# -*- coding: utf-8 -*-
"""
test_markup_bound_gate.py — markup_pct 搜尋下界是否咬住（假最佳檢定）
================================================================================
背景：
  2026-08-28 的新目標函式候選（configs/best_trading_params_candidate.json，
  walk_forward_selected = 窗口4最佳）選出 markup_pct = -3.0，而
  config.py § 10.5 TRADING_PARAM_BOUNDS["markup_pct"] 的下界正好就是 -3.0。

  貼死邊界的解有兩種可能，兩者的處置完全相反：
    (A) 真極大  —— 目標函式在 -3.0 附近達峰，下界沒有限制到搜尋 → 放寬邊界無意義。
    (B) 假最佳  —— 目標函式在 -3.0 以下仍持續上升，只是被邊界截斷 → 邊界該放寬；
                   且需警覺優化器正往「更依賴回測搓合假設」的方向挖分數（見下）。

  為何 (B) 特別危險：trading_sim.py:566-570 的限價搓合規則是
  「當日 low <= 限價 → 一定以限價成交」。markup_pct 越負（掛越深的折價單），
  成交就越依賴這條樂觀假設。實測路徑 1（2026-01-02~08-27）：
      現行部署 markup=-1.0 → 盤中觸價撮合佔 80%（41/51 筆）
      本候選   markup=-3.0 → 盤中觸價撮合佔 94%（29/31 筆）
  實盤上盤中瞬間觸價未必排得到隊，這個比例就是回測與實盤的落差風險。
  若優化器在放寬後還想往更深跑，代表分數有一部分是從搓合假設裡挖出來的，
  那就該先修搓合模型，而不是放寬邊界讓它繼續挖。

方法論（為何 Stage 1 是主判準、Stage 2 只是佐證）：
  直覺做法是「把下界改成 -5.0 重跑一次優化，看它跑到哪」。但那一場同時改變了
  兩件事：邊界，以及 Optuna 的整條隨機搜尋軌跡。本專案已有直接證據顯示
  優化器的 run-to-run 變異極大——新目標函式的第 1 名（窗口4最佳，得分 0.7994）
  與第 2 名（窗口3最佳，0.7292）分數只差 0.07，裁判區報酬中位數卻差 75pp
  （+65.49% vs -9.24%）。在這種變異下，單次重跑的結果無法歸因給邊界。
  對照 EXPERIMENTS_PENDING.md 方法論鐵律 #1（禁止用單路徑回測差決策）。

  故本檢定以 OFAT（One-Factor-At-A-Time，同 scripts/param_sensitivity.py 的精神）
  為主判準：固定候選向量的其餘 8 維，只掃 markup_pct，看目標函式的形狀。
  單一變因、零隨機性、可重現。

Stage 1（預設）— OFAT 掃描
  對 markup_pct in [-5.0 .. -1.0] step 0.5，固定候選其餘維度，各算：
    * 調參區間（config.TRADING_OPT_START/END_DATE）的優化器目標函式得分
      —— 直接呼叫 optimize_trading_params.run_simulation_scoring，與優化器同一支程式碼
    * 裁判區 9 條起點偏移路徑的報酬／回撤中位數
      —— 對齊 scripts/validate_candidate_params.py 的 OFFSETS，不用單路徑
    * 盤中觸價撮合佔比 —— 量化上述搓合假設依賴度

  判準：
    邊界咬住(B) ⇔ 目標函式得分在 markup < -3.0 的區間仍高於 -3.0 處的得分。
    否則為 (A)，-3.0 是真極大，config.py 的下界不需要動。

Stage 2（--reopt，選用）— 真的重跑優化
  in-memory monkeypatch TRADING_PARAM_BOUNDS 後跑兩場 Optuna（下界 -5.0 vs -3.0），
  共用同一個 TPESampler seed 以壓低（但無法消除）軌跡差異。只作佐證，不作判準。

隔離保證（照 tests/test_regime_window_gate.py 既有慣例）：
  * 只在記憶體中改 optimize_trading_params.TRADING_PARAM_BOUNDS，不寫 config.py。
  * 不寫 configs/ 任何檔案（不產生候選檔）。
  * 量測撮合方式時需要 Excel 明細，故把 trading_sim.BASE_DIR 暫時指向 tests/_sandbox_markup/，
    報表落在該處而非專案 reports/；離開即還原。
  * 全程唯讀 configs/best_trading_params_candidate.json。

執行：
  cd D:\\VScode_Stock\\Stock
  python tests/test_markup_bound_gate.py                      # Stage 1（約 7 分鐘）
  python tests/test_markup_bound_gate.py --markups -5,-4,-3,-2  # 自訂掃描點
  python tests/test_markup_bound_gate.py --reopt -t 150        # 加跑 Stage 2
"""

import os
import sys
import json
import shutil
import argparse
import contextlib
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

import optimize_trading_params as opt
import trading_sim as ts
from config import (MAX_POSITIONS, TRADING_OPT_START_DATE, TRADING_OPT_END_DATE,
                    TRADING_PARAM_BOUNDS, REGIME_MAX_POSITIONS)
# 裁判區的多起點切法只有一個定義處，直接沿用避免漂移
from validate_candidate_params import OFFSETS, trading_days

CANDIDATE = os.path.join(ROOT_DIR, "configs", "best_trading_params_candidate.json")
SANDBOX = os.path.join(BASE_DIR, "_sandbox_markup")
PROD_LOW = TRADING_PARAM_BOUNDS["markup_pct"][0]   # 現行下界，判準的分界點


# ── 隔離工具 ────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def sandbox_reports():
    """把 trading_sim 的報表輸出導到 tests/_sandbox_markup/reports/。

    trading_sim.BASE_DIR 另有一處用途是讀 scripts/stock_categories.json（純顯示用的
    股票中文名，不影響損益），故一併複製進沙盒，使沙盒內行為與生產完全一致。
    DATA_PATH／MODEL_DIR 在 import 時就已綁定絕對路徑，不受本 patch 影響。
    """
    os.makedirs(os.path.join(SANDBOX, "scripts"), exist_ok=True)
    src = os.path.join(ROOT_DIR, "scripts", "stock_categories.json")
    dst = os.path.join(SANDBOX, "scripts", "stock_categories.json")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
    old = ts.BASE_DIR
    ts.BASE_DIR = SANDBOX
    try:
        yield os.path.join(SANDBOX, "reports")
    finally:
        ts.BASE_DIR = old


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8")):
        return fn(*a, **kw)


# ── 量測 ────────────────────────────────────────────────────────────────────
def score_on(start, end, params, capital):
    """回傳 (目標函式得分, 報酬%, 回撤%)。與優化器共用同一支評分程式碼。"""
    s, _, ret, mdd = opt.run_simulation_scoring(start, end, params, capital, MAX_POSITIONS)
    return s, ret, mdd * 100.0     # run_simulation 的 max_dd 是小數，total_return 已是 %


def fill_mode(start, end, params, capital):
    """回傳 (買進筆數, 盤中觸價撮合佔比)。需要 Excel 明細，故在沙盒內開 export_report。"""
    with sandbox_reports() as rdir:
        quiet(ts.run_simulation,
              start_date=start, end_date=end, initial_capital=capital,
              max_positions=MAX_POSITIONS,
              mkt_panic_ma5=params["panic_ma5"],
              mkt_panic_breadth=params["panic_breadth"],
              markup_pct=params["markup_pct"],
              regime_buy_threshold={"Bull": params["regime_bull_buy"],
                                    "Sideways": params["regime_sideways_buy"],
                                    "Bear": 99.0},
              regime_bull_trend=params["regime_bull_trend"],
              regime_bear_trend=params["regime_bear_trend"],
              regime_max_positions={"Bull": params["regime_bull_pos"],
                                    "Sideways": params["regime_sideways_pos"],
                                    # 與 run_simulation_scoring 同一來源，確保沙盒行為 == 優化器
                                    "Bear": REGIME_MAX_POSITIONS["Bear"]},
              regime_exit_params=None, export_report=True)
        f = os.path.join(rdir, f"backtest_report_{start}_{end}.xlsx")
        if not os.path.exists(f):
            return 0, float("nan")
        t = pd.read_excel(f, sheet_name="Trade_History")
        col = next((c for c in t.columns if t[c].astype(str).str.contains("撮合").any()), None)
        if col is None:
            return 0, float("nan")
        buys = t[t[col].astype(str).str.contains("撮合")]
        if len(buys) == 0:
            return 0, float("nan")
        intraday = buys[col].astype(str).str.contains("盤中回檔撮合").sum()
        return len(buys), intraday / len(buys)


def judge_paths(params, capital, zone_days):
    """裁判區 9 條起點偏移路徑（終點固定），回傳 (報酬中位數, 回撤中位數)。"""
    end = zone_days[-1]
    rets, mdds = [], []
    for off in OFFSETS:
        if off >= len(zone_days):
            break
        _, r, m = score_on(zone_days[off], end, params, capital)
        rets.append(r)
        mdds.append(m)
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else \
        (sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2.0
    return med(rets), med(mdds)


# ── Stage 2：隔離重跑優化 ───────────────────────────────────────────────────
def reopt(low, start, end, capital, trials, seed):
    """在放寬(或維持)下界的情況下跑一場 Optuna，回傳 (最佳得分, 最佳參數, 全部 trial 的 markup)。

    只改記憶體中的 opt.TRADING_PARAM_BOUNDS（_suggest 由此取邊界），離開即還原；
    不寫任何檔案，故不會產生候選檔。
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    orig = opt.TRADING_PARAM_BOUNDS["markup_pct"]
    opt.TRADING_PARAM_BOUNDS["markup_pct"] = (low, orig[1], orig[2])
    try:
        def objective(trial):
            p = opt.suggest_trial_params(trial, True)
            s, _, _, _ = opt.run_simulation_scoring(start, end, p, capital, MAX_POSITIONS)
            return s
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        quiet(study.optimize, objective, n_trials=trials)
        marks = [t.params["markup_pct"] for t in study.trials
                 if t.value is not None and "markup_pct" in t.params]
        return study.best_value, study.best_params, marks
    finally:
        opt.TRADING_PARAM_BOUNDS["markup_pct"] = orig


def main():
    ap = argparse.ArgumentParser(description="markup_pct 搜尋下界咬住檢定（隔離沙盒）")
    ap.add_argument("-c", "--capital", type=int, default=2000000)
    ap.add_argument("--markups", type=str, default="-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0",
                    help="Stage 1 掃描點（逗號分隔）")
    ap.add_argument("--reopt", action="store_true", help="加跑 Stage 2 隔離重跑優化")
    ap.add_argument("-t", "--trials", type=int, default=150, help="Stage 2 每場 trial 數")
    ap.add_argument("--seed", type=int, default=42, help="Stage 2 兩場共用的 TPESampler seed")
    ap.add_argument("--low", type=float, default=-5.0, help="Stage 2 放寬後的下界")
    args = ap.parse_args()

    base = json.load(open(CANDIDATE, encoding="utf-8"))
    params0 = base["best_params"]
    tune_s, tune_e = TRADING_OPT_START_DATE, TRADING_OPT_END_DATE
    days = trading_days()
    zone = [d for d in days if d > tune_e]

    print("=" * 86)
    print("  markup_pct 搜尋下界咬住檢定（tests/ 隔離沙盒，不動 config.py / configs/ / reports/）")
    print("=" * 86)
    print(f"  受測向量   : {os.path.basename(CANDIDATE)}"
          f"（{base.get('walk_forward_selected', '?')}，best_score={base.get('best_score', 0):.4f}）")
    print(f"  固定維度   : " + ", ".join(f"{k}={v}" for k, v in params0.items() if k != "markup_pct"))
    print(f"  調參區間   : {tune_s} ~ {tune_e}（目標函式得分的計算區間）")
    print(f"  裁判區     : {zone[0]} ~ {zone[-1]}（{len(zone)} 個交易日，{len(OFFSETS)} 條起點偏移路徑）")
    print(f"  現行下界   : markup_pct >= {PROD_LOW}   ← 判準的分界點")
    print(f"  資金       : {args.capital:,}")
    print("=" * 86)

    sweep = [float(x) for x in args.markups.split(",")]
    print(f"\n[Stage 1] OFAT 掃描（單一變因，零隨機性）")
    print(f"{'markup':>8} | {'目標函式得分':>13} | {'調參區報酬':>11} {'調參區MDD':>10} | "
          f"{'裁判區報酬中位':>14} {'裁判區MDD中位':>13} | {'買進':>5} {'盤中觸價撮合':>13}")
    print("-" * 105)

    rows = []
    for mk in sweep:
        p = dict(params0, markup_pct=mk)
        s, ret, mdd = score_on(tune_s, tune_e, p, args.capital)
        zr, zm = judge_paths(p, args.capital, zone)
        nb, fr = fill_mode(zone[0], zone[-1], p, args.capital)
        rows.append((mk, s, ret, mdd, zr, zm, nb, fr))
        mark = "  <= 現行下界" if abs(mk - PROD_LOW) < 1e-9 else ""
        print(f"{mk:8.1f} | {s:13.4f} | {ret:10.2f}% {mdd:9.2f}% | "
              f"{zr:13.2f}% {zm:12.2f}% | {nb:5d} {fr:12.1%}{mark}")

    print("\n" + "=" * 86)
    print("  Stage 1 判讀")
    print("=" * 86)

    at_bound = next((r for r in rows if abs(r[0] - PROD_LOW) < 1e-9), None)
    below = [r for r in rows if r[0] < PROD_LOW - 1e-9]
    best = max(rows, key=lambda r: r[1])

    print(f"  目標函式最高分落在 markup_pct = {best[0]:.1f}（得分 {best[1]:.4f}）")
    if at_bound is None:
        print("  [略過判準] 掃描點未包含現行下界，無從判定；請把 " f"{PROD_LOW}" " 加進 --markups。")
    elif not below:
        print("  [略過判準] 掃描點未包含現行下界以下的值，無從判定。")
    else:
        better = [r for r in below if r[1] > at_bound[1]]
        if better:
            print(f"  → 判定：**下界咬住（假最佳）**。有 {len(better)}/{len(below)} 個更深的 markup "
                  f"得分高於下界處的 {at_bound[1]:.4f}："
                  + "、".join(f"{r[0]:.1f}→{r[1]:.4f}" for r in better))
            print(f"     放寬 config.py TRADING_PARAM_BOUNDS['markup_pct'] 下界有實質意義。")
        else:
            print(f"  → 判定：**下界未咬住（真極大）**。所有更深的 markup 得分都不高於下界處的 "
                  f"{at_bound[1]:.4f}，")
            print(f"     優化器選 {PROD_LOW} 不是被邊界截斷。放寬下界不會改變結果，config.py 不需要動。")

    if at_bound is not None and below:
        worst_fill = max((r for r in below), key=lambda r: (r[7] if r[7] == r[7] else -1))
        print(f"\n  搓合假設依賴度：下界處 {at_bound[7]:.1%}；更深處最高 {worst_fill[7]:.1%}"
              f"（markup={worst_fill[0]:.1f}）。")
        print(f"  這是回測對「當日 low <= 限價即成交」的依賴比例，越高則實盤落差風險越大。")

    print(f"\n  提醒：目標函式得分本身已被證明鑑別力不足（新目標函式第 1／2 名分數差 0.07，")
    print(f"        裁判區報酬中位數卻差 75pp），故上表的『裁判區』兩欄才是實際績效證據，")
    print(f"        目標函式得分只用來回答『邊界有沒有截斷搜尋』這個機械性問題。")

    if args.reopt:
        print("\n" + "=" * 86)
        print(f"  [Stage 2] 隔離重跑優化（各 {args.trials} trials，共用 seed={args.seed}）")
        print("=" * 86)
        for low in (args.low, PROD_LOW):
            bv, bp, marks = reopt(low, tune_s, tune_e, args.capital, args.trials, args.seed)
            deeper = sum(1 for m in marks if m < PROD_LOW - 1e-9)
            print(f"\n  下界 {low:.1f}：最佳得分 {bv:.4f}，最佳 markup_pct = {bp['markup_pct']:.1f}")
            print(f"           全 {len(marks)} 個 trial 中有 {deeper} 個抽到 < {PROD_LOW} 的值"
                  f"（{deeper / max(1, len(marks)):.0%}）")
            print(f"           最佳向量：" + ", ".join(f"{k}={v}" for k, v in bp.items()))
        print(f"\n  ※ Stage 2 混淆了『邊界放寬』與『Optuna 軌跡差異』，只作佐證；判準以 Stage 1 為準。")

    print("\n" + "=" * 86)
    print(f"  沙盒產物在 {SANDBOX}（可直接刪除）。config.py / configs/ / reports/ 全程未被修改。")
    print("=" * 86)


if __name__ == "__main__":
    main()
