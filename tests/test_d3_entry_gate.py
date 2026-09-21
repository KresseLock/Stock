# -*- coding: utf-8 -*-
"""
test_d3_entry_gate.py — 進場 D3 門檻 潔淨 OOS 部署把關（完全隔離）
================================================================================
背景：
  2026-09-02 的生產推論把 2438 翔耀、2419 仲琦、6272 驊陞排進買進清單第 1/2/3 名，
  但同一份清單裡三檔的 D3 分數全為負（-15.7 / -10.8 / -13.6）。也就是說模型
  自己講的是「明天強、三天後弱」，系統卻照買，買進後又被 MIN_HOLD_DAYS 綁住
  （僅停損能提前出場）。

  查證兩處進場邏輯，D3 **完全不參與篩選或排序**，只在報表上印出來給人看：
    trading_sim.py:535      —— 只用 day1_score < current_buy_threshold
    scripts/inference.py:592 —— buy_cond 只有 Day1_net >= eff_buy_threshold

候選改造（唯一變因）：
  進場時額外要求「訊號日 Day3_net >= D3_MIN」，不足者不得進場。
  D3_MIN = DISABLED 即現行（完全不過濾）。

  母體分布（OOS、D1>=12 的 2,179 筆候選）：D3 中位數 +0.9，
  分位數 [5,25,50,75,95] = [-13.3, -5.1, 0.9, 6.1, 16.0]。
  故本次事件的三檔落在候選 D3 分布的底部 5~10%。

實作（比 test_liquidity_gate.py 更自洽，無 index 對齊風險）：
  以 monkeypatch 包裝 day-1 模型，在其 predict 內部**同時**用未包裝的 day-3
  模型對同一份 X 推論出 D3，對 D3 < 門檻的列把 (prob_weak, prob_mid, prob_strong)
  壓成 (1,0,0) → Day1_net = -100，遠低於任何 buy_threshold。
  等價於在 trading_sim.py:535 買進迴圈加一行 `continue`，但不改生產程式碼。

  ※ 過濾條件由 X 自身推論而得，不依賴任何外部 index 對齊 —— 這是本測試
    相對流動性把關的結構性優勢。
  ※ models[3] 本身不包裝，trading_sim 後續獨立呼叫它算 Day3_net 供出場判定
    （D3 轉弱／移動止盈／停損）行為與現行 100% 一致。
  ※ 已知副作用（不影響判定）：trading_sim.py:702 的 bottom-5% 對照組
    （portfolio_spread 診斷欄）會被壓低分數污染；該欄不參與交易決策。

方法論（對齊 CLAUDE.md §4.5 #4 與 #7）：
  * 判準：全 OOS 報酬與回撤雙贏 **且** 多數起點穩健（多起點雙贏 > 半數）。
  * 主判準採固定終點、六起點的長區間掃描，不用季度切片（§4.5 #5：
    MIN_HOLD_DAYS 達 20 天且獲利靠肥尾，短切片會系統性低估績效）。
  * 起點組與 test_liquidity_gate.py／§4.5 #7 完全一致，以利橫向對照。

⚠ 事前預期（寫在跑之前，避免事後合理化）：
  §4.5 #1 已確立「嚴禁用收緊進場來減少停損次數 —— 已驗證會把肥尾贏家一起
  砍掉而崩潰」。D3 門檻形式上仍是收緊進場，很可能重演。本測試的價值在於
  它有明確機制假設（模型內部訊號自相矛盾）而非盲試，且無論結果為何都應
  記入 §4.5 以免日後重做。

隔離保證：
  * 生產 parquet／模型唯讀；不寫 sandbox、不改 config.py／models／data。
  * run_simulation 一律 export_report=False，不寫 reports/。
  * monkeypatch 僅存在於本行程記憶體，finally 還原。

執行：
  cd D:\\VScode_Stock\\Stock
  python tests/test_d3_entry_gate.py
  python tests/test_d3_entry_gate.py --d3mins -15,-10,-5,0,5
  python tests/test_d3_entry_gate.py --starts 2025-08-01,2026-01-01

════════════════════════════════════════════════════════════════════════════════
實測結果（2026-09-08，6 起點 × 6 門檻）：❌ 全面否決，不移植
════════════════════════════════════════════════════════════════════════════════
  **沒有任何門檻在任何指標上贏過現行。** 全 OOS 報酬差全為負：
      D3>=-15 −24.9pp ／ -10 −63.7pp ／ -5 −103.0pp ／ 0 −68.5pp ／ +5 −110.4pp
  多起點雙贏 0～1/6。曲線非單調（轉折 2 次：+106→+81→+43→+3→+38→−4），
  D3>=-5 掉到 +3.4% 但 D3>=0 又回到 +37.9% —— 隨機破壞，不是有方向的效應。

  **決定性證據（機制層面）**：最寬鬆的 D3>=-15 只擋掉 21.7% 的樣本，
  曝險 84.7%→84.4%、均持倉 4.32→4.32 **幾乎沒變**，報酬卻掉了 25pp。
  部位數與曝險都沒降、只是換了一批標的進來，報酬就崩 —— 代表被擋掉的
  不是「會虧的股」，而是**會賺的股**。且過濾越多回撤越糟（D3>=-5 的
  MDD 33.6%，遠劣於現行 22.2%），完全不是「避開弱勢股」的樣子。

  **為何必然失敗（這才是重點）**：D3 負值不是「別買」的訊號，而是
  「進場後準備要跑」的訊號。本系統的分工是**進場吃 D1 動能、出場靠 D3
  轉弱＋移動止盈＋ATR 停損**（`inference.py:504` 的 D3 轉弱判定即為此）。
  把 D3 搬到進場端當篩選條件，等於把出場端的工具挪去做它不該做的事，
  同時砍掉 §4.5 #1 所說的肥尾贏家來源 —— D1 強／D3 弱的標的正是那些
  「短線爆發、隨後回落」的部位，系統本來就靠出場端在它們衝高時鎖利。

  教訓：**訊號有其專屬的作用位置，跨端挪用即失效。** 這是 §4.5 #1
  「嚴禁收緊進場」的第三個實證，也補上一條新規則：D1/D2/D3 各有分工，
  不要把出場訊號拿去做進場過濾（反之亦然）。

  ※ 資料附註：本測試重算的 2026-09-02 分數與當日 prediction_20260902.txt
    快照略有出入（例：2419 D1 +22.0 vs 當日 +15.7），因特徵檔在 9/2 之後
    有籌碼資料回補。方向與量級一致，不影響把關結論。
"""

import os
import sys
import io
import json
import argparse
import contextlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import trading_sim
from trading_sim import run_simulation
from config import BACKTEST_DATE, MAX_POSITIONS, SIM_DEFAULT_CAPITAL, BUY_THRESHOLD
from config import FEATURES_PARQUET as DATA_PATH   # 資料路徑唯一來源：config.py § 0
FEATURE_COLS_PATH = os.path.join(ROOT_DIR, "models", "feature_cols.json")

DISABLED = -999.0                    # 代表「不過濾」的哨兵值
DEFAULT_D3MINS = "-15,-10,-5,0,5"

# 固定終點、多起點（§4.5 #7）。與 test_liquidity_gate.py 同組以利對照。
DEFAULT_STARTS = "2025-08-01,2025-10-01,2025-12-01,2026-01-01,2026-02-02,2026-04-01"

CASE_SIDS = ["2438", "2419", "6272", "2382", "3189"]
CASE_SIGNAL_DATE = "2026-09-02"


def net_score(model, X):
    """(P(強) - P(弱)) × 100，與 trading_sim.py:208 的 DayN_net 公式一致。"""
    p = np.asarray(model.predict(X))
    return (p[:, 2] - p[:, 0]) * 100.0


class _Day1D3Filter:
    """
    包裝 day-1 模型：對「同一份 X 推論出的 D3 < d3_min」的列，
    輸出 (prob_weak, prob_mid, prob_strong) = (1, 0, 0) → Day1_net = -100。
    """

    def __init__(self, inner1, model3, d3_min):
        self._inner = inner1
        self._model3 = model3
        self._d3_min = d3_min
        self.n_rows = 0
        self.n_blocked = 0

    def predict(self, X, *args, **kwargs):
        preds = np.asarray(self._inner.predict(X, *args, **kwargs))
        if preds.ndim != 2 or preds.shape[1] != 3:
            return preds
        if self._d3_min <= DISABLED:
            return preds
        d3 = net_score(self._model3, X)
        mask = d3 < self._d3_min
        self.n_rows = int(len(mask))
        self.n_blocked = int(mask.sum())
        if self.n_blocked:
            preds = preds.copy()
            preds[mask, 0] = 1.0
            preds[mask, 1] = 0.0
            preds[mask, 2] = 0.0
        return preds

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_quiet(**kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_simulation(**kwargs)
    if result is None:
        return None
    total_return, max_dd, history = result
    return total_return, max_dd * 100.0, history


def run_with_d3min(d3_min, start, end, capital, max_pos):
    """以指定 D3 門檻跑生產回測。d3_min <= DISABLED → 完全不過濾（現行基準）。"""
    orig_load = trading_sim.load_models
    holder = {}

    def _patched():
        models = orig_load()
        # models[3] 保持未包裝：出場端的 Day3_net 計算不受影響。
        wrapped = _Day1D3Filter(models[1], models[3], d3_min)
        models[1] = wrapped
        holder["w"] = wrapped
        return models

    trading_sim.load_models = _patched
    try:
        res = run_quiet(start_date=start, end_date=end,
                        initial_capital=capital, max_positions=max_pos,
                        export_report=False)
    finally:
        trading_sim.load_models = orig_load

    w = holder.get("w")
    return res, (w.n_blocked if w else 0), (w.n_rows if w else 0)


def exposure_stats(history):
    try:
        h = pd.DataFrame(history)
        daily_mkt = h["portfolio_return"] - h["portfolio_alpha"]
        bench = float(np.prod(1.0 + daily_mkt.values) - 1.0) * 100.0
        exposure = float((h["invested"] / h["equity"]).mean()) * 100.0
        n_pos = float(h["n_positions"].mean())
        return bench, exposure, n_pos
    except Exception as e:
        print(f"    [警告] history 解析失敗：{e}")
        return float("nan"), float("nan"), float("nan")


def case_diagnosis(d3mins):
    """印出事件當日各標的的 D1／D3，說明會被哪些門檻擋下。"""
    print(f"\n  [案例診斷] {CASE_SIGNAL_DATE} 訊號日 D1／D3 分數：")
    try:
        df = pd.read_parquet(DATA_PATH)
        sig = df[pd.to_datetime(df["date"]) == pd.to_datetime(CASE_SIGNAL_DATE)]
        if sig.empty:
            print("    （查無該日資料）")
            return
        fc = json.load(open(FEATURE_COLS_PATH, encoding="utf-8"))
        X = sig.reindex(columns=fc).astype(np.float32)
        m = trading_sim.load_models()
        d1, d3 = net_score(m[1], X), net_score(m[3], X)
        out = pd.DataFrame({"stock_id": sig["stock_id"].astype(str).values, "d1": d1, "d3": d3})
        for sid in CASE_SIDS:
            r = out[out["stock_id"] == sid]
            if r.empty:
                print(f"    {sid}: （查無資料）")
                continue
            v1, v3 = float(r["d1"].iloc[0]), float(r["d3"].iloc[0])
            blocked = [g for g in d3mins if g > DISABLED and v3 < g]
            tag = f"被擋於門檻 {blocked}" if blocked else "全數門檻皆通過"
            print(f"    {sid}: D1 {v1:+6.1f}  D3 {v3:+6.1f}  → {tag}")
    except Exception as e:
        print(f"    [警告] 案例診斷失敗：{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3mins", default=DEFAULT_D3MINS, help="D3 下限，逗號分隔")
    ap.add_argument("--starts", default=DEFAULT_STARTS)
    ap.add_argument("--capital", type=int, default=SIM_DEFAULT_CAPITAL)
    args = ap.parse_args()

    d3mins = [float(x) for x in args.d3mins.split(",") if x.strip()]
    d3mins = [DISABLED] + [g for g in d3mins if g > DISABLED]
    starts = [s.strip() for s in args.starts.split(",") if s.strip()]

    df_dates = pd.read_parquet(DATA_PATH, columns=["date"])
    latest = str(pd.to_datetime(df_dates["date"]).max().date())
    oos_start = str(pd.to_datetime(str(BACKTEST_DATE)).date())

    def lab(g):
        return "現行" if g <= DISABLED else f"{g:+.0f}"

    print("=" * 96)
    print("  進場 D3 門檻 潔淨 OOS 部署把關（方法論對齊 §4.5 #4／#7）")
    print("=" * 96)
    print(f"  潔淨 OOS 上界 : {oos_start} ~ {latest}（模型訓練截點之後）")
    print(f"  候選條件      : 訊號日 Day3_net >= 門檻，否則不得進場")
    print(f"  門檻掃描      : {[lab(g) for g in d3mins]}（現行 = 不過濾）")
    print(f"  多起點        : {starts}（終點固定 {latest}）")
    print(f"  引擎          : 生產 trading_sim.run_simulation（T+1 限價搓合、T+2 交割）")
    print(f"  資金 {args.capital:,} | 最大持股 {MAX_POSITIONS} | BUY_THRESHOLD {BUY_THRESHOLD}")
    print("  唯一變因為進場 D3 門檻；模型、特徵、風控參數、出場邏輯全同。")
    print("  ⚠ 事前預期：§4.5 #1 已判定「收緊進場會砍掉肥尾贏家」，本候選可能重演。\n")

    rows = []
    total = len(starts) * len(d3mins)
    done = 0
    for start in starts:
        for g in d3mins:
            done += 1
            print(f"  [{done:>2}/{total}] 回測 起點={start} D3門檻={lab(g)} ...", flush=True)
            res, n_blocked, n_rows = run_with_d3min(g, start, latest, args.capital, MAX_POSITIONS)
            if res is None:
                print("      [警告] 無結果，跳過")
                continue
            ret, mdd, hist = res
            bench, exp, npos = exposure_stats(hist)
            rows.append({
                "起點": start, "D3門檻": g,
                "報酬%": round(ret, 2), "MDD%": round(mdd, 2),
                "Calmar": round(ret / abs(mdd), 2) if abs(mdd) > 1e-9 else float("nan"),
                "大盤%": round(bench, 2), "曝險%": round(exp, 1), "均持倉": round(npos, 2),
                "擋比%": round(n_blocked / n_rows * 100, 1) if n_rows else 0.0,
            })

    if not rows:
        print("\n[錯誤] 全部回測皆無結果。")
        return

    df_r = pd.DataFrame(rows)
    df_r["門檻"] = df_r["D3門檻"].map(lab)

    print("\n" + "=" * 96)
    print("  逐窗結果")
    print("=" * 96)
    for start in starts:
        sub = df_r[df_r["起點"] == start]
        if sub.empty:
            continue
        print(f"\n  ── 起點 {start} ~ {latest} " + "─" * 40)
        print(sub[["門檻", "報酬%", "MDD%", "Calmar", "大盤%", "曝險%", "均持倉", "擋比%"]]
              .to_string(index=False))

    print("\n" + "=" * 96)
    print("  ⚖️ 候選 vs 現行判定")
    print("=" * 96)
    print("  判準（§4.5 #4）：全 OOS 報酬與回撤雙贏，且多數起點穩健。")
    print("  §4.5 #7 附加：單調不是穩健的證據，多起點一致才是。\n")

    full_start = starts[0]
    summary = []
    for g in d3mins:
        if g <= DISABLED:
            continue
        wins_ret = wins_mdd = wins_both = n = 0
        for start in starts:
            b = df_r[(df_r["起點"] == start) & (df_r["D3門檻"] <= DISABLED)]
            c = df_r[(df_r["起點"] == start) & (df_r["D3門檻"] == g)]
            if b.empty or c.empty:
                continue
            n += 1
            d_ret = c.iloc[0]["報酬%"] - b.iloc[0]["報酬%"]
            d_mdd = c.iloc[0]["MDD%"] - b.iloc[0]["MDD%"]     # 正值 = 回撤更深
            wins_ret += d_ret >= 0
            wins_mdd += d_mdd <= 0
            wins_both += (d_ret >= 0) and (d_mdd <= 0)

        fb = df_r[(df_r["起點"] == full_start) & (df_r["D3門檻"] <= DISABLED)]
        fc_ = df_r[(df_r["起點"] == full_start) & (df_r["D3門檻"] == g)]
        if fb.empty or fc_.empty:
            continue
        f_ret = fc_.iloc[0]["報酬%"] - fb.iloc[0]["報酬%"]
        f_mdd = fc_.iloc[0]["MDD%"] - fb.iloc[0]["MDD%"]
        full_ok = (f_ret >= 0) and (f_mdd <= 0)
        robust = n > 0 and wins_both > n / 2
        summary.append({
            "D3門檻": lab(g),
            "全OOS報酬差pp": round(f_ret, 2),
            "全OOS回撤差pp": round(f_mdd, 2),
            "全OOS雙贏": "✓" if full_ok else "✗",
            "起點報酬勝": f"{wins_ret}/{n}",
            "起點回撤勝": f"{wins_mdd}/{n}",
            "起點雙贏": f"{wins_both}/{n}",
            "判定": "✓ 通過" if (full_ok and robust) else "✗ 未通過",
        })

    df_s = pd.DataFrame(summary)
    print(df_s.to_string(index=False))
    passed = [r["D3門檻"] for r in summary if r["判定"].startswith("✓")]

    full_curve = df_r[df_r["起點"] == full_start].sort_values("D3門檻")
    if len(full_curve) >= 3:
        rets = full_curve["報酬%"].to_numpy()
        diffs = np.diff(rets)
        sign_changes = int((np.diff(np.sign(diffs)) != 0).sum())
        print(f"\n  全 OOS 報酬對門檻的曲線：{[f'{v:+.1f}' for v in rets]}"
              f"（門檻 {[lab(g) for g in full_curve['D3門檻']]}）")
        print(f"  轉折次數 = {sign_changes}"
              f"{'（大致單調）' if sign_changes <= 1 else '（非單調 → 疑雜訊擬合，§4.5 #7）'}")

    print("\n" + "=" * 96)
    if passed:
        print(f"  結論：D3 門檻 {passed} 在潔淨 OOS 通過雙贏＋多數起點穩健。")
        print("  ⚠ 移植前須確認通過的門檻不是只由少數起點驅動；若移植，")
        print("    應在 config.py 新增 ENTRY_D3_MIN 常數，並於 trading_sim.py:535")
        print("    買進迴圈與 scripts/inference.py:592 的 buy_cond 共用。")
    else:
        print("  結論：無門檻能同時滿足全 OOS 雙贏與多數起點穩健 → 不移植，維持現行。")
        print("  這將是 §4.5 #1「收緊進場必砍肥尾贏家」的又一個實證案例。")
    print("=" * 96)

    case_diagnosis(d3mins)

    print("\n  [Note] 未修改 config.py／models／data 或任何生產檔案；")
    print("         export_report=False，未寫 reports/；monkeypatch 已於 finally 還原。")


if __name__ == "__main__":
    main()
