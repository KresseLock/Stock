# -*- coding: utf-8 -*-
"""
test_regime_window_gate.py — REGIME_TREND_WINDOW 潔淨 OOS 部署把關
================================================================================
背景：
  tests/test_short_backtest.py 的 sandbox 掃描顯示，生產值 REGIME_TREND_WINDOW = 10
  在乾淨 OOS 的表現可能不如其他窗口。但該掃描用的是 sandbox 簡化引擎
  （T+0 收盤成交、無滑價、無 T+2 交割），不足以支撐部署決策。

  本腳本改用**生產 trading_sim.run_simulation**（T+1 限價搓合、T+2 交割、
  完整風控）複驗，並套用與 run_workflow_experiment.py「B7.5 候選 vs 現行」
  相同的把關方法論。

為何不能直接跑 run_workflow_experiment.py 的 B7.5：
  B7.5 比較的是 best_trading_params.json（候選 mode_b vs 現行）。而
  REGIME_TREND_WINDOW 並不在該檔的搜尋空間內（優化器只調 regime_bull_trend /
  regime_bear_trend 兩個門檻值），它是 config.py:139 的寫死常數。
  直接跑 B7.5，候選與現行的 REGIME_TREND_WINDOW 都會是 10，測不到本候選。
  故在此沿用其方法論、改以本參數為唯一變因。

方法論（對齊 B7.5，run_workflow_experiment.py:1005-1013）：
  * 只在潔淨 OOS 視窗比較。全週期回測前段屬樣本內、報酬受 lookahead 灌水，
    會系統性偏袒越激進的參數，使比較淪為誤導性訊號。
  * 兩邊同模型、同區間、同風控參數，唯一差異為 REGIME_TREND_WINDOW。
  * 現行（incumbent）= config.py 目前部署值；候選（candidate）= 其餘窗口。

判準（B7.5 精神：不可以拉高回撤換報酬）：
  候選需同時滿足 報酬 >= 現行 且 MDD 不劣於現行，才算勝出。

隔離保證：
  * export_report=False，不寫 reports/；不修改 config.py 或任何生產檔案。
  * 僅在記憶體中 monkeypatch trading_sim 的模組層常數（同 CLAUDE.md §4.5 既有慣例）。

執行：
  cd D:\\VScode_Stock\\Stock
  python tests/test_regime_window_gate.py
  python tests/test_regime_window_gate.py --windows 3,5,7,10,15,20
"""

import os
import sys
import io
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
from config import (BACKTEST_DATE, MAX_POSITIONS, SIM_DEFAULT_CAPITAL,
                    REGIME_TREND_WINDOW as PROD_WINDOW)

DATA_PATH = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")


def run_quiet(**kwargs):
    """呼叫 run_simulation 並吞掉冗長 stdout，只取回傳值。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_simulation(**kwargs)
    if result is None:
        return None
    total_return, max_dd, history = result
    return total_return, max_dd * 100.0, history


def exposure_stats(history):
    """
    自 history 還原 benchmark、曝險度與空手比率，用以解釋報酬差異來源。
    history 為 list of dict，需先轉 DataFrame（同 param_sensitivity.reconstruct）。
    """
    try:
        h = pd.DataFrame(history)
        daily_mkt = h["portfolio_return"] - h["portfolio_alpha"]
        bench = float(np.prod(1.0 + daily_mkt.values) - 1.0) * 100.0
        exposure = float((h["invested"] / h["equity"]).mean()) * 100.0
        flat = float((h["invested"] / h["equity"] < 0.05).mean()) * 100.0
        return bench, exposure, flat
    except Exception as e:
        print(f"    [警告] history 解析失敗：{e}")
        return float("nan"), float("nan"), float("nan")


def run_with_window(window, start, end, capital, max_pos):
    """
    以指定 REGIME_TREND_WINDOW 跑生產回測。
    trading_sim 於 run_simulation 內以模組層全域查用該常數，故直接覆寫模組屬性即可。
    min_periods 需一併夾住，否則 window < min_periods 時 rolling 全為 NaN。
    """
    old_w = trading_sim.REGIME_TREND_WINDOW
    old_m = trading_sim.REGIME_TREND_MIN_PERIODS
    trading_sim.REGIME_TREND_WINDOW = window
    trading_sim.REGIME_TREND_MIN_PERIODS = min(old_m, window)
    try:
        return run_quiet(start_date=start, end_date=end,
                         initial_capital=capital, max_positions=max_pos,
                         export_report=False)
    finally:
        trading_sim.REGIME_TREND_WINDOW = old_w
        trading_sim.REGIME_TREND_MIN_PERIODS = old_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="3,5,7,10,15,20")
    ap.add_argument("--capital", type=int, default=SIM_DEFAULT_CAPITAL)
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",") if x.strip()]

    df = pd.read_parquet(DATA_PATH, columns=["date"])
    latest = str(pd.to_datetime(df["date"]).max().date())
    oos_start = str(pd.to_datetime(BACKTEST_DATE).date())

    print("=" * 88)
    print("  REGIME_TREND_WINDOW 潔淨 OOS 部署把關（方法論對齊 B7.5）")
    print("=" * 88)
    print(f"  潔淨 OOS 視窗 : {oos_start} ~ {latest}（模型訓練截點之後）")
    print(f"  現行部署值    : REGIME_TREND_WINDOW = {PROD_WINDOW}")
    print(f"  候選          : {[w for w in windows if w != PROD_WINDOW]}")
    print(f"  引擎          : 生產 trading_sim.run_simulation（T+1 限價搓合、T+2 交割）")
    print(f"  資金 {args.capital:,} | 最大持股 {MAX_POSITIONS}")
    print("  唯一變因為 REGIME_TREND_WINDOW，風控參數沿用目前部署值。\n")

    rows = []
    for w in windows:
        print(f"  [回測] REGIME_TREND_WINDOW = {w} ...", flush=True)
        res = run_with_window(w, oos_start, latest, args.capital, MAX_POSITIONS)
        if res is None:
            print(f"    [警告] 窗口 {w} 回測無結果，跳過")
            continue
        ret, mdd, hist = res
        bench, exp, flat = exposure_stats(hist)
        rows.append({
            "窗口": w,
            "身分": "現行" if w == PROD_WINDOW else "候選",
            "報酬%": round(ret, 2),
            "MDD%": round(mdd, 2),
            "Calmar": round(ret / abs(mdd), 2) if abs(mdd) > 1e-9 else float("nan"),
            "大盤%": round(bench, 2),
            "曝險%": round(exp, 1),
            "空手日%": round(flat, 1),
        })

    if not rows:
        print("\n[錯誤] 全部回測皆無結果。")
        return

    df_r = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("  潔淨 OOS 回測結果")
    print("=" * 88)
    print(df_r.to_string(index=False))

    inc = df_r[df_r["窗口"] == PROD_WINDOW]
    if inc.empty:
        print(f"\n[警告] 未包含現行值 {PROD_WINDOW}，無法比較。")
        return
    inc = inc.iloc[0]

    print("\n" + "=" * 88)
    print("  ⚖️ 候選 vs 現行判定")
    print("=" * 88)
    print(f"  現行 窗口={PROD_WINDOW}：報酬 {inc['報酬%']:+.2f}%  MDD {inc['MDD%']:.2f}%  "
          f"Calmar {inc['Calmar']}")

    winners = []
    for _, r in df_r[df_r["窗口"] != PROD_WINDOW].iterrows():
        d_ret = r["報酬%"] - inc["報酬%"]
        # trading_sim 回傳的 MDD 為正值（如 18.44 代表 -18.44% 回撤），故數值越小越好。
        d_mdd = r["MDD%"] - inc["MDD%"]
        better_ret = d_ret >= 0
        better_mdd = d_mdd <= 0
        ok = better_ret and better_mdd
        if ok:
            winners.append(int(r["窗口"]))
        mark = "✓ 勝出" if ok else ("△ 部分" if better_ret or better_mdd else "✗ 落敗")
        print(f"    窗口={int(r['窗口']):>2}：報酬 {r['報酬%']:+8.2f}% ({d_ret:+7.2f}pp)  "
              f"MDD {r['MDD%']:6.2f}% ({d_mdd:+6.2f}pp 回撤{'較淺' if d_mdd < 0 else '較深'})  "
              f"曝險 {r['曝險%']:.1f}%   {mark}")

    print("\n  判準：報酬 >= 現行 且 MDD 不劣於現行（不可拉高回撤換報酬）")
    print("\n" + "=" * 88)
    if winners:
        print(f"  結論：候選窗口 {winners} 在潔淨 OOS 雙指標皆不劣於現行")
        print(f"  ⚠ 但潔淨 OOS 僅約一年、且只含一次急跌（2026-06/07），樣本量不足以")
        print(f"    支撐直接改生產值。建議先納入 optimize_trading_params.py 搜尋空間，")
        print(f"    走完整 walk-forward 穩定度驗證後再議。")
    else:
        print(f"  結論：無候選窗口能在潔淨 OOS 同時不劣於現行 → 維持現行 {PROD_WINDOW}")
    print("=" * 88)
    print("\n  [Note] 未修改 config.py 或任何生產檔案；export_report=False，未寫 reports/。")


if __name__ == "__main__":
    main()
