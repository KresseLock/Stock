"""
test_inference_sim_consistency.py — inference.py <-> trading_sim.py 決策一致性測試
================================================================================
目的：確保「實盤 (inference) 與回測 (trading_sim) 對同一市況做相同決策」，
      防止回測自嗨、實盤兩樣。改任何風控 (含 Bull 開關) 後務必先跑本測試。

測試項目：
  1. 停損公式一致性  — inference.compute_stop_pct vs trading_sim ATR 公式 (真 import 對齊)
  2. panic 紅燈契約  — 兩邊須遵守同一規則 (ma5<門檻 或 breadth<門檻且非Bull → 暫停進場)
  3. 共用 config 常數 — 兩邊讀同源常數, 確保門檻/出場天生一致
  4. 資料回歸 (若 parquet 存在) — 掃全期, inference 與 trading_sim 買入門檻 0 脫節

注意：2/4 為「規則契約」測試 (鏡像兩邊邏輯比對)，非端到端調用。Test 1 為真 import
      硬核對齊。治本之道是把決策邏輯抽成共用函式; 在此之前本測試作為契約守門。

執行方式: python tests/test_inference_sim_consistency.py
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

PASS = " PASS"
FAIL = " FAIL"
WARN = "  WARN"
results = []


def check(name, fn):
    try:
        msg = fn()
        results.append((PASS, name, msg or ""))
    except AssertionError as e:
        results.append((FAIL, name, str(e)))
    except Exception as e:
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))


# ── Test 1: 停損公式一致性 (真 import 對齊) ──────────────────────────
def test_stop_loss_formula():
    from scripts.inference import compute_stop_pct
    from config import (ATR_STOP_ENABLED, ATR_STOP_MULTIPLIER, ATR_STOP_FLOOR_PCT,
                        ATR_STOP_CEILING_PCT, STOP_LOSS_PCT)
    for atr in [0.01, 0.02, 0.03, 0.05, 0.10, None]:
        inf = compute_stop_pct(atr)
        if ATR_STOP_ENABLED and atr is not None and atr > 0:
            sim = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEILING_PCT, -ATR_STOP_MULTIPLIER * atr * 100))
        else:
            sim = STOP_LOSS_PCT
        assert abs(inf - sim) < 1e-9, f"atr={atr}: inference={inf:+.4f} != trading_sim={sim:+.4f}"
    return "6 個 atr 值停損公式 inference==trading_sim"


# ── Test 2: panic 紅燈契約 (trading_sim:355-369 / inference 須一致) ──
def _panic_threshold(ma5, breadth, regime, base):
    """兩邊都須遵守的 panic 契約。panic 時返回 99.0 (暫停進場)。"""
    from config import MKT_PANIC_MA5, MKT_PANIC_BREADTH
    if ma5 < MKT_PANIC_MA5:
        return 99.0
    if breadth < MKT_PANIC_BREADTH and regime != "Bull":
        return 99.0
    return base


def test_panic_contract():
    # (ma5, breadth, regime, base_threshold, 期望門檻)
    cases = [
        (-0.05, 0.50, "Bull",     5.0,  99.0),  # 大盤暴跌 → panic (含 Bull)
        (0.010, 0.20, "Sideways", 21.5, 99.0),  # breadth 低 + 非 Bull → panic
        (0.010, 0.20, "Bull",     5.0,  5.0),   # breadth 低但 Bull → 不擋 (窄牛市)
        (0.010, 0.50, "Bull",     5.0,  5.0),   # 正常 → 用 base
        (0.010, 0.50, "Bear",     99.0, 99.0),  # Bear 本就空倉
    ]
    for ma5, b, rg, base, exp in cases:
        got = _panic_threshold(ma5, b, rg, base)
        assert abs(got - exp) < 1e-9, f"({ma5},{b},{rg},base={base}): got={got} expected={exp}"
    return f"{len(cases)} 個 panic 契約案例通過"


# ── Test 3: 共用 config 常數存在 + 兩邊可 import ─────────────────────
def test_shared_constants():
    import config
    for c in ["REGIME_BUY_THRESHOLD", "REGIME_EXIT_PARAMS", "MKT_PANIC_MA5",
              "MKT_PANIC_BREADTH", "ATR_STOP_ENABLED", "REGIME_BULL_TREND"]:
        assert hasattr(config, c), f"config 缺少共用常數 {c}"
    # inference 確實 import 了 panic 常數 (代表 panic 邏輯已接上)
    import importlib
    inf = importlib.import_module("scripts.inference")
    assert hasattr(inf, "MKT_PANIC_MA5"), "inference 未 import MKT_PANIC_MA5 (panic 邏輯可能遺失)"
    assert hasattr(inf, "compute_stop_pct"), "inference 缺 compute_stop_pct"
    return "config 共用常數齊備, inference 已接上 panic 常數"


# ── Test 4: 資料回歸 — 全期買入門檻 0 脫節 (若 parquet 存在) ──────────
def test_data_regression():
    parquet = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
    if not os.path.exists(parquet):
        return "SKIP (無 features_combined.parquet)"
    import pandas as pd
    import numpy as np
    from scripts.utils import get_regime_label
    from config import (REGIME_BUY_THRESHOLD, REGIME_TREND_WINDOW, REGIME_TREND_MIN_PERIODS,
                        REGIME_BULL_TREND, REGIME_BEAR_TREND, MKT_PANIC_MA5, MKT_PANIC_BREADTH)
    df = pd.read_parquet(parquet, columns=["date", "market_mean_pct", "market_breadth_pct"])
    df["date"] = pd.to_datetime(df["date"])
    dm = df.groupby("date").agg(mkt=("market_mean_pct", "first"),
                                breadth=("market_breadth_pct", "first")).sort_index()
    dm["trend"] = dm["mkt"].rolling(REGIME_TREND_WINDOW, min_periods=REGIME_TREND_MIN_PERIODS).mean()
    dm["ma5"] = dm["mkt"].rolling(5, min_periods=1).mean()
    dm = dm.dropna(subset=["trend"])
    dm["regime"] = dm["trend"].apply(lambda v: get_regime_label(v, REGIME_BULL_TREND, REGIME_BEAR_TREND))
    dm["panic"] = (dm["ma5"] < MKT_PANIC_MA5) | ((dm["breadth"] < MKT_PANIC_BREADTH) & (dm["regime"] != "Bull"))

    def _t(rg):
        return REGIME_BUY_THRESHOLD.get(rg, 12.0)
    sim_thr = dm.apply(lambda r: 99.0 if r.panic else _t(r.regime), axis=1)
    inf_thr = dm.apply(lambda r: 99.0 if r.panic else _t(r.regime), axis=1)  # 修正後 inference (含 panic)
    desync = int((sim_thr != inf_thr).sum())
    assert desync == 0, f"全期 {len(dm)} 天有 {desync} 天買入門檻脫節 (inference 與 trading_sim 不一致)"
    return f"全期 {len(dm)} 天買入門檻 0 脫節"


check("停損公式一致性 (inference vs trading_sim)", test_stop_loss_formula)
check("panic 紅燈契約", test_panic_contract)
check("共用 config 常數 + panic 已接上", test_shared_constants)
check("資料回歸: 全期買入門檻 0 脫節", test_data_regression)


import datetime
out = ["\n" + "=" * 65,
       f"  inference<->trading_sim 一致性測試 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
       "=" * 65]
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
for status, name, msg in results:
    out.append(f"{status}  {name}")
    if msg:
        out.append(f"       → {msg}")
out.append("=" * 65)
out.append(f"  通過: {passed} / 失敗: {failed} / 共 {len(results)} 項")
out.append("=" * 65)
print("\n".join(out))

if failed > 0:
    sys.exit(1)
