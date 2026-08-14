# -*- coding: utf-8 -*-
"""
test_factor_params_contract.py — 因子參數落地契約測試
================================================================================
守的是一條契約：**best_factors.json 宣告的技術指標參數，必須真的出現在特徵矩陣裡。**

2026-08-14 以前這條契約是破的：feature_engineering._compute_ta 讀模組全域，
但它在 joblib 子行程執行，子行程重新 import 模組後看不到 auto_pipeline
在父行程做的 setattr，於是 MA/RSI/ATR/KD/MACD/Boll/VolMA 全部靜默退回預設值，
只有在父行程計算的 CHIPS_SUM_WINDOWS 生效。
（背景與影響：tests/FACTOR_OBJECTIVE_PLAN.md §1）

這類失效不會拋錯、不會有警告，只能靠契約測試抓。

執行：
    python tests/test_factor_params_contract.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROD_PARQUET = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
BEST_FACTORS = os.path.join(ROOT_DIR, "configs", "best_factors.json")


def _synthetic_ohlcv(n_stocks=3, n_days=300, seed=0):
    """純合成 OHLCV，不碰任何生產資料。"""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    for i in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        rows.append(pd.DataFrame({
            "stock_id": f"T{i:04d}",
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.005, n_days)),
            "high": close * (1 + abs(rng.normal(0, 0.01, n_days))),
            "low": close * (1 - abs(rng.normal(0, 0.01, n_days))),
            "close": close,
            "volume": rng.integers(1_000, 100_000, n_days).astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


# ── 測試 1：參數必須穿透 joblib 子行程 ────────────────────────────────
def test_ta_params_survive_multiprocessing():
    import scripts.feature_engineering as fe

    p = dict(fe.current_factor_params())
    p.update({"MA_WINDOWS": [3, 8, 21, 55], "RSI_PERIOD": 11, "ATR_PERIOD": 9,
              "KD_PERIOD": 6, "VOL_MA_WINDOW": 4})

    df = _synthetic_ohlcv()
    out = pd.concat(
        Parallel(n_jobs=2)(delayed(partial(fe._compute_ta, p=p))(g)
                           for _, g in df.groupby("stock_id")),
        ignore_index=True,
    )

    expected = ["ma3", "ma55", "bias3", "rsi11", "atr9", "atr9_pct",
                "k6", "d6", "vol_ma4", "vol_ratio4", "atr_pct"]
    missing = [c for c in expected if c not in out.columns]
    assert not missing, f"因子參數未穿透子行程，缺少欄位：{missing}"

    leaked = [c for c in ("ma9", "ma20", "rsi7", "atr18", "k15", "vol_ma8")
              if c in out.columns]
    assert not leaked, f"仍在使用模組預設值，出現不該有的欄位：{leaked}"

    # 正規別名必須等於帶週期的那一欄
    assert np.allclose(out["atr_pct"].fillna(0), out["atr9_pct"].fillna(0)), \
        "atr_pct 別名與 atr9_pct 不一致"

    print("  [PASS] 測試 1：因子參數穿透 joblib 子行程，atr_pct 別名正確")


# ── 測試 2：APPLY_BEST_FACTORS_TA=False 能重現修復前行為 ──────────────
def test_apply_switch_reverts_to_defaults():
    import scripts.feature_engineering as fe

    orig_rsi, orig_switch = fe.RSI_PERIOD, fe.APPLY_BEST_FACTORS_TA
    try:
        fe.RSI_PERIOD = 99                      # 模擬 auto_pipeline 的 setattr
        fe.APPLY_BEST_FACTORS_TA = False
        assert fe.current_factor_params()["RSI_PERIOD"] == fe._TA_DEFAULTS["RSI_PERIOD"], \
            "開關關閉時未退回模組預設值"

        fe.APPLY_BEST_FACTORS_TA = True
        assert fe.current_factor_params()["RSI_PERIOD"] == 99, \
            "開關開啟時未採用覆寫值"
    finally:
        fe.RSI_PERIOD, fe.APPLY_BEST_FACTORS_TA = orig_rsi, orig_switch

    print("  [PASS] 測試 2：APPLY_BEST_FACTORS_TA 開關雙向有效")


# ── 測試 3：生產特徵檔的欄名與 best_factors.json 一致 ─────────────────
def test_production_parquet_matches_effective_params():
    """生產特徵檔的欄名，必須與「目前設定下實際會生效的參數」一致。

    比對基準隨 APPLY_BEST_FACTORS_TA 變動：
      True  → 技術指標欄名須反映 best_factors.json
      False → 技術指標欄名須為模組預設值（刻意維持修復前行為）
    籌碼視窗一律取自 best_factors.json（它在父行程計算，不受該開關影響）。

    這支測試存在的理由：2026-08-14 之前 json 說 A、特徵是 B，沒有任何機制會叫出來。
    """
    if not (os.path.exists(PROD_PARQUET) and os.path.exists(BEST_FACTORS)):
        print("  [SKIP] 測試 3：找不到生產特徵檔或 best_factors.json")
        return True

    import pyarrow.parquet as pq
    import scripts.feature_engineering as fe

    cols = set(pq.ParquetFile(PROD_PARQUET).schema.names)
    with open(BEST_FACTORS, encoding="utf-8") as f:
        bf = json.load(f).get("best_params_for_run_feature_engineering", {})
    if not bf:
        print("  [SKIP] 測試 3：best_factors.json 無參數內容")
        return True

    # 模擬 auto_pipeline._apply_best_params 的覆寫後，取實際生效值
    orig = {k: getattr(fe, k) for k in fe._TA_OVERRIDABLE_KEYS}
    try:
        for k in fe._TA_OVERRIDABLE_KEYS:
            if k in bf:
                setattr(fe, k, bf[k])
        eff = fe.current_factor_params()
    finally:
        for k, v in orig.items():
            setattr(fe, k, v)

    mode = "best_factors" if fe.APPLY_BEST_FACTORS_TA else "模組預設值(APPLY_BEST_FACTORS_TA=False)"
    checks = [(f"ma{w}", f"MA_WINDOWS[{i}]={w}") for i, w in enumerate(eff["MA_WINDOWS"])]
    checks += [
        (f"rsi{eff['RSI_PERIOD']}",       f"RSI_PERIOD={eff['RSI_PERIOD']}"),
        (f"atr{eff['ATR_PERIOD']}_pct",   f"ATR_PERIOD={eff['ATR_PERIOD']}"),
        (f"k{eff['KD_PERIOD']}",          f"KD_PERIOD={eff['KD_PERIOD']}"),
        (f"vol_ma{eff['VOL_MA_WINDOW']}", f"VOL_MA_WINDOW={eff['VOL_MA_WINDOW']}"),
    ]
    checks += [(f"fini_net_sum{w}", f"CHIPS_SUM_WINDOWS 含 {w}") for w in bf["CHIPS_SUM_WINDOWS"]]

    bad = [f"{why}（缺 {col}）" for col, why in checks if col not in cols]
    if bad:
        print(f"  [FAIL] 測試 3：生產特徵檔與實際生效參數不一致（基準：{mode}）")
        for b in bad:
            print(f"          - {b}")
        print("          → 請執行 auto_pipeline.py -s feature 重建特徵矩陣")
        return False

    print(f"  [PASS] 測試 3：生產特徵檔欄名與實際生效參數一致（基準：{mode}）")
    if "atr_pct" not in cols:
        print("         [注意] 特徵檔尚無 atr_pct 正規別名（重建後才會有）；"
              "風控端已由 get_atr_pct_col() 退回 atr<N>_pct，行為不受影響")
    return True


def main():
    print("=" * 78)
    print("因子參數落地契約測試")
    print("=" * 78)
    ok = True
    test_ta_params_survive_multiprocessing()
    test_apply_switch_reverts_to_defaults()
    ok &= test_production_parquet_matches_effective_params()
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
