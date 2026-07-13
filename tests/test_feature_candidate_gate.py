# -*- coding: utf-8 -*-
"""
test_feature_candidate_gate.py — 候選特徵（fortune 移植）潔淨 OOS 正式把關（完全隔離）
================================================================================
目的：比照 run_workflow_experiment.py 的「潔淨 OOS 候選 vs 現行」把關方法論，判定
      是否採用移植自 fortune-main 的 5 個候選特徵（close_z20/z60 + cal_dow/month/
      days_to_settle）。與該腳本不同處：那支比較的是「風控參數」候選，本支比較的是
      「特徵」候選——唯一變因是「模型有沒有這 5 欄」，其餘（切分、風控、參數）全同。

隔離保證（不影響原系統，全部在 sandbox 內）：
  * 生產 parquet 唯讀載入；所有 parquet/模型/feature_cols 只寫入 scratchpad sandbox。
  * 複用生產 `scripts/train.main` 與 `trading_sim.run_simulation`（與實盤邏輯 100% 一致），
    但把其 DATA_PATH / MODEL_DIR / FEATURE_COLS_PATH / EXCLUDE_FEATURES monkeypatch 到 sandbox。
  * trading_sim 以 export_report=False 呼叫 → 不寫 reports/。
  * config.py / data/ / models/ / reports/ 一個位元都不動。

潔淨 OOS：train 內建 config.BACKTEST_DATE(=2025-08-01) 截斷 → 2025-08 之後為未見樣本。

執行：python tests/test_feature_candidate_gate.py
"""
import os
import sys
import shutil
import contextlib
import io

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

SANDBOX = os.path.join(
    os.environ.get("TEMP", "/tmp"), "claude",
    "D--VScode-Stock-Stock", "c15ae241-a498-43c6-aa71-f743073b2186",
    "scratchpad", "gate_sandbox")
os.makedirs(SANDBOX, exist_ok=True)

PROD_PARQUET = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
NEW_FEATS = ["close_z20", "close_z60", "cal_dow", "cal_month", "cal_days_to_settle"]
Z_FEATS = ["close_z20", "close_z60"]
CAL_FEATS = ["cal_dow", "cal_month", "cal_days_to_settle"]

# 潔淨 OOS 窗（全在 BACKTEST_DATE=2025-08-01 之後）
WINDOWS = [
    ("2025Q3+", "2025-08-01", "2025-10-31"),
    ("2025Q4", "2025-11-01", "2026-01-31"),
    ("2026Q1", "2026-02-01", "2026-04-30"),
    ("2026Q2", "2026-05-01", "2026-06-30"),
    ("全OOS", "2025-08-01", "2026-06-30"),
]
CAPITAL = 1_000_000


def _quiet(fn, *a, **k):
    """吞掉生產腳本的大量 print，只留我們要的結果。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def build_sandbox_parquet():
    """讀生產 parquet（唯讀），用生產函式加 5 欄，寫入 sandbox。"""
    from feature_engineering import _add_calendar_features  # 生產碼
    df = pd.read_parquet(PROD_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    def _z(g):
        c = g["close"]
        for w in (20, 60):
            m = c.rolling(w, min_periods=w).mean()
            s = c.rolling(w, min_periods=w).std()
            g[f"close_z{w}"] = (c - m) / (s + 1e-9)
        return g
    df = df.groupby("stock_id", group_keys=False).apply(_z)
    df = _add_calendar_features(df)
    for c in NEW_FEATS:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    out = os.path.join(SANDBOX, "features_with_candidates.parquet")
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"  sandbox parquet: {df.shape[0]} 列 x {df.shape[1]} 欄（+{len(NEW_FEATS)} 候選欄）")
    return out


def train_variant(tag, parquet_path, exclude):
    """複用生產 train.main，I/O 全導向 sandbox。回傳該變體的模型目錄。"""
    import train as T
    mdl_dir = os.path.join(SANDBOX, f"models_{tag}")
    os.makedirs(mdl_dir, exist_ok=True)
    # monkeypatch 生產 train 的模組級路徑與排除清單
    T.DATA_PATH = parquet_path
    T.MODEL_DIR = mdl_dir
    T.FEATURE_COLS_PATH = os.path.join(mdl_dir, "feature_cols.json")
    T.EXCLUDE_FEATURES = exclude
    _quiet(T.main)
    return mdl_dir


def backtest(mdl_dir, parquet_path, start, end):
    """複用生產 run_simulation，讀 sandbox 模型/parquet，不寫 reports。"""
    import trading_sim as S
    S.DATA_PATH = parquet_path
    S.MODEL_DIR = mdl_dir
    ret, dd, _ = _quiet(S.run_simulation, start, end, CAPITAL, S.MAX_POSITIONS,
                        export_report=False)
    return ret, dd * 100.0   # run_simulation 的 dd 為小數，統一轉為百分比


def main():
    print("=" * 78)
    print("候選特徵潔淨 OOS 正式把關（隔離於 sandbox，不影響生產）")
    print(f"  sandbox: {SANDBOX}")
    print("=" * 78)

    parquet = build_sandbox_parquet()

    variants = {
        "baseline": NEW_FEATS,   # 排除全部新欄 = 現行
        "full": [],              # 含全部 5 新欄 = 候選
        "z_only": CAL_FEATS,     # 只含 z-score
        "cal_only": Z_FEATS,     # 只含日曆
    }
    print("\n訓練 4 變體（潔淨截斷 train ≤ 2025-08-01）...")
    mdirs = {}
    for tag, excl in variants.items():
        mdirs[tag] = train_variant(tag, parquet, excl)
        n = len(pd.read_json(os.path.join(mdirs[tag], "feature_cols.json")))
        print(f"  [{tag:8s}] 特徵數={n}")

    print("\n跨窗回測（報酬% / 回撤% / Calmar=報酬÷回撤）...")
    header = f"{'變體':<10}" + "".join(f"{w[0]:>16}" for w in WINDOWS)
    print(header)
    res = {t: {} for t in variants}
    for tag in variants:
        row = f"{tag:<10}"
        for wname, s, e in WINDOWS:
            ret, dd = backtest(mdirs[tag], parquet, s, e)
            res[tag][wname] = (ret, dd)
            calmar = ret / dd if dd else float("nan")
            row += f"{f'{ret:+.0f}/{dd:.1f}/{calmar:.1f}':>16}"
        print(row)

    # ── 判決 ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("判決（候選 full vs 現行 baseline）")
    print("=" * 78)
    b_full = res["baseline"]["全OOS"]
    c_full = res["full"]["全OOS"]
    print(f"  全OOS：baseline 報酬 {b_full[0]:+.1f}% / 回撤 {b_full[1]:.1f}%"
          f"  →  candidate 報酬 {c_full[0]:+.1f}% / 回撤 {c_full[1]:.1f}%")

    sub = [w[0] for w in WINDOWS if w[0] != "全OOS"]
    cand_wins = sum(res["full"][w][0] > res["baseline"][w][0] for w in sub)
    full_ret_better = c_full[0] > b_full[0]
    full_dd_better = c_full[1] < b_full[1]
    b_cal = b_full[0] / b_full[1]
    c_cal = c_full[0] / c_full[1]

    print(f"  全OOS Calmar：baseline {b_cal:.2f} → candidate {c_cal:.2f}"
          f"（{'改善' if c_cal > b_cal else '退步'}）")
    print(f"  子窗報酬勝出：candidate 於 {cand_wins}/{len(sub)} 個子窗贏過 baseline")

    if full_ret_better and full_dd_better and cand_wins >= len(sub) * 0.6:
        verdict = "✅ 通過：全OOS 報酬與回撤雙贏，且多數子窗穩健勝出 → 建議採用並重建生產特徵"
    elif c_cal > b_cal and cand_wins >= len(sub) * 0.5:
        verdict = "🟡 邊際通過：全OOS 風險調整較佳但子窗不穩定 → 可採用但需持續紙上驗證觀察"
    else:
        verdict = ("❌ 未通過：全OOS 看似改善多由少數窗驅動、子窗不穩健 → 疑過擬合，"
                   "維持現行、勿重建生產特徵")
    print(f"\n  {verdict}")
    # 拆解佐證
    print("\n  拆解佐證（各組是否單獨穩健）：")
    for tag in ("z_only", "cal_only"):
        t_full = res[tag]["全OOS"]
        print(f"    {tag:8s} 全OOS 報酬 {t_full[0]:+.1f}% / 回撤 {t_full[1]:.1f}%"
              f"（vs baseline {b_full[0]:+.1f}%/{b_full[1]:.1f}%）")
    print("=" * 78)


if __name__ == "__main__":
    main()
