# -*- coding: utf-8 -*-
"""
test_chip_flow_normalization.py — 法人籌碼特徵改造 潔淨 OOS 正式把關（完全隔離）
================================================================================
比照 test_feature_candidate_gate.py 的方法論，但這裡的「候選」不是新增欄位，而是
「改造既有籌碼特徵」，唯一變因是模型看到的籌碼欄是原始版還是改造版，其餘（切分、
風控、參數、非籌碼特徵）全同。

兩項改造（本實驗把兩項合併，並各自單獨拆解佐證）：
  item1 成交量正規化：fini/sitc/dealer/inst 的「淨額(股數)」與「滾動合計」除以當日成交量
                     → 消除大小股不可比的 Level Bias；_streak 本就是天數(scale-free)故沿用。
  item2 自營商純化  ：dealer_net(總額 = 自行買賣 + 避險造市) 換成僅「自行買賣」淨額
                     → 剔除權證避險的機械性雜訊（實測 2330 避險佔比逾 6 成）。

四變體：
  baseline    = 現行生產（20 個原始籌碼特徵）
  norm_only   = 只做 item1（四法人皆正規化，dealer 仍用總額）
  dealer_only = 只做 item2（dealer 換自行買賣原始股數，其餘維持原始）
  full        = item1 + item2

隔離保證（不影響生產）：
  * 生產 parquet 唯讀；所有 parquet/模型/feature_cols 只寫入系統暫存 sandbox。
  * 複用生產 train.main 與 trading_sim.run_simulation（與實盤 100% 一致），
    僅 monkeypatch 其 DATA_PATH/MODEL_DIR/FEATURE_COLS_PATH/EXCLUDE_FEATURES 到 sandbox。
  * trading_sim 以 export_report=False 呼叫，不寫 reports/。config.py/data/models 一位元不動。

潔淨 OOS：train 內建 config.BACKTEST_DATE(=2025-08-01) 截斷 → 之後為未見樣本。

執行：python tests/test_chip_flow_normalization.py
"""
import os
import sys
import glob
import tempfile
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

SANDBOX = os.path.join(tempfile.gettempdir(), "stock_chipflow_sandbox")
os.makedirs(SANDBOX, exist_ok=True)

PROD_PARQUET = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
RAW_CHIPS_DIR = os.path.join(ROOT_DIR, "data", "raw_chips")
CAPITAL = 1_000_000

INSTS = ["fini_net", "sitc_net", "dealer_net", "inst_net_total"]
DEALER_SELF_RAW_COL = "自營商買賣超股數(自行買賣)"

# 潔淨 OOS 窗（全在 BACKTEST_DATE=2025-08-01 之後），與 fortune 把關一致以利對照
WINDOWS = [
    ("2025Q3+", "2025-08-01", "2025-10-31"),
    ("2025Q4", "2025-11-01", "2026-01-31"),
    ("2026Q1", "2026-02-01", "2026-04-30"),
    ("2026Q2", "2026-05-01", "2026-06-30"),
    ("全OOS", "2025-08-01", "2026-06-30"),
]


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def _streak_and_sums(df, base_col, windows):
    """完全複製生產 _compute_chips_features 的 streak 與 rolling-sum 公式（df 須先按 stock_id,date 排序）。"""
    sign = np.sign(df[base_col])
    tmp = pd.DataFrame({"sid": df["stock_id"].values, "s": sign.values}, index=df.index)
    df[f"{base_col}_streak"] = (
        tmp.groupby("sid")["s"].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        ) * sign
    )
    for w in windows:
        df[f"{base_col}_sum{w}"] = df.groupby("stock_id")[base_col].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )


def _load_dealer_self(target_stocks, date_set):
    """從原始 chips.csv 逐日取『自營商買賣超股數(自行買賣)』，回傳 (stock_id,date,dealer_self_net)。"""
    frames = []
    files = sorted(glob.glob(os.path.join(RAW_CHIPS_DIR, "*_chips.csv")))
    hit = 0
    for fp in files:
        d = os.path.basename(fp).split("_")[0]
        if d not in date_set:
            continue
        try:
            raw = pd.read_csv(fp, dtype=str, usecols=["證券代號", DEALER_SELF_RAW_COL])
        except (ValueError, Exception):
            continue  # 舊格式無此欄 → 該日略過（留 NaN）
        raw["證券代號"] = raw["證券代號"].astype(str).str.strip()
        raw = raw[raw["證券代號"].isin(target_stocks)]
        if raw.empty:
            continue
        raw["dealer_self_net"] = pd.to_numeric(
            raw[DEALER_SELF_RAW_COL].str.replace(",", "", regex=False), errors="coerce"
        )
        raw["date"] = pd.to_datetime(d, format="%Y%m%d")
        frames.append(raw[["證券代號", "date", "dealer_self_net"]].rename(columns={"證券代號": "stock_id"}))
        hit += 1
    print(f"  自行買賣原始檔命中 {hit} 天")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_sandbox_parquet():
    df = pd.read_parquet(PROD_PARQUET)
    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    windows = sorted(int(c.split("sum")[1]) for c in df.columns if c.startswith("fini_net_sum"))
    print(f"  chips sum 視窗（沿用生產）：{windows}")

    vol = df["volume"].replace(0, np.nan)

    # ── item1：四法人 淨額與滾動合計 正規化（除以當日成交量） ──
    for src in INSTS:
        norm = f"{src}_norm"
        df[norm] = (df[src] / vol).replace([np.inf, -np.inf], np.nan)
        for w in windows:
            df[f"{norm}_sum{w}"] = df.groupby("stock_id")[norm].transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
        # norm 的 streak 與原始同號，直接沿用既有 *_streak，不重算

    # ── item2：自營商『自行買賣』原始股數 → 併入 ──
    date_set = set(df["date"].dt.strftime("%Y%m%d"))
    ds = _load_dealer_self(set(df["stock_id"].unique()), date_set)
    if ds.empty:
        raise RuntimeError("找不到任何『自行買賣』原始資料，無法進行 item2。")
    df = df.merge(ds, on=["stock_id", "date"], how="left")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    cover = df["dealer_self_net"].notna().mean()
    print(f"  dealer_self_net 覆蓋率：{cover:.1%}")

    # dealer_self 原始版（dealer_only 用）：完整 streak+sums
    _streak_and_sums(df, "dealer_self_net", windows)
    # dealer_self 正規化版（full 用）
    df["dealer_self_net_norm"] = (df["dealer_self_net"] / vol).replace([np.inf, -np.inf], np.nan)
    _streak_and_sums(df, "dealer_self_net_norm", windows)

    out = os.path.join(SANDBOX, "features_chipflow.parquet")
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"  sandbox parquet：{df.shape[0]} 列 x {df.shape[1]} 欄")
    return out, windows


def _sumcols(base, windows):
    return [f"{base}_sum{w}" for w in windows]


def build_variant_active_sets(windows):
    """每個變體要『訓練用』的籌碼欄集合；EXCLUDE = 全籌碼欄宇宙 − 該集合。"""
    def fam_raw(src):      # 原始：level + streak + sums
        return [src, f"{src}_streak"] + _sumcols(src, windows)
    def fam_norm(src):     # 正規化：norm level + 原 streak(沿用) + norm sums
        return [f"{src}_norm", f"{src}_streak"] + _sumcols(f"{src}_norm", windows)

    baseline = []
    for s in INSTS:
        baseline += fam_raw(s)

    norm_only = []
    for s in INSTS:
        norm_only += fam_norm(s)

    dealer_only = fam_raw("fini_net") + fam_raw("sitc_net") + fam_raw("inst_net_total") \
        + ["dealer_self_net", "dealer_self_net_streak"] + _sumcols("dealer_self_net", windows)

    full = fam_norm("fini_net") + fam_norm("sitc_net") + fam_norm("inst_net_total") \
        + ["dealer_self_net_norm", "dealer_self_net_norm_streak"] + _sumcols("dealer_self_net_norm", windows)

    active = {"baseline": baseline, "norm_only": norm_only,
              "dealer_only": dealer_only, "full": full}
    universe = sorted(set().union(*active.values()))
    return active, universe


def train_variant(tag, parquet_path, exclude):
    import train as T
    mdl_dir = os.path.join(SANDBOX, f"models_{tag}")
    os.makedirs(mdl_dir, exist_ok=True)
    T.DATA_PATH = parquet_path
    T.MODEL_DIR = mdl_dir
    T.FEATURE_COLS_PATH = os.path.join(mdl_dir, "feature_cols.json")
    T.EXCLUDE_FEATURES = exclude
    _quiet(T.main)
    return mdl_dir


def backtest(mdl_dir, parquet_path, start, end):
    import trading_sim as S
    S.DATA_PATH = parquet_path
    S.MODEL_DIR = mdl_dir
    ret, dd, _ = _quiet(S.run_simulation, start, end, CAPITAL, S.MAX_POSITIONS,
                        export_report=False)
    return ret, dd * 100.0


def main():
    print("=" * 78)
    print("法人籌碼特徵改造 潔淨 OOS 正式把關（隔離於 sandbox，不影響生產）")
    print(f"  sandbox: {SANDBOX}")
    print("=" * 78)

    parquet, windows = build_sandbox_parquet()
    active, universe = build_variant_active_sets(windows)

    print("\n訓練 4 變體（潔淨截斷 train ≤ 2025-08-01）...")
    mdirs = {}
    for tag in ("baseline", "norm_only", "dealer_only", "full"):
        exclude = [c for c in universe if c not in active[tag]]
        mdirs[tag] = train_variant(tag, parquet, exclude)
        import json as _json
        n = len(_json.load(open(os.path.join(mdirs[tag], "feature_cols.json"), encoding="utf-8")))
        print(f"  [{tag:11s}] 特徵數={n}（籌碼欄 {len(active[tag])}，其餘同 baseline）")

    print("\n跨窗回測（報酬% / 回撤% / Calmar=報酬÷回撤）...")
    header = f"{'變體':<12}" + "".join(f"{w[0]:>16}" for w in WINDOWS)
    print(header)
    res = {t: {} for t in active}
    for tag in ("baseline", "norm_only", "dealer_only", "full"):
        row = f"{tag:<12}"
        for wname, s, e in WINDOWS:
            ret, dd = backtest(mdirs[tag], parquet, s, e)
            res[tag][wname] = (ret, dd)
            calmar = ret / dd if dd else float("nan")
            row += f"{f'{ret:+.0f}/{dd:.1f}/{calmar:.1f}':>16}"
        print(row)

    # ── 判決 ──
    print("\n" + "=" * 78)
    print("判決（full vs baseline）")
    print("=" * 78)
    b = res["baseline"]["全OOS"]
    c = res["full"]["全OOS"]
    b_cal = b[0] / b[1] if b[1] else float("nan")
    c_cal = c[0] / c[1] if c[1] else float("nan")
    print(f"  全OOS：baseline 報酬 {b[0]:+.1f}% / 回撤 {b[1]:.1f}% / Calmar {b_cal:.2f}")
    print(f"         full     報酬 {c[0]:+.1f}% / 回撤 {c[1]:.1f}% / Calmar {c_cal:.2f}")

    sub = [w[0] for w in WINDOWS if w[0] != "全OOS"]
    wins = sum(res["full"][w][0] > res["baseline"][w][0] for w in sub)
    ret_better = c[0] > b[0]
    dd_better = c[1] < b[1]
    print(f"  子窗報酬勝出：full 於 {wins}/{len(sub)} 個子窗贏過 baseline")

    if ret_better and dd_better and wins >= len(sub) * 0.6:
        verdict = "✅ 通過：全OOS 報酬與回撤雙贏且多數子窗穩健 → 建議移植進生產"
    elif c_cal > b_cal and wins >= len(sub) * 0.5:
        verdict = "🟡 邊際通過：風險調整較佳但子窗不穩定 → 可移植但需持續紙上驗證"
    else:
        verdict = "❌ 未通過：改善多由少數窗驅動或雙輸 → 疑過擬合，維持現行"
    print(f"\n  {verdict}")

    print("\n  拆解佐證（哪一項在驅動）：")
    for tag in ("norm_only", "dealer_only"):
        t = res[tag]["全OOS"]
        print(f"    {tag:11s} 全OOS 報酬 {t[0]:+.1f}% / 回撤 {t[1]:.1f}%"
              f"（vs baseline {b[0]:+.1f}%/{b[1]:.1f}%）")
    print("=" * 78)


if __name__ == "__main__":
    main()
