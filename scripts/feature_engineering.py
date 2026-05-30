"""
feature_engineering.py — 台灣股市特徵工程模組 (TWSE + FinMind 整合版)
====================================================
輸入: data/ 下的 TWSE 籌碼與 FinMind 財報 CSV
輸出: data/features/features_combined.parquet

特徵分組:
  A. 技術面 (OHLCV + TA)    — 均線、KD、RSI、MACD、布林帶、ATR、成交量比
  B. 法人籌碼                — 外資/投信/自營 買賣超、連續買超天數
  C. 信用交易 (融資券/借券)  — 融資餘額、融券餘額、借券餘額、資券比
  D. 持股分級 (TDCC)         — 大戶(百張+)比例、散戶比例、集中度
  E. 基本面與估值 (FinMind)  — 月營收、季報 (EPS/ROE/毛利率)、PER、PBR
  F. 市場情緒                — 當沖比、大盤法人淨買金額
  G. 標籤                    — 次日漲跌幅 (next_ret)，用於 LightGBM 分類/回歸
"""

import datetime
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FEAT_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════
def _to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)

def load_target_stocks(file_path: str = "Stocks.txt") -> list:
    fp = os.path.join(BASE_DIR, "..", file_path)
    if not os.path.exists(fp): fp = file_path
    if not os.path.exists(fp): return ["2330"]
    with open(fp, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ══════════════════════════════════════════════════════
# A. 技術面特徵
# ══════════════════════════════════════════════════════
def _compute_ta(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v, o = g["close"], g["high"], g["low"], g["volume"], g["open"]
    for w in [5, 10, 20, 60]: g[f"ma{w}"] = c.rolling(w, min_periods=1).mean()
    g["ma5_over_ma20"] = g["ma5"] / (g["ma20"] + 1e-9)
    std20 = c.rolling(20, min_periods=5).std()
    g["boll_up"]  = g["ma20"] + 2 * std20
    g["boll_dn"]  = g["ma20"] - 2 * std20
    g["boll_pct"] = (c - g["boll_dn"]) / (g["boll_up"] - g["boll_dn"] + 1e-9)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    g["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    g["macd"] = ema12 - ema26
    g["macd_sig"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_sig"]
    low9, high9 = l.rolling(9, min_periods=1).min(), h.rolling(9, min_periods=1).max()
    rsv = (c - low9) / (high9 - low9 + 1e-9) * 100
    g["k9"] = rsv.ewm(com=2, adjust=False).mean()
    g["d9"] = g["k9"].ewm(com=2, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=1).mean()
    g["atr14_pct"] = g["atr14"] / (c + 1e-9)
    g["vol_ma5"] = v.rolling(5, min_periods=1).mean()
    g["vol_ratio5"] = v / (g["vol_ma5"] + 1)
    g["ret1"], g["ret5"] = c.pct_change(1), c.pct_change(5)
    g["amplitude"] = (h - l) / (o + 1e-9)
    # 預測未來 1, 2, 3 天的累積報酬率
    g["next_ret_1"] = (c.shift(-1) / c) - 1
    g["next_ret_2"] = (c.shift(-2) / c) - 1
    g["next_ret_3"] = (c.shift(-3) / c) - 1
    return g

# ══════════════════════════════════════════════════════
# B. 法人籌碼特徵
# ══════════════════════════════════════════════════════
def _load_chips_one_day(date_str: str, target_stocks: list) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_chips.csv")
    df = _read_csv(path)
    if df.empty: return pd.DataFrame()
    df["證券代號"] = df["證券代號"].astype(str).str.strip()
    df = df[df["證券代號"].isin(target_stocks)].copy()
    if df.empty: return pd.DataFrame()
    col_map = {"證券代號": "stock_id", "外陸資買賣超股數(不含外資自營商)": "fini_net", 
               "投信買賣超股數": "sitc_net", "自營商買賣超股數": "dealer_net", "三大法人買賣超股數": "inst_net_total"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ["fini_net", "sitc_net", "dealer_net", "inst_net_total"]:
        if col in df.columns: df[col] = _to_float(df[col])
    df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
    return df[["stock_id", "date"] + [c for c in ["fini_net", "sitc_net", "dealer_net", "inst_net_total"] if c in df.columns]]

def _compute_chips_features(df_chips: pd.DataFrame) -> pd.DataFrame:
    df_chips = df_chips.sort_values(["stock_id", "date"])
    for col in ["fini_net", "sitc_net", "dealer_net", "inst_net_total"]:
        if col not in df_chips.columns: continue
        sign = np.sign(df_chips.groupby("stock_id")[col].transform(lambda x: x))
        df_chips[f"{col}_streak"] = sign.groupby(df_chips["stock_id"]).transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        ) * sign
        for w in [3, 5, 10]:
            df_chips[f"{col}_sum{w}"] = df_chips.groupby("stock_id")[col].transform(lambda x: x.rolling(w, min_periods=1).sum())
    return df_chips

# ══════════════════════════════════════════════════════
# C. 信用交易特徵
# ══════════════════════════════════════════════════════
def _load_margin_one_day(date_str: str, target_stocks: list) -> pd.DataFrame:
    df_m = _read_csv(os.path.join(DATA_DIR, "raw_margin", f"{date_str}_margin.csv"))
    if not df_m.empty:
        df_m["代號"] = df_m["代號"].astype(str).str.strip()
        df_m = df_m[df_m["代號"].isin(target_stocks)].copy()
        df_m = df_m.rename(columns={"代號": "stock_id", "今日餘額": "margin_bal", "今日餘額.1": "short_bal", "資券互抵": "offset"})
        for col in ["margin_bal", "short_bal", "offset"]:
            if col in df_m.columns: df_m[col] = _to_float(df_m[col])
        df_m = df_m[["stock_id"] + [c for c in ["margin_bal", "short_bal", "offset"] if c in df_m.columns]]
    df_s = _read_csv(os.path.join(DATA_DIR, "raw_margin", f"{date_str}_sbl.csv"))
    if not df_s.empty:
        df_s["代號"] = df_s["代號"].astype(str).str.strip()
        df_s = df_s[df_s["代號"].isin(target_stocks)].copy()
        df_s = df_s.rename(columns={"代號": "stock_id", "當日餘額": "sbl_bal", "當日賣出": "sbl_sell"})
        for col in ["sbl_bal", "sbl_sell"]:
            if col in df_s.columns: df_s[col] = _to_float(df_s[col])
        df_s = df_s[["stock_id"] + [c for c in ["sbl_bal", "sbl_sell"] if c in df_s.columns]]
    
    if df_m.empty and df_s.empty: return pd.DataFrame()
    elif df_m.empty: df = df_s
    elif df_s.empty: df = df_m
    else: df = pd.merge(df_m, df_s, on="stock_id", how="outer")
    df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
    return df

# ══════════════════════════════════════════════════════
# D. 持股分級特徵
# ══════════════════════════════════════════════════════
def _load_shareholding_all() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "raw_shareholding", "*_shareholding.csv")))
    if not paths: return pd.DataFrame()
    dfs = []
    for p in paths:
        df = _read_csv(p)
        if df.empty: continue
        df["證券代號"] = df["證券代號"].astype(str).str.strip()
        df["持股分級"] = pd.to_numeric(df["持股分級"], errors="coerce")
        df["比例"]     = _to_float(df["占集保庫存數比例%"])
        df["資料日期"] = pd.to_datetime(df["資料日期"].astype(str), format="%Y%m%d", errors="coerce")
        dfs.append(df)
    if not dfs: return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    detail = raw[raw["持股分級"] < 17].copy()

    def _agg(grp):
        total_pct = grp["比例"].sum()
        return pd.Series({
            "big_holder_pct":   grp[grp["持股分級"] >= 11]["比例"].sum(),
            "small_holder_pct": grp[grp["持股分級"] <= 2]["比例"].sum(),
            "holder_hhi":       ((grp["比例"] / (total_pct + 1e-9)) ** 2).sum(),
        })
    return detail.groupby(["資料日期", "證券代號"]).apply(_agg).reset_index().rename(columns={"資料日期": "sh_date", "證券代號": "stock_id"})


# ══════════════════════════════════════════════════════
# E. 基本面與估值 (FinMind)
# ══════════════════════════════════════════════════════
def _load_finmind_fundamentals(stock_id: str, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    df_out = pd.DataFrame({"date": all_dates, "stock_id": stock_id})
    
    # 1. 月營收 (Monthly)
    rev_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_monthly_revenue.csv")
    df_rev = _read_csv(rev_path)
    if not df_rev.empty:
        df_rev["date"] = pd.to_datetime(df_rev["date"])
        df_rev["revenue"] = _to_float(df_rev["revenue"])
        df_rev = df_rev[["date", "revenue"]].drop_duplicates("date")
        df_out = pd.merge(df_out, df_rev, on="date", how="outer").sort_values("date")
        df_out["revenue"] = df_out["revenue"].ffill()
        
    # 2. 財務報表 (Quarterly)
    stmt_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_financial_stmt.csv")
    df_stmt = _read_csv(stmt_path)
    if not df_stmt.empty:
        df_stmt["date"] = pd.to_datetime(df_stmt["date"])
        df_stmt["value"] = _to_float(df_stmt["value"])
        piv = df_stmt.pivot_table(index="date", columns="type", values="value").reset_index()
        keep_cols = ["date"]
        for c in ["EPS", "毛利率", "營業利益率", "稅後淨利率", "ROE", "ROA"]:
            if c in piv.columns: keep_cols.append(c)
        piv = piv[keep_cols]
        df_out = pd.merge(df_out, piv, on="date", how="outer").sort_values("date")
        for c in keep_cols:
            if c != "date": df_out[c] = df_out[c].ffill()
                
    # 3. 本益比估值 (Daily)
    per_path = os.path.join(DATA_DIR, "raw_per", f"{stock_id}_per.csv")
    df_per = _read_csv(per_path)
    if not df_per.empty:
        df_per["date"] = pd.to_datetime(df_per["date"])
        df_per["PER"] = _to_float(df_per["PER"])
        df_per["PBR"] = _to_float(df_per["PBR"])
        df_per["dividend_yield"] = _to_float(df_per["dividend_yield"])
        df_per = df_per[["date", "PER", "PBR", "dividend_yield"]].drop_duplicates("date")
        df_out = pd.merge(df_out, df_per, on="date", how="left")
        
    df_out = df_out[df_out["date"].isin(all_dates)].copy()
    return df_out


# ══════════════════════════════════════════════════════
# F. 市場情緒特徵
# ══════════════════════════════════════════════════════
def _load_market_sentiment(date_str: str) -> dict:
    result = {}
    dt_path = os.path.join(DATA_DIR, "raw_daytrading", f"{date_str}_daytrading.csv")
    df_dt = _read_csv(dt_path)
    if not df_dt.empty:
        pct_col = next((c for c in df_dt.columns if "比重" in c and "股數" in c), None)
        if pct_col: result["daytrading_pct"] = _to_float(df_dt[pct_col]).iloc[0]

    inst_path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_inst_total.csv")
    df_inst = _read_csv(inst_path)
    if not df_inst.empty and "買賣差額" in df_inst.columns:
        total_row = df_inst[df_inst["單位名稱"].astype(str).str.contains("合計", na=False)]
        if not total_row.empty: result["mkt_inst_net"] = _to_float(total_row["買賣差額"]).iloc[0]
    return result

# ══════════════════════════════════════════════════════
# 主建置函式
# ══════════════════════════════════════════════════════
def build_features(date_str: str, target_stocks: list) -> pd.DataFrame:
    df_p = _read_csv(os.path.join(DATA_DIR, "raw_price", f"{date_str}_price.csv"))
    if df_p.empty: return None
    df_p["證券代號"] = df_p["證券代號"].astype(str).str.strip()
    df_p = df_p[df_p["證券代號"].isin(target_stocks)].copy()
    if df_p.empty: return None

    df_p = df_p.rename(columns={"證券代號": "stock_id", "開盤價": "open", "最高價": "high", 
                                "最低價": "low", "收盤價": "close", "成交股數": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_p.columns: df_p[col] = _to_float(df_p[col])
    df_p["date"] = pd.to_datetime(date_str, format="%Y%m%d")

    df_c = _load_chips_one_day(date_str, target_stocks)
    df_m = _load_margin_one_day(date_str, target_stocks)
    sentiment = _load_market_sentiment(date_str)

    merged = df_p
    if not df_c.empty: merged = pd.merge(merged, df_c.drop(columns=["date"], errors="ignore"), on="stock_id", how="left")
    if not df_m.empty: merged = pd.merge(merged, df_m.drop(columns=["date"], errors="ignore"), on="stock_id", how="left")
    for k, v in sentiment.items(): merged[k] = v
    return merged

def process_all_history_features(start_date_obj: datetime.date, end_date_obj: datetime.date):
    output_path = os.path.join(FEAT_DIR, "features_combined.parquet")
    target_stocks = load_target_stocks()
    print(f"  目標股票: {len(target_stocks)} 檔")

    # Step 1: 逐日建立截面
    delta = datetime.timedelta(days=1)
    curr, all_dfs = start_date_obj, []
    while curr <= end_date_obj:
        day_df = build_features(curr.strftime("%Y%m%d"), target_stocks)
        if day_df is not None and not day_df.empty: all_dfs.append(day_df)
        curr += delta

    if not all_dfs:
        print("  [錯誤] 找不到任何原始資料，中止。")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # Step 2: 技術面與籌碼衍生特徵
    print("  計算技術面與連續買賣超特徵...")
    df = df.groupby("stock_id", group_keys=False).apply(_compute_ta)
    if "fini_net" in df.columns: df = _compute_chips_features(df)

    # Step 3: 合入持股分級 (週更, ffill)
    sh = _load_shareholding_all()
    if not sh.empty:
        print("  合入持股分級資料...")
        sh = sh[sh["stock_id"].isin(target_stocks)].copy()
        idx = pd.MultiIndex.from_product([sorted(df["stock_id"].unique()), sorted(df["date"].unique())], names=["stock_id", "date"])
        sh_daily = sh.set_index(["stock_id", "sh_date"]).reindex(idx).groupby(level=0).ffill().reset_index().rename(columns={"date": "date"})
        sh_cols = ["stock_id","date","big_holder_pct","small_holder_pct","holder_hhi"]
        df = pd.merge(df, sh_daily[[c for c in sh_cols if c in sh_daily.columns]], on=["stock_id","date"], how="left")

    # Step 4: 合入 FinMind 基本面與估值 (月/季更 ffill, 日更 join)
    print("  合入 FinMind 財報與估值特徵...")
    fm_dfs = []
    for stock_id in target_stocks:
        stock_dates = df[df["stock_id"] == stock_id]["date"]
        fm_dfs.append(_load_finmind_fundamentals(stock_id, stock_dates))
    if fm_dfs:
        fm_all = pd.concat(fm_dfs, ignore_index=True)
        df = pd.merge(df, fm_all, on=["stock_id", "date"], how="left")

    # 剔除標籤缺失的最後三天(因為需要未來三天的股價才能算標籤)
    if "next_ret_3" in df.columns: df = df.dropna(subset=["next_ret_1", "next_ret_2", "next_ret_3"])

    # 排序與存檔
    id_cols = ["stock_id", "date"]
    df = df[id_cols + [c for c in df.columns if c not in id_cols]]
    df.to_parquet(output_path, engine="pyarrow", index=False)
    
    print(f"  特徵矩陣建置完成！共 {df.shape[0]} 筆樣本，{df.shape[1]-2} 個特徵欄位。")
    print(f"  存至: {output_path}")