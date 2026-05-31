# -*- coding: utf-8 -*-
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
  G. 標籤                    — 未來 N 天報酬率，用於 LightGBM 分類/回歸

注意: 以下模組級別的全域變數可由 run_feature_engineering.py 覆寫，
      以便從外部靈活調整所有因子參數，無需修改本檔案。
"""

import datetime
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FEAT_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# 可由外部覆寫的全域參數 (預設值)
# 實際生效的值由 run_feature_engineering.py 的設定區決定
# ══════════════════════════════════════════════════════

# 技術指標參數
MA_WINDOWS      = [5, 10, 20, 60]
RSI_PERIOD      = 14
ATR_PERIOD      = 14
KD_PERIOD       = 9
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
VOL_MA_WINDOW   = 5
BOLL_WINDOW     = 20
BOLL_STD_MULT   = 2.0

# 籌碼滾動加總週期
CHIPS_SUM_WINDOWS = [3, 5, 10]

# 預測天數
FORECAST_DAYS   = [1, 2, 3]

# 因子模組開關
ENABLE_CHIPS        = True
ENABLE_MARGIN       = True
ENABLE_SHAREHOLDING = True
ENABLE_FINMIND      = True
ENABLE_SENTIMENT    = True


# ══════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════
def _to_float(series: pd.Series) -> pd.Series:
    # 移除千分位逗號，其餘交給 pd.to_numeric 處理 (遇到 "--" 或無法轉換的字串會安全地變成 NaN)
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def load_target_stocks(file_path: str = "Stocks.txt") -> list:
    fp = os.path.join(BASE_DIR, "..", file_path)
    if not os.path.exists(fp): fp = file_path
    if not os.path.exists(fp): return ["2330"]
    with open(fp, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _is_weekend(date_obj: datetime.date) -> bool:
    """判斷是否為週六(5)或週日(6)，台股不開盤，直接跳過"""
    return date_obj.weekday() >= 5


# ══════════════════════════════════════════════════════
# A. 技術面特徵
# ══════════════════════════════════════════════════════
def _compute_ta(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v, o = g["close"], g["high"], g["low"], g["volume"], g["open"]

    # 均線 (由外部參數 MA_WINDOWS 控制)
    for w in MA_WINDOWS:
        g[f"ma{w}"] = c.rolling(w, min_periods=1).mean()
    if len(MA_WINDOWS) >= 2:
        short, long_ = MA_WINDOWS[0], MA_WINDOWS[2] if len(MA_WINDOWS) > 2 else MA_WINDOWS[-1]
        g["ma_short_over_long"] = g[f"ma{short}"] / (g[f"ma{long_}"] + 1e-9)

    # 布林通道
    std_boll = c.rolling(BOLL_WINDOW, min_periods=5).std()
    g["boll_mid"] = c.rolling(BOLL_WINDOW, min_periods=1).mean()
    g["boll_up"]  = g["boll_mid"] + BOLL_STD_MULT * std_boll
    g["boll_dn"]  = g["boll_mid"] - BOLL_STD_MULT * std_boll
    g["boll_pct"] = (c - g["boll_dn"]) / (g["boll_up"] - g["boll_dn"] + 1e-9)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD, min_periods=1).mean()
    g[f"rsi{RSI_PERIOD}"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    g["macd"]      = ema_fast - ema_slow
    g["macd_sig"]  = g["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_sig"]

    # KD 隨機指標
    low_n  = l.rolling(KD_PERIOD, min_periods=1).min()
    high_n = h.rolling(KD_PERIOD, min_periods=1).max()
    rsv = (c - low_n) / (high_n - low_n + 1e-9) * 100
    g[f"k{KD_PERIOD}"] = rsv.ewm(com=2, adjust=False).mean()
    g[f"d{KD_PERIOD}"] = g[f"k{KD_PERIOD}"].ewm(com=2, adjust=False).mean()

    # ATR (真實波幅)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g[f"atr{ATR_PERIOD}"]     = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    g[f"atr{ATR_PERIOD}_pct"] = g[f"atr{ATR_PERIOD}"] / (c + 1e-9)

    # 成交量
    g[f"vol_ma{VOL_MA_WINDOW}"]    = v.rolling(VOL_MA_WINDOW, min_periods=1).mean()
    g[f"vol_ratio{VOL_MA_WINDOW}"] = v / (g[f"vol_ma{VOL_MA_WINDOW}"] + 1)

    # 報酬率
    g["ret1"]      = c.pct_change(1)
    g["ret5"]      = c.pct_change(5)
    g["amplitude"] = (h - l) / (o + 1e-9)

    # 標籤：未來 N 天累積報酬率
    for day in FORECAST_DAYS:
        g[f"next_ret_{day}"] = (c.shift(-day) / c) - 1

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
        sign_col = f"{col}_sign"
        df_chips[sign_col] = np.sign(df_chips[col])
        df_chips[f"{col}_streak"] = df_chips.groupby("stock_id")[sign_col].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        ) * df_chips[sign_col]
        df_chips.drop(columns=[sign_col], inplace=True)
        for w in CHIPS_SUM_WINDOWS:
            df_chips[f"{col}_sum{w}"] = df_chips.groupby("stock_id")[col].transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
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
    return detail.groupby(["資料日期", "證券代號"]).apply(_agg).reset_index().rename(
        columns={"資料日期": "sh_date", "證券代號": "stock_id"})


def _shift_finmind_date(d, rtype):
    """
    動態推移財報日期以消除前視偏差 (Look-ahead Bias)。
    """
    if rtype == "rev":
        # FinMind 月營收的 date 為當月 1 號 (如 2020-01-01)，代表去年 12 月的營收。
        # 該筆資料最晚於 1 月 10 日發布，所以直接推到同月 10 號即可。
        return pd.Timestamp(year=d.year, month=d.month, day=10)
    elif rtype == "stmt":
        if d.month == 3: return pd.Timestamp(year=d.year, month=5, day=15)
        elif d.month == 6: return pd.Timestamp(year=d.year, month=8, day=14)
        elif d.month == 9: return pd.Timestamp(year=d.year, month=11, day=14)
        elif d.month == 12:
            # 台灣年報(Q4)最晚申報截止日為隔年 3/31。
            # 雖然部分大型股(如台積電)可能提早在2月甚至1月公告，
            # 但為了絕對避免前視偏差，此處刻意採用最保守的 3/31 作為邊界，不作提前。
            return pd.Timestamp(year=d.year+1, month=3, day=31)
        return d + pd.Timedelta(days=45)
    return d

# ══════════════════════════════════════════════════════
# E. 基本面與估值 (FinMind)
# ══════════════════════════════════════════════════════
def _load_finmind_fundamentals(stock_id: str, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    merge_dfs = []

    # 1. 月營收
    rev_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_monthly_revenue.csv")
    df_rev = _read_csv(rev_path)
    if not df_rev.empty:
        df_rev["date"]    = pd.to_datetime(df_rev["date"]).apply(lambda x: _shift_finmind_date(x, "rev"))
        df_rev["revenue"] = _to_float(df_rev["revenue"])
        merge_dfs.append(df_rev[["date", "revenue"]].drop_duplicates("date"))

    # 2. 綜合損益表 (季報)
    stmt_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_financial_stmt.csv")
    df_stmt = _read_csv(stmt_path)
    if not df_stmt.empty:
        df_stmt["date"]  = pd.to_datetime(df_stmt["date"]).apply(lambda x: _shift_finmind_date(x, "stmt"))
        df_stmt["value"] = _to_float(df_stmt["value"])
        piv = df_stmt.pivot_table(index="date", columns="type", values="value").reset_index()
        keep_cols = ["date"]
        for c in ["EPS", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes", "Revenue"]:
            if c in piv.columns: keep_cols.append(c)
        merge_dfs.append(piv[keep_cols])

    # 3. 資產負債表 (季報)
    bal_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_balance_sheet.csv")
    df_bal = _read_csv(bal_path)
    if not df_bal.empty:
        df_bal["date"]  = pd.to_datetime(df_bal["date"]).apply(lambda x: _shift_finmind_date(x, "stmt"))
        df_bal["value"] = _to_float(df_bal["value"])
        piv = df_bal.pivot_table(index="date", columns="type", values="value").reset_index()
        keep_cols = ["date"]
        for c in ["TotalAssets", "Liabilities", "Equity"]:
            if c in piv.columns: keep_cols.append(c)
        merge_dfs.append(piv[keep_cols])

    # 4. 現金流量表 (季報)
    cf_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_cashflow.csv")
    df_cf = _read_csv(cf_path)
    if not df_cf.empty:
        df_cf["date"]  = pd.to_datetime(df_cf["date"]).apply(lambda x: _shift_finmind_date(x, "stmt"))
        df_cf["value"] = _to_float(df_cf["value"])
        piv = df_cf.pivot_table(index="date", columns="type", values="value").reset_index()
        keep_cols = ["date"]
        for c in ["CashFlowsFromOperatingActivities", "CashProvidedByInvestingActivities", "CashFlowsProvidedFromFinancingActivities"]:
            if c in piv.columns: keep_cols.append(c)
        merge_dfs.append(piv[keep_cols])

    # 5. 股利政策 (年/季)
    div_path = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_dividend.csv")
    df_div = _read_csv(div_path)
    if not df_div.empty:
        # 注：股利(cash_dividend)通常來自宣告日或除權息日，FinMind 提供之 date 多為已知發布日
        # 若為決議年度，後續除息日也多半於當年 6~8 月前發生，因此不強制推移，直接使用其 date。
        df_div["date"]  = pd.to_datetime(df_div["date"])
        df_div["value"] = _to_float(df_div["CashEarningsDistribution"]) if "CashEarningsDistribution" in df_div.columns else np.nan
        df_div = df_div[["date", "value"]].dropna().drop_duplicates("date").rename(columns={"value": "cash_dividend"})
        merge_dfs.append(df_div)

    if not merge_dfs:
        return pd.DataFrame({"date": all_dates, "stock_id": stock_id})

    # 先組合所有可能的日期 (交易日 + 財報發布日)，再一次性 merge 與 ffill
    all_report_dates = pd.concat([mdf["date"] for mdf in merge_dfs]).drop_duplicates()
    full_dates = pd.DatetimeIndex(sorted(set(all_dates) | set(all_report_dates)))
    df_out = pd.DataFrame({"date": full_dates, "stock_id": stock_id})

    for mdf in merge_dfs:
        df_out = pd.merge(df_out, mdf, on="date", how="left")

    df_out = df_out.sort_values("date").ffill()
    df_out = df_out[df_out["date"].isin(all_dates)].copy()
    return df_out


# ══════════════════════════════════════════════════════
# F. 市場情緒特徵
# ══════════════════════════════════════════════════════
def _load_market_sentiment(date_str: str) -> dict:
    result = {}
    
    # 1. 讀取期交所外資台指期未平倉淨額
    taifex_path = os.path.join(DATA_DIR, "raw_taifex", f"{date_str}_taifex_inst.csv")
    df_tf = _read_csv(taifex_path)
    if not df_tf.empty and "商品名稱" in df_tf.columns and "身份別" in df_tf.columns and "多空未平倉口數淨額" in df_tf.columns:
        # 尋找 商品名稱='TXF' 且 身份別包含'外資'
        txf = df_tf[df_tf["商品名稱"].astype(str).str.contains("TXF", na=False)]
        fini_txf = txf[txf["身份別"].astype(str).str.contains("外資", na=False)]
        if not fini_txf.empty:
            result["taifex_txf_fini_net_oi"] = _to_float(fini_txf["多空未平倉口數淨額"]).iloc[0]

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

    merged = df_p

    # B. 法人籌碼 (以及外資持股、當沖)
    if ENABLE_CHIPS:
        df_c = _load_chips_one_day(date_str, target_stocks)
        if not df_c.empty:
            merged = pd.merge(merged, df_c.drop(columns=["date"], errors="ignore"), on="stock_id", how="left")
            
        # 讀取當沖
        df_dt = _read_csv(os.path.join(DATA_DIR, "raw_chips", f"{date_str}_daytrading.csv"))
        if not df_dt.empty:
            df_dt["證券代號"] = df_dt["證券代號"].astype(str).str.strip()
            df_dt = df_dt[df_dt["證券代號"].isin(target_stocks)][["證券代號", "當日沖銷交易成交股數"]]
            df_dt = df_dt.rename(columns={"證券代號": "stock_id", "當日沖銷交易成交股數": "daytrading_vol"})
            df_dt["daytrading_vol"] = _to_float(df_dt["daytrading_vol"])
            merged = pd.merge(merged, df_dt, on="stock_id", how="left")
            if "volume" in merged.columns:
                merged["daytrading_pct"] = merged["daytrading_vol"] / (merged["volume"] + 1e-9)

        # 讀取外資持股
        df_fh = _read_csv(os.path.join(DATA_DIR, "raw_chips", f"{date_str}_fini_holding.csv"))
        if not df_fh.empty:
            df_fh["證券代號"] = df_fh["證券代號"].astype(str).str.strip()
            df_fh = df_fh[df_fh["證券代號"].isin(target_stocks)][["證券代號", "外資及陸資投資持股率"]]
            df_fh = df_fh.rename(columns={"證券代號": "stock_id", "外資及陸資投資持股率": "fini_holding_pct"})
            df_fh["fini_holding_pct"] = _to_float(df_fh["fini_holding_pct"])
            merged = pd.merge(merged, df_fh, on="stock_id", how="left")

    # C. 信用交易 (以及信用管制)
    if ENABLE_MARGIN:
        df_m = _load_margin_one_day(date_str, target_stocks)
        if not df_m.empty:
            merged = pd.merge(merged, df_m.drop(columns=["date"], errors="ignore"), on="stock_id", how="left")
            
        # 讀取信用限額
        df_cl = _read_csv(os.path.join(DATA_DIR, "raw_margin", f"{date_str}_credit_limit.csv"))
        if not df_cl.empty:
            df_cl["證券代號"] = df_cl["證券代號"].astype(str).str.strip()
            df_cl = df_cl[df_cl["證券代號"].isin(target_stocks)][["證券代號", "融資限額", "融券限額"]]
            df_cl = df_cl.rename(columns={"證券代號": "stock_id", "融資限額": "margin_quota", "融券限額": "short_quota"})
            df_cl["margin_quota"] = _to_float(df_cl["margin_quota"])
            df_cl["short_quota"]  = _to_float(df_cl["short_quota"])
            merged = pd.merge(merged, df_cl, on="stock_id", how="left")

    # TWSE 官方版 PER/PBR
    df_per = _read_csv(os.path.join(DATA_DIR, "raw_twse_per", f"{date_str}_twse_per.csv"))
    if not df_per.empty:
        df_per["證券代號"] = df_per["證券代號"].astype(str).str.strip()
        df_per = df_per[df_per["證券代號"].isin(target_stocks)][["證券代號", "本益比", "股價淨值比", "殖利率(%)"]]
        df_per = df_per.rename(columns={"證券代號": "stock_id", "本益比": "PER", "股價淨值比": "PBR", "殖利率(%)": "dividend_yield"})
        for c in ["PER", "PBR", "dividend_yield"]:
            df_per[c] = _to_float(df_per[c])
        merged = pd.merge(merged, df_per, on="stock_id", how="left")

    # F. 市場情緒 (大盤級)
    if ENABLE_SENTIMENT:
        sentiment = _load_market_sentiment(date_str)
        for k, v in sentiment.items():
            merged[k] = v

    return merged


def process_all_history_features(start_date_obj: datetime.date, end_date_obj: datetime.date, override_target_stocks: list = None):
    output_path = os.path.join(FEAT_DIR, "features_combined.parquet")
    
    if override_target_stocks is not None:
        target_stocks = override_target_stocks
    else:
        target_stocks = load_target_stocks()
        
    print(f"  目標股票: {len(target_stocks)} 檔")

    # Step 1: 逐日建立截面 (自動跳過週六週日)
    delta = datetime.timedelta(days=1)
    curr, all_dfs = start_date_obj, []
    while curr <= end_date_obj:
        if not _is_weekend(curr):
            day_df = build_features(curr.strftime("%Y%m%d"), target_stocks)
            if day_df is not None and not day_df.empty:
                all_dfs.append(day_df)
        curr += delta

    if not all_dfs:
        print("  [錯誤] 找不到任何原始資料，中止。")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # Step 2: 技術面 (A) + 法人籌碼衍生特徵 (B)
    print("  計算技術面特徵...")
    df = df.groupby("stock_id", group_keys=False).apply(_compute_ta)
    if ENABLE_CHIPS and "fini_net" in df.columns:
        print("  計算籌碼連續買賣超特徵...")
        df = _compute_chips_features(df)

    # Step 3: 持股分級 (D, 週更, ffill)
    if ENABLE_SHAREHOLDING:
        sh = _load_shareholding_all()
        if not sh.empty:
            print("  合入持股分級資料...")
            sh = sh[sh["stock_id"].isin(target_stocks)].copy()
            idx = pd.MultiIndex.from_product(
                [sorted(df["stock_id"].unique()), sorted(df["date"].unique())],
                names=["stock_id", "date"]
            )
            sh_daily = (sh.rename(columns={"sh_date": "date"})
                          .set_index(["stock_id", "date"])
                          .reindex(idx)
                          .groupby(level=0).ffill()
                          .reset_index())
            sh_cols = ["stock_id", "date", "big_holder_pct", "small_holder_pct", "holder_hhi"]
            df = pd.merge(df, sh_daily[[c for c in sh_cols if c in sh_daily.columns]], on=["stock_id", "date"], how="left")

    # Step 4: FinMind 基本面 (E, 月/季更 ffill)
    if ENABLE_FINMIND:
        print("  合入 FinMind 財報與估值特徵...")
        fm_dfs = []
        for stock_id in target_stocks:
            stock_dates = df[df["stock_id"] == stock_id]["date"]
            fm_dfs.append(_load_finmind_fundamentals(stock_id, stock_dates))
        if fm_dfs:
            fm_all = pd.concat(fm_dfs, ignore_index=True)
            df = pd.merge(df, fm_all, on=["stock_id", "date"], how="left")

    # 剔除標籤缺失的最後 N 天
    label_cols = [f"next_ret_{d}" for d in FORECAST_DAYS if f"next_ret_{d}" in df.columns]
    if label_cols:
        df = df.dropna(subset=label_cols)

    # 排序與存檔
    id_cols = ["stock_id", "date"]
    df = df[id_cols + [c for c in df.columns if c not in id_cols]]
    df.to_parquet(output_path, engine="pyarrow", index=False)

    print(f"  特徵矩陣建置完成！共 {df.shape[0]} 筆樣本，{df.shape[1]-2} 個特徵欄位。")
    print(f"  存至: {output_path}")