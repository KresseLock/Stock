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
from joblib import Parallel, delayed

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
MA_WINDOWS           = [9, 20, 32, 52]
RSI_PERIOD           = 7
ATR_PERIOD           = 18
KD_PERIOD            = 15
MACD_FAST            = 14
MACD_SLOW            = 38
MACD_SIGNAL          = 5
BOLL_WINDOW          = 35
BOLL_STD_MULT        = 2.49
VOL_MA_WINDOW        = 8

# 籌碼滾動加總週期
CHIPS_SUM_WINDOWS    = [6, 15, 21]

# 預測天數
FORECAST_DAYS   = [1, 2, 3]

# 因子模組開關
ENABLE_CHIPS        = True
ENABLE_MARGIN       = True
ENABLE_SHAREHOLDING = True
ENABLE_FINMIND      = True
ENABLE_SENTIMENT    = True

# 多核心運算設定 (-1 = 使用全部核心)
N_JOBS              = -1



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
    try:
        from utils import load_target_stocks as _load
        return _load(file_path)
    except ImportError:
        import sys
        sys.path.append(os.path.join(BASE_DIR, ".."))
        from utils import load_target_stocks as _load
        return _load(file_path)

def _is_weekend(date_obj: datetime.date) -> bool:
    """判斷是否為週六(5)或週日(6)，台股不開盤，直接跳過"""
    return date_obj.weekday() >= 5


# ══════════════════════════════════════════════════════
# A. 技術面特徵
# ══════════════════════════════════════════════════════
def _compute_ta(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v, o = g["close"], g["high"], g["low"], g["volume"], g["open"]

    # 均線與乖離率 (由外部參數 MA_WINDOWS 控制)
    for w in MA_WINDOWS:
        g[f"ma{w}"] = c.rolling(w, min_periods=1).mean()
        g[f"bias{w}"] = (c - g[f"ma{w}"]) / (g[f"ma{w}"] + 1e-9)  # 乖離率 (Bias Ratio)
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

    # 標籤 (預測未來 N 天收益率)
    for d in FORECAST_DAYS:
        g[f"next_ret_{d}"] = c.shift(-d) / c - 1

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
    df_chips = df_chips.sort_values(["stock_id", "date"]).reset_index(drop=True)
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
        if df_m.empty:
            df_m = pd.DataFrame()
        else:
            df_m = df_m.rename(columns={"代號": "stock_id", "今日餘額": "margin_bal", "今日餘額.1": "short_bal", "資券互抵": "offset"})
            for col in ["margin_bal", "short_bal", "offset"]:
                if col in df_m.columns: df_m[col] = _to_float(df_m[col])
            df_m = df_m[["stock_id"] + [c for c in ["margin_bal", "short_bal", "offset"] if c in df_m.columns]]
    df_s = _read_csv(os.path.join(DATA_DIR, "raw_margin", f"{date_str}_sbl.csv"))
    if not df_s.empty:
        df_s["代號"] = df_s["代號"].astype(str).str.strip()
        df_s = df_s[df_s["代號"].isin(target_stocks)].copy()
        if df_s.empty:
            df_s = pd.DataFrame()
        else:
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
    res = (detail.groupby(["資料日期", "證券代號"])
                  .apply(_agg, include_groups=False)
                  .reset_index()
                  .rename(columns={"資料日期": "sh_date", "證券代號": "stock_id"}))
    
    # 集保資料通常為週五發布（最晚週六早），嚴格推移至下一個交易日 (BDay) 生效，避免週末前視偏差
    res["sh_date"] = res["sh_date"] + pd.tseries.offsets.BDay(1)
    return res
def _shift_finmind_date(d, rtype):
    """
    動態推移財報日期以消除前視偏差 (Look-ahead Bias)。
    """
    if rtype == "rev":
        # 很多公司10號晚上公告，嚴格延後到11號避免前視偏差
        return pd.Timestamp(year=d.year, month=d.month, day=11)
    elif rtype == "stmt":
        if d.month == 3: return pd.Timestamp(year=d.year, month=5, day=16)
        elif d.month == 6: return pd.Timestamp(year=d.year, month=8, day=15)
        elif d.month == 9: return pd.Timestamp(year=d.year, month=11, day=15)
        elif d.month == 12:
            return pd.Timestamp(year=d.year+1, month=4, day=1)
        else:
            return pd.NaT
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
        df_stmt = df_stmt.dropna(subset=["date"])
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
        df_bal = df_bal.dropna(subset=["date"])
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
        df_cf = df_cf.dropna(subset=["date"])
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
            if "當日沖銷交易成交股數" not in df_dt.columns:
                df_dt = pd.DataFrame()
            else:
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
            
            # 支援舊版 CT152816 與新版 MI_QFIIS 的欄位名稱
            col_name = "外資及陸資投資持股率"
            if "全體外資及陸資持股比率" in df_fh.columns:
                col_name = "全體外資及陸資持股比率"
                
            if col_name in df_fh.columns:
                df_fh = df_fh[df_fh["證券代號"].isin(target_stocks)][["證券代號", col_name]]
                df_fh = df_fh.rename(columns={"證券代號": "stock_id", col_name: "fini_holding_pct"})
                df_fh["fini_holding_pct"] = _to_float(df_fh["fini_holding_pct"])
                merged = pd.merge(merged, df_fh, on="stock_id", how="left")
            else:
                print(f"  [警告] fini_holding_pct 欄位找不到，現有欄位: {df_fh.columns.tolist()}")

    # C. 信用交易 (以及信用管制)
    if ENABLE_MARGIN:
        df_m = _load_margin_one_day(date_str, target_stocks)
        if not df_m.empty:
            merged = pd.merge(merged, df_m.drop(columns=["date"], errors="ignore"), on="stock_id", how="left")
            
        # 讀取信用限額
        df_cl = _read_csv(os.path.join(DATA_DIR, "raw_margin", f"{date_str}_credit_limit.csv"))
        if not df_cl.empty:
            df_cl["證券代號"] = df_cl["證券代號"].astype(str).str.strip()
            required_cols = ["融資限額", "融券限額"]
            if not all(c in df_cl.columns for c in required_cols):
                df_cl = pd.DataFrame()
            else:
                df_cl = df_cl[df_cl["證券代號"].isin(target_stocks)][["證券代號"] + required_cols]
                df_cl = df_cl.rename(columns={"證券代號": "stock_id", "融資限額": "margin_quota", "融券限額": "short_quota"})
                df_cl["margin_quota"] = _to_float(df_cl["margin_quota"])
                df_cl["short_quota"]  = _to_float(df_cl["short_quota"])
                merged = pd.merge(merged, df_cl, on="stock_id", how="left")

    # TWSE 官方版 PER/PBR
    df_per = _read_csv(os.path.join(DATA_DIR, "raw_twse_per", f"{date_str}_twse_per.csv"))
    if not df_per.empty:
        df_per["證券代號"] = df_per["證券代號"].astype(str).str.strip()
        required_cols = ["本益比", "股價淨值比", "殖利率(%)"]
        if not all(c in df_per.columns for c in required_cols):
            df_per = pd.DataFrame()
        else:
            df_per = df_per[df_per["證券代號"].isin(target_stocks)][["證券代號"] + required_cols]
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


def _transform_levels(g):
    # 信用交易變化
    for col in ["margin_bal", "short_bal", "sbl_bal"]:
        if col in g.columns:
            g[f"{col}_chg_5d"] = g[col].pct_change(5)
            g.drop(columns=[col], inplace=True)
            
    # 估值指標 Z-score (252天 = 一年)
    for col in ["PER", "PBR"]:
        if col in g.columns:
            g[f"{col}_zscore"] = (g[col] - g[col].rolling(252, min_periods=20).mean()) / (g[col].rolling(252, min_periods=20).std() + 1e-9)
            g.drop(columns=[col], inplace=True)
            
    # 財報成長率 (ffill 後，252天約為 YoY, 21天約為 MoM, 63天約為 QoQ)
    if "Revenue" in g.columns:
        g["Revenue_mom"] = g["Revenue"].pct_change(21)
        g["Revenue_yoy"] = g["Revenue"].pct_change(252)
    if "EPS" in g.columns:
        g["EPS_qoq"] = g["EPS"].pct_change(63)
        g["EPS_yoy"] = g["EPS"].pct_change(252)
    return g


def process_all_history_features(start_date_obj: datetime.date, end_date_obj: datetime.date, override_target_stocks: list = None):
    output_path = os.path.join(FEAT_DIR, "features_combined.parquet")
    
    if override_target_stocks is not None:
        target_stocks = override_target_stocks
    else:
        target_stocks = load_target_stocks()
        
    print(f"  目標股票: {len(target_stocks)} 檔")

    # Step 1: 逐日建立截面 (自動跳過週六週日與無開市的國定假日)
    dates_to_process = []
    delta = datetime.timedelta(days=1)
    curr = start_date_obj
    while curr <= end_date_obj:
        if not _is_weekend(curr):
            date_str = curr.strftime("%Y%m%d")
            # 判斷當天是否有收盤價日報 CSV，若無則代表是國定假日或無開市交易，直接跳過以提升效率
            price_path = os.path.join(DATA_DIR, "raw_price", f"{date_str}_price.csv")
            if os.path.exists(price_path):
                dates_to_process.append(date_str)
        curr += delta

    print(f"  準備處理 {len(dates_to_process)} 個交易日資料，啟動多核心平行處理...")
    all_dfs = Parallel(n_jobs=N_JOBS)(
        delayed(build_features)(d, target_stocks) for d in dates_to_process
    )
    all_dfs = [df for df in all_dfs if df is not None and not df.empty]

    if not all_dfs:
        print("  [錯誤] 找不到任何原始資料，中止。")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    def apply_parallel(df_grouped, func):
        res = Parallel(n_jobs=N_JOBS)(delayed(func)(group) for _, group in df_grouped)
        return pd.concat(res, ignore_index=True)

    # Step 2: 技術面 (A) + 法人籌碼衍生特徵 (B)
    print("  計算技術面特徵 (含乖離率)... (多核心運算中)")
    df = apply_parallel(df.groupby("stock_id"), _compute_ta)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    
    print("  計算相對大盤強弱指標 (Cross-Sectional RS) 與總體市場/板塊特徵 (方案 B)...")
    # 算出個股日漲跌幅
    df["close_pct"] = df.groupby("stock_id")["close"].pct_change()
    
    # 1. 計算總體市場特徵 (市場日均報酬、市場日上漲比例)
    market_stats = df.groupby("date").agg(
        market_mean_pct=("close_pct", "mean"),
        market_breadth_pct=("close_pct", lambda x: (x > 0).mean())
    ).reset_index()
    
    # 時間序列滾動平均 (避開未來資料)
    market_stats = market_stats.sort_values("date").reset_index(drop=True)
    market_stats["market_mean_ma5"] = market_stats["market_mean_pct"].rolling(5, min_periods=1).mean()
    market_stats["market_mean_ma20"] = market_stats["market_mean_pct"].rolling(20, min_periods=1).mean()
    market_stats["market_breadth_ma5"] = market_stats["market_breadth_pct"].rolling(5, min_periods=1).mean()
    market_stats["market_breadth_ma20"] = market_stats["market_breadth_pct"].rolling(20, min_periods=1).mean()
    
    # 合併市場特徵回主 df
    df = pd.merge(df, market_stats, on="date", how="left")
    
    # 計算相對大盤強弱 (RS)
    df["RS_1d"] = df["close_pct"] - df["market_mean_pct"]
    
    # 5日相對大盤強弱 (RS_5d)
    df["close_pct_5d"] = df.groupby("stock_id")["close"].pct_change(periods=5)
    df["RS_5d"] = df["close_pct_5d"] - df.groupby("date")["close_pct_5d"].transform("mean")
    
    # 2. 計算產業板塊情緒特徵 (Sector Strength)
    import json
    cat_path = os.path.join(BASE_DIR, "..", "stock_categories.json")
    stock_to_industry = {}
    if os.path.exists(cat_path):
        try:
            with open(cat_path, "r", encoding="utf-8") as f:
                cats = json.load(f)
            for ind, stocks in cats.items():
                for sid in stocks.keys():
                    stock_to_industry[sid] = ind
            print(f"  成功讀取 stock_categories.json，已建立 {len(stock_to_industry)} 檔股票的產業分類映射。")
        except Exception as e:
            print(f"  [警告] 讀取 stock_categories.json 失敗: {e}")
            
    if stock_to_industry:
        df["stock_industry"] = df["stock_id"].map(stock_to_industry).fillna("未分類")
        
        # 每日各板塊的平均報酬
        sector_mean = df.groupby(["date", "stock_industry"])["close_pct"].mean().reset_index(name="sector_mean_pct")
        
        # 時間序列滾動平均 (5天)
        sector_mean = sector_mean.sort_values(["stock_industry", "date"]).reset_index(drop=True)
        sector_mean["sector_mean_ma5"] = sector_mean.groupby("stock_industry")["sector_mean_pct"].transform(lambda x: x.rolling(5, min_periods=1).mean())
        
        # 合併回主 df
        df = pd.merge(df, sector_mean, on=["date", "stock_industry"], how="left")
        df = df.drop(columns=["stock_industry"])
        
    df = df.drop(columns=["close_pct", "close_pct_5d"])

    if ENABLE_CHIPS and "fini_net" in df.columns:
        print("  計算籌碼連續買賣超特徵...")
        df = _compute_chips_features(df)

    # Step 3: 持股分級 (D, 週更, ffill)
    if ENABLE_SHAREHOLDING:
        sh = _load_shareholding_all()
        if not sh.empty:
            print("  合入持股分級資料...")
            sh = sh[sh["stock_id"].isin(target_stocks)].copy()
            # 使用 pd.merge + ffill 避免 MultiIndex 產生超大稀疏矩陣
            base_dates = df[["stock_id", "date"]].drop_duplicates()
            sh = sh.rename(columns={"sh_date": "date"})
            sh_daily = pd.merge(base_dates, sh, on=["stock_id", "date"], how="left")
            sh_daily = sh_daily.sort_values(["stock_id", "date"])
            cols_to_fill = ["big_holder_pct", "small_holder_pct", "holder_hhi"]
            sh_daily[cols_to_fill] = sh_daily.groupby("stock_id")[cols_to_fill].ffill()
            df = pd.merge(df, sh_daily[["stock_id", "date"] + [c for c in cols_to_fill if c in sh_daily.columns]], on=["stock_id", "date"], how="left")

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

    # Step 5: Convert Level Features to Ratio/Change (避免模型記住股票身份)
    print("  轉換絕對值特徵為相對比例 (消除 Level Bias)... (多核心運算中)")
    df = apply_parallel(df.groupby("stock_id"), _transform_levels)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    
    # 丟棄其餘的絕對數值財報欄位與額度欄位
    level_cols = ["Revenue", "EPS", "TotalAssets", "Liabilities", "Equity", 
                  "GrossProfit", "OperatingIncome", "IncomeAfterTaxes", 
                  "CashFlowsFromOperatingActivities", "CashProvidedByInvestingActivities", 
                  "CashFlowsProvidedFromFinancingActivities", "cash_dividend",
                  "margin_quota", "short_quota"]
    df = df.drop(columns=[c for c in level_cols if c in df.columns], errors="ignore")

    # Step 6: 產生每日橫截面排序標籤 (Quantile Ranking Label) - 方案 C: 混合絕對與相對標籤
    print("  計算每日橫截面排序標籤 (Quantile Label)... (方案 C)")
    for d in FORECAST_DAYS:
        ret_col = f"next_ret_{d}"
        if ret_col in df.columns:
            # 每天在橫截面上針對未來的真實報酬做百分位排序
            rank = df.groupby("date")[ret_col].rank(pct=True)
            
            # 混合標籤設計：
            # 強勢股 (2): 相對排名前 20% 且「絕對報酬必須 > 0%」 (空頭崩盤時不勉強發出買入訊號)
            is_strong = (rank >= 0.8) & (df[ret_col] > 0.0)
            
            # 弱勢股 (0): 相對排名後 20% 或「絕對報酬 < -2.0%」 (即便大盤都跌，大跌的個股依然是弱勢)
            is_weak = (rank <= 0.2) | (df[ret_col] < -0.02)
            
            # 中性股 (1): 其他情況
            df[f"label_{d}"] = np.where(is_strong, 2, np.where(is_weak, 0, 1))
            
            # 處理原本 ret 為 NaN 的地方，讓 label 也保持 NaN
            df.loc[df[ret_col].isna(), f"label_{d}"] = np.nan

    # 剔除標籤缺失的最後 N 天
    label_cols = [f"next_ret_{d}" for d in FORECAST_DAYS if f"next_ret_{d}" in df.columns]
    label_cols += [f"label_{d}" for d in FORECAST_DAYS if f"label_{d}" in df.columns]
    if label_cols:
        pass # 保留最新這幾天的資料，不能在這裡 dropna，否則 inference.py 會永遠落後 3 天

    # 排序與存檔
    id_cols = ["stock_id", "date"]
    df = df[id_cols + [c for c in df.columns if c not in id_cols]]
    df.to_parquet(output_path, engine="pyarrow", index=False)

    print(f"  特徵矩陣建置完成！共 {df.shape[0]} 筆樣本，{df.shape[1]-2} 個特徵欄位。")
    print(f"  存至: {output_path}")