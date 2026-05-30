"""
scraper.py — 台灣股市多源資料爬蟲 (整合 TWSE + FinMind)
====================================================
資料來源: 
  1. 台灣證券交易所 (TWSE) + 集保中心 (TDCC) [免費無限制]
  2. FinMind [需 Token，負責財報與估值]

儲存路徑:
  data/raw_price/         - 股價行情 (TWSE)
  data/raw_chips/         - 法人買賣超 (TWSE)
  data/raw_margin/        - 融資券、借券 (TWSE)
  data/raw_daytrading/    - 當沖統計 (TWSE)
  data/raw_shareholding/  - 持股分級 (TDCC)
  data/raw_financial/     - 財報、月營收 (FinMind)
  data/raw_per/           - 本益比估值 (FinMind)
"""

import datetime
import io
import os
import random
import time
import warnings
import pandas as pd
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

try:
    from taiwan_holidays.taiwan_calendar import TaiwanCalendar
    tw_cal = TaiwanCalendar()
except ImportError:
    tw_cal = None


# ── 建立所有資料夾 ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

DIRS = [
    "raw_price", "raw_chips", "raw_margin", 
    "raw_daytrading", "raw_shareholding",
    "raw_financial", "raw_per"
]
for folder in DIRS:
    os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

# ── 共用 Headers & 工具 ─────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

def _clean_html(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"<[^>]+>", "", regex=True)

def _already_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def _save_csv(df: pd.DataFrame, path: str):
    _clean_html(df).to_csv(path, index=False, encoding="utf-8-sig")

def _polite_sleep(lo: float = 1.5, hi: float = 3.0):
    time.sleep(random.uniform(lo, hi))


# =====================================================================
# 模組 1: TWSE 每日全市場資料 (免費無限制)
# =====================================================================

def _fetch_twse_json(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200: return None
        body = r.text.strip()
        if not body or body[0] == "<": return None
        return r.json()
    except:
        return None

def crawl_daily_price(date_str: str) -> bool:
    path = os.path.join(DATA_DIR, "raw_price", f"{date_str}_price.csv")
    if _already_exists(path): return True
    url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={date_str}&type=ALL"
    data = _fetch_twse_json(url)
    if not data: return False
    
    df = None
    if "tables" in data:
        for tbl in data["tables"]:
            if "fields" in tbl and "證券代號" in tbl.get("fields", []):
                df = pd.DataFrame(tbl["data"], columns=tbl["fields"])
                break
    elif "data9" in data:
        df = pd.DataFrame(data["data9"], columns=data["fields9"])
        
    if df is None or df.empty: return False
    _save_csv(df, path)
    return True

def crawl_daily_chips(date_str: str) -> bool:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_chips.csv")
    if _already_exists(path): return True
    url = f"https://www.twse.com.tw/fund/T86?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data and data.get("data"):
        _save_csv(pd.DataFrame(data["data"], columns=data["fields"]), path)
        return True
    return False

def crawl_daily_institution_total(date_str: str) -> bool:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_inst_total.csv")
    if _already_exists(path): return True
    url = f"https://www.twse.com.tw/fund/BFI82U?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data and data.get("data"):
        _save_csv(pd.DataFrame(data["data"], columns=data["fields"]), path)
        return True
    return False

def crawl_daily_margin(date_str: str) -> bool:
    path = os.path.join(DATA_DIR, "raw_margin", f"{date_str}_margin.csv")
    if _already_exists(path): return True
    url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data and data.get("tables") and len(data["tables"]) >= 2 and data["tables"][1].get("data"):
        tbl = data["tables"][1]
        _save_csv(pd.DataFrame(tbl["data"], columns=tbl["fields"]), path)
        return True
    return False

def crawl_daily_sbl(date_str: str) -> bool:
    path = os.path.join(DATA_DIR, "raw_margin", f"{date_str}_sbl.csv")
    if _already_exists(path): return True
    url = f"https://www.twse.com.tw/exchangeReport/TWT93U?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data and data.get("data"):
        _save_csv(pd.DataFrame(data["data"], columns=data["fields"]), path)
        return True
    return False

def crawl_daily_daytrading(date_str: str) -> bool:
    path = os.path.join(DATA_DIR, "raw_daytrading", f"{date_str}_daytrading.csv")
    if _already_exists(path): return True
    url = f"https://www.twse.com.tw/exchangeReport/TWTB4U?response=json&date={date_str}&selectType=MS"
    data = _fetch_twse_json(url)
    if data and data.get("tables") and data["tables"][0].get("data"):
        tbl = data["tables"][0]
        _save_csv(pd.DataFrame(tbl["data"], columns=tbl["fields"]), path)
        return True
    return False

def crawl_weekly_shareholding() -> bool:
    url = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, verify=False)
        if r.status_code != 200: return False
        df = pd.read_csv(io.StringIO(r.text))
        df["證券代號"] = df["證券代號"].astype(str).str.strip()
        dates = df["資料日期"].unique()
        if len(dates) == 0: return False
        date_str = str(dates[0])
        path = os.path.join(DATA_DIR, "raw_shareholding", f"{date_str}_shareholding.csv")
        if not _already_exists(path):
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  [持股分級] {date_str} 下載成功 ({len(df)} 筆)")
        return True
    except:
        return False

def _is_holiday(date_obj: datetime.date) -> bool:
    if tw_cal is None: return False
    try: return tw_cal.is_holiday(date_obj)
    except: return False


# =====================================================================
# 模組 2: FinMind 個股財報與進階估值 (需 Token)
# =====================================================================

def _get_finmind_token():
    token_path = os.path.join(BASE_DIR, "..", "FINMIND_TOKEN.txt")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            t = f.read().strip()
            if t: return t
    return os.environ.get("FINMIND_TOKEN", "")

FINMIND_TOKEN = _get_finmind_token()
FM_BASE_URL = "https://api.finmindtrade.com/api/v4/data"

def _fm_get(dataset: str, start_date: str, end_date: str, data_id: str, max_retry=3):
    params = {"dataset": dataset, "start_date": start_date, "end_date": end_date, "data_id": data_id}
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"} if FINMIND_TOKEN else {}
    
    for attempt in range(max_retry):
        try:
            r = requests.get(FM_BASE_URL, params=params, headers=headers, timeout=30)
            if r.status_code in (429, 402):
                wait = 60 * (attempt + 1)
                print(f"    [FinMind] 限速中，等待 {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 200 and r.json().get("status") == 200:
                data = r.json().get("data", [])
                if data: return pd.DataFrame(data)
            break
        except:
            time.sleep(5)
    return pd.DataFrame()

def _crawl_fm_dataset(dataset_name: str, stock_id: str, start_date: str, end_date: str, output_path: str):
    """通用的 FinMind 增量下載邏輯"""
    if _already_exists(output_path):
        existing = pd.read_csv(output_path, encoding="utf-8-sig")
        last_date = pd.to_datetime(existing["date"]).max()
        start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if start_date > end_date:
            return True
            
    df = _fm_get(dataset_name, start_date, end_date, stock_id)
    if df.empty: return False

    if os.path.exists(output_path):
        old = pd.read_csv(output_path, encoding="utf-8-sig")
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["date"])

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return True


# =====================================================================
# 主下載控制
# =====================================================================

def download_history_data(start_date_obj: datetime.date, end_date_obj: datetime.date, target_stocks: list = None):
    """
    批次下載 TWSE 與 FinMind 所有資料
    """
    delta = datetime.timedelta(days=1)
    
    # ── 1. 下載 TWSE 每日全市場資料 ──
    print("\n[啟動] TWSE 全市場資料爬蟲 (免 Token)")
    crawl_weekly_shareholding()
    
    curr = start_date_obj
    while curr <= end_date_obj:
        d_str = curr.strftime("%Y%m%d")
        if _is_holiday(curr):
            curr += delta
            continue
            
        print(f"  抓取 TWSE {d_str}...", end="\r")
        if crawl_daily_price(d_str):
            crawl_daily_chips(d_str)
            crawl_daily_institution_total(d_str)
            crawl_daily_margin(d_str)
            crawl_daily_sbl(d_str)
            crawl_daily_daytrading(d_str)
            _polite_sleep()
        curr += delta
    print("\n  TWSE 爬蟲完成。")


    # ── 2. 下載 FinMind 個股財報與估值 ──
    if not target_stocks:
        return
        
    start_str = start_date_obj.strftime("%Y-%m-%d")
    end_str = end_date_obj.strftime("%Y-%m-%d")
    
    print("\n[啟動] FinMind 基本面資料爬蟲")
    if not FINMIND_TOKEN:
        print("  ⚠️ 尚未設定 FINMIND_TOKEN.txt，使用免費額度 (300次/小時)。")

    for stock_id in target_stocks:
        print(f"  抓取 FinMind {stock_id}...", end=" ")
        
        # 月營收
        p_rev = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_monthly_revenue.csv")
        _crawl_fm_dataset("TaiwanStockMonthRevenue", stock_id, start_str, end_str, p_rev)
        _polite_sleep(1, 2)
        
        # 綜合損益表 (EPS, 毛利率等)
        p_stmt = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_financial_stmt.csv")
        _crawl_fm_dataset("TaiwanStockFinancialStatements", stock_id, start_str, end_str, p_stmt)
        _polite_sleep(1, 2)
        
        # 本益比 (PER, PBR, Yield)
        p_per = os.path.join(DATA_DIR, "raw_per", f"{stock_id}_per.csv")
        _crawl_fm_dataset("TaiwanStockPER", stock_id, start_str, end_str, p_per)
        _polite_sleep(1, 2)
        
        print("OK")
    print("  FinMind 爬蟲完成。")