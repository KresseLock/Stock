"""
scraper.py — 台灣股市多源資料爬蟲 (整合 TWSE + TAIFEX + FinMind)
====================================================
資料來源分流 (T+1/T+2 量化藍圖): 
  1. 台灣證券交易所 (TWSE) [免費每日]：價量、籌碼、資券、借券、本益比、當沖、外資持股、信用管制
  2. 台灣期貨交易所 (TAIFEX) [免費每日]：外資台指期未平倉 (大盤多空指標)
  3. 集保中心 (TDCC) [免費每週]：大戶持股分級
  4. FinMind [需 Token 每月/季]：月營收、綜合損益表、資產負債表、現金流量表、股利

儲存路徑:
  data/raw_price/        - 股價行情 (TWSE)
  data/raw_chips/        - 法人買賣超、當沖、外資持股 (TWSE)
  data/raw_margin/       - 融資券、借券、信用管制 (TWSE)
  data/raw_twse_per/     - 官方版個股本益比/PBR (TWSE)
  data/raw_taifex/       - 期貨三大法人未平倉 (TAIFEX)
  data/raw_shareholding/ - 持股分級 (TDCC)
  data/raw_financial/    - 財報、月營收、股利 (FinMind)

skip_dates.json 說明:
  key 格式: "{dataset}_{YYYYMMDD}"
  reason:
    "market_closed"    - price API 確認當天休市 (颱風假/補假等)，整天所有 dataset 同步標記
    "no_data"          - 該 dataset 明確回傳無資料 (例如當沖制度實施前)
    "unexpected_format"- API 有回應但格式不符預期，暫時跳過等格式確認後可手動清除
    "empty_response"   - API 有回應但資料為空
"""

import datetime
import io
import json
import os
import random
import time
import warnings
import pandas as pd
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

try:
    from taiwan_holidays.taiwan_calendar import TaiwanCalendar
    _th = TaiwanCalendar()
    _check_holiday = _th.is_holiday
except Exception:
    _check_holiday = lambda d: False


# ── 建立所有資料夾 ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

DIRS = [
    "raw_price", "raw_chips", "raw_margin",
    "raw_twse_per", "raw_taifex", "raw_shareholding",
    "raw_financial"
]
for folder in DIRS:
    os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

# ── 所有每日 dataset 名稱清單 (price 排第一，其餘為子資料) ──
_ALL_DATASETS = [
    "price", "chips", "twse_per", "taifex_inst",
    "margin", "sbl", "daytrading", "fini_holding", "credit_limit",
]
_SUB_DATASETS = _ALL_DATASETS[1:]  # price 以外的 8 個

# ── ETF 清單 ──────────────────────────────────────────
_CATEGORIES_PATH = os.path.join(BASE_DIR, "..", "stock_categories.json")

def _load_etf_set() -> set:
    try:
        with open(_CATEGORIES_PATH, "r", encoding="utf-8") as f:
            cats = json.load(f)
        return set(cats.get("ETF", {}).keys())
    except Exception:
        return set()

ETF_SET = _load_etf_set()

# ── 無財報股票快取 ────────────────────────────────────
_NO_DATA_PATH         = os.path.join(DATA_DIR, "no_finmind_data.json")
_NO_DATA_RECHECK_DAYS = 90

def _load_no_data_cache() -> dict:
    if os.path.exists(_NO_DATA_PATH):
        try:
            with open(_NO_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_no_data_cache(cache: dict):
    with open(_NO_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _is_confirmed_no_data(cache: dict, stock_id: str) -> bool:
    entry = cache.get(stock_id)
    if not isinstance(entry, dict) or entry.get("status") != "confirmed":
        return False
    confirmed_date = datetime.date.fromisoformat(entry["confirmed_date"])
    return (datetime.date.today() - confirmed_date).days < _NO_DATA_RECHECK_DAYS

def _record_no_data(cache: dict, stock_id: str) -> str:
    """回傳 'pending' 或 'confirmed'"""
    entry = cache.get(stock_id)
    today = datetime.date.today().isoformat()
    if isinstance(entry, dict) and entry.get("status") == "pending":
        cache[stock_id] = {"status": "confirmed", "confirmed_date": today}
        _save_no_data_cache(cache)
        return "confirmed"
    else:
        cache[stock_id] = {"status": "pending", "first_seen": today}
        _save_no_data_cache(cache)
        return "pending"

def _reset_no_data(cache: dict, stock_id: str):
    if stock_id in cache:
        del cache[stock_id]
        _save_no_data_cache(cache)

# ── 失敗日期計數 ──────────────────────────────────────
FAIL_LOG_PATH    = os.path.join(DATA_DIR, "failed_dates.json")
SKIP_AFTER_FAILS = 3

def _load_fail_log() -> dict:
    if os.path.exists(FAIL_LOG_PATH):
        try:
            with open(FAIL_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_fail_log(log: dict):
    with open(FAIL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ── Skip Dates 快取 (取代空白 CSV) ──────────────────
SKIP_DATES_PATH = os.path.join(DATA_DIR, "skip_dates.json")

def _load_skip_dates() -> dict:
    if os.path.exists(SKIP_DATES_PATH):
        try:
            with open(SKIP_DATES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_skip_dates(skip: dict):
    with open(SKIP_DATES_PATH, "w", encoding="utf-8") as f:
        json.dump(skip, f, ensure_ascii=False, indent=2)

def _mark_skip_date(skip: dict, dataset: str, date_str: str, reason: str = "no_data"):
    """標記單一 dataset 的日期，並立即寫檔。
    若需要批次標記多個 dataset，請使用 _mark_skip_dates_batch 以減少寫檔次數。"""
    skip[f"{dataset}_{date_str}"] = {
        "reason": reason,
        "confirmed_at": datetime.date.today().isoformat()
    }
    _save_skip_dates(skip)

def _mark_skip_dates_batch(skip: dict, datasets: list, date_str: str, reason: str = "no_data"):
    """批次標記多個 dataset 的同一天，只寫一次檔。"""
    today = datetime.date.today().isoformat()
    for ds in datasets:
        skip[f"{ds}_{date_str}"] = {"reason": reason, "confirmed_at": today}
    _save_skip_dates(skip)

def _is_skip_date(skip: dict, dataset: str, date_str: str) -> bool:
    return f"{dataset}_{date_str}" in skip

def _is_market_closed(skip: dict, date_str: str) -> bool:
    """price 被確認為休市日（market_closed）才回傳 True，其他 reason 不算。"""
    entry = skip.get(f"price_{date_str}", {})
    return entry.get("reason") == "market_closed"

# ── 共用 Headers & 工具 ──────────────────────────────
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

def _create_df_safely(rows: list, fields: list) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    num_cols = max(len(r) for r in rows)
    fields = list(fields)
    if len(fields) < num_cols:
        fields.extend([f"extra_{i}" for i in range(num_cols - len(fields))])
    elif len(fields) > num_cols:
        fields = fields[:num_cols]
    padded_rows = [r + [None] * (num_cols - len(r)) for r in rows]
    return pd.DataFrame(padded_rows, columns=fields)

def _polite_sleep(lo: float = 1.5, hi: float = 3.0):
    time.sleep(random.uniform(lo, hi))


# =====================================================================
# 模組 1: TWSE & TAIFEX 每日全市場資料 (免費無限制)
# =====================================================================

def _fetch_twse_json(url: str):
    """
    回傳:
      dict      = API 正常且有內容
      "NO_DATA" = API 正常但明確無資料 (含 404 網頁、很抱歉、data=[] 等)
      None      = 網路/解析錯誤或被 Ban
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        body = r.text.strip()

        # 證交所對某些過舊日期回傳 HTTP 200 但內容是 HTML (含 404)
        if not body or body[0] == "<":
            if "<title>404</title>" in body:
                return "NO_DATA"
            return None

        data = r.json()

        stat = data.get("stat", "")
        if "很抱歉" in stat or "沒有" in stat:
            return "NO_DATA"

        # 攔截 data 欄位為空 list 且沒有 tables 的情況
        if isinstance(data.get("data"), list) and not data["data"] and not data.get("tables"):
            return "NO_DATA"

        return data
    except KeyboardInterrupt:
        raise
    except Exception:
        return None


def crawl_daily_price(date_str: str, skip: dict) -> str:
    """
    三態回傳 (字串，語意比 bool 更清晰):
      "exists"       - 檔案已存在，不需要動作
      "market_closed"- API 確認當天休市，已寫入 skip
      "ok"           - 成功下載並存檔
      "error"        - 網路錯誤或解析失敗，應計入 fail_log
    """
    path = os.path.join(DATA_DIR, "raw_price", f"{date_str}_price.csv")
    if _already_exists(path):
        return "exists"
    if _is_skip_date(skip, "price", date_str):
        # 已在 skip 中；若是 market_closed 主迴圈會提前攔截，此處只做防呆
        return "exists"

    url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={date_str}&type=ALL"
    data = _fetch_twse_json(url)

    if data is None:
        return "error"

    if data == "NO_DATA":
        # 批次標記整天所有 dataset，只寫一次檔
        _mark_skip_dates_batch(skip, _ALL_DATASETS, date_str, reason="market_closed")
        return "market_closed"

    df = None
    if "tables" in data:
        for tbl in data["tables"]:
            if "fields" in tbl and "證券代號" in tbl.get("fields", []):
                df = _create_df_safely(tbl.get("data", []), tbl.get("fields", []))
                break
    elif "data9" in data:
        df = _create_df_safely(data["data9"], data.get("fields9", []))

    if df is None or df.empty:
        return "error"

    _save_csv(df, path)
    return "ok"


def crawl_daily_chips(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_chips.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "chips", date_str): return True

    url = f"https://www.twse.com.tw/fund/T86?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "chips", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        _mark_skip_date(skip, "chips", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_margin(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_margin", f"{date_str}_margin.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "margin", date_str): return True

    url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "margin", date_str, reason="no_data")
        return True

    # MI_MARGN 固定在 tables[1]；若結構異常先嘗試 tables[0]，再嘗試 data 欄位
    df = pd.DataFrame()
    tables = data.get("tables", [])
    if len(tables) >= 2 and tables[1].get("fields"):
        df = _create_df_safely(tables[1].get("data", []), tables[1].get("fields", []))
    elif len(tables) >= 1 and tables[0].get("fields"):
        df = _create_df_safely(tables[0].get("data", []), tables[0].get("fields", []))
    elif data.get("data"):
        df = _create_df_safely(data["data"], data.get("fields", []))

    if df.empty:
        # 格式完全不符才標記，避免誤殺有資料的日期
        _mark_skip_date(skip, "margin", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_sbl(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_margin", f"{date_str}_sbl.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "sbl", date_str): return True

    url = f"https://www.twse.com.tw/exchangeReport/TWT93U?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "sbl", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        _mark_skip_date(skip, "sbl", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_twse_per(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_twse_per", f"{date_str}_twse_per.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "twse_per", date_str): return True

    url = f"https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "twse_per", date_str, reason="no_data")
        return True

    df = pd.DataFrame()
    if data.get("data"):
        df = _create_df_safely(data["data"], data.get("fields", []))
    elif data.get("tables") and data["tables"][0].get("fields"):
        df = _create_df_safely(data["tables"][0].get("data", []), data["tables"][0].get("fields", []))

    if df.empty:
        _mark_skip_date(skip, "twse_per", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_daytrading(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_daytrading.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "daytrading", date_str): return True

    url = f"https://www.twse.com.tw/exchangeReport/TWTB4U?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "daytrading", date_str, reason="no_data")
        return True

    df = pd.DataFrame()
    if data.get("data"):
        df = _create_df_safely(data["data"], data.get("fields", []))
    elif data.get("tables"):
        for tbl in data["tables"]:
            if tbl.get("data"):
                df = _create_df_safely(tbl["data"], tbl.get("fields", []))
                break

    if df.empty:
        _mark_skip_date(skip, "daytrading", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_fini_holding(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_fini_holding.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "fini_holding", date_str): return True

    url = f"https://www.twse.com.tw/fund/MI_QFIIS?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "fini_holding", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        _mark_skip_date(skip, "fini_holding", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_credit_limit(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_margin", f"{date_str}_credit_limit.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "credit_limit", date_str): return True

    url = f"https://www.twse.com.tw/exchangeReport/TWT38U?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        _mark_skip_date(skip, "credit_limit", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        _mark_skip_date(skip, "credit_limit", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_taifex_inst(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_taifex", f"{date_str}_taifex_inst.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "taifex_inst", date_str): return True

    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
    query_date = dt.strftime("%Y/%m/%d")
    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    payload = {"queryStartDate": query_date, "queryEndDate": query_date, "commodityId": "TXF"}
    try:
        r = requests.post(url, data=payload, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return False
        try:
            text = r.content.decode("utf-8")
        except UnicodeDecodeError:
            text = r.content.decode("big5", errors="ignore")

        # 期交所只保留近 3 年，太舊的日期回傳 HTML 或「查無資料」
        if ("查無資料" in text or len(text.strip()) < 50
                or "<html" in text.lower() or "<!doctype" in text.lower()):
            _mark_skip_date(skip, "taifex_inst", date_str, reason="no_data")
            return True

        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            _mark_skip_date(skip, "taifex_inst", date_str, reason="empty_response")
            return True
        _save_csv(df, path)
        return True
    except Exception as e:
        print(f"    [期交所] {date_str} 抓取失敗: {e}")
        return False


def crawl_weekly_shareholding() -> bool:
    url = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, verify=False)
        if r.status_code != 200:
            return False
        try:
            text = r.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = r.content.decode("big5", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        df["證券代號"] = df["證券代號"].astype(str).str.strip()
        dates = df["資料日期"].unique()
        if len(dates) == 0:
            return False
        date_str = str(dates[0])
        path = os.path.join(DATA_DIR, "raw_shareholding", f"{date_str}_shareholding.csv")
        if not _already_exists(path):
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  [持股分級] {date_str} 下載成功 ({len(df)} 筆)")
        return True
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  [警告] 持股分級下載失敗: {e}")
        return False


def _is_holiday(date_obj: datetime.date) -> bool:
    if date_obj.weekday() >= 5:
        return True
    try:
        return _check_holiday(date_obj)
    except Exception:
        return False


# =====================================================================
# 模組 2: FinMind 個股財報 (需 Token, 具備自動休眠與三態回傳)
# =====================================================================

def _get_finmind_token():
    token_path = os.path.join(BASE_DIR, "..", "FINMIND_TOKEN.txt")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            t = f.read().strip()
            if t:
                return t
    return os.environ.get("FINMIND_TOKEN", "")

FINMIND_TOKEN = _get_finmind_token()
FM_BASE_URL = "https://api.finmindtrade.com/api/v4/data"


def _fm_get(dataset: str, start_date: str, end_date: str, data_id: str, max_retry: int = 5):
    params = {
        "dataset":    dataset,
        "start_date": start_date,
        "end_date":   end_date,
        "data_id":    data_id,
    }
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN

    for attempt in range(max_retry):
        try:
            r = requests.get(FM_BASE_URL, params=params, timeout=30)

            if r.status_code in (429, 402):
                resume_time = datetime.datetime.now() + datetime.timedelta(seconds=3600)
                print(f"\n    [FinMind] 觸發限速 (狀態碼 {r.status_code})，每小時額度已用盡。")
                print(f"    [FinMind] 預計於 {resume_time.strftime('%H:%M:%S')} 額度重置，自動繼續...")
                time.sleep(3600)
                continue

            if r.status_code == 200:
                js = r.json()
                if js.get("status") == 200:
                    data = js.get("data", [])
                    return pd.DataFrame(data) if data else pd.DataFrame()

            print(f"    [FinMind] 異常狀態碼 {r.status_code}，稍後重試...")
            time.sleep(5)

        except KeyboardInterrupt:
            print("\n[系統] 收到中斷指令，強制結束爬蟲。")
            raise
        except Exception as e:
            print(f"    [FinMind] 網路異常: {e}")
            time.sleep(5)

    return None


def _crawl_fm_dataset(dataset_name: str, stock_id: str, start_date: str, end_date: str, output_path: str):
    """
    三態回傳:
      True  = 成功（有資料 或 歷史資料已是最新）
      False = API/網路錯誤
      None  = 200 OK 但完全無資料（從未有過任何紀錄）

    12 小時快取: 檔案已存在且 12 小時內已查過 → 直接回傳 True，不打 API。
    增量更新: 從既有資料的最後一筆日期 +1 天開始抓，避免重複下載。
    注意: CSV 存的是 FinMind 原始日期（未做 look-ahead 推移），
          推移只在 feature_engineering.py 讀取時才做，
          確保增量更新的日期邊界與 API 查詢參數一致。
    """
    DATE_COL_CANDIDATES = ["date", "revenue_date", "calendarDate", "period"]

    try:
        if _already_exists(output_path):
            # 12 小時內已確認過，跳過
            if time.time() - os.path.getmtime(output_path) < 43200:
                return True

            existing = pd.read_csv(output_path, encoding="utf-8-sig")
            date_col = next((c for c in DATE_COL_CANDIDATES if c in existing.columns), None)
            if date_col and not existing.empty:
                last_date = pd.to_datetime(existing[date_col]).max()
                start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                if start_date > end_date:
                    os.utime(output_path, (time.time(), time.time()))
                    return True

        df = _fm_get(dataset_name, start_date, end_date, stock_id)

        if df is None:
            return False

        if df.empty:
            if _already_exists(output_path):
                os.utime(output_path, (time.time(), time.time()))
                return True
            return None

        if os.path.exists(output_path):
            old_df = pd.read_csv(output_path, encoding="utf-8-sig")
            df = pd.concat([old_df, df], ignore_index=True)

            dedup_cols = []
            if "stock_id" in df.columns:
                dedup_cols.append("stock_id")
            curr_date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
            if curr_date_col:
                dedup_cols.append(curr_date_col)
            if "type" in df.columns:
                dedup_cols.append("type")
            if dedup_cols:
                df = df.drop_duplicates(subset=dedup_cols)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return True

    except Exception as e:
        print(f"\n    [錯誤] {dataset_name} / {stock_id} 發生例外: {e}")
        return False


# =====================================================================
# 主下載控制
# =====================================================================

def download_history_data(
    start_date_obj: datetime.date,
    end_date_obj: datetime.date,
    target_stocks: list = None,
):
    delta = datetime.timedelta(days=1)

    print("\n[啟動] 官方全市場資料爬蟲 (免 Token 吃到飽)")
    crawl_weekly_shareholding()

    fail_log        = _load_fail_log()
    skip_dates      = _load_skip_dates()
    downloaded_days = 0
    skipped_days    = 0
    holiday_days    = 0
    auto_skip_days  = 0

    curr = start_date_obj
    while curr <= end_date_obj:
        d_str = curr.strftime("%Y%m%d")

        # ── 永久略過：累計失敗超過上限的日期 ──────────────
        if fail_log.get(d_str, 0) >= SKIP_AFTER_FAILS:
            auto_skip_days += 1
            curr += delta
            continue

        # ── 週末 / 已知假日：taiwan_holidays 套件攔截 ─────
        if _is_holiday(curr):
            holiday_days += 1
            curr += delta
            continue

        # ── price API 已確認當天休市 (颱風假/補假等) ───────
        # crawl_daily_price 回傳 "market_closed" 時會批次寫入所有 dataset，
        # 因此這裡判斷後可以直接跳過整天，不需要再打任何子資料 API。
        if _is_market_closed(skip_dates, d_str):
            skipped_days += 1
            curr += delta
            continue

        # ── 9 個 dataset 全部處理完畢才跳過 ───────────────
        files_required = [
            ("price",        os.path.join(DATA_DIR, "raw_price",    f"{d_str}_price.csv")),
            ("chips",        os.path.join(DATA_DIR, "raw_chips",    f"{d_str}_chips.csv")),
            ("twse_per",     os.path.join(DATA_DIR, "raw_twse_per", f"{d_str}_twse_per.csv")),
            ("taifex_inst",  os.path.join(DATA_DIR, "raw_taifex",   f"{d_str}_taifex_inst.csv")),
            ("margin",       os.path.join(DATA_DIR, "raw_margin",   f"{d_str}_margin.csv")),
            ("sbl",          os.path.join(DATA_DIR, "raw_margin",   f"{d_str}_sbl.csv")),
            ("daytrading",   os.path.join(DATA_DIR, "raw_chips",    f"{d_str}_daytrading.csv")),
            ("fini_holding", os.path.join(DATA_DIR, "raw_chips",    f"{d_str}_fini_holding.csv")),
            ("credit_limit", os.path.join(DATA_DIR, "raw_margin",   f"{d_str}_credit_limit.csv")),
        ]

        def _done(ds, p):
            return _already_exists(p) or _is_skip_date(skip_dates, ds, d_str)

        if all(_done(ds, p) for ds, p in files_required):
            skipped_days += 1
            curr += delta
            continue

        # ── price 已完成（有檔案），補抓缺失的子資料 ────────
        price_ds, price_path = files_required[0]
        if _done(price_ds, price_path):
            missing_names = [
                p.split(os.sep)[-1]
                for ds, p in files_required[1:]
                if not _done(ds, p)
            ]
            if missing_names:
                print(f"  [補抓] {d_str} 補齊缺失: {missing_names}")
                results = [
                    crawl_daily_chips(d_str, skip_dates),
                    crawl_daily_margin(d_str, skip_dates),
                    crawl_daily_sbl(d_str, skip_dates),
                    crawl_daily_twse_per(d_str, skip_dates),
                    crawl_daily_taifex_inst(d_str, skip_dates),
                    crawl_daily_daytrading(d_str, skip_dates),
                    crawl_daily_fini_holding(d_str, skip_dates),
                    crawl_daily_credit_limit(d_str, skip_dates),
                ]
                if all(results):
                    downloaded_days += 1
                else:
                    print(f"    [警告] {d_str} 部分補抓失敗，下次重試")
                _polite_sleep()
            curr += delta
            continue

        # ── 全新日期：先抓 price，再抓子資料 ────────────────
        print(f"  [下載] 官方日報 {d_str}...", end=" ", flush=True)
        price_result = crawl_daily_price(d_str, skip_dates)

        if price_result == "market_closed":
            # 批次 skip 已在 crawl_daily_price 內完成，直接跳過
            print("休市")
            skipped_days += 1
            curr += delta
            continue

        if price_result == "error":
            fail_log[d_str] = fail_log.get(d_str, 0) + 1
            _save_fail_log(fail_log)
            count     = fail_log[d_str]
            remaining = SKIP_AFTER_FAILS - count
            if remaining > 0:
                print(f"\n    [警告] {d_str} 股價抓取失敗 (第 {count} 次，再失敗 {remaining} 次後永久略過)")
            else:
                print(f"\n    [略過] {d_str} 累計失敗 {count} 次，列入永久略過名單")
            curr += delta
            continue

        # price_result in ("ok", "exists") → 開市，繼續抓子資料
        results = [
            crawl_daily_chips(d_str, skip_dates),
            crawl_daily_margin(d_str, skip_dates),
            crawl_daily_sbl(d_str, skip_dates),
            crawl_daily_twse_per(d_str, skip_dates),
            crawl_daily_taifex_inst(d_str, skip_dates),
            crawl_daily_daytrading(d_str, skip_dates),
            crawl_daily_fini_holding(d_str, skip_dates),
            crawl_daily_credit_limit(d_str, skip_dates),
        ]
        if all(results):
            downloaded_days += 1
            print("OK")
        else:
            print("部分失敗 (下次補抓)")

        if d_str in fail_log:
            del fail_log[d_str]
            _save_fail_log(fail_log)

        _polite_sleep()
        curr += delta

    print(
        f"\n  官方爬蟲完成。"
        f" (新下載: {downloaded_days} 天"
        f" | 快取跳過: {skipped_days} 天"
        f" | 假日: {holiday_days} 天"
        f" | 永久略過: {auto_skip_days} 天)"
    )

    if not target_stocks:
        return

    # ── FinMind 基本面爬蟲 ────────────────────────────────
    start_str = start_date_obj.strftime("%Y-%m-%d")
    end_str   = end_date_obj.strftime("%Y-%m-%d")

    print("\n[啟動] FinMind 基本面資料爬蟲")
    if not FINMIND_TOKEN:
        print("  [注意] 尚未設定 FINMIND_TOKEN.txt，將使用免費額度。")

    no_data_cache = _load_no_data_cache()

    total        = len(target_stocks)
    skipped_etf  = 0
    skipped_cache = 0
    updated      = 0
    partial_miss = 0
    errors_total = 0

    for idx, stock_id in enumerate(target_stocks, 1):
        print(f"  FinMind 進度: {idx}/{total} ({stock_id})          ", end="\r", flush=True)

        if stock_id in ETF_SET:
            skipped_etf += 1
            continue

        if _is_confirmed_no_data(no_data_cache, stock_id):
            skipped_cache += 1
            continue

        results = []
        datasets = [
            ("營收",   "TaiwanStockMonthRevenue",       f"{stock_id}_monthly_revenue.csv"),
            ("損益表", "TaiwanStockFinancialStatements", f"{stock_id}_financial_stmt.csv"),
            ("資產表", "TaiwanStockBalanceSheet",        f"{stock_id}_balance_sheet.csv"),
            ("現金流", "TaiwanStockCashFlowsStatement",  f"{stock_id}_cashflow.csv"),
            ("股利",   "TaiwanStockDividend",            f"{stock_id}_dividend.csv"),
        ]
        for label, dataset, fname in datasets:
            path = os.path.join(DATA_DIR, "raw_financial", fname)
            results.append((label, _crawl_fm_dataset(dataset, stock_id, start_str, end_str, path)))
            _polite_sleep(1, 2)

        errors   = [name for name, ok in results if ok is False]
        no_data  = [name for name, ok in results if ok is None]

        if errors:
            if stock_id in no_data_cache and no_data_cache[stock_id].get("status") == "pending":
                _reset_no_data(no_data_cache, stock_id)
            errors_total += 1
            print(f"  [錯誤] FinMind {stock_id} 抓取失敗: {errors}                    ")
        elif no_data:
            if len(no_data) == len(results):
                status = _record_no_data(no_data_cache, stock_id)
                if status == "pending":
                    print(f"  [首次無資料] FinMind {stock_id} → 待下次二次確認後快取      ")
                else:
                    print(f"  [確認無財報] FinMind {stock_id} → 已快取，{_NO_DATA_RECHECK_DAYS}天後重新確認")
                skipped_cache += 1
            else:
                partial_miss += 1
        else:
            _reset_no_data(no_data_cache, stock_id)
            any_new = any(
                os.path.exists(os.path.join(DATA_DIR, "raw_financial", fname))
                and time.time() - os.path.getmtime(
                    os.path.join(DATA_DIR, "raw_financial", fname)) < 60
                for _, _, fname in datasets
            )
            if any_new:
                updated += 1
                print(f"  [更新] FinMind {stock_id} 新資料已寫入                         ")

    print(
        f"  FinMind 爬蟲完成。"
        f"(略過ETF: {skipped_etf}"
        f" | 略過快取: {skipped_cache}"
        f" | 新更新: {updated}"
        f" | 部分缺項: {partial_miss}"
        f" | 錯誤: {errors_total})"
    )