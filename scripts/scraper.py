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
    "market_closed"     - price API 確認當天休市 (颱風假/補假等)，整天所有 dataset 同步標記
    "no_data"           - 該 dataset 明確回傳無資料 (例如當沖制度實施前)
    "unexpected_format" - API 有回應但格式不符預期，可手動清除後重試
    "empty_response"    - API 有回應但資料為空
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
    "raw_financial",
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


# ── FinMind 基本面快取天數設定 ──────────────────────────
# 設定基本面資料更新天數間隔。預設為 7 (7天更新一次)。
# 例如填入 3 代表 3 天更新一次，填入 30 代表 30 天更新一次。
_FINMIND_CACHE_DAYS = 15
_FINMIND_CACHE_SECONDS = _FINMIND_CACHE_DAYS * 86400



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


# ── 局部缺失資料快取 (個別報表) ─────────────────────
MISSING_FM_PATH = os.path.join(DATA_DIR, "missing_fm_datasets.json")


def _load_missing_fm() -> dict:
    if os.path.exists(MISSING_FM_PATH):
        try:
            with open(MISSING_FM_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_missing_fm(cache: dict):
    with open(MISSING_FM_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


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
    """標記單一 dataset 的日期，並立即寫檔。"""
    skip[f"{dataset}_{date_str}"] = {
        "reason": reason,
        "confirmed_at": datetime.date.today().isoformat(),
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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
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
    if not rows:
        return pd.DataFrame()
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
    while True:
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
            
            # 攔截維護時間 (1:30 PM - 1:45 PM) 或其他伺服器維護 智慧等待
            if any(k in stat for k in ["暫停查詢", "結算時間", "維護"]):
                print(f"\n    ⏳ [系統提示] 證交所伺服器結算或維護中 (原因: {stat})，暫停 30 分鐘後自動重試...", end="", flush=True)
                time.sleep(1800)
                continue

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
    三態回傳 (字串):
      "exists"        - 檔案已存在，不需要動作
      "market_closed" - API 確認當天休市，已寫入 skip
      "ok"            - 成功下載並存檔
      "skip"          - 有 skip 記錄但 reason 不是 market_closed
      "error"         - 網路錯誤或解析失敗，應計入 fail_log
    """
    path = os.path.join(DATA_DIR, "raw_price", f"{date_str}_price.csv")
    if _already_exists(path):
        return "exists"

    if _is_skip_date(skip, "price", date_str):
        entry = skip.get(f"price_{date_str}", {})
        if entry.get("reason") == "market_closed":
            return "market_closed"
        return "skip"

    url = (
        f"https://www.twse.com.tw/exchangeReport/MI_INDEX"
        f"?response=json&date={date_str}&type=ALL"
    )
    data = _fetch_twse_json(url)

    if data is None:
        return "error"

    if data == "NO_DATA":
        today_str = datetime.date.today().strftime("%Y%m%d")
        if date_str < today_str:
            _mark_skip_dates_batch(skip, _ALL_DATASETS, date_str, reason="market_closed")
            return "market_closed"
        else:
            print(f"  [提示] 今日 ({date_str}) 資料在證交所尚未上架或正在更新，暫不標記為休市，待稍後重試。")
            return "error"

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

    url = (
        f"https://www.twse.com.tw/exchangeReport/MI_MARGN"
        f"?response=json&date={date_str}&selectType=ALL"
    )
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
        _mark_skip_date(skip, "margin", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_sbl(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_margin", f"{date_str}_sbl.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "sbl", date_str): return True

    url = (
        f"https://www.twse.com.tw/exchangeReport/TWT93U"
        f"?response=json&date={date_str}&selectType=ALL"
    )
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

    url = (
        f"https://www.twse.com.tw/exchangeReport/BWIBBU_d"
        f"?response=json&date={date_str}&selectType=ALL"
    )
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
        df = _create_df_safely(
            data["tables"][0].get("data", []),
            data["tables"][0].get("fields", []),
        )

    if df.empty:
        _mark_skip_date(skip, "twse_per", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_daytrading(date_str: str, skip: dict) -> bool:
    path = os.path.join(DATA_DIR, "raw_chips", f"{date_str}_daytrading.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "daytrading", date_str): return True

    url = (
        f"https://www.twse.com.tw/exchangeReport/TWTB4U"
        f"?response=json&date={date_str}&selectType=ALL"
    )
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

    url = (
        f"https://www.twse.com.tw/fund/MI_QFIIS"
        f"?response=json&date={date_str}&selectType=ALL"
    )
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

    url = (
        f"https://www.twse.com.tw/exchangeReport/TWT38U"
        f"?response=json&date={date_str}&selectType=ALL"
    )
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

    # 支援 20260531 / 2026-05-31 / 2026/05/31 三種格式輸入
    try:
        clean_date = date_str.replace("-", "").replace("/", "")
        query_date = datetime.datetime.strptime(clean_date, "%Y%m%d").strftime("%Y/%m/%d")
    except Exception:
        query_date = date_str

    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    payload = {
        "queryStartDate": query_date,
        "queryEndDate":   query_date,
        "commodityId":    "TXF",
    }
    try:
        r = requests.post(url, data=payload, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return False
        try:
            text = r.content.decode("utf-8")
        except UnicodeDecodeError:
            text = r.content.decode("big5", errors="ignore")

        # 期交所只保留近 3 年，太舊的日期回傳 HTML 或「查無資料」
        if (
            "查無資料" in text
            or len(text.strip()) < 50
            or "<html" in text.lower()
            or "<!doctype" in text.lower()
        ):
            _mark_skip_date(skip, "taifex_inst", date_str, reason="no_data")
            return True

        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            _mark_skip_date(skip, "taifex_inst", date_str, reason="empty_response")
            return True
        _save_csv(df, path)
        return True
    except Exception as e:
        print(f"    [期交所] {date_str} 抓取失敗: {type(e).__name__} - {e}")
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
        print(f"  [警告] 持股分級下載失敗: {type(e).__name__} - {e}")
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

def _get_finmind_token() -> str:
    token_path = os.path.join(BASE_DIR, "..", "FINMIND_TOKEN.txt")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            t = f.read().strip()
            if t:
                return t
    return os.environ.get("FINMIND_TOKEN", "")


FINMIND_TOKEN = _get_finmind_token()
FM_BASE_URL   = "https://api.finmindtrade.com/api/v4/data"


def _fm_get(
    dataset:   str,
    start_date: str,
    end_date:   str,
    data_id:   str,
    max_retry: int = 5,
):
    params = {
        "dataset":    dataset,
        "start_date": start_date,
        "end_date":   end_date,
        "data_id":    data_id,
    }
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN

    attempt = 0
    while attempt < max_retry:
        try:
            r = requests.get(FM_BASE_URL, params=params, timeout=30)

            if r.status_code in (429, 402):
                # 免費帳號每小時 600 次；429/402 = 額度耗盡，等待 1 小時重置
                resume_time = datetime.datetime.now() + datetime.timedelta(seconds=3600)
                print(f"\n    [FinMind] 觸發限速 (狀態碼 {r.status_code})，每小時額度已用盡。")
                print(f"    [FinMind] 預計於 {resume_time.strftime('%H:%M:%S')} 額度重置，自動繼續...")
                time.sleep(3600)
                # 限速不扣重試次數，繼續等待
                continue

            if r.status_code == 200:
                js = r.json()
                if js.get("status") == 200:
                    data = js.get("data", [])
                    return pd.DataFrame(data) if data else pd.DataFrame()

            print(f"    [FinMind] 異常狀態碼 {r.status_code}，稍後重試...")
            time.sleep(5)
            attempt += 1

        except KeyboardInterrupt:
            print("\n[系統] 收到中斷指令，強制結束爬蟲。")
            raise
        except Exception as e:
            print(f"    [FinMind] 網路異常: {type(e).__name__} - {e}")
            time.sleep(5)
            attempt += 1

    return None


def _crawl_fm_dataset(
    dataset_name: str,
    stock_id:     str,
    start_date:   str,
    end_date:     str,
    output_path:  str,
    missing_cache: dict,
):
    """
    三態回傳:
      True       = 成功（有資料 或 歷史資料已是最新）
      False      = API/網路錯誤
      "skipped"  = 12 小時快取 或 missing_fm_cache 快取，跳過
      None       = 200 OK 但完全無資料（從未有過任何紀錄）

    注意: CSV 存的是 FinMind 原始日期（未做 look-ahead 推移），
          推移只在 feature_engineering.py 讀取時才做，
          確保增量更新的日期邊界與 API 查詢參數一致。
    """
    DATE_COL_CANDIDATES = ["date", "revenue_date", "calendarDate", "period"]

    # ── missing_fm_cache 個別快取：90 天內跳過 ─────────────
    cache_key = f"{stock_id}_{dataset_name}"
    if cache_key in missing_cache:
        try:
            last_check = datetime.date.fromisoformat(missing_cache[cache_key])
            if (datetime.date.today() - last_check).days < _NO_DATA_RECHECK_DAYS:
                return "skipped"
        except Exception:
            pass

    try:
        if _already_exists(output_path):
            # ── 增量更新快取：檔案仍在指定快取天數內，不打 API ─────────────
            if time.time() - os.path.getmtime(output_path) < _FINMIND_CACHE_SECONDS:
                return "skipped"

            # ── 增量更新：從上次最後一筆 +1 天開始 ──────────
            existing = pd.read_csv(output_path, encoding="utf-8-sig")
            date_col = next((c for c in DATE_COL_CANDIDATES if c in existing.columns), None)
            if date_col and not existing.empty:
                last_date  = pd.to_datetime(existing[date_col]).max()
                start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                if start_date > end_date:
                    # 已是最新，更新時間戳避免下次重複讀 CSV
                    os.utime(output_path, (time.time(), time.time()))
                    return "skipped"

        df = _fm_get(dataset_name, start_date, end_date, stock_id)

        if df is None:
            return False  # 伺服器異常或網路斷線

        if df.empty:
            if _already_exists(output_path):
                # 有歷史資料，只是目前沒有新資料
                os.utime(output_path, (time.time(), time.time()))
                return "skipped"
            return None  # 從未有任何資料

        # ── 合併舊資料並去重 ─────────────────────────────
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
        print(f"\n    [錯誤] {dataset_name} / {stock_id} 發生例外: {type(e).__name__} - {e}")
        return False


# =====================================================================
# 主下載控制
# =====================================================================

def download_history_data(
    start_date_obj: datetime.date,
    end_date_obj:   datetime.date,
    target_stocks:  list = None,
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

        # ── price 已完成（有檔案或已 skip），補抓缺失子資料 ─
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
            # 批次 skip 已在 crawl_daily_price 內完成
            if d_str in fail_log:
                del fail_log[d_str]
                _save_fail_log(fail_log)
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
                print(
                    f"\n    [警告] {d_str} 股價抓取失敗 "
                    f"(第 {count} 次，再失敗 {remaining} 次後永久略過)"
                )
            else:
                print(f"\n    [略過] {d_str} 累計失敗 {count} 次，列入永久略過名單")
            _polite_sleep()
            curr += delta
            continue

        if price_result == "skip":
            # 有 skip 記錄但 reason 不是 market_closed（例如 empty_response）
            skipped_days += 1  # ← 修正：納入統計，避免天數對不上
            _polite_sleep()
            curr += delta
            continue

        # 如果之前有失敗紀錄，但這次成功了，就把它從失敗名單移除
        if d_str in fail_log:
            del fail_log[d_str]
            _save_fail_log(fail_log)

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

    no_data_cache    = _load_no_data_cache()
    missing_fm_cache = _load_missing_fm()   # ← 迴圈外只讀一次，減少磁碟 I/O

    total         = len(target_stocks)
    skipped_etf   = 0
    skipped_cache = 0
    updated       = 0
    partial_miss  = 0
    errors_total  = 0

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
            res  = _crawl_fm_dataset(dataset, stock_id, start_str, end_str, path, missing_fm_cache)
            results.append((label, res))
            if res != "skipped":
                _polite_sleep(1, 2)

        errors  = [name for name, ok in results if ok is False]
        no_data = [name for name, ok in results if ok is None]

        if errors:
            if stock_id in no_data_cache and no_data_cache[stock_id].get("status") == "pending":
                _reset_no_data(no_data_cache, stock_id)
            errors_total += 1
            print(f"  [錯誤] FinMind {stock_id} 抓取失敗: {errors}                    ")

        elif no_data:
            if len(no_data) == len(results):
                # 5 張表全無 → 兩階段確認後快取
                status = _record_no_data(no_data_cache, stock_id)
                if status == "pending":
                    print(f"  [首次無資料] FinMind {stock_id} → 待下次二次確認後快取      ")
                else:
                    print(
                        f"  [確認無財報] FinMind {stock_id} → "
                        f"已快取，{_NO_DATA_RECHECK_DAYS} 天後重新確認"
                    )
                skipped_cache += 1
            else:
                # 局部缺失 → 寫入 missing_fm_cache，90 天內跳過該 dataset
                # 注：若 FinMind 之後補上資料，需等 90 天後才會重抓
                partial_miss += 1
                for label in no_data:
                    ds_api_name = next(ds for l, ds, _ in datasets if l == label)
                    missing_fm_cache[f"{stock_id}_{ds_api_name}"] = (
                        datetime.date.today().isoformat()
                    )
                _save_missing_fm(missing_fm_cache)
                print(f"  [局部缺漏] FinMind {stock_id} 缺 {no_data} → 已加入快取       ")

        else:
            _reset_no_data(no_data_cache, stock_id)
            any_new = any(ok is True for _, ok in results)
            if any_new:
                updated += 1
                print(f"  [更新] FinMind {stock_id} 新資料已寫入                         ")

    print(
        f"  FinMind 爬蟲完成。"
        f" (略過ETF: {skipped_etf}"
        f" | 略過快取: {skipped_cache}"
        f" | 新更新: {updated}"
        f" | 部分缺項: {partial_miss}"
        f" | 錯誤: {errors_total})"
    )