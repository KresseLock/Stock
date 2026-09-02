"""
scraper.py — 台灣股市多源資料爬蟲 (整合 TWSE + TAIFEX + FinMind)
====================================================
資料來源分流 (T+1/T+2 量化藍圖):
  1. 台灣證券交易所 (TWSE) [免費每日]：價量、籌碼、資券、借券、本益比、當沖、外資持股
  2. 台灣期貨交易所 (TAIFEX) [免費每日]：外資台指期未平倉 (大盤多空指標)
  3. 集保中心 (TDCC) [免費每週]：大戶持股分級
  4. FinMind [需 Token 每月/季]：月營收、綜合損益表、資產負債表、現金流量表、股利

儲存路徑:
  data/raw_price/        - 股價行情 (TWSE)
  data/raw_chips/        - 法人買賣超、當沖、外資持股 (TWSE)
  data/raw_margin/       - 融資券、借券 (TWSE)
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

import csv
import datetime
import io
import json
import os
import random
import sys
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


class FinMindLimitExceeded(Exception):
    """Raised when FinMind API limit (429/402) is reached."""
    pass


class _SingleInstanceLock:
    """跨平台單一實例鎖：防止兩個 scraper 同時跑而互相覆寫 failed_dates.json / skip_dates.json。
    以對 lock 檔的作業系統建議鎖實作 (Windows: msvcrt / POSIX: fcntl)，行程結束自動釋放，
    不會像「鎖檔存在即視為佔用」那樣在當機後留下死鎖。"""

    def __init__(self, lock_path):
        self.lock_path = lock_path
        self._fh = None

    def __enter__(self):
        self._fh = open(self.lock_path, "w")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"偵測到另一個 scraper 實例正在佔用 "
                f"{os.path.basename(self.lock_path)}。"
                "請等待其結束，避免兩個實例互相覆寫同一份狀態檔。"
            )
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._fh.close()
            self._fh = None

try:
    from taiwan_holidays.taiwan_calendar import TaiwanCalendar
    _th = TaiwanCalendar()
    _check_holiday = _th.is_holiday
except Exception:
    _check_holiday = lambda d: False


# ── 建立所有資料夾 ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
# ★ 資料路徑的唯一來源是 config.py 的 § 0，這裡只負責把它取進來。
#   任何新增的路徑都必須由下面這幾個常數組出來，不要在函式裡另外拼
#   os.path.join(..., "data", ...) 或寫死資料夾名稱——一旦出現第二種拼法
#   （尤其大小寫不同的 "Data"），在 Windows 上因為檔名大小寫不敏感會恰好
#   指向同一處而看不出問題，搬到 Linux/macOS 就會分裂成兩個資料目錄，
#   症狀是「明明抓過卻又重抓一遍」，而且沒有任何錯誤訊息。
#
# ★ fallback 的用途：本檔要能在沒有 config.py 的環境單獨執行（例如只把
#   scripts/ 複製到另一台機器補資料）。因此下面重複了一份等價定義——
#   **這是刻意的重複，修改 config.py § 0 的目錄名稱時務必同步這裡**，
#   兩邊不一致正是上面那個「分裂成兩個目錄」的成因。
try:
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    from config import (
        DATA_DIR,
        RAW_PRICE_DIR, RAW_CHIPS_DIR, RAW_MARGIN_DIR, RAW_TWSE_PER_DIR,
        RAW_TAIFEX_DIR, RAW_SHAREHOLDING_DIR, RAW_FINANCIAL_DIR,
    )
except ImportError:
    # normpath 去掉 ".."：os.path.join(PARENT_DIR, "data") 已無 ".."，
    # 但仍保留 normpath 以確保與 config.py 算出的字串完全一致。
    DATA_DIR = os.path.normpath(os.path.join(PARENT_DIR, "data"))
    RAW_PRICE_DIR        = os.path.join(DATA_DIR, "raw_price")
    RAW_CHIPS_DIR        = os.path.join(DATA_DIR, "raw_chips")
    RAW_MARGIN_DIR       = os.path.join(DATA_DIR, "raw_margin")
    RAW_TWSE_PER_DIR     = os.path.join(DATA_DIR, "raw_twse_per")
    RAW_TAIFEX_DIR       = os.path.join(DATA_DIR, "raw_taifex")
    RAW_SHAREHOLDING_DIR = os.path.join(DATA_DIR, "raw_shareholding")
    RAW_FINANCIAL_DIR    = os.path.join(DATA_DIR, "raw_financial")

# 逐日資料所在的資料夾（check_data_integrity 掃描、幽靈日清理都用這一份）
TWSE_DAILY_DIRS = [
    RAW_PRICE_DIR, RAW_CHIPS_DIR, RAW_MARGIN_DIR,
    RAW_TWSE_PER_DIR, RAW_TAIFEX_DIR,
]

# ── 分來源鎖檔 ────────────────────────────────────────
#
# 為什麼不是一把鎖：TWSE 段（含 TDCC）與 FinMind 段寫入的狀態檔完全不重疊——
#   TWSE 段    → failed_dates.json / skip_dates.json
#   FinMind 段 → no_finmind_data.json / missing_fm_datasets.json
# 兩段之間沒有共用可變狀態，因此可以安全地在兩個終端機並行執行。
# 但同一段仍然只能有一個實例，否則會回到「互相覆寫狀態檔」的老問題。
#
# 規則：--source twse 取 twse 鎖、--source finmind 取 finmind 鎖、
#       --source all（預設）兩把都取。因此 all 與任一單獨模式仍會互斥，
#       不會出現「all 正在跑，另一個終端機又開一個 twse」的重複抓取。
LOCK_FILES = {
    "twse":    os.path.join(DATA_DIR, "scraper_twse.lock"),
    "finmind": os.path.join(DATA_DIR, "scraper_finmind.lock"),
}


def _locks_for(source: str) -> list:
    """回傳指定來源需要取得的鎖檔路徑清單。"""
    if source == "all":
        return [LOCK_FILES["twse"], LOCK_FILES["finmind"]]
    return [LOCK_FILES[source]]


DIRS = TWSE_DAILY_DIRS + [RAW_SHAREHOLDING_DIR, RAW_FINANCIAL_DIR]
for _folder in DIRS:
    os.makedirs(_folder, exist_ok=True)

# ── 所有每日 dataset 名稱清單 (price 排第一，其餘為子資料) ──
_ALL_DATASETS = [
    "price", "chips", "twse_per", "taifex_inst",
    "margin", "sbl", "daytrading", "fini_holding",
]
_SUB_DATASETS = _ALL_DATASETS[1:]  # price 以外的 7 個

# ── ETF 清單 ──────────────────────────────────────────
_CATEGORIES_PATH = os.path.join(BASE_DIR, "stock_categories.json")


def _load_etf_set() -> set:
    try:
        with open(_CATEGORIES_PATH, "r", encoding="utf-8") as f:
            cats = json.load(f)
        return set(cats.get("ETF", {}).keys())
    except Exception:
        return set()


# ── 抓取母體：由 data/raw_price 推導 (FINMIND_FETCH_MODE="listed") ──
#
# 為什麼需要第三種母體來源：
#   limited/all 兩種模式都以 stock_categories.json 的產業對照為準，而該檔是 FinMind
#   的全市場清單（含大量上櫃股與特別股）。實測 limited 模式的 1,258 檔目標中，有 723 檔
#   在 TWSE 沒有價量資料或沒有獨立財報（上櫃股、3036A 這類特別股），生產流水線根本用不到，
#   額度純浪費；同時漏掉 591 檔有價量的上市普通股（台泥 1101、亞泥 1102、統一 1216 等
#   TRAIN_INDUSTRIES=False 的產業），它們的財報會停在最後一次抓取的版本。
#
# listed 模式改以「本系統自己抓下來的 data/raw_price」推導母體：凡是在 TWSE 掛牌交易過的
# 上市普通股都在裡面，不依賴任何外部檔案或另一套系統的產出。
#
# ★ 掃全部歷史而不是只掃近期，是為了含入已下市標的（實測 51 檔）。
#   只掃近期會讓母體只剩「活到今天」的公司，用這種母體補來的財報做回測就是倖存者偏誤，
#   而且不會有任何錯誤訊息——數量看起來很正常，正是它危險的地方。
_UNIVERSE_MIN_EXPECTED = 500   # 低於此數視為清單解析出錯，大聲警告（實測約 1,142 檔）
_UNIVERSE_CACHE_PATH = os.path.join(DATA_DIR, "universe_cache.json")


def _is_common_stock(sid: str) -> bool:
    """上市普通股：4 位純數字且非 00 開頭。

    順帶排除 00 開頭的 ETF/受益證券，以及 3036A/3702A/8112A 這類特別股 ——
    特別股在 FinMind 沒有獨立財報，抓它們只會拿到空回應並吃掉額度。"""
    return len(sid) == 4 and sid.isdigit() and not sid.startswith("00")


def _scan_price_files(price_dir: str, fnames: list) -> set:
    """從指定的價格檔中取出所有上市普通股代號。"""
    out = set()
    for fname in fnames:
        # 用 csv 而非 pandas：這裡只需要代號那一欄，pandas 卻會把整張表
        # (約 1,100 列 × 16 欄) 建成 DataFrame 再整個丟掉。單檔差距看不出來，
        # 乘以 1,600 個檔就是啟動時多等的那分鐘。
        try:
            with open(os.path.join(price_dir, fname), "r",
                      encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                idx = next((i for i, c in enumerate(header)
                            if "證券代號" in str(c) or "股票代號" in str(c)), None)
                if idx is None:
                    continue
                for row in reader:
                    # 欄數不足的壞列直接跳過，等同 pandas 的 on_bad_lines="skip"
                    if len(row) > idx:
                        sid = row[idx].strip()
                        if _is_common_stock(sid):
                            out.add(sid)
        except Exception:
            continue
    return out


def _universe_from_raw_price() -> set:
    """
    掃 data/raw_price 全歷史，推導上市普通股母體（含已下市）。

    ★ 增量快取：母體是「每日代號的聯集」，只增不減，所以新增的價格檔只可能加入新代號，
      不可能讓舊代號消失。因此快取存「已掃到哪一個檔」+ 集合本身，下次只掃該檔之後的新檔。
      全掃 1,600 個檔約 50 秒，只發生一次；日常增量掃 1~2 個檔在毫秒級。
      若快取記錄的最後一檔已不存在（例如被 -c 清掉幽靈日），就整個重掃——往安全方向倒。
    """
    price_dir = RAW_PRICE_DIR
    if not os.path.isdir(price_dir):
        return set()
    files = sorted(f for f in os.listdir(price_dir) if f.endswith("_price.csv"))
    if not files:
        return set()

    cached_stocks, scanned_upto = set(), None
    try:
        with open(_UNIVERSE_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("scanned_upto") in files:
            cached_stocks = set(cached.get("stocks", []))
            scanned_upto = cached["scanned_upto"]
    except Exception:
        pass

    if scanned_upto is None:
        pending = files
        print(f"  [母體] 首次建立清單，掃描 {len(pending)} 個價格檔（約需 1 分鐘，之後只掃新增檔）...")
    else:
        pending = files[files.index(scanned_upto) + 1:]

    out = cached_stocks | _scan_price_files(price_dir, pending)

    if pending:
        # 原子寫入：避免中斷時留下截斷的 JSON，下次讀到壞快取又要重掃
        try:
            tmp = _UNIVERSE_CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"scanned_upto": files[-1], "stocks": sorted(out)}, f)
            os.replace(tmp, _UNIVERSE_CACHE_PATH)
        except Exception:
            pass

    return out


def _load_watchlist_stocks() -> set:
    """讀 Stocks.txt 自選股；用絕對路徑，相對路徑會依 cwd 而定造成靜默讀不到。"""
    path = os.path.join(PARENT_DIR, "Stocks.txt")
    try:
        from scripts.utils import load_target_stocks
    except ImportError:
        try:
            from utils import load_target_stocks
        except Exception as exc:
            print(f"  [警告] 找不到 Stocks.txt 解析器：{type(exc).__name__}: {exc}")
            return set()
    try:
        return set(load_target_stocks(path))
    except Exception as exc:
        print(f"  [警告] 讀取 Stocks.txt 失敗：{type(exc).__name__}: {exc}")
        return set()


def resolve_target_stocks() -> list:
    """
    listed 模式的清單解析：data/raw_price 全歷史上市普通股 ∪ Stocks.txt 自選股。

    把來源與數量印出來不是為了好看：這類清單的失敗模式是「靜默縮成幾十檔」——
    抓取不會報錯，只是不再抓。來源與數量顯示在畫面上，問題才看得見。
    """
    from_stocks_txt = _load_watchlist_stocks()
    universe = _universe_from_raw_price()

    if universe:
        source = "data/raw_price（全歷史推導，含已下市）"
    else:
        # 保底：至少把自選股抓回來，但這代表資料層有問題，不能安靜帶過
        print("  [警告] data/raw_price 無法推導母體，僅抓 Stocks.txt 自選股。")
        print("     請先執行 python scripts/scraper.py -s twse 下載歷史價格資料。")
        source = "Stocks.txt（保底模式）"

    # 自選股一律聯集：它們可能含上櫃股或剛掛牌、尚未進入價格檔的標的
    targets = sorted(universe | from_stocks_txt)

    print(f"  母體來源: {source}")
    print(f"  上市普通股 {len(universe)} 檔 + 自選股 {len(from_stocks_txt)} 檔"
          f" → 去重後 {len(targets)} 檔")

    if len(targets) < _UNIVERSE_MIN_EXPECTED:
        print(f"  ★★ 警告：清單只有 {len(targets)} 檔，遠低於預期的 "
              f"{_UNIVERSE_MIN_EXPECTED}+ 檔。")
        print("     請確認 data/raw_price 是否有完整歷史（執行 "
              "python scripts/scraper.py -s twse 補齊），")
        print(f"     或刪除 {os.path.basename(_UNIVERSE_CACHE_PATH)} 後重跑以強制重建清單。")
    return targets


# ── FinMind 基本面快取天數設定 ──────────────────────────
try:
    from config import FINMIND_CACHE_DAYS as _FINMIND_CACHE_DAYS
except ImportError:
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
    # 原子寫入：先寫暫存檔再 os.replace，避免磁碟滿/中斷時留下截斷的 JSON。
    tmp_path = FAIL_LOG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, FAIL_LOG_PATH)


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


# 累計「實際送出的 HTTP 請求數」。
# 用途：日期迴圈用它判斷某一天有沒有真的碰網路。禮貌性等待的意義是「別連續打人家的
# 伺服器」，所以它該綁在請求上，而不是綁在「這天有缺檔」上——當缺的東西全部由快取、
# 既有檔案或期交所保存期界線就地解決時（例如 skip_dates.json 遺失後重建那 826 天），
# 一個請求都沒發卻照睡 2 秒，就是白白多等半小時。
_REQUESTS_MADE = 0


# =====================================================================
# 模組 1: TWSE & TAIFEX 每日全市場資料 (免費無限制)
# =====================================================================

# TWSE 對過於頻繁的查詢會回「查詢過於頻繁」限流訊息 (HTTP 200 但 data 為空)，
# 若不攔截會被下方「data 為空」判斷誤標為 NO_DATA。這正是先前整批歷史被寫成
# 假 no_data / unexpected_format 的成因，故偵測到限流一律退避重試，不當成無資料。
TWSE_THROTTLE_KEYWORDS  = ("查詢過於頻繁", "過於頻繁")
TWSE_THROTTLE_MAX_WAITS = 8    # 連續限流退避上限，超過視為本次失敗 (回 None，下次再試)
TWSE_THROTTLE_WAIT_SEC  = 30   # 每次退避基礎秒數 (線性遞增，單次上限 300 秒)

# 證交所 WAF 會對查詢過量的來源 IP 封鎖「資料查詢路徑」：HTTP 307 + 一頁
# "FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED." 的 HTML。
# 判別重點：此時 www.twse.com.tw 首頁與 openapi.twse.com.tw 仍可正常回 200，
# 所以不是網路、DNS 或端點失效，是這台機器的 IP 被擋。
#
# 這種狀態必須立即中止整輪，理由有二：
#   1. 重試無效，只會延長封鎖時間。
#   2. 絕不能計入 fail_log —— 被擋期間每個日期都會 error，累計 SKIP_AFTER_FAILS 次
#      就被列入「永久略過」名單，解封後再也不會補抓，等於把暫時性封鎖變成永久資料缺洞。
TWSE_BLOCK_KEYWORDS = ("FOR SECURITY REASONS", "CAN NOT BE ACCESSED")

# 連線異常（DNS 失敗、連線被拒、逾時）的重試policy，刻意與上面的限流退避分開計。
# 兩者性質不同：限流是伺服器叫你慢一點，等久有用；連線異常則可能是整條網路斷了，
# 套用 8 次 × 最長 300 秒的長退避只會把「抓不到」變成「每個日期卡 18 分鐘」。
# 短退避 5/10/15 秒，約 30 秒內認賠，讓失敗計數與下次重試機制去處理。
TWSE_NETWORK_MAX_RETRY = 3
TWSE_NETWORK_WAIT_SEC  = 5

# 每次 TWSE 請求前的最小間隔。原本只在「日期迴圈」尾端 sleep 一次，但每個交易日
# 會打 7 個 TWSE endpoint，實際速率是 3~5 req/s —— 這正是 IP 被封鎖的直接原因。
# 改為對每次請求節流，把整體速率壓到約 1 req/s。
TWSE_REQUEST_GAP = (0.7, 1.4)


class TwseAccessBlocked(Exception):
    """來源 IP 被證交所 WAF 封鎖，本輪抓取應立即中止（不記失敗、不寫 skip）。"""


# 期交所 futContractsDateDown 只提供近約 3 年的資料，更早的日期一律回 HTTP 200 +
# 617 bytes 的 HTML 首頁。舊版每次執行都會為每一天各發一次 POST 並 _polite_sleep()，
# 光這一項就佔掉數十分鐘；又因整輪跑不完就被中斷，skip_dates 只累積到少數幾筆，
# 下次重跑再從頭浪費一次。故在送出請求前先用日期界線攔截並寫入 skip，讓它只發生一次。
TAIFEX_HISTORY_DAYS = 365 * 3  # 早於「今天 - 此天數」的日期不再向期交所查詢


def _fetch_twse_json(url: str):
    """
    回傳:
      dict      = API 正常且有內容
      "NO_DATA" = API 正常但明確無資料 (含 404 網頁、很抱歉、data=[] 等)
      None      = 網路/解析錯誤、被 Ban 或連續限流放棄

    另有一種不回傳、直接拋 TwseAccessBlocked 的情況：來源 IP 被 WAF 封鎖。
    """
    throttle_waits = 0
    net_retries    = 0
    while True:
        try:
            global _REQUESTS_MADE
            _REQUESTS_MADE += 1
            _polite_sleep(*TWSE_REQUEST_GAP)   # 逐次請求節流，見 TWSE_REQUEST_GAP 說明
            r = requests.get(url, headers=HEADERS, timeout=15)
            body = r.text.strip()

            # ── WAF 封鎖：整輪中止，不重試、不計失敗 ──────────
            # 必須排在其他判斷之前：封鎖頁是 HTTP 307 + HTML，會分別被下方
            # 「非 200」判成 None、被「非 JSON」誤觸 8 輪退避重試，兩者都看不出原因。
            if any(k in body for k in TWSE_BLOCK_KEYWORDS):
                raise TwseAccessBlocked(
                    f"證交所已封鎖本機 IP 的資料查詢 (HTTP {r.status_code})。\n"
                    f"       這是查詢速率過高觸發的暫時性封鎖，通常數小時後自動解除；\n"
                    f"       重試無效且會延長封鎖，故立即中止本輪抓取。\n"
                    f"       本輪未完成的日期不會計入 failed_dates.json，解封後原地續抓即可。"
                )

            if r.status_code != 200:
                if r.status_code in (429, 403, 500, 502, 503, 504):
                    throttle_waits += 1
                    if throttle_waits > TWSE_THROTTLE_MAX_WAITS:
                        print(f"\n    [系統提示] 連續 {throttle_waits} 次收到 HTTP {r.status_code} 限流/錯誤，放棄本次查詢。", flush=True)
                        return None
                    wait = min(TWSE_THROTTLE_WAIT_SEC * throttle_waits, 300)
                    print(f"\n    [系統提示] 證交所伺服器回應 HTTP {r.status_code}，第 {throttle_waits}/{TWSE_THROTTLE_MAX_WAITS} 次退避 {wait} 秒後重試...", flush=True)
                    time.sleep(wait)
                    continue
                # 原本靜默 return None，使得失敗原因完全看不出來。
                print(f"\n    [系統提示] 非預期的 HTTP {r.status_code}，本次查詢視為失敗。", flush=True)
                return None

            # 證交所對某些過舊日期回傳 HTTP 200 但內容是 HTML (含 404)
            if not body or body[0] == "<":
                if "<title>404</title>" in body:
                    return "NO_DATA"
                throttle_waits += 1
                if throttle_waits > TWSE_THROTTLE_MAX_WAITS:
                    print(f"\n    [系統提示] 連續 {throttle_waits} 次收到非 JSON 網頁，放棄本次查詢。", flush=True)
                    return None
                wait = min(TWSE_THROTTLE_WAIT_SEC * throttle_waits, 300)
                print(f"\n    [系統提示] 證交所回應 HTML 非 JSON (可能暫時被擋)，第 {throttle_waits}/{TWSE_THROTTLE_MAX_WAITS} 次退避 {wait} 秒後重試...", flush=True)
                time.sleep(wait)
                continue

            data = r.json()

            stat = data.get("stat", "")

            # 攔截限流「查詢過於頻繁」：須早於下方空 data 的 NO_DATA 判斷，退避後重試。
            if any(k in stat for k in TWSE_THROTTLE_KEYWORDS):
                throttle_waits += 1
                if throttle_waits > TWSE_THROTTLE_MAX_WAITS:
                    print(f"\n    [系統提示] 連續 {throttle_waits} 次仍被證交所限流 (原因: {stat})，放棄本次查詢，待下次重試。", flush=True)
                    return None
                wait = min(TWSE_THROTTLE_WAIT_SEC * throttle_waits, 300)
                print(f"\n    [系統提示] 證交所限流 (原因: {stat})，第 {throttle_waits}/{TWSE_THROTTLE_MAX_WAITS} 次退避 {wait} 秒後重試...", flush=True)
                time.sleep(wait)
                continue

            # 攔截維護時間 (1:30 PM - 1:45 PM) 或其他伺服器維護 智慧等待
            if any(k in stat for k in ["暫停查詢", "結算時間", "維護"]):
                print(f"\n    [系統提示] 證交所伺服器結算或維護中 (原因: {stat})，暫停 30 分鐘後自動重試...", end="", flush=True)
                time.sleep(1800)
                continue

            if "很抱歉" in stat or "沒有" in stat:
                return "NO_DATA"

            # 攔截 data 欄位為空 list 且沒有 tables 的情況
            if isinstance(data.get("data"), list) and not data["data"] and not data.get("tables"):
                return "NO_DATA"

            return data
        except (KeyboardInterrupt, TwseAccessBlocked):
            raise
        except Exception as e:
            net_retries += 1
            if net_retries > TWSE_NETWORK_MAX_RETRY:
                print(f"\n    [系統提示] 連線異常 ({type(e).__name__}) 連續 {net_retries - 1} 次重試仍失敗，本次查詢視為失敗。", flush=True)
                return None
            wait = TWSE_NETWORK_WAIT_SEC * net_retries
            print(f"\n    [系統提示] 連線異常 ({type(e).__name__})，第 {net_retries}/{TWSE_NETWORK_MAX_RETRY} 次退避 {wait} 秒後重試...", flush=True)
            time.sleep(wait)
            continue


def crawl_daily_price(date_str: str, skip: dict) -> str:
    """
    三態回傳 (字串):
      "exists"        - 檔案已存在，不需要動作
      "market_closed" - API 確認當天休市，已寫入 skip
      "ok"            - 成功下載並存檔
      "skip"          - 有 skip 記錄但 reason 不是 market_closed
      "error"         - 網路錯誤或解析失敗，應計入 fail_log
      "not_ready"     - 今日資料尚未上架，不計入 fail_log，待稍後重試
    """
    path = os.path.join(RAW_PRICE_DIR, f"{date_str}_price.csv")
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
            return "not_ready"

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
    path = os.path.join(RAW_CHIPS_DIR, f"{date_str}_chips.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "chips", date_str): return True

    url = f"https://www.twse.com.tw/fund/T86?response=json&date={date_str}&selectType=ALL"
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "chips", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "chips", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_margin(date_str: str, skip: dict) -> bool:
    path = os.path.join(RAW_MARGIN_DIR, f"{date_str}_margin.csv")
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
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
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
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "margin", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_sbl(date_str: str, skip: dict) -> bool:
    path = os.path.join(RAW_MARGIN_DIR, f"{date_str}_sbl.csv")
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
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "sbl", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "sbl", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_twse_per(date_str: str, skip: dict) -> bool:
    path = os.path.join(RAW_TWSE_PER_DIR, f"{date_str}_twse_per.csv")
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
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
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
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "twse_per", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_daytrading(date_str: str, skip: dict) -> bool:
    path = os.path.join(RAW_CHIPS_DIR, f"{date_str}_daytrading.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "daytrading", date_str): return True

    url = (
        f"https://www.twse.com.tw/exchangeReport/TWTB4U"
        f"?response=json&date={date_str}"
    )
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "daytrading", date_str, reason="no_data")
        return True

    df = pd.DataFrame()
    if data.get("data"):
        df = _create_df_safely(data["data"], data.get("fields", []))
    elif data.get("tables"):
        # tables[0] 為市場統計總表(僅1列)，個股表以「證券代號」欄位辨識
        for tbl in data["tables"]:
            if tbl.get("data") and "證券代號" in tbl.get("fields", []):
                df = _create_df_safely(tbl["data"], tbl.get("fields", []))
                break

    if df.empty:
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "daytrading", date_str, reason="unexpected_format")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_fini_holding(date_str: str, skip: dict) -> bool:
    path = os.path.join(RAW_CHIPS_DIR, f"{date_str}_fini_holding.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "fini_holding", date_str): return True

    url = (
        f"https://www.twse.com.tw/fund/MI_QFIIS"
        f"?response=json&date={date_str}&selectType=ALLBUT0999"
    )
    data = _fetch_twse_json(url)
    if data is None:
        return False
    if data == "NO_DATA":
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "fini_holding", date_str, reason="no_data")
        return True

    df = _create_df_safely(data.get("data", []), data.get("fields", []))
    if df.empty:
        if date_str >= datetime.date.today().strftime("%Y%m%d"):
            return False
        _mark_skip_date(skip, "fini_holding", date_str, reason="empty_response")
        return True
    _save_csv(df, path)
    return True


def crawl_daily_taifex_inst(date_str: str, skip: dict) -> bool:
    path = os.path.join(RAW_TAIFEX_DIR, f"{date_str}_taifex_inst.csv")
    if _already_exists(path): return True
    if _is_skip_date(skip, "taifex_inst", date_str): return True

    # 支援 20260531 / 2026-05-31 / 2026/05/31 三種格式輸入
    date_obj = None
    try:
        clean_date = date_str.replace("-", "").replace("/", "")
        date_obj   = datetime.datetime.strptime(clean_date, "%Y%m%d").date()
        query_date = date_obj.strftime("%Y/%m/%d")
    except Exception:
        query_date = date_str

    # ── 超出期交所保存期：不發請求，直接標記略過 ─────────
    # 這些日期查詢必定回 HTML 首頁，下方本來也會判定 no_data，
    # 差別在於這裡省下一次 POST（timeout 15 秒）與外層的 _polite_sleep()。
    if date_obj is not None:
        cutoff = datetime.date.today() - datetime.timedelta(days=TAIFEX_HISTORY_DAYS)
        if date_obj < cutoff:
            _mark_skip_date(skip, "taifex_inst", date_str, reason="no_data")
            return True

    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    payload = {
        "queryStartDate": query_date,
        "queryEndDate":   query_date,
        "commodityId":    "TXF",
    }
    try:
        global _REQUESTS_MADE
        _REQUESTS_MADE += 1
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
            if date_str >= datetime.date.today().strftime("%Y%m%d"):
                return False
            _mark_skip_date(skip, "taifex_inst", date_str, reason="no_data")
            return True

        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            if date_str >= datetime.date.today().strftime("%Y%m%d"):
                return False
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
        path = os.path.join(RAW_SHAREHOLDING_DIR, f"{date_str}_shareholding.csv")
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
    token_path = os.path.join(PARENT_DIR, "FINMIND_TOKEN.txt")
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

    try:
        from config import FINMIND_MAX_LIMIT_WAITS as _MAX_LIMIT_WAITS
    except Exception:
        _MAX_LIMIT_WAITS = 6

    attempt = 0
    limit_waits = 0
    while attempt < max_retry:
        try:
            r = requests.get(FM_BASE_URL, params=params, timeout=30)

            if r.status_code in (429, 402):
                if os.environ.get("SKIP_ON_FINMIND_LIMIT") == "1":
                    print(f"\n    [FinMind] 觸發限速 (狀態碼 {r.status_code})，因設定 SKIP_ON_FINMIND_LIMIT，直接中斷下載以利後續執行。")
                    raise FinMindLimitExceeded("FinMind API quota limit reached.")

                # 等待重置有上限，避免無人值守時無限卡住（每次約 1 小時）
                if limit_waits >= _MAX_LIMIT_WAITS:
                    print(f"\n    [FinMind] 已連續等待 {limit_waits} 次額度重置仍受限，超過上限 {_MAX_LIMIT_WAITS} 次，放棄本次下載。")
                    raise FinMindLimitExceeded("FinMind API quota limit reached (max waits exceeded).")
                limit_waits += 1

                # 免費帳號每小時 600 次；429/402 = 額度耗盡，等待 1 小時重置
                resume_time = datetime.datetime.now() + datetime.timedelta(seconds=3600)
                print(f"\n    [FinMind] 觸發限速 (狀態碼 {r.status_code})，每小時額度已用盡。(第 {limit_waits}/{_MAX_LIMIT_WAITS} 次等待)")
                print(f"    [FinMind] 預計於 {resume_time.strftime('%H:%M:%S')} 額度重置，自動繼續...")
                for _ in range(60):
                    time.sleep(60)
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

        except FinMindLimitExceeded:
            raise
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
    force:        bool = False,
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
    if not force:
        cache_key = f"{stock_id}_{dataset_name}"
        if cache_key in missing_cache:
            try:
                last_check = datetime.date.fromisoformat(missing_cache[cache_key])
                if (datetime.date.today() - last_check).days < _NO_DATA_RECHECK_DAYS:
                    return "skipped"
            except Exception:
                pass

    try:
        if _already_exists(output_path) and not force:
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

    except FinMindLimitExceeded:
        raise
    except Exception as e:
        print(f"\n    [錯誤] {dataset_name} / {stock_id} 發生例外: {type(e).__name__} - {e}")
        return False


# =====================================================================
# 主下載控制
# =====================================================================

def _run_twse_stage(
    start_date_obj: datetime.date,
    end_date_obj:   datetime.date,
):
    """TWSE / TAIFEX / TDCC 逐日爬取。只讀寫 failed_dates.json 與 skip_dates.json。"""
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

        # ── 8 個 dataset 全部處理完畢才跳過 ───────────────
        files_required = [
            ("price",        os.path.join(RAW_PRICE_DIR,    f"{d_str}_price.csv")),
            ("chips",        os.path.join(RAW_CHIPS_DIR,    f"{d_str}_chips.csv")),
            ("twse_per",     os.path.join(RAW_TWSE_PER_DIR, f"{d_str}_twse_per.csv")),
            ("taifex_inst",  os.path.join(RAW_TAIFEX_DIR,   f"{d_str}_taifex_inst.csv")),
            ("margin",       os.path.join(RAW_MARGIN_DIR,   f"{d_str}_margin.csv")),
            ("sbl",          os.path.join(RAW_MARGIN_DIR,   f"{d_str}_sbl.csv")),
            ("daytrading",   os.path.join(RAW_CHIPS_DIR,    f"{d_str}_daytrading.csv")),
            ("fini_holding", os.path.join(RAW_CHIPS_DIR,    f"{d_str}_fini_holding.csv")),
        ]

        def _done(ds, p):
            return _already_exists(p) or _is_skip_date(skip_dates, ds, d_str)

        if all(_done(ds, p) for ds, p in files_required):
            skipped_days += 1
            curr += delta
            continue

        # ── price 已完成（有檔案或已 skip），補抓缺失子資料 ─
        _reqs_before = _REQUESTS_MADE
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
                ]
                if all(results):
                    downloaded_days += 1
                else:
                    print(f"    [警告] {d_str} 部分補抓失敗，下次重試")
                if _REQUESTS_MADE > _reqs_before:
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

        if price_result == "not_ready":
            # 今日資料尚未上架，不計入失敗次數且不列入 skip，下次仍可重試
            _polite_sleep()
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


def _run_finmind_stage(
    start_date_obj: datetime.date,
    end_date_obj:   datetime.date,
    target_stocks:  list = None,
):
    """FinMind 基本面爬取。只讀寫 no_finmind_data.json 與 missing_fm_datasets.json。"""
    if not target_stocks:
        print("\n[略過] FinMind 階段：股票清單為空。")
        return

    start_str = start_date_obj.strftime("%Y-%m-%d")
    end_str   = end_date_obj.strftime("%Y-%m-%d")

    print("\n[啟動] FinMind 基本面資料爬蟲")
    if not FINMIND_TOKEN:
        print("  [注意] 尚未設定 FINMIND_TOKEN.txt，將使用免費額度。")

    no_data_cache    = _load_no_data_cache()
    missing_fm_cache = _load_missing_fm()   # ← 迴圈外只讀一次，減少磁碟 I/O

    total         = len(target_stocks)
    idx           = 0   # 迴圈未進入就拋額度例外時，except 區塊仍需要 idx
    skipped_etf   = 0
    skipped_cache = 0
    updated       = 0
    partial_miss  = 0
    errors_total  = 0

    try:
        for idx, stock_id in enumerate(target_stocks, 1):
            print(f"  FinMind 進度: {idx}/{total} ({stock_id})          ", end="\r", flush=True)

            # ETF_SET 來自 stock_categories.json，該檔缺漏時會是空集合，
            # 於是 ETF 照樣被抓。補上不依賴外部檔案的結構性判斷當第二道防線，
            # 同時擋掉沒有獨立財報的特別股 (3036A / 3702A / 8112A)。
            if stock_id in ETF_SET or not _is_common_stock(stock_id):
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
                path = os.path.join(RAW_FINANCIAL_DIR, fname)
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
    except FinMindLimitExceeded:
        print(f"\n[中斷] 偵測到 FinMind API 額度用盡！提前結束 FinMind 爬蟲流程。")
        print(
            f"  中斷前進度: {idx}/{total} 檔"
            f" (略過ETF: {skipped_etf}"
            f" | 略過快取: {skipped_cache}"
            f" | 新更新: {updated}"
            f" | 部分缺項: {partial_miss}"
            f" | 錯誤: {errors_total})"
        )
        raise


def download_history_data(
    start_date_obj: datetime.date,
    end_date_obj:   datetime.date,
    target_stocks:  list = None,
    source:         str = "all",
):
    """
    依 source 決定跑哪些階段。

    source="twse"    只跑證交所/期交所/集保（不需要 Token，慢在逐日 request）
    source="finmind" 只跑 FinMind 財報（慢在逐檔 API，且受額度限制）
    source="all"     兩段依序跑（預設，與舊行為相同）

    兩段沒有共用可變狀態，因此 twse 與 finmind 可以在兩個終端機同時執行。
    """
    if source not in ("all", "twse", "finmind"):
        raise ValueError(f"未知的 source: {source!r} (可用: all / twse / finmind)")

    if source in ("all", "twse"):
        _run_twse_stage(start_date_obj, end_date_obj)

    if source in ("all", "finmind"):
        _run_finmind_stage(start_date_obj, end_date_obj, target_stocks)


# =====================================================================
# 合併功能：分類、補件與完整性修復工具
# =====================================================================

def fetch_industry_categories():
    """自動抓取台股全市場分類與代碼並寫入 stock_categories.json (取代 fetch_categories.py)"""
    print("=" * 60)
    print("  開始下載台股全市場產業分類資料 (FinMind)")
    print("=" * 60)
    output_json = os.path.join(BASE_DIR, "stock_categories.json")
    url = "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo"
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()
        if data.get("msg") != "success":
            print(f"API 回應錯誤: {data.get('msg')}")
            return False
        stock_list = data.get("data", [])
        print(f"  成功取得 {len(stock_list)} 筆資料，開始進行分類整理...")
        from collections import defaultdict
        categories = defaultdict(dict)
        for item in stock_list:
            stock_id = item.get("stock_id", "")
            stock_name = item.get("stock_name", "")
            industry = item.get("industry_category", "未分類")
            if stock_id.isalnum() and 4 <= len(stock_id) <= 6:
                categories[industry][stock_id] = stock_name
        if "" in categories:
            del categories[""]
        sorted_categories = {}
        for ind, stocks in categories.items():
            sorted_categories[ind] = dict(sorted(stocks.items(), key=lambda x: x[0]))
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(sorted_categories, f, ensure_ascii=False, indent=4)
        print("-" * 60)
        print(f"  [完成] 產業分類已儲存至: {output_json}")
        print(f"  共分出 {len(sorted_categories)} 個產業類別。")
        return True
    except Exception as e:
        print(f"[錯誤] 下載分類資料失敗: {e}")
        return False


def patch_stock_finmind(stock_id: str):
    """手動強制補抓特定股票之最新財報，不走 15 天快取與 90 天空值快取 (取代 patch_finmind.py)"""
    print("=" * 60)
    print(f"  強制手動補抓 FinMind 財報資料 (代號: {stock_id})")
    print("=" * 60)
    try:
        from config import START_DATE as _START_DATE
    except ImportError:
        _START_DATE = datetime.date(2020, 1, 1)
        
    start_str = _START_DATE.strftime("%Y-%m-%d")
    end_str = datetime.date.today().strftime("%Y-%m-%d")
    
    datasets = [
        ("營收",   "TaiwanStockMonthRevenue",       f"{stock_id}_monthly_revenue.csv"),
        ("損益表", "TaiwanStockFinancialStatements", f"{stock_id}_financial_stmt.csv"),
        ("資產表", "TaiwanStockBalanceSheet",        f"{stock_id}_balance_sheet.csv"),
        ("現金流", "TaiwanStockCashFlowsStatement",  f"{stock_id}_cashflow.csv"),
        ("股利",   "TaiwanStockDividend",            f"{stock_id}_dividend.csv"),
    ]
    for label, dataset, fname in datasets:
        path = os.path.join(RAW_FINANCIAL_DIR, fname)
        print(f"  -> 正在抓取 {label}...", end=" ", flush=True)
        res = _crawl_fm_dataset(dataset, stock_id, start_str, end_str, path, {}, force=True)
        if res is True:
            print("OK")
        elif res == "skipped":
            print("已是最新 (跳過)")
        elif res is None:
            print("無資料")
        else:
            print("失敗")
        _polite_sleep(1, 2)
    print("=" * 60)


def check_data_integrity():
    """完整資料庫完整性修復工具，檢查空檔、欄位缺失與極端價格幽靈資料 (取代 check_data.py)"""
    import glob
    try:
        from config import GHOST_DATA_PCT_THRESHOLD
    except ImportError:
        GHOST_DATA_PCT_THRESHOLD = 0.15
    financial_dir = RAW_FINANCIAL_DIR
    specs = {
        "monthly_revenue": ["date", "revenue"],
        "financial_stmt":  ["date", "type", "value"],
        "balance_sheet":   ["date", "type", "value"],
        "cashflow":        ["date", "type", "value"],
        "dividend":        ["date", "CashEarningsDistribution"]
    }

    # 1. 檢查 FinMind
    print("==================================================")
    print("  [1/3] 檢查 FinMind 財報與營收檔案...")
    print("==================================================")
    fm_files = glob.glob(os.path.join(financial_dir, "*.csv"))
    fm_del = 0
    for fpath in fm_files:
        fname = os.path.basename(fpath)
        parts = fname.replace(".csv", "").split("_", 1)
        if len(parts) != 2: continue
        sid, ds_name = parts
        if ds_name not in specs: continue
        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig")
            if df.empty or any(c not in df.columns for c in specs[ds_name]):
                os.remove(fpath)
                fm_del += 1
        except Exception:
            os.remove(fpath)
            fm_del += 1
    print(f"  FinMind 檢查完畢。共刪除 {fm_del} 個異常/損毀檔案。\n")

    # 2. 檢查證交所
    print("==================================================")
    print("  [2/3] 檢查 證交所/期交所 歷史資料檔...")
    print("==================================================")
    twse_del = 0
    for dir_path in TWSE_DAILY_DIRS:
        if not os.path.exists(dir_path): continue
        for fpath in glob.glob(os.path.join(dir_path, "*.csv")):
            try:
                size = os.path.getsize(fpath)
                if size <= 3:
                    os.remove(fpath)
                    twse_del += 1
                    continue
                df = pd.read_csv(fpath, encoding="utf-8-sig", dtype=str)
                if df.empty:
                    os.remove(fpath)
                    twse_del += 1
                    continue
                cols = list(df.columns)
                if len(cols) == 1 and ("html" in cols[0].lower() or "很抱歉" in cols[0]):
                    os.remove(fpath)
                    twse_del += 1
            except Exception:
                os.remove(fpath)
                twse_del += 1
    print(f"  證交所檢查完畢。共刪除 {twse_del} 個損毀/錯誤網頁檔案。\n")

    # 3. 幽靈價格檢查
    print("==================================================")
    print("  [3/3] 檢查極端價格異常 (休市假數據/幽靈資料)...")
    print("==================================================")
    price_files = sorted(glob.glob(os.path.join(RAW_PRICE_DIR, "*_price.csv")))
    history = []
    for fpath in price_files:
        date_str = os.path.basename(fpath).split("_")[0]
        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig", dtype=str)
            row = df[df.iloc[:, 0] == "0050"]
            if not row.empty:
                close_col = next((c for c in df.columns if "收盤價" in c), None)
                if close_col:
                    close_price = float(row[close_col].values[0].replace(",", ""))
                    history.append({"date": date_str, "file": fpath, "price": close_price})
        except Exception:
            continue
    # 用模組層既有常數，避免同一份檔案在兩處各拼一次路徑而日後走岔
    skip_dates_path = SKIP_DATES_PATH
    fail_log_path = FAIL_LOG_PATH
    skip_dates = {}
    if os.path.exists(skip_dates_path):
        try:
            with open(skip_dates_path, "r", encoding="utf-8") as f: skip_dates = json.load(f)
        except Exception: pass
    fail_log = {}
    if os.path.exists(fail_log_path):
        try:
            with open(fail_log_path, "r", encoding="utf-8") as f: fail_log = json.load(f)
        except Exception: pass

    ghost_del = 0
    for i in range(1, len(history)):
        prev, curr = history[i-1], history[i]
        pct = abs((curr["price"] - prev["price"]) / prev["price"])
        if pct > GHOST_DATA_PCT_THRESHOLD:
            # 判斷是「真實的股票分割/永久跳空/大型除權息」還是「單日的幽靈資料異常」
            # 幽靈資料特色：只有單日異常暴跌/暴漲，隔天就立刻彈回原本的價格水準
            # 股票分割特色：價格跳空後，隔天會維持在新價格附近（永久性位移）
            is_anomaly = True
            if i + 1 < len(history):
                nxt = history[i+1]
                pct_change_curr_to_next = abs((nxt["price"] - curr["price"]) / curr["price"])
                pct_change_prev_to_next = abs((nxt["price"] - prev["price"]) / prev["price"])
                
                # 如果隔天價格與當天價格相近 (波動 <= GHOST_DATA_PCT_THRESHOLD)，且與前一天相比仍有巨大跳空，代表這是永久性分割/跳空
                if pct_change_curr_to_next <= GHOST_DATA_PCT_THRESHOLD and pct_change_prev_to_next > GHOST_DATA_PCT_THRESHOLD:
                    is_anomaly = False
            else:
                # 若為歷史最後一天，在無法比對隔天情況下，為了避免分割/跳空造成連續刪除的骨牌效應，
                # 只有當價格極端異常 (例如 <= 0) 時才判定為異常，其餘跳空先保留，待隔天有新資料時再行判定
                if curr["price"] > 0:
                    is_anomaly = False
            
            if is_anomaly:
                bad_date = curr["date"]
                print(f"  [幽靈資料] {bad_date} 發現 0050 價格異常跳空！")
                for d in TWSE_DAILY_DIRS:
                    for bad_f in glob.glob(os.path.join(d, f"{bad_date}_*.csv")):
                        os.remove(bad_f)
                for k in list(skip_dates.keys()):
                    if k.endswith(f"_{bad_date}"):
                        del skip_dates[k]
                if bad_date in fail_log: del fail_log[bad_date]
                ghost_del += 1
                curr["price"] = prev["price"]

    if ghost_del > 0:
        with open(skip_dates_path, "w", encoding="utf-8") as f:
            json.dump(skip_dates, f, indent=4, ensure_ascii=False)
        with open(fail_log_path, "w", encoding="utf-8") as f:
            json.dump(fail_log, f, indent=4, ensure_ascii=False)
    print(f"  幽靈資料檢查完畢。共清理 {ghost_del} 天的異常幽靈日資料。\n")
    print("=" * 50)
    print("  資料庫完整性校驗與修復工作已全部執行完成！")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="台股多源資料爬蟲與維護中心")
    parser.add_argument("-p", "--patch", type=str, help="手動強制補抓指定股票代號的 FinMind 財報資料")
    parser.add_argument("-fc", "--fc", "--fetch-categories", dest="fetch_categories", action="store_true", help="重新抓取並更新全市場產業分類對照表")
    parser.add_argument("-c", "--check", action="store_true", help="執行資料庫損毀/異常/極端價格幽靈資料校驗與修復")
    parser.add_argument(
        "-s", "--source",
        choices=["all", "twse", "finmind"],
        default="all",
        help=(
            "選擇要抓的資料來源。twse=證交所/期交所/集保；finmind=財報基本面；"
            "all=兩者依序（預設）。twse 與 finmind 可在兩個終端機同時執行。"
        ),
    )
    
    args = parser.parse_args()
    
    if args.patch:
        patch_stock_finmind(args.patch)
    elif args.fetch_categories:
        fetch_industry_categories()
    elif args.check:
        check_data_integrity()
    else:
        # 預設執行標準資料增量抓取 (原 main.py 入口)
        _SOURCE_LABEL = {
            "all":     "TWSE + FinMind (完整)",
            "twse":    "TWSE / TAIFEX / TDCC (官方免 Token)",
            "finmind": "FinMind 基本面財報",
        }[args.source]
        print("=" * 60)
        print(f"  啟動資料增量下載流程... [來源: {_SOURCE_LABEL}]")
        print("=" * 60)
        try:
            from config import START_DATE, FINMIND_FETCH_MODE
        except ImportError:
            START_DATE = datetime.date(2020, 1, 1)
            FINMIND_FETCH_MODE = "listed"

        # 讀取股票清單 (從中央 config 的 FINMIND_FETCH_MODE 控制)
        # 清單只有 FinMind 階段會用到；--source twse 時直接略過解析，
        # 省下掃描價格檔 / 讀 stock_categories.json 的時間，也不印無關訊息。
        stock_list = None
        if args.source != "twse":
            if FINMIND_FETCH_MODE == "listed":
                # 由 data/raw_price 推導，詳見 resolve_target_stocks() 上方的說明
                stock_list = resolve_target_stocks()
            else:
                target_stocks = set(_load_watchlist_stocks())

                # 根據 FINMIND_FETCH_MODE 加載產業股票
                cat_path = os.path.join(BASE_DIR, "stock_categories.json")
                if not os.path.exists(cat_path):
                    # 舊版在此靜默略過，清單會縮成只剩自選股卻毫無錯誤訊息
                    print(f"[警告] 找不到 {cat_path}，清單將只剩 Stocks.txt 自選股。")
                    print("       請先執行 python scripts/scraper.py -fc 產生產業對照表。")
                    categories = {}
                else:
                    with open(cat_path, "r", encoding="utf-8") as f:
                        categories = json.load(f)

                if FINMIND_FETCH_MODE == "all":
                    for ind_name, stocks_dict in categories.items():
                        target_stocks.update(stocks_dict.keys())
                else:
                    # limited 模式: 載入 config.py 中設定為 True 的產業股票
                    try:
                        from config import TRAIN_INDUSTRIES
                        for ind_name, is_enabled in TRAIN_INDUSTRIES.items():
                            if is_enabled and ind_name in categories:
                                target_stocks.update(categories[ind_name].keys())
                    except Exception as e:
                        print(f"[警告] 無法載入 config 產業清單 ({e})")

                stock_list = sorted(target_stocks)

            print(f"下載目標股票數量: {len(stock_list)} 檔 (模式: {FINMIND_FETCH_MODE})")

        from contextlib import ExitStack
        try:
            with ExitStack() as _stack:
                for _lock_path in _locks_for(args.source):
                    _stack.enter_context(_SingleInstanceLock(_lock_path))
                download_history_data(
                    START_DATE, datetime.date.today(),
                    target_stocks=stock_list,
                    source=args.source,
                )
        except TwseAccessBlocked as e:
            # 來源 IP 被證交所 WAF 擋下。已抓到的檔案都保留，未完成的日期也未被
            # 記成失敗，等封鎖解除後重跑即可從中斷處續抓。
            print(f"\n[中止] {e}")
            print("       建議：等待數小時後重跑；若要更保守，可把 scraper.py 的")
            print("       TWSE_REQUEST_GAP 調大 (例如 (1.5, 2.5)) 再執行。")
            sys.exit(1)
        except RuntimeError as e:
            # 另一個 scraper 實例正在執行 (單一實例鎖被佔用)
            print(f"[中止] {e}")
            sys.exit(1)
        except FinMindLimitExceeded:
            # 退出碼 99 通知上層 Pipeline 可以跳過並繼續
            sys.exit(99)
