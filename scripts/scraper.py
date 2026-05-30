"""
scraper.py — 台灣股市多源資料爬蟲 (完整版)
====================================================
資料來源: 台灣證券交易所 (TWSE) + 集保中心 (TDCC)
----------------------------------------------------
已確認可用的 API 端點 (2025-05):

  [日更] 股價行情          MI_INDEX    → data/raw_price/
  [日更] 三大法人個股       T86         → data/raw_chips/
  [日更] 法人買賣金額彙總   BFI82U      → data/raw_chips/
  [日更] 融資融券個股+彙總  MI_MARGN    → data/raw_margin/
  [日更] 借券+融券個股      TWT93U      → data/raw_margin/
  [日更] 當沖統計           TWTB4U      → data/raw_daytrading/
  [週更] 持股分級 (TDCC)    TDCC CSV    → data/raw_shareholding/

說明:
  * 外資持股 (MI_QFIIS) 改用 T86 的外資欄位替代 (更穩定)
  * 融資融券: MI_MARGN tables[1] 含每股融資/融券完整數據
  * 持股分級: 集保 CSV 週更，用 verify=False 繞過 SSL 問題
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
for folder in [
    "data/raw_price",
    "data/raw_chips",
    "data/raw_margin",
    "data/raw_daytrading",
    "data/raw_shareholding",
]:
    os.makedirs(folder, exist_ok=True)

# ── 共用 Headers ────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}

# ── 工具函式 ────────────────────────────────────────────

def _clean_html(df: pd.DataFrame) -> pd.DataFrame:
    """移除 DataFrame 中所有欄位的 HTML 標籤"""
    return df.replace(r"<[^>]+>", "", regex=True)


def _fetch_json(url: str, timeout: int = 15, verify: bool = True):
    """發送 GET 請求並回傳 JSON dict；失敗回傳 None"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=verify)
        if r.status_code != 200:
            return None
        body = r.text.strip()
        if not body or body[0] == "<":
            return None  # 被 WAF 擋住，回傳 HTML
        return r.json()
    except Exception:
        return None


def _already_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _save_csv(df: pd.DataFrame, path: str):
    _clean_html(df).to_csv(path, index=False, encoding="utf-8-sig")


def _polite_sleep(lo: float = 1.5, hi: float = 3.0):
    time.sleep(random.uniform(lo, hi))


# ══════════════════════════════════════════════════════
# 1. 全市場股價行情 (MI_INDEX)
#    欄位: 證券代號、開高低收、成交量、成交筆數
# ══════════════════════════════════════════════════════

def crawl_daily_price(date_str: str) -> bool:
    """抓取並儲存指定日期的全市場股價 (上市)"""
    path = f"data/raw_price/{date_str}_price.csv"
    if _already_exists(path):
        print(f"  [股價] {date_str} 已存在，略過。")
        return True

    url = (
        "https://www.twse.com.tw/exchangeReport/MI_INDEX"
        f"?response=json&date={date_str}&type=ALL"
    )
    data = _fetch_json(url)
    if data is None:
        print(f"  [股價] {date_str} 請求失敗。")
        return False

    # 新版 API → tables[]；舊版 → data9
    df = None
    if "tables" in data:
        for tbl in data["tables"]:
            if "fields" in tbl and "證券代號" in tbl.get("fields", []):
                df = pd.DataFrame(tbl["data"], columns=tbl["fields"])
                break
    elif "data9" in data:
        df = pd.DataFrame(data["data9"], columns=data["fields9"])

    if df is None or df.empty:
        print(f"  [股價] {date_str} 無開盤資料。")
        return False

    _save_csv(df, path)
    print(f"  [股價] {date_str} 下載成功 ({len(df)} 筆)")
    return True


# ══════════════════════════════════════════════════════
# 2. 三大法人個股買賣超 (T86)
#    欄位: 外資/投信/自營 各別買進/賣出/買賣超股數
# ══════════════════════════════════════════════════════

def crawl_daily_chips(date_str: str) -> bool:
    """三大法人每股買賣超 (T86)"""
    path = f"data/raw_chips/{date_str}_chips.csv"
    if _already_exists(path):
        print(f"  [法人] {date_str} 已存在，略過。")
        return True

    url = (
        "https://www.twse.com.tw/fund/T86"
        f"?response=json&date={date_str}&selectType=ALL"
    )
    data = _fetch_json(url)
    if data is None or "data" not in data or not data["data"]:
        print(f"  [法人] {date_str} 無資料。")
        return False

    df = pd.DataFrame(data["data"], columns=data["fields"])
    _save_csv(df, path)
    print(f"  [法人] {date_str} 下載成功 ({len(df)} 筆)")
    return True


# ══════════════════════════════════════════════════════
# 3. 三大法人合計買賣金額彙總 (BFI82U)
#    欄位: 自營商/投信/外資 買進/賣出/差額 (全市場金額)
#    用途: 計算大盤法人淨買力，作為系統性因子
# ══════════════════════════════════════════════════════

def crawl_daily_institution_total(date_str: str) -> bool:
    """全市場三大法人合計買賣金額"""
    path = f"data/raw_chips/{date_str}_inst_total.csv"
    if _already_exists(path):
        print(f"  [法人匯總] {date_str} 已存在，略過。")
        return True

    url = (
        "https://www.twse.com.tw/fund/BFI82U"
        f"?response=json&date={date_str}&selectType=ALL"
    )
    data = _fetch_json(url)
    if data is None or "data" not in data or not data["data"]:
        print(f"  [法人匯總] {date_str} 無資料。")
        return False

    df = pd.DataFrame(data["data"], columns=data["fields"])
    _save_csv(df, path)
    print(f"  [法人匯總] {date_str} 下載成功 ({len(df)} 筆)")
    return True


# ══════════════════════════════════════════════════════
# 4. 融資融券個股彙總 (MI_MARGN)
#    欄位: 融資買進/賣出/現償/前日餘額/今日餘額
#          融券賣出/買進/現償/前日餘額/今日餘額/資券互抵
#    用途: 散戶槓桿信號，融券做空指標
# ══════════════════════════════════════════════════════

def crawl_daily_margin(date_str: str) -> bool:
    """融資融券每股明細 (MI_MARGN tables[1])"""
    path = f"data/raw_margin/{date_str}_margin.csv"
    if _already_exists(path):
        print(f"  [融資券] {date_str} 已存在，略過。")
        return True

    url = (
        "https://www.twse.com.tw/exchangeReport/MI_MARGN"
        f"?response=json&date={date_str}&selectType=ALL"
    )
    data = _fetch_json(url)
    if data is None:
        print(f"  [融資券] {date_str} 請求失敗。")
        return False

    # tables[1] 是每股明細; tables[0] 是全市場彙總統計
    tables = data.get("tables", [])
    if len(tables) < 2 or not tables[1].get("data"):
        print(f"  [融資券] {date_str} 無資料。")
        return False

    tbl = tables[1]
    df = pd.DataFrame(tbl["data"], columns=tbl["fields"])
    _save_csv(df, path)
    print(f"  [融資券] {date_str} 下載成功 ({len(df)} 筆)")
    return True


# ══════════════════════════════════════════════════════
# 5. 借券個股餘額 (TWT93U)
#    欄位: 借券前日餘額、賣出、買進、今日餘額
#          融券前日餘額、賣出、買進、今日餘額
#    用途: 法人做空的直接代理變數（借券=主力空單）
# ══════════════════════════════════════════════════════

def crawl_daily_sbl(date_str: str) -> bool:
    """借券+融券個股餘額 (TWT93U)"""
    path = f"data/raw_margin/{date_str}_sbl.csv"
    if _already_exists(path):
        print(f"  [借券] {date_str} 已存在，略過。")
        return True

    url = (
        "https://www.twse.com.tw/exchangeReport/TWT93U"
        f"?response=json&date={date_str}&selectType=ALL"
    )
    data = _fetch_json(url)
    if data is None or "data" not in data or not data["data"]:
        print(f"  [借券] {date_str} 無資料。")
        return False

    df = pd.DataFrame(data["data"], columns=data["fields"])
    _save_csv(df, path)
    print(f"  [借券] {date_str} 下載成功 ({len(df)} 筆)")
    return True


# ══════════════════════════════════════════════════════
# 6. 當日沖銷統計 (TWTB4U)
#    欄位: 當沖成交股數、占市場比重、買進賣出金額
#    用途: 當沖比例高 = 短線投機氣氛強 (情緒指標)
# ══════════════════════════════════════════════════════

def crawl_daily_daytrading(date_str: str) -> bool:
    """當日沖銷交易統計 (TWTB4U)"""
    path = f"data/raw_daytrading/{date_str}_daytrading.csv"
    if _already_exists(path):
        print(f"  [當沖] {date_str} 已存在，略過。")
        return True

    url = (
        "https://www.twse.com.tw/exchangeReport/TWTB4U"
        f"?response=json&date={date_str}&selectType=MS"
    )
    data = _fetch_json(url)
    if data is None:
        print(f"  [當沖] {date_str} 無資料。")
        return False

    tables = data.get("tables", [])
    if not tables or not tables[0].get("fields") or not tables[0].get("data"):
        print(f"  [當沖] {date_str} 無資料。")
        return False

    tbl = tables[0]
    df = pd.DataFrame(tbl["data"], columns=tbl["fields"])
    _save_csv(df, path)
    print(f"  [當沖] {date_str} 下載成功 ({len(df)} 筆)")
    return True


# ══════════════════════════════════════════════════════
# 7. 持股分級 (集保 TDCC — 週更)
#    欄位: 證券代號、持股分級(1-17)、人數、股數、比例
#    說明:
#      分級 1  = 1~999 股 (未滿 1 張，散戶)
#      分級 2  = 1,000~5,000 股 (1~5 張)
#      分級 10 = 10,001~50,000 股 (10~50 張)
#      分級 15 = 400,001~1,000,000 股 (400~1000 張, 大戶)
#      分級 16 = 1,000,001 股以上 (千張以上, 主力)
#      分級 17 = 合計
#    用途: 計算「大戶比例上升/下降」的籌碼轉移因子
#    頻率: 每週五更新 (非每日)
# ══════════════════════════════════════════════════════

def crawl_weekly_shareholding() -> bool:
    """
    下載 TDCC 最新一期全市場持股分級 CSV。
    TDCC 每週更新，下載後以「資料日期」作為檔名。
    """
    # 先試著取得最新資料，看是哪個日期
    url = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
    try:
        import warnings
        warnings.filterwarnings("ignore")
        r = requests.get(url, headers=HEADERS, timeout=60, verify=False)
        if r.status_code != 200:
            print("  [持股分級] 請求失敗。")
            return False

        df = pd.read_csv(io.StringIO(r.text))
        # 清理股票代號欄位的空格
        df["證券代號"] = df["證券代號"].astype(str).str.strip()

        # 取得本次資料的日期
        dates = df["資料日期"].unique()
        if len(dates) == 0:
            print("  [持股分級] 無日期資訊。")
            return False

        date_str = str(dates[0])  # e.g. 20260529
        path = f"data/raw_shareholding/{date_str}_shareholding.csv"

        if _already_exists(path):
            print(f"  [持股分級] {date_str} 已存在，略過。")
            return True

        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  [持股分級] {date_str} 下載成功 ({len(df)} 筆，{df['證券代號'].nunique()} 支股票)")
        return True

    except Exception as e:
        print(f"  [持股分級] 下載失敗: {e}")
        return False


# ══════════════════════════════════════════════════════
# 主流程: 批次下載
# ══════════════════════════════════════════════════════

def _is_holiday(date_obj: datetime.date) -> bool:
    """利用 taiwan-holidays 判斷假日，若套件沒有該日資料則回傳 False"""
    if tw_cal is None:
        return False
    try:
        return tw_cal.is_holiday(date_obj)
    except ValueError:
        return False  # 套件沒該日資料時，預設嘗試爬取


def download_history_data(start_date_obj: datetime.date, end_date_obj: datetime.date):
    """
    批次下載所有資料類型。

    資料類型:
      - 每日: 股價、三大法人、法人金額彙總、融資融券、借券、當沖
      - 每週: 持股分級 (週五自動觸發，或每次執行時下載最新一期)

    防封鎖:
      每個成功的網路請求後隨機延遲 1.5~3 秒，
      每天所有資料下完後再額外延遲 3~6 秒。
    """
    delta = datetime.timedelta(days=1)
    curr = start_date_obj

    print("開始執行歷史資料下載任務...")
    print(f"  範圍: {start_date_obj} ~ {end_date_obj}")
    print()

    # 程式啟動時先下載一次最新持股分級週報
    print("── 持股分級週報 ─────────────────────")
    crawl_weekly_shareholding()
    print()
    _polite_sleep(2, 4)

    while curr <= end_date_obj:
        date_str = curr.strftime("%Y%m%d")

        # ── 假日過濾 ─────────────────────────────────
        if _is_holiday(curr):
            print(f"  [休市] {date_str} 是假日，跳過。")
            curr += delta
            continue

        print(f"── {date_str} ──────────────────")

        # ── 1. 股價 ──────────────────────────────────
        has_open = crawl_daily_price(date_str)
        if not has_open:
            # 可能真的沒開盤（補假、臨時停市），跳過後續
            curr += delta
            continue

        _polite_sleep()

        # ── 2. 三大法人個股買賣超 ─────────────────────
        crawl_daily_chips(date_str)
        _polite_sleep()

        # ── 3. 法人合計買賣金額 ───────────────────────
        crawl_daily_institution_total(date_str)
        _polite_sleep()

        # ── 4. 融資融券個股明細 ───────────────────────
        crawl_daily_margin(date_str)
        _polite_sleep()

        # ── 5. 借券個股餘額 ───────────────────────────
        crawl_daily_sbl(date_str)
        _polite_sleep()

        # ── 6. 當日沖銷統計 ───────────────────────────
        crawl_daily_daytrading(date_str)
        _polite_sleep(2.0, 4.0)  # 每天結束後多等一會

        print()
        curr += delta

    print("歷史資料下載任務結束。")