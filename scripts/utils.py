import os

# 統一將 BASE_DIR 設定為專案根目錄 (即 scripts/ 的上一層)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_stocks_file(file_path: str = "Stocks.txt") -> dict:
    """
    解析 Stocks.txt。
    支援兩種格式：
      格式 A（僅代號）  : 每行一個股票代號，例如 "2330"
      格式 B（含成本）  : 代號,買進成本價，例如 "2330,950.0"

    回傳 dict: { stock_id: buy_cost_or_None }
    """
    watchlist = {}
    
    # 支援絕對與相對路徑
    fp = file_path
    if not os.path.isabs(fp):
        fp = os.path.join(BASE_DIR, file_path)
        
    if not os.path.exists(fp):
        # 備用降級防呆：往上一層或同級目錄尋找
        alt_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if os.path.exists(alt_fp):
            fp = alt_fp
            
    if not os.path.exists(fp):
        return watchlist

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            sid = parts[0].strip()
            cost = None
            if len(parts) >= 2:
                try:
                    cost = float(parts[1].strip())
                except ValueError:
                    pass
            watchlist[sid] = cost

    return watchlist

def load_target_stocks(file_path: str = "Stocks.txt") -> list:
    """
    解析 Stocks.txt 並回傳所有股票代號的列表。
    """
    watchlist = parse_stocks_file(file_path)
    if not watchlist:
        return ["2330"]
    return list(watchlist.keys())

def parse_stocks_detailed(file_path: str = "Stocks.txt") -> dict:
    """
    解析 Stocks.txt。
    支援五種格式：
      格式 A (僅代號)         : 2330
      格式 B (含成本)         : 2330,950.0
      格式 C (成本與股數)      : 2330,950.0,1000
      格式 D (含買入日停損價)   : 2330,950.0,1000,910.5
      格式 E (含買入日期)       : 2330,950.0,1000,910.5,2026-06-20

    第 4 欄為「買入日鎖定的 ATR 停損價」（由 inference.py 買進建議提供），
    填了之後 inference.py 以此精確判定停損；未填則退回當日 ATR 近似值。
    第 5 欄為「買入日期」（YYYY-MM-DD 或 YYYYMMDD）；填了之後 inference.py 才能
    比照 trading_sim.py：D3轉弱／移動止盈出場須滿 MIN_HOLD_DAYS（停損不限），
    並啟用移動止盈判定；未填則退回「無視持有天數，D3 轉弱即建議賣」。

    回傳 dict: { stock_id: {"cost": ..., "shares": ..., "stop_price": ..., "buy_date": ...} }
    （buy_date 為原始字串或 None，交由呼叫端解析為日期。）

    ⚠️ 同代號多筆（不同時期／價位買進）時，後者會覆蓋前者、只保留最後一筆。
       需逐筆保留（每筆各自成本／停損價／買入日）請改用 parse_stocks_lots()。
    """
    detailed_watchlist = {}
    fp = _resolve_stocks_path(file_path)
    if not fp:
        return detailed_watchlist

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_stock_line(line)
            if parsed:
                sid, info = parsed
                detailed_watchlist[sid] = info

    return detailed_watchlist


def parse_stocks_lots(file_path: str = "Stocks.txt") -> list:
    """
    逐筆解析 Stocks.txt（同 parse_stocks_detailed 的欄位格式），但**保留同代號多筆**，
    供「同一檔不同時期／價位買進」的 multi-lot 逐筆停損／出場判定使用。

    回傳 list（保留檔案原始出現順序）：
      [ {"stock_id": ..., "cost": ..., "shares": ..., "stop_price": ..., "buy_date": ...}, ... ]
    """
    lots = []
    fp = _resolve_stocks_path(file_path)
    if not fp:
        return lots

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_stock_line(line)
            if parsed:
                sid, info = parsed
                lots.append({"stock_id": sid, **info})

    return lots


def _resolve_stocks_path(file_path: str):
    """解析 Stocks.txt 路徑（支援絕對／相對），找不到回傳 None。"""
    fp = file_path
    if not os.path.isabs(fp):
        fp = os.path.join(BASE_DIR, file_path)
    if not os.path.exists(fp):
        # 備用降級防呆
        alt_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if os.path.exists(alt_fp):
            fp = alt_fp
    return fp if os.path.exists(fp) else None


def _parse_stock_line(line: str):
    """解析單行 Stocks.txt；空行或註解回傳 None，否則回傳 (sid, info dict)。"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    sid = parts[0].strip()
    cost = shares = stop_price = buy_date = None
    if len(parts) >= 2:
        try:
            cost = float(parts[1].strip())
        except ValueError:
            pass
    if len(parts) >= 3:
        try:
            shares = int(float(parts[2].strip()))
        except ValueError:
            pass
    if len(parts) >= 4:
        try:
            stop_price = float(parts[3].strip())
        except ValueError:
            pass
    if len(parts) >= 5:
        _bd = parts[4].strip()
        if _bd:
            buy_date = _bd
    return sid, {"cost": cost, "shares": shares, "stop_price": stop_price, "buy_date": buy_date}


def filter_stocks_by_train_industries(df, target_col="stock_id") -> "pd.DataFrame":
    """
    根據 config.py 中的 TRAIN_INDUSTRIES 設定與 stock_categories.json 檔案，
    以及 Stocks.txt，過濾 DataFrame 中的股票。
    """
    import numpy as np
    import json
    
    # 從中央控制面板 config 載入 TRAIN_INDUSTRIES
    try:
        from config import TRAIN_INDUSTRIES
    except ImportError:
        # 降級嘗試 (適用於移位後的 scripts/ 子目錄執行)
        try:
            import sys
            PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if PARENT_DIR not in sys.path:
                sys.path.insert(0, PARENT_DIR)
            from config import TRAIN_INDUSTRIES
        except ImportError:
            print("[utils] 無法載入 config 中的 TRAIN_INDUSTRIES，跳過過濾")
            return df

    cat_path = os.path.join(BASE_DIR, "scripts", "stock_categories.json")
    if not os.path.exists(cat_path):
        print("[utils] 找不到 stock_categories.json，跳過過濾")
        return df

    with open(cat_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    allowed_stocks = set()
    for ind_name, is_enabled in TRAIN_INDUSTRIES.items():
        if is_enabled and ind_name in categories:
            allowed_stocks.update(categories[ind_name].keys())

    # 確保 Stocks.txt 裡的自選股一定保留在股票池中
    try:
        allowed_stocks.update(load_target_stocks("Stocks.txt"))
    except Exception:
        pass

    # 偵測 df[target_col] 的型態，將 allowed_stocks 進行對齊以提升效能
    if not df.empty:
        sample = df[target_col].iloc[0]
        if isinstance(sample, (int, np.integer)):
            # 轉換為整數集合
            allowed_set = {int(x) for x in allowed_stocks if x.isdigit()}
        else:
            allowed_set = {str(x) for x in allowed_stocks}
        df_filtered = df[df[target_col].isin(allowed_set)].copy()
        return df_filtered
        
    return df


def get_atr_pct_col(df) -> str:
    """回傳 ATR 百分比欄名，找不到時回傳 None。

    ATR 欄名帶週期（atr18_pct / atr23_pct…），會隨 best_factors.json 的 ATR_PERIOD 變動。
    2026-08-14 起 feature_engineering 會額外產生期間無關的正規別名 atr_pct，
    此處優先取用；舊特徵檔沒有該欄時，退回既有的 atr<N>_pct，維持向後相容。
    """
    if "atr_pct" in df.columns:
        return "atr_pct"
    legacy = sorted(c for c in df.columns if c.startswith("atr") and c.endswith("_pct"))
    return legacy[0] if legacy else None


def get_regime_label(trend_val: float, bull_trend: float, bear_trend: float) -> str:
    """根據大盤滾動均值區分市況 (Bull / Bear / Sideways)"""
    import pandas as pd
    if pd.isna(trend_val):
        return "Sideways"
    if trend_val > bull_trend:
        return "Bull"
    elif trend_val < bear_trend:
        return "Bear"
    return "Sideways"

