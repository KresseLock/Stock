import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        # 降級嘗試：如果是相對路徑且調用自 scripts/ 子目錄，往上一層尋找
        parent_fp = os.path.join(BASE_DIR, "..", file_path)
        if os.path.exists(parent_fp):
            fp = parent_fp
            
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
    支援三種格式：
      格式 A (僅代號)    : 2330
      格式 B (含成本)    : 2330,950.0
      格式 C (成本與股數) : 2330,950.0,1000

    回傳 dict: { stock_id: {"cost": buy_cost_or_None, "shares": shares_or_None} }
    """
    detailed_watchlist = {}
    
    # 支援絕對與相對路徑
    fp = file_path
    if not os.path.isabs(fp):
        fp = os.path.join(BASE_DIR, file_path)
        
    if not os.path.exists(fp):
        # 降級嘗試：往上一層尋找
        parent_fp = os.path.join(BASE_DIR, "..", file_path)
        if os.path.exists(parent_fp):
            fp = parent_fp
            
    if not os.path.exists(fp):
        return detailed_watchlist

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            sid = parts[0].strip()
            cost = None
            shares = None
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
            detailed_watchlist[sid] = {"cost": cost, "shares": shares}

    return detailed_watchlist


def filter_stocks_by_train_industries(df, target_col="stock_id") -> "pd.DataFrame":
    """
    根據 train.py 中的 TRAIN_INDUSTRIES 設定與 stock_categories.json 檔案，
    以及 Stocks.txt，過濾 DataFrame 中的股票。
    """
    import numpy as np
    import json
    
    # 延遲載入 train 模組的 TRAIN_INDUSTRIES 避免循環引用
    try:
        from train import TRAIN_INDUSTRIES
    except ImportError:
        print("[utils] 無法載入 train 中的 TRAIN_INDUSTRIES，跳過過濾")
        return df

    cat_path = os.path.join(BASE_DIR, "stock_categories.json")
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

