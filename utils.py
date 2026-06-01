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
