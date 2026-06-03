import os
import glob
import pandas as pd
import json

# 專案路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FINANCIAL_DIR = os.path.join(DATA_DIR, "raw_financial")
CATEGORIES_PATH = os.path.join(BASE_DIR, "stock_categories.json")
NO_DATA_PATH = os.path.join(DATA_DIR, "no_finmind_data.json")

# 定義每個資料集應該具備的「必備欄位」
DATASET_SPECS = {
    "monthly_revenue": ["date", "revenue"],
    "financial_stmt":  ["date", "type", "value"],
    "balance_sheet":   ["date", "type", "value"],
    "cashflow":        ["date", "type", "value"],
    "dividend":        ["date", "CashEarningsDistribution"]
}

def load_etfs():
    try:
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            cats = json.load(f)
        return set(cats.get("ETF", {}).keys())
    except Exception:
        return set()

def load_no_data_stocks():
    if os.path.exists(NO_DATA_PATH):
        try:
            with open(NO_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 只取 confirmed 的股票
            return {k for k, v in data.items() if isinstance(v, dict) and v.get("status") == "confirmed"}
        except Exception:
            pass
    return set()

def check_and_clean_finmind_data():
    if not os.path.exists(FINANCIAL_DIR):
        print(f"資料夾 {FINANCIAL_DIR} 不存在。")
        return

    etfs = load_etfs()
    no_data_stocks = load_no_data_stocks()
    
    # 掃描所有財報檔案
    all_files = glob.glob(os.path.join(FINANCIAL_DIR, "*.csv"))
    if not all_files:
        print("沒有找到任何財報 CSV 檔案。")
        return

    deleted_count = 0
    checked_count = 0

    print("==================================================")
    print("  啟動 FinMind 歷史資料完整性檢查與修復工具")
    print("==================================================")
    print("邏輯說明：")
    print("1. 檢查每個 CSV 是否擁有正確的核心欄位。")
    print("2. 檢查檔案是否為空。")
    print("3. 若發現欄位缺失或破損，直接刪除該檔案，讓 scraper.py 之後自動重抓。")
    print("==================================================\n")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        # 解析 stock_id 和 dataset_name
        # 檔名格式: {stock_id}_{dataset_name}.csv
        parts = filename.replace(".csv", "").split("_", 1)
        if len(parts) != 2:
            continue
            
        stock_id, dataset_name = parts
        
        # ETF 或已知無資料股票，正常來說不會有檔案，若有可以保留或檢查
        if stock_id in etfs:
            continue

        if dataset_name not in DATASET_SPECS:
            continue

        required_cols = DATASET_SPECS[dataset_name]
        checked_count += 1
        
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            
            # 檢查 1: 檔案是否為空
            if df.empty:
                print(f"[異常] {filename} 裡面沒有任何數據 (空表) -> 刪除檔案！")
                os.remove(file_path)
                deleted_count += 1
                continue
                
            # 檢查 2: 核心欄位是否存在
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"[異常] {filename} 缺少必要欄位 {missing_cols} -> 刪除檔案！")
                os.remove(file_path)
                deleted_count += 1
                continue
                
        except Exception as e:
            # 檢查 3: 檔案損毀無法讀取
            print(f"[損毀] {filename} 無法讀取 ({e}) -> 刪除檔案！")
            os.remove(file_path)
            deleted_count += 1
            continue

    print("\n==================================================")
    print(f"檢查完畢！")
    print(f"共檢查 {checked_count} 個 FinMind 財報檔案。")
    print(f"共發現並刪除 {deleted_count} 個異常/缺失欄位的檔案。")
    if deleted_count > 0:
        print("提示: 請在稍後執行 `python main.py`，系統將自動從頭補抓這些被刪除的股票資料！")
    else:
        print("恭喜！所有現存的 FinMind 資料欄位皆完好無損，不需重抓！")
    print("==================================================")

def check_and_clean_twse_data():
    twse_dirs = ["raw_price", "raw_chips", "raw_margin", "raw_twse_per", "raw_taifex"]
    deleted_count = 0
    checked_count = 0

    print("==================================================")
    print("  啟動 證交所/期交所 歷史資料完整性檢查與修復工具")
    print("==================================================")
    
    for d in twse_dirs:
        dir_path = os.path.join(DATA_DIR, d)
        if not os.path.exists(dir_path):
            continue
            
        all_files = glob.glob(os.path.join(dir_path, "*.csv"))
        for file_path in all_files:
            filename = os.path.basename(file_path)
            checked_count += 1
            
            try:
                size = os.path.getsize(file_path)
                if size <= 3:
                    print(f"[異常] {d}/{filename} 檔案大小極小(<=3 bytes) -> 刪除舊版空檔標記！")
                    os.remove(file_path)
                    deleted_count += 1
                    continue
                    
                df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
                
                if df.empty:
                    print(f"[異常] {d}/{filename} 裡面沒有任何數據 (空表) -> 刪除舊版空檔標記！")
                    os.remove(file_path)
                    deleted_count += 1
                    continue
                    
                # 簡單防呆：判斷是否是不小心存成了 HTML 錯誤頁面 (通常欄位數量極少且沒有預期的證券代號)
                cols = list(df.columns)
                if len(cols) == 1 and ("html" in cols[0].lower() or "很抱歉" in cols[0]):
                    print(f"[異常] {d}/{filename} 疑似存成錯誤網頁內容 -> 刪除檔案！")
                    os.remove(file_path)
                    deleted_count += 1
                    continue
                    
            except Exception as e:
                if os.path.getsize(file_path) <= 3:
                    print(f"[異常] {d}/{filename} 檔案大小極小無法讀取 -> 刪除舊版空檔標記！")
                    os.remove(file_path)
                    deleted_count += 1
                    continue
                else:
                    print(f"[損毀] {d}/{filename} 無法讀取 ({e}) -> 刪除檔案！")
                    os.remove(file_path)
                    deleted_count += 1
                    continue
                
    print("\n==================================================")
    print(f"證交所資料檢查完畢！")
    print(f"共檢查 {checked_count} 個官方 CSV 檔案。")
    print(f"共發現並刪除 {deleted_count} 個異常/舊版空檔檔案。")
    print("==================================================\n")

def check_price_anomalies():
    """
    掃描 raw_price 目錄，檢查 0050 的收盤價。
    因為台股有 10% 漲跌幅限制，若 0050 單日跳空超過 15%，
    絕對是證交所 API 吐回了舊日期的假資料（通常發生在假補班或颱風天）。
    這時我們要把這天的所有檔案刪除，並加入 skip_dates.json 防止重抓。
    """
    print("==================================================")
    print("  啟動 極端價格異常 (幽靈資料) 偵測工具")
    print("==================================================")
    
    price_dir = os.path.join(DATA_DIR, "raw_price")
    if not os.path.exists(price_dir):
        return
        
    all_files = sorted(glob.glob(os.path.join(price_dir, "*_price.csv")))
    
    # 讀取 0050 歷史價格
    history = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        date_str = filename.split("_")[0]
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
            # 找到 0050
            row = df[df.iloc[:, 0] == "0050"]
            if not row.empty:
                # 收盤價通常在第 8 或 9 欄 (收盤價)
                cols = list(df.columns)
                close_col = next((c for c in cols if "收盤價" in c), None)
                if close_col:
                    close_price = float(row[close_col].values[0].replace(",", ""))
                    history.append({"date": date_str, "file": file_path, "price": close_price})
        except Exception:
            continue
            
    # 讀取現有的 skip_dates 與 failed_dates 以便清除紀錄
    skip_dates_path = os.path.join(DATA_DIR, "skip_dates.json")
    fail_log_path = os.path.join(DATA_DIR, "failed_dates.json")
    
    skip_dates = {}
    if os.path.exists(skip_dates_path):
        try:
            with open(skip_dates_path, "r", encoding="utf-8") as f:
                skip_dates = json.load(f)
        except Exception:
            pass
            
    fail_log = {}
    if os.path.exists(fail_log_path):
        try:
            with open(fail_log_path, "r", encoding="utf-8") as f:
                fail_log = json.load(f)
        except Exception:
            pass

    deleted_count = 0

    for i in range(1, len(history)):
        prev = history[i-1]
        curr = history[i]
        pct_change = abs((curr["price"] - prev["price"]) / prev["price"])
        
        if pct_change > 0.15:
            bad_date = curr["date"]
            print(f"[幽靈資料] {bad_date} 發現 0050 異常跳空 (前日:{prev['price']} -> 本日:{curr['price']})！")
            
            # 刪除這天的所有相關檔案
            for d in ["raw_price", "raw_chips", "raw_margin", "raw_twse_per", "raw_taifex"]:
                bad_file_pattern = os.path.join(DATA_DIR, d, f"{bad_date}_*.csv")
                for bad_f in glob.glob(bad_file_pattern):
                    os.remove(bad_f)
                    print(f"  - 刪除假檔案: {os.path.basename(bad_f)}")
            
            # 從黑名單中移除，確保 scraper.py 下次會勇敢去抓
            removed_from_cache = False
            if bad_date in skip_dates:
                del skip_dates[bad_date]
                removed_from_cache = True
            if bad_date in fail_log:
                del fail_log[bad_date]
                removed_from_cache = True
            
            if removed_from_cache:
                print(f"  - 已從 JSON 快取中移除 {bad_date}，確保下次重新抓取。")

            deleted_count += 1
            
            # 把 curr 的 price 改回 prev，避免下一次比較又觸發異常 (假跌回)
            curr["price"] = prev["price"]

    if deleted_count > 0:
        # 回寫 JSON 檔案
        with open(skip_dates_path, "w", encoding="utf-8") as f:
            json.dump(skip_dates, f, indent=4, ensure_ascii=False)
        with open(fail_log_path, "w", encoding="utf-8") as f:
            json.dump(fail_log, f, indent=4, ensure_ascii=False)
            
        print(f"\n共清除 {deleted_count} 天的幽靈假資料，請在下次執行 main.py 時重新補抓！")
        print("提示: 幽靈資料已清除，未來補抓完成後，請記得執行 run_feature_engineering.py 重算乾淨的特徵！")
    else:
        print("未發現任何幽靈假資料，資料庫非常健康！")
    print("==================================================\n")

if __name__ == "__main__":
    check_and_clean_finmind_data()
    check_and_clean_twse_data()
    check_price_anomalies()
