import os
import glob
import pandas as pd
import json

# 專案路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FINANCIAL_DIR = os.path.join(DATA_DIR, "raw_financial")
CATEGORIES_PATH = os.path.join(BASE_DIR, "..", "stock_categories.json")
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

if __name__ == "__main__":
    check_and_clean_finmind_data()
