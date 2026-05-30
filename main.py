import datetime
from scripts.scraper import download_history_data
from scripts.feature_engineering import process_all_history_features

if __name__ == "__main__":
    print("=== 台灣股市量化交易系統啟動 ===")
    
    # 定義回溯時間區間
    start_date = datetime.date(2025, 5, 1)
    end_date = datetime.date(2026, 5, 29)
    
    # 【步驟 1】下載原始資料
    # 自動檢查是否已有檔案，避免重複下載浪費時間
    download_history_data(start_date, end_date)
    
    print("-" * 40)
    
    # 【步驟 2】特徵工程
    # 自動讀取 Stocks.txt 並打包成 features_combined.parquet
    process_all_history_features(start_date, end_date)
    
    print("=== 前置資料準備任務完美結束 ===")