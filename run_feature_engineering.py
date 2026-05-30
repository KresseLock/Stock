"""
run_feature_engineering.py — 獨立特徵工程提取工具
====================================================
使用場景:
  當您修改了 Stocks.txt 裡面的股票清單，但「不需要」重新從網路抓取資料時，
  執行本程式將直接讀取本地 data/ 下的現有 raw data，
  重新為 Stocks.txt 中的股票進行特徵工程計算，並更新特徵矩陣。

使用方式:
  1. 修改根目錄的 Stocks.txt (新增、刪除或調整股票代號)。
  2. 執行本程式: python run_feature_engineering.py
  3. 產出特徵矩陣: data/features/features_combined.parquet (供 LightGBM 使用)
"""

import datetime
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 將當前目錄加入 Python 搜尋路徑以利 import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from scripts.feature_engineering import process_all_history_features, load_target_stocks

def check_local_data_integrity(target_stocks: list):
    """
    檢查目標股票在本地是否已有下載好的 FinMind 資料。
    如果沒有，印出友善提示，告知用戶可能需要先運行一次 main.py。
    """
    data_dir = os.path.join(BASE_DIR, "data")
    missing_finmind_stocks = []
    
    for stock in target_stocks:
        # 檢查該個股是否有月營收或財報
        rev_path = os.path.join(data_dir, "raw_financial", f"{stock}_monthly_revenue.csv")
        stmt_path = os.path.join(data_dir, "raw_financial", f"{stock}_financial_stmt.csv")
        
        if not os.path.exists(rev_path) or not os.path.exists(stmt_path):
            missing_finmind_stocks.append(stock)
            
    if missing_finmind_stocks:
        print("=" * 60)
        print("  [注意] 發現新加入的股票在本地無 FinMind 歷史資料：")
        print(f"  缺少的股票: {missing_finmind_stocks}")
        print("  提示: 若要完整提取這些股票的基本面特徵，建議先執行 `python main.py` 下載資料。")
        print("  (若繼續執行，程式仍會運作，但這些股票的月營收、財報特徵將會為空值)")
        print("=" * 60)
        print()

if __name__ == "__main__":
    print("=" * 50)
    print("  獨立特徵工程提取工具啟動 (僅使用本地現有數據)")
    print("=" * 50)
    print()

    # 1. 載入當前 Stocks.txt 清單
    target_stocks = load_target_stocks("Stocks.txt")
    print(f"當前目標股票數量: {len(target_stocks)} 檔")
    print(f"股票清單: {target_stocks}")
    print()

    # 2. 檢查本地資料完整性
    check_local_data_integrity(target_stocks)

    # 3. 設定分析時間區間 (與 main.py 保持一致)
    start_date = datetime.date(2024, 1, 1)    # 歷史回溯起點
    end_date = datetime.date.today()        # 自動抓取今日日期

    print(f"分析時間區間: {start_date} ~ {end_date}")
    print("開始重新計算特徵值...")
    print("-" * 50)

    # 4. 呼叫特徵工程主函式
    try:
        process_all_history_features(start_date, end_date)
        print("-" * 50)
        print("  [完成] 特徵值重新提取完成！")
        print("  您現在可以使用 data/features/features_combined.parquet 進行 LightGBM 訓練與推理。")
    except Exception as e:
        print()
        print(f"[錯誤] 特徵提取過程中發生錯誤: {e}")
        print("請確認 data/ 目錄下是否存有對應日期的 TWSE 價格/籌碼日報 CSV。")
        
    print("=" * 50)
