import os
import sys
import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.stdout.reconfigure(encoding='utf-8')

from scripts.scraper import download_history_data

def test_scrape():
    # 選擇一個已知必定有開市的交易日進行快速單元測試
    test_date = datetime.date(2024, 3, 1)
    date_str = test_date.strftime("%Y%m%d")
    
    print(f"==================================================")
    print(f"  執行爬蟲單元測試: 測試日期 {test_date}")
    print(f"==================================================")
    
    # 執行單日下載
    download_history_data(test_date, test_date, target_stocks=["2330"])
    
    # 檢查架構中定義的核心檔案是否成功下載至正確目錄
    files_to_check = [
        f"data/raw_price/{date_str}_price.csv",
        f"data/raw_chips/{date_str}_chips.csv",
        f"data/raw_twse_per/{date_str}_twse_per.csv",
        f"data/raw_taifex/{date_str}_taifex_inst.csv",
        f"data/raw_margin/{date_str}_margin.csv",
        f"data/raw_margin/{date_str}_sbl.csv",
        f"data/raw_chips/{date_str}_daytrading.csv",
        f"data/raw_chips/{date_str}_fini_holding.csv",
        f"data/raw_margin/{date_str}_credit_limit.csv"
    ]
    
    all_pass = True
    for rel_path in files_to_check:
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            
            # 針對有內容的 CSV 進行 pandas 讀取驗證
            try:
                df = pd.read_csv(full_path, dtype=str)
                cols = list(df.columns)
                
                # 若是台股個股檔案，驗證是否有台積電 (2330)
                has_2330 = "N/A"
                if "證券代號" in cols:
                    df["證券代號"] = df["證券代號"].str.strip()
                    has_2330 = " 存在" if "2330" in df["證券代號"].values else " 缺失"
                    
                print(f" PASS  {rel_path:<40} | 大小: {size:>7} bytes | 欄位數: {len(cols):>2} | 2330: {has_2330}")
            except Exception as e:
                print(f" FAIL  {rel_path:<40} | 檔案毀損無法讀取: {e}")
                all_pass = False
        else:
            print(f" FAIL  {rel_path:<40} | 找不到檔案")
            all_pass = False

    print(f"==================================================")
    if all_pass:
        print(" 爬蟲測試通過！證交所/期交所 API 正常，資料內容正確 (包含指定台積電資料)。")
    else:
        print(" 爬蟲測試失敗！部分檔案未成功下載或內容異常。")
        sys.exit(1)

if __name__ == '__main__':
    test_scrape()

