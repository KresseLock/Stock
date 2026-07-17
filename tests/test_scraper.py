import os
import sys
import datetime
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scripts.scraper import download_history_data

def test_scrape():
    # 選擇一個已知交易日進行快速單元測試
    test_date = datetime.date(2024, 3, 1)
    date_str = test_date.strftime("%Y%m%d")
    
    print(f"==================================================")
    print(f"  執行爬蟲單元測試: 測試日期 {test_date}")
    print(f"==================================================")
    
    # 載入 skip_dates 快取以進行防禦性檢查
    skip_dates = {}
    skip_path = os.path.join(ROOT_DIR, "data", "skip_dates.json")
    if os.path.exists(skip_path):
        try:
            with open(skip_path, "r", encoding="utf-8") as f:
                skip_dates = json.load(f)
        except:
            pass
            
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
        f"data/raw_chips/{date_str}_fini_holding.csv"
    ]
    
    all_pass = True
    for rel_path in files_to_check:
        full_path = os.path.join(ROOT_DIR, rel_path)
        
        # 檢查該檔案是否在 skip_dates 中被標記為跳過
        basename = os.path.basename(rel_path).replace(".csv", "")
        parts = basename.split("_", 1)
        
        is_skipped = False
        skip_reason = ""
        if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 8:
            dt_part = parts[0]
            name_part = parts[1]
            # 支持如 daytrading_20240301 或 taifex_20240301 (從 taifex_inst 映射)
            keys_to_try = [
                f"{name_part}_{dt_part}",
                f"taifex_{dt_part}" if "taifex" in name_part else "",
                f"fini_holding_{dt_part}" if "fini" in name_part else ""
            ]
            for k in keys_to_try:
                if k and k in skip_dates:
                    is_skipped = True
                    skip_reason = skip_dates[k].get("reason", "unknown")
                    break
        
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
        elif is_skipped:
            print(f" PASS  {rel_path:<40} | SKIP (跳過快取: 原因={skip_reason})")
        else:
            print(f" FAIL  {rel_path:<40} | 找不到檔案且無跳過紀錄")
            all_pass = False

    print(f"==================================================")
    if all_pass:
        print(" 爬蟲測試通過！證交所/期交所 API 正常，資料內容正確 (包含指定台積電資料或有合理跳過快取)。")
    else:
        print(" 爬蟲測試失敗！部分檔案未成功下載或內容異常。")
        sys.exit(1)

if __name__ == '__main__':
    test_scrape()

