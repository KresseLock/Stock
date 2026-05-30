import datetime
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scripts.scraper import download_history_data

def test():
    # 找一個確定有開盤的近期交易日測試 (例如 2026-05-20 週三)
    test_date = datetime.date(2026, 5, 20) 
    test_stocks = ["2330"]
    
    print("=" * 60)
    print(f"  [單日沙盒測試] 測試日期: {test_date} | 測試股票: {test_stocks[0]}")
    print("=" * 60)
    
    try:
        download_history_data(test_date, test_date, test_stocks)
        print("\n" + "=" * 60)
        print("  ✅ 測試通過！TWSE 全市場與 FinMind 個股資料皆成功寫入 CSV。")
        print("=" * 60)
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print(f"  ❌ 測試失敗！發生錯誤: {e}")
        traceback.print_exc()
        print("=" * 60)

if __name__ == "__main__":
    test()
