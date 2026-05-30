import os
import sys
from datetime import datetime, timedelta
from scripts.scraper import download_history_data

def test_scrape():
    stocks = ['2049', '4931']
    print(f"Testing scraper for stocks: {stocks}")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2000)
    download_history_data(start_date, end_date, target_stocks=stocks)
    
    # Check if files exist
    for stock in stocks:
        file_path = f"data/history/{stock}.csv"
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"File {file_path} exists, size: {size} bytes")
            # print first few lines
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"  Rows count: {len(lines)}")
                print(f"  Head: {lines[0].strip()} | {lines[1].strip() if len(lines) > 1 else ''}")
        else:
            print(f"File {file_path} DOES NOT EXIST.")

if __name__ == '__main__':
    test_scrape()
