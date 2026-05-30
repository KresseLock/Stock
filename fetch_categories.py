"""
fetch_categories.py — 自動抓取台股全市場分類與代碼
======================================================
從 FinMind 抓取 TaiwanStockInfo，將全台股依據產業類別分類，
並將結果統一儲存到單一 JSON 檔案中，避免產生太多 txt 檔。
"""
import os
import sys
import json
import requests
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(BASE_DIR, "stock_categories.json")

def main():
    print("=" * 60)
    print("  開始下載台股全市場產業分類資料 (FinMind)")
    print("=" * 60)
    
    url = "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo"
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        if data.get("msg") != "success":
            print(f"API 回應錯誤: {data.get('msg')}")
            return
            
        stock_list = data.get("data", [])
        print(f"  成功取得 {len(stock_list)} 筆資料，開始進行分類整理...")
        
        # 使用 defaultdict 來依照 industry_category 分類
        categories = defaultdict(dict)
        
        for item in stock_list:
            stock_id = item.get("stock_id", "")
            stock_name = item.get("stock_name", "")
            industry = item.get("industry_category", "未分類")
            
            # 只保留英數混合的股票代碼 (一般股4碼，ETF/特別股可能到5-6碼，例如 00403A, 00981A)
            if stock_id.isalnum() and 4 <= len(stock_id) <= 6:
                categories[industry][stock_id] = stock_name
                
        # 移除空的或是無效的分類
        if "" in categories:
            del categories[""]
            
        # 依據代碼排序
        sorted_categories = {}
        for ind, stocks in categories.items():
            sorted_categories[ind] = dict(sorted(stocks.items(), key=lambda x: x[0]))
            
        # 存成 JSON 檔案
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(sorted_categories, f, ensure_ascii=False, indent=4)
            
        print("-" * 60)
        print(f"  [完成] 產業分類已儲存至: {OUTPUT_JSON}")
        print(f"  共分出 {len(sorted_categories)} 個產業類別。")
        print("  包含的產業有：")
        
        # 列出前幾個產業作為預覽
        industry_names = list(sorted_categories.keys())
        for ind in industry_names[:10]:
            print(f"    - {ind} ({len(sorted_categories[ind])} 檔)")
        if len(industry_names) > 10:
            print(f"    ... 等共 {len(industry_names)} 個類別。")
        print("=" * 60)

    except Exception as e:
        print(f"[錯誤] 下載分類資料失敗: {e}")

if __name__ == "__main__":
    main()
