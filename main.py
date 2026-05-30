"""
main.py — 台灣股市量化交易系統入口
====================================================
執行順序:
  步驟 1: 多源資料下載 (TWSE 股價/籌碼 + FinMind 財報/估值)
  步驟 2: 特徵工程 (合併所有資料，輸出 features_combined.parquet)

使用方式:
  1. 在根目錄建立 FINMIND_TOKEN.txt 並貼入 Token (可選)
  2. 確認 Stocks.txt 有你要追蹤的股票代號 (每行一個)
  3. 執行 python main.py
"""

import datetime
import os

from scripts.scraper             import download_history_data
from scripts.feature_engineering import process_all_history_features

# ── 讀取股票清單 ──────────────────────────────────────

def load_stocks(file_path: str = "Stocks.txt") -> list:
    if not os.path.exists(file_path):
        print(f"[警告] 找不到 {file_path}，使用預設股票清單。")
        return ["2330", "2317", "2454"]
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    print("=" * 50)
    print("  台灣股市量化交易系統啟動")
    print("=" * 50)
    print()

    # ── 時間區間設定 ─────────────────────────────────
    start_date     = datetime.date(2024, 1, 1)    # 歷史回溯起點 (建議至少 5 年)
    end_date       = datetime.date.today()        # 自動抓取今日日期

    # ── 股票清單 ─────────────────────────────────────
    stock_list = load_stocks("Stocks.txt")
    print(f"目標股票 ({len(stock_list)} 檔): {stock_list}")
    print()

    # ════════════════════════════════════════════════
    # 步驟 1: 多源資料下載 (TWSE + FinMind)
    # ════════════════════════════════════════════════
    print("=" * 50)
    print("  步驟 1: 多源資料下載")
    print("=" * 50)
    print("  包含: 股價 / 法人籌碼 / 融資券 / 當沖 / 財報 / 本益比")
    print()
    download_history_data(start_date, end_date, target_stocks=stock_list)
    print()

    # ════════════════════════════════════════════════
    # 步驟 2: 特徵工程
    # ════════════════════════════════════════════════
    print("=" * 50)
    print("  步驟 2: 特徵工程")
    print("=" * 50)
    print("  輸出: data/features/features_combined.parquet")
    print()
    process_all_history_features(start_date, end_date)
    print()

    print("=" * 50)
    print("  所有前置資料準備任務完成！")
    print("=" * 50)