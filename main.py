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

# ── 抓取模式設定 ──────────────────────────────────────
# "limited" : 只抓取 Stocks.txt 以及 auto_pipeline.py (TRAIN_INDUSTRIES) 勾選的產業 (節省 FinMind Token)
# "all"     : 抓取全市場所有股票 (會消耗極大量 FinMind Token，請確定有足夠額度)
FINMIND_FETCH_MODE = "limited"

# ── 讀取股票清單 ──────────────────────────────────────

def load_stocks(mode="limited") -> list:
    """讀取股票清單"""
    if mode == "all":
        try:
            import json
            with open("stock_categories.json", "r", encoding="utf-8") as f:
                categories = json.load(f)
            all_stocks = set()
            for ind_name, stocks_dict in categories.items():
                for stock_id in stocks_dict.keys():
                    all_stocks.add(stock_id)
            return sorted(list(all_stocks))
        except Exception as e:
            print(f"[警告] 無法讀取全市場股票 ({e})，退回 limited 模式。")
            
    # limited 模式: 透過 auto_pipeline 讀取 TRAIN_INDUSTRIES 與 Stocks.txt
    try:
        import auto_pipeline
        return auto_pipeline._get_training_stocks()
    except Exception as e:
        print(f"[警告] 無法讀取產業清單 ({e})，只使用 Stocks.txt。")
        from utils import load_target_stocks
        return load_target_stocks("Stocks.txt")


if __name__ == "__main__":
    print("=" * 50)
    print("  台灣股市量化交易系統啟動")
    print("=" * 50)
    print()

    # ── 時間區間設定 ─────────────────────────────────
    start_date     = datetime.date(2020, 1, 1)    # 歷史回溯起點 (建議至少 5 年)
    end_date       = datetime.date.today()        # 自動抓取今日日期

    # ── 股票清單 ─────────────────────────────────────
    stock_list = load_stocks(FINMIND_FETCH_MODE)
    if FINMIND_FETCH_MODE == "all":
        print(f"下載目標股票 (全市場模式): 共 {len(stock_list)} 檔")
    else:
        if len(stock_list) <= 20:
            print(f"下載目標股票 (限定模式, {len(stock_list)} 檔): {stock_list}")
        else:
            print(f"下載目標股票 (限定模式, {len(stock_list)} 檔): {stock_list[:10]} ... 等")
        print(f"  (來源: Stocks.txt 以及 auto_pipeline.py 中設定為 True 的產業)")
    print()

    try:
        from scripts.scraper import FinMindLimitExceeded
    except ImportError:
        class FinMindLimitExceeded(Exception):
            pass

    # ════════════════════════════════════════════════
    # 步驟 1: 多源資料下載 (TWSE + FinMind)
    # ════════════════════════════════════════════════
    print("=" * 50)
    print("  步驟 1: 多源資料下載")
    print("=" * 50)
    print("  包含: 股價 / 法人籌碼 / 融資券 / 當沖 / 財報 / 本益比")
    print()
    try:
        download_history_data(start_date, end_date, target_stocks=stock_list)
    except FinMindLimitExceeded:
        print("\n[提示] 偵測到 FinMind API 額度用盡，本階段先結束並回報狀態碼 99 以跳過。")
        import sys
        sys.exit(99)
    print()

    # ════════════════════════════════════════════════
    # 步驟 2: 特徵工程 (已停用)
    # 說明: 每日特徵重建將統一由 auto_pipeline.py 讀取最佳參數進行，
    #       此處已註解以避免重複計算，大幅提升每日更新數據的效率。
    # ════════════════════════════════════════════════
    # print("=" * 50)
    # print("  步驟 2: 特徵工程")
    # print("=" * 50)
    # print("  輸出: data/features/features_combined.parquet")
    # print()
    # process_all_history_features(start_date, end_date, override_target_stocks=stock_list)
    # print()

    print("=" * 50)
    print("  所有前置資料準備任務完成！")
    print("=" * 50)