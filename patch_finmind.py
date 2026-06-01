# -*- coding: utf-8 -*-
"""
patch_finmind.py — FinMind 基本面強制更新與完整性檢查工具 (一鍵回補)
================================================================
本腳本提供以下三大功能：
  1. 獨立清空快取：自動將指定股票從 `no_finmind_data.json` 與 `missing_fm_datasets.json` 排除。
  2. 繞過 12 小時限制：修改本機 CSV 檔案時間戳記，強制發送 API 進行最精準的「缺漏比對」。
  3. 增量去重合併：如果發現本機 CSV 有少天數的缺漏，自動向 API 下載並與舊資料完美合併去重。
"""

import os
import sys
import time
import json
import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from scripts.scraper import _crawl_fm_dataset, DATA_DIR, _polite_sleep
from utils import load_target_stocks

# 快取檔案路徑
NO_DATA_PATH = os.path.join(DATA_DIR, "no_finmind_data.json")
MISSING_FM_PATH = os.path.join(DATA_DIR, "missing_fm_datasets.json")


def clean_skip_caches(stock_ids: list):
    """
    將指定股票在快取 JSON (no_finmind_data.json, missing_fm_datasets.json) 中的略過標記清除，
    使爬蟲重新將其納入檢查。
    """
    print("-" * 75)
    print("  [步驟 1] 清理 FinMind 略過快取標記...")
    print("-" * 75)

    # 1. 清理 no_finmind_data.json (全無資料快取)
    if os.path.exists(NO_DATA_PATH):
        try:
            with open(NO_DATA_PATH, "r", encoding="utf-8") as f:
                no_data = json.load(f)
            cleared = 0
            for sid in stock_ids:
                if sid in no_data:
                    del no_data[sid]
                    cleared += 1
            if cleared > 0:
                with open(NO_DATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(no_data, f, ensure_ascii=False, indent=2)
                print(f"    [v] no_finmind_data.json: 已清除 {cleared} 檔股票的略過標記")
            else:
                print("    [v] no_finmind_data.json: 無需清理")
        except Exception as e:
            print(f"    [警告] 清理 no_finmind_data.json 失敗: {e}")
    else:
        print("    [v] no_finmind_data.json 不存在，無需清理")

    # 2. 清理 missing_fm_datasets.json (局部缺失快取)
    if os.path.exists(MISSING_FM_PATH):
        try:
            with open(MISSING_FM_PATH, "r", encoding="utf-8") as f:
                missing = json.load(f)
            keys_to_del = []
            for k in list(missing.keys()):
                # key 格式為 {stock_id}_{dataset_name}
                for sid in stock_ids:
                    if k.startswith(f"{sid}_"):
                        keys_to_del.append(k)
            if keys_to_del:
                for k in keys_to_del:
                    del missing[k]
                with open(MISSING_FM_PATH, "w", encoding="utf-8") as f:
                    json.dump(missing, f, ensure_ascii=False, indent=2)
                print(f"    [v] missing_fm_datasets.json: 已清除 {len(keys_to_del)} 個略過項目")
            else:
                print("    [v] missing_fm_datasets.json: 無需清理")
        except Exception as e:
            print(f"    [警告] 清理 missing_fm_datasets.json 失敗: {e}")
    else:
        print("    [v] missing_fm_datasets.json 不存在，無需清理")


def force_check_and_patch(stock_ids: list, start_str: str, end_str: str):
    """
    強制對指定股票進行基本面完整性比對與補丁。
    """
    # 1. 先清理略過快取 JSON，釋放封鎖
    clean_skip_caches(stock_ids)
    
    print("\n" + "-" * 75)
    print("  [步驟 2] 破除 12小時快取，進行精準「缺漏比對與增量回補」...")
    print("-" * 75)
    
    # 建立一個臨時的空 missing_cache，確保本次任務中不會被任何快取干擾
    temp_missing_cache = {}
    
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, stock_id in enumerate(stock_ids, 1):
        print(f"\n [{idx}/{len(stock_ids)}] 強制檢查 FinMind {stock_id}...")
        results = []
        
        datasets = [
            ("營收",   "TaiwanStockMonthRevenue",       f"{stock_id}_monthly_revenue.csv"),
            ("損益表", "TaiwanStockFinancialStatements", f"{stock_id}_financial_stmt.csv"),
            ("資產表", "TaiwanStockBalanceSheet",        f"{stock_id}_balance_sheet.csv"),
            ("現金流", "TaiwanStockCashFlowsStatement",  f"{stock_id}_cashflow.csv"),
            ("股利",   "TaiwanStockDividend",            f"{stock_id}_dividend.csv"),
        ]
        
        for label, dataset, fname in datasets:
            path = os.path.join(DATA_DIR, "raw_financial", fname)
            
            #  破除 12 小時新鮮快取：如果檔案存在，將修改時間往回拉 24 小時，強制發送 API 比對
            if os.path.exists(path):
                old_time = time.time() - 86400
                os.utime(path, (old_time, old_time))
                
            print(f"    - 檢查 {label:<4}...", end=" ", flush=True)
            # 呼叫原始爬蟲，傳入完整的 6 個參數 (修復了原版 patch_finmind.py 的 Bug!)
            res = _crawl_fm_dataset(dataset, stock_id, start_str, end_str, path, temp_missing_cache)
            results.append((label, res))
            
            if res == "skipped":
                print("已是最新 (無缺漏)")
                skipped_count += 1
            elif res is True:
                print("更新成功 (已下載缺漏並合併去重)")
                success_count += 1
            elif res is False:
                print("更新失敗 (網路或 Token 限額)")
                failed_count += 1
            elif res is None:
                print("無資料")
                failed_count += 1
                
            if res != "skipped":
                _polite_sleep(1, 2)

    print("\n" + "=" * 75)
    print("  強制基本面補丁完成！")
    print(f"  總檢查細項: {success_count + skipped_count + failed_count} 項")
    print(f"  [v] 更新成功 (補齊缺漏): {success_count} 項")
    print(f"  [v] 已是最新 (無須補齊): {skipped_count} 項")
    if failed_count > 0:
        print(f"   失敗或無資料: {failed_count} 項")
    print("=" * 75)


def main():
    print("=" * 75)
    print("  台灣股市量化交易系統 ─ FinMind 基本面強制回補與缺漏比對工具")
    print("=" * 75)

    # 預設讀取 Stocks.txt 內的所有自選與持倉股進行全面強制檢查
    stock_list = load_target_stocks("Stocks.txt")
    if not stock_list:
        print("[錯誤] Stocks.txt 為空或不存在！")
        return

    print(f"  偵測到 Stocks.txt 目標股票 (共 {len(stock_list)} 檔): {stock_list}")
    
    # 時間區間設定：強制從 2019 年開始比對到今天，確保完整無缺漏
    start_str = "2019-01-01"
    end_str = datetime.date.today().strftime("%Y-%m-%d")
    
    force_check_and_patch(stock_list, start_str, end_str)


if __name__ == "__main__":
    main()
