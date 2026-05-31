# -*- coding: utf-8 -*-
"""
inference.py — LightGBM 模型多天期推理 (Day 1 ~ Day 3)
====================================================
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import lightgbm as lgb
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def get_stock_name(stock_id, date_str):
    price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
    if os.path.exists(price_file):
        try:
            df_price = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
            df_price["證券代號"] = df_price["證券代號"].str.strip()
            df_price["證券名稱"] = df_price["證券名稱"].str.strip()
            return df_price.set_index("證券代號")["證券名稱"].get(stock_id, "")
        except Exception:
            return ""
    return ""

def main():
    print("=" * 75)
    print("  啟動 LightGBM 多天期預測推理 (未來 3 天)")
    print("=" * 75)
    
    if not os.path.exists(DATA_PATH):
        print(f"[錯誤] 找不到特徵檔: {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    latest_date = df["date"].max()
    print(f"  最新資料日期: {latest_date.date()}")

    # ── 只對 Stocks.txt 的自選股進行預測 ──────────────────
    # 訓練時可能使用了數百家公司，但排行榜只顯示您關注的股票
    watchlist_path = os.path.join(BASE_DIR, "Stocks.txt")
    if os.path.exists(watchlist_path):
        with open(watchlist_path, "r", encoding="utf-8") as f:
            watchlist = [line.strip() for line in f if line.strip()]
    else:
        watchlist = sorted(df["stock_id"].unique().tolist())
    print(f"  自選股清單  : {watchlist} ({len(watchlist)} 檔)")

    missing_in_latest = set(watchlist) - set(df[(df["date"] == latest_date)]["stock_id"])
    if missing_in_latest:
        print(f"  [警告] 以下股票在最新日期無資料，跳過預測: {list(missing_in_latest)}")
        
    # ── 對「全市場」所有股票進行預測 (用於尋找最強的 Top-10) ──────────────────
    df_latest_market = df[df["date"] == latest_date].copy()
    if df_latest_market.empty:
        print("[錯誤] 最新日期無任何股票資料。")
        return

    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    ignore_cols = ["stock_id", "date"] + target_cols

    # 優先讀取訓練時固定的特徵欄位清單，確保與模型完全一致
    feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        print(f"  特徵欄位來源: models/feature_cols.json ({len(feature_cols)} 欄)")
    else:
        # fallback：動態推導 (僅在 feature_cols.json 不存在時使用)
        print("  [警告] 找不到 feature_cols.json，改用動態推導特徵欄位 (可能與訓練時不一致)")
        numeric_cols = df_latest.select_dtypes(include=[np.number, bool]).columns
        feature_cols = [c for c in numeric_cols if c not in ignore_cols]

    # 以固定欄位 reindex，缺少的欄位補 NaN（LightGBM 能處理），並統一轉為 float32
    X_latest_market = df_latest_market.reindex(columns=feature_cols).astype(np.float32)
    
    results_market = df_latest_market[["stock_id"]].copy()
    if "close" in df_latest_market.columns:
        results_market["close"] = df_latest_market["close"]
    else:
        results_market["close"] = 0.0
        
    # 逐一載入 3 天的模型並預測
    for days in [1, 2, 3]:
        model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days}.txt")
        if not os.path.exists(model_path):
            print(f"[錯誤] 找不到模型 {model_path}，請重新執行 train.py")
            return
        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X_latest_market)
        if len(preds.shape) == 2 and preds.shape[1] == 3:
            prob_strong = preds[:, 2] # Class 2 (強勢) 機率
            prob_weak = preds[:, 0]   # Class 0 (弱勢) 機率
        else:
            prob_strong = preds[:, 1] if len(preds.shape) == 2 else preds
            prob_weak = 1 - prob_strong
            
        net_score = prob_strong - prob_weak
        
        results_market[f"Day{days}_net"] = net_score * 100
        results_market[f"Day{days}_weak"] = prob_weak * 100
        
    # 將全市場預測結果過濾出「自選股」清單
    results = results_market[results_market["stock_id"].isin(watchlist)].copy()
    results = results.sort_values(by="Day3_net", ascending=False).reset_index(drop=True)
    
    # ── 取出全市場最強 Top 10 (以 Day 1 為主) ──
    results_top10 = results_market.sort_values(by="Day1_net", ascending=False).head(10).reset_index(drop=True)
    
    factors_file = os.path.join(BASE_DIR, "best_factors.json")
    factors_info = "  [未找到最佳化因子檔，使用系統預設因子參數]"
    if os.path.exists(factors_file):
        try:
            with open(factors_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = data.get("best_params_for_run_feature_engineering", {})
            opt_date = data.get("optimized_at", "未知")
            lines = [f"  [使用的因子參數 (最佳化時間: {opt_date})]"]
            for k, v in params.items():
                lines.append(f"    {k:<20} = {v}")
            factors_info = "\n".join(lines)
        except Exception as e:
            print(f"  [警告] 讀取 best_factors.json 失敗: {e}")

    output_lines = []
    # 讀取股票中文名稱對照表
    stock_names = {}
    price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{latest_date.strftime('%Y%m%d')}_price.csv")
    if os.path.exists(price_file):
        try:
            df_price = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
            df_price["證券代號"] = df_price["證券代號"].str.strip()
            df_price["證券名稱"] = df_price["證券名稱"].str.strip()
            stock_names = df_price.set_index("證券代號")["證券名稱"].to_dict()
        except Exception:
            pass

    output_lines.append("\n" + "=" * 80)
    output_lines.append(f"  [結果] 未來三天多空綜合分數預測 (預測基準日: {latest_date.date()})")
    output_lines.append("=" * 80)
    output_lines.append(f"{'排名':<3} | {'股票 (代號+名稱)':<12} | {'收盤價':<6} | {'D1多空分數':<9} | {'D2多空分數':<9} | {'D3多空分數':<9} | {'趨勢分析'}")
    output_lines.append("-" * 80)
    
    for i, row in enumerate(results.itertuples(), start=1):
        stock_id = row.stock_id
        close_price = row.close
        d1, d2, d3 = row.Day1_net, row.Day2_net, row.Day3_net
        w1, w2, w3 = row.Day1_weak, row.Day2_weak, row.Day3_weak
        
        d1_s = f"{d1:+.1f}%"
        d2_s = f"{d2:+.1f}%"
        d3_s = f"{d3:+.1f}%"
        
        # 簡單趨勢判定 (基於多空淨分數 net_score)
        # d1, d2, d3 現在是淨分數 (-100 ~ +100)
        if d1 > 12 and d2 > 8: trend = "強勢多頭 (發動中)"
        elif d1 > 5 and d2 > 0: trend = "偏多 (醞釀中)"
        elif d1 < -15 and d2 < -15: trend = "極度弱勢 (空頭)"
        elif d1 < -5 and d2 < -5: trend = "偏空 (轉弱)"
        elif d1 > 8 and d3 < 0: trend = "短多長空 (當沖佳)"
        else: trend = "震盪整理"
            
        cname = stock_names.get(stock_id, "")
        display_name = f"{stock_id} {cname}"
        
        # 由於中文字元對齊問題，使用手動填充全形/半形空白或直接保留足夠寬度
        # 這裡為求簡單，給予固定長度再配上格式化
        output_lines.append(f" {i:<3} | {display_name:<16} | {close_price:>6.2f} | {d1_s:>10} | {d2_s:>10} | {d3_s:>10} |  {trend}")
        
    output_lines.append("=" * 80)
    
    # ── 新增: 將全市場 Top-10 榜單加入 output_lines ──
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append(f"  [全市場尋寶] 當日 AI 嚴選 Top-10 潛力飆股 (預測基準日: {latest_date.date()})")
    output_lines.append("=" * 80)
    output_lines.append(f"{'排名':<3} | {'股票 (代號+名稱)':<14} | {'收盤價':<6} | {'D1多空分數':<9} | {'D2多空分數':<9} | {'D3多空分數':<9}")
    output_lines.append("-" * 80)
    
    for i, row in results_top10.iterrows():
        sid = str(row['stock_id'])
        cname = stock_names.get(sid, "")
        stock_display = f"{sid} {cname}"
        
        d1_net = row.get('Day1_net', np.nan)
        d2_net = row.get('Day2_net', np.nan)
        d3_net = row.get('Day3_net', np.nan)
        
        d1_s = f"{d1_net:+.1f}%" if pd.notna(d1_net) else "--"
        d2_s = f"{d2_net:+.1f}%" if pd.notna(d2_net) else "--"
        d3_s = f"{d3_net:+.1f}%" if pd.notna(d3_net) else "--"
        
        output_lines.append(f" {i+1:<3} | {stock_display:<16} | {row['close']:>6.2f} | {d1_s:>10} | {d2_s:>10} | {d3_s:>10}")
        
    output_lines.append("=" * 80)
    output_lines.append(factors_info)
    output_lines.append("=" * 80)
    output_lines.append("  [聲明] 模型預測結果僅供量化研究參考，不構成實際投資建議。")
    
    final_output = "\n".join(output_lines)
    print(final_output)
    
    # Save to file
    pred_dir = os.path.join(BASE_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_file = os.path.join(pred_dir, f"prediction_{latest_date.strftime('%Y%m%d')}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_output)
    print(f"\n  [儲存] 本次預測結果已存檔至: {out_file}")

if __name__ == "__main__":
    main()
