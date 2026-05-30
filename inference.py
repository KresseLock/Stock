"""
inference.py — LightGBM 模型多天期推理 (Day 1 ~ Day 3)
====================================================
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import lightgbm as lgb
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

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
    
    df_latest = df[df["date"] == latest_date].copy()
    if df_latest.empty:
        print("[錯誤] 最新日期無資料。")
        return
        
    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    ignore_cols = ["stock_id", "date"] + target_cols
    numeric_cols = df_latest.select_dtypes(include=[np.number, bool]).columns
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]
    
    X_latest = df_latest[feature_cols]
    
    results = df_latest[["stock_id"]].copy()
    if "close" in df_latest.columns:
        results["close"] = df_latest["close"]
    else:
        results["close"] = 0.0
        
    # 逐一載入 3 天的模型並預測
    for days in [1, 2, 3]:
        model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days}.txt")
        if not os.path.exists(model_path):
            print(f"[錯誤] 找不到模型 {model_path}，請重新執行 train.py")
            return
        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X_latest)
        results[f"Day{days}_pct"] = preds * 100
        
    # 以 Day 3 的預測漲跌幅來排序 (看長一點的趨勢)
    results = results.sort_values(by="Day3_pct", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 75)
    print(f"  [結果] 未來三天走勢預測 (預測基準日: {latest_date.date()})")
    print("=" * 75)
    print(f"{'排名':<3} | {'股票':<6} | {'收盤價':<8} | {'預測 Day 1':<10} | {'預測 Day 2':<10} | {'預測 Day 3':<10} | {'趨勢分析'}")
    print("-" * 75)
    
    for i, row in results.iterrows():
        stock_id = row["stock_id"]
        close_price = row["close"]
        d1, d2, d3 = row["Day1_pct"], row["Day2_pct"], row["Day3_pct"]
        
        d1_s = f"+{d1:.2f}%" if d1 > 0 else f"{d1:.2f}%"
        d2_s = f"+{d2:.2f}%" if d2 > 0 else f"{d2:.2f}%"
        d3_s = f"+{d3:.2f}%" if d3 > 0 else f"{d3:.2f}%"
        
        # 簡單趨勢判定
        if d1 > 0 and d2 > d1 and d3 > d2: trend = "強勢多頭 (連漲)"
        elif d1 < 0 and d2 < d1 and d3 < d2: trend = "強勢空頭 (連跌)"
        elif d3 > 0: trend = "震盪偏多"
        else: trend = "震盪偏空"
            
        print(f" {i+1:<3} |  {stock_id:<5} |  {close_price:<7.2f} |  {d1_s:<10} |  {d2_s:<10} |  {d3_s:<10} |  {trend}")
        
    print("=" * 75)
    print("  [聲明] 模型預測結果僅供量化研究參考，不構成實際投資建議。")

if __name__ == "__main__":
    main()
