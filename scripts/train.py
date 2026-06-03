# -*- coding: utf-8 -*-
"""
train.py — LightGBM 模型訓練程式 (預測未來 3 天)
====================================================
"""
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import lightgbm as lgb
import pandas as pd
import numpy as np

# 統一設定路徑與環境
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 載入中央控制面板 config ──────────────────────────────────
try:
    from config import TRAIN_INDUSTRIES, TRAIN_N_JOBS
    N_JOBS = TRAIN_N_JOBS
except ImportError:
    TRAIN_INDUSTRIES = {}
    N_JOBS = -1


def train_model(df, feature_cols, target_col, days_ahead):
    print(f"\n" + "="*50)
    print(f"  [開始訓練] 預測未來 {days_ahead} 天 (標籤: {target_col})")
    print("="*50)
    
    # 以日期分位數切割：70% 訓練, 10% 驗證, 20% 測試
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    train_end_date = unique_dates[int(n_dates * 0.7)]
    valid_end_date = unique_dates[int(n_dates * 0.8)]
    
    train_df = df[df["date"] < train_end_date]
    valid_df = df[(df["date"] >= train_end_date) & (df["date"] < valid_end_date)]
    test_df  = df[df["date"] >= valid_end_date]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col].astype(int)
    X_valid, y_valid = valid_df[feature_cols], valid_df[target_col].astype(int)
    X_test, y_test   = test_df[feature_cols],  test_df[target_col].astype(int)
    
    # 分類模型
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + days_ahead,
        n_jobs=N_JOBS,
        verbose=-1,
        objective="multiclass",
        num_class=3,
        class_weight="balanced"
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # 預測「強勢上漲 (Class 2)」的機率
    preds_proba = model.predict_proba(X_test)
    if preds_proba.shape[1] == 3:
        prob_strong = preds_proba[:, 2]
    else:
        prob_strong = preds_proba[:, 1] # fallback for binary
        
    # 計算 Daily Top-20 Precision (每天挑選機率最高的前 20 檔)
    test_df = test_df.copy()
    test_df["prob_strong"] = prob_strong
    
    daily_pick = (
        test_df
        .sort_values(["date", "prob_strong"], ascending=[True, False])
        .groupby("date")
        .head(20)
    )
    win_rate = (daily_pick[target_col] == 2).mean() * 100
    
    print(f"  訓練集: {len(train_df)} 筆 ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    print(f"  驗證集: {len(valid_df)} 筆 ({valid_df['date'].min().date()} ~ {valid_df['date'].max().date()})")
    print(f"  測試集: {len(test_df)} 筆 ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")
    print(f"  模型最佳迭代次數: {model.best_iteration_}")
    print(f"  測試集每日 Top-20 強勢股命中率 (Daily Top-K Precision): {win_rate:.2f}%")
    
    model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days_ahead}.txt")
    model.booster_.save_model(model_path)
    print(f"  模型已儲存至: {model_path}")

def main():
    print("=" * 50)
    print("  啟動 LightGBM 多天期預測訓練 (Day 1 ~ Day 3)")
    print("=" * 50)
    
    if not os.path.exists(DATA_PATH):
        print(f"[錯誤] 找不到特徵檔: {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    
    # ── 根據 TRAIN_INDUSTRIES 過濾訓練集股票 ─────────────────────
    cat_path = os.path.join(BASE_DIR, "scripts", "stock_categories.json")
    if os.path.exists(cat_path):
        with open(cat_path, "r", encoding="utf-8") as f:
            categories = json.load(f)
            
        allowed_stocks = set()
        for ind_name, is_enabled in TRAIN_INDUSTRIES.items():
            if is_enabled and ind_name in categories:
                allowed_stocks.update(categories[ind_name].keys())
                
        # 載入 Stocks.txt 自選股並確保其一定保留在訓練集中
        try:
            # 優先加載 scripts/utils.py
            from scripts.utils import load_target_stocks
            allowed_stocks.update(load_target_stocks("Stocks.txt"))
        except ImportError:
            try:
                from utils import load_target_stocks
                allowed_stocks.update(load_target_stocks("Stocks.txt"))
            except Exception as e:
                print(f"  [警告] 載入 Stocks.txt 自選股失敗: {e}")
            
        # 過濾資料
        before_count = df["stock_id"].nunique()
        df_filtered = df[df["stock_id"].isin(allowed_stocks)].copy()
        after_count = df_filtered["stock_id"].nunique()
        print(f"  [訓練篩選] 套用 TRAIN_INDUSTRIES 篩選：原本有 {before_count} 檔股票，篩選後剩餘 {after_count} 檔股票進行訓練。")
        df = df_filtered
    else:
        print("  [警告] 找不到 stock_categories.json，不進行訓練產業篩選")
    # ──────────────────────────────────────────────────────────
    
    label_cols = ["label_1", "label_2", "label_3"]
    ret_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    for col in label_cols:
        if col not in df.columns:
            print(f"[錯誤] 缺少標籤欄位 {col}，請重新執行特徵工程。")
            return
            
    df = df.dropna(subset=label_cols).copy()
    ignore_cols = ["stock_id", "date"] + label_cols + ret_cols
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]
    
    # 統一將特徵轉為 float32
    df[feature_cols] = df[feature_cols].astype(np.float32)
    
    print(f"  總樣本數: {len(df)}")
    print(f"  特徵數量: {len(feature_cols)}")
    
    # 儲存 feature_cols
    with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"  特徵欄位清單已儲存至: {FEATURE_COLS_PATH}")
    
    # 分別訓練 3 天的模型
    for days in [1, 2, 3]:
        train_model(df, feature_cols, f"label_{days}", days)

if __name__ == "__main__":
    main()
