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
    
# 資料路徑唯一來源：config.py § 0
try:
    from config import FEATURES_PARQUET as DATA_PATH
except ImportError:
    # ↓ fallback 必須與 config.py § 0 一致（見該節說明）
    DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 載入中央控制面板 config ──────────────────────────────────
try:
    from config import (
        TRAIN_INDUSTRIES, TRAIN_N_JOBS,
        LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_MAX_DEPTH, LGBM_NUM_LEAVES,
        LGBM_SUBSAMPLE, LGBM_COLSAMPLE, LGBM_EARLY_STOPPING,
        SAMPLE_WEIGHT_DROP_THRESHOLD, SAMPLE_WEIGHT_PENALTY,
        DEFAULT_DECAY_LAMBDA,
        EXCLUDE_FEATURES,
        BACKTEST_DATE,
        TRAIN_SPLIT_RATIO, VALID_SPLIT_RATIO,
    )
    N_JOBS = TRAIN_N_JOBS
except ImportError:
    TRAIN_INDUSTRIES = {}
    N_JOBS = -1
    LGBM_N_ESTIMATORS = 300; LGBM_LEARNING_RATE = 0.03; LGBM_MAX_DEPTH = 4
    LGBM_NUM_LEAVES = 15;   LGBM_SUBSAMPLE = 0.8;      LGBM_COLSAMPLE = 0.8
    LGBM_EARLY_STOPPING = 30
    SAMPLE_WEIGHT_DROP_THRESHOLD = -0.05; SAMPLE_WEIGHT_PENALTY = 2.0
    DEFAULT_DECAY_LAMBDA = 0.002
    EXCLUDE_FEATURES = []
    BACKTEST_DATE = None
    TRAIN_SPLIT_RATIO = 0.70; VALID_SPLIT_RATIO = 0.80

# 共用過濾工具
try:
    from scripts.utils import filter_stocks_by_train_industries
except ImportError:
    from utils import filter_stocks_by_train_industries


def train_model(df, feature_cols, target_col, days_ahead):
    print(f"\n" + "="*50)
    print(f"  [開始訓練] 預測未來 {days_ahead} 天 (標籤: {target_col})")
    print("="*50)
    
    # 根據策略 2 (損失權重)：若未來 3 天內有任一天大跌 <= SAMPLE_WEIGHT_DROP_THRESHOLD，給予懲罰權重以抑止 MDD
    df = df.copy()
    ret_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    if all(c in df.columns for c in ret_cols):
        min_future_ret = df[ret_cols].min(axis=1)
        df["sample_weight"] = np.where(min_future_ret <= SAMPLE_WEIGHT_DROP_THRESHOLD, SAMPLE_WEIGHT_PENALTY, 1.0)
    else:
        df["sample_weight"] = 1.0
    
    # 以日期分位數切割（比例由 config 控制，預設 70% 訓練, 10% 驗證, 20% 測試）
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    train_end_date = unique_dates[int(n_dates * TRAIN_SPLIT_RATIO)]
    valid_end_date = unique_dates[int(n_dates * VALID_SPLIT_RATIO)]
    
    train_df = df[df["date"] < train_end_date]
    valid_df = df[(df["date"] >= train_end_date) & (df["date"] < valid_end_date)]
    test_df  = df[df["date"] >= valid_end_date]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col].astype(int)
    X_valid, y_valid = valid_df[feature_cols], valid_df[target_col].astype(int)
    X_test, y_test   = test_df[feature_cols],  test_df[target_col].astype(int)
    
    # 計算時間衰減權重 (越近的資料，權重越高，防範機制不包含 regime_weight)
    train_dates = pd.to_datetime(train_df["date"])
    max_date = train_dates.max()
    decay_days = (max_date - train_dates).dt.days
    time_weight = np.exp(-DEFAULT_DECAY_LAMBDA * decay_days)
    
    w_train = train_df["sample_weight"].values * time_weight.values
    w_valid = valid_df["sample_weight"].values
    
    # 分類模型
    model = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        subsample=LGBM_SUBSAMPLE,
        colsample_bytree=LGBM_COLSAMPLE,
        random_state=42 + days_ahead,
        n_jobs=N_JOBS,
        verbose=-1,
        objective="multiclass",
        num_class=3,
        class_weight="balanced"
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING, verbose=False)]
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
        print(f"[錯誤] 找不到特徵檔: {DATA_PATH}。請先執行 auto_pipeline.py -s feature 生成特徵 Parquet 檔。")
        return
        
    df = pd.read_parquet(DATA_PATH)
    
    # ── 若為模式 A 歷史截斷模式，限制訓練與驗證資料在 BACKTEST_DATE 以前 ──
    if BACKTEST_DATE is not None:
        cutoff_ts = pd.to_datetime(BACKTEST_DATE)
        before_len = len(df)
        df["date_temp"] = pd.to_datetime(df["date"])
        df = df[df["date_temp"] <= cutoff_ts].copy()
        df.drop(columns=["date_temp"], inplace=True)
        print(f"  [防洩漏保護] 偵測到歷史截斷模式 (config.BACKTEST_DATE={BACKTEST_DATE})：")
        print(f"      訓練與驗證資料限制在 {cutoff_ts.date()} 以前（樣本數：{before_len} -> {len(df)}）。")

    # ── 使用統一過濾器 filter_stocks_by_train_industries 篩選訓練集股票 ──────
    # 注意: 此函式已處理 Stocks.txt 自選股強制保留 與 stock_id 類型對齊
    before_count = df["stock_id"].nunique()
    df = filter_stocks_by_train_industries(df)
    after_count = df["stock_id"].nunique()
    print(f"  [訓練篩選] 套用 TRAIN_INDUSTRIES 篩選：{after_count} 檔進行訓練。")
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
    feature_cols = [c for c in feature_cols if c not in EXCLUDE_FEATURES]

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
