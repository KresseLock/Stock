"""
train.py — LightGBM 模型訓練程式 (預測未來 3 天)
====================================================
修正：
  - 使用日期百分位切割訓練/測試集，避免同一天的股票資料被切割到不同集合 (Data Leakage)
  - 訓練完成後將 feature_cols 存成 models/feature_cols.json，供 inference.py 使用固定欄位
"""
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(df, feature_cols, target_col, days_ahead):
    print(f"\n" + "="*50)
    print(f"  [開始訓練] 預測未來 {days_ahead} 天 (標籤: {target_col})")
    print("="*50)
    
    # 以日期百分位切割，確保同一天的所有股票都在同一個集合裡 (避免 Data Leakage)
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    train_df = df[df["date"] < split_date]
    test_df  = df[df["date"] >= split_date]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test   = test_df[feature_cols],  test_df[target_col]
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + days_ahead,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    y_test_dir = np.sign(y_test)
    preds_dir = np.sign(preds)
    valid_idx = y_test_dir != 0
    dir_acc = np.mean(y_test_dir[valid_idx] == preds_dir[valid_idx]) * 100
    
    print(f"  訓練集: {len(train_df)} 筆 ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    print(f"  測試集: {len(test_df)} 筆 ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")
    print(f"  模型最佳迭代次數: {model.best_iteration_}")
    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | 方向勝率: {dir_acc:.2f}%")
    
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
    
    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    for col in target_cols:
        if col not in df.columns:
            print(f"[錯誤] 缺少標籤欄位 {col}，請重新執行 python run_feature_engineering.py")
            return
            
    df = df.dropna(subset=target_cols).copy()
    ignore_cols = ["stock_id", "date"] + target_cols
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]
    
    print(f"  總樣本數: {len(df)}")
    print(f"  特徵數量: {len(feature_cols)}")
    
    # 儲存 feature_cols，供 inference.py 使用固定欄位清單，避免動態推導不一致
    with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"  特徵欄位清單已儲存至: {FEATURE_COLS_PATH}")
    
    # 分別訓練 3 天的模型
    for days in [1, 2, 3]:
        train_model(df, feature_cols, f"next_ret_{days}", days)

if __name__ == "__main__":
    main()
