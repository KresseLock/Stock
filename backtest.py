import os
import sys
import datetime
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")

def run_backtest(backtest_date_str):
    print("=" * 60)
    print(f"  啟動時光機回測模式 (Backtest) - 指定日期: {backtest_date_str}")
    print("=" * 60)

    try:
        backtest_date = pd.to_datetime(backtest_date_str)
    except Exception as e:
        print(f"[錯誤] 日期格式解析失敗 ({backtest_date_str})，請使用 YYYY-MM-DD 或 YYYYMMDD 格式。")
        return

    if not os.path.exists(DATA_PATH):
        print(f"[錯誤] 找不到特徵檔: {DATA_PATH}。請先執行 run_feature_engineering.py")
        return

    print("載入歷史特徵矩陣...")
    df = pd.read_parquet(DATA_PATH)
    
    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    for col in target_cols:
        if col not in df.columns:
            print(f"[錯誤] 缺少標籤欄位 {col}，請檢查特徵檔。")
            return

    # 確保資料依時間排序
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # 找出剛好等於指定日期的那一天的所有股票資料
    df_eval = df[df["date"] == backtest_date].copy()
    if df_eval.empty:
        print(f"[錯誤] 特徵檔中找不到日期為 {backtest_date_str} 的任何資料。請確認那天台股是否有開市。")
        return

    # 取出指定日期「之前」的所有資料做為訓練集
    df_train_all = df[df["date"] < backtest_date].copy()
    if df_train_all.empty:
        print(f"[錯誤] 找不到 {backtest_date_str} 之前的任何訓練資料。請選擇更晚的日期。")
        return

    # 決定特徵欄位
    label_cols = ["label_1", "label_2", "label_3"]
    ret_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    ignore_cols = ["stock_id", "date"] + label_cols + ret_cols
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]

    print(f"準備就緒。將使用 {backtest_date.date()} 之前的資料訓練模型。")
    print(f"目標預測基準日: {backtest_date.date()} (共 {len(df_eval)} 檔股票)")
    print("-" * 60)

    results = []

    for days_ahead in [1, 2, 3]:
        label_col = f"label_{days_ahead}"
        ret_col = f"next_ret_{days_ahead}"
        
        # 準備訓練集 (去除該預測天數為 NaN 的資料)
        train_clean = df_train_all.dropna(subset=[label_col]).copy()
        
        # 將訓練集切出 20% 當作 early stopping 的驗證集
        unique_dates = sorted(train_clean["date"].unique())
        if len(unique_dates) < 10:
            print(f"[警告] 訓練資料天數過少 ({len(unique_dates)}天)，跳過預測 Day {days_ahead}")
            continue
            
        split_date = unique_dates[int(len(unique_dates) * 0.8)]
        X_train = train_clean[train_clean["date"] < split_date][feature_cols]
        y_train = train_clean[train_clean["date"] < split_date][label_col].astype(int)
        X_valid = train_clean[train_clean["date"] >= split_date][feature_cols]
        y_valid = train_clean[train_clean["date"] >= split_date][label_col].astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + days_ahead,
            n_jobs=-1,
            verbose=-1,
            objective="multiclass",
            num_class=3,
            class_weight="balanced"
        )
        
        print(f"訓練 Day {days_ahead} 模型中... ", end="", flush=True)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        print("完成!")

        # 預測指定日期的未來表現
        eval_clean = df_eval.dropna(subset=[label_col]).copy()
        if eval_clean.empty:
            print(f"  [警告] 基準日 {backtest_date.date()} 缺乏有效的未來 {days_ahead} 天真實報酬資料可供對比。")
            continue
            
        X_eval = eval_clean[feature_cols]
        y_real_ret = eval_clean[ret_col]
        y_real_label = eval_clean[label_col]
        
        preds_proba = model.predict_proba(X_eval)
        if preds_proba.shape[1] == 3:
            prob_strong = preds_proba[:, 2]  # Class 2 (強勢) 的機率
            prob_weak = preds_proba[:, 0]    # Class 0 (弱勢) 的機率
        else:
            prob_strong = preds_proba[:, 1]
            prob_weak = 1 - prob_strong
            
        net_score = prob_strong - prob_weak
            
        eval_clean["pred_prob"] = prob_strong
        eval_clean["pred_weak"] = prob_weak
        eval_clean["net_score"] = net_score
        eval_clean["real_ret"] = y_real_ret
        eval_clean["real_label"] = y_real_label
        df_eval.loc[eval_clean.index, f"pred_{days_ahead}"] = net_score
        df_eval.loc[eval_clean.index, f"pred_weak_{days_ahead}"] = prob_weak
        
        # 評估每日 Top-K 的勝率 (若全市場只有 17 檔，選前 20 等同於買下全市場，故改為動態 Top-3)
        top_k_num = 3 if len(eval_clean) < 50 else 20
        daily_pick = (
            eval_clean
            .sort_values(["date", "net_score"], ascending=[True, False])
            .groupby("date")
            .head(top_k_num)
        )
        
        # 勝率 = 這些最強勢股票中，真正被標記為 2 (強勢) 或漲幅 > 0 的比例
        # 若有命中 Label=2 (代表這檔股票在當天所有股票中排名前 20%)
        top20_hit = (daily_pick["real_label"] == 2).mean() * 100
        top_20_return = daily_pick["real_ret"].mean() * 100
        
        # 為了比較，也算一下全市場勝率 (這時看大於 0 就好，因為市場可能多半跌)
        dir_acc = np.mean(eval_clean["real_ret"] > 0) * 100
        
        results.append({
            "days": days_ahead,
            "acc": dir_acc,
            "top20_acc": top20_hit if pd.notna(top20_hit) else 0,
            "top20_return": top_20_return if pd.notna(top_20_return) else 0
        })
        
        print(f"  ▶ 預測未來 {days_ahead} 天 (Day {days_ahead}) 結果:")
        print(f"    - 測試集全市場勝率 (>0%): {dir_acc:.2f}% (評估 {len(eval_clean)} 檔)")
        print(f"    - 模型看多前 {top_k_num} 檔真實強勢率 (命中 Label=2): {results[-1]['top20_acc']:.2f}%")
        print(f"    - 模型看多前 {top_k_num} 檔實際平均報酬: {results[-1]['top20_return']:.2f}%")
        print("-" * 60)

    print("=" * 60)
    print("  回測總結報告 (全市場)")
    print("=" * 60)
    for r in results:
        print(f"Day {r['days']}: 全市場勝率 {r['acc']:.1f}% | 嚴選 Top-{top_k_num} 命中強勢股機率 {r['top20_acc']:.1f}% (平均獲利 {r['top20_return']:.2f}%)")
    
    # ── 印出 Stocks.txt 專屬的詳細預測表 ──
    watchlist_path = os.path.join(BASE_DIR, "Stocks.txt")
    if os.path.exists(watchlist_path):
        with open(watchlist_path, "r", encoding="utf-8") as f:
            watchlist = [line.split(",")[0].strip() for line in f if line.strip() and not line.strip().startswith("#")]
            
        df_watch = df_eval[df_eval["stock_id"].isin(watchlist)].copy()
        if not df_watch.empty:
            # 取得股票名稱
            stock_names = {}
            date_str = backtest_date.strftime("%Y%m%d")
            price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
            if os.path.exists(price_file):
                try:
                    df_p = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
                    df_p["證券代號"] = df_p["證券代號"].str.strip()
                    df_p["證券名稱"] = df_p["證券名稱"].str.strip()
                    stock_names = df_p.set_index("證券代號")["證券名稱"].to_dict()
                except: pass

            print("\n" + "=" * 105)
            print(f"  自選股 (Stocks.txt) 詳細回測清單 - 基準日: {date_str}")
            print("=" * 105)
            print(f"{'排名':<3} | {'股票 (代號+名稱)':<14} | {'收盤價':<6} | {'D1多空分數':<9} | {'Day1真實':<8} | {'D2多空分數':<9} | {'Day2真實':<8} | {'D3多空分數':<9} | {'Day3真實':<8}")
            print("-" * 105)
            
            # 依 Day 1 預測降冪排序
            if "pred_1" in df_watch.columns:
                df_watch = df_watch.sort_values("pred_1", ascending=False)
                
            for i, row in enumerate(df_watch.itertuples(), start=1):
                sid = row.stock_id
                name = stock_names.get(sid, "")
                disp_name = f"{sid} {name}"
                close_p = row.close if hasattr(row, "close") and pd.notna(row.close) else 0.0
                
                # 取得預測
                p1 = getattr(row, "pred_1", np.nan)
                p2 = getattr(row, "pred_2", np.nan)
                p3 = getattr(row, "pred_3", np.nan)
                
                p1_s = f"{p1*100:+.1f}%" if pd.notna(p1) else "   --   "
                p2_s = f"{p2*100:+.1f}%" if pd.notna(p2) else "   --   "
                p3_s = f"{p3*100:+.1f}%" if pd.notna(p3) else "   --   "
                
                # 取得真實
                r1 = getattr(row, "next_ret_1", np.nan)
                r2 = getattr(row, "next_ret_2", np.nan)
                r3 = getattr(row, "next_ret_3", np.nan)
                r1_s = f"{r1*100:+.1f}%" if pd.notna(r1) else "  --  "
                r2_s = f"{r2*100:+.1f}%" if pd.notna(r2) else "  --  "
                r3_s = f"{r3*100:+.1f}%" if pd.notna(r3) else "  --  "
                
                print(f" {i:<3} | {disp_name:<16} | {close_p:>6.2f} | {p1_s:>10} | {r1_s:>8} | {p2_s:>10} | {r2_s:>8} | {p3_s:>10} | {r3_s:>8}")
            print("=" * 105)

    # ── 新增: 印出當日全市場真正的 Top 10 強勢股 (以 Day 1 為主) ──
    print()
    print("=========================================================================================================")
    print(f"  當日全市場嚴選 Top-10 飆股清單 (以 Day 1 分數排序) - 基準日: {backtest_date_str}")
    print("=========================================================================================================")
    print(f"{'排名':<4} | {'股票 (代號+名稱)':<15} | {'收盤價':<7} | {'D1多空分數':<11} | {'Day1真實':<8} | {'D2多空分數':<11} | {'Day2真實':<8} | {'D3多空分數':<11} | {'Day3真實':<8}")
    print("-" * 105)
    
    # 簡單輔助函數取名字
    def get_stock_name(sid, date_str):
        price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
        if os.path.exists(price_file):
            try:
                df_p = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
                name = df_p[df_p["證券代號"].str.strip() == sid]["證券名稱"].values
                return name[0].strip() if len(name) > 0 else ""
            except: pass
        return ""

    top10_market = df_eval.sort_values(["pred_1"], ascending=False).head(10)
    for i, (_, row) in enumerate(top10_market.iterrows(), 1):
        sid = str(row['stock_id'])
        name = get_stock_name(sid, backtest_date_str)
        stock_display = f"{sid} {name}"
        c = row.get('close', 0.0)
        
        p1 = row.get('pred_1', np.nan)
        p2 = row.get('pred_2', np.nan)
        p3 = row.get('pred_3', np.nan)
        r1 = row.get('next_ret_1', np.nan)
        r2 = row.get('next_ret_2', np.nan)
        r3 = row.get('next_ret_3', np.nan)
        
        p1_str = f"{p1*100:>+8.1f}%" if pd.notna(p1) else f"{'--':>9}"
        p2_str = f"{p2*100:>+8.1f}%" if pd.notna(p2) else f"{'--':>9}"
        p3_str = f"{p3*100:>+8.1f}%" if pd.notna(p3) else f"{'--':>9}"
        r1_str = f"{r1*100:>+7.1f}%" if pd.notna(r1) else f"{'--':>8}"
        r2_str = f"{r2*100:>+7.1f}%" if pd.notna(r2) else f"{'--':>8}"
        r3_str = f"{r3*100:>+7.1f}%" if pd.notna(r3) else f"{'--':>8}"
        
        print(f" {i:<3} | {stock_display:<14} | {c:>7.2f} | {p1_str:<12} | {r1_str:<9} | {p2_str:<12} | {r2_str:<9} | {p3_str:<12} | {r3_str:<9}")
    print("=========================================================================================================")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python backtest.py <YYYYMMDD>")
    parser = argparse.ArgumentParser(description="時光機回測工具")
    parser.add_argument("date", type=str, help="欲進行預測的基準日期 (格式: YYYYMMDD 或 YYYY-MM-DD)")
    args = parser.parse_args()
    
    run_backtest(args.date)
