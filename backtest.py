# -*- coding: utf-8 -*-
"""
backtest.py — 時光機回測工具 (極簡流線對齊版)
===========================================
"""
import os
import sys
import datetime
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")


def load_watchlist_detailed() -> dict:
    try:
        from utils import parse_stocks_detailed
        return parse_stocks_detailed("Stocks.txt")
    except Exception:
        try:
            from utils import parse_stocks_file
            simple = parse_stocks_file("Stocks.txt")
            return {sid: {"cost": cost, "shares": None} for sid, cost in simple.items()}
        except Exception:
            return {}


def get_stock_name(sid, date_str):
    price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
    if os.path.exists(price_file):
        try:
            df_price = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
            df_price["證券代號"] = df_price["證券代號"].str.strip()
            df_price["證券名稱"] = df_price["證券名稱"].str.strip()
            return df_price.set_index("證券代號")["證券名稱"].get(sid, "")
        except Exception:
            return ""
    return ""


def run_backtest(backtest_date_str):
    print("=" * 70)
    print(f"  啟動時光機回測模式 (指定日期: {backtest_date_str})")
    print("=" * 70)

    try:
        backtest_date = pd.to_datetime(backtest_date_str)
    except Exception:
        print(f"[錯誤] 日期格式解析失敗 ({backtest_date_str})，請使用 YYYY-MM-DD 或 YYYYMMDD。")
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

    # 找出指定日期的所有股票資料
    df_eval = df[df["date"] == backtest_date].copy()
    if df_eval.empty:
        print(f"[錯誤] 找不到日期 {backtest_date_str} 的任何資料。請確認該日台股是否有開市。")
        return

    # 取出指定日期之前的資料做為訓練集
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

    print(f"  基準日: {backtest_date.date()} (共 {len(df_eval)} 檔股票) | 訓練樣本: {len(df_train_all)} 筆")
    print("-" * 70)

    results = []

    for days_ahead in [1, 2, 3]:
        label_col = f"label_{days_ahead}"
        ret_col = f"next_ret_{days_ahead}"
        
        train_clean = df_train_all.dropna(subset=[label_col]).copy()
        
        unique_dates = sorted(train_clean["date"].unique())
        if len(unique_dates) < 10:
            print(f"  [警告] 訓練天數過少 ({len(unique_dates)}天)，跳過 Day {days_ahead}")
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
        
        print(f"  訓練 Day {days_ahead} 模型中... ", end="", flush=True)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        print("完成")

        eval_clean = df_eval.dropna(subset=[label_col]).copy()
        if eval_clean.empty:
            continue
            
        X_eval = eval_clean[feature_cols]
        y_real_ret = eval_clean[ret_col]
        y_real_label = eval_clean[label_col]
        
        preds_proba = model.predict_proba(X_eval)
        if preds_proba.shape[1] == 3:
            prob_strong = preds_proba[:, 2]
            prob_weak = preds_proba[:, 0]
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
        
        top_k_num = 3 if len(eval_clean) < 50 else 20
        daily_pick = (
            eval_clean
            .sort_values(["date", "net_score"], ascending=[True, False])
            .groupby("date")
            .head(top_k_num)
        )
        
        top20_hit = (daily_pick["real_label"] == 2).mean() * 100
        top_20_return = daily_pick["real_ret"].mean() * 100
        dir_acc = np.mean(eval_clean["real_ret"] > 0) * 100
        
        results.append({
            "days": days_ahead,
            "acc": dir_acc,
            "top20_acc": top20_hit if pd.notna(top20_hit) else 0,
            "top20_return": top_20_return if pd.notna(top_20_return) else 0
        })

    print("-" * 70)
    print("   [全市場回測總結]")
    print("-" * 70)
    for r in results:
        top_k_num = 3 if len(df_eval) < 50 else 20
        print(f"  Day {r['days']}: 全市場勝率 {r['acc']:.1f}% | 嚴選 Top-{top_k_num} 命中率 {r['top20_acc']:.1f}% (平均獲利 {r['top20_return']:+.2f}%)")

    # ── 自選追蹤詳細對比 ──
    watchlist = load_watchlist_detailed()
    if watchlist:
        df_watch = df_eval[df_eval["stock_id"].isin(watchlist.keys())].copy()
        if not df_watch.empty:
            stock_names = {}
            date_str = backtest_date.strftime("%Y%m%d")
            price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
            if os.path.exists(price_file):
                try:
                    df_p = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
                    df_p["證券代號"] = df_p["證券代號"].str.strip()
                    df_p["證券名稱"] = df_p["證券名稱"].str.strip()
                    stock_names = df_p.set_index("證券代號")["證券名稱"].to_dict()
                except Exception:
                    pass

            print("\n" + "=" * 118)
            print(f"   [持倉與自選詳細回測表] 預測基準日: {backtest_date.date()}")
            print("=" * 118)
            print(
                f"{'排名':<3} | {'類型':<2} | {'股票 (代號+名稱)':<14} | {'收盤價':<6} | "
                f"{'D1預測':<6} | {'D1真實':<6} | {'D2預測':<6} | {'D2真實':<6} | "
                f"{'D3預測':<6} | {'D3真實':<6} | {'勝率'}"
            )
            print("-" * 118)
            
            if "pred_1" in df_watch.columns:
                df_watch = df_watch.sort_values("pred_1", ascending=False)
                
            for i, row in enumerate(df_watch.itertuples(), start=1):
                sid = row.stock_id
                cname = stock_names.get(sid, "")
                disp_name = f"{sid} {cname}"
                close_p = row.close if hasattr(row, "close") and pd.notna(row.close) else 0.0
                
                p1 = getattr(row, "pred_1", np.nan)
                p2 = getattr(row, "pred_2", np.nan)
                p3 = getattr(row, "pred_3", np.nan)
                
                p1_s = f"{p1*100:+.1f}%" if pd.notna(p1) else "  --  "
                p2_s = f"{p2*100:+.1f}%" if pd.notna(p2) else "  --  "
                p3_s = f"{p3*100:+.1f}%" if pd.notna(p3) else "  --  "
                
                r1 = getattr(row, "next_ret_1", np.nan)
                r2 = getattr(row, "next_ret_2", np.nan)
                r3 = getattr(row, "next_ret_3", np.nan)
                
                r1_p = close_p * (1 + r1) if pd.notna(r1) and close_p > 0 else np.nan
                r2_p = close_p * (1 + r2) if pd.notna(r2) and close_p > 0 else np.nan
                r3_p = close_p * (1 + r3) if pd.notna(r3) and close_p > 0 else np.nan
                
                r1_s = f"{r1_p:>6.2f}" if pd.notna(r1_p) else "  --  "
                r2_s = f"{r2_p:>6.2f}" if pd.notna(r2_p) else "  --  "
                r3_s = f"{r3_p:>6.2f}" if pd.notna(r3_p) else "  --  "
                
                # 計算勝率 (預測方向與實際漲跌方向一致的比例)
                hits = 0
                valid_days = 0
                for p_val, r_val in [(p1, r1), (p2, r2), (p3, r3)]:
                    if pd.notna(p_val) and pd.notna(r_val):
                        valid_days += 1
                        if (p_val > 0 and r_val > 0) or (p_val < 0 and r_val < 0) or (p_val == 0 and r_val == 0):
                            hits += 1
                win_rate_s = f"{int((hits / valid_days) * 100)}%" if valid_days > 0 else "  --  "
                
                info = watchlist.get(sid, {"cost": None, "shares": None})
                is_holding = info.get("cost") is not None
                type_s = "持倉" if is_holding else "自選"
                
                print(
                    f" {i:<3} | {type_s:<2} | {disp_name:<18} | {close_p:>6.2f} | "
                    f"{p1_s:>6} | {r1_s:>6} | {p2_s:>6} | {r2_s:>6} | "
                    f"{p3_s:>6} | {r3_s:>6} | {win_rate_s:>4}"
                )
            print("=" * 118)

    # ── 全市場強勢 Top-5 飆股回測 ──
    if "pred_1" in df_eval.columns:
        top5_market = df_eval.sort_values(["pred_1"], ascending=False).head(5)
        stock_names = {}
        date_str = backtest_date.strftime("%Y%m%d")
        price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
        if os.path.exists(price_file):
            try:
                df_p = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
                df_p["證券代號"] = df_p["證券代號"].str.strip()
                df_p["證券名稱"] = df_p["證券名稱"].str.strip()
                stock_names = df_p.set_index("證券代號")["證券名稱"].to_dict()
            except Exception:
                pass

        print("\n" + "=" * 108)
        print(f"   全市場強勢 Top-5 飆股回測 (基準日: {backtest_date_str})")
        print("=" * 108)
        print(
            f"{'排名':<3} | {'股票 (代號+名稱)':<14} | {'收盤價':<6} | "
            f"{'D1預測':<6} | {'D1真實':<6} | {'D2預測':<6} | {'D2真實':<6} | "
            f"{'D3預測':<6} | {'D3真實':<6} | {'勝率'}"
        )
        print("-" * 108)
        
        for i, row in enumerate(top5_market.itertuples(), start=1):
            sid = str(row.stock_id)
            cname = stock_names.get(sid, "")
            disp_name = f"{sid} {cname}"
            close_p = row.close if hasattr(row, "close") and pd.notna(row.close) else 0.0
            
            p1 = getattr(row, "pred_1", np.nan)
            p2 = getattr(row, "pred_2", np.nan)
            p3 = getattr(row, "pred_3", np.nan)
            p1_s = f"{p1*100:+.1f}%" if pd.notna(p1) else "  --  "
            p2_s = f"{p2*100:+.1f}%" if pd.notna(p2) else "  --  "
            p3_s = f"{p3*100:+.1f}%" if pd.notna(p3) else "  --  "
            
            r1 = getattr(row, "next_ret_1", np.nan)
            r2 = getattr(row, "next_ret_2", np.nan)
            r3 = getattr(row, "next_ret_3", np.nan)
            
            r1_p = close_p * (1 + r1) if pd.notna(r1) and close_p > 0 else np.nan
            r2_p = close_p * (1 + r2) if pd.notna(r2) and close_p > 0 else np.nan
            r3_p = close_p * (1 + r3) if pd.notna(r3) and close_p > 0 else np.nan
            
            r1_s = f"{r1_p:>6.2f}" if pd.notna(r1_p) else "  --  "
            r2_s = f"{r2_p:>6.2f}" if pd.notna(r2_p) else "  --  "
            r3_s = f"{r3_p:>6.2f}" if pd.notna(r3_p) else "  --  "
            
            # 計算勝率
            hits = 0
            valid_days = 0
            for p_val, r_val in [(p1, r1), (p2, r2), (p3, r3)]:
                if pd.notna(p_val) and pd.notna(r_val):
                    valid_days += 1
                    if (p_val > 0 and r_val > 0) or (p_val < 0 and r_val < 0) or (p_val == 0 and r_val == 0):
                        hits += 1
            win_rate_s = f"{int((hits / valid_days) * 100)}%" if valid_days > 0 else "  --  "
            
            print(
                f" {i:<3} | {disp_name:<18} | {close_p:>6.2f} | "
                f"{p1_s:>6} | {r1_s:>6} | {p2_s:>6} | {r2_s:>6} | "
                f"{p3_s:>6} | {r3_s:>6} | {win_rate_s:>4}"
            )
        print("=" * 108)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python backtest.py <YYYYMMDD>")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="時光機回測工具")
    parser.add_argument("date", type=str, help="欲進行預測的基準日期 (格式: YYYYMMDD 或 YYYY-MM-DD)")
    args = parser.parse_args()
    
    run_backtest(args.date)
