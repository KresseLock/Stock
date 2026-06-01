"""
run_feature_engineering.py — 獨立特徵工程提取工具
====================================================
使用場景:
  當您修改了 Stocks.txt 裡面的股票清單，但「不需要」重新從網路抓取資料時，
  執行本程式將直接讀取本地 data/ 下的現有 raw data，
  重新為 Stocks.txt 中的股票進行特徵工程計算，並更新特徵矩陣。

使用方式:
  1. 修改下方「使用者設定區」的參數 (時間區間、因子開關)。
  2. 修改根目錄的 Stocks.txt (新增、刪除或調整股票代號)。
  3. 執行本程式: python run_feature_engineering.py
  4. 產出特徵矩陣: data/features/features_combined.parquet (供 LightGBM 使用)
"""

import datetime
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ╔══════════════════════════════════════════════════════╗
# ║              使用者設定區 (User Config)              ║
# ╚══════════════════════════════════════════════════════╝

# ── 時間區間 ────────────────────────────────────────────
START_DATE = datetime.date(2020, 1, 1)   # 歷史回溯起點 (建議至少 3~5 年)
END_DATE   = datetime.date.today()       # 結束日期 (today = 自動抓取今日)

# ── 回測模式設定 ─────────────────────────────────────────
# 設定 True 後執行程式，它會詢問您輸入一個「切割日期」
# 程式將使用該日期之前的資料訓練模型，然後預測未來 3 天，
# 並與資料集內真實股價逐一比對，告訴您預測是否正確。
BACKTEST_MODE = False
BACKTEST_DATE = ""  # 格式: "YYYYMMDD"，留空則執行時提示輸入

# ── 技術指標參數 ────────────────────────────────────────
MA_WINDOWS           = [8, 20, 34, 107]
RSI_PERIOD           = 18
ATR_PERIOD           = 23
KD_PERIOD            = 9
MACD_FAST            = 7
MACD_SLOW            = 30
MACD_SIGNAL          = 13
BOLL_WINDOW          = 15
BOLL_STD_MULT        = 2.66
VOL_MA_WINDOW        = 3

# ── 籌碼滾動加總週期 ────────────────────────────────────
# 法人買賣超「滾動加總」的時間視窗（例如：[3,5,10] = 計算近3日、近5日、近10日的買賣超合計）
CHIPS_SUM_WINDOWS    = [4, 7, 25]

# ── 預測天數設定 ────────────────────────────────────────
# 產生幾天的「未來標籤」供 LightGBM 訓練
# [1, 2, 3] = 同時產生 next_ret_1, next_ret_2, next_ret_3
FORECAST_DAYS   = [1, 2, 3]

# ── 因子模組開關 (True=啟用, False=停用) ───────────────
ENABLE_CHIPS        = True   # B. 法人籌碼 (法人買賣超、當沖比例、外資持股比例)
ENABLE_MARGIN       = True   # C. 信用交易 (資券餘額、借券、信用管制限額)
ENABLE_SHAREHOLDING = True   # D. 持股分級 (集保大戶/散戶比例、HHI集中度)
ENABLE_FINMIND      = True   # E. FinMind 基本面 (月營收/三表季報/股利)
ENABLE_SENTIMENT    = True   # F. 市場情緒 (期交所外資台指期未平倉淨額)

# ── 多核心平行運算設定 ────────────────────────────────────
# 設定 -1 為使用全部核心 (建議)，若怕電腦過載可設定為具體數字 (例如 4 或 8)
N_JOBS              = -1


# ╔══════════════════════════════════════════════════════╗
# ║              以下為程式邏輯，一般不需修改              ║
# ╚══════════════════════════════════════════════════════╝

from scripts.feature_engineering import process_all_history_features, load_target_stocks

def check_local_data_integrity(target_stocks: list):
    """
    檢查目標股票在本地是否已有下載好的 FinMind 資料。
    如果沒有，印出友善提示，告知用戶可能需要先運行一次 main.py。
    """
    data_dir = os.path.join(BASE_DIR, "data")
    missing_finmind_stocks = []

    for stock in target_stocks:
        rev_path  = os.path.join(data_dir, "raw_financial", f"{stock}_monthly_revenue.csv")
        stmt_path = os.path.join(data_dir, "raw_financial", f"{stock}_financial_stmt.csv")
        bal_path  = os.path.join(data_dir, "raw_financial", f"{stock}_balance_sheet.csv")
        cf_path   = os.path.join(data_dir, "raw_financial", f"{stock}_cashflow.csv")
        div_path  = os.path.join(data_dir, "raw_financial", f"{stock}_dividend.csv")
        
        missing = False
        for p in [rev_path, stmt_path, bal_path, cf_path, div_path]:
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                missing = True
                break
                
        if missing:
            missing_finmind_stocks.append(stock)

    if missing_finmind_stocks:
        print("=" * 60)
        print("  [注意] 發現股票在本地無完整的 FinMind 歷史資料：")
        print(f"  缺少的股票: {missing_finmind_stocks}")
        print("  提示: 若要完整提取這些股票的財報特徵 (含資產負債表、現金流量表、股利等)，建議先執行 `python main.py`。")
        print("  (若繼續執行，這些股票的特徵將會為空值)")
        print("=" * 60)
        print()

# ── 將使用者設定區的參數傳入特徵工程模組 ────────────────
import scripts.feature_engineering as fe_module

fe_module.MA_WINDOWS        = MA_WINDOWS
fe_module.RSI_PERIOD        = RSI_PERIOD
fe_module.ATR_PERIOD        = ATR_PERIOD
fe_module.KD_PERIOD         = KD_PERIOD
fe_module.MACD_FAST         = MACD_FAST
fe_module.MACD_SLOW         = MACD_SLOW
fe_module.MACD_SIGNAL       = MACD_SIGNAL
fe_module.VOL_MA_WINDOW     = VOL_MA_WINDOW
fe_module.BOLL_WINDOW       = BOLL_WINDOW
fe_module.BOLL_STD_MULT     = BOLL_STD_MULT
fe_module.CHIPS_SUM_WINDOWS = CHIPS_SUM_WINDOWS
fe_module.FORECAST_DAYS     = FORECAST_DAYS
fe_module.ENABLE_CHIPS        = ENABLE_CHIPS
fe_module.ENABLE_MARGIN       = ENABLE_MARGIN
fe_module.ENABLE_SHAREHOLDING = ENABLE_SHAREHOLDING
fe_module.ENABLE_FINMIND      = ENABLE_FINMIND
fe_module.ENABLE_SENTIMENT    = ENABLE_SENTIMENT
fe_module.N_JOBS              = N_JOBS


if __name__ == "__main__":
    print("=" * 60)
    print("  獨立特徵工程提取工具啟動 (僅使用本地現有數據)")
    print("=" * 60)
    print()

    # 印出目前的因子設定，讓使用者確認
    print("[當前因子設定]")
    print(f"  時間區間      : {START_DATE} ~ {END_DATE}")
    print(f"  均線週期      : {MA_WINDOWS}")
    print(f"  RSI 週期      : {RSI_PERIOD} 天")
    print(f"  預測天數      : {FORECAST_DAYS}")
    print(f"  法人籌碼      : {'啟用' if ENABLE_CHIPS        else '停用'}")
    print(f"  信用交易      : {'啟用' if ENABLE_MARGIN       else '停用'}")
    print(f"  持股分級      : {'啟用' if ENABLE_SHAREHOLDING else '停用'}")
    print(f"  FinMind 基本面: {'啟用' if ENABLE_FINMIND      else '停用'}")
    print(f"  市場情緒      : {'啟用' if ENABLE_SENTIMENT    else '停用'}")
    print()

    # 載入當前 Stocks.txt 清單
    target_stocks = load_target_stocks("Stocks.txt")
    print(f"目標股票數量: {len(target_stocks)} 檔")
    print(f"股票清單    : {target_stocks}")
    print()

    # 若 FinMind 啟用，才檢查本地財報完整性
    if ENABLE_FINMIND:
        check_local_data_integrity(target_stocks)

    print(f"分析時間區間: {START_DATE} ~ {END_DATE}")
    print("開始重新計算特徵值...")
    print("-" * 60)

    try:
        process_all_history_features(START_DATE, END_DATE, override_target_stocks=target_stocks)
        print("-" * 60)
        print("  [完成] 特徵值重新提取完成！")
        print("  您現在可以使用 data/features/features_combined.parquet 進行 LightGBM 訓練與推理。")
    except Exception as e:
        import traceback
        print()
        print(f"[錯誤] 特徵提取過程中發生錯誤: {e}")
        traceback.print_exc()
        print("請確認 data/ 目錄下是否存有對應日期的 TWSE 價格/籌碼日報 CSV。")

    # ── 回測模式 ─────────────────────────────────────────
    if BACKTEST_MODE:
        print()
        print("=" * 60)
        print("  [回測模式] 預測結果 vs 真實股價比對")
        print("=" * 60)

        # 取得回測日期
        bt_date_str = BACKTEST_DATE.strip()
        if not bt_date_str:
            bt_date_str = input("  請輸入回測切割日期 (格式 YYYYMMDD，例如 20240501): ").strip()

        try:
            import lightgbm as lgb
            import numpy as np
            import pandas as pd

            bt_date = pd.to_datetime(bt_date_str, format="%Y%m%d")
            parquet_path = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")

            if not os.path.exists(parquet_path):
                print("[錯誤] 找不到特徵檔，請先完成特徵工程再執行回測模式。")
            else:
                df_all = pd.read_parquet(parquet_path)
                df_all["date"] = pd.to_datetime(df_all["date"])

                # 確認回測日期有在資料集內
                available_dates = sorted(df_all["date"].unique())
                if bt_date not in available_dates:
                    # 找最近的交易日
                    close_dates = [d for d in available_dates if d <= bt_date]
                    if not close_dates:
                        print(f"[錯誤] {bt_date_str} 之前沒有任何資料。")
                        raise ValueError("日期超出範圍")
                    bt_date = close_dates[-1]
                    print(f"  [提示] 自動調整為最近的交易日: {bt_date.date()}")

                label_cols  = [f"next_ret_{d}" for d in FORECAST_DAYS]
                ignore_cols = ["stock_id", "date"] + label_cols
                numeric_cols = df_all.select_dtypes(include=[np.number, bool]).columns
                feature_cols = [c for c in numeric_cols if c not in ignore_cols]

                # 訓練集 = 回測日期之前的資料
                train_df = df_all[df_all["date"] < bt_date].dropna(subset=label_cols)
                # 測試集 = 回測當天的資料 (用來預測未來 N 天)
                test_df  = df_all[df_all["date"] == bt_date].copy()

                if train_df.empty or test_df.empty:
                    print("[錯誤] 訓練集或測試集為空，請確認日期範圍。")
                else:
                    print(f"\n  回測切割日期  : {bt_date.date()}")
                    print(f"  訓練集樣本數  : {len(train_df)} 筆")
                    print(f"  預測對象股票  : {sorted(test_df['stock_id'].unique())}")
                    print()

                    X_train = train_df[feature_cols]
                    X_test  = test_df[feature_cols]

                    all_correct = []

                    for day in FORECAST_DAYS:
                        label = f"next_ret_{day}"
                        if label not in df_all.columns:
                            continue

                        y_train = train_df[label]

                        # 訓練 LightGBM (參數已與 train.py 對齊)
                        model = lgb.LGBMRegressor(
                            n_estimators=500, learning_rate=0.03,
                            max_depth=6, num_leaves=31,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42 + day, n_jobs=N_JOBS, verbose=-1
                        )
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        print(f"  ── 預測第 {day} 天 ({bt_date.date()} 後 {day} 個交易日) ──")
                        print(f"  {'股票':<6} | {'預測漲跌':<10} | {'實際漲跌':<10} | {'方向正確?':<8}")
                        print(f"  {'-'*50}")

                        for idx, (_, row) in enumerate(test_df.iterrows()):
                            stock_id   = row["stock_id"]
                            pred_ret   = preds[idx]
                            actual_ret = row.get(label, np.nan)

                            pred_str   = f"+{pred_ret*100:.2f}%" if pred_ret > 0 else f"{pred_ret*100:.2f}%"
                            actual_str = f"+{actual_ret*100:.2f}%" if actual_ret > 0 else f"{actual_ret*100:.2f}%"

                            if np.isnan(actual_ret):
                                correct_str = "無真實資料"
                                all_correct.append(None)
                            else:
                                is_correct = np.sign(pred_ret) == np.sign(actual_ret)
                                correct_str = "[O] 正確" if is_correct else "[X] 錯誤"
                                all_correct.append(is_correct)

                            print(f"  {stock_id:<6} | {pred_str:<10} | {actual_str:<10} | {correct_str}")
                        print()

                    # 計算整體勝率
                    valid = [x for x in all_correct if x is not None]
                    if valid:
                        win_rate = sum(valid) / len(valid) * 100
                        print(f"  [回測總結] 本次回測方向預測勝率: {win_rate:.1f}%  ({sum(valid)}/{len(valid)} 次正確)")
                        print(f"  提示: 可調整設定區的因子參數後，重新執行並比較勝率變化。")
                        print(f"        若想自動最佳化因子，請執行: python optimize_factors.py")

        except Exception as e:
            import traceback
            print(f"[回測錯誤] {e}")
            traceback.print_exc()

    print("=" * 60)

