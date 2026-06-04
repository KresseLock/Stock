# -*- coding: utf-8 -*-
"""
optimize_factors.py — 因子貝葉斯自動最佳化 (Optuna TPE)
================================================================
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import threading

_lock = threading.Lock()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 隱藏 Optuna 的系統詳細 log，由我們自己控制輸出
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 統一將 BASE_DIR 設為專案根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 載入中央控制面板 config ──────────────────────────────────
try:
    from config import BACKTEST_DATE, OPTIMIZATION_TRIALS, EARLY_STOPPING_ROUNDS, OPTUNA_N_JOBS
    MAX_ITERATIONS = OPTIMIZATION_TRIALS
    N_JOBS = OPTUNA_N_JOBS
except ImportError:
    BACKTEST_DATE  = "20250801"
    MAX_ITERATIONS = 400
    EARLY_STOPPING_ROUNDS = None
    N_JOBS = min(os.cpu_count() or 1, 4)

# 預測哪幾天的勝率作為評分基準
FORECAST_DAYS  = [1, 2, 3]

# 最佳化結果儲存路徑 (保存在根目錄)
RESULT_PATH    = os.path.join(BASE_DIR, "best_factors.json")

# ── 參數搜尋邊界 ─────────────────────────────────────────
BOUNDS = {
    # 均線 (短 → 中1 → 中2 → 長，採偏移量設計)
    "ma_short":         ( 3,  9),   # 短均線絕對值
    "ma_mid1_offset":   ( 1, 15),   # 中均線1 = ma_short + offset
    "ma_mid2_offset":   ( 1, 25),   # 中均線2 = ma_mid1  + offset
    "ma_long_offset":   (10, 80),   # 長均線  = ma_mid2  + offset

    # 振盪指標 (絕對值)
    "rsi_period":       ( 7, 30),
    "kd_period":        ( 5, 20),
    "atr_period":       ( 7, 28),

    # MACD (慢線 = 快線 + 偏移量，保證 fast < slow)
    "macd_fast":        ( 6, 18),
    "macd_slow_offset": ( 5, 35),   # 慢線 = fast + offset
    "macd_signal":      ( 5, 15),

    # 布林通道
    "boll_window":      (10, 35),
    "boll_std_x100":    (150, 300), # 實際值 = boll_std_x100 / 100.0 (1.5 ~ 3.0)

    # 成交量均線
    "vol_ma":           ( 3, 15),

    # 籌碼視窗 (採偏移量設計)
    "chips_w1":         ( 2,  7),
    "chips_w2_offset":  ( 2, 10),   # w2 = w1 + offset
    "chips_w3_offset":  ( 5, 20),   # w3 = w2 + offset
}


import re

# 識別 TA 欄位前綴 (這些欄位每輪都需重新計算)
_TA_PREFIXES = (
    "boll_", "rsi", "macd", "atr", "vol_ma", "vol_ratio",
    "ret1", "ret5", "amplitude"
)
_KD_PATTERN = re.compile(r'^[kd]\d+$')
_MA_PATTERN = re.compile(r'^ma\d+$')

def _is_ta_col(col: str) -> bool:
    if _KD_PATTERN.match(col): return True
    if _MA_PATTERN.match(col): return True
    if col == "ma_short_over_long": return True
    return any(col.startswith(p) for p in _TA_PREFIXES)


def _decode_params(trial) -> dict:
    """從 optuna trial 解碼出真正的參數值（將偏移量轉為絕對值）"""
    ma_short = trial.suggest_int("ma_short", *BOUNDS["ma_short"])
    ma_mid1  = ma_short + trial.suggest_int("ma_mid1_offset", *BOUNDS["ma_mid1_offset"])
    ma_mid2  = ma_mid1  + trial.suggest_int("ma_mid2_offset", *BOUNDS["ma_mid2_offset"])
    ma_long  = ma_mid2  + trial.suggest_int("ma_long_offset", *BOUNDS["ma_long_offset"])
    
    assert ma_mid1 > ma_short and ma_mid2 > ma_mid1 and ma_long > ma_mid2, "均線週期必須嚴格遞增"

    macd_fast  = trial.suggest_int("macd_fast",        *BOUNDS["macd_fast"])
    macd_slow  = macd_fast + trial.suggest_int("macd_slow_offset", *BOUNDS["macd_slow_offset"])

    chips_w1   = trial.suggest_int("chips_w1",         *BOUNDS["chips_w1"])
    chips_w2   = chips_w1 + trial.suggest_int("chips_w2_offset",  *BOUNDS["chips_w2_offset"])
    chips_w3   = chips_w2 + trial.suggest_int("chips_w3_offset",  *BOUNDS["chips_w3_offset"])

    return {
        "ma_short":    ma_short,
        "ma_mid1":     ma_mid1,
        "ma_mid2":     ma_mid2,
        "ma_long":     ma_long,
        "rsi_period":  trial.suggest_int("rsi_period",  *BOUNDS["rsi_period"]),
        "kd_period":   trial.suggest_int("kd_period",   *BOUNDS["kd_period"]),
        "atr_period":  trial.suggest_int("atr_period",  *BOUNDS["atr_period"]),
        "macd_fast":   macd_fast,
        "macd_slow":   macd_slow,
        "macd_signal": trial.suggest_int("macd_signal", *BOUNDS["macd_signal"]),
        "boll_window": trial.suggest_int("boll_window", *BOUNDS["boll_window"]),
        "boll_std":    trial.suggest_int("boll_std_x100", *BOUNDS["boll_std_x100"]) / 100.0,
        "vol_ma":      trial.suggest_int("vol_ma",      *BOUNDS["vol_ma"]),
        "chips_w1":    chips_w1,
        "chips_w2":    chips_w2,
        "chips_w3":    chips_w3,
    }


def compute_ta(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """根據參數 p 從 OHLCV 計算所有 TA 特徵（純記憶體操作）"""
    result_dfs = []
    for stock_id, g_orig in df.groupby("stock_id"):
        g = g_orig.sort_values("date")
        c, h, l, v, o = (g[col].astype(float) for col in ["close","high","low","volume","open"])

        new_features = {}
        # 均線
        short, long_ = p["ma_short"], p["ma_long"]
        for w in [p["ma_short"], p["ma_mid1"], p["ma_mid2"], p["ma_long"]]:
            new_features[f"ma{w}"] = c.rolling(w, min_periods=1).mean()
        new_features["ma_short_over_long"] = new_features[f"ma{short}"] / (new_features[f"ma{long_}"] + 1e-9)

        # 布林通道
        bm = c.rolling(p["boll_window"], min_periods=1).mean()
        bs = c.rolling(p["boll_window"], min_periods=5).std()
        new_features["boll_mid"] = bm
        new_features["boll_up"]  = bm + p["boll_std"] * bs
        new_features["boll_dn"]  = bm - p["boll_std"] * bs
        new_features["boll_pct"] = (c - new_features["boll_dn"]) / (new_features["boll_up"] - new_features["boll_dn"] + 1e-9)

        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(p["rsi_period"], min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(p["rsi_period"], min_periods=1).mean()
        new_features[f"rsi{p['rsi_period']}"] = 100 - 100 / (1 + gain / (loss + 1e-9))

        # MACD
        ef = c.ewm(span=p["macd_fast"],   adjust=False).mean()
        es = c.ewm(span=p["macd_slow"],   adjust=False).mean()
        macd = ef - es
        macd_sig  = macd.ewm(span=p["macd_signal"], adjust=False).mean()
        new_features["macd"]      = macd
        new_features["macd_sig"]  = macd_sig
        new_features["macd_hist"] = macd - macd_sig

        # KD
        ln = l.rolling(p["kd_period"], min_periods=1).min()
        hn = h.rolling(p["kd_period"], min_periods=1).max()
        rsv = (c - ln) / (hn - ln + 1e-9) * 100
        k_col = f"k{p['kd_period']}"
        d_col = f"d{p['kd_period']}"
        new_features[k_col] = rsv.ewm(com=2, adjust=False).mean()
        new_features[d_col] = new_features[k_col].ewm(com=2, adjust=False).mean()

        # ATR
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        new_features[f"atr{p['atr_period']}"]     = tr.rolling(p["atr_period"], min_periods=1).mean()
        new_features[f"atr{p['atr_period']}_pct"] = new_features[f"atr{p['atr_period']}"] / (c + 1e-9)

        # 成交量
        new_features[f"vol_ma{p['vol_ma']}"]    = v.rolling(p["vol_ma"], min_periods=1).mean()
        new_features[f"vol_ratio{p['vol_ma']}"] = v / (new_features[f"vol_ma{p['vol_ma']}"] + 1)

        # 報酬率
        new_features["ret1"]      = c.pct_change(1)
        new_features["ret5"]      = c.pct_change(5)
        new_features["amplitude"] = (h - l) / (o + 1e-9)

        result_dfs.append(g.assign(**new_features))

    return pd.concat(result_dfs, ignore_index=True)


def compute_chips_features(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """重新計算籌碼連續買賣超（視窗由參數決定）"""
    df = df.sort_values(["stock_id", "date"]).copy()
    for col in ["fini_net", "sitc_net", "dealer_net", "inst_net_total"]:
        if col not in df.columns: continue
        sign_col = f"{col}_sign"
        df[sign_col] = np.sign(df[col])
        df[f"{col}_streak"] = df.groupby("stock_id")[sign_col].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1) * df[sign_col]
        df.drop(columns=[sign_col], inplace=True)
        for w in [p["chips_w1"], p["chips_w2"], p["chips_w3"]]:
            df[f"{col}_sum{w}"] = df.groupby("stock_id")[col].transform(
                lambda x: x.rolling(w, min_periods=1).sum())
    return df


# 非 TA 穩定特徵欄位（每輪不需重新計算，直接從記憶體讀取）
_STABLE_NON_TA_COLS = [
    "fini_net", "sitc_net", "dealer_net", "inst_net_total",
    "margin_bal", "short_bal", "sbl_bal", "sbl_sell",
    "big_holder_pct", "small_holder_pct", "holder_hhi",
    "revenue", "EPS", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes", "Revenue",
    "TotalAssets", "Liabilities", "Equity",
    "CashFlowsFromOperatingActivities", "CashProvidedByInvestingActivities", "CashFlowsProvidedFromFinancingActivities",
    "cash_dividend",
    "PER", "PBR", "dividend_yield", 
    "daytrading_pct", "fini_holding_pct", "margin_quota", "short_quota",
    "taifex_txf_fini_net_oi"
]


def make_objective(df_base: pd.DataFrame, bt_date: pd.Timestamp,
                   best_holder: dict) -> callable:
    label_cols = [f"label_{d}" for d in FORECAST_DAYS]
    ret_cols = [f"next_ret_{d}" for d in FORECAST_DAYS]
    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    # 預先準備穩定特徵（每輪都用同一份）
    stable_cols_exist = [c for c in _STABLE_NON_TA_COLS if c in df_base.columns]

    def objective(trial) -> float:
        p = _decode_params(trial)

        # 計算 TA (記憶體操作)
        df_ta     = compute_ta(df_base[["stock_id","date"] + ohlcv_cols].copy(), p)
        df_chips  = compute_chips_features(df_ta.copy(), p)

        # 合併穩定特徵與標籤
        merge_cols = ["stock_id", "date"] + stable_cols_exist
        available_labels = [c for c in label_cols if c in df_base.columns]
        available_rets = [c for c in ret_cols if c in df_base.columns]
        df_merged = pd.merge(
            df_chips,
            df_base[merge_cols + available_labels + available_rets].drop_duplicates(["stock_id","date"]),
            on=["stock_id","date"], how="left"
        )

        # 切分訓練/測試
        ignore_cols  = ["stock_id", "date"] + available_labels + available_rets
        numeric_cols = df_merged.select_dtypes(include=[np.number, bool]).columns
        feat_cols    = [c for c in numeric_cols if c not in ignore_cols]

        train_df = df_merged[df_merged["date"] < bt_date].dropna(subset=available_labels)
        test_df  = df_merged[df_merged["date"] >= bt_date].dropna(subset=available_labels)

        if train_df.empty or test_df.empty or len(feat_cols) == 0:
            return 0.0

        X_tr, X_te = train_df[feat_cols], test_df[feat_cols]
        day_scores  = []

        for day in FORECAST_DAYS:
            label = f"label_{day}"
            if label not in df_merged.columns: continue
            
            y_train = train_df[label].astype(int)
            y_test  = test_df[label].astype(int)
            
            # 計算樣本權重：如果未來 3 天內有任一天大跌 <= -5%，給予 2.0 倍懲罰權重
            weight_ret_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
            if all(c in train_df.columns for c in weight_ret_cols):
                min_future_ret = train_df[weight_ret_cols].min(axis=1)
                w_train = np.where(min_future_ret <= -0.05, 2.0, 1.0)
            else:
                w_train = np.ones(len(train_df))
            
            model = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.03, max_depth=4,
                num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=1, verbose=-1,
                class_weight="balanced"
            )
            model.fit(X_tr, y_train, sample_weight=w_train)
            
            preds_proba = model.predict_proba(X_te)
            if preds_proba.shape[1] == 3:
                prob_strong = preds_proba[:, 2]
                prob_weak   = preds_proba[:, 0]
            else:
                prob_strong = preds_proba[:, 1]
                prob_weak   = 1.0 - prob_strong
                
            net_score = prob_strong - prob_weak
            
            # 使用測試集進行 Top-K 評估
            test_day_df = test_df[["date", label]].copy()
            test_day_df["net_score"] = net_score
            
            top_k_num = 3 if len(test_day_df["date"].unique()) > 0 and len(test_day_df) / len(test_day_df["date"].unique()) < 50 else 20
            
            daily_pick = (
                test_day_df
                .sort_values(["date", "net_score"], ascending=[True, False])
                .groupby("date")
                .head(top_k_num)
            )
            
            hit_rate = (daily_pick[label] == 2).mean() * 100
            day_scores.append(hit_rate)

        avg = float(np.mean(day_scores)) if day_scores else 0.0

        # 更新最佳解快取
        with _lock:
            if avg > best_holder.get("score", -1):
                best_holder["score"]  = avg
                best_holder["params"] = p.copy()

        return avg

    return objective


def print_progress(study, trial):
    try:
        n    = trial.number + 1
        val  = trial.value if trial.value is not None else 0.0
        best = study.best_value
        p    = study.best_trial.params

        is_best = abs(val - best) < 1e-6
        if is_best or n % 50 == 0 or n == 1:
            tag   = " << 新最佳解!" if is_best else ""
            ma_s  = p.get("ma_short", "?")
            rsi   = p.get("rsi_period", "?")
            mf    = p.get("macd_fast", "?")
            ms_os = p.get("macd_slow_offset", "?")
            bw    = p.get("boll_window", "?")
            print(f"  [{n:>4}/{MAX_ITERATIONS}] "
                  f"本輪={val:.1f}%  最佳={best:.1f}%  "
                  f"(RSI={rsi} MA_short={ma_s} MACD={mf}/{mf+(ms_os or 0)} Boll={bw})"
                  f"{tag}", flush=True)

        if EARLY_STOPPING_ROUNDS is not None and EARLY_STOPPING_ROUNDS > 0:
            best_trial_number = study.best_trial.number
            current_trial_number = trial.number
            rounds_without_improvement = current_trial_number - best_trial_number
            if rounds_without_improvement >= EARLY_STOPPING_ROUNDS:
                print(f"\n  [提早結束] 連續 {EARLY_STOPPING_ROUNDS} 次未找到更好的參數，觸發 Early Stopping！", flush=True)
                study.stop()
    except Exception:
        pass


def main():
    print("=" * 65)
    print("  因子貝葉斯最佳化 (Optuna TPE)")
    print("=" * 65)
    print(f"  回測切割日期: {BACKTEST_DATE}")
    print(f"  最大迭代次數: {MAX_ITERATIONS}  (貝葉斯效率 = Random Search x 200倍)")
    print(f"  搜尋參數個數: {len(BOUNDS)} 個連續參數")
    print()

    parquet_path = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
    if not os.path.exists(parquet_path):
        print("[錯誤] 找不到特徵檔，請先執行特徵工程計算。")
        return

    print("[1] 載入特徵數據 (僅此一次)...")
    df_base = pd.read_parquet(parquet_path)
    
    # ── 根據 config 的產業與自選設定過濾股票 ──────────────────
    try:
        from scripts.utils import filter_stocks_by_train_industries
    except ImportError:
        from utils import filter_stocks_by_train_industries
        
    before_cnt = df_base["stock_id"].nunique()
    df_base = filter_stocks_by_train_industries(df_base)
    after_cnt = df_base["stock_id"].nunique()
    print(f"  [最佳化過濾] 依 config 產業設定篩選：{after_cnt} 檔進行因子最佳化")
    
    df_base["date"] = pd.to_datetime(df_base["date"])

    bt_date = pd.to_datetime(BACKTEST_DATE, format="%Y%m%d")
    avail   = sorted(df_base["date"].unique())
    close   = [d for d in avail if d <= bt_date]
    if not close:
        print(f"[錯誤] 回測日期 {BACKTEST_DATE} 之前沒有資料。")
        return
    bt_date = close[-1]
    print(f"   資料筆數   : {len(df_base)}")
    print(f"   回測切割日 : {bt_date.date()}")
    print(f"   訓練樣本   : {len(df_base[df_base['date'] < bt_date])}")
    print(f"   評估樣本   : {len(df_base[df_base['date'] >= bt_date])}")
    print()

    # ── 步驟 2: 建立 study 並執行最佳化 ──────────
    print("[2] 啟動 Optuna 貝葉斯最佳化...")
    print(f"    (並行數: {N_JOBS}，每 50 輪或出現新最佳解時顯示進度)")
    print("-" * 65)

    best_holder = {"score": -1.0, "params": {}}
    objective   = make_objective(df_base, bt_date, best_holder)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=40
        ),
    )
    try:
        study.optimize(
            objective, 
            n_trials=MAX_ITERATIONS, 
            callbacks=[print_progress],
            n_jobs=N_JOBS
        )
    except KeyboardInterrupt:
        print("\n  [中斷] 使用者按下 Ctrl+C，強制結束最佳化執行緒...")
        os._exit(1)

    # ── 步驟 3: 輸出最佳結果 ────────────────────────────
    best_trial  = study.best_trial
    best_score  = study.best_value
    raw_p       = best_trial.params

    ma_short = raw_p["ma_short"]
    ma_mid1  = ma_short + raw_p["ma_mid1_offset"]
    ma_mid2  = ma_mid1  + raw_p["ma_mid2_offset"]
    ma_long  = ma_mid2  + raw_p["ma_long_offset"]
    macd_fast = raw_p["macd_fast"]
    macd_slow = macd_fast + raw_p["macd_slow_offset"]
    chips_w1  = raw_p["chips_w1"]
    chips_w2  = chips_w1 + raw_p["chips_w2_offset"]
    chips_w3  = chips_w2 + raw_p["chips_w3_offset"]

    best_config = {
        "MA_WINDOWS":        [ma_short, ma_mid1, ma_mid2, ma_long],
        "RSI_PERIOD":        raw_p["rsi_period"],
        "ATR_PERIOD":        raw_p["atr_period"],
        "KD_PERIOD":         raw_p["kd_period"],
        "MACD_FAST":         macd_fast,
        "MACD_SLOW":         macd_slow,
        "MACD_SIGNAL":       raw_p["macd_signal"],
        "BOLL_WINDOW":       raw_p["boll_window"],
        "BOLL_STD_MULT":     raw_p["boll_std_x100"] / 100.0,
        "VOL_MA_WINDOW":     raw_p["vol_ma"],
        "CHIPS_SUM_WINDOWS": [chips_w1, chips_w2, chips_w3],
    }

    print()
    print("=" * 65)
    print(f"  [完成] Optuna {MAX_ITERATIONS} 次貝葉斯搜尋結束！")
    print(f"  歷史最佳平均勝率: {best_score:.2f}%")
    print("=" * 65)
    print()
    print("已找到最佳參數組。")

    # ── 儲存結果 ────────────────────────────────────────
    result = {
        "optimized_at":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":       "Optuna TPE (Bayesian)",
        "iterations":      MAX_ITERATIONS,
        "best_score_avg":  best_score,
        "backtest_date":   BACKTEST_DATE,
        "best_params_for_run_feature_engineering": best_config,
        "top10_trials": [
            {"trial": t.number, "score": t.value}
            for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:10]
        ]
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print(f"  最佳參數已儲存至: {RESULT_PATH}")
    print()


if __name__ == "__main__":
    main()
