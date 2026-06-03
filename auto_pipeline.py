# -*- coding: utf-8 -*-
"""
auto_pipeline.py — 一鍵自動化量化交易流水線 (CLI 步驟路由版)
================================================================
執行順序:
  步驟 1. 貝葉斯因子最佳化 (scripts/optimize_factors.py)
  步驟 2. 載入並套用最佳因子參數 (scripts/feature_engineering.py)
  步驟 3. 重建特徵工程矩陣 (scripts/feature_engineering.py)
  步驟 4. 訓練 LightGBM 模型 (scripts/train.py)
  步驟 5. 模型預測推理 (scripts/inference.py)

用法:
  python auto_pipeline.py                 # 一鍵跑完完整流程 (由 config.py 控制是否執行最佳化)
  python auto_pipeline.py --step optimize # 僅執行步驟 1 因子最佳化
  python auto_pipeline.py --step feature  # 僅執行步驟 2+3 特徵工程
  python auto_pipeline.py --step train    # 僅執行步驟 4 訓練模型
  python auto_pipeline.py --step inference # 僅執行步驟 5 模型推理
"""

import os
import sys
import json
import datetime
import time
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 載入中央控制面板 config 設定 ──────────────────────────────
try:
    from config import (
        RUN_OPTIMIZATION,
        OPTIMIZATION_TRIALS,
        EARLY_STOPPING_ROUNDS,
        BACKTEST_DATE,
        START_DATE,
        TRAIN_INDUSTRIES,
        FEAT_N_JOBS,
        TRAIN_N_JOBS,
        OPTUNA_N_JOBS
    )
except ImportError:
    RUN_OPTIMIZATION      = False
    OPTIMIZATION_TRIALS   = 600
    EARLY_STOPPING_ROUNDS = 200
    BACKTEST_DATE         = "20250801"
    START_DATE            = datetime.date(2020, 1, 1)
    FEAT_N_JOBS           = -1
    TRAIN_N_JOBS          = -1
    OPTUNA_N_JOBS         = 6
    TRAIN_INDUSTRIES      = {}

END_DATE = datetime.date.today()
BEST_FACTORS_PATH = os.path.join(BASE_DIR, "best_factors.json")

# 引入位於 scripts/ 的新模組
import scripts.feature_engineering as fe_module
from scripts.feature_engineering import process_all_history_features
from scripts.utils import load_target_stocks


def _banner(step: str, title: str):
    print()
    print("=" * 65)
    print(f"  步驟 {step}: {title}")
    print("=" * 65)


def _apply_best_params(params: dict):
    """將 best_factors.json 中的最佳參數套用到 feature_engineering 模組"""
    mapping = {
        "MA_WINDOWS":        "MA_WINDOWS",
        "RSI_PERIOD":        "RSI_PERIOD",
        "ATR_PERIOD":        "ATR_PERIOD",
        "KD_PERIOD":         "KD_PERIOD",
        "MACD_FAST":         "MACD_FAST",
        "MACD_SLOW":         "MACD_SLOW",
        "MACD_SIGNAL":       "MACD_SIGNAL",
        "BOLL_WINDOW":       "BOLL_WINDOW",
        "BOLL_STD_MULT":     "BOLL_STD_MULT",
        "VOL_MA_WINDOW":     "VOL_MA_WINDOW",
        "CHIPS_SUM_WINDOWS": "CHIPS_SUM_WINDOWS",
    }
    for json_key, module_attr in mapping.items():
        if json_key in params and hasattr(fe_module, module_attr):
            setattr(fe_module, module_attr, params[json_key])
    # 固定預測天數
    fe_module.FORECAST_DAYS = [1, 2, 3]
    print(f"  [套用參數] MA={params.get('MA_WINDOWS')}  "
          f"RSI={params.get('RSI_PERIOD')}  "
          f"MACD={params.get('MACD_FAST')}/{params.get('MACD_SLOW')}  "
          f"Boll={params.get('BOLL_WINDOW')}/{params.get('BOLL_STD_MULT')}")


def _resolve_backtest_date() -> str:
    """解析回測切割日期：None 時自動計算為一年前最近的交易日"""
    if BACKTEST_DATE is not None:
        return str(BACKTEST_DATE)

    candidate = datetime.date.today() - datetime.timedelta(days=365)
    
    try:
        from taiwan_holidays.taiwan_calendar import TaiwanCalendar
        th = TaiwanCalendar()
        check_holiday = th.is_holiday
    except Exception:
        check_holiday = lambda d: False
        
    for _ in range(10):
        weekday = candidate.weekday()          # 0=週一 … 6=週日
        is_holiday = check_holiday(candidate)  
        if weekday < 5 and not is_holiday:     # 非週末 且 非假日
            break
        candidate -= datetime.timedelta(days=1)

    result = candidate.strftime("%Y%m%d")
    print(f"  [自動回測日期] 一年前最近交易日: {result}")
    return result


def step1_optimize(bt: str):
    """步驟 1: 執行 Optuna 貝葉斯最佳化"""
    import scripts.optimize_factors as of_module

    # 覆寫最佳化模組的設定 (統一由 config.py 控制)
    of_module.MAX_ITERATIONS = OPTIMIZATION_TRIALS
    of_module.EARLY_STOPPING_ROUNDS = EARLY_STOPPING_ROUNDS
    of_module.BACKTEST_DATE  = bt
    of_module.N_JOBS = OPTUNA_N_JOBS
    print(f"  [注意] 最佳化期間進度輸出可能因多執行緒而順序不一，屬正常現象")
    print(f"  開始 Optuna 貝葉斯最佳化，共 {OPTIMIZATION_TRIALS} 輪...")
    if EARLY_STOPPING_ROUNDS:
        print(f"  (啟用 Early Stopping: 連續 {EARLY_STOPPING_ROUNDS} 輪無進展則提早結束)")
    print(f"  回測切割日期: {bt} (訓練/評估分界點)")
    print(f"  (每 50 輪或出現新最佳解時顯示進度)")
    of_module.main()


def step2_load_params() -> dict:
    """步驟 2: 讀取最佳參數並套用到特徵工程模組"""
    if not os.path.exists(BEST_FACTORS_PATH):
        print("  [提示] 找不到 best_factors.json，將不會覆寫 feature_engineering，直接使用其預設因子參數繼續。")
        return {}

    with open(BEST_FACTORS_PATH, "r", encoding="utf-8") as f:
        result = json.load(f)

    params = result.get("best_params_for_run_feature_engineering", {})
    best_score = result.get("best_score_avg", 0)
    optimized_at = result.get("optimized_at", "未知")

    print(f"  最佳化時間  : {optimized_at}")
    print(f"  歷史最佳勝率: {best_score:.2f}%")
    _apply_best_params(params)
    return params


def _get_training_stocks() -> list:
    """根據 TRAIN_INDUSTRIES 設定與 Stocks.txt 組合出要訓練的股票清單"""
    target_stocks = set(load_target_stocks("Stocks.txt"))  # Stocks.txt 一定要包含
    cat_path = os.path.join(BASE_DIR, "stock_categories.json")
    
    if os.path.exists(cat_path):
        with open(cat_path, "r", encoding="utf-8") as f:
            categories = json.load(f)
            
        for ind_name, is_enabled in TRAIN_INDUSTRIES.items():
            if is_enabled and ind_name in categories:
                for stock_id in categories[ind_name].keys():
                    target_stocks.add(stock_id)
    else:
        print("  [警告] 找不到 stock_categories.json，只使用 Stocks.txt 的股票。")
        
    return sorted(list(target_stocks))


def step3_feature_engineering():
    """步驟 3: 重建特徵矩陣"""
    train_stocks = _get_training_stocks()
    print(f"  目標訓練股票: {len(train_stocks)} 檔")
    if len(train_stocks) <= 20:
        print(f"  清單: {train_stocks}")
    else:
        print(f"  清單: {train_stocks[:10]} ... 等 {len(train_stocks)} 檔")
        
    fe_module.N_JOBS = FEAT_N_JOBS
    process_all_history_features(START_DATE, END_DATE, override_target_stocks=train_stocks)
    return train_stocks


def step4_train():
    """步驟 4: 訓練 LightGBM 模型"""
    import scripts.train as train_module
    train_module.N_JOBS = TRAIN_N_JOBS
    train_module.main()


def step5_inference():
    """步驟 5: 推理預測"""
    import scripts.inference as inference_module
    inference_module.main()


def main():
    parser = argparse.ArgumentParser(description="一鍵自動化量化交易流水線")
    parser.add_argument(
        "-s", "--step", 
        type=str, 
        choices=["optimize", "o", "feature", "f", "train", "t", "inference", "i", "all", "a"], 
        default="all",
        help="執行單一指定步驟 (optimize/o | feature/f | train/t | inference/i | all/a)"
    )
    args = parser.parse_args()

    # 簡碼對齊映射
    step_map = {
        "o": "optimize",
        "f": "feature",
        "t": "train",
        "i": "inference",
        "a": "all",
        "optimize": "optimize",
        "feature": "feature",
        "train": "train",
        "inference": "inference",
        "all": "all"
    }
    target_step = step_map[args.step]

    t_total_start = time.time()

    print("=" * 65)
    print("  一鍵自動化量化交易流水線 (Auto Pipeline)")
    print("=" * 65)
    print(f"  執行時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  目標步驟: {target_step.upper()}")

    bt = _resolve_backtest_date()

    if target_step == "optimize":
        _banner("1", f"單獨執行：貝葉斯因子最佳化 (Optuna TPE, {OPTIMIZATION_TRIALS} 輪)")
        # 最佳化前需要建立基礎特徵，供 Optuna 讀取
        step2_load_params()
        step3_feature_engineering()
        step1_optimize(bt)

    elif target_step == "feature":
        _banner("3", "單獨執行：重建特徵矩陣 (Feature Engineering)")
        step2_load_params()
        step3_feature_engineering()

    elif target_step == "train":
        _banner("4", "單獨執行：訓練 LightGBM 模型 (Day 1 ~ Day 3)")
        step4_train()

    elif target_step == "inference":
        _banner("5", "單獨執行：推理預測 — 未來 3 天走勢排行榜與下單建議")
        step5_inference()

    else:
        # 預設執行完整流程
        if RUN_OPTIMIZATION:
            # 最佳化前置準備
            _banner("0", "最佳化前置準備：以現有參數重建最新特徵矩陣")
            t0 = time.time()
            step2_load_params()
            step3_feature_engineering()
            print(f"  [耗時] {time.time()-t0:.1f} 秒")

            # 執行最佳化
            _banner("1", f"貝葉斯因子最佳化 (Optuna TPE, {OPTIMIZATION_TRIALS} 輪, 回測切割={bt})")
            t0 = time.time()
            step1_optimize(bt)
            print(f"  [耗時] {time.time()-t0:.1f} 秒")

            # 最佳化後，載入新參數
            _banner("2", "載入最新最佳因子參數")
            t0 = time.time()
            step2_load_params()
            print(f"  [耗時] {time.time()-t0:.1f} 秒")

            # 使用新參數重新特徵工程
            _banner("3", "以最佳因子參數重建最終特徵矩陣")
            t0 = time.time()
            step3_feature_engineering()
            print(f"  [耗時] {time.time()-t0:.1f} 秒")
        else:
            _banner("1", "跳過因子最佳化 (RUN_OPTIMIZATION=False)")
            print("  直接使用上次最佳化的 best_factors.json")

            _banner("2", "載入最佳因子參數")
            t0 = time.time()
            step2_load_params()
            print(f"  [耗時] {time.time()-t0:.1f} 秒")

            _banner("3", "重建特徵矩陣 (Feature Engineering)")
            t0 = time.time()
            step3_feature_engineering()
            print(f"  [耗時] {time.time()-t0:.1f} 秒")

        _banner("4", "訓練 LightGBM 模型 (Day 1 ~ Day 3)")
        t0 = time.time()
        step4_train()
        print(f"  [耗時] {time.time()-t0:.1f} 秒")

        _banner("5", "推理預測 — 未來 3 天走勢排行榜")
        t0 = time.time()
        step5_inference()
        print(f"  [耗時] {time.time()-t0:.1f} 秒")

        # 總結
        total = time.time() - t_total_start
        print()
        print("=" * 65)
        print(f"  [流水線完成] 總耗時: {total/60:.1f} 分鐘")
        print(f"  模型已儲存至 : models/")
        print(f"  特徵已儲存至 : data/features/features_combined.parquet")
        print(f"  最佳因子存於 : best_factors.json")
        print("=" * 65)


if __name__ == "__main__":
    main()
