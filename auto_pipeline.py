"""
auto_pipeline.py — 一鍵自動化量化交易流水線
================================================================
執行順序:
  步驟 1. [可選] 貝葉斯因子最佳化 (optimize_factors.py)
           → 自動搜尋最佳 RSI/MA/MACD/布林通道等因子參數
           → 結果寫入 best_factors.json
  步驟 2. 自動讀取 best_factors.json，套用最佳參數
  步驟 3. 重建特徵矩陣 (run_feature_engineering.py)
           → 輸出 data/features/features_combined.parquet
  步驟 4. 訓練 LightGBM 模型 (train.py)
           → 輸出 models/lgbm_model_1.txt ~ lgbm_model_3.txt
  步驟 5. 推理預測輸出 (inference.py)
           → 印出未來 3 天走勢預測排行榜

使用方式:
  python auto_pipeline.py
"""

import os
import sys
import json
import datetime
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ╔══════════════════════════════════════════════════════╗
# ║              一鍵流水線設定區 (請自行調整)            ║
# ╚══════════════════════════════════════════════════════╝

# ── 電腦資源設定 (多核心運算) ──────────────────────────────
# -1 代表使用全部核心 (預設最快)
# 若怕運算時電腦卡頓，可以設定為具體的核心數量 (例如 4 或 8)
FEAT_N_JOBS       = -1   # 特徵工程平行運算 (影響最鉅)
TRAIN_N_JOBS      = -1   # LightGBM 訓練模型使用核心數
OPTUNA_N_JOBS     = 8    # 最佳化時同時啟動的 trial 數量 (建議 2~4 即可，太高會卡死)

# ── 步驟 1 設定：因子最佳化 ─────────────────────────────
# True  = 每次執行都重新跑貝葉斯最佳化 (耗時數分鐘~數十分鐘)
# False = 跳過最佳化，直接使用上一次的 best_factors.json
RUN_OPTIMIZATION    = True

# 若 RUN_OPTIMIZATION=False 但 best_factors.json 不存在，是否使用預設參數繼續？
FALLBACK_TO_DEFAULT = True

# 最佳化迭代次數 (此處可覆寫 optimize_factors.py 裡的 MAX_ITERATIONS)
OPTIMIZATION_TRIALS = 600

# 提早結束機制 (Early Stopping)：連續 N 輪未找到更好的解就提早結束 (None=不提早結束)
EARLY_STOPPING_ROUNDS = 300

# ── 回測切割日期 ────────────────────────────────────────
# 最佳化使用此日期之「前」的資料訓練，之「後」的資料評估勝率。
# 建議設為「約一年前」：樣本充足，又能反映近期市場規律。
# 設為 None 時，程式自動計算為「今日減一年」。
BACKTEST_DATE = "20250801"   # None = 自動設為一年前 | 或填入字串如 "20250101"

# ── 步驟 3 設定：特徵工程時間區間 ──────────────────────
START_DATE = datetime.date(2020, 1, 1)   # 歷史回溯起點
END_DATE   = datetime.date.today()       # 自動設為今日

# ── 步驟 3 設定：訓練產業選擇 ────────────────────────
# 設定 True 代表要拿該產業的「所有股票」加入訓練集 (讓模型學習通用規律)
# 設定 False 代表排除該產業 (加快訓練速度或排除不相關類股)
# 註：無論如何設定，Stocks.txt 裡面的自選股「一定」會加入訓練與最後的預測。
TRAIN_INDUSTRIES = {
    "半導體業": True,        "電子零組件業": True,      "電腦及週邊設備業": True,
    "光電業": True,          "電子通路業": True,        "其他電子業": True,
    "電子工業": False,       "通信網路業": False,       "資訊服務業": False,
    "電子商務業": False,     "生技醫療業": False,       "化學工業": False,
    "化學生技醫療": False,   "塑膠工業": False,         "橡膠工業": False,
    "電機機械": False,       "汽車工業": False,         "航運業": False,
    "鋼鐵工業": False,       "建材營造": False,         "玻璃陶瓷": False,
    "水泥工業": False,       "造紙工業": False,         "紡織纖維": False,
    "食品工業": False,       "農業科技業": False,       "農業科技": False,
    "貿易百貨": False,       "觀光事業": False,         "觀光餐旅": False,
    "金融保險": False,       "金融業": False,           "油電燃氣業": False,
    "綠能環保": False,       "綠能環保類": False,       "居家生活": False,
    "居家生活類": False,     "運動休閒": False,         "運動休閒類": False,
    "數位雲端": False,       "數位雲端類": False,       "文化創意業": False,
    "存託憑證": False,       "創新板股票": False,       "創新版股票": False,
    "ETF": False,            "其他電子類": False,       "其他": False
}

# ── 步驟 1&2 設定：最佳化結果檔路徑 ────────────────────
BEST_FACTORS_PATH = os.path.join(BASE_DIR, "best_factors.json")

# ╔══════════════════════════════════════════════════════╗
# ║              以下為流水線核心邏輯，一般不需修改        ║
# ╚══════════════════════════════════════════════════════╝

import scripts.feature_engineering as fe_module
from scripts.feature_engineering import process_all_history_features, load_target_stocks


def _banner(step: int, title: str):
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
    
    # 嘗試載入 taiwan_holidays，若未安裝則退化為只判斷週末
    try:
        from taiwan_holidays.taiwan_calendar import TaiwanCalendar
        th = TaiwanCalendar()
        check_holiday = th.is_holiday
    except Exception:
        check_holiday = lambda d: False
        
    # 往回找最近的交易日（最多找 10 天，避免無限迴圈）
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
    import optimize_factors as of_module

    # 覆寫最佳化模組的設定 (統一由 auto_pipeline.py 控制)
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
        if FALLBACK_TO_DEFAULT:
            print("  [提示] 找不到 best_factors.json，將不會覆寫 feature_engineering，直接使用其預設因子參數繼續。")
            return {}
        else:
            raise FileNotFoundError(
                f"找不到 {BEST_FACTORS_PATH}，請先執行最佳化或設定 FALLBACK_TO_DEFAULT=True"
            )

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
    target_stocks = set(load_target_stocks())  # Stocks.txt 一定要包含
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
        
    # 直接傳遞 override_target_stocks 給 feature_engineering
    fe_module.N_JOBS = FEAT_N_JOBS
    process_all_history_features(START_DATE, END_DATE, override_target_stocks=train_stocks)
        
    return train_stocks


def step4_train():
    """步驟 4: 訓練 LightGBM 模型"""
    import train as train_module
    train_module.N_JOBS = TRAIN_N_JOBS
    train_module.main()


def step5_inference():
    """步驟 5: 推理預測"""
    import inference as inference_module
    inference_module.main()


def main():
    t_total_start = time.time()

    print("=" * 65)
    print("  一鍵自動化量化交易流水線 (Auto Pipeline)")
    print("=" * 65)
    print(f"  執行時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  執行步驟: {'最佳化 → ' if RUN_OPTIMIZATION else '(跳過最佳化) → '}"
          f"載入參數 → 特徵工程 → 訓練 → 推理")

    # ── 步驟 1: 因子最佳化 ──────────────────────────────
    bt = _resolve_backtest_date()
    if RUN_OPTIMIZATION:
        _banner(1, f"貝葉斯因子最佳化 (Optuna TPE, {OPTIMIZATION_TRIALS} 輪, 回測切割={bt})")
        t0 = time.time()
        step1_optimize(bt)
        print(f"  [耗時] {time.time()-t0:.1f} 秒")
    else:
        _banner(1, "跳過因子最佳化 (RUN_OPTIMIZATION=False)")
        print("  直接使用上次最佳化的 best_factors.json")

    # ── 步驟 2: 載入最佳參數 ────────────────────────────
    _banner(2, "載入最佳因子參數")
    t0 = time.time()
    step2_load_params()
    print(f"  [耗時] {time.time()-t0:.1f} 秒")

    # ── 步驟 3: 特徵工程 ────────────────────────────────
    _banner(3, "重建特徵矩陣 (Feature Engineering)")
    t0 = time.time()
    step3_feature_engineering()
    print(f"  [耗時] {time.time()-t0:.1f} 秒")

    # ── 步驟 4: 訓練 ────────────────────────────────────
    _banner(4, "訓練 LightGBM 模型 (Day 1 ~ Day 3)")
    t0 = time.time()
    step4_train()
    print(f"  [耗時] {time.time()-t0:.1f} 秒")

    # ── 步驟 5: 推理 ────────────────────────────────────
    _banner(5, "推理預測 — 未來 3 天走勢排行榜")
    t0 = time.time()
    step5_inference()
    print(f"  [耗時] {time.time()-t0:.1f} 秒")

    # ── 總結 ─────────────────────────────────────────────
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
