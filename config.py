# -*- coding: utf-8 -*-
"""
config.py — 台灣股市量化交易系統中央控制面板
=========================================
本檔案為專案唯一設定檔，所有常數、參數與調參範圍均在此宣告。
嚴禁在個別腳本中進行分散式寫死。
"""
import datetime

# ── 1. 訓練產業篩選 (原 train.py 設定) ────────────────────────
# 設定 True 代表要將該產業股票納入 LightGBM 訓練集。自選股 (Stocks.txt) 必定保留。
TRAIN_INDUSTRIES = {
    "半導體業": True,        "電子零組件業": True,      "電腦及週邊設備業": True,
    "光電業": True,          "電子通路業": True,        "其他電子業": True,
    "電子工業": True,        "通信網路業": True,        "資訊服務業": True,
    "電子商務業": True,     "生技醫療業": False,       "化學工業": False,
    "化學生技醫療": False,   "塑膠工業": False,         "橡膠工業": False,
    "電機機械": True,       "汽車工業": False,         "航運業": False,
    "鋼鐵工業": False,       "建材營造": False,         "玻璃陶瓷": False,
    "水泥工業": False,       "造紙工業": False,         "紡織纖維": False,
    "食品工業": False,       "農業科技業": False,       "農業科技": False,
    "貿易百貨": False,       "觀光事業": False,         "觀光餐旅": False,
    "金融保險": False,       "金融業": False,           "油電燃氣業": False,
    "綠能環保": False,       "綠能環保類": False,       "居家生活": False,
    "居家生活類": False,     "運動休閒": False,         "運動休閒類": False,
    "數位雲端": False,       "數位雲端類": False,       "文化創意業": False,
    "存託憑證": False,       "創新板股票": False,       "創新版股票": False,
    "ETF": False,            "other": False,            "其他": False,
}

# ── 2. 交易與實戰策略參數 (inference.py & trading_sim.py 共享) ──
BUY_THRESHOLD  = 10.0      # Day1 多空淨分數達此值才觸發買進 (%)
SELL_THRESHOLD = 0.0       # Day3 多空淨分數低於此值觸發賣出 (%)
STOP_LOSS_PCT  = -8.0      # 固定的個股停損趴數 (%)
MAX_POSITIONS  = 5         # 最大持倉上限檔數

FEE_RATE       = 0.001425  # 單邊券商手續費
TAX_RATE       = 0.003     # 賣出證交稅 (非當沖)

# ── 2.1 系統性風控與移動止盈參數 ───────────────────────────────
MKT_PANIC_MA5     = -0.010 # 大盤 5 日滾動平均報酬率避險門檻 (-1.0% 代表 -0.010)
MKT_PANIC_BREADTH = 0.30   # 全市場上漲家數比例避險門檻 (30% 代表 0.30)
TS_ACTIVATION_PCT = 10.0   # 個股浮盈達到此百分比才開啟移動止盈 (%)
TS_PULLBACK_PCT   = -6.0   # 啟動後自最高收盤價回撤此百分比執行止盈 (%)

# ── 2.2 限價掛單加價幅度 (inference.py & trading_sim.py 共享) ──
# 根據 D1 多空信心分數動態決定建議加價幅度，兩個腳本必須保持一致
ORDER_MARKUP_HIGH_SCORE = 30.0  # D1 分數 >= 此值時，使用高加價幅度
ORDER_MARKUP_MID_SCORE  = 20.0  # D1 分數 >= 此值時，使用中加價幅度
ORDER_MARKUP_HIGH_PCT   = 2.5   # 高信心加價幅度 (%)
ORDER_MARKUP_MID_PCT    = 2.0   # 中信心加價幅度 (%)
ORDER_MARKUP_LOW_PCT    = 1.5   # 標準加價幅度 (%)

# ── 3. 數據爬蟲與時間區間設定 ─────────────────────────────────
# limited: 只下載 TRAIN_INDUSTRIES=True 產業與自選股 (節省 Token) | all: 全市場
FINMIND_FETCH_MODE = "limited"
START_DATE = datetime.date(2020, 1, 1)  # 數據回溯起點 (建議至少 5 年)
FINMIND_CACHE_DAYS = 15                 # FinMind 基本面資料快取更新間隔天數

# ── 4. 機器學習與平行資源設定 ─────────────────────────────────
RUN_OPTIMIZATION      = True# 是否在流水線執行時重新啟動 Optuna 調參
OPTIMIZATION_TRIALS   = 600# Optuna 最佳化最大輪數
EARLY_STOPPING_ROUNDS = 250# Optuna Early Stopping 輪數 (None 代表不提早結束)
BACKTEST_DATE         = "20250801"# 訓練與測試的切分分界點 (樣本外評估起點)

FEAT_N_JOBS           = -1      # 特徵工程平行核心數 (-1 為最大核心)
TRAIN_N_JOBS          = -1      # LightGBM 訓練核心數
OPTUNA_N_JOBS         = 6       # Optuna 並行搜尋線程數

# ── 5. 雲端備份設定 ───────────────────────────────────────────
RCLONE_REMOTE_NAME    = "StockSync" # rclone 遠端設定名稱
RCLONE_DEST_PATH      = "StockData" # Google Drive 儲存之目標資料夾名稱

# ── 6. LightGBM 模型超參數 (train.py / backtest.py / optimize_factors.py 共享) ──
# 正式訓練模型 (train.py / backtest.py) 使用完整樹數，Optuna 評估使用精簡樹數
LGBM_N_ESTIMATORS        = 300    # 正式訓練樹數量
LGBM_OPTUNA_N_ESTIMATORS = 100    # Optuna 評估用精簡樹數量 (加速搜尋)
LGBM_LEARNING_RATE       = 0.03   # 學習率
LGBM_MAX_DEPTH           = 4      # 最大樹深
LGBM_NUM_LEAVES          = 15     # 最大葉節點數
LGBM_SUBSAMPLE           = 0.8    # 每棵樹的樣本採樣比例
LGBM_COLSAMPLE           = 0.8    # 每棵樹的特徵採樣比例
LGBM_EARLY_STOPPING      = 30     # LightGBM 訓練早停輪數 (用於 train.py / backtest.py)
LGBM_BT_EARLY_STOPPING   = 20     # 時光機回測 (backtest.py) 早停輪數

# ── 7. 樣本懲罰權重設定 (train.py / backtest.py / optimize_factors.py 共享) ──
# 若未來 3 天內最低報酬率 <= 此閾值，賦予懲罰倍數以抑止最大回撤
SAMPLE_WEIGHT_DROP_THRESHOLD = -0.05  # 跌幅門檻 (-5%)
SAMPLE_WEIGHT_PENALTY        = 2.0    # 懲罰權重倍數

# ── 8. 標籤設計參數 (feature_engineering.py) ─────────────────
# 方案 C：絕對與相對混合標籤設計
LABEL_STRONG_QUANTILE = 0.80   # 強勢股 (label=2) 橫截面百分位排名門檻 (前 20%)
LABEL_WEAK_QUANTILE   = 0.20   # 弱勢股 (label=0) 橫截面百分位排名門檻 (後 20%)
LABEL_STRONG_MIN_RET  = 0.00   # 強勢股絕對報酬率必須 > 此值 (空頭崩盤防線)
LABEL_WEAK_MAX_RET    = -0.02  # 絕對跌幅超過此值強制歸類弱勢 (大跌個股防線)

# ── 9. 交易回測預設參數 (trading_sim.py CLI 預設值) ──────────
SIM_DEFAULT_START   = "2026-01-01"  # 回測預設起始日期
SIM_DEFAULT_END     = "2026-06-30"  # 回測預設結束日期
SIM_DEFAULT_CAPITAL = 1_000_000     # 回測預設初始資金

# ── 10. Optuna 因子搜尋邊界 (optimize_factors.py) ─────────────
# 格式：(最小值, 最大值)，均為整數範圍
OPTUNA_BOUNDS = {
    # 均線 (短 → 中1 → 中2 → 長，採偏移量設計確保嚴格遞增)
    "ma_short":         ( 3,  9),   # 短均線絕對值
    "ma_mid1_offset":   ( 1, 15),   # 中均線1 = ma_short + offset
    "ma_mid2_offset":   ( 1, 25),   # 中均線2 = ma_mid1  + offset
    "ma_long_offset":   (10, 80),   # 長均線  = ma_mid2  + offset
    # 振盪指標
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

# ── 11. 自動加載並套用最佳化交易風控參數 (若有 best_trading_params.json) ──
import os
import json
import multiprocessing

_base_dir = os.path.dirname(os.path.abspath(__file__))
_best_params_path = os.path.join(_base_dir, "best_trading_params.json")
if os.path.exists(_best_params_path):
    try:
        with open(_best_params_path, "r", encoding="utf-8") as _f:
            _opt_data = json.load(_f)
            _params = _opt_data.get("best_params", {})
            
            if "buy_threshold" in _params:
                BUY_THRESHOLD = float(_params["buy_threshold"])
            if "stop_loss" in _params:
                STOP_LOSS_PCT = float(_params["stop_loss"])
            if "panic_ma5" in _params:
                MKT_PANIC_MA5 = float(_params["panic_ma5"])
            if "panic_breadth" in _params:
                MKT_PANIC_BREADTH = float(_params["panic_breadth"])
            if "ts_activation" in _params:
                TS_ACTIVATION_PCT = float(_params["ts_activation"])
            if "ts_pullback" in _params:
                TS_PULLBACK_PCT = float(_params["ts_pullback"])
                
            if multiprocessing.current_process().name == 'MainProcess':
                print(f"[系統提示] 偵測到 {os.path.basename(_best_params_path)}，已自動套用最佳化交易風控參數。")
    except Exception as _e:
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"[警告] 讀取最佳化交易參數檔失敗: {_e}")



