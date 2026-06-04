# -*- coding: utf-8 -*-
"""
config.py — 台灣股市量化交易系統中央控制面板
=========================================
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

# ── 2. 交易與實戰策略參數 (原 inference.py & trading_sim.py 共享) ──
BUY_THRESHOLD  = 10.0      # Day1 多空淨分數達此值才觸發買進 (%)
SELL_THRESHOLD = 0.0       # Day3 多空淨分數低於此值觸發賣出 (%)
STOP_LOSS_PCT  = -8.0      # 固定的個股停損趴數 (%)
MAX_POSITIONS  = 5         # 最大持倉上限檔數

FEE_RATE       = 0.001425  # 單邊券商手續費
TAX_RATE       = 0.003     # 賣出證交稅 (非當沖)

# ── 2.1 系統性風控與移動止盈參數 (新版) ───────────────────────────
MKT_PANIC_MA5     = -0.010 # 大盤 5 日滾動平均報酬率避險門檻 (-1.0% 代表 -0.010)
MKT_PANIC_BREADTH = 0.30   # 全市場上漲家數比例避險門檻 (30% 代表 0.30)
TS_ACTIVATION_PCT = 10.0   # 個股浮盈達到此百分比才開啟移動止盈 (%)
TS_PULLBACK_PCT   = -6.0   # 啟動後自最高收盤價回撤此百分比執行止盈 (%)

# ── 3. 數據爬蟲與時間區間設定 (原 main.py & auto_pipeline.py 共享) ──
# limited: 只下載 TRAIN_INDUSTRIES=True 產業與自選股 (節省 Token) | all: 全市場
FINMIND_FETCH_MODE = "limited"
START_DATE = datetime.date(2020, 1, 1)  # 數據回溯起點 (建議至少 5 年)
FINMIND_CACHE_DAYS = 15                 # FinMind 基本面資料快取更新間隔天數 (如 15 天)

# ── 4. 機器學習與平行資源設定 (原 auto_pipeline.py & optimize_factors.py) ──
RUN_OPTIMIZATION      = False   # 是否在流水線執行時重新啟動 Optuna 調參
OPTIMIZATION_TRIALS   = 600     # Optuna 最佳化最大輪數
EARLY_STOPPING_ROUNDS = 200     # Optuna 提早結束輪數 (None 代表不提早結束)
BACKTEST_DATE         = "20250801" # 訓練與測試的切分分界點 (樣本外評估起點)

FEAT_N_JOBS           = -1      # 特徵工程平行核心數 (-1 為最大核心)
TRAIN_N_JOBS          = -1      # LightGBM 訓練核心數
OPTUNA_N_JOBS         = 6       # Optuna 並行搜尋線程數

# ── 5. 雲端備份設定 (原 StockSync.py) ───────────────────────────
RCLONE_REMOTE_NAME    = "StockSync" # rclone 遠端設定名稱
RCLONE_DEST_PATH      = "StockData" # Google Drive 儲存之目標資料夾名稱

