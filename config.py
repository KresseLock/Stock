# -*- coding: utf-8 -*-
"""
config.py — 台灣股市量化交易系統中央控制面板
=========================================
本檔案為專案唯一設定檔，所有常數、參數與調參範圍均在此宣告。
嚴禁在個別腳本中進行分散式寫死。

快速索引（哪個節影響哪支腳本）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 1    → scraper.py（limited 模式篩選）、train.py（訓練產業）
  § 2    → trading_sim.py、inference.py（核心買賣／停損／手續費）
  § 2.1  → trading_sim.py、inference.py（大盤避險紅燈 + 移動止盈）
  § 2.2  → trading_sim.py、inference.py（ATR 動態停損）
  § 2.3  → trading_sim.py、inference.py（Regime 動態買入門檻）
           ※ 僅在「未顯式指定 buy_threshold」時生效
           ※ CLI --buy_threshold 與 param_sensitivity.py 靜態掃描不受影響
  § 2.4  → optimize_trading_params.py（評分函式權重）
           ※ 改動後須重跑優化才生效（同時使風控 checkpoint 失效）
  § 2.5  → inference.py、trading_sim.py（限價掛單動態加價幅度）
  § 3    → scraper.py、Auto_RUN.py（爬蟲時間區間設定）
  § 4    → auto_pipeline.py、run_workflow_experiment.py、
           train.py、backtest.py（流程控制與 ML 訓練切分點）
  § 5    → scripts/StockSync.py、Auto_RUN.py（雲端備份）
  § 6    → train.py、backtest.py、optimize_factors.py（LightGBM 超參數）
  § 7    → train.py、backtest.py、optimize_factors.py（樣本權重、時間衰減）
           feature_engineering.py（MOMENTUM_WINDOWS 動能特徵週期）
  § 8    → feature_engineering.py（混合標籤設計）
  § 9    → trading_sim.py（CLI 預設值，非風控邏輯）
  § 10   → optimize_factors.py（Optuna 技術指標搜尋邊界）
  § 10.5 → optimize_trading_params.py（Optuna 風控搜尋邊界）
  § 11   → 自動加載（啟動時執行，覆寫 § 2～§ 2.3 預設值，勿修改）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import datetime
import os
import json
import multiprocessing


# ── 1. 訓練產業篩選 ──────────────────────────────────────────────
# *** scraper.py（limited 模式）、train.py（訓練集篩選）使用 ***
# True = 納入 LightGBM 訓練集；自選股 (Stocks.txt) 必定保留，不受此限。
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


# ── 2. 交易策略核心參數 ───────────────────────────────────────────
# *** trading_sim.py、inference.py 共用 ***
# 注意：stop_loss 在 ATR_STOP_ENABLED=True 時由 § 2.2 動態計算覆蓋，此值作為 fallback。
BUY_THRESHOLD    = 12.0      # Day1 多空淨分數達此值才觸發買進 (%)
SELL_THRESHOLD   = 0.0       # Day3 多空淨分數低於此值觸發賣出 (%)
STOP_LOSS_PCT    = -8.0      # 固定個股停損 (%)；ATR_STOP_ENABLED=True 時被動態值覆蓋
MAX_POSITIONS    = 5         # 最大持倉上限檔數
MIN_HOLD_DAYS    = 20        # 最少持股天數（防止頻繁交易）
ORDER_MARKUP_PCT = -2.0      # 預設掛單溢價 (%)，負數 = 折價逢低買進；None = 改用 D1 信心動態加價

# ★ 交易摩擦成本唯一定義處 ★ 全系統手續費以此為準（trading_sim.py 買賣兩端皆引用）。
#   未來若券商手續費調降／談到折讓（如 6 折），只改這一行即可，無須動任何腳本。
#   例：6 折 ⇒ FEE_RATE = 0.001425 * 0.6 = 0.000855
FEE_RATE         = 0.001425  # 單邊券商手續費（0.1425%，定價未打折）
TAX_RATE         = 0.003     # 賣出證交稅（0.3%，非當沖）
LOT_SIZE         = 1000      # 整股交易單位（1 張 = 1000 股）；回測買進股數無條件捨去到此倍數
ODD_LOT_ENABLED  = False     # True=買不起整張的高價股退而用零股買進(吃揭示賣價)；False=純整張(買不起整張即跳過)


# ── 2.1 系統性大盤避險與移動止盈 ─────────────────────────────────
# *** trading_sim.py、inference.py 共用 ***
MKT_PANIC_MA5     = -0.010  # 大盤 5 日滾動均報酬避險門檻（-1.0% → -0.010）
MKT_PANIC_BREADTH = 0.30    # 全市場上漲家數比例避險門檻（30% → 0.30）
                             # ※ Bull regime 下此 breadth 紅燈停用，僅 Sideways/Bear 啟用
TS_ACTIVATION_PCT = 10.0    # 個股浮盈達此 % 才開啟移動止盈
TS_PULLBACK_PCT   = -6.0    # 開啟移動止盈後，自最高收盤回撤此 % 出場


# ── 2.2 ATR 動態停損 ──────────────────────────────────────────────
# *** trading_sim.py、inference.py 共用 ***
# True  → 停損 = 買入成本 × (1 - ATR_STOP_MULTIPLIER × atr18_pct)，依個股波動自適應
# False → 沿用 § 2 的固定 STOP_LOSS_PCT，行為與舊版完全相同
ATR_STOP_ENABLED     = True
ATR_STOP_MULTIPLIER  = 1.5    # 停損距離 = N 倍 ATR（越大越寬鬆）
ATR_STOP_FLOOR_PCT   = -15.0  # ATR 停損絕對下限（防低流動性股異常 ATR 導致停損過寬）
ATR_STOP_CEILING_PCT = -5.0   # ATR 停損絕對上限（防牛市正常回撤即觸發停損）

# ── 2.3a 動能混合排序確認天數（Hysteresis）────────────────────
# *** trading_sim.py、inference.py 共用 ***
# Bull regime 需連續達到此天數，才啟用 30/70 動能混合排序；
# 任何一天非 Bull 立即重置。目的：過濾熊牛轉換振盪假突破。
MOMENTUM_BULL_CONFIRM_DAYS = 4


# ── 2.3 市況過濾器：Regime 動態買入門檻（趨勢市進攻、震盪/空頭防守）──
# *** trading_sim.py、inference.py 共用 ***
# 重要：僅在「未顯式指定 buy_threshold」時生效。
#       CLI 覆寫（--buy_threshold）與 param_sensitivity.py 靜態掃描走固定值，不受影響。
REGIME_ADAPTIVE_ENABLED = True
REGIME_BUY_THRESHOLD = {
    "Bull":     5.0,    # 趨勢多頭：低門檻積極進攻
    "Sideways": 21.5,   # 震盪盤整：高門檻防守（滿倉必爆，高門檻為最不爛解）
    "Bear":     99.0,   # 空頭：實質空倉持現金
}
# Regime 選擇性曝險：依昨日 regime 動態調整最大持股檔數（時間性降曝險）。
# 與 REGIME_BUY_THRESHOLD 同步——僅在「未顯式指定 buy_threshold」時生效。
# 動機：門檻只控制「要不要開新倉」，此處控制「曝險量」。Bull 維持滿倉吃肥尾，
#       Sideways/Bear 降檔數以化解「滿倉必爆」，不強制平倉（靠自然汰換降至上限，避免振盪洗價）。
REGIME_MAX_POSITIONS = {
    "Bull":     5,    # 趨勢多頭：滿倉集中吃肥尾（與 MAX_POSITIONS 一致）
    "Sideways": 3,    # 震盪盤整：降曝險
    "Bear":     1,    # 空頭：極低曝險（實質近空手）
}
# 進場端 Bull 確認天數：Bull 需連續 N 天才在「進場端」生效（買入門檻／檔數上限／breadth 紅燈豁免），
# 未確認前進場端視同 Sideways；出場端（EXIT_REGIME_LAG 遲滯）與動能混合（MOMENTUM_BULL_CONFIRM_DAYS）不受影響。
# 動機：regime 閃爍時「Bull 第 1 天」門檻驟降湧入低分股是回撤主因（2025H2 回撤段 22/38 筆進場、−49 萬）。
# 已通過多起點＋多窗口驗證（2026-07-04，30 條回測）：回撤段修復 5/5 路徑成立、全期報酬中位數 +44pp、
# MDD 改善中位數 +17pp；唯一系統性成本＝健康期（2024 型）中位數 −15.5pp 保險費。N=2 為劑量最優
# （事後歸因：Bull 第 1 天進場全期 −27.5 萬、第 2 天起 +89.3 萬，故 N≥3 會誤殺贏家）。
# None 或 0 = 停用（回到舊行為）。trading_sim.py 與 inference.py 共用。
ENTRY_BULL_CONFIRM_DAYS = 2

# 市況分類以大盤 REGIME_TREND_WINDOW 日滾動均日報酬為主軸（不用 breadth 判 Bull，
# 因 2026 為權值股窄牛市，breadth 低但趨勢強，用 breadth 會嚴重低估多頭環境）
REGIME_BULL_TREND        = 0.0015  # 滾動均日報酬 > 此值 → Bull（趨勢多頭）
REGIME_BEAR_TREND        = -0.002  # 滾動均日報酬 < 此值 → Bear，其餘為 Sideways
REGIME_TREND_WINDOW      = 10      # 滾動視窗天數（20→10：去滯後，牛市起漲時不被誤判 Sideways）
REGIME_TREND_MIN_PERIODS = 5       # 最少有效天數（不足時退回 Sideways）

# ── 2.3.1 市況自適應出場參數（Regime-Adaptive Exit）──────────────────────────
# *** trading_sim.py、inference.py 共用；optimize_trading_params.py 優化時停用 ***
# EXIT_REGIME_LAG：切出 Bull 出場模式所需的連續非 Bull 天數（切入 Bull 立即生效）
# 不對稱遲滯：防止牛市短暫整理誤觸保守出場，同時在真正熊市 7 天後自動切換保護。
EXIT_REGIME_LAG = 7
REGIME_EXIT_PARAMS = {
    "Bull": {
        "sell_threshold": -8.5,   # 牛市訊號出場門檻（高容忍，讓贏家跑）
        "ts_activation":  20.5,   # 浮盈達 +20.5% 才啟動移動止盈（捕捉完整趨勢）
        "ts_pullback":    -9.5,   # 自高點回撤 9.5% 止盈
        "min_hold_days":  23,     # 最少持股 23 天（牛市趨勢持倉）
    },
    "Sideways": {
        "sell_threshold": -18.5,  # 橫盤出場門檻
        "ts_activation":  8.0,    # 橫盤及早保護利潤
        "ts_pullback":    -7.5,
        "min_hold_days":  11,
    },
    "Bear": {
        "sell_threshold": -14.0,  # 熊市主動出場（-14% 比橫盤 -18.5% 更積極）
        "ts_activation":  7.0,    # 熊市小利即保
        "ts_pullback":    -5.5,
        "min_hold_days":  5,      # 最短持倉，快速換手
    },
}


# ── 2.4 Optuna 風控調參評分函式權重 ──────────────────────────────
# *** optimize_trading_params.py 使用 ***
# 評分邏輯：combined_score = Σregime[ALPHA·alpha + SPREAD·spread + CALMAR·calmar]
#           − MDD_PENALTY_WEIGHT · max(0, 全期MDD% − MDD_TOLERANCE)
# 修改後須重跑優化才生效（同時使風控 checkpoint 失效，見 run_workflow_experiment.py）
PORTFOLIO_ALPHA_WEIGHT  = 0.6    # per-regime 組合 Alpha 權重（核心）
PORTFOLIO_SPREAD_WEIGHT = 0.2    # per-regime 多空 Spread 權重（輔助）
CALMAR_SCORE_WEIGHT     = 0.2    # per-regime Calmar 比率權重（想重視回撤可調高）
MDD_TOLERANCE           = 20.0   # 全期 MDD 容忍線 (%)，超過才扣分
MDD_PENALTY_WEIGHT      = 0.05   # 每超出 1% MDD 的線性扣分權重（設 0 停用）


# ── 2.5 限價掛單動態加價幅度 ─────────────────────────────────────
# *** inference.py、trading_sim.py 共用；兩支腳本須保持一致 ***
ORDER_MARKUP_HIGH_SCORE = 30.0  # D1 分數 ≥ 此值 → 使用高加價幅度
ORDER_MARKUP_MID_SCORE  = 20.0  # D1 分數 ≥ 此值 → 使用中加價幅度
ORDER_MARKUP_HIGH_PCT   = 2.5   # 高信心加價幅度 (%)
ORDER_MARKUP_MID_PCT    = 2.0   # 中信心加價幅度 (%)
ORDER_MARKUP_LOW_PCT    = 1.5   # 標準加價幅度 (%)


# ── 3. 數據爬蟲與時間區間 ────────────────────────────────────────
# *** scraper.py、Auto_RUN.py 使用 ***
FINMIND_FETCH_MODE = "limited"          # "limited" = 僅下載 TRAIN_INDUSTRIES=True 產業（省 Token）
                                         # "all" = 全市場下載
START_DATE         = datetime.date(2020, 1, 1)  # 數據回溯起點（建議至少 5 年）
FINMIND_CACHE_DAYS = 15                  # FinMind 基本面資料快取更新間隔天數
GHOST_DATA_PCT_THRESHOLD = 0.15          # 幽靈資料/極端價格跳空判定門檻 (15%)
FINMIND_MAX_LIMIT_WAITS = 6              # 單次呼叫遇 429/402 限速時，最多等待重置幾次（每次約 1 小時）；超過視為額度耗盡放棄，避免無人值守時無限卡住



# ── 4. 機器學習流程控制 ───────────────────────────────────────────
# *** auto_pipeline.py、run_workflow_experiment.py、train.py、backtest.py 使用 ***
RUN_OPTIMIZATION      = False       # 是否在流水線執行時重新啟動 Optuna 調參
OPTIMIZATION_TRIALS   = 600         # Optuna 最佳化最大輪數
EARLY_STOPPING_ROUNDS = 200         # Optuna Early Stopping 輪數（None = 不提早結束）
BACKTEST_DATE         = "20250801"  # 訓練／測試切分分界點（OOS 評估起點）
                                    # 設為 None → 模式 B（滾動重訓，用於實盤生產）

FEAT_N_JOBS   = -1   # 特徵工程平行核心數（-1 = 最大）
TRAIN_N_JOBS  = -1   # LightGBM 訓練核心數
OPTUNA_N_JOBS = 6    # Optuna 並行搜尋線程數


# ── 5. 雲端備份設定 ───────────────────────────────────────────────
# *** scripts/StockSync.py、Auto_RUN.py 使用 ***
RCLONE_REMOTE_NAME = "StockSync"  # rclone 遠端設定名稱
RCLONE_DEST_PATH   = "StockData"  # Google Drive 目標資料夾名稱


# ── 6. LightGBM 模型超參數 ────────────────────────────────────────
# *** train.py、backtest.py、optimize_factors.py 共用 ***
LGBM_N_ESTIMATORS        = 300   # 正式訓練樹數量（train.py / backtest.py）
LGBM_OPTUNA_N_ESTIMATORS = 100   # Optuna 評估用精簡樹數量（加速搜尋）
LGBM_LEARNING_RATE       = 0.03  # 學習率
LGBM_MAX_DEPTH           = 4     # 最大樹深
LGBM_NUM_LEAVES          = 15    # 最大葉節點數
LGBM_SUBSAMPLE           = 0.8   # 每棵樹的樣本採樣比例
LGBM_COLSAMPLE           = 0.8   # 每棵樹的特徵採樣比例
LGBM_EARLY_STOPPING      = 30    # 訓練早停輪數（train.py / backtest.py）
LGBM_BT_EARLY_STOPPING   = 20    # 時光機回測早停輪數（backtest.py 專用）


# ── 7. 訓練樣本權重、時間衰減與特徵控制 ───────────────────────────
# *** train.py、backtest.py、optimize_factors.py 共用 ***
# *** MOMENTUM_WINDOWS 同時由 feature_engineering.py 讀取 ***

# 大跌懲罰：未來 3 日最低報酬 <= 門檻時，樣本權重乘以懲罰倍數，強制模型迴避大跌股
SAMPLE_WEIGHT_DROP_THRESHOLD = -0.05  # 跌幅門檻（-5%）
SAMPLE_WEIGHT_PENALTY        = 2.0    # 懲罰倍數

# 時間衰減：近期樣本權重較高，舊樣本指數衰減；半衰期 = ln(2) / lambda ≈ 346 天
DEFAULT_DECAY_LAMBDA = 0.0            # 預設衰減係數（0 = 不衰減；0.005 ≈ 半年半衰期）
DECAY_LAMBDA_GRID    = [0.0, 0.001, 0.002, 0.003, 0.005]  # optimize_factors.py 搜尋網格

# IC 反轉因子排除：填入欄位名稱後下次訓練生效；實驗確認效果前保持空列表
EXCLUDE_FEATURES = []

# 動能特徵週期：同時計算 ret{w}（多週期報酬率）與 RS_{w}d（相對大盤強弱）
# 修改後需重跑 auto_pipeline.py -s f 重建 parquet，再重新訓練
MOMENTUM_WINDOWS = [3, 10, 20]


# ── 8. 混合標籤設計（方案 C） ─────────────────────────────────────
# *** feature_engineering.py 使用 ***
LABEL_STRONG_QUANTILE = 0.80   # 強勢股 (label=2)：橫截面前 20% 排名門檻
LABEL_WEAK_QUANTILE   = 0.20   # 弱勢股 (label=0)：橫截面後 20% 排名門檻
LABEL_STRONG_MIN_RET  = 0.00   # 強勢股絕對報酬必須 > 此值（崩盤日防線，Class 2 不得負報酬）
LABEL_WEAK_MAX_RET    = -0.02  # 跌幅超過此值強制歸弱勢（Class 0 大跌防線）


# ── 9. 交易回測 CLI 預設值 ─────────────────────────────────────────
# *** trading_sim.py 使用（僅供 CLI --start/--end/--capital 的預設值，非風控邏輯）***
SIM_DEFAULT_START   = "2026-01-01"
SIM_DEFAULT_END     = "2026-06-30"
SIM_DEFAULT_CAPITAL = 1_000_000


# ── 10. Optuna 因子搜尋邊界 ───────────────────────────────────────
# *** optimize_factors.py 使用 ***
# 格式：(最小值, 最大值)，均為整數範圍
OPTUNA_BOUNDS = {
    # 均線（短→中1→中2→長，偏移量設計確保嚴格遞增）
    "ma_short":         ( 3,  9),
    "ma_mid1_offset":   ( 1, 15),   # 中均線1 = ma_short + offset
    "ma_mid2_offset":   ( 1, 25),   # 中均線2 = ma_mid1  + offset
    "ma_long_offset":   (10, 80),   # 長均線  = ma_mid2  + offset
    # 振盪指標
    "rsi_period":       ( 7, 30),
    "kd_period":        ( 5, 20),
    "atr_period":       ( 7, 28),
    # MACD（慢線 = 快線 + 偏移量，保證 fast < slow）
    "macd_fast":        ( 6, 18),
    "macd_slow_offset": ( 5, 35),
    "macd_signal":      ( 5, 15),
    # 布林通道
    "boll_window":      (10, 35),
    "boll_std_x100":    (150, 300),  # 實際值 = boll_std_x100 / 100.0（1.5 ~ 3.0）
    # 成交量均線
    "vol_ma":           ( 3, 15),
    # 籌碼視窗（偏移量設計）
    "chips_w1":         ( 2,  7),
    "chips_w2_offset":  ( 2, 10),   # w2 = w1 + offset
    "chips_w3_offset":  ( 5, 20),   # w3 = w2 + offset
}


# ── 10.5 Optuna 交易風控搜尋邊界 ─────────────────────────────────
# *** optimize_trading_params.py 使用 ***

# 風控調參預設區間。起點統一（optimize_trading_params.py 預設與
# run_workflow_experiment.py 三階段共用）；終點依用途分流：
#   optimize_trading_params.py 預設 → TRADING_OPT_END_DATE（留其後區間當
#     「候選 vs 現行參數」的公平裁判區，候選檔經對比勝出才部署）
#   run_workflow_experiment.py     → mode A／潔淨 OOS 用 BACKTEST_DATE，mode B 用最新資料日
TRADING_OPT_START_DATE = "2022-01-02"  # 含 2022 完整熊市（-32%），風控參數的核心壓力樣本
TRADING_OPT_END_DATE   = "2025-12-31"  # 裁判區起點的前一日（2026-01 起保留為對比驗證用）

# 格式：3-tuple = float 搜尋空間（最小, 最大, 步長）；2-tuple = int 搜尋空間（最小, 最大）
TRADING_PARAM_BOUNDS = {
    "buy_threshold":        (5.0,    25.0,   0.5),
    "sell_threshold":       (-20.0,   5.0,   0.5),
    "stop_loss":            (-15.0,  -3.0,   0.5),
    "panic_ma5":            (-0.025,  0.00,  0.001),
    "panic_breadth":        (0.15,    0.45,  0.01),
    "ts_activation":        (5.0,    25.0,   0.5),
    "ts_pullback":          (-12.0,  -1.0,   0.5),
    "min_hold_days":        (1, 25),
    "markup_pct":           (-3.0,    2.0,   0.5),
    # 市況過濾器參數（--regime 模式下搜尋，取代靜態 buy_threshold）
    "regime_bull_buy":      (0.0,    15.0,   0.5),
    # sideways_buy 下界 8.0 曾為「高門檻防守」保底，2026-07 優化貼死下界(8.0)；
    # 現由 regime_sideways_pos 控曝險，門檻可放行往下探索。
    "regime_sideways_buy":  (5.0,    25.0,   0.5),
    # 2026-07 優化 bull_trend=0.0035 距上界 0.004 僅一步，放寬供下輪探索。
    "regime_bull_trend":    (0.0005,  0.006, 0.0005),
    "regime_bear_trend":    (-0.004, -0.0005, 0.0005),
    # 市況選擇性曝險：各 regime 持股檔數上限（int）。Bull 亦納入搜尋——實測基準滿倉(5)
    # 過度曝險，模型第 4~5 名部位為淨拖累，降 Bull 檔數可同時改善報酬與回撤。
    "regime_bull_pos":      (1, 5),
    "regime_sideways_pos":  (1, 5),
    "regime_bear_pos":      (1, 5),
}


# ── 11. 自動加載最佳化交易風控參數（啟動時自動執行，勿修改）────────
# configs/best_trading_params.json 存在時，自動覆寫 § 2 ~ § 2.3 的預設值；
# 任何模組 import config 即觸發此段，無須手動呼叫。
_base_dir = os.path.dirname(os.path.abspath(__file__))
_best_params_path = os.path.join(_base_dir, "configs", "best_trading_params.json")
if os.path.exists(_best_params_path):
    try:
        with open(_best_params_path, "r", encoding="utf-8") as _f:
            _opt_data = json.load(_f)
            _params = _opt_data.get("best_params", {})

            if "buy_threshold" in _params:
                BUY_THRESHOLD = float(_params["buy_threshold"])
            if "sell_threshold" in _params:
                SELL_THRESHOLD = float(_params["sell_threshold"])
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
            if "min_hold_days" in _params:
                MIN_HOLD_DAYS = int(_params["min_hold_days"])
            if "markup_pct" in _params:
                markup_val = _params["markup_pct"]
                ORDER_MARKUP_PCT = float(markup_val) if markup_val is not None else None

            # 市況過濾器（由 --regime 模式的 optimize_trading_params.py 寫入）
            if "regime_bull_buy" in _params:
                REGIME_BUY_THRESHOLD = {
                    "Bull":     float(_params["regime_bull_buy"]),
                    "Sideways": float(_params.get("regime_sideways_buy", REGIME_BUY_THRESHOLD["Sideways"])),
                    "Bear":     float(_params.get("regime_bear_buy",     REGIME_BUY_THRESHOLD["Bear"])),
                }
            if "regime_bull_trend" in _params:
                REGIME_BULL_TREND = float(_params["regime_bull_trend"])
            if "regime_bear_trend" in _params:
                REGIME_BEAR_TREND = float(_params["regime_bear_trend"])
            # 市況選擇性曝險：各 regime 持股檔數上限。Bull 若無 regime_bull_pos（舊參數檔）
            # 則回退 MAX_POSITIONS（滿倉），確保向後相容。
            if "regime_sideways_pos" in _params:
                REGIME_MAX_POSITIONS = {
                    "Bull":     int(_params.get("regime_bull_pos", MAX_POSITIONS)),
                    "Sideways": int(_params["regime_sideways_pos"]),
                    "Bear":     int(_params.get("regime_bear_pos", REGIME_MAX_POSITIONS["Bear"])),
                }

            if multiprocessing.current_process().name == 'MainProcess':
                print(f"[系統提示] 偵測到 {os.path.basename(_best_params_path)}，已自動套用最佳化交易風控參數。")
    except Exception as _e:
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"[警告] 讀取最佳化交易參數檔失敗: {_e}")
