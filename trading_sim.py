import os
import sys
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 載入中央控制面板接單加價幅度與回測預設値 (CLI 預設不在 run_simulation 內装載) ──
try:
    from config import (
        ORDER_MARKUP_HIGH_SCORE, ORDER_MARKUP_MID_SCORE,
        ORDER_MARKUP_HIGH_PCT, ORDER_MARKUP_MID_PCT, ORDER_MARKUP_LOW_PCT,
        SIM_DEFAULT_START, SIM_DEFAULT_END, SIM_DEFAULT_CAPITAL,
        MAX_POSITIONS, REGIME_BULL_TREND, REGIME_BEAR_TREND,
        REGIME_TREND_WINDOW, REGIME_TREND_MIN_PERIODS,
    )
except ImportError:
    ORDER_MARKUP_HIGH_SCORE = 30.0; ORDER_MARKUP_MID_SCORE = 20.0
    ORDER_MARKUP_HIGH_PCT   = 2.5;  ORDER_MARKUP_MID_PCT   = 2.0; ORDER_MARKUP_LOW_PCT = 1.5
    SIM_DEFAULT_START = "2026-01-01"; SIM_DEFAULT_END = "2026-06-30"; SIM_DEFAULT_CAPITAL = 1_000_000
    MAX_POSITIONS = 5
    REGIME_BULL_TREND = 0.002; REGIME_BEAR_TREND = -0.002
    REGIME_TREND_WINDOW = 20; REGIME_TREND_MIN_PERIODS = 5

def get_tw_tick_size(price: float) -> float:
    if price < 10:
        return 0.01
    elif price < 50:
        return 0.05
    elif price < 100:
        return 0.1
    elif price < 500:
        return 0.5
    elif price < 1000:
        return 1.0
    else:
        return 5.0

def round_to_tick(price: float) -> float:
    if price <= 0:
        return 0.0
    tick = get_tw_tick_size(price)
    rounded = round(price / tick) * tick
    new_tick = get_tw_tick_size(rounded)
    if new_tick != tick:
        rounded = round(price / new_tick) * new_tick
    return rounded

def load_models():
    models = {}
    for days in [1, 2, 3]:
        model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days}.txt")
        if not os.path.exists(model_path):
            print(f"[錯誤] 找不到預先訓練的模型: {model_path}。請先執行 auto_pipeline.py 進行模型訓練與生成。")
            sys.exit(1)
        models[days] = lgb.Booster(model_file=model_path)
    return models

def run_simulation(start_date, end_date, initial_capital, max_positions,
                   mkt_panic_ma5=None, mkt_panic_breadth=None,
                   buy_threshold=None, sell_threshold=None, stop_loss_pct=None,
                   ts_activation_pct=None, ts_pullback_pct=None,
                   min_hold_days=None, markup_pct=None,
                   regime_buy_threshold=None, regime_bull_trend=None, regime_bear_trend=None,
                   regime_max_positions=None, regime_exit_params=None,
                   exit_regime_lag=None):
    print("=" * 70)
    print(f"  啟動量化交易回測 (Out-of-Sample, T+1 限價搓合 + 雙風控防線版)")
    print(f"  期間: {start_date} 到 {end_date}")
    print(f"  初始資金: {initial_capital:,} | 最大持股數: {max_positions}")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print("[錯誤] 找不到 features_combined.parquet")
        return

    # 1. 載入資料並過濾日期
    df = pd.read_parquet(DATA_PATH)

    # 收盤「最後揭示賣價」(字串型)正規化為數值，供高價股零股買進取成交價（吃賣單）。
    # 漲停無賣單時為 '0.00'，會在買進邏輯退回限價/開盤價。舊版 parquet 無此欄則為 NaN。
    if '最後揭示賣價' in df.columns:
        df['last_ask'] = pd.to_numeric(
            df['最後揭示賣價'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    else:
        df['last_ask'] = float('nan')

    # 計算每日大盤特徵與 20 日滾動均值，用以劃分 Regime
    df_daily_mkt = df.groupby("date")[["market_mean_pct", "market_breadth_pct"]].first().reset_index()
    df_daily_mkt["market_trend_20d"] = df_daily_mkt["market_mean_pct"].rolling(window=REGIME_TREND_WINDOW, min_periods=REGIME_TREND_MIN_PERIODS).mean()
    df_daily_mkt["market_breadth_20d"] = df_daily_mkt["market_breadth_pct"].rolling(window=REGIME_TREND_WINDOW, min_periods=REGIME_TREND_MIN_PERIODS).mean()
    
    # 用 np.select 劃分 Bull, Bear, Sideways (趨勢導向：以 market_trend_20d 為主軸)。
    # 不用 breadth 判 Bull，因 2026 為權值股窄牛市 (趨勢強但 breadth 低)，用 breadth 會漏判多頭。
    # 趨勢門檻可由外部覆寫 (供 optimize_trading_params.py 校準)，否則用 config 預設。
    _bull_trend = regime_bull_trend if regime_bull_trend is not None else REGIME_BULL_TREND
    _bear_trend = regime_bear_trend if regime_bear_trend is not None else REGIME_BEAR_TREND
    
    try:
        from scripts.utils import filter_stocks_by_train_industries, get_regime_label
    except ImportError:
        from utils import filter_stocks_by_train_industries, get_regime_label

    df_daily_mkt["regime"] = df_daily_mkt["market_trend_20d"].apply(lambda v: get_regime_label(v, _bull_trend, _bear_trend))
    
    # 把 regime 合併回 df
    df = df.merge(df_daily_mkt[["date", "market_trend_20d", "market_breadth_20d", "regime"]], on="date", how="left")
    
    # ── 根據 train.py 中的 TRAIN_INDUSTRIES 過濾股票 ──────────────────
    try:
        from scripts.utils import filter_stocks_by_train_industries
    except ImportError:
        from utils import filter_stocks_by_train_industries
    before_cnt = df["stock_id"].nunique()
    df = filter_stocks_by_train_industries(df)
    after_cnt = df["stock_id"].nunique()
    print(f"  [模擬過濾] 依 train.py 產業設定篩選：{before_cnt} → {after_cnt} 檔進行回測模擬")
    # ──────────────────────────────────────────────────────────
    
    try:
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as e:
        print("\n[日期格式錯誤] 您輸入的日期無法被解析！")
        print(" 請確認 --start 和 --end 使用標準格式，例如: 2026-01-02 或 2026/01/02")
        print(f"錯誤細節: {e}\n")
        return

    all_dates = sorted(df['date'].unique())
    if not all_dates:
        print("[錯誤] 資料庫中無任何交易日期。")
        return

    # 找出與 start_dt 和 end_dt 對應的資料庫日期索引
    start_idx = None
    for i, d in enumerate(all_dates):
        if d >= start_dt:
            start_idx = i
            break

    end_idx = None
    for i, d in enumerate(all_dates):
        if d <= end_dt:
            end_idx = i

    if start_idx is None or end_idx is None or start_idx > end_idx:
        min_date = all_dates[0].date()
        max_date = all_dates[-1].date()
        print(f"\n[查無資料] 在指定的區間 ({start_date} ~ {end_date}) 內沒有找到任何特徵資料！")
        print(f" 提示：目前資料庫中擁有的資料涵蓋範圍為 {min_date} 到 {max_date}。")
        return

    # 如果是資料庫的第一天，往前無訊號，自動向後順延一天
    if start_idx == 0:
        print("  [提示] 由於指定的起始日期是資料庫的第一天，無法取得前一日訊號。交易將從第二天開始執行。")
        start_idx = 1
        if start_idx > end_idx:
            print("[錯誤] 資料區間過短，無法執行回測。")
            return

    # 需要處理預測的日期區間：從訊號日 (start_idx - 1) 到結束日 (end_idx)
    needed_dates = all_dates[start_idx - 1 : end_idx + 1]
    df_sim = df[df['date'].isin(needed_dates)].copy()

    # 2. 決定特徵欄位並進行預測
    print("載入模型並進行全區間預測...")
    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    ignore_cols = ["stock_id", "date"] + target_cols
    
    feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        print(f"[錯誤] 找不到 {feature_cols_path}。請先執行 auto_pipeline.py。")
        return

    X_sim = df_sim.reindex(columns=feature_cols).astype(np.float32)
    models = load_models()

    for days in [1, 2, 3]:
        preds = models[days].predict(X_sim)
        if len(preds.shape) == 2 and preds.shape[1] == 3:
            prob_strong = preds[:, 2]
            prob_weak = preds[:, 0]
        else:
            print(f"[錯誤] 模型輸出格式不如預期 (preds.shape={preds.shape})，請確認是否為 3 類別分類模型！")
            sys.exit(1)
            
        df_sim[f'Day{days}_net'] = (prob_strong - prob_weak) * 100

    # 3. 模擬交易迴圈
    print("開始模擬每日交易...\n")
    
    available_cash = initial_capital  # 可用資金 (購買力)
    bank_cash = initial_capital       # 銀行帳戶實質餘額
    pending_settlements = {}          # 預計交割項目: date_obj -> net_amount (T+1/T+2 待交割款)

    positions = {}  # stock_id -> {'shares': X, 'buy_price': Y, 'buy_date': Z, 'max_close_price': W}
    history = []
    trades = []     # 交易明細紀錄
    
    # ── 載入中央控制面板 config ──────────────────────────────────
    try:
        from config import (
            BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS_PCT, MAX_POSITIONS,
            FEE_RATE, TAX_RATE, MKT_PANIC_MA5, MKT_PANIC_BREADTH,
            TS_ACTIVATION_PCT, TS_PULLBACK_PCT, MIN_HOLD_DAYS, ORDER_MARKUP_PCT,
            REGIME_ADAPTIVE_ENABLED, REGIME_BUY_THRESHOLD, REGIME_MAX_POSITIONS,
            ATR_STOP_ENABLED, ATR_STOP_MULTIPLIER, ATR_STOP_FLOOR_PCT, ATR_STOP_CEILING_PCT,
            MOMENTUM_BULL_CONFIRM_DAYS, LOT_SIZE, ODD_LOT_ENABLED,
            REGIME_EXIT_PARAMS as _CFG_REGIME_EXIT, EXIT_REGIME_LAG as _CFG_EXIT_LAG,
        )
    except ImportError:
        BUY_THRESHOLD     = 10.0
        SELL_THRESHOLD    = 0.0
        STOP_LOSS_PCT     = -8.0
        FEE_RATE          = 0.001425
        TAX_RATE          = 0.003
        MKT_PANIC_MA5     = -0.010  # 與 config.py 黃金風控參數一致 (-1.0%)
        MKT_PANIC_BREADTH = 0.30    # 與 config.py 黃金風控參數一致 (30%)
        TS_ACTIVATION_PCT = 10.0
        TS_PULLBACK_PCT   = -6.0
        MIN_HOLD_DAYS     = 1
        ORDER_MARKUP_PCT  = None
        REGIME_ADAPTIVE_ENABLED = False
        REGIME_BUY_THRESHOLD = {}
        REGIME_MAX_POSITIONS = {}
        ATR_STOP_ENABLED = False
        ATR_STOP_MULTIPLIER = 1.5
        ATR_STOP_FLOOR_PCT = -15.0
        ATR_STOP_CEILING_PCT = -3.0
        MOMENTUM_BULL_CONFIRM_DAYS = 5
        LOT_SIZE = 1000
        ODD_LOT_ENABLED = True
        _CFG_REGIME_EXIT = {}
        _CFG_EXIT_LAG = 1

    # 市況過濾器精度控管：僅在「未顯式指定 buy_threshold」時啟用動態門檻，
    # 避免破壞 CLI 覆寫與 param_sensitivity.py 的靜態掃描。
    # regime 買入門檻字典可由外部覆寫 (供 optimize_trading_params.py 校準)。
    _regime_buy_thr = regime_buy_threshold if regime_buy_threshold is not None else REGIME_BUY_THRESHOLD
    _regime_max_pos = regime_max_positions if regime_max_positions is not None else REGIME_MAX_POSITIONS
    use_regime_filter = (buy_threshold is None) and REGIME_ADAPTIVE_ENABLED
    if use_regime_filter:
        print(f"  [市況過濾器] 已啟用動態買入門檻 (依昨日 regime)：{_regime_buy_thr}")

    # 外部參數覆蓋
    if buy_threshold is not None:
        BUY_THRESHOLD = buy_threshold
    if sell_threshold is not None:
        SELL_THRESHOLD = sell_threshold
    if stop_loss_pct is not None:
        STOP_LOSS_PCT = stop_loss_pct
    if mkt_panic_ma5 is not None:
        MKT_PANIC_MA5 = mkt_panic_ma5
    if mkt_panic_breadth is not None:
        MKT_PANIC_BREADTH = mkt_panic_breadth
    if ts_activation_pct is not None:
        TS_ACTIVATION_PCT = ts_activation_pct
    if ts_pullback_pct is not None:
        TS_PULLBACK_PCT = ts_pullback_pct
    if min_hold_days is not None:
        MIN_HOLD_DAYS = min_hold_days
    if markup_pct is not None:
        ORDER_MARKUP_PCT = markup_pct
    # regime_exit_params=None → 從 config 載入（部署預設）
    # regime_exit_params={} → 明確停用（optimize_trading_params.py 優化時使用）
    if regime_exit_params is None:
        _regime_exit = _CFG_REGIME_EXIT
        if exit_regime_lag is None:
            exit_regime_lag = _CFG_EXIT_LAG
    else:
        _regime_exit = regime_exit_params
    if exit_regime_lag is None:
        exit_regime_lag = 1

    # 建立股票名稱對照表
    stock_names = {}
    cat_path = os.path.join(BASE_DIR, "scripts", "stock_categories.json")
    if os.path.exists(cat_path):
        try:
            with open(cat_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            for cat, stocks in categories.items():
                for sid, sname in stocks.items():
                    stock_names[str(sid)] = sname
        except Exception as e:
            print(f"[警告] 讀取 stock_categories.json 失敗: {e}")

    # 執行交易的日期範圍 (T+1 執行日)
    trading_dates = all_dates[start_idx : end_idx + 1]

    # Hysteresis 計數器：連續 Bull 天數，非 Bull 立即重置
    consecutive_bull_days = 0
    # 出場 regime 遲滯：切入 Bull 立即生效，切出需連續 N 天才換
    _exit_hysteresis = int(exit_regime_lag) if exit_regime_lag is not None else 1
    _exit_non_bull_streak = 0
    _exit_regime = 'Sideways'

    for idx_today, today in enumerate(trading_dates):
        start_trade_idx = len(trades)
        
        # 訊號日為前一個交易日
        prev_day = all_dates[start_idx - 1 + idx_today]
        
        # --- A0. 處理今日交割款 (T+2 餘額交割) ---
        today_date = today.date()
        settled_amount = 0
        for s_date in list(pending_settlements.keys()):
            if s_date <= today_date:
                settled_amount += pending_settlements.pop(s_date)
        bank_cash += settled_amount
        
        today_data = df_sim[df_sim['date'] == today].drop_duplicates(subset=['stock_id']).set_index('stock_id')
        prev_data = df_sim[df_sim['date'] == prev_day].drop_duplicates(subset=['stock_id']).set_index('stock_id')
        
        # 獲取今日市況
        regime = "Sideways"
        if not today_data.empty and 'regime' in today_data.columns:
            regime = today_data['regime'].iloc[0]

        # --- A1. 讀取大盤指標與判斷硬風控紅燈 (Market Sentiment Filter) ---
        mkt_ma5 = 0.0
        mkt_breadth = 1.0
        if not prev_data.empty:
            first_row = prev_data.iloc[0]
            if 'market_mean_ma5' in prev_data.columns:
                mkt_ma5 = first_row['market_mean_ma5']
            if 'market_breadth_pct' in prev_data.columns:
                mkt_breadth = first_row['market_breadth_pct']
                
        # 讀「昨日(訊號日)」regime，供 breadth 紅燈在 Bull 放寬與市況過濾器共用。
        # 刻意讀 prev_data 而非 today_data，因 today 的 regime 含今日市場報酬，用於今日買進屬前視偏差。
        signal_regime = None
        if not prev_data.empty and 'regime' in prev_data.columns:
            signal_regime = prev_data['regime'].iloc[0]

        # Hysteresis：連續 Bull 天數計數（非 Bull 立即重置，防止振盪期誤觸動能模式）
        if signal_regime == 'Bull':
            consecutive_bull_days += 1
        else:
            consecutive_bull_days = 0

        # 出場 regime 遲滯：切入 Bull 立即生效；切出需連續 N 天非 Bull 才切換
        if signal_regime == 'Bull':
            _exit_non_bull_streak = 0
            _exit_regime = 'Bull'
        elif signal_regime is not None:
            _exit_non_bull_streak += 1
            if _exit_non_bull_streak >= _exit_hysteresis:
                _exit_regime = signal_regime

        is_market_panic = False
        panic_reason = ""
        if mkt_ma5 < MKT_PANIC_MA5:
            is_market_panic = True
            panic_reason = f"大盤 5 日滾動平均報酬率過低 ({mkt_ma5 * 100:+.2f}%)，低於風控門檻 {MKT_PANIC_MA5 * 100:+.2f}%"
        elif mkt_breadth < MKT_PANIC_BREADTH and signal_regime != "Bull":
            # 窄牛市 (breadth 低但趨勢強) 不應被 breadth 紅燈擋下，故 Bull regime 停用 breadth 紅燈，僅 Sideways/Bear 啟用。
            is_market_panic = True
            panic_reason = f"全市場上漲比例過低 ({mkt_breadth * 100:.1f}%)，低於風控門檻 {MKT_PANIC_BREADTH * 100:.1f}%"

        # 市況過濾器：依昨日 regime 動態決定買入門檻 (趨勢市進攻、震盪/空頭防守)。
        if use_regime_filter and signal_regime is not None:
            current_buy_threshold = _regime_buy_thr.get(signal_regime, BUY_THRESHOLD)
        else:
            current_buy_threshold = BUY_THRESHOLD
        if is_market_panic:
            current_buy_threshold = 99.0
            print(f"  [風控警示] {today.date()} 觸發大盤避險紅燈！原因: {panic_reason}。今日起暫停任何新股買進。")

        # 市況選擇性曝險：依昨日 regime 調整最大持股檔數（與買入門檻同步生效，僅在未顯式指定 buy_threshold 時）。
        # 超過上限時不強制平倉，僅停止補倉，靠自然汰換（停損／Day3／止盈）降至上限，避免振盪洗價。
        if use_regime_filter and signal_regime is not None:
            eff_max_positions = _regime_max_pos.get(signal_regime, max_positions)
        else:
            eff_max_positions = max_positions

        # 市況出場：依遲滯後 regime 動態切換出場參數（Bull 立即生效，非 Bull 需連續 N 天）
        if _regime_exit and _exit_regime in _regime_exit:
            _re = _regime_exit[_exit_regime]
            eff_sell_thr = _re.get('sell_threshold', SELL_THRESHOLD)
            eff_ts_act   = _re.get('ts_activation',  TS_ACTIVATION_PCT)
            eff_ts_pull  = _re.get('ts_pullback',    TS_PULLBACK_PCT)
            eff_min_hold = _re.get('min_hold_days',  MIN_HOLD_DAYS)
        else:
            eff_sell_thr = SELL_THRESHOLD
            eff_ts_act   = TS_ACTIVATION_PCT
            eff_ts_pull  = TS_PULLBACK_PCT
            eff_min_hold = MIN_HOLD_DAYS

        # --- B. 賣出邏輯 (昨日訊號 Day3 < 0，跌破停損，或是觸發移動止盈) ---
        sells_today = []
        for sid, pos in list(positions.items()):
            if sid in prev_data.index and sid in today_data.index:
                day3_score = prev_data.loc[sid, 'Day3_net']
                
                open_T1 = today_data.loc[sid, 'open']
                low_T1 = today_data.loc[sid, 'low']
                close_T1 = today_data.loc[sid, 'close']
                
                # 價格防呆補值
                if pd.isna(open_T1) or open_T1 <= 0:
                    open_T1 = pos['current_price']
                if pd.isna(low_T1) or low_T1 <= 0:
                    low_T1 = open_T1
                if pd.isna(close_T1) or close_T1 <= 0:
                    close_T1 = pos['current_price']
                
                actual_cost_price = pos['buy_price'] * (1 + FEE_RATE)
                _stop_pct = pos.get('atr_stop_pct', STOP_LOSS_PCT)
                stop_loss_price = actual_cost_price * (1 + _stop_pct / 100.0)
                
                # 判定 1: 階段式寬鬆移動止盈 (浮盈達到 +10% 啟動門檻，且高點收盤回撤 6%)
                triggered_trailing_stop = False
                if 'max_close_price' in pos and pos['max_close_price'] >= actual_cost_price * (1.0 + eff_ts_act / 100.0):
                    trailing_stop_price = pos['max_close_price'] * (1.0 + eff_ts_pull / 100.0)  # 自最高收盤回撤一定趴數
                    # 我們在 T 日收盤發現跌破止盈點，則在今日 (T+1) 執行賣出
                    # 這裡為了簡化，在昨天的收盤價跌破昨日高點回撤線時，今天開盤直接賣
                    # 使用 pos['current_price'] 作為安全 fallback，避免 close_prev 未定義的 NameError
                    prev_close = prev_data.loc[sid, 'close'] if 'close' in prev_data.columns else pos.get('current_price', 0)
                    if not pd.isna(prev_close) and prev_close > 0 and prev_close <= trailing_stop_price:
                        triggered_trailing_stop = True
                
                # 計算持股天數 (如果沒有 buy_idx 則預設為 999 避免阻礙賣出)
                buy_idx = pos.get('buy_idx', None)
                held_trading_days = (idx_today - buy_idx) if buy_idx is not None else 999
                
                # 賣出判定優先級：停損不受最少持股天數限制，但 Day3 訊號轉弱與移動止盈必須符合 MIN_HOLD_DAYS 限制
                if open_T1 <= stop_loss_price:
                    sells_today.append((sid, open_T1, f"觸發停損(開盤跳空:{stop_loss_price:.2f})"))
                elif low_T1 <= stop_loss_price:
                    sells_today.append((sid, stop_loss_price, f"觸發停損(盤中:{stop_loss_price:.2f})"))
                elif held_trading_days >= eff_min_hold:
                    if day3_score < eff_sell_thr:
                        sells_today.append((sid, open_T1, f"Day3預測轉弱 ({day3_score:.1f}% < {eff_sell_thr}%)"))
                    elif triggered_trailing_stop:
                        sells_today.append((sid, open_T1, f"觸發移動止盈(高點 {pos['max_close_price']:.2f} 回撤{abs(eff_ts_pull)}%)"))

        today_sells_amount = 0
        for sid, sell_price, reason in sells_today:
            pos = positions.pop(sid)
            gross = pos['shares'] * sell_price
            sell_fee = gross * FEE_RATE
            sell_tax = gross * TAX_RATE
            net_proceeds = gross - sell_fee - sell_tax

            # 賣出當天即刻釋出可用資金 (購買力)
            available_cash += net_proceeds
            today_sells_amount += net_proceeds

            profit = net_proceeds - (pos['shares'] * pos['buy_price'] * (1 + FEE_RATE))
            profit_pct = profit / (pos['shares'] * pos['buy_price'] * (1 + FEE_RATE)) * 100
            cname = stock_names.get(sid, "")
            
            stock_value = sum(p['shares'] * p['current_price'] for p in positions.values())
            total_equity = available_cash + stock_value
            
            trades.append({
                'Date': today.date(),
                'Action': '賣出',
                'Stock_ID': sid,
                'Stock_Name': cname,
                'Price': sell_price,
                'Shares': pos['shares'],
                'Amount': net_proceeds,
                'Fee': sell_fee,
                'Tax': sell_tax,
                'Profit': profit,
                'Profit_Pct(%)': profit_pct,
                'Current_Cash': available_cash,
                'Stock_Value': stock_value,
                'Total_Equity': total_equity,
                'Reason': reason
            })

        # --- C. 買進邏輯 (根據昨日訊號做挑選，並在今日執行限價掛單撮合) ---
        # 排除已持有的股票，依昨日的 Day 1 分數降冪排序
        buy_candidates = prev_data[~prev_data.index.isin(positions.keys())].copy()

        # Bull regime 動能混合：排序用混合分數，閾值過濾仍用原始模型分數，避免破壞風控邏輯。
        # Hysteresis：需連續 MOMENTUM_BULL_CONFIRM_DAYS 天 Bull 才啟用，防止熊牛轉換振盪期誤觸。
        momentum_active = (signal_regime == 'Bull' and consecutive_bull_days >= MOMENTUM_BULL_CONFIRM_DAYS)
        if momentum_active and 'RS_20d' in buy_candidates.columns:
            rs_rank = buy_candidates['RS_20d'].rank(pct=True, na_option='bottom') * 100
            buy_candidates['_sort_score'] = 0.30 * buy_candidates['Day1_net'] + 0.70 * rs_rank
        else:
            buy_candidates['_sort_score'] = buy_candidates['Day1_net']

        buy_candidates = buy_candidates.sort_values('_sort_score', ascending=False)
        
        today_buys_amount = 0
        for sid in buy_candidates.index:
            if len(positions) >= eff_max_positions:
                break
                
            day1_score = prev_data.loc[sid, 'Day1_net']
            # 使用當前風控調整後的 BUY_THRESHOLD (如果觸發避險紅燈則為 99.0%)
            if day1_score < current_buy_threshold:
                continue
                
            # 必須有今日開盤、盤中及收盤價格資料
            if sid not in today_data.index:
                continue
                
            close_prev = prev_data.loc[sid, 'close']
            if pd.isna(close_prev) or close_prev <= 0:
                continue
                
            open_T1 = today_data.loc[sid, 'open']
            low_T1 = today_data.loc[sid, 'low']
            close_T1 = today_data.loc[sid, 'close']
            
            if pd.isna(open_T1) or open_T1 <= 0 or pd.isna(low_T1) or low_T1 <= 0:
                continue
                
            # 根據 D1 分數動態決定加價幅度，或是採用固定的 ORDER_MARKUP_PCT
            if ORDER_MARKUP_PCT is not None:
                target_pct = ORDER_MARKUP_PCT
            else:
                if day1_score >= ORDER_MARKUP_HIGH_SCORE:
                    target_pct = ORDER_MARKUP_HIGH_PCT
                elif day1_score >= ORDER_MARKUP_MID_SCORE:
                    target_pct = ORDER_MARKUP_MID_PCT
                else:
                    target_pct = ORDER_MARKUP_LOW_PCT
                
            raw_target_price = close_prev * (1 + target_pct / 100.0)
            limit_price = round_to_tick(raw_target_price)
            
            # --- 限價掛單搓合機制 ---
            buy_price = None
            if open_T1 <= limit_price:
                # 1. 隔天開盤價未高於加價限價 ➔ 買進，成交價為開盤價
                buy_price = open_T1
                reason_detail = f"D1分數強勢 ({day1_score:.1f}%) | 開盤撮合 (限價:{limit_price:.2f} >= 開盤:{open_T1:.2f})"
            elif low_T1 <= limit_price:
                # 2. 隔天開盤跳空高於限價，但盤中有回檔跌破限價 ➔ 買進，成交價為限價
                buy_price = limit_price
                reason_detail = f"D1分數強勢 ({day1_score:.1f}%) | 盤中回檔撮合 (開盤:{open_T1:.2f} > 限價:{limit_price:.2f} >= 最低:{low_T1:.2f})"
            else:
                # 3. 隔天盤中最低價仍然高於限價 ➔ 買不到，跳過
                continue
            
            # 動態分配剩餘資金：以「基準」MAX_POSITIONS 為分母，使單檔權重維持 ~1/MAX_POSITIONS。
            # 故 regime 降檔數（eff_max_positions）時是讓資金「閒置」以降總曝險，而非把同筆資金集中到更少更大的部位。
            available_slots = max(1, max_positions - len(positions))
            target_investment = available_cash / available_slots
            invest_amount = min(target_investment, available_cash)
            if invest_amount < 1000:
                continue
                
            # 台股以「張」(1000股) 為交易單位。買得起 ≥1 張 → 只買整張（零頭忽略，貼近多數散戶）；
            # 連 1 張都買不起的高價股 → 退而以零股買進，成交價採收盤「最後揭示賣價」(吃賣單)。
            _affordable = int((invest_amount / (1 + FEE_RATE)) // buy_price)
            whole_lot_shares = (_affordable // LOT_SIZE) * LOT_SIZE
            if whole_lot_shares >= LOT_SIZE:
                max_shares = whole_lot_shares
                fill_price = buy_price
                fill_note = ""
            elif not ODD_LOT_ENABLED:
                # 純整張模式：連 1 張都買不起就跳過此檔
                continue
            else:
                _ask = today_data.loc[sid, 'last_ask'] if 'last_ask' in today_data.columns else None
                fill_price = float(_ask) if (_ask is not None and not pd.isna(_ask) and float(_ask) > 0) else buy_price
                # 零股最多買到 1 張－1 股（買得起整張時不會走此分支）
                max_shares = min(int((invest_amount / (1 + FEE_RATE)) // fill_price), LOT_SIZE - 1)
                fill_note = f" | 零股買進 (買不起整張，揭示賣價:{fill_price:.2f})"
            if max_shares > 0:
                buy_fee = max_shares * fill_price * FEE_RATE
                cost = max_shares * fill_price + buy_fee
                available_cash -= cost
                today_buys_amount += cost
                
                # 計算個股 ATR 停損百分比（買入當下鎖定，後續不再變動）
                _atr_pct = prev_data.loc[sid, 'atr18_pct'] if 'atr18_pct' in prev_data.columns else None
                if ATR_STOP_ENABLED and _atr_pct is not None and not pd.isna(_atr_pct) and float(_atr_pct) > 0:
                    _raw = -ATR_STOP_MULTIPLIER * float(_atr_pct) * 100
                    _atr_stop_pct = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEILING_PCT, _raw))
                else:
                    _atr_stop_pct = STOP_LOSS_PCT

                positions[sid] = {
                    'shares': max_shares,
                    'buy_price': fill_price,
                    'current_price': close_T1 if not pd.isna(close_T1) and close_T1 > 0 else fill_price,
                    'buy_date': today,
                    'max_close_price': close_T1 if not pd.isna(close_T1) and close_T1 > 0 else fill_price,
                    'buy_idx': idx_today,
                    'atr_stop_pct': _atr_stop_pct,
                }
                cname = stock_names.get(sid, "")
                
                stock_value = sum(p['shares'] * p['current_price'] for p in positions.values())
                total_equity = available_cash + stock_value
                
                trades.append({
                    'Date': today.date(),
                    'Action': '買進',
                    'Stock_ID': sid,
                    'Stock_Name': cname,
                    'Price': fill_price,
                    'Shares': max_shares,
                    'Amount': cost,
                    'Fee': buy_fee,
                    'Tax': 0.0,
                    'Profit': 0.0,
                    'Profit_Pct(%)': 0.0,
                    'Current_Cash': available_cash,
                    'Stock_Value': stock_value,
                    'Total_Equity': total_equity,
                    'Reason': reason_detail + fill_note
                })

        # --- C2. 計算今日交易之 T+2 淨交割金額並加入待交割佇列 ---
        net_change = today_sells_amount - today_buys_amount
        if net_change != 0:
            if idx_today + 2 < len(trading_dates):
                settlement_date = trading_dates[idx_today + 2].date()
            else:
                curr_dt = today
                added = 0
                while added < 2:
                    curr_dt += pd.Timedelta(days=1)
                    if curr_dt.weekday() < 5:
                        added += 1
                settlement_date = curr_dt.date()
                
            if settlement_date in pending_settlements:
                pending_settlements[settlement_date] += net_change
            else:
                pending_settlements[settlement_date] = net_change

        # --- A. 更新今日持倉價格與歷史最高收盤價 (移至日終，避免移動止盈前視偏差) ---
        for sid, pos in positions.items():
            if sid in today_data.index:
                new_price = today_data.loc[sid, 'close']
                if not pd.isna(new_price) and new_price > 0:
                    pos['current_price'] = new_price
                    if 'max_close_price' not in pos:
                        pos['max_close_price'] = new_price
                    else:
                        pos['max_close_price'] = max(pos['max_close_price'], new_price)

        # --- D. 計算今日日終淨值並寫入歷史 ---
        final_stock_value = sum(pos['shares'] * pos['current_price'] for pos in positions.values())
        pending_cash = sum(pending_settlements.values())
        current_equity = bank_cash + pending_cash + final_stock_value
        
        # 計算今日組合回報與 Alpha/Spread
        if idx_today == 0:
            prev_equity = initial_capital
        else:
            prev_equity = history[-1]['equity']
            
        portfolio_return = (current_equity / prev_equity - 1.0)
        
        mkt_ret = today_data['market_mean_pct'].iloc[0] if not today_data.empty and 'market_mean_pct' in today_data.columns else 0.0
        portfolio_alpha = portfolio_return - mkt_ret
        
        # 尋找昨日預測最低的 5% 股票，計算其今日的實際均報作為對照 (Short leg)
        if not prev_data.empty:
            bot_k_size = max(1, int(len(prev_data) * 0.05))
            bottom_stocks = prev_data.sort_values('Day1_net', ascending=True).head(bot_k_size)
            bottom_mean_ret = bottom_stocks['next_ret_1'].mean() if not bottom_stocks.empty else 0.0
        else:
            bottom_mean_ret = 0.0
            
        portfolio_spread = portfolio_return - bottom_mean_ret
        
        history.append({
            'date': today.date(),
            'equity': current_equity,
            'cash': available_cash,
            'bank_cash': bank_cash,
            'pending_cash': pending_cash,
            'invested': final_stock_value,
            'regime': regime,
            'n_positions': len(positions),
            'portfolio_return': portfolio_return,
            'portfolio_alpha': portfolio_alpha,
            'portfolio_spread': portfolio_spread
        })

        # 統一將今日產生的明細欄位更新為日終淨值
        for i in range(start_trade_idx, len(trades)):
            trades[i]['Current_Cash'] = available_cash
            trades[i]['Stock_Value'] = final_stock_value
            trades[i]['Total_Equity'] = current_equity

    cash = available_cash

    # 4. 結算與報表
    print("\n" + "=" * 70)
    print("  回測結束 - 績效結算")
    print("=" * 70)
    
    final_equity = bank_cash + sum(pending_settlements.values()) + sum(pos['shares'] * pos['current_price'] for pos in positions.values())
    total_return = ((final_equity / initial_capital) - 1) * 100
    
    print(f"期初資金: {initial_capital:,.0f}")
    print(f"期末總值: {final_equity:,.0f}")
    print(f"區間報酬: {total_return:+.2f}%")
    
    # 計算最大回撤 (Max Drawdown)
    max_dd = 0
    if history:
        equity_curve = [h['equity'] for h in history]
        peak = equity_curve[0]
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
            
    print(f"最大回撤: -{max_dd*100:.2f}%")
    holdings_str = [f"{sid} {stock_names.get(sid, '')}({int(pos['shares'])}股)" for sid, pos in positions.items()]
    print(f"最終持有: {holdings_str}")

    # 交易成本總覽：判斷是否進出過於頻繁、利潤被手續費／證交稅吃掉
    _tr = pd.DataFrame(trades)
    if not _tr.empty:
        n_buy = int((_tr['Action'] == '買進').sum())
        n_sell = int((_tr['Action'] == '賣出').sum())
        total_fee = float(_tr['Fee'].sum())
        total_tax = float(_tr['Tax'].sum())
        total_cost = total_fee + total_tax
        net_profit = final_equity - initial_capital
        # 毛利＝淨利＋交易成本（若不收費用本可賺到的金額），用以衡量成本侵蝕比例
        gross_profit = net_profit + total_cost
        cost_vs_gross = (total_cost / gross_profit * 100) if gross_profit > 0 else float('nan')
        print(f"\n  交易成本總覽")
        print(f"  {'-'*40}")
        print(f"  完成回合(買/賣): {n_buy} / {n_sell}")
        print(f"  總手續費      : {total_fee:>14,.0f}")
        print(f"  總證交稅      : {total_tax:>14,.0f}")
        print(f"  交易成本合計  : {total_cost:>14,.0f}  (佔期初資金 {total_cost/initial_capital*100:.2f}%)")
        print(f"  區間淨利      : {net_profit:>14,.0f}")
        print(f"  成本侵蝕毛利  : {cost_vs_gross:.1f}%  (毛利中有多少被費用/稅吃掉)")
        print(f"  {'-'*40}")

    # 曝險率報告：每日持倉數 × 市況，診斷閾值是否在 Bull 浪費 alpha
    if history:
        _h = pd.DataFrame(history)
        print(f"\n  曝險率報告（持倉格位 × 市況）")
        print(f"  {'='*54}")
        print(f"  {'市況':<10}{'天數':>6}{'平均持倉':>10}{'0檔%':>8}{'1-2檔%':>9}{'滿倉%':>9}")
        print(f"  {'-'*52}")
        for _reg in ['Bull', 'Sideways', 'Bear', 'All']:
            _sub = _h if _reg == 'All' else _h[_h['regime'] == _reg]
            if _sub.empty:
                continue
            _n = len(_sub)
            _avg = _sub['n_positions'].mean()
            _p0   = (_sub['n_positions'] == 0).sum() / _n * 100
            _p12  = (_sub['n_positions'].between(1, 2)).sum() / _n * 100
            _pfull = (_sub['n_positions'] == max_positions).sum() / _n * 100
            _label = f"[{_reg}]" if _reg != 'All' else '[全期]'
            print(f"  {_label:<10}{_n:>6}{_avg:>10.1f}{_p0:>7.1f}%{_p12:>8.1f}%{_pfull:>8.1f}%")
        _bull_h = _h[_h['regime'] == 'Bull']
        if not _bull_h.empty:
            _bull_under = (_bull_h['n_positions'] <= 2).sum() / len(_bull_h) * 100
            _flag = " [!] 閾值可能浪費 alpha" if _bull_under > 30 else " [OK] 曝險充足"
            print(f"  {'='*52}")
            print(f"  Bull 低持倉(0-2檔)佔 {_bull_under:.1f}%{_flag}")
    
    # 建立報表 DataFrame
    df_history = pd.DataFrame(history)
    df_trades = pd.DataFrame(trades)
    
    final_holdings_list = []
    for sid, pos in positions.items():
        actual_cost_price = pos['buy_price'] * (1 + FEE_RATE)
        final_holdings_list.append({
            'Stock_ID': sid,
            'Stock_Name': stock_names.get(sid, ""),
            'Shares': pos['shares'],
            'Buy_Date': pos['buy_date'].date(),
            'Buy_Price': pos['buy_price'],
            'Current_Price': pos['current_price'],
            'Market_Value': pos['shares'] * pos['current_price'],
            'Unrealized_Profit(%)': (pos['current_price'] - actual_cost_price) / actual_cost_price * 100
        })
    df_holdings = pd.DataFrame(final_holdings_list)

    # 翻譯 DataFrame 欄位名稱
    df_history.rename(columns={
        'date': '日期', 
        'equity': '總資產', 
        'cash': '可用資金(購買力)', 
        'bank_cash': '銀行帳戶實質餘額', 
        'pending_cash': '待交割金額', 
        'invested': '持股市值'
    }, inplace=True)
    df_trades.rename(columns={
        'Date': '日期', 'Action': '操作', 'Stock_ID': '股票編號', 'Stock_Name': '股票名稱',
        'Price': '價格', 'Shares': '股數', 'Amount': '金額',
        'Fee': '手續費', 'Tax': '證交稅', 'Profit': '利潤',
        'Profit_Pct(%)': '利潤率(%)', 'Current_Cash': '可用資金(購買力)',
        'Stock_Value': '持股市值', 'Total_Equity': '總資產', 'Reason': '原因'
    }, inplace=True)
    df_holdings.rename(columns={
        'Stock_ID': '股票編號', 'Stock_Name': '股票名稱', 'Shares': '股數',
        'Buy_Date': '買入日期', 'Buy_Price': '買入價格', 'Current_Price': '當前價格',
        'Market_Value': '市值', 'Unrealized_Profit(%)': '未實現利潤(%)'
    }, inplace=True)

    # 輸出報表
    try:
        report_dir = os.path.join(BASE_DIR, "reports")
        os.makedirs(report_dir, exist_ok=True)
        safe_start = start_date.replace("/", "-")
        safe_end = end_date.replace("/", "-")
        
        try:
            excel_path = os.path.join(report_dir, f"backtest_report_{safe_start}_{safe_end}.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_history.to_excel(writer, sheet_name='Equity_Curve', index=False)
                df_trades.to_excel(writer, sheet_name='Trade_History', index=False)
                df_holdings.to_excel(writer, sheet_name='Final_Holdings', index=False)
            print(f"\n[成功] 完整回測報表已匯出 (Excel格式): {excel_path}")
        except ImportError:
            p1 = os.path.join(report_dir, f"backtest_equity_{safe_start}_{safe_end}.csv")
            p2 = os.path.join(report_dir, f"backtest_trades_{safe_start}_{safe_end}.csv")
            p3 = os.path.join(report_dir, f"backtest_holdings_{safe_start}_{safe_end}.csv")
            df_history.to_csv(p1, index=False, encoding="utf-8-sig")
            df_trades.to_csv(p2, index=False, encoding="utf-8-sig")
            df_holdings.to_csv(p3, index=False, encoding="utf-8-sig")
            print(f"\n[警告] 未安裝 openpyxl，報表已分拆匯出至 CSV:")
            print(f"  - 淨值曲線: {p1}")
            print(f"  - 交易明細: {p2}")
            print(f"  - 最終持股: {p3}")
    except Exception as e:
        print(f"匯出報表失敗: {e}")
        
    print("=" * 70)
    return total_return, max_dd, history

class CustomHelpParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"\n[參數輸入錯誤] {message}")
        print("=" * 70)
        print(" 正確的執行範例:")
        print(f"   python trading_sim.py -s {SIM_DEFAULT_START} -e {SIM_DEFAULT_END} -c {SIM_DEFAULT_CAPITAL} -m {MAX_POSITIONS}\n")
        print("參數說明:")
        print(f"  -s, --start    回測起始日期 (預設: {SIM_DEFAULT_START}，格式: YYYY-MM-DD)")
        print(f"  -e, --end      回測結束日期 (預設: {SIM_DEFAULT_END}，格式: YYYY-MM-DD)")
        print(f"  -c, --capital  初始資金 (預設: {SIM_DEFAULT_CAPITAL})")
        print(f"  -m, --max_pos  最大持股檔數 (預設: {MAX_POSITIONS})")
        print("=" * 70 + "\n")
        sys.exit(2)

if __name__ == "__main__":
    parser = CustomHelpParser(description="量化模型自動交易回測")
    parser.add_argument("-s", "--start", type=str, default=SIM_DEFAULT_START, help="回測起始日期 (YYYY-MM-DD)")
    parser.add_argument("-e", "--end",   type=str, default=SIM_DEFAULT_END,   help="回測結束日期 (YYYY-MM-DD)")
    parser.add_argument("-c", "--capital", type=int, default=SIM_DEFAULT_CAPITAL, help="初始資金")
    parser.add_argument("-m", "--max_pos", type=int, default=MAX_POSITIONS, help="最大持股檔數")
    parser.add_argument("--panic_ma5", type=float, default=None, help="大盤 5 日滾動平均報酬率避險門檻 (例如 -0.005)")
    parser.add_argument("--panic_breadth", type=float, default=None, help="全市場上漲家數比例避險門檻 (例如 0.35)")
    parser.add_argument("--buy_threshold", type=float, default=None, help="買入多空淨分數門檻 (%)")
    parser.add_argument("--sell_threshold", type=float, default=None, help="賣出多空淨分數門檻 (%)")
    parser.add_argument("--stop_loss", type=float, default=None, help="固定的個股停損趴數 (%)")
    parser.add_argument("--ts_activation", type=float, default=None, help="移動止盈啟動門檻 (%)")
    parser.add_argument("--ts_pullback", type=float, default=None, help="移動止盈回撤門檻 (%)")
    parser.add_argument("--min_hold_days", type=int, default=None, help="最少持股天數 (防止頻繁交易)")
    parser.add_argument("--markup_pct", type=float, default=None, help="掛單溢價幅度 (%)，負數為折價買進")
    # 市況出場參數（逐 regime 覆蓋，不傳則沿用全域預設）
    parser.add_argument("--bull_sell",   type=float, default=None, help="Bull regime 賣出門檻 (%)")
    parser.add_argument("--bull_ts_act", type=float, default=None, help="Bull regime 移動止盈啟動 (%)")
    parser.add_argument("--bull_ts_pull",type=float, default=None, help="Bull regime 移動止盈回撤 (%)")
    parser.add_argument("--bull_hold",   type=int,   default=None, help="Bull regime 最少持股天數")
    parser.add_argument("--side_sell",   type=float, default=None, help="Sideways regime 賣出門檻 (%)")
    parser.add_argument("--side_ts_act", type=float, default=None, help="Sideways regime 移動止盈啟動 (%)")
    parser.add_argument("--side_ts_pull",type=float, default=None, help="Sideways regime 移動止盈回撤 (%)")
    parser.add_argument("--side_hold",   type=int,   default=None, help="Sideways regime 最少持股天數")
    parser.add_argument("--bear_sell",   type=float, default=None, help="Bear regime 賣出門檻 (%)")
    parser.add_argument("--bear_ts_act", type=float, default=None, help="Bear regime 移動止盈啟動 (%)")
    parser.add_argument("--bear_ts_pull",type=float, default=None, help="Bear regime 移動止盈回撤 (%)")
    parser.add_argument("--bear_hold",   type=int,   default=None, help="Bear regime 最少持股天數")
    parser.add_argument("--exit_lag",    type=int,   default=None, help="切出 Bull 出場模式所需的連續非 Bull 天數 (不傳則沿用 config EXIT_REGIME_LAG)")

    args = parser.parse_args()

    _regime_exit_params = {}
    bull_p = {k: v for k, v in {'sell_threshold': args.bull_sell, 'ts_activation': args.bull_ts_act,
              'ts_pullback': args.bull_ts_pull, 'min_hold_days': args.bull_hold}.items() if v is not None}
    if bull_p: _regime_exit_params['Bull'] = bull_p
    side_p = {k: v for k, v in {'sell_threshold': args.side_sell, 'ts_activation': args.side_ts_act,
              'ts_pullback': args.side_ts_pull, 'min_hold_days': args.side_hold}.items() if v is not None}
    if side_p: _regime_exit_params['Sideways'] = side_p
    bear_p = {k: v for k, v in {'sell_threshold': args.bear_sell, 'ts_activation': args.bear_ts_act,
              'ts_pullback': args.bear_ts_pull, 'min_hold_days': args.bear_hold}.items() if v is not None}
    if bear_p: _regime_exit_params['Bear'] = bear_p

    run_simulation(
        args.start, args.end, args.capital, args.max_pos,
        mkt_panic_ma5=args.panic_ma5,
        mkt_panic_breadth=args.panic_breadth,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        stop_loss_pct=args.stop_loss,
        ts_activation_pct=args.ts_activation,
        ts_pullback_pct=args.ts_pullback,
        min_hold_days=args.min_hold_days,
        markup_pct=args.markup_pct,
        regime_exit_params=_regime_exit_params or None,
        exit_regime_lag=args.exit_lag,
    )
