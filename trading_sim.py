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
        MAX_POSITIONS,
    )
except ImportError:
    ORDER_MARKUP_HIGH_SCORE = 30.0; ORDER_MARKUP_MID_SCORE = 20.0
    ORDER_MARKUP_HIGH_PCT   = 2.5;  ORDER_MARKUP_MID_PCT   = 2.0; ORDER_MARKUP_LOW_PCT = 1.5
    SIM_DEFAULT_START = "2026-01-01"; SIM_DEFAULT_END = "2026-06-30"; SIM_DEFAULT_CAPITAL = 1_000_000
    MAX_POSITIONS = 5

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
                   buy_threshold=None, stop_loss_pct=None,
                   ts_activation_pct=None, ts_pullback_pct=None):
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
            TS_ACTIVATION_PCT, TS_PULLBACK_PCT
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

    # 外部參數覆蓋
    if buy_threshold is not None:
        BUY_THRESHOLD = buy_threshold
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
        
        today_data = df_sim[df_sim['date'] == today].set_index('stock_id')
        prev_data = df_sim[df_sim['date'] == prev_day].set_index('stock_id')
        
        # --- A. 更新今日持倉價格與歷史最高收盤價 ---
        for sid, pos in positions.items():
            if sid in today_data.index:
                new_price = today_data.loc[sid, 'close']
                if not pd.isna(new_price) and new_price > 0:
                    pos['current_price'] = new_price
                    # 更新買入後的最高收盤價
                    if 'max_close_price' not in pos:
                        pos['max_close_price'] = new_price
                    else:
                        pos['max_close_price'] = max(pos['max_close_price'], new_price)

        # --- A1. 讀取大盤指標與判斷硬風控紅燈 (Market Sentiment Filter) ---
        mkt_ma5 = 0.0
        mkt_breadth = 1.0
        if not prev_data.empty:
            first_row = prev_data.iloc[0]
            if 'market_mean_ma5' in prev_data.columns:
                mkt_ma5 = first_row['market_mean_ma5']
            if 'market_breadth_pct' in prev_data.columns:
                mkt_breadth = first_row['market_breadth_pct']
                
        is_market_panic = False
        panic_reason = ""
        if mkt_ma5 < MKT_PANIC_MA5:
            is_market_panic = True
            panic_reason = f"大盤 5 日滾動平均報酬率過低 ({mkt_ma5 * 100:+.2f}%)，低於風控門檻 {MKT_PANIC_MA5 * 100:+.2f}%"
        elif mkt_breadth < MKT_PANIC_BREADTH:
            is_market_panic = True
            panic_reason = f"全市場上漲比例過低 ({mkt_breadth * 100:.1f}%)，低於風控門檻 {MKT_PANIC_BREADTH * 100:.1f}%"

        current_buy_threshold = BUY_THRESHOLD
        if is_market_panic:
            current_buy_threshold = 99.0
            print(f"  [風控警示] {today.date()} 觸發大盤避險紅燈！原因: {panic_reason}。今日起暫停任何新股買進。")

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
                stop_loss_price = actual_cost_price * (1 + STOP_LOSS_PCT / 100.0)
                
                # 判定 1: 階段式寬鬆移動止盈 (浮盈達到 +10% 啟動門檻，且高點收盤回撤 6%)
                triggered_trailing_stop = False
                if 'max_close_price' in pos and pos['max_close_price'] >= actual_cost_price * (1.0 + TS_ACTIVATION_PCT / 100.0):
                    trailing_stop_price = pos['max_close_price'] * (1.0 + TS_PULLBACK_PCT / 100.0)  # 自最高收盤回撤一定趴數
                    # 我們在 T 日收盤發現跌破止盈點，則在今日 (T+1) 執行賣出
                    # 這裡為了簡化，在昨天的收盤價跌破昨日高點回撤線時，今天開盤直接賣
                    # 使用 pos['current_price'] 作為安全 fallback，避免 close_prev 未定義的 NameError
                    prev_close = prev_data.loc[sid, 'close'] if 'close' in prev_data.columns else pos.get('current_price', 0)
                    if not pd.isna(prev_close) and prev_close > 0 and prev_close <= trailing_stop_price:
                        triggered_trailing_stop = True
                
                # 賣出判定優先級
                if day3_score < SELL_THRESHOLD:
                    sells_today.append((sid, open_T1, "Day3預測轉弱"))
                elif open_T1 <= stop_loss_price:
                    sells_today.append((sid, open_T1, "觸發-8%停損(開盤跳空)"))
                elif low_T1 <= stop_loss_price:
                    sells_today.append((sid, stop_loss_price, "觸發-8%停損(盤中)"))
                elif triggered_trailing_stop:
                    sells_today.append((sid, open_T1, f"觸發移動止盈(高點 {pos['max_close_price']:.2f} 回撤{abs(TS_PULLBACK_PCT)}%)"))

        today_sells_amount = 0
        for sid, sell_price, reason in sells_today:
            pos = positions.pop(sid)
            gross = pos['shares'] * sell_price
            net_proceeds = gross * (1 - FEE_RATE - TAX_RATE)
            
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
        buy_candidates = buy_candidates.sort_values('Day1_net', ascending=False)
        
        today_buys_amount = 0
        for sid in buy_candidates.index:
            if len(positions) >= max_positions:
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
                
            # 根據 D1 分數動態決定加價幅度 (與 scripts/inference.py 對齊，共用 config.py ORDER_MARKUP_* 常數)
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
            
            # 動態分配剩餘資金
            available_slots = max_positions - len(positions)
            target_investment = available_cash / available_slots
            invest_amount = min(target_investment, available_cash)
            if invest_amount < 1000:
                continue
                
            max_shares = int((invest_amount / (1 + FEE_RATE)) // buy_price)
            if max_shares > 0:
                cost = max_shares * buy_price * (1 + FEE_RATE)
                available_cash -= cost
                today_buys_amount += cost
                
                positions[sid] = {
                    'shares': max_shares,
                    'buy_price': buy_price,
                    'current_price': close_T1 if not pd.isna(close_T1) and close_T1 > 0 else buy_price,
                    'buy_date': today,
                    'max_close_price': close_T1 if not pd.isna(close_T1) and close_T1 > 0 else buy_price
                }
                cname = stock_names.get(sid, "")
                
                stock_value = sum(p['shares'] * p['current_price'] for p in positions.values())
                total_equity = available_cash + stock_value
                
                trades.append({
                    'Date': today.date(),
                    'Action': '買進',
                    'Stock_ID': sid,
                    'Stock_Name': cname,
                    'Price': buy_price,
                    'Shares': max_shares,
                    'Amount': cost,
                    'Profit': 0.0,
                    'Profit_Pct(%)': 0.0,
                    'Current_Cash': available_cash,
                    'Stock_Value': stock_value,
                    'Total_Equity': total_equity,
                    'Reason': reason_detail
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

        # --- D. 計算今日日終淨值並寫入歷史 ---
        final_stock_value = sum(pos['shares'] * pos['current_price'] for pos in positions.values())
        pending_cash = sum(pending_settlements.values())
        current_equity = bank_cash + pending_cash + final_stock_value
        
        history.append({
            'date': today.date(),
            'equity': current_equity,
            'cash': available_cash,
            'bank_cash': bank_cash,
            'pending_cash': pending_cash,
            'invested': final_stock_value
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
    
    final_equity = cash + sum(pos['shares'] * pos['current_price'] for pos in positions.values())
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
        'Price': '價格', 'Shares': '股數', 'Amount': '金額', 'Profit': '利潤', 
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
    parser.add_argument("--stop_loss", type=float, default=None, help="固定的個股停損趴數 (%)")
    parser.add_argument("--ts_activation", type=float, default=None, help="移動止盈啟動門檻 (%)")
    parser.add_argument("--ts_pullback", type=float, default=None, help="移動止盈回撤門檻 (%)")
    
    args = parser.parse_args()
    run_simulation(
        args.start, args.end, args.capital, args.max_pos,
        mkt_panic_ma5=args.panic_ma5,
        mkt_panic_breadth=args.panic_breadth,
        buy_threshold=args.buy_threshold,
        stop_loss_pct=args.stop_loss,
        ts_activation_pct=args.ts_activation,
        ts_pullback_pct=args.ts_pullback
    )
