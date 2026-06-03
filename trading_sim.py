import os
import sys
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_models():
    models = {}
    for days in [1, 2, 3]:
        model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days}.txt")
        if not os.path.exists(model_path):
            print(f"[錯誤] 找不到預先訓練的模型: {model_path}。請先執行 train.py。")
            sys.exit(1)
        models[days] = lgb.Booster(model_file=model_path)
    return models

def run_simulation(start_date, end_date, initial_capital, max_positions):
    print("=" * 70)
    print(f"  啟動量化交易回測 (Out-of-Sample)")
    print(f"  期間: {start_date} 到 {end_date}")
    print(f"  初始資金: {initial_capital:,} | 最大持股數: {max_positions}")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print("[錯誤] 找不到 features_combined.parquet")
        return

    # 1. 載入資料並過濾日期
    df = pd.read_parquet(DATA_PATH)
    
    # ── 根據 train.py 中的 TRAIN_INDUSTRIES 過濾股票 ──────────────────
    from utils import filter_stocks_by_train_industries
    before_cnt = df["stock_id"].nunique()
    df = filter_stocks_by_train_industries(df)
    after_cnt = df["stock_id"].nunique()
    print(f"  [模擬過濾] 依 train.py 產業設定篩選：原本 {before_cnt} 檔，剩餘 {after_cnt} 檔進行回測模擬")
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
        
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
    df_sim = df[mask].copy()

    if df_sim.empty:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        print(f"\n[查無資料] 在指定的區間 ({start_date} ~ {end_date}) 內沒有找到任何特徵資料！")
        print(f" 提示：目前資料庫 (features_combined.parquet) 中擁有的資料涵蓋範圍為 {min_date} 到 {max_date}。")
        print("請確認您輸入的日期是否超出此範圍 (例如輸入了未來的日期)，或者該區間內剛好全部都是假日。")
        return

    # 2. 決定特徵欄位並進行預測
    print("載入模型並進行全區間預測...")
    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    ignore_cols = ["stock_id", "date"] + target_cols
    
    feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        print("[錯誤] 找不到 feature_cols.json")
        return

    X_sim = df_sim.reindex(columns=feature_cols).astype(np.float32)
    models = load_models()

    for days in [1, 2, 3]:
        preds = models[days].predict(X_sim)
        # Bug 4 修復：嚴格確保輸出格式為 3 類別
        if len(preds.shape) == 2 and preds.shape[1] == 3:
            prob_strong = preds[:, 2]
            prob_weak = preds[:, 0]
        else:
            print(f"[錯誤] 模型輸出格式不如預期 (preds.shape={preds.shape})，請確認是否為 3 類別分類模型！")
            sys.exit(1)
            
        df_sim[f'Day{days}_net'] = (prob_strong - prob_weak) * 100

    # 3. 模擬交易迴圈
    print("開始模擬每日交易...\n")
    
    # T+2 餘額交割帳戶設計
    available_cash = initial_capital  # 可用資金 (購買力)
    bank_cash = initial_capital       # 銀行帳戶實質餘額
    pending_settlements = {}          # 預計交割項目: date_obj -> net_amount (T+1/T+2 待交割款)

    positions = {}  # stock_id -> {'shares': X, 'buy_price': Y, 'buy_date': Z}
    history = []
    trades = []     # 交易明細紀錄
    
    dates = sorted(df_sim['date'].unique())
    
    # 交易手續費與稅金設定
    FEE_RATE = 0.001425  # 券商手續費
    TAX_RATE = 0.003     # 證交稅
    
    # 嘗試讀取 stock_categories.json 來建立股票名稱對照表
    stock_names = {}
    cat_path = os.path.join(BASE_DIR, "stock_categories.json")
    if os.path.exists(cat_path):
        try:
            with open(cat_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            for cat, stocks in categories.items():
                for sid, sname in stocks.items():
                    stock_names[str(sid)] = sname
        except Exception as e:
            print(f"[警告] 讀取 stock_categories.json 失敗: {e}")

    for idx_today, today in enumerate(dates):
        # 紀錄今天新增交易記錄的起點索引，以便在收盤後統一更新資金欄位為日終一致狀態 (解決問題 2)
        start_trade_idx = len(trades)
        
        # --- A0. 處理今日交割款 (T+2 餘額交割) ---
        today_date = today.date()
        settled_amount = 0
        for s_date in list(pending_settlements.keys()):
            if s_date <= today_date:
                settled_amount += pending_settlements.pop(s_date)
        bank_cash += settled_amount
        
        today_data = df_sim[df_sim['date'] == today].set_index('stock_id')
        
        # --- A. 更新今日價格 ---
        for sid, pos in positions.items():
            if sid in today_data.index:
                new_price = today_data.loc[sid, 'close']
                if not pd.isna(new_price) and new_price > 0:
                    pos['current_price'] = new_price

        # --- B. 賣出邏輯 (Day3 分數 < 0，或是跌破 -8% 停損) ---
        sells_today = []
        for sid, pos in list(positions.items()):
            if sid in today_data.index:
                day3_score = today_data.loc[sid, 'Day3_net']
                actual_cost_price = pos['buy_price'] * (1 + FEE_RATE)
                current_profit_pct = (pos['current_price'] - actual_cost_price) / actual_cost_price * 100
                
                # 賣出條件 1: 模型預測未來3天轉弱
                # 賣出條件 2: 觸發 -8% 固定停損保險機制
                if day3_score < 0:
                    sells_today.append((sid, "Day3預測轉弱"))
                elif current_profit_pct <= -8.0:
                    sells_today.append((sid, "觸發-8%停損"))
                    
        today_sells_amount = 0
        for sid, reason in sells_today:
            pos = positions.pop(sid)
            sell_price = pos['current_price']
            gross = pos['shares'] * sell_price
            net_proceeds = gross * (1 - FEE_RATE - TAX_RATE)
            
            # 賣出當天即刻增加可用資金 (購買力)
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

        # --- C. 買進邏輯 (挑選 Day1 最強的股票) ---
        # 排除已持有的股票，依 Day 1 分數降冪排序
        buy_candidates = today_data[~today_data.index.isin(positions.keys())].copy()
        buy_candidates = buy_candidates.sort_values('Day1_net', ascending=False)
        
        today_buys_amount = 0
        # Bug 2 修復：將判定是否已滿倉的邏輯移至迴圈內
        for sid in buy_candidates.index:
            if len(positions) >= max_positions:
                break
                
            day1_score = today_data.loc[sid, 'Day1_net']
            if day1_score < 10:
                continue
                
            buy_price = today_data.loc[sid, 'close']
            if pd.isna(buy_price) or buy_price <= 0:
                continue
            
            # Bug 3 修復：每次買進時，根據「剩餘可用資金(購買力)」與「剩餘槽位」動態計算可投入金額
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
                    'current_price': buy_price,
                    'buy_date': today
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
                    'Reason': f"D1分數強勢 ({day1_score:.1f}%)"
                })

        # --- C2. 計算今日交易之 T+2 淨交割金額並加入待交割佇列 ---
        net_change = today_sells_amount - today_buys_amount
        if net_change != 0:
            # 尋找 T+2 交割日 (以 dates 列表為準)
            if idx_today + 2 < len(dates):
                settlement_date = dates[idx_today + 2].date()
            else:
                # 若超出模擬區間，則估計下兩個工作日
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

        # --- D. 計算今日最終淨值並寫入 history (Bug 1 修復) ---
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

        # 統一將今天發生的所有交易明細中的資金欄位更新為當天收盤後的最終狀態 (解決問題 2)
        for i in range(start_trade_idx, len(trades)):
            trades[i]['Current_Cash'] = available_cash
            trades[i]['Stock_Value'] = final_stock_value
            trades[i]['Total_Equity'] = current_equity

    # 在迴圈外保留原本變數以相容後續結算報告與 Excel 寫入
    cash = available_cash

    # 4. 結算與報表
    print("\n" + "=" * 70)
    print("  回測結束 - 績效結算")
    print("=" * 70)
    
    # Bug 6 修復：獨立重算最後一天的總淨值
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

    # 將 DataFrame 的欄位名稱翻譯成中文再匯出
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
        
        # 嘗試匯出為單一 Excel (多個分頁)
        try:
            excel_path = os.path.join(report_dir, f"backtest_report_{safe_start}_{safe_end}.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_history.to_excel(writer, sheet_name='Equity_Curve', index=False)
                df_trades.to_excel(writer, sheet_name='Trade_History', index=False)
                df_holdings.to_excel(writer, sheet_name='Final_Holdings', index=False)
            print(f"\n[成功] 完整回測報表已匯出 (Excel格式): {excel_path}")
        except ImportError:
            # 若無 openpyxl，降級匯出為三個獨立 CSV
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
            print(" (提示: 執行 pip install openpyxl 即可匯出包含分頁的單一 Excel 檔)")
    except Exception as e:
        print(f"匯出報表失敗: {e}")
        
    print("=" * 70)

class CustomHelpParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"\n[參數輸入錯誤] {message}")
        print("=" * 70)
        print(" 正確的執行範例:")
        print("   python trading_sim.py --start 2026-01-02 --end 2026-06-25 --capital 1000000 --max_pos 5\n")
        print("參數說明:")
        print("  --start    回測起始日期 (預設: 2026-01-01，格式: YYYY-MM-DD)")
        print("  --end      回測結束日期 (預設: 2026-06-30，格式: YYYY-MM-DD)")
        print("  --capital  初始資金 (預設: 1000000)")
        print("  --max_pos  最大持股檔數 (預設: 5)")
        print("=" * 70 + "\n")
        sys.exit(2)

if __name__ == "__main__":
    parser = CustomHelpParser(description="量化模型自動交易回測")
    parser.add_argument("--start", type=str, default="2026-01-01", help="回測起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-06-30", help="回測結束日期 (YYYY-MM-DD)")
    parser.add_argument("--capital", type=int, default=1000000, help="初始資金")
    parser.add_argument("--max_pos", type=int, default=5, help="最大持股檔數")
    
    args = parser.parse_args()
    run_simulation(args.start, args.end, args.capital, args.max_pos)
