# -*- coding: utf-8 -*-
"""
analyze_real_pnl.py — 台股實盤交易損益與永續持倉追蹤系統
==================================================
本腳本會自動讀取根目錄最新下載的「交易紀錄.xls」，
與本地累計歷史交易檔案「real_trade_history.csv」進行增量合併（多重集聯集去重），
解決證券商系統 Excel 僅保留 30 天紀錄的問題。
接著，計算出系統實盤的已實現損益（已平倉交易）與未實現損益（浮動持倉），
並將詳細報告輸出至終端機與 reports/real_trading_report.md。
"""

import os
import re
import sys
import glob
import shutil
import pandas as pd
import numpy as np
from datetime import datetime

# 設定根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入中央常數控制
from config import FEE_RATE, TAX_RATE

def parse_shares(qty_str):
    """解析張數或股數，若包含『張』字眼則自動乘以 1000"""
    if pd.isna(qty_str):
        return 0
    qty_str = str(qty_str).strip()
    match = re.search(r"(\d+(\.\d+)?)", qty_str)
    if not match:
        return 0
    val = float(match.group(1))
    if "張" in qty_str:
        return int(val * 1000)
    return int(val)

def parse_stock_info(info_str):
    """解析股票代號與名稱 (例如 '2415 - 錩新' -> '2415', '錩新')"""
    if pd.isna(info_str):
        return "UNKNOWN", "UNKNOWN"
    info_str = str(info_str).strip()
    parts = info_str.split("-")
    sid = parts[0].strip()
    sname = parts[1].strip() if len(parts) > 1 else sid
    return sid, sname

def get_latest_prices():
    """從 data/raw_price/ 中讀取最新一天的股票收盤價"""
    from config import RAW_PRICE_DIR as price_dir   # 唯一來源：config.py § 0
    csv_files = sorted(glob.glob(os.path.join(price_dir, "*_price.csv")))
    if not csv_files:
        return {}, "無資料"
    latest_file = csv_files[-1]
    date_str = os.path.basename(latest_file).split("_")[0]
    try:
        date_formatted = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        date_formatted = date_str
        
    df = pd.read_csv(latest_file)
    sid_col = df.columns[0]
    close_col = df.columns[8]
    
    prices = {}
    for _, row in df.iterrows():
        sid = str(row[sid_col]).strip()
        val = row[close_col]
        try:
            if pd.isna(val):
                continue
            val_str = str(val).replace(",", "").strip()
            if val_str == '--':
                continue
            prices[sid] = float(val_str)
        except ValueError:
            pass
    return prices, date_formatted

def merge_trade_history(history_df, new_df):
    """
    增量合併歷史紀錄與新下載紀錄。
    使用多重集聯集 (Multiset Union) 演算法，在去重合併的同時，
    保留同一天內交易特徵（價格、張數、金額等）完全一致的多筆合法交易。
    """
    if history_df is None or history_df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return history_df

    feature_cols = [
        "stock_info", "trade_type", "price", "qty_str", 
        "fee", "tax", "borrow_fee", "amount", "date"
    ]
    
    # 確保兩者都只保留特徵欄位，且日期格式一致
    history_df = history_df[feature_cols].copy()
    new_df = new_df[feature_cols].copy()
    
    history_df["date"] = pd.to_datetime(history_df["date"]).dt.strftime("%Y-%m-%d")
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    
    # 分組統計各特徵組合的出現次數
    hist_counts = history_df.groupby(feature_cols).size().to_frame('hist_count')
    new_counts = new_df.groupby(feature_cols).size().to_frame('new_count')
    
    # 外部合併次數
    merged_counts = hist_counts.join(new_counts, how='outer').fillna(0)
    
    # 聯集取最大值：若 30 天前交易，new_count=0, 取 hist_count；
    # 若為 30 天內新增交易，new_count>=hist_count, 取 new_count，完美去重並追加。
    merged_counts['final_count'] = merged_counts[['hist_count', 'new_count']].max(axis=1)
    
    rows = []
    for idx, row in merged_counts.iterrows():
        count = int(row['final_count'])
        if count > 0:
            for _ in range(count):
                rows.append(idx)
                
    result_df = pd.DataFrame(rows, columns=feature_cols)
    return result_df

def main():
    history_dir = os.path.join(BASE_DIR, "trading_history")
    os.makedirs(history_dir, exist_ok=True)
    
    xls_path = os.path.join(history_dir, "交易紀錄.xls")
    history_csv_path = os.path.join(history_dir, "real_trade_history.csv")
    
    # 檢查是否有 any 有效的交易紀錄來源
    has_xls = os.path.exists(xls_path)
    has_history = os.path.exists(history_csv_path)
    
    if not has_xls and not has_history:
        return
 
    new_df = None
    if has_xls:
        print("正在讀取新下載的交易紀錄...")
        try:
            new_df = pd.read_excel(xls_path)
            if len(new_df.columns) < 9:
                print("[警告] 交易紀錄.xls 的欄位數量不足 9 個，將略過此 Excel 檔案。")
                new_df = None
            else:
                # 重命名欄位
                mapping = {
                    new_df.columns[0]: "stock_info",
                    new_df.columns[1]: "trade_type",
                    new_df.columns[2]: "price",
                    new_df.columns[3]: "qty_str",
                    new_df.columns[4]: "fee",
                    new_df.columns[5]: "tax",
                    new_df.columns[6]: "borrow_fee",
                    new_df.columns[7]: "amount",
                    new_df.columns[8]: "date"
                }
                new_df = new_df.rename(columns=mapping)
        except Exception as e:
            print(f"[警告] 無法讀取 Excel 檔案，將僅使用本地歷史紀錄: {e}")
            new_df = None
 
    # 讀取現有歷史紀錄（先讀取，實際備份延後到確認有新增交易時才進行）
    history_df = None
    if has_history:
        print("發現既有歷史紀錄檔案 real_trade_history.csv...")
        try:
            history_df = pd.read_csv(history_csv_path)
        except Exception as e:
            print(f"\n[嚴重錯誤] 無法讀取已存在的歷史交易紀錄 real_trade_history.csv: {e}")
            print("為保護歷史交易資料不被截斷覆寫，程式已立即安全中止。")
            print("請確認該 CSV 檔案是否正被 Excel 等軟體開啟鎖定，或檔案格式是否毀損。")
            sys.exit(1)

    # 合併新舊資料
    if new_df is not None or history_df is not None:
        merged_df = merge_trade_history(history_df, new_df)
    else:
        print("[警告] 無法取得任何有效的交易紀錄，跳過損益計算。")
        return
    
    # 確保依日期排序
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df = merged_df.sort_values("date", ascending=True).reset_index(drop=True)
    
    # 判斷合併後是否真的新增了交易（多重集聯集必為超集，筆數相等即代表無新增）
    hist_count = len(history_df) if history_df is not None else 0
    new_record_count = len(merged_df) - hist_count
    has_new_records = has_xls and new_df is not None and new_record_count > 0

    if has_new_records:
        # 僅在確定有新增交易時，才備份舊檔並寫回，避免每次執行都產生無意義的 .bak
        if has_history:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(history_dir, f"real_trade_history_{timestamp}.csv.bak")
                shutil.copy2(history_csv_path, backup_path)
                print(f"  [安全備份] 已成功備份舊歷史紀錄至: {os.path.basename(backup_path)}")

                # 自動清理過舊的備份檔，只保留最近 5 份
                backups = sorted(glob.glob(os.path.join(history_dir, "real_trade_history_*.csv.bak")))
                if len(backups) > 5:
                    for old_backup in backups[:-5]:
                        os.remove(old_backup)
                        print(f"  [備份清理] 已自動清理過期備份: {os.path.basename(old_backup)}")
            except Exception as e:
                print(f"  [警告] 備份或清理舊檔案時失敗，但流程將繼續: {e}")

        save_df = merged_df.copy()
        save_df["date"] = save_df["date"].dt.strftime("%Y-%m-%d")
        save_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
        print(f"合併完成！本次新增 {new_record_count} 筆，共計 {len(save_df)} 筆永續交易紀錄，已更新至 real_trade_history.csv。")
    elif has_xls and new_df is not None:
        # 有讀到 Excel 但無新增交易，略過備份與 CSV 寫入，但照常重新計算最新持倉損益並更新報告
        print("本次交易紀錄.xls 無新增交易，歷史紀錄維持不變，將照常更新最新持倉損益報告。")

    # 載入最新收盤價
    latest_prices, price_date = get_latest_prices()

    # FIFO 配對變數
    buy_queues = {}     # {sid: [{"date": ..., "price": ..., "shares": ..., "amount": ...}]}
    closed_trades = []  # 已平倉交易明細
    
    for _, row in merged_df.iterrows():
        sid, sname = parse_stock_info(row["stock_info"])
        trade_type = row["trade_type"]
        price = float(row["price"])
        shares = parse_shares(row["qty_str"])
        amount = float(row["amount"])
        date_str = row["date"].strftime("%Y-%m-%d")
        
        is_buy = "買進" in trade_type or "\u8cb7\u9032" in trade_type
        is_sell = "賣出" in trade_type or "\u8ce3\u51fa" in trade_type
        
        if is_buy:
            if sid not in buy_queues:
                buy_queues[sid] = []
            buy_queues[sid].append({
                "date": date_str,
                "price": price,
                "shares": shares,
                "amount": amount  # 買入為負數，支出
            })
        elif is_sell:
            sell_shares_left = shares
            sell_amount_left = amount  # 賣出為正數，收入
            
            if sid not in buy_queues or not buy_queues[sid]:
                continue
                
            while sell_shares_left > 0 and buy_queues[sid]:
                buy_item = buy_queues[sid][0]
                b_shares = buy_item["shares"]
                b_amount = buy_item["amount"]
                b_price = buy_item["price"]
                b_date = buy_item["date"]
                
                if sell_shares_left < b_shares:
                    matched_b_amount = b_amount * (sell_shares_left / b_shares)
                    matched_pnl = sell_amount_left + matched_b_amount
                    
                    closed_trades.append({
                        "stock_id": sid,
                        "stock_name": sname,
                        "buy_date": b_date,
                        "sell_date": date_str,
                        "shares": sell_shares_left,
                        "buy_price": b_price,
                        "sell_price": price,
                        "pnl": matched_pnl
                    })
                    
                    buy_item["shares"] = b_shares - sell_shares_left
                    buy_item["amount"] = b_amount - matched_b_amount
                    sell_shares_left = 0
                else:
                    matched_b_amount = b_amount
                    matched_sell_amount = sell_amount_left * (b_shares / sell_shares_left)
                    matched_pnl = matched_sell_amount + matched_b_amount
                    
                    closed_trades.append({
                        "stock_id": sid,
                        "stock_name": sname,
                        "buy_date": b_date,
                        "sell_date": date_str,
                        "shares": b_shares,
                        "buy_price": b_price,
                        "sell_price": price,
                        "pnl": matched_pnl
                    })
                    
                    sell_shares_left -= b_shares
                    sell_amount_left -= matched_sell_amount
                    buy_queues[sid].pop(0)

    # 彙整在庫持倉
    open_positions = []
    for sid, queue in buy_queues.items():
        active_items = [item for item in queue if item["shares"] > 0]
        if not active_items:
            continue
            
        sname = "未知"
        for _, row in merged_df.iterrows():
            rsid, rsname = parse_stock_info(row["stock_info"])
            if rsid == sid:
                sname = rsname
                break
                
        total_shares = sum(item["shares"] for item in active_items)
        total_cost = sum(abs(item["amount"]) for item in active_items)
        avg_price = total_cost / total_shares if total_shares > 0 else 0
        
        cur_price = latest_prices.get(sid, None)
        if cur_price is not None:
            market_value = total_shares * cur_price
            # 從 config 載入的 FEE_RATE 與 TAX_RATE 進行未實現損益預估
            estimated_sell_fee = market_value * FEE_RATE
            estimated_sell_tax = market_value * TAX_RATE
            net_market_value = market_value - estimated_sell_fee - estimated_sell_tax
            unrealized_pnl = net_market_value - total_cost
            roi = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0
        else:
            cur_price = np.nan
            market_value = np.nan
            unrealized_pnl = np.nan
            roi = np.nan
            
        open_positions.append({
            "stock_id": sid,
            "stock_name": sname,
            "shares": total_shares,
            "total_cost": total_cost,
            "avg_cost_price": avg_price,
            "current_price": cur_price,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "roi": roi
        })

    # 績效統計
    total_realized_pnl = sum(t["pnl"] for t in closed_trades)
    winning_trades = [t for t in closed_trades if t["pnl"] > 0]
    losing_trades = [t for t in closed_trades if t["pnl"] <= 0]
    total_trades_count = len(closed_trades)
    win_rate = (len(winning_trades) / total_trades_count * 100) if total_trades_count > 0 else 0
    
    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = sum(abs(t["pnl"]) for t in losing_trades)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 1.0)
    
    avg_win = (gross_profit / len(winning_trades)) if len(winning_trades) > 0 else 0
    avg_loss = (gross_loss / len(losing_trades)) if len(losing_trades) > 0 else 0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else np.inf

    total_unrealized_pnl = sum(p["unrealized_pnl"] for p in open_positions if not pd.isna(p["unrealized_pnl"]))
    total_portfolio_cost = sum(p["total_cost"] for p in open_positions)
    total_portfolio_value = sum(p["market_value"] for p in open_positions if not pd.isna(p["market_value"]))
    
    # 輸出終端機
    print("\n" + "=" * 70)
    print("                台股實盤交易系統損益與持倉報告")
    print("=" * 70)
    print(f"  報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  市價日期: {price_date}")
    print("-" * 70)
    print(f"  【已實現績效指標】 (已平倉累計)")
    print(f"    * 已實現總損益 : {total_realized_pnl:,.0f} 元")
    print(f"    * 總交易筆數   : {total_trades_count} 筆")
    print(f"    * 勝率 (Win %) : {win_rate:.1f}% ({len(winning_trades)} 勝 / {len(losing_trades)} 敗)")
    print(f"    * 獲利因子 (PF): {profit_factor:.2f}")
    print(f"    * 均盈 / 均虧  : {avg_win:,.0f} 元 / {avg_loss:,.0f} 元 (盈虧比: {win_loss_ratio:.2f})")
    print("-" * 70)
    print(f"  【未實現持倉損益】 (手頭庫存)")
    if open_positions:
        print(f"    * 持倉總成本   : {total_portfolio_cost:,.0f} 元")
        if not pd.isna(total_portfolio_value):
            print(f"    * 持倉預估市值 : {total_portfolio_value:,.0f} 元")
            print(f"    * 未實現總損益 : {total_unrealized_pnl:,.0f} 元")
            portfolio_roi = (total_unrealized_pnl / total_portfolio_cost) * 100 if total_portfolio_cost > 0 else 0
            print(f"    * 預估浮動投報 : {portfolio_roi:.2f}%")
        else:
            print(f"    * 未實現總損益 : 無最新市價，無法估算")
    else:
        print("    * 目前無任何庫存持倉。")
    print("=" * 70)

    # 輸出明細
    print("\n【已平倉交易明細】")
    if closed_trades:
        print(f"{'股票代號':<8}{'名稱':<8}{'買入日期':<12}{'賣出日期':<12}{'張數':<6}{'買入均價':<10}{'賣出均價':<10}{'淨損益':<10}")
        print("-" * 80)
        for t in closed_trades:
            lots = t["shares"] / 1000
            lots_str = f"{lots:.1f}張" if lots % 1 == 0 or lots >= 0.1 else f"{t['shares']}股"
            print(f"{t['stock_id']:<10}{t['stock_name']:<8}{t['buy_date']:<14}{t['sell_date']:<14}{lots_str:<8}{t['buy_price']:<12.2f}{t['sell_price']:<12.2f}{t['pnl']:+,.0f}")
    else:
        print("  無已平倉歷史交易紀錄。")

    print("\n【在庫持倉庫存明細】")
    if open_positions:
        print(f"{'股票代號':<8}{'名稱':<8}{'張數':<6}{'成本均價':<10}{'最新市價':<10}{'庫存總成本':<12}{'未實現損益':<12}{'預估ROI':<8}")
        print("-" * 90)
        for p in open_positions:
            lots = p["shares"] / 1000
            lots_str = f"{lots:.1f}張" if lots % 1 == 0 or lots >= 0.1 else f"{p['shares']}股"
            pnl_str = f"{p['unrealized_pnl']:+,.0f}" if not pd.isna(p["unrealized_pnl"]) else "N/A"
            roi_str = f"{p['roi']:.2f}%" if not pd.isna(p["roi"]) else "N/A"
            cur_price_str = f"{p['current_price']:.2f}" if not pd.isna(p["current_price"]) else "N/A"
            print(f"{p['stock_id']:<10}{p['stock_name']:<8}{lots_str:<8}{p['avg_cost_price']:<12.2f}{cur_price_str:<12}{p['total_cost']:<14,.0f}{pnl_str:<14}{roi_str:<10}")
    else:
        print("  目前無任何庫存持倉。")

    # 輸出 Markdown 報告
    report_md_path = os.path.join(history_dir, "real_trading_report.md")
    
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 台灣股市量化交易系統 — 實盤交易績效報告\n\n")
        f.write(f"> **報告生成時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"> **最新市價日期**：`{price_date}`  \n")
        f.write(f"> **永續歷史交易檔**：`real_trade_history.csv` (已保存共 {len(merged_df)} 筆歷史交易紀錄)\n\n")
        
        # 提示區塊說明增量更新與安全備份機制
        f.write(f"> [!NOTE]\n")
        f.write(f"> **增量合併與安全備份機制已啟用**：\n")
        f.write(f"> 1. 系統會自動將新下載的 30 天 `交易紀錄.xls` 合併至 `real_trade_history.csv`，永久保存所有交易紀錄。\n")
        f.write(f"> 2. 每次執行時，舊有的歷史資料都會自動被複製為 `real_trade_history_YYYYMMDD_HHMMSS.csv.bak` 進行安全防護。\n")
        f.write(f"> 3. 本腳本之交易摩擦常數已完全同步 `config.py`：手續費率 `{(FEE_RATE*100):.4f}%`、證交稅率 `{(TAX_RATE*100):.2f}%`。\n\n")
        
        # 績效摘要表格
        f.write(f"## 📈 績效摘要指標\n\n")
        f.write(f"| 指標項目 | 數值 | 備註說明 |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        
        pnl_color = "red" if total_realized_pnl >= 0 else "green"
        f.write(f"| **已實現總損益** | <span style='color:{pnl_color}; font-weight:bold;'>{total_realized_pnl:+,.0f} 元</span> | 扣除所有交易費與證交稅 |\n")
        f.write(f"| **總交易筆數** | {total_trades_count} 筆 | 賣出平倉與買入配對筆數 |\n")
        f.write(f"| **勝率 (Win Rate)** | {win_rate:.2f}% | 共 {len(winning_trades)} 勝，{len(losing_trades)} 敗 |\n")
        f.write(f"| **獲利因子 (Profit Factor)** | {profit_factor:.2f} | 總毛利 / 總毛損 |\n")
        f.write(f"| **平均獲利金額** | {avg_win:+,.0f} 元 | 獲利交易之平均獲利 |\n")
        f.write(f"| **平均虧損金額** | -{avg_loss:,.0f} 元 | 虧損交易之平均虧損 |\n")
        f.write(f"| **盈虧比 (W/L Ratio)** | {win_loss_ratio:.2f} | 平均獲利 / 平均虧損 |\n")
        
        if open_positions:
            upnl_color = "red" if total_unrealized_pnl >= 0 else "green"
            portfolio_roi = (total_unrealized_pnl / total_portfolio_cost) * 100 if total_portfolio_cost > 0 else 0
            f.write(f"| **未實現持倉總成本** | {total_portfolio_cost:,.0f} 元 | 當前在庫持倉之總買入金額 |\n")
            if not pd.isna(total_portfolio_value):
                f.write(f"| **持倉預估總市值** | {total_portfolio_value:,.0f} 元 | 依最新收盤價換算市值 |\n")
                f.write(f"| **未實現帳面損益** | <span style='color:{upnl_color}; font-weight:bold;'>{total_unrealized_pnl:+,.0f} 元</span> | 扣除預估賣出費稅之淨損益 |\n")
                f.write(f"| **持倉預估報酬率** | <span style='color:{upnl_color}; font-weight:bold;'>{portfolio_roi:+.2f}%</span> | 帳面損益 / 總成本 |\n")
        f.write("\n")
        
        # 已平倉交易明細
        f.write(f"## 🧾 已平倉交易明細 (已實現)\n\n")
        if closed_trades:
            f.write(f"| 股票代號 | 股票名稱 | 買入日期 | 賣出日期 | 張/股數 | 買入均價 | 賣出均價 | 淨損益 (元) | 交易投報 |\n")
            f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for t in closed_trades:
                lots = t["shares"] / 1000
                lots_str = f"{lots:.1f}張" if lots % 1 == 0 or lots >= 0.1 else f"{t['shares']}股"
                cost_est = t["buy_price"] * t["shares"]
                roi_est = (t["pnl"] / cost_est) * 100 if cost_est > 0 else 0
                
                color = "red" if t["pnl"] >= 0 else "green"
                f.write(f"| `{t['stock_id']}` | {t['stock_name']} | {t['buy_date']} | {t['sell_date']} | {lots_str} | {t['buy_price']:.2f} | {t['sell_price']:.2f} | <span style='color:{color}; font-weight:bold;'>{t['pnl']:+,.0f}</span> | <span style='color:{color};'>{roi_est:+.2f}%</span> |\n")
        else:
            f.write(f"*無已平倉歷史交易紀錄。*\n")
        f.write("\n")
        
        # 在庫持倉明細
        f.write(f"## 💼 在庫持倉庫存明細 (未實現)\n\n")
        if open_positions:
            f.write(f"| 股票代號 | 股票名稱 | 持有張/股數 | 成本均價 | 最新市價 | 庫存總成本 | 未實現損益 | 預估 ROI |\n")
            f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for p in open_positions:
                lots = p["shares"] / 1000
                lots_str = f"{lots:.1f}張" if lots % 1 == 0 or lots >= 0.1 else f"{p['shares']}股"
                
                pnl_val = p["unrealized_pnl"]
                roi_val = p["roi"]
                
                if not pd.isna(pnl_val):
                    color = "red" if pnl_val >= 0 else "green"
                    pnl_str = f"{pnl_val:+,.0f}"
                    roi_str = f"{roi_val:+.2f}%"
                    color_tag_start = f"<span style='color:{color}; font-weight:bold;'>"
                    color_tag_end = "</span>"
                else:
                    color_tag_start = ""
                    color_tag_end = ""
                    pnl_str = "N/A"
                    roi_str = "N/A"
                
                cur_price_str = f"{p['current_price']:.2f}" if not pd.isna(p["current_price"]) else "N/A"
                f.write(f"| `{p['stock_id']}` | {p['stock_name']} | {lots_str} | {p['avg_cost_price']:.2f} | {cur_price_str} | {p['total_cost']:,.0f} | {color_tag_start}{pnl_str}{color_tag_end} | {color_tag_start}{roi_str}{color_tag_end} |\n")
        else:
            f.write(f"*目前無任何庫存持倉。*\n")
            
        f.write("\n---\n")
        f.write("> **💡 交易提示**：本報告由 `analyze_real_pnl.py` 自動根據 `交易紀錄.xls` 配對生成。  \n")
        f.write("> 台股交易成本說明：買入成本為實際支出（含手續費）；持倉未實現損益已預先扣除中央常數設定的手續費率及證交稅率，此為保守預估值。  \n")

    print(f"\n[成功] 詳細的 Markdown 績效報告已輸出至: trading_history/real_trading_report.md")

if __name__ == "__main__":
    main()
