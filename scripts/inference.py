# -*- coding: utf-8 -*-
"""
inference.py — LightGBM 模型多天期推理 (極簡流線版)
====================================================
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import lightgbm as lgb
import pandas as pd
import numpy as np

# 統一設定路徑與環境
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 載入中央控制面板 config ──────────────────────────────────
try:
    from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS_PCT, MAX_POSITIONS, FEE_RATE, TAX_RATE
except ImportError:
    BUY_THRESHOLD   = 10.0   
    SELL_THRESHOLD  = 0.0    
    STOP_LOSS_PCT   = -8.0   
    FEE_RATE        = 0.001425
    TAX_RATE        = 0.003
    MAX_POSITIONS   = 5


def load_watchlist_detailed() -> dict:
    try:
        from scripts.utils import parse_stocks_detailed
        return parse_stocks_detailed("Stocks.txt")
    except ImportError:
        try:
            from utils import parse_stocks_detailed
            return parse_stocks_detailed("Stocks.txt")
        except Exception:
            return {}


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


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[錯誤] 找不到特徵檔: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    
    # ── 根據 config.py 中的 TRAIN_INDUSTRIES 過濾股票 ──────────────────
    try:
        from scripts.utils import filter_stocks_by_train_industries
        before_cnt = df["stock_id"].nunique()
        df = filter_stocks_by_train_industries(df)
        after_cnt = df["stock_id"].nunique()
        print(f"  [推理過濾] 依 config 產業設定篩選：原本 {before_cnt} 檔，剩餘 {after_cnt} 檔進行推理")
    except Exception as e:
        print(f"  [警告] 篩選過濾器執行失敗 ({e})，使用全特徵進行推理")
    # ──────────────────────────────────────────────────────────
    
    latest_date = df["date"].max()
    date_str = latest_date.strftime("%Y%m%d")

    watchlist = load_watchlist_detailed()
    if not watchlist:
        watchlist = {sid: {"cost": None, "shares": None} for sid in sorted(df["stock_id"].unique().tolist())}

    df_latest_market = df[df["date"] == latest_date].copy()
    if df_latest_market.empty:
        print("[錯誤] 最新日期無任何股票資料。")
        return

    target_cols = ["next_ret_1", "next_ret_2", "next_ret_3"]
    ignore_cols = ["stock_id", "date"] + target_cols

    feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        numeric_cols = df_latest_market.select_dtypes(include=[np.number, bool]).columns
        feature_cols = [c for c in numeric_cols if c not in ignore_cols]

    X_latest_market = df_latest_market.reindex(columns=feature_cols).astype(np.float32)
    results_market = df_latest_market[["stock_id"]].copy()
    results_market["close"] = df_latest_market["close"].values if "close" in df_latest_market.columns else 0.0

    for days in [1, 2, 3]:
        model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days}.txt")
        if not os.path.exists(model_path):
            print(f"[錯誤] 找不到模型 {model_path}")
            return
        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X_latest_market)

        if len(preds.shape) == 2 and preds.shape[1] == 3:
            prob_strong = preds[:, 2]
            prob_weak   = preds[:, 0]
        else:
            print(f"[錯誤] 模型輸出格式不符，預期 3 類別！")
            return

        net_score = prob_strong - prob_weak
        results_market[f"Day{days}_net"]  = net_score * 100
        results_market[f"Day{days}_weak"] = prob_weak * 100

    stock_names = {}
    etf_set = set()
    cat_path = os.path.join(BASE_DIR, "scripts", "stock_categories.json")
    if os.path.exists(cat_path):
        try:
            with open(cat_path, "r", encoding="utf-8") as f:
                categories = json.load(f)
            for cat, stocks in categories.items():
                for sid, sname in stocks.items():
                    stock_names[str(sid)] = sname
            etf_set = set(str(k) for k in categories.get("ETF", {}).keys())
        except Exception:
            pass

    results_watchlist = (
        results_market[results_market["stock_id"].isin(watchlist.keys())]
        .copy()
        .sort_values("Day1_net", ascending=False)
        .reset_index(drop=True)
    )

    results_top5    = results_market.sort_values("Day1_net", ascending=False).head(5).reset_index(drop=True)
    results_bottom5 = results_market.sort_values("Day1_net", ascending=True ).head(5).reset_index(drop=True)

    factors_file = os.path.join(BASE_DIR, "best_factors.json")
    factors_info = "  [因子參數] 未找到最佳化參數，使用系統預設值"
    if os.path.exists(factors_file):
        try:
            with open(factors_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = data.get("best_params_for_run_feature_engineering", {})
            factors_info = f"  [因子參數] MA={p.get('MA_WINDOWS')} | RSI={p.get('RSI_PERIOD')} | ATR={p.get('ATR_PERIOD')} | KD={p.get('KD_PERIOD')} | MACD={p.get('MACD_FAST')}/{p.get('MACD_SLOW')}/{p.get('MACD_SIGNAL')} | Boll={p.get('BOLL_WINDOW')}/{p.get('BOLL_STD_MULT')} | VolMA={p.get('VOL_MA_WINDOW')} | Chips={p.get('CHIPS_SUM_WINDOWS')}"
        except Exception:
            pass

    output_lines = []
    
    stocks_file = os.path.join(BASE_DIR, "Stocks.txt")
    has_real_watchlist = os.path.exists(stocks_file) and os.path.getsize(stocks_file) > 0
    
    if has_real_watchlist:
        actual_holdings = {sid: info["cost"] for sid, info in watchlist.items() if info["cost"] is not None}
        track_only = {sid: info["cost"] for sid, info in watchlist.items() if info["cost"] is None}
    else:
        actual_holdings = {}
        track_only = {}

    if has_real_watchlist:
        output_lines.append("\n" + "=" * 108)
        output_lines.append(f"   [持倉與自選追蹤表] 預測基準日: {latest_date.date()}")
        output_lines.append("=" * 108)
        output_lines.append(
            f"{'排名':<3} | {'類型':<2} | {'股票 (代號+名稱)':<14} | {'收盤價':<6} | "
            f"{'D1多空':<6} | {'D2多空':<6} | {'D3多空':<6} | {'買進成本':<6} | {'持股股數':<6} | {'損益%':<6} | {'出場損益':<8} | {'趨勢'}"
        )
        output_lines.append("-" * 108)

        for i, row in enumerate(results_watchlist.itertuples(), start=1):
            sid        = row.stock_id
            close_p    = row.close
            d1, d2, d3 = row.Day1_net, row.Day2_net, row.Day3_net
            cname      = stock_names.get(sid, "")
            disp_name  = f"{sid} {cname}"

            d1_s = f"{d1:+.1f}%"
            d2_s = f"{d2:+.1f}%"
            d3_s = f"{d3:+.1f}%"

            is_holding = sid in actual_holdings
            type_s = "持倉" if is_holding else "自選"
            
            info = watchlist.get(sid, {"cost": None, "shares": None})
            cost = info.get("cost")
            shares = info.get("shares")
            
            pnl_pct = 0.0
            pnl_amt = 0.0
            
            if is_holding and cost and cost > 0:
                actual_cost = cost * (1 + FEE_RATE)
                pnl_pct     = (close_p - actual_cost) / actual_cost * 100
                cost_s      = f"{cost:.2f}"
                pnl_s       = f"{pnl_pct:+.1f}%"
                
                if shares and shares > 0:
                    shares_s = f"{shares:,}"
                    invested_amt = cost * shares * (1 + FEE_RATE)
                    current_val = close_p * shares * (1 - FEE_RATE - TAX_RATE)
                    pnl_amt = current_val - invested_amt
                    pnl_amt_s = f"{pnl_amt:+.0f}"
                else:
                    shares_s = "  --  "
                    pnl_amt_s = "   --   "
            else:
                cost_s = "  --  "
                shares_s = "  --  "
                pnl_s  = "  --  "
                pnl_amt_s = "   --   "

            if   d1 > 12 and d2 > 8:   trend = "強勢多"
            elif d1 > 5  and d2 > 0:   trend = "偏多"
            elif d1 < -15 and d2 < -15: trend = "極空"
            elif d1 < -5  and d2 < -5:  trend = "偏空"
            elif d1 > 8  and d3 < 0:   trend = "短多"
            else:                        trend = "震盪"

            output_lines.append(
                f" {i:<3} | {type_s:<2} | {disp_name:<18} | {close_p:>6.2f} | "
                f"{d1_s:>6} | {d2_s:>6} | {d3_s:>6} | "
                f"{cost_s:>6} | {shares_s:>6} | {pnl_s:>6} | {pnl_amt_s:>8} |  {trend}"
            )

        output_lines.append("=" * 108)
    else:
        output_lines.append("\n" + "=" * 90)
        output_lines.append("   [提示] 目前為全市場掃描。可在 Stocks.txt 中填寫「代號,買入價,股數」啟用損益追蹤。")
        output_lines.append("=" * 90)

    sells_list = []
    for sid, cost in actual_holdings.items():
        row_match = results_market[results_market["stock_id"] == sid]
        if not row_match.empty:
            row = row_match.iloc[0]
            d3 = row["Day3_net"]
            close_p = row["close"]
            reasons = []
            if d3 < SELL_THRESHOLD:
                reasons.append(f"D3轉弱({d3:+.1f}%)")
            pnl_pct = 0.0
            if cost and cost > 0:
                actual_cost = cost * (1 + FEE_RATE)
                pnl_pct = (close_p - actual_cost) / actual_cost * 100
                if pnl_pct <= STOP_LOSS_PCT:
                    reasons.append(f"停損({pnl_pct:.1f}%)")
            if reasons:
                sells_list.append({
                    "stock_id": sid,
                    "name": stock_names.get(sid, ""),
                    "close": close_p,
                    "reason": "、".join(reasons)
                })

    sells_sids = {x["stock_id"] for x in sells_list}
    remaining_holdings = set(actual_holdings.keys()) - sells_sids
    current_hold_count = len(actual_holdings)
    sells_count = len(sells_list)
    remaining_hold_count = len(remaining_holdings)
    available_slots = max(0, MAX_POSITIONS - remaining_hold_count)

    output_lines.append("")
    output_lines.append("=" * 90)
    output_lines.append(f"   [實戰下單指令] 買進D1 >= {BUY_THRESHOLD:.0f}% | 賣出D3 < {SELL_THRESHOLD:.0f}% | 停損 {STOP_LOSS_PCT:.0f}%")
    output_lines.append("=" * 90)

    output_lines.append("   明日建議賣出掛單 (開盤賣出釋出倉位):")
    if sells_list:
        for item in sells_list:
            output_lines.append(f"     賣出 -> {item['stock_id']:<5} {item['name']:<5} (收盤 {item['close']:.2f} | 原因: {item['reason']})")
    else:
        output_lines.append("    [v] 目前無持倉股觸發賣出/停損訊號")
    output_lines.append("-" * 90)

    output_lines.append(f"   明日建議買進掛單 (持倉:{current_hold_count} -> 開盤剩餘:{remaining_hold_count} | 需填補空位:{available_slots} 檔):")
    
    buy_cond = (
        (results_market["Day1_net"] >= BUY_THRESHOLD) &
        (~results_market["stock_id"].isin(remaining_holdings)) &
        (~results_market["stock_id"].isin(sells_sids)) &
        (~results_market["stock_id"].astype(str).isin(etf_set))
    )
    buy_candidates = results_market[buy_cond].sort_values("Day1_net", ascending=False).head(5)
    
    if buy_candidates.empty:
        output_lines.append("     全市場無符合條件之標的")
    else:
        for idx, row in enumerate(buy_candidates.itertuples(), start=1):
            sid   = str(row.stock_id)
            cname = stock_names.get(sid, "")
            d1    = row.Day1_net
            d3    = row.Day3_net
            close_p = row.close
            
            # 根據 D1 多空信心分數動態決定建議加價幅度 (D1 >= 30% 建議 +2.5%; >= 20% 建議 +2.0%; 否則 +1.5%)
            if d1 >= 30.0:
                target_pct = 2.5
            elif d1 >= 20.0:
                target_pct = 2.0
            else:
                target_pct = 1.5
            
            raw_target_price = close_p * (1 + target_pct / 100.0)
            target_price = round_to_tick(raw_target_price)
            
            tag = "[優先買進]" if (available_slots > 0 and idx <= available_slots) else "[觀察遞補]"
            output_lines.append(
                f"    {tag} 第 {idx} 檔 -> {sid:<5} {cname:<5} (收盤 {close_p:>6.2f} | D1 {d1:>+5.1f}% | D3 {d3:>+5.1f}% | 建議掛 {target_pct:+.1f}% -> {target_price:>6.2f})"
            )
        output_lines.append("")
        output_lines.append("     [掛單提醒] 建議掛單價已依 D1 信心動態加價並自動對齊台股 Tick 升降單位 (D1>=30%加2.5%, >=20%加2.0%, 其他加1.5%)。")
    output_lines.append("-" * 90)

    if has_real_watchlist and len(track_only) > 0:
        track_buys_df = results_watchlist[
            (results_watchlist["stock_id"].isin(track_only.keys())) &
            (results_watchlist["Day1_net"] >= BUY_THRESHOLD)
        ]
        if not track_buys_df.empty:
            output_lines.append("   自選追蹤強勢股已達買進門檻 (D1 >= +10.0%):")
            for row in track_buys_df.itertuples():
                sid   = str(row.stock_id)
                cname = stock_names.get(sid, "")
                d1    = row.Day1_net
                output_lines.append(f"     {sid:<5} {cname:<5} (D1 {d1:>+5.1f}%)")
            output_lines.append("-" * 90)

    buy_sids = [str(x) for x in buy_candidates["stock_id"].tolist()]
    top5_sids = [str(x) for x in results_top5["stock_id"].tolist()]
    if set(buy_sids) == set(top5_sids):
        output_lines.append("   全市場強勢 Top-5: (與上方建議買進清單相同，已合併顯示)")
    else:
        output_lines.append("   全市場強勢 Top-5:")
        for i, row in enumerate(results_top5.itertuples(), 1):
            sid   = str(row.stock_id)
            cname = stock_names.get(sid, "")
            output_lines.append(f"    {i}. {sid:<5} {cname:<5} (收盤 {row.close:>6.2f} | D1 {row.Day1_net:>+5.1f}% | D3 {row.Day3_net:>+5.1f}%)")

    output_lines.append("")
    output_lines.append("   全市場弱勢 Bottom-5 (建議避開):")
    for i, row in enumerate(results_bottom5.itertuples(), 1):
        sid   = str(row.stock_id)
        cname = stock_names.get(sid, "")
        output_lines.append(f"    {i}. {sid:<5} {cname:<5} (收盤 {row.close:>6.2f} | D1 {row.Day1_net:>+5.1f}% | D3 {row.Day3_net:>+5.1f}%)")

    output_lines.append("=" * 90)
    output_lines.append(factors_info)
    output_lines.append("=" * 90)
    output_lines.append("  [聲明] 本預測僅供研究參考，不構成實際投資建議。")

    final_output = "\n".join(output_lines)
    print(final_output)

    pred_dir = os.path.join(BASE_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_file = os.path.join(pred_dir, f"prediction_{date_str}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_output)
    print(f"\n  [儲存] 本次預測結果已存檔至: {out_file}")


if __name__ == "__main__":
    main()
