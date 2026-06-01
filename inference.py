# -*- coding: utf-8 -*-
"""
inference.py — LightGBM 模型多天期推理 (Day 1 ~ Day 3)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 與 trading_sim.py 一致的策略參數 ──────────────────────────────────────────
BUY_THRESHOLD   = 10.0   # Day1_net >= 此值才建議買進
SELL_THRESHOLD  = 0.0    # Day3_net <  此值則建議賣出 (持倉股)
STOP_LOSS_PCT   = -8.0   # 持倉跌幅 <= 此值觸發停損建議 (相對買進成本)
FEE_RATE        = 0.001425
TAX_RATE        = 0.003
MAX_BUY_SHOW    = 10     # 買進建議最多顯示幾檔


def get_stock_name(stock_id, date_str):
    price_file = os.path.join(BASE_DIR, "data", "raw_price", f"{date_str}_price.csv")
    if os.path.exists(price_file):
        try:
            df_price = pd.read_csv(price_file, usecols=["證券代號", "證券名稱"], dtype=str)
            df_price["證券代號"] = df_price["證券代號"].str.strip()
            df_price["證券名稱"] = df_price["證券名稱"].str.strip()
            return df_price.set_index("證券代號")["證券名稱"].get(stock_id, "")
        except Exception:
            return ""
    return ""


def load_watchlist_with_cost() -> dict:
    """
    讀取 Stocks.txt。
    呼叫共用 utils.py 的解析函數以維持代碼統一。
    """
    from utils import parse_stocks_file
    return parse_stocks_file("Stocks.txt")


def main():
    print("=" * 75)
    print("  啟動 LightGBM 多天期預測推理 (未來 3 天)")
    print("=" * 75)

    if not os.path.exists(DATA_PATH):
        print(f"[錯誤] 找不到特徵檔: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    latest_date = df["date"].max()
    date_str = latest_date.strftime("%Y%m%d")
    print(f"  最新資料日期: {latest_date.date()}")

    # ── 讀取持倉清單 (Stocks.txt) ───────────────────────────────────────────────
    watchlist = load_watchlist_with_cost()
    if not watchlist:
        print("  [警告] Stocks.txt 不存在或為空，將對全市場所有股票進行預測。")
        watchlist = {sid: None for sid in sorted(df["stock_id"].unique().tolist())}

    print(f"  持倉清單    : {list(watchlist.keys())} ({len(watchlist)} 檔)")

    missing_in_latest = set(watchlist.keys()) - set(df[df["date"] == latest_date]["stock_id"])
    if missing_in_latest:
        print(f"  [警告] 以下股票在最新日期無資料，跳過預測: {list(missing_in_latest)}")

    # ── 對「全市場」所有股票進行預測 ────────────────────────────────────────────
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
        print(f"  特徵欄位來源: models/feature_cols.json ({len(feature_cols)} 欄)")
    else:
        print("  [警告] 找不到 feature_cols.json，改用動態推導特徵欄位 (可能與訓練時不一致)")
        numeric_cols = df_latest_market.select_dtypes(include=[np.number, bool]).columns
        feature_cols = [c for c in numeric_cols if c not in ignore_cols]

    X_latest_market = df_latest_market.reindex(columns=feature_cols).astype(np.float32)

    results_market = df_latest_market[["stock_id"]].copy()
    results_market["close"] = df_latest_market["close"].values if "close" in df_latest_market.columns else 0.0

    for days in [1, 2, 3]:
        model_path = os.path.join(MODEL_DIR, f"lgbm_model_{days}.txt")
        if not os.path.exists(model_path):
            print(f"[錯誤] 找不到模型 {model_path}，請重新執行 train.py")
            return
        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X_latest_market)

        if len(preds.shape) == 2 and preds.shape[1] == 3:
            prob_strong = preds[:, 2]
            prob_weak   = preds[:, 0]
        else:
            print(f"[錯誤] 模型輸出格式不如預期 (preds.shape={preds.shape})，請確認是否為 3 類別分類模型！")
            return

        net_score = prob_strong - prob_weak
        results_market[f"Day{days}_net"]  = net_score * 100
        results_market[f"Day{days}_weak"] = prob_weak * 100

    # ── 讀取股票中文名稱 ────────────────────────────────────────────────────────
    stock_names = {}
    cat_path = os.path.join(BASE_DIR, "stock_categories.json")
    if os.path.exists(cat_path):
        try:
            with open(cat_path, "r", encoding="utf-8") as f:
                categories = json.load(f)
            for cat, stocks in categories.items():
                for sid, sname in stocks.items():
                    stock_names[str(sid)] = sname
        except Exception as e:
            print(f"  [警告] 讀取 stock_categories.json 失敗: {e}")

    # ── 過濾出持倉股票的預測結果 ────────────────────────────────────────────────
    results_watchlist = (
        results_market[results_market["stock_id"].isin(watchlist.keys())]
        .copy()
        .sort_values("Day1_net", ascending=False)
        .reset_index(drop=True)
    )

    # ── 全市場 Top / Bottom 10 ──────────────────────────────────────────────────
    results_top10    = results_market.sort_values("Day1_net", ascending=False).head(10).reset_index(drop=True)
    results_bottom10 = results_market.sort_values("Day1_net", ascending=True ).head(10).reset_index(drop=True)

    # ── 讀取因子參數資訊 ────────────────────────────────────────────────────────
    factors_file = os.path.join(BASE_DIR, "best_factors.json")
    factors_info = "  [未找到最佳化因子檔，使用系統預設因子參數]"
    if os.path.exists(factors_file):
        try:
            with open(factors_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            params   = data.get("best_params_for_run_feature_engineering", {})
            opt_date = data.get("optimized_at", "未知")
            lines = [f"  [使用的因子參數 (最佳化時間: {opt_date})]"]
            for k, v in params.items():
                lines.append(f"    {k:<20} = {v}")
            factors_info = "\n".join(lines)
        except Exception as e:
            print(f"  [警告] 讀取 best_factors.json 失敗: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # 輸出組裝
    # ════════════════════════════════════════════════════════════════════════════
    output_lines = []

    # ── 1. 持倉追蹤表 ────────────────────────────────────────────────────────
    output_lines.append("\n" + "=" * 80)
    output_lines.append(f"  [持倉追蹤] 未來三天多空綜合分數 (預測基準日: {latest_date.date()})")
    output_lines.append("=" * 80)
    output_lines.append(
        f"{'排名':<3} | {'股票 (代號+名稱)':<12} | {'收盤價':<6} | "
        f"{'D1多空':<8} | {'D2多空':<8} | {'D3多空':<8} | {'買進成本':<8} | {'浮動損益':<8} | {'趨勢分析'}"
    )
    output_lines.append("-" * 95)

    for i, row in enumerate(results_watchlist.itertuples(), start=1):
        sid        = row.stock_id
        close_p    = row.close
        d1, d2, d3 = row.Day1_net, row.Day2_net, row.Day3_net
        cname      = stock_names.get(sid, "")
        disp_name  = f"{sid} {cname}"

        d1_s = f"{d1:+.1f}%"
        d2_s = f"{d2:+.1f}%"
        d3_s = f"{d3:+.1f}%"

        # 浮動損益（若 Stocks.txt 有填買進成本才顯示）
        cost = watchlist.get(sid)
        if cost and cost > 0:
            actual_cost = cost * (1 + FEE_RATE)
            pnl_pct     = (close_p - actual_cost) / actual_cost * 100
            cost_s      = f"{cost:.2f}"
            pnl_s       = f"{pnl_pct:+.1f}%"
        else:
            cost_s = "  --  "
            pnl_s  = "  --  "

        # 趨勢判定
        if   d1 > 12 and d2 > 8:   trend = "強勢多頭 (發動中)"
        elif d1 > 5  and d2 > 0:   trend = "偏多 (醞釀中)"
        elif d1 < -15 and d2 < -15: trend = "極度弱勢 (空頭)"
        elif d1 < -5  and d2 < -5:  trend = "偏空 (轉弱)"
        elif d1 > 8  and d3 < 0:   trend = "短多長空 (留意賣點)"
        else:                        trend = "震盪整理"

        output_lines.append(
            f" {i:<3} | {disp_name:<16} | {close_p:>6.2f} | "
            f"{d1_s:>8} | {d2_s:>8} | {d3_s:>8} | "
            f"{cost_s:>8} | {pnl_s:>8} |  {trend}"
        )

    output_lines.append("=" * 80)

    # ── 2. 策略行動建議 (對齊 trading_sim.py 邏輯) ───────────────────────────
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append(
        f"  [策略行動建議]  "
        f"買進條件: Day1 >= {BUY_THRESHOLD:.0f}%  |  "
        f"賣出條件: Day3 < {SELL_THRESHOLD:.0f}%  |  "
        f"停損線: {STOP_LOSS_PCT:.0f}%"
    )
    output_lines.append("=" * 80)

    # ── 2a. 建議賣出（持倉股）─────────────────────────────────────────────
    output_lines.append("  🔴 建議賣出 / 停損 (持倉股觸發條件):")
    sell_any = False

    for _, row in results_watchlist.iterrows():
        sid     = str(row["stock_id"])
        cname   = stock_names.get(sid, "")
        d3      = row["Day3_net"]
        close_p = row["close"]
        cost    = watchlist.get(sid)

        reasons = []

        # 條件 1：Day3 轉弱
        if d3 < SELL_THRESHOLD:
            reasons.append(f"Day3預測轉弱 ({d3:+.1f}%)")

        # 條件 2：固定停損（有填成本才判斷）
        if cost and cost > 0:
            actual_cost = cost * (1 + FEE_RATE)
            pnl_pct     = (close_p - actual_cost) / actual_cost * 100
            if pnl_pct <= STOP_LOSS_PCT:
                reasons.append(f"觸發 {STOP_LOSS_PCT:.0f}% 停損 (現虧 {pnl_pct:.1f}%)")

        if reasons:
            sell_any = True
            reason_str = "、".join(reasons)
            output_lines.append(f"    賣出: {sid:<6} {cname:<6}  原因: {reason_str}")

    if not sell_any:
        output_lines.append("    ✅ 持倉中目前無觸發賣出訊號")

    output_lines.append("")

    # ── 2b. 建議買進（全市場新倉）─────────────────────────────────────────
    output_lines.append(f"  🟢 建議買進 (全市場 Day1 >= {BUY_THRESHOLD:.0f}%，前 {MAX_BUY_SHOW} 檔，排除已持倉):")
    buy_candidates = (
        results_market[
            (results_market["Day1_net"] >= BUY_THRESHOLD) &
            (~results_market["stock_id"].isin(watchlist.keys()))   # 排除已持倉
        ]
        .sort_values("Day1_net", ascending=False)
        .head(MAX_BUY_SHOW)
    )

    if buy_candidates.empty:
        output_lines.append("    ⚪ 目前全市場無符合買進條件之標的（大盤可能偏弱，建議保留現金）")
    else:
        for _, row in buy_candidates.iterrows():
            sid   = str(row["stock_id"])
            cname = stock_names.get(sid, "")
            d1    = row["Day1_net"]
            d3    = row["Day3_net"]
            close_p = row["close"]
            output_lines.append(
                f"    買進: {sid:<6} {cname:<6}  "
                f"收盤: {close_p:>7.2f}  "
                f"D1分數: {d1:>+5.1f}%  "
                f"D3分數: {d3:>+5.1f}%"
            )

    output_lines.append("")

    # ── 2c. 注意：持倉中符合買進條件（加碼提示）─────────────────────────────
    add_position = results_watchlist[results_watchlist["Day1_net"] >= BUY_THRESHOLD]
    if not add_position.empty:
        output_lines.append(f"  🔵 持倉中同時符合買進條件 (可考慮加碼):")
        for _, row in add_position.iterrows():
            sid   = str(row["stock_id"])
            cname = stock_names.get(sid, "")
            d1    = row["Day1_net"]
            output_lines.append(f"    加碼: {sid:<6} {cname:<6}  D1分數: {d1:>+5.1f}%")
        output_lines.append("")

    # ── 3. 全市場 Top10 / Bottom10 ───────────────────────────────────────────
    output_lines.append("=" * 80)
    output_lines.append("  [全市場掃描] Top-10 強勢 & Bottom-10 弱勢 (以 Day1 排序)")
    output_lines.append("=" * 80)
    output_lines.append(f"  🚀 強勢 Top-10:")
    for i, row in enumerate(results_top10.itertuples(), 1):
        sid   = str(row.stock_id)
        cname = stock_names.get(sid, "")
        output_lines.append(
            f"    {i:>2}. {sid:<6} {cname:<8}  "
            f"收盤: {row.close:>7.2f}  "
            f"D1: {row.Day1_net:>+5.1f}%  D3: {row.Day3_net:>+5.1f}%"
        )

    output_lines.append("")
    output_lines.append(f"  ⚠️  弱勢 Bottom-10 (建議避開):")
    for i, row in enumerate(results_bottom10.itertuples(), 1):
        sid   = str(row.stock_id)
        cname = stock_names.get(sid, "")
        output_lines.append(
            f"    {i:>2}. {sid:<6} {cname:<8}  "
            f"收盤: {row.close:>7.2f}  "
            f"D1: {row.Day1_net:>+5.1f}%  D3: {row.Day3_net:>+5.1f}%"
        )

    output_lines.append("=" * 80)
    output_lines.append(factors_info)
    output_lines.append("=" * 80)
    output_lines.append("  [聲明] 模型預測結果僅供量化研究參考，不構成實際投資建議。")

    final_output = "\n".join(output_lines)
    print(final_output)

    # ── 儲存預測結果 ────────────────────────────────────────────────────────
    pred_dir = os.path.join(BASE_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_file = os.path.join(pred_dir, f"prediction_{date_str}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_output)
    print(f"\n  [儲存] 本次預測結果已存檔至: {out_file}")


if __name__ == "__main__":
    main()
