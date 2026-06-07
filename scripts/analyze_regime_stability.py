# -*- coding: utf-8 -*-
"""
analyze_regime_stability.py — 訊號與市況穩定性多層級診斷分析器 (Quant Researcher Edition)
==================================================================================
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr

# 統一設定路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "regime_stability_report.txt")

def calculate_psi(expected, actual, num_bins=10):
    """計算 Population Stability Index (PSI)"""
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # 依 Expected (樣本內) 的分位點切分區間
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    # 處理重複的邊界點
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
        
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    expected_cats = pd.cut(expected, bins=bins, labels=False, include_lowest=True)
    actual_cats = pd.cut(actual, bins=bins, labels=False, include_lowest=True)
    
    expected_counts = pd.Series(expected_cats).value_counts(sort=False)
    actual_counts = pd.Series(actual_cats).value_counts(sort=False)
    
    # 補零
    all_indices = range(len(bins) - 1)
    expected_counts = expected_counts.reindex(all_indices, fill_value=0)
    actual_counts = actual_counts.reindex(all_indices, fill_value=0)
    
    # 計算佔比
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # 避開 0 值以計算對數
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # PSI 公式
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value

def main():
    print("=" * 80)
    print("  量化訊號與市況穩定性診斷分析 (scripts/analyze_regime_stability.py)")
    print("=" * 80)
    
    if not os.path.exists(DATA_PATH):
        print(f"[錯誤] 找不到特徵資料檔: {DATA_PATH}")
        sys.exit(1)
        
    print("載入歷史特徵矩陣與對齊產業...")
    df = pd.read_parquet(DATA_PATH)
    from scripts.utils import filter_stocks_by_train_industries
    df = filter_stocks_by_train_industries(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 載入模型特徵欄位與模型
    feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
        
    model_path = os.path.join(MODEL_DIR, "lgbm_model_1.txt")
    if not os.path.exists(model_path):
        print(f"[錯誤] 找不到 Day 1 模型: {model_path}")
        sys.exit(1)
    model = lgb.Booster(model_file=model_path)
    
    print("生成 Day 1 預測 Net Score...")
    X = df.reindex(columns=feature_cols).astype(np.float32)
    preds = model.predict(X)
    df["net_score"] = (preds[:, 2] - preds[:, 0]) * 100
    
    split_date = pd.to_datetime("2025-08-01")
    df_is = df[df["date"] <= split_date].copy()
    df_oos = df[df["date"] > split_date].copy()
    
    # ── 1. 劃分市場 Regime ──
    # 使用大盤日均回報 (market_mean_pct) 計算 20 日滾動平均做為趨勢指標
    # 上漲家數比 (market_breadth_pct) 做為寬度指標
    df_daily_mkt = df.groupby("date")[["market_mean_pct", "market_breadth_pct"]].first().reset_index()
    df_daily_mkt["market_trend_20d"] = df_daily_mkt["market_mean_pct"].rolling(window=20, min_periods=5).mean()
    df_daily_mkt["market_breadth_20d"] = df_daily_mkt["market_breadth_pct"].rolling(window=20, min_periods=5).mean()
    
    df = df.merge(df_daily_mkt[["date", "market_trend_20d", "market_breadth_20d"]], on="date", how="left")
    
    # 定義 Regime:
    # - Bull: 20日均回報 > 0 且 20日上漲家數比 > 0.50
    # - Bear: 20日均回報 < 0 且 20日上漲家數比 < 0.50
    # - Sideways: 盤整震盪
    conditions = [
        (df["market_trend_20d"] > 0) & (df["market_breadth_20d"] > 0.50),
        (df["market_trend_20d"] < 0) & (df["market_breadth_20d"] < 0.50)
    ]
    df["regime"] = np.select(conditions, ["Bull", "Bear"], default="Sideways")
    
    # ── 2. 按日期計算每日指標 ──
    daily_stats = []
    
    for date, g in df.groupby("date"):
        if len(g) < 10:
            continue
        regime = g["regime"].iloc[0]
        scores = g["net_score"].values
        returns = g["next_ret_1"].values
        mkt_ret = g["market_mean_pct"].iloc[0]
        
        # Spearman RankIC
        rank_ic = 0.0
        if np.std(scores) > 0 and np.std(returns) > 0:
            rank_ic, _ = spearmanr(scores, returns)
            
        # Long-Short Spread (Top 5% vs Bottom 5%)
        g_sorted = g.sort_values("net_score", ascending=False)
        n_stocks = len(g)
        k5 = max(1, int(n_stocks * 0.05))
        top_k = g_sorted.head(k5)
        bot_k = g_sorted.tail(k5)
        
        top_ret = top_k["next_ret_1"].mean()
        bot_ret = bot_k["next_ret_1"].mean()
        ls_spread = top_ret - bot_ret
        
        # Top Buckets (1%, 5%, 10%)
        k1 = max(1, int(n_stocks * 0.01))
        k10 = max(1, int(n_stocks * 0.10))
        
        t1_ret = g_sorted.head(k1)["next_ret_1"].mean()
        t5_ret = top_ret
        t10_ret = g_sorted.head(k10)["next_ret_1"].mean()
        
        daily_stats.append({
            "date": date,
            "regime": regime,
            "rank_ic": rank_ic,
            "ls_spread": ls_spread,
            "market_ret": mkt_ret,
            "top1_ret": t1_ret,
            "top5_ret": t5_ret,
            "top10_ret": t10_ret,
        })
        
    df_daily = pd.DataFrame(daily_stats)
    df_daily["is_oos"] = np.where(df_daily["date"] <= split_date, "IS", "OOS")
    
    # ── 3. 計算 60 日滾動 RankIC 趨勢 ──
    df_daily["rolling_rank_ic_60d"] = df_daily["rank_ic"].rolling(60).mean()
    df_daily["rolling_spread_60d"] = df_daily["ls_spread"].rolling(60).mean()
    
    # ── 4. 市況歸因分析 (Regime Attribution) ──
    regime_summary = df_daily.groupby(["is_oos", "regime"]).agg(
        Days=("date", "count"),
        Mean_RankIC=("rank_ic", "mean"),
        Mean_LS_Spread=("ls_spread", lambda x: np.mean(x) * 100),
        Spread_t_stat=("ls_spread", lambda x: np.mean(x) / (np.std(x) / np.sqrt(len(x))) if np.std(x) > 0 else 0.0)
    ).reset_index()
    
    # 計算整體 IS 與 OOS (All)
    overall_summary = df_daily.groupby("is_oos").agg(
        Days=("date", "count"),
        Mean_RankIC=("rank_ic", "mean"),
        Mean_LS_Spread=("ls_spread", lambda x: np.mean(x) * 100),
        Spread_t_stat=("ls_spread", lambda x: np.mean(x) / (np.std(x) / np.sqrt(len(x))) if np.std(x) > 0 else 0.0)
    ).reset_index()
    overall_summary["regime"] = "All"
    
    regime_summary = pd.concat([regime_summary, overall_summary], ignore_index=True)
    regime_summary = regime_summary.sort_values(["is_oos", "regime"]).reset_index(drop=True)
    
    # ── 5. 精細尾部排序 (Top 1%, 5%, 10% vs Market Alpha) ──
    tail_summary = []
    for grp in ["IS", "OOS"]:
        sub = df_daily[df_daily["is_oos"] == grp]
        n_days = len(sub)
        tail_summary.append({
            "Group": grp,
            "Market_Ret": sub["market_ret"].mean() * 100,
            "Top1_Ret": sub["top1_ret"].mean() * 100,
            "Top1_Alpha": (sub["top1_ret"] - sub["market_ret"]).mean() * 100,
            "Top5_Ret": sub["top5_ret"].mean() * 100,
            "Top5_Alpha": (sub["top5_ret"] - sub["market_ret"]).mean() * 100,
            "Top10_Ret": sub["top10_ret"].mean() * 100,
            "Top10_Alpha": (sub["top10_ret"] - sub["market_ret"]).mean() * 100,
        })
    df_tail = pd.DataFrame(tail_summary)
    
    # ── 6. PSI 特徵分布漂移 ──
    # 僅對存在於 parquet 中的特徵進行分析
    valid_features = [f for f in feature_cols if f in df.columns]
    print(f"共有 {len(valid_features)}/{len(feature_cols)} 個特徵實際存在於 parquet 中，進行漂移分析...")
    
    psi_stats = []
    for f in valid_features:
        is_feat = df_is[f]
        oos_feat = df_oos[f]
        psi_val = calculate_psi(is_feat, oos_feat, num_bins=10)
        psi_stats.append({"feature": f, "psi": psi_val})
    df_psi = pd.DataFrame(psi_stats).sort_values("psi", ascending=False).reset_index(drop=True)
    
    # ── 7. 因子 Correlation Drift (個別特徵之 RankIC 漂移) ──
    feat_rank_ics = []
    for f in valid_features:
        is_ics = []
        oos_ics = []
        # 僅在特徵有變動時計算
        if df_is[f].std() > 0 and df_oos[f].std() > 0:
            # 計算特徵本身與 next_ret_1 的每日 RankIC
            for date, g in df.groupby("date"):
                if len(g) < 10 or g[f].std() == 0 or g["next_ret_1"].std() == 0:
                    continue
                ric, _ = spearmanr(g[f], g["next_ret_1"])
                if pd.notna(ric):
                    if date <= split_date:
                        is_ics.append(ric)
                    else:
                        oos_ics.append(ric)
            
            mean_is_ric = np.mean(is_ics) if is_ics else 0.0
            mean_oos_ric = np.mean(oos_ics) if oos_ics else 0.0
            drift = mean_oos_ric - mean_is_ric
            feat_rank_ics.append({
                "feature": f,
                "IS_RankIC": mean_is_ric,
                "OOS_RankIC": mean_oos_ric,
                "IC_Drift": drift,
                "Abs_IC_Drift": abs(drift)
            })
    df_feat_drift = pd.DataFrame(feat_rank_ics).sort_values("Abs_IC_Drift", ascending=False).reset_index(drop=True)
    
    # ── 7. 動能疊加測試 (Momentum Overlay Test) ──
    # 目的：驗證模型在昨日大漲股票（強動能 > 2%）與一般股票（弱動能 <= 2%）的選股預測力 (RankIC) 是否有結構性差異
    momentum_stats = []
    
    for date, g in df_oos.groupby("date"):
        if len(g) < 10 or "ret1" not in g.columns:
            continue
            
        g_strong = g[g["ret1"] > 0.02]
        g_weak = g[g["ret1"] <= 0.02]
        
        strong_ic = np.nan
        weak_ic = np.nan
        
        # 計算強動能組 RankIC
        if len(g_strong) >= 5 and g_strong["net_score"].std() > 0 and g_strong["next_ret_1"].std() > 0:
            strong_ic, _ = spearmanr(g_strong["net_score"], g_strong["next_ret_1"])
            
        # 計算弱動能組 RankIC
        if len(g_weak) >= 5 and g_weak["net_score"].std() > 0 and g_weak["next_ret_1"].std() > 0:
            weak_ic, _ = spearmanr(g_weak["net_score"], g_weak["next_ret_1"])
        strong_spread = np.nan
        if len(g_strong) >= 5:
            g_strong_sorted = g_strong.sort_values("net_score", ascending=False)
            k_s = max(1, int(len(g_strong) * 0.20))
            strong_spread = g_strong_sorted.head(k_s)["next_ret_1"].mean() - g_strong_sorted.tail(k_s)["next_ret_1"].mean()
        weak_spread = np.nan
        if len(g_weak) >= 5:
            g_weak_sorted = g_weak.sort_values("net_score", ascending=False)
            k_w = max(1, int(len(g_weak) * 0.20))
            weak_spread = g_weak_sorted.head(k_w)["next_ret_1"].mean() - g_weak_sorted.tail(k_w)["next_ret_1"].mean()
        momentum_stats.append({
            "date": date, "strong_ic": strong_ic, "weak_ic": weak_ic,
            "strong_spread": strong_spread, "weak_spread": weak_spread,
            "strong_count": len(g_strong), "weak_count": len(g_weak)
        })
    df_mom = pd.DataFrame(momentum_stats)
    
    # ── 8. SHAP & 特徵重要性漂移分析 (SHAP & Importance Drift) ──
    print("計算特徵重要性與 SHAP 漂移 (Step 4 & 4.5)...")
    X_is = df_is.reindex(columns=feature_cols).astype(np.float32)
    X_oos = df_oos.reindex(columns=feature_cols).astype(np.float32)
    num_features = len(feature_cols)
    contribs_is = model.predict(X_is, pred_contrib=True)
    contribs_oos = model.predict(X_oos, pred_contrib=True)
    class2_start = 2 * (num_features + 1)
    shap_is = contribs_is[:, class2_start : class2_start + num_features]
    shap_oos = contribs_oos[:, class2_start : class2_start + num_features]
    shap_abs_is = np.mean(np.abs(shap_is), axis=0)
    shap_abs_oos = np.mean(np.abs(shap_oos), axis=0)
    shap_mean_is = np.mean(shap_is, axis=0)
    shap_mean_oos = np.mean(shap_oos, axis=0)
    gain_imp = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")
    sum_gain = sum(gain_imp) if sum(gain_imp) > 0 else 1.0
    sum_split = sum(split_imp) if sum(split_imp) > 0 else 1.0
    sum_shap_is = sum(shap_abs_is) if sum(shap_abs_is) > 0 else 1.0
    gain_imp_pct = (gain_imp / sum_gain) * 100
    split_imp_pct = (split_imp / sum_split) * 100
    shap_imp_pct = (shap_abs_is / sum_shap_is) * 100
    shap_drift_list = []
    for j, feat in enumerate(feature_cols):
        shap_drift_list.append({
            "feature": feat, "gain_pct": gain_imp_pct[j], "split_pct": split_imp_pct[j],
            "shap_is_pct": shap_imp_pct[j], "shap_is_abs": shap_abs_is[j], "shap_oos_abs": shap_abs_oos[j],
            "shap_is_mean": shap_mean_is[j], "shap_oos_mean": shap_mean_oos[j], "shap_mean_drift": shap_mean_oos[j] - shap_mean_is[j]
        })
    df_shap_drift = pd.DataFrame(shap_drift_list).sort_values("shap_oos_abs", ascending=False).reset_index(drop=True)
    
    # ── 9. 生成診斷文字報告 ──
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("======================================================================\n")
        f_out.write("          量化訊號與市況穩定性診斷報告 (Regime Stability Report)\n")
        f_out.write("======================================================================\n\n")
        
        f_out.write(f"劃分日期: {split_date.date()}\n")
        f_out.write(f"樣本內 (IS) 交易天數: {len(df_daily[df_daily['is_oos'] == 'IS'])} 天\n")
        f_out.write(f"樣本外 (OOS) 交易天數: {len(df_daily[df_daily['is_oos'] == 'OOS'])} 天\n\n")
        
        f_out.write("1. 滾動 RankIC 趨勢 (60日滾動最近 10 天變化)\n")
        f_out.write("-" * 70 + "\n")
        rolling_part = df_daily.dropna(subset=["rolling_rank_ic_60d"]).tail(10)
        for _, row in rolling_part.iterrows():
            f_out.write(f"日期: {row['date'].date()} | 60日滾動 RankIC: {row['rolling_rank_ic_60d']:+.4f} | 60日滾動多空價差: {row['rolling_spread_60d']*100:+.3f}%\n")
        f_out.write("\n")
        
        f_out.write("2. 市況歸因分析 (Regime Attribution)\n")
        f_out.write("-" * 70 + "\n")
        f_out.write(f"{'區間':<6} | {'市況':<8} | {'交易天數':<8} | {'平均 RankIC':<12} | {'多空日價差':<10} | {'Spread t-stat':<10}\n")
        f_out.write("-" * 70 + "\n")
        for _, row in regime_summary.iterrows():
            f_out.write(f"{row['is_oos']:<8} | {row['regime']:<10} | {row['Days']:<10} | {row['Mean_RankIC']:+12.4f} | {row['Mean_LS_Spread']:+9.4f}% | {row['Spread_t_stat']:10.3f}\n")
        f_out.write("\n")
        
        f_out.write("3. 尾部排序能力 (Top 1%, 5%, 10% vs Market Alpha)\n")
        f_out.write("-" * 70 + "\n")
        for _, row in df_tail.iterrows():
            f_out.write(f"[{row['Group']}] 市場平均: {row['Market_Ret']:+.3f}%\n")
            f_out.write(f"    - Top 1%  平均報酬: {row['Top1_Ret']:+.3f}% | 超額 Alpha: {row['Top1_Alpha']:+.3f}%\n")
            f_out.write(f"    - Top 5%  平均報酬: {row['Top5_Ret']:+.3f}% | 超額 Alpha: {row['Top5_Alpha']:+.3f}%\n")
            f_out.write(f"    - Top 10% 平均報酬: {row['Top10_Ret']:+.3f}% | 超額 Alpha: {row['Top10_Alpha']:+.3f}%\n")
        f_out.write("\n")
        
        f_out.write("4. 因子分布穩定性 (PSI Drift Top 10)\n")
        f_out.write("-" * 70 + "\n")
        f_out.write("PSI < 0.1 無漂移 | 0.1 <= PSI < 0.25 中度漂移 | PSI >= 0.25 嚴重漂移\n")
        f_out.write("-" * 70 + "\n")
        for idx, row in df_psi.head(10).iterrows():
            status = "嚴重漂移" if row['psi'] >= 0.25 else ("中度漂移" if row['psi'] >= 0.1 else "無漂移")
            f_out.write(f"[{idx+1:>2}] 特徵: {row['feature']:<30} | PSI: {row['psi']:.4f} ({status})\n")
        f_out.write("\n")
        
        f_out.write("5. 因子相關性漂移 (IC Drift Top 10)\n")
        f_out.write("-" * 70 + "\n")
        f_out.write("這顯示哪些因子在樣本外與未來報酬率的關係發生了劇烈變化\n")
        f_out.write("-" * 70 + "\n")
        for idx, row in df_feat_drift.head(10).iterrows():
            f_out.write(f"[{idx+1:>2}] 特徵: {row['feature']:<30} | IS RankIC: {row['IS_RankIC']:+.4f} | OOS RankIC: {row['OOS_RankIC']:+.4f} | Drift: {row['IC_Drift']:+.4f}\n")
        f_out.write("\n")
        
        f_out.write("6. 動能疊加測試 (Momentum Overlay Test)\n")
        f_out.write("-" * 70 + "\n")
        f_out.write("目的: 驗證昨日大漲股票(強動能 > 2%)與一般股票(弱動能 <= 2%)在樣本外的預測力差異\n")
        f_out.write("-" * 70 + "\n")
        
        mean_weak_ic = df_mom["weak_ic"].mean() if not df_mom.empty else np.nan
        mean_strong_ic = df_mom["strong_ic"].mean() if not df_mom.empty else np.nan
        mean_weak_spread = df_mom["weak_spread"].mean() if not df_mom.empty else np.nan
        mean_strong_spread = df_mom["strong_spread"].mean() if not df_mom.empty else np.nan
        mean_weak_count = df_mom["weak_count"].mean() if not df_mom.empty else 0
        mean_strong_count = df_mom["strong_count"].mean() if not df_mom.empty else 0
        
        f_out.write(f"弱動能組 (Yesterday Ret <= 2%) 平均 RankIC: {mean_weak_ic:+.4f} | 平均多空價差: {mean_weak_spread*100:+.3f}% (日均個股數: {mean_weak_count:.1f})\n")
        f_out.write(f"強動能組 (Yesterday Ret > 2%)  平均 RankIC: {mean_strong_ic:+.4f} | 平均多空價差: {mean_strong_spread*100:+.3f}% (日均個股數: {mean_strong_count:.1f})\n")
        f_out.write("-" * 70 + "\n")
        f_out.write("診斷指引:\n")
        f_out.write("- 若 [弱動能組 IC] 顯著高於 [強動能組 IC] (例如 0.04 vs -0.01)，代表模型在強趨勢股上失效(Mean Reversion與動能相衝)\n")
        f_out.write("- 若兩組 IC 接近，代表 Alpha 衰退並非由 Momentum 風格切換單一因子所主導\n\n")
        
        # 新增第 7 部分：SHAP 與特徵重要性漂移對比
        f_out.write("7. SHAP 與特徵重要性漂移 (SHAP & Importance Drift Top 15)\n")
        f_out.write("-" * 80 + "\n")
        f_out.write(f"{'特徵名稱':<28} | {'Gain%':<6} | {'Split%':<6} | {'IS_SHAP_A':<9} | {'OOS_SHAP_A':<10} | {'IS_SHAP_M':<9} | {'OOS_SHAP_M':<10} | {'Drift(Mean)':<11}\n")
        f_out.write("-" * 80 + "\n")
        for idx, row in df_shap_drift.head(15).iterrows():
            f_out.write(f"{row['feature']:<28} | {row['gain_pct']:5.1f}% | {row['split_pct']:5.1f}% | {row['shap_is_abs']:9.4f} | {row['shap_oos_abs']:10.4f} | {row['shap_is_mean']:+9.4f} | {row['shap_oos_mean']:+10.4f} | {row['shap_mean_drift']:+11.4f}\n")
        f_out.write("-" * 80 + "\n")
        f_out.write("診斷指引:\n")
        f_out.write("- 比較 IS_SHAP_M 與 OOS_SHAP_M (SHAP Direction)，若正負號反轉，即為市場機制反轉之鐵證\n")
        f_out.write("- 比較 Gain% (擬合) 與 OOS_SHAP_A (實質)，若 Gain% 高但 OOS SHAP 極低，說明特徵過度擬合且失效\n")
            
    # 同步印出報告至控制台
    with open(REPORT_PATH, "r", encoding="utf-8") as f_in:
        print(f_in.read())
        
    print(f"\n[成功] 診斷分析完成，完整報告已存檔至: {REPORT_PATH}\n")

if __name__ == "__main__":
    main()
