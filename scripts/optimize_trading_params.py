# -*- coding: utf-8 -*-
"""
optimize_trading_params.py — 交易策略與避險風控參數貝葉斯自動最佳化 (Optuna TPE)
==================================================================================
用法:
  # 預設執行 (從 config.py 讀取參數):
  python scripts/optimize_trading_params.py

  # 自訂參數執行:
  python scripts/optimize_trading_params.py -t 150 -s 2021-01-01 -e 2025-08-01 -c 2000000
  或使用長參數:
  python scripts/optimize_trading_params.py --trials 150 --start 2021-01-01 --end 2025-08-01 --capital 2000000
"""

import os
import sys
import json
import datetime
import argparse
import contextlib
import io
import optuna

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import signal
def _force_exit_sigint(signum, frame):
    print("\n[使用者中斷] 偵測到 Ctrl+C，強制終止所有執行緒並結束程式...", flush=True)
    import os
    os._exit(1)

signal.signal(signal.SIGINT, _force_exit_sigint)

# 隱藏 Optuna 的系統詳細 log，由我們自己控制輸出
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 統一將 BASE_DIR 設為專案根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 載入中央控制面板與模擬交易器
try:
    from config import (
        BACKTEST_DATE, MAX_POSITIONS, OPTIMIZATION_TRIALS, EARLY_STOPPING_ROUNDS
    )
    from trading_sim import run_simulation
except ImportError as e:
    print(f"[錯誤] 載入核心模組失敗: {e}")
    sys.exit(1)

# 計算預設的訓練期（Optuna 最佳化區間）
# 為了避免前視偏差與對樣本外數據（Test Set）進行過度擬合，
# 最佳化區間預設截止於 BACKTEST_DATE（樣本外評估的分界點）
dt_end = None
if BACKTEST_DATE is not None:
    try:
        dt_end = datetime.datetime.strptime(str(BACKTEST_DATE), "%Y%m%d")
    except ValueError:
        try:
            dt_end = datetime.datetime.strptime(str(BACKTEST_DATE), "%Y-%m-%d")
        except ValueError:
            pass

if dt_end is None:
    dt_end = datetime.datetime(2025, 8, 1)

# 預設結束時間：BACKTEST_DATE
default_end_date = dt_end.strftime("%Y-%m-%d")
# 預設起始時間：結束日期的 2.5 年前，以保證有足夠長的歷史數據進行多牛熊周期驗證
default_start_date = (dt_end - datetime.timedelta(days=365 * 2.5)).strftime("%Y-%m-%d")

# 最佳化結果存檔路徑
RESULT_PATH = os.path.join(BASE_DIR, "configs", "best_trading_params.json")


def run_simulation_scoring(start_date, end_date, trial_params, capital, max_pos):
    import io
    import contextlib
    from trading_sim import run_simulation
    import numpy as np
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            total_return, max_dd, history = run_simulation(
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
                max_positions=max_pos,
                mkt_panic_ma5=trial_params["panic_ma5"],
                mkt_panic_breadth=trial_params["panic_breadth"],
                buy_threshold=trial_params["buy_threshold"],
                stop_loss_pct=trial_params["stop_loss"],
                ts_activation_pct=trial_params["ts_activation"],
                ts_pullback_pct=trial_params["ts_pullback"]
            )
        except Exception:
            return -999.0, {}, 0.0, 0.0
            
    weighted_sum = 0.0
    total_w = 0.0
    regime_details = {}
    
    for r in ["Bull", "Bear", "Sideways"]:
        sub_hist = [h for h in history if h.get('regime', 'Sideways') == r]
        n_days_r = len(sub_hist)
        if n_days_r == 0:
            continue
            
        mean_alpha = np.mean([h['portfolio_alpha'] for h in sub_hist])
        mean_spread = np.mean([h['portfolio_spread'] for h in sub_hist])
        
        regime_equity = [h['equity'] for h in sub_hist]
        regime_initial = regime_equity[0]
        regime_final = regime_equity[-1]
        regime_ret = (regime_final / regime_initial - 1.0) if regime_initial > 0 else 0.0
        
        regime_max_dd = 0.0
        peak = regime_equity[0]
        for e in regime_equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0.0
            if dd > regime_max_dd:
                regime_max_dd = dd
                
        regime_calmar = (regime_ret * 100) / ((regime_max_dd * 100) + 2.0)
        
        # 套用 0.6 Alpha + 0.2 Spread + 0.2 Calmar 權重公式
        sub_score = 0.6 * (mean_alpha * 100) + 0.2 * (mean_spread * 100) + 0.2 * regime_calmar
        
        # 平方根天數權重歸一化
        w_r = min(1.0, np.sqrt(n_days_r / 60.0))
        
        weighted_sum += w_r * sub_score
        total_w += w_r
        
        regime_details[r] = {
            "days": n_days_r,
            "weight": w_r,
            "alpha_pct": mean_alpha * 100,
            "spread_pct": mean_spread * 100,
            "return_pct": regime_ret * 100,
            "mdd_pct": regime_max_dd * 100,
            "score": sub_score
        }
        
    combined_score = (weighted_sum / total_w) if total_w > 0 else -999.0
    return combined_score, regime_details, total_return, max_dd


def main():
    parser = argparse.ArgumentParser(description="交易與風控參數貝葉斯最佳化工具 (Optuna)")
    parser.add_argument("-s", "--start", type=str, default=default_start_date, help="回測起始日期 (YYYY-MM-DD)")
    parser.add_argument("-e", "--end", type=str, default=default_end_date, help="回測結束日期 (YYYY-MM-DD)")
    parser.add_argument("-c", "--capital", type=int, default=1000000, help="回測初始資金")
    parser.add_argument("-m", "--max_pos", type=int, default=MAX_POSITIONS, help="最大持持股上限檔數")
    parser.add_argument("-t", "--trials", type=int, default=OPTIMIZATION_TRIALS, help="最佳化搜尋輪數")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="並行搜尋執行緒數")
    parser.add_argument("-wf", "--walk_forward", action="store_true", help="是否啟用 Walk-Forward 參數穩定性檢驗")
    
    args = parser.parse_args()
    
    import numpy as np
    
    print("=" * 70)
    print("  交易與避險風控參數貝葉斯自動調參 (Optuna TPE)")
    print("=" * 70)
    print(f"  調參資料區間 : {args.start} 至 {args.end}")
    print(f"  模擬資金規模 : {args.capital:,} | 最大持股 slots: {args.max_pos}")
    print(f"  預期搜尋輪數 : {args.trials} 輪")
    print(f"  結果輸出存檔 : {RESULT_PATH}")
    print(f"  Walk-Forward : {'是' if args.walk_forward else '否'}")
    print("=" * 70)
    
    feature_cols_path = os.path.join(BASE_DIR, "models", "feature_cols.json")
    if not os.path.exists(feature_cols_path):
        print(f"[錯誤] 找不到 feature_cols.json，請先執行 auto_pipeline.py 訓練模型。")
        sys.exit(1)
        
    if args.walk_forward:
        # ── Walk-Forward 模式 ──
        start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(args.end, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        
        # 每個子窗口長度設為總長度的 65%
        window_days = int(total_days * 0.65)
        step_days = int((total_days - window_days) / 3) if total_days > window_days else 10
        
        WINDOWS = []
        for i in range(4):
            w_start = start_dt + datetime.timedelta(days=i * step_days)
            w_end = w_start + datetime.timedelta(days=window_days)
            if i == 3 or w_end > end_dt:
                w_end = end_dt
            WINDOWS.append((w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
            
        all_best_params = []
        wf_trials = max(20, args.trials // 4)
        
        for idx, (ws, we) in enumerate(WINDOWS):
            print(f"\n[WF 窗口 {idx+1}/4] 優化區間: {ws} 至 {we} (搜尋 {wf_trials} 輪)...")
            
            def wf_objective(trial):
                trial_params = {
                    "buy_threshold": trial.suggest_float("buy_threshold", 5.0, 25.0, step=0.5),
                    "stop_loss": trial.suggest_float("stop_loss", -15.0, -3.0, step=0.5),
                    "panic_ma5": trial.suggest_float("panic_ma5", -0.025, 0.00, step=0.001),
                    "panic_breadth": trial.suggest_float("panic_breadth", 0.15, 0.45, step=0.01),
                    "ts_activation": trial.suggest_float("ts_activation", 5.0, 25.0, step=0.5),
                    "ts_pullback": trial.suggest_float("ts_pullback", -12.0, -1.0, step=0.5)
                }
                score, _, _, _ = run_simulation_scoring(ws, we, trial_params, args.capital, args.max_pos)
                return score
                
            study = optuna.create_study(direction="maximize")
            study.optimize(wf_objective, n_trials=wf_trials, n_jobs=args.jobs)
            print(f"  [窗口 {idx+1} 完成] 最佳得分: {study.best_value:.4f} | 參數: {study.best_params}")
            all_best_params.append(study.best_params)
            
        # 統計與部署
        param_names = ["buy_threshold", "stop_loss", "panic_ma5", "panic_breadth", "ts_activation", "ts_pullback"]
        wf_results = {}
        median_params = {}
        
        print("\n" + "=" * 75)
        print("  [Walk-Forward 參數穩定性統計報表 (Step 6 / Phase 3)]")
        print("=" * 75)
        print(f"{'參數名稱':<15} | {'中位數(Median)':<15} | {'四分位距(IQR)':<15} | {'變異係數(CV)':<12} | {'穩定度':<6}")
        print("-" * 75)
        
        for p in param_names:
            vals = np.array([bp[p] for bp in all_best_params])
            med_val = np.median(vals)
            q75, q25 = np.percentile(vals, [75, 25])
            iqr_val = q75 - q25
            std_val = np.std(vals)
            mean_val = np.mean(vals)
            cv_val = std_val / abs(mean_val) if abs(mean_val) > 0 else 0.0
            stability = "穩定" if cv_val < 0.15 else "需注意" if cv_val < 0.30 else "不穩定"
            
            if p in ["panic_ma5", "panic_breadth"]:
                print(f"{p:<15} | {med_val*100:13.2f}% | {iqr_val*100:13.2f}% | {cv_val:12.3f} | {stability}")
            else:
                print(f"{p:<15} | {med_val:14.2f}  | {iqr_val:14.2f}  | {cv_val:12.3f} | {stability}")
                
            median_params[p] = float(med_val)
            wf_results[p] = {
                "values": [float(v) for v in vals],
                "median": float(med_val),
                "iqr": float(iqr_val),
                "cv": float(cv_val),
                "stability": stability
            }
            
        # 寫入最優中位數參數 JSON
        output_data = {
            "best_score": 0.0,
            "optimized_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": f"{args.start} to {args.end}",
            "best_params": median_params,
            "overall_metrics": {"return_pct": 0.0, "mdd_pct": 0.0},
            "walk_forward_metrics": wf_results
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n[成功] Walk-Forward 最佳中位數參數已匯出至: {RESULT_PATH}\n")
        
    else:
        # ── 標準單區間優化模式 ──
        def objective(trial):
            trial_params = {
                "buy_threshold": trial.suggest_float("buy_threshold", 5.0, 25.0, step=0.5),
                "stop_loss": trial.suggest_float("stop_loss", -15.0, -3.0, step=0.5),
                "panic_ma5": trial.suggest_float("panic_ma5", -0.025, 0.00, step=0.001),
                "panic_breadth": trial.suggest_float("panic_breadth", 0.15, 0.45, step=0.01),
                "ts_activation": trial.suggest_float("ts_activation", 5.0, 25.0, step=0.5),
                "ts_pullback": trial.suggest_float("ts_pullback", -12.0, -1.0, step=0.5)
            }
            
            score, regime_details, total_ret, max_dd = run_simulation_scoring(
                args.start, args.end, trial_params, args.capital, args.max_pos
            )
            
            # 記錄優化屬性
            for r in ["Bull", "Bear", "Sideways"]:
                details = regime_details.get(r, {"days": 0, "alpha_pct": 0.0, "spread_pct": 0.0, "return_pct": 0.0, "mdd_pct": 0.0, "score": 0.0})
                trial.set_user_attr(f"regime_days_{r}", int(details["days"]))
                trial.set_user_attr(f"regime_alpha_{r}", float(details["alpha_pct"]))
                trial.set_user_attr(f"regime_spread_{r}", float(details["spread_pct"]))
                trial.set_user_attr(f"regime_ret_{r}", float(details["return_pct"]))
                trial.set_user_attr(f"regime_mdd_{r}", float(details["mdd_pct"]))
                trial.set_user_attr(f"regime_score_{r}", float(details["score"]))
                
            trial.set_user_attr("overall_return", float(total_ret))
            trial.set_user_attr("overall_mdd", float(max_dd * 100))
            return score

        study = optuna.create_study(direction="maximize")
        
        import threading
        _print_lock = threading.Lock()

        def callback(study, trial):
            try:
                n = trial.number + 1
                val = trial.value if trial.value is not None else -999.0
                best = study.best_value
                is_best = abs(val - best) < 1e-6
                
                if is_best or n % 50 == 0 or n == 1:
                    tag = " << 新最佳解!" if is_best else ""
                    sub_info = []
                    for r in ["Bull", "Bear", "Sideways"]:
                        sub_score = trial.user_attrs.get(f"regime_score_{r}", 0.0)
                        sub_info.append(f"{r}:{sub_score:.2f}")
                    sub_str = " | ".join(sub_info)
                    
                    with _print_lock:
                        print(
                            f"  [{n:>4}/{args.trials:>4}] "
                            f"綜合得分={val:.2f} ({sub_str})  最佳={best:.2f} | "
                            f"Buy門檻: {trial.params['buy_threshold']}% | "
                            f"停損: {trial.params['stop_loss']}% | "
                            f"大盤MA5: {trial.params['panic_ma5']*100:.1f}% | "
                            f"上漲比例: {trial.params['panic_breadth']*100:.0f}% | "
                            f"移動止盈: 達 {trial.params['ts_activation']}% 回撤 {trial.params['ts_pullback']}%"
                            f"{tag}",
                            flush=True
                        )
            except Exception:
                pass
            
            if EARLY_STOPPING_ROUNDS is not None and EARLY_STOPPING_ROUNDS > 0:
                try:
                    best_trial_number = study.best_trial.number
                    current_trial_number = trial.number
                    rounds_without_improvement = current_trial_number - best_trial_number
                    if rounds_without_improvement >= EARLY_STOPPING_ROUNDS:
                        print(f"\n  [提早結束] 連續 {EARLY_STOPPING_ROUNDS} 次未找到更好的參數，觸發 Early Stopping！", flush=True)
                        study.stop()
                except Exception:
                    pass

        print("\n[開始調參] 正在執行貝葉斯搜尋最佳配置...")
        study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, callbacks=[callback])
        
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        
        print("\n" + "=" * 70)
        print("  [最佳化搜尋完成] 最佳風控策略配置如下：")
        print("=" * 70)
        print(f"  最佳綜合得分 (Regime-Robust Score): {best_value:.4f}")
        for r in ["Bull", "Bear", "Sideways"]:
            r_days = best_trial.user_attrs.get(f"regime_days_{r}", 0)
            r_ret = best_trial.user_attrs.get(f"regime_ret_{r}", 0.0)
            r_mdd = best_trial.user_attrs.get(f"regime_mdd_{r}", 0.0)
            r_score = best_trial.user_attrs.get(f"regime_score_{r}", 0.0)
            print(f"    - {r:<8} 市況績效 ({r_days:>3}天): 報酬率 {r_ret:+.2f}% | 最大回撤 {r_mdd:+.2f}% | 分數 {r_score:.2f}")
        print(f"    - 全區間整體績效: 報酬率 {best_trial.user_attrs.get('overall_return', 0.0):+.2f}% | 最大回撤 {best_trial.user_attrs.get('overall_mdd', 0.0):+.2f}%")
        print("-" * 70)
        print(f"  1. 買進分數門檻 (buy_threshold)  : {best_params['buy_threshold']:.1f}%")
        print(f"  2. 個股固定停損 (stop_loss)      : {best_params['stop_loss']:.1f}%")
        print(f"  3. 大盤5日報酬門檻 (panic_ma5)   : {best_params['panic_ma5']*100:.2f}% (實數: {best_params['panic_ma5']:.4f})")
        print(f"  4. 全市場上漲比例 (panic_breadth): {best_params['panic_breadth']*100:.1f}% (實數: {best_params['panic_breadth']:.2f})")
        print(f"  5. 移動止盈啟動線 (ts_activation): {best_params['ts_activation']:.1f}%")
        print(f"  6. 移動止盈回撤線 (ts_pullback)  : {best_params['ts_pullback']:.1f}%")
        print("=" * 70)
        
        output_data = {
            "best_score": best_value,
            "optimized_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": f"{args.start} to {args.end}",
            "best_params": best_params,
            "overall_metrics": {
                "return_pct": best_trial.user_attrs.get("overall_return", 0.0),
                "mdd_pct": best_trial.user_attrs.get("overall_mdd", 0.0)
            },
            "regime_metrics": {
                r: {
                    "days": best_trial.user_attrs.get(f"regime_days_{r}", 0),
                    "return_pct": best_trial.user_attrs.get(f"regime_ret_{r}", 0.0),
                    "mdd_pct": best_trial.user_attrs.get(f"regime_mdd_{r}", 0.0),
                    "alpha_pct": best_trial.user_attrs.get(f"regime_alpha_{r}", 0.0),
                    "spread_pct": best_trial.user_attrs.get(f"regime_spread_{r}", 0.0),
                    "score": best_trial.user_attrs.get(f"regime_score_{r}", 0.0)
                }
                for r in ["Bull", "Bear", "Sideways"]
            }
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n[成功] 最佳參數已匯出至: {RESULT_PATH}\n")


if __name__ == "__main__":
    main()
