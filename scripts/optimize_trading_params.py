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
RESULT_PATH = os.path.join(BASE_DIR, "best_trading_params.json")


def main():
    parser = argparse.ArgumentParser(description="交易與風控參數貝葉斯最佳化工具 (Optuna)")
    parser.add_argument("-s", "--start", type=str, default=default_start_date, help="回測起始日期 (YYYY-MM-DD)")
    parser.add_argument("-e", "--end", type=str, default=default_end_date, help="回測結束日期 (YYYY-MM-DD)")
    parser.add_argument("-c", "--capital", type=int, default=1000000, help="回測初始資金")
    parser.add_argument("-m", "--max_pos", type=int, default=MAX_POSITIONS, help="最大持持股上限檔數")
    parser.add_argument("-t", "--trials", type=int, default=OPTIMIZATION_TRIALS, help="最佳化搜尋輪數")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="並行搜尋執行緒數 (交易模擬包含I/O寫入，建議設為 1)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  交易與避險風控參數貝葉斯自動調參 (Optuna TPE)")
    print("=" * 70)
    print(f"  調參資料區間 : {args.start} 至 {args.end}")
    print(f"  模擬資金規模 : {args.capital:,} | 最大持股 slots: {args.max_pos}")
    print(f"  預期搜尋輪數 : {args.trials} 輪")
    print(f"  結果輸出存檔 : {RESULT_PATH}")
    print("=" * 70)
    
    # 檢查是否有模型與特徵
    feature_cols_path = os.path.join(BASE_DIR, "models", "feature_cols.json")
    if not os.path.exists(feature_cols_path):
        print(f"[錯誤] 找不到 feature_cols.json，請先執行 auto_pipeline.py 訓練模型。")
        sys.exit(1)
        
    def objective(trial):
        # 1. 定義參數的貝葉斯搜尋邊界
        buy_threshold = trial.suggest_float("buy_threshold", 5.0, 25.0, step=0.5)
        stop_loss = trial.suggest_float("stop_loss", -15.0, -3.0, step=0.5)
        panic_ma5 = trial.suggest_float("panic_ma5", -0.025, 0.00, step=0.001)
        panic_breadth = trial.suggest_float("panic_breadth", 0.15, 0.45, step=0.01)
        ts_activation = trial.suggest_float("ts_activation", 5.0, 25.0, step=0.5)
        ts_pullback = trial.suggest_float("ts_pullback", -12.0, -1.0, step=0.5)
        
        # 2. 執行模擬交易 (使用 contextlib 遮蔽交易明細的大量 print 輸出)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                total_return, max_dd, history = run_simulation(
                    start_date=args.start,
                    end_date=args.end,
                    initial_capital=args.capital,
                    max_positions=args.max_pos,
                    mkt_panic_ma5=panic_ma5,
                    mkt_panic_breadth=panic_breadth,
                    buy_threshold=buy_threshold,
                    stop_loss_pct=stop_loss,
                    ts_activation_pct=ts_activation,
                    ts_pullback_pct=ts_pullback
                )
            except Exception as e:
                # 發生運行錯誤時給予罰分
                return -999.0
                
        # 3. 計算多市況魯棒性適應度分數 (Regime-Robust CV)
        # 將全區間 Chronological (時間順序) 切分為 3 個子區間，分別計算 Calmar 分數
        n_days = len(history)
        if n_days < 3:
            # 歷史天數過短，退化為計算整體區間分數
            score = total_return - 2.0 * abs(max_dd * 100)
            return score
            
        chunk_size = n_days // 3
        sub_scores = []
        sub_details = []
        
        for i in range(3):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 2 else n_days
            sub_hist = history[start_idx:end_idx]
            
            if not sub_hist:
                sub_scores.append(-999.0)
                sub_details.append((0.0, 0.0))
                continue
                
            sub_initial = sub_hist[0]['equity']
            sub_final = sub_hist[-1]['equity']
            if sub_initial <= 0:
                sub_initial = 1.0
            sub_ret = ((sub_final / sub_initial) - 1) * 100
            
            sub_max_dd = 0.0
            peak = sub_hist[0]['equity']
            for h in sub_hist:
                e = h['equity']
                if e > peak:
                    peak = e
                denom = peak if peak > 0 else 1.0
                dd = (peak - e) / denom
                if dd > sub_max_dd:
                    sub_max_dd = dd
                    
            # 採用直觀的線性回撤懲罰法 (Score = Return - 2.0 * MDD)
            # 避免除以 MDD 造成的壓抑與壓縮效應
            sub_score = sub_ret - 2.0 * (sub_max_dd * 100)
                
            sub_scores.append(sub_score)
            sub_details.append((sub_ret, sub_max_dd * 100))
            
        # 4. 記錄子區間表現屬性，供 callback 列印與存檔
        for i, (sub_ret, sub_mdd) in enumerate(sub_details):
            trial.set_user_attr(f"sub_ret_{i+1}", float(sub_ret))
            trial.set_user_attr(f"sub_mdd_{i+1}", float(sub_mdd))
            trial.set_user_attr(f"sub_score_{i+1}", float(sub_scores[i]))
            
        trial.set_user_attr("overall_return", float(total_return))
        trial.set_user_attr("overall_mdd", float(max_dd * 100))
        
        # 5. 結合為多區間魯棒得分
        # 若所有子區間皆為正回報，使用調和平均值 (對差勁的單一區間進行重度懲罰)
        # 若有任何子區間為負回報，則直接回傳最小值 (Maximin 防守導向)
        if all(s > 0 for s in sub_scores):
            combined_score = len(sub_scores) / sum(1.0 / (s + 1e-6) for s in sub_scores)
        else:
            combined_score = min(sub_scores)
            
        return combined_score

    # 建立 Optuna 研討室，目標為最大化適應度分數
    study = optuna.create_study(direction="maximize")
    
    # 用於多執行緒安全控制的 lock
    import threading
    _print_lock = threading.Lock()

    # 註冊自訂進度條/日誌輸出
    def callback(study, trial):
        try:
            n = trial.number + 1
            val = trial.value if trial.value is not None else -999.0
            best = study.best_value
            
            is_best = abs(val - best) < 1e-6
            if is_best or n % 50 == 0 or n == 1:
                tag = " << 新最佳解!" if is_best else ""
                
                # 取得子區間分數詳情
                sub_info = []
                for i in range(3):
                    sub_score = trial.user_attrs.get(f"sub_score_{i+1}", 0.0)
                    sub_info.append(f"P{i+1}:{sub_score:.1f}")
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
        
        # 實作 Early Stopping
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
    
    # 輸出最佳參數
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    print("\n" + "=" * 70)
    print("  [最佳化搜尋完成] 最佳風控策略配置如下：")
    print("=" * 70)
    print(f"  最佳綜合得分 (Regime-Robust Score): {best_value:.4f}")
    for i in range(3):
        sub_ret = best_trial.user_attrs.get(f"sub_ret_{i+1}", 0.0)
        sub_mdd = best_trial.user_attrs.get(f"sub_mdd_{i+1}", 0.0)
        sub_score = best_trial.user_attrs.get(f"sub_score_{i+1}", 0.0)
        print(f"    - 子區間 {i+1} 績效: 報酬率 {sub_ret:+.2f}% | 最大回撤 {sub_mdd:+.2f}% | 分數 {sub_score:.2f}")
    print(f"    - 全區間整體績效: 報酬率 {best_trial.user_attrs.get('overall_return', 0.0):+.2f}% | 最大回撤 {best_trial.user_attrs.get('overall_mdd', 0.0):+.2f}%")
    print("-" * 70)
    print(f"  1. 買進分數門檻 (buy_threshold)  : {best_params['buy_threshold']:.1f}%")
    print(f"  2. 個股固定停損 (stop_loss)      : {best_params['stop_loss']:.1f}%")
    print(f"  3. 大盤5日報酬門檻 (panic_ma5)   : {best_params['panic_ma5']*100:.2f}% (實數: {best_params['panic_ma5']:.4f})")
    print(f"  4. 全市場上漲比例 (panic_breadth): {best_params['panic_breadth']*100:.1f}% (實數: {best_params['panic_breadth']:.2f})")
    print(f"  5. 移動止盈啟動線 (ts_activation): {best_params['ts_activation']:.1f}%")
    print(f"  6. 移動止盈回撤線 (ts_pullback)  : {best_params['ts_pullback']:.1f}%")
    print("=" * 70)
    
    # 寫入 JSON 存檔 (包含整體與子區間表現屬性)
    output_data = {
        "best_score": best_value,
        "optimized_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_range": f"{args.start} to {args.end}",
        "best_params": best_params,
        "overall_metrics": {
            "return_pct": best_trial.user_attrs.get("overall_return", 0.0),
            "mdd_pct": best_trial.user_attrs.get("overall_mdd", 0.0)
        },
        "sub_periods": [
            {
                "period": i + 1,
                "return_pct": best_trial.user_attrs.get(f"sub_ret_{i+1}", 0.0),
                "mdd_pct": best_trial.user_attrs.get(f"sub_mdd_{i+1}", 0.0),
                "score": best_trial.user_attrs.get(f"sub_score_{i+1}", 0.0)
            }
            for i in range(3)
        ]
    }
    
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[成功] 最佳參數已匯出至: {RESULT_PATH}\n")


if __name__ == "__main__":
    main()
