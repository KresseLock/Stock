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

輸出行為:
  預設寫入 configs/best_trading_params_candidate.json（候選檔，不影響部署）。
  config.py § 11 只自動載入 best_trading_params.json，故新參數須先以 OOS 回測驗證，
  確認優於現行參數後，加 --deploy 重跑或手動將候選檔複製為 best_trading_params.json 才會生效。
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
        MAX_POSITIONS, OPTIMIZATION_TRIALS, EARLY_STOPPING_ROUNDS,
        MDD_TOLERANCE, MDD_PENALTY_WEIGHT,
        PORTFOLIO_ALPHA_WEIGHT, PORTFOLIO_SPREAD_WEIGHT, CALMAR_SCORE_WEIGHT,
        TRADING_PARAM_BOUNDS, TRADING_OPT_START_DATE, TRADING_OPT_END_DATE,
        SELL_THRESHOLD, STOP_LOSS_PCT, TS_ACTIVATION_PCT, TS_PULLBACK_PCT, MIN_HOLD_DAYS,
    )
    from trading_sim import run_simulation
except ImportError as e:
    print(f"[錯誤] 載入核心模組失敗: {e}")
    sys.exit(1)

# 預設調參區間（可被 -s / -e 覆寫，常數集中於 config.py § 10.5）：
# 起點含 2022 完整熊市作壓力樣本；終點刻意停在裁判區之前，
# 使候選參數可與現行參數在「候選未見過」的區間公平對比後再部署。
default_start_date = TRADING_OPT_START_DATE
default_end_date = TRADING_OPT_END_DATE

# 部署參數檔（config.py § 11 啟動時自動載入）與候選參數檔路徑。
# 預設寫入候選檔，避免未經 OOS 驗證的優化結果直接覆蓋部署中的參數；--deploy 才寫部署檔。
RESULT_PATH = os.path.join(BASE_DIR, "configs", "best_trading_params.json")
CANDIDATE_PATH = os.path.join(BASE_DIR, "configs", "best_trading_params_candidate.json")


def _suggest(trial, name):
    """依 config.TRADING_PARAM_BOUNDS 的邊界定義建議參數 (3-tuple=float, 2-tuple=int)。"""
    bounds = TRADING_PARAM_BOUNDS[name]
    if len(bounds) == 3:
        lo, hi, step = bounds
        return trial.suggest_float(name, lo, hi, step=step)
    lo, hi = bounds
    return trial.suggest_int(name, lo, hi)


def search_space_names(regime_mode):
    """搜尋空間參數名稱清單（暖啟動相容性檢查與 suggest 共用，避免兩處清單漂移）。"""
    names = ["panic_ma5", "panic_breadth", "markup_pct"]
    if regime_mode:
        # 市況過濾器：趨勢市低門檻進攻、震盪市高門檻防守、空頭固定空倉(99)
        names += ["regime_bull_buy", "regime_sideways_buy",
                  "regime_bull_trend", "regime_bear_trend",
                  "regime_bull_pos", "regime_sideways_pos", "regime_bear_pos"]
    else:
        names.append("buy_threshold")
    return names


def suggest_trial_params(trial, regime_mode):
    """集中定義 Optuna 搜尋空間。regime_mode 時搜尋 REGIME_* 動態門檻，否則搜尋靜態 buy_threshold。
    搜尋邊界統一由 config.TRADING_PARAM_BOUNDS 控制 (常數集中原則)。

    僅搜尋「部署時真正生效」的參數：panic_*（大盤恐慌門檻）、markup_pct（限價搓合溢價）為全域生效。
    stop_loss 由 ATR 動態停損覆蓋 (config.ATR_STOP_ENABLED=True，搜尋值僅為 ATR 缺值 fallback)，
    sell_threshold / ts_activation / ts_pullback / min_hold_days 在部署時由 config.REGIME_EXIT_PARAMS
    依市況覆蓋（trading_sim.py 載入 regime_exit_params=None 即套用），故全數移出搜尋空間，
    避免 Optuna 在「不會被部署」的維度空轉並污染 Walk-Forward 穩定度報表。"""
    return {name: _suggest(trial, name) for name in search_space_names(regime_mode)}


def run_simulation_scoring(start_date, end_date, trial_params, capital, max_pos):
    import io
    import contextlib
    from trading_sim import run_simulation
    import numpy as np
    
    sim_kwargs = dict(
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        max_positions=max_pos,
        mkt_panic_ma5=trial_params["panic_ma5"],
        mkt_panic_breadth=trial_params["panic_breadth"],
        markup_pct=trial_params["markup_pct"],
        # 與實盤部署完全一致：regime_exit_params=None → trading_sim 載入 config.REGIME_EXIT_PARAMS
        # 依市況切換出場 (sell_threshold/ts_*/min_hold_days)；stop_loss 不傳 → 由 ATR 動態停損接管。
        # 這三類參數不在此搜尋 (見 suggest_trial_params)，確保「優化評分 == 部署行為」。
        regime_exit_params=None,
        export_report=False,
    )
    if "regime_bull_buy" in trial_params:
        # 市況過濾器模式：搜尋 regime 動態門檻，buy_threshold 留空 (None) 以啟用過濾器
        sim_kwargs["regime_buy_threshold"] = {
            "Bull":     trial_params["regime_bull_buy"],
            "Sideways": trial_params["regime_sideways_buy"],
            "Bear":     99.0,
        }
        sim_kwargs["regime_bull_trend"] = trial_params["regime_bull_trend"]
        sim_kwargs["regime_bear_trend"] = trial_params["regime_bear_trend"]
        # 市況選擇性曝險：各 regime 持股檔數上限均由優化器決定（含 Bull，舊版固定滿倉）
        sim_kwargs["regime_max_positions"] = {
            "Bull":     trial_params.get("regime_bull_pos", max_pos),
            "Sideways": trial_params["regime_sideways_pos"],
            "Bear":     trial_params["regime_bear_pos"],
        }
    else:
        sim_kwargs["buy_threshold"] = trial_params["buy_threshold"]

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            total_return, max_dd, history = run_simulation(**sim_kwargs)
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
        
        # 套用 Alpha / Spread / Calmar 權重公式 (權重集中於 config.py 第 119-121 行)
        sub_score = (PORTFOLIO_ALPHA_WEIGHT * (mean_alpha * 100)
                     + PORTFOLIO_SPREAD_WEIGHT * (mean_spread * 100)
                     + CALMAR_SCORE_WEIGHT * regime_calmar)
        
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

    # 全期 MDD 懲罰：上面的 combined_score 只彙整 per-regime alpha/spread/calmar，
    # 看不到跨 regime 交界 (Bull→崩盤) 的全期回撤。在此對全期 max_dd 超出容忍線的部分線性扣分，
    # 逼優化器避開「高報酬但 -44% MDD」的角落解 (WF 與標準模式共用此回傳值，故兩者皆生效)。
    if combined_score > -999.0 and MDD_PENALTY_WEIGHT > 0:
        mdd_excess = max(0.0, max_dd * 100 - MDD_TOLERANCE)
        combined_score -= MDD_PENALTY_WEIGHT * mdd_excess

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
    parser.add_argument("--regime", action="store_true",
                        help="市況過濾器模式：搜尋 REGIME_* 動態門檻 (Bull/Sideways 門檻與趨勢分界)，而非靜態 buy_threshold")
    parser.add_argument("--recency_weight", type=float, default=1.0,
                        help="WFO 近期加權係數：>1 時讓最新窗口擁有更高權重（例: 2 = 每窗口權重是前一個的 2 倍），預設 1.0 為普通中位數")
    parser.add_argument("--deploy", action="store_true",
                        help="直接寫入 best_trading_params.json（立即部署生效）；"
                             "預設寫入 best_trading_params_candidate.json，經 OOS 驗證後再手動部署")

    args = parser.parse_args()

    import numpy as np

    # 結果輸出路徑：預設候選檔（不影響部署），--deploy 才覆寫部署檔
    out_path = RESULT_PATH if args.deploy else CANDIDATE_PATH

    print("=" * 70)
    print("  交易與避險風控參數貝葉斯自動調參 (Optuna TPE)")
    print("=" * 70)
    print(f"  調參資料區間 : {args.start} 至 {args.end}")
    print(f"  模擬資金規模 : {args.capital:,} | 最大持股 slots: {args.max_pos}")
    print(f"  預期搜尋輪數 : {args.trials} 輪")
    print(f"  結果輸出存檔 : {out_path}")
    if not args.deploy:
        print("  部署模式     : 否（候選檔；驗證後手動複製為 best_trading_params.json，或加 --deploy 重跑）")
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
                trial_params = suggest_trial_params(trial, args.regime)
                score, _, _, _ = run_simulation_scoring(ws, we, trial_params, args.capital, args.max_pos)
                return score
                
            study = optuna.create_study(direction="maximize")
            with open(os.devnull, "w", encoding="utf-8") as _devnull, contextlib.redirect_stdout(_devnull):
                study.optimize(wf_objective, n_trials=wf_trials, n_jobs=args.jobs)
            print(f"  [窗口 {idx+1} 完成] 最佳得分: {study.best_value:.4f} | 參數: {study.best_params}")
            all_best_params.append(study.best_params)
            
        # 統計與部署 (參數名稱依模式動態取得)
        param_names = list(all_best_params[0].keys()) if all_best_params else []
        wf_results = {}
        median_params = {}
        
        print("\n" + "=" * 75)
        print("  [Walk-Forward 參數穩定性統計報表 (Step 6 / Phase 3)]")
        print("=" * 75)
        print(f"{'參數名稱':<15} | {'中位數(Median)':<15} | {'四分位距(IQR)':<15} | {'變異係數(CV)':<12} | {'穩定度':<6}")
        print("-" * 75)
        
        def _weighted_median(raw_vals, rw):
            """加權中位數：rw=1 退化為普通中位數，>1 時讓最新窗口（index 最大）主導。
            weights[i] = rw^i，最舊窗口 = rw^0 = 1，最新窗口 = rw^(n-1)。"""
            if rw <= 1.0:
                return float(np.median(raw_vals))
            n = len(raw_vals)
            raw_w = [rw ** i for i in range(n)]
            min_w = min(raw_w)
            expanded = []
            for v, w in zip(raw_vals, raw_w):
                expanded.extend([float(v)] * max(1, round(w / min_w)))
            expanded.sort()
            m = len(expanded)
            if m % 2 == 0:
                return (expanded[m // 2 - 1] + expanded[m // 2]) / 2.0
            return expanded[m // 2]

        rw = args.recency_weight
        if rw > 1.0:
            print(f"\n  [加權中位數] recency_weight={rw}，窗口權重 = {[round(rw**i, 2) for i in range(len(all_best_params))]}")
            print("  (穩定度統計仍用普通中位數/IQR，best_params 採加權中位數以反映近期市況)")

        for p in param_names:
            vals = np.array([bp[p] for bp in all_best_params])
            med_val = np.median(vals)
            q75, q25 = np.percentile(vals, [75, 25])
            iqr_val = q75 - q25
            std_val = np.std(vals)
            mean_val = np.mean(vals)
            cv_val = std_val / abs(mean_val) if abs(mean_val) > 0 else 0.0
            stability = "穩定" if cv_val < 0.15 else "需注意" if cv_val < 0.30 else "不穩定"
            deploy_val = _weighted_median(list(vals), rw)

            # 整數型參數 (TRADING_PARAM_BOUNDS 2-tuple，如 regime_*_pos)：窗口數為偶數時中位數會落在
            # x.5，而 config.py § 11 載入時以 int() 截斷 (3.5 → 3)，導致參數檔記錄值與實際部署值不一致。
            # 此處以相同的截斷規則先取整，使 JSON 記錄 == 實際部署（不改變既有部署行為）。
            # 型別判定沿用 _suggest 的規則，以 TRADING_PARAM_BOUNDS 為單一來源。
            if len(TRADING_PARAM_BOUNDS.get(p, ())) == 2:
                deploy_val = int(deploy_val)

            if p in ["panic_ma5", "panic_breadth"]:
                print(f"{p:<15} | {deploy_val*100:13.2f}% | {iqr_val*100:13.2f}% | {cv_val:12.3f} | {stability}")
            else:
                print(f"{p:<15} | {deploy_val:14.2f}  | {iqr_val:14.2f}  | {cv_val:12.3f} | {stability}")

            median_params[p] = deploy_val
            wf_results[p] = {
                "values": [float(v) for v in vals],
                "median": float(med_val),
                "deployed": deploy_val,
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
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"\n[成功] Walk-Forward 最佳中位數參數已匯出至: {out_path}")
        if not args.deploy:
            print("[提示] 此為候選參數，尚未部署。請先以 OOS 區間回測驗證，"
                  "確認優於現行參數後再複製為 best_trading_params.json。")
        print()
        
    else:
        # ── 標準單區間優化模式 ──
        def objective(trial):
            trial_params = suggest_trial_params(trial, args.regime)

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

        # 暖啟動：若存在上一輪結果，將最佳參數 enqueue 為 trial #0，
        # 讓 TPE 從已知好解附近展開搜尋，避免浪費前幾輪在純隨機探索。
        # 來源優先序：候選檔（最新一次搜尋結果）→ 部署檔。
        _warm_src = CANDIDATE_PATH if os.path.exists(CANDIDATE_PATH) else RESULT_PATH
        if os.path.exists(_warm_src):
            try:
                with open(_warm_src, encoding="utf-8") as _f:
                    _prev = json.load(_f)
                _prev_params = _prev.get("best_params", {})
                if _prev_params:
                    # 確認與目前搜尋模式相容（regime / 非 regime）
                    _expected = set(search_space_names(args.regime))
                    _filtered = {k: v for k, v in _prev_params.items() if k in _expected}
                    if len(_filtered) == len(_expected):
                        study.enqueue_trial(_filtered)
                        print(f"  [暖啟動] 已從 {os.path.basename(_warm_src)} 載入上輪最佳參數作為起點"
                              f"（得分: {_prev.get('best_score', '?')}）")
                    else:
                        print("  [暖啟動] 上輪使用不同模式（regime/非regime），跳過暖啟動")
            except Exception as _e:
                print(f"  [暖啟動] 讀取失敗，跳過（{_e}）")

        import threading
        _print_lock = threading.Lock()

        # -j 多執行緒時 run_simulation_scoring 內的 redirect_stdout 會全域替換 sys.stdout
        # （非執行緒安全），callback 若用預設 stdout 列印，進度會被別的 trial 緩衝吃掉。
        # 故先鎖住真正的 console 供進度列印，optimize 期間全域 stdout 導向 devnull 吸收漏出的模擬輸出。
        _real_stdout = sys.stdout

        def _save_checkpoint(study, trial):
            """發現新最佳解時立即寫入 JSON，防止中途中斷遺失進度。"""
            try:
                _bt = study.best_trial
                _chk = {
                    "best_score": study.best_value,
                    "optimized_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "date_range": f"{args.start} to {args.end}",
                    "best_params": _bt.params,
                    "overall_metrics": {
                        "return_pct": _bt.user_attrs.get("overall_return", 0.0),
                        "mdd_pct": _bt.user_attrs.get("overall_mdd", 0.0)
                    },
                    "regime_metrics": {
                        r: {
                            "days": _bt.user_attrs.get(f"regime_days_{r}", 0),
                            "return_pct": _bt.user_attrs.get(f"regime_ret_{r}", 0.0),
                            "mdd_pct": _bt.user_attrs.get(f"regime_mdd_{r}", 0.0),
                            "alpha_pct": _bt.user_attrs.get(f"regime_alpha_{r}", 0.0),
                            "spread_pct": _bt.user_attrs.get(f"regime_spread_{r}", 0.0),
                            "score": _bt.user_attrs.get(f"regime_score_{r}", 0.0)
                        }
                        for r in ["Bull", "Bear", "Sideways"]
                    }
                }
                with open(out_path, "w", encoding="utf-8") as _f_chk:
                    json.dump(_chk, _f_chk, indent=4, ensure_ascii=False)
            except Exception:
                pass  # 不因存檔失敗而中斷優化

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

                    tp = trial.params
                    if "regime_bull_buy" in tp:
                        buy_str = (f"市況門檻[多:{tp['regime_bull_buy']}/盤:{tp['regime_sideways_buy']}]% "
                                   f"趨勢界[多>{tp['regime_bull_trend']:.4f}/空<{tp['regime_bear_trend']:.4f}]")
                    else:
                        buy_str = f"Buy門檻: {tp['buy_threshold']}%"
                    with _print_lock:
                        print(
                            f"  [{n:>4}/{args.trials:>4}] "
                            f"綜合得分={val:.2f} ({sub_str})  最佳={best:.2f} | "
                            f"{buy_str} | "
                            f"Sell門檻: {tp.get('sell_threshold', SELL_THRESHOLD)}% | "
                            f"停損: {tp.get('stop_loss', STOP_LOSS_PCT)}% | "
                            f"大盤MA5: {tp['panic_ma5']*100:.1f}% | "
                            f"上漲比例: {tp['panic_breadth']*100:.0f}% | "
                            f"移動止盈: 達 {tp.get('ts_activation', TS_ACTIVATION_PCT)}% 回撤 {tp.get('ts_pullback', TS_PULLBACK_PCT)}% | "
                            f"持股天數: {tp.get('min_hold_days', MIN_HOLD_DAYS)}天 | 折溢價: {tp['markup_pct']}%"
                            f"{tag}",
                            file=_real_stdout, flush=True
                        )
            except Exception:
                pass

            # 新最佳解即時寫入 checkpoint
            # 超過 EARLY_STOPPING_ROUNDS 才開始存，避免探索期前幾十輪頻繁寫檔
            try:
                val = trial.value if trial.value is not None else -999.0
                n = trial.number + 1
                past_exploration = (
                    EARLY_STOPPING_ROUNDS is None
                    or EARLY_STOPPING_ROUNDS <= 0
                    or n > EARLY_STOPPING_ROUNDS
                )
                if past_exploration and study.best_value is not None and abs(val - study.best_value) < 1e-6:
                    _save_checkpoint(study, trial)
            except Exception:
                pass

            if EARLY_STOPPING_ROUNDS is not None and EARLY_STOPPING_ROUNDS > 0:
                try:
                    best_trial_number = study.best_trial.number
                    current_trial_number = trial.number
                    rounds_without_improvement = current_trial_number - best_trial_number
                    if rounds_without_improvement >= EARLY_STOPPING_ROUNDS:
                        print(f"\n  [提早結束] 連續 {EARLY_STOPPING_ROUNDS} 次未找到更好的參數，觸發 Early Stopping！",
                              file=_real_stdout, flush=True)
                        study.stop()
                except Exception:
                    pass

        print("\n[開始調參] 正在執行貝葉斯搜尋最佳配置...")
        with open(os.devnull, "w", encoding="utf-8") as _devnull, contextlib.redirect_stdout(_devnull):
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
        if "regime_bull_buy" in best_params:
            print(f"  1. 市況動態買入門檻 (regime):")
            print(f"       - Bull 多頭進攻門檻 : {best_params['regime_bull_buy']:.1f}%")
            print(f"       - Sideways 防守門檻 : {best_params['regime_sideways_buy']:.1f}%")
            print(f"       - Bear 空頭         : 99.0% (實質空倉)")
            print(f"       - 趨勢分界 (t20)    : Bull > {best_params['regime_bull_trend']:.4f} | Bear < {best_params['regime_bear_trend']:.4f}")
            if "regime_bull_pos" in best_params:
                print(f"       - 持股檔數上限      : Bull {int(best_params['regime_bull_pos'])} | "
                      f"Sideways {int(best_params['regime_sideways_pos'])} | Bear {int(best_params['regime_bear_pos'])} 檔")
        else:
            print(f"  1. 買進分數門檻 (buy_threshold)  : {best_params['buy_threshold']:.1f}%")
        print(f"  1.5 賣出分數門檻 (sell_threshold): {best_params.get('sell_threshold', SELL_THRESHOLD):.1f}%")
        print(f"  2. 個股固定停損 (stop_loss)      : {best_params.get('stop_loss', STOP_LOSS_PCT):.1f}%")
        print(f"  3. 大盤5日報酬門檻 (panic_ma5)   : {best_params['panic_ma5']*100:.2f}% (實數: {best_params['panic_ma5']:.4f})")
        print(f"  4. 全市場上漲比例 (panic_breadth): {best_params['panic_breadth']*100:.1f}% (實數: {best_params['panic_breadth']:.2f})")
        print(f"  5. 移動止盈啟動線 (ts_activation): {best_params.get('ts_activation', TS_ACTIVATION_PCT):.1f}%")
        print(f"  6. 移動止盈回撤線 (ts_pullback)  : {best_params.get('ts_pullback', TS_PULLBACK_PCT):.1f}%")
        print(f"  7. 最少持股天數 (min_hold_days)  : {best_params.get('min_hold_days', MIN_HOLD_DAYS)}天")
        print(f"  8. 掛單溢價幅度 (markup_pct)     : {best_params['markup_pct']:.1f}%")
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
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"\n[成功] 最佳參數已匯出至: {out_path}")
        if not args.deploy:
            print("[提示] 此為候選參數，尚未部署（config.py 只自動載入 best_trading_params.json）。")
            print("       建議先用現行參數未見過的 OOS 區間跑 trading_sim.py 對比，"
                  "確認更優後再複製為 best_trading_params.json。")
        print()


if __name__ == "__main__":
    main()
