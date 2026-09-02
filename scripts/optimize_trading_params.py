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
        MDD_TOLERANCE, MDD_PENALTY_WEIGHT, REGIME_MAX_POSITIONS,
        WF_STABILITY_CV_WARN, WF_STABILITY_CV_BAD,
        WF_HOLDOUT_RATIO, WF_HOLDOUT_MIN_DAYS, WF_SELECT_RULE, WF_PROGRESS_EVERY,
        PORTFOLIO_ALPHA_WEIGHT, PORTFOLIO_SPREAD_WEIGHT, CALMAR_SCORE_WEIGHT,
        SCORE_TRADING_DAYS_PER_YEAR, SCORE_ANN_FACTOR_MAX,
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


def _suggest(trial, name, lo_min=None, hi_max=None):
    """依 config.TRADING_PARAM_BOUNDS 的邊界定義建議參數 (3-tuple=float, 2-tuple=int)。

    lo_min / hi_max 可再收緊該次 trial 的上下界，供 suggest_trial_params 施加單調約束；
    兩者皆為「與 config 邊界取交集」，故 TRADING_PARAM_BOUNDS 仍是唯一的邊界來源。
    """
    bounds = TRADING_PARAM_BOUNDS[name]
    if len(bounds) == 3:
        lo, hi, step = bounds
        if lo_min is not None:
            lo = max(lo, lo_min)
        if hi_max is not None:
            hi = min(hi, hi_max)
        return trial.suggest_float(name, lo, max(lo, hi), step=step)
    lo, hi = bounds
    if lo_min is not None:
        lo = max(lo, int(lo_min))
    if hi_max is not None:
        hi = min(hi, int(hi_max))
    return trial.suggest_int(name, lo, max(lo, hi))


def snap_to_bounds(name, value):
    """把值夾進 TRADING_PARAM_BOUNDS 範圍並對齊 step 網格，回傳可被 enqueue 的值。

    暖啟動專用。Walk-Forward 產出的中位數經常落在網格外（例：panic_ma5 四窗中位數
    -0.0145，而 step=0.001；regime_sideways_buy 11.25，而 step=0.5），Optuna 的
    enqueue_trial 會判定 "out of range"、丟棄該值改回隨機抽樣，只留下一行 UserWarning，
    使「從上輪最佳解展開搜尋」實際上從未生效。

    僅調整搜尋起點，不動任何部署值——參數檔記錄的原始中位數才是實際部署值。
    """
    bounds = TRADING_PARAM_BOUNDS[name]
    if len(bounds) == 3:
        lo, hi, step = bounds
        v = lo + round((min(max(float(value), lo), hi) - lo) / step) * step
        # lo + n*step 會累積浮點誤差（產生 0.30000000000000004 之類），四捨五入後再夾一次
        return round(min(max(v, lo), hi), 10)
    lo, hi = bounds
    return int(min(max(round(float(value)), lo), hi))


def enforce_monotonic(params):
    """就地修復參數向量的單調約束，回傳同一個 dict（非 regime 模式為 no-op）。

    約束定義見 suggest_trial_params。任何「不是由 suggest_trial_params 直接產生」的向量
    都可能違反它——暖啟動讀進來的舊參數檔（加約束之前產生的）、以及 Walk-Forward 把不同
    窗口的值混搭出來的向量（如穩定維度取中位數、不穩定維度沿用現行部署值）皆是。
    違反時該維度的 suggest 邊界會被收緊，實際跑的參數就不是這個向量了。
    """
    if "regime_bull_buy" not in params:
        return params
    params["regime_sideways_buy"] = max(params["regime_sideways_buy"], params["regime_bull_buy"])
    params["regime_sideways_pos"] = min(params["regime_sideways_pos"], params["regime_bull_pos"])
    if "regime_bear_pos" in params:   # 舊檔才有，見 search_space_names
        params["regime_bear_pos"] = min(params["regime_bear_pos"], params["regime_sideways_pos"])
    return params


def select_by_deploy_gate(scored):
    """從候選向量池依**部署判準**挑一組，回傳 (選中名稱, Pareto 前緣名稱清單)。

    `scored` 為 [(目標函式得分, 名稱, 參數, 報酬%, 回撤小數)]，量測區間由呼叫端決定
    （預設是四窗都沒看過的尾端 holdout）。

    兩段式：
      1. **Pareto 前緣（硬門檻）**：剔除「存在另一個向量在報酬與回撤上同時不差、且至少一項更好」
         的向量。被支配＝依 §4.5 #4 雙贏判準必輸，不可能是對的選擇。
      2. **前緣內取 Calmar 最高**：calmar = 報酬% / 回撤%。

    為何不直接用 Optuna 目標函式得分排序（舊行為，`WF_SELECT_RULE="objective"`）：
    目標函式與部署判準是兩把不同的尺，實測會選到被雙面支配的向量——2026-08-31 兩次獨立煙霧測試
    兩次都發生（一次是窗口4最佳 +6.42%/5.09% 輸給窗口1最佳 −5.36%/16.57%，一次是窗口3最佳
    +14.25%/15.90% 輸給窗口1最佳 −2.26%/16.82%）。詳見 scripts/EXPERIMENTS_PENDING.md 缺口#2。

    為何 Pareto 前緣要當硬門檻而不是只排 Calmar：報酬為負時 calmar 會**獎勵更大的回撤**
    （−2%/50% 算出來的 −0.04 高於 −2%/10% 的 −0.2），而 holdout 上多數向量報酬為負是常態，
    直接排 Calmar 會踩進這個陷阱；前緣過濾把這類向量先擋掉。

    注意：目標函式仍照常驅動每個窗口內的 Optuna 搜尋，本函式只決定「最後挑哪一組交付」。
    """
    items = [(_n, _r, _m * 100.0) for _s, _n, _, _r, _m in scored]   # 回撤統一轉成正的百分比
    front = []
    for name, ret, mdd in items:
        dominated = any(
            (o_ret >= ret and o_mdd <= mdd) and (o_ret > ret or o_mdd < mdd)
            for o_name, o_ret, o_mdd in items if o_name != name
        )
        if not dominated:
            front.append(name)
    pool = [(n, r, m) for n, r, m in items if n in front] or items
    _calmar = lambda r, m: r / m if m > 1e-9 else (float("inf") if r > 0 else r)
    best = max(pool, key=lambda x: _calmar(x[1], x[2]))
    return best[0], front


def search_space_names(regime_mode):
    """搜尋空間參數名稱清單（暖啟動相容性檢查與 suggest 共用，避免兩處清單漂移）。

    **順序有意義**：suggest_trial_params 的單調約束以先前已抽樣的值收緊後續邊界，
    故 regime_bull_buy 須早於 regime_sideways_buy、regime_bull_pos 須早於
    regime_sideways_pos。調整此清單順序前請一併檢查該處。

    regime_bear_pos 不在此清單：Bear 的買入門檻固定 99.0（實質空倉、不在搜尋空間），
    而 eff_max_positions 只用於擋新進場（trading_sim.py:523），故 Bear 檔數上限沒有任何
    生效路徑，是死維度——留著只會空耗搜尋預算並在穩定度報表產生假數字。
    部署時由 config.REGIME_MAX_POSITIONS["Bear"] 提供固定值。
    若日後 Bear 改為可進場，須同時把它加回本清單與單調約束。"""
    names = ["panic_ma5", "panic_breadth", "markup_pct"]
    if regime_mode:
        # 市況過濾器：趨勢市低門檻進攻、震盪市高門檻防守、空頭固定空倉(99)
        names += ["regime_bull_buy", "regime_sideways_buy",
                  "regime_bull_trend", "regime_bear_trend",
                  "regime_bull_pos", "regime_sideways_pos"]
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
    避免 Optuna 在「不會被部署」的維度空轉並污染 Walk-Forward 穩定度報表。

    regime 模式另施加**單調約束**（CLAUDE.md §4.5 #2「選擇性降曝險」的經濟意義要求）：
        regime_bull_pos >= regime_sideways_pos                      （多頭曝險不得低於震盪）
        regime_bull_buy <= regime_sideways_buy                      （多頭進場門檻不得高於震盪）
    動機是**規則的經濟意義，不是績效**：無約束時優化器會回傳倒置解（實例：2026-08-26 候選
    bull_pos=2 < bear_pos=3，即「多頭只准持 2 檔、空頭反而准持 3 檔」；bear_pos 已於
    2026-08-27 移出搜尋空間，見 search_space_names），這種解使 REGIME_MAX_POSITIONS 的
    選擇性降曝險完全失效，退化成 CLAUDE.md §4.5 #2 所述的「均勻降風險」。

    **勿拿績效替這個約束背書**：2026-08-27 以 scripts/validate_candidate_params.py 跑 9 條
    多起點（裁判區 2026-01-02~08-27）對照無約束的同源候選，結果是「報酬 9/9 贏、回撤 9/9 輸、
    雙贏 0/9」，且加約束的向量在調參區間的目標函式得分反而更低。約束保住的是規則的合理性，
    報酬與回撤仍受 §4.5 #2 的不可分割性支配。（早期註記的「Calmar 2.67 -> 2.28」為單路徑
    結論，已被多起點推翻，見 EXPERIMENTS_PENDING.md 方法論鐵律 #1／#2。）

    另因中位數對逐點支配具單調性（x_i >= y_i for all i  =>  median(x) >= median(y)，
    int() 截斷亦保序），各窗口滿足約束即保證中位數向量也滿足，不會再拼出倒置組合。"""
    params = {}
    for name in search_space_names(regime_mode):
        if name == "regime_sideways_buy":
            params[name] = _suggest(trial, name, lo_min=params["regime_bull_buy"])
        elif name == "regime_sideways_pos":
            params[name] = _suggest(trial, name, hi_max=params["regime_bull_pos"])
        else:
            params[name] = _suggest(trial, name)
    return params


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
            # Bear 不在搜尋空間（見 search_space_names）；沿用 config 固定值，
            # 舊參數檔若還帶著 regime_bear_pos 仍照舊生效，確保新舊檔行為一致。
            "Bear":     trial_params.get("regime_bear_pos", REGIME_MAX_POSITIONS["Bear"]),
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
        
        # per-regime 報酬／回撤：以「僅該 regime 交易日」的日報酬複利重建淨值曲線。
        # 舊版取 h['equity'] 首尾相除，但 regime 日並不連續（Bull 日散落全期），
        # 會把夾在中間的 Sideways/Bear 日損益一併計入，使三個 regime 的 regime_ret
        # 幾乎都退化成全期報酬、regime_max_dd 亦混入其他 regime 的回撤。
        regime_curve = []
        _eq = 1.0
        for h in sub_hist:
            _eq *= (1.0 + h['portfolio_return'])
            regime_curve.append(_eq)
        regime_ret = regime_curve[-1] - 1.0

        regime_max_dd = 0.0
        peak = 1.0   # 起始淨值，確保該 regime 首日即下跌也計入回撤
        for e in regime_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0.0
            if dd > regime_max_dd:
                regime_max_dd = dd
                
        # 年化 regime_ret 再算 calmar。舊版直接用累積複利報酬當分子，天數越多灌得越大
        # （487 個 Bull 日 → +825.79%，分母只有這些日子自己的回撤 8.46% → calmar 78.93），
        # 使 Bull 單項佔總分 99.6%、alpha/spread 的權重完全失效，並系統性獎勵「Bull 全押」
        # ——其代價是 regime 交界的全期回撤，per-regime 曲線看不到它。詳見 config.py §2.4。
        # ann_factor 上限防短 regime 爆炸（如 10 天 +5% 年化成 +240%）。
        ann_factor = min(SCORE_TRADING_DAYS_PER_YEAR / n_days_r, SCORE_ANN_FACTOR_MAX)
        regime_ret_ann = (1.0 + regime_ret) ** ann_factor - 1.0 if regime_ret > -1.0 else -1.0
        regime_calmar = (regime_ret_ann * 100) / ((regime_max_dd * 100) + 2.0)
        
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
            "return_ann_pct": regime_ret_ann * 100,
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

    # 全程 tee：主控台看得到進度，reports/ 下的時間戳 log 留得下記錄（見 utils.start_tee_log）
    from scripts.utils import start_tee_log
    _log_path = start_tee_log("optimize_trading_params")

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
    print(f"  執行記錄存檔 : {os.path.relpath(_log_path, BASE_DIR)}")
    print("=" * 70)

    # 裁判區污染警示（守門 1/3）：調參終點越過 config.TRADING_OPT_END_DATE 時，本輪參數已「看過」
    # 原本保留給裁判驗證的區間，日後任何落在該區間的對照對它而言都是樣本內。
    # 刻意只警示、不擋：run_workflow_experiment.py B6（模式 B ＝實盤部署組）設計上就以
    # 「-e 最新資料日 + --deploy」產出參數，實盤部署本來就該用全部可得資料調參，擋下會打死整條流水線。
    # 真正該擋的是「拿看過裁判區的參數當對照」，那一關在 scripts/validate_candidate_params.py。
    if args.end > TRADING_OPT_END_DATE:
        _clean_start = (datetime.datetime.strptime(args.end, "%Y-%m-%d")
                        + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  [裁判區警示] 調參終點 {args.end} 已越過 TRADING_OPT_END_DATE ({TRADING_OPT_END_DATE})，")
        print(f"               本輪參數已看過至 {args.end} 的資料，該段不可再用於驗證本參數；")
        print(f"               其乾淨裁判區起點為 {_clean_start}。")
        print("=" * 70)

    feature_cols_path = os.path.join(BASE_DIR, "models", "feature_cols.json")
    if not os.path.exists(feature_cols_path):
        print(f"[錯誤] 找不到 feature_cols.json，請先執行 auto_pipeline.py 訓練模型。")
        sys.exit(1)
        
    if args.walk_forward:
        # ── Walk-Forward 模式 ──
        start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(args.end, "%Y-%m-%d")
        span_days = (end_dt - start_dt).days

        # ── 尾端 holdout ────────────────────────────────────────────────────
        # 調參區間尾端切一段出來，四個子窗口都不許看，只用於下方「候選向量評分守門」的排序。
        # 病灶：先前向量池是拿「整段調參區間」評分排序，而四窗各佔總長 65%、彼此高度重疊，
        # 每個窗口自己的訓練期都落在那段裡面 → in-sample 選美，排第一不代表前瞻泛化力。
        # 實證（2026-08-31）：第 1 名 0.7994 與第 2 名 0.7292 只差 0.07，裁判區報酬中位數卻
        # 差 70pp 以上（+61.44% vs -9.24%），排序等於建立在各窗自己的樣本內表現上。
        # 代價：holdout 段不再參與參數擬合（四窗搜不到那段），只當裁判。
        # 停用條件：WF_HOLDOUT_RATIO = 0，或切出來不足 WF_HOLDOUT_MIN_DAYS（區間太短，
        # 如煙霧測試）→ 退回舊行為並明講，不靜默改變語意。
        _holdout_days = int(span_days * WF_HOLDOUT_RATIO)
        if WF_HOLDOUT_RATIO > 0 and _holdout_days >= WF_HOLDOUT_MIN_DAYS:
            fit_end_dt = end_dt - datetime.timedelta(days=_holdout_days)
            holdout_start = (fit_end_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            holdout_end = args.end
        else:
            fit_end_dt = end_dt
            holdout_start = holdout_end = None
            if WF_HOLDOUT_RATIO > 0:
                print("")
                print(f"  [WF holdout] 停用：依 WF_HOLDOUT_RATIO={WF_HOLDOUT_RATIO} 只切得出 "
                      f"{_holdout_days} 天，低於 WF_HOLDOUT_MIN_DAYS={WF_HOLDOUT_MIN_DAYS}；")
                print("               向量排序退回在整段調參區間上進行（in-sample 選美，結果僅供參考）。")

        fit_end = fit_end_dt.strftime("%Y-%m-%d")
        total_days = (fit_end_dt - start_dt).days

        # 每個子窗口長度設為擬合區間長度的 65%
        window_days = int(total_days * 0.65)
        step_days = int((total_days - window_days) / 3) if total_days > window_days else 10
        
        WINDOWS = []
        for i in range(4):
            w_start = start_dt + datetime.timedelta(days=i * step_days)
            w_end = w_start + datetime.timedelta(days=window_days)
            if i == 3 or w_end > fit_end_dt:
                w_end = fit_end_dt
            WINDOWS.append((w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))

        if holdout_start:
            print("")
            print(f"  [WF holdout] 擬合區間 {args.start} ~ {fit_end}（四窗只在此區間內搜尋）")
            print(f"               排序區間 {holdout_start} ~ {holdout_end}（{_holdout_days} 天，"
                  f"四窗都沒看過，只用來替可交付向量排序）")
            
        all_best_params = []
        all_best_scores = []
        wf_trials = max(20, args.trials // 4)
        
        for idx, (ws, we) in enumerate(WINDOWS):
            print(f"\n[WF 窗口 {idx+1}/4] 優化區間: {ws} 至 {we} (搜尋 {wf_trials} 輪)...")
            
            def wf_objective(trial):
                trial_params = suggest_trial_params(trial, args.regime)
                score, _, _, _ = run_simulation_scoring(ws, we, trial_params, args.capital, args.max_pos)
                return score
                
            study = optuna.create_study(direction="maximize")
            # 進度回報：study.optimize 整段被 redirect_stdout 導去 devnull（壓掉 Optuna 與回測的雜訊），
            # 所以 callback 必須抓住重導向「之前」的 stdout 才印得出來。flush 是必要的——
            # 輸出被 `>` 重導成檔案時 Python 走塊緩衝，不 flush 會讓 log 長時間停在 0 bytes。
            _real_stdout = sys.stdout

            def _wf_progress(_study, _trial, _idx=idx):
                n = _trial.number + 1
                if not WF_PROGRESS_EVERY or (n % WF_PROGRESS_EVERY and n != wf_trials):
                    return
                try:
                    _bv = f"{_study.best_value:.4f}"
                except ValueError:      # 全部 trial 都失敗時 best_value 會拋例外
                    _bv = "n/a"
                print(f"    ... 窗口 {_idx+1}: {n}/{wf_trials} 輪，目前最佳得分 {_bv}",
                      file=_real_stdout, flush=True)

            with open(os.devnull, "w", encoding="utf-8") as _devnull, contextlib.redirect_stdout(_devnull):
                study.optimize(wf_objective, n_trials=wf_trials, n_jobs=args.jobs,
                               callbacks=[_wf_progress])
            print(f"  [窗口 {idx+1} 完成] 最佳得分: {study.best_value:.4f} | 參數: {study.best_params}")
            all_best_params.append(study.best_params)
            # 各窗最佳分數先前只印在螢幕上、沒有進 JSON，事後無從比對中位數向量是否退步
            all_best_scores.append(float(study.best_value))
            
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
            stability = ("穩定" if cv_val < WF_STABILITY_CV_WARN
                         else "需注意" if cv_val < WF_STABILITY_CV_BAD else "不穩定")
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
            
        # ── 候選向量評分守門 ────────────────────────────────────────────────
        # 逐維取中位數會拼出「沒有任何窗口驗證過」的組合：某窗選的寬鬆 Bull 判定，可能跟
        # 另一窗選的曝險設定焊在一起。實測 2026-08-26 那輪候選在調參區間的目標函式得分
        # -0.4446、全期 MDD 48.88%（同期對照 +0.7120 / 23.13%）——優化器若看過該組合會
        # 直接淘汰，它卻因為 best_score 硬寫 0.0、從未被評分而被當成候選送出。
        # 故在寫檔前把所有可交付向量放到同一個區間上真的各跑一次，取分數最高者。
        # 這不是再一次搜尋（向量都已事先決定），只是把「從未被評分」的漏洞補起來。
        # 該區間預設為上方切出的尾端 holdout（四窗都沒看過）；WF_HOLDOUT_RATIO=0 或區間太短
        # 時退回整段調參區間，那屬 in-sample 選美，排名僅供參考。
        vector_pool = [("中位數", dict(median_params))]
        for _i, _bp in enumerate(all_best_params):
            vector_pool.append((f"窗口{_i+1}最佳", dict(_bp)))

        _unstable = [_p for _p in param_names if wf_results[_p]["stability"] == "不穩定"]
        _dep_params = {}
        if os.path.exists(RESULT_PATH):
            try:
                with open(RESULT_PATH, encoding="utf-8") as _f:
                    _dep_params = json.load(_f).get("best_params", {})
            except Exception:
                _dep_params = {}
        if _unstable and all(_p in _dep_params for _p in _unstable):
            # 保守中位數：CV >= WF_STABILITY_CV_BAD 的維度沿用現行部署值，只讓穩定維度更新。
            # 動機：不穩定維度的窗口間中位數等於取在雜訊上（實例：regime_bull_trend 四窗
            # 0.0015/0.0025/0.0015/0.0010，而它正是把 Bull 天數從 207 撐到 435 的那個參數）。
            # 這同樣是混搭出來的向量，所以不直接採用，一律丟進下面的評分一起比。
            _cons = dict(median_params)
            for _p in _unstable:
                _v = float(_dep_params[_p])
                _cons[_p] = int(_v) if len(TRADING_PARAM_BOUNDS.get(_p, ())) == 2 else _v
            vector_pool.append(("保守中位數", enforce_monotonic(_cons)))

        # 排序區間：有 holdout 就用 holdout（四窗都沒看過），否則退回整段調參區間。
        score_start = holdout_start or args.start
        score_end = holdout_end or args.end
        _zone_label = "尾端 holdout" if holdout_start else "整段調參區間 (in-sample)"

        print("")
        print("=" * 75)
        print(f"  [候選向量評分] {_zone_label} {score_start} ~ {score_end}，共 {len(vector_pool)} 組")
        print("=" * 75)
        print(f"{'向量':<12} | {'目標函式得分':>12} | {'區間報酬':>10} | {'區間MDD':>9}")
        print("-" * 75)
        scored = []
        for _name, _vec in vector_pool:
            _sc, _, _ret, _mdd = run_simulation_scoring(score_start, score_end, _vec,
                                                        args.capital, args.max_pos)
            scored.append((_sc, _name, _vec, _ret, _mdd))
            print(f"{_name:<12} | {_sc:>12.4f} | {_ret:>9.2f}% | {_mdd*100:>8.2f}%", flush=True)

        # ── 挑一組交付 ──────────────────────────────────────────────────────
        # 預設用部署判準（Pareto 前緣 → Calmar），不是 Optuna 目標函式得分：兩者是不同的尺，
        # 後者實測會選到被雙面支配的向量（見 select_by_deploy_gate 的說明與 EXPERIMENTS_PENDING.md）。
        _obj_top = max(scored, key=lambda x: x[0])[1]
        _front = []
        if WF_SELECT_RULE == "deploy_gate":
            best_name, _front = select_by_deploy_gate(scored)
        else:
            best_name = _obj_top
        best_score_wf, _, best_vec, best_ret, best_mdd = next(
            (_s, _n, _v, _r, _m) for _s, _n, _v, _r, _m in scored if _n == best_name)

        print("-" * 75)
        if WF_SELECT_RULE == "deploy_gate":
            _calmar = best_ret / (best_mdd * 100.0) if best_mdd > 1e-11 else float("nan")
            print(f"  [選向量] 規則＝部署判準（Pareto 前緣 → Calmar）；"
                  f"前緣：{'、'.join(_front) if _front else '（全被支配，退回全體）'}")
            print(f"           → 選「{best_name}」：報酬 {best_ret:+.2f}%、回撤 {best_mdd*100:.2f}%、"
                  f"Calmar {_calmar:.2f}（目標函式得分 {best_score_wf:.4f}）")
            if _obj_top != best_name:
                _dominated = _obj_top not in _front
                print(f"           ※ 目標函式得分最高的是「{_obj_top}」"
                      + ("，但它被支配（報酬與回撤同時輸給前緣向量），依部署判準排除。"
                         if _dominated else "，它在前緣內但 Calmar 較低。"))
                print("             兩把尺不一致本身就是缺口#2 記載的病灶，判讀時要記上一筆。")
        else:
            print(f"  [選向量] 規則＝目標函式得分（舊行為）→ 選「{best_name}」，得分 {best_score_wf:.4f}")
        if best_name != "中位數":
            print("           中位數是逐維混搭的向量、未必被任何窗口驗證過，落敗屬正常現象，不是 bug。")
        print("=" * 75)

        # deployed 欄位的語意是「實際寫入參數檔的值」，選到別的向量時必須同步
        for _p in param_names:
            wf_results[_p]["deployed"] = best_vec[_p]

        # 同一個向量池、同一支評分程式碼，另在整段調參區間上再評一次分。
        # 目的**不是**拿來選（選用 holdout），而是把「舊規則（全區間排序）會選誰」直接記進候選檔：
        # 兩份排名共用同一個向量池、零隨機性，是「排序區間」這個單一變因的乾淨對照。
        # 沒有它的話，新舊兩輪候選的差異會混雜三件事（排序區間、四窗範圍縮短、Optuna 無固定種子），
        # 事後無從歸因。順帶提供全段績效，overall_metrics 的語意固定是「date_range 全段」。
        full_scored = {}
        if holdout_start:
            for _name, _vec in vector_pool:
                _fs, _, _fr, _fm = run_simulation_scoring(args.start, args.end, _vec,
                                                         args.capital, args.max_pos)
                full_scored[_name] = {"score": float(_fs), "return_pct": float(_fr),
                                      "mdd_pct": float(_fm * 100)}
            _old_rule_pick = max(full_scored.items(), key=lambda kv: kv[1]["score"])[0]
            full_ret = full_scored[best_name]["return_pct"]
            full_mdd = full_scored[best_name]["mdd_pct"] / 100.0
            print("")
            print(f"  [規則對照] 舊流程（整段調參區間 × 目標函式得分）會選「{_old_rule_pick}」，"
                  f"本輪選「{best_name}」"
                  + ("——兩者相同，本輪的流程改動未改變選擇。" if _old_rule_pick == best_name
                     else "——兩者不同。"))
            print(f"             拆解：換排序區間（整段 → holdout）後目標函式會選「{_obj_top}」；"
                  f"再換排序依據（目標函式 → 部署判準）後選「{best_name}」。")
        else:
            full_ret, full_mdd = best_ret, best_mdd

        output_data = {
            "best_score": float(best_score_wf),
            "optimized_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": f"{args.start} to {args.end}",
            "best_params": best_vec,
            "overall_metrics": {"return_pct": float(full_ret), "mdd_pct": float(full_mdd * 100)},
            "walk_forward_selected": best_name,
            # 排序依據的區間與該區間上的實際績效。best_score 是在此區間上算的，不是全段。
            "walk_forward_score_range": f"{score_start} to {score_end}",
            "walk_forward_holdout": ({"start": holdout_start, "end": holdout_end,
                                      "ratio": WF_HOLDOUT_RATIO,
                                      "selected_return_pct": float(best_ret),
                                      "selected_mdd_pct": float(best_mdd * 100)}
                                     if holdout_start else None),
            "walk_forward_windows": [
                {"start": _ws, "end": _we, "best_score": _sc, "best_params": _bp}
                for (_ws, _we), _sc, _bp in zip(WINDOWS, all_best_scores, all_best_params)
            ],
            "walk_forward_vector_scores": {
                _n: {"score": float(_s), "return_pct": float(_r), "mdd_pct": float(_m * 100)}
                for _s, _n, _, _r, _m in scored
            },
            # 同一向量池在整段調參區間上的評分＝「舊規則會選誰」，供事後歸因（見上方規則對照）
            "walk_forward_vector_scores_full_range": full_scored or None,
            # 選向量的規則、Pareto 前緣，以及兩把尺各自會選誰——歸因用，勿刪
            "walk_forward_selection": {
                "rule": WF_SELECT_RULE,
                "pareto_front": _front or None,
                "objective_top_on_score_range": _obj_top,
                "objective_top_on_full_range": (
                    max(full_scored.items(), key=lambda kv: kv[1]["score"])[0]
                    if full_scored else None),
            },
            "walk_forward_metrics": wf_results
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"\n[成功] Walk-Forward 參數（採用「{best_name}」）已匯出至: {out_path}")
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
                        # enqueue 的值必須是搜尋空間「抽得出來」的點，否則 Optuna 靜默丟棄該值：
                        #   (1) 對齊 step 網格（見 snap_to_bounds）
                        #   (2) 滿足 suggest_trial_params 的單調約束——舊參數檔可能產生於加約束之前
                        #       （實例：best_trading_params_candidate.rejected_20260826.json 的
                        #        bull_pos=2 < bear_pos=3），違反時對應維度的邊界會被收緊而丟棄 enqueue 值。
                        _warm = enforce_monotonic(
                            {k: snap_to_bounds(k, v) for k, v in _filtered.items()})
                        _adj = sorted(k for k in _warm if float(_warm[k]) != float(_filtered[k]))
                        study.enqueue_trial(_warm)
                        print(f"  [暖啟動] 已從 {os.path.basename(_warm_src)} 載入上輪最佳參數作為起點"
                              f"（得分: {_prev.get('best_score', '?')}）")
                        if _adj:
                            print(f"           其中 {len(_adj)} 個參數已調整以落入搜尋空間"
                                  f"（{', '.join(_adj)}）；僅影響搜尋起點，不影響部署值")
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
                _bear_pos = int(best_params.get('regime_bear_pos', REGIME_MAX_POSITIONS['Bear']))
                print(f"       - 持股檔數上限      : Bull {int(best_params['regime_bull_pos'])} | "
                      f"Sideways {int(best_params['regime_sideways_pos'])} | Bear {_bear_pos} 檔(固定)")
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
