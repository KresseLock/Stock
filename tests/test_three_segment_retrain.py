# -*- coding: utf-8 -*-
"""三段切分重訓實驗：模型訓練 | 風控調參 | 乾淨裁判 三段互不重疊（說明見 THREE_SEGMENT_RETRAIN_PLAN.md）。

    模型（含因子最佳化、驗證集） 2017-12 ~ 2024-12-31
    風控參數調參                2025-01-02 ~ 2025-12-31（模型沒看過）
    乾淨裁判區                  2026-01-02 ~ 資料最後一天（模型與參數都沒看過）

流程：
  0. 取得執行鎖（擋 Auto_RUN）＋心跳；快照現行 models/、特徵檔、因子與風控參數
  1. 因子最佳化（optimize_factors，截斷 2024-12-31）       ← --skip-factors 可略過
  2. 以新因子重建特徵檔
  3. 重訓 LightGBM（訓練集 ~2024-06-30，驗證集 2024-07 ~ 2024-12，供 early stopping）
  4. 快照候選（模型＋特徵檔＋因子）
  5. 在候選模型上調風控參數（optimize_trading_params --regime -wf，2025 全年）
  6. 裁判區 2×2 把關（模型 現行/候選 × 參數 舊/新），9 條多起點路徑＋三段 2026 事件
  7. finally：無論成功、失敗或 Ctrl+C，一律還原所有生產檔案、釋放執行鎖

每一步完成即寫入實驗目錄，中斷後以 --exp-dir <目錄> 續跑會跳過已完成的步驟。

用法：
  python tests/test_three_segment_retrain.py --dry-run      # 只印計畫與切分日期，不改任何檔案
  python tests/test_three_segment_retrain.py                # 正式執行（數小時）
  python tests/test_three_segment_retrain.py --exp-dir models/_exp_three_segment_YYYYmmdd_HHMMSS  # 續跑
"""

import argparse
import contextlib
import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402  只讀；切分日期以 module 屬性在本行程內覆寫，不改 config.py
from config import FEATURES_PARQUET, MAX_POSITIONS  # noqa: E402
from scripts.validate_candidate_params import OFFSETS, _median  # noqa: E402  與既有把關同一組路徑
from scripts.utils import (start_tee_log, model_fingerprint, acquire_run_lock,  # noqa: E402
                           release_run_lock, start_run_lock_heartbeat)

# ── 三段切分（本實驗的定義；改這裡就是另一個實驗）────────────────────
MODEL_CUTOFF   = "20241231"     # 因子最佳化與訓練資料截斷日（BACKTEST_DATE）
TRAIN_END_FROM = "2024-07-01"   # 訓練集結束點＝此日起的第一個交易日；其後到截斷日為驗證集
PARAM_START    = "2025-01-02"   # 風控調參區間
PARAM_END      = "2025-12-31"
ZONE_FROM      = "2026-01-02"   # 裁判區起點（模型與新參數都沒看過）
EVENT_WINDOWS = [               # 2026 三次下跌，僅供觀察系統反應，不參與判準（§4.5 #5）
    ("2026 1~3月", "2026-01-02", "2026-03-13"),
    ("2026 7月股災", "2026-06-15", "2026-08-31"),
    ("2026 10月下殺", "2026-09-15", "2026-10-30"),
]

MODEL_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
MODEL_FILES = ("lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json")
FACTORS = os.path.join(CONFIG_DIR, "best_factors.json")
PARAMS = os.path.join(CONFIG_DIR, "best_trading_params.json")
PARAM_CANDIDATE = os.path.join(CONFIG_DIR, "best_trading_params_candidate.json")
# 「舊參數」軸：mode_a 調參至 2025-08-01，沒看過裁判區，是乾淨的對照。
# 生產 best_trading_params.json 調參至 2026-07-23、看過大半個裁判區，只列為參考格，不進判準。
PARAMS_CLEAN_OLD = os.path.join(CONFIG_DIR, "best_trading_params_mode_a.json")
# 執行期間會被覆寫、結束時必須還原到原狀的生產檔案（不存在者結束時刪除）
GUARDED = [FACTORS, FACTORS + ".bak", PARAMS, PARAM_CANDIDATE, FEATURES_PARQUET] + \
          [os.path.join(MODEL_DIR, n) for n in MODEL_FILES]


# ══════════════════════════════════════════════════════════════════════
# 檔案快照／還原
# ══════════════════════════════════════════════════════════════════════
def _snap_name(path):
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "__")


def snapshot_guarded(dst):
    """把 GUARDED 現況存進 dst，並記錄哪些原本不存在。"""
    os.makedirs(dst, exist_ok=True)
    missing = []
    for p in GUARDED:
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, _snap_name(p)))
        else:
            missing.append(_snap_name(p))
    with open(os.path.join(dst, "_missing.json"), "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=1)
    with open(os.path.join(dst, "_DONE"), "w") as f:
        f.write(model_fingerprint())


def restore_guarded(src):
    with open(os.path.join(src, "_missing.json"), encoding="utf-8") as f:
        missing = set(json.load(f))
    for p in GUARDED:
        name = _snap_name(p)
        if name in missing:
            if os.path.exists(p):
                os.remove(p)
        else:
            shutil.copy2(os.path.join(src, name), p)


def save_set(dst, params_src=None):
    """存一組「模型＋特徵檔＋因子」（＋選配風控參數）。"""
    os.makedirs(dst, exist_ok=True)
    for n in MODEL_FILES:
        shutil.copy2(os.path.join(MODEL_DIR, n), dst)
    shutil.copy2(FEATURES_PARQUET, os.path.join(dst, "features_combined.parquet"))
    shutil.copy2(FACTORS, os.path.join(dst, "best_factors.json"))
    if params_src:
        shutil.copy2(params_src, os.path.join(dst, "best_trading_params.json"))
    with open(os.path.join(dst, "_DONE"), "w") as f:
        f.write(model_fingerprint())


def install_set(src):
    """模型、特徵檔、因子三者必須一起換：因子不同 → 特徵欄名不同 → 只換模型會被 reindex 靜默補 NaN。"""
    for n in MODEL_FILES:
        shutil.copy2(os.path.join(src, n), os.path.join(MODEL_DIR, n))
    shutil.copy2(os.path.join(src, "features_combined.parquet"), FEATURES_PARQUET)
    shutil.copy2(os.path.join(src, "best_factors.json"), FACTORS)


def _rel(p):
    try:
        return os.path.relpath(p, BASE_DIR)
    except ValueError:      # 不同磁碟機
        return p


def done(d):
    return os.path.exists(os.path.join(d, "_DONE"))


# ══════════════════════════════════════════════════════════════════════
# 切分比例：反算 TRAIN/VALID_SPLIT_RATIO，讓 train.py 的日期分位切在指定日期
# ══════════════════════════════════════════════════════════════════════
def compute_split_ratios():
    """複製 train.py main() 的前處理（截斷 → 產業過濾 → dropna 標籤）取得同一組 unique dates。"""
    import pandas as pd
    from scripts.utils import filter_stocks_by_train_industries
    labels = ["label_1", "label_2", "label_3"]
    df = pd.read_parquet(FEATURES_PARQUET, columns=["stock_id", "date"] + labels)
    df = df[pd.to_datetime(df["date"]) <= pd.to_datetime(MODEL_CUTOFF)]
    df = filter_stocks_by_train_industries(df).dropna(subset=labels)
    dates = sorted(pd.to_datetime(df["date"].unique()))
    n = len(dates)
    k = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TRAIN_END_FROM))
    # +0.5 / -0.5 讓 int(n*ratio) 精確落在 k 與 n-1，避免浮點誤差差一格
    tr, va = (k + 0.5) / n, (n - 0.5) / n
    assert dates[int(n * tr)] == dates[k] and int(n * va) == n - 1
    return tr, va, {
        "訓練集": f"{dates[0].date()} ~ {dates[k - 1].date()}",
        "驗證集": f"{dates[k].date()} ~ {dates[n - 2].date()}",
        "交易日數": n,
    }


# ══════════════════════════════════════════════════════════════════════
# 回測（子行程：config.py 在 import 時讀 best_trading_params.json，換參數必須另起行程）
# ══════════════════════════════════════════════════════════════════════
_RUNNER = r"""
import sys, io, json, contextlib
sys.path.insert(0, sys.argv[1])
import trading_sim as ts
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = ts.run_simulation(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), export_report=False)
if not res:
    print("@@RESULT@@" + json.dumps({"error": buf.getvalue()[-800:]}))
else:
    ret, _, hist = res
    eq = [h["equity"] for h in hist]
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v); mdd = min(mdd, v / peak - 1)
    exp = sum(h["invested"] / h["equity"] for h in hist) / len(hist)
    print("@@RESULT@@" + json.dumps({"ret": ret, "mdd": mdd * 100, "exp": exp * 100}))
"""


def run_backtest(start, end, capital):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-c", _RUNNER, BASE_DIR, start, end, str(capital), str(MAX_POSITIONS)],
                          cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = proc.stdout.decode("utf-8", errors="replace")
    line = next((l for l in out.splitlines() if l.startswith("@@RESULT@@")), None)
    if line is None:
        raise RuntimeError(f"回測失敗（exit={proc.returncode}）：\n" + "\n".join(out.splitlines()[-15:]))
    r = json.loads(line[len("@@RESULT@@"):])
    if "error" in r:
        raise RuntimeError("回測無結果：\n" + r["error"])
    return r


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════
def banner(t):
    print("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78, flush=True)


def phase_build_candidate(exp, args):
    """步驟 1~4：因子最佳化 → 特徵 → 訓練 → 快照候選。全部在本行程，以 module 屬性覆寫截斷日。"""
    config.BACKTEST_DATE = MODEL_CUTOFF       # optimize_factors 的防洩漏保護讀這個
    import auto_pipeline as ap                # 其內 step 函式自帶 feature_engineering 參數套用

    if args.skip_factors:
        banner("步驟 1/7：略過因子最佳化（--skip-factors，沿用現行 best_factors.json）")
        print("  ⚠ 現行因子是截斷 2025-08-01 搜出來的，看過調參區 2025-01~08；裁判區 2026 仍乾淨。")
    else:
        banner(f"步驟 1/7：因子最佳化（截斷 {MODEL_CUTOFF}，{args.factor_trials} 輪）")
        ap.OPTIMIZATION_TRIALS = args.factor_trials   # step1_optimize 以此設定 Optuna 輪數
        ap.step2_load_params()
        ap.step3_feature_engineering()        # 先以現行因子建好基礎特徵供 Optuna 讀取
        ap.step1_optimize(MODEL_CUTOFF)       # 寫入 configs/best_factors.json（結束時還原）

    banner("步驟 2/7：以（新）因子重建特徵檔")
    ap.step2_load_params()
    ap.step3_feature_engineering()

    banner("步驟 3/7：重訓 LightGBM")
    tr, va, info = compute_split_ratios()
    for k, v in info.items():
        print(f"  {k}: {v}")
    config.TRAIN_SPLIT_RATIO, config.VALID_SPLIT_RATIO = tr, va
    import scripts.train as train_module      # 必須在覆寫 config 之後才 import（train 以 from config import 取值）
    assert train_module.BACKTEST_DATE == MODEL_CUTOFF and train_module.TRAIN_SPLIT_RATIO == tr
    train_module.N_JOBS = config.TRAIN_N_JOBS
    train_module.main()

    banner("步驟 4/7：快照候選模型")
    save_set(os.path.join(exp, "candidate"))
    with open(os.path.join(exp, "candidate", "split.json"), "w", encoding="utf-8") as f:
        json.dump({"MODEL_CUTOFF": MODEL_CUTOFF, "TRAIN_SPLIT_RATIO": tr, "VALID_SPLIT_RATIO": va,
                   "skip_factors": args.skip_factors, **info}, f, ensure_ascii=False, indent=1)
    print(f"  候選模型指紋: {model_fingerprint()}")


def phase_tune_params(exp, args):
    banner(f"步驟 5/7：在候選模型上調風控參數（{PARAM_START} ~ {PARAM_END}）")
    cand = os.path.join(exp, "candidate")
    install_set(cand)
    # 調參時的基底參數用乾淨的 mode_a，避免未被搜尋的鍵沿用看過裁判區的生產參數
    shutil.copy2(PARAMS_CLEAN_OLD, PARAMS)
    cmd = [sys.executable, os.path.join("scripts", "optimize_trading_params.py"),
           "--regime", "-wf", "-j", str(args.jobs), "-c", str(args.capital),
           "-t", str(args.param_trials), "-s", PARAM_START, "-e", PARAM_END]
    print("  " + " ".join(cmd), flush=True)
    if os.path.exists(PARAM_CANDIDATE):
        os.remove(PARAM_CANDIDATE)            # 確保讀到的是本輪產出（原檔已在快照內）
    subprocess.run(cmd, cwd=BASE_DIR, check=True)
    if not os.path.exists(PARAM_CANDIDATE):
        raise RuntimeError("optimize_trading_params 沒有產出 best_trading_params_candidate.json")
    shutil.copy2(PARAM_CANDIDATE, os.path.join(cand, "best_trading_params.json"))


def phase_gate(exp, args):
    banner("步驟 6/7：裁判區 2×2 把關")
    import pandas as pd
    inc, cand = os.path.join(exp, "incumbent"), os.path.join(exp, "candidate")
    models = {"現行模型": None, "候選模型": cand}
    params = {"舊參數": PARAMS_CLEAN_OLD, "新參數": os.path.join(cand, "best_trading_params.json")}
    cells = [(m, p) for m in models for p in params] + [("現行模型", "生產參數")]
    params["生產參數"] = os.path.join(inc, _snap_name(PARAMS))

    days = sorted(str(d)[:10] for d in pd.read_parquet(os.path.join(cand, "features_combined.parquet"),
                                                       columns=["date"])["date"].unique())
    days = [d for d in days if d >= ZONE_FROM]
    zone_end = days[-1]
    paths = [(days[o], zone_end) for o in OFFSETS]
    print(f"  裁判區 {days[0]} ~ {zone_end}（{len(days)} 交易日），{len(paths)} 條路徑，終點固定")

    res_path = os.path.join(exp, "gate_runs.json")
    runs = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
    jobs = [(c, s, e) for c in cells for (s, e) in paths] + \
           [(c, s, e) for c in cells[:4] for (_, s, e) in EVENT_WINDOWS]
    cur_model = None
    for i, (cell, s, e) in enumerate(jobs, 1):
        key = f"{cell[0]}|{cell[1]}|{s}|{e}"
        if key in runs:
            continue
        if cur_model != cell[0]:
            if cell[0] == "現行模型":
                for n in MODEL_FILES:
                    shutil.copy2(os.path.join(inc, _snap_name(os.path.join(MODEL_DIR, n))), os.path.join(MODEL_DIR, n))
                shutil.copy2(os.path.join(inc, _snap_name(FEATURES_PARQUET)), FEATURES_PARQUET)
                shutil.copy2(os.path.join(inc, _snap_name(FACTORS)), FACTORS)
            else:
                install_set(cand)
            cur_model = cell[0]
        shutil.copy2(params[cell[1]], PARAMS)
        t0 = time.time()
        r = run_backtest(s, e, args.capital)
        runs[key] = r
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False, indent=1)
        print(f"  [{i}/{len(jobs)}] {cell[0]}×{cell[1]}  {s}~{e}  報酬 {r['ret']:+8.2f}%  "
              f"回撤 {r['mdd']:7.2f}%  曝險 {r['exp']:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    return runs, cells, paths, zone_end


def report(exp, runs, cells, paths, zone_end):
    banner("步驟 7/7：結果")
    get = lambda c, s, e: runs[f"{c[0]}|{c[1]}|{s}|{e}"]
    med = {c: (_median([get(c, s, e)["ret"] for s, e in paths]),
               _median([get(c, s, e)["mdd"] for s, e in paths])) for c in cells}
    L = []
    L.append(f"# 三段切分重訓實驗結果\n\n實驗目錄：`{_rel(exp)}`  ")
    L.append(f"裁判區：{paths[0][0]} ~ {zone_end}，{len(paths)} 條路徑（起點後挪 {'/'.join(map(str, OFFSETS))} 交易日、終點固定）\n")
    sp = json.load(open(os.path.join(exp, "candidate", "split.json"), encoding="utf-8"))
    L.append(f"候選切分：訓練集 {sp['訓練集']}、驗證集 {sp['驗證集']}、因子最佳化 {'略過' if sp['skip_factors'] else '重跑'}\n")
    L.append("## 2×2 中位數（報酬% / 回撤%，回撤越接近 0 越好）\n")
    L.append("| | 舊參數（mode_a，乾淨） | 新參數（本輪） |\n|---|---|---|")
    for m in ("現行模型", "候選模型"):
        L.append(f"| {m} | " + " | ".join(f"{med[(m, p)][0]:+.2f}% / {med[(m, p)][1]:.2f}%" for p in ("舊參數", "新參數")) + " |")
    r = med[("現行模型", "生產參數")]
    L.append(f"\n參考（不進判準，生產參數看過 2026-01~07）：現行模型 × 生產參數 {r[0]:+.2f}% / {r[1]:.2f}%\n")

    def cmp(a, b):
        both = sum(get(b, s, e)["ret"] > get(a, s, e)["ret"] and get(b, s, e)["mdd"] > get(a, s, e)["mdd"]
                   for s, e in paths)
        return med[b][0] - med[a][0], med[b][1] - med[a][1], both
    L.append("## 歸因（固定一軸、只動另一軸）\n\n| 比較 | 報酬差 | 回撤差 | 逐路徑雙贏 |\n|---|---|---|---|")
    A, B, C, D = ("現行模型", "舊參數"), ("現行模型", "新參數"), ("候選模型", "舊參數"), ("候選模型", "新參數")
    for name, a, b in [("新參數的效果（候選模型上）", C, D), ("新參數的效果（現行模型上）", A, B),
                       ("新模型的效果（舊參數下）", A, C), ("新模型的效果（新參數下）", B, D),
                       ("★ 部署決策：候選×新 vs 現行×舊", A, D)]:
        dr, dm, both = cmp(a, b)
        L.append(f"| {name} | {dr:+.2f}pp | {dm:+.2f}pp | {both}/{len(paths)} |")
    dr, dm, both = cmp(A, D)
    passed = dr > 0 and dm > 0 and both >= 5
    L.append("\n**判準**：報酬與回撤中位數雙贏，且逐路徑雙贏 ≥ 5/9（§4.5 #4）。")
    L.append(f"**結果**：{'✅ 通過' if passed else '❌ 未通過'}\n")
    L.append("## 2026 事件段（僅觀察，不進判準）\n\n| 事件 | " + " | ".join(f"{m}×{p}" for m, p in cells[:4]) + " |\n|---|" + "---|" * 4)
    for name, s, e in EVENT_WINDOWS:
        L.append(f"| {name} {s}~{e} | " + " | ".join(
            f"{get(c, s, e)['ret']:+.1f}% / {get(c, s, e)['mdd']:.1f}%" for c in cells[:4]) + " |")
    L.append("\n## 逐路徑明細\n\n| 起點 | " + " | ".join(f"{m}×{p}" for m, p in cells) + " |\n|---|" + "---|" * len(cells))
    for s, e in paths:
        L.append(f"| {s} | " + " | ".join(f"{get(c, s, e)['ret']:+.1f} / {get(c, s, e)['mdd']:.1f}" for c in cells) + " |")
    text = "\n".join(L)
    with open(os.path.join(exp, "RESULT.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  結果已寫入 {_rel(os.path.join(exp, 'RESULT.md'))}")


def main():
    ap = argparse.ArgumentParser(description="三段切分重訓實驗（訓練 | 調參 | 裁判 互不重疊）")
    ap.add_argument("--exp-dir", help="續跑既有實驗目錄（跳過已完成步驟）")
    ap.add_argument("--skip-factors", action="store_true", help="略過因子最佳化，沿用現行 best_factors.json")
    ap.add_argument("--factor-trials", type=int, default=config.OPTIMIZATION_TRIALS)
    ap.add_argument("--param-trials", type=int, default=config.OPTIMIZATION_TRIALS)
    ap.add_argument("-j", "--jobs", type=int, default=4, help="風控調參並行數")
    ap.add_argument("-c", "--capital", type=int, default=2000000)
    ap.add_argument("--dry-run", action="store_true", help="只印計畫與切分日期，不改任何檔案")
    args = ap.parse_args()

    if args.dry_run:
        print("[dry-run] 不取鎖、不改檔。")
        print(f"  模型截斷 {MODEL_CUTOFF}｜調參 {PARAM_START}~{PARAM_END}｜裁判區 {ZONE_FROM}~")
        print(f"  因子最佳化: {'略過' if args.skip_factors else f'{args.factor_trials} 輪'}｜風控調參: {args.param_trials} 輪, -j {args.jobs}")
        print(f"  現行模型指紋: {model_fingerprint()}")
        for p in [PARAMS_CLEAN_OLD, PARAMS]:
            d = json.load(open(p, encoding="utf-8"))
            print(f"  {os.path.basename(p)}: 調參區間 {d.get('date_range')}")
        tr, va, info = compute_split_ratios()
        print(f"  依「現行」特徵檔預估切分（因子重跑後交易日不變，切點相同）: {info}, ratio={tr:.6f}/{va:.6f}")
        return 0

    exp = os.path.abspath(args.exp_dir) if args.exp_dir else \
        os.path.join(MODEL_DIR, f"_exp_three_segment_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(exp, exist_ok=True)
    log = start_tee_log("test_three_segment_retrain")
    print(f"  實驗目錄: {_rel(exp)}\n  執行記錄: {os.path.relpath(log, BASE_DIR)}")

    if not acquire_run_lock("test_three_segment_retrain"):
        return 2
    start_run_lock_heartbeat()                # 全程可能超過 6 小時的鎖殘留判定門檻
    inc = os.path.join(exp, "incumbent")
    try:
        if not done(inc):                     # 續跑時絕不重新快照（此時現況可能已是候選）
            banner("步驟 0/7：快照現行生產檔案")
            snapshot_guarded(inc)
        print(f"  現行模型指紋: {open(os.path.join(inc, '_DONE')).read()}")

        if not done(os.path.join(exp, "candidate")):
            phase_build_candidate(exp, args)
        else:
            print("  [續跑] 候選模型已存在，略過步驟 1~4")
        if not os.path.exists(os.path.join(exp, "candidate", "best_trading_params.json")):
            phase_tune_params(exp, args)
        else:
            print("  [續跑] 候選風控參數已存在，略過步驟 5")
        report(exp, *phase_gate(exp, args))
        return 0
    finally:
        if done(inc):                         # 快照未完成就中斷時，生產檔案尚未被動過，不可拿殘缺快照還原
            restore_guarded(inc)
            print(f"\n[還原] 生產檔案已放回實驗前狀態；models/ 指紋 {model_fingerprint()}（應等於上方現行指紋）")
        release_run_lock()


if __name__ == "__main__":
    sys.exit(main())
