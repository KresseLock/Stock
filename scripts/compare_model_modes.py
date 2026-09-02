"""
模型時效性對照實驗
======================================================================

目的：驗證「近期資料沒被訓練進去」是否為近月回測績效不佳的主因。

背景（實測事實，2026-08-13）：
  * config.BACKTEST_DATE 只決定「資料截斷點」，真正 fit 的訓練集終點由
    train.py 的日期分位切分 (config.TRAIN_SPLIT_RATIO / VALID_SPLIT_RATIO) 決定。
  * 現行生產模型：資料截至 2025-08-01，但實際 fit 只到 ~2023-11。
  * 只把 BACKTEST_DATE 設 None：fit 終點僅推進到 ~2024-08，最近一年仍進不了訓練集。

候選（同一回測區間、同一組風控參數）：
  incumbent  現行 models/ 內的模型（不重訓，直接回測）
  b1         BACKTEST_DATE=None，切分比例維持 config 原值           ── 只重訓模型
  b2         BACKTEST_DATE=回測起始日前一日 + 高比例切分            ── 只重訓模型
  b2full     同 b2 的截斷設定，但完整重跑 optimize→feature→train    ── 含貝葉斯因子調參
             （b1/b2 沿用現行 best_factors.json，b2full 會重新搜尋因子）

為何沒有 b1full：optimize_factors.py 的防洩漏保護看 config.BACKTEST_DATE，
設 None 時不截斷，會把回測區間納入因子搜尋目標（lookahead）。故完整重訓
只能用「截斷於回測起始日前一日」的設定，None 型無法做乾淨對照。

安全機制：
  * 開跑前把 models/ 快照到 models/_backup_incumbent_<TS>/；full 模式另備份
    configs/best_factors.json 與 data/features/features_combined.parquet
  * 備份資訊寫入 models/_restore_manifest.json；正常結束才刪除。若行程被強制
    中止（斷電／關機）導致殘留，下次啟動會擋下並要求先 --restore-latest
  * 候選訓練成果保留在 models/_candidate_<tag>/（full 模式含 best_factors 與
    特徵檔），比較結束不會刪除 —— 結果較好時 --promote 直接套用，不必重跑
  * 無論成功、失敗或 Ctrl+C，finally 一律還原 config.py 變數、models/ 與資產
  * 全程輸出同時寫入 reports/compare_model_modes_<TS>.log

用法：
  python scripts/compare_model_modes.py                    # 只重訓模型的三方比較（數分鐘）
  python scripts/compare_model_modes.py --full             # 含貝葉斯因子調參（數小時，建議夜間跑）
  python scripts/compare_model_modes.py --full --factor-trials 200   # 降 trials 快速看方向
  python scripts/compare_model_modes.py --only b1 b2 b2full          # 自選候選組合
  python scripts/compare_model_modes.py --reuse-candidate  # 沿用已保留的候選模型，不重訓
  python scripts/compare_model_modes.py --promote b2full   # 採用候選（含特徵檔與因子參數）
  python scripts/compare_model_modes.py --restore-latest   # 緊急還原（行程被強制中止時）
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CONFIG_PATH = os.path.join(BASE_DIR, "config.py")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
BEST_FACTORS_PATH = os.path.join(BASE_DIR, "configs", "best_factors.json")
from config import FEATURES_PARQUET as FEATURE_PATH   # 唯一來源：config.py § 0

SELF_REL_PATH = os.path.relpath(os.path.abspath(__file__), BASE_DIR).replace(os.sep, "/")
MODEL_FILES = ("lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json")
# full 模式額外備份的資產：候選名 → 專案內實際路徑
ASSET_PATHS = {"best_factors.json": BEST_FACTORS_PATH, "features_combined.parquet": FEATURE_PATH}
CONFIG_VARS = ("BACKTEST_DATE", "TRAIN_SPLIT_RATIO", "VALID_SPLIT_RATIO", "OPTIMIZATION_TRIALS",
               "APPLY_BEST_FACTORS_TA")
BACKUP_PREFIX = "_backup_incumbent_"
CANDIDATE_PREFIX = "_candidate_"
MANIFEST_PATH = os.path.join(MODEL_DIR, "_restore_manifest.json")

# 本次比較窗（使用者指定；非策略常數，故以 CLI 預設值形式留在實驗腳本內）
DEFAULT_START = "2026-04-01"
DEFAULT_END = "2026-07-31"
# b2／b2full 的高比例切分：train 盡量往截斷點推，仍保留 valid 供 early stopping、test 供評估
B2_TRAIN_SPLIT = 0.95
B2_VALID_SPLIT = 0.98


# ── 輸出同時落檔 ────────────────────────────────────────────────────────
class _Tee:
    """stdout 分流到終端與 log 檔（長時間執行必備，事後可回溯）"""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")

    def write(self, s):
        self.terminal.write(s)
        self.terminal.flush()  # 使用者若再用 > 重導向，不加這行會看不到即時進度
        self.log.write(s)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ── config.py 讀寫（與 run_workflow_experiment.update_config_var 同法） ──────
def read_config_var(name):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(rf"^(\s*{name}\s*=\s*)([^\n#]+)", content, flags=re.MULTILINE)
    return m.group(2).strip() if m else None


def write_config_var(name, value_str):
    """替換變數值，保留值與行尾註解之間的空白（否則反覆改寫會把註解黏上來）"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    new_content, count = re.subn(
        rf"^(\s*{name}\s*=\s*)([^\n#]*?)([ \t]*)(#[^\n]*)?$",
        lambda m: f"{m.group(1)}{value_str}{m.group(3)}{m.group(4) or ''}",
        content,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError(f"無法更新 config.py 中的 {name}（請確認該變數存在且未被註解）")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  [Config變更] {name} = {value_str}")


# ── 模型與資產的快照／還原 ──────────────────────────────────────────────
def snapshot(dst_dir, with_assets=False):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in MODEL_FILES:
        src = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(src):
            raise FileNotFoundError(f"找不到模型檔: {src}")
        shutil.copy2(src, os.path.join(dst_dir, fn))
    if with_assets:
        for name, src in ASSET_PATHS.items():
            if os.path.exists(src):
                t0 = time.time()
                shutil.copy2(src, os.path.join(dst_dir, name))
                print(f"  [備份] {name} ({os.path.getsize(src) / 2**20:.0f} MB, {time.time() - t0:.1f}s)")
    return dst_dir


def restore(src_dir, with_assets=True):
    """還原模型；with_assets=True 時，來源目錄內若存在資產副本也一併還原"""
    for fn in MODEL_FILES:
        src = os.path.join(src_dir, fn)
        if not os.path.exists(src):
            raise FileNotFoundError(f"備份缺少 {fn}: {src}")
        shutil.copy2(src, os.path.join(MODEL_DIR, fn))
    restored = []
    if with_assets:
        for name, dst in ASSET_PATHS.items():
            src = os.path.join(src_dir, name)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(name)
    return restored


def latest_backup_dir():
    if not os.path.isdir(MODEL_DIR):
        return None
    dirs = sorted(d for d in os.listdir(MODEL_DIR) if d.startswith(BACKUP_PREFIX))
    return os.path.join(MODEL_DIR, dirs[-1]) if dirs else None


def candidate_dir(tag):
    return os.path.join(MODEL_DIR, f"{CANDIDATE_PREFIX}{tag}")


def write_manifest(backup_dir, orig_cfg):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now().isoformat(timespec="seconds"),
                   "backup_dir": backup_dir, "config": orig_cfg}, f, ensure_ascii=False, indent=2)


def clear_manifest():
    if os.path.exists(MANIFEST_PATH):
        os.remove(MANIFEST_PATH)


# ── 流水線步驟（子行程執行，確保讀到改寫後的 config.py） ──────────────────
def run_pipeline_step(step):
    cmd = [sys.executable, os.path.join(BASE_DIR, "auto_pipeline.py"), "-s", step]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    print(f"\n  [執行] auto_pipeline.py -s {step}    ({datetime.now():%H:%M:%S} 起)")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"步驟 {step} 失敗（exit code {proc.returncode}），請看上方輸出")
    print(f"  [完成] {step} 耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    return lines


def parse_fit_range(train_log):
    """自 train.py 輸出擷取實際 fit 的訓練集／驗證集期間"""
    out = {}
    for key, label in (("train", "訓練集"), ("valid", "驗證集")):
        for line in train_log:
            m = re.search(rf"{label}:.*?\((\d{{4}}-\d{{2}}-\d{{2}}) ~ (\d{{4}}-\d{{2}}-\d{{2}})\)", line)
            if m:
                out[key] = f"{m.group(1)} ~ {m.group(2)}"
                break
    return out


def estimate_fit_range(backtest_date, train_ratio, valid_ratio):
    """以特徵檔日期推算切分邊界（供 incumbent 這種無訓練 log 的情況參考）"""
    try:
        df = pd.read_parquet(FEATURE_PATH, columns=["date", "label_1"]).dropna(subset=["label_1"])
    except Exception as e:
        return {"train": f"（推算失敗：{e}）"}
    d = pd.to_datetime(df["date"])
    if backtest_date:
        d = d[d <= pd.to_datetime(backtest_date)]
    u = sorted(d.unique())
    n = len(u)
    if n == 0:
        return {"train": "（無資料）"}
    tr_end = pd.Timestamp(u[min(int(n * train_ratio), n - 1)])
    va_end = pd.Timestamp(u[min(int(n * valid_ratio), n - 1)])
    return {
        "train": f"{pd.Timestamp(u[0]).date()} ~ {tr_end.date()}（推算）",
        "valid": f"{tr_end.date()} ~ {va_end.date()}（推算）",
    }


def read_factor_params():
    """讀取目前生效的技術指標參數，供報告對照"""
    if not os.path.exists(BEST_FACTORS_PATH):
        return {}
    try:
        with open(BEST_FACTORS_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        return {"optimized_at": j.get("optimized_at"), "backtest_date": j.get("backtest_date"),
                "iterations": j.get("iterations"),
                "params": j.get("best_params_for_run_feature_engineering", {})}
    except Exception:
        return {}


# ── 回測 ────────────────────────────────────────────────────────────────
def run_backtest(tag, start, end, capital, max_pos):
    import trading_sim  # 延遲載入；run_simulation 內部每次都重讀 models/，故換模型後可重複呼叫

    total_return, max_dd, _ = trading_sim.run_simulation(
        start, end, capital, max_pos, export_report=True
    )
    src = os.path.join(REPORT_DIR, f"backtest_report_{start}_{end}.xlsx")
    dst = os.path.join(REPORT_DIR, f"backtest_report_{start}_{end}__{tag}.xlsx")
    if os.path.exists(src):
        os.replace(src, dst)
    else:
        dst = None
    res = {"tag": tag, "total_return": total_return, "max_dd": max_dd * 100, "report": dst}
    res.update(summarize_trades(dst))
    return res


def summarize_trades(xlsx_path):
    empty = {"n_buy": 0, "n_sell": 0, "win_rate": None, "avg_pnl": None}
    if not xlsx_path or not os.path.exists(xlsx_path):
        return empty
    try:
        th = pd.read_excel(xlsx_path, sheet_name="Trade_History")
    except Exception:
        return empty
    if th.empty or "操作" not in th.columns:
        return empty
    op = th["操作"].astype(str)
    sell = th[op.str.contains("賣")]
    pnl = sell["利潤率(%)"] if "利潤率(%)" in sell.columns else pd.Series(dtype=float)
    return {
        "n_buy": int(op.str.contains("買").sum()),
        "n_sell": int(len(sell)),
        "win_rate": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "avg_pnl": float(pnl.mean()) if len(pnl) else None,
    }


# ── 報告 ────────────────────────────────────────────────────────────────
def _f(v, spec="+.2f"):
    return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"


def calmar(ret, mdd):
    if not isinstance(ret, (int, float)) or not isinstance(mdd, (int, float)) or mdd <= 0:
        return None
    return ret / mdd


def build_report(results, args, ts, orig_cfg, log_path):
    inc = next((r for r in results if r["tag"] == "incumbent"), None)
    has_full = any(r.get("full_pipeline") for r in results)
    lines = [
        "# 模型時效性對照報告",
        "",
        f"- 產生時間：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 回測區間：`{args.start}` ~ `{args.end}`（連續區間，非按月切割）",
        f"- 初始資金：{args.capital:,} ｜ 最大持股：{args.max_pos} 檔",
        "- 風控參數：完全沿用 config.py 現值（含 regime 動態門檻），各組唯一變因為模型／因子參數",
        "- config 原始值（實驗結束已還原）：" + "、".join(f"`{k}={v}`" for k, v in orig_cfg.items()),
        f"- 完整執行 log：`{os.path.relpath(log_path, BASE_DIR)}`",
        "",
        "## 一、候選定義",
        "",
        "| 代號 | 說明 | 資料截斷 | 切分比例 | 因子調參 | 實際 fit 訓練集 | early stopping 驗證集 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['tag']}` | {r.get('name', '')} | `{r.get('cfg_backtest_date', '-')}` | "
            f"`{r.get('cfg_split', '-')}` | {r.get('factor_note', '沿用現行')} | "
            f"{r.get('fit', {}).get('train', 'N/A')} | {r.get('fit', {}).get('valid', 'N/A')} |"
        )

    lines += [
        "",
        "## 二、績效比較",
        "",
        "| 指標 | " + " | ".join(f"`{r['tag']}`" for r in results) + " |",
        "|---|" + "---|" * len(results),
        "| 區間報酬 (%) | " + " | ".join(_f(r["total_return"]) for r in results) + " |",
        "| 最大回撤 MDD (%) | " + " | ".join(_f(-r["max_dd"]) for r in results) + " |",
        "| Calmar (報酬/MDD) | " + " | ".join(_f(calmar(r["total_return"], r["max_dd"]), ".2f") for r in results) + " |",
        "| 買進筆數 | " + " | ".join(str(r["n_buy"]) for r in results) + " |",
        "| 賣出筆數 | " + " | ".join(str(r["n_sell"]) for r in results) + " |",
        "| 賣出勝率 (%) | " + " | ".join(_f(r["win_rate"], ".0f") for r in results) + " |",
        "| 平均每筆損益 (%) | " + " | ".join(_f(r["avg_pnl"]) for r in results) + " |",
        "",
    ]

    if inc:
        lines += ["## 三、相對現行模型的增減", "", "| 候選 | 報酬差 (pp) | MDD 差 (pp，負=回撤改善) | 判定 |", "|---|---|---|---|"]
        for r in results:
            if r["tag"] == "incumbent":
                continue
            d_ret = r["total_return"] - inc["total_return"]
            d_mdd = r["max_dd"] - inc["max_dd"]
            if d_ret > 0 and d_mdd <= 0:
                verdict = "✅ 雙贏，建議採用"
            elif d_ret > 0:
                verdict = "⚠️ 報酬升但回撤變大，屬取捨"
            elif d_ret <= 0 and d_mdd <= 0:
                verdict = "⚠️ 回撤改善但報酬變差，屬取捨"
            else:
                verdict = "❌ 雙輸，不建議採用"
            lines.append(f"| `{r['tag']}` | {d_ret:+.2f} | {d_mdd:+.2f} | {verdict} |")
        lines.append("")

    for r in results:
        if r.get("factor_params"):
            lines += [f"### `{r['tag']}` 重新搜尋出的技術指標參數", "", "```json",
                      json.dumps(r["factor_params"], ensure_ascii=False, indent=2), "```", ""]

    lines += [
        "## 四、判讀注意事項",
        "",
        "1. **採用判準採雙贏制**（呼應 CLAUDE.md §4.5 方法論）：報酬與回撤須同時不劣於現行，"
        "單看報酬改善不算數。",
        f"2. **樣本量極小**：本區間僅 {(pd.to_datetime(args.end) - pd.to_datetime(args.start)).days} 天、"
        f"每組僅 {max((r['n_sell'] for r in results), default=0)} 筆以內的平倉交易，單次結果的噪音很大；"
        "任何候選在此勝出都只能視為「值得進一步用完整 OOS 驗證」，不足以直接下結論。",
        "3. **b1 無 lookahead**：訓練資料涵蓋期止於 valid 尾端，回測區間落在 test 段，未參與訓練。",
        f"4. **b2／b2full 無 lookahead**：資料截斷點設在回測起始日前一日 (`{args.b2_cutoff}`)，"
        "訓練、early stopping 與因子搜尋都看不到回測區間。",
        "5. **高比例切分的代價**：驗證集比例壓到極小（early stopping 樣本變少），模型選點穩定度會下降，"
        "這是換取「訓練集涵蓋近期」的必要取捨。",
    ]
    if has_full:
        lines += [
            "6. **完整重跑候選的變因數**：`b2full` 同時換了因子參數、特徵矩陣、訓練窗與模型，"
            "與 incumbent 的差異無法單獨歸因；`tafix` 則刻意固定訓練設定，唯一變因為"
            "「best_factors.json 的技術指標參數是否真的進入特徵矩陣」。",
            "7. **b1/b2 為何沒有 full 版本**：`optimize_factors.py` 的防洩漏保護看 `config.BACKTEST_DATE`，"
            "設 `None` 時不截斷，會把回測區間納入因子搜尋（lookahead），故 `None` 型無法做乾淨的完整重訓對照。",
        ]
    lines += ["", "## 五、產出檔案", ""]
    for r in results:
        if r.get("report"):
            lines.append(f"- `{os.path.relpath(r['report'], BASE_DIR)}`（{r['tag']} 明細）")
    for r in results:
        if r["tag"] != "incumbent" and os.path.isdir(candidate_dir(r["tag"])):
            extra = "，含 best_factors 與特徵檔" if r.get("full_pipeline") else ""
            lines.append(f"- `{os.path.relpath(candidate_dir(r['tag']), BASE_DIR)}/`（{r['tag']} 訓練成果，已保留{extra}）")
    lines += [
        f"- `{os.path.relpath(log_path, BASE_DIR)}`（完整執行 log）",
        "",
        "## 六、要採用某個候選時",
        "",
        "```bash",
        f"python {SELF_REL_PATH} --promote <tag>",
        "```",
        "",
        "promote 會替換 `models/` 內的模型檔；若該候選是完整重跑產生的，會**一併還原它的 "
        "`configs/best_factors.json` 與特徵檔**，確保模型與特徵矩陣相符（兩者不一致會讓推論靜默失真）。",
        "",
        "promote **不會改 config.py**。若決定長期採用某候選，需自行把該候選的 `BACKTEST_DATE` / "
        "`TRAIN_SPLIT_RATIO` / `VALID_SPLIT_RATIO`（見上表）寫回 config.py，否則下次 "
        "`auto_pipeline.py -s train` 會用舊設定重訓覆蓋掉。",
        "",
    ]

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"model_mode_compare_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def print_summary(results):
    print("\n" + "=" * 78)
    print("  比較結果")
    print("=" * 78)
    print(f"  {'候選':<12}{'報酬(%)':>12}{'MDD(%)':>12}{'Calmar':>10}{'買/賣':>10}{'勝率(%)':>10}")
    for r in results:
        cal = calmar(r["total_return"], r["max_dd"])
        cal_s = f"{cal:.2f}" if cal is not None else "N/A"
        trade_s = f"{r['n_buy']}/{r['n_sell']}"
        win_s = f"{r['win_rate']:.0f}" if r["win_rate"] is not None else "N/A"
        print(f"  {r['tag']:<12}{r['total_return']:>+12.2f}{-r['max_dd']:>12.2f}"
              f"{cal_s:>10}{trade_s:>10}{win_s:>10}")
    print("=" * 78)


# ── 子指令 ──────────────────────────────────────────────────────────────
def do_promote(tag):
    src = candidate_dir(tag)
    if not os.path.isdir(src):
        print(f"[錯誤] 找不到候選模型目錄: {src}")
        return 1
    meta = {}
    meta_path = os.path.join(src, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  候選 `{tag}` 訓練於 {meta.get('trained_at')}，config 設定：{meta.get('config')}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = snapshot(os.path.join(MODEL_DIR, f"{BACKUP_PREFIX}before_promote_{ts}"),
                      with_assets=meta.get("full_pipeline", False))
    restored = restore(src, with_assets=True)
    print(f"\n[完成] 已將候選 `{tag}` 套用至 models/（原內容備份於 {os.path.relpath(backup, BASE_DIR)}）")
    if restored:
        print(f"       同時還原了：{', '.join(restored)} —— 模型與特徵矩陣已對齊，無須重跑特徵工程")
    elif meta.get("full_pipeline"):
        print("[警告] 該候選標記為完整重跑，但目錄內缺少特徵檔／因子檔；"
              "請先執行 auto_pipeline.py -s feature 重建特徵，否則模型與特徵不匹配")
    if meta.get("config"):
        print("[提醒] config.py 未被修改。若要長期採用此候選，請自行寫回下列設定，"
              "否則下次訓練會用舊設定覆蓋：")
        for k, v in meta["config"].items():
            print(f"        {k} = {v}")
    return 0


def do_restore_latest():
    """優先依 manifest 完整還原（含 config 與資產），否則退回還原最新模型備份"""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            man = json.load(f)
        print(f"  發現未完成的實驗紀錄（建立於 {man.get('created_at')}）")
        for k, v in (man.get("config") or {}).items():
            write_config_var(k, v)
        restored = restore(man["backup_dir"], with_assets=True)
        print(f"[完成] 已自 {os.path.relpath(man['backup_dir'], BASE_DIR)} 還原 models/"
              + (f" 與 {', '.join(restored)}" if restored else ""))
        clear_manifest()
        return 0
    backup = latest_backup_dir()
    if not backup:
        print("[錯誤] 找不到任何備份目錄")
        return 1
    restored = restore(backup, with_assets=True)
    print(f"[完成] 已自 {os.path.relpath(backup, BASE_DIR)} 還原 models/"
          + (f" 與 {', '.join(restored)}" if restored else ""))
    print("[提醒] 無 manifest 可用，config.py 的 BACKTEST_DATE / TRAIN_SPLIT_RATIO / "
          "VALID_SPLIT_RATIO / OPTIMIZATION_TRIALS 請自行確認是否為原值")
    return 0


# ── 主流程 ──────────────────────────────────────────────────────────────
def build_candidates(args, orig_cfg):
    _same_train_cfg = {k: orig_cfg[k] for k in
                       ("BACKTEST_DATE", "TRAIN_SPLIT_RATIO", "VALID_SPLIT_RATIO", "OPTIMIZATION_TRIALS")}
    all_defs = {
        "tafix": {
            "tag": "tafix",
            "name": "因子參數落地修復（訓練設定與現行完全相同，僅重建特徵與模型）",
            # 唯一變因＝best_factors.json 的技術指標參數有沒有真的進到特徵矩陣。
            # 2026-08-14 前這些參數在多核心路徑被子行程忽略（tests/FACTOR_OBJECTIVE_PLAN.md §1），
            # 故 incumbent 等同 APPLY_BEST_FACTORS_TA=False 的狀態。
            "cfg": dict(_same_train_cfg, APPLY_BEST_FACTORS_TA="True"),
            "steps": ["feature", "train"],
            "factor_note": "沿用現行 best_factors.json（首次真正生效）",
            "full_pipeline": True,   # 會重建特徵檔 → 需備份/還原/隨候選保存
        },
        "b1": {
            "tag": "b1",
            "name": "解除資料截斷，切分比例不變",
            "cfg": {"BACKTEST_DATE": "None",
                    "TRAIN_SPLIT_RATIO": orig_cfg["TRAIN_SPLIT_RATIO"],
                    "VALID_SPLIT_RATIO": orig_cfg["VALID_SPLIT_RATIO"],
                    "OPTIMIZATION_TRIALS": orig_cfg["OPTIMIZATION_TRIALS"]},
            "steps": ["train"],
            "factor_note": "沿用現行",
        },
        "b2": {
            "tag": "b2",
            "name": "截斷於回測前一日 + 高比例切分（訓練集涵蓋近期）",
            "cfg": {"BACKTEST_DATE": f'"{args.b2_cutoff}"',
                    "TRAIN_SPLIT_RATIO": str(B2_TRAIN_SPLIT),
                    "VALID_SPLIT_RATIO": str(B2_VALID_SPLIT),
                    "OPTIMIZATION_TRIALS": orig_cfg["OPTIMIZATION_TRIALS"]},
            "steps": ["train"],
            "factor_note": "沿用現行",
        },
        "b2full": {
            "tag": "b2full",
            "name": "同 b2 截斷設定，完整重跑 optimize→feature→train",
            "cfg": {"BACKTEST_DATE": f'"{args.b2_cutoff}"',
                    "TRAIN_SPLIT_RATIO": str(B2_TRAIN_SPLIT),
                    "VALID_SPLIT_RATIO": str(B2_VALID_SPLIT),
                    "OPTIMIZATION_TRIALS": str(args.factor_trials)},
            "steps": ["optimize", "feature", "train"],
            "factor_note": f"重新搜尋 {args.factor_trials} trials",
            "full_pipeline": True,
        },
    }
    return [all_defs[t] for t in args.only if t in all_defs]


def main():
    parser = argparse.ArgumentParser(description="模型時效性對照實驗")
    parser.add_argument("-s", "--start", default=DEFAULT_START, help="回測起始日 (YYYY-MM-DD)")
    parser.add_argument("-e", "--end", default=DEFAULT_END, help="回測結束日 (YYYY-MM-DD)")
    parser.add_argument("-c", "--capital", type=int, default=2_000_000, help="初始資金")
    parser.add_argument("-m", "--max_pos", type=int, default=None, help="最大持股檔數（預設讀 config.MAX_POSITIONS）")
    parser.add_argument("--only", nargs="+", choices=["tafix", "b1", "b2", "b2full"], default=None,
                        help="指定候選組合（預設 b1 b2；加 --full 則為 b2full）")
    parser.add_argument("--full", action="store_true",
                        help="完整重跑 optimize→feature→train（含貝葉斯因子調參，數小時等級）")
    parser.add_argument("--factor-trials", type=int, default=None,
                        help="b2full 的 Optuna 因子搜尋輪數（預設讀 config.OPTIMIZATION_TRIALS）")
    parser.add_argument("--reuse-candidate", action="store_true", help="沿用 models/_candidate_<tag>/ 既有成果，不重跑")
    parser.add_argument("--promote", help="把指定候選（b1/b2/b2full）蓋回 models/ 後結束")
    parser.add_argument("--restore-latest", action="store_true", help="自最新備份還原後結束")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # 防 Windows cp950 主控台遇中文中斷長時間任務
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.promote:
        return do_promote(args.promote)
    if args.restore_latest:
        return do_restore_latest()

    if os.path.exists(MANIFEST_PATH):
        print(f"[錯誤] 偵測到未完成的實驗紀錄 {os.path.relpath(MANIFEST_PATH, BASE_DIR)}，"
              "代表上次執行未正常還原（可能被強制中止）。")
        print(f"       請先執行: python {SELF_REL_PATH} --restore-latest")
        return 1

    from config import MAX_POSITIONS, OPTIMIZATION_TRIALS
    if args.max_pos is None:
        args.max_pos = MAX_POSITIONS
    if args.factor_trials is None:
        args.factor_trials = OPTIMIZATION_TRIALS
    if args.only is None:
        args.only = ["b2full"] if args.full else ["b1", "b2"]
    elif args.full and "b2full" not in args.only:
        args.only.append("b2full")

    if not os.path.exists(FEATURE_PATH):
        print(f"[錯誤] 找不到特徵檔 {FEATURE_PATH}，請先執行 auto_pipeline.py -s feature")
        return 1

    orig_cfg = {k: read_config_var(k) for k in CONFIG_VARS}
    missing = [k for k, v in orig_cfg.items() if v is None]
    if missing:
        print(f"[錯誤] config.py 缺少變數 {missing}；TRAIN_SPLIT_RATIO / VALID_SPLIT_RATIO 為本實驗前提，請先加入 config.py")
        return 1

    args.b2_cutoff = (pd.to_datetime(args.start) - pd.Timedelta(days=1)).strftime("%Y%m%d")
    candidates = build_candidates(args, orig_cfg)
    if not candidates:
        print("[錯誤] 未選定任何候選")
        return 1
    need_assets = any(c.get("full_pipeline") for c in candidates)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(REPORT_DIR, exist_ok=True)
    log_path = os.path.join(REPORT_DIR, f"compare_model_modes_{ts}.log")
    sys.stdout = _Tee(log_path)

    print("=" * 78)
    print("  模型時效性對照實驗")
    print("=" * 78)
    print(f"  開始時間 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  回測區間 : {args.start} ~ {args.end}")
    print(f"  初始資金 : {args.capital:,} ｜ 最大持股: {args.max_pos}")
    print(f"  候選     : incumbent + {', '.join(c['tag'] for c in candidates)}")
    print(f"  config 原值: {orig_cfg}")
    print(f"  執行 log : {os.path.relpath(log_path, BASE_DIR)}")
    if need_assets:
        print("\n  ⚠️  含完整重跑（因子調參）候選，預計耗時數小時；")
        print("      期間 config.py、models/、configs/best_factors.json 與特徵檔會暫時被替換，")
        print("      請勿同時執行 Auto_RUN.py / run_daily.ps1。")
        print("      若行程被強制中止，重跑前先執行 --restore-latest 還原。")

    backup_dir = snapshot(os.path.join(MODEL_DIR, f"{BACKUP_PREFIX}{ts}"), with_assets=need_assets)
    write_manifest(backup_dir, orig_cfg)
    print(f"  現行狀態已備份至: {os.path.relpath(backup_dir, BASE_DIR)}")

    results = []
    try:
        total_steps = len(candidates) + 1
        print("\n" + "─" * 78 + f"\n  [1/{total_steps}] 回測現行模型 (incumbent)\n" + "─" * 78)
        inc = run_backtest("incumbent", args.start, args.end, args.capital, args.max_pos)
        inc["name"] = "現行生產模型（不重訓）"
        inc["cfg_backtest_date"] = orig_cfg["BACKTEST_DATE"]
        inc["cfg_split"] = f"{orig_cfg['TRAIN_SPLIT_RATIO']} / {orig_cfg['VALID_SPLIT_RATIO']}"
        bf = read_factor_params()
        inc["factor_note"] = (f"沿用現行（{bf.get('optimized_at', '?')}、"
                              f"截斷 {bf.get('backtest_date', '?')}、{bf.get('iterations', '?')} trials）")
        bt = orig_cfg["BACKTEST_DATE"].strip("\"'")
        inc["fit"] = estimate_fit_range(
            None if bt == "None" else bt,
            float(orig_cfg["TRAIN_SPLIT_RATIO"]), float(orig_cfg["VALID_SPLIT_RATIO"]),
        )
        results.append(inc)

        for i, cand in enumerate(candidates, start=2):
            tag = cand["tag"]
            print("\n" + "─" * 78 + f"\n  [{i}/{total_steps}] 候選 {tag}：{cand['name']}\n" + "─" * 78)
            cdir = candidate_dir(tag)
            fit, factor_params = {}, None

            if args.reuse_candidate and os.path.isdir(cdir):
                print(f"  [沿用] 既有候選成果 {os.path.relpath(cdir, BASE_DIR)}（跳過重跑）")
                meta_path = os.path.join(cdir, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    fit, factor_params = meta.get("fit", {}), meta.get("factor_params")
                restored = restore(cdir, with_assets=True)
                if cand.get("full_pipeline") and not restored:
                    print("  [警告] 該候選缺少特徵檔副本，模型與現行特徵矩陣可能不匹配，結果僅供參考")
            else:
                for k, v in cand["cfg"].items():
                    write_config_var(k, v)
                logs = {}
                for step in cand["steps"]:
                    logs[step] = run_pipeline_step(step)
                fit = parse_fit_range(logs.get("train", []))
                if cand.get("full_pipeline"):
                    factor_params = read_factor_params().get("params")
                snapshot(cdir, with_assets=cand.get("full_pipeline", False))
                with open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"tag": tag, "name": cand["name"],
                               "trained_at": datetime.now().isoformat(timespec="seconds"),
                               "config": cand["cfg"], "fit": fit,
                               "full_pipeline": cand.get("full_pipeline", False),
                               "factor_params": factor_params}, f, ensure_ascii=False, indent=2)
                print(f"  [保留] 候選成果已存至 {os.path.relpath(cdir, BASE_DIR)}")

            res = run_backtest(tag, args.start, args.end, args.capital, args.max_pos)
            res.update({"name": cand["name"], "cfg_backtest_date": cand["cfg"]["BACKTEST_DATE"],
                        "cfg_split": f"{cand['cfg']['TRAIN_SPLIT_RATIO']} / {cand['cfg']['VALID_SPLIT_RATIO']}",
                        "factor_note": cand["factor_note"], "fit": fit,
                        "full_pipeline": cand.get("full_pipeline", False),
                        "factor_params": factor_params})
            results.append(res)
    finally:
        print("\n" + "─" * 78 + "\n  還原環境\n" + "─" * 78)
        for k, v in orig_cfg.items():
            try:
                write_config_var(k, v)
            except Exception as e:
                print(f"  [警告] 還原 {k} 失敗: {e}")
        try:
            restored = restore(backup_dir, with_assets=True)
            print(f"  models/ 已自 {os.path.relpath(backup_dir, BASE_DIR)} 還原"
                  + (f"，並還原 {', '.join(restored)}" if restored else ""))
            clear_manifest()
        except Exception as e:
            print(f"  [警告] 還原失敗: {e}；請執行 python {SELF_REL_PATH} --restore-latest")

    if len(results) > 1:
        print_summary(results)
        path = build_report(results, args, ts, orig_cfg, log_path)
        print(f"\n[完成] 比較報告: {os.path.relpath(path, BASE_DIR)}")
        print(f"       執行 log : {os.path.relpath(log_path, BASE_DIR)}")
        print(f"       候選成果已保留，若要採用: python {SELF_REL_PATH} --promote <tag>")
    print(f"  結束時間 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
