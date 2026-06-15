# -*- coding: utf-8 -*-
"""
參數敏感度自動診斷器 (Parameter Sensitivity Diagnostic)
========================================================
定位：與 optimize_trading_params.py 互補。
  - optimize_trading_params.py 負責「找」最佳參數 (Optuna 黑箱)。
  - 本工具負責「解釋」參數：每個風控參數在不同市況下如何牽動
    報酬 / 回撤 / 資金曝險 / 對大盤的超額擷取，讓你看著數據手動決策，
    而不是盲目相信優化器的單一結果。

手法：單因子掃描 (OFAT, One-Factor-At-A-Time)
  以 configs/best_trading_params.json 為基準，每次只掃描「一個」參數，
  其餘固定為現行最佳值，跨多個市況區間回測，輸出敏感度表與自動判讀。

從每次回測的 history 反推三項歸因指標：
  1. 大盤 Benchmark(beta) = 每日大盤報酬連乘 (每日大盤報酬 = portfolio_return - portfolio_alpha)
  2. 平均曝險率 exposure = mean(持股市值 / 總資產)  ← 量化資金利用率
  3. Capture = 策略報酬 / 大盤報酬           ← >1 勝過 beta(有 alpha)，<1 跑輸 beta
  另附 OptScore = Return - 2*MDD (重現 optimize_trading_params.py 的評分邏輯)

執行：
  python scripts/param_sensitivity.py                 # 預設掃 3 個高影響參數
  python scripts/param_sensitivity.py --full          # 掃全部 6 個參數 (較久)
  python scripts/param_sensitivity.py -p buy_threshold # 只掃指定參數
  python scripts/param_sensitivity.py -c 2000000
"""
import os
import sys
import io
import json
import argparse
import contextlib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from trading_sim import run_simulation

BEST_PARAMS_PATH = os.path.join(BASE_DIR, "configs", "best_trading_params.json")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "param_sensitivity_report.md")

# ── 預設市況區間 (刻意選性質相反的兩段，檢驗參數的跨市況穩定性) ──
DEFAULT_PERIODS = [
    ("2025_震盪市", "2025-01-02", "2025-12-31"),
    ("2026_大多頭", "2026-01-02", "2026-06-11"),
]

# ── 掃描定義：sweep 名稱 -> (run_simulation 的 kwarg 名稱, 掃描值清單, 說明) ──
SWEEP_DEFS = {
    "buy_threshold":  ("buy_threshold",     [0.0, 5.0, 10.0, 15.0, 21.5], "Day1 多空淨分數買入門檻 (%)"),
    "panic_breadth":  ("mkt_panic_breadth", [0.0, 0.15, 0.25, 0.33, 0.45], "大盤上漲家數避險紅燈門檻"),
    "stop_loss":      ("stop_loss_pct",     [-15.0, -12.0, -9.0, -6.0, -4.0], "個股固定停損 (%)"),
    # --full 才會掃描以下三項
    "ts_activation":  ("ts_activation_pct", [8.0, 12.0, 15.0, 20.0], "移動止盈啟動門檻 (%)"),
    "ts_pullback":    ("ts_pullback_pct",   [-3.0, -6.0, -9.0, -12.0], "移動止盈回撤出場 (%)"),
    "sell_threshold": ("sell_threshold",    [-10.0, -5.0, 0.0, 5.0], "Day3 轉弱賣出門檻 (%)"),
}
CORE_SWEEPS = ["buy_threshold", "panic_breadth", "stop_loss"]

# best_trading_params.json 的 key -> run_simulation 的 kwarg 名稱
PARAM_KEY_MAP = {
    "buy_threshold": "buy_threshold",
    "sell_threshold": "sell_threshold",
    "stop_loss": "stop_loss_pct",
    "panic_ma5": "mkt_panic_ma5",
    "panic_breadth": "mkt_panic_breadth",
    "ts_activation": "ts_activation_pct",
    "ts_pullback": "ts_pullback_pct",
    "min_hold_days": "min_hold_days",
    "markup_pct": "markup_pct",
}


def load_baseline():
    """讀取現行最佳參數作為 OFAT 的固定基準。找不到則回傳空 dict (用 config 預設)。"""
    if not os.path.exists(BEST_PARAMS_PATH):
        return {}
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        bp = json.load(f).get("best_params", {})
    return {PARAM_KEY_MAP[k]: v for k, v in bp.items() if k in PARAM_KEY_MAP}


def reconstruct(history):
    """從 history 反推 benchmark 報酬、平均曝險率、空倉天數佔比。"""
    h = pd.DataFrame(history)
    daily_mkt = h["portfolio_return"] - h["portfolio_alpha"]
    bench_cum = float(np.prod(1.0 + daily_mkt.values) - 1.0) * 100.0
    exposure = float((h["invested"] / h["equity"]).mean()) * 100.0
    flat_pct = float((h["invested"] / h["equity"] < 0.05).mean()) * 100.0
    return bench_cum, exposure, flat_pct


def run_quiet(**kwargs):
    """呼叫 run_simulation 並吞掉冗長 stdout，只取回傳值。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_simulation(**kwargs)
    if result is None:
        return None
    total_return, max_dd, history = result
    return total_return, max_dd * 100.0, history


def sweep_one_param(sweep_name, baseline, periods, capital, max_pos):
    """對單一參數做 OFAT 掃描，回傳 list of dict。"""
    kwarg_name, values, _desc = SWEEP_DEFS[sweep_name]
    rows = []
    for pname, start, end in periods:
        for val in values:
            kwargs = dict(baseline)
            kwargs[kwarg_name] = val
            kwargs.update(start_date=start, end_date=end,
                          initial_capital=capital, max_positions=max_pos)
            print(f"[掃描] {sweep_name}={val} | {pname} ...", flush=True)
            res = run_quiet(**kwargs)
            if res is None:
                continue
            ret, mdd, history = res
            bench, exposure, flat = reconstruct(history)
            capture = (ret / bench) if abs(bench) > 1e-9 else float("nan")
            rows.append({
                "period": pname, "value": val, "ret": ret, "mdd": mdd,
                "bench": bench, "capture": capture, "exposure": exposure,
                "flat": flat, "opt_score": ret - 2.0 * mdd,
            })
    return rows


def diagnose_param(sweep_name, rows):
    """根據掃描結果產生自動判讀文字。"""
    df = pd.DataFrame(rows)
    lines = []
    best_by_score = {}
    for pname in df["period"].unique():
        sub = df[df["period"] == pname].sort_values("value")
        exp_span = sub["exposure"].max() - sub["exposure"].min()
        ret_span = sub["ret"].max() - sub["ret"].min()
        b_ret = sub.loc[sub["ret"].idxmax()]
        b_score = sub.loc[sub["opt_score"].idxmax()]
        best_by_score[pname] = b_score["value"]
        tags = []
        if exp_span > 50:
            tags.append("曝險旋鈕")
        if ret_span < 2 and exp_span < 10:
            tags.append("此區間幾乎無作用")
        tag_str = (" / ".join(tags)) if tags else "影響中性"
        lines.append(
            f"  - [{pname}] 標籤: {tag_str} | 報酬最佳值={b_ret['value']} "
            f"(報酬 {b_ret['ret']:+.1f}%, 曝險 {b_ret['exposure']:.0f}%) | "
            f"OptScore最佳值={b_score['value']} (Score {b_score['opt_score']:+.1f})"
        )
    # 跨市況穩定性
    uniq = set(best_by_score.values())
    if len(uniq) > 1:
        lines.append(f"  ⚠ 跨市況不穩定：各區間的 OptScore 最佳值不一致 {best_by_score} → "
                     f"靜態單一值無法通吃，是過擬合風險來源。")
    else:
        lines.append(f"  ✓ 跨市況一致：OptScore 最佳值皆為 {list(uniq)[0]}，此參數較穩健。")
    return lines


# 跨市況報酬擺幅低於此值，視為「此基準下幾乎無作用」(惰性)，其穩定性無參考價值
MEANINGFUL_RET_SPAN = 3.0


def build_summary(all_data, baseline):
    """彙整所有掃描結果，產生可直接照做的行動建議清單。"""
    lines = ["## 🧭 綜合判讀與行動建議 (Executive Summary)\n"]
    stable_recs, unstable, inert, exposure_dials = [], [], [], []

    for name, rows in all_data.items():
        df = pd.DataFrame(rows)
        best_by_score, exp_span_max, ret_span_max = {}, 0.0, 0.0
        for p in df["period"].unique():
            sub = df[df["period"] == p]
            best_by_score[p] = sub.loc[sub["opt_score"].idxmax(), "value"]
            exp_span_max = max(exp_span_max, sub["exposure"].max() - sub["exposure"].min())
            ret_span_max = max(ret_span_max, sub["ret"].max() - sub["ret"].min())
        kw = SWEEP_DEFS[name][0]
        cur = baseline.get(kw, "?")
        if ret_span_max < MEANINGFUL_RET_SPAN:
            # 在現行基準下幾乎不影響報酬 → 多半是被空倉狀態掩蓋，穩定與否都無參考價值
            inert.append((name, cur, ret_span_max))
        elif len(set(best_by_score.values())) == 1:
            stable_recs.append((name, list(best_by_score.values())[0], cur))
        else:
            unstable.append((name, best_by_score, cur))
        if exp_span_max > 50:
            exposure_dials.append(name)

    # 資金利用率體檢：用 buy_threshold 掃描中「現行值」那格的曝險判斷是否常年空手
    if "buy_threshold" in all_data:
        df = pd.DataFrame(all_data["buy_threshold"])
        cur_bt = baseline.get("buy_threshold")
        cur_rows = df[np.isclose(df["value"], cur_bt)] if cur_bt is not None else df.iloc[0:0]
        if not cur_rows.empty:
            avg_exp = cur_rows["exposure"].mean()
            if avg_exp < 30:
                lines.append(
                    f"**🔴 資金利用率警示**：現行 `buy_threshold={cur_bt}` 下平均曝險僅 "
                    f"**{avg_exp:.0f}%**，資金常年空手、嚴重錯失行情——這是首要瓶頸。\n"
                )

    if stable_recs:
        lines.append("**✅ 跨市況穩健、可安全調整的參數（建議直接採用）：**")
        for name, val, cur in stable_recs:
            lines.append(f"- `{name}`：兩市況 OptScore 最佳值一致 = **{val}**（現行 {cur}）→ 可直接改。")
        lines.append("")
    if unstable:
        lines.append("**⚠ 跨市況不穩定、勿設靜態單一值（需靠有效市況過濾器動態調整）：**")
        for name, bbs, cur in unstable:
            bbs_str = ", ".join(f"{k}={v}" for k, v in bbs.items())
            lines.append(f"- `{name}`：各市況最佳值分歧（{bbs_str}，現行 {cur}）→ 靜態值必過擬合。")
        lines.append("")
    if inert:
        lines.append("**⊘ 現行基準下惰性、建議值不可信（須在「滿倉狀態」重測）：**")
        for name, cur, span in inert:
            lines.append(f"- `{name}`：跨市況報酬擺幅僅 {span:.1f}%（現行 {cur}）→ 因現行 buy_threshold 過高、"
                         f"幾乎空倉，出場/賣出邏輯無部位可作用。須在低 buy_threshold(滿倉)下重掃才有意義。")
        lines.append("")
    if exposure_dials:
        lines.append(f"**🎚️ 曝險旋鈕（直接控制市場參與度，是報酬主閥）：** {', '.join(f'`{n}`' for n in exposure_dials)}\n")

    lines.append("> 判讀邏輯：① 穩健參數直接照建議值改；② 不穩定參數別硬設靜態值，"
                 "真正該補的是「趨勢市進攻、震盪市防守」的市況過濾器；③ 惰性參數須先調低 buy_threshold "
                 "讓資金進場後重掃，否則建議值是空倉假象。改完再重跑 `optimize_trading_params.py`。\n")
    return lines


def fmt_table(rows):
    df = pd.DataFrame(rows)
    out = []
    out.append("| 市況 | 參數值 | 策略報酬 | 大盤beta | Capture | MDD | 平均曝險 | 空倉% | OptScore |")
    out.append("|------|--------|---------|---------|---------|-----|---------|-------|----------|")
    last = None
    for _, r in df.iterrows():
        if last is not None and r["period"] != last:
            out.append("| | | | | | | | | |")
        last = r["period"]
        out.append(
            f"| {r['period']} | {r['value']} | {r['ret']:+.2f}% | {r['bench']:+.2f}% | "
            f"{r['capture']:+.2f} | {r['mdd']:.1f}% | {r['exposure']:.1f}% | "
            f"{r['flat']:.0f}% | {r['opt_score']:+.1f} |"
        )
    return out


def assemble_report(all_data, baseline, capital, max_pos):
    """由原始數據組裝完整 Markdown 報告 (表頭 + 執行摘要 + 各參數明細)。"""
    report = ["# 參數敏感度自動診斷報告 (Parameter Sensitivity)\n"]
    report.append(f"- 初始資金: {capital:,} | 最大持股: {max_pos}")
    report.append(f"- OFAT 基準 (其餘參數固定值): `{baseline}`")
    report.append("- Capture = 策略報酬 / 大盤beta；>1 表勝過大盤(有alpha)，<1 表跑輸大盤")
    report.append("- OptScore = Return − 2×MDD，重現 optimize_trading_params.py 的評分邏輯\n")
    if all_data:
        report.extend(build_summary(all_data, baseline))
        report.append("\n---\n")
    for name, rows in all_data.items():
        desc = SWEEP_DEFS[name][2]
        report.append(f"## 參數: `{name}` — {desc}\n")
        report.extend(fmt_table(rows))
        report.append("\n**自動判讀：**")
        report.extend(diagnose_param(name, rows))
        report.append("")
    return "\n".join(report)


def main():
    ap = argparse.ArgumentParser(description="參數敏感度自動診斷器 (OFAT 掃描)")
    ap.add_argument("-c", "--capital", type=int, default=2_000_000, help="回測初始資金")
    ap.add_argument("-m", "--max_pos", type=int, default=5, help="最大持股數")
    ap.add_argument("-p", "--param", type=str, default=None,
                    help=f"只掃描指定參數 (可選: {', '.join(SWEEP_DEFS.keys())})")
    ap.add_argument("--full", action="store_true", help="掃描全部 6 個參數 (預設只掃 3 個高影響參數)")
    ap.add_argument("--from-json", action="store_true",
                    help="不重跑回測，直接用上次的 JSON 數據重新產生報告 (秒級)")
    ap.add_argument("--base", type=str, default=None,
                    help="覆寫 OFAT 固定基準，格式 'buy_threshold=5,stop_loss=-4'。"
                         "用於在『滿倉狀態(低 buy_threshold)』下重測出場類惰性參數。")
    args = ap.parse_args()

    data_path = REPORT_PATH.replace(".md", "_data.json")

    # ── 模式 A：純重生報告 (讀 JSON，不跑回測) ──
    if getattr(args, "from_json", False):
        if not os.path.exists(data_path):
            print(f"[錯誤] 找不到快取數據 {data_path}，請先正常執行一次。")
            return
        with open(data_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        all_data = cache["sweeps"]
        baseline = cache["baseline"]
        text = assemble_report(all_data, baseline, cache["capital"], args.max_pos)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[成功] 已用快取數據重生報告: {REPORT_PATH}")
        return

    baseline = load_baseline()
    # 套用 --base 覆寫 (友善參數名 -> run_simulation kwarg)
    if args.base:
        for kv in args.base.split(","):
            k, v = kv.split("=")
            k = k.strip()
            if k not in SWEEP_DEFS:
                print(f"[錯誤] --base 含未知參數 {k}，可選: {list(SWEEP_DEFS.keys())}")
                return
            baseline[SWEEP_DEFS[k][0]] = float(v)
        print(f"  [基準覆寫] 已套用 --base：{args.base}")

    if args.param:
        sweep_list = [p.strip() for p in args.param.split(",")]
        bad = [p for p in sweep_list if p not in SWEEP_DEFS]
        if bad:
            print(f"[錯誤] 未知參數 {bad}，可選: {list(SWEEP_DEFS.keys())}")
            return
    elif args.full:
        sweep_list = list(SWEEP_DEFS.keys())
    else:
        sweep_list = CORE_SWEEPS

    # --base 重掃輸出到獨立檔名，避免覆寫主報告
    report_path = REPORT_PATH.replace(".md", "_rescan.md") if args.base else REPORT_PATH
    data_path = report_path.replace(".md", "_data.json")

    print("=" * 70)
    print("  參數敏感度自動診斷器 (OFAT)")
    print(f"  基準參數 (固定值): {baseline}")
    print(f"  掃描參數: {sweep_list}")
    print(f"  市況區間: {[p[0] for p in DEFAULT_PERIODS]}")
    print("=" * 70)

    # 1. 先跑完所有掃描，累積原始數據
    all_data = {}
    for sweep_name in sweep_list:
        _, _, desc = SWEEP_DEFS[sweep_name]
        print(f"\n>>> 掃描參數: {sweep_name} ({desc})")
        rows = sweep_one_param(sweep_name, baseline, DEFAULT_PERIODS, args.capital, args.max_pos)
        if rows:
            all_data[sweep_name] = rows

    # 2. 組裝並寫出報告
    text = assemble_report(all_data, baseline, args.capital, args.max_pos)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    # 3. 原始數據存成 JSON sidecar，之後重新分析免再跑回測
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "capital": args.capital,
                   "periods": [p[0] for p in DEFAULT_PERIODS], "sweeps": all_data},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[成功] 敏感度報告已存檔 (Markdown): {report_path}")
    print(f"[成功] 原始數據已存檔 (JSON): {data_path}")


if __name__ == "__main__":
    main()
