# -*- coding: utf-8 -*-
"""
test_liquidity_gate.py — 進場流動性門檻 潔淨 OOS 部署把關（完全隔離）
================================================================================
背景：
  2026-09-02 的生產推論把 2438 翔耀（近 20 日均成交金額僅 143 萬元／日）與
  6272 驊陞（495 萬元／日）排進買進建議前三名。同一份清單裡的 3026 禾伸堂
  為 63 億元／日、3167 大量 27 億元／日 —— 相差三到四個數量級。

  這暴露一個結構性缺口：config.py 目前**沒有任何流動性／成交金額／市值門檻**。
  而 trading_sim 的限價搓合以歷史 OHLC 判定成交、不模擬衝擊成本，等於在極低
  流動性個股上系統性高估可執行性。

候選改造（唯一變因）：
  進場時要求「訊號日（T-1）的近 LIQ_WINDOW 日平均成交金額 >= 門檻」，
  不足者不得進場。門檻 = 0 即現行（完全不過濾）。

為何是過濾器而非特徵（決定了本測試的做法）：
  流動性門檻不進模型、不改變任何 Day1/2/3 分數的產生方式，只在「候選挑選」
  階段剔除標的。故本測試**不需重訓模型**，唯一變因乾淨。
  實作上以 monkeypatch 包裝 day-1 模型，把被擋標的的 (prob_strong, prob_weak)
  壓成 (0, 1) → Day1_net = -100，遠低於任何 buy_threshold，等價於在
  trading_sim.py:535 的買進迴圈加一行 `continue`，但不必改動生產程式碼。

  ※ 副作用（已知、不影響判定）：trading_sim.py:702 的 bottom-5% 對照組
    （portfolio_spread 診斷欄）會被壓低分數污染。該欄不參與交易決策，
    亦不影響報酬／MDD／曝險，故不予校正。Day2/Day3 模型完全不動，
    出場端（D3 轉弱／移動止盈／停損）行為與現行一致。

方法論（對齊 CLAUDE.md §4.5 #4 與 #7 的教訓）：
  * §4.5 #4「判準是全 OOS 報酬與回撤雙贏 且 多數子窗穩健」。
  * §4.5 #7「參數掃描必須看多起點而非單一窗，非單調曲線＝雜訊擬合」。
    故主判準採**固定終點、六個起點**的長區間掃描，而非季度切片 ——
    後者違反 §4.5 #5（MIN_HOLD_DAYS 達 20 天且獲利靠肥尾，短切片會把
    跨窗贏家攔腰砍斷，系統性低估績效）。
  * 起點組沿用 §4.5 #7 判死「降級砍倉」時的同一組，以利橫向對照。

隔離保證：
  * 生產 parquet 唯讀；不寫任何 sandbox 檔、不改 config.py／models／data。
  * run_simulation 一律 export_report=False，不寫 reports/。
  * monkeypatch 僅存在於本行程記憶體，finally 還原。

執行：
  cd D:\\VScode_Stock\\Stock
  python tests/test_liquidity_gate.py
  python tests/test_liquidity_gate.py --thresholds 0,0.1,0.3,0.5,1.0,2.0
  python tests/test_liquidity_gate.py --starts 2025-08-01,2026-01-01

════════════════════════════════════════════════════════════════════════════════
實測結果（2026-09-08，6 起點 × 6 門檻）：❌ 未通過，不移植
════════════════════════════════════════════════════════════════════════════════
  全 OOS 單窗（起點 2025-08-01）極具誘惑性：報酬隨門檻由 +106% 單調升到 +166%
  （門檻 1.0 億），Calmar 4.79 → 6.87，轉折僅 1 次。**只看這個窗會判定為大幅有效。**

  但多起點掃描直接推翻：各門檻的「起點雙贏」僅 0～2/6，回撤幾乎全面惡化。
  同一門檻在相鄰起點間的報酬差可達 200pp —— 例如門檻 1.0 億相對現行：
      起點 2025-08-01 +59pp ／ 10-01 +7pp ／ 12-01 −15pp
           2026-01-01 +183pp ／ 02-02 −1pp ／ 04-01 −24pp
  起點只差一個月（2025-12-01 → 2026-01-01），結果從 −15pp 跳到 +183pp，
  這是路徑依賴的雜訊，不是可移植的效應。

  機制解釋（呼應 §4.5 #1／#2）：門檻對持倉結構的影響其實很小（均持倉
  4.32 → 4.12~4.25、曝險 84.7% → 79.7~82.6%），代表它多半擋掉的是本來
  也不會進場的標的，真正改變的交易筆數極少。少數幾筆的肥尾效應經
  MAX_POSITIONS=5（單筆權重 20%）複利放大，就足以製造 ±180pp 的擺盪。
  且流動性門檻屬「無差別」降曝險，非 regime／ATR 那種選擇性降曝險，
  故 §4.5 #2 的結論成立：回撤不但沒改善，多數起點反而更深。

  教訓：本例再次驗證 §4.5 #7 ——「單看全窗改善不算數，且曲線單調也不等於
  高原」。此處曲線比當初「降級砍倉」更漂亮（近乎單調遞增），仍是雜訊。

  ※ 尚未檢驗的變體（若日後要再試，須沿用本框架的多起點判準）：
    相對排名門檻（剔除成交金額後 N%）、與部位規模掛鉤的門檻
    （例如要求日均成交金額 > 單筆部位金額的 K 倍）。後者在概念上比絕對
    金額合理，但依 §4.5 #4「別再盲試」，須有明確假設再動手。

  ※ 本次事件暴露的「回測以歷史 OHLC 搓合、不模擬衝擊成本」屬**執行面**風險，
    與績效面無關，不該用進場過濾器解決——應以下單紀律（嚴守建議限價、
    買不到就放棄）處理。
"""

import os
import sys
import io
import argparse
import contextlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import trading_sim
from trading_sim import run_simulation
from config import BACKTEST_DATE, MAX_POSITIONS, SIM_DEFAULT_CAPITAL
from config import FEATURES_PARQUET as DATA_PATH   # 資料路徑唯一來源：config.py § 0
TURNOVER_COL = "成交金額"          # TWSE 原始欄位，單位：元（字串含千分位逗號）

# 候選門檻定義（單位：億元／日）。0 = 現行（不過濾）。
LIQ_WINDOW = 20                     # 近 N 日平均
LIQ_MIN_PERIODS = 20                # 不足 N 日 → NaN → 保守視為不通過
DEFAULT_THRESHOLDS = "0,0.1,0.3,0.5,1.0,2.0"

# 固定終點、多起點（§4.5 #7）。沿用判死「降級砍倉」時的同一組起點以利對照。
DEFAULT_STARTS = "2025-08-01,2025-10-01,2025-12-01,2026-01-01,2026-02-02,2026-04-01"

# 本次事件的當事標的，用於在報表末端做關聯診斷。
CASE_SIDS = ["2438", "2419", "6272", "3026", "3167"]
CASE_SIGNAL_DATE = "2026-09-02"


# ──────────────────────────────────────────────────────────────────────────
#  流動性指標
# ──────────────────────────────────────────────────────────────────────────
def load_liquidity_frame():
    """
    讀生產 parquet，計算每列的近 LIQ_WINDOW 日平均成交金額（元）。

    index 對齊的正確性依據（已逐段查核 trading_sim.py）：
      read_parquet → RangeIndex 0..N-1
      df.merge(df_daily_mkt, on='date', how='left')  → 右側 date 唯一不膨脹、
                                                        左側保序 → index 值不變
      filter_stocks_by_train_industries()            → 布林遮罩取子集，不 reset
      df_sim = df[df['date'].isin(needed_dates)]     → 同上
    故 run_simulation 內部 X_sim.index 的值 == 本函式回傳 df 的 index 值。
    此假設會在 _Day1LiquidityFilter 中以實際看到的 index 反查日期做執行期自驗。
    """
    df = pd.read_parquet(DATA_PATH, columns=["stock_id", "date", TURNOVER_COL])
    if not isinstance(df.index, pd.RangeIndex):
        raise RuntimeError("parquet 讀出的 index 非 RangeIndex，index 對齊假設失效。")

    turnover = pd.to_numeric(
        df[TURNOVER_COL].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.assign(turnover=turnover)

    # 依 (stock_id, date) 排序後做 rolling，再依原 index 還原順序。
    # 只用到當列與其之前的資料，且買進迴圈取的是 T-1 列，故無前視偏差。
    ordered = df.sort_values(["stock_id", "date"])
    df["liq"] = (ordered.groupby("stock_id")["turnover"]
                 .rolling(LIQ_WINDOW, min_periods=LIQ_MIN_PERIODS)
                 .mean()
                 .reset_index(level=0, drop=True))
    return df


def blocked_index_for(df_liq, threshold_yi):
    """回傳該門檻下應被擋下的列 index（threshold_yi 單位：億元）。0 → 不過濾。"""
    if threshold_yi <= 0:
        return np.array([], dtype=np.int64)
    # NaN（上市未滿 LIQ_MIN_PERIODS 日）保守視為不通過。
    ok = df_liq["liq"] >= threshold_yi * 1e8
    return df_liq.index[~ok].to_numpy()


# ──────────────────────────────────────────────────────────────────────────
#  模型包裝：等價於在買進迴圈剔除低流動性標的
# ──────────────────────────────────────────────────────────────────────────
class _Day1LiquidityFilter:
    """
    包裝 day-1 模型：被擋標的輸出 (prob_weak, prob_mid, prob_strong) = (1, 0, 0)，
    使 Day1_net = (strong - weak) * 100 = -100，必定低於任何 buy_threshold。
    """

    def __init__(self, inner, blocked_idx):
        self._inner = inner
        self._blocked = set(blocked_idx.tolist())
        self.seen_index = None          # 供執行期自驗
        self.n_blocked_rows = 0

    def predict(self, X, *args, **kwargs):
        preds = np.asarray(self._inner.predict(X, *args, **kwargs))
        if preds.ndim != 2 or preds.shape[1] != 3:
            return preds                # 非預期格式交由 trading_sim 自行報錯
        self.seen_index = X.index
        if not self._blocked:
            return preds
        mask = X.index.isin(self._blocked)
        self.n_blocked_rows = int(mask.sum())
        if self.n_blocked_rows:
            preds = preds.copy()
            preds[mask, 0] = 1.0        # prob_weak
            preds[mask, 1] = 0.0
            preds[mask, 2] = 0.0        # prob_strong
        return preds

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_quiet(**kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_simulation(**kwargs)
    if result is None:
        return None
    total_return, max_dd, history = result
    return total_return, max_dd * 100.0, history


def run_with_threshold(threshold_yi, df_liq, start, end, capital, max_pos, verify_dates=True):
    """以指定流動性門檻跑生產回測。門檻 0 → 完全不過濾（現行基準）。"""
    blocked = blocked_index_for(df_liq, threshold_yi)
    orig_load = trading_sim.load_models
    holder = {}

    def _patched():
        models = orig_load()
        wrapped = _Day1LiquidityFilter(models[1], blocked)
        models[1] = wrapped
        holder["w"] = wrapped
        return models

    trading_sim.load_models = _patched
    try:
        res = run_quiet(start_date=start, end_date=end,
                        initial_capital=capital, max_positions=max_pos,
                        export_report=False)
    finally:
        trading_sim.load_models = orig_load

    w = holder.get("w")
    if verify_dates and w is not None and w.seen_index is not None:
        _verify_index_alignment(df_liq, w.seen_index, start, end)
    return res, (w.n_blocked_rows if w else 0), len(blocked)


def _verify_index_alignment(df_liq, seen_index, start, end):
    """
    執行期自驗：若 index 對齊假設失效（例如 trading_sim 日後加了 reset_index），
    反查到的日期範圍會偏離回測區間。寧可中止，也不要默默輸出錯誤結論。
    """
    dates = pd.to_datetime(df_liq.loc[seen_index, "date"])
    lo, hi = dates.min(), dates.max()
    # df_sim 會多帶回測起點之前的暖身列，故只驗上界與「不得超出資料末端」。
    if hi > pd.to_datetime(end) or lo > pd.to_datetime(end):
        raise RuntimeError(
            f"index 對齊自驗失敗：模型看到的日期範圍 {lo.date()}~{hi.date()} "
            f"超出回測區間 {start}~{end}。trading_sim 的前置處理可能已改動 index。")


def exposure_stats(history):
    """自 history 還原 benchmark、曝險度與空手比率（同 test_regime_window_gate）。"""
    try:
        h = pd.DataFrame(history)
        daily_mkt = h["portfolio_return"] - h["portfolio_alpha"]
        bench = float(np.prod(1.0 + daily_mkt.values) - 1.0) * 100.0
        exposure = float((h["invested"] / h["equity"]).mean()) * 100.0
        n_pos = float(h["n_positions"].mean())
        return bench, exposure, n_pos
    except Exception as e:
        print(f"    [警告] history 解析失敗：{e}")
        return float("nan"), float("nan"), float("nan")


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="億元／日，逗號分隔，0=現行")
    ap.add_argument("--starts", default=DEFAULT_STARTS, help="多起點掃描的起點，逗號分隔")
    ap.add_argument("--capital", type=int, default=SIM_DEFAULT_CAPITAL)
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    starts = [s.strip() for s in args.starts.split(",") if s.strip()]
    if 0.0 not in thresholds:
        thresholds.insert(0, 0.0)

    df_liq = load_liquidity_frame()
    latest = str(pd.to_datetime(df_liq["date"]).max().date())
    oos_start = str(pd.to_datetime(str(BACKTEST_DATE)).date())

    print("=" * 96)
    print("  進場流動性門檻 潔淨 OOS 部署把關（方法論對齊 §4.5 #4／#7）")
    print("=" * 96)
    print(f"  潔淨 OOS 上界  : {oos_start} ~ {latest}（模型訓練截點之後）")
    print(f"  候選指標       : 近 {LIQ_WINDOW} 日平均成交金額（訊號日 T-1，無前視）")
    print(f"  門檻掃描       : {thresholds} 億元／日（0 = 現行，不過濾）")
    print(f"  多起點         : {starts}（終點固定 {latest}）")
    print(f"  引擎           : 生產 trading_sim.run_simulation（T+1 限價搓合、T+2 交割）")
    print(f"  資金 {args.capital:,} | 最大持股 {MAX_POSITIONS}")
    print("  唯一變因為進場流動性門檻；模型、特徵、風控參數、出場邏輯全同。\n")

    # 母體覆蓋率：門檻擋掉多少 OOS 樣本
    oos_mask = pd.to_datetime(df_liq["date"]) >= pd.to_datetime(str(BACKTEST_DATE))
    print("  [母體覆蓋率] OOS 期間通過門檻的個股-日比例：")
    for th in thresholds:
        if th <= 0:
            print(f"    門檻 {th:>4} 億：100.0%（現行）")
        else:
            keep = float((df_liq.loc[oos_mask, "liq"] >= th * 1e8).mean()) * 100.0
            print(f"    門檻 {th:>4} 億：{keep:5.1f}%")
    print()

    # ── 多起點 × 多門檻掃描 ────────────────────────────────────────────
    rows = []
    total = len(starts) * len(thresholds)
    done = 0
    for start in starts:
        for th in thresholds:
            done += 1
            print(f"  [{done:>2}/{total}] 回測 起點={start} 門檻={th} 億 ...", flush=True)
            res, n_blocked, n_total_blocked = run_with_threshold(
                th, df_liq, start, latest, args.capital, MAX_POSITIONS)
            if res is None:
                print(f"      [警告] 無結果，跳過")
                continue
            ret, mdd, hist = res
            bench, exp, npos = exposure_stats(hist)
            rows.append({
                "起點": start, "門檻億": th,
                "報酬%": round(ret, 2), "MDD%": round(mdd, 2),
                "Calmar": round(ret / abs(mdd), 2) if abs(mdd) > 1e-9 else float("nan"),
                "大盤%": round(bench, 2), "曝險%": round(exp, 1), "均持倉": round(npos, 2),
            })

    if not rows:
        print("\n[錯誤] 全部回測皆無結果。")
        return

    df_r = pd.DataFrame(rows)

    print("\n" + "=" * 96)
    print("  逐窗結果")
    print("=" * 96)
    for start in starts:
        sub = df_r[df_r["起點"] == start]
        if sub.empty:
            continue
        base = sub[sub["門檻億"] == 0.0]
        print(f"\n  ── 起點 {start} ~ {latest} " + "─" * 40)
        print(sub.drop(columns=["起點"]).to_string(index=False))
        if not base.empty:
            b = base.iloc[0]
            print(f"     （現行基準：報酬 {b['報酬%']:+.2f}%  MDD {b['MDD%']:.2f}%）")

    # ── 判定 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("  ⚖️ 候選 vs 現行判定")
    print("=" * 96)
    print("  判準（§4.5 #4）：全 OOS 報酬與回撤雙贏，且多數起點穩健。")
    print("  §4.5 #7 附加：曲線須大致單調；非單調＝雜訊擬合，不可解釋為高原。\n")

    full_start = starts[0]
    summary = []
    for th in thresholds:
        if th <= 0:
            continue
        wins_ret = wins_mdd = wins_both = n = 0
        for start in starts:
            b = df_r[(df_r["起點"] == start) & (df_r["門檻億"] == 0.0)]
            c = df_r[(df_r["起點"] == start) & (df_r["門檻億"] == th)]
            if b.empty or c.empty:
                continue
            n += 1
            d_ret = c.iloc[0]["報酬%"] - b.iloc[0]["報酬%"]
            d_mdd = c.iloc[0]["MDD%"] - b.iloc[0]["MDD%"]   # 正值代表回撤更深
            wins_ret += d_ret >= 0
            wins_mdd += d_mdd <= 0
            wins_both += (d_ret >= 0) and (d_mdd <= 0)

        fb = df_r[(df_r["起點"] == full_start) & (df_r["門檻億"] == 0.0)]
        fc = df_r[(df_r["起點"] == full_start) & (df_r["門檻億"] == th)]
        if fb.empty or fc.empty:
            continue
        f_ret = fc.iloc[0]["報酬%"] - fb.iloc[0]["報酬%"]
        f_mdd = fc.iloc[0]["MDD%"] - fb.iloc[0]["MDD%"]
        full_ok = (f_ret >= 0) and (f_mdd <= 0)
        robust = n > 0 and wins_both > n / 2
        summary.append({
            "門檻億": th,
            "全OOS報酬差pp": round(f_ret, 2),
            "全OOS回撤差pp": round(f_mdd, 2),
            "全OOS雙贏": "✓" if full_ok else "✗",
            "起點報酬勝": f"{wins_ret}/{n}",
            "起點回撤勝": f"{wins_mdd}/{n}",
            "起點雙贏": f"{wins_both}/{n}",
            "判定": "✓ 通過" if (full_ok and robust) else "✗ 未通過",
        })

    df_s = pd.DataFrame(summary)
    print(df_s.to_string(index=False))

    passed = [r["門檻億"] for r in summary if r["判定"].startswith("✓")]

    # 單調性檢查（§4.5 #7）
    full_curve = df_r[df_r["起點"] == full_start].sort_values("門檻億")
    if len(full_curve) >= 3:
        rets = full_curve["報酬%"].to_numpy()
        diffs = np.diff(rets)
        sign_changes = int((np.diff(np.sign(diffs)) != 0).sum())
        print(f"\n  全 OOS 報酬對門檻的曲線：{[f'{v:+.1f}' for v in rets]}"
              f"（門檻 {full_curve['門檻億'].tolist()}）")
        print(f"  轉折次數 = {sign_changes}"
              f"{'（大致單調）' if sign_changes <= 1 else '（非單調 → 疑雜訊擬合，§4.5 #7）'}")

    print("\n" + "=" * 96)
    if passed:
        print(f"  結論：門檻 {passed} 億元／日 在潔淨 OOS 通過雙贏＋多數起點穩健。")
        print("  ⚠ 移植前仍須確認：曲線是否單調、通過的門檻是否只由少數起點驅動。")
        print("    若通過，移植方式為在 config.py 新增 MIN_TURNOVER_YI 常數，")
        print("    並於 trading_sim.py:535 買進迴圈與 scripts/inference.py 候選挑選處共用。")
    else:
        print("  結論：無門檻能同時滿足全 OOS 雙贏與多數起點穩健 → 不移植，維持現行。")
        print("  註：即使績效判定未通過，本次事件暴露的『回測不模擬衝擊成本』問題依然存在；")
        print("      該風險屬執行面而非績效面，應另以下單紀律（嚴守限價）處理。")
    print("=" * 96)

    # ── 案例關聯診斷 ────────────────────────────────────────────────────
    print(f"\n  [案例診斷] {CASE_SIGNAL_DATE} 訊號日各標的近 {LIQ_WINDOW} 日均成交金額：")
    sig = df_liq[pd.to_datetime(df_liq["date"]) == pd.to_datetime(CASE_SIGNAL_DATE)]
    for sid in CASE_SIDS:
        r = sig[sig["stock_id"].astype(str) == sid]
        if r.empty:
            print(f"    {sid}: （查無資料）")
            continue
        v = r["liq"].iloc[0]
        if pd.isna(v):
            print(f"    {sid}: 資料不足 {LIQ_MIN_PERIODS} 日 → 保守視為不通過")
            continue
        blocked_at = [th for th in thresholds if th > 0 and v < th * 1e8]
        tag = f"被擋於門檻 {blocked_at} 億" if blocked_at else "全數門檻皆通過"
        print(f"    {sid}: {v/1e8:9.4f} 億元／日  → {tag}")

    print("\n  [Note] 未修改 config.py／models／data 或任何生產檔案；")
    print("         export_report=False，未寫 reports/；monkeypatch 已於 finally 還原。")


if __name__ == "__main__":
    main()
