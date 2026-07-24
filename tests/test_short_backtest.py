# -*- coding: utf-8 -*-
"""
test_short_backtest.py — Step 2：放空回測模擬（v2 修正版）
================================================================================
目的：
  在同一套引擎下比較四種方案，使「差異只來自放空覆蓋層」，而非實作差異：
    A 純多頭基準      ：Bear 空倉
    B 多頭 + 反向 ETF ：Bear 時買 00632R
    C 多頭 + 個股放空 ：Bear/Sideways 均可放空
    D 多頭 + 個股放空 ：僅 Bear 可放空（Bear-only）

v2 相對 v1 的修正（v1 的方案比較不可用，原因如下）：
  F1 多頭槽位被結構性卡死【最嚴重】：v1 方案 C 的多頭上限寫成
     (LONG_MAX_POS - SHORT_MAX_POS) = 5-3 = 2 檔，且不論當下是否真的持有空單；
     方案 A 卻是滿血 5 檔。於是「C 牛市輸給 A」有一大部分是多頭倉位被砍半的
     機會成本，而非放空拖累——歸因完全失真。v2 各方案多頭容量一律相同。
  F2 空頭部位不複利：v1 用固定 INITIAL_CAPITAL 計算空單股數，多頭卻用當下 cash。
     資產成長數倍後，空頭名目仍停在初始值，放空影響力被系統性稀釋到近乎無感。
     v2 改用「當下權益」計算，與多頭同基準。
  F3 持有期與訊號錯配：Step 1（v2）實測放空 alpha 僅在持有 1~3 交易日顯著，
     10 日已被交易成本吃掉（真OOS alpha 0.423% < 成本 0.605%）。v1 設 12 日，
     落在無統計證據的區間。v2 預設 3 日並提供 --short-hold 掃描。
  F4 融券保證金未計：v1 註明「做空進場不動 cash」，但台股融券需繳保證金
     （成數約 90%），放空實際會佔用資金並與多頭競爭資本。v2 納入保證金佔用。
  F5 ETF 損益計算錯誤：v1 用 etf_value 反推前一日價值當成本，記錄的 pnl 並非
     相對進場成本的損益。v2 明確追蹤 ETF 成本基礎。
  F6 持有天數用日曆日：v1 的 (date - entry_date).days 是日曆日，5 日曆日僅約
     3.5 個交易日，與 Step 1 的交易日口徑不一致。v2 一律以交易日計算。
  F7 多空損益混記：v1 把多頭與空頭 pnl 併入同一個 trades 清單，無法得知放空
     本身是賺是賠。v2 分開統計。
  F8 Regime 前視：v1 用「當日」大盤趨勢決定當日進場，生產系統（CLAUDE.md §8）
     讀「昨日」以防前視偏差。v2 對齊生產，改讀昨日。

執行：
  cd D:\\VScode_Stock\\Stock
  python tests/test_short_backtest.py
  python tests/test_short_backtest.py --short-hold 1,3,5,12   # 持有期敏感度掃描

隔離保證：
  * 生產 parquet / model 唯讀，不寫入任何生產檔案。
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROD_PARQUET   = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
MODEL_DIR      = os.path.join(ROOT_DIR, "models")
FEAT_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# ── 載入 config ────────────────────────────────────────────────────────────────
try:
    from config import (
        REGIME_BULL_TREND, REGIME_BEAR_TREND,
        REGIME_TREND_WINDOW, REGIME_TREND_MIN_PERIODS,
        FEE_RATE, TAX_RATE,
        ATR_STOP_MULTIPLIER, ATR_STOP_FLOOR_PCT, ATR_STOP_CEILING_PCT,
        BACKTEST_DATE,
        LABEL_WEAK_QUANTILE, SAMPLE_WEIGHT_PENALTY, DEFAULT_DECAY_LAMBDA,
        LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_MAX_DEPTH, LGBM_NUM_LEAVES,
        LGBM_SUBSAMPLE, LGBM_COLSAMPLE, TRAIN_N_JOBS,
    )
    from scripts.utils import get_regime_label, filter_stocks_by_train_industries
except ImportError:
    from utils import get_regime_label, filter_stocks_by_train_industries
    REGIME_BULL_TREND = 0.0015; REGIME_BEAR_TREND = -0.002
    REGIME_TREND_WINDOW = 10;   REGIME_TREND_MIN_PERIODS = 5
    FEE_RATE = 0.001425; TAX_RATE = 0.003
    ATR_STOP_MULTIPLIER = 1.5
    ATR_STOP_FLOOR_PCT = -15.0; ATR_STOP_CEILING_PCT = -5.0
    BACKTEST_DATE = "20250801"
    LABEL_WEAK_QUANTILE = 0.20; SAMPLE_WEIGHT_PENALTY = 2.0; DEFAULT_DECAY_LAMBDA = 0.0
    LGBM_N_ESTIMATORS = 300; LGBM_LEARNING_RATE = 0.03; LGBM_MAX_DEPTH = 4
    LGBM_NUM_LEAVES = 15;    LGBM_SUBSAMPLE = 0.8;      LGBM_COLSAMPLE = 0.8
    TRAIN_N_JOBS = -1

# ── 策略參數 ────────────────────────────────────────────────────────────────────
BACKTEST_START  = "2022-01-01"
BACKTEST_END    = "2026-07-20"
INITIAL_CAPITAL = 1_000_000
LOT             = 1000          # 1 張 = 1000 股

# 多頭（四方案共用，確保差異只來自放空覆蓋層）
LONG_BUY_THR    = 12.0
LONG_MAX_POS    = 5
LONG_ALLOC      = 0.18
LONG_HOLD_DAYS  = 5             # 交易日
LONG_TS_ACTIVATE = 10.0
LONG_TS_PULLBACK = 6.0

# 放空
SHORT_SCORE_THR    = 10.0
SHORT_MAX_POS      = 3
SHORT_ALLOC        = 0.15       # 佔「當下權益」比例（F2：不再用初始資金）
SHORT_TS_ACTIVATE  = 8.0
SHORT_TS_PULLBACK  = 4.0
SHORT_MAX_HOLD     = 3          # 交易日（F3：依 Step 1 實測 1~3 日才有 alpha）
BORROW_RATE_YR     = 0.005
SHORT_MARGIN_RATE  = 0.90       # F4：台股融券保證金成數

# 獨立訓練空頭模型（--short-source trained）專屬常數：對稱鏡像修正
SHORT_LABEL_MAX_RET      = 0.00   # 偏差②：對稱鏡像 long 的 LABEL_STRONG_MIN_RET=0.0
SAMPLE_WEIGHT_RISE_THRES = 0.05   # 偏差①：max_future_ret >= +5% 的樣本給懲罰


def build_short_label(df):
    """偏差②：對稱鏡像空頭標籤。is_short = 橫截面後 20% AND 絕對報酬<0（用 & 而非生產的 |，保鑑別力）。"""
    rank1 = df.groupby("date")["next_ret_1"].rank(pct=True)
    is_short = (rank1 <= LABEL_WEAK_QUANTILE) & (df["next_ret_1"] < SHORT_LABEL_MAX_RET)
    return is_short.astype(int)


def build_short_sample_weight(df):
    """偏差①：反轉樣本權重——罰「未來大漲」樣本（迴避牛市誤發空訊號）。"""
    max_future = df[["next_ret_1", "next_ret_2", "next_ret_3"]].max(axis=1)
    return np.where(max_future >= SAMPLE_WEIGHT_RISE_THRES, SAMPLE_WEIGHT_PENALTY, 1.0)


def train_short_model(train_df, feature_cols):
    """記憶體內訓練 binary 空頭分類器（prob_short），沿用生產 LGBM 超參與時間衰減。"""
    X = train_df.reindex(columns=feature_cols).astype(np.float32)
    y = train_df["y_short"].astype(int)
    dates = pd.to_datetime(train_df["date"])
    time_w = np.exp(-DEFAULT_DECAY_LAMBDA * (dates.max() - dates).dt.days)
    w = train_df["short_sw"].values * time_w.values
    model = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS, learning_rate=LGBM_LEARNING_RATE,
        max_depth=LGBM_MAX_DEPTH, num_leaves=LGBM_NUM_LEAVES,
        subsample=LGBM_SUBSAMPLE, colsample_bytree=LGBM_COLSAMPLE,
        random_state=42, n_jobs=TRAIN_N_JOBS, verbose=-1,
        objective="binary", class_weight="balanced",
    )
    model.fit(X, y, sample_weight=w)
    return model


# ETF 方案
ETF_ALLOC      = 0.30
ETF_DAILY_COST = 0.010 / 252

# 子視窗（標記洩漏狀態：生產模型訓練截至 BACKTEST_DATE）
_CUT = str(pd.to_datetime(BACKTEST_DATE).date())
WINDOWS = [
    ("熊市 2022",     "2022-01-05", "2022-10-25", "樣本內"),
    ("熊轉牛 2022H2", "2022-10-26", "2023-06-30", "樣本內"),
    ("牛市 2023-24",  "2023-07-01", "2024-12-31", "樣本內"),
    # 短促崩跌壓力窗：2022 是 9 個月的慢熊，這兩段是數週內的急跌，
    # regime 過濾器的反應速度在此才會受到真正考驗。
    ("2024急跌",      "2024-07-11", "2024-08-06", "樣本內"),
    ("2026急跌",      "2026-06-01", "2026-07-20", "真OOS"),
    ("真OOS",         _CUT,         BACKTEST_END, "真OOS"),
    ("全期",          BACKTEST_START, BACKTEST_END, "混合"),
]


# ════════════════════════════════════════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════════════════════════════════════════

def atr_stop_pct(atr_val) -> float:
    """多頭 ATR 動態停損（負值，如 -6.5）"""
    if atr_val is not None and not pd.isna(atr_val) and float(atr_val) > 0:
        raw = ATR_STOP_MULTIPLIER * float(atr_val) * 100
        return max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEILING_PCT, -raw))
    return -8.0


def atr_stop_rise(atr_val) -> float:
    """空頭 ATR 動態停損（正值：股票漲超此 % 即停損），為多頭停損的鏡像。"""
    return abs(atr_stop_pct(atr_val))


def build_regime(df: pd.DataFrame, window: int = REGIME_TREND_WINDOW,
                 bear: float = REGIME_BEAR_TREND, bull: float = REGIME_BULL_TREND) -> dict:
    """
    每日 Regime。F8：改讀「昨日」趨勢，與生產系統（CLAUDE.md §8）一致，避免前視。
    window/bear/bull 可覆寫，供敏感度掃描使用（生產值來自 config.py）。
    """
    daily = df.groupby("date")["market_mean_pct"].first().reset_index().sort_values("date")
    daily["trend"] = daily["market_mean_pct"].rolling(
        window, min_periods=min(REGIME_TREND_MIN_PERIODS, window)
    ).mean().shift(1)
    daily["regime"] = daily["trend"].apply(lambda v: get_regime_label(v, bull, bear))
    return dict(zip(pd.to_datetime(daily["date"]), daily["regime"]))


def compute_metrics(cap_series: list, long_trades: list, short_trades: list) -> dict:
    """F7：多空損益分開統計，才看得出放空本身是賺是賠。"""
    if len(cap_series) < 2:
        return {}
    cap = np.array(cap_series, dtype=float)
    total_ret = (cap[-1] - cap[0]) / cap[0] * 100
    peak = np.maximum.accumulate(cap)
    mdd = float(((cap - peak) / peak * 100).min())
    calmar = round(total_ret / abs(mdd), 2) if mdd < -0.01 else float("inf")

    def wr(ts): return round(sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100, 1) if ts else 0.0
    def tot(ts): return round(sum(t["pnl"] for t in ts), 0)
    return {
        "累積報酬%": round(total_ret, 2),
        "MDD%": round(mdd, 2),
        "Calmar": calmar,
        "多筆數": len(long_trades),
        "多勝率%": wr(long_trades),
        "多損益": tot(long_trades),
        "空筆數": len(short_trades),
        "空勝率%": wr(short_trades),
        "空損益": tot(short_trades),
    }


def short_pnl_by_window(short_trades: list) -> dict:
    """
    逐窗口放空損益。用來判定「總報酬變好」究竟來自放空獲利，
    還是來自保證金佔用改變多頭部位規模所產生的副作用。
    """
    out = {}
    for name, ws, we, _tag in WINDOWS:
        sel = [t["pnl"] for t in short_trades if ws <= str(t["date"].date()) <= we]
        out[name] = round(sum(sel), 0) if sel else 0.0
    return out


def sub_returns(cap_by_date: dict) -> dict:
    out = {}
    for name, ws, we, _tag in WINDOWS:
        sub = sorted((d, v) for d, v in cap_by_date.items() if ws <= str(d.date()) <= we)
        out[name] = round((sub[-1][1] - sub[0][1]) / sub[0][1] * 100, 2) if len(sub) >= 2 else float("nan")
    return out


# ════════════════════════════════════════════════════════════════════════════════
# 統一回測引擎（四方案共用，避免 v1 各方案實作分歧產生的 confound）
# ════════════════════════════════════════════════════════════════════════════════

def build_day_map(df_scored) -> tuple:
    """
    預先切分每日資料。原本在迴圈內對全表做 df[df.date==d] 過濾，
    每天掃 76 萬列，敏感度掃描要跑十幾輪會慢到不可用。
    """
    all_dates = sorted(d for d in df_scored["date"].unique()
                       if BACKTEST_START <= str(pd.Timestamp(d).date()) <= BACKTEST_END)
    day_map = {d: g.set_index("stock_id") for d, g in df_scored.groupby("date")}
    return day_map, all_dates


def run_strategy(day_map, all_dates, regime_map, *, short_regimes=frozenset(),
                 etf_enabled=False, short_max_hold=SHORT_MAX_HOLD):
    """
    參數：
      short_regimes  : 允許放空的 regime 集合。空集合 = 不放空。
      etf_enabled    : Bear 時是否買進反向 ETF。
      short_max_hold : 空頭最長持有（交易日）。

    資金模型：
      多頭進場 cash -= entry*shares*(1+FEE)
      多頭出場 cash += exit *shares*(1-FEE-TAX)
      空頭進場 cash -= entry*shares*SHORT_MARGIN_RATE      （F4：融券保證金佔用）
      空頭出場 cash += 保證金 + 淨損益
      權益 = cash + 多頭市值 + 空頭浮動損益 + ETF 市值
    """
    cash = float(INITIAL_CAPITAL)
    long_pos, short_pos = {}, {}
    etf_value, etf_cost = 0.0, 0.0          # F5：明確追蹤 ETF 成本基礎
    long_trades, short_trades = [], []
    cap_curve, cap_by_date = [], {}

    # F6：以交易日序號計算持有天數（v1 用日曆日，口徑與 Step 1 不一致）
    day_idx = {d: i for i, d in enumerate(all_dates)}

    for date in all_dates:
        i = day_idx[date]
        regime = regime_map.get(date, "Sideways")
        day = day_map[date]
        prices = day["close"].to_dict()
        mkt_ret = float(day["market_mean_pct"].iloc[0]) if len(day) else 0.0

        # ── ETF 每日 mark-to-market ────────────────────────────────
        if etf_value > 0:
            etf_value *= (1 + (-mkt_ret - ETF_DAILY_COST))

        # ── ETF 出場：離開 Bear ────────────────────────────────────
        if etf_value > 0 and regime != "Bear":
            proceeds = etf_value * (1 - FEE_RATE - TAX_RATE)
            cash += proceeds
            short_trades.append({"date": date, "pnl": proceeds - etf_cost})  # F5：對比真實成本基礎
            etf_value, etf_cost = 0.0, 0.0

        # ── 空頭出場 ───────────────────────────────────────────────
        for sid in list(short_pos):
            pos = short_pos[sid]
            px = prices.get(sid)
            hold = i - pos["entry_idx"]
            if px is None or px <= 0:
                # 當日無報價（停牌／下市）：沿用最後成交價，超過持有上限仍強制平倉，
                # 否則保證金會被永久鎖死而扭曲後續部位規模。
                if hold < short_max_hold:
                    continue
                px = pos["last_px"]
            pos["last_px"] = px
            rise_pct = (px - pos["entry"]) / pos["entry"] * 100
            pos["floor"] = min(pos["floor"], px)
            max_profit = (pos["entry"] - pos["floor"]) / pos["entry"] * 100
            ts_exit = (max_profit >= SHORT_TS_ACTIVATE and
                       (px - pos["floor"]) / pos["entry"] * 100 >= SHORT_TS_PULLBACK)

            if (rise_pct >= pos["stop_rise"] or hold >= short_max_hold
                    or ts_exit or regime == "Bull"):
                notional = pos["entry"] * pos["shares"]
                borrow = notional * BORROW_RATE_YR * max(1, hold) / 252
                costs = (borrow + notional * FEE_RATE + px * pos["shares"] * FEE_RATE
                         + notional * TAX_RATE)
                net = (pos["entry"] - px) * pos["shares"] - costs
                cash += pos["margin"] + net
                short_trades.append({"date": date, "pnl": net})
                short_pos.pop(sid)

        # ── 多頭出場 ───────────────────────────────────────────────
        for sid in list(long_pos):
            pos = long_pos[sid]
            px = prices.get(sid)
            hold = i - pos["entry_idx"]
            if px is None or px <= 0:
                if hold < LONG_HOLD_DAYS:
                    continue
                px = pos["last_px"]
            pos["last_px"] = px
            pct = (px - pos["entry"]) / pos["entry"] * 100
            pos["peak"] = max(pos["peak"], px)
            ts_trail = ((pos["peak"] - pos["entry"]) / pos["entry"] * 100 >= LONG_TS_ACTIVATE
                        and (pos["peak"] - px) / pos["peak"] * 100 >= LONG_TS_PULLBACK)
            if (pct <= pos["stop_pct"] or hold >= LONG_HOLD_DAYS
                    or ts_trail or regime == "Bear"):
                proceeds = px * pos["shares"] * (1 - FEE_RATE - TAX_RATE)
                cash += proceeds
                long_trades.append({"date": date, "pnl": proceeds - pos["cost"]})
                long_pos.pop(sid)

        # ── 當下權益（供空頭部位規模計算，F2）─────────────────────
        equity = (cash
                  + sum(prices.get(s, p["entry"]) * p["shares"] for s, p in long_pos.items())
                  + sum((p["entry"] - prices.get(s, p["entry"])) * p["shares"] + p["margin"]
                        for s, p in short_pos.items())
                  + etf_value)

        # ── ETF 進場 ───────────────────────────────────────────────
        if etf_enabled and regime == "Bear" and etf_value <= 0:
            invest = cash * ETF_ALLOC
            if invest * (1 + FEE_RATE) <= cash and invest > 0:
                cash -= invest * (1 + FEE_RATE)
                etf_value, etf_cost = invest, invest * (1 + FEE_RATE)

        # ── 空頭進場 ───────────────────────────────────────────────
        if regime in short_regimes and len(short_pos) < SHORT_MAX_POS:
            slots = SHORT_MAX_POS - len(short_pos)
            cands = day[(~day.index.isin(short_pos)) &
                        (day["short_score_1"] >= SHORT_SCORE_THR)].nlargest(slots, "short_score_1")
            for sid, row in cands.iterrows():
                px = row["close"]
                if px is None or px <= 0:
                    continue
                shares = int(equity * SHORT_ALLOC / px / LOT) * LOT
                if shares <= 0:
                    continue
                margin = px * shares * SHORT_MARGIN_RATE
                if margin > cash:
                    continue
                cash -= margin
                short_pos[sid] = {"entry": px, "shares": shares, "margin": margin,
                                  "stop_rise": atr_stop_rise(row.get("atr18_pct")),
                                  "entry_idx": i, "floor": px, "last_px": px}

        # ── 多頭進場（F1：容量固定為 LONG_MAX_POS，不因放空而縮減）──
        if regime != "Bear" and len(long_pos) < LONG_MAX_POS:
            slots = LONG_MAX_POS - len(long_pos)
            cands = day[(~day.index.isin(long_pos)) &
                        (day["long_score_1"] >= LONG_BUY_THR)].nlargest(slots, "long_score_1")
            for sid, row in cands.iterrows():
                px = row["close"]
                if px is None or px <= 0:
                    continue
                shares = int(cash * LONG_ALLOC / px / LOT) * LOT
                if shares <= 0:
                    continue
                cost = px * shares * (1 + FEE_RATE)
                if cost > cash:
                    continue
                cash -= cost
                long_pos[sid] = {"entry": px, "shares": shares, "cost": cost,
                                 "stop_pct": atr_stop_pct(row.get("atr18_pct")),
                                 "entry_idx": i, "peak": px, "last_px": px}

        total = (cash
                 + sum(prices.get(s, p["entry"]) * p["shares"] for s, p in long_pos.items())
                 + sum((p["entry"] - prices.get(s, p["entry"])) * p["shares"] + p["margin"]
                       for s, p in short_pos.items())
                 + etf_value)
        cap_curve.append(total)
        cap_by_date[date] = total

    return cap_curve, long_trades, short_trades, cap_by_date


# ════════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-hold", default="", help="空頭持有期掃描，如 1,3,5,12")
    ap.add_argument("--regime-scan", default="", help="Regime 趨勢窗口掃描，如 3,5,7,10,15,20")
    ap.add_argument("--short-source", choices=["prod", "trained"], default="prod",
                    help="空頭分數來源：prod=生產 prob_weak（已被 plan 否決）；"
                         "trained=獨立訓練的 bias-corrected 空頭模型（生產切法，截 BACKTEST_DATE）")
    args = ap.parse_args()

    print("=" * 92)
    print("  test_short_backtest.py — Step 2（v2）：放空回測模擬")
    print(f"  區間 {BACKTEST_START} ~ {BACKTEST_END}  初始資金 {INITIAL_CAPITAL:,}"
          f"  空頭持有上限 {SHORT_MAX_HOLD} 交易日")
    print("=" * 92)

    print("\n[1/4] 載入資料與模型...")
    df = pd.read_parquet(PROD_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = filter_stocks_by_train_industries(df)
    with open(FEAT_COLS_PATH) as f:
        feature_cols = json.load(f)
    models = {d: lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgbm_model_{d}.txt"))
              for d in [1]}
    print(f"  資料 {len(df):,} 筆  特徵 {len(feature_cols)} 個")

    print("[2/4] 推論分數...")
    df_v = df.dropna(subset=["close"]).copy()
    X = df_v.reindex(columns=feature_cols).astype(np.float32)
    p = models[1].predict(X)

    # 多頭分數一律用生產模型（本實驗只替換空頭訊號來源，確保差異只來自空頭側）
    df_v["long_score_1"] = (p[:, 2] - p[:, 0]) * 100

    if args.short_source == "prod":
        # 生產 prob_weak（= prob_weak - prob_strong），plan 已否決，作為基準
        df_v["short_score_1"] = (p[:, 0] - p[:, 2]) * 100
        print("  空頭來源：prod（生產 prob_weak，已被 plan 否決之基準）")
    else:
        # 獨立訓練 bias-corrected 空頭模型：修正①樣本權重反轉、②標籤對稱鏡像。
        # 生產切法：訓練截至 config.BACKTEST_DATE，與生產多頭模型同一驗證慣例
        # （截點前為樣本內、截點後為乾淨 OOS），故與 long 端 apples-to-apples。
        cut = pd.to_datetime(BACKTEST_DATE)
        dtr = df_v.dropna(subset=["next_ret_1", "next_ret_2", "next_ret_3"]).copy()
        dtr["y_short"] = build_short_label(dtr)
        dtr["short_sw"] = build_short_sample_weight(dtr)
        train_df = dtr[dtr["date"] <= cut]
        print(f"  空頭來源：trained（生產切法，訓練截至 {cut.date()}，"
              f"{len(train_df):,} 筆，正類 {train_df['y_short'].mean()*100:.1f}%）")
        short_model = train_short_model(train_df, feature_cols)
        ps = short_model.predict_proba(X)[:, 1]
        df_v["short_score_1"] = ps * 100
        print("  ✓ 空頭模型記憶體內訓練完成（未落地），prob_short 取代 prob_weak")

    print("[3/4] 計算 Regime（讀昨日趨勢，對齊生產）...")
    regime_map = build_regime(df_v)
    cnt = {"Bull": 0, "Sideways": 0, "Bear": 0}
    for d, r in regime_map.items():
        if BACKTEST_START <= str(d.date()) <= BACKTEST_END:
            cnt[r] = cnt.get(r, 0) + 1
    tot = sum(cnt.values())
    print(f"  Bull={cnt['Bull']}天({cnt['Bull']/tot*100:.0f}%)  "
          f"Sideways={cnt['Sideways']}天({cnt['Sideways']/tot*100:.0f}%)  "
          f"Bear={cnt['Bear']}天({cnt['Bear']/tot*100:.0f}%)")

    print("\n[4/4] 執行四方案回測...\n")
    plans = {
        "A 純多頭":       dict(short_regimes=frozenset(), etf_enabled=False),
        "B 反向ETF":      dict(short_regimes=frozenset(), etf_enabled=True),
        "C 放空(熊+橫盤)": dict(short_regimes=frozenset({"Bear", "Sideways"}), etf_enabled=False),
        "D 放空(僅熊市)":  dict(short_regimes=frozenset({"Bear"}), etf_enabled=False),
    }
    day_map, all_dates = build_day_map(df_v)
    results, short_books = {}, {}
    for name, kw in plans.items():
        cap, lt, st, byd = run_strategy(day_map, all_dates, regime_map, **kw)
        results[name] = {**compute_metrics(cap, lt, st), **sub_returns(byd)}
        short_books[name] = short_pnl_by_window(st)
        r = results[name]
        print(f"  {name:<16} 累積 {r['累積報酬%']:>+9.1f}%  MDD {r['MDD%']:>7.1f}%  "
              f"多{r['多筆數']:>4}筆/空{r['空筆數']:>4}筆")

    df_r = pd.DataFrame(results).T

    print("\n" + "=" * 92)
    print("  【主要指標】")
    print("=" * 92)
    print(df_r[["累積報酬%", "MDD%", "Calmar", "多筆數", "多勝率%", "空筆數", "空勝率%", "空損益"]].to_string())

    print("\n" + "=" * 92)
    print("  【各市況子窗口報酬%】（真OOS 為訓練截點後的乾淨樣本）")
    print("=" * 92)
    print(df_r[[n for n, *_ in WINDOWS]].to_string())

    print("\n" + "=" * 92)
    print("  【放空側逐窗口損益（元）】關鍵歸因：總報酬變好是否真的來自放空獲利？")
    print("=" * 92)
    print(pd.DataFrame(short_books).T.to_string())
    print("\n  若某方案總報酬勝過 A，但放空損益為負，代表優勢並非來自放空 alpha，")
    print("  而是融券保證金佔用改變了多頭部位規模（LONG_ALLOC 依當下 cash 逐檔遞減）")
    print("  所產生的部位規模副作用——屬實作假象，不可作為採用放空的依據。")

    print("\n" + "=" * 92)
    print("  【結論】vs 方案 A")
    print("=" * 92)
    a = results["A 純多頭"]
    for name, r in results.items():
        if name == "A 純多頭":
            continue
        mdd_diff = r["MDD%"] - a["MDD%"]
        print(f"\n  {name}：")
        print(f"    全期報酬差 {r['累積報酬%'] - a['累積報酬%']:+.2f}%   "
              f"MDD {abs(mdd_diff):.2f}pp {'較深(更糟)' if mdd_diff < 0 else '較淺(更好)'}   "
              f"Calmar {r['Calmar']} vs {a['Calmar']}")
        print(f"    熊市2022 差 {r['熊市 2022'] - a['熊市 2022']:+.2f}%   "
              f"真OOS 差 {r['真OOS'] - a['真OOS']:+.2f}%")
        print(f"    放空側自身損益 {r['空損益']:+,.0f} 元（{r['空筆數']} 筆，勝率 {r['空勝率%']}%）")

    # ── 持有期敏感度掃描 ──────────────────────────────────────────
    if args.short_hold:
        holds = [int(x) for x in args.short_hold.split(",") if x.strip()]
        print("\n" + "=" * 92)
        print("  【空頭持有期敏感度】方案 D（僅熊市放空）")
        print("  Step 1 實測：放空 alpha 僅在 1~3 交易日顯著，10 日已被成本吃掉")
        print("=" * 92)
        rows = []
        for h in holds:
            cap, lt, st, byd = run_strategy(
                day_map, all_dates, regime_map, short_regimes=frozenset({"Bear"}),
                etf_enabled=False, short_max_hold=h)
            m = compute_metrics(cap, lt, st)
            rows.append({"持有期": h, "累積報酬%": m["累積報酬%"], "MDD%": m["MDD%"],
                         "Calmar": m["Calmar"], "空筆數": m["空筆數"],
                         "空勝率%": m["空勝率%"], "空損益": m["空損益"]})
        print(pd.DataFrame(rows).to_string(index=False))

    # ── Regime 窗口敏感度掃描（OFAT）────────────────────────────
    if args.regime_scan:
        wins = [int(x) for x in args.regime_scan.split(",") if x.strip()]
        print("\n" + "=" * 92)
        print("  【Regime 趨勢窗口敏感度掃描（OFAT）】")
        print(f"  生產值 REGIME_TREND_WINDOW = {REGIME_TREND_WINDOW}；窗口越短反應越快但越易 whipsaw")
        print("  兩個問題分開看：(1) 多頭防守是否改善 (2) 放空側自身損益是否由負轉正")
        print("=" * 92)
        rows = []
        for wlen in wins:
            rm = build_regime(df_v, window=wlen)
            bear_all = sum(1 for d, r in rm.items()
                           if BACKTEST_START <= str(d.date()) <= BACKTEST_END and r == "Bear")
            bear_26 = sum(1 for d, r in rm.items()
                          if "2026-06-01" <= str(d.date()) <= "2026-07-20" and r == "Bear")
            capA, ltA, stA, bydA = run_strategy(day_map, all_dates, rm)
            mA, wA = compute_metrics(capA, ltA, stA), sub_returns(bydA)
            capC, ltC, stC, bydC = run_strategy(
                day_map, all_dates, rm, short_regimes=frozenset({"Bear", "Sideways"}))
            mC, wC = compute_metrics(capC, ltC, stC), sub_returns(bydC)
            sbC = short_pnl_by_window(stC)
            capD, ltD, stD, bydD = run_strategy(
                day_map, all_dates, rm, short_regimes=frozenset({"Bear"}))
            sbD = short_pnl_by_window(stD)
            rows.append({
                "窗口": wlen, "Bear天數": bear_all, "26急跌Bear天": f"{bear_26}/33",
                "A全期%": mA["累積報酬%"], "A_MDD%": mA["MDD%"],
                "A真OOS%": wA["真OOS"], "A_26急跌%": wA["2026急跌"], "A_24急跌%": wA["2024急跌"],
                "C全期%": mC["累積報酬%"],
                "C空損益_真OOS": sbC["真OOS"], "C空損益_26急跌": sbC["2026急跌"],
                "D空損益_真OOS": sbD["真OOS"],
            })
        df_scan = pd.DataFrame(rows)
        print("\n  [多頭防守：方案 A]")
        print(df_scan[["窗口", "Bear天數", "26急跌Bear天", "A全期%", "A_MDD%",
                       "A真OOS%", "A_26急跌%", "A_24急跌%"]].to_string(index=False))
        print("\n  [放空是否有用：放空側自身損益，元]")
        print(df_scan[["窗口", "C全期%", "C空損益_真OOS", "C空損益_26急跌",
                       "D空損益_真OOS"]].to_string(index=False))
        print("\n  判讀：若「C/D 空損益_真OOS」在所有窗口皆為負，代表放空無效與 regime")
        print("  反應速度無關——換再快的偵測也救不回來，問題出在空方選股力本身。")
        print("  ⚠ 掃描僅有一次乾淨 OOS 急跌（2026）可供判斷，挑出的最佳窗口極可能是")
        print("    對該次事件過擬合，不可直接當成生產參數（同 CLAUDE.md §4.5 之戒）。")

    print(f"\n  [Note] 收盤價成交、無滑價，絕對數字僅供方案間相對比較。")
    print(f"  [Note] 四方案共用同一引擎，多頭容量一致，差異只來自放空覆蓋層。")
    print(f"  [Note] 樣本內窗口（2022~2025/07）指標會被灌水，請以「真OOS」欄為準。")


if __name__ == "__main__":
    main()
