# -*- coding: utf-8 -*-
"""
test_fortune_ported_signals.py — 從 fortune-main 移植候選訊號的「有沒有 alpha」輕量回測
================================================================================
目的：把 fortune-main (港股系統) 幾個頭條策略「思想」移植到台股，用我方真實
      features_combined.parquet 量測其對未來報酬 (next_ret_1/3) 的預測力，
      在「投入重訓成本前」先篩掉沒有 edge 的候選。

為何是輕量回測 (不重訓)：
  正式驗證一個特徵要 feature_engineering→train→inference→trading_sim (數小時)。
  本測試改為「訊號 vs 未來報酬」的橫斷面 IC / 條件勝率 / 分位差診斷，直接回答
  『這個訊號本身帶不帶正交 alpha』。有 edge 才值得升級成正式特徵重訓。

移植的候選 (對應 fortune-main/README.md 章節)：
  S1  Z-Score 抄底 (§1.5 異常檢測)：宣稱「價格異常+當日下跌→抄底 5日勝率 72%」
  S2  市場情緒過濾 (§1.4)：驗證我方既有 market_breadth 的分層 edge (fortune 宣稱 +8.7% acc)
  S3  日曆效應 (§1.3 特徵表)：台指結算日(每月第三個週三)/月份/星期，我方完全沒有
  S4  高信心虧損尾 (§1.3 風險提示)：fortune 警告高信心錯誤可虧 -73%，檢查我方 label=2 尾部

無前視：所有訊號在 t 日僅用 t 日(收盤已知)及過去資料；標的 next_ret_N 為 t 之後的未來報酬。

執行：python tests/test_fortune_ported_signals.py
"""
import os
import sys

import numpy as np
import pandas as pd

try:  # Windows 主控台預設 cp950，強制 UTF-8 避免中文亂碼
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

PARQUET = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
RET_CLIP = 0.25   # 報酬截尾 (±25%)，壓低低價股資料雜訊對均值/IC 的扭曲


# ══════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════
def _clip_ret(s):
    return s.clip(-RET_CLIP, RET_CLIP)


def winrate(ret):
    ret = ret.dropna()
    return (ret > 0).mean() if len(ret) else np.nan


def summarize(ret, label):
    ret = _clip_ret(ret.dropna())
    n = len(ret)
    if n == 0:
        return f"{label:32s} n=0"
    return (f"{label:32s} n={n:7d}  勝率={ (ret>0).mean():6.2%}  "
            f"均值={ret.mean():+7.3%}  中位={ret.median():+7.3%}")


def rank_ic(signal, fwd, by_date):
    """逐日橫斷面 Spearman RankIC，回傳 (平均 IC, IC t-stat, 覆蓋天數)。"""
    df = pd.DataFrame({"s": signal, "f": fwd, "d": by_date}).dropna()
    ics = []
    for _, g in df.groupby("d"):
        if len(g) >= 20 and g["s"].nunique() > 1:
            ics.append(g["s"].corr(g["f"], method="spearman"))
    ics = pd.Series(ics).dropna()
    if len(ics) < 5:
        return np.nan, np.nan, len(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std(ddof=1) > 0 else np.nan
    return ics.mean(), t, len(ics)


def decile_spread(signal, fwd):
    """依訊號分 10 組，回傳 (最高組勝率, 最低組勝率, 最高-最低均值差)。"""
    df = pd.DataFrame({"s": signal, "f": _clip_ret(fwd)}).dropna()
    if df["s"].nunique() < 10 or len(df) < 1000:
        return None
    try:
        df["q"] = pd.qcut(df["s"].rank(method="first"), 10, labels=False)
    except ValueError:
        return None
    top, bot = df[df.q == 9]["f"], df[df.q == 0]["f"]
    return (top > 0).mean(), (bot > 0).mean(), top.mean() - bot.mean()


# ══════════════════════════════════════════════════════
# 載入
# ══════════════════════════════════════════════════════
def load():
    if not os.path.exists(PARQUET):
        print(f"[SKIP] 找不到 {PARQUET}")
        sys.exit(0)
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    print(f"載入 {len(df):,} 列 / {df['stock_id'].nunique()} 檔 / "
          f"{df['date'].min():%Y-%m-%d}~{df['date'].max():%Y-%m-%d}\n")
    return df


# ══════════════════════════════════════════════════════
# S1: Z-Score 抄底 (fortune §1.5)
# ══════════════════════════════════════════════════════
def s1_zscore_dip(df):
    print("=" * 78)
    print("S1  Z-Score 抄底 (fortune 宣稱: 價格異常+當日下跌 → 5日勝率 72%)")
    print("=" * 78)
    g = df.groupby("stock_id")["close"]
    mean20 = g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    std20 = g.transform(lambda s: s.rolling(20, min_periods=20).std())
    z = (df["close"] - mean20) / std20  # t 日收盤已知，無前視
    fwd = df["next_ret_3"]

    base = summarize(fwd, "  [基準] 全體")
    oversold = summarize(fwd[(z < -2)], "  z<-2 (超賣)")
    dip = summarize(fwd[(z < -2) & (df["ret1"] < 0)], "  z<-2 且當日下跌 (抄底)")
    overshoot = summarize(fwd[(z > 2)], "  z>+2 (超買, 對照)")
    print(base); print(oversold); print(dip); print(overshoot)

    ic, t, nd = rank_ic(-z, fwd, df["date"])   # 負 z (越超賣越看多) 對未來報酬
    print(f"  RankIC(-z vs next_ret_3): {ic:+.4f}  t={t:+.2f}  ({nd} 天)")
    print("  解讀：fortune 的『72% 勝率二元抄底』不轉移 (門檻集勝率僅微幅>基準且均值仍負);")
    print("        但 -z 連續值對次日報酬有穩健正 RankIC → 均值回歸 alpha 存在，")
    print("        宜當『連續特徵』餵模型 (呼應共整合殘差方向)，勿當二元進場規則\n")
    return {"抄底勝率": winrate(_clip_ret(fwd[(z < -2) & (df['ret1'] < 0)])),
            "基準勝率": winrate(_clip_ret(fwd)), "z_rankic": ic}


# ══════════════════════════════════════════════════════
# S2: 市場情緒過濾 (fortune §1.4) — 驗證我方既有 breadth
# ══════════════════════════════════════════════════════
def s2_market_sentiment(df):
    print("=" * 78)
    print("S2  市場情緒分層 (驗證我方既有 market_breadth 的時序 edge; fortune 宣稱過濾 +8.7% acc)")
    print("=" * 78)
    # breadth 是市場級 (同日全股同值) → 只能做「時序」而非橫斷面分層。
    # 用 t 日收盤已知的 breadth_ma5 分環境，看『之後』全體平均報酬是否隨環境變化 (無前視)。
    b = df["market_breadth_ma5"]
    fwd = df["next_ret_3"]
    tiers = [("extreme_bear b<0.20", b < 0.20),
             ("bear 0.20~0.30", (b >= 0.20) & (b < 0.30)),
             ("weak 0.30~0.40", (b >= 0.30) & (b < 0.40)),
             ("normal 0.40~0.55", (b >= 0.40) & (b < 0.55)),
             ("greed b>=0.55", b >= 0.55)]
    for name, mask in tiers:
        print(summarize(fwd[mask], f"  全體 @ {name}"))
    print("  解讀：若健康環境(normal/greed)整體勝率明顯高於 bear → 印證我方 REGIME 依市況")
    print("        調整買入門檻/檔數的方向正確 (低迷市況少進場)\n")


# ══════════════════════════════════════════════════════
# S3: 日曆效應 (fortune §1.3) — 我方完全沒有
# ══════════════════════════════════════════════════════
def _third_wednesday(dates):
    """台指期/選擇權結算日 = 每月第三個星期三。回傳每列對應該月結算日。"""
    d = pd.DatetimeIndex(dates)
    firsts = d.to_period("M").to_timestamp()
    # 該月第一天的星期 (Mon=0..Sun=6)，週三=2
    dow_first = firsts.dayofweek
    days_to_first_wed = (2 - dow_first) % 7
    third_wed = firsts + pd.to_timedelta(days_to_first_wed + 14, unit="D")
    return third_wed


def s3_calendar(df):
    print("=" * 78)
    print("S3  日曆效應 (台指結算=每月第三個週三 / 星期 / 月份; 我方完全沒有此類特徵)")
    print("=" * 78)
    d = df["date"]
    fwd1, fwd3 = df["next_ret_1"], df["next_ret_3"]

    # (a) 星期效應
    dow_names = ["週一", "週二", "週三", "週四", "週五"]
    print("  -- 星期效應 (next_ret_1) --")
    for i, nm in enumerate(dow_names):
        print(summarize(fwd1[d.dt.dayofweek == i], f"    {nm}"))

    # (b) 結算日效應
    settle = _third_wednesday(d.values)
    days_to = (settle - d).dt.days
    print("  -- 台指結算週效應 (next_ret_3) --")
    print(summarize(fwd3[days_to.between(1, 3)], "    結算日前 1~3 交易日"))
    print(summarize(fwd3[days_to == 0], "    結算日當日"))
    print(summarize(fwd3[days_to.between(-3, -1)], "    結算日後 1~3 交易日"))
    print(summarize(fwd3[~days_to.between(-3, 3)], "    非結算週"))

    # (c) 月份效應 (作為橫斷面訊號的 IC 沒意義，看均值分佈)
    print("  -- 月份效應 (next_ret_3 均值, 台股習見 Q1 作夢/Q3 淡季) --")
    m = d.dt.month
    line = "    "
    for mm in range(1, 13):
        r = _clip_ret(fwd3[m == mm]).mean()
        line += f"{mm:>2d}月:{r:+5.2%}  "
        if mm == 6:
            line += "\n    "
    print(line)
    print("  解讀：若某星期/結算窗口的勝率或均值系統性偏離 → 加成 0-lookahead 正交特徵，成本極低\n")


# ══════════════════════════════════════════════════════
# S4: 高信心虧損尾 (fortune §1.3 風險提示)
# ══════════════════════════════════════════════════════
def s4_tail_risk(df):
    print("=" * 78)
    print("S4  『看似強勢』setup 的虧損尾檢查 (fortune 警告: 高信心錯誤可虧 -73%, 必設止損)")
    print("=" * 78)
    # 不能用 label (=未來報酬定義, 會前視)。改用 t 日已知的技術面『強勢』代理:
    #   動能強 (ret20 前 25%) 且站上月線 (close>ma20)。這是實盤真能在 t 日選出的集合。
    fwd = df["next_ret_3"]
    up_trend = df["close"] > df["ma20"]
    mom_top = df["ret20"] >= df["ret20"].quantile(0.75)
    strong = up_trend & mom_top
    r = fwd[strong].dropna()
    ra = fwd.dropna()
    print(f"  『強勢動能』setup (close>ma20 且 ret20 前25%) n={len(r):,}  (全體 n={len(ra):,})")
    print(f"    平均報酬 {r.mean():+.3%} (全體 {ra.mean():+.3%})  勝率 {(r>0).mean():.2%}")
    print(f"    虧損 <= -5%: {(r<=-0.05).mean():.2%}  <= -10%: {(r<=-0.10).mean():.2%}  "
          f"<= -20%: {(r<=-0.20).mean():.2%}")
    print(f"    最大單筆虧損 {r.min():+.2%}  |  第 1 百分位 {r.quantile(0.01):+.2%}")
    print("  解讀：即便是技術面最強勢的 setup，左尾仍肥厚 → 印證我方 ATR 停損必要 (§4.5-3)\n")


def main():
    df = load()
    s1_zscore_dip(df)
    s2_market_sentiment(df)
    s3_calendar(df)
    s4_tail_risk(df)
    print("=" * 78)
    print("結論見上；有顯著 edge 的候選才升級為正式特徵 (改 feature_engineering.py) 並跑完整重訓回測。")
    print("=" * 78)


if __name__ == "__main__":
    main()
