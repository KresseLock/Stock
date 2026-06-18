# -*- coding: utf-8 -*-
"""
record_paper_trades.py — Paper Trading 掛單簿自動記錄器
========================================================
無腦自動化前瞻驗證的核心：把每日 predictions/prediction_<date>.txt 的買/賣掛單
自動寫進 paper_trading/trades.csv，並在隔日價格資料到位後，依「真實次日價格」
自動判定限價單是否成交、FIFO 配對計算實現損益%。

完全無前視偏差：訊號來自當日預測，成交判定只用「掛單日之後」才出現的價格資料。

用法：
  python paper_trading/record_paper_trades.py            # 處理最新一份預測 + 回補所有 pending
  python paper_trading/record_paper_trades.py -d 20260618 # 指定預測基準日

設計邊界（刻意不做，交給 trading_sim.py）：
  總資產 / 損益金額 / 股數 等需完整資金配置與 T+2 交割模擬者，請用 trading_sim.py。
  本工具只負責「掛單簿 + 成交判定 + 損益%」這段無腦可自動化的部分。
"""
import os
import re
import csv
import glob
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, "predictions")
PRICE_DIR = os.path.join(BASE_DIR, "data", "raw_price")
TRADES_CSV = os.path.join(BASE_DIR, "paper_trading", "trades.csv")

FEE_RATE = 0.001425      # 單邊手續費
TAX_RATE = 0.003         # 賣出證交稅
ROUND_TRIP_COST = FEE_RATE * 2 + TAX_RATE  # ≈ 0.585%

COLUMNS = ["掛單日期", "股票代號", "股票名稱", "方向", "訊號類型", "訊號值%",
           "建議掛單價", "當日收盤價", "regime", "是否成交", "成交日期", "成交價",
           "出場日期", "出場價", "出場原因", "持有天數", "損益%", "損益金額", "備註"]

# ── 解析正則 ──────────────────────────────────────────────
RE_REGIME = re.compile(r"市況[:：]\s*(\w+)")
RE_BUY = re.compile(
    r"第\s*\d+\s*檔\s*->\s*(\S+)\s+(\S+).*?D1\s*([+-]?[\d.]+)%.*?建議掛.*?->\s*([\d.]+)\)"
)
RE_SELL = re.compile(r"賣出\s*->\s*(\S+)\s+(\S+)\s*\(收盤\s*([\d.]+)")


def list_price_dates():
    """回傳已有價格資料的交易日 (YYYYMMDD)，升序。"""
    dates = []
    for p in glob.glob(os.path.join(PRICE_DIR, "*_price.csv")):
        name = os.path.basename(p)
        d = name.split("_")[0]
        if d.isdigit() and len(d) == 8:
            dates.append(d)
    return sorted(dates)


def next_trading_day(base_date, price_dates):
    """掛單於 base_date 收盤後產生 → 次一交易日開盤執行。"""
    for d in price_dates:
        if d > base_date:
            return d
    return None


def load_price_day(date):
    """讀取某交易日價格，回傳 {代號: {open, high, low, close}}。"""
    path = os.path.join(PRICE_DIR, f"{date}_price.csv")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("證券代號") or "").strip()
            if not code:
                continue
            def num(key):
                v = (row.get(key) or "").replace(",", "").strip()
                try:
                    return float(v)
                except ValueError:
                    return None
            out[code] = {"open": num("開盤價"), "high": num("最高價"),
                         "low": num("最低價"), "close": num("收盤價")}
    return out


def parse_prediction(base_date):
    """解析 prediction_<date>.txt，回傳 (regime, [買單], [賣單])。"""
    path = os.path.join(PRED_DIR, f"prediction_{base_date}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到預測檔: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = RE_REGIME.search(text)
    regime = m.group(1) if m else ""
    buys = [{"code": c, "name": n, "d1": float(d1), "limit": float(lim)}
            for c, n, d1, lim in RE_BUY.findall(text)]
    sells = [{"code": c, "name": n, "close": float(cl)}
             for c, n, cl in RE_SELL.findall(text)]
    return regime, buys, sells


def read_trades():
    if not os.path.exists(TRADES_CSV):
        return []
    with open(TRADES_CSV, "r", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_trades(rows):
    with open(TRADES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def blank_row():
    return {k: "" for k in COLUMNS}


RE_SRC = re.compile(r"來源預測\s*([\d-]+)")


def src_base_date(row):
    """從備註取回該列來源預測的基準日 (YYYYMMDD)。"""
    m = RE_SRC.search(row.get("備註", ""))
    return m.group(1).replace("-", "") if m else ""


def already_recorded(rows, base_date, code, direction):
    """以『來源預測基準日 + 代號 + 方向』為穩定去重鍵（掛單日會由空→實際，不能當鍵）。"""
    return any(src_base_date(r) == base_date and r["股票代號"] == code
               and r["方向"] == direction for r in rows)


def fmt_date(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def main():
    ap = argparse.ArgumentParser(description="Paper Trading 掛單簿自動記錄器")
    ap.add_argument("-d", "--date", default=None,
                    help="預測基準日 YYYYMMDD（預設：最新一份預測）")
    args = ap.parse_args()

    # 決定要處理的預測基準日
    if args.date:
        base_date = args.date.replace("-", "")
    else:
        preds = sorted(glob.glob(os.path.join(PRED_DIR, "prediction_*.txt")))
        if not preds:
            print("[錯誤] predictions/ 沒有任何預測檔。")
            return
        base_date = os.path.basename(preds[-1]).replace("prediction_", "").replace(".txt", "")

    price_dates = list_price_dates()
    rows = read_trades()
    # 移除範例列
    rows = [r for r in rows if "範例" not in r.get("備註", "") and "初始基準" not in r.get("備註", "")]

    # ── 步驟 1：附加當日預測的新掛單（掛單日 = 次一交易日，未到則先留空）──
    regime, buys, sells = parse_prediction(base_date)
    exec_day = next_trading_day(base_date, price_dates)        # 次一交易日；未到則 None
    exec_str = fmt_date(exec_day) if exec_day else ""
    src_note = f"來源預測 {fmt_date(base_date)}"
    added = 0
    for b in buys:
        if already_recorded(rows, base_date, b["code"], "買"):
            continue
        r = blank_row()
        r.update({"掛單日期": exec_str, "股票代號": b["code"], "股票名稱": b["name"],
                  "方向": "買", "訊號類型": "D1", "訊號值%": f"{b['d1']:+.1f}",
                  "建議掛單價": f"{b['limit']:.2f}", "regime": regime, "備註": src_note})
        rows.append(r)
        added += 1
    for s in sells:
        if already_recorded(rows, base_date, s["code"], "賣"):
            continue
        r = blank_row()
        r.update({"掛單日期": exec_str, "股票代號": s["code"], "股票名稱": s["name"],
                  "方向": "賣", "訊號類型": "開盤賣", "當日收盤價": f"{s['close']:.2f}",
                  "regime": regime, "備註": src_note})
        rows.append(r)
        added += 1

    # ── 步驟 2：回補所有 pending 掛單的成交判定 ────────────
    # 成交一律以「來源預測基準日之後的第一個交易日」價格判定，杜絕用預測前/當日價格搓合（前視偏差）。
    resolved = 0
    for r in rows:
        if r["是否成交"]:  # 已判定過
            continue
        base = src_base_date(r)
        ed = next_trading_day(base, price_dates) if base else None
        if ed is None:                       # 次一交易日尚未產生 → 維持 pending
            continue
        r["掛單日期"] = fmt_date(ed)          # 次一交易日已到，補上實際執行日
        prices = load_price_day(ed)
        if prices is None:
            continue
        px = prices.get(r["股票代號"])
        if px is None:
            r["是否成交"] = "N"
            r["備註"] = (r["備註"] + " | 無當日價格").strip(" |")
            resolved += 1
            continue
        if r["方向"] == "買":
            limit = float(r["建議掛單價"])
            # 限價買：當日最低 <= 掛單價 才成交；開盤即低於掛價則以開盤成交（更優）
            if px["low"] is not None and px["low"] <= limit:
                fill = min(limit, px["open"]) if px["open"] else limit
                r["是否成交"], r["成交日期"], r["成交價"] = "Y", r["掛單日期"], f"{fill:.2f}"
            else:
                r["是否成交"] = "N"
        else:  # 開盤賣
            if px["open"]:
                r["是否成交"], r["成交日期"], r["成交價"] = "Y", r["掛單日期"], f"{px['open']:.2f}"
            else:
                r["是否成交"] = "N"
        resolved += 1

    # ── 步驟 3：FIFO 配對賣單 → 計算買進持倉的實現損益% ────
    pnl_filled = 0
    open_buys = {}  # code -> list of buy rows (已成交、未出場)，依掛單日排序
    for r in sorted(rows, key=lambda x: (x["掛單日期"], x["方向"])):
        code = r["股票代號"]
        if r["方向"] == "買" and r["是否成交"] == "Y":
            open_buys.setdefault(code, []).append(r)
        elif r["方向"] == "賣" and r["是否成交"] == "Y":
            lots = open_buys.get(code, [])
            sell_price = float(r["成交價"])
            for buy in lots:
                if buy["出場日期"]:  # 已配對過
                    continue
                buy_price = float(buy["成交價"])
                ret = (sell_price / buy_price - 1 - ROUND_TRIP_COST) * 100
                buy["出場日期"] = r["成交日期"]
                buy["出場價"] = f"{sell_price:.2f}"
                buy["出場原因"] = "訊號賣出(開盤)"
                # 持有天數（交易日差）
                try:
                    bi = price_dates.index(buy["成交日期"].replace("-", ""))
                    si = price_dates.index(r["成交日期"].replace("-", ""))
                    buy["持有天數"] = str(si - bi)
                except ValueError:
                    pass
                buy["損益%"] = f"{ret:+.2f}"
                pnl_filled += 1
                break  # 一張賣單配一筆買進（簡化：每檔每日一單）

    # 依掛單日期、方向排序輸出
    rows.sort(key=lambda x: (x["掛單日期"], x["股票代號"], x["方向"]))
    write_trades(rows)

    print(f"[完成] 預測基準日 {fmt_date(base_date)} → 執行日 {exec_str or '(次一交易日尚未產生，pending)'}")
    print(f"  regime={regime} | 新增掛單 {added} 筆 | 本次判定成交 {resolved} 筆 | 新配對損益 {pnl_filled} 筆")
    print(f"  掛單簿: {TRADES_CSV}（共 {len(rows)} 列）")


if __name__ == "__main__":
    main()
