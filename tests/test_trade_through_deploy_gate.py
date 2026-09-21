# -*- coding: utf-8 -*-
"""
test_trade_through_deploy_gate.py — 「穿價才成交 + 換用 oldselect 參數」是否勝過原系統
======================================================================================
背景：EXPERIMENTS_PENDING.md「已結案檢定：穿價才成交」留下兩件待決：
  (1) config.FILL_REQUIRE_TRADE_THROUGH 預設是否改為 True
  (2) 生產風控參數是否換成 best_trading_params_candidate.oldselect_20260828.json
原則：先在 tests/ 驗證「改完是否比原本好」，贏了才動生產系統。

受測系統：
  原系統   = configs/best_trading_params.json（現行部署，調參至 2026-07-23）
  候選系統 = configs/best_trading_params_candidate.oldselect_20260828.json（調參至 2025-12-31）
各在兩種成交規則、兩個模型實現下，跑裁判區 9 條起點偏移路徑（終點固定）。

判準：
  * 主判準（決定動不動生產）：成交規則固定為「穿價才成交」（較貼近實盤），候選系統對原系統
        報酬中位數勝 且 回撤中位數勝 且 逐路徑雙贏 >= 5/9，
    並且在**每一個模型實現**上都成立——EXPERIMENTS_PENDING.md 鐵律 #8：只補了 5 支已下市股的財報，
    裁判區報酬中位數就差 +15~25pp，單一實現上的勝負不構成證據。
  * 觸價規則下的同規則對照：只作參考。
  * 跨規則比較（候選@穿價 vs 原系統@觸價）：只印出，不作判準。成交規則是量尺，換量尺不會改變
    實盤績效，只會改變回測量出來的數字，拿兩把量尺的讀數互比沒有意義。

關於待決 (1)：開關只影響回測怎麼算成交，inference.py 與實盤下單完全不受影響，
所以它不能用「績效有沒有變好」來決定。本測試對它能回答的是「各系統的回測報酬有多少建立在
『剛好碰到限價就成交』上」（下方「成交規則依賴度」一節）。

單向有效：原系統參數看過裁判區（2026-01-02 起），比較天然偏袒原系統。候選勝出＝強證據；
候選落敗不能證明候選較差，但依「贏了才動」的原則，落敗即維持原系統。

隔離保證：
  * 模型實現：in-process patch trading_sim.MODEL_DIR（run_simulation 每次呼叫才載入模型），不動 models/。
  * 成交規則：in-process patch config.FILL_REQUIRE_TRADE_THROUGH（run_simulation 每次呼叫才 import），不寫 config.py。
  * configs/ 全程唯讀；只寫 reports/ 下的執行記錄。

執行：
  python tests/test_trade_through_deploy_gate.py                  # 72 次回測
  python tests/test_trade_through_deploy_gate.py --end 2026-09-10  # 釘住終點以便重現
"""

import argparse
import contextlib
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import trading_sim as ts
import optimize_trading_params as opt
from config import MAX_POSITIONS
from validate_candidate_params import OFFSETS, trading_days, _median
from scripts.utils import start_tee_log, model_fingerprint

INCUMBENT = os.path.join(ROOT_DIR, "configs", "best_trading_params.json")
CANDIDATE = os.path.join(ROOT_DIR, "configs", "best_trading_params_candidate.oldselect_20260828.json")
REALIZATIONS = {
    "生產模型": os.path.join(ROOT_DIR, "models"),
    "19:50實現": os.path.join(ROOT_DIR, "models", "_candidate_backfill_20260910"),
}
RULES = {"觸價即成交": False, "穿價才成交": True}
MAIN_RULE = "穿價才成交"
ZONE_FROM = "2026-01-02"
MODEL_FILES = ("lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json")


def load_params(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["best_params"], (d.get("date_range") or "").split(" to ")[-1].strip()


def run(params, start, end, capital):
    """回傳 (報酬%, 回撤%)，回撤為負值。與優化器、markup 檢定同一支評分程式碼。"""
    with contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8")):
        score, _, ret, mdd = opt.run_simulation_scoring(start, end, params, capital, MAX_POSITIONS)
    if score == -999.0:   # run_simulation_scoring 會吞掉例外，必須自己擋
        raise RuntimeError(f"回測失敗：{start} ~ {end}")
    return ret, -mdd * 100.0


def compare(base, cand):
    """cand 相對 base：(報酬中位數差, 回撤中位數差, 報酬贏, 回撤贏, 雙贏)。"""
    dr = _median([r for r, _ in cand]) - _median([r for r, _ in base])
    dm = _median([m for _, m in cand]) - _median([m for _, m in base])
    wr = sum(rc > rb for (rb, _), (rc, _) in zip(base, cand))
    wm = sum(mc > mb for (_, mb), (_, mc) in zip(base, cand))
    both = sum(rc > rb and mc > mb for (rb, mb), (rc, mc) in zip(base, cand))
    return dr, dm, wr, wm, both


def fmt_cmp(res):
    dr, dm, wr, wm, both = res
    return f"報酬 {dr:+8.2f}pp（贏 {wr}/9）  回撤 {dm:+7.2f}pp（贏 {wm}/9）  雙贏 {both}/9"


def main():
    ap = argparse.ArgumentParser(description="穿價才成交 + oldselect 參數 vs 原系統（隔離把關）")
    ap.add_argument("-c", "--capital", type=int, default=2000000)
    ap.add_argument("--end", type=str, default=None, help="裁判區終點（預設為特徵檔最新交易日）")
    args = ap.parse_args()

    for label, path in (("原系統參數", INCUMBENT), ("候選系統參數", CANDIDATE)):
        if not os.path.exists(path):
            print(f"[錯誤] 找不到{label}：{path}")
            return 2
    for label, mdir in REALIZATIONS.items():
        missing = [n for n in MODEL_FILES if not os.path.exists(os.path.join(mdir, n))]
        if missing:
            print(f"[錯誤] 模型實現「{label}」缺檔：{missing}")
            return 2

    systems = {"原系統": load_params(INCUMBENT), "候選系統": load_params(CANDIDATE)}
    cand_end = systems["候選系統"][1]
    if not cand_end or cand_end >= ZONE_FROM:
        print(f"[錯誤] 候選參數調參至 {cand_end!r}，未早於裁判區起點 {ZONE_FROM}，不是乾淨候選。")
        return 2

    days = [d for d in trading_days() if d >= ZONE_FROM and (args.end is None or d <= args.end)]
    if len(days) <= max(OFFSETS):
        print(f"[錯誤] 裁判區只有 {len(days)} 個交易日，不足以跑 {len(OFFSETS)} 條路徑。")
        return 2
    paths = [(days[o], days[-1]) for o in OFFSETS]

    log_path = start_tee_log("test_trade_through_deploy_gate")
    print("=" * 92)
    print("  穿價才成交 + oldselect 參數 vs 原系統（tests/ 隔離把關，不動 config.py / configs/ / models/）")
    print("=" * 92)
    print(f"  原系統   : {os.path.basename(INCUMBENT)}（調參至 {systems['原系統'][1]}，看過裁判區 → 單向有效）")
    print(f"  候選系統 : {os.path.basename(CANDIDATE)}（調參至 {cand_end}）")
    print(f"  模型實現 : " + "、".join(f"{k}={os.path.relpath(v, ROOT_DIR)}（指紋 {model_fingerprint(v)}）"
                                       for k, v in REALIZATIONS.items()))
    print(f"  裁判區   : {days[0]} ~ {days[-1]}（{len(days)} 個交易日），{len(paths)} 條起點偏移路徑，終點固定")
    print(f"  資金     : {args.capital:,}")
    print(f"  執行記錄 : {os.path.relpath(log_path, ROOT_DIR)}")
    print("=" * 92)

    res = {}
    orig_model_dir, orig_rule = ts.MODEL_DIR, config.FILL_REQUIRE_TRADE_THROUGH
    try:
        for real, mdir in REALIZATIONS.items():
            ts.MODEL_DIR = mdir
            for rule, flag in RULES.items():
                config.FILL_REQUIRE_TRADE_THROUGH = flag
                for sname, (params, _) in systems.items():
                    rows = []
                    for start, end in paths:
                        rows.append(run(params, start, end, args.capital))
                    res[(real, rule, sname)] = rows
                    rs = [r for r, _ in rows]
                    ms = [m for _, m in rows]
                    print(f"  [{real} | {rule} | {sname:<4}] 報酬中位 {_median(rs):+8.2f}%  回撤中位 {_median(ms):7.2f}%"
                          f"  報酬範圍 {min(rs):+.2f} ~ {max(rs):+.2f}（級距 {max(rs) - min(rs):.1f}pp）", flush=True)
    finally:
        ts.MODEL_DIR, config.FILL_REQUIRE_TRADE_THROUGH = orig_model_dir, orig_rule

    verdicts = []
    for real in REALIZATIONS:
        print("\n" + "=" * 92)
        print(f"  模型實現：{real}")
        print("=" * 92)
        print("  同規則對照（候選系統相對原系統）")
        for rule in RULES:
            c = compare(res[(real, rule, "原系統")], res[(real, rule, "候選系統")])
            tag = "★ 主判準" if rule == MAIN_RULE else "  參考  "
            print(f"    {tag} [{rule}] {fmt_cmp(c)}")
            if rule == MAIN_RULE:
                dr, dm, _, _, both = c
                verdicts.append((real, dr > 0 and dm > 0 and both >= 5, c))

        print("  成交規則依賴度（同一系統，觸價 → 穿價；報酬掉越多＝越依賴剛好觸價就成交）")
        for sname in systems:
            c = compare(res[(real, "觸價即成交", sname)], res[(real, "穿價才成交", sname)])
            print(f"             [{sname:<4}] {fmt_cmp(c)}")

        c = compare(res[(real, "觸價即成交", "原系統")], res[(real, MAIN_RULE, "候選系統")])
        print("  跨規則（候選@穿價 vs 原系統@觸價）—— 兩把量尺互比，只列出、不作判準")
        print(f"                    {fmt_cmp(c)}")

    passed = all(ok for _, ok, _ in verdicts)
    print("\n" + "=" * 92)
    print("  結論")
    print("=" * 92)
    for real, ok, (dr, dm, _, _, both) in verdicts:
        why = [t for t, good in (("報酬中位數未勝", dr > 0), ("回撤中位數未勝", dm > 0),
                                 (f"逐路徑雙贏僅 {both}/9", both >= 5)) if not good]
        print(f"  [{real}] {'✅ 通過' if ok else '❌ 未通過：' + '、'.join(why)}")
    if passed:
        print("\n  ✅ 候選系統在所有模型實現上、穿價規則下皆勝過原系統 → 可進入部署評估（仍須使用者確認才動生產）。")
    else:
        print("\n  ❌ 候選系統未在所有模型實現上勝過原系統 → 依「贏了才動」原則，維持原系統，不動生產。")
        print("     ※ 單向有效：原系統看過裁判區，落敗不代表候選較差，只代表沒有足夠證據換掉原系統。")
    print("  ※ FILL_REQUIRE_TRADE_THROUGH 是回測量尺、不影響實盤，不以本表績效決定是否採用；見上方「成交規則依賴度」。")
    print("=" * 92)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
