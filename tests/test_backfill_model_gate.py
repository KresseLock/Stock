# -*- coding: utf-8 -*-
"""訓練資料回補至 2017-12-18 的模型把關（EXPERIMENTS_PENDING.md 回補章節 Step 6）。

比較的是**模型**，不是風控參數：`configs/best_trading_params.json` 全程不動，
唯一變因是 `models/` 裡的 LightGBM 檔（現行 = 訓練集 2020-01-02 起／候選 = 2017-12-18 起，
兩者訓練終點同為 2023-11-24，由 config.py 的 TRAIN/VALID_SPLIT_RATIO 反算固定）。

紀律沿用既有把關框架：
  - 終點固定、只挪起點的 9 條路徑（方法論鐵律 #4；禁止切段，CLAUDE.md §4.5 #5）。
  - 判準：報酬與回撤中位數雙贏，且逐路徑雙贏 >= 5/9（§4.5 #4）。
  - 單向有效：`best_trading_params.json` 是拿現行模型的預測調出來的，天然偏袒現行，
    故候選勝出＝強證據，候選落敗不可據此判死回補這個方向。

用法：python tests/test_backfill_model_gate.py [-c 資金]
"""

import argparse
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 直接複用體檢器的回測封裝：子行程編碼已在該處釘死 cp950（該檔坑 #11），別自行重寫
from scripts.validate_candidate_params import OFFSETS, run_backtest, trading_days, _median
from scripts.utils import start_tee_log

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILES = ("lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json")

# 兩組模型的來源目錄
from config import DATA_DIR   # 資料路徑唯一來源：config.py § 0
INCUMBENT = os.path.join(DATA_DIR, "_backup_pre_backfill_20260910", "models_backup")
CANDIDATE = os.path.join(MODEL_DIR, "_candidate_backfill_20260910")

# 共同乾淨 OOS 起點：兩組模型都截斷於 BACKTEST_DATE=20250801，該日之後對雙方皆為樣本外
OOS_FROM = "2025-08-04"


def install(src):
    """把一組模型檔搬進 models/ 根目錄（trading_sim.py 只認那裡）。"""
    for name in MODEL_FILES:
        path = os.path.join(src, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少模型檔：{path}")
        shutil.copy2(path, os.path.join(MODEL_DIR, name))


def main():
    ap = argparse.ArgumentParser(description="回補訓練資料的模型把關（多起點對照）")
    ap.add_argument("-c", "--capital", type=int, default=2000000)
    args = ap.parse_args()

    days = [d for d in trading_days() if d >= OOS_FROM]
    if len(days) <= max(OFFSETS):
        print(f"[錯誤] {OOS_FROM} 之後只有 {len(days)} 個交易日，不足以跑 {len(OFFSETS)} 條路徑。")
        return 1
    zone_end = days[-1]
    paths = [(days[o], zone_end) for o in OFFSETS]

    log_path = start_tee_log("test_backfill_model_gate")
    print("=" * 78)
    print("  回補訓練資料（2020-01-02 → 2017-12-18 起）模型把關")
    print("=" * 78)
    print(f"  現行     : {os.path.relpath(INCUMBENT, BASE_DIR)}")
    print(f"  候選     : {os.path.relpath(CANDIDATE, BASE_DIR)}")
    print(f"  裁判區   : {days[0]} ~ {zone_end}（{len(days)} 個交易日，對兩組模型都乾淨）")
    print(f"  路徑     : {len(OFFSETS)} 條，起點各往後挪 {'/'.join(map(str, OFFSETS))} 個交易日，終點固定")
    print(f"  風控參數 : configs/best_trading_params.json（全程不動＝單一變因）")
    print(f"  資金     : {args.capital:,}")
    print(f"  執行記錄 : {os.path.relpath(log_path, BASE_DIR)}")
    print("=" * 78)

    res = {}
    # 生產 models/ 未必等於候選副本（重訓會換實現，見 EXPERIMENTS_PENDING.md 鐵律 #8），
    # 故還原「測試前的 models/」，而不是固定裝回候選副本
    snapshot = os.path.join(MODEL_DIR, "_pre_gate_snapshot")
    os.makedirs(snapshot, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(os.path.join(MODEL_DIR, name), snapshot)
    try:
        for label, src in (("現行", INCUMBENT), ("候選", CANDIDATE)):
            install(src)
            print(f"\n[{label}] 模型已就位，開始 {len(paths)} 條路徑")
            rows = []
            for i, (start, end) in enumerate(paths, 1):
                ret, mdd = run_backtest(start, end, args.capital)
                rows.append((start, ret, mdd))
                print(f"  路徑 {i}/{len(paths)}  {start} ~ {end}  報酬 {ret:+8.2f}%  回撤 {mdd:7.2f}%")
            res[label] = rows
    finally:
        # 落敗時的回退由 Step 0 的備份負責，不在此處代為決定
        install(snapshot)
        shutil.rmtree(snapshot)
        print("\n[還原] models/ 已放回測試前的模型")

    print("\n" + "=" * 78)
    print("  逐路徑對照（回撤為負值，越接近 0 越好）")
    print("=" * 78)
    print(f"  {'起點':<12}{'現行報酬':>11}{'候選報酬':>11}{'報酬差':>10}"
          f"{'現行回撤':>11}{'候選回撤':>11}{'回撤差':>10}  雙贏")
    both = 0
    for (start, r0, m0), (_, r1, m1) in zip(res["現行"], res["候選"]):
        win = (r1 > r0) and (m1 > m0)          # 回撤為負，較大＝較淺
        both += win
        print(f"  {start:<12}{r0:>10.2f}%{r1:>10.2f}%{r1 - r0:>+9.2f}pp"
              f"{m0:>10.2f}%{m1:>10.2f}%{m1 - m0:>+9.2f}pp  {'✅' if win else '❌'}")

    med = {k: (_median([r for _, r, _ in v]), _median([m for _, _, m in v])) for k, v in res.items()}
    print(f"\n  報酬中位數  現行 {med['現行'][0]:+.2f}%  候選 {med['候選'][0]:+.2f}%"
          f"  ({med['候選'][0] - med['現行'][0]:+.2f}pp)")
    print(f"  回撤中位數  現行 {med['現行'][1]:.2f}%  候選 {med['候選'][1]:.2f}%"
          f"  ({med['候選'][1] - med['現行'][1]:+.2f}pp)")
    ret_win = med["候選"][0] > med["現行"][0]
    mdd_win = med["候選"][1] > med["現行"][1]
    print(f"  逐路徑雙贏  {both}/{len(paths)}")

    passed = ret_win and mdd_win and both >= 5
    print("\n" + "=" * 78)
    if passed:
        print("  ✅ 通過：報酬與回撤中位數雙贏，且逐路徑雙贏 >= 5/9。")
        print("     風控參數偏袒現行模型，候選仍勝出＝強證據，回補方向值得採用。")
    else:
        why = []
        if not ret_win:
            why.append("報酬中位數未勝")
        if not mdd_win:
            why.append("回撤中位數未勝")
        if both < 5:
            why.append(f"逐路徑雙贏僅 {both}/9")
        print(f"  ❌ 未通過：{'、'.join(why)}。")
        print("     ⚠️ 單向有效：best_trading_params.json 是拿現行模型的預測調出來的，")
        print("        落敗無法區分「候選較差」與「參數為現行模型量身訂做」，不可據此判死回補方向。")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
