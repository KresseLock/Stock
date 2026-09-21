# -*- coding: utf-8 -*-
"""回補模型 × 風控參數的 2×2 歸因對照（EXPERIMENTS_PENDING.md 回補章節 待決處置 A）。

**為什麼要 2×2**：這一輪同時動了模型（回補訓練資料）與風控參數（在候選模型上重調），
只跑「候選模型 × 新參數」會落入鐵律 #6——無論結果好壞都無法分辨是哪一層在起作用。
故四組全跑，其中「現行模型 × 新參數」是純粹的歸因對照。

| | 舊參數（乾淨） | 新參數（本輪） |
| :--- | :--- | :--- |
| 現行模型 | 基準 | **歸因對照**：新參數本身好不好 |
| 候選模型 | ＝ Step 6 的候選組 | 本輪主角 |

**參數軸為何不用 `best_trading_params.json`**：它調參至 2026-07-23，看過整個裁判區，
拿它當對照會系統性偏袒它（該檔坑 #5）。改用 `oldselect_20260828`——同樣調參至 2025-12-31，
且是本專案唯一通過過乾淨對照雙贏判準的參數。

**裁判區 2026-01-02 起**：新參數調參至 2025-12-31，兩組模型都截斷於 BACKTEST_DATE=20250801，
故該區間對四組全部乾淨。

用法：python tests/test_backfill_param_pairing_gate.py [-c 資金]
"""

import argparse
import itertools
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.validate_candidate_params import OFFSETS, run_backtest, trading_days, _median
from scripts.utils import start_tee_log

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILES = ("lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json")
DEPLOYED = os.path.join(BASE_DIR, "configs", "best_trading_params.json")

from config import DATA_DIR   # 資料路徑唯一來源：config.py § 0

MODELS = {
    "現行模型": os.path.join(DATA_DIR, "_backup_pre_backfill_20260910", "models_backup"),
    "候選模型": os.path.join(MODEL_DIR, "_candidate_backfill_20260910"),
}
PARAMS = {
    "舊參數": os.path.join(BASE_DIR, "configs", "best_trading_params_candidate.oldselect_20260828.json"),
    "新參數": os.path.join(BASE_DIR, "configs", "best_trading_params_candidate.json"),
}
ZONE_FROM = "2026-01-02"


def install_model(src):
    for name in MODEL_FILES:
        path = os.path.join(src, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少模型檔：{path}")
        shutil.copy2(path, os.path.join(MODEL_DIR, name))


def main():
    ap = argparse.ArgumentParser(description="回補模型 × 風控參數 2×2 歸因對照")
    ap.add_argument("-c", "--capital", type=int, default=2000000)
    args = ap.parse_args()

    for label, path in list(MODELS.items()) + list(PARAMS.items()):
        if not os.path.exists(path):
            print(f"[錯誤] 找不到 {label}：{path}")
            if label == "新參數":
                print("       請先執行 python scripts/optimize_trading_params.py --regime -wf -j 4 -c 2000000")
            return 1

    days = [d for d in trading_days() if d >= ZONE_FROM]
    if len(days) <= max(OFFSETS):
        print(f"[錯誤] {ZONE_FROM} 之後只有 {len(days)} 個交易日，不足以跑 {len(OFFSETS)} 條路徑。")
        return 1
    zone_end = days[-1]
    paths = [(days[o], zone_end) for o in OFFSETS]

    log_path = start_tee_log("test_backfill_param_pairing_gate")
    print("=" * 78)
    print("  回補模型 × 風控參數 2×2 歸因對照")
    print("=" * 78)
    print(f"  裁判區   : {days[0]} ~ {zone_end}（{len(days)} 個交易日，對四組全部乾淨）")
    print(f"  路徑     : {len(OFFSETS)} 條，起點各往後挪 {'/'.join(map(str, OFFSETS))} 個交易日，終點固定")
    print(f"  資金     : {args.capital:,}")
    print(f"  執行記錄 : {os.path.relpath(log_path, BASE_DIR)}")
    print("=" * 78)

    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    param_backup = f"{DEPLOYED}.bak_pairing_{stamp}"
    shutil.copy2(DEPLOYED, param_backup)
    print(f"[備份] 現行風控參數已備份至 {os.path.basename(param_backup)}")
    # 生產 models/ 未必等於候選副本（重訓會換實現，見 EXPERIMENTS_PENDING.md 鐵律 #8），
    # 故還原「測試前的 models/」，而不是固定裝回候選副本
    model_snapshot = os.path.join(MODEL_DIR, f"_pre_pairing_{stamp}")
    os.makedirs(model_snapshot)
    for name in MODEL_FILES:
        shutil.copy2(os.path.join(MODEL_DIR, name), model_snapshot)

    res = {}
    try:
        for (m_label, m_src), (p_label, p_src) in itertools.product(MODELS.items(), PARAMS.items()):
            install_model(m_src)
            shutil.copy2(p_src, DEPLOYED)
            cell = f"{m_label} × {p_label}"
            print(f"\n[{cell}] 開始 {len(paths)} 條路徑")
            rows = []
            for i, (start, end) in enumerate(paths, 1):
                ret, mdd = run_backtest(start, end, args.capital)
                rows.append((start, ret, mdd))
                print(f"  路徑 {i}/{len(paths)}  {start} ~ {end}  報酬 {ret:+8.2f}%  回撤 {mdd:7.2f}%")
            res[cell] = rows
    finally:
        shutil.copy2(param_backup, DEPLOYED)
        install_model(model_snapshot)
        shutil.rmtree(model_snapshot)
        print(f"\n[還原] 風控參數已放回 {os.path.basename(DEPLOYED)}；models/ 已放回測試前的模型")

    med = {k: (_median([r for _, r, _ in v]), _median([m for _, _, m in v])) for k, v in res.items()}

    print("\n" + "=" * 78)
    print("  2×2 中位數（報酬% / 回撤%；回撤為負值，越接近 0 越好）")
    print("=" * 78)
    print(f"  {'':<12}{'舊參數':>22}{'新參數':>22}")
    for m in MODELS:
        cells = "".join(f"{med[f'{m} × {p}'][0]:>+11.2f}%{med[f'{m} × {p}'][1]:>10.2f}%" for p in PARAMS)
        print(f"  {m:<12}{cells}")

    def cmp_pair(a, b):
        """b 相對 a 的中位數差，與逐路徑雙贏數。"""
        both = sum((r1 > r0) and (m1 > m0)
                   for (_, r0, m0), (_, r1, m1) in zip(res[a], res[b]))
        return med[b][0] - med[a][0], med[b][1] - med[a][1], both

    print("\n" + "=" * 78)
    print("  歸因（每列固定一軸、只動另一軸）")
    print("=" * 78)
    print(f"  {'比較':<34}{'報酬差':>11}{'回撤差':>11}  雙贏")
    rows = [
        ("新參數的效果（在候選模型上）", "候選模型 × 舊參數", "候選模型 × 新參數"),
        ("新參數的效果（在現行模型上）", "現行模型 × 舊參數", "現行模型 × 新參數"),
        ("回補的效果（在舊參數下）",     "現行模型 × 舊參數", "候選模型 × 舊參數"),
        ("回補的效果（在新參數下）",     "現行模型 × 新參數", "候選模型 × 新參數"),
        ("★ 部署決策：新組合 vs 現有最佳", "現行模型 × 舊參數", "候選模型 × 新參數"),
    ]
    for name, a, b in rows:
        dr, dm, both = cmp_pair(a, b)
        print(f"  {name:<34}{dr:>+10.2f}pp{dm:>+10.2f}pp  {both}/{len(paths)}")

    dr, dm, both = cmp_pair("現行模型 × 舊參數", "候選模型 × 新參數")
    passed = dr > 0 and dm > 0 and both >= 5
    print("\n" + "=" * 78)
    if passed:
        print("  ✅ 新組合通過部署判準（報酬與回撤中位數雙贏，逐路徑雙贏 >= 5/9）。")
        print("     判讀前仍須確認：新參數的 markup_pct 若貼邊界或極端值，先跑")
        print("     python tests/test_markup_bound_gate.py 排除單格噪音峰（該檔坑 #6）。")
    else:
        why = [t for t, ok in (("報酬中位數未勝", dr > 0), ("回撤中位數未勝", dm > 0),
                               (f"逐路徑雙贏僅 {both}/9", both >= 5)) if not ok]
        print(f"  ❌ 新組合未通過部署判準：{'、'.join(why)}。")
    print("     ※ 交互作用看上表前兩列：新參數只在候選模型上有效 → 屬配對效應；")
    print("       兩者皆有效 → 是參數本身的改良；皆無效 → 這一輪優化沒抓到東西。")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
