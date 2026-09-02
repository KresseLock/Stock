"""候選風控參數體檢：自動挑乾淨對照組 + 9 條多起點路徑對打。

用法（不需要任何參數）：
    python scripts/validate_candidate_params.py

    --baseline <路徑或檔名>  指定對照組，繞過下方第 2 點的自動挑選。
        主要用途是「明知不公平也要比」：現行部署檔看過裁判區時，自動模式會把它換掉，
        但「候選能不能取代現行」這個問題只有拿現行來比才答得出來。此時該比較是
        **單向有效檢定**——對照佔了看過答案的便宜，候選勝出才是強證據；候選落敗則無法
        區分「候選真的較差」與「對照被偏袒」，不可據此判死候選。工具會自行印出這段解讀。

三件事它自己決定，不用記：
  1. 裁判區 = 候選調參終點的次一交易日 ~ 資料最新交易日（候選沒看過的區間）。
  2. 對照組 = 現行部署檔；若現行部署檔也看過裁判區（比較會偏袒它），自動改用 configs/ 裡
     調參終點最晚、且早於裁判區起點的參數檔，並印出換掉的理由。
  3. 9 條路徑 = 起點各往後挪 0~16 個交易日、終點固定。

為何這樣切（踩過的坑，改之前先讀）：
  * 終點固定、只挪起點：CLAUDE.md 4.5 #5——本策略 MIN_HOLD_DAYS 11~23 天且獲利靠肥尾，
    把區間切成小段（尤其按月）會月初空手重建倉、月底截斷跨段贏家，系統性低估績效。
    評估一律用連續長區間，多起點靠「挪起點」而不是「切段」。
  * 要多起點不要單路徑：EXPERIMENTS_PENDING.md 方法論鐵律 #1／#2——單路徑差值可能被單筆
    極端交易主宰，同一策略換起點曾見 34%~83% 的漂移，須以中位數為準。
  * 逐路徑數字全印：CLAUDE.md 4.5 #7 教訓 (b)——只看彙總數字會把雜訊擬合成「高原」，
    必須看得到離散度與逐路徑勝負。
  * 判準是雙贏：CLAUDE.md 4.5 #4——報酬與回撤同時不劣於對照才算勝出，不可拉高回撤換報酬。

為何用 subprocess：config.py 在 import 時就讀取 best_trading_params.json 並覆寫常數，
同一個 process 內無法切換參數，故每組都另起 process 跑 trading_sim.py。

安全性：執行期間會暫時替換 configs/best_trading_params.json，結束時（含中途例外／Ctrl+C）
一律還原備份。不會自動部署——是否採用由使用者自行決定。

輸出訊息一律只用 cp950 可編碼字元（Windows 主控台預設 cp950，警告符號等會 UnicodeEncodeError）。
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# DATA_PATH 是特徵檔（交易日曆來源）的單一定義處，直接沿用避免路徑漂移。
from trading_sim import DATA_PATH

DEPLOYED = os.path.join(BASE_DIR, "configs", "best_trading_params.json")
CANDIDATE = os.path.join(BASE_DIR, "configs", "best_trading_params_candidate.json")
PARAM_GLOB = os.path.join(BASE_DIR, "configs", "best_trading_params*.json")

# 9 條路徑的起點位移（交易日）。終點固定，見檔頭「為何這樣切」。
OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16]
# 裁判區最短長度（交易日）。低於此值不足以容納數個完整持有週期（MIN_HOLD_DAYS 11~23 天），
# 結果會被單一持有週期主宰，只警示不中斷。
MIN_ZONE_DAYS = 60

_RE_RET = re.compile(r"區間報酬:\s*([+-]?[\d.]+)%")
_RE_MDD = re.compile(r"最大回撤:\s*([+-]?[\d.]+)%")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _train_end(path):
    """取參數檔的調參終點（date_range "2022-01-02 to 2025-12-31" 取後半）。
    非參數檔或舊格式回傳 None。"""
    try:
        parts = (_load(path).get("date_range") or "").split(" to ")
    except Exception:
        return None
    return parts[-1].strip() if len(parts) == 2 else None


def trading_days():
    """特徵檔裡的交易日（YYYY-MM-DD 字串，已排序）——與 trading_sim 同一份資料。"""
    import pandas as pd
    dates = pd.read_parquet(DATA_PATH, columns=["date"])["date"].unique()
    return sorted(str(d)[:10] for d in dates)


def resolve_baseline_arg(name):
    """把 --baseline 的值解成路徑：先當成路徑，再當成 configs/ 底下的檔名。"""
    for cand in (name, os.path.join(BASE_DIR, "configs", name)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def pick_baseline(zone_start):
    """挑對照組：現行部署檔優先；它若看過裁判區就換掉。

    回傳 (路徑, 調參終點, 換掉的理由或 None)。找不到乾淨對照時回傳 (None, None, 理由)。
    """
    dep_end = _train_end(DEPLOYED)
    if dep_end is not None and dep_end < zone_start:
        return DEPLOYED, dep_end, None

    reason = (f"現行部署檔調參至 {dep_end}，已看過裁判區（{zone_start} 起），"
              f"拿它當對照會系統性偏袒它，任何守紀律的候選都不可能贏")
    pool = []
    for path in glob.glob(PARAM_GLOB):
        if os.path.abspath(path) in (os.path.abspath(DEPLOYED), os.path.abspath(CANDIDATE)):
            continue
        te = _train_end(path)
        if te is not None and te < zone_start:
            pool.append((te, _load(path).get("optimized_at", ""), path))
    if not pool:
        return None, None, reason + "；且 configs/ 裡找不到任何調參終點早於裁判區的替代對照檔"
    # 調參終點越晚、資訊量越接近候選；同終點再比 optimized_at
    te, _, path = max(pool)
    return path, te, reason


def run_backtest(start, end, capital):
    """另起 process 跑 trading_sim.py，回傳 (報酬%, 回撤%)。回撤為負值。"""
    # 子行程的輸出編碼必須釘死：若外部環境設了 PYTHONIOENCODING=utf-8，子行程會改吐 UTF-8，
    # 而下方是硬解 cp950，會整段變亂碼、regex 找不到數字 → 誤報「無法解析回測輸出」。
    env = dict(os.environ, PYTHONIOENCODING="cp950")
    proc = subprocess.run(
        [sys.executable, "trading_sim.py", "--start", start, "--end", end, "-c", str(capital)],
        cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    # trading_sim.py 在 Windows 主控台輸出為 cp950，非 UTF-8
    out = proc.stdout.decode("cp950", errors="replace")
    m_ret, m_mdd = _RE_RET.search(out), _RE_MDD.search(out)
    if not (m_ret and m_mdd):
        tail = "\n".join(out.splitlines()[-15:])
        raise RuntimeError(f"無法解析回測輸出（exit={proc.returncode}）：\n{tail}")
    return float(m_ret.group(1)), float(m_mdd.group(1))


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    ap = argparse.ArgumentParser(description="候選風控參數體檢（自動挑對照組 + 多起點）")
    ap.add_argument("-c", "--capital", type=int, default=2000000,
                    help="初始資金（預設 2000000，與 run_workflow_experiment.py 的實驗資金一致）")
    ap.add_argument("--baseline", default=None,
                    help="指定對照組參數檔（路徑或 configs/ 底下的檔名），繞過自動挑選。"
                         "對照若看過裁判區，比較會偏袒它，結論只在候選勝出時有效——見檔頭說明")
    args = ap.parse_args()

    if not os.path.exists(CANDIDATE):
        print(f"[錯誤] 找不到候選檔：{CANDIDATE}")
        print("       請先執行 python scripts/optimize_trading_params.py --regime -wf")
        return 1
    if not os.path.exists(DEPLOYED):
        print(f"[錯誤] 找不到現行部署檔：{DEPLOYED}")
        return 1

    cand_end = _train_end(CANDIDATE)
    if cand_end is None:
        print("[錯誤] 候選檔沒有 date_range，無法判斷它看過哪些資料，不能做乾淨對照。")
        return 1

    days = trading_days()
    zone_days = [d for d in days if d > cand_end]
    if len(zone_days) <= max(OFFSETS):
        print(f"[錯誤] 候選調參至 {cand_end}，其後只有 {len(zone_days)} 個交易日，"
              f"不足以跑 {len(OFFSETS)} 條起點相差 {max(OFFSETS)} 天的路徑。")
        return 1

    zone_start, zone_end = zone_days[0], zone_days[-1]
    if args.baseline:
        baseline = resolve_baseline_arg(args.baseline)
        if baseline is None:
            print(f"[錯誤] 找不到指定的對照檔：{args.baseline}")
            return 1
        if os.path.abspath(baseline) == os.path.abspath(CANDIDATE):
            print("[錯誤] 對照組不能就是候選檔本身。")
            return 1
        base_end, swap_reason = _train_end(baseline), None
        if base_end is None:
            print(f"[錯誤] 對照檔 {os.path.basename(baseline)} 沒有 date_range，"
                  "無法判斷它看過哪些資料。")
            return 1
    else:
        baseline, base_end, swap_reason = pick_baseline(zone_start)
        if baseline is None:
            print(f"[錯誤] 找不到乾淨的對照組：{swap_reason}。")
            return 1
    # 對照看過裁判區 = 比較偏袒對照，結論只在候選勝出時有效（單向有效檢定）
    base_dirty = base_end >= zone_start

    paths = [(zone_days[o], zone_end) for o in OFFSETS]
    offset_str = "/".join(str(o) for o in OFFSETS)

    # 全程 tee：8/28 那輪就是因為結果只印在主控台，關機後整輪數字全丟（見 utils.start_tee_log）
    from scripts.utils import start_tee_log
    _log_path = start_tee_log("validate_candidate_params")

    print("=" * 78)
    print("  候選參數體檢（自動模式）")
    print("=" * 78)
    print(f"  候選     : {os.path.basename(CANDIDATE)}（調參至 {cand_end}）")
    print(f"  對照     : {os.path.basename(baseline)}（調參至 {base_end}）")
    if swap_reason:
        print(f"             ※ 已自動換掉現行部署檔：{swap_reason}")
    if base_dirty:
        print(f"             ※ 對照調參至 {base_end}，已涵蓋裁判區 → 比較偏袒對照組。")
        print("               單向有效檢定：候選勝出才算數，落敗不可判死（見結論段）")
    print(f"  裁判區   : {zone_start} ~ {zone_end}（兩邊都沒看過，共 {len(zone_days)} 個交易日）")
    print(f"  路徑     : {len(OFFSETS)} 條，起點各往後挪 {offset_str} 個交易日，終點固定")
    print(f"  資金     : {args.capital:,}")
    print(f"  執行記錄 : {os.path.relpath(_log_path, BASE_DIR)}")
    if len(zone_days) - max(OFFSETS) < MIN_ZONE_DAYS:
        print(f"  [注意] 最短路徑僅 {len(zone_days) - max(OFFSETS)} 個交易日，低於建議的 {MIN_ZONE_DAYS} 天；")
        print("         MIN_HOLD_DAYS 為 11~23 天，區間太短會被單一持有週期主宰，結論僅供參考。")
    print("=" * 78)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{DEPLOYED}.bak_validate_{stamp}"
    shutil.copy2(DEPLOYED, backup)
    print(f"[備份] 現行參數已備份至 {os.path.basename(backup)}")

    res = {}
    try:
        for label, src in (("對照", baseline), ("候選", CANDIDATE)):
            # --baseline 指定現行部署檔時 src 就是 DEPLOYED，copy2 同檔會拋 SameFileError
            if os.path.abspath(src) != os.path.abspath(DEPLOYED):
                shutil.copy2(src, DEPLOYED)
            print(f"\n[{label}] {os.path.basename(src)}")
            for i, (s, e) in enumerate(paths, 1):
                r, m = run_backtest(s, e, args.capital)
                res[(label, i)] = (r, m)
                print(f"  路徑 {i}/{len(paths)}  起點 {s}   報酬 {r:+7.2f}%   回撤 {m:7.2f}%", flush=True)
    finally:
        shutil.copy2(backup, DEPLOYED)
        print(f"\n[還原] 現行參數已從備份還原（{os.path.basename(backup)} 保留備查）")

    idx = list(range(1, len(paths) + 1))
    b_ret = [res[("對照", i)][0] for i in idx]
    b_mdd = [res[("對照", i)][1] for i in idx]
    c_ret = [res[("候選", i)][0] for i in idx]
    c_mdd = [res[("候選", i)][1] for i in idx]
    # 回撤是負值，越接近 0 越好，故「不劣於」是 >=
    wins = sum(1 for i in idx
               if res[("候選", i)][0] > res[("對照", i)][0]
               and res[("候選", i)][1] >= res[("對照", i)][1])

    med_b_ret, med_c_ret = _median(b_ret), _median(c_ret)
    med_b_mdd, med_c_mdd = _median(b_mdd), _median(c_mdd)

    print("\n" + "=" * 78)
    print(f"  {len(paths)} 條路徑總結")
    print("=" * 78)
    _mark = lambda better: "（候選較優）" if better else "（候選較差）"
    print(f"  報酬中位數   對照 {med_b_ret:+.2f}%   候選 {med_c_ret:+.2f}%"
          f"   差 {med_c_ret - med_b_ret:+.2f}pp {_mark(med_c_ret > med_b_ret)}")
    print(f"  報酬範圍     對照 {min(b_ret):+.2f}~{max(b_ret):+.2f}%"
          f"   候選 {min(c_ret):+.2f}~{max(c_ret):+.2f}%")
    print(f"  回撤中位數   對照 {med_b_mdd:.2f}%   候選 {med_c_mdd:.2f}%"
          f"   差 {med_c_mdd - med_b_mdd:+.2f}pp {_mark(med_c_mdd >= med_b_mdd)}")
    print(f"  回撤範圍     對照 {min(b_mdd):.2f}~{max(b_mdd):.2f}%"
          f"   候選 {min(c_mdd):.2f}~{max(c_mdd):.2f}%")
    print(f"  逐路徑報酬贏 {sum(1 for i in idx if c_ret[i-1] > b_ret[i-1])} / {len(paths)}"
          f"   逐路徑回撤贏 {sum(1 for i in idx if c_mdd[i-1] >= b_mdd[i-1])} / {len(paths)}"
          f"   逐路徑雙贏 {wins} / {len(paths)}")

    med_win = (med_c_ret > med_b_ret) and (med_c_mdd >= med_b_mdd)
    most_win = wins * 2 > len(paths)
    print("-" * 78)
    print("  判準（CLAUDE.md 4.5 #4）：報酬與回撤中位數雙贏，且過半路徑雙贏，才算勝出。")
    if med_win and most_win:
        print(f"  → 候選勝出：中位數雙贏，且 {wins}/{len(paths)} 條路徑雙贏。")
    elif med_win:
        print(f"  → 需再確認：中位數雙贏，但只有 {wins}/{len(paths)} 條路徑雙贏，"
              "改善可能由少數起點驅動。")
    else:
        _why = []
        if med_c_ret <= med_b_ret:
            _why.append("報酬中位數未勝出")
        if med_c_mdd < med_b_mdd:
            _why.append("回撤中位數劣化")
        print(f"  → 候選未勝出：{'、'.join(_why)}，未達雙贏（逐路徑雙贏 {wins}/{len(paths)}）。")
    if swap_reason:
        print("\n  本次對照組不是現行部署檔，所以這個結論回答的是「候選比乾淨基準好不好」，")
        print("  不是「要不要取代現行部署檔」。現行部署檔已看過裁判區，無法公平比較。")
    if base_dirty:
        print()
        print("  本次對照組看過裁判區、佔了看過答案的便宜，所以這是單向有效檢定：")
        if med_win and most_win:
            print("  候選在被偏袒的對照面前仍雙贏 → 強證據，可考慮部署。")
        else:
            print("  候選未勝出，但無法區分「候選真的較差」與「對照被偏袒」，")
            print("  不可據此判死候選；要判死須另找沒看過裁判區的乾淨對照（自動模式）。")
    print(f"\n  注意：{len(OFFSETS)} 條路徑起點只差 {max(OFFSETS)} 個交易日、區間高度重疊，")
    print("        中位數反映的是「起點敏感度」而非 9 個獨立樣本；離散度大就代表結論脆弱。")
    print("\n  要採用候選時自行執行（本工具不會自動部署）：")
    print("    Copy-Item configs/best_trading_params_candidate.json configs/best_trading_params.json -Force")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
