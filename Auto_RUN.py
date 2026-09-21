# -*- coding: utf-8 -*-
"""
Auto_RUN.py — 一鍵執行全流程主控腳本 (CLI 步驟路由版)
==================================================
本腳本負責將整個量化交易系統的各個步驟進行「完全解耦」的一鍵順序執行：
  1. 增量下載最新資料: scripts/scraper.py
  2. 重建特徵與推理（不重訓）: auto_pipeline.py -s feature、-s inference
  3. 雲端備份: scripts/StockSync.py

用法:
  python Auto_RUN.py                 # 一鍵跑完下載 ➔ 特徵/推理 ➔ 雲端備份
  python Auto_RUN.py --step download # 僅下載今日最新資料 (scraper)
  python Auto_RUN.py --step predict  # 僅重建特徵並推理預測 (pipeline，不重訓)
  python Auto_RUN.py --step backup   # 僅執行雲端備份 (StockSync)

為何每日不重訓（CLAUDE.md §4.5 #9）：訓練資料被 BACKTEST_DATE 截斷而固定，每日重訓學不到新資料，
卻會因歷史資料補抓、Stocks.txt 持倉變動、產業表更新而悄悄換掉已把關的模型。
要換模型請手動執行 python auto_pipeline.py -s train，並先走把關。
"""
import os
import subprocess
import sys
import time
import argparse
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 執行鎖與 run_workflow_experiment.py 共用（路徑／逾時門檻在 config.py § 0），兩者互斥
from scripts.utils import model_fingerprint, acquire_run_lock, release_run_lock


def run_script(script_path: str, is_scraper: bool = False, skip_on_limit: str = None, extra_args: list = None) -> bool:
    """執行指定的 Python 腳本並監控結果"""
    full_path = os.path.join(BASE_DIR, script_path)
    if not os.path.exists(full_path):
        print(f"\n[錯誤] 找不到腳本檔案: {script_path} (路徑: {full_path})")
        return False

    print("\n" + "=" * 70)
    print(f"  正在執行: {script_path}")
    print("=" * 70)
    
    t_start = time.time()
    try:
        # 使用當前虛擬環境的 Python 執行檔執行子腳本，並將輸出實時印出到終端機
        env = os.environ.copy()
        if skip_on_limit is not None:
            env["SKIP_ON_FINMIND_LIMIT"] = skip_on_limit
        else:
            # 不指定時（如完整流程），優先讀取既有環境變數設定；若無設定則預設為 "1" (直接跳過以利自動化)
            if "SKIP_ON_FINMIND_LIMIT" not in env:
                env["SKIP_ON_FINMIND_LIMIT"] = "1"
        
        cmd = [sys.executable, full_path]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, check=True, env=env)
        t_elapsed = time.time() - t_start
        print(f"\n[成功] {script_path} 執行完成！耗時: {t_elapsed:.1f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        if is_scraper and e.returncode == 99:
            t_elapsed = time.time() - t_start
            print(f"\n[注意] {script_path} 觸發 FinMind API 額度限制 (已自動跳過，不中斷全流程)，流程繼續！耗時: {t_elapsed:.1f} 秒")
            # 標記本次資料不完整：供下游 inference 在預測輸出標註、供最終 banner 警示
            os.environ["RUN_DATA_INCOMPLETE"] = "1"
            return True
        print(f"\n[失敗] {script_path} 執行中斷，錯誤代碼: {e.returncode}")
        return False
    except Exception as ex:
        print(f"\n[系統錯誤] 無法啟動 {script_path}: {ex}")
        return False


def main():
    parser = argparse.ArgumentParser(description="台股量化交易系統一鍵自動化執行主控台")
    parser.add_argument(
        "-s", "--step",
        type=str,
        choices=["download", "d", "D", "predict", "p", "P", "backup", "b", "B", "all", "a", "A"],
        default="all",
        help="執行指定步驟 (download/d/D | predict/p/P | backup/b/B | all/a/A)"
    )
    parser.add_argument(
        "-d", "--date",
        type=str,
        default=None,
        help="指定推理/預測的基準日期 (格式: YYYYMMDD 或 YYYY-MM-DD，僅在預測/predict/pipeline步驟中生效)"
    )
    args = parser.parse_args()

    # 簡碼對齊映射
    step_map = {
        "d": "download",
        "D": "download",
        "p": "predict",
        "P": "predict",
        "b": "backup",
        "B": "backup",
        "a": "all",
        "A": "all",
        "download": "download",
        "predict": "predict",
        "backup": "backup",
        "all": "all"
    }
    target_step = step_map[args.step]

    # 單例執行鎖（與 run_workflow_experiment.py 共用）：防止併發覆寫 features/models/predictions，
    # 也防止在實驗流程改寫 config.py、切換模型的期間產出正式預測
    if not acquire_run_lock("Auto_RUN"):
        sys.exit(1)
    atexit.register(release_run_lock)  # 涵蓋正常結束、sys.exit 與 KeyboardInterrupt 各出口

    t_total_start = time.time()
    
    print("=" * 75)
    print("  台灣股市量化交易系統 — 一鍵自動化執行主控台 (Auto_RUN)")
    print("=" * 75)
    print(f"  啟動時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  目標步驟: {target_step.upper()}")
    if target_step in ("predict", "all"):
        print(f"  模型指紋: {model_fingerprint()}（每日不重訓，指紋應維持不變；變了代表 models/ 被動過）")

    extra_args = []
    if args.date:
        extra_args = ["--date", args.date]

    if target_step == "download":
        # 單獨下載模式下，強制 skip_on_limit="0" (若額度滿了則等待 1 小時後繼續抓取，不直接跳過)
        success = run_script(os.path.join("scripts", "scraper.py"), is_scraper=True, skip_on_limit="0")
        if not success:
            sys.exit(1)

    elif target_step == "predict":
        # 只重建特徵＋推論，不重訓（見檔頭說明）；--date 只對推論生效
        for step_args in (["-s", "feature"], ["-s", "inference"] + extra_args):
            if not run_script("auto_pipeline.py", extra_args=step_args):
                sys.exit(1)

    elif target_step == "backup":
        success = run_script(os.path.join("scripts", "StockSync.py"), extra_args=extra_args)
        if not success:
            sys.exit(1)

    else:
        # 執行完整流程
        # 特徵與推論分兩步呼叫，刻意不跑 auto_pipeline 的 all（all 會重訓，見檔頭說明）
        steps = [
            (os.path.join("scripts", "scraper.py"), True, None),
            ("auto_pipeline.py", False, ["-s", "feature"]),
            ("auto_pipeline.py", False, ["-s", "inference"] + extra_args),
            (os.path.join("scripts", "StockSync.py"), False, extra_args)
        ]
        
        for idx, (script, is_scr, e_args) in enumerate(steps, 1):
            print(f"\n>>> 進入第 {idx}/{len(steps)} 階段...")
            success = run_script(script, is_scraper=is_scr, skip_on_limit=None, extra_args=e_args)
            if not success:
                print("\n" + "!" * 75)
                print(f"  [中斷警告] 流程在執行到 {script} 時發生錯誤中斷！")
                print("  後續步驟已停止執行，請先修復上述錯誤。")
                print("!" * 75)
                sys.exit(1)
                
        # 先執行實盤損益與持倉分析，再宣告成功——避免「成功」banner 之後又冒出分析輸出/錯誤而誤導 log 判讀
        analyze_script = "analyze_real_pnl.py"
        if os.path.exists(os.path.join(BASE_DIR, analyze_script)):
            analyze_ok = run_script(analyze_script)
        else:
            analyze_ok = None  # 無此腳本，非成功也非失敗

        total_time = time.time() - t_total_start
        data_incomplete = os.environ.get("RUN_DATA_INCOMPLETE") == "1"
        print("\n" + "=" * 75)
        if data_incomplete:
            print(f"  [全流程完成（資料不完整）] 總耗時: {total_time/60:.1f} 分鐘")
            print(f"  ⚠ 本次上游下載曾觸發 FinMind 額度限制被跳過，籌碼/基本面可能為舊值；")
            print(f"    預測輸出已標註此情形，判讀訊號時請留意。")
        else:
            print(f"  [全流程執行成功] 總耗時: {total_time/60:.1f} 分鐘")
        if analyze_ok is False:
            print(f"  ⚠ 實盤損益分析 (analyze_real_pnl.py) 執行失敗，請檢視上方輸出。")
        print(f"  雲端同步已完成，您可以前往 Google Drive 查看最新結果！")
        print("=" * 75)


if __name__ == "__main__":
    main()
