# -*- coding: utf-8 -*-
"""
Auto_RUN.py — 一鍵執行全流程主控腳本 (CLI 步驟路由版)
==================================================
本腳本負責將整個量化交易系統的各個步驟進行「完全解耦」的一鍵順序執行：
  1. 增量下載最新資料: scripts/scraper.py
  2. 因子加載、重建特徵、訓練與推理: auto_pipeline.py
  3. 雲端備份: scripts/StockSync.py

用法:
  python Auto_RUN.py                 # 一鍵跑完下載 ➔ 預測/訓練 ➔ 雲端備份
  python Auto_RUN.py --step download # 僅下載今日最新資料 (scraper)
  python Auto_RUN.py --step predict  # 僅重建特徵、訓練模型並推理預測 (pipeline)
  python Auto_RUN.py --step backup   # 僅執行雲端備份 (StockSync)
"""
import os
import subprocess
import sys
import time
import argparse
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(BASE_DIR, "data", "auto_run.lock")
LOCK_STALE_SECONDS = 6 * 3600  # 逾此時間視為前次崩潰殘留的死鎖，可安全接管


def acquire_lock() -> bool:
    """取得單例執行鎖，避免排程與手動同時執行而互相覆寫特徵/模型/預測檔。"""
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    if os.path.exists(LOCK_PATH):
        age = time.time() - os.path.getmtime(LOCK_PATH)
        if age < LOCK_STALE_SECONDS:
            print("\n" + "!" * 75)
            print(f"  [已在執行中] 偵測到鎖檔 {LOCK_PATH}")
            print(f"  另一個 Auto_RUN 可能正在執行 (鎖檔建立於 {age/60:.0f} 分鐘前)。")
            print(f"  若確定無其他行程執行，請手動刪除該鎖檔後重試。")
            print("!" * 75)
            return False
        print(f"[提示] 偵測到逾時鎖檔 (已 {age/3600:.1f} 小時)，視為前次殘留並接管。")
    try:
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return True
    except OSError as e:
        print(f"[錯誤] 無法建立鎖檔: {e}")
        return False


def release_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except OSError:
        pass


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

    # 單例執行鎖：防止排程與手動同時執行造成 features/models/predictions 併發覆寫
    if not acquire_lock():
        sys.exit(1)
    atexit.register(release_lock)  # 涵蓋正常結束、sys.exit 與 KeyboardInterrupt 各出口

    t_total_start = time.time()
    
    print("=" * 75)
    print("  台灣股市量化交易系統 — 一鍵自動化執行主控台 (Auto_RUN)")
    print("=" * 75)
    print(f"  啟動時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  目標步驟: {target_step.upper()}")

    extra_args = []
    if args.date:
        extra_args = ["--date", args.date]

    if target_step == "download":
        # 單獨下載模式下，強制 skip_on_limit="0" (若額度滿了則等待 1 小時後繼續抓取，不直接跳過)
        success = run_script(os.path.join("scripts", "scraper.py"), is_scraper=True, skip_on_limit="0")
        if not success:
            sys.exit(1)

    elif target_step == "predict":
        # 執行 auto_pipeline.py
        success = run_script("auto_pipeline.py", extra_args=extra_args)
        if not success:
            sys.exit(1)

    elif target_step == "backup":
        success = run_script(os.path.join("scripts", "StockSync.py"), extra_args=extra_args)
        if not success:
            sys.exit(1)

    else:
        # 執行完整流程
        steps = [
            (os.path.join("scripts", "scraper.py"), True, None),
            ("auto_pipeline.py", False, extra_args),
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
