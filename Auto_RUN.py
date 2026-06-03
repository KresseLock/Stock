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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_path: str, is_scraper: bool = False) -> bool:
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
        # 加入環境變數以指示 FinMind API 遇到流量限制時直接跳過，不等待 1 小時
        env = os.environ.copy()
        env["SKIP_ON_FINMIND_LIMIT"] = "1"
        
        subprocess.run([sys.executable, full_path], check=True, env=env)
        t_elapsed = time.time() - t_start
        print(f"\n[成功] {script_path} 執行完成！耗時: {t_elapsed:.1f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        if is_scraper and e.returncode == 99:
            t_elapsed = time.time() - t_start
            print(f"\n[注意] {script_path} 觸發 FinMind API 額度限制 (已自動跳過，不中斷全流程)，流程繼續！耗時: {t_elapsed:.1f} 秒")
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
        choices=["download", "d", "predict", "p", "backup", "b", "all", "a"],
        default="all",
        help="執行指定步驟 (download/d | predict/p | backup/b | all/a)"
    )
    args = parser.parse_args()

    # 簡碼對齊映射
    step_map = {
        "d": "download",
        "p": "predict",
        "b": "backup",
        "a": "all",
        "download": "download",
        "predict": "predict",
        "backup": "backup",
        "all": "all"
    }
    target_step = step_map[args.step]

    t_total_start = time.time()
    
    print("=" * 75)
    print("  台灣股市量化交易系統 — 一鍵自動化執行主控台 (Auto_RUN)")
    print("=" * 75)
    print(f"  啟動時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  目標步驟: {target_step.upper()}")

    if target_step == "download":
        success = run_script(os.path.join("scripts", "scraper.py"), is_scraper=True)
        if not success:
            sys.exit(1)

    elif target_step == "predict":
        # 執行 auto_pipeline.py
        success = run_script("auto_pipeline.py")
        if not success:
            sys.exit(1)

    elif target_step == "backup":
        success = run_script(os.path.join("scripts", "StockSync.py"))
        if not success:
            sys.exit(1)

    else:
        # 執行完整流程
        steps = [
            (os.path.join("scripts", "scraper.py"), True),
            ("auto_pipeline.py", False),
            (os.path.join("scripts", "StockSync.py"), False)
        ]
        
        for idx, (script, is_scr) in enumerate(steps, 1):
            print(f"\n>>> 進入第 {idx}/{len(steps)} 階段...")
            success = run_script(script, is_scraper=is_scr)
            if not success:
                print("\n" + "!" * 75)
                print(f"  [中斷警告] 流程在執行到 {script} 時發生錯誤中斷！")
                print("  後續步驟已停止執行，請先修復上述錯誤。")
                print("!" * 75)
                sys.exit(1)
                
        total_time = time.time() - t_total_start
        print("\n" + "=" * 75)
        print(f"  🎉 [全流程執行成功] 總耗時: {total_time/60:.1f} 分鐘")
        print(f"  雲端同步已完成，您可以前往 Google Drive 查看最新結果！")
        print("=" * 75)


if __name__ == "__main__":
    main()
