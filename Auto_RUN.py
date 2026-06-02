# -*- coding: utf-8 -*-
"""
Auto_RUN.py — 一鍵執行全流程主控腳本
==================================================
本腳本負責將整個量化交易系統的各個步驟進行「完全解耦」的一鍵順序執行：
  1. 執行 main.py            (下載最新資料)
  2. 執行 auto_pipeline.py    (因子加載、特徵工程與模型訓練)
  3. 執行 inference.py       (生成最新預測排行榜與下單建議)
  4. 執行 StockSync.py       (將預測結果備份上傳至 Google Drive)

如果其中任何一個腳本執行失敗，將會立即中斷並回報錯誤，確保流程正確性。
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name: str) -> bool:
    """執行指定的 Python 腳本並監控結果"""
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"\n[錯誤] 找不到腳本檔案: {script_name} (路徑: {script_path})")
        return False

    print("\n" + "=" * 70)
    print(f"  正在執行: {script_name}")
    print("=" * 70)
    
    t_start = time.time()
    try:
        # 使用當前虛擬環境的 Python 執行檔執行子腳本，並將輸出實時印出到終端機
        subprocess.run([sys.executable, script_path], check=True)
        t_elapsed = time.time() - t_start
        print(f"\n[成功] {script_name} 執行完成！耗時: {t_elapsed:.1f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[失敗] {script_name} 執行中斷，錯誤代碼: {e.returncode}")
        return False
    except Exception as ex:
        print(f"\n[系統錯誤] 無法啟動 {script_name}: {ex}")
        return False


def main():
    t_total_start = time.time()
    
    print("=" * 75)
    print("  台灣股市量化交易系統 — 一鍵自動化執行主控台 (Auto_RUN)")
    print("=" * 75)
    print(f"  啟動時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 依序執行的解耦腳本清單
    scripts_to_run = [
        "main.py",
        "auto_pipeline.py",
        "inference.py",
        "StockSync.py"
    ]
    
    for idx, script in enumerate(scripts_to_run, 1):
        print(f"\n>>> 進入第 {idx}/{len(scripts_to_run)} 階段...")
        success = run_script(script)
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
