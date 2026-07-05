# -*- coding: utf-8 -*-
"""
StockSync.py — 台灣股市量化預測結果雲端自動備份工具
==================================================
使用 rclone 將 predictions/ 資料夾中產生的預測結果文字檔 (.txt) 
自動複製並備份到您的 Google Drive (StockSync 遠端)。
"""
import os
import subprocess
import sys

# 統一設定路徑與環境
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ╔══════════════════════════════════════════════════════╗
# ║                  雲端與路徑設定                      ║
# ╚══════════════════════════════════════════════════════╝
# 載入 config.py 的設定，若無則使用預設值
try:
    from config import RCLONE_REMOTE_NAME, RCLONE_DEST_PATH
except ImportError:
    RCLONE_REMOTE_NAME = "StockSync"
    RCLONE_DEST_PATH = "StockData"


def check_rclone_installed() -> bool:
    """檢查環境中是否可調用 rclone"""
    try:
        subprocess.run(["rclone", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    print("=" * 60)
    print("        StockSync — 預測結果雲端備份啟動")
    print("=" * 60)

    # 1. 檢查 rclone 命令是否可用
    if not check_rclone_installed():
        print("[錯誤] 找不到 rclone 指令！")
        print("  請確認 rclone.exe 已經被放置在 venv\\Scripts\\ 目錄中，")
        print("  或是已加入您 Windows 系統的環境變數 PATH 中。")
        sys.exit(1)

    # 2. 定義本地預測資料夾路徑
    pred_dir = os.path.join(BASE_DIR, "predictions")

    if not os.path.exists(pred_dir):
        print(f"[錯誤] 本地找不到 predictions 資料夾：{pred_dir}")
        print("  請先執行推理程式 (python scripts/inference.py) 以產生預測結果。")
        sys.exit(1)

    # 3. 找出最新的預測 .txt 檔案（依檔名排序，日期越大越新）
    txt_files = sorted(
        [f for f in os.listdir(pred_dir) if f.endswith(".txt")]
    )
    if not txt_files:
        print("[提示] predictions/ 資料夾下目前沒有任何預測 .txt 檔案。")
        print("=" * 60)
        return

    latest_file = txt_files[-1]
    print(f"共 {len(txt_files)} 個預測檔案，僅上傳最新：{latest_file}")
    print("-" * 60)

    # 4. 執行備份命令 — 只上傳最新的單一檔案
    remote_dest = f"{RCLONE_REMOTE_NAME}:{RCLONE_DEST_PATH}"
    local_path = os.path.join(pred_dir, latest_file)
    print(f"正在備份至雲端目標 -> {remote_dest} ...")

    cmd = ["rclone", "copyto", local_path, f"{remote_dest}/{latest_file}"]
    try:
        subprocess.run(cmd, check=True)
        print("-" * 60)
        print("  [成功] 最新預測結果檔案同步完成！")
        print("  您可以打開手機或電腦上的 Google Drive，確認資料夾內容。")
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"  [失敗] rclone 執行失敗，錯誤代碼: {e.returncode}")
        print("  請確認您在 rclone config 中的連線與帳戶狀態是否正確。")

    print("=" * 60)


if __name__ == "__main__":
    main()
