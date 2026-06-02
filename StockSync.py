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

# ╔══════════════════════════════════════════════════════╗
# ║                  雲端與路徑設定                      ║
# ╚══════════════════════════════════════════════════════╝
# 您剛才在 rclone config 設定的遠端名稱
RCLONE_REMOTE_NAME = "StockSync"

# 雲端硬碟的目標目錄名稱
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions")

    if not os.path.exists(pred_dir):
        print(f"[錯誤] 本地找不到 predictions 資料夾：{pred_dir}")
        print("  請先執行推理程式 (python inference.py) 以產生預測結果。")
        sys.exit(1)

    # 3. 計算並列出本地要上傳的 txt 檔案
    txt_files = [f for f in os.listdir(pred_dir) if f.endswith(".txt")]
    if not txt_files:
        print("[提示] predictions/ 資料夾下目前沒有任何預測 .txt 檔案。")
    else:
        print(f"發現 {len(txt_files)} 個預測檔案準備備份：")
        for f in txt_files:
            print(f"  - {f}")
        print("-" * 60)

    # 4. 執行備份命令
    # 使用 copy 而非 sync，可以避免當您刪除本地舊檔案時，雲端舊的歷史預測紀錄也被同步刪除
    remote_dest = f"{RCLONE_REMOTE_NAME}:{RCLONE_DEST_PATH}"
    print(f"正在備份至雲端目標 -> {remote_dest} ...")
    
    cmd = ["rclone", "copy", pred_dir, remote_dest]
    try:
        subprocess.run(cmd, check=True)
        print("-" * 60)
        print("  [成功] 預測結果檔案同步完成！")
        print("  您可以打開手機或電腦上的 Google Drive，確認資料夾內容。")
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"  [失敗] rclone 執行失敗，錯誤代碼: {e.returncode}")
        print("  請確認您在 rclone config 中的連線與帳戶狀態是否正確。")

    print("=" * 60)


if __name__ == "__main__":
    main()
