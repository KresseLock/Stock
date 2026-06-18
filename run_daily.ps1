# run_daily.ps1 — 每日一鍵生產 + Paper Trading 自動記錄
# 用法：右鍵以 PowerShell 執行，或在終端機 `.\run_daily.ps1`
# 流程：① Auto_RUN -s all（下載→重訓→推理→備份 Drive） ② 自動把掛單寫入 paper_trading/trades.csv 並回補成交
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location $ROOT
$env:PYTHONIOENCODING = "utf-8"   # 避免中文輸出亂碼

Write-Host "==== [1/2] Auto_RUN -s all ====" -ForegroundColor Cyan
python Auto_RUN.py -s all
if ($LASTEXITCODE -ne 0) { Write-Host "[中斷] Auto_RUN 失敗，跳過記錄。" -ForegroundColor Red; exit 1 }

Write-Host "`n==== [2/2] 記錄 Paper Trading 掛單簿 ====" -ForegroundColor Cyan
python paper_trading\record_paper_trades.py
if ($LASTEXITCODE -ne 0) { Write-Host "[警告] 掛單記錄失敗。" -ForegroundColor Yellow; exit 1 }

Write-Host "`n[全部完成] 最新掛單簿： paper_trading\trades.csv" -ForegroundColor Green
