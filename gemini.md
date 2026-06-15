# 台灣股市量化交易系統 - AI 運行與環境狀態指南 (gemini.md)

本檔案記錄了系統初始化狀態、執行命令速查與開發規範，方便 AI 在載入工作區時能快速預覽。

---

## 🟢 1. 目前工作區狀態 (最後更新: 2026-06-11)

* **Python 虛擬環境**: `.\venv\Scripts\python.exe`
* **資料最新日期**: `2026-06-11` (已下載完整價格與籌碼 CSV 數據)
* **資料完整性檢查**: 已執行 `scraper.py -c` 完成數據清理與驗證 (修復/清理 240 天的潛在異常 CSV 檔)
* **產業分類表**: 已執行 `scraper.py -fc` 重建 [stock_categories.json](scripts/stock_categories.json)，包含 **57 個產業板塊與 ETF 分類**
* **單元測試狀態**: 執行 [test_pipeline.py](tests/test_pipeline.py) 結果為 **18 / 18 全部通過 (PASS)**

---

## 🛠️ 2. 核心執行指令速查

### A. 日常生產與排程
* **一鍵執行全流程 (下載 ➔ 預測/訓練 ➔ 雲端備份)**:
  ```powershell
  .\venv\Scripts\python.exe Auto_RUN.py
  ```
* **僅增量下載今日資料**:
  ```powershell
  .\venv\Scripts\python.exe Auto_RUN.py --step download
  ```
* **僅推理今日新預測與建議**:
  ```powershell
  .\venv\Scripts\python.exe Auto_RUN.py --step predict
  ```

### B. 研發、最佳化與模擬交易
* **執行雙階段沙盒實驗 (模式 A/B)**:
  ```powershell
  .\venv\Scripts\python.exe run_workflow_experiment.py
  ```
* **執行 OOS 模擬交易 (回測)**:
  ```powershell
  .\venv\Scripts\python.exe trading_sim.py -s 2023-01-01 -e 2026-06-05 -c 2000000
  ```
* **執行特徵與 SHAP 穩定性診斷**:
  ```powershell
  .\venv\Scripts\python.exe scripts/analyze_regime_stability.py
  ```
* **單獨執行交易風控參數優化 (Walk-Forward)**:
  ```powershell
  .\venv\Scripts\python.exe scripts/optimize_trading_params.py -t 400 -s 2023-01-01 -e 2026-06-01 -c 2000000 -wf
  ```

---

## 📌 3. AI 開發與修改規範 (Rules)

1. **中央常數控制**: 所有常數、門檻與交易參數必須寫在 [config.py](config.py) 中，**嚴禁在腳本中寫死常數**。
2. **最佳化風控參數讀取優先級**: 
   * 系統啟動時會自動加載 `configs/best_trading_params.json`。
   * **市況過濾器精度**: `REGIME_*` 參數（在 [config.py](config.py) 中）會依昨日大盤狀態動態調整買入門檻。**僅在「未顯式指定 buy_threshold」時生效**，CLI 命令列參數與敏感度掃描（OFAT）仍使用靜態門檻以防被覆寫。
3. **無損備份機制**: 
   * 執行研發實驗前，必須備份 `config.py`、`configs/best_factors.json` 和 `configs/best_trading_params.json` 為 `.workflow.bak`。
   * 不論執行是否因錯誤或 Ctrl+C 中斷，`finally` 區塊必須 100% 自動還原備份檔案。
4. **Walk-Forward 參數決策**:
   * 當執行 WFO (Walk-Forward Optimization) 調參時，最終部署參數**必須使用中位數 (Median)**，嚴禁使用平均值 (Mean)，以抵禦極端噪訊。
5. **不干涉無關程式碼**: 修改時採**外科手術式修改 (Surgical Changes)**，僅觸動需求程式碼，保留原有的註解、Docstring。
6. **data 目錄唯讀保護 (嚴禁修改)**:
   * AI 擁有 [data](file:///D:/VScode_Stock/Stock/data) 目錄的**讀取權限**以進行結構與數據分析（已在 `.claudeignore` 豁免允許讀取），但**嚴禁任何修改、寫入、覆寫或刪除該目錄及其下任何檔案與子目錄的行為**。
   * 任何資料的增刪、清理或特徵產生，必須完全依賴現有系統腳本（例如 `scraper.py` 或 `feature_engineering.py`），AI 在修改或重構程式碼時，不得新增會對該目錄寫入的臨時檔案或變更其內容。

---

## 🧠 4. 寫 Code 行為準則 (Coding Behavior Guidelines)

* **Think Before Coding (謀定而後動)** — 不通靈、不通融疑惑、主動說明折衷方案。
  * 清楚說明假設。若有不確定，務必發問。
  * 若有多種解讀方式，一併呈列選項，不自行通靈決定。
  * 若有更簡單的解法，主動告知。若需求不合理，應予退回。
* **Simplicity First (簡單優先)** — 以最精簡的程式碼解決問題。不做通往未來的猜測。
  * 不實作未被要求的功能，不為單次使用的程式碼做抽象化設計。
  * 對於不可能發生的場景，不寫多餘的錯誤處理。
* **Surgical Changes (外科手術式修改)** — 僅修改必須變動的程式碼，留下的垃圾自己清理。
  * 不去重構沒有壞掉的程式碼。
  * 完美契合既有的排版與 Coding Style。
