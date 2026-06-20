# 台灣股市量化交易系統 - AI 預測與因子導入指南 (agents.md)

本檔案為 AI 專門指南，專注於因子導入、特徵工程、模型預測與優化流程，去除無關雜訊與問題記錄。

---

## 1. 因子與特徵工程 (Factor Ingestion & Feature Engineering)

* **資料唯讀保護**：[data/](file:///D:/Vscode_workspace/Stock/data) 目錄對 AI 為唯讀，嚴禁任何修改、寫入或刪除該目錄及其下檔案的行為。
* **因子參數優化**：[optimize_factors.py](file:///D:/Vscode_workspace/Stock/scripts/optimize_factors.py) 使用 Optuna 尋找最佳技術指標參數，輸出至 [best_factors.json](file:///D:/Vscode_workspace/Stock/configs/best_factors.json)。
* **特徵計算**：[feature_engineering.py](file:///D:/Vscode_workspace/Stock/scripts/feature_engineering.py) 載入最佳因子，計算四大類特徵，輸出 `features_combined.parquet`：
  1. *技術指標*：MA, KD, RSI, MACD, Boll, ATR, Vol MA (已移除 Level Bias，轉為比例/變化率)。
  2. *法人籌碼*：外資/投信/自營商買賣超、連買天數、滾動累積籌碼。
  3. *市場感知*：全市場日報酬均值、5日/20日市場寬度。
  4. *板塊強度*：依 [stock_categories.json](file:///D:/Vscode_workspace/Stock/scripts/stock_categories.json) 計算 57 個產業每日平均報酬與強度。

---

## 2. 模型訓練與預測推論 (Model Training & Inference)

* **模型架構**：[train.py](file:///D:/Vscode_workspace/Stock/scripts/train.py) 訓練三個 LightGBM 分類模型，預測未來 1、2、3 天之強勢/弱勢/中性標籤。
* **大跌樣本懲罰 (Loss Weighting)**：若未來 3 天內最低收益率低於 `SAMPLE_WEIGHT_DROP_THRESHOLD` (預設 -5%)，將該樣本權重乘以 `SAMPLE_WEIGHT_PENALTY` (預設 2.0)，以避開大跌風險個股。
* **特徵對齊**：推論時的特徵必須與 [feature_cols.json](file:///D:/Vscode_workspace/Stock/models/feature_cols.json) 的特徵名稱 100% 對齊。
* **預測推理與掛單**：[inference.py](file:///D:/Vscode_workspace/Stock/scripts/inference.py) 計算未來 3 天的多空分數，並對齊台灣股市 Tick Size 四捨五入，依信心強度提供開盤建議掛單限價。

---

## 3. 交易風控與最佳化 (Risk Control & Optimization)

* **風控優化**：[optimize_trading_params.py](file:///D:/Vscode_workspace/Stock/scripts/optimize_trading_params.py) 優化交易風控參數並存至 [best_trading_params.json](file:///D:/Vscode_workspace/Stock/configs/best_trading_params.json)。
* **Walk-Forward 參數決策**：在 WFO 調參時，最終部署參數必須使用中位數 (Median)，嚴禁使用平均值 (Mean)，以抵禦極端噪訊。
* **市況過濾器**：`REGIME_*` 參數（在 [config.py](file:///D:/Vscode_workspace/Stock/config.py) 中）依昨日大盤狀態動態調整買入門檻。僅在未指定 `buy_threshold` 時生效，以防覆寫敏感度掃描 (OFAT)。

---

## 4. 研發與生產工作流 (Research & Production Workflow)

* **中央常數控制**：所有常數、門檻與交易參數必須寫在 [config.py](file:///D:/Vscode_workspace/Stock/config.py) 中，嚴禁在腳本中寫死。
* **無損備份機制**：執行研發實驗前，必須備份 `config.py`、`configs/best_factors.json` 和 `configs/best_trading_params.json` 為 `.workflow.bak`。`finally` 區塊必須 100% 自動還原。
* **雙階段實驗**：[run_workflow_experiment.py](file:///D:/Vscode_workspace/Stock/run_workflow_experiment.py) 協調模式 A (研究模式，鎖定 2025-08-01) 與模式 B (生產模式，全數據滾動重訓)，包含 Stage C 潔淨 OOS 風控驗證，以評估風控參數泛化防禦力。
* **生產部署**：實盤運行前需將 [best_trading_params_mode_b.json](file:///D:/Vscode_workspace/Stock/configs/best_trading_params_mode_b.json) 複製覆寫為預設的 [best_trading_params.json](file:///D:/Vscode_workspace/Stock/configs/best_trading_params.json)，接著每日以 [Auto_RUN.py](file:///D:/Vscode_workspace/Stock/Auto_RUN.py) 運行全流程。

**Think Before Coding** — Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

**Simplicity First** — Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked. No abstractions for single-use code.
- No error handling for impossible scenarios.

**Surgical Changes** — Touch only what you must. Clean up only your own mess.
- Don't refactor things that aren't broken.
- Match existing style.