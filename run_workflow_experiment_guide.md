# 🎛️ run_workflow_experiment.py 使用說明與技術架構指南

本指南詳細介紹 `run_workflow_experiment.py` 腳本的設計理念、調用架構、執行參數、數據加載機制以及如何將實驗結果套用至您的實盤生產系統中。

---

## 🎯 1. 核心設計理念與工作流

量化交易系統的致命傷是**前視偏差 (Lookahead Bias)** 與**過擬合 (Overfitting)**。為了防止系統在歷史數據上調出「看似完美」但在實盤中迅速失效的參數，本腳本實現了**雙階段時光機沙盒實驗**：

*   **🟢 模式 A (研究模式 - 樣本外測試)**：
    將時光倒回 `2025-08-01`。模型完全不知道之後大盤會漲到四萬點。我們在 `2025-08-01` 之前的數據上調參並訓練模型，隨後在 `2025-08-02 ~ 2026-06-05` 這段未知的「超級牛市」中進行回測，用以**驗證策略在全新市場環境下的泛化防禦力**。
*   **🔵 模式 B (生產模式 - 全數據覆蓋)**：
    模型每天重訓並吸收包含最新大牛市的完整特徵。同時，風控優化器也在包含這段牛市的全週期上調參，用以**優化出一組最契合當前大波動行情的黃金避險參數**，防止避險過度敏感而在牛市中少賺。

---

## 🔌 2. 系統調用架構圖

本實驗腳本扮演「總控制台」的角色，透過 `subprocess` 沙盒進程安全地調用系統中各個模組：

```mermaid
flowchart TD
    RunExp["🎛️ run_workflow_experiment.py\n(備份、配置更新與還原)"]
    
    subgraph ModeA ["🟢 模式 A (截斷 2025-08-01)"]
        A_Opt["1. 因子最佳化\nauto_pipeline.py -s o"] --> A_Feat["2. 重建特徵\nauto_pipeline.py -s f"]
        A_Feat --> A_Train["3. 訓練模型 A\nauto_pipeline.py -s t"]
        A_Train --> A_Diag["4. 訊號穩定性分析\nanalyze_regime_stability.py"]
        A_Diag --> A_Param["5. 風控最佳化\noptimize_trading_params.py"]
        A_Param --> A_Sim["6. OOS 模擬交易\ntrading_sim.py"]
    end
    
    subgraph ModeB ["🔵 模式 B (全數據無截斷)"]
        B_Feat["1. 重建特徵\nauto_pipeline.py -s f"] --> B_Train["2. 重訓模型 B\nauto_pipeline.py -s t"]
        B_Train --> B_Param["3. 風控最佳化\noptimize_trading_params.py"]
        B_Param --> B_Sim["4. 全週期模擬交易\ntrading_sim.py"]
        B_Train --> B_Infer["5. 推理預測\nauto_pipeline.py -s i"]
    end

    RunExp -->|設定 BACKTEST_DATE=20250801| ModeA
    RunExp -->|設定 BACKTEST_DATE=None| ModeB
    ModeA -->|寫入與備份| TA_Param["best_trading_params_mode_a.json"]
    ModeB -->|寫入與備份| TB_Param["best_trading_params_mode_b.json"]
```

---

## 📝 3. 執行參數與命令列選項 (CLI Options)

本腳本提供了靈活的命令列參數，方便您在不修改代碼的情況下調整實驗規模：

| 短參數 | 長參數 | 類型 | 預設值 | 說明 |
| :--- | :--- | :--- | :--- | :--- |
| `-f` | `--factor_trials` | `int` | `30` | 模式 A 中 `optimize_factors.py` 的最大搜尋輪數。 |
| `-fe`| `--factor_early_stopping`| `str`| `"15"` | 因子搜尋無進展多少輪後自動終止 (`None` 代表不啟用)。 |
| `-t` | `--trading_trials` | `int` | `100` | 模式 A / 模式 B 中 `optimize_trading_params.py` 的最大搜尋輪數。 |
| `-te`| `--trading_early_stopping`| `str`| `"30"` | 風控搜尋無進展多少輪後自動終止 (`None` 代表不啟用)。 |
| `-c` | `--capital` | `int` | `2000000`| 模擬交易與風控調參時的初始資金量。 |
| ❌ | `--skip_factor_opt` | `bool`| `False` | 模式 A 中是否跳過因子調參，直接沿用現有的 `best_factors.json`。 |

> [!TIP]
> 預設的因子搜尋設為 `30` 輪、風控搜尋設為 `100` 輪，是為了讓您能在數十分鐘內快速跑完完整實驗。在進行正式的研究與發表前，建議將風控調參設定為 `200` ~ `600` 輪以獲得最優解。

---

## 🔍 4. 詳細執行步驟、參數與載入數據

### 📂 步驟 1：安全備份與初始化
*   **載入數據與狀態**：檢測根目錄是否存在 `config.py`、`best_factors.json` 和 `best_trading_params.json`。
*   **執行動作**：將上述三檔複製並備份為 `.workflow.bak` 檔。如果原本沒有 `best_factors.json` 或 `best_trading_params.json`，則記錄其 nonexistent 狀態。
*   **目的**：確保在實驗過程中，不論發生程式崩潰、斷電或手動終止，都能在退出時 100% 還原主系統的狀態。

---

### 🟢 步驟 2：模式 A (研究模式) 運行

#### 2.1 因子指標優化 (Factor Optimization)
*   **調用指令**：`python auto_pipeline.py -s o`
*   **寫入配置**：
    *   將 `config.py` 中的 `BACKTEST_DATE` 設為 `"20250801"`。
    *   將 `RUN_OPTIMIZATION` 設為 `True`，將 `OPTIMIZATION_TRIALS` 設為 `-f` 參數值，將 `EARLY_STOPPING_ROUNDS` 設為 `-fe` 參數值。
*   **載入數據**：加載 `data/raw_price/` 的每日收盤行情與 `config.py` 中的 `TRAIN_INDUSTRIES` 對應股票。
*   **執行內容**：Optuna 在 2025-08-01 之前的訓練集上尋找最佳技術指標參數（MA、RSI、MACD 等均線窗口），以最大化分類勝率。
*   **輸出產出**：寫入最佳因子到 `best_factors.json`。

#### 2.2 重建特徵與訓練模型 A
*   **調用指令**：`python auto_pipeline.py -s f` 與 `python auto_pipeline.py -s t`
*   **載入數據**：
    *   讀取剛生成的 `best_factors.json`。
    *   讀取 `data/` 目錄下三大法人籌碼、融資融券、大戶持股與基本面季報。
*   **執行內容**：以最佳因子重新計算所有股票的技術指標與總體/板塊特徵，並只採用 `2025-08-01` 之前的數據訓練 LightGBM 分類模型。
*   **輸出產出**：儲存模型至 `models/lgbm_model_1.txt`，特徵欄位順序存至 `models/feature_cols.json`。

#### 2.3 訊號穩定性診斷
*   **調用指令**：`python scripts/analyze_regime_stability.py`
*   **載入數據**：載入整個特徵 Parquet 檔與剛剛訓練好的模型 `lgbm_model_1.txt`。
*   **執行內容**：計算 2025-08-01 之後「樣本外測試集 (OOS)」的 RankIC、第一名組別超額 Alpha 獲利率、特徵 PSI 漂移度。
*   **輸出產出**：將訊號診斷日誌寫入 `reports/regime_stability_report.txt`，實驗腳本會將其複製備份為 [reports/mode_a_regime_stability_report.txt](reports/mode_a_regime_stability_report.txt)。

#### 2.4 歷史風控參數調參
*   **調用指令**：`python scripts/optimize_trading_params.py -t <trials> -s 2021-01-02 -e 2025-08-01 -c <capital>`
*   **寫入配置**：將 `config.py` 中的 `EARLY_STOPPING_ROUNDS` 設為 `-te` 參數值。
*   **載入數據**：加載 `features_combined.parquet`、`models/lgbm_model_1.txt` 與歷史大盤指數。
*   **執行內容**：限制在 2025-08-01 以前的多空市況中（**絕對不讓 Optuna 看到未來牛市**），尋找最優風控配置（如停損線、避險紅燈）。
*   **輸出產出**：最佳參數寫入 `best_trading_params.json`，實驗腳本將其備份為 `best_trading_params_mode_a.json`。

#### 2.5 樣本外模擬交易 (時光機回測)
*   **調用指令**：`python trading_sim.py -s 2025-08-02 -e 2026-06-05 -c <capital>`
*   **載入數據**：加載 Model A 模型、`best_trading_params.json`（模式 A 風控）、`Stocks.txt` 自選股與 OOS 期間的股價及大盤數據。
*   **執行內容**：在 `2025-08-02 ~ 2026-06-05` 超級牛市中模擬實盤交易，模擬台股 T+2 交割機制，計入手續費與稅金。
*   **輸出產出**：解析回測 terminal 輸出的「區間報酬」與「最大回撤」，寫入實驗日誌。

---

### 🔵 步驟 3：模式 B (實盤模式) 運行

#### 3.1 沿用因子與特徵重建
*   **調用指令**：`python auto_pipeline.py -s f`
*   **寫入配置**：
    *   將 `config.py` 中的 `BACKTEST_DATE` 設為 `None`。
    *   將 `RUN_OPTIMIZATION` 設為 `False`（沿用模式 A 優化出的 `best_factors.json`，以節省時間）。
*   **執行內容**：將最完整的歷史數據（截至今日）合併重建。由於 `BACKTEST_DATE = None`，流水線會自動將特徵分界點設置在「一年前的最近交易日」，實現動態滾動。

#### 3.2 重訓模型 B (包含牛市)
*   **調用指令**：`python auto_pipeline.py -s t`
*   **執行內容**：重新訓練 LightGBM 模型。此時訓練集會自動滾動涵蓋 2025-08 到 2026-06 的超級牛市數據。模型大腦擁有了最新高點大盤的特徵分佈與選股記憶。
*   **輸出產出**：覆蓋更新 `models/lgbm_model_1.txt`。

#### 3.3 全週期風控參數優化 (覆蓋牛市)
*   **調用指令**：`python scripts/optimize_trading_params.py -t <trials> -s 2023-01-01 -e 2026-06-01 -c <capital>`
*   **寫入配置**：將 `config.py` 中的 `EARLY_STOPPING_ROUNDS` 設為 `-te` 參數值。
*   **執行內容**：在涵蓋這段超級牛市的完整週期上進行風控優化。Optuna 此時能夠親眼見識到牛市的劇烈個股波動和大盤的極端強勢，藉以放寬避險紅燈與停損範圍，以免在牛市中被過早震盪洗出場。
*   **輸出產出**：最佳參數寫入 `best_trading_params.json`，實驗腳本將其備份為 `best_trading_params_mode_b.json`。

#### 3.4 全週期模擬交易回測
*   **調用指令**：`python trading_sim.py -s 2023-01-01 -e 2026-06-05 -c <capital>`
*   **執行內容**：套用 Model B 與模式 B 風控進行全週期回測，並解析其回報與回撤。

#### 3.5 生成明日實盤推理下單建議
*   **調用指令**：`python auto_pipeline.py -s i`
*   **執行內容**：以最新重訓完的模型 B，結合剛剛生成的模式 B 最佳風控參數，直接為明天的台股交易日輸出具體的買賣掛單限價指引。

---

### 📊 步驟 4：對比報告生成與安全還原
*   **執行內容**：
    1.  讀取模式 A 訊號診斷報告中的相關 RankIC 指標。
    2.  將模式 A 與模式 B 的參數、報酬與 MDD 以表格對比格式寫入 [reports/workflow_experiment_report.md](reports/workflow_experiment_report.md)。
    3.  從備份檔 `.workflow.bak` 中，完美還原 `config.py`、`best_factors.json` 和 `best_trading_params.json` 到最原始狀態，確保實盤無任何副作用。

---

## ⚙️ 5. 參數套用 (Release) 行動指南

當您執行完實驗並仔細評估了 [reports/workflow_experiment_report.md](reports/workflow_experiment_report.md) 後，如何將新參數上線？

### 🔹 情境一：我想採用模式 B 優化出的黃金風控配置
這通常是最佳選擇，因為模式 B 經歷了最新大牛市的洗禮，其風控配置最適應當前市場：
1. 在根目錄中，將 `best_trading_params_mode_b.json` 複製並**重新命名為 `best_trading_params.json`**（覆蓋原本的檔案）。
2. 下午收盤後，照常執行 `python Auto_RUN.py`。系統會自動偵測並讀取此 JSON 檔，明日推理掛單將直接套用這組黃金風控。

### 🔹 情境二：我想採用實驗優化出的最優因子 (MA, RSI 週期等)
如果您重新執行了因子最佳化（即沒有使用 `--skip_factor_opt`），且發現新因子的 RankIC 顯著高於舊因子：
1. 因子參數已經在實驗中被寫入 `best_factors.json`（已在實驗結束時被還原）。您可以在對比報告中查看參數，或將實驗過程產生的備份因子套用。
2. 讓系統正式使用新因子，您必須手動執行特徵工程與模型重新訓練，以將新因子嵌入模型中：
   ```powershell
   # 1. 根據新因子重新計算特徵
   python auto_pipeline.py -s f
   # 2. 重新訓練模型
   python auto_pipeline.py -s t
   ```
