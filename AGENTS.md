# 台灣股市量化交易系統 - 架構與 AI 導航指南 (AGENTS.md)

本文件專為 AI Agent 設計。當您進入本專案時，請先閱讀此文件以快速掌握系統全貌、資料流與核心模組，避免重新分析程式碼而浪費 Token。

---

## 1. 系統設計理念與工作流 (Workflow)
這是一個**一條龍全自動化的台股量化預測與模擬交易系統**，設計成能在 T+1 / T+2 的藍圖下，根據每日收盤後的資料，自動化產出未來的交易預測並執行回測模擬。

### 核心升級：絕對收益系統 (2026/06 升級)
系統內建五大實戰級別機制，徹底解決了傳統相對選股「在市場崩盤時也滿倉跌較少股票」與「實戰交易磨擦」的盲點：
1. **方案 B (總體與板塊特徵)：** 特徵工程自動注入全市場日報酬平均、市場寬度（上漲比例）及其 5日/20日 滾動趨勢；同時透過 `scripts/stock_categories.json` 計算各產業板塊的每日平均表現與滾動強度，給予模型宏觀視野。
2. **方案 C (絕對與相對混合標籤)：** 標籤設計強制規定強勢股（`Label=2`）除滿足相對排名前 20% 外，**未來絕對報酬率必須大於 0%**，否則歸為中性。
3. **方案 D (大跌樣本權重懲罰)：** 在 `train.py` 訓練 LightGBM 時，當樣本在未來 3 天內有任一天大跌 <= `SAMPLE_WEIGHT_DROP_THRESHOLD` (預設 -5%)，將該樣本權重乘以 `SAMPLE_WEIGHT_PENALTY` (預設 2.0)，強迫模型在選股時避開這類高回撤標的。
4. **方案 E (智慧限價溢價與台灣 Tick Size 對齊)：** 在 `inference.py` 和 `trading_sim.py` 中，根據 D1 信心分數動態溢價 `+1.5% ~ +2.5%` 並自動對齊台灣股市的報價升降單位 (Tick Size) 進行買入限價搓合。
5. **風控優化：** 這些機制與 `trading_sim.py` 的信心分數門檻相輔相成。本系統經過 6年 (2020-2026) 全週期數據回測，調校出平衡穩健的黃金風控參數（5日均回報門檻 -1.0% / 上漲家數比例 30.0% / 個股停損 -8.0%），在大變動年份（如 2022 熊市）依然能有效保護資產，實現真正的絕對收益！
6. **產業過濾對齊與節流優化**：`auto_pipeline.py` 動態導入 `config.py` 中的 `TRAIN_INDUSTRIES` 設定，避免下載與特徵化不需訓練的產業，大幅節省 FinMind API 額度並確保全系統篩選一致。

### 雙階段參數最佳化研發流程（調參工作流）
為防範前視偏差（Lookahead Bias）與對測試集過度擬合，最佳化調參應遵循以下雙階段工作流：
1. **第一階段：技術指標與因子最佳化 (Feature Tuning)**
   - 執行 `auto_pipeline.py -s o` (調用 `optimize_factors.py`)，在歷史訓練區間上通過 Optuna TPE 尋找最佳技術指標參數，輸出至 `configs/best_factors.json`。
   - 隨後必須執行 `auto_pipeline.py -s f` (重建特徵 parquet) 與 `auto_pipeline.py -s t` (重新訓練 LightGBM 模型)，將新因子套用到特徵與模型中。
2. **第二階段：交易策略與避險風控最佳化 (Trading/Risk Tuning)**
   - 執行 `scripts/optimize_trading_params.py`，載入已應用新因子的特徵 parquet 與模型，在相同歷史區間上搜尋能最大化平滑 Calmar 比率（報酬率 / 最大回撤）的交易風控參數組合，輸出至 `configs/best_trading_params.json`。
   - 此 JSON 檔會由 `config.py` 在載入時自動加載並覆寫常數，使全系統（回測、模擬、推理）即時自動套用新風控值。

### 核心流水線 (Pipeline) 流程
`多源爬蟲 (scraper.py)` -> `超參數最佳化 (optimize_factors.py)` -> `特徵工程 (feature_engineering.py)` -> `模型訓練 (train.py)` -> `推論預測 (inference.py)` -> `交易回測 (trading_sim.py)`

---

## 2. 核心模組與 Python 腳本對照表

### ⚙️ 中央控制面板 (Centralized Panel)
- **`config.py`**: **專案唯一設定檔**。所有訓練產業、交易門檻、爬蟲區間、超參數、資源核心數均在此設定。

### 🕷️ 資料獲取與清洗 (Data Scraping & ETL)
- **`scripts/scraper.py`**: **整合資料維護中心**。
  - **免費資料**: TWSE (股價、籌碼、資券、借券、本益比、當沖、外資持股、信用限額)、TAIFEX (外資台指期未平倉)、TDCC (每週大戶持股)。
  - **FinMind (需 Token)**: 月營收、三大報表、股利。
  - **特色**: 具備嚴格的 9 檔案歷史跳過機制、15天快取、FinMind 空資料快取機制 (`no_finmind_data.json`)，以及失敗略過機制 (`failed_dates.json`)。
  - **限額容錯阻斷**：支援 `SKIP_ON_FINMIND_LIMIT="1"` 環境變數，觸發限速拋出 `FinMindLimitExceeded` 異常終止，主控腳本捕獲並自動跳過。
  - **CLI 整合**：
    - 正常抓取 (無參數)：增量下載歷史資料庫。
    - `-p <sid>` / `--patch <sid>`：針對性補抓特定股票 (例如2330) 缺漏之財報 (原 `patch_finmind.py`)。
    - `-fc` / `--fc` / `--fetch-categories`：更新台股產業分類對照表 (原 `fetch_categories.py`)。
    - `-c` / `--check`：掃描並刪除損毀或價格異常的歷史資料日 (原 `check_data.py`)。

### ⚙️ 特徵工程 (Feature Engineering)
- **`scripts/feature_engineering.py`**: **核心特徵生成模組**。
  - 計算個股技術指標：MA、KD、RSI、MACD、布林通道、ATR。
  - **總體與板塊情緒（方案 B）**：每日計算全市場平均報酬與上漲比例，並結合滾動均線；自動載入產業映射，每日計算板塊平均報酬與 rolling 情緒。
  - **混合型預測標籤（方案 C）**：預測未來 1~3 天收益率，強勢標籤（2）強制鎖定為絕對上漲股，最終輸出 `features_combined.parquet`。
  - **效能優化**：Step 1 自動偵測當天是否有收盤價日報 CSV，**自動剔除國定假日與不開市交易日**，大為節省 joblib 平行處理無效日期的開銷。
  - **CLI 整合**：支援 `--backtest <YYYYMMDD>` 單日回測，快速驗證當天模型的預測精度。

### 🤖 模型最佳化、訓練與預測 (Modeling & Inference)
- **`scripts/optimize_factors.py`**: **特徵與超參數最佳化腳本**。使用 `Optuna` 框架，在不偷看未來的驗證集上，動態尋找勝率最高的技術指標參數組合並輸出 `configs/best_factors.json`。
  - **自動板塊篩選**：載入數據時會依據 `config.py` 中的 `TRAIN_INDUSTRIES` 進行股票過濾，確保尋找出來的最佳參數是專門契合勾選的產業特性。
- **`scripts/train.py`**: **LightGBM 模型訓練腳本**。讀取 parquet 特徵檔，依照時間軸進行 strict 日期分位切割，訓練 LightGBM 模型並將特徵欄位儲存至 `models/feature_cols.json`。
  - **訓練產業選擇 (`TRAIN_INDUSTRIES`)**：在 `config.py` 設定勾選欲訓練的板塊，排除其他產業雜訊，同時無條件保留 `Stocks.txt` 自選股。
  - **樣本大跌懲罰機制**：讀取 `SAMPLE_WEIGHT_DROP_THRESHOLD` 與 `SAMPLE_WEIGHT_PENALTY` 對未來大跌樣本進行加權懲罰，優化模型避險能力。
- **`scripts/inference.py`**: **預測推論腳本**。載入最新一天資料與模型，推論目標清單股票未來 3 天的多空分數並印出排名表。
  - **實戰倉位與板塊對齊**：自動與策略的「最多 5 檔持倉上限、Day1 >= 10.0% 買入、Day3 < 0% 賣出、-8% 停損」策略對齊；同時自動以 `config.py` 設定過濾股票，只推薦您選中的產業。
  - **智慧下單建議與限價溢價**：自動區分 `Stocks.txt` 內實質持倉與自選股，動態計算明日可用空位，並在每日收盤後根據 D1 信心分數與 `ORDER_MARKUP_*` 溢價設定，結合 `round_to_tick` 函數，提供明日「開盤買進/賣出」具體掛單價格建議。
- **`scripts/backtest.py`**: **時光機回測模式腳本**。指定單一基準日，以該日前的數據訓練模型，回測該日未來 3 天全市場及自選股的實質利潤與勝率。
- **`scripts/analyze_regime_stability.py`**: **訊號穩定性與市場狀態診斷器**。
  - 計算 `BACKTEST_DATE` 分界後的「樣本外 (OOS)  」訊號健康指標：RankIC、第一名組別超額 Alpha 獲利率、特徵 PSI 漂移度。
  - 輸出將以 IS/OOS 分層表格顯示，含 Bear/Bull/整體 All 行，便於實驗腳本自動解析。
  - 執行時機：模式 A 訓練完成後驗證 OOS 泛化力；實盤期每月定期監控特徵漂移（PSI >= 0.25 且 RankIC 衰退 → 需重新設計因子）。

### ☁️ 雲端備份與自動化控制 (Cloud Sync & Master Control)
- **`scripts/StockSync.py`**: **雲端自動備份腳本**。使用 rclone 把 `predictions/` 下產生的預測建議文字檔同步拷貝至 Google Drive (RCLONE_REMOTE_NAME 遠端)。
- **`Auto_RUN.py`**: **生產流程主控腳本**。一鍵順序執行全流程（scraper ➔ pipeline ➔ StockSync）。
  - **CLI 步驟分流**：支援 `-s` / `--step` 參數，可帶入完整值或簡碼（`download`/`d`、`predict`/`p`、`backup`/`b`）執行單一步驟。
- **`auto_pipeline.py`**: **機器學習研發流水線入口**。
  - **CLI 步驟分流**：支援 `-s` / `--step` 參數，可帶入完整值或簡碼（`optimize`/`o`、`feature`/`f`、`train`/`t`、`inference`/`i`）執行單一模型訓練步驟。其中單獨執行 `optimize`/`o` 步驟時會自動無視 `config.py` 中的 `RUN_OPTIMIZATION` 限制強行啟動因子調參。
- **`run_workflow_experiment.py`**: **雙階段實驗自動化控制台（沙盒時光機）**。一鍵完成模式 A（研究期，截斷 2025-08-01，包含因子最佳化、模型 A 訓練、訊號穩定性診斷、歷史風控調參、樣本外 OOS 回測）與模式 B（生產推理期，全數據重訓最新模型、全週期風控參數優化、全週期回測、推理預測）的所有研發流程。
  - **具備 `try...finally` 異常安全機制**與 `Ctrl+C` 子進程主動關閉功能，100% 還原主系統。
  - **Checkpoint 斷點續傳**：兩模式各步驟均有備份檔儲存中間結果（`configs/best_factors_mode_a.json`、`configs/best_trading_params_mode_a.json`、`configs/best_trading_params_mode_b.json` 等）。中途崩潰後重新執行同一指令即可從斷點續跑，無需重跑 Optuna 調參。
  - **CLI 參數**：`-f`/`--factor_trials`、`-t`/`--trading_trials`、`-c`/`--capital`、`--skip_factor_opt`（跳過因子調參）、`--fresh`（忽略所有 Checkpoint 強制全部重跑）。
  - 另存各模式參數與績效對比報告 [reports/workflow_experiment_report.md](reports/workflow_experiment_report.md)。
- **`run_workflow_experiment_guide.md`**: **雙階段自動化實驗指南**。詳細記述實驗流程、CLI 參數、腳本調用關係、數據流向與參數套用發布行動指南。包含：如何判讀實驗報告（RankIC / Alpha / PSI / MDD / Calmar 判斷標準）、OOS 績效不佳時的分層調整策略、台股市況背景與風控旋鈕調整方向對照表、Checkpoint 檔案對照表與何時建議重新執行實驗。
- **安全性與敏感資料防護**：
  - **`rclone.exe` 執行檔**：需另行下載，可放置於虛擬環境的 `venv/Scripts/` 目錄中以方便調用。
  - **敏感金鑰防外洩**：任何包含 rclone 登入 OAuth Token 的 `config`、`.config` 資料夾或 `rclone.conf` 設定檔，**屬於極度敏感金鑰，已列入 Git 與 AI 忽略清單，絕對禁止提交與外洩**。

### 📊 實戰回測與模擬交易 (Simulation)
- **`trading_sim.py`**: **實戰級交易模擬回測器 (Out-of-Sample)**。
  - 模擬真實交易：支持自訂時間區間、初始資金、買入多空信心分數門檻、個股固定停損（如 `-8%`）或轉弱出場。
  - **動態參數覆蓋與限價搓合**：支援透過 CLI 參數（如 `--panic_ma5`, `--panic_breadth`, `--stop_loss` 等）直接覆蓋中央 `config.py` 設定；模擬交易買入時套用與 `inference.py` 一致的信心分數限價溢價與台灣 Tick Size 報價對齊搓合，避免過於樂觀的回測結果。
  - **真實 T+2 交割機制模擬**：引進 `pending_settlements` 待交割帳戶，將資金拆分為「可用資金（購買力）」與「銀行實質餘額」，以正確計算台股賣股後資金同日再買入的實際運作，以及 T+2 交割時對銀行餘額和權益的真實影響。
  - 資金與資產一致性：日終統一將當天發生的所有交易明細之 `Current_Cash`、`Stock_Value` 與 `Total_Equity` 更新為收盤後的最終狀態。
  - 輸出與降級：支援 Excel 多分頁高規格匯出，並內建 CSV 穩健降級機制。
- **`scripts/optimize_trading_params.py`**: **交易策略與避險參數自動調參器 (Optuna TPE)**。
  - 貝葉斯超參數調參：使用 Optuna 自動化對 `trading_sim.py` 的風控與策略參數組合進行最佳化，以最大化平滑 Calmar 比率為目標，將最優參數結果輸出至 `configs/best_trading_params.json`。

### 🛠️ 共享核心工具與測試 (Utilities & Tests)
- **`scripts/utils.py`**: **全系統共享股票解析與過濾工具**。
  - 統一 `Stocks.txt` 的解析邏輯（格式 A/B/C 成本與股數欄位），支援 subdirectory 與 fallback。
  - 提供 `filter_stocks_by_train_industries(df)` 統一過濾器，自動以 `config.py` 的設定篩選 DataFrame 中的個股（支持 int/str 類型 stock_id 與自選股強制保留）。
- **`scripts/tools/clean_stocks.py`**: **自選股清單清理工具**。讀取 `Stocks.txt`，對照 `scripts/stock_categories.json` 過濾無效的股票代號，並自動處理重複項。
- **`tests/test_pipeline.py`**: **全系統整合測試腳本**。驗證 18 項核心邏輯，包含日期切割、特徵檔完整性、`utils.py` 解析器單元測試等。
- **`tests/test_scraper.py`**: **爬蟲單元測試腳本**。快速測試證交所與期交所 API 下載功能，自動整合 `skip_dates.json` 防呆略過機制。
- **`tests/test_finmind.py`**: **FinMind 資料單元測試**。

---

## 3. 資料目錄結構與核心快取檔案 (Data & Cache)

### `data/` 目錄結構
- **`raw_price/`**: 每日收盤行情與成交量 (`*_price.csv`，來自 TWSE `MI_INDEX`)。
- **`raw_chips/`**: 籌碼與交易面資料。三大法人買賣超 (`*_chips.csv`)、當沖統計 (`*_daytrading.csv`)、外資持股比例 (`*_fini_holding.csv`)。
- **`raw_margin/`**: 信用交易資料。融資融券餘額 (`*_margin.csv`)、借券餘額 (`*_sbl.csv`)、額度總量管制 (`*_credit_limit.csv`)。
- **`raw_twse_per/`**: 個股官方本益比、殖利率及淨值比 (`*_twse_per.csv`)。
- **`raw_taifex/`**: 期貨大盤外資多空未平倉淨額 (`*_taifex_inst.csv`)。
- **`raw_shareholding/`**: 每週集保戶股權分散表 (`*_shareholding.csv`)。
- **`raw_financial/`**: FinMind 基本面財報。營收、三表季報與歷年股利。

### 核心設定與快取檔案 (.json)
1. **`configs/best_factors.json`**: 記錄 Optuna 找出的最佳技術指標參數，由 `feature_engineering.py` 自動讀取。
2. **`configs/best_trading_params.json`**: 記錄 Optuna 找出的最佳交易策略與風控參數，作為策略調整與覆蓋依據。
3. **`configs/best_factors_mode_a.json`**: 實驗模式 A 找出的最佳因子備份。實驗結束後可將此檔複製為 `configs/best_factors.json` 使用。
4. **`configs/best_trading_params_mode_a.json`**: 實驗模式 A 的最佳風控參數備份（在 2021-01-02 ~ 2025-08-01 歷史市況下調參，天然偏保守）。
5. **`configs/best_trading_params_mode_b.json`**: 實驗模式 B 的最佳風控參數備份（涵蓋大牛市全週期調參，更適合當前市場）。**實盤生產推薦直接複製為 `configs/best_trading_params.json` 使用。**
6. **`reports/workflow_experiment_results.json`**: 實驗進度快取檔。記錄各步驟的完成結果，支援中斷後續傳（Checkpoint 機制）。
7. **`scripts/stock_categories.json`**: 產業分類與 ETF 清單，為特徵工程的板塊情緒、爬蟲跳過 ETF 任務及交易模擬器股票名稱轉換的共用依據。
8. **`models/feature_cols.json`**: 記錄模型訓練當下的所有特徵名稱，確保推論特徵順序與數量 100% 一致。
9. **`data/failed_dates.json`**: TWSE/TAIFEX 的失敗計數器。失敗超過 3 次則判定為「無開市/假補班」並永久略過。
10. **`data/no_finmind_data.json`**: FinMind 股票空值快取（快取 90 天）。
11. **`data/skip_dates.json`**: 官方爬蟲跳過快取，記錄無資料或格式異常的日期，並註記 `reason`以供除錯。
12. **`data/missing_fm_datasets.json`**: FinMind 局部財報缺漏快取。

---

## 4. 未來擴展方向與 AI 注意事項
1. **資料維護**: 若發現特徵檔中缺少某項資料，請優先檢查 `scraper.py` 的跳過機制，並利用 `scraper.py --check` 進行檢驗與修復。
2. **常數宣告限制**: 任何常數、參數或調參範圍變數，都必須宣告在 [config.py](config.py) 中，嚴禁在個別腳本中進行分散式寫死。
3. **參數動態命名**: 在新增技術指標時，請務必與 `optimize_factors.py` 聯動，確保 `feature_engineering.py` 所產生的特徵欄位名稱是動態且可被模型讀取的。
4. **文件連結相對路徑規範**: 任何在專案說明文檔（如 `README.md`、`AGENTS.md`、`run_workflow_experiment_guide.md` 等）中新增、修改或提及專案內檔案的跳轉連結時，**必須且只能使用相對路徑**（例如 `[config.py](config.py)`），**嚴禁使用包含本地絕對路徑的 file:/// 協議**（例如 `[config.py](file:///D:/Vscode_workspace/Stock/config.py)`）。這能確保說明文檔在不同環境與開發者電腦之間具備完全的移植性。 (注意：此規範僅適用於專案內的說明文件，AI Agent 在與使用者交談的對話視窗中仍應遵循 file:/// 協議提供 clickable 連結)。
5. **實驗腳本的 Checkpoint 續傳機制**：`run_workflow_experiment.py` 使用多檔 JSON 作為 Checkpoint。若 AI 發現實驗未完成或報告顯示「未完成」，應檢查 `reports/workflow_experiment_results.json` 的內容：若 `mode_b` 為空 `{}`，代表模式 B 尚未執行；若模擬結果為 `0.0`，代表先前有編碼解析錯誤，實驗會自動重跑該步驟。
6. **模式 A 保守 vs 模式 B 過鬆的正常行為**：模式 A 的風控參數在包含 2022 熊市的歷史數據上調出，天然偏保守。模式 B 則包含大牛市數據。若 AI 需評估實驗結果，模式 B 的風控參數比模式 A 更寬鬆是正常現象，不代表模式 B 過擬合。
7. **測試執行免除規範**：當變更僅限於說明文檔、註解或組態設定檔（如 `AGENTS.md`、`README.md`、`.gitignore`、`.agyignore` 等），且與核心 `.py` 執行檔沒有任何關聯時，**無須執行 `tests/test_pipeline.py`**，以節省開發時間與系統開銷。