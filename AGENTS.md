# 台灣股市量化交易系統 - 架構與 AI 導航指南 (AGENTS.md)

本文件專為 AI Agent 設計。當您進入本專案時，請先閱讀此文件以快速掌握系統全貌、資料流與核心模組，避免重新分析程式碼而浪費 Token。

---

## 1. 系統設計理念與工作流 (Workflow)
這是一個**一條龍全自動化的台股量化預測與模擬交易系統**，設計成能在 T+1 / T+2 的藍圖下，根據每日收盤後的資料，自動化產出未來的交易預測並執行回測模擬。

### 核心升級：絕對收益系統 (2026/06 升級)
系統內建兩大實戰級別防線，徹底解決了傳統相對選股「在市場崩盤時也滿倉跌較少股票」的盲點：
1. **方案 B (總體與板塊特徵)：** 特徵工程自動注入全市場日報酬平均、市場寬度（上漲比例）及其 5日/20日 滾動趨勢；同時透過 `stock_categories.json` 計算各產業板塊的每日平均表現與滾動強度，給予模型宏觀視野。
2. **方案 C (絕對與相對混合標籤)：** 標籤設計強制規定強勢股（`Label=2`）除滿足相對排名前 20% 外，**未來絕對報酬率必須大於 0%**，否則歸為中性。
3. **風控效果：** 這兩者與 `trading_sim.py` 的信心分數門檻相輔相成。在大盤大跌日，全市場多空分數會自適應下滑，系統會**自動判定無股可買而 100% 空倉避險（報酬率 0.00%）**，完美避開系統性崩盤！

### 核心流水線 (Pipeline) 流程
`多源爬蟲 (Scraping)` -> `超參數最佳化 (Optuna)` -> `特徵工程 (ETL)` -> `模型訓練 (LightGBM)` -> `推論預測 (Inference)` -> `交易回測 (Simulation)`

---

## 2. 核心模組與 Python 腳本對照表

### ️ 資料獲取與清洗 (Data Scraping & ETL)
- **`scripts/scraper.py`**: **核心爬蟲模組**。
  - **免費資料**: TWSE (股價、籌碼、資券、借券、本益比、當沖、外資持股、信用限額)、TAIFEX (外資台指期未平倉)、TDCC (每週大戶持股)。
  - **FinMind (需 Token)**: 月營收、三大報表、股利。
  - **特色**: 具備嚴格的 9 檔案歷史跳過機制、12小時快取、FinMind 兩次確認空資料防呆機制 (`no_finmind_data.json`)，以及失敗略過機制 (`failed_dates.json`)。
- **`main.py`**: **歷史資料爬蟲入口腳本**。負責呼叫 `scraper.py` 執行全市場歷史股價與基本面資料的抓取。
- **`fetch_categories.py`**: 抓取並更新台股產業分類及 ETF 清單至 `stock_categories.json`。
- **`patch_finmind.py`**: **FinMind 針對性補抓工具**。當發現特定股票有財報缺漏時，可單獨強制補抓，無需重新執行整個市場的爬蟲。
- **`scripts/check_data.py`**: **資料完整性修復工具**。自動掃描並刪除空檔或缺少核心欄位的異常檔案，讓爬蟲下次自動回補。

### 特徵工程 (Feature Engineering)
- **`scripts/feature_engineering.py`**: **核心特徵生成模組**。
  - 計算個股技術指標：MA、KD、RSI、MACD、布林通道、ATR。
  - **總體與板塊情緒（方案 B）**：每日計算全市場平均報酬與上漲比例，並結合滾動均線；自動載入產業映射，每日計算板塊平均報酬與 rolling 情緒。
  - **混合型預測標籤（方案 C）**：預測未來 1~3 天收益率，強勢標籤（2）強制鎖定為絕對上漲股，最終輸出 `features_combined.parquet`。
  - **效能優化**：Step 1 自動偵測當天是否有收盤價日報 CSV，**自動剔除國定假日與不開市交易日**，大幅減少 joblib 平行處理無效日期的開銷。
- **`run_feature_engineering.py`**: 獨立執行特徵工程的測試入口腳本。

### 模型最佳化、訓練與預測 (Modeling & Inference)
- **`optimize_factors.py`**: **特徵與超參數最佳化腳本**。使用 `Optuna` 框架，在不偷看未來的驗證集上，動態尋找勝率最高的技術指標參數組合並輸出 `best_factors.json`。
- **`train.py`**: **LightGBM 模型訓練腳本**。讀取 parquet 特徵檔，依照時間軸進行 strict 日期分位切割，訓練 LightGBM 模型並將特徵欄位儲存至 `models/feature_cols.json`。
- **`inference.py`**: **預測推論腳本**。載入最新一天資料與模型，推論目標清單股票未來 3 天的多空分數並印出排名表。
  - **實戰倉位對齊**：自動與 `trading_sim.py` 的「最多 5 檔持倉上限、Day1 >= 10.0% 買入、Day3 < 0% 賣出、-8% 停損」策略完全對齊。
  - **智慧下單建議**：自動區分 `Stocks.txt` 內實質持倉（有買入成本）與自選股（無成本），動態計算明日可用空位，並在每日收盤後提供明日「開盤買進/賣出」的雲端智慧單具體掛單建議。

### 實戰回測與模擬交易 (Simulation)
- **`trading_sim.py`**: **實戰級交易模擬回測器 (Out-of-Sample)**。
  - 模擬真實交易：支持自訂時間區間、初始資金、買入多空信心分數門檻（預設 `>= 10%` 信心）、個股固定停損（如 `-8%`）或轉弱出場。
  - 資金與資產一致性：日終統一將同一天發生的所有交易明細之 `Current_Cash`、`Stock_Value` 與 `Total_Equity` 更新為當天收盤後的最終狀態。
  - 輸出與降級：支援 Excel 多分頁高規格匯出（含歷史淨值與交易明細分頁），並內建 CSV 穩健降級機制。
- **`backtest.py`**: **時光機回測模式腳本**。指定單一基準日，以該日前的數據訓練模型，回測該日未來 3 天全市場及自選股的實質利潤與勝率。

### ️ 共享核心工具與測試 (Utilities & Tests)
- **`utils.py`**: **全系統共享股票解析工具**。統一 `Stocks.txt` 的解析邏輯（格式 A 及格式 B 成本欄位），提供單一事實來源並自動支援 subdirectory 與 fallback。
- **`tests/test_pipeline.py`**: **全系統整合測試腳本**。驗證 19 項核心邏輯，包含日期切割、特徵檔完整性（排除 Level Bias 絕對值特徵，包含方案 B 特徵）、`utils.py` 解析器單元測試等。
- **`tests/test_scraper.py`**: **爬蟲單元測試腳本**。快速測試證交所與期交所 API 下載功能，自動整合 `skip_dates.json` 防呆略過機制。
- **`tests/test_finmind.py`**: **FinMind 資料單元測試**。自動以 `__file__` 動態解析路徑，檢驗基本面資料與歷年修改時間。

---

## 3. 資料目錄結構與核心快取檔案 (Data & Cache)

### `data/` 目錄結構
- **`raw_price/`**: 每日收盤行情與成交量 (`*_price.csv`，來自 TWSE `MI_INDEX`)。核心開市判斷基準。
- **`raw_chips/`**: 籌碼與交易面資料。三大法人買賣超 (`*_chips.csv`)、當沖統計 (`*_daytrading.csv`)、外資持股比例 (`*_fini_holding.csv`)。
- **`raw_margin/`**: 信用交易資料。融資融券餘額 (`*_margin.csv`)、借券餘額 (`*_sbl.csv`)、額度總量管制 (`*_credit_limit.csv`)。
- **`raw_twse_per/`**: 個股官方本益比、殖利率及淨值比 (`*_twse_per.csv`)。
- **`raw_taifex/`**: 期貨大盤外資多空未平倉淨額 (`*_taifex_inst.csv`)。
- **`raw_shareholding/`**: 每週集保戶股權分散表 (`*_shareholding.csv`)。
- **`raw_financial/`**: FinMind 基本面財報。營收、三表季報與歷年股利。

### 核心設定與快取檔案 (.json)
1. **`best_factors.json`**: 記錄 Optuna 找出的最佳技術指標參數，由 `feature_engineering.py` 自動讀取。
2. **`stock_categories.json`**: 產業分類與 ETF 清單，為特徵工程的板塊情緒、爬蟲跳過 ETF 任務及交易模擬器股票名稱轉換的共用依據。
3. **`models/feature_cols.json`**: 記錄模型訓練當下的所有特徵名稱，確保推論特徵順序與數量 100% 一致。
4. **`data/failed_dates.json`**: TWSE/TAIFEX 的失敗計數器。失敗超過 3 次則判定為「無開市/假補班」並永久略過。
5. **`data/no_finmind_data.json`**: FinMind 股票空值快取（快取 90 天）。
6. **`data/skip_dates.json`**: 官方爬蟲跳過快取，記錄無資料或格式異常的日期，並註記 `reason`（如 `no_data`, `unexpected_format`）以供除錯。
7. **`data/missing_fm_datasets.json`**: FinMind 局部財報缺漏快取。

---

## 4. 未來擴展方向與 AI 注意事項
1. **資料維護**: 若發現特徵檔中缺少某項資料，請優先檢查 `scraper.py` 的跳過機制，並利用 `check_data.py` 將損毀的 CSV 清除。
2. **參數動態命名**: 在新增技術指標時，請務必與 `optimize_factors.py` 聯動，確保 `feature_engineering.py` 所產生的特徵欄位名稱是動態且可被模型讀取的。
3. **GPU Sentiment (規劃中)**: 未來可引入本地 Llama 模型，分析新聞情緒並將其轉化為情緒因子併入特徵矩陣中。