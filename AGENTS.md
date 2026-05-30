# 📈 台灣股市量化交易系統 - 架構與 AI 導航指南 (AGENTS.md)

本文件專為 AI Agent 設計。當您進入本專案時，請先閱讀此文件以快速掌握系統全貌、資料流與核心模組，避免重新分析程式碼而浪費 Token。

---

## 1. 系統設計理念與工作流 (Workflow)
這是一個**一條龍全自動化的台股量化預測系統**，設計成能在 T+1 / T+2 的藍圖下，根據每日收盤後的資料，自動化產出未來的交易預測。

系統具備高度的**容錯性與容錯快取**，並透過 `Optuna` 尋找最佳技術指標參數，最後交由 `LightGBM` 進行訓練與推論。
核心流水線 (Pipeline) 流程如下：
`多源爬蟲 (Scraping)` ➔ `超參數最佳化 (Optuna)` ➔ `特徵工程 (ETL)` ➔ `模型訓練 (LightGBM)` ➔ `推論預測 (Inference)`

---

## 2. 核心模組與 Python 腳本對照表

### 🕷️ 資料獲取與清洗 (Data Scraping & ETL)
- **`scripts/scraper.py`**: **核心爬蟲模組**。
  - **免費資料**: TWSE (股價、籌碼、資券、借券、本益比、當沖、外資持股、信用限額)、TAIFEX (外資台指期未平倉)、TDCC (每週大戶持股)。
  - **FinMind (需 Token)**: 月營收、三大報表、股利。
  - **特色**: 具備嚴格的 9 檔案歷史跳過機制、12小時快取、FinMind 兩次確認空資料防呆機制 (`no_finmind_data.json`)，以及失敗略過機制 (`failed_dates.json`)。
- **`main.py`**: **歷史資料爬蟲入口腳本**。負責呼叫 `scraper.py` 執行全市場歷史股價與基本面資料的抓取。
- **`fetch_categories.py`**: 抓取並更新台股產業分類及 ETF 清單至 `stock_categories.json`。
- **`scripts/check_data.py`**: **資料完整性修復工具**。掃描已下載的 FinMind 財報 CSV，自動刪除空檔或缺少核心欄位 (`date`, `revenue`, `type`, `value` 等) 的異常檔案，讓爬蟲下次自動回補。

### 🧬 特徵工程 (Feature Engineering)
- **`scripts/feature_engineering.py`**: **核心特徵生成模組**。
  - 計算技術指標：MA、KD、RSI、MACD、布林通道、ATR。
  - 融合大盤情緒 (期交所未平倉)、籌碼 (外資/投信/自營商)、財報 (營收/EPS/股利) 與持股分級。
  - 產生預測標籤 (未來 1~3 天漲跌幅)，最終輸出供模型訓練的 `features_combined.parquet`。
- **`run_feature_engineering.py`**: 獨立執行特徵工程的測試入口腳本。

### 🤖 模型最佳化、訓練與預測 (Modeling & Inference)
- **`optimize_factors.py`**: **特徵與超參數最佳化腳本**。使用 `Optuna` 框架，在不偷看未來的驗證集上，動態尋找勝率最高的技術指標參數組合 (如最佳均線天數、KD週期等) 並輸出 `best_factors.json` 供後續流程套用。
- **`train.py`**: **LightGBM 模型訓練腳本**。讀取 parquet 特徵檔，依照時間軸切割訓練集與驗證集，訓練 LightGBM 模型並將模型檔儲存至 `models/`。
- **`inference.py`**: **預測推論腳本**。載入最新一天的資料與訓練好的模型，推論目標清單股票未來 3 天的漲跌機率與走勢分析，並在終端機印出易讀的排名表 (包含股票代號與中文名稱)。

### 🔄 自動化與測試 (Orchestration & Testing)
- **`auto_pipeline.py`**: **系統全自動主流程腳本**。將上述所有步驟一條龍串接，只要執行這支程式，系統就會自動執行：補抓資料 -> Optuna找參數 -> 算特徵 -> 訓練 -> 預測。支援指定 `TRAIN_INDUSTRIES` (依產業篩選)。
- **`test_pipeline.py`**: **全系統整合測試腳本**。檢查各模組的 import、技術指標參數動態命名邏輯、特徵檔的欄位完整性 (檢查 7 大財務與籌碼特徵)，以及確保 `scraper.py` 具備正確的防呆機制。

---

## 3. 未來擴展方向與 AI 注意事項
1. **資料維護**: 若發現特徵檔中缺少某項資料，請優先檢查 `scraper.py` 的跳過機制，並利用 `check_data.py` 將損毀的 CSV 清除。
2. **參數動態命名**: 在新增技術指標時，請務必與 `optimize_factors.py` 聯動，確保 `feature_engineering.py` 所產生的特徵欄位名稱 (如 `k20`, `rsi18`) 是動態且可被模型讀取的。
3. **GPU Sentiment (規劃中)**: 未來可引入本地 Llama 模型，分析新聞情緒並將其轉化為情緒因子 (Sentiment Scores) 併入 `features_combined.parquet` 中。