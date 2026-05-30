# 專案 Agent 職責說明

## 1. CPU_Cleaner (數據清洗與因子生成)
- **職責**: 負責 ETL 流程，整合股價、籌碼、財報。
- **輸入**: 原始 CSV/API 資料。
- **輸出**: `features_combined.parquet`。

## 2. GPU_Sentiment (新聞情緒分析)
- **職責**: 執行本地 Llama 3，分析新聞情緒。
- **設備**: 使用 3060Ti (CUDA)。
- **輸出**: `sentiment_scores.csv`。

## 3. Core_Predictor (LightGBM 綜合預測)
- **職責**: 融合因子，訓練 LightGBM 模型。
- **任務**: 執行 `train.py` 與 `inference.py`。
- **設備**: CPU 密集 (9700X)。

---

## 4. 專案 Python 程式碼架構與功能對照表
為方便後續 AI 代理快速理解系統架構，以下是本專案核心 Python 腳本的功能地圖：

### 🔄 自動化與主控端 (Main Controllers)
- **`auto_pipeline.py`**: **系統全自動主流程腳本**。負責依照設定的參數，一條龍串接：啟動爬蟲 -> 超參數最佳化 -> 特徵工程 -> 模型訓練 -> 最終預測。
- **`main.py`**: **歷史資料爬蟲入口腳本**。負責呼叫 `scraper.py` 執行台股全市場歷史股價、籌碼、財報等資料的抓取任務。

### 🕷️ 資料獲取與清洗 (Data Scraping & ETL)
- **`scripts/scraper.py`**: **核心爬蟲模組**。負責抓取 TWSE (證交所)、TAIFEX (期交所)、FinMind 財報與集保中心大戶持股。內建 12 小時快取與嚴格的失敗跳過與二次確認機制。
- **`fetch_categories.py`**: 抓取並更新台股產業分類及 ETF 清單 (更新 `stock_categories.json`)。
- **`scripts/check_data.py`**: **資料完整性修復工具**。快速掃描已下載的 FinMind 財報 CSV，自動刪除空檔或缺失欄位的異常檔案，以利系統自動回補。

### 🧬 特徵工程 (Feature Engineering)
- **`scripts/feature_engineering.py`**: **核心特徵生成模組**。計算技術指標 (MA, KD, RSI, MACD 等)、組合法人籌碼與財報資料，產生預測目標標籤，最終輸出供模型訓練的 `.parquet` 特徵檔。
- **`run_feature_engineering.py`**: 獨立執行特徵工程的腳本 (功能多半已被 `auto_pipeline.py` 整合，可用作單獨執行或測試)。

### 🤖 模型訓練與預測 (Modeling & Inference)
- **`optimize_factors.py`**: **特徵與超參數最佳化腳本**。使用 Optuna 框架，動態尋找勝率最高的技術指標參數組合 (如最佳的均線天數、KD週期等) 並輸出 `best_factors.json`。
- **`train.py`**: **模型訓練腳本**。讀取 parquet 特徵檔，切割訓練與驗證集，訓練 LightGBM 模型，並將模型儲存至 `models/`。
- **`inference.py`**: **預測推論腳本**。載入最新一天的特徵與訓練好的 LightGBM 模型，推論目標清單股票未來 3 天的漲跌機率。

### 🧪 測試與驗證 (Testing)
- **`test_pipeline.py`**: **全系統整合測試腳本**。檢查模組 import、檢查參數動態命名、檢查特徵檔 (`parquet`) 的欄位完整性，以及測試略過機制，確保整個量化系統正常運作。