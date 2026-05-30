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