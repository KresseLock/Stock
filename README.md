# 🇹🇼 台灣股市量化交易系統
### Taiwan Stock Quantitative Trading System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-ML_Engine-2ecc71?style=for-the-badge&logo=leaflet&logoColor=white"/>
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter_Search-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  一條龍全自動化台股量化預測系統 · LightGBM 多天期分類預測 · Optuna 貝葉斯調參 · Out-of-Sample 實戰回測
</p>

---

## 🎯 系統核心特色

> 傳統相對選股模型在「市場崩盤日」依然會滿倉買入（在全大跌日買入跌最少的股票），導致資產隨大盤沉淪。本系統透過三層防禦機制徹底解決這個問題。

### 三層絕對收益防禦架構

```
第一層：宏觀感知         第二層：混合型標籤        第三層：智慧空倉
────────────────         ──────────────────        ────────────────
注入全市場日報酬          強勢股需同時滿足：         Day1 分數全面
均值、市場寬度            · 相對排名前 20%           下滑時，系統
（上漲比例）與            · 絕對報酬率 > 0%          自動判定無股
板塊趨勢強度              ，崩盤日不產生             可買，100% 空
                          買入標籤                   倉持現金避險
```

### 多源數據融合

| 數據來源 | 內容 |
|---------|------|
| 📊 **TWSE 證交所** | 收盤行情、三大法人、融資融券、本益比、殖利率 |
| 📈 **TAIFEX 期交所** | 台指期外資未平倉淨額 |
| 🏦 **TDCC 集保所** | 每週大戶持股分級 |
| 📋 **FinMind** | 月營收、季報 EPS/ROE/毛利率、現金流量 |

---

## 🔄 系統工作流程

```
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
│  數據爬蟲     │───▶│  貝葉斯因子調參 │───▶│   特徵工程重建    │
│ scraper.py   │     │optimize_factors│     │feature_engineer  │
└──────────────┘     └────────────────┘     └──────────────────┘
                           ↓ best_factors.json      ↓ .parquet
                    ┌──────────────┐     ┌──────────────────┐
                    │  推理預測     │◀───│   模型訓練        │
                    │ inference.py │     │    train.py      │
                    └──────────────┘     └──────────────────┘
                           ↓ 多空分數
                    ┌──────────────┐
                    │  交易模擬回測 │
                    │trading_sim.py│
                    └──────────────┘
```

---

## ✨ 功能模組詳解

<details>
<summary><b>🕷️ 多源容錯爬蟲 (scraper.py)</b></summary>

- 免 Token 抓取證交所收盤行情、三大法人買賣超、融資融券、借券餘額、當沖比例、外資持股、本益比、殖利率
- 期交所台指期外資未平倉、集保所每週大戶持股
- 整合 FinMind 月營收與三大財務報表（支援 `_FINMIND_CACHE_DAYS` 自訂天數快取，預設 7 天，可自由調校）
- 具備失敗計數略過（`failed_dates.json`）與空值快取機制，節省 Token 與網路開銷

</details>

<details>
<summary><b>⚙️ 精密特徵工程 (feature_engineering.py)</b></summary>

- **技術指標**：Moving Average、KD、RSI、MACD、布林通道、ATR、成交量比（參數均可由 Optuna 最佳化）
- **法人籌碼**：外資/投信/自營買賣超、連續買超天數、滾動累積籌碼
- **市場感知**：全市場日報酬均值、市場寬度（上漲比例）5日/20日滾動趨勢
- **板塊強度**：依 `stock_categories.json` 計算各產業每日平均報酬與滾動強度
- **自動消除 Level Bias**：絕對金額轉為比例/變化率，避免模型記住個股身份

</details>

<details>
<summary><b>🔬 貝葉斯超參數最佳化 (optimize_factors.py)</b></summary>

- Optuna TPE 框架自動搜尋技術指標最佳參數組合
- 嚴格日期分界防止前視偏差（Lookahead Bias）
- 支援 Early Stopping（連續 N 輪無進展自動終止）
- 結果存至 `best_factors.json`，供後續流水線復用

</details>

<details>
<summary><b>🤖 智能推論預測 (inference.py)</b></summary>

- 讀取最新一天資料，載入 LightGBM 推論未來 1~3 天多空分數
- 自動對齊持倉清單，計算即時浮動損益（需在 `Stocks.txt` 填入成本）
- 依交易模擬器策略參數自動輸出明日建議買進 / 賣出掛單
- **智慧限價掛單**：自動依預測信心強度動態推薦溢價幅度（+1.5% ~ +2.5%），並自動對齊台灣股市報價升降單位 (Tick Size) 四捨五入，直接輸出開盤建議掛單限價，省去人工計算
- 自動排除 ETF，優先顯示「可立即行動」的高分標的

</details>

<details>
<summary><b>📊 實戰交易模擬器 (trading_sim.py)</b></summary>

- 模擬真實手續費（0.1425%）與證交稅（0.3%）
- 個股固定停損（-8%）+ 信號轉弱出場雙重保護
- 剩餘現金動態配倉（不固定每檔金額，依剩餘槽位均分）
- 支援零股交易精算，回測結束輸出多分頁 Excel 報表

</details>

---

## 📁 目錄結構

```
Stock/
├── 📂 data/                    # 原始 CSV、快取 JSON、特徵矩陣 Parquet
├── 📂 models/                  # LightGBM 模型檔 + feature_cols.json
├── 📂 reports/                 # 回測輸出的 Excel / CSV 績效報表
├── 📂 scripts/
│   ├── check_data.py           # 資料完整性修復工具
│   ├── feature_engineering.py  # 核心特徵與標籤提取模組
│   └── scraper.py              # 多源容錯資料爬蟲
├── 📂 tests/
│   ├── test_finmind.py         # FinMind 財報單元測試
│   ├── test_pipeline.py        # 全流程整合測試（19 項）
│   └── test_scraper.py         # 證交所 API 單元測試
├── 📂 predictions/             # 每日推理結果存檔
├── auto_pipeline.py            # ⭐ 一鍵式自動化流水線
├── backtest.py                 # 時光機單日回測器
├── inference.py                # 多空分數排行榜 + 下單指引
├── optimize_factors.py         # Optuna 參數最佳化器
├── trading_sim.py              # 實戰量化模擬交易器
├── train.py                    # LightGBM 模型訓練器
├── utils.py                    # 共享股票解析工具
├── Stocks.txt                  # ⭐ 自選股 / 持倉清單
├── best_factors.json           # 最佳化因子參數存檔
└── requirements.txt            # 依賴套件清單
```

---

## 🚀 快速開始

### Step 1 — 建立環境

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2 — 設定自選股清單

編輯根目錄的 `Stocks.txt`，支援三種格式：

```text
# 格式 A：僅追蹤（不計算損益）
2330

# 格式 B：有買進成本（顯示損益%）
2317,161.0

# 格式 C：完整持倉（顯示損益金額）
2454,680.5,1000
```

> 若要抓取基本面財務報表，請在根目錄建立 `FINMIND_TOKEN.txt` 並貼入 FinMind API Token（支援免費免登入額度）。

### Step 3 — 一鍵啟動全自動流水線

```powershell
# 執行順序：Optuna 調參 → 套用參數 → 重建特徵 → 訓練模型 → 推理預測
python auto_pipeline.py
```

### Step 4 — 執行策略回測

```powershell
# 回測 2026 年上半年，初始資金 100 萬，最大持股 5 檔
python trading_sim.py --start 2026-01-02 --end 2026-06-25 --capital 1000000 --max_pos 5

# 回測 2025 全年，初始資金 50 萬
python trading_sim.py --start 2025-01-02 --end 2025-12-30 --capital 500000 --max_pos 5
```

回測結束後，報表自動輸出至 `reports/` 資料夾（Excel 多分頁格式）。

### Step 5 — 執行系統測試

```powershell
# 確認 19 項整合指標全數通過
python tests/test_pipeline.py
```

---

## ⚙️ 核心策略參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `BUY_THRESHOLD` | `10.0%` | Day1 多空淨分數達此值才觸發買進 |
| `SELL_THRESHOLD` | `0.0%` | Day3 多空淨分數低於此值觸發賣出 |
| `STOP_LOSS_PCT` | `-8.0%` | 相對買進成本的固定停損線 |
| `MAX_POSITIONS` | `5 檔` | 最大同時持股數 |
| `FEE_RATE` | `0.1425%` | 單邊券商手續費 |
| `TAX_RATE` | `0.3%` | 賣出證交稅（非當沖） |
| **總進出成本** | **≈ 0.585%** | 模型選股獲利需超越此值才有淨利 |

---

## 📈 回測架構說明

```
訓練資料邊界                     可回測區間
      │                              │
2020-01-01 ──────────── 2025-08-01 ────────────── 今日
      │<── 模型訓練資料 ──>│<──── Out-of-Sample ────>│
                          ↑                         ↑
                    BACKTEST_DATE             特徵庫每日更新
                  （訓練/驗證分界）         （END_DATE = today）
```

> **Out-of-Sample 保證**：模型訓練截止於 `20250801`，回測若設定在此日期之後，所有推論均屬完全樣本外，不存在未來函數（Lookahead Bias）問題。

---

## 🛡️ 開發規範

- **新增特徵**：修改 `scripts/feature_engineering.py` 後，必須重新執行 `auto_pipeline.py` 重建特徵矩陣與重新訓練
- **修改策略參數**：`trading_sim.py` 與 `inference.py` 頂端的常數區塊需同步修改，確保回測與實盤邏輯一致
- **Stocks.txt 格式**：所有解析邏輯統一走 `utils.py`，勿在各模組重複實作
- **提交前**：執行 `python tests/test_pipeline.py` 確認全數通過

---

## 📄 授權

本專案採用 [MIT License](https://opensource.org/licenses/MIT) 授權。

**免責聲明**：本系統所有預測結果與回測報告僅供量化研究參考，**不構成任何實際投資建議**。投資涉及風險，請自行審慎評估。
