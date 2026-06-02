# 🇹🇼 台灣股市量化交易系統
### Taiwan Stock Quantitative Trading System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-ML_Engine-2ecc71?style=for-the-badge&logo=leaflet&logoColor=white"/>
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter_Search-FF6F00?style=for-the-badge"/>
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
                     ┌──────────────┐        ┌──────────────────┐
                     │  推理預測     │ ◀───  │   模型訓練        │
                     │ inference.py │        │    train.py      │
                     └──────────────┘        └──────────────────┘
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

```text
Stock/
├── 📂 data/                    # 原始 CSV、快取 JSON、特徵 Parquet
├── 📂 models/                  # LightGBM 訓練模型檔 + feature_cols.json
├── 📂 reports/                 # 回測輸出的 Excel / CSV 績效報表
├── 📂 scripts/
│   ├── check_data.py           # 資料完整性修復工具
│   ├── feature_engineering.py  # 核心特徵與標籤提取模組
│   └── scraper.py              # 多源容錯資料爬蟲模組
├── 📂 tests/
│   ├── test_finmind.py         # FinMind 財報單元測試
│   ├── test_pipeline.py        # 全流程整合測試（19 項 100% PASS）
│   └── test_scraper.py         # 證交所 API 單元測試
├── 📂 predictions/             # 每日推理結果存檔 (.txt)
├── .agyignore                  # AI 開發排除過濾設定 (防干擾)
├── .gitignore                  # Git 版本控制忽略設定
├── AGENTS.md                   # AI Agent 架構與導航指南 (開發必讀)
├── Auto_RUN.py                 # 一鍵順序執行全流程主控腳本 (完全解耦)
├── auto_pipeline.py            # 一鍵式自動化調參-訓練-推理流水線
├── backtest.py                 # 時光機單日回測器 (樣本外評估)
├── fetch_categories.py         # 產業分類與 ETF 下載工具
├── inference.py                # 多空分數預測排行榜 + 智慧限價掛單指引
├── main.py                     # 全市場資料下載與歷史庫初始化入口
├── optimize_factors.py         # Optuna 貝葉斯超參數最佳化器
├── patch_finmind.py            # FinMind 基本面個股缺漏強制補丁工具
├── run_feature_engineering.py  # 獨立特徵工程執行入口
├── StockSync.py                # 雲端自動備份上傳工具 (調用 rclone)
├── trading_sim.py              # 實戰級量化模擬交易器 (回測引擎)
├── train.py                    # LightGBM 多天期分類模型訓練器
├── utils.py                    # 全系統共享股票解析工具
├── Stocks.txt                  # 自選股 / 實質持倉清單
├── best_factors.json           # 最佳化技術指標參數存檔
├── stock_categories.json       # 產業分類與 ETF 全系統共享對照表
├── FINMIND_TOKEN.txt           # FinMind API 金鑰存放檔 (可選)
└── requirements.txt            # Python 依賴套件清單
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

### Step 3 — 初始化歷史資料庫

首次下載本專案或需要回補最新歷史數據時，請執行 `main.py` 開啟全自動爬蟲下載股價、法人籌碼與基本面資料庫：

```powershell
python main.py
```

> 💡 **爬蟲防呆與雙重快取機制**：
> * **官方全市場爬蟲**：採取「成功歷史檔案永久跳過（不重複下載）」與「開市日自動判定」，並具備失敗達 3 次則寫入 `failed_dates.json` 永久略過的避雷防呆機制。
> * **FinMind 基本面爬蟲**：採取「`_FINMIND_CACHE_DAYS` 自訂天數快取（預設 7 天）」與「確認無財報 / 局部缺漏 90 天快取」機制，極大化節省 API 比對的等待時間與 Token 額度。
> 首次抓取歷史資料（依您的股票檔數而定）因需要建立完整歷史庫可能需要較長時間，後續每日增量回補均可在數秒至數分鐘內迅速完成。

### Step 4 — 一鍵啟動全自動流水線與雲端同步

本系統支援兩種自動化執行方式：

#### 方案甲：一鍵主控執行 (完全解耦，最推薦)
我們將所有功能腳本進行了完全解耦，並由 `Auto_RUN.py` 主控執行。它會依序啟動並監控以下步驟：
1. 增量下載今日最新行情與法人籌碼數據 (`main.py`)
2. 重建特徵工程與模型訓練 (`auto_pipeline.py`)
3. 執行推理預測產出多空排行榜與建議下單指令 (`inference.py`)
4. 調用 rclone 將最新預測 txt 建議單同步上傳至雲端硬碟備份 (`StockSync.py`)

執行指令：
```powershell
python Auto_RUN.py
```

#### 方案乙：分步單獨執行 (模組化執行)
如果您只想執行特定的核心流程，可以直接單獨運行以下腳本：
```powershell
# 只執行：因子載入 → 特徵生成 → 模型訓練 → 推理預測 (不包含資料下載與雲端上傳)
python auto_pipeline.py

# 只執行：推理預測 (讀取現有特徵與模型，直接輸出今日排行榜與下單建議)
python inference.py

# 只執行：雲端備份 (將 predictions/ 底下的預測建議 txt 上傳至雲端)
python StockSync.py
```

> [!WARNING]
> #### ☁️ rclone 雲端備份環境與安全性注意事項
> 
> 1. **執行檔需另行下載**：`rclone` 本身是用 Go 語言編寫的獨立工具，`pip` 安裝的只是 Python 的封裝庫。您需要前往 [rclone 官網](https://rclone.org/downloads/) 下載適用於 Windows 的 `rclone.exe` 執行檔，並將其放入虛擬環境的 `venv/Scripts/` 目錄中，或者加入系統的環境變數 PATH 中。
> 2. **Token 與登入金鑰安全**：透過 `rclone config` 登入雲端硬碟後所產生的金鑰資訊會儲存於 `config` / `.config` 資料夾或 `rclone.conf` 設定檔中。**這些金鑰與權限 Token 屬於高度敏感私鑰，已自動列入 `.gitignore` 與 `.agyignore` 中，絕對禁止提交至 Git 倉庫，亦不可外洩或上傳雲端**。

> [!IMPORTANT]
> #### 💡 重新訓練與大數據調參流程 (更換追蹤產業與重跑優化時的正確步驟)
>
> 當您修改了 `auto_pipeline.py` 中的 `TRAIN_INDUSTRIES`（例如只保留科技股以縮小訓練範圍），或者半年後大盤環境變更想重跑因子優化時，因為流水線「先優化、後建檔」的特性，正確的「三步走」流程是：
>
> 1. **第一步（產生全市場特徵檔）**：
>    * 在 `auto_pipeline.py` 中，將所有產業都改成 `True`。
>    * 設定 `RUN_OPTIMIZATION = False`（不進行優化，先建檔）。
>    * 執行 `python auto_pipeline.py`。
>    *(此步驟目的在於在硬碟中產生包含「全市場股票」的特徵 Parquet 檔案供優化器讀取)*
> 
> 2. **第二步（執行大數據優化）**：
>    * 維持所有產業為 `True`。
>    * 設定 `RUN_OPTIMIZATION = True`。
>    * 執行 `python auto_pipeline.py`。
>    *(此步驟會讀取第一步建立的全市場特徵檔，進行貝葉斯優化，並把最佳解寫入 `best_factors.json` 中)*
> 
> 3. **第三步（縮小範圍並訓練）**：
>    * 將不想追蹤的產業改為 `False`（例如只留下科技股）。
>    * 設定 `RUN_OPTIMIZATION = False`（直接套用最佳化完成的結果，免重複優化）。
>    * 執行 `python auto_pipeline.py`。
>    *(此步驟會載入剛才優化好的最佳參數，並「只為科技股」重新計算特徵與訓練模型，既能保證參數在大樣本下的通用性，又不易在小樣本下產生過度擬合)*

### Step 5 — 執行策略回測

```powershell
# 回測 2026 年上半年，初始資金 100 萬，最大持股 5 檔
python trading_sim.py --start 2026-01-02 --end 2026-06-25 --capital 1000000 --max_pos 5

# 回測 2025 全年，初始資金 50 萬
python trading_sim.py --start 2025-01-02 --end 2025-12-30 --capital 500000 --max_pos 5
```

回測結束後，報表自動輸出至 `reports/` 資料夾（Excel 多分頁格式）。

### Step 6 — 執行系統整合測試

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
**免責聲明**：本系統所有預測結果與回測報告僅供量化研究參考，**不構成任何實際投資建議**。投資涉及風險，請自行審慎評估。
