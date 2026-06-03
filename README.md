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
┌──────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  數據爬蟲     │───▶│   貝葉斯因子調參    │───▶│   特徵工程重建    │
│  scraper.py  │     │ optimize_factors.py │     │feature_engineering│
└──────────────┘     └─────────────────────┘     └───────────────────┘
                            ↓ best_factors.json           ↓ .parquet
                      ┌──────────────────┐       ┌───────────────────┐
                      │     推理預測     │ ◀───  │     模型訓練      │
                      │   inference.py   │       │     train.py      │
                      └──────────────────┘       └───────────────────┘
                            ↓ 多空分數
                      ┌──────────────────┐
                      │   交易模擬回測   │
                      │  trading_sim.py  │
                      └──────────────────┘
```

---

## ✨ 功能模組詳解

<details>
<summary><b>🕷️ 多源容錯爬蟲 (scripts/scraper.py)</b></summary>

- 整合原 `main.py`、`patch_finmind.py`、`fetch_categories.py`、`check_data.py` 入口
- 免 Token 抓取證交所收盤行情、三大法人買賣超、融資融券、借券餘額、當沖比例、外資持股、本益比、殖利率
- 期交所台指期外資未平倉、集保所每週大戶持股
- 整合 FinMind 月營收與三大財務報表（支援 `FINMIND_CACHE_DAYS` 自訂天數快取，預設 15 天，可由 `config.py` 自由調校）
- 具備失敗計數略過（`failed_dates.json`）與空值快取機制，節省 Token 與網絡開銷
- **限額自動跳過**：支援環境變數 `SKIP_ON_FINMIND_LIMIT = "1"`，遭遇 API 429/402 限制時拋出 `FinMindLimitExceeded` 並由主控 `Auto_RUN.py` 自動跳過爬蟲，不卡死流程。

</details>

<details>
<summary><b>⚙️ 精密特徵工程 (scripts/feature_engineering.py)</b></summary>

- **技術指標**：Moving Average、KD、RSI、MACD、布林通道、ATR、成交量比（參數均可由 Optuna 最佳化）
- **法人籌碼**：外資/投信/自營買賣超、連續買超天數、滾動累積籌碼
- **市場感知**：全市場日報酬均值、市場寬度（上漲比例）5日/20日滾動趨勢
- **板塊強度**：依 `scripts/stock_categories.json` 計算各產業每日平均報酬與滾動強度
- **自動消除 Level Bias**：絕對金額轉為比例/變化率，避免模型記住個股身份

</details>

<details>
<summary><b>🔬 貝葉斯超參數最佳化 (scripts/optimize_factors.py)</b></summary>

- Optuna TPE 框架自動搜尋技術指標最佳參數組合
- 嚴格日期分界防止前視偏差（Lookahead Bias）
- 支援 Early Stopping（連續 N 輪無進展自動終止）
- 結果存至 `best_factors.json`，供後續流水線復用

</details>

<details>
<summary><b>🤖 智能推論預測 (scripts/inference.py)</b></summary>

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
- **真實 T+2 交割制度模擬**：細分「購買力（可用資金）」與「銀行實質餘額（T+2 扣/入款）」，賣出股票當天資金可立即滾動買入，但實質款項於兩日後才完成交割。
- 支援零股交易精算，回測結束輸出多分頁 Excel 報表

</details>

---

## 📁 目錄結構

```text
Stock/
├── 📂 data/                    # 原始 CSV、快取 JSON、特徵 Parquet
├── 📂 models/                  # LightGBM 訓練模型檔 + feature_cols.json
├── 📂 reports/                 # 回測輸出的 Excel / CSV 績效報表
├── 📂 predictions/             # 每日推理結果存檔 (.txt)
├── 📂 scripts/                 # 重構後所有邏輯與腳本的放置處
│   ├── 📂 tools/
│   │   └── clean_stocks.py     # 自選股清單 (Stocks.txt) 驗證與重複清理工具
│   ├── scraper.py              # 多源容錯資料爬蟲 (整合下載、補件、檢驗、產業分類)
│   ├── feature_engineering.py  # 核心特徵與標籤提取模組
│   ├── train.py                # LightGBM 多天期分類模型訓練器
│   ├── inference.py            # 多空分數預測排行榜 + 智慧限價掛單指引
│   ├── optimize_factors.py     # Optuna 貝葉斯超參數最佳化器
│   ├── backtest.py             # 時光機單日回測器 (樣本外評估)
│   ├── utils.py                # 全系統共享股票解析工具
│   ├── stock_categories.json   # 產業分類與 ETF 全系統共享對照表
│   └── StockSync.py            # 雲端自動備份上傳工具 (調用 rclone)
├── 📂 tests/
│   ├── test_finmind.py         # FinMind 財報單元測試
│   ├── test_pipeline.py        # 全流程整合測試 (18 項 100% PASS)
│   └── test_scraper.py         # 證交所 API 單元測試
├── .agyignore                  # AI 開發排除過濾設定 (防干擾)
├── .gitignore                  # Git 版本控制忽略設定
├── AGENTS.md                   # AI Agent 架構與導航指南 (開發必讀)
├── Auto_RUN.py                 # 一鍵順序執行全流程主控腳本 (支援 CLI `-s/--step` 簡碼路由)
├── auto_pipeline.py            # 一鍵式自動化流水線入口 (支援 CLI `-s/--step` 簡碼路由)
├── config.py                   # 系統中央控制面板 (所有策略、爬蟲、調參參數皆在此設定)
├── trading_sim.py              # 實戰級量化模擬交易器 (回測引擎)
├── Stocks.txt                  # 自選股 / 實質持倉清單
├── best_factors.json           # 最佳化技術指標參數存檔
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

### Step 3 — 系統配置設定 (config.py)

專案已將所有分散於各檔案的設定參數，統一集中在根目錄的 [config.py](file:///D:/VScode_Stock/Stock/config.py) 中。在開始之前，您可以隨時打開此檔案修改：
* `TRAIN_INDUSTRIES`：要加入訓練與交易的產業板塊 (自選股 `Stocks.txt` 必定保留)。
* `BUY_THRESHOLD` / `SELL_THRESHOLD` / `STOP_LOSS_PCT`：買賣策略分數與停損門檻。
* `START_DATE` / `FINMIND_FETCH_MODE`：爬蟲下載的時間起點與過濾模式。
* `RUN_OPTIMIZATION` / `FEAT_N_JOBS` / `TRAIN_N_JOBS`：最佳化開關與多核心平行核心數設定。

### Step 4 — 一鍵自動化執行與步驟路由

#### 方案甲：一鍵主控執行（生產流程）
執行 `Auto_RUN.py` 會自動進行一鍵生產流程：增量下載今日最新行情 ➔ 重建特徵工程與訓練模型 ➔ 產出推理排名與掛單建議 ➔ 雲端硬碟同步備份。

> [!NOTE]
> 參數選項 (`--step` / `-s`) 與步驟值 (完整拼寫 / 簡寫首字) 可以任意混合搭配使用。例如：`--step download`, `-s download`, `--step d`, `-s d` 全數等價。

```powershell
# 1. 執行全套生產流程
python Auto_RUN.py

# 2. 僅執行特定生產步驟
python Auto_RUN.py -s d             # 僅增量下載今日最新資料 (同 --step download)
python Auto_RUN.py -s p             # 僅重新建立特徵、訓練與模型推理 (同 --step predict)
python Auto_RUN.py -s b             # 僅執行雲端備份 (同 --step backup)
```

#### 方案乙：一鍵流水線執行（研發與最佳化流程）
執行 `auto_pipeline.py` 會自動執行機器學習的研發流程：參數載入 ➔ 特徵矩陣重建 ➔ 模型訓練 ➔ 模型推理。

> [!NOTE]
> 參數選項 (`--step` / `-s`) 與步驟值 (完整拼寫 / 簡寫首字) 可以任意混合搭配使用。例如：`--step optimize`, `-s optimize`, `--step o`, `-s o` 全數等價。

```powershell
# 1. 執行完整研發流水線 (是否跑 Optuna 調參取決於 config.py 中的 RUN_OPTIMIZATION 設定)
python auto_pipeline.py

# 2. 僅執行特定研發步驟
python auto_pipeline.py -s o        # 單獨啟動 Optuna 調參 (同 --step optimize，會無視 config.py 中的限制強行執行)
python auto_pipeline.py -s f        # 單獨重建特徵矩陣 (同 --step feature)
python auto_pipeline.py -s t        # 單獨訓練 LightGBM 模型 (同 --step train)
python auto_pipeline.py -s i        # 單獨輸出今日推理結果 (同 --step inference)
```

#### 方案丙：資料庫維護中心 (scripts/scraper.py)
所有資料抓取、特定補件、完整性維護功能，統一整合在 `scraper.py` 的 CLI 參數中。

> [!NOTE]
> 所有功能與參數選項均支援「完整拼寫」或「簡短簡碼」自由搭配使用。例如：`--patch 2330`, `-p 2330` 或 `--fetch-categories`, `-fc` 全數等價。

```powershell
# 1. 正常增量下載今日最新資料
python scripts/scraper.py

# 2. 針對性補足特定股票 (例如 2330) 的歷史財報與基本面資料
python scripts/scraper.py -p 2330               # 針對性補件 (同 --patch 2330)

# 3. 重新下載並更新全市場產業分類與對照表 (scripts/stock_categories.json)
python scripts/scraper.py -fc                   # 更新分類表 (同 --fetch-categories)

# 4. 掃描並刪除損毀、格式異常或含極端價格的歷史日報資料
python scripts/scraper.py -c                    # 校驗與修復 (同 --check)
```

> [!WARNING]
> #### ☁️ rclone 雲端備份環境與安全性注意事項
> 
> 1. **執行檔需另行下載**：`rclone` 本身是用 Go 語言編寫的獨立工具，您需要前往 [rclone 官網](https://rclone.org/downloads/) 下載適用於 Windows 的 `rclone.exe` 執行檔，並將其放入虛擬環境的 `venv/Scripts/` 目錄中，或者加入系統的環境變數 PATH 中。
> 2. **Token 與登入金鑰安全**：透過 `rclone config` 登入雲端硬碟後所產生的金鑰資訊會儲存於 `config` / `.config` 資料夾或 `rclone.conf` 設定檔中。**這些金鑰與權限 Token 屬於高度敏感私鑰，已自動列入 `.gitignore` 與 `.agyignore` 中，絕對禁止提交至 Git 倉庫，亦不可外洩或上傳雲端**。

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
# 確認 18 項整合指標全數通過
python tests/test_pipeline.py
```

---

## ⚙️ 核心策略參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `BUY_THRESHOLD` | `10.0%` | Day1 多空淨分數達此值才觸發買進 (在 config.py 中設定) |
| `SELL_THRESHOLD` | `0.0%` | Day3 多空淨分數低於此值觸發賣出 (在 config.py 中設定) |
| `STOP_LOSS_PCT` | `-8.0%` | 相對買進成本的固定停損線 (在 config.py 中設定) |
| `MAX_POSITIONS` | `5 檔` | 最大同時持股數 (在 config.py 中設定) |
| `FEE_RATE` | `0.1425%` | 單邊券商手續費 (在 config.py 中設定) |
| `TAX_RATE` | `0.3%` | 賣出證交稅（非當沖，在 config.py 中設定） |
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

- **中央設定檔 (config.py)**：所有常數變數與機器學習核心配置均統一在 [config.py](file:///D:/VScode_Stock/Stock/config.py) 修改，禁止在單一模組腳本中硬編碼。
- **新增特徵**：修改 `scripts/feature_engineering.py` 後，必須重新執行 `auto_pipeline.py -s f` (或 `--step feature`) 重建特徵工程 parquet 檔並重新訓練。
- **編輯自選股清單**：請直接修改 `Stocks.txt`。可以使用輔助工具 `python scripts/tools/clean_stocks.py` 自動清理重複或無效的股票代碼。
- **提交前**：務必執行 `python tests/test_pipeline.py` 確認全數通過。

---

## 📄 授權
**免責聲明**：本系統所有預測結果與回測報告僅供量化研究參考，**不構成任何實際投資建議**。投資涉及風險，請自行審慎評估。
