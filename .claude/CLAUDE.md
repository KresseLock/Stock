# 台灣股市量化交易系統 - AI 導航指南 (CLAUDE.md)

## 1. 系統概覽
全自動台股量化預測回測系統。核心流程：
`scraper.py` → `optimize_factors.py` → `feature_engineering.py` → `train.py` → `inference.py` → `trading_sim.py`

唯一設定檔：[config.py](config.py)（所有常數、產業、門檻均在此，**嚴禁在腳本中分散寫死**）。

---

## 2. 核心腳本速查

| 腳本 | 功能摘要 |
|------|---------|
| `scripts/scraper.py` | 增量下載 TWSE/TAIFEX/FinMind。`-p <sid>` 補財報、`-fc` 更新產業表、`-c` 檢查清理 |
| `scripts/feature_engineering.py` | 技術指標＋板塊情緒（方案B）＋混合標籤（方案C），輸出 `features_combined.parquet` |
| `scripts/optimize_factors.py` | Optuna TPE 尋最佳技術指標參數 → `configs/best_factors.json` |
| `scripts/train.py` | 訓練 LightGBM，大跌樣本加權懲罰，輸出 `models/feature_cols.json` |
| `scripts/inference.py` | 推論未來3天多空分數，提供含 Tick Size 對齊的掛單建議 |
| `scripts/optimize_trading_params.py` | Optuna 最佳化風控參數 → `configs/best_trading_params.json` |
| `trading_sim.py` | OOS 回測，含 T+2 交割模擬、限價溢價搓合、Excel 輸出 |
| `Auto_RUN.py` | 生產主控：`-s download/predict/backup/all`（預設 all），`-d YYYYMMDD` 指定推理日；download 模式額度不足會等待而非跳過 |
| `auto_pipeline.py` | 研發流水線：`-s optimize/feature/train/inference` |
| `run_workflow_experiment.py` | 雙模式實驗（模式A研究期截至2025-08-01／模式B生產期），含 Checkpoint 斷點續傳 |
| `scripts/analyze_regime_stability.py` | OOS 訊號健康診斷：RankIC / Alpha / PSI 漂移 |
| `scripts/param_sensitivity.py` | 參數敏感度診斷（OFAT）：跨市況掃單一風控參數，解釋優化器為何選某值 → `reports/param_sensitivity_report.md` |
| `scripts/StockSync.py` | rclone 同步 `predictions/` 至 Google Drive |

---

## 3. 重要設定與快取檔案
- `configs/best_factors.json` — 最佳技術指標參數（`feature_engineering.py` 自動讀取）
- `configs/best_trading_params.json` — 最佳風控參數（`config.py` 啟動時自動覆寫常數）
- `configs/best_trading_params_mode_b.json` — **實盤推薦**，複製為 `best_trading_params.json` 使用
- `reports/workflow_experiment_results.json` — 實驗 Checkpoint 快取
- `scripts/stock_categories.json` — 產業分類與 ETF 清單（全系統共用）
- `models/feature_cols.json` — 訓練特徵名稱，推論時須 100% 對齊
- `data/skip_dates.json` / `data/failed_dates.json` — 爬蟲跳過與失敗快取
- `Stocks.txt` — 自選股／持倉清單；**格式 D 第 4 欄＝買入日鎖定 ATR 停損價**（`代號,成本,股數,停損價`，由 `inference.py` 買進建議提供），填了 `inference.py` 走精確價判停損，未填退回當日 ATR 近似

---

## 4. AI 作業規範
1. **常數集中**：任何常數必須宣告在 `config.py`，嚴禁在腳本中寫死。
2. **動態特徵命名**：新增技術指標需同步 `optimize_factors.py`，確保欄位名稱動態可讀。
3. **文件連結**：說明文件中只能用相對路徑（如 `[config.py](config.py)`），禁用 `file:///` 絕對路徑。
4. **Checkpoint 故障排查**：`mode_b={}` 代表模式B未執行；`sim=0.0` 代表解析錯誤（會自動重跑）。
5. **模式A vs B 風控差異**：B 比 A 寬鬆是正常現象（B 含大牛市數據），不代表過擬合。
6. **免測試情境**：僅修改文件／設定檔時無須執行 `tests/test_pipeline.py`。
7. **敏感金鑰**：`rclone.conf` 已列入 Git 忽略，絕對禁止提交。
8. **市況過濾器精度控管**：`REGIME_*`（config.py）依昨日大盤趨勢動態調整買入門檻，`trading_sim.py`／`inference.py` 共用。**僅在「未顯式指定 buy_threshold」時生效**，故 `param_sensitivity.py` 與 CLI `--buy_threshold` 走靜態值不受影響——修改時務必維持此優先序，否則會破壞敏感度掃描。regime 讀「昨日」狀態以防前視偏差。

---

## 4.5 策略不變量（已驗證，勿重蹈覆轍）

> 以下為跨多輪實驗驗證的結論，違反它們的「改進」已實測會讓績效崩潰。改策略前先讀。

1. **進場訊號弱、獲利靠出場**：進場分數 IC 低、十分位勝率近乎平坦（最高分組勝率僅約 52%）。獲利來自出場吃肥尾（Day3 持有／移動止盈）與 regime 曝險，不是靠進場命中率。**嚴禁用「拉高 `buy_threshold`／收緊進場」來減少停損次數**——已驗證會把肥尾贏家一起砍掉而崩潰（+59%→-9%）。要降回撤只能動出場／停損／regime 曝險，不能動進場嚴格度。
2. **回撤與報酬不可分割**：均勻降風險（縮 ATR 部位、無差別分散）會等比例砍報酬。唯有**選擇性**降曝險能破對稱——即 regime 導向（震盪／空頭降門檻、降檔數 `REGIME_MAX_POSITIONS`）與個股波動導向（ATR 停損）。已驗證 `REGIME_MAX_POSITIONS` 完整 OOS 同時 +13pp 報酬、−0.5pp 回撤（非取捨）。
3. **停損採 ATR 動態（生產預設）且兩端已對齊**：`trading_sim.py` 與 `inference.py` 共用同一套 ATR 停損（`-ATR_STOP_MULTIPLIER × atr18_pct`，夾 `[ATR_STOP_FLOOR_PCT, ATR_STOP_CEILING_PCT]`）。停損掃描已驗證 ATR 動態全面勝固定 -6/-8/-10%。`inference.py` 是日快照、用當日 ATR 近似買入日；`Stocks.txt` 第 4 欄填了買入日鎖定停損價才完全等價 `trading_sim`。改停損務必跑掃描（設 `config.ATR_STOP_ENABLED` 再呼叫 `run_simulation`），不要改進場門檻。

---

## 5. Coding Behavior Guidelines

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
