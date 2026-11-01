# 台灣股市量化交易系統 - AI 導航指南 (CLAUDE.md)

## 1. 系統概覽
全自動台股量化預測回測系統。核心流程：
`scraper.py` → `optimize_factors.py` → `feature_engineering.py` → `train.py` → `inference.py` → `trading_sim.py`

唯一設定檔：[config.py](config.py)（所有常數、產業、門檻均在此，**嚴禁在腳本中分散寫死**）。

---

## 2. 核心腳本速查

| 腳本 | 功能摘要 |
|------|---------|
| `scripts/scraper.py` | 增量下載 TWSE/TAIFEX/FinMind。`-s twse/finmind` 只跑單一來源（分鎖，可並行）、`-p <sid>` 補財報、`-fc` 更新產業表、`-c` 檢查清理。FinMind 母體由 `FINMIND_FETCH_MODE` 決定（`listed` 由 `data/raw_price` 全歷史推導，快取於 `data/universe_cache.json`） |
| `scripts/feature_engineering.py` | 技術指標＋板塊情緒（方案B）＋混合標籤（方案C），輸出 `features_combined.parquet` |
| `scripts/optimize_factors.py` | Optuna TPE 尋最佳技術指標參數 → `configs/best_factors.json` |
| `scripts/train.py` | 訓練 LightGBM，大跌樣本加權懲罰，輸出 `models/feature_cols.json` |
| `scripts/inference.py` | 推論未來3天多空分數，提供含 Tick Size 對齊的掛單建議 |
| `scripts/optimize_trading_params.py` | Optuna 最佳化風控參數 → `configs/best_trading_params.json` |
| `trading_sim.py` | OOS 回測，含 T+2 交割模擬、限價溢價搓合、Excel 輸出 |
| `Auto_RUN.py` | 生產主控：`-s download/predict/backup/all`（預設 all），`-d YYYYMMDD` 指定推理日；download 模式額度不足會等待而非跳過。**predict／all 只重建特徵＋推論、不重訓**（§4.5 #9），並印出模型指紋。與 `run_workflow_experiment.py` 共用執行鎖（`config.RUN_LOCK_PATH`），不可同時執行 |
| `auto_pipeline.py` | 研發流水線：`-s optimize/feature/train/inference`；不帶 `-s`（all）**會重訓**，生產請改走 `Auto_RUN.py` |
| `run_workflow_experiment.py` | 雙模式實驗（模式A研究期截至2025-08-01／模式B生產期），含 Checkpoint 斷點續傳。結束時（含例外）還原執行前的 `models/`＋特徵檔，Model B 只留在 `models/*.mode_b`；B8 實驗預測移至 `reports/workflow_mode_b_*`，不覆蓋 `predictions/` |
| `scripts/analyze_regime_stability.py` | OOS 訊號健康診斷：RankIC / Alpha / PSI 漂移 |
| `scripts/param_sensitivity.py` | 參數敏感度診斷（OFAT）：跨市況掃單一風控參數，解釋優化器為何選某值 → `reports/param_sensitivity_report.md` |
| `scripts/compare_model_modes.py` | 模型時效性三方對照（現行／解除截斷／訓練集涵蓋近期）：自動備份→重訓→同區間回測→出報告→還原環境，候選模型保留供 `--promote` 採用 |
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
- `Stocks.txt` — 自選股／持倉清單；**格式 D 第 4 欄＝買入日鎖定 ATR 停損價**（`代號,成本,股數,停損價`，由 `inference.py` 買進建議提供），填了 `inference.py` 走精確價判停損，未填退回當日 ATR 近似。**格式 E 第 5 欄＝買入日期**（`代號,成本,股數,停損價,買入日期`，YYYY-MM-DD）；填了 `inference.py` 才比照 `trading_sim.py`：D3轉弱／移動止盈出場須滿 `MIN_HOLD_DAYS`（停損不限）且啟用移動止盈判定，未填退回「無視持有天數、D3 轉弱即建議賣」

---

## 4. AI 作業規範
1. **常數集中**：任何常數必須宣告在 `config.py`，嚴禁在腳本中寫死。
   **資料路徑同理**：一律取自 `config.py` § 0（`DATA_DIR` / `RAW_*_DIR` / `FEATURES_PARQUET`），嚴禁自行拼接 `data` 目錄——大小寫不同的拼法（`data` vs `Data`）在 Windows 上會恰好指向同一處而看不出問題，移植到 Linux/macOS 才會分裂成兩個資料目錄，症狀是「明明抓過卻又整批重抓」。少數腳本（`scraper.py`、`trading_sim.py`、`inference.py` 等）在 `except ImportError` 內留有等價 fallback 以支援脫離 config 獨立執行，**改 § 0 的目錄名稱時必須同步這些 fallback**。
2. **動態特徵命名**：新增技術指標需同步 `optimize_factors.py`，確保欄位名稱動態可讀。
3. **文件連結**：說明文件中只能用相對路徑（如 `[config.py](config.py)`），禁用 `file:///` 絕對路徑。
4. **Checkpoint 故障排查**：`mode_b={}` 代表模式B未執行；`sim=0.0` 代表解析錯誤（會自動重跑）。
5. **模式A vs B 風控差異**：B 比 A 寬鬆是正常現象（B 含大牛市數據），不代表過擬合。
6. **免測試情境**：僅修改文件／設定檔時無須執行 `tests/test_pipeline.py`。
7. **敏感金鑰**：`rclone.conf` 已列入 Git 忽略，絕對禁止提交。
8. **市況過濾器精度控管**：`REGIME_*`（config.py）依昨日大盤趨勢動態調整買入門檻，`trading_sim.py`／`inference.py` 共用。**僅在「未顯式指定 buy_threshold」時生效**，故 `param_sensitivity.py` 與 CLI `--buy_threshold` 走靜態值不受影響——修改時務必維持此優先序，否則會破壞敏感度掃描。regime 讀「昨日」狀態以防前視偏差。
9. **README 禁寫績效數字**：修改 `README.md` 時，嚴禁寫入任何具體回測績效數字（報酬率、MDD、Calmar、勝率等），因為這些數字會隨執行時的資料區間、資金、模型與參數版本而變動，寫死在文件裡等同暗示「保證績效」，會誤導讀者。要說明機制或案例時，只描述**方向與相對關係**（例如「候選在乾淨 OOS 明顯輸給現行」），不寫絕對百分比；實際數字一律留給使用者自行執行對應指令查看。此規則與 README 既有「系統績效」一節的原則一致，修改任何段落都適用，不限該節。
10. **禁止代為 commit／push**：一律不執行 `git commit`、`git push` 或任何改寫歷史的指令（`rebase`、`reset --hard`、`filter-repo` 等）。修改完程式碼後只需說明改了什麼、留在工作區，**由使用者自行 commit**。即使使用者說「做完這件事」也不含 commit；要 commit 會由使用者明確指示。

---

## 4.5 策略不變量（已驗證，勿重蹈覆轍）

> 以下為跨多輪實驗驗證的結論，違反它們的「改進」已實測會讓績效崩潰。改策略前先讀。

1. **進場訊號弱、獲利靠出場**：進場分數 IC 低、十分位勝率近乎平坦（最高分組勝率僅約 52%）。獲利來自出場吃肥尾（Day3 持有／移動止盈）與 regime 曝險，不是靠進場命中率。**嚴禁用「拉高 `buy_threshold`／收緊進場」來減少停損次數**——已驗證會把肥尾贏家一起砍掉而崩潰（+59%→-9%）。要降回撤只能動出場／停損／regime 曝險，不能動進場嚴格度。
2. **回撤與報酬不可分割**：均勻降風險（縮 ATR 部位、無差別分散）會等比例砍報酬。唯有**選擇性**降曝險能破對稱——即 regime 導向（震盪／空頭降門檻、降檔數 `REGIME_MAX_POSITIONS`）與個股波動導向（ATR 停損）。已驗證 `REGIME_MAX_POSITIONS` 完整 OOS 同時 +13pp 報酬、−0.5pp 回撤（非取捨）。
3. **停損採 ATR 動態（生產預設）且兩端已對齊**：`trading_sim.py` 與 `inference.py` 共用同一套 ATR 停損（`-ATR_STOP_MULTIPLIER × atr18_pct`，夾 `[ATR_STOP_FLOOR_PCT, ATR_STOP_CEILING_PCT]`）。停損掃描已驗證 ATR 動態全面勝固定 -6/-8/-10%。`inference.py` 是日快照、用當日 ATR 近似買入日；`Stocks.txt` 第 4 欄填了買入日鎖定停損價才完全等價 `trading_sim`。改停損務必跑掃描（設 `config.ATR_STOP_ENABLED` 再呼叫 `run_simulation`），不要改進場門檻。
4. **進場端改造（新特徵／新過濾器）屢試屢敗，別再盲試——先過潔淨 OOS 把關**：呼應 #1（進場訊號本就弱），多輪候選在隔離潔淨 OOS（train 截 `BACKTEST_DATE`、複用生產 train/sim）皆被否決。**已驗證否決、勿重做**：
   - **法人淨額成交量正規化＝有害**（`tests/test_chip_flow_normalization.py`，2026-07-16）：把 fini/sitc/dealer/inst 淨額與滾動合計除以成交量，全 OOS 報酬 +224%→+97% 且回撤變大。教訓：**法人淨額的「絕對量級」本身帶訊號**（大額買超≠小額買超），正規化把量級丟掉即毀訊號。「絕對值＝Level Bias 雜訊、該正規化」的直覺**只對財報/估值 level 成立，對法人流量不成立**。
   - **自營商改用「自行買賣」淨額（剔除避險造市）＝平手**（同檔測試）：+238% vs baseline +224%、Calmar 11.9 vs 11.6，在單次回測噪音內，不值得為它多維護原始 chips.csv 重讀與新欄相依。
   - **fortune 移植 z-score／日曆特徵＝否決**（`tests/test_feature_candidate_gate.py`）：僅 1/4 子窗勝出、疑過擬合。
   - **進場流動性門檻（近 20 日均成交金額下限）＝否決**（`tests/test_liquidity_gate.py`，2026-09-08）：起因是 2026-09-02 推論把 2438 翔耀（日均成交 143 萬元）、6272 驊陞（495 萬元）排進買進前三名，而同份清單的 3026 禾伸堂為 63 億元——`config.py` 確實沒有任何流動性／市值門檻。6 起點 × 6 門檻（0／0.1／0.3／0.5／1／2 億元）掃描結果：**全 OOS 單窗漂亮到會騙人**（報酬隨門檻由 +106% 單調升至 +166%、Calmar 4.79→6.87、轉折僅 1 次），但多起點雙贏僅 0～2/6、回撤幾乎全面惡化。門檻 1 億對現行的逐起點報酬差為 `+59 / +7 / −15 / +183 / −1 / −24` pp——起點只差一個月（2025-12-01→2026-01-01）就從 −15pp 跳到 +183pp。
     **為何是雜訊**：門檻對持倉結構影響極小（均持倉 4.32→4.12~4.25、曝險 84.7%→79.7~82.6%），代表它多半擋掉的是本來也不會進場的標的，真正改變的交易筆數極少；少數肥尾經 `MAX_POSITIONS=5`（單筆權重 20%）複利放大即足以造成 ±180pp 擺盪。且它屬**無差別**降曝險而非 regime／ATR 的選擇性降曝險，故 #2 成立：回撤不降反升。
     **關鍵區分（別搞混）**：低流動性股進榜暴露的是「回測以歷史 OHLC 搓合、不模擬衝擊成本」的**執行面**風險，不是績效面問題，應以下單紀律（嚴守建議限價、買不到就放棄）處理，不要用進場過濾器解。未試變體：相對排名門檻、與部位規模掛鉤的門檻（日均成交 > 單筆部位的 K 倍）；要試須沿用本框架的多起點判準。
   - **進場 D3 門檻＝否決**（`tests/test_d3_entry_gate.py`，2026-09-08）：要求「訊號日 `Day3_net` >= 門檻」才准進場，5 個門檻全 OOS 報酬差全為負、多起點雙贏 0～1/6。**機制與教訓見 #8**（D3 是出場訊號，挪到進場端即失效）。
   - **方法論**：任何進場端候選（新特徵或新過濾器）必先寫進上述其一的把關框架跑贏才可移植；**判準是「全 OOS 報酬與回撤雙贏 且 多數子窗穩健」**，單看全窗改善不算數（易由少數窗驅動）。想再從籌碼榨訊號，別走正規化淨額（死路）；未試、且不需新資料的方向：法人分歧度（`sign(fini)×sign(sitc)` 同向/對作）。
5. **禁用「按月切割」評估策略——會系統性低估績效**：本策略 `MIN_HOLD_DAYS` 為 11～23 天且獲利靠肥尾（見 #1），按月分段回測會月初空手重建倉、月底強制截斷未平倉，把跨月贏家攔腰砍斷。已實測（2026-08-13，同模型同參數）：連續區間 2026-04-01~07-31 的報酬，遠高於同期四份單月報表的複利結果。評估一律用連續區間，單月報表只能看交易明細、不能拿來論績效。
6. **「模型訓練資料不夠新」不是近月績效不佳的解釋（只重訓模型、完整重跑皆已驗證）**：`BACKTEST_DATE` 只決定資料截斷點，真正 fit 的訓練集終點由 `TRAIN_SPLIT_RATIO` 決定（截斷點的 70%）。已用 `scripts/compare_model_modes.py` 三方對照（2026-08-13，2026-04~07 區間）：訓練集終點 2023-11（現行）／2024-08／2025-12 三者，績效與「資料新鮮度」**無單調關係**，且兩個較新的候選皆雙輸現行。
   - **範圍限制**：該實驗只重跑 `train.py`，`best_factors.json`（2026-06-21、`backtest_date=20250801`）與 `best_trading_params.json` 固定、特徵檔共用，屬單一變因設計。
   - **完整重跑也已否決**（`tests/test_three_segment_retrain.py`，2026-11-01，詳見 `scripts/EXPERIMENTS_PENDING.md`）：三層互不重疊——因子＋訓練至 2024-12-31｜調參 2025 全年｜裁判區 2026-01-02~10-30（9 條起點偏移路徑）。「候選模型×新參數」對「現行模型×`mode_a`」報酬與回撤雙輸、逐路徑 0/9。**主因在參數端**：新參數在兩個模型上都 0/9 輸 `mode_a`，walk-forward 診斷 9 參數有 7 個不穩定、四窗中位數解在自己的調參年就是負報酬，且選出 `regime_bull_pos=5`（等於取消 Bull 降檔，違反 #2）。候選模型在中立的 `mode_a` 下也較弱（0/9）。教訓：**調參區只有一年撐不起穩定解**，日後若重試，調參區至少跨兩年、涵蓋多種市況。
   - **補測時的硬限制**：`optimize_factors.py` 的防洩漏保護看 `config.BACKTEST_DATE`，設 `None` 時**不截斷**，會把回測區間納入因子搜尋（lookahead）。故完整重訓比較只能用「截斷於回測起始日前一日」的設定，`None` 型無法做乾淨對照。
   - ※ 單窗 30 筆交易、噪音大，反向結論（舊模型較優）同樣不成立，要動模型時效性須先跑多窗驗證。
7. **「降級砍倉／閃現部位提前出清」已二度否決，程式碼已刪除，勿再實作**：構想是「regime 由 Bull 降級且持倉超過 `REGIME_MAX_POSITIONS` 時，出清買後 ≤N 個交易日的 Bull 進場部位（LIFO、開盤價、不受 `MIN_HOLD_DAYS` 限制）」。**第一次否決**（`scripts/EXPERIMENTS_PENDING.md`）：全面砍倉版本 −53.8 萬、誤殺 20 筆贏家。**第二次否決**（2026-08-26）：改成只砍「閃現部位」的收斂版本，`DOWNGRADE_FLASH_TRIM=12` 曾短暫進入 `config.py`（未 commit），現已從 `config.py`／`trading_sim.py`／`scripts/inference.py` 完整移除。
   - **判死數字**：終點固定 2026-08-25、六個起點的多起點掃描，trim=12 對比停用的報酬差為 `+75 / +31 / +22 / −72 / −48 / +21` pp（起點 2025-08-02、10-01、12-01、2026-01-01、02-02、04-01）。**報酬期望值≈+5pp、變異達 ±70pp**；回撤雖 6/6 起點改善約 5pp，Calmar 僅 4/6 改善。
   - **為何必然失敗（機制層面，這才是重點）**：該規則觸發 16 次中有 **13 次砍的是獲利部位**，而 §4.5 #1 已確立本策略獲利完全來自出場端吃肥尾。它砍的正是「短期暴漲」的部位——那恰恰是肥尾贏家的早期特徵。一旦砍到就是數十 pp 的損失，這不是可調參數能修好的，是**規則定義與獲利來源直接對立**。
   - **當初為何誤判為「有效」**：取值依據只看單一全窗（2025-08-02~2026-08-25）的雙贏，而該窗的優勢**全部由 2025 下半年 5 個月產生並被複利放大**（前段抬高期末資金再吃後段），2026 那 8 個月其實大幅為負。呼應 #4 判準：**單看全窗改善不算數**。且掃描曲線根本不單調（trim=3 反而大輸 trim=0 達 38pp），當時卻被描述成「單調遞增後轉平的高原」——16 個觸發樣本撐不起任何「高原」結論。
   - **教訓（可推廣）**：(a) 任何新出場規則都必須先問「它會不會砍到肥尾贏家」，若觸發樣本多數為獲利部位就該直接懷疑；(b) 參數掃描必須看**多起點**而非單一窗，且非單調曲線＝雜訊擬合，不可解釋成高原——反過來也不成立：#4 的流動性門檻全窗曲線近乎完全單調（轉折 1 次）仍是雜訊，**單調不是穩健的證據，多起點一致才是**；(c) 提防與 `REGIME_MAX_POSITIONS` 的隱性耦合——當 Sideways 上限等於 `MAX_POSITIONS` 時該規則實質只在 Bear 生效，換一組風控參數它的作用面就完全改變，這種「參數依賴的觸發率」本身就使結論不可移植。
8. **訊號有專屬作用位置，跨端挪用即失效——D3 是出場訊號，不可拿來當進場過濾**：`Day1_net`／`Day3_net` 在本系統有明確分工：**進場吃 D1 動能，出場靠 D3 轉弱＋移動止盈＋ATR 停損**（`scripts/inference.py:504` 的 D3 轉弱判定即為此而設）。進場端只看 D1、完全不看 D3（`trading_sim.py:535`、`scripts/inference.py:592`），這是**設計，不是疏漏**。
   - **已驗證否決**（`tests/test_d3_entry_gate.py`，2026-09-08）：加一道「訊號日 `Day3_net >= 門檻` 才准進場」，掃 6 起點 × 5 門檻（-15／-10／-5／0／+5）：**全 OOS 報酬差全為負**（−24.9／−63.7／−103.0／−68.5／−110.4 pp），多起點雙贏 0～1/6，曲線非單調（轉折 2 次）。**無任何門檻在任何指標上贏過現行。**
   - **決定性證據**：最寬鬆的 `D3>=-15` 只擋掉 21.7% 樣本，曝險 84.7%→84.4%、均持倉 4.32→4.32 **幾乎沒變**，報酬卻掉 25pp。部位數與曝險都沒降、只是換了一批標的進來，報酬就崩 → 被擋掉的不是「會虧的股」，而是**會賺的股**。且過濾越多回撤越糟（`D3>=-5` 的 MDD 33.6% vs 現行 22.2%），完全不是「避開弱勢股」該有的樣子。
   - **機制（為何必然失敗）**：D3 負值不是「別買」，而是「進場後準備要跑」。D1 強／D3 弱的標的正是「短線爆發、隨後回落」型，恰恰是 #1 所說的肥尾贏家來源——系統本來就靠出場端在它們衝高時鎖利。把 D3 搬到進場端＝把出場工具挪去做它不該做的事，同時砍掉獲利來源本身。
   - **推廣**：任何「把某端訊號拿去另一端用」的構想（D3 當進場過濾、D1 當出場條件等），都必須先問「這個訊號在現行架構中負責什麼」。本例正是由「模型說 D3 弱卻照買＝自相矛盾」的誤讀出發，實測證明那是**分工而非矛盾**——別把分工誤讀成 bug。
9. **生產模型不每日重訓——換模型是一次部署，必須手動＋把關**（2026-09-12 起）：`Auto_RUN.py` 的 predict／all 只跑 `auto_pipeline.py -s feature` 與 `-s inference`，不再跑 `-s train`。
   - **為何重訓沒有好處**：`train.py` 以 `BACKTEST_DATE` 截斷資料，訓練／驗證集是固定的，每天重訓學不到任何新資料；`train.py` 又可逐位元重現（`random_state` 固定），輸入不變時產出完全相同。
   - **為何重訓有害**：只要截斷日以前的輸入有任何變動，就會產生一個新的、沒把關過的模型實現——FinMind／TWSE 補抓舊資料、`Stocks.txt` 持倉變動（`filter_stocks_by_train_industries` 會把持倉股強制併入訓練母體）、`scraper.py -fc` 更新產業表。實例：2026-09-10 補抓到 5 支已下市股 2017-2019 的財報（僅佔該兩年 0.16% 列），當晚的每日重訓就把通過把關的模型換掉，裁判區報酬中位數差 +15~25pp，事後手動比 md5 才發現（`scripts/EXPERIMENTS_PENDING.md` 鐵律 #8）。
   - **要換模型時**：手動 `python auto_pipeline.py -s train` → 模型×參數 2×2 把關（鐵律 #7）→ 通過才留用，否則還原。以 `scripts/utils.py` 的 `model_fingerprint()` 對帳；`Auto_RUN.py` 每次執行會印出當前指紋，**指紋變了卻沒人手動重訓＝有東西動了 `models/`**。
   - **連帶防呆**：`inference.py` 在特徵檔缺少 `feature_cols.json` 任一欄時直接中止（不再每日重訓後，改特徵程式的欄名會讓 reindex 靜默補 NaN）；`run_workflow_experiment.py` 結束時還原執行前的 `models/`＋特徵檔，B8 實驗預測不覆蓋 `predictions/`。
   - **勿誤用**：`auto_pipeline.py` 不帶 `-s`（all）仍會重訓，那是研發用途，別拿來當每日生產指令。
10. **績效評估一律對照同期 0050 買進持有，「系統 vs 系統」勝出不等於有用**（2026-11-01）：把關工具（`validate_candidate_params.py`、2×2 把關等）只比兩組系統設定，回答不了「能不能打贏什麼都不做」。任何「值得上線／有 alpha」的結論，都必須在**乾淨區間**（模型與參數皆沒看過）附上同期 0050 的報酬與回撤；以 `data/raw_price` 收盤價計時只有價格報酬，須註明不含息（含息基準更高）。
    - **現況**：乾淨 alpha 時有時無——2026 年贏、從 2025-08 起算則輸，量級落在執行面（觸價成交假設）與起點離散度可吞掉的範圍內，判定為**證據不足**。數字與推論見 `scripts/EXPERIMENTS_PENDING.md`「現況評估：對 0050 的乾淨 alpha」。
    - **推論**：回測已無乾淨區間可用，下一步是前瞻資料（紙上交易／小額實盤、真實成交價）對 0050，而不是再調參或重訓。看過裁判區的生產參數績效（如對 2025 下半年的高報酬）不可拿來當作 alpha。

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
