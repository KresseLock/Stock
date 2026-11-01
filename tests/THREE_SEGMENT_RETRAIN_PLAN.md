# 三段切分重訓實驗（THREE_SEGMENT_RETRAIN_PLAN）

腳本：[test_three_segment_retrain.py](test_three_segment_retrain.py)
建立日期：2026-11-01

---

## 1. 為什麼要跑

### 現況的問題
現行生產系統三層資料的切法互相重疊：

| 層 | 用到的資料 | 問題 |
|---|---|---|
| 模型（LightGBM） | 訓練集 2017-12 ~ 2023-11-24，驗證集到 2024-06-20 | 最新的訓練資料已近三年前 |
| 因子參數 `best_factors.json` | 截斷 2025-08-01 | — |
| 風控參數 `best_trading_params.json` | 2022-01-02 ~ 2026-07-23 | **看過 2026 年 1~7 月**，所以 2026 年任何回測對它而言都是樣本內，績效會被高估 |

結果是：**目前沒有一段區間能對「現行模型 × 現行參數」做真正乾淨的考試。** 只有 2026-08 以後是乾淨的，但只有三個月。

### 為什麼不能「模型訓練到 2025-08、參數調 2023~2026-07」
參數若在模型「看過的期間」調，模型在那段的分數會過度樂觀（實測：看過答案的 2020/2022 RankIC ≈ 0.15，沒看過的期間只有 0.06~0.12），參數會被調成配合虛高的分數，上線後失準。所以三層必須**依序、互不重疊**。

### 本實驗的切法

```
模型（因子最佳化＋訓練＋驗證）  2017-12 ━━━━━━━━━━━━━━ 2024-12-31
                                  訓練集 ~2024-06-28｜驗證集 2024-07-01 ~ 2024-12-30（early stopping 用）
風控參數調參                                              2025-01-02 ━━━━ 2025-12-31
乾淨裁判區                                                                2026-01-02 ━━━ 資料最後一天
```

- 裁判區 2026 年對候選的模型與參數**完全沒看過**，且涵蓋三次下跌（1~3 月、7 月股災、10 月下殺）。
- 調參區 2025 年對候選模型是樣本外，且同時有對等關稅崩跌與下半年大多頭兩種市況。

### 這個實驗回答／不回答什麼
- **回答**：用較新的資料、並以乾淨方式重跑整條流程（因子 → 特徵 → 訓練 → 調參），能否勝過現行。
- **不回答**：「轉空退場後錯過 V 轉反彈」的兩邊打臉問題。那是 regime 判定（20 日均報酬）本身落後的結構問題，換資料區間修不好，需另案處理。
- 既有反證（CLAUDE.md §4.5 #6）：2026-08-13「只重訓模型」的對照，兩個資料較新的候選皆雙輸現行。本實驗差別在於**完整重跑因子最佳化與調參**，那是 #6 明確寫著「尚未檢驗」的部分。

---

## 2. 腳本做了什麼

| 步驟 | 內容 | 寫到哪 |
|---|---|---|
| 0 | 取執行鎖（擋 `Auto_RUN.py`）＋心跳；快照現行 `models/`、特徵檔、`best_factors.json`、`best_trading_params*.json` | `models/_exp_three_segment_<時間>/incumbent/` |
| 1 | 因子最佳化 `optimize_factors`，截斷 2024-12-31（`--skip-factors` 可略過） | `configs/best_factors.json`（暫時） |
| 2 | 以新因子重建特徵檔 | `data/features/features_combined.parquet`（暫時） |
| 3 | 重訓 LightGBM：`BACKTEST_DATE=20241231`，切分比例自動反算成訓練集到 2024-06-28 | `models/`（暫時） |
| 4 | 快照候選（模型＋特徵檔＋因子三者成組） | `.../candidate/` |
| 5 | 在候選模型上調風控參數：`optimize_trading_params --regime -wf -s 2025-01-02 -e 2025-12-31`；基底參數用乾淨的 `mode_a` | `.../candidate/best_trading_params.json` |
| 6 | 裁判區 2×2 把關（下節）＋三段事件觀察 | `.../gate_runs.json`（逐筆即時寫入） |
| 7 | 產出報告；**finally 無論成功／失敗／Ctrl+C 一律還原所有生產檔案、釋放鎖** | `.../RESULT.md` |

設計要點：
- **只在本行程以 module 屬性覆寫 `config.BACKTEST_DATE` 與切分比例，不改 `config.py`。**
- 模型、特徵檔、因子**必須成組切換**：因子不同 → 特徵欄名不同 → 只換模型會被 reindex 靜默補 NaN（記憶 `feature-cols-version-mismatch`）。
- 回測在子行程以 `export_report=False` 執行，不會在 `reports/` 留下 Excel/CSV。
- 中斷可續跑：`--exp-dir` 指向同一個實驗目錄，已完成的步驟（候選模型、候選參數、每一筆回測）會跳過；續跑時**不會重新快照**（避免把候選誤當現行）。

---

## 3. 怎麼執行

```powershell
# 1) 先確認計畫與切分日期（唯讀，幾十秒）
python tests/test_three_segment_retrain.py --dry-run

# 2) 正式執行（背景跑，數小時到半天以上，視機器）
python tests/test_three_segment_retrain.py

# 中斷後續跑（目錄名稱見第一次執行時印出的「實驗目錄」）
python tests/test_three_segment_retrain.py --exp-dir models/_exp_three_segment_YYYYmmdd_HHMMSS
```

選項：`--skip-factors`（省掉最久的因子最佳化，但現行因子看過 2025-01~08，調參區不再完全乾淨；裁判區仍乾淨）、`--factor-trials` / `--param-trials`（預設 `config.OPTIMIZATION_TRIALS`）、`-j`（調參並行數，預設 4）、`-c`（資金，預設 200 萬）。

### 執行期間的注意事項
- **`Auto_RUN.py` 會被執行鎖擋下**（印出「已在執行中」後結束），實驗期間當天沒有每日預測。排程若固定晚上跑，建議挑不需要當日建議的日子開始。
- **不要在實驗期間手動跑 `trading_sim.py`、`inference.py` 或 `auto_pipeline.py`**：它們不檢查執行鎖，而此時 `models/`、特徵檔、參數可能是候選版本，得到的結果不代表生產。
- 執行記錄：`reports/test_three_segment_retrain_*.log`（tee，含完整輸出）；調參另有 `reports/optimize_trading_params_*.log`。
- 實驗目錄含兩份特徵檔（各約 400MB），放在 `models/` 底下（已被 `.gitignore` 排除）。

---

## 4. 執行完後怎麼驗證

### (a) 生產環境已還原（先做這個）
- log 最後一行「[還原] … models/ 指紋 X」的 X 必須等於開頭「現行模型指紋」。
- 比對：`python -c "from scripts.utils import model_fingerprint; print(model_fingerprint())"` 應為實驗前的值（2026-11-01 時為 `949e10535b3a`）。
- `configs/best_trading_params.json` 的 `date_range` 應仍是 `2022-01-02 to 2026-07-23`；`configs/best_factors.json` 的 `backtest_date` 應仍是 `20250801`。
- `git status` 中 `configs/`、`models/feature_cols.json` 不應出現變更。

### (b) 候選本身是否照設計產生
- `candidate/split.json`：訓練集應為 `2017-12-18 ~ 2024-06-28`、驗證集 `2024-07-01 ~ 2024-12-30`。
- log 步驟 3 中 `train.py` 印出的「訓練集／驗證集」日期區間須與上面一致。
- `candidate/best_factors.json` 的 `backtest_date` 應為 `20241231`（未加 `--skip-factors` 時）。
- `candidate/best_trading_params.json` 的 `date_range` 應為 `2025-01-02 to 2025-12-31`。

### (c) 判讀 `RESULT.md`
2×2 的兩軸：模型（現行／候選）× 參數（舊＝`mode_a`，調參至 2025-08-01、沒看過裁判區／新＝本輪）。

- **部署判準**（沿用 §4.5 #4）：「候選模型 × 新參數」對「現行模型 × 舊參數」，**報酬中位數與回撤中位數雙贏，且 9 條路徑中逐路徑雙贏 ≥ 5 條**。三者缺一即不通過。
- **歸因**：看「新模型的效果」兩列與「新參數的效果」兩列——判斷改善來自模型、參數，或兩者配對。只有配對有效（交叉格都不好）＝配對效應，可移植性較差。
- **參考格「現行模型 × 生產參數」不進判準**：生產參數看過 2026-01~07，天然偏袒，拿它當對照會誤導。
- **事件段**只看系統在三次下跌中的反應（持股、回撤），**不可用來判優劣**（§4.5 #5：短區間會截斷跨區間贏家）。
- 通過後仍須注意：新參數若有值貼在搜尋邊界（例如 `markup_pct`），先跑 `tests/test_markup_bound_gate.py` 排除單格噪音峰。

### (d) 通過／不通過之後
- **通過**：不要直接覆蓋。由使用者決定後手動部署（候選模型三檔＋`feature_cols.json` 複製回 `models/`、`best_factors.json`、`best_trading_params.json`），並記錄新的模型指紋（§4.5 #9）。部署後 `Auto_RUN.py` 會重建特徵，與候選特徵檔一致。
- **不通過**：生產已自動還原，無須動作。結論寫入 `scripts/EXPERIMENTS_PENDING.md`，並依 §4.5 #6 的精神註明：本次是「完整重跑」的單一實驗，裁判區僅約 10 個月，結論強度有限。

---

## 5. 已知限制
- 調參區只有一年；`-wf` 四個子窗口各約該年的 65%，參數穩定度可能不如過去以 2022~2025 調參的版本。
- 截斷日前最後 3 個交易日的標籤會用到 2025-01 初的報酬（label 前看 1~3 天），量極小，現行設計同樣如此。
- 因子最佳化會以現行 `best_factors.json` 暖啟動（只當第一個試驗點、仍以截斷前資料評分），不構成洩漏，但搜尋起點偏向現行解。
- 裁判區單一、約 10 個月；即使通過，也只是「這段期間」的證據。
