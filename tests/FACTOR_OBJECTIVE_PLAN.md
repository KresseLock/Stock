# 因子管線修復 × 多窗把關 × 目標函式重構（系統規劃書）

> 建立 2026-08-14 ／ 狀態：**規劃中，尚未動工**
> 起因：[../reports/compare_model_modes_20260813_224948.log](../reports/compare_model_modes_20260813_224948.log) 顯示
> 「Optuna 代理目標從 21.93% 改善到 22.19%，實際回測卻從 +37.68% 掉到 +20.41%」。
> 追查代理目標為何與 P&L 反向時，發現一個**更上游的機械性缺陷**（見 §1），
> 它改變了整份規劃的優先序：目標函式重構（原第 1 步）必須排在管線修復（新增第 −1 步）之後。
>
> 相關：[../scripts/compare_model_modes.md](../scripts/compare_model_modes.md)、[../.claude/CLAUDE.md](../.claude/CLAUDE.md) §4.5

---

## 0. TL;DR

| 步驟 | 名稱 | 為什麼 | 產出 | 估時 |
| :--- | :--- | :--- | :--- | :--- |
| **−1** ✅ | 因子參數傳遞修復 | Optuna 搜出的 TA 參數在生產特徵中**完全沒生效**（16 個參數只有 3 個真的落地） | `feature_engineering.py` 改為顯式傳參 + 契約測試 | 半天 |
| **0** | 多窗把關框架 | 單窗 30 筆交易，任何改動的效果都埋在噪音裡，無法判定 | `tests/test_multiwindow_gate.py` | 1 天 |
| **1** | 目標函式重構 | 代理目標優化「命中率＋1~3 日」，策略獲利靠「幅度＋20 日肥尾」，兩者不同調 | `optimize_factors.py` 可切換目標 + config 常數 | 1 天 + 數小時跑批 |

**一句話**：先讓因子參數真的生效（−1），再建能判定「有沒有用」的量尺（0），最後才改優化目標（1）。順序不可對調——在缺陷未修、量尺未建之前改目標函式，得到的任何結論都不可採信。

**核心結論（先講清楚）**：`--full` 那輪實驗的績效差異，**不是**「重新搜尋的 16 個因子參數」造成的，而是只有 `CHIPS_SUM_WINDOWS` 一個參數（外加訓練窗設定）造成的。其餘 11 個技術指標參數從頭到尾是死的。

---

## 1. 新發現：因子參數在生產特徵中大部分沒有生效

### 1.1 症狀（決定性證據）

`models/_candidate_b2full/` 是 2026-08-14 00:31 用**當時剛搜出的新因子**重建的特徵檔。比對它的
`best_factors.json` 與實際欄位：

| 參數 | best_factors.json 要求 | 特徵檔實際欄位 | 生效？ |
| :--- | :--- | :--- | :---: |
| `MA_WINDOWS` | `[4, 11, 12, 92]` | `ma9, ma20, ma32, ma52` | ❌ |
| `RSI_PERIOD` | `16` | `rsi7` | ❌ |
| `ATR_PERIOD` | `26` | `atr18`, `atr18_pct` | ❌ |
| `KD_PERIOD` | `13` | `k15, d15` | ❌ |
| `VOL_MA_WINDOW` | `3` | `vol_ma8, vol_ratio8` | ❌ |
| `BOLL_WINDOW` / `BOLL_STD_MULT` | `17` / `2.27` | `boll_*`（名稱不含期間，無法從欄名判定，但與上列同函式計算，故同樣無效） | ❌ |
| `MACD_FAST/SLOW/SIGNAL` | `16/29/6` | `macd, macd_sig, macd_hist`（同上） | ❌ |
| `CHIPS_SUM_WINDOWS` | `[7, 10, 19]` | `*_sum7, *_sum10, *_sum19` | ✅ |

`ma9/ma20/ma32/ma52`、`rsi7`、`atr18`、`k15`、`vol_ma8` 全部等於
[../scripts/feature_engineering.py](../scripts/feature_engineering.py) 第 61~73 行的**模組預設值**。
生產特徵檔 `data/features/features_combined.parquet` 也是同一組預設值欄位（其 best_factors 為
`MA=[4,19,43,95] / RSI=8 / ATR=23 / KD=5 / VOL=6`，同樣沒生效）。

### 1.2 病因

[../scripts/feature_engineering.py](../scripts/feature_engineering.py):584

```python
def apply_parallel(df_grouped, func):
    res = Parallel(n_jobs=N_JOBS)(delayed(func)(group) for _, group in df_grouped)
    return pd.concat(res, ignore_index=True)

df = apply_parallel(df.groupby("stock_id"), _compute_ta)   # ← 問題在這
```

`_compute_ta`（第 140~211 行）用的是**模組全域變數** `MA_WINDOWS` / `RSI_PERIOD` / `ATR_PERIOD` …。
而 [../auto_pipeline.py](../auto_pipeline.py):72 `_apply_best_params()` 是用 `setattr(fe_module, ...)`
在**父行程**覆寫這些全域。

`joblib.Parallel` 預設 loky backend → 另開行程 → Windows 為 spawn 模式 → 子行程**重新 import 模組**
→ 只看得到模組原始預設值，看不到父行程的 `setattr`。

最小重現（已實測）：

```
父行程 setattr 前 RSI_PERIOD = 7
父行程 setattr 後 RSI_PERIOD = 16
子行程看到的 RSI_PERIOD  = [7, 7]      ← joblib n_jobs=2
n_jobs=1 看到的         = [16]        ← 同行程時才正確
```

籌碼特徵（第 666 行「計算籌碼連續買賣超特徵…」）在**父行程**計算，沒有經過 `Parallel`，
所以 `CHIPS_SUM_WINDOWS` 是唯一生效的參數——與 §1.1 的觀察完全吻合。

`config.FEAT_N_JOBS = -1`（[../config.py](../config.py):214）代表生產環境永遠走多行程路徑，
亦即**這個缺陷在生產上 100% 觸發**。

### 1.3 影響範圍（哪些既有結論要重讀）

| 既有結論 | 是否受影響 | 說明 |
| :--- | :---: | :--- |
| `--full` 那輪 `b2full` 雙輸現行 | **⚠️ 需重讀** | 差異只來自 `CHIPS_SUM_WINDOWS` + 訓練窗，不是「16 個因子」。原報告「b2full 的變因不只一個」的說法仍對，但變因清單錯了 |
| CLAUDE.md §4.5 #6「完整重跑 pipeline 沒用」 | **⚠️ 需重讀** | 該實驗實際上沒有真的重跑因子，等於未檢驗 |
| `b1` / `b2`（只重訓模型） | ✅ 不受影響 | 兩者沿用現行特徵檔，沒有動因子 |
| §4.5 #1 進場訊號弱、獲利靠出場 | ✅ 不受影響 | 由回測行為推得，與因子生效與否無關 |
| §4.5 #2 regime 選擇性曝險有效 | ✅ 不受影響 | 風控層，與特徵無關 |
| §4.5 #4 法人正規化／LTR／fortune 特徵否決 | ✅ 不受影響 | 三者都在 sandbox 內自建特徵，未經 `apply_parallel` 路徑 |
| `best_trading_params.json` 風控參數 | ✅ 不受影響 | 在既有特徵上優化，特徵沒變，參數仍有效 |
| Optuna 歷次搜尋的「最佳勝率」數字 | ✅ 數字本身有效 | `optimize_factors.py` 用 `compute_ta(df, p)` 顯式傳參、且 Optuna `n_jobs` 是**執行緒**（同行程），所以目標函式測的是真的參數組。**問題是它測的東西從未被生產採用** |

> 最後一列是本案的核心荒謬處：**優化器評估的特徵配置，與生產實際產生的特徵配置，是兩套不同的東西。**
> 這本身就足以製造「代理目標改善、實際績效變差」，且與 §5 的度量錯位是**兩個獨立的問題**，都要修。

### 1.4 這不是「修好就變好」——修復本身是一次重大變更

修好之後，`best_factors.json` 裡那些沉睡已久的 TA 參數會**突然開始生效**，
生產特徵矩陣會整體改變（`ma9→ma4`、`rsi7→rsi8`、`atr18→atr23`…）。這等同換掉整組特徵，
**必須經 §4 的多窗把關才能進生產**，不可直接上線。

因此 Step −1 的交付定義是：「讓參數**可以**正確生效，並提供開關與驗證」，
而不是「讓新參數立刻生效」。

---

## 2. 目標系統框架

### 2.1 分層與契約

```
L0 資料層   scraper.py                     → data/raw_*                  契約：日期完整性、無重複
L1 因子層   optimize_factors.py            → configs/best_factors.json   契約：目標函式與 P&L 同調（§5）
L2 特徵層   feature_engineering.py         → features_combined.parquet   契約：L1 參數 100% materialize（§3）
L3 模型層   train.py                       → models/*.txt + feature_cols 契約：與 L2 欄位對齊、無 lookahead
L4 策略層   trading_sim.py / inference.py  → 回測與掛單建議              契約：兩端出場規則一致
────────────────────────────────────────────────────────────────────────────────
L5 把關層   tests/test_multiwindow_gate.py                               契約：L1~L3 任何變更的唯一部署入口
```

**核心設計原則（本次規劃的主張）**：

1. **每層之間用顯式參數傳遞，不用模組全域**——§1 的缺陷正是違反此原則的代價。
   全域在單行程下「看起來會動」，一旦進入多行程／多執行緒就靜默失效，而且**不會報錯**。
2. **每層都要有一個廉價的契約測試**，在 pipeline 內建執行，讓靜默失效變成大聲失敗（§3.5）。
3. **L5 是唯一部署閘門**：L1~L3 的任何變更（新因子、新特徵、新模型、新目標函式）
   一律先過多窗把關，判準統一（§4.5）。這取代目前「單窗回測 + 人工判讀」的做法。
4. **代理目標必須與 P&L 同調，但不可等於 P&L**——直接拿 `trading_sim` 當目標函式
   會把出場、regime、風控一起過擬合到單一區間，比現在更危險（§5.6）。

### 2.2 資料流（修復後）

```
best_factors.json ──┐
                    ├──► current_factor_params() ──► partial(_compute_ta, p=params)
config 預設值    ──┘        （父行程快照）              │
                                                       ▼
                                          joblib workers（顯式收到 params）
                                                       │
                                                       ▼
                                          features_combined.parquet
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          ▼                            ▼                            ▼
                   契約測試（§3.5）              train.py（L3）            多窗把關（L5）
                   欄名 == 參數？                                          候選 vs 現行
```

---

## 3. Step −1：因子參數傳遞修復

### 3.1 修法：顯式傳參（父行程快照 → 子行程接收）

**(a) 新增參數快照函式**（`feature_engineering.py`，放在 §「可由外部覆寫的全域參數」區塊之後）

```python
_FACTOR_KEYS = (
    "MA_WINDOWS", "RSI_PERIOD", "ATR_PERIOD", "KD_PERIOD",
    "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL",
    "BOLL_WINDOW", "BOLL_STD_MULT", "VOL_MA_WINDOW",
    "CHIPS_SUM_WINDOWS", "MOMENTUM_WINDOWS",
)

def current_factor_params() -> dict:
    """快照目前模組全域的因子參數。

    必須在父行程呼叫並把回傳值顯式傳給 worker——joblib 子行程會重新 import
    本模組，看不到 auto_pipeline._apply_best_params() 的 setattr 結果。
    """
    g = globals()
    return {k: g[k] for k in _FACTOR_KEYS if k in g}
```

**(b) `_compute_ta` 改為接受參數**（原本讀全域的 ~40 行改讀 `p`）

```python
def _compute_ta(g, p=None):
    p = p or current_factor_params()          # 單獨呼叫時仍可用全域，保持向後相容
    ma_windows   = p["MA_WINDOWS"]
    rsi_period   = p["RSI_PERIOD"]
    atr_period   = p["ATR_PERIOD"]
    kd_period    = p["KD_PERIOD"]
    macd_fast    = p["MACD_FAST"]
    macd_slow    = p["MACD_SLOW"]
    macd_signal  = p["MACD_SIGNAL"]
    boll_window  = p["BOLL_WINDOW"]
    boll_std     = p["BOLL_STD_MULT"]
    vol_ma       = p["VOL_MA_WINDOW"]
    # ... 以下函式主體把 MA_WINDOWS → ma_windows 等逐一替換（純機械替換，不改邏輯）
```

**(c) 呼叫端顯式傳入**（`feature_engineering.py`:583-584）

```python
from functools import partial

_p = current_factor_params()
print(f"  計算技術面特徵 (含乖離率)... (多核心運算中)")
print(f"  [生效因子] MA={_p['MA_WINDOWS']} RSI={_p['RSI_PERIOD']} ATR={_p['ATR_PERIOD']} "
      f"KD={_p['KD_PERIOD']} MACD={_p['MACD_FAST']}/{_p['MACD_SLOW']}/{_p['MACD_SIGNAL']} "
      f"Boll={_p['BOLL_WINDOW']}/{_p['BOLL_STD_MULT']} VolMA={_p['VOL_MA_WINDOW']} "
      f"Chips={_p['CHIPS_SUM_WINDOWS']}")
df = apply_parallel(df.groupby("stock_id"), partial(_compute_ta, p=_p))
```

> `[生效因子]` 這行是刻意加的**稽核輸出**：日後任何人看 log 就能確認實際生效的參數，
> 而不是只看到 `auto_pipeline` 那行「打算套用」的參數。兩行對不上就是出事了。

**(d) 為多窗把關開放輸出路徑與參數注入**（`process_all_history_features` 簽名）

```python
def process_all_history_features(start_date_obj, end_date_obj,
                                 override_target_stocks=None,
                                 factor_params=None,      # 新增：顯式指定因子（sandbox 用）
                                 out_path=None):          # 新增：輸出到指定路徑（sandbox 用）
    ...
    _p = factor_params or current_factor_params()
    ...
    output_path = out_path or os.path.join(PARENT_DIR, "data", "features", "features_combined.parquet")
```

這兩個參數是 §4 多窗把關能在 sandbox 內建候選特徵的前提，預設值維持現行行為、生產零影響。

**(e) 檢查其他 `Parallel` 路徑**
`feature_engineering.py`:566 的 `build_features(d, target_stocks)` 也走子行程。
實作時須確認它是否讀取任何因子全域（初判為否，僅組裝當日原始資料），若有則比照 (b) 處理。

### 3.2 ATR 欄名地雷（修復會踩爆，必須同批處理）

修好之後 `ATR_PERIOD=23` 會生效，特徵檔的欄位變成 `atr23_pct`，而下列位置**寫死 `atr18_pct`**：

| 位置 | 後果 |
| :--- | :--- |
| [../trading_sim.py](../trading_sim.py):620 | `'atr18_pct' in prev_data.columns` 為 False → `_atr_stop_pct = STOP_LOSS_PCT` → **ATR 動態停損靜默失效，退回固定 −8%** |
| [../scripts/inference.py](../scripts/inference.py):280,492,493,507,596 | 停損價建議全部變 NaN／退回固定值 |
| [test_short_backtest.py](test_short_backtest.py):390,410 | 同上（僅測試腳本） |

這正是 §4.5 #3「`trading_sim` 與 `inference` 共用同一套 ATR 停損」會被無聲破壞的路徑，
且**不會有任何錯誤訊息**（`if ... in columns else None` 把它吞掉了）。

**解法：導入期間無關的正規欄名。** 在 `_compute_ta` 末尾加一行：

```python
g["atr_pct"] = g[f"atr{atr_period}_pct"]   # 期間無關別名，風控引擎一律讀這欄
```

在 [../scripts/utils.py](../scripts/utils.py) 加共用取值器：

```python
def get_atr_pct_col(df) -> str | None:
    """回傳 ATR 百分比欄名。優先用期間無關別名，舊特徵檔退回 atr18_pct。"""
    if "atr_pct" in df.columns:
        return "atr_pct"
    legacy = [c for c in df.columns if c.startswith("atr") and c.endswith("_pct")]
    return legacy[0] if legacy else None
```

`trading_sim.py` 與 `inference.py` 改用它。**保留舊欄位 fallback**，讓現有特徵檔不必立刻重建。

> 替代方案（不建議）：把 `atr_period` 從 `OPTUNA_BOUNDS` 移除、鎖死 18。
> 這樣少改幾行，但等於承認「有一個因子永遠不能優化」，且沒解決 MA/RSI/KD/VolMA 的同類問題。

### 3.3 修復後的既有資產怎麼辦

| 資產 | 處置 |
| :--- | :--- |
| `configs/best_factors.json`（2026-06-21 搜出） | **不自動啟用**。它是在「TA 有效」的假設下搜出的，但從未被生產驗證過。列為 §4 的候選之一 |
| `data/features/features_combined.parquet` | 保持不動，直到候選通過把關 |
| `models/*.txt` | 同上。特徵沒換就不必重訓 |
| `models/_candidate_b2full/` | 保留作為反例紀錄，**不 promote** |

### 3.4 提供關閉開關

在 [../config.py](../config.py) 新增（放在既有因子相關區塊）：

```python
# 因子參數是否實際套用到技術指標計算。
# False = 維持 2026-08-14 修復前的行為（TA 用模組預設值，僅籌碼視窗生效），
#         供 A/B 對照與緊急回退使用。修復後預設 True。
APPLY_BEST_FACTORS_TA = True
```

`current_factor_params()` 在 `False` 時回傳模組預設值。這讓「修復前 vs 修復後」
可以在多窗把關裡當成兩個候選直接比較——**這是驗證修復是否真的有益的唯一正當方式**。

### 3.5 契約測試（新增 `tests/test_factor_params_contract.py`）

廉價、秒級、可放進 `tests/test_pipeline.py` 一起跑：

```python
def test_ta_params_survive_multiprocessing():
    """因子參數必須穿透 joblib 子行程——防止 §1 的缺陷復發。"""
    import scripts.feature_engineering as fe
    from functools import partial
    from joblib import Parallel, delayed

    p = dict(fe.current_factor_params())
    p.update({"MA_WINDOWS": [3, 8, 21, 55], "RSI_PERIOD": 11,
              "ATR_PERIOD": 9, "KD_PERIOD": 6, "VOL_MA_WINDOW": 4})

    df = _synthetic_ohlcv(n_stocks=3, n_days=300)      # 純合成資料，不碰生產檔
    out = pd.concat(Parallel(n_jobs=2)(
        delayed(partial(fe._compute_ta, p=p))(g) for _, g in df.groupby("stock_id")))

    for col in ("ma3", "ma55", "rsi11", "atr9", "atr9_pct", "k6", "vol_ma4", "atr_pct"):
        assert col in out.columns, f"因子參數未生效：缺少 {col}"
    for col in ("ma9", "rsi7", "atr18", "k15", "vol_ma8"):
        assert col not in out.columns, f"仍在使用模組預設值：出現 {col}"


def test_production_parquet_matches_best_factors():
    """生產特徵檔的欄名必須與 best_factors.json 一致（可能 xfail，直到候選上線）。"""
    # 讀 parquet schema（不載入資料）+ best_factors.json，逐項比對
    # 這支測試就是 §1.1 那張表的自動化版本
```

第二支測試是**這次缺陷本來就該被抓到的守門員**——它會讓「json 說 A、特徵是 B」直接失敗。

### 3.6 驗收條件

- [ ] `test_factor_params_contract.py` 兩支全綠
- [ ] 執行 `auto_pipeline.py -s feature` 後，log 的 `[生效因子]` 與 `[套用參數]` 完全一致
- [ ] 新產生的 parquet 欄名與 `best_factors.json` 逐項對得上
- [ ] `atr_pct` 欄存在；`trading_sim` / `inference` 在新舊兩種特徵檔上都能取到停損值
- [ ] `APPLY_BEST_FACTORS_TA=False` 時，產出的 parquet 與現行生產檔 **byte-identical**
      （證明修復沒引入其他非預期變異，方法同 `compare_model_modes.md` §8 的一致性驗證）

---

## 4. Step 0：多窗把關框架

### 4.1 為什麼必須先有

`compare_model_modes.md` §4.2 已經寫明：4 個月、每組約 30 筆平倉交易，
「單窗差異噪音足以蓋過真實效果」。目前每次實驗都要靠人工在報告裡重複這段但書，
且沒有任何機制阻止「單窗贏了就上線」。把它變成程式，判準才會被強制執行。

### 4.2 設計原則

| 原則 | 做法 |
| :--- | :--- |
| **完全隔離** | 一切寫入 `tempfile.gettempdir()/stock_gate_sandbox/`；生產 parquet 唯讀；`config.py`／`models/`／`reports/` 一個位元都不動 |
| **複用生產引擎** | monkeypatch `train.DATA_PATH/MODEL_DIR/FEATURE_COLS_PATH`、`trading_sim.DATA_PATH/MODEL_DIR`；`run_simulation(export_report=False)` |
| **單一變因** | 候選與基準共用同一截斷點、同一風控參數、同一資金與檔數 |
| **乾淨 OOS** | 所有視窗都在訓練截斷點之後 |
| **不動進場嚴格度** | `buy_threshold=None`，走 regime 動態門檻（§4.5 #1／CLAUDE.md #8） |

前四項的實作範本已經存在：[test_feature_candidate_gate.py](test_feature_candidate_gate.py)。
本框架是它的泛化版——把「候選＝特徵欄」擴充成「候選＝任意 L1~L3 變更」。

### 4.3 視窗切法

生產訓練截斷點 `BACKTEST_DATE = 20250801`，資料最新到 2026-08-13，
乾淨 OOS 約 12.5 個月：

```python
WINDOWS = [
    ("2025Q3+", "2025-08-01", "2025-10-31"),
    ("2025Q4",  "2025-11-01", "2026-01-31"),
    ("2026Q1",  "2026-02-01", "2026-04-30"),
    ("2026Q2",  "2026-05-01", "2026-07-31"),
    ("全OOS",   "2025-08-01", "2026-07-31"),   # 連續區間，不是四個子窗的複利
]
```

**「全OOS」必須是連續區間回測，不能由子窗結果複利推算**——CLAUDE.md §4.5 #5：
`MIN_HOLD_DAYS=20`、獲利靠肥尾，切割會月初空手重建倉、月底截斷未平倉，系統性低估績效。
子窗只用來看**穩健度**（勝出是否集中在少數期間），絕對績效一律看全OOS。

### 4.4 程式骨架（`tests/test_multiwindow_gate.py`）

```python
"""
test_multiwindow_gate.py — L1~L3 變更的統一多窗部署把關

用法：
  python tests/test_multiwindow_gate.py --candidate factors:configs/best_factors.json
  python tests/test_multiwindow_gate.py --candidate objective:expectancy --trials 600
  python tests/test_multiwindow_gate.py --candidate ta_fix          # §3 修復本身
  python tests/test_multiwindow_gate.py --seeds 42,7,2024           # 種子穩健度
"""

SANDBOX = os.path.join(tempfile.gettempdir(), "stock_gate_sandbox")
PROD_PARQUET = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
CAPITAL = 2_000_000        # 對齊 compare_model_modes.py 預設，便於橫向比對


# ── 候選定義：每個候選要能回答「怎麼生出特徵檔」 ──────────────────
class Candidate:
    """tag / 說明 / 產生 sandbox 特徵檔的方法"""
    def build_parquet(self) -> str: ...


class BaselineCandidate(Candidate):
    """現行：直接用生產 parquet（唯讀）"""
    def build_parquet(self):
        return PROD_PARQUET


class FactorCandidate(Candidate):
    """指定一組因子參數，重建特徵到 sandbox（需 §3(d) 的 factor_params/out_path）"""
    def __init__(self, tag, params):
        self.tag, self.params = tag, params

    def build_parquet(self):
        import scripts.feature_engineering as fe
        out = os.path.join(SANDBOX, f"features_{self.tag}.parquet")
        if os.path.exists(out):                       # 快取：重建一次約 3 分鐘
            return out
        _quiet(fe.process_all_history_features,
               START_DATE, END_DATE,
               factor_params=self.params, out_path=out)
        return out


# ── 訓練與回測：monkeypatch 到 sandbox，不碰生產 ──────────────────
def train_variant(tag, parquet_path, seed=42, cutoff=None):
    import train as T
    mdl = os.path.join(SANDBOX, f"models_{tag}_s{seed}")
    os.makedirs(mdl, exist_ok=True)
    T.DATA_PATH, T.MODEL_DIR = parquet_path, mdl
    T.FEATURE_COLS_PATH = os.path.join(mdl, "feature_cols.json")
    if cutoff is not None:
        T.BACKTEST_DATE = cutoff        # 模組全域，main() 呼叫時才讀 → monkeypatch 有效
    T.LGBM_SEED_BASE = seed             # 見 0.2：train.py:106 的 42 需提為模組變數
    _quiet(T.main)
    return mdl


def backtest(mdl_dir, parquet_path, start, end):
    import trading_sim as S
    S.DATA_PATH, S.MODEL_DIR = parquet_path, mdl_dir
    ret, dd, _ = _quiet(S.run_simulation, start, end, CAPITAL, S.MAX_POSITIONS,
                        export_report=False)     # buy_threshold 不指定 → 走 regime 動態門檻
    return ret, dd * 100.0


# ── 判決 ────────────────────────────────────────────────────────
def verdict(base, cand, sub_windows):
    """判準與 CLAUDE.md §4.5 #4 一致：全窗雙贏 且 多數子窗穩健"""
    b_ret, b_dd = base["全OOS"]
    c_ret, c_dd = cand["全OOS"]
    wins = sum(cand[w][0] > base[w][0] for w in sub_windows)
    full_win = (c_ret > b_ret) and (c_dd <= b_dd)

    if full_win and wins >= len(sub_windows) * 0.6:
        return "PASS",     "✅ 全OOS 雙贏且多數子窗穩健 → 建議採用"
    if full_win:
        return "MARGINAL", "🟡 全OOS 雙贏但子窗不穩（勝出集中於少數期間）→ 需再觀察，勿逕行上線"
    if c_ret > b_ret:
        return "FAIL",     "⚠️ 報酬升但回撤變差 → 取捨，不算勝出"
    return "FAIL",         "❌ 未通過 → 維持現行"
```

**輸出**：`reports/gate_<candidate>_<TS>.md` + 同名 `.log`，含每窗
報酬／MDD／Calmar／買賣筆數，以及跨種子的離散度。

### 4.5 判準（統一，不再逐案討論）

| 條件 | 判定 |
| :--- | :--- |
| 全OOS 報酬↑ 且 MDD 不變差 **且** ≥60% 子窗報酬勝出 | ✅ PASS，可採用 |
| 全OOS 雙贏但子窗 <60% | 🟡 MARGINAL，繼續觀察，不上線 |
| 其餘 | ❌ FAIL |

**補充守則**：

- **交易筆數 <20 的視窗標記為「不具參考性」**，不計入子窗勝出統計（避免用 3 筆交易的窗投票）。
- **跨種子檢查**：`--seeds` 至少跑 3 個 LightGBM 種子；若候選的勝負隨種子翻轉，
  一律降為 FAIL。這是最便宜的過擬合偵測（每個種子只多 0.5 分鐘訓練）。
- **判定寫進報告，不靠人工判讀**。

### 4.6 已知限制（誠實列出）

1. **子窗仍然不長**：3 個月／窗 ≈ 20~30 筆交易。這個框架**降低**噪音誤判機率，不能消除。
2. **子窗之間不獨立**：共用同一個訓練截斷點與同一個模型，四個窗的誤差是相關的。
   它檢定的是「時間穩健度」，不是統計顯著性。
3. **回測期已被人看過**：2025-08 之後的資料，作者與歷次實驗都已反覆檢視，
   嚴格說有「研究者自由度」污染。真正乾淨的驗證只有**向前的紙上驗證**。
4. **`FactorCandidate` 需要重建特徵**（約 3 分鐘／候選 + 310MB／份）。
   sandbox 要做容量控管，建議加 `--clean` 清理。

---

## 5. Step 1：Optuna 目標函式重構

### 5.1 病因：兩重度量錯位

[../scripts/optimize_factors.py](../scripts/optimize_factors.py):326

```python
hit_rate = (daily_pick[label] == 2).mean() * 100
day_scores.append(hit_rate)
```

| 錯位 | 優化器在乎的 | 策略實際靠的 |
| :--- | :--- | :--- |
| **幅度盲** | 命中率（label==2 的比例），漲 0.1% 與漲 15% 同分 | 肥尾幅度（§4.5 #1：進場勝率與獲利無關，`b1` 勝率最高卻報酬最差） |
| **期程錯位** | 1~3 日前瞻報酬（`FORECAST_DAYS=[1,2,3]`） | `MIN_HOLD_DAYS = 20`，實際持有 20 日以上 |
| **無下行成本** | 不區分「小跌」與「−12% 被 ATR 停損掃出」 | 停損是真實現金損失，且會佔用槽位 |
| **單一評估區塊** | 截斷點前 1.5 年一整塊（`optimize_factors.py`:426） | — 容易對該塊過擬合 |

`label==2` 的定義（`feature_engineering.py`:717）是
`(橫截面排名 ≥ 80%) & (next_ret > 0)`——**是個二元門檻，量級資訊在標籤生成當下就被丟棄了**。
所以目標函式再怎麼精修，都精修不出幅度。

### 5.2 三種目標定義（可切換，同一段程式）

| 名稱 | 定義 | 修掉哪個錯位 | 風險 |
| :--- | :--- | :--- | :--- |
| `precision` | 現行 Top-K 命中率 | — | 基準線，必須保留供對照 |
| `expectancy` | Top-K 的 `mean(next_ret_H)` | 幅度盲 | 低。H=3 時與現行同期程，變因單一 |
| `stop_aware` | Top-K 的「停損感知期望值」（見下） | 幅度盲＋期程＋下行成本 | 中。最貼近 P&L，但多了假設 |

**`stop_aware` 定義**：對每一檔被選中的股票，以買入日鎖定的 ATR 停損為界，
用未來 H 日的價格路徑近似真實出場：

```
stop_pct_i = clip(-ATR_STOP_MULTIPLIER × atr_pct_i, ATR_STOP_FLOOR_PCT, ATR_STOP_CEILING_PCT)

pnl_i = stop_pct_i                      若 fwd_low_min_H_i ≤ stop_pct_i   （期間內被掃出）
        fwd_ret_H_i                     否則                              （抱滿 H 日）
```

- `fwd_low_min_H` 用**盤中最低價**算，對齊 `trading_sim.py`:459 的「觸發停損(盤中)」判定。
- `H` 取 `FACTOR_OBJ_HORIZON`，建議 20（對齊 `MIN_HOLD_DAYS`）。
- `atr_pct` 用 §3.2 的正規欄名。

### 5.3 實作（`optimize_factors.py`）

**(a) 前瞻量在 `make_objective` 內預計算一次**——它們與 trial 參數無關，
每 trial 重算是純浪費（600 trials × 72 萬列）：

```python
def make_objective(df_base, bt_date, best_holder):
    ...
    H = FACTOR_OBJ_HORIZON
    if FACTOR_OBJECTIVE != "precision":
        g = df_base.sort_values(["stock_id", "date"]).groupby("stock_id")
        df_base[f"fwd_ret_{H}"] = g["close"].shift(-H) / df_base["close"] - 1
        # 未來 H 日的最低「盤中低點 vs 買價」跌幅
        df_base[f"fwd_low_{H}"] = (
            g["low"].transform(lambda s: s.shift(-H).rolling(H, min_periods=1).min())
            / df_base["close"] - 1
        )
        atr_col = get_atr_pct_col(df_base)
        raw = -ATR_STOP_MULTIPLIER * df_base[atr_col]
        df_base["stop_frac"] = raw.clip(lower=ATR_STOP_FLOOR_PCT / 100,
                                        upper=ATR_STOP_CEILING_PCT / 100)
```

> ⚠️ `fwd_low` 的 shift/rolling 組合方向極易寫反，**實作後必須用手算的小例子驗證**
> （3 檔 × 10 天的合成資料，人工核對每一格）。寫反會製造前視偏差，
> 而且因為分數會「變好」，非常難從結果察覺。這是本步驟最高風險的 3 行程式。

**(b) 評分改為可切換 + 分塊穩健化**：

```python
def _score_picks(daily_pick, label, H):
    if FACTOR_OBJECTIVE == "precision":
        return (daily_pick[label] == 2).mean() * 100
    if FACTOR_OBJECTIVE == "expectancy":
        return daily_pick[f"fwd_ret_{H}"].mean() * 100
    # stop_aware
    stopped = daily_pick[f"fwd_low_{H}"] <= daily_pick["stop_frac"]
    pnl = np.where(stopped, daily_pick["stop_frac"], daily_pick[f"fwd_ret_{H}"])
    return float(np.nanmean(pnl)) * 100


def _blocked_score(daily_pick, label, H):
    """把評估期切 N 塊各自評分，回傳 mean − λ·std。

    懲罰跨期離散度：一組只在某半年特別好的因子，不該贏過各期都穩的因子。
    這是 §4 多窗把關的精神在目標函式內的縮影。
    """
    blocks = pd.qcut(daily_pick["date"].rank(method="dense"),
                     FACTOR_OBJ_N_BLOCKS, labels=False, duplicates="drop")
    scores = [_score_picks(daily_pick[blocks == b], label, H)
              for b in sorted(pd.unique(blocks)) if (blocks == b).sum() >= 50]
    if not scores:
        return 0.0
    return float(np.mean(scores) - FACTOR_OBJ_DISPERSION_PENALTY * np.std(scores))
```

**(c) `best_factors.json` 記錄目標函式**（否則日後無法分辨兩次搜尋的分數可不可比）：

```python
result = {
    ...,
    "objective":         FACTOR_OBJECTIVE,
    "objective_horizon": FACTOR_OBJ_HORIZON,
    "objective_topk":    FACTOR_OBJ_TOPK,
    "best_score_avg":    best_score,
}
```

> **不同目標函式的 `best_score_avg` 絕對不可互相比較**。
> 這次 log 的「21.93% → 22.19%」看起來像進步，其實兩者評估集不同（截斷 20250801 vs 20260331），
> 本來就不可比。加上 `objective` 欄位後，報告可以自動擋掉這種誤讀。

### 5.4 新增 config 常數（[../config.py](../config.py)，遵守常數集中原則）

```python
# ── 因子最佳化目標函式（§ tests/FACTOR_OBJECTIVE_PLAN.md）────────────
# "precision"  Top-K 命中率（2026-08 之前的行為，保留供對照）
# "expectancy" Top-K 平均前瞻報酬——讓肥尾幅度進入目標
# "stop_aware" 停損感知期望值——再計入 ATR 停損造成的左尾截斷
FACTOR_OBJECTIVE            = "precision"
FACTOR_OBJ_HORIZON          = 20     # 前瞻天數；建議對齊 MIN_HOLD_DAYS
FACTOR_OBJ_TOPK             = 20     # 每日取前 K 檔評分
FACTOR_OBJ_N_BLOCKS         = 4      # 評估期切幾塊做穩健度懲罰
FACTOR_OBJ_DISPERSION_PENALTY = 0.5  # score = mean − λ × std(blocks)
```

預設維持 `precision`，**修改預設值本身就是一次需要過 §4 把關的變更**。

順帶修掉一個既有的隱性寫死（`optimize_factors.py`:317）：

```python
top_k_num = 3 if ... < 50 else 20     # ← 魔術數字，改讀 FACTOR_OBJ_TOPK
```

### 5.5 實驗矩陣

修復（§3）與把關（§4）就緒後，用同一組視窗跑：

| 候選 | 目標函式 | H | 說明 |
| :--- | :--- | :--- | :--- |
| `baseline` | — | — | 現行生產（TA 參數未生效的狀態） |
| `ta_fix` | precision | 3 | **只修 §1 缺陷**，目標函式不動 → 分離「修復本身」的影響 |
| `obj_exp` | expectancy | 3 | 只換幅度，期程不變 → 分離「幅度」的影響 |
| `obj_exp20` | expectancy | 20 | 幅度 + 期程對齊 |
| `obj_stop20` | stop_aware | 20 | 全套 |

**必須這樣拆**，否則會重蹈 `b2full` 的覆轍：一次動三個變因，贏了也不知道是誰的功勞
（`compare_model_modes.md` §4.2 第 2 點）。每個候選都要重跑 optimize（~1.7 小時）
+ feature（~3 分）+ train（~0.5 分）+ 5 窗回測（~2.5 分）≈ 2 小時，
五個候選一夜跑完。

### 5.6 明確不做的事

- **不把 `trading_sim.run_simulation` 當目標函式**。600 trials 直接優化回測損益
  ＝把出場、regime、風控一起過擬合到單一區間，比現行的錯位更危險。
  代理目標要**與 P&L 同調**，不是**等於 P&L**。
- **不改 `FORECAST_DAYS` / label 定義**。改標籤期程會連動 `train.py`、`inference.py`、
  `trading_sim.py` 的 Day1/Day2/Day3 分數語意，是另一個數量級的變更。
  本步驟只改「用什麼尺去評價因子」，不改模型預測什麼。
- **不動進場門檻**。§4.5 #1 已驗證會崩潰。

---

## 6. 如何接回現行系統

### 6.1 部署路徑

```
候選通過 §4 把關（PASS）
   │
   ├─► configs/best_factors.json           ← 新因子參數
   ├─► auto_pipeline.py -s feature         ← 重建生產特徵（3 分鐘）
   ├─► auto_pipeline.py -s train           ← 重訓模型（0.5 分鐘）
   └─► 契約測試 test_factor_params_contract ← 確認欄名與 json 一致
   │
   ▼
生產 inference.py / trading_sim.py 自動使用新特徵（無需改動）
   │
   ▼
紙上驗證（run_daily.ps1 + record_paper_trades.py）觀察 4~8 週後才調整實盤部位
```

### 6.2 需要同步更新的文件

| 文件 | 更新內容 |
| :--- | :--- |
| [../.claude/CLAUDE.md](../.claude/CLAUDE.md) §4.5 #6 | 加註「該實驗的 TA 因子實際未生效，結論範圍需重讀」 |
| [../scripts/compare_model_modes.md](../scripts/compare_model_modes.md) §7 | 補第二輪 `--full` 結果，並註記變因實為 CHIPS_SUM_WINDOWS + 訓練窗 |
| [../.claude/CLAUDE.md](../.claude/CLAUDE.md) §4.5 | 新增一條不變量：「L1~L3 變更一律走 `test_multiwindow_gate.py`」 |
| [../README.md](../README.md) | 若因子生效改變生產行為，更新操作說明（**不得寫入績效數字**，CLAUDE.md #9） |

### 6.3 回退方案

| 情境 | 回退 |
| :--- | :--- |
| 修復後績效變差 | `APPLY_BEST_FACTORS_TA = False`（§3.4），行為回到 2026-08-14 之前 |
| 新目標函式搜出的因子變差 | `FACTOR_OBJECTIVE = "precision"` + 還原舊 `best_factors.json` + 重建特徵 |
| 特徵檔／模型不匹配 | `compare_model_modes.py --promote <tag>`（會一併還原特徵檔與因子檔） |

**新舊特徵檔必須成對更換**——只換模型不換特徵會讓推論靜默失真
（`compare_model_modes.md` §6 已記載此陷阱）。

---

## 7. 禁止事項（已驗證否決，勿重做）

摘自 [../.claude/CLAUDE.md](../.claude/CLAUDE.md) §4.5 與各測試檔，本規劃**全程不得觸碰**：

| 禁止 | 原因 | 出處 |
| :--- | :--- | :--- |
| 拉高 `buy_threshold` / 收緊進場來減少停損 | 已實測 +59% → −9%，肥尾贏家被一起砍掉 | §4.5 #1 |
| 均勻降風險（縮 ATR 部位、無差別分散） | 等比例砍報酬，回撤與報酬不可分割 | §4.5 #2 |
| 法人淨額做成交量正規化 | 全 OOS +224% → +97%，量級本身帶訊號 | `test_chip_flow_normalization.py` |
| 自營商改用「自行買賣」淨額 | 平手，不值得多維護一份資料相依 | 同上 |
| fortune z-score／日曆特徵 | 僅 1/4 子窗勝出，疑過擬合 | `test_feature_candidate_gate.py` |
| Learning-to-Rank（lambdarank） | 敗給現行 multiclass | 2026-07-11 實驗（腳本已不在工作區，結論僅存於專案記憶） |
| 個股放空／反向 ETF | 真OOS 放空側全面虧損，結構性瓶頸 | [SHORT_STRATEGY_PLAN.md](SHORT_STRATEGY_PLAN.md) |
| 按月切割評估績效 | 系統性低估（月初空手、月底截斷肥尾） | §4.5 #5 |
| 再靠「重訓模型讓資料更新」救績效 | b1/b2/b2full 三度雙輸 | §4.5 #6 |
| `BACKTEST_DATE=None` 下跑 `optimize_factors` | 防洩漏保護失效，回測區間進入目標函式＝lookahead | `compare_model_modes.md` §3 |

另外三條**方法論禁令**（本規劃自身必須遵守）：

- **不得用單窗結果下部署決定**——即使贏很多。
- **不得比較不同目標函式的分數**——`best_score_avg` 跨目標無意義。
- **sandbox 掃描只能產生假設，部署決策一律以生產引擎複驗**
  （`SHORT_STRATEGY_PLAN.md` §4.3 有前車之鑑：sandbox 說 regime 窗口 10 最差，
  生產引擎複驗後窗口 10 實為壓倒性最佳）。

---

## 8. 風險登記表

| # | 風險 | 機率 | 影響 | 緩解 |
| :-: | :--- | :--: | :--- | :--- |
| R1 | `fwd_low` 的 shift/rolling 寫反 → 前視偏差 | 中 | **極高**（分數變好，難察覺） | 合成資料手算逐格驗證；分數異常高（>2×baseline）視為紅旗 |
| R2 | 修復後 ATR 欄名改變 → 停損靜默失效 | **高**（若不處理必然發生） | 高 | §3.2 正規欄名 + fallback；把關報告須列出停損觸發次數，為 0 即異常 |
| R3 | 修復後特徵全變 → 現行風控參數不再適用 | 中 | 中 | 把關通過後重跑 `optimize_trading_params.py`，並再過一次把關 |
| R4 | sandbox 特徵檔佔滿磁碟（310MB × N） | 中 | 低 | 快取複用 + `--clean`；候選數控制在 5 個內 |
| R5 | 五個候選一次全跑，結果互相污染 | 低 | 中 | 每候選獨立 sandbox 子目錄；`compare_model_modes.py` 的 manifest 機制可借鑑 |
| R6 | 修復本身讓績效變差，但已經改了很多程式 | 中 | 低 | §3.4 開關可一鍵回退；程式修復與參數啟用是兩件事 |
| R7 | 20 日前瞻 + 每日取樣 → 樣本重疊嚴重，有效樣本數遠低於名目 | **高** | 中 | 不對小差異下結論；分塊評分（§5.3b）已部分緩解；最終判定以 §4 把關為準 |
| R8 | 研究者自由度：同一段 OOS 被反覆使用 | 高 | 中 | 誠實記載於報告；最終仍以向前紙上驗證為準 |

---

## 9. 執行順序與檢查點

```
Step −1  修復（半天）  ── ✅ 已完成 2026-08-14（程式修復；尚未重建生產特徵）
  −1.1  _compute_ta 顯式傳參 + current_factor_params()
  −1.2  ATR 正規欄名 atr_pct + get_atr_pct_col()（trading_sim / inference 同步）
  −1.3  process_all_history_features 加 factor_params / out_path
  −1.4  config: APPLY_BEST_FACTORS_TA
  −1.5  tests/test_factor_params_contract.py
  ✅ 檢查點（2026-08-14 實測結果）：
       · 契約測試 1/2 綠燈；測試 3 依設計標紅（生產特徵檔仍是修復前的舊檔）
       · 預設值路徑等價：22 個 TA 欄 × 4,815 列與現行生產特徵檔逐格 bit-exact
       · 回測回歸：2026-04-01~07-31 得 +37.68% / −15.75% / 30 筆，與修復前完全相同
       · tests/test_pipeline.py 18/18、tests/test_inference_sim_consistency.py 4/4 通過
  ⏸ 未做（刻意）：重建生產特徵矩陣。重建等於一次啟用 10 個沉睡因子參數，
       屬 §1.4 所述的重大變更，須先有 Step 0 的把關框架

Step 0  把關框架（1 天）
  0.1   tests/test_multiwindow_gate.py 骨架 + Candidate 抽象
  0.2   train.py:106 的 random_state=42+days_ahead → LGBM_SEED_BASE+days_ahead（供 --seeds）
  0.3   用「現行 vs 現行」空跑一次驗證框架本身（兩邊數字必須完全相同）
  ✅ 檢查點：空跑對照零差異；報告格式可讀；全程未寫入生產目錄

Step 1  目標函式（1 天 + 一夜跑批）
  1.1   config 新增 5 個常數 + optimize_factors 讀取
  1.2   前瞻量預計算（R1 逐格驗證後才繼續）
  1.3   _score_picks / _blocked_score
  1.4   best_factors.json 記錄 objective 欄位
  1.5   跑 §5.5 五候選矩陣（夜間）
  ✅ 檢查點：每候選的把關判定 + 跨種子一致性

決策點
  任一候選 PASS → §6.1 部署路徑 → 紙上驗證 4~8 週
  全部 FAIL     → 記錄於本文件 §10，收束此線，轉向 §4.5 #4 未試方向（法人分歧度）
```

**Step 0 的 0.3「現行 vs 現行空跑」是最容易被跳過、卻最重要的一步**——
它證明框架本身沒有引入變異。`compare_model_modes.md` §8 做過同性質驗證並因此建立了信任，
本框架必須比照辦理。

---

## 10. 結果紀錄（待填）

> 實驗完成後在此追記，格式比照 [SHORT_STRATEGY_PLAN.md](SHORT_STRATEGY_PLAN.md) §1：
> 只記結論與決定性證據，不記過程。若全數 FAIL，明確寫下「此線關閉」與重啟前提條件。

### 10.1 `tafix` 單窗初測（2026-08-14，尚未經多窗把關）

修復完成後，用 [../scripts/compare_model_modes.py](../scripts/compare_model_modes.py) 新增的
`tafix` 候選做單一變因對照：訓練設定（`BACKTEST_DATE=20250801`、切分 0.70/0.80）與現行完全相同，
只重建特徵與模型，**唯一變因＝best_factors.json 的技術指標參數是否真的進入特徵矩陣**。
回測 2026-04-01~07-31、200 萬、5 檔。

| 指標 | `incumbent` | `tafix` |
| :--- | ---: | ---: |
| 區間報酬 (%) | **+37.68** | +24.07 |
| 最大回撤 (%) | **−15.75** | −24.59 |
| Calmar | **2.39** | 0.98 |
| 買/賣筆數 | 30/30 | 31/31 |
| 賣出勝率 (%) | 33 | **35** |

**判定：❌ 雙輸 → 不採用，`APPLY_BEST_FACTORS_TA` 維持 `False`。**

三點觀察：

1. **修復本身確認生效**：log 的 `[生效因子]` 與 `[套用參數]` 首次一致
   （`MA=[4,19,43,95] RSI=8 ATR=23 KD=5 MACD=16/23/8 Boll=14/2.16 VolMA=6`），
   訓練切分與現行完全相同（訓練集 2020-01-02 ~ 2023-11-24），確認訓練窗不是變因。
2. **再次印證 §4.5 #1**：`tafix` 賣出勝率最高（35%）卻報酬最低。而它用的因子正是 Optuna
   以「Top-K 命中率」為目標搜出來的——**優化器要什麼就得到什麼：命中率上去了，錢少了**。
   這是 §5 度量錯位論點的直接證據。亦即這些參數「啟用後更差」，很可能不是參數壞，
   而是**它們被一把錯的尺選出來的**。
3. **單窗噪音極大**：首輪因 `atr_pct` 別名誤入訓練特徵（與 `atr23_pct` 完全共線的重複欄），
   得到 +36.48% / −23.12%；移除該重複欄後變 +24.07% / −24.59%。**僅僅一個重複特徵欄就讓報酬差 12pp**
   ——足以說明單窗 30 筆交易對微擾動有多敏感，也說明「因子啟用有害」這個結論本身同樣不可靠。
   （該重複欄已由 `config.EXCLUDE_FEATURES = ["atr_pct"]` 排除。）

> 因此本輪的正確結論是：**「在目前這把尺選出的因子下，啟用沒有好處」**，
> 而不是「因子參數生效有害」。要分辨這兩者，必須先做 §5 的目標函式重構，再重跑本對照。

### 10.2 多窗把關結果（待填）

| 候選 | 全OOS 報酬 | 全OOS MDD | 子窗勝出 | 跨種子一致 | 判定 |
| :--- | ---: | ---: | :-: | :-: | :--- |
| `baseline` | — | — | — | — | 基準 |
| `tafix` | | | | | |
| `obj_exp` | | | | | |
| `obj_exp20` | | | | | |
| `obj_stop20` | | | | | |

---

## 11. 相關文件

- [../.claude/CLAUDE.md](../.claude/CLAUDE.md) §4.5 — 策略不變量（禁止事項的權威來源）
- [../scripts/compare_model_modes.md](../scripts/compare_model_modes.md) — 模型時效性實驗；§4 的判讀方法論、§6 的環境還原機制均為本規劃借鑑對象
- [test_feature_candidate_gate.py](test_feature_candidate_gate.py) — §4 隔離 sandbox 的實作範本
- [SHORT_STRATEGY_PLAN.md](SHORT_STRATEGY_PLAN.md) — 規劃文件格式範本；§4.3 的方法論教訓（恆等式陷阱、sandbox 須生產複驗）適用於本規劃
- [../scripts/EXPERIMENTS_PENDING.md](../scripts/EXPERIMENTS_PENDING.md) — 其他待辦／已結案實驗方向
- [../reports/compare_model_modes_20260813_224948.log](../reports/compare_model_modes_20260813_224948.log) — 本規劃的起點
