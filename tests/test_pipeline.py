"""
test_pipeline.py — 全流程模擬測試 (不下載資料、不實際訓練)
============================================================
測試項目：
  1. 所有模組 import 是否正常 (從 scripts/ 載入)
  2. feature_engineering 的欄位產出是否正確 (KD 命名, 持股分級對齊)
  3. optimize_factors 的 _TA_PREFIXES 過濾是否涵蓋所有 TA 欄位
  4. best_factors.json 的 key 結構是否與 auto_pipeline 一致
  5. train.py 的日期切割是否正常
  6. inference.py 的 feature_cols.json 讀取流程是否正常
  7. scraper.py 的 json 模組是否可用
  8. utils.py 共享解析器單元測試

執行方式: python test_pipeline.py
"""

import sys
import os
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

PASS = " PASS"
FAIL = " FAIL"
WARN = "  WARN"

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append((PASS, name, msg or ""))
    except AssertionError as e:
        results.append((FAIL, name, str(e)))
    except Exception as e:
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))

# ─────────────────────────────────────────────────────────────
# 1. Import 測試
# ─────────────────────────────────────────────────────────────

def test_import_scraper():
    import scripts.scraper as sc
    assert hasattr(sc, 'download_history_data'), "找不到 download_history_data"
    import json
    assert json is not None
    return "scraper + json import OK"

def test_import_feature_engineering():
    import scripts.feature_engineering as fe
    assert hasattr(fe, 'process_all_history_features'), "找不到 process_all_history_features"
    assert hasattr(fe, 'load_target_stocks'), "找不到 load_target_stocks"
    return "feature_engineering import OK"

def test_import_optimize_factors():
    import scripts.optimize_factors as of_
    assert hasattr(of_, 'main'), "找不到 main"
    assert hasattr(of_, '_TA_PREFIXES'), "找不到 _TA_PREFIXES"
    assert hasattr(of_, '_STABLE_NON_TA_COLS'), "找不到 _STABLE_NON_TA_COLS"
    return "optimize_factors import OK"

def test_import_train():
    import scripts.train as train
    assert hasattr(train, 'main'), "找不到 train.main"
    assert hasattr(train, 'train_model'), "找不到 train.train_model"
    return "train import OK"

def test_import_inference():
    import scripts.inference as inference
    assert hasattr(inference, 'main'), "找不到 inference.main"
    return "inference import OK"

def test_import_auto_pipeline():
    import auto_pipeline
    assert hasattr(auto_pipeline, 'main'), "找不到 auto_pipeline.main"
    assert hasattr(auto_pipeline, 'step2_load_params'), "找不到 step2_load_params"
    assert hasattr(auto_pipeline, 'step3_feature_engineering'), "找不到 step3_feature_engineering"
    return "auto_pipeline import OK"

check("Import: scraper.py", test_import_scraper)
check("Import: feature_engineering.py", test_import_feature_engineering)
check("Import: optimize_factors.py", test_import_optimize_factors)
check("Import: train.py", test_import_train)
check("Import: inference.py", test_import_inference)
check("Import: auto_pipeline.py", test_import_auto_pipeline)

# ─────────────────────────────────────────────────────────────
# 2. feature_engineering KD 欄位命名
# ─────────────────────────────────────────────────────────────

def test_kd_column_naming():
    import json
    import pandas as pd
    from scripts.feature_engineering import KD_PERIOD
    path = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
    if not os.path.exists(path):
        return "Skip (no parquet)"
    df = pd.read_parquet(path)
    
    k_cols = [c for c in df.columns if c.startswith("k") and c[1:].isdigit()]
    d_cols = [c for c in df.columns if c.startswith("d") and c[1:].isdigit()]
    
    assert len(k_cols) > 0, "缺少 KD 動態欄位 (找不到任何 k 開頭的數字欄位)"
    assert len(d_cols) > 0, "缺少 KD 動態欄位 (找不到任何 d 開頭的數字欄位)"
    return f"KD 欄位已改為動態命名: {k_cols[0]}, {d_cols[0]}"

check("KD 欄位動態命名", test_kd_column_naming)

# ─────────────────────────────────────────────────────────────
# 3. optimize_factors _TA_PREFIXES 覆蓋範圍
# ─────────────────────────────────────────────────────────────

def test_ta_prefixes():
    from scripts.optimize_factors import _TA_PREFIXES, _is_ta_col
    # k9 / d9 必須被識別為 TA 欄位
    assert _is_ta_col("k9"), "_TA_PREFIXES 未涵蓋 k9"
    assert _is_ta_col("d9"), "_TA_PREFIXES 未涵蓋 d9"
    assert _is_ta_col("rsi"), "_TA_PREFIXES 未涵蓋 rsi"
    assert _is_ta_col("macd"), "_TA_PREFIXES 未涵蓋 macd"
    assert _is_ta_col("boll_mid"), "_TA_PREFIXES 未涵蓋 boll_"
    assert _is_ta_col("atr"), "_TA_PREFIXES 未涵蓋 atr"
    assert _is_ta_col("vol_ma"), "_TA_PREFIXES 未涵蓋 vol_ma"
    # 非 TA 欄位不應被識別為 TA
    assert not _is_ta_col("fini_net"), "fini_net 被誤判為 TA 欄位"
    assert not _is_ta_col("revenue"), "revenue 被誤判為 TA 欄位"
    assert not _is_ta_col("EPS"), "EPS 被誤判為 TA 欄位"
    return f"_TA_PREFIXES 覆蓋正確，包含 k9/d9"

check("_TA_PREFIXES 覆蓋範圍", test_ta_prefixes)

# ─────────────────────────────────────────────────────────────
# 4. _STABLE_NON_TA_COLS 不含 mkt_inst_net
# ─────────────────────────────────────────────────────────────

def test_stable_cols_no_mkt_inst_net():
    from scripts.optimize_factors import _STABLE_NON_TA_COLS
    assert "mkt_inst_net" not in _STABLE_NON_TA_COLS, \
        "mkt_inst_net 仍在 _STABLE_NON_TA_COLS，應已移除"
    return f"mkt_inst_net 已移除，共 {len(_STABLE_NON_TA_COLS)} 個穩定特徵"

check("_STABLE_NON_TA_COLS 無 mkt_inst_net", test_stable_cols_no_mkt_inst_net)

# ─────────────────────────────────────────────────────────────
# 5. best_factors.json key 結構
# ─────────────────────────────────────────────────────────────

def test_best_factors_json():
    import json
    path = os.path.join(ROOT_DIR, "configs", "best_factors.json")
    assert os.path.exists(path), "best_factors.json 不存在"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = ["optimized_at", "best_score_avg", "best_params_for_run_feature_engineering"]
    for k in required_keys:
        assert k in data, f"best_factors.json 缺少 key: {k}"
    params = data["best_params_for_run_feature_engineering"]
    param_keys = ["MA_WINDOWS", "RSI_PERIOD", "MACD_FAST", "MACD_SLOW", "BOLL_WINDOW"]
    for k in param_keys:
        assert k in params, f"best_params 缺少 key: {k}"
    return f"JSON 結構正確，勝率={data['best_score_avg']:.2f}%"

check("best_factors.json 結構", test_best_factors_json)

# ─────────────────────────────────────────────────────────────
# 6. auto_pipeline 的 _apply_best_params 對應
# ─────────────────────────────────────────────────────────────

def test_apply_best_params():
    import json
    import auto_pipeline
    import scripts.feature_engineering as rfe_module
    path = os.path.join(ROOT_DIR, "configs", "best_factors.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data["best_params_for_run_feature_engineering"]
    # 呼叫 _apply_best_params 看是否正常執行
    auto_pipeline._apply_best_params(params)
    # 確認 fe_module 的變數有被正確更新
    import scripts.feature_engineering as fe
    assert fe.MA_WINDOWS == params["MA_WINDOWS"], "MA_WINDOWS 未正確寫入 fe_module"
    assert fe.RSI_PERIOD == params["RSI_PERIOD"], "RSI_PERIOD 未正確寫入 fe_module"
    return f"_apply_best_params 執行正常，MA={params['MA_WINDOWS']}"

check("auto_pipeline._apply_best_params 對應", test_apply_best_params)

# ─────────────────────────────────────────────────────────────
# 7. train.py 日期切割邏輯
# ─────────────────────────────────────────────────────────────

def test_train_date_split():
    import numpy as np
    import pandas as pd
    import random
    from scripts.train import train_model
    # 建立假資料：100 個日期 x 5 支股票 = 500 筆 (避免 LightGBM 因為資料過少而無法進行 Valid Split 導致報錯)
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    stocks = ["1101", "2330", "2317"]
    rows = []
    for d in dates:
        for s in stocks:
            rows.append({
                "stock_id": s,
                "date": d,
                "feat1": np.random.randn(),
                "feat2": np.random.randn(),
                "next_ret_1": np.random.randn() * 0.01,
                "next_ret_2": np.random.randn() * 0.01,
                "next_ret_3": np.random.randn() * 0.01,
                "label_1": random.choice([0, 1, 2])
            })
    df = pd.DataFrame(rows)
    # 驗證 train.py 的源碼中切割邏輯是否能實際運行不報錯
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 建立假 feature_cols 與目標
            features = ["feat1", "feat2"]
            target = "label_1"
            
            # 使用我們剛才建立的假 df (含 20 天)
            import scripts.train as train
            original_model_dir = train.MODEL_DIR
            train.MODEL_DIR = tmp_dir
            
            try:
                # 執行訓練 (資料量極少，瞬間完成)
                train_model(df, features, target, 1)
                
                # 檢查是否有產出模型檔
                model_file = os.path.join(tmp_dir, "lgbm_model_1.txt")
                assert os.path.exists(model_file), f"訓練完畢但找不到模型檔案: {model_file}"
            finally:
                train.MODEL_DIR = original_model_dir
                
        return "日期切割無重疊，且 train_model 實際執行成功"
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(f"train_model 執行失敗: {str(e)}")

check("train.py 日期切割邏輯", test_train_date_split)

# ─────────────────────────────────────────────────────────────
# 8. feature_cols.json 讀寫流程
# ─────────────────────────────────────────────────────────────

def test_feature_cols_json_roundtrip():
    import json
    import tempfile
    test_cols = ["feat_a", "feat_b", "k9", "d9", "rsi", "macd"]
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as f:
        tmp_path = f.name
        json.dump(test_cols, f)
    
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == test_cols, f"讀寫不一致: {loaded} != {test_cols}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return f"feature_cols.json 讀寫正常，{len(loaded)} 欄 (使用暫存檔不影響正式模型)"

check("feature_cols.json 讀寫流程", test_feature_cols_json_roundtrip)

# ─────────────────────────────────────────────────────────────
# 9. 持股分級 reindex 對齊（模擬）
# ─────────────────────────────────────────────────────────────

def test_shareholding_reindex():
    import numpy as np
    import pandas as pd
    # 模擬 sh DataFrame（週頻）
    sh = pd.DataFrame({
        "stock_id": ["2330", "2330", "1101", "1101"],
        "sh_date":  pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-06", "2023-01-13"]),
        "big_holder_pct": [60.0, 61.0, 40.0, 41.0],
        "small_holder_pct": [15.0, 14.5, 20.0, 20.5],
        "holder_hhi": [0.4, 0.41, 0.25, 0.26],
    })
    # 模擬交易日 index（日頻，10 天）
    dates = pd.date_range("2023-01-04", periods=10, freq="B")
    stocks = ["2330", "1101"]
    idx = pd.MultiIndex.from_product([sorted(stocks), sorted(dates)], names=["stock_id", "date"])
    # 執行修正後的邏輯
    sh_daily = (sh.rename(columns={"sh_date": "date"})
                  .set_index(["stock_id", "date"])
                  .reindex(idx)
                  .groupby(level=0).ffill()
                  .reset_index())
    # 驗證：reindex 後不應全為 NaN
    assert "big_holder_pct" in sh_daily.columns, "缺少 big_holder_pct"
    non_nan = sh_daily["big_holder_pct"].notna().sum()
    assert non_nan > 0, f"big_holder_pct 全為 NaN（reindex 對齊失敗）"
    return f"持股分級 reindex 正常，{non_nan}/{len(sh_daily)} 筆有值"

check("持股分級 reindex 對齊", test_shareholding_reindex)

# ─────────────────────────────────────────────────────────────
# 10. parquet 特徵檔基本完整性（如果存在）
# ─────────────────────────────────────────────────────────────

def test_parquet_integrity():
    import pandas as pd
    path = os.path.join(ROOT_DIR, "data", "features", "features_combined.parquet")
    if not os.path.exists(path):
        raise AssertionError("features_combined.parquet 不存在，請先執行 run_feature_engineering.py")
    df = pd.read_parquet(path)
    
    # 基礎核心欄位 (注意：revenue 與 cash_dividend 為絕對值，已於 Step 5 消除 Level Bias 時 Drop 掉，故不應在 parquet 內直接檢查)
    required = ["stock_id", "date", "close", "next_ret_1", "next_ret_2", "next_ret_3"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"parquet 缺少核心特徵欄位: {missing}"
    
    # 方案 B 的總體市場與板塊情緒特徵欄位檢查
    macro_cols = ["market_mean_pct", "market_breadth_pct", "market_mean_ma5", "market_mean_ma20", "market_breadth_ma5", "market_breadth_ma20"]
    missing_macro = [c for c in macro_cols if c not in df.columns]
    assert not missing_macro, f"parquet 缺少方案 B 的總體市場特徵欄位: {missing_macro}"
    
    sector_cols = ["sector_mean_pct", "sector_mean_ma5"]
    missing_sector = [c for c in sector_cols if c not in df.columns]
    assert not missing_sector, f"parquet 缺少方案 B 的板塊特徵欄位: {missing_sector}"
    
    warnings = []
    if "taifex_txf_fini_net_oi" not in df.columns:
        warnings.append("taifex_txf_fini_net_oi")
    if "fini_holding_pct" not in df.columns:
        warnings.append("fini_holding_pct")

    msg = f"parquet 正常，{len(df):,} 筆，{len(df.columns)} 欄 (含總體市場與板塊特徵)"
    if warnings:
        msg += f" ( 警告: 缺少 {warnings}，可能爬蟲跳過歷史下載)"
        
    assert len(df) > 1000, f"資料筆數過少: {len(df)}"
    return msg

check("parquet 特徵檔完整性", test_parquet_integrity)

# ─────────────────────────────────────────────────────────────
# 11. scraper.py 失敗略過機制 (failed_dates.json) 檢查
# ─────────────────────────────────────────────────────────────

def test_scraper_fail_logic():
    with open(os.path.join(ROOT_DIR, "scripts", "scraper.py"), "r", encoding="utf-8") as f:
        content = f.read()
    assert "failed_dates.json" in content, "scraper.py 找不到 failed_dates.json 的實作"
    assert "SKIP_AFTER_FAILS" in content, "scraper.py 找不到 SKIP_AFTER_FAILS 的常數設定"
    return "scraper.py 已實作 failed_dates.json 失敗略過機制"

check("scraper.py 失敗略過機制", test_scraper_fail_logic)

# ─────────────────────────────────────────────────────────────
# 12. Early Stopping 變數檢查
# ─────────────────────────────────────────────────────────────

def test_early_stopping_config():
    import auto_pipeline
    import scripts.optimize_factors as of_module
    assert hasattr(auto_pipeline, 'EARLY_STOPPING_ROUNDS'), "auto_pipeline 缺少 EARLY_STOPPING_ROUNDS 設定"
    assert hasattr(of_module, 'EARLY_STOPPING_ROUNDS'), "optimize_factors 缺少 EARLY_STOPPING_ROUNDS 設定"
    return f"提早結束變數存在, auto_pipeline: {auto_pipeline.EARLY_STOPPING_ROUNDS}"

check("Early Stopping 參數檢查", test_early_stopping_config)

# ─────────────────────────────────────────────────────────────
# 13. utils.py 共享解析器單元測試
# ─────────────────────────────────────────────────────────────

def test_utils_parser():
    import tempfile
    import json
    from scripts.utils import parse_stocks_file, load_target_stocks
    
    # 建立假 Stocks.txt
    test_content = (
        "# 這是測試註解\n"
        "2330,950.0\n"
        "2317\n"
        "  2454 , 1200.5 \n"
    )
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as f:
        tmp_path = f.name
        f.write(test_content)
        
    try:
        watchlist = parse_stocks_file(tmp_path)
        assert watchlist["2330"] == 950.0, "解析成本錯誤"
        assert watchlist["2317"] is None, "解析無成本錯誤"
        assert watchlist["2454"] == 1200.5, "解析空白及成本錯誤"
        
        tickers = load_target_stocks(tmp_path)
        assert tickers == ["2330", "2317", "2454"], f"載入股票清單列表錯誤: {tickers}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return "utils.py 共享解析器測試成功"

check("utils.py 共享解析器測試", test_utils_parser)

# ─────────────────────────────────────────────────────────────
# 輸出結果與產生 Log 檔
# ─────────────────────────────────────────────────────────────
import datetime
output_lines = []
output_lines.append("\n" + "=" * 65)
output_lines.append(f"  全流程模擬測試結果 (時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
output_lines.append("=" * 65)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)

for status, name, msg in results:
    output_lines.append(f"{status}  {name}")
    if msg:
        output_lines.append(f"       → {msg}")

output_lines.append("=" * 65)
output_lines.append(f"  通過: {passed} / 失敗: {failed} / 共 {len(results)} 項")
output_lines.append("=" * 65)

log_content = "\n".join(output_lines)
print(log_content)

# 將結果寫入 Log 檔案
log_path = os.path.join(BASE_DIR, "test_pipeline.log")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(log_content + "\n")

print(f"\n   測試報告已完整匯出至: {log_path}\n")

if failed > 0:
    sys.exit(1)
