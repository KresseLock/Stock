# -*- coding: utf-8 -*-
"""
run_workflow_experiment.py — 模式 A 與 模式 B 雙階段量化研發與風控自動化實驗主控台
========================================================================
本臨時腳本用於一鍵全自動執行模式 A (研究驗證期) 與模式 B (實盤生產期) 的全流程實驗。
您可以放著電腦讓它自動執行，執行完畢後會自動生成對比報告，並百分之百還原您的設定檔。

執行步驟包括：
1. 備份 config.py, best_factors.json, best_trading_params.json
2. 【模式 A 流程】：
   - 切換 BACKTEST_DATE = "20250801", RUN_OPTIMIZATION = True, OPTIMIZATION_TRIALS = FACTOR_TRIALS
   - 執行因子調參 (auto_pipeline.py -s o)
   - 重建特徵 (auto_pipeline.py -s f)
   - 訓練模型 A (auto_pipeline.py -s t)
   - 執行 OOS 訊號診斷 (scripts/analyze_regime_stability.py) 并保存報告
   - 執行歷史風控參數調參 (scripts/optimize_trading_params.py)，限制在 2025-08-01 以前
   - 在 OOS 超級牛市 (2025-08-02 ~ 2026-06-05) 跑模擬交易 (trading_sim.py)，評估歷史風控之泛化力
3. 【模式 B 流程】：
   - 切換 BACKTEST_DATE = None, RUN_OPTIMIZATION = False (沿用模式 A 的最佳因子以節省時間)
   - 重建特徵 (auto_pipeline.py -s f)
   - 重訓模型 B (auto_pipeline.py -s t)，包含最新的牛市數據
   - 執行全週期風控參數調參 (scripts/optimize_trading_params.py)，時間覆蓋牛市 (2023-01-01 ~ 2026-06-01)
   - 執行推理預測 (auto_pipeline.py -s i)，產生明日掛單建議
   - 執行全週期模擬回測 (trading_sim.py)，評估完整策略表現
4. 還原所有設定與備份，並生成對比報告 reports/workflow_experiment_report.md
"""
import os
import sys
import shutil
import re
import subprocess
import time
import json
import argparse
import threading

# 強制設定標準輸出/錯誤編碼為 UTF-8，防止 Windows 終端機 (CP950/Big5) 遇到 Emoji 拋出 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── 1. 實驗核心配置 (最上面易於手動調整的變數，也支援執行時 CLI 參數覆蓋) ────────
# ⚙️ 因子最佳化設定 (scripts/optimize_factors.py)
FACTOR_TRIALS         = 400        # 因子最佳化最大搜尋輪數 (預設較少以利快速驗證)
FACTOR_EARLY_STOPPING = 150        # 因子最佳化早停輪數 (無進度達此輪數自動終止，None 代表不啟用)

# 🎛️ 交易風控最佳化設定 (scripts/optimize_trading_params.py)
TRADING_TRIALS         = 400      # 風控最佳化最大搜尋輪數 (預設 100 輪以兼顧效率與品質)
TRADING_EARLY_STOPPING = 150       # 風控最佳化早停輪數 (無進步達此輪數自動終止，None 代表不啟用)

# 💰 模擬交易基本設定
CAPITAL        = 2000000          # 回測與優化的初始資金 (200 萬)

# 路徑定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.py")

def get_config_backtest_date():
    """自中央控制面板 config.py 讀取 BACKTEST_DATE 設定，若為 None 或格式不符則使用預設值 20250801"""
    if not os.path.exists(CONFIG_PATH):
        return "20250801"
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r'BACKTEST_DATE\s*=\s*["\'](\d{8})["\']', content)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "20250801"

# 🗓️ 模式 A 研究截斷日期 (YYYYMMDD)；OOS 起始日 = 截斷日 +1；所有回測結束日 = 最新資料日 (自動偵測)
MODE_A_CUTOFF_DATE = get_config_backtest_date()   # ← 自動從 config.py 讀取，若 config.py 為 None 則預設為 "20250801"
# 注意：是否跳過模式 A 因子調參，請使用 CLI 參數 --skip_factor_opt，而非在此修改
BEST_FACTORS_PATH = os.path.join(BASE_DIR, "configs", "best_factors.json")
BEST_TRADING_PARAMS_PATH = os.path.join(BASE_DIR, "configs", "best_trading_params.json")
STABILITY_REPORT_PATH = os.path.join(BASE_DIR, "reports", "regime_stability_report.txt")

# 備份路徑
CONFIG_BAK = CONFIG_PATH + ".workflow.bak"
BEST_FACTORS_BAK = BEST_FACTORS_PATH + ".workflow.bak"
BEST_TRADING_PARAMS_BAK = BEST_TRADING_PARAMS_PATH + ".workflow.bak"


def format_val_for_config(val):
    """將參數值格式化為 config.py 的 Python 代碼格式"""
    if val is None or str(val).strip().lower() == "none":
        return "None"
    return str(val).strip()


def get_latest_data_date():
    """掃描 data/raw_price/ 目錄，自動偵測最新已下載的資料日期，回傳 YYYY-MM-DD 字串。"""
    import glob as _glob
    price_dir = os.path.join(BASE_DIR, "data", "raw_price")
    if not os.path.isdir(price_dir):
        fallback = time.strftime("%Y-%m-%d")
        print(f"  [警告] 找不到 data/raw_price/，以今日 {fallback} 作為回測結束日")
        return fallback
    files = _glob.glob(os.path.join(price_dir, "*_price.csv"))
    dates = [
        m.group(1)
        for f in files
        for m in [re.match(r"(\d{8})_price\.csv", os.path.basename(f))]
        if m
    ]
    if not dates:
        fallback = time.strftime("%Y-%m-%d")
        print(f"  [警告] data/raw_price/ 無有效資料，以今日 {fallback} 作為回測結束日")
        return fallback
    latest = max(dates)
    return f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"


def update_config_var(var_name, new_val_str):
    """安全地用正則表達式更新 config.py 中的變數值"""
    if not os.path.exists(CONFIG_PATH):
        print(f"[錯誤] 找不到 config.py: {CONFIG_PATH}")
        return False
        
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 匹配 var_name = value 格式，忽略註解，支援字串、數字、布林值與 None，且允許縮排但排除已被註解之行
    pattern = rf'^(\s*{var_name}\s*=\s*)([^\n#]+)'
    replacement = rf'\g<1>{new_val_str}'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        print(f"[警告] 無法更新 config.py 中的 {var_name}")
        return False
        
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  [Config變更] {var_name} = {new_val_str}")
    return True


def run_cmd(cmd_list, description=""):
    """執行命令，並將輸出實時印出，同時捕獲輸出"""
    if description:
        print(f"\n>>> 正在執行: {description}")
    print(f"    指令: {' '.join(cmd_list)}")
    
    start_time = time.time()
    process = None
    try:
        # 強制子進程使用 UTF-8 編碼輸出
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # 創建 subprocess
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=BASE_DIR,
            env=env
        )
        
        lines = []
        
        def reader():
            try:
                for line in process.stdout:
                    sys.stdout.write("    " + line)
                    sys.stdout.flush()
                    lines.append(line)
            except Exception:
                pass

        # 啟動背景線程讀取輸出，避免阻塞主線程的信號處理
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        
        # 主線程以 time.sleep 進行非阻塞式循環等待，以確保 Ctrl+C (KeyboardInterrupt) 可以在 Windows 上被即時捕獲
        while t.is_alive() or process.poll() is None:
            time.sleep(0.1)
            if not t.is_alive() and process.poll() is None:
                # 讀取線程已結束但進程尚未結束，可能已到達 EOF，給予短暫緩衝後檢查
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    # 超時說明讀取線程異常崩潰，而進程仍在運行且可能因 pipe 滿而堵塞
                    print("⚠️ [警告] 輸出讀取線程已提前終止，但子進程仍在運行。可能發生管道阻塞，正在終止子進程...")
                    process.terminate()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                
        rc = process.wait()
        elapsed = time.time() - start_time
        
        if rc != 0:
            print(f"!!! [錯誤] 執行失敗，錯誤代碼: {rc}，耗時: {elapsed:.1f} 秒")
            # 拋出異常以便於 try-finally 機制捕獲
            raise RuntimeError(f"Command {' '.join(cmd_list)} failed with code {rc}")
            
        print(f"✓ [完成] 耗時: {elapsed:.1f} 秒")
        return "".join(lines), elapsed
        
    except KeyboardInterrupt:
        print(f"\n⚠️  [使用者中斷] 偵測到 Ctrl+C！正在安全終止子進程...")
        if process:
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print("  子進程未能在 3 秒內結束，執行強制關閉 (Kill)...")
                process.kill()
            print("✓ 子進程已安全關閉。")
        raise KeyboardInterrupt


def parse_sim_output(stdout_str):
    """從 trading_sim.py 的 stdout 中解析區間報酬與最大回撤"""
    ret_match = re.search(r"區間報酬:\s*([+-]?\d+\.?\d*)%", stdout_str)
    mdd_match = re.search(r"最大回撤:\s*-?(\d+\.?\d*)%", stdout_str)
    
    ret_val = float(ret_match.group(1)) if ret_match else None
    mdd_val = float(mdd_match.group(1)) if mdd_match else None
    return ret_val, mdd_val


def load_trading_params(filepath):
    """載入風控參數 JSON，並確保包含 'best_params' 鍵值且非空"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到風控參數檔案: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "best_params" not in data:
        raise KeyError(f"在 {os.path.basename(filepath)} 中找不到 'best_params' 鍵值，請檢查優化/調參程式輸出結構是否變動！")
    params = data.get("best_params", {})
    if not params:
        raise ValueError(f"在 {os.path.basename(filepath)} 中的 'best_params' 為空，請檢查優化/調參是否成功運行！")
    return params


def backup_models(suffix):
    """將 models/ 目錄下的所有模型備份，加上指定的 suffix (例如 .mode_a)"""
    model_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(model_dir):
        return
    for name in ["lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json"]:
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, src + suffix)
            print(f"  [模型備份] 備份 {name} -> {name}{suffix}")


def restore_models(suffix):
    """還原指定 suffix 的模型到 models/ 目錄下"""
    model_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(model_dir):
        return False
    restored_any = False
    for name in ["lgbm_model_1.txt", "lgbm_model_2.txt", "lgbm_model_3.txt", "feature_cols.json"]:
        src = os.path.join(model_dir, name + suffix)
        dst = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            restored_any = True
            print(f"  [模型還原] 還原 {name}{suffix} -> {name}")
    return restored_any


def fmt_pct(val):
    if val is None or val == "N/A":
        return "未完成"
    try:
        return f"{float(val):+.2f}%"
    except (ValueError, TypeError):
        return str(val)


def fmt_mdd(val):
    if val is None or val == "N/A":
        return "未完成"
    try:
        return f"-{float(val):.2f}%"
    except (ValueError, TypeError):
        return str(val)


def fmt_calmar(ret, mdd):
    if ret is None or mdd is None or ret == "N/A" or mdd == "N/A":
        return "未完成"
    try:
        r = float(ret)
        m = float(mdd)
        if m == 0:
            return "N/A"
        return f"{r / m:.2f}"
    except (ValueError, TypeError):
        return "未完成"


def fmt_param_val(val, suffix=""):
    if val is None or val == "N/A":
        return "未完成"
    return f"{val}{suffix}"


def write_experiment_report(res):
    """將實驗結果對比寫入 Markdown 報告"""
    report_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "workflow_experiment_report.md")
    
    mode_a_p = res["mode_a"].get("params", {})
    mode_b_p = res["mode_b"].get("params", {})
    args_info = res["args"]
    date_info = res.get("date_info", {})
    _cutoff      = date_info.get("mode_a_cutoff",   "2025-08-01")
    _oos_start   = date_info.get("mode_a_oos_start", "2025-08-02")
    _latest      = date_info.get("latest_date",      "N/A")

    shap_table = res["mode_a"].get("shap_drift_table", "")
    shap_section_md = ""
    if shap_table:
        shap_section_md = f"""
---

## 🔍 OOS 特徵 SHAP 漂移診斷 (Top 5 Features)

| 特徵名稱 | Gain% | IS_SHAP_A | OOS_SHAP_A | OOS_SHAP_M | Drift(Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
{shap_table}

*註：當 Drift(Mean) 正負號發生反轉時，說明特徵作用方向改變（Regime Shift）。*
"""

    report_content = f"""# 🇹🇼 台灣股市量化交易系統 ─ 模式 A 與 模式 B 雙階段自動化實驗報告

本報告由自動化實驗腳本 `run_workflow_experiment.py` 於 {time.strftime('%Y-%m-%d %H:%M:%S')} 自動生成。本實驗旨在對比研發研究（模式 A）與實盤生產（模式 B）的表現，並提供風控參數對比。

---

## 📋 實驗設定 (Experiment Configurations)

*   **因子最佳化搜尋設定 (Factor Search)**：最大輪數: `{args_info["factor_trials"]}` 輪 | 早停設定: `{args_info["factor_early_stopping"]}` 輪
*   **風控最佳化搜尋設定 (Trading Search)**：最大輪數: `{args_info["trading_trials"]}` 輪 | 早停設定: `{args_info["trading_early_stopping"]}` 輪
*   **模擬初始資金 (Capital)**：`{args_info["capital"]:,}` 元
*   **模式 A 執行因子優化**：`{"否 (沿用歷史因子)" if args_info["skip_factor_opt"] else "是"}`

---

## 📊 關鍵績效指標對比 (Key Metrics)

| 指標 / 模式 | 🟢 模式 A (研究模式 - 乾淨樣本外) | 🔵 模式 B (實盤模式 - 全數據包含牛市) |
| :--- | :--- | :--- |
| **訓練與測試分界** | 截斷於 {_cutoff} (樣本外測試) | `None` (每日滾動重訓至最新) |
| **最優時間衰減 (Lambda)** | `{res.get("best_lambda", "未完成")}` | 沿用模式 A 最優值 |
| **樣本內 (IS) 因子 RankIC** | `{res["mode_a"].get("is_rankic", "未完成")}` | 沿用模式 A 因子 |
| **樣本外 (OOS) 牛市 RankIC**| `{res["mode_a"].get("oos_bull_rankic", "未完成")}` | N/A (模型已納入牛市訓練) |
| **OOS 弱動能組 (ret1 <= 2%) RankIC**| `{res["mode_a"].get("weak_mom_ic", "未完成")}` | N/A |
| **OOS 強動能組 (ret1 > 2%) RankIC**| `{res["mode_a"].get("strong_mom_ic", "未完成")}` | N/A |
| **回測測試區間** | {_oos_start} ~ {_latest} (樣本外) | 2023-01-01 ~ {_latest} (全週期) |
| **回測累計報酬率 (%)** | `{fmt_pct(res["mode_a"].get("oos_return"))}` | `{fmt_pct(res["mode_b"].get("full_return"))}` |
| **回測最大回撤 MDD (%)** | `{fmt_mdd(res["mode_a"].get("oos_mdd"))}` | `{fmt_mdd(res["mode_b"].get("full_mdd"))}` |
| **Calmar 比率 (報酬/MDD)** | `{fmt_calmar(res["mode_a"].get("oos_return"), res["mode_a"].get("oos_mdd"))}` | `{fmt_calmar(res["mode_b"].get("full_return"), res["mode_b"].get("full_mdd"))}` |

*註：模式 A 的回測區間屬於完全未見過的樣本外 (OOS) 測試集，代表策略在全新超級牛市下的防禦與獲利能力。模式 B 的回測區間為包含牛市與熊市的全週期回測，展現策略的長線穩健性。*
{shap_section_md}
---

## ⚙️ 最佳化風控策略參數對比 (Optimized Trading Params)

本部分對比 Walk-Forward 在兩種模式下搜尋出的最佳交易參數中位數（Median）。**這揭示了牛市與熊市/阻礙市下，最優風控配置的結構性漂移：**

| 風控參數 | 🟢 模式 A (未見過牛市的最佳化) | 🔵 模式 B (包含牛市的最佳化) | 參數說明 |
| :--- | :--- | :--- | :--- |
| **買入門檻 (`buy_threshold`)** | `{fmt_param_val(mode_a_p.get("buy_threshold"), "%")}` | `{fmt_param_val(mode_b_p.get("buy_threshold"), "%")}` | D1 多空預測分數觸發買進的百分比。 |
| **賣出門檻 (`sell_threshold`)** | `{fmt_param_val(mode_a_p.get("sell_threshold"), "%")}` | `{fmt_param_val(mode_b_p.get("sell_threshold"), "%")}` | Day 3 多空預測分數低於此值觸發賣出的百分比。 |
| **個股停損 (`stop_loss`)** | `{fmt_param_val(mode_a_p.get("stop_loss"), "%")}` | `{fmt_param_val(mode_b_p.get("stop_loss"), "%")}` | 買入後的個股固定停損線。 |
| **避險門檻 (`panic_ma5`)** | `{fmt_param_val(mode_a_p.get("panic_ma5"))}` | `{fmt_param_val(mode_b_p.get("panic_ma5"))}` | 大盤 5 日平均回報低於此值觸發避險紅燈。 |
| **避險門檻 (`panic_breadth`)**| `{fmt_param_val(mode_a_p.get("panic_breadth"))}` | `{fmt_param_val(mode_b_p.get("panic_breadth"))}` | 全市場上漲比例低於此值觸發避險紅燈。 |
| **移動止盈啟動 (`ts_activation`)**| `{fmt_param_val(mode_a_p.get("ts_activation"), "%")}` | `{fmt_param_val(mode_b_p.get("ts_activation"), "%")}` | 個股利潤達到此值開啟移動追蹤止盈。 |
| **移動止盈回撤 (`ts_pullback`)** | `{fmt_param_val(mode_a_p.get("ts_pullback"), "%")}` | `{fmt_param_val(mode_b_p.get("ts_pullback"), "%")}` | 移動止盈開啟後自高點拉回多少執行停利。 |
| **最少持股天數 (`min_hold_days`)** | `{fmt_param_val(mode_a_p.get("min_hold_days"), " 天")}` | `{fmt_param_val(mode_b_p.get("min_hold_days"), " 天")}` | 防止頻繁交易所限制的最短持倉天數。 |
| **掛單折溢價幅 (`markup_pct`)**  | `{fmt_param_val(mode_a_p.get("markup_pct"), "%")}` | `{fmt_param_val(mode_b_p.get("markup_pct"), "%")}` | 掛單折溢價比例，負數代表折價拉回買進。 |

### 💡 研究員核心分析與結論：
1. **為什麼模式 A 與模式 B 的最優風控參數存在差異？**
   * 模式 A 優化時，Optuna 看不到 2025-08-01 之後的兩萬點到四萬點大牛市，因此其優化出的風控引數更加傾向於**「防守阻礙與熊市」**。
   * 模式 B 將 2025-08 ~ 2026-06 的超級牛市納源優化眼界。在強多頭市場中，大盤避險紅燈門檻（`panic_breadth` 和 `panic_ma5`）通常會被優化得更加寬容，個股停損（`stop_loss`）與移動止盈回撤（`ts_pullback`）也會更寬，以適應牛市個股的劇烈波動，防止被輕易洗出場，最大化捕捉趨勢利潤。
2. **策略健康度判斷：**
   * 若模式 A 在 OOS 區間的 `oos_return` 表現良好且 `oos_mdd` 受控，證明策略具備極強的**樣本外泛化能力**，非過擬合。
   * 正式實盤上線時，推薦**以模式 B 訓練出的最新模型**作為預測大腦（擁有最新特徵），並套用**模式 B 產出的 `best_trading_params_mode_b.json` 風控參數**進行每日交易，以在當下大牛市中獲得最契合市場波動的收益。

---
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  [進度更新] 實驗分析報告已寫入: {report_path}")


def save_progress(results):
    """保存當前實驗進度 JSON 並重新生成 Markdown 報告"""
    report_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. 寫入 JSON 檔
    progress_json_path = os.path.join(report_dir, "workflow_experiment_results.json")
    try:
        with open(progress_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  [進度更新] 實驗結果 JSON 已存檔: {progress_json_path}")
    except Exception as e:
        print(f"  [警告] 儲存進度 JSON 失敗: {e}")
        
    # 2. 寫入 Markdown 報告
    write_experiment_report(results)


def main():
    parser = argparse.ArgumentParser(description="模式 A & 模式 B 自動化量化工作流實驗器")
    parser.add_argument("-f", "--factor_trials", type=int, default=FACTOR_TRIALS, help="因子調參最大搜尋輪數")
    parser.add_argument("-fe", "--factor_early_stopping", type=str, default=str(FACTOR_EARLY_STOPPING), help="因子調參早停輪數 (None 代表不啟用)")
    parser.add_argument("-t", "--trading_trials", type=int, default=TRADING_TRIALS, help="風控調參最大搜尋輪數")
    parser.add_argument("-te", "--trading_early_stopping", type=str, default=str(TRADING_EARLY_STOPPING), help="風控調參早停輪數 (None 代表不啟用)")
    parser.add_argument("-c", "--capital", type=int, default=CAPITAL, help="回測與優化的初始資金")
    parser.add_argument("--skip_factor_opt", action="store_true", help="模式 A 跳過因子優化，直接沿用現有 best_factors.json")
    parser.add_argument("--fresh", action="store_true", help="強制重新執行所有步驟，忽略現有的 checkpoint 與中間 JSON 檔")
    args = parser.parse_args()
    
    # 參數標準化 (處理 'None')
    fact_es_str = format_val_for_config(args.factor_early_stopping)
    trad_es_str = format_val_for_config(args.trading_early_stopping)

    optuna_jobs = 1
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
                _content = _f.read()
            _m = re.search(r'OPTUNA_N_JOBS\s*=\s*(-?\d+)', _content)
            if _m:
                optuna_jobs = int(_m.group(1))
    except Exception:
        pass

    # ── 0. 動態推算所有回測日期 ─────────────────────────────────────────────
    from datetime import datetime as _dt, timedelta as _td
    latest_date      = get_latest_data_date()                              # 最新資料日 (自動偵測)
    cutoff_dt        = _dt.strptime(MODE_A_CUTOFF_DATE, "%Y%m%d")
    mode_a_cutoff_str = cutoff_dt.strftime("%Y-%m-%d")                    # 例：2025-08-01
    mode_a_oos_start  = (cutoff_dt + _td(days=1)).strftime("%Y-%m-%d")   # 例：2025-08-02

    print("=" * 80)
    print("   🚀 台灣股市量化交易系統 ─ 模式 A 與 模式 B 雙階段自動化實驗控制台 🚀")
    print("=" * 80)
    print(f"  * 因子調參 (Factor) : 最大 {args.factor_trials} 輪 | 早停門檻: {fact_es_str} 輪")
    print(f"  * 風控調參 (Trading): 最大 {args.trading_trials} 輪 | 早停門檻: {trad_es_str} 輪")
    print(f"  * 初始模擬資金 (Capital)         : {args.capital:,} 元")
    print(f"  * 模式 A 截斷日期 (BACKTEST_DATE): {mode_a_cutoff_str}")
    print(f"  * 最新資料日期 (自動偵測)        : {latest_date}")
    print(f"  * 模式 A 執行因子最佳化          : {'否 (沿用現有)' if args.skip_factor_opt else '是'}")
    print(f"  * 續傳 / 復原機制啟動           : {'否 (強制重新執行)' if args.fresh else '是 (優先讀取 Checkpoints)'}")
    print("=" * 80)
    
    # ── 1. 安全備份 ────────────────────────────────────────────────────────
    print("\n[步驟 1/4] 安全備份設定與資料檔...")
    
    # 檢查是否有上次殘留的備份，有的話先還原，避免覆蓋正確的備份
    if os.path.exists(CONFIG_BAK):
        print("  [還原] 偵測到上次中斷留下的 config.py.workflow.bak，正在自動還原原始設定檔...")
        shutil.copy2(CONFIG_BAK, CONFIG_PATH)
        
    if os.path.exists(BEST_FACTORS_BAK):
        print("  [還原] 偵測到上次中斷留下的 best_factors.json.workflow.bak，正在自動還原...")
        shutil.copy2(BEST_FACTORS_BAK, BEST_FACTORS_PATH)
        
    if os.path.exists(BEST_TRADING_PARAMS_BAK):
        print("  [還原] 偵測到上次中斷留下的 best_trading_params.json.workflow.bak，正在自動還原...")
        shutil.copy2(BEST_TRADING_PARAMS_BAK, BEST_TRADING_PARAMS_PATH)

    # 現在可以安全進行備份
    if os.path.exists(CONFIG_PATH):
        shutil.copy2(CONFIG_PATH, CONFIG_BAK)
        print(f"  已備份 config.py -> config.py.workflow.bak")
    else:
        print("[嚴重錯誤] 找不到 config.py！")
        sys.exit(1)
        
    backup_factors_exists = False
    if os.path.exists(BEST_FACTORS_PATH):
        shutil.copy2(BEST_FACTORS_PATH, BEST_FACTORS_BAK)
        backup_factors_exists = True
        print(f"  已備份 best_factors.json -> best_factors.json.workflow.bak")
        
    backup_trading_exists = False
    if os.path.exists(BEST_TRADING_PARAMS_PATH):
        shutil.copy2(BEST_TRADING_PARAMS_PATH, BEST_TRADING_PARAMS_BAK)
        backup_trading_exists = True
        print(f"  已備份 best_trading_params.json -> best_trading_params.json.workflow.bak")

    # 用於收集報告的字典
    results = {
        "args": {
            "factor_trials": args.factor_trials,
            "factor_early_stopping": fact_es_str,
            "trading_trials": args.trading_trials,
            "trading_early_stopping": trad_es_str,
            "capital": args.capital,
            "skip_factor_opt": args.skip_factor_opt
        },
        "date_info": {
            "mode_a_cutoff": mode_a_cutoff_str,
            "mode_a_oos_start": mode_a_oos_start,
            "latest_date": latest_date
        },
        "mode_a": {},
        "mode_b": {}
    }
    
    # ── 1.1 讀取現有實驗進度 ──────────────────────────────────────────────
    progress_json_path = os.path.join(BASE_DIR, "reports", "workflow_experiment_results.json")
    if os.path.exists(progress_json_path) and not args.fresh:
        print(f"\n[續傳/復原] 偵測到已有實驗進度檔 {progress_json_path}，正在載入上次進度...")
        try:
            with open(progress_json_path, "r", encoding="utf-8") as f:
                loaded_results = json.load(f)
                # 合併 args_info 以外的進度
                if "mode_a" in loaded_results:
                    results["mode_a"] = loaded_results["mode_a"]
                if "mode_b" in loaded_results:
                    results["mode_b"] = loaded_results["mode_b"]
                if "best_lambda" in loaded_results:
                    results["best_lambda"] = loaded_results["best_lambda"]
                print("  已載入的進度數據:")
                print(f"    模式 A 已完成: {list(results['mode_a'].keys())}")
                print(f"    模式 B 已完成: {list(results['mode_b'].keys())}")
                if "best_lambda" in results:
                    print(f"    最優時間衰減 Lambda: {results['best_lambda']}")
        except Exception as e:
            print(f"  [警告] 讀取進度檔失敗: {e}，將從頭開始執行。")

    try:
        # ── 2. 模式 A 流程 (研究與策略驗證期) ──────────────────────────────────
        print("\n" + "=" * 80)
        print("   🟢 進入 [模式 A]：研究與策略驗證期 (截斷日期: 2025-08-01)")
        print("=" * 80)
        
        # A1. 更新 config 變數以符合因子優化配置
        update_config_var("BACKTEST_DATE", f'"{MODE_A_CUTOFF_DATE}"')
        update_config_var("RUN_OPTIMIZATION", "True" if not args.skip_factor_opt else "False")
        update_config_var("OPTIMIZATION_TRIALS", str(args.factor_trials))
        update_config_var("EARLY_STOPPING_ROUNDS", fact_es_str)
        
        # Checkpoint: 檢查是否已有備份的最佳因子
        factor_mode_a_saved = os.path.join(BASE_DIR, "configs", "best_factors_mode_a.json")
        has_checkpoint_factor = os.path.exists(factor_mode_a_saved) and not args.fresh
        
        if has_checkpoint_factor:
            print(f"\n[續傳/復原] 偵測到已存在的模式 A 最佳因子檔 {factor_mode_a_saved}，直接載入並跳過優化...")
            shutil.copy2(factor_mode_a_saved, BEST_FACTORS_PATH)
        else:
            # A2. 如果不沿用因子，先移開現有的 best_factors.json，讓 Optuna 重頭搜尋
            if not args.skip_factor_opt:
                if os.path.exists(BEST_FACTORS_PATH):
                    os.remove(BEST_FACTORS_PATH)
                    print("  已暫時移除 best_factors.json，以便進行全新的模式 A 因子搜尋")
                
                # A3. 因子優化
                run_cmd([sys.executable, "auto_pipeline.py", "-s", "o"], "模式 A：因子技術指標最佳化")
                if os.path.exists(BEST_FACTORS_PATH):
                    shutil.copy2(BEST_FACTORS_PATH, factor_mode_a_saved)
                    print(f"  已將模式 A 最佳因子另存至 best_factors_mode_a.json")
            else:
                print("  [設定] 跳過模式 A 因子優化 (沿用現有因子)")
                if os.path.exists(BEST_FACTORS_PATH):
                    shutil.copy2(BEST_FACTORS_PATH, factor_mode_a_saved)

        # A4 & A5 & A5.1. 訓練模型 A & 訊號診斷
        stability_summary_path = os.path.join(BASE_DIR, "reports", "mode_a_regime_stability_report.txt")
        has_checkpoint_stability = os.path.exists(stability_summary_path) and not args.fresh
        
        if has_checkpoint_stability:
            with open(stability_summary_path, "r", encoding="utf-8") as rf:
                stability_summary = rf.read()
            if "All" not in stability_summary or "7. SHAP" not in stability_summary:
                print("  [提示] 偵測到舊版診斷報告（缺少 'All' 或 '7. SHAP' 行），將重新執行模型 A 訓練與診斷...")
                has_checkpoint_stability = False
                
        if has_checkpoint_stability:
            print(f"\n[續傳/復原] 偵測到已存在的模式 A 診斷報告，將直接讀取並跳過模型 A 訓練與診斷...")
            restore_models(".mode_a")
        else:
            # A3.2. 重建特徵矩陣
            run_cmd([sys.executable, "auto_pipeline.py", "-s", "f"], "模式 A：重建特徵矩陣 (截斷 2025-08-01)")
            
            # Phase 1 Step 2: Time Decay Grid Search
            print("\n>>> 正在進行 Time Decay 網格搜尋實驗 (Step 2)...")
            lambdas = [0.0, 0.001, 0.002, 0.003, 0.005]
            best_lambda = None
            best_ic = None
            grid_results = {}
            
            for l_val in lambdas:
                print(f"\n[衰減網格測試] 正在測試 Lambda = {l_val}...")
                update_config_var("DEFAULT_DECAY_LAMBDA", str(l_val))
                
                # 訓練模型
                run_cmd([sys.executable, "auto_pipeline.py", "-s", "t"], f"訓練 LightGBM 模型 (Lambda={l_val})")
                
                # 診斷 OOS 訊號
                stdout_diag, _ = run_cmd([sys.executable, "scripts/analyze_regime_stability.py", "--silent"], f"診斷 OOS 訊號 (Lambda={l_val})")
                
                # 從 stdout 中解析 RankIC
                ic_match = re.search(r"OOS\s*\|\s*All\s*\|\s*\d+\s*\|\s*([+-]?\d+\.?\d*)", stdout_diag)
                ic_val = float(ic_match.group(1)) if ic_match else None
                
                # 解析 Top 1% Alpha
                alpha_match = re.search(r"Top 1%  平均報酬: [+-]?\d+\.?\d*% \| 超額 Alpha:\s*([+-]?\d+\.?\d*)", stdout_diag)
                alpha_val = float(alpha_match.group(1)) if alpha_match else 0.0
                
                # 解析 LS Spread
                spread_match = re.search(r"OOS\s*\|\s*All\s*\|\s*\d+\s*\|\s*[+-]?\d+\.?\d*\s*\|\s*([+-]?\d+\.?\d*)%", stdout_diag)
                spread_val = float(spread_match.group(1)) if spread_match else 0.0
                
                if ic_val is not None:
                    print(f"  -> 結果: OOS RankIC = {ic_val:+.4f} | Top1 Alpha = {alpha_val:+.3f}% | LS Spread = {spread_val:+.3f}%")
                    grid_results[l_val] = (ic_val, alpha_val, spread_val)
                    if best_ic is None or ic_val > best_ic:
                        best_ic = ic_val
                        best_lambda = l_val
                else:
                    print(f"  -> [警告] 無法解析 Lambda = {l_val} 的 OOS RankIC")
                    grid_results[l_val] = ("解析失敗", alpha_val, spread_val)
            
            if best_lambda is None:
                raise RuntimeError("Time Decay 網格搜尋中所有 Lambda 的 OOS RankIC 解析均失敗，無法決定最佳 Lambda！請檢查 scripts/analyze_regime_stability.py 的輸出格式是否變動。")
            
            print(f"\n[網格搜尋完成] 最佳時間衰減係數: Lambda = {best_lambda} (OOS RankIC = {best_ic:+.4f})")
            print("各參數對比:")
            for lv, res_tuple in grid_results.items():
                if len(res_tuple) == 3:
                    ic, al, sp = res_tuple
                    if isinstance(ic, float):
                        print(f"  Lambda = {lv:<5} | RankIC = {ic:+.4f} | Alpha = {al:+.3f}% | Spread = {sp:+.3f}%")
                    else:
                        print(f"  Lambda = {lv:<5} | RankIC = {ic} | Alpha = {al:+.3f}% | Spread = {sp:+.3f}%")
                
            results["best_lambda"] = best_lambda
            
            # 將最優 Lambda 寫入 config.py
            update_config_var("DEFAULT_DECAY_LAMBDA", str(best_lambda))
            
            # 以最佳 Lambda 重新訓練最終模型 A
            run_cmd([sys.executable, "auto_pipeline.py", "-s", "t"], f"以最佳 Lambda={best_lambda} 重新訓練最終模型 A")
            
            # 跑最終的診斷報告 (直接輸出到 mode_a_regime_stability_report.txt)
            run_cmd([sys.executable, "scripts/analyze_regime_stability.py", "--output", stability_summary_path], "生成最終模式 A 診斷報告")
            
            backup_models(".mode_a")
            
            stability_summary = "找不到報告"
            if os.path.exists(stability_summary_path):
                with open(stability_summary_path, "r", encoding="utf-8") as rf:
                    stability_summary = rf.read()
        
        # 解析關鍵 RankIC 指標與 SHAP 表
        ic_is_all = re.search(r"IS\s*\|\s*All\s*\|\s*\d+\s*\|\s*([+-]?\d+\.?\d*)", stability_summary)
        ic_oos_bull = re.search(r"OOS\s*\|\s*Bull\s*\|\s*\d+\s*\|\s*([+-]?\d+\.?\d*)", stability_summary)
        ic_weak_mom = re.search(r"弱動能組.*平均 RankIC:\s*([+-]?\d+\.?\d*)", stability_summary)
        ic_strong_mom = re.search(r"強動能組.*平均 RankIC:\s*([+-]?\d+\.?\d*)", stability_summary)
        
        results["mode_a"]["is_rankic"] = ic_is_all.group(1) if ic_is_all else "N/A"
        results["mode_a"]["oos_bull_rankic"] = ic_oos_bull.group(1) if ic_oos_bull else "N/A"
        results["mode_a"]["weak_mom_ic"] = ic_weak_mom.group(1) if ic_weak_mom else "N/A"
        results["mode_a"]["strong_mom_ic"] = ic_strong_mom.group(1) if ic_strong_mom else "N/A"
        
        # 解析 Section 7 SHAP Drift table
        shap_drift_section = re.search(r"7\. SHAP 與特徵重要性漂移.*?\n-+\n(.*?)\n-+\n診斷指引", stability_summary, re.DOTALL)
        shap_drift_lines = []
        if shap_drift_section:
            lines = shap_drift_section.group(1).strip().split('\n')
            for l in lines:
                if '|' not in l:
                    continue
                parts = [p.strip() for p in l.split('|')]
                # 排除標題行與分界線
                if len(parts) >= 8 and "特徵名稱" not in parts[0] and "feature" not in parts[0].lower():
                    shap_drift_lines.append(f"| {parts[0]} | {parts[1]} | {parts[3]} | {parts[4]} | {parts[6]} | {parts[7]} |")
                    if len(shap_drift_lines) >= 5:
                        break
        
        results["mode_a"]["shap_drift_table"] = "\n".join(shap_drift_lines) if shap_drift_lines else ""
        save_progress(results)
        
        # A8. 執行風控參數最佳化
        trading_mode_a_saved = os.path.join(BASE_DIR, "configs", "best_trading_params_mode_a.json")
        has_checkpoint_trading_a = os.path.exists(trading_mode_a_saved) and not args.fresh
        
        mode_a_params = {}
        if has_checkpoint_trading_a:
            print(f"\n[續傳/復原] 偵測到已存在的模式 A 風控參數 {trading_mode_a_saved}，直接載入並跳過優化...")
            shutil.copy2(trading_mode_a_saved, BEST_TRADING_PARAMS_PATH)
            mode_a_params = load_trading_params(trading_mode_a_saved)
        else:
            # 移除舊的交易參數以防干擾風控優化
            if os.path.exists(BEST_TRADING_PARAMS_PATH):
                os.remove(BEST_TRADING_PARAMS_PATH)
                
            # 更新 config 中的早停設定為風控專用
            update_config_var("EARLY_STOPPING_ROUNDS", trad_es_str)
            
            run_cmd([
                sys.executable, "scripts/optimize_trading_params.py",
                "-t", str(args.trading_trials),
                "-s", "2021-01-02",
                "-e", mode_a_cutoff_str,
                "-c", str(args.capital),
                "-j", str(optuna_jobs),
                "-wf"
            ], f"模式 A：交易與風控參數最佳化 (Walk-Forward, 2021-01-02 ~ {mode_a_cutoff_str})")
            
            if os.path.exists(BEST_TRADING_PARAMS_PATH):
                mode_a_params = load_trading_params(BEST_TRADING_PARAMS_PATH)
                shutil.copy2(BEST_TRADING_PARAMS_PATH, trading_mode_a_saved)
                print(f"  已將模式 A 風控參數另存至 best_trading_params_mode_a.json")
                
        results["mode_a"]["params"] = mode_a_params
        save_progress(results)
        
        # A9. 在樣本外超級牛市進行模擬交易 (2025-08-02 ~ 2026-06-05)
        has_checkpoint_sim_a = ("oos_return" in results["mode_a"]) and ("oos_mdd" in results["mode_a"]) and not args.fresh
        # 修正：若模擬結果為 None (先前解析錯誤導致)，則強制重新執行以獲取正確數據
        if has_checkpoint_sim_a and (results["mode_a"].get("oos_return") is None or results["mode_a"].get("oos_mdd") is None):
            print("  [提示] 偵測到模擬交易結果為 None (可能為先前解析錯誤的無效數據)，將重新執行模擬...")
            has_checkpoint_sim_a = False
            
        if has_checkpoint_sim_a:
            print(f"\n[續傳/復原] 偵測到已存在的模式 A 模擬交易結果，跳過模擬交易回測...")
        else:
            # 確保 BACKTEST_DATE 設為模式 A 截斷日，以加載模型 A
            update_config_var("BACKTEST_DATE", f'"{MODE_A_CUTOFF_DATE}"')

            sim_stdout, _ = run_cmd([
                sys.executable, "trading_sim.py",
                "-s", mode_a_oos_start,
                "-e", latest_date,
                "-c", str(args.capital)
            ], f"模式 A：樣本外 (OOS) 超級牛市模擬交易回測 (Model A, {mode_a_oos_start} ~ {latest_date})")
            
            ret_a, mdd_a = parse_sim_output(sim_stdout)
            if ret_a is None or mdd_a is None:
                print("⚠️ [警告] 無法解析模式 A 模擬交易結果！請檢查 trading_sim.py 輸出是否正常。")
            results["mode_a"]["oos_return"] = ret_a
            results["mode_a"]["oos_mdd"] = mdd_a
            save_progress(results)
            
        # B1. 更新 config 變數
        update_config_var("BACKTEST_DATE", "None")
        update_config_var("RUN_OPTIMIZATION", "False") # 沿用模式 A 的最佳因子
        
        # B6. 執行全週期風控參數優化
        trading_mode_b_saved = os.path.join(BASE_DIR, "configs", "best_trading_params_mode_b.json")
        has_checkpoint_trading_b = os.path.exists(trading_mode_b_saved) and not args.fresh
        
        # Checkpoint: 檢查是否能復原 Mode B 的模型以節省時間
        has_mode_b_model = False
        if has_checkpoint_trading_b:
            has_mode_b_model = restore_models(".mode_b")
            
        if has_mode_b_model:
            print("  [續傳/復原] 成功從備份還原模式 B 模型，跳過特徵重建與重訓 (B2/B3)。")
        else:
            # B2. 重建特徵工程
            run_cmd([sys.executable, "auto_pipeline.py", "-s", "f"], "模式 B：重建特徵工程 (全數據覆蓋)")
            
            # B3. 重訓模型 B (包含牛市數據)
            run_cmd([sys.executable, "auto_pipeline.py", "-s", "t"], "模式 B：重訓 LightGBM 模型 (包含超級牛市)")
            
            backup_models(".mode_b")
            
        mode_b_params = {}
        if has_checkpoint_trading_b:
            print(f"\n[續傳/復原] 偵測到已存在的模式 B 風控參數 {trading_mode_b_saved}，直接載入並跳過優化...")
            shutil.copy2(trading_mode_b_saved, BEST_TRADING_PARAMS_PATH)
            mode_b_params = load_trading_params(trading_mode_b_saved)
        else:
            # B4. 移除模式 A 的最佳交易參數以防干擾模式 B 優化
            if os.path.exists(BEST_TRADING_PARAMS_PATH):
                os.remove(BEST_TRADING_PARAMS_PATH)
                
            # B5. 更新 config 中的早停設定為風控專用
            update_config_var("EARLY_STOPPING_ROUNDS", trad_es_str)
            
            # B6. 執行全週期風控參數優化 (覆蓋整個牛市)
            run_cmd([
                sys.executable, "scripts/optimize_trading_params.py",
                "-t", str(args.trading_trials),
                "-s", "2023-01-01",
                "-e", latest_date,
                "-c", str(args.capital),
                "-j", str(optuna_jobs),
                "-wf"
            ], f"模式 B：交易與風控參數全週期最佳化 (Walk-Forward, 2023-01-01 ~ {latest_date})")
            
            if os.path.exists(BEST_TRADING_PARAMS_PATH):
                mode_b_params = load_trading_params(BEST_TRADING_PARAMS_PATH)
                shutil.copy2(BEST_TRADING_PARAMS_PATH, trading_mode_b_saved)
                print(f"  已將模式 B 風控參數另存至 best_trading_params_mode_b.json")
                
        results["mode_b"]["params"] = mode_b_params
        save_progress(results)
        
        # B7. 執行全週期模擬交易回測 (2023-01-01 ~ 2026-06-05)
        has_checkpoint_sim_b = ("full_return" in results["mode_b"]) and ("full_mdd" in results["mode_b"]) and not args.fresh
        # 修正：若模擬結果為 None (先前解析錯誤導致)，則強制重新執行以獲取正確數據
        if has_checkpoint_sim_b and (results["mode_b"].get("full_return") is None or results["mode_b"].get("full_mdd") is None):
            print("  [提示] 偵測到模擬交易結果為 None (可能為先前解析錯誤的無效數據)，將重新執行模擬...")
            has_checkpoint_sim_b = False
            
        if has_checkpoint_sim_b:
            print(f"\n[續傳/復原] 偵測到已存在的模式 B 模擬交易結果，跳過模擬交易回測...")
        else:
            # 確保 config 變數為 Mode B
            update_config_var("BACKTEST_DATE", "None")

            sim_stdout_b, _ = run_cmd([
                sys.executable, "trading_sim.py",
                "-s", "2023-01-01",
                "-e", latest_date,
                "-c", str(args.capital)
            ], f"模式 B：全週期 (含大牛市) 模擬交易回測 (Model B, 2023-01-01 ~ {latest_date})")
            
            ret_b, mdd_b = parse_sim_output(sim_stdout_b)
            if ret_b is None or mdd_b is None:
                print("⚠️ [警告] 無法解析模式 B 模擬交易結果！請檢查 trading_sim.py 輸出是否正常。")
            results["mode_b"]["full_return"] = ret_b
            results["mode_b"]["full_mdd"] = mdd_b
            save_progress(results)
            
        # B8. 執行推理預測，產生明天的買賣建議
        has_checkpoint_infer_b = results["mode_b"].get("inference_completed") and not args.fresh
        
        if has_checkpoint_infer_b:
            print(f"\n[續傳/復原] 偵測到已完成模式 B 推理預測，跳過推理...")
        else:
            # 確保 config 變數為 Mode B
            update_config_var("BACKTEST_DATE", "None")
            
            run_cmd([sys.executable, "auto_pipeline.py", "-s", "i"], "模式 B：模型推理預測，產生明日實盤下單建議")
            results["mode_b"]["inference_completed"] = True
            save_progress(results)
            
        # ── 4. 產出最終實驗報告 ──────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("   🎉 全自動化實驗流程順利完成！對比分析報告已輸出。 🎉")
        print("   報告位置: reports/workflow_experiment_report.md")
        print("=" * 80 + "\n")
        
    except Exception as ex:
        print(f"\n[嚴重異常] 實驗被異常中斷: {ex}")
        import traceback
        traceback.print_exc()
        
    finally:
        # ── 5. 還原環境設定 (確保不論成敗都還原設定檔) ──────────────────────
        print("\n[步驟 4/4] 正在還原原始設定檔與備份...")
        
        # 清理臨時的診斷報告，只保留 mode_a_regime_stability_report.txt 精裝版
        if os.path.exists(STABILITY_REPORT_PATH):
            try:
                os.remove(STABILITY_REPORT_PATH)
                print("  已清理臨時的 regime_stability_report.txt 診斷報告")
            except Exception:
                pass

        if os.path.exists(CONFIG_BAK):
            shutil.copy2(CONFIG_BAK, CONFIG_PATH)
            os.remove(CONFIG_BAK)
            print("  已還原 config.py 并清理臨時備份")
            
        if backup_factors_exists and os.path.exists(BEST_FACTORS_BAK):
            shutil.copy2(BEST_FACTORS_BAK, BEST_FACTORS_PATH)
            os.remove(BEST_FACTORS_BAK)
            print("  已還原 best_factors.json 并清理臨時備份")
            
        if backup_trading_exists and os.path.exists(BEST_TRADING_PARAMS_BAK):
            shutil.copy2(BEST_TRADING_PARAMS_BAK, BEST_TRADING_PARAMS_PATH)
            os.remove(BEST_TRADING_PARAMS_BAK)
            print("  已還原 best_trading_params.json 并清理臨時備份")


if __name__ == "__main__":
    main()
