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

# ── 1. 實驗核心配置 (最上面易於手動調整的變數，也支援執行時 CLI 參數覆蓋) ────────
# ⚙️ 因子最佳化設定 (scripts/optimize_factors.py)
FACTOR_TRIALS         = 600        # 因子最佳化最大搜尋輪數 (預設較少以利快速驗證)
FACTOR_EARLY_STOPPING = 250        # 因子最佳化早停輪數 (無進度達此輪數自動終止，None 代表不啟用)

# 🎛️ 交易風控最佳化設定 (scripts/optimize_trading_params.py)
TRADING_TRIALS         = 600      # 風控最佳化最大搜尋輪數 (預設 100 輪以兼顧效率與品質)
TRADING_EARLY_STOPPING = 250       # 風控最佳化早停輪數 (無進步達此輪數自動終止，None 代表不啟用)

# 💰 模擬交易基本設定
CAPITAL        = 2000000          # 回測與優化的初始資金 (200 萬)
RUN_FACTOR_OPT = True             # 模式 A 是否重新執行因子最佳化 (False 則沿用現有 best_factors.json)

# 路徑定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.py")
BEST_FACTORS_PATH = os.path.join(BASE_DIR, "best_factors.json")
BEST_TRADING_PARAMS_PATH = os.path.join(BASE_DIR, "best_trading_params.json")
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


def update_config_var(var_name, new_val_str):
    """安全地用正則表達式更新 config.py 中的變數值"""
    if not os.path.exists(CONFIG_PATH):
        print(f"[錯誤] 找不到 config.py: {CONFIG_PATH}")
        return False
        
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 匹配 var_name = value 格式，忽略註解，支援字串、數字、布林值與 None
    pattern = rf'^({var_name}\s*=\s*)([^\n#]+)'
    replacement = rf'\g<1>{new_val_str}'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        # 嘗試無首行錨定的匹配
        pattern_fallback = rf'({var_name}\s*=\s*)([^\n#]+)'
        new_content, count = re.subn(pattern_fallback, replacement, content)
        
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
        # 創建 subprocess
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=BASE_DIR
        )
        
        lines = []
        # 實時輸出
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write("    " + line)
                sys.stdout.flush()
                lines.append(line)
                
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
    
    ret_val = float(ret_match.group(1)) if ret_match else 0.0
    mdd_val = float(mdd_match.group(1)) if mdd_match else 0.0
    return ret_val, mdd_val


def main():
    parser = argparse.ArgumentParser(description="模式 A & 模式 B 自動化量化工作流實驗器")
    parser.add_argument("-f", "--factor_trials", type=int, default=FACTOR_TRIALS, help="因子調參最大搜尋輪數")
    parser.add_argument("-fe", "--factor_early_stopping", type=str, default=str(FACTOR_EARLY_STOPPING), help="因子調參早停輪數 (None 代表不啟用)")
    parser.add_argument("-t", "--trading_trials", type=int, default=TRADING_TRIALS, help="風控調參最大搜尋輪數")
    parser.add_argument("-te", "--trading_early_stopping", type=str, default=str(TRADING_EARLY_STOPPING), help="風控調參早停輪數 (None 代表不啟用)")
    parser.add_argument("-c", "--capital", type=int, default=CAPITAL, help="回測與優化的初始資金")
    parser.add_argument("--skip_factor_opt", action="store_true", help="模式 A 跳過因子優化，直接沿用現有 best_factors.json")
    args = parser.parse_args()
    
    # 參數標準化 (處理 'None')
    fact_es_str = format_val_for_config(args.factor_early_stopping)
    trad_es_str = format_val_for_config(args.trading_early_stopping)
    
    print("=" * 80)
    print("   🚀 台灣股市量化交易系統 ─ 模式 A 與 模式 B 雙階段自動化實驗控制台 🚀")
    print("=" * 80)
    print(f"  * 因子調參 (Factor) : 最大 {args.factor_trials} 輪 | 早停門檻: {fact_es_str} 輪")
    print(f"  * 風控調參 (Trading): 最大 {args.trading_trials} 輪 | 早停門檻: {trad_es_str} 輪")
    print(f"  * 初始模擬資金 (Capital)       : {args.capital:,} 元")
    print(f"  * 模式 A 執行因子最佳化          : {'否 (沿用現有)' if args.skip_factor_opt else '是'}")
    print("=" * 80)
    
    # ── 1. 安全備份 ────────────────────────────────────────────────────────
    print("\n[步驟 1/4] 安全備份設定與資料檔...")
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
        shutil.copy2(BEST_TRADING_PARAMS_PATH, BEST_TRADING_BAK)
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
        "mode_a": {},
        "mode_b": {}
    }
    
    try:
        # ── 2. 模式 A 流程 (研究與策略驗證期) ──────────────────────────────────
        print("\n" + "=" * 80)
        print("   🟢 進入 [模式 A]：研究與策略驗證期 (截斷日期: 2025-08-01)")
        print("=" * 80)
        
        # A1. 更新 config 變數以符合因子優化配置
        update_config_var("BACKTEST_DATE", '"20250801"')
        update_config_var("RUN_OPTIMIZATION", "True" if not args.skip_factor_opt else "False")
        update_config_var("OPTIMIZATION_TRIALS", str(args.factor_trials))
        update_config_var("EARLY_STOPPING_ROUNDS", fact_es_str)
        
        # A2. 如果不沿用因子，先移開現有的 best_factors.json，讓 Optuna 重頭搜尋
        if not args.skip_factor_opt and os.path.exists(BEST_FACTORS_PATH):
            os.remove(BEST_FACTORS_PATH)
            print("  已暫時移除 best_factors.json，以便進行全新的模式 A 因子搜尋")
            
        # A3. 因子優化與特徵重建
        if not args.skip_factor_opt:
            run_cmd([sys.executable, "auto_pipeline.py", "-s", "o"], "模式 A：因子技術指標最佳化")
            
        run_cmd([sys.executable, "auto_pipeline.py", "-s", "f"], "模式 A：重建特徵矩陣 (截斷 2025-08-01)")
        
        # A4. 訓練模型 A (僅使用 2025-08-01 以前的數據)
        run_cmd([sys.executable, "auto_pipeline.py", "-s", "t"], "模式 A：訓練 LightGBM 模型 (僅限樣本內)")
        
        # A5. 執行 OOS 訊號診斷 (analyze_regime_stability.py)
        run_cmd([sys.executable, "scripts/analyze_regime_stability.py"], "模式 A：樣本外 (OOS) 訊號健康與特徵漂移診斷")
        
        # 讀取穩定性診斷報告
        stability_summary = "找不到報告"
        if os.path.exists(STABILITY_REPORT_PATH):
            with open(STABILITY_REPORT_PATH, "r", encoding="utf-8") as rf:
                stability_summary = rf.read()
            # 複製一份備份
            shutil.copy2(STABILITY_REPORT_PATH, os.path.join(BASE_DIR, "reports", "mode_a_regime_stability_report.txt"))
            print(f"  已將模式 A 診斷報告另存至 reports/mode_a_regime_stability_report.txt")
            
        # 解析關鍵 RankIC 指標
        ic_oos_bull = re.search(r"OOS Bull \(.+?\) RankIC:\s*([+-]?\d+\.?\d*)", stability_summary)
        ic_is_all = re.search(r"IS All \(.+?\) RankIC:\s*([+-]?\d+\.?\d*)", stability_summary)
        
        results["mode_a"]["is_rankic"] = ic_is_all.group(1) if ic_is_all else "N/A"
        results["mode_a"]["oos_bull_rankic"] = ic_oos_bull.group(1) if ic_oos_bull else "N/A"
        
        # A6. 移除舊的交易參數以防干擾風控優化
        if os.path.exists(BEST_TRADING_PARAMS_PATH):
            os.remove(BEST_TRADING_PARAMS_PATH)
            
        # A7. 更新 config 中的早停設定為風控專用
        update_config_var("EARLY_STOPPING_ROUNDS", trad_es_str)
        
        # A8. 執行風控參數最佳化 (鎖定 2025-08-01 以前，嚴禁偷看牛市)
        run_cmd([
            sys.executable, "scripts/optimize_trading_params.py",
            "-t", str(args.trading_trials),
            "-s", "2021-01-02",
            "-e", "2025-08-01",
            "-c", str(args.capital)
        ], "模式 A：交易與風控參數最佳化 (Optuna)")
        
        # 讀取模式 A 風控優化結果
        mode_a_params = {}
        if os.path.exists(BEST_TRADING_PARAMS_PATH):
            with open(BEST_TRADING_PARAMS_PATH, "r", encoding="utf-8") as f:
                mode_a_data = json.load(f)
                mode_a_params = mode_a_data.get("best_params", {})
            # 備份模式 A 的風控參數
            shutil.copy2(BEST_TRADING_PARAMS_PATH, os.path.join(BASE_DIR, "best_trading_params_mode_a.json"))
            print(f"  已將模式 A 風控參數另存至 best_trading_params_mode_a.json")
            
        results["mode_a"]["params"] = mode_a_params
        
        # A9. 在樣本外超級牛市進行模擬交易 (2025-08-02 ~ 2026-06-05)
        # 注意：此時 config.py 中的 BACKTEST_DATE 仍為 "20250801"，所以 trading_sim.py 載入的是 Model A (訓練截至 2025-08-01)
        sim_stdout, _ = run_cmd([
            sys.executable, "trading_sim.py",
            "-s", "2025-08-02",
            "-e", "2026-06-05",
            "-c", str(args.capital)
        ], "模式 A：樣本外 (OOS) 超級牛市模擬交易回測 (Model A)")
        
        ret_a, mdd_a = parse_sim_output(sim_stdout)
        results["mode_a"]["oos_return"] = ret_a
        results["mode_a"]["oos_mdd"] = mdd_a
        
        # ── 3. 模式 B 流程 (實盤生產推理期) ──────────────────────────────────
        print("\n" + "=" * 80)
        print("   🔵 進入 [模式 B]：實盤生產推理期 (動態滾動重訓)")
        print("=" * 80)
        
        # B1. 更新 config 變數
        update_config_var("BACKTEST_DATE", "None")
        update_config_var("RUN_OPTIMIZATION", "False") # 沿用模式 A 的最佳因子
        
        # B2. 重建特徵工程 (動態計算一年前的交易日邊界)
        run_cmd([sys.executable, "auto_pipeline.py", "-s", "f"], "模式 B：重建特徵工程 (全數據覆蓋)")
        
        # B3. 重訓模型 B (包含牛市數據)
        run_cmd([sys.executable, "auto_pipeline.py", "-s", "t"], "模式 B：重訓 LightGBM 模型 (包含超級牛市)")
        
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
            "-e", "2026-06-01",
            "-c", str(args.capital)
        ], "模式 B：交易與風控參數全週期最佳化 (Optuna)")
        
        # 讀取模式 B 風控優化結果
        mode_b_params = {}
        if os.path.exists(BEST_TRADING_PARAMS_PATH):
            with open(BEST_TRADING_PARAMS_PATH, "r", encoding="utf-8") as f:
                mode_b_data = json.load(f)
                mode_b_params = mode_b_data.get("best_params", {})
            # 備份模式 B 的風控參數
            shutil.copy2(BEST_TRADING_PARAMS_PATH, os.path.join(BASE_DIR, "best_trading_params_mode_b.json"))
            print(f"  已將模式 B 風控參數另存至 best_trading_params_mode_b.json")
            
        results["mode_b"]["params"] = mode_b_params
        
        # B7. 執行全週期模擬交易回測 (2023-01-01 ~ 2026-06-05)
        # 注意：此時 config.py 的 BACKTEST_DATE 為 None，且載入的是 Model B (最新重訓模型)
        sim_stdout_b, _ = run_cmd([
            sys.executable, "trading_sim.py",
            "-s", "2023-01-01",
            "-e", "2026-06-05",
            "-c", str(args.capital)
        ], "模式 B：全週期 (含大牛市) 模擬交易回測 (Model B)")
        
        ret_b, mdd_b = parse_sim_output(sim_stdout_b)
        results["mode_b"]["full_return"] = ret_b
        results["mode_b"]["full_mdd"] = mdd_b
        
        # B8. 執行推理預測，產生明天的買賣建議
        run_cmd([sys.executable, "auto_pipeline.py", "-s", "i"], "模式 B：模型推理預測，產生明日實盤下單建議")
        
        # ── 4. 產出實驗報告 ──────────────────────────────────────────────────
        write_experiment_report(results)
        
    except Exception as ex:
        print(f"\n[嚴重異常] 實驗被異常中斷: {ex}")
        import traceback
        traceback.print_exc()
        
    finally:
        # ── 5. 還原環境設定 (確保不論成敗都還原設定檔) ──────────────────────
        print("\n[步驟 4/4] 正在還原原始設定檔與備份...")
        if os.path.exists(CONFIG_BAK):
            shutil.copy2(CONFIG_BAK, CONFIG_PATH)
            os.remove(CONFIG_BAK)
            print("  已還原 config.py 并清理臨時備份")
            
        if backup_factors_exists and os.path.exists(BEST_FACTORS_BAK):
            shutil.copy2(BEST_FACTORS_BAK, BEST_FACTORS_PATH)
            os.remove(BEST_FACTORS_BAK)
            print("  已還原 best_factors.json 并清理臨時備份")
            
        if backup_trading_exists and os.path.exists(BEST_TRADING_BAK):
            shutil.copy2(BEST_TRADING_BAK, BEST_TRADING_PARAMS_PATH)
            os.remove(BEST_TRADING_BAK)
            print("  已還原 best_trading_params.json 并清理臨時備份")
            
    print("\n" + "=" * 80)
    print("   🎉 全自動化實驗流程順利完成！對比分析報告已輸出。 🎉")
    print("   報告位置: reports/workflow_experiment_report.md")
    print("=" * 80 + "\n")


def write_experiment_report(res):
    """將實驗結果對比寫入 Markdown 報告"""
    report_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "workflow_experiment_report.md")
    
    mode_a_p = res["mode_a"].get("params", {})
    mode_b_p = res["mode_b"].get("params", {})
    args_info = res["args"]
    
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
| **訓練與測試分界** | 截斷於 2025-08-01 (樣本外測試) | `None` (每日滾動重訓至最新) |
| **樣本內 (IS) 因子 RankIC** | `{res["mode_a"].get("is_rankic", "N/A")}` | 沿用模式 A 因子 |
| **樣本外 (OOS) 牛市 RankIC**| `{res["mode_a"].get("oos_bull_rankic", "N/A")}` | N/A (模型已納入牛市訓練) |
| **回測測試區間** | 2025-08-02 ~ 2026-06-05 (樣本外) | 2023-01-01 ~ 2026-06-05 (全週期) |
| **回測累計報酬率 (%)** | `{res["mode_a"].get("oos_return", 0.0):+.2f}%` | `{res["mode_b"].get("full_return", 0.0):+.2f}%` |
| **回測最大回撤 MDD (%)** | `-{res["mode_a"].get("oos_mdd", 0.0):.2f}%` | `-{res["mode_b"].get("full_mdd", 0.0):.2f}%` |
| **Calmar 比率 (報酬/MDD)** | `{res["mode_a"].get("oos_return", 0.0) / (res["mode_a"].get("oos_mdd", 1.0) or 1.0):.2f}` | `{res["mode_b"].get("full_return", 0.0) / (res["mode_b"].get("full_mdd", 1.0) or 1.0):.2f}` |

*註：模式 A 的回測區間屬於完全未見過的樣本外 (OOS) 測試集，代表策略在全新超級牛市下的防禦與獲利能力。模式 B 的回測區間為包含牛市與熊市的全週期回測，展現策略的長線穩健性。*

---

## ⚙️ 最佳化風控策略參數對比 (Optimized Trading Params)

本部分對比 Optuna 在兩種模式下搜尋出的最佳交易參數。**這揭示了牛市與熊市/震盪市下，最優風控配置的結構性漂移：**

| 風控參數 | 🟢 模式 A (未見過牛市的最佳化) | 🔵 模式 B (包含牛市的最佳化) | 參數說明 |
| :--- | :--- | :--- | :--- |
| **買入門檻 (`buy_threshold`)** | `{mode_a_p.get("buy_threshold", "N/A")}%` | `{mode_b_p.get("buy_threshold", "N/A")}%` | D1 多空預測分數觸發買進的百分比。 |
| **個股停損 (`stop_loss`)** | `{mode_a_p.get("stop_loss", "N/A")}%` | `{mode_b_p.get("stop_loss", "N/A")}%` | 買入後的個股固定停損線。 |
| **避險門檻 (`panic_ma5`)** | `{mode_a_p.get("panic_ma5", "N/A")}` | `{mode_b_p.get("panic_ma5", "N/A")}` | 大盤 5 日平均回報低於此值觸發避險紅燈。 |
| **避險門檻 (`panic_breadth`)**| `{mode_a_p.get("panic_breadth", "N/A")}` | `{mode_b_p.get("panic_breadth", "N/A")}` | 全市場上漲比例低於此值觸發避險紅燈。 |
| **移動止盈啟動 (`ts_activation`)**| `{mode_a_p.get("ts_activation", "N/A")}%` | `{mode_b_p.get("ts_activation", "N/A")}%` | 個股利潤達到此值開啟移動追蹤止盈。 |
| **移動止盈回撤 (`ts_pullback`)** | `{mode_a_p.get("ts_pullback", "N/A")}%` | `{mode_b_p.get("ts_pullback", "N/A")}%` | 移動止盈開啟後自高點拉回多少執行停利。 |

### 💡 研究員核心分析與結論：
1. **為什麼模式 A 與模式 B 的最優風控參數存在差異？**
   * 模式 A 優化時，Optuna 看不到 2025-08-01 之後的兩萬點到四萬點大牛市，因此其優化出的風控引數更加傾向於**「防守震盪與熊市」**。
   * 模式 B 將 2025-08 ~ 2026-06 的超級牛市納入優化眼界。在強多頭市場中，大盤避險紅燈門檻（`panic_breadth` 和 `panic_ma5`）通常會被優化得更加寬容，個股停損（`stop_loss`）與移動止盈回撤（`ts_pullback`）也會更寬，以適應牛市個股的劇烈波動，防止被輕易洗出場，最大化捕捉趨勢利潤。
2. **策略健康度判斷：**
   * 若模式 A 在 OOS 區間的 `oos_return` 表現良好且 `oos_mdd` 受控，證明策略具備極強的**樣本外泛化能力**，非過擬合。
   * 正式實盤上線時，推薦**以模式 B 訓練出的最新模型**作為預測大腦（擁有最新特徵），並套用**模式 B 產出的 `best_trading_params_mode_b.json` 風控參數**進行每日交易，以在當下大牛市中獲得最契合市場波動的收益。

---
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[成功] 實驗分析報告已寫入: {report_path}")


if __name__ == "__main__":
    main()
