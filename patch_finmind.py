import os
from scripts.scraper import _crawl_fm_dataset, DATA_DIR, _polite_sleep

def patch():
    target_stocks = ["2049", "4931"]
    start_str = "2019-01-01"
    end_str = "2026-05-30"
    
    for stock_id in target_stocks:
        print(f"  抓取 FinMind {stock_id}...", end=" ", flush=True)
        results = []
        
        p_rev = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_monthly_revenue.csv")
        results.append(("營收", _crawl_fm_dataset("TaiwanStockMonthRevenue", stock_id, start_str, end_str, p_rev)))
        _polite_sleep(1, 2)
        
        p_stmt = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_financial_stmt.csv")
        results.append(("損益表", _crawl_fm_dataset("TaiwanStockFinancialStatements", stock_id, start_str, end_str, p_stmt)))
        _polite_sleep(1, 2)
        
        p_bal = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_balance_sheet.csv")
        results.append(("資產表", _crawl_fm_dataset("TaiwanStockBalanceSheet", stock_id, start_str, end_str, p_bal)))
        _polite_sleep(1, 2)
        
        p_cf = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_cashflow.csv")
        results.append(("現金流", _crawl_fm_dataset("TaiwanStockCashFlowsStatement", stock_id, start_str, end_str, p_cf)))
        _polite_sleep(1, 2)
        
        p_div = os.path.join(DATA_DIR, "raw_financial", f"{stock_id}_dividend.csv")
        results.append(("股利", _crawl_fm_dataset("TaiwanStockDividend", stock_id, start_str, end_str, p_div)))
        _polite_sleep(1, 2)
        
        errors  = [name for name, ok in results if ok is False]
        no_data = [name for name, ok in results if ok is None]
        
        if errors:
            print(f"抓取失敗: {errors}")
        elif no_data:
            print(f"OK (無資料: {no_data})")
        else:
            print("OK")

if __name__ == '__main__':
    patch()
