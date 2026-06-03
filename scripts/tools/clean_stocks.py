import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_stocks(stocks_file="Stocks.txt", categories_file="stock_categories.json"):
    # 統一將 base_dir 設為專案根目錄 (即 scripts/tools/ 的上上上層)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stocks_path = os.path.join(base_dir, stocks_file)
    categories_path = os.path.join(base_dir, "scripts", categories_file)

    if not os.path.exists(categories_path):
        print(f"錯誤: 找不到分類檔案 {categories_path}")
        return
    if not os.path.exists(stocks_path):
        print(f"錯誤: 找不到股票檔案 {stocks_path}")
        return

    # 1. 載入 stock_categories.json 取得所有合法的股票代號與名稱
    with open(categories_path, "r", encoding="utf-8") as f:
        categories = json.load(f)
    
    valid_stocks = {}  # stock_id -> (name, category)
    for category, stocks in categories.items():
        for sid, name in stocks.items():
            valid_stocks[sid] = (name, category)
            
    # 2. 讀取 Stocks.txt
    with open(stocks_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 3. 處理每一行
    # stock_entries: stock_id -> (original_line, score)
    stock_entries = {}
    ignored_lines = []
    invalid_stocks_in_file = []
    duplicates_info = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            ignored_lines.append(line)
            continue
            
        parts = [p.strip() for p in stripped.split(",")]
        stock_id = parts[0]
        
        # 驗證代號是否存在於 stock_categories.json
        if stock_id not in valid_stocks:
            invalid_stocks_in_file.append(stripped)
            continue
            
        # 計算詳細度分數 (parts 的長度，欄位越多分數越高)
        score = len(parts)
        
        if stock_id in stock_entries:
            prev_line, prev_score = stock_entries[stock_id]
            duplicates_info.append({
                "stock_id": stock_id,
                "name": valid_stocks[stock_id][0],
                "prev": prev_line.strip(),
                "curr": stripped,
                "kept": stripped if score > prev_score else prev_line.strip()
            })
            if score > prev_score:
                stock_entries[stock_id] = (line, score)
        else:
            stock_entries[stock_id] = (line, score)

    # 4. 保持原本檔案中的出現順序 (排除重複和無效的股票)
    seen_ids = []
    new_lines = []
    
    # 保留原本的註解或空白行
    new_lines.extend(ignored_lines)
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        stock_id = parts[0]
        
        if stock_id in stock_entries and stock_id not in seen_ids:
            seen_ids.append(stock_id)
            orig_line, _ = stock_entries[stock_id]
            # 確保有換行符
            if not orig_line.endswith("\n"):
                orig_line += "\n"
            new_lines.append(orig_line)

    # 5. 寫回 Stocks.txt
    with open(stocks_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # 6. 印出詳細清理報告
    print("=" * 60)
    print("  股票清單 (Stocks.txt) 清理完成報告")
    print("=" * 60)
    print(f"原本行數: {len(lines)} 行")
    print(f"清理後行數: {len(new_lines)} 行")
    print(f"保留的有效股票數: {len(seen_ids)} 檔")
    print("-" * 60)

    if duplicates_info:
        print("【重複處理紀錄】:")
        for idx, info in enumerate(duplicates_info, 1):
            print(f"  {idx}. 股票: {info['stock_id']} ({info['name']})")
            print(f"     - 衝突項目 A: {info['prev']}")
            print(f"     - 衝突項目 B: {info['curr']}")
            print(f"     => 保留較完整/最新的項目: {info['kept']}")
            print()
    else:
        print("  未發現重複股票。")

    if invalid_stocks_in_file:
        print("-" * 60)
        print("【排除的無效股票代號】(不在 stock_categories.json 中):")
        for item in invalid_stocks_in_file:
            print(f"  - {item}")
    print("=" * 60)

if __name__ == "__main__":
    clean_stocks()
