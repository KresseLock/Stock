import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def _norm_num(s):
    """把數字欄位正規化為 float 以利比對（"24.450"=="24.45"，"9000"=="9000.0"）；非數字保留原字串。"""
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return s


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

    # 3. 逐行解析為「買進批次 (lot)」。支援同代號多筆（不同時期／價位買進）。
    ignored_lines = []            # 註解／空白行
    invalid_stocks_in_file = []   # 不在 stock_categories.json 的代號
    entries = []                  # 依原始順序保留的每筆有效 lot

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

        cost = parts[1] if len(parts) >= 2 and parts[1] else None
        shares = parts[2] if len(parts) >= 3 and parts[2] else None
        buy_date = parts[4] if len(parts) >= 5 and parts[4] else None
        entries.append({
            "stock_id": stock_id,
            "cost": _norm_num(cost),
            "shares": _norm_num(shares),
            "buy_date": buy_date,
            "is_holding": cost is not None,
            "field_count": len(parts),
            "line": line,
            "stripped": stripped,
        })

    # 4. 去重（lot 感知）：
    #    - 純自選（無成本）若該代號已有持倉 → 視為冗餘自選，移除。
    #    - 完全相同的 lot（同代號/成本/股數/買入日）→ 只留欄位最完整者。
    #    - 不同成本或不同買入日 → 視為不同買進批次 (lot)，全部保留。
    sid_has_holding = {e["stock_id"] for e in entries if e["is_holding"]}
    kept = []                 # 依原始順序保留的 lot
    seen_identity = {}        # identity -> kept 索引
    duplicates_info = []      # 完全重複行（已合併）
    redundant_bare_info = []  # 已有持倉、又出現純代號自選

    for e in entries:
        if not e["is_holding"] and e["stock_id"] in sid_has_holding:
            redundant_bare_info.append(e["stripped"])
            continue

        identity = (e["stock_id"], e["cost"], e["shares"], e["buy_date"])
        if identity in seen_identity:
            prev = kept[seen_identity[identity]]
            keep_curr = e["field_count"] > prev["field_count"]
            duplicates_info.append({
                "stock_id": e["stock_id"],
                "name": valid_stocks[e["stock_id"]][0],
                "prev": prev["stripped"],
                "curr": e["stripped"],
                "kept": e["stripped"] if keep_curr else prev["stripped"],
            })
            if keep_curr:
                kept[seen_identity[identity]] = e
        else:
            seen_identity[identity] = len(kept)
            kept.append(e)

    # 5. 組出寫回內容（保留註解 + 依原始順序的 lot）
    new_lines = []
    new_lines.extend(ignored_lines)
    for e in kept:
        orig_line = e["line"]
        if not orig_line.endswith("\n"):
            orig_line += "\n"
        new_lines.append(orig_line)

    with open(stocks_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # 6. 印出詳細清理報告
    distinct_sids = len({e["stock_id"] for e in kept})
    print("=" * 60)
    print("  股票清單 (Stocks.txt) 清理完成報告")
    print("=" * 60)
    print(f"原本行數: {len(lines)} 行")
    print(f"清理後行數: {len(new_lines)} 行")
    print(f"保留的有效買進批次: {len(kept)} 筆 ({distinct_sids} 檔)")
    print("-" * 60)

    if duplicates_info:
        print("【完全重複行（已合併，同代號/成本/股數/買入日）】:")
        for idx, info in enumerate(duplicates_info, 1):
            print(f"  {idx}. 股票: {info['stock_id']} ({info['name']})")
            print(f"     - 衝突項目 A: {info['prev']}")
            print(f"     - 衝突項目 B: {info['curr']}")
            print(f"     => 保留較完整的項目: {info['kept']}")
            print()
    else:
        print("  未發現完全重複行。")

    if redundant_bare_info:
        print("-" * 60)
        print("【移除的冗餘自選代號】(該代號已有持倉紀錄，純代號自選為多餘):")
        for item in redundant_bare_info:
            print(f"  - {item}")

    if invalid_stocks_in_file:
        print("-" * 60)
        print("【排除的無效股票代號】(不在 stock_categories.json 中):")
        for item in invalid_stocks_in_file:
            print(f"  - {item}")
    print("=" * 60)


if __name__ == "__main__":
    clean_stocks()
