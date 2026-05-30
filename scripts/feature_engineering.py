import os
import pandas as pd

# 建立輸出資料夾
os.makedirs("data/features", exist_ok=True)

def load_target_stocks(file_path="Stocks.txt"):
    """讀取股票清單，過濾空白行"""
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到 {file_path}，將使用預設名單。")
        return ["2330"]
        
    with open(file_path, "r", encoding="utf-8") as f:
        # 去除換行符號與空白，並排除空字串
        stocks = [line.strip() for line in f.readlines() if line.strip()]
    return stocks

def build_features(date_str: str, target_stocks: list):
    """讀取當日原始 CSV，篩選出目標股票，並轉換成乾淨特徵"""
    price_file = f"data/raw_price/{date_str}_price.csv"
    chips_file = f"data/raw_chips/{date_str}_chips.csv"
    
    if not os.path.exists(price_file):
        return None
        
    # 1. 讀取並清洗股價
    df_p = pd.read_csv(price_file)
    df_p["證券代號"] = df_p["證券代號"].astype(str).str.strip()
    df_p = df_p[df_p["證券代號"].isin(target_stocks)]
    
    p_cols = {"證券代號": "stock_id", "開盤價": "open", "最高價": "high", "最低價": "low", "收盤價": "close", "成交股數": "volume"}
    df_p = df_p[[c for c in p_cols.keys() if c in df_p.columns]].rename(columns=p_cols)
    
    # 2. 讀取並清洗籌碼
    if os.path.exists(chips_file):
        df_c = pd.read_csv(chips_file)
        df_c["證券代號"] = df_c["證券代號"].astype(str).str.strip()
        df_c = df_c[df_c["證券代號"].isin(target_stocks)]
        
        c_cols = {"證券代號": "stock_id", "外陸資買賣超股數(不含外資自營商)": "foreign_buy", "投信買賣超股數": "sitc_buy"}
        df_c = df_c[[c for c in c_cols.keys() if c in df_c.columns]].rename(columns=c_cols)
    else:
        df_c = pd.DataFrame(columns=["stock_id", "foreign_buy", "sitc_buy"])
        
    # 3. 合併
    merged = pd.merge(df_p, df_c, on="stock_id", how="left").fillna(0)
    
    # 清洗數值
    for col in ["open", "high", "low", "close", "volume", "foreign_buy", "sitc_buy"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.replace(",", "")
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            
    merged["date"] = pd.to_datetime(date_str, format="%Y%m%d")
    
    # 範例因子計算
    merged["inst_ratio"] = (merged["foreign_buy"] + merged["sitc_buy"]) / (merged["volume"] + 1)
    merged["amplitude"] = (merged["high"] - merged["low"]) / (merged["open"] + 1)
    
    return merged

def process_all_history_features(start_date_obj, end_date_obj):
    """建置特徵大表並存為 Parquet 格式"""
    import datetime
    
    # 從外部檔案讀取股票清單
    target_stocks = load_target_stocks("Stocks.txt")
    print(f"📊 已載入 {len(target_stocks)} 檔目標股票，開始計算特徵...")
    
    delta = datetime.timedelta(days=1)
    curr = start_date_obj
    all_dfs = []
    
    while curr <= end_date_obj:
        date_str = curr.strftime("%Y%m%d")
        day_df = build_features(date_str, target_stocks)
        if day_df is not None and not day_df.empty:
            all_dfs.append(day_df)
        curr += delta
        
    if all_dfs:
        final_history_table = pd.concat(all_dfs, ignore_index=True)
        final_history_table = final_history_table.sort_values(by=["stock_id", "date"]).reset_index(drop=True)
        
        # 輸出為 Parquet 格式 (使用 pyarrow 引擎)
        save_path = "data/features/features_combined.parquet"
        final_history_table.to_parquet(save_path, engine="pyarrow", index=False)
        print(f"✅ 特徵矩陣建置完成！已存至 {save_path} (Parquet 格式)")
    else:
        print("❌ 找不到任何可轉特徵的原始資料。")