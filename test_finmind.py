# 快速驗證新邏輯
import os

def _already_exists(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            header = f.readline()
            data   = f.readline()
        return bool(header.strip()) and bool(data.strip())
    except Exception:
        return False

fin_dir = r"D:\Vscode_workspace\Stock\data\raw_financial"
for fname in sorted(os.listdir(fin_dir))[:5]:
    path = os.path.join(fin_dir, fname)
    print(f"{fname:40s}  exists={_already_exists(path)}")