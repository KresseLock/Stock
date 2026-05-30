import os, time

fin_dir = r"D:\Vscode_workspace\Stock\data\raw_financial"
files = [f for f in os.listdir(fin_dir) if f.startswith("1336")]
for f in sorted(files):
    path = os.path.join(fin_dir, f)
    mtime = os.path.getmtime(path)
    age_hours = (time.time() - mtime) / 3600
    print(f"{f:45s}  修改時間: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}  ({age_hours:.1f} 小時前)")