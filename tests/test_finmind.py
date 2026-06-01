import os, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
fin_dir = os.path.join(ROOT_DIR, "data", "raw_financial")
files = [f for f in os.listdir(fin_dir) if f.startswith("1336")]
for f in sorted(files):
    path = os.path.join(fin_dir, f)
    mtime = os.path.getmtime(path)
    age_hours = (time.time() - mtime) / 3600
    print(f"{f:45s}  修改時間: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}  ({age_hours:.1f} 小時前)")