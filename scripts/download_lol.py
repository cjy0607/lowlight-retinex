"""从 HuggingFace 镜像下载 LOLv1 的 15 组测试图与 60 组训练图。

用法：python scripts/download_lol.py [--train 60]
"""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://hf-mirror.com/datasets/Benxelua/LOL_v1"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "LOL")


def ls(path):
    url = f"{BASE}/tree/main/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    return [x["path"] for x in data if x["type"] == "file" and x["path"].endswith(".png")]


def get(path):
    out = os.path.join(ROOT, path)
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    req = urllib.request.Request(f"{BASE}/resolve/main/{path}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
        f.write(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    a = ap.parse_args()
    ev = ls("eval15/low") + ls("eval15/high")
    tr = ls("our485/low") + ls("our485/high")
    names = sorted({os.path.basename(p) for p in tr}, key=lambda s: int(s.split(".")[0]))[: a.train]
    tr = [p for p in tr if os.path.basename(p) in set(names)]
    print(f"eval {len(ev)} files, train {len(tr)} files")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(get, ev + tr))
    print("done ->", ROOT)


if __name__ == "__main__":
    main()
