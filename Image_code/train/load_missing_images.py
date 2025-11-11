#!/usr/bin/env python3
import csv
import os
import requests

# ==============================
# ✅ MODIFY THESE THREE PATHS
# ==============================
TSV_INFO = "../dataset/WikiArt-info.tsv"     # 你上传的 info 文件
MISSING_IDS_FILE = "missing_images.txt"     # 保存缺失 id 的 txt
SAVE_DIR = "../dataset/images"              # 下载到你的 images folder
# ==============================


# 读取 missing ids
with open(MISSING_IDS_FILE, "r", encoding="utf-8") as f:
    missing_ids = [line.strip() for line in f if line.strip()]


print(f"[INFO] Missing id count: {len(missing_ids)}")


# 读取 TSV Image info
print("[INFO] Loading WikiArt-info.tsv ...")
with open(TSV_INFO, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

# ---- 自动识别 URL column ----
url_col = None
for c in reader.fieldnames:
    if "url" in c.lower():
        url_col = c
        break

if url_col is None:
    raise RuntimeError("❌ 没找到 URL 列，检查 TSV header！")


# id → url 映射
id_to_url = {}
for r in rows:
    _id = r.get("ID", "").strip()
    if _id in missing_ids:
        id_to_url[_id] = r[url_col].strip()


print(f"[INFO] Found {len(id_to_url)} urls in TSV")


# ======== 开始下载 ========
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

for img_id, url in id_to_url.items():

    save_path = os.path.join(SAVE_DIR, img_id + ".jpg")

    if os.path.exists(save_path):
        print(f"[SKIP] exists: {save_path}")
        continue

    try:
        print(f"[DOWNLOADING] {img_id} → {save_path}")
        resp = requests.get(url, headers=headers, timeout=20)

        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(resp.content)
        else:
            print(f"[FAIL] HTTP {resp.status_code} for {img_id}")

    except Exception as e:
        print(f"[ERROR] failed: {img_id} ({e})")

print("\n✅ DONE: images saved to", SAVE_DIR)