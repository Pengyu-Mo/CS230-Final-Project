import os
import csv
from pathlib import Path

tsv_file = "../dataset/WikiArt-Emotions-Ag3.tsv"
img_root = "../dataset/images"

print("\n========== Checking missing images ==========")

# 读取 TSV 的所有 ID
ids = []
with open(tsv_file, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        ids.append(row["ID"])

print(f"Total IDs in TSV: {len(ids)}")

# 建立 images 文件夹中实际存在的文件映射（支持 jpg/jpeg/png/JPG/etc）
existing = {Path(p).stem for p in Path(img_root).rglob("*") if p.is_file()}

missing = [id for id in ids if id not in existing]

print(f"Images found     : {len(existing)}")
print(f"Missing images   : {len(missing)}")

if missing:
    print("\n=== Missing image IDs (first 20) ===")
    for mid in missing[:20]:
        print(mid)

    # 保存缺失列表
    with open("missing_images.txt", "w") as f:
        f.write("\n".join(missing))

    print("\n⚠️ Missing image list saved to: missing_images.txt")

else:
    print("\n✅ No missing images. All IDs have corresponding image files.")