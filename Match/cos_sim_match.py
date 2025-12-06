import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 读入两个 npz =====
npz1 = np.load("image_features_test_with_model_pred.npz", allow_pickle=True)
npz2 = np.load("hubert_testset_features_768_backbone_test_with_model_pred.npz", allow_pickle=True)

feat1 = npz1['features']   # [N1, D]
paths1 = npz1['paths']     # [N1]
feat2 = npz2['features']   # [N2, D]
paths2 = npz2['paths']     # [N2]

print("feat1:", feat1.shape, "feat2:", feat2.shape)

# ===== 2. L2 归一化 =====
def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

feat1_norm = l2_normalize(feat1, axis=1)
feat2_norm = l2_normalize(feat2, axis=1)

# ===== 3. 计算相似度矩阵：cosine = x @ y^T =====
sim = feat1_norm @ feat2_norm.T    # [N1, N2]

# ------------------------------------------------------------------
# Part A: 统计分布（Top-1, Worst-1, Random baseline）
# ------------------------------------------------------------------
N1, N2 = sim.shape

# 每个 query 的 top-1 similarity
top1_sims = sim.max(axis=1)              # [N1]

# 每个 query 的最小 similarity（最差匹配）
worst1_sims = sim.min(axis=1)            # [N1]

# 随机 baseline
rng = np.random.default_rng(42)
rand_idx = rng.integers(low=0, high=N2, size=N1)
rand_sims = sim[np.arange(N1), rand_idx]

def print_stats(name, arr):
    print(f"\n== {name} ==")
    print(f"  mean     : {arr.mean():.4f}")
    print(f"  std      : {arr.std():.4f}")
    print(f"  min / max: {arr.min():.4f} / {arr.max():.4f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:02d}     : {np.percentile(arr, p):.4f}")

print_stats("Top-1 similarity", top1_sims)
print_stats("Worst-1 similarity", worst1_sims)
print_stats("Random-pair similarity", rand_sims)

# 保存分布
np.savez("similarity_distributions_full.npz",
         top1_sims=top1_sims,
         worst1_sims=worst1_sims,
         rand_sims=rand_sims)

# ------------------------------------------------------------------
# Part B: 绘制直方图比较
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))

bins = 40

plt.hist(rand_sims, bins=bins, alpha=0.5, label="Random pairs", color="gray")
plt.hist(worst1_sims, bins=bins, alpha=0.5, label="Worst-1 similarity", color="r")
plt.hist(top1_sims, bins=bins, alpha=0.5, label="Top-1 similarity", color="b")

plt.xlabel("Cosine Similarity", fontsize=12)
plt.ylabel("Count", fontsize=12)
#plt.title("Pre-VAD Embedding Similarity Score Distributions", fontsize=14)
plt.legend()

plt.tight_layout()
plt.savefig("similarity_distribution_hist_full.png", dpi=300)
plt.show()

print("\nHistogram saved to similarity_distribution_hist_full.png")

# ------------------------------------------------------------------
# Part C: 随机输出 10 个 query 的 Top-1 匹配
# ------------------------------------------------------------------
num_samples = 40
rng = np.random.default_rng(123)

sample_indices = rng.choice(N1, size=num_samples, replace=False)

print("\n===== Random 10 Query Top-1 Matches =====")

for idx in sample_indices:
    row = sim[idx]                     # similarity row [N2]
    top1_idx = np.argmax(row)          # best match
    top1_sim = row[top1_idx]

    print(f"\nQuery image  : {paths1[idx]}")
    print(f"Matched audio: {paths2[top1_idx]}")
    print(f"Cosine similarity = {top1_sim:.4f}")
