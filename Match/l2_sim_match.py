import numpy as np
import matplotlib.pyplot as plt

npz1 = np.load("image_features_test_with_model_pred.npz", allow_pickle=True)
npz2 = np.load("hubert_testset_features_768_backbone_test_with_model_pred.npz", allow_pickle=True)

feat1 = npz1["vad_pred"]   # [N1, D]
paths1 = npz1["paths"]
feat2 = npz2["vad_pred"]   # [N2, D]
paths2 = npz2["paths"]

print("feat1:", feat1.shape, "feat2:", feat2.shape)

feat1_sq = np.sum(feat1**2, axis=1, keepdims=True)      # [N1,1]
feat2_sq = np.sum(feat2**2, axis=1, keepdims=True).T    # [1,N2]

dot = feat1 @ feat2.T                                    # [N1,N2]

dist = np.sqrt(np.maximum(feat1_sq + feat2_sq - 2*dot, 1e-12))

N1, N2 = dist.shape

top1_l2 = dist.min(axis=1)

worst1_l2 = dist.max(axis=1)

# Random baseline
rng = np.random.default_rng(42)
rand_idx = rng.integers(low=0, high=N2, size=N1)
rand_l2 = dist[np.arange(N1), rand_idx]

def print_stats(name, arr):
    print(f"\n== {name} ==")
    print(f"  mean     : {arr.mean():.4f}")
    print(f"  std      : {arr.std():.4f}")
    print(f"  min / max: {arr.min():.4f} / {arr.max():.4f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:02d}     : {np.percentile(arr, p):.4f}")

print_stats("Top-1 L2 distance", top1_l2)
print_stats("Worst-1 L2 distance", worst1_l2)
print_stats("Random-pair L2 distance", rand_l2)

np.savez("similarity_distributions_l2.npz",
         top1_l2=top1_l2,
         worst1_l2=worst1_l2,
         rand_l2=rand_l2)

plt.figure(figsize=(8,5))

bins = 40

plt.hist(rand_l2, bins=bins, alpha=0.5, label="Random pairs", color="gray")
plt.hist(worst1_l2, bins=bins, alpha=0.5, label="Worst-1 distance", color="r")
plt.hist(top1_l2, bins=bins, alpha=0.5, label="Top-1 distance", color="b")

plt.xlabel("L2 Distance", fontsize=12)
plt.ylabel("Count", fontsize=12)
#plt.title("Pre-VAD Embedding L2 Distance Distributions", fontsize=14)
plt.legend()

plt.tight_layout()
plt.savefig("vad_similarity_distribution_hist_l2.png", dpi=300)
plt.show()

print("\nHistogram saved to similarity_distribution_hist_l2.png")
