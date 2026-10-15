import os
import re
import numpy as np
from sklearn.decomposition import PCA

folder = "data/training_data/clean_samples"
files = [f for f in os.listdir(folder) if f.endswith(".npy")]
files = sorted(files, key=lambda f: int(re.match(r"\d+", f).group()))

frame = 0

X = np.array([np.load(os.path.join(folder, f), mmap_mode="r")[frame, :, :3].ravel() for f in files])

print("X shape:", X.shape)

pca = PCA(n_components=0.95, svd_solver="full")

X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print("Variance retained:", sum(pca.explained_variance_ratio_))