# PCA of geometric point cloud

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

folder = "data/training_data/mixed_samples"
files = [f for f in os.listdir(folder) if f.endswith(".npy")]
files = sorted(files, key=lambda f: int(re.match(r"\d+", f).group()))

files_dem = "data/training_data/mixed_demographics.csv"
data_dem = pd.read_csv(files_dem)
sex = data_dem["sex"]

frame = 0  # try other frames from 0 to 9
X = np.array([np.load(os.path.join(folder, f), mmap_mode="r")[frame, :, :3].ravel()for f in files])
print("X shape:", X.shape) # Shape (900,54000)

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
print("X_pca shape:", X_pca.shape) # Shape (900,6)

scores_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
scores_df["filename"] = files

scores_df["label"] = scores_df["filename"].apply(
    lambda filename: "pMI" if "pMI" in filename else "healthy"
)

sns.set_style("white")

g = sns.pairplot(scores_df, corner=True)

# Add index labels to each scatter plot
for i in range(1, 6):
    for j in range(i):
        ax = g.axes[i, j]
        for idx, row in scores_df.iterrows():
            ax.annotate(
                str(idx),
                (row[f"PC{j+1}"], row[f"PC{i+1}"]),
                fontsize=5,
                alpha=0.7
            )

plt.savefig("results/pca_pairplot_sex_index.png", dpi=300)
plt.show()