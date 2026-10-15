# PCA + getting ready for feature creation

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


folder = "data/training_data/mixed_samples"

########### Here add in any others that are weird to take out ################
files = [f for f in os.listdir(folder) if f.endswith(".npy") and f != "658healthy.npy"]
files = sorted(files, key=lambda f: int(re.match(r"\d+", f).group()))

files_dem = "data/training_data/mixed_demographics.csv"
data_dem = pd.read_csv(files_dem, index_col=0)

file_ids = [
    int(re.match(r"\d+", filename).group())
    for filename in files
]

data_dem = data_dem.loc[file_ids].reset_index(drop=True)

print("Number of files:", len(files))
print("Number of demographic rows:", len(data_dem))


sex = data_dem["sex"]



def run_pca(folder, files, components=32):
    X = np.array([np.load(os.path.join(folder, f), mmap_mode="r")[:, :, :].ravel()for f in files])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca, scaler

X_pca, pca, scaler = run_pca(folder, files, components=32)

scores_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
scores_df["filename"] = files

scores_df["label"] = scores_df["filename"].apply(
    lambda filename: "pMI" if "pMI" in filename else "healthy"
)


scores_df["sex"] = sex

