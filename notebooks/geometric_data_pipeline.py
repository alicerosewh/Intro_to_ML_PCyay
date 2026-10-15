# PCA + getting ready for feature creation

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler


folder = "data/training_data/clean_samples"

########### Here add in any others that are weird to take out ################
files = [f for f in os.listdir(folder) if f.endswith(".npy")]
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

sample = np.load(os.path.join(folder, files[0]), mmap_mode="r")

values = np.unique(sample[:, :, 3])

print("Number of unique values:", len(values))
print("Values:", values)

def run_pca(folder, files, components=32):
    samples = [
        np.load(os.path.join(folder, filename), mmap_mode="r")
        for filename in files
    ]

    encoder = OneHotEncoder(sparse_output=False)

    region_arrays = [
        sample[:, :, 3].reshape(-1, 1)
        for sample in samples
    ]

    encoder.fit(np.vstack(region_arrays))

    X = np.array([
        np.concatenate([
            sample[:, :, :3].ravel(),
            encoder.transform(region).ravel()
        ])
        for sample, region in zip(samples, region_arrays)
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca, scaler, encoder

X_pca, pca, scaler, encoder = run_pca(folder, files, components=32)




