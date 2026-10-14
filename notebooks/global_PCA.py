import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

folder = "data/training_data/mixed_samples"
files = sorted([f for f in os.listdir(folder) if f.endswith(".npy") and f != "658healthy.npy"], key=lambda f: int(re.match(r"\d+", f).group()))
data = [np.load(os.path.join(folder, f)) for f in files]

def extract_frame_features(frame):
    xyz = frame[:, :3]
    intensity = frame[:, 3]

    mean_xyz = np.mean(xyz, axis=0)
    std_xyz = np.std(xyz, axis=0)

    centroid = mean_xyz
    distances = np.linalg.norm(xyz - centroid, axis=1)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    mean_intensity = np.mean(intensity)
    std_intensity = np.std(intensity)

    features = np.concatenate([
        mean_xyz,                  
        std_xyz,                   
        [mean_distance],           
        [std_distance],            
        [mean_intensity],          
        [std_intensity]            
    ])

    return features

def extract_sample_features(sample):
    frame_features = np.array([extract_frame_features(frame) for frame in sample[:10]])

    temporal_differences = np.diff(frame_features,axis=0)

    spatial_features = frame_features.flatten()

    temporal_features = temporal_differences.flatten()

    sample_features = np.concatenate([spatial_features, temporal_features])

    return sample_features

X = np.array([extract_sample_features(sample) for sample in data])

y = np.array(["pMI" if "pMI" in f else "Non-pMI" for f in files])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=6)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance ratios:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {variance*100:.2f}%")

print(
    f"\nTotal explained variance: "
    f"{pca.explained_variance_ratio_.sum()*100:.2f}%"
)

pc_names = [f"PC{i+1}" for i in range(6)]

df_pca = pd.DataFrame(X_pca, columns=pc_names)

df_pca["Class"] = y

sns.set_style("white")

g = sns.pairplot(df_pca, vars=pc_names, hue="Class",
    palette={"pMI": "red", "Non-pMI": "blue"},
    plot_kws={"alpha": 0.6, "s": 20, "edgecolor": "none"},
    diag_kws={"alpha": 0.5},
    height=2.2,
    corner=True)

for i in range(6):
    variance = pca.explained_variance_ratio_[i] * 100
    g.axes[-1, i].set_xlabel(f"PC{i+1} ({variance:.1f}%)")

    g.axes[i, 0].set_ylabel(f"PC{i+1} ({variance:.1f}%)")

g.figure.suptitle(
    "Sample-Level Temporal PCA (6 Components)", fontsize=16, y=1.02)

g.savefig(
    "results/sample_temporal_pca_pairplot.png", dpi=300, bbox_inches="tight")

plt.show()