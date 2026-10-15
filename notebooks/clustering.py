import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import shutil

folder = "data/training_data/mixed_samples"
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

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_pca)

print("Cluster assignments:", clusters)
print("Cluster 0:", np.sum(clusters == 0), "anatomies")
print("Cluster 1:", np.sum(clusters == 1), "anatomies")

cluster_1_files = [f for f, c in zip(files, clusters) if c == 1]

data = [np.load(os.path.join(folder, f)) for f in cluster_1_files]

n_frames = 10
interval = 100  # 100 milliseconds, 10 FPS

sampled_data = [[d[frame][::5] for frame in range(n_frames)] for d in data]

fig = plt.figure(figsize=(24,15))
scatters = []

for i, (f, d) in enumerate(zip(cluster_1_files, sampled_data)):
    p = d[0]
    ax = fig.add_subplot(5, 8, i + 1, projection="3d")
    sc = ax.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        c=p[:, 3],
        s=0.5,
        vmin=min(frame[:, 3].min() for frame in d),
        vmax=max(frame[:, 3].max() for frame in d)
    )
    ax.set_title(
        f[:-4],
        color="red" if "pMI" in f else "black"
    )
    ax.set_axis_off()
    all_points = np.concatenate(d, axis=0)
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())
    scatters.append(sc)

frame_text = fig.suptitle("Frame 1 / 10", fontsize=16)

plt.tight_layout()

def update(frame):
    for sc, d in zip(scatters, sampled_data):
        p = d[frame]
        sc._offsets3d = (
            p[:, 0],
            p[:, 1],
            p[:, 2]
        )
        sc.set_array(p[:, 3])
    frame_text.set_text(f"Frame {frame + 1} / {n_frames}")
    return scatters + [frame_text]

ani = FuncAnimation(
    fig,
    update,
    frames=range(n_frames),
    interval=interval,
    blit=False,
    repeat=True,
    cache_frame_data=False
)

ani.save(
    "results/weird_samples_animation.gif",
    writer="pillow",
    fps=10,
    dpi=100
)

# Original dataset
folder = "data/training_data/mixed_samples"

# New directories
clean_folder = "data/training_data/clean_samples"
bad_folder = "data/training_data/impossible_samples"

os.makedirs(clean_folder, exist_ok=True)
os.makedirs(bad_folder, exist_ok=True)

# Cluster 1 contains impossible anatomies
bad_cluster = 1

# Separate filenames
valid_files = [f for f, c in zip(files, clusters) if c != bad_cluster]
invalid_files = [f for f, c in zip(files, clusters) if c == bad_cluster]

print("Valid anatomies:", len(valid_files))
print("Impossible anatomies:", len(invalid_files))

# Copy files into their respective directories
for f in valid_files:
    shutil.copy2(
        os.path.join(folder, f),
        os.path.join(clean_folder, f)
    )

for f in invalid_files:
    shutil.copy2(
        os.path.join(folder, f),
        os.path.join(bad_folder, f)
    )

print("Dataset successfully separated!")