
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import os
import re

folder = "data/training_data/mixed_samples"
files = [f for f in os.listdir(folder) if f.endswith(".npy")]
files = sorted(files, key=lambda f: int(re.match(r"\d+", f).group()))[:10]
data = [np.load(os.path.join(folder, f)) for f in files]

n_frames = 10
interval = 100  # 100 milliseconds, 10 FPS

sampled_data = [
    [d[frame][::5] for frame in range(n_frames)]
    for d in data
]

fig = plt.figure(figsize=(15, 8))
scatters = []

for i, (f, d) in enumerate(zip(files, sampled_data)):
    p = d[0]
    ax = fig.add_subplot(2, 5, i + 1, projection="3d")
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
    "results/10_samples_animation.gif",
    writer="pillow",
    fps=10,
    dpi=100
)

plt.show()