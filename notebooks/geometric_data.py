import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import re
import plotly.io as pio

"""
# One investigation

mi_sample = np.load("data/training_data/mixed_samples/2pMI.npy")
# mi_sample shape is (10,18000,4) - 10 time frames, 18000 points on the hearts surface, 4 columns (x,y,z, part of heart)

print("unique values in column 3:", len(np.unique(mi_sample[0, :, 3])))

vals, counts = np.unique(mi_sample[0, :, 3], return_counts=True)
print(vals, counts)

pts = mi_sample[0]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3], s=1)
plt.show()

"""

folder = "data/training_data/mixed_samples"
files = [f for f in os.listdir(folder) if f.endswith(".npy")]
files = sorted(files, key=lambda f: int(re.match(r"\d+", f).group()))[:10]
data = [np.load(os.path.join(folder, f)) for f in files]

frame = 0  # try other frames from 0 to 9
fig = plt.figure(figsize=(15, 8))
for i, (f, d) in enumerate(zip(files, data)):
    p = d[frame][::5]
    ax = fig.add_subplot(2, 5, i + 1, projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 3], s=0.5)
    ax.set_title(f[:-4], color="red" if "pMI" in f else "black")
    ax.set_axis_off()
plt.tight_layout()
plt.savefig("results/first_timestep_10_samples.png", dpi=300)
plt.show()


