import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import plotly.io as pio


mi_sample = np.load("data/training_data/mixed_samples/2pMI.npy")
pts = mi_sample[0]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3], s=1)
plt.show()