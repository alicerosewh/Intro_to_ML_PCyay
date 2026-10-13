import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import plotly.io as pio


mi_sample = np.load("data/training_data/mixed_samples/2pMI.npy")
pts = mi_sample[0]

fig = go.Figure(go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode="markers",
    marker=dict(size=2, color=pts[:, 3], colorscale="Viridis", showscale=True),
))
fig.show()