import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./data/training_data/mixed_demographics.csv")

continuous_df = df.select_dtypes(include="number").drop(columns=["index"])

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(continuous_df.columns),
    figsize=(3 * len(continuous_df.columns), 5)
)

for ax, col in zip(axes, continuous_df.columns):
    ax.violinplot(continuous_df[col].dropna())
    ax.set_title(col)
    ax.set_xticks([])
    ax.set_ylabel(['age (years)', 'BMI (kg/m^2)', 'height (cm)', 'weight (kg)', 'diastolic BP (mmHg)', 'systolic BP (mmHg)'][list(continuous_df.columns).index(col)])

fig.suptitle("Distribution of Continuous Variables", fontsize=16)

plt.tight_layout()
plt.savefig("./results/violin_plots.png")
plt.show()