import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df1 = pd.read_csv("data/zscores_from_primary_screening.csv")
df2 = pd.read_csv("results/zscore_intended_edits.csv")
df = df1.merge(df2, left_on="variant", right_on="variant", how="outer")
df = df.dropna()

slope, intercept = np.polyfit(df["zscore_weighted_fc_%reads"], df["zscore_primary_screening"], 1)
r_squared = r2_score(df["zscore_primary_screening"],df["zscore_weighted_fc_%reads"] * slope + intercept)
df.plot.scatter(x="zscore_weighted_fc_%reads", y="zscore_primary_screening")
for i,row in df.iterrows():
    plt.text(row["zscore_weighted_fc_%reads"], row["zscore_primary_screening"], row["variant"], alpha=0.5, fontsize=8)
plt.plot(df["zscore_weighted_fc_%reads"], df["zscore_weighted_fc_%reads"] * slope + intercept, color="red")
plt.title(f"r_squared: {round(r_squared, 4)}")
plt.grid("on")
plt.tight_layout()
plt.savefig("results/zscore_intended_edits_vs_zscore_primary_screening.png")
