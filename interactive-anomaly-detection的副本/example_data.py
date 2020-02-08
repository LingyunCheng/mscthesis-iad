import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

################### Loading the data
df = pd.read_csv("5544c26ecd130b116b28a36c.csv", index_col=0, parse_dates=True)
t1, t2 = datetime(2018, 3, 12), datetime(2018, 3, 17) # compressor failure period: from t1 to t2

################### Some plotting (time series)
print("df columns:", df.columns)
ax = df.plot(subplots=True) # Plot the data (with all features) over time
ax = df[t1 : t2].plot(subplots=True, color="black", ax=ax) # plot in black the period where the failure happened
plt.show()

################### Preparing the data matrix X and corresponding labels y
# Adding a column "labels" withs values -1 (normal) or +1 (abnormal/failure)
df.loc[:, "labels"] = -1
df.loc[t1 : t2, "labels"] = 1

X = df.values[:, :-2]  # We ignore the two last columns (corresponding to "OutsideTemp_mean" and "labels")
y = df["labels"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

################### More plotting (scatter plots ...)
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel("Compressor_mean")
ax1.set_ylabel("AditionalHeat_mean")

ax2.set_xlabel("Compressor_mean")
ax2.set_ylabel("HotWater_mean")

colors = df["OutsideTemp_mean"].values # Using the outside temperature as colors (to show some context)

cax = ax1.scatter(X[:, 0], X[:, 1], marker=".", c=colors, label="Normal")
cax = ax2.scatter(X[:, 0], X[:, 2], marker=".", c=colors, label="Normal")

X1 = X[y == 1]  # subset of anomalous data (where label is +1)
# Re-plotting anomalies with a different marker (*) and color (black)
ax1.scatter(X1[:, 0], X1[:, 1], marker="*", color="black", label="Compressor failure")
ax2.scatter(X1[:, 0], X1[:, 2], marker="*", color="black", label="Compressor failure")

plt.colorbar(cax)
plt.legend()
plt.show()
