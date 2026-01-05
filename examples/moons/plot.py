import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the original data with its predictions
df_original = pd.read_csv("moons_original_preds.csv")

# Load the boundary data
df_boundary = pd.read_csv("moons_boundary_preds.csv")

# Prepare data for contour plot
grid_size = int(np.sqrt(len(df_boundary)))
xx = df_boundary["x"].values.reshape(grid_size, grid_size)
yy = df_boundary["y"].values.reshape(grid_size, grid_size)
Z = df_boundary["pred"].values.reshape(grid_size, grid_size)

# Create the plot
plt.figure(figsize=(10, 8))

# Plot the decision boundary and filled regions
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
plt.colorbar(label="Model Prediction Score")

error = (df_original["pred"] - df_original["actual"]) ** 2
# Plot the original data points
# Class 1
plt.scatter(
    df_original[df_original["actual"] == 1]["x"],
    df_original[df_original["actual"] == 1]["y"],
    s=20 + error[df_original["actual"] == 1] * 8,
    c="blue",
    edgecolors="k",
    label="Actual Class 1",
)
# Class -1
plt.scatter(
    df_original[df_original["actual"] == -1]["x"],
    df_original[df_original["actual"] == -1]["y"],
    s=20 + error[df_original["actual"] == -1] * 8,
    c="red",
    edgecolors="k",
    label="Actual Class -1",
)

# Finish
plt.title("CppGrad Model Decision Boundary on Moons Dataset")
plt.xlabel("Feature 1 (x)")
plt.ylabel("Feature 2 (y)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
