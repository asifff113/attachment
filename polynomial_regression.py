import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Generate sine-wave dataset.
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# Build polynomial regression model with degree 2.
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Predict and evaluate.
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)

# Plot actual vs predicted.
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=18, label="Actual", alpha=0.8)
plt.plot(X, y_pred, color="crimson", linewidth=2.5, label="Predicted")
plt.title(f"Polynomial Regression (degree=2) | MSE: {mse:.6f}")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

# Save graph for submission/repo and show it locally.
plt.savefig("polynomial_regression_degree2.png", dpi=200)
plt.close()

# Print MSE for the task requirement.
print(f"MSE (Degree 2): {mse}")
