import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Generate sine-wave dataset.
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# Function to train and predict for a given polynomial degree.
def train_and_predict(degree):
	poly = PolynomialFeatures(degree=degree)
	X_poly = poly.fit_transform(X)

	model = LinearRegression()
	model.fit(X_poly, y)

	y_pred = model.predict(X_poly)
	mse = mean_squared_error(y, y_pred)

	return y_pred, mse


# Train models.
y_pred2, mse2 = train_and_predict(2)
y_pred5, mse5 = train_and_predict(5)

# Plot side-by-side comparison for degree 2 and degree 5.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, degree, mse in zip(
	axes,
	[y_pred2, y_pred5],
	[2, 5],
	[mse2, mse5],
):
	ax.scatter(X, y, s=10, label="Actual")
	ax.plot(X, y_pred, label="Predicted")
	ax.set_title(f"Degree = {degree} | MSE: {mse:.2e}")
	ax.legend()

plt.suptitle("Polynomial Regression Comparison", fontsize=14)
plt.tight_layout()

# Save comparison graph for submission/repo.
plt.savefig("polynomial_regression_degree2.png", dpi=200)
plt.close()

# Print exact MSE values.
print("MSE (Degree 2):", mse2)
print("MSE (Degree 5):", mse5)
