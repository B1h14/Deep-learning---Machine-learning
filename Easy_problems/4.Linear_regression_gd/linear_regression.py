import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Ensure y is a column vector
	if y.ndim == 1:
		y = y.reshape(-1, 1)
	m, n = X.shape
	theta = np.zeros((n, 1))
	for _ in range(iterations):
		predictions = X @ theta
		errors = predictions - y
		gradient = (X.T @ errors) / m
		theta -= alpha * gradient
	return np.round(theta, 4)