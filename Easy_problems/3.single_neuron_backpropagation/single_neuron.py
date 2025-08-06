import numpy as np
def sigmoid(x: float) -> float:
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = initial_weights
    bias = initial_bias
    mse_values = []

    for epoch in range(epochs):
        # Forward pass
        linear_output = np.dot(features, weights) + bias
        predictions = sigmoid(linear_output)

        # Compute loss (Mean Squared Error)
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(mse)

        # Backward pass (Gradient Descent)
        error = predictions - labels
        weights_gradient = 2*np.dot(features.T, error * predictions * (1 - predictions)) / len(labels)
        bias_gradient = np.mean(error * predictions * (1 - predictions))

        # Update weights and bias
        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient

    return weights, bias, mse_values