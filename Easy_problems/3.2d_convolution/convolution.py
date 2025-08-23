import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Your code here
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1
    output_matrix = np.zeros((output_height, output_width))

    # Apply padding to the input matrix
    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')

    for i in range(0, output_height):
        for j in range(0, output_width):
            # Extract the current region
            region = padded_input[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
            # Apply the convolution operation (element-wise multiplication and sum)
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix
