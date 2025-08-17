import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute the full Singular Value Decomposition (SVD) of a 2x2 real matrix.

    Args:
        A (np.ndarray): A 2x2 real matrix.

    Returns:
        tuple: (U, S, V_T)
            - U: 2x2 orthogonal matrix
            - S: 2x2 diagonal matrix with singular values in descending order
            - V_T: 2x2 orthogonal matrix (transpose of V)
    """
    if A.shape != (2, 2):
        return -1

    # Step 1: Compute AᵀA
    AtA = A.T @ A

    # Step 2: Eigen-decompose AᵀA using rotation
    if np.isclose(AtA[0, 0], AtA[1, 1]):
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan2(2 * AtA[0, 1], AtA[0, 0] - AtA[1, 1])
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    V = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])
    
    m = V.T @ AtA @ V
    singular_values = np.sqrt([m[0, 0], m[1, 1]])

    if singular_values[0] < singular_values[1]:
        singular_values = singular_values[::-1]
        V = V[:, ::-1]

    Σ_inv = np.diag(1 / singular_values)
    U = A @ V @ Σ_inv

    Σ = np.diag(singular_values)

    return (U, singular_values, V.T)
