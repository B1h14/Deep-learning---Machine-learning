import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    # calculate initial residual vector
    if x0 is not None:
        x_k = x0
    else:
        x_k = np.zeros_like(b) #initial guess
    r_k = b - A @ x_k # initial residual vector
    p_k = r_k.copy() # initial search direction
    for k in range(n):
        alpha_k = r_k.T @ r_k / (p_k.T @ A @ p_k) # step size
        x_k = x_k + alpha_k * p_k # update solution
        r_k_ = r_k - alpha_k * (A @ p_k) # update residual
        if np.linalg.norm(r_k_) < tol: # check for convergence
            break
        beta_k = r_k_.T @ r_k_ / (r_k.T @ r_k) # update parameter
        p_k = r_k_ + beta_k * p_k # update search direction
        r_k = r_k_
    return x_k