def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	# Direct method using the numpy library to compute eigenvalues
	# eigenvalues, _ = np.linalg.eig(np.array(matrix))
	# return eigenvalues.real.tolist()

    # Calculating the matrix's eigenvalues by solving the characteristic polynomial
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    det = a * d - b * c
    trace = a + d
    # The characteristic polynomial is λ^2 - (a+d)λ + (ad-bc) = 0
    # Solving for λ using the quadratic formula
    discriminant = trace**2 - 4 * det
    if discriminant < 0:
        # Complex eigenvalues
        real_part = trace / 2
        imaginary_part = ((-discriminant)**0.5) / 2
        sol_1 = complex(real_part, imaginary_part)
        sol_2 = complex(real_part, -imaginary_part)
        return [sol_1, sol_2]
    else:
        # Real eigenvalues
        return [(trace + discriminant**0.5) / 2, (trace - discriminant**0.5) / 2]
