def determinant_4x4(matrix: list[list[int|float]]) -> float:
	n = len(matrix)
	m = len(matrix[0])
	assert n == m 
	det = 0
	if n == 1:
		return matrix[0][0]
	else:
		for i in range(n):
			submatrix = [row[1:] for row in (matrix[:i] + matrix[i+1:])]
			det += ((-1)**(i)) * matrix[i][0] * determinant_4x4(submatrix)
		return det