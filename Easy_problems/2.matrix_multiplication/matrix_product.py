def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    n = len(a)
    m = len(b[0])
    p = len(b)
    assert len(a[0]) == p, "Incompatible matrices for multiplication"
    result = [[0 for _ in range(m)] for __ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += a[i][k] * b[k][j]
    return result
