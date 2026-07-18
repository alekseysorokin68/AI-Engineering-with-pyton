import math
import random

random.seed(42)


# ============================================================
#  Gaussian Elimination с частичным пивотингом
# ============================================================

def gaussian_elimination(A, b):
    n = len(b)
    # Копируем
    Ab = [row[:] + [bi] for row, bi in zip(A, b)]

    for k in range(n):
        # Частичный пивотинг
        max_val = abs(Ab[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(Ab[i][k]) > max_val:
                max_val = abs(Ab[i][k])
                max_row = i
        Ab[k], Ab[max_row] = Ab[max_row], Ab[k]

        if abs(Ab[k][k]) < 1e-12:
            raise ValueError(f"Матрица вырожденна на пивоте {k}")

        # Прямой ход
        for i in range(k + 1, n):
            m = Ab[i][k] / Ab[k][k]
            for j in range(k, n + 1):
                Ab[i][j] -= m * Ab[k][j]

    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i][n] - sum(Ab[i][j] * x[j] for j in range(i + 1, n))) / Ab[i][i]

    return x


# ============================================================
#  LU Decomposition
# ============================================================

def lu_decompose(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [row[:] for row in A]

    for i in range(n):
        L[i][i] = 1.0

    for k in range(n):
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U

def lu_solve(L, U, b):
    n = len(b)
    # Прямая подстановка: Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Обратная подстановка: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


# ============================================================
#  Cholesky Decomposition
# ============================================================

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if A[i][i] - s <= 0:
                    raise ValueError("Матрица не положительно определённая")
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L

def cholesky_solve(L, b):
    n = len(b)
    # Прямая: Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    # Обратная: L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]

    return x


# ============================================================
#  Normal Equations (Linear Regression)
# ============================================================

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matT_matMul(A, B):
    AT = transpose(A)
    rows, cols = len(AT), len(B[0])
    inner = len(AT[0])
    return [[sum(AT[i][k] * B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]

def mat_vecMul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def least_squares_normal(X, y):
    XtX = matT_matMul(X, X)
    # X^T y
    XT = transpose(X)
    Xty = [sum(XT[i][k] * y[k] for k in range(len(y))) for i in range(len(XT))]
    return gaussian_elimination(XtX, Xty)

def ridge_regression(X, y, lam):
    n = len(X[0])
    XtX = matT_matMul(X, X)
    # Добавляем lambda * I
    for i in range(n):
        XtX[i][i] += lam
    XT = transpose(X)
    Xty = [sum(XT[i][k] * y[k] for k in range(len(y))) for i in range(len(XT))]
    L = cholesky(XtX)
    return cholesky_solve(L, Xty)


# ============================================================
#  Condition Number
# ============================================================

def singular_values(A):
    """Упрощённое вычисление сингулярных значений через A^T A"""
    AtA = matT_matMul(A, A)
    n = len(AtA)
    # QR итерации (упрощённо)
    eigenvalues = []
    for i in range(n):
        eigenvalues.append(AtA[i][i])
    eigenvalues.sort(reverse=True)
    return [math.sqrt(max(0, ev)) for ev in eigenvalues]

def condition_number(A):
    svs = singular_values(A)
    if svs[-1] < 1e-10:
        return float('inf')
    return svs[0] / svs[-1]


# ============================================================
#  Вспомогательные
# ============================================================

def mat_mult(A, B):
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(cols_a)) for j in range(cols_b)] for i in range(rows_a)]

def vec_mult(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def residual_norm(A, x, b):
    Ax = vec_mult(A, x)
    return math.sqrt(sum((ax - bi)**2 for ax, bi in zip(Ax, b)))


# ============================================================
#  Демо 1: Gaussian Elimination
# ============================================================

print("=" * 55)
print("ДЕМО 1: Gaussian Elimination")
print("=" * 55)

A1 = [
    [2, 1, 1],
    [4, 3, 3],
    [2, 3, 1],
]
b1 = [8, 20, 12]

x1 = gaussian_elimination(A1, b1)
print(f"\nA = {A1}")
print(f"b = {b1}")
print(f"Решение: x = {[f'{xi:.4f}' for xi in x1]}")

# Проверка
Ax = vec_mult(A1, x1)
print(f"A @ x = {[f'{ai:.4f}' for ai in Ax]}")
print(f"Residual: {residual_norm(A1, x1, b1):.2e}")


# ============================================================
#  Демо 2: LU Decomposition
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: LU Decomposition")
print("=" * 55)

A2 = [
    [2, 1, 1],
    [4, 3, 3],
    [2, 3, 1],
]

L, U = lu_decompose(A2)
print(f"\nA = LU:")
print(f"L = [[{L[0][0]:.1f}, {L[0][1]:.1f}, {L[0][2]:.1f}]")
for row in L[1:]:
    print(f"     [{row[0]:.1f}, {row[1]:.1f}, {row[2]:.1f}]")

print(f"\nU = [[{U[0][0]:.1f}, {U[0][1]:.1f}, {U[0][2]:.1f}]")
for row in U[1:]:
    print(f"     [{row[0]:.1f}, {row[1]:.1f}, {row[2]:.1f}]")

# Проверка: L @ U ≈ A
LU = mat_mult(L, U)
print(f"\nL @ U ≈ A: {all(abs(LU[i][j] - A2[i][j]) < 1e-10 for i in range(3) for j in range(3))}")


# ============================================================
#  Демо 3: Cholesky Decomposition
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Cholesky Decomposition")
print("=" * 55)

A3 = [
    [4, 2],
    [2, 5],
]

L3 = cholesky(A3)
print(f"\nA = [[4, 2], [2, 5]]")
print(f"L = [[{L3[0][0]:.4f}, {L3[0][1]:.4f}]")
print(f"     [{L3[1][0]:.4f}, {L3[1][1]:.4f}]]")
print(f"\nL @ L^T = A: {all(abs(sum(L3[i][k]*L3[j][k] for k in range(2)) - A3[i][j]) < 1e-10 for i in range(2) for j in range(2))}")


# ============================================================
#  Демо 4: Least Squares — Linear Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Least Squares (Linear Regression)")
print("=" * 55)

# Генерируем данные: y = 2x + 1 + noise
random.seed(42)
n_samples = 20
X_data = [[1, i] for i in range(n_samples)]  # intercept + slope
y_data = [2 * i + 1 + random.gauss(0, 0.5) for i in range(n_samples)]

w_ols = least_squares_normal(X_data, y_data)
print(f"\ny = 2x + 1 + шум, {n_samples} сэмплов")
print(f"Решение (normal equations): w = [{w_ols[0]:.4f}, {w_ols[1]:.4f}]")
print(f"Ожидается:                 w = [1.0000, 2.0000]")


# ============================================================
#  Демо 5: Ridge Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Ridge Regression")
print("=" * 55)

w_ridge = ridge_regression(X_data, y_data, lam=1.0)
print(f"\nRidge (λ=1.0): w = [{w_ridge[0]:.4f}, {w_ridge[1]:.4f}]")
print(f"OLS:           w = [{w_ols[0]:.4f}, {w_ols[1]:.4f}]")
print(f"\n→ Ridge сдвигает веса к нулю (регуляризация)")


# ============================================================
#  Демо 6: Condition Number
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Condition Number")
print("=" * 55)

# Хорошо обусловленная
A_good = [[2, 0], [0, 1]]
# Плохо обусловленная
A_bad = [[1, 1], [1, 1.0001]]

print(f"\nХорошая: [[2,0],[0,1]]")
print(f"  κ ≈ {condition_number(A_good):.2f}")

print(f"\nПлохая: [[1,1],[1,1.0001]]")
print(f"  κ ≈ {condition_number(A_bad):.2e}")
print(f"  → Теряем ~8 знаков точности!")


# ============================================================
#  Демо 5: Проверка — OLS vs NumPy-style
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Проверка решений")
print("=" * 55)

# Система из 3 уравнений
A7 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10],
]
b7 = [6, 15, 27]

x_gauss = gaussian_elimination(A7, b7)
L7, U7 = lu_decompose(A7)
x_lu = lu_solve(L7, U7, b7)

print(f"\nСистема: [[1,2,3],[4,5,6],[7,8,10]] x = [6,15,27]")
print(f"Gaussian: x = {[f'{xi:.6f}' for xi in x_gauss]}")
print(f"LU:       x = {[f'{xi:.6f}' for xi in x_lu]}")
print(f"Residual (Gauss): {residual_norm(A7, x_gauss, b7):.2e}")
print(f"Residual (LU):    {residual_norm(A7, x_lu, b7):.2e}")
