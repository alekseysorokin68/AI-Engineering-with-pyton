import math
import random

random.seed(42)


# ============================================================
#  SVD через power iteration
# ============================================================

def power_iteration(M, num_iters=100):
    """Находит собственный вектор для максимального собственного числа"""
    n = len(M[0])
    v = [random.gauss(0, 1) for _ in range(n)]
    norm = math.sqrt(sum(x**2 for x in v))
    v = [x / norm for x in v]

    for _ in range(num_iters):
        # Mv
        Mv = [sum(M[i][j] * v[j] for j in range(n)) for i in range(len(M))]
        norm = math.sqrt(sum(x**2 for x in Mv))
        if norm < 1e-10:
            break
        v = [x / norm for x in Mv]

    # Eigenvalue = v^T M v
    Mv = [sum(M[i][j] * v[j] for j in range(n)) for i in range(len(M))]
    eigenvalue = sum(v[i] * Mv[i] for i in range(n))
    return eigenvalue, v


def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def mat_mul(A, B):
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_a))
    return result


def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def outer_product(u, v):
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]


def mat_norm(A):
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A)) for j in range(len(A[0]))))


def svd_power_iteration(A, k=None):
    """SVD через power iteration + дефляцию"""
    m, n = len(A), len(A[0])
    if k is None:
        k = min(m, n)

    A_res = [row[:] for row in A]
    sigmas = []
    us = []
    vs = []

    for _ in range(k):
        AtA = mat_mul(mat_transpose(A_res), A_res)
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = math.sqrt(eigenvalue)

        # u = A @ v / sigma
        Av = [sum(A_res[i][j] * v[j] for j in range(n)) for i in range(m)]
        u = [x / sigma for x in Av]

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        # Дефляция: A = A - sigma * u * v^T
        A_res = mat_sub(A_res, outer_product(u, [sigma * vi for vi in v]))

    return us, sigmas, vs


def reconstruct(us, sigmas, vs, m, n):
    """Восстановление матрицы из SVD"""
    result = [[0.0] * n for _ in range(m)]
    for k in range(len(sigmas)):
        for i in range(m):
            for j in range(n):
                result[i][j] += sigmas[k] * us[k][i] * vs[k][j]
    return result


# ============================================================
#  Вспомогательные функции
# ============================================================

def print_matrix(A, name="", precision=3):
    if name:
        print(f"{name}:")
    for row in A:
        print("  [" + ", ".join(f"{x:{precision+4}.{precision}f}" for x in row) + "]")


# ============================================================
#  Демо 1: SVD на маленькой матрице
# ============================================================

print("=" * 55)
print("ДЕМО 1: SVD через power iteration")
print("=" * 55)

A = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]

us, sigmas, vs = svd_power_iteration(A)

print(f"\nСингулярные значения: {[f'{s:.4f}' for s in sigmas]}")
print(f"Компонент U: {len(us)} векторов")
print(f"Компонент V: {len(vs)} векторов")

A_recon = reconstruct(us, sigmas, vs, 3, 3)
error = mat_norm(mat_sub(A, A_recon))
print(f"Ошибка реконструкции: {error:.6f}")


# ============================================================
#  Демо 2: Truncated SVD — сжатие
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Truncated SVD — разные ранги")
print("=" * 55)

# Большая матрица 6×5
random.seed(42)
m, n = 6, 5
B = [[random.gauss(0, 1) for _ in range(n)] for _ in range(m)]

us_full, sigmas_full, vs_full = svd_power_iteration(B)
print(f"\nПолные сингулярные значения: {[f'{s:.3f}' for s in sigmas_full]}")

for k in range(1, len(sigmas_full) + 1):
    B_k = reconstruct(us_full[:k], sigmas_full[:k], vs_full[:k], m, n)
    error = mat_norm(mat_sub(B, B_k)) / mat_norm(B)
    orig_size = m * n
    comp_size = k * (m + n + 1)
    ratio = comp_size / orig_size
    print(f"  k={k}: ошибка={error:.4f}, сжатие={ratio:.1%}")


# ============================================================
#  Демо 3: Сжатие изображения (симуляция)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Сжатие изображения (100×80)")
print("=" * 55)

# Генерируем "изображение" с структурой
random.seed(42)
img_rows, img_cols = 100, 80
img = []
for i in range(img_rows):
    row = []
    for j in range(img_cols):
        # Низкочастотный сигнал + шум
        val = math.sin(i / 10) * math.cos(j / 8) + random.gauss(0, 0.3)
        row.append(val)
    img.append(row)

us_img, sigmas_img, vs_img = svd_power_iteration(img)
print(f"Сингулярные значения (top 10): {[f'{s:.2f}' for s in sigmas_img[:10]]}")

for k in [1, 3, 5, 10, 20]:
    img_k = reconstruct(us_img[:k], sigmas_img[:k], vs_img[:k], img_rows, img_cols)
    error = mat_norm(mat_sub(img, img_k)) / mat_norm(img)
    orig = img_rows * img_cols
    comp = k * (img_rows + img_cols + 1)
    print(f"  k={k:>2d}: ошибка={error:.4f}, размер={comp}/{orig} ({comp/orig:.1%})")


# ============================================================
#  Демо 4: Шумоподавление
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Шумоподавление через SVD")
print("=" * 55)

# Чистый сигнал
random.seed(42)
clean = []
for i in range(50):
    row = []
    for j in range(40):
        val = math.sin(i / 5) * math.cos(j / 4)
        row.append(val)
    clean.append(row)

# Добавляем шум
noise_level = 0.5
noisy = [[clean[i][j] + random.gauss(0, noise_level)
          for j in range(40)] for i in range(50)]

us_n, sigmas_n, vs_n = svd_power_iteration(noisy)
print(f"Сингулярные значения (top 10): {[f'{s:.2f}' for s in sigmas_n[:10]]}")

error_noisy = mat_norm(mat_sub(noisy, clean))
print(f"\nОшибка (шумной - чистая): {error_noisy:.4f}")

for k in [1, 3, 5, 10]:
    denoised = reconstruct(us_n[:k], sigmas_n[:k], vs_n[:k], 50, 40)
    error_clean = mat_norm(mat_sub(denoised, clean))
    improvement = (1 - error_clean / error_noisy) * 100
    print(f"  k={k:>2d}: ошибка={error_clean:.4f}, улучшение={improvement:.1f}%")


# ============================================================
#  Демо 5: Псевдообратная (least squares)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Псевдообратная для least squares")
print("=" * 55)

# Переопределённая система: 3 уравнения, 2 неизвестных
# x + y = 3
# 2x + y = 5
# 3x + y = 7
A_ls = [
    [1.0, 1.0],
    [2.0, 1.0],
    [3.0, 1.0],
]
b_ls = [3.0, 5.0, 7.0]

us_ls, sigmas_ls, vs_ls = svd_power_iteration(A_ls)

# Псевдообратная: A+ = V Σ+ U^T
# Сначала Σ+ (транспонируем и инвертируем ненулевые)
sigma_inv = [1.0 / s for s in sigmas_ls]

# x = V Σ+ U^T b
# U^T b
Ut_b = [sum(us_ls[k][i] * b_ls[i] for i in range(3)) for k in range(len(sigmas_ls))]

# Σ+ (U^T b)
scaled = [sigma_inv[k] * Ut_b[k] for k in range(len(sigmas_ls))]

# V (scaled)
x = [sum(vs_ls[k][j] * scaled[k] for k in range(len(sigmas_ls))) for j in range(2)]

print(f"Система: x+y=3, 2x+y=5, 3x+y=7")
print(f"Решение (псевдообратная): x={x[0]:.4f}, y={x[1]:.4f}")

# Проверка: среднеквадратичная ошибка
residuals = [(A_ls[i][0]*x[0] + A_ls[i][1]*x[1] - b_ls[i]) for i in range(3)]
mse = sum(r**2 for r in residuals) / 3
print(f"MSE: {mse:.6f}")


# ============================================================
#  Демо 6: Рекомендательная система (концептуально)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Рекомендательная система (латентные факторы)")
print("=" * 55)

# Матрица рейтингов: 5 пользователей × 4 фильма
# (? = пропущенный рейтинг, заменяем средним)
ratings_raw = [
    [5.0, 3.0, 4.0, 4.0],
    [4.0, 0.0, 5.0, 3.0],
    [3.0, 1.0, 3.0, 5.0],
    [3.0, 3.0, 0.0, 3.0],
    [1.0, 5.0, 4.0, 2.0],
]

# Заполняем пропуски средними по строкам
ratings = []
for row in ratings_raw:
    known = [x for x in row if x > 0]
    mean = sum(known) / len(known) if known else 3.0
    ratings.append([x if x > 0 else mean for x in row])

print("Рейтинги (заполненные):")
users = ["User1", "User2", "User3", "User4", "User5"]
movies = ["Film1", "Film2", "Film3", "Film4"]
print(f"  {'':8s}", end="")
for m in movies:
    print(f"{m:>8s}", end="")
print()
for u, row in zip(users, ratings):
    print(f"  {u:8s}", end="")
    for v in row:
        print(f"{v:8.1f}", end="")
    print()

us_r, sigmas_r, vs_r = svd_power_iteration(ratings)
print(f"\nЛатентные факторы (singular values): {[f'{s:.2f}' for s in sigmas_r]}")
print(f"Rank-2 приближение: улавливает {sum(sigmas_r[:2])/sum(sigmas_r)*100:.0f}% структуры")


# ============================================================
#  Демо 7: Связь SVD с PCA
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: PCA = SVD на центрированных данных")
print("=" * 55)

# Генерируем данные 2D с корреляцией
random.seed(42)
n_samples = 200
X = []
for _ in range(n_samples):
    x1 = random.gauss(0, 2)
    x2 = 0.8 * x1 + random.gauss(0, 1)
    X.append([x1, x2])

# Центрируем
means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(2)]
X_centered = [[X[i][j] - means[j] for j in range(2)] for i in range(n_samples)]

# SVD
us_pca, sigmas_pca, vs_pca = svd_power_iteration(X_centered)

print(f"Сингулярные значения: {[f'{s:.2f}' for s in sigmas_pca]}")
print(f"Объяснённая дисперсия: {[f'{s**2/sum(s**2 for s in sigmas_pca):.2%}' for s in sigmas_pca]}")
print(f"\nPCA через SVD: principal components = правые сингулярные векторы V")
