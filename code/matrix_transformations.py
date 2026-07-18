import math

# === Базовые операции ===

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]


# === Демо 1: Базовые преобразования ===

print("=" * 50)
print("ДЕМО 1: Базовые преобразования")
print("=" * 50)

point = [1.0, 0.0]
angle = math.pi / 4  # 45 градусов

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Повернуть (1,0) на 45°: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Масштаб (1,1) на (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Сдвиг (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Отражение (2,1) через y: ({reflected[0]:.1f}, {reflected[1]:.1f})")


# === Демо 2: Композиция — порядок важен ===

print("\n" + "=" * 50)
print("ДЕМО 2: Композиция (порядок важен!)")
print("=" * 50)

R = rotation_2d(math.pi / 2)  # 90 градусов
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Повернуть 90° → Масштаб(2,0.5): ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Масштаб(2,0.5) → Повернуть 90°: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Одинаковые? {result1 == result2}")


# === Демо 3: Собственные числа и векторы ===

print("\n" + "=" * 50)
print("ДЕМО 3: Собственные числа и векторы")
print("=" * 50)

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Матрица: {A}")
print(f"Собственные числа: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"\n  λ={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A @ v = {[round(x,4) for x in result]}")
    print(f"    λ * v = {[round(x,4) for x in scaled]}")


# === Демо 4: Определитель ===

print("\n" + "=" * 50)
print("ДЕМО 4: Определитель = масштаб объёма")
print("=" * 50)

print(f"det(поворот 45°)    = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(масштаб 2,3)    = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(сдвиг kx=1)     = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(отражение y)    = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(вырожденная)    = {det_2x2(singular):.1f}")
print("  → Столбцы пропорциональны, пространство схлопывается в линию")


# === Демо 5: Круг из 8 точек через 3 преобразования ===

print("\n" + "=" * 50)
print("ДЕМО 5: 8 точек круга → поворот 30° → масштаб → сдвиг")
print("=" * 50)

# Создаём 8 точек на круге
angles = [i * math.pi / 4 for i in range(8)]
circle = [[math.cos(a), math.sin(a)] for a in angles]

# Три матрицы
R30 = rotation_2d(math.pi / 6)      # поворот 30°
S15 = scaling_2d(1.5, 0.8)          # масштаб
Sh = shearing_2d(0.3, 0)            # сдвиг

# Композиция: Sh @ S15 @ R30
composed = mat_mul(Sh, mat_mul(S15, R30))

print("Точка | До → После")
for i, p in enumerate(circle):
    transformed = mat_vec_mul(composed, p)
    print(f"  P{i}: ({p[0]:+.3f}, {p[1]:+.3f}) → ({transformed[0]:+.3f}, {transformed[1]:+.3f})")

det_composed = det_2x2(composed)
det_product = det_2x2(R30) * det_2x2(S15) * det_2x2(Sh)
print(f"\ndet(композиция) = {det_composed:.4f}")
print(f"det(R) * det(S) * det(Sh) = {det_product:.4f}")
print(f"Совпадают? {abs(det_composed - det_product) < 1e-10}")
