import math
import random

random.seed(42)


# ============================================================
#  PCA с нуля
# ============================================================

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        n_samples = len(X)
        n_features = len(X[0])

        # Центрирование
        self.mean = [sum(X[i][j] for i in range(n_samples)) / n_samples
                     for j in range(n_features)]
        X_centered = [[X[i][j] - self.mean[j] for j in range(n_features)]
                      for i in range(n_samples)]

        # Ковариационная матрица
        cov = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                cov[i][j] = sum(X_centered[k][i] * X_centered[k][j]
                               for k in range(n_samples)) / (n_samples - 1)

        # Собственные числа и векторы (симметричная матрица)
        eigenvalues, eigenvectors = self._eigh(cov)

        # Сортировка по убыванию
        idx = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
        eigenvalues = [eigenvalues[i] for i in idx]
        eigenvectors = [eigenvectors[i] for i in idx]

        self.eigenvalues = eigenvalues[:self.n_components]
        self.components = [eigenvectors[i][:self.n_components] for i in range(n_features)]
        # components[j][k] = j-й признак в k-й компоненте

        total_var = sum(eigenvalues)
        self.explained_variance_ratio_ = [ev / total_var for ev in self.eigenvalues]

        return self

    def transform(self, X):
        n_samples = len(X)
        n_features = len(X[0])
        X_centered = [[X[i][j] - self.mean[j] for j in range(n_features)]
                      for i in range(n_samples)]

        result = []
        for i in range(n_samples):
            row = []
            for k in range(self.n_components):
                val = sum(X_centered[i][j] * self.components[j][k]
                         for j in range(n_features))
                row.append(val)
            result.append(row)
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _eigh(self, matrix):
        """Собственные числа симметричной матрицы (QR-алгоритм)"""
        n = len(matrix)
        A = [row[:] for row in matrix]
        eigenvectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for _ in range(100):
            # QR разложение (упрощённое для симметричных)
            Q, R = self._qr(A)
            A = [[sum(R[i][k] * Q[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
            eigenvectors = [[sum(eigenvectors[i][k] * Q[j][k] for k in range(n))
                            for j in range(n)] for i in range(n)]

        eigenvalues = [A[i][i] for i in range(n)]
        return eigenvalues, eigenvectors

    def _qr(self, A):
        """QR разложение через Грамма-Шмидта"""
        n = len(A)
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]

        for j in range(n):
            v = [A[i][j] for i in range(n)]
            for k in range(j):
                dot = sum(Q[i][k] * A[i][j] for i in range(n))
                R[k][j] = dot
                v = [v[i] - dot * Q[i][k] for i in range(n)]
            norm = math.sqrt(sum(x**2 for x in v))
            R[j][j] = norm
            if norm > 1e-10:
                for i in range(n):
                    Q[i][j] = v[i] / norm

        return Q, R


# ============================================================
#  Вспомогательные функции
# ============================================================

def generate_spiral(n_points=200, noise=0.5):
    """Две спирали — нелинейные данные"""
    points = []
    labels = []
    for i in range(n_points):
        t = i / n_points * 3 * math.pi + random.gauss(0, noise * 0.1)
        # Спираль 1
        x1 = t * math.cos(t) + random.gauss(0, noise)
        y1 = t * math.sin(t) + random.gauss(0, noise)
        points.append([x1, y1])
        labels.append(0)
        # Спираль 2 (сдвинутая)
        x2 = t * math.cos(t + math.pi) + random.gauss(0, noise)
        y2 = t * math.sin(t + math.pi) + random.gauss(0, noise)
        points.append([x2, y2])
        labels.append(1)
    return points, labels

def generate_concentric_circles(n_points=200, noise=0.1):
    """Два концентрических кольца"""
    points = []
    labels = []
    for i in range(n_points):
        angle = random.uniform(0, 2 * math.pi)
        # Внутреннее кольцо
        r1 = 1.0 + random.gauss(0, noise)
        points.append([r1 * math.cos(angle), r1 * math.sin(angle)])
        labels.append(0)
        # Внешнее кольцо
        r2 = 3.0 + random.gauss(0, noise)
        points.append([r2 * math.cos(angle), r2 * math.sin(angle)])
        labels.append(1)
    return points, labels

def mean_squared_error(original, reconstructed):
    total = 0
    for orig, recon in zip(original, reconstructed):
        total += sum((o - r)**2 for o, r in zip(orig, recon))
    return total / (len(original) * len(original[0]))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ============================================================
#  Демо 1: PCA на синтетических данных
# ============================================================

print("=" * 55)
print("ДЕМО 1: PCA на синтетических 3D данных")
print("=" * 55)

# Генерируем данные: x1, x2, x3 = 0.5*x1 + 0.3*x2 + шум
random.seed(42)
n = 300
X_synth = []
for _ in range(n):
    x1 = random.gauss(0, 2)
    x2 = random.gauss(0, 1)
    x3 = 0.5 * x1 + 0.3 * x2 + random.gauss(0, 0.5)
    X_synth.append([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synth)

print(f"Исходные данные: {len(X_synth)}×3")
print(f"После PCA:       {len(X_reduced)}×2")
print(f"Explained variance: {[f'{v:.4f}' for v in pca.explained_variance_ratio_]}")
print(f"Суммарно: {sum(pca.explained_variance_ratio_):.4f}")


# ============================================================
#  Демо 2: Reconstruction error
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Reconstruction error")
print("=" * 55)

# Восстанавливаем данные из 2D
mean = pca.mean
components = pca.components
X_reconstructed = []
for i in range(len(X_reduced)):
    row = []
    for j in range(3):
        val = mean[j] + sum(X_reduced[i][k] * components[j][k] for k in range(2))
        row.append(val)
    X_reconstructed.append(row)

mse = mean_squared_error(X_synth, X_reconstructed)
print(f"MSE восстановления: {mse:.6f}")
print(f"Дисперсия, потерянная: {1 - sum(pca.explained_variance_ratio_):.4f}")


# ============================================================
#  Демо 3: PCA — разные n_components
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Explained variance для разных k")
print("=" * 55)

for k in [1, 2, 3]:
    pca_k = PCA(n_components=k)
    pca_k.fit(X_synth)
    total = sum(pca_k.explained_variance_ratio_)
    print(f"  k={k}: explained variance = {total:.4f}")


# ============================================================
#  Демо 4: Ковариационная матрица
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Ковариационная матрица")
print("=" * 55)

n_samples = len(X_synth)
n_feat = 3
means = [sum(X_synth[i][j] for i in range(n_samples)) / n_samples for j in range(n_feat)]
cov = [[0.0] * n_feat for _ in range(n_feat)]
for i in range(n_feat):
    for j in range(n_feat):
        cov[i][j] = sum((X_synth[k][i] - means[i]) * (X_synth[k][j] - means[j])
                        for k in range(n_samples)) / (n_samples - 1)

print("Ковариационная матрица:")
for row in cov:
    print(f"  [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")


# ============================================================
#  Демо 5: PCA на двойных спиралях (линейная vs нелинейная)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Спирали — PCA не может разделить")
print("=" * 55)

spiral_points, spiral_labels = generate_spiral(100, noise=0.3)
pca_spiral = PCA(n_components=1)
X_spiral_1d = pca_spiral.fit_transform(spiral_points)

# Считаем, насколько хорошо PCA разделяет классы
class0 = [x[0] for x, l in zip(X_spiral_1d, spiral_labels) if l == 0]
class1 = [x[0] for x, l in zip(X_spiral_1d, spiral_labels) if l == 1]

mean0 = sum(class0) / len(class0)
mean1 = sum(class1) / len(class1)
var0 = sum((x - mean0)**2 for x in class0) / len(class0)
var1 = sum((x - mean1)**2 for x in class1) / len(class1)

print(f"PCA (линейная): проекция на 1D")
print(f"  Класс 0: среднее={mean0:.2f}, дисперсия={var0:.2f}")
print(f"  Класс 1: среднее={mean1:.2f}, дисперсия={var1:.2f}")
print(f"  Разделение: {'плохое' if abs(mean0 - mean1) < 2 else 'нормальное'}")
print(f"  → Спирали перекрываются в 1D — PCA не помогает")


# ============================================================
#  Демо 6: Концентрические кольца — PCA vs Kernel PCA
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Концентрические кольца")
print("=" * 55)

circle_points, circle_labels = generate_concentric_circles(100, noise=0.1)

# Стандартная PCA
pca_circle = PCA(n_components=1)
X_circle_1d = pca_circle.fit_transform(circle_points)

class0_c = [x[0] for x, l in zip(X_circle_1d, circle_labels) if l == 0]
class1_c = [x[0] for x, l in zip(X_circle_1d, circle_labels) if l == 1]
mean0_c = sum(class0_c) / len(class0_c)
mean1_c = sum(class1_c) / len(class1_c)

print(f"Стандартная PCA → 1D:")
print(f"  Класс 0: среднее={mean0_c:.2f}")
print(f"  Класс 1: среднее={mean1_c:.2f}")
print(f"  → Кольца перекрываются (оба на одной оси)")

# Kernel PCA (RBF)
print(f"\nKernel PCA (RBF) → 1D:")
gamma = 0.5
n_c = len(circle_points)
# Ядерная матрица
K = [[0.0] * n_c for _ in range(n_c)]
for i in range(n_c):
    for j in range(n_c):
        dist = sum((circle_points[i][k] - circle_points[j][k])**2 for k in range(2))
        K[i][j] = math.exp(-gamma * dist)

# Центрирование ядра
ones = [[1.0/n_c] * n_c for _ in range(n_c)]
K_centered = [[K[i][j] - sum(K[i][k] for k in range(n_c))/n_c
               - sum(K[k][j] for k in range(n_c))/n_c
               + sum(sum(K[k][l] for l in range(n_c)) for k in range(n_c))/n_c**2
               for j in range(n_c)] for i in range(n_c)]

# Собственные значения (упрощённо — берём первый компонент)
eigvals = [sum(K_centered[i][i] for i in range(n_c))]
X_kpca = []
for i in range(n_c):
    val = sum(K_centered[i][j] for j in range(n_c)) / math.sqrt(n_c * max(eigvals[0], 1e-10))
    X_kpca.append([val])

class0_k = [x[0] for x, l in zip(X_kpca, circle_labels) if l == 0]
class1_k = [x[0] for x, l in zip(X_kpca, circle_labels) if l == 1]
mean0_k = sum(class0_k) / len(class0_k)
mean1_k = sum(class1_k) / len(class1_k)

print(f"  Класс 0: среднее={mean0_k:.2f}")
print(f"  Класс 1: среднее={mean1_k:.2f}")
print(f"  → Кольца разделены (RBF kernel проецирует на разные радиусы)")


# ============================================================
#  Демо 7: Сравнение PCA vs t-SNE (концептуально)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Сравнение методов")
print("=" * 55)

print(f"""
┌─────────────┬──────────────────┬────────────────────┬──────────┐
│ Метод       │ Зачем            │ Сохраняет          │ Скорость │
├─────────────┼──────────────────┼────────────────────┼──────────┤
│ PCA         │ Препроцессинг    │ Глобальную диспер. │ Быстро   │
│ t-SNE       │ Визуализация 2D  │ Локальных соседей  │ Медленно │
│ UMAP        │ Визуализация     │ Лок.+глобальную    │ Средне   │
│ Kernel PCA  │ Нелинейная проекц│ Сложные структуры  │ Средне   │
└─────────────┴──────────────────┴────────────────────┴──────────┘

PCA:    784 → 50 (препроцессинг), 784 → 2 (визуализация)
t-SNE:  784 → 2 (только визуализация)
UMAP:   784 → 2 (визуализация + структура)
""")


# ============================================================
#  Демо 8: Выбор числа компонент (elbow method)
# ============================================================

print("=" * 55)
print("ДЕМО 8: Elbow method — сколько компонент оставить")
print("=" * 55)

# Генерируем данные с 10 признаками, но有效的 dimensionality = 3
random.seed(42)
X_high = []
for _ in range(200):
    z1 = random.gauss(0, 3)
    z2 = random.gauss(0, 2)
    z3 = random.gauss(0, 1)
    x = [
        z1 + random.gauss(0, 0.1),
        z2 + random.gauss(0, 0.1),
        z3 + random.gauss(0, 0.1),
        0.5 * z1 + random.gauss(0, 0.5),
        0.3 * z2 + random.gauss(0, 0.5),
        random.gauss(0, 0.3),
        random.gauss(0, 0.3),
        random.gauss(0, 0.3),
        random.gauss(0, 0.3),
        random.gauss(0, 0.3),
    ]
    X_high.append(x)

print(f"Данные: 200×10 (эффективная размерность ≈ 3)\n")
print(f"{'k':>3}  {'Explained':>10}  {'Cumulative':>10}")
print("-" * 28)

cumulative = 0
for k in range(1, 11):
    pca_k = PCA(n_components=k)
    pca_k.fit(X_high)
    var_k = pca_k.explained_variance_ratio_[-1]
    cumulative = sum(pca_k.explained_variance_ratio_)
    marker = " ← elbow" if k == 3 else ""
    print(f"{k:>3}  {var_k:>10.4f}  {cumulative:>10.4f}{marker}")
