import math
import random

random.seed(42)


# ============================================================
#  Нормы
# ============================================================

def l1_norm(x):
    return sum(abs(v) for v in x)

def l2_norm(x):
    return math.sqrt(sum(v**2 for v in x))

def linf_norm(x):
    return max(abs(v) for v in x)

def lp_norm(x, p):
    return sum(abs(v)**p for v in x) ** (1/p)


# ============================================================
#  Расстояния
# ============================================================

def l1_distance(a, b):
    return l1_norm([ai - bi for ai, bi in zip(a, b)])

def l2_distance(a, b):
    return l2_norm([ai - bi for ai, bi in zip(a, b)])

def linf_distance(a, b):
    return linf_norm([ai - bi for ai, bi in zip(a, b)])

def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)

def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


# ============================================================
#  Mahalanobis
# ============================================================

def covariance_matrix(data):
    n = len(data)
    d = len(data[0])
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum((data[k][i] - means[i]) * (data[k][j] - means[j])
                           for k in range(n)) / n
    return cov

def inverse_2x2(m):
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < 1e-10:
        return [[1, 0], [0, 1]]
    return [[m[1][1]/det, -m[0][1]/det], [-m[1][0]/det, m[0][0]/det]]

def mahalanobis_distance(a, b, cov_inv):
    d = len(a)
    diff = [ai - bi for ai, bi in zip(a, b)]
    # (diff)^T @ cov_inv @ diff
    middle = [sum(cov_inv[i][j] * diff[j] for j in range(d)) for i in range(d)]
    return math.sqrt(sum(diff[i] * middle[i] for i in range(d)))


# ============================================================
#  Jaccard
# ============================================================

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def jaccard_distance(set_a, set_b):
    return 1 - jaccard_similarity(set_a, set_b)


# ============================================================
#  Edit Distance (Levenshtein)
# ============================================================

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # удаление
                                   dp[i][j-1],      # вставка
                                   dp[i-1][j-1])    # замена
    return dp[m][n]


# ============================================================
#  KL Divergence
# ============================================================

def kl_divergence(p, q, epsilon=1e-10):
    kl = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, epsilon)
        qi = max(qi, epsilon)
        kl += pi * math.log(pi / qi)
    return kl


# ============================================================
#  Wasserstein Distance (1D)
# ============================================================

def wasserstein_1d(p, q):
    """Wasserstein distance для 1D распределений через CDF"""
    n = len(p)
    # CDF
    cdf_p = []
    cdf_q = []
    sum_p = 0
    sum_q = 0
    for i in range(n):
        sum_p += p[i]
        sum_q += q[i]
        cdf_p.append(sum_p)
        cdf_q.append(sum_q)
    # L1 расстояние между CDF
    return sum(abs(a - b) for a, b in zip(cdf_p, cdf_q))


# ============================================================
#  Демо 1: L1, L2, L∞ нормы
# ============================================================

print("=" * 55)
print("ДЕМО 1: L1, L2, L∞ нормы")
print("=" * 55)

a = [1, 2, 3]
b = [4, 0, 6]

print(f"\na = {a}")
print(f"b = {b}")
print(f"\nL1 расстояние:   {l1_distance(a, b)}  (|4-1| + |0-2| + |6-3| = 3+2+3)")
print(f"L2 расстояние:   {l2_distance(a, b):.4f}  (√(9+4+9) = √22)")
print(f"L∞ расстояние:   {linf_distance(a, b)}  (max(3,2,3))")

print(f"\nПорядок: L∞ ≤ L2 ≤ L1: {linf_distance(a,b) <= l2_distance(a,b) <= l1_distance(a,b)}")


# ============================================================
#  Демо 2: Lp нормы — разные формы
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Lp нормы — единичные шары")
print("=" * 55)

# Точки на единичном шаре для разных p
for p in [1, 2, 3, 10, 100]:
    points = []
    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        x = math.cos(angle)
        y = math.sin(angle)
        # Нормализуем по Lp
        norm = (abs(x)**p + abs(y)**p) ** (1/p)
        points.append((x/norm, y/norm))
    print(f"\nL{p}:")
    for pt in points[:4]:
        print(f"  ({pt[0]:6.3f}, {pt[1]:6.3f})")


# ============================================================
#  Демо 3: Cosine vs Dot Product
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Cosine vs Dot Product")
print("=" * 55)

a = [3, 0]
b = [1, 0]
c = [0, 1]

print(f"\na = {a}, b = {b}, c = {c}")
print(f"\ndot(a,b) = {dot_product(a,b)}, cos(a,b) = {cosine_similarity(a,b):.4f}")
print(f"dot(a,c) = {dot_product(a,c)}, cos(a,c) = {cosine_similarity(a,c):.4f}")

# Разные по длине, одинакового направления
d = [1, 1]
e = [10, 10]
print(f"\nd = {d}, e = {e} (одно направление, разная длина)")
print(f"dot(d,e) = {dot_product(d,e)}, cos(d,e) = {cosine_similarity(d,e):.4f}")
print(f"→ Cosine = 1.0 (одно направление), dot = 20 (разная мощность)")


# ============================================================
#  Демо 4: Mahalanobis
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Mahalanobis vs Euclidean")
print("=" * 55)

# Данные с корреляцией рост-вес
data = [
    [170, 65], [175, 70], [180, 75], [165, 60],
    [172, 68], [178, 73], [168, 62], [176, 71],
    [182, 78], [174, 69],
]

cov = covariance_matrix(data)
cov_inv = inverse_2x2(cov)

mean = [sum(d[0] for d in data)/len(data), sum(d[1] for d in data)/len(data)]

# Нормальная точка
normal = [175, 70]
# Выброс: рост 160, вес 90 (тяжёлый для своего роста)
outlier = [160, 90]

euc_normal = l2_distance(normal, mean)
euc_outlier = l2_distance(outlier, mean)
mah_normal = mahalanobis_distance(normal, mean, cov_inv)
mah_outlier = mahalanobis_distance(outlier, mean, cov_inv)

print(f"\nСреднее: рост={mean[0]:.0f}, вес={mean[1]:.0f}")
print(f"Нормальная точка: рост=175, вес=70")
print(f"Выброс:           рост=160, вес=90")
print(f"\n                  Euclidean  Mahalanobis")
print(f"  Нормальная:     {euc_normal:8.2f}    {mah_normal:8.2f}")
print(f"  Выброс:         {euc_outlier:8.2f}    {mah_outlier:8.2f}")
print(f"\n→ Mahalanobis правильно определяет выброс (большое расстояние)")


# ============================================================
#  Демо 5: Jaccard
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Jaccard Similarity")
print("=" * 55)

A = {"кошка", "собака", "рыба"}
B = {"кошка", "птица", "рыба", "змея"}

print(f"\nA = {A}")
print(f"B = {B}")
print(f"Пересечение: {A & B}")
print(f"Объединение: {A | B}")
print(f"Jaccard similarity = {jaccard_similarity(A, B):.2f}")
print(f"Jaccard distance   = {jaccard_distance(A, B):.2f}")


# ============================================================
#  Демо 6: Edit Distance
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Edit Distance (Levenshtein)")
print("=" * 55)

pairs = [
    ("kitten", "sitting"),
    ("hello", "hello"),
    ("book", "back"),
    ("aturday", "saturday"),
]

for s1, s2 in pairs:
    dist = edit_distance(s1, s2)
    print(f"  '{s1}' → '{s2}': {dist}")


# ============================================================
#  Демо 7: KL Divergence (не симметрична!)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: KL Divergence (не симметрична)")
print("=" * 55)

P = [0.7, 0.2, 0.1]
Q = [0.1, 0.1, 0.8]

print(f"\nP = {P}")
print(f"Q = {Q}")
print(f"\nKL(P||Q) = {kl_divergence(P, Q):.4f}")
print(f"KL(Q||P) = {kl_divergence(Q, P):.4f}")
print(f"\n→ KL(P||Q) ≠ KL(Q||P) — это НЕ расстояние!")


# ============================================================
#  Демо 8: Wasserstein Distance
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Wasserstein Distance")
print("=" * 55)

P1 = [0.5, 0.5, 0, 0]
Q1 = [0, 0, 0.5, 0.5]
P2 = [0.25, 0.25, 0.25, 0.25]
Q2 = [0, 0, 0.5, 0.5]

print(f"\nP1 = {P1}")
print(f"Q1 = {Q1}")
print(f"Wasserstein(P1, Q1) = {wasserstein_1d(P1, Q1)}")

print(f"\nP2 = {P2}")
print(f"Q2 = {Q2}")
print(f"Wasserstein(P2, Q2) = {wasserstein_1d(P2, Q2)}")

print(f"\n→ Wasserstein даёт осмысленный градиент, даже если распределения не пересекаются")


# ============================================================
#  Демо 9: Одни данные — разные ближайшие соседи
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 9: Разные метрики — разные соседи")
print("=" * 55)

data_points = [
    [1, 1],
    [2, 2],
    [10, 1],
    [10, 2],
    [5, 5],
]

query = [0, 0]

print(f"\nТочки: {data_points}")
print(f"Запрос: {query}\n")

# L1
dists_l1 = [(l1_distance(query, p), i) for i, p in enumerate(data_points)]
dists_l1.sort()
print(f"L1 (Манхэттен): ближайший = точка {dists_l1[0][1]} (расст={dists_l1[0][0]})")

# L2
dists_l2 = [(l2_distance(query, p), i) for i, p in enumerate(data_points)]
dists_l2.sort()
print(f"L2 (Евклидова): ближайший = точка {dists_l2[0][1]} (расст={dists_l2[0][1]:.2f})")

# Cosine
dists_cos = [(cosine_distance(query, p), i) for i, p in enumerate(data_points)]
dists_cos.sort()
print(f"Cosine:          ближайший = точка {dists_cos[0][1]} (расст={dists_cos[0][0]:.4f})")
