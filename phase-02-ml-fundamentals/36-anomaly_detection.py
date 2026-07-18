"""
36 - Anomaly Detection (с нуля на Python)

Методы обнаружения аномалий:
1. Z-score
2. IQR (межквартильный размах)
3. Isolation Forest (упрощённо)
4. Threshold-based detection
"""

import math
import random

random.seed(42)


# ─── Утилиты ────────────────────────────────────────────────────────────────

def mean(arr):
    return sum(arr) / len(arr)


def median(arr):
    s = sorted(arr)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def stdev(arr):
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))


def percentile(arr, p):
    """Простой percentil (linear interpolation)."""
    s = sorted(arr)
    k = (p / 100.0) * (len(s) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ─── 1. Z-score ─────────────────────────────────────────────────────────────

def zscore_anomalies(data, threshold=2.0):
    """
    Обнаружение аномалий по Z-score.

    Z = (x - mean) / std
    |Z| > threshold  -->  аномалия

    Возвращает список (index, value, z_score) для аномалий.
    """
    m = mean(data)
    s = stdev(data)
    if s == 0:
        return []

    results = []
    for i, x in enumerate(data):
        z = (x - m) / s
        if abs(z) > threshold:
            results.append((i, x, z))
    return results


# ─── 2. IQR ──────────────────────────────────────────────────────────────────

def iqr_anomalies(data, k=1.5):
    """
    Обнаружение аномалий по межквартильному размаху (IQR).

    Q1 = 25-й процентиль, Q3 = 75-й процентиль
    IQR = Q3 - Q1
    Нижняя граница = Q1 - k * IQR
    Верхняя граница = Q3 + k * IQR
    Значение за пределами --> аномалия

    Возвращает список (index, value, lower, upper) для аномалий.
    """
    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    results = []
    for i, x in enumerate(data):
        if x < lower or x > upper:
            results.append((i, x, lower, upper))
    return results


# ─── 3. Isolation Forest (упрощённо) ────────────────────────────────────────

class IsolationTree:
    """Одиночное дерево изолирования."""

    def __init__(self, depth=0, max_depth=10):
        self.depth = depth
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = False

    def fit(self, X):
        self.size = len(X)
        if self.depth >= self.max_depth or len(X) <= 1:
            self.is_leaf = True
            return

        n_features = len(X[0])
        self.split_feature = random.randint(0, n_features - 1)

        col = [row[self.split_feature] for row in X]
        lo, hi = min(col), max(col)
        if lo == hi:
            self.is_leaf = True
            return

        self.split_value = random.uniform(lo, hi)

        left = [row for row in X if row[self.split_feature] < self.split_value]
        right = [row for row in X if row[self.split_feature] >= self.split_value]

        self.left = IsolationTree(self.depth + 1, self.max_depth)
        self.right = IsolationTree(self.depth + 1, self.max_depth)
        self.left.fit(left)
        self.right.fit(right)

    def path_length(self, x):
        """Количество分裂ов для разделения точки x."""
        if self.is_leaf:
            return self.depth
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x)
        return self.right.path_length(x)


class IsolationForestModel:
    """
    Упрощённый Isolation Forest.

    - n_trees деревьев
    - path_length чем меньше --> тем более изолирована точка --> аномалия
    - anomaly_score = 2^(-mean_path / c(n)), где c(n) -- средняя длина пути в BST
    """

    def __init__(self, n_trees=100, max_depth=10, contamination=0.05):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.contamination = contamination
        self.trees = []
        self.threshold = None

    def _c(self, n):
        """Средняя длина unsuccessful search в BST."""
        if n <= 1:
            return 0
        return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

    def fit(self, data):
        """Обучение на списке списков [[f1, f2, ...], ...]."""
        self.trees = []
        n = len(data)
        sample_size = min(256, n)
        avg_path = self._c(sample_size)

        for _ in range(self.n_trees):
            sample = random.choices(data, k=sample_size)
            tree = IsolationTree(max_depth=self.max_depth)
            tree.fit(sample)
            self.trees.append(tree)

        scores = []
        for x in data:
            paths = [t.path_length(x) for t in self.trees]
            avg = sum(paths) / len(paths)
            score = 2.0 ** (-avg / avg_path) if avg_path > 0 else 0
            scores.append(score)

        scores_sorted = sorted(scores, reverse=True)
        idx = max(1, int(n * self.contamination))
        self.threshold = scores_sorted[idx - 1]
        return scores

    def predict(self, data, scores):
        """Возвращает список (index, value, score) для аномалий."""
        results = []
        for i, (x, s) in enumerate(zip(data, scores)):
            if s >= self.threshold:
                results.append((i, x, s))
        return results


# ─── 4. Threshold-based ─────────────────────────────────────────────────────

def threshold_anomalies(data, low=None, high=None):
    """
    Обнаружение аномалий по явным порогам.

    Если low задан: x < low --> аномалия
    Если high задан: x > high --> аномалия

    Возвращает список (index, value) для аномалий.
    """
    results = []
    for i, x in enumerate(data):
        if (low is not None and x < low) or (high is not None and x > high):
            results.append((i, x))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#   ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def generate_normal_data(n=100, mu=50, sigma=5):
    """Генерация нормальных данных + случайные аномалии."""
    data = []
    for _ in range(n):
        data.append(random.gauss(mu, sigma))
    return data


def add_anomalies(data, n_anomalies=5, low=-20, high=120):
    """Добавить явные аномалии."""
    indices = random.sample(range(len(data)), min(n_anomalies, len(data)))
    for i in indices:
        if random.random() < 0.5:
            data[i] = random.uniform(low, mu - 3 * sigma)
        else:
            data[i] = random.uniform(mu + 3 * sigma, high)
    return data, indices


# ═══════════════════════════════════════════════════════════════════════════════
#   ДЕМО 1: Z-score
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ДЕМО 1: Z-score метод")
print("=" * 70)

mu, sigma = 50, 5
data = generate_normal_data(100, mu, sigma)
data, anomaly_indices = add_anomalies(data, 5)

print(f"\nДанные: {len(data)} точек (mu={mu}, sigma={sigma}, + 5 аномалий)")
print(f"Индекс аномалий: {anomaly_indices}")
print(f"\nДанные (первые 20): {['{:.1f}'.format(x) for x in data[:20]]}")

for t in [2.0, 1.5, 3.0]:
    anoms = zscore_anomalies(data, threshold=t)
    print(f"\n  Z-score threshold={t}: найдено {len(anoms)} аномалий")
    for idx, val, z in anoms[:8]:
        marker = " <-- аномалия" if idx in anomaly_indices else ""
        print(f"    [{idx:3d}] value={val:7.2f}  z={z:+6.2f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
#   ДЕМО 2: IQR
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ДЕМО 2: IQR метод")
print("=" * 70)

q1 = percentile(data, 25)
q3 = percentile(data, 75)
iqr_val = q3 - q1

print(f"\nQ1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr_val:.2f}")

for k in [1.5, 2.0, 1.0]:
    anoms = iqr_anomalies(data, k=k)
    lo = q1 - k * iqr_val
    hi = q3 + k * iqr_val
    print(f"\n  k={k}: границы [{lo:.2f}, {hi:.2f}]")
    print(f"    Найдено {len(anoms)} аномалий")
    for idx, val, lower, upper in anoms[:8]:
        marker = " <-- аномалия" if idx in anomaly_indices else ""
        print(f"    [{idx:3d}] value={val:7.2f}  (границы: {lower:.2f}..{upper:.2f}){marker}")

# ═══════════════════════════════════════════════════════════════════════════════
#   ДЕМО 3: Сравнение методов
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ДЕМО 3: Сравнение методов")
print("=" * 70)

random.seed(42)
mu, sigma = 50, 5
data = generate_normal_data(200, mu, sigma)
true_anomalies = set()

# Добавляем 10 аномалий в Known locations
for i in random.sample(range(200), 10):
    data[i] = random.uniform(-10, 110) if random.random() < 0.5 else random.uniform(-10, 0)
    true_anomalies.add(i)

zscore_anoms = zscore_anomalies(data, threshold=2.5)
iqr_anoms = iqr_anomalies(data, k=1.5)
thres_anoms = threshold_anomalies(data, low=30, high=70)

zset = {a[0] for a in zscore_anoms}
iqset = {a[0] for a in iqr_anoms}
thset = {a[0] for a in thres_anoms}

all_detected = zset | iqset | thset
all_true = true_anomalies
true_positives = all_detected & all_true
false_positives = all_detected - all_true
false_negatives = all_true - all_detected

def f1_score(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)

precision = len(true_positives) / len(all_detected) if all_detected else 0
recall = len(true_positives) / len(all_true) if all_true else 0
f1 = f1_score(precision, recall)

print(f"\nИстинные аномалии: {len(true_anomalies)}")
print(f"Z-score (thr=2.5): {len(zset)} детекций | TP={len(zset & all_true)} | FP={len(zset - all_true)}")
print(f"IQR (k=1.5):       {len(iqset)} детекций | TP={len(iqset & all_true)} | FP={len(iqset - all_true)}")
print(f"Threshold [30,70]:  {len(thset)} детекций | TP={len(thset & all_true)} | FP={len(thset - all_true)}")

print(f"\nОбъединённый: {len(all_detected)} детекций | TP={len(true_positives)} | FP={len(false_positives)} | FN={len(false_negatives)}")
print(f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

print(f"\nTrue Positives (индексы): {sorted(true_positives)[:15]}")
print(f"False Positives (индексы): {sorted(false_positives)[:15]}")
print(f"False Negatives (индексы): {sorted(false_negatives)[:15]}")

# ═══════════════════════════════════════════════════════════════════════════════
#   ДЕМО 4: Влияние порога
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ДЕМО 4: Влияние порога")
print("=" * 70)

random.seed(42)
mu, sigma = 50, 5
data = generate_normal_data(200, mu, sigma)
true_anom_set = set()
for i in random.sample(range(200), 8):
    data[i] = random.uniform(-15, 115) if random.random() < 0.5 else random.uniform(-15, 5)
    true_anom_set.add(i)

print(f"\nДанные: {len(data)} точек, 8 аномалий")
print(f"\n{'Threshold':>10} {'Detected':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print("-" * 72)

for t in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    anoms = zscore_anomalies(data, threshold=t)
    detected = {a[0] for a in anoms}
    tp = len(detected & true_anom_set)
    fp = len(detected - true_anom_set)
    fn = len(true_anom_set - detected)
    prec = tp / len(detected) if detected else 0
    rec = tp / len(true_anom_set) if true_anom_set else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    print(f"{t:>10.1f} {len(detected):>10} {tp:>6} {fp:>6} {fn:>6} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

print("\n  Наблюдение:")
print("  - Низкий порог (1.0-1.5): много FP, высокий Recall")
print("  - Высокий порог (3.5-4.0): мало FP, но могут пропускаться аномалии")
print("  - Оптимальный порог зависит от бизнес-стоимости FP vs FN")

# ═══════════════════════════════════════════════════════════════════════════════
#   Дополнительно: Isolation Forest демо
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ДОПОЛНИТЕЛЬНО: Isolation Forest (2D данные)")
print("=" * 70)

random.seed(42)

# Генерируем 2D данные: основной кластер + аномалии
normal_points = []
for _ in range(200):
    normal_points.append([random.gauss(0, 1), random.gauss(0, 1)])

anomaly_points = []
for _ in range(10):
    x = random.uniform(-6, 6)
    y = random.uniform(-6, 6)
    anomaly_points.append([x, y])

all_points = normal_points + anomaly_points
true_labels = [0] * 200 + [1] * 10

model = IsolationForestModel(n_trees=100, max_depth=10, contamination=0.05)
scores = model.fit(all_points)
anoms = model.predict(all_points, scores)

print(f"\nДанные: {len(all_points)} точек (200 нормальных + 10 аномалий)")
print(f"Isolation Forest: {len(anoms)} детекций")

tp = sum(1 for i, _, _ in anoms if true_labels[i] == 1)
fp = sum(1 for i, _, _ in anoms if true_labels[i] == 0)
print(f"TP={tp}, FP={fp}")
print(f"\nТоп-10 аномалий по score:")
for idx, val, sc in sorted(anoms, key=lambda x: -x[2])[:10]:
    label = "anomaly" if true_labels[idx] == 1 else "normal"
    print(f"  [{idx:3d}] pos=({val[0]:+.2f}, {val[1]:+.2f})  score={sc:.4f}  actual={label}")

print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)
print("""
Методы обнаружения аномалий:

1. Z-score -- простой метод для одномерных данных с нормальным распределением.
   Работает плохо при heavy-tail распределениях.

2. IQR -- робастный к выбросам (использует медиану/квартили).
   Не предполагает нормального распределения.

3. Isolation Forest -- работает в многомерном пространстве.
   Не требует нормального распределения. Чем проще изолировать точку,
   тем вероятнее она аномалия.

4. Threshold -- простой и интерпретируемый. Подходит когда известны
   физические/бизнес-ограничения.

Выбор метода зависит от:
- Размерности данных (1D vs N-dimensional)
- Распределения данных
- Бизнес-стоимости ошибок (FP vs FN)
- Необходимость интерпретируемости
""")
