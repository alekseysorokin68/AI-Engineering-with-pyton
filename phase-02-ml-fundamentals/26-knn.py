"""
K-Nearest Neighbors (KNN) — реализация с нуля на Python

Алгоритм классификации, который определяет класс объекта по большинству
среди K ближайших соседей в пространстве признаков.

Содержание:
  1. Реализация класса KNN (fit, predict)
  2. Метрики расстояния: Евклидова, Манхэттен, косинусная
  3. Влияние K на результат
  4. Демо: разные K, сравнение метрик, влияние K на границу
"""

import math
import random

random.seed(42)


# ============================================================
# 1. Метрики расстояния
# ============================================================

def euclidean_distance(a, b):
    """Евклидово расстояние: sqrt(sum((ai - bi)^2))"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def manhattan_distance(a, b):
    """Расстояние Манхэттена: sum(|ai - bi|)"""
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def cosine_distance(a, b):
    """Косинусное расстояние: 1 - cos(angle)"""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


METRICS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "cosine": cosine_distance,
}


# ============================================================
# 2. Класс KNN
# ============================================================

class KNN:
    """
    K-Nearest Neighbors классификатор.

    Параметры:
        k (int): количество ближайших соседей
        metric (str): метрика расстояния ('euclidean', 'manhattan', 'cosine')
    """

    def __init__(self, k=3, metric="euclidean"):
        if k < 1:
            raise ValueError("k должно быть >= 1")
        if metric not in METRICS:
            raise ValueError(f"Неизвестная метрика: {metric}. Доступны: {list(METRICS.keys())}")
        self.k = k
        self.metric = metric
        self.dist_fn = METRICS[metric]
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """Обучение: запоминаем обучающую выборку."""
        self.X_train = [list(x) for x in X]
        self.y_train = list(y)
        return self

    def _get_neighbors(self, x):
        """Находим K ближайших соседей для точки x."""
        distances = []
        for i, x_train in enumerate(self.X_train):
            d = self.dist_fn(x, x_train)
            distances.append((d, self.y_train[i]))
        distances.sort(key=lambda pair: pair[0])
        return distances[:self.k]

    def _vote(self, neighbors):
        """Голосование: выбираем класс с максимальным числом голосов."""
        counts = {}
        for _, label in neighbors:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    def predict_single(self, x):
        """Предсказание для одной точки."""
        neighbors = self._get_neighbors(x)
        return self._vote(neighbors)

    def predict(self, X):
        """Предсказание для набора точек."""
        return [self.predict_single(x) for x in X]

    def score(self, X, y):
        """Accuracy: доля правильных ответов."""
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)


# ============================================================
# 3. Вспомогательные функции для генерации данных
# ============================================================

def generate_cluster(cx, cy, n, spread=0.5):
    """Генерация кластера точек вокруг (cx, cy)."""
    points = []
    for _ in range(n):
        x = cx + random.gauss(0, spread)
        y = cy + random.gauss(0, spread)
        points.append((x, y))
    return points


def generate_2class_data(n_per_class=50):
    """Два пересекающихся кластера для классификации."""
    random.seed(42)
    X = []
    y = []
    for x, label in [(generate_cluster(2, 2, n_per_class, 1.2), 0),
                     (generate_cluster(4, 4, n_per_class, 1.2), 1)]:
        X.extend(x)
        y.extend(label if isinstance(label, list) else [label] * len(x))
    return X, y


def generate_multiclass_data():
    """Три класса для задачи с несколькими метками."""
    random.seed(42)
    X = []
    y = []
    centers = [(1, 1), (5, 5), (9, 1)]
    labels = [0, 1, 2]
    for (cx, cy), label in zip(centers, labels):
        points = generate_cluster(cx, cy, 30, 0.7)
        X.extend(points)
        y.extend([label] * len(points))
    return X, y


def make_meshgrid(X, step=0.2, margin=1.0):
    """Сетка для визуализации разделяющей границы."""
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    xx = []
    yy = []
    y_val = y_min
    while y_val <= y_max:
        x_val = x_min
        row_x = []
        row_y = []
        while x_val <= x_max:
            row_x.append(round(x_val, 2))
            row_y.append(round(y_val, 2))
            x_val += step
        xx.append(row_x)
        yy.append(row_y)
        y_val += step
    return xx, yy


def print_ascii_grid(xx, yy, predictions, X_train, y_train, resolution=5):
    """
    ASCII-визуализация разделяющей границы.
    resolution — шаг по сетке для ускорения вывода.
    """
    n_rows = len(xx)
    n_cols = len(xx[0])
    symbols = ["·", "█", "▪", "○"]
    color_labels = {0: ".", 1: "#", 2: "o"}

    for i in range(0, n_rows, resolution):
        row = ""
        for j in range(0, n_cols, resolution):
            cls = predictions[i * n_cols + j]
            row += symbols[cls % len(symbols)]
        print(row)
    print()


# ============================================================
# 4. Демонстрации
# ============================================================

def demo_1_different_k():
    """Демо 1: Разные значения K (1, 3, 5, 10)"""
    print("=" * 65)
    print("ДЕМО 1: Влияние K на качество классификации")
    print("=" * 65)

    X, y = generate_2class_data(50)

    # Разделим на train/test вручную
    random.seed(42)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"\nОбучающая выборка: {len(X_train)} точек")
    print(f"Тестовая выборка:  {len(X_test)} точек")
    print(f"\n{'K':>4} | {'Accuracy':>10}")
    print("-" * 20)

    results = []
    for k in [1, 3, 5, 10]:
        model = KNN(k=k, metric="euclidean")
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results.append((k, acc))
        print(f"{k:>4} | {acc:>10.4f}")

    print("\nВывод:")
    print("  K=1  — максимальное переобучение (шум влияет сильно)")
    print("  K=3  — компромисс, но может быть нестабильно")
    print("  K=5  — устойчивая классификация")
    print("  K=10 — возможное недообучение, граница сглаживается")
    print(f"\nЛучший K в эксперименте: {max(results, key=lambda x: x[1])[0]}")
    print()


def demo_2_distance_metrics():
    """Демо 2: Сравнение метрик расстояния"""
    print("=" * 65)
    print("ДЕМО 2: Сравнение метрик расстояния")
    print("=" * 65)

    X, y = generate_2class_data(50)

    random.seed(42)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"\n{'Метрика':<15} | {'K=3':>10} | {'K=5':>10}")
    print("-" * 40)

    for metric_name in ["euclidean", "manhattan", "cosine"]:
        accs = []
        for k in [3, 5]:
            model = KNN(k=k, metric=metric_name)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            accs.append(acc)
        print(f"{metric_name:<15} | {accs[0]:>10.4f} | {accs[1]:>10.4f}")

    print("\nВывод:")
    print("  Евклидова — стандартный выбор для числовых признаков")
    print("  Манхэттен — устойчивее к выбросам (сумма |Δ| вместо Δ²)")
    print("  Косинусная — ориентирована на направление, а не длину вектора")
    print()


def demo_3_decision_boundary():
    """Демо 3: Влияние K на разделяющую границу"""
    print("=" * 65)
    print("ДЕМО 3: Влияние K на разделяющую границу (ASCII-визуализация)")
    print("=" * 65)

    X, y = generate_multiclass_data()

    for k in [1, 5, 15]:
        model = KNN(k=k, metric="euclidean")
        model.fit(X, y)

        xx, yy = make_meshgrid(X, step=0.5, margin=1.0)
        grid_points = []
        for i in range(len(xx)):
            for j in range(len(xx[i])):
                grid_points.append((xx[i][j], yy[i][j]))

        predictions = model.predict(grid_points)

        print(f"\n--- K = {k} (n_classes=3) ---")
        print("  · = класс 0, █ = класс 1, ▪ = класс 2")
        print()

        n_rows = len(xx)
        n_cols = len(xx[0])
        symbols = ["·", "█", "▪"]

        # Печатаем网格
        for i in range(n_rows):
            row = ""
            for j in range(n_cols):
                cls = predictions[i * n_cols + j]
                row += symbols[cls % len(symbols)]
            print(row)

    print("\nВывод:")
    print("  K=1  — граница очень извилистая, подстраивается под каждый объект")
    print("  K=5  — граница более гладкая, устойчивая")
    print("  K=15 — граница сильно сглажена, может игнорировать локальные структуры")
    print()


def demo_4_manhattan_vs_euclidean_detail():
    """Дополнительная демонстрация: подробное сравнение Евклидовой и Манхэттен."""
    print("=" * 65)
    print("ДОПОЛНИТЕЛЬНО: Евклидово vs Манхэттен расстояние")
    print("=" * 65)

    a = (3, 4)
    b = (0, 0)

    euc = euclidean_distance(a, b)
    man = manhattan_distance(a, b)
    cos = cosine_distance(a, b)

    print(f"\nТочки: A={a}, B={b}")
    print(f"  Евклидово расстояние: {euc:.4f}")
    print(f"  Манхэттен расстояние: {man:.4f}")
    print(f"  Косинусное расстояние: {cos:.4f}")

    print("\nИнтерпретация:")
    print(f"  Евклидово (прямая): sqrt(3²+4²) = sqrt(25) = {euc}")
    print(f"  Манхэттен (по осям): |3-0|+|4-0| = {man}")
    print(f"  Косинусное (угол): 1 - cos(angle) = {cos:.4f}")

    # Пример с тремя точками — разные результаты метрик
    print("\n--- Три точки: P1=(1,1), P2=(2,2), P3=(5,1) ---")
    print("Ближайший к P1 по каждой метрике:")
    p1 = (1, 1)
    p2 = (2, 2)
    p3 = (5, 1)

    for name, fn in METRICS.items():
        d2 = fn(p1, p2)
        d3 = fn(p1, p3)
        closer = "P2" if d2 < d3 else "P3"
        print(f"  {name:<12}: d(P1,P2)={d2:.4f}, d(P1,P3)={d3:.4f} → ближе {closer}")
    print()


# ============================================================
# 5. Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     K-Nearest Neighbors (KNN) — реализация с нуля           ║")
    print("║     Без sklearn, чистый Python                              ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    demo_1_different_k()
    demo_2_distance_metrics()
    demo_3_decision_boundary()
    demo_4_manhattan_vs_euclidean_detail()

    print("=" * 65)
    print("Итого по KNN:")
    print("=" * 65)
    print("""
  K-Nearest Neighbors — один из простейших алгоритмов ML:

  Принцип работы:
    1. Запоминаем все обучающие данные (lazy learning)
    2. Для новой точки считаем расстояния до всех объектов
    3. Берём K ближайших соседей
    4. Мажоритарным голосованием определяем класс

  Метрики расстояния:
    • Евклидова: sqrt(ΣΔ²) — чувствительна к выбросам
    • Манхэттен: Σ|Δ|    — устойчивее к выбросам
    • Косинусная: 1-cos(θ) — ориентация важнее длины

  Выбор K:
    • K=1: переобучение, шумовые выбросы
    • K=среднее: компромисс
    • K=большое: недообучение, сглаживание границ

  Достоинства:
    + Простота реализации и понимания
    + Не требует обучения (ленивое обучение)
    + Автоматически подходит для нелинейных границ

  Недостатки:
    - Медленно на больших данных O(n*d) на каждый запрос
    - Чувствителен к масштабу признаков
    - Плохо работает в высоких размерностях (проклятие размерности)
""")
