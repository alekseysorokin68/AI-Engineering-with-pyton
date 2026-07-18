"""
Методы подбора гиперпараметров с нуля на Python.

Реализованы без sklearn:
- Grid Search (полный перебор)
- Random Search

Демо:
1. Grid Search для learning rate и regularization
2. Random Search vs Grid Search
3. Влияние K в KNN
4. Влияние max_depth в дереве решений
"""

import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

random.seed(42)

# =============================================================================
# УТИЛИТЫ
# =============================================================================

def train_test_split(
    X: List[List[float]],
    y: List[float],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """Разбиение данных на train/test."""
    rng = random.Random(seed)
    n = len(X)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def accuracy_score(y_true: List[float], y_pred: List[float]) -> float:
    """Accuracy для классификации."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """MSE для регрессии."""
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Евклидово расстояние."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def generate_classification_data(
    n_samples: int = 200,
    n_features: int = 2,
    seed: int = 42,
) -> Tuple[List[List[float]], List[float]]:
    """Генерация синтетических данных для классификации (2 класса)."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_samples):
        cls = rng.randint(0, 1)
        features = [rng.gauss(cls * 2 - 1, 0.8) for _ in range(n_features)]
        X.append(features)
        y.append(float(cls))
    return X, y


def generate_regression_data(
    n_samples: int = 200,
    seed: int = 42,
) -> Tuple[List[List[float]], List[float]]:
    """Генерация синтетических данных для регрессии."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_samples):
        x1 = rng.uniform(-3, 3)
        x2 = rng.uniform(-3, 3)
        noise = rng.gauss(0, 0.5)
        target = 2 * x1 - x2 ** 2 + noise
        X.append([x1, x2])
        y.append(target)
    return X, y


# =============================================================================
# МОДЕЛИ
# =============================================================================

class LinearRegressionSGD:
    """Линейная регрессия с градиентным спуском."""

    def __init__(self, lr: float = 0.01, regularization: float = 0.0, epochs: int = 100):
        self.lr = lr
        self.regularization = regularization
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)
        for _ in range(self.epochs):
            for i in range(n):
                pred = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                error = pred - y[i]
                for j in range(n_features):
                    grad = error * X[i][j] + self.regularization * self.weights[j]
                    self.weights[j] -= self.lr * grad
                self.bias -= self.lr * error

    def predict(self, X: List[List[float]]) -> List[float]:
        return [sum(w * x for w, x in zip(self.weights, row)) + self.bias for row in X]

    def score(self, X: List[List[float]], y: List[float]) -> float:
        """Возвращает -MSE (чем больше, тем лучше) для совместимости с Grid Search."""
        preds = self.predict(X)
        mse = mean_squared_error(y, preds)
        return -mse


class LogisticRegressionSGD:
    """Логистическая регрессия (SGD) для бинарной классификации."""

    def __init__(self, lr: float = 0.01, regularization: float = 0.0, epochs: int = 100):
        self.lr = lr
        self.regularization = regularization
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        for _ in range(self.epochs):
            for i in range(len(X)):
                z = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                pred = self._sigmoid(z)
                error = pred - y[i]
                for j in range(n_features):
                    grad = error * X[i][j] + self.regularization * self.weights[j]
                    self.weights[j] -= self.lr * grad
                self.bias -= self.lr * error

    def predict(self, X: List[List[float]]) -> List[float]:
        return [1.0 if self._sigmoid(sum(w * x for w, x in zip(self.weights, row)) + self.bias) >= 0.5 else 0.0 for row in X]

    def score(self, X: List[List[float]], y: List[float]) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)


class KNNClassifier:
    """K-ближайших соседей для классификации."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for x in X:
            dists = [(euclidean_distance(x, xt), yt) for xt, yt in zip(self.X_train, self.y_train)]
            dists.sort(key=lambda d: d[0])
            neighbors = [d[1] for d in dists[:self.k]]
            counts: Dict[float, int] = {}
            for label in neighbors:
                counts[label] = counts.get(label, 0) + 1
            predictions.append(max(counts, key=counts.get))
        return predictions

    def score(self, X: List[List[float]], y: List[float]) -> float:
        return accuracy_score(y, self.predict(X))


class DecisionTreeClassifier:
    """Простое дерево решений для классификации."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[Dict] = None

    def _gini(self, labels: List[float]) -> float:
        if not labels:
            return 0.0
        n = len(labels)
        classes = set(labels)
        imp = 1.0
        for c in classes:
            p = labels.count(c) / n
            imp -= p ** 2
        return imp

    def _best_split(self, X: List[List[float]], y: List[float]) -> Optional[Tuple[int, float]]:
        best_gain = -1.0
        best_feat, best_val = 0, 0.0
        parent_imp = self._gini(y)
        n = len(y)
        for feat in range(len(X[0])):
            values = sorted(set(row[feat] for row in X))
            for i in range(len(values) - 1):
                val = (values[i] + values[i + 1]) / 2
                left_y = [y[j] for j in range(n) if X[j][feat] <= val]
                right_y = [y[j] for j in range(n) if X[j][feat] > val]
                if not left_y or not right_y:
                    continue
                imp = (len(left_y) / n) * self._gini(left_y) + (len(right_y) / n) * self._gini(right_y)
                gain = parent_imp - imp
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_val = feat, val
        if best_gain <= 0:
            return None
        return best_feat, best_val

    def _build(self, X: List[List[float]], y: List[float], depth: int) -> Dict:
        classes = list(set(y))
        if len(classes) == 1:
            return {"leaf": True, "class": classes[0]}
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            counts: Dict[float, int] = {}
            for c in y:
                counts[c] = counts.get(c, 0) + 1
            return {"leaf": True, "class": max(counts, key=counts.get)}

        split = self._best_split(X, y)
        if split is None:
            counts = {}
            for c in y:
                counts[c] = counts.get(c, 0) + 1
            return {"leaf": True, "class": max(counts, key=counts.get)}

        feat, val = split
        left_idx = [i for i in range(len(y)) if X[i][feat] <= val]
        right_idx = [i for i in range(len(y)) if X[i][feat] > val]

        return {
            "leaf": False,
            "feature": feat,
            "threshold": val,
            "left": self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1),
            "right": self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1),
        }

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.tree = self._build(X, y, 0)

    def _predict_one(self, row: List[float], node: Dict) -> float:
        if node["leaf"]:
            return node["class"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self._predict_one(row, self.tree) for row in X]

    def score(self, X: List[List[float]], y: List[float]) -> float:
        return accuracy_score(y, self.predict(X))


# =============================================================================
# GRID SEARCH
# =============================================================================

def grid_search(
    model_class: type,
    param_grid: Dict[str, List[Any]],
    X_train: List[List[float]],
    y_train: List[float],
    X_val: List[List[float]],
    y_val: List[float],
    score_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Полный перебор всех комбинаций гиперпараметров.

    param_grid: {"lr": [0.001, 0.01, 0.1], "regularization": [0.0, 0.1]}
    score_fn: функция(model, X, y) -> float (больше = лучше)
    """
    if score_fn is None:
        score_fn = lambda m, X, y: m.score(X, y)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Генерация всех комбинаций
    combos = [{}]
    for key in keys:
        new_combos = []
        for combo in combos:
            for val in values[keys.index(key)]:
                combo_copy = dict(combo)
                combo_copy[key] = val
                new_combos.append(combo_copy)
        combos = new_combos

    best_score = -float("inf")
    best_params = {}
    results = []

    for params in combos:
        model = model_class(**params)
        model.fit(X_train, y_train)
        score = score_fn(model, X_val, y_val)
        results.append({"params": dict(params), "score": score})
        if score > best_score:
            best_score = score
            best_params = dict(params)

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"best_params": best_params, "best_score": best_score, "all_results": results}


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def random_search(
    model_class: type,
    param_distributions: Dict[str, List[Any]],
    n_iter: int = 50,
    X_train: Optional[List[List[float]]] = None,
    y_train: Optional[List[float]] = None,
    X_val: Optional[List[List[float]]] = None,
    y_val: Optional[List[float]] = None,
    score_fn: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Случайный поиск по пространству гиперпараметров.

    param_distributions: {"lr": [0.001, 0.01, 0.1], "k": [3, 5, 7, 9]}
    n_iter: количество случайных комбинаций
    """
    rng = random.Random(seed)

    if score_fn is None:
        score_fn = lambda m, X, y: m.score(X, y)

    best_score = -float("inf")
    best_params = {}
    results = []

    for _ in range(n_iter):
        params = {key: rng.choice(values) for key, values in param_distributions.items()}
        model = model_class(**params)
        model.fit(X_train, y_train)
        score = score_fn(model, X_val, y_val)
        results.append({"params": dict(params), "score": score})
        if score > best_score:
            best_score = score
            best_params = dict(params)

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"best_params": best_params, "best_score": best_score, "all_results": results}


# =============================================================================
# ВИЗУАЛИЗАЦИЯ (текстовая, через print)
# =============================================================================

def print_heatmap(
    results: List[Dict],
    param_x: str,
    param_y: str,
    score_key: str = "score",
    width: int = 50,
    height: int = 20,
) -> None:
    """Текстовая тепловая карта по двум параметрам."""
    # Извлекаем уникальные значения
    x_vals = sorted(set(r["params"][param_x] for r in results))
    y_vals = sorted(set(r["params"][param_y] for r in results))

    if not x_vals or not y_vals:
        print("  Нет данных для тепловой карты")
        return

    # Построение матрицы
    matrix: Dict[Tuple, float] = {}
    for r in results:
        key = (r["params"][param_x], r["params"][param_y])
        matrix[key] = r[score_key]

    all_scores = [v for v in matrix.values() if v != -float("inf") and not (isinstance(v, float) and math.isnan(v))]
    if not all_scores:
        print("  Все scores = -inf/NaN")
        return
    min_s = min(all_scores)
    max_s = max(all_scores)
    score_range = max_s - min_s if max_s != min_s else 1.0

    chars = " .:-=+*#%@"

    print(f"\n  Тепловая карта: {param_y} (строки) x {param_x} (столбцы)")
    print(f"  Чем темнее ('@'), тем лучше score\n")

    # Заголовок столбцов
    col_width = max(8, width // max(len(x_vals), 1))
    header = " " * 12
    for xv in x_vals:
        header += f"{str(xv):>{col_width}}"
    print(header)
    print("  " + "-" * len(header))

    for yv in y_vals:
        row_label = f"  {str(yv):>8} | "
        row = row_label
        for xv in x_vals:
            val = matrix.get((xv, yv), -float("inf"))
            if val == -float("inf") or (isinstance(val, float) and math.isnan(val)):
                cell = "?" * col_width
            else:
                normalized = (val - min_s) / score_range if score_range > 0 else 0
                char_idx = int(normalized * (len(chars) - 1))
                char_idx = max(0, min(char_idx, len(chars) - 1))
                cell = chars[char_idx] * col_width
            row += cell
        print(row)

    print(f"\n  Score range: [{min_s:.4f}, {max_s:.4f}]")


def print_bar_chart(
    data: List[Tuple[str, float]],
    title: str = "Bar Chart",
    bar_width: int = 40,
) -> None:
    """Текстовая горизонтальная диаграмма."""
    print(f"\n  {title}")
    print("  " + "=" * (bar_width + 30))

    if not data:
        print("  (пусто)")
        return

    max_val = max(abs(v) for _, v in data) or 1.0

    for label, value in data:
        bar_len = int((abs(value) / max_val) * bar_width)
        bar = "#" * bar_len
        print(f"  {label:>20s} | {bar} {value:.4f}")

    print()


def print_comparison_table(
    grid_results: List[Dict],
    random_results: List[Dict],
    top_n: int = 5,
) -> None:
    """Таблица сравнения Grid Search и Random Search."""
    print(f"\n  Сравнение: Top-{top_n} результатов")
    print("  " + "=" * 70)

    grid_top = grid_results[:top_n]
    rand_top = random_results[:top_n]

    print(f"  {'Rank':>4} | {'Grid Search':>30} | {'Random Search':>30}")
    print(f"  {'':->4}-+-{'':->30}-+-{'':->30}")

    for i in range(top_n):
        g = grid_top[i] if i < len(grid_top) else {"score": float("nan"), "params": {}}
        r = rand_top[i] if i < len(rand_top) else {"score": float("nan"), "params": {}}
        g_str = f"score={g['score']:.4f} {g['params']}"
        r_str = f"score={r['score']:.4f} {r['params']}"
        if len(g_str) > 30:
            g_str = g_str[:27] + "..."
        if len(r_str) > 30:
            r_str = r_str[:27] + "..."
        print(f"  {i+1:>4} | {g_str:>30} | {r_str:>30}")

    print()


# =============================================================================
# ДЕМО 1: Grid Search для learning rate и regularization
# =============================================================================

def demo_1_grid_search_lr_reg() -> None:
    """Grid Search для learning rate и regularization в линейной регрессии."""
    print("=" * 70)
    print("ДЕМО 1: Grid Search для learning rate и regularization")
    print("=" * 70)

    X, y = generate_regression_data(n_samples=200, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

    param_grid = {
        "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        "regularization": [0.0, 0.001, 0.01, 0.1, 1.0],
    }

    print(f"\n  Пространство параметров:")
    print(f"    learning rate:     {param_grid['lr']}")
    print(f"    regularization:   {param_grid['regularization']}")
    print(f"  Всего комбинаций: {len(param_grid['lr']) * len(param_grid['regularization'])}")

    result = grid_search(
        model_class=LinearRegressionSGD,
        param_grid=param_grid,
        X_train=X_train_inner,
        y_train=y_train_inner,
        X_val=X_val,
        y_val=y_val,
    )

    print(f"\n  Лучшие параметры: {result['best_params']}")
    print(f"  Лучший score (val -MSE): {result['best_score']:.4f}")

    # Тестовая оценка
    best_model = LinearRegressionSGD(**result["best_params"])
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"  Test score (-MSE): {test_score:.4f}")

    # Тепловая карта
    print_heatmap(result["all_results"], "lr", "regularization")

    # Топ-5 результатов
    print("  Топ-5 комбинаций:")
    for i, r in enumerate(result["all_results"][:5]):
        print(f"    {i+1}. {r['params']} -> score={r['score']:.4f}")

    print()


# =============================================================================
# ДЕМО 2: Random Search vs Grid Search
# =============================================================================

def demo_2_random_vs_grid() -> None:
    """Сравнение Random Search и Grid Search."""
    print("=" * 70)
    print("ДЕМО 2: Random Search vs Grid Search")
    print("=" * 70)

    X, y = generate_classification_data(n_samples=200, n_features=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

    param_space = {
        "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        "regularization": [0.0, 0.001, 0.01, 0.1, 1.0],
    }

    total_combos = len(param_space["lr"]) * len(param_space["regularization"])
    print(f"\n  Пространство: {total_combos} комбинаций")

    # Grid Search
    grid_result = grid_search(
        model_class=LogisticRegressionSGD,
        param_grid=param_space,
        X_train=X_train_inner,
        y_train=y_train_inner,
        X_val=X_val,
        y_val=y_val,
    )

    # Random Search с разным числом итераций
    n_iters = [5, 10, 20, total_combos]
    random_results_by_iter = {}

    for n_iter in n_iters:
        rand_result = random_search(
            model_class=LogisticRegressionSGD,
            param_distributions=param_space,
            n_iter=min(n_iter, total_combos),
            X_train=X_train_inner,
            y_train=y_train_inner,
            X_val=X_val,
            y_val=y_val,
        )
        random_results_by_iter[n_iter] = rand_result

    print(f"\n  Grid Search: лучший score = {grid_result['best_score']:.4f}")
    print(f"  Параметры: {grid_result['best_params']}")

    chart_data = []
    chart_data.append((f"Grid ({total_combos})", grid_result["best_score"]))

    print("\n  Random Search (разные n_iter):")
    for n_iter in n_iters:
        res = random_results_by_iter[n_iter]
        print(f"    n_iter={n_iter:>3d}: best score={res['best_score']:.4f}, params={res['best_params']}")
        chart_data.append((f"Random({n_iter})", res["best_score"]))

    print_bar_chart(chart_data, "Сравнение best score по методам")

    # Сравнительная таблица
    print_comparison_table(
        grid_result["all_results"],
        random_results_by_iter[total_combos]["all_results"],
        top_n=5,
    )

    # Вывод: сколько итераций нужно Random Search чтобы найти лучшее
    print("  Вывод:")
    best_grid = grid_result["best_score"]
    for n_iter in n_iters:
        found = random_results_by_iter[n_iter]["best_score"]
        pct = (found / best_grid * 100) if best_grid != 0 else 0
        print(f"    Random({n_iter}) нашёл {pct:.1f}% от лучшего Grid Search")

    print()


# =============================================================================
# ДЕМО 3: Влияние K в KNN
# =============================================================================

def demo_3_knn_k() -> None:
    """Влияние числа соседей K в KNN."""
    print("=" * 70)
    print("ДЕМО 3: Влияние K в K-ближайших соседей")
    print("=" * 70)

    X, y = generate_classification_data(n_samples=300, n_features=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

    k_values = [1, 2, 3, 5, 7, 9, 11, 15, 21, 31, 51, 71]

    print(f"\n  Тестируемые K: {k_values}")

    train_scores = []
    val_scores = []
    chart_data = []

    for k in k_values:
        model = KNNClassifier(k=k)
        model.fit(X_train_inner, y_train_inner)
        train_score = model.score(X_train_inner, y_train_inner)
        val_score = model.score(X_val, y_val)
        train_scores.append((k, train_score))
        val_scores.append((k, val_score))
        chart_data.append((f"K={k}", val_score))
        print(f"  K={k:>2d}: train_acc={train_score:.4f}, val_acc={val_score:.4f}")

    # Оптимальный K
    best_k = max(val_scores, key=lambda x: x[1])
    print(f"\n  Оптимальное K (val): {best_k[0]} (accuracy={best_k[1]:.4f})")

    # Тестовая оценка оптимальной модели
    best_model = KNNClassifier(k=best_k[0])
    best_model.fit(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    print(f"  Test accuracy: {test_acc:.4f}")

    # Визуализация: train vs val accuracy по K
    print("\n  Train vs Validation Accuracy:")
    print(f"  {'K':>4} | {'Train':>8} | {'Val':>8} | Гистограмма (train=#, val=*)")
    print("  " + "-" * 55)
    for k, (tr, vl) in zip(k_values, zip([s for _, s in train_scores], [s for _, s in val_scores])):
        bar_len = int(vl * 30)
        bar = "#" * int(tr * 30) + " | " + "*" * bar_len
        print(f"  {k:>4} | {tr:.4f}   | {vl:.4f}   | {'#' * int(tr * 30)}")
        print(f"       |         |           | {'*' * int(vl * 30)}")

    print("\n  Вывод:")
    print("    - При K=1 модель переобучается (train=100%, val низкая)")
    print("    - С увеличением K val accuracy растёт, потом падает (underfitting)")
    print("    - Оптимальный баланс где-то в середине")

    print()


# =============================================================================
# ДЕМО 4: Влияние max_depth в дереве решений
# =============================================================================

def demo_4_tree_depth() -> None:
    """Влияние max_depth в дереве решений."""
    print("=" * 70)
    print("ДЕМО 4: Влияние max_depth в дереве решений")
    print("=" * 70)

    X, y = generate_classification_data(n_samples=300, n_features=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

    depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]

    print(f"\n  Тестируемые max_depth: {depths}")

    train_scores = []
    val_scores = []

    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, min_samples_split=2)
        model.fit(X_train_inner, y_train_inner)
        train_acc = model.score(X_train_inner, y_train_inner)
        val_acc = model.score(X_val, y_val)
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        print(f"  depth={d:>3d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # График train vs val
    print("\n  Визуализация accuracy vs max_depth:")
    print(f"  {'depth':>6} | {'Train':>6} | {'Val':>6} | Гистограмма")
    print("  " + "-" * 60)
    for d, tr, vl in zip(depths, train_scores, val_scores):
        tr_bar = "#" * int(tr * 40)
        vl_bar = "*" * int(vl * 40)
        print(f"  {d:>6} | {tr:.4f} | {vl:.4f} | T:{tr_bar}")
        print(f"         |        |         | V:{vl_bar}")

    # Оптимальная глубина
    best_idx = max(range(len(val_scores)), key=lambda i: val_scores[i])
    best_depth = depths[best_idx]
    best_val = val_scores[best_idx]

    print(f"\n  Оптимальная глубина (val): {best_depth} (accuracy={best_val:.4f})")

    # Тестовая оценка
    best_model = DecisionTreeClassifier(max_depth=best_depth)
    best_model.fit(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    print(f"  Test accuracy: {test_acc:.4f}")

    # Анализ переобучения
    print("\n  Анализ переобучения:")
    for d in [1, 5, best_depth, 50]:
        if d in depths:
            idx = depths.index(d)
            gap = train_scores[idx] - val_scores[idx]
            status = "переобучение" if gap > 0.1 else "хороший баланс" if gap < 0.05 else "нормально"
            print(f"    depth={d:>3d}: train-val gap = {gap:.4f} ({status})")

    print("\n  Вывод:")
    print("    - Маленькая глубина = underfitting (обе accuracy низкие)")
    print("    - Большая глубина = overfitting (train ~100%, val падает)")
    print("    - Лучшая глубина балансирует bias и variance")

    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  МЕТОДЫ ПОДБОРА ГИПЕРПАРАМЕТРОВ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()

    demo_1_grid_search_lr_reg()
    demo_2_random_vs_grid()
    demo_3_knn_k()
    demo_4_tree_depth()

    print("=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print("""
  1. Grid Search — полный перебор, гарантированно находит лучшую
     комбинацию, но экспоненциально растёт с числом параметров.

  2. Random Search — быстрее, часто находит сопоставимый результат
     за меньше итераций. Хорошо работает когда некоторые параметры
     важнее других.

  3. K в KNN — малое K = переобучение, большое = underfitting.
     Оптимальное K зависит от данных.

  4. max_depth в дереве решений — прямая связь с переобучением.
     Нужно выбирать по validation set.

  Все методы требуют разделения данных: train / validation / test.
    """)
