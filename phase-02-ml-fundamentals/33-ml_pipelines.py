"""
ML Pipelines & Experiments — с нуля на Python
==============================================
Без sklearn. Только стандартная библиотека + math + pickle.

Содержимое:
  1. Трансформеры: StandardScaler, MinMaxScaler, PolynomialFeatures
  2. Модели: LinearRegression, LogisticRegression
  3. Pipeline — цепочка трансформаций + модель
  4. ExperimentTracker — логирование и сравнение экспериментов
  5. Сохранение/загрузка моделей через pickle
"""

import math
import random
import pickle
import os
import copy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


# ============================================================================
# Вспомогательные функции
# ============================================================================

def dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: list) -> list:
    return [list(row) for row in zip(*matrix)]


def matmul(A: list, B: list) -> list:
    BT = transpose(B)
    return [[dot(row_a, col_b) for col_b in BT] for row_a in A]


def matvec(A: list, v: list) -> list:
    return [dot(row, v) for row in A]


def eye(n: int) -> list:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_add(A: list, B: list) -> list:
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mat_scale(A: list, s: float) -> list:
    return [[a * s for a in row] for row in A]


def vec_sub(a: list, b: list) -> list:
    return [x - y for x, y in zip(a, b)]


def vec_add(a: list, b: list) -> list:
    return [x + y for x, y in zip(a, b)]


def vec_scale(v: list, s: float) -> list:
    return [x * s for x in v]


def l2_norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def print_matrix(M: list, name: str = "", cols: int = 4):
    if name:
        print(f"  {name}:")
    for row in M:
        formatted = ", ".join(f"{v:8.4f}" for v in row[:cols])
        print(f"    [{formatted}]")


def print_vector(v: list, name: str = "", fmt: str = ".4f"):
    if name:
        print(f"  {name}:")
    formatted = ", ".join(f"{x:{fmt}}" for x in v[:8])
    if len(v) > 8:
        formatted += f", ... ({len(v)} total)"
    print(f"    [{formatted}]")


# ============================================================================
# 1. Трансформеры
# ============================================================================

class Transformer:
    """Базовый класс для трансформаций."""

    def fit(self, X: list) -> "Transformer":
        raise NotImplementedError

    def transform(self, X: list) -> list:
        raise NotImplementedError

    def fit_transform(self, X: list) -> list:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: list) -> list:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class StandardScaler(Transformer):
    """Z-score нормализация: (x - mean) / std"""

    def __init__(self):
        self.mean_: list = []
        self.std_: list = []
        self.n_features_: int = 0

    def fit(self, X: list) -> "StandardScaler":
        self.n_features_ = len(X[0])
        n = len(X)
        self.mean_ = [sum(X[i][j] for i in range(n)) / n
                       for j in range(self.n_features_)]
        self.std_ = [max(
            math.sqrt(sum((X[i][j] - self.mean_[j]) ** 2 for i in range(n)) / n),
            1e-8
        ) for j in range(self.n_features_)]
        return self

    def transform(self, X: list) -> list:
        return [[(x - self.mean_[j]) / self.std_[j]
                 for j, x in enumerate(row)] for row in X]

    def inverse_transform(self, X: list) -> list:
        return [[x * self.std_[j] + self.mean_[j]
                 for j, x in enumerate(row)] for row in X]


class MinMaxScaler(Transformer):
    """Мин-макс в диапазон [0, 1]"""

    def __init__(self):
        self.min_: list = []
        self.max_: list = []

    def fit(self, X: list) -> "MinMaxScaler":
        n_feat = len(X[0])
        self.min_ = [min(X[i][j] for i in range(len(X))) for j in range(n_feat)]
        self.max_ = [max(X[i][j] for i in range(len(X))) for j in range(n_feat)]
        return self

    def transform(self, X: list) -> list:
        return [[(x - self.min_[j]) / max(self.max_[j] - self.min_[j], 1e-8)
                 for j, x in enumerate(row)] for row in X]

    def inverse_transform(self, X: list) -> list:
        return [[x * (self.max_[j] - self.min_[j]) + self.min_[j]
                 for j, x in enumerate(row)] for row in X]


class IdentityTransformer(Transformer):
    """Трансформация-заглушка (ничего не делает)."""

    def fit(self, X: list) -> "IdentityTransformer":
        return self

    def transform(self, X: list) -> list:
        return [row[:] for row in X]

    def inverse_transform(self, X: list) -> list:
        return [row[:] for row in X]


class PCA(Transformer):
    """Простая PCA: центрирование + проекция на k компонент через
    итеративную ортогонализацию (Gram-Schmidt)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean_: list = []
        self.components_: list = []

    def fit(self, X: list) -> "PCA":
        n = len(X)
        d = len(X[0])
        self.mean_ = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        centered = [[X[i][j] - self.mean_[j] for j in range(d)] for i in range(n)]

        self.components_ = []
        for _ in range(self.n_components):
            v = [random.gauss(0, 1) for _ in range(d)]
            for _ in range(20):
                Av = matvec(centered, v)
                new_v = [0.0] * d
                for row_idx in range(n):
                    for j in range(d):
                        new_v[j] += centered[row_idx][j] * Av[row_idx]
                for comp in self.components_:
                    proj = dot(new_v, comp)
                    new_v = vec_sub(new_v, vec_scale(comp, proj))
                norm = l2_norm(new_v)
                if norm > 1e-10:
                    new_v = vec_scale(new_v, 1.0 / norm)
                v = new_v
            self.components_.append(v)
        return self

    def transform(self, X: list) -> list:
        d = len(X[0])
        centered = [[X[i][j] - self.mean_[j] for j in range(d)] for i in range(len(X))]
        return [[dot(row, comp) for comp in self.components_] for row in centered]

    def inverse_transform(self, X: list) -> list:
        n = len(X)
        d = len(self.components_)
        result = []
        for i in range(n):
            row = self.mean_[:]
            for j in range(d):
                row = vec_add(row, vec_scale(self.components_[j], X[i][j]))
            result.append(row)
        return result


# ============================================================================
# 2. Модели
# ============================================================================

class BaseModel:
    """Базовый класс модели."""

    def fit(self, X: list, y: list) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: list) -> list:
        raise NotImplementedError

    def score(self, X: list, y: list) -> float:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class LinearRegression(BaseModel):
    """Линейная регрессия: градиентный спуск."""

    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights: list = []
        self.bias: float = 0.0

    def fit(self, X: list, y: list) -> "LinearRegression":
        n, d = len(X), len(X[0])
        self.weights = [random.gauss(0, 0.01) for _ in range(d)]
        self.bias = 0.0

        for _ in range(self.n_iter):
            preds = self.predict(X)
            errors = vec_sub(preds, y)
            grad_w = [sum(errors[i] * X[i][j] for i in range(n)) / n
                       for j in range(d)]
            grad_b = sum(errors) / n
            self.weights = vec_sub(self.weights, vec_scale(grad_w, self.lr))
            self.bias -= self.lr * grad_b
        return self

    def predict(self, X: list) -> list:
        return [dot(row, self.weights) + self.bias for row in X]

    def score(self, X: list, y: list) -> float:
        preds = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((a - b) ** 2 for a, b in zip(y, preds))
        ss_tot = sum((a - y_mean) ** 2 for a in y)
        return 1.0 - ss_res / max(ss_tot, 1e-8)


class LogisticRegression(BaseModel):
    """Логистическая регрессия: градиентный спуск."""

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights: list = []
        self.bias: float = 0.0

    def _sigmoid(self, z: float) -> float:
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: list, y: list) -> "LogisticRegression":
        n, d = len(X), len(X[0])
        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.n_iter):
            preds = self.predict_proba(X)
            errors = vec_sub(preds, y)
            grad_w = [sum(errors[i] * X[i][j] for i in range(n)) / n
                       for j in range(d)]
            grad_b = sum(errors) / n
            self.weights = vec_sub(self.weights, vec_scale(grad_w, self.lr))
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X: list) -> list:
        return [self._sigmoid(dot(row, self.weights) + self.bias) for row in X]

    def predict(self, X: list, threshold: float = 0.5) -> list:
        return [1.0 if p >= threshold else 0.0 for p in self.predict_proba(X)]

    def score(self, X: list, y: list) -> float:
        preds = self.predict(X)
        return sum(1 for a, b in zip(preds, y) if a == b) / len(y)


# ============================================================================
# 3. Pipeline
# ============================================================================

class Pipeline:
    """Цепочка трансформаций + финальная модель.

    Пример:
        pipe = Pipeline(
            steps=[("scaler", StandardScaler()), ("pca", PCA(2))],
            model=LinearRegression(lr=0.01, n_iter=500)
        )
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
    """

    def __init__(self, steps: List[Tuple[str, Transformer]], model: BaseModel):
        self.steps = steps
        self.model = model
        self._fitted = False

    def _apply_transforms(self, X: list, inverse: bool = False) -> list:
        if inverse:
            for name, t in reversed(self.steps):
                X = t.inverse_transform(X)
            return X
        for name, t in self.steps:
            X = t.transform(X)
        return X

    def fit(self, X: list, y: list) -> "Pipeline":
        for name, t in self.steps:
            X = t.fit_transform(X)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: list) -> list:
        X_t = self._apply_transforms(X)
        return self.model.predict(X_t)

    def score(self, X: list, y: list) -> float:
        X_t = self._apply_transforms(X)
        return self.model.score(X_t, y)

    def get_params(self) -> dict:
        return {
            "steps": [(n, str(t)) for n, t in self.steps],
            "model": str(self.model),
        }

    def summary(self) -> str:
        parts = ["Pipeline:"]
        for name, t in self.steps:
            parts.append(f"  [{name}] {t}")
        parts.append(f"  [model] {self.model}")
        return "\n".join(parts)

    def __repr__(self):
        return self.summary()


# ============================================================================
# 4. Experiment Tracker
# ============================================================================

class ExperimentRecord:
    """Один эксперимент."""

    def __init__(self, name: str, params: dict, metrics: dict):
        self.name = name
        self.params = params
        self.metrics = metrics
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __repr__(self):
        return f"Experiment({self.name}, metrics={self.metrics})"


class ExperimentTracker:
    """Сбор и сравнение результатов экспериментов."""

    def __init__(self):
        self.records: List[ExperimentRecord] = []
        self.log_lines: List[str] = []

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.log_lines.append(line)
        print(f"  LOG: {message}")

    def record(self, name: str, params: dict, metrics: dict) -> ExperimentRecord:
        exp = ExperimentRecord(name, params, metrics)
        self.records.append(exp)
        self.log(f"Recorded: {name} | metrics={metrics}")
        return exp

    def compare(self, metric_key: str = "score") -> List[dict]:
        """Сортировка экспериментов по метрике (убывание)."""
        items = []
        for r in self.records:
            if metric_key in r.metrics:
                items.append({
                    "name": r.name,
                    metric_key: r.metrics[metric_key],
                    "params": r.params,
                    "timestamp": r.timestamp,
                })
        items.sort(key=lambda x: x[metric_key], reverse=True)
        return items

    def best(self, metric_key: str = "score") -> Optional[dict]:
        ranked = self.compare(metric_key)
        return ranked[0] if ranked else None

    def print_comparison(self, metric_key: str = "score"):
        ranked = self.compare(metric_key)
        if not ranked:
            print("  Нет экспериментов для сравнения.")
            return
        print(f"\n  {'Ранг':<5} {'Эксперимент':<30} {metric_key:<12} {'Параметры'}")
        print(f"  {'─'*5} {'─'*30} {'─'*12} {'─'*30}")
        for i, r in enumerate(ranked, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            print(f"  {i:<5} {r['name']:<30} {r[metric_key]:<12.4f} {params_str}")

    def save_log(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(line + "\n")
        print(f"\n  Лог сохранён: {path}")

    def print_all_records(self):
        print(f"\n  Всего экспериментов: {len(self.records)}")
        for i, r in enumerate(self.records, 1):
            print(f"  {i}. {r.name} @ {r.timestamp}")
            print(f"     Params: {r.params}")
            print(f"     Metrics: {r.metrics}")


# ============================================================================
# 5. Сохранение / загрузка моделей
# ============================================================================

def save_model(model: Any, path: str):
    """Сохранение Pipeline или модели через pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Модель сохранена: {path} ({os.path.getsize(path)} bytes)")


def load_model(path: str) -> Any:
    """Загрузка модели из pickle-файла."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Модель загружена: {path}")
    return model


# ============================================================================
# Генерация данных
# ============================================================================

def make_regression_data(n: int = 100, n_features: int = 3,
                         noise: float = 0.5, seed: int = 42):
    """Линейная регрессия: y = w·x + b + noise."""
    random.seed(seed)
    weights_true = [random.uniform(-3, 3) for _ in range(n_features)]
    bias_true = random.uniform(-5, 5)
    X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n)]
    y = [sum(w * x for w, x in zip(weights_true, row)) + bias_true
         + random.gauss(0, noise) for row in X]
    return X, y, weights_true, bias_true


def make_classification_data(n: int = 100, n_features: int = 2,
                             seed: int = 42):
    """Бинарная классификация: y = 1 если w·x + b > 0."""
    random.seed(seed)
    weights_true = [random.uniform(-2, 2) for _ in range(n_features)]
    bias_true = random.uniform(-2, 2)
    X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n)]
    logits = [sum(w * x for w, x in zip(weights_true, row)) + bias_true
              for row in X]
    y = [1.0 if l > 0 else 0.0 for l in logits]
    return X, y


def train_test_split(X: list, y: list, test_ratio: float = 0.2,
                     seed: int = 42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


# ============================================================================
# ДЕМОНСТРАЦИИ
# ============================================================================

def demo_pipeline():
    """Демо 1: Пайплайн — нормализация → PCA → модель."""
    print("=" * 70)
    print("  ДЕМО 1: Pipeline (трансформации + модель)")
    print("=" * 70)

    X, y, w_true, b_true = make_regression_data(
        n=200, n_features=5, noise=0.5, seed=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    print(f"\n  Данные: {len(X_train)} train, {len(X_test)} test, "
          f"{len(X[0])} признаков")
    print_vector(w_true, "Истинные веса")
    print(f"  Истинное смещение: {b_true:.4f}")

    # --- Pipeline: StandardScaler → PCA(3) → LinearRegression ---
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=3)),
        ],
        model=LinearRegression(lr=0.05, n_iter=800),
    )
    print(f"\n  {pipe.summary()}")

    pipe.fit(X_train, y_train)
    r2 = pipe.score(X_test, y_test)
    print(f"\n  R² на тесте: {r2:.4f}")

    preds = pipe.predict(X_test)
    mse = sum((p - t) ** 2 for p, t in zip(preds, y_test)) / len(y_test)
    print(f"  MSE на тесте: {mse:.4f}")

    # --- Без пайплайна: та же модель на сырых данных ---
    print(f"\n  --- Сравнение: без пайплайна ---")
    lr_plain = LinearRegression(lr=0.05, n_iter=800)
    lr_plain.fit(X_train, y_train)
    r2_plain = lr_plain.score(X_test, y_test)
    preds_plain = lr_plain.predict(X_test)
    mse_plain = sum((p - t) ** 2 for p, t in zip(preds_plain, y_test)) / len(y_test)
    print(f"  R² (без пайплайна): {r2_plain:.4f}")
    print(f"  MSE (без пайплайна): {mse_plain:.4f}")

    print(f"\n  Pipeline показал R²={r2:.4f} vs {r2_plain:.4f} (без пайплайна)")

    return pipe


def demo_experiments():
    """Демо 2: Сравнение нескольких экспериментов."""
    print("\n" + "=" * 70)
    print("  ДЕМО 2: Сравнение экспериментов")
    print("=" * 70)

    X, y, _, _ = make_regression_data(n=150, n_features=4, noise=1.0, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    tracker = ExperimentTracker()

    configs = [
        ("Baseline (сырые данные)",
         {"steps": [], "lr": 0.01, "n_iter": 500},
         IdentityTransformer(), 500, 0.01),
        ("StandardScaler → LinearReg",
         {"steps": ["StandardScaler"], "lr": 0.05, "n_iter": 500},
         StandardScaler(), 500, 0.05),
        ("MinMaxScaler → LinearReg",
         {"steps": ["MinMaxScaler"], "lr": 0.05, "n_iter": 500},
         MinMaxScaler(), 500, 0.05),
        ("StandardScaler → PCA(2) → LR",
         {"steps": ["StandardScaler", "PCA(2)"], "lr": 0.05, "n_iter": 800},
         None, 800, 0.05),
        ("StandardScaler → PCA(3) → LR",
         {"steps": ["StandardScaler", "PCA(3)"], "lr": 0.05, "n_iter": 800},
         None, 800, 0.05),
    ]

    for name, params, transformer, n_iter, lr in configs:
        tracker.log(f"Запуск: {name}")

        if transformer is not None:
            pipe = Pipeline(
                steps=[("transform", transformer)],
                model=LinearRegression(lr=lr, n_iter=n_iter),
            )
        else:
            # StandardScaler + PCA
            steps = [("scaler", StandardScaler())]
            if "PCA(2)" in str(params["steps"]):
                steps.append(("pca", PCA(2)))
            else:
                steps.append(("pca", PCA(3)))
            pipe = Pipeline(
                steps=steps,
                model=LinearRegression(lr=lr, n_iter=n_iter),
            )

        pipe.fit(X_train, y_train)
        r2 = pipe.score(X_test, y_test)
        preds = pipe.predict(X_test)
        mse = sum((p - t) ** 2 for p, t in zip(preds, y_test)) / len(y_test)

        metrics = {"r2": round(r2, 4), "mse": round(mse, 4)}
        tracker.record(name, params, metrics)

    tracker.print_comparison("r2")
    best = tracker.best("r2")
    print(f"\n  Лучший: {best['name']} (R² = {best['r2']:.4f})")

    return tracker


def demo_metric_logging():
    """Демо 3: Детальное логирование метрик."""
    print("\n" + "=" * 70)
    print("  ДЕМО 3: Логирование метрик эксперимента")
    print("=" * 70)

    X, y = make_classification_data(n=200, n_features=3, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    tracker = ExperimentTracker()
    tracker.log("Начало классификационного эксперимента")

    # --- Эксперимент 1: StandardScaler + LogisticRegression ---
    tracker.log("Эксперимент A: StandardScaler + LogisticRegression")
    pipe_a = Pipeline(
        steps=[("scaler", StandardScaler())],
        model=LogisticRegression(lr=0.5, n_iter=1000),
    )
    pipe_a.fit(X_train, y_train)
    acc_a = pipe_a.score(X_test, y_test)
    preds_a = pipe_a.predict(X_test)

    tp = sum(1 for p, t in zip(preds_a, y_test) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds_a, y_test) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds_a, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds_a, y_test) if p == 0 and t == 1)
    precision_a = tp / max(tp + fp, 1)
    recall_a = tp / max(tp + fn, 1)
    f1_a = 2 * precision_a * recall_a / max(precision_a + recall_a, 1e-8)

    metrics_a = {
        "accuracy": round(acc_a, 4),
        "precision": round(precision_a, 4),
        "recall": round(recall_a, 4),
        "f1": round(f1_a, 4),
        "confusion_matrix": f"TN={tn} FP={fp} FN={fn} TP={tp}",
    }
    tracker.record("A: StdScaler + LogReg", {"lr": 0.5, "n_iter": 1000}, metrics_a)

    print(f"\n  Метрики эксперимента A:")
    for k, v in metrics_a.items():
        print(f"    {k}: {v}")

    # --- Эксперимент 2: PCA(2) + LogisticRegression ---
    tracker.log("Эксперимент B: PCA(2) + LogisticRegression")
    pipe_b = Pipeline(
        steps=[("scaler", StandardScaler()), ("pca", PCA(2))],
        model=LogisticRegression(lr=0.5, n_iter=1000),
    )
    pipe_b.fit(X_train, y_train)
    acc_b = pipe_b.score(X_test, y_test)
    preds_b = pipe_b.predict(X_test)

    tp = sum(1 for p, t in zip(preds_b, y_test) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds_b, y_test) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds_b, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds_b, y_test) if p == 0 and t == 1)
    precision_b = tp / max(tp + fp, 1)
    recall_b = tp / max(tp + fn, 1)
    f1_b = 2 * precision_b * recall_b / max(precision_b + recall_b, 1e-8)

    metrics_b = {
        "accuracy": round(acc_b, 4),
        "precision": round(precision_b, 4),
        "recall": round(recall_b, 4),
        "f1": round(f1_b, 4),
        "confusion_matrix": f"TN={tn} FP={fp} FN={fn} TP={tp}",
    }
    tracker.record("B: PCA(2) + LogReg", {"lr": 0.5, "n_iter": 1000, "pca": 2}, metrics_b)

    print(f"\n  Метрики эксперимента B:")
    for k, v in metrics_b.items():
        print(f"    {k}: {v}")

    # --- Сравнение ---
    tracker.print_comparison("accuracy")

    # --- Сохранение лучшей модели ---
    best = tracker.best("accuracy")
    print(f"\n  Лучшая модель: {best['name']} (accuracy = {best['accuracy']:.4f})")

    return tracker


def demo_save_load():
    """Бонус: сохранение и загрузка модели."""
    print("\n" + "=" * 70)
    print("  БОНУС: Сохранение / загрузка модели (pickle)")
    print("=" * 70)

    X, y, _, _ = make_regression_data(n=100, n_features=3, noise=0.3, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    pipe = Pipeline(
        steps=[("scaler", StandardScaler())],
        model=LinearRegression(lr=0.05, n_iter=600),
    )
    pipe.fit(X_train, y_train)
    score_before = pipe.score(X_test, y_test)
    print(f"\n  R² перед сохранением: {score_before:.4f}")

    # Сохраняем
    save_path = os.path.join(os.path.dirname(__file__) or ".", "saved_pipeline.pkl")
    save_model(pipe, save_path)

    # Загружаем
    loaded = load_model(save_path)
    score_after = loaded.score(X_test, y_test)
    print(f"  R² после загрузки: {score_after:.4f}")
    print(f"  Совпадают: {'ДА' if abs(score_before - score_after) < 1e-6 else 'НЕТ'}")

    # Удаляем файл
    os.remove(save_path)
    print(f"  Файл удалён.")

    # Также сохраняем экспериментальный трекер
    tracker = ExperimentTracker()
    tracker.record("test", {"lr": 0.05}, {"r2": score_before})
    tracker_path = os.path.join(os.path.dirname(__file__) or ".", "tracker_log.txt")
    tracker.save_log(tracker_path)
    os.remove(tracker_path)


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     ML Pipelines & Experiments — с нуля на Python          ║")
    print("║     (без sklearn)                                          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    demo_pipeline()
    demo_experiments()
    demo_metric_logging()
    demo_save_load()

    print("\n" + "=" * 70)
    print("  Все демо завершены!")
    print("=" * 70)
