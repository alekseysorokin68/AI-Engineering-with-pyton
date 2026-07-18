"""
38. Отбор признаков (Feature Selection) — с нуля, без sklearn.

Filter methods:  корреляция, взаимная информация
Wrapper methods: forward selection, backward elimination
Embedded methods: L1-регуляризация (Lasso)

Все алгоритмы реализованы на чистом Python + numpy.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable

np.random.seed(42)


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Корреляция Пирсона."""
    x, y = x - x.mean(), y - y.mean()
    num = np.dot(x, y)
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return num / den if den > 0 else 0.0


def spearman_rank(arr: np.ndarray) -> np.ndarray:
    """Ранговая трансформация (аналог scipy.stats.rankdata)."""
    ranks = np.empty_like(arr, dtype=float)
    temp = arr.argsort()
    ranks[temp] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Корреляция Спирмена через ранги."""
    return pearsonr(spearman_rank(x), spearman_rank(y))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Взаимная информация I(X;Y) — непараметрическая оценка через гистограмму.
    """
    xy = np.column_stack([x, y])
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-нормализация по столбцам."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-12
    return (X - mu) / sigma, mu, sigma


# ─────────────────────────────────────────────
# Линейная модель (мини-реализация)
# ─────────────────────────────────────────────

class LinearModel:
    """Мини-класс для линейной регрессии / классификации."""

    def __init__(self, n_features: int):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def fit_ols(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01,
                epochs: int = 3000):
        """Обучение градиентным спуском (OLS)."""
        n = len(X)
        for _ in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            dw = (2 / n) * (X.T @ error)
            db = (2 / n) * error.sum()
            self.w -= lr * dw
            self.b -= lr * db

    def fit_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
                  lr: float = 0.01, epochs: int = 3000):
        """Ridge-регрессия (L2)."""
        n = len(X)
        for _ in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            dw = (2 / n) * (X.T @ error) + alpha * self.w
            db = (2 / n) * error.sum()
            self.w -= lr * dw
            self.b -= lr * db

    def fit_lasso(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1,
                  lr: float = 0.01, epochs: int = 3000):
        """Lasso-регрессия (L1) — субградиент."""
        n = len(X)
        for _ in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            dw = (2 / n) * (X.T @ error)
            # L1-штраф: субградиент
            dw += alpha * np.sign(self.w)
            db = (2 / n) * error.sum()
            self.w -= lr * dw
            self.b -= lr * db


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


# ─────────────────────────────────────────────
# 1. FILTER METHODS
# ─────────────────────────────────────────────

def filter_correlation(X: np.ndarray, y: np.ndarray,
                       threshold: float = 0.2) -> List[int]:
    """
    Отбор признаков по абсолютной корреляции с целевой переменной.
    Остаются только признаки с |r| >= threshold.
    """
    scores = {}
    for j in range(X.shape[1]):
        scores[j] = abs(pearsonr(X[:, j], y))
    selected = [j for j, s in scores.items() if s >= threshold]
    return sorted(selected, key=lambda j: scores[j], reverse=True)


def filter_mutual_information(X: np.ndarray, y: np.ndarray,
                              k: int = 5) -> List[int]:
    """
    Отбор k признаков с наибольшей взаимной информацией с y.
    """
    scores = {}
    for j in range(X.shape[1]):
        scores[j] = mutual_information(X[:, j], y, bins=15)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]


# ─────────────────────────────────────────────
# 2. WRAPPER METHODS
# ─────────────────────────────────────────────

def forward_selection(X: np.ndarray, y: np.ndarray,
                      k_features: int = 5,
                      metric_fn: Optional[Callable] = None) -> List[int]:
    """
    Пошаговый (forward) отбор: на каждом шаге добавляем признак,
    дающий наибольшее улучшение метрики.
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp: -mse(yt, yp)   # максимизируем (-MSE)

    n_samples, n_feats = X.shape
    selected: List[int] = []
    remaining = list(range(n_feats))
    best_score = -np.inf

    for step in range(min(k_features, n_feats)):
        best_feat = -1
        for feat in remaining:
            trial = selected + [feat]
            model = LinearModel(len(trial))
            X_sub = X[:, trial]
            model.fit_ols(X_sub, y, lr=0.05, epochs=5000)
            score = metric_fn(y, model.predict(X_sub))
            if score > best_score:
                best_score = score
                best_feat = feat
        if best_feat == -1:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)

    return selected


def backward_elimination(X: np.ndarray, y: np.ndarray,
                         k_features: int = 3,
                         metric_fn: Optional[Callable] = None) -> List[int]:
    """
    Обратное (backward) исключение: начинаем со всех признаков,
    на каждом шаге удаляем наименее важный.
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp: -mse(yt, yp)

    n_feats = X.shape[1]
    selected = list(range(n_feats))

    while len(selected) > k_features:
        worst_feat = -1
        best_score = -np.inf
        for feat in selected:
            trial = [f for f in selected if f != feat]
            if not trial:
                continue
            model = LinearModel(len(trial))
            X_sub = X[:, trial]
            model.fit_ols(X_sub, y, lr=0.05, epochs=5000)
            score = metric_fn(y, model.predict(X_sub))
            if score > best_score:
                best_score = score
                worst_feat = feat
        if worst_feat == -1:
            break
        selected.remove(worst_feat)

    return selected


# ─────────────────────────────────────────────
# 3. EMBEDDED METHOD — L1 (Lasso)
# ─────────────────────────────────────────────

def embedded_lasso(X: np.ndarray, y: np.ndarray,
                   alpha: float = 0.1,
                   threshold: float = 1e-3) -> List[int]:
    """
    Lasso-регреляция: признаки с |w| > threshold считаются отобранными.
    """
    model = LinearModel(X.shape[1])
    model.fit_lasso(X, y, alpha=alpha, lr=0.01, epochs=5000)
    selected = [j for j in range(len(model.w)) if abs(model.w[j]) > threshold]
    return selected


# ═══════════════════════════════════════════════
# ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════

def generate_dataset(n_samples=200, n_features=10, seed=42):
    """
    Генерирует синтетический датасет:
    - 3 информативных признака (линейно связаны с y)
    - 7 шумовых признаков
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)

    # Целевая функция — линейная комбинация первых 3 + шум
    true_weights = np.zeros(n_features)
    true_weights[0] = 3.0
    true_weights[2] = -2.0
    true_weights[5] = 1.5
    y = X @ true_weights + rng.randn(n_samples) * 0.5

    feature_names = [f"f{j}" for j in range(n_features)]
    return X, y, feature_names, true_weights


def demo_1():
    """Демо 1: Отбор по корреляции с целевой переменной."""
    print("=" * 65)
    print("  ДЕМО 1: FILTER — Корреляция Пирсона и взаимная информация")
    print("=" * 65)

    X, y, names, tw = generate_dataset()

    # Корреляция
    print("\n▸ Корреляция Пирсона каждого признака с y:")
    print("-" * 45)
    for j in range(X.shape[1]):
        r = pearsonr(X[:, j], y)
        bar = "█" * int(abs(r) * 20)
        marker = "  ← информативный" if j in [0, 2, 5] else ""
        print(f"  {names[j]}:  r = {r:+.3f}  {bar}{marker}")

    selected_corr = filter_correlation(X, y, threshold=0.15)
    print(f"\n  Отобраны (|r| >= 0.15): {[names[j] for j in selected_corr]}")

    # Взаимная информация
    print("\n▸ Взаимная информация I(X;Y):")
    print("-" * 45)
    for j in range(X.shape[1]):
        mi = mutual_information(X[:, j], y, bins=15)
        bar = "█" * int(mi * 20)
        marker = "  ← информативный" if j in [0, 2, 5] else ""
        print(f"  {names[j]}:  MI = {mi:.4f}  {bar}{marker}")

    selected_mi = filter_mutual_information(X, y, k=3)
    print(f"\n  Top-3 по MI: {[names[j] for j in selected_mi]}")
    print()


def demo_2():
    """Демо 2: Forward selection (обёрточный метод)."""
    print("=" * 65)
    print("  ДЕМО 2: WRAPPER — Forward Selection")
    print("=" * 65)

    X, y, names, tw = generate_dataset()

    print("\n▸ Пошаговый отбор (добавляем по одному признаку):\n")
    for step in range(1, 6):
        sel = forward_selection(X, y, k_features=step)
        model = LinearModel(len(sel))
        X_sub = X[:, sel]
        model.fit_ols(X_sub, y, lr=0.05, epochs=5000)
        score = mse(y, model.predict(X_sub))
        print(f"  Шаг {step}:  отобраны {[names[j] for j in sel]}"
              f"  → MSE = {score:.4f}")

    print("\n▸ Сравнение: forward vs backward")
    print("-" * 50)

    fwd = forward_selection(X, y, k_features=3)
    bwd = backward_elimination(X, y, k_features=3)

    print(f"  Forward selection (3 фичи): {[names[j] for j in fwd]}")
    print(f"  Backward elimination (3 фичи): {[names[j] for j in bwd]}")

    # Сравним MSE
    model_fwd = LinearModel(len(fwd))
    model_fwd.fit_ols(X[:, fwd], y, lr=0.05, epochs=5000)
    mse_fwd = mse(y, model_fwd.predict(X[:, fwd]))

    model_bwd = LinearModel(len(bwd))
    model_bwd.fit_ols(X[:, bwd], y, lr=0.05, epochs=5000)
    mse_bwd = mse(y, model_bwd.predict(X[:, bwd]))

    print(f"  MSE (forward):  {mse_fwd:.4f}")
    print(f"  MSE (backward): {mse_bwd:.4f}")
    print()


def demo_3():
    """Демо 3: L1-регуляризация (Lasso) — обнуление весов."""
    print("=" * 65)
    print("  ДЕМО 3: EMBEDDED — Lasso (L1-регуляризация)")
    print("=" * 65)

    X, y, names, tw = generate_dataset()
    X_std = standardize(X)[0]

    alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

    print("\n▸ Веса при разных значениях alpha:\n")
    print(f"  {'alpha':>7s}", end="")
    for j in range(X.shape[1]):
        print(f"  {names[j]:>6s}", end="")
    print(f"  {'MSE':>8s}")
    print("  " + "-" * 65)

    for alpha in alphas:
        model = LinearModel(X_std.shape[1])
        model.fit_lasso(X_std, y, alpha=alpha, lr=0.01, epochs=5000)
        score = mse(y, model.predict(X_std))
        print(f"  {alpha:7.3f}", end="")
        for j in range(len(model.w)):
            w = model.w[j]
            if abs(w) < 1e-3:
                print(f"  {'0':>6s}", end="")
            else:
                print(f"  {w:+6.2f}", end="")
        print(f"  {score:8.4f}")

    print("\n▸ При alpha = 0.1 отбираются признаки:")
    sel = embedded_lasso(X_std, y, alpha=0.1)
    print(f"  {[names[j] for j in sel]}")
    print(f"  Настоящие информативные: {[names[j] for j in [0, 2, 5]]}")

    # Показываем, как shapely-нулевые веса исключают шум
    model = LinearModel(X_std.shape[1])
    model.fit_lasso(X_std, y, alpha=0.1, lr=0.01, epochs=5000)
    print("\n▸ Все веса Lasso (alpha=0.1):")
    for j in range(len(model.w)):
        marker = "✓" if abs(model.w[j]) > 1e-3 else "✗ (обнулён)"
        print(f"  {names[j]}: w = {model.w[j]:+.4f}  {marker}")
    print()


def demo_4():
    """Демо 4: Сравнение всех методов."""
    print("=" * 65)
    print("  ДЕМО 4: СРАВНЕНИЕ МЕТОДОВ ОТБОРА ПРИЗНАКОВ")
    print("=" * 65)

    X, y, names, tw = generate_dataset()
    X_std = standardize(X)[0]

    true_features = {0, 2, 5}

    # Методы
    methods = {}

    # Filter: корреляция
    sel_corr = filter_correlation(X, y, threshold=0.15)
    methods["Filter (correlation)"] = set(sel_corr)

    # Filter: MI
    sel_mi = filter_mutual_information(X, y, k=4)
    methods["Filter (MI, top-4)"] = set(sel_mi)

    # Wrapper: forward
    sel_fwd = forward_selection(X, y, k_features=4)
    methods["Forward selection"] = set(sel_fwd)

    # Wrapper: backward
    sel_bwd = backward_elimination(X, y, k_features=4)
    methods["Backward elimination"] = set(sel_bwd)

    # Embedded: Lasso
    sel_lasso = embedded_lasso(X_std, y, alpha=0.1)
    methods["Lasso (L1)"] = set(sel_lasso)

    print(f"\n  Настоящие информативные признаки: "
          f"{[names[j] for j in sorted(true_features)]}\n")

    print(f"  {'Метод':<25s} {'Отобраны':<20s} {'TP':>3s} {'FP':>3s} {'FN':>3s}")
    print("  " + "-" * 55)

    results = {}
    for name, sel in methods.items():
        tp = len(sel & true_features)
        fp = len(sel - true_features)
        fn = len(true_features - sel)
        selected_str = ", ".join(sorted(names[j] for j in sel))
        print(f"  {name:<25s} {selected_str:<20s} {tp:>3d} {fp:>3d} {fn:>3d}")

    # Итоговая таблица MSE
    print("\n▸ MSE моделей с отобранными признаками:")
    print("-" * 55)
    for name, sel in methods.items():
        if len(sel) == 0:
            print(f"  {name:<25s} — нет признаков!")
            continue
        idx = sorted(sel)
        model = LinearModel(len(idx))
        model.fit_ols(X[:, idx], y, lr=0.05, epochs=5000)
        score = mse(y, model.predict(X[:, idx]))
        print(f"  {name:<25s} MSE = {score:.4f}")

    # Модель со всеми признаками
    model_all = LinearModel(X.shape[1])
    model_all.fit_ols(X, y, lr=0.05, epochs=5000)
    mse_all = mse(y, model_all.predict(X))
    print(f"  {'Все признаки (baseline)':<25s} MSE = {mse_all:.4f}")

    # Модель только с настоящими
    model_true = LinearModel(3)
    model_true.fit_ols(X[:, [0, 2, 5]], y, lr=0.05, epochs=5000)
    mse_true = mse(y, model_true.predict(X[:, [0, 2, 5]]))
    print(f"  {'Только истинные фичи':<25s} MSE = {mse_true:.4f}")

    print("\n  ВЫВОД: Лучший метод отбора — тот, что минимизирует MSE")
    print("  при минимальном числе признаков (принцип парсимонии).")
    print()


# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  38. ОТБОР ПРИЗНАКОВ (FEATURE SELECTION) — РЕАЛИЗАЦИЯ С НУЛЯ")
    print("  Все методы: filter / wrapper / embedded")
    print("=" * 65 + "\n")

    demo_1()
    demo_2()
    demo_3()
    demo_4()

    print("=" * 65)
    print("  Все демонстрации завершены.")
    print("=" * 65)
