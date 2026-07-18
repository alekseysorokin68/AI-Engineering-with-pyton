"""
29. Оценка моделей (Model Evaluation)
======================================
Основы оценки качества моделей машинного обучения с нуля на Python.

Содержание:
    1. Train/Test split
    2. K-Fold Cross-Validation
    3. Метрики: Accuracy, Precision, Recall, F1, MSE, R²
    4. Confusion Matrix
    5. ROC curve и AUC

Все реализовано без sklearn — только numpy и собственные функции.
"""

import random
import math

# ============================================================
# Вспомогательные функции
# ============================================================

def generate_data(n_samples=200, seed=42):
    """Генерация синтетических данных для классификации (2 класса)."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        # Линейная граница: x1 + x2 > 0 → класс 1, иначе → класс 0
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        data.append(([x1, x2], label))
    return data


def generate_regression_data(n_samples=100, seed=42):
    """Генерация синтетических данных для регрессии."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        x = random.uniform(0, 10)
        noise = random.gauss(0, 1)
        y = 2.5 * x + 3.0 + noise  # y = 2.5x + 3 + шум
        data.append(([x], y))
    return data


def simple_linear_regression(X_train, y_train):
    """Простая линейная регрессия (одна переменная) методом наименьших квадратов."""
    x_vals = [row[0] for row in X_train]
    y_vals = list(y_train)
    n = len(x_vals)
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - mean_x) ** 2 for x in x_vals)

    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    return slope, intercept


def simple_logistic_regression(X_train, y_train, lr=0.1, epochs=200):
    """Простая логистическая регрессия градиентным спуском."""
    n_features = len(X_train[0])
    weights = [0.0] * n_features
    bias = 0.0

    def sigmoid(z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    for _ in range(epochs):
        for i in range(len(X_train)):
            z = bias + sum(w * x for w, x in zip(weights, X_train[i]))
            pred = sigmoid(z)
            error = pred - y_train[i]
            for j in range(n_features):
                weights[j] -= lr * error * X_train[i][j]
            bias -= lr * error

    return weights, bias


def predict_logistic(X, weights, bias):
    """Предсказание логистической регрессии."""
    preds = []
    for row in X:
        z = bias + sum(w * x for w, x in zip(weights, row))
        prob = 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))
        preds.append(prob)
    return preds


def predict_linear(X, slope, intercept):
    """Предсказание линейной регрессии."""
    return [slope * row[0] + intercept for row in X]


def train_test_split(data, test_ratio=0.3, seed=42):
    """
    Разделение данных на обучающую и тестовую выборки.
    
    Args:
        data: список кортежей (features, label)
        test_ratio: доля тестовых данных (по умолчанию 0.3 = 30%)
        seed: зерно генератора случайных чисел
    
    Returns:
        (X_train, y_train, X_test, y_test)
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_ratio)
    test_set = shuffled[:n_test]
    train_set = shuffled[n_test:]

    X_train = [item[0] for item in train_set]
    y_train = [item[1] for item in train_set]
    X_test = [item[0] for item in test_set]
    y_test = [item[1] for item in test_set]

    return X_train, y_train, X_test, y_test


# ============================================================
# МЕТРИКИ
# ============================================================

def accuracy(y_true, y_pred):
    """Доля правильных ответов: (TP + TN) / (TP + TN + FP + FN)"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if len(y_true) > 0 else 0.0


def precision(y_true, y_pred, positive=1):
    """Точность (Precision): TP / (TP + FP)"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, positive=1):
    """Полнота (Recall): TP / (TP + FN)"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, positive=1):
    """F1-мера: гармоническое среднее Precision и Recall"""
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mse(y_true, y_pred):
    """Среднеквадратичная ошибка: mean((y_true - y_pred)^2)"""
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n if n > 0 else 0.0


def r_squared(y_true, y_pred):
    """Коэффициент детерминации R²: 1 - SS_res / SS_tot"""
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_y = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def confusion_matrix(y_true, y_pred, positive=1):
    """
    Матрица ошибок (Confusion Matrix):
    
                    Predicted
                    Pos     Neg
    Actual  Pos    TP      FN
            Neg    FP      TN
    
    Returns: (tp, fp, fn, tn)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p != positive)
    return tp, fp, fn, tn


def roc_curve(y_true, y_scores, thresholds=None):
    """
    ROC-кривая (упрощённо).
    
    Для каждого порога threshold:
        - Если score >= threshold → предсказание = 1
        - Вычисляем TPR и FPR
    
    Returns: (fpr_list, tpr_list, threshold_list)
    """
    if thresholds is None:
        thresholds = sorted(set(y_scores), reverse=True)
        thresholds = [t + 0.001 for t in thresholds]  # небольшой сдвиг

    tp_all = sum(1 for t in y_true if t == 1)
    tn_all = sum(1 for t in y_true if t != 1)

    fpr_list = []
    tpr_list = []

    for threshold in thresholds:
        tp = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s >= threshold)
        fp = sum(1 for t, s in zip(y_true, y_scores) if t != 1 and s >= threshold)
        fn = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s < threshold)
        tn = sum(1 for t, s in zip(y_true, y_scores) if t != 1 and s < threshold)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc(fpr_list, tpr_list):
    """
    Площадь под ROC-кривой (AUC) методом трапеций.
    """
    # Сортируем по FPR
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        dx = fpr_sorted[i] - fpr_sorted[i - 1]
        avg_h = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2.0
        area += dx * avg_h

    return abs(area)


# ============================================================
# K-Fold Cross-Validation
# ============================================================

def k_fold_split(data, k, seed=42):
    """
    Разделение данных на k фолдов (блоков).
    
    Returns: список из k кортежей (train_indices, test_indices)
    """
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    fold_size = len(indices) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(indices)
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        folds.append((train_indices, test_indices))

    return folds


def k_fold_cross_validation(data, k, model_type="classification", seed=42):
    """
    K-Fold Cross-Validation.
    
    Args:
        data: список кортежей (features, label)
        k: количество фолдов
        model_type: "classification" или "regression"
        seed: зерно генератора
    
    Returns:
        Средняя метрика по всем фолдам
    """
    folds = k_fold_split(data, k, seed)
    scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = [data[i][0] for i in train_idx]
        y_train = [data[i][1] for i in train_idx]
        X_test = [data[i][0] for i in test_idx]
        y_test = [data[i][1] for i in test_idx]

        if model_type == "classification":
            weights, bias = simple_logistic_regression(X_train, y_train, lr=0.1, epochs=200)
            y_scores = predict_logistic(X_test, weights, bias)
            y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
            score = accuracy(y_test, y_pred)
        else:
            slope, intercept = simple_linear_regression(X_train, y_train)
            y_pred = predict_linear(X_test, slope, intercept)
            score = r_squared(y_test, y_pred)

        scores.append(score)

    return scores


# ============================================================
# Демонстрации
# ============================================================

def demo1_train_test_split():
    """Демо 1: Train/Test split с разными пропорциями."""
    print("=" * 65)
    print("ДЕМО 1: Train/Test Split")
    print("=" * 65)

    data = generate_data(200, seed=42)
    ratios = [("70/30", 0.3), ("50/50", 0.5), ("80/20", 0.2)]

    for name, test_ratio in ratios:
        X_train, y_train, X_test, y_test = train_test_split(data, test_ratio, seed=42)
        train_pos = sum(y_train)
        train_neg = len(y_train) - train_pos
        test_pos = sum(y_test)
        test_neg = len(y_test) - test_pos

        print(f"\n  Пропорция {name}:")
        print(f"    Всего: {len(data)}")
        print(f"    Train: {len(X_train)} (класс 0: {train_neg}, класс 1: {train_pos})")
        print(f"    Test:  {len(X_test)} (класс 0: {test_neg}, класс 1: {test_pos})")

        # Быстрая классификация для демонстрации
        weights, bias = simple_logistic_regression(X_train, y_train, lr=0.1, epochs=200)
        y_scores = predict_logistic(X_test, weights, bias)
        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
        acc = accuracy(y_test, y_pred)
        print(f"    Accuracy: {acc:.4f}")

    print()


def demo2_kfold_cv():
    """Демо 2: K-Fold Cross-Validation с разными k."""
    print("=" * 65)
    print("ДЕМО 2: K-Fold Cross-Validation")
    print("=" * 65)

    data = generate_data(200, seed=42)
    k_values = [3, 5, 10]

    for k in k_values:
        scores = k_fold_cross_validation(data, k, model_type="classification", seed=42)
        mean_score = sum(scores) / len(scores)
        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

        print(f"\n  K = {k} Fold Cross-Validation:")
        for i, s in enumerate(scores):
            print(f"    Fold {i + 1}: Accuracy = {s:.4f}")
        print(f"    Среднее: {mean_score:.4f} (+/- {std_score:.4f})")

    print()


def demo3_all_metrics():
    """Демо 3: Все метрики на одном наборе данных."""
    print("=" * 65)
    print("ДЕМО 3: Все метрики на одном наборе данных")
    print("=" * 65)

    # --- Классификация ---
    print("\n  --- Классификация ---")
    data = generate_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(data, 0.3, seed=42)

    weights, bias = simple_logistic_regression(X_train, y_train, lr=0.1, epochs=200)
    y_scores = predict_logistic(X_test, weights, bias)
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    print(f"    Samples: test = {len(y_test)}")
    print(f"    Accuracy:   {accuracy(y_test, y_pred):.4f}")
    print(f"    Precision:  {precision(y_test, y_pred, positive=1):.4f}")
    print(f"    Recall:     {recall(y_test, y_pred, positive=1):.4f}")
    print(f"    F1-score:   {f1_score(y_test, y_pred, positive=1):.4f}")

    # --- Регрессия ---
    print("\n  --- Регрессия ---")
    reg_data = generate_regression_data(100, seed=42)
    X_train_r, y_train_r, X_test_r, y_test_r = train_test_split(reg_data, 0.3, seed=42)

    slope, intercept = simple_linear_regression(X_train_r, y_train_r)
    y_pred_r = predict_linear(X_test_r, slope, intercept)

    print(f"    Истинная модель: y = 2.5x + 3.0")
    print(f"    Обученная модель: y = {slope:.4f}x + {intercept:.4f}")
    print(f"    MSE:        {mse(y_test_r, y_pred_r):.4f}")
    print(f"    R²:         {r_squared(y_test_r, y_pred_r):.4f}")

    print()


def demo4_confusion_matrix():
    """Демо 4: Confusion Matrix и её интерпретация."""
    print("=" * 65)
    print("ДЕМО 4: Confusion Matrix и интерпретация")
    print("=" * 65)

    data = generate_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(data, 0.3, seed=42)

    weights, bias = simple_logistic_regression(X_train, y_train, lr=0.1, epochs=200)
    y_scores = predict_logistic(X_test, weights, bias)
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    tp, fp, fn, tn = confusion_matrix(y_test, y_pred)

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                   Pos     Neg")
    print(f"    Actual  Pos    {tp:>4}    {fn:>4}")
    print(f"            Neg    {fp:>4}    {tn:>4}")

    print(f"\n  Интерпретация:")
    print(f"    TP (True Positive):  {tp} — модель верно предсказала класс 1")
    print(f"    FP (False Positive): {fp} — ложно предсказала класс 1 (ошибка I рода)")
    print(f"    FN (False Negative): {fn} — ложно предсказала класс 0 (ошибка II рода)")
    print(f"    TN (True Negative):  {tn} — модель верно предсказала класс 0")

    print(f"\n  Метрики из Confusion Matrix:")
    print(f"    Accuracy:  {(tp + tn) / (tp + fp + fn + tn):.4f}")
    if (tp + fp) > 0:
        print(f"    Precision: {tp / (tp + fp):.4f}")
    if (tp + fn) > 0:
        print(f"    Recall:    {tp / (tp + fn):.4f}")
    p_val = precision(y_test, y_pred)
    r_val = recall(y_test, y_pred)
    if (p_val + r_val) > 0:
        print(f"    F1:        {2 * p_val * r_val / (p_val + r_val):.4f}")

    # --- ROC и AUC ---
    print(f"\n  --- ROC Curve и AUC ---")
    fpr_list, tpr_list, thresholds = roc_curve(y_test, y_scores)

    # Визуализация ROC-кривой текстом
    print(f"\n    ROC-кривая (точки):")
    step = max(1, len(fpr_list) // 8)
    print(f"    {'FPR':>8} {'TPR':>8}")
    print(f"    {'-'*18}")
    for i in range(0, len(fpr_list), step):
        print(f"    {fpr_list[i]:>8.4f} {tpr_list[i]:>8.4f}")
    # Последняя точка
    if (len(fpr_list) - 1) % step != 0:
        print(f"    {fpr_list[-1]:>8.4f} {tpr_list[-1]:>8.4f}")

    # Простая текстовая визуализация
    print(f"\n    ASCII ROC-кривая:")
    grid_size = 20
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

    # Заполняем кривую
    for f, t in zip(fpr_list, tpr_list):
        row = int((1 - t) * (grid_size - 1))
        col = int(f * (grid_size - 1))
        row = max(0, min(grid_size - 1, row))
        col = max(0, min(grid_size - 1, col))
        grid[row][col] = '*'

    # Диагональ (случайный классификатор)
    for i in range(grid_size):
        grid[i][i] = '-'

    for row in grid:
        print(f"    {''.join(row)}")

    print(f"    * = ROC-кривая, - = случайный классификатор")
    print(f"    (ось Y = TPR, ось X = FPR, 0 сверху-слева)")

    area = auc(fpr_list, tpr_list)
    print(f"\n    AUC = {area:.4f}")
    if area >= 0.9:
        print(f"    Интерпретация: отлично (> 0.9)")
    elif area >= 0.7:
        print(f"    Интерпретация: хорошо (0.7 - 0.9)")
    elif area >= 0.5:
        print(f"    Интерпретация: удовлетворительно (0.5 - 0.7)")
    else:
        print(f"    Интерпретация: плохо (< 0.5)")

    print()


# ============================================================
# Основная программа
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  ОСНОВЫ ОЦЕНКИ МОДЕЛЕЙ (Model Evaluation)")
    print("  Все метрики реализованы с нуля, без sklearn")
    print("=" * 65)
    print()

    demo1_train_test_split()
    demo2_kfold_cv()
    demo3_all_metrics()
    demo4_confusion_matrix()

    print("=" * 65)
    print("  ИТОГИ")
    print("=" * 65)
    print("""
  Ключевые концепции:

  1. Train/Test Split — разделяем данные, чтобы проверять модель
     на данных, которые она не видела при обучении.

  2. K-Fold Cross-Validation — делим данные на k частей, по очереди
     каждую используем как тест. Усредняем результат для надёжности.

  3. Метрики классификации:
     - Accuracy  — доля правильных ответов
     - Precision — из предсказанных "1", сколько действительно "1"
     - Recall    — из реальных "1", сколько модель нашла
     - F1        — баланс Precision и Recall

  4. Метрики регрессии:
     - MSE — средняя квадратичная ошибка
     - R²  — доля объяснённой дисперсии (1 = идеально)

  5. Confusion Matrix — показывает TP, FP, FN, TN — что именно
     модель путает.

  6. ROC/AUC — показывает качество классификатора при разных порогах.
     AUC = 1.0 — идеально, AUC = 0.5 — как случайный угадывание.
""")
