"""
Bias-Variance Tradeoff — демонстрация с нуля на Python.

Ключевые концепции:
- Bias (смещение): ошибка из-за слишком простой модели
- Variance (дисперсия): ошибка из-за слишком сложной модели
- Bias-Variance Decomposition: MSE = Bias² + Variance + Irreducible Error
- Learning Curves: как ошибка зависит от размера обучающей выборки

Зависимости: только numpy (для удобства) + math, copy.
"""

import math
import copy
import random

random.seed(42)

# ============================================================
# 1. Вспомогательные функции
# ============================================================

def generate_data(n=30, noise_std=0.5):
    """Генерирует синтетические данные: y = sin(1.5*x) + noise."""
    data = []
    for _ in range(n):
        x = random.uniform(-3, 3)
        y = math.sin(1.5 * x) + random.gauss(0, noise_std)
        data.append((x, y))
    return data


def poly_features(x, degree):
    """Создаёт полиномиальные признаки [1, x, x^2, ..., x^degree]."""
    return [x ** i for i in range(degree + 1)]


def poly_predict(x, coeffs):
    """Вычисляет предсказание полинома с заданными коэффициентами."""
    features = poly_features(x, len(coeffs) - 1)
    return sum(c * f for c, f in zip(coeffs, features))


def mean_squared_error(data, coeffs):
    """Вычисляет MSE на наборе данных."""
    if not data:
        return 0.0
    total = 0.0
    for x, y in data:
        pred = poly_predict(x, coeffs)
        total += (y - pred) ** 2
    return total / len(data)


def linear_regression_fit(X, y):
    """
    Решает задачу линейной регрессии: X @ w = y
    через Normal Equation: w = (X^T X)^{-1} X^T y
    
    X: список списков (матрица признаков)
    y: список значений
    """
    n = len(X)
    m = len(X[0])
    
    # X^T X
    XtX = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += X[k][i] * X[k][j]
            XtX[i][j] = s
    
    # X^T y
    Xty = [0.0] * m
    for i in range(m):
        s = 0.0
        for k in range(n):
            s += X[k][i] * y[k]
        Xty[i] = s
    
    # Regularization (ridge) для численной стабильности
    lam = 1e-8
    for i in range(m):
        XtX[i][i] += lam
    
    # Решение через Gaussian elimination
    w = solve_linear_system(XtX, Xty)
    return w


def solve_linear_system(A, b):
    """Решает систему Ax = b методом Гаусса с частичным выбором."""
    n = len(A)
    # Копируем
    M = [row[:] for row in A]
    rhs = b[:]
    
    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        rhs[col], rhs[max_row] = rhs[max_row], rhs[col]
        
        if abs(M[col][col]) < 1e-14:
            continue
        
        # Eliminate
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n):
                M[row][j] -= factor * M[col][j]
            rhs[row] -= factor * rhs[col]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i] if abs(M[i][i]) > 1e-14 else 0.0
    
    return x


def fit_polynomial(data, degree):
    """Обучает полиномиальную регрессию заданной степени."""
    X = [poly_features(x, degree) for x, _ in data]
    y = [val for _, val in data]
    coeffs = linear_regression_fit(X, y)
    return coeffs


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# Демо 1: Полиномиальная регрессия (степень 1 vs 3 vs 10)
# ============================================================

def demo1_polynomial_regression():
    print_separator("Демо 1: Полиномиальная регрессия")
    
    data = generate_data(n=30, noise_std=0.3)
    degrees = [1, 3, 10]
    coeffs_dict = {}
    
    for degree in degrees:
        coeffs = fit_polynomial(data, degree)
        coeffs_dict[degree] = coeffs
        mse = mean_squared_error(data, coeffs)
        
        print(f"--- Степень полинома: {degree} ---")
        print(f"  Коэффициенты: {[round(c, 4) for c in coeffs]}")
        print(f"  MSE на train: {mse:.6f}")
        
        if degree == 1:
            print(f"  Модель: y = {coeffs[0]:.3f} + {coeffs[1]:.3f}*x")
            print(f"  -> Underfitting: слишком простая, не улавливает нелинейность")
        elif degree == 3:
            print(f"  Модель: полином 3-й степени")
            print(f"  -> Good fit: баланс сложности и обобщения")
        else:
            print(f"  Модель: полином 10-й степени ({len(coeffs)} коэффициентов)")
            print(f"  -> Overfitting: идеально проходит через точки, но плохо обобщает")
        print()
    
    # Предсказания на новых точках
    test_points = [-2.5, -1.0, 0.0, 1.0, 2.5]
    print("Предсказания на тестовых точках:")
    print(f"{'x':>6} | {'Степень 1':>12} | {'Степень 3':>12} | {'Степень 10':>12} | {'sin(1.5x)':>12}")
    print("-" * 65)
    for x in test_points:
        true_val = math.sin(1.5 * x)
        pred1 = poly_predict(x, coeffs_dict[1])
        pred3 = poly_predict(x, coeffs_dict[3])
        pred10 = poly_predict(x, coeffs_dict[10])
        print(f"{x:6.1f} | {pred1:12.4f} | {pred3:12.4f} | {pred10:12.4f} | {true_val:12.4f}")


# ============================================================
# Демо 2: Bias-Variance Decomposition
# ============================================================

def bias_variance_decomposition(n_datasets=200, n_train=30, degree=3, noise_std=0.3):
    """
    Разложение Bias-Variance через усреднение по нескольким датасетам.
    
    Для каждого нового датасета:
    1. Генерируем данные из истинной функции + шум
    2. Обучаем модель
    3. Записываем предсказания
    
    Bias² = E[(f_true - E[f_hat])²]
    Variance = E[(f_hat - E[f_hat])²]
    """
    test_x = 0.5  # Точка, в которой считаем decomposition
    true_y = math.sin(1.5 * test_x)
    
    predictions = []
    
    for _ in range(n_datasets):
        data = generate_data(n=n_train, noise_std=noise_std)
        coeffs = fit_polynomial(data, degree)
        pred = poly_predict(test_x, coeffs)
        predictions.append(pred)
    
    # Среднее предсказание
    mean_pred = sum(predictions) / len(predictions)
    
    # Bias² = (mean_pred - true_y)^2
    bias_squared = (mean_pred - true_y) ** 2
    
    # Variance = E[(pred - mean_pred)^2]
    variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
    
    # Irreducible error = noise²
    irreducible = noise_std ** 2
    
    # MSE = Bias² + Variance + Irreducible
    mse_total = sum((p - true_y) ** 2 for p in predictions) / len(predictions)
    mse_check = bias_squared + variance + irreducible
    
    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "irreducible": irreducible,
        "mse_total": mse_total,
        "mse_check": mse_check,
        "mean_pred": mean_pred,
        "true_y": true_y,
        "predictions": predictions,
    }


def demo2_bias_variance():
    print_separator("Демо 2: Bias-Variance Decomposition")
    
    degrees = [1, 3, 10]
    n_datasets = 300
    n_train = 30
    
    print(f"Параметры: {n_datasets} датасетов, train_size={n_train}, noise=0.3")
    print(f"Точка для анализа: x=0.5, true y=sin(1.5*0.5)={math.sin(0.75):.4f}\n")
    
    results = {}
    
    for degree in degrees:
        res = bias_variance_decomposition(
            n_datasets=n_datasets,
            n_train=n_train,
            degree=degree,
            noise_std=0.3
        )
        results[degree] = res
        
        print(f"--- Степень {degree} ---")
        print(f"  Bias² (смещение):     {res['bias_squared']:.6f}")
        print(f"  Variance (дисперсия): {res['variance']:.6f}")
        print(f"  Irreducible (шум²):   {res['irreducible']:.6f}")
        print(f"  MSE (分解):            {res['mse_check']:.6f}")
        print(f"  MSE (измеренный):     {res['mse_total']:.6f}")
        print(f"  Среднее предсказание:  {res['mean_pred']:.4f} (true: {res['true_y']:.4f})")
        
        if degree == 1:
            print(f"  -> Высокий Bias, низкая Variance = Underfitting")
        elif degree == 3:
            print(f"  -> Сбалансированный Bias и Variance = Good Fit")
        else:
            print(f"  -> Низкий Bias, высокая Variance = Overfitting")
        print()
    
    # Визуальная сводка (текстовая гистограмма)
    print("Сводка Bias² vs Variance:")
    print(f"{'Степень':>8} | {'Bias²':>10} | {'Variance':>10} | {'Сумма':>10} | {'Диагноз':>15}")
    print("-" * 65)
    for d in degrees:
        r = results[d]
        diag = "Underfit" if r['bias_squared'] > r['variance'] else \
               "Overfit" if r['variance'] > r['bias_squared'] * 2 else "Good Fit"
        print(f"{d:>8} | {r['bias_squared']:10.6f} | {r['variance']:10.6f} | "
              f"{r['bias_squared']+r['variance']:10.6f} | {diag:>15}")


# ============================================================
# Демо 3: Learning Curves
# ============================================================

def compute_learning_curve(degree, train_sizes, n_test=500, noise_std=0.3, n_trials=50):
    """
    Вычисляет learning curves для заданной степени полинома.
    
    Для каждого размера обучающей выборки:
    1. Генерируем n_trials разных train наборов
    2. Для каждого обучаем модель
    3. Считаем ошибку на большом тестовом наборе (без шума)
    4. Усредняем
    """
    # Фиксированный тестовый набор (без шума для чистоты)
    random.seed(123)
    test_data = [(x, math.sin(1.5 * x)) for x in [i * 6 / n_test - 3 for i in range(n_test)]]
    random.seed(42)
    
    train_errors = []
    test_errors = []
    
    for size in train_sizes:
        t_errors = []
        te_errors = []
        
        for _ in range(n_trials):
            data = generate_data(n=size, noise_std=noise_std)
            coeffs = fit_polynomial(data, degree)
            
            train_mse = mean_squared_error(data, coeffs)
            test_mse = mean_squared_error(test_data, coeffs)
            
            t_errors.append(train_mse)
            te_errors.append(test_mse)
        
        train_errors.append(sum(t_errors) / len(t_errors))
        test_errors.append(sum(te_errors) / len(te_errors))
    
    return train_errors, test_errors


def demo3_learning_curves():
    print_separator("Демо 3: Learning Curves")
    
    train_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    degrees = [1, 3, 10]
    
    all_curves = {}
    
    for degree in degrees:
        train_err, test_err = compute_learning_curve(
            degree=degree,
            train_sizes=train_sizes,
            n_test=500,
            noise_std=0.3,
            n_trials=40
        )
        all_curves[degree] = (train_err, test_err)
    
    # Вывод таблицы
    for degree in degrees:
        train_err, test_err = all_curves[degree]
        print(f"--- Степень {degree} ---")
        print(f"{'Размер':>8} | {'Train MSE':>12} | {'Test MSE':>12} | {'Разница':>10}")
        print("-" * 50)
        for i, size in enumerate(train_sizes):
            gap = test_err[i] - train_err[i]
            print(f"{size:>8} | {train_err[i]:12.6f} | {test_err[i]:12.6f} | {gap:10.6f}")
        print()
    
    # Текстовая "визуализация" learning curves
    print("Текстовая визуализация Learning Curves (Train vs Test):")
    print()
    for degree in degrees:
        train_err, test_err = all_curves[degree]
        print(f"Степень {degree}:")
        print(f"  Train MSE: ", end="")
        max_te = max(test_err)
        for te in test_err:
            bar_len = int(te / max_te * 30)
            print(f"{'█' * bar_len}", end=" ")
        print()
        print(f"  Test MSE:  ", end="")
        for te in test_err:
            bar_len = int(te / max_te * 30)
            print(f"{'▓' * bar_len}", end=" ")
        print()
        print(f"  █ = Train, ▓ = Test")
        print()


# ============================================================
# Демо 4: Сравнение Underfitting / Good Fit / Overfitting
# ============================================================

def demo4_comparison():
    print_separator("Демо 4: Underfitting vs Good Fit vs Overfitting")
    
    # Генерируем данные
    data = generate_data(n=40, noise_std=0.3)
    random.seed(42)  # Сброс для воспроизводимости далее
    
    # Тестовый набор
    test_data = generate_data(n=200, noise_std=0.3)
    
    # Обучаем модели разных степеней
    models = {
        "Underfit (degree=1)": (1, fit_polynomial(data, 1)),
        "Good Fit (degree=3)": (3, fit_polynomial(data, 3)),
        "Overfit (degree=15)": (15, fit_polynomial(data, 15)),
    }
    
    # Количественные метрики
    print("1. Количественные метрики:")
    print(f"{'Модель':>25} | {'Train MSE':>12} | {'Test MSE':>12} | {'Разница':>10}")
    print("-" * 65)
    
    for name, (degree, coeffs) in models.items():
        train_mse = mean_squared_error(data, coeffs)
        test_mse = mean_squared_error(test_data, coeffs)
        gap = test_mse - train_mse
        print(f"{name:>25} | {train_mse:12.6f} | {test_mse:12.6f} | {gap:10.6f}")
    
    # Характеристики
    print("\n2. Характеристики каждого случая:\n")
    
    characteristics = [
        ("Underfit (degree=1)", [
            "Модель слишком простая",
            "Не способна уловить нелинейную зависимость",
            "Высокий Bias (смещение)",
            "Низкая Variance (дисперсия)",
            "Плохо как на train, так и на test",
            "Решение: увеличить сложность модели"
        ]),
        ("Good Fit (degree=3)", [
            "Модель оптимально сложная",
            "Хорошо улавливает основной паттерн",
            "Баланс Bias и Variance",
            "Небольшой разрыв train/test",
            "Хорошо обобщает на новые данные",
            "Идеальное состояние для ML модели"
        ]),
        ("Overfit (degree=15)", [
            "Модель слишком сложная",
            "Запоминает шум в данных",
            "Низкий Bias (смещение)",
            "Высокая Variance (дисперсия)",
            "Отлично на train, плохо на test",
            "Решение: регуляризация, меньше признаков, больше данных"
        ]),
    ]
    
    for name, chars in characteristics:
        print(f"  {name}:")
        for char in chars:
            print(f"    - {char}")
        print()
    
    # Предсказания на тестовых точках
    print("3. Предсказания на тестовых точках:")
    test_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print(f"{'x':>6} | {'True':>8} | {'Underfit':>10} | {'Good Fit':>10} | {'Overfit':>10}")
    print("-" * 55)
    for x in test_x:
        true_val = math.sin(1.5 * x)
        preds = {}
        for name, (degree, coeffs) in models.items():
            preds[name] = poly_predict(x, coeffs)
        print(f"{x:6.1f} | {true_val:8.4f} | "
              f"{preds['Underfit (degree=1)']:10.4f} | "
              f"{preds['Good Fit (degree=3)']:10.4f} | "
              f"{preds['Overfit (degree=15)']:10.4f}")
    
    # Выводы
    print("\n4. Ключевые выводы:")
    print("   - Underfit: высокий Bias, низкая Variance -> модель слишком проста")
    print("   - Good Fit: баланс Bias и Variance -> модель обобщает")
    print("   - Overfit: низкий Bias, высокая Variance -> модель переобучена")
    print()
    print("   Bias-Variance Tradeoff:")
    print("   MSE = Bias² + Variance + Irreducible Error")
    print("   Нельзя одновременно минимизировать Bias и Variance —")
    print("   нужно найти баланс (оптимальную сложность модели).")


# ============================================================
# Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BIAS-VARIANCE TRADEOFF — Демонстрация")
    print("  Полиномиальная регрессия с нуля на Python")
    print("=" * 60)
    
    demo1_polynomial_regression()
    demo2_bias_variance()
    demo3_learning_curves()
    demo4_comparison()
    
    print_separator("Итог")
    print("Bias-Variance Tradeoff — центральная концепция ML:")
    print()
    print("  Underfit (high bias, low variance):")
    print("    - Модель слишком простая")
    print("    - Решение: сложнее модель, больше признаков")
    print()
    print("  Good fit (balanced):")
    print("    - Оптимальная сложность")
    print("    - Минимальная общая ошибка")
    print()
    print("  Overfit (low bias, high variance):")
    print("    - Модель слишком сложная, запоминает шум")
    print("    - Решение: регуляризация, больше данных, ансамбли")
    print()
    print("  Формула:")
    print("    E[MSE] = Bias²(f̂) + Var(f̂) + σ²")
    print("    где σ² — неустранимая ошибка (шум в данных)")
