"""
35. Основы работы с временными рядами на Python

Содержимое:
1. Генерация временного ряда (синусоида + тренд + шум)
2. Скользящее среднее (Moving Average)
3. Экспоненциальное сглаживание
4. Предсказание (наивное и линейная регрессия)
"""

import math
import random

random.seed(42)

# ============================================================
# 1. ГЕНЕРАЦИЯ ВРЕМЕННОГО РЯДА
# ============================================================

def generate_time_series(n_points=200, trend_slope=0.02, noise_level=0.5, seed=42):
    """Генерация временного ряда: синусоида + тренд + шум."""
    random.seed(seed)
    series = []
    for i in range(n_points):
        sine_val = math.sin(2 * math.pi * i / 50)
        trend_val = trend_slope * i
        noise_val = random.gauss(0, noise_level)
        series.append(sine_val + trend_val + noise_val)
    return series


# ============================================================
# 2. СКОЛЬЗЯЩЕЕ СРЕДНЕЕ (MOVING AVERAGE)
# ============================================================

def moving_average(series, window):
    """Скользящее среднее с окном заданного размера."""
    result = []
    for i in range(len(series)):
        if i < window - 1:
            result.append(None)
        else:
            window_slice = series[i - window + 1: i + 1]
            result.append(sum(window_slice) / window)
    return result


# ============================================================
# 3. ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ
# ============================================================

def exponential_smoothing(series, alpha):
    """Простое экспоненциальное сглаживание.
    
    alpha: коэффициент сглаживания (0 < alpha <= 1)
           alpha близко к 1 — быстрая реакция на изменения
           alpha близко к 0 — сильное сглаживание
    """
    result = [series[0]]
    for i in range(1, len(series)):
        smoothed = alpha * series[i] + (1 - alpha) * result[i - 1]
        result.append(smoothed)
    return result


# ============================================================
# 4. ПРЕДСКАЗАНИЕ
# ============================================================

def naive_forecast(series, n_forecast=10):
    """Наивное предсказание: последнее значение повторяется."""
    last_value = series[-1]
    return [last_value] * n_forecast


def linear_regression_forecast(series, n_forecast=10):
    """Линейная регрессия для предсказания будущих значений.
    
    Использует формулу метода наименьших квадратов:
        y = a + b * x
    где b = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - (sum(x))^2)
        a = mean(y) - b * mean(x)
    """
    n = len(series)
    x_values = list(range(n))
    y_values = series

    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)

    # Коэффициенты линейной регрессии
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    a = (sum_y - b * sum_x) / n

    # Предсказание
    forecast = []
    for i in range(1, n_forecast + 1):
        future_x = n + i - 1
        forecast.append(a + b * future_x)

    return forecast, a, b


def calculate_mse(actual, predicted):
    """Среднеквадратичная ошибка (MSE)."""
    n = len(actual)
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n


def calculate_mae(actual, predicted):
    """Средняя абсолютная ошибка (MAE)."""
    n = len(actual)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / n


# ============================================================
# ДЕМОНСТРАЦИЯ
# ============================================================

def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo1_generation():
    """Демо 1: Генерация и визуализация временного ряда."""
    print_separator("Демо 1: Генерация временного ряда")

    series = generate_time_series(n_points=100)

    print(f"\nТипичные значения ряда (первые 15 точек):")
    for i in range(15):
        print(f"  t={i:3d}: {series[i]:+.4f}")

    print(f"\nСтатистика:")
    print(f"  Количество точек: {len(series)}")
    print(f"  Среднее:          {sum(series)/len(series):+.4f}")

    min_val = min(series)
    max_val = max(series)
    print(f"  Минимум:          {min_val:+.4f}")
    print(f"  Максимум:         {max_val:+.4f}")

    variance = sum((x - sum(series)/len(series))**2 for x in series) / len(series)
    print(f"  Дисперсия:        {variance:.4f}")

    print(f"\nПростая текстовая визуализация (амплитуда 0-39 символов):")
    width = 40
    for i in range(0, len(series), 5):
        normalized = int((series[i] - min_val) / (max_val - min_val) * (width - 1))
        bar = ' ' * normalized + '*'
        print(f"  t={i:3d} |{bar:<{width}}| {series[i]:+.2f}")


def demo2_moving_average():
    """Демо 2: Скользящее среднее с разными окнами."""
    print_separator("Демо 2: Скользящее среднее (Moving Average)")

    series = generate_time_series(n_points=100)
    windows = [5, 10, 20]

    print("\nСравнение скользящих средних с разными окнами:")
    print(f"{'t':>4} | {'Исходный':>9} | {'MA-5':>9} | {'MA-10':>9} | {'MA-20':>9}")
    print("-" * 55)

    for w in windows:
        ma = moving_average(series, w)

    for i in range(0, 50, 5):
        ma5 = moving_average(series, 5)
        ma10 = moving_average(series, 10)
        ma20 = moving_average(series, 20)

        original = series[i]
        val5 = f"{ma5[i]:+.4f}" if ma5[i] is not None else "  N/A"
        val10 = f"{ma10[i]:+.4f}" if ma10[i] is not None else "  N/A"
        val20 = f"{ma20[i]:+.4f}" if ma20[i] is not None else "  N/A"

        print(f"{i:4d} | {original:+9.4f} | {val5:>9} | {val10:>9} | {val20:>9}")

    print("\nЭффект разных окон:")
    for w in windows:
        ma = moving_average(series, w)
        valid = [v for v in ma if v is not None]
        variance_before = sum((x - sum(series)/len(series))**2 for x in series) / len(series)
        mean_ma = sum(valid) / len(valid)
        variance_after = sum((x - mean_ma)**2 for x in valid) / len(valid)
        print(f"  Окно {w:2d}: дисперсия {variance_before:.4f} → {variance_after:.4f} "
              f"(снижение на {(1 - variance_after/variance_before)*100:.1f}%)")


def demo3_exponential_smoothing():
    """Демо 3: Экспоненциальное сглаживание."""
    print_separator("Демо 3: Экспоненциальное сглаживание")

    series = generate_time_series(n_points=100)
    alphas = [0.1, 0.3, 0.5, 0.9]

    print("\nСравнение сглаживания с разными alpha:")
    print(f"{'t':>4} | {'Исходный':>9} | {'α=0.1':>9} | {'α=0.3':>9} | {'α=0.5':>9} | {'α=0.9':>9}")
    print("-" * 70)

    smoothed = {}
    for alpha in alphas:
        smoothed[alpha] = exponential_smoothing(series, alpha)

    for i in range(0, 50, 5):
        row = f"{i:4d} | {series[i]:+9.4f}"
        for alpha in alphas:
            row += f" | {smoothed[alpha][i]:+9.4f}"
        print(row)

    print("\nХарактеристики сглаживания:")
    for alpha in alphas:
        s = smoothed[alpha]
        mse = calculate_mse(series, s)
        print(f"  alpha={alpha}: MSE к исходному = {mse:.4f} "
              f"(реакция на изменения: {'быстрая' if alpha > 0.5 else 'медленная'})")


def demo4_forecasting():
    """Демо 4: Предсказание следующих значений."""
    print_separator("Демо 4: Предсказание следующих значений")

    series = generate_time_series(n_points=100)

    # Наивное предсказание
    naive = naive_forecast(series, n_forecast=10)
    print("\nНаивное предсказание (последнее значение):")
    print(f"  Последнее известное значение: {series[-1]:+.4f}")
    print(f"  Все предсказанные значения:    {naive[0]:+.4f}")

    # Линейная регрессия
    lr_forecast, intercept, slope = linear_regression_forecast(series, n_forecast=10)
    print(f"\nЛинейная регрессия:")
    print(f"  Модель: y = {intercept:+.4f} + {slope:+.4f} * t")
    print(f"  Направление: {'возрастающий' if slope > 0 else 'убывающий'} тренд")

    print("\nПредсказания на 10 шагов вперёд:")
    print(f"{'Шаг':>4} | {'Наивное':>9} | {'Регрессия':>9}")
    print("-" * 30)
    for i in range(10):
        print(f"{i+1:4d} | {naive[i]:+9.4f} | {lr_forecast[i]:+9.4f}")

    # Оценка качества на последних 20 точках
    test_size = 20
    train_series = series[:-test_size]
    test_series = series[-test_size:]

    naive_test = naive_forecast(train_series, n_forecast=test_size)
    lr_test, _, _ = linear_regression_forecast(train_series, n_forecast=test_size)

    naive_mse = calculate_mse(test_series, naive_test)
    lr_mse = calculate_mse(test_series, lr_test)
    naive_mae = calculate_mae(test_series, naive_test)
    lr_mae = calculate_mae(test_series, lr_test)

    print(f"\nОценка качества (последние {test_size} точек как тест):")
    print(f"  {'Метод':<20} | {'MSE':>8} | {'MAE':>8}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'Наивное':<20} | {naive_mse:8.4f} | {naive_mae:8.4f}")
    print(f"  {'Линейная регрессия':<20} | {lr_mse:8.4f} | {lr_mae:8.4f}")

    winner = "Линейная регрессия" if lr_mse < naive_mse else "Наивное"
    print(f"\n  Лучший метод по MSE: {winner}")


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ОСНОВЫ РАБОТЫ С ВРЕМЕННЫМИ РЯДАМИ НА PYTHON")
    print("=" * 60)

    demo1_generation()
    demo2_moving_average()
    demo3_exponential_smoothing()
    demo4_forecasting()

    print(f"\n{'=' * 60}")
    print("  ИТОГИ")
    print(f"{'=' * 60}")
    print("""
  • Временной ряд — последовательность значений, упорядоченных по времени.
  • Скользящее среднее сглаживает шум; большее окно → сильнее сглаживание.
  • Экспоненциальное сглаживание: alpha управляет балансом между
    историей и новыми данными.
  • Наивное предсказание: простое, но часто удивительно конкурентоспособное.
  • Линейная регрессия улавливает тренд, но не сезонность.
  • Для сезонных рядов нужны SARIMA, Prophet или нейросети.
""")
