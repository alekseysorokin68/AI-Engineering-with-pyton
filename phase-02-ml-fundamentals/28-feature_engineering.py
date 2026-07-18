"""
28 - Feature Engineering: основы с нуля на Python
================================================
Набор методов преобразования и отбора признаков для подготовки данных к ML.
"""

import random
import math
from typing import List

random.seed(42)

# ──────────────────────────── Утилиты ────────────────────────────

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def std(xs: List[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def min_max_range(xs: List[float]):
    return min(xs), max(xs), max(xs) - min(xs)

def correlation(xs: List[float], ys: List[float]) -> float:
    """Коэффициент корреляции Пирсона."""
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x * den_y == 0:
        return 0.0
    return num / (den_x * den_y)

# ═══════════════════════════════════════════════════════════════════
#  1. СТАНДАРТИЗАЦИЯ (z-score нормализация)
# ═══════════════════════════════════════════════════════════════════

def zscore_normalize(xs: List[float]) -> List[float]:
    """
    z = (x - mean) / std
    Результат: среднее ≈ 0, стандартное отклонение ≈ 1.
    """
    m, s = mean(xs), std(xs)
    if s == 0:
        return [0.0] * len(xs)
    return [(x - m) / s for x in xs]

# ═══════════════════════════════════════════════════════════════════
#  2. MIN-MAX НОРМАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════

def minmax_normalize(xs: List[float], a: float = 0.0, b: float = 1.0) -> List[float]:
    """
    x' = a + (x - x_min) / (x_max - x_min) * (b - a)
    Результат: значения лежат в [a, b].
    """
    lo, hi, rng = min_max_range(xs)
    if rng == 0:
        return [a] * len(xs)
    return [a + (x - lo) / rng * (b - a) for x in xs]

# ═══════════════════════════════════════════════════════════════════
#  3. ONE-HOT ENCODING
# ═══════════════════════════════════════════════════════════════════

def one_hot_encode(categories: List[str]) -> List[List[int]]:
    """
    Преобразует список категорий в бинарные вектора.
    """
    unique = sorted(set(categories))
    cat_to_idx = {c: i for i, c in enumerate(unique)}
    encoded = []
    for cat in categories:
        vec = [0] * len(unique)
        vec[cat_to_idx[cat]] = 1
        encoded.append(vec)
    return encoded, unique

# ═══════════════════════════════════════════════════════════════════
#  4. ПОЛИНОМИАЛЬНЫЕ ПРИЗНАКИ
# ═══════════════════════════════════════════════════════════════════

def poly_features(xs: List[float], ys: List[float], degree: int = 2) -> List[List[float]]:
    """
    Создаёт полиномиальные признаки:
    [x, y] → [x, y, x^2, x*y, y^2, ..., x^degree, ..., y^degree]
    """
    features = []
    for i in range(len(xs)):
        row = [xs[i], ys[i]]
        for deg in range(2, degree + 1):
            for j in range(deg + 1):
                row.append(xs[i] ** (deg - j) * ys[i] ** j)
        features.append(row)
    return features

# ═══════════════════════════════════════════════════════════════════
#  5. ВЫБОР ПРИЗНАКОВ ПО КОРРЕЛЯЦИИ
# ═══════════════════════════════════════════════════════════════════

def select_by_correlation(
    features: List[List[float]],
    target: List[float],
    threshold: float = 0.5,
) -> List[int]:
    """
    Возвращает индексы признаков, |корреляция| >= threshold с целевой переменной.
    """
    selected = []
    for col in range(len(features[0])):
        col_vals = [row[col] for row in features]
        corr = abs(correlation(col_vals, target))
        if corr >= threshold:
            selected.append(col)
    return selected


# ═══════════════════════════════════════════════════════════════════
#  ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════════════════════════

def demo_1_standartization_vs_minmax():
    print("=" * 65)
    print("ДЕМО 1: Стандартизация vs Min-Max нормализация")
    print("=" * 65)

    raw = [10, 20, 30, 40, 50, 100, 200]
    print(f"\n  Исходные данные:  {raw}")
    print(f"  Среднее = {mean(raw):.2f},  Стд.откл. = {std(raw):.2f}")
    print(f"  Min = {min(raw)},  Max = {max(raw)}")

    zs = zscore_normalize(raw)
    print(f"\n  --- Z-Score (стандартизация) ---")
    print(f"  Результат: {[round(v, 4) for v in zs]}")
    print(f"  Среднее = {mean(zs):.4f},  Стд.откл. = {std(zs):.4f}")
    print(f"  Минимум = {min(zs):.4f},  Максимум = {max(zs):.4f}")
    print(f"  → Среднее ≈ 0, стандартное отклонение ≈ 1")
    print(f"  → Структура распределения сохраняется")

    mm = minmax_normalize(raw)
    print(f"\n  --- Min-Max (нормализация в [0, 1]) ---")
    print(f"  Результат: {[round(v, 4) for v in mm]}")
    print(f"  Среднее = {mean(mm):.4f},  Стд.откл. = {std(mm):.4f}")
    print(f"  Минимум = {min(mm):.4f},  Максимум = {max(mm):.4f}")
    print(f"  → Все значения строго в диапазоне [0, 1]")
    print(f"  → Минимум всегда = 0, Максимум всегда = 1")

    mm_custom = minmax_normalize(raw, a=-1, b=1)
    print(f"\n  --- Min-Max в диапазоне [-1, 1] ---")
    print(f"  Результат: {[round(v, 4) for v in mm_custom]}")
    print(f"  Минимум = {min(mm_custom):.4f},  Максимум = {max(mm_custom):.4f}")

    print("\n  ВЫВОД:")
    print("  • Z-Score: хорошо для алгоритмов, предполагающих нормальность")
    print("    (линейная регрессия, SVM, PCA). Не ограничен диапазоном.")
    print("  • Min-Max: хорошо для нейросетей, kNN, алгоритмов на расстояниях.")
    print("    Ограничен фиксированным диапазоном. Чувствителен к выбросам.")


def demo_2_one_hot_encoding():
    print("\n" + "=" * 65)
    print("ДЕМО 2: One-Hot Encoding")
    print("=" * 65)

    colors = ["красный", "синий", "зелёный", "красный", "синий", "красный"]
    print(f"\n  Исходные категории: {colors}")

    encoded, unique = one_hot_encode(colors)
    print(f"\n  Уникальные категории: {unique}")
    print(f"  Число признаков после кодирования: {len(unique)}\n")

    print(f"  {'Категория':<15} {'One-Hot вектор'}")
    print(f"  {'-'*40}")
    for cat, vec in zip(colors, encoded):
        print(f"  {cat:<15} {vec}")

    print(f"\n  ВЫВОД:")
    print("  • Каждая категория → бинарный вектор длины N (число категорий)")
    print("  • Ровно один элемент = 1, остальные = 0")
    print("  • Ловушка фиктивных переменных (dummy variable trap):")
    print("    чтобы избежать мультиколлинеарности, можно удалить один столбец.")

    # Пример: one-hot для числовых бинов
    print(f"\n  --- One-Hot для числовых диапазонов ---")
    ages = [25, 17, 35, 42, 19, 30]
    bins = [("молодой", 0, 20), ("средний", 20, 35), ("пожилой", 35, 100)]
    binned = []
    for age in ages:
        for label, lo, hi in bins:
            if lo <= age < hi:
                binned.append(label)
                break
        else:
            binned.append("пожилой")
    print(f"  Возраста:  {ages}")
    print(f"  Бины:      {binned}")
    enc_bins, u_bins = one_hot_encode(binned)
    print(f"  Категории: {u_bins}")
    for age, cat, vec in zip(ages, binned, enc_bins):
        print(f"    age={age:>2} → {cat:<10} → {vec}")


def demo_3_polynomial_features():
    print("\n" + "=" * 65)
    print("ДЕМО 3: Полиномиальные признаки")
    print("=" * 65)

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 5.0, 4.0, 5.0]

    print(f"\n  Исходные признаки:")
    for x, y in zip(xs, ys):
        print(f"    x={x:.1f}, y={y:.1f}")

    deg2 = poly_features(xs, ys, degree=2)
    labels = ["x", "y", "x²", "x·y", "y²"]
    print(f"\n  --- Степень 2 (degree=2) ---")
    print(f"  Столбцы: {labels}")
    for row in deg2:
        print(f"    {[round(v, 2) for v in row]}")

    deg3 = poly_features(xs, ys, degree=3)
    labels3 = ["x", "y", "x²", "x·y", "y²", "x³", "x²·y", "x·y²", "y³"]
    print(f"\n  --- Степень 3 (degree=3) ---")
    print(f"  Столбцы: {labels3}")
    for row in deg3:
        print(f"    {[round(v, 2) for v in row]}")

    print(f"\n  Число признаков:")
    print(f"    2 исходных → степень 2: {len(deg2[0])} признаков")
    print(f"    2 исходных → степень 3: {len(deg3[0])} признаков")

    print(f"\n  ВЫВОД:")
    print("  • Полиномиальные признаки позволяют линейной модели")
    print("    аппроксимировать нелинейные зависимости")
    print("  • Формула: число признаков = C(n+d, d) - 1,")
    print("    где n — число исходных, d — степень")
    print("  • Опасность: взрывного роста числа признаков (curse of dimensionality)")


def demo_4_feature_selection_correlation():
    print("\n" + "=" * 65)
    print("ДЕМО 4: Отбор признаков по корреляции")
    print("=" * 65)

    n = 20
    random.seed(42)

    x1 = [random.gauss(0, 1) for _ in range(n)]              # сильно коррелирует
    x2 = [random.gauss(0, 1) for _ in range(n)]              # слабо коррелирует (шум)
    x3 = [0.5 * x1[i] + random.gauss(0, 0.3) for i in range(n)]  # средняя корреляция

    target = [2.0 * x1[i] + 3.0 for i in range(n)]  # целевая переменная

    print(f"\n  Число наблюдений: {n}")
    print(f"  Целевая переменная: target = 2·x1 + 3 + шум")
    print(f"\n  Признаки:")
    print(f"    x1: сильно коррелирован с target (линейная зависимость)")
    print(f"    x2: случайный шум (нулевая корреляция)")
    print(f"    x3: средняя корреляция с x1")

    features = [[x1[i], x2[i], x3[i]] for i in range(n)]
    feature_names = ["x1", "x2", "x3"]

    print(f"\n  Корреляции с target:")
    for i, name in enumerate(feature_names):
        col = [row[i] for row in features]
        corr = correlation(col, target)
        print(f"    {name}: r = {corr:+.4f}")

    print(f"\n  Матрица корреляций между признаками:")
    print(f"    {'':>6}", end="")
    for name in feature_names:
        print(f"{name:>8}", end="")
    print()
    for i, name_i in enumerate(feature_names):
        col_i = [row[i] for row in features]
        print(f"    {name_i:>6}", end="")
        for j, name_j in enumerate(feature_names):
            col_j = [row[j] for row in features]
            corr = correlation(col_i, col_j)
            print(f"{corr:>8.3f}", end="")
        print()

    thresholds = [0.3, 0.5, 0.8]
    for thr in thresholds:
        selected = select_by_correlation(features, target, threshold=thr)
        names = [feature_names[i] for i in selected]
        print(f"\n  Порог |r| >= {thr}: отобраны → {names if names else '(пусто)'}")

    print(f"\n  ВЫВОД:")
    print("  • Признаки с |r| >= 0.5 считаются значимыми")
    print("  • Признаки с |r| < 0.3 — кандидаты на удаление")
    print("  • Корреляция ≠ причинность: высокий r не гарантирует причинной связи")
    print("  • Важно: корреляция ловит только линейные зависимости!")


# ═══════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  28 - Feature Engineering: основы с нуля на Python          ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    demo_1_standartization_vs_minmax()
    demo_2_one_hot_encoding()
    demo_3_polynomial_features()
    demo_4_feature_selection_correlation()

    print("\n" + "=" * 65)
    print("ИТОГО: основные методы Feature Engineering")
    print("=" * 65)
    print("""
  1. Стандартизация (z-score):
     z = (x - mean) / std  →  среднее=0, стд=1
     Использование: SVM, линейная регрессия, PCA

  2. Min-Max нормализация:
     x' = (x - min) / (max - min)  →  [0, 1]
     Использование: нейросети, kNN, алгоритмы на расстояниях

  3. One-Hot Encoding:
     Категория → бинарный вектор
     Использование: все алгоритмы, принимающие числа

  4. Полиномиальные признаки:
     [x, y] → [x, y, x², xy, y², ...]
     Использование: линейные модели для нелинейных зависимостей

  5. Отбор признаков по корреляции:
     Удаление признаков с |r| < порога
     Использование: предобработка данных перед обучением
""")
