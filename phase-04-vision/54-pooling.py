"""
Пулинг (Pooling) — свёрточные нейросети
========================================
Pooling (подвыборка) — операция уменьшения пространственных размерностей
feature map. Ключевые типы:
  • Max Pooling  — максимум в окне
  • Average Pooling — среднее в окне
  • Global Average Pooling — среднее по всему пространственному размеру

Пулинг делает представления инвариантными к малым сдвигам и уменьшает
число параметров и вычислений в следующих слоях.
"""

import random

random.seed(42)

# ============================================================
#  Вспомогательные функции
# ============================================================

def create_matrix(rows, cols, low=0.0, high=10.0):
    """Создаёт матрицу (rows x cols) со случайными числами."""
    return [[round(random.uniform(low, high), 2) for _ in range(cols)] for _ in range(rows)]


def create_3d(depth, rows, cols, low=0.0, high=10.0):
    """Создаёт 3D-тензор (depth x rows x cols)."""
    return [create_matrix(rows, cols, low, high) for _ in range(depth)]


def print_matrix(mat, title="", fmt=".2f"):
    """Красиво печатает матрицу."""
    if title:
        print(f"  {title}")
    for row in mat:
        print("  [" + ", ".join(f"{v:{fmt}}" for v in row) + "]")
    print()


def print_tensor(tensor, title="", fmt=".2f"):
    """Красиво печатает 3D-тензор (список матриц)."""
    if title:
        print(f"  {title}")
    for i, layer in enumerate(tensor):
        print(f"  --- слой {i} ---")
        for row in layer:
            print("  [" + ", ".join(f"{v:{fmt}}" for v in row) + "]")
    print()


# ============================================================
#  1. Max Pooling (2D)
# ============================================================

def max_pool_2d(matrix, pool_size=2, stride=None):
    """
    Max pooling по 2D-матрице.

    Параметры:
        matrix    — входная матрица (список списков)
        pool_size — размер окна (квадратное pool_size x pool_size)
        stride    — шаг окна (по умолчанию = pool_size)

    Возвращает:
        Выходную матрицу после max pooling.
    """
    if stride is None:
        stride = pool_size

    rows = len(matrix)
    cols = len(matrix[0])
    out_rows = (rows - pool_size) // stride + 1
    out_cols = (cols - pool_size) // stride + 1

    result = []
    for i in range(out_rows):
        row_out = []
        for j in range(out_cols):
            # Извлекаем окно pool_size x pool_size
            window = []
            for di in range(pool_size):
                for dj in range(pool_size):
                    window.append(matrix[i * stride + di][j * stride + dj])
            row_out.append(max(window))
        result.append(row_out)

    return result


# ============================================================
#  2. Average Pooling (2D)
# ============================================================

def avg_pool_2d(matrix, pool_size=2, stride=None):
    """
    Average pooling по 2D-матрице.

    Параметры:
        matrix    — входная матрица
        pool_size — размер окна
        stride    — шаг окна

    Возвращает:
        Выходную матрицу после average pooling.
    """
    if stride is None:
        stride = pool_size

    rows = len(matrix)
    cols = len(matrix[0])
    out_rows = (rows - pool_size) // stride + 1
    out_cols = (cols - pool_size) // stride + 1

    result = []
    for i in range(out_rows):
        row_out = []
        for j in range(out_cols):
            window = []
            for di in range(pool_size):
                for dj in range(pool_size):
                    window.append(matrix[i * stride + di][j * stride + dj])
            row_out.append(round(sum(window) / len(window), 4))
        result.append(row_out)

    return result


# ============================================================
#  3. Global Average Pooling (3D тензор → вектор)
# ============================================================

def global_avg_pool_3d(tensor):
    """
    Global average pooling по 3D-тензору (channels x height x width).

    Для каждого канала вычисляется среднее значение
    по всему пространственному размеру (height × width).

    Параметры:
        tensor — 3D-тензор: список из depth матриц (height x width)

    Возвращает:
        Список средних значений по каждому каналу.
    """
    result = []
    for channel in tensor:
        total = 0.0
        count = 0
        for row in channel:
            for val in row:
                total += val
                count += 1
        result.append(round(total / count, 4))
    return result


def global_avg_pool_2d(matrix):
    """Global average pooling для одной 2D-матрицы."""
    total = 0.0
    count = 0
    for row in matrix:
        for val in row:
            total += val
            count += 1
    return round(total / count, 4)


# ============================================================
#  4. Max Pooling (3D тензор) — по каждому каналу
# ============================================================

def max_pool_3d(tensor, pool_size=2, stride=None):
    """Max pooling для 3D-тензора (применяется к каждому каналу отдельно)."""
    return [max_pool_2d(channel, pool_size, stride) for channel in tensor]


def avg_pool_3d(tensor, pool_size=2, stride=None):
    """Average pooling для 3D-тензора (по каждому каналу)."""
    return [avg_pool_2d(channel, pool_size, stride) for channel in tensor]


# ============================================================
#  Демонстрация
# ============================================================

def demo1_max_pooling():
    """Демо 1: Max pooling 2x2 на матрице 4x4."""
    print("=" * 60)
    print("  ДЕМО 1: Max Pooling 2x2")
    print("=" * 60)

    matrix = create_matrix(4, 4, low=1.0, high=20.0)
    print_matrix(matrix, "Входная матрица 4x4:")

    result = max_pool_2d(matrix, pool_size=2)
    print_matrix(result, "После Max Pooling 2x2 (размер 2x2):")

    # Покажем окно и результат
    print("  Как работает:")
    print(f"  Окно [0:2, 0:2] → max = {max([matrix[i][j] for i in range(2) for j in range(2)]):.2f}")
    print(f"  Окно [0:2, 2:4] → max = {max([matrix[i][j] for i in range(2) for j in range(2, 4)]):.2f}")
    print(f"  Окно [2:4, 0:2] → max = {max([matrix[i][j] for i in range(2, 4) for j in range(2)]):.2f}")
    print(f"  Окно [2:4, 2:4] → max = {max([matrix[i][j] for i in range(2, 4) for j in range(2, 4)]):.2f}")
    print()


def demo2_avg_pooling():
    """Демо 2: Average pooling 2x2 на матрице 4x4."""
    print("=" * 60)
    print("  ДЕМО 2: Average Pooling 2x2")
    print("=" * 60)

    matrix = create_matrix(4, 4, low=1.0, high=20.0)
    print_matrix(matrix, "Входная матрица 4x4:")

    result = avg_pool_2d(matrix, pool_size=2)
    print_matrix(result, "После Average Pooling 2x2 (размер 2x2):")

    # Покажем расчёт одного окна
    window_vals = [matrix[i][j] for i in range(2) for j in range(2)]
    print("  Как работает:")
    print(f"  Окно [0:2, 0:2] → среднее = {sum(window_vals)/len(window_vals):.4f}")
    print(f"  (получено из {window_vals})")
    print()


def demo3_global_avg_pooling():
    """Демо 3: Global average pooling на 3D-тензоре."""
    print("=" * 60)
    print("  ДЕМО 3: Global Average Pooling")
    print("=" * 60)

    tensor = create_3d(3, 4, 4, low=0.0, high=20.0)
    print_tensor(tensor, "Входной 3D-тензор (3 канала, 4x4 каждый):")

    result = global_avg_pool_3d(tensor)
    print("  Global Average Pooling:")
    for i, (ch, val) in enumerate(zip(tensor, result)):
        count = len(ch) * len(ch[0])
        print(f"    Канал {i}: среднее всех {count} элементов = {val}")
    print()
    print(f"  Вход:  тензор 3 x 4 x 4 = {3*4*4} значений")
    print(f"  Выход: вектор длины {len(result)} (по одному числу на канал)")
    print()


def demo4_comparison():
    """Демо 4: Сравнение Max vs Average Pooling."""
    print("=" * 60)
    print("  ДЕМО 4: Сравнение Max vs Average Pooling")
    print("=" * 60)

    matrix = create_matrix(6, 6, low=1.0, high=20.0)
    print_matrix(matrix, "Входная матрица 6x6:")

    max_result = max_pool_2d(matrix, pool_size=2)
    avg_result = avg_pool_2d(matrix, pool_size=2)

    print_matrix(max_result, "Max Pooling 2x2 → 3x3:")
    print_matrix(avg_result, "Average Pooling 2x2 → 3x3:")

    # Разница
    diff = []
    for i in range(len(max_result)):
        row = []
        for j in range(len(max_result[0])):
            row.append(round(max_result[i][j] - avg_result[i][j], 4))
        diff.append(row)
    print_matrix(diff, "Разница (max - avg):")
    print("  Max pooling сохраняет яркие FEATURES (максимумы).")
    print("  Average pooling сохраняет ОБЩУЮ информацию (сглаживает).")
    print()

    # Демонстрация с stride
    print("-" * 40)
    print("  Влияние stride:")
    print("-" * 40)

    max_s1 = max_pool_2d(matrix, pool_size=2, stride=1)
    max_s2 = max_pool_2d(matrix, pool_size=2, stride=2)

    print(f"  stride=1 → выход {len(max_s1)}x{len(max_s1[0])}")
    print(f"  stride=2 → выход {len(max_s2)}x{len(max_s2[0])}")
    print()


def demo_dimensionality():
    """Демонстрация влияния пулинга на размерность."""
    print("=" * 60)
    print("  ВЛИЯНИЕ ПУЛИНГА НА РАЗМЕРНОСТЬ")
    print("=" * 60)

    H, W, C = 32, 32, 3
    print(f"  Вход: {C} x {H} x {W} (каналы x высота x ширина)")
    print(f"  Всего элементов: {C * H * W}")
    print()

    for ps in [2, 4]:
        out_H = H // ps
        out_W = W // ps
        print(f"  Pool size {ps}x{ps}:  → {C} x {out_H} x {out_W}  "
              f"= {C * out_H * out_W} элементов  "
              f"(сжатие {(H * W) / (out_H * out_W):.0f}x)")
    print()

    print(f"  Global Average Pooling: → вектор длины {C}")
    print(f"    Сжатие: {(H * W) / 1:.0f}x по пространству")
    print()
    print("  Формула выходного размера:")
    print("    H_out = (H - pool_size) / stride + 1")
    print("    W_out = (W - pool_size) / stride + 1")
    print()


# ============================================================
#  Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          ПУЛИНГ (POOLING) СЛОИ — РЕАЛИЗАЦИЯ            ║")
    print("║          Random seed: 42                                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo1_max_pooling()
    demo2_avg_pooling()
    demo3_global_avg_pooling()
    demo4_comparison()
    demo_dimensionality()

    print("=" * 60)
    print("  ИТОГИ")
    print("=" * 60)
    print("  1. Max pooling: извлекает сильнейшие активации,")
    print("     хорош для обнаружения features.")
    print("  2. Average pooling: сглаживает, сохраняет общую")
    print("     статистику, менее чувствителен к шуму.")
    print("  3. Global average pooling: заменяет полносвязный слой,")
    print("     сильно уменьшает число параметров.")
    print("  4. Pooling делает модель инвариантной к малым сдвигам")
    print("     и уменьшает вычислительную сложность.")
    print("=" * 60)
