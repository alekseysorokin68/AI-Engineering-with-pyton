"""
Quantization: INT8 и INT4
==========================

Основы квантизации весов нейросетей — сжатие моделей за счёт
уменьшения точности чисел (float32 → int8 / int4).

Включает:
1. INT8 квантизация
2. INT4 квантизация (упрощённо)
3. Влияние на точность
4. Speed vs accuracy tradeoff

Все реализации самодостаточные (без numpy/torch/transformers/bitsandbytes).
"""

import random
import math
import time

random.seed(42)


# ============================================================
# Утилиты
# ============================================================

def rand_list(n, lo=-1.0, hi=1.0):
    """Создать список из n случайных float в [lo, hi]."""
    return [random.uniform(lo, hi) for _ in range(n)]


def rand_matrix(rows, cols, lo=-1.0, hi=1.0):
    """Создать матрицу rows×cols случайных float."""
    return [rand_list(cols, lo, hi) for _ in range(rows)]


def mat_mul(A, B):
    """Умножение матриц A (m×k) × B (k×n) → (m×n)."""
    m, k, n = len(A), len(A[0]), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def vec_mul(A, x):
    """Умножение матрицы A (m×n) на вектор x (n) → (m)."""
    m, n = len(A), len(A[0])
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(m)]


def frobenius_error(original, approx):
    """Относительная ошибка Фробениуса."""
    n, m = len(original), len(original[0])
    sq_err = 0.0
    sq_norm = 0.0
    for i in range(n):
        for j in range(m):
            sq_err += (original[i][j] - approx[i][j]) ** 2
            sq_norm += original[i][j] ** 2
    return math.sqrt(sq_err) / math.sqrt(sq_norm) if sq_norm > 0 else 0.0


def cosine_sim(a, b):
    """Косинусное сходство двух векторов."""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai ** 2 for ai in a))
    nb = math.sqrt(sum(bi ** 2 for bi in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ============================================================
# 1. INT8 квантизация
# ============================================================

def quantize_int8(weights):
    """
    Symmetric INT8 квантизация:
      scale = max(|w|) / 127
      q = round(w / scale)   → int8 [-127, 127]
    """
    max_val = max(abs(w) for w in weights)
    scale = max_val / 127.0 if max_val > 0 else 1.0
    quantized = [max(-127, min(127, round(w / scale))) for w in weights]
    return quantized, scale


def dequantize_int8(quantized, scale):
    """Обратная INT8 квантизация."""
    return [q * scale for q in quantized]


def quantize_int8_matrix(matrix):
    """Квантизация всей матрицы с общим scale."""
    all_vals = [w for row in matrix for w in row]
    max_val = max(abs(w) for w in all_vals)
    scale = max_val / 127.0 if max_val > 0 else 1.0
    quantized = [
        [max(-127, min(127, round(w / scale))) for w in row]
        for row in matrix
    ]
    return quantized, scale


def dequantize_int8_matrix(quantized, scale):
    """Обратная квантизация матрицы."""
    return [[q * scale for q in row] for row in quantized]


# ============================================================
# 2. INT4 квантизация (упрощённо)
# ============================================================

def quantize_int4(weights):
    """
    Symmetric INT4 квантизация:
      scale = max(|w|) / 7
      q = round(w / scale)   → int4 [-7, 7]
    """
    max_val = max(abs(w) for w in weights)
    scale = max_val / 7.0 if max_val > 0 else 1.0
    quantized = [max(-7, min(7, round(w / scale))) for w in weights]
    return quantized, scale


def dequantize_int4(quantized, scale):
    """Обратная INT4 квантизация."""
    return [q * scale for q in quantized]


def quantize_int4_matrix(matrix):
    """Квантизация матрицы INT4."""
    all_vals = [w for row in matrix for w in row]
    max_val = max(abs(w) for w in all_vals)
    scale = max_val / 7.0 if max_val > 0 else 1.0
    quantized = [
        [max(-7, min(7, round(w / scale))) for w in row]
        for row in matrix
    ]
    return quantized, scale


def dequantize_int4_matrix(quantized, scale):
    """Обратная квантизация INT4."""
    return [[q * scale for q in row] for row in quantized]


# ============================================================
# 3. Групповая квантизация (better precision)
# ============================================================

def quantize_int4_grouped(weights, group_size=32):
    """
    Групповая INT4 квантизация:
    Каждые group_size весов квантизуются со своим scale.
    Лучшая точность за счёт локальных диапазонов.
    """
    scales = []
    quantized = []
    for i in range(0, len(weights), group_size):
        group = weights[i:i + group_size]
        max_val = max(abs(w) for w in group)
        s = max_val / 7.0 if max_val > 0 else 1.0
        scales.append(s)
        quantized.extend([max(-7, min(7, round(w / s))) for w in group])
    return quantized, scales


def dequantize_int4_grouped(quantized, scales, group_size=32):
    """Обратная групповая квантизация."""
    result = []
    for i in range(len(scales)):
        start = i * group_size
        s = scales[i]
        for q in quantized[start:start + group_size]:
            result.append(q * s)
    return result


# ============================================================
# Демо 1: INT8 квантизация
# ============================================================

def demo_int8():
    print("=" * 60)
    print("Демо 1: INT8 квантизация")
    print("=" * 60)

    random.seed(42)
    weights = rand_list(16, lo=-2.0, hi=2.0)

    print(f"\nИсходные веса ({len(weights)} шт):")
    for i in range(0, len(weights), 4):
        chunk = [f"{w:+.4f}" for w in weights[i:i + 4]]
        print(f"  [{', '.join(chunk)}]")

    q_int8, scale = quantize_int8(weights)
    recovered = dequantize_int8(q_int8, scale)

    print(f"\nINT8 scale = {scale:.6f}")
    print(f"INT8 квантизованные ({len(q_int8)} шт):")
    for i in range(0, len(q_int8), 8):
        chunk = [f"{q:3d}" for q in q_int8[i:i + 8]]
        print(f"  [{', '.join(chunk)}]")

    errors = [abs(weights[i] - recovered[i]) for i in range(len(weights))]
    max_err = max(errors)
    avg_err = sum(errors) / len(errors)

    print(f"\nВосстановленные веса:")
    for i in range(0, len(recovered), 4):
        chunk = [f"{r:+.4f}" for r in recovered[i:i + 4]]
        print(f"  [{', '.join(chunk)}]")

    print(f"\nСтатистика ошибок:")
    print(f"  Макс. ошибка:    {max_err:.6f}")
    print(f"  Средняя ошибка:  {avg_err:.6f}")

    # Матричная квантизация
    print(f"\n--- Матрица 4×4 ---")
    mat = rand_matrix(4, 4, -1.0, 1.0)
    q_mat, s_mat = quantize_int8_matrix(mat)
    rec_mat = dequantize_int8_matrix(q_mat, s_mat)
    err = frobenius_error(mat, rec_mat)

    print(f"Исходная матрица:")
    for row in mat:
        print(f"  [{', '.join(f'{v:+.3f}' for v in row)}]")
    print(f"Квантизованная (int8):")
    for row in q_mat:
        print(f"  [{', '.join(f'{q:4d}' for q in row)}]")
    print(f"Относительная ошибка Фробениуса: {err * 100:.4f}%")

    print()


# ============================================================
# Демо 2: INT4 квантизация
# ============================================================

def demo_int4():
    print("=" * 60)
    print("Демо 2: INT4 квантизация")
    print("=" * 60)

    random.seed(42)
    weights = rand_list(16, lo=-1.5, hi=1.5)

    print(f"\nИсходные веса ({len(weights)} шт):")
    for i in range(0, len(weights), 4):
        chunk = [f"{w:+.4f}" for w in weights[i:i + 4]]
        print(f"  [{', '.join(chunk)}]")

    # Обычная INT4
    q_int4, scale = quantize_int4(weights)
    recovered = dequantize_int4(q_int4, scale)

    errors_std = [abs(weights[i] - recovered[i]) for i in range(len(weights))]
    avg_err_std = sum(errors_std) / len(errors_std)

    print(f"\n--- Обычная INT4 ---")
    print(f"Scale = {scale:.6f}")
    print(f"Квантизованные: {q_int4}")
    print(f"Средняя ошибка: {avg_err_std:.6f}")

    # Групповая INT4
    random.seed(42)
    weights2 = rand_list(64, lo=-1.5, hi=1.5)
    q_grp, scales = quantize_int4_grouped(weights2, group_size=16)
    rec_grp = dequantize_int4_grouped(q_grp, scales, group_size=16)

    errors_grp = [abs(weights2[i] - rec_grp[i]) for i in range(len(weights2))]
    avg_err_grp = sum(errors_grp) / len(errors_grp)

    print(f"\n--- Групповая INT4 (group_size=16) ---")
    print(f"Количество групп: {len(scales)}")
    print(f"Средняя ошибка:  {avg_err_grp:.6f}")
    print(f"Улучшение:       {(1 - avg_err_grp / avg_err_std) * 100:+.1f}%")

    # Размер сжатия
    print(f"\n--- Сжатие ---")
    original_bits = len(weights2) * 32
    int4_bits = len(q_grp) * 4
    print(f"Оригинал (float32): {original_bits} бит")
    print(f"INT4:               {int4_bits} бит")
    print(f"Коэффициент:        {original_bits / int4_bits:.1f}x")

    print()


# ============================================================
# Демо 3: Влияние на точность
# ============================================================

def demo_accuracy():
    print("=" * 60)
    print("Демо 3: Влияние квантизации на точность")
    print("=" * 60)

    random.seed(42)

    # Симуляция: матричное умножение
    rows, cols, vec_len = 8, 16, 16
    W = rand_matrix(rows, cols, -1.0, 1.0)
    x = rand_list(vec_len, -1.0, 1.0)

    # Оригинальный результат
    y_orig = vec_mul(W, x)

    # INT8
    W_q8, s8 = quantize_int8_matrix(W)
    W_rec8 = dequantize_int8_matrix(W_q8, s8)
    y_int8 = vec_mul(W_rec8, x)

    # INT4
    W_flat = [w for row in W for w in row]
    q4, s4 = quantize_int4(W_flat)
    W_rec4_flat = dequantize_int4(q4, s4)
    W_rec4 = [W_rec4_flat[i * cols:(i + 1) * cols] for i in range(rows)]
    y_int4 = vec_mul(W_rec4, x)

    # Групповая INT4
    q4g, s4g = quantize_int4_grouped(W_flat, group_size=8)
    W_rec4g_flat = dequantize_int4_grouped(q4g, s4g, group_size=8)
    W_rec4g = [W_rec4g_flat[i * cols:(i + 1) * cols] for i in range(rows)]
    y_int4g = vec_mul(W_rec4g, x)

    print(f"\nМатрица {rows}×{vec_len}, вектор длины {vec_len}")
    print(f"\nСравнение выходов (первые 8 элементов):")
    print(f"  {'Original':>12s}  {'INT8':>12s}  {'INT4':>12s}  {'INT4-group':>12s}")
    for i in range(min(8, rows)):
        print(f"  {y_orig[i]:+12.6f}  {y_int8[i]:+12.6f}  {y_int4[i]:+12.6f}  {y_int4g[i]:+12.6f}")

    # Косинусное сходство
    cos_8 = cosine_sim(y_orig, y_int8)
    cos_4 = cosine_sim(y_orig, y_int4)
    cos_4g = cosine_sim(y_orig, y_int4g)

    print(f"\nКосинусное сходство с оригиналом:")
    print(f"  INT8:       {cos_8:.6f}")
    print(f"  INT4:       {cos_4:.6f}")
    print(f"  INT4-group: {cos_4g:.6f}")

    # MSE
    mse_8 = sum((y_orig[i] - y_int8[i]) ** 2 for i in range(rows)) / rows
    mse_4 = sum((y_orig[i] - y_int4[i]) ** 2 for i in range(rows)) / rows
    mse_4g = sum((y_orig[i] - y_int4g[i]) ** 2 for i in range(rows)) / rows

    print(f"\nMSE (среднеквадратичная ошибка):")
    print(f"  INT8:       {mse_8:.8f}")
    print(f"  INT4:       {mse_4:.8f}")
    print(f"  INT4-group: {mse_4g:.8f}")

    print()


# ============================================================
# Демо 4: Размер моделей (Speed vs Accuracy tradeoff)
# ============================================================

def demo_model_sizes():
    print("=" * 60)
    print("Демо 4: Размер моделей — Speed vs Accuracy tradeoff")
    print("=" * 60)

    # Симуляция Llama-7B
    params_b = 7.0
    params = int(params_b * 1e9)

    print(f"\nМодель: ~{params_b:.0f}B параметров ({params:,})")
    print(f"{'Формат':<15s} {'Бит/параметр':<14s} {'Размер (GB)':<14s} {'Сжатие':<11s} {'Точность*':<11s}")
    print("-" * 65)

    formats = [
        ("FP32", 32, 1.0),
        ("FP16", 16, 1.0),
        ("INT8", 8, 1.0),
        ("INT4", 4, 1.0),
        ("INT4-group", 4, 0.02),  # ~2% overhead for scales
    ]

    fp32_size = params * 32 / (8 * 1024 ** 3)

    for name, bits, overhead in formats:
        size_gb = params * bits * (1 + overhead) / (8 * 1024 ** 3)
        compression = fp32_size / size_gb
        # Эмпирическая точность (относительно FP32)
        accuracy_map = {"FP32": 100.0, "FP16": 99.99, "INT8": 99.5, "INT4": 96.0, "INT4-group": 97.5}
        acc = accuracy_map[name]
        print(f"{name:<15s} {bits:<14d} {size_gb:<14.2f} {compression:.2f}x     {acc:.2f}%")

    print(f"\n* Точность — эмпирические значения для типичных NLP задач")

    # Speed-accuracy tradeoff
    print(f"\n--- Speed vs Accuracy Tradeoff ---")
    print(f"{'Метод':<18s} {'Отн. скорость':<16s} {'Отн. точность':<16s} {'Коэфф. эффективности':<20s}")
    print("-" * 70)

    methods = [
        ("FP32 baseline", 1.0, 1.00),
        ("FP16", 1.5, 1.00),
        ("INT8 (absmax)", 2.0, 0.995),
        ("INT4 (absmax)", 3.5, 0.960),
        ("INT4 (grouped)", 3.2, 0.975),
        ("INT4 + mixed", 3.0, 0.985),
    ]

    for name, speed, acc in methods:
        efficiency = speed * acc
        bar_len = int(efficiency * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{name:<18s} {speed:.2f}x       {acc:<16.3f} {bar} {efficiency:.2f}")

    # Практические рекомендации
    print(f"\n--- Практические рекомендации ---")
    print(f"1. FP16  — стандарт для инференса; минимум потерь, 2x экономия")
    print(f"2. INT8  — хорош для серверного инференса; ~2x ускорение на GPU")
    print(f"3. INT4  — для работы на потребительском железе (RTX 3090, Mac M2)")
    print(f"4. GROUPED INT4 —最佳 точность при малом размере; рекомендовано для LLM")

    # Стоимость инференса
    print(f"\n--- Стоимость инференса (1000 токенов, ~7B модель) ---")
    cost_data = [
        ("FP32", 7.0, "~30 ms/token"),
        ("FP16", 3.5, "~15 ms/token"),
        ("INT8", 1.75, "~8 ms/token"),
        ("INT4", 0.875, "~5 ms/token"),
    ]
    print(f"{'Формат':<10s} {'VRAM (GB)':<12s} {'Скорость':<18s} {'Экономия VRAM':<15s}")
    print("-" * 55)
    for name, vram, speed in cost_data:
        saving = f"{7.0 / vram:.1f}x"
        print(f"{name:<10s} {vram:<12.3f} {speed:<18s} {saving:<15s}")

    print()


# ============================================================
# Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  КВАНТИЗАЦИЯ: INT8 и INT4")
    print("  Основы сжатия весов нейросетей")
    print("=" * 60)
    print()

    demo_int8()
    demo_int4()
    demo_accuracy()
    demo_model_sizes()

    print("=" * 60)
    print("  Итоги:")
    print("  - INT8: простая квантизация, 4x сжатие, ~99.5% точности")
    print("  - INT4: агрессивное сжатие, 8x, ~96% точности")
    print("  - Групповая квантизация улучшает INT4 до ~97.5%")
    print("  - Tradeoff: размер ↔ скорость ↔ точность")
    print("=" * 60)
