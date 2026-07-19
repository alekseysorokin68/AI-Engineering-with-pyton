"""210 — GPU & Accelerators: CUDA основы, mixed precision, tensor cores

Темы:
  1. Архитектура GPU (SMs, warps, иерархия памяти, occupancy)
  2. Основы CUDA (kernels, thread blocks, grid, memory coalescing)
  3. Mixed Precision Training (FP16/BF16, loss scaling, AMP)
  4. Tensor Cores (ускорение матричных умножений, TF32, FP8)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# =============================================================================
# Демо 1: Архитектура GPU
# =============================================================================
def demo_gpu_architecture():
    """Демонстрация ключевых концепций архитектуры GPU"""
    print("=" * 70)
    print("ДЕМО 1: АРХИТЕКТУРА GPU — SMs, warps, иерархия памяти, occupancy")
    print("=" * 70)

    # --- 1.1 Модель Streaming Multiprocessor (SM) ---
    print("\n--- 1.1 Streaming Multiprocessor (SM) ---")

    # Класс, моделирующий один SM
    class StreamingMultiprocessor:
        """Модель одного SM GPU"""

        def __init__(self, cuda_cores=64, shared_mem_kb=64, registers=65536):
            self.cuda_cores = cuda_cores  # ядра CUDA
            self.shared_mem_bytes = shared_mem_kb * 1024  # разделяемая память
            self.registers = registers  # регистры на SM
            self.active_warps = 0
            self.max_warps = 48  # максимум warps на SM (Ampere)

        def calculate_occupancy(self, block_size, regs_per_thread, shared_per_block):
            """Вычисление occupancy — доли активных warps от максимума"""
            # Warp = 32 потока
            warps_per_block = math.ceil(block_size / 32)

            # Ограничение по warp slots
            max_blocks_by_warps = self.max_warps // warps_per_block

            # Ограничение по регистрам
            regs_per_block = regs_per_thread * block_size
            max_blocks_by_regs = self.registers // regs_per_block if regs_per_block > 0 else float('inf')

            # Ограничение по разделяемой памяти
            max_blocks_by_shared = self.shared_mem_bytes // shared_per_block if shared_per_block > 0 else float('inf')

            # Реальное число блоков — минимум из всех ограничений
            active_blocks = min(max_blocks_by_warps, max_blocks_by_regs, max_blocks_by_shared)
            active_blocks = max(active_blocks, 1)

            active_warps = active_blocks * warps_per_block
            active_warps = min(active_warps, self.max_warps)

            occupancy = active_warps / self.max_warps
            return {
                "active_blocks": active_blocks,
                "warps_per_block": warps_per_block,
                "active_warps": active_warps,
                "occupancy_pct": occupancy * 100
            }

    sm = StreamingMultiprocessor(cuda_cores=64, shared_mem_kb=64, registers=65536)

    # Сравниваем разные размеры блоков
    configs = [
        (64, 32, 4096),   # 64 потока, 32 регистров, 4 KB shared
        (128, 64, 8192),  # 128 потоков, 64 регистра, 8 KB shared
        (256, 64, 16384), # 256 потоков, 64 регистра, 16 KB shared
        (512, 32, 8192),  # 512 потоков, 32 регистра, 8 KB shared
    ]

    print(f"SM: {sm.cuda_cores} CUDA cores, {sm.shared_mem_bytes // 1024} KB shared mem, "
          f"{sm.registers} registers, max {sm.max_warps} warps")
    print()

    for block_size, regs, shared in configs:
        info = sm.calculate_occupancy(block_size, regs, shared)
        print(f"  Block={block_size:4d} threads, {regs:2d} regs/thread, "
              f"{shared // 1024:2d} KB shared → "
              f"blocks={info['active_blocks']}, "
              f"warps={info['active_warps']}, "
              f"occupancy={info['occupancy_pct']:.1f}%")

    # --- 1.2 Иерархия памяти GPU ---
    print("\n--- 1.2 Иерархия памяти GPU ---")

    memory_hierarchy = {
        "Регистры": {"объём": "256 КБ/SM", "задержка": "1 цикл", "пропуск": "~19 ТБ/с"},
        "Shared Memory": {"объём": "48-164 КБ/SM", "задержка": "~20-30 циклов", "пропуск": "~19 ТБ/с"},
        "L1 Cache": {"объём": "128 КБ/SM", "задержка": "~30 циклов", "пропуск": "~19 ТБ/с"},
        "L2 Cache": {"объём": "40 МБ (всего)", "задержка": "~200 циклов", "пропуск": "~5 ТБ/с"},
        "HBM (DRAM)": {"объём": "80 ГБ (A100)", "задержка": "~400-800 циклов", "пропуск": "2 ТБ/с"},
    }

    print("  Уровень          | Объём          | Задержка      | Пропуск")
    print("  " + "-" * 65)
    for name, props in memory_hierarchy.items():
        print(f"  {name:<17}| {props['объём']:<15}| {props['задержка']:<14}| {props['пропуск']}")

    # Время доступа в наносекундах (относительные)
    print("\n  Относительные задержки доступа:")
    latencies = {"Регистры": 1, "Shared/L1": 30, "L2": 200, "HBM": 600}
    for name, cycles in latencies.items():
        bar = "█" * min(cycles // 5, 60)
        print(f"    {name:<12} {cycles:>4} циклов  {bar}")

    # --- 1.3 Модель warp scheduling ---
    print("\n--- 1.3 Warp Scheduling и divergence ---")

    # Демонстрация warp divergence
    print("  Warp divergence — когда потоки в warp выполняют разные ветки:")
    print()

    warp_size = 32

    # Пример: if-else divergence
    conditions = [random.choice([True, False]) for _ in range(warp_size)]
    true_count = sum(conditions)
    false_count = warp_size - true_count

    print(f"  Условие: {true_count} потоков TRUE, {false_count} потоков FALSE")
    print(f"  Без divergence: 1 итерация")
    print(f"  С divergence: 2 итерации (обе ветки выполняются последовательно)")
    print(f"  Потеря эффективности: {((2/1) - 1) * 100:.0f}%")

    # Сериализация — все потоки в одном warp
    print(f"\n  Эффективность warp при divergence:")
    for true_pct in [100, 75, 50, 25, 0]:
        efficiency = max(true_pct, 100 - true_pct) / 100 * 100
        print(f"    {true_pct:3d}% true threads → эффективность: {efficiency:.0f}%")

    # --- 1.4 Occupancy и производительность ---
    print("\n--- 1.4 Влияние occupancy на производительность ---")

    # Модель: задержка вычислений скрывается при достаточном числе warps
    def compute_hide_latency(num_active_warps, compute_per_warp, memory_latency):
        """Модель: сколько warps нужно для скрытия задержки памяти"""
        # Warps, ожидающие памяти, должны компенсироваться вычисляющими warp'ами
        compute_time_per_warp = compute_per_warp
        warps_needed = math.ceil(memory_latency / compute_time_per_warp)
        return warps_needed

    print("  Для скрытия задержки памяти (latency hiding):")
    print("  memory_latency / compute_per_warp = warps_needed\n")

    for latency in [100, 200, 400, 800]:
        for compute in [10, 20, 50]:
            needed = compute_hide_latency(1, compute, latency)
            status = "OK" if needed <= 48 else "МАЛО"
            print(f"    latency={latency:>4}, compute={compute:>2} → "
                  f"нужно warps: {needed:>3} {status}")

    print("\n" + "=" * 70)
    print("ВЫВОД: Высокая occupancy = больше warps = лучше скрытие задержки памяти")
    print("=" * 70)


# =============================================================================
# Демо 2: Основы CUDA
# =============================================================================
def demo_cuda_basics():
    """Демонстрация CUDA programming model: kernels, blocks, grid, coalescing"""
    print("\n" + "=" * 70)
    print("ДЕМО 2: ОСНОВЫ CUDA — kernels, thread blocks, grid, memory coalescing")
    print("=" * 70)

    # --- 2.1 Вычислительная модель CUDA ---
    print("\n--- 2.1 Вычислительная модель CUDA ---")

    # Модель ядра (kernel)
    print("  CUDA Kernel — функция, выполняемая на GPU:")
    print("  __global__ void vector_add(float* a, float* b, float* c, int n) {")
    print("      int idx = blockIdx.x * blockDim.x + threadIdx.x;")
    print("      if (idx < n) c[idx] = a[idx] + b[idx];")
    print("  }")
    print()

    # Маппинг thread → индекс
    def compute_thread_index(block_idx, thread_idx, block_dim):
        """Вычисление глобального индекса потока"""
        return block_idx * block_dim + thread_idx

    # Пример: 3 блока по 4 потока
    block_dim = 4
    num_blocks = 3
    print(f"  blockDim.x = {block_dim}, gridDim.x = {num_blocks}")
    print(f"  Всего потоков: {block_dim * num_blocks}")
    print()

    print("  blockIdx | threadIdx | Глобальный индекс")
    print("  " + "-" * 45)
    for bx in range(num_blocks):
        for tx in range(block_dim):
            global_idx = compute_thread_index(bx, tx, block_dim)
            print(f"  {bx:>8} | {tx:>9} | {global_idx:>16}")

    # --- 2.2 Запуск kernel'а (host-side) ---
    print("\n--- 2.2 Запуск kernel'а (host → device) ---")

    # Модель данных
    n = 16
    host_a = [random.uniform(-1, 1) for _ in range(n)]
    host_b = [random.uniform(-1, 1) for _ in range(n)]

    # На CPU (host)
    host_result = [a + b for a, b in zip(host_a, host_b)]

    # Модель GPU kernel
    block_size = 4
    grid_size = math.ceil(n / block_size)

    def vector_add_kernel(a, b, n, block_size, grid_size):
        """Модель CUDA kernel на Python"""
        result = [0.0] * n
        for bx in range(grid_size):
            for tx in range(block_size):
                idx = bx * block_size + tx
                if idx < n:
                    result[idx] = a[idx] + b[idx]
        return result

    device_result = vector_add_kernel(host_a, host_b, n, block_size, grid_size)

    print(f"  n={n}, block_size={block_size}, grid_size={grid_size}")
    print(f"  Host и Device результаты совпадают: {host_result == device_result}")
    print(f"  Примеры: host[0]={host_result[0]:.4f}, device[0]={device_result[0]:.4f}")

    # --- 2.3 Memory Coalescing ---
    print("\n--- 2.3 Memory Coalescing ---")

    print("  Coalesced access: потоки в warp читают соседние адреса")
    print("  → 1 транзакция памяти на весь warp\n")

    # Модель: считаем транзакции
    def count_memory_transactions(access_pattern, warp_size=32):
        """Подсчёт транзакций памяти для different паттернов доступа"""
        transactions = 0
        cache_line = 128  # байт в cache line

        if access_pattern == "coalesced":
            # Все 32 потока читают подряд (4 байта каждый)
            # = 1 транзакция (128 байт cache line)
            transactions = 1
        elif access_pattern == "strided_4":
            # Каждый поток через 4 элемента (stride=16 байт)
            # = 4 транзакции (4 cache line'а)
            transactions = 4
        elif access_pattern == "strided_32":
            # Stride = 32 элемента (128 байт)
            # = 32 транзакции
            transactions = 32
        elif access_pattern == "random":
            # Случайные адреса — каждая транзакция отдельно
            transactions = warp_size

        return transactions

    patterns = ["coalesced", "strided_4", "strided_32", "random"]
    for pat in patterns:
        trans = count_memory_transactions(pat)
        efficiency = 32 / trans if trans > 0 else 0
        bar = "█" * min(trans, 32)
        print(f"  {pat:<15} → {trans:>3} транзакций, эффективность: {efficiency:.1%}")

    # Пример coalesced vs strided
    print("\n  Пример: чтение массива float[1024]")
    print("  Coalesced: thread 0 → a[0], thread 1 → a[1], ..., thread 31 → a[31]")
    print("    → 1 транзакция на warp (128 байт)")
    print("  Strided(4): thread 0 → a[0], thread 1 → a[4], ..., thread 31 → a[128]")
    print("    → 4 транзакции на warp (4 × 128 байт)")

    # --- 2.4 Shared Memory и bank conflicts ---
    print("\n--- 2.4 Shared Memory и bank conflicts ---")

    def count_bank_conflicts(access_pattern, num_banks=32):
        """Подсчёт bank conflicts в shared memory"""
        if access_pattern == "no_conflict":
            # Каждый поток в不同的 bank
            return 1  # 1-way (без конфликта)
        elif access_pattern == "broadcast":
            # Все читают один адрес → broadcast
            return 1  # broadcast без конфликта
        elif access_pattern == "sequential":
            # stride-1 доступ
            return 1
        elif access_pattern == "stride_2":
            # stride-2 → 2-way bank conflict
            return 2
        elif access_pattern == "stride_32":
            # stride = num_banks → все потоки в одном bank!
            return 32  # 32-way bank conflict!
        else:
            return 1

    print("  Shared Memory: 32 banks (по 4 байта каждый)")
    print()
    for pat in ["no_conflict", "broadcast", "sequential", "stride_2", "stride_32"]:
        conflicts = count_bank_conflicts(pat)
        print(f"  {pat:<20} → {conflicts:>2}-way bank conflict")

    print("\n  Формула: Bank Conflict = количество потоков, попадающих в один bank")
    print("  32-way conflict = 32x замедление доступа к shared memory!")

    print("\n" + "=" * 70)
    print("ВЫВОД: Coalesced access к global memory + bank-conflict-free shared memory = максимум пропускной способности")
    print("=" * 70)


# =============================================================================
# Демо 3: Mixed Precision Training
# =============================================================================
def demo_mixed_precision():
    """Демонстрация mixed precision: FP16/BF16, loss scaling, AMP"""
    print("\n" + "=" * 70)
    print("ДЕМО 3: MIXED PRECISION TRAINING — FP16/BF16, loss scaling, AMP")
    print("=" * 70)

    # --- 3.1 Форматы с плавающей точкой ---
    print("\n--- 3.1 Форматы с плавающей точкой ---")

    class FloatFormat:
        """Модель формата с плавающей точкой"""

        def __init__(self, name, exponent_bits, mantissa_bits):
            self.name = name
            self.exp_bits = exponent_bits
            self.mantissa_bits = mantissa_bits
            self.total_bits = 1 + exponent_bits + mantissa_bits  # sign + exp + mantissa

            # Диапазон
            self.max_exp = (2 ** exponent_bits - 1) // 2  # bias
            self.max_val = (2 - 2**(-mantissa_bits)) * (2 ** self.max_exp)
            self.min_normal = 2 ** (1 - self.max_exp)
            self.epsilon = 2 ** (-mantissa_bits)

        def info(self):
            return (f"{self.name}: {self.total_bits} bit "
                    f"(1 sign + {self.exp_bits} exp + {self.mantissa_bits} mantissa), "
                    f"max={self.max_val:.2e}, eps={self.epsilon:.2e}")

    formats = [
        FloatFormat("FP32", 8, 23),
        FloatFormat("TF32", 8, 10),
        FloatFormat("FP16", 5, 10),
        FloatFormat("BF16", 8, 7),
        FloatFormat("FP8-E4M3", 4, 3),
        FloatFormat("FP8-E5M2", 5, 2),
    ]

    print(f"  {'Формат':<12} {'Бит':<6} {'Максимум':<12} {'Epsilon':<12}")
    print("  " + "-" * 42)
    for f in formats:
        print(f"  {f.name:<12} {f.total_bits:<6} {f.max_val:<12.2e} {f.epsilon:<12.2e}")

    # --- 3.2 Precision comparison на задаче ---
    print("\n--- 3.2 Сравнение точности на задаче ---")

    # Модель умножения матриц с разной точностью
    def matmul_precision(a, b, precision="fp32"):
        """Модель матричного умножения с разной точностью"""
        n = len(a)
        result = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    val = a[i][k] * b[k][j]
                    # Округление в зависимости от формата
                    if precision == "fp16":
                        val = round(val, 3)  # ~3 десятичных знака
                    elif precision == "bf16":
                        val = round(val, 2)  # ~2 десятичных знака
                    elif precision == "fp8":
                        val = round(val, 1)  # ~1 десятичный знак
                    acc += val
                    if precision == "fp16":
                        acc = round(acc, 3)
                    elif precision == "bf16":
                        acc = round(acc, 2)
                    elif precision == "fp8":
                        acc = round(acc, 1)
                result[i][j] = acc
        return result

    # Генерируем матрицы
    n = 4
    random.seed(42)
    a = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    b = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]

    # Эталон FP32
    ref = matmul_precision(a, b, "fp32")

    print("  Матрица A (4x4), Матрица B (4x4)")
    print(f"  Эталон (FP32) результат[0][0] = {ref[0][0]:.6f}")
    print()

    for prec in ["fp16", "bf16", "fp8"]:
        result = matmul_precision(a, b, prec)
        error = abs(result[0][0] - ref[0][0])
        rel_error = error / abs(ref[0][0]) * 100 if ref[0][0] != 0 else 0
        print(f"  {prec.upper():>5} результат[0][0] = {result[0][0]:>10.6f}, "
              f"abs误差={error:.6f}, rel误差={rel_error:.2f}%")

    # --- 3.3 Loss Scaling ---
    print("\n--- 3.3 Loss Scaling для FP16 ---")

    # Проблема: малые градиенты уходят в 0 при FP16
    print("  Проблема: FP16 min_normal ≈ 6.1e-5")
    print("  Малые градиенты (< 6.1e-5) → стёрты в 0!\n")

    # Модель grad scaling
    def simulate_loss_scaling(learning_rates, grad_magnitudes, scale_factor):
        """Модель: loss scaling спасает малые градиенты"""
        fp16_min_normal = 6.103515625e-05
        results = []

        for lr, grad in zip(learning_rates, grad_magnitudes):
            # Без scaling
            grad_no_scale = grad
            underflow_no_scale = abs(grad_no_scale) < fp16_min_normal

            # С loss scaling
            grad_scaled = grad * scale_factor
            grad_reduced = grad_scaled  # делим на scale_factor после backward
            underflow_scaled = abs(grad) < fp16_min_normal / scale_factor

            results.append({
                "lr": lr,
                "grad": grad,
                "no_scale": underflow_no_scale,
                "scaled": underflow_scaled,
                "scale": scale_factor
            })

        return results

    lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    grads = [0.1, 0.01, 1e-4, 1e-6, 1e-8]

    print(f"  {'Gradient':<12} {'Без scaling':<14} {'С scaling (1024)':<16}")
    print("  " + "-" * 42)
    for lr, g in zip(lrs, grads):
        no_scale_underflow = abs(g) < 6.103515625e-05
        scale_factor = 1024
        scaled_underflow = abs(g * scale_factor) < 6.103515625e-05

        print(f"  {g:<12.1e} {'underflow!' if no_scale_underflow else 'OK':<14} "
              f"{'underflow!' if scaled_underflow else 'OK':<16}")

    # Динамическое scaling
    print("\n  Динамический loss scaling:")
    print("  1. Начинаем с scale_factor = 2^16")
    print("  2. Если нет overflow → scale_factor *= 2")
    print("  3. Если overflow → scale_factor /= 2, пропускаем шаг")
    print()

    scale = 2**16
    overflow_count = 0
    for step in range(10):
        # Симуляция: случайный overflow
        has_overflow = random.random() < 0.2
        if has_overflow:
            scale = max(scale // 2, 1)
            overflow_count += 1
            status = "OVERFLOW → scale /= 2"
        else:
            scale = min(scale * 2, 2**30)
            status = "OK → scale *= 2"
        print(f"  Step {step:>2}: scale={scale:>10}, {status}")

    # --- 3.4 AMP (Automatic Mixed Precision) ---
    print("\n--- 3.4 Automatic Mixed Precision (AMP) ---")

    print("  AMP Policy: какие операции в каком формате:")
    print()

    op_policy = {
        "Matrix Multiply": "FP16/BF16 (Tensor Core)",
        "Convolution": "FP16/BF16 (Tensor Core)",
        "BatchNorm": "FP32 ( numerically sensitive )",
        "Softmax": "FP32",
        "Loss computation": "FP32",
        "Gradient AllReduce": "FP16 ( экономит bandwidth )",
        "Optimizer step": "FP32 ( master weights )",
    }

    for op, policy in op_policy.items():
        print(f"    {op:<25} → {policy}")

    print("\n  AMP память: master weights в FP32 + копия в FP16")
    print("  Экономия: ~50% памяти для activations")
    print("  Ускорение: ~2x на Tensor Core операциях")

    print("\n" + "=" * 70)
    print("ВЫВОД: Mixed precision даёт 2x ускорение + 50% экономии памяти с минимальной потерей точности")
    print("=" * 70)


# =============================================================================
# Демо 4: Tensor Cores
# =============================================================================
def demo_tensor_cores():
    """Демонстрация Tensor Cores: matrix multiply acceleration, TF32, FP8"""
    print("\n" + "=" * 70)
    print("ДЕМО 4: TENSOR CORES — ускорение матричных умножений, TF32, FP8")
    print("=" * 70)

    # --- 4.1 Что такое Tensor Core ---
    print("\n--- 4.1 Что такое Tensor Core ---")

    print("  Обычное ядро CUDA:    1 FMA (fused multiply-add) за такт")
    print("  Tensor Core:          D = A × B + C  (матрицы 16x16)")
    print("                        за 1 такт — в 16 раз больше!\n")

    # Модель: CUDA core vs Tensor Core
    def compute_fma_count(m, n, k, tile_size=16):
        """Подсчёт FMA операций для матричного умножения M×K × K×N"""
        total_fma = m * n * k  # каждая позиция = K умножений-сложений

        # CUDA cores: по 1 FMA за такт на ядро
        cuda_cycles = total_fma  # без параллелизма

        # Tensor cores: tile_size×tile_size×tile_size за такт
        tiles_m = math.ceil(m / tile_size)
        tiles_n = math.ceil(n / tile_size)
        tiles_k = math.ceil(k / tile_size)
        tensor_cycles = tiles_m * tiles_n * tiles_k

        speedup = cuda_cycles / tensor_cycles if tensor_cycles > 0 else 0

        return {
            "total_fma": total_fma,
            "cuda_cycles": cuda_cycles,
            "tensor_cycles": tensor_cycles,
            "speedup": speedup
        }

    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024),
             (4096, 4096, 4096)]

    print("  Сравнение CUDA Core vs Tensor Core (M×K × K×N):")
    print(f"  {'Размер':<18} {'FMA ops':<15} {'CUDA cycles':<14} {'Tensor cycles':<15} {'Speedup':<10}")
    print("  " + "-" * 70)

    for m, n, k in sizes:
        info = compute_fma_count(m, n, k)
        print(f"  {m}×{k}×{n:<10} {info['total_fma']:<15,} {info['cuda_cycles']:<14,} "
              f"{info['tensor_cycles']:<15,} {info['speedup']:<10.1f}x")

    # --- 4.2 TF32 (TensorFloat-32) ---
    print("\n--- 4.2 TF32 (TensorFloat-32) ---")

    print("  TF32 = FP32 range + FP16 precision")
    print("  1 sign + 8 exponent + 10 mantissa = 19 bit\n")

    print("  Форматы для Tensor Cores:")
    formats_tc = [
        ("TF32", 19, "8 exp + 10 man", "FP32 range, FP16 precision"),
        ("FP16", 16, "5 exp + 10 man", "Полная точность FP16"),
        ("BF16", 16, "8 exp + 7 man", "FP32 range, reduced precision"),
        ("FP8-E4M3", 8, "4 exp + 3 man", "Максимальная throughput"),
        ("FP8-E5M2", 8, "5 exp + 2 man", "Больше range, меньше precision"),
    ]

    print(f"  {'Формат':<12} {'Бит':<5} {'Структура':<18} {'Описание':<35}")
    print("  " + "-" * 70)
    for name, bits, struct, desc in formats_tc:
        print(f"  {name:<12} {bits:<5} {struct:<18} {desc}")

    # TF32 vs FP32 точность
    print("\n  TF32 vs FP32 точность (модель):")

    def tf32_round(value):
        """Модель TF32: 10-bit mantissa (vs 23-bit FP32)"""
        if value == 0:
            return 0.0
        # Округляем до ~3 знаков после запятой
        return round(value, 3)

    random.seed(42)
    test_values = [random.uniform(-100, 100) for _ in range(8)]

    print("    FP32 значение     TF32 значение     Разница")
    print("    " + "-" * 45)
    for v in test_values:
        tf32_v = tf32_round(v)
        diff = abs(v - tf32_v)
        print(f"    {v:>16.8f}   {tf32_v:>16.8f}   {diff:.8f}")

    # --- 4.3 Throughput Tensor Cores (TFLOPS) ---
    print("\n--- 4.3 Throughput Tensor Cores ---")

    gpu_specs = {
        "A100 (FP32 CUDA)": {"tflops": 19.5, "mem_tb_s": 2.0, "mem_gb": 80},
        "A100 (TF32 TC)": {"tflops": 156, "mem_tb_s": 2.0, "mem_gb": 80},
        "A100 (FP16 TC)": {"tflops": 312, "mem_tb_s": 2.0, "mem_gb": 80},
        "H100 (FP32 CUDA)": {"tflops": 67, "mem_tb_s": 3.35, "mem_gb": 80},
        "H100 (FP16 TC)": {"tflops": 989, "mem_tb_s": 3.35, "mem_gb": 80},
        "H100 (FP8 TC)": {"tflops": 1979, "mem_tb_s": 3.35, "mem_gb": 80},
    }

    print(f"  {'GPU + Format':<24} {'TFLOPS':<10} {'Bandwidth':<12} {'Memory':<10}")
    print("  " + "-" * 56)
    for name, specs in gpu_specs.items():
        print(f"  {name:<24} {specs['tflops']:<10.0f} {specs['mem_tb_s']:<12.1f} "
              f"{specs['mem_gb']:<10}")

    # Arithmetic intensity
    print("\n  Arithmetic Intensity (AI) = FLOPs / Bytes")
    print("  AI > 算力/带宽 → compute-bound (Tensor Core helps)")
    print("  AI < 算力/带宽 → memory-bound ( bandwidth helps)\n")

    for name, specs in gpu_specs.items():
        ai = specs['tflops'] * 1000 / (specs['mem_tb_s'] * 1e6)  # TFLOPS/TB/s
        bound = "compute" if ai > 100 else "memory"
        print(f"  {name:<24} AI ≈ {ai:.1f} → {bound}-bound")

    # --- 4.4 FP8 Training ---
    print("\n--- 4.4 FP8 Training ---")

    print("  FP8 Enables:")
    print("    • 2x throughput vs FP16 (H100)")
    print("    • 2x memory savings")
    print("    • Requires careful loss scaling\n")

    # Модель FP8 quantization
    def quantize_to_fp8_e4m3(value):
        """Модель FP8 E4M3: 4-bit exponent, 3-bit mantissa"""
        if value == 0:
            return 0.0
        # Ограничиваем диапазон
        max_val = 448.0
        min_val = 0.001953125  # 2^(-8)
        value = max(min(value, max_val), -max_val)
        if abs(value) < min_val:
            return 0.0
        # Округляем до 3 бит мантиссы (~1 десятичный знак)
        return round(value, 1)

    print("  FP8 E4M3 квантизация:")
    values = [3.14159, 0.00123, 100.5, 0.0001, -42.7]
    for v in values:
        q = quantize_to_fp8_e4m3(v)
        error = abs(v - q)
        print(f"    {v:>12.5f} → {q:>12.5f}  (error: {error:.5f})")

    # Training loop с FP8
    print("\n  Training loop с FP8:")
    print("    1. Forward pass: activations в FP8")
    print("    2. Backward pass: градиенты в FP8")
    print("    3. Weight update: master weights в FP32")
    print("    4. Quantize weights → FP8 для следующего шага")

    # Результаты FP8 vs FP16
    print("\n  Эмпирические результаты (GPT-3 175B на H100):")
    results = [
        ("FP32 (baseline)", "1.0x", "100%"),
        ("TF32", "1.6x", "99.8%"),
        ("FP16/BF16", "3.2x", "99.7%"),
        ("FP8", "6.4x", "99.5%"),
    ]
    print(f"    {'Format':<20} {'Speedup':<10} {'Accuracy retention'}")
    print("    " + "-" * 45)
    for fmt, speed, acc in results:
        print(f"    {fmt:<20} {speed:<10} {acc}")

    print("\n" + "=" * 70)
    print("ВЫВОД: Tensor Cores дают 10-16x ускорение GEMM, FP8 добавляет ещё 2x")
    print("=" * 70)


# =============================================================================
# Запуск всех демонстраций
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("УРОК 210: GPU & ACCELERATORS")
    print("CUDA основы, mixed precision, tensor cores")
    print("=" * 70)
    print()

    demo_gpu_architecture()
    demo_cuda_basics()
    demo_mixed_precision()
    demo_tensor_cores()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 70)
