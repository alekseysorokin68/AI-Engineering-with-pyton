"""
Weight Initialization Methods from Scratch
==========================================
Реализация методов инициализации весов нейронных сетей:
- Random Normal
- Xavier/Glorot (uniform и normal)
- He (uniform и normal)

Влияние на затухание/взрыв градиентов.
Без внешних зависимостей (numpy, torch, sklearn).
"""

import random
import math

random.seed(42)


# ============================================================
# Утилиты: генерация случайных чисел с разными распределениями
# ============================================================

def rand_normal(mu=0.0, sigma=1.0):
    """Box-Muller transform: генерация числа из N(mu, sigma^2)."""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z


def rand_uniform(low, high):
    """Случайное число из U[low, high)."""
    return low + random.random() * (high - low)


# ============================================================
# Методы инициализации весов
# ============================================================

def random_normal_init(rows, cols, mu=0.0, sigma=0.01):
    """
    Random Normal Initialization
    -----------------------------
    W ~ N(mu, sigma^2)

    Классический подход: маленькие случайные числа around 0.
    Проблема: sigma подбирается вручную, нет теоретического обоснования.
    """
    return [[rand_normal(mu, sigma) for _ in range(cols)] for _ in range(rows)]


def xavier_uniform_init(rows, cols):
    """
    Xavier/Glorot Uniform Initialization
    --------------------------------------
    W ~ U[-limit, +limit], limit = sqrt(6 / (fan_in + fan_out))

    Обеспечивает сохранение дисперсии сигнала в прямом и обратном проходе.
    Подходит для сигмоид/tanh активаций (линейные в области определения).
    """
    limit = math.sqrt(6.0 / (rows + cols))
    return [[rand_uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def xavier_normal_init(rows, cols):
    """
    Xavier/Glorot Normal Initialization
    -------------------------------------
    W ~ N(0, sqrt(2 / (fan_in + fan_out)))

    Аналогичная идея, но нормальное распределение вместо равномерного.
    """
    sigma = math.sqrt(2.0 / (rows + cols))
    return [[rand_normal(0, sigma) for _ in range(cols)] for _ in range(rows)]


def he_uniform_init(rows, cols):
    """
    He Uniform Initialization
    --------------------------
    W ~ U[-limit, +limit], limit = sqrt(6 / fan_in)

    Разработана для ReLU: учитывает, что ReLU обнуляет ~50% нейронов.
    fan_in — количество входящих связей (rows).
    """
    limit = math.sqrt(6.0 / rows)
    return [[rand_uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def he_normal_init(rows, cols):
    """
    He Normal Initialization
    -------------------------
    W ~ N(0, sqrt(2 / fan_in))

    Оптимальна для ReLU и её вариантов (LeakyReLU, PReLU).
    """
    sigma = math.sqrt(2.0 / rows)
    return [[rand_normal(0, sigma) for _ in range(cols)] for _ in range(rows)]


# ============================================================
# Вспомогательные функции для анализа
# ============================================================

def matrix_stats(W):
    """Вычисление статистик матрицы весов."""
    flat = [w for row in W for w in row]
    n = len(flat)
    mean = sum(flat) / n
    variance = sum((x - mean) ** 2 for x in flat) / n
    std = math.sqrt(variance)
    min_val = min(flat)
    max_val = max(flat)
    return {"mean": mean, "std": std, "min": min_val, "max": max_val, "count": n}


def sigmoid(x):
    """Сигмоида."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x):
    """ReLU."""
    return max(0.0, x)


def relu_derivative(x):
    """Производная ReLU."""
    return 1.0 if x > 0 else 0.0


def sigmoid_derivative(x):
    """Производная сигмоиды по выходу sig(x)."""
    return x * (1.0 - x)


def mat_vec_mul(W, v):
    """Умножение матрицы на вектор."""
    return [sum(W[i][j] * v[j] for j in range(len(v))) for i in range(len(W))]


def print_matrix_sample(W, name, rows=4, cols=6):
    """Печать фрагмента матрицы."""
    print(f"\n  {name} (фрагмент {rows}x{cols} из {len(W)}x{len(W[0])}):")
    for i in range(min(rows, len(W))):
        vals = " ".join(f"{W[i][j]:+.4f}" for j in range(min(cols, len(W[0]))))
        print(f"    [{vals} ...]")


def print_stats(name, stats):
    """Печать статистик."""
    print(f"  {name:30s} | mean={stats['mean']:+.6f} | std={stats['std']:.6f} "
          f"| range=[{stats['min']:+.4f}, {stats['max']:+.4f}]")


# ============================================================
# ДЕМО 1: Все методы инициализации — визуальное сравнение
# ============================================================

def demo1_initialization_methods():
    print("=" * 70)
    print("ДЕМО 1: Все методы инициализации весов")
    print("=" * 70)

    fan_in, fan_out = 128, 64  # типичный полносвязный слой

    methods = {
        "Random Normal (σ=0.01)":    random_normal_init(fan_in, fan_out, 0, 0.01),
        "Random Normal (σ=0.1)":     random_normal_init(fan_in, fan_out, 0, 0.1),
        "Xavier Uniform":            xavier_uniform_init(fan_in, fan_out),
        "Xavier Normal":             xavier_normal_init(fan_in, fan_out),
        "He Uniform":                he_uniform_init(fan_in, fan_out),
        "He Normal":                 he_normal_init(fan_in, fan_out),
    }

    print(f"\nРазмер матрицы: {fan_in} x {fan_out} (fan_in={fan_in}, fan_out={fan_out})")
    print(f"fan_in + fan_out = {fan_in + fan_out}")
    print(f"sqrt(2/fan_in)  = {math.sqrt(2/fan_in):.4f}  (He σ)")
    print(f"sqrt(2/(f+fo))  = {math.sqrt(2/(fan_in+fan_out)):.4f}  (Xavier σ)")
    print()

    for name, W in methods.items():
        stats = matrix_stats(W)
        print_stats(name, stats)
        print_matrix_sample(W, name, rows=3, cols=8)

    print("\n" + "-" * 70)
    print("ВЫВОД:")
    print("  - Random Normal (σ=0.01): веса ОЧЕНЬ малы → градиенты затухнут")
    print("  - Random Normal (σ=0.1):  лучше, но нет гарантий")
    print("  - Xavier: дисперсия ≈ 2/(fan_in+fan_out) → баланс для sigmoid/tanh")
    print("  - He: дисперсия ≈ 2/fan_in → компенсирует обнуление ReLU")


# ============================================================
# ДЕМО 2: Статистика весов — теория vs практика
# ============================================================

def demo2_weight_statistics():
    print("\n\n" + "=" * 70)
    print("ДЕМО 2: Статистика весов (теория vs практика)")
    print("=" * 70)

    random.seed(42)

    configs = [
        (64, 32),    # маленький слой
        (128, 128),  # квадратный
        (256, 64),   # сужение
        (64, 256),   # расширение
    ]

    print("\n{'Слой':>14s} | {'Теор. Xavier σ':>14s} | {'Практ. Xavier σ':>15s} "
          "| {'Теор. He σ':>12s} | {'Практ. He σ':>13s}")
    print("-" * 80)

    for fan_in, fan_out in configs:
        xavier_theory = math.sqrt(2.0 / (fan_in + fan_out))
        he_theory = math.sqrt(2.0 / fan_in)

        W_xavier = xavier_normal_init(fan_in, fan_out)
        W_he = he_normal_init(fan_in, fan_out)

        xavier_actual = matrix_stats(W_xavier)["std"]
        he_actual = matrix_stats(W_he)["std"]

        print(f"  {fan_in}x{fan_out:>6d} | {xavier_theory:14.6f} | {xavier_actual:15.6f} "
              f"| {he_theory:12.6f} | {he_actual:13.6f}")

    print("\nВЫВОД: Практическая σ ≈ теоретической (при достаточном кол-ве элементов).")
    print("  Чем больше матрица, тем точнее совпадение (закон больших чисел).")


# ============================================================
# ДЕМО 3: Проблема затухающих/взрывающихся градиентов
# ============================================================

def demo3_vanishing_gradients():
    print("\n\n" + "=" * 70)
    print("ДЕМО 3: Затухание и взрыв градиентов")
    print("=" * 70)

    random.seed(42)

    layer_sizes = [64, 64, 64, 64, 64, 64]  # 6 слоёв

    def simulate_backward(weights_list, activation="sigmoid"):
        """Обратный проход: произведение якобианов через слои."""
        grad_norm = 1.0
        history = [grad_norm]

        for i in range(len(weights_list) - 1, -1, -1):
            W = weights_list[i]
            fan_in = len(W)
            fan_out = len(W[0])

            # Средняя якобиана: E[∂a/∂z]取决于 активации
            if activation == "sigmoid":
                # Для sigmoid: max производной = 0.25, типичная ≈ 0.2
                jacobian_avg = 0.25
            elif activation == "relu":
                # Для ReLU: ~50% нейронов активны → avg = 0.5
                jacobian_avg = 0.5
            else:
                jacobian_avg = 0.25

            # Масштабирование весами
            weight_scale = math.sqrt(2.0 / fan_in)  # He-инициализация
            grad_norm *= jacobian_avg * weight_scale * math.sqrt(fan_in)
            history.append(grad_norm)

        return history

    # --- Сигмоида + разные инициализации ---
    print("\n--- Сигмоида + разные инициализации (6 слоёв, 64→64) ---\n")

    init_methods = {
        "Случайная (σ=0.01)": lambda fi, fo: random_normal_init(fi, fo, 0, 0.01),
        "Xavier":             xavier_normal_init,
        "He":                 he_normal_init,
    }

    for name, init_fn in init_methods.items():
        weights = [init_fn(layer_sizes[i], layer_sizes[i + 1])
                   for i in range(len(layer_sizes) - 1)]

        # Прямой проход: масштабирование сигнала
        signal = 1.0
        signal_history = [signal]
        for i, W in enumerate(weights):
            fan_in = len(W)
            fan_out = len(W[0])
            stats = matrix_stats(W)
            # Средний выход слоя (при единичном входе)
            signal *= stats["std"] * math.sqrt(fan_in) * 0.5  # 0.5 avg activation
            signal_history.append(signal)

        print(f"  {name}:")
        print(f"    Прямой:  {['%.4e' % s for s in signal_history]}")
        print(f"    Финальный сигнал: {signal_history[-1]:.4e}")

    # --- ReLU: затухание vs взрыв ---
    print("\n--- ReLU + He: затухание vs взрыв при разных длинах ---\n")

    for depth in [3, 6, 10, 15]:
        weights = [he_normal_init(layer_sizes[min(i, len(layer_sizes) - 2)],
                                  layer_sizes[min(i + 1, len(layer_sizes) - 1)])
                   for i in range(depth)]

        signal = 1.0
        for W in weights:
            fan_in = len(W)
            signal *= 0.5 * math.sqrt(2.0 / fan_in) * math.sqrt(fan_in)

        print(f"  {depth:2d} слоёв: финальный сигнал = {signal:.4e}", end="")
        if abs(signal) < 1e-10:
            print("  ← ЗАТУХАНИЕ!")
        elif abs(signal) > 1e10:
            print("  ← ВЗРЫВ!")
        else:
            print("  ← стабильно")

    print("\n" + "-" * 70)
    print("ВЫВОД:")
    print("  Сигмоида: градиенты затухают экспоненциально (max ∂σ/∂x = 0.25)")
    print("  ReLU + He: сигнал сохраняется лучше (нет затухания через 0)")
    print("  Глубокие сети (>10 слоёв): даже с He нужно batch norm / residual")


# ============================================================
# ДЕМО 4: Сравнение на практике — имитация обучения
# ============================================================

def demo4_practical_comparison():
    print("\n\n" + "=" * 70)
    print("ДЕМО 4: Сравнение на практике — имитация обучения")
    print("=" * 70)

    random.seed(42)

    # Простая сеть: 8 → 16 → 16 → 1 (классификация)
    architecture = [8, 16, 16, 1]
    lr = 0.01
    epochs = 200

    def forward_pass(X_batch, weights, biases, activation="relu"):
        """Прямой проход по сети."""
        activations = [X_batch]
        zs = []  # pre-activation значения

        current = X_batch
        for layer_idx in range(len(weights)):
            W = weights[layer_idx]
            b = biases[layer_idx]
            fan_in = len(W)
            fan_out = len(W[0])

            # Z = W @ a + b
            new_current = []
            for j in range(fan_out):
                z = b[j]
                for i in range(fan_in):
                    z += W[i][j] * current[i]
                zs.append(z)

                if layer_idx < len(weights) - 1:
                    # Скрытые слои: ReLU
                    new_current.append(relu(z))
                else:
                    # Выходной слой: линейный (для MSE)
                    new_current.append(z)
            current = new_current
            activations.append(current)

        return current[0], activations, zs

    def compute_loss(predictions, targets):
        """MSE loss."""
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

    def train_network(init_name, init_fn, X_train, y_train):
        """Обучение сети с заданной инициализацией."""
        random.seed(42)

        weights = []
        biases = []
        for i in range(len(architecture) - 1):
            W = init_fn(architecture[i], architecture[i + 1])
            b = [0.0] * architecture[i + 1]
            weights.append(W)
            biases.append(b)

        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in zip(X_train, y_train):
                pred, activations, zs = forward_pass(x, weights, biases)
                loss = (pred - y) ** 2
                total_loss += loss

                # Упрощённый обратный проход (градиентный спуск)
                # dL/dpred = 2*(pred - y) / n
                delta = 2.0 * (pred - y) / len(y_train)

                for layer_idx in range(len(weights) - 1, -1, -1):
                    fan_in = len(weights[layer_idx])
                    fan_out = len(weights[layer_idx][0])

                    # Обновление биасов
                    biases[layer_idx][0] -= lr * delta

                    # Градиенты по весам
                    new_delta = [0.0] * fan_in
                    for j in range(fan_out):
                        for i in range(fan_in):
                            grad = delta * activations[layer_idx][i]
                            weights[layer_idx][i][j] -= lr * grad
                            new_delta[i] += delta * weights[layer_idx][i][j]

                    # Производная активации (ReLU для скрытых слоёв)
                    if layer_idx > 0:
                        for i in range(fan_in):
                            if zs[len(zs) - fan_in + i - 1] <= 0:
                                new_delta[i] = 0.0

                    delta = sum(new_delta) / max(fan_in, 1)

            avg_loss = total_loss / len(y_train)
            losses.append(avg_loss)

        return losses

    # Генерация данных: XOR-подобная задача
    random.seed(123)
    n_samples = 50
    X_train = [[random.gauss(0, 1) for _ in range(8)] for _ in range(n_samples)]
    # Простая целевая функция: знак суммы первых 4 признаков
    y_train = [1.0 if sum(x[:4]) > 0 else -1.0 for x in X_train]

    methods = {
        "Random Normal σ=0.01": lambda r, c: random_normal_init(r, c, 0, 0.01),
        "Random Normal σ=0.1":  lambda r, c: random_normal_init(r, c, 0, 0.1),
        "Xavier Uniform":       xavier_uniform_init,
        "Xavier Normal":        xavier_normal_init,
        "He Uniform":           he_uniform_init,
        "He Normal":            he_normal_init,
    }

    print(f"\nАрхитектура: {' → '.join(map(str, architecture))}")
    print(f"Эпох: {epochs}, Learning rate: {lr}, Данных: {n_samples}\n")

    print(f"{'Метод':>25s} | {'Loss (эп. 1)':>14s} | {'Loss (эп. 50)':>14s} "
          "| {'Loss (эп. 200)':>15s} | {'Финал':>10s}")
    print("-" * 90)

    results = {}
    for name, init_fn in methods.items():
        losses = train_network(name, init_fn, X_train, y_train)
        results[name] = losses

        final = losses[-1]
        status = "OK" if final < 0.5 else ("slow" if final < 2.0 else "FAIL")

        print(f"  {name:>23s} | {losses[0]:14.6f} | {losses[49]:14.6f} "
              f"| {losses[-1]:15.6f} | {status:>10s}")

    print("\n" + "-" * 70)
    print("ВЫВОД:")
    print("  - σ=0.01: градиенты слишком малы → обучение практически не идёт")
    print("  - σ=0.1:  лучше, но Xavier/He обычно надёжнее")
    print("  - Xavier: хорош для sigmoid/tanh, работает и с ReLU")
    print("  - He: оптимальна для ReLU-сетей, быстрая сходимость")
    print("  - Правильная инициализация = критический гиперпараметр!")


# ============================================================
# Главная функция
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + " Weight Initialization Methods — From Scratch".center(68) + "║")
    print("║" + " Методы инициализации весов нейронных сетей".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    demo1_initialization_methods()
    demo2_weight_statistics()
    demo3_vanishing_gradients()
    demo4_practical_comparison()

    print("\n\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 70)
    print("""
  ┌─────────────────────┬──────────────────────┬────────────────────────────┐
  │ Метод               │ Формула              │ Когда использовать         │
  ├─────────────────────┼──────────────────────┼────────────────────────────┤
  │ Random Normal       │ N(0, σ²)             │ Простые сети, σ подбирается│
  │ Xavier Uniform      │ U[-√(6/(f+o)), +...] │ Sigmoid, Tanh              │
  │ Xavier Normal       │ N(0, 2/(f+o))        │ Sigmoid, Tanh              │
  │ He Uniform          │ U[-√(6/f), +...]     │ ReLU и варианты            │
  │ He Normal           │ N(0, 2/f)            │ ReLU и варианты            │
  └─────────────────────┴──────────────────────┴────────────────────────────┘

  f = fan_in (входящие связи), o = fan_out (исходящие связи)

  Ключевые правила:
  1. Sigmoid/Tanh → Xavier
  2. ReLU/LeakyReLU/PReLU → He
  3. Слишком малые веса → затухание градиентов
  4. Слишком большие веса → взрыв градиентов
  5. Нулевая инициализация → симметрия (все нейроны учатся одинаково)
""")
