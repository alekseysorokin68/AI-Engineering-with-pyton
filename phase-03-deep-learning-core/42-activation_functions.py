"""
42. Функции активации с нуля на Python
========================================

Содержание:
    1. Сигмоид, Tanh, ReLU, Leaky ReLU, GELU, Swish
    2. Производные всех функций
    3. Сравнение свойств
    4. Проблема затухающих/взрывающихся градиентов

Демонстрации:
    Demo 1: Все функции активации
    Demo 2: Производные функций
    Demo 3: Проблема затухающих градиентов
    Demo 4: Сравнение на практике

Без внешних зависимостей (numpy, torch, sklearn).
"""

import random
import math

random.seed(42)

# ============================================================
# 1. ФУНКЦИИ АКТИВАЦИИ
# ============================================================

def sigmoid(x):
    """Сигмоид: σ(x) = 1 / (1 + e^(-x))"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def sigmoid_derivative(x):
    """Производная сигмоиды: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh_func(x):
    """Гиперболический тангенс: tanh(x)"""
    return math.tanh(x)

def tanh_derivative(x):
    """Производная tanh: tanh'(x) = 1 - tanh²(x)"""
    t = math.tanh(x)
    return 1.0 - t * t

def relu(x):
    """ReLU: max(0, x)"""
    return max(0.0, x)

def relu_derivative(x):
    """Производная ReLU: 1 если x > 0, иначе 0"""
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: max(alpha * x, x)"""
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    """Производная Leaky ReLU: 1 если x > 0, иначе alpha"""
    return 1.0 if x > 0 else alpha

def gelu(x):
    """GELU: x * Φ(x), где Φ —CDF нормального распределения
    Аппроксимация: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))"""
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))

def gelu_derivative(x):
    """Производная GELU (численная аппроксимация)"""
    h = 1e-5
    return (gelu(x + h) - gelu(x - h)) / (2.0 * h)

def swish(x, beta=1.0):
    """Swish: x * σ(βx)"""
    return x * sigmoid(beta * x)

def swish_derivative(x, beta=1.0):
    """Производная Swish (численная аппроксимация)"""
    h = 1e-5
    return (swish(x + h, beta) - swish(x - h, beta)) / (2.0 * h)


# ============================================================
# 2. СПРАВОЧНАЯ ИНФОРМАЦИЯ О СВОЙСТВАХ
# ============================================================

def print_properties():
    """Вывод свойств каждой функции активации."""
    props = [
        ("Sigmoid", "(0, 1)", "0.25", "Нет", "Да", "Выходной слой бинарной классификации"),
        ("Tanh",    "(-1, 1)", "1.0", "Нет", "Да", "Рекуррентные сети, нормализация"),
        ("ReLU",    "[0, ∞)",  "0 или 1", "Да", "Нет", "Скрытые слои CNN/MLP"),
        ("Leaky ReLU", "(-∞, ∞)", "0.01 или 1", "Нет", "Нет", "Избегает mёртвых нейронов"),
        ("GELU",    "(-∞, ∞)", "≈ 0.5 при x=0", "Нет", "Нет", "Transformer (BERT, GPT)"),
        ("Swish",   "(-∞, ∞)", "≈ 0.5 при x=0", "Нет", "Нет", "Эффективнее ReLU в глубоких сетях"),
    ]
    print("  {:<12} {:<14} {:<16} {:<8} {:<8} {}".format(
        "Функция", "Диапазон", "Макс. градиент", "Мёртвые", "Затух.", "Применение"
    ))
    print("  " + "-" * 90)
    for name, rng, grad, dead, vanish, usage in props:
        print("  {:<12} {:<14} {:<16} {:<8} {:<8} {}".format(
            name, rng, grad, dead, vanish, usage
        ))


# ============================================================
# ДЕМОНСТРАЦИИ
# ============================================================

def demo1_all_functions():
    """Demo 1: Все функции активации."""
    print("=" * 65)
    print("DEMO 1: Все функции активации")
    print("=" * 65)
    test_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    print("\n  Значения функций на сетке [-3, 3]:\n")
    print("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "x", "Sigmoid", "Tanh", "ReLU", "LeakyReLU", "GELU", "Swish"
    ))
    print("  " + "-" * 65)

    for x in test_values:
        print("  {:>6.1f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            x,
            sigmoid(x),
            tanh_func(x),
            relu(x),
            leaky_relu(x),
            gelu(x),
            swish(x)
        ))

    print("\n  Ключевые наблюдения:")
    print("  - Sigmoid сжимает вход в диапазон (0, 1)")
    print("  - Tanh центрирован вокруг 0, диапазон (-1, 1)")
    print("  - ReLU простая, но mёртвые нейроны при x < 0")
    print("  - GELU и Swish — гладкие аппроксимации ReLU")
    print()


def demo2_derivatives():
    """Demo 2: Производные функций."""
    print("=" * 65)
    print("DEMO 2: Производные функций активации")
    print("=" * 65)
    test_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    print("\n  Значения производных:\n")
    print("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "x", "Sigmoid'", "Tanh'", "ReLU'", "LeakyReLU'", "GELU'", "Swish'"
    ))
    print("  " + "-" * 65)

    for x in test_values:
        print("  {:>6.1f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            x,
            sigmoid_derivative(x),
            tanh_derivative(x),
            relu_derivative(x),
            leaky_relu_derivative(x),
            gelu_derivative(x),
            swish_derivative(x)
        ))

    print("\n  Анализ производных:")
    print("  - Sigmoid': максимум = 0.25 при x=0 → градиенты затухают")
    print("  - Tanh':   максимум = 1.0 при x=0 → лучше сигмоиды")
    print("  - ReLU':   0 при x<0, 1 при x>0 → нет затухания, но mёртвые нейроны")
    print("  - Leaky ReLU': всегда > 0 → нет mёртвых нейронов")
    print("  - GELU' и Swish': гладкие, ненулевые производные")
    print()


def demo3_vanishing_gradients():
    """Demo 3: Проблема затухающих градиентов."""
    print("=" * 65)
    print("DEMO 3: Проблема затухающих/взрывающихся градиентов")
    print("=" * 65)

    # Симуляция прохождения градиента через 10 слоёв
    num_layers = 10
    initial_gradient = 1.0

    print(f"\n  Симуляция: начальный градиент = {initial_gradient}")
    print(f"  Количество слоёв: {num_layers}\n")

    # Сигмоид
    grad = initial_gradient
    sig_chain = [grad]
    for _ in range(num_layers):
        # Максимальная производная сигмоиды = 0.25
        grad *= 0.25
        sig_chain.append(grad)
    print("  Sigmoid (макс. градиент слоя = 0.25):")
    print("  Слои:  {}".format(" -> ".join(f"{g:.6f}" for g in sig_chain)))
    print("  Итого: градиент уменьшился в {:.0f} раз\n".format(
        initial_gradient / max(sig_chain[-1], 1e-30)
    ))

    # Tanh
    grad = initial_gradient
    tanh_chain = [grad]
    for _ in range(num_layers):
        grad *= 1.0  # максимальная производная tanh = 1.0
        tanh_chain.append(grad)
    print("  Tanh (макс. градиент слоя = 1.0):")
    print("  Слои:  {}".format(" -> ".join(f"{g:.6f}" for g in tanh_chain)))
    print("  Итого: градиент НЕ затухает (макс. градиент = 1)\n")

    # ReLU
    grad = initial_gradient
    relu_chain = [grad]
    for _ in range(num_layers):
        grad *= 1.0  # производная ReLU = 1 при x > 0
        relu_chain.append(grad)
    print("  ReLU (градиент слоя = 1 при x > 0):")
    print("  Слои:  {}".format(" -> ".join(f"{g:.6f}" for g in relu_chain)))
    print("  Итого: градиент НЕ затухает при положительных входах\n")

    # Пример затухания через 20 слоёв с сигмоидой
    deep_grad = initial_gradient
    deep_layers = 20
    for _ in range(deep_layers):
        deep_grad *= 0.25

    print("  Глубокая сеть (20 слоёв, Sigmoid):")
    print(f"  Начальный градиент: {initial_gradient}")
    print(f"  Градиент на входе:  {deep_grad:.2e}")
    print("  → Градиент практически исчез (проблема затухания)\n")

    # Пример взрывающихся градиентов
    print("  Пример взрывающихся градиентов (большие веса):")
    exploding_grad = initial_gradient
    big_weight = 1.5
    for i in range(10):
        exploding_grad *= big_weight
    print(f"  Вес = {big_weight}, 10 слоёв: {exploding_grad:.2e}")
    print("  → Градиент экспоненциально растёт (проблема взрыва)\n")


def demo4_practical_comparison():
    """Demo 4: Сравнение на практике."""
    print("=" * 65)
    print("DEMO 4: Сравнение на практике")
    print("=" * 65)

    # Симуляция простой нейронной сети: forward pass + backward pass
    random.seed(42)
    input_size = 4
    hidden_size = 3
    output_size = 1

    # Инициализация весов
    W1 = [[random.gauss(0, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
    b1 = [0.0] * hidden_size
    W2 = [[random.gauss(0, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
    b2 = [0.0] * output_size

    x = [1.0, 0.5, -0.3, 0.8]  # вход

    print(f"\n  Вход: {x}")
    print(f"  W1 (4x3): {[round(w, 3) for row in W1 for w in row]}")
    print(f"  W2 (3x1): {[round(w, 3) for row in W2 for w in row]}")

    activations = {
        "Sigmoid": (sigmoid, sigmoid_derivative),
        "Tanh":    (tanh_func, tanh_derivative),
        "ReLU":    (relu, relu_derivative),
    }

    print("\n  Forward pass + Backward pass:\n")

    for name, (act_fn, act_deriv) in activations.items():
        # Forward pass
        z1 = []
        for j in range(hidden_size):
            s = b1[j]
            for i in range(input_size):
                s += x[i] * W1[i][j]
            z1.append(s)

        a1 = [act_fn(z) for z in z1]

        z2 = []
        for k in range(output_size):
            s = b2[k]
            for j in range(hidden_size):
                s += a1[j] * W2[j][k]
            z2.append(s)

        y_pred = z2[0]  # линейный выход

        # Backward pass (MSE loss: L = 0.5 * (y_pred - target)^2)
        target = 0.7
        dL_dy = y_pred - target

        # Градиенты по W2
        dW2 = [[dL_dy * a1[j] for k in range(output_size)] for j in range(hidden_size)]

        # Градиенты по скрытому слою
        da1 = [dL_dy * W2[j][0] for j in range(hidden_size)]
        dz1 = [da1[j] * act_deriv(z1[j]) for j in range(hidden_size)]

        # Градиенты по W1
        dW1 = [[dz1[j] * x[i] for j in range(hidden_size)] for i in range(input_size)]

        grad_norm_sq = sum(dW1[i][j] ** 2 for i in range(input_size) for j in range(hidden_size))

        print(f"  {name}:")
        print(f"    Выход:     {y_pred:.4f}")
        print(f"    Затухание: max|dW1| = {max(abs(dW1[i][j]) for i in range(input_size) for j in range(hidden_size)):.4f}")
        print(f"    L2 градиента: {math.sqrt(grad_norm_sq):.4f}")
        print()

    # Сравнение: цепочка из 5 слоёв
    print("  Цепочка из 5 слоёв (накопление градиентов):\n")
    for name, (act_fn, act_deriv) in activations.items():
        grad = 1.0
        for _ in range(5):
            # Типичное значение производной
            if name == "Sigmoid":
                grad *= 0.2  # средняя производная
            elif name == "Tanh":
                grad *= 0.6
            elif name == "ReLU":
                grad *= 1.0  # при x > 0
        print(f"    {name:<10}: градиент после 5 слоёв = {grad:.4f}")

    print("\n  Вывод:")
    print("  - ReLU лучше сохраняет градиент (но mёртвые нейроны)")
    print("  - Tanh лучше сигмоиды, но всё ещё затухает")
    print("  - Сигмоид хуже всех для глубоких сетей")
    print("  - Leaky ReLU / GELU / Swish — компромиссы")
    print()


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ФУНКЦИИ АКТИВАЦИИ — ОСНОВЫ С НУЛЯ")
    print("=" * 65)

    print("\n  Справочная таблица свойств:")
    print()
    print_properties()

    print()
    demo1_all_functions()
    demo2_derivatives()
    demo3_vanishing_gradients()
    demo4_practical_comparison()

    print("=" * 65)
    print("  Все демонстрации завершены!")
    print("=" * 65)
