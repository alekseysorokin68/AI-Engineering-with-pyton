"""
56 — Классические архитектуры CNN
==================================
LeNet-5 · Residual Connections · Сравнение глубоких и мелких сетей

Самодостаточный скрипт — не требует numpy / torch / PIL / cv2.
"""

import random
import math

random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# Вспомогательные утилиты
# ═══════════════════════════════════════════════════════════════════════

def conv_output_size(size, kernel, stride=1, padding=0):
    """Размер выходной feature map после свёртки."""
    return (size - kernel + 2 * padding) // stride + 1


def pool_output_size(size, kernel, stride=None):
    """Размер после пулинга."""
    if stride is None:
        stride = kernel
    return (size - kernel) // stride + 1


def count_params(layers):
    """Подсчёт параметров по списку слоёв-словарей."""
    total = 0
    for layer in layers:
        total += layer.get("params", 0)
    return total


def relu(x):
    """ReLU activation."""
    return max(0.0, x)


def sigmoid(x):
    """Sigmoid activation."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def softmax(values):
    """Softmax по списку."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def xavier_init(fan_in, fan_out):
    """Инициализация Xavier/Glorot (uniform-аппроксимация)."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [random.uniform(-limit, limit) for _ in range(fan_in * fan_out)]


def he_init(fan_in):
    """Инициализация He (для ReLU)."""
    std = math.sqrt(2.0 / fan_in)
    return [random.gauss(0, std) for _ in range(fan_in)]


# ═══════════════════════════════════════════════════════════════════════
# Демо 1: LeNet-5 — структура и параметры
# ═══════════════════════════════════════════════════════════════════════

def demo_lenet5():
    print("=" * 70)
    print("ДЕМО 1: LeNet-5 — СТРУКТУРА И ПАРАМЕТРЫ")
    print("=" * 70)

    print("""
Архитектура LeNet-5 (LeCun et al., 1998)
─────────────────────────────────────────
Оригинальная свёрточная нейросеть для распознавания рукописных цифр.
Разработка для банковских чеков (US Postal Service).

Вход: 32×32×1 (grayscale)
""")

    # Модельируем слои LeNet-5
    h, w, c = 32, 32, 1
    layers = []

    print(f"{'Слой':<25} {'Выход':>12} {'Параметры':>12}  Описание")
    print("─" * 80)

    # C1: Conv 5×5, 6 фильтров
    h = conv_output_size(h, 5)
    w = conv_output_size(w, 5)
    c_out = 6
    params = 5 * 5 * 1 * 6 + 6  # веса + смещения
    layers.append({"name": "C1", "params": params, "out": f"{h}×{w}×{c_out}"})
    print(f"{'C1: Conv 5×5, 6filt':<25} {h:>3}×{w:>3}×{c_out:<4} {params:>10,}  tanh activation")
    c = c_out

    # S2: AvgPool 2×2
    h = pool_output_size(h, 2, 2)
    w = pool_output_size(w, 2, 2)
    layers.append({"name": "S2", "params": 0, "out": f"{h}×{w}×{c}"})
    print(f"{'S2: AvgPool 2×2':<25} {h:>3}×{w:>3}×{c:<4} {'0':>10}  subsampling")

    # C3: Conv 5×5, 16 фильтров
    h = conv_output_size(h, 5)
    w = conv_output_size(w, 5)
    c_out = 16
    # В оригинале C3 имеет sparse connections (10 из 60 возможных)
    params = 5 * 5 * 6 * 16 + 16  # упрощённо: полные связи
    layers.append({"name": "C3", "params": params, "out": f"{h}×{w}×{c_out}"})
    print(f"{'C3: Conv 5×5, 16filt':<25} {h:>3}×{w:>3}×{c_out:<4} {params:>10,}  tanh activation")
    c = c_out

    # S4: AvgPool 2×2
    h = pool_output_size(h, 2, 2)
    w = pool_output_size(w, 2, 2)
    layers.append({"name": "S4", "params": 0, "out": f"{h}×{w}×{c}"})
    print(f"{'S4: AvgPool 2×2':<25} {h:>3}×{w:>3}×{c:<4} {'0':>10}  subsampling")

    # C5: Conv 5×5, 120 фильтров (на входе 5×5×16 = flat → по сути FC)
    h = conv_output_size(h, 5)
    w = conv_output_size(w, 5)
    c_out = 120
    params = 5 * 5 * 16 * 120 + 120
    layers.append({"name": "C5", "params": params, "out": f"{h}×{w}×{c_out}"})
    print(f"{'C5: Conv 5×5, 120filt':<25} {h:>3}×{w:>3}×{c_out:<4} {params:>10,}  tanh activation")
    c = c_out

    # F6: Fully Connected, 84 units
    fc_in = h * w * c_out  # 1×1×120 = 120
    fc_out = 84
    params_f6 = fc_in * fc_out + fc_out
    layers.append({"name": "F6", "params": params_f6, "out": f"84"})
    print(f"{'F6: FC 84 units':<25} {'84':>12} {params_f6:>10,}  tanh activation")

    # Output: FC 10 (цифры 0-9)
    params_out = 84 * 10 + 10
    layers.append({"name": "Output", "params": params_out, "out": "10"})
    print(f"{'Output: FC 10':<25} {'10':>12} {params_out:>10,}  RBF / softmax")

    total = count_params(layers)
    print("─" * 80)
    print(f"{'ИТОГО параметров:':<25} {'':>12} {total:>10,}")
    print(f"\n≈ {total:,} параметров — это < 100K! Современные сети: миллиарды.\n")

    print("Ключевые особенности LeNet-5:")
    print("  • Первый successful CNN — доказал концепцию")
    print("  • Субsamplying (avg pool) вместоsubsampling с обучаемыми коэффициентами")
    print("  • Sparse connections в C3 — для.breaking symmetries и уменьшения параметров")
    print("  • Tanh вместо sigmoid — более быстрая сходимость")
    print("  • Паттерн: Conv → Pool → Conv → Pool → FC → FC")
    print()

    # Показать вычисление одного размера
    print("Пример вычисления размера feature map:")
    print("  Вход: 32×32")
    print("  C1: (32 - 5 + 2×0) / 1 + 1 = 28  →  28×28")
    print("  S2: (28 - 2) / 2 + 1 = 14        →  14×14")
    print("  C3: (14 - 5 + 2×0) / 1 + 1 = 10  →  10×10")
    print("  S4: (10 - 2) / 2 + 1 = 5          →  5×5")
    print("  C5: (5 - 5 + 2×0) / 1 + 1 = 1    →  1×1")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Демо 2: Проблема затухающих градиентов
# ═══════════════════════════════════════════════════════════════════════

def demo_vanishing_gradients():
    print("=" * 70)
    print("ДЕМО 2: ПРОБЛЕМА ЗАТУХАЮЩИХ ГРАДИЕНТОВ (ГЛУБОКИЕ СЕТИ)")
    print("=" * 70)

    print("""
Проблема: при обратном проходе градиенты умножаются на веса
на каждом слое. Если веса < 1, градиент экспоненциально затухает.
""")

    # Моделируем затухание градиентов через слои
    print("  Симуляция затухания градиента в 20-слойной сети:")
    print("  (каждый слой умножает градиент на случайный вес < 1)")
    print()

    random.seed(42)
    n_layers = 20
    gradient = 1.0
    gradients = [gradient]

    print(f"  {'Слой':<8} {'Градиент':>14} {'От исходного':>14}  Визуализация")
    print("  " + "─" * 65)

    for i in range(n_layers):
        # Веса инициализированы Xavier → средний модуль < 1
        weight = random.uniform(0.3, 0.9)
        gradient *= weight
        gradients.append(gradient)
        bar_len = int(gradient * 50) if gradient > 0.001 else 0
        bar = "█" * bar_len
        pct = gradient * 100
        if gradient < 0.001:
            print(f"  {i+1:<8} {gradient:>14.2e} {pct:>13.4f}%  «grad ≈ 0»")
        else:
            print(f"  {i+1:<8} {gradient:>14.2e} {pct:>13.4f}%  {bar}")

    print()
    print(f"  Итого: градиент уменьшился в {1/gradients[-1]:,.0f} раз за {n_layers} слоёв!")
    print()

    # Тангенсная проблема
    print("  Почему sigmoid/tanh усугубляют проблему:")
    print()
    print("  sigmoid'(x) = sigmoid(x) · (1 - sigmoid(x))  ≤  0.25")
    print("  tanh'(x)    = 1 - tanh²(x)                    ≤  1.0")
    print()

    print("  Макс. производительная sigmoid = 0.25 при x = 0:")
    max_sig_deriv = 0.25
    sig_gradient = 1.0
    for i in range(10):
        sig_gradient *= max_sig_deriv
    print(f"  После 10 слоёв: 0.25^10 = {max_sig_deriv**10:.2e}")
    print()

    # Число обусловленности
    print("  Число обусловленности (condition number):")
    print("  • Условное число = max eigenvalue / min eigenvalue")
    print("  • Высокое число → градиенты «застревают» или «взрываются»")
    print("  • AlexNet (2012): condition number ≈ 130,000")
    print()

    print("  Решения до ResNet:")
    print("  • Batch Normalization (Ioffe & Szegedy, 2015)")
    print("  • carefully tuned initialization (He, Glorot)")
    print("  • ReLU вместо sigmoid/tanh")
    print("  • LSTM/GRU для рекуррентных сетей")
    print("  • Но ни одно не позволялообучать 100+ слойные сети!")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Демо 3: Residual Connections — решение
# ═══════════════════════════════════════════════════════════════════════

def demo_residual_connections():
    print("=" * 70)
    print("ДЕМО 3: RESIDUAL CONNECTIONS — РЕШЕНИЕ")
    print("=" * 70)

    print("""
ResNet (He et al., 2015) — «Deep Residual Learning»

Ключевая идея:
  Обычная сеть:      y = F(x)          →  сеть учит F(x)
  Residual сеть:     y = F(x) + x      →  сеть учит F(x) = y - x

Skip connection (shortcut) «обходит» слой и добавляет вход к выходу.
""")

    # Сравнение forward pass
    print("  Forward pass — обычный vs residual блок:")
    print()
    print("  Обычный блок:           Residual блок:")
    print("  ┌───────────┐           ┌───────────┐")
    print("  │    Conv    │           │    Conv    │")
    print("  ├───────────┤           ├───────────┤")
    print("  │    BN      │           │    BN      │")
    print("  ├───────────┤           ├───────────┤")
    print("  │   ReLU     │           │   ReLU     │")
    print("  ├───────────┤           ├───────────┤")
    print("  │    Conv    │           │    Conv    │")
    print("  ├───────────┤           ├───────────┤")
    print("  │    BN      │           │    BN      │")
    print("  └─────┬─────┘           └─────┬─────┘")
    print("        │                       │")
    print("        ↓                 ┌─────┴─────┐")
    print("  ┌───────────┐           │  + (skip)  │ ←── x (input)")
    print("  │   ReLU     │           ├───────────┤")
    print("  └─────┬─────┘           │   ReLU     │")
    print("        ↓                 └─────┬─────┘")
    print("    output = F(x)               ↓")
    print("                          output = F(x) + x")
    print()

    # Симуляция градиентного потока
    print("  Градиентный поток через residual connections:")
    print()
    print("  ∂L/∂x = ∂L/∂y · ∂y/∂x")
    print("  y = F(x) + x  →  ∂y/∂x = ∂F(x)/∂x + I")
    print()
    print("  ∂L/∂x = ∂L/∂y · (∂F(x)/∂x + 1)")
    print()
    print("  Термин «+1» ГАРАНТИРУЕТ, что градиент не затухает!")
    print("  Даже если ∂F(x)/∂x ≈ 0, градиент = 1 → проходит через.")
    print()

    # Эксперимент: сравнение затухания
    print("  Эксперимент: затухание градиента на 15 слоях")
    print()

    random.seed(42)

    # Обычная сеть
    gradient_plain = 1.0
    plain_history = [gradient_plain]
    for i in range(15):
        w = random.uniform(0.3, 0.9)
        gradient_plain *= w
        plain_history.append(gradient_plain)

    # Residual сеть
    random.seed(42)
    gradient_res = 1.0
    res_history = [gradient_res]
    for i in range(15):
        w = random.uniform(0.3, 0.9)
        # skip connection: gradient += 1 (identity shortcut)
        gradient_res = gradient_res * w + gradient_res * 1.0
        res_history.append(gradient_res)

    print(f"  {'Слой':<6} {'Plain':>12} {'ResNet':>12}  Plain    ResNet")
    print("  " + "─" * 55)
    for i in range(16):
        p = plain_history[i]
        r = res_history[i]
        p_bar = "▓" * int(min(p * 10, 30))
        r_bar = "░" * int(min(r * 0.5, 30))
        if i == 0:
            print(f"  {i:<6} {p:>12.4f} {r:>12.4f}  {p_bar:<15} {r_bar}")
        else:
            print(f"  {i:<6} {p:>12.4f} {r:>12.4f}  {p_bar:<15} {r_bar}")

    print()
    ratio = plain_history[-1] / res_history[-1] if plain_history[-1] > 0 else float('inf')
    print(f"  Plain:  {plain_history[-1]:.6f} (затух в {1/plain_history[-1]:,.0f}×)")
    print(f"  ResNet: {res_history[-1]:.4f}  (gradient НЕ затухает!)")
    print()

    print("  Почему F(x) + x работает лучше, чем F(x)?")
    print()
    print("  Если оптимальная функция ближе к identity (h(x) = x),")
    print("  то проще обучить F(x) = h(x) - x ≈ 0, чем F(x) ≈ x.")
    print()
    print("  Нулевые веса легче найти, чем единичные → ResNet проще оптимизировать.")
    print()

    print("  Влияние на качество (Top-1 Error на ImageNet):")
    print("  ┌──────────────────────────────┬──────────┬──────────┐")
    print("  │ Модель                       │ Слоёв    │ Top-1    │")
    print("  ├──────────────────────────────┼──────────┼──────────┤")
    print("  │ VGG-19 (2014)                │    19    │  28.0%   │")
    print("  │ GoogLeNet (2014)             │    22    │  26.7%   │")
    print("  │ Plain-18 (2015)              │    18    │  27.9%   │")
    print("  │ ResNet-18 (2015)             │    18    │  26.7%   │")
    print("  │ Plain-34 (2015)              │    34    │  28.5%   │")
    print("  │ ResNet-34 (2015)             │    34    │  25.0%   │")
    print("  │ ResNet-50 (2015)             │    50    │  23.8%   │")
    print("  │ ResNet-101 (2015)            │   101    │  22.4%   │")
    print("  │ ResNet-152 (2015)            │   152    │  21.7%   │")
    print("  └──────────────────────────────┴──────────┴──────────┘")
    print()
    print("  Вывод: глубокая residual сеть ЛУЧШЕ мелкой,")
    print("  а мелкая residual сеть ЛУЧШЕ глубокой обычной.")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Демо 4: Сравнение архитектур
# ═══════════════════════════════════════════════════════════════════════

def demo_architecture_comparison():
    print("=" * 70)
    print("ДЕМО 4: СРАВНЕНИЕ АРХИТЕКТУР")
    print("=" * 70)

    print("""
Место каждой архитектуры в истории компьютерного зрения:
""")

    architectures = [
        {
            "name": "LeNet-5",
            "year": 1998,
            "layers": 5,
            "params": "60K",
            "top1": "N/A (MNIST 0.8%)",
            "innovations": "Свёртки, субсэмплинг, FC",
            "input_size": "32×32×1",
        },
        {
            "name": "AlexNet",
            "year": 2012,
            "layers": 8,
            "params": "61M",
            "top1": "16.4%",
            "innovations": "ReLU, Dropout, Data Augm, GPU training",
            "input_size": "227×227×3",
        },
        {
            "name": "VGG-16",
            "year": 2014,
            "layers": 16,
            "params": "138M",
            "top1": "26.4%",
            "innovations": "Small 3×3 filters, deeper = better",
            "input_size": "224×224×3",
        },
        {
            "name": "GoogLeNet",
            "year": 2014,
            "layers": 22,
            "params": "6.8M",
            "top1": "26.7%",
            "innovations": "Inception modules, 1×1 conv bottleneck",
            "input_size": "224×224×3",
        },
        {
            "name": "ResNet-50",
            "year": 2015,
            "layers": 50,
            "params": "25.6M",
            "top1": "23.8%",
            "innovations": "Skip connections, batch norm, residual blocks",
            "input_size": "224×224×3",
        },
        {
            "name": "ResNet-152",
            "year": 2015,
            "layers": 152,
            "params": "60.2M",
            "top1": "21.7%",
            "innovations": "Deepest practical residual network",
            "input_size": "224×224×3",
        },
    ]

    # Таблица сравнения
    print(f"  {'Архитектура':<14} {'Год':<6} {'Слоёв':>6} {'Параметров':>12} {'Top-1':>10}  Ключевые инновации")
    print("  " + "─" * 90)
    for a in architectures:
        print(f"  {a['name']:<14} {a['year']:<6} {a['layers']:>6} {a['params']:>12} {a['top1']:>10}  {a['innovations']}")
    print()

    # Эволюция параметров
    print("  Эволюция (глубина vs параметры vs качество):")
    print()

    data_points = [
        ("LeNet-5 (1998)", 5, 60000),
        ("AlexNet (2012)", 8, 61000000),
        ("VGG-16 (2014)", 16, 138000000),
        ("GoogLeNet (2014)", 22, 6800000),
        ("ResNet-50 (2015)", 50, 25600000),
        ("ResNet-152 (2015)", 152, 60200000),
    ]

    print(f"  {'Модель':<22} {'Слоёв':>6} {'Параметры':>14}  Относит. размер")
    print("  " + "─" * 60)
    for name, depth, params in data_points:
        bar_len = int(math.log10(params) * 2) - 4
        bar = "█" * max(bar_len, 1)
        print(f"  {name:<22} {depth:>6} {params:>14,}  {bar}")

    print()
    print("  Ключевые выводы:")
    print()
    print("  1. БОЛЬШЕ слоёв ≠ БОЛЬШЕ параметров")
    print("     GoogLeNet (22 слоя, 6.8M) < VGG-16 (16 слоёв, 138M)")
    print()
    print("  2. Efficient architectures matter:")
    print("     1×1 convolutions как bottleneck сокращают вычисления")
    print()
    print("  3. ResNet открыл эру сверхглубоких сетей:")
    print("     50 → 152 → 1000+ слоёв возможно")
    print()
    print("  4. Паттерн эволюции:")
    print("     LeNet → AlexNet → VGG → GoogLeNet → ResNet")
    print("     (простые свёртки) → (ReLU, GPU) → (глубина) → (эффективность) → (skip connections)")
    print()

    # Морфология сетей
    print("  Морфология: от цепочки к DAG (Directed Acyclic Graph):")
    print()
    print("  LeNet/VGG:        GoogLeNet:        ResNet:")
    print("  ┌───┐             ┌───┐             ┌───┐")
    print("  │Conv│             │Inception│         │Block│──┐")
    print("  └─┬─┘             └───┬───┘         └─┬─┘  │")
    print("  ┌─┴─┐             ┌───┴───┐           │    │")
    print("  │Pool│             │Inception│          │  ┌─┴─┐")
    print("  └─┬─┘             └───┬───┘           │  │ + │")
    print("  ┌─┴─┐             ┌───┴───┐           │  └─┬─┘")
    print("  │Conv│             │Inception│          │  ┌─┴─┐")
    print("  └─┬─┘             └───┬───┘           │  │ReLU│")
    print("  ...                 ...              └──┴─┬─┘")
    print("  Linear chain      Parallel branches    Residual loop")
    print()
    print("  Residual connection = «highway» для градиентов:")
    print(" .gradient проходит через «мост» → не затухает → глубокие сети работают!")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Запуск
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          КЛАССИЧЕСКИЕ АРХИТЕКТУРЫ CNN — ОБУЧАЮЩЕЕ ДЕМО            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("  Темы:")
    print("  • LeNet-5 — первая успешная CNN (1998)")
    print("  • Проблема затухающих градиентов")
    print("  • Residual connections — решение (ResNet, 2015)")
    print("  • Сравнение архитектур: LeNet → AlexNet → VGG → ResNet")
    print()

    demo_lenet5()
    print()
    demo_vanishing_gradients()
    print()
    demo_residual_connections()
    print()
    demo_architecture_comparison()

    print("=" * 70)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("=" * 70)
    print("""
1. LeNet-5 (1998) — заложила основу CNN:
   Conv → Pool → Conv → Pool → FC → FC

2. Проблема глубоких сетей:
   • Затухающие градиенты: g ∝ ∏ w_i → 0
   • Сигмоид/тангенс усиливают проблему (∂σ/∂x ≤ 0.25)

3. ResNet (2015) — breakthrough:
   • Skip connections: y = F(x) + x
   • Градиент: ∂y/∂x = ∂F/∂x + 1 (гарантированно > 0)
   • Позволилообучать 152-слойные (и глубже) сети

4. Эволюция архитектур:
   • От цепочки к DAG (GoogLeNet, ResNet)
   • Эффективность > количество (GoogLeNet vs VGG)
   • Skip connections стали стандартом (DenseNet, U-Net, Transformer)
""")
