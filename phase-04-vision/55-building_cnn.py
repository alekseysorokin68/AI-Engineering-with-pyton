"""
55 — Построение CNN с нуля на Python
====================================
Свёрточная нейросеть без NumPy / PyTorch — только lists + math.

Классы:
  ConvLayer     — свёрточный слой (multi-channel input → multi-channel output)
  PoolingLayer  — max-pooling 2×2
  CNN           — полная архитектура: Conv → ReLU → Pool → Flatten → FC → Softmax
"""

import random
import math

random.seed(42)

# ─────────────────────────── вспомогательные функции ───────────────────────────

def relu(x):
    return x if x > 0 else 0.0


def softmax(vec):
    mx = max(vec)
    exps = [math.exp(v - mx) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


# ─────────────────────── свёрточный слой (ConvLayer) ───────────────────────────

class ConvLayer:
    """Многоканальная свёртка 2D.

    Параметры
    ---------
    in_channels  : int — количество входных каналов (например 1 или 3)
    out_channels : int — количество фильтров (ядро)
    kernel_size  : int — размер квадратного ядра (по умолчанию 3)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # K[out][in][ky][kx]
        self.K = [
            [
                [[random.uniform(-0.5, 0.5) for _ in range(kernel_size)]
                 for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias = [0.0 for _ in range(out_channels)]

    def forward(self, inputs):
        """
        inputs — список из in_channels матриц (list of list of list).
        Возвращает out_channels матриц того же spatial-размера (same-padding).
        """
        C_in = self.in_channels
        H = len(inputs[0])
        W = len(inputs[0][0])
        K = self.kernel_size
        pad = K // 2  # same-padding

        # паддинг по краям
        padded = []
        for c in range(C_in):
            p = []
            for y in range(-pad, H + pad):
                row = []
                for x in range(-pad, W + pad):
                    if 0 <= y < H and 0 <= x < W:
                        row.append(inputs[c][y][x])
                    else:
                        row.append(0.0)
                p.append(row)
            padded.append(p)

        outputs = []
        for oc in range(self.out_channels):
            out_mat = []
            for y in range(H):
                row = []
                for x in range(W):
                    acc = 0.0
                    for ic in range(C_in):
                        for ky in range(K):
                            for kx in range(K):
                                acc += self.K[oc][ic][ky][kx] * padded[ic][y + ky][x + kx]
                    row.append(acc + self.bias[oc])
                out_mat.append(row)
            outputs.append(out_mat)
        return outputs

    def param_count(self):
        return self.out_channels * self.in_channels * self.kernel_size ** 2 + self.out_channels


# ───────────────────── pooling-слой (2×2 max) ────────────────────────────────

class PoolingLayer:
    """Max-pooling 2×2, stride 2."""

    def forward(self, inputs):
        """inputs — list of 2D-матриц (каналов)."""
        outputs = []
        for mat in inputs:
            H = len(mat)
            W = len(mat[0])
            out_h = H // 2
            out_w = W // 2
            pooled = []
            for y in range(out_h):
                row = []
                for x in range(out_w):
                    block = [
                        mat[y * 2][x * 2],
                        mat[y * 2][x * 2 + 1] if x * 2 + 1 < W else mat[y * 2][x * 2],
                        mat[y * 2 + 1][x * 2] if y * 2 + 1 < H else mat[y * 2][x * 2],
                        mat[y * 2 + 1][x * 2 + 1] if (y * 2 + 1 < H and x * 2 + 1 < W) else mat[y * 2][x * 2],
                    ]
                    row.append(max(block))
                pooled.append(row)
            outputs.append(pooled)
        return outputs

    def param_count(self):
        return 0


# ──────────────────────── полная CNN ──────────────────────────────────────────

class CNN:
    """Двухблочная свёрточная сеть:
       Conv1(1→8,3×3) → ReLU → Pool →
       Conv2(8→16,3×3) → ReLU → Pool →
       Flatten → FC → Softmax

    Параметры
    ---------
    in_channels  : int — входные каналы изображения
    num_classes  : int — количество классов
    h, w         : int — размер входного изображения (H × W)
    """

    def __init__(self, in_channels=1, num_classes=10, h=8, w=8):
        self.conv1 = ConvLayer(in_channels, out_channels=8, kernel_size=3)
        self.conv2 = ConvLayer(8, out_channels=16, kernel_size=3)
        self.pool = PoolingLayer()
        self.h = h
        self.w = w
        self.num_classes = num_classes

        # после двух conv (same) + pool (÷2) каждый → spatial уменьшается в 2 раза
        self.feat_h = h // 4
        self.feat_w = w // 4
        flat_size = 16 * self.feat_h * self.feat_w

        # FC
        self.W_fc = [[random.uniform(-0.3, 0.3) for _ in range(flat_size)]
                     for _ in range(num_classes)]
        self.b_fc = [0.0 for _ in range(num_classes)]

    def forward(self, x):
        """x — входная матрица (single-channel) или список каналов."""
        if isinstance(x[0][0], (int, float)):
            x = [x]  # один канал

        # Блок 1: Conv1 → ReLU → Pool
        c1 = self.conv1.forward(x)
        r1 = [[[relu(v) for v in row] for row in mat] for mat in c1]
        p1 = self.pool.forward(r1)

        # Блок 2: Conv2 → ReLU → Pool
        c2 = self.conv2.forward(p1)
        r2 = [[[relu(v) for v in row] for row in mat] for mat in c2]
        p2 = self.pool.forward(r2)

        # Flatten
        flat = []
        for mat in p2:
            for row in mat:
                flat.extend(row)

        # FC
        logits = []
        for i in range(self.num_classes):
            acc = self.b_fc[i]
            for j in range(len(flat)):
                acc += self.W_fc[i][j] * flat[j]
            logits.append(acc)

        probs = softmax(logits)
        return probs

    def predict(self, x):
        probs = self.forward(x)
        return probs.index(max(probs))

    def param_count(self):
        return (self.conv1.param_count() + self.conv2.param_count() +
                self.pool.param_count() +
                len(self.W_fc) * len(self.W_fc[0]) + len(self.b_fc))


# ════════════════════════════════ ДЕМО ═══════════════════════════════════════

def make_image(h, w, pattern="checker"):
    """Создаёт простое изображение."""
    if pattern == "checker":
        return [[1.0 if (i + j) % 2 == 0 else 0.0 for j in range(w)] for i in range(h)]
    elif pattern == "gradient":
        return [[i / h + j / w for j in range(w)] for i in range(h)]
    elif pattern == "random":
        return [[random.random() for _ in range(w)] for _ in range(h)]
    return [[0.0] * w for _ in range(h)]


def print_matrix(mat, title=""):
    if title:
        print(f"  {title}:")
    for row in mat:
        print("    [" + " ".join(f"{v:7.4f}" for v in row) + "]")


# ─── Демо 1: ConvLayer — forward pass ────────────────────────────────────────

print("=" * 70)
print("Демо 1: ConvLayer — свёрточный слой")
print("=" * 70)

img = make_image(8, 8, "checker")
print("Входное изображение 8×8 (checker):")
print_matrix(img)

conv = ConvLayer(in_channels=1, out_channels=3, kernel_size=3)
out = conv.forward([img])

for ch in range(len(out)):
    print(f"\nВыходной канал {ch}:")
    print_matrix(out[ch])

print(f"\nПараметров ConvLayer: {conv.param_count()}")
print()


# ─── Демо 2: CNN — forward pass ──────────────────────────────────────────────

print("=" * 70)
print("Демо 2: CNN — полный forward pass (2 conv-блока)")
print("=" * 70)

cnn = CNN(in_channels=1, num_classes=4, h=8, w=8)

img1 = make_image(8, 8, "checker")
img2 = make_image(8, 8, "gradient")
img3 = make_image(8, 8, "random")

for idx, (image, label) in enumerate([(img1, "checker"), (img2, "gradient"), (img3, "random")]):
    probs = cnn.forward(image)
    pred = probs.index(max(probs))
    print(f"\nИзображение: {label}")
    print(f"  Вероятности: [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  Предсказанный класс: {pred}")

print()


# ─── Демо 3: Сравнение с fully connected ─────────────────────────────────────

print("=" * 70)
print("Демо 3: CNN vs Fully Connected — сравнение")
print("=" * 70)

H, W = 32, 32
in_ch = 1
num_cls = 10
hidden_fc = 256  # скрытый слой полносвязной сети

# CNN (2 conv-блока)
cnn_model = CNN(in_channels=in_ch, num_classes=num_cls, h=H, w=W)
cnn_params = cnn_model.param_count()

# Fully Connected (1 скрытый слой 256 нейронов)
fc_input = in_ch * H * W
fc_hidden_params = fc_input * hidden_fc + hidden_fc       # Input → Hidden
fc_output_params = hidden_fc * num_cls + num_cls          # Hidden → Output
fc_params = fc_hidden_params + fc_output_params

print(f"  Размер входа: {in_ch} × {H} × {W} = {fc_input} пикселей")
print(f"  Выходных классов: {num_cls}")
print()
print(f"  CNN (2 conv-блока + FC):    {cnn_params:>10} параметров")
print(f"  FC (скрытый слой {hidden_fc}):     {fc_params:>10} параметров")
print(f"  Коэффициент сжатия:         {fc_params / cnn_params:>8.1f}x")
print()
print("  CNN эффективнее за счёт:")
print("    • Weight sharing — одно ядро 3×3 для всего изображения")
print("    • Sparse connections — каждый нейрон видит только 3×3 окно")
print("    • Pooling — уменьшает пространственную размерность")


# ─── Демо 4: Количество параметров при разных архитектурах ───────────────────

print()
print("=" * 70)
print("Демо 4: Количество параметров при разных архитектурах")
print("=" * 70)

# Вход 32×32, 1 канал
architectures = [
    ("2 conv → FC(10)",      CNN(in_channels=1, num_classes=10, h=32, w=32)),
    ("2 conv → FC(100)",     CNN(in_channels=1, num_classes=100, h=32, w=32)),
    ("2 conv → FC(256)",     CNN(in_channels=1, num_classes=256, h=32, w=32)),
]

fc_architectures = [
    ("FC(256) → FC(10)",     32 * 32 * 256 + 256 + 256 * 10 + 10),
    ("FC(256) → FC(100)",    32 * 32 * 256 + 256 + 256 * 100 + 100),
    ("FC(256) → FC(256)",    32 * 32 * 256 + 256 + 256 * 256 + 256),
]

print(f"  {'Архитектура':<25} {'CNN':>10} {'FC':>10} {'Сжатие':>8}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
for (c_name, c_model), (f_name, f_params) in zip(architectures, fc_architectures):
    c_params = c_model.param_count()
    ratio = f_params / c_params
    print(f"  {c_name:<25} {c_params:>10} {f_params:>10} {ratio:>6.1f}x")

print()
print("Вывод: на изображениях CNN требует в разы меньше параметров")
print("  за счёт weight sharing и local connectivity.")
print()
print("═" * 70)
print("Файл самодостаточен: только random + math, без NumPy / PyTorch.")
print("═" * 70)
