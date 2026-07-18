"""
83 — Autoencoders: Encoder, Decoder, Latent Space Visualization
Pure Python (no numpy/torch/tensorflow). random.seed(42).
"""

import random
import math

random.seed(42)

# ─────────────────────────── helpers ───────────────────────────

def mat_zeros(r, c):
    return [[0.0] * c for _ in range(r)]

def mat_mul(a, b):
    """Multiply matrix a (m×n) by b (n×p)."""
    m, n, p = len(a), len(a[0]), len(b[0])
    out = mat_zeros(m, p)
    for i in range(m):
        for k in range(n):
            if a[i][k] == 0:
                continue
            for j in range(p):
                out[i][j] += a[i][k] * b[k][j]
    return out

def mat_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def mat_sub(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def mat_scale(a, s):
    return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]

def mat_transpose(a):
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]

def vec_to_col(v):
    return [[x] for x in v]

def col_to_vec(m):
    return [row[0] for row in m]

def mat_apply(m, fn):
    return [[fn(m[i][j]) for j in range(len(m[0]))] for i in range(len(m))]

def mat_rand(r, c, lo=-0.5, hi=0.5):
    return [[random.uniform(lo, hi) for _ in range(c)] for _ in range(r)]

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(y):
    return y * (1.0 - y)

def relu(x):
    return max(0.0, x)

def relu_deriv(y):
    return 1.0 if y > 0 else 0.0

def tanh_fn(x):
    return math.tanh(x)

def tanh_deriv(y):
    return 1.0 - y * y

def mse_loss(pred, target):
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)

def identity(x):
    return x

def identity_deriv(y):
    return 1.0

# ─────────────────────── SimpleAutoencoder ─────────────────────

class SimpleAutoencoder:
    """Two-layer autoencoder: input → latent → input."""

    def __init__(self, input_size, hidden_size, lr=0.1, activation="sigmoid"):
        self.lr = lr
        acts = {"sigmoid": (sigmoid, sigmoid_deriv),
                "relu": (relu, relu_deriv),
                "tanh": (tanh_fn, tanh_deriv)}
        self.act, self.act_d = acts.get(activation, acts["sigmoid"])

        # Xavier-ish init
        s1 = math.sqrt(2.0 / input_size)
        s2 = math.sqrt(2.0 / hidden_size)
        self.W_enc = mat_rand(hidden_size, input_size, -s1, s1)
        self.b_enc = [0.0] * hidden_size
        self.W_dec = mat_rand(input_size, hidden_size, -s2, s2)
        self.b_dec = [0.0] * input_size

    def forward(self, x):
        """Encode then decode; return (latent, output).
        W_enc: hidden×input, W_dec: input×hidden."""
        h = len(self.W_enc)
        n = len(x)

        # encoder: z = W_enc @ x + b_enc  → hidden×1
        z_lin = [0.0] * h
        for j in range(h):
            s = self.b_enc[j]
            for k in range(n):
                s += self.W_enc[j][k] * x[k]
            z_lin[j] = s
        self._z = [self.act(v) for v in z_lin]
        self._z_lin = z_lin

        # decoder: o = W_dec @ z + b_dec  → input×1
        o_lin = [0.0] * n
        for i in range(n):
            s = self.b_dec[i]
            for j in range(h):
                s += self.W_dec[i][j] * self._z[j]
            o_lin[i] = s
        self._o = [self.act(v) for v in o_lin]
        self._o_lin = o_lin
        return self._z, self._o

    def backward(self, x, target):
        """Backprop + weight update; return loss."""
        self.forward(x)
        n = len(x)
        loss = mse_loss(self._o, target)

        # output error
        out_err = [(self._o[i] - target[i]) * self.act_d(self._o[i]) for i in range(n)]

        # decoder gradient
        for i in range(n):
            for j in range(len(self._z)):
                self.W_dec[i][j] -= self.lr * out_err[i] * self._z[j]
            self.b_dec[i] -= self.lr * out_err[i]

        # hidden error
        hid_err = [0.0] * len(self._z)
        for j in range(len(self._z)):
            for i in range(n):
                hid_err[j] += out_err[i] * self.W_dec[i][j]
            hid_err[j] *= self.act_d(self._z[j])

        # encoder gradient
        for j in range(len(self._z)):
            for k in range(n):
                self.W_enc[j][k] -= self.lr * hid_err[j] * x[k]
            self.b_enc[j] -= self.lr * hid_err[j]

        return loss

    def train(self, data, epochs=200, verbose_every=50):
        losses = []
        for ep in range(1, epochs + 1):
            total = 0.0
            for sample in data:
                total += self.backward(sample, sample)
            avg = total / len(data)
            losses.append(avg)
            if verbose_every and ep % verbose_every == 0:
                print(f"  epoch {ep:>4d}  loss {avg:.6f}")
        return losses

    def compress(self, x):
        z, _ = self.forward(x)
        return z

    def decompress(self, z):
        """Given a latent vector, run the decoder only."""
        n = len(self.W_dec)
        h = len(z)
        o_lin = [0.0] * n
        for i in range(n):
            s = self.b_dec[i]
            for j in range(h):
                s += self.W_dec[i][j] * z[j]
            o_lin[i] = s
        return [self.act(v) for v in o_lin]

# ─────────────────────── MNIST-like data ───────────────────────

def make_digit(pattern, noise=0.0):
    """Create 7×5 binary-ish vector from a pattern grid."""
    vec = []
    for row in pattern:
        for v in row:
            val = v if noise == 0 else max(0.0, min(1.0, v + random.gauss(0, noise)))
            vec.append(val)
    return vec

DIGITS_PATTERNS = {
    0: [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
    1: [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]],
    2: [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
    3: [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[1,1,1,1,1]],
    4: [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1]],
    5: [[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[1,1,1,1,1]],
    6: [[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
    7: [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
    8: [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
    9: [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[1,1,1,1,1]],
}

def make_dataset(n_per_digit=5, noise=0.05):
    data = []
    for d in range(10):
        for _ in range(n_per_digit):
            data.append(make_digit(DIGITS_PATTERNS[d], noise=noise))
    return data

def ascii_art(vec, w=5, h=7):
    lines = []
    for r in range(h):
        row = ""
        for c in range(w):
            val = vec[r * w + c]
            row += "█" if val > 0.5 else ("░" if val > 0.2 else " ")
        lines.append(row)
    return "\n".join(lines)

def vec_str(v, prec=2):
    return "[" + ", ".join(f"{x:.{prec}f}" for x in v) + "]"

# ───────────────── DEMO 1: Encoder-Decoder Architecture ──────

def demo1():
    print("=" * 60)
    print("DEMO 1: Encoder-Decoder архитектура")
    print("=" * 60)

    input_size = 35  # 7×5
    hidden_size = 8

    ae = SimpleAutoencoder(input_size, hidden_size, lr=0.5, activation="sigmoid")

    print(f"\nInput size:  {input_size} (7×5 пикселей)")
    print(f"Latent size: {hidden_size} (сжатие в {input_size // hidden_size}×)")
    print(f"Activation:  sigmoid")
    print(f"\nEncoder weights shape: {len(ae.W_enc)}×{len(ae.W_enc[0])}")
    print(f"Decoder weights shape: {len(ae.W_dec)}×{len(ae.W_dec[0])}")

    sample = make_digit(DIGITS_PATTERNS[0])
    z, out = ae.forward(sample)

    print(f"\nПример: цифра 0")
    print(f"Input  (first 10):  {vec_str(sample[:10])}")
    print(f"Latent (compressed): {vec_str(z)}")
    print(f"Output (first 10):  {vec_str(out[:10])}")
    print(f"\nLatent vector — это сжатое представление входных данных.")
    print(f"Из {input_size} чисел → {hidden_size} чисел → восстановление.")

# ───────────────── DEMO 2: Training on Simple Data ───────────

def demo2():
    print("\n" + "=" * 60)
    print("DEMO 2: Обучение на MNIST-подобных данных")
    print("=" * 60)

    input_size = 35
    hidden_size = 8

    data = make_dataset(n_per_digit=3, noise=0.08)
    print(f"\nДатасет: {len(data)}样本, 10 классов (цифры 0-9)")
    print(f"Размерность: {input_size} → {hidden_size} → {input_size}")

    ae = SimpleAutoencoder(input_size, hidden_size, lr=1.0, activation="sigmoid")

    print("\nОбучение (200 эпох):")
    losses = ae.train(data, epochs=200, verbose_every=50)

    print(f"\nНачальная loss: {losses[0]:.6f}")
    print(f"Конечная  loss: {losses[-1]:.6f}")
    print(f"Снижение:      {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    # qualitative check
    for digit in [0, 3, 7]:
        sample = make_digit(DIGITS_PATTERNS[digit])
        _, out = ae.forward(sample)
        matches = sum(1 for p, t in zip(out, sample) if abs(p - t) < 0.3)
        print(f"\nЦифра {digit}: совпадение {matches}/{input_size} пикселей")

# ───────────────── DEMO 3: Compression & Reconstruction ─────

def demo3():
    print("\n" + "=" * 60)
    print("DEMO 3: Сжатие и восстановление")
    print("=" * 60)

    input_size = 35
    hidden_size = 6
    ae = SimpleAutoencoder(input_size, hidden_size, lr=1.0, activation="sigmoid")

    data = make_dataset(n_per_digit=5, noise=0.06)
    ae.train(data, epochs=300, verbose_every=100)

    print("\nОригинал → Latent → Восстановление:")
    print("-" * 50)

    for digit in [0, 4, 8]:
        sample = make_digit(DIGITS_PATTERNS[digit], noise=0.03)
        z = ae.compress(sample)
        rec = ae.decompress(z)

        print(f"\nЦифра {digit}:")
        print(f"  Latent: {vec_str(z)}")
        print(f"  Сжатие: {input_size} → {hidden_size} ({input_size / hidden_size:.1f}×)")

        orig_art = ascii_art(sample)
        rec_art = ascii_art(rec)
        print(f"  Оригинал:         Восстановление:")
        for lo, lr in zip(orig_art.split("\n"), rec_art.split("\n")):
            print(f"    {lo}       {lr}")

    print(f"\nКоэффициент сжатия: {input_size / hidden_size:.1f}×")
    print(f"Безопасная передача: {hidden_size} чисел вместо {input_size}")

# ───────────────── DEMO 4: Latent Space Interpolation ───────

def demo4():
    print("\n" + "=" * 60)
    print("DEMO 4: Latent space интерполяция")
    print("=" * 60)

    input_size = 35
    hidden_size = 6
    ae = SimpleAutoencoder(input_size, hidden_size, lr=1.0, activation="sigmoid")

    data = make_dataset(n_per_digit=5, noise=0.05)
    ae.train(data, epochs=300, verbose_every=0)

    # Get latent vectors for digits 1 and 7
    z1 = ae.compress(make_digit(DIGITS_PATTERNS[1]))
    z7 = ae.compress(make_digit(DIGITS_PATTERNS[7]))

    print(f"\nЛатентный вектор '1': {vec_str(z1)}")
    print(f"Латентный вектор '7': {vec_str(z7)}")

    print("\nИнтерполяция от '1' к '7' (5 шагов):")
    print("-" * 50)

    steps = 5
    for i in range(steps + 1):
        t = i / steps
        z_interp = [z1[j] * (1 - t) + z7[j] * t for j in range(hidden_size)]
        decoded = ae.decompress(z_interp)
        art = ascii_art(decoded)
        print(f"\nt = {t:.1f}  ({vec_str(z_interp[:3])}...)")
        for line in art.split("\n"):
            print(f"  {line}")

    # Grid interpolation between 4 digits
    print("\n\nДвумерная интерполяция (2×2 сетка):")
    print("-" * 50)

    z_a = ae.compress(make_digit(DIGITS_PATTERNS[2]))
    z_b = ae.compress(make_digit(DIGITS_PATTERNS[5]))
    z_c = ae.compress(make_digit(DIGITS_PATTERNS[3]))
    z_d = ae.compress(make_digit(DIGITS_PATTERNS[9]))

    labels = [("2", z_a), ("5", z_b), ("3", z_c), ("9", z_d)]

    # Show corners and center
    grid_data = [
        ("Угол 1 (≈2)", z_a),
        ("Угол 2 (≈5)", z_b),
        ("Центр (смесь)", [0.25 * z_a[j] + 0.25 * z_b[j] + 0.25 * z_c[j] + 0.25 * z_d[j]
                            for j in range(hidden_size)]),
        ("Угол 3 (≈3)", z_c),
        ("Угол 4 (≈9)", z_d),
    ]

    for name, z in grid_data:
        decoded = ae.decompress(z)
        art = ascii_art(decoded)
        print(f"\n{name}:")
        for line in art.split("\n"):
            print(f"  {line}")

    print("\nИнтерполяция показывает плавные переходы в latent space.")
    print("Между любыми двумя точками — непрерывный путь.")

# ─────────────────────── Main ─────────────────────────────────

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║     AUTOENCODERS — Encoder / Decoder / Latent Space        ║")
    print("║          Pure Python • MNIST-like • No Libraries           ║")
    print("╚" + "═" * 58 + "╝")

    demo1()
    demo2()
    demo3()
    demo4()

    print("\n" + "=" * 60)
    print("Итого:")
    print("  • Autoencoder: input → encoder → latent → decoder → output")
    print("  • Обучение методом backpropagation на мини-батчах")
    print("  • Latent space хранит сжатое представление данных")
    print("  • Интерполяция = плавный переход между образами")
    print("  • Основа для VAE, GAN, диффузионных моделей")
    print("=" * 60)
