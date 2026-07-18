import math
import random

random.seed(42)


# ============================================================
#  Complex class
# ============================================================

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def __repr__(self):
        if abs(self.imag) < 1e-10:
            return f"{self.real:.4f}"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.4f} {sign} {abs(self.imag):.4f}i"


# ============================================================
#  Вспомогательные функции
# ============================================================

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result

def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result


# ============================================================
#  Демо 1: Комплексная арифметика
# ============================================================

print("=" * 55)
print("ДЕМО 1: Комплексная арифметика")
print("=" * 55)

z1 = Complex(3, 2)
z2 = Complex(1, 4)

print(f"\nz1 = {z1}")
print(f"z2 = {z2}")
print(f"\nz1 + z2 = {z1 + z2}")
print(f"z1 - z2 = {z1 - z2}")
print(f"z1 × z2 = {z1 * z2}")
print(f"z1 / z2 = {z1 / z2}")
print(f"|z1| = {z1.magnitude():.4f}")
print(f"arg(z1) = {z1.phase():.4f} rad = {math.degrees(z1.phase()):.2f}°")
print(f"conjugate(z1) = {z1.conjugate()}")
print(f"z1 × conj(z1) = {(z1 * z1.conjugate()).real:.4f} (всегда вещественное)")


# ============================================================
#  Демо 2: Euler's formula
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Euler's formula e^(iθ) = cos θ + i sin θ")
print("=" * 55)

for theta in [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]:
    e = euler(theta)
    print(f"  e^(i×{theta/math.pi:.2f}π) = {e}  (|z|={e.magnitude():.4f})")

result = euler(math.pi) + Complex(1, 0)
print(f"\ne^(iπ) + 1 = {result}  ←最美的 формула")


# ============================================================
#  Демо 3: Комплексное умножение = поворот
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Комплексное умножение = поворот")
print("=" * 55)

point = Complex(3, 4)
print(f"\nТочка: {point} (|z|={point.magnitude():.4f})")

for angle_deg in [30, 45, 90, 180]:
    angle = math.radians(angle_deg)
    rotated = point * euler(angle)
    print(f"  Поворот на {angle_deg:>3d}°: {rotated}  (|z|={rotated.magnitude():.4f})")

print(f"\n→ Модуль сохраняется, меняется только угол!")


# ============================================================
#  Демо 4: Корни единицы
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Корни единицы")
print("=" * 55)

for N in [4, 6, 8]:
    roots = roots_of_unity(N)
    print(f"\nN={N}: {[f'{r}' for r in roots]}")
    print(f"  |w_k| = {[f'{r.magnitude():.4f}' for r in roots]}")
    total = roots[0]
    for r in roots[1:]:
        total = total + r
    print(f"  Σw_k = {total} (ожидается 0)")


# ============================================================
#  Демо 5: DFT
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: DFT — анализ частот")
print("=" * 55)

# Сигнал: sin(2π×3×t) + 0.5×sin(2π×7×t)
N = 32
signal = []
for n in range(N):
    t = n / N
    val = math.sin(2 * math.pi * 3 * t) + 0.5 * math.sin(2 * math.pi * 7 * t)
    signal.append(val)

spectrum = dft(signal)

print(f"\nСигнал: sin(2π×3×t) + 0.5×sin(2π×7×t)")
print(f"Сэмплов: {N}")
print(f"\n{'k':>4} {'|X[k]|':>10} {'Частота':>10}")
print("-" * 28)

for k in range(N // 2):
    mag = spectrum[k].magnitude()
    if mag > 0.5:
        print(f"{k:>4} {mag:>10.4f} {k:>10}")

print(f"\n→ Пики на k=3 и k=7, амплитуда k=7 в 2× меньше")


# ============================================================
#  Демо 6: DFT → IDFT (обратное преобразование)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: DFT → IDFT (реконструкция)")
print("=" * 55)

original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
print(f"\nИсходный сигнал: {original}")

spectrum = dft(original)
reconstructed = idft(spectrum)

print(f"После DFT → IDFT:")
for orig, recon in zip(original, reconstructed):
    print(f"  {orig:.4f} → {recon.real:.4f} (ошибка: {abs(orig - recon.real):.2e})")


# ============================================================
#  Демо 7: Euler's formula и поворотная матрица
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Комплексное умножение = матрица поворота")
print("=" * 55)

x, y = 3.0, 4.0
theta = math.pi / 6  # 30°

# Комплексное умножение
z = Complex(x, y) * euler(theta)

# Матрица поворота
cos_t, sin_t = math.cos(theta), math.sin(theta)
mx = x * cos_t - y * sin_t
my = x * sin_t + y * cos_t

print(f"\nТочка: ({x}, {y}), поворот на 30°")
print(f"  Комплексное: ({z.real:.4f}, {z.imag:.4f})")
print(f"  Матрица:     ({mx:.4f}, {my:.4f})")
print(f"  Разница:     {abs(z.real - mx) + abs(z.imag - my):.2e}")
print(f"\n→ Результаты идентичны!")


# ============================================================
#  Демо 8: Связь с Transformer (sinusoidal PE)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Sinusoidal Positional Encoding")
print("=" * 55)

d_model = 16
pos = 5
print(f"\nPE(pos={pos}, d_model={d_model}):")
print(f"{'i':>4} {'freq':>10} {'sin':>10} {'cos':>10} {'|e^iθ|':>10}")
print("-" * 50)

for i in range(0, d_model, 2):
    freq = 1.0 / (10000 ** (i / d_model))
    theta = pos * freq
    s = math.sin(theta)
    c = math.cos(theta)
    mag = math.sqrt(s**2 + c**2)
    print(f"{i//2:>4} {freq:>10.6f} {s:>10.4f} {c:>10.4f} {mag:>10.4f}")

print(f"\n→ sin и cos — это real и imag части e^(iθ)")
print(f"→ Низкие частоты: медленное изменение (грубая позиция)")
print(f"→ Высокие частоты: быстрое изменение (точная позиция)")
