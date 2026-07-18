import math
import random

random.seed(42)


# ============================================================
#  Complex class (из урока 19)
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

    def __repr__(self):
        if abs(self.imag) < 1e-10:
            return f"{self.real:.4f}"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.4f} {sign} {abs(self.imag):.4f}i"


# ============================================================
#  DFT (O(N²))
# ============================================================

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = Complex(signal[n]) if not isinstance(signal[n], Complex) else signal[n]
            total = total + xn * w
        result.append(total)
    return result


# ============================================================
#  IDFT
# ============================================================

def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + spectrum[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result


# ============================================================
#  FFT (Cooley-Tukey, O(N log N))
# ============================================================

def fft(signal):
    N = len(signal)
    if N <= 1:
        x = signal[0] if isinstance(signal[0], Complex) else Complex(signal[0])
        return [x]
    if N % 2 != 0:
        return dft(signal)

    even = fft([signal[i] for i in range(0, N, 2)])
    odd = fft([signal[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result


# ============================================================
#  Анализ спектра
# ============================================================

def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def magnitude_spectrum(X):
    return [xk.magnitude() for xk in X]

def phase_spectrum(X):
    return [xk.phase() for xk in X]


# ============================================================
#  Свёртка через FFT
# ============================================================

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]

def convolve_direct(x, h):
    N = len(x) + len(h) - 1
    result = [0.0] * N
    for i in range(len(x)):
        for j in range(len(h)):
            result[i + j] += x[i] * h[j]
    return result


# ============================================================
#  Окна
# ============================================================

def hann_window(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]

def hamming_window(N):
    return [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]


# ============================================================
#  Демо 1: DFT vs FFT
# ============================================================

print("=" * 55)
print("ДЕМО 1: DFT vs FFT — одинаковый результат")
print("=" * 55)

signal = [math.sin(2 * math.pi * 3 * n / 16) for n in range(16)]
X_dft = dft(signal)
X_fft = fft(signal)

print(f"\nСигнал: sin(2π×3×n/16), N=16")
print(f"\n{'k':>4} {'DFT |X[k]|':>12} {'FFT |X[k]|':>12} {'Разница':>12}")
print("-" * 44)

for k in range(8):
    mag_dft = X_dft[k].magnitude()
    mag_fft = X_fft[k].magnitude()
    diff = abs(mag_dft - mag_fft)
    marker = "← пик" if mag_fft > 1.0 else ""
    print(f"{k:>4} {mag_dft:>12.4f} {mag_fft:>12.4f} {diff:>12.2e} {marker}")


# ============================================================
#  Демо 2: Анализ частот
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Анализ частот — спектральный анализ")
print("=" * 55)

# Сигнал: 5 Гц + 12 Гц (с разными амплитудами)
N = 64
fs = 64  # частота дискретизации
signal2 = []
for n in range(N):
    t = n / fs
    val = math.sin(2 * math.pi * 5 * t) + 0.5 * math.sin(2 * math.pi * 12 * t)
    signal2.append(val)

X2 = fft(signal2)
power = power_spectrum(X2)

print(f"\nСигнал: sin(2π×5×t) + 0.5×sin(2π×12×t)")
print(f"N={N}, fs={fs} Гц")
print(f"\n{'Частота (Гц)':>14} {'|X[k]|':>10} {'Power':>12}")
print("-" * 38)

for k in range(N // 2):
    freq = k * fs / N
    mag = X2[k].magnitude()
    pwr = power[k]
    if mag > 1.0:
        print(f"{freq:>14.1f} {mag:>10.4f} {pwr:>12.2f}")


# ============================================================
#  Демо 3: DFT → IDFT (реконструкция)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: DFT → IDFT — идеальная реконструкция")
print("=" * 55)

original = [math.sin(2 * math.pi * 5 * n / 64) for n in range(64)]
X = fft(original)
reconstructed = idft(X)

max_error = max(abs(orig - recon.real) for orig, recon in zip(original, reconstructed))
print(f"\nN=64, max ошибка реконструкции: {max_error:.2e}")
print(f"→ Идеальная реконструкция, информация не теряется")


# ============================================================
#  Демо 4: Convolution theorem
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Теорема о свёртке")
print("=" * 55)

x = [1, 2, 3, 4]
h = [1, 1, 1]

direct = convolve_direct(x, h)
fft_conv = convolve_fft(x, h)

print(f"\nx = {x}")
print(f"h = {h}")
print(f"\nПрямая свёртка:   {[f'{v:.2f}' for v in direct]}")
print(f"Свёртка через FFT: {[f'{v:.2f}' for v in fft_conv[:len(direct)]]}")
print(f"Разница: {max(abs(a - b) for a, b in zip(direct, fft_conv)):.2e}")


# ============================================================
#  Демо 5: Окна (windowing)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Окна — борьба со спектральным утечанием")
print("=" * 55)

# Два близких сигнала: 10 Гц и 12 Гц
N_win = 128
fs_win = 128
t_signal = [math.sin(2*math.pi*10*n/fs_win) + math.sin(2*math.pi*12*n/fs_win)
            for n in range(N_win)]

windows = {
    "Без окна": [1.0] * N_win,
    "Hann": hann_window(N_win),
    "Hamming": hamming_window(N_win),
}

for name, w in windows.items():
    windowed = [t * wi for t, wi in zip(t_signal, w)]
    X_w = fft(windowed)
    power_w = power_spectrum(X_w)

    # Ищем пики
    peaks = []
    for k in range(1, N_win // 2):
        freq = k * fs_win / N_win
        if power_w[k] > power_w[k-1] and power_w[k] > power_w[k+1] and power_w[k] > 100:
            peaks.append((freq, power_w[k]))

    print(f"\n{name}:")
    for freq, pwr in peaks:
        print(f"  Пик: {freq:.0f} Гц (power={pwr:.0f})")


# ============================================================
#  Демо 6: Связь с CNN и Transformer
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Связь с ML")
print("=" * 55)

print(f"""
┌─────────────────────────────────────────────────────────┐
│ Применение FFT в ML                                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ CNN:                                                    │
│   Свёртка фильтра × вход = IFFT(FFT(фильтр) · FFT(вход))│
│   Прямая свёртка O(N·M), через FFT O(N log N)          │
│                                                         │
│ FNet (2021):                                            │
│   Заменяет self-attention на FFT                        │
│   Сложность: O(N log N) вместо O(N²)                   │
│                                                         │
│ Sinusoidal PE:                                          │
│   PE(pos) = [sin(pos/freq), cos(pos/freq)]             │
│   = real и imag части e^(i·pos·freq)                    │
│                                                         │
│ Аудио модели (Whisper, DeepSpeech):                     │
│   Вход: мел-спектрограмма (STFT → мел-шкала)          │
│                                                         │
│ Time series:                                            │
│   Поиск периодичных паттернов через спектральный анализ │
└─────────────────────────────────────────────────────────┘
""")


# ============================================================
#  Демо 7: Свойства DFT
# ============================================================

print("=" * 55)
print("ДЕМО 7: Свойства DFT")
print("=" * 55)

print(f"""
┌────────────────────┬──────────────────────┬──────────────────────┐
│ Свойство           │ Временная область    │ Частотная область    │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Линейность         │ a·x + b·y            │ a·X + b·Y            │
│ Сдвиг во времени   │ x[n-k]               │ X[f]·e^(-2πifk/N)    │
│ Сдвиг частоты      │ x[n]·e^(2πif₀n/N)   │ X[f-f₀]              │
│ Свёртка            │ x * h                │ X·H (поэлементно)    │
│ Умножение          │ x·h (поэлементно)    │ X*H (свёртка, /N)    │
│ Парсеваль          │ Σ|x[n]|²             │ (1/N)·Σ|X[k]|²       │
└────────────────────┴──────────────────────┴──────────────────────┘

→ Энергия сохраняется через преобразование (Парсеваль)
""")
