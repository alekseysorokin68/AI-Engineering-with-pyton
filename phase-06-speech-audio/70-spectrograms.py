"""
Spectrograms & Mel-Scale — от STFT до MFCC
============================================
Самодостаточный файл: только Python stdlib + math.
Никаких numpy, scipy, librosa, torch.

Темы:
  1. STFT (Short-Time Fourier Transform)
  2. Спектрограмма
  3. Мел-шкала и мел-фильтры
  4. MFCC (Mel-Frequency Cepstral Coefficients)

Демо:
  Демо 1: STFT — разложение сигнала на окна
  Демо 2: Спектрограмма (ASCII-визуализация)
  Демо 3: Мел-фильтры
  Демо 4: MFCC извлечение
"""

import math
import random

random.seed(42)


# ============================================================
# 1. Утилиты: комплексная арифметика
# ============================================================

def cplx(re, im=0.0):
    return (re, im)


def cplx_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cplx_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def cplx_mul(a, b):
    return (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])


def cplx_abs(a):
    return math.sqrt(a[0]**2 + a[1]**2)


def cplx_phase(a):
    return math.atan2(a[1], a[0])


# ============================================================
# 2. DFT / FFT (Cooley-Tukey radix-2)
# ============================================================

def dft(x):
    """Очень медленный DFT — O(N^2). Используем как fallback."""
    N = len(x)
    X = []
    for k in range(N):
        s = cplx(0.0, 0.0)
        for n in range(N):
            angle = -2.0 * math.pi * k * n / N
            w = cplx(math.cos(angle), math.sin(angle))
            s = cplx_add(s, cplx_mul(x[n], w))
        X.append(s)
    return X


def _fft_radix2(x):
    """Cooley-Tukey FFT, требует N = 2^k."""
    N = len(x)
    if N <= 1:
        return x[:]
    if N % 2 != 0:
        return dft(x)

    even = _fft_radix2(x[0::2])
    odd = _fft_radix2(x[1::2])

    result = [cplx(0.0)] * N
    for k in range(N // 2):
        angle = -2.0 * math.pi * k / N
        tw = cplx(math.cos(angle), math.sin(angle))
        t = cplx_mul(tw, odd[k])
        result[k] = cplx_add(even[k], t)
        result[k + N // 2] = cplx_sub(even[k], t)
    return result


def fft(x):
    """FFT с дополнением нулями до ближайшей степени двойки."""
    N = len(x)
    M = 1
    while M < N:
        M <<= 1
    padded = x + [cplx(0.0)] * (M - N)
    return _fft_radix2(padded)


# ============================================================
# 3. Генерация тестовых сигналов
# ============================================================

def sine_wave(freq, sr, duration, amplitude=1.0):
    """Генерация синусоиды."""
    n_samples = int(sr * duration)
    return [amplitude * math.sin(2.0 * math.pi * freq * n / sr)
            for n in range(n_samples)]


def chirp_signal(sr, duration, f0=100, f1=4000):
    """Линейный чирп: частота нарастает от f0 до f1."""
    n_samples = int(sr * duration)
    return [math.sin(2.0 * math.pi * (f0 + (f1 - f0) * n / n_samples) * n / sr)
            for n in range(n_samples)]


def noise_with_sine(sr, duration, freq=440, noise_level=0.3):
    """Синусоида + шум."""
    sig = sine_wave(freq, sr, duration)
    return [s + noise_level * (random.random() - 0.5) * 2 for s in sig]


# ============================================================
# 4. Оконные функции
# ============================================================

def hamming_window(N):
    """Окно Хэмминга."""
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * n / (N - 1))
            for n in range(N)]


def hann_window(N):
    """Окно Ханна."""
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * n / (N - 1)))
            for n in range(N)]


def rectangular_window(N):
    return [1.0] * N


# ============================================================
# 5. STFT (Short-Time Fourier Transform)
# ============================================================

def stft(signal, n_fft=256, hop_length=None, win_func=None):
    """
    Short-Time Fourier Transform.

    Параметры:
        signal     — список отсчётов (float)
        n_fft      — размер окна FFT (степень двойки)
        hop_length — шаг окна (по умолчанию n_fft // 2)
        win_func   — оконная функция (по умолчанию Хэмминг)

    Возвращает:
        list of list of complex — спектры каждого окна
        N_fft — размер FFT (с дополнением нулями)
    """
    if hop_length is None:
        hop_length = n_fft // 2
    if win_func is None:
        win_func = hamming_window

    window = win_func(n_fft)
    frames = []

    # Дополняем сигнал нулями для последнего окна
    padded = signal + [0.0] * n_fft

    for start in range(0, len(signal), hop_length):
        frame = padded[start:start + n_fft]
        if len(frame) < n_fft:
            frame = frame + [0.0] * (n_fft - len(frame))
        # Применяем окно
        windowed = [frame[i] * window[i] for i in range(n_fft)]
        # FFT
        spectrum = fft([cplx(v) for v in windowed])
        frames.append(spectrum)

    return frames, n_fft


# ============================================================
# 6. Спектрограмма (мощность в дБ)
# ============================================================

def power_spectrum_db(spectrum):
    """Мощностной спектр в дБ (одно окно)."""
    eps = 1e-10
    return [10.0 * math.log10(cplx_abs(s)**2 + eps) for s in spectrum]


def spectrogram(signal, n_fft=256, hop_length=None, win_func=None):
    """
    Спектрограмма: список мощностных спектров в дБ для каждого окна.

    Возвращает:
        specs — list of list of float (dB)
        freqs — список частот для каждого бина
    """
    frames, N = stft(signal, n_fft, hop_length, win_func)
    specs = [power_spectrum_db(f) for f in frames]
    # Частоты бинов
    sr_hint = None  # не знаем sr здесь
    freqs = list(range(len(specs[0])))  # просто индексы бинов
    return specs, freqs


# ============================================================
# 7. Мел-шкала
# ============================================================

def hz_to_mel(hz):
    """Гц → мел."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """Мел → Гц."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ============================================================
# 8. Мел-фильтры (bank)
# ============================================================

def mel_filterbank(n_mels, n_fft, sr=16000, fmin=0.0, fmax=None):
    """
    Построение банка мел-фильтров.

    Параметры:
        n_mels  — количество мел-фильтров
        n_fft   — размер FFT
        sr      — частота дискретизации
        fmin    — минимальная частота
        fmax    — максимальная частота (по умолчанию sr/2)

    Возвращает:
        list of list of float — каждый фильтр (длина = n_fft // 2 + 1)
    """
    if fmax is None:
        fmax = sr / 2.0

    n_freqs = n_fft // 2 + 1
    # Граничные точки в мел-шкале
    mel_low = hz_to_mel(fmin)
    mel_high = hz_to_mel(fmax)
    mel_points = [mel_low + i * (mel_high - mel_low) / (n_mels + 1)
                  for i in range(n_mels + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]

    # Переводим частоты в индексы бинов
    bin_points = [int(math.floor((n_fft + 1) * f / sr)) for f in hz_points]

    filters = []
    for m in range(n_mels):
        filt = [0.0] * n_freqs
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]

        # Восходящий склон
        for k in range(left, center + 1):
            if center != left:
                filt[k] = (k - left) / (center - left)
        # Нисходящий склон
        for k in range(center, right + 1):
            if right != center:
                filt[k] = (right - k) / (right - center)

        filters.append(filt)

    return filters


# ============================================================
# 9. Mel-Spectrogram
# ============================================================

def mel_spectrogram(signal, n_fft=256, hop_length=None, n_mels=40,
                    sr=16000, fmin=0.0, fmax=None, win_func=None):
    """
    Мел-спектрограмма: энергия по мел-фильтрам для каждого окна.
    """
    specs, _ = spectrogram(signal, n_fft, hop_length, win_func)
    filters = mel_filterbank(n_mels, n_fft, sr, fmin, fmax)
    n_freqs = n_fft // 2 + 1

    mel_specs = []
    for spec in specs:
        mel_frame = []
        for filt in filters:
            energy = sum(spec[i] * filt[i] for i in range(n_freqs))
            mel_frame.append(energy)
        mel_specs.append(mel_frame)
    return mel_specs


# ============================================================
# 10. DCT-II (Discrete Cosine Transform)
# ============================================================

def dct_ii(x):
    """DCT-II (ортонормированный)."""
    N = len(x)
    result = []
    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(math.pi * k * (2.0 * n + 1) / (2.0 * N))
        result.append(s)
    return result


def idct_ii(x):
    """Обратный DCT-II."""
    N = len(x)
    result = []
    for n in range(N):
        s = 0.0
        for k in range(N):
            coeff = 1.0 if k == 0 else 2.0
            s += coeff * x[k] * math.cos(math.pi * k * (2.0 * n + 1) / (2.0 * N))
        result.append(s / (2.0 * N))
    return result


# ============================================================
# 11. MFCC (Mel-Frequency Cepstral Coefficients)
# ============================================================

def mfcc(signal, n_fft=256, hop_length=None, n_mels=40,
         n_mfcc=13, sr=16000, fmin=0.0, fmax=None, win_func=None):
    """
    Извлечение MFCC из аудиосигнала.

    Этапы:
      1. STFT → мощностной спектр
      2. Мел-фильтры → мел-спектрограмма (энергия в мел-бинах)
      3. log() по каждой мел-полосе
      4. DCT → цепстральные коэффициенты
      5. Берём первые n_mfcc коэффициентов

    Возвращает:
        list of list of float — MFCC для каждого фрейма
    """
    mel_specs = mel_spectrogram(signal, n_fft, hop_length, n_mels,
                                sr, fmin, fmax, win_func)
    eps = 1e-10

    mfcc_frames = []
    for mel_frame in mel_specs:
        # Логарифм энергии
        log_mel = [math.log(max(e, eps)) for e in mel_frame]
        # DCT-II
        cepstral = dct_ii(log_mel)
        # Берём первые n_mfcc коэффициентов
        mfcc_frames.append(cepstral[:n_mfcc])

    return mfcc_frames


# ============================================================
# 12. ASCII-визуализация
# ============================================================

def ascii_heatmap(data, width=60, height=15, label=""):
    """
    Простейшая ASCII-визуализация 2D-матрицы.

    data: list of list of float
    """
    if not data:
        print("  (пусто)")
        return

    flat = [v for row in data for v in row]
    mn, mx = min(flat), max(flat)
    rng = mx - mn if mx > mn else 1.0

    chars = " .:-=+*#%@"

    n_rows = len(data)
    n_cols = len(data[0])

    # Масштабируем до нужного размера
    row_step = max(1, n_rows // height)
    col_step = max(1, n_cols // width)

    print(f"  {label}")
    print(f"  {'':>4}", end="")
    print("+" + "-" * min(n_cols, width) + "+")

    for r in range(0, min(n_rows, height * row_step), row_step):
        print(f"  {r:>4}|", end="")
        for c in range(0, min(n_cols, width * col_step), col_step):
            val = data[r][c]
            idx = int((val - mn) / rng * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            print(chars[idx], end="")
        print("|")

    print(f"  {'':>4}", end="")
    print("+" + "-" * min(n_cols, width) + "+")
    print(f"  [{mn:.1f} .. {mx:.1f}]")


def ascii_bar(values, width=50, label=""):
    """Горизонтальная шкала для одномерного массива."""
    if not values:
        return
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0
    chars = " |/=*#%@"
    print(f"  {label}")
    for i, v in enumerate(values):
        bar_len = int((v - mn) / rng * width)
        bar = chars[min(len(chars)-1, bar_len // max(1, width // len(chars)))] * bar_len
        print(f"  {i:>3} | {bar} {v:.2f}")


# ============================================================
#                     ДЕМОНСТРАЦИИ
# ============================================================

def demo_1_stft():
    """Демо 1: STFT — разложение сигнала на окна."""
    print("=" * 65)
    print("  ДЕМО 1: STFT — разложение на окна")
    print("=" * 65)

    sr = 1000  # Частота дискретизации 1 кГц
    duration = 0.128  # 128 мс → 128 отсчётов
    freq = 100  # 100 Гц синусоида

    signal = sine_wave(freq, sr, duration)
    print(f"\n  Сигнал: синусоида {freq} Гц, sr={sr} Гц, "
          f"длительность={duration*1000:.0f} мс ({len(signal)} отсчётов)")

    # STFT с окном 32 отсчёта
    n_fft = 32
    hop = 16
    print(f"  Размер окна FFT: {n_fft}, шаг: {hop}")
    print(f"  Количество окон: 1 + (128 - 32) / 16 = 7\n")

    frames, N = stft(signal, n_fft=n_fft, hop_length=hop)
    print(f"  STFT результат: {len(frames)} фреймов, каждый — {N} комплексных чисел\n")

    # Показываем магнитуду для каждого окна
    print("  Магнитуда спектра (первые 5 бинов для каждого окна):")
    print(f"  {'Окно':>5}  {'Bin0':>8}  {'Bin1':>8}  {'Bin2':>8}  {'Bin3':>8}  {'Bin4':>8}")
    print("  " + "-" * 55)
    for i, frame in enumerate(frames):
        mags = [cplx_abs(frame[j]) for j in range(min(5, len(frame)))]
        row = f"  {i:>5}  " + "  ".join(f"{m:>8.3f}" for m in mags)
        print(row)

    # Проверяем, что основная энергия в бине, соответствующем 100 Гц
    bin_of_interest = int(freq * n_fft / sr)
    print(f"\n  Частотный бин для {freq} Гц: {bin_of_interest}")
    for i, frame in enumerate(frames):
        mag = cplx_abs(frame[bin_of_interest])
        bar = "#" * int(mag * 5)
        print(f"  Окно {i}: Bin[{bin_of_interest}] = {mag:.3f}  {bar}")

    print()


def demo_2_spectrogram():
    """Демо 2: Спектрограмма (ASCII-визуализация)."""
    print("=" * 65)
    print("  ДЕМО 2: Спектрограмма")
    print("=" * 65)

    sr = 8000
    duration = 0.5  # 500 мс

    # Чирп: частота нарастает от 100 до 2000 Гц
    signal = chirp_signal(sr, duration, f0=100, f1=2000)
    print(f"\n  Сигнал: чирп 100→2000 Гц, sr={sr} Гц, "
          f"длительность={duration*1000:.0f} мс ({len(signal)} отсчётов)")

    n_fft = 128
    hop = 32
    specs, freqs = spectrogram(signal, n_fft=n_fft, hop_length=hop)

    print(f"  Размер окна: {n_fft}, шаг: {hop}, фреймов: {len(specs)}")
    print(f"  Бинов в спектре: {len(specs[0])}\n")

    # ASCII-визуализация
    ascii_heatmap(specs, width=50, height=12,
                  label="Спектрограмма (дБ) — горизонтально время, вертикально частота")

    # Показываем пики в первом и последнем фрейме
    print(f"\n  Пиковые частоты:")
    for idx in [0, len(specs)//2, len(specs)-1]:
        spec = specs[idx]
        peak_bin = spec.index(max(spec))
        freq_est = peak_bin * sr / n_fft
        print(f"  Фрейм {idx:>3}: пик на бине {peak_bin:>3} → {freq_est:.0f} Гц "
              f"(ожидается ~{100 + (2000-100)*idx/(len(specs)-1):.0f} Гц)")

    print()


def demo_3_mel_filters():
    """Демо 3: Мел-фильтры."""
    print("=" * 65)
    print("  ДЕМО 3: Мел-фильтры")
    print("=" * 65)

    sr = 16000
    n_fft = 256
    n_mels = 10  # меньше для наглядности

    filters = mel_filterbank(n_mels, n_fft, sr)
    n_freqs = n_fft // 2 + 1
    freq_res = sr / n_fft  # разрешение по частоте

    print(f"\n  Параметры: sr={sr} Гц, n_fft={n_fft}, n_mels={n_mels}")
    print(f"  Разрешение по частоте: {freq_res:.1f} Гц")
    print(f"  Диапазон: 0 — {sr//2} Гц\n")

    # Показываем границы мел-фильтров
    mel_points = []
    for i in range(n_mels + 2):
        m = hz_to_mel(0) + i * (hz_to_mel(sr/2) - hz_to_mel(0)) / (n_mels + 1)
        hz = mel_to_hz(m)
        mel_points.append(hz)
    print("  Граничные частоты фильтров (Гц):")
    for i, hz in enumerate(mel_points):
        print(f"    Фильтр {i}: {hz:>7.1f} Гц  (мел: {hz_to_mel(hz):>6.1f})")

    # ASCII-визуализация фильтров
    print(f"\n  Мел-фильтры (каждый символ ≈ 1 фильтр по высоте):")
    ascii_heatmap(filters, width=50, height=n_mels,
                  label="Фильтры × бины (горизонтально = частота)")

    # Свойства мел-шкалы
    print(f"\n  Свойства мел-шкалы:")
    test_freqs = [100, 500, 1000, 2000, 4000, 8000]
    for f in test_freqs:
        m = hz_to_mel(f)
        back = mel_to_hz(m)
        print(f"    {f:>5} Гц → {m:>7.1f} мел → {back:>7.1f} Гц")

    print()


def demo_4_mfcc():
    """Демо 4: MFCC извлечение."""
    print("=" * 65)
    print("  ДЕМО 4: MFCC извлечение")
    print("=" * 65)

    sr = 16000
    duration = 0.3

    # Сигнал: два тона + шум
    sig1 = sine_wave(300, sr, duration, amplitude=1.0)
    sig2 = sine_wave(1000, sr, duration, amplitude=0.5)
    signal = [s1 + s2 + 0.1 * (random.random() - 0.5) * 2
              for s1, s2 in zip(sig1, sig2)]

    print(f"\n  Сигнал: 300 Гц + 1000 Гц + шум, sr={sr} Гц, "
          f"длительность={duration*1000:.0f} мс ({len(signal)} отсчётов)")

    n_fft = 256
    hop = 128
    n_mels = 26
    n_mfcc = 13

    mfcc_frames = mfcc(signal, n_fft=n_fft, hop_length=hop,
                       n_mels=n_mels, n_mfcc=n_mfcc, sr=sr)

    print(f"  Параметры: n_fft={n_fft}, hop={hop}, n_mels={n_mels}, "
          f"n_mfcc={n_mfcc}")
    print(f"  Фреймов: {len(mfcc_frames)}\n")

    # Таблица MFCC
    print(f"  MFCC коэффициенты для каждого фрейма:")
    header = f"  {'Фрейм':>6}"
    for c in range(n_mfcc):
        header += f"  MFCC{c:>2}"
    print(header)
    print("  " + "-" * (8 + n_mfcc * 7))

    for i, frame in enumerate(mfcc_frames):
        row = f"  {i:>6}"
        for v in frame:
            row += f"  {v:>6.2f}"
        print(row)

    # Первые 5 фреймов — MFCC0 (нормальная энергия)
    print(f"\n  MFCC0 (нормальная энергия) по фреймам:")
    ascii_bar([f[0] for f in mfcc_frames[:10]], width=40,
              label="MFCC0")

    # Восстановление спектра из MFCC (обратное преобразование)
    print(f"\n  Демонстрация обратного DCT (восстановление лог-мел-спектра):")
    test_mel = [f[0] for f in mfcc_frames]  # первый фрейм, все мел-коэффициенты
    # Возьмём полный набор из 26 мел-фильтров
    mel_specs = mel_spectrogram(signal, n_fft=n_fft, hop_length=hop,
                                n_mels=n_mels, sr=sr)
    log_mel_orig = [math.log(max(e, 1e-10)) for e in mel_specs[0]]
    cepstral_full = dct_ii(log_mel_orig)
    log_mel_recon = idct_ii(cepstral_full)

    print(f"  Фрейм 0: исходный log-мел-спектр vs восстановленный")
    print(f"  {'Бин':>4}  {'Оригинал':>10}  {'Восстановл.':>12}  {'Разница':>10}")
    print("  " + "-" * 42)
    for i in range(min(10, len(log_mel_orig))):
        diff = abs(log_mel_orig[i] - log_mel_recon[i])
        print(f"  {i:>4}  {log_mel_orig[i]:>10.3f}  {log_mel_recon[i]:>12.3f}  {diff:>10.6f}")

    print(f"\n  MFCC — это компактное представление спектра сигнала.")
    print(f"  Первые коэффициенты кодируют общую форму спектра (тембр),")
    print(f"  а высокие — детали и шум.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Spectrograms & Mel-Scale  —  STFT → Спектрограмма → MFCC      ║")
    print("║  Самодостаточный файл: только Python stdlib + math              ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    demo_1_stft()
    demo_2_spectrogram()
    demo_3_mel_filters()
    demo_4_mfcc()

    print("=" * 65)
    print("  Все демонстрации завершены.")
    print("  Файл: phase-06-speech-audio/70-spectrograms.py")
    print("=" * 65)
