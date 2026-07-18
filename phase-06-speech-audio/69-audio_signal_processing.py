"""
69 - Основы обработки аудио сигналов на Python (без внешних библиотек)
=====================================================================

Реализовано с нуля:
- Генерация синусоидальных сигналов
- Спектральный анализ (DFT/FFT)
- Фильтрация (low-pass, high-pass)
- Окнонание (Hann, Hamming)
"""

import math
import random

random.seed(42)


# ──────────────────────────────────────────────────────────────────────
# 1. Генерация синусоидальных сигналов
# ──────────────────────────────────────────────────────────────────────

def generate_sine(frequency, sample_rate, duration, amplitude=1.0, phase=0.0):
    """Генерация синусоидального сигнала.

    Args:
        frequency: частота в Гц
        sample_rate: частота дискретизации (Гц)
        duration: длительность в секундах
        amplitude: амплитуда
        phase: начальная фаза в радианах

    Returns:
        список отсчётов сигнала
    """
    n_samples = int(sample_rate * duration)
    signal = []
    for i in range(n_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t + phase)
        signal.append(value)
    return signal


def generate_square(frequency, sample_rate, duration, amplitude=1.0):
    """Генерация меандра (прямоугольный сигнал)."""
    n_samples = int(sample_rate * duration)
    signal = []
    for i in range(n_samples):
        t = i / sample_rate
        phase = (2 * math.pi * frequency * t) % (2 * math.pi)
        value = amplitude if phase < math.pi else -amplitude
        signal.append(value)
    return signal


def generate_sawtooth(frequency, sample_rate, duration, amplitude=1.0):
    """Генерация пилообразного сигнала."""
    n_samples = int(sample_rate * duration)
    signal = []
    period = 1.0 / frequency
    for i in range(n_samples):
        t = i / sample_rate
        phase = (t % period) / period  # от 0 до 1
        value = amplitude * (2 * phase - 1)
        signal.append(value)
    return signal


def generate_noised_sine(frequency, sample_rate, duration, amplitude=1.0,
                         noise_level=0.2):
    """Синусоида + аддитивный гауссов шум."""
    signal = generate_sine(frequency, sample_rate, duration, amplitude)
    noisy = []
    for v in signal:
        noise = random.gauss(0, noise_level)
        noisy.append(v + noise)
    return noisy


# ──────────────────────────────────────────────────────────────────────
# 2. Спектральный анализ (DFT / FFT)
# ──────────────────────────────────────────────────────────────────────

def dft(signal):
    """Дискретное преобразование Фурье (наивная реализация O(N^2)).

    Returns:
        список комплексных коэффициентов [X[0], X[1], ..., X[N-1]]
    """
    N = len(signal)
    spectrum = []
    for k in range(N):
        real_sum = 0.0
        imag_sum = 0.0
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            real_sum += signal[n] * math.cos(angle)
            imag_sum += signal[n] * math.sin(angle)
        spectrum.append(complex(real_sum, imag_sum))
    return spectrum


def fft(signal):
    """Быстрое преобразование Фурье (Cooley-Tukey, рекурсивное).

    Требует длину сигнала — степень двойки.
    Возвращает список комплексных коэффициентов.
    """
    N = len(signal)
    if N <= 1:
        return [complex(x) for x in signal]

    if N & (N - 1) != 0:
        # Длина не степень двойки — дополним нулями до ближайшей степени
        next_pow2 = 1
        while next_pow2 < N:
            next_pow2 <<= 1
        signal = list(signal) + [0.0] * (next_pow2 - N)
        N = next_pow2

    if N <= 8:
        return dft(signal)

    even = fft(signal[0::2])
    odd = fft(signal[1::2])

    result = [complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t

    return result


def magnitude_spectrum(signal, sample_rate):
    """Вычисление амплитудного спектра.

    Returns:
        (freqs, magnitudes) — список частот и амплитуд
    """
    N = len(signal)
    spectrum = fft(signal)
    n_unique = N // 2 + 1

    freqs = []
    magnitudes = []
    for k in range(n_unique):
        freq = k * sample_rate / N
        mag = math.sqrt(spectrum[k].real ** 2 + spectrum[k].imag ** 2) / N
        freqs.append(freq)
        magnitudes.append(mag)

    return freqs, magnitudes


def phase_spectrum(signal, sample_rate):
    """Вычисление фазового спектра."""
    N = len(signal)
    spectrum = fft(signal)
    n_unique = N // 2 + 1

    freqs = []
    phases = []
    for k in range(n_unique):
        freq = k * sample_rate / N
        phase = math.atan2(spectrum[k].imag, spectrum[k].real)
        freqs.append(freq)
        phases.append(phase)

    return freqs, phases


# ──────────────────────────────────────────────────────────────────────
# 3. Фильтрация
# ──────────────────────────────────────────────────────────────────────

def ideal_lowpass(signal, sample_rate, cutoff_freq):
    """Идеальный низкочастотный фильтр (в частотной области).

    Обнуляет все гармоники выше cutoff_freq.
    """
    N = len(signal)
    spectrum = fft(signal)

    filtered = []
    for k in range(N):
        freq = k * sample_rate / N if k <= N // 2 else (N - k) * sample_rate / N
        if freq <= cutoff_freq:
            filtered.append(spectrum[k])
        else:
            filtered.append(complex(0))

    # Обратное DFT
    result = _ifft(filtered)
    return [v.real for v in result]


def ideal_highpass(signal, sample_rate, cutoff_freq):
    """Идеальный высокочастотный фильтр (в частотной области).

    Обнуляет все гармоники ниже cutoff_freq.
    """
    N = len(signal)
    spectrum = fft(signal)

    filtered = []
    for k in range(N):
        freq = k * sample_rate / N if k <= N // 2 else (N - k) * sample_rate / N
        if freq >= cutoff_freq:
            filtered.append(spectrum[k])
        else:
            filtered.append(complex(0))

    result = _ifft(filtered)
    return [v.real for v in result]


def ideal_bandpass(signal, sample_rate, low_freq, high_freq):
    """Идеальный полосовой фильтр."""
    N = len(signal)
    spectrum = fft(signal)

    filtered = []
    for k in range(N):
        freq = k * sample_rate / N if k <= N // 2 else (N - k) * sample_rate / N
        if low_freq <= freq <= high_freq:
            filtered.append(spectrum[k])
        else:
            filtered.append(complex(0))

    result = _ifft(filtered)
    return [v.real for v in result]


def moving_average_filter(signal, window_size):
    """Фильтр скользящего средного (простое сглаживание)."""
    result = []
    half_w = window_size // 2
    N = len(signal)
    for i in range(N):
        start = max(0, i - half_w)
        end = min(N, i + half_w + 1)
        window = signal[start:end]
        avg = sum(window) / len(window)
        result.append(avg)
    return result


def _ifft(spectrum):
    """Обратное DFT (наивное, O(N^2))."""
    N = len(spectrum)
    result = []
    for n in range(N):
        real_sum = 0.0
        imag_sum = 0.0
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            real_sum += spectrum[k].real * math.cos(angle) - spectrum[k].imag * math.sin(angle)
            imag_sum += spectrum[k].real * math.sin(angle) + spectrum[k].imag * math.cos(angle)
        result.append(complex(real_sum / N, imag_sum / N))
    return result


# ──────────────────────────────────────────────────────────────────────
# 4. Окнонание (Windowing)
# ──────────────────────────────────────────────────────────────────────

def rectangular_window(N):
    """Прямоугольное окно (без оконнания)."""
    return [1.0] * N


def hann_window(N):
    """Окно Ханна (raised cosine).

    w[n] = 0.5 * (1 - cos(2πn / (N-1)))
    """
    if N <= 1:
        return [1.0]
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]


def hamming_window(N):
    """Окно Хэмминга.

    w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
    """
    if N <= 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]


def blackman_window(N):
    """Окно Блэкмана.

    w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
    """
    if N <= 1:
        return [1.0]
    return [0.42 - 0.5 * math.cos(2 * math.pi * n / (N - 1)) +
            0.08 * math.cos(4 * math.pi * n / (N - 1)) for n in range(N)]


def apply_window(signal, window_func):
    """Применение оконной функции к сигналу."""
    N = len(signal)
    w = window_func(N)
    return [signal[i] * w[i] for i in range(N)]


# ──────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────

def rms(signal):
    """Корень из среднего квадрата (мощность сигнала)."""
    return math.sqrt(sum(v * v for v in signal) / len(signal))


def peak_amplitude(signal):
    """Пиковая амплитуда."""
    return max(abs(v) for v in signal)


def signal_stats(signal, name="signal"):
    """Вывод статистики сигнала."""
    print(f"  [{name}]")
    print(f"    Длина:           {len(signal)} отсчётов")
    print(f"    Пиковая ампл.:   {peak_amplitude(signal):.6f}")
    print(f"    RMS:             {rms(signal):.6f}")
    print(f"    Мин:             {min(signal):.6f}")
    print(f"    Макс:            {max(signal):.6f}")


def print_spectrum_peaks(freqs, magnitudes, top_n=5):
    """Вывод топ-N пиков спектра."""
    indexed = list(enumerate(magnitudes))
    indexed.sort(key=lambda x: x[1], reverse=True)
    print(f"    Топ-{top_n} пиков спектра:")
    for rank, (idx, mag) in enumerate(indexed[:top_n], 1):
        print(f"      {rank}. {freqs[idx]:.1f} Гц  ампл. = {mag:.6f}")


def draw_ascii_bar(value, max_value, width=40):
    """Асси-бар для визуализации."""
    if max_value == 0:
        filled = 0
    else:
        filled = int(abs(value) / max_value * width)
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


# ──────────────────────────────────────────────────────────────────────
# ДЕМО
# ──────────────────────────────────────────────────────────────────────

def demo1_signal_generation():
    """Демо 1: Генерация и визуализация различных сигналов."""
    print("=" * 70)
    print("ДЕМО 1: Генерация и визуализация сигналов")
    print("=" * 70)

    sr = 8000  # частота дискретизации
    dur = 0.05  # 50 мс (400 отсчётов — удобно для визуализации)

    # Синусоида 440 Гц (нота La)
    sine = generate_sine(440, sr, dur)
    print("\n[Синусоида 440 Гц, 8 кГц, 50 мс]")
    signal_stats(sine, "sine_440Hz")
    print("    Визуализация (первые 80 отсчётов):")
    for i in range(0, min(80, len(sine)), 8):
        chunk = sine[i:i + 8]
        avg = sum(chunk) / len(chunk)
        print(f"      [{i:3d}]: {draw_ascii_bar(avg, 1.0, 40)}  {avg:+.3f}")

    # Меандр 200 Гц
    square = generate_square(200, sr, dur)
    print(f"\n[Меандр 200 Гц]")
    signal_stats(square, "square_200Hz")
    print("    Визуализация (первые 80 отсчётов):")
    for i in range(0, min(80, len(square)), 8):
        chunk = square[i:i + 8]
        avg = sum(chunk) / len(chunk)
        print(f"      [{i:3d}]: {draw_ascii_bar(avg, 1.0, 40)}  {avg:+.3f}")

    # Пила 300 Гц
    sawtooth = generate_sawtooth(300, sr, dur)
    print(f"\n[Пилообразный 300 Гц]")
    signal_stats(sawtooth, "sawtooth_300Hz")
    print("    Визуализация (первые 80 отсчётов):")
    for i in range(0, min(80, len(sawtooth)), 8):
        chunk = sawtooth[i:i + 8]
        avg = sum(chunk) / len(chunk)
        print(f"      [{i:3d}]: {draw_ascii_bar(avg, 1.0, 40)}  {avg:+.3f}")

    # Суммарный сигнал (аккорд: 220 + 440 + 880 Гц)
    dur2 = 0.05
    n = int(sr * dur2)
    chord = []
    for i in range(n):
        t = i / sr
        val = (math.sin(2 * math.pi * 220 * t) +
               math.sin(2 * math.pi * 440 * t) +
               math.sin(2 * math.pi * 880 * t)) / 3
        chord.append(val)
    print(f"\n[Суммарный сигнал (220 + 440 + 880 Гц)]")
    signal_stats(chord, "chord")
    print("    Визуализация (первые 80 отсчётов):")
    for i in range(0, min(80, len(chord)), 8):
        chunk = chord[i:i + 8]
        avg = sum(chunk) / len(chunk)
        print(f"      [{i:3d}]: {draw_ascii_bar(avg, 1.0, 40)}  {avg:+.3f}")

    print()


def demo2_spectral_analysis():
    """Демо 2: Спектральный анализ."""
    print("=" * 70)
    print("ДЕМО 2: Спектральный анализ (FFT)")
    print("=" * 70)

    sr = 1000  # 1 кГц
    dur = 0.16  # 160 отсчётов
    n = int(sr * dur)

    # Сигнал: 50 Гц + 120 Гц + шум
    signal = []
    for i in range(n):
        t = i / sr
        val = (math.sin(2 * math.pi * 50 * t) +
               0.5 * math.sin(2 * math.pi * 120 * t) +
               random.gauss(0, 0.1))
        signal.append(val)

    print(f"\n[Сигнал: 50 Гц + 120 Гц + шум]")
    print(f"  Частота дискретизации: {sr} Гц")
    print(f"  Длительность: {dur} с ({n} отсчётов)")
    signal_stats(signal, "composite")

    freqs, mags = magnitude_spectrum(signal, sr)

    print(f"\n[Амплитудный спектр (первые {min(30, len(freqs))} бинов)]")
    max_mag = max(mags) if mags else 1.0
    for k in range(min(30, len(freqs))):
        if freqs[k] > 0:
            print(f"    {freqs[k]:6.1f} Гц  |{draw_ascii_bar(mags[k], max_mag, 40)}| {mags[k]:.4f}")

    print_spectrum_peaks(freqs, mags, top_n=5)

    # Фазовый спектр
    _, phases = phase_spectrum(signal, sr)
    print(f"\n[Фазовый спектр (первые 15 бинов)]")
    for k in range(min(15, len(phases))):
        if freqs[k] > 0:
            print(f"    {freqs[k]:6.1f} Гц  фаза = {phases[k]:+.4f} рад ({math.degrees(phases[k]):+.1f}°)")

    print()


def demo3_filtering():
    """Демо 3: Фильтрация."""
    print("=" * 70)
    print("ДЕМО 3: Фильтрация сигналов")
    print("=" * 70)

    sr = 1000
    dur = 0.16
    n = int(sr * dur)

    # Сигнал: 30 Гц (низкая) + 200 Гц (высокая)
    signal = []
    for i in range(n):
        t = i / sr
        low = math.sin(2 * math.pi * 30 * t)
        high = 0.7 * math.sin(2 * math.pi * 200 * t)
        signal.append(low + high)

    print(f"\n[Исходный сигнал: 30 Гц + 200 Гц]")
    signal_stats(signal, "mixed")

    # Low-pass (пропускаем < 100 Гц)
    lp = ideal_lowpass(signal, sr, 100)
    print("\n[Low-pass фильтр: cutoff = 100 Гц]")
    signal_stats(lp, "lowpass")
    print(f"    Ожидание: остаётся 30 Гц, 200 Гц подавлен")
    freqs_lp, mags_lp = magnitude_spectrum(lp, sr)
    print_spectrum_peaks(freqs_lp, mags_lp, top_n=3)

    # High-pass (пропускаем > 100 Гц)
    hp = ideal_highpass(signal, sr, 100)
    print("\n[High-pass фильтр: cutoff = 100 Гц]")
    signal_stats(hp, "highpass")
    print(f"    Ожидание: остаётся 200 Гц, 30 Гц подавлен")
    freqs_hp, mags_hp = magnitude_spectrum(hp, sr)
    print_spectrum_peaks(freqs_hp, mags_hp, top_n=3)

    # Bandpass (пропускаем 50–150 Гц)
    bp = ideal_bandpass(signal, sr, 50, 150)
    print("\n[Bandpass фильтр: 50–150 Гц]")
    signal_stats(bp, "bandpass")
    print(f"    Ожидание: обе гармоники подавлены")
    freqs_bp, mags_bp = magnitude_spectrum(bp, sr)
    print_spectrum_peaks(freqs_bp, mags_bp, top_n=3)

    # Moving average
    noisy = generate_sine(100, sr, 0.1)
    for i in range(len(noisy)):
        noisy[i] += random.gauss(0, 0.3)

    smoothed = moving_average_filter(noisy, 5)
    print(f"\n[Скользящее среднее: окно = 5]")
    print(f"    До сглаживания RMS: {rms(noisy):.4f}")
    print(f"    После сглаживания RMS: {rms(smoothed):.4f}")

    print()


def demo4_windowing():
    """Демо 4: Окнонание (Hann, Hamming, Blackman)."""
    print("=" * 70)
    print("ДЕМО 4: Окнонание (windowing)")
    print("=" * 70)

    N = 64  # длина окна

    print(f"\n[Оконные функции, N = {N}]")
    print()

    windows = {
        "Rectangular": rectangular_window(N),
        "Hann":        hann_window(N),
        "Hamming":     hamming_window(N),
        "Blackman":    blackman_window(N),
    }

    # Визуализация окон
    for name, w in windows.items():
        print(f"  {name} window:")
        max_w = max(w)
        for i in range(0, N, 4):
            bar = draw_ascii_bar(w[i], 1.0, 32)
            print(f"    [{i:2d}]: {bar} {w[i]:.3f}")
        print()

    # Сравнение свойств
    print("  Сравнение свойств:")
    print(f"  {'Окно':<14} {'Энергия':>10} {'Макс':>8} {'Мин':>8}")
    print("  " + "-" * 44)
    for name, w in windows.items():
        energy = sum(v * v for v in w)
        print(f"  {name:<14} {energy:>10.2f} {max(w):>8.4f} {min(w):>8.4f}")

    # Эффект оконнания на спектр
    sr = 1000
    dur = 0.064
    n = int(sr * dur)
    freq = 100  # 100 Гц

    signal = generate_sine(freq, sr, dur)
    print(f"\n[Эффект оконнания на спектр]")
    print(f"  Сигнал: {freq} Гц, {sr} Гц, {n} отсчётов")

    windows_funcs = {
        "Rectangular": rectangular_window,
        "Hann":        hann_window,
        "Hamming":     hamming_window,
        "Blackman":    blackman_window,
    }

    for name, wf in windows_funcs.items():
        windowed = apply_window(signal, wf)
        freqs, mags = magnitude_spectrum(windowed, sr)

        # Находим основной пик
        peak_idx = 0
        for k in range(len(mags)):
            if mags[k] > mags[peak_idx]:
                peak_idx = k

        # Спектральная утечка — сумма боковых лепестков
        side_lobe_sum = sum(mags) - mags[peak_idx]

        print(f"\n  [{name}]")
        print(f"    Пик: {freqs[peak_idx]:.1f} Гц, ампл. = {mags[peak_idx]:.4f}")
        print(f"    Боковые лепестки (суммарная энергия): {side_lobe_sum:.4f}")

        # Визуализация спектра (первые 20 бинов)
        max_mag = max(mags[:20])
        for k in range(min(20, len(freqs))):
            if freqs[k] > 0:
                print(f"      {freqs[k]:5.1f} Гц  |{draw_ascii_bar(mags[k], max_mag, 30)}| {mags[k]:.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────
# Основная программа
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    ОСНОВЫ ОБРАБОТКИ АУДИО СИГНАЛОВ НА PYTHON                  ║")
    print("║    (без numpy, scipy, librosa, torch)                         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    demo1_signal_generation()
    demo2_spectral_analysis()
    demo3_filtering()
    demo4_windowing()

    print("=" * 70)
    print("Все демонстрации завершены.")
    print("=" * 70)
