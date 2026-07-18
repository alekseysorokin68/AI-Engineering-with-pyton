"""
Аугментация аудио: методы без внешних зависимостей
===================================================
Самодостаточный модуль — только stdlib Python.

Методы:
  1. Добавление гауссова шума
  2. Time stretching (изменение скорости)
  3. Pitch shifting (изменение высоты тона)
  4. Скользящее среднее (smoothing)
"""

import math
import random


# ──────────────────────────── Вспомогательные функции ────────────────────────────

def generate_sine_wave(freq: float, duration: float, sample_rate: int) -> list[float]:
    """Генерация синусоиды заданной частоты."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        samples.append(math.sin(2 * math.pi * freq * t))
    return samples


def generate_combined_wave(durations_freqs: list[tuple[float, float]], sample_rate: int) -> list[float]:
    """Генерация составного сигнала из нескольких синусоид."""
    samples = []
    for duration, freq in durations_freqs:
        n_samples = int(sample_rate * duration)
        for i in range(n_samples):
            t = i / sample_rate
            samples.append(math.sin(2 * math.pi * freq * t))
    return samples


def rms_energy(samples: list[float]) -> float:
    """Корень из среднего квадрата (RMS) сигнала."""
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def normalize(samples: list[float], target_rms: float = 0.1) -> list[float]:
    """Нормализация сигнала к целевому RMS."""
    current_rms = rms_energy(samples)
    if current_rms < 1e-10:
        return samples[:]
    scale = target_rms / current_rms
    return [s * scale for s in samples]


def signal_info(name: str, samples: list[float], sample_rate: int) -> dict:
    """Информация о сигнале: длина, RMS, min, max."""
    return {
        "name": name,
        "samples": len(samples),
        "duration_s": round(len(samples) / sample_rate, 4),
        "rms": round(rms_energy(samples), 6),
        "min": round(min(samples), 6),
        "max": round(max(samples), 6),
    }


# ──────────────────────── 1. Добавление гауссова шума ────────────────────────

def add_gaussian_noise(samples: list[float], snr_db: float = 20.0) -> list[float]:
    """
    Добавление аддитивного гауссова шума.

    snr_db — целевое отношение сигнал/шум в дБ.
    """
    signal_power = sum(s * s for s in samples) / len(samples) if samples else 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = math.sqrt(max(noise_power, 1e-20))
    noisy = []
    for s in samples:
        noise = random.gauss(0, noise_std)
        noisy.append(s + noise)
    return noisy


# ──────────────────────── 2. Time stretching ────────────────────────

def time_stretch(samples: list[float], rate: float) -> list[float]:
    """
    Изменение скорости без изменения высоты тона.

    rate < 1  → замедление (длиннее)
    rate > 1  → ускорение (короче)
    Метод: линейная интерполяция с растяжением/сжатием.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    new_len = int(len(samples) / rate)
    stretched = []
    for i in range(new_len):
        src_pos = i * rate
        idx0 = int(src_pos)
        idx1 = min(idx0 + 1, len(samples) - 1)
        frac = src_pos - idx0
        val = samples[idx0] * (1.0 - frac) + samples[idx1] * frac
        stretched.append(val)
    return stretched


# ──────────────────────── 3. Pitch shifting ────────────────────────

def pitch_shift_resample(samples: list[float], semitones: float) -> list[float]:
    """
    Изменение высоты тона методом ресемплинга.

    semitones > 0 → выше
    semitones < 0 → ниже
    Метод: ресемплинг с последующим выравниванием длины.
    """
    factor = 2 ** (semitones / 12.0)
    # Ресемплинг: берём через интервал
    new_len = int(len(samples) * factor)
    shifted = []
    for i in range(new_len):
        src_pos = i / factor
        idx0 = int(src_pos)
        idx1 = min(idx0 + 1, len(samples) - 1)
        frac = src_pos - idx0
        val = samples[idx0] * (1.0 - frac) + samples[idx1] * frac
        shifted.append(val)
    return shifted


# ──────────────────────── 4. Скользящее среднее ────────────────────────

def sliding_average(samples: list[float], window_size: int) -> list[float]:
    """
    Сглаживание сигнала скользящим средним.

    window_size — размер окна (нечётное предпочтительно).
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    smoothed = []
    half_w = window_size // 2
    for i in range(len(samples)):
        start = max(0, i - half_w)
        end = min(len(samples), i + half_w + 1)
        window = samples[start:end]
        avg = sum(window) / len(window)
        smoothed.append(avg)
    return smoothed


# ──────────────────────── Демо ────────────────────────

def demo_1_noise():
    """Демо 1: Добавление гауссова шума."""
    print("=" * 70)
    print("ДЕМО 1: Добавление гауссова шума")
    print("=" * 70)

    sr = 8000
    signal = generate_sine_wave(freq=440, duration=0.5, sample_rate=sr)
    signal = normalize(signal, target_rms=0.3)

    snr_values = [40, 20, 10, 0]
    for snr in snr_values:
        noisy = add_gaussian_noise(signal, snr_db=snr)
        info = signal_info(f"SNR={snr}dB", noisy, sr)
        print(f"  SNR={snr:>3d}dB → len={info['samples']}, "
              f"rms={info['rms']:.4f}, "
              f"range=[{info['min']:.4f}, {info['max']:.4f}]")

    # Демонстрация разброса шума
    random.seed(42)
    noisy_20 = add_gaussian_noise(signal, snr_db=20)
    diff = [noisy_20[i] - signal[i] for i in range(len(signal))]
    print(f"  Шум при SNR=20dB: std≈{rms_energy(diff):.4f}")
    print()


def demo_2_time_stretch():
    """Демо 2: Time stretching."""
    print("=" * 70)
    print("ДЕМО 2: Time stretching (изменение скорости)")
    print("=" * 70)

    sr = 8000
    signal = generate_combined_wave(
        [(0.15, 440), (0.15, 880), (0.15, 660)],
        sample_rate=sr
    )
    signal = normalize(signal, target_rms=0.3)
    orig_info = signal_info("Оригинал", signal, sr)
    print(f"  Оригинал: len={orig_info['samples']}, dur={orig_info['duration_s']}s")

    rates = [0.5, 0.75, 1.0, 1.5, 2.0]
    for rate in rates:
        stretched = time_stretch(signal, rate)
        info = signal_info(f"rate={rate}", stretched, sr)
        print(f"  rate={rate:<4} → len={info['samples']:>5}, "
              f"dur={info['duration_s']:.3f}s, "
              f"rms={info['rms']:.4f}")

    print()


def demo_3_pitch_shift():
    """Демо 3: Pitch shifting."""
    print("=" * 70)
    print("ДЕМО 3: Pitch shifting (изменение высоты тона)")
    print("=" * 70)

    sr = 8000
    signal = generate_sine_wave(freq=440, duration=0.3, sample_rate=sr)
    signal = normalize(signal, target_rms=0.3)
    orig_info = signal_info("Оригинал (440Hz)", signal, sr)
    print(f"  Оригинал: len={orig_info['samples']}, "
          f"dur={orig_info['duration_s']}s")

    semitone_values = [-12, -7, -5, 0, 5, 7, 12]
    for sem in semitone_values:
        shifted = pitch_shift_resample(signal, sem)
        info = signal_info(f"{sem:+d} semitones", shifted, sr)
        expected_freq = 440 * (2 ** (sem / 12.0))
        print(f"  {sem:>+3d} semitones → len={info['samples']:>5}, "
              f"dur={info['duration_s']:.4f}s, "
              f"ожид. частота≈{expected_freq:.1f}Hz")

    print()


def demo_4_comparison():
    """Демо 4: Сравнение методов аугментации."""
    print("=" * 70)
    print("ДЕМО 4: Сравнение методов аугментации")
    print("=" * 70)

    sr = 8000
    signal = generate_combined_wave(
        [(0.1, 440), (0.1, 880), (0.1, 550), (0.1, 1100)],
        sample_rate=sr
    )
    signal = normalize(signal, target_rms=0.3)
    orig_info = signal_info("Оригинал", signal, sr)

    print(f"  Оригинал: {orig_info['samples']} samples, "
          f"{orig_info['duration_s']}s, rms={orig_info['rms']:.4f}\n")

    # Сравнение
    methods = [
        ("+ шум SNR=15dB", lambda s: add_gaussian_noise(s, snr_db=15)),
        ("×0.75 speed",    lambda s: time_stretch(s, 0.75)),
        ("×1.33 speed",    lambda s: time_stretch(s, 1.33)),
        ("+7 semitones",   lambda s: pitch_shift_resample(s, 7)),
        ("-5 semitones",   lambda s: pitch_shift_resample(s, -5)),
        ("smooth w=5",     lambda s: sliding_average(s, 5)),
        ("smooth w=15",    lambda s: sliding_average(s, 15)),
    ]

    print(f"  {'Метод':<20} {'Samples':>8} {'Duration':>10} {'RMS':>8} {'ΔLen':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for name, aug_fn in methods:
        augmented = aug_fn(signal)
        info = signal_info(name, augmented, sr)
        delta_len = len(augmented) - len(signal)
        print(f"  {name:<20} {info['samples']:>8} {info['duration_s']:>10.4f} "
              f"{info['rms']:>8.4f} {delta_len:>+8}")

    print()

    # Комбинированная аугментация
    print("  --- Комбинированная аугментация ---")
    combined = signal[:]
    combined = add_gaussian_noise(combined, snr_db=20)
    combined = time_stretch(combined, 0.9)
    combined = pitch_shift_resample(combined, 3)
    combined = sliding_average(combined, 3)
    info = signal_info("pipeline", combined, sr)
    print(f"  noise → stretch(0.9) → pitch(+3) → smooth(3)")
    print(f"  Результат: {info['samples']} samples, "
          f"{info['duration_s']:.4f}s, rms={info['rms']:.4f}")
    print()


# ──────────────────────── Точка входа ────────────────────────

if __name__ == "__main__":
    random.seed(42)
    print("Аугментация аудио — самодостаточный модуль (stdlib only)")
    print(f"Используется Python random с seed=42\n")

    demo_1_noise()
    demo_2_time_stretch()
    demo_3_pitch_shift()
    demo_4_comparison()

    print("Готово.")
