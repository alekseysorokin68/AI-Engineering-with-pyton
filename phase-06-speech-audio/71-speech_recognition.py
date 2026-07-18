"""
71 — Основы распознавания речи на Python (самодостаточный файл)

Реализуем с нуля:
  • MFCC-признаки (MFCC — Mel-Frequency Cepstral Coefficients)
  • DTW  (Dynamic Time Warping) для выравнивания и сравнения аудиосигналов
  • HMM  (Hidden Markov Model) для предсказания скрытых состояний

Все依赖и — только Python stdlib (math, random, statistics, collections).
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
#  0. УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def log2(x: float) -> float:
    """log2 без импорта math.log2 на старых питонах."""
    if x <= 0:
        return -100.0
    return math.log(x) / math.log(2)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DFT / FFT  (реализация с нуля)
# ═══════════════════════════════════════════════════════════════════════════════

def dft(x: List[float]) -> List[complex]:
    """Дискретное преобразование Фурье O(N²)."""
    N = len(x)
    result: List[complex] = []
    for k in range(N):
        s = 0 + 0j
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            s += x[n] * complex(math.cos(angle), math.sin(angle))
        result.append(s)
    return result


def fft_radix2(x: List[float]) -> List[complex]:
    """Быстрое преобразование Фурье (radix-2, рекурсия), O(N log N).
    Длина N должна быть степенью 2; иначе падаем обратно на DFT."""
    N = len(x)
    if N <= 1:
        return [complex(v, 0) for v in x]
    if N & (N - 1) != 0:          # не степень двойки
        return dft(x)

    even = fft_radix2(x[0::2])
    odd  = fft_radix2(x[1::2])
    T = [complex(math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N))
         * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]


def power_spectrum(frame: List[float], n_fft: int) -> List[float]:
    """Спектр мощности (односторонний)."""
    # zero-pad / обрезаем
    padded = frame[:n_fft] + [0.0] * max(0, n_fft - len(frame))
    spec = fft_radix2(padded)
    mag = [abs(s) for s in spec[:n_fft // 2 + 1]]
    return [m * m / n_fft for m in mag]


# ═══════════════════════════════════════════════════════════════════════════════
#  2. МЕЛОВАЯ ПЕРЕМЕННАЯ ШКАЛА + MEL-ФИЛЬТРЫ
# ═══════════════════════════════════════════════════════════════════════════════

def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> List[List[float]]:
    """Создать набор треугольных мел-фильтров.
    Возвращает матрицу [n_filters][n_fft//2 + 1]."""
    low_mel  = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = [low_mel + i * (high_mel - low_mel) / (n_filters + 1)
                  for i in range(n_filters + 2)]
    hz_points  = [mel_to_hz(m) for m in mel_points]
    bin_points = [int(math.floor((n_fft + 1) * f / sample_rate)) for f in hz_points]

    filters: List[List[float]] = []
    for i in range(n_filters):
        filt = [0.0] * (n_fft // 2 + 1)
        left   = bin_points[i]
        center = bin_points[i + 1]
        right  = bin_points[i + 2]
        for j in range(left, min(center + 1, len(filt))):
            if center != left:
                filt[j] = (j - left) / (center - left)
        for j in range(center, min(right + 1, len(filt))):
            if right != center:
                filt[j] = (right - j) / (right - center)
        filters.append(filt)
    return filters


# ═══════════════════════════════════════════════════════════════════════════════
#  3. ДКП (DCT-II) для перехода от лог-мел-энергии к MFCC
# ═══════════════════════════════════════════════════════════════════════════════

def dct2(x: List[float]) -> List[float]:
    """DCT-II (orthogonal normalisation omitted for simplicity)."""
    N = len(x)
    result: List[float] = []
    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        result.append(s)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MFCC ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ═══════════════════════════════════════════════════════════════════════════════

def pre_emphasis(signal: List[float], coeff: float = 0.97) -> List[float]:
    """Предварительное усиление высоких частот."""
    return [signal[0]] + [signal[i] - coeff * signal[i - 1]
                          for i in range(1, len(signal))]


def framing(signal: List[float], frame_len: int, hop_len: int) -> List[List[float]]:
    """Разбить сигнал на перекрывающиеся фреймы."""
    frames: List[List[float]] = []
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frames.append(signal[start:start + frame_len])
    return frames


def hamming_window(n: int) -> List[float]:
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def mfcc(signal: List[float], sample_rate: int = 16000,
         n_mfcc: int = 13, n_mels: int = 26,
         n_fft: int = 512, frame_len_ms: float = 25.0,
         hop_len_ms: float = 10.0) -> List[List[float]]:
    """
    Извлечь MFCC-признаки из аналогового сигнала.

    Параметры
    ----------
    signal      : отсчёты (float, нормализованные -1..1)
    sample_rate : частота дискретизации, Гц
    n_mfcc      : количество коэффициентов (обычно 13)
    n_mels      : число мел-фильтров (обычно 26)
    n_fft       : размер окна FFT (степень 2)
    frame_len_ms: длина фрейма, мс
    hop_len_ms  : шаг фрейма, мс

    Возвращает
    ----------
    Список фреймов, каждый — список из n_mfcc значений.
    """
    frame_len = int(sample_rate * frame_len_ms / 1000)
    hop_len   = int(sample_rate * hop_len_ms / 1000)

    sig = pre_emphasis(signal)
    frames = framing(sig, frame_len, hop_len)

    window = hamming_window(frame_len)
    filters = mel_filterbank(n_mels, n_fft, sample_rate)

    mfcc_frames: List[List[float]] = []
    for frame in frames:
        # применяем окно
        windowed = [f * w for f, w in zip(frame, window)]
        # спектр мощности
        ps = power_spectrum(windowed, n_fft)
        # мел-энергия
        mel_energies = []
        for filt in filters:
            e = sum(p * f for p, f in zip(ps, filt))
            mel_energies.append(math.log(e + 1e-10))
        # DCT → MFCC
        cep = dct2(mel_energies)[:n_mfcc]
        mfcc_frames.append(cep)

    return mfcc_frames


# ═══════════════════════════════════════════════════════════════════════════════
#  5. DTW (DYNAMIC TIME WARPING)
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dtw(seq1: List[List[float]], seq2: List[List[float]],
        dist_fn=euclidean_distance) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Dynamic Time Warping.

    Возвращает (cost, path) где path — список пар индексов (i, j).
    """
    n, m = len(seq1), len(seq2)
    # матрица стоимостей
    cost = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    # матрица предков для восстановления пути
    prev = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = dist_fn(seq1[i - 1], seq2[j - 1])
            candidates = [
                (cost[i - 1][j]     + d, (i - 1, j)),
                (cost[i][j - 1]     + d, (i, j - 1)),
                (cost[i - 1][j - 1] + d, (i - 1, j - 1)),
            ]
            best, parent = min(candidates, key=lambda x: x[0])
            cost[i][j] = best
            prev[i][j] = parent

    # восстанавливаем путь
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if prev[i][j] is not None:
            i, j = prev[i][j]
        else:
            break
    path.reverse()
    return cost[n][m], path


def dtw_distance_normalized(seq1: List[List[float]], seq2: List[List[float]]) -> float:
    """DTW-расстояние, нормализованное по длине пути."""
    cost, path = dtw(seq1, seq2)
    return cost / len(path) if path else float('inf')


# ═══════════════════════════════════════════════════════════════════════════════
#  6. HMM  (HIDDEN MARKOV MODEL) — Наивный реализация
# ═══════════════════════════════════════════════════════════════════════════════

class HMM:
    """
    Простая HMM с дискретными наблюдениями.

    Параметры
    ----------
    n_states : количество скрытых состояний
    n_obs    : количество различных наблюдений (алфавит)
    """

    def __init__(self, n_states: int, n_obs: int):
        self.n_states = n_states
        self.n_obs = n_obs
        # Инициализируем случайно, но нормализуем
        self.pi  = self._random_simplex(n_states)           # начальные вероятности
        self.A   = [self._random_simplex(n_states) for _ in range(n_states)]  # переходы
        self.B   = [self._random_simplex(n_obs) for _ in range(n_states)]     # эмиссии

    @staticmethod
    def _random_simplex(n: int) -> List[float]:
        vals = [random.random() for _ in range(n)]
        s = sum(vals)
        return [v / s for v in vals]

    # — Вывод (Forward algorithm) —

    def forward(self, obs: List[int]) -> List[List[float]]:
        """Алгоритм прямого прохода. Возвращает матрицу α[T][N]."""
        T = len(obs)
        N = self.n_states
        alpha = [[0.0] * N for _ in range(T)]

        # t = 0
        for i in range(N):
            alpha[0][i] = self.pi[i] * self.B[i][obs[0]]

        for t in range(1, T):
            for j in range(N):
                s = sum(alpha[t - 1][i] * self.A[i][j] for i in range(N))
                alpha[t][j] = s * self.B[j][obs[t]]
        return alpha

    def backward(self, obs: List[int]) -> List[List[float]]:
        """Алгоритм обратного прохода. Возвращает матрицу β[T][N]."""
        T = len(obs)
        N = self.n_states
        beta = [[0.0] * N for _ in range(T)]

        # t = T-1
        for i in range(N):
            beta[T - 1][i] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t][i] = sum(
                    self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j]
                    for j in range(N)
                )
        return beta

    def log_likelihood(self, obs: List[int]) -> float:
        """Log P(obs | модель) через forward-проход."""
        alpha = self.forward(obs)
        return math.log(sum(alpha[-1]) + 1e-300)

    # — Витерби (Viterbi) —

    def viterbi(self, obs: List[int]) -> Tuple[List[int], float]:
        """Алгоритм Витерби — наиболее вероятная последовательность состояний."""
        T = len(obs)
        N = self.n_states

        # log-вероятности для численной стабильности
        def log(x): return math.log(x + 1e-300)

        delta = [[0.0] * N for _ in range(T)]
        psi   = [[0]   * N for _ in range(T)]

        for i in range(N):
            delta[0][i] = log(self.pi[i]) + log(self.B[i][obs[0]])

        for t in range(1, T):
            for j in range(N):
                candidates = [
                    delta[t - 1][i] + log(self.A[i][j])
                    for i in range(N)
                ]
                best_i = candidates.index(max(candidates))
                psi[t][j] = best_i
                delta[t][j] = candidates[best_i] + log(self.B[j][obs[t]])

        # backtrack
        states = [0] * T
        states[T - 1] = delta[T - 1].index(max(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1][states[t + 1]]

        return states, delta[T - 1][states[T - 1]]

    # — Baum-Welch (EM) —

    def baum_welch(self, obs: List[int], n_iter: int = 20) -> float:
        """Обучение методом Baum-Welch. Возвращает финальный log-likelihood."""
        T = len(obs)
        N = self.n_states

        for iteration in range(n_iter):
            alpha = self.forward(obs)
            beta  = self.backward(obs)

            # γ(t, i)
            gamma = [[0.0] * N for _ in range(T)]
            for t in range(T):
                denom = sum(alpha[t][i] * beta[t][i] for i in range(N))
                for i in range(N):
                    gamma[t][i] = alpha[t][i] * beta[t][i] / (denom + 1e-300)

            # ξ(t, i, j)
            xi = [[[0.0] * N for _ in range(N)] for _ in range(T - 1)]
            for t in range(T - 1):
                denom = sum(
                    alpha[t][i] * self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j]
                    for i in range(N) for j in range(N)
                )
                for i in range(N):
                    for j in range(N):
                        xi[t][i][j] = (alpha[t][i] * self.A[i][j] *
                                        self.B[j][obs[t + 1]] * beta[t + 1][j]) / (denom + 1e-300)

            # обновление π
            for i in range(N):
                self.pi[i] = gamma[0][i]

            # обновление A
            for i in range(N):
                denom = sum(gamma[t][i] for t in range(T - 1))
                for j in range(N):
                    numer = sum(xi[t][i][j] for t in range(T - 1))
                    self.A[i][j] = numer / (denom + 1e-300)

            # обновление B
            for i in range(N):
                denom = sum(gamma[t][i] for t in range(T))
                for k in range(self.n_obs):
                    numer = sum(gamma[t][i] for t in range(T) if obs[t] == k)
                    self.B[i][k] = numer / (denom + 1e-300)

        return self.log_likelihood(obs)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. СИНТЕТИЧЕСКИЕ ГЕНЕРАТОРЫ СИГНАЛОВ
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sine(freq: float, duration: float, sample_rate: int = 16000,
                  noise_level: float = 0.0) -> List[float]:
    """Синусоида заданной частоты + опциональный шум."""
    n_samples = int(sample_rate * duration)
    signal = []
    for i in range(n_samples):
        t = i / sample_rate
        val = math.sin(2 * math.pi * freq * t)
        if noise_level > 0:
            val += random.gauss(0, noise_level)
        signal.append(val)
    return signal


def generate_chirp(f0: float, f1: float, duration: float,
                   sample_rate: int = 16000) -> List[float]:
    """Линейно возрастающая частота (chirp)."""
    n_samples = int(sample_rate * duration)
    signal = []
    for i in range(n_samples):
        t = i / sample_rate
        freq = f0 + (f1 - f0) * t / duration
        signal.append(math.sin(2 * math.pi * freq * t))
    return signal


def generate_dtmf(digit: str, duration: float = 0.15,
                  sample_rate: int = 16000) -> List[float]:
    """DTMF-тон (телефонная тональная сигнализация)."""
    dtmf_freqs = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
        '0': (941, 1336), '*': (941, 1209), '#': (941, 1477),
    }
    if digit not in dtmf_freqs:
        raise ValueError(f"Неизвестная цифра: {digit}")
    f_low, f_high = dtmf_freqs[digit]
    n_samples = int(sample_rate * duration)
    return [0.5 * (math.sin(2 * math.pi * f_low * i / sample_rate) +
                    math.sin(2 * math.pi * f_high * i / sample_rate))
            for i in range(n_samples)]


# ═══════════════════════════════════════════════════════════════════════════════
#  ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def demo_mfcc():
    """Демо 1: MFCC-извлечение признаков."""
    print("=" * 70)
    print("  ДЕМО 1 — MFCC (Mel-Frequency Cepstral Coefficients)")
    print("=" * 70)

    sr = 16000
    sig = generate_sine(440, 0.5, sr, noise_level=0.05)

    print(f"\n  Сигнал: синусоида 440 Гц, 0.5 с, Fs = {sr} Гц")
    print(f"  Отсчётов: {len(sig)}")

    frames = mfcc(sig, sample_rate=sr, n_mfcc=13)
    print(f"  Фреймов (после фрейминга): {len(frames)}")
    print(f"  Коэффициентов на фрейм: {len(frames[0])}")

    print("\n  Первые 5 фреймов (MFCC[0..12]):")
    for idx in range(min(5, len(frames))):
        vals = ", ".join(f"{v:8.4f}" for v in frames[idx])
        print(f"    Фрейм {idx}: [{vals}]")

    # Средние и стд по всем фреймам
    n_mfcc = len(frames[0])
    means = [statistics.mean(f[i] for f in frames) for i in range(n_mfcc)]
    stds  = [statistics.stdev(f[i] for f in frames) if len(frames) > 1 else 0
             for i in range(n_mfcc)]

    print("\n  Средние MFCC по всему сигналу:")
    for i in range(n_mfcc):
        print(f"    MFCC[{i:2d}]:  mean = {means[i]:8.4f}  std = {stds[i]:8.4f}")

    # Сравним MFCC для разных частот
    print("\n  Сравнение MFCC для разных частот тона:")
    for freq in [220, 440, 880, 1760]:
        s = generate_sine(freq, 0.3, sr)
        f = mfcc(s, sample_rate=sr, n_mfcc=6)
        avg = [statistics.mean(fr[i] for fr in f) for i in range(6)]
        vals = ", ".join(f"{v:7.3f}" for v in avg)
        print(f"    {freq:5d} Гц → MFCC[0..5] mean: [{vals}]")

    print()


def demo_dtw():
    """Демо 2: DTW — сравнение двух сигналов."""
    print("=" * 70)
    print("  ДЕМО 2 — DTW (Dynamic Time Warping)")
    print("=" * 70)

    sr = 16000

    # Генерируем «эталонный» сигнал и три «тестовых»
    ref = generate_sine(440, 0.3, sr)
    same_freq  = generate_sine(440, 0.35, sr)          # та же частота, чуть длиннее
    diff_freq  = generate_sine(880, 0.3, sr)           # другая частота
    chirp_sig  = generate_chirp(200, 800, 0.3, sr)     # чирп

    # Извлекаем MFCC
    ref_mfcc    = mfcc(ref, sr, n_mfcc=13)
    same_mfcc   = mfcc(same_freq, sr, n_mfcc=13)
    diff_mfcc   = mfcc(diff_freq, sr, n_mfcc=13)
    chirp_mfcc  = mfcc(chirp_sig, sr, n_mfcc=13)

    print(f"\n  Эталон: синусоида 440 Гц, 0.3 с → {len(ref_mfcc)} фреймов")
    print(f"  Тест 1: синусоида 440 Гц, 0.35 с → {len(same_mfcc)} фреймов")
    print(f"  Тест 2: синусоида 880 Гц, 0.3 с  → {len(diff_mfcc)} фреймов")
    print(f"  Тест 3: чирп 200→800 Гц, 0.3 с   → {len(chirp_mfcc)} фреймов")

    pairs = [
        ("тот же 440 Гц", same_mfcc),
        ("другой 880 Гц", diff_mfcc),
        ("чирп 200-800",  chirp_mfcc),
    ]

    print("\n  DTW-расстояния (нормализованные):")
    for name, test_mfcc in pairs:
        cost, path = dtw(ref_mfcc, test_mfcc)
        norm = cost / len(path) if path else float('inf')
        print(f"    {name:20s} → cost = {cost:10.2f}  norm = {norm:8.4f}  путь = {len(path)} шагов")

    # Детали выравнивания
    print("\n  Пример выравнивания (440 vs 440):")
    _, path = dtw(ref_mfcc, same_mfcc)
    for i, j in path[:10]:
        print(f"    ref[{i:3d}] ↔ test[{j:3d}]")
    print(f"    ... (всего {len(path)} пар)")

    print()


def demo_hmm():
    """Демо 3: HMM — предсказание скрытых состояний."""
    print("=" * 70)
    print("  ДЕМО 3 — HMM (Hidden Markov Model)")
    print("=" * 70)

    # Создаём модель: 3 скрытых состояния (холод/тепло/жара), 4 наблюдения
    hmm = HMM(n_states=3, n_obs=4)
    state_names = ["Холод", "Тепло", "Жара"]
    obs_names   = ["Куртка", "Рубашка", "Футболка", "Шорты"]

    print(f"\n  Модель: {hmm.n_states} скрытых состояний × {hmm.n_obs} наблюдений")
    print(f"  Состояния: {state_names}")
    print(f"  Наблюдения: {obs_names}")

    # Синтезируем «обучающую» последовательность
    # Паттерн: холод→тепло→жара→тепло→холод (по сезонам)
    hidden_seq = [0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0]
    # Наблюдения: на основе скрытых состояний
    obs_probs = {
        0: [0.6, 0.3, 0.1, 0.0],  # Холод → куртка/рубашка
        1: [0.1, 0.3, 0.4, 0.2],  # Тепло → рубашка/футболка
        2: [0.0, 0.1, 0.3, 0.6],  # Жара  → футболка/шорты
    }
    obs_seq = []
    for s in hidden_seq:
        r = random.random()
        cum = 0.0
        for k in range(4):
            cum += obs_probs[s][k]
            if r <= cum:
                obs_seq.append(k)
                break
        else:
            obs_seq.append(3)

    print(f"\n  Обучающая последовательность ({len(obs_seq)} шагов):")
    print(f"    Скрытые:  {hidden_seq}")
    print(f"    Наблюд.:  {obs_seq}")

    # Baum-Welch обучение
    print("\n  Baum-Welch обучение (20 итераций)...")
    ll = hmm.baum_welch(obs_seq, n_iter=20)
    print(f"  Финальный log-likelihood: {ll:.4f}")

    # Витерби
    states_pred, vit_score = hmm.viterbi(obs_seq)
    print(f"\n  Витерби предсказание:")
    print(f"    Предсказ.: {states_pred}")
    print(f"    Истинное : {hidden_seq}")
    # Точность
    correct = sum(1 for p, t in zip(states_pred, hidden_seq) if p == t)
    print(f"    Точность: {correct}/{len(hidden_seq)} = {correct/len(hidden_seq)*100:.1f}%")

    # Forward
    ll_after = hmm.log_likelihood(obs_seq)
    print(f"\n  Log-likelihood после обучения: {ll_after:.4f}")

    # Новая последовательность
    print("\n  Предсказание на новой последовательности:")
    new_obs = [0, 1, 2, 2, 1, 0, 0, 1, 2, 1]
    new_states, new_score = hmm.viterbi(new_obs)
    print(f"    Наблюдения: {new_obs}")
    print(f"    Состояния:  {new_states}")
    print(f"    Присвоения:")
    for t, (o, s) in enumerate(zip(new_obs, new_states)):
        print(f"      t={t:2d}: {obs_names[o]:10s} → {state_names[s]}")

    print()


def demo_command_recognition():
    """Демо 4: Простое распознавание голосовых команд."""
    print("=" * 70)
    print("  ДЕМО 4 — Простое распознавание команд (DTW + MFCC)")
    print("=" * 70)

    sr = 16000

    # Имитируем «команды» разными тональными паттернами:
    # «Вперёд»  — высокий тон  (440 Гц)
    # «Назад»   — низкий тон   (220 Гц)
    # «Стоп»    — быстро затухающий (чирп 800→200)
    # «Поворот» — пульс (440 Гц с паузами)

    def make_forward():
        return generate_sine(440, 0.25, sr)

    def make_back():
        return generate_sine(220, 0.25, sr)

    def make_stop():
        return generate_chirp(800, 200, 0.25, sr)

    def make_turn():
        """Пульсирующий тон."""
        sig = []
        for i in range(int(sr * 0.25)):
            t = i / sr
            env = max(0, math.sin(2 * math.pi * 8 * t))
            sig.append(env * math.sin(2 * math.pi * 440 * t))
        return sig

    commands = {
        "Вперёд":  make_forward,
        "Назад":   make_back,
        "Стоп":    make_stop,
        "Поворот": make_turn,
    }

    # Создаём эталоны (по 3 каждого для усреднения)
    print("\n  Создаём эталонные шаблоны (по 3 образца на команду)...")
    templates: Dict[str, List[List[List[float]]]] = {}
    for name, gen_fn in commands.items():
        templates[name] = []
        for _ in range(3):
            sig = gen_fn()
            feat = mfcc(sig, sr, n_mfcc=13)
            templates[name].append(feat)

    for name, tmpls in templates.items():
        avg_len = statistics.mean(len(t) for t in tmpls)
        print(f"    «{name:8s}» → {len(tmpls)} шаблонов, ~{avg_len:.0f} фреймов каждый")

    # Создаём «тестовые» команды (с небольшим шумом)
    def add_noise(sig, level=0.03):
        return [s + random.gauss(0, level) for s in sig]

    test_cases = [
        ("Вперёд (чистый)",  make_forward()),
        ("Назад (+ шум)",    add_noise(make_back())),
        ("Стоп (чистый)",    make_stop()),
        ("Поворот (+ шум)",  add_noise(make_turn())),
        ("Вперёд (чирп?!)",  generate_chirp(300, 600, 0.25, sr)),  # неизвестный
    ]

    print("\n  Распознавание команд:")
    print("  " + "-" * 66)

    for test_name, test_sig in test_cases:
        test_feat = mfcc(test_sig, sr, n_mfcc=13)

        # Сравниваем с каждым шаблоном, берём лучший
        best_cmd = None
        best_dist = float('inf')
        all_dists = {}

        for cmd_name, cmd_templates in templates.items():
            # Среднее DTW-расстояние до всех шаблонов
            dists = []
            for tmpl in cmd_templates:
                d = dtw_distance_normalized(tmpl, test_feat)
                dists.append(d)
            avg_d = statistics.mean(dists)
            all_dists[cmd_name] = avg_d
            if avg_d < best_dist:
                best_dist = avg_d
                best_cmd = cmd_name

        # Порог для «неизвестной» команды
        sorted_dists = sorted(all_dists.values())
        margin = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) > 1 else float('inf')
        confidence = "высокая" if margin > 0.1 else ("средняя" if margin > 0.05 else "низкая")

        dists_str = " | ".join(f"{k}: {v:.3f}" for k, v in sorted(all_dists.items()))
        print(f"\n  Тест: {test_name}")
        print(f"    Расстояния: {dists_str}")
        print(f"    → Результат: «{best_cmd}»  (маржа: {margin:.4f}, уверенность: {confidence})")

    print()
    print("  " + "=" * 66)
    print("  Вывод: DTW + MFCC позволяют различать простые тональные")
    print("  команды даже без нейросетей. Для реальных задач используют")
    print("  HMM/GMM + Language Model или end-to-end нейросети (Wav2Vec, Whisper).")
    print("  " + "=" * 66)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "▓" * 70)
    print("  71 — ОСНОВЫ РАСПОЗНАВАНИЯ РЕЧИ НА PYTHON")
    print("  Самодостаточный файл: только stdlib Python")
    print("▓" * 70 + "\n")

    demo_mfcc()
    demo_dtw()
    demo_hmm()
    demo_command_recognition()

    print("▓" * 70)
    print("  Все демо завершены!")
    print("▓" * 70)
