"""
Speaker Verification — основы верификации говорящего.

Самодостаточный файл: не требует numpy, scipy, librosa, torch.
Использует только random, math и стандартную библиотеку Python.
"""

import math
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def magnitude(v):
    return math.sqrt(dot(v, v))


def cosine_similarity(a, b):
    mag_a, mag_b = magnitude(a), magnitude(b)
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot(a, b) / (mag_a * mag_b)


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def stdev(lst):
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst)) if len(lst) > 1 else 0.0


def normalize(v):
    mag = magnitude(v)
    if mag < 1e-12:
        return [0.0] * len(v)
    return [x / mag for x in v]


# ─────────────────────────────────────────────────────────────
# 1. Извлечение признаков (MFCC, energy)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_audio(duration_sec=1.0, sample_rate=16000, speaker_id=0):
    """
    Генерация синтетического аудиосигнала для демонстрации.
    speaker_id влияет на частоты и амплитуду (индивидуальные «форманты»).
    """
    n_samples = int(duration_sec * sample_rate)
    freq_base = 100.0 + speaker_id * 15.0  # базовая частота
    harmonic_ratio = 1.5 + speaker_id * 0.2  # соотношение гармоник
    amplitude = 0.5 + speaker_id * 0.05

    signal = []
    for i in range(n_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * freq_base * t)
        value += (amplitude * 0.4) * math.sin(2 * math.pi * freq_base * harmonic_ratio * t)
        value += (amplitude * 0.2) * math.sin(2 * math.pi * freq_base * 2.5 * t)
        # добавляем шум
        value += random.gauss(0, 0.05)
        signal.append(value)
    return signal


def compute_energy(signal, frame_size=256):
    """Вычисление энергии по фреймам."""
    energies = []
    for i in range(0, len(signal) - frame_size + 1, frame_size // 2):
        frame = signal[i:i + frame_size]
        energy = sum(x ** 2 for x in frame) / frame_size
        energies.append(energy)
    return energies


def discrete_cosine_transform(frame, n_coeffs=13):
    """Упрощённое DCT-II для получения MFCC-подобных коэффициентов."""
    n = len(frame)
    coeffs = []
    for k in range(n_coeffs):
        s = 0.0
        for i in range(n):
            s += frame[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        coeffs.append(s)
    return coeffs


def compute_mfcc(signal, n_mfcc=13, frame_size=256):
    """Вычисление MFCC-подобных коэффициентов по фреймам."""
    mfcc_frames = []
    for i in range(0, len(signal) - frame_size + 1, frame_size // 2):
        frame = signal[i:i + frame_size]
        # оконная функция (Хэмминг)
        windowed = [frame[j] * (0.54 - 0.46 * math.cos(2 * math.pi * j / (frame_size - 1)))
                     for j in range(frame_size)]
        coeffs = discrete_cosine_transform(windowed, n_mfcc)
        mfcc_frames.append(coeffs)
    return mfcc_frames


def extract_features(signal):
    """
    Извлечение вектора признаков из аудиосигнала.
    Возвращает: усреднённые MFCC + средняя энергия + std энергии.
    """
    energies = compute_energy(signal)
    mfcc_frames = compute_mfcc(signal)

    # усреднение MFCC по фреймам
    n_mfcc = len(mfcc_frames[0]) if mfcc_frames else 13
    avg_mfcc = []
    for c in range(n_mfcc):
        col = [frame[c] for frame in mfcc_frames]
        avg_mfcc.append(mean(col))

    # энергетические признаки
    avg_energy = mean(energies)
    std_energy = stdev(energies)

    # объединяем всё в один вектор
    feature_vector = avg_mfcc + [avg_energy, std_energy]
    return feature_vector, mfcc_frames, energies


# ─────────────────────────────────────────────────────────────
# 2. Сравнение говорящих (cosine similarity)
# ─────────────────────────────────────────────────────────────

def compare_speakers(features_a, features_b):
    """Сравнение двух векторов признаков."""
    norm_a = normalize(features_a)
    norm_b = normalize(features_b)
    return cosine_similarity(norm_a, norm_b)


def build_speaker_profile(signals):
    """
    Построение профиля говорящего по нескольким записям.
    Профиль = усреднённый нормализованный вектор признаков.
    """
    all_features = [extract_features(sig)[0] for sig in signals]
    n_feat = len(all_features[0])
    profile = [0.0] * n_feat
    for feat in all_features:
        norm = normalize(feat)
        for i in range(n_feat):
            profile[i] += norm[i]
    return [x / len(all_features) for x in profile]


# ─────────────────────────────────────────────────────────────
# 3. Пороговые решения
# ─────────────────────────────────────────────────────────────

def verify_speaker(score, threshold=0.85):
    """Решение: один говорящий или разные."""
    if score >= threshold:
        return "IDENTICAL", score - threshold
    else:
        return "DIFFERENT", threshold - score


def compute_eer(scores_genuine, scores_impostor):
    """
    Equal Error Rate — порог, при котором FAR = FRR.
    """
    all_thresholds = sorted(set(scores_genuine + scores_impostor))
    best_eer = 1.0
    best_thr = 0.0

    for thr in all_thresholds:
        far = sum(1 for s in scores_impostor if s >= thr) / len(scores_impostor) if scores_impostor else 0
        frr = sum(1 for s in scores_genuine if s < thr) / len(scores_genuine) if scores_genuine else 0
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer = eer
            best_thr = thr
            if far == frr:
                break

    return best_eer, best_thr


def compute_far_frr(scores_genuine, scores_impostor, threshold):
    """Вычисление FAR и FRR для заданного порога."""
    far = sum(1 for s in scores_impostor if s >= threshold) / len(scores_impostor) if scores_impostor else 0
    frr = sum(1 for s in scores_impostor if s < threshold) / len(scores_impostor) if scores_impostor else 0
    frr_genuine = sum(1 for s in scores_genuine if s < threshold) / len(scores_genuine) if scores_genuine else 0
    return far, frr_genuine


# ─────────────────────────────────────────────────────────────
# Демо
# ─────────────────────────────────────────────────────────────

def demo_1_feature_extraction():
    print("=" * 60)
    print("ДЕМО 1: Извлечение признаков из аудио")
    print("=" * 60)

    speaker_id = 0
    signal = generate_synthetic_audio(duration_sec=0.5, speaker_id=speaker_id)
    feature_vector, mfcc_frames, energies = extract_features(signal)

    print(f"  Длина сигнала:       {len(signal)} сэмплов")
    print(f"  Количество фреймов:  {len(mfcc_frames)}")
    print(f"  MFCC коэффициенты:   {len(feature_vector) - 2}")
    print(f"  Размер вектора:      {len(feature_vector)}")
    print()
    print("  Первые 5 MFCC (усреднённые):")
    for i, val in enumerate(feature_vector[:5]):
        print(f"    MFCC[{i}]: {val:+.4f}")
    print()
    print(f"  Средняя энергия:     {feature_vector[-2]:.6f}")
    print(f"  Std энергии:         {feature_vector[-1]:.6f}")
    print()


def demo_2_speaker_comparison():
    print("=" * 60)
    print("ДЕМО 2: Сравнение двух говорящих")
    print("=" * 60)

    # один и тот же говорящий (разные записи)
    signals_same_1 = [generate_synthetic_audio(0.5, speaker_id=1) for _ in range(3)]
    signals_same_2 = [generate_synthetic_audio(0.5, speaker_id=1) for _ in range(3)]

    profile_a = build_speaker_profile(signals_same_1)
    profile_b = build_speaker_profile(signals_same_2)

    score_same = compare_speakers(profile_a, profile_b)
    verdict_same, margin_same = verify_speaker(score_same)

    print(f"  Говорящий A vs Говорящий A (разные записи)")
    print(f"    Cosine similarity: {score_same:+.4f}")
    print(f"    Решение:           {verdict_same} (запас: {margin_same:+.4f})")
    print()

    # разные говорящие
    signals_diff_1 = [generate_synthetic_audio(0.5, speaker_id=1) for _ in range(3)]
    signals_diff_2 = [generate_synthetic_audio(0.5, speaker_id=5) for _ in range(3)]

    profile_c = build_speaker_profile(signals_diff_1)
    profile_d = build_speaker_profile(signals_diff_2)

    score_diff = compare_speakers(profile_c, profile_d)
    verdict_diff, margin_diff = verify_speaker(score_diff)

    print(f"  Говорящий A vs Говорящий B (разные)")
    print(f"    Cosine similarity: {score_diff:+.4f}")
    print(f"    Решение:           {verdict_diff} (запас: {margin_diff:+.4f})")
    print()


def demo_3_threshold_classification():
    print("=" * 60)
    print("ДЕМО 3: Порог классификации")
    print("=" * 60)

    # генерируем score'ы
    genuine_scores = []
    for _ in range(20):
        sig1 = generate_synthetic_audio(0.5, speaker_id=3)
        sig2 = generate_synthetic_audio(0.5, speaker_id=3)
        feat1 = extract_features(sig1)[0]
        feat2 = extract_features(sig2)[0]
        genuine_scores.append(compare_speakers(feat1, feat2))

    impostor_scores = []
    for _ in range(20):
        sig1 = generate_synthetic_audio(0.5, speaker_id=3)
        sig2 = generate_synthetic_audio(0.5, speaker_id=7)
        feat1 = extract_features(sig1)[0]
        feat2 = extract_features(sig2)[0]
        impostor_scores.append(compare_speakers(feat1, feat2))

    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    print(f"  {'Порог':>8} | {'True Accept':>12} | {'False Reject':>13} | {'False Accept':>13} | {'True Reject':>12}")
    print("  " + "-" * 65)
    for thr in thresholds:
        ta = sum(1 for s in genuine_scores if s >= thr)
        fr = sum(1 for s in genuine_scores if s < thr)
        fa = sum(1 for s in impostor_scores if s >= thr)
        tr = sum(1 for s in impostor_scores if s < thr)
        print(f"  {thr:>8.2f} | {ta:>12} / 20 | {fr:>13} / 20 | {fa:>13} / 20 | {tr:>12} / 20")
    print()


def demo_4_accuracy_evaluation():
    print("=" * 60)
    print("ДЕМО 4: Оценка точности")
    print("=" * 60)

    # данные для оценки
    genuine_scores = []
    for _ in range(30):
        sid = random.choice(range(5))
        sig1 = generate_synthetic_audio(0.5, speaker_id=sid)
        sig2 = generate_synthetic_audio(0.5, speaker_id=sid)
        feat1 = extract_features(sig1)[0]
        feat2 = extract_features(sig2)[0]
        genuine_scores.append(compare_speakers(feat1, feat2))

    impostor_scores = []
    for _ in range(30):
        sid1, sid2 = random.sample(range(10), 2)
        sig1 = generate_synthetic_audio(0.5, speaker_id=sid1)
        sig2 = generate_synthetic_audio(0.5, speaker_id=sid2)
        feat1 = extract_features(sig1)[0]
        feat2 = extract_features(sig2)[0]
        impostor_scores.append(compare_speakers(feat1, feat2))

    # EER
    eer, eer_thr = compute_eer(genuine_scores, impostor_scores)
    print(f"  Equal Error Rate (EER): {eer:.4f}")
    print(f"  EER порог:             {eer_thr:.4f}")
    print()

    # метрики при EER-пороге
    far, frr = compute_far_frr(genuine_scores, impostor_scores, eer_thr)
    accuracy = 1.0 - (far + frr) / 2.0
    print(f"  При пороге {eer_thr:.4f}:")
    print(f"    FAR (False Accept Rate): {far:.4f}")
    print(f"    FRR (False Reject Rate): {frr:.4f}")
    print(f"    Точность (Accuracy):     {accuracy:.4f}")
    print()

    # статистика
    print(f"  Статистика скоров (genuine):")
    print(f"    min={min(genuine_scores):.4f}  max={max(genuine_scores):.4f}  "
          f"mean={mean(genuine_scores):.4f}  std={stdev(genuine_scores):.4f}")
    print(f"  Статистика скоров (impostor):")
    print(f"    min={min(impostor_scores):.4f}  max={max(impostor_scores):.4f}  "
          f"mean={mean(impostor_scores):.4f}  std={stdev(impostor_scores):.4f}")
    print()


# ─────────────────────────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("Speaker Verification — Основы верификации говорящего")
    print("Самодостаточный файл (без внешних зависимостей)")
    print()

    demo_1_feature_extraction()
    demo_2_speaker_comparison()
    demo_3_threshold_classification()
    demo_4_accuracy_evaluation()

    print("=" * 60)
    print("Все демонстрации завершены.")
    print("=" * 60)
