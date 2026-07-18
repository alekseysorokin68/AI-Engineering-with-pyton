"""
73. Текст-в-речь (Text-to-Speech) — основы синтеза речи
======================================================

Формантный синтез — метод синтеза речи, основанный на моделировании
формантных частот голосового тракта. Форманты — это области усиления
спектра звука, определяемые формой рта и положением языка.

Ключевые концепции:
- Форманты (F1, F2, F3...) — резонансные частоты голосового тракта
- Основной тон (F0) — частота вибраций голосовых связок
- Обертоновая структура — гармоники основного тона
- Фрикативные согласные — шумовые компоненты речи

Самодостаточный файл — не требует внешних библиотек (numpy, scipy и т.д.)
"""

import math
import random
import struct
import wave

# ============================================================================
# Утилиты
# ============================================================================

def sine_wave(frequency, duration, sample_rate=22050):
    """Генерация синусоидальной волны"""
    n_samples = int(duration * sample_rate)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        samples.append(math.sin(2 * math.pi * frequency * t))
    return samples


def harmonic_series(f0, n_harmonics, amplitudes=None):
    """Генерация ряда гармоник
    
    Args:
        f0: основная частота
        n_harmonics: количество гармоник
        amplitudes: список амплитуд для каждой гармоники (опционально)
    
    Returns:
        функция, генерирующая samples волны
    """
    if amplitudes is None:
        amplitudes = [1.0 / (i + 1) for i in range(n_harmonics)]
    
    def generator(duration, sample_rate=22050):
        n_samples = int(duration * sample_rate)
        samples = [0.0] * n_samples
        for h in range(n_harmonics):
            freq = f0 * (h + 1)
            amp = amplitudes[h] if h < len(amplitudes) else amplitudes[-1]
            for i in range(n_samples):
                t = i / sample_rate
                samples[i] += amp * math.sin(2 * math.pi * freq * t)
        return samples
    
    return generator


def apply_envelope(samples, attack=0.05, decay=0.1, sustain_level=0.7, release=0.1, sample_rate=22050):
    """Применение ADSR-огибачки к сэмплам
    
    Args:
        samples: список амплитуд
        attack: время нарастания (секунды)
        decay: время затухания (секунды)
        sustain_level: уровень сустейна (0.0 - 1.0)
        release: время затухания (секунды)
        sample_rate: частота дискретизации
    
    Returns:
        список сэмплов с применённой огибающей
    """
    n_samples = len(samples)
    envelope = [0.0] * n_samples
    
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = max(0, n_samples - attack_samples - decay_samples - release_samples)
    
    idx = 0
    
    # Attack
    for i in range(min(attack_samples, n_samples)):
        envelope[idx] = i / attack_samples if attack_samples > 0 else 1.0
        idx += 1
    
    # Decay
    for i in range(min(decay_samples, n_samples - idx)):
        t = i / decay_samples if decay_samples > 0 else 1.0
        envelope[idx] = 1.0 - (1.0 - sustain_level) * t
        idx += 1
    
    # Sustain
    for i in range(min(sustain_samples, n_samples - idx)):
        envelope[idx] = sustain_level
        idx += 1
    
    # Release
    for i in range(min(release_samples, n_samples - idx)):
        t = i / release_samples if release_samples > 0 else 1.0
        envelope[idx] = sustain_level * (1.0 - t)
        idx += 1
    
    # Заполнение оставшихся сэмплов нулями
    while idx < n_samples:
        envelope[idx] = 0.0
        idx += 1
    
    return [samples[i] * envelope[i] for i in range(n_samples)]


def add_noise(samples, amplitude=0.01):
    """Добавление случайного шума к сэмплам"""
    random.seed(42)
    return [s + random.uniform(-amplitude, amplitude) for s in samples]


def normalize(samples, target_peak=0.9):
    """Нормализация сэмплов к целевому пиковому значению"""
    peak = max(abs(s) for s in samples) if samples else 1.0
    if peak == 0:
        return samples
    return [s * target_peak / peak for s in samples]


def concatenate_sounds(*sound_lists, gap_ms=50, sample_rate=22050):
    """Конкатенация нескольких наборов сэмплов с паузами между ними
    
    Args:
        *sound_lists: наборы сэмплов для конкатенации
        gap_ms: длина паузы в миллисекундах
        sample_rate: частота дискретизации
    
    Returns:
        конкатенированный список сэмплов
    """
    gap_samples = int(gap_ms / 1000 * sample_rate)
    result = []
    for i, sound in enumerate(sound_lists):
        result.extend(sound)
        if i < len(sound_lists) - 1:
            result.extend([0.0] * gap_samples)
    return result


def save_wav(filename, samples, sample_rate=22050):
    """Сохранение сэмплов в WAV-файл"""
    n_samples = len(samples)
    n_channels = 1
    sample_width = 2  # 16-bit
    
    # Конвертация float [-1, 1] в int16
    int_samples = []
    for s in samples:
        clamped = max(-1.0, min(1.0, s))
        int_samples.append(int(clamped * 32767))
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        data = struct.pack(f'<{n_samples}h', *int_samples)
        wf.writeframes(data)


def rms(samples):
    """Вычисление RMS-значения (среднеквадратичная амплитуда)"""
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def peak(samples):
    """Вычисление пикового значения"""
    if not samples:
        return 0.0
    return max(abs(s) for s in samples)


def duration_sec(samples, sample_rate=22050):
    """Длительность сэмплов в секундах"""
    return len(samples) / sample_rate


def frequency_spectrum(samples, sample_rate=22050, n_bins=10):
    """Простой анализ спектра (разбиение на частотные полосы)"""
    n = len(samples)
    if n == 0:
        return [0.0] * n_bins
    
    # Разбиение на полосы
    bin_size = sample_rate // 2 // n_bins
    spectrum = [0.0] * n_bins
    
    for i in range(n):
        t = i / sample_rate
        freq = i * sample_rate / n if n > 0 else 0
        bin_idx = min(int(freq / bin_size), n_bins - 1)
        spectrum[bin_idx] += abs(samples[i])
    
    # Нормализация
    max_val = max(spectrum) if spectrum else 1.0
    if max_val > 0:
        spectrum = [v / max_val for v in spectrum]
    
    return spectrum


# ============================================================================
# ФОНАТИЧЕСКАЯ ТАБЛИЦА — параметры формант для русских фонем
# ============================================================================

# Форманты определяют качество гласных: F1 (открытость), F2 (передний/задний)
# F3 влияет на индивидуальные особенности голоса
VOWEL_FORMANTS = {
    # Фонема: (F0_базовый, F1, F2, F3, амплитуда)
    'А': (120, 800, 1200, 2500, 1.0),
    'О': (120, 500, 900, 2500, 0.9),
    'У': (120, 350, 800, 2500, 0.8),
    'Э': (120, 600, 1800, 2500, 0.85),
    'Ы': (120, 400, 1500, 2500, 0.75),
    'И': (120, 300, 2200, 3000, 0.9),
    'Е': (130, 400, 2000, 2800, 0.85),
    'Ё': (130, 450, 1600, 2600, 0.8),
}

# Параметры для согласных
CONSONANT_PARAMS = {
    # Фонема: (тип, F0, F1, F2, F3, амплитуда, длительность_мс)
    'Б': ('plosive', 100, 200, 1000, 2500, 0.6, 50),
    'В': ('fricative', 100, 300, 1500, 2500, 0.4, 60),
    'Г': ('plosive', 100, 250, 1100, 2500, 0.5, 50),
    'Д': ('plosive', 100, 300, 1700, 2500, 0.6, 50),
    'Ж': ('fricative', 100, 200, 1800, 2500, 0.5, 70),
    'З': ('fricative', 100, 250, 1600, 2500, 0.45, 65),
    'К': ('plosive', 100, 300, 1200, 2500, 0.5, 45),
    'Л': ('liquid', 120, 350, 1200, 2800, 0.55, 80),
    'М': ('nasal', 100, 250, 1000, 2500, 0.5, 70),
    'Н': ('nasal', 100, 300, 1500, 2500, 0.5, 65),
    'П': ('plosive', 100, 200, 1000, 2500, 0.6, 45),
    'Р': ('trill', 100, 300, 1500, 2500, 0.55, 70),
    'С': ('fricative', 100, 250, 5000, 7000, 0.5, 60),
    'Т': ('plosive', 100, 300, 1800, 2500, 0.6, 45),
    'Ф': ('fricative', 100, 250, 4500, 6500, 0.45, 55),
    'Х': ('fricative', 100, 300, 1500, 3000, 0.5, 60),
    'Ц': ('affricate', 100, 300, 1800, 2500, 0.5, 65),
    'Ч': ('affricate', 100, 250, 2000, 2800, 0.5, 55),
    'Ш': ('fricative', 100, 200, 2200, 3500, 0.55, 65),
    'Щ': ('fricative', 100, 250, 2400, 3500, 0.5, 70),
}


# ============================================================================
# ГЕНЕРАТОРЫ ЗВУКОВ
# ============================================================================

def generate_vowel_sound(phoneme, duration=0.3, sample_rate=22050, vibrato_depth=2.0):
    """Генерация звука гласной фонемы
    
    Args:
        phoneme: символ гласной (А, О, У и т.д.)
        duration: длительность в секундах
        sample_rate: частота дискретизации
        vibrato_depth: глубина вибрато (в Гц)
    
    Returns:
        список сэмплов
    """
    if phoneme not in VOWEL_FORMANTS:
        raise ValueError(f"Неизвестная гласная: {phoneme}")
    
    f0, f1, f2, f3, amp = VOWEL_FORMANTS[phoneme]
    n_samples = int(duration * sample_rate)
    samples = [0.0] * n_samples
    
    for i in range(n_samples):
        t = i / sample_rate
        
        # Модуляция основного тона (вибрато)
        vibrato = vibrato_depth * math.sin(2 * math.pi * 5.0 * t)
        current_f0 = f0 + vibrato
        
        # Основной тон + гармоники
        harmonic = 0.0
        for h in range(1, 10):
            amplitude = 1.0 / (h * 0.8)  # Более естественное затухание
            harmonic += amplitude * math.sin(2 * math.pi * current_f0 * h * t)
        
        # Формантные фильтры (упрощённая модель)
        # F1 резонанс
        f1_response = 1.0 / (1.0 + ((current_f0 - f1) / (f1 * 0.15)) ** 2)
        # F2 резонанс
        f2_response = 1.0 / (1.0 + ((current_f0 - f2) / (f2 * 0.12)) ** 2)
        # F3 резонанс
        f3_response = 0.5 / (1.0 + ((current_f0 - f3) / (f3 * 0.1)) ** 2)
        
        formant_sum = f1_response + f2_response + f3_response
        if formant_sum > 0:
            formant_sum /= 3
        
        samples[i] = amp * harmonic * (0.5 + 0.5 * formant_sum)
    
    # Применение огибачки
    samples = apply_envelope(samples, attack=0.03, decay=0.05, sustain_level=0.8, release=0.05, sample_rate=sample_rate)
    
    return samples


def generate_consonant_sound(phoneme, duration=None, sample_rate=22050):
    """Генерация звука согласной фонемы
    
    Args:
        phoneme: символ согласной
        duration: длительность (если None, берётся из параметров фонемы)
        sample_rate: частота дискретизации
    
    Returns:
        список сэмплов
    """
    if phoneme not in CONSONANT_PARAMS:
        raise ValueError(f"Неизвестная согласная: {phoneme}")
    
    con_type, f0, f1, f2, f3, amp, default_duration = CONSONANT_PARAMS[phoneme]
    
    if duration is None:
        duration = default_duration / 1000.0
    
    n_samples = int(duration * sample_rate)
    samples = [0.0] * n_samples
    
    if con_type == 'plosive':
        # Взрывные согласные: быстрое нарастание и затухание
        for i in range(n_samples):
            t = i / sample_rate
            # Шумовой компонент
            random.seed(42 + i % 1000)
            noise = random.uniform(-1, 1)
            # Экспоненциальное затухание
            decay = math.exp(-t * 50)
            samples[i] = amp * noise * decay
    
    elif con_type == 'fricative':
        # Фрикативные согласные: шум
        for i in range(n_samples):
            t = i / sample_rate
            random.seed(42 + i % 1000)
            noise = random.uniform(-1, 1)
            # Модуляция шума формантами
            mod = 1.0 + 0.5 * math.sin(2 * math.pi * f1 * t)
            samples[i] = amp * noise * mod * 0.7
    
    elif con_type == 'nasal':
        # Носовые согласные: основной тон + шум
        for i in range(n_samples):
            t = i / sample_rate
            # Основной тон
            tone = math.sin(2 * math.pi * f0 * t)
            # Шум
            random.seed(42 + i % 1000)
            noise = random.uniform(-0.3, 0.3)
            samples[i] = amp * (0.7 * tone + 0.3 * noise)
    
    elif con_type == 'liquid':
        # Плавные согласные: тон с быстрой модуляцией
        for i in range(n_samples):
            t = i / sample_rate
            # Основной тон с модуляцией
            mod_freq = 20 + 10 * math.sin(2 * math.pi * 3 * t)
            tone = math.sin(2 * math.pi * f0 * t + 2 * math.sin(2 * math.pi * mod_freq * t))
            samples[i] = amp * tone * 0.8
    
    elif con_type == 'trill':
        # Дрожащие согласные (Р): быстрая амплитудная модуляция
        for i in range(n_samples):
            t = i / sample_rate
            # Основной тон
            tone = math.sin(2 * math.pi * f0 * t)
            # Амплитудная модуляция (дрожание)
            trill = 0.5 + 0.5 * math.sin(2 * math.pi * 25 * t)
            samples[i] = amp * tone * trill * 0.8
    
    elif con_type == 'affricate':
        # Аффрикаты: переход от шума к тону
        for i in range(n_samples):
            t = i / sample_rate
            transition = min(1.0, t / (duration * 0.3))  # Переход за 30% времени
            # Шумовая компонента
            random.seed(42 + i % 1000)
            noise = random.uniform(-1, 1)
            # Тоновая компонента
            tone = math.sin(2 * math.pi * f0 * t)
            samples[i] = amp * ((1 - transition) * noise + transition * tone) * 0.7
    
    # Огибачка
    samples = apply_envelope(samples, attack=0.01, decay=0.02, sustain_level=0.9, release=0.01, sample_rate=sample_rate)
    
    return samples


# ============================================================================
# ФОРМАНТНЫЙ СИНТЕЗ
# ============================================================================

def formant_synthesize(phoneme, duration=0.3, f0=None, sample_rate=22050):
    """Формантный синтез фонемы
    
    Args:
        phoneme: символ фонемы
        duration: длительность в секундах
        f0: основная частота (если None, используется стандартная)
        sample_rate: частота дискретизации
    
    Returns:
        (samples, params) — сэмплы и использованные параметры
    """
    random.seed(42)
    
    if phoneme in VOWEL_FORMANTS:
        # Гласная
        f0_default, f1, f2, f3, amp = VOWEL_FORMANTS[phoneme]
        if f0 is None:
            f0 = f0_default
        
        n_samples = int(duration * sample_rate)
        samples = [0.0] * n_samples
        
        # Параметры для анализа
        params = {
            'type': 'vowel',
            'phoneme': phoneme,
            'f0': f0,
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'amplitude': amp,
            'duration_ms': duration * 1000,
        }
        
        # Генерация с формантными резонансами
        for i in range(n_samples):
            t = i / sample_rate
            
            # Основной тон с гармониками
            signal = 0.0
            for h in range(1, 12):
                harmonic_amp = 1.0 / (h ** 0.7)
                signal += harmonic_amp * math.sin(2 * math.pi * f0 * h * t)
            
            # Формантные полюсы (упрощённая модель)
            # Используем суммирование синусоид на формантных частотах
            f1_sig = 0.4 * math.sin(2 * math.pi * f1 * t)
            f2_sig = 0.3 * math.sin(2 * math.pi * f2 * t)
            f3_sig = 0.15 * math.sin(2 * math.pi * f3 * t)
            
            formants = f1_sig + f2_sig + f3_sig
            
            # Комбинация гармонического и формантного сигнала
            samples[i] = amp * (0.5 * signal + 0.5 * formants)
        
        # Огибачка
        samples = apply_envelope(samples, attack=0.04, decay=0.08, sustain_level=0.75, release=0.08, sample_rate=sample_rate)
        
    elif phoneme in CONSONANT_PARAMS:
        # Согласная
        samples = generate_consonant_sound(phoneme, sample_rate=sample_rate)
        params = {
            'type': 'consonant',
            'phoneme': phoneme,
            'type_detail': CONSONANT_PARAMS[phoneme][0],
            'duration_ms': len(samples) / sample_rate * 1000,
        }
    
    else:
        raise ValueError(f"Неизвестная фонема: {phoneme}")
    
    return samples, params


# ============================================================================
# КОНКАТЕНАЦИЯ СЛОВ
# ============================================================================

def phonemize_word(word):
    """Разбиение слова на фонемы (упрощённое)
    
    Args:
        word: слово в верхнем регистре
    
    Returns:
        список фонем
    """
    word = word.upper().strip()
    phonemes = []
    i = 0
    
    while i < len(word):
        char = word[i]
        
        # Двузначные фонемы
        if i + 1 < len(word):
            bigram = word[i:i+2]
            if bigram in ('ШЧ', 'ЦС', 'ДЖ', 'ДЗ'):
                phonemes.append(bigram)
                i += 2
                continue
        
        # Однозначные фонемы
        phonemes.append(char)
        i += 1
    
    return phonemes


def synthesize_word(word, base_f0=120, duration_per_phoneme=0.2, sample_rate=22050):
    """Синтез слова из набора фонем
    
    Args:
        word: слово для синтеза
        base_f0: базовая основная частота
        duration_per_phoneme: длительность одной фонемы (секунды)
        sample_rate: частота дискретизации
    
    Returns:
        (samples, phoneme_info) — сэмплы и информация о фонемах
    """
    phonemes = phonemize_word(word)
    all_samples = []
    phoneme_info = []
    
    for i, phoneme in enumerate(phonemes):
        # Небольшая вариация длительности
        random.seed(42 + i)
        duration_var = duration_per_phoneme * (0.8 + 0.4 * random.random())
        
        # Вариация F0 (интонация)
        f0_var = base_f0 * (1.0 + 0.05 * math.sin(2 * math.pi * 0.5 * i / len(phonemes)))
        
        try:
            samples, params = formant_synthesize(phoneme, duration=duration_var, f0=f0_var, sample_rate=sample_rate)
            all_samples.append(samples)
            phoneme_info.append(params)
        except ValueError:
            # Пропускаем неизвестные символы
            continue
    
    # Конкатенация с короткими паузами
    if all_samples:
        result = concatenate_sounds(*all_samples, gap_ms=30, sample_rate=sample_rate)
    else:
        result = []
    
    return result, phoneme_info


# ============================================================================
# ДЕМОНСТРАЦИЯ 1: Генерация базовых тонов
# ============================================================================

def demo1_basic_tones():
    """Демонстрация генерации базовых тонов"""
    print("=" * 70)
    print("ДЕМО 1: ГЕНЕРАЦИЯ БАЗОВЫХ ТОНОВ")
    print("=" * 70)
    print()
    
    sample_rate = 22050
    
    # 1.1 Простой синусоидальный тон
    print("1.1 Простой синусоидальный тон (440 Гц, 0.5 сек)")
    tone_a4 = sine_wave(440, 0.5, sample_rate)
    print(f"    Количество сэмплов: {len(tone_a4)}")
    print(f"    RMS: {rms(tone_a4):.4f}")
    print(f"    Пик: {peak(tone_a4):.4f}")
    print(f"    Длительность: {duration_sec(tone_a4, sample_rate):.3f} сек")
    print()
    
    # 1.2 Ряд гармоник
    print("1.2 Ряд гармоник (f0=220 Гц, 8 гармоник)")
    harmonic_gen = harmonic_series(220, 8)
    harmonic_tone = harmonic_gen(0.5, sample_rate)
    print(f"    Количество сэмплов: {len(harmonic_tone)}")
    print(f"    RMS: {rms(harmonic_tone):.4f}")
    print(f"    Пик: {peak(harmonic_tone):.4f}")
    print()
    
    # 1.3 Различные основные частоты
    print("1.3 Различные основные частоты:")
    frequencies = {
        'C4 (до4)': 261.63,
        'D4 (ре4)': 293.66,
        'E4 (ми4)': 329.63,
        'F4 (фа4)': 349.23,
        'G4 (соль4)': 392.00,
        'A4 (ля4)': 440.00,
        'B4 (си4)': 493.88,
    }
    
    for name, freq in frequencies.items():
        tone = sine_wave(freq, 0.3, sample_rate)
        print(f"    {name}: {freq:.2f} Гц, RMS={rms(tone):.4f}, пик={peak(tone):.4f}")
    print()
    
    # 1.4 Огибачка ADSR
    print("1.4 Влияние ADSR-огибачки:")
    tone = sine_wave(440, 0.5, sample_rate)
    
    configs = [
        ("Без огибачки", 0, 0, 1.0, 0),
        ("Быстрое нарастание", 0.01, 0.02, 0.8, 0.05),
        ("Длинное нарастание", 0.2, 0.1, 0.6, 0.2),
        ("Короткий импульс", 0.005, 0.05, 0.3, 0.1),
    ]
    
    for name, a, d, s, r in configs:
        shaped = apply_envelope(tone, attack=a, decay=d, sustain_level=s, release=r, sample_rate=sample_rate)
        print(f"    {name}: RMS={rms(shaped):.4f}, пик={peak(shaped):.4f}")
    print()
    
    # 1.5 Нормализация
    print("1.5 Влияние нормализации:")
    quiet_tone = sine_wave(440, 0.3, sample_rate)
    quiet_tone = [s * 0.1 for s in quiet_tone]  # Тихий тон
    print(f"    До нормализации: RMS={rms(quiet_tone):.6f}, пик={peak(quiet_tone):.4f}")
    
    normalized = normalize(quiet_tone, target_peak=0.9)
    print(f"    После нормализации: RMS={rms(normalized):.6f}, пик={peak(normalized):.4f}")
    print()


# ============================================================================
# ДЕМОНСТРАЦИЯ 2: Формантный синтез
# ============================================================================

def demo2_formant_synthesis():
    """Демонстрация формантного синтеза"""
    print("=" * 70)
    print("ДЕМО 2: ФОРМАНТНЫЙ СИНТЕЗ")
    print("=" * 70)
    print()
    
    sample_rate = 22050
    
    # 2.1 Синтез гласных
    print("2.1 Синтез гласных фонем:")
    print("-" * 50)
    
    vowels = ['А', 'О', 'У', 'Э', 'Ы', 'И', 'Е', 'Ё']
    for vowel in vowels:
        samples, params = formant_synthesize(vowel, duration=0.3, sample_rate=sample_rate)
        print(f"    {vowel}: F0={params['f0']} Гц, F1={params['f1']} Гц, "
              f"F2={params['f2']} Гц, F3={params['f3']} Гц")
        print(f"         RMS={rms(samples):.4f}, длительность={duration_sec(samples, sample_rate):.3f} сек")
    print()
    
    # 2.2 Синтез согласных
    print("2.2 Синтез согласных фонем:")
    print("-" * 50)
    
    consonants = ['Б', 'В', 'Г', 'Д', 'Ж', 'З', 'К', 'Л', 'М', 'Н']
    for con in consonants:
        samples, params = formant_synthesize(con, sample_rate=sample_rate)
        print(f"    {con}: тип={params['type_detail']}, "
              f"длительность={params['duration_ms']:.1f} мс")
        print(f"         RMS={rms(samples):.4f}")
    print()
    
    # 2.3 Влияние F0 на качество
    print("2.3 Влияние основной частоты (F0) на гласную 'А':")
    print("-" * 50)
    
    f0_values = [80, 100, 120, 150, 200, 250]
    for f0 in f0_values:
        samples, params = formant_synthesize('А', duration=0.3, f0=f0, sample_rate=sample_rate)
        print(f"    F0={f0} Гц: RMS={rms(samples):.4f}, пик={peak(samples):.4f}")
    print()
    
    # 2.4 Спектральный анализ
    print("2.4 Спектральный анализ гласных:")
    print("-" * 50)
    
    for vowel in ['А', 'О', 'У', 'И']:
        samples, _ = formant_synthesize(vowel, duration=0.5, sample_rate=sample_rate)
        spectrum = frequency_spectrum(samples, sample_rate, n_bins=8)
        spectrum_str = ', '.join([f'{v:.2f}' for v in spectrum])
        print(f"    {vowel}: [{spectrum_str}]")
    print()


# ============================================================================
# ДЕМОНСТРАЦИЯ 3: Конкатенация слов
# ============================================================================

def demo3_word_concatenation():
    """Демонстрация конкатенации слов"""
    print("=" * 70)
    print("ДЕМО 3: КОНКАТЕНАЦИЯ СЛОВ")
    print("=" * 70)
    print()
    
    sample_rate = 22050
    
    # 3.1 Разбиение слов на фонемы
    print("3.1 Разбиение слов на фонемы:")
    print("-" * 50)
    
    words = ['А', 'О', 'МА', 'ПА', 'БА', 'ДОМ', 'СЛОВО', 'МАМА', 'ПАПА']
    for word in words:
        phonemes = phonemize_word(word)
        print(f"    {word} -> {phonemes}")
    print()
    
    # 3.2 Синтез отдельных слов
    print("3.2 Синтез слов:")
    print("-" * 50)
    
    test_words = ['МА', 'ПА', 'ДА', 'БА', 'ДОМ']
    for word in test_words:
        samples, info = synthesize_word(word, base_f0=120, duration_per_phoneme=0.15, sample_rate=sample_rate)
        phonemes_str = [p['phoneme'] for p in info]
        print(f"    {word}: фонемы={phonemes_str}")
        print(f"         RMS={rms(samples):.4f}, длительность={duration_sec(samples, sample_rate):.3f} сек")
        print(f"         сэмплов={len(samples)}")
    print()
    
    # 3.3 Конкатенация фразы
    print("3.3 Конкатенация фразы 'МАМА МАЛА':")
    print("-" * 50)
    
    word1_samples, info1 = synthesize_word('МАМА', base_f0=120, duration_per_phoneme=0.12, sample_rate=sample_rate)
    word2_samples, info2 = synthesize_word('МАЛА', base_f0=115, duration_per_phoneme=0.12, sample_rate=sample_rate)
    
    # Добавляем паузу между словами
    gap_samples = int(0.1 * sample_rate)  # 100 мс пауза
    phrase = word1_samples + [0.0] * gap_samples + word2_samples
    
    print(f"    Слово 1 'МАМА': {len(word1_samples)} сэмплов, {duration_sec(word1_samples, sample_rate):.3f} сек")
    print(f"    Пауза: {gap_samples} сэмплов, 0.100 сек")
    print(f"    Слово 2 'МАЛА': {len(word2_samples)} сэмплов, {duration_sec(word2_samples, sample_rate):.3f} сек")
    print(f"    Итого фраза: {len(phrase)} сэмплов, {duration_sec(phrase, sample_rate):.3f} сек")
    print(f"    RMS фразы: {rms(phrase):.4f}")
    print()
    
    # 3.4 Сохранение в WAV
    print("3.4 Сохранение синтезированной речи в WAV-файлы:")
    print("-" * 50)
    
    wav_files = []
    
    # Сохраняем отдельные фонемы
    for vowel in ['А', 'О', 'У']:
        samples, _ = formant_synthesize(vowel, duration=0.3, sample_rate=sample_rate)
        filename = f"demo_vowel_{vowel}.wav"
        save_wav(filename, normalize(samples), sample_rate)
        wav_files.append(filename)
        print(f"    Сохранён: {filename}")
    
    # Сохраняем слово
    word_samples, _ = synthesize_word('МАМА', base_f0=120, duration_per_phoneme=0.15, sample_rate=sample_rate)
    filename = "demo_word_МАМА.wav"
    save_wav(filename, normalize(word_samples), sample_rate)
    wav_files.append(filename)
    print(f"    Сохранён: {filename}")
    
    # Сохраняем фразу
    filename = "demo_phrase_МАМА_МАЛА.wav"
    save_wav(filename, normalize(phrase), sample_rate)
    wav_files.append(filename)
    print(f"    Сохранён: {filename}")
    print()
    
    return wav_files


# ============================================================================
# ДЕМОНСТРАЦИЯ 4: Влияние параметров на качество
# ============================================================================

def demo4_parameter_influence():
    """Демонстрация влияния параметров на качество синтеза"""
    print("=" * 70)
    print("ДЕМО 4: ВЛИЯНИЕ ПАРАМЕТРОВ НА КАЧЕСТВО")
    print("=" * 70)
    print()
    
    sample_rate = 22050
    
    # 4.1 Количество гармоник
    print("4.1 Влияние количества гармоник:")
    print("-" * 50)
    
    for n_harmonics in [3, 5, 8, 12, 20]:
        gen = harmonic_series(220, n_harmonics)
        tone = gen(0.3, sample_rate)
        print(f"    {n_harmonics} гармоник: RMS={rms(tone):.4f}, пик={peak(tone):.4f}")
    print()
    
    # 4.2 Глубина вибрато
    print("4.2 Влияние глубины вибрато:")
    print("-" * 50)
    
    for vibrato_depth in [0.0, 1.0, 2.0, 5.0, 10.0]:
        samples, _ = formant_synthesize('А', duration=0.3, sample_rate=sample_rate)
        # Модифицируем с разной глубиной вибрато
        n_samples = len(samples)
        modified = [0.0] * n_samples
        for i in range(n_samples):
            t = i / sample_rate
            vibrato = vibrato_depth * math.sin(2 * math.pi * 5.0 * t)
            # Простая модуляция
            mod = 1.0 + 0.1 * vibrato / 10.0
            modified[i] = samples[i] * mod
        
        print(f"    Вибрато {vibrato_depth:.1f} Гц: RMS={rms(modified):.4f}, пик={peak(modified):.4f}")
    print()
    
    # 4.3 Частота дискретизации
    print("4.3 Влияние частоты дискретизации:")
    print("-" * 50)
    
    for sr in [8000, 16000, 22050, 44100, 48000]:
        samples, _ = formant_synthesize('А', duration=0.3, sample_rate=sr)
        print(f"    {sr} Гц: {len(samples)} сэмплов, "
              f"RMS={rms(samples):.4f}, "
              f"разрешение={sr/1000:.1f} кГц")
    print()
    
    # 4.4 Длительность фонемы
    print("4.4 Влияние длительности фонемы:")
    print("-" * 50)
    
    for dur_ms in [50, 100, 150, 200, 300, 500]:
        dur = dur_ms / 1000.0
        samples, _ = formant_synthesize('А', duration=dur, sample_rate=sample_rate)
        print(f"    {dur_ms} мс: {len(samples)} сэмплов, "
              f"RMS={rms(samples):.4f}")
    print()
    
    # 4.5 ADSR-параметры
    print("4.5 Влияние ADSR-параметров на гласную 'А':")
    print("-" * 50)
    
    base_samples, _ = formant_synthesize('А', duration=0.5, sample_rate=sample_rate)
    
    adsr_configs = [
        ("Стандартный", 0.04, 0.08, 0.75, 0.08),
        ("Быстрый старт", 0.005, 0.02, 0.9, 0.02),
        ("Медленный старт", 0.2, 0.1, 0.6, 0.2),
        ("Перкуссия", 0.001, 0.05, 0.2, 0.3),
        ("Орган", 0.05, 0.01, 0.95, 0.05),
    ]
    
    for name, a, d, s, r in adsr_configs:
        shaped = apply_envelope(base_samples, attack=a, decay=d, sustain_level=s, release=r, sample_rate=sample_rate)
        print(f"    {name}: A={a:.3f}, D={d:.3f}, S={s:.2f}, R={r:.3f} -> "
              f"RMS={rms(shaped):.4f}")
    print()
    
    # 4.6 Сравнение методов синтеза согласных
    print("4.6 Сравнение типов согласных:")
    print("-" * 50)
    
    con_types = {}
    for phoneme, (con_type, *_) in CONSONANT_PARAMS.items():
        if con_type not in con_types:
            con_types[con_type] = []
        con_types[con_type].append(phoneme)
    
    for con_type, phonemes in con_types.items():
        print(f"    {con_type}: {', '.join(phonemes)}")
    print()
    
    # 4.7 Итоговая сводка
    print("4.7 Итоговая сводка влияния параметров:")
    print("-" * 50)
    print("    1. F0 (основная частота): определяет высоту тона")
    print("       - Низкое F0 (80-120 Гц): мужской голос")
    print("       - Высокое F0 (180-300 Гц): женский/детский голос")
    print()
    print("    2. F1, F2, F3 (форманты): определяют качество гласных")
    print("       - F1: открытость рта (высокое=открытое, низкое=закрытое)")
    print("       - F2: передний/задний резонанс (высокое=переднее, низкое=заднее)")
    print("       - F3: индивидуальные особенности")
    print()
    print("    3. ADSR-огибачка: определяет временную динамику")
    print("       - Attack: быстрый=ударный, медленный=плавный")
    print("       - Decay/Sustain: определяет характер звука")
    print("       - Release: определяет затухание")
    print()
    print("    4. Количество гармоник: влияет на богатство тембра")
    print("       - Мало гармоник (3-5): простой, синтетический звук")
    print("       - Много гармоник (12-20): богатый, естественный звук")
    print()
    print("    5. Частота дискретизации: влияет на верхнюю границу частот")
    print("       - 8 кГц: телефонное качество (до 4 кГц)")
    print("       - 22 кГц: стандартное качество речи")
    print("       - 44+ кГц: высокое качество, полный спектр")
    print()


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Запуск всех демонстраций"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "ТЕКСТ-В-РЕЧЬ: ОСНОВЫ СИНТЕЗА РЕЧИ".center(68) + "║")
    print("║" + "Формантный синтез и конкатенация фонем".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Установка seed для воспроизводимости
    random.seed(42)
    
    # Запуск демонстраций
    demo1_basic_tones()
    demo2_formant_synthesis()
    wav_files = demo3_word_concatenation()
    demo4_parameter_influence()
    
    # Финальное резюме
    print("=" * 70)
    print("ИТОГОВОЕ РЕЗЮМЕ")
    print("=" * 70)
    print()
    print("Файл демонстрирует основы синтеза речи:")
    print()
    print("1. ФОРМАНТНЫЙ СИНТЕЗ:")
    print("   - Моделирование формантных частот F1, F2, F3")
    print("   - Генерация гласных с резонансными полюсами")
    print("   - Моделирование согласных (взрывные, фрикативные и др.)")
    print()
    print("2. ГЕНЕРАЦИЯ ТОНОВ:")
    print("   - Основной тон (F0) с гармонической структурой")
    print("   - ADSR-огибачка для временной динамики")
    print("   - Вибрато и модуляция")
    print()
    print("3. КОНКАТЕНАЦИЯ:")
    print("   - Разбиение слов на фонемы")
    print("   - Последовательный синтез фонем")
    print("   - Объединение с паузами")
    print()
    print("4. ВЛИЯНИЕ ПАРАМЕТРОВ:")
    print("   - F0, форманты, огибачка, частота дискретизации")
    print("   - Количества гармоник, длительность фонем")
    print()
    
    if wav_files:
        print("Созданные WAV-файлы:")
        for f in wav_files:
            print(f"  - {f}")
    print()
    
    print("Примечание: для прослушивания результатов используйте")
    print("аудиоплеер или библиотеки Python (pydub, soundfile и т.д.)")
    print()


if __name__ == "__main__":
    main()
