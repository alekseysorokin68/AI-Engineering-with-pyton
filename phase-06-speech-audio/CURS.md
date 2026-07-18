# Phase 6: Speech & Audio

> Обработка речи и аудио — от спектрограмм до распознавания речи.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 69 | [Audio Signal Processing](#урок-69-audio-signal-processing) | [Код](69-audio_signal_processing.py) |
| 70 | [Spectrograms & Mel Scale](#урок-70-spectrograms--mel-scale) | [Код](70-spectrograms.py) |
| 71 | [Speech Recognition](#урок-71-speech-recognition) | [Код](71-speech_recognition.py) |
| 72 | [Speaker Verification](#урок-72-speaker-verification) | [Код](72-speaker_verification.py) |
| 73 | [Text-to-Speech](#урок-73-text-to-speech) | [Код](73-text_to_speech.py) |
| 74 | [Music Generation](#урок-74-music-generation) | [Код](74-music_generation.py) |
| 75 | [Audio Augmentation](#урок-75-audio-augmentation) | [Код](75-audio_augmentation.py) |

---

## Урок 69: Audio Signal Processing

### Основы

| Концепция | Описание |
|---|---|
| Синусоида | Базовый строительный блок звука |
| FFT | Быстрое преобразование Фурье → частоты |
| Фильтры | Low-pass, high-pass, band-pass |
| Windowing | Окнонание (Hann, Hamming) |

---

## Урок 70: Spectrograms & Mel Scale

### STFT

```
STFT: FFT на наложенных окнах → 2D спектрограмма
Ось X: время, Ось Y: частота, Яркость: энергия
```

### MFCC

```
STFT → мел-фильтры → log → DCT → MFCC коэффициенты
```

---

## Урок 71: Speech Recognition

### Методы

| Метод | Описание |
|---|---|
| MFCC | Извлечение признаков |
| DTW | Сравнение временных рядов |
| HMM | Модель скрытых состояний |

---

## Урок 72: Speaker Verification

### Пайплайн

```
Аудио → MFCC признаки → Эмбеддинг говорящего → Cosine similarity → Решение
```

---

## Урок 73: Text-to-Speech

### Формантный синтез

```
Гласные: суммирование гармоник + резонансные полюса (F1, F2, F3)
Согласные: 6 типов (взрывные, фрикативные, носовые, ...)
Конкатенация: синтез фонем → слова → предложения
```

---

## Урок 74: Music Generation

### Методы

| Метод | Описание |
|---|---|
| Марковские цепи | Генерация мелодий по статистике |
| Пентатоника | 5 нот без диссонансов |
| Аккордовые прогрессии | I-IV-V-I, I-V-vi-IV |

---

## Урок 75: Audio Augmentation

### Методы

| Метод | Описание |
|---|---|
| Шум | Добавление гауссова шума |
| Time stretching | Изменение скорости |
| Pitch shifting | Изменение высоты тона |
| Smoothing | Скользящее среднее |
