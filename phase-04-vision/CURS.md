# Phase 4: Vision

> Computer Vision — от свёрточных сетей до детекции объектов.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 52 | [Image Representations](#урок-52-image-representations) | [Код](52-image_representations.py) |
| 53 | [Convolution Operation](#урок-53-convolution-operation) | [Код](53-convolution.py) |
| 54 | [Pooling Layers](#урок-54-pooling-layers) | [Код](54-pooling.py) |
| 55 | [Building a CNN](#урок-55-building-a-cnn-from-scratch) | [Код](55-building_cnn.py) |
| 56 | [Classic Architectures](#урок-56-classic-architectures) | [Код](56-classic_architectures.py) |
| 57 | [Transfer Learning](#урок-57-transfer-learning) | [Код](57-transfer_learning.py) |
| 58 | [Data Augmentation](#урок-58-data-augmentation) | [Код](58-data_augmentation.py) |
| 59 | [Object Detection Basics](#урок-59-object-detection-basics) | [Код](59-object_detection.py) |
| 60 | [Image Segmentation](#урок-60-image-segmentation) | [Код](60-image_segmentation.py) |

---

## Урок 52: Image Representations

### Изображение = матрица пикселей

```
Grayscale:  H × W (яркость 0-255)
RGB:        H × W × 3 (красный, зелёный, синий)
```

### Фильтры

| Фильтр | Ядро | Эффект |
|---|---|---|
| Box blur | 1/9 × [[1,1,1],[1,1,1],[1,1,1]] | Размытие |
| Sharpen | [[0,-1,0],[-1,5,-1],[0,-1,0]] | Резкость |
| Edge detection | [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] | Границы |

---

## Урок 53: Convolution Operation

### 2D свёртка

```
output[i][j] = Σ kernel[m][n] × input[i+m][j+n]
```

### Padding

| Тип | Описание |
|---|---|
| Valid | Без padding → уменьшает размер |
| Same | Zero padding → сохраняет размер |

### Stride

```
H_out = (H - kernel + 2×padding) / stride + 1
```

---

## Урок 54: Pooling Layers

### Типы pooling

| Тип | Описание |
|---|---|
| Max pooling | Максимальное значение в окне |
| Average pooling | Среднее в окне |
| Global average pooling | Среднее по всему пространству |

---

## Урок 55: Building a CNN

### Архитектура

```
Input → [Conv → ReLU → Pool] × N → Flatten → [FC → ReLU] × M → Output
```

### Преимущества CNN vs FC

- Weight sharing: меньше параметров
- Sparse connections: локальные паттерны
- Иерархические признаки: low-level → high-level

---

## Урок 56: Classic Architectures

### Эволюция

| Год | Модель | Глубина | Параметры |
|---|---|---|---|
| 1998 | LeNet-5 | 7 | 61K |
| 2012 | AlexNet | 8 | 60M |
| 2014 | VGG-16 | 16 | 138M |
| 2015 | ResNet-50 | 50 | 25M |

### Residual Connections

```
output = F(x) + x

Градиент: ∂output/∂x = ∂F/∂x + 1 (незатухающий)
```

---

## Урок 57: Transfer Learning

### Подходы

| Подход | Описание | Когда |
|---|---|---|
| Feature extraction | Заморозить backbone, обучить FC | Мало данных |
| Fine-tuning | Разморозить последние слои | Средне данных |
| Full fine-tuning | Обучить всё | Много данных |

---

## Урок 58: Data Augmentation

### Методы

| Метод | Описание |
|---|---|
| Flip | Горизонтальный/вертикальный |
| Rotation | Поворот на угол |
| Brightness | Изменение яркости |
| Crop | Обрезка |
| Noise | Гауссов шум |

---

## Урок 59: Object Detection

### Метрики

| Метрика | Формула |
|---|---|
| IoU | intersection / union |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| mAP | Average precision по классам |

### NMS

1. Сортировка по confidence
2. Выбираем лучший
3. Подавляем все с IoU > threshold

---

## Урок 60: Image Segmentation

### Типы сегментации

| Тип | Описание |
|---|---|
| Пороговая | Разделение по яркости |
| K-Means | Кластеризация пикселей |
| Connected components | Связные области |

### Метрики

| Метрика | Формула |
|---|---|
| IoU | intersection / union |
| Dice | 2 × intersection / (total1 + total2) |
| Pixel accuracy | correct / total |
