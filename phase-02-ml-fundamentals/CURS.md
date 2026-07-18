# Phase 2: ML Fundamentals

> Основы машинного обучения — от линейной регрессии до ансамблей.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 21 | [Что такое ML](#урок-21-что-такое-machine-learning) | [Код](21-ml_intro.py) |
| 22 | [Linear Regression](#урок-22-linear-regression) | [Код](22-linear_regression.py) |
| 23 | [Logistic Regression](#урок-23-logistic-regression) | [Код](23-logistic_regression.py) |
| 24 | [Decision Trees & Random Forests](#урок-24-decision-trees--random-forests) | [Код](24-decision_trees.py) |
| 25 | [Support Vector Machines](#урок-25-support-vector-machines) | [Код](25-support_vector_machines.py) |
| 26 | [KNN & Distance Metrics](#урок-26-knn--distance-metrics) | [Код](26-knn.py) |
| 27 | [Unsupervised Learning](#урок-27-unsupervised-learning--k-means-dbscan) | [Код](27-unsupervised_learning.py) |
| 28 | [Feature Engineering & Selection](#урок-28-feature-engineering--selection) | [Код](28-feature_engineering.py) |
| 29 | [Model Evaluation](#урок-29-model-evaluation) | [Код](29-model_evaluation.py) |
| 30 | [Bias, Variance & Learning Curve](#урок-30-bias-variance-tradeoff) | [Код](30-bias-variance.py) |
| 31 | [Ensemble Methods](#урок-31-ensemble-methods) | [Код](31-ensemble_methods.py) |
| 32 | [Hyperparameter Tuning](#урок-32-hyperparameter-tuning) | [Код](32-hyperparameter_tuning.py) |
| 33 | [ML Pipelines](#урок-33-ml-pipelines) | [Код](33-ml_pipelines.py) |
| 34 | [Naive Bayes](#урок-34-naive-bayes) | [Код](34-naive_bayes.py) |
| 35 | [Time Series](#урок-35-time-series) | [Код](35-time_series.py) |
| 36 | [Anomaly Detection](#урок-36-anomaly-detection) | [Код](36-anomaly_detection.py) |
| 37 | [Imbalanced Data](#урок-37-imbalanced-data) | [Код](37-imbalanced_data.py) |
| 38 | [Feature Selection](#урок-38-feature-selection) | [Код](38-feature_selection.py) |

---

## Урок 21: Что такое Machine Learning

### Три типа ML

| Тип | Данные | Задача |
|---|---|---|
| **Supervised** | Вход-выход пары | Классификация, регрессия |
| **Unsupervised** | Только входы | Кластеризация, PCA |
| **Reinforcement** | Агент + среда | Оптимизация политики |

### Train / Validation / Test

```
Training (70%): модель учится
Validation (15%): настройка гиперпараметров
Test (15%): финальная оценка (священный!)
```

### Overfitting vs Underfitting

```
Underfitting: слишком простая → high bias
Overfitting: слишком сложная → high variance
Good fit: баланс → хорошо обобщает

Total error = Bias² + Variance + noise
```

---

## Урок 22: Linear Regression

### Модель

```
y = wx + b
```

### Cost Function (MSE)

```
MSE = (1/n) × Σ(y_pred - y_actual)²
```

### Gradient Descent

```
dMSE/dw = (2/n) × Σ(y_pred - y) × x
w = w - lr × dMSE/dw
```

### Normal Equation

```
w = (XᵀX)⁻¹ × Xᵀy  (закрытая форма, O(n³))
```

### Ridge (L2 регуляризация)

```
Cost = MSE + λ × Σw²  → сжимает веса к нулю
```

---

## Урок 23: Logistic Regression

### Sigmoid Function

```
sigmoid(z) = 1 / (1 + e^(-z))

z → +∞: sigmoid → 1
z → -∞: sigmoid → 0
z = 0:  sigmoid = 0.5
```

### Модель

```
z = wx + b         (линейная комбинация)
p = sigmoid(z)     (вероятность)

p >= 0.5 → класс 1
p < 0.5  → класс 0
```

### Binary Cross-Entropy Loss

```
L = -(1/n) × Σ[y·log(p) + (1-y)·log(1-p)]

y=1, p→1: loss→0 (правильно)
y=1, p→0: loss→∞ (неправильно)
```

### Метрики

| Метрика | Формула | Когда важна |
|---|---|---|
| Precision | TP/(TP+FP) | Ложные позитивы дороги |
| Recall | TP/(TP+FN) | Ложные негативы дороги |
| F1 | 2×P×R/(P+R) | Баланс |

### Multi-class (Softmax)

```
softmax(zᵢ) = exp(zᵢ) / Σexp(zⱼ)

Класс с макс. вероятностью = предсказание
```

---

## Урок 24: Decision Trees & Random Forests

### Gini Impurity

```
Gini(S) = 1 - Σpₖ²

0 = чисто, 0.5 = макс. нечистота (50/50)
```

### Entropy

```
Entropy(S) = -Σpₖ·log₂(pₖ)

0 = чисто, 1 = макс. неопределённость
```

### Information Gain

```
IG = Impurity(родитель) - weighted_avg(Impurity(дети))

Чем больше IG → тем лучше разбиение
```

### Random Forest

```
Данные → Bootstrap samples → Дерево 1 (случайные признаки)
                           → Дерево 2 (случайные признаки)
                           → ...
                           → Дерево N → Majority vote

Bagging: каждое дерево на своём bootstrap (~63% уникальных)
Feature randomisation: случайный подмножество признаков на каждом разбиении
```

### Когда деревья лучше нейросетей

- Табличные данные (structured data)
- Маленькие датасеты (< 10k строк)
- Нужна интерпретируемость
- Смешанные типы признаков

---

## Урок 25: Support Vector Machines

### Maximum Margin

```
minimize    (1/2) ||w||²
subject to  yᵢ(wᵀxᵢ + b) ≥ 1

Максимальный отступ = 2 / ||w||
```

### Soft Margin (параметр C)

```
minimize    (1/2) ||w||² + C × Σξᵢ

C большой → мало нарушений, узкий отступ → переобучение
C малый → больше нарушений, широкий отступ → недообучение
```

### Hinge Loss

```
loss = max(0, 1 - y × f(x))

y×f(x) ≥ 1: loss = 0
y×f(x) < 1: loss линейно растёт
```

### Kernel Trick

```
Линейный:     K(x, z) = x · z
Полиномиальный: K(x, z) = (x·z + c)ᵈ
RBF:           K(x, z) = exp(-γ‖x-z‖²)

RBF → бесконечномерное пространство → любая гладкая граница
```

### SVM vs Logistic Regression

| | SVM | Log. Regression |
|---|---|---|
| Loss | Hinge | Log |
| Решения | Разреженные (SV) | Все точки |
| Ядра | Да | Нет |

---

## Урок 26: KNN & Distance Metrics

### K-Nearest Neighbors

```
1. Запомни все обучающие данные
2. Для новой точки найди K ближайших соседей
3. Прогноз = самый частый класс среди соседей
```

### Метрики расстояния

| Метрика | Формула | Когда |
|---|---|---|
| Евклидова | √(Σ(xᵢ-yᵢ)²) | Пространственные данные |
| Манхэттен | Σ|xᵢ-yᵢ| | Разреженные данные |
| Косинусная | 1 - (x·y)/(‖x‖‖y‖) | Текст, эмбеддинги |

### Влияние K

```
K=1: переобучение (шум)
K=big: недообучение (размытая граница)
```

---

## Урок 27: Unsupervised Learning — K-Means, DBSCAN

### K-Means

```
1. Выбери K случайных центров
2. Назначь каждую точку ближайшему центру
3. Пересчитай центры как среднее кластеров
4. Повторяй до сходимости
```

### DBSCAN

```
1. Найди core points (≥ min_samples соседей в eps)
2. Расширь кластеры через core points
3. Точки без core neighbors = шум
```

### Elbow Method

Ищи "локоть" на графике inertia vs K — где inertia перестаёт резко падать.

---

## Урок 28: Feature Engineering & Selection

### Методы нормализации

| Метод | Формула | Когда |
|---|---|---|
| Z-score | (x - μ) / σ | Нормальные данные |
| Min-Max | (x - min) / (max - min) | Ограничённый диапазон |

### One-Hot Encoding

```
Категория → вектор из 0 и 1
"кошка" → [1, 0, 0]
"собака" → [0, 1, 0]
```

---

## Урок 29: Model Evaluation

### Метрики

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### K-Fold Cross-Validation

Разбить данные на K частей, K раз обучить на K-1, протестировать на оставшейся.

---

## Урок 30: Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

Underfitting: high bias, low variance
Overfitting: low bias, high variance
Good fit: баланс
```

### Learning Curves

График train/test error vs размер данных. Сходимость = good fit.

---

## Урок 31: Ensemble Methods

### Bagging

```
Bootstrap samples → N деревьев → Majority vote
Уменьшает variance
```

### AdaBoost

```
1. Обучи слабую модель
2. Увеличь веса ошибочных сэмплов
3. Обучи следующую модель на взвешенных данных
4. Повтори
```

---

## Урок 32: Hyperparameter Tuning

### Grid Search

Полный перебор всех комбинаций параметров.

### Random Search

Случайный выбор комбинаций — быстрее при большом пространстве параметров.

---

## Урок 33: ML Pipelines

### Пайплайн

```
Данные → Трансформер 1 → Трансформер 2 → Модель → Предсказание
```

Каждый шаг: fit/transform на train, transform на test.

---

## Урок 34: Naive Bayes

### Формула

```
P(class|features) ∝ P(class) × ∏P(feature_i|class)

Laplace smoothing: P = (count + 1) / (total + vocab_size)
```

---

## Урок 35: Time Series

### Методы

| Метод | Формула | Когда |
|---|---|---|
| Moving Average | Среднее по окну | Сглаживание тренда |
| Exponential Smoothing | α×x_t + (1-α)×预报 | Экспоненциальный тренд |

---

## Урок 36: Anomaly Detection

### Z-score

```
z = (x - μ) / σ
|z| > 3 → аномалия
```

### IQR

```
IQR = Q3 - Q1
Аномалии: x < Q1 - 1.5×IQR или x > Q3 + 1.5×IQR
```

---

## Урок 37: Handling Imbalanced Data

### Методы

| Метод | Как |
|---|---|
| Oversampling | Дублировать миноритарный класс |
| Undersampling | Уменьшить мажоритарный класс |
| Class weights | Увеличить вес миноритарного класса в loss |

---

## Урок 38: Feature Selection

### Методы

| Тип | Метод | Как |
|---|---|---|
| Filter | Корреляция, MI | Ранжировать признаки по связи с целью |
| Wrapper | Forward/Backward | Добавлять/убирать признаки, проверять модель |
| Embedded | L1 (Lasso) | Регуляризация обнуляет веса ненужных признаков |
