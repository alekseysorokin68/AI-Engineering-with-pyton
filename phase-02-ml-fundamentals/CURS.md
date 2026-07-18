# Phase 2: ML Fundamentals

> Основы машинного обучения — от линейной регрессии до ансамблей.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 21 | [Что такое ML](#урок-21) | [Код](21-ml_intro.py) |
| 22 | [Linear Regression](#урок-22) | [Код](22-linear_regression.py) |
| 23 | [Logistic Regression](#урок-23) | [Код](23-logistic_regression.py) |
| 24 | [Decision Trees & Random Forests](#урок-24) | [Код](24-decision_trees.py) |
| 25 | [Support Vector Machines](#урок-25) | [Код](25-support_vector_machines.py) |
| 26 | [KNN & Distance Metrics](#урок-26) | [Код](26-knn.py) |
| 27 | [Unsupervised Learning](#урок-27) | [Код](27-unsupervised_learning.py) |
| 28 | [Feature Engineering & Selection](#урок-28) | [Код](28-feature_engineering.py) |
| 29 | [Model Evaluation](#урок-29) | [Код](29-model_evaluation.py) |
| 30 | [Bias, Variance & Learning Curve](#урок-30) | [Код](30-bias-variance.py) |
| 31 | [Ensemble Methods](#урок-31) | [Код](31-ensemble_methods.py) |
| 32 | [Hyperparameter Tuning](#урок-32) | [Код](32-hyperparameter_tuning.py) |
| 33 | [ML Pipelines](#урок-33) | [Код](33-ml_pipelines.py) |
| 34 | [Naive Bayes](#урок-34) | [Код](34-naive_bayes.py) |
| 35 | [Time Series](#урок-35) | [Код](35-time_series.py) |
| 36 | [Anomaly Detection](#урок-36) | [Код](36-anomaly_detection.py) |
| 37 | [Imbalanced Data](#урок-37) | [Код](37-imbalanced_data.py) |
| 38 | [Feature Selection](#урок-38) | [Код](38-feature_selection.py) |

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
