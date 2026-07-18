# Phase 2: ML Fundamentals

> Основы машинного обучения — от линейной регрессии до ансамблей.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 21 | [Что такое ML](#урок-21) | [Код](21-ml_intro.py) |
| 22 | [Linear Regression](#урок-22) | [Код](22-linear_regression.py) |
| 23 | Logistic Regression & Classification | — |
| 24 | Decision Trees & Random Forests | — |
| 25 | Support Vector Machines | — |
| 26 | KNN & Distance Metrics | — |
| 27 | Unsupervised Learning: K-Means, DBSCAN | — |
| 28 | Feature Engineering & Selection | — |
| 29 | Model Evaluation: Metrics, Cross-Validation | — |
| 30 | Bias, Variance & the Learning Curve | — |
| 31 | Ensemble Methods: Boosting, Bagging, Stacking | — |
| 32 | Hyperparameter Tuning | — |
| 33 | ML Pipelines & Experiment Tracking | — |
| 34 | Naive Bayes | — |
| 35 | Time Series Fundamentals | — |
| 36 | Anomaly Detection | — |
| 37 | Handling Imbalanced Data | — |
| 38 | Feature Selection | — |

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
