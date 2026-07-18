# Phase 3: Deep Learning Core

> Нейросети с нуля — от перцептрона до трансформеров.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 39 | [The Perceptron](#урок-39-the-perceptron) | [Код](39-perceptron.py) |
| 40 | [Multi-Layer Networks](#урок-40-multi-layer-networks--forward-pass) | [Код](40-multi_layer_networks.py) |
| 41 | [Backpropagation](#урок-41-backpropagation-from-scratch) | [Код](41-backpropagation.py) |
| 42 | [Activation Functions](#урок-42-activation-functions) | [Код](42-activation_functions.py) |
| 43 | [Loss Functions](#урок-43-loss-functions) | [Код](43-loss_functions.py) |
| 44 | [Optimizers](#урок-44-optimizers) | [Код](44-optimizers.py) |
| 45 | [Weight Initialization](#урок-45-weight-initialization) | [Код](45-weight_initialization.py) |
| 46 | [Batch Normalization](#урок-46-batch-normalization) | [Код](46-batch_normalization.py) |
| 47 | [Dropout & Regularization](#урок-47-dropout--regularization) | [Код](47-dropout_regularization.py) |
| 48 | [Learning Rate Schedules](#урок-48-learning-rate-schedules) | [Код](48-learning_rate_schedules.py) |
| 49 | [Neural Network Framework](#урок-49-neural-network-framework) | [Код](49-neural_network_framework.py) |
| 50 | [Training Loop](#урок-50-training-loop-from-scratch) | [Код](50-training_loop.py) |
| 51 | [Debugging Neural Networks](#урок-51-debugging-neural-networks) | [Код](51-debugging_neural_networks.py) |

---

## Урок 39: The Perceptron

### Перцепptron

```
output = sign(w·x + b)

w = w + lr × (target - predicted) × x
```

Линейно разделимые данные → сходимость гарантирована.

**Не может решить XOR!**

---

## Урок 40: Multi-Layer Networks

### Нейросеть

```
Layer: [Neuron, Neuron, ...]
MLP:   [Layer 1, Layer 2, ...]

Forward: input → linear → activation → linear → activation → output
```

### Активации

| Формула | Когда |
|---|---|
| sigmoid: 1/(1+e⁻ˣ) | Выход ∈ (0,1) |
| tanh: (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Выход ∈ (-1,1) |
| ReLU: max(0,x) | Скрытые слои |

---

## Урок 41: Backpropagation

### Цепное правило через граф

```
1. Forward pass: вычислить значения
2. Topological sort: упорядочить узлы
3. Backward pass: seed=1, идти назад, chain rule
```

### Gradient checking

```
relative_error = |analytical - numerical| / max(|analytical|, |numerical|)
< 1e-5: нормально
```

---

## Урок 42: Activation Functions

| Функция | Формула | Производная | Проблема |
|---|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | σ(1-σ) | Затухание градиентов |
| ReLU | max(0,x) | 0 или 1 | Мёртвые нейроны |
| GELU | x·Φ(x) | Φ(x) + x·φ(x) | Вычислительно дороже |

---

## Урок 43: Loss Functions

| Функция | Формула | Когда |
|---|---|---|
| MSE | (1/n)Σ(pred-actual)² | Регрессия |
| Binary CE | -Σ[y·log(p)+(1-y)·log(1-p)] | Бинарная классификация |
| Categorical CE | -Σyₖ·log(pₖ) | Мультикласс |
| Huber | Компромисс MSE/MAE | Робастность к выбросам |

---

## Урок 44: Optimizers

### Сравнение

| Оптимизатор | Что использует | Когда |
|---|---|---|
| SGD | Только gradient | Baseline |
| SGD+Momentum | 1-й момент | Ускорение |
| Adam | 1-й + 2-й моменты | Дефолт для DL |
| AdamW | Adam + weight decay | Трансформеры |

---

## Урок 45: Weight Initialization

| Метод | Формула | Когда |
|---|---|---|
| Random N(0,σ) | σ = 0.01 | Не рекомендуется |
| Xavier/Glorot | σ = √(2/(fan_in+fan_out)) | Sigmoid, tanh |
| He | σ = √(2/fan_in) | ReLU |

---

## Урок 46: Batch Normalization

```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ × x̂ + β

Train: μ, σ² от батча
Inference: running μ, σ² (EMA)
```

Ускоряет сходимость, позволяет больший lr.

---

## Урок 47: Dropout & Regularization

| Метод | Как работает |
|---|---|
| Dropout | Случайное обнуление p% нейронов |
| L2 (Ridge) | penalty = λ × Σw² → малые веса |
| L1 (Lasso) | penalty = λ × Σ|w| → разреженные веса |
| Early stopping | Стоп когда val loss растёт |

---

## Урок 48: Learning Rate Schedules

| Расписание | Формула | Когда |
|---|---|---|
| Step decay | lr × factor каждые N эпох | Простое |
| Exponential | lr₀ × decay^t | Плавное |
| Cosine | lr_min + ½(lr_max-lr_min)(1+cos(πt/T)) | Трансформеры |
| Warmup + decay | Линейный рост → decay | Крупные модели |

---

## Урок 49: Neural Network Framework

```
Value → автодифференцирование
Neuron → sum(w*x) + b → activation
Layer → [Neuron, ...]
MLP → [Layer, ...]
```

---

## Урок 50: Training Loop

```
for epoch in range(epochs):
    1. Forward pass
    2. Compute loss
    3. Zero gradients
    4. Backward pass
    5. Update weights
    6. Evaluate on validation set
```

---

## Урок 51: Debugging Neural Networks

### Чек-лист

- [ ] Gradient checking (relative_error < 1e-5)
- [ ] Гистограммы весов и градиентов
- [ ] Loss curve (train vs val)
- [ ] Проверка на NaN/inf
- [ ] Совместимость форматов
