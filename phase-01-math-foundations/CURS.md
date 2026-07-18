# Phase 1: Math Foundations

> Математика под капотом AI/ML — от линейной алгебры до преобразования Фурье.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 01 | [Линейная алгебра — интуиция](#урок-01-линейная-алгебра--интуиция) | — |
| 02 | [Векторы, матрицы и операции](#урок-02-векторы-матрицы-и-операции) | — |
| 03 | [Матричные преобразования](#урок-03-матричные-преобразования) | [Код](03-matrix_transformations.py) |
| 04 | [Calculus для ML](#урок-04-calculus-for-ml) | [Код](04-calculus_for_ml.py) |
| 05 | [Chain Rule & Autograd](#урок-05-chain-rule--automatic-differentiation) | [Код](05-chain_rule_autodiff.py) |
| 06 | [Вероятности и распределения](#урок-06-probability--distributions) | [Код](06-probability_distributions.py) |
| 07 | [Bayes' Theorem](#урок-07-bayes-theorem) | [Код](07-bayes_theorem.py) |
| 08 | [Оптимизация](#урок-08-optimization) | [Код](08-optimization.py) |
| 09 | [Теория информации](#урок-09-information-theory) | [Код](09-information_theory.py) |
| 10 | [Размерность](#урок-10-dimensionality-reduction) | [Код](10-dimensionality_reduction.py) |
| 11 | [SVD](#урок-11-svd) | [Код](11-svd.py) |
| 12 | [Tensor Operations](#урок-12-tensor-operations) | [Код](12-tensor_operations.py) |
| 13 | [Numerical Stability](#урок-13-numerical-stability) | [Код](13-numerical_stability.py) |
| 14 | [Нормы и расстояния](#урок-14-norms-and-distances) | [Код](14-norms_distances.py) |
| 15 | [Статистика для ML](#урок-15-statistics-for-ml) | [Код](15-statistics_for_ml.py) |
| 16 | [Методы семплирования](#урок-16-sampling-methods) | [Код](16-sampling_methods.py) |
| 17 | [Линейные системы](#урок-17-linear-systems) | [Код](17-linear_systems.py) |
| 18 | [Выпуклая оптимизация](#урок-18-convex-optimization) | [Код](18-convex_optimization.py) |
| 19 | [Комплексные числа](#урок-19-complex-numbers) | [Код](19-complex_numbers.py) |
| 20 | [Преобразование Фурье](#урок-20-fourier-transform) | [Код](20-fourier_transform.py) |

---

## Урок 01: Линейная алгебра — интуиция

### Векторы — это точки (и направления)

Вектор — список чисел. Числа = координаты в пространстве.

```
v = [3, 2]  — 2D вектор, длина √(9+4) = √13
```

В AI векторы представляют всё:
- Слово → вектор из 768 чисел (эмбеддинг)
- Изображение → вектор из миллионов пикселей
- Пользователь → вектор предпочтений

### Скалярное произведение (Dot Product)

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Одно направление:  a · b > 0  (похожи)
Перпендикулярны:   a · b = 0  (не связаны)
Противоположные:   a · b < 0  (различны)
```

### Проекция

```
proj_b(a) = (a·b / b·b) × b
```

### Процесс Грамма-Шмидта

Преобразование векторов в **ортонормальный базис**.

---

## Урок 02: Векторы, матрицы и операции

### Формула умножения

```
(m × n) @ (n × p) = (m × p)
```

### Broadcasting

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

→  | 11  22  33 |
   | 14  25  36 |
```

### Нейросеть = матричные операции

```
output = relu(W @ x + b)
```

---

## Урок 03: Матричные преобразования

### Типы преобразований

| Преобразование | Матрица (2D) |
|---|---|
| **Поворот** | `[[cos θ, -sin θ], [sin θ, cos θ]]` |
| **Масштаб** | `[[sx, 0], [0, sy]]` |
| **Сдвиг** | `[[1, kx], [ky, 1]]` |
| **Отражение** | `[[-1, 0], [0, 1]]` |

### Собственные числа и векторы

```
A @ v = λ * v
```

### Определитель

```
det = 1   → область сохранена
det = 0   → вырожденная матрица
```

---

## Урок 04: Calculus для ML

### Производная

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

### Градиентный спуск

```
w_new = w_old - learning_rate * dL/dw
```

### Цепное правило

```
y = f(g(x))  →  dy/dx = f'(g(x)) * g'(x)
```

### Ряд Тейлора

```
f(x+h) ≈ f(x) + f'(x)·h + ½f''(x)·h²
```

---

## Урок 05: Chain Rule & Autograd

### Автоматическое дифференцирование

1. **Value** — обёртка (значение + градиент)
2. **Graph recording** — запись операций
3. **Backward** — обратный обход → chain rule

### Reverse mode = backpropagation

```
Один обратный проход для всех градиентов
```

---

## Урок 06: Probability & Distributions

### Распределения

| Распределение | Где в ML |
|---|---|
| **Bernoulli** | Бинарная классификация |
| **Normal** | Веса, шум, градиенты |

### Softmax и Cross-entropy

```
softmax(zᵢ) = exp(zᵢ) / Σexp(zⱼ)
L = -log_softmax(logits)[target]
```

---

## Урок 07: Bayes' Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)
```

### Мнемоника

- **Априори = регуляризация**
- **Постериор = обновлённая вероятность**
- **Beta-Binomial = онлайн-обучение**

---

## Урок 08: Optimization

### Adam

```
w = w - lr × m̂ / (√v̂ + ε)
```

### Расписания lr

| Расписание | Когда |
|---|---|
| Cosine | Трансформеры |
| Warmup + decay | Крупные модели |

---

## Урок 09: Information Theory

### Ключевые формулы

```
Энтропия:    H(P) = -Σ p(x) · log(p(x))
Cross-entropy: H(P,Q) = -Σ p(x) · log(q(x))
KL:           D_KL(P||Q) = H(P,Q) - H(P)
```

---

## Урок 10: Dimensionality Reduction

### PCA

```
1. Центрируем
2. Ковариационная матрица
3. Собственная декомпозиция
4. Проекция на top k компонент
```

### Когда что

| Метод | Зачем |
|---|---|
| PCA | Препроцессинг |
| t-SNE | Визуализация 2D |
| UMAP | Визуализация + структура |

---

## Урок 11: SVD

```
A = U × Σ × Vᵀ
```

### PCA = SVD

```
Principal components = правые сингулярные V
```

---

## Урок 12: Tensor Operations

### Einsum

```
"ik,kj->ij"     — matmul
"bhtd,bhsd->bhts" — attention scores
```

---

## Урок 13: Numerical Stability

### Log-Sum-Exp Trick

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

---

## Урок 14: Norms and Distances

| Норма | Когда |
|---|---|
| L1 | Разреженные данные |
| L2 | Пространственные данные |
| Cosine | Текст, эмбеддинги |

---

## Урок 15: Statistics for ML

### Ключевые концепции

```
95% CI = x̄ ± 1.96 × (s/√n)
Bootstrap: пересэмплирование с возвратом
Cohen's d: размер эффекта
```

---

## Урок 16: Sampling Methods

### Методы

| Метод | Когда |
|---|---|
| Inverse CDF | Есть CDF |
| Temperature | LLM генерация |
| Top-k / Top-p | Контроль разнообразия |

---

## Урок 17: Linear Systems

### Normal Equations

```
w = (XᵀX)⁻¹ × Xᵀy
```

---

## Урок 18: Convex Optimization

### Метод Ньютона

```
x_new = x - H⁻¹ × grad
```

### KKT и регуляризация

```
L2: min Loss(w) s.t. ||w||² ≤ t
L1: min Loss(w) s.t. ||w||₁ ≤ t
```

---

## Урок 19: Complex Numbers

### Euler's formula

```
e^(iθ) = cos θ + i sin θ
```

---

## Урок 20: Fourier Transform

### FFT

```
DFT: O(N²)
FFT: O(N log N)
```

### Теорема о свёртке

```
Свёртка во времени = умножение в частотной области
```
