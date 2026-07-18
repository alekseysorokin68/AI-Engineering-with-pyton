# AI Engineering from Scratch — Конспект

> Все формулы, таблицы и связи между уроками.

## Фазы курса

| Фаза | Описание | Уроки | Конспект |
|------|----------|-------|----------|
| Phase 1 | Math Foundations | 01-20 | [CURS.md](phase-01-math-foundations/CURS.md) |
| Phase 2 | ML Fundamentals | 21-38 | [CURS.md](phase-02-ml-fundamentals/CURS.md) |
| Phase 3 | Deep Learning Core | 39-51 | [CURS.md](phase-03-deep-learning-core/CURS.md) |
| Phase 4 | Vision | 52-60 | [CURS.md](phase-04-vision/CURS.md) |
| Phase 5 | NLP | 61-68 | [CURS.md](phase-05-nlp/CURS.md) |
| Phase 6 | Speech & Audio | 69-75 | [CURS.md](phase-06-speech-audio/CURS.md) |
| Phase 7 | Reinforcement Learning | 76-82 | [CURS.md](phase-07-reinforcement-learning/CURS.md) |
| Phase 8 | Generative AI | 83-90 | [CURS.md](phase-08-generative-ai/CURS.md) |

## Формулы для справки

```python
# Скалярное произведение
a · b = sum(ai * bi)

# Матричное умножение
(m × n) @ (n × p) = (m × p)

# Определитель 2×2
det([[a,b],[c,d]]) = ad - bc

# Собственные числа (2×2)
λ² - (a+d)λ + (ad-bc) = 0

# Проекция
proj_b(a) = (a·b / b·b) × b

# Числовая производная
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

# Градиентный спуск
w = w - lr * dL/dw

# Softmax
softmax(z) = exp(z) / sum(exp(z))
log_softmax(z) = z - log(sum(exp(z)))

# Cross-entropy
L = -log_softmax(logits)[target]

# Нормальное распределение
f(x) = exp(-(x-μ)² / (2σ²)) / √(2πσ²)

# Байес
P(A|B) = P(B|A) * P(A) / P(B)

# Наивный Байес
score(class) = P(class) * prod(P(feature_i | class))

# Beta-Binomial обновление
Beta(a, b) + s successes, f failures → Beta(a+s, b+f)

# SGD + Momentum
v = beta * v + gradient
w = w - lr * v

# Adam
m = beta1 * m + (1-beta1) * g
v = beta2 * v + (1-beta2) * g^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

# Информационное содержание
I(x) = -log(p(x))

# Энтропия
H(P) = -sum(p(x) * log(p(x)))

# Перекрёстная энтропия
H(P, Q) = -sum(p(x) * log(q(x)))

# KL-расхождение
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))

# Взаимная информация
I(X; Y) = H(X) + H(Y) - H(X,Y)

# Перплексия
Perplexity = 2^H(P,Q)  (биты)  или  e^H(P,Q)  (наты)

# Label smoothing
soft = (1-epsilon) * hard + epsilon / num_classes

# Normal Equations
w = (X^T X)^-1 X^T y

# Ridge Regression
w = (X^T X + λI)^-1 X^T y
```
