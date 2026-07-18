# AI Engineering from Scratch

Практический курс по математике для AI/ML — от линейной алгебры до методов семплирования. Каждый урок: теория + рабочий код на Python без фреймворков.

> **[Конспект теории (CURS.md)](CURS.md)** — все формулы, таблицы и связи между уроками в одном файле.

## О чём этот курс

Чтобы понимать, как работают нейросети, PyTorch и LLM, нужно знать математику **под капотом**. Этот курс проходит Phase 1 курса [AI Engineering from Scratch](https://aiengineeringfromscratch.com/) — Math Foundations.

Все алгоритмы реализованы **с нуля**: autograd движок, оптимизаторы, PCA, SVD — без импорта `torch` или `sklearn`. Только Python и понимание.

## Уроки

| # | Урок | Что изучаем |
|---|------|-------------|
| 01 | Линейная алгебра — интуиция | Векторы, dot product, проекция, ранг, Грамм-Шмидт |
| 02 | Векторы, матрицы и операции | Матричное умножение, broadcasting, слой нейросети |
| 03 | [Матричные преобразования](code/matrix_transformations.py) | Поворот, масштаб, собственные числа, определитель |
| 04 | [Calculus для ML](code/calculus_for_ml.py) | Производные, градиенты, гессиан, ряд Тейлора |
| 05 | [Chain Rule & Autograd](code/chain_rule_autodiff.py) | Цепное правило, micrograd, gradient checking |
| 06 | [Вероятности и распределения](code/probability_distributions.py) | Bernoulli, Normal, softmax, cross-entropy |
| 07 | [Bayes' Theorem](code/bayes_theorem.py) | Наивный Байес, A/B тестирование, Beta-Binomial |
| 08 | [Оптимизация](code/optimization.py) | GD, SGD, Momentum, Adam, расписания lr |
| 09 | [Теория информации](code/information_theory.py) | Энтропия, KL-расхождение, взаимная информация |
| 10 | [Размерность](code/dimensionality_reduction.py) | PCA, t-SNE, UMAP, Kernel PCA |
| 11 | [SVD](code/svd.py) | Сжатие изображений, шумоподавление, рекомендации |
| 12 | [Tensor Operations](code/tensor_operations.py) | Shape, broadcasting, einsum, multi-head attention |
| 13 | [Numerical Stability](code/numerical_stability.py) | Overflow, stable softmax, gradient checking, mixed precision |
| 14 | [Нормы и расстояния](code/norms_distances.py) | L1, L2, cosine, Mahalanобис, Jaccard, расстояние редактирования |
| 15 | [Статистика для ML](code/statistics_for_ml.py) | Корреляция, t-test, bootstrap, Cohen's d, A/B тестирование |
| 16 | [Методы семплирования](code/sampling_methods.py) | Inverse CDF, rejection, temperature, top-k, top-p, MCMC |

## Быстрый старт

```bash
git clone https://github.com/alekseysorokin68/AI-Engineering-with-pyton.git
cd AI-Engineering-with-pyton
python code/optimization.py
```

Или через VS Code: открой папку → `code/` → файл → **F5**.

## Структура

```
AI-Engineering-with-pyton/
├── README.md
├── CURS.md
├── code/
│   ├── matrix_transformations.py
│   ├── calculus_for_ml.py
│   ├── chain_rule_autodiff.py
│   ├── probability_distributions.py
│   ├── bayes_theorem.py
│   ├── optimization.py
│   ├── information_theory.py
│   ├── dimensionality_reduction.py
│   ├── svd.py
│   ├── tensor_operations.py
│   ├── numerical_stability.py
│   ├── norms_distances.py
│   ├── statistics_for_ml.py
│   └── sampling_methods.py
├── requirements.txt
└── LICENSE
```

## Конспект

Файл `CURS.md` содержит конспект всей теории по урокам 01-16 — формулы, таблицы, связи между темами. Удобно для повторения перед собеседованием или экзаменом.

## Лицензия

MIT
