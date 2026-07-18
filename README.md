# AI Engineering from Scratch

Практический курс по математике для AI/ML — от линейной алгебры до ансамблей методов. Каждый урок: теория + рабочий код на Python без фреймворков.

> **[Конспект теории (CURS.md)](CURS.md)** — все формулы, таблицы и связи между уроками.

## О чём этот курс

Чтобы понимать, как работают нейросети, PyTorch и LLM, нужно знать математику **под капотом**. Этот курс проходит Phase 1-2 курса [AI Engineering from Scratch](https://aiengineeringfromscratch.com/) — Math Foundations + ML Fundamentals.

Все алгоритмы реализованы **с нуля**: autograd движок, оптимизаторы, PCA, SVD — без импорта `torch` или `sklearn`. Только Python и понимание.

## Структура курса

### Phase 1: Math Foundations (уроки 01-20)

| # | Урок | Что изучаем |
|---|------|-------------|
| 01 | Линейная алгебра — интуиция | Векторы, dot product, проекция, ранг, Грамм-Шмидт |
| 02 | Векторы, матрицы и операции | Матричное умножение, broadcasting, слой нейросети |
| 03 | [Матричные преобразования](phase-01-math-foundations/03-matrix_transformations.py) | Поворот, масштаб, собственные числа, определитель |
| 04 | [Calculus для ML](phase-01-math-foundations/04-calculus_for_ml.py) | Производные, градиенты, гессиан, ряд Тейлора |
| 05 | [Chain Rule & Autograd](phase-01-math-foundations/05-chain_rule_autodiff.py) | Цепное правило, micrograd, gradient checking |
| 06 | [Вероятности и распределения](phase-01-math-foundations/06-probability_distributions.py) | Bernoulli, Normal, softmax, cross-entropy |
| 07 | [Bayes' Theorem](phase-01-math-foundations/07-bayes_theorem.py) | Наивный Байес, A/B тестирование, Beta-Binomial |
| 08 | [Оптимизация](phase-01-math-foundations/08-optimization.py) | GD, SGD, Momentum, Adam, расписания lr |
| 09 | [Теория информации](phase-01-math-foundations/09-information_theory.py) | Энтропия, KL-расхождение, взаимная информация |
| 10 | [Размерность](phase-01-math-foundations/10-dimensionality_reduction.py) | PCA, t-SNE, UMAP, Kernel PCA |
| 11 | [SVD](phase-01-math-foundations/11-svd.py) | Сжатие изображений, шумоподавление, рекомендации |
| 12 | [Tensor Operations](phase-01-math-foundations/12-tensor_operations.py) | Shape, broadcasting, einsum, multi-head attention |
| 13 | [Numerical Stability](phase-01-math-foundations/13-numerical_stability.py) | Overflow, stable softmax, gradient checking, mixed precision |
| 14 | [Нормы и расстояния](phase-01-math-foundations/14-norms_distances.py) | L1, L2, cosine, Mahalanобис, Jaccard, расстояние редактирования |
| 15 | [Статистика для ML](phase-01-math-foundations/15-statistics_for_ml.py) | Корреляция, t-test, bootstrap, Cohen's d, A/B тестирование |
| 16 | [Методы семплирования](phase-01-math-foundations/16-sampling_methods.py) | Inverse CDF, rejection, temperature, top-k, top-p, MCMC |
| 17 | [Линейные системы](phase-01-math-foundations/17-linear_systems.py) | Gaussian elimination, LU, Cholesky, least squares, ridge regression |
| 18 | [Выпуклая оптимизация](phase-01-math-foundations/18-convex_optimization.py) | Выпуклость, Ньютон, множители Лагранжа, KKT, L1/L2 регуляризация |
| 19 | [Комплексные числа](phase-01-math-foundations/19-complex_numbers.py) | Комплексная арифметика, Euler, DFT, RoPE в Transformer |
| 20 | [Преобразование Фурье](phase-01-math-foundations/20-fourier_transform.py) | DFT, FFT, спектральный анализ, свёртка, спектрограммы |

### Phase 2: ML Fundamentals (уроки 21-38)

| # | Урок | Что изучаем |
|---|------|-------------|
| 21 | [Что такое ML](phase-02-ml-fundamentals/21-ml_intro.py) | Типы ML, classification/regression, overfitting, bias-variance |
| 22 | [Linear Regression](phase-02-ml-fundamentals/22-linear_regression.py) | GD vs Normal Equation, polynomial, Ridge регуляризация |
| 23 | [Logistic Regression](phase-02-ml-fundamentals/23-logistic_regression.py) | Sigmoid, binary cross-entropy, softmax, метрики |
| 24 | [Decision Trees & Random Forests](phase-02-ml-fundamentals/24-decision_trees.py) | Gini, entropy, information gain, bootstrap, feature importance |
| 25-38 | Будущие уроки | SVM, KNN, и др. |

## Быстрый старт

```bash
git clone https://github.com/alekseysorokin68/AI-Engineering-with-pyton.git
cd AI-Engineering-with-pyton
python phase-01-math-foundations/08-optimization.py
```

Или через VS Code: открой папку → `phase-01-math-foundations/` → файл → **F5**.

## Конспект

Файл `CURS.md` — оглавление со ссылками на конспекты каждой фазы. В каждой папке фазы свой `CURS.md` с полным конспектом.

## Лицензия

MIT
