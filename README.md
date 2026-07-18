# AI Engineering from Scratch

Практический курс по математике для AI/ML — от линейной алгебры до SVD. Каждый урок: теория + рабочий код на Python без фреймворков.

## О чём этот курс

Чтобы понимать, как работают нейросети, PyTorch и LLM, нужно знать математику **под капотом**. Этот курс проходит Phase 1 курса [AI Engineering from Scratch](https://aiengineeringfromscratch.com/) — Math Foundations.

Все алгоритмы реализованы **с нуля**: autograd движок, оптимизаторы, PCA, SVD — без импорта `torch` или `sklearn`. Только Python и понимание.

## Уроки

| # | Урок | Что изучаем |
|---|------|-------------|
| 01 | Линейная алгебра — интуиция | Векторы, dot product, проекция, ранг, Грамм-Шмидт |
| 02 | Векторы, матрицы и операции | Матричное умножение, broadcasting, слой нейросети |
| 03 | Матричные преобразования | Поворот, масштаб, собственные числа, определитель |
| 04 | Calculus для ML | Производные, градиенты, гессиан, ряд Тейлора |
| 05 | Chain Rule & Autograd | Цепное правило, micrograd, gradient checking |
| 06 | Вероятности и распределения | Bernoulli, Normal, softmax, cross-entropy |
| 07 | Bayes' Theorem | Наивный Байес, A/B тестирование, Beta-Binomial |
| 08 | Оптимизация | GD, SGD, Momentum, Adam, расписания lr |
| 09 | Теория информации | Энтропия, KL-расхождение, взаимная информация |
| 10 | Размерность | PCA, t-SNE, UMAP, Kernel PCA |
| 11 | SVD | Сжатие изображений, шумоподавление, рекомендации |

## Быстрый старт

```bash
git clone https://github.com/alekseysorokin68/curs_ai.git
cd curs_ai
python optimization.py
```

Или через VS Code: открой папку → файл → **F5**.

## Структура

```
curs_ai/
├── CURS.md                       — конспект теории (все уроки)
├── matrix_transformations.py     — урок 03
├── calculus_for_ml.py            — урок 04
├── chain_rule_autodiff.py        — урок 05
├── probability_distributions.py  — урок 06
├── bayes_theorem.py              — урок 07
├── optimization.py               — урок 08
├── information_theory.py         — урок 09
├── dimensionality_reduction.py   — урок 10
├── svd.py                        — урок 11
└── requirements.txt
```

## Конспект

Файл `CURS.md` содержит конспект всей теории по урокам 01-11 — формулы, таблицы, связи между темами. Удобно для повторения перед собеседованием или экзаменом.

## Лицензия

MIT
