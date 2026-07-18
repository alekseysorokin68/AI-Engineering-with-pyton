# AI Engineering from Scratch — Курс

Файлы к курсу "AI Engineering from Scratch" (Phase 1: Math Foundations).

## Материалы

- **`CURS.md`** — конспект теории по всем урокам (для повторения)

## Уроки (текст)

| # | Тема |
|---|------|
| 01 | Линейная алгебра — интуиция (векторы, dot product, проекция, ранг) |
| 02 | Векторы, матрицы и операции (умножение, broadcasting, нейросеть-слой) |

## Уроки (код)

| # | Файл | Тема |
|---|------|------|
| 03 | `matrix_transformations.py` | Матричные преобразования, собственные числа |
| 04 | `calculus_for_ml.py` | Производные, градиентный спуск |
| 05 | `chain_rule_autodiff.py` | Цепное правило, autograd с нуля |
| 06 | `probability_distributions.py` | Вероятности, распределения, softmax, cross-entropy |
| 07 | `bayes_theorem.py` | Байес, наивный Байес, A/B тестирование |
| 08 | `optimization.py` | GD, SGD, Momentum, Adam, расписания lr |
| 09 | `information_theory.py` | Энтропия, cross-entropy, KL, взаимная информация |
| 10 | `dimensionality_reduction.py` | PCA, t-SNE, UMAP, Kernel PCA |
| 11 | `svd.py` | SVD, truncated SVD, сжатие, шумоподавление |

## Запуск

```bash
python matrix_transformations.py
python calculus_for_ml.py
python chain_rule_autodiff.py
python probability_distributions.py
python bayes_theorem.py
python optimization.py
python information_theory.py
python dimensionality_reduction.py
python svd.py
```

Или через VS Code: открой файл → F5 (или Ctrl+F5).

## Структура

```
curs_ai/
├── CURS.md                    — конспект теории (все уроки)
├── matrix_transformations.py  — урок 03
├── calculus_for_ml.py         — урок 04
├── chain_rule_autodiff.py     — урок 05
├── probability_distributions.py — урок 06
├── bayes_theorem.py           — урок 07
├── optimization.py            — урок 08
├── information_theory.py      — урок 09
├── dimensionality_reduction.py — урок 10
├── svd.py                     — урок 11
├── requirements.txt
└── .vscode/
```
