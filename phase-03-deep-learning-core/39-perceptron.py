"""
39 - Perceptron from Scratch
============================
Реализация перцептрона Розенблатта без использования ML-библиотек.
Демонстрация работы на линейно разделимых данных, визуализация границы
решения и показ limitations (XOR).
"""

import random

random.seed(42)


# ============================================================
# Класс Perceptron
# ============================================================

class Perceptron:
    """Однослойный перцептрон Розенблатта."""

    def __init__(self, n_features, lr=0.1):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = lr
        self.errors_per_epoch = []

    def _dot(self, x):
        s = self.bias
        for w, xi in zip(self.weights, x):
            s += w * xi
        return s

    def predict_one(self, x):
        return 1 if self._dot(x) >= 0 else 0

    def fit(self, X, y, max_epochs=100):
        for epoch in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict_one(xi)
                err = yi - pred
                if err != 0:
                    errors += 1
                    for j in range(len(self.weights)):
                        self.weights[j] += self.lr * err * xi[j]
                    self.bias += self.lr * err
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X):
        return [self.predict_one(xi) for xi in X]


# ============================================================
# Генерация данных
# ============================================================

def make_linear_data(n=40):
    """Линейно разделимые данные: класс 0 — верхний левый угол,
    класс 1 — нижний правый угол. Разделяющая линия: x1 + x2 = 1."""
    X, y = [], []
    for _ in range(n // 2):
        x1 = random.uniform(0.0, 0.4)
        x2 = random.uniform(0.6, 1.0)
        X.append([x1, x2])
        y.append(0)
    for _ in range(n // 2):
        x1 = random.uniform(0.6, 1.0)
        x2 = random.uniform(0.0, 0.4)
        X.append([x1, x2])
        y.append(1)
    return X, y


X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_xor = [0, 1, 1, 0]


# ============================================================
# ASCII-визуализация границы решения
# ============================================================

def ascii_decision_boundary(clf, width=50, height=25):
    """Рисует ASCII-карту decision boundary. '.' = класс 0, '#' = класс 1."""
    grid = []
    for row in range(height):
        line = []
        for col in range(width):
            x1 = col / (width - 1)
            x2 = 1.0 - row / (height - 1)   # y-ось инвертирована
            pred = clf.predict_one([x1, x2])
            line.append('#' if pred == 1 else '.')
        grid.append(''.join(line))
    return grid


def print_ascii_grid(grid, X, y):
    """Печатает ASCII-карту с отмеченными точками данных."""
    W = len(grid[0])
    H = len(grid)
    canvas = [list(row) for row in grid]

    for xi, yi in zip(X, y):
        col = int(round(xi[0] * (W - 1)))
        row = int(round((1.0 - xi[1]) * (H - 1)))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))
        canvas[row][col] = 'O' if yi == 0 else 'X'

    print('+' + '-' * W + '+')
    for r in canvas:
        print('|' + ''.join(r) + '|')
    print('+' + '-' * W + '+')
    print('  O = class 0 | X = class 1 | . = predicted 0 | # = predicted 1')


# ============================================================
# Демо 1: Перцепptron на линейно разделимых данных
# ============================================================

def demo1():
    print('=' * 60)
    print('Демо 1: Перцептрон на линейно разделимых данных')
    print('=' * 60)

    X, y = make_linear_data(40)
    clf = Perceptron(n_features=2, lr=0.1)
    clf.fit(X, y, max_epochs=50)

    preds = clf.predict(X)
    correct = sum(p == t for p, t in zip(preds, y))
    accuracy = correct / len(y) * 100

    print(f'\nОбучающая выборка: {len(y)} образцов')
    print(f'Веса: {[round(w, 4) for w in clf.weights]}')
    print(f'Bias:  {round(clf.bias, 4)}')
    print(f'Точность на обучении: {correct}/{len(y)} = {accuracy:.1f}%')

    # random guessing
    rand_preds = [random.randint(0, 1) for _ in y]
    rand_correct = sum(p == t for p, t in zip(rand_preds, y))
    print(f'Random guessing:     {rand_correct}/{len(y)} = {rand_correct / len(y) * 100:.1f}%')
    print()


# ============================================================
# Демо 2: Граница решения (ASCII)
# ============================================================

def demo2():
    print('=' * 60)
    print('Демо 2: Граница решения (ASCII-визуализация)')
    print('=' * 60)

    X, y = make_linear_data(40)
    clf = Perceptron(n_features=2, lr=0.1)
    clf.fit(X, y, max_epochs=50)

    grid = ascii_decision_boundary(clf, width=50, height=25)
    print()
    print_ascii_grid(grid, X, y)
    print(f'Веса: {[round(w, 4) for w in clf.weights]}, bias: {round(clf.bias, 4)}')
    print()


# ============================================================
# Демо 3: Количество итераций до сходимости
# ============================================================

def demo3():
    print('=' * 60)
    print('Демо 3: Количество итераций до сходимости')
    print('=' * 60)

    X, y = make_linear_data(40)
    clf = Perceptron(n_features=2, lr=0.1)
    clf.fit(X, y, max_epochs=100)

    converged_epoch = len(clf.errors_per_epoch)
    for i, e in enumerate(clf.errors_per_epoch):
        if e == 0:
            converged_epoch = i + 1
            break

    print(f'\nОшибок по эпохам:')
    for i, e in enumerate(clf.errors_per_epoch):
        bar = '#' * e
        print(f'  Эпоха {i + 1:2d}: {e:2d} ошибок  {bar}')

    print(f'\nСходимость достигнута на эпохе: {converged_epoch}')
    print()


# ============================================================
# Демо 4: Перцептрон НЕ может решить XOR
# ============================================================

def demo4():
    print('=' * 60)
    print('Демо 4: Перцептрон НЕ может решить XOR')
    print('=' * 60)

    clf = Perceptron(n_features=2, lr=0.1)
    clf.fit(X_xor, y_xor, max_epochs=50)

    preds = clf.predict(X_xor)

    print('\nXOR таблица:')
    print('  x1  x2  |  target  predict')
    print('  --------|------------------')
    for xi, ti, pi in zip(X_xor, y_xor, preds):
        mark = '  OK' if ti == pi else '  FAIL'
        print(f'   {xi[0]}   {xi[1]}  |    {ti}       {pi}{mark}')

    correct = sum(p == t for p, t in zip(preds, y_xor))
    print(f'\nТочность: {correct}/4 = {correct / 4 * 100:.0f}%')
    print('Перцептрон не способен решить XOR — линейно неразделимая задача!')
    print('Решение: либо добавить скрытый слой (MLP), либо нелинейные признаки.')
    print()

    # ASCII-визуализация XOR
    print('Decision boundary для XOR:')
    grid = ascii_decision_boundary(clf, width=30, height=15)
    xor_points = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    canvas = [list(row) for row in grid]
    for x1, x2, label in xor_points:
        col = int(round(x1 * 29))
        row = int(round((1.0 - x2) * 14))
        canvas[row][col] = 'O' if label == 0 else 'X'
    print('+' + '-' * 30 + '+')
    for r in canvas:
        print('|' + ''.join(r) + '|')
    print('+' + '-' * 30 + '+')
    print('  O = 0  X = 1 — одна линия не может разделить XOR')
    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
    demo4()
