import math
import random

random.seed(42)


# ============================================================
#  Nearest Centroid Classifier с нуля
# ============================================================

class NearestCentroid:
    def __init__(self):
        self.centroids = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = list(set(y))
        for c in self.classes:
            points = [X[i] for i in range(len(y)) if y[i] == c]
            n = len(points)
            d = len(points[0])
            centroid = [sum(points[i][j] for i in range(n)) / n for j in range(d)]
            self.centroids[c] = centroid

    def predict(self, X):
        predictions = []
        for x in X:
            best_class = None
            best_dist = float('inf')
            for c, centroid in self.centroids.items():
                dist = math.sqrt(sum((xi - ci)**2 for xi, ci in zip(x, centroid)))
                if dist < best_dist:
                    best_dist = dist
                    best_class = c
            predictions.append(best_class)
        return predictions


# ============================================================
#  Вспомогательные функции
# ============================================================

def accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

def generate_data(n_per_class=100, separation=2.0):
    """Генерирует 2 класса в 2D"""
    X = []
    y = []
    for _ in range(n_per_class):
        # Класс 0: центр в (separation/2, separation/2)
        x1 = random.gauss(separation / 2, 1.0)
        x2 = random.gauss(separation / 2, 1.0)
        X.append([x1, x2])
        y.append(0)
    for _ in range(n_per_class):
        # Класс 1: центр в (-separation/2, -separation/2)
        x1 = random.gauss(-separation / 2, 1.0)
        x2 = random.gauss(-separation / 2, 1.0)
        X.append([x1, x2])
        y.append(1)
    return X, y

def train_test_split(X, y, test_ratio=0.3):
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


# ============================================================
#  Демо 1: Supervised vs Unsupervised vs RL
# ============================================================

print("=" * 55)
print("ДЕМО 1: Три типа машинного обучения")
print("=" * 55)

print(f"""
┌───────────────────┬──────────────────────┬─────────────────────────┐
│ Тип               │ Данные               │ Задача                  │
├───────────────────┼──────────────────────┼─────────────────────────┤
│ Supervised        │ Вход + выход (метка) │ Классификация, регрессия│
│ Unsupervised      │ Только входы         │ Кластеризация, PCA      │
│ Reinforcement     │ Агент + среда        │ Оптимизация политики    │
└───────────────────┴──────────────────────┴─────────────────────────┘

Примеры:
  Supervised:    "Вот фото котов и собак, научись различать"
  Unsupervised:  "Вот покупки клиентов, найди группы"
  Reinforcement: "Играй в шахматы, +1 за победу"
""")


# ============================================================
#  Демо 2: Classification vs Regression
# ============================================================

print("=" * 55)
print("ДЕМО 2: Classification vs Regression")
print("=" * 55)

print(f"""
┌──────────────────┬──────────────────────┬─────────────────────────┐
│                  │ Classification       │ Regression              │
├──────────────────┼──────────────────────┼─────────────────────────┤
│ Выход            │ Категории            │ Числа                   │
│ Пример           │ "Спам или нет?"      │ "Цена дома?"            │
│ Loss             │ Cross-entropy        │ MSE, MAE                │
│ Решение          │ Граница классов      │ Кривая, подходящая данные│
└──────────────────┴──────────────────────┴─────────────────────────┘
""")


# ============================================================
#  Демо 3: Train / Validation / Test Split
# ============================================================

print("=" * 55)
print("ДЕМО 3: Train / Validation / Test Split")
print("=" * 55)

X, y = generate_data(150, 2.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3)

print(f"\nВсего сэмплов: {len(X)}")
print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test:  {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")


# ============================================================
#  Демо 4: Nearest Centroid Classifier
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Nearest Centroid Classifier")
print("=" * 55)

clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print(f"\nОбучающие данные: {len(X_train)} сэмплов")
print(f"Центр класса 0: ({clf.centroids[0][0]:.3f}, {clf.centroids[0][1]:.3f})")
print(f"Центр класса 1: ({clf.centroids[1][0]:.3f}, {clf.centroids[1][1]:.3f})")
print(f"\nAccuracy на test: {acc:.4f}")


# ============================================================
#  Демо 5: Baseline сравнение
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Baseline — случайное угадывание")
print("=" * 55)

# Случайный baseline
random_preds = [random.choice([0, 1]) for _ in range(len(y_test))]
random_acc = accuracy(y_test, random_preds)

# Baseline: всегда предсказываем самый частый класс
from collections import Counter
most_common = Counter(y_train).most_common(1)[0][0]
majority_preds = [most_common] * len(y_test)
majority_acc = accuracy(y_test, majority_preds)

print(f"\nСлучайное угадывание: {random_acc:.4f} (ожидается ~0.5)")
print(f"Всегда класс {most_common}: {majority_acc:.4f}")
print(f"Nearest Centroid:      {acc:.4f}")
print(f"\n→ Nearest Centroid >> baseline, модель работает!")


# ============================================================
#  Демо 6: Overfitting vs Underfitting
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Overfitting vs Underfitting")
print("=" * 55)

# Underfitting: линейная модель на нелинейных данных
print(f"\nUnderfitting:")
print(f"  Модель слишком простая → high bias")
print(f"  Train error высокий, test error высокий")
print(f"  Пример: прямая линия на кривых данных")

# Overfitting: слишком сложная модель
print(f"\nOverfitting:")
print(f"  Модель слишком сложная → high variance")
print(f"  Train error низкий, test error высокий")
print(f"  Пример: полином степени 20 на 10 точках")
print(f"  Запоминает шум, не обобщает")

# Good fit
print(f"\nGood Fit:")
print(f"  Правильная сложность → баланс bias-variance")
print(f"  Train error и test error оба приемлемы")


# ============================================================
#  Демо 7: Bias-Variance Tradeoff
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Bias-Variance Tradeoff")
print("=" * 55)

print(f"""
┌──────────────────┬──────────┬──────────┬──────────────┐
│ Сложность модели │ Bias     │ Variance │ Результат    │
├──────────────────┼──────────┼──────────┼──────────────┤
│ Слишком низкая   │ Высокий  │ Низкий   │ Underfitting │
│ Сбалансированная │ Средний  │ Средний  │ Good fit     │
│ Слишком высокая  │ Низкий   │ Высокий  │ Overfitting  │
└──────────────────┴──────────┴──────────┴──────────────┘

Total error = Bias² + Variance + Irreducible noise

→ Нельзя уменьшить шум данных, но можно найти баланс
""")


# ============================================================
#  Демо 8: Когда НЕ использовать ML
# ============================================================

print("=" * 55)
print("ДЕМО 8: Когда НЕ использовать ML")
print("=" * 55)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│ Когда ML НЕ нужен:                                          │
├─────────────────────────────────────────────────────────────┤
│ ✓ Правила простые и понятные (if-else)                      │
│ ✓ Нет данных или очень мало                                 │
│ ✓ Нужна 100% гарантия (медицина, ядерные реакторы)         │
│ ✓ Достаточно таблицы или эвристики                          │
│ ✓ Нужна объяснимость (кредиты, страхование)                 │
│ ✓ Правила меняются быстрее, чем переобучение                │
└─────────────────────────────────────────────────────────────┘
""")
