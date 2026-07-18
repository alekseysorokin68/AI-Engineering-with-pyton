"""
31 - Ансамблевые методы (Ensemble Methods) с нуля
==================================================
Реализация ансамблевых методов без sklearn:
- Decision Tree (базовая модель)
- Bagging (Bootstrap Aggregating)
- AdaBoost
- Voting Classifier (Majority Vote)
- Сравнение всех подходов

Ансамбли объединяют несколько «слабых» моделей в одну «сильную»,
снижая variance (Bagging) или bias+variance (Boosting).
"""

import random
import math
from collections import Counter

random.seed(42)

# ============================================================
# 1. СИНТЕТИЧЕСКИЕ ДАННЫЕ
# ============================================================

def generate_dataset(n=200, noise=0.1):
    """
    Генерация двумерного классификационного датасета.
    Класс 0: центр (-1, -1), класс 1: центр (1, 1).
    """
    X, y = [], []
    for _ in range(n):
        label = random.randint(0, 1)
        if label == 0:
            x1 = random.gauss(-1, 1)
            x2 = random.gauss(-1, 1)
        else:
            x1 = random.gauss(1, 1)
            x2 = random.gauss(1, 1)
        if random.random() < noise:
            label = 1 - label
        X.append([x1, x2])
        y.append(label)
    return X, y


def train_test_split(X, y, test_ratio=0.3):
    """Разбиение на train/test."""
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


# ============================================================
# 2. РЕШАЮЩЕЕ ДЕРЕВО (базовая модель)
# ============================================================

class DecisionStump:
    """
    Решающий пень (Decision Stump) — дерево глубины 1.
    Ищет лучший порог по одному признаку.
    """

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_label = None
        self.right_label = None

    def fit(self, X, y, sample_weights=None):
        """Обучение: перебор признаков и порогов."""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        if sample_weights is None:
            sample_weights = [1.0 / n_samples] * n_samples

        best_gini = float('inf')
        best_config = None

        for feat in range(n_features):
            # Уникальные значения признака как кандидаты на порог
            values = sorted(set(X[i][feat] for i in range(n_samples)))
            thresholds = []
            for i in range(len(values) - 1):
                thresholds.append((values[i] + values[i + 1]) / 2)
            # Добавляем границы за крайними значениями
            thresholds.insert(0, values[0] - 1)
            thresholds.append(values[-1] + 1)

            for thr in thresholds:
                left_y, left_w = [], []
                right_y, right_w = [], []
                for i in range(n_samples):
                    if X[i][feat] <= thr:
                        left_y.append(y[i])
                        left_w.append(sample_weights[i])
                    else:
                        right_y.append(y[i])
                        right_w.append(sample_weights[i])

                if not left_y or not right_y:
                    continue

                # Взвешенный Gini
                gini = self._weighted_gini(left_y, left_w) * sum(left_w) + \
                       self._weighted_gini(right_y, right_w) * sum(right_w)

                if gini < best_gini:
                    best_gini = gini
                    best_config = (feat, thr, left_y, left_w, right_y, right_w)

        if best_config is None:
            self.feature_idx = 0
            self.threshold = 0
            self.left_label = Counter(y).most_common(1)[0][0]
            self.right_label = self.left_label
            return

        feat, thr, left_y, left_w, right_y, right_w = best_config
        self.feature_idx = feat
        self.threshold = thr
        self.left_label = self._majority(left_y)
        self.right_label = self._majority(right_y)

    def _weighted_gini(self, labels, weights):
        """Взвешенный коэффициент Джини."""
        total = sum(weights)
        if total == 0:
            return 0
        counts = {}
        for lbl, w in zip(labels, weights):
            counts[lbl] = counts.get(lbl, 0) + w
        impurity = 1.0
        for c in counts.values():
            p = c / total
            impurity -= p * p
        return impurity

    def _majority(self, labels):
        """Мажоритарная метка."""
        return Counter(labels).most_common(1)[0][0]

    def predict_one(self, x):
        if x[self.feature_idx] <= self.threshold:
            return self.left_label
        else:
            return self.right_label

    def predict(self, X):
        return [self.predict_one(x) for x in X]


# ============================================================
# 3. BAGGING (Bootstrap Aggregating)
# ============================================================

class BaggingClassifier:
    """
    Bagging: обучает n_models базовых моделей на bootstrap-выборках.
    Финальное предсказание — мажоритарное голосование.
    """

    def __init__(self, n_models=10):
        self.n_models = n_models
        self.models = []

    def fit(self, X, y):
        """Обучение на bootstrap-выборках."""
        self.models = []
        n = len(X)

        for m in range(self.n_models):
            # Bootstrap: выборка с возвращением
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            stump = DecisionStump()
            stump.fit(X_boot, y_boot)
            self.models.append(stump)

        return self

    def predict(self, X):
        """Мажоритарное голосование."""
        all_preds = [model.predict(X) for model in self.models]

        results = []
        for i in range(len(X)):
            votes = [all_preds[m][i] for m in range(self.n_models)]
            results.append(Counter(votes).most_common(1)[0][0])
        return results


# ============================================================
# 4. ADABOOST
# ============================================================

class AdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting):
    1. Инициализировать веса образцов равномерно.
    2. Обучить слабую модель на взвешенных данных.
    3. Вычислить ошибку модели.
    4. Вычислить вес модели (alpha).
    5. Обновить веса образцов: ошибочные получают больший вес.
    6. Повторить n_rounds раз.
    """

    def __init__(self, n_rounds=10):
        self.n_rounds = n_rounds
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = len(X)
        # Инициализация весов образцов
        weights = [1.0 / n] * n

        self.stumps = []
        self.alphas = []

        for r in range(self.n_rounds):
            # Обучаем пень на текущих весах
            stump = DecisionStump()
            stump.fit(X, y, sample_weights=weights)
            preds = stump.predict(X)

            # Вычисляем ошибку (взвешенная доля неправильных)
            err = 0.0
            for i in range(n):
                if preds[i] != y[i]:
                    err += weights[i]

            err = max(err, 1e-10)  # защита от деления на 0

            # Вес модели: чем меньше ошибка, тем больше alpha
            alpha = 0.5 * math.log((1 - err) / err)
            self.stumps.append(stump)
            self.alphas.append(alpha)

            # Обновляем веса образцов
            new_weights = []
            for i in range(n):
                if preds[i] == y[i]:
                    new_w = weights[i] * math.exp(-alpha)
                else:
                    new_w = weights[i] * math.exp(alpha)
                new_weights.append(new_w)

            # Нормализация весов
            total = sum(new_weights)
            weights = [w / total for w in new_weights]

        return self

    def predict(self, X):
        """Взвешенное голосование: sum(alpha * prediction)."""
        n = len(X)
        results = []
        for i in range(n):
            score = 0.0
            for stump, alpha in zip(self.stumps, self.alphas):
                pred = stump.predict_one(X[i])
                score += alpha * (1 if pred == 1 else -1)
            results.append(1 if score >= 0 else 0)
        return results


# ============================================================
# 5. VOTING CLASSIFIER
# ============================================================

class VotingClassifier:
    """
    Voting Classifier: объединяет n моделей разного типа.
    Голосование: мажоритарное (hard voting).
    """

    def __init__(self, models):
        """
        models: список кортежей (name, model), где model имеет
        методы fit(X, y) и predict(X).
        """
        self.models = models

    def fit(self, X, y):
        """Обучение всех моделей."""
        for name, model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """Мажоритарное голосование."""
        all_preds = []
        for name, model in self.models:
            preds = model.predict(X)
            all_preds.append(preds)

        results = []
        for i in range(len(X)):
            votes = [all_preds[m][i] for m in range(len(self.models))]
            results.append(Counter(votes).most_common(1)[0][0])
        return results


# ============================================================
# 6. МЕТРИКИ
# ============================================================

def accuracy(y_true, y_pred):
    """Accuracy: доля правильных ответов."""
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def precision(y_true, y_pred, label=1):
    """Precision для заданного класса."""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true, y_pred, label=1):
    """Recall для заданного класса."""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1(y_true, y_pred, label=1):
    """F1-score для заданного класса."""
    p = precision(y_true, y_pred, label)
    r = recall(y_true, y_pred, label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def print_metrics(name, y_true, y_pred):
    """Вывод метрик модели."""
    acc = accuracy(y_true, y_pred)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f = f1(y_true, y_pred)
    print(f"  {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {p:.4f}")
    print(f"    Recall:    {r:.4f}")
    print(f"    F1-score:  {f:.4f}")
    return acc


# ============================================================
# ДЕМОНСТРАЦИИ
# ============================================================

def demo1_bagging():
    """Демо 1: Bagging vs одиночное дерево"""
    print("=" * 60)
    print("ДЕМО 1: Bagging vs одиночное дерево")
    print("=" * 60)

    X, y = generate_dataset(n=200, noise=0.15)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # Одиночное дерево
    single = DecisionStump()
    single.fit(X_train, y_train)
    pred_single = single.predict(X_test)

    # Bagging из 20 деревьев
    bagging = BaggingClassifier(n_models=20)
    bagging.fit(X_train, y_train)
    pred_bagging = bagging.predict(X_test)

    print(f"\n  Размер обучающей выборки: {len(X_train)}")
    print(f"  Размер тестовой выборки:  {len(X_test)}")
    print()
    print_metrics("Одиночное дерево (depth=1)", y_test, pred_single)
    print()
    print_metrics("Bagging (20 деревьев)", y_test, pred_bagging)

    acc_single = accuracy(y_test, pred_single)
    acc_bagging = accuracy(y_test, pred_bagging)
    print(f"\n  Улучшение Bagging: +{(acc_bagging - acc_single)*100:.1f}%")
    print()


def demo2_adaboost():
    """Демо 2: AdaBoost — последовательное обучение"""
    print("=" * 60)
    print("ДЕМО 2: AdaBoost — последовательное обучение")
    print("=" * 60)

    X, y = generate_dataset(n=200, noise=0.15)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # Анализ по раундам
    print("\n  Процесс обучения AdaBoost:")
    print("  " + "-" * 45)
    print(f"  {'Раунд':<8} {'Модель weight':<15} {'Accuracy':<10}")
    print("  " + "-" * 45)

    for n_rounds in [1, 2, 3, 5, 10, 20]:
        adaboost = AdaBoostClassifier(n_rounds=n_rounds)
        adaboost.fit(X_train, y_train)
        preds = adaboost.predict(X_test)
        acc = accuracy(y_test, preds)

        # Показываем вес последней добавленной модели
        last_alpha = adaboost.alphas[-1]
        print(f"  {n_rounds:<8} {last_alpha:<15.4f} {acc:<10.4f}")

    # Детали: что происходит внутри
    print("\n  ДеталиAdaBoost (5 раундов):")
    adaboost = AdaBoostClassifier(n_rounds=5)
    adaboost.fit(X_train, y_train)

    for i, (stump, alpha) in enumerate(zip(adaboost.stumps, adaboost.alphas)):
        print(f"    Раунд {i+1}: пень по признаку {stump.feature_idx}, "
              f"порог={stump.threshold:.3f}, вес модели alpha={alpha:.4f}")
    print()


def demo3_voting():
    """Демо 3: Voting classifier (3 модели)"""
    print("=" * 60)
    print("ДЕМО 3: Voting Classifier (3 модели)")
    print("=" * 60)

    X, y = generate_dataset(n=200, noise=0.15)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # Три разных «слабых» классификатора
    bagging_5 = BaggingClassifier(n_models=5)
    bagging_20 = BaggingClassifier(n_models=20)
    adaboost_10 = AdaBoostClassifier(n_rounds=10)

    voting = VotingClassifier([
        ("Bagging(5)", bagging_5),
        ("Bagging(20)", bagging_20),
        ("AdaBoost(10)", adaboost_10),
    ])

    voting.fit(X_train, y_train)
    pred_voting = voting.predict(X_test)

    # Отдельные предсказания для сравнения
    pred_b5 = bagging_5.predict(X_test)
    pred_b20 = bagging_20.predict(X_test)
    pred_ab = adaboost_10.predict(X_test)

    print(f"\n  Индивидуальные результаты:")
    print_metrics("Bagging (5 деревьев)", y_test, pred_b5)
    print()
    print_metrics("Bagging (20 деревьев)", y_test, pred_b20)
    print()
    print_metrics("AdaBoost (10 раундов)", y_test, pred_ab)
    print()
    print_metrics("Voting (мажоритарное)", y_test, pred_voting)

    # Анализ голосов
    print("\n  Анализ голосования (первые 20 тестовых образцов):")
    all_preds = [pred_b5[:20], pred_b20[:20], pred_ab[:20], pred_voting[:20]]
    names = ["B5", "B20", "Ada", "Vote", "True"]
    print(f"  {'#':<4} {' '.join(f'{n:>6}' for n in names)}")
    print("  " + "-" * 40)
    for i in range(min(20, len(y_test))):
        vals = [str(p[i]) for p in all_preds] + [str(y_test[i])]
        print(f"  {i:<4} {' '.join(f'{v:>6}' for v in vals)}")
    print()


def demo4_comparison():
    """Демо 4: Сравнение всех методов"""
    print("=" * 60)
    print("ДЕМО 4: Сравнение всех методов")
    print("=" * 60)

    # Несколько датасетов с разным уровнем шума
    configs = [
        ("Мало шума (5%)", 0.05),
        ("Средний шум (15%)", 0.15),
        ("Много шума (25%)", 0.25),
    ]

    results = {}

    for dataset_name, noise in configs:
        print(f"\n  Датасет: {dataset_name}")
        print("  " + "-" * 50)

        X, y = generate_dataset(n=200, noise=noise)
        X_train, y_train, X_test, y_test = train_test_split(X, y)

        methods = {}

        # Одиночное дерево
        tree = DecisionStump()
        tree.fit(X_train, y_train)
        methods["Одиночное дерево"] = tree.predict(X_test)

        # Bagging (10)
        bag10 = BaggingClassifier(n_models=10)
        bag10.fit(X_train, y_train)
        methods["Bagging (10)"] = bag10.predict(X_test)

        # Bagging (30)
        bag30 = BaggingClassifier(n_models=30)
        bag30.fit(X_train, y_train)
        methods["Bagging (30)"] = bag30.predict(X_test)

        # AdaBoost (5)
        ada5 = AdaBoostClassifier(n_rounds=5)
        ada5.fit(X_train, y_train)
        methods["AdaBoost (5)"] = ada5.predict(X_test)

        # AdaBoost (15)
        ada15 = AdaBoostClassifier(n_rounds=15)
        ada15.fit(X_train, y_train)
        methods["AdaBoost (15)"] = ada15.predict(X_test)

        # Voting
        voting = VotingClassifier([
            ("B10", BaggingClassifier(n_models=10)),
            ("A10", AdaBoostClassifier(n_rounds=10)),
        ])
        voting.fit(X_train, y_train)
        methods["Voting"] = voting.predict(X_test)

        print(f"  {'Метод':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("  " + "-" * 65)

        for name, preds in methods.items():
            acc = accuracy(y_test, preds)
            p = precision(y_test, preds)
            r = recall(y_test, preds)
            f = f1(y_test, preds)
            results.setdefault(name, []).append(acc)
            print(f"  {name:<25} {acc:<10.4f} {p:<10.4f} {r:<10.4f} {f:<10.4f}")

    # Средние результаты
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ (среднее по 3 датасетам)")
    print("=" * 60)
    print(f"\n  {'Метод':<25} {'Ср. Accuracy':<15}")
    print("  " + "-" * 40)
    for name, accs in sorted(results.items(), key=lambda x: -sum(x[1])):
        avg = sum(accs) / len(accs)
        bar = "#" * int(avg * 40)
        print(f"  {name:<25} {avg:<15.4f} {bar}")
    print()


# ============================================================
# ГЛАВНАЯ ПРОГРАММА
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║   Ансамблевые методы (Ensemble Methods) с нуля        ║")
    print("║   Bagging • AdaBoost • Voting • Сравнение             ║")
    print("╚" + "═" * 58 + "╝")
    print()

    demo1_bagging()
    demo2_adaboost()
    demo3_voting()
    demo4_comparison()

    print("=" * 60)
    print("КЛЮЧЕВЫЕ ВЫВОДЫ")
    print("=" * 60)
    print("""
  1. BAGGING снижает дисперсию (variance), обучая модели на
     разных bootstrap-подвыборках и голосуя. Эффективен для
     переобучённых моделей (глубоких деревьев).

  2. ADABOOST последовательно обучает модели, фокусируя
     внимание на ошибках предыдущих. Снижает и bias, и variance.
     Каждая модель получает вес обратно пропорционально ошибке.

  3. VOTING объединяет разные типы моделей. Разнообразие
     моделей (heterogeneity) — ключ к лучшему результату.

  4. ансамбли ВСЕГДА лучше одиночной слабой модели, но
     стоят дороже по вычислениям и сложнее для интерпретации.
""")
