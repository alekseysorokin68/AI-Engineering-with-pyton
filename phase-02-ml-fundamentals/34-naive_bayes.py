"""
Наивный Байес (Naive Bayes) — реализация с нуля

Алгоритм классификации, основанный на теореме Байеса с допущением
условной независимости признаков.

Теорема Байеса:
    P(C|X) = P(X|C) * P(C) / P(X)

Разновидности:
    1. GaussianNB — признаки распределены нормально (непрерывные данные)
    2. MultinomialNB — признаки — частоты/подсчёты (текстовые данные)

Все вычисления в log-пространстве для численной стабильности.
Laplace smoothing предотвращает нулевые вероятности.
"""

import math
import random

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def mean(values):
    """Среднее арифметическое."""
    return sum(values) / len(values)


def variance(values, mu):
    """Дисперсия (с параметром mu, не выборочная)."""
    return sum((x - mu) ** 2 for x in values) / len(values)


def normal_pdf(x, mu, sigma_sq):
    """Плотность нормального распределения (log)."""
    if sigma_sq == 0:
        return 0.0
    return -0.5 * (math.log(2 * math.pi * sigma_sq) + (x - mu) ** 2 / sigma_sq)


def log_sum_exp(log_values):
    """Численно стабильный log(exp(a1) + exp(a2) + ...)."""
    max_val = max(log_values)
    if max_val == float('-inf'):
        return float('-inf')
    total = sum(math.exp(v - max_val) for v in log_values)
    return max_val + math.log(total)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes для непрерывных признаков.

    Каждый признак i для класса c моделируется нормальным распределением
    N(mu_c_i, sigma_c_i^2).
    """

    def __init__(self):
        self.classes = []       # уникальные классы
        self.priors = {}        # log P(C)
        self.means = {}         # means[cls][feature_idx]
        self.variances = {}     # variances[cls][feature_idx]

    def fit(self, X, y):
        """Обучение: оценка параметров распределений."""
        self.classes = sorted(set(y))
        n_features = len(X[0])

        for cls in self.classes:
            indices = [i for i, yi in enumerate(y) if yi == cls]
            X_cls = [X[i] for i in indices]

            # log prior: P(C) = N_C / N
            self.priors[cls] = math.log(len(indices) / len(y))

            # среднее и дисперсия по каждому признаку
            self.means[cls] = []
            self.variances[cls] = []
            for j in range(n_features):
                feature_vals = [row[j] for row in X_cls]
                mu = mean(feature_vals)
                var = variance(feature_vals, mu)
                # Защита от нулевой дисперсии
                if var < 1e-9:
                    var = 1e-9
                self.means[cls].append(mu)
                self.variances[cls].append(var)

    def _log_likelihood(self, x, cls):
        """log P(X|C) для конкретного класса (условная независимость)."""
        ll = 0.0
        for j, xj in enumerate(x):
            ll += normal_pdf(xj, self.means[cls][j], self.variances[cls][j])
        return ll

    def predict_one(self, x):
        """Предсказание класса для одного образца."""
        log_posteriors = []
        for cls in self.classes:
            lp = self.priors[cls] + self._log_likelihood(x, cls)
            log_posteriors.append((lp, cls))
        log_posteriors.sort(reverse=True)
        return log_posteriors[0][1]

    def predict(self, X):
        """Предсказание классов для набора образцов."""
        return [self.predict_one(x) for x in X]

    def score(self, X, y):
        """Точность (accuracy)."""
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# Multinomial Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes для дискретных признаков (подсчёты/частоты).

    Используется для классификации текста: TF-векторы или подсчёт слов.
    P(x_i | C) = (count(x_i, C) + alpha) / (sum(count(·, C)) + alpha * V)

    alpha — параметр Laplace smoothing.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha          # Laplace smoothing параметр
        self.classes = []
        self.priors = {}            # log P(C)
        self.feature_log_probs = {} # log P(x_i | C) для каждого признака

    def fit(self, X, y):
        """
        Обучение.
        X — матрица подсчётов (n_samples x n_features), значения >= 0.
        y — метки классов.
        """
        self.classes = sorted(set(y))
        n_samples = len(X)
        n_features = len(X[0])

        for cls in self.classes:
            indices = [i for i, yi in enumerate(y) if yi == cls]
            X_cls = [X[i] for i in indices]
            n_cls = len(X_cls)

            # log prior
            self.priors[cls] = math.log(n_cls / n_samples)

            # сумма подсчётов по всем признакам для класса
            feature_counts = [0.0] * n_features
            for row in X_cls:
                for j in range(n_features):
                    feature_counts[j] += row[j]

            total_count = sum(feature_counts)

            # log P(x_i | C) с Laplace smoothing
            self.feature_log_probs[cls] = [
                math.log((feature_counts[j] + self.alpha) / (total_count + self.alpha * n_features))
                for j in range(n_features)
            ]

    def predict_one(self, x):
        """Предсказание для одного образца."""
        log_posteriors = []
        for cls in self.classes:
            ll = sum(xj * self.feature_log_probs[cls][j] for j, xj in enumerate(x))
            lp = self.priors[cls] + ll
            log_posteriors.append((lp, cls))
        log_posteriors.sort(reverse=True)
        return log_posteriors[0][1]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# Минимальная логистическая регрессия (для сравнения)
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegressionSimple:
    """
    Простая логистическая регрессия (бинарная, one-vs-rest для мультикласса).
    Градиентный спуск.
    """

    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.classes = []
        self.weights = {}
        self.biases = {}

    def _sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        self.classes = sorted(set(y))
        n_features = len(X[0])

        for cls in self.classes:
            # Бинарная метка: 1 если класс, иначе 0
            y_bin = [1 if yi == cls else 0 for yi in y]

            w = [0.0] * n_features
            b = 0.0

            for _ in range(self.n_iter):
                for i, xi in enumerate(X):
                    z = sum(xi[j] * w[j] for j in range(n_features)) + b
                    pred = self._sigmoid(z)
                    error = pred - y_bin[i]
                    for j in range(n_features):
                        w[j] -= self.lr * error * xi[j]
                    b -= self.lr * error

            self.weights[cls] = w
            self.biases[cls] = b

    def predict_one(self, x):
        best_cls = self.classes[0]
        best_score = float('-inf')
        for cls in self.classes:
            z = sum(x[j] * self.weights[cls][j] for j in range(len(x))) + self.biases[cls]
            if z > best_score:
                best_score = z
                best_cls = cls
        return best_cls

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# Генерация данных
# ─────────────────────────────────────────────────────────────────────────────

def generate_gaussian_data():
    """Генерация 2D Gaussian данных для 3 классов."""
    random.seed(42)

    centers = [
        (2.0, 2.0),
        (6.0, 2.0),
        (4.0, 6.0),
    ]
    spread = 0.8

    X, y = [], []
    for cls_idx, (cx, cy) in enumerate(centers):
        for _ in range(30):
            x1 = random.gauss(cx, spread)
            x2 = random.gauss(cy, spread)
            X.append([x1, x2])
            y.append(cls_idx)
    return X, y


def generate_text_data():
    """
    Генерация текстовых данных (мешок слов).
    Каждый документ — вектор подсчётов слов из словаря.
    """
    random.seed(42)

    vocab = [
        "python", "java", "code", "function", "class",
        "data", "algorithm", "run", "test", "debug",
        "machine", "learning", "model", "train", "predict",
        "neural", "network", "layer", "loss", "train",
        "stock", "market", "price", "trade", "invest",
        "goal", "team", "score", "game", "play",
    ]

    # Шаблоны для каждого класса (индексы слов с высокой вероятностью)
    templates = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],       # programming
        1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # ML
        2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  # sports/finance
    }

    X, y = [], []
    for cls_idx, template in templates.items():
        for _ in range(40):
            doc = [0] * len(vocab)
            # Каждое слово из шаблона появляется 1-3 раза
            for word_idx in template:
                doc[word_idx] = random.randint(1, 3)
            # Добавляем немного шума (случайные слова)
            for _ in range(random.randint(0, 3)):
                noise_idx = random.randint(0, len(vocab) - 1)
                doc[noise_idx] += 1
            X.append(doc)
            y.append(cls_idx)

    return X, y, vocab


# ─────────────────────────────────────────────────────────────────────────────
# Демонстрации
# ─────────────────────────────────────────────────────────────────────────────

def demo1_gaussian_nb():
    """Демо 1: Gaussian Naive Bayes на числовых данных."""
    print("=" * 70)
    print("ДЕМО 1: Gaussian Naive Bayes — классификация числовых данных")
    print("=" * 70)

    X, y = generate_gaussian_data()
    print(f"\nДатасет: {len(X)} образцов, {len(set(y))} класса, {len(X[0])} признака")
    print(f"Классы: 0, 1, 2")
    print(f"Пример образца: X[0] = {[round(v, 2) for v in X[0]]}, y[0] = {y[0]}")

    # Разделяем на train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Оценка параметров
    print("\n--- Обученные параметры ---")
    for cls in model.classes:
        print(f"\n  Класс {cls}:")
        print(f"    log P(C) = {model.priors[cls]:.4f}  "
              f"(P(C) ≈ {math.exp(model.priors[cls]):.4f})")
        for j in range(len(X[0])):
            print(f"    Признак {j}: μ = {model.means[cls][j]:.4f}, "
                  f"σ² = {model.variances[cls][j]:.4f}")

    # Предсказания
    accuracy = model.score(X_test, y_test)
    print(f"\nТочность на test: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)})")

    # Показать несколько предсказаний
    print("\n--- Примеры предсказаний ---")
    for i in range(min(5, len(X_test))):
        pred = model.predict_one(X_test[i])
        true = y_test[i]
        status = "✓" if pred == true else "✗"
        print(f"  {status}  Предсказано: {pred}, Истинно: {true}, "
              f"X = {[round(v, 2) for v in X_test[i]]}")

    print()


def demo2_multinomial_nb():
    """Демо 2: Multinomial Naive Bayes для классификации текста."""
    print("=" * 70)
    print("ДЕМО 2: Multinomial Naive Bayes — классификация текста")
    print("=" * 70)

    X, y, vocab = generate_text_data()
    class_names = ["Programming", "Machine Learning", "Sports/Finance"]

    print(f"\nСловарь: {len(vocab)} слов")
    print(f"Классы: {class_names}")
    print(f"Датасет: {len(X)} документов")

    # Train/test split
    all_indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(all_indices)
    split = int(0.8 * len(X))
    train_idx = all_indices[:split]
    test_idx = all_indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nТочность: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)})")

    # Показать log P(слово | класс) для ключевых слов
    print("\n--- Топ-5 слов по log P(word | class) ---")
    for cls_idx, cls_name in enumerate(class_names):
        log_probs = model.feature_log_probs[cls_idx]
        top_words = sorted(range(len(vocab)), key=lambda j: log_probs[j], reverse=True)[:5]
        words = [(vocab[j], log_probs[j]) for j in top_words]
        print(f"\n  {cls_name}:")
        for word, lp in words:
            print(f"    P('{word}' | {cls_name}) = exp({lp:.4f}) = {math.exp(lp):.6f}")

    # Показать предсказания
    print("\n--- Примеры предсказаний ---")
    for i in range(min(5, len(X_test))):
        pred = model.predict_one(X_test[i])
        true = y_test[i]
        status = "✓" if pred == true else "✗"
        # Найти ненулевые слова
        nonzero = [vocab[j] for j in range(len(vocab)) if X_test[i][j] > 0]
        doc_text = " ".join(nonzero[:5])
        print(f"  {status}  «{doc_text}...»")
        print(f"       → Предсказано: {class_names[pred]}, Истинно: {class_names[true]}")

    print()


def demo3_laplace_smoothing():
    """Демо 3: Влияние Laplace smoothing."""
    print("=" * 70)
    print("ДЕМО 3: Влияние Laplace smoothing (параметр alpha)")
    print("=" * 70)

    X, y, vocab = generate_text_data()

    # Train/test split
    all_indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(all_indices)
    split = int(0.8 * len(X))
    train_idx = all_indices[:split]
    test_idx = all_indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"\n{'alpha':>8} | {'Train Acc':>10} | {'Test Acc':>10} | Эффект")
    print("-" * 60)

    best_acc = 0
    best_alpha = 1.0

    for alpha in alphas:
        model = MultinomialNaiveBayes(alpha=alpha)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        if test_acc > best_acc:
            best_acc = test_acc
            best_alpha = alpha

        # Определить эффект
        effect = ""
        if train_acc - test_acc > 0.1:
            effect = "переобучение"
        elif test_acc < 0.5:
            effect = "недообучение"
        elif abs(train_acc - test_acc) < 0.05:
            effect = "хорошая генерализация"
        else:
            effect = "умеренный gap"

        print(f"{alpha:>8.3f} | {train_acc:>10.4f} | {test_acc:>10.4f} | {effect}")

    print(f"\nЛучший alpha: {best_alpha} (test accuracy: {best_acc:.4f})")

    # Показать, как alpha влияет на распределение вероятностей
    print("\n--- Демонстрация сглаживания на одном слове ---")
    model_no_smooth = MultinomialNaiveBayes(alpha=0.001)
    model_no_smooth.fit(X_train, y_train)

    model_smooth = MultinomialNaiveBayes(alpha=1.0)
    model_smooth.fit(X_train, y_train)

    test_word_idx = vocab.index("python")
    print(f"\nСлово '{vocab[test_word_idx]}' (индекс {test_word_idx}):")
    for cls in range(3):
        p_ns = math.exp(model_no_smooth.feature_log_probs[cls][test_word_idx])
        p_s = math.exp(model_smooth.feature_log_probs[cls][test_word_idx])
        print(f"  Класс {cls}: без сглаживания = {p_ns:.6f}, с alpha=1.0 = {p_s:.6f}")

    # Показать, что log-пространство предотвращает underflow
    print("\n--- Зачем log-пространство? ---")
    tiny_prob = 1e-300
    print(f"  Вероятность: {tiny_prob}")
    print(f"  В обычном пространстве: 0.0 (underflow!)")
    print(f"  В log-пространстве: {math.log(tiny_prob):.1f} (сохраняется)")
    print(f"  log(1e-300) = {math.log(1e-300):.1f}")
    print(f"  log(1e-10) = {math.log(1e-10):.1f}")
    print(f"  log(0.5) = {math.log(0.5):.4f}")

    print()


def demo4_comparison():
    """Демо 4: Сравнение с логистической регрессией."""
    print("=" * 70)
    print("ДЕМО 4: Сравнение Naive Bayes с Logistic Regression")
    print("=" * 70)

    # --- Бинарная классификация ---
    print("\n--- Бинарная классификация (2 класса) ---")
    random.seed(42)
    X, y = [], []
    for _ in range(50):
        X.append([random.gauss(2.0, 1.0), random.gauss(2.0, 1.0)])
        y.append(0)
    for _ in range(50):
        X.append([random.gauss(5.0, 1.0), random.gauss(5.0, 1.0)])
        y.append(1)

    # Train/test split
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(X))
    X_train = [X[i] for i in indices[:split]]
    y_train = [y[i] for i in indices[:split]]
    X_test = [X[i] for i in indices[split:]]
    y_test = [y[i] for i in indices[split:]]

    # Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    gnb_acc = gnb.score(X_test, y_test)

    # Logistic Regression
    lr = LogisticRegressionSimple(lr=0.5, n_iter=500)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)

    print(f"\n  GaussianNB accuracy:   {gnb_acc:.4f}")
    print(f"  LogisticReg accuracy:  {lr_acc:.4f}")

    # --- Мультиклассовая классификация ---
    print("\n--- Мультиклассовая классификация (3 класса) ---")
    X_multi, y_multi = generate_gaussian_data()

    indices = list(range(len(X_multi)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(X_multi))
    X_train_m = [X_multi[i] for i in indices[:split]]
    y_train_m = [y_multi[i] for i in indices[:split]]
    X_test_m = [X_multi[i] for i in indices[split:]]
    y_test_m = [y_multi[i] for i in indices[split:]]

    gnb_m = GaussianNaiveBayes()
    gnb_m.fit(X_train_m, y_train_m)
    gnb_m_acc = gnb_m.score(X_test_m, y_test_m)

    lr_m = LogisticRegressionSimple(lr=0.5, n_iter=500)
    lr_m.fit(X_train_m, y_train_m)
    lr_m_acc = lr_m.score(X_test_m, y_test_m)

    print(f"\n  GaussianNB accuracy:   {gnb_m_acc:.4f}")
    print(f"  LogisticReg accuracy:  {lr_m_acc:.4f}")

    # --- Сводная таблица ---
    print("\n--- Сводная таблица ---")
    print(f"{'Метод':<25} | {'Бинарная':>10} | {'Мультикласс':>12}")
    print("-" * 52)
    print(f"{'Gaussian Naive Bayes':<25} | {gnb_acc:>10.4f} | {gnb_m_acc:>12.4f}")
    print(f"{'Logistic Regression':<25} | {lr_acc:>10.4f} | {lr_m_acc:>12.4f}")

    # --- Ключевые различия ---
    print("\n--- Ключевые различия ---")
    print("""
  Gaussian Naive Bayes:
    (+) Быстрое обучение (один проход по данным)
    (+) Хорошо работает при малом объёме данных
    (+) Естественно обрабатывает мультикласс
    (-) Допущение независимости признаков (часто не выполняется)
    (-) Предполагает нормальное распределение признаков

  Logistic Regression:
    (+) Не предполагает распределение признаков
    (+) Обучает разделяющую гиперплоскость
    (+) Обычно точнее при большом объёме данных
    (-) Медленнее при большом числе признаков
    (-) Требует итеративной оптимизации
""")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("\n" + "█" * 70)
    print("  НАИВНЫЙ БАЙЕС (NAIVE BAYES) — РЕАЛИЗАЦИЯ С НУЛЯ")
    print("█" * 70)

    demo1_gaussian_nb()
    demo2_multinomial_nb()
    demo3_laplace_smoothing()
    demo4_comparison()

    print("=" * 70)
    print("  ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 70)
