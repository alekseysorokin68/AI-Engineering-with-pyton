import math
import random

random.seed(42)


# ============================================================
#  Sigmoid
# ============================================================

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


# ============================================================
#  Logistic Regression — Binary
# ============================================================

class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


# ============================================================
#  Classification Metrics
# ============================================================

class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_report(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


# ============================================================
#  Softmax Regression — Multi-class
# ============================================================

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


# ============================================================
#  Генерация данных
# ============================================================

def generate_binary_data(n=200):
    X = []
    y = []
    for _ in range(n // 2):
        X.append([random.gauss(2, 1), random.gauss(2, 1)])
        y.append(0)
    for _ in range(n // 2):
        X.append([random.gauss(5, 1), random.gauss(5, 1)])
        y.append(1)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)

def generate_multiclass_data(n_per_class=50):
    centers = [(1, 1), (5, 1), (3, 5)]
    X, y = [], []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_class):
            X.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
            y.append(label)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


# ============================================================
#  Демо 1: Sigmoid
# ============================================================

print("=" * 55)
print("ДЕМО 1: Sigmoid Function")
print("=" * 55)

print(f"\nz     | sigmoid(z)")
print("-" * 25)
for z in [-5, -2, -1, 0, 1, 2, 5]:
    print(f"{z:5.1f} | {sigmoid(z):.4f}")

print(f"\nПроизводная sigmoid'(z) = sigmoid(z) × (1 - sigmoid(z))")
for z in [-2, 0, 2]:
    s = sigmoid(z)
    print(f"  z={z}: sigmoid'={s*(1-s):.4f}")


# ============================================================
#  Демо 2: Binary Logistic Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Binary Logistic Regression")
print("=" * 55)

X, y = generate_binary_data(200)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nДанные: 200 сэмплов, 2 класса, 2 признака")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")


# ============================================================
#  Демо 3: Classification Metrics
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Classification Metrics")
print("=" * 55)

y_pred_test = [model.predict(x) for x in X_test]
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()


# ============================================================
#  Демо 4: Decision Boundary
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Decision Boundary")
print("=" * 55)

w1, w2 = model.weights
b = model.bias
print(f"\nГраница: {w1:.4f}×x1 + {w2:.4f}×x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Выражаем x2: x2 = {-w1/w2:.4f}×x1 + {-b/w2:.4f}")

print(f"\nПредсказания вблизи границы:")
for point in [[3.0, 3.0], [3.5, 3.5], [4.0, 4.0], [2.5, 2.5], [5.0, 5.0]]:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] → prob={prob:.4f}, класс={pred}")


# ============================================================
#  Демо 5: Multi-class Softmax
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Multi-class Softmax Regression")
print("=" * 55)

X_3, y_3 = generate_multiclass_data(50)
split_3 = int(0.8 * len(X_3))
X_train_3, X_test_3 = X_3[:split_3], X_3[split_3:]
y_train_3, y_test_3 = y_3[:split_3], y_3[split_3:]

print(f"\n3 класса, центры: (1,1), (5,1), (3,5)")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print(f"\nПримеры предсказаний:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  Истинный: {y_test_3[i]}, Предсказанный: {pred}, Вероятности: [{', '.join(f'{p:.3f}' for p in probs)}]")


# ============================================================
#  Демо 6: Threshold Tuning
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Threshold Tuning")
print("=" * 55)

print(f"\n{'Порог':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")

print(f"\n→ Порог < 0.5: больше Recall (ловим больше positives)")
print(f"→ Порог > 0.5: больше Precision (меньше ложных срабатываний)")
