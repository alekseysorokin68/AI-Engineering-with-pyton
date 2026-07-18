import math
import random

random.seed(42)


# ============================================================
#  Вспомогательные
# ============================================================

def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


# ============================================================
#  Hinge Loss
# ============================================================

def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n


# ============================================================
#  Linear SVM — Gradient Descent
# ============================================================

class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]

    def accuracy(self, X, y):
        preds = self.predict(X)
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return correct / len(y)


# ============================================================
#  Kernel SVM (RBF через kernel matrix)
# ============================================================

class KernelSVM:
    def __init__(self, C=1.0, gamma=0.5, n_epochs=100):
        self.C = C
        self.gamma = gamma
        self.n_epochs = n_epochs

    def rbf_kernel(self, x, z):
        diff = [xi - zi for xi, zi in zip(x, z)]
        return math.exp(-self.gamma * dot(diff, diff))

    def fit(self, X, y):
        n = len(X)
        # Kernel matrix
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                K[i][j] = self.rbf_kernel(X[i], X[j])

        # SGD on dual
        self.alphas = [0.0] * n
        self.X_train = X
        self.y_train = y

        for epoch in range(self.n_epochs):
            for i in range(n):
                decision = sum(self.alphas[j] * self.y_train[j] * K[i][j]
                              for j in range(n))
                if self.y_train[i] * decision < 1:
                    self.alphas[i] += self.lr_dual
                    self.alphas[i] = min(self.alphas[i], self.C)

    def predict(self, X):
        result = []
        for x in X:
            decision = sum(self.alphas[i] * self.y_train[i] *
                          self.rbf_kernel(self.X_train[i], x)
                          for i in range(len(self.X_train)))
            result.append(1 if decision >= 0 else -1)
        return result

    @property
    def lr_dual(self):
        return 1.0 / (self.C + 1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return correct / len(y)


# ============================================================
#  Кернелы
# ============================================================

def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))


# ============================================================
#  Support Vectors
# ============================================================

def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors


# ============================================================
#  Генерация данных
# ============================================================

def generate_linear_data(n_per_class=50):
    X, y = [], []
    for _ in range(n_per_class):
        X.append([random.gauss(1, 0.8), random.gauss(1, 0.8)])
        y.append(-1)
    for _ in range(n_per_class):
        X.append([random.gauss(4, 0.8), random.gauss(4, 0.8)])
        y.append(1)
    combined = list(zip(X, y))
    random.shuffle(combined)
    return zip(*combined)

def generate_noisy_data(n_per_class=50, noise=1.5):
    X, y = [], []
    for _ in range(n_per_class):
        X.append([random.gauss(1, noise), random.gauss(1, noise)])
        y.append(-1)
    for _ in range(n_per_class):
        X.append([random.gauss(3, noise), random.gauss(3, noise)])
        y.append(1)
    combined = list(zip(X, y))
    random.shuffle(combined)
    return zip(*combined)

def generate_circular_data(n_per_class=50):
    X, y = [], []
    for _ in range(n_per_class):
        angle = random.uniform(0, 2 * math.pi)
        r = random.gauss(1.0, 0.2)
        X.append([r * math.cos(angle), r * math.sin(angle)])
        y.append(-1)
    for _ in range(n_per_class):
        angle = random.uniform(0, 2 * math.pi)
        r = random.gauss(3.0, 0.2)
        X.append([r * math.cos(angle), r * math.sin(angle)])
        y.append(1)
    combined = list(zip(X, y))
    random.shuffle(combined)
    return zip(*combined)


# ============================================================
#  Демо 1: Hinge Loss
# ============================================================

print("=" * 55)
print("ДЕМО 1: Hinge Loss")
print("=" * 55)

print(f"\ny×f(x) | Hinge Loss")
print("-" * 30)
for margin in [-2, -1, 0, 0.5, 1, 2, 3]:
    loss = max(0, 1 - margin)
    print(f"  {margin:5.1f} | {loss:.1f}")

print(f"\n→ y×f(x) ≥ 1: loss = 0 (правильно, за пределами отступа)")
print(f"→ y×f(x) < 1: loss линейно растёт")


# ============================================================
#  Демо 2: Linear SVM
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Linear SVM")
print("=" * 55)

X, y = generate_linear_data(50)
X, y = list(X), list(y)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = list(y[:split]), list(y[split:])

print(f"\nДанные: 100 сэмплов, 2 класса")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

svm = LinearSVM(lr=0.001, lambda_param=0.01, n_epochs=500)
svm.fit(X_train, y_train)

train_acc = svm.accuracy(X_train, y_train)
test_acc = svm.accuracy(X_test, y_test)
loss = hinge_loss(X_train, y_train, svm.w, svm.b)

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")
print(f"Hinge loss:     {loss:.4f}")
print(f"Weights: [{svm.w[0]:.4f}, {svm.w[1]:.4f}]")
print(f"Bias: {svm.b:.4f}")

# Support vectors
sv_indices = find_support_vectors(X_train, y_train, svm.w, svm.b)
print(f"\nSupport vectors: {len(sv_indices)} из {len(X_train)}")


# ============================================================
#  Демо 3: Влияние параметра C
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Влияние C (soft margin)")
print("=" * 55)

X_noisy, y_noisy = generate_noisy_data(50, noise=1.5)
X_noisy, y_noisy = list(X_noisy), list(y_noisy)
split_n = int(0.8 * len(X_noisy))
Xn_train, Xn_test = X_noisy[:split_n], X_noisy[split_n:]
yn_train, yn_test = list(y_noisy[:split_n]), list(y_noisy[split_n:])

print(f"\n{'C':>8} {'Train':>8} {'Test':>8} {'Loss':>10}")
print("-" * 38)

for lambda_param in [0.001, 0.01, 0.1, 1.0, 10.0]:
    svm_n = LinearSVM(lr=0.001, lambda_param=lambda_param, n_epochs=500)
    svm_n.fit(Xn_train, yn_train)
    train_acc = svm_n.accuracy(Xn_train, yn_train)
    test_acc = svm_n.accuracy(Xn_test, yn_test)
    loss_n = hinge_loss(Xn_train, yn_train, svm_n.w, svm_n.b)
    print(f"{lambda_param:>8.3f} {train_acc:>8.4f} {test_acc:>8.4f} {loss_n:>10.4f}")

print(f"\n→ Большой C (малый λ): узкий отступ, fewer ошибок, переобучение")
print(f"→ Малый C (большой λ): широкий отступ, больше ошибок, недообучение")


# ============================================================
#  Демо 4: Kernel Trick — RBF
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Kernel Trick — RBF")
print("=" * 55)

X_circ, y_circ = generate_circular_data(30)
X_circ, y_circ = list(X_circ), list(y_circ)

print(f"\nДанные: 2 концентрических кольца")
print(f"Линейный SVM не может разделить!")

# Kernel matrix
print(f"\nKernel matrix (RBF, γ=0.5):")
print(f"{'':>8}", end="")
for j in range(min(6, len(X_circ))):
    print(f"  [{j}]", end="")
print()
for i in range(min(6, len(X_circ))):
    print(f"  [{i}]  ", end="")
    for j in range(min(6, len(X_circ))):
        k = rbf_kernel(X_circ[i], X_circ[j], gamma=0.5)
        print(f" {k:.2f}", end="")
    print()

print(f"\n→ Точки одного класса: K ≈ 1.0 (близко)")
print(f"→ Точки разных классов: K ≈ 0.0 (далеко)")
print(f"→ В ядерном пространстве данные разделимы!")


# ============================================================
#  Демо 5: Сравнение ядер
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Сравнение ядер")
print("=" * 55)

x1 = [1.0, 2.0]
x2 = [3.0, 4.0]

print(f"\nx1 = {x1}, x2 = {x2}")
print(f"Linear kernel:    {linear_kernel(x1, x2):.4f}")
print(f"Polynomial (d=3): {polynomial_kernel(x1, x2, degree=3):.4f}")
print(f"RBF (γ=0.5):      {rbf_kernel(x1, x2, gamma=0.5):.4f}")


# ============================================================
#  Демо 6: Сравнение SVM vs Logistic Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: SVM vs Logistic Regression")
print("=" * 55)

print(f"""
┌──────────────────────┬──────────────────────┬─────────────────────┐
│                      │ SVM                  │ Log. Regression     │
├──────────────────────┼──────────────────────┼─────────────────────┤
│ Loss                 │ Hinge (max(0,1-yf)) │ Log (log(1+e^(-yf)))│
│ Решения              │ Разреженные (SV)     │ Все точки            │
│ Ядра                 │ Да (RBF, poly)       │ Нет                 │
│ Интерпретируемость   │ Средняя              │ Высокая             │
│ Маленькие данные     │ Отлично              │ Хорошо              │
│ Большие данные       │ Медленно (O(n²))     │ Быстро (O(n))       │
└──────────────────────┴──────────────────────┴─────────────────────┘
""")
