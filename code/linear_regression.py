import math
import random

random.seed(42)


# ============================================================
#  Linear Regression — Gradient Descent
# ============================================================

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y))
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
#  Linear Regression — Normal Equation
# ============================================================

class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
#  Multiple Linear Regression
# ============================================================

class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
#  Ridge Regression (L2)
# ============================================================

class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f}")
        return self


# ============================================================
#  Polynomial Regression
# ============================================================

class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]

    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.6f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
#  Вспомогательные
# ============================================================

def standardize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(len(X)):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds


# ============================================================
#  Демо 1: Linear Regression (GD)
# ============================================================

print("=" * 55)
print("ДЕМО 1: Linear Regression — Gradient Descent")
print("=" * 55)

TRUE_W = 3.0
TRUE_B = 7.0
X = [random.uniform(0, 10) for _ in range(100)]
y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X]

print(f"\nИстинная модель: y = {TRUE_W}x + {TRUE_B} + noise")
model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)

print(f"\nНашли:     y = {model.w:.4f}x + {model.b:.4f}")
print(f"Истинная:  y = {TRUE_W}x + {TRUE_B}")
print(f"R²: {model.r_squared(X, y):.4f}")


# ============================================================
#  Демо 2: Normal Equation
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Normal Equation (закрытая форма)")
print("=" * 55)

model_normal = LinearRegressionNormal()
model_normal.fit(X, y)

print(f"\nНашли:     y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
print(f"Истинная:  y = {TRUE_W}x + {TRUE_B}")
print(f"R²: {model_normal.r_squared(X, y):.4f}")


# ============================================================
#  Демо 3: Multiple Linear Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Multiple Linear Regression (3 признака)")
print("=" * 55)

random.seed(42)
N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)

X_scaled, x_means, x_stds = standardize(X_multi)
y_mean_val = sum(y_multi) / len(y_multi)
y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]

print(f"\nПризнаки: размер, спальни, возраст")
print(f"Цель: цена дома")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nВеса (стандартизированные): {[round(w, 4) for w in multi_model.weights]}")
print(f"R²: {multi_model.r_squared(X_scaled, y_scaled):.4f}")
print(f"\nИнтерпретация:")
print(f"  Размер:     {multi_model.weights[0]:.4f} (чем больше дом → дороже)")
print(f"  Спальни:    {multi_model.weights[1]:.4f} (больше спален → дороже)")
print(f"  Возраст:    {multi_model.weights[2]:.4f} (старше → дешевле)")


# ============================================================
#  Демо 4: Polynomial Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Polynomial Regression")
print("=" * 55)

X_poly = [x / 10.0 for x in range(0, 50)]
y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

x_max = max(abs(x) for x in X_poly)
X_poly_norm = [x / x_max for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

print(f"\nИстинная модель: y = 0.5x² - 2x + 3")

poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"\nСтепень 2: R² = {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"Степень 5: R² = {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

print(f"\n→ Степень 2 хорошо подходит, степень 5 может переобучиться")


# ============================================================
#  Демо 5: Ridge Regression
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Ridge Regression (L2 регуляризация)")
print("=" * 55)

ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nRidge веса:  {[round(w, 4) for w in ridge.weights]}")
print(f"Обычные:    {[round(w, 4) for w in multi_model.weights]}")
print(f"\n→ Ridge веса МЕНЬШЕ (сжаты к нулю) из-за L2 штрафа")


# ============================================================
#  Демо 6: Сравнение методов
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Сравнение методов")
print("=" * 55)

print(f"""
┌──────────────────────┬──────────────────────┬─────────────────────────┐
│ Метод                │ Стоимость            │ Когда использовать     │
├──────────────────────┼──────────────────────┼─────────────────────────┤
│ Gradient Descent     │ O(n × epochs)        │ Большие данные, онлайн  │
│ Normal Equation      │ O(n³) инверсия       │ Маленькие данные       │
│ Ridge (L2)           │ O(n × epochs)        │ Много признаков        │
│ Polynomial           │ O(n × epochs × d)    │ Нелинейная зависимость │
└──────────────────────┴──────────────────────┴─────────────────────────┘
""")
