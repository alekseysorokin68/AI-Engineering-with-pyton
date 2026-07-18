import math
import random

random.seed(42)


# === Демо 1: Числовая производная ===

print("=" * 55)
print("ДЕМО 1: Числовая vs аналитическая производная")
print("=" * 55)

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  числовая={numerical:.6f}  аналитическая={analytical:.1f}")


# === Демо 2: Градиент для функции двух переменных ===

print("\n" + "=" * 55)
print("ДЕМО 2: Градиент f(x,y) = x² + 3xy + y²")
print("=" * 55)

def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Числовой градиент в (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Аналитический:             [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")


# === Демо 3: Градиентный спуск для f(x) = x² ===

print("\n" + "=" * 55)
print("ДЕМО 3: Градиентный спуск f(x) = x², старт x=5")
print("=" * 55)

x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    if step < 5 or step == 19:
        print(f"шаг {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
    elif step == 5:
        print("  ...")


# === Демо 4: Градиентный спуск 2D ===

print("\n" + "=" * 55)
print("ДЕМО 4: Градиентный спуск f(x,y) = x² + y², старт (4,3)")
print("=" * 55)

def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"шаг {step:2d}  ({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")


# === Демо 5: Сравнение числовых и аналитических производных ===

print("\n" + "=" * 55)
print("ДЕМО 5: Сравнение производных (x=2)")
print("=" * 55)

test_functions = [
    ("x²",       lambda x: x**2,          lambda x: 2*x),
    ("x³",       lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Функция':<12} {'Числовая':>12} {'Аналитич.':>12} {'Ошибка':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")


# === Демо 6: Гессиан — седловая точка vs чаша ===

print("\n" + "=" * 55)
print("ДЕМО 6: Гессиан (матрица вторых производных)")
print("=" * 55)

def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)

print(f"Седло Hessian:   {H_saddle}  ← смешанные знаки = седловая точка")
print(f"Чаша Hessian:    {H_bowl}    ← оба положительные = минимум")


# === Демо 7: Ряд Тейлора для sin(x) ===

print("\n" + "=" * 55)
print("ДЕМО 7: Аппроксимация Тейлора для sin(x) от 0")
print("=" * 55)

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
print(f"{'h':>6} {'sin(h)':>10} {'1-й порядок':>12} {'2-й порядок':>12}")
print("-" * 44)
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"{h:6.1f} {true_val:10.4f} {t1:12.4f} {t2:12.4f}")


# === Демо 8: Нейросеть — линейная регрессия с нуля ===

print("\n" + "=" * 55)
print("ДЕМО 8: Линейная регрессия y = 2x + 1 (градиентный спуск)")
print("=" * 55)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"эпоха {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nНашли:   y = {w:.2f}x + {b:.2f}")
print(f"Правильно: y = 2x + 1")
