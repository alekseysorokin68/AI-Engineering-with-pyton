import math
import random

random.seed(42)


# ============================================================
#  Функция Розенброка (классический бенчмарк)
# ============================================================

def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


# ============================================================
#  Оптимизаторы
# ============================================================

class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


# ============================================================
#  Функция оптимизации
# ============================================================

def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    losses = [func(params)]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
        losses.append(func(params))
    return history, losses


# ============================================================
#  Демо 1: Сравнение оптимизаторов на Розенброке
# ============================================================

print("=" * 55)
print("ДЕМО 1: Сравнение GD, SGD+Momentum, Adam")
print("=" * 55)

start = [-1.0, 1.0]
steps = 5000

gd_hist, gd_loss = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start, steps)
sgd_hist, sgd_loss = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start, steps)
adam_hist, adam_loss = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start, steps)

print(f"\n{'Оптимизатор':<15} {'x':<12} {'y':<12} {'loss':<15}")
print("-" * 54)
for name, hist, loss in [("GD", gd_hist, gd_loss), ("SGD+Momentum", sgd_hist, sgd_loss), ("Adam", adam_hist, adam_loss)]:
    final = hist[-1]
    print(f"{name:<15} {final[0]:<12.6f} {final[1]:<12.6f} {loss[-1]:<15.8f}")

print(f"\nМинимум Розенброка: x=1, y=1, loss=0")


# ============================================================
#  Демо 2: Сравнение learning rates для GD
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Влияние learning rate на GD")
print("=" * 55)

for lr in [0.0001, 0.0003, 0.0005, 0.0007, 0.001]:
    hist, loss = optimize(GradientDescent(lr=lr), rosenbrock, rosenbrock_gradient, start, steps=3000)
    final = hist[-1]
    converged = "✓" if loss[-1] < 0.01 else "✗"
    print(f"  lr={lr:.4f}  loss={loss[-1]:<12.6f}  x={final[0]:.4f}  y={final[1]:.4f}  {converged}")


# ============================================================
#  Демо 3: Влияние momentum
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Влияние momentum на SGD")
print("=" * 55)

for beta in [0.0, 0.5, 0.9, 0.99]:
    hist, loss = optimize(SGDMomentum(lr=0.0001, momentum=beta), rosenbrock, rosenbrock_gradient, start, steps=5000)
    final = hist[-1]
    print(f"  momentum={beta:.2f}  loss={loss[-1]:<12.6f}  x={final[0]:.4f}  y={final[1]:.4f}")


# ============================================================
#  Демо 4: Седловая точка f(x,y) = x² - y²
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Седловая точка — escape из (0.01, 0.01)")
print("=" * 55)

def saddle(params):
    x, y = params
    return x ** 2 - y ** 2

def saddle_gradient(params):
    x, y = params
    return [2 * x, -2 * y]

saddle_start = [0.01, 0.01]
saddle_steps = 200

gd_h, gd_l = optimize(GradientDescent(lr=0.1), saddle, saddle_gradient, saddle_start, saddle_steps)
sgd_h, sgd_l = optimize(SGDMomentum(lr=0.1, momentum=0.9), saddle, saddle_gradient, saddle_start, saddle_steps)
adam_h, adam_l = optimize(Adam(lr=0.1), saddle, saddle_gradient, saddle_start, saddle_steps)

print(f"  Старт: (0.01, 0.01), f={saddle(saddle_start):.6f}")
header = f"  {'Оптимизатор':<15} {'Итого x':<10} {'Итого y':<10} {'Итого loss':<12} Сбежал?"
print(f"\n{header}")
print("  " + "-" * 60)
for name, hist, loss in [("GD", gd_h, gd_l), ("SGD+Momentum", sgd_h, sgd_l), ("Adam", adam_h, adam_l)]:
    final = hist[-1]
    escaped = "✓" if abs(final[1]) > 0.01 else "✗"
    print(f"  {name:<15} {final[0]:<10.4f} {final[1]:<10.4f} {loss[-1]:<12.6f} {escaped}")


# ============================================================
#  Демо 5: Learning rate schedules
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Learning rate schedules")
print("=" * 55)

lr_0 = 0.01
total_steps = 200

# Step decay
print("\nStep Decay (factor=0.5 каждые 50 шагов):")
for step in range(0, total_steps, 50):
    lr = lr_0 * (0.5 ** (step // 50))
    print(f"  шаг {step:3d}: lr={lr:.6f}")

# Cosine annealing
print("\nCosine Annealing:")
for step in [0, 50, 100, 150, 199]:
    T = total_steps
    lr_min = 0.001
    lr = lr_min + 0.5 * (lr_0 - lr_min) * (1 + math.cos(math.pi * step / T))
    print(f"  шаг {step:3d}: lr={lr:.6f}")

# Warmup + decay
print("\nWarmup (20 шагов) + Linear Decay:")
for step in [0, 10, 20, 50, 100, 150, 199]:
    if step < 20:
        lr = lr_0 * step / 20
    else:
        lr = lr_0 * (1 - (step - 20) / (total_steps - 20))
    print(f"  шаг {step:3d}: lr={lr:.6f}")


# ============================================================
#  Демо 6: Adam vs SGD на простой функции
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Adam vs SGD — пошаговое сравнение")
print("=" * 55)

def simple_loss(params):
    x, y = params
    return (x - 3) ** 2 + (y + 2) ** 2

def simple_grad(params):
    x, y = params
    return [2 * (x - 3), 2 * (y + 2)]

start = [0.0, 0.0]

gd_opt = GradientDescent(lr=0.1)
adam_opt = Adam(lr=0.1)

gd_params = list(start)
adam_params = list(start)

print(f"\n  Старт: (0, 0), loss={simple_loss(start):.2f}")
print(f"  Цель: x=3, y=-2, loss=0\n")
print(f"  {'Шаг':<6} {'GD x':<10} {'GD y':<10} {'GD loss':<10} {'Adam x':<10} {'Adam y':<10} {'Adam loss':<10}")
print("  " + "-" * 66)

for step in range(21):
    if step in [0, 1, 2, 5, 10, 20]:
        print(f"  {step:<6} {gd_params[0]:<10.4f} {gd_params[1]:<10.4f} {simple_loss(gd_params):<10.4f} {adam_params[0]:<10.4f} {adam_params[1]:<10.4f} {simple_loss(adam_params):<10.4f}")
    gd_grads = simple_grad(gd_params)
    gd_params = gd_opt.step(gd_params, gd_grads)
    adam_grads = simple_grad(adam_params)
    adam_params = adam_opt.step(adam_params, adam_grads)


# ============================================================
#  Демо 7: Нейросеть с разными оптимизаторами
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Нейросеть — GD vs Adam (линейная регрессия)")
print("=" * 55)

# Простая линейная регрессия y = 2x + 1
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

def train_linear(optimizer_class, lr, epochs=200):
    w = random.gauss(0, 1)
    b = random.gauss(0, 1)
    opt = optimizer_class(lr=lr)
    for epoch in range(epochs):
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
        params = opt.step([w, b], [dw, db])
        w, b = params
    return w, b, total_loss

print(f"\n  Цель: y = 2x + 1\n")
print(f"  {'Оптимизатор':<20} {'w':<10} {'b':<10} {'loss':<12}")
print("  " + "-" * 52)

for name, cls, lr in [("GD", GradientDescent, 0.01), ("SGD+Momentum", SGDMomentum, 0.01), ("Adam", Adam, 0.01)]:
    w, b, loss = train_linear(cls, lr)
    print(f"  {name:<20} {w:<10.4f} {b:<10.4f} {loss:<12.6f}")

print(f"\n  Правильно: w=2.0, b=1.0")
