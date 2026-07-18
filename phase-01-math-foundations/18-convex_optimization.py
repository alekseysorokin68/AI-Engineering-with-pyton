import math
import random

random.seed(42)


# ============================================================
#  Проверка выпуклости
# ============================================================

def check_convexity(f, dim, bounds=(-5, 5), samples=1000):
    violations = 0
    for _ in range(samples):
        x = [random.uniform(*bounds) for _ in range(dim)]
        y = [random.uniform(*bounds) for _ in range(dim)]
        t = random.uniform(0, 1)
        mid = [t * xi + (1 - t) * yi for xi, yi in zip(x, y)]
        lhs = f(mid)
        rhs = t * f(x) + (1 - t) * f(y)
        if lhs > rhs + 1e-10:
            violations += 1
    return violations == 0, violations


# ============================================================
#  Gradient Descent
# ============================================================

def gradient_descent(grad_f, x0, lr=0.01, steps=1000, tol=1e-10):
    x = list(x0)
    history = [x[:]]
    for _ in range(steps):
        try:
            g = grad_f(x)
            x = [xi - lr * gi for xi, gi in zip(x, g)]
            history.append(x[:])
            grad_norm = sum(gi**2 for gi in g)
            if grad_norm < tol or math.isinf(grad_norm) or math.isnan(grad_norm):
                break
        except (OverflowError, ValueError):
            break
    return history


# ============================================================
#  Newton's Method
# ============================================================

def newtons_method(grad_f, hessian_f, x0, steps=50, tol=1e-12):
    x = list(x0)
    history = [x[:]]
    for _ in range(steps):
        g = grad_f(x)
        H = hessian_f(x)
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if abs(det) < 1e-15:
            break
        H_inv = [
            [H[1][1] / det, -H[0][1] / det],
            [-H[1][0] / det, H[0][0] / det],
        ]
        dx = [
            H_inv[0][0] * g[0] + H_inv[0][1] * g[1],
            H_inv[1][0] * g[0] + H_inv[1][1] * g[1],
        ]
        x = [x[0] - dx[0], x[1] - dx[1]]
        history.append(x[:])
        if sum(gi**2 for gi in g) < tol:
            break
    return history


# ============================================================
#  Lagrange Multipliers
# ============================================================

def lagrange_solve(f_grad, g_val, g_grad, x0, lr=0.01,
                   lr_lambda=0.01, steps=5000):
    x = list(x0)
    lam = 0.0
    history = []
    for _ in range(steps):
        fg = f_grad(x)
        gv = g_val(x)
        gg = g_grad(x)
        x = [
            xi - lr * (fgi + lam * ggi)
            for xi, fgi, ggi in zip(x, fg, gg)
        ]
        lam = lam + lr_lambda * gv
        history.append((x[:], lam, gv))
    return history


# ============================================================
#  Вспомогательные
# ============================================================

def norm(x):
    return math.sqrt(sum(xi**2 for xi in x))

def mat_eigenvalues_2x2(H):
    trace = H[0][0] + H[1][1]
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    disc = trace**2 - 4 * det
    if disc < 0:
        return [trace/2, trace/2]
    return [(trace + math.sqrt(disc)) / 2, (trace - math.sqrt(disc)) / 2]


# ============================================================
#  Демо 1: Проверка выпуклости
# ============================================================

print("=" * 55)
print("ДЕМО 1: Проверка выпуклости")
print("=" * 55)

functions = [
    ("x²", lambda x: x[0]**2, 1),
    ("x³", lambda x: x[0]**3, 1),
    ("|x|", lambda x: abs(x[0]), 1),
    ("eˣ", lambda x: math.exp(x[0]), 1),
    ("sin(x)", lambda x: math.sin(x[0]), 1),
    ("x² + y²", lambda x: x[0]**2 + x[1]**2, 2),
    ("x·y", lambda x: x[0] * x[1], 2),
    ("max(x,0)", lambda x: max(0, x[0]), 1),
]

for name, f, dim in functions:
    is_convex, violations = check_convexity(f, dim, samples=5000)
    status = "✓ выпуклая" if is_convex else f"✗ не выпуклая ({violations} нарушений)"
    print(f"  {name:<12} {status}")


# ============================================================
#  Демо 2: Newton vs Gradient Descent
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Newton vs Gradient Descent")
print("=" * 55)

# f(x,y) = 5x² + y² (вытянутая долина)
def quadratic(x):
    return 5 * x[0]**2 + x[1]**2

def quadratic_grad(x):
    return [10 * x[0], 2 * x[1]]

def quadratic_hessian(x):
    return [[10, 0], [0, 2]]

start = [10.0, 10.0]
print(f"\nf(x,y) = 5x² + y², старт: {start}")

gd_hist = gradient_descent(quadratic_grad, start, lr=0.05, steps=500)
newton_hist = newtons_method(quadratic_grad, quadratic_hessian, start)

print(f"\nGradient Descent:")
print(f"  Шагов: {len(gd_hist)-1}")
print(f"  Финальная точка: ({gd_hist[-1][0]:.6f}, {gd_hist[-1][1]:.6f})")
print(f"  Loss: {quadratic(gd_hist[-1]):.2e}")

print(f"\nNewton's Method:")
print(f"  Шагов: {len(newton_hist)-1}")
print(f"  Финальная точка: ({newton_hist[-1][0]:.6f}, {newton_hist[-1][1]:.6f})")
print(f"  Loss: {quadratic(newton_hist[-1]):.2e}")

print(f"\n→ Newton: 1 шаг (точен для квадратичных)")
print(f"→ GD: ~100+ шагов (долина вытянута, κ=5)")


# ============================================================
#  Демо 3: Lagrange Multipliers
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Lagrange Multipliers")
print("=" * 55)

# min f(x,y) = x² + y² s.t. x + y = 1
def f_grad(x):
    return [2 * x[0], 2 * x[1]]

def g_val(x):
    return x[0] + x[1] - 1

def g_grad(x):
    return [1.0, 1.0]

hist = lagrange_solve(f_grad, g_val, g_grad, [5.0, 5.0], steps=10000)
final_x, final_lam, final_g = hist[-1]

print(f"\nmin f(x,y) = x² + y²  s.t. x + y = 1")
print(f"Решение: x={final_x[0]:.4f}, y={final_x[1]:.4f}")
print(f"λ = {final_lam:.4f} (ожидается -1)")
print(f"g(x) = {final_g:.6f} (ожидается 0)")
print(f"Проверка: x+y = {final_x[0]+final_x[1]:.4f} (ожидается 1)")


# ============================================================
#  Демо 4: L1 vs L2 регуляризация (геометрия)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: L1 vs L2 — геометрия ограничений")
print("=" * 55)

print(f"""
┌─────────────────────────────────────────────┐
│ L2 ограничение: ||w||² ≤ t  → КРУГ         │
│                                             │
│     ╭───╮         Контуры loss — эллипсы    │
│    ╱     ╲        Касание на гладкой        │
│   │   •   │       границе → все веса ≠ 0   │
│    ╲     ╱                                  │
│     ╰───╯         → Ridge: веса мелкие,     │
│                    но не нулевые             │
├─────────────────────────────────────────────┤
│ L1 ограничение: ||w||₁ ≤ t  → РОМБ         │
│                                             │
│       ╱╲          Контуры loss — эллипсы    │
│      ╱  ╲         Касание на УГЛУ          │
│     ╱  • ╲        (на оси) → один вес = 0  │
│      ╲  ╱                                   │
│       ╲╱          → LASSO: спарсивность,    │
│                    автовыбор признаков       │
└─────────────────────────────────────────────┘
""")


# ============================================================
#  Демо 5: Condition Number и Newton
# ============================================================

print("=" * 55)
print("ДЕМО 5: Condition Number и скорость сходимости")
print("=" * 55)

# Разные condition number
configs = [
    (1, "f = x² + y²"),
    (5, "f = 5x² + y²"),
    (50, "f = 50x² + y²"),
    (500, "f = 500x² + y²"),
]

for kappa, name in configs:
    def make_grad(k):
        return lambda x: [2*k*x[0], 2*x[1]]

    gd = gradient_descent(make_grad(kappa), [10.0, 10.0], lr=0.05, steps=1000)
    loss_gd = gd[-1][0]**2 * kappa + gd[-1][1]**2

    newton = newtons_method(make_grad(kappa),
                           lambda x: [[2*kappa, 0], [0, 2]],
                           [10.0, 10.0])
    loss_n = newton[-1][0]**2 * kappa + newton[-1][1]**2

    print(f"\n  {name} (κ={kappa})")
    print(f"    GD: {len(gd)-1} шагов, loss={loss_gd:.2e}")
    print(f"    Newton: {len(newton)-1} шагов, loss={loss_n:.2e}")


# ============================================================
#  Демо 6: Выпуклые vs невыпуклые задачи в ML
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Выпуклые vs невыпуклые задачи в ML")
print("=" * 55)

print(f"""
┌──────────────────────┬──────────┬──────────────────────┐
│ Задача               │ Выпуклая?│ Почему               │
├──────────────────────┼──────────┼──────────────────────┤
│ Linear regression    │ Да       │ Loss = квадратичная  │
│ Logistic regression  │ Да       │ Log-loss выпуклая    │
│ SVM                  │ Да       │ Hinge loss выпуклая  │
│ LASSO / Ridge        │ Да       │ Сумма выпуклых       │
│ Нейросеть (любая)    │ Нет      │ Нелинейные активации │
│ k-means              │ Нет      │ Дискретные присвоения│
│ Matrix factorization │ Нет      │ Произведение неизвестных│
└──────────────────────┴──────────┴──────────────────────┘

→ Линейные модели + выпуклые loss = гарантия глобального оптимума
→ Нейросети = no guarantees, но SGD работает "достаточно хорошо"
""")


# ============================================================
#  Демо 7: Hessian eigenvalues
# ============================================================

print("=" * 55)
print("ДЕМО 7: Hessian eigenvalues — кривизна")
print("=" * 55)

# Rosenbrock
def rosenbrock_grad(x):
    x1, x2 = x
    dfdx1 = -2*(1-x1) + 200*(x2-x1**2)*(-2*x1)
    dfdx2 = 200*(x2-x1**2)
    return [dfdx1, dfdx2]

def rosenbrock_hessian(x):
    x1, x2 = x
    h11 = 2 + 200*(2*(x2-x1**2) + 4*x1**2)
    h12 = 200*(-2*x1)
    h22 = 200
    return [[h11, h12], [h12, h22]]

# В минимуме (1,1)
H_min = rosenbrock_hessian([1.0, 1.0])
ev_min = mat_eigenvalues_2x2(H_min)

# Далеко от минимума (-1, 1)
H_far = rosenbrock_hessian([-1.0, 1.0])
ev_far = mat_eigenvalues_2x2(H_far)

print(f"\nRosenbrock: f(x,y) = (1-x)² + 100(y-x²)²")
print(f"\nВ минимуме (1,1):")
print(f"  Hessian: [[{H_min[0][0]:.0f}, {H_min[0][1]:.0f}], [{H_min[1][0]:.0f}, {H_min[1][1]:.0f}]]")
print(f"  Eigenvalues: {[f'{ev:.2f}' for ev in ev_min]}")
print(f"  → Оба положительные: локальный минимум")

print(f"\nДалеко (-1, 1):")
print(f"  Hessian: [[{H_far[0][0]:.0f}, {H_far[0][1]:.0f}], [{H_far[1][0]:.0f}, {H_far[1][1]:.0f}]]")
print(f"  Eigenvalues: {[f'{ev:.2f}' for ev in ev_far]}")
print(f"  → Смешанные знаки: седловая точка")
