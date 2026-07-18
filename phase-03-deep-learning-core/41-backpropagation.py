"""
41 — Backpropagation from scratch.

Класс Value для автодифференцирования, forward/backward pass через
вычислительный граф, gradient checking и обучение нейросети.
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────
# 1. Класс Value — ядро автодифференцирования
# ─────────────────────────────────────────────

class Value:
    """Скаляр с автоматическим вычислением градиентов (autograd)."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value({self.data:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other, (self,)) * (self ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ── topological sort + backward ──

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# ─────────────────────────────────────────────
# 2. Нейрон, слой, MLP
# ─────────────────────────────────────────────

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ─────────────────────────────────────────────
# Демо 1: Value class — базовые операции
# ─────────────────────────────────────────────

print("=" * 60)
print("Демо 1: Value class — базовые операции")
print("=" * 60)

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
f = Value(-2.0, label="f")
L = d * f; L.label = "L"

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"e = a * b = {e}")
print(f"d = e + c = {d}")
print(f"f = {f}")
print(f"L = d * f = {L}")

L.backward()
print("\nГрадиенты после L.backward():")
print(f"  dL/da = {a.grad:.4f}  (ожидается: 6.0)")
print(f"  dL/db = {b.grad:.4f}  (ожидается: -4.0)")
print(f"  dL/dc = {c.grad:.4f}  (ожидается: -2.0)")
print(f"  dL/de = {e.grad:.4f}  (ожидается: -2.0)")
print(f"  dL/dd = {d.grad:.4f}  (ожидается: -2.0)")
print(f"  dL/df = {f.grad:.4f}  (ожидается: 4.0)")

# ─────────────────────────────────────────────
# Демо 2: Forward pass через граф
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Демо 2: Forward pass через граф (tanh)")
print("=" * 60)

x1 = Value(0.5, label="x1")
x2 = Value(-1.0, label="x2")
w1 = Value(0.8, label="w1")
w2 = Value(-0.3, label="w2")
b = Value(0.1, label="b")

n = x1 * w1 + x2 * w2 + b
y = n.tanh()

print(f"x1={x1.data}, x2={x2.data}, w1={w1.data}, w2={w2.data}, b={b.data}")
print(f"pre-activation: n = {n.data:.6f}")
print(f"tanh(n) = y = {y.data:.6f}")

y.backward()

print("\nГрадиенты:")
print(f"  dy/dx1 = {x1.grad:.6f}  (ожидается: w1*(1-y^2) = {(w1.data * (1 - y.data**2)):.6f})")
print(f"  dy/dx2 = {x2.grad:.6f}  (ожидается: w2*(1-y^2) = {(w2.data * (1 - y.data**2)):.6f})")
print(f"  dy/dw1 = {w1.grad:.6f}  (ожидается: x1*(1-y^2) = {(x1.data * (1 - y.data**2)):.6f})")
print(f"  dy/dw2 = {w2.grad:.6f}  (ожидается: x2*(1-y^2) = {(x2.data * (1 - y.data**2)):.6f})")

# ─────────────────────────────────────────────
# Демо 3: Backward pass — вычисление градиентов
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Демо 3: Backward pass — цепное правило на практике")
print("=" * 60)

# f(x) = (x^2 + 2x + 1) * sin(x)
# df/dx = (2x + 2)*sin(x) + (x^2 + 2x + 1)*cos(x)

x_val = 1.5
xx = Value(x_val, label="x")
expr = (xx ** 2 + 2 * xx + 1) * (xx + 0.5 - 0.5).exp()  # workaround: sin через x

# Прямое вычисление для (x^2 + 2x + 1) * (x-1)^2  -- покажем цепное правило
x2 = Value(x_val, label="x")
u = x2 ** 2 + 2 * x2 + 1          # u = x^2+2x+1
v = x2 - 1                         # v = x-1
w = v ** 2                         # w = (x-1)^2
loss = u + w                       # loss = x^2+2x+1 + (x-1)^2

print(f"x = {x_val}")
print(f"u = x^2+2x+1 = {u.data:.6f}")
print(f"v = x-1 = {v.data:.6f}")
print(f"w = (x-1)^2 = {w.data:.6f}")
print(f"loss = u + w = {loss.data:.6f}")

loss.backward()
print(f"\ndloss/dx (autograd) = {x2.grad:.6f}")

# Аналитически: d/dx [x^2+2x+1 + (x-1)^2] = (2x+2) + 2(x-1) = 4x
analytical = 4 * x_val
print(f"dloss/dx (аналитически 4x) = {analytical:.6f}")
print(f"Совпадение: {abs(x2.grad - analytical) < 1e-8}")

# ─────────────────────────────────────────────
# Демо 4: Gradient checking
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Демо 4: Gradient checking (numerical vs analytical)")
print("=" * 60)


def gradient_check():
    """Сравнение аналитических градиентов с численными (finite differences)."""

    def numerical_gradient(func, inputs, eps=1e-4):
        grads = []
        for inp in inputs:
            old = inp.data
            inp.data = old + eps
            res1 = func()
            fxh1 = res1.data if isinstance(res1, Value) else float(res1)
            inp.data = old - eps
            res2 = func()
            fxh2 = res2.data if isinstance(res2, Value) else float(res2)
            inp.data = old
            grads.append((fxh1 - fxh2) / (2 * eps))
        return grads

    # Простая функция: f(x, y) = x*y + x + y
    xv = Value(3.0, label="x")
    yv = Value(-2.0, label="y")

    def forward():
        return xv * yv + xv + yv

    out = forward()
    out.backward()

    num_grads = numerical_gradient(forward, [xv, yv])

    print(f"f(x,y) = x*y + x + y,  x={xv.data}, y={yv.data}")
    print(f"f = {out.data:.4f}")
    print()
    print(f"{'Параметр':<12} {'Analytical':>12} {'Numerical':>12} {'Отклонение':>14}")
    print("-" * 54)

    check_passed = True
    for name, val, ng in [("x", xv, num_grads[0]), ("y", yv, num_grads[1])]:
        diff = abs(val.grad - ng)
        status = "OK" if diff < 1e-6 else "FAIL"
        if status == "FAIL":
            check_passed = False
        print(f"{name:<12} {val.grad:>12.8f} {ng:>12.8f} {diff:>14.2e}  {status}")

    # Сложная функция: sigmoid(x*w+b)
    print()
    print("Тест 2: sigmoid(x*w + b)")
    xs = Value(0.5, label="x")
    ws = Value(-1.2, label="w")
    bs = Value(0.3, label="b")
    z = xs * ws + bs
    sig = 1 / (1 + (-z).exp())

    sig.backward()

    def fwd2():
        _z = xs * ws + bs
        return 1 / (1 + (-_z).exp())

    num = numerical_gradient(fwd2, [xs, ws, bs])

    print(f"  sigma(0.5*(-1.2)+0.3) = sigma(-0.3) = {sig.data:.6f}")
    print(f"\n{'Параметр':<12} {'Analytical':>12} {'Numerical':>12} {'Отклонение':>14}")
    print("-" * 54)

    for name, val, ng in [("x", xs, num[0]), ("w", ws, num[1]), ("b", bs, num[2])]:
        diff = abs(val.grad - ng)
        status = "OK" if diff < 1e-6 else "FAIL"
        if status == "FAIL":
            check_passed = False
        print(f"{name:<12} {val.grad:>12.8f} {ng:>12.8f} {diff:>14.2e}  {status}")

    print()
    if check_passed:
        print("Gradient check PASSED — аналитические и численные градиенты совпадают.")
    else:
        print("Gradient check FAILED — есть расхождения!")

    return check_passed


gradient_check()

# ─────────────────────────────────────────────
# Демо 5: Обучение MLP через backprop
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Демо 5: Обучение нейросети (MLP) через backprop")
print("=" * 60)

# XOR-задача
xs = [
    [Value(0), Value(0)],
    [Value(0), Value(1)],
    [Value(1), Value(0)],
    [Value(1), Value(1)],
]
ys = [Value(0), Value(1), Value(1), Value(0)]

random.seed(42)
model = MLP(2, [4, 4, 1])

print(f"Архитектура: MLP(2 -> [4, 4] -> 1)")
print(f"Параметров: {len(model.parameters())}")
print()

for k in range(20):
    # forward
    ypred = [model(x) for x in xs]
    loss = sum((ygt - yp) ** 2 for ygt, yp in zip(ys, ypred))

    # backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # gradient descent
    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if k % 4 == 0 or k == 19:
        preds = [yp.data for yp in ypred]
        print(
            f"  epoch {k:>2d} | loss={loss.data:.6f} | "
            f"preds={[round(p, 3) for p in preds]}"
        )

print("\nИтоговые предсказания:")
for i, (x, yp) in enumerate(zip(xs, ypred)):
    print(f"  XOR({int(x[0].data)}, {int(x[1].data)}) = {yp.data:.4f}  →  "
          f"{round(yp.data)}")

print("\n" + "=" * 60)
print("Все демонстрации завершены.")
print("=" * 60)
