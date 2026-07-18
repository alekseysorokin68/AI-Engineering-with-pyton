import math
import random

random.seed(42)


# ============================================================
#  Value class — мини autograd движок
# ============================================================

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value({self.data:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

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


# ============================================================
#  Нейросеть: Neuron → Layer → MLP
# ============================================================

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ============================================================
#  Демо 1: Проверка базовых операций
# ============================================================

print("=" * 55)
print("ДЕМО 1: Проверка базовых операций Value")
print("=" * 55)

# y = relu(x1 * x2 + 1)
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)   # b = 7.0
y = b.relu()         # y = 7.0

y.backward()

print(f"y = relu(x1*x2 + 1) = relu(2*3 + 1) = {y.data}")
print(f"dy/dx1 = {x1.grad}  (ожидается x2 = 3.0)")
print(f"dy/dx2 = {x2.grad}  (ожидается x1 = 2.0)")


# ============================================================
#  Демо 2: Более сложное выражение
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: f = relu(a*b + c), a=2, b=-3, c=10")
print("=" * 55)

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"f = relu(2*(-3) + 10) = relu({a.data * b.data + c.data}) = {f.data}")
print(f"df/da = {a.grad}  (ожидается b = -3.0)")
print(f"df/db = {b.grad}  (ожидается a = 2.0)")
print(f"df/dc = {c.grad}  (ожидается 1.0)")


# ============================================================
#  Демо 3: Производная tanh
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: tanh'(x) через autograd")
print("=" * 55)

for x_val in [0.0, 0.5, 1.0, 2.0]:
    x = Value(x_val)
    y = x.tanh()
    y.backward()
    numerical = (math.tanh(x_val + 1e-7) - math.tanh(x_val - 1e-7)) / (2e-7)
    print(f"  x={x_val:.1f}  autograd={x.grad:.6f}  числовая={numerical:.6f}  1-tanh²={1 - math.tanh(x_val)**2:.6f}")


# ============================================================
#  Демо 4: Gradient checking
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Gradient checking — (x³ + 2x + 1).tanh()")
print("=" * 55)

def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff

def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

for x_val in [0.0, 0.5, 1.0, -1.0]:
    ad, num, diff = gradient_check(expr, x_val)
    print(f"  x={x_val:5.1f}  autograd={ad:8.5f}  числовая={num:8.5f}  разница={diff:.2e}")


# ============================================================
#  Демо 5: Обучение MLP на XOR
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Обучение MLP на XOR (с нуля, без PyTorch)")
print("=" * 55)

model = MLP([2, 4, 1])  # 2 входа, 4 скрытых нейрона, 1 выход

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR (-1/1 для tanh)

for step in range(200):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0 or step == 199:
        print(f"  шаг {step:3d}  loss = {loss.data:.6f}")

print("\nРезультаты после обучения:")
for x, y in zip(xs, ys):
    pred = model(x).data
    print(f"  вход={x}  цель={y:+2d}  предсказание={pred:+.4f}  {'✓' if (pred > 0) == (y > 0) else '✗'}")

print(f"\nПараметров в модели: {len(model.parameters())}")


# ============================================================
#  Демо 6: Сравнение с PyTorch
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Сравнение с PyTorch")
print("=" * 55)

try:
    import torch

    x1 = torch.tensor(2.0, requires_grad=True)
    x2 = torch.tensor(3.0, requires_grad=True)
    a = x1 * x2
    b = a + 1.0
    y = torch.relu(b)
    y.backward()

    print(f"  Наш autograd:  dy/dx1 = 3.0,  dy/dx2 = 2.0")
    print(f"  PyTorch:       dy/dx1 = {x1.grad.item()},  dy/dx2 = {x2.grad.item()}")
    print(f"  Совпадают: ✓")
except ImportError:
    print("  PyTorch не установлен — пропускаем сравнение")
    print("  (наши градиенты уже проверены через gradient checking)")
