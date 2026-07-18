"""
49 — Фреймворк нейросетей с нуля.
Все компоненты: Value, Neuron, Layer, MLP.
Backpropagation через топологическую сортировку.
"""

import random
import math

random.seed(42)


# ─── Value ───────────────────────────────────────────────────────────────────

class Value:
    """Число с автодифференцированием (forward + backward)."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value({self.label}={self.data:.4f})"

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

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

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

    def __truediv__(self, other):
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

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
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Топологическая сортировка + обратное распространение."""
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


# ─── Neuron ──────────────────────────────────────────────────────────────────

class Neuron:
    """Один нейрон: взвешенная сумма + функция активации."""

    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        return act

    def parameters(self):
        return self.w + [self.b]


# ─── Layer ───────────────────────────────────────────────────────────────────

class Layer:
    """Полносвязный слой из nout нейронов по nin входов."""

    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


# ─── MLP ─────────────────────────────────────────────────────────────────────

class MLP:
    """Многослойный персептрон."""

    def __init__(self, nin, nouts, activation='tanh'):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activation)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ─── Утилита: MSE loss ──────────────────────────────────────────────────────

def mse_loss(targets, preds):
    if not isinstance(preds, list):
        preds = [preds]
    return sum((t - p) ** 2 for t, p in zip(targets, preds))


# ─── Утилита: train loop ────────────────────────────────────────────────────

def train(mlp, X_train, y_train, epochs=200, lr=0.05, verbose_every=50):
    """Простой SGD train loop. Возвращает список loss по эпохам."""
    history = []

    for epoch in range(epochs):
        # forward
        ypred = [mlp(x) for x in X_train]
        loss = mse_loss(y_train, ypred)

        # backward
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in mlp.parameters():
            p.data -= lr * p.grad

        history.append(loss.data)

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>4d} | loss {loss.data:.6f}")

    return history


# ─── Утилита: визуализация ASCII-графика ─────────────────────────────────────

def plot_ascii(history, width=50, height=15):
    """Рисует график loss по эпохам в консоли."""
    mn, mx = min(history), max(history)
    rng = mx - mn if mx != mn else 1.0

    print("\n  Loss по эпохам:")
    print("  " + "─" * (width + 2))

    # rows top → bottom
    for row in range(height - 1, -1, -1):
        threshold = mn + (row / (height - 1)) * rng
        line = ""
        step = max(1, len(history) // width)
        for i in range(0, len(history), step):
            if history[i] >= threshold:
                line += "█"
            else:
                line += " "
        label = f"{threshold:8.4f}" if row in (0, height // 2, height - 1) else "         "
        print(f"  {label}│{line}│")

    print("  " + " " * 9 + "└" + "─" * width + "┘")
    print("  " + " " * 10 + "0" + " " * (width // 2 - 2) + f"epoch={len(history)}")


# ═════════════════════════════════════════════════════════════════════════════
#  ДЕМО
# ═════════════════════════════════════════════════════════════════════════════

def demo1_value():
    print("=" * 65)
    print("  ДЕМО 1: Value — базовые операции и автоматическое дифференцирование")
    print("=" * 65)

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    print(f"\n  L = (a * b + c) * f")
    print(f"  a = {a.data}, b = {b.data}, c = {c.data}, f = {f.data}")
    print(f"  L = ({a.data} * {b.data} + {c.data}) * {f.data} = {L.data:.4f}")

    L.backward()

    print(f"\n  After backward:")
    print(f"    dL/da = {a.grad:.4f}   (ожидается: b * f = {b.data * f.data})")
    print(f"    dL/db = {b.grad:.4f}   (ожидается: a * f = {a.data * f.data})")
    print(f"    dL/dc = {c.grad:.4f}   (ожидается: f = {f.data})")
    print(f"    dL/df = {f.grad:.4f}   (ожидается: d = {d.data})")

    print("\n  ReLU и sigmoid:")
    x = Value(0.5, label='x')
    print(f"    x = {x.data}")
    print(f"    relu(x)  = {x.relu().data:.4f}")
    x2 = Value(0.5, label='x2')
    print(f"    sigmoid(x) = {x2.sigmoid().data:.4f}")
    x3 = Value(0.5, label='x3')
    print(f"    tanh(x)    = {x3.tanh().data:.4f}")


def demo2_mlp():
    print("\n" + "=" * 65)
    print("  ДЕМО 2: MLP — forward pass")
    print("=" * 65)

    mlp = MLP(3, [4, 4, 1], activation='tanh')
    n_params = len(mlp.parameters())
    print(f"\n  Архитектура: 3 → [4, 4] → 1 (tanh)")
    print(f"  Количество параметров: {n_params}")

    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = mlp(x)
    print(f"\n  Forward: mlp([1, 2, 3]) = {out.data:.4f}")

    print("\n  Параметры слоёв:")
    for i, layer in enumerate(mlp.layers):
        print(f"    Layer {i}: {len(layer.neurons)} нейронов, "
              f"{len(layer.parameters())} параметров")


def demo3_xor():
    print("\n" + "=" * 65)
    print("  ДЕМО 3: Обучение MLP на XOR")
    print("=" * 65)

    X_train = [
        [Value(0), Value(0)],
        [Value(0), Value(1)],
        [Value(1), Value(0)],
        [Value(1), Value(1)],
    ]
    y_train = [Value(0), Value(1), Value(1), Value(0)]

    mlp = MLP(2, [4, 1], activation='tanh')
    print(f"\n  Архитектура: 2 → [4] → 1 (tanh)")
    print(f"  Эпохи: 300, LR: 0.05\n")

    history = train(mlp, X_train, y_train, epochs=300, lr=0.05, verbose_every=50)

    print("\n  Предсказания после обучения:")
    for i, x in enumerate(X_train):
        pred = mlp(x)
        print(f"    输入 {[int(x[0].data), int(x[1].data)]} → "
              f"{pred.data:.4f} (ожидалось: {int(y_train[i].data)})")


def demo4_visualization():
    print("\n" + "=" * 65)
    print("  ДЕМО 4: Визуализация loss по эпохам")
    print("=" * 65)

    X_train = [
        [Value(0), Value(0)],
        [Value(0), Value(1)],
        [Value(1), Value(0)],
        [Value(1), Value(1)],
    ]
    y_train = [Value(0), Value(1), Value(1), Value(0)]

    random.seed(42)
    mlp = MLP(2, [8, 1], activation='tanh')
    print(f"\n  Архитектура: 2 → [8] → 1 (tanh)")
    print(f"  Эпохи: 500, LR: 0.05\n")

    history = train(mlp, X_train, y_train, epochs=500, lr=0.05, verbose_every=100)

    plot_ascii(history, width=55, height=14)

    print("\n  Финальные предсказания:")
    for i, x in enumerate(X_train):
        pred = mlp(x)
        print(f"    输入 {[int(x[0].data), int(x[1].data)]} → {pred.data:.4f}")


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo1_value()
    demo2_mlp()
    demo3_xor()
    demo4_visualization()
    print("\n" + "=" * 65)
    print("  Все демонстрации завершены.")
    print("=" * 65)
