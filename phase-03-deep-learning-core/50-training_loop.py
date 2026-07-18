"""
50 — Полный цикл обучения нейросети (Training Loop)
====================================================

Нейросеть строится С НУЛЯ на чистом Python:
  • Dense-слои, активации, loss-функции
  • Forward pass → Loss → Backward pass → Weight update
  • Train / Validation раздельные циклы
  • Никаких импортов numpy, torch, sklearn

Демонстрации:
  Demo 1 — Forward + Backward pass пошагово
  Demo 2 — Полный train loop (mini-batch SGD)
  Demo 3 — Validation: train vs test loss
  Demo 4 — Сравнение гиперпараметров (lr, batch_size)
"""

import random
import math

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def rand_matrix(rows, cols):
    """Xavier-инициализация."""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def rand_vector(n):
    return [random.uniform(-0.01, 0.01) for _ in range(n)]


def zeros_vector(n):
    return [0.0] * n


def mat_vec_mul(mat, vec):
    """mat: (m×n), vec: (n,) → (m,)"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def outer(a, b):
    """(m,) × (n,) → (m×n)"""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vec_mul_scalar(v, s):
    return [x * s for x in v]


# ═══════════════════════════════════════════════════════════════════════════════
# АКТИВАЦИИ И ИХ ПРОИЗВОДНЫЕ
# ═══════════════════════════════════════════════════════════════════════════════

def relu(x):
    return max(0.0, x)


def relu_grad(x):
    return 1.0 if x > 0 else 0.0


def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_grad_from_output(out):
    return out * (1.0 - out)


def tanh_fn(x):
    return math.tanh(x)


def tanh_grad_from_output(out):
    return 1.0 - out * out


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS-ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def mse_loss(pred, target):
    n = len(pred)
    return sum((pred[i] - target[i]) ** 2 for i in range(n)) / n


def mse_grad(pred, target):
    n = len(pred)
    return [2.0 * (pred[i] - target[i]) / n for i in range(n)]


def bce_loss(pred, target):
    eps = 1e-7
    n = len(pred)
    total = 0.0
    for i in range(n):
        p = max(eps, min(1 - eps, pred[i]))
        total -= target[i] * math.log(p) + (1 - target[i]) * math.log(1 - p)
    return total / n


def bce_grad(pred, target):
    eps = 1e-7
    n = len(pred)
    return [
        (-target[i] / max(eps, pred[i]) + (1 - target[i]) / max(eps, 1 - pred[i])) / n
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER: Dense-слой (forward + backward + param update)
# ═══════════════════════════════════════════════════════════════════════════════

class DenseLayer:
    """
    Линейный слой: z = W·x + b
    Поддерживает.relu / .sigmoid / .tanh
    """

    def __init__(self, in_dim, out_dim, activation="relu"):
        self.W = rand_matrix(out_dim, in_dim)
        self.b = rand_vector(out_dim)
        self.activation = activation

        # Кэш для backward pass
        self.input = None
        self.z = None          # pre-activation
        self.a = None          # post-activation
        self.dW = None
        self.db = None

    def forward(self, x):
        """x: (in_dim,) → (out_dim,)"""
        self.input = x[:]
        self.z = mat_vec_mul(self.W, x)
        self.z = vec_add(self.z, self.b)

        if self.activation == "relu":
            self.a = [relu(v) for v in self.z]
        elif self.activation == "sigmoid":
            self.a = [sigmoid(v) for v in self.z]
        elif self.activation == "tanh":
            self.a = [tanh_fn(v) for v in self.z]
        else:
            self.a = self.z[:]

        return self.a

    def backward(self, grad_output, learning_rate):
        """
        grad_output: (out_dim,) — градиент loss по a этого слоя.
        Вычисляет dW, db и передаёт grad_input предыдущему слою.
        """
        out_dim = len(self.a)
        in_dim = len(self.input)

        # Производная активации
        if self.activation == "relu":
            act_grad = [relu_grad(self.z[i]) for i in range(out_dim)]
        elif self.activation == "sigmoid":
            act_grad = [sigmoid_grad_from_output(self.a[i]) for i in range(out_dim)]
        elif self.activation == "tanh":
            act_grad = [tanh_grad_from_output(self.a[i]) for i in range(out_dim)]
        else:
            act_grad = [1.0] * out_dim

        # delta = grad_output ⊙ activation'(z)
        delta = [grad_output[i] * act_grad[i] for i in range(out_dim)]

        # Градиенты параметров
        self.dW = outer(delta, self.input)      # (out_dim × in_dim)
        self.db = delta[:]

        # Градиент для предыдущего слоя
        # grad_input[j] = sum_i(delta[i] * W[i][j])
        grad_input = [0.0] * in_dim
        for i in range(out_dim):
            for j in range(in_dim):
                grad_input[j] += self.W[i][j] * delta[i]

        return grad_input


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """Последовательность Dense-слоёв."""

    def __init__(self):
        self.layers = []

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, learning_rate):
        """Цепное правило: grad от последнего слоя → к первому."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def update_weights(self, learning_rate):
        """SGD: W -= lr * dW, b -= lr * db."""
        for layer in self.layers:
            if layer.dW is not None:
                out_dim = len(layer.W)
                in_dim = len(layer.W[0])
                for i in range(out_dim):
                    for j in range(in_dim):
                        layer.W[i][j] -= learning_rate * layer.dW[i][j]
                    layer.b[i] -= learning_rate * layer.db[i]


# ═══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ДАННЫХ: XOR-задача (4 точки → 1 выход)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_xor_data(n_samples=400):
    """XOR с шумом: y = x1 ⊕ x2 + noise."""
    X, y = [], []
    for _ in range(n_samples):
        x1 = random.random() * 2 - 1
        x2 = random.random() * 2 - 1
        label = 1.0 if (x1 > 0) != (x2 > 0) else 0.0
        noise = random.gauss(0, 0.05)
        X.append([x1, x2])
        y.append([label + noise])
    return X, y


def generate_spiral_data(n_points=200):
    """Two-class spiral (harder than XOR)."""
    X, y = [], []
    for i in range(n_points):
        t = i / n_points * 4 * math.pi + random.gauss(0, 0.2)
        r = i / n_points * 0.8 + random.gauss(0, 0.05)
        X.append([r * math.cos(t), r * math.sin(t)])
        y.append([1.0])
    for i in range(n_points):
        t = i / n_points * 4 * math.pi + math.pi + random.gauss(0, 0.2)
        r = i / n_points * 0.8 + random.gauss(0, 0.05)
        X.append([r * math.cos(t), r * math.sin(t)])
        y.append([0.0])
    return X, y


def train_test_split(X, y, test_ratio=0.2):
    """Простой random split."""
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


def make_batches(X, y, batch_size):
    """Разбить данные на мини-батчи."""
    indices = list(range(len(X)))
    random.shuffle(indices)
    batches = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_idx = indices[start:end]
        bx = [X[i] for i in batch_idx]
        by = [y[i] for i in batch_idx]
        batches.append((bx, by))
    return batches


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL ЦИКЛЫ
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(net, X, y, batch_size, learning_rate, loss_fn, loss_grad_fn):
    """Одна эпоха: forward → loss → backward → update."""
    batches = make_batches(X, y, batch_size)
    total_loss = 0.0
    n_samples = 0

    for bx, by in batches:
        batch_loss = 0.0
        for xi, yi in zip(bx, by):
            # Forward
            pred = net.forward(xi)
            # Loss
            loss = loss_fn(pred, yi)
            batch_loss += loss
            n_samples += 1
            # Backward
            grad = loss_grad_fn(pred, yi)
            net.backward(grad, learning_rate)
            # Update
            net.update_weights(learning_rate)

        total_loss += batch_loss

    return total_loss / n_samples


def evaluate(net, X, y, loss_fn):
    """Forward-only: средний loss на тесте."""
    total_loss = 0.0
    for xi, yi in zip(X, y):
        pred = net.forward(xi)
        total_loss += loss_fn(pred, yi)
    return total_loss / len(X)


def predict(net, x):
    """Forward-only: предсказание одного样本."""
    return net.forward(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ПОСТРОЕНИЯ НЕЙРОСЕТИ
# ═══════════════════════════════════════════════════════════════════════════════

def build_network(layer_specs):
    """
    layer_specs: список (out_dim, activation).
    Входной размер определяется автоматически из первого batch.
    """
    net = NeuralNetwork()
    for i, (out_dim, act) in enumerate(layer_specs):
        if i == 0:
            net.add(DenseLayer(2, out_dim, activation=act))  # default input=2
        else:
            prev_dim = layer_specs[i - 1][0]
            net.add(DenseLayer(prev_dim, out_dim, activation=act))
    return net


def build_network_with_input(in_dim, layer_specs):
    net = NeuralNetwork()
    prev = in_dim
    for out_dim, act in layer_specs:
        net.add(DenseLayer(prev, out_dim, activation=act))
        prev = out_dim
    return net


def accuracy(net, X, y, threshold=0.5):
    """Accuracy для бинарной классификации."""
    correct = 0
    for xi, yi in zip(X, y):
        pred = net.forward(xi)
        predicted_class = 1.0 if pred[0] > threshold else 0.0
        target_class = 1.0 if yi[0] > 0.5 else 0.0
        if predicted_class == target_class:
            correct += 1
    return correct / len(X)


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 1: Forward + Backward Pass — пошагово
# ═══════════════════════════════════════════════════════════════════════════════

def demo1_forward_backward():
    print("=" * 72)
    print("DEMO 1: Forward + Backward Pass — пошагово")
    print("=" * 72)

    random.seed(42)

    # Простая сеть: 2 → 4 (relu) → 1 (sigmoid)
    net = NeuralNetwork()
    net.add(DenseLayer(2, 4, activation="relu"))
    net.add(DenseLayer(4, 1, activation="sigmoid"))

    x = [0.5, -0.3]
    target = [1.0]

    print(f"\nВход:       x = {x}")
    print(f"Цель:       y = {target}\n")

    # --- FORWARD ---
    print("--- FORWARD PASS ---")
    pred = net.forward(x)
    print(f"  Слой 1 (relu):    {net.layers[0].a}")
    print(f"  Слой 2 (sigmoid): {net.layers[1].a}")
    print(f"  Выход (pred):     {pred}")

    loss = mse_loss(pred, target)
    print(f"\n  Loss (MSE):       {loss:.6f}")

    # --- BACKWARD ---
    print("\n--- BACKWARD PASS ---")
    grad = mse_grad(pred, target)
    print(f"  Градиент loss:    {grad}")

    net.backward(grad, learning_rate=0.1)

    print(f"  Слой 2 dW shape:  {len(net.layers[1].dW)}×{len(net.layers[1].dW[0])}")
    print(f"  Слой 2 db:        {net.layers[1].db}")
    print(f"  Слой 1 dW shape:  {len(net.layers[0].dW)}×{len(net.layers[0].dW[0])}")
    print(f"  Слой 1 db:        {net.layers[0].db}")

    # --- WEIGHT UPDATE ---
    print("\n--- WEIGHT UPDATE (lr=0.1) ---")
    old_w2 = net.layers[1].W[0][0]
    net.update_weights(learning_rate=0.1)
    new_w2 = net.layers[1].W[0][0]
    print(f"  W[2][0][0]: {old_w2:.6f} → {new_w2:.6f}")

    # Проверяем, что loss уменьшился
    pred2 = net.forward(x)
    loss2 = mse_loss(pred2, target)
    print(f"\n  После обновления:")
    print(f"  Новый pred:       {pred2}")
    print(f"  Новый loss:       {loss2:.6f}  (было {loss:.6f})")
    print(f"  Loss уменьшился:  {loss2 < loss}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 2: Полный Train Loop (XOR + spiral)
# ═══════════════════════════════════════════════════════════════════════════════

def demo2_full_train_loop():
    print("=" * 72)
    print("DEMO 2: Полный Train Loop — XOR dataset")
    print("=" * 72)

    random.seed(42)

    X, y = generate_xor_data(n_samples=400)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

    print(f"\nДанные:     {len(X_train)} train, {len(X_test)} test")
    print(f"Архитектура: 2 → 8 (relu) → 4 (relu) → 1 (sigmoid)")
    print(f"Optimizer:   SGD, lr=0.05, batch_size=16")
    print(f"Loss:        BCE\n")

    net = build_network_with_input(2, [
        (8, "relu"),
        (4, "relu"),
        (1, "sigmoid"),
    ])

    epochs = 50
    lr = 0.05
    batch_size = 16

    print(f"{'Эпоха':>6} | {'Train Loss':>12} | {'Test Loss':>12} | {'Train Acc':>10} | {'Test Acc':>10}")
    print("-" * 66)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(net, X_train, y_train, batch_size, lr, bce_loss, bce_grad)
        test_loss = evaluate(net, X_test, y_test, bce_loss)

        if epoch % 5 == 0 or epoch == 1:
            train_acc = accuracy(net, X_train, y_train)
            test_acc = accuracy(net, X_test, y_test)
            print(f"{epoch:>6} | {train_loss:>12.6f} | {test_loss:>12.6f} | {train_acc:>9.1%} | {test_acc:>9.1%}")

    final_train_acc = accuracy(net, X_train, y_train)
    final_test_acc = accuracy(net, X_test, y_test)
    print(f"\nФинал: train_acc={final_train_acc:.1%}, test_acc={final_test_acc:.1%}")

    # Проверим отдельные предсказания
    print("\nПримеры предсказаний:")
    test_points = [
        ([0.9, 0.9], "ожидается 0 (оба положительные)"),
        ([0.9, -0.9], "ожидается 1 (разные знаки)"),
        ([-0.5, 0.5], "ожидается 1 (разные знаки)"),
        ([-0.8, -0.3], "ожидается 0 (оба отрицательные)"),
    ]
    for xi, desc in test_points:
        pred = predict(net, xi)
        label = "1" if pred[0] > 0.5 else "0"
        print(f"  {xi} → {pred[0]:.4f} (class={label})  [{desc}]")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 3: Validation — Train vs Test Loss (Spiral)
# ═══════════════════════════════════════════════════════════════════════════════

def demo3_validation():
    print("=" * 72)
    print("DEMO 3: Validation — Train vs Test Loss (Spiral dataset)")
    print("=" * 72)

    random.seed(42)

    X, y = generate_spiral_data(n_points=150)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

    print(f"\nДанные:     {len(X_train)} train, {len(X_test)} test (spiral)")
    print(f"Архитектура: 2 → 16 (relu) → 8 (relu) → 1 (sigmoid)")
    print(f"Optimizer:   SGD, lr=0.02, batch_size=32")
    print(f"Loss:        BCE\n")

    net = build_network_with_input(2, [
        (16, "relu"),
        (8, "relu"),
        (1, "sigmoid"),
    ])

    epochs = 80
    lr = 0.02
    batch_size = 32

    train_losses = []
    test_losses = []

    print(f"{'Эпоха':>6} | {'Train Loss':>12} | {'Test Loss':>12} | {'Δ(test-train)':>14}")
    print("-" * 56)

    for epoch in range(1, epochs + 1):
        tl = train_one_epoch(net, X_train, y_train, batch_size, lr, bce_loss, bce_grad)
        vl = evaluate(net, X_test, y_test, bce_loss)
        train_losses.append(tl)
        test_losses.append(vl)

        if epoch % 10 == 0 or epoch == 1:
            delta = vl - tl
            print(f"{epoch:>6} | {tl:>12.6f} | {vl:>12.6f} | {delta:>+14.6f}")

    # Overfitting analysis
    min_test_epoch = test_losses.index(min(test_losses)) + 1
    min_test_loss = min(test_losses)

    print(f"\nЛучший test loss: {min_test_loss:.6f} на эпохе {min_test_epoch}")
    print(f"Финальный train loss: {train_losses[-1]:.6f}")
    print(f"Финальный test loss:  {test_losses[-1]:.6f}")

    if test_losses[-1] > min_test_loss * 1.2:
        print("→ Видно начало переобучения: test loss растёт при падающем train loss")
    else:
        print("→ Модель стабильна: train и test loss близки")

    # ASCII-график
    print("\n--- ASCII-график losses ---")
    all_losses = train_losses + test_losses
    max_loss = max(all_losses)
    min_loss = min(all_losses)
    width = 50

    def bar(val):
        norm = (val - min_loss) / (max_loss - min_loss + 1e-9)
        return "#" * int(norm * width)

    for epoch in range(0, epochs, 5):
        tl = train_losses[epoch]
        vl = test_losses[epoch]
        print(f"  E{epoch + 1:>3} T|{bar(tl):<50}| {tl:.4f}")
        print(f"       V|{bar(vl):<50}| {vl:.4f}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 4: Сравнение Гиперпараметров
# ═══════════════════════════════════════════════════════════════════════════════

def demo4_hyperparameter_comparison():
    print("=" * 72)
    print("DEMO 4: Сравнение Гиперпараметров (lr × batch_size)")
    print("=" * 72)

    random.seed(42)
    X, y = generate_xor_data(n_samples=400)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

    configs = [
        {"name": "lr=0.5, bs=8",   "lr": 0.5,   "batch": 8},
        {"name": "lr=0.1, bs=8",   "lr": 0.1,   "batch": 8},
        {"name": "lr=0.01, bs=8",  "lr": 0.01,  "batch": 8},
        {"name": "lr=0.1, bs=32",  "lr": 0.1,   "batch": 32},
        {"name": "lr=0.1, bs=128", "lr": 0.1,   "batch": 128},
        {"name": "lr=0.01, bs=128", "lr": 0.01,  "batch": 128},
    ]

    epochs = 40
    results = []

    for cfg in configs:
        random.seed(42)  # Одинаковый старт для честного сравнения
        net = build_network_with_input(2, [
            (8, "relu"),
            (4, "relu"),
            (1, "sigmoid"),
        ])

        print(f"\n--- {cfg['name']} ---")
        history = {"train": [], "test": []}

        for epoch in range(1, epochs + 1):
            tl = train_one_epoch(net, X_train, y_train, cfg["batch"], cfg["lr"], bce_loss, bce_grad)
            vl = evaluate(net, X_test, y_test, bce_loss)
            history["train"].append(tl)
            history["test"].append(vl)

        final_train = history["train"][-1]
        final_test = history["test"][-1]
        min_test = min(history["test"])
        min_test_ep = history["test"].index(min_test) + 1
        test_acc = accuracy(net, X_test, y_test)

        results.append({
            "name": cfg["name"],
            "final_train": final_train,
            "final_test": final_test,
            "min_test": min_test,
            "min_test_ep": min_test_ep,
            "test_acc": test_acc,
        })

        print(f"  Final train: {final_train:.6f}, final test: {final_test:.6f}, "
              f"best test: {min_test:.6f} (ep {min_test_ep}), acc: {test_acc:.1%}")

    # Сводная таблица
    print("\n\n" + "=" * 78)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 78)
    print(f"{'Конфигурация':>20} | {'Train Loss':>11} | {'Test Loss':>11} | {'Best Test':>10} | {'Test Acc':>9}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:>20} | {r['final_train']:>11.6f} | {r['final_test']:>11.6f} | "
              f"{r['min_test']:>10.6f} | {r['test_acc']:>8.1%}")

    # Вывод
    best = min(results, key=lambda r: r["final_test"])
    worst = max(results, key=lambda r: r["final_test"])

    print(f"\nЛучшая:  {best['name']}  (test_acc={best['test_acc']:.1%})")
    print(f"Худшая:  {worst['name']}  (test_acc={worst['test_acc']:.1%})")

    print("\nВыводы:")
    print("  • Большой lr (0.5) — нестабильное обучение, потери не падают")
    print("  • Маленький lr (0.01) — медленно, но стабильно")
    print("  • Оптимальный lr (0.1) с малым batch — быстрая и точная сходимость")
    print("  • Большой batch (128) — менее точное обновление градиентов")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo1_forward_backward()
    demo2_full_train_loop()
    demo3_validation()
    demo4_hyperparameter_comparison()

    print("=" * 72)
    print("Все 4 демонстрации завершены.")
    print("=" * 72)
