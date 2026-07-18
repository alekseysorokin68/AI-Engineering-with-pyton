"""
Multi-Layer Neural Network from Scratch
========================================
Многослойная нейросеть, реализованная с нуля на Python.
Классы: Neuron, Layer, MLP
Активации: sigmoid, tanh, ReLU
Обучение на XOR
"""

import random
import math

random.seed(42)


# ===================== Activation Functions =====================

def sigmoid(x):
    """Сигмоида: σ(x) = 1 / (1 + e^(-x))"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def sigmoid_derivative(output):
    """Производная сигмоиды: σ'(x) = σ(x) * (1 - σ(x))"""
    return output * (1.0 - output)


def tanh_act(x):
    """Гиперболический тангенс"""
    return math.tanh(x)


def tanh_derivative(output):
    """Производная tanh: 1 - tanh²(x)"""
    return 1.0 - output * output


def relu(x):
    """ReLU: max(0, x)"""
    return x if x > 0 else 0.0


def relu_derivative(output):
    """Производная ReLU: 1 если x > 0, иначе 0"""
    return 1.0 if output > 0 else 0.0


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh_act, tanh_derivative),
    "relu": (relu, relu_derivative),
}


# ===================== Neuron =====================

class Neuron:
    """Один нейрон: взвешенная сумма + активация."""

    def __init__(self, n_inputs, activation="sigmoid"):
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        self.bias = random.uniform(-1.0, 1.0)
        self.activation_fn, self.activation_deriv = ACTIVATIONS[activation]
        self.output = 0.0
        self.delta = 0.0
        self._raw_input = [0.0] * n_inputs

    def forward(self, inputs):
        """Вычисление выхода нейрона."""
        self._raw_input = inputs
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_fn(total)
        return self.output

    def update_weights(self, inputs, lr):
        """Обновление весов по градиентному спуску."""
        for i in range(len(self.weights)):
            self.weights[i] += lr * self.delta * inputs[i]
        self.bias += lr * self.delta


# ===================== Layer =====================

class Layer:
    """Полносвязный слой из нейронов."""

    def __init__(self, n_neurons, n_inputs, activation="sigmoid"):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def forward(self, inputs):
        """Forward pass: каждый нейрон получает одни и те же входы."""
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, target, prev_layer_outputs, lr):
        """Backpropagation для скрытого или выходного слоя."""
        for i, neuron in enumerate(self.neurons):
            if target is not None:
                error = target[i] - neuron.output
                neuron.delta = error * neuron.activation_deriv(neuron.output)
            else:
                error = 0.0
                neuron.delta = 0.0
            neuron.update_weights(prev_layer_outputs, lr)

    @property
    def outputs(self):
        return [n.output for n in self.neurons]

    @property
    def n_neurons(self):
        return len(self.neurons)


# ===================== MLP =====================

class MLP:
    """Многослойный перцептрон."""

    def __init__(self, layer_sizes, activations=None):
        """
        layer_sizes: [n_input, n_hidden1, n_hidden2, ..., n_output]
        activations: список активаций для каждого скрытого слоя (не для входного).
                     Если None — все слои используют sigmoid.
        """
        if activations is None:
            activations = ["sigmoid"] * (len(layer_sizes) - 1)

        self.layers = []
        for i in range(1, len(layer_sizes)):
            act = activations[min(i - 1, len(activations) - 1)]
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1], act))

    def forward(self, inputs):
        """Forward pass через все слои."""
        self._last_input = inputs
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, targets, lr):
        """Backpropagation от последнего слоя к первому."""
        n_layers = len(self.layers)

        # Выходной слой
        output_layer = self.layers[-1]
        prev_outputs = self.layers[-2].outputs if n_layers > 1 else self._last_input
        output_layer.backward(targets, prev_outputs, lr)

        # Скрытые слои (от предпоследнего к первому)
        for i in range(n_layers - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j, neuron in enumerate(layer.neurons):
                error = sum(
                    next_neuron.weights[j] * next_neuron.delta
                    for next_neuron in next_layer.neurons
                )
                neuron.delta = error * neuron.activation_deriv(neuron.output)

            prev_outputs = self.layers[i - 1].outputs if i > 0 else self._last_input
            for neuron in layer.neurons:
                neuron.update_weights(prev_outputs, lr)

    def train(self, X, y, epochs=1000, lr=0.5):
        """Обучение на данных X, y."""
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                output = self.forward(xi)
                total_loss += sum((t - o) ** 2 for t, o in zip(yi, output))
                self.backward(yi, lr)
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch + 1:>5d} | MSE = {avg_loss:.6f}")
        return losses


# ===================== DEMO 1: Forward Pass через 2 слоя =====================

def demo1_forward_pass():
    print("=" * 60)
    print("DEMO 1: Forward pass через 2 скрытых слоя")
    print("=" * 60)

    random.seed(42)
    mlp = MLP([3, 4, 2], activations=["sigmoid", "sigmoid"])

    inputs = [0.5, 0.3, -0.7]
    output = mlp.forward(inputs)

    print(f"Вход:  {inputs}")
    print(f"Выход: {[round(v, 6) for v in output]}")
    print()

    print("Структура сети:")
    print(f"  Входной слой:   3 нейрона")
    for i, layer in enumerate(mlp.layers, 1):
        print(f"  Слой {i}:         {layer.n_neurons} нейронов")
    print()

    print("Веса первого нейрона первого скрытого слоя:")
    n = mlp.layers[0].neurons[0]
    print(f"  weights = {[round(w, 4) for w in n.weights]}")
    print(f"  bias    = {n.bias:.4f}")
    print()


# ===================== DEMO 2: Сравнение активаций =====================

def demo2_activation_comparison():
    print("=" * 60)
    print("DEMO 2: Сравнение функций активации (sigmoid vs tanh vs ReLU)")
    print("=" * 60)

    random.seed(42)
    test_input = 0.7

    print(f"\nВход x = {test_input}\n")

    print(f"{'Функция':<12} {'Значение':<12} {'Производная':<12}")
    print("-" * 36)

    for name, (fn, deriv) in ACTIVATIONS.items():
        val = fn(test_input)
        d = deriv(fn(test_input))
        print(f"{name:<12} {val:<12.6f} {d:<12.6f}")

    # Поведение на краях
    print("\n--- Поведение на экстремальных значениях ---\n")
    test_vals = [-10.0, -1.0, 0.0, 1.0, 10.0]
    print(f"{'x':<8} {'sigmoid':<12} {'tanh':<12} {'relu':<12}")
    print("-" * 44)
    for x in test_vals:
        s = sigmoid(x)
        t = tanh_act(x)
        r = relu(x)
        print(f"{x:<8.1f} {s:<12.6f} {t:<12.6f} {r:<12.6f}")

    # Сравнение обучения на XOR с разными активациями
    print("\n--- Сравнение сходимости на XOR (500 эпох) ---\n")

    X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_xor = [[0], [1], [1], [0]]

    for act_name in ["sigmoid", "tanh"]:
        random.seed(42)
        mlp = MLP([2, 4, 1], activations=[act_name])
        print(f"Активация: {act_name}")
        mlp.train(X_xor, y_xor, epochs=500, lr=0.5)
        results = [mlp.forward(xi) for xi in X_xor]
        for xi, yi, ri in zip(X_xor, y_xor, results):
            print(f"  {xi} → {ri[0]:.4f}  (ожидается {yi[0]})")
        print()


# ===================== DEMO 3: Обучение MLP на XOR =====================

def demo3_xor_training():
    print("=" * 60)
    print("DEMO 3: Обучение MLP на XOR")
    print("=" * 60)

    random.seed(42)

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    print("\nАрхитектура: [2] → [4] → [1]")
    print("Активация: sigmoid")
    print("Learning rate: 0.5")
    print("Эпохи: 2000\n")

    mlp = MLP([2, 4, 1], activations=["sigmoid", "sigmoid"])

    losses = mlp.train(X, y, epochs=2000, lr=0.5)

    print("\n--- Результаты после обучения ---\n")
    print(f"{'Вход':<12} {'Выход':<12} {'Ожидается':<12} {'Решение':<10}")
    print("-" * 46)

    for xi, yi in zip(X, y):
        output = mlp.forward(xi)
        pred = 1 if output[0] >= 0.5 else 0
        match = "✓" if pred == yi[0] else "✗"
        print(f"{str(xi):<12} {output[0]:<12.6f} {yi[0]:<12} {pred:<10} {match}")

    print(f"\nФинальная MSE: {losses[-1]:.6f}")


# ===================== DEMO 4: Визуализация ошибки =====================

def demo4_loss_visualization():
    print("\n" + "=" * 60)
    print("DEMO 4: Визуализация ошибки по эпохам")
    print("=" * 60)

    random.seed(42)

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    mlp = MLP([2, 4, 1], activations=["sigmoid"])
    losses = mlp.train(X, y, epochs=2000, lr=0.5)

    # ASCII-график
    print("\n--- MSE по эпохам (ASCII-график) ---\n")

    max_loss = max(losses)
    min_loss = min(losses)
    width = 50

    # Показываем каждую 100-ю эпоху
    sample_indices = list(range(0, len(losses), 100))
    if (len(losses) - 1) not in sample_indices:
        sample_indices.append(len(losses) - 1)

    print(f"{'Эпоха':<8} {'MSE':<12} График")
    print("-" * (8 + 12 + width + 5))

    for idx in sample_indices:
        loss = losses[idx]
        bar_len = int((loss / max_loss) * width) if max_loss > 0 else 0
        bar = "█" * bar_len + "░" * (width - bar_len)
        print(f"{idx + 1:<8} {loss:<12.6f} |{bar}|")

    # Сравнение: несколько архитектур
    print("\n--- Сравнение архитектур (2000 эпох) ---\n")

    configs = [
        ("[2] → [2] → [1]",   [2, 2, 1]),
        ("[2] → [4] → [1]",   [2, 4, 1]),
        ("[2] → [8] → [1]",   [2, 8, 1]),
        ("[2] → [4] → [4] → [1]", [2, 4, 4, 1]),
    ]

    print(f"{'Архитектура':<30} {'MSE (финальная)':<20} {'MSE (эпоха 1000)':<20}")
    print("-" * 70)

    for name, sizes in configs:
        random.seed(42)
        mlp = MLP(sizes, activations=["sigmoid"] * (len(sizes) - 1))
        losses = mlp.train(X, y, epochs=2000, lr=0.5)
        mid_loss = losses[999] if len(losses) > 999 else losses[-1]
        print(f"{name:<30} {losses[-1]:<20.6f} {mid_loss:<20.6f}")

    print()


# ===================== Main =====================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Multi-Layer Neural Network from Scratch                ║")
    print("║  Многослойная нейросеть: Neuron → Layer → MLP           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    demo1_forward_pass()
    demo2_activation_comparison()
    demo3_xor_training()
    demo4_loss_visualization()

    print("=" * 60)
    print("Все демонстрации завершены.")
    print("=" * 60)
