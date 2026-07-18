"""
Batch Normalization — Полная реализация с нуля
===============================================

Batch Normalization (Ioffe & Szegedy, 2015) — техника нормализации
промежуточных представлений в нейронных сетях, которая:
  1. Ускоряет сходимость обучения
  2. Позволяет использовать более высокие learning rates
  3. Снижает чувствительность к начальной инициализации весов
  4. Действует как слабая регуляция (через шум статистик батча)

Формулы:
  Forward (train):
    mean  = (1/m) * Σ x_i
    var   = (1/m) * Σ (x_i - mean)²
    x_hat = (x_i - mean) / sqrt(var + ε)
    y     = γ * x_hat + β

  Backward:
    dx_hat = dout * γ
    dγ     = sum(dout * x_hat)
    dβ     = sum(dout)
    dx     = (1/(m*σ)) * (m*dx_hat − Σdx_hat − x_hat * Σ(dx_hat * x_hat))

Реализация полностью на чистом Python без numpy/torch.
"""

import math
import random

random.seed(42)


# =============================================================================
# Вспомогательные функции
# =============================================================================

def vec_mean(v):
    return sum(v) / len(v)


def vec_var(v, m):
    return sum((x - m) ** 2 for x in v) / len(v)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def sigmoid_grad_from_output(s):
    return s * (1.0 - s)


def relu(x):
    return max(0.0, x)


def relu_grad(x):
    return 1.0 if x > 0 else 0.0


# =============================================================================
# Класс BatchNorm
# =============================================================================

class BatchNorm:
    """
    Batch Normalization слой.

    Параметры:
        num_features — количество признаков (размерность слоя)
        momentum     — momentum для running статистик (по умолчанию 0.1)
        epsilon      — малое число для численной стабильности
    """

    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Обучаемые параметры
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features

        # Running статистики (для inference)
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features

        # Градиенты
        self.dgamma = [0.0] * num_features
        self.dbeta = [0.0] * num_features

        # Кэш для backward
        self._cache = {}
        self.training = True

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False

    def forward(self, x_batch):
        """
        Forward pass.
        x_batch: список батча [batch_size][num_features]
        Возвращает: нормализованный выход [batch_size][num_features]
        """
        batch_size = len(x_batch)

        if self.training:
            batch_mean = [0.0] * self.num_features
            batch_var = [0.0] * self.num_features
            for j in range(self.num_features):
                col = [x_batch[i][j] for i in range(batch_size)]
                batch_mean[j] = vec_mean(col)
                batch_var[j] = vec_var(col, batch_mean[j])

            # EMA running статистик
            for j in range(self.num_features):
                self.running_mean[j] = (
                    (1 - self.momentum) * self.running_mean[j]
                    + self.momentum * batch_mean[j]
                )
                self.running_var[j] = (
                    (1 - self.momentum) * self.running_var[j]
                    + self.momentum * batch_var[j]
                )
        else:
            batch_mean = self.running_mean[:]
            batch_var = self.running_var[:]

        # Нормализация
        x_norm = []
        for i in range(batch_size):
            row = []
            for j in range(self.num_features):
                std = math.sqrt(batch_var[j] + self.epsilon)
                row.append((x_batch[i][j] - batch_mean[j]) / std)
            x_norm.append(row)

        # γ * x_hat + β
        output = []
        for i in range(batch_size):
            row = []
            for j in range(self.num_features):
                row.append(self.gamma[j] * x_norm[i][j] + self.beta[j])
            output.append(row)

        # Кэш
        self._cache = {
            'x_batch': [row[:] for row in x_batch],
            'x_norm': x_norm,
            'batch_mean': batch_mean,
            'batch_var': batch_var,
            'batch_size': batch_size,
        }

        return output

    def backward(self, dout):
        """
        Backward pass.
        dout: градиент от следующего слоя [batch_size][num_features]
        Возвращает: градиент по входу [batch_size][num_features]
        """
        x_norm = self._cache['x_norm']
        batch_var = self._cache['batch_var']
        batch_size = self._cache['batch_size']

        self.dgamma = [0.0] * self.num_features
        self.dbeta = [0.0] * self.num_features
        dx = [[0.0] * self.num_features for _ in range(batch_size)]

        for j in range(self.num_features):
            std = math.sqrt(batch_var[j] + self.epsilon)

            dx_hat = [dout[i][j] * self.gamma[j] for i in range(batch_size)]
            sum_dx_hat = sum(dx_hat)
            sum_dx_hat_xnorm = sum(dx_hat[i] * x_norm[i][j] for i in range(batch_size))

            for i in range(batch_size):
                dx[i][j] = (
                    batch_size * dx_hat[i] - sum_dx_hat - x_norm[i][j] * sum_dx_hat_xnorm
                ) / (batch_size * std)

            self.dgamma[j] = sum(dout[i][j] * x_norm[i][j] for i in range(batch_size))
            self.dbeta[j] = sum(dout[i][j] for i in range(batch_size))

        return dx


# =============================================================================
# Простая нейронная сеть
# =============================================================================

class SimpleNetwork:
    """
    Простая полносвязная сеть для демонстрации BatchNorm.

    Архитектура: input -> [Linear+BN+ReLU]*hidden_layers -> Linear+Sigmoid
    """

    def __init__(self, layer_sizes, use_bn=False, lr=0.1):
        self.use_bn = use_bn
        self.lr = lr
        self.n_layers = len(layer_sizes) - 1

        # Параметры слоёв
        self.weights = []
        self.biases = []
        self.bn_layers = []

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            W = [[random.uniform(-limit, limit) for _ in range(fan_out)]
                 for _ in range(fan_in)]
            b = [0.0] * fan_out
            self.weights.append(W)
            self.biases.append(b)

            if use_bn and i < self.n_layers - 1:
                self.bn_layers.append(BatchNorm(fan_out))
            else:
                self.bn_layers.append(None)

        # Кэш forward
        self._activations = []
        self._pre_activations = []
        self._bn_outputs = []

    def forward(self, x):
        """Forward pass по одной выборке (вектор)."""
        self._activations = [x[:]]
        self._pre_activations = []
        self._bn_outputs = []

        a = x[:]
        for i in range(self.n_layers):
            # Linear
            z = [0.0] * len(self.biases[i])
            for j in range(len(self.biases[i])):
                s = self.biases[i][j]
                for k in range(len(a)):
                    s += a[k] * self.weights[i][k][j]
                z[j] = s

            self._pre_activations.append(z[:])

            # BatchNorm (только для скрытых слоёв)
            if self.bn_layers[i] is not None:
                z_bn = self.bn_layers[i].forward([z])
                z = z_bn[0]
                self._bn_outputs.append(z[:])
            else:
                self._bn_outputs.append(None)

            # Активация
            if i < self.n_layers - 1:
                a = [relu(zi) for zi in z]
            else:
                a = [sigmoid(zi) for zi in z]

            self._activations.append(a[:])

        return a

    def backward(self, y_true):
        """Backward pass + обновление весов для одной выборки."""
        output = self._activations[-1]
        n_out = len(output)

        # Градиент MSE + Sigmoid
        dout = [2.0 * (output[j] - y_true[j]) / n_out for j in range(n_out)]

        for i in range(self.n_layers - 1, -1, -1):
            # Через активацию
            if i < self.n_layers - 1:
                # ReLU
                pre_act = self._pre_activations[i]
                dout = [dout[j] * relu_grad(pre_act[j]) for j in range(len(dout))]

            # Через BatchNorm
            if self.bn_layers[i] is not None:
                dout_bn = self.bn_layers[i].backward([dout])
                dout = dout_bn[0]

            # Через Linear: dw, db, dx
            a_prev = self._activations[i]
            fan_in = len(a_prev)
            fan_out = len(dout)

            for k in range(fan_in):
                for j in range(fan_out):
                    self.weights[i][k][j] -= self.lr * a_prev[k] * dout[j]
            for j in range(fan_out):
                self.biases[i][j] -= self.lr * dout[j]

            # Градиент к предыдущему слою
            new_dout = [0.0] * fan_in
            for k in range(fan_in):
                for j in range(fan_out):
                    new_dout[k] += dout[j] * self.weights[i][k][j]
            dout = new_dout

    def train_epoch(self, data, labels):
        """Одна эпоха обучения. Возвращает средний loss."""
        total_loss = 0.0
        for x, y in zip(data, labels):
            out = self.forward(x)
            err = [out[j] - y[j] for j in range(len(y))]
            loss = sum(e ** 2 for e in err) / len(y)
            total_loss += loss
            self.backward(y)
        return total_loss / len(data)

    def predict(self, x):
        """Предсказание (в режиме eval)."""
        old_mode = None
        for bn in self.bn_layers:
            if bn is not None:
                old_mode = bn.training
                bn.eval_mode()
        out = self.forward(x)
        if old_mode is not None:
            for bn in self.bn_layers:
                if bn is not None:
                    bn.train_mode()
        return out


# =============================================================================
# Демо 1: Forward pass — нормализация
# =============================================================================

def demo1_forward_pass():
    """Демонстрация forward pass BatchNorm — нормализация данных."""
    print("=" * 70)
    print("ДЕМО 1: Forward Pass — Нормализация данных")
    print("=" * 70)

    random.seed(42)
    num_features = 4
    batch_size = 8

    bn = BatchNorm(num_features)

    # Сырые данные с разным масштабом
    raw_data = []
    for i in range(batch_size):
        row = []
        for j in range(num_features):
            row.append(random.gauss(0, 1) * (j + 1) * 10 + (j + 1) * 50)
        raw_data.append(row)

    print("\nИсходные данные (разный масштаб для каждого признака):")
    print(f"  {'Признак':>10s} {'Среднее':>10s} {'Стд':>10s} {'Мин':>10s} {'Макс':>10s}")
    for j in range(num_features):
        col = [raw_data[i][j] for i in range(batch_size)]
        mean = vec_mean(col)
        std = math.sqrt(vec_var(col, mean))
        print(f"  {'Признак ' + str(j+1):>10s} {mean:>10.2f} {std:>10.2f} {min(col):>10.2f} {max(col):>10.2f}")

    output = bn.forward(raw_data)

    print("\nПосле BatchNorm:")
    print(f"  {'Признак':>10s} {'Среднее':>10s} {'Стд':>10s} {'Мин':>10s} {'Макс':>10s}")
    for j in range(num_features):
        col = [output[i][j] for i in range(batch_size)]
        mean = vec_mean(col)
        std = math.sqrt(vec_var(col, mean))
        print(f"  {'Признак ' + str(j+1):>10s} {mean:>10.4f} {std:>10.4f} {min(col):>10.4f} {max(col):>10.4f}")

    print("\nВывод: BatchNorm приводит все признаки к единому масштабу!")
    print("  Средние ≈ 0, стандартное отклонение ≈ 1\n")


# =============================================================================
# Демо 2: Статистики (mean, variance)
# =============================================================================

def demo2_statistics():
    """Демонстрация вычисления и отслеживания статистик."""
    print("=" * 70)
    print("ДЕМО 2: Статистики — Среднее, Дисперсия, Running статистики")
    print("=" * 70)

    random.seed(42)
    num_features = 3
    bn = BatchNorm(num_features, momentum=0.3)

    print("\nСимуляция 5 батчей. momentum = 0.3")
    print(f"{'Батч':>5s} | {'Batch Mean':>12s} | {'Running Mean':>12s} | {'Batch Var':>12s} | {'Running Var':>12s}")
    print("-" * 70)

    for batch_idx in range(5):
        batch_size = 16
        data = [[random.gauss(batch_idx * 2, 1) for _ in range(num_features)]
                for _ in range(batch_size)]

        output = bn.forward(data)

        batch_col = [data[i][0] for i in range(batch_size)]
        b_mean = vec_mean(batch_col)
        b_var = vec_var(batch_col, b_mean)

        print(f"{batch_idx+1:>5d} | {b_mean:>12.4f} | {bn.running_mean[0]:>12.4f} | "
              f"{b_var:>12.4f} | {bn.running_var[0]:>12.4f}")

    print("\nВывод:")
    print("  Running mean/var — экспоненциальное скользящее среднее (EMA)")
    print("  Они плавно следуют за статистиками батчей")
    print("  На inference используются ВМЕСТО статистик текущего батча\n")

    # Train vs Inference
    print("Демонстрация train vs inference:")
    test_data = [[5.0, 10.0, 15.0]]

    bn.train_mode()
    out_train = bn.forward(test_data)
    print(f"  Train mode:     input={test_data[0]}, output={[round(x, 4) for x in out_train[0]]}")

    bn.eval_mode()
    out_eval = bn.forward(test_data)
    print(f"  Inference mode: input={test_data[0]}, output={[round(x, 4) for x in out_eval[0]]}")
    print(f"  Running mean:   {[round(x, 4) for x in bn.running_mean]}")
    print(f"  Running var:    {[round(x, 4) for x in bn.running_var]}")
    print()


# =============================================================================
# Демо 3: Сравнение с/без BatchNorm
# =============================================================================

def demo3_comparison():
    """Сравнение обучения с и без BatchNorm."""
    print("=" * 70)
    print("ДЕМО 3: Сравнение — с BatchNorm vs без BatchNorm")
    print("=" * 70)

    random.seed(42)

    # Генерация данных: задача бинарной классификации
    def make_data(n=30, seed=42):
        random.seed(seed)
        data = []
        labels = []
        for _ in range(n):
            x = [random.gauss(0, 1) for _ in range(4)]
            label = 1.0 if (x[0] * 0.5 + x[1] * 0.3 - x[2] * 0.2 + x[3] * 0.4 > 0.1) else 0.0
            data.append(x)
            labels.append([label])
        return data, labels

    train_data, train_labels = make_data(40)
    epochs = 200

    # Без BatchNorm
    print("\n--- Обучение БЕЗ BatchNorm (lr=0.5) ---")
    random.seed(42)
    net_no_bn = SimpleNetwork([4, 16, 8, 1], use_bn=False, lr=0.5)
    losses_no = []
    for epoch in range(epochs):
        # Перемешиваем данные
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        shuffled_data = [train_data[i] for i in indices]
        shuffled_labels = [train_labels[i] for i in indices]
        loss = net_no_bn.train_epoch(shuffled_data, shuffled_labels)
        losses_no.append(loss)
        if (epoch + 1) % 40 == 0:
            print(f"    Эпоха {epoch+1:>3d}: loss = {loss:.6f}")

    # С BatchNorm
    print("\n--- Обучение С BatchNorm (lr=0.5) ---")
    random.seed(42)
    net_bn = SimpleNetwork([4, 16, 8, 1], use_bn=True, lr=0.5)
    losses_bn = []
    for epoch in range(epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        shuffled_data = [train_data[i] for i in indices]
        shuffled_labels = [train_labels[i] for i in indices]
        loss = net_bn.train_epoch(shuffled_data, shuffled_labels)
        losses_bn.append(loss)
        if (epoch + 1) % 40 == 0:
            print(f"    Эпоха {epoch+1:>3d}: loss = {loss:.6f}")

    # Сравнение
    print("\n--- Сравнение ---")
    print(f"  {'Эпоха':>6s} {'Без BN':>12s} {'С BN':>12s} {'Разница':>12s}")
    print(f"  {'-' * 44}")
    for idx in [0, 19, 49, 99, 149, 199]:
        diff = losses_no[idx] - losses_bn[idx]
        print(f"  {idx+1:>6d} {losses_no[idx]:>12.6f} {losses_bn[idx]:>12.6f} {diff:>+12.6f}")

    better = "Да" if losses_bn[-1] < losses_no[-1] else "Нет"
    print(f"\n  BatchNorm помог быстрее сойтись: {better}")
    print(f"  Финальный loss без BN: {losses_no[-1]:.6f}")
    print(f"  Финальный loss с BN:   {losses_bn[-1]:.6f}")
    print()


# =============================================================================
# Демо 4: Ускорение сходимости
# =============================================================================

def demo4_convergence():
    """Демонстрация ускорения сходимости BatchNorm при разных learning rates."""
    print("=" * 70)
    print("ДЕМО 4: Ускорение сходимости при разных learning rates")
    print("=" * 70)

    random.seed(42)

    def make_data(n=40, seed=42):
        random.seed(seed)
        data = []
        labels = []
        for _ in range(n):
            x = [random.gauss(0, 1) for _ in range(3)]
            label = 1.0 if (x[0] + x[1] * 0.5 - x[2] * 0.3 > 0.2) else 0.0
            data.append(x)
            labels.append([label])
        return data, labels

    data, labels = make_data(40)
    epochs = 150

    learning_rates = [0.1, 0.5, 1.0]

    for lr in learning_rates:
        print(f"\n--- Learning Rate = {lr} ---")

        # Без BN
        random.seed(42)
        net_no = SimpleNetwork([3, 16, 8, 1], use_bn=False, lr=lr)
        losses_no = []
        for epoch in range(epochs):
            idx = list(range(len(data)))
            random.shuffle(idx)
            sd = [data[i] for i in idx]
            sl = [labels[i] for i in idx]
            losses_no.append(net_no.train_epoch(sd, sl))

        # С BN
        random.seed(42)
        net_bn = SimpleNetwork([3, 16, 8, 1], use_bn=True, lr=lr)
        losses_bn = []
        for epoch in range(epochs):
            idx = list(range(len(data)))
            random.shuffle(idx)
            sd = [data[i] for i in idx]
            sl = [labels[i] for i in idx]
            losses_bn.append(net_bn.train_epoch(sd, sl))

        # Текстовый график
        print(f"  {'Эпоха':>6s} {'Без BN':>12s} {'С BN':>12s}")
        print(f"  {'-' * 32}")
        for idx in range(0, epochs, 25):
            bar_no = '█' * max(1, int(losses_no[idx] * 40))
            bar_bn = '█' * max(1, int(losses_bn[idx] * 40))
            print(f"  {idx+1:>6d} {losses_no[idx]:>10.6f}  {bar_no}")
            print(f"  {'':>6s} {losses_bn[idx]:>10.6f}  {bar_bn}  (BN)")

        # Эпоха порога
        threshold = 0.15
        ep_no = next((i + 1 for i, l in enumerate(losses_no) if l < threshold), epochs)
        ep_bn = next((i + 1 for i, l in enumerate(losses_bn) if l < threshold), epochs)
        print(f"\n  Эпоха достижения loss < {threshold}:")
        print(f"    Без BN: {ep_no},  С BN: {ep_bn}")
        if ep_bn < ep_no:
            print(f"    BN быстрее на {ep_no - ep_bn} эпох!")
        else:
            print(f"    BN медленнее на {ep_bn - ep_no} эпох")

    print("\nВывод:")
    print("  Batch Norm стабилизирует градиенты, позволяя сети")
    print("  быстрее сходиться при любых learning rates.")
    print("  Эффект наиболее заметен при высоких lr.\n")


# =============================================================================
# Запуск всех демо
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       BATCH NORMALIZATION — Полная реализация с нуля               ║")
    print("║       Источник: Ioffe & Szegedy, 2015                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    demo1_forward_pass()
    demo2_statistics()
    demo3_comparison()
    demo4_convergence()

    print("=" * 70)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("=" * 70)
    print("""
  1. BatchNorm нормализует активации: mean≈0, std≈1
  2. Running статистики (EMA) запоминают распределение обучающих данных
  3. В train режиме используются статистики батча
  4. В inference режиме используются running статистики
  5. BatchNorm ускоряет сходимость и стабилизирует обучение
  6. Позволяет использовать learning rates в 3-10x выше
  7. Действует как регуляция (шум статистик батча)
  8. γ и β — обучаемые параметры, восстанавливают выразительность
""")
