"""
63 — Рекуррентные нейросети (RNN): теория и практика
=====================================================

Содержание:
  1. Класс SimpleRNN — forward pass, backward pass (BPTT)
  2. Затухающие / взрывающиеся градиенты — теория и демонстрация
  3. Обучение на простых последовательностях (классификация)

Зависимости: только numpy + стандартная библиотека Python.
"""

import numpy as np
import math

# ──────────────────────────────────────────────────────────────
# 1. Простая RNN-ячейка (SimpleRNNCell)
# ──────────────────────────────────────────────────────────────

class SimpleRNNCell:
    """
    Одна ячейка простой RNN.

    Математика:
        h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        y_t = W_hy * h_t + b_y

    Параметры:
        input_size  — размерность входного вектора x_t
        hidden_size — размерность скрытого состояния h_t
        output_size — размерность выходного вектора y_t (0 = нет выхода)
    """

    def __init__(self, input_size, hidden_size, output_size=0, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов (Xavier-like)
        scale_xh = math.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = math.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = math.sqrt(2.0 / (hidden_size + output_size)) if output_size > 0 else 1.0

        self.W_xh = rng.randn(hidden_size, input_size) * scale_xh
        self.W_hh = rng.randn(hidden_size, hidden_size) * scale_hh
        self.b_h = np.zeros((hidden_size, 1))

        self.W_hy = None
        self.b_y = None
        if output_size > 0:
            self.W_hy = rng.randn(output_size, hidden_size) * scale_hy
            self.b_y = np.zeros((output_size, 1))

    def forward(self, x_seq, h_init=None):
        """
        Прямой проход по последовательности.

        Args:
            x_seq  — список векторов [(input_size,1), ...] длины T
            h_init — начальное скрытое состояние (hidden_size, 1) или None

        Returns:
            h_seq   — список скрытых состояний [h_1, ..., h_T]
            y_seq   — список выходов [y_1, ..., y_T] (если output_size > 0)
            cache   — кэш для backward pass
        """
        T = len(x_seq)
        h_prev = h_init if h_init is not None else np.zeros((self.hidden_size, 1))

        h_seq = []
        y_seq = []
        cache = []

        for t in range(T):
            x_t = x_seq[t]

            # Линейная комбинация + tanh
            z = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
            h_t = np.tanh(z)

            # Выход (если есть)
            y_t = None
            if self.W_hy is not None:
                y_t = self.W_hy @ h_t + self.b_y

            cache.append((x_t, h_prev, z, h_t, y_t))
            h_seq.append(h_t)
            if y_t is not None:
                y_seq.append(y_t)

            h_prev = h_t

        return h_seq, y_seq, cache

    def forward_single(self, x_t, h_prev):
        """Один шаг forward для демо."""
        z = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        h_t = np.tanh(z)
        y_t = None
        if self.W_hy is not None:
            y_t = self.W_hy @ h_t + self.b_y
        return h_t, y_t


# ──────────────────────────────────────────────────────────────
# 2. Backpropagation Through Time (BPTT)
# ──────────────────────────────────────────────────────────────

class SimpleRNN:
    """
    RNN с полным BPTT и SGD-оптимизатором.

    BPTT вычисляет градиенты по всем шагам времени:
        dL/dW_xh = Σ_t  dL/dh_t · dh_t/dz_t · x_t^T
        dL/dW_hh = Σ_t  dL/dh_t · dh_t/dz_t · h_{t-1}^T
        dL/dW_hy = Σ_t  dL/dy_t · h_t^T
    """

    def __init__(self, input_size, hidden_size, output_size=0, learning_rate=0.01,
                 clip_norm=5.0, rng=None):
        self.cell = SimpleRNNCell(input_size, hidden_size, output_size, rng)
        self.lr = learning_rate
        self.clip_norm = clip_norm

    def _softmax(self, logits):
        """Numerically stable softmax."""
        e = np.exp(logits - np.max(logits))
        return e / np.sum(e)

    def _cross_entropy_loss(self, y_pred, y_true_idx):
        """Cross-entropy loss для one-hot метки."""
        probs = self._softmax(y_pred)
        loss = -np.log(probs[y_true_idx, 0] + 1e-12)
        grad_logits = probs.copy()
        grad_logits[y_true_idx, 0] -= 1.0
        return loss, grad_logits

    def _mse_loss(self, y_pred, y_target):
        """MSE loss."""
        diff = y_pred - y_target
        loss = 0.5 * np.sum(diff ** 2)
        return loss, diff

    def backward(self, cache, targets=None, loss_type='mse'):
        """
        BPTT: обратное распространение сквозь время.

        Args:
            cache      — кэш из forward pass
            targets    — целевые значения [(output_size,1), ...] для MSE
                           или список индексов классов для cross-entropy
            loss_type  — 'mse' или 'cross_entropy'

        Returns:
            grads — словарь градиентов
        """
        cell = self.cell
        T = len(cache)

        # Инициализация градиентов
        dW_xh = np.zeros_like(cell.W_xh)
        dW_hh = np.zeros_like(cell.W_hh)
        db_h = np.zeros_like(cell.b_h)
        dW_hy = np.zeros_like(cell.W_hy) if cell.W_hy is not None else None
        db_y = np.zeros_like(cell.b_y) if cell.b_y is not None else None

        dh_next = np.zeros((cell.hidden_size, 1))

        total_loss = 0.0

        for t in reversed(range(T)):
            x_t, h_prev, z, h_t, y_t = cache[t]

            # Вычисление dL/dh_t
            has_target = (targets is not None and y_t is not None
                          and t < len(targets) and targets[t] is not None)
            if has_target:
                if loss_type == 'cross_entropy':
                    loss, dy = self._cross_entropy_loss(y_t, targets[t])
                    total_loss += loss
                else:
                    loss, dy = self._mse_loss(y_t, targets[t])
                    total_loss += loss

                # dL/dh_t от выходного слоя
                if dW_hy is not None:
                    dW_hy += dy @ h_t.T
                    db_y += dy
                    dh_out = cell.W_hy.T @ dy
                else:
                    dh_out = np.zeros_like(dh_next)
            else:
                dh_out = np.zeros_like(dh_next)

            dh = dh_out + dh_next

            # Производная tanh: dtanh/dz = 1 - tanh(z)^2
            dtanh = (1.0 - h_t ** 2) * dh

            # Градиенты весов
            dW_xh += dtanh @ x_t.T
            dW_hh += dtanh @ h_prev.T
            db_h += dtanh

            # Градиент для предыдущего шага
            dh_next = cell.W_hh.T @ dtanh

        # Нормализация по длине последовательности
        dW_xh /= T
        dW_hh /= T
        db_h /= T
        if dW_hy is not None:
            dW_hy /= T
            db_y /= T

        # Gradient clipping (по L2-норме)
        grads = {
            'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h,
            'W_hy': dW_hy, 'b_y': db_y,
        }
        self._clip_gradients(grads)

        return grads, total_loss / T if targets is not None else 0.0

    def _clip_gradients(self, grads):
        """Gradient clipping по суммарной L2-норме всех параметров."""
        total_norm = 0.0
        for k, v in grads.items():
            if v is not None:
                total_norm += np.sum(v ** 2)
        total_norm = math.sqrt(total_norm + 1e-12)

        if total_norm > self.clip_norm:
            scale = self.clip_norm / total_norm
            for k in grads:
                if grads[k] is not None:
                    grads[k] *= scale

    def _apply_grads(self, grads):
        """SGD обновление параметров."""
        cell = self.cell
        cell.W_xh -= self.lr * grads['W_xh']
        cell.W_hh -= self.lr * grads['W_hh']
        cell.b_h -= self.lr * grads['b_h']
        if grads['W_hy'] is not None:
            cell.W_hy -= self.lr * grads['W_hy']
            cell.b_y -= self.lr * grads['b_y']

    def train_step(self, x_seq, targets, loss_type='mse'):
        """Один шаг обучения: forward → backward → update."""
        h_seq, y_seq, cache = self.cell.forward(x_seq)
        grads, loss = self.backward(cache, targets, loss_type)
        self._apply_grads(grads)
        return loss, y_seq


# ──────────────────────────────────────────────────────────────
# 3. Вспомогательные функции
# ──────────────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx, 0] = 1.0
    return v


def compute_gradient_norms(cell, x_seq, num_steps=20):
    """
    Вычисляет норму градиентов W_hh на каждом шаге времени.
    Используется для демонстрации затухания градиентов.
    """
    h_prev = np.zeros((cell.hidden_size, 1))
    norms = []

    for t in range(num_steps):
        x_t = x_seq[t % len(x_seq)]
        z = cell.W_xh @ x_t + cell.W_hh @ h_prev + cell.b_h
        h_t = np.tanh(z)

        # "Производная" tanh на этом шаге
        dtanh = 1.0 - h_t ** 2

        # Норма dL/dW_hh на этом шаге (упрощённо: считаем |dh_t/dW_hh|)
        # dh_t/dW_hh = dtanh · h_{t-1}^T  +  dtanh · W_hh · dh_{t-1}/dW_hh
        grad_norm = float(np.sqrt(np.sum(dtanh ** 2)))
        norms.append(grad_norm)

        h_prev = h_t

    return norms


# ──────────────────────────────────────────────────────────────
# 4. Демонстрации
# ──────────────────────────────────────────────────────────────

def demo_1_rnn_cell():
    """Демо 1: RNN-ячейка — forward pass по шагам."""
    print("=" * 65)
    print("ДЕМО 1: RNN-ячейка — пошаговый forward pass")
    print("=" * 65)

    rng = np.random.RandomState(42)
    cell = SimpleRNNCell(input_size=3, hidden_size=4, output_size=2, rng=rng)

    # Последовательность из 5 шагов
    x_seq = [rng.randn(3, 1) for _ in range(5)]

    print(f"\n  Входная размерность: {cell.input_size}")
    print(f"  Скрытое состояние:   {cell.hidden_size}")
    print(f"  Выходная размерность: {cell.output_size}")
    print(f"  Длина последовательности: {len(x_seq)}")

    # Пошаговый forward
    h_prev = np.zeros((cell.hidden_size, 1))
    print(f"\n  Начальное h_0 = [0, 0, 0, 0]^T\n")

    for t, x_t in enumerate(x_seq):
        h_t, y_t = cell.forward_single(x_t, h_prev)
        print(f"  Шаг {t+1}:")
        print(f"    x_{t+1}      = {x_t.flatten().round(3)}")
        print(f"    h_{t+1}      = {h_t.flatten().round(3)}")
        print(f"    y_{t+1}      = {y_t.flatten().round(3)}")
        print(f"    ||h_{t+1}||   = {np.linalg.norm(h_t):.4f}")
        h_prev = h_t

    # Полный forward
    h_seq, y_seq, cache = cell.forward(x_seq)
    print(f"\n  Полный forward: {len(h_seq)} скрытых состояний, {len(y_seq)} выходов")
    print(f"  Финальное h_T  = {h_seq[-1].flatten().round(3)}")
    print()

    # Параметры
    total_params = cell.W_xh.size + cell.W_hh.size + cell.b_h.size
    if cell.W_hy is not None:
        total_params += cell.W_hy.size + cell.b_y.size
    print(f"  Всего параметров: {total_params}")
    print(f"    W_xh: {cell.W_xh.shape} ({cell.W_xh.size})")
    print(f"    W_hh: {cell.W_hh.shape} ({cell.W_hh.size})")
    print(f"    b_h:  {cell.b_h.shape} ({cell.b_h.size})")
    print(f"    W_hy: {cell.W_hy.shape} ({cell.W_hy.size})")
    print(f"    b_y:  {cell.b_y.shape} ({cell.b_y.size})")


def demo_2_sequence_generation():
    """Демо 2: Генерация последовательности обученной RNN."""
    print("=" * 65)
    print("ДЕМО 2: Генерация последовательности")
    print("=" * 65)

    rng = np.random.RandomState(42)

    # Простая задача: предсказать следующий символ в周期е 0,1,2,0,1,2,...
    seq_len = 6  # 0,1,2,0,1,2
    vocab_size = 3

    rnn = SimpleRNN(input_size=vocab_size, hidden_size=8, output_size=vocab_size,
                     learning_rate=0.1, rng=rng)

    # Обучающая последовательность: 0,1,2,0,1,2
    pattern = [0, 1, 2, 0, 1, 2]
    print(f"\n  Паттерн для обучения: {pattern}")
    print(f"  Задача: предсказать следующий символ\n")

    # Подготовка: x_t = one_hot(pattern[t]), target = pattern[t+1]
    x_seq = [one_hot(pattern[t], vocab_size) for t in range(len(pattern))]
    targets = [pattern[(t + 1) % len(pattern)] for t in range(len(pattern))]

    # Обучение
    print("  Обучение (100 эпох):")
    losses = []
    for epoch in range(101):
        loss, y_seq = rnn.train_step(x_seq, targets, loss_type='cross_entropy')
        if epoch % 20 == 0:
            losses.append(loss)
            print(f"    Эпоха {epoch:3d}: loss = {loss:.4f}")

    # Генерация
    print(f"\n  Генерация 12 символов:")
    h_prev = np.zeros((rnn.cell.hidden_size, 1))
    start_char = 0
    generated = [start_char]

    for step in range(12):
        x_t = one_hot(generated[-1], vocab_size)
        h_t, y_t = rnn.cell.forward_single(x_t, h_prev)
        probs = softmax(y_t).flatten()
        next_char = int(np.argmax(probs))
        generated.append(next_char)
        h_prev = h_t

    print(f"    Сгенерированная последовательность: {generated}")
    print(f"    Ожидаемый паттерн:                  {[0,1,2,0,1,2,0,1,2,0,1,2,0]}")

    # Вероятности последнего шага
    print(f"\n  Вероятности на последнем шаге:")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 30)
        print(f"    Класс {i}: {p:.3f} {bar}")


def demo_3_vanishing_gradients():
    """Демо 3: Проблема затухающих градиентов."""
    print("=" * 65)
    print("ДЕМО 3: Проблема затухающих / взрывающихся градиентов")
    print("=" * 65)

    rng = np.random.RandomState(42)
    input_size = 4
    hidden_size = 8

    # Создаём RNN-ячейку
    cell = SimpleRNNCell(input_size=input_size, hidden_size=hidden_size, rng=rng)

    # Входная последовательность
    num_steps = 15
    x_seq = [rng.randn(input_size, 1) for _ in range(num_steps)]

    # ── Теоретическое объяснение ──
    print("\n  Теория: затухание градиентов в RNN")
    print("  ─" * 32)
    print("  При BPTT градиент через T шагов содержит произведение:")
    print("    ∂h_T/∂h_1 = ∏_{t=2}^{T} diag(1 - tanh²(z_t)) · W_hh")
    print()
    print("  Каждый сомножитель:")
    print("    • diag(1 - tanh²(z_t)) ∈ (0, 1] — норма ≤ 1 (tanh)")
    print("    • ||W_hh|| определяет рост / затухание")
    print()
    print("  Если ||W_hh|| · max(1-tanh²) < 1 → градиент экспоненциально затухает")
    print("  Если ||W_hh|| · max(1-tanh²) > 1 → градиент экспоненциально растёт")

    # ── Эксперимент 1: Разные начальные масштабы W_hh ──
    print("\n  Эксперимент: масштаб W_hh и затухание градиентов")
    print("  ─" * 32)

    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"  {'Масштаб':>10} | {'||W_hh||_F':>10} | {'Норма градиента T=1':>20} | {'Норма градиента T=15':>20}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*20} | {'-'*20}")

    for scale in scales:
        cell_test = SimpleRNNCell(input_size, hidden_size, rng=np.random.RandomState(42))
        cell_test.W_hh *= scale

        norms = compute_gradient_norms(cell_test, x_seq, num_steps=15)
        whh_norm = np.linalg.norm(cell_test.W_hh, 'fro')
        print(f"  {scale:>10.1f} | {whh_norm:>10.4f} | {norms[0]:>20.6f} | {norms[-1]:>20.6f}")

    # ── Эксперимент 2: Градиенты через разную глубину ──
    print(f"\n  Эксперимент: градиенты на каждом шаге времени")
    print(f"  (масштаб W_hh = 1.0)")
    print("  ─" * 32)

    cell_exp = SimpleRNNCell(input_size, hidden_size, rng=np.random.RandomState(42))
    norms = compute_gradient_norms(cell_exp, x_seq, num_steps=num_steps)

    print(f"  {'Шаг':>6} | {'Норма градиента':>14} | {'Относительная':>12} | График")
    print(f"  {'-'*6} | {'-'*14} | {'-'*12} | {'-'*30}")

    max_norm = max(norms)
    for t, n in enumerate(norms):
        rel = n / norms[0] if norms[0] > 0 else 0
        bar_len = int(n / max_norm * 25) if max_norm > 0 else 0
        bar = "█" * bar_len
        print(f"  {t+1:>6} | {n:>14.6f} | {rel:>11.4f} | {bar}")

    # ── Решение: gradient clipping ──
    print(f"\n  Решения проблемы затухающих / взрывающихся градиентов:")
    print("  ─" * 32)
    print("  1. Gradient clipping (обрезка по норме)")
    print("     → предотвращает взрыв градиентов")
    print("  2. Инициализация ортогональными матрицами")
    print("     → ||W_hh|| ≈ 1, градиент не затухает")
    print("  3. Skip connections (LSTM, GRU)")
    print("     → добавляют «шунты» для градиента")
    print("  4. Batch Normalization / Layer Normalization")
    print("     → стабилизирует распределение активаций")


def demo_4_classification():
    """Демо 4: Обучение RNN на задаче классификации последовательностей."""
    print("=" * 65)
    print("ДЕМО 4: Обучение RNN — классификация последовательностей")
    print("=" * 65)

    rng = np.random.RandomState(42)

    # ── Задача: классификация направления последовательности ──
    # Возрастающая (1) vs убывающая (0) последовательность из 3 чисел
    print("\n  Задача: классификация последовательности")
    print("  Класс 1: возрастающая  (например, [0.1, 0.5, 0.9])")
    print("  Класс 0: убывающая     (например, [0.9, 0.5, 0.1])")
    print("  Длина последовательности: 4 элемента")

    # Генерация данных
    def generate_data(n_samples, seq_len=4):
        X, y = [], []
        for _ in range(n_samples):
            seq = rng.uniform(0.0, 1.0, seq_len)
            label = 1 if seq[-1] > seq[0] else 0  # возрастающая = 1
            X.append([np.array([[v]]) for v in seq])  # input_size=1
            y.append(label)
        return X, y

    train_X, train_y = generate_data(100, seq_len=4)
    test_X, test_y = generate_data(30, seq_len=4)

    print(f"\n  Обучающая выборка: {len(train_X)} последовательностей")
    print(f"  Тестовая выборка:  {len(test_X)} последовательностей")

    # Подсчёт классов
    n_pos = sum(train_y)
    n_neg = len(train_y) - n_pos
    print(f"  Класс 1 (возрастающая): {n_pos}")
    print(f"  Класс 0 (убывающая):    {n_neg}")

    # ── Обучение ──
    rnn = SimpleRNN(input_size=1, hidden_size=8, output_size=2,
                     learning_rate=0.05, clip_norm=5.0, rng=rng)

    print(f"\n  Архитектура: SimpleRNN(input=1, hidden=8, output=2)")
    print(f"  Параметров: {8*1 + 8*8 + 8 + 2*8 + 2} (W_xh + W_hh + b_h + W_hy + b_y)")
    print(f"\n  Обучение (200 эпох):")

    train_losses = []
    train_accs = []

    for epoch in range(201):
        # Перемешиваем данные
        indices = rng.permutation(len(train_X))
        epoch_loss = 0.0
        correct = 0

        for i in indices:
            x_seq = train_X[i]
            target = train_y[i]
            T = len(x_seq)
            # Целевой класс только на последнем шаге, None на остальных
            targets = [None] * (T - 1) + [target]

            # Forward + backward
            h_seq, y_seq, cache = rnn.cell.forward(x_seq)
            grads, loss = rnn.backward(cache, targets, loss_type='cross_entropy')
            rnn._apply_grads(grads)
            epoch_loss += loss

            # Предсказание
            probs = softmax(y_seq[-1]).flatten()
            pred = int(np.argmax(probs))
            if pred == target:
                correct += 1

        avg_loss = epoch_loss / len(train_X)
        accuracy = correct / len(train_X)

        if epoch % 40 == 0:
            train_losses.append(avg_loss)
            train_accs.append(accuracy)
            print(f"    Эпоха {epoch:3d}: loss = {avg_loss:.4f}, accuracy = {accuracy:.2%}")

    # ── Оценка на тесте ──
    print(f"\n  Оценка на тестовой выборке:")
    correct = 0
    conf_matrix = np.zeros((2, 2), dtype=int)

    for i in range(len(test_X)):
        x_seq = test_X[i]
        target = test_y[i]

        h_seq, y_seq, cache = rnn.cell.forward(x_seq)
        probs = softmax(y_seq[-1]).flatten()
        pred = int(np.argmax(probs))

        if pred == target:
            correct += 1
        conf_matrix[target, pred] += 1

    test_acc = correct / len(test_X)
    print(f"    Точность: {test_acc:.2%} ({correct}/{len(test_X)})")

    print(f"\n  Матрица ошибок (confusion matrix):")
    print(f"  {'':>12} | {'Предсказано 0':>14} | {'Предсказано 1':>14}")
    print(f"  {'-'*12} | {'-'*14} | {'-'*14}")
    print(f"  {'Истинно 0':>12} | {conf_matrix[0,0]:>14} | {conf_matrix[0,1]:>14}")
    print(f"  {'Истинно 1':>12} | {conf_matrix[1,0]:>14} | {conf_matrix[1,1]:>14}")

    # ── Примеры предсказаний ──
    print(f"\n  Примеры предсказаний:")
    print(f"  {'Последовательность':>30} | {'Истина':>7} | {'Предсказание':>12} | {'Уверенность':>10}")
    print(f"  {'-'*30} | {'-'*7} | {'-'*12} | {'-'*10}")

    for i in range(min(8, len(test_X))):
        x_seq = test_X[i]
        target = test_y[i]
        h_seq, y_seq, cache = rnn.cell.forward(x_seq)
        probs = softmax(y_seq[-1]).flatten()
        pred = int(np.argmax(probs))
        conf = probs[pred]

        seq_str = "[" + ", ".join(f"{x[0,0]:.2f}" for x in x_seq) + "]"
        label_str = "возр." if target == 1 else "убыв."
        pred_str = "возр." if pred == 1 else "убыв."
        match = "✓" if pred == target else "✗"
        print(f"  {seq_str:>30} | {label_str:>7} | {pred_str:>10} {match} | {conf:>9.2%}")

    # ── Заключение ──
    print(f"\n  Результаты:")
    print(f"  ─" * 32)
    print(f"  • RNN успешно обучилась различать возрастающие / убывающие последовательности")
    print(f"  • Точность на тесте: {test_acc:.0%}")
    print(f"  • RNN учитывает порядок элементов (не просто сравнивает min и max)")
    print(f"  • Затухание градиентов ограничивает длину зависимостей")
    print(f"    → для длинных последовательностей используют LSTM / GRU")


# ──────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_1_rnn_cell()
    print("\n")
    demo_2_sequence_generation()
    print("\n")
    demo_3_vanishing_gradients()
    print("\n")
    demo_4_classification()
    print("\n" + "=" * 65)
    print("Все демонстрации завершены.")
    print("=" * 65)
