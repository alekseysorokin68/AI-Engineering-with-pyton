"""183 — Continual Learning: катастрофическое забывание, EWC, прогрессивные сети

Темы:
  1. Catastrophic Forgetting — concept drift, task interference, replay buffers
  2. Elastic Weight Consolidation — Fisher information, penalty terms, importance weights
  3. Progressive Networks — lateral connections, column expansion, capacity management
  4. Knowledge Distillation — teacher-student, soft targets, feature matching

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import statistics

random.seed(42)

# ========================================================================
# Общие классы и функции (модульный уровень)
# ========================================================================

class LinearClassifier:
    """Простой линейный классификатор на чистом Python."""
    def __init__(self, input_dim, lr=0.01):
        self.weights = [0.0] * input_dim
        self.bias = 0.0
        self.lr = lr

    def predict(self, x):
        """Скалярное произведение + смещение."""
        s = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, s))))  # sigmoid

    def train_step(self, x, target):
        """Один шаг градиентного спуска на бинарной кросс-энтропии."""
        pred = self.predict(x)
        error = pred - target
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * error * x[i]
        self.bias -= self.lr * error
        return -target * math.log(pred + 1e-10) - (1 - target) * math.log(1 - pred + 1e-10)

    def accuracy(self, data, labels):
        """Вычисление точности на наборе данных."""
        correct = sum(1 for x, y in zip(data, labels) if (self.predict(x) > 0.5) == y)
        return correct / len(data)


def generate_task(n, seed, offset=0.0):
    """Генерация линейно разделимого набора данных."""
    random.seed(seed)
    data = []
    labels = []
    for _ in range(n):
        x = [random.gauss(offset, 1.0) for _ in range(4)]
        label = 1 if sum(x) > 0 else 0
        data.append(x)
        labels.append(label)
    return data, labels


# ========================================================================
# 1. Catastrophic Forgetting
# ========================================================================

def demo_catastrophic_forgetting():
    """Демонстрация катастрофического забывания и техник смягчения."""
    print("=" * 70)
    print("1. КАТАСТРОФИЧЕСКОЕ ЗАБЫВАНИЕ")
    print("=" * 70)

    # --- 1a. Модель простого линейного классификатора ---
    print("\n--- 1a. Линейный классификатор и task interference ---\n")

    # Task A: положительное смещение, Task B: отрицательное смещение
    train_a, labels_a = generate_task(200, seed=1, offset=0.8)
    train_b, labels_b = generate_task(200, seed=2, offset=-0.8)
    test_a, test_labels_a = generate_task(100, seed=3, offset=0.8)
    test_b, test_labels_b = generate_task(100, seed=4, offset=-0.8)

    # Обучаем на Task A
    model = LinearClassifier(4, lr=0.05)
    loss_history = []
    for epoch in range(50):
        epoch_loss = 0.0
        for x, y in zip(train_a, labels_a):
            epoch_loss += model.train_step(x, y)
        loss_history.append(epoch_loss / len(train_a))

    acc_a_after_a = model.accuracy(test_a, test_labels_a)
    acc_b_after_a = model.accuracy(test_b, test_labels_b)
    print(f"После обучения на Task A:")
    print(f"  Точность на Task A: {acc_a_after_a:.3f}")
    print(f"  Точность на Task B: {acc_b_after_a:.3f} (ещё не изучали)")
    print(f"  Начальный loss: {loss_history[0]:.4f} → финальный: {loss_history[-1]:.4f}")

    # Теперь обучаем на Task B — классическое забывание!
    for epoch in range(50):
        for x, y in zip(train_b, labels_b):
            model.train_step(x, y)

    acc_a_after_b = model.accuracy(test_a, test_labels_a)
    acc_b_after_b = model.accuracy(test_b, test_labels_b)
    print(f"\nПосле обучения на Task B (забыли Task A!):")
    print(f"  Точность на Task A: {acc_a_after_b:.3f} ← СНИЗИЛАСЬ!")
    print(f"  Точность на Task B: {acc_b_after_b:.3f}")
    print(f"\n  Формула: ΔAcc_A = {acc_a_after_a:.3f} → {acc_a_after_b:.3f} "
          f"(потеря: {acc_a_after_a - acc_a_after_b:.3f})")

    # --- 1b. Replay Buffer ---
    print("\n--- 1b. Replay Buffer — повторное использование старых данных ---\n")

    def train_with_replay(model, new_data, new_labels, old_data, old_labels,
                          replay_ratio=0.3, epochs=50):
        """Обучение с буфером повторного воспроизведения."""
        random.seed(42)
        for _ in range(epochs):
            for x, y in zip(new_data, new_labels):
                model.train_step(x, y)
                # С вероятностью replay_ratio повторяем старый пример
                if random.random() < replay_ratio and old_data:
                    idx = random.randint(0, len(old_data) - 1)
                    model.train_step(old_data[idx], old_labels[idx])

    model_replay = LinearClassifier(4, lr=0.05)
    # Сначала обучаем на Task A
    for _ in range(50):
        for x, y in zip(train_a, labels_a):
            model_replay.train_step(x, y)

    # Сохраняем буфер из Task A
    buffer_indices = random.sample(range(len(train_a)), min(40, len(train_a)))
    replay_buffer = [train_a[i] for i in buffer_indices]
    replay_labels = [labels_a[i] for i in buffer_indices]

    # Обучаем на Task B с replay buffer
    train_with_replay(model_replay, train_b, labels_b,
                      replay_buffer, replay_labels, replay_ratio=0.3, epochs=50)

    acc_a_replay = model_replay.accuracy(test_a, test_labels_a)
    acc_b_replay = model_replay.accuracy(test_b, test_labels_b)
    print(f"  Replay Buffer (размер={len(replay_buffer)}):")
    print(f"  Точность на Task A: {acc_a_replay:.3f} (было {acc_a_after_b:.3f} без replay)")
    print(f"  Точность на Task B: {acc_b_replay:.3f}")
    print(f"  Улучшение на Task A: +{acc_a_replay - acc_a_after_b:.3f}")

    # --- 1c. Concept Drift — обнаружение сдвига ---
    print("\n--- 1c. Обнаружение concept drift ---\n")

    def detect_drift(scores, window=20, threshold=0.15):
        """Простой скользящий детектор дрифта по среднему скользящему."""
        drift_points = []
        for i in range(window, len(scores)):
            prev = statistics.mean(scores[i - window:i])
            curr = statistics.mean(scores[i:i + window]) if i + window <= len(scores) else statistics.mean(scores[i:])
            if abs(curr - prev) > threshold:
                drift_points.append(i)
        return drift_points

    # Симуляция: сначала высокая точность, потом резкий спад
    random.seed(42)
    accuracy_stream = [0.85 + random.gauss(0, 0.02) for _ in range(50)]
    accuracy_stream += [0.85 - 0.3 + random.gauss(0, 0.02) for _ in range(50)]  # дрифт!
    accuracy_stream += [0.55 + random.gauss(0, 0.02) for _ in range(50)]       # стабилизация

    drifts = detect_drift(accuracy_stream, window=15, threshold=0.1)
    print(f"  Длина потока метрик: {len(accuracy_stream)}")
    print(f"  Начальная точность: {statistics.mean(accuracy_stream[:15]):.3f}")
    print(f"  Обнаруженные точки дрифта: {drifts[:5]}...")
    print(f"  Формула: |μ_recent - μ_prev| > threshold ({0.1})")

    # --- 1d. Multi-task replay с ядром памяти ---
    print("\n--- 1d. Ядро памяти (episodic memory) для 3 задач ---\n")

    class EpisodicMemory:
        """Простое ядро памяти с ограничением по ёмкости."""
        def __init__(self, capacity=50):
            self.capacity = capacity
            self.buffer = []
            self.labels_buf = []

        def add(self, x, y):
            """Добавление примера с заменой по стратегии reservoir sampling."""
            if len(self.buffer) < self.capacity:
                self.buffer.append(x)
                self.labels_buf.append(y)
            else:
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = x
                self.labels_buf[idx] = y

        def sample(self, n):
            """Случайный выбор n примеров из буфера."""
            n = min(n, len(self.buffer))
            indices = random.sample(range(len(self.buffer)), n)
            return [self.buffer[i] for i in indices], [self.labels_buf[i] for i in indices]

    memory = EpisodicMemory(capacity=30)
    tasks_data = [
        generate_task(100, seed=10, offset=1.0),
        generate_task(100, seed=11, offset=-0.5),
        generate_task(100, seed=12, offset=0.3),
    ]

    task_names = ["A (+1.0)", "B (-0.5)", "C (+0.3)"]
    mem_model = LinearClassifier(4, lr=0.05)
    task_accs = []

    for t_idx, ((t_data, t_labels), t_name) in enumerate(zip(tasks_data, task_names)):
        # Заполняем ядро памяти из текущей задачи
        for x, y in zip(t_data[:20], t_labels[:20]):
            memory.add(x, y)

        # Обучаем на текущей задаче + выборка из памяти
        for _ in range(30):
            for x, y in zip(t_data, t_labels):
                mem_model.train_step(x, y)
            mem_x, mem_y = memory.sample(10)
            for mx, my in zip(mem_x, mem_y):
                mem_model.train_step(mx, my)

        # Проверяем на ВСЕХ задачах
        accs = []
        for test_d, test_l in tasks_data:
            accs.append(mem_model.accuracy(test_d, test_l))
        task_accs.append(accs)
        print(f"  После Task {t_name}: {[f'{a:.3f}' for a in accs]}")

    print(f"\n  Финальная матрица точности (строки = после задачи, столбцы = тест):")
    for i, (row, name) in enumerate(zip(task_accs, task_names)):
        print(f"    После {name}: {[f'{v:.3f}' for v in row]}")
    print(f"  Ядро памяти: capacity={memory.capacity}, "
          f"заполнено={len(memory.buffer)}")


# ========================================================================
# 2. Elastic Weight Consolidation (EWC)
# ========================================================================

def demo_ewc():
    """Демонстрация метода EWC для защиты важных весов."""
    print("\n" + "=" * 70)
    print("2. ELASTIC WEIGHT CONSOLIDATION (EWC)")
    print("=" * 70)

    # --- 2a. Fisher Information Matrix ---
    print("\n--- 2a. Информация Фишера ---\n")

    def compute_fisher(model, data, labels):
        """Вычисление диагональной матрицы Фишера для линейной модели."""
        fisher = [0.0] * len(model.weights)
        for x, y in zip(data, labels):
            pred = model.predict(x)
            # Производная sigmoid: σ'(z) = σ(z)(1 - σ(z))
            grad_scale = pred * (1 - pred)
            for i in range(len(fisher)):
                fisher[i] += (grad_scale * x[i]) ** 2
        # Усредняем по числу примеров
        n = len(data)
        return [f / n for f in fisher]

    # Обучаем модель на Task A и вычисляем Fisher
    fisher_model = LinearClassifier(4, lr=0.05)
    train_a, labels_a = generate_task(200, seed=1, offset=0.8)
    test_a, test_labels_a = generate_task(100, seed=3, offset=0.8)
    train_b, labels_b = generate_task(200, seed=2, offset=-0.8)
    test_b, test_labels_b = generate_task(100, seed=4, offset=-0.8)

    for _ in range(50):
        for x, y in zip(train_a, labels_a):
            fisher_model.train_step(x, y)

    # Сохраняем оптимальные веса Task A
    optimal_weights = fisher_model.weights[:]
    optimal_bias = fisher_model.bias

    # Вычисляем информацию Фишера
    fisher = compute_fisher(fisher_model, train_a, labels_a)
    print(f"  Веса Task A (оптимальные): {[f'{w:.4f}' for w in optimal_weights]}")
    print(f"  Информация Фишера:        {[f'{f:.4f}' for f in fisher]}")
    print(f"\n  Интерпретация: большие значения Fisher → вес важен для Task A")
    print(f"  Формула: F_i = E[ (∂L/∂w_i)² ] — ожидание квадрата градиента")

    # --- 2b. EWC Penalty ---
    print("\n--- 2b. Штраф EWC при обучении на Task B ---\n")

    def ewc_penalty(current_weights, optimal_weights, fisher):
        """Вычисление штрафа EWC: Σ F_i * (w_i - w*_i)²."""
        penalty = 0.0
        for cw, ow, f in zip(current_weights, optimal_weights, fisher):
            penalty += f * (cw - ow) ** 2
        return penalty

    # Без EWC: просто обучаем на Task B
    model_no_ewc = LinearClassifier(4, lr=0.05)
    for w_idx, w in enumerate(optimal_weights):
        model_no_ewc.weights[w_idx] = w
    model_no_ewc.bias = optimal_bias

    for _ in range(50):
        for x, y in zip(train_b, labels_b):
            model_no_ewc.train_step(x, y)

    # С EWC: добавляем штраф
    model_ewc = LinearClassifier(4, lr=0.05)
    for w_idx, w in enumerate(optimal_weights):
        model_ewc.weights[w_idx] = w
    model_ewc.bias = optimal_bias

    lambda_ewc = 100.0  # коэффициент важности старой задачи

    for _ in range(50):
        for x, y in zip(train_b, labels_b):
            # Стандартный градиент
            pred = model_ewc.predict(x)
            error = pred - y
            lr = model_ewc.lr
            for i in range(len(model_ewc.weights)):
                # Градиент задачи + EWC штраф
                grad_task = error * x[i]
                grad_ewc = 2 * lambda_ewc * fisher[i] * (model_ewc.weights[i] - optimal_weights[i])
                model_ewc.weights[i] -= lr * (grad_task + grad_ewc)
            model_ewc.bias -= lr * error

    penalty_no_ewc = ewc_penalty(model_no_ewc.weights, optimal_weights, fisher)
    penalty_ewc = ewc_penalty(model_ewc.weights, optimal_weights, fisher)

    acc_a_no_ewc = model_no_ewc.accuracy(test_a, test_labels_a)
    acc_a_ewc = model_ewc.accuracy(test_a, test_labels_a)
    acc_b_no_ewc = model_no_ewc.accuracy(test_b, test_labels_b)
    acc_b_ewc = model_ewc.accuracy(test_b, test_labels_b)

    print(f"  λ (коэффициент EWC) = {lambda_ewc}")
    print(f"\n  Без EWC:")
    print(f"    Штраф: {penalty_no_ewc:.4f}")
    print(f"    Task A: {acc_a_no_ewc:.3f}, Task B: {acc_b_no_ewc:.3f}")
    print(f"\n  С EWC:")
    print(f"    Штраф: {penalty_ewc:.4f}")
    print(f"    Task A: {acc_a_ewc:.3f}, Task B: {acc_b_ewc:.3f}")
    print(f"\n  Формула EWC: L = L_task + λ/2 · Σ F_i·(w_i - w*_i)²")

    # --- 2c. Влияние λ ---
    print("\n--- 2c. Анализ влияния λ на баланс задач ---\n")

    lambdas = [0, 1, 10, 50, 100, 500, 1000]
    results = []

    for lam in lambdas:
        m = LinearClassifier(4, lr=0.05)
        for i, w in enumerate(optimal_weights):
            m.weights[i] = w
        m.bias = optimal_bias

        for _ in range(50):
            for x, y in zip(train_b, labels_b):
                pred = m.predict(x)
                error = pred - y
                lr = m.lr
                for i in range(len(m.weights)):
                    grad_task = error * x[i]
                    grad_ewc = 2 * lam * fisher[i] * (m.weights[i] - optimal_weights[i])
                    m.weights[i] -= lr * (grad_task + grad_ewc)
                m.bias -= lr * error

        a_acc = m.accuracy(test_a, test_labels_a)
        b_acc = m.accuracy(test_b, test_labels_b)
        results.append((lam, a_acc, b_acc))

    print(f"  {'λ':>8} | {'Task A':>8} | {'Task B':>8} | {'Средняя':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for lam, a_acc, b_acc in results:
        avg = (a_acc + b_acc) / 2
        print(f"  {lam:>8} | {a_acc:>8.3f} | {b_acc:>8.3f} | {avg:>8.3f}")

    print(f"\n  Вывод: λ → 0 = забываем Task A; λ → ∞ = не учим Task B")
    print(f"  Оптимальный баланс: λ ≈ {100}")

    # --- 2d. Мульти-сессионное EWC ---
    print("\n--- 2d. Последовательное обучение 4 задач с EWC ---\n")

    tasks_multi = [
        generate_task(100, seed=20, offset=2.0),
        generate_task(100, seed=21, offset=-1.5),
        generate_task(100, seed=22, offset=0.5),
        generate_task(100, seed=23, offset=-0.3),
    ]
    test_multi = [
        generate_task(50, seed=30, offset=2.0),
        generate_task(50, seed=31, offset=-1.5),
        generate_task(50, seed=32, offset=0.5),
        generate_task(50, seed=33, offset=-0.3),
    ]

    # Накопленный EWC: храним оптимальные веса и Fisher для каждой задачи
    all_optimal = []
    all_fisher = []

    multi_model = LinearClassifier(4, lr=0.05)
    multi_results = []

    for t_idx, (t_data, t_labels) in enumerate(tasks_multi):
        # Обучаем на текущей задаче с EWC от ВСЕХ предыдущих
        for _ in range(40):
            for x, y in zip(t_data, t_labels):
                pred = multi_model.predict(x)
                error = pred - y
                lr = multi_model.lr
                for i in range(len(multi_model.weights)):
                    grad_task = error * x[i]
                    # Суммируем EWC штрафы от всех предыдущих задач
                    grad_ewc_total = 0.0
                    for opt_w, f_vals in zip(all_optimal, all_fisher):
                        grad_ewc_total += 2 * lambda_ewc * f_vals[i] * (multi_model.weights[i] - opt_w[i])
                    multi_model.weights[i] -= lr * (grad_task + grad_ewc_total)
                multi_model.bias -= lr * error

        # Сохраняем оптимальные веса и Fisher для текущей задачи
        all_optimal.append(multi_model.weights[:])
        all_fisher.append(compute_fisher(multi_model, t_data, t_labels))

        # Тестируем на всех задачах
        accs = []
        for test_d, test_l in test_multi:
            accs.append(multi_model.accuracy(test_d, test_l))
        multi_results.append(accs)
        print(f"  После Task {t_idx + 1}: {[f'{a:.3f}' for a in accs]}")

    print(f"\n  Матрица забывания (Task A → Task D):")
    for i, row in enumerate(multi_results):
        print(f"    Task {i + 1}: {[f'{v:.3f}' for v in row]}")


# ========================================================================
# 3. Progressive Networks
# ========================================================================

def demo_progressive_networks():
    """Демонстрация прогрессивных сетей с наращиванием столбцов."""
    print("\n" + "=" * 70)
    print("3. PROGRESSIVE NETWORKS")
    print("=" * 70)

    # --- 3a. Базовая колонка (столбец) ---
    print("\n--- 3a. Колонка — один столбец прогрессивной сети ---\n")

    class Column:
        """Один столбец прогрессивной сети."""
        def __init__(self, input_dim, hidden_dim, output_dim, column_id):
            self.column_id = column_id
            random.seed(42 + column_id)
            # Инициализация весов Xavier
            self.W_hidden = [[random.gauss(0, math.sqrt(2.0 / input_dim))
                              for _ in range(hidden_dim)] for _ in range(input_dim)]
            # W_output: hidden_dim → output_dim (вектор для скалярного выхода)
            self.W_output = [random.gauss(0, math.sqrt(2.0 / hidden_dim))
                             for _ in range(hidden_dim)]
            self.b_hidden = [0.0] * hidden_dim
            self.b_output = 0.0
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

        def forward(self, x):
            """Прямой проход через колонку."""
            hidden = []
            for j in range(self.hidden_dim):
                s = sum(x[i] * self.W_hidden[i][j] for i in range(min(len(x), self.input_dim))) + self.b_hidden[j]
                hidden.append(max(0, s))  # ReLU
            out = sum(hidden[j] * self.W_output[j] for j in range(self.hidden_dim)) + self.b_output
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, out)))), hidden

    col = Column(4, 8, 1, column_id=0)
    test_x = [0.5, -0.3, 0.8, 0.1]
    pred, hidden = col.forward(test_x)
    print(f"  Колонка 0: input_dim={col.input_dim}, hidden={col.hidden_dim}, output={col.output_dim}")
    print(f"  Вход: {[f'{v:.2f}' for v in test_x]}")
    print(f"  Скрытое представление: {[f'{h:.3f}' for h in hidden]}")
    print(f"  Выход (sigmoid): {pred:.4f}")

    # --- 3b. Боковые соединения ---
    print("\n--- 3b. Боковые (lateral) соединения между столбцами ---\n")

    class ProgressiveNetwork:
        """Прогрессивная сеть с боковыми соединениями."""
        def __init__(self, input_dim):
            self.columns = []
            self.lateral_weights = []
            self.input_dim = input_dim

        def add_column(self, hidden_dim, output_dim):
            """Добавление нового столбца с боковыми соединениями."""
            col_id = len(self.columns)
            prev_output_dim = self.input_dim

            if self.columns:
                prev_output_dim = self.columns[-1].hidden_dim

            col = Column(prev_output_dim, hidden_dim, output_dim, col_id)

            # Боковые веса: от скрытого слоя каждого предыдущего столбца
            lateral = []
            for prev_col in self.columns:
                l_w = [random.gauss(0, math.sqrt(2.0 / prev_col.hidden_dim))
                       for _ in range(hidden_dim)]
                lateral.append(l_w)
            self.lateral_weights.append(lateral)
            self.columns.append(col)
            return col

        def forward(self, x):
            """Прямой проход через все столбцы."""
            all_hidden = []
            current_input = x

            for idx, col in enumerate(self.columns):
                # Суммируем вход от предыдущего столбца + боковые соединения
                enhanced_input = list(current_input)
                for prev_idx in range(idx):
                    l_w = self.lateral_weights[idx][prev_idx]
                    for j in range(col.hidden_dim):
                        if j < len(enhanced_input):
                            enhanced_input[j] += all_hidden[prev_idx][j] * l_w[j] * 0.1
                        else:
                            enhanced_input.append(all_hidden[prev_idx][j] * l_w[j] * 0.1)

                out, hidden = col.forward(enhanced_input[:col.input_dim])
                all_hidden.append(hidden)

            return out

    pnet = ProgressiveNetwork(4)
    pnet.add_column(hidden_dim=6, output_dim=1)   # Столбец 1
    pnet.add_column(hidden_dim=6, output_dim=1)   # Столбец 2
    pnet.add_column(hidden_dim=6, output_dim=1)   # Столбец 3

    test_vectors = [[0.5, -0.3, 0.8, 0.1],
                    [-0.2, 0.7, 0.1, -0.5],
                    [0.9, 0.9, 0.9, 0.9]]

    print(f"  Столбцов: {len(pnet.columns)}")
    for i, x in enumerate(test_vectors):
        out = pnet.forward(x)
        print(f"  Вход {i + 1}: {[f'{v:.2f}' for v in x]} → выход: {out:.4f}")

    # --- 3c. Управление ёмкостью ---
    print("\n--- 3c. Управление ёмкостью столбцов ---\n")

    def parameter_budget(num_columns, hidden_dims):
        """Подсчёт общего числа параметров."""
        total = 0
        details = []
        input_dim = 4
        for i, h in enumerate(hidden_dims):
            # Параметры столбца
            col_params = input_dim * h + h + h + 1  # W_hidden, b_hidden, W_output, b_output
            # Боковые параметры
            lateral_params = i * h
            total += col_params + lateral_params
            details.append(f"Столбец {i}: {col_params} (колонка) + {lateral_params} (боковые) = {col_params + lateral_params}")
            input_dim = h
        return total, details

    configs = [
        [8, 8, 8],
        [16, 16, 16],
        [8, 12, 16],
        [16, 8, 4],
    ]

    print(f"  {'Конфигурация':<25} | {'Параметры':>10}")
    print(f"  {'-'*25}-+-{'-'*10}")
    for config in configs:
        total, _ = parameter_budget(len(config), config)
        config_str = " → ".join(str(c) for c in config)
        print(f"  {config_str:<25} | {total:>10}")

    # --- 3d. Frozen columns ---
    print("\n--- 3d. Замороженные столбцы и дообучение ---\n")

    class FrozenProgressiveNet:
        """Прогрессивная сеть с замороженными старыми столбцами."""
        def __init__(self):
            self.frozen_columns = []
            self.active_column = None
            self.train_steps_log = []

        def freeze_column(self, column):
            """Заморозить столбец (замерзнуть = не обновлять веса)."""
            frozen = {
                'weights': [row[:] for row in column.W_hidden],
                'output': column.W_output[:],
                'hidden_dim': column.hidden_dim,
                'frozen': True
            }
            self.frozen_columns.append(frozen)
            print(f"  Заморожен столбец с hidden_dim={column.hidden_dim}")

        def compute_trainable_params(self):
            """Подсчёт обучаемых параметров."""
            active = 0
            if self.active_column:
                c = self.active_column
                active = c.input_dim * c.hidden_dim + c.hidden_dim + c.hidden_dim + 1
            frozen = sum(
                col['hidden_dim'] * 4 + col['hidden_dim'] + col['hidden_dim'] + 1
                for col in self.frozen_columns
            )
            return active, frozen

    fpn = FrozenProgressiveNet()
    col1 = Column(4, 8, 1, 0)
    col2 = Column(8, 8, 1, 1)
    col3 = Column(8, 8, 1, 2)

    fpn.freeze_column(col1)
    active, frozen = fpn.compute_trainable_params()
    print(f"  Активных: {active}, замороженных: {frozen}, "
          f"ratio: {active / (active + frozen):.1%}")

    fpn.active_column = col2
    active, frozen = fpn.compute_trainable_params()
    print(f"\n  После добавления столбца 2:")
    print(f"  Активных: {active}, замороженных: {frozen}, "
          f"ratio: {active / (active + frozen):.1%}")

    fpn.freeze_column(col2)
    fpn.active_column = col3
    active, frozen = fpn.compute_trainable_params()
    print(f"\n  После добавления столбца 3:")
    print(f"  Активных: {active}, замороженных: {frozen}, "
          f"ratio: {active / (active + frozen):.1%}")


# ========================================================================
# 4. Knowledge Distillation
# ========================================================================

def demo_knowledge_distillation():
    """Демонстрация дистилляции знаний: teacher → student."""
    print("\n" + "=" * 70)
    print("4. KNOWLEDGE DISTILLATION")
    print("=" * 70)

    # --- 4a. Teacher и Student модели ---
    print("\n--- 4a. Teacher (большая) и Student (маленькая) модели ---\n")

    class TeacherModel:
        """Большая модель с 2 скрытыми слоями."""
        def __init__(self):
            random.seed(42)
            # Слой 1: 4 → 12
            self.W1 = [[random.gauss(0, 0.3) for _ in range(12)] for _ in range(4)]
            self.b1 = [0.0] * 12
            # Слой 2: 12 → 8
            self.W2 = [[random.gauss(0, 0.3) for _ in range(8)] for _ in range(12)]
            self.b2 = [0.0] * 8
            # Выход: 8 → 1
            self.W3 = [random.gauss(0, 0.3) for _ in range(8)]
            self.b3 = 0.0

        def forward(self, x):
            """Прямой проход с возвратом промежуточных представлений."""
            # Слой 1
            h1 = []
            for j in range(12):
                s = sum(x[i] * self.W1[i][j] for i in range(4)) + self.b1[j]
                h1.append(max(0, s))  # ReLU
            # Слой 2
            h2 = []
            for j in range(8):
                s = sum(h1[i] * self.W2[i][j] for i in range(12)) + self.b2[j]
                h2.append(max(0, s))
            # Выход
            out = sum(h2[j] * self.W3[j] for j in range(8)) + self.b3
            sigm = 1.0 / (1.0 + math.exp(-max(-500, min(500, out))))
            return sigm, h1, h2

    class StudentModel:
        """Маленькая модель с 1 скрытым слоем."""
        def __init__(self):
            random.seed(100)
            self.W1 = [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(4)]
            self.b1 = [0.0] * 4
            self.W2 = [random.gauss(0, 0.5) for _ in range(4)]
            self.b2 = 0.0
            self.lr = 0.01

        def forward(self, x):
            """Прямой проход с возвратом скрытого слоя."""
            h = []
            for j in range(4):
                s = sum(x[i] * self.W1[i][j] for i in range(4)) + self.b1[j]
                h.append(max(0, s))
            out = sum(h[j] * self.W2[j] for j in range(4)) + self.b2
            sigm = 1.0 / (1.0 + math.exp(-max(-500, min(500, out))))
            return sigm, h

    teacher = TeacherModel()
    student = StudentModel()

    params_teacher = 4 * 12 + 12 + 12 * 8 + 8 + 8 + 1  # W1, b1, W2, b2, W3, b3
    params_student = 4 * 4 + 4 + 4 + 1

    test_x = [0.5, -0.3, 0.8, 0.1]
    t_out, t_h1, t_h2 = teacher.forward(test_x)
    s_out, s_h = student.forward(test_x)

    print(f"  Teacher: 4 → 12 → 8 → 1 ({params_teacher} параметров)")
    print(f"  Student: 4 → 4 → 1 ({params_student} параметров)")
    print(f"  Коэффициент сжатия: {params_teacher / params_student:.1f}x")
    print(f"\n  Вход: {[f'{v:.2f}' for v in test_x]}")
    print(f"  Teacher выход: {t_out:.4f}")
    print(f"  Student выход: {s_out:.4f}")
    print(f"  Teacher hidden[0:4]: {[f'{h:.3f}' for h in t_h2[:4]]}")
    print(f"  Student hidden: {[f'{h:.3f}' for h in s_h]}")

    # --- 4b. Soft Targets с температурой ---
    print("\n--- 4b. Soft targets и температура ---\n")

    def softmax_temperature(logits, temperature):
        """Softmax с температурой T."""
        scaled = [logit / temperature for logit in logits]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    # Симуляция логитов для 5 классов
    logits = [2.0, 1.0, 0.5, 0.1, -1.0]
    class_names = ["Кот", "Собака", "Птица", "Рыба", "Змея"]

    print(f"  Логиты: {logits}")
    print(f"  {'T':>5} | ", end="")
    for name in class_names:
        print(f"{name:>8}", end=" | ")
    print()

    for temp in [1, 2, 5, 10]:
        probs = softmax_temperature(logits, temp)
        print(f"  {temp:>5} | ", end="")
        for p in probs:
            print(f"{p:>8.4f}", end=" | ")
        print()

    print(f"\n  Формула: p_i = exp(z_i / T) / Σ_j exp(z_j / T)")
    print(f"  T=1: стандартный softmax (жёсткие цели)")
    print(f"  T→∞: равномерное распределение (мягкие цели)")
    print(f"  Мягкие цели несут информацию о «похожести» классов")

    # --- 4c. Дистилляция: комбинированный loss ---
    print("\n--- 4c. Обучение student с дистилляцией ---\n")

    def distillation_loss(student_pred, teacher_pred, true_label, T=3.0, alpha=0.5):
        """Потеря дистилляции: alpha * soft_loss + (1-alpha) * hard_loss."""
        # Hard loss: бинарная кросс-энтропия
        hard = -true_label * math.log(student_pred + 1e-10) - \
               (1 - true_label) * math.log(1 - student_pred + 1e-10)

        # Soft loss: KL-расстояние (аппроксимация для бинарного случая)
        s_soft = student_pred / T
        t_soft = teacher_pred / T
        soft = t_soft * math.log(t_soft / (s_soft + 1e-10) + 1e-10) + \
               (1 - t_soft) * math.log((1 - t_soft) / (1 - s_soft + 1e-10) + 1e-10)

        return alpha * T * T * soft + (1 - alpha) * hard

    # Генерируем данные
    random.seed(42)
    train_data = [(random.gauss(0, 1), 1 if random.random() > 0.5 else 0) for _ in range(200)]

    def train_student(student, teacher, data, T=3.0, alpha=0.5, epochs=30):
        """Обучение student с дистилляцией."""
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for x_val, y in data:
                x = [x_val, x_val ** 2, x_val ** 3, x_val * 0.5]  # расширение признаков
                t_pred, _, _ = teacher.forward(x)
                s_pred, _ = student.forward(x)

                loss = distillation_loss(s_pred, t_pred, y, T=T, alpha=alpha)
                epoch_loss += loss

                # Градиентный спуск (упрощённый)
                error_s = s_pred - (alpha * t_pred + (1 - alpha) * y)
                for i in range(len(student.W1)):
                    for j in range(len(student.W1[i])):
                        student.W1[i][j] -= student.lr * error_s * x[i]
                for j in range(len(student.W2)):
                    student.W2[j] -= student.lr * error_s * student.W1[j % len(student.W1)][0]
            losses.append(epoch_loss / len(data))
        return losses

    student_distill = StudentModel()
    losses = train_student(student_distill, teacher, train_data, T=3.0, alpha=0.5, epochs=30)

    # Тестирование
    correct_distill = 0
    for x_val, y in train_data:
        x = [x_val, x_val ** 2, x_val ** 3, x_val * 0.5]
        s_pred, _ = student_distill.forward(x)
        if (s_pred > 0.5) == y:
            correct_distill += 1

    print(f"  T=3.0, α=0.5 (50% soft + 50% hard)")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Точность: {correct_distill}/{len(train_data)} = "
          f"{correct_distill / len(train_data):.3f}")
    print(f"\n  Формула: L = α·T²·KL(p_s^T ‖ p_t^T) + (1-α)·H(y, p_s)")

    # --- 4d. Feature Matching ---
    print("\n--- 4d. Feature Matching — совпадение представлений ---\n")

    def feature_matching_loss(student_hidden, teacher_hidden):
        """L2 расстояние между средними представлениями скрытых слоёв."""
        # Усредняем по скрытым нейронам
        s_mean = statistics.mean(student_hidden) if student_hidden else 0
        t_mean = statistics.mean(teacher_hidden) if teacher_hidden else 0
        return (s_mean - t_mean) ** 2

    random.seed(42)
    test_inputs = [[random.gauss(0, 1) for _ in range(4)] for _ in range(50)]

    # Teacher представления
    teacher_features = []
    for x in test_inputs:
        _, _, t_h2 = teacher.forward(x)
        teacher_features.append(t_h2)

    # Student представления (до и после feature matching)
    student_before_features = []
    student_after_features = []
    student_fm = StudentModel()

    for x in test_inputs:
        _, s_h = student_fm.forward(x)
        student_before_features.append(s_h)

    # Простая корректировка через feature matching
    t_means = [statistics.mean([f[i] for f in teacher_features])
               for i in range(len(teacher_features[0]))]
    s_means = [statistics.mean([f[i] for f in student_before_features])
               for i in range(len(student_before_features[0]))]

    # Сдвигаем веса student для лучшего совпадения
    correction = [(t - s) * 0.01 for t, s in zip(t_means, s_means)]
    for i in range(len(student_fm.W2)):
        student_fm.W2[i] += correction[i % len(correction)]

    for x in test_inputs:
        _, s_h = student_fm.forward(x)
        student_after_features.append(s_h)

    # Сравниваем
    loss_before = 0.0
    loss_after = 0.0
    for sf, tf in zip(student_before_features, teacher_features):
        for i in range(min(len(sf), len(tf))):
            loss_before += (sf[i] - tf[i]) ** 2
            loss_after += (sf[i] - tf[i]) ** 2
    loss_before /= len(test_inputs)
    loss_after /= len(test_inputs)

    print(f"  Teacher features (среднее по 50 примеров):")
    print(f"    {[f'{m:.3f}' for m in t_means]}")
    print(f"\n  Student features ДО feature matching:")
    print(f"    {[f'{m:.3f}' for m in s_means]}")
    print(f"    MSE → Teacher: {loss_before:.4f}")

    s_means_after = [statistics.mean([f[i] for f in student_after_features])
                     for i in range(len(student_after_features[0]))]
    print(f"\n  Student features ПОСЛЕ feature matching:")
    print(f"    {[f'{m:.3f}' for m in s_means_after]}")
    print(f"    MSE → Teacher: {loss_after:.4f}")
    print(f"\n  Формула: L_FM = ||E[h_student] - E[h_teacher]||²")
    print(f"  Feature matching заставляет student学习中期表示, не только выход")


# ========================================================================
# Точка входа
# ========================================================================

if __name__ == "__main__":
    demo_catastrophic_forgetting()
    demo_ewc()
    demo_progressive_networks()
    demo_knowledge_distillation()
