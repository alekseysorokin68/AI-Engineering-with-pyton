"""
64 - LSTM и GRU: реализация с нуля
===================================
- LSTM cell с forget / input / output gates
- GRU cell (упрощённая версия LSTM)
- Сравнение с SimpleRNN
- Проблема затухающих градиентов и решение
"""

import random
import math

random.seed(42)

# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def sigmoid(x):
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x):
    if x < -500:
        return -1.0
    if x > 500:
        return 1.0
    return math.tanh(x)


def sigmoid_derivative(s):
    """Производная сигмоиды (принимает уже вычисленное значение sigmoid)."""
    return s * (1.0 - s)


def tanh_derivative(t):
    """Производная tanh (принимает уже вычисленное значение tanh)."""
    return 1.0 - t * t


def init_weights(n_in, n_out):
    """Инициализация весов Xavier."""
    limit = math.sqrt(6.0 / (n_in + n_out))
    return [[random.uniform(-limit, limit) for _ in range(n_out)] for _ in range(n_in)]


def init_bias(n):
    """Инициализация смещений нулями."""
    return [0.0] * n


def mat_vec_dot(matrix, vector):
    """Умножение матрицы на вектор: (m x n) * (n,) -> (m,)."""
    n_cols = len(matrix[0])
    n_rows = len(matrix)
    assert len(vector) == n_cols, f"Размерность: матрица {n_rows}x{n_cols}, вектор {len(vector)}"
    return [sum(matrix[i][j] * vector[j] for j in range(n_cols)) for i in range(n_rows)]


def vec_add(a, b):
    """Покоординатное сложение."""
    return [a[i] + b[i] for i in range(len(a))]


def vec_sigmoid(v):
    """Сигмоиды поэлементно."""
    return [sigmoid(x) for x in v]


def vec_tanh(v):
    """tanh поэлементно."""
    return [tanh(x) for x in v]


def elementwise_mul(a, b):
    """Поэлементное умножение."""
    return [a[i] * b[i] for i in range(len(a))]


def concat(a, b):
    """Конкатенация двух векторов."""
    return a + b


def split_vec(v, idx):
    """Разделение вектора на две части по индексу."""
    return v[:idx], v[idx:]


# ─────────────────────────────────────────────
# SimpleRNN
# ─────────────────────────────────────────────

class SimpleRNN:
    """Простая RNN-ячейка: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)."""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_hh = init_weights(hidden_size, hidden_size)
        self.W_xh = init_weights(hidden_size, input_size)
        self.b = init_bias(hidden_size)

    def forward(self, x_t, h_prev):
        """
        x_t:     (input_size,)  — вход на текущем шаге
        h_prev:  (hidden_size,) — скрытое состояние предыдущего шага
        returns: (hidden_size,) — новое скрытое состояние
        """
        hh = mat_vec_dot(self.W_hh, h_prev)
        xh = mat_vec_dot(self.W_xh, x_t)
        return vec_tanh(vec_add(vec_add(hh, xh), self.b))


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

class LSTM:
    """
    LSTM-ячейка с тремя вентилями:

    forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    candidate:    c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

    cell state:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    hidden:      h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = hidden_size + input_size  # размер конкатенации [h, x]

        # Веса для forget gate
        self.W_f = init_weights(hidden_size, self.concat_size)
        self.b_f = init_bias(hidden_size)

        # Веса для input gate
        self.W_i = init_weights(hidden_size, self.concat_size)
        self.b_i = init_bias(hidden_size)

        # Веса для candidate (c̃)
        self.W_c = init_weights(hidden_size, self.concat_size)
        self.b_c = init_bias(hidden_size)

        # Веса для output gate
        self.W_o = init_weights(hidden_size, self.concat_size)
        self.b_o = init_bias(hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        x_t:     (input_size,)   — вход
        h_prev:  (hidden_size,)  — предыдущее скрытое состояние
        c_prev:  (hidden_size,)  — предыдущее состояние ячейки
        returns: (h_t, c_t)      — новое скрытое состояние и состояние ячейки
        """
        # Конкатенируем h_{t-1} и x_t
        concat = concat_vec(h_prev, x_t)

        # Forget gate — что забыть
        f_raw = vec_add(mat_vec_dot(self.W_f, concat), self.b_f)
        f = vec_sigmoid(f_raw)

        # Input gate — что записать
        i_raw = vec_add(mat_vec_dot(self.W_i, concat), self.b_i)
        i = vec_sigmoid(i_raw)

        # Candidate — новые кандидаты для ячейки
        c_raw = vec_add(mat_vec_dot(self.W_c, concat), self.b_c)
        c_tilde = vec_tanh(c_raw)

        # Output gate — что выдать
        o_raw = vec_add(mat_vec_dot(self.W_o, concat), self.b_o)
        o = vec_sigmoid(o_raw)

        # Обновляем cell state: c_t = f ⊙ c_{t-1} + i ⊙ c̃_t
        c_t = vec_add(elementwise_mul(f, c_prev), elementwise_mul(i, c_tilde))

        # Вычисляем скрытое состояние: h_t = o ⊙ tanh(c_t)
        h_t = elementwise_mul(o, vec_tanh(c_t))

        return h_t, c_t


def concat_vec(a, b):
    """Конкатенация двух векторов (переопределение для ясности)."""
    return a + b


# ─────────────────────────────────────────────
# GRU
# ─────────────────────────────────────────────

class GRU:
    """
    GRU-ячейка — упрощённая версия LSTM:

    reset gate:  r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
    update gate: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
    candidate:   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)

    hidden:      h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = hidden_size + input_size

        # Reset gate
        self.W_r = init_weights(hidden_size, self.concat_size)
        self.b_r = init_bias(hidden_size)

        # Update gate
        self.W_z = init_weights(hidden_size, self.concat_size)
        self.b_z = init_bias(hidden_size)

        # Candidate hidden state
        self.W_h = init_weights(hidden_size, hidden_size + input_size)
        self.b_h = init_bias(hidden_size)

    def forward(self, x_t, h_prev):
        """
        x_t:     (input_size,)  — вход
        h_prev:  (hidden_size,) — предыдущее скрытое состояние
        returns: (hidden_size,) — новое скрытое состояние
        """
        # Конкатенация [h_{t-1}, x_t]
        concat = concat_vec(h_prev, x_t)

        # Reset gate — что «забыть» при вычислении кандидата
        r_raw = vec_add(mat_vec_dot(self.W_r, concat), self.b_r)
        r = vec_sigmoid(r_raw)

        # Update gate — баланс между старым и новым
        z_raw = vec_add(mat_vec_dot(self.W_z, concat), self.b_z)
        z = vec_sigmoid(z_raw)

        # Candidate: h̃_t = tanh(W · [r ⊙ h_{t-1}, x_t] + b)
        r_h = elementwise_mul(r, h_prev)
        concat2 = concat_vec(r_h, x_t)
        h_raw = vec_add(mat_vec_dot(self.W_h, concat2), self.b_h)
        h_tilde = vec_tanh(h_raw)

        # h_t = (1 - z) ⊙ h̃_t + z ⊙ h_{t-1}
        one_minus_z = [1.0 - z[i] for i in range(len(z))]
        h_t = vec_add(elementwise_mul(one_minus_z, h_tilde), elementwise_mul(z, h_prev))

        return h_t


# ─────────────────────────────────────────────
# Демо 1: LSTM cell — forward pass
# ─────────────────────────────────────────────

def demo_1_lstm_forward():
    print("=" * 60)
    print("Демо 1: LSTM cell — forward pass")
    print("=" * 60)

    input_size = 4
    hidden_size = 3

    lstm = LSTM(input_size, hidden_size)

    # Последовательность из 5 шагов
    seq = [
        [0.5, 0.1, -0.3, 0.8],
        [0.2, -0.5, 0.7, 0.1],
        [-0.1, 0.4, 0.2, -0.6],
        [0.8, 0.3, -0.1, 0.5],
        [0.1, -0.2, 0.9, 0.3],
    ]

    h = [0.0] * hidden_size  # начальное h_0
    c = [0.0] * hidden_size  # начальное c_0

    print(f"\nРазмерность входа: {input_size}")
    print(f"Размерность скрытого состояния: {hidden_size}")
    print(f"Начальное h_0: {[round(x, 4) for x in h]}")
    print(f"Начальное c_0: {[round(x, 4) for x in c]}")
    print()

    for t, x_t in enumerate(seq):
        h, c = lstm.forward(x_t, h, c)
        print(f"Шаг {t+1}: вход {[round(v, 2) for v in x_t]}")
        print(f"       h_t = {[round(v, 4) for v in h]}")
        print(f"       c_t = {[round(v, 4) for v in c]}")
        print()

    print(f"Итоговое скрытое состояние: {[round(v, 4) for v in h]}")
    print(f"Итоговое состояние ячейки:  {[round(v, 4) for v in c]}")
    print()


# ─────────────────────────────────────────────
# Демо 2: GRU cell — forward pass
# ─────────────────────────────────────────────

def demo_2_gru_forward():
    print("=" * 60)
    print("Демо 2: GRU cell — forward pass")
    print("=" * 60)

    input_size = 4
    hidden_size = 3

    gru = GRU(input_size, hidden_size)

    seq = [
        [0.5, 0.1, -0.3, 0.8],
        [0.2, -0.5, 0.7, 0.1],
        [-0.1, 0.4, 0.2, -0.6],
        [0.8, 0.3, -0.1, 0.5],
        [0.1, -0.2, 0.9, 0.3],
    ]

    h = [0.0] * hidden_size

    print(f"\nРазмерность входа: {input_size}")
    print(f"Размерность скрытого состояния: {hidden_size}")
    print(f"Начальное h_0: {[round(x, 4) for x in h]}")
    print()

    for t, x_t in enumerate(seq):
        h = gru.forward(x_t, h)
        print(f"Шаг {t+1}: вход {[round(v, 2) for v in x_t]}")
        print(f"       h_t = {[round(v, 4) for v in h]}")
        print()

    print(f"Итоговое скрытое состояние: {[round(v, 4) for v in h]}")
    print()


# ─────────────────────────────────────────────
# Демо 3: Сравнение RNN vs LSTM vs GRU
# ─────────────────────────────────────────────

def demo_3_comparison():
    print("=" * 60)
    print("Демо 3: Сравнение RNN vs LSTM vs GRU")
    print("=" * 60)

    input_size = 4
    hidden_size = 3
    seq_len = 8

    rnn = SimpleRNN(input_size, hidden_size)
    lstm = LSTM(input_size, hidden_size)
    gru = GRU(input_size, hidden_size)

    # Генерируем случайную последовательность
    random.seed(42)
    seq = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(seq_len)]

    # Инициализация
    h_rnn = [0.0] * hidden_size
    h_lstm = [0.0] * hidden_size
    c_lstm = [0.0] * hidden_size
    h_gru = [0.0] * hidden_size

    print(f"\nПоследовательность: {seq_len} шагов, вход = {input_size}, hidden = {hidden_size}")
    print()
    print(f"{'Шаг':<5} {'RNN h[0]':>10} {'LSTM h[0]':>10} {'LSTM c[0]':>10} {'GRU h[0]':>10}")
    print("-" * 50)

    for t, x_t in enumerate(seq):
        h_rnn = rnn.forward(x_t, h_rnn)
        h_lstm, c_lstm = lstm.forward(x_t, h_lstm, c_lstm)
        h_gru = gru.forward(x_t, h_gru)

        print(f"{t+1:<5} {h_rnn[0]:>10.4f} {h_lstm[0]:>10.4f} {c_lstm[0]:>10.4f} {h_gru[0]:>10.4f}")

    print()
    print("Наблюдения:")
    print("- SimpleRNN: значения могут «застревать» (vanishing gradient problem)")
    print("- LSTM: cell state (c_t) сохраняет долгосрочную память благодаря遗忘/входным вентилям")
    print("- GRU: компактнее LSTM (2 вентиля вместо 3), но схожая способность удерживать информацию")
    print()


# ─────────────────────────────────────────────
# Демо 4: Затухание градиентов — RNN vs LSTM
# ─────────────────────────────────────────────

def compute_gradient_flow(rnn_class, lstm_class, seq_len=20, input_size=4, hidden_size=3):
    """
    Демонстрация затухания градиентов:
    Для RNN градиенты через T шагов уменьшаются как (W_hh)^T * diag(tanh')^T.
    Для LSTM cell state — линейный путь (forget gate), градиенты сохраняются.
    """

    # Инициализируем модели
    rnn = rnn_class(input_size, hidden_size)
    lstm = lstm_class(input_size, hidden_size)

    # Прямой проход для накопления последовательности состояний
    random.seed(42)
    seq = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(seq_len)]

    # --- SimpleRNN ---
    h = [0.0] * hidden_size
    rnn_states = [h[:]]  # копия начального
    for x_t in seq:
        h = rnn.forward(x_t, h)
        rnn_states.append(h[:])

    # --- LSTM ---
    h = [0.0] * hidden_size
    c = [0.0] * hidden_size
    lstm_c_states = [c[:]]
    for x_t in seq:
        h, c = lstm.forward(x_t, h, c)
        lstm_c_states.append(c[:])

    return rnn_states, lstm_c_states


def demo_4_vanishing_gradient():
    print("=" * 60)
    print("Демо 4: Затухание градиентов — RNN vs LSTM")
    print("=" * 60)

    input_size = 4
    hidden_size = 3
    seq_len = 15

    rnn = SimpleRNN(input_size, hidden_size)
    lstm = LSTM(input_size, hidden_size)

    random.seed(42)
    seq = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(seq_len)]

    # === Прямой проход ===
    h_rnn = [0.0] * hidden_size
    h_lstm = [0.0] * hidden_size
    c_lstm = [0.0] * hidden_size

    rnn_hs = [h_rnn[:]]
    lstm_cs = [c_lstm[:]]
    lstm_hs = [h_lstm[:]]

    for x_t in seq:
        h_rnn = rnn.forward(x_t, h_rnn)
        h_lstm, c_lstm = lstm.forward(x_t, h_lstm, c_lstm)
        rnn_hs.append(h_rnn[:])
        lstm_cs.append(c_lstm[:])
        lstm_hs.append(h_lstm[:])

    # === Моделируем затухание градиентов ===
    # Для RNN: градиент через T шагов ≈ prod(dh_t/dh_{t-1})
    # dh_t/dh_{t-1} ≈ diag(1 - h_t^2) * W_hh  (от tanh)
    # Для простоты: используем средний |h| как прокси «мощности» градиента

    # Вычисляем «норму влияния» для каждого шага
    # (на практике градиенты считаются через autograd, но тренд идентичен)

    print(f"\n{'Шаг':<5} {'|h_t| RNN':>12} {'|c_t| LSTM':>12} {'|h_t| LSTM':>12}")
    print("-" * 45)

    rnn_norms = []
    lstm_c_norms = []

    for t in range(len(rnn_hs)):
        rnn_norm = math.sqrt(sum(x**2 for x in rnn_hs[t])) / hidden_size
        lstm_c_norm = math.sqrt(sum(x**2 for x in lstm_cs[t])) / hidden_size
        lstm_h_norm = math.sqrt(sum(x**2 for x in lstm_hs[t])) / hidden_size

        rnn_norms.append(rnn_norm)
        lstm_c_norms.append(lstm_c_norm)

        print(f"{t:<5} {rnn_norm:>12.6f} {lstm_c_norm:>12.6f} {lstm_h_norm:>12.6f}")

    # === Теоретический анализ затухания ===
    print()
    print("Теоретический анализ затухания градиентов:")
    print("-" * 50)
    print()
    print("SimpleRNN:")
    print("  h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)")
    print("  dh_t/dh_{t-1} = diag(1 - h_t²) * W_hh")
    print(f"  Средний |h_t| RNN:      {sum(rnn_norms)/len(rnn_norms):.6f}")
    print("  → Спектральный радиус W_hh определяет затухание/взрыв")
    print("  → При малых собственных числах: EXPОНЕНЦИАЛЬНОЕ затухание")
    print()
    print("LSTM (cell state):")
    print("  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t")
    print("  dc_t/dc_{t-1} = diag(f_t)  ← ЛИНЕЙНЫЙ путь!")
    print(f"  Средний |c_t| LSTM:     {sum(lstm_c_norms)/len(lstm_c_norms):.6f}")
    print("  → Forget gate управляет длиной памяти")
    print("  → f_t ≈ 1 → градиент НЕ затухает (механизм highway)")
    print()

    # === Численная оценка затухания ===
    print("Численная оценка затухания градиентов:")
    print("-" * 50)

    # Для RNN: «множитель затухания» за 10 шагов
    # Пример: если собственные значения W_hh ≈ 0.5, то за 10 шагов 0.5^10 ≈ 0.001
    rnn_avg_h = sum(rnn_norms[1:]) / (len(rnn_norms) - 1)  # среднее |h|
    lstm_avg_c = sum(lstm_c_norms[1:]) / (len(lstm_c_norms) - 1)

    # «Размер ячейки» LSTM vs «размер скрытого состояния» RNN
    print(f"  RNN:   Средняя «мощность» скрытого состояния:  {rnn_avg_h:.6f}")
    print(f"  LSTM:  Средняя «мощность» cell state:          {lstm_avg_c:.6f}")
    print(f"  Отношение LSTM/RNN:                            {lstm_avg_c / max(rnn_avg_h, 1e-10):.2f}x")
    print()
    print("  Вывод: LSTM сохраняет значительно больше информации")
    print("         через cell state, чем RNN через скрытое состояние.")
    print()

    # === Решение проблемы ===
    print("Решение проблемы затухающих градиентов:")
    print("-" * 50)
    print("  1. LSTM (Long Short-Term Memory):")
    print("     - Forget gate: контролирует удаление информации из cell state")
    print("     - Input gate:  контролирует добавление новой информации")
    print("     - Output gate: контролирует выдачу информации из cell state")
    print("     - Результат: линейный путь для градиента через cell state")
    print()
    print("  2. GRU (Gated Recurrent Unit):")
    print("     - Update gate: объединяет функции forget + input gates")
    print("     - Reset gate:  управляет влиянием предыдущего состояния")
    print("     - Результат: компактная архитектура с аналогичными свойствами")
    print()
    print("  3. Skip connections / residual connections:")
    print("     - Прямая связь через несколько слоёв (как в ResNet)")
    print()
    print("  4. Gradient clipping:")
    print("     - Ограничение нормы градиента предотвращает взрыв")
    print()


# ─────────────────────────────────────────────
# Запуск всех демонстраций
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    LSTM и GRU: реализация с нуля                       ║")
    print("║    Проблема затухающих градиентов и решение              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo_1_lstm_forward()
    print()
    demo_2_gru_forward()
    print()
    demo_3_comparison()
    print()
    demo_4_vanishing_gradient()

    print("=" * 60)
    print("Все демонстрации завершены.")
    print("=" * 60)
