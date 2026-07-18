"""
51 — Отладка нейросетей с нуля (Debugging Neural Networks)
==========================================================

Нейросеть строится С НУЛЯ на чистом Python:
  • Dense-слои, активации, loss, backpropagation
  • Gradient checking (numerical vs analytical)
  • Визуализация гистограмм весов и градиентов (текстовая)
  • Отслеживание loss по эпохам
  • Диагностика проблем (NaN, exploding/vanishing gradients)
  • Никаких импортов numpy, torch, sklearn

Демонстрации:
  Demo 1 — Gradient checking: аналитические vs числовые градиенты
  Demo 2 — Статистики весов и градиентов (гистограммы)
  Demo 3 — Loss curve: нормальное vs проблемное обучение
  Demo 4 — Диагностика NaN / inf
"""

import random
import math

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ: матричные операции
# ═══════════════════════════════════════════════════════════════════════════════

def rand_matrix(rows, cols, scale=1.0):
    limit = scale * math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def zeros_matrix(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def zeros_vector(n):
    return [0.0] * n


def rand_vector(n, scale=0.01):
    return [random.uniform(-scale, scale) for _ in range(n)]


def mat_vec_mul(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def vec_mat_mul(vec, mat):
    """(n,) × (n×m) → (m,)"""
    return [sum(vec[i] * mat[i][j] for i in range(len(vec))) for j in range(len(mat[0]))]


def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vec_scale(v, s):
    return [x * s for x in v]


def mat_scale(M, s):
    return [[M[i][j] * s for j in range(len(M[0]))] for i in range(len(M))]


def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def flatten_params(weights, biases):
    """Разворачивает список матриц весов и векторов смещений в один плоский список."""
    flat = []
    for W in weights:
        for row in W:
            flat.extend(row)
    for b in biases:
        flat.extend(b)
    return flat


def unflatten_params(flat, weight_shapes, bias_shapes):
    """Восстанавливает веса и смещения из плоского списка."""
    weights, biases = [], []
    idx = 0
    for (r, c) in weight_shapes:
        W = []
        for i in range(r):
            W.append(flat[idx:idx + c])
            idx += c
        weights.append(W)
    for n in bias_shapes:
        biases.append(flat[idx:idx + n])
        idx += n
    return weights, biases


# ═══════════════════════════════════════════════════════════════════════════════
# АКТИВАЦИИ И LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def sigmoid_derivative(out):
    return out * (1.0 - out)


def relu(x):
    return max(0.0, x)


def relu_derivative(out):
    return 1.0 if out > 0 else 0.0


def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy_loss(predicted, target):
    """Binary cross-entropy (для sigmoid-выхода): L = -Σ[t·log(a) + (1-t)·log(1-a)]"""
    eps = 1e-12
    loss = 0.0
    for i in range(len(predicted)):
        loss -= target[i] * math.log(max(predicted[i], eps))
        loss -= (1 - target[i]) * math.log(max(1 - predicted[i], eps))
    return loss


def mse_loss(predicted, target):
    return sum((predicted[i] - target[i]) ** 2 for i in range(len(predicted))) / len(predicted)


# ═══════════════════════════════════════════════════════════════════════════════
# НЕЙРОННАЯ СЕТЬ (простая feedforward)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleNetwork:
    """Полносвязная сеть с произвольным числом скрытых слоёв."""

    def __init__(self, layer_sizes, activation='sigmoid', lr=0.1):
        """
        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation
        self.lr = lr
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            W = rand_matrix(layer_sizes[i + 1], layer_sizes[i])  # (output × input)
            b = zeros_vector(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)
        # Кэш для backprop
        self._pre_act = []
        self._post_act = []

    def _activate(self, z):
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        return sigmoid(z)

    def _activate_deriv(self, out):
        if self.activation == 'sigmoid':
            return sigmoid_derivative(out)
        elif self.activation == 'relu':
            return relu_derivative(out)
        return sigmoid_derivative(out)

    def forward(self, x):
        """Forward pass. Возвращает выход сети и кэширует промежуточные значения."""
        self._pre_act = []
        self._post_act = []
        a = list(x)
        self._post_act.append(a)

        for i in range(self.num_layers):
            z = vec_add(mat_vec_mul(self.weights[i], a), self.biases[i])
            self._pre_act.append(z)
            a = [self._activate(zi) for zi in z]
            self._post_act.append(a)
        return a

    def backward(self, target):
        """Backward pass. Возвращает словарь градиентов {layer: {'W': ..., 'b': ...}}."""
        L = self.num_layers
        grads = {}

        # Выходной слой: dL/dz для softmax+CE = (predicted - target)
        output = self._post_act[L]
        delta = vec_sub(output, target)

        grads[L - 1] = {
            'W': outer(delta, self._post_act[L - 1]),
            'b': list(delta),
        }

        for i in range(L - 2, -1, -1):
            # error = W_next^T @ delta, then multiply by f'(z_i)
            W_next_t = [[self.weights[i + 1][r][c] for r in range(len(self.weights[i + 1]))]
                        for c in range(len(self.weights[i + 1][0]))]
            error = mat_vec_mul(W_next_t, delta)
            deriv = [self._activate_deriv(a) for a in self._post_act[i + 1]]
            delta = [error[j] * deriv[j] for j in range(len(error))]

            grads[i] = {
                'W': outer(delta, self._post_act[i]),
                'b': list(delta),
            }

        return grads

    def train_step(self, x, target):
        """Один шаг обучения. Возвращает loss и градиенты."""
        output = self.forward(x)
        loss = cross_entropy_loss(output, target)
        grads = self.backward(target)
        # Обновление весов
        for i in range(self.num_layers):
            for r in range(len(self.weights[i])):
                for c in range(len(self.weights[i][0])):
                    self.weights[i][r][c] -= self.lr * grads[i]['W'][r][c]
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= self.lr * grads[i]['b'][j]
        return loss, grads


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT CHECKING
# ═══════════════════════════════════════════════════════════════════════════════

def numerical_gradient_check(net, x, target, epsilon=1e-5):
    """
    Сравнивает аналитические градиенты (backprop) с числовыми (finite differences).

    Для каждого параметра θ:
      numerical_grad = [J(θ + ε) - J(θ - ε)] / (2ε)
    """
    # Получаем аналитические градиенты
    net.forward(x)
    analytical_grads = net.backward(target)

    # Начальный loss
    output = net.forward(x)
    base_loss = cross_entropy_loss(output, target)

    # Собираем все параметры
    all_grads_analytical = []
    all_grads_numerical = []
    param_locations = []  # (layer, 'W'/'b', row, col)

    # Для ускорения: проверяем только подмножество параметров
    check_indices = []
    idx = 0
    for li in range(net.num_layers):
        for r in range(len(net.weights[li])):
            for c in range(len(net.weights[li][0])):
                check_indices.append((li, 'W', r, c))
                idx += 1
        for j in range(len(net.biases[li])):
            check_indices.append((li, 'b', j, -1))
            idx += 1

    # Берём случайные 20 параметров для проверки
    random.shuffle(check_indices)
    samples = check_indices[:20]

    max_diff = 0.0
    for li, kind, row, col in samples:
        # Сохраняем оригинальное значение
        if kind == 'W':
            orig = net.weights[li][row][col]

            # J(θ + ε)
            net.weights[li][row][col] = orig + epsilon
            out_plus = net.forward(x)
            loss_plus = cross_entropy_loss(out_plus, target)

            # J(θ - ε)
            net.weights[li][row][col] = orig - epsilon
            out_minus = net.forward(x)
            loss_minus = cross_entropy_loss(out_minus, target)

            # Восстанавливаем
            net.weights[li][row][col] = orig

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = analytical_grads[li]['W'][row][col]
        else:
            orig = net.biases[li][row]

            net.biases[li][row] = orig + epsilon
            out_plus = net.forward(x)
            loss_plus = cross_entropy_loss(out_plus, target)

            net.biases[li][row] = orig - epsilon
            out_minus = net.forward(x)
            loss_minus = cross_entropy_loss(out_minus, target)

            net.biases[li][row] = orig

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = analytical_grads[li]['b'][row]

        denom = max(abs(numerical), abs(analytical), 1e-12)
        diff = abs(numerical - analytical) / denom
        max_diff = max(max_diff, diff)
        all_grads_analytical.append(analytical)
        all_grads_numerical.append(numerical)

    return {
        'max_relative_error': max_diff,
        'analytical_samples': all_grads_analytical[:5],
        'numerical_samples': all_grads_numerical[:5],
        'passed': max_diff < 1e-4,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕКСТОВЫЕ ГИСТОГРАММЫ
# ═══════════════════════════════════════════════════════════════════════════════

def text_histogram(values, title="", num_bins=20, width=50):
    """Строит текстовую гистограмму из списка чисел."""
    if not values:
        print(f"  {title}: [пусто]")
        return

    lo = min(values)
    hi = max(values)
    mean_v = sum(values) / len(values)
    std_v = (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5

    if lo == hi:
        bins = {0: len(values)}
        step = 0
    else:
        step = (hi - lo) / num_bins
        bins = {}
        for v in values:
            b = min(int((v - lo) / step), num_bins - 1) if step > 0 else 0
            bins[b] = bins.get(b, 0) + 1

    max_count = max(bins.values()) if bins else 1

    print(f"  {title}")
    print(f"  n={len(values)}  mean={mean_v:+.6f}  std={std_v:.6f}  min={lo:+.6f}  max={hi:+.6f}")
    for b in sorted(bins.keys()):
        count = bins[b]
        bar_len = int(count / max_count * width)
        if step > 0:
            lo_edge = lo + b * step
            hi_edge = lo_edge + step
            label = f"[{lo_edge:+.4f}, {hi_edge:+.4f})"
        else:
            label = f"[{lo:.4f}]"
        print(f"  {label:>22s} |{'#' * bar_len}")
    print()


def collect_all_weights(net):
    """Собирает все веса в один плоский список."""
    result = []
    for W in net.weights:
        for row in W:
            result.extend(row)
    return result


def collect_all_biases(net):
    result = []
    for b in net.biases:
        result.extend(b)
    return result


def collect_all_grads(grads, net):
    result = []
    for i in range(net.num_layers):
        for row in grads[i]['W']:
            result.extend(row)
        result.extend(grads[i]['b'])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА NaN / INF / EXPLODING GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def check_nan_inf(values, label=""):
    """Проверяет наличие NaN и Inf в списке значений."""
    nan_count = 0
    inf_count = 0
    for v in values:
        if v != v:  # NaN check
            nan_count += 1
        elif abs(v) == float('inf'):
            inf_count += 1
    return nan_count, inf_count


def diagnose_training(losses, grads_per_epoch, net):
    """Полная диагностика проблем обучения."""
    report = []

    # 1. Проверка loss
    if losses and losses[-1] != losses[-1]:
        report.append("[КРИТИЧНО] Loss содержит NaN — модель сломана!")
    if losses and abs(losses[-1]) == float('inf'):
        report.append("[КРИТИЧНО] Loss = inf — взрыв градиентов!")

    # 2. Монотонный рост loss (должен убывать)
    if len(losses) > 10:
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        if last_10 > first_10 * 1.5:
            report.append("[ПРЕДУПРЕЖДЕНИЕ] Loss растёт — слишком большой lr или плохая инициализация")

    # 3. Loss застрял на константе
    if len(losses) > 20:
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        if abs(first_10 - last_10) < 1e-6:
            report.append("[ПРЕДУПРЕЖДЕНИЕ] Loss не меняется — модель не учится (vanishing grad / слишком малый lr)")

    # 4. Проверка градиентов на NaN/Inf
    for epoch_i, grads in enumerate(grads_per_epoch):
        all_g = collect_all_grads(grads, net)
        nan_c, inf_c = check_nan_inf(all_g)
        if nan_c > 0:
            report.append(f"[КРИТИЧНО] Epoch {epoch_i}: {nan_c} NaN-градиентов")
        if inf_c > 0:
            report.append(f"[КРИТИЧНО] Epoch {epoch_i}: {inf_c} Inf-градиентов")

    # 5. Exploding gradients: норма градиентов > 10
    for epoch_i, grads in enumerate(grads_per_epoch):
        all_g = collect_all_grads(grads, net)
        grad_norm = sum(v ** 2 for v in all_g) ** 0.5
        if grad_norm > 10.0:
            report.append(f"[ПРЕДУПРЕЖДЕНИЕ] Epoch {epoch_i}: grad_norm={grad_norm:.2f} — exploding gradients")

    # 6. Vanishing gradients: норма < 1e-7
    for epoch_i, grads in enumerate(grads_per_epoch):
        all_g = collect_all_grads(grads, net)
        grad_norm = sum(v ** 2 for v in all_g) ** 0.5
        if 0 < grad_norm < 1e-7:
            report.append(f"[ПРЕДУПРЕЖДЕНИЕ] Epoch {epoch_i}: grad_norm={grad_norm:.2e} — vanishing gradients")

    # 7. Проверка весов
    all_w = collect_all_weights(net)
    nan_c, inf_c = check_nan_inf(all_w, "weights")
    if nan_c > 0:
        report.append(f"[КРИТИЧНО] Веса содержат {nan_c} NaN-значений")
    if inf_c > 0:
        report.append(f"[КРИТИЧНО] Веса содержат {inf_c} Inf-значений")

    w_norm = sum(v ** 2 for v in all_w) ** 0.5
    if w_norm > 100:
        report.append(f"[ПРЕДУПРЕЖДЕНИЕ] Норма весов={w_norm:.2f} — веса взрываются")

    if not report:
        report.append("[OK] Проблем не обнаружено")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def generate_xor_data(n=100):
    """Генерирует XOR-данные."""
    X, Y = [], []
    for _ in range(n):
        x1 = random.choice([0.0, 1.0])
        x2 = random.choice([0.0, 1.0])
        y = [1.0, 0.0] if int(x1) != int(x2) else [0.0, 1.0]
        X.append([x1, x2])
        Y.append(y)
    return X, Y


def train_and_track(net, X, Y, epochs=200, print_every=50):
    """Обучает сеть и возвращает историю loss и градиентов."""
    loss_history = []
    grads_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_grads = None
        for x, y in zip(X, Y):
            loss, grads = net.train_step(x, y)
            epoch_loss += loss
            epoch_grads = grads  # сохраняем последние градиенты

        avg_loss = epoch_loss / len(X)
        loss_history.append(avg_loss)
        grads_history.append(epoch_grads)

        if (epoch + 1) % print_every == 0:
            print(f"    Epoch {epoch+1:>4d}: loss = {avg_loss:.6f}")

    return loss_history, grads_history


def text_bar_chart(values, title="", width=40):
    """Простая текстовая столбчатая диаграмма (для loss curve)."""
    print(f"\n  {title}")
    print(f"  {'=' * (width + 25)}")
    lo = min(values)
    hi = max(values)
    for i, v in enumerate(values):
        bar_len = int((v - lo) / (hi - lo + 1e-12) * width) if hi > lo else width // 2
        marker = "<---" if v == hi or v == lo else ""
        print(f"  epoch {i:>3d} | {'#' * bar_len:<{width}s} {v:.4f} {marker}")
    print(f"  {'=' * (width + 25)}")


# ───────────────────────────────────────────────────────────────────────────────
# Demo 1: Gradient Checking
# ───────────────────────────────────────────────────────────────────────────────

def demo_gradient_checking():
    print("=" * 72)
    print("DEMO 1: GRADIENT CHECKING (Числовые vs Аналитические градиенты)")
    print("=" * 72)
    print()
    print("Идея: сравниваем градиенты из backpropagation с числовыми")
    print("приближениями: ∂J/∂θ ≈ [J(θ+ε) - J(θ-ε)] / 2ε")
    print()

    random.seed(42)
    net = SimpleNetwork([2, 3, 2], activation='sigmoid', lr=0.1)

    X, Y = generate_xor_data(4)

    print("Проверка на 4 примерах XOR:")
    for i, (x, y) in enumerate(zip(X, Y)):
        result = numerical_gradient_check(net, x, y, epsilon=1e-5)
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"\n  Пример {i+1}: x={x}, y={y}")
        print(f"    Макс. относительная ошибка: {result['max_relative_error']:.2e}")
        print(f"    Результат: {status}")
        print(f"    Аналитические (первые 5):    {[f'{g:+.8f}' for g in result['analytical_samples']]}")
        print(f"    Числовые (первые 5):         {[f'{g:+.8f}' for g in result['numerical_samples']]}")

    print("\n" + "-" * 72)
    print("ВЫВОД: если relative error < 1e-4, backpropagation корректна.")
    print("Это базовая проверка перед запуском обучения.")


# ───────────────────────────────────────────────────────────────────────────────
# Demo 2: Статистики весов и градиентов
# ───────────────────────────────────────────────────────────────────────────────

def demo_weight_gradient_stats():
    print("\n" + "=" * 72)
    print("DEMO 2: СТАТИСТИКИ ВЕСОВ И ГРАДИЕНТОВ (Гистограммы)")
    print("=" * 72)
    print()
    print("После обучения отслеживаем распределение весов и градиентов.")
    print("Хорошие признаки: симметричное распределение, разумный масштаб.")
    print("Плохие признаки: все в нуле, все на satuрации, огромные значения.")
    print()

    random.seed(42)
    net = SimpleNetwork([2, 8, 4, 2], activation='sigmoid', lr=0.5)

    X, Y = generate_xor_data(100)
    print("Обучение на XOR (100 примеров, 2 скрытых слоя: 8 и 4 нейрона)...\n")
    loss_history, grads_history = train_and_track(net, X, Y, epochs=100, print_every=25)

    print("\n--- Гистограмма ВЕСОВ после обучения ---")
    text_histogram(collect_all_weights(net), title="Все веса (weights)")

    print("--- Гистограмма СМЕЩЕНИЙ после обучения ---")
    text_histogram(collect_all_biases(net), title="Все смещения (biases)")

    print("--- Гистограмма ГРАДИЕНТОВ (последняя эпоха) ---")
    last_grads = grads_history[-1]
    text_histogram(collect_all_grads(last_grads, net), title="Все градиенты (grads)")

    print("--- Покрытое по слоям (last epoch grads) ---")
    for i in range(net.num_layers):
        layer_W_grads = []
        for row in last_grads[i]['W']:
            layer_W_grads.extend(row)
        layer_b_grads = list(last_grads[i]['b'])
        print(f"\n  Слой {i}: W shape ({net.layer_sizes[i+1]}x{net.layer_sizes[i]}), "
              f"b shape ({net.layer_sizes[i+1]})")
        text_histogram(layer_W_grads, title=f"  Grads W (слой {i})", num_bins=10, width=30)
        text_histogram(layer_b_grads, title=f"  Grads b (слой {i})", num_bins=5, width=30)


# ───────────────────────────────────────────────────────────────────────────────
# Demo 3: Loss Curve — нормальное vs проблемное обучение
# ───────────────────────────────────────────────────────────────────────────────

def demo_loss_curves():
    print("\n" + "=" * 72)
    print("DEMO 3: LOSS CURVE — НОРМАЛЬНОЕ vs ПРОБЛЕМНОЕ ОБУЧЕНИЕ")
    print("=" * 72)
    print()

    # --- Нормальное обучение ---
    print("--- 3a. Нормальное обучение (lr=0.5, sigmoid) ---\n")
    random.seed(42)
    net_ok = SimpleNetwork([2, 8, 2], activation='sigmoid', lr=0.5)
    X, Y = generate_xor_data(100)
    loss_ok, _ = train_and_track(net_ok, X, Y, epochs=100, print_every=25)

    # Показываем кривую loss
    print("\n  Loss curve (нормальное):")
    sampled = [loss_ok[i] for i in range(0, len(loss_ok), 5)]
    text_bar_chart(sampled, title="Loss каждые 5 эпох", width=35)
    final_acc = sum(1 for i in range(len(X)) if
                    (net_ok.forward(X[i])[0] > 0.5) == (Y[i][0] > 0.5)) / len(X)
    print(f"  Точность после обучения: {final_acc*100:.0f}%\n")

    # --- Слишком большой lr ---
    print("--- 3b. Слишком большой lr=5.0 (exploding loss) ---\n")
    random.seed(42)
    net_bad_lr = SimpleNetwork([2, 8, 2], activation='sigmoid', lr=5.0)
    loss_bad, _ = train_and_track(net_bad_lr, X, Y, epochs=100, print_every=25)

    sampled_bad = [loss_bad[i] for i in range(0, len(loss_bad), 5)]
    text_bar_chart(sampled_bad, title="Loss каждые 5 эпох (lr=5.0)", width=35)

    if loss_bad[-1] > 10:
        print("  ⚠ Loss ВЗРОС — модель разошлась из-за слишком большого lr!\n")
    else:
        print(f"  Loss = {loss_bad[-1]:.4f} (модель нестабильна)\n")

    # --- Слишком малый lr ---
    print("--- 3c. Слишком малый lr=0.001 (медленное обучение) ---\n")
    random.seed(42)
    net_slow = SimpleNetwork([2, 8, 2], activation='sigmoid', lr=0.001)
    loss_slow, _ = train_and_track(net_slow, X, Y, epochs=100, print_every=25)

    sampled_slow = [loss_slow[i] for i in range(0, len(loss_slow), 5)]
    text_bar_chart(sampled_slow, title="Loss каждые 5 эпох (lr=0.001)", width=35)
    final_acc_slow = sum(1 for i in range(len(X)) if
                         (net_slow.forward(X[i])[0] > 0.5) == (Y[i][0] > 0.5)) / len(X)
    print(f"  Точность после 100 эпох: {final_acc_slow*100:.0f}%")
    print("  lr=0.001 слишком мал — loss почти не убывает.\n")

    # --- Сравнение ---
    print("--- 3d. Сравнение lr=0.1, 0.5, 1.0, 5.0 ---\n")
    lrs = [0.1, 0.5, 1.0, 5.0]
    final_losses = []
    for lr in lrs:
        random.seed(42)
        net_t = SimpleNetwork([2, 8, 2], activation='sigmoid', lr=lr)
        loss_t, _ = train_and_track(net_t, X, Y, epochs=100, print_every=999)
        final_losses.append(loss_t[-1])

    print(f"  {'lr':>6s} | {'Final Loss':>12s} | {'Статус':>20s}")
    print(f"  {'-'*6} | {'-'*12} | {'-'*20}")
    for lr, fl in zip(lrs, final_losses):
        if fl < 0.1:
            status = "Хорошо"
        elif fl < 0.7:
            status = "Медленно"
        elif fl < 5.0:
            status = "Нестабильно"
        else:
            status = "Разошлась!"
        print(f"  {lr:>6.1f} | {fl:>12.6f} | {status:>20s}")


# ───────────────────────────────────────────────────────────────────────────────
# Demo 4: Диагностика NaN / inf
# ───────────────────────────────────────────────────────────────────────────────

def demo_nan_diagnosis():
    print("\n" + "=" * 72)
    print("DEMO 4: ДИАГНОСТИКА NaN / INF В ГРАДИЕНТАХ И ВЕСАХ")
    print("=" * 72)
    print()

    # --- 4a. Обучение с проблемным lr (вызывает NaN) ---
    print("--- 4a. Обучение с lr=100.0 (вызывает NaN в градиентах) ---\n")
    random.seed(42)
    net_nan = SimpleNetwork([2, 8, 2], activation='sigmoid', lr=100.0)
    X, Y = generate_xor_data(50)
    loss_nan = []
    grads_nan = []

    for epoch in range(50):
        epoch_loss = 0.0
        last_grads = None
        for x, y in zip(X, Y):
            output = net_nan.forward(x)
            loss = cross_entropy_loss(output, y)

            # Проверяем loss перед backprop
            if loss != loss:  # NaN check
                break
            grads = net_nan.backward(y)

            # Обновляем веса вручную с проверкой
            for i in range(net_nan.num_layers):
                for r in range(len(net_nan.weights[i])):
                    for c in range(len(net_nan.weights[i][0])):
                        new_val = net_nan.weights[i][r][c] - net_nan.lr * grads[i]['W'][r][c]
                        if new_val != new_val or abs(new_val) == float('inf'):
                            grads[i]['W'][r][c] = 0  # clamp
                            new_val = net_nan.weights[i][r][c]
                        net_nan.weights[i][r][c] = new_val
                for j in range(len(net_nan.biases[i])):
                    new_val = net_nan.biases[i][j] - net_nan.lr * grads[i]['b'][j]
                    if new_val != new_val or abs(new_val) == float('inf'):
                        grads[i]['b'][j] = 0
                        new_val = net_nan.biases[i][j]
                    net_nan.biases[i][j] = new_val

            epoch_loss += loss if loss == loss else 0
            last_grads = grads

        if loss != loss:
            loss_nan.append(float('nan'))
        else:
            loss_nan.append(epoch_loss / len(X))
        grads_nan.append(last_grads)

        if epoch < 10 or (epoch + 1) % 10 == 0:
            w_vals = collect_all_weights(net_nan)
            nan_c, inf_c = check_nan_inf(w_vals)
            grad_norm = 0
            if last_grads:
                all_g = collect_all_grads(last_grads, net_nan)
                grad_norm = sum(v ** 2 for v in all_g if v == v) ** 0.5
            print(f"    Epoch {epoch+1:>3d}: loss={loss_nan[-1] if loss_nan[-1] == loss_nan[-1] else float('nan'):.2f}  "
                  f"nan_in_weights={nan_c}  grad_norm={grad_norm:.1f}")

    # --- Диагностика ---
    print("\n--- Диагностический отчёт ---")
    report = diagnose_training(loss_nan, grads_nan, net_nan)
    for line in report:
        print(f"  {line}")

    # --- 4b. Демонстрация детекции конкретных проблем ---
    print("\n" + "-" * 72)
    print("--- 4b. Детекция типичных проблем (симуляция) ---\n")

    # Simulate: normal loss that suddenly explodes
    sim_losses = [2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.85, 0.8, 0.75,
                  0.8, 1.2, 3.5, 10.0, 50.0, float('nan')]
    # dummy_net has 2 layers: [2,2,2]
    sim_grads = []
    for _ in range(16):
        sim_grads.append({0: {'W': [[0.1, 0.05], [0.05, 0.1]], 'b': [0.05, 0.05]},
                          1: {'W': [[0.1, 0.05], [0.05, 0.1]], 'b': [0.05, 0.05]}})
    sim_grads[-2][1]['W'][0][0] = float('inf')

    random.seed(42)
    dummy_net = SimpleNetwork([2, 2, 2])

    print("  Симулированный сценарий: loss нормально убывает, затем взрывается.")
    print()
    report_sim = diagnose_training(sim_losses, sim_grads, dummy_net)
    for line in report_sim:
        print(f"  {line}")

    # --- 4c. Практические советы ---
    print("\n" + "-" * 72)
    print("--- 4c. Практические советы по отладке ---")
    print()
    print("  1. GRADIENT CHECKING — всегда запускайте перед обучением.")
    print("     Если numerical ≈ analytical, backpropagation корректна.")
    print()
    print("  2. НОРМА ГРАДИЕНТОВ — отслеживайте ||∇L|| на каждой эпохе.")
    print("     • Норма растёт экспоненциально → lr слишком большой.")
    print("     • Норма ≈ 0 → vanishing gradients (проверьте активации).")
    print()
    print("  3. LOSS CURVE — всегда визуализируйте loss по эпохам.")
    print("     • Loss не убывает → lr слишком малый / bad init.")
    print("     • Loss скачет → lr слишком большой / нет нормализации.")
    print()
    print("  4. NaN/INF — используйте gradient clipping (clip_by_norm).")
    print("     if ||grad|| > max_norm: grad = grad * max_norm / ||grad||")
    print()
    print("  5. ВЕСА — проверяйте их распределение после init и после обучения.")
    print("     • Все ≈ 0 → сеть не выучила ничего.")
    print("     • Все ≈ константу → bottleneck.")
    print("     • Экстремальные значения → нужно регуляризация.")

    print("\n" + "=" * 72)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_gradient_checking()
    demo_weight_gradient_stats()
    demo_loss_curves()
    demo_nan_diagnosis()
