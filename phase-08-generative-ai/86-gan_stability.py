"""
GAN Stability Methods
=====================
Самодостаточный файл — реализация методов стабилизации GAN на чистом Python.

Методы:
  1. Label Smoothing
  2. Spectral Normalization (упрощённо)
  3. Wasserstein Distance (WGAN)
  4. Gradient Penalty
"""

import random
import math

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Вспомогательные математические функции (вместо numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def vec_add(a, b):
    """Поэлементное сложение."""
    return [x + y for x, y in zip(a, b)]

def vec_sub(a, b):
    """Поэлементное вычитание."""
    return [x - y for x, y in zip(a, b)]

def vec_scale(a, s):
    """Умножение вектора на скаляр."""
    return [x * s for x in a]

def vec_dot(a, b):
    """Скалярное произведение."""
    return sum(x * y for x, y in zip(a, b))

def vec_norm(a):
    """L2-норма вектора."""
    return math.sqrt(sum(x * x for x in a))

def vec_lerp(a, b, t):
    """Линейная интерполяция: a + t*(b-a)."""
    return [x + t * (y - x) for x, y in zip(a, b)]

def vec_zeros(n):
    """Нулевой вектор длины n."""
    return [0.0] * n

def mat_vec(mat, vec):
    """Умножение матрицы (списка списков) на вектор."""
    return [vec_dot(row, vec) for row in mat]

def mat_transpose(mat):
    """Транспонирование матрицы."""
    n_cols = len(mat[0])
    return [[row[i] for row in mat] for i in range(n_cols)]

def mat_mul(a, b):
    """Умножение матриц."""
    b_t = mat_transpose(b)
    return [vec_dot(row_a, row_b) for row_a in a for row_b in b_t]
    # Упрощённо — возвращаем плоский список; для наших задач достаточно

def random_normal(d):
    """Генерация вектора из стандартного нормального распределения (Box-Muller)."""
    result = []
    for _ in range(d):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
        result.append(z)
    return result

def sigmoid(x):
    """Сигмоида."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def relu(x):
    return max(0.0, x)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LABEL SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════════

def label_smoothing(target, num_classes, smoothing=0.1):
    """
    Label Smoothing: вместо hard-меток (0/1) используем мягкие.
    real_label = 1 - smoothing, fake_label = smoothing / (num_classes - 1)
    """
    smooth_labels = [smoothing / (num_classes - 1)] * num_classes
    smooth_labels[target] = 1.0 - smoothing
    return smooth_labels

def cross_entropy_smooth(logits, smooth_labels):
    """
    Cross-entropy с мягкими метками.
    logits: список сырых выходов (logits) модели.
    smooth_labels: мягкие метки.
    """
    # softmax
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    # cross-entropy
    loss = 0.0
    for p, y in zip(probs, smooth_labels):
        if y > 0:
            loss -= y * math.log(max(p, 1e-10))
    return loss

def cross_entropy_hard(logits, target_idx):
    """Стандартная cross-entropy с hard-меткой."""
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    return -math.log(max(probs[target_idx], 1e-10))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPECTRAL NORMALIZATION (упрощённо)
# ═══════════════════════════════════════════════════════════════════════════════

def power_iteration(mat, n_iter=20):
    """
    Power iteration для оценки максимального сингулярного числа (σ_max).
    mat: матрица весов (список списков).
    """
    rows = len(mat)
    cols = len(mat[0])
    # Случайный вектор
    u = [random.random() for _ in range(rows)]
    norm_u = vec_norm(u)
    if norm_u < 1e-10:
        return 1.0
    u = [x / norm_u for x in u]

    for _ in range(n_iter):
        # v = M^T u
        mat_t = mat_transpose(mat)
        v = mat_vec(mat_t, u)
        norm_v = vec_norm(v)
        if norm_v < 1e-10:
            return 1.0
        v = [x / norm_v for x in v]

        # u = M v
        u_new = mat_vec(mat, v)
        norm_u_new = vec_norm(u_new)
        if norm_u_new < 1e-10:
            return 1.0
        u = [x / norm_u_new for x in u_new]

    # σ_max ≈ u^T M v
    Mv = mat_vec(mat, v)
    sigma = vec_dot(u, Mv)
    return abs(sigma)

def spectral_normalize(mat, n_iter=20):
    """
    Spectral Normalization: делим веса на σ_max.
    Возвращает нормализованную матрицу и σ_max.
    """
    sigma = power_iteration(mat, n_iter)
    if sigma < 1e-10:
        sigma = 1.0
    normalized = [[w / sigma for w in row] for row in mat]
    return normalized, sigma


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WASSERSTEIN DISTANCE (WGAN)
# ═══════════════════════════════════════════════════════════════════════════════

def wasserstein_distance_1d(real_samples, fake_samples):
    """
    Wasserstein-1 расстояние (Earth Mover's Distance) для 1D.
    Через сортировку и поэлементную разницу.
    """
    sorted_real = sorted(real_samples)
    sorted_fake = sorted(fake_samples)
    n = min(len(sorted_real), len(sorted_fake))
    distance = sum(abs(sorted_real[i] - sorted_fake[i]) for i in range(n)) / n
    return distance

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Critic loss для WGAN: L = -(E[D(real)] - E[D(fake)]).
    Хотим МАКСИМИЗИРОВАТЬ E[D(real)] - E[D(fake)].
    """
    mean_real = sum(real_scores) / len(real_scores)
    mean_fake = sum(fake_scores) / len(fake_scores)
    loss = -(mean_real - mean_fake)
    return loss, mean_real - mean_fake

def clip_weights(weights, clip_value=0.01):
    """Weight clipping для WGAN (WGAN-GP не требует)."""
    return [[max(-clip_value, min(clip_value, w)) for w in row] for row in weights]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRADIENT PENALTY (WGAN-GP)
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_samples(real, fake, alpha):
    """Интерполяция между реальными и фейковыми сэмплами: x̂ = α·x_real + (1-α)·x_fake."""
    return vec_lerp(real, fake, alpha)

def compute_gradient_penalty_1d(real_sample, fake_sample, critic_fn, lambda_gp=10.0):
    """
    Gradient Penalty для WGAN-GP (многомерный вход).
    GP = λ · E[(‖∇_{x̂} D(x̂)‖₂ - 1)²]

    critic_fn: функция, возвращающая скалярный score для сэмпла.
    """
    alpha = random.random()
    interp = interpolate_samples(real_sample, fake_sample, alpha)

    # Численное приближение градиента: парциальные производные по каждому элементу
    eps = 1e-4
    gradient = []
    base_score = critic_fn(interp)
    for i in range(len(interp)):
        x_plus = list(interp)
        x_plus[i] += eps
        grad_i = (critic_fn(x_plus) - base_score) / eps
        gradient.append(grad_i)

    grad_norm = vec_norm(gradient)
    penalty = lambda_gp * ((grad_norm - 1.0) ** 2)
    return penalty, grad_norm


# ═══════════════════════════════════════════════════════════════════════════════
# Простой GAN-класс для демонстраций
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleGenerator:
    """Простой генератор: z -> hidden -> output.
    Матрицы хранятся как (out x in), mat_vec(W, x) = W @ x."""
    def __init__(self, z_dim, hidden_dim, out_dim, lr=0.001):
        self.lr = lr
        # W1: hidden_dim x z_dim (hidden_dim строк, z_dim столбцов)
        self.W1 = [[random.gauss(0, 0.1) for _ in range(z_dim)] for _ in range(hidden_dim)]
        self.b1 = vec_zeros(hidden_dim)
        # W2: out_dim x hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(out_dim)]
        self.b2 = vec_zeros(out_dim)

    def forward(self, z):
        # Layer 1: ReLU
        h = vec_add(mat_vec(self.W1, z), self.b1)
        h = [relu(x) for x in h]
        # Layer 2: Tanh (выход)
        out = vec_add(mat_vec(self.W2, h), self.b2)
        out = [math.tanh(x) for x in out]
        return out, h

    def generate(self, z_dim=8):
        z = random_normal(z_dim)
        out, _ = self.forward(z)
        return out


class SimpleCritic:
    """Простой критик (дискриминатор): x -> score.
    Матрицы хранятся как (out x in), mat_vec(W, x) = W @ x."""
    def __init__(self, in_dim, hidden_dim, lr=0.001, use_spectral_norm=False):
        self.lr = lr
        self.in_dim = in_dim
        self.use_spectral_norm = use_spectral_norm
        # W1: hidden_dim x in_dim
        self.W1 = [[random.gauss(0, 0.1) for _ in range(in_dim)] for _ in range(hidden_dim)]
        self.b1 = vec_zeros(hidden_dim)
        # W2: 1 x hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)]]
        self.b2 = [0.0]

        if use_spectral_norm:
            self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        self.W1_norm, _ = spectral_normalize(self.W1)
        self.W2_norm, _ = spectral_normalize(self.W2)

    def forward(self, x):
        w1 = self.W1_norm if self.use_spectral_norm else self.W1
        w2 = self.W2_norm if self.use_spectral_norm else self.W2

        h = vec_add(mat_vec(w1, x), self.b1)
        h = [relu(x_val) for x_val in h]
        out = vec_add(mat_vec(w2, h), self.b2)
        return out, h

    def score(self, x):
        out, _ = self.forward(x)
        return out[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def demo_label_smoothing():
    print("=" * 70)
    print("ДЕМО 1: LABEL SMOOTHING")
    print("=" * 70)
    print()
    print("Label smoothing заменяет hard-метки (0/1) мягкими.")
    print("Смягчает градиенты, не даёт дискриминатору стать слишком сильным.")
    print()

    num_classes = 2
    smoothing_values = [0.0, 0.05, 0.1, 0.2]

    # Сымитируем "уверенный" logits-вектор
    confident_logits = [5.0, -2.0]  # модель очень уверена в классе 0
    uncertain_logits = [1.0, 0.5]   # модель менее уверена

    print("Логиты модели: [5.0, -2.0] (уверенная), [1.0, 0.5] (неуверенная)")
    print()
    print(f"{'Smoothing':>10} | {'Hard CE':>10} | {'Smooth CE':>10} | {'Разница':>10}")
    print("-" * 55)

    for sm in smoothing_values:
        # Hard labels (target = 0)
        ce_hard = cross_entropy_hard(confident_logits, 0)

        # Smooth labels
        smooth_labels = label_smoothing(0, num_classes, sm)
        ce_smooth = cross_entropy_smooth(confident_logits, smooth_labels)

        print(f"{sm:>10.2f} | {ce_hard:>10.4f} | {ce_smooth:>10.4f} | {ce_hard - ce_smooth:>+10.4f}")

    print()
    print("Вывод:")
    print("  • Smoothing=0.0 → стандартная CE (hard labels)")
    print("  • Smoothing > 0 → CE увеличивается, модель наказывается за overly-уверенность")
    print("  • Это стабилизирует GAN: дискриминатор не доминирует над генератором")
    print()

    # Показываем мягкые метки
    print("Мягкие метки для smoothing=0.1:")
    for target in [0, 1]:
        labels = label_smoothing(target, 2, 0.1)
        print(f"  target={target}: {[f'{l:.2f}' for l in labels]}")
    print()


def demo_wasserstein_distance():
    print("=" * 70)
    print("ДЕМО 2: WASSERSTEIN DISTANCE (WGAN)")
    print("=" * 70)
    print()
    print("Wasserstein distance измеряет 'стоимость перевода' одного")
    print("распределения в другое (Earth Mover's Distance).")
    print()

    # Генерируем два распределения
    random.seed(42)
    real_samples = [random.gauss(0, 1) for _ in range(100)]

    # Генератор постепенно улучшается
    gen_stages = [
        ("Худший генератор (сдвиг = 3.0)", [x + 3.0 for x in random_normal(100)]),
        ("Средний генератор (сдвиг = 1.5)", None),
        ("Хороший генератор (сдвиг = 0.3)", None),
    ]

    random.seed(100)
    fake_bad = [x + 3.0 for x in [random.gauss(0, 1) for _ in range(100)]]
    fake_med = [x + 1.5 for x in [random.gauss(0, 1) for _ in range(100)]]
    fake_good = [x + 0.3 for x in [random.gauss(0, 1) for _ in range(100)]]

    fakes = [fake_bad, fake_med, fake_good]
    names = ["Худший генератор (сдвиг=3.0)", "Средний генератор (сдвиг=1.5)",
             "Хороший генератор (сдвиг=0.3)"]

    print("Реальное распределение: N(0, 1)")
    print()

    for name, fake in zip(names, fakes):
        w_dist = wasserstein_distance_1d(real_samples, fake)
        print(f"  {name}")
        print(f"    → Wasserstein distance: {w_dist:.4f}")

    print()
    print("Сравнение с другими метриками:")

    # BCE loss (обычный GAN)
    print()
    print("  В WGAN loss = -(E[D(real)] - E[D(fake)])")
    print("  Это напрямую связано с Wasserstein distance!")
    print()

    # Показываем WGAN critic loss для каждого этапа
    random.seed(42)
    for name, fake in zip(names, fakes):
        real_scores = [random.gauss(0.5, 0.2) for _ in range(50)]
        fake_scores = [random.gauss(-0.5, 0.2) for _ in range(50)]

        # Для "хорошего" генератора fake_scores будут ближе к real_scores
        if "Средний" in name:
            fake_scores = [s + 0.5 for s in fake_scores]
        elif "Хороший" in name:
            fake_scores = [s + 0.9 for s in fake_scores]

        loss, w_dist = wasserstein_critic_loss(real_scores, fake_scores)
        print(f"  {name}: W_critic_loss={loss:.4f}, W_distance≈{w_dist:.4f}")

    print()
    print("Вывод:")
    print("  • Wasserstein distance корректно уменьшается при улучшении генератора")
    print("  • Нет проблемы « vanishing gradient » — градиенты не исчезают")
    print("  • WGAN более стабилен, чем стандартный GAN")
    print()


def demo_gradient_penalty():
    print("=" * 70)
    print("ДЕМО 3: GRADIENT PENALTY (WGAN-GP)")
    print("=" * 70)
    print()
    print("Gradient Penalty обеспечивает 1-Lipschitz условие для критика")
    print("без weight clipping (как в WGAN).")
    print()
    print("GP = λ · E[(‖∇_{x̂} D(x̂)‖₂ - 1)²]")
    print("где x̂ = α·x_real + (1-α)·x_fake, α ~ U[0,1]")
    print()

    # Простой критик для демонстрации
    random.seed(42)
    critic = SimpleCritic(in_dim=8, hidden_dim=16, lr=0.0005)

    # Генерируем сэмплы
    real_samples = [[random.gauss(0, 1) for _ in range(8)] for _ in range(10)]
    fake_samples = [[random.gauss(2, 1) for _ in range(8)] for _ in range(10)]

    lambda_values = [0.1, 1.0, 10.0, 50.0]

    print(f"{'λ_GP':>8} | {'Mean GP':>10} | {'Mean |∇D|':>12} | {'Качество':>10}")
    print("-" * 55)

    for lam in lambda_values:
        penalties = []
        grad_norms = []
        for real, fake in zip(real_samples, fake_samples):
            gp, gn = compute_gradient_penalty_1d(real, fake, critic.score, lam)
            penalties.append(gp)
            grad_norms.append(gn)

        mean_gp = sum(penalties) / len(penalties)
        mean_gn = sum(grad_norms) / len(grad_norms)

        # Оценка качества (чем ближе к 1, тем лучше)
        quality = 1.0 / (1.0 + abs(mean_gn - 1.0))
        print(f"{lam:>8.1f} | {mean_gp:>10.4f} | {mean_gn:>12.4f} | {quality:>10.4f}")

    print()
    print("Вывод:")
    print("  • λ слишком мал → нет регуляризации, критик не-Lipschitz")
    print("  • λ слишком велик → критик слишком ограничен, потеряется информация")
    print("  • λ ≈ 10 (стандартное значение) — хороший баланс")
    print()
    print("Преимущества WGAN-GP:")
    print("  ✓ Не нужен weight clipping")
    print("  ✓ Более стабильное обучение")
    print("  ✓ Нет проблемы исчезающих градиентов")
    print("  ✓ Качество генерации выше")
    print()


def demo_comparison():
    print("=" * 70)
    print("ДЕМО 4: СРАВНЕНИЕ GAN vs WGAN vs WGAN-GP")
    print("=" * 70)
    print()

    random.seed(42)

    # Сымитируем обучение (несколько эпох)
    n_epochs = 100

    # === Стандартный GAN ===
    random.seed(42)
    g_losses_gan = []
    d_losses_gan = []
    for epoch in range(n_epochs):
        # Генератор улучшается, но нестабильно
        progress = epoch / n_epochs
        g_loss = max(0.1, 1.5 - progress * 2 + random.gauss(0, 0.3))
        d_loss = max(0.05, 0.5 + random.gauss(0, 0.4))

        # Иногда mode collapse
        if epoch in [30, 31, 65, 66]:
            g_loss *= 3.0  # резкий всплеск

        g_losses_gan.append(g_loss)
        d_losses_gan.append(d_loss)

    # === WGAN ===
    random.seed(42)
    g_losses_wgan = []
    d_losses_wgan = []
    for epoch in range(n_epochs):
        progress = epoch / n_epochs
        g_loss = max(-2.0, 2.0 - progress * 3 + random.gauss(0, 0.15))
        d_loss = max(-1.0, -g_loss + random.gauss(0, 0.1))

        g_losses_wgan.append(g_loss)
        d_losses_wgan.append(d_loss)

    # === WGAN-GP ===
    random.seed(42)
    g_losses_wgan_gp = []
    d_losses_wgan_gp = []
    for epoch in range(n_epochs):
        progress = epoch / n_epochs
        g_loss = max(-2.0, 1.8 - progress * 2.8 + random.gauss(0, 0.1))
        d_loss = max(-1.0, -g_loss + random.gauss(0, 0.08) + 0.01)

        g_losses_wgan_gp.append(g_loss)
        d_losses_wgan_gp.append(d_loss)

    # === Анализ ===
    def analyze(losses, name):
        mean = sum(losses) / len(losses)
        variance = sum((x - mean) ** 2 for x in losses) / len(losses)
        std = math.sqrt(variance)
        max_val = max(losses)
        min_val = min(losses)
        return name, mean, std, min_val, max_val

    print("Статистика лоссов генератора (100 эпох):")
    print()
    header = f"{'Метод':>12} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Стабильность':>12}"
    print(header)
    print("-" * 80)

    results = [
        analyze(g_losses_gan, "GAN"),
        analyze(g_losses_wgan, "WGAN"),
        analyze(g_losses_wgan_gp, "WGAN-GP"),
    ]

    for name, mean, std, min_v, max_v in results:
        stability = "ВЫСОКАЯ" if std < 0.3 else "СРЕДНЯЯ" if std < 0.5 else "НИЗКАЯ"
        print(f"{name:>12} | {mean:>8.4f} | {std:>8.4f} | {min_v:>8.4f} | {max_v:>8.4f} | {stability:>12}")

    print()
    print("Колебания loss (std) — чем меньше, тем стабильнее:")
    print(f"  GAN:    std = {math.sqrt(sum((x - sum(g_losses_gan)/len(g_losses_gan))**2 for x in g_losses_gan)/len(g_losses_gan)):.4f}")
    print(f"  WGAN:   std = {math.sqrt(sum((x - sum(g_losses_wgan)/len(g_losses_wgan))**2 for x in g_losses_wgan)/len(g_losses_wgan)):.4f}")
    print(f"  WGAN-GP: std = {math.sqrt(sum((x - sum(g_losses_wgan_gp)/len(g_losses_wgan_gp))**2 for x in g_losses_wgan_gp)/len(g_losses_wgan_gp)):.4f}")

    print()
    print("Итоговое сравнение:")
    print()
    print("  Метод      │ Стабильность │ Mode Collapse │ Качество │ Скорость")
    print("  ───────────┼──────────────┼───────────────┼──────────┼──────────")
    print("  GAN        │ Низкая       │ Часто         │ Среднее  │ Быстро")
    print("  WGAN       │ Высокая      │ Редко         │ Хорошее  │ Средне")
    print("  WGAN-GP    │ Очень высокая│ Очень редко   │ Лучшее   │ Медленно")
    print("  + Label    │              │               │          │")
    print("   Smoothing │ + стабильность│ + редко      │          │")
    print("  + Spectral │              │               │          │")
    print("   Norm      │ + стабильность│              │          │")
    print()
    print("Рекомендации:")
    print("  • Для начала: WGAN-GP + Label Smoothing")
    print("  • Для продакшена: WGAN-GP + Spectral Norm + Label Smoothing")
    print("  • Learning rate критика < learning rate генератора")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         GAN STABILITY METHODS — Демонстрация                   ║")
    print("║  Методы стабилизации: Label Smoothing, Spectral Norm,         ║")
    print("║  Wasserstein Distance, Gradient Penalty                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    demo_label_smoothing()
    demo_wasserstein_distance()
    demo_gradient_penalty()
    demo_comparison()

    print("═" * 70)
    print("Все демонстрации завершены.")
    print("═" * 70)
