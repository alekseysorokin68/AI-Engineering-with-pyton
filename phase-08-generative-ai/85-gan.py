"""
85 - Generative Adversarial Network (GAN)
==========================================

Самодостаточная реализация GAN на чистом Python (без внешних зависимостей).

Компоненты:
  - Generator: генерирует данные из шума
  - Discriminator: отличает реальные данные от сгенерированных
  - Adversarial Training: двухагентное обучение
  - Метрики: fake ratio, loss, accuracy

Демо:
  1. Generator — генерация данных
  2. Discriminator — классификация
  3. Adversarial training (полный цикл)
  4. Эволюция генератора по эпохам
"""

import random
import math

random.seed(42)

# ============================================================
# УТИЛИТЫ
# ============================================================

def sigmoid(x: float) -> float:
    """Сигмоида с защитой от переполнения."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def sigmoid_derivative(out: float) -> float:
    """Производная сигмоиды: s * (1 - s)."""
    return out * (1.0 - out)


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(out: float) -> float:
    return 1.0 - out * out


def mse_loss(predicted: list[float], target: list[float]) -> float:
    """Среднеквадратичная ошибка."""
    n = len(predicted)
    return sum((p - t) ** 2 for p, t in zip(predicted, target)) / n


def binary_cross_entropy(predicted: list[float], target: list[float]) -> float:
    """Бинарная кросс-энтропия."""
    eps = 1e-7
    n = len(predicted)
    total = 0.0
    for p, t in zip(predicted, target):
        p = max(eps, min(1.0 - eps, p))
        total += -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))
    return total / n


def Xavier_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Инициализация весов Xavier."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [[random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]


def print_separator(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# ГЕНЕРАТОР
# ============================================================

class Generator:
    """
    Генератор: принимает вектор шума (latent) и выдаёт данные.
    Архитектура: latent_dim -> 16 -> 8 -> output_dim
    Активации: ReLU (скрытые), Tanh (выход)
    """

    def __init__(self, latent_dim: int, output_dim: int, lr: float = 0.01):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.lr = lr

        # Слой 1: latent_dim -> 16
        self.W1 = Xavier_init(latent_dim, 16)
        self.b1 = [0.0] * 16

        # Слой 2: 16 -> output_dim
        self.W2 = Xavier_init(16, output_dim)
        self.b2 = [0.0] * output_dim

    def forward(self, z: list[float]) -> tuple[list[float], dict]:
        """Прямой проход. Возвращает (output, cache)."""
        # Слой 1: ReLU
        h1_raw = [sum(z[i] * self.W1[i][j] for i in range(self.latent_dim)) + self.b1[j]
                   for j in range(16)]
        h1 = [relu(x) for x in h1_raw]

        # Слой 2: Tanh
        out_raw = [sum(h1[i] * self.W2[i][j] for i in range(16)) + self.b2[j]
                   for j in range(self.output_dim)]
        out = [tanh(x) for x in out_raw]

        cache = {"z": z, "h1_raw": h1_raw, "h1": h1, "out_raw": out_raw, "out": out}
        return out, cache

    def backward(self, d_out: list[float], cache: dict):
        """Обратное проход через генератор."""
        z = cache["z"]
        h1_raw = cache["h1_raw"]
        h1 = cache["h1"]

        # Градиент слоя 2
        d_out_raw = [d_out[j] * tanh_derivative(cache["out"][j])
                     for j in range(self.output_dim)]

        # Градиенты W2, b2
        dW2 = [[h1[i] * d_out_raw[j] for j in range(self.output_dim)]
               for i in range(16)]
        db2 = list(d_out_raw)

        # Градиент на h1
        d_h1 = [sum(self.W2[i][j] * d_out_raw[j] for j in range(self.output_dim))
                for i in range(16)]

        # Градиент слоя 1 (ReLU)
        d_h1_raw = [d_h1[i] * relu_derivative(h1_raw[i]) for i in range(16)]

        # Градиенты W1, b1
        dW1 = [[z[i] * d_h1_raw[j] for j in range(16)]
               for i in range(self.latent_dim)]
        db1 = list(d_h1_raw)

        # Обновление весов
        for i in range(self.latent_dim):
            for j in range(16):
                self.W1[i][j] -= self.lr * dW1[i][j]
        for j in range(16):
            self.b1[j] -= self.lr * db1[j]

        for i in range(16):
            for j in range(self.output_dim):
                self.W2[i][j] -= self.lr * dW2[i][j]
        for j in range(self.output_dim):
            self.b2[j] -= self.lr * db2[j]


# ============================================================
# ДИСКРИМИНАТОР
# ============================================================

class Discriminator:
    """
    Дискриминатор: принимает данные и выдаёт вероятность "реальности".
    Архитектура: input_dim -> 16 -> 8 -> 1
    Активации: ReLU (скрытые), Sigmoid (выход)
    """

    def __init__(self, input_dim: int, lr: float = 0.01):
        self.input_dim = input_dim
        self.lr = lr

        # Слой 1: input_dim -> 16
        self.W1 = Xavier_init(input_dim, 16)
        self.b1 = [0.0] * 16

        # Слой 2: 16 -> 1
        self.W2 = Xavier_init(16, 1)
        self.b2 = [0.0]

    def forward(self, x: list[float]) -> tuple[float, dict]:
        """Прямой проход. Возвращает (probability, cache)."""
        # Слой 1: ReLU
        h1_raw = [sum(x[i] * self.W1[i][j] for i in range(self.input_dim)) + self.b1[j]
                   for j in range(16)]
        h1 = [relu(v) for v in h1_raw]

        # Слой 2: Sigmoid
        out_raw = sum(h1[i] * self.W2[i][0] for i in range(16)) + self.b2[0]
        prob = sigmoid(out_raw)

        cache = {"x": x, "h1_raw": h1_raw, "h1": h1, "out_raw": out_raw, "prob": prob}
        return prob, cache

    def backward(self, d_prob: float, cache: dict):
        """Обратное проход через дискриминатор."""
        x = cache["x"]
        h1_raw = cache["h1_raw"]
        h1 = cache["h1"]
        prob = cache["prob"]

        # Градиент выходного слоя
        d_out_raw = d_prob * sigmoid_derivative(prob)

        # Градиенты W2, b2
        dW2 = [[h1[i] * d_out_raw] for i in range(16)]
        db2 = [d_out_raw]

        # Градиент на h1
        d_h1 = [self.W2[i][0] * d_out_raw for i in range(16)]

        # Градиент слоя 1 (ReLU)
        d_h1_raw = [d_h1[i] * relu_derivative(h1_raw[i]) for i in range(16)]

        # Градиенты W1, b1
        dW1 = [[x[i] * d_h1_raw[j] for j in range(16)]
               for i in range(self.input_dim)]
        db1 = list(d_h1_raw)

        # Обновление весов
        for i in range(self.input_dim):
            for j in range(16):
                self.W1[i][j] -= self.lr * dW1[i][j]
        for j in range(16):
            self.b1[j] -= self.lr * db1[j]

        for i in range(16):
            self.W2[i][0] -= self.lr * dW2[i][0]
        self.b2[0] -= self.lr * db2[0]

    def classify(self, x: list[float]) -> str:
        prob, _ = self.forward(x)
        return "REAL" if prob >= 0.5 else "FAKE"


# ============================================================
# GAN (ADVERSARIAL TRAINING)
# ============================================================

class GAN:
    """
    Generative Adversarial Network.

    Цикл обучения:
      1. Тренируем D на реальных данных (label=1) и фейковых (label=0)
      2. Тренируем G через D: генерируем фейки, хотим чтобы D сказал 1
    """

    def __init__(self, latent_dim: int, data_dim: int, lr: float = 0.01):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.G = Generator(latent_dim, data_dim, lr)
        self.D = Discriminator(data_dim, lr)
        self.history = {"g_loss": [], "d_loss": [], "d_acc": [], "fake_ratio": []}

    def generate_noise(self) -> list[float]:
        return [random.gauss(0, 1) for _ in range(self.latent_dim)]

    def train_step(self, real_data: list[list[float]]) -> dict:
        """Один шаг обучения GAN."""
        batch_size = len(real_data)
        smooth = 0.1  # Label smoothing

        # --- Шаг 1: Тренируем Discriminator ---
        d_losses = []
        d_correct = 0

        for real_sample in real_data:
            # Реальные данные -> D ->应该输出 ~1
            prob_real, cache_real = self.D.forward(real_sample)
            loss_real = -math.log(max(prob_real, 1e-7))
            d_correct += 1 if prob_real >= 0.5 else 0

            # Градиент для реальных данных (label = 1 - smooth)
            d_prob_real = -(1.0 - smooth) / max(prob_real, 1e-7)
            self.D.backward(d_prob_real, cache_real)

            # Фейковые данные -> D ->应该输出 ~0
            z = self.generate_noise()
            fake_sample, _ = self.G.forward(z)
            prob_fake, cache_fake = self.D.forward(fake_sample)
            loss_fake = -math.log(max(1.0 - prob_fake, 1e-7))
            d_correct += 1 if prob_fake < 0.5 else 0

            # Градиент для фейковых данных (label = 0)
            d_prob_fake = 1.0 / max(1.0 - prob_fake, 1e-7)
            self.D.backward(d_prob_fake, cache_fake)

            d_losses.append((loss_real + loss_fake) / 2)

        # --- Шаг 2: Тренируем Generator ---
        g_losses = []
        fake_ratio_sum = 0.0

        for _ in range(batch_size):
            z = self.generate_noise()
            fake_sample, g_cache = self.G.forward(z)
            prob, d_cache = self.D.forward(fake_sample)

            fake_ratio_sum += 1.0 - prob  # Доля "фейка" по мнению D

            # Генератор хочет, чтобы D выдал 1
            # dL/d(prob) для BCE: -(target/prob - (1-target)/(1-prob))
            # target = 1: dL/d(prob) = -1/prob
            # D backward: dL/d(h1) через dL/d(prob) * sigmoid'
            # Но проще: градиент D по входу = prob - 1 (для target=1)

            # Вычисляем градиент D по входу
            d_input_grad = [0.0] * self.data_dim
            d_prob = prob - 1.0  # Градиент BCE по выходу D (target=1)

            # Проходим через D
            h1 = d_cache["h1"]
            d_out_raw = d_prob * sigmoid_derivative(prob)

            # Через W2
            d_h1 = [self.D.W2[i][0] * d_out_raw for i in range(16)]
            h1_raw = d_cache["h1_raw"]
            d_h1_raw = [d_h1[i] * relu_derivative(h1_raw[i]) for i in range(16)]

            # Через W1 -> градиент по входу D (= выходу G)
            for j in range(self.data_dim):
                d_input_grad[j] = sum(self.D.W1[j][i] * d_h1_raw[i] for i in range(16))

            # Обновляем G
            self.G.backward(d_input_grad, g_cache)
            g_losses.append(-math.log(max(prob, 1e-7)))

        # Метрики
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)
        d_acc = d_correct / (2 * batch_size) * 100
        avg_fake_ratio = fake_ratio_sum / batch_size

        self.history["g_loss"].append(avg_g_loss)
        self.history["d_loss"].append(avg_d_loss)
        self.history["d_acc"].append(d_acc)
        self.history["fake_ratio"].append(avg_fake_ratio)

        return {
            "g_loss": avg_g_loss,
            "d_loss": avg_d_loss,
            "d_acc": d_acc,
            "fake_ratio": avg_fake_ratio,
        }


# ============================================================
# ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================

def generate_real_data(n_samples: int, data_dim: int) -> list[list[float]]:
    """
    Генерирует 'реальные' данные: смесь двух кластеров.
    Кластер 1: центр в (0.5, 0.5, ..., 0.5)
    Кластер 2: центр в (-0.5, -0.5, ..., -0.5)
    """
    data = []
    for _ in range(n_samples):
        center = [0.5] * data_dim if random.random() < 0.5 else [-0.5] * data_dim
        sample = [c + random.gauss(0, 0.15) for c in center]
        data.append(sample)
    return data


# ============================================================
# ДЕМО 1: ГЕНЕРАТОР — ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================

def demo1_generator():
    print_separator("DEMO 1: Generator — Генерация данных")

    latent_dim = 4
    data_dim = 3
    gen = Generator(latent_dim, data_dim, lr=0.05)

    print("\nИнициализация генератора:")
    print(f"  Вход (шум): {latent_dim} измерений")
    print(f"  Выход (данные): {data_dim} измерений")
    print(f"  Архитектура: {latent_dim} -> 16 (ReLU) -> {data_dim} (Tanh)")

    print("\n--- Случайные вектора шума -> Сгенерированные данные ---\n")
    for i in range(5):
        z = [random.gauss(0, 1) for _ in range(latent_dim)]
        generated, _ = gen.forward(z)
        z_str = ", ".join(f"{v:+.3f}" for v in z)
        g_str = ", ".join(f"{v:+.3f}" for v in generated)
        print(f"  Z[{i}] = [{z_str}]")
        print(f"  G(Z) = [{g_str}]")
        print()

    # Проверяем статистику выхода Tanh
    print("--- Статистика выхода (50 сэмплов) ---\n")
    outputs = []
    for _ in range(50):
        z = [random.gauss(0, 1) for _ in range(latent_dim)]
        out, _ = gen.forward(z)
        outputs.append(out)

    means = [sum(outputs[s][d] for s in range(50)) / 50 for d in range(data_dim)]
    stds = [math.sqrt(sum((outputs[s][d] - means[d]) ** 2 for s in range(50)) / 50) for d in range(data_dim)]

    for d in range(data_dim):
        print(f"  Выход[{d}]: mean={means[d]:+.4f}, std={stds[d]:.4f}")

    print(f"\n  Выходные значения в диапазоне [-1, 1] (Tanh)")


# ============================================================
# ДЕМО 2: ДИСКРИМИНАТОР — КЛАССИФИКАЦИЯ
# ============================================================

def demo2_discriminator():
    print_separator("DEMO 2: Discriminator — Классификация")

    data_dim = 3
    disc = Discriminator(data_dim, lr=0.05)

    # Генерируем данные
    real_data = generate_real_data(20, data_dim)
    fake_data = [[random.uniform(-1, 1) for _ in range(data_dim)] for _ in range(20)]

    print("\nТренировка дискриминатора (50 итераций)...\n")

    for epoch in range(50):
        for sample in real_data:
            prob, cache = disc.forward(sample)
            # Label = 0.9 (smoothed)
            d_prob = -(0.9) / max(prob, 1e-7)
            disc.backward(d_prob, cache)

        for sample in fake_data:
            prob, cache = disc.forward(sample)
            # Label = 0.1 (smoothed)
            d_prob = 1.0 / max(1.0 - prob, 1e-7)
            disc.backward(d_prob, cache)

        if (epoch + 1) % 10 == 0:
            correct = 0
            for s in real_data:
                if disc.classify(s) == "REAL":
                    correct += 1
            for s in fake_data:
                if disc.classify(s) == "FAKE":
                    correct += 1
            acc = correct / 40 * 100
            print(f"  Epoch {epoch + 1:3d}: Accuracy = {acc:.1f}%")

    print("\n--- Классификация новых данных ---\n")
    test_real = generate_real_data(5, data_dim)
    test_fake = [[random.uniform(-1, 1) for _ in range(data_dim)] for _ in range(5)]

    print("  Реальные данные:")
    for i, s in enumerate(test_real):
        prob, _ = disc.forward(s)
        label = disc.classify(s)
        vals = ", ".join(f"{v:+.3f}" for v in s)
        print(f"    [{vals}] -> P(real)={prob:.4f} => {label}")

    print("\n  Фейковые данные (случайный шум):")
    for i, s in enumerate(test_fake):
        prob, _ = disc.forward(s)
        label = disc.classify(s)
        vals = ", ".join(f"{v:+.3f}" for v in s)
        print(f"    [{vals}] -> P(real)={prob:.4f} => {label}")


# ============================================================
# ДЕМО 3: ADVERSARIAL TRAINING
# ============================================================

def demo3_adversarial():
    print_separator("DEMO 3: Adversarial Training (полный цикл)")

    latent_dim = 4
    data_dim = 3
    gan = GAN(latent_dim, data_dim, lr=0.02)

    n_epochs = 80
    batch_size = 16

    print(f"\n  Архитектура:")
    print(f"    Generator:    {latent_dim} -> 16 (ReLU) -> {data_dim} (Tanh)")
    print(f"    Discriminator: {data_dim} -> 16 (ReLU) -> 1 (Sigmoid)")
    print(f"    Эпох: {n_epochs}, Batch size: {batch_size}")
    print(f"    Label smoothing: 0.1")
    print()

    print("--- Прогресс обучения ---\n")

    for epoch in range(n_epochs):
        real_data = generate_real_data(batch_size, data_dim)
        metrics = gan.train_step(real_data)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: "
                  f"G_loss={metrics['g_loss']:.4f} | "
                  f"D_loss={metrics['d_loss']:.4f} | "
                  f"D_acc={metrics['d_acc']:.1f}% | "
                  f"Fake_ratio={metrics['fake_ratio']:.4f}")

    print("\n--- Финальные метрики ---\n")
    print(f"  G loss (финальная): {gan.history['g_loss'][-1]:.4f}")
    print(f"  D loss (финальная): {gan.history['d_loss'][-1]:.4f}")
    print(f"  D accuracy (финальная): {gan.history['d_acc'][-1]:.1f}%")
    print(f"  Fake ratio (финальная): {gan.history['fake_ratio'][-1]:.4f}")

    # Генерация примеров
    print("\n--- Сгенерированные данные (после обучения) ---\n")
    for i in range(5):
        z = gan.generate_noise()
        generated, _ = gan.G.forward(z)
        prob, _ = gan.D.forward(generated)
        g_str = ", ".join(f"{v:+.3f}" for v in generated)
        print(f"  Sample {i}: [{g_str}]  D(real?)={prob:.4f}")


# ============================================================
# ДЕМО 4: ЭВОЛЮЦИЯ ГЕНЕРАТОРА
# ============================================================

def demo4_evolution():
    print_separator("DEMO 4: Эволюция генератора")

    latent_dim = 4
    data_dim = 2  # 2D для наглядности
    gan = GAN(latent_dim, data_dim, lr=0.02)

    n_epochs = 100
    batch_size = 16
    checkpoints = [1, 10, 25, 50, 75, 100]

    print(f"\n  Параметры:")
    print(f"    Размерность данных: {data_dim}D (для визуализации)")
    print(f"    Эпох: {n_epochs}")
    print(f"    Чекпоинты: {checkpoints}")
    print()

    print("--- Эволюция генератора ---\n")

    for epoch in range(n_epochs):
        real_data = generate_real_data(batch_size, data_dim)
        gan.train_step(real_data)

        if (epoch + 1) in checkpoints:
            # Генерируем 8 сэмплов для статистики
            samples = []
            for _ in range(8):
                z = gan.generate_noise()
                out, _ = gan.G.forward(z)
                samples.append(out)

            # Статистика
            means = [sum(s[d] for s in samples) / len(samples) for d in range(data_dim)]
            stds = [math.sqrt(sum((s[d] - means[d]) ** 2 for s in samples) / len(samples))
                    for d in range(data_dim)]

            # D accuracy на контрольных данных
            test_real = generate_real_data(10, data_dim)
            test_fake = []
            for _ in range(10):
                z = gan.generate_noise()
                fake, _ = gan.G.forward(z)
                test_fake.append(fake)

            correct = sum(1 for s in test_real if gan.D.classify(s) == "REAL")
            correct += sum(1 for s in test_fake if gan.D.classify(s) == "FAKE")
            acc = correct / 20 * 100

            print(f"  Epoch {epoch + 1:3d}:")
            print(f"    Статистика выхода: "
                  f"mean=[{means[0]:+.3f}, {means[1]:+.3f}], "
                  f"std=[{stds[0]:.3f}, {stds[1]:.3f}]")
            print(f"    D accuracy: {acc:.1f}% | "
                  f"G_loss: {gan.history['g_loss'][-1]:.4f} | "
                  f"Fake_ratio: {gan.history['fake_ratio'][-1]:.4f}")

            # Показываем 3 сэмпл
            for i in range(3):
                vals = ", ".join(f"{v:+.3f}" for v in samples[i])
                print(f"    Пример {i}: [{vals}]")
            print()

    # Итоговая эволюция
    print("--- Сравнение: Начало vs Конец ---\n")

    # Начальные сэмплы (без обучения)
    gen_fresh = Generator(latent_dim, data_dim, lr=0.02)
    print("  Начальные (случайные) генерации:")
    for i in range(3):
        z = [random.gauss(0, 1) for _ in range(latent_dim)]
        out, _ = gen_fresh.forward(z)
        vals = ", ".join(f"{v:+.3f}" for v in out)
        print(f"    [{vals}]")

    print("\n  Обученные генерации:")
    for i in range(3):
        z = gan.generate_noise()
        out, _ = gan.G.forward(z)
        vals = ", ".join(f"{v:+.3f}" for v in out)
        print(f"    [{vals}]")

    # Целевое распределение
    real_data = generate_real_data(100, data_dim)
    real_means = [sum(s[d] for s in real_data) / len(real_data) for d in range(data_dim)]
    print(f"\n  Целевое распределение (реальные данные):")
    print(f"    mean = [{real_means[0]:+.3f}, {real_means[1]:+.3f}]")


# ============================================================
# ГЛАВНЫЙ ЗАПУСК
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Generative Adversarial Network (GAN)")
    print("  Самодостаточная реализация на чистом Python")
    print("=" * 60)

    demo1_generator()
    demo2_discriminator()
    demo3_adversarial()
    demo4_evolution()

    print()
    print("=" * 60)
    print("  Все демонстрации завершены!")
    print("=" * 60)
