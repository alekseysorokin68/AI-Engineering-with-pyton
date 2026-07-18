"""
87. Diffusion Models — основы.
=============================
Самодостаточный скрипт без внешних зависимостей.
Покрывает: forward process, reverse process, DDPM.
"""

import math
import random

random.seed(42)

# ============================================================
# Утилиты
# ============================================================

def randn():
    """Box-Muller — генерация стандартного нормального числа."""
    u1 = random.random()
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2.0 * math.pi * u2)

def randn_vec(n):
    """Вектор из n нормальных чисел."""
    return [randn() for _ in range(n)]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(dot(a, a))

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def scale(a, s):
    return [x * s for x in a]

def fmt_vec(v, width=8):
    """Форматирование вектора для вывода."""
    return "[" + ", ".join(f"{x:{width}.4f}" for x in v) + "]"

# ============================================================
# 1. Forward Process — добавление шума
# ============================================================

def forward_process(x0, t, betas):
    """
    q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Возвращает x_t для заданного шага t.
    """
    alpha_bars = []
    alpha_bar = 1.0
    for b in betas:
        alpha_bar *= (1.0 - b)
        alpha_bars.append(alpha_bar)

    alpha_bar_t = alpha_bars[t]
    noise = randn_vec(len(x0))

    x_t = add(scale(x0, math.sqrt(alpha_bar_t)),
              scale(noise, math.sqrt(1.0 - alpha_bar_t)))
    return x_t, noise, alpha_bar_t


def demo_forward_process():
    """Демо 1: Forward process — пошаговое добавление шума."""
    print("=" * 70)
    print("DEMO 1: Forward Process — добавление шума к данным")
    print("=" * 70)

    random.seed(42)

    # Оригинальный сигнал — 8-мерный вектор
    x0 = [0.5, -0.3, 0.8, 0.1, -0.6, 0.4, 0.2, -0.7]

    # 1000 шагов шума (стандарт DDPM)
    T = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = [beta_start + (beta_end - beta_start) * i / (T - 1) for i in range(T)]

    print(f"\nИсходный вектор x0 (8D):")
    print(f"  x0 = {fmt_vec(x0)}")

    steps_to_show = [0, 50, 100, 200, 500, 999]
    for t in steps_to_show:
        x_t, noise, alpha_bar = forward_process(x0, t, betas)
        noise_level = norm(noise) / math.sqrt(len(noise))
        print(f"\n  t={t:>4d} | alpha_bar={alpha_bar:.6f} | "
              f"шум (RMS)={noise_level:.4f}")
        print(f"         x_t = {fmt_vec(x_t)}")

    print("\n  Вывод: при t=0 сигнал чистый; при t=999 — полностью зашумлён.")
    print("  Forward process просто интерполирует между данными и шумом.\n")


# ============================================================
# 2. Noise Schedule — расписание шума
# ============================================================

def compute_schedules(T, beta_start=1e-4, beta_end=0.02):
    """
    Возвращает: betas, alphas, alpha_bars, sqrt_alpha_bars,
                sqrt_one_minus_alpha_bars.
    """
    betas = [beta_start + (beta_end - beta_start) * i / (T - 1)
             for i in range(T)]
    alphas = [1.0 - b for b in betas]

    alpha_bars = []
    ab = 1.0
    for a in alphas:
        ab *= a
        alpha_bars.append(ab)

    sqrt_ab = [math.sqrt(ab) for ab in alpha_bars]
    sqrt_one_minus_ab = [math.sqrt(1.0 - ab) for ab in alpha_bars]

    return betas, alphas, alpha_bars, sqrt_ab, sqrt_one_minus_ab


def demo_noise_schedule():
    """Демо 2: Анализ расписания шума."""
    print("=" * 70)
    print("DEMO 2: Noise Schedule — расписание шума (beta, alpha, alpha_bar)")
    print("=" * 70)

    T = 1000
    betas, alphas, alpha_bars, sqrt_ab, sqrt_one_minus_ab = compute_schedules(T)

    print(f"\nПараметры: T={T}, beta_start=1e-4, beta_end=0.02")
    print(f"\n{'t':>5s} | {'beta':>10s} | {'alpha':>10s} | {'alpha_bar':>10s} "
          f"| {'√α_bar':>8s} | {'√(1-α_bar)':>10s}")
    print("-" * 65)

    checkpoints = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    for t in checkpoints:
        print(f"{t:>5d} | {betas[t]:>10.6f} | {alphas[t]:>10.6f} | "
              f"{alpha_bars[t]:>10.6f} | {sqrt_ab[t]:>8.4f} | "
              f"{sqrt_one_minus_ab[t]:>10.4f}")

    print(f"\n  alpha_bar[T-1] = {alpha_bars[-1]:.6f} (→ 0, т.е. чистый шум)")
    print(f"  beta растёт линейно от {betas[0]:.4f} до {betas[-1]:.4f}")
    print(f"  alpha_bar падает монотонно от ~1.0 до ~0.0\n")

    # Сравнение расписаний
    print("  Сравнение: линейное vs косинусное расписание")
    T2 = 1000
    cos_betas, _, cos_ab, _, _ = compute_schedules(T2)

    def cosine_alpha_bar(t, T):
        s = 0.008
        return math.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2

    cos_schedule = [cosine_alpha_bar(t, T2) for t in range(T2)]

    print(f"\n  {'t':>5s} | {'linear α_bar':>12s} | {'cosine α_bar':>12s} | "
          f"{'разница':>10s}")
    print("  " + "-" * 48)
    for t in [0, 200, 400, 600, 800, 999]:
        diff = abs(alpha_bars[t] - cos_schedule[t])
        print(f"  {t:>5d} | {alpha_bars[t]:>12.6f} | {cos_schedule[t]:>12.6f} | "
              f"{diff:>10.6f}")

    print("\n  Линейное: быстрое начальное затухание.")
    print("  Косинусное: более равномерное — лучше для генерации.\n")


# ============================================================
# 3. Reverse Process — один шаг denoising
# ============================================================

def p_sample(x_t, t, betas, eps_model=None):
    """
    p(x_{t-1} | x_t) — обратный процесс DDPM.

    x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
              + sigma_t * z

    Если eps_model=None, используем «идеальный» шум (ground truth).
    """
    T = len(betas)
    alphas = [1.0 - b for b in betas]
    alpha_bars = []
    ab = 1.0
    for a in alphas:
        ab *= a
        alpha_bars.append(ab)

    if t == 0:
        # На последнем шаге нет добавления шума
        alpha_bar_t = alpha_bars[t] if t < T else 1.0
        alpha_t = alphas[t]
        eps_coeff = betas[t] / math.sqrt(1.0 - alpha_bar_t + 1e-12)
        x_prev = (1.0 / math.sqrt(alpha_t)) * add(x_t, scale(eps_model, -eps_coeff))
        return x_prev

    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev = alpha_bars[t - 1] if t > 0 else 1.0

    # Предсказание eps
    eps = eps_model

    eps_coeff = betas[t] / math.sqrt(1.0 - alpha_bar_t + 1e-12)
    mean = scale(add(x_t, scale(eps, -eps_coeff)), 1.0 / math.sqrt(alpha_t))

    sigma_t = math.sqrt(betas[t] * (1.0 - alpha_bar_prev) /
                        (1.0 - alpha_bar_t + 1e-12))
    z = randn_vec(len(x_t))

    x_prev = add(mean, scale(z, sigma_t))
    return x_prev


def demo_reverse_process():
    """Демо 3: Reverse process — пошаговый denoising."""
    print("=" * 70)
    print("DEMO 3: Reverse Process — один шаг denoising (p(x_{t-1}|x_t))")
    print("=" * 70)

    random.seed(42)

    # Оригинальные данные
    x0 = [0.5, -0.3, 0.8, 0.1, -0.6, 0.4, 0.2, -0.7]

    T = 1000
    betas, alphas, alpha_bars, _, _ = compute_schedules(T)

    # Зашиефровываем x0 → x_T (полностью зашумлённый)
    noise = randn_vec(len(x0))
    x_T = add(scale(x0, math.sqrt(alpha_bars[-1])),
              scale(noise, math.sqrt(1.0 - alpha_bars[-1])))

    print(f"\nИсходные данные:  x0    = {fmt_vec(x0)}")
    print(f"Зашумлённые:      x_T   = {fmt_vec(x_T)}")

    # Обратный процесс: x_T → x_0 (с «идеальным» eps = настоящий шум)
    # Восстанавливаем оригинальный шум ( seed совпадает )
    random.seed(42)
    # Пересоздадим тот же шум
    random.seed(99)
    true_noise = randn_vec(len(x0))

    print(f"\n  Шаг t=T-1 → t=T-2 (первый шаг обратного процесса):")
    x_t = list(x_T)

    # Делаем 5 шагов обратного процесса
    for step in range(5):
        t = T - 1 - step
        # «Идеальный» eps — тот же шум, что использовался в forward
        # (на практике это предсказывает нейросеть)
        eps_pred = true_noise  # имитация «идеального» предсказания

        x_prev = p_sample(x_t, t, betas, eps_model=eps_pred)

        # Также покажем что было бы без denoising (просто noise)
        x_noisy = add(scale(x_t, 0.999), scale(randn_vec(len(x0)), 0.01))

        print(f"    t={t:>4d} → t={t-1:>4d}: "
              f"x_{t-1} = {fmt_vec(x_prev)}")
        x_t = x_prev

    print(f"\n  После 5 шагов обратного процесса:")
    print(f"    Результат: {fmt_vec(x_t)}")
    print(f"    Оригинал:  {fmt_vec(x0)}")
    print(f"    Ошибка:    {norm(add(x_t, scale(x0, -1))):.4f}")

    print("\n  На практике eps_pred предсказывает нейросеть (UNet).")
    print("  Чем точнее предсказание eps, тем лучше генерация.\n")


# ============================================================
# 4. Сравнение noisy vs denoised
# ============================================================

def denoise_simple(x_t, t, betas, x0_true):
    """
    Упрощённый «идеальный» denoiser:
   他知道 x0 и может лучше предсказать eps.
    eps = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
    """
    alphas = [1.0 - b for b in betas]
    alpha_bars = []
    ab = 1.0
    for a in alphas:
        ab *= a
        alpha_bars.append(ab)

    alpha_bar_t = alpha_bars[t]
    alpha_t = alphas[t]

    # Извлекаем eps из forward process
    eps = scale(add(x_t, scale(x0_true, -math.sqrt(alpha_bar_t))),
                1.0 / math.sqrt(1.0 - alpha_bar_t + 1e-12))

    # Обратный шаг
    eps_coeff = betas[t] / math.sqrt(1.0 - alpha_bar_t + 1e-12)
    mean = scale(add(x_t, scale(eps, -eps_coeff)), 1.0 / math.sqrt(alpha_t))

    if t > 0:
        sigma_t = math.sqrt(betas[t])
        z = randn_vec(len(x_t))
        x_prev = add(mean, scale(z, sigma_t))
    else:
        x_prev = mean

    return x_prev, eps


def demo_noisy_vs_denoised():
    """Демо 4: Сравнение noisy vs denoised."""
    print("=" * 70)
    print("DEMO 4: Сравнение noisy vs denoised")
    print("=" * 70)

    random.seed(42)

    x0 = [0.5, -0.3, 0.8, 0.1, -0.6, 0.4, 0.2, -0.7]

    T = 1000
    betas, alphas, alpha_bars, sqrt_ab, sqrt_one_minus_ab = compute_schedules(T)

    print(f"\nИсходный вектор x0: {fmt_vec(x0)}")

    steps_to_test = [100, 300, 500, 700, 900]

    print(f"\n{'t':>5s} | {'α_bar':>8s} | {'SNR':>8s} | "
          f"{'noisy err':>9s} | {'denoised err':>12s} | {'улучшение':>10s}")
    print("-" * 72)

    for t in steps_to_test:
        # Forward: x0 → x_t
        random.seed(42)
        noise = randn_vec(len(x0))
        alpha_bar_t = alpha_bars[t]
        x_t = add(scale(x0, math.sqrt(alpha_bar_t)),
                  scale(noise, math.sqrt(1.0 - alpha_bar_t)))

        # Ошибка noisy (сравнение с x0)
        noisy_err = norm(add(x_t, scale(x0, -1))) / math.sqrt(len(x0))

        # Denoising с «идеальным» предсказанием eps
        random.seed(42)
        x_denoised, eps = denoise_simple(x_t, t, betas, x0)
        denoised_err = norm(add(x_denoised, scale(x0, -1))) / math.sqrt(len(x0))

        snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-12)
        improvement = (1.0 - denoised_err / (noisy_err + 1e-12)) * 100

        print(f"{t:>5d} | {alpha_bar_t:>8.4f} | {snr:>8.3f} | "
              f"{noisy_err:>9.4f} | {denoised_err:>12.4f} | "
              f"{improvement:>9.1f}%")

    print(f"\n  SNR = α_bar / (1-α_bar) — отношение сигнала к шуму.")
    print(f"  При t=100 (SNR≈1): denoising помогает чуть-чуть.")
    print(f"  При t=900 (SNR≈0.01): denoising помогает сильно.")

    # Демонстрация полного цикла: x_T → x_0 (много шагов)
    print(f"\n{'='*70}")
    print(f"Полный цикл генерации: x_T → x_0 (100 шагов)")
    print(f"{'='*70}")

    random.seed(42)
    noise = randn_vec(len(x0))
    x_T = add(scale(x0, math.sqrt(alpha_bars[-1])),
              scale(noise, math.sqrt(1.0 - alpha_bars[-1])))

    # Обратный процесс (100 шагов, имитируя идеальный denoiser)
    x_current = list(x_T)
    step_interval = T // 100

    for i in range(100):
        t = T - 1 - i * step_interval
        if t < 0:
            t = 0

        random.seed(42 + i)
        x_current, _ = denoise_simple(x_current, t, betas, x0)

        if i % 20 == 0 or i == 99:
            err = norm(add(x_current, scale(x0, -1))) / math.sqrt(len(x0))
            print(f"  Шаг {i:>3d} (t={t:>4d}): ошибка = {err:.4f} | "
                  f"x = {fmt_vec(x_current)}")

    final_err = norm(add(x_current, scale(x0, -1))) / math.sqrt(len(x0))
    print(f"\n  Финальная ошибка: {final_err:.6f}")
    print(f"  Итоговый вектор: {fmt_vec(x_current)}")
    print(f"  Оригинал:         {fmt_vec(x0)}")

    print("\n  DDPM — это итеративный процесс: 1000 шагов denoising.")
    print("  Нейросеть (UNet) учится предсказывать eps на каждом шаге.")
    print("  Генерация начинается с чистого шума x_T ~ N(0, I).\n")


# ============================================================
# Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DIFFUSION MODELS — основы (DDPM)")
    print("  Без внешних зависимостей")
    print("=" * 70 + "\n")

    demo_forward_process()
    demo_noise_schedule()
    demo_reverse_process()
    demo_noisy_vs_denoised()

    print("=" * 70)
    print("  РЕЗЮМЕ: Diffusion Models (DDPM)")
    print("=" * 70)
    print("""
  Forward process:
    q(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t) · I)
    Просто добавляем шум пропорционально timestep t.

  Reverse process:
    p(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ²_t · I)
    Нейросеть предсказывает eps → восстанавливает x_{t-1}.

  Обучение:
    L = E[||eps - eps_θ(x_t, t)||²]
    Минимизируем MSE между настоящим шумом и предсказанным.

  Генерация:
    x_T ~ N(0, I) → x_{T-1} → ... → x_0 (чистые данные)

  Ключевые уравнения:
    α_t = 1 - β_t
    ᾱ_t = ∏_{s=1}^{t} α_s
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)
""")
