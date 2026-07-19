"""206 — A/B Testing for ML: статистическое тестирование, multi-armed bandits

Темы:
  1. Statistical Testing (проверка гипотез, p-value, доверительные интервалы)
  2. A/B Test Design (размер выборки, длительность, рандомизация)
  3. Multi-Armed Bandits (epsilon-greedy, UCB, Thompson Sampling)
  4. Online Evaluation (interleaving, shadow testing, canary releases)

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

# ============================================================================
# Демо 1: Statistical Testing — гипотезы, p-value, доверительные интервалы
# ============================================================================

def demo_statistical_testing():
    """Демонстрация статистических тестов: гипотезы, p-value, CI."""
    print("=" * 70)
    print("ДЕМО 1: Statistical Testing — гипотезы, p-value, доверительные интервалы")
    print("=" * 70)

    # --- 1.1 Z-test для пропорций (конверсия) ---
    print("\n--- 1.1 Z-test для пропорций (конверсия A/B) ---")
    # Формула: Z = (p_A - p_B) / sqrt(p*(1-p) * (1/n_A + 1/n_B))
    # где p = (x_A + x_B) / (n_A + n_B) — объединённая пропорция

    n_a = 5000  # контроль
    n_b = 5000  # вариант
    conv_a = 350  # конверсии в контрольной
    conv_b = 400  # конверсии в варианте

    p_a = conv_a / n_a  # 0.07
    p_b = conv_b / n_b  # 0.08

    # Объединённая пропорция
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    # Стандартная ошибка
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))

    # Z-статистика
    z_stat = (p_b - p_a) / se

    # Приближённое p-value через функцию нормального распределения
    # Используем аппроксимацию: Φ(z) ≈ 1 - φ(z) * (a1*t + a2*t^2 + a3*t^3)
    # где t = 1 / (1 + 0.2316419 * |z|), φ(z) = e^(-z²/2) / sqrt(2π)
    def normal_cdf(z):
        """Аппроксимация CDF стандартного нормального распределения."""
        if z < 0:
            return 1 - normal_cdf(-z)
        a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        t = 1 / (1 + 0.2316419 * z)
        phi = math.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
        return 1 - phi * (a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5)

    p_value_two = 2 * (1 - normal_cdf(abs(z_stat)))  # двухсторонний тест
    p_value_one = 1 - normal_cdf(z_stat)  # односторонний тест (B > A)

    print(f"Контроль (A): n={n_a}, конверсия={p_a:.3f} ({conv_a})")
    print(f"Вариант  (B): n={n_b}, конверсия={p_b:.3f} ({conv_b})")
    print(f"Объединённая пропорция: p={p_pool:.4f}")
    print(f"Стандартная ошибка: SE={se:.6f}")
    print(f"Z-статистика: Z={z_stat:.4f}")
    print(f"p-value (двухсторонний): {p_value_two:.6f}")
    print(f"p-value (односторонний): {p_value_one:.6f}")
    print(f"Решение (α=0.05): {'ОТВЕРГАЕМ H0 — разница значима' if p_value_two < 0.05 else 'НЕ ОТВЕРГАЕМ H0'}")
    print(f"Относительное улучшение: {(p_b - p_a) / p_a * 100:.1f}%")

    # --- 1.2 Доверительный интервал ---
    print("\n--- 1.2 Доверительный интервал для разницы пропорций ---")
    # Формула: CI = (p_B - p_A) ± z_α/2 * SE
    z_95 = 1.96  # z для 95% CI
    z_99 = 2.576  # z для 99% CI

    diff = p_b - p_a
    ci_95_lower = diff - z_95 * se
    ci_95_upper = diff + z_95 * se
    ci_99_lower = diff - z_99 * se
    ci_99_upper = diff + z_99 * se

    print(f"Разница (B - A): {diff:.4f}")
    print(f"95% CI: [{ci_95_lower:.4f}, {ci_95_upper:.4f}]")
    print(f"99% CI: [{ci_99_lower:.4f}, {ci_99_upper:.4f}]")
    print(f"Интерпретация: {'Улучшение значимо' if ci_95_lower > 0 else 'Различие незначимо'} "
          f"(95% CI не содержит 0: {'да' if ci_95_lower > 0 else 'нет'})")

    # --- 1.3 T-test для непрерывных метрик ---
    print("\n--- 1.3 T-test для средних (непрерывная метрика) ---")
    random.seed(42)
    # Время на сайте: контроль vs вариант
    control_time = [random.gauss(120, 30) for _ in range(200)]
    variant_time = [random.gauss(128, 32) for _ in range(200)]

    mean_c = statistics.mean(control_time)
    mean_v = statistics.mean(variant_time)
    std_c = statistics.stdev(control_time)
    std_v = statistics.stdev(variant_time)
    n_c = len(control_time)
    n_v = len(variant_time)

    # Welch's t-test
    se_welch = math.sqrt(std_c**2/n_c + std_v**2/n_v)
    t_stat = (mean_v - mean_c) / se_welch

    # Степени свободы (Welch-Satterthwaite)
    num = (std_c**2/n_c + std_v**2/n_v) ** 2
    den = (std_c**2/n_c)**2 / (n_c - 1) + (std_v**2/n_v)**2 / (n_v - 1)
    df = num / den

    # Аппроксимация p-value для t-распределения
    # Используем нормальную аппроксимацию для больших df
    p_t = 2 * (1 - normal_cdf(abs(t_stat)))

    print(f"Контроль: mean={mean_c:.2f}с, std={std_c:.2f}, n={n_c}")
    print(f"Вариант:  mean={mean_v:.2f}с, std={std_v:.2f}, n={n_v}")
    print(f"Разница: {mean_v - mean_c:+.2f}с ({(mean_v-mean_c)/mean_c*100:+.1f}%)")
    print(f"Welch t-statistic: {t_stat:.4f}")
    print(f"Степени свободы: {df:.1f}")
    print(f"p-value (двухсторонний): {p_t:.6f}")
    print(f"Решение (α=0.05): {'ОТВЕРГАЕМ H0' if p_t < 0.05 else 'НЕ ОТВЕРГАЕМ H0'}")

    # --- 1.4 Множественные сравнения (Bonferroni) ---
    print("\n--- 1.4 Коррекция Бонферрони (множественные тесты) ---")
    metrics = ["CTR", "конверсия", "время_на_сайте", "доход_на_пользователя"]
    p_values_raw = [0.021, 0.043, 0.127, 0.038]
    alpha = 0.05
    n_tests = len(metrics)

    alpha_bonf = alpha / n_tests  # скорректированный уровень значимости
    print(f"Количество тестов: {n_tests}")
    print(f"Базовый α: {alpha}")
    print(f"Скорректированный α (Бонферрони): {alpha_bonf:.4f}")
    print()
    print(f"{'Метрика':25s} | {'p-value':>8} | {'Значимо (raw)':>14} | {'Значимо (Bonf)':>14}")
    print("-" * 70)
    for metric, pval in zip(metrics, p_values_raw):
        raw_sig = "Да" if pval < alpha else "Нет"
        bonf_sig = "Да" if pval < alpha_bonf else "Нет"
        print(f"{metric:25s} | {pval:8.4f} | {raw_sig:>14} | {bonf_sig:>14}")

    # Метод Холма (более мощный)
    print("\nМетод Холма (степенчатая коррекция):")
    indexed = sorted(enumerate(p_values_raw), key=lambda x: x[1])
    for rank, (idx, pval) in enumerate(indexed, 1):
        holm_alpha = alpha / (n_tests - rank + 1)
        sig = "Да" if pval < holm_alpha else "Нет"
        print(f"  {metrics[idx]:25s}: p={pval:.4f}, α_holm={holm_alpha:.4f} -> {sig}")

    print("\n[OK] Statistical Testing — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 2: A/B Test Design — размер выборки, длительность, рандомизация
# ============================================================================

def demo_ab_test_design():
    """Демонстрация дизайна A/B тестов: sample size, duration, randomization."""
    print("=" * 70)
    print("ДЕМО 2: A/B Test Design — размер выборки, длительность, рандомизация")
    print("=" * 70)

    # --- 1. Расчёт размера выборки ---
    print("\n--- 2.1 Расчёт минимального размера выборки ---")
    # Формула: n = (z_α/2 + z_β)^2 * 2 * p * (1-p) / δ^2
    # где p — базовая пропорция, δ — ожидаемая разница

    def sample_size_two_proportions(p1, mde, alpha=0.05, power=0.8):
        """Расчёт размера выборки для двух пропорций."""
        p2 = p1 * (1 + mde)
        # z-значения
        z_alpha = 1.96  # для alpha=0.05 двухсторонний
        z_beta = 0.842   # для power=0.80
        # Формула
        numerator = (z_alpha + z_beta) ** 2 * (p1*(1-p1) + p2*(1-p2))
        denominator = (p2 - p1) ** 2
        return math.ceil(numerator / denominator)

    print("Параметры теста:")
    print(f"  Базовая конверсия (p): 10%")
    print(f"  Минимальное обнаруживаемое улучшение (MDE): 5%, 10%, 20%")
    print()
    print(f"{'MDE':>6} | {'Конверсия B':>12} | {'n на группу':>12} | {'Всего':>12}")
    print("-" * 50)
    for mde_pct in [5, 10, 20]:
        n = sample_size_two_proportions(0.10, mde_pct / 100)
        p2 = 0.10 * (1 + mde_pct / 100)
        print(f"{mde_pct:5d}% | {p2:11.3f} | {n:12d} | {2*n:12d}")

    # --- 2. Длительность теста ---
    print("\n--- 2.2 Расчёт длительности теста ---")
    daily_users = 10000  # ежедневные пользователи
    n_per_group = sample_size_two_proportions(0.10, 0.05)  # MDE=5%
    total_needed = 2 * n_per_group
    days_needed = math.ceil(total_needed / daily_users)

    print(f"Ежедневные пользователи: {daily_users}")
    print(f"Необходимый размер на группу: {n_per_group}")
    print(f"Всего необходимо: {total_needed}")
    print(f"Минимальная длительность: {days_needed} дней")
    print(f"Рекомендуемая длительность: {days_needed * 2} дней (учёт недельной сезонности)")

    # --- 3. Рандомизация (Hash-based) ---
    print("\n--- 2.3 Рандомизация пользователей (Hash-based) ---")
    def hash_bucket(user_id, experiment_name, n_buckets=100):
        """Хэширование user_id + experiment для детерминированного разбиения."""
        hash_input = f"{user_id}:{experiment_name}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return hash_val % n_buckets

    experiment_name = "recommendation_v2"
    users = [f"user_{i:04d}" for i in range(20)]
    groups = collections.Counter()
    print(f"Эксперимент: {experiment_name}")
    print(f"{'User':>12} | {'Hash % 100':>10} | {'Группа':>10}")
    print("-" * 40)
    for user in users:
        bucket = hash_bucket(user, experiment_name)
        group = "control" if bucket < 50 else "variant"
        groups[group] += 1
        print(f"{user:>12} | {bucket:>10} | {group:>10}")

    print(f"\nРаспределение: {dict(groups)}")
    print(f"Баланс: {abs(groups['control'] - groups['variant']) / sum(groups.values()) * 100:.1f}% "
          f"отклонение")

    # --- 4. Проверка_SRM (Sample Ratio Mismatch) ---
    print("\n--- 2.4 Sample Ratio Mismatch (SRM) проверка ---")
    # Хи-квадрат тест: O_observed vs O_expected
    observed_control = 4980
    observed_variant = 5020
    total = observed_control + observed_variant
    expected_ratio = 0.5
    expected_control = total * expected_ratio
    expected_variant = total * expected_ratio

    chi_sq = ((observed_control - expected_control)**2 / expected_control +
              (observed_variant - expected_variant)**2 / expected_variant)

    # df = 1, при alpha=0.05 критическое значение = 3.841
    critical_value = 3.841

    print(f"Ожидаемое соотношение: 50/50")
    print(f"Наблюдаемое: control={observed_control}, variant={observed_variant}")
    print(f"Ожидаемое: control={expected_control:.0f}, variant={expected_variant:.0f}")
    print(f"Chi-квадрат: {chi_sq:.4f}")
    print(f"Критическое значение (α=0.05, df=1): {critical_value}")
    print(f"SRM: {'ЕСТЬ (проблема с рандомизацией!)' if chi_sq > critical_value else 'Нет (норма)'}")

    print("\n[OK] A/B Test Design — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 3: Multi-Armed Bandits — epsilon-greedy, UCB, Thompson Sampling
# ============================================================================

def demo_multi_armed_bandits():
    """Демонстрация многоруких бандитов: epsilon-greedy, UCB, Thompson."""
    print("=" * 70)
    print("ДЕМО 3: Multi-Armed Bandits — epsilon-greedy, UCB, Thompson Sampling")
    print("=" * 70)

    # Определяем бандитов с известными вероятностями выигрыша
    random.seed(42)
    n_arms = 5
    true_probs = [0.15, 0.30, 0.45, 0.25, 0.35]  # истинные CTR
    arm_names = ["Баннер А", "Видео B", "Текст C", "Карусель D", "Попап E"]

    print(f"Истинные вероятности выигрыша:")
    for name, prob in zip(arm_names, true_probs):
        print(f"  {name}: {prob:.2f}")
    print(f"Оптимальный выбор: {arm_names[true_probs.index(max(true_probs))]} "
          f"(p={max(true_probs):.2f})")

    # --- 1. Epsilon-Greedy ---
    print("\n--- 3.1 Epsilon-Greedy стратегия ---")
    epsilon = 0.1
    n_rounds = 1000

    q_values = [0.0] * n_arms  # оценки ценности
    n_pulls = [0] * n_arms     # количество pulls
    total_reward_eg = 0
    best_arm_count = 0

    for t in range(n_rounds):
        # Выбор действия
        if random.random() < epsilon:
            # Исследование: случайный выбор
            arm = random.randint(0, n_arms - 1)
        else:
            # Использование: лучший по оценкам
            arm = max(range(n_arms), key=lambda a: q_values[a])

        # Получение награды
        reward = 1 if random.random() < true_probs[arm] else 0
        total_reward_eg += reward

        # Обновление оценки (incremental mean)
        n_pulls[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / n_pulls[arm]

        if arm == true_probs.index(max(true_probs)):
            best_arm_count += 1

    print(f"Epsilon: {epsilon}, Раундов: {n_rounds}")
    print(f"Итоговые оценки:")
    for i in range(n_arms):
        print(f"  {arm_names[i]:12s}: Q={q_values[i]:.4f}, pulls={n_pulls[i]}")
    print(f"Выбор оптимального баннера: {best_arm_count}/{n_rounds} "
          f"({best_arm_count/n_rounds*100:.1f}%)")
    print(f"Средняя награда: {total_reward_eg/n_rounds:.4f}")

    # --- 2. UCB (Upper Confidence Bound) ---
    print("\n--- 3.2 UCB1 стратегия ---")
    # Формула: UCB1(a) = Q(a) + sqrt(2 * ln(t) / N(a))
    ucb_q = [0.0] * n_arms
    ucb_pulls = [0] * n_arms
    total_reward_ucb = 0

    # Инициализация: pull каждый arm один раз
    for arm in range(n_arms):
        reward = 1 if random.random() < true_probs[arm] else 0
        ucb_pulls[arm] = 1
        ucb_q[arm] = reward
        total_reward_ucb += reward

    for t in range(n_arms + 1, n_rounds + 1):
        # Вычисляем UCB1 для каждого действия
        ucb_values = []
        for a in range(n_arms):
            if ucb_pulls[a] == 0:
                ucb_values.append(float('inf'))
            else:
                exploration = math.sqrt(2 * math.log(t) / ucb_pulls[a])
                ucb_values.append(ucb_q[a] + exploration)

        arm = ucb_values.index(max(ucb_values))
        reward = 1 if random.random() < true_probs[arm] else 0
        total_reward_ucb += reward
        ucb_pulls[arm] += 1
        ucb_q[arm] += (reward - ucb_q[arm]) / ucb_pulls[arm]

    print(f"Раундов: {n_rounds}")
    print(f"Итоговые оценки:")
    for i in range(n_arms):
        print(f"  {arm_names[i]:12s}: Q={ucb_q[i]:.4f}, pulls={ucb_pulls[i]}")
    print(f"Средняя награда: {total_reward_ucb/n_rounds:.4f}")

    # --- 3. Thompson Sampling ---
    print("\n--- 3.3 Thompson Sampling ---")
    # Модель: Beta(α, β) для каждой руки
    alpha_params = [1.0] * n_arms  # начальные α (успехи + 1)
    beta_params = [1.0] * n_arms   # начальные β (неудачи + 1)
    total_reward_ts = 0
    ts_history = []

    for t in range(n_rounds):
        # Сэмплируем из Beta для каждого действия
        sampled = [random.betavariate(alpha_params[a], beta_params[a])
                   for a in range(n_arms)]
        arm = sampled.index(max(sampled))

        reward = 1 if random.random() < true_probs[arm] else 0
        total_reward_ts += reward

        # Обновление: успех → α+1, неудача → β+1
        if reward == 1:
            alpha_params[arm] += 1
        else:
            beta_params[arm] += 1

        ts_history.append((t, arm, reward))

    print(f"Раундов: {n_rounds}")
    print(f"Итоговые параметры Beta(α, β):")
    for i in range(n_arms):
        mean_est = alpha_params[i] / (alpha_params[i] + beta_params[i])
        print(f"  {arm_names[i]:12s}: Beta({alpha_params[i]:.0f}, {beta_params[i]:.0f}), "
              f"mean={mean_est:.4f}")
    print(f"Средняя награда: {total_reward_ts/n_rounds:.4f}")

    # --- 4. Сравнение стратегий ---
    print("\n--- 3.4 Сравнение всех стратегий ---")
    def run_strategy(strategy_name, n_rounds=1000, **kwargs):
        """Запуск стратегии бандита и возврат cumulative reward."""
        random.seed(42)
        q = [0.0] * n_arms
        pulls = [0] * n_arms
        total = 0
        cumulative = []

        for t in range(n_rounds):
            if strategy_name == "random":
                arm = random.randint(0, n_arms - 1)
            elif strategy_name == "greedy":
                arm = max(range(n_arms), key=lambda a: q[a])
            elif strategy_name == "epsilon_greedy":
                eps = kwargs.get("epsilon", 0.1)
                arm = random.randint(0, n_arms - 1) if random.random() < eps else \
                      max(range(n_arms), key=lambda a: q[a])
            elif strategy_name == "ucb":
                if t < n_arms:
                    arm = t
                else:
                    ucb_vals = [q[a] + math.sqrt(2*math.log(t)/pulls[a]) if pulls[a]>0
                                else float('inf') for a in range(n_arms)]
                    arm = ucb_vals.index(max(ucb_vals))

            reward = 1 if random.random() < true_probs[arm] else 0
            total += reward
            pulls[arm] += 1
            q[arm] += (reward - q[arm]) / pulls[arm]
            cumulative.append(total)

        return cumulative

    strategies = {
        "Случайный": run_strategy("random"),
        "Жадный": run_strategy("greedy"),
        "ε-Greedy (ε=0.1)": run_strategy("epsilon_greedy", epsilon=0.1),
        "UCB1": run_strategy("ucb"),
    }

    print(f"{'Стратегия':20s} | {'Награда (1000)':>14} | {'Средняя':>10} | {'% от оптимума':>14}")
    print("-" * 65)
    optimal_avg = max(true_probs)
    for name, cum in strategies.items():
        total_r = cum[-1]
        avg_r = total_r / len(cum)
        pct = avg_r / optimal_avg * 100
        print(f"{name:20s} | {total_r:14d} | {avg_r:10.4f} | {pct:13.1f}%")

    print("\n[OK] Multi-Armed Bandits — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 4: Online Evaluation — interleaving, shadow testing, canary
# ============================================================================

def demo_online_evaluation():
    """Демонстрация онлайн-оценки: interleaving, shadow testing, canary."""
    print("=" * 70)
    print("ДЕМО 4: Online Evaluation — interleaving, shadow testing, canary")
    print("=" * 70)

    # --- 1. Interleaving ---
    print("\n--- 4.1 Interleaving (Team Draft) ---")
    # Метод: перемешиваем результаты двух алгоритмов
    random.seed(42)

    def team_draft_interleave(rankings_a, rankings_b, n=5):
        """Team Draft Interleaving: перемешиваем топ-N из двух ранжирований."""
        interleaved = []
        teams = {"A": 0, "B": 0}
        pointers = {"A": 0, "B": 0}
        last_team = None

        while len(interleaved) < n:
            # Чередуем: если последний был от A, берём от B, и наоборот
            if last_team is None:
                team = random.choice(["A", "B"])
            else:
                team = "B" if last_team == "A" else "A"

            # Берём следующий неиспользованный документ из выбранного рейтинга
            ranking = rankings_a if team == "A" else rankings_b
            while pointers[team] < len(ranking) and ranking[pointers[team]] in interleaved:
                pointers[team] += 1

            if pointers[team] < len(ranking):
                doc = ranking[pointers[team]]
                interleaved.append(doc)
                teams[team] += 1
                last_team = team
                pointers[team] += 1
            else:
                # Если закончились документы, переключаемся
                last_team = team

        return interleaved, teams

    # Два алгоритма ранжирования (ID документов)
    ranking_a = ["doc_12", "doc_5", "doc_8", "doc_3", "doc_15", "doc_7", "doc_2"]
    ranking_b = ["doc_5", "doc_12", "doc_3", "doc_20", "doc_8", "doc_1", "doc_9"]

    interleaved, team_sizes = team_draft_interleave(ranking_a, ranking_b, n=5)
    print(f"Ранжирование A: {ranking_a[:5]}")
    print(f"Ранжирование B: {ranking_b[:5]}")
    print(f"Interleaved:    {interleaved}")
    print(f"Вклад команд:   A={team_sizes['A']}, B={team_sizes['B']}")

    # Симулируем клики
    clicks = random.sample(interleaved[:3], min(2, len(interleaved[:3])))
    winner = None
    for doc in clicks:
        if doc in ranking_a[:3]:
            print(f"  Клик на {doc} -> A получает очко")
        if doc in ranking_b[:3]:
            print(f"  Клик на {doc} -> B получает очко")

    # --- 2. Shadow Testing ---
    print("\n--- 4.2 Shadow Testing (параллельный запуск) ---")
    # Запускаем модель-кандидат параллельно с продакшн, но не показываем результат

    random.seed(42)
    n_requests = 100
    shadow_results = {"production": {"correct": 0, "total": 0},
                      "shadow": {"correct": 0, "total": 0}}

    for i in range(n_requests):
        # Имитация: production модель точнее на 60%, shadow на 55%
        prod_correct = random.random() < 0.60
        shadow_correct = random.random() < 0.55

        shadow_results["production"]["total"] += 1
        shadow_results["shadow"]["total"] += 1
        shadow_results["production"]["correct"] += int(prod_correct)
        shadow_results["shadow"]["correct"] += int(shadow_correct)

    prod_acc = shadow_results["production"]["correct"] / shadow_results["production"]["total"]
    shadow_acc = shadow_results["shadow"]["correct"] / shadow_results["shadow"]["total"]

    print(f"Запросов: {n_requests}")
    print(f"Production модель: accuracy = {prod_acc:.3f}")
    print(f"Shadow модель:     accuracy = {shadow_acc:.3f}")
    print(f"Разница: {shadow_acc - prod_acc:+.3f}")
    print(f"Рекомендация: {'ЗАМЕНИТЬ' if shadow_acc > prod_acc else 'ОСТАВИТЬ'} production модель")

    # Сравнение латентности
    prod_latencies = [random.expovariate(1/80) + 20 for _ in range(n_requests)]
    shadow_latencies = [random.expovariate(1/90) + 25 for _ in range(n_requests)]

    print(f"\nЛатентность:")
    print(f"  Production: avg={statistics.mean(prod_latencies):.1f}мс, "
          f"p99={sorted(prod_latencies)[int(0.99*n_requests)]:.1f}мс")
    print(f"  Shadow:     avg={statistics.mean(shadow_latencies):.1f}мс, "
          f"p99={sorted(shadow_latencies)[int(0.99*n_requests)]:.1f}мс")

    # --- 3. Canary Release ---
    print("\n--- 4.3 Canary Release (ступенчатый rollout) ---")
    canary_stages = [
        {"traffic_pct": 1, "duration_hours": 1, "status": "completed", "errors": 2},
        {"traffic_pct": 5, "duration_hours": 2, "status": "completed", "errors": 8},
        {"traffic_pct": 25, "duration_hours": 24, "status": "completed", "errors": 45},
        {"traffic_pct": 50, "duration_hours": 24, "status": "in_progress", "errors": 30},
        {"traffic_pct": 100, "duration_hours": 0, "status": "pending", "errors": 0},
    ]

    print(f"{'Трафик':>8} | {'Длительность':>14} | {'Статус':>15} | {'Ошибки':>8} | {'Следующий шаг':>15}")
    print("-" * 75)
    for stage in canary_stages:
        status_map = {"completed": "завершено", "in_progress": "идёт", "pending": "ожидание"}
        next_step = "→" if stage["status"] != "pending" else "—"
        print(f"{stage['traffic_pct']:>7d}% | {stage['duration_hours']:>13d}ч | "
              f"{status_map[stage['status']]:>15} | {stage['errors']:>8d} | {next_step:>15}")

    print(f"\nКритерии для продвижения:")
    print(f"  - Error rate < 0.1%")
    print(f"  - Latency p99 < baseline * 1.5")
    print(f"  - Нет P1/P2 инцидентов")

    # --- 4. Online-Offline Correlation ---
    print("\n--- 4.4 Корреляция онлайн/оффлайн метрик ---")
    # Проверяем, согласуются ли оффлайн метрики с онлайн
    experiments = [
        {"name": "exp_001", "offline_auc": 0.85, "online_ctr": 0.12,
         "online_conv": 0.032, "status": "rolled_out"},
        {"name": "exp_002", "offline_auc": 0.87, "online_ctr": 0.14,
         "online_conv": 0.038, "status": "rolled_out"},
        {"name": "exp_003", "offline_auc": 0.82, "online_ctr": 0.10,
         "online_conv": 0.025, "status": "not_rolled_out"},
        {"name": "exp_004", "offline_auc": 0.88, "online_ctr": 0.09,
         "online_conv": 0.020, "status": "analysis_needed"},
        {"name": "exp_005", "offline_auc": 0.86, "online_ctr": 0.13,
         "online_conv": 0.035, "status": "rolled_out"},
    ]

    print(f"{'Эксперимент':>12} | {'Offline AUC':>11} | {'Online CTR':>11} | "
          f"{'Online Conv':>11} | {'Статус':>20}")
    print("-" * 75)
    for exp in experiments:
        print(f"{exp['name']:>12} | {exp['offline_auc']:11.3f} | {exp['online_ctr']:11.3f} | "
              f"{exp['online_conv']:11.3f} | {exp['status']:>20}")

    # Корреляция AUC vs CTR
    aucs = [e["offline_auc"] for e in experiments]
    ctrs = [e["online_ctr"] for e in experiments]

    # Pearson correlation
    n = len(aucs)
    mean_auc = sum(aucs) / n
    mean_ctr = sum(ctrs) / n
    cov = sum((aucs[i] - mean_auc) * (ctrs[i] - mean_ctr) for i in range(n)) / n
    std_auc = math.sqrt(sum((a - mean_auc)**2 for a in aucs) / n)
    std_ctr = math.sqrt(sum((c - mean_ctr)**2 for c in ctrs) / n)
    correlation = cov / (std_auc * std_ctr) if std_auc * std_ctr > 0 else 0

    print(f"\nКорреляция Offline AUC vs Online CTR: {correlation:.4f}")
    print(f"Интерпретация: "
          f"{'сильная' if abs(correlation) > 0.7 else 'умеренная' if abs(correlation) > 0.4 else 'слабая'} "
          f"{'положительная' if correlation > 0 else 'отрицательная'} корреляция")

    print("\n[OK] Online Evaluation — 4 подпримера выполнены.\n")


# ============================================================================
# Точка входа
# ============================================================================

if __name__ == "__main__":
    demo_statistical_testing()
    demo_ab_test_design()
    demo_multi_armed_bandits()
    demo_online_evaluation()
