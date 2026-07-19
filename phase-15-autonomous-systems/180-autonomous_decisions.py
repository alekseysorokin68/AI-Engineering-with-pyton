"""180 — Autonomous Decision Making: неопределённость, оценка рисков, ожидаемая полезность

Темы:
  1. Decision Under Uncertainty — expected utility, maximin, minimax regret
  2. Risk Assessment — risk vs uncertainty, VaR, scenario analysis
  3. Bayesian Decision Theory — prior, likelihood, posterior, expected value of information
  4. Sequential Decisions — dynamic programming, value iteration, policy evaluation

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# ──────────────────────────── 1. Decision Under Uncertainty ────────────────────────────

def demo_decision_under_uncertainty():
    """Ожидаемая полезность, maximin, minimax regret."""
    print("=" * 70)
    print("DEMO 1 — Decision Under Uncertainty")
    print("=" * 70)

    # --- 1a. Матрица решений: строки — стратегии, столбцы — состояния мира ---
    # Каждое значение — выигрыш (payoff) при данной стратегии и состоянии
    strategies = ["Инвестировать", "Сберегать", "Диверсифицировать"]
    states = ["Рост", "Стагнация", "Спад"]
    payoff_matrix = [
        [120, 20, -60],   # Инвестировать
        [30, 25, 20],     # Сберегать
        [70, 35, -10],    # Диверсифицировать
    ]

    print("\n--- 1a. Матрица выигрышей (payoff matrix) ---")
    print(f"{'Стратегия':<20}", end="")
    for s in states:
        print(f"{s:>15}", end="")
    print()
    for i, strat in enumerate(strategies):
        print(f"{strat:<20}", end="")
        for j in range(len(states)):
            print(f"{payoff_matrix[i][j]:>15}", end="")
        print()

    # --- 1b. Ожидаемая полезность (Expected Utility) ---
    # При неизвестных вероятностях используем равные вероятности (принцип Лапласа)
    # EU(a) = sum_j p(j) * u(a, j)
    print("\n--- 1b. Ожидаемая полезность (равные вероятности) ---")
    equal_probs = [1 / len(states)] * len(states)
    for i, strat in enumerate(strategies):
        eu = sum(equal_probs[j] * payoff_matrix[i][j] for j in range(len(states)))
        print(f"  EU({strat}) = {' + '.join(f'1/3×({payoff_matrix[i][j]})' for j in range(len(states)))} = {eu:.2f}")

    best_eu_idx = max(range(len(strategies)), key=lambda i: sum(equal_probs[j] * payoff_matrix[i][j] for j in range(len(states))))
    print(f"  → Оптимальная стратегия по критерию Лапласа: {strategies[best_eu_idx]}")

    # --- 1c. Критерий Wald (maximin) ---
    # Максимизируем минимальный возможный выигрыш
    print("\n--- 1c. Критерий Wald (maximin) ---")
    for i, strat in enumerate(strategies):
        min_val = min(payoff_matrix[i])
        print(f"  min({strat}) = {min_val}")
    best_maximin_idx = max(range(len(strategies)), key=lambda i: min(payoff_matrix[i]))
    print(f"  → Оптимальная стратегия: {strategies[best_maximin_idx]} "
          f"(maximin = {min(payoff_matrix[best_maximin_idx])})")

    # --- 1d. Minimax Regret ---
    # Шкала сожаления: regret(i,j) = max_k payoff(k,j) - payoff(i,j)
    print("\n--- 1d. Minimax Regret ---")
    col_maxs = [max(payoff_matrix[i][j] for i in range(len(strategies))) for j in range(len(states))]
    regret_matrix = []
    for i in range(len(strategies)):
        row = [col_maxs[j] - payoff_matrix[i][j] for j in range(len(states))]
        regret_matrix.append(row)

    print("  Матрица сожалений (regret):")
    print(f"  {'Стратегия':<20}", end="")
    for s in states:
        print(f"{s:>15}", end="")
    print(f"{'Макс. сожаление':>20}")
    for i, strat in enumerate(strategies):
        print(f"  {strat:<18}", end="")
        for j in range(len(states)):
            print(f"{regret_matrix[i][j]:>15}", end="")
        print(f"{max(regret_matrix[i]):>20}")

    best_minimax_idx = min(range(len(strategies)), key=lambda i: max(regret_matrix[i]))
    print(f"  → Оптимальная стратегия: {strategies[best_minimax_idx]} "
          f"(minimax regret = {max(regret_matrix[best_minimax_idx])})")


# ──────────────────────────── 2. Risk Assessment ────────────────────────────

def demo_risk_assessment():
    """VaR, анализ сценариев, разница риск vs неопределённость."""
    print("\n\n" + "=" * 70)
    print("DEMO 2 — Risk Assessment")
    print("=" * 70)

    # --- 2a. Разница между риском и неопределённостью (Ф. Найт) ---
    print("\n--- 2a. Risk vs Uncertainty (Фрэнк Найт) ---")
    print("  РИСК:      известны вероятности исходов (бросок игральной кости)")
    print("  НЕОПРЕДЕЛЁННОСТЬ: вероятности неизвестны (война, изобретение)")
    print("  → При риске можно рассчитать ожидаемую полезность напрямую")
    print("  → При неопределённости нужен другой подход (maximin, robust decision)")

    # --- 2b. Value at Risk (VaR) ---
    # VaR_α = - quantile(returns, α)
    print("\n--- 2b. Value at Risk (VaR) ---")
    # Генерируем распределение доходностей портфеля
    random.seed(42)
    n_simulations = 10000
    portfolio_returns = []
    for _ in range(n_simulations):
        # Модель: нормальное распределение с skewness
        r = random.gauss(-0.001, 0.02)
        # Добавляем тяжёлый хвост через occasional large loss
        if random.random() < 0.05:
            r += random.gauss(-0.05, 0.02)
        portfolio_returns.append(r)

    # Сортируем для вычисления квантилей
    sorted_returns = sorted(portfolio_returns)

    # Вычисляем VaR на разных уровнях доверия
    confidence_levels = [0.90, 0.95, 0.99]
    for cl in confidence_levels:
        idx = int((1 - cl) * len(sorted_returns))
        var_value = -sorted_returns[idx]
        print(f"  VaR_{int(cl * 100)}% = {var_value * 100:.2f}% "
              f"(с вероятностью {int((1 - cl) * 100)}% убыток превысит это значение)")

    # Conditional VaR (CVaR) — среднее потерь за VaR
    for cl in confidence_levels:
        idx = int((1 - cl) * len(sorted_returns))
        tail = sorted_returns[:idx]
        cvar = -sum(tail) / len(tail) if tail else 0
        print(f"  CVaR_{int(cl * 100)}% = {cvar * 100:.2f}% (среднее потерь в хвосте)")

    # --- 2c. Анализ сценариев ---
    print("\n--- 2c. Анализ сценариев (Scenario Analysis) ---")
    scenarios = {
        "Оптимистичный": {"probability": 0.25, "return": 0.15, "impact": "Быстрый рост рынка"},
        "Базовый":       {"probability": 0.50, "return": 0.05, "impact": "Умеренный рост"},
        "Пессимистичный": {"probability": 0.20, "return": -0.08, "impact": "Экономический спад"},
        "Кризис":        {"probability": 0.05, "return": -0.30, "impact": "Финансовый кризис"},
    }

    expected_return = 0
    variance = 0
    print(f"  {'Сценарий':<18}{'Вероятность':>12}{'Доходность':>14}{'Вклад':>14}")
    for name, data in scenarios.items():
        contribution = data["probability"] * data["return"]
        expected_return += contribution
        print(f"  {name:<18}{data['probability']:>12.0%}{data['return']:>14.1%}{contribution:>14.4f}")

    for name, data in scenarios.items():
        variance += data["probability"] * (data["return"] - expected_return) ** 2

    std_dev = math.sqrt(variance)
    print(f"\n  Ожидаемая доходность: {expected_return:.4f} ({expected_return * 100:.2f}%)")
    print(f"  Стандартное отклонение (риск): {std_dev:.4f} ({std_dev * 100:.2f}%)")
    print(f"  Коэффициент变异ации риска: {abs(std_dev / expected_return) if expected_return != 0 else float('inf'):.2f}")

    # --- 2d. Sensitivity analysis ---
    print("\n--- 2d. Чувствительный анализ (Sensitivity Analysis) ---")
    base_revenue = 1000000
    print(f"  Базовая выручка: ${base_revenue:,.0f}")
    print(f"\n  Варьируем один параметр, остальные фиксированы:")
    parameters = [
        ("Цена продажи", 100, 120, base_revenue),
        ("Объём продаж (шт.)", 5000, 6000, base_revenue),
        ("Переменные затраты", 40, 35, -base_revenue),
    ]

    for param_name, low, high, base in parameters:
        impact_low = (low - 100) / 100 * abs(base)
        impact_high = (high - 100) / 100 * abs(base)
        print(f"  {param_name}: {low} → {high} | "
              f"Влияние: [{impact_low:+,.0f}, {impact_high:+,.0f}]")


# ──────────────────────────── 3. Bayesian Decision Theory ────────────────────────────

def demo_bayesian_decision():
    """Байесовская теория решений: prior, likelihood, posterior, EVPI."""
    print("\n\n" + "=" * 70)
    print("DEMO 3 — Bayesian Decision Theory")
    print("=" * 70)

    # --- 3a. Prior, Likelihood, Posterior ---
    print("\n--- 3a. Prior → Likelihood → Posterior ---")
    # Задача: определить, есть ли у пациента редкое заболевание
    # Prior: P(B) = prior prevalence of disease
    prior_disease = 0.01  # 1% заболеваемость в популяции

    # Likelihood: P(+|B) и P(+|¬B) — чувствительность и ложная положительность
    sensitivity = 0.95     # P(+|Болен) = 95%
    false_positive = 0.05  # P(+|Здоров) = 5%

    # По Байесу: P(B|+) = P(+|B) * P(B) / P(+)
    p_positive = sensitivity * prior_disease + false_positive * (1 - prior_disease)
    posterior_disease = (sensitivity * prior_disease) / p_positive

    print(f"  Prior:     P(Болен) = {prior_disease:.3f} ({prior_disease * 100:.1f}%)")
    print(f"  Likelihood: P(+|Болен) = {sensitivity:.2f}, P(+|Здоров) = {false_positive:.2f}")
    print(f"  P(+) = P(+|Болен)·P(Болен) + P(+|Здоров)·P(Здоров)")
    print(f"       = {sensitivity}×{prior_disease} + {false_positive}×{1 - prior_disease} = {p_positive:.4f}")
    print(f"  Posterior: P(Болен|+) = {posterior_disease:.4f} ({posterior_disease * 100:.2f}%)")

    # --- 3b. Обновление при повторном тесте ---
    print("\n--- 3b. Обновление posterior при повторном тесте ---")
    # Используем предыдущий posterior как новый prior
    prior_2 = posterior_disease
    p_positive_2 = sensitivity * prior_2 + false_positive * (1 - prior_2)
    posterior_2 = (sensitivity * prior_2) / p_positive_2

    print(f"  Новый prior: P(Болен) = {prior_2:.4f}")
    print(f"  P(Болен|+,+) = {posterior_2:.4f} ({posterior_2 * 100:.2f}%)")
    print(f"  Два положительных теста значительно повышают уверенность!")

    # --- 3c. Оптимальное решение по минимальному ожидаемому убытку ---
    print("\n--- 3c. Оптимальное решение по Expected Loss ---")
    # Стоимость различных исходов
    cost_treatment = 500     # Стоимость лечения (если болен и лечим)
    cost_missed = 100000     # Пропущенный случай (болен, но не лечим)
    cost_unnecessary = 2000  # Лишнее лечение (здоров, но лечим)
    cost_true_negative = 0   # Здоров и не лечим — норма

    # Для двух решений: лечить или не лечить
    # Expected Loss(лечить) = P(здоров) × cost_unnecessary + P(болен) × cost_treatment
    # Expected Loss(не лечить) = P(здоров) × cost_true_negative + P(болен) × cost_missed
    p_healthy = 1 - posterior_disease
    loss_treat = p_healthy * cost_unnecessary + posterior_disease * cost_treatment
    loss_no_treat = p_healthy * cost_true_negative + posterior_disease * cost_missed

    print(f"  P(Болен|+) = {posterior_disease:.4f}, P(Здоров|+) = {p_healthy:.4f}")
    print(f"  Expected Loss(лечить)    = {p_healthy:.4f}×${cost_unnecessary} + {posterior_disease:.4f}×${cost_treatment}")
    print(f"                          = ${loss_treat:,.2f}")
    print(f"  Expected Loss(не лечить) = {p_healthy:.4f}×$0 + {posterior_disease:.4f}×${cost_missed:,}")
    print(f"                          = ${loss_no_treat:,.2f}")
    print(f"  → Решение: {'ЛЕЧИТЬ' if loss_treat < loss_no_treat else 'НЕ ЛЕЧИТЬ'} "
          f"(ожидаемый убыток: ${min(loss_treat, loss_no_treat):,.2f})")

    # --- 3d. Expected Value of Information (EVPI) ---
    print("\n--- 3d. Expected Value of Perfect Information (EVPI) ---")
    # EVPI = Expected Loss без информации - Expected Loss с информацией
    # Без информации: оптимальное решение based on prior
    loss_prior_treat = (1 - prior_disease) * cost_unnecessary + prior_disease * cost_treatment
    loss_prior_no_treat = (1 - prior_disease) * 0 + prior_disease * cost_missed
    best_prior = min(loss_prior_treat, loss_prior_no_treat)

    # С идеальной информацией: лечим только больных, здоровых — нет
    loss_perfect = prior_disease * cost_treatment + (1 - prior_disease) * 0

    evpi = best_prior - loss_perfect
    print(f"  Стоимость идеального теста: ${cost_treatment:,.0f} за процедуру")
    print(f"  Expected Loss (без теста, лучшее решение): ${best_prior:,.2f}")
    print(f"  Expected Loss (с идеальным тестом):        ${loss_perfect:,.2f}")
    print(f"  EVPI = ${evpi:,.2f}")
    print(f"  → Если точный тест стоит меньше ${evpi:,.2f}, его стоит провести!")


# ──────────────────────────── 4. Sequential Decisions ────────────────────────────

def demo_sequential_decisions():
    """Динамическое программирование, value iteration, policy evaluation."""
    print("\n\n" + "=" * 70)
    print("DEMO 4 — Sequential Decisions (Value Iteration)")
    print("=" * 70)

    # --- 4a. MDP: коротко определяем задачу ---
    print("\n--- 4a. Markov Decision Process (MDP) ---")
    # Простая сетка 3×3: агент移动 в сетке, цель — максимизировать награду
    # Состояния: (row, col), центральное состояние — цель (terminal)
    rows, cols = 3, 3
    goal = (1, 1)
    actions = ["up", "down", "left", "right"]
    # Дисконтирование
    gamma = 0.9

    print(f"  Сетка: {rows}×{cols}, цель: {goal}, γ = {gamma}")
    print(f"  Действия: {actions}")

    # Функция перехода:Deterministic movement with boundary checking
    def step(state, action):
        r, c = state
        if action == "up":    r -= 1
        elif action == "down":  r += 1
        elif action == "left":  c -= 1
        elif action == "right": c += 1
        # Ограничиваем границами сетки
        r = max(0, min(rows - 1, r))
        c = max(0, min(cols - 1, c))
        return (r, c)

    def reward(state):
        if state == goal:
            return 10  # Награда за достижение цели
        return -0.1    # Штаг за каждый шаг

    # --- 4b. Policy Evaluation ---
    print("\n--- 4b. Policy Evaluation (итеративная) ---")
    # Оцениваем случайную политику (равные вероятности для каждого действия)
    V = [[0.0] * cols for _ in range(rows)]

    def policy_evaluation(V, iterations=50):
        for _ in range(iterations):
            V_new = [[0.0] * cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if (r, c) == goal:
                        V_new[r][c] = 0  # Терминальное состояние
                        continue
                    # Среднее по всем действиям (случайная политика)
                    total = 0
                    for a in actions:
                        next_s = step((r, c), a)
                        total += reward(next_s) + gamma * V[next_s[0]][next_s[1]]
                    V_new[r][c] = total / len(actions)
            V = V_new
        return V

    V = policy_evaluation(V)
    print("  V(s) после 50 итераций (случайная политика):")
    print("  ┌─────┬──────────┬──────────┬──────────┐")
    print("  │     │ col=0    │ col=1    │ col=2    │")
    print("  ├─────┼──────────┼──────────┼──────────┤")
    for r in range(rows):
        print(f"  │ r={r} │", end="")
        for c in range(cols):
            print(f"{V[r][c]:>9.4f}│", end="")
        print()
    print("  └─────┴──────────┴──────────┴──────────┘")

    # --- 4c. Value Iteration ---
    print("\n--- 4c. Value Iteration ---")
    V2 = [[0.0] * cols for _ in range(rows)]
    policy = [[None] * cols for _ in range(rows)]

    n_iterations = 20
    for iteration in range(n_iterations):
        V_new = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if (r, c) == goal:
                    V_new[r][c] = 0
                    continue
                # Для каждого действия вычисляем Q(s,a)
                q_values = []
                for a in actions:
                    next_s = step((r, c), a)
                    q = reward(next_s) + gamma * V2[next_s[0]][next_s[1]]
                    q_values.append(q)
                # Берём максимум (жадная политика)
                V_new[r][c] = max(q_values)
                best_action_idx = q_values.index(max(q_values))
                policy[r][c] = actions[best_action_idx]
        V2 = V_new

    print(f"  V(s) после {n_iterations} итераций value iteration:")
    print("  ┌─────┬──────────┬──────────┬──────────┐")
    print("  │     │ col=0    │ col=1    │ col=2    │")
    print("  ├─────┼──────────┼──────────┼──────────┤")
    for r in range(rows):
        print(f"  │ r={r} │", end="")
        for c in range(cols):
            print(f"{V2[r][c]:>9.4f}│", end="")
        print()
    print("  └─────┴──────────┴──────────┴──────────┘")

    # --- 4d. Извлечение оптимальной политики ---
    print("\n--- 4d. Оптимальная политика (policy) ---")
    arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    print("  Оптимальные действия из каждого состояния:")
    print("  ┌─────┬──────────┬──────────┬──────────┐")
    print("  │     │ col=0    │ col=1    │ col=2    │")
    print("  ├─────┼──────────┼──────────┼──────────┤")
    for r in range(rows):
        print(f"  │ r={r} │", end="")
        for c in range(cols):
            if (r, c) == goal:
                print(f"   ЦЕЛЬ   │", end="")
            else:
                a = policy[r][c]
                print(f"  {arrows[a]:>6}   │", end="")
        print()
    print("  └─────┴──────────┴──────────┴──────────┘")

    # Симуляция: идём от (0,0) к цели
    print("\n  Симуляция пути от (0,0) к (1,1):")
    state = (0, 0)
    path = [state]
    for step_num in range(20):
        if state == goal:
            break
        a = policy[state[0]][state[1]]
        state = step(state, a)
        path.append(state)

    path_str = " → ".join(f"({r},{c})" for r, c in path)
    print(f"  {path_str}")
    print(f"  Длина пути: {len(path) - 1} шагов")


if __name__ == "__main__":
    demo_decision_under_uncertainty()
    demo_risk_assessment()
    demo_bayesian_decision()
    demo_sequential_decisions()
