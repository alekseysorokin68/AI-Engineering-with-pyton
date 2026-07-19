"""174 — World Models: предиктивные модели, симуляция, мысленная симуляция

Темы:
  1. Predictive Models (предсказание состояния, предсказание награды, модели переходов)
  2. Mental Simulation (планирование смотрением вперёд, концепции MCTS)
  3. Model Learning (обучение динамики из опыта, ошибка предсказания)
  4. World Model Planning (планирование в стиле Dreamer, оптимизация воображением)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import itertools

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ПРЕДИКТИВНЫЕ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def demo_predictive_models():
    """Предиктивные модели: предсказание состояния, награды и переходов."""
    print("=" * 70)
    print("ДЕМО 1: PREDICTIVE MODELS — предиктивные модели")
    print("=" * 70)

    # --- 1a. Простейшая модель перехода состояний ---
    print("\n--- 1a. Модель перехода (Transition Model) ---")
    # Модель: s' = f(s, a) — детерминированная среда
    # Переходы: (состояние, действие) -> новое_состояние
    transition = {
        (0, "left"): 1,   (0, "right"): 2,
        (1, "left"): 0,   (1, "right"): 3,
        (2, "left"): 0,   (2, "right"): 3,
        (3, "left"): 1,   (3, "right"): 2,
    }
    state = 0
    actions = ["right", "right", "left", "right"]
    print(f"Начальное состояние: s={state}")
    for a in actions:
        old_s = state
        state = transition[(state, a)]
        print(f"  s={old_s}, a={a} -> s'={state}")

    # --- 1b. Модель предсказания награды ---
    print("\n--- 1b. Модель предсказания награды (Reward Prediction) ---")
    # Награда зависит от состояния и действия
    rewards = {
        (0, "right"): 0.1,  (1, "right"): 0.5,  (2, "right"): 1.0, (3, "right"): -0.5,
        (0, "left"): 0.0,   (1, "left"): 0.0,   (2, "left"): 0.0,  (3, "left"): -1.0,
    }
    # Суммарная награда по траектории
    state = 0
    total_reward = 0.0
    trajectory = ["right", "right", "right"]
    print(f"Начальное состояние: s=0, планируемое действие: {trajectory}")
    for a in trajectory:
        r = rewards[(state, a)]
        total_reward += r
        state = transition[(state, a)]
        print(f"  s_prev={state if a == 'right' else '?'} a={a} r={r:.1f} R_total={total_reward:.1f}")
    print(f"Предсказанная суммарная награда: {total_reward:.1f}")

    # --- 1c. Вероятностная модель перехода ---
    print("\n--- 1c. Вероятностная модель перехода (Stochastic Transition) ---")
    # P(s'|s, a) — распределение вероятностей
    stochastic = {
        0: {0: 0.1, 1: 0.7, 2: 0.2},  # из состояния 0
        1: {1: 0.2, 3: 0.8},            # из состояния 1
        2: {2: 0.3, 3: 0.7},            # из состояния 2
    }
    # Сэмплируем 5 траекторий из состояния 0
    print("5 сэмплированных переходов из состояния 0:")
    for i in range(5):
        dist = stochastic[0]
        states_list = list(dist.keys())
        probs = list(dist.values())
        # Ручной сэмплинг
        r_val = random.random()
        cumulative = 0.0
        result = states_list[0]
        for s_idx, p in enumerate(probs):
            cumulative += p
            if r_val <= cumulative:
                result = states_list[s_idx]
                break
        print(f"  Сэмпл {i+1}: s'= {result} (вероятности: {dist})")

    # --- 1d. Ансамбль предиктивных моделей ---
    print("\n--- 1d. Ансамбль моделей (Ensemble Prediction) ---")
    # Каждая модель — линейная: y = w*x + b
    random.seed(42)
    X = [random.uniform(0, 10) for _ in range(8)]
    Y = [2.5 * x + 1.0 + random.gauss(0, 1.0) for x in X]

    def linear_fit(xs, ys):
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        w = (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-12)
        b = (sy - w * sx) / n
        return w, b

    # Три модели на разных подвыборках
    models = []
    for seed_val in [10, 20, 30]:
        random.seed(seed_val)
        indices = random.sample(range(len(X)), k=max(3, len(X) // 2))
        xs_sub = [X[i] for i in indices]
        ys_sub = [Y[i] for i in indices]
        w, b = linear_fit(xs_sub, ys_sub)
        models.append((w, b))
        print(f"  Модель seed={seed_val}: y = {w:.3f}*x + {b:.3f}")

    # Ансамбль: среднее предсказание
    x_test = 5.0
    preds = [w * x_test + b for w, b in models]
    ensemble_pred = sum(preds) / len(preds)
    print(f"\nТестовая точка x={x_test}")
    for i, p in enumerate(preds):
        print(f"  Модель {i+1}: предсказание = {p:.3f}")
    print(f"  Ансамбль (среднее): {ensemble_pred:.3f}")
    print(f"  Истинное значение:  {2.5 * x_test + 1.0:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. МЫСЛЕННАЯ СИМУЛЯЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def demo_mental_simulation():
    """Мысленная симуляция: look-ahead планирование и MCTS."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: MENTAL SIMULATION — мысленная симуляция")
    print("=" * 70)

    # --- 2a. Look-ahead планирование (глубина 1) ---
    print("\n--- 2a. Look-Ahead планирование (глубина 1) ---")
    # Пространство: grid 3x3, цель — попасть в (2,2)
    goal = (2, 2)
    moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    def heuristic(pos, g):
        """Манхэттенское расстояние до цели."""
        return abs(pos[0] - g[0]) + abs(pos[1] - g[1])

    state = (0, 0)
    print(f"Начальная позиция: {state}, цель: {goal}")
    print("Жадный look-ahead (глубина 1):")
    for step in range(5):
        best_move = None
        best_dist = float("inf")
        for name, (dr, dc) in moves.items():
            ns = (state[0] + dr, state[1] + dc)
            if 0 <= ns[0] <= 2 and 0 <= ns[1] <= 2:
                d = heuristic(ns, goal)
                if d < best_dist:
                    best_dist = d
                    best_move = name
        if best_move:
            dr, dc = moves[best_move]
            state = (state[0] + dr, state[1] + dc)
            print(f"  Шаг {step+1}: {best_move} -> {state} (dist={best_dist})")

    # --- 2b. Look-ahead глубина 2 ---
    print("\n--- 2b. Look-Ahead планирование (глубина 2) ---")
    state = (0, 0)
    print(f"Начальная позиция: {state}, цель: {goal}")
    print("Планирование с глубиной 2 (minimax минус расстояние):")
    for step in range(4):
        best_move = None
        best_score = -float("inf")
        for name, (dr, dc) in moves.items():
            s1 = (state[0] + dr, state[1] + dc)
            if not (0 <= s1[0] <= 2 and 0 <= s1[1] <= 2):
                continue
            # Смотрим на лучший следующий шаг
            min_next_dist = float("inf")
            for _, (dr2, dc2) in moves.items():
                s2 = (s1[0] + dr2, s1[1] + dc2)
                if 0 <= s2[0] <= 2 and 0 <= s2[1] <= 2:
                    min_next_dist = min(min_next_dist, heuristic(s2, goal))
            score = -min_next_dist  # максимизируем (чем меньше расстояние, тем лучше)
            if score > best_score:
                best_score = score
                best_move = name
        if best_move:
            dr, dc = moves[best_move]
            state = (state[0] + dr, state[1] + dc)
            print(f"  Шаг {step+1}: {best_move} -> {state}")

    # --- 2c. MCTS: UCB1 выбор ---
    print("\n--- 2c. MCTS: UCB1 выбор узлов ---")
    # Узлы дерева: имя -> (посещения, суммарная_награда)
    tree = {
        "root": [0, 0.0],
        "A": [10, 7.0],
        "B": [5, 2.0],
        "C": [2, 1.8],
        "D": [1, 0.9],
    }
    c_ucb = 1.41  # exploration constant
    N_total = sum(v[0] for v in tree.values())
    print(f"Дерево: { {k: (v[0], v[1]) for k, v in tree.items()} }")
    print(f"N_total = {N_total}, c = {c_ucb}")
    print("UCB1 = Q/N + c * sqrt(ln(N_parent) / N)")
    print()

    for name in ["A", "B", "C", "D"]:
        visits = tree[name][0]
        q_value = tree[name][1]
        avg_reward = q_value / visits if visits > 0 else 0
        exploration = c_ucb * math.sqrt(math.log(N_total + 1) / (visits + 1))
        ucb1 = avg_reward + exploration
        print(f"  {name}: Q/N={avg_reward:.3f}, exploration={exploration:.3f}, UCB1={ucb1:.3f}")

    # --- 2d. Rollout симуляция (случайная политика) ---
    print("\n--- 2d. Rollout симуляция (случайная политика) ---")
    random.seed(42)
    # Среда: цепь с 5 состояниями, награды в состояниях 4
    rewards_chain = [0, 0, 0, 0, 10]
    n_rollouts = 6
    gamma = 0.9
    print(f"Цепь состояний 0..4, награда=10 в состоянии 4, gamma={gamma}")
    print(f"Количество роллаутов: {n_rollouts}")
    returns_list = []
    for i in range(n_rollouts):
        s = random.randint(0, 2)
        trajectory = [s]
        for _ in range(20):
            # Действие: случайный шаг влево/вправо
            a = random.choice([-1, 1])
            s = max(0, min(4, s + a))
            trajectory.append(s)
            if s == 4:
                break
        # Считаем discounted return
        G = 0.0
        for t, st in enumerate(trajectory):
            G += (gamma ** t) * rewards_chain[st]
        returns_list.append(G)
        print(f"  Роллаут {i+1}: старт={trajectory[0]}, путь={trajectory[:8]}{'...' if len(trajectory)>8 else ''}, G={G:.2f}")
    avg_return = sum(returns_list) / len(returns_list)
    print(f"Средний return: {avg_return:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ОБУЧЕНИЕ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def demo_model_learning():
    """Обучение модели динамики из опыта, анализ ошибки предсказания."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: MODEL LEARNING — обучение модели из опыта")
    print("=" * 70)

    # --- 3a. Сбор опыта и обучение динамики ---
    print("\n--- 3a. Обучение динамики из опыта ---")
    # Истинная динамика: s' = 0.9 * s + 0.1 * a + noise
    random.seed(42)
    true_w = 0.9
    true_b = 0.1
    experiences = []
    for _ in range(30):
        s = random.uniform(-1, 1)
        a = random.choice([-1, 0, 1])
        noise = random.gauss(0, 0.05)
        s_next = true_w * s + true_b * a + noise
        experiences.append((s, a, s_next))

    # Обучаем линейную модель: s' = w * s + b * a
    # Аналитическое решение (метод наименьших квадратов)
    n = len(experiences)
    # Нормальные уравнения для [w, b]
    # Матрица: [[sum(s^2), sum(s*a)], [sum(s*a), sum(a^2)]]
    s_s = sum(s * s for s, a, _ in experiences)
    s_a = sum(s * a for s, a, _ in experiences)
    a_a = sum(a * a for s, a, _ in experiences)
    s_y = sum(s * sn for s, _, sn in experiences)
    a_y = sum(a * sn for _, a, sn in experiences)

    det = s_s * a_a - s_a * s_a
    w_hat = (a_a * s_y - s_a * a_y) / det
    b_hat = (s_s * a_y - s_a * s_y) / det
    print(f"Истинная модель: s' = {true_w}*s + {true_b}*a")
    print(f"Обученная модель: s' = {w_hat:.4f}*s + {b_hat:.4f}*a")
    print(f"Ошибки: dw={abs(w_hat - true_w):.4f}, db={abs(b_hat - true_b):.4f}")

    # --- 3b. Ошибка предсказания по эпохам ---
    print("\n--- 3b. Ошибка предсказания по эпохам (MSE) ---")
    random.seed(42)
    data_X = [random.uniform(0, 10) for _ in range(50)]
    data_Y = [3.0 * x + 2.0 + random.gauss(0, 2.0) for x in data_X]

    # Градиентный спуск
    w, b = 0.0, 0.0
    lr = 0.001
    epochs = 20
    mse_history = []
    for epoch in range(epochs):
        # Предсказание
        preds = [w * x + b for x in data_X]
        errors = [p - y for p, y in zip(preds, data_Y)]
        mse = sum(e * e for e in errors) / len(errors)
        mse_history.append(mse)
        # Градиенты
        dw = 2.0 / len(data_X) * sum(e * x for e, x in zip(errors, data_X))
        db = 2.0 / len(data_X) * sum(errors)
        w -= lr * dw
        b -= lr * db
        if epoch % 4 == 0 or epoch == epochs - 1:
            print(f"  Эпоха {epoch+1:2d}: MSE={mse:.4f}, w={w:.4f}, b={b:.4f}")

    print(f"\nСравнение MSE: эпоха 1 = {mse_history[0]:.4f}, эпоха {epochs} = {mse_history[-1]:.4f}")

    # --- 3c. Сравнение моделей по ошибке ---
    print("\n--- 3c. Сравнение моделей по ошибке предсказания ---")
    random.seed(42)
    test_X = [random.uniform(0, 10) for _ in range(20)]
    test_Y = [3.0 * x + 2.0 + random.gauss(0, 1.5) for x in test_X]

    models_pred = {
        "Линейная (w=3, b=2)": [3.0 * x + 2.0 for x in test_X],
        "Константа (mean)": [sum(test_Y) / len(test_Y)] * len(test_X),
        "Случайный лес (симуляция)": [3.0 * x + 2.0 + random.gauss(0, 0.5) for x in test_X],
    }
    for name, preds in models_pred.items():
        mse_val = sum((p - y) ** 2 for p, y in zip(preds, test_Y)) / len(test_Y)
        mae_val = sum(abs(p - y) for p, y in zip(preds, test_Y)) / len(test_Y)
        print(f"  {name}:")
        print(f"    MSE={mse_val:.4f}, MAE={mae_val:.4f}")

    # --- 3d. Анализ Residuals ---
    print("\n--- 3d. Анализ остатков (Residual Analysis) ---")
    random.seed(42)
    residuals = [random.gauss(0, 1.0) for _ in range(30)]
    mean_r = sum(residuals) / len(residuals)
    var_r = sum((r - mean_r) ** 2 for r in residuals) / len(residuals)
    print(f"Среднее остатков: {mean_r:.4f} (ожидается ~0)")
    print(f"Дисперсия остатков: {var_r:.4f}")

    # Автокорреляция (лаг 1)
    n_r = len(residuals)
    cov = sum(residuals[i] * residuals[i+1] for i in range(n_r - 1)) / (n_r - 1)
    autocorr = cov / (var_r + 1e-12)
    print(f"Автокорреляция (лаг 1): {autocorr:.4f} (ожидается ~0 для IID)")
    print("Нормальность: проверка через соотношение min/max к std:")
    std_r = math.sqrt(var_r)
    print(f"  min/std = {min(residuals)/std_r:.3f}, max/std = {max(residuals)/std_r:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ПЛАНИРОВАНИЕ ПО МОДЕЛИ МИРА
# ══════════════════════════════════════════════════════════════════════════════

def demo_world_model_planning():
    """Планирование в стиле Dreamer: оптимизация воображением."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: WORLD MODEL PLANNING — оптимизация воображением")
    print("=" * 70)

    # --- 4a. Dreamer-style: модель мира + оптимизация в латентном пространстве ---
    print("\n--- 4a. Dreamer-style: модель мира + оптимизация воображением ---")
    # Латентное состояние — 2D, модель предсказывает s' = A*s + B*a + c
    random.seed(42)
    A = [[0.9, 0.1], [0.1, 0.8]]
    B = [[0.5, 0.0], [0.0, 0.3]]
    c = [0.01, -0.01]

    def world_step(state, action):
        """Шаг модели мира."""
        s0 = A[0][0]*state[0] + A[0][1]*state[1] + B[0][0]*action[0] + B[0][1]*action[1] + c[0]
        s1 = A[1][0]*state[0] + A[1][1]*state[1] + B[1][0]*action[0] + B[1][1]*action[1] + c[1]
        return [s0, s1]

    def reward_fn(state):
        """Функция награды: близость к целевой точке (1, 1)."""
        return -((state[0] - 1.0) ** 2 + (state[1] - 1.0) ** 2)

    # Оптимизация: кандидаты действий, выбор лучшего
    s = [0.0, 0.0]
    gamma = 0.95
    H = 3  # горизонт планирования
    n_candidates = 20
    print(f"Начальное состояние: {s}, горизонт H={H}, кандидатов={n_candidates}")
    print("Оптимизация: случайные последовательности действий -> выбор лучшей")
    print()

    best_return = -float("inf")
    best_actions = None
    for _ in range(n_candidates):
        candidate_actions = [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)] for _ in range(H)]
        s_sim = list(s)
        G = 0.0
        for t, a in enumerate(candidate_actions):
            s_sim = world_step(s_sim, a)
            G += (gamma ** t) * reward_fn(s_sim)
        if G > best_return:
            best_return = G
            best_actions = candidate_actions

    print(f"Лучшая последовательность действий (G={best_return:.3f}):")
    for t, a in enumerate(best_actions):
        s = world_step(s, a)
        print(f"  t={t}: a=[{a[0]:.3f}, {a[1]:.3f}] -> s=[{s[0]:.3f}, {s[1]:.3f}]")

    # --- 4b. Многопробное планирование (Best-of-N) ---
    print("\n--- 4b. Многопробное планирование (Best-of-N) ---")
    random.seed(42)
    s_start = [0.0, 0.0]
    N = 50
    H_plan = 4
    returns = []
    print(f"Запуск {N} симуляций с горизонтом {H_plan}:")

    for i in range(N):
        s_sim = list(s_start)
        G = 0.0
        for t in range(H_plan):
            a = [random.gauss(0, 0.3), random.gauss(0, 0.3)]
            s_sim = world_step(s_sim, a)
            G += (gamma ** t) * reward_fn(s_sim)
        returns.append(G)

    returns_sorted = sorted(returns, reverse=True)
    print(f"  Min return:   {min(returns):.3f}")
    print(f"  Median:       {returns_sorted[N // 2]:.3f}")
    print(f"  Max return:   {max(returns):.3f}")
    print(f"  Top-5 среднее: {sum(returns_sorted[:5])/5:.3f}")
    print(f"  Top-10 среднее: {sum(returns_sorted[:10])/10:.3f}")

    # --- 4c. Iterative Refinement (CEM-like) ---
    print("\n--- 4c. Центрированные моменты (CEM-like) оптимизация ---")
    random.seed(42)
    # Параметризуем действия: a_t = mu_t + sigma_t * noise
    mu = [[0.0, 0.0]] * H_plan
    sigma = [[0.5, 0.5]] * H_plan
    n_pop = 30
    n_elite = 5

    for iteration in range(5):
        all_returns = []
        for _ in range(n_pop):
            s_sim = [0.0, 0.0]
            G = 0.0
            actions_taken = []
            for t in range(H_plan):
                a = [mu[t][0] + sigma[t][0] * random.gauss(0, 1),
                     mu[t][1] + sigma[t][1] * random.gauss(0, 1)]
                actions_taken.append(a)
                s_sim = world_step(s_sim, a)
                G += (gamma ** t) * reward_fn(s_sim)
            all_returns.append((G, actions_taken))

        all_returns.sort(key=lambda x: x[0], reverse=True)
        elite = all_returns[:n_elite]

        # Обновляем mu и sigma по элитным образцам
        for t in range(H_plan):
            elite_actions = [e[1][t] for e in elite]
            mu[t] = [sum(a[0] for a in elite_actions) / n_elite,
                      sum(a[1] for a in elite_actions) / n_elite]
            var_t0 = sum((a[0] - mu[t][0]) ** 2 for a in elite_actions) / n_elite + 0.01
            var_t1 = sum((a[1] - mu[t][1]) ** 2 for a in elite_actions) / n_elite + 0.01
            sigma[t] = [math.sqrt(var_t0), math.sqrt(var_t1)]

        best_G = elite[0][0]
        print(f"  Итерация {iteration+1}: лучший G={best_G:.3f}, sigma_mean={sum(s[0] for s in sigma)/H_plan:.4f}")

    # --- 4d. Сравнение: случайное vs оптимизированное ---
    print("\n--- 4d. Сравнение: случайное vs оптимизированное планирование ---")
    random.seed(42)
    s_random = [0.0, 0.0]
    s_optim = [0.0, 0.0]
    H_final = 5

    # Случайная политика
    random_returns = []
    for _ in range(100):
        s_sim = [0.0, 0.0]
        G = 0.0
        for t in range(H_final):
            a = [random.gauss(0, 0.5), random.gauss(0, 0.5)]
            s_sim = world_step(s_sim, a)
            G += (gamma ** t) * reward_fn(s_sim)
        random_returns.append(G)

    # Оптимизированная (CEM)
    mu_opt = [[0.3, 0.2], [0.3, 0.2], [0.2, 0.2], [0.15, 0.15], [0.1, 0.1]]
    s_sim_opt = [0.0, 0.0]
    G_opt = 0.0
    for t in range(H_final):
        s_sim_opt = world_step(s_sim_opt, mu_opt[t])
        G_opt += (gamma ** t) * reward_fn(s_sim_opt)

    avg_random = sum(random_returns) / len(random_returns)
    print(f"Случайная политика (100 запусков): средний G = {avg_random:.3f}")
    print(f"Оптимизированная политика: G = {G_opt:.3f}")
    print(f"Улучшение: {(G_opt - avg_random) / abs(avg_random) * 100:.1f}%")
    print(f"\nТраектория оптимизированной политики:")
    s_show = [0.0, 0.0]
    for t in range(H_final):
        s_show = world_step(s_show, mu_opt[t])
        print(f"  t={t+1}: s=[{s_show[0]:.3f}, {s_show[1]:.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_predictive_models()
    demo_mental_simulation()
    demo_model_learning()
    demo_world_model_planning()
    print("\n" + "=" * 70)
    print("Все демо завершены: World Models")
    print("=" * 70)
