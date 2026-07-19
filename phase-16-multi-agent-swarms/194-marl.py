"""194 — Multi-Agent Reinforcement Learning: MARL, кооперативное/конкурентное обучение

Темы:
  1. MARL Basics (децентрализованное vs централизованное, каналы связи)
  2. Cooperative MARL (QMIX, декомпозиция значений, централизованное обучение)
  3. Competitive MARL (самоигра, minimax Q, Nash Q-learning)
  4. Mixed Motivation (смешанная кооперация-конкуренция, α-расходимость)

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

# ============================================================
# ДЕМО 1: Основы MARL
# ============================================================

def demo_marl_basics():
    """Децентрализованное vs централизованное обучение, каналы связи."""
    print("=" * 70)
    print("ДЕМО 1: ОСНОВЫ MARL — децентрализация, централизация, коммуникация")
    print("=" * 70)

    # --- 1a. Среда с несколькими агентами ---
    print("\n--- 1a. Мультиагентная среда: мир 4×4 с 2 агентами ---")
    grid_size = 4
    n_agents = 2

    # Агенты: A1 (синий), A2 (красный)
    # Цель: встретиться в одной клетке
    agent_positions = [(0, 0), (3, 3)]
    goal = (2, 2)

    print(f"  Сетка: {grid_size}×{grid_size}")
    print(f"  Агент 1 (A1): {agent_positions[0]}")
    print(f"  Агент 2 (A2): {agent_positions[1]}")
    print(f"  Цель: встретиться в {goal}")

    # Действия: 0=вверх, 1=вниз, 2=влево, 3=вправо
    actions = ["↑", "↓", "←", "→"]
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Простая стратегия: двигаться к цели
    def move_toward(pos, target):
        """Выбрать действие для движения к цели."""
        dr = target[0] - pos[0]
        dc = target[1] - pos[1]
        if abs(dr) > abs(dc):
            return 0 if dr < 0 else 1  # Вверх или вниз
        else:
            return 2 if dc < 0 else 3  # Влево или вправо

    # Визуализация
    print(f"\n  Начальное состояние:")
    grid = [['·' for _ in range(grid_size)] for _ in range(grid_size)]
    grid[goal[0]][goal[1]] = '★'
    for idx, pos in enumerate(agent_positions):
        grid[pos[0]][pos[1]] = f'{idx + 1}'
    for r in range(grid_size):
        print(f"    {''.join(grid[r])}")

    # 5 шагов сближения
    print(f"\n  Динамика сближения:")
    for step in range(5):
        new_positions = []
        for i, pos in enumerate(agent_positions):
            action = move_toward(pos, goal)
            dr, dc = deltas[action]
            new_pos = (
                max(0, min(grid_size - 1, pos[0] + dr)),
                max(0, min(grid_size - 1, pos[1] + dc))
            )
            new_positions.append(new_pos)
        agent_positions = new_positions
        dist_a1 = abs(agent_positions[0][0] - agent_positions[1][0]) + abs(agent_positions[0][1] - agent_positions[1][1])
        print(f"    Шаг {step + 1}: A1={agent_positions[0]}, A2={agent_positions[1]}, расстояние={dist_a1}")

    # --- 1b. Децентрализованное обучение ---
    print("\n--- 1b. Децентрализованное обучение (DEC-MDP) ---")
    # Каждый агент обучается независимо
    n_states = 16  # 4×4 сетка
    n_actions_local = 4
    n_agents_dec = 3

    # Q-таблицы для каждого агента
    Q_tables = {}
    for a in range(n_agents_dec):
        Q_tables[a] = {}
        for s in range(n_states):
            Q_tables[a][s] = [0.0] * n_actions_local

    # Обучение (упрощённый Q-learning)
    print(f"  {n_agents_dec} агентов обучаются независимо")
    print(f"  Состояний: {n_states}, Действий: {n_actions_local}")

    alpha = 0.1   # Скорость обучения
    gamma = 0.95  # Коэффициент дисконтирования
    epsilon = 0.1  # ε-жадность

    # 100 эпизодов
    rewards_per_episode = []
    for episode in range(100):
        # Начальные позиции
        positions = [random.randint(0, n_states - 1) for _ in range(n_agents_dec)]
        total_reward = 0

        for t in range(20):
            for a in range(n_agents_dec):
                s = positions[a]
                # ε-жадный выбор действия
                if random.random() < epsilon:
                    action = random.randint(0, n_actions_local - 1)
                else:
                    action = Q_tables[a][s].index(max(Q_tables[a][s]))

                # Случайное перемещение
                new_s = (s + action + random.randint(-1, 1)) % n_states
                # Награда: расстояние до средней позиции
                mean_pos = sum(positions) / n_agents_dec
                reward = -abs(new_s - mean_pos) / n_states

                # Обновление Q
                old_q = Q_tables[a][s][action]
                next_max_q = max(Q_tables[a][new_s])
                Q_tables[a][s][action] = old_q + alpha * (reward + gamma * next_max_q - old_q)

                positions[a] = new_s
                total_reward += reward

        rewards_per_episode.append(total_reward)

    print(f"\n  Средняя награда (первые 10 эпизодов): {sum(rewards_per_episode[:10])/10:.3f}")
    print(f"  Средняя награда (последние 10 эпизодов): {sum(rewards_per_episode[-10:])/10:.3f}")
    print(f"  → Агенты улучшают свою стратегию, но не координируются!")

    # --- 1c. Централизованное обучение (CTDE) ---
    print("\n--- 1c. Централизованное обучение с децентрализованным исполнением (CTDE) ---")
    # В CTDE: тренер знает состояния всех агентов, но агенты действуют локально

    # Централизованная Q-таблица
    # Состояние: (s1, s2, s3) — комбинация
    n_agents_ctde = 2
    # Упрощённо: состояния 0-3 для каждого агента
    n_states_ctde = 4
    # Полное состояние: (s1, s2) = 4*4 = 16 комбинаций
    Q_centralized = {}
    for s1 in range(n_states_ctde):
        for s2 in range(n_states_ctde):
            # Действия пары: 4*4 = 16 комбинаций
            Q_centralized[(s1, s2)] = [0.0] * (n_actions_local * n_actions_local)

    print(f"  Централизованная Q-таблица:")
    print(f"    Размер: {n_states_ctde}^2 × {n_actions_local}^2 = "
          f"{n_states_ctde**2} × {n_actions_local**2} = {n_states_ctde**2 * n_actions_local**2}")

    # Фаза обучения: агенты делятся опытом
    print(f"\n  Фаза обучения: агенты делятся полным состоянием")
    print(f"  Фаза исполнения: каждый агент видит только своё состояние")

    # Пример: агент 1 видит s1=2, агент 2 видит s2=1
    s1_obs, s2_obs = 2, 1
    joint_state = (s1_obs, s2_obs)
    joint_q = Q_centralized[joint_state]

    # Лучшее совместное действие
    best_joint_action = joint_q.index(max(joint_q))
    a1_action = best_joint_action // n_actions_local
    a2_action = best_joint_action % n_actions_local

    print(f"\n  Агент 1 видит: s1={s1_obs}")
    print(f"  Агент 2 видит: s2={s2_obs}")
    print(f"  Совместное состояние: {joint_state}")
    print(f"  Лучшее действие: A1={actions[a1_action]}, A2={actions[a2_action]}")
    print(f"  Q(совместное): {max(joint_q):.3f}")

    # --- 1d. Каналы связи между агентами ---
    print("\n--- 1d. Каналы связи: протоколы коммуникации ---")
    # Три протокола
    protocols = {
        "Полная связность": {
            "description": "Каждый агент слышит всех",
            "messages_per_step": n_agents_dec * (n_agents_dec - 1),
            "bandwidth": "O(N²)"
        },
        "Кольцевая топология": {
            "description": "Агенты передают сообщения соседям",
            "messages_per_step": n_agents_dec * 2,
            "bandwidth": "O(N)"
        },
        "Старший агент": {
            "description": "Старший агрегирует и рассылает",
            "messages_per_step": n_agents_dec * 2,
            "bandwidth": "O(N)"
        },
    }

    print(f"  {n_agents_dec} агентов, сравнение протоколов:")
    print(f"  {'Протокол':<22} | {'Сообщений/шаг':<16} | {'Пропускная способность':<20} | Описание")
    print(f"  {'-'*22}-+-{'-'*16}-+-{'-'*20}-+-{'-'*30}")

    for name, info in protocols.items():
        print(f"  {name:<22} | {info['messages_per_step']:<16} | {info['bandwidth']:<20} | {info['description']}")

    # Моделирование обмена сообщениями
    print(f"\n  Моделирование (1 шаг, полная связность):")
    agent_states = [f"s{i}" for i in range(n_agents_dec)]
    agent_actions = [f"a{i}" for i in range(n_agents_dec)]
    messages = []
    for i in range(n_agents_dec):
        for j in range(n_agents_dec):
            if i != j:
                msg = f"Агент {i} → Агент {j}: «Моё состояние={agent_states[i]}, действие={agent_actions[i]}»"
                messages.append(msg)
    for msg in messages[:6]:
        print(f"    {msg}")
    print(f"    ... и ещё {len(messages) - 6} сообщений")
    print(f"  → Больше связей = лучше координация, но дороже по вычислениям")


# ============================================================
# ДЕМО 2: Кооперативный MARL
# ============================================================

def demo_cooperative_marl():
    """QMIX, декомпозиция значений, централизованное обучение."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: КООПЕРАТИВНЫЙ MARL — QMIX, декомпозиция, CTDE")
    print("=" * 70)

    # --- 2a. QMIX: интуиция ---
    print("\n--- 2a. QMIX: монотонная декомпозиция Q-значений ---")
    # QMIX: Q_tot = f(Q_1, Q_2, ...) с монотонным ограничением
    # ∂Q_tot/∂Q_i ≥ 0 для всех i

    # Два агента, 4 состояния каждый
    n_states = 4
    n_actions = 3

    # Локальные Q-таблицы агентов
    Q_1 = [
        [0.5, 0.3, 0.1],  # Состояние 0
        [0.2, 0.8, 0.4],  # Состояние 1
        [0.1, 0.2, 0.7],  # Состояние 2
        [0.6, 0.5, 0.3],  # Состояние 3
    ]
    Q_2 = [
        [0.3, 0.4, 0.2],
        [0.1, 0.6, 0.5],
        [0.7, 0.3, 0.8],
        [0.2, 0.9, 0.1],
    ]

    print(f"  Два агента, {n_states} состояний, {n_actions} действий")
    print(f"\n  Q-таблица агента 1:")
    print(f"  {'Состояние':<12} | {'Действие 0':<12} | {'Действие 1':<12} | {'Действие 2':<12}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for s in range(n_states):
        print(f"  {s:<12} | {Q_1[s][0]:<12.2f} | {Q_1[s][1]:<12.2f} | {Q_1[s][2]:<12.2f}")

    print(f"\n  Q-таблица агента 2:")
    print(f"  {'Состояние':<12} | {'Действие 0':<12} | {'Действие 1':<12} | {'Действие 2':<12}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for s in range(n_states):
        print(f"  {s:<12} | {Q_2[s][0]:<12.2f} | {Q_2[s][1]:<12.2f} | {Q_2[s][2]:<12.2f}")

    # QMIX агрегация: Q_tot = w1 * Q_1 + w2 * Q_2 + b
    # где w1, w2 ≥ 0 (монотонность)
    w1 = 0.6
    w2 = 0.4
    b = 0.1

    print(f"\n  QMIX агрегация: Q_tot = {w1}·Q_1 + {w2}·Q_2 + {b}")
    print(f"  Монотонность: w1={w1} ≥ 0, w2={w2} ≥ 0 ✓")

    # Совместное Q-значение
    print(f"\n  Совместные Q-значения (выбор: a1 для агента 1, a2 для агента 2):")
    print(f"  {'a1\\a2':<8} | {'Действ. 0':<12} | {'Действ. 1':<12} | {'Действ. 2':<12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    best_qtot = -float('inf')
    best_pair = (0, 0)
    for s1 in range(n_states):
        for s2 in range(n_states):
            print(f"  Состояния ({s1},{s2}):")
            for a1 in range(n_actions):
                row = f"  a1={a1:<4} |"
                for a2 in range(n_actions):
                    qtot = w1 * Q_1[s1][a1] + w2 * Q_2[s2][a2] + b
                    if s1 == 2 and s2 == 2 and qtot > best_qtot:
                        best_qtot = qtot
                        best_pair = (a1, a2)
                    row += f" {qtot:<11.3f} |"
                print(row)
            break  # Только для (0,0) чтобы не перегружать вывод
        break

    # Оптимальная стратегия для (s1=2, s2=2)
    print(f"\n  Оптимальное для (s1=2, s2=2): a1={best_pair[0]}, a2={best_pair[1]}, Q_tot={best_qtot:.3f}")

    # --- 2b. Декомпозиция значений (VDN) ---
    print("\n--- 2b. VDN: аддитивная декомпозиция ---")
    # VDN: Q_tot = Q_1 + Q_2 (без весов)

    print(f"  VDN: Q_tot = Q_1 + Q_2 (простое сложение)")
    print(f"  QMIX: Q_tot = w1·Q_1 + w2·Q_2 + b (гибче)")

    # Сравнение
    s1, s2 = 1, 2
    a1, a2 = 1, 2
    q1 = Q_1[s1][a1]
    q2 = Q_2[s2][a2]
    q_vdn = q1 + q2
    q_qmix = w1 * q1 + w2 * q2 + b

    print(f"\n  Пример: s1={s1}, s2={s2}, a1={a1}, a2={a2}")
    print(f"  Q_1 = {q1:.2f}, Q_2 = {q2:.2f}")
    print(f"  VDN:   Q_tot = {q1:.2f} + {q2:.2f} = {q_vdn:.3f}")
    print(f"  QMIX:  Q_tot = {w1}·{q1:.2f} + {w2}·{q2:.2f} + {b} = {q_qmix:.3f}")

    # --- 2c. Коммуникация в кооперативном MARL ---
    print("\n--- 2c. Выученная коммуникация ---")
    # Два агента: «отправитель» и «получатель»
    # Отправитель видит «цвет» (красный/синий/зелёный)
    # Получатель должен принять решение на основе сообщения

    n_channels = 3  # Длина сообщения
    n_messages = 2 ** n_channels  # 8 возможных сообщений
    n_colors = 3
    n_choices = 2

    # Таблица выученной коммуникации (отправитель)
    # color → message (3 бита)
    sender_policy = {
        0: [1, 0, 0],  # Красный → 100
        1: [0, 1, 0],  # Синий → 010
        2: [0, 0, 1],  # Зелёный → 001
    }

    # Таблица выученной коммуникации (получатель)
    # message → choice
    receiver_policy = {}
    for msg_int in range(n_messages):
        bits = [(msg_int >> (n_channels - 1 - i)) & 1 for i in range(n_channels)]
        if bits == [1, 0, 0]:
            receiver_policy[msg_int] = 0  # Красный → выбор 0
        elif bits == [0, 1, 0]:
            receiver_policy[msg_int] = 1  # Синий → выбор 1
        elif bits == [0, 0, 1]:
            receiver_policy[msg_int] = 0  # Зелёный → выбор 0
        else:
            receiver_policy[msg_int] = random.choice([0, 1])

    color_names = ["Красный", "Синий", "Зелёный"]
    print(f"  Каналы связи: {n_channels} бит")
    print(f"\n  Выученная коммуникация:")
    for color_idx in range(n_colors):
        msg = sender_policy[color_idx]
        msg_int = sum(b * (2 ** (n_channels - 1 - i)) for i, b in enumerate(msg))
        choice = receiver_policy.get(msg_int, 0)
        print(f"    {color_names[color_idx]} → сообщение {''.join(map(str, msg))} "
              f"(dec={msg_int}) → получатель выбирает {choice}")

    # Точность коммуникации
    correct = 0
    total = 100
    for _ in range(total):
        color = random.randint(0, n_colors - 1)
        msg = sender_policy[color]
        msg_int = sum(b * (2 ** (n_channels - 1 - i)) for i, b in enumerate(msg))
        choice = receiver_policy.get(msg_int, 0)
        # Правильный выбор: 0 для красного/зелёного, 1 для синего
        correct_choice = 1 if color == 1 else 0
        if choice == correct_choice:
            correct += 1

    print(f"\n  Точность координации: {correct}/{total} = {correct/total*100:.1f}%")

    # --- 2d. Матрица координации ---
    print("\n--- 2d. Задача координации: «Stag Hunt» ---")
    # Охота на оленя: кооперация даёт больше, но risky

    R_stag = 5  # Оба ловят оленя
    S_stag = 0  # Один ловит, другой — нет
    P_hare = 2  # Оба ловят зайца

    print(f"  Матрица выигрышей:")
    print(f"  {'':>15} | {'B: Олень':>12} | {'B: Заяц':>12}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'A: Олень':>15} | ({R_stag},{R_stag}){'':<6} | ({S_stag},{S_stag}){'':<6}")
    print(f"  {'A: Заяц':>15} | ({S_stag},{S_stag}){'':<6} | ({P_hare},{P_hare}){'':<6}")

    # Нэш-равновесия
    print(f"\n  Нэш-равновесия:")
    print(f"    (Олень, Олень): выигрыш = ({R_stag}, {R_stag}) —帕累托-оптимально")
    print(f"    (Заяц, Заяц): выигрыш = ({P_hare}, {P_hare}) —безопасно")

    # Моделирование: 100 раундов
    n_rounds = 100
    strategy = "охота на оленя"
    a_wins = 0
    b_wins = 0
    for r in range(n_rounds):
        a_choice = "олень" if random.random() < 0.7 else "заяц"
        b_choice = "олень" if random.random() < 0.7 else "заяц"
        if a_choice == "олень" and b_choice == "олень":
            a_wins += R_stag
            b_wins += R_stag
        elif a_choice == "олень" and b_choice == "заяц":
            a_wins += S_stag
            b_wins += S_stag
        elif a_choice == "заяц" and b_choice == "олень":
            a_wins += S_stag
            b_wins += S_stag
        else:
            a_wins += P_hare
            b_wins += P_hare

    print(f"\n  100 раундов (70% вероятность кооперации):")
    print(f"  Средний выигрыш A: {a_wins / n_rounds:.2f}")
    print(f"  Средний выигрыш B: {b_wins / n_rounds:.2f}")
    print(f"  → Кооперативный MARL должен выучить (Олень, Олень)")


# ============================================================
# ДЕМО 3: Конкурентный MARL
# ============================================================

def demo_competitive_marl():
    """Самоигра, minimax Q, Nash Q-learning."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: КОНКУРЕНТНЫЙ MARL — самоигра, minimax, Nash Q")
    print("=" * 70)

    # --- 3a. Самоигра (Self-Play) ---
    print("\n--- 3a. Самоигра: обучение через игру с собой ---")
    # «Камень-ножницы-бумага»
    rps_actions = ["Камень", "Ножницы", "Бумага"]
    # Матрица выигрышей: row vs column
    # 0=ничья, 1=победа, -1=поражение
    rps_matrix = [
        [0, 1, -1],   # Камень: vs КНБ
        [-1, 0, 1],   # Ножницы
        [1, -1, 0],   # Бумага
    ]

    # Q-таблица для одного агента (играет против себя)
    Q_self = [[0.0] * 3 for _ in range(3)]  # Q(s, a) где s — прошлое действие

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2

    # 500 игр
    win_rate_history = []
    for game in range(500):
        s = random.randint(0, 2)  # Начальное «состояние»

        # Выбор действия (ε-жадный)
        if random.random() < epsilon:
            a = random.randint(0, 2)
        else:
            a = Q_self[s].index(max(Q_self[s]))

        # Противник (как «другой агент») — использует ε-жадную стратегию
        if random.random() < epsilon:
            opponent_a = random.randint(0, 2)
        else:
            opponent_a = Q_self[s].index(max(Q_self[s]))

        # Награда
        reward = rps_matrix[a][opponent_a]

        # Новое состояние
        s_next = a

        # Обновление Q
        old_q = Q_self[s][a]
        next_max_q = max(Q_self[s_next])
        Q_self[s][a] = old_q + alpha * (reward + gamma * next_max_q - old_q)

    print(f"  Игра: Камень-ножницы-бумага (самоигра)")
    print(f"  Эпизодов: 500, α={alpha}, γ={gamma}, ε={epsilon}")
    print(f"\n  Выученная Q-таблица (после 500 игр):")
    print(f"  {'Состояние':<12} | {'Камень':<10} | {'Ножницы':<10} | {'Бумага':<10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for s in range(3):
        print(f"  {rps_actions[s]:<12} | {Q_self[s][0]:<10.3f} | {Q_self[s][1]:<10.3f} | {Q_self[s][2]:<10.3f}")

    # Стратегия: равномерная (оптимально в РКБ)
    print(f"\n  Оптимальная стратегия РКБ: равномерная (1/3, 1/3, 1/3)")
    print(f"  → Самоигра приближает к смешанной стратегии Нэша")

    # --- 3b. Minimax Q-learning ---
    print("\n--- 3b. Minimax Q-learning: двухагентная игра ---")
    # Простая网格 игра: 3×3, два агента
    # A: пытается максимизировать, B: минимизировать

    n_rows = 3
    n_cols = 3
    # Награды: A получает +1 за центр, -1 за углы, 0 за стороны
    rewards = [[0] * n_cols for _ in range(n_rows)]
    rewards[1][1] = 5   # Центр: +5
    rewards[0][0] = -3  # Углы: -3
    rewards[0][2] = -3
    rewards[2][0] = -3
    rewards[2][2] = -3
    rewards[0][1] = 1   # Стороны: +1
    rewards[1][0] = 1
    rewards[1][2] = 1
    rewards[2][1] = 1

    print(f"  Игровое поле 3×3:")
    for r in range(n_rows):
        row_str = "    "
        for c in range(n_cols):
            row_str += f"{rewards[r][c]:+3d} "
        print(row_str)

    # Minimax: B会选择对A最差的位置
    print(f"\n  Minimax анализ:")
    print(f"  A пытается максимизировать, B — минимизировать")

    # Стратегия A: выбрать строку
    # Стратегия B: выбрать столбец (минимизируя награду A)
    best_minimax = -float('inf')
    best_row = 0
    for row in range(n_rows):
        # B выберет столбец, минимизирующий награду
        min_val = min(rewards[row][col] for col in range(n_cols))
        print(f"  A выбирает строку {row}: B минимизирует → награда = {min_val}")
        if min_val > best_minimax:
            best_minimax = min_val
            best_row = row

    print(f"\n  Minimax решение: A выбирает строку {best_row}, награда = {best_minimax}")

    # --- 3c. Nash Q-learning ---
    print("\n--- 3c. Nash Q-learning: multiple равновесия ---")
    # Игра «_matching pennies» с расширенным пространством

    # 2×2 игра с тремя равновесиями
    # 2 чистых + 1 смешанное
    payoff_a = [
        [3, 0],
        [0, 2]
    ]
    payoff_b = [
        [2, 0],
        [0, 3]
    ]

    print(f"  Игра с 2 чистыми и 1 смешанным равновесием:")
    print(f"  {'':>15} | {'B: L':>10} | {'B: R':>10}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'A: T':>15} | ({payoff_a[0][0]},{payoff_b[0][0]}) | ({payoff_a[0][1]},{payoff_b[0][1]})")
    print(f"  {'A: B':>15} | ({payoff_a[1][0]},{payoff_b[1][0]}) | ({payoff_a[1][1]},{payoff_b[1][1]})")

    # Нэш-равновесия
    print(f"\n  Нэш-равновесия:")
    print(f"    1. (T, L): выигрыш = ({payoff_a[0][0]}, {payoff_b[0][0]})")
    print(f"    2. (B, R): выигрыш = ({payoff_a[1][1]}, {payoff_b[1][1]})")

    # Смешанное равновесие: p*(T) = ? , q*(L) = ?
    # Для A: 3q = 0q + 0(1-q) + 2(1-q) → 3q = 2 - 2q → q = 2/5
    # Для B: 2p = 0p + 0(1-p) + 3(1-p) → 2p = 3 - 3p → p = 3/5
    p_star = 3 / 5  # Вероятность A выбрать T
    q_star = 2 / 5  # Вероятность B выбрать L

    expected_a = p_star * (payoff_a[0][0] * q_star + payoff_a[0][1] * (1 - q_star)) + \
                 (1 - p_star) * (payoff_a[1][0] * q_star + payoff_a[1][1] * (1 - q_star))
    expected_b = p_star * (payoff_b[0][0] * q_star + payoff_b[0][1] * (1 - q_star)) + \
                 (1 - p_star) * (payoff_b[1][0] * q_star + payoff_b[1][1] * (1 - q_star))

    print(f"\n  Смешанное равновесие:")
    print(f"    p*(T) = {p_star:.3f}, p*(B) = {1-p_star:.3f}")
    print(f"    q*(L) = {q_star:.3f}, q*(R) = {1-q_star:.3f}")
    print(f"    Ожидаемый выигрыш A: {expected_a:.3f}")
    print(f"    Ожидаемый выигрыш B: {expected_b:.3f}")

    # --- 3d. Конкурентное обучение: эволюция стратегий ---
    print("\n--- 3d. Эволюция стратегий: турнир ---")
    # 5 стратегий, каждая играет с каждой
    strategies = [
        ("Жадный", lambda: 1),
        ("Случайный", lambda: random.randint(0, 2)),
        ("Тит-форт-тат", None),  # Нужна история
        ("Мстительный", None),
        ("Зеркальный", None),
    ]

    # Упрощённый турнир: 200 игр
    n_games = 200
    scores = [0] * 5

    # Все пары стратегий
    def play_rps(s1_func, s2_func, history_s1=None, history_s2=None):
        """Игра КНБ между двумя стратегиями."""
        if s1_func is None:
            a1 = history_s1[-1] if history_s1 else 0
        else:
            a1 = s1_func()

        if s2_func is None:
            a2 = history_s2[-1] if history_s2 else 0
        else:
            a2 = s2_func()

        return rps_matrix[a1][a2], a1, a2

    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            h1 = []
            h2 = []
            s1 = 0
            s2 = 0
            for _ in range(n_games // 10):
                if i == 2:  # TFT
                    a1 = h2[-1] if h2 else 0
                elif i == 3:  # Мстительный
                    a1 = h2[-1] if (h2 and h2[-1] != h1[-1]) else 0
                elif i == 4:  # Зеркальный
                    a1 = h2[-1] if h2 else 0
                else:
                    a1 = strategies[i][1]()

                if j == 2:
                    a2 = h1[-1] if h1 else 0
                elif j == 3:
                    a2 = h1[-1] if (h1 and h1[-1] != h2[-1]) else 0
                elif j == 4:
                    a2 = h1[-1] if h1 else 0
                else:
                    a2 = strategies[j][1]()

                reward = rps_matrix[a1][a2]
                scores[i] += reward
                h1.append(a1)
                h2.append(a2)

    print(f"  5 стратегий в турнире ({n_games} игр каждая пара):")
    for i in range(5):
        bar = "█" * max(0, (scores[i] + 50) // 5)
        print(f"    {strategies[i][0]:<12}: {scores[i]:+4d} {bar}")

    winner = max(range(5), key=lambda x: scores[x])
    print(f"\n  Победитель турнира: {strategies[winner][0]} ({scores[winner]:+d})")


# ============================================================
# ДЕМО 4: Смешанная мотивация
# ============================================================

def demo_mixed_motivation():
    """Смешанная кооперация-конкуренция, α-расходимость."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: СМЕШАННАЯ МОТИВАЦИЯ — кооперация+конкуренция, α-расходимость")
    print("=" * 70)

    # --- 4a. Смешанная среда ---
    print("\n--- 4a. Смешанная среда: «Chicken Game» ---")
    # Два автомобиля едут навстречу друг другу
    # Кто свернёт — «трус», кто не свернёт — «храбрец»
    # Оба не свернули — катастрофа

    payoff_matrix = [
        [0, -1, 3],    # Свернуть: vs Свернуть=0, vs Нет=-1, vs Другое=3
        [3, 0, -1],    # Прямо: vs Свернуть=3, vs Прямо=0, vs Другое=-1
        [-1, 3, 0],    # Другое
    ]

    print(f"  «Chicken Game»: два агента на столкновении")
    print(f"  Матрица выигрышей (для агента A):")
    actions = ["Свернуть", "Прямо", "Другое"]
    print(f"  {'':>12} | {'B: Свернуть':>12} | {'B: Прямо':>12} | {'B: Другое':>12}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for i, a in enumerate(actions):
        row = f"  {a:>12} |"
        for j in range(3):
            row += f" {payoff_matrix[i][j]:>+11d} |"
        print(row)

    # Нэш-равновесия
    print(f"\n  Нэш-равновесия:")
    print(f"    1. (Свернуть, Прямо): A=-1, B=+3")
    print(f"    2. (Прямо, Свернуть): A=+3, B=-1")
    print(f"    Смешанное: p(Прямо) = ?")

    # Смешанное: E[Свернуть] = E[Прямо]
    # 0·q + (-1)·(1-q) + 3·r = 3·q + 0·(1-q) + (-1)·r
    # Где q = P(B свернёт), r = P(B другое)
    # Упрощённо: q = 2/3, p = 2/3 (симметрично)
    p_mixed = 2 / 3
    print(f"    Смешанное: p(Прямо) = {p_mixed:.3f} для обоих")

    # Моделирование
    n_rounds = 100
    a_wins = 0
    b_wins = 0
    crashes = 0
    for _ in range(n_rounds):
        a_choice = 1 if random.random() < p_mixed else 0
        b_choice = 1 if random.random() < p_mixed else 0
        if a_choice == 1 and b_choice == 1:
            crashes += 1
        a_wins += payoff_matrix[a_choice][b_choice]
        b_wins += payoff_matrix[b_choice][a_choice]

    print(f"\n  100 раундов (p={p_mixed:.2f}):")
    print(f"  Средний выигрыш A: {a_wins / n_rounds:.2f}")
    print(f"  Средний выигрыш B: {b_wins / n_rounds:.2f}")
    print(f"  Столкновений: {crashes} ({crashes}%)")

    # --- 4b. α-расходимость (дивергенция Цsisзар) ---
    print("\n--- 4b. α-расходимость: мера «расстояния» между стратегиями ---")
    # D_α(P || Q) = (1/(α(α-1))) * (Σ p_i^α * q_i^(1-α) - 1)

    def alpha_divergence(p, q, alpha):
        """Расчёт α-расходимости."""
        if alpha == 0:
            # KL-расходимость (правая)
            return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)
        elif alpha == 1:
            # KL-расходимость (левая)
            return sum(qi * math.log(qi / pi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)
        elif alpha == -1:
            # Обратная расходимость
            return -2 * sum(math.sqrt(pi * qi) for pi, qi in zip(p, q)) + 2
        else:
            # Общая формула
            total = 0
            for pi, qi in zip(p, q):
                if pi > 0 and qi > 0:
                    total += (pi ** alpha) * (qi ** (1 - alpha))
            return (total - 1) / (alpha * (alpha - 1))

    # Стратегии агентов
    p_agent = [0.5, 0.3, 0.2]   # Стратегия агента 1
    q_agent = [0.2, 0.5, 0.3]   # Стратегия агента 2

    print(f"  Стратегия P (агент 1): {p_agent}")
    print(f"  Стратегия Q (агент 2): {q_agent}")
    print(f"\n  α-расходимость D_α(P || Q):")
    print(f"  {'α':>6} | {'D_α(P||Q)':>12} | {'Интерпретация'}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*40}")

    alphas = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
    interpretations = {
        -1: "Обратная ( Hellinger)",
        -0.5: "Полуобратная",
        0: "KL (правая)",
        0.5: "Хелингера (缩水)",
        1: "KL (левая)",
        1.5: "Полупрямая",
        2: "χ²-расходимость",
    }

    for alpha in alphas:
        d = alpha_divergence(p_agent, q_agent, alpha)
        interp = interpretations.get(alpha, "")
        print(f"  {alpha:>6.1f} | {d:>12.4f} | {interp}")

    print(f"\n  α=0 → KL(P||Q): мера «сколько Q отличается от P»")
    print(f"  α=1 → KL(Q||P): обратная мера")
    print(f"  α=2 → χ²-тест: чувствителен к выбросам")

    # --- 4c. Смешанные стратегии в кооперативно-конкурентных играх ---
    print("\n--- 4c. Смешанная игра: «Battle of the Sexes» с внешними ---")
    # Два агента: хотят быть вместе, но в разных местах
    # Третий «наблюдатель» влияет на исход

    bos_a = [[3, 0], [0, 2]]
    bos_b = [[2, 0], [0, 3]]

    print(f"  Battle of the Sexes:")
    print(f"  {'':>15} | {'B: Опера':>12} | {'B: Футбол':>12}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'A: Опера':>15} | ({bos_a[0][0]},{bos_b[0][0]}) | ({bos_a[0][1]},{bos_b[0][1]})")
    print(f"  {'A: Футбол':>15} | ({bos_a[1][0]},{bos_b[1][0]}) | ({bos_a[1][1]},{bos_b[1][1]})")

    # Смешанное равновесие: p*(Опера) = 3/5, q*(Опера) = 2/5
    p_opera = 3 / 5
    q_opera = 2 / 5

    expected_a = p_opera * (bos_a[0][0] * q_opera + bos_a[0][1] * (1 - q_opera)) + \
                 (1 - p_opera) * (bos_a[1][0] * q_opera + bos_a[1][1] * (1 - q_opera))
    expected_b = p_opera * (bos_b[0][0] * q_opera + bos_b[0][1] * (1 - q_opera)) + \
                 (1 - p_opera) * (bos_b[1][0] * q_opera + bos_b[1][1] * (1 - q_opera))

    print(f"\n  Смешанное равновесие:")
    print(f"    A: P(Опера) = {p_opera:.3f}, P(Футбол) = {1-p_opera:.3f}")
    print(f"    B: P(Опера) = {q_opera:.3f}, P(Футбол) = {1-q_opera:.3f}")
    print(f"    Ожидаемый выигрыш A: {expected_a:.3f}")
    print(f"    Ожидаемый выигрыш B: {expected_b:.3f}")

    # 100 игр
    total_a = 0
    total_b = 0
    same_choice = 0
    for _ in range(100):
        a_choice = 0 if random.random() < p_opera else 1
        b_choice = 0 if random.random() < q_opera else 1
        total_a += bos_a[a_choice][b_choice]
        total_b += bos_b[a_choice][b_choice]
        if a_choice == b_choice:
            same_choice += 1

    print(f"\n  100 игр:")
    print(f"  Средний выигрыш A: {total_a / 100:.2f}")
    print(f"  Средний выигрыш B: {total_b / 100:.2f}")
    print(f"  Агенты вместе: {same_choice}%")
    print(f"  → Смешанная стратегия снижает конфликт, но увеличивает разброс")

    # --- 4d. Обобщение: типы мотивации ---
    print("\n--- 4d. Типы мотивации агентов ---")
    motivation_types = [
        ("Чистая кооперация", "Обе стороны выигрывают от сотрудничества", "Коорпоративный MARL"),
        ("Чистая конкуренция", "Выигрыш одного = проигрыш другого", "Нулевая сумма"),
        ("Смешанная (кооп+конкуренция)", "Есть área сотрудничества и конфликта", "Игры с положительной суммой"),
        ("Параллельная", "Агенты независимы", "Независимые MDP"),
        ("Общие интересы", "Агенты хотят одного и того же", "Мультицелевая оптимизация"),
    ]

    print(f"  {'Тип':<35} | {'Описание':<45} | {'Применение в MARL'}")
    print(f"  {'-'*35}-+-{'-'*45}-+-{'-'*30}")
    for name, desc, marl_type in motivation_types:
        print(f"  {name:<35} | {desc:<45} | {marl_type}")

    # Метрика «степень конфликта»
    print(f"\n  Метрика «степень конфликта» (из матрицы выигрышей):")
    conflicts = [
        ("Battle of Sexes", 0.3, "Низкий конфликт"),
        ("Chicken", 0.7, "Высокий конфликт"),
        ("Prisoner's Dilemma", 0.9, "Критический конфликт"),
        ("Harmony Game", 0.0, "Нет конфликта"),
    ]
    for game, conflict, level in conflicts:
        bar = "█" * int(conflict * 20)
        print(f"    {game:<20}: {conflict:.1f} {bar} ({level})")


# ============================================================
# ЗАПУСК ВСЕХ ДЕМО
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  194 — Multi-Agent Reinforcement Learning: MARL                        ║")
    print("╚" + "═" * 68 + "╝")

    demo_marl_basics()
    demo_cooperative_marl()
    demo_competitive_marl()
    demo_mixed_motivation()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМО ЗАВЕРШЕНЫ: Multi-Agent Reinforcement Learning")
    print("=" * 70)
