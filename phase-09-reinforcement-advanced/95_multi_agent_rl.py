"""
95 — Multi-Agent Reinforcement Learning (Multi-Agent RL)

Основы обучения агентов в среде с несколькими агентами.

Компоненты:
  1. Cooperating agents — агенты сотрудничают для достижения общей цели
  2. Competitive agents — агенты соревнуются за ресурс
  3. Communication — агенты обмениваются сообщениями
  4. Emergent behavior — возникновение стратегий без явных правил

Каждый агент: Q-learning с табличным представлением.
Среда: простые сетки / сценарии на чистом Python.
"""

import random
import math
from typing import Dict, List, Tuple, Optional

random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# Базовый агент
# ═══════════════════════════════════════════════════════════════════════════════

class QAgent:
    """Q-learning агент с табличным Q-значениями."""

    def __init__(
        self,
        agent_id: str,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[tuple, List[float]] = {}

    def get_q(self, state) -> List[float]:
        key = state if isinstance(state, tuple) else (state,)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.n_actions
        return self.q_table[key]

    def choose_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_vals = self.get_q(state)
        max_q = max(q_vals)
        best = [i for i, v in enumerate(q_vals) if v == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, done=False):
        key_s = state if isinstance(state, tuple) else (state,)
        key_ns = next_state if isinstance(next_state, tuple) else (next_state,)
        q = self.get_q(key_s)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.get_q(key_ns))
        q[action] += self.lr * (target - q[action])

    def policy_summary(self, state) -> List[float]:
        return self.get_q(state)[:]


# ═══════════════════════════════════════════════════════════════════════════════
# Демо 1: Cooperating Agents — совместное решение задачи
# ═══════════════════════════════════════════════════════════════════════════════

def demo_cooperating_agents():
    """
    Два агента на 4×4 сетке должны вместе довести «груз» от (0,0) до (3,3).
    Агент 1 двигает груз по строкам (вправо/вниз).
    Агент 2 двигает груз по столбцам (вниз/вправо).
    Оба получают +10, только когда груз достиг цели; -1 за каждый шаг.
    Общий reward делится поровну.
    """
    print("=" * 70)
    print("ДЕМО 1: Cooperating Agents — совместное решение задачи")
    print("=" * 70)
    print("Два агента совместно тащат груз из (0,0) в (3,3) на сетке 4×4.\n")

    GRID = 4
    GOAL = (3, 3)
    MAX_STEPS = 20
    ACTIONS_1 = [0, 1]  # 0=right, 1=down  (агент 1)
    ACTIONS_2 = [0, 1]  # 0=down, 1=right  (агент 2)
    N_EPISODES = 300

    agent1 = QAgent("A1", len(ACTIONS_1), lr=0.15, gamma=0.9, epsilon=0.25)
    agent2 = QAgent("A2", len(ACTIONS_2), lr=0.15, gamma=0.9, epsilon=0.25)

    def step(state: Tuple[int, int], a1: int, a2: int):
        r, c = state
        if a1 == 1:
            r = min(r + 1, GRID - 1)
        if a2 == 0:
            r = min(r + 1, GRID - 1)
        else:
            c = min(c + 1, GRID - 1)
        next_state = (r, c)
        if next_state == GOAL:
            reward = 10.0
            done = True
        else:
            reward = -1.0
            done = False
        return next_state, reward, done

    for ep in range(N_EPISODES):
        state = (0, 0)
        for _ in range(MAX_STEPS):
            a1 = agent1.choose_action(state)
            a2 = agent2.choose_action(state)
            next_state, reward, done = step(state, a1, a2)
            half = reward / 2.0
            agent1.update(state, a1, half, next_state, done)
            agent2.update(state, a2, half, next_state, done)
            state = next_state
            if done:
                break

    # Жадная политика
    state = (0, 0)
    path = [state]
    for _ in range(MAX_STEPS):
        a1 = agent1.choose_action(state)
        a2 = agent2.choose_action(state)
        state, _, done = step(state, a1, a2)
        path.append(state)
        if done:
            break

    print(f"Обучение: {N_EPISODES} эпизодов.")
    print(f"Путь груза (жадная политика): {path}")
    print(f"Достигнута цель: {path[-1] == GOAL}")

    for s in [(0, 0), (1, 1), (2, 2)]:
        q1 = agent1.policy_summary(s)
        q2 = agent2.policy_summary(s)
        a1_best = "right" if q1[0] >= q1[1] else "down"
        a2_best = "down" if q2[0] >= q2[1] else "right"
        print(f"  Позиция {s}: A1 → {a1_best}, A2 → {a2_best}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Демо 2: Competitive Agents — соревнование
# ═══════════════════════════════════════════════════════════════════════════════

def demo_competitive_agents():
    """
    Два агента бегут по дорожке длины 10 к финишу (ячейка 9).
    Кто первый доберётся — получает +10, проигравший 0.
    Каждый шаг — -0.1 (штраф за время).
    """
    print("=" * 70)
    print("ДЕМО 2: Competitive Agents — соревнование")
    print("=" * 70)
    print("Два агента бегут по дорожке длины 10 к финишу.\n")

    TRACK_LEN = 10
    GOAL_POS = 9
    ACTIONS = [0, 1]  # 0=остановиться, 1=идти вперёд
    N_EPISODES = 500

    agent1 = QAgent("Racer1", 2, lr=0.15, gamma=0.9, epsilon=0.3)
    agent2 = QAgent("Racer2", 2, lr=0.15, gamma=0.9, epsilon=0.3)

    def race_step(pos1, pos2, a1, a2):
        """Возвращает (new_pos1, new_pos2, r1, r2, done)."""
        new_p1 = min(pos1 + a1, GOAL_POS)
        new_p2 = min(pos2 + a2, GOAL_POS)
        done = (new_p1 == GOAL_POS) or (new_p2 == GOAL_POS)
        if done:
            if new_p1 == GOAL_POS and new_p2 == GOAL_POS:
                r1, r2 = 5.0, 5.0
            elif new_p1 == GOAL_POS:
                r1, r2 = 10.0, 0.0
            else:
                r1, r2 = 0.0, 10.0
        else:
            r1, r2 = -0.1, -0.1
        return new_p1, new_p2, r1, r2, done

    wins = {"Racer1": 0, "Racer2": 0, "draw": 0}

    for ep in range(N_EPISODES):
        p1, p2 = 0, 0
        state1 = (p1,)
        state2 = (p2,)
        for _ in range(30):
            a1 = agent1.choose_action(state1)
            a2 = agent2.choose_action(state2)
            np1, np2, r1, r2, done = race_step(p1, p2, a1, a2)
            ns1, ns2 = (np1,), (np2,)
            agent1.update(state1, a1, r1, ns1, done)
            agent2.update(state2, a2, r2, ns2, done)
            state1, state2 = ns1, ns2
            p1, p2 = np1, np2
            if done:
                if p1 == GOAL_POS and p2 == GOAL_POS:
                    wins["draw"] += 1
                elif p1 == GOAL_POS:
                    wins["Racer1"] += 1
                else:
                    wins["Racer2"] += 1
                break

    print(f"Обучение: {N_EPISODES} эпизодов.")
    print(f"Статистика побед: Racer1={wins['Racer1']}, Racer2={wins['Racer2']}, "
          f"ничьи={wins['draw']}")

    # Симуляция одной гонки
    p1, p2 = 0, 0
    trace1, trace2 = [p1], [p2]
    for _ in range(30):
        a1 = agent1.choose_action((p1,))
        a2 = agent2.choose_action((p2,))
        p1, p2 = min(p1 + a1, GOAL_POS), min(p2 + a2, GOAL_POS)
        trace1.append(p1)
        trace2.append(p2)
        if p1 == GOAL_POS or p2 == GOAL_POS:
            break

    print(f"Пример гонки: Racer1={trace1}, Racer2={trace2}")

    # Q-значения для позиции 5
    q1_at5 = agent1.policy_summary((5,))
    q2_at5 = agent2.policy_summary((5,))
    print(f"Racer1 в позиции 5: Q=[stay={q1_at5[0]:.2f}, move={q1_at5[1]:.2f}]")
    print(f"Racer2 в позиции 5: Q=[stay={q2_at5[0]:.2f}, move={q2_at5[1]:.2f}]")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Демо 3: Communication — обмен информацией
# ═══════════════════════════════════════════════════════════════════════════════

def demo_communication():
    """
    Три агента-исследователя на сетке 6×6. Каждый видит только 3×3 окрестность.
    Агенты могут передавать «метку» — сообщение о типе объекта на своей клетке.
    Сообщения записываются в общее хранилище. Q-таблица учитывает историю сообщений.
    """
    print("=" * 70)
    print("ДЕМО 3: Communication — обмен информацией")
    print("=" * 70)
    print("Три агента探索 6×6 сетку, обмениваясь метками о найденных объектах.\n")

    GRID = 6
    N_AGENTS = 3
    # Объекты на сетке: {позиция: тип}
    objects = {
        (1, 1): "food",
        (2, 4): "gem",
        (4, 0): "food",
        (5, 3): "gem",
        (0, 5): "food",
    }
    REWARD_MAP = {"food": 3.0, "gem": 5.0}
    MAX_STEPS = 15
    ACTIONS = [0, 1, 2, 3]  # up, down, left, right
    ACTION_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MSG_TYPES = ["none", "food", "gem"]
    N_EPISODES = 400

    # Общее хранилище сообщений: список сообщений за эпизод
    message_board: List[Tuple[int, str]] = []

    class CommAgent(QAgent):
        """Агент с учётом сообщений в состоянии."""

        def __init__(self, agent_id: int):
            super().__init__(agent_id, len(ACTIONS), lr=0.12, gamma=0.88, epsilon=0.3)
            self.discovered = set()

        def make_state(self, pos, board_summary):
            return (pos, board_summary)

    agents = [CommAgent(i) for i in range(N_AGENTS)]
    total_rewards_by_agent = [0.0] * N_AGENTS
    total_communications = 0

    for ep in range(N_EPISODES):
        # Случайные стартовые позиции
        starts = [(0, 0), (0, GRID - 1), (GRID - 1, 0)]
        positions = list(starts)
        message_board.clear()
        discovered_per_agent = [set() for _ in range(N_AGENTS)]

        for _ in range(MAX_STEPS):
            for i, agent in enumerate(agents):
                r, c = positions[i]
                # Проверяем объект
                if (r, c) in objects and (r, c) not in discovered_per_agent[i]:
                    obj_type = objects[(r, c)]
                    discovered_per_agent[i].add((r, c))
                    message_board.append((i, obj_type))
                    total_communications += 1

                # Сводка сообщений: сколько каких объектов
                food_msgs = sum(1 for _, t in message_board if t == "food")
                gem_msgs = sum(1 for _, t in message_board if t == "gem")
                board_summary = (min(food_msgs, 5), min(gem_msgs, 5))
                state = agent.make_state(positions[i], board_summary)

                a = agent.choose_action(state)
                dr, dc = ACTION_DELTA[a]
                nr = max(0, min(GRID - 1, r + dr))
                nc = max(0, min(GRID - 1, c + dc))
                next_pos = (nr, nc)
                reward = -0.1

                if next_pos in objects:
                    reward += REWARD_MAP.get(objects[next_pos], 0)

                next_board = (min(food_msgs, 5), min(gem_msgs, 5))
                next_state = agent.make_state(next_pos, next_board)
                done = False
                agent.update(state, a, reward, next_state, done)
                positions[i] = next_pos
                total_rewards_by_agent[i] += reward

    print(f"Обучение: {N_EPISODES} эпизодов.")
    print(f"Всего сообщений за обучение: {total_communications}")
    for i in range(N_AGENTS):
        print(f"  Агент {i}: общий reward = {total_rewards_by_agent[i]:.1f}, "
              f"уникальных состояний в Q = {len(agents[i].q_table)}")

    # Финальный эпизод
    positions = list(starts)
    message_board.clear()
    discovered_per_agent = [set() for _ in range(N_AGENTS)]
    print("\nФинальный эпизод (жадная политика):")
    for step_i in range(MAX_STEPS):
        step_msgs = []
        for i, agent in enumerate(agents):
            r, c = positions[i]
            if (r, c) in objects and (r, c) not in discovered_per_agent[i]:
                obj_type = objects[(r, c)]
                discovered_per_agent[i].add((r, c))
                message_board.append((i, obj_type))
                step_msgs.append(f"  A{i} на {positions[i]} нашёл {obj_type}")

        for msg in step_msgs:
            print(msg)

        for i, agent in enumerate(agents):
            food_msgs = sum(1 for _, t in message_board if t == "food")
            gem_msgs = sum(1 for _, t in message_board if t == "gem")
            board_summary = (min(food_msgs, 5), min(gem_msgs, 5))
            state = agent.make_state(positions[i], board_summary)
            a = agent.choose_action(state)
            dr, dc = ACTION_DELTA[a]
            nr = max(0, min(GRID - 1, positions[i][0] + dr))
            nc = max(0, min(GRID - 1, positions[i][1] + dc))
            positions[i] = (nr, nc)

    print(f"Общее хранилище сообщений: {message_board}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Демо 4: Emergent Behavior — возникновение стратегий
# ═══════════════════════════════════════════════════════════════════════════════

def demo_emergent_behavior():
    """
    Пять агентов на кольцевой дорожке (12 ячеек). Каждый агент движется по кольцу.
    Если агент попадает на ячейку, занятую другим — минус reward.
    Если агент занимает пустую ячейку рядом с другим — плюс reward (бонус за компанию).
    Без явных правил агенты обнаружат паттерны: формирование кластеров,
    чередование позиций, «стайное» движение.
    """
    print("=" * 70)
    print("ДЕМО 4: Emergent Behavior — возникновение стратегий")
    print("=" * 70)
    print("5 агентов на кольцевой дорожке (12 ячеек).")
    print("Правила: штраф за столкновение, бонус за соседство.\n")

    RING_LEN = 12
    N_AGENTS = 5
    ACTIONS = [0, 1, 2]  # 0=стоять, 1=вперёд (+1), 2=назад (-1)
    N_EPISODES = 600
    MAX_STEPS = 25

    COLLISION_PENALTY = -5.0
    PROXIMITY_BONUS = 2.0
    STEP_PENALTY = -0.05

    agents = [
        QAgent(f"Agent{i}", len(ACTIONS), lr=0.1, gamma=0.92, epsilon=0.3)
        for i in range(N_AGENTS)
    ]

    def ring_step(positions: List[int], actions: List[int]):
        n = len(positions)
        new_positions = []
        for i in range(n):
            if actions[i] == 1:
                new_positions.append((positions[i] + 1) % RING_LEN)
            elif actions[i] == 2:
                new_positions.append((positions[i] - 1) % RING_LEN)
            else:
                new_positions.append(positions[i])

        rewards = [0.0] * n
        occupied = {}
        for i, p in enumerate(new_positions):
            occupied.setdefault(p, []).append(i)

        for i in range(n):
            rewards[i] += STEP_PENALTY
            # Столкновение
            if len(occupied[new_positions[i]]) > 1:
                rewards[i] += COLLISION_PENALTY
            # Бонус за ближайшего соседа
            for j in range(n):
                if i == j:
                    continue
                dist = min(abs(new_positions[i] - new_positions[j]),
                           RING_LEN - abs(new_positions[i] - new_positions[j]))
                if dist <= 2:
                    rewards[i] += PROXIMITY_BONUS
                    break

        return new_positions, rewards

    # Метрики emergence
    cluster_scores = []
    collision_counts = []

    for ep in range(N_EPISODES):
        positions = [i * 2 for i in range(N_AGENTS)]  # старт: 0, 2, 4, 6, 8
        ep_collisions = 0

        for _ in range(MAX_STEPS):
            actions = [agent.choose_action((pos,))
                       for agent, pos in zip(agents, positions)]
            next_positions, rewards = ring_step(positions, actions)

            for i in range(N_AGENTS):
                ns = (next_positions[i],)
                agents[i].update(positions[i], actions[i], rewards[i], ns)

            ep_collisions += sum(1 for i in range(N_AGENTS)
                                 if next_positions.count(next_positions[i]) > 1)
            positions = next_positions

        # Метрика кластеризации: среднее расстояние до ближайшего соседа
        min_dists = []
        for i in range(N_AGENTS):
            dists = [min(abs(positions[i] - positions[j]),
                         RING_LEN - abs(positions[i] - positions[j]))
                     for j in range(N_AGENTS) if i != j]
            min_dists.append(min(dists))
        avg_cluster = sum(min_dists) / len(min_dists) if min_dists else 0
        cluster_scores.append(avg_cluster)
        collision_counts.append(ep_collisions)

    print(f"Обучение: {N_EPISODES} эпизодов.")
    print(f"Среднее расстояние до ближайшего соседа (первые 50): "
          f"{sum(cluster_scores[:50])/50:.2f}")
    print(f"Среднее расстояние до ближайшего соседа (последние 50): "
          f"{sum(cluster_scores[-50:])/50:.2f}")
    print(f"Столкновений за эпизод (первые 50): "
          f"{sum(collision_counts[:50])/50:.1f}")
    print(f"Столкновений за эпизод (последние 50): "
          f"{sum(collision_counts[-50:])/50:.1f}")

    # Финальная симуляция
    positions = [i * 2 for i in range(N_AGENTS)]
    print(f"\nФинальная симуляция (жадная политика):")
    print(f"  Старт: {positions}")
    for t in range(MAX_STEPS):
        actions = [agent.choose_action((pos,))
                   for agent, pos in zip(agents, positions)]
        positions, rewards = ring_step(positions, actions)
        act_names = ["stay", "fwd", "back"]
        print(f"  Шаг {t + 1}: позиции={positions}, "
              f"действия={[act_names[a] for a in actions]}, "
              f"rewards={[f'{r:.1f}' for r in rewards]}")

    # Анализ паттернов
    print("\nАнализ стратегий:")
    for i, agent in enumerate(agents):
        stay_q = []
        fwd_q = []
        back_q = []
        for pos in range(RING_LEN):
            q = agent.policy_summary((pos,))
            stay_q.append(q[0])
            fwd_q.append(q[1])
            back_q.append(q[2])
        avg_stay = sum(stay_q) / len(stay_q)
        avg_fwd = sum(fwd_q) / len(fwd_q)
        avg_back = sum(back_q) / len(back_q)
        dominant = "stay" if avg_stay >= max(avg_fwd, avg_back) else (
            "forward" if avg_fwd >= avg_back else "backward")
        print(f"  Агент {i}: средние Q=[stay={avg_stay:.2f}, "
              f"fwd={avg_fwd:.2f}, back={avg_back:.2f}] → доминант: {dominant}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Multi-Agent Reinforcement Learning — основы")
    print("=" * 70)
    print()

    demo_cooperating_agents()
    demo_competitive_agents()
    demo_communication()
    demo_emergent_behavior()

    print("=" * 70)
    print("Итог: Multi-Agent RL")
    print("=" * 70)
    print("""
Ключевые концепции:

1. Cooperating Agents:
   — Агенты разделяют общую награду
   — Кооперация возникает через общий reward
   — Каждый агент учит свою часть задачи

2. Competitive Agents:
   — Агенты максимизируют свой reward за счёт другого
   — Возникает гонка, преследование, блокировка
   — Дифференциальная награда стимулирует конкуренцию

3. Communication:
   — Агенты делятся информацией через «сообщения»
   — Сообщения расширяют наблюдаемость среды
   — Q-таблица учитывает историю входящих сообщений

4. Emergent Behavior:
   — Стратегии возникают из взаимодействия агентов
   — Простые правила → сложное поведение
   — Кластеризация, чередование, стайное движение

Расширения:
   — CTDE (Centralized Training, Decentralized Execution)
   — MADDPG, QMIX, CommNet
   — Self-play и турнирное обучение
   — Адверсиальное обучение (GAN-подобные схемы)
""")
