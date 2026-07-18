"""
94 - SARSA vs Q-Learning: сравнение on-policy и off-policy методов
==================================================================

Реализация с нуля: SARSA (on-policy) и Q-Learning (off-policy)
на windy gridworld — среде с "ветром", сдвигающим агента.

Демо:
  1) SARSA — обучение
  2) Q-Learning — обучение
  3) Разница в политике (on-policy vs off-policy)
  4) Сравнение на windy gridworld
"""

import random
import math

random.seed(42)

# ─── Константы ───────────────────────────────────────────────────
ALPHA = 0.1          # скорость обучения
GAMMA = 0.99         # дисконтирование
EPSILON = 0.1        #epsilon-greedy
EPISODES = 200       # число эпизодов для обучения
MAX_STEPS = 200      # макс. шагов в эпизоде

# Действия: 0=up, 1=right, 2=down, 3=left
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}
DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


# ─── Windy Gridworld ─────────────────────────────────────────────
class WindyGridworld:
    """
    7x10 gridworld (как в Sutton & Barto, Example 6.5).
    Ветер дует вверх в столбцах 3-8 с разной силой.

    Старт: (3, 0), Цель: (3, 7)
    """

    def __init__(self, use_wind=True):
        self.rows = 7
        self.cols = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.use_wind = use_wind
        # ветер: сила сдвига вверх по столбцам
        # столбец: 0  1  2  3  4  5  6  7  8  9
        self.wind = [0,  0,  0,  1,  1,  1,  2,  2,  1,  0]

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        """Выполнить действие. Возвращает (next_state, reward, done)."""
        dr, dc = DELTA[action]
        r, c = self.pos
        nr = r + dr
        nc = c + dc

        # ветер: сдвигает вверх
        if self.use_wind:
            wind_strength = self.wind[nc] if 0 <= nc < self.cols else 0
            nr -= wind_strength

        # границы
        nr = max(0, min(self.rows - 1, nr))
        nc = max(0, min(self.cols - 1, nc))

        self.pos = (nr, nc)

        if self.pos == self.goal:
            return self.pos, 0, True
        return self.pos, -1, False


# ─── Epsilon-greedy выбор действия ───────────────────────────────
def epsilon_greedy(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_vals = q_table[state]
    max_q = max(q_vals)
    best = [a for a in ACTIONS if q_vals[a] == max_q]
    return random.choice(best)


# ─── SARSA: on-policy TD(0) ─────────────────────────────────────
def sarsa(env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
    """
    SARSA — State-Action-Reward-State-Action.
    Обновляет Q по действию, КОТОРОЕ ДЕЙСТВИТЕЛЬНО БУДЕТ ВЫПОЛНЕНО.
    On-policy: учитывает текущую epsilon-greedy политику.
    """
    q = {}
    for r in range(env.rows):
        for c in range(env.cols):
            q[(r, c)] = [0.0] * len(ACTIONS)

    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(q, state, epsilon)
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                q[state][action] += alpha * (reward - q[state][action])
                break

            next_action = epsilon_greedy(q, next_state, epsilon)

            # SARSA обновление: используем Q(next_state, next_action)
            td_target = reward + gamma * q[next_state][next_action]
            q[state][action] += alpha * (td_target - q[state][action])

            state = next_state
            action = next_action

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

    return q, rewards_per_episode, steps_per_episode


# ─── Q-Learning: off-policy TD(0) ───────────────────────────────
def q_learning(env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
    """
    Q-Learning — off-policy метод.
    Обновляет Q по ЛУЧШЕМУ действию из Q-таблицы (greedy target),
    хотя агент действует по epsilon-greedy.
    """
    q = {}
    for r in range(env.rows):
        for c in range(env.cols):
            q[(r, c)] = [0.0] * len(ACTIONS)

    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            action = epsilon_greedy(q, state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                q[state][action] += alpha * (reward - q[state][action])
                break

            # Q-Learning обновление: используем max_a Q(next_state, a)
            max_next_q = max(q[next_state])
            td_target = reward + gamma * max_next_q
            q[state][action] += alpha * (td_target - q[state][action])

            state = next_state

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

    return q, rewards_per_episode, steps_per_episode


# ─── Утилиты ─────────────────────────────────────────────────────
def avg_last(values, n=20):
    """Среднее по последним n значениям."""
    return sum(values[-n:]) / n


def extract_policy(q_table, env):
    """Извлечь greedy-политику из Q-таблицы."""
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            q_vals = q_table[state]
            max_q = max(q_vals)
            best = [a for a in ACTIONS if q_vals[a] == max_q]
            policy[state] = best[0]
    return policy


def simulate(env, q_table, max_steps=MAX_STEPS):
    """Прогон greed-политики и вернуть путь."""
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        q_vals = q_table[state]
        action = q_vals.index(max(q_vals))
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    return path


def print_grid(path, env, title=""):
    """Нарисовать сетку с путём."""
    if title:
        print(f"\n  {title}")

    grid = [["." for _ in range(env.cols)] for _ in range(env.rows)]

    for i, (r, c) in enumerate(path):
        if i == 0:
            grid[r][c] = "S"
        elif (r, c) == env.goal:
            grid[r][c] = "G"
        else:
            grid[r][c] = "*"

    grid[env.goal[0]][env.goal[1]] = "G"
    grid[env.start[0]][env.start[1]] = "S"

    print("    ", end="")
    for c in range(env.cols):
        print(f"{c:3}", end="")
    print()

    for r in range(env.rows):
        print(f"  {r:2} ", end="")
        for c in range(env.cols):
            print(f" {grid[r][c]:>2}", end="")
        print()


def print_q_summary(q_table, env):
    """Показать топ-5 состояний с наибольшим Q-значением."""
    all_q = []
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            max_q = max(q_table[state])
            all_q.append((max_q, state))
    all_q.sort(reverse=True)

    print("    Топ-5 состояний по Q-значению:")
    for max_q, state in all_q[:5]:
        best_action = ACTION_NAMES[q_table[state].index(max(q_table[state]))]
        print(f"      {state}: max_Q={max_q:.2f}  action={best_action}")


# ─── ДЕМО 1: SARSA — обучение ───────────────────────────────────
def demo1_sarsa():
    print("=" * 65)
    print("  ДЕМО 1: SARSA — On-Policy TD(0) обучение")
    print("=" * 65)
    print()

    env = WindyGridworld(use_wind=False)
    q_sarsa, rewards_sarsa, steps_sarsa = sarsa(env, episodes=EPISODES)

    print(f"  Эпизодов: {EPISODES}")
    print(f"  Гиперпараметры: α={ALPHA}, γ={GAMMA}, ε={EPSILON}")
    print()
    print("  Прогресс обучения (среднее по последним 20 эпизодам):")
    for milestone in [10, 50, 100, 150, 200]:
        avg_r = avg_last(rewards_sarsa[:milestone], min(20, milestone))
        avg_s = avg_last(steps_sarsa[:milestone], min(20, milestone))
        print(f"    Эпизод {milestone:>3}: reward={avg_r:>7.2f}  steps={avg_s:>6.1f}")

    print()
    print("  Финальная greedy-политика (SARSA):")
    policy = extract_policy(q_sarsa, env)
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("    ", end="")
    for c in range(env.cols):
        print(f"{c:3}", end="")
    print()
    for r in range(env.rows):
        print(f"  {r:2} ", end="")
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal:
                print("  G", end="")
            else:
                print(f"  {arrows[policy[state]]}", end="")
        print()

    path = simulate(env, q_sarsa)
    print()
    print_grid(path, env, "Путь greedy-политики SARSA:")
    print(f"    Длина пути: {len(path) - 1} шагов")

    print_q_summary(q_sarsa, env)
    print()


# ─── ДЕМО 2: Q-Learning — обучение ──────────────────────────────
def demo2_qlearning():
    print("=" * 65)
    print("  ДЕМО 2: Q-Learning — Off-Policy TD(0) обучение")
    print("=" * 65)
    print()

    env = WindyGridworld(use_wind=False)
    q_ql, rewards_ql, steps_ql = q_learning(env, episodes=EPISODES)

    print(f"  Эпизодов: {EPISODES}")
    print(f"  Гиперпараметры: α={ALPHA}, γ={GAMMA}, ε={EPSILON}")
    print()
    print("  Прогресс обучения (среднее по последним 20 эпизодам):")
    for milestone in [10, 50, 100, 150, 200]:
        avg_r = avg_last(rewards_ql[:milestone], min(20, milestone))
        avg_s = avg_last(steps_ql[:milestone], min(20, milestone))
        print(f"    Эпизод {milestone:>3}: reward={avg_r:>7.2f}  steps={avg_s:>6.1f}")

    print()
    print("  Финальная greedy-политика (Q-Learning):")
    policy = extract_policy(q_ql, env)
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("    ", end="")
    for c in range(env.cols):
        print(f"{c:3}", end="")
    print()
    for r in range(env.rows):
        print(f"  {r:2} ", end="")
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal:
                print("  G", end="")
            else:
                print(f"  {arrows[policy[state]]}", end="")
        print()

    path = simulate(env, q_ql)
    print()
    print_grid(path, env, "Путь greedy-политики Q-Learning:")
    print(f"    Длина пути: {len(path) - 1} шагов")

    print_q_summary(q_ql, env)
    print()


# ─── ДЕМО 3: Разница в политике (on-policy vs off-policy) ───────
def demo3_policy_difference():
    print("=" * 65)
    print("  ДЕМО 3: Разница в политике — on-policy vs off-policy")
    print("=" * 65)
    print()

    env = WindyGridworld(use_wind=False)

    # Обучаем оба метода
    q_sarsa, rewards_s, steps_s = sarsa(env, episodes=EPISODES)
    q_ql, rewards_q, steps_q = q_learning(env, episodes=EPISODES)

    policy_sarsa = extract_policy(q_sarsa, env)
    policy_ql = extract_policy(q_ql, env)

    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    # Показать политику SARSA
    print("  Политика SARSA (on-policy):")
    print("    ", end="")
    for c in range(env.cols):
        print(f"{c:3}", end="")
    print()
    for r in range(env.rows):
        print(f"  {r:2} ", end="")
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal:
                print("  G", end="")
            else:
                print(f"  {arrows[policy_sarsa[state]]}", end="")
        print()

    print()

    # Показать политику Q-Learning
    print("  Политика Q-Learning (off-policy):")
    print("    ", end="")
    for c in range(env.cols):
        print(f"{c:3}", end="")
    print()
    for r in range(env.rows):
        print(f"  {r:2} ", end="")
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal:
                print("  G", end="")
            else:
                print(f"  {arrows[policy_ql[state]]}", end="")
        print()

    # Подсчёт различий
    diff_count = 0
    total = env.rows * env.cols
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if policy_sarsa[state] != policy_ql[state]:
                diff_count += 1

    print()
    print(f"  Состояний с различной политикой: {diff_count}/{total}")
    print(f"  ({diff_count / total * 100:.1f}% отличаются)")
    print()
    print("  Ключевое отличие:")
    print("    SARSA  — учитывает exploration (epsilon-greedy), более осторожен")
    print("    Q-Learning — учитывает оптимум (max Q), более агрессивен")
    print()

    # Траектории
    path_sarsa = simulate(env, q_sarsa)
    path_ql = simulate(env, q_ql)

    print_grid(path_sarsa, env, "Путь SARSA (on-policy):")
    print(f"    Длина: {len(path_sarsa) - 1} шагов")
    print()
    print_grid(path_ql, env, "Путь Q-Learning (off-policy):")
    print(f"    Длина: {len(path_ql) - 1} шагов")
    print()


# ─── ДЕМО 4: Сравнение на windy gridworld ───────────────────────
def demo4_windy_comparison():
    print("=" * 65)
    print("  ДЕМО 4: Сравнение SARSA vs Q-Learning на Windy Gridworld")
    print("=" * 65)
    print()

    # Windy environment
    env_windy = WindyGridworld(use_wind=True)

    print("  Windy Gridworld:")
    print("    Сетка 7x10, старт=(3,0), цель=(3,7)")
    print("    Ветер дует вверх в столбцах 3-8:")
    print("    Столбец:  0  1  2  3  4  5  6  7  8  9")
    print("    Ветер:    0  0  0  1  1  1  2  2  1  0")
    print()

    # Запускаем с разными seed для разнообразия
    random.seed(42)
    q_sarsa_w, rewards_sw, steps_sw = sarsa(env_windy, episodes=EPISODES)
    random.seed(42)
    q_ql_w, rewards_qw, steps_qw = q_learning(env_windy, episodes=EPISODES)

    print("  Сравнение обучения:")
    print(f"    {'Эпизод':>8} | {'SARSA steps':>12} | {'QL steps':>12} | {'Разница':>8}")
    print("    " + "-" * 50)
    for milestone in [20, 50, 100, 150, 200]:
        avg_s = avg_last(steps_sw[:milestone], min(20, milestone))
        avg_q = avg_last(steps_qw[:milestone], min(20, milestone))
        diff = avg_q - avg_s
        marker = "<-- SARSA быстрее" if diff > 5 else ("--> QL быстрее" if diff < -5 else "")
        print(f"    {milestone:>8} | {avg_s:>12.1f} | {avg_q:>12.1f} | {diff:>+8.1f} {marker}")

    print()

    # Greedy политики
    path_sw = simulate(env_windy, q_sarsa_w)
    path_qw = simulate(env_windy, q_ql_w)

    print_grid(path_sw, env_windy, "SARSA на windy gridworld:")
    print(f"    Длина пути: {len(path_sw) - 1} шагов")
    print()
    print_grid(path_qw, env_windy, "Q-Learning на windy gridworld:")
    print(f"    Длина пути: {len(path_qw) - 1} шагов")

    print()
    print("  Анализ результатов:")
    print("  " + "-" * 60)

    policy_sw = extract_policy(q_sarsa_w, env_windy)
    policy_qw = extract_policy(q_ql_w, env_windy)

    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    # Сравнение политик
    print("  SARSA (on-policy) ищет безопасный путь:")
    print("    Учитывает, что exploration сepsilon=0.1 приведёт")
    print("    к случайным действиям → избегает опасных переходов")
    print()
    print("  Q-Learning (off-policy) ищет оптимальный путь:")
    print("    Обновляет Q по max_a Q(s',a), игнорируя exploration")
    print("    → может найти более короткий, но рискованный путь")
    print()

    # Q-значения у цели
    goal = env_windy.goal
    print(f"  Q-значения у цели {goal}:")
    print(f"    SARSA:      {[f'{v:.2f}' for v in q_sarsa_w[goal]]}")
    print(f"    Q-Learning: {[f'{v:.2f}' for v in q_ql_w[goal]]}")

    # Разница в стратегии у опасных состояний
    print()
    print("  Ключевые различия в поведении:")
    print("    1. SARSA: осторожный → длиннее, но стабильнее путь")
    print("    2. Q-Learning: агрессивный → короче, но может 'уронить' агента")
    print("    3. На windy gridworld: ветер делает SARSA ещё осторожнее")
    print("    4. Q-Learning может игнорировать ветер в оптимизации")
    print()


# ─── Запуск всех демо ───────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  " + "=" * 65)
    print("  SARSA vs Q-Learning: On-Policy vs Off-Policy")
    print("  " + "=" * 65)
    print()
    print("  Теория:")
    print("    SARSA (State-Action-Reward-State-Action):")
    print("      Q(s,a) += α [r + γ Q(s',a') - Q(s,a)]")
    print("      где a' — действие, которое ДЕЙСТВИТЕЛЬНО будет выполнено")
    print()
    print("    Q-Learning:")
    print("      Q(s,a) += α [r + γ max_a' Q(s',a') - Q(s,a)]")
    print("      где max_a' — лучшее действие по Q-таблице")
    print()
    print("    Ключевая разница:")
    print("      SARSA:  целевая Q(s', a') — on-policy (учитывает exploration)")
    print("      QL:     целевая max Q(s',a') — off-policy (игнорирует exploration)")
    print()

    demo1_sarsa()
    demo2_qlearning()
    demo3_policy_difference()
    demo4_windy_comparison()

    print("  " + "=" * 65)
    print("  Итог: SARSA — безопаснее, Q-Learning — агрессивнее")
    print("  " + "=" * 65)
