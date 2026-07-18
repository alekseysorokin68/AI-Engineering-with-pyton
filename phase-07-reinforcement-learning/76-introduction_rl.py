"""
Основы обучения с подкреплением (Reinforcement Learning).

Ключевые концепции:
- Среда (Environment): мир, в котором действует агент
- Состояние (State): описание текущей ситуации
- Действие (Action): выбор агента
- Вознаграждение (Reward): числовая оценка за действие
- Политика (Policy): стратегия выбора действий
- Эпизод (Episode): полная последовательность взаимодействия
"""

import random
import time

random.seed(42)

# ============================================================
# 1. СРЕДА (ENVIRONMENT)
# ============================================================

class GridWorld:
    """
    Простая сеточная среда 5x5.

    Сетка:
      '.' — пустая клетка (проходимая)
      'S' — стартовая позиция агента
      'G' — цель (агент получает награду +10)
      'X' — стена (непроходимая)
      'T' — ловушка (агент получает награду -5)

    Агент может двигаться: вверх, вниз, влево, вправо.
    """

    ACTIONS = ["up", "down", "left", "right"]
    ACTION_DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self, grid=None):
        if grid is None:
            grid = [
                ["S", ".", ".", ".", "."],
                [".", "X", "X", ".", "."],
                [".", ".", ".", "X", "."],
                [".", "X", "T", ".", "."],
                [".", ".", ".", ".", "G"],
            ]
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.agent_pos = self._find("S")
        self.goal_pos = self._find("G")
        self.start_pos = self.agent_pos
        self.steps = 0
        self.max_steps = 100

    def _find(self, symbol):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == symbol:
                    return (r, c)
        return None

    def reset(self):
        """Сброс среды в начальное состояние."""
        self.agent_pos = self.start_pos
        self.steps = 0
        return self.agent_pos

    def get_state(self):
        """Текущее состояние = позиция агента."""
        return self.agent_pos

    def get_states_count(self):
        """Общее количество возможных состояний."""
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != "X":
                    count += 1
        return count

    def step(self, action):
        """
        Выполнить действие и вернуть (новое_состояние, награда, завершён_ли).
        """
        self.steps += 1
        dr, dc = self.ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Проверка границ и стен
        if (0 <= new_r < self.rows and 0 <= new_c < self.cols
                and self.grid[new_r][new_c] != "X"):
            self.agent_pos = (new_r, new_c)
        # Иначе агент остаётся на месте

        cell = self.grid[self.agent_pos[0]][self.agent_pos[1]]

        # Определяем награду
        done = False
        if cell == "G":
            reward = 10.0
            done = True
        elif cell == "T":
            reward = -5.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -1.0
            done = True
        else:
            reward = -0.1  # маленький штраф за каждый шаг (поощрение к скорости)

        return self.get_state(), reward, done

    def render(self):
        """Отрисовать текущее состояние среды."""
        symbols = []
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row_str += " A"
                else:
                    row_str += f" {self.grid[r][c]}"
            symbols.append(row_str)
        return "\n".join(symbols)


# ============================================================
# 2. АГЕНТ (AGENT)
# ============================================================

class RandomAgent:
    """Агент со случайной политикой — выбирает действия случайно."""

    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, state):
        """Случайный выбор действия."""
        return random.choice(self.actions)

    def learn(self, state, action, reward, next_state, done):
        """Случайный агент не обучается."""
        pass


class QLearningAgent:
    """
    Агент с Q-learning — обучается через таблицу Q-значений.

    Q(s, a) = Q(s, a) + lr * (reward + gamma * max(Q(s', a')) - Q(s, a))
    """

    def __init__(self, actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # {(state, action): q_value}

    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Epsilon-greedy политика: с вероятностью epsilon — случайное действие."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Выбираем действие с максимальным Q-значением
        q_values = [(self._get_q(state, a), a) for a in self.actions]
        max_q = max(q_values, key=lambda x: x[0])
        # Если есть несколько действий с одинаковым Q — выбираем случайно среди них
        best = [a for q, a in q_values if q == max_q[0]]
        return random.choice(best)

    def learn(self, state, action, reward, next_state, done):
        """Обновление Q-таблицы."""
        current_q = self._get_q(state, action)
        if done:
            target = reward
        else:
            max_next_q = max(self._get_q(next_state, a) for a in self.actions)
            target = reward + self.gamma * max_next_q
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q


# ============================================================
# 3. ВОЗНАГРАЖДЕНИЕ (REWARD) — обёртка
# ============================================================

class RewardTracker:
    """Отслеживает статистику вознаграждений по эпизодам."""

    def __init__(self):
        self.episode_rewards = []

    def start_episode(self):
        self.current_reward = 0.0

    def add(self, reward):
        self.current_reward += reward

    def end_episode(self):
        self.episode_rewards.append(self.current_reward)
        return self.current_reward

    def average(self, n=10):
        if len(self.episode_rewards) < n:
            n = len(self.episode_rewards)
        if n == 0:
            return 0.0
        return sum(self.episode_rewards[-n:]) / n


# ============================================================
# 4. ЦИКЛ ВЗАИМОДЕЙСТВИЯ (INTERACTION LOOP)
# ============================================================

def run_episode(env, agent, tracker, verbose=False):
    """Запуск одного эпизода взаимодействия агента со средой."""
    state = env.reset()
    tracker.start_episode()
    total_reward = 0.0
    steps = 0
    done = False

    if verbose:
        print(f"\nСтарт: {state}")

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        tracker.add(reward)
        total_reward += reward
        steps += 1
        state = next_state

        if verbose:
            print(f"  Шаг {steps}: действие={action}, "
                  f"позиция={state}, награда={reward:.2f}, done={done}")

    episode_reward = tracker.end_episode()
    return episode_reward, steps


def train_agent(env, agent, n_episodes=500, verbose_interval=100):
    """Обучение агента на n эпизодах."""
    tracker = RewardTracker()
    for ep in range(1, n_episodes + 1):
        reward, steps = run_episode(env, agent, tracker)
        if ep % verbose_interval == 0:
            avg = tracker.average(verbose_interval)
            print(f"  Эпизод {ep:>5d} | "
                  f"Средняя награда (последние {verbose_interval}): {avg:>7.2f} | "
                  f"Q-таблица: {len(agent.q_table)} записей")
    return tracker


# ============================================================
# ДЕМО 1: Простая среда (Grid World)
# ============================================================

def demo_1_environment():
    print("=" * 60)
    print("ДЕМО 1: Среда — Grid World 5x5")
    print("=" * 60)

    env = GridWorld()

    print("\nНачальное состояние среды:")
    print(env.render())

    print(f"\nРазмер сетки: {env.rows}x{env.cols}")
    print(f"Количество состояний: {env.get_states_count()}")
    print(f"Действия: {env.ACTIONS}")
    print(f"Цель (G): награда +10")
    print(f"Ловушка (T): награда -5")
    print(f"Стены (X): непроходимы")
    print(f"Каждый шаг: награда -0.1")

    # Демонстрация хода
    print("\n--- Демонстрация: 5 случайных шагов ---")
    env.reset()
    for i in range(5):
        action = random.choice(env.ACTIONS)
        state, reward, done = env.step(action)
        print(f"  Шаг {i+1}: действие={action:>5s} -> позиция={state}, "
              f"награда={reward:.1f}, завершён={done}")
        if done:
            print("  Эпизод завершён!")
            break

    print()


# ============================================================
# ДЕМО 2: Агент со случайной политикой
# ============================================================

def demo_2_random_agent():
    print("=" * 60)
    print("ДЕМО 2: Агент со случайной политикой")
    print("=" * 60)

    env = GridWorld()
    agent = RandomAgent(env.ACTIONS)
    tracker = RewardTracker()

    print("\nЗапуск 10 эпизодов со случайной политикой:\n")

    for ep in range(1, 11):
        reward, steps = run_episode(env, agent, tracker)
        print(f"  Эпизод {ep:>2d}: "
              f"награда={reward:>6.2f}, шагов={steps:>3d}")

    avg = tracker.average(10)
    print(f"\nСредняя награда за 10 эпизодов: {avg:.2f}")

    # Показать, что случайный агент часто попадает в ловушку
    trap_count = sum(1 for r in tracker.episode_rewards if r < 0)
    print(f"Эпизодов с отрицательной наградой: {trap_count}/10")
    print()


# ============================================================
# ДЕМО 3: Цикл взаимодействия (подробный)
# ============================================================

def demo_3_interaction_loop():
    print("=" * 60)
    print("ДЕМО 3: Цикл взаимодействия (подробный разбор)")
    print("=" * 60)

    env = GridWorld()
    agent = QLearningAgent(env.ACTIONS, epsilon=1.0)  # полный рандом для наглядности
    tracker = RewardTracker()

    print("\nОдин эпизод с подробным выводом:\n")
    state = env.reset()
    tracker.start_episode()
    total_reward = 0.0
    steps = 0
    done = False

    print("Начальная позиция:", state)
    print(f"{'Шаг':>4s} | {'Действие':>8s} | {'Позиция':>10s} | {'Награда':>8s} | {'Этап'}")
    print("-" * 55)

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        total_reward += reward
        steps += 1
        state = next_state

        if reward > 5:
            stage = "ЦЕЛЬ!"
        elif reward < -1:
            stage = "ЛОВУШКА!"
        elif done and reward < 0:
            stage = "тайм-аут"
        else:
            stage = "шаг"

        print(f"{steps:>4d} | {action:>8s} | {str(state):>10s} | {reward:>8.2f} | {stage}")

    print("-" * 55)
    print(f"\nИтого: {steps} шагов, общая награда: {total_reward:.2f}")
    print()


# ============================================================
# ДЕМО 4: Сравнение случайного vs обученного агента
# ============================================================

def demo_4_comparison():
    print("=" * 60)
    print("ДЕМО 4: Сравнение случайного vs обученного агента")
    print("=" * 60)

    # --- Случайный агент ---
    print("\n--- Случайный агент (500 эпизодов) ---\n")
    env1 = GridWorld()
    random_ag = RandomAgent(env1.ACTIONS)
    tracker1 = RewardTracker()
    for ep in range(1, 501):
        run_episode(env1, random_ag, tracker1)
    avg_random = tracker1.average(500)
    print(f"  Средняя награда за 500 эпизодов: {avg_random:.2f}")

    # --- Обученный агент (Q-learning) ---
    print("\n--- Q-learning агент (обучение: 500 эпизодов) ---\n")
    env2 = GridWorld()
    q_agent = QLearningAgent(env2.ACTIONS, learning_rate=0.1, gamma=0.99, epsilon=0.3)
    train_agent(env2, q_agent, n_episodes=500, verbose_interval=100)

    # --- Оценка обученного агента (без exploration) ---
    print("\n--- Оценка обученного агента (epsilon=0, 50 эпизодов) ---\n")
    env3 = GridWorld()
    eval_agent = QLearningAgent(env3.ACTIONS, learning_rate=0, gamma=0.99, epsilon=0)
    eval_agent.q_table = q_agent.q_table.copy()  # копируем выученную Q-таблицу
    tracker3 = RewardTracker()
    for ep in range(1, 51):
        reward, steps = run_episode(env3, eval_agent, tracker3)
        if ep % 10 == 0:
            avg = tracker3.average(10)
            print(f"  Эпизод {ep:>2d}: награда={reward:>6.2f}, "
                  f"шагов={steps:>3d}, средняя={avg:.2f}")

    avg_trained = tracker3.average(50)
    print(f"\n  Средняя награда обученного агента: {avg_trained:.2f}")
    print(f"  Средняя награда случайного агента:  {avg_random:.2f}")
    print(f"  Улучшение: {avg_trained - avg_random:+.2f}")

    # --- Пример найденного маршрута ---
    print("\n--- Пример оптимального маршрута обученного агента ---\n")
    env4 = GridWorld()
    eval_agent2 = QLearningAgent(env4.ACTIONS, learning_rate=0, gamma=0.99, epsilon=0)
    eval_agent2.q_table = q_agent.q_table.copy()
    state = env4.reset()
    print(env4.render())
    done = False
    step = 0
    while not done:
        action = eval_agent2.choose_action(state)
        state, reward, done = env4.step(action)
        step += 1
        print(f"\n  Шаг {step}: {action}")
        print(env4.render())
    print(f"\n  Итого шагов: {step}, финальная награда: {reward:.2f}")

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("ОБУЧЕНИЕ С ПОДКРЕПЛЕНИЕМ: ОСНОВНЫЕ КОНЦЕПЦИИ")
    print("=" * 60)
    print()
    print("Концепции RL:")
    print("  Agent (Агент)     — принимает решения")
    print("  Environment       — мир, в котором действует агент")
    print("  State (Состояние) — текущая ситуация")
    print("  Action (Действие) — выбор агента")
    print("  Reward (Награда)  — обратная связь от среды")
    print("  Policy (Политика) — стратегия выбора действий")
    print("  Q-learning        — обучение ценности действий")
    print()

    demo_1_environment()
    demo_2_random_agent()
    demo_3_interaction_loop()
    demo_4_comparison()

    print("=" * 60)
    print("ВСЕ ДЕМО ЗАВЕРШЕНЫ")
    print("=" * 60)
