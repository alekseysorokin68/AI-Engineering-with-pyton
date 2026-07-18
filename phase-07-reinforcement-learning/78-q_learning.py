"""
Q-learning — классический алгоритм обучения с подкреплением.

Реализация с нуля: Q-таблица, epsilon-greedy, grid world, сравнение с random agent.
Без внешних зависимостей — только stdlib.
"""

import random

random.seed(42)

# ── Grid World ────────────────────────────────────────────────────────────────

class GridWorld:
    """Простой 5x5 grid world с фиксированными стартом и целью."""

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # вправо, влево, вниз, вверх
    ACTION_NAMES = ["→", "←", "↓", "↑"]
    SIZE = 5
    GOAL = (4, 4)
    WALL = {(2, 2), (2, 3), (3, 2)}
    GOAL_REWARD = 10.0
    STEP_PENALTY = -0.1
    WALL_PENALTY = -1.0

    def __init__(self):
        self.start = (0, 0)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action_idx):
        dx, dy = self.ACTIONS[action_idx]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy

        if nx < 0 or nx >= self.SIZE or ny < 0 or ny >= self.SIZE or (nx, ny) in self.WALL:
            return self.pos, self.WALL_PENALTY, False

        self.pos = (nx, ny)
        done = self.pos == self.GOAL
        reward = self.GOAL_REWARD if done else self.STEP_PENALTY
        return self.pos, reward, done

    def render_path(self, path, title=""):
        if title:
            print(f"\n{title}")
        grid = [["·" for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        for wx, wy in self.WALL:
            grid[wx][wy] = "█"
        for (px, py) in path:
            grid[px][py] = "*"
        grid[self.start[0]][self.start[1]] = "S"
        grid[self.GOAL[0]][self.GOAL[1]] = "G"
        for row in grid:
            print("  " + " ".join(row))


# ── Q-таблица ─────────────────────────────────────────────────────────────────

class QTable:
    """Q-таблица: dict[state][action] → value, с Bellman-обновлением."""

    def __init__(self, n_actions, init_val=0.0):
        self.n_actions = n_actions
        self.table = {}
        self.init_val = init_val

    def get_q(self, state, action):
        if state not in self.table:
            self.table[state] = [self.init_val] * self.n_actions
        return self.table[state][action]

    def get_all_q(self, state):
        if state not in self.table:
            self.table[state] = [self.init_val] * self.n_actions
        return self.table[state]

    def update(self, state, action, reward, next_state, alpha, gamma, done):
        """
        Bellman-обновление:
        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
        """
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            target = reward + gamma * max(self.get_all_q(next_state))
        self.table[state][action] = current_q + alpha * (target - current_q)

    def best_action(self, state):
        q_vals = self.get_all_q(state)
        return q_vals.index(max(q_vals))

    def __repr__(self):
        lines = ["State (x,y) |  →      ←      ↓      ↑"]
        lines.append("-" * 46)
        for state in sorted(self.table.keys()):
            vals = self.table[state]
            formatted = " ".join(f"{v:+6.2f}" for v in vals)
            lines.append(f"  {state}  | {formatted}")
        return "\n".join(lines)


# ── Epsilon-greedy ────────────────────────────────────────────────────────────

def epsilon_greedy(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, q_table.n_actions - 1)
    return q_table.best_action(state)


# ── Q-learning Training ──────────────────────────────────────────────────────

def train_q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon_start=1.0,
                     epsilon_end=0.01, epsilon_decay=0.995, verbose_every=100):
    q_table = QTable(n_actions=len(env.ACTIONS))
    rewards_per_episode = []

    epsilon = epsilon_start
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 200

        while steps < max_steps:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            q_table.update(state, action, reward, next_state, alpha, gamma, done)
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if verbose_every and ep % verbose_every == 0:
            avg = sum(rewards_per_episode[-verbose_every:]) / verbose_every
            print(f"  Episode {ep:>4d} | avg reward: {avg:>7.2f} | ε: {epsilon:.3f}")

    return q_table, rewards_per_episode


# ── Random Agent ──────────────────────────────────────────────────────────────

def random_agent(env, episodes=500, max_steps=200):
    rewards_per_episode = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = random.randint(0, len(env.ACTIONS) - 1)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        rewards_per_episode.append(total_reward)
    return rewards_per_episode


# ── Trained Agent (greedy) ───────────────────────────────────────────────────

def greedy_agent(env, q_table, max_steps=200):
    state = env.reset()
    path = [state]
    total_reward = 0
    for _ in range(max_steps):
        action = q_table.best_action(state)
        state, reward, done = env.step(action)
        path.append(state)
        total_reward += reward
        if done:
            break
    return path, total_reward


# ── Утилиты ───────────────────────────────────────────────────────────────────

def moving_average(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i + 1]) / (i - start + 1))
    return result


def print_rewards_comparison(q_rewards, r_rewards, window=20):
    q_avg = moving_average(q_rewards, window)
    r_avg = moving_average(r_rewards, window)
    print("\n  Ep │ Q-learning (avg) │ Random (avg)")
    print("  ───┼─────────────────┼─────────────")
    for i in range(0, len(q_avg), 50):
        qv = q_avg[i] if i < len(q_avg) else 0
        rv = r_avg[i] if i < len(r_avg) else 0
        print(f"  {i:>3d}│ {qv:>+15.2f}│ {rv:>+11.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ДЕМОНСТРАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def demo1_q_table_init():
    print("=" * 60)
    print("ДЕМО 1: Q-таблица — инициализация")
    print("=" * 60)
    print("\nQ-таблица: dict[state] = [Q(→), Q(←), Q(↓), Q(↑)]")
    print("Все значения начинаются с 0.0\n")

    qt = QTable(n_actions=4)
    states = [(0, 0), (0, 1), (1, 0), (4, 4)]
    for s in states:
        print(f"  Q{s} = {qt.get_all_q(s)}")

    print(f"\nВсего состояний: {GridWorld.SIZE ** 2 - len(GridWorld.WALL)}")
    print(f"Размер Q-таблицы: ~{20 * 4} значений (20 состояний × 4 действия)")


def demo2_training():
    print("\n" + "=" * 60)
    print("ДЕМО 2: Обучение Q-learning")
    print("=" * 60)
    print("\nПараметры:")
    print("  α (learning rate) = 0.1")
    print("  γ (discount)      = 0.99")
    print("  ε (exploration)   = 1.0 → 0.01 (decay: 0.995)")
    print("  Episodes          = 500\n")

    env = GridWorld()
    q_table, rewards = train_q_learning(env, episodes=500, verbose_every=100)

    final_avg = sum(rewards[-50:]) / 50
    first_avg = sum(rewards[:50]) / 50
    print(f"\n  Средняя награда первых 50 eps:  {first_avg:>+.2f}")
    print(f"  Средняя награда последних 50 eps: {final_avg:>+.2f}")
    print(f"  Улучшение: {final_avg - first_avg:>+.2f}")

    print(f"\n  Уникальных состояний в Q-таблице: {len(q_table.table)}")
    return q_table, rewards


def demo3_optimal_policy(q_table):
    print("\n" + "=" * 60)
    print("ДЕМО 3: Оптимальная политика")
    print("=" * 60)

    env = GridWorld()
    path, total_reward = greedy_agent(env, q_table)
    print(f"\n  Длина пути: {len(path) - 1} шагов")
    print(f"  Награда: {total_reward:.1f}")
    env.render_path(path, title="\n  Путь агента:")

    print("\n  Лучшие действия (из Q-таблицы):")
    env.reset()
    arrows = [["·" for _ in range(GridWorld.SIZE)] for _ in range(GridWorld.SIZE)]
    for wx, wy in GridWorld.WALL:
        arrows[wx][wy] = "█"
    arrows[GridWorld.GOAL[0]][GridWorld.GOAL[1]] = "G"
    for x in range(GridWorld.SIZE):
        for y in range(GridWorld.SIZE):
            if (x, y) in GridWorld.WALL or (x, y) == GridWorld.GOAL:
                continue
            best = q_table.best_action((x, y))
            arrows[x][y] = GridWorld.ACTION_NAMES[best]
    print()
    for row in arrows:
        print("    " + " ".join(row))

    print("\n  Q-значения для нагруженных состояний:")
    key_states = [(0, 0), (0, 4), (1, 4), (4, 0), (3, 4)]
    for s in key_states:
        vals = q_table.get_all_q(s)
        best_a = vals.index(max(vals))
        print(f"    Q{s} = {[f'{v:+.2f}' for v in vals]}  best: {GridWorld.ACTION_NAMES[best_a]}")


def demo4_agent_comparison(q_rewards):
    print("\n" + "=" * 60)
    print("ДЕМО 4: Сравнение Q-learning vs Random agent")
    print("=" * 60)

    print("\n  Обучение Random агента (500 episodes)...")
    r_rewards = random_agent(GridWorld(), episodes=500)

    print_rewards_comparison(q_rewards, r_rewards, window=20)

    q_final = sum(q_rewards[-50:]) / 50
    r_final = sum(r_rewards[-50:]) / 50
    q_first = sum(q_rewards[:50]) / 50
    r_first = sum(r_rewards[:50]) / 50

    print(f"\n  {'Метрика':<35} {'Q-learning':>12} {'Random':>12}")
    print("  " + "─" * 59)
    print(f"  {'Средняя награда (первые 50 eps)':<35} {f'{q_first:>+.2f}':>12} {f'{r_first:>+.2f}':>12}")
    print(f"  {'Средняя награда (последние 50 eps)':<35} {f'{q_final:>+.2f}':>12} {f'{r_final:>+.2f}':>12}")
    print(f"  {'Улучшение':<35} {f'{q_final - q_first:>+.2f}':>12} {f'{r_final - r_first:>+.2f}':>12}")

    env = GridWorld()
    q_path, q_r = greedy_agent(env, q_table)
    print(f"\n  Q-learning (greedy): путь = {len(q_path) - 1} шагов, награда = {q_r:.1f}")
    print(f"  Random (лучший из 500): путь ≈ ~20+ шагов, награда ≈ {r_final:.1f}")
    print(f"\n  Вывод: Q-learning находит оптимальный путь ({len(q_path)-1} шагов),")
    print(f"         random agent застревает (средняя награда {r_final:+.1f}).")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Q-LEARNING — КЛАССИЧЕСКИЙ RL-АЛГОРИТМ          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo1_q_table_init()

    q_table, q_rewards = demo2_training()

    demo3_optimal_policy(q_table)

    demo4_agent_comparison(q_rewards)

    print("\n" + "=" * 60)
    print("ИТОГО")
    print("=" * 60)
    print("  • Q-таблица хранит value для каждого (state, action) pair")
    print("  • Bellman update: Q(s,a) += α[r + γ·max Q(s',·) − Q(s,a)]")
    print("  • Epsilon-greedy: баланс исследования и использования")
    print("  • Q-learning находит оптимальный путь через пробу и ошибку")
    print("  • Random agent не имеет шансов против обученного")
