"""
Monte Carlo методы в Reinforcement Learning
============================================
Самодостаточный файл: random.seed(42), без внешних зависимостей.

Содержание:
  1. Monte Carlo policy evaluation (first-visit и every-visit)
  2. Monte Carlo control с ε-greedy политикой
  3. Сравнение с TD(0) методом
"""

import random
from collections import defaultdict

random.seed(42)

# ============================================================
# 1. Среда: Gridworld 4x4 (детерминированная)
# ============================================================

class GridWorld:
    """4x4 gridworld. S=0 (top-left) — start, T=15 (bottom-right) — terminal.
    Действия: 0=up, 1=right, 2=down, 3=left.
    Награда: -1 за каждый шаг, 0 при достижении терминала.
    """

    ACTIONS = [0, 1, 2, 3]  # up, right, down, left
    DIR = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ROWS, COLS = 4, 4
    START = 0
    TERMINAL = 15

    def __init__(self):
        self.n_states = self.ROWS * self.COLS
        self.n_actions = len(self.ACTIONS)

    def _rc(self, s):
        return s // self.COLS, s % self.COLS

    def _s(self, r, c):
        return r * self.COLS + c

    def step(self, state, action):
        if state == self.TERMINAL:
            return state, 0.0, True
        r, c = self._rc(state)
        dr, dc = self.DIR[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
            ns = self._s(nr, nc)
        else:
            ns = state  # bumps into wall
        done = ns == self.TERMINAL
        reward = 0.0 if done else -1.0
        return ns, reward, done

    def reset(self):
        return self.START


# ============================================================
# 2. Генерация эпизода (episode)
# ============================================================

def generate_episode(env, policy):
    """Генерирует один эпизод: список (state, action, reward)."""
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
    return episode


# ============================================================
# 3. Monte Carlo Policy Evaluation
# ============================================================

def mc_prediction_first_visit(env, policy, n_episodes=5000):
    """First-visit MC: обновляем V(s) только при первом посещении s в эпизоде."""
    returns = defaultdict(list)
    V = [0.0] * env.n_states

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + G  # return от шага t
            if s not in visited:
                returns[s].append(G)
                V[s] = sum(returns[s]) / len(returns[s])
                visited.add(s)
    return V


def mc_prediction_every_visit(env, policy, n_episodes=5000):
    """Every-visit MC: обновляем V(s) при каждом посещении s в эпизоде."""
    returns = defaultdict(list)
    V = [0.0] * env.n_states

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + G
            returns[s].append(G)
            V[s] = sum(returns[s]) / len(returns[s])
    return V


# ============================================================
# 4. Monte Carlo Control (ε-greedy)
# ============================================================

def mc_control(env, n_episodes=10000, epsilon=0.1, gamma=1.0):
    """First-visit MC control с ε-greedy политикой.
    Возвращает (Q, policy)."""
    Q = [[0.0] * env.n_actions for _ in range(env.n_states)]
    returns_count = [[0] * env.n_actions for _ in range(env.n_states)]
    returns_sum = [[0.0] * env.n_actions for _ in range(env.n_states)]

    def make_policy(state):
        if random.random() < epsilon:
            return random.choice(env.ACTIONS)
        return Q[state].index(max(Q[state]))

    for ep in range(n_episodes):
        episode = generate_episode(env, make_policy)
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            pair = (s, a)
            if pair not in visited:
                visited.add(pair)
                returns_count[s][a] += 1
                returns_sum[s][a] += G
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]

    # итоговая жадная политика
    final_policy = [Q[s].index(max(Q[s])) for s in range(env.n_states)]
    return Q, final_policy


# ============================================================
# 5. TD(0) для сравнения
# ============================================================

def td0_evaluation(env, policy, alpha=0.1, n_episodes=5000, gamma=1.0):
    """TD(0) prediction для сравнения с MC."""
    V = [0.0] * env.n_states
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(state, action)
            target = reward if done else reward + gamma * V[next_state]
            V[state] += alpha * (target - V[state])
            state = next_state
    return V


# ============================================================
# Демонстрации
# ============================================================

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


def print_grid(values, title):
    env = GridWorld()
    print(f"\n  {title}")
    print("  " + "-" * 24)
    for r in range(env.ROWS):
        row_vals = []
        for c in range(env.COLS):
            s = env._s(r, c)
            if s == env.TERMINAL:
                row_vals.append("  TERM")
            else:
                row_vals.append(f"{values[s]:6.2f}")
        print("  | " + " | ".join(row_vals) + " |")
    print("  " + "-" * 24)


def print_policy(policy, title):
    env = GridWorld()
    print(f"\n  {title}")
    print("  " + "-" * 28)
    for r in range(env.ROWS):
        row_vals = []
        for c in range(env.COLS):
            s = env._s(r, c)
            if s == env.TERMINAL:
                row_vals.append(" TERM ")
            else:
                row_vals.append(f" {ACTION_NAMES[policy[s]]:>5}")
        print("  | " + " | ".join(row_vals) + " |")
    print("  " + "-" * 28)


def demo1_mc_evaluation():
    """Демо 1: Monte Carlo оценка ценности."""
    print("=" * 60)
    print("  ДЕМО 1: Monte Carlo Policy Evaluation")
    print("=" * 60)

    env = GridWorld()
    # Жадная политика: всегда идти вниз-вправо (к терминалу)
    def greedy_policy(state):
        return 2 if state < 12 else 1  # down if not last row, else right

    V_fv = mc_prediction_first_visit(env, greedy_policy, n_episodes=5000)
    V_ev = mc_prediction_every_visit(env, greedy_policy, n_episodes=5000)

    print_grid(V_fv, "First-Visit MC V(s)")
    print_grid(V_ev, "Every-Visit MC V(s)")

    diff = sum(abs(V_fv[s] - V_ev[s]) for s in range(env.n_states))
    print(f"\n  Суммарная |разница| между first/every visit: {diff:.4f}")
    print("  (Должна быть очень мала — обе оценки сходятся к V^π)")


def demo2_first_vs_every():
    """Демо 2: First-visit vs Every-visit — разная скорость сходимости."""
    print("\n" + "=" * 60)
    print("  ДЕМО 2: First-visit vs Every-visit — Сходимость")
    print("=" * 60)

    env = GridWorld()
    # Случайная политика — создаёт вариативность в returns
    def random_policy(state):
        return random.choice(env.ACTIONS)

    V_fv = [0.0] * env.n_states
    V_ev = [0.0] * env.n_states
    ret_count_fv = defaultdict(int)
    ret_sum_fv = defaultdict(float)
    ret_count_ev = defaultdict(int)
    ret_sum_ev = defaultdict(float)

    # эталон — много эпизодов со случайной политикой
    V_ref = mc_prediction_first_visit(env, random_policy, n_episodes=10000)

    print("\n  Случайная политика — отслеживаем сходимость V(s=0):")
    print(f"  Эталон V(0) = {V_ref[0]:.3f}\n")

    checkpoints = [10, 50, 100, 500, 1000, 2000]
    for ep in range(1, max(checkpoints) + 1):
        episode = generate_episode(env, random_policy)
        G = 0.0
        visited_fv = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + G

            # Every-visit
            ret_count_ev[s] += 1
            ret_sum_ev[s] += G
            V_ev[s] = ret_sum_ev[s] / ret_count_ev[s]

            # First-visit
            if s not in visited_fv:
                ret_count_fv[s] += 1
                ret_sum_fv[s] += G
                V_fv[s] = ret_sum_fv[s] / ret_count_fv[s]
                visited_fv.add(s)

        if ep in checkpoints:
            fv_err = sum(abs(V_fv[s] - V_ref[s]) for s in range(env.n_states))
            ev_err = sum(abs(V_ev[s] - V_ref[s]) for s in range(env.n_states))
            print(f"  Эп {ep:>5d}  V_fv(0)={V_fv[0]:6.2f}  V_ev(0)={V_ev[0]:6.2f}  "
                  f"|FV err|={fv_err:.2f}  |EV err|={ev_err:.2f}")

    print("\n  First-visit: каждое состояние учитывается 1 раз за эпизод →")
    print("  меньше variance, но potentially медленнее накопление данных.")
    print("  Every-visit: каждый визит учитывается → больше данных, но")
    print("  вложенность посещений увеличивает variance оценки.")


def demo3_mc_control():
    """Демо 3: MC control — обучение ε-greedy политики."""
    print("\n" + "=" * 60)
    print("  ДЕМО 3: MC Control (ε-greedy) — Обучение политики")
    print("=" * 60)

    env = GridWorld()
    Q, policy = mc_control(env, n_episodes=10000, epsilon=0.1)

    print_policy(policy, "Выученная политика (ε-greedy MC)")
    print("\n  Ценности Q(s,a) для каждого состояния:")
    for s in range(env.n_states):
        if s == env.TERMINAL:
            continue
        q_str = "  ".join(f"{ACTION_NAMES[a]}:{Q[s][a]:6.2f}" for a in range(4))
        print(f"    S{s:>2d}: {q_str}")

    # Проверка: насколько хорошо политика попадает в терминал
    state = env.reset()
    steps = 0
    while state != env.TERMINAL and steps < 20:
        state, _, _ = env.step(state, policy[state])
        steps += 1
    print(f"\n  Длина пути от S0 до терминала: {steps} шагов")


def demo4_mc_vs_td():
    """Демо 4: Сравнение MC и TD(0)."""
    print("\n" + "=" * 60)
    print("  ДЕМО 4: MC vs TD(0) Сравнение")
    print("=" * 60)

    env = GridWorld()
    def policy(state):
        return 2 if state < 12 else 1

    V_mc = mc_prediction_first_visit(env, policy, n_episodes=3000)
    V_td = td0_evaluation(env, policy, alpha=0.1, n_episodes=3000)

    print_grid(V_mc, "MC First-Visit V(s) [3000 ep]")
    print_grid(V_td, "TD(0) V(s) [3000 ep, α=0.1]")

    diff = sum(abs(V_mc[s] - V_td[s]) for s in range(env.n_states))
    print(f"\n  Суммарная |разница| MC vs TD: {diff:.4f}")

    print("\n  Ключевые различия MC vs TD:")
    print("  ┌───────────────────────┬──────────────────────────────────┐")
    print("  │ Свойство              │ MC vs TD                         │")
    print("  ├───────────────────────┼──────────────────────────────────┤")
    print("  │ Обучение              │ После эпизода vs пошагово        │")
    print("  │ Смещение (bias)       │ Нет (V^π) vs Да (≈ V^π)         │")
    print("  │ Дисперсия             │ Высокая vs Низкая                │")
    print("  │ Конечные эпизоды      │ Требует vs Не требует            │")
    print("  │ Онлайн обучение      │ Нет vs Да                        │")
    print("  │ Сходимость           │ Медленнее при γ=1 vs Быстрее     │")
    print("  └───────────────────────┴──────────────────────────────────┘")


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    demo1_mc_evaluation()
    demo2_first_vs_every()
    demo3_mc_control()
    demo4_mc_vs_td()
    print("\n" + "=" * 60)
    print("  Все демонстрации завершены.")
    print("=" * 60)
