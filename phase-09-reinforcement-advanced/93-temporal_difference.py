"""
Temporal Difference (TD) Learning — основы.
=============================================
TD объединяет идеи MC (обучение по опыта) и DP (обучение по модели):
  - Как MC: работает с реальными траекториями (без знания модели)
  - Как DP: обновляет оценки на основе других оценок (bootstrapping)

Ключевые методы:
  1. TD(0)          — одношаговое обновление V(s)
  2. TD(λ)          — eligibility traces для ускорения распространения оценки
  3. SARSA           — предсказание Q(s,a) по.policy
  4. Сравнение TD vs MC по скорости сходимости

Самодостаточный файл: только random + collections.
"""

import random
from collections import defaultdict
from typing import List, Tuple, Dict

random.seed(42)

# ============================================================================
# Утилиты: простой MDP — случайная сетка 4x4 с фиксированными наградами
# ============================================================================

class GridWorld:
    """
    4x4 сетка.
    Состояния: 0..15 (row*4+col).
    Терминальные: 15 (win, +10) и 7 (pit, -10).
    Действия: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
    Стоимость шага: -1 (штраф за каждый ход).
    """

    ACTIONS = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT
    ROWS, COLS = 4, 4
    TERMINAL_WIN = 15
    TERMINAL_PIT = 7
    STEP_REWARD = -1.0
    WIN_REWARD = 10.0
    PIT_REWARD = -10.0

    def __init__(self):
        self.n_states = self.ROWS * self.COLS
        self.n_actions = len(self.ACTIONS)
        self.terminal = {self.TERMINAL_WIN, self.TERMINAL_PIT}

    def reset(self) -> int:
        """Начальное состояние — случайная клетка (не терминальная)."""
        while True:
            s = random.randint(0, self.n_states - 1)
            if s not in self.terminal:
                return s

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Выполнить действие. Возвращает (next_state, reward, done)."""
        if state in self.terminal:
            return state, 0.0, True

        row, col = divmod(state, self.COLS)

        # Движение с вероятностью детерминированной
        if action == 0:    # UP
            row = max(0, row - 1)
        elif action == 1:  # RIGHT
            col = min(self.COLS - 1, col + 1)
        elif action == 2:  # DOWN
            row = min(self.ROWS - 1, row + 1)
        elif action == 3:  # LEFT
            col = max(0, col - 1)

        next_state = row * self.COLS + col

        if next_state == self.TERMINAL_WIN:
            return next_state, self.WIN_REWARD, True
        elif next_state == self.TERMINAL_PIT:
            return next_state, self.PIT_REWARD, True
        else:
            return next_state, self.STEP_REWARD, False

    def get_reward(self, state: int) -> float:
        if state == self.TERMINAL_WIN:
            return self.WIN_REWARD
        elif state == self.TERMINAL_PIT:
            return self.PIT_REWARD
        return self.STEP_REWARD


def generate_episode(env: GridWorld, policy: dict = None, max_steps: int = 100
                     ) -> List[Tuple[int, int, float]]:
    """Генерация одного эпизода: [(state, action, reward), ...]."""
    episode = []
    state = env.reset()
    for _ in range(max_steps):
        if state in env.terminal:
            break
        if policy is not None:
            action = policy[state]
        else:
            action = random.choice(env.ACTIONS)
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


# ============================================================================
# 1. TD(0) — одношаговое обновление ценности состояний
# ============================================================================

def td_zero(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
            gamma: float = 0.99) -> Dict[int, float]:
    """
    TD(0) предсказание V(s).

    Обновление: V(s) <- V(s) + α [R + γV(s') - V(s)]
    """
    V = defaultdict(float)

    for ep in range(n_episodes):
        state = env.reset()
        while state not in env.terminal:
            action = random.choice(env.ACTIONS)
            next_state, reward, done = env.step(state, action)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]

            V[state] += alpha * (td_target - V[state])
            state = next_state

    return dict(V)


# ============================================================================
# 2. TD(λ) — eligibility traces
# ============================================================================

def td_lambda(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
              gamma: float = 0.99, lam: float = 0.5) -> Dict[int, float]:
    """
    TD(λ) с eligibility traces.

    Trace:  e(s) <- γλ·e(s) + 1   (при посещении s)
    Update: V(s) <- V(s) + α·δ·e(s) для ВСЕХ s
    δ = R + γV(s') - V(s)  (TD error)
    """
    V = defaultdict(float)

    for ep in range(n_episodes):
        e = defaultdict(float)  # traces
        state = env.reset()

        while state not in env.terminal:
            action = random.choice(env.ACTIONS)
            next_state, reward, done = env.step(state, action)

            if done:
                td_error = reward - V[state]
            else:
                td_error = reward + gamma * V[next_state] - V[state]

            e[state] += 1.0  # обновить trace текущего состояния

            # Обновить ВСЕ оценки
            for s in list(e.keys()):
                V[s] += alpha * td_error * e[s]
                e[s] *= gamma * lam  # затухание

            state = next_state

    return dict(V)


# ============================================================================
# 3. SARSA prediction — Q(s,a)
# ============================================================================

def sarsa_prediction(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
                     gamma: float = 0.99) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    SARSA: Q(s,a) <- Q(s,a) + α [R + γQ(s',a') - Q(s,a)]
    Использует ε-greedy политику для сбора данных.
    """
    Q = defaultdict(lambda: [0.0] * env.n_actions)
    eps = 0.2

    for ep in range(n_episodes):
        state = env.reset()
        # Выбрать действие по ε-greedy
        if random.random() < eps:
            action = random.choice(env.ACTIONS)
        else:
            action = max(range(env.n_actions), key=lambda a: Q[state][a])

        while state not in env.terminal:
            next_state, reward, done = env.step(state, action)

            if done:
                td_target = reward
            else:
                # Выбрать следующее действие по ε-greedy
                if random.random() < eps:
                    next_action = random.choice(env.ACTIONS)
                else:
                    next_action = max(range(env.n_actions),
                                      key=lambda a: Q[next_state][a])
                td_target = reward + gamma * Q[next_state][next_action]

            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            if not done:
                action = next_action

    # Извлечь политику и оптимальные ценности
    policy = {}
    V = {}
    for s in range(env.n_states):
        best_a = max(range(env.n_actions), key=lambda a: Q[s][a])
        policy[s] = best_a
        V[s] = max(Q[s])

    return dict(V), policy


# ============================================================================
# 4. Monte Carlo для сравнения с TD
# ============================================================================

def monte_carlo_prediction(env: GridWorld, n_episodes: int = 500,
                           gamma: float = 0.99) -> Dict[int, float]:
    """
    MC first-visit предсказание V(s).
    V(s) <- V(s) + α [G_t - V(s)]
    """
    V = defaultdict(float)
    counts = defaultdict(int)

    for ep in range(n_episodes):
        episode = generate_episode(env)
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G

            if state not in visited:
                visited.add(state)
                counts[state] += 1
                V[state] += (G - V[state]) / counts[state]

    return dict(V)


# ============================================================================
# Демо 1: TD(0) — оценка ценности
# ============================================================================

def demo_1_td_zero():
    print("=" * 70)
    print("ДЕМО 1: TD(0) — оценка ценности состояний")
    print("=" * 70)

    env = GridWorld()
    V = td_zero(env, n_episodes=1000, alpha=0.1, gamma=0.99)

    print("\nV(s) после TD(0) (1000 эпизодов):")
    print("-" * 40)
    for row in range(env.ROWS):
        row_vals = []
        for col in range(env.COLS):
            s = row * env.COLS + col
            if s == env.TERMINAL_WIN:
                row_vals.append(" WIN ")
            elif s == env.TERMINAL_PIT:
                row_vals.append(" PIT ")
            else:
                row_vals.append(f"{V.get(s, 0.0):+6.2f}")
        print("  ".join(row_vals))

    print(f"\nV(WIN) = {V.get(env.TERMINAL_WIN, 0):.2f} (ожидается: +10)")
    print(f"V(PIT) = {V.get(env.TERMINAL_PIT, 0):.2f} (ожидается: -10)")
    print()


# ============================================================================
# Демо 2: TD(λ) — eligibility traces
# ============================================================================

def demo_2_td_lambda():
    print("=" * 70)
    print("ДЕМО 2: TD(λ) — eligibility traces")
    print("=" * 70)

    env = GridWorld()

    # Сравнение разных значений λ
    lambdas = [0.0, 0.3, 0.5, 0.8, 1.0]
    results = {}

    for lam in lambdas:
        random.seed(42)
        V = td_lambda(env, n_episodes=1000, alpha=0.1, gamma=0.99, lam=lam)
        results[lam] = V

    print("\nСравнение TD(λ) для разных λ (V для ключевых состояний):")
    print("-" * 60)
    print(f"{'λ':>5}  {'V(1)':>8}  {'V(5)':>8}  {'V(10)':>8}  {'V(14)':>8}")
    print("-" * 60)

    for lam in lambdas:
        V = results[lam]
        print(f"{lam:5.1f}  {V.get(1, 0):+8.2f}  {V.get(5, 0):+8.2f}  "
              f"{V.get(10, 0):+8.2f}  {V.get(14, 0):+8.2f}")

    print("\nВывод:")
    print("  λ=0.0: чистый TD(0) — быстрое обучение, но смещение")
    print("  λ=1.0: чистый MC — медленное, но несмещённое оценка")
    print("  λ=0.5: компромисс — баланс скорости и точности")
    print()


# ============================================================================
# Демо 3: SARSA prediction
# ============================================================================

def demo_3_sarsa():
    print("=" * 70)
    print("ДЕМО 3: SARSA prediction — Q(s,a)")
    print("=" * 70)

    env = GridWorld()
    V, policy = sarsa_prediction(env, n_episodes=1000, alpha=0.1, gamma=0.99)

    action_symbols = ["↑", "→", "↓", "←"]

    print("\nОптимальная политика (SARSA):")
    print("-" * 30)
    for row in range(env.ROWS):
        cells = []
        for col in range(env.COLS):
            s = row * env.COLS + col
            if s == env.TERMINAL_WIN:
                cells.append(" WIN ")
            elif s == env.TERMINAL_PIT:
                cells.append(" PIT ")
            else:
                cells.append(f"  {action_symbols[policy[s]]}  ")
        print("  ".join(cells))

    print("\nV(s) из SARSA:")
    print("-" * 30)
    for row in range(env.ROWS):
        row_vals = []
        for col in range(env.COLS):
            s = row * env.COLS + col
            if s in (env.TERMINAL_WIN, env.TERMINAL_PIT):
                row_vals.append(" --- ")
            else:
                row_vals.append(f"{V.get(s, 0.0):+6.2f}")
        print("  ".join(row_vals))
    print()


# ============================================================================
# Демо 4: TD vs MC — скорость сходимости
# ============================================================================

def demo_4_td_vs_mc():
    print("=" * 70)
    print("ДЕМО 4: TD vs MC — сравнение скорости сходимости")
    print("=" * 70)

    env = GridWorld()

    # Настоящие значения (из TD(0) с большим числом эпизодов)
    random.seed(42)
    V_true = td_zero(env, n_episodes=50000, alpha=0.01, gamma=0.99)

    # Сравнение сходимости
    episode_counts = [50, 100, 200, 500, 1000, 2000]

    print("\nСредняя абсолютная ошибка V(s) от эталона:")
    print("-" * 65)
    print(f"{'Эпизодов':>10}  {'TD(0)':>12}  {'TD(λ=0.5)':>12}  {'MC':>12}")
    print("-" * 65)

    for n_ep in episode_counts:
        # TD(0)
        random.seed(42)
        V_td0 = td_zero(env, n_episodes=n_ep, alpha=0.1, gamma=0.99)
        err_td0 = _mean_abs_error(V_td0, V_true, env)

        # TD(λ)
        random.seed(42)
        V_tdl = td_lambda(env, n_episodes=n_ep, alpha=0.1, gamma=0.99, lam=0.5)
        err_tdl = _mean_abs_error(V_tdl, V_true, env)

        # MC
        random.seed(42)
        V_mc = monte_carlo_prediction(env, n_episodes=n_ep, gamma=0.99)
        err_mc = _mean_abs_error(V_mc, V_true, env)

        print(f"{n_ep:>10}  {err_td0:>12.3f}  {err_tdl:>12.3f}  {err_mc:>12.3f}")

    print("\nВывод:")
    print("  - TD(0) обычно сходится быстрее MC (bootstrap уменьшает дисперсию)")
    print("  - TD(λ) балансирует: быстрее MC, точнее TD(0)")
    print("  - MC имеет большую дисперсию из-за полных траекторий")
    print("  - TD лучше для сред с большими/бесконечными траекториями")
    print()


def _mean_abs_error(V_est: dict, V_true: dict, env: GridWorld) -> float:
    """Средняя абсолютная ошибка по не-терминальным состояниям."""
    errors = []
    for s in range(env.n_states):
        if s not in env.terminal:
            errors.append(abs(V_est.get(s, 0.0) - V_true.get(s, 0.0)))
    return sum(errors) / len(errors) if errors else 0.0


# ============================================================================
# Запуск всех демо
# ============================================================================

if __name__ == "__main__":
    random.seed(42)

    demo_1_td_zero()
    demo_2_td_lambda()
    demo_3_sarsa()
    demo_4_td_vs_mc()

    print("=" * 70)
    print("КРАТКАЯ СВОДКА: TD Learning")
    print("=" * 70)
    print("""
TD(0):
  V(s) <- V(s) + α [R + γV(s') - V(s)]
  Bootstrapping: оценка на основе другой оценки.

TD(λ):
  e(s) <- γλ·e(s) + 1  (при посещении)
  V(s) <- V(s) + α·δ·e(s)
  λ управляет trade-off: bias (λ→0) vs variance (λ→1).

SARSA:
  Q(s,a) <- Q(s,a) + α [R + γQ(s',a') - Q(s,a)]
  On-policy: учитывает текущую политику.

TD vs MC:
  TD:  быстрее, меньше дисперсия, bias от bootstrapping
  MC:  медленнее, больше дисперсия, несмещённый
""")
