"""
80 — Policy Gradient Methods (REINFORCE)
=========================================

Политические градиентные методы: алгоритм REINFORCE, политические сети,
baseline для снижения дисперсии. Всё на чистом Python.

Содержание:
  Демо 1 — Policy Network: forward pass, softmax, sampling
  Демо 2 — REINFORCE: сбор эпизодов и вычисление GAE
  Демо 3 — Baseline: вычитание среднего вознаграждения
  Демо 4 — Обучение Policy Gradient агента в табличном окружении
"""

import random
import math

random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────

def sigmoid(x):
    """Сигмоида."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def softmax(values):
    """Softmax по списку значений."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    s = sum(exps)
    return [e / s for e in exps]

def log_softmax(values):
    """Log-softmax по списку значений."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    s = sum(exps)
    return [math.log(e / s) for e in exps]

def sample_from_probs(probs):
    """Сэмплирование индекса из распределения вероятностей."""
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


# ──────────────────────────────────────────────────────────────────────
# Демо 1 — Policy Network: forward pass
# ──────────────────────────────────────────────────────────────────────

class PolicyNetwork:
    """
    Простая политическая сеть: state -> action logits -> softmax -> вероятности.

    Архитектура:
      input (state_dim) -> hidden (hidden_dim, ReLU) -> output (action_dim)
    """

    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.01):
        self.lr = lr
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Инициализация весов (Xavier-like)
        scale1 = math.sqrt(2.0 / (state_dim + hidden_dim))
        self.W1 = [[random.gauss(0, scale1) for _ in range(hidden_dim)]
                    for _ in range(state_dim)]
        self.b1 = [0.0] * hidden_dim

        scale2 = math.sqrt(2.0 / (hidden_dim + action_dim))
        self.W2 = [[random.gauss(0, scale2) for _ in range(action_dim)]
                    for _ in range(hidden_dim)]
        self.b2 = [0.0] * action_dim

    def forward(self, state):
        """Forward pass: state -> action probabilities."""
        # Скрытый слой
        hidden = []
        for j in range(self.hidden_dim):
            s = self.b1[j]
            for i in range(self.state_dim):
                s += state[i] * self.W1[i][j]
            hidden.append(max(0.0, s))  # ReLU

        # Выходной слой
        logits = []
        for k in range(self.action_dim):
            s = self.b2[k]
            for j in range(self.hidden_dim):
                s += hidden[j] * self.W2[j][k]
            logits.append(s)

        return softmax(logits), hidden, logits

    def get_action(self, state):
        """Выбрать действие по политике."""
        probs, _, _ = self.forward(state)
        action = sample_from_probs(probs)
        return action, probs

    def backward(self, state, hidden, logits, action, advantage):
        """
        Backward pass: вычисляем градиенты и обновляем веса.

        Градиент лог-вероятности:
          d log pi(a|s) / d logits = one_hot(a) - softmax(logits)
        Умножаем на advantage (G - baseline).
        """
        probs = softmax(logits)

        # Градиент по logits
        d_logits = [probs[k] - (1.0 if k == action else 0.0)
                    for k in range(self.action_dim)]

        # Градиент по W2, b2
        d_W2 = [[hidden[j] * d_logits[k] * advantage
                 for k in range(self.action_dim)]
                for j in range(self.hidden_dim)]
        d_b2 = [d_logits[k] * advantage for k in range(self.action_dim)]

        # Градиент по hidden
        d_hidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            for k in range(self.action_dim):
                d_hidden[j] += self.W2[j][k] * d_logits[k] * advantage

        # ReLU backward
        for j in range(self.hidden_dim):
            if hidden[j] <= 0:
                d_hidden[j] = 0.0

        # Градиент по W1, b1
        d_W1 = [[state[i] * d_hidden[j]
                 for j in range(self.hidden_dim)]
                for i in range(self.state_dim)]
        d_b1 = [d_hidden[j] for j in range(self.hidden_dim)]

        # Обновление весов (SGD)
        for i in range(self.state_dim):
            for j in range(self.hidden_dim):
                self.W1[i][j] -= self.lr * d_W1[i][j]
        for j in range(self.hidden_dim):
            self.b1[j] -= self.lr * d_b1[j]

        for j in range(self.hidden_dim):
            for k in range(self.action_dim):
                self.W2[j][k] -= self.lr * d_W2[j][k]
        for k in range(self.action_dim):
            self.b2[k] -= self.lr * d_b2[k]


# ══════════════════════════════════════════════════════════════════════
#  Демо 1 — Policy network: forward pass
# ══════════════════════════════════════════════════════════════════════

def demo1_policy_network():
    print("=" * 60)
    print("Демо 1: Policy Network — forward pass")
    print("=" * 60)

    net = PolicyNetwork(state_dim=4, hidden_dim=8, action_dim=2, lr=0.01)

    states = [
        [0.1, -0.5, 0.3, 0.8],
        [-1.0, 0.2, 0.0, -0.7],
        [0.5, 0.5, 0.5, 0.5],
    ]

    for idx, s in enumerate(states):
        probs, _, logits = net.forward(s)
        action, _ = net.get_action(s)
        print(f"\nСостояние {idx+1}: {s}")
        print(f"  Логиты:     {[round(l, 4) for l in logits]}")
        print(f"  Вероятности: {[round(p, 4) for p in probs]}")
        print(f"  Выбрано действие: {action}")

    # Проверка: сумма вероятностей = 1
    probs, _, _ = net.forward(states[0])
    print(f"\n  Сумма вероятностей: {sum(probs):.6f} (ожидается 1.0)")

    # Пример обучения: одно обновление
    print("\n--- Одно обновление (advantage = +1.0, action = 0) ---")
    probs_before, _, _ = net.forward(states[0])
    print(f"  Вероятности ДО:  {[round(p, 4) for p in probs_before]}")

    hidden, logits = net.forward(states[0])[1:]
    net.backward(states[0], hidden, logits, 0, advantage=1.0)

    probs_after, _, _ = net.forward(states[0])
    print(f"  Вероятности ПОСЛЕ: {[round(p, 4) for p in probs_after]}")
    print(f"  Действие 0 стало вероятнее: {probs_after[0] > probs_before[0]}")


# ──────────────────────────────────────────────────────────────────────
# Демо 2 — REINFORCE: сбор эпизодов
# ──────────────────────────────────────────────────────────────────────

class SimpleGridEnv:
    """
    Простое сеточное окружение 4x4.
    Агент移动 по клеткам, цель — дойти до правого нижнего угла.
    Состояние: [x, y] нормализовано в [0, 1].
    Действия: 0=вверх, 1=вправо, 2=вниз, 3=влево.
    Награда: -0.04 за каждый шаг, +1.0 за достижение цели.
    Макс. 100 шагов.
    """

    def __init__(self, size=4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self._obs()

    def _obs(self):
        return [self.pos[0] / (self.size - 1),
                self.pos[1] / (self.size - 1)]

    def step(self, action):
        x, y = self.pos
        if action == 0:   # вверх
            x = max(0, x - 1)
        elif action == 1:  # вправо
            y = min(self.size - 1, y + 1)
        elif action == 2:  # вниз
            x = min(self.size - 1, x + 1)
        elif action == 3:  # влево
            y = max(0, y - 1)

        self.pos = (x, y)
        self.steps += 1

        done = (self.pos == self.goal) or (self.steps >= 100)
        reward = 1.0 if self.pos == self.goal else -0.04
        return self._obs(), reward, done


def collect_episode(env, policy):
    """
    Собрать один эпизод: список (state, action, reward).
    """
    trajectory = []
    state = env.reset()
    done = False

    while not done:
        action, probs = policy.get_action(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state

    return trajectory


def compute_returns(trajectory, gamma=1.0):
    """
    Вычислить G_t = sum_{k=t}^{T-1} gamma^(k-t) * r_k для каждого шага.
    """
    T = len(trajectory)
    returns = [0.0] * T
    G = 0.0
    for t in reversed(range(T)):
        G = trajectory[t][2] + gamma * G
        returns[t] = G
    return returns


def demo2_reinforce():
    print("\n" + "=" * 60)
    print("Демо 2: REINFORCE — сбор эпизодов и вычисление G")
    print("=" * 60)

    env = SimpleGridEnv(size=4)
    net = PolicyNetwork(state_dim=2, hidden_dim=8, action_dim=4, lr=0.01)

    # Соберём несколько эпизодов
    trajectories = []
    for ep in range(5):
        traj = collect_episode(env, net)
        returns = compute_returns(traj, gamma=0.99)
        trajectories.append((traj, returns))

        total_r = sum(r for _, _, r in traj)
        print(f"\nЭпизод {ep+1}: {len(traj)} шагов, суммарная награда = {total_r:.3f}")
        print(f"  Первые 5 шагов:")
        for t in range(min(5, len(traj))):
            s, a, r = traj[t]
            print(f"    t={t}: state={[round(x,2) for x in s]}, "
                  f"action={a}, reward={r:.2f}, G={returns[t]:.3f}")

    # Статистика
    test_trajs = [collect_episode(env, net) for _ in range(2)]
    all_returns = [sum(r for _, _, r in traj) for traj in test_trajs]
    print(f"\nСредняя награда за 2 новых эпизода: {sum(all_returns)/len(all_returns):.3f}")


# ──────────────────────────────────────────────────────────────────────
# Демо 3 — Baseline: вычитание среднего вознаграждения
# ──────────────────────────────────────────────────────────────────────

def demo3_baseline():
    print("\n" + "=" * 60)
    print("Демо 3: Baseline — вычитание среднего вознаграждения")
    print("=" * 60)

    # Демонстрация на массиве наград
    rewards_episode1 = [0.1, 0.2, -0.3, 0.5, 1.0]
    rewards_episode2 = [-0.1, -0.2, 0.3, -0.1, 0.2]

    returns1 = compute_returns(
        [(None, None, r) for r in rewards_episode1], gamma=1.0)
    returns2 = compute_returns(
        [(None, None, r) for r in rewards_episode2], gamma=1.0)

    print("\nЭпизод 1:")
    print(f"  Награды:     {rewards_episode1}")
    print(f"  Returns G_t: {[round(g, 3) for g in returns1]}")

    print("\nЭпизод 2:")
    print(f"  Награды:     {rewards_episode2}")
    print(f"  Returns G_t: {[round(g, 3) for g in returns2]}")

    # Без baseline
    print("\n--- Без baseline (advantage = G_t) ---")
    print(f"  Эп.1 advantage: {[round(g, 3) for g in returns1]}")
    print(f"  Эп.2 advantage: {[round(g, 3) for g in returns2]}")

    # С baseline (среднее G)
    mean_return1 = sum(returns1) / len(returns1)
    mean_return2 = sum(returns2) / len(returns2)

    adv1 = [g - mean_return1 for g in returns1]
    adv2 = [g - mean_return2 for g in returns2]

    print(f"\n--- С baseline (advantage = G_t - mean(G)) ---")
    print(f"  Базeline эп.1: {mean_return1:.3f}")
    print(f"  Advantage эп.1: {[round(a, 3) for a in adv1]}")
    print(f"  Базeline эп.2: {mean_return2:.3f}")
    print(f"  Advantage эп.2: {[round(a, 3) for a in adv2]}")

    # Сравнение дисперсии
    def variance(vals):
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / len(vals)

    var_no = variance(returns1 + returns2)
    var_yes = variance(adv1 + adv2)
    print(f"\nДисперсия advantage без baseline: {var_no:.4f}")
    print(f"Дисперсия advantage с baseline:   {var_yes:.4f}")
    print(f"Снижение дисперсии: {(1 - var_yes/var_no)*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# Демо 4 — Обучение Policy Gradient агента
# ──────────────────────────────────────────────────────────────────────

class REINFORCEAgent:
    """
    REINFORCE агент с baseline.
    """

    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.005, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim, lr)
        self.gamma = gamma
        self.episode_returns = []  # для running baseline

    def select_action(self, state):
        return self.policy.get_action(state)

    def update(self, trajectory):
        """Обновить политику по REINFORCE с baseline."""
        # Вычислить returns
        T = len(trajectory)
        returns = compute_returns(trajectory, self.gamma)

        # Running baseline: среднее по всем предыдущим G_0
        G0 = returns[0]
        self.episode_returns.append(G0)
        if len(self.episode_returns) > 100:
            self.episode_returns.pop(0)
        baseline = sum(self.episode_returns) / len(self.episode_returns)

        # Advantage = G_t - baseline
        advantages = [g - baseline for g in returns]

        # Обновление весов
        for t in range(T):
            state, action, _ = trajectory[t]
            probs, hidden, logits = self.policy.forward(state)
            self.policy.backward(state, hidden, logits, action, advantages[t])

        return sum(r for _, _, r in trajectory), baseline


def demo4_training():
    print("\n" + "=" * 60)
    print("Демо 4: Обучение Policy Gradient агента")
    print("=" * 60)

    env = SimpleGridEnv(size=4)
    agent = REINFORCEAgent(
        state_dim=2, hidden_dim=16, action_dim=4, lr=0.005, gamma=0.95
    )

    # Тренировка
    n_episodes = 500
    rewards_history = []

    for ep in range(n_episodes):
        traj = collect_episode(env, agent.policy)
        total_reward, baseline = agent.update(traj)
        rewards_history.append(total_reward)

        if (ep + 1) % 100 == 0:
            recent = rewards_history[-100:]
            avg = sum(recent) / len(recent)
            print(f"  Эпизод {ep+1:4d} | "
                  f"Ср. награда (100): {avg:7.3f} | "
                  f"Длина эпизода: {len(traj):3d} | "
                  f"Baseline: {baseline:.3f}")

    # Тестирование обученного агента
    print("\n--- Тестирование обученного агента (10 эпизодов) ---")
    test_rewards = []
    for ep in range(10):
        traj = collect_episode(env, agent.policy)
        total_r = sum(r for _, _, r in traj)
        test_rewards.append(total_r)
        print(f"  Тест {ep+1}: награда = {total_r:.3f}, шагов = {len(traj)}")

    print(f"\n  Средняя тестовая награда: {sum(test_rewards)/len(test_rewards):.3f}")

    # Визуализация политики
    print("\n--- Выученная политика (4x4 сетка) ---")
    action_symbols = ["↑", "→", "↓", "←"]
    for x in range(4):
        row = ""
        for y in range(4):
            if x == 3 and y == 3:
                row += " G "
            else:
                obs = [x / 3, y / 3]
                probs, _, _ = agent.policy.forward(obs)
                best = probs.index(max(probs))
                row += f" {action_symbols[best]} "
        print(f"  {row}")


# ══════════════════════════════════════════════════════════════════════
#  Запуск всех демо
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo1_policy_network()
    demo2_reinforce()
    demo3_baseline()
    demo4_training()

    print("\n" + "=" * 60)
    print("Все демонстрации Policy Gradient завершены!")
    print("=" * 60)
