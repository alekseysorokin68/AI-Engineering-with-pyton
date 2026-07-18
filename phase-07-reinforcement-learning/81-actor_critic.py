"""
81. Actor-Critic Methods
========================
Actor-Critic = Policy (Actor) + Value Function (Critic) в одном агенте.

Ключевые идеи:
- Actor: управляет поведением (политика π(a|s))
- Critic: оценивает ценность состояний/действий (V(s) или Q(s,a))
- Advantage = Q(s,a) - V(s) — насколько действие лучше среднего
- A2C: Advantage Actor-Critic — один из самых стабильных методов

Сравнение:
- REINFORCE: высокая дисперсия, не использует критик
- A2C: низкая дисперсия, использует advantage для стабилизации
"""

import random
import math

random.seed(42)


# ============================================================
# 1. Математические утилиты
# ============================================================

def sigmoid(x):
    """Сигмоида для выхода критика."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def softmax(values):
    """Softmax для распределения действий актора."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def normal_pdf(x, mu, sigma):
    """Плотность нормального распределения."""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exp_part = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coeff * exp_part


# ============================================================
# 2. Простая нейросеть (MLP) — реализация с нуля
# ============================================================

class SimpleMLP:
    """Простая многослойная нейросеть без внешних зависимостей."""

    _counter = 0  # Автоинкремент для уникальных seed по умолчанию

    def __init__(self, layer_sizes, seed=None):
        """
        Args:
            layer_sizes: [input, hidden1, ..., output]
            seed: seed для генерации весов (None = автоинкремент)
        """
        if seed is None:
            seed = SimpleMLP._counter
            SimpleMLP._counter += 1
        self.rng = random.Random(seed)
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            # Xavier инициализация
            limit = math.sqrt(6.0 / (in_size + out_size))
            weights = [[self.rng.uniform(-limit, limit) for _ in range(out_size)]
                       for _ in range(in_size)]
            bias = [0.0] * out_size
            self.layers.append({'weights': weights, 'bias': bias})

    def forward(self, x):
        """Forward pass, возвращает (output, caches) для backward."""
        caches = []
        current = x

        for layer in self.layers:
            # Linear: y = Wx + b
            z = []
            for j in range(len(layer['bias'])):
                s = layer['bias'][j]
                for i, xi in enumerate(current):
                    s += xi * layer['weights'][i][j]
                z.append(s)

            # ReLU для скрытых слоёв, identity для выхода
            if layer is not self.layers[-1]:
                a = [max(0, v) for v in z]  # ReLU (скрытые слои)
            else:
                a = list(z)  # Identity (выходной слой)
            caches.append({'input': current, 'z': z, 'a': a})
            current = a

        return current, caches

    def backward(self, caches, output_grad, lr=0.01, grad_clip=1.0):
        """Backward pass с обновлением весов и клиппингом градиентов."""

        def clip(v, c):
            return max(-c, min(c, v))

        grads = [None] * len(self.layers)
        dA = output_grad

        for l in range(len(self.layers) - 1, -1, -1):
            cache = caches[l]
            z = cache['z']
            a_prev = cache['input']
            layer = self.layers[l]

            # ReLU производная (для скрытых слоёв)
            if l < len(self.layers) - 1:
                dZ = [dA[i] * (1.0 if z[i] > 0 else 0.0)
                      for i in range(len(z))]
            else:
                dZ = dA

            # Вычисляем градиенты весов
            dW = []
            for i in range(len(a_prev)):
                row = []
                for j in range(len(dZ)):
                    row.append(a_prev[i] * dZ[j])
                dW.append(row)

            dB = list(dZ)

            # Клиппинг градиентов
            dW = [[clip(v, grad_clip) for v in row] for row in dW]
            dB = [clip(v, grad_clip) for v in dB]

            # Обновляем веса
            for i in range(len(layer['weights'])):
                for j in range(len(layer['weights'][i])):
                    layer['weights'][i][j] -= lr * dW[i][j]
            for j in range(len(layer['bias'])):
                layer['bias'][j] -= lr * dB[j]

            # Передаём градиент назад
            if l > 0:
                dA = []
                for i in range(len(a_prev)):
                    s = 0
                    for j in range(len(dZ)):
                        s += layer['weights'][i][j] * dZ[j]
                    dA.append(s)


# ============================================================
# 3. Actor (Политика) — дискретное пространство действий
# ============================================================

class Actor:
    """
    Actor: нейросеть, выдающая распределение вероятностей действий.
    π(a|s) = softmax(θ(s))
    """

    def __init__(self, state_dim, action_dim, hidden_dim=32, lr=0.001):
        self.action_dim = action_dim
        self.lr = lr
        self.network = SimpleMLP([state_dim, hidden_dim, action_dim])

    def get_action(self, state):
        """Выбрать действие по распределению вероятностей."""
        logits, self.last_caches = self.network.forward(state)
        probs = softmax(logits)
        # Сэмплирование
        r = random.random()
        cumulative = 0.0
        for a, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return a, probs
        return action_dim - 1, probs

    def update(self, advantage, action, log_prob=None):
        """
        Обновление политики: ∇J ≈ advantage * ∇log π(a|s)
        Для дискретного случая: градиент = (probs_one_hot - probs) * advantage
        """
        probs = self.last_probs
        d_logits = list(probs)

        # Вычитаем 1 у выбранного действия
        d_logits[action] -= 1.0

        # Умножаем на advantage (с минусом, т.к. мы минимизируем)
        output_grad = [d_logits[i] * advantage for i in range(len(d_logits))]

        self.network.backward(self.last_caches, output_grad, self.lr)


class ActorDiscrete:
    """Обёртка для дискретного актора с хранением лог-вероятностей."""

    def __init__(self, state_dim, action_dim, hidden_dim=32, lr=0.001):
        self.action_dim = action_dim
        self.lr = lr
        self.network = SimpleMLP([state_dim, hidden_dim, action_dim])

    def get_action(self, state):
        logits, caches = self.network.forward(state)
        probs = softmax(logits)

        # Сэмплирование
        r = random.random()
        cumulative = 0.0
        chosen = self.action_dim - 1
        for a, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                chosen = a
                break

        self.last_caches = caches
        self.last_probs = probs
        self.last_action = chosen

        return chosen, probs

    def update(self, advantage):
        probs = self.last_probs
        action = self.last_action

        # Градиент политики: (one_hot(action) - probs) * advantage
        d_logits = list(probs)
        d_logits[action] -= 1.0

        output_grad = [d_logits[i] * advantage for i in range(len(d_logits))]

        self.network.backward(self.last_caches, output_grad, self.lr)

        return max(probs)


# ============================================================
# 4. Critic (Функция ценности)
# ============================================================

class Critic:
    """
    Critic: нейросеть, оценивающая ценность состояния V(s).
    Обучается через TD-ошибку: L = (V(s) - V_target)^2
    """

    def __init__(self, state_dim, hidden_dim=32, lr=0.001):
        self.lr = lr
        self.network = SimpleMLP([state_dim, hidden_dim, 1])

    def get_value(self, state):
        """Оценить V(s)."""
        output, caches = self.network.forward(state)
        self.last_caches = caches
        return output[0]

    def update(self, td_error, clip=1.0):
        """
        Обновление критика: минимизируем (V(s) - target)^2
        Градиент: 2 * (V(s) - target) ≈ 2 * td_error
        """
        td_error = max(-clip, min(clip, td_error))
        output_grad = [2.0 * td_error]
        self.network.backward(self.last_caches, output_grad, self.lr)


# ============================================================
# 5. Симулирование среды (Simple Grid)
# ============================================================

class SimpleGridEnv:
    """
    Простая сетка 4x4. Агент движется к цели, получая награды.
    Дискретные действия: 0=вверх, 1=вниз, 2=влево, 3=вправо.
    """

    def __init__(self, size=4, seed=42):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """Возвращает нормализованное состояние."""
        return [
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size,
            self.goal_pos[0] / self.size,
            self.goal_pos[1] / self.size,
        ]

    def step(self, action):
        """Выполнить действие, вернуть (new_state, reward, done)."""
        self.steps += 1

        # Движение
        if action == 0:   # вверх
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # вниз
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # влево
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # вправо
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        # Проверка цели
        if self.agent_pos == self.goal_pos:
            return self._get_state(), 1.0, True

        # Награда: расстояние до цели
        dist = abs(self.agent_pos[0] - self.goal_pos[0]) + \
               abs(self.agent_pos[1] - self.goal_pos[1])
        reward = -0.1 * dist / (2 * self.size)

        # Штраф за слишком длинные эпизоды
        if self.steps >= 100:
            return self._get_state(), -0.5, True

        return self._get_state(), reward, False


# ============================================================
# 6. REINFORCE (базовый метод)
# ============================================================

class REINFORCE:
    """
    REINFORCE:olicy gradient без критика.
    ∇J = E[G_t * ∇log π(a_t|s_t)]
    """

    def __init__(self, state_dim, action_dim, lr=0.005):
        self.actor = ActorDiscrete(state_dim, action_dim, lr=lr)

    def select_action(self, state):
        return self.actor.get_action(state)

    def train_episode(self, env, gamma=0.99):
        """Обучение на одном эпизоде."""
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False

        while not done:
            action, _ = self.actor.get_action(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Вычисляем G_t (returns)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Нормализуем returns (baseline)
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5 + 1e-8

        # Обновляем политику
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            advantage = (returns[t] - mean_r) / std_r

            # Выполняем forward для получения кэшей
            logits, caches = self.actor.network.forward(state)
            probs = softmax(logits)

            # Обратное распространение
            d_logits = list(probs)
            d_logits[action] -= 1.0
            output_grad = [d_logits[i] * advantage for i in range(len(d_logits))]
            self.actor.network.backward(caches, output_grad, self.actor.lr)

        return sum(rewards)


# ============================================================
# 7. A2C (Advantage Actor-Critic)
# ============================================================

class A2C:
    """
    Advantage Actor-Critic:
    - Actor: π(a|s), обновляется через advantage
    - Critic: V(s), обновляется через TD-ошибку
    - Advantage = r + γV(s') - V(s) = TD-ошибка
    """

    def __init__(self, state_dim, action_dim, hidden_dim=32,
                 actor_lr=0.001, critic_lr=0.001):
        self.actor = ActorDiscrete(state_dim, action_dim, hidden_dim, actor_lr)
        self.critic = Critic(state_dim, hidden_dim, critic_lr)

    def select_action(self, state):
        return self.actor.get_action(state)

    def train_episode(self, env, gamma=0.99):
        """Обучение на одном эпизоде с advantage."""
        total_reward = 0
        state = env.reset()
        done = False

        prev_value = 0.0

        while not done:
            # Actor: выбрать действие
            action, probs = self.actor.get_action(state)

            # Critic: оценить текущее состояние
            value = self.critic.get_value(state)

            # Выполнить действие
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Critic: оценить следующее состояние
            if done:
                next_value = 0.0
            else:
                next_value = self.critic.get_value(next_state)

            # Advantage = r + γV(s') - V(s)
            advantage = reward + gamma * next_value - value

            # Обновить критик (TD-ошибка)
            td_error = advantage
            self.critic.update(td_error)

            # Обновить актор (через advantage)
            self.actor.update(advantage)

            prev_value = value
            state = next_state

        return total_reward


# ============================================================
# 8. Демонстрации
# ============================================================

def demo_1_actor_critic_architectures():
    """Демо 1: Архитектуры Actor и Critic сетей."""
    print("=" * 60)
    print("ДЕМО 1: Архитектуры Actor и Critic")
    print("=" * 60)

    state_dim = 4
    action_dim = 4

    # Создаём Actor
    actor = ActorDiscrete(state_dim, action_dim, hidden_dim=16)
    # Создаём Critic
    critic = Critic(state_dim, hidden_dim=16)

    print(f"\n--- Actor (Политика) ---")
    print(f"  Вход: {state_dim} (состояние)")
    print(f"  Скрытые слои: 16 нейронов (ReLU)")
    print(f"  Выход: {action_dim} (logits для softmax → π(a|s))")
    print(f"  Параметры: {sum(len(l['weights'][0]) * len(l['weights']) + len(l['bias']) for l in actor.network.layers)}")

    print(f"\n--- Critic (Функция ценности) ---")
    print(f"  Вход: {state_dim} (состояние)")
    print(f"  Скрытые слои: 16 нейронов (ReLU)")
    print(f"  Выход: 1 (V(s) — ценность состояния)")
    print(f"  Параметры: {sum(len(l['weights'][0]) * len(l['weights']) + len(l['bias']) for l in critic.network.layers)}")

    # Forward pass демонстрация
    test_state = [0.0, 0.0, 1.0, 1.0]
    logits, _ = actor.network.forward(test_state)
    probs = softmax(logits)
    value = critic.get_value(test_state)

    print(f"\n--- Пример forward pass ---")
    print(f"  Состояние: {test_state}")
    print(f"  Actor logits: {[f'{l:.3f}' for l in logits]}")
    print(f"  Actor probs: {[f'{p:.3f}' for p in probs]}")
    print(f"  Critic V(s): {value:.4f}")

    # Два агента с разными начальными весами
    actor1 = ActorDiscrete(state_dim, action_dim)
    actor2 = ActorDiscrete(state_dim, action_dim)

    _, probs1 = actor1.get_action(test_state)
    _, probs2 = actor2.get_action(test_state)

    print(f"\n--- Инициализация случайных весов ---")
    print(f"  Agent 1 probs: {[f'{p:.3f}' for p in probs1]}")
    print(f"  Agent 2 probs: {[f'{p:.3f}' for p in probs2]}")
    print(f"  → Разные веса = разные политики\n")


def demo_2_advantage_computation():
    """Демо 2: Вычисление advantage."""
    print("=" * 60)
    print("ДЕМО 2: Advantage Computation")
    print("=" * 60)

    gamma = 0.99

    print(f"\n--- TD-ошибка = Advantage ---")
    print(f"  Advantage(s,a) = r + γ·V(s') - V(s)")
    print(f"  Advantage > 0 → действие лучше среднего")
    print(f"  Advantage < 0 → действие хуже среднего")
    print(f"  Advantage = 0 → действие среднее\n")

    # Примеры вычисления advantage
    scenarios = [
        {"V_s": 0.5, "reward": 1.0, "V_s_next": 0.8, "desc": "Хорошее действие"},
        {"V_s": 0.5, "reward": 0.1, "V_s_next": 0.4, "desc": "Плохое действие"},
        {"V_s": 0.5, "reward": 0.5, "V_s_next": 0.5, "desc": "Нейтральное действие"},
        {"V_s": 0.8, "reward": 0.0, "V_s_next": 0.6, "desc": "Состояние ухудшилось"},
    ]

    print(f"  {'Описание':<25} {'V(s)':>6} {'r':>6} {'V(s\')':>6} {'Advantage':>10}")
    print(f"  {'-' * 55}")

    for sc in scenarios:
        adv = sc["reward"] + gamma * sc["V_s_next"] - sc["V_s"]
        marker = "+" if adv > 0 else ("-" if adv < 0 else "=")
        print(f"  {sc['desc']:<25} {sc['V_s']:>6.2f} {sc['reward']:>6.2f} "
              f"{sc['V_s_next']:>6.2f} {adv:>+10.4f}  [{marker}]")

    # Нормализованный advantage
    print(f"\n--- Нормализация advantage (batch) ---")
    advantages_raw = [0.5, -0.3, 0.1, 0.8, -0.2, 0.4, -0.1, 0.6]
    print(f"  Raw advantages: {[f'{a:+.2f}' for a in advantages_raw]}")

    mean_a = sum(advantages_raw) / len(advantages_raw)
    std_a = (sum((a - mean_a) ** 2 for a in advantages_raw) / len(advantages_raw)) ** 0.5
    advantages_norm = [(a - mean_a) / (std_a + 1e-8) for a in advantages_raw]
    print(f"  Normalized:     {[f'{a:+.3f}' for a in advantages_norm]}")
    print(f"  Mean: {mean_a:.4f}, Std: {std_a:.4f}")

    print(f"\n--- TD(λ) — N-step returns ---")
    rewards = [0.1, 0.1, 0.1, 1.0, 0.0]
    V_values = [0.3, 0.4, 0.5, 0.9, 0.0]

    for n in [1, 3, 5]:
        # N-step return
        G_n = 0
        for i in range(min(n, len(rewards))):
            G_n += (gamma ** i) * rewards[i]
        if n < len(rewards):
            G_n += (gamma ** n) * V_values[n]

        adv = G_n - V_values[0]
        print(f"  {n}-step return: {G_n:.4f}, Advantage: {adv:+.4f}")

    print()


def demo_3_a2c_training():
    """Демо 3: Обучение A2C агента."""
    print("=" * 60)
    print("ДЕМО 3: Обучение A2C")
    print("=" * 60)

    env = SimpleGridEnv(size=4, seed=42)
    agent = A2C(state_dim=4, action_dim=4, hidden_dim=16,
                actor_lr=0.005, critic_lr=0.001)

    print(f"\nАгент: A2C (Advantage Actor-Critic)")
    print(f"Среда: Grid 4x4, цель: (3,3)")
    print(f"Гиперпараметры: γ=0.99, actor_lr=0.005, critic_lr=0.001\n")

    rewards_history = []

    for episode in range(1, 51):
        reward = agent.train_episode(env, gamma=0.99)
        rewards_history.append(reward)

        if episode % 10 == 0:
            avg = sum(rewards_history[-10:]) / 10
            print(f"  Эпизод {episode:>3}: награда={reward:>+.3f}, "
                  f"средняя за 10={avg:>+.3f}")

    # Финальная оценка
    print(f"\n--- Финальное тестирование (10 эпизодов) ---")
    test_rewards = []
    for _ in range(10):
        env_test = SimpleGridEnv(size=4, seed=random.randint(0, 1000))
        state = env_test.reset()
        total_r = 0
        done = False
        while not done:
            action, _ = agent.select_action(state)
            state, reward, done = env_test.step(action)
            total_r += reward
        test_rewards.append(total_r)

    print(f"  Средняя награда: {sum(test_rewards)/len(test_rewards):+.3f}")
    print(f"  Мин: {min(test_rewards):+.3f}, Макс: {max(test_rewards):+.3f}")

    # Демонстрация работы критика
    print(f"\n--- Критик оценивает позиции ---")
    env_demo = SimpleGridEnv(size=4, seed=42)
    positions = [(0, 0), (0, 3), (3, 0), (2, 2), (3, 3)]
    for pos in positions:
        env_demo.agent_pos = list(pos)
        state = env_demo._get_state()
        value = agent.critic.get_value(state)
        dist = abs(pos[0] - 3) + abs(pos[1] - 3)
        print(f"  Позиция {pos} (расстояние до цели={dist}): V(s)={value:.4f}")

    print()


def demo_4_reinforce_vs_a2c():
    """Демо 4: Сравнение REINFORCE и A2C."""
    print("=" * 60)
    print("ДЕМО 4: REINFORCE vs A2C")
    print("=" * 60)

    # Запускаем оба алгоритма
    n_episodes = 100

    print(f"\nЗапуск {n_episodes} эпизодов каждого алгоритма...")
    print(f"  REINFORCE: lr=0.005")
    print(f"  A2C: actor_lr=0.005, critic_lr=0.001\n")

    # REINFORCE
    random.seed(42)
    env_r = SimpleGridEnv(size=4, seed=42)
    reinforce = REINFORCE(state_dim=4, action_dim=4, lr=0.005)

    r_rewards = []
    for ep in range(n_episodes):
        r = reinforce.train_episode(env_r, gamma=0.99)
        r_rewards.append(r)

    # A2C
    random.seed(42)
    env_a = SimpleGridEnv(size=4, seed=42)
    a2c = A2C(state_dim=4, action_dim=4, hidden_dim=16,
              actor_lr=0.005, critic_lr=0.001)

    a_rewards = []
    for ep in range(n_episodes):
        r = a2c.train_episode(env_a, gamma=0.99)
        a_rewards.append(r)

    # Сравнение
    print(f"  {'Эпизод':<10} {'REINFORCE':>12} {'A2C':>12} {'Лучший':>10}")
    print(f"  {'-' * 46}")

    for ep in [0, 9, 19, 49, 74, 99]:
        r_r = r_rewards[ep]
        r_a = a_rewards[ep]
        winner = "A2C" if r_a > r_r else ("REINFORCE" if r_r > r_a else "=")
        print(f"  {ep+1:<10} {r_r:>+12.3f} {r_a:>+12.3f} {winner:>10}")

    # Статистика по окнам
    print(f"\n--- Средняя награда по окнам (20 эпизодов) ---")
    print(f"  {'Окно':<10} {'REINFORCE':>12} {'A2C':>12} {'Δ (A2C-R)':>10}")
    print(f"  {'-' * 46}")

    for start in range(0, n_episodes, 20):
        end = min(start + 20, n_episodes)
        avg_r = sum(r_rewards[start:end]) / (end - start)
        avg_a = sum(a_rewards[start:end]) / (end - start)
        delta = avg_a - avg_r
        print(f"  {start+1:>3}-{end:<3}    {avg_r:>+12.3f} {avg_a:>+12.3f} {delta:>+10.3f}")

    # Финальные метрики
    print(f"\n--- Финальные метрики ---")
    last_20_r = r_rewards[-20:]
    last_20_a = a_rewards[-20:]

    print(f"  REINFORCE (последние 20):")
    print(f"    Среднее: {sum(last_20_r)/len(last_20_r):>+.3f}")
    print(f"    Стабильность (std): {(sum((x-sum(last_20_r)/20)**2 for x in last_20_r)/20)**0.5:.3f}")

    print(f"  A2C (последние 20):")
    print(f"    Среднее: {sum(last_20_a)/len(last_20_a):>+.3f}")
    print(f"    Стабильность (std): {(sum((x-sum(last_20_a)/20)**2 for x in last_20_a)/20)**0.5:.3f}")

    print(f"\n--- Ключевые различия ---")
    print(f"  REINFORCE:")
    print(f"    • Использует полные returns (G_t)")
    print(f"    • Высокая дисперсия градиентов")
    print(f"    • Не использует критик")
    print(f"    • Полный он-policy")

    print(f"\n  A2C:")
    print(f"    • Использует advantage (G_t - V(s))")
    print(f"    • Критик стабилизирует обучение")
    print(f"    • Низкая дисперсия")
    print(f"    • Параллельные эпизоды возможны")

    print(f"\n  → A2C обычно стабильнее и обучается быстрее\n")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ACTOR-CRITIC МЕТОДЫ")
    print("  Основы + A2C + Advantage + Сравнение с REINFORCE")
    print("=" * 60 + "\n")

    demo_1_actor_critic_architectures()
    demo_2_advantage_computation()
    demo_3_a2c_training()
    demo_4_reinforce_vs_a2c()

    print("=" * 60)
    print("  ВЫВОДЫ")
    print("=" * 60)
    print("""
  1. Actor-Critic совмещает политику и функцию ценности
  2. Advantage (A = r + γV(s') - V(s)) снижает дисперсию
  3. A2C стабильнее REINFORCE за счёт критика
  4. Критик даёт baseline без дополнительной случайности
  5. A2C — основа для PPO, SAC и других методов

  Математика:
    REINFORCE: ∇J = E[G_t · ∇log π(a|s)]
    A2C:       ∇J = E[A_t · ∇log π(a|s)]
    Critic:    Δw = α · (r + γV(s') - V(s)) · ∇V(s)
""")
