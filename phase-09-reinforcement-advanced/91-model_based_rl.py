"""
91 - Model-Based Reinforcement Learning
=======================================
Основы Model-Based RL: модель среды, планирование, Dyna-Q.
"""

import random

random.seed(42)


# ============================================================
# 1. Среда: 4x4 сетка (Grid World) с 4 действиями
# ============================================================

class GridWorld:
    """Простая сетка 4x4. Агент移动 (0,0) → (3,3). Яма в (1,1)."""

    SIZE = 4
    ACTIONS = [0, 1, 2, 3]  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    TERMINAL = (3, 3)
    PIT = (1, 1)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, state, action):
        """Возвращает (next_state, reward, done)."""
        if state == self.TERMINAL:
            return state, 0.0, True
        if state == self.PIT:
            return state, -5.0, True

        dr, dc = self.DELTAS[action]
        r, c = state[0] + dr, state[1] + dc
        r = max(0, min(self.SIZE - 1, r))
        c = max(0, min(self.SIZE - 1, c))
        ns = (r, c)

        if ns == self.TERMINAL:
            return ns, 10.0, True
        if ns == self.PIT:
            return ns, -5.0, True
        return ns, -0.1, False

    def all_state_action_pairs(self):
        pairs = []
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                s = (r, c)
                for a in self.ACTIONS:
                    pairs.append((s, a))
        return pairs


# ============================================================
# 2. Модель среды (World Model)
# ============================================================

class TransitionModel:
    """
    Учит статистику переходов: count[s,a,s'] и reward[s,a,s'].
    Возвращает предсказание по частоте.
    """

    def __init__(self, env):
        self.env = env
        self.counts = {}   # (s,a,s') → int
        self.rewards = {}  # (s,a,s') → sum of rewards
        self.visited = set()  # (s,a)

    def update(self, s, a, r, s2):
        key = (s, a, s2)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.rewards[key] = self.rewards.get(key, 0.0) + r
        self.visited.add((s, a))

    def predict(self, s, a):
        """
        Возвращает список (next_state, reward, probability) —
        все возможные исходы для (s, a).
        """
        total = 0
        results = {}
        for r in range(self.env.SIZE):
            for c in range(self.env.SIZE):
                ns = (r, c)
                key = (s, a, ns)
                if key in self.counts:
                    total += self.counts[key]
                    avg_r = self.rewards[key] / self.counts[key]
                    results[ns] = avg_r

        if total == 0:
            return [(s, 0.0, 1.0)]

        return [(ns, r, self.counts.get((s, a, ns), 0) / total)
                for ns, r in results.items()]

    def sample(self, s, a):
        """Сэмплирует один переход из модели."""
        outcomes = self.predict(s, a)
        if len(outcomes) == 1 and outcomes[0][2] == 0:
            return outcomes[0][0], outcomes[0][1]
        total = sum(o[2] for o in outcomes)
        r_val = random.random() * total
        cumul = 0
        for ns, reward, prob in outcomes:
            cumul += prob
            if r_val <= cumul:
                return ns, reward
        return outcomes[-1][0], outcomes[-1][1]


# ============================================================
# 3. Планирование (Planning) — поиск лучшей стратегии через модель
# ============================================================

def value_iteration_model(model, env, gamma=0.95, iterations=50):
    """
    Value Iteration, используя модель переходов.
    Возвращает таблицу Q(s,a).
    """
    Q = {}
    for s, a in env.all_state_action_pairs():
        Q[(s, a)] = 0.0

    for _ in range(iterations):
        Q_new = dict(Q)
        for s, a in env.all_state_action_pairs():
            if s == env.TERMINAL or s == env.PIT:
                Q_new[(s, a)] = 0.0
                continue
            outcomes = model.predict(s, a)
            val = 0
            for ns, reward, prob in outcomes:
                max_q_next = max(Q[(ns, act)] for act in env.ACTIONS)
                val += prob * (reward + gamma * max_q_next)
            Q_new[(s, a)] = val
        Q = Q_new
    return Q


# ============================================================
# 4. Dyna-Q алгоритм
# ============================================================

def dyna_q(env, n_planning=10, episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1):
    """
    Dyna-Q: реальные + модельные переходы.
    """
    model = TransitionModel(env)
    Q = {}
    for s, a in env.all_state_action_pairs():
        Q[(s, a)] = 0.0

    steps_per_episode = []

    for ep in range(episodes):
        s = env.reset()
        steps = 0

        while s != env.TERMINAL and s != env.PIT and steps < 200:
            # ε-greedy выбор действия
            if random.random() < epsilon:
                a = random.choice(env.ACTIONS)
            else:
                a = max(env.ACTIONS, key=lambda act: Q[(s, act)])

            s2, r, done = env.step(s, a)

            # Обновляем Q по реальному переходу
            best_next = max(Q[(s2, act)] for act in env.ACTIONS)
            Q[(s, a)] += alpha * (r + gamma * best_next * (1 - int(done)) - Q[(s, a)])

            # Запоминаем в модель
            model.update(s, a, r, s2)

            # Планирование: сэмплируем из модели
            for _ in range(n_planning):
                s_rand, a_rand = random.choice(
                    [(ss, aa) for ss, aa in env.all_state_action_pairs()
                     if (ss, aa) in model.visited]
                )
                s2_rand, r_rand = model.sample(s_rand, a_rand)
                best_next_rand = max(Q[(s2_rand, act)] for act in env.ACTIONS)
                Q[(s_rand, a_rand)] += alpha * (
                    r_rand + gamma * best_next_rand - Q[(s_rand, a_rand)]
                )

            s = s2
            steps += 1

        steps_per_episode.append(steps)

    return Q, steps_per_episode


# ============================================================
# 5. Q-Learning (Model-Free) для сравнения
# ============================================================

def q_learning(env, episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1):
    Q = {}
    for s, a in env.all_state_action_pairs():
        Q[(s, a)] = 0.0

    steps_per_episode = []

    for ep in range(episodes):
        s = env.reset()
        steps = 0

        while s != env.TERMINAL and s != env.PIT and steps < 200:
            if random.random() < epsilon:
                a = random.choice(env.ACTIONS)
            else:
                a = max(env.ACTIONS, key=lambda act: Q[(s, act)])

            s2, r, done = env.step(s, a)
            best_next = max(Q[(s2, act)] for act in env.ACTIONS)
            Q[(s, a)] += alpha * (r + gamma * best_next * (1 - int(done)) - Q[(s, a)])

            s = s2
            steps += 1

        steps_per_episode.append(steps)

    return Q, steps_per_episode


# ============================================================
# ДЕМО
# ============================================================

def demo_1():
    """Демо 1: Модель среды — предсказание переходов."""
    print("=" * 60)
    print("ДЕМО 1: Модель среды — предсказание переходов")
    print("=" * 60)

    env = GridWorld()
    model = TransitionModel(env)

    # Собираем опыт
    transitions = [
        ((0, 0), 1, (0, 1)),
        ((0, 1), 2, (1, 1)),
        ((0, 0), 2, (1, 0)),
        ((1, 0), 1, (1, 1)),
        ((2, 2), 1, (2, 3)),
        ((2, 3), 2, (3, 3)),
    ]
    for s, a, s2 in transitions:
        r = env.step(s, a)[1]
        model.update(s, a, r, s2)

    test_cases = [
        ((0, 0), 1, "RIGHT"),
        ((0, 0), 2, "DOWN"),
        ((2, 3), 2, "DOWN"),
    ]

    for s, a, action_name in test_cases:
        outcomes = model.predict(s, a)
        print(f"\n  Состояние {s}, действие {action_name} ({a}):")
        for ns, reward, prob in outcomes:
            print(f"    → Состояние {ns}, награда {reward:.1f}, "
                  f"вероятность {prob:.2f}")

    # Сэмплирование
    print("\n  Сэмплирование 5 переходов из (0,0) → RIGHT:")
    for i in range(5):
        ns, r = model.sample((0, 0), 1)
        print(f"    [{i+1}] → {ns}, награда {r:.1f}")

    print()


def demo_2():
    """Демо 2: Планирование через модель."""
    print("=" * 60)
    print("ДЕМО 2: Планирование через модель (Value Iteration)")
    print("=" * 60)

    env = GridWorld()
    model = TransitionModel(env)

    # Заполняем модель реальными переходами
    for s, a in env.all_state_action_pairs():
        s2, r, done = env.step(s, a)
        model.update(s, a, r, s2)

    Q = value_iteration_model(model, env)

    print("\n  Оптимальные Q-значения (ключевые состояния):")
    key_states = [
        ((0, 0), "старт"), ((0, 2), "верхний ряд"),
        ((2, 0), "левый столбец"), ((2, 2), "перед финишем"),
    ]
    for s, desc in key_states:
        print(f"\n  {s} ({desc}):")
        for a in range(4):
            name = env.ACTION_NAMES[a]
            print(f"    {name:>5s}: Q = {Q[(s, a)]:+.3f}")

    print("\n  Оптимальная политика:")
    for r in range(env.SIZE):
        row = ""
        for c in range(env.SIZE):
            s = (r, c)
            if s == env.TERMINAL:
                row += " GOAL  "
            elif s == env.PIT:
                row += "  PIT  "
            else:
                best_a = max(env.ACTIONS, key=lambda a: Q[(s, a)])
                row += f" {env.ACTION_NAMES[best_a]:>4s}  "
        print(f"    {row}")

    print()


def demo_3():
    """Демо 3: Dyna-Q — обучение с планированием."""
    print("=" * 60)
    print("ДЕМО 3: Dyna-Q — обучение с планированием")
    print("=" * 60)

    env = GridWorld()

    # Запускаем Dyna-Q с разным количеством планирований
    configs = [
        (0, "Без планирования (только реальные переходы)"),
        (5, "Планирование: 5 итераций"),
        (20, "Планирование: 20 итераций"),
    ]

    all_results = {}
    for n_plan, label in configs:
        random.seed(42)
        Q, steps = dyna_q(env, n_planning=n_plan, episodes=150)
        all_results[label] = steps
        avg_start = sum(steps[:10]) / 10
        avg_end = sum(steps[-10:]) / 10
        print(f"\n  {label}:")
        print(f"    Начало (эп. 1-10):   среднее {avg_start:.1f} шагов")
        print(f"    Конец  (эп. 141-150): среднее {avg_end:.1f} шагов")

    # Сравнение кривых обучения
    print("\n  Сравнение (шагов к решению, каждые 30 эпизодов):")
    print(f"  {'Эпизод':>8s}  {'Без планир.':>12s}  {'5 план.':>10s}  {'20 план.':>10s}")
    print("  " + "-" * 45)
    labels = list(all_results.keys())
    for i in range(0, 150, 30):
        vals = [all_results[l][i] for l in labels]
        print(f"  {i+1:>8d}  {vals[0]:>12.1f}  {vals[1]:>10.1f}  {vals[2]:>10.1f}")

    print()


def demo_4():
    """Демо 4: Сравнение model-based vs model-free."""
    print("=" * 60)
    print("ДЕМО 4: Сравнение Model-Based vs Model-Free")
    print("=" * 60)

    env = GridWorld()
    n_episodes = 200

    # Dyna-Q (model-based)
    random.seed(42)
    dyna_steps = dyna_q(env, n_planning=10, episodes=n_episodes)[1]

    # Q-Learning (model-free)
    random.seed(42)
    q_steps = q_learning(env, episodes=n_episodes)[1]

    # Анализ
    def avg_last(steps, n=20):
        return sum(steps[-n:]) / n

    print(f"\n  Среднее кол-во шагов к финишу (последние {20} эпизодов):")
    print(f"    Dyna-Q (model-based):  {avg_last(dyna_steps):.2f}")
    print(f"    Q-Learning (model-free): {avg_last(q_steps):.2f}")

    # Эффективность по сбору данных
    print("\n  Эффективность сбора данных (шагов к решению):")
    print(f"  {'Эпизод':>8s}  {'Dyna-Q':>10s}  {'Q-Learn':>10s}  {'Экономия':>10s}")
    print("  " + "-" * 45)
    for i in range(0, n_episodes, 40):
        d = dyna_steps[i]
        q = q_steps[i]
        saving = (1 - d / max(q, 1)) * 100
        print(f"  {i+1:>8d}  {d:>10.1f}  {q:>10.1f}  {saving:>+9.1f}%")

    print("\n  Ключевые преимущества:")
    print("    Model-Based (Dyna-Q):")
    print("      + Быстрее сходится за счёт планирования")
    print("      + Эффективнее использует каждый реальный переход")
    print("      + Может планировать без взаимодействия со средой")
    print("    Model-Free (Q-Learning):")
    print("      + Проще реализовать")
    print("      + Не нужна модель (нет ошибок модели)")
    print("      + Устойчив к неточностям модели")
    print("\n  Вывод: Dyna-Q значительно быстрее сходится,")
    print("  т.к. каждый реальный переход 'используется' повторно")
    print("  через планирование в модели.")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    print("Phase 09, Файл 91: Model-Based Reinforcement Learning")
    print()

    demo_1()
    demo_2()
    demo_3()
    demo_4()

    print("=" * 60)
    print("Все демо завершены!")
    print("=" * 60)
