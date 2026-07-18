"""
79 - Deep Q-Networks (DQN): основы и практика

Самодостаточный файл без внешних зависимостей (numpy, torch, gym).
Полная реализация DQN с нулевого уровня.

Ключевые концепции:
- Аппроксимация Q-функции нейросетью
- Experience Replay Buffer
- Target Network (фиксированные веса)
- Epsilon-greedy исследование с затуханием

Запуск: python 79-dqn.py
"""
import sys
import math
import random
from collections import deque

random.seed(42)

def p(msg=""):
    print(msg)
    sys.stdout.flush()


# =============================================================================
# БЛОК 1: Среда — Grid World 4x4
# =============================================================================

class GridWorld:
    SIZE = 4
    NAMES = {0: "вверх", 1: "вправо", 2: "вниз", 3: "влево"}
    DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.goal = (3, 3)
        self.steps = 0
        return self.pos

    def step(self, action):
        dr, dc = self.DELTAS[action]
        r, c = self.pos
        self.pos = (max(0, min(3, r + dr)), max(0, min(3, c + dc)))
        self.steps += 1
        if self.pos == self.goal:
            return self.pos, 10.0, True
        dist = abs(3 - self.pos[0]) + abs(3 - self.pos[1])
        done = self.steps >= 100
        return self.pos, (-5.0 if done else -0.1 - 0.01 * dist), done

    def idx(self, s):
        return s[0] * 4 + s[1]


# =============================================================================
# БЛОК 2: Нейросеть для Q-функции
# =============================================================================

class QNetwork:
    """
    Нейросеть для аппроксимации Q(s,a).

    Архитектура (линейная модель — однослойный перцептрон):
      input:  one-hot вектор [16] для текущего состояния
      output: Q-значения [4] для каждого действия

      Q(s, a) = W[s*4 + a]

    16 состояний x 4 действия = 64 параметра.
    В реальных задачах ( Atari, robotic control ) используется
    многослойная свёрточная/полносвязная сеть с тысячами параметров.
    """
    def __init__(self, n_states=16, n_actions=4, lr=0.1):
        self.W = [random.gauss(0, 0.01) for _ in range(n_states * n_actions)]
        self.lr = lr

    def predict(self, si):
        b = si * 4
        return [self.W[b], self.W[b+1], self.W[b+2], self.W[b+3]]

    def train_step(self, si, target_q):
        b = si * 4
        for a in range(4):
            # SGD: w -= lr * d(Loss)/dw, Loss = (pred - target)^2
            self.W[b + a] -= self.lr * 2.0 * (self.W[b + a] - target_q[a])

    def copy_from(self, other):
        self.W = other.W[:]


# =============================================================================
# БЛОК 3: Experience Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Буфер повторного проигрывания опыта.

    Хранит кортежи (state, action, reward, next_state, done).
    Позволяет извлекать случайные мини-батчи, разрывая
    корреляцию между последовательными транзициями.
    """
    def __init__(self, cap=1000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        return ([t[i] for t in batch] for i in range(5))

    def __len__(self):
        return len(self.buf)


# =============================================================================
# БЛОК 4: DQN-агент
# =============================================================================

class DQNAgent:
    """
    DQN-агент.

    Компоненты:
    1. q_net — основная сеть, обучается на каждом шаге
    2. target_net — фиксированная копия, обновляется каждые N эпизодов
    3. replay — буфер опыта для мини-батчей
    4. epsilon — скорость исследования (greedy vs explore)
    """
    def __init__(self, lr=0.1):
        self.q_net = QNetwork(lr=lr)
        self.target_net = QNetwork(lr=lr)
        self.target_net.copy_from(self.q_net)
        self.replay = ReplayBuffer(5000)
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32

    def sync(self):
        """Скопировать веса q_net -> target_net."""
        self.target_net.copy_from(self.q_net)

    def act(self, si):
        """Epsilon-greedy выбор действия."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        q = self.q_net.predict(si)
        return q.index(max(q))

    def learn(self):
        """Обучение на мини-батче из replay buffer."""
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, nss, dones = list(self.replay.sample(self.batch_size))
        for i in range(len(states)):
            si, a, r, nsi, d = states[i], actions[i], rewards[i], nss[i], dones[i]
            q = self.q_net.predict(si)
            tgt = q[:]
            # Bellman: target = r + gamma * max_a' Q_target(s', a')
            tgt[a] = r + (0.0 if d else self.gamma * max(self.target_net.predict(nsi)))
            self.q_net.train_step(si, tgt)
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# =============================================================================
# БЛОК 5: Tabular Q-Learning
# =============================================================================

class TabularQL:
    """Классический tabular Q-learning для сравнения."""
    def __init__(self, lr=0.1, gamma=0.99):
        self.q = [[0.0] * 4 for _ in range(16)]
        self.lr, self.gamma = lr, gamma
        self.epsilon = 1.0

    def act(self, si):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return self.q[si].index(max(self.q[si]))

    def learn(self, s, a, r, ns, done):
        tgt = r if done else r + self.gamma * max(self.q[ns])
        self.q[s][a] += self.lr * (tgt - self.q[s][a])

    def decay_eps(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def q_of(net, idx):
    return net.predict(idx) if hasattr(net, 'predict') else net[idx]

def print_policy(net, env, title):
    p(f"\n{title}:")
    p("  " + "-" * (env.SIZE * 6))
    for r in range(env.SIZE):
        s = "  "
        for c in range(env.SIZE):
            i = env.idx((r, c))
            if (r, c) == env.goal:
                s += "  GOAL "
            else:
                q = q_of(net, i)
                s += f" {env.NAMES[q.index(max(q))]:^4s} "
        p(s)
    p("  " + "-" * (env.SIZE * 6))

def run_episode(env, net):
    s = env.reset()
    total = 0; path = []
    while True:
        q = q_of(net, env.idx(s))
        a = q.index(max(q))
        path.append((s[0], s[1], env.NAMES[a]))
        s, r, d = env.step(a)
        total += r
        if d:
            break
    return total, path, s


# =============================================================================
# ДЕМО 1: Нейросеть для Q-функции
# =============================================================================

def demo1():
    p("=" * 70)
    p("ДЕМО 1: Нейросеть для аппроксимации Q-функции")
    p("=" * 70)

    env = GridWorld()
    net = QNetwork(lr=0.1)
    p("\nАрхитектура: one-hot(16) -> Q-values(4)")
    p("Параметров: 16 * 4 = 64 (матрица весов W)")
    p("Это однослойный перцептрон — базовая нейросеть.")
    p("В реальных задачах добавляются скрытые слои с ReLU.")

    p("\n--- Q(0,0) ДО обучения ---")
    for i, nm in GridWorld.NAMES.items():
        p(f"  {nm:>6s}: {net.predict(0)[i]:+.4f}")

    # Обучение (online, без replay — как baseline)
    for ep in range(100):
        s = env.reset()
        done = False
        while not done:
            si = env.idx(s)
            qv = net.predict(si)
            a = qv.index(max(qv))
            ns, r, done = env.step(a)
            tgt = qv[:]
            tgt[a] = r + (0.0 if done else 0.99 * max(net.predict(env.idx(ns))))
            net.train_step(si, tgt)
            s = ns

    p("\n--- Q(0,0) ПОСЛЕ 100 эпизодов ---")
    for i, nm in GridWorld.NAMES.items():
        p(f"  {nm:>6s}: {net.predict(0)[i]:+.4f}")

    print_policy(net, env, "Выученная политика")

    p("\nПрохождение:")
    rew, path, final = run_episode(env, net)
    p(f"  {len(path)} шагов, награда={rew:.2f}, цель={'Да' if final == env.goal else 'Нет'}")
    for r, c, a in path:
        p(f"    ({r},{c}) -> {a}")

    p("\nКлючевой момент: нейросеть АППРОКСИМИРУЕТ Q-функцию.")
    p("Для 16 состояний хватает таблицы (tabular).")
    p("Но для миллионов состояний (Atari, роботы) нужна нейросеть!")


# =============================================================================
# ДЕМО 2: Experience Replay
# =============================================================================

def demo2():
    p("\n" + "=" * 70)
    p("ДЕМО 2: Experience Replay Buffer")
    p("=" * 70)

    buf = ReplayBuffer(20)
    env = GridWorld()
    p("\nЗаполнение буфера (20 транзиций)...")
    for _ in range(20):
        s = env.reset()
        a = random.randint(0, 3)
        ns, r, d = env.step(a)
        buf.push(env.idx(s), a, r, env.idx(ns), d)
    p(f"  Размер: {len(buf)} / 20")

    p("\nСлучайный батч из 5 транзиций:")
    states, actions, rewards, nss, dones = list(buf.sample(5))
    for i in range(5):
        p(f"  s=({states[i]//4},{states[i]%4}) a={GridWorld.NAMES[actions[i]]} "
          f"-> ns=({nss[i]//4},{nss[i]%4}) r={rewards[i]:+.2f} done={dones[i]}")

    p("\nСтатистика (100 случайных батчей по 5):")
    rs = []
    for _ in range(100):
        _, _, r, _, _ = list(buf.sample(5))
        rs.extend(r)
    p(f"  Средняя награда: {sum(rs)/len(rs):+.4f}")

    p("\nДемонстрация перезаписи (cap=5, добавляем 10):")
    sm = ReplayBuffer(5)
    for i in range(10):
        sm.push(i, 0, float(i), 0, i == 9)
    p(f"  Размер: {len(sm)}, награды: {[t[2] for t in sm.buf]}")
    p("  (Старые записи перезаписаны новыми)")

    p("\nЗачем нужен Replay?")
    p("  - Последовательные транзиции коррелированы")
    p("  - Случайные батчи разрывают эту корреляцию")
    p("  - Один опыт используется многократно (эффективность)")


# =============================================================================
# ДЕМО 3: Обучение DQN
# =============================================================================

def demo3():
    p("\n" + "=" * 70)
    p("ДЕМО 3: Обучение DQN-агента")
    p("=" * 70)

    env = GridWorld()
    agent = DQNAgent(lr=0.1)
    EP = 300

    p(f"\nПараметры:")
    p(f"  Episodes: {EP}")
    p(f"  Epsilon: {agent.epsilon} -> {agent.eps_min} (decay={agent.eps_decay})")
    p(f"  Gamma: {agent.gamma}")
    p(f"  LR: 0.1, Batch: 32")
    p(f"  Replay buffer: 5000")
    p(f"  Target sync: каждые 20 эпизодов")

    hist = []
    p("\nОбучение...")
    for ep in range(EP):
        s = env.reset()
        tr = 0
        done = False
        while not done:
            si = env.idx(s)
            a = agent.act(si)
            ns, r, done = env.step(a)
            agent.replay.push(si, a, r, env.idx(ns), done)
            agent.learn()
            s = ns
            tr += r
        hist.append(tr)
        if (ep + 1) % 20 == 0:
            agent.sync()
            avg = sum(hist[-20:]) / 20
            p(f"  Эпизод {ep+1:3d}: avg={avg:+7.2f}, eps={agent.epsilon:.3f}")

    first = sum(hist[:20]) / 20
    last = sum(hist[-20:]) / 20
    p(f"\nИтоги:")
    p(f"  Epsilon финальный: {agent.epsilon:.4f}")
    p(f"  Средняя (первые 20): {first:+.2f}")
    p(f"  Средняя (последние 20): {last:+.2f}")
    p(f"  Лучшая награда за эпизод: {max(hist):+.2f}")
    p(f"  Улучшение: {last - first:+.2f}")

    print_policy(agent.q_net, env, "DQN политика (выученная)")
    rew, path, final = run_episode(env, agent.q_net)
    p(f"\nПрохождение: {len(path)} шагов, награда={rew:.2f}, цель={'Да' if final == env.goal else 'Нет'}")
    for r, c, a in path:
        p(f"  ({r},{c}) -> {a}")


# =============================================================================
# ДЕМО 4: Сравнение DQN vs Tabular
# =============================================================================

def demo4():
    p("\n" + "=" * 70)
    p("ДЕМО 4: Сравнение DQN vs Tabular Q-Learning")
    p("=" * 70)
    EP = 300

    # --- Tabular ---
    p("\n--- Tabular Q-Learning ---")
    random.seed(42)
    env1 = GridWorld()
    tab = TabularQL(lr=0.1)
    tr = []
    for _ in range(EP):
        s = env1.reset()
        r = 0; done = False
        while not done:
            si = env1.idx(s); a = tab.act(si)
            ns, rew, done = env1.step(a)
            tab.learn(si, a, rew, env1.idx(ns), done)
            s = ns; r += rew
        tab.decay_eps(); tr.append(r)

    t1 = sum(tr[:20]) / 20; t2 = sum(tr[-20:]) / 20
    p(f"  Средняя (первые 20): {t1:+.2f}")
    p(f"  Средняя (последние 20): {t2:+.2f}")
    p(f"  Улучшение: {t2-t1:+.2f}")
    print_policy(tab.q, env1, "Tabular политика")
    rw, pp, _ = run_episode(env1, tab.q)
    p(f"  Прохождение: {len(pp)} шагов, награда={rw:.2f}")

    # --- DQN ---
    p("\n--- DQN ---")
    random.seed(42)
    env2 = GridWorld()
    agent = DQNAgent(lr=0.1)
    dr = []
    for ep in range(EP):
        s = env2.reset(); r = 0; done = False
        while not done:
            si = env2.idx(s); a = agent.act(si)
            ns, rew, done = env2.step(a)
            agent.replay.push(si, a, rew, env2.idx(ns), done)
            agent.learn(); s = ns; r += rew
        dr.append(r)
        if (ep + 1) % 20 == 0:
            agent.sync()

    d1 = sum(dr[:20]) / 20; d2 = sum(dr[-20:]) / 20
    p(f"  Средняя (первые 20): {d1:+.2f}")
    p(f"  Средняя (последние 20): {d2:+.2f}")
    p(f"  Улучшение: {d2-d1:+.2f}")
    print_policy(agent.q_net, env2, "DQN политика")
    rw, pp, _ = run_episode(env2, agent.q_net)
    p(f"  Прохождение: {len(pp)} шагов, награда={rw:.2f}")

    # --- Сравнение ---
    p("\n" + "-" * 55)
    p("ИТОГОВОЕ СРАВНЕНИЕ:")
    p("-" * 55)
    p(f"{'Метрика':<30} {'Tabular':>10} {'DQN':>10}")
    p("-" * 55)
    p(f"{'Награда (первые 20)':<30} {t1:>+10.2f} {d1:>+10.2f}")
    p(f"{'Награда (последние 20)':<30} {t2:>+10.2f} {d2:>+10.2f}")
    p(f"{'Улучшение':<30} {t2-t1:>+10.2f} {d2-d1:>+10.2f}")

    p(f"\nВыводы:")
    p(f"  1. Tabular Q-learning: точное решение для малых дискретных задач")
    p(f"  2. DQN с аппроксимацией конкурирует с tabular")
    p(f"  3. Target network стабилизирует обучение DQN")
    p(f"  4. Experience replay устраняет корреляцию транзиций")
    p(f"  5. DQN масштабируется: для Atari/роботов нужна глубокая сеть")
    p(f"     с Conv-слоями (50K+ параметров вместо 64)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    p("79 - Deep Q-Networks (DQN): основы и практика")
    p("Реализация на чистом Python (numpy/torch не используются)\n")

    demo1()
    demo2()
    demo3()
    demo4()

    p("\n" + "=" * 70)
    p("ВСЕ ДЕМО ЗАВЕРШЕНЫ")
    p("=" * 70)
