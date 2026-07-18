"""
77 — Multi-Armed Bandits
========================
Основы стратегий исследования-эксплуатации (explore-exploit):
  • Epsilon-Greedy
  • UCB (Upper Confidence Bound)
  • Thompson Sampling

Без внешних зависимостей — только random + math.
"""

import random
import math
from collections import defaultdict

random.seed(42)

# ────────────────────────────────────────────────────────────────
# Среда: K-armed bandit
# ────────────────────────────────────────────────────────────────

class Bandit:
    """Классический K-armed bandit с нормальными наградами."""

    def __init__(self, k: int = 10, seed: int = 42):
        rng = random.Random(seed)
        self.k = k
        # Истинные средние награды (случайные из N(0,1))
        self.q_true = [rng.gauss(0, 1) for _ in range(k)]
        # Стандартное отклонение наград
        self.sigma = 1.0

    def reward(self, action: int) -> float:
        """Сэмпл награды для выбранного action."""
        return random.gauss(self.q_true[action], self.sigma)

    def best_action(self) -> int:
        """Индекс лучшего action (для подсчёта regret)."""
        return self.q_true.index(max(self.q_true))

    def __repr__(self):
        return f"Bandit(k={self.k})"


# ────────────────────────────────────────────────────────────────
# Стратегии
# ────────────────────────────────────────────────────────────────

class EpsilonGreedy:
    """Epsilon-Greedy стратегия.

    С вероятностью epsilon выбирает случайный action (explore),
    иначе — action с максимальной оценкой Q (exploit).
    """

    def __init__(self, k: int, epsilon: float = 0.1, seed: int = 42):
        self.k = k
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.Q = [0.0] * k          # оценки средних наград
        self.N = [0] * k            # количество выборов каждого action
        self.t = 0                  # общий счётчик шагов

    def select_action(self) -> int:
        self.t += 1
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.k - 1)
        # Break ties randomly: если несколько Q равны — выбрать случайно
        max_q = max(self.Q)
        best = [a for a in range(self.k) if self.Q[a] == max_q]
        return self.rng.choice(best)

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def __repr__(self):
        return f"EpsilonGreedy(eps={self.epsilon})"


class UCB1:
    """Upper Confidence Bound стратегия.

    Выбирает action, максимизирующий:
        Q(a) + c * sqrt(ln(t) / N(a))
    """

    def __init__(self, k: int, c: float = 1.0, seed: int = 42):
        self.k = k
        self.c = c
        self.rng = random.Random(seed)
        self.Q = [0.0] * k
        self.N = [0] * k
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        # Сначала перебрать все action хотя бы раз
        for a in range(self.k):
            if self.N[a] == 0:
                return a
        ucb_values = []
        for a in range(self.k):
            confidence = self.c * math.sqrt(math.log(self.t) / self.N[a])
            ucb_values.append(self.Q[a] + confidence)
        max_ucb = max(ucb_values)
        best = [a for a in range(self.k) if ucb_values[a] == max_ucb]
        return self.rng.choice(best)

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def __repr__(self):
        return f"UCB1(c={self.c})"


class ThompsonSampling:
    """Thompson Sampling для Gaussian bandits.

    Моделирует каждую руку Beta-распределением (binomial) или
    Normal-Normal (для непрерывных наград).
    Для простоты используем Normal-Normal conjugacy:
        prior:       N(mu_0, sigma_0^2)
        likelihood:  N(theta, sigma^2)
        posterior:   N(mu_n, sigma_n^2)
    """

    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.rng = random.Random(seed)
        # Априорные параметры N(mu_0, sigma_0^2) для каждого action
        self.mu_prior = [0.0] * k       # mu_0 = 0
        self.sigma2_prior = [1.0] * k   # sigma_0^2 = 1
        # Наблюдённые данные (сумма наград и количество)
        self.sum_reward = [0.0] * k
        self.N = [0] * k
        self.sigma_noise = 1.0  # известная дисперсия шума

    def select_action(self) -> int:
        samples = []
        for a in range(self.k):
            # Posterior params: sigma_n^2 = 1 / (1/sigma_0^2 + N/sigma_noise^2)
            sigma2_post = 1.0 / (1.0 / self.sigma2_prior[a] + self.N[a] / self.sigma_noise**2)
            # mu_n = sigma_n^2 * (mu_0/sigma_0^2 + sum_r/sigma_noise^2)
            mu_post = sigma2_post * (
                self.mu_prior[a] / self.sigma2_prior[a]
                + self.sum_reward[a] / self.sigma_noise**2
            )
            sample = self.rng.gauss(mu_post, math.sqrt(sigma2_post))
            samples.append(sample)
        max_val = max(samples)
        best = [a for a in range(self.k) if samples[a] == max_val]
        return self.rng.choice(best)

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.sum_reward[action] += reward

    def __repr__(self):
        return "ThompsonSampling"


# ────────────────────────────────────────────────────────────────
# Утилита: прокат (run one experiment)
# ────────────────────────────────────────────────────────────────

def run_episode(bandit, strategy, steps: int = 1000):
    """Запускает одну эпизодную симуляцию.

    Возвращает:
        rewards  — список полученных наград по шагам
        regrets  — список мгновенных regret по шагам
    """
    best = bandit.best_action()
    rewards = []
    regrets = []
    for _ in range(steps):
        a = strategy.select_action()
        r = bandit.reward(a)
        strategy.update(a, r)
        rewards.append(r)
        regrets.append(bandit.q_true[best] - bandit.q_true[a])
    return rewards, regrets


def cumulative(values):
    """Кумулятивная сумма."""
    result = []
    s = 0
    for v in values:
        s += v
        result.append(s)
    return result


def running_average(values, window=100):
    """Скользящее среднее."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


# ────────────────────────────────────────────────────────────────
# Демо 1: Epsilon-Greedy (разные ε)
# ────────────────────────────────────────────────────────────────

def demo_epsilon_greedy():
    print("=" * 70)
    print("ДЕМО 1: Epsilon-Greedy — влияние ε на regret")
    print("=" * 70)

    k = 10
    steps = 1000
    epsilons = [0.0, 0.01, 0.1, 0.2, 0.5]

    print(f"\nBandit: {k} рук, {steps} шагов")
    print(f"{'ε':>6} | {'Средняя награда (последние 200)':>32} | {'Cumul. Regret':>14}")
    print("-" * 60)

    for eps in epsilons:
        bandit = Bandit(k=k)
        strategy = EpsilonGreedy(k=k, epsilon=eps, seed=42)
        rewards, regrets = run_episode(bandit, strategy, steps)
        avg_reward = sum(rewards[-200:]) / 200
        cum_regret = sum(regrets)
        print(f"{eps:>6.2f} | {avg_reward:>32.3f} | {cum_regret:>14.1f}")

    print()
    print("Вывод: eps=0 (чистая exploit) — высокий regret из-за жадности.")
    print("       eps=0.1 — хороший баланс explore/exploit.")
    print("       eps=0.5 — слишком много случайных действий.")


# ────────────────────────────────────────────────────────────────
# Демо 2: UCB
# ────────────────────────────────────────────────────────────────

def demo_ucb():
    print("\n" + "=" * 70)
    print("ДЕМО 2: UCB1 — Upper Confidence Bound")
    print("=" * 70)

    k = 10
    steps = 1000
    c_values = [0.5, 1.0, 2.0, 5.0]

    bandit = Bandit(k=k)
    print(f"\nBandit: {k} рук, {steps} шагов")
    print(f"Истинные Q: {[round(q, 2) for q in bandit.q_true]}")
    print(f"Лучший action: {bandit.best_action()} (Q={bandit.q_true[bandit.best_action()]:.2f})\n")

    print(f"{'c':>6} | {'Средняя нагр. (посл. 200)':>26} | {'Cumul. Regret':>14} | {'Лучший action %':>16}")
    print("-" * 72)

    for c in c_values:
        bandit = Bandit(k=k)
        strategy = UCB1(k=k, c=c, seed=42)
        rewards, regrets = run_episode(bandit, strategy, steps)
        avg_reward = sum(rewards[-200:]) / 200
        cum_regret = sum(regrets)
        best_pct = sum(1 for a in strategy.N if a == bandit.best_action()) / steps * 100
        print(f"{c:>6.1f} | {avg_reward:>26.3f} | {cum_regret:>14.1f} | {best_pct:>15.1f}%")

    print()
    print("Вывод: c=1.0 — стандартный UCB, хороший результат.")
    print("       c=0.5 — консервативнее (меньше exploration).")
    print("       c=5.0 — агрессивный exploration, больше regret на старте.")


# ────────────────────────────────────────────────────────────────
# Демо 3: Thompson Sampling
# ────────────────────────────────────────────────────────────────

def demo_thompson():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Thompson Sampling")
    print("=" * 70)

    k = 10
    steps = 1000

    bandit = Bandit(k=k)
    strategy = ThompsonSampling(k=k, seed=42)
    rewards, regrets = run_episode(bandit, strategy, steps)

    print(f"\nBandit: {k} рук, {steps} шагов")
    print(f"Истинные Q: {[round(q, 2) for q in bandit.q_true]}")
    print(f"Лучший action: {bandit.best_action()}\n")

    print("Распределение выборов по action:")
    total = sum(strategy.N)
    for a in range(k):
        bar = "#" * int(strategy.N[a] / total * 40)
        print(f"  Action {a}: Q={bandit.q_true[a]:>6.2f}  "
              f"N={strategy.N[a]:>4} ({strategy.N[a]/total*100:>5.1f}%) {bar}")

    print(f"\nСредняя награда (первые 100): {sum(rewards[:100])/100:.3f}")
    print(f"Средняя награда (последние 100): {sum(rewards[-100:])/100:.3f}")
    print(f"Cumulative regret: {sum(regrets):.1f}")

    print()
    print("Вывод: Thompson Sampling быстро фокусируется на лучших action.")
    print("       Байесовская posterior автоматически балансирует")
    print("       exploration и exploitation без гиперпараметров.")


# ────────────────────────────────────────────────────────────────
# Демо 4: Сравнение всех стратегий
# ────────────────────────────────────────────────────────────────

def demo_comparison():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Сравнение всех стратегий")
    print("=" * 70)

    k = 10
    steps = 2000
    n_runs = 20  # усредняем по нескольким запускам

    strategies_config = [
        ("ε-greedy ε=0.01", lambda: EpsilonGreedy(k=k, epsilon=0.01)),
        ("ε-greedy ε=0.1",  lambda: EpsilonGreedy(k=k, epsilon=0.1)),
        ("ε-greedy ε=0.2",  lambda: EpsilonGreedy(k=k, epsilon=0.2)),
        ("UCB1 c=1.0",      lambda: UCB1(k=k, c=1.0)),
        ("UCB1 c=2.0",      lambda: UCB1(k=k, c=2.0)),
        ("Thompson Sampling", lambda: ThompsonSampling(k=k)),
    ]

    print(f"\nБенчмарк: {k} рук, {steps} шагов, {n_runs} запусков\n")

    results = {}
    for name, make_strategy in strategies_config:
        all_regrets = []
        all_rewards = []
        for run in range(n_runs):
            bandit = Bandit(k=k, seed=run)
            strategy = make_strategy()
            strategy.rng = random.Random(run)
            rewards, regrets = run_episode(bandit, strategy, steps)
            all_regrets.append(sum(regrets))
            all_rewards.append(sum(rewards[-500:]) / 500)
        results[name] = {
            "avg_regret": sum(all_regrets) / len(all_regrets),
            "avg_reward_last500": sum(all_rewards) / len(all_rewards),
        }

    print(f"{'Стратегия':<22} | {'Ср. Cumul. Regret':>18} | {'Ср. награда (посл. 500)':>24}")
    print("-" * 70)

    # Сортируем по регрету
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_regret"])
    for name, res in sorted_results:
        print(f"{name:<22} | {res['avg_regret']:>18.1f} | {res['avg_reward_last500']:>24.3f}")

    print()
    print("═══ ИТОГИ ═══")
    print()
    print("1. Epsilon-Greedy: простая, но ε нужно подбирать.")
    print("   - eps=0.01: мало exploration → может застрять.")
    print("   - eps=0.1: хороший компромисс.")
    print("   - eps=0.2: много exploration → расточительно.")
    print()
    print("2. UCB1: детерминированный exploration, не требует ε.")
    print("   Автоматически снижает exploration со временем.")
    print()
    print("3. Thompson Sampling: байесовский подход.")
    print("   - Самый адаптивный из трёх.")
    print("   - Не требует подбора гиперпараметров.")
    print("   - Обычно показывает лучший результат.")
    print()
    print("На практике Thompson Sampling и UCB1 — предпочтительные выборы.")
    print("Epsilon-Greedy — хорош как baseline и для понимания концепции.")


# ────────────────────────────────────────────────────────────────
# Точка входа
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MULTI-ARMED BANDITS: Explore vs Exploit            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    demo_epsilon_greedy()
    demo_ucb()
    demo_thompson()
    demo_comparison()

    print("\n" + "=" * 70)
    print("Все демо завершены.")
    print("=" * 70)
