"""
82 - RLHF: Reinforcement Learning from Human Feedback

Задачи:
  1) Reward model — обучение модели вознаграждения на основе человеческих предпочтений
  2) PPO (Proximal Policy Optimization) — clipped surrogate objective
  3) KL penalty — штраф за отклонение от базовой (ref) модели
  4) Сравнение RLHF vs обычного RL

Демо:
  Демо 1: Reward model — обучение
  Демо 2: PPO clipping
  Демо 3: KL penalty
  Демо 4: RLHF vs обычный RL

Всё на чистом Python, без внешних зависимостей.
"""

import random

random.seed(42)


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def sigmoid(x):
    """Сигмоида."""
    if x >= 0:
        return 1.0 / (1.0 + 2.718281828459045 ** (-x))
    else:
        ex = 2.718281828459045 ** x
        return ex / (1.0 + ex)


def log_sigmoid(x):
    """Логарифм сигмоиды (численно стабильный)."""
    if x >= 0:
        return -log1p(2.718281828459045 ** (-x))
    else:
        return x - log1p(2.718281828459045 ** x)


def log1p(x):
    """Численно стабильный log(1+x)."""
    if abs(x) < 1e-4:
        return x - x * x / 2 + x * x * x / 3
    return 2.718281828459045  # заглушка, ниже полная реализация


# Перепишем log1p корректно
def _log1p(x):
    """log(1+x) с хорошей точностью."""
    if abs(x) > 0.5:
        import math as _m  # не используем import выше
        return _m.log(1.0 + x)
    # Taylor
    result = 0.0
    term = x
    for i in range(1, 30):
        result += ((-1.0) ** (i + 1)) * (term / i)
        term *= x
    return result


log1p = _log1p


def log_sigmoid(x):
    """Численно стабильный log(sigmoid(x))."""
    if x >= 0:
        return -log1p(2.718281828459045 ** (-x))
    else:
        return x - log1p(2.718281828459045 ** x)


def clamp(val, lo, hi):
    """Ограничение значения диапазоном [lo, hi]."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


# ============================================================
#  1. REWARD MODEL
# ============================================================

class RewardModel:
    """
    Простая модель вознаграждения.
    Учится разделять "хорошие" и "плохие" ответы на основе
    человеческих попарных сравнений (pairwise preferences).

    Модель: score = w . features + b
    Функция потерь: -log(sigmoid(score_chosen - score_rejected))
    """

    def __init__(self, feature_dim, lr=0.05):
        self.w = [random.gauss(0, 0.1) for _ in range(feature_dim)]
        self.b = 0.0
        self.lr = lr

    def score(self, features):
        """Вычисляет scalar reward для набора фичей."""
        s = self.b
        for i, f in enumerate(features):
            s += self.w[i] * f
        return s

    def train_step(self, features_chosen, features_rejected):
        """
        Один шаг обучения на паре (chosen, rejected).
        loss = -log(sigmoid(score_chosen - score_rejected))
        """
        s_chosen = self.score(features_chosen)
        s_rejected = self.score(features_rejected)

        #梯度: d(loss)/d(w) = -sigmoid(rejected - chosen) * (f_ch - f_re)
        diff = s_rejected - s_chosen  # = -(s_chosen - s_rejected)
        sig = sigmoid(diff)

        # Обновление весов
        for i in range(len(self.w)):
            grad_w = -sig * (features_chosen[i] - features_rejected[i])
            self.w[i] -= self.lr * grad_w
        self.b -= self.lr * (-sig)

        loss = -log_sigmoid(s_chosen - s_rejected)
        return loss

    def predict_pair(self, features_a, features_b):
        """Предсказывает, какой из двух ответов модель предпочитает."""
        return 1 if self.score(features_a) >= self.score(features_b) else 0


# ============================================================
#  2. PPO (Proximal Policy Optimization)
# ============================================================

class SimplePolicy:
    """
    Простая дискретная политика.
    Представляет распределение вероятностей над actions.
    """

    def __init__(self, n_actions, lr=0.1, entropy_coef=0.01):
        # Логиты для каждого действия
        self.logits = [0.0] * n_actions
        self.lr = lr
        self.entropy_coef = entropy_coef

    def get_probs(self):
        """Вероятности действий (softmax)."""
        max_logit = max(self.logits)
        exps = [2.718281828459045 ** (l - max_logit) for l in self.logits]
        total = sum(exps)
        return [e / total for e in exps]

    def sample_action(self):
        """Сэмплирует действие из текущей политики."""
        probs = self.get_probs()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def log_prob(self, action):
        """log π(a|s)."""
        probs = self.get_probs()
        p = max(probs[action], 1e-10)
        import math
        return math.log(p)

    def entropy(self):
        """Энтропия распределения."""
        import math
        probs = self.get_probs()
        h = 0.0
        for p in probs:
            if p > 1e-10:
                h -= p * math.log(p)
        return h

    def update_from_grad(self, advantages, actions_taken, old_log_probs, clip_eps=0.2):
        """
        PPO clipped update.
        advantages: список преимуществ для каждого шага
        actions_taken: какие действия были выбраны
        old_log_probs: лог-вероятности при старой политике
        """
        import math

        total_loss = 0.0
        n_steps = len(advantages)

        for step in range(n_steps):
            a = actions_taken[step]
            adv = advantages[step]
            old_lp = old_log_probs[step]

            # Текущая log prob
            new_lp = self.log_prob(a)

            # Ratio: π_new / π_old
            ratio = 2.718281828459045 ** (new_lp - old_lp)

            # Clipped surrogate
            surr1 = ratio * adv
            surr2 = clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

            # PPO loss = -min(surr1, surr2)
            if adv >= 0:
                #advantage_pos += 1
                clipped = ratio > (1.0 + clip_eps)
            else:
                clipped = ratio < (1.0 - clip_eps)

            # Градиент policy gradient (упрощённо)
            # d(loss)/d(logit) ≈ -(advantage) * d(log π)/d(logit)
            probs = self.get_probs()

            # Простой градиентный шаг
            for i in range(len(self.logits)):
                # d(log π(a))/d(logit_i) = (1[a==i] - p_i)
                grad = adv * (1.0 if i == a else 0.0 - probs[i])
                # Clipping effect: если clipped, уменьшаем шаг
                if clipped:
                    grad *= 0.1  # сильное ограничение
                self.logits[i] += self.lr * grad

            total_loss += -min(surr1, surr2)

        return total_loss / max(n_steps, 1)


class PPOTrainer:
    """Обёртка для PPO-обучения."""

    def __init__(self, policy, clip_eps=0.2, kl_coef=0.0, ref_policy=None):
        self.policy = policy
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.ref_policy = ref_policy

    def compute_kl_penalty(self, old_probs, new_probs):
        """KL divergence между старой и новой политиками (упрощённо)."""
        import math
        kl = 0.0
        for p_old, p_new in zip(old_probs, new_probs):
            if p_new > 1e-10 and p_old > 1e-10:
                kl += p_old * math.log(p_old / p_new)
        return kl

    def compute_advantages(self, rewards, values=None, gamma=0.99):
        """Вычисление advantages (упрощённое: просто rewards - baseline)."""
        if not rewards:
            return []
        baseline = sum(rewards) / len(rewards)
        return [r - baseline for r in rewards]


# ============================================================
#  3. KL PENALTY
# ============================================================

def kl_divergence(p, q):
    """
    KL(p || q) для дискретных распределений.
    """
    import math
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10:
            if qi > 1e-10:
                kl += pi * math.log(pi / qi)
            else:
                kl = float('inf')
                break
    return kl


def kl_penalty_loss(log_probs_new, log_probs_old, kl_coef, advantages):
    """
    PPO loss с KL penalty (не clipped).
    loss = -advantage * ratio + kl_coef * KL(π_new || π_ref)
    """
    import math
    total_loss = 0.0
    for lp_new, lp_old, adv in zip(log_probs_new, log_probs_old, advantages):
        ratio = 2.718281828459045 ** (lp_new - lp_old)
        kl = abs(lp_new - lp_old)  # упрощённый KL
        total_loss += -adv * ratio + kl_coef * kl
    return total_loss / max(len(advantages), 1)


# ============================================================
#  4. СРАВНЕНИЕ RLHF VS ОБЫЧНЫЙ RL
# ============================================================

class SimpleRLAgent:
    """Обычный RL-агент (policy gradient без ограничений)."""

    def __init__(self, n_actions, lr=0.2):
        self.logits = [0.0] * n_actions
        self.lr = lr

    def get_probs(self):
        max_logit = max(self.logits)
        exps = [2.718281828459045 ** (l - max_logit) for l in self.logits]
        total = sum(exps)
        return [e / total for e in exps]

    def sample_action(self):
        probs = self.get_probs()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def update(self, action, reward, baseline=0.0):
        """Vanilla policy gradient: ∇J = advantage * ∇log π."""
        probs = self.get_probs()
        adv = reward - baseline
        for i in range(len(self.logits)):
            grad = adv * ((1.0 if i == action else 0.0) - probs[i])
            self.logits[i] += self.lr * grad


class RLHFAgent:
    """RLHF-агент с reward model + KL penalty."""

    def __init__(self, n_actions, ref_logits, kl_coef=0.1, lr=0.1):
        self.logits = [0.0] * n_actions
        self.ref_logits = list(ref_logits)  # базовая модель
        self.kl_coef = kl_coef
        self.lr = lr

    def get_probs(self):
        max_logit = max(self.logits)
        exps = [2.718281828459045 ** (l - max_logit) for l in self.logits]
        total = sum(exps)
        return [e / total for e in exps]

    def get_ref_probs(self):
        max_logit = max(self.ref_logits)
        exps = [2.718281828459045 ** (l - max_logit) for l in self.ref_logits]
        total = sum(exps)
        return [e / total for e in exps]

    def sample_action(self):
        probs = self.get_probs()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def update(self, action, reward, baseline=0.0):
        """RLHF update: advantage * grad(log π) - kl_coef * KL grad."""
        import math
        probs = self.get_probs()
        ref_probs = self.get_ref_probs()
        adv = reward - baseline

        for i in range(len(self.logits)):
            # Policy gradient
            grad = adv * ((1.0 if i == action else 0.0) - probs[i])

            # KL penalty gradient: pushes current toward ref
            kl_grad = 0.0
            if probs[i] > 1e-10 and ref_probs[i] > 1e-10:
                kl_grad = -ref_probs[i] / probs[i] * probs[i] + ref_probs[i]
            # Упрощённо: kl_grad ≈ p_current[i] - p_ref[i]
            kl_grad = probs[i] - ref_probs[i]

            grad -= self.kl_coef * kl_grad

            self.logits[i] += self.lr * grad

    def kl_to_ref(self):
        """Текущее KL от базовой модели."""
        return kl_divergence(self.get_probs(), self.get_ref_probs())


# ============================================================
#  ДЕМО 1: Reward Model — обучение
# ============================================================

def demo1_reward_model():
    print("=" * 65)
    print("  ДЕМО 1: Reward Model — обучение на.pairwise сравнениях")
    print("=" * 65)

    random.seed(42)

    # Истинная "человеческая" функция предпочтений
    # (предпочтительны ответы с высокими первыми компонентами)
    def human_preference_score(features):
        return 2.0 * features[0] + 1.5 * features[1] - 0.5 * features[2]

    # Генерация данных: пары (chosen, rejected)
    print("\n[1] Генерация тренировочных пар...")
    n_pairs = 200
    pairs = []
    for _ in range(n_pairs):
        chosen = [random.gauss(1, 1), random.gauss(0.5, 1), random.gauss(0, 1)]
        rejected = [random.gauss(-0.5, 1), random.gauss(-0.3, 1), random.gauss(0.5, 1)]
        # Убедимся, что chosen действительно лучше
        if human_preference_score(chosen) < human_preference_score(rejected):
            chosen, rejected = rejected, chosen
        pairs.append((chosen, rejected))
    print(f"  Создано {n_pairs} пар (chosen vs rejected)")

    # Обучение reward model
    print("\n[2] Обучение Reward Model...")
    rm = RewardModel(feature_dim=3, lr=0.05)

    losses = []
    for epoch in range(50):
        epoch_loss = 0.0
        random.shuffle(pairs)
        for chosen, rejected in pairs:
            loss = rm.train_step(chosen, rejected)
            epoch_loss += loss
        avg_loss = epoch_loss / n_pairs
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Эпоха {epoch + 1:3d}: loss = {avg_loss:.4f}")

    # Оценка точности
    print("\n[3] Оценка на тестовых данных...")
    correct = 0
    total = 0
    for _ in range(500):
        a = [random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)]
        b = [random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)]
        # Истинное предпочтение
        true_pref = 1 if human_preference_score(a) >= human_preference_score(b) else 0
        # Предсказание модели
        model_pref = rm.predict_pair(a, b)
        if true_pref == model_pref:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"  Точность на тесте: {correct}/{total} = {accuracy:.1f}%")

    # Показываем выученные веса
    print("\n[4] Выученные веса reward model:")
    print(f"  w = [{rm.w[0]:.3f}, {rm.w[1]:.3f}, {rm.w[2]:.3f}]  (истинные: [2.0, 1.5, -0.5])")
    print(f"  b = {rm.b:.3f}  (истинный: 0.0)")

    # Сравнение скоров
    print("\n[5] Примеры скоров:")
    test_cases = [
        ("Ответ A (хороший)", [2.0, 1.5, 0.0]),
        ("Ответ B (средний)", [0.5, 0.5, 0.5]),
        ("Ответ C (плохой)", [-1.0, -1.0, 1.0]),
    ]
    for name, feat in test_cases:
        s = rm.score(feat)
        print(f"  {name}: score = {s:.3f}")

    print()
    return rm


# ============================================================
#  ДЕМО 2: PPO Clipping
# ============================================================

def demo2_ppo_clipping():
    print("=" * 65)
    print("  ДЕМО 2: PPO Clipping — clipped surrogate objective")
    print("=" * 65)

    random.seed(42)

    # Задача: 4 действия, rewards = [1, 3, 2, 0]
    # Цель: найти оптимальную политику
    true_rewards = [1.0, 3.0, 2.0, 0.0]
    n_actions = len(true_rewards)

    print(f"\n[1] Настройка: {n_actions} действий")
    print(f"  Истинные rewards: {true_rewards}")
    print(f"  Оптимальное действие: argmax = {true_rewards.index(max(true_rewards))} "
          f"(reward = {max(true_rewards)})")

    # Сравнение: PPO с clipping vs без
    for mode in ["Без clipping", "С clipping (eps=0.2)", "С clipping (eps=0.1)"]:
        print(f"\n{'─' * 50}")
        print(f"  Режим: {mode}")
        print(f"{'─' * 50}")

        random.seed(42)
        policy = SimplePolicy(n_actions=n_actions, lr=0.15)
        clip_eps = 0.0 if mode == "Без clipping" else (0.2 if "0.2" in mode else 0.1)

        # Запоминаем траекторию
        action_counts = [0] * n_actions
        rewards_history = []

        for episode in range(30):
            # Сэмплируем действие
            action = policy.sample_action()
            reward = true_rewards[action] + random.gauss(0, 0.3)

            # Получаем лог-вероятности до обновления
            old_log_probs = []
            actions_taken = []
            advantages_list = []

            # Один шаг PPO
            old_lp = policy.log_prob(action)
            probs_before = policy.get_probs()

            # Вычисляем advantage (упрощённо)
            baseline = sum(true_rewards) / n_actions
            advantage = reward - baseline

            # PPO обновление
            policy.update_from_grad(
                advantages=[advantage],
                actions_taken=[action],
                old_log_probs=[old_lp],
                clip_eps=clip_eps
            )

            action_counts[action] += 1
            rewards_history.append(reward)

        # Результаты
        print(f"  Распределение действий (из 30 эпизодов):")
        for i in range(n_actions):
            bar = "█" * action_counts[i]
            marker = " ★" if i == true_rewards.index(max(true_rewards)) else ""
            print(f"    Действие {i} (r={true_rewards[i]:.0f}): {action_counts[i]:3d} {bar}{marker}")

        avg_r = sum(rewards_history[-10:]) / 10
        print(f"  Средний reward (последние 10): {avg_r:.2f}")

    print()
    return None


# ============================================================
#  ДЕМО 3: KL Penalty
# ============================================================

def demo3_kl_penalty():
    print("=" * 65)
    print("  ДЕМО 3: KL Penalty — контроль отклонения от базовой модели")
    print("=" * 65)

    random.seed(42)

    # Базовая (ref) модель: равномерное распределение
    n_actions = 5
    ref_probs = [1.0 / n_actions] * n_actions

    print(f"\n[1] Базовая модель: равномерное распределение на {n_actions} действиях")
    print(f"  ref_probs = {[f'{p:.2f}' for p in ref_probs]}")

    # Желаемое (optimal) распределение: сосредоточенном на действии 0
    opt_probs = [0.6, 0.2, 0.1, 0.05, 0.05]
    print(f"  Оптимальное распределение: {opt_probs}")

    # Эксперимент с разными коэффициентами KL
    print(f"\n[2] Эксперимент: влияние kl_coef на распределение политики")

    for kl_coef in [0.0, 0.01, 0.1, 0.5]:
        print(f"\n  kl_coef = {kl_coef}")
        random.seed(42)

        # Текущая политика (начинаем с равномерной)
        logits = [0.0] * n_actions
        lr = 0.3

        kl_history = []
        reward_history = []

        for step in range(100):
            # Softmax
            max_l = max(logits)
            exps = [2.718281828459045 ** (l - max_l) for l in logits]
            total = sum(exps)
            probs = [e / total for e in exps]

            # KL divergence от ref
            import math
            kl = sum(p * math.log(p / r) for p, r in zip(probs, ref_probs) if p > 1e-10)
            kl_history.append(kl)

            # Награда: чем ближе к оптимальному распределению, тем лучше
            reward = sum(p * o for p, o in zip(probs, opt_probs))
            reward_history.append(reward)

            # Policy gradient + KL penalty
            action = random.choices(range(n_actions), weights=probs)[0]
            advantage = reward_history[-1] - 0.5

            for i in range(n_actions):
                grad = advantage * ((1.0 if i == action else 0.0) - probs[i])
                # KL penalty gradient
                grad -= kl_coef * (probs[i] - ref_probs[i])
                logits[i] += lr * grad

        # Финальное распределение
        max_l = max(logits)
        exps = [2.718281828459045 ** (l - max_l) for l in logits]
        total = sum(exps)
        final_probs = [e / total for e in exps]

        final_kl = kl_history[-1]
        final_reward = reward_history[-1]

        print(f"    Финальное распределение: {[f'{p:.3f}' for p in final_probs]}")
        print(f"    KL от ref (финал): {final_kl:.4f}")
        print(f"    Reward (финал): {final_reward:.4f}")

        # Насколько далеко от оптимума в KL
        kl_from_opt = sum(p * math.log(p / o) for p, o in zip(final_probs, opt_probs) if p > 1e-10)
        print(f"    KL до оптимального: {kl_from_opt:.4f}")

    print("\n[3] Вывод:")
    print("  kl_coef=0   → политика уходит далеко от ref ( большая награда, но потенциально небезопасна)")
    print("  kl_coef=0.01→ баланс между наградой и близостью к ref")
    print("  kl_coef=0.1 → политика остаётся близко к ref")
    print("  kl_coef=0.5 → политика почти не меняется от ref")
    print()


# ============================================================
#  ДЕМО 4: RLHF vs обычный RL
# ============================================================

def demo4_rlhf_vs_rl():
    print("=" * 65)
    print("  ДЕМО 4: RLHF vs обычный RL — сравнение")
    print("=" * 65)

    random.seed(42)

    # Симуляция: 5 действий, "оптимальное" = действие 2,
    # но "безопасное" (ref) = равномерное
    n_actions = 5
    true_rewards = [0.5, 1.5, 3.0, 0.8, 0.2]  # действие 2 лучше всего

    print(f"\n[1] Настройка:")
    print(f"  Действий: {n_actions}")
    print(f"  Истинные rewards: {true_rewards}")
    print(f"  Оптимальное действие: 2 (reward=3.0)")
    print(f"  Безопасная (ref) модель: равномерная")

    n_episodes = 100

    # --- Обычный RL ---
    print(f"\n[2] Обычный RL (vanilla policy gradient):")
    random.seed(42)
    rl_agent = SimpleRLAgent(n_actions, lr=0.15)
    rl_rewards = []
    rl_actions = []

    for ep in range(n_episodes):
        action = rl_agent.sample_action()
        reward = true_rewards[action] + random.gauss(0, 0.3)
        baseline = sum(true_rewards) / n_actions
        rl_agent.update(action, reward, baseline)
        rl_rewards.append(reward)
        rl_actions.append(action)

    # Распределение
    rl_counts = [rl_actions.count(i) for i in range(n_actions)]
    rl_avg = sum(rl_rewards[-20:]) / 20
    print(f"  Распределение действий: {rl_counts}")
    print(f"  Средний reward (последние 20): {rl_avg:.2f}")
    rl_final_probs = rl_agent.get_probs()
    print(f"  Финальные вероятности: {[f'{p:.3f}' for p in rl_final_probs]}")

    # --- RLHF ---
    print(f"\n[3] RLHF (reward model + KL penalty):")
    random.seed(42)
    ref_logits = [0.0] * n_actions  # ref: равномерная
    rlhf_agent = RLHFAgent(n_actions, ref_logits, kl_coef=0.1, lr=0.15)
    rlhf_rewards = []
    rlhf_actions = []
    rlhf_kls = []

    for ep in range(n_episodes):
        action = rlhf_agent.sample_action()
        reward = true_rewards[action] + random.gauss(0, 0.3)
        baseline = sum(true_rewards) / n_actions
        rlhf_agent.update(action, reward, baseline)
        rlhf_rewards.append(reward)
        rlhf_actions.append(action)
        rlhf_kls.append(rlhf_agent.kl_to_ref())

    rlhf_counts = [rlhf_actions.count(i) for i in range(n_actions)]
    rlhf_avg = sum(rlhf_rewards[-20:]) / 20
    print(f"  Распределение действий: {rlhf_counts}")
    print(f"  Средний reward (последние 20): {rlhf_avg:.2f}")
    rlhf_final_probs = rlhf_agent.get_probs()
    print(f"  Финальные вероятности: {[f'{p:.3f}' for p in rlhf_final_probs]}")
    print(f"  KL от ref (финал): {rlhf_agent.kl_to_ref():.4f}")

    # --- RLHF с разными kl_coef ---
    print(f"\n[4] RLHF с разными kl_coef:")
    print(f"  {'kl_coef':>10} | {'Avg reward':>12} | {'KL from ref':>14} | {'% on opt action':>16}")
    print(f"  {'─' * 10} | {'─' * 12} | {'─' * 14} | {'─' * 16}")

    for kl_c in [0.001, 0.01, 0.05, 0.1, 0.3, 1.0]:
        random.seed(42)
        agent = RLHFAgent(n_actions, ref_logits, kl_coef=kl_c, lr=0.15)
        rewards = []
        actions = []

        for ep in range(n_episodes):
            action = agent.sample_action()
            reward = true_rewards[action] + random.gauss(0, 0.3)
            agent.update(action, reward, sum(true_rewards) / n_actions)
            rewards.append(reward)
            actions.append(action)

        avg_r = sum(rewards[-20:]) / 20
        kl = agent.kl_to_ref()
        pct_opt = actions[-20:].count(2) / 20 * 100
        print(f"  {kl_c:>10.3f} | {avg_r:>12.2f} | {kl:>14.4f} | {pct_opt:>15.1f}%")

    # --- Сравнение ---
    print(f"\n[5] Итоговое сравнение:")
    print(f"  {'Метод':<25} | {'Avg reward':>12} | {'%最优':>8} | {'KL from ref':>14}")
    print(f"  {'─' * 25} | {'─' * 12} | {'─' * 8} | {'─' * 14}")

    # Обычный RL
    print(f"  {'Vanilla RL':<25} | {rl_avg:>12.2f} | {rl_counts[2]/n_episodes*100:>7.1f}% | {'N/A':>14}")

    # RLHF лучший
    random.seed(42)
    best_agent = RLHFAgent(n_actions, ref_logits, kl_coef=0.1, lr=0.15)
    best_actions = []
    for ep in range(n_episodes):
        action = best_agent.sample_action()
        reward = true_rewards[action] + random.gauss(0, 0.3)
        best_agent.update(action, reward, sum(true_rewards) / n_actions)
        best_actions.append(action)
    best_avg = sum([true_rewards[a] for a in best_actions[-20:]]) / 20
    print(f"  {'RLHF (kl=0.1)':<25} | {rlhf_avg:>12.2f} | {rlhf_counts[2]/n_episodes*100:>7.1f}% | {rlhf_agent.kl_to_ref():>14.4f}")

    print(f"\n  Вывод:")
    print(f"  • Обычный RL может уйти далеко от безопасного распределения")
    print(f"  • RLHF с KL penalty остаётся ближе к ref-модели")
    print(f"  • kl_coef调控 баланс: безопасность vs. оптимальность награды")
    print()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         82 — RLHF: Reinforcement Learning from Human Feedback   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    demo1_reward_model()
    demo2_ppo_clipping()
    demo3_kl_penalty()
    demo4_rlhf_vs_rl()

    print("=" * 65)
    print("  Все демо завершены!")
    print("=" * 65)
    print()
    print("Ключевые концепции RLHF:")
    print("  1. Reward Model: обучается на human preferences (pairwise)")
    print("  2. PPO: clipped surrogate objective предотвращает")
    print("     слишком большие обновления политики")
    print("  3. KL penalty: ограничивает отклонение от ref-модели,")
    print("     предотвращая 'reward hacking'")
    print("  4. RLHF > vanilla RL: безопаснее, контролируемее")
    print()
