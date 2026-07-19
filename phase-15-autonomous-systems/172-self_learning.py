"""172 — Self-Learning Systems: онлайн-обучение, накопление опыта

Темы:
  1. Online Learning (incremental updates, experience replay buffer)
  2. Experience Accumulation (trajectory storage, pattern extraction, generalization)
  3. Reward Shaping (sparse vs dense rewards, intrinsic motivation)
  4. Curriculum Learning (difficulty progression, self-paced learning)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# ============================================================
# Демо 1: Онлайн-обучение
# ============================================================

def demo_online_learning():
    """
    Демонстрация онлайн-обучения:
    - Инкрементные обновления параметров
    - Experience Replay Buffer
    - Сравнение с пакетным обучением
    """
    print("=" * 70)
    print("ДЕМО 1: ОНЛАЙН-ОБУЧЕНИЕ")
    print("=" * 70)

    # Подзадача 1: Линейная регрессия с инкрементным обновлением
    print("\n--- Подзадача 1: Инкрементное обучение (SGD) ---\n")

    class OnlineLinearRegressor:
        """
        Линейная регрессия с онайн-обновлением.
        Формула обновления: w = w - lr * (prediction - target) * x
        Это стохастический градиентный спуск (SGD).
        """

        def __init__(self, n_features: int, learning_rate: float = 0.01):
            self.weights = [0.0] * n_features
            self.bias = 0.0
            self.lr = learning_rate
            self.update_count = 0

        def predict(self, x: list) -> float:
            """Предсказание: y = w·x + b"""
            return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

        def update(self, x: list, target: float):
            """
            Одно обновление по SGD:
            error = prediction - target
            w_i = w_i - lr * error * x_i
            bias = bias - lr * error
            """
            prediction = self.predict(x)
            error = prediction - target

            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * error * x[i]
            self.bias -= self.lr * error
            self.update_count += 1

        def mse(self, X: list, y: list) -> float:
            """Среднеквадратичная ошибка."""
            predictions = [self.predict(xi) for xi in X]
            return sum((p - t) ** 2 for p, t in zip(predictions, y)) / len(y)

    # Генерация данных: y = 3*x1 + 2*x2 - 1*x3 + noise
    random.seed(42)
    true_weights = [3.0, 2.0, -1.0]
    true_bias = 5.0

    def generate_data(n: int):
        X = []
        y = []
        for _ in range(n):
            x = [random.uniform(-10, 10) for _ in range(3)]
            noise = random.gauss(0, 0.5)
            target = sum(w * xi for w, xi in zip(true_weights, x)) + true_bias + noise
            X.append(x)
            y.append(target)
        return X, y

    # Пакетное обучение (для сравнения)
    X_train, y_train = generate_data(200)
    X_test, y_test = generate_data(50)

    # Online learning — по одному примеру за раз
    online_model = OnlineLinearRegressor(3, learning_rate=0.001)
    online_errors = []

    for i, (x, target) in enumerate(zip(X_train, y_train)):
        online_model.update(x, target)
        if (i + 1) % 20 == 0:
            mse = online_model.mse(X_test, y_test)
            online_errors.append((i + 1, mse))

    print("Истинные параметры: w=[3.0, 2.0, -1.0], bias=5.0")
    print(f"Обученные параметры: w=[{', '.join(f'{w:.3f}' for w in online_model.weights)}], bias={online_model.bias:.3f}")
    print(f"Количество обновлений: {online_model.update_count}")
    print(f"\nСходимость MSE (каждые 20 шагов):")
    print(f"{'Шаг':<10} {'MSE':<12} {'График'}")
    print("-" * 50)
    for step, mse in online_errors:
        bar = "█" * int(max(0, 1 - mse / 10) * 30)
        print(f"{step:<10} {mse:<12.4f} {bar}")

    # Подзадача 2: Experience Replay Buffer
    print("\n--- Подзадача 2: Experience Replay Buffer ---\n")

    class ReplayBuffer:
        """
        Буфер воспроизведения опыта для offline-обучения.
        Хранит кортежи (state, action, reward, next_state, done).
        При обучении случайно выбирает мини-пакеты.
        """

        def __init__(self, capacity: int = 1000):
            self.buffer = []
            self.capacity = capacity
            self.position = 0

        def push(self, state, action, reward, next_state, done: bool):
            """Добавление опыта в буфер."""
            experience = (state, action, reward, next_state, done)
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                # Перезапись старых записей (циклический буфер)
                self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size: int) -> list:
            """Случайная выборка опыта из буфера."""
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))

        def __len__(self) -> int:
            return len(self.buffer)

        def get_statistics(self) -> dict:
            """Статистика буфера."""
            if not self.buffer:
                return {"size": 0}

            rewards = [e[2] for e in self.buffer]
            dones = [e[4] for e in self.buffer]
            return {
                "size": len(self.buffer),
                "avg_reward": sum(rewards) / len(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "terminal_ratio": sum(dones) / len(dones),
            }

    # Симуляция опыта агента
    buffer = ReplayBuffer(capacity=500)

    # Симуляция 200 шагов взаимодействия со средой
    for step in range(200):
        state = random.uniform(-1, 1)
        action = random.choice(["left", "right", "stay"])

        # Простая среда: награда зависит от действия и состояния
        if action == "right" and state > 0:
            reward = random.uniform(0.5, 1.5)
        elif action == "left" and state < 0:
            reward = random.uniform(0.5, 1.5)
        elif action == "stay":
            reward = random.uniform(-0.5, 0.5)
        else:
            reward = random.uniform(-1.5, -0.5)

        next_state = state + random.gauss(0, 0.1)
        done = abs(next_state) > 2.0

        buffer.push(state, action, reward, next_state, done)

    stats = buffer.get_statistics()
    print(f"Буфер опыта (ёмкость 500):")
    print(f"  Размер: {stats['size']}")
    print(f"  Средняя награда: {stats['avg_reward']:.4f}")
    print(f"  Диапазон наград: [{stats['min_reward']:.4f}, {stats['max_reward']:.4f}]")
    print(f"  Доля завершающих шагов: {stats['terminal_ratio']:.2%}")

    # Демонстрация семплирования
    print(f"\nСемплирование мини-пакета (размер 5):")
    batch = buffer.sample(5)
    for i, (s, a, r, ns, d) in enumerate(batch):
        print(f"  {i+1}. state={s:.3f}, action={a}, reward={r:.3f}, next={ns:.3f}, done={d}")

    # Подзадача 3: Сравнение стратегий семплирования
    print("\n--- Подзадача 3: Стратегии семплирования из буфера ---\n")

    class PrioritizedReplayBuffer:
        """
        Буфер с приоритетным семплированием.
        Опыт с большей ошибкой обучения выбирается чаще.
        """

        def __init__(self, capacity: int = 500, alpha: float = 0.6):
            self.buffer = []
            self.capacity = capacity
            self.priorities = []
            self.alpha = alpha  # Степень приоритета

        def push(self, state, action, reward, next_state, done: bool, td_error: float = 1.0):
            """Добавление опыта с приоритетом."""
            priority = abs(td_error) ** self.alpha
            if len(self.buffer) < self.capacity:
                self.buffer.append((state, action, reward, next_state, done))
                self.priorities.append(priority)
            else:
                # Заменяем наименее приоритетный
                min_idx = self.priorities.index(min(self.priorities))
                self.buffer[min_idx] = (state, action, reward, next_state, done)
                self.priorities[min_idx] = priority

        def sample(self, batch_size: int) -> list:
            """Семплирование с учётом приоритетов."""
            total = sum(self.priorities)
            probabilities = [p / total for p in self.priorities]

            indices = []
            for _ in range(min(batch_size, len(self.buffer))):
                r = random.random()
                cumulative = 0
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        indices.append(i)
                        break

            return [self.buffer[i] for i in indices]

    # Заполняем оба буфера одними и теми же данными
    uniform_buffer = ReplayBuffer(capacity=200)
    prioritized_buffer = PrioritizedReplayBuffer(capacity=200, alpha=0.7)

    for step in range(200):
        state = random.uniform(-1, 1)
        action = random.choice(["left", "right", "stay"])
        if action == "right" and state > 0:
            reward = random.uniform(0.5, 1.5)
        elif action == "left" and state < 0:
            reward = random.uniform(0.5, 1.5)
        else:
            reward = random.uniform(-1.0, 1.0)
        next_state = state + random.gauss(0, 0.1)
        done = abs(next_state) > 2.0
        td_error = abs(reward - random.uniform(0, 1))

        uniform_buffer.push(state, action, reward, next_state, done)
        prioritized_buffer.push(state, action, reward, next_state, done, td_error)

    # Сравнение распределений наград в семплированных батчах
    print("Сравнение стратегий семплирования (1000 батчей по 32):")
    print(f"\n{'Стратегия':<25} {'Средняя награда':<20} {'Std награды':<15} {'Разнообразие'}")
    print("-" * 75)

    for name, buf in [("Uniform", uniform_buffer), ("Prioritized", prioritized_buffer)]:
        batch_rewards = []
        for _ in range(1000):
            batch = buf.sample(32)
            batch_reward = sum(e[2] for e in batch) / len(batch)
            batch_rewards.append(batch_reward)

        avg = sum(batch_rewards) / len(batch_rewards)
        std = math.sqrt(sum((r - avg) ** 2 for r in batch_rewards) / len(batch_rewards))
        unique_actions = len(set(e[1] for e in buf.buffer))
        print(f"{name:<25} {avg:<20.4f} {std:<15.4f} {unique_actions} уникальных действий")

    # Подзадача 4: Адаптивная скорость обучения
    print("\n--- Подзадача 4: Адаптивная скорость обучения ---\n")

    class AdaptiveLearningRate:
        """
        Адаптивная скорость обучения.
        Формула Adam: m = beta1*m + (1-beta1)*grad
                      v = beta2*v + (1-beta2)*grad^2
                      w = w - lr * m_hat / (sqrt(v_hat) + eps)
        """

        def __init__(self, n_features: int, lr: float = 0.001):
            self.weights = [0.0] * n_features
            self.bias = 0.0
            self.lr = lr
            self.m = [0.0] * n_features  # Первый момент
            self.v = [0.0] * n_features  # Второй момент
            self.mb = 0.0
            self.vb = 0.0
            self.t = 0  # Временной шаг
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-8

        def predict(self, x: list) -> float:
            return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

        def update(self, x: list, target: float):
            """Обновление с Adam-оптимизатором."""
            self.t += 1
            prediction = self.predict(x)
            error = prediction - target
            grad = [error * xi for xi in x]

            for i in range(len(self.weights)):
                # Обновление моментов
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad[i] ** 2

                # Коррекция смещения
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Обновление весов
                self.weights[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            # Обновление bias
            self.mb = self.beta1 * self.mb + (1 - self.beta1) * error
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * error ** 2
            mb_hat = self.mb / (1 - self.beta1 ** self.t)
            vb_hat = self.vb / (1 - self.beta2 ** self.t)
            self.bias -= self.lr * mb_hat / (math.sqrt(vb_hat) + self.eps)

    # Сравнение SGD и Adam
    X, y = generate_data(300)

    sgd_model = OnlineLinearRegressor(3, learning_rate=0.001)
    adam_model = AdaptiveLearningRate(3, lr=0.001)

    sgd_errors = []
    adam_errors = []

    for i in range(len(X)):
        sgd_model.update(X[i], y[i])
        adam_model.update(X[i], y[i])

        if (i + 1) % 30 == 0:
            sgd_mse = sgd_model.mse(X_test, y_test)
            adam_preds = [adam_model.predict(xi) for xi in X_test]
            adam_mse = sum((p - t) ** 2 for p, t in zip(adam_preds, y_test)) / len(y_test)
            sgd_errors.append(sgd_mse)
            adam_errors.append(adam_mse)

    print("Сравнение SGD vs Adam (MSE по итерациям):")
    print(f"{'Итерация':<12} {'SGD':<15} {'Adam':<15} {'Лучше'}")
    print("-" * 55)
    for i in range(len(sgd_errors)):
        step = (i + 1) * 30
        winner = "SGD" if sgd_errors[i] < adam_errors[i] else "Adam"
        print(f"{step:<12} {sgd_errors[i]:<15.4f} {adam_errors[i]:<15.4f} {winner}")

    print(f"\nИтоговые веса:")
    print(f"  SGD:  w=[{', '.join(f'{w:.3f}' for w in sgd_model.weights)}], bias={sgd_model.bias:.3f}")
    print(f"  Adam: w=[{', '.join(f'{w:.3f}' for w in adam_model.weights)}], bias={adam_model.bias:.3f}")
    print(f"  Истинные: w=[3.0, 2.0, -1.0], bias=5.0")

    print()


# ============================================================
# Демо 2: Накопление опыта
# ============================================================

def demo_experience_accumulation():
    """
    Демонстрация накопления опыта:
    - Хранение траекторий
    - Извлечение паттернов
    - Обобщение опыта
    """
    print("=" * 70)
    print("ДЕМО 2: НАКОПЛЕНИЕ ОПЫТА")
    print("=" * 70)

    # Подзадача 1: Хранение траекторий
    print("\n--- Подзадача 1: Хранение траекторий ---\n")

    class TrajectoryStorage:
        """Хранилище траекторий — последовательностей состояний и действий."""

        def __init__(self):
            self.trajectories = []

        def add_trajectory(self, states: list, actions: list, rewards: list):
            """Добавление траектории."""
            assert len(states) == len(actions) + 1, f"states={len(states)}, actions={len(actions)}, rewards={len(rewards)}"
            self.trajectories.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "total_reward": sum(rewards),
                "length": len(actions),
            })

        def get_best_trajectories(self, n: int = 5) -> list:
            """Получение N лучших траекторий по суммарной награде."""
            sorted_trajs = sorted(self.trajectories, key=lambda t: t["total_reward"], reverse=True)
            return sorted_trajs[:n]

        def get_statistics(self) -> dict:
            """Статистика по всем траекториям."""
            if not self.trajectories:
                return {}

            total_rewards = [t["total_reward"] for t in self.trajectories]
            lengths = [t["length"] for t in self.trajectories]
            return {
                "count": len(self.trajectories),
                "avg_reward": sum(total_rewards) / len(total_rewards),
                "max_reward": max(total_rewards),
                "min_reward": min(total_rewards),
                "avg_length": sum(lengths) / len(lengths),
            }

    storage = TrajectoryStorage()

    # Симуляция 50 траекторий
    for traj_idx in range(50):
        length = random.randint(5, 20)
        states = [random.uniform(-1, 1) for _ in range(length + 1)]
        actions = [random.choice(["left", "right", "up", "down"]) for _ in range(length)]
        rewards = [random.gauss(0.5, 0.3) for _ in range(length)]
        storage.add_trajectory(states, actions, rewards)

    stats = storage.get_statistics()
    print("Статистика хранилища траекторий:")
    print(f"  Количество: {stats['count']}")
    print(f"  Средняя награда: {stats['avg_reward']:.4f}")
    print(f"  Макс. награда: {stats['max_reward']:.4f}")
    print(f"  Мин. награда: {stats['min_reward']:.4f}")
    print(f"  Средняя длина: {stats['avg_length']:.1f}")

    print("\nЛучшие 3 траектории:")
    best = storage.get_best_trajectories(3)
    for i, traj in enumerate(best):
        print(f"  #{i+1}: награда={traj['total_reward']:.4f}, длина={traj['length']}, "
              f"действия={traj['actions'][:5]}{'...' if traj['length'] > 5 else ''}")

    # Подзадача 2: Извлечение паттернов
    print("\n--- Подзадача 2: Извлечение паттернов ---\n")

    class PatternExtractor:
        """Извлечение паттернов из последовательностей действий."""

        def __init__(self, min_support: float = 0.1):
            self.min_support = min_support

        def find_ngrams(self, sequences: list, n: int) -> dict:
            """Поиск n-грамм в последовательностях."""
            ngram_counts = collections.Counter()
            total_sequences = len(sequences)

            for seq in sequences:
                seen = set()
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i + n])
                    if ngram not in seen:
                        ngram_counts[ngram] += 1
                        seen.add(ngram)

            # Фильтрация по минимальной поддержке
            frequent = {
                ngram: count / total_sequences
                for ngram, count in ngram_counts.items()
                if count / total_sequences >= self.min_support
            }
            return dict(sorted(frequent.items(), key=lambda x: x[1], reverse=True))

        def find_recurring_patterns(self, sequences: list) -> list:
            """Поиск повторяющихся паттернов (подпоследовательности)."""
            all_patterns = collections.Counter()

            for seq in sequences:
                # Ищем все подпоследовательности длины 2-5
                for n in range(2, min(6, len(seq) + 1)):
                    for i in range(len(seq) - n + 1):
                        pattern = tuple(seq[i:i + n])
                        all_patterns[pattern] += 1

            # Фильтруем по частоте
            common = [(p, c) for p, c in all_patterns.most_common(10)]
            return common

    # Собираем данные из хранилища
    action_sequences = [t["actions"] for t in storage.trajectories]
    extractor = PatternExtractor(min_support=0.1)

    # Поиск биграмм
    bigrams = extractor.find_ngrams(action_sequences, 2)
    print("Топ-10 биграмм действий (поддержка >= 10%):")
    print(f"{'Паттерн':<25} {'Поддержка':<12} {'График'}")
    print("-" * 55)
    for pattern, support in list(bigrams.items())[:10]:
        bar = "█" * int(support * 50)
        print(f"{str(pattern):<25} {support:<12.3f} {bar}")

    # Поиск повторяющихся паттернов
    patterns = extractor.find_recurring_patterns(action_sequences)
    print(f"\nТоп-5 повторяющихся паттернов:")
    for pattern, count in patterns[:5]:
        print(f"  {pattern}: встречается {count} раз")

    # Подзадача 3: Обобщение опыта
    print("\n--- Подзадача 3: Обобщение опыта ---\n")

    class ExperienceGeneralizer:
        """Обобщение опыта — извлечение общих правил из конкретных примеров."""

        def __init__(self):
            self.state_action_rewards = []  # (state_feature, action, reward)

        def add_experience(self, state_features: dict, action: str, reward: float):
            self.state_action_rewards.append((state_features, action, reward))

        def compute_rules(self) -> dict:
            """
            Вычисление правил вида: если состояние特征 X > порог, то действие Y.
            Метрика: information gain по награде.
            """
            rules = {}

            # Группируем по действиям
            by_action = collections.defaultdict(list)
            for features, action, reward in self.state_action_rewards:
                by_action[action].append((features, reward))

            # Для каждого действия ищем лучший порог
            for action, experiences in by_action.items():
                if len(experiences) < 5:
                    continue

                # Смотрим на числовые特征
                for feature_name in experiences[0][0].keys():
                    values = [(exp[0][feature_name], exp[1]) for exp in experiences]
                    if not all(isinstance(v, (int, float)) for v, _ in values):
                        continue

                    # Пробуем разные пороги
                    feature_values = sorted(set(v for v, _ in values))
                    best_threshold = None
                    best_gain = -1

                    for i in range(len(feature_values) - 1):
                        threshold = (feature_values[i] + feature_values[i + 1]) / 2

                        # Information gain
                        left_rewards = [r for v, r in values if v <= threshold]
                        right_rewards = [r for v, r in values if v > threshold]

                        if not left_rewards or not right_rewards:
                            continue

                        # Энтропия до разделения
                        all_rewards = [r for _, r in values]
                        mean_all = sum(all_rewards) / len(all_rewards)
                        var_all = sum((r - mean_all) ** 2 for r in all_rewards) / len(all_rewards)

                        # Энтропия после
                        mean_left = sum(left_rewards) / len(left_rewards)
                        var_left = sum((r - mean_left) ** 2 for r in left_rewards) / len(left_rewards)
                        mean_right = sum(right_rewards) / len(right_rewards)
                        var_right = sum((r - mean_right) ** 2 for r in right_rewards) / len(right_rewards)

                        # Взвешенное уменьшение дисперсии
                        n = len(values)
                        gain = var_all - (len(left_rewards) / n * var_left + len(right_rewards) / n * var_right)

                        if gain > best_gain:
                            best_gain = gain
                            best_threshold = threshold

                    if best_threshold is not None and best_gain > 0.01:
                        rules[(action, feature_name)] = {
                            "threshold": best_threshold,
                            "gain": best_gain,
                            "direction": ">",
                        }

            return rules

    generalizer = ExperienceGeneralizer()

    # Генерация опыта с известными зависимостями
    for _ in range(500):
        temp = random.uniform(0, 100)
        humidity = random.uniform(0, 100)
        light = random.uniform(0, 100)

        # Правила: если temp > 70, включить охлаждение
        # если humidity < 30, увлажнить
        if temp > 70:
            action = "охладить"
            reward = 1.0 + random.gauss(0, 0.1)
        elif humidity < 30:
            action = "увлажнить"
            reward = 0.8 + random.gauss(0, 0.1)
        elif light < 40:
            action = "осветить"
            reward = 0.6 + random.gauss(0, 0.1)
        else:
            action = "ничего"
            reward = 0.2 + random.gauss(0, 0.1)

        generalizer.add_experience({"temp": temp, "humidity": humidity, "light": light}, action, reward)

    rules = generalizer.compute_rules()
    print("Обнаруженные правила:")
    for (action, feature), info in rules.items():
        print(f"  Если {feature} {info['direction']} {info['threshold']:.1f} → {action} "
              f"(information gain: {info['gain']:.4f})")

    # Подзадача 4: Кластеризация опыта
    print("\n--- Подзадача 4: Кластеризация опыта ---\n")

    def simple_kmeans(data: list, k: int, max_iters: int = 50) -> tuple:
        """
        Простой K-means без внешних библиотек.
        Возвращает (центроиды,.labels).
        """
        n = len(data)
        dim = len(data[0])

        # Инициализация центроидов случайными точками
        centroids = [list(data[i]) for i in random.sample(range(n), k)]
        labels = [0] * n

        for iteration in range(max_iters):
            # Назначение точек ближайшему центроиду
            changed = False
            for i, point in enumerate(data):
                min_dist = float("inf")
                best_cluster = 0
                for j, centroid in enumerate(centroids):
                    dist = sum((p - c) ** 2 for p, c in zip(point, centroid))
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = j
                if labels[i] != best_cluster:
                    labels[i] = best_cluster
                    changed = True

            if not changed:
                break

            # Обновление центроидов
            for j in range(k):
                cluster_points = [data[i] for i in range(n) if labels[i] == j]
                if cluster_points:
                    centroids[j] = [
                        sum(p[d] for p in cluster_points) / len(cluster_points)
                        for d in range(dim)
                    ]

        return centroids, labels

    # Генерация данных с 3 кластерами
    cluster_data = []
    true_labels = []

    centers = [(2, 2), (8, 8), (2, 8)]
    for center_idx, center in enumerate(centers):
        for _ in range(30):
            point = (
                center[0] + random.gauss(0, 1),
                center[1] + random.gauss(0, 1),
            )
            cluster_data.append(point)
            true_labels.append(center_idx)

    # K-means
    centroids, labels = simple_kmeans(cluster_data, k=3)

    # Оценка качества кластеризации
    correct = 0
    for i in range(len(cluster_data)):
        if labels[i] == true_labels[i]:
            correct += 1

    accuracy = correct / len(cluster_data)
    print(f"K-means кластеризация (3 кластера, {len(cluster_data)} точек):")
    print(f"  Точность: {accuracy:.2%}")
    print(f"  Центроиды:")
    for i, c in enumerate(centroids):
        print(f"    Кластер {i}: ({c[0]:.2f}, {c[1]:.2f})")

    # Визуализация (текстовая)
    print("\n  Текстовая визуализация кластеров (первые 60 точек):")
    grid_size = 20
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    for i, (x, y) in enumerate(cluster_data[:60]):
        gx = int((x / 12) * (grid_size - 1))
        gy = int((y / 12) * (grid_size - 1))
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        symbols = ["●", "■", "▲"]
        grid[grid_size - 1 - gy][gx] = symbols[labels[i] % 3]

    for row in grid:
        print("  " + "".join(row))

    print()


# ============================================================
# Демо 3: Формирование наград
# ============================================================

def demo_reward_shaping():
    """
    Демонстрация формирования наград:
    - Sparse vs dense rewards
    - Intrinsic motivation (внутренняя мотивация)
    - Reward shaping
    """
    print("=" * 70)
    print("ДЕМО 3: ФОРМИРОВАНИЕ НАГРАД (REWARD SHAPING)")
    print("=" * 70)

    # Подзадача 1: Sparse vs Dense rewards
    print("\n--- Подзадача 1: Sparse vs Dense Rewards ---\n")

    class GridWorld:
        """Простая среда —网格世界 с целями и препятствиями."""

        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height
            self.agent_pos = [0, 0]
            self.goals = []
            self.obstacles = []

        def set_goals(self, goals: list):
            self.goals = goals

        def set_obstacles(self, obstacles: list):
            self.obstacles = obstacles

        def reset(self):
            self.agent_pos = [0, 0]
            return tuple(self.agent_pos)

        def step(self, action: str) -> tuple:
            """Выполнение действия. Возвращает (new_state, reward, done)."""
            moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
            dx, dy = moves.get(action, (0, 0))
            new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]

            # Проверка границ
            new_pos[0] = max(0, min(self.height - 1, new_pos[0]))
            new_pos[1] = max(0, min(self.width - 1, new_pos[1]))

            # Проверка препятствий
            if tuple(new_pos) in self.obstacles:
                return tuple(self.agent_pos), -1.0, False

            self.agent_pos = new_pos
            state = tuple(self.agent_pos)

            # Проверка целей
            if state in self.goals:
                return state, 10.0, True

            return state, 0.0, False

        def sparse_reward(self, state, reward, done) -> float:
            """Sparse reward: награда только за достижение цели."""
            return reward if done else 0.0

        def dense_reward(self, state, reward, done) -> float:
            """
            Dense reward: штраф за каждый шаг + бонус за приближение к цели.
            Формула: reward = -0.1 + distance_improvement * 0.5
            """
            if done:
                return 10.0

            # Расстояние до ближайшей цели
            min_dist_before = min(
                abs(state[0] - g[0]) + abs(state[1] - g[1])
                for g in self.goals
            )

            return -0.1  # Базовый штраф за каждый шаг

    # Сравнение sparse и dense rewards
    env = GridWorld(10, 10)
    env.set_goals([(9, 9)])
    env.set_obstacles([(3, 3), (3, 4), (4, 3), (7, 7), (7, 8)])

    def run_episode(env, reward_fn, max_steps=50):
        """Запуск одного эпизода со случайной политикой."""
        state = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = random.choice(["up", "down", "left", "right"])
            next_state, env_reward, done = env.step(action)
            r = reward_fn(next_state, env_reward, done)
            total_reward += r
            steps += 1
            state = next_state
            if done:
                break

        return total_reward, steps, done

    print("Сравнение sparse и dense rewards (500 эпизодов):")
    print(f"\n{'Тип награды':<20} {'Средняя награда':<20} {'Доля успеха':<15} {'Средняя длина'}")
    print("-" * 70)

    for name, reward_fn in [("Sparse", env.sparse_reward), ("Dense", env.dense_reward)]:
        rewards = []
        successes = []
        lengths = []

        for _ in range(500):
            total_r, steps, done = run_episode(env, reward_fn)
            rewards.append(total_r)
            successes.append(done)
            lengths.append(steps)

        avg_reward = sum(rewards) / len(rewards)
        success_rate = sum(successes) / len(successes)
        avg_length = sum(lengths) / len(lengths)

        print(f"{name:<20} {avg_reward:<20.4f} {success_rate:<15.2%} {avg_length:.1f}")

    # Подзадача 2: Intrinsic motivation (внутренняя мотивация)
    print("\n--- Подзадача 2: Intrinsic Motivation ---\n")

    class IntrinsicMotivation:
        """
        Внутренняя мотивация — награда за исследование новых состояний.
        Используется, когда внешние награды sparse.

        Типы:
        1. Curiosity: награда за предсказание ошибок модели
        2. Coverage: награда за посещение новых состояний
        3. Information gain: награда за уменьшение неопределённости
        """

        def __init__(self):
            self.visited = collections.Counter()
            self.total_visits = 0

        def novelty_reward(self, state: tuple) -> float:
            """
            Награда за новизну состояния.
            Формула: reward = 1 / (1 + visit_count)
            """
            count = self.visited[state]
            return 1.0 / (1.0 + count)

        def coverage_reward(self, state: tuple) -> float:
            """
            Награда за покрытие.
            Формула: reward = new_state_bonus / total_states_explored
            """
            self.visited[state] += 1
            self.total_visits += 1
            unique_states = len(self.visited)

            # Бонус за новое состояние
            if self.visited[state] == 1:
                return 1.0
            # Убывающая награда за повторное посещение
            return 0.5 / self.visited[state]

        def information_gain(self, state: tuple) -> float:
            """
            Награда за прирост информации.
            Формула: ΔH = -log(p(state)) = log(1/p(state))
            """
            self.visited[state] += 1
            self.total_visits += 1

            p_state = self.visited[state] / self.total_visits
            if p_state > 0:
                return -math.log2(p_state)
            return 0.0

    intrinsic = IntrinsicMotivation()

    # Симуляция исследования среды
    print("Исследование среды (30 шагов, случайные действия):\n")
    print(f"{'Шаг':<6} {'Состояние':<12} {'Novelty':<12} {'Coverage':<12} {'Info Gain'}")
    print("-" * 60)

    for step in range(30):
        state = (random.randint(0, 9), random.randint(0, 9))
        novelty = intrinsic.novelty_reward(state)
        coverage = intrinsic.coverage_reward(state)
        info = intrinsic.information_gain(state)

        if step < 15 or step == 29:  # Показываем первые 15 и последний
            print(f"{step+1:<6} {str(state):<12} {novelty:<12.4f} {coverage:<12.4f} {info:.4f}")
        elif step == 15:
            print("  ...")

    print(f"\nУникальных состояний: {len(intrinsic.visited)} из 100")
    print(f"Всего посещений: {intrinsic.total_visits}")

    # Подзадача 3: Reward shaping
    print("\n--- Подзадача 3: Reward Shaping ---\n")

    class RewardShaper:
        """
        Преобразование наград для ускорения обучения.
        Формула Potential-Based Shaping:
        F(s, s') = γ * Φ(s') - Φ(s)
        где Φ — потенциальная функция, γ — коэффициент дисконтирования.
        """

        def __init__(self, gamma: float = 0.99):
            self.gamma = gamma
            self.potential_fn = None

        def set_potential_function(self, fn):
            """Установка потенциальной функции Φ(s)."""
            self.potential_fn = fn

        def shape(self, state: tuple, next_state: tuple, original_reward: float) -> float:
            """
            Формирование награды:
            shaped_reward = original_reward + γ * Φ(s') - Φ(s)
            """
            if self.potential_fn is None:
                return original_reward

            phi_s = self.potential_fn(state)
            phi_s_next = self.potential_fn(next_state)
            shaping = self.gamma * phi_s_next - phi_s

            return original_reward + shaping

    # Потенциальные функции
    def distance_to_goal_potential(state, goal=(9, 9)):
        """Потенциальная функция: обратное расстояние до цели."""
        dist = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        return 1.0 / (1.0 + dist)

    def coverage_potential(state, visited):
        """Потенциальная функция: покрытие карты."""
        return len(visited) / 100.0

    shaper = RewardShaper(gamma=0.99)
    shaper.set_potential_function(lambda s: distance_to_goal_potential(s))

    # Сравнение наград
    print("Сравнение оригинальных и сформированных наград:")
    print(f"\n{'Состояние':<15} {'Следующее':<15} {'Оригинал':<12} {'Сформированная':<18} {'Δ'}")
    print("-" * 75)

    for _ in range(10):
        state = (random.randint(0, 9), random.randint(0, 9))
        next_state = (random.randint(0, 9), random.randint(0, 9))
        original = random.uniform(-1, 1)
        shaped = shaper.shape(state, next_state, original)
        delta = shaped - original

        print(f"{str(state):<15} {str(next_state):<15} {original:<12.4f} {shaped:<18.4f} {delta:+.4f}")

    # Подзадача 4: Комбинированная система наград
    print("\n--- Подзадача 4: Комбинированная система наград ---\n")

    class CombinedRewardSystem:
        """Комбинированная система наград, объединяющая несколько источников."""

        def __init__(self):
            self.weights = {
                "task": 1.0,  # Основная задача
                "exploration": 0.3,  # Исследование
                "safety": 0.5,  # Безопасность
                "efficiency": 0.2,  # Эффективность
            }
            self.metrics = collections.defaultdict(float)

        def compute_reward(self, state: dict) -> float:
            """
            Вычисление комбинированной награды.
            R = w1*r1 + w2*r2 + w3*r3 + w4*r4
            """
            rewards = {}

            # Награда за задачу
            if state.get("goal_reached", False):
                rewards["task"] = 10.0
            else:
                rewards["task"] = -0.1

            # Награда за исследование
            rewards["exploration"] = state.get("novelty", 0.0)

            # Награда за безопасность
            if state.get("collision", False):
                rewards["safety"] = -5.0
            else:
                rewards["safety"] = 0.1

            # Награда за эффективность
            rewards["efficiency"] = -state.get("energy_used", 0.0) * 0.01

            # Взвешенная сумма
            total = sum(self.weights[k] * rewards[k] for k in self.weights)

            # Обновление метрик
            for k, v in rewards.items():
                self.metrics[k] += v

            return total

    reward_system = CombinedRewardSystem()

    # Симуляция 20 эпизодов
    print("Симуляция 20 эпизодов с комбинированной системой наград:\n")
    print(f"{'Эпизод':<10} {'Задача':<10} {'Исслед.':<10} {'Безоп.':<10} {'Эффек.':<10} {'Итого':<10}")
    print("-" * 60)

    episode_rewards = []
    for ep in range(20):
        state = {
            "goal_reached": random.random() < 0.15,
            "novelty": random.uniform(0, 1),
            "collision": random.random() < 0.1,
            "energy_used": random.uniform(0, 10),
        }
        total = reward_system.compute_reward(state)
        episode_rewards.append(total)

        if ep < 10 or ep == 19:
            print(f"{ep+1:<10} "
                  f"{'✓' if state['goal_reached'] else '✗':<10} "
                  f"{state['novelty']:<10.3f} "
                  f"{'✗' if state['collision'] else '✓':<10} "
                  f"{state['energy_used']:<10.3f} "
                  f"{total:<10.4f}")

    avg = sum(episode_rewards) / len(episode_rewards)
    print(f"\nСредняя награда за эпизод: {avg:.4f}")
    print("Средние метрики:")
    for k, v in reward_system.metrics.items():
        print(f"  {k}: {v / 20:.4f}")

    print()


# ============================================================
# Демо 4: Curriculum Learning
# ============================================================

def demo_curriculum_learning():
    """
    Демонстрация обучения по учебной программе:
    - Прогрессия сложности
    - Self-paced learning
    - Адаптивная сложность
    """
    print("=" * 70)
    print("ДЕМО 4: CURRICULUM LEARNING (ОБУЧЕНИЕ ПО УЧЕБНОЙ ПРОГРАММЕ)")
    print("=" * 70)

    # Подзадача 1: Прогрессия сложности
    print("\n--- Подзадача 1: Прогрессия сложности ---\n")

    class DifficultyProgression:
        """Прогрессия сложности задач с течением времени."""

        def __init__(self, initial_difficulty: float = 0.1):
            self.current_difficulty = initial_difficulty
            self.history = []
            self.success_threshold = 0.8  # Порог успеха для повышения сложности

        def get_difficulty(self) -> float:
            """Получение текущей сложности."""
            return self.current_difficulty

        def update(self, success_rate: float):
            """
            Обновление сложности на основе процента успеха.
            Если success_rate > порога → повышаем сложность.
            Если success_rate < порога/2 → понижаем сложность.
            """
            self.history.append({
                "difficulty": self.current_difficulty,
                "success_rate": success_rate,
            })

            if success_rate > self.success_threshold:
                # Повышение сложности (экспоненциально)
                self.current_difficulty *= 1.2
            elif success_rate < self.success_threshold / 2:
                # Понижение сложности
                self.current_difficulty *= 0.8

            self.current_difficulty = max(0.01, min(1.0, self.current_difficulty))

    progression = DifficultyProgression(0.1)

    print("Симуляция прогрессии сложности (20 этапов):\n")
    print(f"{'Этап':<8} {'Сложность':<12} {'Успех':<10} {'Действие'}")
    print("-" * 55)

    for stage in range(20):
        difficulty = progression.get_difficulty()

        # Имитация процента успеха (зависит от сложности)
        base_success = 1.0 - difficulty * 0.8
        noise = random.gauss(0, 0.1)
        success_rate = max(0, min(1, base_success + noise))

        old_diff = difficulty
        progression.update(success_rate)

        action = "↑ повышение" if progression.current_difficulty > old_diff else \
                 "↓ понижение" if progression.current_difficulty < old_diff else "→ без изменений"

        if stage < 10 or stage == 19:
            print(f"{stage+1:<8} {difficulty:<12.4f} {success_rate:<10.2%} {action}")

    # Подзадача 2: Self-paced learning
    print("\n--- Подзадача 2: Self-paced Learning ---\n")

    class SelfPacedLearner:
        """
        Self-paced learning — ученик выбирает сложность задач
        на основе своей текущей компетенции.

        Формула: P(выбрать задачу) = sigmoid(λ * (competence - difficulty))
        """

        def __init__(self, initial_competence: float = 0.3):
            self.competence = initial_competence
            self.learning_rate = 0.05
            self.tasks_completed = 0
            self.total_reward = 0

        def task_selection_prob(self, difficulty: float, lam: float = 5.0) -> float:
            """
            Вероятность выбора задачи определённой сложности.
            sigmoid(λ * (competence - difficulty))
            """
            z = lam * (self.competence - difficulty)
            return 1.0 / (1.0 + math.exp(-z))

        def complete_task(self, difficulty: float, success: bool):
            """Обработка выполнения задачи."""
            self.tasks_completed += 1

            if success:
                # Повышение компетенции пропорционально сложности
                self.competence += self.learning_rate * difficulty
                self.total_reward += difficulty
            else:
                # Небольшое снижение компетенции при неудаче
                self.competence -= self.learning_rate * 0.1

            self.competence = max(0.0, min(1.0, self.competence))

    learner = SelfPacedLearner(0.3)

    # Доступные задачи разной сложности
    task_difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("Self-paced обучение (30 шагов):\n")
    print(f"{'Шаг':<6} {'Компетенция':<14} {'Выбранная':<12} {'Вероятность':<14} {'Успех'}")
    print("-" * 60)

    for step in range(30):
        # Выбор задачи на основе вероятностей
        probs = [learner.task_selection_prob(d) for d in task_difficulties]
        total_prob = sum(probs)
        probs_normalized = [p / total_prob for p in probs]

        # Семплирование
        r = random.random()
        cumulative = 0
        chosen_idx = 0
        for i, p in enumerate(probs_normalized):
            cumulative += p
            if r <= cumulative:
                chosen_idx = i
                break

        chosen_diff = task_difficulties[chosen_idx]
        chosen_prob = probs[chosen_idx]

        # Имитация выполнения (успех зависит от компетенции)
        success_prob = learner.competence / chosen_diff if chosen_diff > 0 else 1.0
        success_prob = max(0.1, min(0.95, success_prob))
        success = random.random() < success_prob

        learner.complete_task(chosen_diff, success)

        if step < 15 or step == 29:
            status = "✓" if success else "✗"
            print(f"{step+1:<6} {learner.competence:<14.4f} {chosen_diff:<12.2f} "
                  f"{chosen_prob:<14.4f} {status}")

    print(f"\nИтого: компетенция={learner.competence:.4f}, задач={learner.tasks_completed}")

    # Подзадача 3: Адаптивная сложность
    print("\n--- Подзадача 3: Адаптивная сложность ---\n")

    class AdaptiveDifficulty:
        """
        Адаптивная сложность — подбор сложности под текущий уровень ученика.
        Использует Zone of Proximal Development (ZPD):
        задачи должны быть не слишком лёгкими и не слишком сложными.
        """

        def __init__(self):
            self.student_level = 0.5
            self.zpd_lower = 0.2  # Нижняя граница ZPD
            self.zpd_upper = 0.8  # Верхняя граница ZPD

        def recommend_difficulty(self) -> float:
            """
            Рекомендация сложности в пределах ZPD.
            Оптимальная сложность = student_level * (zpd_lower + zpd_upper) / 2
            """
            optimal = self.student_level * (self.zpd_lower + self.zpd_upper) / 2
            # Добавляем небольшой随机ный выбор в пределах ZPD
            noise = random.uniform(-0.1, 0.1)
            return max(self.zpd_lower, min(self.zpd_upper, optimal + noise))

        def update_level(self, task_difficulty: float, success: bool):
            """Обновление уровня ученика."""
            if success:
                # Успешное выполнение сложной задачи — больший рост
                growth = 0.05 * task_difficulty
                self.student_level += growth
            else:
                # Неудача — небольшое снижение
                self.student_level -= 0.02

            self.student_level = max(0.1, min(1.0, self.student_level))

        def get_zpd(self) -> tuple:
            """Получение текущей ZPD."""
            lower = self.student_level * self.zpd_lower
            upper = self.student_level * self.zpd_upper
            return (lower, upper)

    adaptive = AdaptiveDifficulty()

    print("Адаптивная сложность с ZPD (Zone of Proximal Development):\n")
    print(f"{'Шаг':<6} {'Уровень':<10} {'ZPD':<20} {'Рекоменд.':<12} {'Успех'}")
    print("-" * 60)

    for step in range(20):
        difficulty = adaptive.recommend_difficulty()
        zpd = adaptive.get_zpd()

        # Успех зависит от того, насколько сложность близка к уровню ученика
        distance = abs(adaptive.student_level - difficulty)
        success_prob = max(0.2, 1.0 - distance * 2)
        success = random.random() < success_prob

        adaptive.update_level(difficulty, success)

        if step < 10 or step == 19:
            status = "✓" if success else "✗"
            print(f"{step+1:<6} {adaptive.student_level:<10.4f} "
                  f"[{zpd[0]:.3f}, {zpd[1]:.3f}]  {difficulty:<12.4f} {status}")

    # Подзадача 4: Полная система Curriculum Learning
    print("\n--- Подзадача 4: Полная система Curriculum Learning ---\n")

    class CurriculumLearningSystem:
        """
        Полная система curriculum learning:
        - Множество этапов с нарастающей сложностью
        - Критерии перехода между этапами
        - Адаптивная скорость
        """

        def __init__(self):
            self.stages = []
            self.current_stage = 0
            self.stage_metrics = []

        def add_stage(self, name: str, difficulty_range: tuple, passing_score: float):
            """Добавление этапа curriculum."""
            self.stages.append({
                "name": name,
                "difficulty_range": difficulty_range,
                "passing_score": passing_score,
                "attempts": 0,
                "successes": 0,
            })

        def get_current_task(self) -> dict:
            """Генерация задачи текущего этапа."""
            if self.current_stage >= len(self.stages):
                return None

            stage = self.stages[self.current_stage]
            difficulty = random.uniform(*stage["difficulty_range"])

            return {
                "stage": stage["name"],
                "difficulty": difficulty,
            }

        def evaluate(self, success: bool):
            """Оценка результата выполнения задачи."""
            stage = self.stages[self.current_stage]
            stage["attempts"] += 1
            if success:
                stage["successes"] += 1

            # Проверка критерия перехода
            if stage["attempts"] >= 10:
                score = stage["successes"] / stage["attempts"]
                self.stage_metrics.append({
                    "stage": stage["name"],
                    "score": score,
                    "passing": score >= stage["passing_score"],
                })

                if score >= stage["passing_score"] and self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
                    return True  # Переход к следующему этапу

            return False

        def get_progress(self) -> list:
            """Получение прогресса по всем этапам."""
            progress = []
            for i, stage in enumerate(self.stages):
                if stage["attempts"] > 0:
                    score = stage["successes"] / stage["attempts"]
                else:
                    score = 0.0

                progress.append({
                    "name": stage["name"],
                    "score": score,
                    "completed": i < self.current_stage,
                    "current": i == self.current_stage,
                })
            return progress

    # Создание curriculum
    curriculum = CurriculumLearningSystem()
    curriculum.add_stage("Базовые навыки", (0.1, 0.3), 0.7)
    curriculum.add_stage("Средний уровень", (0.3, 0.6), 0.75)
    curriculum.add_stage("Продвинутый", (0.6, 0.8), 0.8)
    curriculum.add_stage("Экспертный", (0.8, 1.0), 0.85)

    print("Запуск Curriculum Learning (максимум 100 задач):\n")

    total_tasks = 0
    for _ in range(100):
        task = curriculum.get_current_task()
        if task is None:
            break

        # Имитация выполнения
        success = random.random() < (1.0 - task["difficulty"] * 0.5)
        promoted = curriculum.evaluate(success)
        total_tasks += 1

        if promoted:
            print(f"  → Переход к этапу: {curriculum.stages[curriculum.current_stage]['name']}")

    # Вывод прогресса
    print(f"\nВсего задач выполнено: {total_tasks}\n")
    print("Прогресс по этапам:")
    print(f"{'Этап':<25} {'Балл':<10} {'Статус'}")
    print("-" * 50)

    for stage_info in curriculum.get_progress():
        status = "✓ пройден" if stage_info["completed"] else \
                 "→ текущий" if stage_info["current"] else "⏳ ожидание"
        bar = "█" * int(stage_info["score"] * 20)
        print(f"{stage_info['name']:<25} {stage_info['score']:<10.2%} {status} {bar}")

    print()


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    demo_online_learning()
    demo_experience_accumulation()
    demo_reward_shaping()
    demo_curriculum_learning()
