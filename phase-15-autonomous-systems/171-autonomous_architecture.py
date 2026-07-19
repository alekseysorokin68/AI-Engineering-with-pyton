"""171 — Autonomous System Architecture: полный стек автономии, принятие решений

Темы:
  1. Autonomy Levels (L0 manual → L5 full autonomy, SAE classification)
  2. Decision Architecture (perception → world model → planner → executor)
  3. State Management (belief state, goal stack, execution context)
  4. Integration Patterns (modular architecture, plugin system, event-driven)

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
# Демо 1: Уровни автономии (L0–L5)
# ============================================================

def demo_autonomy_levels():
    """
    Демонстрация классификации уровней автономии по SAE:
    L0 — полное ручное управление
    L1 — водитель/оператор помогает (ассистент)
    L2 — частичная автоматизация (несколько функций)
    L3 — условная автоматизация (система берёт контроль в определённых условиях)
    L4 — высокая автоматизация (без водителя в определённых зонах)
    L5 — полная автоматизация (без ограничений)
    """
    print("=" * 70)
    print("ДЕМО 1: УРОВНИ АВТОНОМИИ (SAE L0–L5)")
    print("=" * 70)

    # Определение уровней автономии с описаниями и критериями
    autonomy_levels = {
        "L0": {
            "name": "Нет автоматизации",
            "description": "Полное ручное управление, человек выполняет все задачи",
            "human_in_loop": True,
            "decision_speed_ms": 0,
            "reliability": 0.0,
        },
        "L1": {
            "name": "Ассистент водителя",
            "description": "Система помогает с одной функцией ( cruise control, парковка)",
            "human_in_loop": True,
            "decision_speed_ms": 500,
            "reliability": 0.6,
        },
        "L2": {
            "name": "Частичная автоматизация",
            "description": "Система управляет рулём и газом, но человек监控ирует",
            "human_in_loop": True,
            "decision_speed_ms": 200,
            "reliability": 0.8,
        },
        "L3": {
            "name": "Условная автоматизация",
            "description": "Система берёт контроль в определённых условиях, человек готов перехватить",
            "human_in_loop": True,
            "decision_speed_ms": 100,
            "reliability": 0.92,
        },
        "L4": {
            "name": "Высокая автоматизация",
            "description": "Система полностью автономна в определённых зонах",
            "human_in_loop": False,
            "decision_speed_ms": 50,
            "reliability": 0.98,
        },
        "L5": {
            "name": "Полная автоматизация",
            "description": "Система автономна в любых условиях, без ограничений",
            "human_in_loop": False,
            "decision_speed_ms": 20,
            "reliability": 0.999,
        },
    }

    # Подзадача 1: Вывод таблицы уровней
    print("\n--- Подзадача 1: Классификация уровней автономии ---\n")
    print(f"{'Уровень':<6} {'Название':<30} {'Человек в цикле':<18} {'Надёжность'}")
    print("-" * 75)
    for level, info in autonomy_levels.items():
        human = "Да" if info["human_in_loop"] else "Нет"
        print(f"{level:<6} {info['name']:<30} {human:<18} {info['reliability']:.1%}")

    # Подзадача 2: Модель принятия решений на разных уровнях
    print("\n--- Подзадача 2: Модель принятия решений ---\n")

    class DecisionModel:
        """Модель принятия решений, зависящая от уровня автономии."""

        def __init__(self, level: str):
            self.level = level
            self.info = autonomy_levels[level]

        def decide(self, situation: str) -> dict:
            """Принятие решения в зависимости от уровня автономии."""
            decision_time = self.info["decision_speed_ms"]
            # Вероятность успешного решения зависит от уровня
            base_prob = self.info["reliability"]
            # Добавляем небольшой随机ный шум
            noise = random.uniform(-0.05, 0.05)
            actual_prob = max(0, min(1, base_prob + noise))

            success = random.random() < actual_prob
            return {
                "level": self.level,
                "situation": situation,
                "success": success,
                "time_ms": decision_time,
                "human_intervention": self.info["human_in_loop"] and not success,
            }

    # Симуляция сценариев для каждого уровня
    scenarios = ["Препятствие на дороге", "Поворот на перекрёстке", "Парковка"]

    for level in ["L0", "L2", "L4", "L5"]:
        model = DecisionModel(level)
        print(f"Уровень {level}: {autonomy_levels[level]['name']}")
        for scenario in scenarios:
            result = model.decide(scenario)
            status = "УСПЕХ" if result["success"] else "ПРОВАЛ"
            intervention = " (требуется вмешательство)" if result["human_intervention"] else ""
            print(f"  {scenario}: {status} за {result['time_ms']}мс{intervention}")
        print()

    # Подзадача 3: Метрика зрелости автономной системы
    print("--- Подзадача 3: Метрика зрелости автономной системы ---\n")

    def calculate_maturity(level: str, use_cases: int, edge_cases_handled: int) -> float:
        """
        Формула зрелости автономной системы:
        Maturity = level_score * use_case_factor * edge_case_factor
        level_score = L_number / 5 (нормализованный уровень)
        use_case_factor = 1 - exp(-use_cases / 10)
        edge_case_factor = edge_cases_handled / (edge_cases_handled + 5)
        """
        level_num = int(level[1])
        level_score = level_num / 5.0
        use_case_factor = 1 - math.exp(-use_cases / 10.0)
        edge_case_factor = edge_cases_handled / (edge_cases_handled + 5.0)
        maturity = level_score * use_case_factor * edge_case_factor
        return maturity

    # Примеры расчёта зрелости
    configs = [
        ("L1", 2, 1, "Базовый ассистент"),
        ("L3", 8, 4, "Условная автономия (город)"),
        ("L4", 15, 10, "Высокая автономия (шоссе)"),
        ("L5", 30, 20, "Полная автономия"),
    ]

    print(f"{'Конфигурация':<35} {'Зрелость':>10} {'Уровень'}")
    print("-" * 60)
    for level, cases, edge, name in configs:
        m = calculate_maturity(level, cases, edge)
        print(f"{name:<35} {m:>10.3f}  {level}")

    # Подзадача 4: Симуляция перехода между уровнями
    print("\n--- Подзадача 4: Траектория повышения автономии ---\n")

    # Моделируем постепенное повышение уровня автономии
    trajectory = []
    current_reliability = 0.0
    for month in range(1, 13):
        # Каждый месяц надёжность растёт с убывающей отдачей
        improvement = 0.15 * (1 - current_reliability)
        current_reliability += improvement + random.uniform(-0.02, 0.02)
        current_reliability = max(0, min(1, current_reliability))

        # Определяем уровень по надёжности
        if current_reliability < 0.5:
            level = "L0"
        elif current_reliability < 0.65:
            level = "L1"
        elif current_reliability < 0.8:
            level = "L2"
        elif current_reliability < 0.9:
            level = "L3"
        elif current_reliability < 0.98:
            level = "L4"
        else:
            level = "L5"

        trajectory.append((month, current_reliability, level))

    print("Месяц  Надёжность  Уровень  Прогресс")
    print("-" * 55)
    for month, rel, level in trajectory:
        bar = "█" * int(rel * 30) + "░" * (30 - int(rel * 30))
        print(f"  {month:2d}     {rel:.3f}      {level}    {bar}")

    print()


# ============================================================
# Демо 2: Архитектура принятия решений
# ============================================================

def demo_decision_architecture():
    """
    Демонстрация архитектуры принятия решений:
    Восприятие → Модель мира → Планировщик → Исполнитель
    Каждый компонент обрабатывает информацию и передаёт дальше.
    """
    print("=" * 70)
    print("ДЕМО 2: АРХИТЕКТУРА ПРИНЯТИЯ РЕШЕНИЙ")
    print("=" * 70)

    # Компонент 1: Восприятие (Perception)
    class PerceptionModule:
        """Модуль восприятия — собирает и нормализует данные из среды."""

        def __init__(self):
            self.sensors = {}  # Источники данных
            self.noise_level = 0.1  # Уровень шума датчиков

        def add_sensor(self, name: str, value_range: tuple):
            """Регистрация датчика с диапазоном значений."""
            self.sensors[name] = {"range": value_range, "value": None}

        def read_sensors(self) -> dict:
            """Считывание показаний всех датчиков с шумом."""
            readings = {}
            for name, sensor in self.sensors.items():
                low, high = sensor["range"]
                true_value = random.uniform(low, high)
                noise = random.gauss(0, self.noise_level * (high - low))
                noisy_value = true_value + noise
                noisy_value = max(low, min(high, noisy_value))
                readings[name] = round(noisy_value, 3)
                sensor["value"] = noisy_value
            return readings

        def detect_anomalies(self, readings: dict) -> list:
            """Обнаружение аномалий — значений за пределами нормы."""
            anomalies = []
            for name, value in readings.items():
                low, high = self.sensors[name]["range"]
                mid = (low + high) / 2
                std = (high - low) / 6  # 3-сигма правило
                if abs(value - mid) > 2.5 * std:
                    anomalies.append((name, value, "anomaly"))
            return anomalies

    # Компонент 2: Модель мира (World Model)
    class WorldModel:
        """Модель мира — хранит текущее состояние среды и предсказывает будущее."""

        def __init__(self):
            self.state = {}  # Текущее состояние
            self.history = []  # История состояний
            self.prediction_horizon = 3  # Горизонт предсказания (шаги)

        def update(self, observations: dict):
            """Обновление модели мира на основе наблюдений."""
            self.history.append(dict(self.state))
            self.state.update(observations)

        def predict(self, steps: int = 1) -> list:
            """
            Предсказание будущих состояний.
            Формула: state(t+k) = state(t) + sum(velocity * dt^i / i!)
            Простая модель с линейной экстраполяцией.
            """
            if len(self.history) < 2:
                return [dict(self.state)] * steps

            # Вычисляем скорость изменения
            prev = self.history[-1]
            curr = self.state
            velocity = {}
            for key in curr:
                if key in prev:
                    velocity[key] = curr[key] - prev[key]
                else:
                    velocity[key] = 0.0

            predictions = []
            for k in range(1, steps + 1):
                pred = {}
                for key, val in curr.items():
                    vel = velocity.get(key, 0.0)
                    pred[key] = val + vel * k
                predictions.append(pred)
            return predictions

        def get_confidence(self) -> float:
            """Оценка уверенности модели на основе свежести данных."""
            # Чем больше история, тем выше уверенность
            history_factor = min(1.0, len(self.history) / 20.0)
            return 0.5 + 0.5 * history_factor

    # Компонент 3: Планировщик (Planner)
    class Planner:
        """Планировщик — генерирует план действий на основе модели мира."""

        def __init__(self):
            self.actions = []  # Доступные действия
            self.current_plan = []  # Текущий план

        def add_action(self, name: str, preconditions: dict, effects: dict, cost: float):
            """Добавление действия с предусловиями, эффектами и стоимостью."""
            self.actions.append({
                "name": name,
                "preconditions": preconditions,
                "effects": effects,
                "cost": cost,
            })

        def plan(self, current_state: dict, goal: dict) -> list:
            """
            Простой贪心 планировщик: выбирает действие, которое максимально
            приближает к цели при минимальной стоимости.
            Формула: score(action) = (goal_progress) / (cost + 1)
            """
            plan = []
            state = dict(current_state)
            max_steps = 10

            for step in range(max_steps):
                best_action = None
                best_score = -float("inf")

                for action in self.actions:
                    # Проверяем предусловия
                    if all(state.get(k, 0) >= v for k, v in action["preconditions"].items()):
                        # Считаем прогресс к цели
                        goal_progress = 0
                        for gk, gv in goal.items():
                            current_val = state.get(gk, 0)
                            new_val = current_val + action["effects"].get(gk, 0)
                            if gv > current_val:
                                goal_progress += min(new_val - current_val, gv - current_val)

                        score = goal_progress / (action["cost"] + 1)
                        if score > best_score:
                            best_score = score
                            best_action = action

                if best_action is None:
                    break

                # Применяем лучшее действие
                plan.append(best_action["name"])
                for k, v in best_action["effects"].items():
                    state[k] = state.get(k, 0) + v

                # Проверяем, достигнута ли цель
                if all(state.get(k, 0) >= v for k, v in goal.items()):
                    break

            return plan

    # Компонент 4: Исполнитель (Executor)
    class Executor:
        """Исполнитель — выполняет план действий и отслеживает результаты."""

        def __init__(self):
            self.execution_log = []
            self.success_rate = 0.9

        def execute(self, plan: list) -> list:
            """Выполнение плана с вероятностным успехом."""
            results = []
            for i, action in enumerate(plan):
                success = random.random() < self.success_rate
                result = {
                    "step": i + 1,
                    "action": action,
                    "success": success,
                    "timestamp": time.time(),
                }
                self.execution_log.append(result)
                results.append(result)
                if not success:
                    print(f"  [ОШИБКА] Шаг {i+1}: действие '{action}' не выполнено")
            return results

    # Демонстрация полного конвейера
    print("\n--- Полный конвейер: Восприятие → Модель → План → Исполнение ---\n")

    # Инициализация компонентов
    perception = PerceptionModule()
    perception.add_sensor("temperature", (15.0, 35.0))
    perception.add_sensor("humidity", (30.0, 90.0))
    perception.add_sensor("light", (0.0, 1000.0))
    perception.add_sensor("noise", (30.0, 90.0))

    world = WorldModel()
    planner_obj = Planner()

    # Добавляем действия
    planner_obj.add_action("включить_кондиционер", {"temperature": 25}, {"temperature": -3, "power": 2}, 1.5)
    planner_obj.add_action("открыть_окно", {"temperature": 20}, {"temperature": -1, "humidity": -5}, 0.5)
    planner_obj.add_action("включить_освещение", {"light": 200}, {"light": 500, "power": 1}, 0.8)
    planner_obj.add_action("включить_обогрев", {"temperature": 15}, {"temperature": 2, "power": 3}, 2.0)
    planner_obj.add_action("увлажнить_воздух", {"humidity": 40}, {"humidity": 15, "power": 1}, 1.0)

    executor = Executor()

    # Несколько итераций
    goal = {"temperature": 22.0, "humidity": 60.0, "light": 500.0}

    for cycle in range(3):
        print(f"--- Цикл {cycle + 1} ---")

        # Шаг 1: Восприятие
        readings = perception.read_sensors()
        print(f"  [Восприятие] Датчики: {readings}")
        anomalies = perception.detect_anomalies(readings)
        if anomalies:
            print(f"  [Восприятие] Аномалии: {anomalies}")

        # Шаг 2: Обновление модели мира
        world.update(readings)
        confidence = world.get_confidence()
        print(f"  [Модель мира] Уверенность: {confidence:.3f}")

        # Шаг 3: Планирование
        current = dict(world.state)
        plan = planner_obj.plan(current, goal)
        print(f"  [Планировщик] План: {plan}")

        # Шаг 4: Исполнение
        if plan:
            results = executor.execute(plan)
            successes = sum(1 for r in results if r["success"])
            print(f"  [Исполнитель] Выполнено: {successes}/{len(results)}")
        else:
            print("  [Исполнитель] Нет действий для выполнения")
        print()

    # Подзадача 3: Метрики архитектуры
    print("--- Метрики архитектуры ---\n")

    # Latency breakdown
    latencies = {
        "Восприятие": random.uniform(5, 15),
        "Модель мира": random.uniform(2, 8),
        "Планировщик": random.uniform(10, 50),
        "Исполнитель": random.uniform(5, 20),
    }
    total = sum(latencies.values())

    print("Задержка по компонентам:")
    for comp, lat in latencies.items():
        pct = lat / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {comp:<20} {lat:>6.2f}мс  ({pct:>5.1f}%) {bar}")
    print(f"  {'ИТОГО':<20} {total:>6.2f}мс")

    # Подзадача 4: Цикл обратной связи
    print("\n--- Цикл обратной связи ---\n")

    # Модель с обратной связью
    feedback_scores = []
    system_performance = 0.5  # Начальная производительность

    for iteration in range(10):
        # Система принимает решение
        action_quality = random.uniform(0.3, 1.0)
        # Обратная связь корректирует поведение
        feedback = action_quality * 0.3 + system_performance * 0.7
        system_performance = 0.8 * system_performance + 0.2 * feedback
        feedback_scores.append(system_performance)

    print("Итерация  Производительность  График")
    print("-" * 55)
    for i, score in enumerate(feedback_scores):
        bar = "█" * int(score * 30)
        print(f"  {i+1:2d}       {score:.4f}             {bar}")

    print()


# ============================================================
# Демо 3: Управление состоянием
# ============================================================

def demo_state_management():
    """
    Демонстрация управления состоянием автономной системы:
    - Belief state (состояние убеждений)
    - Goal stack (стек целей)
    - Execution context (контекст выполнения)
    """
    print("=" * 70)
    print("ДЕМО 3: УПРАВЛЕНИЕ СОСТОЯНИЕМ")
    print("=" * 70)

    # Подзадача 1: Belief State — вероятностное состояние
    print("\n--- Подзадача 1: Belief State (Состояние убеждений) ---\n")

    class BeliefState:
        """
        Belief state — распределение вероятностей по возможным состояниям мира.
        Используется, когда наблюдения неполные или зашумлённые.
        Формула обновления (Байес): P(s|o) = P(o|s) * P(s) / P(o)
        """

        def __init__(self, states: list):
            # Начальное равномерное распределение
            self.states = states
            self.probabilities = {s: 1.0 / len(states) for s in states}
            self.observations = []

        def update(self, observation: dict):
            """
            Обновление belief state с помощью байесовского вывода.
            observation = {"state": "sunny", "confidence": 0.9}
            """
            observed_state = observation["state"]
            confidence = observation["confidence"]

            # Вероятность наблюдения для каждого состояния
            for state in self.states:
                if state == observed_state:
                    likelihood = confidence
                else:
                    # Чем дальше состояние от наблюдённого, тем меньше правдоподобие
                    # Используем простую метрику расстояния
                    state_idx = self.states.index(state)
                    obs_idx = self.states.index(observed_state)
                    distance = abs(state_idx - obs_idx)
                    likelihood = (1 - confidence) * math.exp(-0.5 * distance)

                # Байесовское обновление
                self.probabilities[state] *= likelihood

            # Нормализация
            total = sum(self.probabilities.values())
            for state in self.states:
                self.probabilities[state] /= total

            self.observations.append(observation)

        def get_most_likely(self) -> str:
            """Получение наиболее вероятного состояния."""
            return max(self.probabilities, key=self.probabilities.get)

        def get_entropy(self) -> float:
            """
            Энтропия belief state — мера неопределённости.
            H = -sum(p * log2(p))
            """
            entropy = 0.0
            for p in self.probabilities.values():
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy

    # Пример: определение погоды по неполным наблюдениям
    weather_states = ["солнечно", "облачно", "дождь", "снег"]
    belief = BeliefState(weather_states)

    print("Начальное распределениеbelief state:")
    for state, prob in belief.probabilities.items():
        print(f"  {state:<15} {prob:.4f} {'█' * int(prob * 40)}")
    print(f"  Энтропия: {belief.get_entropy():.4f} бит")

    # Симуляция наблюдений
    observations = [
        {"state": "облачно", "confidence": 0.7},
        {"state": "облачно", "confidence": 0.8},
        {"state": "дождь", "confidence": 0.6},
        {"state": "дождь", "confidence": 0.9},
    ]

    for obs in observations:
        belief.update(obs)
        print(f"\nНаблюдение: {obs['state']} (уверенность {obs['confidence']})")
        for state, prob in belief.probabilities.items():
            print(f"  {state:<15} {prob:.4f} {'█' * int(prob * 40)}")
        print(f"  Наиболее вероятное: {belief.get_most_likely()}")
        print(f"  Энтропия: {belief.get_entropy():.4f} бит")

    # Подзадача 2: Goal Stack (Стек целей)
    print("\n--- Подзадача 2: Goal Stack (Стек целей) ---\n")

    class GoalStack:
        """
        Стек целей — управление иерархией целей агента.
        Поддерживает вложенность: подцели вкладываются в основную цель.
        """

        def __init__(self):
            self.stack = []
            self.completed = []

        def push(self, goal: str, priority: int = 0):
            """Добавление цели в стек (с учётом приоритета)."""
            self.stack.append({"goal": goal, "priority": priority, "status": "active"})

        def pop(self) -> dict:
            """Извлечение текущей цели из стека."""
            if self.stack:
                return self.stack.pop()
            return None

        def complete_current(self):
            """Отметка текущей цели как выполненной."""
            if self.stack:
                current = self.stack.pop()
                current["status"] = "completed"
                self.completed.append(current)

        def get_current(self) -> dict:
            """Получение текущей (верхней) цели."""
            return self.stack[-1] if self.stack else None

        def suspend_and_push(self, subgoal: str):
            """
            Приостановка текущей цели и добавление подцели.
            Текущая цель остаётся в стеке, подцель добавляется поверх.
            """
            self.push(subgoal)

        def display(self):
            """Отображение стека целей."""
            if not self.stack:
                print("  (стек пуст)")
                return
            for i in range(len(self.stack) - 1, -1, -1):
                goal = self.stack[i]
                prefix = "→ " if i == len(self.stack) - 1 else "  "
                status = " [АКТИВНА]" if i == len(self.stack) - 1 else " [приостановлена]"
                print(f"  {prefix}{goal['goal']}{status}")

    stack = GoalStack()

    # Демонстрация работы со стеком целей
    print("Добавляем основную цель:")
    stack.push("Достигнуть пункта назначения", priority=10)
    stack.display()

    print("\nНавигатор требует уточнение маршрута — приостановка и подцель:")
    stack.suspend_and_push("Построить маршрут")
    stack.display()

    print("\nНужно найти место для парковки — ещё одна подцель:")
    stack.suspend_and_push("Найти парковку")
    stack.display()

    print("\nПарковка найдена — подцель выполнена:")
    stack.complete_current()
    stack.display()

    print("\nМаршрут построен — подцель выполнена:")
    stack.complete_current()
    stack.display()

    print("\nПункт назначения достигнут — основная цель выполнена:")
    stack.complete_current()
    stack.display()

    print(f"\nВыполненные цели: {[g['goal'] for g in stack.completed]}")

    # Подзадача 3: Execution Context (Контекст выполнения)
    print("\n--- Подзадача 3: Execution Context (Контекст выполнения) ---\n")

    class ExecutionContext:
        """
        Контекст выполнения — хранит всю информацию, необходимую
        для выполнения текущей задачи: переменные, таймеры, счётчики.
        """

        def __init__(self):
            self.variables = {}  # Переменные контекста
            self.timers = {}  # Таймеры
            self.counters = {}  # Счётчики
            self.history = []  # История изменений

        def set_var(self, name: str, value):
            """Установка переменной контекста."""
            old = self.variables.get(name)
            self.variables[name] = value
            self.history.append(("set", name, old, value))

        def get_var(self, name: str, default=None):
            """Получение переменной контекста."""
            return self.variables.get(name, default)

        def start_timer(self, name: str):
            """Запуск таймера."""
            self.timers[name] = time.time()

        def stop_timer(self, name: str) -> float:
            """Остановка таймера и возврат времени."""
            if name in self.timers:
                elapsed = time.time() - self.timers[name]
                del self.timers[name]
                return elapsed
            return 0.0

        def increment(self, name: str, step: int = 1) -> int:
            """Инкремент счётчика."""
            self.counters[name] = self.counters.get(name, 0) + step
            return self.counters[name]

        def snapshot(self) -> dict:
            """Снимок текущего контекста."""
            return {
                "variables": dict(self.variables),
                "timers": list(self.timers.keys()),
                "counters": dict(self.counters),
            }

    ctx = ExecutionContext()

    # Симуляция выполнения задачи
    print("Сценарий: обработка заказа в интернет-магазине\n")

    ctx.set_var("order_id", "ORD-2024-001")
    ctx.set_var("customer", "Иван Иванов")
    ctx.set_var("items_count", 3)
    ctx.set_var("total_price", 0.0)

    ctx.start_timer("processing")

    # Обработка каждого товара
    prices = [299.99, 149.50, 89.00]
    for i, price in enumerate(prices):
        step = ctx.increment("items_processed")
        current_total = ctx.get_var("total_price", 0)
        ctx.set_var("total_price", current_total + price)
        print(f"  Товар {step}: +{price:.2f}₽ → итого {ctx.get_var('total_price'):.2f}₽")

    elapsed = ctx.stop_timer("processing")

    # Применяем скидку
    discount = ctx.increment("discount_percent", 10)
    total = ctx.get_var("total_price")
    ctx.set_var("final_price", total * (1 - discount / 100))

    print(f"\n  Скидка: {discount}%")
    print(f"  Итого: {ctx.get_var('final_price'):.2f}₽")
    print(f"  Время обработки: {elapsed*1000:.2f}мс")

    print("\n  Снимок контекста:")
    snapshot = ctx.snapshot()
    for key, val in snapshot["variables"].items():
        print(f"    {key}: {val}")
    print(f"    Счётчики: {snapshot['counters']}")

    # Подзадача 4: Интеграция компонентов состояния
    print("\n--- Подзадача 4: Интеграция компонентов состояния ---\n")

    # Комбинируем все три компонента
    print("Объединённая система состояния:\n")

    # Belief state для среды
    env_belief = BeliefState(["свободно", "занято", "препятствие"])

    # Goal stack для целей
    robot_goals = GoalStack()
    robot_goals.push("Доставить посылку", priority=10)

    # Execution context для текущей задачи
    robot_ctx = ExecutionContext()
    robot_ctx.set_var("position", (0, 0))
    robot_ctx.set_var("battery", 100.0)
    robot_ctx.set_var("speed", 0.0)

    # Симуляция навигации
    steps = [(1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    goal_pos = (3, 3)

    print(f"Цель: {robot_goals.get_current()['goal']} → позиция {goal_pos}")
    print(f"{'Шаг':<6} {'Позиция':<12} {'Батарея':<10} {'Уверенность':<12} {'Действие'}")
    print("-" * 60)

    for i, (dx, dy) in enumerate(steps):
        pos = robot_ctx.get_var("position")
        new_pos = (pos[0] + dx, pos[1] + dy)
        robot_ctx.set_var("position", new_pos)

        battery = robot_ctx.get_var("battery")
        robot_ctx.set_var("battery", battery - 5.0)

        # Обновляем belief state
        if random.random() < 0.7:
            env_belief.update({"state": "свободно", "confidence": 0.8})
        else:
            env_belief.update({"state": "занято", "confidence": 0.6})

        most_likely = env_belief.get_most_likely()
        action = "достигнута" if new_pos == goal_pos else "движение"

        print(f"  {i+1:<4} {str(new_pos):<12} {robot_ctx.get_var('battery'):<10.1f} {most_likely:<12} {action}")

    print(f"\n  Итоговая позиция: {robot_ctx.get_var('position')}")
    print(f"  Оставшаяся батарея: {robot_ctx.get_var('battery'):.1f}%")
    print(f"  Belief state: {env_belief.probabilities}")

    print()


# ============================================================
# Демо 4: Паттерны интеграции
# ============================================================

def demo_integration_patterns():
    """
    Демонстрация паттернов интеграции автономных систем:
    - Модульная архитектура
    - Плагинная система
    - Event-driven архитектура
    """
    print("=" * 70)
    print("ДЕМО 4: ПАТТЕРНЫ ИНТЕГРАЦИИ")
    print("=" * 70)

    # Подзадача 1: Модульная архитектура
    print("\n--- Подзадача 1: Модульная архитектура ---\n")

    class Module:
        """Базовый класс модуля автономной системы."""

        def __init__(self, name: str):
            self.name = name
            self.inputs = {}
            self.outputs = {}
            self.dependencies = []

        def add_dependency(self, module_name: str):
            """Добавление зависимости от другого модуля."""
            self.dependencies.append(module_name)

        def process(self, data: dict) -> dict:
            """Обработка данных (переопределяется в наследниках)."""
            return {"module": self.name, "processed": True, **data}

        def __repr__(self):
            return f"Module({self.name})"

    class PerceptionModuleImpl(Module):
        """Модуль восприятия — обработка данных датчиков."""

        def process(self, data: dict) -> dict:
            raw = data.get("sensor_data", {})
            processed = {}
            for key, value in raw.items():
                # Простая фильтрация: нормализация к [0, 1]
                processed[key] = max(0, min(1, value / 100.0))
            return {"module": self.name, "filtered": processed}

    class PlanningModuleImpl(Module):
        """Модуль планирования — генерация плана действий."""

        def process(self, data: dict) -> dict:
            filtered = data.get("filtered", {})
            # Простое правило: если temperature > 0.7, включить охлаждение
            actions = []
            for key, value in filtered.items():
                if "temp" in key.lower() and value > 0.7:
                    actions.append(f"охладить_{key}")
                elif "light" in key.lower() and value < 0.3:
                    actions.append(f"осветить_{key}")
            return {"module": self.name, "actions": actions}

    class ExecutionModuleImpl(Module):
        """Модуль исполнения — выполнение действий."""

        def process(self, data: dict) -> dict:
            actions = data.get("actions", [])
            results = []
            for action in actions:
                success = random.random() < 0.9  # 90% вероятность успеха
                results.append({"action": action, "success": success})
            return {"module": self.name, "results": results}

    # Создание модулей
    perception = PerceptionModuleImpl("Восприятие")
    planner = PlanningModuleImpl("Планировщик")
    executor = ExecutionModuleImpl("Исполнитель")

    # Установка зависимостей
    planner.add_dependency("Восприятие")
    executor.add_dependency("Планировщик")

    modules = [perception, planner, executor]

    print("Модули системы:")
    for mod in modules:
        deps = ", ".join(mod.dependencies) if mod.dependencies else "нет"
        print(f"  {mod.name} (зависимости: {deps})")

    # Конвейерная обработка
    print("\nКонвейерная обработка:")
    input_data = {"sensor_data": {"temperature": 75, "humidity": 45, "light": 20}}

    current_data = input_data
    for mod in modules:
        result = mod.process(current_data)
        print(f"  {mod.name}: {result}")
        current_data = result

    # Подзадача 2: Плагинная система
    print("\n--- Подзадача 2: Плагинная система ---\n")

    class PluginManager:
        """Менеджер плагинов — регистрация и вызов плагинов."""

        def __init__(self):
            self.plugins = {}  # Имя плагина → класс плагина
            self.instances = {}  # Имя плагина → экземпляр
            self.load_order = []  # Порядок загрузки

        def register(self, name: str, plugin_class):
            """Регистрация класса плагина."""
            self.plugins[name] = plugin_class
            self.load_order.append(name)

        def activate(self, name: str, **kwargs):
            """Активация (создание экземпляра) плагина."""
            if name in self.plugins:
                self.instances[name] = self.plugins[name](**kwargs)
                print(f"  [Плагин] {name} активирован")

        def deactivate(self, name: str):
            """Деактивация плагина."""
            if name in self.instances:
                del self.instances[name]
                print(f"  [Плагин] {name} деактивирован")

        def call(self, name: str, method: str, *args, **kwargs):
            """Вызов метода плагина."""
            if name in self.instances:
                plugin = self.instances[name]
                func = getattr(plugin, method, None)
                if func:
                    return func(*args, **kwargs)
            return None

    # Примеры плагинов
    class LoggerPlugin:
        """Плагин логирования."""

        def __init__(self, log_file: str = "system.log"):
            self.log_file = log_file
            self.entries = []

        def log(self, message: str, level: str = "INFO"):
            entry = {"level": level, "message": message, "timestamp": time.time()}
            self.entries.append(entry)
            return entry

        def get_recent(self, n: int = 5) -> list:
            return self.entries[-n:]

    class MetricsPlugin:
        """Плагин метрик."""

        def __init__(self):
            self.metrics = {}

        def record(self, name: str, value: float):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

        def average(self, name: str) -> float:
            values = self.metrics.get(name, [])
            return sum(values) / len(values) if values else 0.0

    class CachePlugin:
        """Плагин кэширования."""

        def __init__(self, max_size: int = 100):
            self.cache = {}
            self.max_size = max_size
            self.hits = 0
            self.misses = 0

        def get(self, key: str):
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

        def set(self, key: str, value):
            if len(self.cache) >= self.max_size:
                # Удаляем старый элемент (FIFO)
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[key] = value

        def stats(self) -> dict:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}

    # Демонстрация плагинной системы
    manager = PluginManager()

    # Регистрация плагинов
    manager.register("logger", LoggerPlugin)
    manager.register("metrics", MetricsPlugin)
    manager.register("cache", CachePlugin)

    # Активация плагинов
    manager.activate("logger", log_file="autonomous_system.log")
    manager.activate("metrics")
    manager.activate("cache", max_size=50)

    # Использование плагинов
    manager.call("logger", "log", "Система запущена")
    manager.call("logger", "log", "Загружены датчики")
    manager.call("logger", "log", "Плагин cache активен")

    print(f"\n  Логи (последние 3):")
    for entry in manager.call("logger", "get_recent", 3):
        print(f"    [{entry['level']}] {entry['message']}")

    # Запись метрик
    for _ in range(10):
        latency = random.uniform(5, 50)
        manager.call("metrics", "record", "latency", latency)

    avg_latency = manager.call("metrics", "average", "latency")
    print(f"\n  Средняя задержка: {avg_latency:.2f}мс")

    # Использование кэша
    for i in range(15):
        key = f"request_{i % 10}"  # Повторяющиеся ключи
        cached = manager.call("cache", "get", key)
        if cached is None:
            manager.call("cache", "set", key, f"result_{i}")
            result = f"result_{i}"
        else:
            result = cached

    cache_stats = manager.call("cache", "stats")
    print(f"\n  Статистика кэша: попадания={cache_stats['hits']}, "
          f"промахи={cache_stats['misses']}, "
          f"коэффициент={cache_stats['hit_rate']:.2%}")

    # Подзадача 3: Event-driven архитектура
    print("\n--- Подзадача 3: Event-driven архитектура ---\n")

    class EventBus:
        """Шина событий — центральный компонент для event-driven архитектуры."""

        def __init__(self):
            self.subscribers = {}  # Тип события → список обработчиков
            self.event_log = []  # Журнал событий

        def subscribe(self, event_type: str, handler):
            """Подписка на тип события."""
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

        def unsubscribe(self, event_type: str, handler):
            """Отписка от типа события."""
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(handler)

        def publish(self, event_type: str, data: dict):
            """Публикация события."""
            event = {"type": event_type, "data": data, "timestamp": time.time()}
            self.event_log.append(event)

            # Вызов обработчиков
            handlers = self.subscribers.get(event_type, [])
            for handler in handlers:
                handler(event)

            # Глобальные обработчики (подписаны на все события)
            if "*" in self.subscribers:
                for handler in self.subscribers["*"]:
                    handler(event)

    # Создание шины событий
    bus = EventBus()

    # Обработчики событий
    sensor_readings = []
    alerts = []
    all_events = []

    def on_sensor_data(event):
        """Обработчик данных датчиков."""
        sensor_readings.append(event["data"])

    def on_anomaly(event):
        """Обработчик аномалий."""
        alerts.append(f"АНОМАЛИЯ: {event['data']}")

    def on_all_events(event):
        """Глобальный обработчик — логирует все события."""
        all_events.append(event["type"])

    # Подписка на события
    bus.subscribe("sensor_data", on_sensor_data)
    bus.subscribe("anomaly", on_anomaly)
    bus.subscribe("*", on_all_events)

    # Генерация событий
    print("Генерация событий:")
    for i in range(5):
        temperature = random.uniform(20, 40)
        bus.publish("sensor_data", {"temperature": temperature, "source": "temp_sensor"})

        if temperature > 35:
            bus.publish("anomaly", {"type": "high_temperature", "value": temperature})

    print(f"  Получено показаний датчиков: {len(sensor_readings)}")
    print(f"  Сработавшие алерты: {len(alerts)}")
    for alert in alerts:
        print(f"    {alert}")

    print(f"\n  Журнал событий (типы): {collections.Counter(all_events)}")

    # Подзадача 4: Сравнение паттернов
    print("\n--- Подзадача 4: Сравнение паттернов интеграции ---\n")

    patterns = [
        {
            "name": "Модульная архитектура",
            "pros": ["Чёткая структура", "Легко тестировать", "Явные зависимости"],
            "cons": ["Жёсткая связность", "Сложно добавлять компоненты"],
            "best_for": "Стабильные системы с фиксированным набором компонентов",
            "complexity": "Низкая",
        },
        {
            "name": "Плагинная система",
            "pros": ["Гибкость", "Расширяемость", "Независимость компонентов"],
            "cons": ["Сложность управления", "Возможные конфликты плагинов"],
            "best_for": "Системы, требующие динамической настройки",
            "complexity": "Средняя",
        },
        {
            "name": "Event-driven",
            "pros": ["Слабая связность", "Масштабируемость", "Асинхронность"],
            "cons": ["Сложность отладки", "Порядок событий не гарантирован"],
            "best_for": "Распределённые системы, высокая нагрузка",
            "complexity": "Высокая",
        },
    ]

    print(f"{'Паттерн':<25} {'Сложность':<12} {'Лучше всего для'}")
    print("-" * 75)
    for p in patterns:
        print(f"{p['name']:<25} {p['complexity']:<12} {p['best_for']}")

    print()
    for p in patterns:
        print(f"\n{p['name']}:")
        print(f"  + {', '.join(p['pros'])}")
        print(f"  - {', '.join(p['cons'])}")

    print()


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    demo_autonomy_levels()
    demo_decision_architecture()
    demo_state_management()
    demo_integration_patterns()
