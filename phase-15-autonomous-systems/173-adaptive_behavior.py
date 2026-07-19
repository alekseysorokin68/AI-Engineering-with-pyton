"""173 — Adaptive Behavior: деревья поведения, утилитарные системы, GOAP

Темы:
  1. Behavior Trees (selector, sequence, decorator, blackboard)
  2. Utility Systems (scoring actions, curve evaluation, best action selection)
  3. GOAP (Goal-Oriented Action Planning, preconditions, effects, cost)
  4. Adaptive Strategies (context switching, parameter tuning, behavior blending)

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
# Демо 1: Деревья поведения (Behavior Trees)
# ============================================================

def demo_behavior_trees():
    """
    Демонстрация деревьев поведения:
    - Selector (выбор) — выполняет детей по очереди, пока один не успешен
    - Sequence (последовательность) — выполняет детей по очереди, пока все успешны
    - Decorator (декоратор) — модифицирует результат дочернего узла
    - Blackboard (чёрная доска) — общее хранилище данных для узлов
    """
    print("=" * 70)
    print("ДЕМО 1: ДЕРЕВЬЯ ПОВЕДЕНИЯ (BEHAVIOR TREES)")
    print("=" * 70)

    # Blackboard — общее хранилище данных
    class Blackboard:
        """Чёрная доска — общее хранилище данных для всех узлов дерева."""

        def __init__(self):
            self.data = {}

        def set(self, key: str, value):
            self.data[key] = value

        def get(self, key: str, default=None):
            return self.data.get(key, default)

        def has(self, key: str) -> bool:
            return key in self.data

        def display(self):
            print(f"  Blackboard: {self.data}")

    # Статусы узлов
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

    # Базовый узел дерева
    class BTNode:
        """Базовый узел дерева поведения."""

        def __init__(self, name: str):
            self.name = name

        def tick(self, blackboard: Blackboard) -> str:
            raise NotImplementedError

    # Листовые узлы ( Actions и Conditions)
    class Condition(BTNode):
        """Условие — проверяет что-то на чёрной доске."""

        def __init__(self, name: str, key: str, expected_value=None):
            super().__init__(name)
            self.key = key
            self.expected_value = expected_value

        def tick(self, blackboard: Blackboard) -> str:
            value = blackboard.get(self.key)
            if self.expected_value is None:
                return SUCCESS if value else FAILURE
            return SUCCESS if value == self.expected_value else FAILURE

    class Action(BTNode):
        """Действие — выполняет действие и возвращает результат."""

        def __init__(self, name: str, action_fn):
            super().__init__(name)
            self.action_fn = action_fn

        def tick(self, blackboard: Blackboard) -> str:
            return self.action_fn(blackboard)

    # Составные узлы
    class Selector(BTNode):
        """
        Селектор — пробует детей по очереди.
        Возвращает SUCCESS, как только один из детей успешен.
        Возвращает FAILURE, если все дети завершились неудачей.
        """

        def __init__(self, name: str, children: list):
            super().__init__(name)
            self.children = children

        def tick(self, blackboard: Blackboard) -> str:
            for child in self.children:
                result = child.tick(blackboard)
                if result == SUCCESS:
                    return SUCCESS
            return FAILURE

    class Sequence(BTNode):
        """
        Последовательность — выполняет детей по очереди.
        Возвращает FAILURE, как только один из детей завершился неудачей.
        Возвращает SUCCESS, если все дети успешны.
        """

        def __init__(self, name: str, children: list):
            super().__init__(name)
            self.children = children

        def tick(self, blackboard: Blackboard) -> str:
            for child in self.children:
                result = child.tick(blackboard)
                if result == FAILURE:
                    return FAILURE
            return SUCCESS

    class Decorator(BTNode):
        """
        Декоратор — модифицирует результат дочернего узла.
        Типы: Inverter (инвертирует), Repeater (повторяет), Limiter (ограничивает).
        """

        def __init__(self, name: str, child: BTNode, decorator_type: str = "inverter"):
            super().__init__(name)
            self.child = child
            self.decorator_type = decorator_type
            self.execution_count = 0

        def tick(self, blackboard: Blackboard) -> str:
            result = self.child.tick(blackboard)

            if self.decorator_type == "inverter":
                # Инвертирует результат
                if result == SUCCESS:
                    return FAILURE
                elif result == FAILURE:
                    return SUCCESS
                return RUNNING

            elif self.decorator_type == "repeater":
                # Повторяет действие N раз
                max_repeats = blackboard.get("repeater_max", 3)
                self.execution_count += 1
                if self.execution_count >= max_repeats:
                    self.execution_count = 0
                    return SUCCESS
                return RUNNING

            elif self.decorator_type == "succeeder":
                # Всегда возвращает SUCCESS
                return SUCCESS

            elif self.decorator_type == "limiter":
                # Ограничивает количество вызовов
                max_calls = blackboard.get("limiter_max", 2)
                self.execution_count += 1
                if self.execution_count > max_calls:
                    return FAILURE
                return result

            return result

    # Демонстрация 1: Простое дерево поведения для NPC
    print("\n--- Подзадача 1: Дерево поведения для NPC ---\n")

    bb = Blackboard()
    bb.set("health", 80)
    bb.set("enemy_nearby", True)
    bb.set("has_ammo", True)
    bb.set("ally_nearby", False)

    # Дерево решений NPC:
    # Selector:
    #   Sequence [Приоритет 1: Атака]:
    #     Condition: враг рядом?
    #     Condition: есть патроны?
    #     Action: атаковать
    #   Sequence [Приоритет 2: Лечение]:
    #     Condition: здоровье < 30?
    #     Action: использовать аптечку
    #   Sequence [Приоритет 3: Помощь союзнику]:
    #     Condition: союзник рядом?
    #     Action: помочь союзнику
    #   Action [Приоритет 4: Патрулирование]:
    #     Action: патрулировать

    def attack_action(bb):
        print("    → NPC атакует врага!")
        return SUCCESS

    def heal_action(bb):
        print("    → NPC использует аптечку!")
        bb.set("health", min(100, bb.get("health") + 30))
        return SUCCESS

    def help_ally_action(bb):
        print("    → NPC помогает союзнику!")
        return SUCCESS

    def patrol_action(bb):
        print("    → NPC патрулирует территорию...")
        return SUCCESS

    # Построение дерева
    attack_sequence = Sequence("Атака", [
        Condition("Враг рядом?", "enemy_nearby", True),
        Condition("Есть патроны?", "has_ammo", True),
        Action("Атаковать", attack_action),
    ])

    heal_sequence = Sequence("Лечение", [
        Condition("Низкое здоровье?", "health", None),  # Проверяет что значение существует
        Action("Использовать аптечку", heal_action),
    ])

    help_sequence = Sequence("Помощь", [
        Condition("Союзник рядом?", "ally_nearby", True),
        Action("Помочь союзнику", help_ally_action),
    ])

    patrol_action_node = Action("Патрулирование", patrol_action)

    root = Selector("NPC Behavior", [attack_sequence, heal_sequence, help_sequence, patrol_action_node])

    print("Дерево поведения NPC:")
    print("  Selector (корень)")
    print("    ├── Sequence: Атака")
    print("    │   ├── Condition: Враг рядом?")
    print("    │   ├── Condition: Есть патроны?")
    print("    │   └── Action: Атаковать")
    print("    ├── Sequence: Лечение")
    print("    │   ├── Condition: Низкое здоровье?")
    print("    │   └── Action: Использовать аптечку")
    print("    ├── Sequence: Помощь")
    print("    │   ├── Condition: Союзник рядом?")
    print("    │   └── Action: Помочь союзнику")
    print("    └── Action: Патрулирование")

    print(f"\nЧёрная доска: {bb.data}")
    print("\nВыполнение дерева:")
    result = root.tick(bb)
    print(f"\nРезультат: {result}")

    # Подзадача 2: Декораторы
    print("\n--- Подзадача 2: Декораторы ---\n")

    def simple_action(bb):
        print("    → Действие выполнено")
        return SUCCESS

    # Inverter — инвертирует результат
    print("Inverter (инвертор):")
    inverter = Decorator("Inverter", Action("Действие", simple_action), "inverter")
    result = inverter.tick(bb)
    print(f"  Результат: {result} (инвертирован SUCCESS → FAILURE)")

    # Succeeder — всегда возвращает SUCCESS
    print("\nSucceeder (всегда успех):")
    succeeder = Decorator("Succeeder", Action("Действие", simple_action), "succeeder")
    result = succeeder.tick(bb)
    print(f"  Результат: {result}")

    # Limiter — ограничивает количество вызовов
    print("\nLimiter (ограничитель, макс. 2 вызова):")
    bb.set("limiter_max", 2)
    limiter = Decorator("Limiter", Action("Действие", simple_action), "limiter")
    for i in range(4):
        result = limiter.tick(bb)
        print(f"  Вызов {i+1}: {result}")

    # Подзадача 3: Сложное дерево с вложенными селекторами
    print("\n--- Подзадача 3: Иерархическое дерево поведения ---\n")

    # Дерево для умного дома
    home_bb = Blackboard()
    home_bb.set("temperature", 28.0)
    home_bb.set("time_of_day", "evening")
    home_bb.set("occupants", 3)
    home_bb.set("energy_mode", False)

    def adjust_ac(bb):
        temp = bb.get("temperature")
        if temp > 25:
            print(f"    → Кондиционер: охлаждение до 22°C (сейчас {temp}°C)")
        else:
            print(f"    → Кондиционер: обогрев до 22°C (сейчас {temp}°C)")
        return SUCCESS

    def adjust_lights(bb):
        time = bb.get("time_of_day")
        if time == "evening" or time == "night":
            print(f"    → Освещение: вечерний режим (40% яркости)")
        else:
            print(f"    → Освещение: дневной режим (100% яркость)")
        return SUCCESS

    def adjust_music(bb):
        occupants = bb.get("occupants")
        print(f"    → Музыка: релакс playlist ({occupants} человек)")
        return SUCCESS

    def energy_saving(bb):
        print("    → Энергосбережение: выключение ненужных приборов")
        return SUCCESS

    # Иерархическое дерево
    comfort_tree = Selector("Комфорт", [
        Sequence("Климат", [
            Action("Настроить кондиционер", adjust_ac),
        ]),
        Sequence("Освещение", [
            Action("Настроить свет", adjust_lights),
        ]),
    ])

    entertainment_tree = Selector("Развлечения", [
        Sequence("Музыка", [
            Action("Включить музыку", adjust_music),
        ]),
    ])

    energy_tree = Selector("Энергия", [
        Sequence("Энергосбережение", [
            Action("Сэкономить энергию", energy_saving),
        ]),
    ])

    root_home = Selector("Умный дом", [comfort_tree, entertainment_tree, energy_tree])

    print("Дерево умного дома:")
    result = root_home.tick(home_bb)
    print(f"\nРезультат: {result}")

    # Подзадача 4: Сравнение BT с FSM
    print("\n--- Подзадача 4: Сравнение Behavior Trees и FSM ---\n")

    print("Behavior Trees vs Finite State Machine:\n")
    comparisons = [
        ("Расширяемость", "BT: легко добавлять новые ветки", "FSM: нужно менять все переходы"),
        ("Читаемость", "BT: визуальная иерархия", "FSM: таблица переходов"),
        ("Гибкость", "BT: приоритеты через порядок", "FSM: фиксированные переходы"),
        ("Память", "BT: через Blackboard", "FSM: через состояние"),
        ("Параллелизм", "BT: поддерживается (Parallel)", "FSM: сложно"),
        ("Тестируемость", "BT: каждый узел отдельно", "FSM: тест переходов"),
    ]

    print(f"{'Аспект':<18} {'Behavior Trees':<35} {'FSM'}")
    print("-" * 85)
    for aspect, bt, fsm in comparisons:
        print(f"{aspect:<18} {bt:<35} {fsm}")

    print()


# ============================================================
# Демо 2: Утилитарные системы
# ============================================================

def demo_utility_systems():
    """
    Демонстрация утилитарных систем:
    - Подсчёт очков для действий
    - Оценка кривых (Response curves)
    - Выбор лучшего действия
    """
    print("=" * 70)
    print("ДЕМО 2: УТИЛИТАРНЫЕ СИСТЕМЫ (UTILITY SYSTEMS)")
    print("=" * 70)

    # Подзадача 1: Response Curves (Кривые отклика)
    print("\n--- Подзадача 1: Response Curves ---\n")

    class ResponseCurve:
        """
        Response curve — функция преобразования входного значения в выходное.
        Типы: Linear, Exponential, Logarithmic, Logistic, Step.
        """

        @staticmethod
        def linear(x: float, a: float = 1.0, b: float = 0.0) -> float:
            """Линейная: y = a*x + b"""
            return a * x + b

        @staticmethod
        def exponential(x: float, a: float = 1.0, b: float = 2.0) -> float:
            """Экспоненциальная: y = a * b^x"""
            return a * (b ** x)

        @staticmethod
        def logarithmic(x: float, a: float = 1.0, base: float = math.e) -> float:
            """Логарифмическая: y = a * log_b(x + 1)"""
            if x + 1 <= 0:
                return 0
            return a * math.log(x + 1) / math.log(base)

        @staticmethod
        def logistic(x: float, k: float = 1.0, x0: float = 0.5) -> float:
            """Логистическая (сигмоида): y = 1 / (1 + e^(-k*(x-x0)))"""
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))

        @staticmethod
        def power(x: float, a: float = 1.0, n: float = 2.0) -> float:
            """Степенная: y = a * x^n"""
            return a * (max(0, x) ** n)

        @staticmethod
        def gaussian(x: float, mu: float = 0.5, sigma: float = 0.2) -> float:
            """Гауссова: y = e^(-(x-mu)^2 / (2*sigma^2))"""
            return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Визуализация кривых
    print("Response Curves (значения от 0 до 1 с шагом 0.1):\n")
    print(f"{'x':<6} {'Linear':<10} {'Exp':<10} {'Log':<10} {'Logistic':<10} {'Gaussian':<10}")
    print("-" * 60)

    for i in range(11):
        x = i / 10.0
        linear = ResponseCurve.linear(x)
        exp = ResponseCurve.exponential(x, a=0.1, b=2.0)
        log = ResponseCurve.logarithmic(x)
        logistic = ResponseCurve.logistic(x, k=10, x0=0.5)
        gaussian = ResponseCurve.gaussian(x)

        print(f"{x:<6.1f} {linear:<10.3f} {exp:<10.3f} {log:<10.3f} {logistic:<10.3f} {gaussian:<10.3f}")

    # Подзадача 2: Scoring Actions (Подсчёт очков действий)
    print("\n--- Подзадача 2: Scoring Actions ---\n")

    class ActionScorer:
        """Оценщик действий — подсчёт очков на основе 여러 факторов."""

        def __init__(self):
            self.factors = {}  # Имя фактора → (вес, функция оценки)

        def add_factor(self, name: str, weight: float, curve_fn):
            """Добавление фактора оценки."""
            self.factors[name] = (weight, curve_fn)

        def score(self, action: dict) -> float:
            """
            Подсчёт общего очка действия.
            Score = sum(weight_i * curve_i(value_i))
            """
            total = 0.0
            for name, (weight, curve_fn) in self.factors.items():
                value = action.get(name, 0.0)
                total += weight * curve_fn(value)
            return total

    # Пример: оценка действий NPC
    scorer = ActionScorer()
    scorer.add_factor("urgency", 0.3, lambda x: ResponseCurve.logistic(x, k=10, x0=0.5))
    scorer.add_factor("feasibility", 0.4, lambda x: ResponseCurve.linear(x))
    scorer.add_factor("risk", 0.2, lambda x: 1.0 - x)  # Чем меньше риск, тем лучше
    scorer.add_factor("reward", 0.1, lambda x: ResponseCurve.logistic(x, k=5, x0=0.3))

    # Действия NPC
    actions = [
        {"name": "атака", "urgency": 0.9, "feasibility": 0.7, "risk": 0.6, "reward": 0.8},
        {"name": "лечение", "urgency": 0.3, "feasibility": 0.9, "risk": 0.1, "reward": 0.4},
        {"name": "побег", "urgency": 0.5, "feasibility": 0.8, "risk": 0.3, "reward": 0.2},
        {"name": "оборона", "urgency": 0.6, "feasibility": 0.85, "risk": 0.4, "reward": 0.5},
        {"name": "разведка", "urgency": 0.2, "feasibility": 0.6, "risk": 0.5, "reward": 0.6},
    ]

    print("Оценка действий NPC:")
    print(f"{'Действие':<12} {'Срочность':<12} {'Выполним.':<12} {'Риск':<10} {'Награда':<10} {'ИТОГО':<10}")
    print("-" * 65)

    scored_actions = []
    for action in actions:
        score = scorer.score(action)
        scored_actions.append((action["name"], score))
        print(f"{action['name']:<12} {action['urgency']:<12.2f} {action['feasibility']:<12.2f} "
              f"{action['risk']:<10.2f} {action['reward']:<10.2f} {score:<10.4f}")

    # Лучшее действие
    best = max(scored_actions, key=lambda x: x[1])
    print(f"\nЛучшее действие: {best[0]} (оценка: {best[1]:.4f})")

    # Подзадача 3: Best Action Selection (Выбор лучшего действия)
    print("\n--- Подзадача 3: Best Action Selection ---\n")

    class UtilityAI:
        """Утилитарный ИИ — выбор лучшего действия на основе полезности."""

        def __init__(self):
            self.actions = []
            self.consideration_threshold = 0.1  # Порог consideration

        def add_action(self, name: str, utility_fn, weight: float = 1.0):
            """Добавление действия с функцией полезности."""
            self.actions.append({
                "name": name,
                "utility_fn": utility_fn,
                "weight": weight,
                "score": 0.0,
            })

        def evaluate(self, context: dict) -> list:
            """Оценка всех действий в текущем контексте."""
            for action in self.actions:
                raw_score = action["utility_fn"](context)
                action["score"] = raw_score * action["weight"]

            # Сортировка по убыванию полезности
            return sorted(self.actions, key=lambda a: a["score"], reverse=True)

        def select(self, context: dict) -> dict:
            """Выбор лучшего действия."""
            scored = self.evaluate(context)

            # Фильтрация по порогу consideration
            viable = [a for a in scored if a["score"] > self.consideration_threshold]

            if not viable:
                return None

            # Выбор с учётом случайности (epsilon-greedy)
            if random.random() < 0.1:  # epsilon = 0.1
                return random.choice(viable)

            return viable[0]

    # Создание утилитарного ИИ
    ai = UtilityAI()

    # Действия с функциями полезности
    def eat_utility(ctx):
        hunger = ctx.get("hunger", 0)
        return hunger  # Чем голоднее, тем полезнее еда

    def sleep_utility(ctx):
        fatigue = ctx.get("fatigue", 0)
        return fatigue * 1.2  # Сон очень важен при усталости

    def work_utility(ctx):
        money = ctx.get("money", 0)
        hunger = ctx.get("hunger", 0)
        fatigue = ctx.get("fatigue", 0)
        # Работа полезна, но не при голоде или усталости
        if hunger > 0.8 or fatigue > 0.8:
            return 0.1
        return (1 - money) * 0.7  # Чем меньше денег, тем важнее работать

    def socialize_utility(ctx):
        social = ctx.get("social_need", 0)
        return social * 0.5

    def exercise_utility(ctx):
        health = ctx.get("health", 1)
        fatigue = ctx.get("fatigue", 0)
        if fatigue > 0.7:
            return 0.1
        return (1 - health) * 0.6

    ai.add_action("поесть", eat_utility, weight=1.0)
    ai.add_action("спать", sleep_utility, weight=1.2)
    ai.add_action("работать", work_utility, weight=0.8)
    ai.add_action("общаться", socialize_utility, weight=0.5)
    ai.add_action("тренироваться", exercise_utility, weight=0.6)

    # Симуляция дней
    print("Симуляция принятия решений (7 дней):\n")
    context = {
        "hunger": 0.5,
        "fatigue": 0.3,
        "money": 0.4,
        "social_need": 0.6,
        "health": 0.7,
    }

    for day in range(1, 8):
        # Выбор действия
        chosen = ai.select(context)

        print(f"День {day}:")
        print(f"  Состояние: голод={context['hunger']:.2f}, усталость={context['fatigue']:.2f}, "
              f"деньги={context['money']:.2f}")

        # Оценка всех действий
        scored = ai.evaluate(context)
        for action in scored[:3]:
            marker = " ← ВЫБРАНО" if action["name"] == chosen["name"] else ""
            print(f"    {action['name']:<15} {action['score']:.4f}{marker}")

        print(f"  → Действие: {chosen['name']}")

        # Обновление состояния после действия
        if chosen["name"] == "поесть":
            context["hunger"] = max(0, context["hunger"] - 0.4)
        elif chosen["name"] == "спать":
            context["fatigue"] = max(0, context["fatigue"] - 0.5)
        elif chosen["name"] == "работать":
            context["money"] = min(1, context["money"] + 0.2)
            context["fatigue"] = min(1, context["fatigue"] + 0.3)
        elif chosen["name"] == "общаться":
            context["social_need"] = max(0, context["social_need"] - 0.3)
        elif chosen["name"] == "тренироваться":
            context["health"] = min(1, context["health"] + 0.15)
            context["fatigue"] = min(1, context["fatigue"] + 0.2)

        # Постепенное изменение состояния
        context["hunger"] = min(1, context["hunger"] + 0.15)
        context["fatigue"] = min(1, context["fatigue"] + 0.1)
        context["social_need"] = min(1, context["social_need"] + 0.08)

        print()

    # Подзадача 4: Сравнение утилитарных систем
    print("--- Подзадача 4: Сравнение подходов к оценке ---\n")

    approaches = [
        {
            "name": "Весовая сумма",
            "formula": "Score = Σ(wi * fi(x))",
            "pros": "Простота, прозрачность",
            "cons": "Линейность, трудно настроить веса",
        },
        {
            "name": "Response Curves",
            "formula": "Score = Σ(wi * curve_i(fi(x)))",
            "pros": "Нелинейность, гибкость",
            "cons": "Много параметров, сложно калибровать",
        },
        {
            "name": "Дерево решений",
            "formula": "if-else правила",
            "pros": "Интуитивно, легко отлаживать",
            "cons": "Экспоненциальный рост, хрупкость",
        },
        {
            "name": "Нейросеть",
            "formula": "Score = NN(features)",
            "pros": "Автонастройка, сложные паттерны",
            "cons": "Чёрный ящик, нужно много данных",
        },
    ]

    print(f"{'Подход':<20} {'Формула':<30} {'Плюсы':<25} {'Минусы'}")
    print("-" * 100)
    for a in approaches:
        print(f"{a['name']:<20} {a['formula']:<30} {a['pros']:<25} {a['cons']}")

    print()


# ============================================================
# Демо 3: GOAP (Goal-Oriented Action Planning)
# ============================================================

def demo_goap():
    """
    Демонстрация GOAP (Goal-Oriented Action Planning):
    - Preconditions (предусловия)
    - Effects (эффекты)
    - Cost (стоимость)
    - Планирование через A* поиск
    """
    print("=" * 70)
    print("ДЕМО 3: GOAP (GOAL-ORIENTED ACTION PLANNING)")
    print("=" * 70)

    # Подзадача 1: Определение GOAP компонентов
    print("\n--- Подзадача 1: Компоненты GOAP ---\n")

    class GOAPAction:
        """Действие GOAP с предусловиями, эффектами и стоимостью."""

        def __init__(self, name: str, preconditions: dict, effects: dict, cost: float):
            self.name = name
            self.preconditions = preconditions  # Что должно быть true
            self.effects = effects  # Что станет true
            self.cost = cost

        def is_applicable(self, state: dict) -> bool:
            """Проверка: выполнены ли все предусловия."""
            for key, value in self.preconditions.items():
                if state.get(key) != value:
                    return False
            return True

        def apply(self, state: dict) -> dict:
            """Применение эффектов действия к состоянию."""
            new_state = dict(state)
            for key, value in self.effects.items():
                new_state[key] = value
            return new_state

        def __repr__(self):
            return f"GOAPAction({self.name})"

    class GOAPGoal:
        """Цель GOAP — набор желаемых значений состояния."""

        def __init__(self, name: str, desired_state: dict, priority: float = 1.0):
            self.name = name
            self.desired_state = desired_state
            self.priority = priority

        def is_achieved(self, state: dict) -> bool:
            """Проверка: достигнута ли цель."""
            for key, value in self.desired_state.items():
                if state.get(key) != value:
                    return False
            return True

    # Пример: действия фермера
    print("Пример: GOAP для фермера\n")

    actions = [
        GOAPAction(
            "копать_землю",
            preconditions={"есть_инструмент": True, "земля_рыхлая": False},
            effects={"земля_рыхлая": True},
            cost=1.0,
        ),
        GOAPAction(
            "посадить_семена",
            preconditions={"земля_рыхлая": True, "есть_семена": True},
            effects={"семена_посажены": True},
            cost=0.5,
        ),
        GOAPAction(
            "полить_растения",
            preconditions={"семена_посажены": True},
            effects={"растения_политы": True},
            cost=0.3,
        ),
        GOAPAction(
            "собрать_урожай",
            preconditions={"растения_политы": True, "прошло_время": True},
            effects={"урожай_собран": True},
            cost=0.8,
        ),
        GOAPAction(
            "купить_инструмент",
            preconditions={"есть_деньги": True},
            effects={"есть_инструмент": True},
            cost=2.0,
        ),
        GOAPAction(
            "купить_семена",
            preconditions={"есть_деньги": True},
            effects={"есть_семена": True},
            cost=1.0,
        ),
        GOAPAction(
            "подождать",
            preconditions={"семена_посажены": True},
            effects={"прошло_время": True},
            cost=0.1,
        ),
    ]

    # Начальное состояние
    initial_state = {
        "есть_инструмент": True,
        "есть_семена": False,
        "есть_деньги": True,
        "земля_рыхлая": False,
        "семена_посажены": False,
        "растения_политы": False,
        "прошло_время": False,
        "урожай_собран": False,
    }

    print("Начальное состояние:")
    for key, value in initial_state.items():
        print(f"  {key}: {value}")

    print(f"\nДоступные действия:")
    for action in actions:
        print(f"  {action.name}:")
        print(f"    Предусловия: {action.preconditions}")
        print(f"    Эффекты: {action.effects}")
        print(f"    Стоимость: {action.cost}")

    # Подзадача 2: A* планирование
    print("\n--- Подзадача 2: A* Планирование ---\n")

    def goap_plan(initial_state: dict, goal: GOAPGoal, actions: list) -> list:
        """
        A* поиск плана действий.
        f(n) = g(n) + h(n)
        g(n) = суммарная стоимость пути
        h(n) = количество невыполненных условий цели
        """
        # Начальная нода
        start_node = {
            "state": initial_state,
            "path": [],
            "g": 0,
            "h": sum(1 for k, v in goal.desired_state.items() if initial_state.get(k) != v),
        }
        start_node["f"] = start_node["g"] + start_node["h"]

        # Открытый список (приоритетная очередь)
        open_list = [start_node]
        # Закрытый список (посещённые состояния)
        closed_set = set()

        iterations = 0
        max_iterations = 1000

        while open_list and iterations < max_iterations:
            iterations += 1

            # Выбираем ноду с минимальным f
            open_list.sort(key=lambda n: n["f"])
            current = open_list.pop(0)

            # Проверка: достигнута ли цель
            if goal.is_achieved(current["state"]):
                return current["path"]

            # Хэш состояния для проверки посещений
            state_hash = json.dumps(current["state"], sort_keys=True)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)

            # Развёртывание: пробуем все применимые действия
            for action in actions:
                if action.is_applicable(current["state"]):
                    new_state = action.apply(current["state"])
                    new_path = current["path"] + [action.name]
                    new_g = current["g"] + action.cost
                    new_h = sum(1 for k, v in goal.desired_state.items() if new_state.get(k) != v)
                    new_f = new_g + new_h

                    new_node = {
                        "state": new_state,
                        "path": new_path,
                        "g": new_g,
                        "h": new_h,
                        "f": new_f,
                    }

                    new_state_hash = json.dumps(new_state, sort_keys=True)
                    if new_state_hash not in closed_set:
                        open_list.append(new_node)

        return None  # План не найден

    # Планирование: собрать урожай
    goal = GOAPGoal("собрать_урожай", {"урожай_собран": True}, priority=1.0)

    print(f"Цель: {goal.name}")
    print(f"Желаемое состояние: {goal.desired_state}")
    print(f"\nПоиск плана...")

    plan = goap_plan(initial_state, goal, actions)

    if plan:
        print(f"\nНайденный план ({len(plan)} шагов):")
        state = dict(initial_state)
        for i, action_name in enumerate(plan):
            action = next(a for a in actions if a.name == action_name)
            print(f"  {i+1}. {action_name}")
            print(f"     Предусловия: {action.preconditions}")
            state = action.apply(state)
            print(f"     Состояние после: {dict(list(state.items())[:3])}...")
    else:
        print("  План не найден!")

    # Подзадача 3: Стоимость и оптимизация
    print("\n--- Подзадача 3: Стоимость и оптимизация ---\n")

    # Сравнение планов с разными весами стоимости
    print("Влияние стоимости действий на план:\n")

    cost_configs = [
        ("Все действия дешёвые", {a.name: 0.1 for a in actions}),
        ("Только копать дорогое", {"копать_землю": 5.0, **{a.name: 0.1 for a in actions if a.name != "копать_землю"}}),
        ("Купить инструмент дорогое", {"купить_инструмент": 10.0, **{a.name: 0.1 for a in actions if a.name != "купить_инструмент"}}),
    ]

    for config_name, costs in cost_configs:
        # Создаём действия с разными стоимостями
        modified_actions = []
        for a in actions:
            modified_actions.append(GOAPAction(
                a.name,
                dict(a.preconditions),
                dict(a.effects),
                costs.get(a.name, a.cost),
            ))

        plan = goap_plan(initial_state, goal, modified_actions)
        if plan:
            total_cost = sum(costs.get(a, 0.1) for a in plan)
            print(f"  {config_name}:")
            print(f"    План: {' → '.join(plan)}")
            print(f"    Стоимость: {total_cost:.1f}")
        else:
            print(f"  {config_name}: план не найден")
        print()

    # Подзадача 4: GOAP с динамическими целями
    print("--- Подзадача 4: GOAP с динамическими целями ---\n")

    # Симуляция агента, который переключает цели
    goals = [
        GOAPGoal("собрать_урожай", {"урожай_собран": True}, priority=1.0),
        GOAPGoal("полить_растения", {"растения_политы": True}, priority=0.8),
        GOAPGoal("посадить_семена", {"семена_посажены": True}, priority=0.6),
    ]

    state = dict(initial_state)
    all_actions = list(actions)  # Копия

    print("Динамическое планирование (5 итераций):\n")
    print(f"{'Итерация':<10} {'Цель':<20} {'Приоритет':<12} {'План'}")
    print("-" * 70)

    for iteration in range(5):
        # Выбор цели по приоритету (с учётом текущего состояния)
        applicable_goals = []
        for g in goals:
            if not g.is_achieved(state):
                applicable_goals.append(g)

        if not applicable_goals:
            print(f"{iteration+1:<10} {'Все цели достигнуты!':<20}")
            break

        # Выбор лучшей цели
        best_goal = max(applicable_goals, key=lambda g: g.priority)
        plan = goap_plan(state, best_goal, all_actions)

        if plan:
            plan_str = " → ".join(plan[:4])
            if len(plan) > 4:
                plan_str += " → ..."
            print(f"{iteration+1:<10} {best_goal.name:<20} {best_goal.priority:<12.1f} {plan_str}")

            # Выполняем первое действие
            first_action = next(a for a in all_actions if a.name == plan[0])
            state = first_action.apply(state)
        else:
            print(f"{iteration+1:<10} {best_goal.name:<20} {best_goal.priority:<12.1f} (нет плана)")

    print(f"\nИтоговое состояние:")
    for key, value in state.items():
        if value != initial_state.get(key):
            print(f"  {key}: {initial_state.get(key)} → {value}")

    print()


# ============================================================
# Демо 4: Адаптивные стратегии
# ============================================================

def demo_adaptive_strategies():
    """
    Демонстрация адаптивных стратегий:
    - Context switching (переключение контекста)
    - Parameter tuning (настройка параметров)
    - Behavior blending (смешивание поведений)
    """
    print("=" * 70)
    print("ДЕМО 4: АДАПТИВНЫЕ СТРАТЕГИИ")
    print("=" * 70)

    # Подзадача 1: Context Switching
    print("\n--- Подзадача 1: Context Switching ---\n")

    class ContextManager:
        """Менеджер контекстов — переключение между наборами правил."""

        def __init__(self):
            self.contexts = {}
            self.current_context = None
            self.transition_history = []

        def add_context(self, name: str, rules: dict):
            """Добавление контекста с правилами."""
            self.contexts[name] = rules

        def set_context(self, name: str):
            """Установка текущего контекста."""
            if name in self.contexts:
                if self.current_context:
                    self.transition_history.append(
                        (self.current_context, name, time.time())
                    )
                self.current_context = name

        def get_rule(self, key: str):
            """Получение правила из текущего контекста."""
            if self.current_context and self.current_context in self.contexts:
                return self.contexts[self.current_context].get(key)
            return None

        def auto_select(self, environment: dict) -> str:
            """
            Автоматический выбор контекста на основе среды.
            Правило: выбирать контекст с наибольшим количеством совпадений.
            """
            best_context = None
            best_score = -1

            for name, rules in self.contexts.items():
                score = 0
                for key, condition in rules.items():
                    if key in environment:
                        if callable(condition):
                            if condition(environment[key]):
                                score += 1
                        elif environment[key] == condition:
                            score += 1

                if score > best_score:
                    best_score = score
                    best_context = name

            return best_context

    # Пример: NPC с разными контекстами поведения
    manager = ContextManager()

    manager.add_context("мир", {
        "скорость": lambda x: x < 0.3,
        "настроение": "спокойное",
        "приоритет": "исследование",
    })

    manager.add_context("бой", {
        "скорость": lambda x: x > 0.7,
        "настроение": "агрессивное",
        "приоритет": "защита",
    })

    manager.add_context("побег", {
        "здоровье": lambda x: x < 0.3,
        "настроение": "паника",
        "приоритет": "выживание",
    })

    print("Доступные контексты:")
    for name, rules in manager.contexts.items():
        print(f"  {name}: {rules}")

    # Автоматическое переключение
    print("\nАвтоматическое переключение контекстов:\n")
    scenarios = [
        {"скорость": 0.1, "здоровье": 0.9, "настроение": "спокойное"},
        {"скорость": 0.8, "здоровье": 0.7, "настроение": "боевое"},
        {"скорость": 0.5, "здоровье": 0.2, "настроение": "усталость"},
        {"скорость": 0.2, "здоровье": 0.8, "настроение": "спокойное"},
    ]

    for i, env in enumerate(scenarios):
        context = manager.auto_select(env)
        manager.set_context(context)
        print(f"  Сценарий {i+1}: {env}")
        print(f"  → Контекст: {context}")
        print(f"  → Приоритет: {manager.get_rule('приоритет')}")
        print()

    # Подзадача 2: Parameter Tuning
    print("--- Подзадача 2: Parameter Tuning ---\n")

    class ParameterTuner:
        """Настройка параметров в реальном времени."""

        def __init__(self):
            self.parameters = {}
            self.history = []

        def add_parameter(self, name: str, value: float, min_val: float, max_val: float):
            """Добавление параметра с границами."""
            self.parameters[name] = {
                "value": value,
                "min": min_val,
                "max": max_val,
                "step": (max_val - min_val) * 0.1,
            }

        def get(self, name: str) -> float:
            return self.parameters[name]["value"]

        def adjust(self, name: str, feedback: float):
            """
            Настройка параметра на основе обратной связи.
            feedback > 0 → увеличить параметр
            feedback < 0 → уменьшить параметр
            """
            param = self.parameters[name]
            # Адаптивный шаг: чем увереннее обратная связь, тем больше шаг
            step = param["step"] * abs(feedback)
            param["value"] += step * (1 if feedback > 0 else -1)
            param["value"] = max(param["min"], min(param["max"], param["value"]))

            self.history.append({
                "parameter": name,
                "value": param["value"],
                "feedback": feedback,
            })

    tuner = ParameterTuner()
    tuner.add_parameter("скорость_атаки", 1.0, 0.5, 3.0)
    tuner.add_parameter("дальность_обзора", 5.0, 2.0, 10.0)
    tuner.add_parameter("агрессивность", 0.5, 0.0, 1.0)

    print("Настройка параметров NPC (10 итераций):\n")
    print(f"{'Итер.':<8} {'Скорость':<12} {'Обзор':<12} {'Агресс.':<12} {'Обратная связь'}")
    print("-" * 60)

    for i in range(10):
        speed = tuner.get("скорость_атаки")
        view = tuner.get("дальность_обзора")
        aggr = tuner.get("агрессивность")

        # Обратная связь от среды
        feedback_speed = random.uniform(-0.5, 0.5)
        feedback_view = random.uniform(-0.3, 0.3)
        feedback_aggr = random.uniform(-0.4, 0.4)

        tuner.adjust("скорость_атаки", feedback_speed)
        tuner.adjust("дальность_обзора", feedback_view)
        tuner.adjust("агрессивность", feedback_aggr)

        print(f"{i+1:<8} {speed:<12.4f} {view:<12.4f} {aggr:<12.4f} "
              f"спд={feedback_speed:+.3f} обз={feedback_view:+.3f} агр={feedback_aggr:+.3f}")

    # Подзадача 3: Behavior Blending
    print("\n--- Подзадача 3: Behavior Blending ---\n")

    class BehaviorBlender:
        """
        Смешивание поведений — комбинирование нескольких стратегий
        с весами, зависящими от контекста.
        """

        def __init__(self):
            self.behaviors = {}  # Имя → функция поведения
            self.weights = {}  # Имя → текущий вес

        def add_behavior(self, name: str, behavior_fn, initial_weight: float = 0.5):
            """Добавление поведения."""
            self.behaviors[name] = behavior_fn
            self.weights[name] = initial_weight

        def normalize_weights(self):
            """Нормализация весов (сумма = 1)."""
            total = sum(self.weights.values())
            if total > 0:
                for name in self.weights:
                    self.weights[name] /= total

        def blend(self, context: dict) -> dict:
            """
            Смешивание результатов всех поведений.
            Результат = sum(weight_i * behavior_i(context))
            """
            self.normalize_weights()
            blended = {}

            for name, behavior_fn in self.behaviors.items():
                result = behavior_fn(context)
                weight = self.weights[name]

                for key, value in result.items():
                    if key not in blended:
                        blended[key] = 0.0
                    blended[key] += weight * value

            return blended

        def adjust_weights(self, context: dict, feedback: dict):
            """Настройка весов на основе обратной связи."""
            for name, value in feedback.items():
                if name in self.weights:
                    self.weights[name] += value
                    self.weights[name] = max(0.01, min(2.0, self.weights[name]))

    blender = BehaviorBlender()

    # Определяем поведения
    def aggressive_behavior(ctx):
        return {"атака": 0.8, "защита": 0.3, "скорость": 0.9}

    def defensive_behavior(ctx):
        return {"атака": 0.3, "защита": 0.9, "скорость": 0.5}

    def stealth_behavior(ctx):
        return {"атака": 0.5, "защита": 0.4, "скорость": 0.7}

    blender.add_behavior("агрессивное", aggressive_behavior, 0.5)
    blender.add_behavior("защитное", defensive_behavior, 0.3)
    blender.add_behavior("скрытное", stealth_behavior, 0.2)

    print("Смешивание поведений (5 сценариев):\n")

    scenarios = [
        {"враг_рядом": True, "здоровье": 0.8, "союзники": 2},
        {"враг_рядом": False, "здоровье": 0.3, "союзники": 0},
        {"враг_рядом": True, "здоровье": 0.2, "союзники": 1},
        {"враг_рядом": False, "здоровье": 0.9, "союзники": 3},
        {"враг_рядом": True, "здоровье": 0.5, "союзники": 1},
    ]

    for i, ctx in enumerate(scenarios):
        # Адаптация весов на основе контекста
        if ctx["здоровье"] < 0.3:
            blender.adjust_weights(ctx, {"агрессивное": -0.1, "защитное": 0.15, "скрытное": 0.05})
        elif ctx["враг_рядом"]:
            blender.adjust_weights(ctx, {"агрессивное": 0.1, "защитное": -0.05, "скрытное": -0.05})
        else:
            blender.adjust_weights(ctx, {"агрессивное": -0.05, "защитное": -0.05, "скрытное": 0.1})

        blender.normalize_weights()
        result = blender.blend(ctx)

        print(f"Сценарий {i+1}: здоровье={ctx['здоровье']}, враг={'да' if ctx['враг_рядом'] else 'нет'}")
        print(f"  Веса: агр={blender.weights['агрессивное']:.2f}, "
              f"защ={blender.weights['защитное']:.2f}, "
              f"скр={blender.weights['скрытное']:.2f}")
        print(f"  Результат: атака={result['атака']:.3f}, "
              f"защита={result['защита']:.3f}, "
              f"скорость={result['скорость']:.3f}")
        print()

    # Подзадача 4: Полная адаптивная система
    print("--- Подзадача 4: Полная адаптивная система ---\n")

    class AdaptiveAgent:
        """Полностью адаптивный агент, комбинирующий все подходы."""

        def __init__(self):
            self.context_manager = ContextManager()
            self.parameter_tuner = ParameterTuner()
            self.behavior_blender = BehaviorBlender()
            self.performance_history = []

        def setup(self):
            """Настройка агента."""
            # Контексты
            self.context_manager.add_context("исследование", {"режим": "explore"})
            self.context_manager.add_context("бой", {"режим": "combat"})
            self.context_manager.add_context("отдых", {"режим": "rest"})

            # Параметры
            self.parameter_tuner.add_parameter("скорость", 1.0, 0.5, 2.0)
            self.parameter_tuner.add_parameter("осторожность", 0.5, 0.0, 1.0)

            # Поведения
            def explore_behavior(ctx):
                return {"двинуться": 0.8, "осмотреться": 0.6, "взаимодействие": 0.3}

            def combat_behavior(ctx):
                return {"двинуться": 0.5, "атаковать": 0.9, "защититься": 0.7}

            def rest_behavior(ctx):
                return {"двинуться": 0.1, "восстановиться": 0.9, "поесть": 0.5}

            self.behavior_blender.add_behavior("исследование", explore_behavior, 0.4)
            self.behavior_blender.add_behavior("бой", combat_behavior, 0.3)
            self.behavior_blender.add_behavior("отдых", rest_behavior, 0.3)

        def step(self, environment: dict) -> dict:
            """Один шаг работы агента."""
            # 1. Выбор контекста
            context = self.context_manager.auto_select(environment)
            self.context_manager.set_context(context)

            # 2. Смешивание поведений
            blended = self.behavior_blender.blend(environment)

            # 3. Получение параметров
            speed = self.parameter_tuner.get("скорость")
            caution = self.parameter_tuner.get("осторожность")

            # 4. Формирование результата
            result = {
                "контекст": context,
                "поведение": blended,
                "скорость": speed,
                "осторожность": caution,
            }

            # 5. Оценка производительности
            performance = blended.get("двинуться", 0) + blended.get("атаковать", 0) * 0.5
            self.performance_history.append(performance)

            # 6. Настройка параметров
            self.parameter_tuner.adjust("скорость", random.uniform(-0.1, 0.1))

            return result

    agent = AdaptiveAgent()
    agent.setup()

    print("Симуляция адаптивного агента (10 шагов):\n")
    print(f"{'Шаг':<6} {'Контекст':<15} {'Скорость':<10} {'Осторож.':<10} {'Производ.'}")
    print("-" * 60)

    environments = [
        {"режим": "explore", "опасность": 0.1},
        {"режим": "explore", "опасность": 0.2},
        {"режим": "combat", "опасность": 0.8},
        {"режим": "combat", "опасность": 0.9},
        {"режим": "combat", "опасность": 0.7},
        {"режим": "rest", "опасность": 0.1},
        {"режим": "rest", "опасность": 0.05},
        {"режим": "explore", "опасность": 0.15},
        {"режим": "explore", "опасность": 0.1},
        {"режим": "rest", "опасность": 0.0},
    ]

    for i, env in enumerate(environments):
        result = agent.step(env)
        perf = agent.performance_history[-1]
        print(f"{i+1:<6} {result['контекст']:<15} {result['скорость']:<10.4f} "
              f"{result['осторожность']:<10.4f} {perf:.4f}")

    # Средняя производительность
    avg_perf = sum(agent.performance_history) / len(agent.performance_history)
    print(f"\nСредняя производительность: {avg_perf:.4f}")

    # График производительности
    print("\nПроизводительность по шагам:")
    for i, perf in enumerate(agent.performance_history):
        bar = "█" * int(perf * 30)
        print(f"  {i+1:2d} {perf:.4f} {bar}")

    print()


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    demo_behavior_trees()
    demo_utility_systems()
    demo_goap()
    demo_adaptive_strategies()
