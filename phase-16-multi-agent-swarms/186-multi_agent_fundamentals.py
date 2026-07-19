"""
186 — Multi-Agent Fundamentals: общества агентов, протоколы коммуникации, координация

Темы:
  1. Agent Societies — типы агентов, среды, паттерны взаимодействия
  2. Communication Protocols — FIPA ACL, перформативы, структура сообщений
  3. Coordination Mechanisms — совместные намерения, общие планы, социальные законы
  4. Multi-Agent Taxonomy — кооперативные vs конкурентные, централизованные vs декентрализованные

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import itertools

random.seed(42)


# ─────────────────────────── Демо 1: Общества агентов ───────────────────────────

def demo_agent_societies():
    """Демонстрация типов агентов, сред и паттернов взаимодействия."""
    print("=" * 70)
    print("ДЕМО 1: Agent Societies — типы агентов, среды, паттерны взаимодействия")
    print("=" * 70)

    # 1.1 Типы агентов: реактивные, целенаправленные, рациональные
    class ReactiveAgent:
        """Реагентный агент — действует по правилу刺激→реакция."""
        def __init__(self, name, rules):
            self.name = name
            self.rules = rules  # {состояние: действие}

        def decide(self, perception):
            return self.rules.get(perception, "ждать")

    class DeliberativeAgent:
        """Целенаправленный агент — планирует действия на основе модели мира."""
        def __init__(self, name, goal, model):
            self.name = name
            self.goal = goal
            self.model = model  # {состояние: {действие: следующее_состояние}}

        def plan(self, current_state):
            # Простой поиск в глубину до достижения цели
            path = []
            visited = set()
            state = current_state
            for _ in range(10):  # ограничение глубины
                if state == self.goal:
                    return path
                if state in visited:
                    break
                visited.add(state)
                transitions = self.model.get(state, {})
                if not transitions:
                    break
                next_state = min(transitions.keys(), key=lambda s: abs(hash(s) - hash(self.goal)))
                action = transitions[next_state]
                path.append((action, next_state))
                state = next_state
            return path if state == self.goal else []

    # Демонстрация реактивного агента
    reactive_rules = {
        "огонь": "тушить",
        "вода": "плыть",
        "враг": "бежать",
        "добыча": "собирать",
    }
    robot = ReactiveAgent("Робот-1", reactive_rules)
    perceptions = ["огонь", "добыча", "вода", "враг", "солнце"]
    print("Реактивный агент — правила刺激→реакция:")
    for perc in perceptions:
        action = robot.decide(perc)
        print(f"  Восприятие '{perc}' → Действие: '{action}'")

    # Демонстрация целенаправленного агента
    world_model = {
        "дом": {"идти_в_магазин": "магазин", "идти_в_парк": "парк"},
        "магазин": {"купить_еду": "дом", "купить_вещи": "дом"},
        "парк": {"гулять": "дом", "читать": "парк"},
    }
    planner = DeliberativeAgent("Планировщик-1", goal="магазин", model=world_model)
    plan = planner.plan("дом")
    print("\nЦеленаправленный агент — планирование пути:")
    print(f"  Начальное состояние: 'дом', Цель: 'магазин'")
    for i, (action, state) in enumerate(plan, 1):
        print(f"  Шаг {i}: {action} → состояние: {state}")

    # 1.2 Типы сред: дискретная, непрерывная, частично наблюдаемая
    print("\nТипы сред:")
    # Дискретная среда — сетка
    grid_size = 5
    agent_pos = (0, 0)
    goal_pos = (4, 4)
    print(f"  Дискретная среда: сетка {grid_size}x{grid_size}, агент={agent_pos}, цель={goal_pos}")

    # Манхэттенское расстояние — эвристика для планирования
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    print(f"  Расстояние Манхэттена: {manhattan_distance(agent_pos, goal_pos)}")

    # 1.3 Паттерны взаимодействия: дублирование, чередование, иерархия
    print("\nПаттерны взаимодействия:")

    # Дублирование — все агенты получают одну задачу
    tasks = ["анализ_данных", "обучение_модели", "валидация", "деплой"]
    agents_dup = [f"Агент_{i}" for i in range(1, 4)]
    print("  Дублирование (broadcast):")
    for agent in agents_dup:
        task = random.choice(tasks)
        print(f"    {agent} получает задачу: {task}")

    # Чередование — агенты выполняют по очереди
    print("  Чередование (round-robin):")
    for i, agent in enumerate(agents_dup):
        task = tasks[i % len(tasks)]
        print(f"    {agent} выполняет: {task}")

    # Иерархия — координатор распределяет задачи
    print("  Иерархия (координатор):")
    coordinator = "Координатор"
    for agent in agents_dup:
        task = random.choice(tasks)
        print(f"    {coordinator} → {agent}: {task}")

    # 1.4 Простая среда с множеством агентов
    print("\nМногоагентная среда (4 агента, 3 типа):")
    agent_types = ["исполнитель", "наблюдатель", "координатор"]
    agents = []
    for i in range(4):
        agent_type = agent_types[i % 3]
        agents.append({"id": i, "type": agent_type, "pos": (random.randint(0, 4), random.randint(0, 4))})
    for a in agents:
        print(f"  Агент {a['id']}: тип={a['type']}, позиция={a['pos']}")

    print()


# ─────────────────────────── Демо 2: Протоколы коммуникации ───────────────────────────

def demo_communication_protocols():
    """Демонстрация FIPA ACL, перформативов, структуры сообщений."""
    print("=" * 70)
    print("ДЕМО 2: Communication Protocols — FIPA ACL, перформативы, структура сообщений")
    print("=" * 70)

    # 2.1 FIPA ACL — формат сообщений
    # ACL message = performativ + sender + receiver + content + language + ontology
    print("FIPA ACL Message Structure:")
    print("-" * 50)

    acl_message = {
        "performative": "inform",
        "sender": "agent@platform",
        "receiver": "coordinator@platform",
        "content": {"price": 100, "item": "widget"},
        "language": "json",
        "ontology": "trading",
        "conversation-id": hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
        "reply-with": "msg_001",
    }
    print(json.dumps(acl_message, indent=2, ensure_ascii=False))

    # 2.2 Перформативы — типы речевых актов
    print("\nПерформативы FIPA:")
    performatives = {
        "inform": "аргент сообщает факт",
        "request": "аргент просит действие",
        "propose": "аргент предлагает контракт",
        "agree": "аргент соглашается на действие",
        "refuse": "аргент отказывается",
        "confirm": "аргент подтверждает истинность",
        "query-if": "аргент спрашивает истинность",
        "cfp": "призыв к предложению (call for proposal)",
    }
    for perf, desc in performatives.items():
        print(f"  {perf:15s} — {desc}")

    # 2.3 Протокол request-inform
    print("\nПротокол request-inform:")
    print("-" * 50)

    conversation = []
    # Сообщение 1: request от A к B
    msg1 = {
        "performative": "request",
        "sender": "A",
        "receiver": "B",
        "content": "provide_weather_data",
        "time": 0.1,
    }
    conversation.append(msg1)
    print(f"  [t={msg1['time']:.1f}] {msg1['sender']} → {msg1['receiver']}: {msg1['performative']}({msg1['content']})")

    # Сообщение 2: agree от B к A
    msg2 = {
        "performative": "agree",
        "sender": "B",
        "receiver": "A",
        "content": "will_provide_weather",
        "time": 0.2,
    }
    conversation.append(msg2)
    print(f"  [t={msg2['time']:.1f}] {msg2['sender']} → {msg2['receiver']}: {msg2['performative']}({msg2['content']})")

    # Сообщение 3: inform от B к A
    msg3 = {
        "performative": "inform",
        "sender": "B",
        "receiver": "A",
        "content": {"temperature": 22, "humidity": 65, "condition": "sunny"},
        "time": 0.3,
    }
    conversation.append(msg3)
    print(f"  [t={msg3['time']:.1f}] {msg3['sender']} → {msg3['receiver']}: {msg3['performative']}(...)")

    # Сообщение 4: confirm от A к B
    msg4 = {
        "performative": "confirm",
        "sender": "A",
        "receiver": "B",
        "content": "data_received",
        "time": 0.4,
    }
    conversation.append(msg4)
    print(f"  [t={msg4['time']:.1f}] {msg4['sender']} → {msg4['receiver']}: {msg4['performative']}({msg4['content']})")

    # 2.4 Валидация сообщений
    print("\nВалидация ACL-сообщений:")

    def validate_acl(msg, known_performatives):
        """Проверка ACL-сообщения на соответствие стандарту."""
        errors = []
        if msg.get("performative") not in known_performatives:
            errors.append(f"неизвестный перформатив: {msg.get('performative')}")
        if "sender" not in msg:
            errors.append("отсутствует sender")
        if "receiver" not in msg:
            errors.append("отсутствует receiver")
        if "content" not in msg:
            errors.append("отсутствует content")
        return errors

    valid_perf = set(performatives.keys())
    test_messages = [
        {"performative": "inform", "sender": "A", "receiver": "B", "content": "hello"},
        {"performative": "unknown", "sender": "A", "receiver": "B", "content": "test"},
        {"performative": "request", "content": "do_something"},
    ]
    for i, msg in enumerate(test_messages, 1):
        errors = validate_acl(msg, valid_perf)
        status = "ВАЛИДНО" if not errors else f"ОШИБКИ: {errors}"
        print(f"  Сообщение {i}: {status}")

    print()


# ─────────────────────────── Демо 3: Механизмы координации ───────────────────────────

def demo_coordination_mechanisms():
    """Демонстрация совместных намерений, общих планов, социальных законов."""
    print("=" * 70)
    print("ДЕМО 3: Coordination Mechanisms — совместные намерения, общие планы")
    print("=" * 70)

    # 3.1 Совместные намерения (Joint Intentions)
    print("Совместные намерения (Joint Intentions):")
    print("-" * 50)

    class JointIntention:
        """Модель совместного намерения по Cohen & Levesque."""
        def __init__(self, agents, goal):
            self.agents = agents
            self.goal = goal
            self.commitments = {a: "believe_goal" for a in agents}
            self.status = "active"

        def update(self, agent, new_commitment):
            if agent in self.commitments:
                self.commitments[agent] = new_commitment
                self._check_completion()

        def _check_completion(self):
            if all(c == "completed" for c in self.commitments.values()):
                self.status = "completed"
            elif any(c == "abandoned" for c in self.commitments.values()):
                self.status = "degraded"

        def report(self):
            print(f"  Намерение: '{self.goal}'")
            print(f"  Агенты: {self.agents}")
            for a, c in self.commitments.items():
                print(f"    {a}: {c}")
            print(f"  Статус: {self.status}")

    ji = JointIntention(["Робот_А", "Робот_Б"], "построить стену")
    ji.report()
    ji.update("Робот_А", "working")
    ji.update("Робот_Б", "working")
    print("\n  После начала работы:")
    ji.report()
    ji.update("Робот_А", "completed")
    ji.update("Робот_Б", "completed")
    print("\n  После завершения:")
    ji.report()

    # 3.2 Общий план (Shared Plan)
    print("\nОбщий план (Shared Plan):")
    print("-" * 50)

    class SharedPlan:
        """Общий план совместного действия."""
        def __init__(self, goal, steps):
            self.goal = goal
            self.steps = steps  # [(шаг, ответственный, prereq)]
            self.completed = set()

        def can_execute(self, step_idx):
            _, _, prereqs = self.steps[step_idx]
            return all(p in self.completed for p in prereqs)

        def execute(self, step_idx):
            step_name, agent, _ = self.steps[step_idx]
            self.completed.add(step_name)
            return f"  {agent} выполняет '{step_name}'"

    plan = SharedPlan("установка_ПО", [
        ("скачать_установщик", "Админ_1", []),
        ("проверить_целостность", "Админ_2", ["скачать_установщик"]),
        ("установить", "Админ_1", ["скачать_установщик", "проверить_целостность"]),
        ("настроить", "Админ_2", ["установить"]),
        ("протестировать", "Админ_1", ["настроить"]),
    ])

    print(f"  Цель: {plan.goal}")
    print(f"  Шаги: {len(plan.steps)}")
    executed = 0
    for i in range(len(plan.steps)):
        if plan.can_execute(i):
            result = plan.execute(i)
            print(result)
            executed += 1
        else:
            step_name, agent, prereqs = plan.steps[i]
            missing = [p for p in prereqs if p not in plan.completed]
            print(f"  БЛОКИРОВКА: '{step_name}' ждёт: {missing}")

    # 3.3 Социальные законы (Social Laws)
    print("\nСоциальные законы (Social Laws):")
    print("-" * 50)

    social_laws = [
        {"law": "AL1", "desc": "Не пересекать зону другого агента", "priority": 1},
        {"law": "AL2", "desc": "Сообщать о завершении задачи координатору", "priority": 2},
        {"law": "AL3", "desc": "Приоритет выполняется по номеру задачи", "priority": 3},
        {"law": "AL4", "desc": "При конфликте — координатор решает", "priority": 4},
    ]
    for sl in social_laws:
        print(f"  Закон {sl['law']} (приоритет {sl['priority']}): {sl['desc']}")

    # 3.4 Разрешение конфликтов
    print("\nРазрешение конфликтов (mediation):")

    conflicts = [
        {"agents": ("Робот_1", "Робот_2"), "resource": "сверло", "time_slot": "10:00-11:00"},
        {"agents": ("Робот_1", "Робот_3"), "resource": "кран", "time_slot": "14:00-15:00"},
    ]

    def resolve_conflict(conflict, laws):
        """Простое разрешение конфликта по приоритету законов."""
        a1, a2 = conflict["agents"]
        resource = conflict["resource"]
        slot = conflict["time_slot"]
        # Рандомное назначение с учётом приоритетов
        winner = random.choice([a1, a2])
        loser = a1 if winner == a2 else a2
        return {
            "resource": resource,
            "time_slot": slot,
            "winner": winner,
            "loser": loser,
            "action": f"{winner} использует {resource} в {slot}, {loser} перенаправляется"
        }

    for i, conflict in enumerate(conflicts, 1):
        resolution = resolve_conflict(conflict, social_laws)
        print(f"  Конфликт {i}: {resolution['action']}")

    print()


# ─────────────────────────── Демо 4: Таксономия мультиагентных систем ───────────────────────────

def demo_multi_agent_taxonomy():
    """Демонстрация кооперативных vs конкурентных, централизованных vs декентрализованных систем."""
    print("=" * 70)
    print("ДЕМО 4: Multi-Agent Taxonomy — кооперативные vs конкурентные")
    print("=" * 70)

    # 4.1 Кооперативная vs конкурентная
    print("Кооперативные vs конкурентные агенты:")
    print("-" * 50)

    # Кооперативная: агенты делят награду
    def cooperative_reward(rewards):
        """Средняя награда — все получают одинаково."""
        avg = sum(rewards) / len(rewards)
        return [avg] * len(rewards)

    # Конкурентная: только победитель получает награду
    def competitive_reward(rewards):
        """Только лучший получает награду."""
        max_r = max(rewards)
        return [max_r if r == max_r else 0 for r in rewards]

    agent_rewards = [0.8, 0.6, 0.9, 0.7]
    coop = cooperative_reward(agent_rewards)
    comp = competitive_reward(agent_rewards)
    print(f"  Исходные награды: {agent_rewards}")
    print(f"  Кооперативная:    {[round(r, 2) for r in coop]}")
    print(f"  Конкурентная:     {comp}")

    # 4.2 Централизованная vs декентрализованная
    print("\nЦентрализованная vs декентрализованная архитектура:")
    print("-" * 50)

    # Централизованная — один координатор
    class CentralizedCoordinator:
        def __init__(self, n_agents):
            self.n_agents = n_agents
            self.tasks = ["задача_A", "задача_B", "задача_C", "задача_D"]
            self.allocation = {}

        def allocate(self):
            for i in range(self.n_agents):
                task = self.tasks[i % len(self.tasks)]
                self.allocation[f"Агент_{i}"] = task
            return self.allocation

    central = CentralizedCoordinator(4)
    alloc = central.allocate()
    print("  Централизованная (координатор):")
    for agent, task in alloc.items():
        print(f"    {agent} → {task}")

    # Декентрализованная — агенты договариваются сами
    def decentralized_negotiate(agents, tasks, rounds=3):
        """Простая decentralized торговля за задачи."""
        allocation = {}
        available_tasks = list(tasks)
        random.shuffle(available_tasks)
        for agent in agents:
            if available_tasks:
                # Агент выбирает лучшую доступную задачу
                chosen = available_tasks.pop(0)
                allocation[agent] = chosen
        return allocation

    agents = [f"Агент_{i}" for i in range(4)]
    tasks = ["задача_A", "задача_B", "задача_C", "задача_D"]
    dec_alloc = decentralized_negotiate(agents, tasks)
    print("\n  Декентрализованная (самоорганизация):")
    for agent, task in dec_alloc.items():
        print(f"    {agent} → {task}")

    # 4.3 Подтипы кооперативных систем
    print("\nПодтипы кооперативных мультиагентных систем:")
    subtypes = {
        "Team": "агенты делят одну общую цель",
        "Society": "агенты имеют разные цели, но соблюдают нормы",
        "Organization": "иерархическая структура ролей",
        "Market": "экономические механизмы координации",
    }
    for name, desc in subtypes.items():
        print(f"  {name:15s} — {desc}")

    # 4.4 Паттерны распределённого принятия решений
    print("\nПаттерны распределённого принятия решений:")
    patterns = [
        ("majority_vote", "простое голосование большинством"),
        ("consensus", "полный консенсус (все соглашаются)"),
        ("bidding", "аукционный механизм (кто больше заплатит)"),
        ("blackboard", "общая доска — агенты читают/пишут"),
    ]
    for name, desc in patterns:
        print(f"  {name:20s} — {desc}")

    # 4.5 Сравнение архитектур
    print("\nСравнение архитектур:")
    architectures = [
        ("MAS централизованный", "простая координация", "точка отказа", "низкая масштабируемость"),
        ("MAS декентрализованный", "устойчивость", "сложная отладка", "высокая масштабируемость"),
        ("MAS гибридный", "баланс", "средняя сложность", "средняя масштабируемость"),
    ]
    print(f"  {'Архитектура':30s} | {'Плюс':20s} | {'Минус':20s} | {'Масштаб':20s}")
    print(f"  {'-'*30}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    for arch, pro, con, scale in architectures:
        print(f"  {arch:30s} | {pro:20s} | {con:20s} | {scale:20s}")

    print()


if __name__ == "__main__":
    demo_agent_societies()
    demo_communication_protocols()
    demo_coordination_mechanisms()
    demo_multi_agent_taxonomy()
