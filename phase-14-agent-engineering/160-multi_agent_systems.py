"""160 — Multi-Agent Systems: кооперация, коммуникация, координация

Темы:
  1. Agent Communication — передача сообщений, общий черный доска, протоколы
  2. Cooperation Patterns — последовательная, параллельная, иерархическая, P2P кооперация
  3. Task Allocation — назначение ролей, балансировка нагрузки, соответствие способностей
  4. Conflict Resolution — голосование, переговоры, алгоритмы консенсуса

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


# ─────────────────────────────────────────────────────────────
# 1. Agent Communication — передача сообщений, протоколы
# ─────────────────────────────────────────────────────────────

class Message:
    """Сообщение между агентами."""

    def __init__(self, sender: str, receiver: str, content: str, msg_type: str = "inform"):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.msg_type = msg_type  # inform, request, propose, accept, reject
        self.timestamp = time.time()
        self.msg_id = hashlib.md5(f"{sender}{receiver}{content}".encode()).hexdigest()[:8]

    def __repr__(self):
        return f"Msg({self.sender}→{self.receiver}: {self.msg_type})"


class Blackboard:
    """Общий черный доска (Blackboard) — разделяемое хранилище для агентов."""

    def __init__(self):
        self.entries = {}  # {ключ: (значение, автор)}
        self.history = []

    def write(self, key: str, value, author: str):
        """Записать данные на доску."""
        self.entries[key] = (value, author)
        self.history.append(("write", key, author))

    def read(self, key: str):
        """Прочитать данные с доски."""
        return self.entries.get(key, (None, None))

    def list_entries(self) -> list:
        """Список всех записей на доске."""
        return [(k, v, author) for k, (v, author) in self.entries.items()]


class Agent:
    """Базовый агент с почтовым ящиком."""

    def __init__(self, name: str, role: str = "worker"):
        self.name = name
        self.role = role
        self.mailbox = []  # Входящие сообщения
        self.blackboard = None  # Ссылка на общую доску

    def send(self, receiver: "Agent", content: str, msg_type: str = "inform"):
        """Отправить сообщение другому агенту."""
        msg = Message(self.name, receiver.name, content, msg_type)
        receiver.mailbox.append(msg)
        return msg

    def check_mail(self) -> list:
        """Проверить почтовый ящик."""
        messages = self.mailbox.copy()
        self.mailbox.clear()
        return messages

    def attach_blackboard(self, bb: Blackboard):
        """Подключить агента к общей доске."""
        self.blackboard = bb

    def write_to_blackboard(self, key: str, value):
        """Записать данные на общую доску."""
        if self.blackboard:
            self.blackboard.write(key, value, self.name)

    def read_from_blackboard(self, key: str):
        """Прочитать данные с общей доски."""
        if self.blackboard:
            return self.blackboard.read(key)
        return None


def demo_agent_communication():
    """Демонстрация: передача сообщений, черный доска, протоколы."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: КОММУНИКАЦИЯ АГЕНТОВ (Agent Communication)")
    print("=" * 70)

    # --- Пример 1: Прямая передача сообщений ---
    print("\n--- Пример 1: Прямая передача сообщений (message passing) ---")
    agent_a = Agent("Agent_A", "researcher")
    agent_b = Agent("Agent_B", "analyst")
    agent_c = Agent("Agent_C", "writer")

    # Протокол FIPA-подобного обмена
    agent_a.send(agent_b, "Найди данные по продажам за Q3", msg_type="request")
    agent_b.send(agent_a, "Данные готовы: revenue = 1.2M", msg_type="inform")
    agent_b.send(agent_c, "Вот данные для отчёта: revenue = 1.2M", msg_type="inform")
    agent_c.send(agent_b, "Отчёт готов, спасибо за данные!", msg_type="accept")

    print("  Протокол обмена сообщениями (FIPA-style):")
    for agent in [agent_a, agent_b, agent_c]:
        for msg in agent.check_mail():
            print(f"    {msg.sender} → {msg.receiver} [{msg.msg_type}]: {msg.content[:50]}")

    # --- Пример 2: Черный доска ---
    print("\n--- Пример 2: Общий черный доска (Blackboard) ---")
    bb = Blackboard()
    agent_a.attach_blackboard(bb)
    agent_b.attach_blackboard(bb)
    agent_c.attach_blackboard(bb)

    # Агенты записывают свои результаты на доску
    agent_a.write_to_blackboard("task_1_status", "completed")
    agent_b.write_to_blackboard("task_2_status", "in_progress")
    agent_c.write_to_blackboard("task_1_result", "500 строк данных")

    print("  Записи на доске:")
    for key, value, author in bb.list_entries():
        print(f"    [{author}] {key} = {value}")

    # Все агенты видят общее состояние
    for agent in [agent_a, agent_b, agent_c]:
        result = agent.read_from_blackboard("task_1_status")
        print(f"    {agent.name} видит task_1_status: {result[0]}")

    # --- Пример 3: Протоколы обмена ---
    print("\n--- Пример 3: Протоколы обмена (protocols) ---")
    protocols = {
        "request-response": ["request", "inform"],
        "contract-net": ["call_for_proposal", "proposal", "accept", "reject"],
        "publish-subscribe": ["subscribe", "notify"],
        "auction": ["bid", "highest_bid", "winner"],
    }
    print("  Типы агентных протоколов:")
    for proto, steps in protocols.items():
        print(f"    {proto}: {' → '.join(steps)}")

    # Демонстрация contract-net протокола
    print("\n  Пример contract-net протокола:")
    manager = Agent("Manager", "manager")
    worker1 = Agent("Worker_1", "worker")
    worker2 = Agent("Worker_2", "worker")

    # Менеджер отправляет call for proposal
    manager.send(worker1, "call_for_proposal: обработать изображения", msg_type="call_for_proposal")
    manager.send(worker2, "call_for_proposal: обработать изображения", msg_type="call_for_proposal")

    # Рабочие отвечают proposal
    worker1.send(manager, "proposal: GPU кластер, 2 часа", msg_type="proposal")
    worker2.send(manager, "proposal: CPU сервер, 5 часов", msg_type="proposal")

    # Менеджер выбирает лучшее предложение
    all_msgs = manager.check_mail()
    worker1_msg = all_msgs[0]
    worker2_msg = all_msgs[1]
    print(f"    Менеджер получил: {worker1_msg.content} | {worker2_msg.content}")
    manager.send(worker1, "accept: ваше предложение принято", msg_type="accept")
    manager.send(worker2, "reject: лучшее предложение уже выбрано", msg_type="reject")

    # --- Пример 4: Многоагентный диалог ---
    print("\n--- Пример 4: Многоагентный диалог (3 агента) ---")
    agents = [Agent(f"Agent_{i}") for i in range(3)]
    # Агент 0 — инициатор
    agents[0].send(agents[1], "Можешь проверить код на ошибки?", msg_type="request")
    agents[1].send(agents[2], "Нужна помощь с ревью кода", msg_type="request")
    agents[2].send(agents[0], "Я помогу с ревью", msg_type="inform")
    agents[1].send(agents[0], "Нашёл 3 ошибки в файле main.py", msg_type="inform")

    print("  Цепочка сообщений:")
    all_msgs = []
    for a in agents:
        for msg in a.check_mail():
            all_msgs.append(msg)
    for msg in all_msgs:
        print(f"    {msg.sender} → {msg.receiver} [{msg.msg_type}]: {msg.content}")

    print()


# ─────────────────────────────────────────────────────────────
# 2. Cooperation Patterns — паттерны кооперации
# ─────────────────────────────────────────────────────────────

class SequentialPipeline:
    """Последовательная кооперация: агенты работают один за другим (pipeline)."""

    def __init__(self, agents: list):
        self.agents = agents

    def execute(self, initial_input):
        """Выполнить pipeline: каждый агент обрабатывает результат предыдущего."""
        result = initial_input
        for agent in self.agents:
            result = agent(result)
        return result


class ParallelWorkers:
    """Параллельная кооперация: несколько агентов работают одновременно."""

    def __init__(self, agents: list):
        self.agents = agents

    def execute(self, input_data: list) -> list:
        """Выполнить параллельно: каждый агент обрабатывает свою часть."""
        results = []
        for agent, data in zip(self.agents, input_data):
            results.append(agent(data))
        return results


class HierarchicalOrg:
    """Иерархическая кооперация: менеджер → подчинённые агенты."""

    def __init__(self, manager: dict, workers: list):
        self.manager = manager
        self.workers = workers

    def execute(self, task: str) -> dict:
        """Менеджер делит задачу, рабочие выполняют, менеджер агрегирует."""
        # Менеджер разбивает задачу на подзадачи
        subtasks = self.manager["decompose"](task)
        # Рабочие выполняют подзадачи
        results = {}
        for i, worker in enumerate(self.workers):
            subtask = subtasks[i] if i < len(subtasks) else f"default_{i}"
            results[subtask] = worker(subtask)
        # Менеджер агрегирует результаты
        final = self.manager["aggregate"](results)
        return final


class PeerToPeer:
    """P2P кооперация: агенты общаются напрямую, без центрального координатора."""

    def __init__(self, agents: list):
        self.agents = agents
        # Матрица связей: кто с кем общается
        self.connections = {}

    def connect(self, idx1: int, idx2: int):
        """Соединить двух агентов."""
        if idx1 not in self.connections:
            self.connections[idx1] = set()
        if idx2 not in self.connections:
            self.connections[idx2] = set()
        self.connections[idx1].add(idx2)
        self.connections[idx2].add(idx1)

    def broadcast(self, sender_idx: int, message: str) -> dict:
        """Агент отправляет сообщение всем связанным агентам."""
        results = {}
        if sender_idx in self.connections:
            for target_idx in self.connections[sender_idx]:
                results[target_idx] = self.agents[target_idx](message)
        return results


def demo_cooperation_patterns():
    """Демонстрация: последовательная, параллельная, иерархическая, P2P кооперация."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: ПАТТЕРНЫ КООПЕРАЦИИ (Cooperation Patterns)")
    print("=" * 70)

    # --- Пример 1: Последовательный pipeline ---
    print("\n--- Пример 1: Последовательный pipeline (sequential) ---")
    # Цепочка обработки текста
    step1 = lambda text: text.strip().lower()
    step2 = lambda text: re.sub(r'[^\w\s]', '', text)
    step3 = lambda text: ' '.join(text.split())
    step4 = lambda text: text.title()

    pipeline = SequentialPipeline([step1, step2, step3, step4])
    input_text = "  Привет,   МИР! Это   тест.  "
    result = pipeline.execute(input_text)
    print(f"  Вход: '{input_text}'")
    print(f"  Pipeline: strip+lower → remove_punct → normalize → title")
    print(f"  Выход: '{result}'")

    # --- Пример 2: Параллельные рабочие ---
    print("\n--- Пример 2: Параллельная кооперация (parallel) ---")
    # Каждый рабочий обрабатывает свою часть данных
    worker1 = lambda data: f"Обработано (CPU): {data} → нормализация"
    worker2 = lambda data: f"Обработано (GPU): {data} → классификация"
    worker3 = lambda data: f"Обработано (RAM): {data} → агрегация"

    parallel = ParallelWorkers([worker1, worker2, worker3])
    data_parts = ["часть_1", "часть_2", "часть_3"]
    results = parallel.execute(data_parts)
    print(f"  Данные разбиты на {len(data_parts)} частей:")
    for r in results:
        print(f"    {r}")
    print(f"  Скорость: {len(data_parts)}x параллелизм")

    # --- Пример 3: Иерархическая организация ---
    print("\n--- Пример 3: Иерархическая кооперация (hierarchical) ---")
    manager = {
        "decompose": lambda task: ["анализ", "разработка", "тестирование"],
        "aggregate": lambda results: f"Проект завершён: {len(results)} этапов",
    }
    worker_funcs = [
        lambda task: f"{task} — готово за 2ч",
        lambda task: f"{task} — готово за 5ч",
        lambda task: f"{task} — готово за 1ч",
    ]
    hierarchy = HierarchicalOrg(manager, worker_funcs)
    final = hierarchy.execute("Создать MVP приложения")
    print(f"  Задача: 'Создать MVP приложения'")
    print(f"  Менеджер разбил на: анализ, разработка, тестирование")
    print(f"  Результат: {final}")

    # --- Пример 4: P2P кооперация ---
    print("\n--- Пример 4: P2P кооперация (peer-to-peer) ---")
    agents = [
        lambda msg: f"Agent_0 обработал: {msg}",
        lambda msg: f"Agent_1 обработал: {msg}",
        lambda msg: f"Agent_2 обработал: {msg}",
        lambda msg: f"Agent_3 обработал: {msg}",
    ]
    p2p = PeerToPeer(agents)
    p2p.connect(0, 1)
    p2p.connect(0, 2)
    p2p.connect(1, 3)
    p2p.connect(2, 3)

    print(f"  Сеть агентов (topology):")
    print(f"    Agent_0 ↔ Agent_1, Agent_2")
    print(f"    Agent_1 ↔ Agent_0, Agent_3")
    print(f"    Agent_2 ↔ Agent_0, Agent_3")
    print(f"    Agent_3 ↔ Agent_1, Agent_2")

    # Агент 0 рассылает сообщение
    results = p2p.broadcast(0, "новые данные")
    print(f"\n  Agent_0 broadcast 'новые данные':")
    for idx, res in results.items():
        print(f"    → {res}")

    print()


# ─────────────────────────────────────────────────────────────
# 3. Task Allocation — назначение ролей, балансировка нагрузки
# ─────────────────────────────────────────────────────────────

class TaskAllocationSystem:
    """Система распределения задач между агентами."""

    def __init__(self):
        self.agents = {}
        self.tasks = []
        self.assignments = {}

    def register_agent(self, name: str, capabilities: list, load: float = 0.0):
        """Зарегистрировать агента с его способностями."""
        self.agents[name] = {
            "capabilities": capabilities,
            "current_load": load,  # Текущая нагрузка (0.0 - 1.0)
            "completed_tasks": 0,
        }

    def add_task(self, task_id: str, required_capabilities: list, priority: int = 1):
        """Добавить задачу."""
        self.tasks.append({
            "id": task_id,
            "required": required_capabilities,
            "priority": priority,
            "assigned_to": None,
        })

    def capability_match_score(self, agent_name: str, task: dict) -> float:
        """Подсчёт соответствия способностей агента задаче.
        Score = |capabilities ∩ required| / |required|"""
        agent_caps = set(self.agents[agent_name]["capabilities"])
        required_caps = set(task["required"])
        if not required_caps:
            return 0.0
        return len(agent_caps & required_caps) / len(required_caps)

    def allocate_greedy(self):
        """Жадный алгоритм распределения: назначаем задачу лучшему агенту."""
        for task in self.tasks:
            if task["assigned_to"]:
                continue
            best_agent = None
            best_score = -1
            for name, agent in self.agents.items():
                # Итоговый скор: соответствие × (1 - текущая нагрузка) × приоритет
                match = self.capability_match_score(name, task)
                load_factor = 1.0 - agent["current_load"]
                score = match * load_factor * task["priority"]
                if score > best_score:
                    best_score = score
                    best_agent = name
            if best_agent:
                task["assigned_to"] = best_agent
                self.agents[best_agent]["current_load"] = min(
                    1.0, self.agents[best_agent]["current_load"] + 0.2
                )
                self.agents[best_agent]["completed_tasks"] += 1
                self.assignments[task["id"]] = best_agent

    def get_load_distribution(self) -> dict:
        """Получить распределение нагрузки по агентам."""
        return {
            name: {
                "load": agent["current_load"],
                "tasks": agent["completed_tasks"],
                "caps": agent["capabilities"],
            }
            for name, agent in self.agents.items()
        }

    def balance_load(self):
        """Балансировка нагрузки: переназначение задач от перегруженных к свободным."""
        overloaded_threshold = 0.8
        underloaded_threshold = 0.3
        overloaded = [n for n, a in self.agents.items() if a["current_load"] > overloaded_threshold]
        underloaded = [n for n, a in self.agents.items() if a["current_load"] < underloaded_threshold]
        reassignments = []
        for task in self.tasks:
            if task["assigned_to"] in overloaded and underloaded:
                for free_agent in underloaded:
                    match = self.capability_match_score(free_agent, task)
                    if match > 0.5:
                        old = task["assigned_to"]
                        task["assigned_to"] = free_agent
                        self.agents[old]["current_load"] = max(0, self.agents[old]["current_load"] - 0.2)
                        self.agents[free_agent]["current_load"] = min(1.0, self.agents[free_agent]["current_load"] + 0.2)
                        reassignments.append((task["id"], old, free_agent))
                        break
        return reassignments


def demo_task_allocation():
    """Демонстрация: назначение ролей, балансировка нагрузки, соответствие способностей."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: РАСПРЕДЕЛЕНИЕ ЗАДАЧ (Task Allocation)")
    print("=" * 70)

    system = TaskAllocationSystem()

    # --- Пример 1: Регистрация агентов ---
    print("\n--- Пример 1: Регистрация агентов и их способностей ---")
    system.register_agent("DataScientist", ["анализ", "моделирование", "статистика"], load=0.1)
    system.register_agent("Engineer", ["разработка", "оптимизация", "тестирование"], load=0.0)
    system.register_agent("Analyst", ["анализ", "визуализация", "отчётность"], load=0.3)
    system.register_agent("DevOps", ["развёртывание", "мониторинг", "оптимизация"], load=0.5)
    for name, info in system.agents.items():
        print(f"  {name}: способности={info['capabilities']}, нагрузка={info['current_load']:.1f}")

    # --- Пример 2: Добавление и распределение задач ---
    print("\n--- Пример 2: Жадное распределение задач ---")
    tasks = [
        ("T1", ["анализ", "визуализация"], 2),
        ("T2", ["разработка", "тестирование"], 1),
        ("T3", ["развёртывание", "мониторинг"], 1),
        ("T4", ["анализ", "моделирование"], 3),
        ("T5", ["оптимизация", "тестирование"], 2),
    ]
    for task_id, caps, priority in tasks:
        system.add_task(task_id, caps, priority)

    system.allocate_greedy()

    print("  Результаты распределения:")
    for task in system.tasks:
        print(f"    {task['id']} → {task['assigned_to']} "
              f"(нужно: {task['required']}, приоритет: {task['priority']})")

    # --- Пример 3: Соответствие способностей ---
    print("\n--- Пример 3: Подсчёт соответствия способностей ---")
    print("  Формула: match = |caps ∩ required| / |required|")
    test_cases = [
        ("DataScientist", ["анализ", "статистика"]),
        ("Engineer", ["разработка", "анализ"]),
        ("Analyst", ["визуализация", "анализ", "отчётность"]),
    ]
    for agent_name, required in test_cases:
        score = system.capability_match_score(agent_name, {"required": required})
        print(f"    {agent_name} vs {required}: {score:.2f}")

    # --- Пример 4: Балансировка нагрузки ---
    print("\n--- Пример 4: Балансировка нагрузки ---")
    print("  До балансировки:")
    for name, info in system.get_load_distribution().items():
        bar = "█" * int(info["load"] * 10)
        print(f"    {name:15s}: нагрузка={info['load']:.1f} {bar} ({info['tasks']} задач)")

    # Искусственно перегрузим DevOps
    system.agents["DevOps"]["current_load"] = 0.9
    reassignments = system.balance_load()
    print("  После балансировки:")
    for name, info in system.get_load_distribution().items():
        bar = "█" * int(info["load"] * 10)
        print(f"    {name:15s}: нагрузка={info['load']:.1f} {bar} ({info['tasks']} задач)")
    if reassignments:
        print(f"  Переназначения:")
        for task_id, old, new in reassignments:
            print(f"    {task_id}: {old} → {new}")

    print()


# ─────────────────────────────────────────────────────────────
# 4. Conflict Resolution — голосование, переговоры, консенсус
# ─────────────────────────────────────────────────────────────

class VotingSystem:
    """Система голосования агентов."""

    def __init__(self):
        self.agents = []
        self.votes = {}

    def register(self, agent_name: str, weight: float = 1.0):
        """Зарегистрировать агента с весом голоса."""
        self.agents.append({"name": agent_name, "weight": weight})

    def cast_vote(self, agent_name: str, option: str):
        """Проголосовать за вариант."""
        self.votes[agent_name] = option

    def count_votes(self) -> dict:
        """Подсчитать голоса (взвешенное голосование)."""
        results = collections.defaultdict(float)
        for agent in self.agents:
            if agent["name"] in self.votes:
                option = self.votes[agent["name"]]
                results[option] += agent["weight"]
        return dict(results)

    def majority_winner(self) -> tuple:
        """Определить победителя простым большинством."""
        results = self.count_votes()
        if not results:
            return None, 0
        winner = max(results, key=results.get)
        total = sum(results.values())
        return winner, results[winner] / total if total else 0


class NegotiationProtocol:
    """Протокол переговоров между агентами."""

    def __init__(self):
        self.rounds = []
        self.max_rounds = 5

    def propose(self, round_num: int, proposals: dict) -> dict:
        """Раунд переговоров: каждый агент вносит предложение."""
        self.rounds.append(proposals)
        # Ищем общее решение (среднее по числовым предложениям)
        numeric = {}
        for agent, proposal in proposals.items():
            if isinstance(proposal, (int, float)):
                numeric[agent] = proposal
        if numeric:
            avg = sum(numeric.values()) / len(numeric)
            return {"consensus_value": avg, "agreement": True}
        return {"consensus_value": None, "agreement": False}

    def nash_bargaining(self, agent_values: list, disagreement_point: float = 0.0) -> list:
        """Модель торга Нэша: максимизация произведения выигрышей.
        max ∏(u_i(x) - u_i(d))"""
        # Простая линейная модель: каждый агент получает долю
        n = len(agent_values)
        # Нэш-решение: равное распределение при линейных предпочтениях
        total_value = sum(agent_values)
        nash_share = (total_value - disagreement_point) / n
        return [nash_share] * n


class ConsensusAlgorithm:
    """Алгоритм консенсуса (простая модель)."""

    def __init__(self, agents: list, initial_values: dict):
        self.agents = agents
        self.values = dict(initial_values)
        self.history = [dict(initial_values)]

    def iterate(self) -> dict:
        """Один шаг консенсуса: каждый агент обновляет значение как среднее соседей."""
        new_values = {}
        for agent in self.agents:
            neighbors = [v for a, v in self.values.items() if a != agent]
            if neighbors:
                new_values[agent] = sum(neighbors) / len(neighbors)
            else:
                new_values[agent] = self.values[agent]
        self.values = new_values
        self.history.append(dict(new_values))
        return new_values

    def is_converged(self, threshold: float = 0.01) -> bool:
        """Проверить, сошёлся ли алгоритм."""
        if len(self.history) < 2:
            return False
        prev = self.history[-2]
        curr = self.history[-1]
        return all(abs(prev[a] - curr[a]) < threshold for a in self.agents)

    def run_to_convergence(self, max_iter: int = 20) -> int:
        """Запустить до сходимости или max_iter."""
        for i in range(max_iter):
            self.iterate()
            if self.is_converged():
                return i + 1
        return max_iter


def demo_conflict_resolution():
    """Демонстрация: голосование, переговоры, алгоритмы консенсуса."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: РАЗРЕШЕНИЕ КОНФЛИКТОВ (Conflict Resolution)")
    print("=" * 70)

    # --- Пример 1: Голосование ---
    print("\n--- Пример 1: Взвешенное голосование ---")
    vs = VotingSystem()
    vs.register("Agent_A", weight=2.0)  # Старший агент
    vs.register("Agent_B", weight=1.0)
    vs.register("Agent_C", weight=1.5)
    vs.register("Agent_D", weight=1.0)

    # Голоса за варианты
    vs.cast_vote("Agent_A", " модель_1")
    vs.cast_vote("Agent_B", " модель_2")
    vs.cast_vote("Agent_C", " модель_1")
    vs.cast_vote("Agent_D", " модель_2")

    results = vs.count_votes()
    winner, pct = vs.majority_winner()
    print("  Голоса агентов:")
    for agent_name, option in vs.votes.items():
        agent_info = next(a for a in vs.agents if a["name"] == agent_name)
        print(f"    {agent_name} (вес={agent_info['weight']}): {option}")
    print(f"  Итого: {results}")
    print(f"  Победитель: {winner} ({pct:.1%} голосов)")

    # --- Пример 2: Переговоры ---
    print("\n--- Пример 2: Протокол переговоров ---")
    neg = NegotiationProtocol()
    # Раунд 1: агенты предлагают разные бюджеты
    r1 = neg.propose(1, {"Agent_A": 100, "Agent_B": 200, "Agent_C": 150})
    print(f"  Раунд 1: Agent_A=100, Agent_B=200, Agent_C=150")
    print(f"  Консенсус: среднее = {r1['consensus_value']:.1f}")

    # Модель торга Нэша
    print("\n  Модель торга Нэша:")
    print("  max ∏(u_i(x) - u_i(d)), где d — точка разногласий")
    nash = NegotiationProtocol()
    agent_values = [300, 200, 250]
    shares = nash.nash_bargaining(agent_values, disagreement_point=50)
    print(f"  Агенты: {[f'v={v}' for v in agent_values]}")
    print(f"  Точка разногласий: d=50")
    for i, share in enumerate(shares):
        print(f"    Agent_{i}: доля = {share:.1f}")

    # --- Пример 3: Алгоритм консенсуса ---
    print("\n--- Пример 3: Алгоритм консенсуса (среднее соседей) ---")
    agents = ["A", "B", "C", "D"]
    initial = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}
    consensus = ConsensusAlgorithm(agents, initial)

    print(f"  Начальные значения: {initial}")
    steps = consensus.run_to_convergence()
    print(f"  Сходится за {steps} итераций")
    print(f"  Финальные значения:")
    for agent, val in consensus.values.items():
        print(f"    {agent}: {val:.4f}")
    print(f"  Все значения сходятся к: {sum(initial.values()) / len(initial):.4f}")

    # --- Пример 4: Многораундовые переговоры ---
    print("\n--- Пример 4: Многораундовые переговоры ---")
    neg2 = NegotiationProtocol()
    agent_bids = {"A": 300, "B": 100, "C": 200}
    for r in range(1, 5):
        result = neg2.propose(r, agent_bids)
        avg = result["consensus_value"]
        # Каждый агент приближается к среднему на 30%
        for agent in agent_bids:
            agent_bids[agent] = agent_bids[agent] * 0.7 + avg * 0.3
        vals_str = ", ".join(f"{a}={v:.0f}" for a, v in agent_bids.items())
        print(f"  Раунд {r}: {vals_str} → среднее={avg:.1f}")

    print()


# ─────────────────────────────────────────────────────────────
# Запуск всех демонстраций
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  160 — Multi-Agent Systems: кооперация, коммуникация, координация")
    print("=" * 70 + "\n")

    demo_agent_communication()
    demo_cooperation_patterns()
    demo_task_allocation()
    demo_conflict_resolution()

    print("=" * 70)
    print("  Все 4 демонстрации завершены успешно!")
    print("=" * 70)
