"""199 — Scaling Multi-Agent Systems: иерархический контроль, абстракции, масштабирование

Темы:
  1. Hierarchical Control — мета-агенты, делегирование, уровни контроля
  2. Abstraction Layers — команды агентов, руководители команд, стратегический слой
  3. Scalability Patterns — партиционирование, кэширование, асинхронная коммуникация
  4. Performance Optimization — анализ узких мест, профилирование, оптимизация

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
# Демо 1: Hierarchical Control — иерархический контроль
# ============================================================

def demo_hierarchical_control():
    """Иерархическая архитектура управления: мета-агенты, делегирование, контроль."""
    print("=" * 70)
    print("ДЕМО 1: Hierarchical Control — иерархический контроль")
    print("=" * 70)

    # --- 1.1 Мета-агент ---
    print("\n--- 1.1 Мета-агент (Meta-Agent) ---")
    print("Высокоуровневый агент управляет группой подчинённых\n")

    class MetaAgent:
        """Мета-агент: управляет группой подчинённых агентов."""

        def __init__(self, name, skill_areas):
            self.name = name
            self.skill_areas = skill_areas
            self.subordinates = []
            self.task_log = []

        def add_subordinate(self, agent):
            """Добавить подчинённого агента."""
            self.subordinates.append(agent)

        def delegate(self, task):
            """Делегировать задачу наиболее подходящему подчинённому."""
            best_agent = None
            best_score = -1

            for agent in self.subordinates:
                # Оценка: совпадение навыков + текущая загрузка
                skill_match = len(set(task["required_skills"]) & set(agent["skills"]))
                load_factor = 1.0 / (1.0 + agent["current_load"])
                score = skill_match * load_factor
                if score > best_score:
                    best_score = score
                    best_agent = agent

            if best_agent:
                best_agent["current_load"] += task.get("complexity", 1)
                self.task_log.append({
                    "task": task["name"],
                    "assigned_to": best_agent["name"],
                    "score": round(best_score, 3),
                })
                return best_agent["name"]
            return None

        def report(self):
            """Отчёт о делегировании."""
            return {
                "meta_agent": self.name,
                "subordinates": len(self.subordinates),
                "tasks_delegated": len(self.task_log),
                "assignments": self.task_log,
            }

    # Создаём мета-агента
    meta = MetaAgent("Manager-0", ["planning", "coordination"])

    # Подчинённые агенты
    agents = [
        {"name": "Worker-A", "skills": ["coding", "testing"], "current_load": 0},
        {"name": "Worker-B", "skills": ["coding", "review"], "current_load": 0},
        {"name": "Worker-C", "skills": ["testing", "deployment"], "current_load": 0},
        {"name": "Worker-D", "skills": ["coding", "data"], "current_load": 0},
    ]

    for agent in agents:
        meta.add_subordinate(agent)

    # Делегируем задачи
    tasks = [
        {"name": "Фича-1", "required_skills": ["coding", "testing"], "complexity": 3},
        {"name": "Ревью-1", "required_skills": ["review", "coding"], "complexity": 2},
        {"name": "Деплой-1", "required_skills": ["deployment", "testing"], "complexity": 1},
        {"name": "Данные-1", "required_skills": ["data", "coding"], "complexity": 4},
        {"name": "Фича-2", "required_skills": ["coding"], "complexity": 2},
    ]

    print(f"Мета-агент: {meta.name}")
    print(f"Подчинённые: {[a['name'] for a in agents]}")
    print("\nДелегирование задач:")

    for task in tasks:
        assigned = meta.delegate(task)
        print(f"  {task['name']} -> {assigned} (навыки: {task['required_skills']})")

    report = meta.report()
    print(f"\nОтчёт: делегировано {report['tasks_delegated']} задач")

    # --- 1.2 Многоуровневая иерархия ---
    print("\n--- 1.2 Многоуровневая иерархия ---")
    print("CEO -> Managers -> Team Leads -> Workers\n")

    class HierarchicalAgent:
        """Агент в иерархической структуре."""

        def __init__(self, name, role, level):
            self.name = name
            self.role = role
            self.level = level
            self.children = []
            self.parent = None

        def add_child(self, child):
            """Добавить подчинённого."""
            child.parent = self
            self.children.append(child)

        def count_descendants(self):
            """Подсчитать всех потомков."""
            count = len(self.children)
            for child in self.children:
                count += child.count_descendants()
            return count

        def find(self, name):
            """Найти агента по имени в поддереве."""
            if self.name == name:
                return self
            for child in self.children:
                result = child.find(name)
                if result:
                    return result
            return None

        def tree(self, indent=0):
            """Вывести дерево иерархии."""
            prefix = "  " * indent + ("├── " if indent > 0 else "")
            lines = [f"{prefix}{self.name} [{self.role}]"]
            for child in self.children:
                lines.extend(child.tree(indent + 1))
            return lines

    # Строим иерархию
    ceo = HierarchicalAgent("CEO", "Стратегическое управление", 0)

    mgr_eng = HierarchicalAgent("Mgr-Engineering", "Руководитель инженерии", 1)
    mgr_data = HierarchicalAgent("Mgr-Data", "Руководитель данных", 1)
    mgr_ops = HierarchicalAgent("Mgr-Ops", "Руководитель операций", 1)

    ceo.add_child(mgr_eng)
    ceo.add_child(mgr_data)
    ceo.add_child(mgr_ops)

    # Подчинённые managers
    tl_backend = HierarchicalAgent("TL-Backend", "Тимлид бэкенда", 2)
    tl_frontend = HierarchicalAgent("TL-Frontend", "Тимлид фронтенда", 2)
    tl_ml = HierarchicalAgent("TL-ML", "Тимлид ML", 2)

    mgr_eng.add_child(tl_backend)
    mgr_eng.add_child(tl_frontend)
    mgr_data.add_child(tl_ml)

    # Рабочие
    for i in range(3):
        w = HierarchicalAgent(f"Dev-{i}", "Разработчик", 3)
        tl_backend.add_child(w)

    for i in range(2):
        w = HierarchicalAgent(f"ML-{i}", "ML-инженер", 3)
        tl_ml.add_child(w)

    print("Дерево иерархии:")
    for line in ceo.tree():
        print(f"  {line}")

    print(f"\nВсего агентов в дереве: {ceo.count_descendants() + 1}")

    # Поиск агента
    target = ceo.find("TL-ML")
    if target:
        print(f"Найден: {target.name} [{target.role}] (уровень {target.level})")
        # Подсчёт подчинённых
        print(f"  Прямых подчинённых: {len(target.children)}")

    # --- 1.3 Делегирование с эскалацией ---
    print("\n--- 1.3 Делегирование с эскалацией ---")
    print("Задача поднимается вверх, если не может быть решена\n")

    class EscalationSystem:
        """Система делегирования с эскалацией."""

        def __init__(self):
            self.abilities = {
                "worker": ["simple_task", "code_review"],
                "lead": ["simple_task", "code_review", "architecture", "bug_fix"],
                "manager": ["simple_task", "code_review", "architecture", "bug_fix", "planning", "hiring"],
                "director": ["simple_task", "code_review", "architecture", "bug_fix", "planning", "hiring", "strategy"],
            }
            self.log = []

        def solve(self, task, level="worker", max_escalations=3):
            """Попытаться решить задачу с эскалацией."""
            escalations = 0
            current_level = level
            levels = ["worker", "lead", "manager", "director"]

            while current_level in levels:
                can_solve = task in self.abilities.get(current_level, [])
                self.log.append({
                    "level": current_level,
                    "task": task,
                    "can_solve": can_solve,
                    "escalated": not can_solve,
                })

                if can_solve:
                    return current_level, escalations

                escalations += 1
                if escalations > max_escalations:
                    return None, escalations

                idx = levels.index(current_level)
                if idx + 1 < len(levels):
                    current_level = levels[idx + 1]
                else:
                    break

            return None, escalations

    escalation = EscalationSystem()

    tasks_to_solve = ["simple_task", "architecture", "strategy", "code_review", "hiring"]
    print("Обработка задач с эскалацией:")
    for task in tasks_to_solve:
        solved_by, num_esc = escalation.solve(task)
        status = f"решено на уровне '{solved_by}'" if solved_by else "НЕ решено"
        print(f"  Задача '{task}': {status} (эскалаций: {num_esc})")

    # --- 1.4 Tree of Agents ---
    print("\n--- 1.4 Tree of Agents — дерево агентов ---")
    print("Каждый агент — узел дерева, обрабатывает подзадачи\n")

    class AgentNode:
        """Узел дерева агентов."""

        def __init__(self, name, work_time):
            self.name = name
            self.work_time = work_time
            self.children = []
            self.result = None

        def add_child(self, child):
            self.children.append(child)

        def execute(self, task):
            """Выполнить задачу: рекурсивно разделить и выполнить."""
            if not self.children:
                # Лист: выполняем работу
                self.result = f"{self.name}: обработал '{task}' за {self.work_time}мс"
                return self.result

            # Внутренний узел: разделяем и делегируем
            chunk_size = max(1, len(task) // len(self.children))
            results = []
            for i, child in enumerate(self.children):
                sub_task = task[i * chunk_size:(i + 1) * chunk_size]
                if sub_task:
                    results.append(child.execute(sub_task))

            self.result = f"{self.name}: координировал {len(self.children)} дочерних"
            return self.result

    # Строим дерево
    root = AgentNode("Root-Coordinator", 0)

    branch1 = AgentNode("Branch-A", 5)
    branch2 = AgentNode("Branch-B", 3)

    root.add_child(branch1)
    root.add_child(branch2)

    for i in range(2):
        branch1.add_child(AgentNode(f"A-Leaf-{i}", 2))
    for i in range(3):
        branch2.add_child(AgentNode(f"B-Leaf-{i}", 2))

    task_data = "Hello World from hierarchical agents!"
    print(f"Задача: '{task_data}'")
    print("\nВыполнение по дереву:")
    result = root.execute(task_data)
    print(f"  {root.result}")
    for child in root.children:
        print(f"  {child.result}")
        for leaf in child.children:
            if leaf.result:
                print(f"    {leaf.result}")


# ============================================================
# Демо 2: Abstraction Layers — слои абстракции
# ============================================================

def demo_abstraction_layers():
    """Слои абстракции: команды агентов, руководители, стратегический слой."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: Abstraction Layers — слои абстракции")
    print("=" * 70)

    # --- 2.1 Команды агентов (Agent Teams) ---
    print("\n--- 2.1 Команды агентов (Agent Teams) ---")
    print("Группировка агентов по функциональным командам\n")

    class AgentTeam:
        """Команда агентов с общей целью."""

        def __init__(self, name, focus_area):
            self.name = name
            self.focus_area = focus_area
            self.members = []
            self.completed_tasks = []

        def add_member(self, agent_name, role):
            """Добавить участника команды."""
            self.members.append({"name": agent_name, "role": role, "tasks_done": 0})

        def assign_task(self, task):
            """Назначить задачу наименее загруженному участнику."""
            if not self.members:
                return None

            # Находим наименее загруженного
            best = min(self.members, key=lambda m: m["tasks_done"])
            best["tasks_done"] += 1
            self.completed_tasks.append({"task": task, "assigned_to": best["name"]})
            return best["name"]

        def productivity(self):
            """Продуктивность команды."""
            total = sum(m["tasks_done"] for m in self.members)
            return {
                "team": self.name,
                "members": len(self.members),
                "total_tasks": total,
                "per_member": round(total / len(self.members), 1) if self.members else 0,
            }

    # Создаём команды
    teams = {
        "backend": AgentTeam("Backend Team", "API и базы данных"),
        "frontend": AgentTeam("Frontend Team", "UI/UX"),
        "ml": AgentTeam("ML Team", "Модели и инференс"),
    }

    teams["backend"].add_member("Бэк-1", "senior")
    teams["backend"].add_member("Бэк-2", "middle")
    teams["backend"].add_member("Бэк-3", "junior")

    teams["frontend"].add_member("Фронт-1", "senior")
    teams["frontend"].add_member("Фронт-2", "middle")

    teams["ml"].add_member("ML-1", "researcher")
    teams["ml"].add_member("ML-2", "engineer")

    # Распределяем задачи
    all_tasks = {
        "backend": ["API /users", "API /tasks", "DB миграция", "Кэширование"],
        "frontend": ["Главная страница", "Дашборд", "Форма логина"],
        "ml": ["Обучение модели", "Оптимизация инференса", "Датасет"],
    }

    print("Распределение задач по командам:")
    for team_name, tasks in all_tasks.items():
        team = teams[team_name]
        print(f"\n  [{team.name}] ({team.focus_area}):")
        for task in tasks:
            assigned = team.assign_task(task)
            print(f"    '{task}' -> {assigned}")

    print("\nПродуктивность команд:")
    for team in teams.values():
        stats = team.productivity()
        print(f"  {stats['team']}: {stats['total_tasks']} задач, "
              f"{stats['per_member']} на человека")

    # --- 2.2 Team Leaders ---
    print("\n--- 2.2 Team Leaders — руководители команд ---")
    print("Каждый тимлид координирует свою команду\n")

    class TeamLeader:
        """Руководитель команды: координирует и отчитывается."""

        def __init__(self, name, team_size):
            self.name = name
            self.team_size = team_size
            self.assigned = 0
            self.completed = 0
            self.blocked = []

        def receive_task(self, task):
            """Получить задачу от вышестоящего."""
            self.assigned += 1
            # Имитация: выполняем с вероятностью 80%
            if random.random() < 0.8:
                self.completed += 1
                return f"Выполнено: {task}"
            else:
                self.blocked.append(task)
                return f"ЗАБЛОКИРОВАНО: {task}"

        def status(self):
            return {
                "leader": self.name,
                "assigned": self.assigned,
                "completed": self.completed,
                "blocked": len(self.blocked),
                "efficiency": f"{self.completed / max(1, self.assigned) * 100:.0f}%",
            }

    # Создаём тимлидов
    leaders = [
        TeamLeader("Alice", 5),
        TeamLeader("Bob", 4),
        TeamLeader("Carol", 6),
    ]

    project_tasks = [
        "Спроектировать схему БД", "Написать API", "Настроить CI/CD",
        "Обучить модель", "Написать тесты", "Провести код-ревью",
        "Задеплоить на staging", "Оптимизировать запросы",
    ]

    print("Распределение задач по тимлидам:")
    for i, task in enumerate(project_tasks):
        leader = leaders[i % len(leaders)]
        result = leader.receive_task(task)
        print(f"  {leader.name}: {result}")

    print("\nСтатус тимлидов:")
    for leader in leaders:
        stats = leader.status()
        print(f"  {stats['leader']}: назначено={stats['assigned']}, "
              f"выполнено={stats['completed']}, "
              f"заблокировано={stats['blocked']}, "
              f"эффективность={stats['efficiency']}")

    # --- 2.3 Стратегический слой ---
    print("\n--- 2.3 Стратегический слой ---")
    print("Высший уровень: определяет приоритеты и стратегию\n")

    class StrategicLayer:
        """Стратегический слой: определяет приоритеты проектов."""

        def __init__(self):
            self.projects = {}
            self.budget = 1000

        def add_project(self, name, priority, estimated_cost, team_needed):
            """Добавить проект в портфель."""
            self.projects[name] = {
                "priority": priority,
                "cost": estimated_cost,
                "team": team_needed,
                "status": "planned",
            }

        def prioritize(self):
            """Отсортировать проекты по приоритету и бюджету."""
            sorted_projects = sorted(
                self.projects.items(),
                key=lambda x: (x[1]["priority"], -x[1]["cost"]),
                reverse=True,
            )
            return sorted_projects

        def allocate_budget(self):
            """Распределить бюджет по проектам (greedy)."""
            prioritized = self.prioritize()
            allocations = []
            remaining = self.budget

            for name, info in prioritized:
                if info["cost"] <= remaining:
                    allocations.append((name, info["cost"]))
                    remaining -= info["cost"]
                    self.projects[name]["status"] = "approved"
                else:
                    self.projects[name]["status"] = "deferred"

            return allocations, remaining

        def report(self):
            """Отчёт по портфелю проектов."""
            approved = sum(1 for p in self.projects.values() if p["status"] == "approved")
            deferred = sum(1 for p in self.projects.values() if p["status"] == "deferred")
            return {
                "total_projects": len(self.projects),
                "approved": approved,
                "deferred": deferred,
                "budget_used": self.budget - self.allocate_budget()[1] if self.projects else 0,
            }

    strategic = StrategicLayer()

    # Добавляем проекты
    projects = [
        ("Рекомендательная система", 9, 300, "ml"),
        ("Новый API", 7, 200, "backend"),
        ("Мобильное приложение", 8, 400, "frontend"),
        ("Система мониторинга", 6, 150, "ops"),
        ("ML-пайплайн", 10, 250, "ml"),
    ]

    print("Портфель проектов:")
    for name, priority, cost, team in projects:
        strategic.add_project(name, priority, cost, team)
        print(f"  '{name}': приоритет={priority}, стоимость={cost}, команда={team}")

    print("\nПриоритезация:")
    for name, info in strategic.prioritize():
        print(f"  {name}: приоритет={info['priority']}, стоимость={info['cost']}")

    allocations, remaining = strategic.allocate_budget()
    print(f"\nБюджет: {strategic.budget}")
    print("Выделения:")
    for name, cost in allocations:
        print(f"  {name}: {cost}")
    print(f"  Остаток: {remaining}")

    print("\nСтатус проектов:")
    for name, info in strategic.projects.items():
        print(f"  {name}: {info['status']}")

    # --- 2.4 Многослойная архитектура ---
    print("\n--- 2.4 Многослойная архитектура ---")
    print("Стратегический -> Тактический -> Операционный -> Исполнительный\n")

    class MultiLayerArchitecture:
        """Многослойная архитектура с разными уровнями абстракции."""

        def __init__(self):
            self.layers = {
                "strategic": {"tasks": [], "processed": []},
                "tactical": {"tasks": [], "processed": []},
                "operational": {"tasks": [], "processed": []},
                "execution": {"tasks": [], "processed": []},
            }
            self.flow_log = []

        def submit_task(self, task, complexity):
            """Задача поступает на нижний уровень и поднимается при необходимости."""
            current_layer = "execution"
            self.layers[current_layer]["tasks"].append(task)
            self.flow_log.append(f"'{task}' поступила в {current_layer}")

            # Простые задачи обрабатываются сразу
            if complexity <= 2:
                self.layers[current_layer]["processed"].append(task)
                self.flow_log.append(f"'{task}' обработана в {current_layer}")
                return current_layer

            # Сложные задачи поднимаются вверх
            layer_order = ["execution", "operational", "tactical", "strategic"]
            for layer in layer_order:
                if layer_order.index(layer) >= layer_order.index(current_layer):
                    self.layers[layer]["tasks"].append(task)
                    self.flow_log.append(f"'{task}' поднята в {layer}")
                    if complexity <= (layer_order.index(layer) + 1) * 3:
                        self.layers[layer]["processed"].append(task)
                        self.flow_log.append(f"'{task}' обработана в {layer}")
                        return layer

            # Самые сложные — стратегический уровень
            self.layers["strategic"]["processed"].append(task)
            self.flow_log.append(f"'{task}' обработана в strategic")
            return "strategic"

    arch = MultiLayerArchitecture()

    test_tasks = [
        ("Фикс бага", 1),
        ("Написать тесты", 2),
        ("Рефакторинг модуля", 4),
        ("Новая архитектура API", 7),
        ("Проектирование системы", 10),
    ]

    print("Обработка задач в многослойной архитектуре:")
    for task, complexity in test_tasks:
        result_layer = arch.submit_task(task, complexity)
        print(f"  '{task}' (сложность={complexity}) -> обработана в '{result_layer}'")

    print("\nЖурнал потока задач:")
    for entry in arch.flow_log:
        print(f"  {entry}")


# ============================================================
# Демо 3: Scalability Patterns — паттерны масштабирования
# ============================================================

def demo_scalability_patterns():
    """Паттерны масштабирования: партиционирование, кэширование, async."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: Scalability Patterns — паттерны масштабирования")
    print("=" * 70)

    # --- 3.1 Партиционирование ---
    print("\n--- 3.1 Партиционирование (Sharding) ---")
    print("Данные разбиваются по партициям, каждая обрабатывается своим агентом\n")

    class ShardRouter:
        """Маршрутизатор: направляет данные в нужную партицию."""

        def __init__(self, n_shards):
            self.shards = {i: [] for i in range(n_shards)}
            self.n_shards = n_shards

        def route(self, key):
            """Определить шард по ключу (хэш-модуло)."""
            h = int(hashlib.md5(key.encode()).hexdigest(), 16)
            return h % self.n_shards

        def insert(self, key, value):
            """Вставить данные в нужную партицию."""
            shard_id = self.route(key)
            self.shards[shard_id].append({"key": key, "value": value})
            return shard_id

        def get(self, key):
            """Получить данные из партиции."""
            shard_id = self.route(key)
            for item in self.shards[shard_id]:
                if item["key"] == key:
                    return item["value"]
            return None

        def stats(self):
            """Статистика по партициям."""
            return {f"Shard-{i}": len(data) for i, data in self.shards.items()}

    router = ShardRouter(4)

    # Распределяем данные
    users = [
        ("alice", "инженер"),
        ("bob", "дизайнер"),
        ("carol", "менеджер"),
        ("dave", "аналитик"),
        ("eve", "разработчик"),
        ("frank", "тестировщик"),
        ("grace", "архитектор"),
        ("hank", "DevOps"),
    ]

    print("Распределение пользователей по шардам:")
    for user_id, role in users:
        shard = router.insert(user_id, role)
        print(f"  {user_id} -> Shard-{shard} ({role})")

    print("\nЗагрузка шардов:")
    for shard, count in router.stats().items():
        bar = "█" * count
        print(f"  {shard}: {count} записей {bar}")

    # Поиск
    print("\nПоиск:")
    for query in ["alice", "eve", "bob"]:
        result = router.get(query)
        shard = router.route(query)
        print(f"  {query}: Shard-{shard} -> {result}")

    # --- 3.2 Кэширование ---
    print("\n--- 3.2 Кэширование агентов ---")
    print("Часто запрашиваемые результаты кэшируются\n")

    class AgentCache:
        """Кэш результатов агентов с LRU-эвикцией."""

        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = {}
            self.access_order = []
            self.hits = 0
            self.misses = 0

        def get(self, key):
            """Получить из кэша."""
            if key in self.cache:
                self.hits += 1
                # Обновляем порядок доступа
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]

            self.misses += 1
            return None

        def put(self, key, value):
            """Записать в кэш."""
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.capacity:
                # LRU: удаляем самый давно использованный
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

            self.cache[key] = value
            self.access_order.append(key)

        def hit_rate(self):
            """Коэффициент попаданий."""
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0

    cache = AgentCache(capacity=3)

    # Имитация запросов с паттерном (Zipf-like)
    queries = ["model_A", "model_B", "model_C", "model_A", "model_D",
               "model_A", "model_B", "model_E", "model_A", "model_C",
               "model_A", "model_B", "model_A", "model_F", "model_A"]

    print(f"Ёмкость кэша: {cache.capacity}")
    print("\nЗапросы и попадания:")
    for q in queries:
        result = cache.get(q)
        if result is None:
            # Имитация вычисления
            result = f"результат({q})"
            cache.put(q, result)
            status = "MISS (вычислено)"
        else:
            status = f"HIT (из кэша)"

        print(f"  Запрос '{q}': {status}")

    print(f"\nИтого: hits={cache.hits}, misses={cache.misses}, "
          f"hit rate={cache.hit_rate():.1%}")
    print(f"Кэш: {list(cache.cache.keys())}")

    # --- 3.3 Асинхронная коммуникация ---
    print("\n--- 3.3 Асинхронная коммуникация ---")
    print("Сообщения отправляются через очередь, а не напрямую\n")

    class MessageQueue:
        """Простая асинхронная очередь сообщений."""

        def __init__(self):
            self.queues = collections.defaultdict(list)
            self.dead_letter = []
            self.processed = []

        def publish(self, topic, message):
            """Опубликовать сообщение в топик."""
            self.queues[topic].append(message)

        def subscribe(self, topic, handler):
            """Обработать все сообщения в топике."""
            results = []
            while self.queues[topic]:
                msg = self.queues[topic].pop(0)
                try:
                    result = handler(msg)
                    results.append(result)
                    self.processed.append({"topic": topic, "message": msg, "result": result})
                except Exception as e:
                    self.dead_letter.append({"topic": topic, "message": msg, "error": str(e)})
            return results

        def stats(self):
            return {
                "processed": len(self.processed),
                "dead_letter": len(self.dead_letter),
                "pending": {t: len(q) for t, q in self.queues.items()},
            }

    mq = MessageQueue()

    # Публикуем сообщения
    events = [
        ("tasks", {"type": "train", "model": "GPT"}),
        ("tasks", {"type": "evaluate", "model": "GPT"}),
        ("tasks", {"type": "deploy", "model": "GPT"}),
        ("logs", {"level": "info", "message": "Старт"}),
        ("logs", {"level": "error", "message": "Ошибка"}),
        ("metrics", {"cpu": 85, "memory": 70}),
        ("metrics", {"cpu": 92, "memory": 88}),
    ]

    print("Публикация сообщений:")
    for topic, msg in events:
        mq.publish(topic, msg)
        print(f"  -> {topic}: {msg}")

    # Подписываемся и обрабатываем
    print("\nОбработка топика 'tasks':")
    task_results = mq.subscribe("tasks", lambda msg: f"Обработано: {msg['type']}")
    for r in task_results:
        print(f"  {r}")

    print("\nОбработка топика 'logs':")
    log_results = mq.subscribe("logs", lambda msg: f"[{msg['level']}] {msg['message']}")
    for r in log_results:
        print(f"  {r}")

    print("\nОбработка топика 'metrics':")
    metric_results = mq.subscribe("metrics", lambda msg: f"CPU={msg['cpu']}%, MEM={msg['memory']}%")
    for r in metric_results:
        print(f"  {r}")

    stats = mq.stats()
    print(f"\nСтатистика: обработано={stats['processed']}, "
          f"dead_letter={stats['dead_letter']}, "
          f"ожидают={stats['pending']}")

    # --- 3.4 Горизонтальное масштабирование ---
    print("\n--- 3.4 Горизонтальное масштабирование ---")
    print("Добавление агентов при росте нагрузки\n")

    class HorizontalScaler:
        """Система горизонтального масштабирования."""

        def __init__(self, min_agents, max_agents, scale_up_at=80, scale_down_at=30):
            self.min_agents = min_agents
            self.max_agents = max_agents
            self.scale_up_at = scale_up_at
            self.scale_down_at = scale_down_at
            self.agents = [f"Agent-{i}" for i in range(min_agents)]
            self.load_history = []

        def current_load(self):
            """Средняя нагрузка."""
            if not self.agents:
                return 0
            return sum(self.load_history[-len(self.agents):]) / len(self.agents) if self.load_history else 0

        def update_load(self, load):
            """Обновить нагрузку и при необходимости масштабировать."""
            self.load_history.append(load)
            avg_load = self.current_load()
            action = "без изменений"

            if avg_load > self.scale_up_at and len(self.agents) < self.max_agents:
                new_agent = f"Agent-{len(self.agents)}"
                self.agents.append(new_agent)
                action = f"масштабирование ВВЕРХ: +{new_agent}"
            elif avg_load < self.scale_down_at and len(self.agents) > self.min_agents:
                removed = self.agents.pop()
                action = f"масштабирование ВНИЗ: -{removed}"

            return action

    scaler = HorizontalScaler(min_agents=2, max_agents=6, scale_up_at=70, scale_down_at=20)

    print(f"Начало: {len(scaler.agents)} агентов")

    # Симуляция нагрузки
    loads = [50, 60, 75, 85, 90, 80, 60, 40, 25, 15, 20, 30]
    print("\nДинамика масштабирования:")
    for i, load in enumerate(loads):
        action = scaler.update_load(load)
        print(f"  t={i}: нагрузка={load}%, агентов={len(scaler.agents)}, {action}")

    print(f"\nФинал: {len(scaler.agents)} агентов — {scaler.agents}")


# ============================================================
# Демо 4: Performance Optimization — оптимизация производительности
# ============================================================

def demo_performance_optimization():
    """Анализ узких мест, профилирование и оптимизация мультиагентных систем."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: Performance Optimization — оптимизация")
    print("=" * 70)

    # --- 4.1 Анализ узких мест ---
    print("\n--- 4.1 Анализ узких мест (Bottleneck Analysis) ---")
    print("Поиск компонентов, ограничивающих пропускную способность\n")

    class Pipeline:
        """Конвейер обработки с этапами разной скорости."""

        def __init__(self, stages):
            # stages: [(name, capacity_per_sec), ...]
            self.stages = stages

        def throughput(self):
            """Пропускная способность = минимум по всем этапам."""
            return min(cap for _, cap in self.stages)

        def bottleneck(self):
            """Найти узкое место."""
            min_cap = min(cap for _, cap in self.stages)
            for name, cap in self.stages:
                if cap == min_cap:
                    return name, cap

        def analyze(self):
            """Полный анализ пропускной способности."""
            bp_name, bp_cap = self.bottleneck()
            total_time = sum(1 / cap for _, cap in self.stages)
            efficiency = bp_cap / max(cap for _, cap in self.stages)

            return {
                "bottleneck": bp_name,
                "bottleneck_capacity": bp_cap,
                "pipeline_throughput": self.throughput(),
                "total_latency": round(total_time, 4),
                "efficiency": f"{efficiency:.1%}",
            }

    # Конвейер обработки изображений
    pipeline = Pipeline([
        ("Загрузка изображения", 100),      # 100 изображений/сек
        ("Предобработка", 80),               # 80/сек — узкое место!
        ("Инференс модели", 20),             # 20/сек — самое медленное!
        ("Постобработка", 150),
        ("Сохранение", 200),
    ])

    print("Этапы конвейера:")
    for name, cap in pipeline.stages:
        marker = " <-- УЗКОЕ МЕСТО" if cap == pipeline.bottleneck()[1] else ""
        print(f"  {name}: {cap} обработок/сек{marker}")

    analysis = pipeline.analyze()
    print(f"\nАнализ:")
    print(f"  Узкое место: {analysis['bottleneck']}")
    print(f"  Пропускная способность: {analysis['pipeline_throughput']} обработок/сек")
    print(f"  Задержка: {analysis['total_latency']} сек")
    print(f"  Эффективность: {analysis['efficiency']}")

    # --- 4.2 Профилирование агентов ---
    print("\n--- 4.2 Профилирование агентов ---")
    print("Замер времени выполнения операций каждым агентом\n")

    class AgentProfiler:
        """Профилировщик: замеряет время выполнения операций."""

        def __init__(self):
            self.profiles = collections.defaultdict(list)

        def record(self, agent_id, operation, duration_ms):
            """Записать время выполнения операции."""
            self.profiles[agent_id].append({
                "operation": operation,
                "duration_ms": duration_ms,
            })

        def summary(self):
            """Сводка по каждому агенту."""
            summaries = {}
            for agent, records in self.profiles.items():
                durations = [r["duration_ms"] for r in records]
                op_counts = collections.Counter(r["operation"] for r in records)

                # Находим самую медленную операцию
                op_avg = {}
                for op in op_counts:
                    op_durations = [r["duration_ms"] for r in records if r["operation"] == op]
                    op_avg[op] = sum(op_durations) / len(op_durations)

                summaries[agent] = {
                    "total_calls": len(records),
                    "total_ms": sum(durations),
                    "avg_ms": round(sum(durations) / len(durations), 2),
                    "p95_ms": round(sorted(durations)[int(len(durations) * 0.95)], 2) if durations else 0,
                    "slowest_op": max(op_avg, key=op_avg.get) if op_avg else None,
                }
            return summaries

    profiler = AgentProfiler()

    # Имитация работы агентов
    random.seed(42)
    for agent_id in ["Agent-A", "Agent-B", "Agent-C"]:
        operations = ["inference", "preprocess", "postprocess"]
        for _ in range(20):
            op = random.choice(operations)
            # Разное время для разных агентов и операций
            base_times = {"inference": 50, "preprocess": 10, "postprocess": 5}
            agent_factor = {"Agent-A": 1.0, "Agent-B": 1.3, "Agent-C": 0.8}
            duration = base_times[op] * agent_factor[agent_id] * random.uniform(0.8, 1.2)
            profiler.record(agent_id, op, round(duration, 2))

    summary = profiler.summary()
    print("Профиль агентов:")
    for agent, stats in summary.items():
        print(f"\n  {agent}:")
        print(f"    Вызовов: {stats['total_calls']}")
        print(f"    Общее время: {stats['total_ms']:.1f} мс")
        print(f"    Среднее: {stats['avg_ms']} мс")
        print(f"    P95: {stats['p95_ms']} мс")
        print(f"    Самая медленная операция: {stats['slowest_op']}")

    # --- 4.3 Оптимизация памяти ---
    print("\n--- 4.3 Оптимизация памяти агентов ---")
    print("Сжатие состояний агентов, объектные пулы\n")

    class MemoryOptimizedAgent:
        """Агент с оптимизацией памяти: __slots__ и object pool."""

        __slots__ = ["agent_id", "state", "memory_usage"]

        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.state = "idle"
            self.memory_usage = 0

    class ObjectPool:
        """Пул объектов для переиспользования."""

        def __init__(self, factory, initial_size=10):
            self.factory = factory
            self.available = [factory() for _ in range(initial_size)]
            self.in_use = 0
            self.created = initial_size

        def acquire(self):
            """Получить объект из пула."""
            if self.available:
                obj = self.available.pop()
            else:
                obj = self.factory()
                self.created += 1
            self.in_use += 1
            return obj

        def release(self, obj):
            """Вернуть объект в пул."""
            self.in_use -= 1
            self.available.append(obj)

        def stats(self):
            return {
                "available": len(self.available),
                "in_use": self.in_use,
                "total_created": self.created,
                "reuse_rate": f"{(self.created - len(self.available) - self.in_use) / max(1, self.created) * 100:.0f}%",
            }

    # Демонстрация пула
    pool = ObjectPool(lambda: MemoryOptimizedAgent(f"Agent-{random.randint(1000, 9999)}"), initial_size=5)

    print(f"Пул создан: {pool.stats()}")

    # Используем объекты
    acquired_agents = []
    for i in range(8):
        agent = pool.acquire()
        agent.state = "working"
        acquired_agents.append(agent)
        print(f"  Acquired: {agent.agent_id} (state={agent.state})")

    print(f"\nПосле acquire×8: {pool.stats()}")

    # Возвращаем объекты
    for agent in acquired_agents[:5]:
        agent.state = "idle"
        pool.release(agent)

    print(f"После release×5: {pool.stats()}")

    # --- 4.4 Метрики производительности ---
    print("\n--- 4.4 Метрики производительности ---")
    print("Сбор и анализ ключевых метрик системы\n")

    class PerformanceMetrics:
        """Сбор метрик производительности мультиагентной системы."""

        def __init__(self):
            self.metrics = {
                "throughput": [],       # задач/сек
                "latency": [],          # мс на задачу
                "agent_utilization": [], # % использования агентов
                "queue_depth": [],      # глубина очереди
                "error_rate": [],       # % ошибок
            }
            self.alerts = []

        def record(self, **kwargs):
            """Записать значения метрик."""
            for key, value in kwargs.items():
                if key in self.metrics:
                    self.metrics[key].append(value)

            # Проверка порогов
            if kwargs.get("latency", 0) > 100:
                self.alerts.append(f"Высокая задержка: {kwargs['latency']} мс")
            if kwargs.get("error_rate", 0) > 5:
                self.alerts.append(f"Высокий уровень ошибок: {kwargs['error_rate']}%")
            if kwargs.get("agent_utilization", 0) > 90:
                self.alerts.append(f"Высокая загрузка агентов: {kwargs['agent_utilization']}%")

        def summary(self):
            """Сводка по метрикам."""
            result = {}
            for key, values in self.metrics.items():
                if values:
                    result[key] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": round(sum(values) / len(values), 2),
                        "last": values[-1],
                    }
            return result

    metrics = PerformanceMetrics()

    # Имитация сбора метрик
    print("Сбор метрик (10 интервалов):")
    for i in range(10):
        metrics.record(
            throughput=random.uniform(50, 120),
            latency=random.uniform(20, 150),
            agent_utilization=random.uniform(30, 95),
            queue_depth=random.randint(0, 20),
            error_rate=random.uniform(0, 8),
        )
        print(f"  Интервал {i+1}: throughput={metrics.metrics['throughput'][-1]:.1f}, "
              f"latency={metrics.metrics['latency'][-1]:.1f}мс")

    summary = metrics.summary()
    print("\nСводка метрик:")
    for key, stats in summary.items():
        print(f"  {key}: min={stats['min']:.1f}, max={stats['max']:.1f}, "
              f"avg={stats['avg']:.1f}, last={stats['last']:.1f}")

    print(f"\nАлерты ({len(metrics.alerts)}):")
    for alert in metrics.alerts[:5]:
        print(f"  ⚠ {alert}")
    if len(metrics.alerts) > 5:
        print(f"  ... и ещё {len(metrics.alerts) - 5}")


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    demo_hierarchical_control()
    demo_abstraction_layers()
    demo_scalability_patterns()
    demo_performance_optimization()
