"""198 — Task Distribution: аллокация задач, балансировка нагрузки, распределение ресурсов

Темы:
  1. Task Allocation — венгерский алгоритм, аукционные методы, жадное распределение
  2. Load Balancing — round-robin, наименее загруженный, взвешенный, адаптивный
  3. Resource Management — пулы ресурсов, резервирование, перехват
  4. Work Stealing — прокачка задач от свободных агентов к занятым, диффузия нагрузки

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
# Демо 1: Task Allocation — аллокация задач агентам
# ============================================================

def demo_task_allocation():
    """Распределение задач среди агентов: венгерский, аукцион, жадный подход."""
    print("=" * 70)
    print("ДЕМО 1: Task Allocation — аллокация задач")
    print("=" * 70)

    # --- 1.1 Жадное распределение (greedy allocation) ---
    print("\n--- 1.1 Жадное (greedy) распределение задач ---")
    print("Принцип: каждую задачу назначаем агенту с минимальной стоимостью\n")

    # Матрица стоимости: cost[i][j] = стоимость назначения задачи j агенту i
    cost_matrix = [
        [4, 2, 7, 3],  # агент 0
        [2, 5, 1, 6],  # агент 1
        [6, 3, 4, 2],  # агент 2
        [3, 7, 5, 4],  # агент 3
    ]
    agents = ["A0", "A1", "A2", "A3"]
    tasks = ["T0", "T1", "T2", "T3"]

    # Жадный алгоритм: сортируем пары (стоимость, агент, задача) по возрастанию
    assignments = {}
    used_agents = set()
    used_tasks = set()

    # Создаём список всех пар (стоимость, агент, задача)
    all_pairs = []
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix[i])):
            all_pairs.append((cost_matrix[i][j], i, j))

    # Сортируем по стоимости — от дешёвых к дорогим
    all_pairs.sort()

    # Жадно назначаем: берём самую дешёвую пару, если оба свободны
    for cost, agent_idx, task_idx in all_pairs:
        if agent_idx not in used_agents and task_idx not in used_tasks:
            assignments[task_idx] = (agent_idx, cost)
            used_agents.add(agent_idx)
            used_tasks.add(task_idx)

    total_cost = sum(c for _, c in assignments.values())
    print("Матрица стоимости (агент × задача):")
    print(f"  {'':8s}", end="")
    for t in tasks:
        print(f"{t:>6s}", end="")
    print()
    for i, row in enumerate(cost_matrix):
        print(f"  {agents[i]:6s}", end="")
        for v in row:
            print(f"{v:>6d}", end="")
        print()

    print(f"\nЖадное назначение (стоимость: {total_cost}):")
    for t_idx in sorted(assignments):
        a_idx, c = assignments[t_idx]
        print(f"  {tasks[t_idx]} -> {agents[a_idx]} (стоимость={c})")

    # --- 1.2 Аукционный метод ---
    print("\n--- 1.2 Аукционный метод распределения ---")
    print("Агенты делают ставки; задача достаётся с наибольшей ставкой\n")

    # Имитация аукциона: агенты竞争力 (competitiveness) определяет ставку
    task_values = {"T0": 10, "T1": 8, "T2": 12, "T3": 7}
    agent_profiles = {
        "A0": {"skill": 0.9, "budget": 20},  # высокий скилл, большой бюджет
        "A1": {"skill": 0.7, "budget": 15},
        "A2": {"skill": 0.8, "budget": 25},  # средний скилл, большой бюджет
        "A3": {"skill": 0.5, "budget": 10},
    }

    auction_results = {}
    for task_name, base_value in task_values.items():
        bids = {}
        for agent_name, profile in agent_profiles.items():
            # Ставка = базовая ценность × скилл × случайный коэффициент конкуренции
            bid = base_value * profile["skill"] * random.uniform(0.8, 1.2)
            bid = min(bid, profile["budget"])  # не больше бюджета
            bids[agent_name] = round(bid, 2)

        # Победитель — максимум ставки
        winner = max(bids, key=bids.get)
        auction_results[task_name] = (winner, bids[winner])
        print(f"  Задача {task_name} (базовая ценность={base_value}):")
        for name, bid in sorted(bids.items()):
            marker = " <-- ПОБЕДИТЕЛЬ" if name == winner else ""
            print(f"    {name}: ставка={bid:.2f}{marker}")

    print("\nИтог аукциона:")
    for t, (a, b) in auction_results.items():
        print(f"  {t} -> {a} (ставка={b})")

    # --- 1.3 Венгерский алгоритм (концепция) ---
    print("\n--- 1.3 Венгерский алгоритм (оптимальное назначение) ---")
    print("Минимум суммарной стоимости при однозначном назначении\n")

    def hungarian_simple(costs):
        """Упрощённый венгерский алгоритм для квадратных матриц.
        Использует перебор для малых размеров (N<=8)."""
        n = len(costs)
        if n == 1:
            return [0], costs[0][0]

        best_assignment = None
        best_cost = float("inf")

        def backtrack(row, used_cols, current_assignment, current_cost):
            """Рекурсивный перебор всех перестановок столбцов."""
            nonlocal best_assignment, best_cost
            if row == n:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment[:]
                return

            for col in range(n):
                if col not in used_cols:
                    current_assignment[row] = col
                    used_cols.add(col)
                    backtrack(row + 1, used_cols, current_assignment, current_cost + costs[row][col])
                    used_cols.remove(col)

        backtrack(0, set(), [0] * n, 0)
        return best_assignment, best_cost

    assignment, min_cost = hungarian_simple(cost_matrix)

    print(f"Венгерский алгоритм (оптимум: {min_cost}):")
    for i, j in enumerate(assignment):
        print(f"  {agents[i]} -> {tasks[j]} (стоимость={cost_matrix[i][j]})")

    print(f"\nСравнение: жадный={total_cost} vs венгерский={min_cost}")
    if total_cost == min_cost:
        print("  Жадный алгоритм дал оптимальное решение для этого случая!")
    else:
        print(f"  Жадный алгоритм переплатил на {total_cost - min_cost}")

    # --- 1.4 Стохастическое назначение ---
    print("\n--- 1.4 Стохастическое назначение с вероятностями ---")
    print("Задачи распределяются пропорционально способностям агентов\n")

    # Каждый агент имеет «усердие» — вероятность того, что он справится лучше
    agent_fitness = {"A0": 0.95, "A1": 0.7, "A2": 0.85, "A3": 0.6}

    task_difficulty = [0.8, 0.5, 0.9, 0.3]
    stochastic_results = collections.Counter()

    n_trials = 1000
    for _ in range(n_trials):
        for t_idx, diff in enumerate(task_difficulty):
            # Вероятность назначения = fitness / сумма всех fitness
            total_fitness = sum(agent_fitness.values())
            probs = {a: f / total_fitness for a, f in agent_fitness.items()}
            # Выбор агента пропорционально вероятности
            r = random.random()
            cumulative = 0.0
            for agent_name, prob in probs.items():
                cumulative += prob
                if r <= cumulative:
                    stochastic_results[(agent_name, tasks[t_idx])] += 1
                    break

    print(f"Результат {n_trials} итераций (топ назначений):")
    sorted_results = sorted(stochastic_results.items(), key=lambda x: x[1], reverse=True)
    for (agent, task), count in sorted_results[:8]:
        pct = count / n_trials * 100
        bar = "#" * int(pct / 2)
        print(f"  {agent} -> {task}: {count:>4d} раз ({pct:5.1f}%) {bar}")

    print("\nИтого: задачи распределяются пропорционально способностям агентов")


# ============================================================
# Демо 2: Load Balancing — балансировка нагрузки
# ============================================================

def demo_load_balancing():
    """Методы балансировки нагрузки между агентами."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: Load Balancing — балансировка нагрузки")
    print("=" * 70)

    # --- 2.1 Round-Robin ---
    print("\n--- 2.1 Round-Robin — циклическое распределение ---")
    print("Каждая следующая задача идёт следующему агенту по кругу\n")

    agents_rr = ["A0", "A1", "A2"]
    tasks_rr = [f"T{i}" for i in range(10)]

    # Round-robin: индекс агента = (номер задачи) % (число агентов)
    rr_assignments = {}
    for i, task in enumerate(tasks_rr):
        agent = agents_rr[i % len(agents_rr)]
        rr_assignments[task] = agent

    # Подсчёт нагрузки
    rr_loads = collections.Counter(rr_assignments.values())

    print("Распределение:")
    for task in tasks_rr:
        print(f"  {task} -> {rr_assignments[task]}")

    print(f"\nНагрузка: {dict(rr_loads)}")
    print(f"Макс. отклонение от среднего: {max(rr_loads.values()) - min(rr_loads.values())}")

    # --- 2.2 Least-Loaded ---
    print("\n--- 2.2 Least-Loaded — наименее загруженный агент ---")
    print("Новая задача идёт агенту с минимальной текущей нагрузкой\n")

    agent_loads = {"A0": 0, "A1": 0, "A2": 0}
    agent_costs = {"A0": 1.0, "A1": 1.5, "A2": 0.8}  # разная стоимость обработки

    tasks_ll = [("T0", 3), ("T1", 5), ("T2", 2), ("T3", 7), ("T4", 4), ("T5", 6)]

    print("Задачи (имя, требуемое время):")
    ll_assignments = {}
    for task_name, cost in tasks_ll:
        # Выбираем агента с минимальной текущей загрузкой × стоимость
        best_agent = min(agent_loads, key=lambda a: agent_loads[a] * agent_costs[a])
        agent_loads[best_agent] += cost * agent_costs[best_agent]
        ll_assignments[task_name] = (best_agent, cost)
        print(f"  {task_name} (время={cost}) -> {best_agent} (нагрузка={agent_loads[best_agent]:.1f})")

    print(f"\nИтоговая нагрузка: {dict(agent_loads)}")

    # --- 2.3 Взвешенный round-robin ---
    print("\n--- 2.3 Weighted Round-Robin ---")
    print("Агенты с бо́льшим весом получают больше задач\n")

    weighted_agents = {
        "A0": {"weight": 3, "tasks_assigned": []},  # мощный сервер
        "A1": {"weight": 2, "tasks_assigned": []},  # средний сервер
        "A2": {"weight": 1, "tasks_assigned": []},  # слабый сервер
    }

    # Создаём «взвешенную очередь»: повторяем агента weight раз
    weighted_queue = []
    for name, info in weighted_agents.items():
        weighted_queue.extend([name] * info["weight"])

    # Распределяем 12 задач по взвешенной очереди
    tasks_wrr = [f"T{i}" for i in range(12)]
    for i, task in enumerate(tasks_wrr):
        agent = weighted_queue[i % len(weighted_queue)]
        weighted_agents[agent]["tasks_assigned"].append(task)

    print(f"Взвешенная очередь: {weighted_queue}")
    print("\nРаспределение:")
    for name, info in weighted_agents.items():
        count = len(info["tasks_assigned"])
        print(f"  {name} (вес={info['weight']}): {count} задач — {info['tasks_assigned']}")

    # --- 2.4 Адаптивный балансировщик ---
    print("\n--- 2.4 Адаптивный балансировщик ---")
    print("Динамически перераспределяет задачи при изменении нагрузки\n")

    class AdaptiveBalancer:
        """Балансировщик, который перераспределяет задачи при дисбалансе."""

        def __init__(self, n_agents):
            self.queues = [[] for _ in range(n_agents)]
            self.capacities = [10, 8, 12, 6]  # вместимость каждого агента

        def add_task(self, task_id):
            """Добавить задачу на наименее загруженный агент."""
            loads = [len(q) for q in self.queues]
            best = min(range(len(self.queues)), key=lambda i: loads[i] / self.capacities[i])
            self.queues[best].append(task_id)

        def rebalance(self):
            """Перераспределить задачи, если есть дисбаланс."""
            loads = [len(q) / c for q, c in zip(self.queues, self.capacities)]
            max_load = max(loads)
            min_load = min(loads)

            if max_load - min_load < 0.3:
                return 0  # баланс достаточный

            moved = 0
            for i in range(len(self.queues)):
                if loads[i] == max_load and len(self.queues[i]) > 1:
                    # Перекинуть последнюю задачу на наименее загруженного
                    target = min(range(len(self.queues)), key=lambda j: loads[j])
                    if target != i:
                        task = self.queues[i].pop()
                        self.queues[target].append(task)
                        loads[i] = len(self.queues[i]) / self.capacities[i]
                        loads[target] = len(self.queues[target]) / self.capacities[target]
                        moved += 1
            return moved

        def status(self):
            """Возвращает текущее состояние."""
            return {
                f"A{i}": {"tasks": len(q), "capacity": c, "load": f"{len(q)/c:.1%}"}
                for i, (q, c) in enumerate(zip(self.queues, self.capacities))
            }

    balancer = AdaptiveBalancer(4)

    # Добавляем задачи неравномерно
    print("Добавляем 15 задач неравномерно...")
    for i in range(15):
        balancer.add_task(f"T{i}")

    print("До балансировки:")
    for name, info in balancer.status().items():
        print(f"  {name}: {info['tasks']} задач, загрузка={info['load']}")

    # Перераспределяем
    moves = balancer.rebalance()
    print(f"\nПосле перераспределения (перемещено: {moves}):")
    for name, info in balancer.status().items():
        print(f"  {name}: {info['tasks']} задач, загрузка={info['load']}")

    print("\nАдаптивный балансировщик поддерживает равномерную нагрузку")


# ============================================================
# Демо 3: Resource Management — управление ресурсами
# ============================================================

def demo_resource_management():
    """Пулы ресурсов, резервирование и перехват ресурсов."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: Resource Management — управление ресурсами")
    print("=" * 70)

    # --- 3.1 Пул ресурсов ---
    print("\n--- 3.1 Пул ресурсов (Resource Pool) ---")
    print("Пул GPU-устройств, выделяемых агентам\n")

    class ResourcePool:
        """Пул ресурсов с выдачей и возвратом."""

        def __init__(self, resources):
            self.available = list(resources)
            self.allocated = {}

        def acquire(self, agent_id, resource_type=None):
            """Запросить ресурс из пула."""
            for r in self.available:
                if resource_type is None or r["type"] == resource_type:
                    self.available.remove(r)
                    self.allocated[agent_id] = r
                    return r
            return None  # нет свободных ресурсов

        def release(self, agent_id):
            """Вернуть ресурс в пул."""
            if agent_id in self.allocated:
                resource = self.allocated.pop(agent_id)
                self.available.append(resource)
                return resource
            return None

        def status(self):
            """Состояние пула."""
            return {
                "available": len(self.available),
                "allocated": len(self.allocated),
                "total": len(self.available) + len(self.allocated),
            }

    # Создаём пул GPU
    gpu_pool = ResourcePool([
        {"id": "GPU-0", "type": "A100", "memory_gb": 80},
        {"id": "GPU-1", "type": "A100", "memory_gb": 80},
        {"id": "GPU-2", "type": "V100", "memory_gb": 32},
        {"id": "GPU-3", "type": "T4", "memory_gb": 16},
    ])

    print(f"Пул создан: {gpu_pool.status()}")

    # Агенты запрашивают ресурсы
    agents_to_allocate = ["Agent-0", "Agent-1", "Agent-2", "Agent-3", "Agent-4"]
    for agent in agents_to_allocate:
        resource = gpu_pool.acquire(agent, resource_type="A100")
        if resource:
            print(f"  {agent} -> {resource['id']} ({resource['type']}, {resource['memory_gb']}GB)")
        else:
            resource = gpu_pool.acquire(agent)  # берём любой доступный
            if resource:
                print(f"  {agent} -> {resource['id']} ({resource['type']}, {resource['memory_gb']}GB)")
            else:
                print(f"  {agent} -> ОЖИДАНИЕ (нет свободных ресурсов)")

    print(f"\nСостояние пула: {gpu_pool.status()}")

    # Освобождаем ресурс
    freed = gpu_pool.release("Agent-0")
    print(f"\nAgent-0 освободил: {freed}")
    print(f"Состояние пула: {gpu_pool.status()}")

    # --- 3.2 Резервирование ресурсов ---
    print("\n--- 3.2 Резервирование ресурсов ---")
    print("Агенты резервируют ресурсы на будущее\n")

    class ReservationSystem:
        """Система резервирования с временными окнами."""

        def __init__(self):
            self.reservations = []

        def reserve(self, agent_id, resource, start_time, duration):
            """Зарезервировать ресурс на заданное время."""
            end_time = start_time + duration

            # Проверяем конфликты с существующими резервациями
            for res in self.reservations:
                if res["resource"] == resource:
                    existing_start = res["start"]
                    existing_end = res["start"] + res["duration"]
                    # Проверка пересечения интервалов
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False, f"Конфликт с {res['agent']}"

            self.reservations.append({
                "agent": agent_id,
                "resource": resource,
                "start": start_time,
                "duration": duration,
            })
            return True, "Зарезервировано"

        def get_availability(self, resource, time_point):
            """Проверить доступность ресурса в момент времени."""
            for res in self.reservations:
                if res["resource"] == resource:
                    if res["start"] <= time_point < res["start"] + res["duration"]:
                        return False, res["agent"]
            return True, None

    reservation_sys = ReservationSystem()

    # Резервируем GPU на разное время
    reservations = [
        ("Agent-A", "GPU-0", 0, 5),
        ("Agent-B", "GPU-0", 3, 4),  # конфликт с Agent-A
        ("Agent-C", "GPU-1", 2, 6),
        ("Agent-D", "GPU-0", 6, 3),  # после Agent-A
    ]

    for agent, resource, start, duration in reservations:
        success, msg = reservation_sys.reserve(agent, resource, start, duration)
        status = "OK" if success else "ОТКАЗ"
        print(f"  {agent} резервирует {resource} [t={start}..{start+duration}]: {status} ({msg})")

    print("\nПроверка доступности GPU-0 в t=4:")
    avail, holder = reservation_sys.get_availability("GPU-0", 4)
    if avail:
        print("  Доступен")
    else:
        print(f"  Занят агентом {holder}")

    # --- 3.3 Перехват ресурсов (preemption) ---
    print("\n--- 3.3 Перехват ресурсов (Preemption) ---")
    print("Высокоприоритетные задачи могут перехватить ресурсы\n")

    class PreemptiveAllocator:
        """Аллокатор с приоритетами и перехватом."""

        def __init__(self):
            self.allocations = {}  # agent -> {"resource", "priority", "task"}

        def allocate(self, agent_id, resource, priority, task):
            """Выделить ресурс с проверкой перехвата."""
            # Если ресурс уже выдан
            for other_agent, alloc in self.allocations.items():
                if alloc["resource"] == resource:
                    if priority > alloc["priority"]:
                        # Перехватываем!
                        preempted_task = alloc["task"]
                        del self.allocations[other_agent]
                        self.allocations[agent_id] = {
                            "resource": resource,
                            "priority": priority,
                            "task": task,
                        }
                        return True, f"Перехвачено у {other_agent} (задача {preempted_task})"
                    else:
                        return False, f"Приоритет {priority} <= текущего {alloc['priority']}"

            self.allocations[agent_id] = {
                "resource": resource,
                "priority": priority,
                "task": task,
            }
            return True, "Выделено"

    preemptor = PreemptiveAllocator()

    # Низкоприоритетная задача получает ресурс
    ok, msg = preemptor.allocate("Low-Agent", "GPU-0", priority=1, task="обучение")
    print(f"  Low-Agent (приоритет=1) -> GPU-0: {ok} — {msg}")

    # Ещё одна низкоприоритетная задача
    ok, msg = preemptor.allocate("Low-Agent-2", "GPU-0", priority=2, task="дообучение")
    print(f"  Low-Agent-2 (приоритет=2) -> GPU-0: {ok} — {msg}")

    # Высокоприоритетная задача перехватывает
    ok, msg = preemptor.allocate("High-Agent", "GPU-0", priority=10, task="инференс")
    print(f"  High-Agent (приоритет=10) -> GPU-0: {ok} — {msg}")

    # Ещё одна низкоприоритетная — уже не перехватит
    ok, msg = preemptor.allocate("Low-Agent-3", "GPU-0", priority=3, task="тестирование")
    print(f"  Low-Agent-3 (приоритет=3) -> GPU-0: {ok} — {msg}")

    print("\nТекущие выделения:")
    for agent, alloc in preemptor.allocations.items():
        print(f"  {agent}: {alloc['resource']} (приоритет={alloc['priority']}, задача={alloc['task']})")

    # --- 3.4 Динамическое масштабирование ---
    print("\n--- 3.4 Динамическое масштабирование пула ---")
    print("Пул автоматически расширяется при нехватке ресурсов\n")

    class ScalablePool:
        """Пул ресурсов с автоматическим масштабированием."""

        def __init__(self, initial_size, max_size, scale_up_threshold=0.8):
            self.resources = [{"id": f"R-{i}", "active": True} for i in range(initial_size)]
            self.max_size = max_size
            self.scale_up_threshold = scale_up_threshold
            self.allocated = 0
            self.history = []

        def utilization(self):
            """Текущая загрузка пула."""
            active = sum(1 for r in self.resources if r["active"])
            return self.allocated / active if active > 0 else 0

        def acquire(self):
            """Попытка выделить ресурс с масштабированием."""
            active = [r for r in self.resources if r["active"]]
            if self.allocated < len(active):
                self.allocated += 1
                return True

            # Нужно масштабировать
            if len(self.resources) < self.max_size:
                new_resource = {"id": f"R-{len(self.resources)}", "active": True}
                self.resources.append(new_resource)
                self.allocated += 1
                self.history.append(f"Масштабирование: добавлен {new_resource['id']}")
                return True

            return False  # достигнут максимум

        def release(self):
            """Освободить ресурс."""
            if self.allocated > 0:
                self.allocated -= 1

    pool = ScalablePool(initial_size=3, max_size=6, scale_up_threshold=0.8)

    print(f"Начальный размер пула: {len(pool.resources)}")
    print(f"Максимум: {pool.max_size}")

    # Заполняем пул
    for i in range(8):
        ok = pool.acquire()
        util = pool.utilization()
        print(f"  Запрос {i+1}: {'OK' if ok else 'ОТКАЗ'} | ресурсов={len(pool.resources)}, загрузка={util:.0%}")

    print(f"\nИстория масштабирования:")
    for event in pool.history:
        print(f"  {event}")

    print(f"\nФинальное состояние: ресурсов={len(pool.resources)}, занято={pool.allocated}")


# ============================================================
# Демо 4: Work Stealing — кража задач
# ============================================================

def demo_work_stealing():
    """Метод work stealing: свободные агенты крадут задачи у занятых."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: Work Stealing — кража задач")
    print("=" * 70)

    # --- 4.1 Базовый work stealing ---
    print("\n--- 4.1 Базовый механизм work stealing ---")
    print("Свободный агент берёт задачу из очереди самого загруженного\n")

    class WorkStealingQueue:
        """Очередь задач с поддержкой кражи."""

        def __init__(self, owner):
            self.owner = owner
            self.tasks = []

        def push(self, task):
            """Добавить задачу в конец очереди (LIFO для owner)."""
            self.tasks.append(task)

        def pop(self):
            """Взять задачу из конца (LIFO)."""
            if self.tasks:
                return self.tasks.pop()
            return None

        def steal(self):
            """Украсть задачу из начала (FIFO для thieves — минимум конфликтов)."""
            if self.tasks:
                return self.tasks.pop(0)
            return None

        def size(self):
            return len(self.tasks)

    # Создаём очереди для 4 агентов
    queues = {f"A{i}": WorkStealingQueue(f"A{i}") for i in range(4)}

    # Загружаем задачи неравномерно
    tasks_distribution = {
        "A0": ["T0", "T1", "T2", "T3", "T4", "T5"],  # 6 задач
        "A1": ["T6", "T7", "T8"],                       # 3 задачи
        "A2": ["T9", "T10", "T11", "T12"],              # 4 задачи
        "A3": [],                                         # 0 задач
    }

    for agent, tasks in tasks_distribution.items():
        for task in tasks:
            queues[agent].push(task)

    print("Начальное состояние:")
    for name, q in queues.items():
        print(f"  {name}: {q.size()} задач — {q.tasks}")

    # Симуляция: A3 свободен и крадёт задачу у A0
    print("\nСимуляция: A3 свободен, крадёт задачу у A0")
    stolen = queues["A3"].steal()  # крадём из начала очереди A0
    print(f"  A3 украл {stolen} из очереди A0")
    print(f"  A0: {queues['A0'].tasks}")
    print(f"  A3: {queues['A3'].tasks}")

    # --- 4.2 Двойная очередь (deque) ---
    print("\n--- 4.2 Двойная очередь для work stealing ---")
    print("Owner добавляет в конец, thieves крадут из начала\n")

    class DequeWorkStealing:
        """Двойная очередь: owner push/pop с конца, thieves steal с начала."""

        def __init__(self, owner_id):
            self.owner = owner_id
            self.left = []   # начало (для кражи)
            self.right = []  # конец (для owner)

        def push(self, task):
            """Owner добавляет задачу."""
            self.right.append(task)

        def pop(self):
            """Owner берёт задачу (LIFO с правого конца)."""
            if self.right:
                return self.right.pop()
            # Переливаем из левой части
            self.left, self.right = self.right, self.left[::-1]
            return self.right.pop() if self.right else None

        def steal(self):
            """Thief крадёт задачу (FIFO из левого конца)."""
            if self.left:
                return self.left.pop(0)
            if self.right:
                return self.right.pop(0)
            return None

        def total(self):
            return len(self.left) + len(self.right)

    deques = {f"A{i}": DequeWorkStealing(f"A{i}") for i in range(4)}

    # A0 получает много задач
    for i in range(8):
        deques["A0"].push(f"T{i}")

    print(f"До кражи: A0.left={deques['A0'].left}, A0.right={deques['A0'].right}")

    # Кража из разных частей
    for _ in range(3):
        stolen = deques["A0"].steal()
        print(f"  Украдено: {stolen} | A0.left={deques['A0'].left}, A0.right={deques['A0'].right}")

    # --- 4.3 Work sharing vs work stealing ---
    print("\n--- 4.3 Work Sharing vs Work Stealing ---")
    print("Sharing: загруженный агент раздаёт. Stealing: свободный забирает\n")

    def simulate_work_distribution(method, agent_tasks, n_steps):
        """Симуляция распределения задач."""
        history = []
        tasks = dict(agent_tasks)  # копия

        for step in range(n_steps):
            loads = {a: len(t) for a, t in tasks.items()}

            if method == "sharing":
                # Самый загруженный раздаёт самому свободному
                max_agent = max(loads, key=loads.get)
                min_agent = min(loads, key=loads.get)
                if loads[max_agent] - loads[min_agent] > 1:
                    task = tasks[max_agent].pop()
                    tasks[min_agent].append(task)
                    history.append((step, f"{max_agent} -> {min_agent}", f"T{task}"))
            else:  # stealing
                # Самый свободный крадёт у самого загруженного
                min_agent = min(loads, key=loads.get)
                max_agent = max(loads, key=loads.get)
                if loads[max_agent] - loads[min_agent] > 1 and tasks[max_agent]:
                    task = tasks[max_agent].pop(0)  # steal from front
                    tasks[min_agent].append(task)
                    history.append((step, f"{min_agent} <- {max_agent}", f"T{task}"))

        return tasks, history

    initial = {
        "A0": list(range(10)),
        "A1": list(range(10, 13)),
        "A2": list(range(13, 15)),
        "A3": list(range(15, 16)),
    }

    print("Начальное распределение:")
    for a, t in initial.items():
        print(f"  {a}: {len(t)} задач")

    # Sharing
    tasks_sharing, history_sharing = simulate_work_distribution(
        "sharing", {k: list(v) for k, v in initial.items()}, 15
    )
    print(f"\nWork Sharing (перемещений: {len(history_sharing)}):")
    for step, direction, task in history_sharing[:5]:
        print(f"  Шаг {step}: {direction} ({task})")
    if len(history_sharing) > 5:
        print(f"  ... и ещё {len(history_sharing) - 5}")
    print("  Итог:", {a: len(t) for a, t in tasks_sharing.items()})

    # Stealing
    tasks_stealing, history_stealing = simulate_work_distribution(
        "stealing", {k: list(v) for k, v in initial.items()}, 15
    )
    print(f"\nWork Stealing (перемещений: {len(history_stealing)}):")
    for step, direction, task in history_stealing[:5]:
        print(f"  Шаг {step}: {direction} ({task})")
    if len(history_stealing) > 5:
        print(f"  ... и ещё {len(history_stealing) - 5}")
    print("  Итог:", {a: len(t) for a, t in tasks_stealing.items()})

    # --- 4.4 Диффузия нагрузки ---
    print("\n--- 4.4 Диффузия нагрузки (Load Diffusion) ---")
    print("Задачи «протекают» от загруженных к свободным агентам\n")

    def load_diffusion(loads, n_rounds, diffusion_rate=0.3):
        """Модель диффузии: нагрузка выравнивается пропорционально разнице."""
        history = [loads[:]]
        agents = len(loads)

        for round_num in range(n_rounds):
            new_loads = loads[:]
            for i in range(agents):
                for j in range(i + 1, agents):
                    # Задачи текут от более загруженного к менее загруженному
                    diff = loads[i] - loads[j]
                    if diff > 0:
                        transfer = int(diff * diffusion_rate)
                        if transfer > 0:
                            new_loads[i] -= transfer
                            new_loads[j] += transfer
                    elif diff < 0:
                        transfer = int(-diff * diffusion_rate)
                        if transfer > 0:
                            new_loads[j] -= transfer
                            new_loads[i] += transfer
            loads = new_loads
            history.append(loads[:])

        return history

    initial_loads = [20, 5, 15, 2, 18, 3]
    print(f"Начальная нагрузка агентов: {initial_loads}")
    print(f"Среднее: {sum(initial_loads)/len(initial_loads):.1f}, разброс: {max(initial_loads) - min(initial_loads)}")

    diffusion_history = load_diffusion(initial_loads, n_rounds=8, diffusion_rate=0.3)

    print("\nДиффузия по раундам:")
    for i, loads in enumerate(diffusion_history):
        variance = sum((x - sum(loads)/len(loads))**2 for x in loads) / len(loads)
        bar = " | ".join(f"{l:3d}" for l in loads)
        print(f"  Раунд {i}: [{bar}]  дисперсия={variance:.1f}")

    final = diffusion_history[-1]
    print(f"\nФинал: {final}")
    print(f"Разброс уменьшился: {max(initial_loads) - min(initial_loads)} -> {max(final) - min(final)}")


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    demo_task_allocation()
    demo_load_balancing()
    demo_resource_management()
    demo_work_stealing()
