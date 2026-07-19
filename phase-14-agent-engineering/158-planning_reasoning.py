"""158 — Planning & Reasoning: декомпозиция задач, ReAct, chain of thought

Темы:
  1. Task Decomposition (hierarchical planning, subtask generation)
  2. ReAct Pattern (Reason + Act loop, observation formatting)
  3. Plan-and-Execute (generate plan → execute steps → replan)
  4. Reflection & Self-Correction (error detection, backtracking, retry strategies)

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
# 1. Task Decomposition — декомпозиция задач на подзадачи
# ============================================================

def demo_task_decomposition():
    print("=" * 70)
    print("DEMO 1: Task Decomposition — иерархическое планирование")
    print("=" * 70)

    # --- 1.1 Простая декомпозиция ---
    print("\n--- 1.1 Простая декомпозиция задачи ---")

    def decompose_task(task, depth=0, max_depth=3):
        """Декомпозиция задачи на подзадачи (рекурсивно)."""
        # Базовый случай: не углубляемся дальше max_depth
        if depth >= max_depth:
            return {"task": task, "subtasks": [], "leaf": True}

        # Генерация подзадач (имитация)
        subtask_templates = {
            "написать код": ["спроектировать архитектуру", "реализовать функции",
                           "написать тесты", "задокументировать"],
            "исследовать тему": ["определить область", "найти источники",
                                "проанализировать данные", "сформулировать выводы"],
            "спроектировать систему": ["определить требования", "выбрать технологию",
                                      "создать прототип", "провести тестирование"],
        }

        subtasks = subtask_templates.get(task, [f"шаг {i+1}: {task}" for i in range(3)])

        # Рекурсивная декомпозиция подзадач
        decomposed = []
        for subtask in subtasks:
            if depth < max_depth - 1 and random.random() < 0.5:
                decomposed.append(decompose_task(subtask, depth + 1, max_depth))
            else:
                decomposed.append({"task": subtask, "subtasks": [], "leaf": True})

        return {"task": task, "subtasks": decomposed, "leaf": False}

    # Декомпозиция задачи
    plan = decompose_task("написать код")

    def print_plan(node, indent=0):
        """Печать плана с отступами."""
        prefix = "  " * indent
        leaf = " [ЛИСТ]" if node["leaf"] else ""
        print(f"  {prefix}- {node['task']}{leaf}")
        for sub in node["subtasks"]:
            print_plan(sub, indent + 1)

    print("  Задача: 'написать код'")
    print_plan(plan)

    # --- 1.2 Подсчёт подзадач ---
    print("\n--- 1.2 Подсчёт подзадач ---")

    def count_tasks(node):
        """Подсчёт общего количества подзадач."""
        count = 1
        for sub in node["subtasks"]:
            count += count_tasks(sub)
        return count

    def count_leaves(node):
        """Подсчёт листовых (конечных) подзадач."""
        if node["leaf"]:
            return 1
        count = 0
        for sub in node["subtasks"]:
            count += count_leaves(sub)
        return count

    total = count_tasks(plan)
    leaves = count_leaves(plan)
    print(f"  Всего задач: {total}")
    print(f"  Листовых (конечных): {leaves}")
    print(f"  Промежуточных: {total - leaves}")

    # --- 1.3 Иерархия задач с приоритетами ---
    print("\n--- 1.3 Иерархия задач с приоритетами ---")

    class TaskNode:
        """Узел задачи с приоритетом и зависимостями."""

        def __init__(self, name, priority=0, dependencies=None):
            self.name = name
            self.priority = priority
            self.dependencies = dependencies or []
            self.subtasks = []
            self.status = "pending"

        def add_subtask(self, subtask):
            """Добавление подзадачи."""
            self.subtasks.append(subtask)

        def get_execution_order(self, visited=None):
            """Определение порядка выполнения (拓扑排序)."""
            if visited is None:
                visited = set()

            order = []
            for dep_name in self.dependencies:
                if dep_name not in visited:
                    visited.add(dep_name)
                    order.append(dep_name)

            if self.name not in visited:
                visited.add(self.name)
                order.append(self.name)

            for sub in self.subtasks:
                order.extend(sub.get_execution_order(visited))

            return order

    # Создание дерева задач
    root = TaskNode("Проект", priority=0)
    design = TaskNode("Дизайн", priority=3)
    design.add_subtask(TaskNode("UI макет", priority=2, dependencies=["Дизайн"]))
    design.add_subtask(TaskNode("API дизайн", priority=1, dependencies=["Дизайн"]))

    impl = TaskNode("Реализация", priority=5, dependencies=["Дизайн"])
    impl.add_subtask(TaskNode("Backend", priority=4, dependencies=["Реализация"]))
    impl.add_subtask(TaskNode("Frontend", priority=3, dependencies=["Реализация"]))

    test = TaskNode("Тестирование", priority=2, dependencies=["Реализация"])

    root.add_subtask(design)
    root.add_subtask(impl)
    root.add_subtask(test)

    # Порядок выполнения
    order = root.get_execution_order()
    print(f"  Порядок выполнения: {order}")

    # Сортировка по приоритету
    all_tasks = []
    def collect_tasks(node):
        all_tasks.append((node.name, node.priority))
        for sub in node.subtasks:
            collect_tasks(sub)
    collect_tasks(root)

    by_priority = sorted(all_tasks, key=lambda x: -x[1])
    print("  По приоритету:")
    for name, prio in by_priority:
        bar = "█" * prio
        print(f"    {name}: {prio} {bar}")

    # --- 1.4 Параллельная декомпозиция ---
    print("\n--- 1.4 Параллельная декомпозиция (волновая) ---")

    def wave_decomposition(task, max_waves=4):
        """Декомпозиция волнами — каждая волна разбивает предыдущую."""
        waves = [[task]]

        for wave_num in range(max_waves - 1):
            next_wave = []
            for t in waves[-1]:
                # Разбиение на 2-4 подзадачи
                n_sub = random.randint(2, 4)
                for i in range(n_sub):
                    next_wave.append(f"{t}.{i+1}")
            waves.append(next_wave)

        return waves

    waves = wave_decomposition("Проект")
    for i, wave in enumerate(waves):
        print(f"  Волна {i+1} ({len(wave)} задач): {wave[:3]}{'...' if len(wave) > 3 else ''}")

    print()


# ============================================================
# 2. ReAct Pattern — Reason + Act loop
# ============================================================

def demo_react_pattern():
    print("=" * 70)
    print("DEMO 2: ReAct Pattern — цикл Reason + Act + Observe")
    print("=" * 70)

    # --- 2.1 Базовый ReAct цикл ---
    print("\n--- 2.1 Базовый ReAct цикл ---")

    class ReActAgent:
        """Агент, реализующий паттерн ReAct."""

        def __init__(self, tools=None):
            self.tools = tools or {}
            self.trajectory = []
            self.max_steps = 5

        def reason(self, observation, goal):
            """Шаг рассуждения: анализ наблюдения и формулировка следующего действия."""
            # Простая логика принятия решений
            if observation is None:
                return {"thought": "Начинаю выполнение задачи", "action": "search",
                       "action_input": goal}

            if "найдено" in str(observation).lower():
                return {"thought": "Получил достаточно информации, анализирую",
                       "action": "analyze", "action_input": observation}

            return {"thought": "Продолжаю поиск", "action": "search",
                   "action_input": goal}

        def act(self, action_name, action_input):
            """Шаг действия: вызов инструмента."""
            if action_name in self.tools:
                result = self.tools[action_name](action_input)
                return {"action": action_name, "input": action_input, "result": result}
            return {"action": action_name, "input": action_input,
                   "result": f"Инструмент '{action_name}' не найден"}

        def observe(self, action_result):
            """Шаг наблюдения: форматирование результата."""
            return f"Результат: {action_result['result']}"

        def run(self, goal):
            """Основной цикл ReAct."""
            observation = None

            for step in range(self.max_steps):
                # Reason
                decision = self.reason(observation, goal)
                print(f"  Шаг {step+1}:")
                print(f"    Thought: {decision['thought']}")
                print(f"    Action: {decision['action']}({decision['action_input']})")

                # Act
                action_result = self.act(decision["action"], decision["action_input"])

                # Observe
                observation = self.observe(action_result)
                print(f"    Observation: {observation}")

                self.trajectory.append({
                    "step": step + 1,
                    "thought": decision["thought"],
                    "action": decision["action"],
                    "observation": observation,
                })

                # Условие остановки
                if decision["action"] == "analyze":
                    print(f"    → Задача выполнена на шаге {step+1}")
                    break

            return self.trajectory

    # Простые инструменты
    tools = {
        "search": lambda q: f"Найдена информация по запросу '{q}'",
        "analyze": lambda data: f"Анализ завершён: {data}",
    }

    agent = ReActAgent(tools)
    trajectory = agent.run("найти информацию о Python")

    # --- 2.2 Форматирование наблюдений ---
    print("\n--- 2.2 Форматирование наблюдений ---")

    def format_observation(obs_type, data):
        """Форматирование наблюдения для ReAct."""
        formatters = {
            "text": lambda d: f"Текст: {d}",
            "search_result": lambda d: f"Результат поиска ({len(d)} записей): {d[:3]}...",
            "error": lambda d: f"Ошибка: {d}",
            "numeric": lambda d: f"Число: {d} (тип: {type(d).__name__})",
            "dict": lambda d: f"Данные: {json.dumps(d, ensure_ascii=False)[:100]}",
        }
        formatter = formatters.get(obs_type, lambda d: f"{obs_type}: {d}")
        return formatter(data)

    observations = [
        ("text", "Python — высокоуровневый язык программирования"),
        ("search_result", ["Python 3.12", "Python 3.11", "Python 3.10", "Python 2.7"]),
        ("error", "Timeout: превышен лимит времени"),
        ("numeric", 42),
        ("dict", {"status": "ok", "count": 15, "query": "python"}),
    ]

    for obs_type, data in observations:
        formatted = format_observation(obs_type, data)
        print(f"  {formatted}")

    # --- 2.3 Tracing (трассировка) ---
    print("\n--- 2.3 Трассировка ReAct цикла ---")

    class ReActTracer:
        """Трассировщик для анализа ReAct цикла."""

        def __init__(self):
            self.trace = []

        def log(self, step_type, content, duration=0):
            """Логирование шага."""
            self.trace.append({
                "type": step_type,
                "content": content,
                "duration": duration,
                "timestamp": len(self.trace),
            })

        def get_summary(self):
            """Получение сводки по трассировке."""
            type_counts = collections.Counter(t["type"] for t in self.trace)
            total_duration = sum(t["duration"] for t in self.trace)
            return {
                "steps": len(self.trace),
                "by_type": dict(type_counts),
                "total_duration": total_duration,
            }

        def print_trace(self):
            """Печать трассировки."""
            for entry in self.trace:
                icon = {"reason": "💭", "act": "⚡", "observe": "👁️"}.get(entry["type"], "•")
                print(f"    {icon} [{entry['type']}] {entry['content'][:60]}")

    tracer = ReActTracer()

    # Имитация ReAct цикла
    tracer.log("reason", "Анализирую задачу: найти максимум в списке")
    tracer.log("act", "Вызываю инструмент: find_max([3, 7, 2, 9, 4])", 0.001)
    tracer.log("observe", "Результат: 9")
    tracer.log("reason", "Получил результат, проверяю корректность")
    tracer.log("act", "Вызываю инструмент: verify(9, [3, 7, 2, 9, 4])", 0.0005)
    tracer.log("observe", "Верификация пройдена: 9确实是最大值")

    print("  Трассировка ReAct:")
    tracer.print_trace()

    summary = tracer.get_summary()
    print(f"\n  Сводка: {summary}")

    # --- 2.4 ReAct с ограниченной длиной контекста ---
    print("\n--- 2.4 ReAct с управлением контекстом ---")

    def manage_react_context(trajectory, max_history=3):
        """Управление длиной контекста в ReAct."""
        if len(trajectory) <= max_history:
            return trajectory

        # Оставляем последние max_history шагов + резюме старых
        old_steps = trajectory[:-max_history]
        recent_steps = trajectory[-max_history:]

        # Создание резюме
        summary = {
            "type": "summary",
            "content": f"Ранее выполнено {len(old_steps)} шагов",
            "key_findings": [],
        }

        for step in old_steps:
            if step.get("observation"):
                summary["key_findings"].append(step["observation"][:50])

        return [summary] + recent_steps

    # Длинная траектория
    long_trajectory = [
        {"step": i, "thought": f"Рассуждение {i}", "action": f"действие {i}",
         "observation": f"Наблюдение {i}"}
        for i in range(1, 8)
    ]

    print(f"  Полная траектория: {len(long_trajectory)} шагов")
    managed = manage_react_context(long_trajectory, max_history=3)
    print(f"  После управления контекстом: {len(managed)} элементов")
    for item in managed:
        if isinstance(item, dict) and item.get("type") == "summary":
            print(f"    [РЕЗЮМЕ] {item['content']}")
            print(f"      Ключевые находки: {item['key_findings'][:2]}")
        else:
            print(f"    [Шаг {item['step']}] {item['observation']}")

    print()


# ============================================================
# 3. Plan-and-Execute — генерация плана и выполнение шагов
# ============================================================

def demo_plan_and_execute():
    print("=" * 70)
    print("DEMO 3: Plan-and-Execute — план → выполнение → перепланирование")
    print("=" * 70)

    # --- 3.1 Генерация плана ---
    print("\n--- 3.1 Генерация плана из задачи ---")

    class Planner:
        """Генератор планов из задач."""

        def __init__(self):
            self.plan_templates = {
                "исследование": [
                    "Определить область исследования",
                    "Найти релевантные источники",
                    "Проанализировать данные",
                    "Сформулировать выводы",
                ],
                "разработка": [
                    "Определить требования",
                    "Спроектировать архитектуру",
                    "Реализовать компоненты",
                    "Протестировать",
                    "Задеплоить",
                ],
                "анализ": [
                    "Собрать данные",
                    "Очистить данные",
                    "Провести анализ",
                    "Визуализировать результаты",
                    "Подготовить отчёт",
                ],
            }

        def generate_plan(self, task_type, context=""):
            """Генерация плана по типу задачи."""
            steps = self.plan_templates.get(task_type, [
                "Шаг 1: Подготовка",
                "Шаг 2: Основная работа",
                "Шаг 3: Проверка",
            ])

            plan = []
            for i, step in enumerate(steps):
                plan.append({
                    "id": i + 1,
                    "description": step,
                    "status": "pending",
                    "estimated_time": random.randint(5, 30),
                    "dependencies": [i] if i > 0 else [],
                })

            return plan

    planner = Planner()
    plan = planner.generate_plan("анализ", "Данные о продажах")

    print("  План:")
    for step in plan:
        deps = f" (зависимости: {step['dependencies']})" if step["dependencies"] else ""
        print(f"    {step['id']}. {step['description']} "
              f"[~{step['estimated_time']}мин]{deps}")

    # --- 3.2 Выполнение плана ---
    print("\n--- 3.2 Выполнение плана ---")

    class PlanExecutor:
        """Исполнитель плана."""

        def __init__(self):
            self.results = []

        def execute_step(self, step):
            """Выполнение одного шага плана."""
            start = time.time()
            # Имитация выполнения
            success = random.random() > 0.15  # 85% успеха
            duration = time.time() - start

            result = {
                "step_id": step["id"],
                "description": step["description"],
                "success": success,
                "duration": duration,
                "output": f"Результат шага {step['id']}" if success else None,
                "error": f"Ошибка в шаге {step['id']}" if not success else None,
            }

            self.results.append(result)
            return result

        def execute_plan(self, plan):
            """Выполнение всего плана."""
            for step in plan:
                result = self.execute_step(step)
                status = "OK" if result["success"] else f"ОШИБКА: {result['error']}"
                print(f"    Шаг {result['step_id']}: {status}")

            return self.results

    executor = PlanExecutor()
    results = executor.execute_plan(plan)

    # Статистика
    successes = sum(1 for r in results if r["success"])
    print(f"\n  Итого: {successes}/{len(results)} шагов выполнено успешно")

    # --- 3.3 Перепланирование ---
    print("\n--- 3.3 Перепланирование при ошибках ---")

    def replan(original_plan, failed_steps, new_context=""):
        """Перепланирование с учётом ошибок."""
        new_plan = []

        for step in original_plan:
            if step["id"] in failed_steps:
                # Модификация провалившегося шага
                new_plan.append({
                    **step,
                    "id": step["id"],
                    "description": f"[РЕТРАЙ] {step['description']}",
                    "status": "retry",
                    "estimated_time": step["estimated_time"] * 2,
                })
            else:
                new_plan.append(step)

        # Добавление шага проверки
        new_plan.append({
            "id": len(original_plan) + 1,
            "description": "Проверка результатов после перепланирования",
            "status": "pending",
            "estimated_time": 5,
            "dependencies": [],
        })

        return new_plan

    failed = [r["step_id"] for r in results if not r["success"]]
    if failed:
        print(f"  Провалившиеся шаги: {failed}")
        new_plan = replan(plan, failed)
        print("  Новый план:")
        for step in new_plan:
            status = f"[{step['status'].upper()}]" if step['status'] != 'pending' else ""
            print(f"    {step['id']}. {step['description']} {status}")
    else:
        print("  Все шаги выполнены успешно, перепланирование не требуется")

    # --- 3.4 Plan-and-Execute с replan ---
    print("\n--- 3.4 Полный цикл Plan-and-Execute с replan ---")

    class PlanAndExecuteAgent:
        """Агент с полным циклом Plan-and-Execute."""

        def __init__(self, planner, executor):
            self.planner = planner
            self.executor = executor
            self.max_replans = 2
            self.plan_history = []

        def solve(self, task_type, context=""):
            """Решение задачи с перепланированием."""
            plan = self.planner.generate_plan(task_type, context)
            self.plan_history.append({"plan": plan.copy(), "attempt": 1})

            for attempt in range(self.max_replans + 1):
                print(f"\n  Попытка {attempt + 1}:")

                # Выполнение
                results = self.executor.execute_plan(plan)

                # Проверка
                failed = [r["step_id"] for r in results if not r["success"]]

                if not failed:
                    print(f"  → Задача решена за {attempt + 1} попытку")
                    return {"success": True, "attempts": attempt + 1, "results": results}

                if attempt < self.max_replans:
                    print(f"  → Перепланирование (ошибки: {failed})")
                    plan = replan(plan, failed)
                    self.plan_history.append({"plan": plan.copy(), "attempt": attempt + 2})

            return {"success": False, "attempts": self.max_replans + 1, "results": results}

    agent = PlanAndExecuteAgent(planner, executor)
    result = agent.solve("исследование", "Новые данные")

    print(f"\n  Итог: {'успех' if result['success'] else 'неудача'}, "
          f"попыток: {result['attempts']}")

    print()


# ============================================================
# 4. Reflection & Self-Correction — рефлексия и самокоррекция
# ============================================================

def demo_reflection():
    print("=" * 70)
    print("DEMO 4: Reflection & Self-Correction — анализ ошибок и исправление")
    print("=" * 70)

    # --- 4.1 Детекция ошибок ---
    print("\n--- 4.1 Детекция ошибок в рассуждениях ---")

    class ReflectionEngine:
        """Движок рефлексии для анализа рассуждений."""

        def __init__(self):
            self.error_patterns = {
                "противоречие": ["однако", "но", "вопреки", "противоречит"],
                "неуверенность": ["возможно", "наверное", "кажется", "也许"],
                "неполнота": ["неизвестно", "нет данных", "нужно уточнить"],
            }

        def analyze(self, reasoning_text):
            """Анализ текста рассуждений на наличие ошибок."""
            issues = []

            for error_type, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if pattern in reasoning_text.lower():
                        issues.append({
                            "type": error_type,
                            "pattern": pattern,
                            "context": reasoning_text[:100],
                        })

            return issues

    engine = ReflectionEngine()

    test_reasoning = [
        "Python — лучший язык, однако Go быстрее для серверов",
        "Результат: 42. Возможно, стоит перепроверить расчёты",
        "Нужно уточнить требования клиента перед началом работы",
        "Код написан и протестирован. Все тесты пройдены.",
    ]

    for text in test_reasoning:
        issues = engine.analyze(text)
        if issues:
            print(f"  Текст: '{text[:50]}...'")
            for issue in issues:
                print(f"    ⚠ {issue['type']}: '{issue['pattern']}'")

    # --- 4.2 Backtracking ---
    print("\n--- 4.2 Backtracking (возврат к предыдущему состоянию) ---")

    class CheckpointManager:
        """Менеджер контрольных точек для backtracking."""

        def __init__(self):
            self.checkpoints = []

        def save(self, state, description=""):
            """Сохранение контрольной точки."""
            checkpoint = {
                "id": len(self.checkpoints) + 1,
                "state": state.copy() if isinstance(state, dict) else state,
                "description": description,
            }
            self.checkpoints.append(checkpoint)
            return checkpoint["id"]

        def restore(self, checkpoint_id):
            """Восстановление из контрольной точки."""
            for cp in self.checkpoints:
                if cp["id"] == checkpoint_id:
                    return cp["state"]
            return None

        def list_checkpoints(self):
            """Список всех контрольных точек."""
            return [(cp["id"], cp["description"]) for cp in self.checkpoints]

    manager = CheckpointManager()

    # Имитация работы с контрольными точками
    state = {"step": 0, "data": [], "score": 0}
    manager.save(state, "Начальное состояние")

    # Шаг 1
    state["step"] = 1
    state["data"].append("шаг1")
    state["score"] = 10
    manager.save(state, "После шага 1")

    # Шаг 2
    state["step"] = 2
    state["data"].append("шаг2")
    state["score"] = 5  # Ошибка! Счёт уменьшился
    manager.save(state, "После шага 2 (ошибка)")

    print(f"  Контрольные точки: {manager.list_checkpoints()}")

    # Backtracking к контрольной точке 2
    restored = manager.restore(2)
    print(f"  Восстановлено из точки 2: {restored}")

    # --- 4.3 Retry стратегии ---
    print("\n--- 4.3 Стратегии повтора (retry strategies) ---")

    def retry_with_strategy(func, args, strategy="exponential"):
        """Повторный вызов с различными стратегиями."""
        strategies = {
            "exponential": lambda attempt: min(2 ** attempt, 10),
            "linear": lambda attempt: attempt + 1,
            "fixed": lambda attempt: 1,
            "fibonacci": lambda attempt: [1, 1, 2, 3, 5, 8][min(attempt, 5)],
        }

        delay_fn = strategies.get(strategy, strategies["fixed"])
        attempts = []

        for attempt in range(5):
            start = time.time()
            try:
                result = func(*args)
                duration = time.time() - start
                attempts.append({"attempt": attempt + 1, "success": True,
                               "duration": duration})
                return {"success": True, "result": result, "attempts": attempts}
            except Exception as e:
                duration = time.time() - start
                delay = delay_fn(attempt)
                attempts.append({"attempt": attempt + 1, "success": False,
                               "error": str(e), "delay": delay, "duration": duration})
                if attempt < 4:
                    time.sleep(delay * 0.01)  # Ускоренная задержка для демо

        return {"success": False, "attempts": attempts}

    # Функция, котораяucceeds на 3-й попытке
    call_count = [0]
    def flaky_function(x):
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError(f"Ошибка попытки {call_count[0]}")
        return x * 10

    for strategy in ["fixed", "linear", "exponential"]:
        call_count[0] = 0
        result = retry_with_strategy(flaky_function, (5,), strategy)
        successes = sum(1 for a in result["attempts"] if a["success"])
        print(f"  Стратегия '{strategy}': "
              f"{successes}/{len(result['attempts'])} попыток, "
              f"итог={result.get('result', 'ОШИБКА')}")

    # --- 4.4 Самокоррекция через рефлексию ---
    print("\n--- 4.4 Самокоррекция через рефлексию ---")

    class SelfCorrectingAgent:
        """Агент с самокоррекцией через рефлексию."""

        def __init__(self, max_reflections=3):
            self.max_reflections = max_reflections
            self.reflection_log = []

        def solve_with_reflection(self, problem, solution_fn):
            """Решение проблемы с рефлексией."""
            best_solution = None
            best_score = -1

            for reflection in range(self.max_reflections):
                # Генерация решения
                solution = solution_fn(problem, reflection)
                score = self.evaluate(solution, problem)

                self.reflection_log.append({
                    "reflection": reflection + 1,
                    "solution": str(solution)[:50],
                    "score": score,
                })

                print(f"  Рефлексия {reflection + 1}: "
                      f"решение={str(solution)[:30]}..., оценка={score}")

                if score > best_score:
                    best_score = score
                    best_solution = solution

                # Если оценка достаточно высока — останавливаемся
                if score >= 0.9:
                    print(f"  → Достигнута высокая оценка, остановка")
                    break

            return best_solution, best_score

        def evaluate(self, solution, problem):
            """Оценка решения."""
            # Простая эвристика: чем длиннее решение и чем больше ключевых слов
            keywords = ["решение", "анализ", "результат", "вывод"]
            score = 0.0

            solution_str = str(solution)
            # Базовая оценка за длину
            score += min(len(solution_str) / 100, 0.5)

            # Оценка за ключевые слова
            for kw in keywords:
                if kw in solution_str.lower():
                    score += 0.1

            # Случайный фактор (имитация)
            score += random.random() * 0.2

            return min(score, 1.0)

    def generate_solution(problem, reflection_num):
        """Генерация решения (улучшается с каждой рефлексией)."""
        base = f"Решение проблемы '{problem}'"
        if reflection_num == 0:
            return base
        elif reflection_num == 1:
            return f"{base}: анализ данных и формирование результата"
        else:
            return f"{base}: детальный анализ, решение и подтверждение результата"

    agent = SelfCorrectingAgent(max_reflections=3)
    solution, score = agent.solve_with_reflection(
        "Оптимизация производительности",
        generate_solution
    )

    print(f"\n  Лучшее решение: '{solution}'")
    print(f"  Оценка: {score:.2f}")

    # Журнал рефлексии
    print("\n  Журнал рефлексии:")
    for entry in agent.reflection_log:
        bar = "█" * int(entry["score"] * 20)
        print(f"    Рефлексия {entry['reflection']}: "
              f"оценка={entry['score']:.2f} {bar}")

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  УРОК 158: Planning & Reasoning")
    print("  Декомпозиция задач, ReAct, chain of thought")
    print("=" * 70 + "\n")

    demo_task_decomposition()
    demo_react_pattern()
    demo_plan_and_execute()
    demo_reflection()

    print("=" * 70)
    print("  Все демонстрации завершены!")
    print("=" * 70)
