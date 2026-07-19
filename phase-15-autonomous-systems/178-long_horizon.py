"""178 — Long-Horizon Autonomy: цепочки задач, персистентность, восстановление

Темы:
  1. Task Chaining — граф зависимостей, параллельное выполнение
  2. State Persistence — контрольные точки, восстановление, сериализация
  3. Failure Recovery — повтор с backoff, цепочки fallback, эскалация
  4. Progress Tracking — вехи, оценка завершения, отчёты о прогрессе

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import heapq

random.seed(42)

# =============================================================================
# 1. Task Chaining — цепочки задач
# =============================================================================

def demo_task_chaining():
    """Демонстрация графа зависимостей задач и параллельного выполнения."""
    print("=" * 70)
    print("DEMO 1: Task Chaining — цепочки задач")
    print("=" * 70)

    # --- 1.1 Граф зависимостей (DAG) ---
    print("\n--- 1.1 Граф зависимостей (DAG) ---")

    class TaskDAG:
        """Оriented Acyclic Graph для моделирования зависимостей задач."""

        def __init__(self):
            self.tasks = {}           # task_id → task_info
            self.edges = {}           # task_id → [依赖 tasks]

        def add_task(self, task_id, name, duration=1):
            """Добавить задачу."""
            self.tasks[task_id] = {"name": name, "duration": duration, "status": "pending"}
            self.edges.setdefault(task_id, [])

        def add_dependency(self, task_id, depends_on):
            """task_id зависит от depends_on."""
            self.edges[task_id].append(depends_on)

        def get_ready_tasks(self, completed):
            """Получить задачи, готовые к выполнению."""
            ready = []
            for task_id in self.tasks:
                if task_id in completed:
                    continue
                deps = self.edges.get(task_id, [])
                if all(d in completed for d in deps):
                    ready.append(task_id)
            return ready

        def topological_sort(self):
            """Топологическая сортировка (Кahn's algorithm)."""
            in_degree = {t: 0 for t in self.tasks}
            for task_id, deps in self.edges.items():
                for d in deps:
                    in_degree[task_id] += 1

            queue = [t for t, deg in in_degree.items() if deg == 0]
            result = []

            while queue:
                queue.sort()  # детерминированный порядок
                task = queue.pop(0)
                result.append(task)
                for dependent in self.edges:
                    if task in self.edges[dependent]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

            return result

    dag = TaskDAG()

    # Пайплайн обработки данных
    tasks = [
        ("fetch", "Загрузка данных", 2),
        ("validate", "Валидация", 1),
        ("clean", "Очистка", 3),
        ("transform", "Трансформация", 2),
        ("train", "Обучение модели", 5),
        ("evaluate", "Оценка", 1),
        ("deploy", "Деплой", 2),
    ]

    for tid, name, dur in tasks:
        dag.add_task(tid, name, dur)

    # Зависимости
    dag.add_dependency("validate", "fetch")
    dag.add_dependency("clean", "validate")
    dag.add_dependency("transform", "clean")
    dag.add_dependency("train", "transform")
    dag.add_dependency("evaluate", "train")
    dag.add_dependency("deploy", "evaluate")

    order = dag.topological_sort()
    print("  Топологический порядок выполнения:")
    for i, tid in enumerate(order):
        info = dag.tasks[tid]
        print(f"    {i+1}. [{tid}] {info['name']} (длительность: {info['duration']})")

    # --- 1.2 Параллельное выполнение независимых задач ---
    print("\n--- 1.2 Параллельное выполнение независимых задач ---")

    def simulate_parallel(dag, max_workers=2):
        """Имитация параллельного выполнения."""
        completed = set()
        time_step = 0
        running = {}  # task_id → time_left

        print(f"  Макс. параллельных воркеров: {max_workers}")

        while len(completed) < len(dag.tasks):
            # Запускаем готовые задачи
            ready = dag.get_ready_tasks(completed)
            for task_id in ready:
                if len(running) < max_workers and task_id not in running:
                    dur = dag.tasks[task_id]["duration"]
                    running[task_id] = dur
                    print(f"  t={time_step}: запуск [{task_id}] {dag.tasks[task_id]['name']}")

            # Шаг времени
            finished = []
            for task_id in list(running):
                running[task_id] -= 1
                if running[task_id] <= 0:
                    finished.append(task_id)

            for task_id in finished:
                del running[task_id]
                completed.add(task_id)
                print(f"  t={time_step+1}: завершена [{task_id}] {dag.tasks[task_id]['name']}")

            time_step += 1

        return time_step

    total_time = simulate_parallel(dag, max_workers=2)
    print(f"\n  Общее время (2 воркера): {total_time}")

    # --- 1.3 Критический путь ---
    print("\n--- 1.3 Критический путь (Critical Path) ---")

    def find_critical_path(dag):
        """Найти самый длинный путь в DAG (критический путь)."""
        # Динамическое программирование: longest path
        order = dag.topological_sort()
        dist = {t: 0 for t in dag.tasks}
        prev = {t: None for t in dag.tasks}

        for task_id in order:
            dur = dag.tasks[task_id]["duration"]
            for dependent in dag.edges:
                if task_id in dag.edges[dependent]:
                    new_dist = dist[task_id] + dur
                    if new_dist > dist[dependent]:
                        dist[dependent] = new_dist
                        prev[dependent] = task_id

        # Найти конец критического пути
        end_task = max(dist, key=dist.get)
        path = []
        current = end_task
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

        return path, dist[end_task] + dag.tasks[end_task]["duration"]

    path, length = find_critical_path(dag)
    print(f"  Критический путь (длина={length}):")
    for i, tid in enumerate(path):
        name = dag.tasks[tid]["name"]
        dur = dag.tasks[tid]["duration"]
        print(f"    {i+1}. [{tid}] {name} ({dur})")

    # --- 1.4 Конвейерная обработка ---
    print("\n--- 1.4 Конвейерная обработка (Pipeline) ---")

    class Pipeline:
        """Конвейер обработки данных с этапами."""

        def __init__(self):
            self.stages = []

        def add_stage(self, name, process_func):
            """Добавить этап конвейера."""
            self.stages.append({"name": name, "func": process_func})

        def process(self, data):
            """Прогнать данные через все этапы."""
            log = []
            current = data
            for stage in self.stages:
                before = current
                current = stage["func"](current)
                log.append({
                    "stage": stage["name"],
                    "input": str(before)[:50],
                    "output": str(current)[:50],
                })
            return current, log

    pipeline = Pipeline()
    pipeline.add_stage("нормализация", lambda x: x.strip().lower())
    pipeline.add_stage("токенизация", lambda x: x.split())
    pipeline.add_stage("фильтрация", lambda x: [w for w in x if len(w) > 2])
    pipeline.add_stage("подсчёт", lambda x: {"tokens": len(x), "unique": len(set(x)), "words": x})

    text = "  Hello World from the AI Engineering Course  "
    result, log = pipeline.process(text)

    print(f"  Вход: '{text.strip()}'")
    for entry in log:
        print(f"    [{entry['stage']}] → {entry['output']}")
    print(f"  Результат: {result}")


# =============================================================================
# 2. State Persistence — сохранение состояния
# =============================================================================

def demo_state_persistence():
    """Демонстрация чекпоинтов и восстановления состояния."""
    print("\n" + "=" * 70)
    print("DEMO 2: State Persistence — персистентность")
    print("=" * 70)

    # --- 2.1 Контрольные точки (Checkpoints) ---
    print("\n--- 2.1 Контрольные точки (Checkpoints) ---")

    class CheckpointManager:
        """Управление контрольными точками состояния."""

        def __init__(self):
            self.checkpoints = []
            self.storage = {}  # имитация хранилища

        def save(self, state, label=""):
            """Сохранить контрольную точку."""
            cp_id = len(self.checkpoints)
            snapshot = {
                "id": cp_id,
                "label": label,
                "state": json.loads(json.dumps(state)),  # глубокая копия
                "hash": hashlib.md5(json.dumps(state).encode()).hexdigest()[:8],
            }
            self.checkpoints.append(snapshot)
            self.storage[cp_id] = snapshot
            print(f"  💾 Чекпоинт #{cp_id} [{label}]: hash={snapshot['hash']}")
            return cp_id

        def load(self, cp_id):
            """Загрузить состояние из контрольной точки."""
            if cp_id in self.storage:
                print(f"  📂 Загружен чекпоинт #{cp_id}: {self.storage[cp_id]['label']}")
                return json.loads(json.dumps(self.storage[cp_id]["state"]))
            return None

        def list_checkpoints(self):
            """Показать все чекпоинты."""
            for cp in self.checkpoints:
                print(f"    #{cp['id']}: {cp['label']} (hash={cp['hash']})")

    cm = CheckpointManager()

    # Имитируем обучение модели с чекпоинтами
    state = {"epoch": 0, "loss": 2.5, "accuracy": 0.1, "weights": [0.1, 0.2, 0.3]}

    for epoch in range(1, 6):
        state["epoch"] = epoch
        state["loss"] = max(0.1, 2.5 - epoch * 0.4 + random.uniform(-0.1, 0.1))
        state["accuracy"] = min(0.95, 0.1 + epoch * 0.18)

        if epoch % 2 == 0:  # сохраняем каждые 2 эпохи
            cm.save(state, label=f"epoch_{epoch}")

    print("\n  Все чекпоинты:")
    cm.list_checkpoints()

    # Восстанавливаем лучший (последний чекпоинт)
    restored = cm.load(1)
    print(f"  Восстановленное состояние: epoch={restored['epoch']}, "
          f"loss={restored['loss']:.2f}, accuracy={restored['accuracy']:.2f}")

    # --- 2.2 Сериализация сложных объектов ---
    print("\n--- 2.2 Сериализация сложных объектов ---")

    class SerializableState:
        """Сериализация и десериализация состояния агента."""

        @staticmethod
        def serialize(state_dict):
            """Сериализовать в JSON-строку."""
            # Обрабатываем нестандартные типы
            def convert(obj):
                if isinstance(obj, set):
                    return {"__type__": "set", "items": list(obj)}
                if isinstance(obj, tuple):
                    return {"__type__": "tuple", "items": list(obj)}
                return obj

            cleaned = json.loads(json.dumps(state_dict, default=convert))
            return json.dumps(cleaned, indent=2)

        @staticmethod
        def deserialize(json_str):
            """Десериализовать из JSON-строки."""
            def restore(obj):
                if isinstance(obj, dict) and "__type__" in obj:
                    if obj["__type__"] == "set":
                        return set(obj["items"])
                    if obj["__type__"] == "tuple":
                        return tuple(obj["items"])
                return obj

            return json.loads(json_str, object_hook=restore)

    # Состояние с нестандартными типами
    agent_state = {
        "memory": {"short_term": ["task1", "task2"], "long_term": ["lesson177"]},
        "goals": {"active", "completed"},  # set
        "position": (10, 20),  # tuple
        "config": {"model": "gpt-4", "temperature": 0.7},
    }

    serialized = SerializableState.serialize(agent_state)
    print("  Сериализованное состояние:")
    for line in serialized.split("\n")[:10]:
        print(f"    {line}")
    print("    ...")

    deserialized = SerializableState.deserialize(serialized)
    print(f"\n  Десериализация: типы сохранены")
    print(f"    goals тип: {type(deserialized['goals']).__name__}")
    print(f"    position тип: {type(deserialized['position']).__name__}")

    # --- 2.3 Восстановление после сбоя ---
    print("\n--- 2.3 Восстановление после сбоя (Crash Recovery) ---")

    class CrashRecovery:
        """Механизм восстановления после аварийного завершения."""

        def __init__(self):
            self.journal = []     # журнал операций
            self.committed = []   # подтверждённые операции

        def begin(self, operation):
            """Начать транзакцию."""
            entry = {
                "op": operation,
                "status": "pending",
                "id": len(self.journal),
            }
            self.journal.append(entry)
            return entry["id"]

        def commit(self, op_id):
            """Подтвердить операцию."""
            self.journal[op_id]["status"] = "committed"
            self.committed.append(self.journal[op_id])

        def recover(self):
            """Восстановить незавершённые операции."""
            recovered = []
            for entry in self.journal:
                if entry["status"] == "pending":
                    recovered.append(entry)
            return recovered

    recovery = CrashRecovery()

    # Имитируем серию операций с "сбоем"
    ops = [
        "запись в базу",
        "отправка email",
        "обновление кэша",
        "запись в базу",
        "отправка email",
    ]

    for op in ops:
        op_id = recovery.begin(op)
        if random.random() > 0.3:  # 70% успешных
            recovery.commit(op_id)
            print(f"  ✅ [{op_id}] {op}: подтверждено")
        else:
            print(f"  ❌ [{op_id}] {op}: СБОЙ (осталось pending)")

    pending = recovery.recover()
    print(f"\n  Незавершённые операции (требуют повтора): {len(pending)}")
    for entry in pending:
        print(f"    [{entry['id']}] {entry['op']}")

    # --- 2.4 Версионирование состояния ---
    print("\n--- 2.4 Версионирование состояния ---")

    class StateVersioning:
        """Отслеживание версий состояния."""

        def __init__(self):
            self.versions = []
            self.current = None

        def update(self, state, description=""):
            """Создать новую версию."""
            version = len(self.versions)
            entry = {
                "version": version,
                "state": json.loads(json.dumps(state)),
                "description": description,
                "hash": hashlib.sha256(
                    json.dumps(state, sort_keys=True).encode()
                ).hexdigest()[:12],
            }
            self.versions.append(entry)
            self.current = entry
            return version

        def diff(self, v1, v2):
            """Сравнить две версии."""
            s1 = self.versions[v1]["state"]
            s2 = self.versions[v2]["state"]
            changes = []

            all_keys = set(list(s1.keys()) + list(s2.keys()))
            for key in all_keys:
                if key not in s1:
                    changes.append(f"+ {key}: {s2[key]}")
                elif key not in s2:
                    changes.append(f"- {key}: {s1[key]}")
                elif s1[key] != s2[key]:
                    changes.append(f"~ {key}: {s1[key]} → {s2[key]}")

            return changes

    sv = StateVersioning()

    states = [
        {"progress": 0, "phase": "init", "score": 0},
        {"progress": 25, "phase": "data_load", "score": 0.3},
        {"progress": 50, "phase": "training", "score": 0.6},
        {"progress": 75, "phase": "eval", "score": 0.85},
    ]

    for state in states:
        ver = sv.update(state, description=f"progress={state['progress']}%")
        print(f"  Версия {ver}: {state} (hash={sv.versions[ver]['hash']})")

    print("\n  Diff между v0 и v3:")
    changes = sv.diff(0, 3)
    for c in changes:
        print(f"    {c}")


# =============================================================================
# 3. Failure Recovery — восстановление после ошибок
# =============================================================================

def demo_failure_recovery():
    """Демонстрация стратегий восстановления после ошибок."""
    print("\n" + "=" * 70)
    print("DEMO 3: Failure Recovery — восстановление после ошибок")
    print("=" * 70)

    # --- 3.1 Retry с экспоненциальным backoff ---
    print("\n--- 3.1 Retry с экспоненциальным backoff ---")

    def retry_with_backoff(func, max_retries=5, base_delay=0.01):
        """Повтор с экспоненциальной задержкой."""
        # Формула: delay = base * 2^attempt + random_jitter
        attempts = 0
        results = []

        for attempt in range(max_retries):
            attempts += 1
            try:
                result = func()
                results.append(("SUCCESS", attempt, 0))
                return result, attempts, results
            except Exception as e:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.005)
                results.append(("RETRY", attempt, delay))
                time.sleep(delay)  # в реальности — секунды/минуты

        raise Exception(f"Не удалось после {max_retries} попыток")

    def flaky_api_call():
        """Имитация ненадёжного API (30% успех)."""
        if random.random() < 0.3:
            return "OK: данные получены"
        raise ConnectionError("API недоступен")

    try:
        result, total_attempts, log = retry_with_backoff(flaky_api_call, max_retries=5)
        print(f"  Результат: {result}")
        print(f"  Попыток: {total_attempts}")
        for entry in log:
            status, attempt, delay = entry
            if status == "RETRY":
                print(f"    Попытка {attempt}: RETRY (задержка {delay:.3f}с)")
            else:
                print(f"    Попытка {attempt}: SUCCESS")
    except Exception as e:
        print(f"  Провал: {e}")

    # --- 3.2 Цепочки fallback ---
    print("\n--- 3.2 Цепочки fallback ---")

    class FallbackChain:
        """Цепочка альтернативных стратегий."""

        def __init__(self):
            self.strategies = []

        def add_strategy(self, name, func):
            """Добавить стратегию в цепочку."""
            self.strategies.append((name, func))

        def execute(self, *args, **kwargs):
            """Выполнить, переключаясь при ошибках."""
            errors = []
            for name, func in self.strategies:
                try:
                    result = func(*args, **kwargs)
                    return result, name, errors
                except Exception as e:
                    errors.append((name, str(e)))
                    continue
            raise Exception("Все стратегии исчерпаны")

    chain = FallbackChain()

    # Три стратегии получения данных
    chain.add_strategy("API v1", lambda x: (_ for _ in ()).throw(ConnectionError("timeout")))
    chain.add_strategy("API v2", lambda x: (_ for _ in ()).throw(ConnectionError("rate limit")))
    chain.add_strategy("local_cache", lambda x: {"data": [1, 2, 3], "source": "cache"})

    result, strategy, errors = chain.execute("test")
    print(f"  Успех через: {strategy}")
    print(f"  Данные: {result}")
    print(f"  Проваленные стратегии:")
    for name, err in errors:
        print(f"    [{name}]: {err}")

    # --- 3.3 Circuit Breaker ---
    print("\n--- 3.3 Circuit Breaker (автоматический откат) ---")

    class CircuitBreaker:
        """Паттерн Circuit Breaker для защиты от каскадных сбоев."""

        CLOSED = "CLOSED"     # нормальная работа
        OPEN = "OPEN"         # блокировка (сервис недоступен)
        HALF_OPEN = "HALF_OPEN"  # тестовое восстановление

        def __init__(self, failure_threshold=3, recovery_timeout=5):
            self.state = self.CLOSED
            self.failures = 0
            self.threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.last_failure_time = None
            self.calls = []

        def call(self, func):
            """Выполнить вызов через circuit breaker."""
            if self.state == self.OPEN:
                # Проверяем, прошло ли время восстановления
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    print(f"    Circuit: OPEN → HALF_OPEN (тест)")
                else:
                    self.calls.append(("BLOCKED", 0))
                    raise Exception("Circuit OPEN: вызов заблокирован")

            try:
                result = func()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise

        def _on_success(self):
            """Успешный вызов."""
            self.failures = 0
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                print(f"    Circuit: HALF_OPEN → CLOSED")
            self.calls.append(("OK", 1))

        def _on_failure(self):
            """Неудачный вызов."""
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                self.state = self.OPEN
                print(f"    Circuit: → OPEN (порог {self.threshold} reached)")
            self.calls.append(("FAIL", 0))

    cb = CircuitBreaker(failure_threshold=3)

    def unreliable_service():
        """Ненадёжный сервис."""
        if random.random() < 0.5:
            raise ConnectionError("service down")
        return "response"

    print("  Вызовы через Circuit Breaker:")
    for i in range(8):
        try:
            result = cb.call(unreliable_service)
            print(f"    Вызов {i+1}: OK")
        except Exception as e:
            print(f"    Вызов {i+1}: {e}")

    print(f"  Финальное состояние: {cb.state}")

    # --- 3.4 Эскалация на ручное управление ---
    print("\n--- 3.4 Эскалация на ручное управление ---")

    class EscalationPolicy:
        """Политика эскалации при невозможности автоматического восстановления."""

        def __init__(self):
            self.levels = []    # (уровень, действие)
            self.current_level = 0

        def add_level(self, level, action, description):
            """Добавить уровень эскалации."""
            self.levels.append({
                "level": level,
                "action": action,
                "description": description,
            })

        def escalate(self, error_context):
            """Перейти на следующий уровень эскалации."""
            if self.current_level < len(self.levels) - 1:
                self.current_level += 1
            level = self.levels[self.current_level]
            return level

    policy = EscalationPolicy()
    policy.add_level(1, "retry", "Автоматический повтор")
    policy.add_level(2, "fallback", "Переключение на резервный сервис")
    policy.add_level(3, "notify_ops", "Уведомление команды")
    policy.add_level(4, "manual", "Ручное вмешательство оператора")

    errors = [
        "Connection timeout",
        "Rate limit exceeded",
        "Data corruption detected",
        "Service unavailable for 10 min",
    ]

    for err in errors:
        level_info = policy.escalate(err)
        print(f"  Ошибка: {err}")
        print(f"  → Уровень {level_info['level']}: {level_info['action']} "
              f"({level_info['description']})")


# =============================================================================
# 4. Progress Tracking — отслеживание прогресса
# =============================================================================

def demo_progress_tracking():
    """Демонстрация отслеживания прогресса долгосрочных задач."""
    print("\n" + "=" * 70)
    print("DEMO 4: Progress Tracking — отслеживание прогресса")
    print("=" * 70)

    # --- 4.1 Вехи (Milestones) ---
    print("\n--- 4.1 Вехи (Milestones) ---")

    class MilestoneTracker:
        """Отслеживание ключевых вех проекта."""

        def __init__(self, project_name):
            self.project = project_name
            self.milestones = []

        def add_milestone(self, name, weight=1.0):
            """Добавить веху с весом (вклад в общий прогресс)."""
            self.milestones.append({
                "name": name,
                "weight": weight,
                "completed": False,
            })

        def complete(self, name):
            """Отметить веху как завершённую."""
            for m in self.milestones:
                if m["name"] == name:
                    m["completed"] = True
                    return True
            return False

        def progress_pct(self):
            """Вычислить общий прогресс в процентах."""
            if not self.milestones:
                return 0
            total_weight = sum(m["weight"] for m in self.milestones)
            done_weight = sum(m["weight"] for m in self.milestones if m["completed"])
            return done_weight / total_weight * 100

        def report(self):
            """Вывести отчёт."""
            pct = self.progress_pct()
            filled = "█" * int(pct / 5)
            empty = "░" * (20 - int(pct / 5))
            print(f"\n  Проект: {self.project}")
            print(f"  Прогресс: [{filled}{empty}] {pct:.1f}%")
            for m in self.milestones:
                icon = "✅" if m["completed"] else "⬜"
                print(f"    {icon} {m['name']} (вес: {m['weight']})")

    tracker = MilestoneTracker("AI Pipeline v2")
    tracker.add_milestone("Данные собраны", weight=2.0)
    tracker.add_milestone("Предобработка", weight=1.5)
    tracker.add_milestone("Модель обучена", weight=3.0)
    tracker.add_milestone("Метрики достигнуты", weight=2.0)
    tracker.add_milestone("Деплой", weight=1.5)

    # Последовательно завершаем
    completed = ["Данные собраны", "Предобработка", "Модель обучена"]
    for name in completed:
        tracker.complete(name)
        print(f"  ✅ Завершена: {name}")

    tracker.report()

    # --- 4.2 Оценка времени завершения (ETA) ---
    print("\n--- 4.2 Оценка времени завершения (ETA) ---")

    class ETAEstimator:
        """Оценка времени до завершения на основе прогресса."""

        def __init__(self, total_work):
            self.total = total_work
            self.completed = 0
            self.start_time = time.time()
            self.history = []  # (время, выполнено)

        def update(self, work_done):
            """Обновить прогресс."""
            self.completed += work_done
            elapsed = time.time() - self.start_time
            self.history.append((elapsed, self.completed))

        def eta_seconds(self):
            """Оценка оставшегося времени (секунды)."""
            if not self.history or self.completed == 0:
                return float('inf')

            # Средняя скорость
            last_time, last_done = self.history[-1]
            if last_time == 0:
                return float('inf')
            rate = last_done / last_time  # работа в секунду
            remaining = self.total - self.completed
            return remaining / rate if rate > 0 else float('inf')

        def progress_str(self):
            """Строковое представление прогресса."""
            pct = self.completed / self.total * 100 if self.total > 0 else 0
            eta = self.eta_seconds()
            if eta < 60:
                eta_str = f"{eta:.0f}с"
            elif eta < 3600:
                eta_str = f"{eta/60:.1f}мин"
            else:
                eta_str = f"{eta/3600:.1f}ч"
            return f"{pct:.1f}% (ETA: {eta_str})"

    estimator = ETAEstimator(total_work=100)
    work_steps = [10, 15, 20, 25, 30]

    for step in work_steps:
        estimator.update(step)
        print(f"  Выполнено +{step}: {estimator.progress_str()}")

    # --- 4.3 Отчёты о прогрессе ---
    print("\n--- 4.3 Генерация отчётов о прогрессе ---")

    class ProgressReporter:
        """Генерация структурированных отчётов."""

        def __init__(self, task_name):
            self.task = task_name
            self.events = []

        def log_event(self, event_type, message, data=None):
            """Записать событие."""
            self.events.append({
                "type": event_type,
                "message": message,
                "data": data,
                "seq": len(self.events),
            })

        def summary(self):
            """Краткая сводка."""
            counts = collections.Counter(e["type"] for e in self.events)
            total = len(self.events)
            return {
                "task": self.task,
                "total_events": total,
                "breakdown": dict(counts),
                "last_event": self.events[-1] if self.events else None,
            }

        def format_report(self):
            """Форматированный отчёт."""
            summary = self.summary()
            lines = [
                f"  === Отчёт: {summary['task']} ===",
                f"  Всего событий: {summary['total_events']}",
                f"  По типам: {summary['breakdown']}",
            ]
            if summary["last_event"]:
                last = summary["last_event"]
                lines.append(f"  Последнее: [{last['type']}] {last['message']}")
            return "\n".join(lines)

    reporter = ProgressReporter("Обучение модели v3")
    reporter.log_event("start", "Начало обучения")
    reporter.log_event("progress", "Эпоха 1/10", {"loss": 2.1})
    reporter.log_event("progress", "Эпоха 2/10", {"loss": 1.8})
    reporter.log_event("warning", "Память: 85%", {"memory_mb": 1700})
    reporter.log_event("progress", "Эпоха 3/10", {"loss": 1.5})
    reporter.log_event("checkpoint", "Сохранён чекпоинт", {"path": "model_ep3.pt"})
    reporter.log_event("progress", "Эпоха 4/10", {"loss": 1.2})
    reporter.log_event("complete", "Обучение завершено")

    print(reporter.format_report())

    # --- 4.4 Комбинированное отслеживание ---
    print("\n--- 4.4 Комбинированное отслеживание всех компонентов ---")

    class AutonomousTaskTracker:
        """Комплексный трекер для автономных долгосрочных задач."""

        def __init__(self, name):
            self.name = name
            self.milestones = MilestoneTracker(name)
            self.reporter = ProgressReporter(name)
            self.start_time = time.time()
            self.status = "running"

        def add_milestone(self, name, weight=1.0):
            self.milestones.add_milestone(name, weight)

        def complete_milestone(self, name):
            self.milestones.complete(name)
            self.reporter.log_event("milestone", f"Веха: {name}")

        def fail(self, reason):
            self.status = "failed"
            self.reporter.log_event("error", f"Ошибка: {reason}")

        def finish(self):
            self.status = "completed"
            self.reporter.log_event("complete", "Задача завершена")

        def get_status(self):
            """Полный статус задачи."""
            elapsed = time.time() - self.start_time
            return {
                "name": self.name,
                "status": self.status,
                "progress": self.milestones.progress_pct(),
                "elapsed": f"{elapsed:.1f}с",
                "events": len(self.reporter.events),
            }

    tracker = AutonomousTaskTracker("ML Pipeline Full Run")
    tracker.add_milestone("Данные", 1.0)
    tracker.add_milestone("Обучение", 2.0)
    tracker.add_milestone("Оценка", 1.0)

    tracker.complete_milestone("Данные")
    tracker.complete_milestone("Обучение")

    status = tracker.get_status()
    print(f"  Задача: {status['name']}")
    print(f"  Статус: {status['status']}")
    print(f"  Прогресс: {status['progress']:.1f}%")
    print(f"  Время: {status['elapsed']}")
    print(f"  Событий: {status['events']}")

    tracker.finish()
    final = tracker.get_status()
    print(f"\n  Финальный статус: {final['status']}, прогресс: {final['progress']:.1f}%")


# =============================================================================
# Запуск всех демонстраций
# =============================================================================

if __name__ == "__main__":
    print("УРОК 178: Long-Horizon Autonomy")
    print("Цепочки задач, персистентность, восстановление\n")

    demo_task_chaining()
    demo_state_persistence()
    demo_failure_recovery()
    demo_progress_tracking()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены.")
    print("=" * 70)
