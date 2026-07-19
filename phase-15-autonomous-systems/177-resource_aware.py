"""177 — Resource-Aware Agents: бюджет вычислений, управление временем, приоритеты

Темы:
  1. Compute Budgeting — лимиты токенов, времени, стоимости
  2. Priority Scheduling — срочное vs важное, дедлайны, распределение ресурсов
  3. Lazy Evaluation — отложенное вычисление, кэширование, пропуск лишнего
  4. Resource Monitoring — отслеживание использования, квоты, динамическая корректировка

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
# 1. Compute Budgeting — бюджет вычислений
# =============================================================================

def demo_compute_budgeting():
    """Демонстрация управления вычислительным бюджетом агента."""
    print("=" * 70)
    print("DEMO 1: Compute Budgeting — бюджет вычислений")
    print("=" * 70)

    # --- 1.1 Токенные бюджеты ---
    # Агент работает с лимитом токенов на запрос
    print("\n--- 1.1 Токенные бюджеты ---")

    class TokenBudget:
        """Управление лимитом токенов для LLM-запросов."""

        def __init__(self, max_tokens):
            self.max_tokens = max_tokens          # общий лимит
            self.used_tokens = 0                   # использовано
            self.history = []                      # история запросов

        def estimate_tokens(self, text):
            """Оценка числа токенов (≈ 4 символа на токен для английского)."""
            # Формула: токены ≈ длина_строки / 4
            return max(1, len(text) // 4)

        def can_afford(self, text):
            """Проверка, хватит ли бюджета на этот запрос."""
            needed = self.estimate_tokens(text)
            return (self.used_tokens + needed) <= self.max_tokens

        def spend(self, label, text):
            """Потратить токены на запрос и записать в историю."""
            tokens = self.estimate_tokens(text)
            self.used_tokens += tokens
            self.history.append({"label": label, "tokens": tokens})
            return tokens

    budget = TokenBudget(max_tokens=100)

    # Имитируем серию запросов к LLM
    requests = [
        ("summarize", "Please summarize the main findings of the research paper"),
        ("classify", "Classify this text as positive or negative sentiment"),
        ("generate", "Generate a detailed technical report about neural networks"),
        ("embed", "Create embedding vector for this short phrase"),
    ]

    total_spent = 0
    for label, text in requests:
        if budget.can_afford(text):
            tokens = budget.spend(label, text)
            total_spent += tokens
            remaining = budget.max_tokens - budget.used_tokens
            print(f"  Запрос '{label}': потрачено {tokens} токенов, "
                  f"осталось {remaining}/{budget.max_tokens}")
        else:
            needed = budget.estimate_tokens(text)
            print(f"  Запрос '{label}': ОТКАЗАН — нужно {needed}, "
                  f"осталось {budget.max_tokens - budget.used_tokens}")

    print(f"\n  Итого: потрачено {total_spent} из {budget.max_tokens} токенов")

    # --- 1.2 Лимиты по времени ---
    # Агент должен укладываться в ограничение по времени
    print("\n--- 1.2 Временные лимиты ---")

    class TimeLimit:
        """Ограничение времени на выполнение задачи."""

        def __init__(self, deadline_seconds):
            self.deadline = deadline_seconds
            self.start_time = time.time()

        def remaining(self):
            """Сколько времени осталось."""
            elapsed = time.time() - self.start_time
            return max(0, self.deadline - elapsed)

        def is_expired(self):
            """Проверка: истекло ли время."""
            return self.remaining() <= 0

        def budget_per_task(self, num_tasks):
            """Равномерное распределение времени между задачами."""
            # Формула: время_на_задачу = остаток / число_задач
            return self.remaining() / max(1, num_tasks)

    timer = TimeLimit(deadline_seconds=5.0)
    tasks = ["анализ данных", "генерация отчёта", "валидация", "упаковка"]

    print(f"  Общий дедлайн: {timer.deadline} сек")
    per_task = timer.budget_per_task(len(tasks))
    print(f"  Задач: {len(tasks)}, время на каждую: {per_task:.2f} сек")

    for task in tasks:
        # Имитируем выполнение (с задержкой 0.1 сек)
        time.sleep(0.1)
        remaining = timer.remaining()
        print(f"  Выполнена '{task}': осталось {remaining:.2f} сек")

    # --- 1.3 Бюджет стоимости ---
    # Лимит на сумму, которую агент может потратить
    print("\n--- 1.3 Бюджет стоимости ---")

    class CostCap:
        """Управление денежным бюджетом агента."""

        # Цены за 1000 токенов (в условных единицах)
        PRICE_TABLE = {
            "input": 0.005,    # стоимость входных токенов
            "output": 0.015,   # стоимость выходных токенов
        }

        def __init__(self, budget):
            self.budget = budget
            self.spent = 0.0
            self.log = []

        def estimate_cost(self, input_tokens, output_tokens):
            """Оценка стоимости запроса."""
            # Формула: стоимость = (вход * цену_входа + выход * цену_выхода) / 1000
            cost = (input_tokens * self.PRICE_TABLE["input"] +
                    output_tokens * self.PRICE_TABLE["output"]) / 1000
            return cost

        def try_spend(self, label, input_tokens, output_tokens):
            """Попытка потратить — если хватает бюджета."""
            cost = self.estimate_cost(input_tokens, output_tokens)
            if self.spent + cost <= self.budget:
                self.spent += cost
                self.log.append({"label": label, "cost": cost})
                return True, cost
            return False, cost

    cap = CostCap(budget=0.001)  # бюджет 0.001 единицы

    api_calls = [
        ("chat", 500, 200),
        ("summarize", 1000, 500),
        ("embed", 100, 50),
        ("generate", 2000, 1500),
    ]

    for label, inp, out in api_calls:
        ok, cost = cap.try_spend(label, inp, out)
        status = "OK" if ok else "ОТКАЗАН"
        print(f"  {label}: стоимость ${cost:.5f} — {status} "
              f"(бюджет: ${cap.spent:.5f}/${cap.budget:.5f})")

    # --- 1.4 Комбинированный контроль ---
    # Все три лимита вместе
    print("\n--- 1.4 Комбинированный контроль ресурсов ---")

    class ResourceController:
        """Единый контроллер всех ресурсов агента."""

        def __init__(self, max_tokens, deadline, budget):
            self.tokens = TokenBudget(max_tokens)
            self.time = TimeLimit(deadline)
            self.cost = CostCap(budget)

        def can_execute(self, task_desc, input_tok, output_tok, needed_tok):
            """Проверка всех ограничений перед выполнением."""
            reasons = []
            if not self.tokens.can_afford(task_desc):
                reasons.append("мало токенов")
            if self.time.is_expired():
                reasons.append("время вышло")
            est_cost = self.cost.estimate_cost(input_tok, output_tok)
            if self.cost.spent + est_cost > self.cost.budget:
                reasons.append("превышен бюджет")
            return len(reasons) == 0, reasons

    ctrl = ResourceController(max_tokens=200, deadline=2.0, budget=0.0005)

    test_tasks = [
        ("quick", 100, 50, 30),
        ("medium", 300, 200, 80),
        ("heavy", 500, 400, 130),
    ]

    for name, inp, out, tok in test_tasks:
        ok, reasons = ctrl.can_execute(name, inp, out, tok)
        if ok:
            ctrl.tokens.spend(name, "x" * (tok * 4))
            ctrl.cost.try_spend(name, inp, out)
            print(f"  Задача '{name}': ВЫПОЛНЕНА")
        else:
            print(f"  Задача '{name}': БЛОКИРОВАНА — {', '.join(reasons)}")


# =============================================================================
# 2. Priority Scheduling — планирование по приоритетам
# =============================================================================

def demo_priority_scheduling():
    """Демонстрация планирования задач по приоритетам."""
    print("\n" + "=" * 70)
    print("DEMO 2: Priority Scheduling — планирование по приоритетам")
    print("=" * 70)

    # --- 2.1 Матрица срочное/важное (Эйзенхауэр) ---
    print("\n--- 2.1 Матрица Эйзенхауэра (urgent/important) ---")

    class EisenhowerMatrix:
        """Классификация задач по срочности и важности."""

        def __init__(self):
            self.quadrants = {
                "DO": [],           # срочное + важное
                "SCHEDULE": [],     # важное + несрочное
                "DELEGATE": [],     # срочное + неважное
                "ELIMINATE": [],    # несрочное + неважное
            }

        def classify(self, task_name, urgency, importance):
            """Отнести задачу в один из квадрантов."""
            # Формула: порог = 0.5 для каждого измерения
            if urgency > 0.5 and importance > 0.5:
                quad = "DO"
            elif importance > 0.5 and urgency <= 0.5:
                quad = "SCHEDULE"
            elif urgency > 0.5 and importance <= 0.5:
                quad = "DELEGATE"
            else:
                quad = "ELIMINATE"
            self.quadrants[quad].append((task_name, urgency, importance))
            return quad

        def summary(self):
            """Вывод сводки по квадрантам."""
            labels = {
                "DO": "ДЕЛАТЬ СРАЗУ",
                "SCHEDULE": "ЗАПЛАНИРОВАТЬ",
                "DELEGATE": "ДЕЛЕГИРОВАТЬ",
                "ELIMINATE": "УДАЛИТЬ",
            }
            for q, label in labels.items():
                tasks = self.quadrants[q]
                names = [t[0] for t in tasks]
                print(f"  [{label}]: {', '.join(names) if names else '(пусто)'}")

    matrix = EisenhowerMatrix()
    tasks = [
        ("исправить баг в проде", 0.9, 0.9),
        ("обновить документацию", 0.3, 0.7),
        ("ответить на спам-письмо", 0.8, 0.2),
        ("почистить рабочий стол", 0.2, 0.1),
        ("deploy hotfix", 0.95, 0.85),
        ("рефакторинг legacy", 0.4, 0.8),
    ]

    for name, urg, imp in tasks:
        q = matrix.classify(name, urg, imp)
        print(f"  Задача '{name}': urgency={urg}, importance={imp} → {q}")

    print()
    matrix.summary()

    # --- 2.2 Приоритетная очередь с дедлайнами ---
    print("\n--- 2.2 Приоритетная очередь с дедлайнами ---")

    class DeadlineScheduler:
        """Планировщик с учётом дедлайнов и приоритетов."""

        def __init__(self):
            self.queue = []      # куча (min-heap)
            self.counter = 0     # счётчик для разрыва связей

        def add_task(self, name, priority, deadline):
            """Добавить задачу с приоритетом и дедлайном."""
            # Приоритет в очереди = -(приоритет) + дедлайн/1000
            # Чем выше приоритет и ближе дедлайн — тем раньше выполняем
            effective = -priority + deadline / 1000.0
            entry = (effective, self.counter, name, priority, deadline)
            heapq.heappush(self.queue, entry)
            self.counter += 1

        def pop_next(self):
            """Извлечь задачу с наивысшим приоритетом."""
            if not self.queue:
                return None
            _, _, name, priority, deadline = heapq.heappop(self.queue)
            return name, priority, deadline

        def peek_all(self):
            """Посмотреть очередь (без извлечения)."""
            return [(e[2], e[3], e[4]) for e in sorted(self.queue)]

    scheduler = DeadlineScheduler()
    jobs = [
        ("отчёт для CEO", 10, 1),
        ("тестирование API", 5, 3),
        ("обновление зависимостей", 3, 10),
        ("hotfix критический баг", 10, 0),
        ("код-ревью", 6, 5),
    ]

    for name, prio, dl in jobs:
        scheduler.add_task(name, prio, dl)
        print(f"  Добавлена: '{name}' (приоритет={prio}, дедлайн={dl}ч)")

    print("\n  Порядок выполнения:")
    for i in range(len(scheduler.queue)):
        result = scheduler.pop_next()
        if result:
            name, prio, dl = result
            print(f"    {i+1}. '{name}' — приоритет {prio}, дедлайн {dl}ч")

    # --- 2.3 Взвешенное распределение ресурсов ---
    print("\n--- 2.3 Взвешенное распределение ресурсов ---")

    def weighted_allocation(resources, tasks_with_weights):
        """Распределить ресурсы пропорционально весам задач."""
        # Формула: доля_i = вес_i / сумма_весов * ресурс
        total_weight = sum(w for _, w in tasks_with_weights)
        allocations = []
        for name, weight in tasks_with_weights:
            share = (weight / total_weight) * resources
            allocations.append((name, round(share, 2)))
        return allocations

    total_resources = 1000  # условных единицcompute
    task_weights = [
        ("训练 модели", 40),
        ("инференс", 30),
        ("preprocessing", 20),
        ("мониторинг", 10),
    ]

    allocs = weighted_allocation(total_resources, task_weights)
    for name, share in allocs:
        pct = share / total_resources * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:20s}: {share:6.1f} ({pct:5.1f}%) {bar}")

    # --- 2.4 Динамическое перераспределение ---
    print("\n--- 2.4 Динамическое перераспределение при изменении приоритетов ---")

    class DynamicAllocator:
        """Динамическое перераспределение ресурсов при изменении приоритетов."""

        def __init__(self, total):
            self.total = total
            self.allocations = {}

        def set_priorities(self, task_weights):
            """Пересчитать распределение по новым весам."""
            total_w = sum(w for _, w in task_weights)
            old = dict(self.allocations)
            self.allocations = {}
            for name, weight in task_weights:
                self.allocations[name] = round(weight / total_w * self.total, 1)

            # Показать изменения
            for name in self.allocations:
                new_val = self.allocations[name]
                old_val = old.get(name, 0)
                delta = new_val - old_val
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                print(f"    {name:20s}: {old_val:6.1f} → {new_val:6.1f} ({arrow}{abs(delta):.1f})")

    allocator = DynamicAllocator(total=1000)

    print("  Начальное распределение:")
    allocator.set_priorities([
        ("training", 50),
        ("inference", 30),
        ("monitoring", 20),
    ])

    print("\n  После инцидента (мониторинг becomes critical):")
    allocator.set_priorities([
        ("training", 20),
        ("inference", 20),
        ("monitoring", 60),   # мониторинг получил приоритет
    ])


# =============================================================================
# 3. Lazy Evaluation — отложенное вычисление
# =============================================================================

def demo_lazy_evaluation():
    """Демонстрация отложенного вычисления и кэширования."""
    print("\n" + "=" * 70)
    print("DEMO 3: Lazy Evaluation — отложенное вычисление")
    print("=" * 70)

    # --- 3.1 Ленивые выражения ---
    print("\n--- 3.1 Ленивые выражения (Deferred Computation) ---")

    class LazyValue:
        """Вычисление значения только при обращении."""

        def __init__(self, func, *args, **kwargs):
            self._func = func
            self._args = args
            self._kwargs = kwargs
            self._computed = False
            self._value = None

        def get(self):
            """Вычислить и кэшировать значение."""
            if not self._computed:
                print(f"    [вычисляю] вызов {self._func.__name__}...")
                self._value = self._func(*self._args, **self._kwargs)
                self._computed = True
            return self._value

        @property
        def is_ready(self):
            """Уже вычислено?"""
            return self._computed

    def expensive_computation(n):
        """Имитация дорогого вычисления (факториал)."""
        time.sleep(0.05)  # имитация задержки
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    # Создаём ленивые значения — вычисления ещё НЕ происходит
    lazy_vals = [
        LazyValue(expensive_computation, 5),
        LazyValue(expensive_computation, 10),
        LazyValue(expensive_computation, 15),
    ]

    print("  Созданы 3 ленивых значения (вычисления НЕ запущены):")
    for i, lv in enumerate(lazy_vals):
        print(f"    lazy[{i}]: готов={lv.is_ready}")

    # Обращаемся только к первому — остальные остаются ленивыми
    print(f"\n  Доступ к lazy[0]: {lazy_vals[0].get()}")
    print("  Состояние после обращения:")
    for i, lv in enumerate(lazy_vals):
        print(f"    lazy[{i}]: готов={lv.is_ready}")

    # --- 3.2 Кэширование результатов ---
    print("\n--- 3.2 Кэширование результатов (Memoization) ---")

    class MemoCache:
        """Кэш с автоматическим мемоизацией."""

        def __init__(self):
            self.cache = {}
            self.hits = 0
            self.misses = 0

        def get_or_compute(self, key, compute_func):
            """Получить из кэша или вычислить."""
            if key in self.cache:
                self.hits += 1
                return self.cache[key], "HIT"
            self.misses += 1
            value = compute_func(key)
            self.cache[key] = value
            return value, "MISS"

        def stats(self):
            """Статистика кэша."""
            total = self.hits + self.misses
            rate = self.hits / total * 100 if total > 0 else 0
            return {"hits": self.hits, "misses": self.misses, "hit_rate": rate}

    cache = MemoCache()

    def fibonacci(n):
        """Вычисление числа Фибоначчи (рекурсивно)."""
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    # Запросы с повторами
    queries = [10, 5, 10, 20, 5, 10, 20, 30]
    for q in queries:
        result, status = cache.get_or_compute(q, fibonacci)
        print(f"  fib({q}) = {result} [{status}]")

    stats = cache.stats()
    print(f"\n  Статистика: hits={stats['hits']}, misses={stats['misses']}, "
          f"hit_rate={stats['hit_rate']:.1f}%")

    # --- 3.3 Пропуск ненужных вычислений ---
    print("\n--- 3.3 Пропуск ненужных вычислений (Short-Circuit) ---")

    class FilterChain:
        """Цепочка фильтров с ранним выходом."""

        def __init__(self):
            self.filters = []
            self.skip_count = 0

        def add_filter(self, name, check_func):
            """Добавить фильтр."""
            self.filters.append((name, check_func))

        def evaluate(self, item):
            """Проверить элемент — выйти при первом отказе."""
            for name, check in self.filters:
                if not check(item):
                    self.skip_count += 1
                    return False, name  # ранний выход — дальше не проверяем
            return True, "all_passed"

    chain = FilterChain()
    chain.add_filter("длина > 5", lambda x: len(str(x)) > 5)
    chain.add_filter("чётное", lambda x: x % 2 == 0)
    chain.add_filter("делится на 3", lambda x: x % 3 == 0)
    chain.add_filter("< 100", lambda x: x < 100)

    test_numbers = [12, 6, 24, 100, 36, 48, 7, 999999]
    print("  Фильтры: длина>5, чётное, делится на 3, <100")
    for num in test_numbers:
        passed, reason = chain.evaluate(num)
        status = "ПРОШЁЛ" if passed else f"ОТСЕЧЁН на '{reason}'"
        print(f"    {num}: {status}")

    print(f"  Пропущено проверок (ранний выход): {chain.skip_count}")

    # --- 3.4 Ленивые последовательности ---
    print("\n--- 3.4 Ленивые последовательности (Lazy Sequences) ---")

    def lazy_range(start, stop, step=1):
        """Ленивый генератор диапазона — элементы создаются по одному."""
        current = start
        generated = 0
        while current < stop:
            yield current
            current += step
            generated += 1

    def lazy_map(func, sequence):
        """Ленивое отображение — вычисляет только при итерации."""
        count = 0
        for item in sequence:
            count += 1
            yield func(item)

    # Ленивая цепочка: range → map → take first 5
    print("  Ленивая цепочка: range(1, 1000000) → map(x^2) → take(5)")

    lazy_nums = lazy_range(1, 1_000_000)       # не создаёт миллионы чисел
    lazy_squares = lazy_map(lambda x: x ** 2, lazy_nums)  # не вычисляет квадраты

    # Берём только первые 5 — остальные никогда не вычислятся
    results = []
    for val in lazy_squares:
        results.append(val)
        if len(results) >= 5:
            break

    print(f"  Результат: {results}")
    print(f"  Вычислено элементов из 999999: {len(results)} (остальные пропущены)")


# =============================================================================
# 4. Resource Monitoring — мониторинг ресурсов
# =============================================================================

def demo_resource_monitoring():
    """Демонстрация отслеживания и управления ресурсами."""
    print("\n" + "=" * 70)
    print("DEMO 4: Resource Monitoring — мониторинг ресурсов")
    print("=" * 70)

    # --- 4.1 Трекер использования ---
    print("\n--- 4.1 Трекер использования ресурсов ---")

    class ResourceTracker:
        """Отслеживание потребления ресурсов во времени."""

        def __init__(self):
            self.timeline = []       # временная шкала
            self.totals = {}         # суммарное потребление

        def record(self, resource, amount):
            """Записать потребление ресурса."""
            self.timeline.append({
                "time": len(self.timeline),
                "resource": resource,
                "amount": amount,
            })
            self.totals[resource] = self.totals.get(resource, 0) + amount

        def get_summary(self):
            """Сводка по всем ресурсам."""
            return dict(self.totals)

        def peak_usage(self, resource):
            """Пиковое потребление ресурса."""
            entries = [e["amount"] for e in self.timeline if e["resource"] == resource]
            return max(entries) if entries else 0

    tracker = ResourceTracker()

    # Имитируем потребление ресурсов
    usage_data = [
        ("tokens", 150), ("tokens", 200), ("memory_mb", 256),
        ("tokens", 180), ("memory_mb", 312), ("api_calls", 1),
        ("tokens", 90), ("memory_mb", 128), ("api_calls", 1),
        ("tokens", 300), ("memory_mb", 512), ("api_calls", 1),
    ]

    for resource, amount in usage_data:
        tracker.record(resource, amount)

    summary = tracker.get_summary()
    print("  Суммарное потребление:")
    for res, total in summary.items():
        peak = tracker.peak_usage(res)
        print(f"    {res:12s}: суммарно={total:8}, пик={peak:8}")

    # --- 4.2 Управление квотами ---
    print("\n--- 4.2 Управление квотами (Quota Management) ---")

    class QuotaManager:
        """Управление лимитами на ресурсы."""

        def __init__(self):
            self.quotas = {}      # resource → limit
            self.used = {}        # resource → consumed

        def set_quota(self, resource, limit):
            """Установить лимит на ресурс."""
            self.quotas[resource] = limit
            self.used.setdefault(resource, 0)

        def consume(self, resource, amount):
            """Потребить ресурс. Возвращает (ok, remaining)."""
            limit = self.quotas.get(resource, float('inf'))
            current = self.used.get(resource, 0)
            if current + amount > limit:
                return False, limit - current
            self.used[resource] = current + amount
            return True, limit - self.used[resource]

        def usage_pct(self, resource):
            """Процент использования."""
            limit = self.quotas.get(resource, 1)
            used = self.used.get(resource, 0)
            return used / limit * 100

    qm = QuotaManager()
    qm.set_quota("tokens", 1000)
    qm.set_quota("api_calls", 5)
    qm.set_quota("memory_mb", 2048)

    # Потребление ресурсов
    consumption = [
        ("tokens", 300), ("api_calls", 2), ("memory_mb", 512),
        ("tokens", 200), ("api_calls", 1), ("memory_mb", 1024),
        ("tokens", 400), ("api_calls", 2),
    ]

    for resource, amount in consumption:
        ok, remaining = qm.consume(resource, amount)
        status = "OK" if ok else "ПРЕВЫШЕНИЕ"
        print(f"  Потребление {resource}={amount}: {status} "
              f"(осталось {remaining})")

    print("\n  Итоговая квота:")
    for res in ["tokens", "api_calls", "memory_mb"]:
        pct = qm.usage_pct(res)
        filled = "█" * int(pct / 5)
        empty = "░" * (20 - int(pct / 5))
        print(f"    {res:12s}: {pct:5.1f}% [{filled}{empty}]")

    # --- 4.3 Динамическая корректировка ---
    print("\n--- 4.3 Динамическая корректировка квот ---")

    class AdaptiveQuota:
        """Квота, которая адаптируется к нагрузке."""

        def __init__(self, resource, initial_limit):
            self.resource = resource
            self.base_limit = initial_limit
            self.current_limit = initial_limit
            self.history = []          # история потребления
            self.scale_factor = 0.8    # коэффициент масштабирования

        def record(self, usage):
            """Записать потребление и пересчитать квоту."""
            self.history.append(usage)

            # Корректировка: если потребление > 80% лимита — увеличить
            if len(self.history) >= 3:
                recent_avg = sum(self.history[-3:]) / 3
                pct_used = recent_avg / self.current_limit

                if pct_used > 0.8:
                    # Увеличиваем лимит на 20%
                    self.current_limit = round(self.current_limit * 1.2)
                    action = "УВЕЛИЧЕН"
                elif pct_used < 0.3:
                    # Уменьшаем лимит на 20%
                    self.current_limit = round(self.current_limit * self.scale_factor)
                    action = "УМЕНЬШЕН"
                else:
                    action = "БЕЗ ИЗМЕНЕНИЙ"

                return action
            return "НАКОПЛЕНИЕ ДАННЫХ"

    adaptive = AdaptiveQuota("tokens", 1000)

    # Имитируем нарастающую нагрузку
    workload = [200, 300, 500, 700, 900, 850, 800, 300, 200, 100]
    print(f"  Начальный лимит: {adaptive.current_limit}")
    for i, usage in enumerate(workload):
        action = adaptive.record(usage)
        print(f"  Шаг {i+1}: потребление={usage}, лимит={adaptive.current_limit}, "
              f"действие={action}")

    # --- 4.4 Алерты и мониторинг ---
    print("\n--- 4.4 Система алертов ---")

    class AlertSystem:
        """Система предупреждений при превышении порогов."""

        def __init__(self):
            self.rules = []       # (resource, threshold_pct, severity)
            self.alerts = []

        def add_rule(self, resource, threshold_pct, severity):
            """Добавить правило алерта."""
            self.rules.append((resource, threshold_pct, severity))

        def check(self, resource, current_pct):
            """Проверить и сгенерировать алерты."""
            triggered = []
            for res, threshold, severity in self.rules:
                if res == resource and current_pct >= threshold:
                    alert = {
                        "resource": res,
                        "usage": current_pct,
                        "threshold": threshold,
                        "severity": severity,
                    }
                    self.alerts.append(alert)
                    triggered.append(alert)
            return triggered

    alerts = AlertSystem()
    alerts.add_rule("tokens", 70, "WARNING")
    alerts.add_rule("tokens", 90, "CRITICAL")
    alerts.add_rule("memory", 80, "WARNING")
    alerts.add_rule("memory", 95, "CRITICAL")

    # Проверяем потребление
    checks = [
        ("tokens", 65), ("tokens", 75), ("tokens", 92),
        ("memory", 70), ("memory", 85), ("memory", 96),
    ]

    for resource, pct in checks:
        triggered = alerts.check(resource, pct)
        if triggered:
            for a in triggered:
                icon = "🔴" if a["severity"] == "CRITICAL" else "🟡"
                print(f"  {icon} ALERT [{a['severity']}]: {a['resource']} "
                      f"на {a['usage']}% (порог: {a['threshold']}%)")
        else:
            print(f"  ✅ {resource}={pct}%: норма")

    print(f"\n  Всего алертов: {len(alerts.alerts)}")


# =============================================================================
# Запуск всех демонстраций
# =============================================================================

if __name__ == "__main__":
    print("УРОК 177: Resource-Aware Agents")
    print("Бюджет вычислений, управление временем, приоритеты\n")

    demo_compute_budgeting()
    demo_priority_scheduling()
    demo_lazy_evaluation()
    demo_resource_monitoring()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены.")
    print("=" * 70)
