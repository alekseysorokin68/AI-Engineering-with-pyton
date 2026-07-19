"""
208 — Data Pipelines: ETL, потоковая обработка, качество данных

Темы:
  1. ETL Patterns (извлечение, трансформация, загрузка, ELT, эволюция схемы)
  2. Stream Processing (оконные операции, watermarks, exactly-once семантика)
  3. Data Quality (правила валидации, обнаружение аномалий, проверки полноты)
  4. Pipeline Orchestration (DAG-и, планирование, управление зависимостями)

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


# ──────────────────────────────────────────────────────────────────────
# Демо 1: ETL-паттерны
# ──────────────────────────────────────────────────────────────────────
def demo_etl_patterns():
    """Извлечение, трансформация, загрузка, ELT, эволюция схемы."""
    print("=" * 70)
    print("ДЕМО 1: ETL-паттерны (Extract, Transform, Load)")
    print("=" * 70)

    # --- 1.1 Извлечение данных (Extract) ---
    print("\n--- 1.1 Извлечение данных (Extract) ---")

    class DataSource:
        """Абстракция источника данных."""

        def __init__(self, name: str, source_type: str, data: list):
            self.name = name
            self.source_type = source_type
            self._data = data

        def extract(self) -> list:
            """Извлечение всех записей."""
            print(f"    📥 Извлечено {len(self._data)} записей из {self.name} ({self.source_type})")
            return list(self._data)

        def extract_with_filter(self, predicate) -> list:
            """Извлечение с фильтрацией на источнике."""
            filtered = [r for r in self._data if predicate(r)]
            print(f"    📥 Извлечено {len(filtered)}/{len(self._data)} записей "
                  f"с фильтром из {self.name}")
            return filtered

    # Симуляция данных из разных источников
    raw_orders = [
        {"id": 1, "product": "Laptop", "qty": 2, "price": 999.99, "date": "2024-01-15"},
        {"id": 2, "product": "Mouse", "qty": 10, "price": 29.99, "date": "2024-01-15"},
        {"id": 3, "product": "Keyboard", "qty": 5, "price": 79.99, "date": "2024-01-16"},
        {"id": 4, "product": "Monitor", "qty": 1, "price": 549.99, "date": "2024-01-16"},
        {"id": 5, "product": "Headset", "qty": 3, "price": 149.99, "date": "2024-01-17"},
    ]

    raw_inventory = [
        {"product": "Laptop", "warehouse": "A", "stock": 50},
        {"product": "Mouse", "warehouse": "A", "stock": 200},
        {"product": "Keyboard", "warehouse": "B", "stock": 80},
        {"product": "Monitor", "warehouse": "A", "stock": 30},
        {"product": "Headset", "warehouse": "B", "stock": 120},
    ]

    orders_src = DataSource("orders_db", "PostgreSQL", raw_orders)
    inventory_src = DataSource("inventory_api", "REST API", raw_inventory)

    orders = orders_src.extract()
    inventory = inventory_src.extract()

    # Извлечение с фильтром: только крупные заказы
    big_orders = orders_src.extract_with_filter(lambda r: r["qty"] >= 5)

    # --- 1.2 Трансформация данных (Transform) ---
    print("\n--- 1.2 Трансформация данных (Transform) ---")

    class Transformer:
        """Цепочка трансформаций данных."""

        def __init__(self):
            self.steps = []

        def add_step(self, name: str, func):
            """Добавление шага трансформации."""
            self.steps.append((name, func))
            return self

        def apply(self, data: list) -> list:
            """Применение всех шагов последовательно."""
            result = data
            for name, func in self.steps:
                before_count = len(result)
                result = func(result)
                after_count = len(result)
                delta = after_count - before_count
                sign = "+" if delta >= 0 else ""
                print(f"    🔧 {name}: {before_count} → {after_count} "
                      f"({sign}{delta} записей)")
            return result

    # Определяем цепочку трансформаций
    pipeline = Transformer()
    pipeline.add_step(
        "Удаление дубликатов",
        lambda data: list({r["id"]: r for r in data}.values())
    )
    pipeline.add_step(
        "Валидация цен",
        lambda data: [r for r in data if r["price"] > 0]
    )
    pipeline.add_step(
        "Добавление total_cost",
        lambda data: [{**r, "total_cost": round(r["qty"] * r["price"], 2)} for r in data]
    )
    pipeline.add_step(
        "Фильтрация минимального заказа",
        lambda data: [r for r in data if r["total_cost"] >= 100]
    )
    pipeline.add_step(
        "Нормализация названий",
        lambda data: [{**r, "product": r["product"].lower().strip()} for r in data]
    )

    transformed = pipeline.apply(orders)

    print(f"\n  Результат трансформации:")
    for r in transformed:
        print(f"    {r['product']:<12s} qty={r['qty']:>2d}  "
              f"price={r['price']:>8.2f}  total={r['total_cost']:>10.2f}")

    # --- 1.3 Загрузка данных (Load) ---
    print("\n--- 1.3 Загрузка данных (Load) ---")

    class DataLoader:
        """Симуляция загрузки данных в целевое хранилище."""

        def __init__(self, target: str):
            self.target = target
            self.loaded_records = []
            self.load_stats = {"inserted": 0, "updated": 0, "skipped": 0}

        def load(self, data: list, mode: str = "append", key_field: str = "id"):
            """Загрузка данных в целевое хранилище."""
            print(f"    📤 Загрузка в {self.target} (режим: {mode})")

            if mode == "append":
                self.loaded_records.extend(data)
                self.load_stats["inserted"] += len(data)
                print(f"       Вставлено: {len(data)} записей")

            elif mode == "upsert":
                existing_keys = {r[key_field]: i for i, r in enumerate(self.loaded_records)}
                for record in data:
                    key = record[key_field]
                    if key in existing_keys:
                        idx = existing_keys[key]
                        self.loaded_records[idx] = record
                        self.load_stats["updated"] += 1
                    else:
                        self.loaded_records.append(record)
                        self.load_stats["inserted"] += 1
                print(f"       Вставлено: {self.load_stats['inserted']}, "
                      f"Обновлено: {self.load_stats['updated']}")

            elif mode == "replace":
                self.loaded_records = list(data)
                self.load_stats["inserted"] = len(data)
                print(f"       Таблица заменена: {len(data)} записей")

        def get_stats(self) -> dict:
            """Статистика загрузки."""
            return {
                "total_records": len(self.loaded_records),
                **self.load_stats
            }

    loader = DataLoader("data_warehouse/orders_fact")
    loader.load(transformed, mode="replace")

    stats = loader.get_stats()
    print(f"\n  Статистика загрузки:")
    for k, v in stats.items():
        print(f"    {k:<20s} = {v}")

    # --- 1.4 ELT и эволюция схемы ---
    print("\n--- 1.4 ELT и эволюция схемы (Schema Evolution) ---")

    # ELT: сначала загружаем сырые данные, потом трансформируем внутри хранилища
    print("  ELT-подход (Extract → Load → Transform):")
    print("    1. Extract: извлечение сырых данных")
    print("    2. Load: загрузка в Raw Zone (без трансформаций)")
    print("    3. Transform: трансформация SQL-запросами внутри хранилища")
    print(f"\n  Преимущество ELT: сохраняем исходные данные для повторной обработки")

    # Эволюция сехемы
    schema_v1 = {
        "version": 1,
        "fields": [
            {"name": "id", "type": "int", "nullable": False},
            {"name": "product", "type": "string", "nullable": False},
            {"name": "qty", "type": "int", "nullable": False},
            {"name": "price", "type": "float", "nullable": False},
        ]
    }

    schema_v2 = {
        "version": 2,
        "fields": [
            {"name": "id", "type": "int", "nullable": False},
            {"name": "product", "type": "string", "nullable": False},
            {"name": "qty", "type": "int", "nullable": False},
            {"name": "price", "type": "float", "nullable": False},
            {"name": "currency", "type": "string", "nullable": True, "default": "USD"},  # НОВОЕ
            {"name": "discount_pct", "type": "float", "nullable": True, "default": 0.0},  # НОВОЕ
            {"name": "created_at", "type": "timestamp", "nullable": True},  # НОВОЕ
        ]
    }

    def evolve_schema(data: list, old_schema: dict, new_schema: dict) -> list:
        """Эволюция данных при изменении схемы."""
        old_fields = {f["name"] for f in old_schema["fields"]}
        new_fields = {f["name"] for f in new_schema["fields"]}
        added = new_fields - old_fields
        removed = old_fields - new_fields

        print(f"\n  Эволюция схемы v{old_schema['version']} → v{new_schema['version']}")
        print(f"    Добавлены поля: {added if added else 'нет'}")
        print(f"    Удалены поля:   {removed if removed else 'нет'}")

        # Добавляем дефолтные значения для новых полей
        defaults = {}
        for f in new_schema["fields"]:
            if f["name"] in added and "default" in f:
                defaults[f["name"]] = f["default"]
            elif f["name"] in added:
                defaults[f["name"]] = None

        evolved = []
        for record in data:
            new_record = dict(record)
            for field, default in defaults.items():
                new_record[field] = default
            # Удаляем поля, которых нет в новой схеме
            new_record = {k: v for k, v in new_record.items() if k in new_fields}
            evolved.append(new_record)

        return evolved

    evolved_data = evolve_schema(transformed, schema_v1, schema_v2)
    print(f"\n  Пример эволюционировавшей записи:")
    if evolved_data:
        print(f"    {json.dumps(evolved_data[0], indent=6)}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 2: Потоковая обработка (Stream Processing)
# ──────────────────────────────────────────────────────────────────────
def demo_stream_processing():
    """Оконные операции, watermarks, exactly-once семантика."""
    print("=" * 70)
    print("ДЕМО 2: Потоковая обработка (Stream Processing)")
    print("=" * 70)

    # --- 2.1 Потоковые события ---
    print("\n--- 2.1 Потоковые события (Event Stream) ---")

    # Симуляция потока событий кликов
    events = []
    base_time = 1705300000  # epoch seconds
    event_types = ["click", "view", "purchase", "add_to_cart", "search"]

    for i in range(30):
        event = {
            "event_id": f"evt-{i:04d}",
            "user_id": f"user-{random.randint(1, 5):03d}",
            "event_type": random.choice(event_types),
            "timestamp": base_time + random.randint(0, 300),  # 5 минут окно
            "value": round(random.uniform(1.0, 500.0), 2) if random.random() > 0.3 else 0,
        }
        events.append(event)

    events.sort(key=lambda e: e["timestamp"])

    print(f"  Поток событий: {len(events)} событий за ~5 минут")
    print(f"  Типы событий: {set(e['event_type'] for e in events)}")
    print(f"  Уникальные пользователи: {len(set(e['user_id'] for e in events))}")

    # Примеры событий
    print(f"\n  Примеры событий:")
    for e in events[:5]:
        ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        print(f"    [{ts}] {e['event_type']:<14s} user={e['user_id']} "
              f"value={e['value']:>7.2f}")

    # --- 2.2 Оконные операции (Windowing) ---
    print("\n--- 2.2 Оконные операции (Windowing) ---")

    def tumbling_window(events: list, window_size_sec: int) -> dict:
        """РасFixedSize (tumbling) окна."""
        windows = {}
        for event in events:
            window_start = (event["timestamp"] // window_size_sec) * window_size_sec
            if window_start not in windows:
                windows[window_start] = []
            windows[window_start].append(event)
        return windows

    def sliding_window(events: list, window_size_sec: int, slide_sec: int) -> dict:
        """Скользящие (sliding) окна."""
        if not events:
            return {}
        min_ts = events[0]["timestamp"]
        max_ts = events[-1]["timestamp"]
        windows = {}
        current = min_ts
        while current <= max_ts:
            window_events = [
                e for e in events
                if current <= e["timestamp"] < current + window_size_sec
            ]
            if window_events:
                windows[current] = window_events
            current += slide_sec
        return windows

    # Tumbling окна по 60 секунд
    tumbling = tumbling_window(events, 60)
    print(f"  Tumbling окна (60 сек): {len(tumbling)} окон")
    for start_ts, window_events in sorted(tumbling.items()):
        ts = time.strftime("%H:%M:%S", time.localtime(start_ts))
        types_count = collections.Counter(e["event_type"] for e in window_events)
        print(f"    [{ts}] {len(window_events):>3d} событий: {dict(types_count)}")

    # Sliding окна: 120 сек окно, сдвиг 30 сек
    sliding = sliding_window(events, 120, 30)
    print(f"\n  Sliding окна (120 сек, сдвиг 30 сек): {len(sliding)} окон")
    for i, (start_ts, window_events) in enumerate(sorted(sliding.items())[:5]):
        ts = time.strftime("%H:%M:%S", time.localtime(start_ts))
        total_value = sum(e["value"] for e in window_events)
        print(f"    [{ts}] {len(window_events):>3d} событий, суммарная ценность: {total_value:.2f}")

    # --- 2.3 Watermarks ---
    print("\n--- 2.3 Watermarks (отслеживание задержек) ---")

    class WatermarkTracker:
        """Отслеживание watermarks для обработки задержанных событий."""

        def __init__(self, max_delay_sec: int):
            self.max_delay = max_delay_sec
            self.current_watermark = 0
            self.late_events = []
            self.processed_events = []

        def process_event(self, event: dict) -> str:
            """Обработка события с проверкой watermark."""
            event_ts = event["timestamp"]

            # Обновляем watermark: максимальное время события - допустимая задержка
            self.current_watermark = max(self.current_watermark,
                                         event_ts - self.max_delay)

            if event_ts < self.current_watermark:
                self.late_events.append(event)
                return "LATE"
            else:
                self.processed_events.append(event)
                return "ON_TIME"

    tracker = WatermarkTracker(max_delay_sec=30)

    print(f"  Максимальная задержка: {tracker.max_delay} сек")
    print(f"  Обработка событий:")

    late_count = 0
    on_time_count = 0
    for event in events[:15]:
        status = tracker.process_event(event)
        ts = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
        wm = time.strftime("%H:%M:%S", time.localtime(tracker.current_watermark))
        icon = "✅" if status == "ON_TIME" else "⚠️"
        print(f"    {icon} [{ts}] {event['event_type']:<14s} watermark={wm} → {status}")
        if status == "LATE":
            late_count += 1
        else:
            on_time_count += 1

    print(f"\n  Итого: {on_time_count} вовремя, {late_count} задержанных "
          f"({late_count/(on_time_count+late_count)*100:.0f}%)")

    # --- 2.4 Exactly-once семантика ---
    print("\n--- 2.4 Exactly-once семантика ---")

    class ExactlyOnceProcessor:
        """Симуляция exactly-once обработки через идемпотентность."""

        def __init__(self):
            self.processed_ids = set()
            self.results = []
            self.duplicates_caught = 0

        def process(self, event: dict) -> dict:
            """Обработка события с гарантией exactly-once."""
            event_id = event["event_id"]

            # Проверка идемпотентности: был ли обработан ранее?
            if event_id in self.processed_ids:
                self.duplicates_caught += 1
                return {"status": "duplicate_skipped", "event_id": event_id}

            # Обработка и запись ID
            self.processed_ids.add(event_id)
            result = {
                "status": "processed",
                "event_id": event_id,
                "result": event["value"] * 1.1,  # трансформация
            }
            self.results.append(result)
            return result

    processor = ExactlyOnceProcessor()

    # Симуляция: отправляем события с дубликатами
    test_events = events[:10]
    duplicate_events = [events[2], events[5], events[7]]  # дубликаты

    all_test = test_events + duplicate_events
    random.shuffle(all_test)

    print(f"  Отправлено событий: {len(test_events)} уникальных + "
          f"{len(duplicate_events)} дубликатов = {len(all_test)} всего")
    print(f"\n  Результаты обработки:")
    for event in all_test:
        result = processor.process(event)
        status = result["status"]
        icon = "✅" if status == "processed" else "⏭️"
        print(f"    {icon} {result['event_id']} → {status}")

    print(f"\n  Обработано уникальных: {len(processor.results)}")
    print(f"  Поймано дубликатов:   {processor.duplicates_caught}")
    print(f"  Гарантия exactly-once: "
          f"{'✅ ВЫПОЛНЕНА' if processor.duplicates_caught == len(duplicate_events) else '❌ НАРУШЕНА'}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 3: Качество данных (Data Quality)
# ──────────────────────────────────────────────────────────────────────
def demo_data_quality():
    """Правила валидации, обнаружение аномалий, проверки полноты."""
    print("=" * 70)
    print("ДЕМО 3: Качество данных (Data Quality)")
    print("=" * 70)

    # --- 3.1 Правила валидации ---
    print("\n--- 3.1 Правила валидации (Validation Rules) ---")

    class DataValidator:
        """Валидация данных по набору правил."""

        def __init__(self):
            self.rules = []
            self.violations = []

        def add_rule(self, name: str, check_fn, severity: str = "error"):
            """Добавление правила валидации."""
            self.rules.append({"name": name, "check": check_fn, "severity": severity})

        def validate(self, data: list) -> dict:
            """Проверка данных по всем правилам."""
            self.violations = []
            results = {"passed": 0, "failed": 0, "warnings": 0, "details": []}

            for rule in self.rules:
                rule_violations = []
                for i, record in enumerate(data):
                    if not rule["check"](record):
                        rule_violations.append(i)

                if rule_violations:
                    results["details"].append({
                        "rule": rule["name"],
                        "severity": rule["severity"],
                        "violations": len(rule_violations),
                        "examples": rule_violations[:3],
                    })
                    if rule["severity"] == "error":
                        results["failed"] += len(rule_violations)
                    else:
                        results["warnings"] += len(rule_violations)
                    self.violations.extend(
                        {"rule": rule["name"], "index": i, "severity": rule["severity"]}
                        for i in rule_violations
                    )
                else:
                    results["passed"] += 1

            return results

    # Тестовые данные с дефектами
    test_data = [
        {"name": "Alice", "age": 28, "email": "alice@test.com", "salary": 75000, "dept": "engineering"},
        {"name": "Bob", "age": 35, "email": "bob@test.com", "salary": 85000, "dept": "marketing"},
        {"name": "", "age": -5, "email": "invalid-email", "salary": 50000, "dept": "engineering"},
        {"name": "Diana", "age": 42, "email": "diana@test.com", "salary": 150000, "dept": "engineering"},
        {"name": "Eve", "age": 200, "email": "eve@test.com", "salary": 45000, "dept": ""},
        {"name": "Frank", "age": 31, "email": "frank@test.com", "salary": -1000, "dept": "sales"},
    ]

    validator = DataValidator()
    validator.add_rule("Имя не пустое", lambda r: len(r["name"].strip()) > 0)
    validator.add_rule("Возраст 0-120", lambda r: 0 <= r["age"] <= 120)
    validator.add_rule("Корректный email", lambda r: "@" in r["email"] and "." in r["email"])
    validator.add_rule("Зарплата > 0", lambda r: r["salary"] > 0)
    validator.add_rule("Отдел указан", lambda r: len(r["dept"].strip()) > 0, severity="warning")

    results = validator.validate(test_data)

    print(f"  Данные: {len(test_data)} записей")
    print(f"  Правила: {len(validator.rules)}")
    print(f"\n  Результаты валидации:")
    for detail in results["details"]:
        severity_icon = "❌" if detail["severity"] == "error" else "⚠️"
        print(f"    {severity_icon} {detail['rule']}: {detail['violations']} нарушений "
              f"(индексы: {detail['examples']})")

    print(f"\n  Итого: ✅ {results['passed']} правил пройдено, "
          f"❌ {results['failed']} ошибок, ⚠️ {results['warnings']} предупреждений")

    # --- 3.2 Обнаружение аномалий ---
    print("\n--- 3.2 Обнаружение аномалий (Anomaly Detection) ---")

    def detect_anomalies_iqr(values: list, factor: float = 1.5) -> dict:
        """Обнаружение аномалий методом IQR (межквартильный размах)."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        # Квартили
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        anomalies = [(i, v) for i, v in enumerate(values)
                     if v < lower_bound or v > upper_bound]

        return {
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower_bound, "upper_bound": upper_bound,
            "anomalies": anomalies,
            "total_points": n,
            "anomaly_count": len(anomalies),
        }

    # Симуляция: метрики latency за день
    random.seed(42)
    latencies = [random.gauss(50, 10) for _ in range(100)]
    # Добавляем аномалии
    latencies[15] = 350  # выброс
    latencies[42] = 280  # выброс
    latencies[87] = 5    # слишком низкое

    result = detect_anomalies_iqr(latencies)

    print(f"  Анализ латентности: {result['total_points']} измерений")
    print(f"\n  Статистика:")
    print(f"    Q1 (25-й перцентиль): {result['q1']:.1f} ms")
    print(f"    Q3 (75-й перцентиль): {result['q3']:.1f} ms")
    print(f"    IQR:                   {result['iqr']:.1f} ms")
    print(f"    Нижняя граница:        {result['lower_bound']:.1f} ms")
    print(f"    Верхняя граница:       {result['upper_bound']:.1f} ms")

    print(f"\n  Обнаруженные аномалии ({result['anomaly_count']}):")
    for idx, val in result["anomalies"]:
        print(f"    ⚠️  Измерение [{idx}]: {val:.1f} ms "
              f"{'(слишком высокое)' if val > result['upper_bound'] else '(слишком низкое)'}")

    # --- 3.3 Проверки полноты (Completeness Checks) ---
    print("\n--- 3.3 Проверки полноты (Completeness Checks) ---")

    class CompletenessChecker:
        """Проверка полноты данных по ожидаемой схеме."""

        def __init__(self, schema: dict):
            self.schema = schema  # {field_name: {"required": bool, "type": str}}

        def check(self, data: list) -> dict:
            """Проверка полноты набора данных."""
            report = {
                "total_records": len(data),
                "fields": {},
                "overall_completeness": 0,
            }

            total_cells = 0
            filled_cells = 0

            for field_name, field_spec in self.schema.items():
                non_null = 0
                type_correct = 0
                for record in data:
                    value = record.get(field_name)
                    total_cells += 1
                    if value is not None and value != "":
                        non_null += 1
                        filled_cells += 1
                        # Проверка типа
                        if field_spec.get("type") == "string" and isinstance(value, str):
                            type_correct += 1
                        elif field_spec.get("type") == "number" and isinstance(value, (int, float)):
                            type_correct += 1
                        elif field_spec.get("type") == "string":
                            type_correct += 1  # для прочих

                completeness = non_null / len(data) * 100 if data else 0
                report["fields"][field_name] = {
                    "completeness_pct": completeness,
                    "non_null": non_null,
                    "total": len(data),
                    "required": field_spec.get("required", False),
                }

            report["overall_completeness"] = filled_cells / total_cells * 100 if total_cells else 0
            return report

    schema = {
        "name": {"required": True, "type": "string"},
        "age": {"required": True, "type": "number"},
        "email": {"required": True, "type": "string"},
        "phone": {"required": False, "type": "string"},
        "address": {"required": False, "type": "string"},
    }

    # Данные с пропусками
    data_with_gaps = [
        {"name": "Alice", "age": 28, "email": "alice@test.com", "phone": "+1234", "address": "St 1"},
        {"name": "Bob", "age": 35, "email": "bob@test.com", "phone": None, "address": "St 2"},
        {"name": "Charlie", "age": None, "email": None, "phone": "+5678", "address": None},
        {"name": "Diana", "age": 42, "email": "diana@test.com", "phone": "+9012", "address": "St 4"},
        {"name": None, "age": 20, "email": "eve@test.com", "phone": None, "address": "St 5"},
    ]

    checker = CompletenessChecker(schema)
    report = checker.check(data_with_gaps)

    print(f"  Набор данных: {report['total_records']} записей, {len(schema)} полей")
    print(f"\n  Полнота по полям:")
    for field, info in report["fields"].items():
        req = " [ОБЯЗАТЕЛЬНОЕ]" if info["required"] else ""
        icon = "✅" if info["completeness_pct"] == 100 else ("❌" if info["required"] else "⚠️")
        print(f"    {icon} {field:<15s} {info['completeness_pct']:>5.1f}% "
              f"({info['non_null']}/{info['total']}){req}")

    print(f"\n  Общая полнота: {report['overall_completeness']:.1f}%")

    # --- 3.4 Мониторинг распределения ---
    print("\n--- 3.4 Мониторинг распределения данных ---")

    def distribution_report(values: list, bins: int = 10) -> dict:
        """Анализ распределения значений."""
        min_val = min(values)
        max_val = max(values)
        bin_width = (max_val - min_val) / bins

        histogram = [0] * bins
        for v in values:
            idx = min(int((v - min_val) / bin_width), bins - 1)
            histogram[idx] += 1

        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance)

        return {
            "count": len(values),
            "min": min_val, "max": max_val,
            "mean": mean_val, "std": std_val,
            "histogram": histogram,
            "bin_width": bin_width,
        }

    # Распределение значений зарплат
    salaries = [random.gauss(70000, 15000) for _ in range(200)]
    salaries = [max(20000, min(200000, s)) for s in salaries]

    dist = distribution_report(salaries, bins=8)

    print(f"  Распределение зарплат: {dist['count']} значений")
    print(f"    Мин:  ${dist['min']:>10,.0f}")
    print(f"    Макс: ${dist['max']:>10,.0f}")
    print(f"    Сред: ${dist['mean']:>10,.0f}")
    print(f"    СКО:  ${dist['std']:>10,.0f}")

    print(f"\n  Гистограмма:")
    max_count = max(dist["histogram"])
    for i, count in enumerate(dist["histogram"]):
        bin_start = dist["min"] + i * dist["bin_width"]
        bin_end = bin_start + dist["bin_width"]
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"    ${bin_start:>8,.0f} - ${bin_end:>8,.0f} | {bar} ({count})")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 4: Оркестрация пайплайнов (Pipeline Orchestration)
# ──────────────────────────────────────────────────────────────────────
def demo_pipeline_orchestration():
    """DAG-и, планирование, управление зависимостями."""
    print("=" * 70)
    print("ДЕМО 4: Оркестрация пайплайнов (Pipeline Orchestration)")
    print("=" * 70)

    # --- 4.1 DAG (Directed Acyclic Graph) ---
    print("\n--- 4.1 DAG (ориентированный ациклический граф) ---")

    class DAG:
        """Ориентированный ациклический граф для описания пайплайна."""

        def __init__(self):
            self.nodes = {}  # name -> {task, dependencies, status}
            self.edges = []  # [(from, to)]

        def add_node(self, name: str, task: str, dependencies: list = None):
            """Добавление узла (задачи) в DAG."""
            self.nodes[name] = {
                "task": task,
                "dependencies": dependencies or [],
                "status": "pending",
                "duration_sec": 0,
            }
            for dep in (dependencies or []):
                self.edges.append((dep, name))

        def topological_sort(self) -> list:
            """Топологическая сортировка для определения порядка выполнения."""
            in_degree = {n: 0 for n in self.nodes}
            for _, to_node in self.edges:
                in_degree[to_node] += 1

            queue = [n for n, d in in_degree.items() if d == 0]
            order = []

            while queue:
                queue.sort()  # детерминированный порядок
                node = queue.pop(0)
                order.append(node)
                for _, to_node in self.edges:
                    if _ == node:  # from_node == node
                        in_degree[to_node] -= 1
                        if in_degree[to_node] == 0:
                            queue.append(to_node)

            return order

        def get_parallel_groups(self) -> list:
            """Определение групп задач, которые можно выполнять параллельно."""
            in_degree = {n: 0 for n in self.nodes}
            for _, to_node in self.edges:
                in_degree[to_node] += 1

            remaining = set(self.nodes.keys())
            groups = []

            while remaining:
                # Задачи без незавершённых зависимостей
                ready = sorted([n for n in remaining if in_degree[n] == 0])
                if not ready:
                    break  # цикл (не должно happen для DAG)
                groups.append(ready)
                for node in ready:
                    remaining.remove(node)
                    for _, to_node in self.edges:
                        if _ == node:
                            in_degree[to_node] -= 1

            return groups

    # Создаём DAG для ML-пайплайна
    pipeline = DAG()
    pipeline.add_node("fetch_data", "Загрузка данных из PostgreSQL")
    pipeline.add_node("validate_schema", "Валидация схемы данных", ["fetch_data"])
    pipeline.add_node("clean_data", "Очистка и дедупликация", ["validate_schema"])
    pipeline.add_node("feature_eng", "Инженерия признаков", ["clean_data"])
    pipeline.add_node("train_model", "Обучение модели", ["feature_eng"])
    pipeline.add_node("evaluate", "Оценка качества", ["train_model"])
    pipeline.add_node("register", "Регистрация в реестре", ["evaluate"])
    pipeline.add_node("deploy_staging", "Деплой в staging", ["register"])
    pipeline.add_node("integration_test", "Интеграционные тесты", ["deploy_staging"])
    pipeline.add_node("deploy_prod", "Деплой в production", ["integration_test"])

    # Дополнительные параллельные задачи
    pipeline.add_node("generate_report", "Генерация отчёта", ["evaluate"])
    pipeline.add_node("notify_team", "Уведомление команды", ["deploy_prod", "generate_report"])

    order = pipeline.topological_sort()
    print(f"  Порядок выполнения (topological sort):")
    for i, node in enumerate(order):
        task = pipeline.nodes[node]["task"]
        deps = pipeline.nodes[node]["dependencies"]
        dep_str = f" ← {deps}" if deps else ""
        print(f"    {i+1:>2d}. {node:<25s} {task}{dep_str}")

    groups = pipeline.get_parallel_groups()
    print(f"\n  Параллельные группы:")
    for i, group in enumerate(groups):
        tasks_str = ", ".join(f"{n} ({pipeline.nodes[n]['task'][:20]})" for n in group)
        print(f"    Группа {i+1}: {tasks_str}")

    # --- 4.2 Планирование (Scheduling) ---
    print("\n--- 4.2 Планирование (Scheduling) ---")

    class Schedule:
        """Расписание запуска пайплайнов."""

        def __init__(self, name: str, cron_expr: str):
            self.name = name
            self.cron_expr = cron_expr
            self.next_run = time.time()

        def parse_cron(self) -> dict:
            """Парсинг cron-выражения."""
            parts = self.cron_expr.split()
            return {
                "minute": parts[0] if len(parts) > 0 else "*",
                "hour": parts[1] if len(parts) > 1 else "*",
                "day": parts[2] if len(parts) > 2 else "*",
                "month": parts[3] if len(parts) > 3 else "*",
                "weekday": parts[4] if len(parts) > 4 else "*",
            }

        def describe(self) -> str:
            """Человекочитаемое описание расписания."""
            cron = self.parse_cron()
            parts = []

            if cron["weekday"] != "*":
                days_map = {"1-5": "по будням", "0,6": "по выходным", "*": "ежедневно"}
                parts.append(days_map.get(cron["weekday"], f"дн. недели={cron['weekday']}"))
            else:
                parts.append("ежедневно")

            if cron["hour"] != "*" and cron["minute"] != "*":
                parts.append(f"в {cron['hour']}:{cron['minute'].zfill(2)}")
            elif cron["hour"] != "*":
                parts.append(f"каждый час {cron['hour']}")

            return " ".join(parts)

    schedules = [
        Schedule("daily_etl", "0 2 * * *"),          # каждый день в 2:00
        Schedule("hourly_metrics", "7 * * * *"),      # каждый час на 7-й минуте
        Schedule("weekly_report", "0 9 * * 1"),       # понедельник в 9:00
        Schedule("monthly_retrain", "0 3 1 * *"),     # 1-го числа в 3:00
        Schedule("realtime_monitor", "*/5 * * * *"),   # каждые 5 минут
    ]

    print(f"  Расписания пайплайнов:")
    for s in schedules:
        cron = s.parse_cron()
        desc = s.describe()
        print(f"    {s.name:<25s} cron={s.cron_expr:<15s} → {desc}")

    # --- 4.3 Управление зависимостями ---
    print("\n--- 4.3 Управление зависимостями (Dependency Management) ---")

    class TaskRunner:
        """Выполнитель задач с управлением зависимостями."""

        def __init__(self, dag: DAG):
            self.dag = dag
            self.execution_log = []

        def run(self) -> bool:
            """Выполнение всех задач в топологическом порядке."""
            order = self.dag.topological_sort()

            for node_name in order:
                node = self.dag.nodes[node_name]

                # Проверка зависимостей
                deps_ok = all(
                    self.dag.nodes[dep]["status"] == "completed"
                    for dep in node["dependencies"]
                )

                if not deps_ok:
                    node["status"] = "failed"
                    self.execution_log.append({
                        "node": node_name, "status": "failed",
                        "reason": "Зависимость не выполнена",
                    })
                    print(f"    ❌ {node_name}: ЗАВИСИМОСТЬ НЕ ВЫПОЛНЕНА")
                    return False

                # Выполнение задачи (симуляция)
                node["status"] = "running"
                duration = random.uniform(0.5, 5.0)
                node["duration_sec"] = round(duration, 2)

                # Симуляция: 90% шанс успеха
                success = random.random() < 0.9

                if success:
                    node["status"] = "completed"
                    self.execution_log.append({
                        "node": node_name, "status": "completed",
                        "duration": duration,
                    })
                    print(f"    ✅ {node_name}: выполнено за {duration:.1f}с")
                else:
                    node["status"] = "failed"
                    self.execution_log.append({
                        "node": node_name, "status": "failed",
                        "reason": "Ошибка выполнения",
                    })
                    print(f"    ❌ {node_name}: ОШИБКА ВЫПОЛНЕНИЯ")
                    return False

            return True

    runner = TaskRunner(pipeline)
    random.seed(42)  # фиксируем seed для воспроизводимости
    print(f"  Запуск пайплайна:")
    success = runner.run()

    # Статистика выполнения
    completed = [l for l in runner.execution_log if l["status"] == "completed"]
    total_time = sum(l.get("duration", 0) for l in completed)

    print(f"\n  Результат: {'✅ УСПЕХ' if success else '❌ НЕУДАЧА'}")
    print(f"  Задач выполнено: {len(completed)}/{len(pipeline.nodes)}")
    print(f"  Общее время: {total_time:.1f}с (с учётом параллелизма)")

    # --- 4.4 Мониторинг и алертинг ---
    print("\n--- 4.4 Мониторинг и алертинг ---")

    class PipelineMonitor:
        """Мониторинг пайплайнов с алертингом."""

        def __init__(self):
            self.metrics = []
            self.alerts = []

        def record_metric(self, pipeline_name: str, metric: str, value: float):
            """Запись метрики пайплайна."""
            self.metrics.append({
                "pipeline": pipeline_name,
                "metric": metric,
                "value": value,
                "timestamp": time.time(),
            })

        def check_alert(self, pipeline_name: str, metric: str,
                        threshold: float, direction: str = "above"):
            """Проверка условия для алерта."""
            recent = [
                m for m in self.metrics
                if m["pipeline"] == pipeline_name and m["metric"] == metric
            ]
            if not recent:
                return

            latest = recent[-1]["value"]
            triggered = (direction == "above" and latest > threshold) or \
                        (direction == "below" and latest < threshold)

            if triggered:
                alert = {
                    "pipeline": pipeline_name,
                    "metric": metric,
                    "value": latest,
                    "threshold": threshold,
                    "severity": "critical" if abs(latest - threshold) > threshold * 0.5 else "warning",
                }
                self.alerts.append(alert)

        def status_report(self):
            """Отчёт о статусе пайплайнов."""
            print(f"\n  Метрики пайплайнов:")
            by_pipeline = collections.defaultdict(list)
            for m in self.metrics:
                by_pipeline[m["pipeline"]].append(m)

            for pipeline_name, metrics in by_pipeline.items():
                print(f"\n    {pipeline_name}:")
                for m in metrics:
                    print(f"      {m['metric']:<30s} = {m['value']:.2f}")

            if self.alerts:
                print(f"\n  ⚠️ Алерты ({len(self.alerts)}):")
                for alert in self.alerts:
                    sev = "🔴 КРИТИЧЕСКИЙ" if alert["severity"] == "critical" else "🟡 ПРЕДУПРЕЖДЕНИЕ"
                    print(f"    {sev} {alert['pipeline']}: {alert['metric']}={alert['value']:.2f} "
                          f"(порог: {alert['threshold']:.2f})")
            else:
                print(f"\n  ✅ Алертов нет")

    monitor = PipelineMonitor()

    # Запись метрик
    monitor.record_metric("daily_etl", "duration_sec", 847.3)
    monitor.record_metric("daily_etl", "records_processed", 125000)
    monitor.record_metric("daily_etl", "error_rate", 0.02)
    monitor.record_metric("hourly_metrics", "duration_sec", 45.2)
    monitor.record_metric("hourly_metrics", "latency_ms", 120.5)

    # Проверка алертов
    monitor.check_alert("daily_etl", "error_rate", threshold=0.01, direction="above")
    monitor.check_alert("hourly_metrics", "latency_ms", threshold=200, direction="above")

    monitor.status_report()

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_etl_patterns()
    demo_stream_processing()
    demo_data_quality()
    demo_pipeline_orchestration()
