"""166 — Agent Deployment: масштабирование, мониторинг, управление стоимостью

Темы:
  1. Scaling Patterns (горизонтальное масштабирование, балансировка нагрузки, очереди)
  2. Monitoring (задержка, частота успеха, использование токенов, отслеживание стоимости)
  3. Cost Management (бюджетирование токенов, маршрутизация моделей, кеширование)
  4. Production Concerns (идемпотентность, повторные попытки, таймауты, circuit breaker)

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
# Демо 1: Паттерны масштабирования
# ============================================================
def demo_scaling_patterns():
    """Демонстрация паттернов масштабирования агентов."""
    print("=" * 70)
    print("ДЕМО 1: ПАТТЕРНЫ МАСШТАБИРОВАНИЯ (Scaling Patterns)")
    print("=" * 70)

    # --- 1.1 Горизонтальное масштабирование ---
    print("\n[1.1] Горизонтальное масштабирование (Horizontal Scaling)")
    print("-" * 50)

    class HorizontalScaler:
        """Симуляция горизонтального масштабирования агентов."""

        def __init__(self):
            # Пул экземпляров агентов
            self.instances = []
            self.request_count = 0

        def add_instance(self, instance_id: str, capacity: int = 100):
            """Добавляет новый экземпляр агента."""
            self.instances.append({
                "id": instance_id,
                "capacity": capacity,
                "current_load": 0,
                "status": "healthy",
                "requests_handled": 0
            })

        def route_request(self) -> dict:
            """Маршрутизирует запрос к наименее загруженному экземпляру."""
            # Фильтруем только здоровые экземпляры
            healthy = [i for i in self.instances if i["status"] == "healthy"]

            if not healthy:
                return {"error": "Нет доступных экземпляров"}

            # Выбираем экземпляр с наименьшей нагрузкой
            target = min(healthy, key=lambda x: x["current_load"])
            target["current_load"] += 1
            target["requests_handled"] += 1
            self.request_count += 1

            return {
                "instance_id": target["id"],
                "current_load": target["current_load"],
                "capacity": target["capacity"],
                "utilization": round(target["current_load"] / target["capacity"] * 100, 1)
            }

        def scale_out(self, threshold: float = 80.0):
            """Добавляет экземпляр при превышении порога загрузки."""
            healthy = [i for i in self.instances if i["status"] == "healthy"]
            if not healthy:
                return {"action": "scale_out", "reason": "Нет здоровых экземпляров"}

            avg_utilization = sum(
                i["current_load"] / i["capacity"] * 100 for i in healthy
            ) / len(healthy)

            if avg_utilization > threshold:
                new_id = f"agent-{len(self.instances) + 1}"
                self.add_instance(new_id, capacity=100)
                return {
                    "action": "scale_out",
                    "new_instance": new_id,
                    "avg_utilization": round(avg_utilization, 1),
                    "total_instances": len(self.instances)
                }

            return {"action": "none", "avg_utilization": round(avg_utilization, 1)}

        def get_stats(self) -> dict:
            """Возвращает статистику кластера."""
            healthy = [i for i in self.instances if i["status"] == "healthy"]
            return {
                "total_instances": len(self.instances),
                "healthy_instances": len(healthy),
                "total_requests": self.request_count,
                "avg_load": round(
                    sum(i["current_load"] for i in healthy) / max(len(healthy), 1), 2
                )
            }

    scaler = HorizontalScaler()

    # Добавляем экземпляры
    for i in range(3):
        scaler.add_instance(f"agent-{i+1}", capacity=100)
        print(f"  Добавлен экземпляр agent-{i+1} (ёмкость: 100)")

    # Нагружаем экземпляры
    print("\n  Маршрутизация 5 запросов:")
    for i in range(5):
        result = scaler.route_request()
        print(f"    Запрос {i+1} → {result['instance_id']} "
              f"(загрузка: {result['current_load']}/{result['capacity']}, "
              f"использование: {result['utilization']}%)")

    # Попытка масштабирования
    print("\n  Проверка необходимости масштабирования:")
    for _ in range(3):
        # Генерируем нагрузку
        for _ in range(15):
            scaler.route_request()

        scale_result = scaler.scale_out(threshold=70.0)
        if scale_result["action"] == "scale_out":
            print(f"    МАСШТАБИРОВАНИЕ: добавлен {scale_result['new_instance']} "
                  f"(средняя загрузка: {scale_result['avg_utilization']}%)")
        else:
            print(f"    Масштабирование не требуется (средняя загрузка: {scale_result['avg_utilization']}%)")

    stats = scaler.get_stats()
    print(f"\n  Статистика кластера: {stats}")

    # --- 1.2 Балансировка нагрузки ---
    print("\n[1.2] Балансировка нагрузки (Load Balancing)")
    print("-" * 50)

    class LoadBalancer:
        """Балансировщик нагрузки для агентов."""

        def __init__(self):
            self.backends = []
            self.request_history = []

        def add_backend(self, name: str, weight: int = 1):
            """Добавляет бэкенд с весом."""
            self.backends.append({
                "name": name,
                "weight": weight,
                "requests": 0,
                "response_times": []
            })

        def round_robin(self) -> dict:
            """Стратегия Round Robin."""
            if not self.backends:
                return {"error": "Нет бэкендов"}

            min_requests = min(b["requests"] for b in self.backends)
            target = next(b for b in self.backends if b["requests"] == min_requests)

            target["requests"] += 1
            return {"strategy": "round_robin", "target": target["name"]}

        def weighted_round_robin(self) -> dict:
            """Взвешенная стратегия Round Robin."""
            if not self.backends:
                return {"error": "Нет бэкендов"}

            total_weight = sum(b["weight"] for b in self.backends)
            random_val = random.random() * total_weight

            cumulative = 0
            for backend in self.backends:
                cumulative += backend["weight"]
                if random_val <= cumulative:
                    backend["requests"] += 1
                    return {"strategy": "weighted_rr", "target": backend["name"]}

            # Fallback
            self.backends[-1]["requests"] += 1
            return {"strategy": "weighted_rr", "target": self.backends[-1]["name"]}

        def least_connections(self, active_connections: dict) -> dict:
            """Стратегия наименьших соединений."""
            if not self.backends:
                return {"error": "Нет бэкендов"}

            target = min(self.backends, key=lambda b: active_connections.get(b["name"], 0))
            target["requests"] += 1
            return {
                "strategy": "least_conn",
                "target": target["name"],
                "active": active_connections.get(target["name"], 0)
            }

        def get_stats(self) -> list:
            """Возвращает статистику бэкендов."""
            return [
                {"name": b["name"], "requests": b["requests"], "weight": b["weight"]}
                for b in self.backends
            ]

    lb = LoadBalancer()

    # Добавляем бэкенды с разными весами
    backends = [("agent-fast-1", 3), ("agent-fast-2", 3), ("agent-slow-1", 1)]
    for name, weight in backends:
        lb.add_backend(name, weight)
        print(f"  Бэкенд: {name} (вес: {weight})")

    # Round Robin
    print("\n  Round Robin (10 запросов):")
    for i in range(10):
        result = lb.round_robin()
        print(f"    Запрос {i+1} → {result['target']}")

    # Сброс счётчиков для следующей стратегии
    for b in lb.backends:
        b["requests"] = 0

    # Weighted Round Robin
    print("\n  Weighted Round Robin (15 запросов):")
    for i in range(15):
        result = lb.weighted_round_robin()
        print(f"    Запрос {i+1} → {result['target']}", end="")
        if (i + 1) % 5 == 0:
            print()  # Перенос строки каждые 5 запросов
    print()

    # Least Connections
    print("\n  Least Connections:")
    active = {"agent-fast-1": 5, "agent-fast-2": 2, "agent-slow-1": 8}
    print(f"  Активные соединения: {active}")
    for i in range(6):
        result = lb.least_connections(active)
        active[result["target"]] = active.get(result["target"], 0) + 1
        print(f"    Запрос {i+1} → {result['target']} (активных: {result['active'] + 1})")

    print(f"\n  Статистика: {lb.get_stats()}")

    # --- 1.3 Очередь задач ---
    print("\n[1.3] Очередь задач (Queue-Based Scaling)")
    print("-" * 50)

    class TaskQueue:
        """Очередь задач с динамическим масштабированием."""

        def __init__(self, max_queue_size: int = 100):
            self.max_queue_size = max_queue_size
            self.queue = []
            self.processed = []
            self.workers = 0
            self.tasks_per_worker = 5  # Максимум задач на воркер

        def enqueue(self, task: dict) -> dict:
            """Добавляет задачу в очередь."""
            if len(self.queue) >= self.max_queue_size:
                return {"success": False, "reason": "Очередь переполнена"}

            task["id"] = hashlib.md5(json.dumps(task).encode()).hexdigest()[:8]
            task["enqueued_at"] = time.time()
            task["status"] = "pending"
            self.queue.append(task)

            return {
                "success": True,
                "task_id": task["id"],
                "queue_size": len(self.queue)
            }

        def dequeue(self) -> dict:
            """Извлекает задачу из очереди."""
            if not self.queue:
                return {"error": "Очередь пуста"}

            task = self.queue.pop(0)
            task["status"] = "processing"
            task["started_at"] = time.time()
            return task

        def complete_task(self, task_id: str, result: dict):
            """Отмечает задачу как выполненную."""
            task = next((t for t in self.processed if t["id"] == task_id), None)
            if task:
                task["status"] = "completed"
                task["result"] = result
                task["completed_at"] = time.time()

        def should_scale_workers(self) -> dict:
            """Определяет, нужно ли мен количество воркеров."""
            queue_length = len(self.queue)
            optimal_workers = math.ceil(queue_length / self.tasks_per_worker)
            optimal_workers = max(1, min(optimal_workers, 10))  # Лимит 1-10 воркеров

            return {
                "current_workers": self.workers,
                "optimal_workers": optimal_workers,
                "queue_length": queue_length,
                "action": "scale_up" if optimal_workers > self.workers
                         else "scale_down" if optimal_workers < self.workers
                         else "maintain"
            }

    queue = TaskQueue(max_queue_size=50)

    # Добавляем задачи
    print("  Добавление задач в очередь:")
    tasks = [
        {"type": "code_analysis", "priority": "high", "size": 1024},
        {"type": "data_processing", "priority": "medium", "size": 512},
        {"type": "model_inference", "priority": "high", "size": 2048},
        {"type": "report_generation", "priority": "low", "size": 256},
        {"type": "code_analysis", "priority": "medium", "size": 768}
    ]

    for task in tasks:
        result = queue.enqueue(task)
        print(f"    Задача {result['task_id']}: {task['type']} (очередь: {result['queue_size']})")

    # Обработка задач
    print("\n  Обработка задач:")
    for _ in range(3):
        task = queue.dequeue()
        if "error" not in task:
            print(f"    Задача {task['id']}: {task['type']} → обрабатывается")
            # Имитируем обработку
            queue.processed.append(task)
            queue.complete_task(task["id"], {"output": "done"})

    # Проверка масштабирования
    print("\n  Проверка необходимости масштабирования воркеров:")
    scale_decision = queue.should_scale_workers()
    print(f"    Текущих воркеров: {scale_decision['current_workers']}")
    print(f"    Оптимальное: {scale_decision['optimal_workers']}")
    print(f"    Действие: {scale_decision['action']}")

    # --- 1.4 Автоматическое масштабирование ---
    print("\n[1.4] Автоматическое масштабирование (Auto-Scaling)")
    print("-" * 50)

    class AutoScaler:
        """Автоматический масштабировщик на основе метрик."""

        def __init__(self):
            # Пороговые значения для масштабирования
            self.scale_up_threshold = 70.0    # % использования → масштабирование вверх
            self.scale_down_threshold = 30.0  # % использования → масштабирование вниз
            self.min_instances = 1
            self.max_instances = 10

            self.instances = 2  # Текущее количество экземпляров
            self.metrics_history = []

        def collect_metrics(self, cpu_usage: float, request_rate: float, queue_depth: int):
            """Собирает метрики для принятия решения."""
            metrics = {
                "timestamp": time.time(),
                "cpu_usage": cpu_usage,
                "request_rate": request_rate,
                "queue_depth": queue_depth,
                "instances": self.instances
            }
            self.metrics_history.append(metrics)
            return metrics

        def evaluate_and_scale(self) -> dict:
            """Оценивает метрики и масштабирует при необходимости."""
            if not self.metrics_history:
                return {"action": "none", "reason": "Нет метрик"}

            latest = self.metrics_history[-1]

            # Среднее использование CPU за последние 5 замеров
            recent_cpus = [m["cpu_usage"] for m in self.metrics_history[-5:]]
            avg_cpu = sum(recent_cpus) / len(recent_cpus)

            # Решение о масштабировании
            if avg_cpu > self.scale_up_threshold:
                old_instances = self.instances
                self.instances = min(self.instances + 1, self.max_instances)
                action = "scale_up" if self.instances > old_instances else "max_reached"
            elif avg_cpu < self.scale_down_threshold:
                old_instances = self.instances
                self.instances = max(self.instances - 1, self.min_instances)
                action = "scale_down" if self.instances < old_instances else "min_reached"
            else:
                action = "maintain"

            return {
                "action": action,
                "avg_cpu": round(avg_cpu, 1),
                "instances": self.instances,
                "request_rate": latest["request_rate"],
                "queue_depth": latest["queue_depth"]
            }

    autoscaler = AutoScaler()

    # Симуляция изменения нагрузки
    print("  Симуляция автоматического масштабирования:")
    metric_sequence = [
        (20, 10, 0),    # Низкая нагрузка
        (45, 25, 2),    # Средняя нагрузка
        (75, 50, 5),    # Высокая нагрузка
        (85, 60, 8),    # Очень высокая нагрузка
        (90, 70, 12),   # Критическая нагрузка
        (60, 40, 6),    # Нагрузка снижается
        (30, 20, 2),    # Низкая нагрузка
        (15, 10, 0)     # Минимальная нагрузка
    ]

    for cpu, rate, depth in metric_sequence:
        autoscaler.collect_metrics(cpu, rate, depth)
        result = autoscaler.evaluate_and_scale()
        action_symbol = {"scale_up": "▲", "scale_down": "▼", "maintain": "●"}.get(result["action"], "?")
        print(f"    CPU: {cpu}%, запросов/сек: {rate}, очередь: {depth} → "
              f"{action_symbol} {result['action']} (экземпляров: {result['instances']})")


# ============================================================
# Демо 2: Мониторинг
# ============================================================
def demo_monitoring():
    """Демонстрация систем мониторинга для агентов."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: МОНИТОРИНГ (Monitoring)")
    print("=" * 70)

    # --- 2.1 Отслеживание задержки (Latency) ---
    print("\n[2.1] Отслеживание задержки (Latency Tracking)")
    print("-" * 50)

    class LatencyMonitor:
        """Мониторинг задержки ответов агента."""

        def __init__(self):
            self.latencies = []
            self.percentiles = [50, 90, 95, 99]

        def record(self, latency_ms: float):
            """Записывает измерение задержки."""
            self.latencies.append(latency_ms)

        def get_statistics(self) -> dict:
            """Вычисляет статистику задержки."""
            if not self.latencies:
                return {"count": 0}

            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)

            stats = {
                "count": n,
                "min": round(sorted_latencies[0], 2),
                "max": round(sorted_latencies[-1], 2),
                "mean": round(sum(sorted_latencies) / n, 2),
                "percentiles": {}
            }

            for p in self.percentiles:
                index = int(n * p / 100)
                index = min(index, n - 1)
                stats["percentiles"][f"p{p}"] = round(sorted_latencies[index], 2)

            return stats

        def detect_anomalies(self, threshold_multiplier: float = 2.0) -> list:
            """Обнаруживает аномальные задержки."""
            if len(self.latencies) < 10:
                return []

            mean = sum(self.latencies) / len(self.latencies)
            std = math.sqrt(sum((x - mean) ** 2 for x in self.latencies) / len(self.latencies))

            anomalies = []
            for i, latency in enumerate(self.latencies):
                if latency > mean + threshold_multiplier * std:
                    anomalies.append({
                        "index": i,
                        "latency": round(latency, 2),
                        "z_score": round((latency - mean) / std, 2)
                    })

            return anomalies

    latency_monitor = LatencyMonitor()

    # Генерируем данные о задержке
    print("  Генерация данных о задержке...")
    for _ in range(50):
        # Нормальное распределение с выбросами
        latency = random.gauss(150, 30)
        if random.random() < 0.1:  # 10% выбросов
            latency *= random.uniform(3, 5)
        latency_monitor.record(max(latency, 10))

    stats = latency_monitor.get_statistics()
    print(f"\n  Статистика задержки:")
    print(f"    Количество запросов: {stats['count']}")
    print(f"    Минимум: {stats['min']} мс")
    print(f"    Максимум: {stats['max']} мс")
    print(f"    Среднее: {stats['mean']} мс")
    for p, value in stats['percentiles'].items():
        print(f"    {p}: {value} мс")

    # Обнаружение аномалий
    anomalies = latency_monitor.detect_anomalies(threshold_multiplier=2.5)
    print(f"\n  Обнаружено аномалий: {len(anomalies)}")
    for a in anomalies[:3]:
        print(f"    Запрос {a['index']}: {a['latency']} мс (z-score: {a['z_score']})")

    # --- 2.2 Частота успеха ---
    print("\n[2.2] Частота успеха (Success Rate)")
    print("-" * 50)

    class SuccessRateMonitor:
        """Мониторинг частоты успешных операций."""

        def __init__(self):
            self.results = []  # True = успех, False = неудача
            self.error_types = collections.Counter()

        def record(self, success: bool, error_type: str = None):
            """Записывает результат операции."""
            self.results.append(success)
            if not success and error_type:
                self.error_types[error_type] += 1

        def get_success_rate(self, window: int = None) -> dict:
            """Вычисляет частоту успеха."""
            data = self.results[-window:] if window else self.results

            if not data:
                return {"rate": 0, "count": 0}

            success_count = sum(data)
            total = len(data)
            rate = success_count / total * 100

            return {
                "rate": round(rate, 2),
                "successes": success_count,
                "failures": total - success_count,
                "total": total
            }

        def get_error_distribution(self) -> dict:
            """Возвращает распределение ошибок."""
            total_errors = sum(self.error_types.values())
            if total_errors == 0:
                return {}

            return {
                error_type: {
                    "count": count,
                    "percentage": round(count / total_errors * 100, 1)
                }
                for error_type, count in self.error_types.most_common()
            }

        def calculate_sla(self, target_rate: float = 99.9) -> dict:
            """Рассчитывает SLA (Service Level Agreement)."""
            overall = self.get_success_rate()
            return {
                "target_sla": target_rate,
                "actual_rate": overall["rate"],
                "sla_met": overall["rate"] >= target_rate,
                "downtime_minutes": round(
                    (100 - overall["rate"]) / 100 * 60, 2
                ) if overall["rate"] < 100 else 0
            }

    success_monitor = SuccessRateMonitor()

    # Генерируем данные
    print("  Генерация результатов операций...")
    error_types = ["timeout", "rate_limit", "invalid_input", "server_error", "model_error"]

    for _ in range(100):
        success = random.random() < 0.95  # 95% успеха
        error = random.choice(error_types) if not success else None
        success_monitor.record(success, error)

    # Общая частота успеха
    overall = success_monitor.get_success_rate()
    print(f"\n  Общая частота успеха: {overall['rate']}% ({overall['successes']}/{overall['total']})")

    # За последнее окно
    recent = success_monitor.get_success_rate(window=20)
    print(f"  За последние 20 операций: {recent['rate']}%")

    # Распределение ошибок
    errors = success_monitor.get_error_distribution()
    print(f"\n  Распределение ошибок:")
    for error_type, info in errors.items():
        print(f"    {error_type}: {info['count']} ({info['percentage']}%)")

    # SLA
    sla = success_monitor.calculate_sla(target_rate=99.0)
    print(f"\n  SLA:")
    print(f"    Целевой: {sla['target_sla']}%")
    print(f"    Фактический: {sla['actual_rate']}%")
    print(f"    Выполнен: {'Да' if sla['sla_met'] else 'Нет'}")
    print(f"    Простой: {sla['downtime_minutes']} минут")

    # --- 2.3 Использование токенов ---
    print("\n[2.3] Использование токенов (Token Usage)")
    print("-" * 50)

    class TokenUsageMonitor:
        """Мониторинг использования токенов."""

        def __init__(self):
            self.usage_records = []
            self.models = {}

        def record_usage(self, model: str, input_tokens: int, output_tokens: int,
                        cost_per_1k_input: float, cost_per_1k_output: float):
            """Записывает использование токенов."""
            input_cost = (input_tokens / 1000) * cost_per_1k_input
            output_cost = (output_tokens / 1000) * cost_per_1k_output
            total_cost = input_cost + output_cost

            record = {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": round(total_cost, 6)
            }
            self.usage_records.append(record)

            if model not in self.models:
                self.models[model] = {"tokens": 0, "cost": 0, "calls": 0}
            self.models[model]["tokens"] += record["total_tokens"]
            self.models[model]["cost"] += total_cost
            self.models[model]["calls"] += 1

        def get_summary(self) -> dict:
            """Возвращает сводку использования."""
            total_tokens = sum(r["total_tokens"] for r in self.usage_records)
            total_cost = sum(r["cost"] for r in self.usage_records)

            return {
                "total_calls": len(self.usage_records),
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "avg_tokens_per_call": round(total_tokens / max(len(self.usage_records), 1)),
                "by_model": {
                    model: {
                        "calls": info["calls"],
                        "tokens": info["tokens"],
                        "cost": round(info["cost"], 4)
                    }
                    for model, info in self.models.items()
                }
            }

        def get_usage_by_time(self, interval_seconds: int = 3600) -> list:
            """Группирует использование по временным интервалам."""
            # Для симуляции просто разбиваем на группы
            groups = []
            group_size = max(1, len(self.usage_records) // 5)

            for i in range(0, len(self.usage_records), group_size):
                group = self.usage_records[i:i + group_size]
                groups.append({
                    "period": f"Период {len(groups) + 1}",
                    "calls": len(group),
                    "tokens": sum(r["total_tokens"] for r in group),
                    "cost": round(sum(r["cost"] for r in group), 4)
                })

            return groups

    token_monitor = TokenUsageMonitor()

    # Генерируем данные использования
    print("  Генерация данных использования токенов...")
    models_config = [
        ("gpt-4", 0.03, 0.06),
        ("gpt-3.5-turbo", 0.0015, 0.002),
        ("claude-3", 0.015, 0.075)
    ]

    for _ in range(20):
        model, cost_in, cost_out = random.choice(models_config)
        input_tokens = random.randint(100, 2000)
        output_tokens = random.randint(50, 1000)
        token_monitor.record_usage(model, input_tokens, output_tokens, cost_in, cost_out)

    summary = token_monitor.get_summary()
    print(f"\n  Сводка использования:")
    print(f"    Всего вызовов: {summary['total_calls']}")
    print(f"    Всего токенов: {summary['total_tokens']:,}")
    print(f"    Общая стоимость: ${summary['total_cost']:.4f}")
    print(f"    Среднее токенов/вызов: {summary['avg_tokens_per_call']:,}")

    print(f"\n  По моделям:")
    for model, info in summary['by_model'].items():
        print(f"    {model}: {info['calls']} вызовов, {info['tokens']:,} токенов, ${info['cost']:.4f}")

    # Использование по времени
    time_groups = token_monitor.get_usage_by_time()
    print(f"\n  Использование по периодам:")
    for group in time_groups:
        print(f"    {group['period']}: {group['calls']} вызовов, {group['tokens']:,} токенов")

    # --- 2.4 Отслеживание стоимости ---
    print("\n[2.4] Отслеживание стоимости (Cost Tracking)")
    print("-" * 50)

    class CostTracker:
        """Отслеживание и контроль стоимости."""

        def __init__(self, daily_budget: float = 100.0):
            self.daily_budget = daily_budget
            self.costs = []
            self.alerts = []

        def add_cost(self, category: str, amount: float, description: str = ""):
            """Добавляет запись о стоимости."""
            self.costs.append({
                "category": category,
                "amount": amount,
                "description": description,
                "timestamp": time.time()
            })

            # Проверяем бюджет
            daily_total = self.get_daily_total()
            if daily_total > self.daily_budget * 0.8 and "budget_80" not in self.alerts:
                self.alerts.append("budget_80")
                print(f"    ⚠ ПРЕДУПРЕЖДЕНИЕ: потрачено {daily_total:.2f}/{self.daily_budget:.2f} (80%)")
            if daily_total > self.daily_budget:
                self.alerts.append("budget_exceeded")
                print(f"    ✗ ПРЕВЫШЕН БЮДЖЕТ: {daily_total:.2f} > {self.daily_budget:.2f}")

        def get_daily_total(self) -> float:
            """Возвращает общую сумму за день."""
            return sum(c["amount"] for c in self.costs)

        def get_by_category(self) -> dict:
            """Группирует расходы по категориям."""
            categories = collections.defaultdict(float)
            for cost in self.costs:
                categories[cost["category"]] += cost["amount"]
            return dict(categories)

        def get_cost_forecast(self, days_ahead: int = 7) -> dict:
            """Прогнозирует расходы на основе тренда."""
            if len(self.costs) < 2:
                return {"forecast": 0, "confidence": "low"}

            daily_total = self.get_daily_total()
            # Простой прогноз: линейная экстраполяция
            forecast = daily_total * days_ahead

            return {
                "current_daily": round(daily_total, 2),
                "forecast_days": days_ahead,
                "forecast_total": round(forecast, 2),
                "forecast_daily_avg": round(daily_total, 2),
                "within_budget": forecast <= self.daily_budget * days_ahead
            }

        def optimize_costs(self) -> list:
            """Предлагает оптимизацию расходов."""
            suggestions = []
            categories = self.get_by_category()

            if categories.get("api_calls", 0) > self.daily_budget * 0.5:
                suggestions.append({
                    "category": "api_calls",
                    "current": round(categories["api_calls"], 2),
                    "suggestion": "Рассмотреть кеширование повторяющихся запросов"
                })

            if categories.get("compute", 0) > self.daily_budget * 0.3:
                suggestions.append({
                    "category": "compute",
                    "current": round(categories["compute"], 2),
                    "suggestion": "Использовать более дешёвые модели для простых задач"
                })

            return suggestions

    cost_tracker = CostTracker(daily_budget=50.0)

    # Генерируем расходы
    print("  Генерация расходов...")
    expense_categories = [
        ("api_calls", 15.50, "GPT-4 inference"),
        ("compute", 8.25, "GPU usage"),
        ("storage", 2.10, "S3 storage"),
        ("api_calls", 12.30, "Claude inference"),
        ("monitoring", 3.40, "DataDog"),
        ("compute", 6.80, "EC2 instances"),
        ("api_calls", 9.75, "Embeddings API")
    ]

    for category, amount, desc in expense_categories:
        cost_tracker.add_cost(category, amount, desc)
        print(f"    +${amount:.2f} ({category}: {desc})")

    # Анализ расходов
    daily_total = cost_tracker.get_daily_total()
    print(f"\n  Итого за день: ${daily_total:.2f}")

    categories = cost_tracker.get_by_category()
    print(f"\n  По категориям:")
    for cat, amount in sorted(categories.items(), key=lambda x: -x[1]):
        pct = amount / daily_total * 100
        print(f"    {cat}: ${amount:.2f} ({pct:.1f}%)")

    # Прогноз
    forecast = cost_tracker.get_cost_forecast(days_ahead=7)
    print(f"\n  Прогноз на 7 дней:")
    print(f"    Текущий средний: ${forecast['current_daily']:.2f}/день")
    print(f"    Прогноз: ${forecast['forecast_total']:.2f}")
    print(f"    В рамках бюджета: {'Да' if forecast['within_budget'] else 'Нет'}")

    # Рекомендации
    suggestions = cost_tracker.optimize_costs()
    print(f"\n  Рекомендации по оптимизации:")
    for s in suggestions:
        print(f"    {s['category']} (${s['current']:.2f}): {s['suggestion']}")


# ============================================================
# Демо 3: Управление стоимостью
# ============================================================
def demo_cost_management():
    """Демонстрация стратегий управления стоимостью агентов."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: УПРАВЛЕНИЕ СТОИМОСТЬЮ (Cost Management)")
    print("=" * 70)

    # --- 3.1 Бюджетирование токенов ---
    print("\n[3.1] Бюджетирование токенов (Token Budgeting)")
    print("-" * 50)

    class TokenBudgetManager:
        """Менеджер бюджетов на токены."""

        def __init__(self, daily_budget: int = 1000000):
            self.daily_budget = daily_budget
            self.used_tokens = 0
            self.budget_by_type = {
                "chat": int(daily_budget * 0.5),      # 50% на чат
                "embedding": int(daily_budget * 0.2),   # 20% на эмбеддинги
                "completion": int(daily_budget * 0.3)   # 30% на completion
            }
            self.used_by_type = collections.defaultdict(int)

        def check_budget(self, request_type: str, tokens_needed: int) -> dict:
            """Проверяет, достаточно ли бюджета."""
            # Проверка общего бюджета
            if self.used_tokens + tokens_needed > self.daily_budget:
                remaining = max(0, self.daily_budget - self.used_tokens)
                return {
                    "allowed": False,
                    "reason": "Превышен дневной бюджет",
                    "remaining": remaining,
                    "needed": tokens_needed
                }

            # Проверка бюджета по типу
            type_budget = self.budget_by_type.get(request_type, 0)
            type_used = self.used_by_type[request_type]

            if type_used + tokens_needed > type_budget:
                return {
                    "allowed": False,
                    "reason": f"Превышен бюджет для типа '{request_type}'",
                    "type_remaining": max(0, type_budget - type_used),
                    "needed": tokens_needed
                }

            return {
                "allowed": True,
                "remaining_total": self.daily_budget - self.used_tokens - tokens_needed,
                "remaining_type": type_budget - type_used - tokens_needed
            }

        def consume(self, request_type: str, tokens: int):
            """Потребляет токены из бюджета."""
            self.used_tokens += tokens
            self.used_by_type[request_type] += tokens

        def get_usage_report(self) -> dict:
            """Генерирует отчёт об использовании."""
            return {
                "daily_budget": self.daily_budget,
                "used_total": self.used_tokens,
                "remaining": max(0, self.daily_budget - self.used_tokens),
                "utilization_pct": round(self.used_tokens / self.daily_budget * 100, 1),
                "by_type": {
                    t: {
                        "used": self.used_by_type[t],
                        "budget": self.budget_by_type.get(t, 0),
                        "utilization": round(
                            self.used_by_type[t] / max(self.budget_by_type.get(t, 1), 1) * 100, 1
                        )
                    }
                    for t in self.budget_by_type
                }
            }

    budget_mgr = TokenBudgetManager(daily_budget=500000)

    print(f"  Дневной бюджет: {budget_mgr.daily_budget:,} токенов")
    print(f"  Распределение: chat={budget_mgr.budget_by_type['chat']:,}, "
          f"embedding={budget_mgr.budget_by_type['embedding']:,}, "
          f"completion={budget_mgr.budget_by_type['completion']:,}")

    # Симуляция запросов
    print("\n  Обработка запросов:")
    requests = [
        ("chat", 15000), ("chat", 20000), ("embedding", 5000),
        ("completion", 25000), ("chat", 10000), ("embedding", 3000),
        ("chat", 8000), ("completion", 12000)
    ]

    for req_type, tokens in requests:
        check = budget_mgr.check_budget(req_type, tokens)
        if check["allowed"]:
            budget_mgr.consume(req_type, tokens)
            print(f"    {req_type}: {tokens:,} токенов ✓ (осталось: {check['remaining_total']:,})")
        else:
            print(f"    {req_type}: {tokens:,} токенов ✗ ({check['reason']})")

    # Отчёт
    report = budget_mgr.get_usage_report()
    print(f"\n  Отчёт об использовании:")
    print(f"    Использовано: {report['used_total']:,} / {report['daily_budget']:,} ({report['utilization_pct']}%)")
    for t, info in report['by_type'].items():
        print(f"    {t}: {info['used']:,} / {info['budget']:,} ({info['utilization']}%)")

    # --- 3.2 Маршрутизация моделей ---
    print("\n[3.2] Маршрутизация моделей (Model Routing)")
    print("-" * 50)

    class ModelRouter:
        """Умный маршрутизатор запросов к моделям."""

        def __init__(self):
            # Каталог моделей с характеристиками
            self.models = {
                "gpt-4": {
                    "quality_score": 95,
                    "cost_per_1k_tokens": 0.03,
                    "latency_ms": 2000,
                    "max_tokens": 8192,
                    "strengths": ["reasoning", "code", "analysis"]
                },
                "gpt-3.5-turbo": {
                    "quality_score": 80,
                    "cost_per_1k_tokens": 0.002,
                    "latency_ms": 500,
                    "max_tokens": 4096,
                    "strengths": ["chat", "simple_tasks", "classification"]
                },
                "claude-3-sonnet": {
                    "quality_score": 90,
                    "cost_per_1k_tokens": 0.015,
                    "latency_ms": 1500,
                    "max_tokens": 4096,
                    "strengths": ["reasoning", "writing", "analysis"]
                },
                "claude-3-haiku": {
                    "quality_score": 75,
                    "cost_per_1k_tokens": 0.001,
                    "latency_ms": 300,
                    "max_tokens": 2048,
                    "strengths": ["simple_chat", "classification", "extraction"]
                }
            }

            # Правила маршрутизации
            self.routing_rules = {
                "complexity_threshold": 0.7,  # Выше — сложная задача
                "quality_threshold": 85,       # Минимальное качество
                "max_cost_per_request": 0.05   # Максимальная стоимость запроса
            }

        def analyze_complexity(self, query: str) -> float:
            """Анализирует сложность запроса (0-1)."""
            complexity = 0.5  # Базовая сложность

            # Длина запроса
            if len(query) > 500:
                complexity += 0.1
            elif len(query) < 50:
                complexity -= 0.1

            # Наличие сложных терминов
            complex_terms = ["analyze", "implement", "optimize", "debug", "explain why",
                           "проанализируй", "реализуй", "оптимизируй"]
            for term in complex_terms:
                if term.lower() in query.lower():
                    complexity += 0.05

            # Наличие кода
            if "```" in query or "def " in query or "class " in query:
                complexity += 0.15

            return min(max(complexity, 0), 1)

        def select_model(self, query: str, priority: str = "balanced") -> dict:
            """Выбирает оптимальную модель для запроса."""
            complexity = self.analyze_complexity(query)

            candidates = []
            for name, specs in self.models.items():
                # Базовый скор
                score = specs["quality_score"]

                # Штраф за стоимость
                cost_penalty = specs["cost_per_1k_tokens"] * 1000

                # Штраф за задержку
                latency_penalty = specs["latency_ms"] / 1000

                # Учитываем приоритет
                if priority == "cost":
                    final_score = score - cost_penalty * 50
                elif priority == "speed":
                    final_score = score - latency_penalty * 10
                else:  # balanced
                    final_score = score - cost_penalty * 20 - latency_penalty * 5

                # Сложность запроса влияет на выбор
                if complexity > self.routing_rules["complexity_threshold"]:
                    if "reasoning" in specs["strengths"] or "code" in specs["strengths"]:
                        final_score += 10

                candidates.append((name, final_score, specs))

            # Сортируем по скору
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_model, best_score, best_specs = candidates[0]

            # Рассчитываем стоимость
            estimated_tokens = len(query.split()) * 2  # Грубая оценка
            estimated_cost = (estimated_tokens / 1000) * best_specs["cost_per_1k_tokens"]

            return {
                "selected_model": best_model,
                "complexity": round(complexity, 2),
                "score": round(best_score, 2),
                "estimated_cost": round(estimated_cost, 6),
                "estimated_latency": best_specs["latency_ms"],
                "alternatives": [
                    {"model": name, "score": round(score, 2)}
                    for name, score, _ in candidates[1:3]
                ]
            }

    router = ModelRouter()

    # Тестовые запросы
    queries = [
        ("Привет, как дела?", "balanced"),
        ("Проанализируй производительность этого кода и предложи оптимизации", "balanced"),
        ("Сгенерируй простой ответ на вопрос: что такое Python?", "cost"),
        ("Реализуй алгоритм быстрой сортировки с обработкой дубликатов", "quality"),
        ("Классифицируй этот текст по категориям", "speed")
    ]

    print("  Маршрутизация запросов:")
    for query, priority in queries:
        result = router.select_model(query, priority)
        print(f"\n  Запрос: \"{query[:50]}...\" (приоритет: {priority})")
        print(f"    Выбрана модель: {result['selected_model']}")
        print(f"    Сложность: {result['complexity']}")
        print(f"    Оценка: {result['score']}")
        print(f"    Стоимость: ${result['estimated_cost']:.6f}")
        if result['alternatives']:
            alts = ", ".join([f"{a['model']} ({a['score']})" for a in result['alternatives']])
            print(f"    Альтернативы: {alts}")

    # --- 3.3 Кеширование ---
    print("\n[3.3] Кеширование (Caching)")
    print("-" * 50)

    class ResponseCache:
        """Кеш ответов для снижения стоимости."""

        def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
            self.max_size = max_size
            self.ttl_seconds = ttl_seconds
            self.cache = {}
            self.stats = {"hits": 0, "misses": 0}

        def _make_key(self, query: str, model: str) -> str:
            """Генерирует ключ кеша."""
            # Нормализуем запрос
            normalized = re.sub(r'\s+', ' ', query.lower().strip())
            return hashlib.md5(f"{model}:{normalized}".encode()).hexdigest()

        def get(self, query: str, model: str) -> dict:
            """Получает ответ из кеша."""
            key = self._make_key(query, model)

            if key in self.cache:
                entry = self.cache[key]
                # Проверяем TTL
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    self.stats["hits"] += 1
                    return {
                        "cached": True,
                        "response": entry["response"],
                        "age_seconds": int(time.time() - entry["timestamp"])
                    }
                else:
                    del self.cache[key]

            self.stats["misses"] += 1
            return {"cached": False}

        def set(self, query: str, model: str, response: str):
            """Сохраняет ответ в кеш."""
            key = self._make_key(query, model)

            # Проверяем размер кеша
            if len(self.cache) >= self.max_size:
                # Удаляем самую старую запись
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]

            self.cache[key] = {
                "response": response,
                "model": model,
                "timestamp": time.time(),
                "query_hash": key
            }

        def get_stats(self) -> dict:
            """Возвращает статистику кеша."""
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(total, 1) * 100

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": round(hit_rate, 1)
            }

        def clear_expired(self) -> int:
            """Очищает просроченные записи."""
            current_time = time.time()
            expired = [
                key for key, entry in self.cache.items()
                if current_time - entry["timestamp"] > self.ttl_seconds
            ]
            for key in expired:
                del self.cache[key]
            return len(expired)

    cache = ResponseCache(max_size=5, ttl_seconds=3600)

    # Симуляция запросов
    print("  Симуляция запросов с кешированием:")
    test_queries = [
        ("Что такое Python?", "gpt-4"),
        ("Что такое Python?", "gpt-4"),  # Повторный запрос
        ("Объясни декораторы", "gpt-3.5-turbo"),
        ("Что такое Python?", "gpt-4"),  # Ещё раз
        ("Как работает GIL?", "gpt-4"),
        ("Что такое Python?", "gpt-3.5-turbo"),  # Другая модель
    ]

    for query, model in test_queries:
        # Проверяем кеш
        result = cache.get(query, model)
        if result["cached"]:
            print(f"    КЕШ: \"{query[:30]}...\" ({model}) - попадание (возраст: {result['age_seconds']}с)")
        else:
            # Имитируем генерацию ответа
            response = f"Ответ на: {query[:30]}..."
            cache.set(query, model, response)
            print(f"    ГЕНЕРАЦИЯ: \"{query[:30]}...\" ({model}) - промах")

    # Статистика кеша
    stats = cache.get_stats()
    print(f"\n  Статистика кеша:")
    print(f"    Размер: {stats['size']}/{stats['max_size']}")
    print(f"    Попадания: {stats['hits']}, Промахи: {stats['misses']}")
    print(f"    Процент попаданий: {stats['hit_rate']}%")

    # Очистка
    cleared = cache.clear_expired()
    print(f"    Очищено просроченных: {cleared}")

    # --- 3.4 Оптимизация стоимости ---
    print("\n[3.4] Стратегии оптимизации стоимости")
    print("-" * 50)

    class CostOptimizer:
        """Стратегии оптимизации стоимости."""

        def __init__(self):
            self.strategies = {
                "model_downgrade": {
                    "description": "Использование менее дорогих моделей для простых задач",
                    "potential_savings": 0.6  # 60% экономия
                },
                "response_caching": {
                    "description": "Кеширование повторяющихся ответов",
                    "potential_savings": 0.3  # 30% экономия
                },
                "prompt_optimization": {
                    "description": "Оптимизация промптов для уменьшения токенов",
                    "potential_savings": 0.2  # 20% экономия
                },
                "batch_processing": {
                    "description": "Пакетная обработка для снижения накладных расходов",
                    "potential_savings": 0.15  # 15% экономия
                }
            }

        def analyze_query(self, query: str) -> dict:
            """Анализирует запрос для определения оптимальной стратегии."""
            # Простой анализ сложности
            complexity = 0.5
            if len(query) < 100:
                complexity -= 0.2
            if len(query) > 500:
                complexity += 0.2
            if "код" in query.lower() or "code" in query.lower():
                complexity += 0.1

            # Определяем подходящие стратегии
            applicable_strategies = []
            if complexity < 0.4:
                applicable_strategies.append("model_downgrade")
            applicable_strategies.append("response_caching")
            if len(query) > 200:
                applicable_strategies.append("prompt_optimization")

            return {
                "complexity": round(complexity, 2),
                "applicable_strategies": applicable_strategies,
                "estimated_savings": round(
                    sum(self.strategies[s]["potential_savings"]
                        for s in applicable_strategies) / len(applicable_strategies) * 100, 1
                )
            }

        def calculate_savings(self, original_cost: float, strategies: list) -> dict:
            """Рассчитывает экономию от применения стратегий."""
            savings = 0
            applied = []

            for strategy in strategies:
                if strategy in self.strategies:
                    saving = original_cost * self.strategies[strategy]["potential_savings"]
                    savings += saving
                    applied.append({
                        "strategy": strategy,
                        "description": self.strategies[strategy]["description"],
                        "savings": round(saving, 4)
                    })

            return {
                "original_cost": round(original_cost, 4),
                "total_savings": round(savings, 4),
                "optimized_cost": round(original_cost - savings, 4),
                "savings_percentage": round(savings / max(original_cost, 0.001) * 100, 1),
                "applied_strategies": applied
            }

    optimizer = CostOptimizer()

    # Анализ запросов
    print("  Анализ запросов для оптимизации:")
    test_queries_for_opt = [
        "Привет!",
        "Объясни принцип работы трансформеров с примерами кода",
        "Какой сегодня день?",
        "Проанализируй производительность базы данных и предложи индексы"
    ]

    for query in test_queries_for_opt:
        analysis = optimizer.analyze_query(query)
        print(f"\n  Запрос: \"{query[:50]}...\"" if len(query) > 50 else f"\n  Запрос: \"{query}\"")
        print(f"    Сложность: {analysis['complexity']}")
        print(f"    Стратегии: {analysis['applicable_strategies']}")
        print(f"    Потенциальная экономия: {analysis['estimated_savings']}%")

    # Расчёт экономии
    print("\n  Расчёт экономии:")
    original_cost = 50.0  # $50 в день
    strategies = ["model_downgrade", "response_caching", "prompt_optimization"]

    result = optimizer.calculate_savings(original_cost, strategies)
    print(f"    Оригинальная стоимость: ${result['original_cost']:.2f}")
    print(f"    Экономия: ${result['total_savings']:.2f} ({result['savings_percentage']}%)")
    print(f"    Оптимизированная стоимость: ${result['optimized_cost']:.2f}")
    print(f"\n  Применённые стратегии:")
    for s in result['applied_strategies']:
        print(f"    - {s['description']}: ${s['savings']:.4f}")


# ============================================================
# Демо 4: Продакшн- concerns
# ============================================================
def demo_production_concerns():
    """Демонстрация продакшн-аспектов: идемпотентность, retry, circuit breaker."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: ПРОДАКШН-АСПЕКТЫ (Production Concerns)")
    print("=" * 70)

    # --- 4.1 Идемпотентность ---
    print("\n[4.1] Идемпотентность (Idempotency)")
    print("-" * 50)

    class IdempotencyManager:
        """Менеджер идемпотентных операций."""

        def __init__(self):
            # Хранилище результатов операций
            self.results = {}
            # Счётчики попыток
            self.attempt_counts = collections.Counter()

        def execute_operation(self, operation_id: str, operation_fn) -> dict:
            """Выполняет операцию с гарантией идемпотентности."""
            self.attempt_counts[operation_id] += 1
            attempt = self.attempt_counts[operation_id]

            # Проверяем, была ли операция уже выполнена
            if operation_id in self.results:
                return {
                    "idempotent": True,
                    "result": self.results[operation_id],
                    "attempt": attempt,
                    "message": "Операция уже выполнена, возвращён кешированный результат"
                }

            # Выполняем операцию
            try:
                result = operation_fn()
                self.results[operation_id] = result
                return {
                    "idempotent": False,
                    "result": result,
                    "attempt": attempt,
                    "message": "Операция выполнена впервые"
                }
            except Exception as e:
                return {
                    "idempotent": False,
                    "error": str(e),
                    "attempt": attempt,
                    "message": "Операция завершилась ошибкой"
                }

        def get_stats(self) -> dict:
            """Возвращает статистику."""
            return {
                "total_operations": len(self.results),
                "total_attempts": sum(self.attempt_counts.values()),
                "saved_executions": sum(self.attempt_counts.values()) - len(self.results)
            }

    idempotency_mgr = IdempotencyManager()

    # Симуляция операций с повторными попытками
    print("  Симуляция идемпотентных операций:")

    # Операция: отправка email (должна быть идемпотентной)
    call_count = 0
    def send_email():
        nonlocal call_count
        call_count += 1
        return f"Email отправлен (попытка {call_count})"

    operation_id = "email_user123_report"

    # Первый вызов
    result1 = idempotency_mgr.execute_operation(operation_id, send_email)
    print(f"\n  Вызов 1: {result1['message']}")
    print(f"    Результат: {result1['result']}")

    # Повторный вызов (идемпотентный)
    result2 = idempotency_mgr.execute_operation(operation_id, send_email)
    print(f"\n  Вызов 2: {result2['message']}")
    print(f"    Результат: {result2['result']}")
    print(f"    Идемпотентный: {result2['idempotent']}")

    # Ещё вызов
    result3 = idempotency_mgr.execute_operation(operation_id, send_email)
    print(f"\n  Вызов 3: {result3['message']}")

    # Другая операция
    def process_payment():
        return {"status": "success", "transaction_id": "TXN-001"}

    result4 = idempotency_mgr.execute_operation("payment_order456", process_payment)
    print(f"\n  Оплата: {result4['message']}")
    print(f"    Результат: {result4['result']}")

    stats = idempotency_mgr.get_stats()
    print(f"\n  Статистика:")
    print(f"    Всего операций: {stats['total_operations']}")
    print(f"    Всего попыток: {stats['total_attempts']}")
    print(f"    Сэкономлено выполнений: {stats['saved_executions']}")

    # --- 4.2 Повторные попытки (Retry) ---
    print("\n[4.2] Повторные попытки (Retry)")
    print("-" * 50)

    class RetryManager:
        """Менеджер повторных попыток с экспоненциальной задержкой."""

        def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                     max_delay: float = 30.0, backoff_factor: float = 2.0):
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            self.backoff_factor = backoff_factor
            self.retry_history = []

        def calculate_delays(self) -> list:
            """Рассчитывает задержки для каждой попытки."""
            delays = []
            for attempt in range(self.max_retries):
                delay = self.base_delay * (self.backoff_factor ** attempt)
                delay = min(delay, self.max_delay)
                delays.append(round(delay, 2))
            return delays

        def execute_with_retry(self, operation_fn, retryable_errors: tuple = (Exception,)) -> dict:
            """Выполняет операцию с повторными попытками."""
            delays = self.calculate_delays()
            last_error = None

            for attempt in range(self.max_retries + 1):
                try:
                    result = operation_fn()
                    self.retry_history.append({
                        "attempt": attempt + 1,
                        "success": True,
                        "total_time": sum(delays[:attempt]) if attempt > 0 else 0
                    })
                    return {
                        "success": True,
                        "result": result,
                        "attempts": attempt + 1,
                        "total_delay": sum(delays[:attempt])
                    }
                except retryable_errors as e:
                    last_error = e
                    if attempt < self.max_retries:
                        delay = delays[attempt]
                        print(f"    Попытка {attempt + 1}: ошибка {type(e).__name__}, "
                              f"повтор через {delay}с...")
                        time.sleep(min(delay, 0.1))  # Ускоряем для демо
                    else:
                        self.retry_history.append({
                            "attempt": attempt + 1,
                            "success": False,
                            "error": str(e)
                        })

            return {
                "success": False,
                "error": str(last_error),
                "attempts": self.max_retries + 1,
                "total_delay": sum(delays)
            }

    retry_mgr = RetryManager(max_retries=3, base_delay=0.5)

    # Симуляция операции, которая иногда падает
    print("  Симуляция с retry (операция падает 2 раза, потомucceeds):")
    fail_count = 0

    def unreliable_operation():
        nonlocal fail_count
        fail_count += 1
        if fail_count <= 2:
            raise ConnectionError(f"Сервис недоступен (попытка {fail_count})")
        return {"status": "success", "data": "result"}

    result = retry_mgr.execute_with_retry(unreliable_operation)
    print(f"\n  Результат: success={result['success']}")
    print(f"  Попыток: {result['attempts']}")
    print(f"  Общая задержка: {result['total_delay']}с")

    # Показываем расчёт задержек
    print(f"\n  Расчёт задержек (backoff_factor=2):")
    delays = retry_mgr.calculate_delays()
    for i, delay in enumerate(delays):
        print(f"    Попытка {i+1}: {delay}с")

    # --- 4.3 Таймауты ---
    print("\n[4.3] Таймауты (Timeouts)")
    print("-" * 50)

    class TimeoutManager:
        """Менеджер таймаутов для операций."""

        def __init__(self):
            self.timeout_configs = {
                "api_call": 30,           # 30 секунд на API вызов
                "model_inference": 60,    # 60 секунд на инференс
                "file_operation": 10,     # 10 секунд на файловую операцию
                "database_query": 5       # 5 секунд на запрос к БД
            }
            self.timeout_events = []

        def execute_with_timeout(self, operation_type: str, operation_fn,
                                custom_timeout: float = None) -> dict:
            """Выполняет операцию с таймаутом."""
            timeout = custom_timeout or self.timeout_configs.get(operation_type, 30)
            start_time = time.time()

            # В реальном коде здесь был бы threading.Timer или asyncio.wait_for
            # Для симуляции просто проверяем условие
            try:
                result = operation_fn()
                elapsed = time.time() - start_time

                if elapsed > timeout:
                    self.timeout_events.append({
                        "type": operation_type,
                        "timeout": timeout,
                        "elapsed": round(elapsed, 2),
                        "status": "timeout"
                    })
                    return {
                        "success": False,
                        "error": "Timeout",
                        "timeout": timeout,
                        "elapsed": round(elapsed, 2)
                    }

                self.timeout_events.append({
                    "type": operation_type,
                    "timeout": timeout,
                    "elapsed": round(elapsed, 2),
                    "status": "success"
                })

                return {
                    "success": True,
                    "result": result,
                    "elapsed": round(elapsed, 2),
                    "timeout": timeout
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        def get_timeout_stats(self) -> dict:
            """Возвращает статистику таймаутов."""
            if not self.timeout_events:
                return {"total": 0}

            timeouts = [e for e in self.timeout_events if e["status"] == "timeout"]
            return {
                "total": len(self.timeout_events),
                "timeouts": len(timeouts),
                "timeout_rate": round(len(timeouts) / len(self.timeout_events) * 100, 1),
                "avg_elapsed": round(
                    sum(e["elapsed"] for e in self.timeout_events) / len(self.timeout_events), 2
                )
            }

    timeout_mgr = TimeoutManager()

    # Тестирование таймаутов
    print("  Конфигурация таймаутов:")
    for op_type, timeout in timeout_mgr.timeout_configs.items():
        print(f"    {op_type}: {timeout}с")

    print("\n  Выполнение операций:")
    operations = [
        ("api_call", lambda: {"data": "response"}),
        ("model_inference", lambda: {"text": "generated"}),
        ("database_query", lambda: {"rows": 100}),
        ("file_operation", lambda: {"content": "file_data"})
    ]

    for op_type, fn in operations:
        result = timeout_mgr.execute_with_timeout(op_type, fn)
        status = "✓ УСПЕХ" if result["success"] else f"✗ {result.get('error', 'ERROR')}"
        print(f"    {op_type}: {status} ({result['elapsed']}с / {result.get('timeout', 'N/A')}с)")

    stats = timeout_mgr.get_timeout_stats()
    print(f"\n  Статистика таймаутов:")
    print(f"    Всего операций: {stats['total']}")
    print(f"    Таймаутов: {stats.get('timeouts', 0)}")
    print(f"    Среднее время: {stats.get('avg_elapsed', 0)}с")

    # --- 4.4 Circuit Breaker ---
    print("\n[4.4] Circuit Breaker (Прерыватель цепи)")
    print("-" * 50)

    class CircuitBreaker:
        """Паттерн Circuit Breaker для отказоустойчивости."""

        # Состояния: closed (работает), open (заблокирован), half-open (проверка)
        STATES = {"closed": "CLOSED", "open": "OPEN", "half_open": "HALF-OPEN"}

        def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                     half_open_max: int = 3):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.half_open_max = half_open_max

            self.state = "closed"
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.half_open_attempts = 0
            self.history = []

        def can_execute(self) -> dict:
            """Проверяет, можно ли выполнять операции."""
            if self.state == "closed":
                return {"allowed": True, "state": self.STATES[self.state]}

            elif self.state == "open":
                # Проверяем, прошло ли время восстановления
                if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    self.half_open_attempts = 0
                    return {"allowed": True, "state": self.STATES[self.state],
                           "message": "Переход в half-open"}
                return {"allowed": False, "state": self.STATES[self.state],
                       "retry_after": round(self.recovery_timeout - (time.time() - self.last_failure_time), 1)}

            else:  # half_open
                if self.half_open_attempts < self.half_open_max:
                    return {"allowed": True, "state": self.STATES[self.state]}
                return {"allowed": False, "state": self.STATES[self.state]}

        def record_success(self):
            """Записывает успешное выполнение."""
            self.history.append({"success": True, "state": self.state, "timestamp": time.time()})

            if self.state == "half_open":
                self.success_count += 1
                if self.success_count >= self.half_open_max:
                    self.state = "closed"
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == "closed":
                self.failure_count = max(0, self.failure_count - 1)  # Уменьшаем счётчик

        def record_failure(self):
            """Записывает неудачное выполнение."""
            self.history.append({"success": False, "state": self.state, "timestamp": time.time()})
            self.last_failure_time = time.time()

            if self.state == "half_open":
                self.state = "open"
                self.half_open_attempts = 0
            elif self.state == "closed":
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

        def get_status(self) -> dict:
            """Возвращает текущий статус."""
            return {
                "state": self.STATES[self.state],
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls": len(self.history),
                "recent_failures": sum(1 for h in self.history[-10:] if not h["success"])
            }

    # Симуляция Circuit Breaker
    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    print("  Симуляция Circuit Breaker:")
    print(f"    Порог ошибок: {circuit.failure_threshold}")
    print(f"    Время восстановления: {circuit.recovery_timeout}с")

    # Нормальная работа
    print("\n  Нормальная работа (5 успешных вызовов):")
    for i in range(5):
        check = circuit.can_execute()
        if check["allowed"]:
            circuit.record_success()
            print(f"    Вызов {i+1}: ✓ (состояние: {circuit.STATES[circuit.state]})")

    # Генерируем ошибки
    print("\n  Генерация ошибок:")
    for i in range(4):
        check = circuit.can_execute()
        if check["allowed"]:
            circuit.record_failure()
            print(f"    Вызов {i+1}: ✗ ОШИБКА (состояние: {circuit.STATES[circuit.state]}, "
                  f"ошибок: {circuit.failure_count})")

    # Попытка вызова когда circuit open
    print("\n  Попытка вызова когда circuit breaker открыт:")
    check = circuit.can_execute()
    print(f"    Разрешено: {check['allowed']}")
    print(f"    Состояние: {check['state']}")
    if not check["allowed"]:
        print(f"    Повтор через: {check.get('retry_after', 'N/A')}с")

    # Ждём и проверяем переход в half-open
    print("\n  Ожидание восстановления (2с)...")
    time.sleep(2.1)

    check = circuit.can_execute()
    print(f"    После ожидания:")
    print(f"    Разрешено: {check['allowed']}")
    print(f"    Состояние: {check.get('state', 'N/A')}")

    if check["allowed"] and check.get("state") == "HALF-OPEN":
        # Успешные вызовы в half-open
        print("\n  Успешные вызовы в half-open:")
        for i in range(3):
            check = circuit.can_execute()
            if check["allowed"]:
                circuit.record_success()
                print(f"    Вызов {i+1}: ✓ (состояние: {circuit.STATES[circuit.state]})")

    # Финальный статус
    print(f"\n  Финальный статус:")
    status = circuit.get_status()
    for key, value in status.items():
        print(f"    {key}: {value}")


# ============================================================
# Главная функция
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Модуль 166: Agent Deployment")
    print("Масштабирование, мониторинг, управление стоимостью")
    print("=" * 70)

    demo_scaling_patterns()
    demo_monitoring()
    demo_cost_management()
    demo_production_concerns()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены!")
    print("=" * 70)