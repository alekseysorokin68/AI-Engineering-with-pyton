"""200 — Production Multi-Agent: развёртывание, мониторинг, отладка

Темы:
  1. Deployment Patterns — service mesh, message broker, event bus
  2. Monitoring Multi-Agent — метрики агентов, паттерны коммуникации, узкие места
  3. Debugging Agents — трассировка сообщений, инспекция состояния, воспроизведение
  4. Operations — версионирование, A/B тестирование, откат, планирование ёмкости

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
# Демо 1: Deployment Patterns — паттерны развёртывания
# ============================================================

def demo_deployment_patterns():
    """Service mesh, message broker, event bus для мультиагентных систем."""
    print("=" * 70)
    print("ДЕМО 1: Deployment Patterns — паттерны развёртывания")
    print("=" * 70)

    # --- 1.1 Service Mesh ---
    print("\n--- 1.1 Service Mesh — сервисная сетка ---")
    print("Агенты общаются через прокси (sidecar), а не напрямую\n")

    class ServiceMesh:
        """Сервисная сетка: маршрутизация, балансировка, безопасность."""

        def __init__(self):
            self.services = {}
            self.routes = []
            self.policies = {}
            self.request_log = []

        def register(self, service_name, endpoints):
            """Зарегистрировать сервис с эндпоинтами."""
            self.services[service_name] = {
                "endpoints": endpoints,
                "healthy": [True] * len(endpoints),
                "requests": 0,
            }

        def add_route(self, path, target_service, weight=100):
            """Добавить маршрут."""
            self.routes.append({
                "path": path,
                "target": target_service,
                "weight": weight,
            })

        def route_request(self, path, payload):
            """Маршрутизировать запрос через service mesh."""
            # Находим подходящий маршрут
            for route in self.routes:
                if route["path"] == path:
                    service = self.services[route["target"]]
                    # Балансировка: round-robin по здоровым эндпоинтам
                    healthy_endpoints = [
                        ep for ep, h in zip(service["endpoints"], service["healthy"]) if h
                    ]
                    if healthy_endpoints:
                        ep = healthy_endpoints[service["requests"] % len(healthy_endpoints)]
                        service["requests"] += 1
                        self.request_log.append({
                            "path": path,
                            "target": route["target"],
                            "endpoint": ep,
                            "payload": payload,
                        })
                        return {"status": "ok", "endpoint": ep, "service": route["target"]}

            return {"status": "error", "message": "Маршрут не найден"}

        def health_check(self, service_name, endpoint_idx, is_healthy):
            """Проверка здоровья эндпоинта."""
            if service_name in self.services:
                self.services[service_name]["healthy"][endpoint_idx] = is_healthy

        def stats(self):
            """Статистика service mesh."""
            return {
                "services": len(self.services),
                "routes": len(self.routes),
                "total_requests": len(self.request_log),
                "per_service": {
                    name: info["requests"]
                    for name, info in self.services.items()
                },
            }

    mesh = ServiceMesh()

    # Регистрируем сервисы (агенты)
    mesh.register("agent-inference", ["10.0.1.1:8080", "10.0.1.2:8080", "10.0.1.3:8080"])
    mesh.register("agent-preprocess", ["10.0.2.1:8080", "10.0.2.2:8080"])
    mesh.register("agent-postprocess", ["10.0.3.1:8080"])

    # Маршруты
    mesh.add_route("/predict", "agent-inference")
    mesh.add_route("/preprocess", "agent-preprocess")
    mesh.add_route("/postprocess", "agent-postprocess")

    print("Зарегистрированные сервисы:")
    for name, info in mesh.services.items():
        print(f"  {name}: {info['endpoints']}")

    # Отправляем запросы
    print("\nМаршрутизация запросов:")
    requests = [
        ("/predict", {"input": "image.jpg"}),
        ("/preprocess", {"data": [1, 2, 3]}),
        ("/predict", {"input": "text.txt"}),
        ("/postprocess", {"result": "ok"}),
        ("/predict", {"input": "audio.wav"}),
    ]

    for path, payload in requests:
        result = mesh.route_request(path, payload)
        print(f"  {path} -> {result['service']} ({result.get('endpoint', 'N/A')})")

    # Симулируем падение эндпоинта
    print("\nЭндпоинт 10.0.1.2:8080 упал!")
    mesh.health_check("agent-inference", 1, False)

    # Запросы идут только на здоровые
    result = mesh.route_request("/predict", {"input": "new.jpg"})
    print(f"  /predict -> {result['endpoint']} (только healthy эндпоинты)")

    stats = mesh.stats()
    print(f"\nСтатистика: {json.dumps(stats, indent=2)}")

    # --- 1.2 Message Broker ---
    print("\n--- 1.2 Message Broker — брокер сообщений ---")
    print("Централизованная доставка сообщений между агентами\n")

    class MessageBroker:
        """Брокер сообщений с подписками и持久化."""

        def __init__(self):
            self.topics = collections.defaultdict(list)
            self.subscriptions = collections.defaultdict(list)
            self.message_log = []
            self.dlq = []  # dead letter queue

        def publish(self, topic, message, priority="normal"):
            """Опубликовать сообщение."""
            msg = {
                "topic": topic,
                "payload": message,
                "priority": priority,
                "timestamp": len(self.message_log),
            }
            self.topics[topic].append(msg)
            self.message_log.append(msg)

            # Уведомляем подписчиков
            delivered = 0
            for subscriber in self.subscriptions[topic]:
                try:
                    subscriber(msg)
                    delivered += 1
                except Exception as e:
                    self.dlq.append({"message": msg, "error": str(e)})

            return delivered

        def subscribe(self, topic, callback):
            """Подписаться на топик."""
            self.subscriptions[topic].append(callback)

        def consume(self, topic, consumer_id):
            """Потребить сообщение из топика."""
            if self.topics[topic]:
                msg = self.topics[topic].pop(0)
                return msg
            return None

        def stats(self):
            return {
                "topics": len(self.topics),
                "total_messages": len(self.message_log),
                "pending": {t: len(q) for t, q in self.topics.items()},
                "subscribers": {t: len(s) for t, s in self.subscriptions.items()},
                "dlq_size": len(self.dlq),
            }

    broker = MessageBroker()

    # Подписчики
    received_by_agent = collections.defaultdict(list)

    def agent_handler(agent_name):
        """Создать обработчик для агента."""
        def handler(msg):
            received_by_agent[agent_name].append(msg["payload"])
        return handler

    broker.subscribe("tasks", agent_handler("Agent-A"))
    broker.subscribe("tasks", agent_handler("Agent-B"))
    broker.subscribe("results", agent_handler("Monitor"))
    broker.subscribe("alerts", agent_handler("AlertManager"))

    # Публикуем сообщения
    print("Публикация сообщений:")
    messages = [
        ("tasks", "Обработать изображение 1", "high"),
        ("tasks", "Обработать изображение 2", "normal"),
        ("tasks", "Обработать изображение 3", "normal"),
        ("results", "Результат: классификация=кошка", "normal"),
        ("alerts", "Высокая нагрузка на GPU-0", "high"),
    ]

    for topic, payload, priority in messages:
        delivered = broker.publish(topic, payload, priority)
        print(f"  [{priority}] {topic}: '{payload}' (доставлено: {delivered})")

    print("\nПотребители (round-robin):")
    for agent, msgs in received_by_agent.items():
        print(f"  {agent}: {len(msgs)} сообщений")
        for msg in msgs:
            print(f"    - {msg}")

    stats = broker.stats()
    print(f"\nСтатистика брокера:")
    print(f"  Топиков: {stats['topics']}")
    print(f"  Всего сообщений: {stats['total_messages']}")
    print(f"  Ожидают: {stats['pending']}")
    print(f"  Dead letter: {stats['dlq_size']}")

    # --- 1.3 Event Bus ---
    print("\n--- 1.3 Event Bus — шина событий ---")
    print("События публикуются и потребляются асинхронно\n")

    class EventBus:
        """Шина событий с поддержкой фильтрации и шаблонов."""

        def __init__(self):
            self.handlers = collections.defaultdict(list)
            self.event_history = []

        def on(self, event_pattern, handler):
            """Зарегистрировать обработчик на шаблон события."""
            self.handlers[event_pattern].append(handler)

        def emit(self, event_type, data):
            """Отправить событие."""
            event = {
                "type": event_type,
                "data": data,
                "id": len(self.event_history),
            }
            self.event_history.append(event)

            # Ищем подходящие обработчики
            triggered = []
            for pattern, handlers in self.handlers.items():
                # Простое сопоставление шаблонов (поддержка * и ?)
                if self._match(pattern, event_type):
                    for handler in handlers:
                        handler(event)
                        triggered.append(pattern)

            return triggered

        def _match(self, pattern, text):
            """Сопоставление шаблонов с поддержкой * и ?."""
            # Конвертируем паттерн в regex
            regex = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
            return bool(re.fullmatch(regex, text))

        def history(self, event_type=None, limit=10):
            """История событий с фильтрацией."""
            if event_type:
                filtered = [e for e in self.event_history if e["type"] == event_type]
            else:
                filtered = self.event_history
            return filtered[-limit:]

    bus = EventBus()

    # Регистрируем обработчики
    event_log = []

    def log_handler(event):
        event_log.append(f"[{event['type']}] {event['data']}")

    def alert_handler(event):
        if event["data"].get("severity") == "critical":
            event_log.append(f"КРИТИЧЕСКОЕ: {event['data']}")

    bus.on("agent.*", log_handler)
    bus.on("system.*", log_handler)
    bus.on("system.alert.*", alert_handler)

    # Генерируем события
    print("Генерация событий:")
    events_to_emit = [
        ("agent.started", {"agent_id": "A0", "role": "worker"}),
        ("agent.task.completed", {"task_id": "T1", "duration_ms": 150}),
        ("system.alert.high_load", {"severity": "warning", "cpu": 95}),
        ("system.alert.critical_error", {"severity": "critical", "error": "OOM"}),
        ("agent.stopped", {"agent_id": "A0", "reason": "shutdown"}),
    ]

    for event_type, data in events_to_emit:
        triggered = bus.emit(event_type, data)
        print(f"  {event_type}: обработчики={triggered}")

    print("\nЖурнал обработанных событий:")
    for entry in event_log:
        print(f"  {entry}")

    # --- 1.4 Rolling Deployment ---
    print("\n--- 1.4 Rolling Deployment — поэтапное развёртывание ---")
    print("Обновление агентов по частям без простоя\n")

    class RollingDeployment:
        """Поэтапное развёртывание новых версий агентов."""

        def __init__(self, total_instances, batch_size=2):
            self.total = total_instances
            self.batch_size = batch_size
            self.instances = [{"version": "v1.0", "status": "running"} for _ in range(total_instances)]
            self.deploy_log = []

        def deploy(self, new_version):
            """Поэтапное обновление."""
            self.deploy_log.append(f"Начало развёртывания {new_version}")

            for batch_start in range(0, self.total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.total)
                self.deploy_log.append(
                    f"Пакет {batch_start+1}-{batch_end}: обновление до {new_version}"
                )

                # Обновляем инстансы в пакете
                for i in range(batch_start, batch_end):
                    self.instances[i]["version"] = new_version
                    self.instances[i]["status"] = "updated"

                # Проверка здоровья после обновления пакета
                healthy = all(
                    inst["status"] == "updated"
                    for inst in self.instances[batch_start:batch_end]
                )

                if not healthy:
                    self.deploy_log.append(f"ОШИБКА в пакете {batch_start+1}-{batch_end}! Откат!")
                    return False

                self.deploy_log.append(f"Пакет {batch_start+1}-{batch_end} здоров")

            self.deploy_log.append(f"Развёртывание {new_version} завершено")
            return True

        def rollback(self, target_version):
            """Откат к предыдущей версии."""
            self.deploy_log.append(f"ОТКАТ к {target_version}")
            for inst in self.instances:
                inst["version"] = target_version
                inst["status"] = "running"
            self.deploy_log.append("Откат завершён")

    deployer = RollingDeployment(total_instances=6, batch_size=2)

    print(f"Начальное состояние: {deployer.total} инстансов v1.0")

    # Развёртывание v2.0
    success = deployer.deploy("v2.0")
    print(f"\nРазвёртывание v2.0: {'УСПЕХ' if success else 'НЕУДАЧА'}")

    for entry in deployer.deploy_log:
        print(f"  {entry}")

    # Текущее состояние
    print("\nСостояние инстансов:")
    for i, inst in enumerate(deployer.instances):
        print(f"  Instance-{i}: {inst['version']} ({inst['status']})")


# ============================================================
# Демо 2: Monitoring Multi-Agent — мониторинг
# ============================================================

def demo_monitoring():
    """Метрики агентов, паттерны коммуникации, обнаружение узких мест."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: Monitoring Multi-Agent — мониторинг")
    print("=" * 70)

    # --- 2.1 Метрики агентов ---
    print("\n--- 2.1 Метрики агентов ---")
    print("Сбор и анализ ключевых показателей каждого агента\n")

    class AgentMetrics:
        """Сбор метрик для каждого агента."""

        def __init__(self):
            self.metrics = collections.defaultdict(lambda: {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "avg_latency_ms": 0,
                "memory_mb": 0,
                "cpu_percent": 0,
                "messages_sent": 0,
                "messages_received": 0,
            })

        def update(self, agent_id, **kwargs):
            """Обновить метрики агента."""
            for key, value in kwargs.items():
                if key in self.metrics[agent_id]:
                    if key == "avg_latency_ms":
                        # Скользящее среднее
                        old = self.metrics[agent_id][key]
                        count = self.metrics[agent_id]["tasks_completed"]
                        self.metrics[agent_id][key] = (old * count + value) / (count + 1)
                    else:
                        self.metrics[agent_id][key] = value

        def record_task(self, agent_id, success=True, latency_ms=0):
            """Записать выполнение задачи."""
            if success:
                self.metrics[agent_id]["tasks_completed"] += 1
            else:
                self.metrics[agent_id]["tasks_failed"] += 1

            self.update(agent_id, avg_latency_ms=latency_ms)

        def agent_score(self, agent_id):
            """Комплексная оценка агента (0-100)."""
            m = self.metrics[agent_id]
            total = m["tasks_completed"] + m["tasks_failed"]
            if total == 0:
                return 0

            success_rate = m["tasks_completed"] / total * 100
            latency_score = max(0, 100 - m["avg_latency_ms"])
            resource_score = max(0, 100 - m["cpu_percent"])

            return round(success_rate * 0.5 + latency_score * 0.3 + resource_score * 0.2, 1)

        def report(self):
            """Полный отчёт по всем агентам."""
            report = {}
            for agent_id, m in self.metrics.items():
                total = m["tasks_completed"] + m["tasks_failed"]
                report[agent_id] = {
                    **m,
                    "total_tasks": total,
                    "success_rate": f"{m['tasks_completed'] / max(1, total) * 100:.1f}%",
                    "score": self.agent_score(agent_id),
                }
            return report

    metrics = AgentMetrics()

    # Имитация работы агентов
    agents_data = {
        "Agent-0": {"tasks": 15, "failures": 1, "latency": 45},
        "Agent-1": {"tasks": 12, "failures": 3, "latency": 120},
        "Agent-2": {"tasks": 18, "failures": 0, "latency": 30},
        "Agent-3": {"tasks": 8, "failures": 2, "latency": 80},
    }

    for agent_id, data in agents_data.items():
        for _ in range(data["tasks"] - data["failures"]):
            metrics.record_task(agent_id, success=True,
                                latency_ms=data["latency"] * random.uniform(0.8, 1.2))
        for _ in range(data["failures"]):
            metrics.record_task(agent_id, success=False)

        # Ресурсы
        metrics.update(agent_id,
                       cpu_percent=random.uniform(20, 90),
                       memory_mb=random.uniform(512, 2048),
                       messages_sent=random.randint(10, 100),
                       messages_received=random.randint(10, 100))

    report = metrics.report()
    print("Отчёт по агентам:")
    for agent_id, stats in report.items():
        print(f"\n  {agent_id}:")
        print(f"    Задач: {stats['total_tasks']} (успешных: {stats['success_rate']})")
        print(f"    Средняя задержка: {stats['avg_latency_ms']:.1f} мс")
        print(f"    CPU: {stats['cpu_percent']:.1f}%, RAM: {stats['memory_mb']:.0f} МБ")
        print(f"    Оценка: {stats['score']}/100")

    # --- 2.2 Паттерны коммуникации ---
    print("\n--- 2.2 Паттерны коммуникации ---")
    print("Анализ графа взаимодействий между агентами\n")

    class CommunicationAnalyzer:
        """Анализатор графа коммуникаций между агентами."""

        def __init__(self):
            self.edges = collections.Counter()  # (from, to) -> count
            self.node_degree = collections.Counter()

        def record_communication(self, sender, receiver):
            """Записать коммуникацию."""
            self.edges[(sender, receiver)] += 1
            self.node_degree[sender] += 1
            self.node_degree[receiver] += 1

        def most_active_pairs(self, top_n=5):
            """Самые активные пары агентов."""
            return self.edges.most_common(top_n)

        def central_agents(self, top_n=3):
            """Самые центральные агенты (по degree centrality)."""
            return self.node_degree.most_common(top_n)

        def detect_clusters(self):
            """Обнаружение кластеров (грубо — через компоненты связности)."""
            # Строим граф
            graph = collections.defaultdict(set)
            for (a, b) in self.edges:
                graph[a].add(b)
                graph[b].add(a)

            # BFS для поиска компонент
            visited = set()
            clusters = []

            for node in graph:
                if node not in visited:
                    cluster = []
                    queue = [node]
                    while queue:
                        current = queue.pop(0)
                        if current not in visited:
                            visited.add(current)
                            cluster.append(current)
                            queue.extend(graph[current] - visited)
                    clusters.append(cluster)

            return clusters

        def stats(self):
            """Общая статистика."""
            return {
                "total_communications": sum(self.edges.values()),
                "unique_pairs": len(self.edges),
                "unique_agents": len(self.node_degree),
            }

    analyzer = CommunicationAnalyzer()

    # Имитация коммуникаций
    communications = [
        ("A0", "A1"), ("A0", "A2"), ("A1", "A0"), ("A1", "A3"),
        ("A2", "A0"), ("A2", "A3"), ("A3", "A1"), ("A3", "A4"),
        ("A4", "A3"), ("A4", "A5"), ("A5", "A4"), ("A5", "A6"),
        ("A0", "A1"), ("A0", "A2"), ("A1", "A0"), ("A2", "A3"),
        ("A3", "A4"), ("A4", "A5"), ("A5", "A6"), ("A6", "A5"),
        ("A0", "A1"), ("A3", "A0"), ("A4", "A3"), ("A6", "A5"),
    ]

    for sender, receiver in communications:
        analyzer.record_communication(sender, receiver)

    print("Самые активные пары:")
    for (a, b), count in analyzer.most_active_pairs(5):
        bar = "█" * count
        print(f"  {a} -> {b}: {count} сообщений {bar}")

    print("\nСамые центральные агенты:")
    for agent, degree in analyzer.central_agents(5):
        print(f"  {agent}: degree={degree}")

    clusters = analyzer.detect_clusters()
    print(f"\nОбнаружено кластеров: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Кластер {i}: {cluster}")

    stats = analyzer.stats()
    print(f"\nСтатистика: {stats}")

    # --- 2.3 Детектирование узких мест ---
    print("\n--- 2.3 Детектирование узких мест ---")
    print("Автоматическое обнаружение проблем в системе\n")

    class BottleneckDetector:
        """Детектор узких мест в мультиагентной системе."""

        def __init__(self):
            self.thresholds = {
                "latency_ms": 100,
                "cpu_percent": 85,
                "queue_depth": 50,
                "error_rate_percent": 5,
            }
            self.alerts = []

        def check(self, agent_id, metrics):
            """Проверить метрики агента на аномалии."""
            agent_alerts = []

            for metric, threshold in self.thresholds.items():
                value = metrics.get(metric, 0)
                if value > threshold:
                    severity = "critical" if value > threshold * 1.5 else "warning"
                    alert = {
                        "agent": agent_id,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": severity,
                    }
                    agent_alerts.append(alert)
                    self.alerts.append(alert)

            return agent_alerts

        def summary(self):
            """Сводка по alerts."""
            by_severity = collections.Counter(a["severity"] for a in self.alerts)
            by_agent = collections.Counter(a["agent"] for a in self.alerts)
            by_metric = collections.Counter(a["metric"] for a in self.alerts)

            return {
                "total": len(self.alerts),
                "by_severity": dict(by_severity),
                "by_agent": dict(by_agent),
                "by_metric": dict(by_metric),
            }

    detector = BottleneckDetector()

    # Проверяем агентов
    agents_to_check = {
        "Agent-0": {"latency_ms": 45, "cpu_percent": 60, "queue_depth": 10, "error_rate_percent": 2},
        "Agent-1": {"latency_ms": 150, "cpu_percent": 92, "queue_depth": 75, "error_rate_percent": 8},
        "Agent-2": {"latency_ms": 30, "cpu_percent": 45, "queue_depth": 5, "error_rate_percent": 1},
        "Agent-3": {"latency_ms": 120, "cpu_percent": 88, "queue_depth": 60, "error_rate_percent": 6},
    }

    print("Проверка агентов:")
    for agent_id, agent_metrics in agents_to_check.items():
        alerts = detector.check(agent_id, agent_metrics)
        if alerts:
            print(f"\n  {agent_id}:")
            for alert in alerts:
                print(f"    [{alert['severity'].upper()}] {alert['metric']}: "
                      f"{alert['value']} > порог {alert['threshold']}")
        else:
            print(f"  {agent_id}: норма")

    summary = detector.summary()
    print(f"\nСводка алертов:")
    print(f"  Всего: {summary['total']}")
    print(f"  По серьёзности: {summary['by_severity']}")
    print(f"  По агентам: {summary['by_agent']}")
    print(f"  По метрикам: {summary['by_metric']}")

    # --- 2.4 Дашборд метрик ---
    print("\n--- 2.4 Текстовый дашборд метрик ---")
    print("Визуализация текущего состояния системы\n")

    class Dashboard:
        """Текстовый дашборд для мониторинга."""

        def __init__(self):
            self.data = {}

        def update(self, section, metrics):
            """Обновить данные дашборда."""
            self.data[section] = metrics

        def render(self):
            """Отрисовать дашборд."""
            lines = []
            lines.append("=" * 60)
            lines.append("  DASHBOARD — Multi-Agent System Monitor")
            lines.append("=" * 60)

            for section, metrics in self.data.items():
                lines.append(f"\n  [{section}]")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if "percent" in key or "rate" in key:
                            bar_len = int(value / 5)
                            bar = "█" * bar_len + "░" * (20 - bar_len)
                            lines.append(f"    {key}: {value:.1f}% {bar}")
                        else:
                            lines.append(f"    {key}: {value}")
                    else:
                        lines.append(f"    {key}: {value}")

            lines.append("\n" + "=" * 60)
            return "\n".join(lines)

    dashboard = Dashboard()

    # Обновляем данные
    dashboard.update("Система", {
        "Агентов активно": 4,
        "Задач в очереди": 12,
        "Обработано сегодня": 1547,
        "Средняя задержка": "45 мс",
    })

    dashboard.update("Ресурсы", {
        "CPU использование": 72.5,
        "RAM использование": 68.3,
        "GPU использование": 45.2,
        "Сетевой I/O": 234.5,
    })

    dashboard.update("Ошибки", {
        "Уровень ошибок": 2.3,
        "Неудачные задачи": 3,
        "Dead letter queue": 1,
        "Retry rate": 5.1,
    })

    print(dashboard.render())


# ============================================================
# Демо 3: Debugging Agents — отладка
# ============================================================

def demo_debugging():
    """Трассировка сообщений, инспекция состояния, воспроизведение."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: Debugging Agents — отладка")
    print("=" * 70)

    # --- 3.1 Трассировка сообщений ---
    print("\n--- 3.1 Трассировка сообщений (Message Tracing) ---")
    print("Отслеживание пути сообщения через систему агентов\n")

    class MessageTracer:
        """Трассировщик сообщений: записывает каждый шаг."""

        def __init__(self):
            self.traces = collections.defaultdict(list)
            self.trace_counter = 0

        def start_trace(self, message_id, payload):
            """Начать трассировку сообщения."""
            self.trace_counter += 1
            self.traces[message_id].append({
                "step": 0,
                "event": "created",
                "payload": payload,
                "timestamp": self.trace_counter,
            })

        def trace_step(self, message_id, agent, action, detail=""):
            """Записать шаг трассировки."""
            step = len(self.traces[message_id])
            self.traces[message_id].append({
                "step": step,
                "event": action,
                "agent": agent,
                "detail": detail,
                "timestamp": self.trace_counter + step,
            })

        def get_trace(self, message_id):
            """Получить полную трассировку."""
            return self.traces.get(message_id, [])

        def visualize(self, message_id):
            """Визуализировать путь сообщения."""
            trace = self.get_trace(message_id)
            if not trace:
                return "Трассировка не найдена"

            lines = [f"Трассировка сообщения '{message_id}':"]
            for entry in trace:
                agent = entry.get("agent", "system")
                indent = "  " * (entry["step"] + 1)
                lines.append(f"{indent}[{entry['event']}] {agent}: {entry.get('detail', '')}")
                if entry["step"] < len(trace) - 1:
                    lines.append(f"{indent}  │")
            return "\n".join(lines)

    tracer = MessageTracer()

    # Имитация обработки сообщения
    msg_id = "MSG-001"
    tracer.start_trace(msg_id, {"task": "классификация", "input": "image.jpg"})

    tracer.trace_step(msg_id, "Gateway", "received", "получен запрос от клиента")
    tracer.trace_step(msg_id, "Preprocessor", "processing", "нормализация данных")
    tracer.trace_step(msg_id, "Preprocessor", "forwarded", "отправлено в Router")
    tracer.trace_step(msg_id, "Router", "routed", "маршрутизация в Agent-A")
    tracer.trace_step(msg_id, "Agent-A", "processing", "инференс модели")
    tracer.trace_step(msg_id, "Agent-A", "completed", "результат: кошка (0.95)")
    tracer.trace_step(msg_id, "Gateway", "responded", "ответ отправлен клиенту")

    print(tracer.visualize(msg_id))

    # --- 3.2 Инспекция состояния ---
    print("\n--- 3.2 Инспекция состояния агентов ---")
    print("Снимки состояния для отладки\n")

    class StateInspector:
        """Инспектор состояний: снимки и сравнение."""

        def __init__(self):
            self.snapshots = []

        def take_snapshot(self, agent_id, state):
            """Сделать снимок состояния."""
            snapshot = {
                "agent": agent_id,
                "state": dict(state),
                "hash": hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8],
            }
            self.snapshots.append(snapshot)
            return snapshot

        def compare(self, idx1, idx2):
            """Сравнить два снимка."""
            s1 = self.snapshots[idx1]
            s2 = self.snapshots[idx2]

            if s1["agent"] != s2["agent"]:
                return "Разные агенты"

            changes = {}
            all_keys = set(s1["state"].keys()) | set(s2["state"].keys())
            for key in all_keys:
                v1 = s1["state"].get(key)
                v2 = s2["state"].get(key)
                if v1 != v2:
                    changes[key] = {"before": v1, "after": v2}

            return changes

        def history(self, agent_id=None):
            """История снимков."""
            if agent_id:
                return [s for s in self.snapshots if s["agent"] == agent_id]
            return self.snapshots

    inspector = StateInspector()

    # Снимки состояния агента
    states = [
        {"status": "idle", "queue": [], "memory_mb": 512, "tasks_done": 0},
        {"status": "working", "queue": ["T1"], "memory_mb": 640, "tasks_done": 0},
        {"status": "working", "queue": ["T1", "T2"], "memory_mb": 768, "tasks_done": 1},
        {"status": "working", "queue": ["T2"], "memory_mb": 640, "tasks_done": 2},
        {"status": "idle", "queue": [], "memory_mb": 512, "tasks_done": 3},
    ]

    print("Снимки состояния Agent-A:")
    for i, state in enumerate(states):
        snap = inspector.take_snapshot("Agent-A", state)
        print(f"  Снимок {i}: status={state['status']}, "
              f"queue={state['queue']}, tasks_done={state['tasks_done']}, "
              f"hash={snap['hash']}")

    # Сравнение снимков
    print("\nСравнение снимков 0 и 2:")
    changes = inspector.compare(0, 2)
    if isinstance(changes, dict):
        for key, diff in changes.items():
            print(f"  {key}: {diff['before']} -> {diff['after']}")
    else:
        print(f"  {changes}")

    # --- 3.3 Воспроизведение (Replay) ---
    print("\n--- 3.3 Воспроизведение сценариев (Replay) ---")
    print("Повторное выполнение для воспроизведения багов\n")

    class ReplayEngine:
        """Движок воспроизведения: запись и воспроизведение сценариев."""

        def __init__(self):
            self.recorded_events = []
            self.is_recording = False

        def start_recording(self):
            """Начать запись."""
            self.recorded_events = []
            self.is_recording = True

        def record(self, event_type, data):
            """Записать событие."""
            if self.is_recording:
                self.recorded_events.append({
                    "type": event_type,
                    "data": data,
                    "step": len(self.recorded_events),
                })

        def stop_recording(self):
            """Остановить запись."""
            self.is_recording = False
            return len(self.recorded_events)

        def replay(self, handler):
            """Воспроизвести записанные события."""
            results = []
            for event in self.recorded_events:
                result = handler(event)
                results.append(result)
            return results

    engine = ReplayEngine()

    # Запись сценария
    engine.start_recording()
    engine.record("input", {"task": "T1", "data": [1, 2, 3]})
    engine.record("process", {"agent": "A0", "action": "start"})
    engine.record("output", {"result": "ok", "latency_ms": 45})
    engine.record("input", {"task": "T2", "data": [4, 5, 6]})
    engine.record("error", {"agent": "A0", "error": "timeout"})
    engine.record("retry", {"agent": "A1", "action": "start"})
    engine.record("output", {"result": "ok", "latency_ms": 120})
    n_events = engine.stop_recording()

    print(f"Записано событий: {n_events}")

    # Воспроизведение
    print("\nВоспроизведение сценария:")
    replay_results = engine.replay(lambda event: {
        "step": event["step"],
        "type": event["type"],
        "processed": True,
    })

    for result in replay_results:
        print(f"  Шаг {result['step']}: {result['type']} -> обработано")

    # --- 3.4 Логирование и структурированные логи ---
    print("\n--- 3.4 Структурированное логирование ---")
    print("Логи с контекстом для быстрого поиска проблем\n")

    class StructuredLogger:
        """Структурированный логгер для агентов."""

        def __init__(self):
            self.logs = []
            self.log_id = 0

        def log(self, level, agent, message, **context):
            """Записать структурированный лог."""
            self.log_id += 1
            entry = {
                "id": self.log_id,
                "level": level,
                "agent": agent,
                "message": message,
                "context": context,
            }
            self.logs.append(entry)
            return entry

        def query(self, level=None, agent=None, pattern=None):
            """Поиск по логам."""
            results = self.logs

            if level:
                results = [l for l in results if l["level"] == level]
            if agent:
                results = [l for l in results if l["agent"] == agent]
            if pattern:
                results = [l for l in results if pattern in l["message"]]

            return results

        def format(self, entry):
            """Форматировать лог для вывода."""
            ctx = json.dumps(entry["context"], ensure_ascii=False) if entry["context"] else ""
            return f"[{entry['level']:5s}] {entry['agent']:12s} | {entry['message']} {ctx}"

    logger = StructuredLogger()

    # Генерируем логи
    log_data = [
        ("INFO", "Gateway", "Запрос получен", {"task_id": "T1", "client": "user_42"}),
        ("INFO", "Agent-A", "Начало обработки", {"task_id": "T1", "model": "v2.1"}),
        ("WARN", "Agent-A", "Высокое использование памяти", {"memory_mb": 1800, "threshold": 2048}),
        ("INFO", "Agent-A", "Обработка завершена", {"task_id": "T1", "latency_ms": 45}),
        ("ERROR", "Agent-B", "Таймаут подключения", {"timeout_ms": 5000, "retry_count": 3}),
        ("INFO", "Router", "Перенаправление задачи", {"task_id": "T2", "from_agent": "B", "to_agent": "C"}),
        ("INFO", "Agent-C", "Начало обработки", {"task_id": "T2", "model": "v2.1"}),
        ("ERROR", "Agent-C", "Модель не найдена", {"model": "v3.0", "available": ["v2.0", "v2.1"]}),
        ("INFO", "Agent-C", "Обработка завершена", {"task_id": "T2", "latency_ms": 120}),
    ]

    for level, agent, message, context in log_data:
        logger.log(level, agent, message, **context)

    print("Все логи:")
    for entry in logger.logs:
        print(f"  {logger.format(entry)}")

    # Поиск по фильтрам
    print("\nТолько ERROR логи:")
    errors = logger.query(level="ERROR")
    for entry in errors:
        print(f"  {logger.format(entry)}")

    print("\nЛоги Agent-A:")
    agent_logs = logger.query(agent="Agent-A")
    for entry in agent_logs:
        print(f"  {logger.format(entry)}")

    print("\nПоиск 'таймаут':")
    timeout_logs = logger.query(pattern="Таймаут")
    for entry in timeout_logs:
        print(f"  {logger.format(entry)}")


# ============================================================
# Демо 4: Operations — операционные процессы
# ============================================================

def demo_operations():
    """Версионирование, A/B тестирование, откат, планирование ёмкости."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: Operations — операционные процессы")
    print("=" * 70)

    # --- 4.1 Версионирование ---
    print("\n--- 4.1 Версионирование агентов ---")
    print("Управление версиями моделей и конфигураций агентов\n")

    class VersionManager:
        """Менеджер версий агентов."""

        def __init__(self):
            self.versions = {}
            self.current = {}
            self.history = []

        def register_version(self, agent_id, version, config):
            """Зарегистрировать новую версию."""
            if agent_id not in self.versions:
                self.versions[agent_id] = {}
            self.versions[agent_id][version] = {
                "config": dict(config),
                "deployed": False,
                "metrics": {},
            }

        def deploy(self, agent_id, version):
            """Развернуть версию."""
            if agent_id in self.versions and version in self.versions[agent_id]:
                # Сохраняем предыдущую версию
                prev = self.current.get(agent_id)
                self.current[agent_id] = version
                self.versions[agent_id][version]["deployed"] = True

                self.history.append({
                    "agent": agent_id,
                    "from": prev,
                    "to": version,
                    "action": "deploy",
                })
                return True
            return False

        def rollback(self, agent_id):
            """Откатить к предыдущей версии."""
            agent_versions = self.versions.get(agent_id, {})
            deployed = [v for v, info in agent_versions.items() if info["deployed"]]

            if len(deployed) >= 2:
                current_ver = self.current.get(agent_id)
                prev_ver = deployed[-2]  # предыдущая развёрнутая

                self.current[agent_id] = prev_ver
                self.history.append({
                    "agent": agent_id,
                    "from": current_ver,
                    "to": prev_ver,
                    "action": "rollback",
                })
                return prev_ver
            return None

        def list_versions(self, agent_id):
            """Список версий агента."""
            versions = self.versions.get(agent_id, {})
            return {
                v: {
                    "deployed": info["deployed"],
                    "current": self.current.get(agent_id) == v,
                }
                for v, info in versions.items()
            }

    version_mgr = VersionManager()

    # Регистрируем версии
    configs = [
        ("v1.0", {"model": "baseline", "temperature": 0.7}),
        ("v1.1", {"model": "baseline", "temperature": 0.5}),
        ("v2.0", {"model": "enhanced", "temperature": 0.3}),
        ("v2.1", {"model": "enhanced", "temperature": 0.4, "max_tokens": 2048}),
    ]

    for version, config in configs:
        version_mgr.register_version("Agent-A", version, config)

    # Разворачиваем версии
    print("Развёртывание версий:")
    for version, _ in configs:
        ok = version_mgr.deploy("Agent-A", version)
        print(f"  {version}: {'OK' if ok else 'ОШИБКА'}")

    print(f"\nТекущая версия: {version_mgr.current.get('Agent-A')}")

    # Откат
    rolled_back = version_mgr.rollback("Agent-A")
    print(f"Откат к: {rolled_back}")

    # Список версий
    print("\nВерсии Agent-A:")
    for ver, info in version_mgr.list_versions("Agent-A").items():
        current = " (ТЕКУЩАЯ)" if info["current"] else ""
        deployed = " [развёрнута]" if info["deployed"] else ""
        print(f"  {ver}{current}{deployed}")

    # --- 4.2 A/B тестирование ---
    print("\n--- 4.2 A/B тестирование агентов ---")
    print("Сравнение двух версий на реальном трафике\n")

    class ABTest:
        """A/B тест для сравнения версий агента."""

        def __init__(self, name, variant_a, variant_b, traffic_split=0.5):
            self.name = name
            self.variant_a = variant_a
            self.variant_b = variant_b
            self.traffic_split = traffic_split
            self.results = {"A": [], "B": []}
            self.assignments = []

        def assign(self, request_id):
            """Назначить запрос на вариант A или B."""
            variant = "A" if random.random() < self.traffic_split else "B"
            self.assignments.append({"request": request_id, "variant": variant})
            return variant

        def record_result(self, variant, latency_ms, success):
            """Записать результат."""
            self.results[variant].append({
                "latency_ms": latency_ms,
                "success": success,
            })

        def analyze(self):
            """Анализ результатов."""
            analysis = {}
            for variant in ["A", "B"]:
                results = self.results[variant]
                if not results:
                    continue

                latencies = [r["latency_ms"] for r in results]
                successes = sum(1 for r in results if r["success"])

                analysis[variant] = {
                    "variant": self.variant_a if variant == "A" else self.variant_b,
                    "samples": len(results),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                    "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
                    "success_rate": f"{successes / len(results) * 100:.1f}%",
                }

            return analysis

    # Создаём A/B тест
    ab_test = ABTest(
        name="Модель v2.0 vs v2.1",
        variant_a="v2.0 (baseline)",
        variant_b="v2.1 (enhanced)",
        traffic_split=0.5,
    )

    # Симуляция запросов
    print(f"Тест: {ab_test.name}")
    print(f"Распределение: {ab_test.traffic_split * 100:.0f}% / {(1 - ab_test.traffic_split) * 100:.0f}%")

    for i in range(100):
        variant = ab_test.assign(f"req_{i}")

        if variant == "A":
            latency = random.uniform(30, 80)
            success = random.random() < 0.95
        else:
            latency = random.uniform(25, 60)  # v2.1 быстрее
            success = random.random() < 0.98  # и точнее

        ab_test.record_result(variant, latency, success)

    analysis = ab_test.analyze()
    print("\nРезультаты:")
    for variant, stats in analysis.items():
        print(f"\n  Вариант {variant} ({stats['variant']}):")
        print(f"    Запросов: {stats['samples']}")
        print(f"    Средняя задержка: {stats['avg_latency_ms']} мс")
        print(f"    P95 задержка: {stats['p95_latency_ms']} мс")
        print(f"    Успешность: {stats['success_rate']}")

    # Определение победителя
    if "A" in analysis and "B" in analysis:
        a_score = float(analysis["A"]["success_rate"].rstrip("%")) / analysis["A"]["avg_latency_ms"]
        b_score = float(analysis["B"]["success_rate"].rstrip("%")) / analysis["B"]["avg_latency_ms"]
        winner = "B (v2.1)" if b_score > a_score else "A (v2.0)"
        print(f"\n  Победитель: {winner} (score: A={a_score:.4f}, B={b_score:.4f})")

    # --- 4.3 Планирование ёмкости ---
    print("\n--- 4.3 Планирование ёмкости ---")
    print("Прогнозирование потребности в ресурсах\n")

    class CapacityPlanner:
        """Планировщик ёмкости на основе истории."""

        def __init__(self):
            self.history = []

        def add_datapoint(self, hour, agents, load, response_time_ms):
            """Добавить точку данных."""
            self.history.append({
                "hour": hour,
                "agents": agents,
                "load": load,
                "response_time_ms": response_time_ms,
            })

        def forecast(self, target_load):
            """Прогноз: сколько агентов нужно для заданной нагрузки."""
            if not self.history:
                return None

            # Простая линейная регрессия: agents = f(load)
            loads = [h["load"] for h in self.history]
            agents = [h["agents"] for h in self.history]

            n = len(loads)
            sum_x = sum(loads)
            sum_y = sum(agents)
            sum_xy = sum(x * y for x, y in zip(loads, agents))
            sum_x2 = sum(x * x for x in loads)

            # Коэффициенты линейной регрессии
            denom = n * sum_x2 - sum_x * sum_x
            if denom == 0:
                return None

            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n

            predicted = slope * target_load + intercept
            return max(1, math.ceil(predicted))

        def efficiency_report(self):
            """Отчёт об эффективности использования ресурсов."""
            if not self.history:
                return {}

            loads = [h["load"] for h in self.history]
            agents = [h["agents"] for h in self.history]
            response_times = [h["response_time_ms"] for h in self.history]

            # Утилизация: реальная нагрузка vs доступная ёмкость
            avg_utilization = sum(l / (a * 100) * 100 for l, a in zip(loads, agents)) / len(loads)

            return {
                "avg_load": round(sum(loads) / len(loads), 1),
                "avg_agents": round(sum(agents) / len(agents), 1),
                "avg_response_ms": round(sum(response_times) / len(response_times), 1),
                "avg_utilization": round(avg_utilization, 1),
                "peak_load": max(loads),
                "min_load": min(loads),
            }

    planner = CapacityPlanner()

    # Собираем данные за 24 часа
    print("Сбор данных за 24 часа:")
    for hour in range(24):
        # Имитация паттерна нагрузки: пик днём, спад ночью
        base_load = 30 + 40 * math.sin((hour - 6) * math.pi / 12)
        load = max(10, base_load + random.uniform(-10, 10))
        agents = max(2, int(load / 25) + 1)
        response_time = 50 + load * 0.5 + random.uniform(-5, 5)

        planner.add_datapoint(hour, agents, load, response_time)

    report = planner.efficiency_report()
    print(f"\nОтчёт об эффективности:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Прогноз
    print("\nПрогноз потребности:")
    test_loads = [30, 50, 70, 90, 100]
    for load in test_loads:
        needed = planner.forecast(load)
        print(f"  Нагрузка={load}%: нужно {needed} агентов")

    # --- 4.4 Каталог инцидентов ---
    print("\n--- 4.4 Каталог инцидентов ---")
    print("Регистрация и анализ инцидентов\n")

    class IncidentCatalog:
        """Каталог инцидентов с анализом."""

        def __init__(self):
            self.incidents = []
            self.incident_id = 0

        def report_incident(self, title, severity, affected_agents, root_cause, resolution):
            """Зарегистрировать инцидент."""
            self.incident_id += 1
            incident = {
                "id": f"INC-{self.incident_id:04d}",
                "title": title,
                "severity": severity,
                "affected_agents": affected_agents,
                "root_cause": root_cause,
                "resolution": resolution,
                "status": "resolved",
            }
            self.incidents.append(incident)
            return incident["id"]

        def analyze_patterns(self):
            """Анализ паттернов инцидентов."""
            causes = collections.Counter(i["root_cause"] for i in self.incidents)
            severities = collections.Counter(i["severity"] for i in self.incidents)
            agent_impact = collections.Counter()
            for i in self.incidents:
                for agent in i["affected_agents"]:
                    agent_impact[agent] += 1

            return {
                "total": len(self.incidents),
                "by_cause": dict(causes.most_common()),
                "by_severity": dict(severities),
                "most_affected_agents": dict(agent_impact.most_common(5)),
            }

        def mttr(self):
            """Mean Time To Resolution (в минутах, имитация)."""
            # В реальности считается по timestamps
            return sum(random.randint(5, 60) for _ in self.incidents) / max(1, len(self.incidents))

    catalog = IncidentCatalog()

    # Регистрируем инциденты
    incidents = [
        ("OOM на Agent-A", "critical", ["Agent-A"], "memory leak", "перезапуск + патч"),
        ("Таймаут инференса", "warning", ["Agent-B", "Agent-C"], " overloaded", "autoscaling"),
        ("Потеря сообщений", "critical", ["Router"], "broker overload", "очередь расширена"),
        ("Медленный ответ", "warning", ["Agent-D"], "model loading", "кэширование модели"),
        ("Дублирование задач", "info", ["Agent-A", "Agent-B"], "race condition", "distributed lock"),
    ]

    for title, severity, agents, cause, resolution in incidents:
        inc_id = catalog.report_incident(title, severity, agents, cause, resolution)
        print(f"  {inc_id}: {title} [{severity}]")

    # Анализ
    analysis = catalog.analyze_patterns()
    print(f"\nАнализ инцидентов:")
    print(f"  Всего: {analysis['total']}")
    print(f"  По причинам: {analysis['by_cause']}")
    print(f"  По серьёзности: {analysis['by_severity']}")
    print(f"  Часто затрагиваемые агенты: {analysis['most_affected_agents']}")

    mttr = catalog.mttr()
    print(f"\nСреднее время восстановления (MTTR): {mttr:.1f} минут")


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    demo_deployment_patterns()
    demo_monitoring()
    demo_debugging()
    demo_operations()
