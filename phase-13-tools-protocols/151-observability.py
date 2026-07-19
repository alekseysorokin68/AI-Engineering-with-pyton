"""151 — Monitoring & Observability: Prometheus, Grafana, распределённая трассировка

Темы:
  1. Metrics — счётчики, гистограммы, SLI/SLO/SLA
  2. Structured Logging — JSON-логи, уровни, correlation ID
  3. Distributed Tracing — спаны, контекст трассировки, пропагация
  4. Dashboards & Alerting — панели Grafana, правила алертов, эскалация

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import sqlite3

random.seed(42)

# =============================================================================
# Демо 1: Metrics — счётчики, гистограммы, SLI/SLO/SLA
# =============================================================================

def demo_metrics():
    print("=" * 70)
    print("ДЕМО 1: Metrics — Counter, Histogram, SLI/SLO/SLA")
    print("=" * 70)

    # --- 1.1 Типы метрик Prometheus ---
    # Counter — монотонно возрастающее значение
    # Gauge — значение, которое может расти и уменьшаться
    # Histogram — распределение значений с бакетами
    # Summary — квантили на стороне клиента
    print("\n--- 1.1 Типы метрик Prometheus ---")

    # Counter: количество запросов
    http_requests_total = 0
    for _ in range(150):
        http_requests_total += 1
    print(f"http_requests_total (Counter): {http_requests_total}")
    print("Counter: только вверх, сбрасывается при рестарте процесса")

    # Gauge: текущее значение
    active_connections = 42
    print(f"active_connections (Gauge): {active_connections}")
    print("Gauge: может расти и падать (очередь, температура, память)")

    # Histogram: распределение латентности
    latencies = [random.expovariate(1.0 / 0.15) for _ in range(1000)]  # в секундах
    buckets_ms = [10, 25, 50, 100, 250, 500, 1000]
    histogram = collections.Counter()
    for lat in latencies:
        lat_ms = lat * 1000
        for b in buckets_ms:
            if lat_ms <= b:
                histogram[b] += 1
                break
        else:
            histogram[">1000"] = histogram.get(">1000", 0) + 1

    print("\nГистограмма латентности (1000 запросов):")
    print(f"  {'Бакет':>10s} {'Кол-во':>8s} {'%':>7s}")
    for b in buckets_ms:
        pct = histogram[b] / 1000 * 100
        bar = "█" * int(pct / 2)
        print(f"  ≤{b:>6d}ms  {histogram[b]:8d}  {pct:5.1f}%  {bar}")
    oob = histogram.get(">1000", 0)
    print(f"  >1000ms  {oob:8d}  {oob/1000*100:5.1f}%")

    # --- 1.2 SLI, SLO, SLA ---
    # SLI — Service Level Indicator (конкретная метрика)
    # SLO — Service Level Objective (целевое значение)
    # SLA — Service Level Agreement (юридическое обязательство)
    print("\n--- 1.2 SLI, SLO, SLA ---")

    total_requests = 100000
    successful = 99700
    failed = total_requests - successful

    sli_availability = successful / total_requests * 100
    sli_latency_p99 = 0.250  # 250ms

    slo_availability = 99.9
    slo_latency_p99 = 0.500  # 500ms

    sla_availability = 99.5
    sla_latency_p99 = 1.000  # 1000ms

    print(f"Общее число запросов: {total_requests:,}")
    print(f"Успешных: {successful:,}, Ошибок: {failed:,}")
    print()
    print(f"  {'Уровень':>12s} {'Availability':>12s} {'Latency P99':>12s} {'Статус':>10s}")
    print(f"  {'-'*12:>12s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s}")

    slis = [
        ("SLI (текущее)", f"{sli_availability:.3f}%", f"{sli_latency_p99*1000:.0f}ms"),
        ("SLO (целевое)", f"{slo_availability:.2f}%", f"{slo_latency_p99*1000:.0f}ms"),
        ("SLA (минимум)", f"{sla_availability:.1f}%",  f"{sla_latency_p99*1000:.0f}ms"),
    ]

    for name, avail, lat in slis:
        avail_ok = sli_availability >= float(avail.rstrip("%"))
        lat_ok = sli_latency_p99 <= float(lat.rstrip("ms")) / 1000
        status = "ВЫПОЛНЯЕТСЯ" if (avail_ok and lat_ok) else "НАРУШЕНО"
        print(f"  {name:>12s} {avail:>12s} {lat:>12s} {status:>10s}")

    # --- 1.3 Error budget ---
    # Error budget = 1 - SLO
    print("\n--- 1.3 Error Budget ---")

    error_budget = (100 - slo_availability) / 100
    error_budget_requests = int(total_requests * error_budget)
    consumed = failed
    remaining = max(0, error_budget_requests - consumed)

    print(f"SLO: {slo_availability}% -> Error budget: {error_budget*100:.1f}%")
    print(f"Всего запросов за период: {total_requests:,}")
    print(f"Бюджет ошибок: {error_budget_requests:,} запросов")
    print(f"Потрачено: {consumed:,}, Осталось: {remaining:,}")
    print(f"Использование бюджета: {consumed/error_budget_requests*100:.1f}%")
    if remaining == 0:
        print("  ⚠ ERROR BUDGET ИСЧЕРПАН! Необходимо замедлить деплои!")

    # --- 1.4 Прометеевский язык запросов (PQL) ---
    print("\n--- 1.4 Примеры PQL-запросов ---")

    queries = [
        ("Скорость запросов (RPS)", "rate(http_requests_total[5m])"),
        ("P99 латентность", "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))"),
        ("Доля ошибок", "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])"),
        ("Предсказание на 4ч", "predict_linear(http_requests_total[1h], 4*3600)"),
    ]

    for name, q in queries:
        print(f"  {name}:")
        print(f"    {q}")


# =============================================================================
# Демо 2: Structured Logging — JSON-логи, уровни, correlation ID
# =============================================================================

def demo_structured_logging():
    print("=" * 70)
    print("ДЕМО 2: Structured Logging — JSON, уровни, correlation ID")
    print("=" * 70)

    # --- 2.1 Уровни логирования ---
    print("\n--- 2.1 Уровни логирования ---")

    levels = {
        "DEBUG":    "Подробная отладочная информация (выключается в production)",
        "INFO":     "Обычные операции (запрос обработан, соединение установлено)",
        "WARNING":  "Неожиданное, но не ошибка (старый API, deprecated usage)",
        "ERROR":    "Ошибка, требующая внимания (БД недоступна, таймаут)",
        "CRITICAL": "Критическая ошибка, сервис может упасть",
    }

    for level, desc in levels.items():
        print(f"  {level:10s} -> {desc}")

    # --- 2.2 JSON-формат логов ---
    print("\n--- 2.2 JSON-формат логов ---")

    log_entries = [
        {
            "timestamp": "2025-03-25T10:30:01.123Z",
            "level": "INFO",
            "service": "user-api",
            "message": "Обработка запроса завершена",
            "requestId": "req-abc-123",
            "userId": "user-42",
            "duration_ms": 45,
            "http_status": 200,
        },
        {
            "timestamp": "2025-03-25T10:30:01.456Z",
            "level": "ERROR",
            "service": "user-api",
            "message": "Ошибка подключения к БД",
            "requestId": "req-def-456",
            "error": "connection refused",
            "db_host": "postgres-primary",
            "retry_count": 3,
        },
        {
            "timestamp": "2025-03-25T10:30:02.789Z",
            "level": "WARNING",
            "service": "user-api",
            "message": "Медленный запрос",
            "requestId": "req-ghi-789",
            "duration_ms": 2500,
            "threshold_ms": 1000,
        },
    ]

    for entry in log_entries:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        print()

    # --- 2.3 Correlation ID ---
    # Уникальный ID для трассировки запроса через все микросервисы
    print("\n--- 2.3 Correlation ID — трассировка через микросервисы ---")

    request_id = hashlib.md5(str(random.random()).encode()).hexdigest()[:12]
    services_trace = [
        ("api-gateway",   "Получен входящий запрос", 2),
        ("auth-service",  "Проверка JWT-токена", 5),
        ("user-service",  "Запрос профиля пользователя", 15),
        ("cache-redis",   "Проверка кэша (HIT)", 1),
        ("postgres",      "SELECT * FROM users WHERE id = ?", 8),
    ]

    print(f"Trace ID: {request_id}")
    print()
    cumulative = 0
    for svc, action, latency in services_trace:
        cumulative += latency
        print(f"  [{request_id}] [{svc:14s}] {action} ({latency}ms, cumulative: {cumulative}ms)")

    # --- 2.4 Фильтрация и агрегация логов ---
    print("\n--- 2.4 Фильтрация и агрегация логов ---")

    # Генерируем массив логов
    random.seed(42)
    log_levels = ["DEBUG", "INFO", "INFO", "INFO", "WARNING", "ERROR", "CRITICAL"]
    services = ["user-api", "order-api", "payment-api", "gateway"]

    logs = []
    for i in range(200):
        logs.append({
            "level": random.choice(log_levels),
            "service": random.choice(services),
            "timestamp": f"2025-03-25T{10 + i // 30:02d}:{(i * 17) % 60:02d}:00Z",
            "message": f"Event {i}",
        })

    # Подсчёт по уровням
    level_counts = collections.Counter(l["level"] for l in logs)
    print("Распределение по уровням:")
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        count = level_counts.get(level, 0)
        bar = "█" * (count // 2)
        print(f"  {level:10s}: {count:4d} {bar}")

    # Подсчёт по сервисам
    service_counts = collections.Counter(l["service"] for l in logs)
    print("\nРаспределение по сервисам:")
    for svc, count in service_counts.most_common():
        print(f"  {svc:14s}: {count:4d}")

    # Фильтрация: только ERROR и выше
    errors = [l for l in logs if l["level"] in ("ERROR", "CRITICAL")]
    print(f"\nЛоги уровня ERROR+: {len(errors)} из {len(logs)}")


# =============================================================================
# Демо 3: Distributed Tracing — спаны, контекст, пропагация
# =============================================================================

def demo_distributed_tracing():
    print("=" * 70)
    print("ДЕМО 3: Distributed Tracing — спаны, trace context, пропагация")
    print("=" * 70)

    # --- 3.1 Концепция спанов ---
    # Span — единица работы в распределённой системе
    # Trace — дерево спанов, представляющих путь запроса
    print("\n--- 3.1 Концепция спанов ---")

    trace_id = hashlib.sha256(str(random.random()).encode()).hexdigest()[:32]
    span_id_0 = hashlib.md5(str(random.random()).encode()).hexdigest()[:16]

    spans = [
        {
            "traceId": trace_id,
            "spanId": span_id_0,
            "parentSpanId": None,
            "operationName": "HTTP GET /api/orders",
            "service": "api-gateway",
            "duration_ms": 156,
            "status": "OK",
        },
    ]

    # Дочерние спаны
    parent_id = span_id_0
    child_operations = [
        ("auth.verify_token",     "auth-service",   12),
        ("db.query_orders",       "postgres",        45),
        ("cache.get_orders",      "redis",            3),
        ("rpc.order-service",     "order-service",   89),
        ("  db.find_by_user",     "postgres",        22),
        ("  grpc.inventory.check","inventory-svc",   35),
    ]

    for op, svc, dur in child_operations:
        span_id = hashlib.md5(str(random.random()).encode()).hexdigest()[:16]
        spans.append({
            "traceId": trace_id,
            "spanId": span_id,
            "parentSpanId": parent_id,
            "operationName": op,
            "service": svc,
            "duration_ms": dur,
            "status": "OK",
        })

    print(f"Trace ID: {trace_id}")
    print(f"Количество спанов: {len(spans)}")
    print()

    # Вывод дерева спанов
    for span in spans:
        indent = "  " if span["parentSpanId"] else ""
        prefix = "└─" if span["parentSpanId"] else "●"
        print(f"  {indent}{prefix} [{span['service']:14s}] {span['operationName']:30s} "
              f"{span['duration_ms']:4d}ms")

    # --- 3.2 Trace Context (W3C) ---
    # Формат: traceparent: 00-<trace-id>-<parent-id>-<trace-flags>
    print("\n--- 3.2 Trace Context (W3C标准) ---")

    traceparent = f"00-{trace_id}-{span_id_0}-01"
    print(f"traceparent: {traceparent}")
    print("Формат: 00-<trace-id-32hex>-<parent-span-id-16hex>-<trace-flags-2hex>")
    print("  00 = версия протокола")
    print("  01 = trace-flags (01 = sampled)")

    # Пропагация контекста через HTTP-заголовки
    print("\nПропагация через HTTP-заголовки:")
    print(f"  traceparent: {traceparent}")
    print(f"  tracestate: vendor1=value1,vendor2=value2")

    # --- 3.3 Пропагация между сервисами ---
    print("\n--- 3.3 Пропагация между сервисами ---")

    services_path = [
        {"service": "client",       "action": "Отправка запроса"},
        {"service": "api-gateway",  "action": "Парсинг traceparent, создание span"},
        {"service": "auth-service",  "action": "Чтение traceparent, продолжение trace"},
        {"service": "user-service",  "action": "Чтение traceparent, продолжение trace"},
        {"service": "postgres",      "action": "Драйвер БД создаёт span (автоматически)"},
    ]

    current_span_id = span_id_0
    print()
    for i, s in enumerate(services_path):
        new_span_id = hashlib.md5(f"{random.random()}".encode()).hexdigest()[:16]
        header = f"traceparent: 00-{trace_id}-{current_span_id}-01"
        print(f"  {s['service']:14s} <- {s['action']}")
        print(f"    Заголовок: {header}")
        current_span_id = new_span_id
        print()

    # --- 3.4 Анализ спанов и поиск медленных ---
    print("\n--- 3.4 Анализ спанов — поиск bottle neck ---")

    # Дополним спаны данными
    all_spans = [
        {"op": "GET /api/orders",       "dur": 156, "svc": "api-gateway"},
        {"op": "auth.verify_token",      "dur": 12,  "svc": "auth-service"},
        {"op": "cache.get_orders",       "dur": 3,   "svc": "redis"},
        {"op": "db.query_orders",        "dur": 45,  "svc": "postgres"},
        {"op": "order-service.process",  "dur": 89,  "svc": "order-service"},
        {"op": "db.find_by_user",        "dur": 22,  "svc": "postgres"},
        {"op": "grpc.inventory.check",   "dur": 35,  "svc": "inventory-svc"},
    ]

    # Сортируем по длительности
    sorted_spans = sorted(all_spans, key=lambda s: s["dur"], reverse=True)
    print("Топ-5 самых долгих спанов:")
    for i, s in enumerate(sorted_spans[:5], 1):
        pct = s["dur"] / 156 * 100  # относительно корневого спана
        bar = "█" * int(pct / 5)
        print(f"  {i}. [{s['svc']:14s}] {s['op']:30s} {s['dur']:4d}ms ({pct:.0f}%) {bar}")

    print(f"\nBottle neck: {sorted_spans[0]['svc']} — {sorted_spans[0]['op']} "
          f"({sorted_spans[0]['dur']}ms = {sorted_spans[0]['dur']/156*100:.0f}% общего времени)")


# =============================================================================
# Демо 4: Dashboards & Alerting — Grafana, правила алертов
# =============================================================================

def demo_dashboards_alerting():
    print("=" * 70)
    print("ДЕМО 4: Dashboards & Alerting — Grafana, правила алертов")
    print("=" * 70)

    # --- 4.1 Структура Grafana-дашборда ---
    print("\n--- 4.1 Структура Grafana-дашборда ---")

    dashboard = {
        "title": "AI Engineering — Service Overview",
        "panels": [
            {
                "title": "Request Rate (RPS)",
                "type": "graph",
                "query": "sum(rate(http_requests_total[5m]))",
                "position": {"row": 1, "col": 1, "w": 12, "h": 8},
            },
            {
                "title": "Error Rate (%)",
                "type": "graph",
                "query": "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m])) * 100",
                "position": {"row": 1, "col": 13, "w": 12, "h": 8},
                "thresholds": [{"value": 1, "color": "yellow"}, {"value": 5, "color": "red"}],
            },
            {
                "title": "P99 Latency (ms)",
                "type": "heatmap",
                "query": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) * 1000",
                "position": {"row": 9, "col": 1, "w": 12, "h": 8},
            },
            {
                "title": "Active Connections",
                "type": "stat",
                "query": "active_connections",
                "position": {"row": 9, "col": 13, "w": 6, "h": 4},
            },
        ],
    }

    print(f"Дашборд: {dashboard['title']}")
    print(f"Панелей: {len(dashboard['panels'])}")
    for p in dashboard["panels"]:
        pos = p["position"]
        print(f"  [{p['type']:8s}] {p['title']:28s} | "
              f"row={pos['row']:2d}, col={pos['col']:2d}, {pos['w']}x{pos['h']}")

    # --- 4.2 Правила алертов ---
    print("\n--- 4.2 Правила алертов (Prometheus Alerting Rules) ---")

    alert_rules = [
        {
            "alert": "HighErrorRate",
            "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05',
            "for": "5m",
            "severity": "critical",
            "summary": "Доля ошибок > 5% в течение 5 минут",
        },
        {
            "alert": "HighLatency",
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1.0",
            "for": "10m",
            "severity": "warning",
            "summary": "P99 латентность > 1s в течение 10 минут",
        },
        {
            "alert": "PodCrashLooping",
            "expr": "rate(kube_pod_container_status_restarts_total[15m]) > 0",
            "for": "5m",
            "severity": "critical",
            "summary": "Под перезапускается (CrashLoopBackOff)",
        },
        {
            "alert": "HighMemoryUsage",
            "expr": "container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9",
            "for": "5m",
            "severity": "warning",
            "summary": "Использование памяти > 90% от лимита",
        },
    ]

    for rule in alert_rules:
        print(f"  ALERT: {rule['alert']}")
        print(f"    expr: {rule['expr']}")
        print(f"    for: {rule['for']}, severity: {rule['severity']}")
        print(f"    summary: {rule['summary']}")
        print()

    # --- 4.3 Эскалация алертов ---
    print("\n--- 4.3 Эскалация алертов ---")

    escalation_policy = [
        {"level": 1, "target": "On-call инженер",     "channel": "Slack #alerts",
         "delay": "0м",   "action": "Уведомление в Slack, auto-acknowledge"},
        {"level": 2, "target": "Старший инженер",       "channel": "Telegram + SMS",
         "delay": "15м",  "action": "Повторное уведомление, эскалация если нет ack"},
        {"level": 3, "target": "Team Lead",             "channel": "Звонок",
         "delay": "30м",  "action": "Телефонный звонок, конференция"},
        {"level": 4, "target": "VP Engineering",        "channel": "Звонок + Email",
         "delay": "60м",  "action": "Экстренное совещание"},
    ]

    print(f"{'Level':>6s} {'Target':>20s} {'Channel':>25s} {'Delay':>6s}  Action")
    print(f"{'-'*6:>6s} {'-'*20:>20s} {'-'*25:>25s} {'-'*6:>6s}  {'-'*40}")
    for e in escalation_policy:
        print(f"  {e['level']:>4d}  {e['target']:>20s} {e['channel']:>25s} {e['delay']:>6s}  {e['action']}")

    # --- 4.4 SLO-based алерты ---
    # Горячие и холодные ошибочные бюджеты
    print("\n--- 4.4 SLO-based алерты ---")

    slo_budget = 99.9
    error_budget = 1 - slo_budget / 100
    period_days = 30
    period_seconds = period_days * 86400

    # Burn rate — скорость消耗 ошибочного бюджета
    burn_rates = [
        {"rate": 14.4, "window": "1h", "budget_pct": error_budget * 100,
         "time_to_exhaust": f"{1/14.4*100:.1f}% за 1ч"},
        {"rate": 6.0,  "window": "6h", "budget_pct": error_budget * 100,
         "time_to_exhaust": f"{1/6.0*100:.1f}% за 6ч"},
        {"rate": 1.0,  "window": "3d", "budget_pct": error_budget * 100,
         "time_to_exhaust": f"{1/1.0*100:.1f}% за 3д"},
    ]

    print(f"SLO: {slo_budget}%, Error Budget: {error_budget*100:.2f}% за {period_days} дней")
    print()
    for b in burn_rates:
        # Время до исчерпания бюджета
        exhaust_minutes = (error_budget / b["rate"]) * period_days * 24 * 60
        print(f"  Burn rate: {b['rate']:5.1f}x за {b['window']:>3s} -> "
              f"бюджет исчерпается через {exhaust_minutes:.0f} мин "
              f"({exhaust_minutes/60:.1f} ч)")

    print()
    print("Правило: если burn rate > 1x, бюджет тратится быстрее чем накапливается")
    print("Оптимальная стратегия: 2 окна (short + long) для ложных срабатываний")


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    demo_metrics()
    print()
    demo_structured_logging()
    print()
    demo_distributed_tracing()
    print()
    demo_dashboards_alerting()
