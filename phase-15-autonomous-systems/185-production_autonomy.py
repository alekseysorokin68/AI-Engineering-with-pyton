"""185 — Production Autonomy: соблюдение SLA, деградация, human-in-the-loop

Темы:
  1. SLA Compliance — availability, latency, throughput targets, error budgets
  2. Graceful Degradation — fallback strategies, feature flags, circuit breakers
  3. Human-in-the-Loop — escalation triggers, approval queues, feedback loops
  4. Autonomy Governance — audit trails, compliance checks, review boards

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import statistics

random.seed(42)

# ========================================================================
# 1. SLA Compliance
# ========================================================================

def demo_sla_compliance():
    """Демонстрация мониторинга и расчёта SLA."""
    print("=" * 70)
    print("1. SLA COMPLIANCE")
    print("=" * 70)

    # --- 1a. Расчёт доступности ---
    print("\n--- 1a. Расчёт доступности (Availability) ---\n")

    def calculate_availability(uptime_seconds, total_seconds):
        """Вычисление доступности: A = uptime / total × 100%"""
        return (uptime_seconds / total_seconds) * 100

    def availability_to_downtime(availability_pct, period_days):
        """Перевод доступности в downtime (минуты в месяц)."""
        period_minutes = period_days * 24 * 60
        downtime_minutes = period_minutes * (1 - availability_pct / 100)
        return downtime_minutes

    # SLA-уровни
    sla_levels = [
        ("99%", 99.0, "Базовый"),
        ("99.9%", 99.9, "Высокий"),
        ("99.99%", 99.99, "Критический"),
        ("99.999%", 99.999, "Ультра"),
    ]

    print(f"  {'SLA':<10} {'Доступность':>12} {'Downtime/мес':>14} {'Уровень':>12}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}")
    for name, avail, level in sla_levels:
        downtime = availability_to_downtime(avail, 30)
        print(f"  {name:<10} {avail:>11.3f}% {downtime:>12.2f} мин {level:>12}")

    # Реальные данные
    print(f"\n  Формула: A = Uptime / Total_Time × 100%")
    print(f"  Месяц = 30 дней = 43200 минут")

    uptime = 43197.5  # минуты в месяце
    total = 43200.0
    actual_avail = calculate_availability(uptime, total)
    downtime = total - uptime
    print(f"\n  Реальные данные за месяц:")
    print(f"    Uptime: {uptime:.1f} мин")
    print(f"    Downtime: {downtime:.1f} мин")
    print(f"    Доступность: {actual_avail:.4f}%")
    print(f"    SLA 99.99%: {'ВЫПОЛНЕН' if actual_avail >= 99.99 else 'НАРУШЕН'}")

    # --- 1b. Latency targets ---
    print("\n--- 1b. Мониторинг латентности (Latency Targets) ---\n")

    def calculate_percentiles(latencies):
        """Вычисление перцентилей латентности."""
        sorted_lats = sorted(latencies)
        n = len(sorted_lats)
        return {
            'p50': sorted_lats[int(n * 0.5)],
            'p90': sorted_lats[int(n * 0.9)],
            'p95': sorted_lats[int(n * 0.95)],
            'p99': sorted_lats[int(n * 0.99)],
            'max': sorted_lats[-1],
            'mean': statistics.mean(sorted_lats)
        }

    # Генерация латентностей
    random.seed(42)
    latencies = [max(5, random.gauss(50, 20)) for _ in range(1000)]
    # Добавляем выбросы
    latencies += [random.uniform(200, 500) for _ in range(10)]

    percentiles = calculate_percentiles(latencies)

    # SLA.targets для латентности
    latency_sla = {
        'p50': 60,
        'p95': 150,
        'p99': 300
    }

    print(f"  Пороговые значения SLA:")
    print(f"    p50 ≤ {latency_sla['p50']} мс")
    print(f"    p95 ≤ {latency_sla['p95']} мс")
    print(f"    p99 ≤ {latency_sla['p99']} мс")

    print(f"\n  Реальные значения:")
    for metric, target in latency_sla.items():
        actual = percentiles[metric]
        status = "OK" if actual <= target else "VIOLATION"
        print(f"    {metric}: {actual:.1f} мс (цель: ≤{target}) [{status}]")

    print(f"\n    mean: {percentiles['mean']:.1f} мс")
    print(f"    max: {percentiles['max']:.1f} мс")

    # --- 1c. Error budget ---
    print("\n--- 1c. Error Budget (бюджет ошибок) ---\n")

    def calculate_error_budget(sla_pct, period_days):
        """Расчёт бюджета ошибок на период."""
        period_seconds = period_days * 24 * 3600
        allowed_downtime = period_seconds * (1 - sla_pct / 100)
        return allowed_downtime

    def calculate_budget_remaining(allowed_downtime, used_downtime):
        """Оставшийся бюджет ошибок."""
        remaining = allowed_downtime - used_downtime
        usage_pct = (used_downtime / allowed_downtime) * 100
        return remaining, usage_pct

    # Error budget для 99.9% SLA на 30 дней
    sla_pct = 99.9
    period_days = 30
    allowed = calculate_error_budget(sla_pct, period_days)

    # Использованный downtime
    incidents = [
        {"name": "Инцидент #1", "downtime_min": 5},
        {"name": "Инцидент #2", "downtime_min": 12},
        {"name": "Деплой отката", "downtime_min": 3},
    ]

    total_used = sum(i['downtime_min'] * 60 for i in incidents)
    remaining, usage = calculate_budget_remaining(allowed, total_used)

    print(f"  SLA: {sla_pct}% на {period_days} дней")
    print(f"  Допустимый downtime: {allowed:.0f} сек = {allowed / 60:.1f} мин")
    print(f"\n  Инциденты:")
    for inc in incidents:
        print(f"    {inc['name']}: {inc['downtime_min']} мин ({inc['downtime_min'] * 60} сек)")
    print(f"  Итого использовано: {total_used:.0f} сек = {total_used / 60:.1f} мин")
    print(f"\n  Бюджет ошибок:")
    print(f"    Использовано: {usage:.1f}%")
    print(f"    Осталось: {remaining:.0f} сек ({remaining / 60:.1f} мин)")
    print(f"    Статус: {'OK' if usage < 80 else 'WARNING' if usage < 100 else 'EXHAUSTED'}")

    # --- 1d. Throughput monitoring ---
    print("\n--- 1d. Мониторинг пропускной способности ---\n")

    def throughput_analysis(requests_per_second, capacity_rps, period_minutes=60):
        """Анализ пропускной способности."""
        total_requests = requests_per_second * period_minutes * 60
        max_capacity = capacity_rps * period_minutes * 60
        utilization = (requests_per_second / capacity_rps) * 100

        # Расчёт времени до заполнения при росте
        growth_rate = 0.05  # 5% рост в час
        hours_to_full = math.log(capacity_rps / requests_per_second) / math.log(1 + growth_rate) \
            if requests_per_second < capacity_rps else 0

        return {
            'total_requests': total_requests,
            'max_capacity': max_capacity,
            'utilization': utilization,
            'hours_to_capacity': hours_to_full
        }

    current_rps = 850
    max_rps = 1000

    metrics = throughput_analysis(current_rps, max_rps)
    print(f"  Текущая нагрузка: {current_rps} req/s")
    print(f"  Максимальная: {max_rps} req/s")
    print(f"  Утилизация: {metrics['utilization']:.1f}%")
    print(f"  Запросов за час: {metrics['total_requests']:,}")
    print(f"  Максимум за час: {metrics['max_capacity']:,}")

    # Прогноз
    growth_rates = [0.01, 0.03, 0.05, 0.10]
    print(f"\n  Прогноз времени до заполнения capacity:")
    for gr in growth_rates:
        hours = math.log(max_rps / current_rps) / math.log(1 + gr) \
            if current_rps < max_rps else 0
        days = hours / 24
        print(f"    Рост {gr:.0%}/час: {hours:.1f} ч ({days:.1f} дн)")


# ========================================================================
# 2. Graceful Degradation
# ========================================================================

def demo_graceful_degradation():
    """Демонстрация механизмов优雅ной деградации."""
    print("\n" + "=" * 70)
    print("2. GRACEFUL DEGRADATION")
    print("=" * 70)

    # --- 2a. Fallback strategies ---
    print("\n--- 2a. Стратегии fallback ---\n")

    class ServiceWithFallback:
        """Сервис с несколькими уровнями fallback."""
        def __init__(self, name):
            self.name = name
            self.primary_ok = True
            self.cache_ok = True
            self.default_value = {"status": "degraded", "data": "default"}

        def primary_service(self):
            """Основной сервис."""
            if not self.primary_ok:
                raise ConnectionError("Primary service unavailable")
            return {"status": "ok", "data": "fresh_data", "source": "primary"}

        def cache_fallback(self):
            """Кэш как fallback."""
            if not self.cache_ok:
                raise ConnectionError("Cache unavailable")
            return {"status": "ok", "data": "cached_data", "source": "cache"}

        def default_fallback(self):
            """Значение по умолчанию."""
            return self.default_value

        def get_data(self):
            """Получение данных с каскадным fallback."""
            strategies = [
                ("primary", self.primary_service),
                ("cache", self.cache_fallback),
                ("default", self.default_fallback),
            ]

            for name, func in strategies:
                try:
                    result = func()
                    return result, name
                except Exception as e:
                    continue

            return self.default_value, "none"

    # Тестирование
    scenarios = [
        ("Всё работает", True, True),
        ("Primary упал", False, True),
        ("Всё упало", False, False),
    ]

    for scenario_name, primary_ok, cache_ok in scenarios:
        service = ServiceWithFallback("main")
        service.primary_ok = primary_ok
        service.cache_ok = cache_ok

        result, source = service.get_data()
        print(f"  {scenario_name}:")
        print(f"    Источник: {source}")
        print(f"    Данные: {result}")
        print()

    # --- 2b. Feature flags ---
    print("\n--- 2b. Feature Flags — управление функциональностью ---\n")

    class FeatureFlagSystem:
        """Система feature flags с процентным rollout."""
        def __init__(self):
            self.flags = {}
            self.rollout_hashes = {}

        def set_flag(self, name, enabled, rollout_pct=100):
            """Установка feature flag."""
            self.flags[name] = {
                'enabled': enabled,
                'rollout_pct': rollout_pct
            }

        def is_enabled(self, name, user_id=None):
            """Проверка: включён ли feature для пользователя."""
            if name not in self.flags:
                return False

            flag = self.flags[name]
            if not flag['enabled']:
                return False

            # Процентный rollout на основе user_id
            if flag['rollout_pct'] < 100 and user_id is not None:
                hash_val = int(hashlib.md5(f"{name}:{user_id}".encode()).hexdigest()[:8], 16)
                user_pct = hash_val % 100
                return user_pct < flag['rollout_pct']

            return True

        def get_all_flags(self):
            """Получение всех flags."""
            return self.flags

    flags = FeatureFlagSystem()
    flags.set_flag("new_ui", enabled=True, rollout_pct=25)
    flags.set_flag("dark_mode", enabled=True, rollout_pct=100)
    flags.set_flag("beta_api", enabled=False, rollout_pct=0)

    print(f"  Feature Flags:")
    for name, config in flags.get_all_flags().items():
        print(f"    {name}: enabled={config['enabled']}, rollout={config['rollout_pct']}%")

    # Тестирование для разных пользователей
    print(f"\n  Тест для пользователей:")
    test_users = ["user_001", "user_002", "user_003", "user_004", "user_005"]
    for user in test_users:
        results = {}
        for flag_name in ["new_ui", "dark_mode", "beta_api"]:
            results[flag_name] = "ON" if flags.is_enabled(flag_name, user) else "OFF"
        print(f"    {user}: {results}")

    # --- 2c. Circuit breaker ---
    print("\n--- 2c. Circuit Breaker — разрыв цепи ---\n")

    class CircuitBreaker:
        """Паттерн Circuit Breaker."""
        def __init__(self, failure_threshold=5, recovery_timeout=30):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failure_count = 0
            self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
            self.last_failure_time = None
            self.success_count = 0

        def call(self, func):
            """Вызов функции через circuit breaker."""
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    print(f"    Circuit: OPEN → HALF_OPEN (попытка восстановления)")
                else:
                    print(f"    Circuit: OPEN → отклонено")
                    raise RuntimeError("Circuit breaker OPEN")

            try:
                result = func()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise

        def _on_success(self):
            """Обработка успешного вызова."""
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= 3:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
                    print(f"    Circuit: HALF_OPEN → CLOSED (восстановлен)")
            else:
                self.failure_count = 0

        def _on_failure(self):
            """Обработка неудачного вызова."""
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print(f"    Circuit: CLOSED → OPEN (порог: {self.failure_count})")

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2)

    call_count = 0
    def flaky_service():
        """Нестабильный сервис."""
        nonlocal call_count
        call_count += 1
        if call_count % 4 == 0:
            return "success"
        raise ConnectionError("Service unavailable")

    print(f"  Порог failures: {cb.failure_threshold}")
    print(f"  Recovery timeout: {cb.recovery_timeout} сек")
    print(f"\n  Симуляция вызовов:")

    for i in range(8):
        try:
            result = cb.call(flaky_service)
            print(f"  Вызов {i + 1}: {result} (state={cb.state})")
        except (RuntimeError, ConnectionError) as e:
            print(f"  Вызов {i + 1}: {type(e).__name__} (state={cb.state})")
        time.sleep(0.1)

    # --- 2d. Уровни service ---
    print("\n--- 2d. Multi-tier service с приоритетами ---\n")

    class TieredService:
        """Сервис с приоритетными уровнями."""
        def __init__(self):
            self.tiers = {
                'critical': {'available': True, 'latency': 10, 'weight': 0.5},
                'important': {'available': True, 'latency': 50, 'weight': 0.3},
                'standard': {'available': True, 'latency': 200, 'weight': 0.15},
                'best_effort': {'available': True, 'latency': 1000, 'weight': 0.05},
            }

        def set_availability(self, tier, available):
            """Установка доступности уровня."""
            if tier in self.tiers:
                self.tiers[tier]['available'] = available

        def route_request(self, priority):
            """Маршрутизация запроса по приоритету."""
            # Определяем минимальный уровень для приоритета
            priority_map = {
                'high': ['critical'],
                'medium': ['critical', 'important'],
                'low': ['critical', 'important', 'standard'],
                'any': ['critical', 'important', 'standard', 'best_effort']
            }

            target_tiers = priority_map.get(priority, ['critical'])
            for tier in target_tiers:
                if self.tiers[tier]['available']:
                    return tier, self.tiers[tier]

            return None, None

        def calculate_capacity(self):
            """Расчёт общей доступной ёмкости."""
            total = sum(
                t['weight'] for t in self.tiers.values() if t['available']
            )
            return total

    service = TieredService()

    # Нормальная работа
    print(f"  Нормальная работа:")
    for priority in ['high', 'medium', 'low', 'any']:
        tier, info = service.route_request(priority)
        if tier:
            print(f"    Приоритет '{priority}' → {tier} (latency: {info['latency']} мс)")
        else:
            print(f"    Приоритет '{priority}' → НЕТ ДОСТУПНЫХ УРОВНЕЙ")

    # Деградация: critical недоступен
    print(f"\n  После отказа critical:")
    service.set_availability('critical', False)
    for priority in ['high', 'medium', 'low', 'any']:
        tier, info = service.route_request(priority)
        if tier:
            print(f"    Приоритет '{priority}' → {tier} (latency: {info['latency']} мс)")
        else:
            print(f"    Приоритет '{priority}' → ОТКАЗ")

    print(f"\n  Доступная ёмкость: {service.calculate_capacity():.0%}")


# ========================================================================
# 3. Human-in-the-Loop
# ========================================================================

def demo_human_in_the_loop():
    """Демонстрация систем human-in-the-loop."""
    print("\n" + "=" * 70)
    print("3. HUMAN-IN-THE-LOOP")
    print("=" * 70)

    # --- 3a. Escalation triggers ---
    print("\n--- 3a. Триггеры эскалации ---\n")

    class EscalationSystem:
        """Система автоматической эскалации."""
        def __init__(self):
            self.rules = []
            self.escalation_log = []

        def add_rule(self, name, condition_func, severity, timeout_sec):
            """Добавление правила эскалации."""
            self.rules.append({
                'name': name,
                'condition': condition_func,
                'severity': severity,
                'timeout': timeout_sec
            })

        def check_escalation(self, metrics):
            """Проверка необходимости эскалации."""
            triggered = []
            for rule in self.rules:
                if rule['condition'](metrics):
                    triggered.append({
                        'rule': rule['name'],
                        'severity': rule['severity'],
                        'timeout': rule['timeout'],
                        'timestamp': time.time()
                    })
                    self.escalation_log.append(triggered[-1])
            return triggered

    esc_system = EscalationSystem()

    # Правила эскалации
    esc_system.add_rule(
        "Высокий error rate",
        lambda m: m.get('error_rate', 0) > 0.05,
        severity="critical",
        timeout_sec=300
    )
    esc_system.add_rule(
        "Долгий latency",
        lambda m: m.get('p99_latency', 0) > 500,
        severity="warning",
        timeout_sec=600
    )
    esc_system.add_rule(
        "Низкая доступность",
        lambda m: m.get('availability', 100) < 99.9,
        severity="critical",
        timeout_sec=180
    )
    esc_system.add_rule(
        "Высокая утилизация",
        lambda m: m.get('cpu_usage', 0) > 90,
        severity="warning",
        timeout_sec=900
    )

    # Тестовые метрики
    test_metrics = [
        {"error_rate": 0.08, "p99_latency": 300, "availability": 99.5, "cpu_usage": 75},
        {"error_rate": 0.02, "p99_latency": 600, "availability": 99.95, "cpu_usage": 85},
        {"error_rate": 0.01, "p99_latency": 200, "availability": 99.99, "cpu_usage": 95},
    ]

    for i, metrics in enumerate(test_metrics):
        triggered = esc_system.check_escalation(metrics)
        print(f"  Метрики #{i + 1}: error_rate={metrics['error_rate']}, "
              f"p99={metrics['p99_latency']}ms, avail={metrics['availability']}%")
        if triggered:
            for t in triggered:
                print(f"    ЭСКАЛАЦИЯ: [{t['severity']}] {t['rule']} (timeout: {t['timeout']}с)")
        else:
            print(f"    Эскалация не требуется")

    # --- 3b. Approval queue ---
    print("\n--- 3b. Очередь одобрений (Approval Queue) ---\n")

    class ApprovalQueue:
        """Очередь на одобрение операций."""
        def __init__(self):
            self.queue = []
            self.approved = []
            self.rejected = []

        def submit(self, operation, requester, risk_level, description):
            """Подача заявки на одобрение."""
            item = {
                'id': hashlib.md5(f"{operation}{time.time()}".encode()).hexdigest()[:8],
                'operation': operation,
                'requester': requester,
                'risk_level': risk_level,
                'description': description,
                'status': 'pending',
                'timestamp': time.time()
            }
            self.queue.append(item)
            return item['id']

        def approve(self, item_id, approver):
            """Одобрение операции."""
            for item in self.queue:
                if item['id'] == item_id:
                    item['status'] = 'approved'
                    item['approver'] = approver
                    self.approved.append(item)
                    return True
            return False

        def reject(self, item_id, approver, reason):
            """Отклонение операции."""
            for item in self.queue:
                if item['id'] == item_id:
                    item['status'] = 'rejected'
                    item['approver'] = approver
                    item['reason'] = reason
                    self.rejected.append(item)
                    return True
            return False

        def get_pending(self):
            """Получение ожидающих операций."""
            return [i for i in self.queue if i['status'] == 'pending']

    approval_q = ApprovalQueue()

    # Подача заявок
    ops = [
        ("deploy", "CI/CD", "high", "Деплой v2.1 в production"),
        ("config_change", "DevOps", "medium", "Изменение лимитов DB"),
        ("scale_up", "Auto-scaler", "low", "Увеличение Instances до 10"),
        ("rollback", "Incident", "critical", "Откат к v2.0 из-за бага"),
    ]

    print(f"  Подача заявок на одобрение:")
    for op, requester, risk, desc in ops:
        item_id = approval_q.submit(op, requester, risk, desc)
        print(f"    [{item_id}] {op}: {desc} (риск: {risk})")

    pending = approval_q.get_pending()
    print(f"\n  Ожидающих: {len(pending)}")

    # Одобрение части заявок
    approval_q.approve(pending[0]['id'], "Team Lead")
    approval_q.approve(pending[2]['id'], "SRE Manager")
    approval_q.reject(pending[3]['id'], "CTO", "Сначала нужен постmortem")

    print(f"\n  Результаты:")
    for item in approval_q.approved:
        print(f"    ОДОБРЕНО: {item['operation']} by {item['approver']}")
    for item in approval_q.rejected:
        print(f"    ОТКЛОНЕНО: {item['operation']} — {item['reason']}")

    # --- 3c. Feedback loops ---
    print("\n--- 3c. Петли обратной связи (Feedback Loops) ---\n")

    class FeedbackLoop:
        """Система обратной связи для autonomous decision."""
        def __init__(self):
            self.decisions = []
            self.feedback = []

        def record_decision(self, decision_type, context, action):
            """Запись решения."""
            record = {
                'id': len(self.decisions),
                'type': decision_type,
                'context': context,
                'action': action,
                'timestamp': time.time()
            }
            self.decisions.append(record)
            return record['id']

        def submit_feedback(self, decision_id, rating, comment):
            """Отправка обратной связи."""
            fb = {
                'decision_id': decision_id,
                'rating': rating,
                'comment': comment,
                'timestamp': time.time()
            }
            self.feedback.append(fb)

        def get_accuracy(self):
            """Расчёт точности решений."""
            if not self.feedback:
                return None
            positive = sum(1 for f in self.feedback if f['rating'] > 0)
            return positive / len(self.feedback)

        def get_pattern(self):
            """Поиск паттернов в обратной связи."""
            type_ratings = {}
            for fb in self.feedback:
                decision = self.decisions[fb['decision_id']]
                dtype = decision['type']
                if dtype not in type_ratings:
                    type_ratings[dtype] = []
                type_ratings[dtype].append(fb['rating'])
            return {k: statistics.mean(v) for k, v in type_ratings.items()}

    loop = FeedbackLoop()

    # Запись решений и обратной связи
    decisions_data = [
        ("auto_scale", "cpu > 80%", "scale_up"),
        ("auto_scale", "cpu > 80%", "scale_up"),
        ("auto_heal", "pod crashed", "restart"),
        ("auto_heal", "pod crashed", "restart"),
        ("auto_deploy", "tests passed", "deploy"),
        ("auto_deploy", "tests passed", "deploy"),
        ("auto_rollback", "error_rate > 5%", "rollback"),
    ]

    for dtype, context, action in decisions_data:
        loop.record_decision(dtype, context, action)

    # Обратная связь (4-е решение было неправильным)
    feedback_data = [
        (0, 1, "Масштабирование помогло"),
        (1, 1, "Работает"),
        (2, 1, "Перезапуск решил проблему"),
        (3, 1, "Стабильно"),
        (4, -1, "Деплой сломал staging"),
        (5, 1, "Успешно"),
        (6, 1, "Откат спас ситуацию"),
    ]

    for did, rating, comment in feedback_data:
        loop.submit_feedback(did, rating, comment)

    accuracy = loop.get_accuracy()
    patterns = loop.get_pattern()

    print(f"  Всего решений: {len(loop.decisions)}")
    print(f"  Обратная связь: {len(loop.feedback)}")
    print(f"  Точность решений: {accuracy:.1%}")

    print(f"\n  Средний рейтинг по типам:")
    for dtype, avg_rating in patterns.items():
        emoji = "+" if avg_rating > 0 else "-" if avg_rating < 0 else "="
        print(f"    {dtype}: {avg_rating:+.2f} {emoji}")

    # --- 3d. Approval requirements ---
    print("\n--- 3d. Динамические требования к одобрению ---\n")

    class DynamicApprovalPolicy:
        """Динамическая политика одобрения на основе риска."""
        def __init__(self):
            self.policies = {}

        def add_policy(self, operation, min_approvers, required_roles, auto_approve_below=None):
            """Добавление политики."""
            self.policies[operation] = {
                'min_approvers': min_approvers,
                'required_roles': required_roles,
                'auto_approve_below': auto_approve_below
            }

        def check_approval(self, operation, approvers, risk_score=0):
            """Проверка достаточности одобрений."""
            policy = self.policies.get(operation)
            if not policy:
                return True, "Нет политики — одобрено"

            # Автоодобрение при низком риске
            if policy['auto_approve_below'] and risk_score < policy['auto_approve_below']:
                return True, f"Автоодобрение (риск {risk_score:.2f} < {policy['auto_approve_below']})"

            # Проверка количества
            if len(approvers) < policy['min_approvers']:
                return False, f"Нужно {policy['min_approvers']} одобрений, имеется {len(approvers)}"

            # Проверка ролей
            approver_roles = set(a['role'] for a in approvers)
            missing_roles = set(policy['required_roles']) - approver_roles
            if missing_roles:
                return False, f"Отсутствуют роли: {missing_roles}"

            return True, "Все требования выполнены"

    policy = DynamicApprovalPolicy()
    policy.add_policy("deploy_production", min_approvers=2,
                      required_roles=["sre", "tech_lead"], auto_approve_below=0.3)
    policy.add_policy("database_migration", min_approvers=3,
                      required_roles=["dba", "sre", "security"], auto_approve_below=None)
    policy.add_policy("config_change", min_approvers=1,
                      required_roles=["devops"], auto_approve_below=0.5)

    # Тестовые сценарии
    scenarios = [
        ("deploy_production", [{"name": "Alice", "role": "sre"},
                                {"name": "Bob", "role": "tech_lead"}], 0.1),
        ("deploy_production", [{"name": "Alice", "role": "sre"}], 0.5),
        ("database_migration", [{"name": "Dave", "role": "dba"},
                                 {"name": "Eve", "role": "sre"}], 0.2),
        ("config_change", [], 0.3),
    ]

    for op, approvers, risk in scenarios:
        approved, reason = policy.check_approval(op, approvers, risk)
        print(f"  {op} (риск={risk:.2f}):")
        print(f"    Одобрения: {[a['name'] for a in approvers] if approvers else 'нет'}")
        print(f"    Результат: {'ОДОБРЕНО' if approved else 'ОТКЛОНЕНО'}")
        print(f"    Причина: {reason}")
        print()


# ========================================================================
# 4. Autonomy Governance
# ========================================================================

def demo_autonomy_governance():
    """Демонстрация систем governance для автономных систем."""
    print("\n" + "=" * 70)
    print("4. AUTONOMY GOVERNANCE")
    print("=" * 70)

    # --- 4a. Audit trail ---
    print("\n--- 4a. Журнал аудита (Audit Trail) ---\n")

    class AuditTrail:
        """Журнал аудита для отслеживания действий."""
        def __init__(self):
            self.entries = []

        def log(self, actor, action, resource, details=None, outcome="success"):
            """Запись действия в журнал."""
            entry = {
                'timestamp': time.time(),
                'actor': actor,
                'action': action,
                'resource': resource,
                'details': details or {},
                'outcome': outcome,
                'hash': None
            }
            # Цепочка хэшей
            prev_hash = self.entries[-1]['hash'] if self.entries else "0" * 64
            entry['hash'] = hashlib.sha256(
                f"{prev_hash}{json.dumps(entry, default=str)}".encode()
            ).hexdigest()[:16]
            self.entries.append(entry)
            return entry

        def verify_integrity(self):
            """Проверка целостности журнала."""
            prev_hash = "0" * 16
            for i, entry in enumerate(self.entries):
                expected_hash = entry['hash']
                # Пересчитываем хэш
                recomputed = hashlib.sha256(
                    f"{prev_hash}{json.dumps({k: v for k, v in entry.items() if k != 'hash'}, default=str)}".encode()
                ).hexdigest()[:16]
                if recomputed != expected_hash:
                    return False, i
                prev_hash = expected_hash
            return True, len(self.entries)

        def query(self, actor=None, action=None):
            """Запрос журнала по фильтрам."""
            results = self.entries
            if actor:
                results = [e for e in results if e['actor'] == actor]
            if action:
                results = [e for e in results if e['action'] == action]
            return results

    audit = AuditTrail()

    # Записи аудита
    actions = [
        ("deploy-bot", "deploy", "service-api", {"version": "2.1.0"}, "success"),
        ("sre-alice", "approve", "deploy-api-2.1", {"reason": "tests_pass"}, "success"),
        ("auto-healer", "restart", "pod-xyz-123", {"reason": "OOM"}, "success"),
        ("deploy-bot", "deploy", "service-api", {"version": "2.1.1"}, "success"),
        ("sre-bob", "rollback", "service-api", {"from": "2.1.1", "to": "2.1.0"}, "success"),
    ]

    for actor, action, resource, details, outcome in actions:
        audit.log(actor, action, resource, details, outcome)

    print(f"  Записей в журнале: {len(audit.entries)}")
    print(f"\n  Журнал аудита:")
    for entry in audit.entries:
        print(f"    [{entry['hash']}] {entry['actor']} → {entry['action']} "
              f"({entry['resource']}) — {entry['outcome']}")

    # Проверка целостности
    valid, count = audit.verify_integrity()
    print(f"\n  Целостность: {'OK' if valid else 'НАРУШЕНА'} ({count} записей)")

    # Запросы
    deploys = audit.query(action="deploy")
    print(f"\n  Деплои: {len(deploys)}")
    alice_actions = audit.query(actor="sre-alice")
    print(f"  Действия alice: {len(alice_actions)}")

    # --- 4b. Compliance checks ---
    print("\n--- 4b. Проверки соответствия (Compliance Checks) ---\n")

    class ComplianceChecker:
        """Проверка соответствия политикам."""
        def __init__(self):
            self.rules = []
            self.violations = []

        def add_rule(self, name, check_func, severity, description):
            """Добавление правила."""
            self.rules.append({
                'name': name,
                'check': check_func,
                'severity': severity,
                'description': description
            })

        def check(self, target):
            """Проверка всех правил."""
            results = []
            for rule in self.rules:
                passed = rule['check'](target)
                result = {
                    'rule': rule['name'],
                    'passed': passed,
                    'severity': rule['severity'],
                    'description': rule['description']
                }
                results.append(result)
                if not passed:
                    self.violations.append(result)
            return results

    checker = ComplianceChecker()

    # Правила compliance
    checker.add_rule(
        "no_hardcoded_secrets",
        lambda t: "password" not in t.get('config', '').lower() or
                  "env:" in t.get('config', '').lower(),
        severity="critical",
        description="Запрет хардкода секретов в конфигурации"
    )
    checker.add_rule(
        "encryption_at_rest",
        lambda t: t.get('encrypted', False),
        severity="high",
        description="Данные должны быть зашифрованы"
    )
    checker.add_rule(
        "backup_enabled",
        lambda t: t.get('backup', {}).get('enabled', False),
        severity="medium",
        description="Должны быть включены бэкапы"
    )
    checker.add_rule(
        "logging_enabled",
        lambda t: t.get('logging', {}).get('level') in ['INFO', 'DEBUG'],
        severity="low",
        description="Логирование должно быть включено"
    )

    # Тестовые конфигурации
    configs = [
        {
            'name': 'service-a',
            'config': 'DB_HOST=env:DB_HOST',
            'encrypted': True,
            'backup': {'enabled': True},
            'logging': {'level': 'INFO'}
        },
        {
            'name': 'service-b',
            'config': 'password=hardcoded_secret',
            'encrypted': False,
            'backup': {'enabled': False},
            'logging': {'level': 'WARNING'}
        },
    ]

    for config in configs:
        results = checker.check(config)
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        print(f"  {config['name']}: {passed}/{total} правил пройдено")
        for r in results:
            status = "OK" if r['passed'] else "VIOLATION"
            print(f"    [{status:>8}] [{r['severity']:>8}] {r['rule']}: {r['description']}")
        print()

    print(f"  Всего нарушений: {len(checker.violations)}")

    # --- 4c. Review board simulation ---
    print("\n--- 4c. Simulated Review Board ---\n")

    class ReviewBoard:
        """Симуляция совета по ревью автономных решений."""
        def __init__(self):
            self.submissions = []
            self.decisions = []

        def submit(self, proposal, risk_level, author):
            """Подача предложения на ревью."""
            submission = {
                'id': len(self.submissions),
                'proposal': proposal,
                'risk_level': risk_level,
                'author': author,
                'status': 'pending'
            }
            self.submissions.append(submission)
            return submission['id']

        def review(self, submission_id, reviewer, approve, comments=""):
            """Ревью предложения."""
            if submission_id < len(self.submissions):
                self.submissions[submission_id]['status'] = 'approved' if approve else 'rejected'
                self.decisions.append({
                    'submission_id': submission_id,
                    'reviewer': reviewer,
                    'approve': approve,
                    'comments': comments
                })

        def get_pending(self):
            """Ожидающие ревью."""
            return [s for s in self.submissions if s['status'] == 'pending']

    board = ReviewBoard()

    proposals = [
        ("Автоматический деплой в prod при зелёных тестах", "medium", "DevOps"),
        ("Самостоятельное масштабирование БД", "high", "SRE"),
        ("Автооткат при росте error rate > 10%", "critical", "Platform"),
        ("Автоматическая очистка логов > 30 дней", "low", "SRE"),
    ]

    print(f"  Поданные предложения:")
    for proposal, risk, author in proposals:
        sid = board.submit(proposal, risk, author)
        print(f"    [{sid}] ({risk}) {proposal} — {author}")

    # Ревью
    board.review(0, "SRE Lead", True, "Хорошая идея, тесты покрывают критические сценарии")
    board.review(1, "DBA Lead", False, "Риск данных, нужен approval от DBA")
    board.review(2, "CTO", True, "Критически важно, одобряю")
    board.review(3, "SRE Lead", True, "Безопасная операция")

    print(f"\n  Решения:")
    for d in board.decisions:
        status = "ОДОБРЕНО" if d['approve'] else "ОТКЛОНЕНО"
        print(f"    Proposal #{d['submission_id']}: {status}")
        print(f"      Ревьюер: {d['reviewer']}")
        print(f"      Комментарий: {d['comments']}")

    # --- 4d. Autonomy levels ---
    print("\n--- 4d. Уровни автономии (Autonomy Levels) ---\n")

    autonomy_levels = {
        0: ("Manual", "Все действия выполняются вручную",
            ["Ручное принятие всех решений", "Нет автоматизации"]),
        1: ("Advisory", "Система рекомендует, человек решает",
            ["Рекомендации по действиям", "Человек подтверждает каждое"]),
        2: ("Supervised", "Автоматизация с human approval",
            ["Авто-действия для низкого риска", "Approval для среднего/высокого"]),
        3: ("Conditional", "Автономно в рамках условий",
            ["Автономно в пределах политик", "Эскалация при отклонениях"]),
        4: ("High Autonomy", "Автономно с мониторингом",
            ["Широкая автономия", "Мониторинг и алерты"]),
        5: ("Full Autonomy", "Полная автономность",
            ["Непрерывная работа", "Человек только для edge cases"]),
    }

    print(f"  Модель уровней автономии:")
    print(f"  {'Ур.':>3} | {'Название':<18} | {'Описание':<45}")
    print(f"  {'-'*3}-+-{'-'*18}-+-{'-'*45}")
    for level, (name, desc, _) in autonomy_levels.items():
        print(f"  {level:>3} | {name:<18} | {desc}")

    print(f"\n  Детали по уровням:")
    for level, (name, _, features) in autonomy_levels.items():
        print(f"\n  Level {level} — {name}:")
        for feature in features:
            print(f"    • {feature}")

    # Пример: текущая система на уровне 2
    current_level = 2
    print(f"\n  Текущий уровень системы: {current_level}")
    name, desc, features = autonomy_levels[current_level]
    print(f"  {name}: {desc}")
    print(f"  Возможности: {features[0]}")


# ========================================================================
# Точка входа
# ========================================================================

if __name__ == "__main__":
    demo_sla_compliance()
    demo_graceful_degradation()
    demo_human_in_the_loop()
    demo_autonomy_governance()
