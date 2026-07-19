"""179 — Human-AI Teaming: протоколы передачи, калибровка доверия, эскалация

Темы:
  1. Handoff Protocols — когда передавать, передача контекста, пороги уверенности
  2. Trust Calibration — неопределённость, объяснения, кривые калибровки
  3. Escalation Strategies — уровни серьёзности, маршрутизация, SLA
  4. Collaborative Patterns — циклы проверки, согласование, смешанная инициатива

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
# 1. Handoff Protocols — протоколы передачи управления
# =============================================================================

def demo_handoff_protocols():
    """Демонстрация протоколов передачи управления между человеком и ИИ."""
    print("=" * 70)
    print("DEMO 1: Handoff Protocols — протоколы передачи")
    print("=" * 70)

    # --- 1.1 Определение момента передачи ---
    print("\n--- 1.1 Определение момента передачи (Handoff Triggers) ---")

    class HandoffDecision:
        """Принятие решения о передаче управления."""

        def __init__(self, confidence_threshold=0.7, max_retries=3):
            self.threshold = confidence_threshold
            self.max_retries = max_retries
            self.history = []

        def should_handoff(self, task, confidence, context):
            """Решение: передавать ли управление человеку."""
            reasons = []

            # Проверка порога уверенности
            if confidence < self.threshold:
                reasons.append(f"низкая уверенность ({confidence:.2f} < {self.threshold})")

            # Проверка количества повторов
            retries = context.get("retries", 0)
            if retries >= self.max_retries:
                reasons.append(f"превышен лимит повторов ({retries} >= {self.max_retries})")

            # Проверка критичности задачи
            if context.get("criticality") == "high":
                reasons.append("высокая критичность задачи")

            # Проверка наличия неопределённости в данных
            if context.get("data_quality") == "low":
                reasons.append("низкое качество данных")

            handoff = len(reasons) > 0
            decision = {
                "task": task,
                "confidence": confidence,
                "handoff": handoff,
                "reasons": reasons,
            }
            self.history.append(decision)
            return decision

    decider = HandoffDecision(confidence_threshold=0.75)

    tasks = [
        ("классификация email", 0.92, {"retries": 0, "criticality": "low"}),
        ("отмена заказа", 0.65, {"retries": 0, "criticality": "high"}),
        ("генерация отчёта", 0.70, {"retries": 2, "criticality": "medium"}),
        ("обработка возврата", 0.55, {"retries": 0, "criticality": "high", "data_quality": "low"}),
    ]

    for task_name, conf, ctx in tasks:
        result = decider.should_handoff(task_name, conf, ctx)
        action = "ПЕРЕДАЧА ЧЕЛОВЕКУ" if result["handoff"] else "АВТОНОМНО"
        print(f"  Задача: '{task_name}'")
        print(f"    Уверенность: {conf:.2f} → {action}")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"    Причина: {r}")
        print()

    # --- 1.2 Передача контекста ---
    print("\n--- 1.2 Передача контекста (Context Transfer) ---")

    class ContextTransfer:
        """Упаковка контекста для передачи человеку."""

        def __init__(self):
            self.templates = {}

        def register_template(self, task_type, template):
            """Зарегистрировать шаблон контекста для типа задачи."""
            self.templates[task_type] = template

        def package(self, task_type, data):
            """Упаковать контекст по шаблону."""
            template = self.templates.get(task_type, {})
            package = {}

            for field, source in template.items():
                if callable(source):
                    package[field] = source(data)
                elif source in data:
                    package[field] = data[source]
                else:
                    package[field] = None

            return package

        def format_handoff(self, package):
            """Форматировать контекст для человека."""
            lines = ["  ── КОНТЕКСТ ДЛЯ ОПЕРАТОРА ──"]
            for key, value in package.items():
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    - {k}: {v}")
                elif isinstance(value, list):
                    lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"  {key}: {value}")
            return "\n".join(lines)

    transfer = ContextTransfer()

    # Регистрируем шаблон для жалоб клиентов
    transfer.register_template("complaint", {
        "клиент": "customer_name",
        "проблема": "issue_description",
        "приоритет": lambda d: "КРИТИЧЕСКИЙ" if d.get("amount", 0) > 100 else "обычный",
        "предыдущие_действия": lambda d: d.get("attempts", []),
        "рекомендация_ИИ": "ai_suggestion",
    })

    complaint_data = {
        "customer_name": "Иванов И.И.",
        "issue_description": "Двойное списание средств",
        "amount": 250,
        "attempts": ["авто-ответ", "переключение на tier-2"],
        "ai_suggestion": "Вернуть сумму и применить компенсацию 10%",
    }

    package = transfer.package("complaint", complaint_data)
    print(transfer.format_handoff(package))

    # --- 1.3 Пороги уверенности ---
    print("\n--- 1.3 Пороги уверенности (Confidence Thresholds) ---")

    class ConfidencePolicy:
        """Политика действий на основе уровня уверенности."""

        def __init__(self):
            self.zones = []

        def add_zone(self, min_conf, max_conf, action, description):
            """Добавить зону уверенности."""
            self.zones.append({
                "range": (min_conf, max_conf),
                "action": action,
                "description": description,
            })

        def get_action(self, confidence):
            """Определить действие по уровню уверенности."""
            for zone in self.zones:
                lo, hi = zone["range"]
                if lo <= confidence < hi:
                    return zone
            return None

    policy = ConfidencePolicy()
    policy.add_zone(0.0, 0.3, "ESCALATE", "Критически низкая — немедленная эскалация")
    policy.add_zone(0.3, 0.5, "HUMAN_REVIEW", "Низкая — требуется проверка человека")
    policy.add_zone(0.5, 0.7, "SUPERVISED", "Средняя —执行 с контролем")
    policy.add_zone(0.7, 0.9, "AUTO_WITH_LOG", "Хорошая — автономно с логированием")
    policy.add_zone(0.9, 1.01, "FULL_AUTO", "Отличная — полная автономность")

    test_confidences = [0.15, 0.42, 0.63, 0.78, 0.95]
    for conf in test_confidences:
        zone = policy.get_action(conf)
        print(f"  Уверенность {conf:.2f}: {zone['action']} — {zone['description']}")

    # --- 1.4 Протоколы возврата ---
    print("\n--- 1.4 Протокол возврата (Return Protocol) ---")

    class ReturnProtocol:
        """Протокол возврата управления от человека к ИИ."""

        def __init__(self):
            self.history = []

        def human_resolves(self, task, resolution, notes=""):
            """Человек разрешил задачу."""
            entry = {
                "task": task,
                "resolution": resolution,
                "notes": notes,
                "returned": True,
            }
            self.history.append(entry)
            return entry

        def learn_from_resolution(self, resolution):
            """Извлечь урок из решения человека."""
            lessons = []
            if "одобрено" in resolution.lower():
                lessons.append("кейс одобрен — возможно, расширить автоматизацию")
            if "отклонено" in resolution.lower():
                lessons.append("кейс отклонён — усилить проверки")
            if "изменено" in resolution.lower():
                lessons.append("кейс изменён — обновить правила")
            return lessons

    protocol = ReturnProtocol()

    resolutions = [
        ("возврат средств #1234", "одобritten полностью", "сумма < 100"),
        ("жалоба на сервис", "отклонено — клиент неправ", "нарушение policy"),
        ("запрос на изменение тарифа", "изменено — предложен альтернативный план", ""),
    ]

    for task, resolution, notes in resolutions:
        entry = protocol.human_resolves(task, resolution, notes)
        lessons = protocol.learn_from_resolution(resolution)
        print(f"  Задача: {task}")
        print(f"    Решение: {resolution}")
        if lessons:
            for lesson in lessons:
                print(f"    Урок: {lesson}")


# =============================================================================
# 2. Trust Calibration — калибровка доверия
# =============================================================================

def demo_trust_calibration():
    """Демонстрация калибровки доверия к ИИ-системе."""
    print("\n" + "=" * 70)
    print("DEMO 2: Trust Calibration — калибровка доверия")
    print("=" * 70)

    # --- 2.1 Коммуникация неопределённости ---
    print("\n--- 2.1 Коммуникация неопределённости ---")

    class UncertaintyCommunicator:
        """Формулирование неопределённости для пользователя."""

        def __init__(self):
            self.calibration = {
                (0.0, 0.3): "я не уверен в этом",
                (0.3, 0.5): "есть значительные сомнения",
                (0.5, 0.7): "возможна ошибка",
                (0.7, 0.85): "в целом可靠но, но проверьте",
                (0.85, 1.0): "высокая уверенность",
            }

        def communicate(self, confidence, prediction, alternatives=None):
            """Сформулировать ответ с учётом неопределённости."""
            phrase = ""
            for (lo, hi), text in self.calibration.items():
                if lo <= confidence < hi:
                    phrase = text
                    break

            response = {
                "prediction": prediction,
                "confidence": confidence,
                "phrase": phrase,
                "alternatives": alternatives or [],
                "recommendation": self._recommendation(confidence),
            }
            return response

        def _recommendation(self, confidence):
            """Рекомендация по действию."""
            if confidence < 0.5:
                return "рекомендуется ручная проверка"
            elif confidence < 0.8:
                return "можно доверять, но перепроверьте критичные данные"
            else:
                return "можно действовать автономно"

    communicator = UncertaintyCommunicator()

    predictions = [
        (0.92, "Клиент доволен", [("нейтрально", 0.06), ("недоволен", 0.02)]),
        (0.45, "Спам", [("legitimate", 0.35), ("promotions", 0.20)]),
        (0.68, "Высокий риск оттока", [("средний риск", 0.22), ("низкий", 0.10)]),
        (0.25, "Текущий статус: active", [("inactive", 0.45), ("pending", 0.30)]),
    ]

    for conf, pred, alts in predictions:
        result = communicator.communicate(conf, pred, alts)
        print(f"  Предсказание: '{pred}' (уверенность: {conf:.0%})")
        print(f"    Фраза: '{result['phrase']}'")
        print(f"    Рекомендация: {result['recommendation']}")
        if alts:
            alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alts)
            print(f"    Альтернативы: {alt_str}")
        print()

    # --- 2.2 Кривые калибровки ---
    print("\n--- 2.2 Кривые калибровки (Calibration Curves) ---")

    def compute_calibration(predictions, actuals, n_bins=5):
        """Вычислить калибровочные бины.

        Показывает, насколько предсказанные вероятности
        соответствуют реальной частоте положительных исходов.
        """
        bins = [[] for _ in range(n_bins)]
        bin_edges = [i / n_bins for i in range(n_bins + 1)]

        for pred, actual in zip(predictions, actuals):
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx].append((pred, actual))

        calibration = []
        for i, bin_data in enumerate(bins):
            if bin_data:
                mean_pred = sum(p for p, _ in bin_data) / len(bin_data)
                mean_actual = sum(a for _, a in bin_data) / len(bin_data)
                count = len(bin_data)
                calibration.append({
                    "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                    "mean_predicted": mean_pred,
                    "mean_actual": mean_actual,
                    "count": count,
                    "gap": abs(mean_pred - mean_actual),
                })

        return calibration

    # Имитируем предсказания и реальные исходы
    random.seed(42)
    n_samples = 200
    predictions = [random.random() for _ in range(n_samples)]
    # Добавляем шум к реальным исходам (хорошо калиброванные)
    actuals = [1 if random.random() < p + random.gauss(0, 0.1) else 0
               for p in predictions]
    actuals = [max(0, min(1, a)) for a in actuals]  # ограничиваем

    calibration = compute_calibration(predictions, actuals, n_bins=5)

    print("  Калибровочная таблица:")
    print(f"  {'Бин':12s} {'Предсказано':>12s} {'Фактически':>12s} {'Разница':>10s} {'N':>5s}")
    print("  " + "-" * 55)
    for entry in calibration:
        print(f"  {entry['bin']:12s} {entry['mean_predicted']:12.3f} "
              f"{entry['mean_actual']:12.3f} {entry['gap']:10.3f} {entry['count']:5d}")

    # Оценка калибровки (Expected Calibration Error)
    total_samples = sum(e["count"] for e in calibration)
    ece = sum(e["gap"] * e["count"] for e in calibration) / total_samples
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    if ece < 0.05:
        print("  Оценка: хорошо калибрована")
    elif ece < 0.1:
        print("  Оценка: приемлемая калибровка")
    else:
        print("  Оценка: нужна калибровка (рекалибровка)")

    # --- 2.3 Объяснимость решений ---
    print("\n--- 2.3 Объяснимость решений (Explainability) ---")

    class DecisionExplainer:
        """Генерация объяснений для решений модели."""

        def __init__(self):
            self.feature_names = []

        def explain_classification(self, features, weights, prediction):
            """Объяснение решения классификатора."""
            # Важность признаков = |вес * значение|
            importances = []
            for name, feat_val, weight in zip(self.feature_names, features, weights):
                importance = abs(feat_val * weight)
                direction = "POSITIVELY" if feat_val * weight > 0 else "NEGATIVELY"
                importances.append({
                    "feature": name,
                    "value": feat_val,
                    "weight": weight,
                    "importance": importance,
                    "direction": direction,
                })

            # Сортируем по важности
            importances.sort(key=lambda x: x["importance"], reverse=True)

            explanation = {
                "prediction": prediction,
                "top_factors": importances[:3],
                "formula": self._build_formula(importances[:3]),
            }
            return explanation

        def _build_formula(self, top_factors):
            """Построить формулу объяснения."""
            parts = []
            for f in top_factors:
                sign = "+" if f["direction"] == "POSITIVELY" else "-"
                parts.append(f"{sign}{f['value']:.2f}*{f['weight']:.2f}")
            return " ".join(parts)

    explainer = DecisionExplainer()
    explainer.feature_names = ["цена", "рейтинг", "отзывы", "реклама", "сезон"]

    cases = [
        {
            "features": [0.8, 0.9, 0.7, 0.3, 0.5],
            "weights": [0.3, 0.4, 0.2, 0.1, 0.05],
            "prediction": "покупка",
        },
        {
            "features": [0.2, 0.4, 0.3, 0.1, 0.8],
            "weights": [0.3, 0.4, 0.2, 0.1, 0.05],
            "prediction": "отказ",
        },
    ]

    for case in cases:
        explanation = explainer.explain_classification(
            case["features"], case["weights"], case["prediction"]
        )
        print(f"  Предсказание: {explanation['prediction']}")
        print(f"  Ключевые факторы:")
        for f in explanation["top_factors"]:
            print(f"    {f['feature']:10s}: значение={f['value']:.2f}, "
                  f"вес={f['weight']:.2f}, влияние={f['direction']}")
        print(f"  Формула: {explanation['formula']}")
        print()

    # --- 2.4 Метрики доверия ---
    print("\n--- 2.4 Метрики доверия (Trust Metrics) ---")

    class TrustMetrics:
        """Отслеживание метрик доверия пользователя к системе."""

        def __init__(self):
            self.interactions = []

        def record(self, accepted, confidence, task_type):
            """Записать взаимодействие."""
            self.interactions.append({
                "accepted": accepted,
                "confidence": confidence,
                "task_type": task_type,
            })

        def acceptance_rate(self):
            """Процент принятия рекомендаций."""
            if not self.interactions:
                return 0
            accepted = sum(1 for i in self.interactions if i["accepted"])
            return accepted / len(self.interactions)

        def over_trust_score(self):
            """Оценка избыточного доверия (принятие при низкой уверенности)."""
            low_conf = [i for i in self.interactions if i["confidence"] < 0.5]
            if not low_conf:
                return 0
            accepted_low = sum(1 for i in low_conf if i["accepted"])
            return accepted_low / len(low_conf)

        def under_trust_score(self):
            """Оценка недоверия (отклонение при высокой уверенности)."""
            high_conf = [i for i in self.interactions if i["confidence"] > 0.8]
            if not high_conf:
                return 0
            rejected_high = sum(1 for i in high_conf if not i["accepted"])
            return rejected_high / len(high_conf)

    metrics = TrustMetrics()

    # Имитируем взаимодействия
    for _ in range(50):
        conf = random.random()
        accepted = random.random() < (conf * 0.8 + 0.1)  # более вероятно принять при высокой confident
        task = random.choice(["email", "order", "support"])
        metrics.record(accepted, conf, task)

    print(f"  Всего взаимодействий: {len(metrics.interactions)}")
    print(f"  Процент принятия: {metrics.acceptance_rate():.1%}")
    print(f"  Избыточное доверие (принятие при confident<0.5): {metrics.over_trust_score():.1%}")
    print(f"  Недоверие (отклонение при confident>0.8): {metrics.under_trust_score():.1%}")


# =============================================================================
# 3. Escalation Strategies — стратегии эскалации
# =============================================================================

def demo_escalation_strategies():
    """Демонстрация стратегий эскалации в человеко-ИИ взаимодействии."""
    print("\n" + "=" * 70)
    print("DEMO 3: Escalation Strategies — стратегии эскалации")
    print("=" * 70)

    # --- 3.1 Уровни серьёзности ---
    print("\n--- 3.1 Уровни серьёзности (Severity Levels) ---")

    class SeverityClassifier:
        """Классификация инцидентов по серьёзности."""

        SEVERITY_LEVELS = {
            1: {"name": "INFO", "response_time": "24ч", "action": "логирование"},
            2: {"name": "LOW", "response_time": "8ч", "action": "автоматическое исправление"},
            3: {"name": "MEDIUM", "response_time": "4ч", "action": "уведомление команды"},
            4: {"name": "HIGH", "response_time": "1ч", "action": "немедленное вмешательство"},
            5: {"name": "CRITICAL", "response_time": "15мин", "action": "всех на борт"},
        }

        def __init__(self):
            self.rules = []

        def add_rule(self, condition, severity):
            """Добавить правило классификации."""
            self.rules.append((condition, severity))

        def classify(self, incident):
            """Определить серьёзность инцидента."""
            for condition, severity in self.rules:
                if condition(incident):
                    return severity
            return 1  # по умолчанию INFO

        def get_response(self, severity):
            """Получить параметры реагирования."""
            return self.SEVERITY_LEVELS.get(severity, self.SEVERITY_LEVELS[1])

    classifier = SeverityClassifier()

    # Правила классификации
    classifier.add_rule(lambda i: "потеря_данных" in i, 5)
    classifier.add_rule(lambda i: i.get("users_affected", 0) > 1000, 5)
    classifier.add_rule(lambda i: i.get("revenue_impact", 0) > 10000, 4)
    classifier.add_rule(lambda i: "безопасность" in i, 4)
    classifier.add_rule(lambda i: i.get("users_affected", 0) > 100, 3)
    classifier.add_rule(lambda i: i.get("revenue_impact", 0) > 1000, 3)
    classifier.add_rule(lambda i: "производительность" in i, 2)

    incidents = [
        {"type": "ошибка_в_логине", "users_affected": 50},
        {"type": "потеря_данных", "users_affected": 10},
        {"type": "медленная_загрузка", "revenue_impact": 5000},
        {"type": "баг_в_UI", "users_affected": 10},
        {"type": "утечка_данных", "users_affected": 5000},
    ]

    for incident in incidents:
        severity = classifier.classify(incident)
        response = classifier.get_response(severity)
        print(f"  Инцидент: {incident['type']}")
        print(f"    Серьёзность: {severity} ({response['name']})")
        print(f"    Время реакции: {response['response_time']}")
        print(f"    Действие: {response['action']}")
        print()

    # --- 3.2 Маршрутизация ---
    print("\n--- 3.2 Маршрутизация (Routing) ---")

    class IncidentRouter:
        """Маршрутизация инцидентов по командам/специалистам."""

        def __init__(self):
            self.routes = []
            self.load = {}  # agent → количество активных задач

        def add_route(self, pattern, team, priority=1):
            """Добавить маршрут."""
            self.routes.append({"pattern": pattern, "team": team, "priority": priority})

        def route(self, incident):
            """Найти лучший маршрут для инцидента."""
            matches = []
            for route in self.routes:
                if re.search(route["pattern"], incident.get("type", "")):
                    matches.append(route)

            if not matches:
                return {"team": "L1-support", "reason": "нет подходящего маршрута"}

            # Выбираем по приоритету и загрузке
            matches.sort(key=lambda r: r["priority"])

            best = matches[0]
            team_name = best["team"]
            current_load = self.load.get(team_name, 0)

            return {
                "team": team_name,
                "priority": best["priority"],
                "current_load": current_load,
            }

        def update_load(self, team, delta):
            """Обновить загрузку команды."""
            self.load[team] = self.load.get(team, 0) + delta

    router = IncidentRouter()
    router.add_route(r"login|auth", "auth-team", priority=1)
    router.add_route(r"data|loss|corrupt", "data-team", priority=2)
    router.add_route(r"performance|slow", "platform-team", priority=1)
    router.add_route(r"security|leak|breach", "security-team", priority=3)
    router.add_route(r"ui|interface|frontend", "frontend-team", priority=1)

    # Имитируем загрузку
    router.update_load("auth-team", 3)
    router.update_load("data-team", 5)
    router.update_load("security-team", 1)

    incidents = [
        {"type": "login_timeout", "description": "Таймаут при входе"},
        {"type": "data_corruption", "description": "Повреждение БД"},
        {"type": "slow_query", "description": "Медленный запрос"},
        {"type": "security_leak_detected", "description": "Обнаружена утечка"},
    ]

    for incident in incidents:
        route = router.route(incident)
        print(f"  Инцидент: {incident['description']}")
        print(f"    Маршрут: {route['team']} (приоритет: {route.get('priority', 'N/A')}, "
              f"загрузка: {route.get('current_load', 'N/A')})")

    # --- 3.3 SLA-соответствие ---
    print("\n--- 3.3 SLA-соответствие (SLA Compliance) ---")

    class SLAMonitor:
        """Мониторинг соответствия SLA (Service Level Agreement)."""

        def __init__(self):
            self.agreements = {}
            self.incidents = []

        def add_agreement(self, severity, response_time_min, resolution_time_min):
            """Добавить SLA для уровня серьёзности."""
            self.agreements[severity] = {
                "response_time": response_time_min,
                "resolution_time": resolution_time_min,
            }

        def record_incident(self, severity, response_time, resolution_time):
            """Записать инцидент и проверить SLA."""
            sla = self.agreements.get(severity, {})
            resp_sla = sla.get("response_time", 999)
            res_sla = sla.get("resolution_time", 999)

            result = {
                "severity": severity,
                "response_breach": response_time > resp_sla,
                "resolution_breach": resolution_time > res_sla,
                "response_time": response_time,
                "resolution_time": resolution_time,
                "sla_response": resp_sla,
                "sla_resolution": res_sla,
            }
            self.incidents.append(result)
            return result

        def compliance_report(self):
            """Отчёт о соответствии SLA."""
            total = len(self.incidents)
            if total == 0:
                return {"total": 0, "compliance": 1.0}

            resp_breaches = sum(1 for i in self.incidents if i["response_breach"])
            res_breaches = sum(1 for i in self.incidents if i["resolution_breach"])

            return {
                "total": total,
                "response_compliance": (total - resp_breaches) / total,
                "resolution_compliance": (total - res_breaches) / total,
                "resp_breaches": resp_breaches,
                "res_breaches": res_breaches,
            }

    sla = SLAMonitor()
    sla.add_agreement(1, response_time_min=1440, resolution_time_min=10080)  # 24ч/7д
    sla.add_agreement(3, response_time_min=240, resolution_time_min=1440)     # 4ч/24ч
    sla.add_agreement(5, response_time_min=15, resolution_time_min=120)       # 15мин/2ч

    # Записываем инциденты
    incidents_data = [
        (1, 60, 480),      # INFO — в рамках SLA
        (3, 300, 1200),    # MEDIUM — превышение response
        (5, 10, 90),       # CRITICAL — в рамках SLA
        (3, 180, 900),     # MEDIUM — в рамках SLA
        (5, 20, 180),      # CRITICAL — превышение response
    ]

    for severity, resp_time, res_time in incidents_data:
        result = sla.record_incident(severity, resp_time, res_time)
        status_r = "FAIL" if result["response_breach"] else "OK"
        status_s = "FAIL" if result["resolution_breach"] else "OK"
        print(f"  Severity {severity}: response={resp_time}мин [{status_r}], "
              f"resolution={res_time}мин [{status_s}]")

    report = sla.compliance_report()
    print(f"\n  SLA Compliance:")
    print(f"    Response: {report.get('response_compliance', 1):.1%} "
          f"({report.get('resp_breaches', 0)} нарушений)")
    print(f"    Resolution: {report.get('resolution_compliance', 1):.1%} "
          f"({report.get('res_breaches', 0)} нарушений)")

    # --- 3.4 Эскалационная лестница ---
    print("\n--- 3.4 Эскалационная лестница (Escalation Ladder) ---")

    class EscalationLadder:
        """Ступенчатая эскалация с тайм-аутами."""

        def __init__(self):
            self.levels = []

        def add_level(self, name, timeout_min, action):
            """Добавить ступень эскалации."""
            self.levels.append({
                "name": name,
                "timeout": timeout_min,
                "action": action,
            })

        def simulate(self, incident_id, resolution_time):
            """Имитировать прохождение по ступеням."""
            elapsed = 0
            log = []

            for i, level in enumerate(self.levels):
                if elapsed >= resolution_time:
                    break  # инцидент решён до этой ступени

                log.append({
                    "level": i + 1,
                    "name": level["name"],
                    "action": level["action"],
                    "time": elapsed,
                })

                # Время на этой ступени
                step_time = min(level["timeout"], resolution_time - elapsed)
                elapsed += step_time

            return log

    ladder = EscalationLadder()
    ladder.add_level("auto-resolve", 5, "автоматическое исправление")
    ladder.add_level("L1-support", 30, "базовая поддержка")
    ladder.add_level("L2-engineer", 60, "инженер-специалист")
    ladder.add_level("L3-architect", 120, "архитектор/руководитель")
    ladder.add_level("management", 999, "управление + PR-команда")

    # Три инцидента разной сложности
    test_cases = [
        ("INC-001", 8, "решён автоматически"),
        ("INC-002", 45, "решён на L2"),
        ("INC-003", 200, "дошло до management"),
    ]

    for inc_id, resolution, outcome in test_cases:
        log = ladder.simulate(inc_id, resolution)
        print(f"\n  {inc_id} (решено за {resolution} мин → {outcome}):")
        for entry in log:
            print(f"    Уровень {entry['level']}: {entry['name']} "
                  f"({entry['action']}) @ {entry['time']} мин")


# =============================================================================
# 4. Collaborative Patterns — совместные паттерны работы
# =============================================================================

def demo_collaborative_patterns():
    """Демонстрация паттернов совместной работы человека и ИИ."""
    print("\n" + "=" * 70)
    print("DEMO 4: Collaborative Patterns — совместные паттерны")
    print("=" * 70)

    # --- 4.1 Циклы проверки (Review Loops) ---
    print("\n--- 4.1 Циклы проверки (Review Loops) ---")

    class ReviewLoop:
        """Цикл проверки: ИИ → Человек → Коррекция → ИИ."""

        def __init__(self, max_rounds=3):
            self.max_rounds = max_rounds
            self.rounds = []

        def run(self, initial_output, reviewer, corrector):
            """Запустить цикл проверки."""
            current = initial_output

            for round_num in range(1, self.max_rounds + 1):
                # Человек проверяет
                feedback, approved = reviewer(current, round_num)

                self.rounds.append({
                    "round": round_num,
                    "input": current[:50],
                    "feedback": feedback,
                    "approved": approved,
                })

                if approved:
                    return current, round_num, self.rounds

                # ИИ корректирует
                current = corrector(current, feedback)

            return current, self.max_rounds, self.rounds

    # Пример: редактирование текста
    def reviewer(text, round_num):
        """Имитация ревьюера."""
        issues = {
            1: ("слишком формальный тон", False),
            2: ("добавить примеры", False),
            3: ("отлично, можно публиковать", True),
        }
        return issues.get(round_num, ("OK", True))

    def corrector(text, feedback):
        """Имитация коррекции ИИ."""
        if "формальный" in feedback:
            return text.replace("Уважаемый коллега", "Привет")
        if "примеры" in feedback:
            return text + "\nНапример: case study #1"
        return text

    loop = ReviewLoop(max_rounds=3)
    initial = "Уважаемый коллега, presentaция готова к рассмотрению."
    final, rounds_used, log = loop.run(initial, reviewer, corrector)

    print(f"  Начальный текст: '{initial}'")
    for entry in log:
        print(f"  Раунд {entry['round']}: feedback='{entry['feedback']}', "
              f"approved={entry['approved']}")
    print(f"  Результат (после {rounds_used} раундов): '{final}'")

    # --- 4.2 Approval Workflows ---
    print("\n--- 4.2 Approval Workflows (процедуры согласования) ---")

    class ApprovalWorkflow:
        """Процедура многоуровневого согласования."""

        def __init__(self):
            self.approvers = []

        def add_approver(self, name, role, auto_approve_below=None):
            """Добавить согласующего."""
            self.approvers.append({
                "name": name,
                "role": role,
                "auto_approve_below": auto_approve_below,
            })

        def process(self, request):
            """Обработать заявку на согласование."""
            results = []
            current_status = "pending"

            for approver in self.approvers:
                # Автоматическое согласование для мелких сумм
                if (approver["auto_approve_below"] is not None and
                    request.get("amount", 0) < approver["auto_approve_below"]):
                    decision = "auto_approved"
                else:
                    # Имитация решения
                    decision = "approved" if request.get("amount", 0) < 5000 else "rejected"

                results.append({
                    "approver": approver["name"],
                    "role": approver["role"],
                    "decision": decision,
                })

                if decision == "rejected":
                    current_status = "rejected"
                    break
                current_status = "approved"

            return current_status, results

    workflow = ApprovalWorkflow()
    workflow.add_approver("Менеджер", "team_lead", auto_approve_below=500)
    workflow.add_approver("Директор", "department_head")
    workflow.add_approver("CFO", "finance")

    requests = [
        {"name": "Новая клавиатура", "amount": 150},
        {"name": "Лицензия на ПО", "amount": 2000},
        {"name": "Новый сервер", "amount": 8000},
    ]

    for req in requests:
        status, approvals = workflow.process(req)
        print(f"  Заявка: '{req['name']}' (сумма: {req['amount']})")
        print(f"    Статус: {status}")
        for a in approvals:
            print(f"    {a['approver']} ({a['role']}): {a['decision']}")
        print()

    # --- 4.3 Mixed Initiative (смешанная инициатива) ---
    print("\n--- 4.3 Mixed Initiative (смешанная инициатива) ---")

    class MixedInitiative:
        """Система, где и человек, и ИИ могут инициировать действия."""

        def __init__(self):
            self.initiatives = []
            self.decisions = []

        def ai_proposes(self, action, reason, confidence):
            """ИИ предлагает действие."""
            entry = {"source": "AI", "action": action, "reason": reason,
                     "confidence": confidence}
            self.initiatives.append(entry)
            return entry

        def human_decides(self, initiative, decision):
            """Человек принимает решение по提议."""
            self.decisions.append({
                "initiative": initiative,
                "decision": decision,
            })
            return decision

        def human_initiates(self, action, reason):
            """Человек инициирует действие."""
            entry = {"source": "Human", "action": action, "reason": reason}
            self.initiatives.append(entry)
            return entry

    system = MixedInitiative()

    # ИИ предлагает
    prop1 = system.ai_proposes(
        "увеличить бюджет рекламы",
        "конверсия выросла на 20%",
        0.85
    )
    decision1 = system.human_decides(prop1, "одобрено")
    print(f"  ИИ提议: {prop1['action']}")
    print(f"    Причина: {prop1['reason']}")
    print(f"    Решение человека: {decision1}")

    # Человек инициирует
    prop2 = system.human_initiates(
        "обновить дизайн лендинга",
        "нужен редизайн перед запуском"
    )
    print(f"\n  Человек提议: {prop2['action']}")
    print(f"    Причина: {prop2['reason']}")

    # ИИ предлагает автоматическое действие
    prop3 = system.ai_proposes(
        "автоматически ответить на жалобу",
        "шаблон подходит на 90%",
        0.92
    )
    decision3 = system.human_decides(prop3, "одобрено с редактированием")
    print(f"\n  ИИ提议: {prop3['action']}")
    print(f"    Решение: {decision3}")

    print(f"\n  Всего инициатив: {len(system.initiatives)} "
          f"(ИИ: {sum(1 for i in system.initiatives if i['source']=='AI')}, "
          f"Человек: {sum(1 for i in system.initiatives if i['source']=='Human')})")

    # --- 4.4 Роли и ответственность ---
    print("\n--- 4.4 Матрица ролей и ответственности (RACI) ---")

    class RAMatrix:
        """Матрица ответственности (RACI: Responsible, Accountable, Consulted, Informed)."""

        def __init__(self):
            self.matrix = {}

        def add_task(self, task, roles):
            """Добавить задачу с ролями."""
            self.matrix[task] = roles

        def display(self):
            """Вывести матрицу."""
            if not self.matrix:
                return

            # Собираем все роли
            all_roles = set()
            for roles in self.matrix.values():
                all_roles.update(roles.keys())
            all_roles = sorted(all_roles)

            # Заголовок
            header = f"  {'Задача':30s}"
            for role in all_roles:
                header += f" {role:8s}"
            print(header)
            print("  " + "-" * (30 + 9 * len(all_roles)))

            # Строки
            for task, roles in self.matrix.items():
                row = f"  {task:30s}"
                for role in all_roles:
                    value = roles.get(role, "-")
                    row += f" {value:8s}"
                print(row)

    raci = RAMatrix()
    raci.add_task("Определение требований", {"AI": "C", "PM": "R/A", "Dev": "C", "QA": "I"})
    raci.add_task("Разработка модели", {"AI": "R", "PM": "A", "Dev": "R", "QA": "C"})
    raci.add_task("Тестирование", {"AI": "C", "PM": "I", "Dev": "C", "QA": "R/A"})
    raci.add_task("Деплой", {"AI": "R", "PM": "A", "Dev": "R", "QA": "C"})
    raci.add_task("Мониторинг", {"AI": "R", "PM": "I", "Dev": "C", "QA": "I"})

    print("  R=Responsible  A=Accountable  C=Consulted  I=Informed\n")
    raci.display()


# =============================================================================
# Запуск всех демонстраций
# =============================================================================

if __name__ == "__main__":
    print("УРОК 179: Human-AI Teaming")
    print("Протоколы передачи, калибровка доверия, эскалация\n")

    demo_handoff_protocols()
    demo_trust_calibration()
    demo_escalation_strategies()
    demo_collaborative_patterns()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены.")
    print("=" * 70)
