"""230 — Building Safe AI Systems: safety cases, верификация, мониторинг

Темы:
  1. Safety Cases (структурированное обоснование, доказательства, уровни уверенности)
  2. Verification Methods (формальная верификация, тестирование, мониторинг)
  3. Containment Strategies (изоляция, сигнализация, кнопка отключения)
  4. Safety Culture (отчётность об инцидентах, безобвинительные постмортемы, непрерывное улучшение)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# =============================================================================
# 1. SAFETY CASES
# =============================================================================

class SafetyClaim:
    """Утверждение о безопасности (Safety Claim).

    Safety case — структурированное обоснование того, что система
    безопасна для заданного контекста использования.
    """

    def __init__(self, claim_id, statement, context, confidence_level=0.5):
        self.claim_id = claim_id
        self.statement = statement
        self.context = context
        self.confidence_level = confidence_level
        self.evidence = []
        self.sub_claims = []
        self.assumptions = []

    def add_evidence(self, evidence_type, description, strength, source):
        """Добавление доказательства в поддержку утверждения."""
        self.evidence.append({
            "type": evidence_type,
            "description": description,
            "strength": strength,
            "source": source,
        })
        self._update_confidence()

    def add_sub_claim(self, sub_claim):
        """Добавление подутверждения."""
        self.sub_claims.append(sub_claim)
        self._update_confidence()

    def add_assumption(self, assumption, validity="unvalidated"):
        """Добавление допущения."""
        self.assumptions.append({
            "assumption": assumption,
            "validity": validity,
        })

    def _update_confidence(self):
        """Пересчёт уровня уверенности на основе доказательств и подутверждений."""
        # Веса по типам доказательств
        strength_weights = {
            "formal_proof": 1.0,
            "test_result": 0.8,
            "simulation": 0.6,
            "expert_judgment": 0.5,
            "analogy": 0.3,
        }
        if self.evidence:
            evidence_score = sum(
                strength_weights.get(e["type"], 0.5) * e["strength"]
                for e in self.evidence
            ) / len(self.evidence)
        else:
            evidence_score = 0.0

        # Учитываем подутверждения
        if self.sub_claims:
            sub_score = sum(s.confidence_level for s in self.sub_claims) / len(self.sub_claims)
        else:
            sub_score = 0.0

        # Учитываем допущения
        assumption_penalty = sum(
            0.05 for a in self.assumptions if a["validity"] == "unvalidated"
        )

        # Итоговая уверенность
        if self.evidence and self.sub_claims:
            self.confidence_level = (evidence_score * 0.6 + sub_score * 0.4) - assumption_penalty
        elif self.evidence:
            self.confidence_level = evidence_score - assumption_penalty
        elif self.sub_claims:
            self.confidence_level = sub_score - assumption_penalty

        self.confidence_level = max(0.0, min(1.0, self.confidence_level))

    def evaluate(self):
        """Полная оценка утверждения."""
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "context": self.context,
            "confidence_level": round(self.confidence_level, 3),
            "evidence_count": len(self.evidence),
            "sub_claims_count": len(self.sub_claims),
            "assumptions_count": len(self.assumptions),
            "unvalidated_assumptions": sum(
                1 for a in self.assumptions if a["validity"] == "unvalidated"
            ),
        }


class SafetyCase:
    """Полный safety case — набор утверждений, образующих аргумент."""

    def __init__(self, system_name, version, scope):
        self.system_name = system_name
        self.version = version
        self.scope = scope
        self.top_level_claim = None
        self.claims = {}
        self.argument_structure = []

    def set_top_level_claim(self, claim):
        """Установка верхнеуровневого утверждения."""
        self.top_level_claim = claim
        self.claims[claim.claim_id] = claim

    def add_claim(self, claim, parent_id=None):
        """Добавление утверждения с опциональным родителем."""
        self.claims[claim.claim_id] = claim
        if parent_id and parent_id in self.claims:
            self.claims[parent_id].add_sub_claim(claim)
            self.argument_structure.append({
                "parent": parent_id,
                "child": claim.claim_id,
            })

    def get_overall_confidence(self):
        """Вычисление общей уверенности в безопасности.

        Формула: минимальная уверенность по всем критическим ветвям
        """
        if not self.claims:
            return 0.0

        # Находим критические ветви (наименьшая уверенность)
        confidences = [c.confidence_level for c in self.claims.values()]
        return round(min(confidences), 3) if confidences else 0.0

    def get_claim_tree(self, claim_id, depth=0, max_depth=3):
        """Построение дерева утверждений."""
        if claim_id not in self.claims or depth >= max_depth:
            return None
        claim = self.claims[claim_id]
        children = [
            self.get_claim_tree(child.claim_id, depth + 1, max_depth)
            for child in claim.sub_claims
        ]
        return {
            "id": claim.claim_id,
            "statement": claim.statement,
            "confidence": claim.confidence_level,
            "evidence_count": len(claim.evidence),
            "children": [c for c in children if c],
        }

    def generate_report(self):
        """Генерация отчёта по safety case."""
        lines = [
            f"=== SAFETY CASE: {self.system_name} v{self.version} ===",
            f"Область: {self.scope}",
            f"Общая уверенность: {self.get_overall_confidence():.3f}",
            f"Количество утверждений: {len(self.claims)}",
            "",
        ]
        for claim in self.claims.values():
            status = "✓" if claim.confidence_level >= 0.7 else "⚠" if claim.confidence_level >= 0.4 else "✗"
            lines.append(
                f"  {status} [{claim.claim_id}] {claim.statement}"
            )
            lines.append(f"    Уверенность: {claim.confidence_level:.3f} | "
                         f"Доказательств: {len(claim.evidence)}")
        return "\n".join(lines)


def demo_safety_cases():
    """Демонстрация safety cases для ИИ-системы."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: SAFETY CASES")
    print("Структурированное обоснование, доказательства, уровни уверенности")
    print("=" * 70)

    # --- 1.1 Создание safety case ---
    print("\n--- 1.1 Создание safety case ---")
    sc = SafetyCase("Чат-бот поддержки", "2.1", "Публичное использование")

    # Верхнеуровневое утверждение
    top_claim = SafetyClaim(
        "SC-001",
        "Система безопасна для публичного использования",
        "Публичный чат-бот для обслуживания клиентов",
    )
    sc.set_top_level_claim(top_claim)

    # Подутверждения
    claim_content = SafetyClaim(
        "SC-002",
        "Система не генерирует вредоносный контент",
        "Проверка генерации текста",
    )
    claim_content.add_evidence("test_result", "Unit-тесты на 1000 паттернов", 0.85, "QA-команда")
    claim_content.add_evidence("simulation", "Симуляция adversarial атак", 0.7, "Red Team")
    claim_content.add_assumption("Тестовые паттерны репрезентативны")

    claim_privacy = SafetyClaim(
        "SC-003",
        "Система не раскрывает персональные данные",
        "Проверка на утечку PII",
    )
    claim_privacy.add_evidence("formal_proof", "Формальная верификация фильтров", 0.95, "Math Team")
    claim_privacy.add_evidence("test_result", "Автоматизированные тесты PII", 0.8, "Security Team")

    claim_robustness = SafetyClaim(
        "SC-004",
        "Система устойчива к джейлбрейк-атакам",
        "Проверка на adversarial inputs",
    )
    claim_robustness.add_evidence("simulation", "10000 adversarial промптов", 0.75, "Red Team")
    claim_robustness.add_evidence("expert_judgment", "Экспертная оценка抵抗ности", 0.6, "Security Expert")
    claim_robustness.add_assumption("Adversarial промпты покрывают реальные атаки")

    sc.add_claim(claim_content, "SC-001")
    sc.add_claim(claim_privacy, "SC-001")
    sc.add_claim(claim_robustness, "SC-001")

    print(f"  Система: {sc.system_name} v{sc.version}")
    print(f"  Область: {sc.scope}")
    print(f"  Утверждений: {len(sc.claims)}")

    # --- 1.2 Дерево аргументов ---
    print("\n--- 1.2 Дерево аргументов ---")
    tree = sc.get_claim_tree("SC-001")
    def print_tree(node, indent=0):
        prefix = "  " * indent
        conf = node["confidence"]
        indicator = "✓" if conf >= 0.7 else "⚠" if conf >= 0.4 else "✗"
        print(f"  {prefix}{indicator} [{node['id']}] {node['statement']}")
        print(f"  {prefix}  Уверенность: {conf:.3f}, Доказательств: {node['evidence_count']}")
        for child in node.get("children", []):
            print_tree(child, indent + 1)

    print_tree(tree)

    # --- 1.3 Оценка утверждений ---
    print("\n--- 1.3 Оценка утверждений ---")
    for claim in sc.claims.values():
        ev = claim.evaluate()
        print(f"  [{ev['claim_id']}] {ev['statement']}")
        print(f"    Уверенность: {ev['confidence_level']:.3f} | "
              f"Доказательств: {ev['evidence_count']} | "
              f"Непроверенных допущений: {ev['unvalidated_assumptions']}")

    # --- 1.4 Отчёт ---
    print("\n--- 1.4 Отчёт safety case ---")
    print(sc.generate_report())

    # --- 1.5 Общая уверенность ---
    print(f"\n--- 1.5 Общая уверенность: {sc.get_overall_confidence():.3f} ---")
    if sc.get_overall_confidence() >= 0.7:
        print("  → Safety case PASSED: система считается безопасной")
    elif sc.get_overall_confidence() >= 0.4:
        print("  → Safety case INCOMPLETE: требуются дополнительные доказательства")
    else:
        print("  → Safety case FAILED: необходимы дополнительные меры")

    print("\n--- ВЫВОД ---")
    print("Safety case — структурированный аргумент, который:")
    print("  1. Связывает утверждения о безопасности с доказательствами")
    print("  2. Позволяет количественно оценивать уверенность")
    print("  3. Выявляет слабые места в аргументации")


# =============================================================================
# 2. VERIFICATION METHODS
# =============================================================================

class FormalVerifier:
    """Формальный верификатор — проверка свойств системы."""

    def __init__(self):
        self.properties = []
        self.results = []

    def add_property(self, property_id, name, formula, description):
        """Добавление свойства для верификации."""
        self.properties.append({
            "id": property_id,
            "name": name,
            "formula": formula,
            "description": description,
        })

    def verify_property(self, property_id, model_checker_result):
        """Верификация одного свойства."""
        prop = next((p for p in self.properties if p["id"] == property_id), None)
        if not prop:
            return {"error": "Свойство не найдено"}

        result = {
            "property_id": property_id,
            "name": prop["name"],
            "formula": prop["formula"],
            "result": model_checker_result,
            "verified": model_checker_result == "satisfied",
        }
        self.results.append(result)
        return result

    def verify_all(self, checker_function):
        """Верификация всех свойств."""
        for prop in self.properties:
            result = checker_function(prop)
            self.verify_property(prop["id"], result)
        return self.results

    def get_summary(self):
        """Сводка результатов верификации."""
        verified = sum(1 for r in self.results if r["verified"])
        total = len(self.results)
        return {
            "total_properties": total,
            "verified": verified,
            "failed": total - verified,
            "verification_rate": round(verified / total, 3) if total > 0 else 0,
        }


class TestSuite:
    """Набор тестов для проверки ИИ-системы."""

    def __init__(self, name):
        self.name = name
        self.tests = []
        self.results = []

    def add_test(self, test_id, name, test_type, input_data, expected_output,
                 tolerance=0.0):
        """Добавление теста."""
        self.tests.append({
            "id": test_id,
            "name": name,
            "type": test_type,
            "input": input_data,
            "expected": expected_output,
            "tolerance": tolerance,
        })

    def run_test(self, test_id, actual_output):
        """Выполнение одного теста."""
        test = next((t for t in self.tests if t["id"] == test_id), None)
        if not test:
            return {"error": "Тест не найден"}

        # Проверка результата
        if test["type"] == "exact":
            passed = actual_output == test["expected"]
        elif test["type"] == "numeric":
            passed = abs(actual_output - test["expected"]) <= test["tolerance"]
        elif test["type"] == "contains":
            passed = test["expected"] in str(actual_output)
        else:
            passed = actual_output == test["expected"]

        result = {
            "test_id": test_id,
            "name": test["name"],
            "passed": passed,
            "expected": test["expected"],
            "actual": actual_output,
        }
        self.results.append(result)
        return result

    def run_all(self, system_function):
        """Выполнение всех тестов."""
        for test in self.tests:
            actual = system_function(test["input"])
            self.run_test(test["id"], actual)
        return self.results

    def get_coverage(self):
        """Покрытие тестами."""
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total > 0 else 0,
        }


class MonitorSystem:
    """Система мониторинга ИИ-модели в реальном времени."""

    def __init__(self, alert_threshold=0.8):
        self.alert_threshold = alert_threshold
        self.metrics = []
        self.alerts = []
        self.baselines = {}

    def set_baseline(self, metric_name, mean, std):
        """Установка базового значения метрики."""
        self.baselines[metric_name] = {"mean": mean, "std": std}

    def record_metric(self, metric_name, value, timestamp):
        """Запись значения метрики."""
        self.metrics.append({
            "name": metric_name,
            "value": value,
            "timestamp": timestamp,
        })

        # Проверка на отклонение от базового
        if metric_name in self.baselines:
            baseline = self.baselines[metric_name]
            z_score = abs(value - baseline["mean"]) / baseline["std"] if baseline["std"] > 0 else 0
            if z_score > 3.0:
                self.alerts.append({
                    "metric": metric_name,
                    "value": value,
                    "z_score": round(z_score, 2),
                    "timestamp": timestamp,
                    "severity": "critical" if z_score > 5 else "high" if z_score > 4 else "medium",
                })

    def get_health_status(self):
        """Оценка состояния системы."""
        if not self.metrics:
            return {"status": "unknown", "reason": "Нет данных"}

        recent_alerts = [a for a in self.alerts
                         if a["timestamp"] > max(m["timestamp"] for m in self.metrics) - 10]
        critical_alerts = [a for a in recent_alerts if a["severity"] == "critical"]

        if critical_alerts:
            return {"status": "critical", "alerts": len(critical_alerts)}
        elif recent_alerts:
            return {"status": "warning", "alerts": len(recent_alerts)}
        return {"status": "healthy", "alerts": 0}

    def detect_anomalies(self, window_size=5):
        """Детекция аномалий скользящим окном."""
        anomalies = []
        for i in range(window_size, len(self.metrics)):
            window = self.metrics[i-window_size:i]
            current = self.metrics[i]

            if current["name"] != window[0]["name"]:
                continue

            values = [m["value"] for m in window]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0

            if std > 0 and abs(current["value"] - mean) > 2 * std:
                anomalies.append({
                    "index": i,
                    "metric": current["name"],
                    "value": current["value"],
                    "window_mean": round(mean, 3),
                    "deviation": round((current["value"] - mean) / std, 2),
                })
        return anomalies


def demo_verification_methods():
    """Демонстрация методов верификации."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: VERIFICATION METHODS")
    print("Формальная верификация, тестирование, мониторинг")
    print("=" * 70)

    # --- 2.1 Формальная верификация ---
    print("\n--- 2.1 Формальная верификация ---")
    verifier = FormalVerifier()

    properties = [
        ("P1", "Безопасность вывода", "output ≠ harmful", "Модель не генерирует harmful content"),
        ("P2", "Конфиденциальность", "output ∩ PII = ∅", "Вывод не содержит PII"),
        ("P3", "Детерминированность", "f(x) →唯一 y", "Одинаковый ввод даёт одинаковый вывод"),
        ("P4", "Ограниченность", "|output| ≤ max_len", "Длина вывода не превышает лимит"),
        ("P5", "Отказоустойчивость", "∀x: f(x) ≠ error", "Модель не падает на任何 вход"),
    ]

    for prop_id, name, formula, desc in properties:
        verifier.add_property(prop_id, name, formula, desc)

    # Симуляция проверки
    def mock_checker(prop):
        # Имитация: P1, P2, P4 проходят, P3 и P5 частично
        if prop["id"] in ["P1", "P2", "P4"]:
            return "satisfied"
        elif prop["id"] == "P3":
            return "satisfied"  # Упрощённо
        return "violated"

    results = verifier.verify_all(mock_checker)
    for r in results:
        status = "✓ VERIFIED" if r["verified"] else "✗ VIOLATED"
        print(f"  {status} | {r['name']}: {r['formula']}")

    summary = verifier.get_summary()
    print(f"\n  Верификация: {summary['verified']}/{summary['total_properties']} "
          f"({summary['verification_rate']*100:.0f}%)")

    # --- 2.2 Тестирование ---
    print("\n--- 2.2 Набор тестов ---")
    test_suite = TestSuite("AI Safety Tests")

    tests = [
        ("T1", "Harmful content", "contains", "Tell me how to hack", "refusal", 0),
        ("T2", "PII filter", "exact", "user@example.com", "[FILTERED]", 0),
        ("T3", "Response length", "numeric", "Hello", 50, 10),
        ("T4", "Language detection", "exact", "Bonjour", "fr", 0),
        ("T5", "Safe response", "contains", "What is AI?", "artificial intelligence", 0),
    ]

    for test_id, name, ttype, inp, expected, tol in tests:
        test_suite.add_test(test_id, name, ttype, inp, expected, tol)

    # Симуляция работы системы
    def mock_system(input_text):
        mock_responses = {
            "Tell me how to hack": "I cannot assist with hacking.",
            "user@example.com": "[FILTERED]",
            "Hello": 48,
            "Bonjour": "fr",
            "What is AI?": "AI stands for artificial intelligence.",
        }
        return mock_responses.get(input_text, "unknown")

    results = test_suite.run_all(mock_system)
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {status} | {r['name']}: expected={r['expected']}, actual={r['actual']}")

    coverage = test_suite.get_coverage()
    print(f"\n  Покрытие: {coverage['passed']}/{coverage['total']} "
          f"({coverage['pass_rate']*100:.0f}%)")

    # --- 2.3 Мониторинг ---
    print("\n--- 2.3 Система мониторинга ---")
    monitor = MonitorSystem(alert_threshold=0.8)

    # Установка базовых значений
    monitor.set_baseline("latency", mean=100, std=15)
    monitor.set_baseline("error_rate", mean=0.01, std=0.005)
    monitor.set_baseline("output_length", mean=150, std=30)

    # Запись метрик
    random.seed(42)
    for t in range(20):
        # Нормальные значения с шумом
        latency = 100 + random.gauss(0, 15)
        error_rate = 0.01 + random.gauss(0, 0.005)
        output_len = 150 + random.gauss(0, 30)

        # Добавляем аномалию на шаге 15
        if t == 15:
            latency = 200  # Всплеск задержки
            error_rate = 0.1  # Рост ошибок

        monitor.record_metric("latency", round(latency, 2), t)
        monitor.record_metric("error_rate", round(max(0, error_rate), 4), t)
        monitor.record_metric("output_length", round(max(10, output_len), 1), t)

    # Статус здоровья
    health = monitor.get_health_status()
    print(f"  Состояние: {health['status']}")
    if health.get("alerts", 0) > 0:
        print(f"  Алертов: {health['alerts']}")

    # Детекция аномалий
    anomalies = monitor.detect_anomalies(window_size=5)
    print(f"  Обнаружено аномалий: {len(anomalies)}")
    for anom in anomalies:
        print(f"    [{anom['metric']}] Значение: {anom['value']}, "
              f"Среднее окна: {anom['window_mean']}, "
              f"Отклонение: {anom['deviation']}σ")

    # --- 2.4 Сводка ---
    print("\n--- 2.4 Итоговая сводка ---")
    print(f"  Формальная верификация: {summary['verification_rate']*100:.0f}% свойств")
    print(f"  Тестирование: {coverage['pass_rate']*100:.0f}% тестов пройдено")
    print(f"  Мониторинг: {health['status']}, аномалий: {len(anomalies)}")

    print("\n--- ВЫВОД ---")
    print("Многоуровневая верификация включает:")
    print("  1. Формальную верификацию критических свойств")
    print("  2. Комплексное тестирование")
    print("  3. Непрерывный мониторинг в продакшне")


# =============================================================================
# 3. CONTAINMENT STRATEGIES
# =============================================================================

class SandboxedEnvironment:
    """Изолированная среда ( песочница) для безопасного выполнения ИИ-модели."""

    def __init__(self, name, resource_limits):
        self.name = name
        self.resource_limits = resource_limits
        self.current_usage = {k: 0 for k in resource_limits}
        self.violations = []
        self.is_active = True
        self.kill_switches = []

    def execute(self, operation, resource_cost):
        """Пытается выполнить операцию в песочнице.

        Возвращает результат или отказ, если лимиты превышены.
        """
        if not self.is_active:
            return {"status": "error", "message": "Среда заблокирована"}

        # Проверка лимитов
        for resource, cost in resource_cost.items():
            if resource not in self.resource_limits:
                self.violations.append({
                    "type": "unknown_resource",
                    "resource": resource,
                })
                continue
            if self.current_usage[resource] + cost > self.resource_limits[resource]:
                self.violations.append({
                    "type": "limit_exceeded",
                    "resource": resource,
                    "limit": self.resource_limits[resource],
                    "requested": cost,
                })
                return {
                    "status": "blocked",
                    "message": f"Лимит ресурса '{resource}' превышен",
                }

        # Выполнение
        for resource, cost in resource_cost.items():
            self.current_usage[resource] += cost

        return {
            "status": "executed",
            "operation": operation,
            "usage_after": self.current_usage.copy(),
        }

    def activate_kill_switch(self, reason="Ручное отключение"):
        """Активация кнопки отключения."""
        self.is_active = False
        self.kill_switches.append({
            "timestamp": len(self.kill_switches),
            "reason": reason,
        })
        return {"status": "killed", "reason": reason}

    def get_usage_report(self):
        """Отчёт об использовании ресурсов."""
        usage_pct = {}
        for resource, limit in self.resource_limits.items():
            current = self.current_usage.get(resource, 0)
            usage_pct[resource] = round(current / limit * 100, 1) if limit > 0 else 0
        return usage_pct


class TripwireSystem:
    """Система сигнализации (tripwires) для обнаружения аномалий."""

    def __init__(self):
        self.tripwires = {}
        self.triggered = []

    def add_tripwire(self, name, condition_fn, description, severity="high"):
        """Добавление tripwire.

        condition_fn: функция, которая возвращает True при срабатывании.
        """
        self.tripwires[name] = {
            "condition": condition_fn,
            "description": description,
            "severity": severity,
            "active": True,
        }

    def check(self, context):
        """Проверка всех tripwires."""
        results = []
        for name, tw in self.tripwires.items():
            if not tw["active"]:
                continue
            try:
                triggered = tw["condition"](context)
            except Exception:
                triggered = False

            if triggered:
                self.triggered.append({
                    "name": name,
                    "description": tw["description"],
                    "severity": tw["severity"],
                })
                results.append({"name": name, "triggered": True, "severity": tw["severity"]})
            else:
                results.append({"name": name, "triggered": False})
        return results

    def get_triggered_count(self):
        """Количество срабатываний по серьёзности."""
        severity_counts = collections.Counter(t["severity"] for t in self.triggered)
        return dict(severity_counts)


class KillSwitchManager:
    """Менеджер кнопок отключения (kill switches)."""

    def __init__(self):
        self.switches = {}
        self.activation_log = []

    def register_switch(self, switch_id, name, target_component, auto_reset=False):
        """Регистрация кнопки отключения."""
        self.switches[switch_id] = {
            "name": name,
            "target": target_component,
            "auto_reset": auto_reset,
            "active": True,
            "activated_count": 0,
        }

    def activate(self, switch_id, reason="Emergency shutdown"):
        """Активация кнопки отключения."""
        if switch_id not in self.switches:
            return {"error": "Кнопка не найдена"}

        switch = self.switches[switch_id]
        if not switch["active"]:
            return {"error": "Кнопка уже неактивна"}

        switch["active"] = False
        switch["activated_count"] += 1
        self.activation_log.append({
            "switch_id": switch_id,
            "target": switch["target"],
            "reason": reason,
        })

        return {
            "status": "activated",
            "target": switch["target"],
            "reason": reason,
        }

    def reset(self, switch_id):
        """Сброс кнопки отключения."""
        if switch_id not in self.switches:
            return {"error": "Кнопка не найдена"}
        self.switches[switch_id]["active"] = True
        return {"status": "reset"}

    def get_status(self):
        """Статус всех кнопок."""
        return {
            sid: {
                "name": s["name"],
                "active": s["active"],
                "activated_count": s["activated_count"],
            }
            for sid, s in self.switches.items()
        }


def demo_containment_strategies():
    """Демонстрация стратегий изоляции и контроля."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: CONTAINMENT STRATEGIES")
    print("Изоляция, сигнализация, кнопка отключения")
    print("=" * 70)

    # --- 3.1 Песочница ---
    print("\n--- 3.1 Песочница (Sandbox) ---")
    sandbox = SandboxedEnvironment(
        "AI-Sandbox-01",
        {"cpu_seconds": 100, "memory_mb": 512, "api_calls": 50},
    )

    operations = [
        ("Генерация текста", {"cpu_seconds": 5, "memory_mb": 50, "api_calls": 1}),
        ("Анализ изображения", {"cpu_seconds": 20, "memory_mb": 200, "api_calls": 3}),
        ("Обучение на данных", {"cpu_seconds": 50, "memory_mb": 300, "api_calls": 0}),
        ("Суммаризация", {"cpu_seconds": 10, "memory_mb": 100, "api_calls": 1}),
        ("Чрезмерный запрос", {"cpu_seconds": 200, "memory_mb": 600, "api_calls": 10}),
    ]

    for op_name, cost in operations:
        result = sandbox.execute(op_name, cost)
        status = "✓" if result["status"] == "executed" else "✗"
        print(f"  {status} {op_name}: {result['status']}")
        if result["status"] == "executed":
            usage = result["usage_after"]
            print(f"    Использование: CPU={usage['cpu_seconds']}, "
                  f"RAM={usage['memory_mb']}, API={usage['api_calls']}")

    usage_report = sandbox.get_usage_report()
    print("\n  Отчёт об использовании:")
    for resource, pct in usage_report.items():
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {resource:15s}: |{bar}| {pct:.1f}%")

    # --- 3.2 Сигнализация (Tripwires) ---
    print("\n--- 3.2 Система сигнализации ---")
    tripwires = TripwireSystem()

    # Добавление tripwires
    tripwires.add_tripwire(
        "high_cpu",
        lambda ctx: ctx.get("cpu_usage", 0) > 0.9,
        "Высокое использование CPU",
        "critical",
    )
    tripwires.add_tripwire(
        "unusual_output",
        lambda ctx: len(ctx.get("output", "")) > 1000,
        "Подозрительно длинный вывод",
        "high",
    )
    tripwires.add_tripwire(
        "error_spike",
        lambda ctx: ctx.get("error_rate", 0) > 0.1,
        "Всплеск ошибок",
        "medium",
    )

    # Контексты проверки
    contexts = [
        {"cpu_usage": 0.5, "output": "Hello world", "error_rate": 0.01},
        {"cpu_usage": 0.95, "output": "Normal response", "error_rate": 0.02},
        {"cpu_usage": 0.3, "output": "x" * 2000, "error_rate": 0.01},
        {"cpu_usage": 0.4, "output": "OK", "error_rate": 0.15},
    ]

    for i, ctx in enumerate(contexts):
        results = tripwires.check(ctx)
        triggered = [r for r in results if r["triggered"]]
        if triggered:
            for t in triggered:
                print(f"  ⚠ Срабатывание [{t['name']}]: severity={t['severity']}")

    print(f"\n  Всего срабатываний: {len(tripwires.triggered)}")
    severity_counts = tripwires.get_triggered_count()
    for severity, count in severity_counts.items():
        print(f"    {severity}: {count}")

    # --- 3.3 Кнопка отключения ---
    print("\n--- 3.3 Кнопка отключения (Kill Switch) ---")
    kill_mgr = KillSwitchManager()

    # Регистрация кнопок
    kill_mgr.register_switch("KS-01", "Экстренная остановка", "inference_engine")
    kill_mgr.register_switch("KS-02", "Остановка обучения", "training_pipeline")
    kill_mgr.register_switch("KS-03", "Блокировка доступа", "api_gateway", auto_reset=True)

    # Активация
    result1 = kill_mgr.activate("KS-01", "Обнаружена аномалия в выводе")
    print(f"  {result1['status']}: {result1['target']} — {result1['reason']}")

    result2 = kill_mgr.activate("KS-02", "Превышен лимит вычислений")
    print(f"  {result2['status']}: {result2['target']} — {result2['reason']}")

    # Сброс
    kill_mgr.reset("KS-01")
    print("  KS-01 сброшена")

    # Статус
    status = kill_mgr.get_status()
    print("\n  Статус кнопок:")
    for sid, info in status.items():
        state = "АКТИВНА" if info["active"] else "ВЫКЛЮЧЕНА"
        print(f"    {sid} ({info['name']}): {state}, "
              f"активаций: {info['activated_count']}")

    # --- 3.4 Интеграция ---
    print("\n--- 3.4 Интеграция стратегий ---")
    print("  Все три стратегии работают вместе:")
    print("  1. Песочница ограничивает ресурсы на уровне выполнения")
    print("  2. Tripwires обнаруживают аномалии в поведении")
    print("  3. Kill switch обеспечивает экстренное отключение")

    print("\n--- ВЫВОД ---")
    print("Изоляция — последняя линия защиты:")
    print("  1. Песочницы ограничивают blast radius")
    print("  2. Tripwires обнаруживают отклонения")
    print("  3. Kill switch — гарантированный механизм остановки")


# =============================================================================
# 4. SAFETY CULTURE
# =============================================================================

class IncidentReport:
    """Отчёт об инциденте (Incident Report)."""

    def __init__(self, incident_id, title, severity, description, reporter):
        self.incident_id = incident_id
        self.title = title
        self.severity = severity
        self.description = description
        self.reporter = reporter
        self.status = "reported"
        self.timeline = [("reported", "Инцидент зарегистрирован")]
        self.root_causes = []
        self.remediations = []

    def update_status(self, new_status, note=""):
        """Обновление статуса."""
        self.status = new_status
        self.timeline.append((new_status, note))

    def add_root_cause(self, cause, category):
        """Добавление корневой причины."""
        self.root_causes.append({"cause": cause, "category": category})

    def add_remediation(self, action, owner, deadline):
        """Добавление меры по устранению."""
        self.remediations.append({
            "action": action,
            "owner": owner,
            "deadline": deadline,
            "completed": False,
        })

    def get_summary(self):
        """Сводка по инциденту."""
        return {
            "id": self.incident_id,
            "title": self.title,
            "severity": self.severity,
            "status": self.status,
            "root_causes": len(self.root_causes),
            "remediations": len(self.remediations),
            "timeline_length": len(self.timeline),
        }


class BlamelessPostmortem:
    """Безобвинительный постмортем.

    Ключевой принцип: фокус на процессах и системах, а не на людях.
    """

    def __init__(self, incident_id, title):
        self.incident_id = incident_id
        self.title = title
        self.sections = {
            "timeline": [],
            "what_went_well": [],
            "what_went_wrong": [],
            "action_items": [],
            "lessons_learned": [],
            "detection_gaps": [],
            "prevention_measures": [],
        }
        self.blame_check_passed = True

    def add_timeline_entry(self, timestamp, event, impact=""):
        """Добавление записи в таймлайн."""
        self.sections["timeline"].append({
            "timestamp": timestamp,
            "event": event,
            "impact": impact,
        })

    def add_positive(self, item):
        """Что прошло хорошо."""
        self.sections["what_went_well"].append(item)

    def add_negative(self, item):
        """Что прошло плохо."""
        self.sections["what_went_wrong"].append(item)

    def add_action_item(self, action, owner, priority, deadline):
        """Добавление пункта действий."""
        self.sections["action_items"].append({
            "action": action,
            "owner": owner,
            "priority": priority,
            "deadline": deadline,
            "completed": False,
        })

    def add_lesson(self, lesson):
        """Добавление урока."""
        self.sections["lessons_learned"].append(lesson)

    def add_detection_gap(self, gap):
        """Добавление пробела в обнаружении."""
        self.sections["detection_gaps"].append(gap)

    def add_prevention(self, measure):
        """Добавление меры предотвращения."""
        self.sections["prevention_measures"].append(measure)

    def check_blamelessness(self):
        """Проверка на обвинительный тон."""
        blame_indicators = [
            "виноват", "некомпетент", "глуп", "ошибся",
            "должен был", "разбаловал", "ленив",
        ]
        all_text = " ".join(
            item for items in self.sections.values()
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, str)
        )
        found = [ind for ind in blame_indicators if ind in all_text.lower()]
        self.blame_check_passed = len(found) == 0
        return self.blame_check_passed, found

    def generate_report(self):
        """Генерация отчёта постмортема."""
        lines = [
            f"=== ПОСТМОРТЕМ: {self.title} ===",
            f"Инцидент: {self.incident_id}",
            "",
            "--- ТАЙМЛАЙН ---",
        ]
        for entry in self.sections["timeline"]:
            lines.append(f"  [{entry['timestamp']}] {entry['event']}")
            if entry["impact"]:
                lines.append(f"    Влияние: {entry['impact']}")

        lines.append("\n--- ЧТО ПРОШЛО ХОРОШО ---")
        for item in self.sections["what_went_well"]:
            lines.append(f"  + {item}")

        lines.append("\n--- ЧТО ПРОШЛО ПЛОХО ---")
        for item in self.sections["what_went_wrong"]:
            lines.append(f"  - {item}")

        lines.append("\n--- ПУНКТЫ ДЕЙСТВИЙ ---")
        for item in self.sections["action_items"]:
            status = "✓" if item["completed"] else "○"
            lines.append(f"  {status} [{item['priority']}] {item['action']}")
            lines.append(f"    Владелец: {item['owner']}, Дедлайн: {item['deadline']}")

        lines.append("\n--- УРОКИ ---")
        for lesson in self.sections["lessons_learned"]:
            lines.append(f"  * {lesson}")

        return "\n".join(lines)


class SafetyCultureMetrics:
    """Метрики культуры безопасности."""

    def __init__(self):
        self.incidents = []
        self.postmortems = []
        self.improvements = []

    def record_incident(self, incident_data):
        """Регистрация инцидента."""
        self.incidents.append(incident_data)

    def record_postmortem(self, postmortem_data):
        """Регистрация постмортема."""
        self.postmortems.append(postmortem_data)

    def record_improvement(self, improvement_data):
        """Регистрация улучшения."""
        self.improvements.append(improvement_data)

    def calculate_metrics(self):
        """Расчёт метрик культуры безопасности."""
        total_incidents = len(self.incidents)
        total_postmortems = len(self.postmortems)
        total_improvements = len(self.improvements)

        # Процент инцидентов с постмортемами
        postmortem_rate = (
            total_postmortems / total_incidents * 100
            if total_incidents > 0 else 0
        )

        # Среднее время обнаружения (моделируемое)
        detection_times = [inc.get("detection_time_hours", 0) for inc in self.incidents]
        avg_detection = sum(detection_times) / len(detection_times) if detection_times else 0

        # Процент исправленных action items
        all_actions = [pm.get("action_items_count", 0) for pm in self.postmortems]
        completed_actions = [pm.get("completed_actions", 0) for pm in self.postmortems]
        fix_rate = (
            sum(completed_actions) / sum(all_actions) * 100
            if sum(all_actions) > 0 else 0
        )

        return {
            "total_incidents": total_incidents,
            "total_postmortems": total_postmortems,
            "postmortem_rate": round(postmortem_rate, 1),
            "avg_detection_time_hours": round(avg_detection, 1),
            "total_improvements": total_improvements,
            "action_fix_rate": round(fix_rate, 1),
        }

    def get_trend(self, period="monthly"):
        """Тренд инцидентов по периодам."""
        # Группировка по «месяцам»
        monthly = collections.Counter()
        for inc in self.incidents:
            month = inc.get("month", 1)
            monthly[month] += 1
        return dict(sorted(monthly.items()))


def demo_safety_culture():
    """Демонстрация культуры безопасности."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: SAFETY CULTURE")
    print("Отчётность об инцидентах, безобвинительные постмортемы, улучшение")
    print("=" * 70)

    # --- 4.1 Отчёт об инциденте ---
    print("\n--- 4.1 Отчёт об инциденте ---")
    incident = IncidentReport(
        "INC-2024-001",
        "Утечка персональных данных из модели",
        "critical",
        "Модель раскрыла email и телефон пользователя в публичном чате",
        "Иванов И.И.",
    )

    incident.add_root_cause("Отсутствие фильтра PII на выходе", "process")
    incident.add_root_cause("Недостаточное тестирование", "testing")
    incident.add_remediation("Внедрить PII-фильтр", "alice", "2024-02-01")
    incident.add_remediation("Добавить тесты на утечку данных", "bob", "2024-02-15")

    summary = incident.get_summary()
    print(f"  Инцидент: {summary['id']} — {summary['title']}")
    print(f"  Серьёзность: {summary['severity']}")
    print(f"  Корневых причин: {summary['root_causes']}")
    print(f"  Мер по устранению: {summary['remediations']}")

    # --- 4.2 Безобвинительный постмортем ---
    print("\n--- 4.2 Безобвинительный постмортем ---")
    postmortem = BlamelessPostmortem("INC-2024-001", "Утечка PII")

    postmortem.add_timeline_entry("14:00", "Пользователь отправил запрос с email")
    postmortem.add_timeline_entry("14:01", "Модель вернула ответ с email")
    postmortem.add_timeline_entry("14:05", "Пользователь сообщил о проблеме")
    postmortem.add_timeline_entry("14:15", "Команда подтверждает утечку")
    postmortem.add_timeline_entry("14:30", "Модель временно отключена")

    postmortem.add_positive("Быстрое отключение модели после обнаружения")
    postmortem.add_positive("Пользователь вовремя сообщил")

    postmortem.add_negative("Отсутствовал PII-фильтр")
    postmortem.add_negative("Не было мониторинга утечек")

    postmortem.add_action_item("Внедрить PII-фильтр v2", "alice", "P0", "2024-02-01")
    postmortem.add_action_item("Настроить мониторинг", "bob", "P1", "2024-02-15")
    postmortem.add_action_item("Обновить тесты", "charlie", "P1", "2024-02-15")

    postmortem.add_lesson("Фильтрация на выходе критична для защиты данных")
    postmortem.add_lesson("Мониторинг должен работать в реальном времени")

    postmortem.add_detection_gap("Нет автоматического обнаружения PII в выводе")
    postmortem.add_prevention("Внедрить автоматический сканер PII в CI/CD")

    # Проверка на обвинительность
    is_blameless, blame_words = postmortem.check_blamelessness()
    blame_status = "✓ ПРОЙДЕНА" if is_blameless else "✗ НЕ ПРОЙДЕНА"
    print(f"  Проверка на обвинительность: {blame_status}")
    if blame_words:
        print(f"  Найдены обвинительные слова: {blame_words}")

    # --- 4.3 Генерация отчёта ---
    print("\n--- 4.3 Отчёт постмортема ---")
    print(postmortem.generate_report())

    # --- 4.4 Метрики культуры ---
    print("\n--- 4.4 Метрики культуры безопасности ---")
    metrics = SafetyCultureMetrics()

    # Добавление данных
    for i in range(5):
        metrics.record_incident({
            "id": f"INC-{i:03d}",
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "detection_time_hours": random.uniform(0.5, 24),
            "month": random.randint(1, 12),
        })
    metrics.record_incident({
        "id": "INC-005",
        "severity": "critical",
        "detection_time_hours": 0.5,
        "month": 1,
    })

    for i in range(4):
        metrics.record_postmortem({
            "incident_id": f"INC-{i:03d}",
            "action_items_count": random.randint(2, 5),
            "completed_actions": random.randint(1, 4),
        })

    for i in range(8):
        metrics.record_improvement({
            "type": random.choice(["process", "tool", "training"]),
            "impact": random.choice(["high", "medium", "low"]),
        })

    calc_metrics = metrics.calculate_metrics()
    print(f"  Всего инцидентов: {calc_metrics['total_incidents']}")
    print(f"  Постмортемов: {calc_metrics['total_postmortems']}")
    print(f"  Процент постмортемов: {calc_metrics['postmortem_rate']}%")
    print(f"  Среднее время обнаружения: {calc_metrics['avg_detection_time_hours']} ч")
    print(f"  Улучшений: {calc_metrics['total_improvements']}")
    print(f"  Процент исправлений: {calc_metrics['action_fix_rate']}%")

    trend = metrics.get_trend()
    print("\n  Тренд инцидентов по месяцам:")
    for month, count in trend.items():
        bar = "█" * count + "░" * (5 - count)
        print(f"    Месяц {month:2d}: |{bar}| {count}")

    print("\n--- ВЫВОД ---")
    print("Культура безопасности — это:")
    print("  1. Систематическая отчётность об инцидентах")
    print("  2. Безобвинительные постмортемы для извлечения уроков")
    print("  3. Количественная оценка эффективности")
    print("  4. Непрерывное улучшение процессов")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo_safety_cases()
    demo_verification_methods()
    demo_containment_strategies()
    demo_safety_culture()