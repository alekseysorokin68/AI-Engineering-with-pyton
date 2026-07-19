"""161 — Agent Evaluation: бенчмарки, метрики, режимы отказа

Темы:
  1. Evaluation Metrics — success rate, step efficiency, cost, latency
  2. Benchmark Tasks — tool use accuracy, planning quality, multi-step reasoning
  3. Failure Modes — hallucination, tool misuse, infinite loops, goal deviation
  4. Ablation Studies — tool importance, prompt sensitivity, memory impact

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


# ─────────────────────────────────────────────────────────────
# 1. Evaluation Metrics — метрики оценки агентов
# ─────────────────────────────────────────────────────────────

class AgentEvaluator:
    """Оценщик производительности агентов: метрики success rate, efficiency, cost, latency."""

    def __init__(self):
        self.results = []

    def record(self, task_id: str, success: bool, steps_taken: int,
               steps_optimal: int, cost: float, latency_ms: float,
               agent_name: str = "default"):
        """Записать результат выполнения задачи."""
        self.results.append({
            "task_id": task_id,
            "success": success,
            "steps_taken": steps_taken,
            "steps_optimal": steps_optimal,
            "cost": cost,
            "latency_ms": latency_ms,
            "agent_name": agent_name,
        })

    def success_rate(self) -> float:
        """Success Rate = |успешных задач| / |всех задач| × 100%"""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r["success"])
        return successful / len(self.results) * 100

    def step_efficiency(self) -> float:
        """Step Efficiency = steps_optimal / steps_taken (1.0 = идеально)"""
        if not self.results:
            return 0.0
        efficiencies = []
        for r in self.results:
            if r["steps_taken"] > 0:
                eff = r["steps_optimal"] / r["steps_taken"]
                efficiencies.append(min(1.0, eff))
        return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0

    def avg_cost(self) -> float:
        """Средняя стоимость выполнения задачи."""
        if not self.results:
            return 0.0
        return sum(r["cost"] for r in self.results) / len(self.results)

    def avg_latency(self) -> float:
        """Средняя задержка (latency) в миллисекундах."""
        if not self.results:
            return 0.0
        return sum(r["latency_ms"] for r in self.results) / len(self.results)

    def cost_per_success(self) -> float:
        """Стоимость на одну успешную задачу."""
        successful = [r for r in self.results if r["success"]]
        if not successful:
            return float('inf')
        return sum(r["cost"] for r in successful) / len(successful)

    def composite_score(self) -> float:
        """Композитный скор = α × success_rate + β × step_efficiency - γ × normalized_cost
        Нормализация: cost / max_cost, latency / max_latency"""
        sr = self.success_rate() / 100
        se = self.step_efficiency()
        costs = [r["cost"] for r in self.results] if self.results else [1]
        max_cost = max(costs) if costs else 1
        avg_normalized_cost = sum(r["cost"] for r in self.results) / (len(self.results) * max_cost) if self.results else 0
        # α=0.5, β=0.3, γ=0.2
        score = 0.5 * sr + 0.3 * se - 0.2 * avg_normalized_cost
        return max(0.0, min(1.0, score))

    def report(self) -> dict:
        """Полный отчёт по метрикам."""
        return {
            "tasks_total": len(self.results),
            "tasks_successful": sum(1 for r in self.results if r["success"]),
            "success_rate_%": round(self.success_rate(), 1),
            "step_efficiency": round(self.step_efficiency(), 3),
            "avg_cost": round(self.avg_cost(), 4),
            "avg_latency_ms": round(self.avg_latency(), 1),
            "cost_per_success": round(self.cost_per_success(), 4),
            "composite_score": round(self.composite_score(), 3),
        }


def demo_evaluation_metrics():
    """Демонстрация: success rate, step efficiency, cost, latency."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: МЕТРИКИ ОЦЕНКИ (Evaluation Metrics)")
    print("=" * 70)

    evaluator = AgentEvaluator()

    # --- Пример 1: Запись результатов ---
    print("\n--- Пример 1: Запись результатов выполнения задач ---")
    tasks = [
        ("T1", True, 5, 4, 0.02, 150.0),
        ("T2", True, 3, 3, 0.01, 80.0),
        ("T3", False, 10, 4, 0.05, 300.0),
        ("T4", True, 6, 5, 0.03, 200.0),
        ("T5", True, 4, 4, 0.015, 100.0),
        ("T6", False, 8, 5, 0.04, 250.0),
        ("T7", True, 7, 5, 0.035, 180.0),
        ("T8", True, 3, 3, 0.01, 70.0),
    ]
    for task_id, success, steps, optimal, cost, latency in tasks:
        evaluator.record(task_id, success, steps, optimal, cost, latency)

    print(f"  Записано {len(evaluator.results)} результатов:")
    for r in evaluator.results:
        status = "✓" if r["success"] else "✗"
        print(f"    {r['task_id']}: {status} steps={r['steps_taken']}/{r['steps_optimal']}"
              f" cost=${r['cost']:.3f} latency={r['latency_ms']:.0f}ms")

    # --- Пример 2: Success Rate ---
    print("\n--- Пример 2: Success Rate ---")
    sr = evaluator.success_rate()
    total = len(evaluator.results)
    successful = sum(1 for r in evaluator.results if r["success"])
    print(f"  Формула: SR = |успешных| / |всех| × 100%")
    print(f"  SR = {successful} / {total} × 100% = {sr:.1f}%")

    # --- Пример 3: Step Efficiency ---
    print("\n--- Пример 3: Step Efficiency ---")
    se = evaluator.step_efficiency()
    print(f"  Формула: SE = mean(min(1, steps_optimal / steps_taken))")
    print(f"  Step Efficiency = {se:.3f} ({se:.1%} от идеала)")
    for r in evaluator.results:
        eff = r["steps_optimal"] / r["steps_taken"] if r["steps_taken"] > 0 else 0
        eff_capped = min(1.0, eff)
        print(f"    {r['task_id']}: {r['steps_optimal']}/{r['steps_taken']} = {eff:.2f} → {eff_capped:.2f}")

    # --- Пример 4: Полный отчёт ---
    print("\n--- Пример 4: Композитный скор и полный отчёт ---")
    report = evaluator.report()
    print("  Полный отчёт:")
    for key, value in report.items():
        print(f"    {key}: {value}")
    print(f"\n  Композитная формула:")
    print(f"    Score = 0.5 × SR/100 + 0.3 × SE - 0.2 × avg_norm_cost")
    print(f"    Score = 0.5 × {report['success_rate_%']/100:.3f} + "
          f"0.3 × {report['step_efficiency']:.3f} - "
          f"0.2 × normalized_cost")
    print(f"    Score = {report['composite_score']:.3f}")

    print()


# ─────────────────────────────────────────────────────────────
# 2. Benchmark Tasks — бенчмарки для агентов
# ─────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """Набор тестов для оценки способностей агента."""

    def __init__(self):
        self.tasks = []

    def tool_use_accuracy(self, agent_calls: list, expected_calls: list) -> dict:
        """Оценка точности использования инструментов.
        Accuracy = |correct_calls| / |total_calls|"""
        # Проверяем, какие вызовы агента совпадают с ожидаемыми
        correct = 0
        details = []
        for actual, expected in zip(agent_calls, expected_calls):
            is_correct = actual == expected
            if is_correct:
                correct += 1
            details.append({
                "actual": actual,
                "expected": expected,
                "correct": is_correct,
            })
        accuracy = correct / len(expected_calls) if expected_calls else 0
        return {"accuracy": accuracy, "correct": correct, "total": len(expected_calls), "details": details}

    def planning_quality(self, plan: list, optimal_plan: list) -> dict:
        """Оценка качества планирования.
        Сравниваем порядок и полноту шагов.
        Q = 1 - (edit_distance / max(len(plan), len(optimal)))"""
        # Расстояние Левенштейна (упрощённое)
        def edit_distance(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if s1[i-1] == s2[j-1] else 1
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # удаление
                        dp[i][j-1] + 1,      # вставка
                        dp[i-1][j-1] + cost  # замена
                    )
            return dp[m][n]

        dist = edit_distance(plan, optimal_plan)
        max_len = max(len(plan), len(optimal_plan))
        quality = 1 - dist / max_len if max_len > 0 else 1.0

        # Полнота: доля оптимальных шагов, присутствующих в плане
        plan_set = set(plan)
        optimal_set = set(optimal_plan)
        coverage = len(plan_set & optimal_set) / len(optimal_set) if optimal_set else 0

        return {
            "quality": max(0, quality),
            "edit_distance": dist,
            "coverage": coverage,
            "plan_length": len(plan),
            "optimal_length": len(optimal_plan),
        }

    def multi_step_reasoning(self, steps: list, correct_answers: list) -> dict:
        """Оценка многошагового рассуждения.
        Проверяем промежуточные и итоговый ответы."""
        correct = 0
        details = []
        for i, (step, expected) in enumerate(zip(steps, correct_answers)):
            is_correct = step == expected
            if is_correct:
                correct += 1
            details.append({
                "step": i + 1,
                "actual": step,
                "expected": expected,
                "correct": is_correct,
            })
        return {
            "accuracy": correct / len(correct_answers) if correct_answers else 0,
            "intermediate_correct": correct - (1 if steps and steps[-1] == correct_answers[-1] else 0),
            "final_correct": steps[-1] == correct_answers[-1] if steps and correct_answers else False,
            "details": details,
        }


def demo_benchmark_tasks():
    """Демонстрация: tool use accuracy, planning quality, multi-step reasoning."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: БЕНЧМАРКИ (Benchmark Tasks)")
    print("=" * 70)

    suite = BenchmarkSuite()

    # --- Пример 1: Tool Use Accuracy ---
    print("\n--- Пример 1: Точность использования инструментов ---")
    # Агент вызвал инструменты: web_search, calculator, web_search, file_read
    agent_calls = ["web_search", "calculator", "web_search", "file_read", "calculator"]
    # Ожидалось:           web_search, calculator, file_read, calculator
    expected_calls = ["web_search", "calculator", "file_read", "calculator"]

    result = suite.tool_use_accuracy(agent_calls, expected_calls)
    print(f"  Вызовы агента:   {agent_calls}")
    print(f"  Ожидаемые вызовы: {expected_calls}")
    print(f"  Точность: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    for d in result['details']:
        status = "✓" if d['correct'] else "✗"
        print(f"    {status} actual={d['actual']:15s} expected={d['expected']:15s}")

    # --- Пример 2: Planning Quality ---
    print("\n--- Пример 2: Качество планирования ---")
    # Оптимальный план
    optimal = ["read_file", "parse_data", "validate", "transform", "save", "report"]
    # План агента (хороший, но не идеальный)
    agent_plan = ["read_file", "parse_data", "validate", "transform", "save"]
    # Плохой план агента
    bad_plan = ["save", "read_file", "delete", "parse_data"]

    good_result = suite.planning_quality(agent_plan, optimal)
    bad_result = suite.planning_quality(bad_plan, optimal)

    print(f"  Оптимальный план: {optimal}")
    print(f"\n  Хороший план: {agent_plan}")
    print(f"    Качество: {good_result['quality']:.3f}")
    print(f"    Расстояние Левенштейна: {good_result['edit_distance']}")
    print(f"    Покрытие: {good_result['coverage']:.1%}")

    print(f"\n  Плохой план: {bad_plan}")
    print(f"    Качество: {bad_result['quality']:.3f}")
    print(f"    Расстояние Левенштейна: {bad_result['edit_distance']}")
    print(f"    Покрытие: {bad_result['coverage']:.1%}")

    # --- Пример 3: Multi-Step Reasoning ---
    print("\n--- Пример 3: Многошаговое рассуждение ---")
    # Задача: вычислить выражение шаг за шагом
    # Шаг 1: 2 + 3 = 5
    # Шаг 2: 5 × 4 = 20
    # Шаг 3: 20 / 2 = 10
    # Шаг 4: 10 + 1 = 11
    steps = [5, 20, 10, 11]
    correct = [5, 20, 10, 11]
    result = suite.multi_step_reasoning(steps, correct)
    print(f"  Задача: ((2+3)×4)/2 + 1")
    print(f"  Промежуточные ответы агента: {steps}")
    print(f"  Правильные ответы:           {correct}")
    print(f"  Точность: {result['accuracy']:.1%}")
    print(f"  Все промежуточные шаги верны: {result['final_correct']}")

    # Ошибка на шаге 3
    steps_wrong = [5, 20, 9, 10]
    result_wrong = suite.multi_step_reasoning(steps_wrong, correct)
    print(f"\n  Агент ошибся на шаге 3: {steps_wrong}")
    print(f"  Точность: {result_wrong['accuracy']:.1%}")
    for d in result_wrong['details']:
        status = "✓" if d['correct'] else "✗"
        print(f"    Шаг {d['step']}: {status} actual={d['actual']} expected={d['expected']}")

    # --- Пример 4: Сравнение агентов ---
    print("\n--- Пример 4: Сравнение агентов на бенчмарке ---")
    agents_perf = {
        "Agent_A": {"success_rate": 92, "step_eff": 0.85, "tool_acc": 0.90, "plan_qual": 0.88},
        "Agent_B": {"success_rate": 88, "step_eff": 0.92, "tool_acc": 0.85, "plan_qual": 0.80},
        "Agent_C": {"success_rate": 95, "step_eff": 0.78, "tool_acc": 0.95, "plan_qual": 0.90},
    }
    print(f"  {'Агент':12s} {'SR%':>6s} {'StepEff':>8s} {'ToolAcc':>8s} {'PlanQual':>9s}")
    print(f"  {'-'*45}")
    for name, perf in agents_perf.items():
        print(f"  {name:12s} {perf['success_rate']:5.1f}% {perf['step_eff']:.3f}   {perf['tool_acc']:.3f}   {perf['plan_qual']:.3f}")

    # Композитный скор для каждого агента
    print("\n  Композитный скор (0.3×SR + 0.3×SE + 0.2×TA + 0.2×PQ):")
    for name, perf in agents_perf.items():
        composite = (0.3 * perf['success_rate'] / 100 + 0.3 * perf['step_eff']
                     + 0.2 * perf['tool_acc'] + 0.2 * perf['plan_qual'])
        bar = "█" * int(composite * 40)
        print(f"    {name}: {composite:.3f} {bar}")

    print()


# ─────────────────────────────────────────────────────────────
# 3. Failure Modes — режимы отказа агентов
# ─────────────────────────────────────────────────────────────

class FailureDetector:
    """Детектор режимов отказа агентов."""

    def __init__(self):
        self.failures = []

    def detect_hallucination(self, response: str, context: str) -> dict:
        """Обнаружение галлюцинаций: факты в ответе, которых нет в контексте.
        Простая эвристика: ключевые слова ответа, отсутствующие в контексте."""
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))
        # Слова в ответе, которых нет в контексте (потенциальные галлюцинации)
        hallucinated = response_words - context_words
        total_content_words = len(response_words)
        hallucinated_ratio = len(hallucinated) / total_content_words if total_content_words else 0
        return {
            "is_hallucination": hallucinated_ratio > 0.3,
            "hallucinated_words": list(hallucinated)[:5],
            "ratio": hallucinated_ratio,
            "confidence": "high" if hallucinated_ratio > 0.5 else "medium" if hallucinated_ratio > 0.3 else "low",
        }

    def detect_tool_misuse(self, tool_calls: list, task_description: str) -> dict:
        """Обнаружение неправильного использования инструментов.
        Проверяем: повторные вызовы, вызовы без результатов, нерелевантные инструменты."""
        issues = []
        # Проверка на повторные вызовы с теми же аргументами
        seen_calls = collections.Counter()
        for call in tool_calls:
            call_key = f"{call['tool']}({call.get('args', '')})"
            seen_calls[call_key] += 1
        for call_key, count in seen_calls.items():
            if count > 2:
                issues.append(f"Повторный вызов {count} раз: {call_key}")

        # Проверка на пустые результаты
        empty_results = sum(1 for call in tool_calls if call.get('result', '') == '')
        if empty_results > 0:
            issues.append(f"Пустые результаты: {empty_results} вызов(ов)")

        return {
            "has_misuse": len(issues) > 0,
            "issues": issues,
            "total_calls": len(tool_calls),
        }

    def detect_infinite_loop(self, actions: list, max_repeats: int = 3) -> dict:
        """Обнаружение бесконечных циклов: повторяющиеся действия.
        Проверяем на наличие паттерна, повторяющегося более max_repeats раз."""
        # Ищем повторяющиеся подпоследовательности длины 2-5
        for pattern_len in range(2, min(6, len(actions) // 2 + 1)):
            for start in range(len(actions) - pattern_len * 2 + 1):
                pattern = tuple(actions[start:start + pattern_len])
                repeats = 0
                for i in range(start, len(actions) - pattern_len + 1, pattern_len):
                    if tuple(actions[i:i + pattern_len]) == pattern:
                        repeats += 1
                    else:
                        break
                if repeats >= max_repeats:
                    return {
                        "is_loop": True,
                        "pattern": list(pattern),
                        "repeats": repeats,
                        "position": start,
                    }
        return {"is_loop": False, "pattern": None, "repeats": 0}

    def detect_goal_deviation(self, steps: list, original_goal: str, achieved_goal: str) -> dict:
        """Обнаружение отклонения от цели: агент решал не ту задачу."""
        # Сравниваем ключевые слова цели и результата
        goal_words = set(re.findall(r'\b\w{3,}\b', original_goal.lower()))
        achieved_words = set(re.findall(r'\b\w{3,}\b', achieved_goal.lower()))
        overlap = goal_words & achieved_words
        deviation = 1.0 - len(overlap) / len(goal_words) if goal_words else 0
        return {
            "has_deviation": deviation > 0.5,
            "deviation_score": deviation,
            "goal_overlap": len(overlap),
            "goal_size": len(goal_words),
            "shared_keywords": list(overlap)[:5],
        }


def demo_failure_modes():
    """Демонстрация: галлюцинации, неправильное использование, бесконечные циклы, отклонение от цели."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: РЕЖИМЫ ОТКАЗА (Failure Modes)")
    print("=" * 70)

    detector = FailureDetector()

    # --- Пример 1: Галлюцинации ---
    print("\n--- Пример 1: Обнаружение галлюцинаций ---")
    context = "Python — язык программирования. PyTorch — фреймворк для глубокого обучения."
    response_good = "Python используется для программирования, PyTorch для обучения моделей"
    response_bad = "Python разработан SpaceX для управления ракетами и Mars colonization"

    result_good = detector.detect_hallucination(response_good, context)
    result_bad = detector.detect_hallucination(response_bad, context)

    print(f"  Контекст: '{context}'")
    print(f"\n  Ответ 1 (хороший): '{response_good}'")
    print(f"    Галлюцинация: {result_good['is_hallucination']}, confidence={result_good['confidence']}")
    print(f"    Рatio галлюциногенных слов: {result_good['ratio']:.2f}")
    print(f"\n  Ответ 2 (плохой): '{response_bad}'")
    print(f"    Галлюцинация: {result_bad['is_hallucination']}, confidence={result_bad['confidence']}")
    print(f"    Подозрительные слова: {result_bad['hallucinated_words']}")
    print(f"    Ratio: {result_bad['ratio']:.2f}")

    # --- Пример 2: Tool Misuse ---
    print("\n--- Пример 2: Неправильное использование инструментов ---")
    tool_calls = [
        {"tool": "web_search", "args": "Python docs", "result": "found"},
        {"tool": "web_search", "args": "Python docs", "result": "found"},  # Повтор
        {"tool": "web_search", "args": "Python docs", "result": "found"},  # Повтор
        {"tool": "calculator", "args": "2+2", "result": ""},  # Пустой результат
        {"tool": "file_read", "args": "data.csv", "result": "data..."},
    ]
    result = detector.detect_tool_misuse(tool_calls, "анализ данных")
    print(f"  Инструменты: {[c['tool'] for c in tool_calls]}")
    print(f"  Проблемы обнаружены: {result['has_misuse']}")
    for issue in result['issues']:
        print(f"    ⚠ {issue}")

    # --- Пример 3: Бесконечные циклы ---
    print("\n--- Пример 3: Обнаружение бесконечных циклов ---")
    actions_loop = ["search", "analyze", "retry", "search", "analyze", "retry", "search", "analyze", "retry"]
    actions_normal = ["search", "analyze", "plan", "execute", "verify"]

    result_loop = detector.detect_infinite_loop(actions_loop)
    result_normal = detector.detect_infinite_loop(actions_normal)

    print(f"  Действия (с циклом): {actions_loop}")
    print(f"    Бесконечный цикл: {result_loop['is_loop']}")
    if result_loop['is_loop']:
        print(f"    Паттерн: {result_loop['pattern']}, повторений: {result_loop['repeats']}")

    print(f"\n  Действия (нормальные): {actions_normal}")
    print(f"    Бесконечный цикл: {result_normal['is_loop']}")

    # --- Пример 4: Отклонение от цели ---
    print("\n--- Пример 4: Обнаружение отклонения от цели ---")
    goal = "Обучить модель классификации изображений на датасете MNIST"
    achieved_good = "Модель классификации обучена на датасете MNIST с точностью 98%"
    achieved_bad = "Настроил веб-сервер для развертывания приложения"

    result_good = detector.detect_goal_deviation([], goal, achieved_good)
    result_bad = detector.detect_goal_deviation([], goal, achieved_bad)

    print(f"  Цель: '{goal}'")
    print(f"\n  Результат 1: '{achieved_good}'")
    print(f"    Отклонение: {result_good['has_deviation']} (score={result_good['deviation_score']:.2f})")
    print(f"    Общие ключевые слова: {result_good['shared_keywords']}")

    print(f"\n  Результат 2: '{achieved_bad}'")
    print(f"    Отклонение: {result_bad['has_deviation']} (score={result_bad['deviation_score']:.2f})")
    print(f"    Общие ключевые слова: {result_bad['shared_keywords']}")

    print()


# ─────────────────────────────────────────────────────────────
# 4. Ablation Studies — аблиационные исследования
# ─────────────────────────────────────────────────────────────

class AblationStudy:
    """Аблиационное исследование: влияние компонентов на производительность."""

    def __init__(self, baseline_score: float):
        self.baseline = baseline_score
        self.ablation_results = []

    def remove_component(self, component_name: str, score_without: float, score_with: float):
        """Зафиксировать результат удаления компонента.
        impact = baseline - score_without (больше = важнее компонент)."""
        impact = self.baseline - score_without
        impact_pct = (impact / self.baseline * 100) if self.baseline else 0
        self.ablation_results.append({
            "component": component_name,
            "score_with": score_with,
            "score_without": score_without,
            "impact": impact,
            "impact_pct": impact_pct,
        })

    def rank_by_importance(self) -> list:
        """Ранжировать компоненты по важности (убывание impact)."""
        return sorted(self.ablation_results, key=lambda x: x["impact"], reverse=True)

    def prompt_sensitivity(self, prompt_variants: dict) -> dict:
        """Анализ чувствительности к промпту.
        variance = Var(scores), sensitivity = std/mean"""
        scores = list(prompt_variants.values())
        n = len(scores)
        mean_score = sum(scores) / n
        variance = sum((s - mean_score) ** 2 for s in scores) / n
        std_dev = math.sqrt(variance)
        sensitivity = std_dev / mean_score if mean_score > 0 else 0
        return {
            "variants": prompt_variants,
            "mean": mean_score,
            "std": std_dev,
            "variance": variance,
            "sensitivity": sensitivity,
            "interpretation": (
                "Высокая" if sensitivity > 0.1 else
                "Средняя" if sensitivity > 0.05 else
                "Низкая"
            ),
        }

    def memory_impact(self, memory_configs: dict) -> dict:
        """Влияние размера памяти на производительность.
        Анализ: score = a + b × log(memory_size)"""
        configs = list(memory_configs.items())
        if len(configs) < 2:
            return {"error": "Нужно >= 2 конфигураций"}

        # Линейная регрессия: score = a + b × log(memory)
        x_vals = [math.log2(size) for size, score in configs]
        y_vals = [score for size, score in configs]
        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        # Коэффициент b = Cov(x,y) / Var(x)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / n
        var_x = sum((x - x_mean) ** 2 for x in x_vals) / n
        b = cov / var_x if var_x else 0
        a = y_mean - b * x_mean

        # R² (коэффициент детерминации)
        y_pred = [a + b * x for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0

        return {
            "formula": f"score = {a:.3f} + {b:.3f} × log2(memory_size)",
            "a": a,
            "b": b,
            "r_squared": r_squared,
            "interpretation": (
                "Сильная логарифмическая зависимость" if r_squared > 0.9 else
                "Умеренная зависимость" if r_squared > 0.7 else
                "Слабая зависимость"
            ),
        }


def demo_ablation_studies():
    """Демонстрация: важность инструментов, чувствительность промпта, влияние памяти."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: АБЛИАЦИОННЫЕ ИССЛЕДОВАНИЯ (Ablation Studies)")
    print("=" * 70)

    # --- Пример 1: Tool Importance ---
    print("\n--- Пример 1: Важность компонентов (ablation) ---")
    baseline = 0.85
    ablation = AblationStudy(baseline)

    # Удаляем компоненты и измеряем качество
    ablation.remove_component("web_search", 0.50, 0.85)   # Без поиска — сильно падает
    ablation.remove_component("calculator", 0.78, 0.85)    # Без калькулятора — немного
    ablation.remove_component("memory", 0.60, 0.85)        # Без памяти — сильно падает
    ablation.remove_component("prompt_engineering", 0.70, 0.85)  # Без промптинга — средне
    ablation.remove_component("chain_of_thought", 0.65, 0.85)    # Без CoT — средне

    print(f"  Базовый скор (все компоненты): {baseline}")
    print(f"  Результаты аблиации:")
    ranked = ablation.rank_by_importance()
    for i, r in enumerate(ranked, 1):
        bar = "█" * int(r["impact"] * 100)
        print(f"    {i}. {r['component']:22s}: без={r['score_without']:.2f}, "
              f"impact={r['impact']:.2f} ({r['impact_pct']:.1f}%) {bar}")

    # --- Пример 2: Prompt Sensitivity ---
    print("\n--- Пример 2: Чувствительность к промпту ---")
    # Разные формулировки одного промпта
    prompts = {
        "Прямой": 0.82,
        "С инструкциями": 0.88,
        "С примерами": 0.91,
        "С CoT": 0.87,
        "Разговорный": 0.79,
        "Формальный": 0.85,
    }
    sensitivity_result = ablation.prompt_sensitivity(prompts)
    print(f"  Варианты промпта и их скоры:")
    for variant, score in sorted(prompts.items(), key=lambda x: -x[1]):
        print(f"    {variant:18s}: {score:.2f}")
    print(f"\n  Анализ:")
    print(f"    Среднее: {sensitivity_result['mean']:.3f}")
    print(f"    Стд. отклонение: {sensitivity_result['std']:.3f}")
    print(f"    Чувствительность: {sensitivity_result['sensitivity']:.3f}")
    print(f"    Интерпретация: {sensitivity_result['interpretation']}")

    # --- Пример 3: Memory Impact ---
    print("\n--- Пример 3: Влияние размера памяти ---")
    # memory_size → score (логарифмическая зависимость)
    memory_configs = {
        8: 0.60,
        16: 0.72,
        32: 0.80,
        64: 0.85,
        128: 0.88,
        256: 0.90,
        512: 0.91,
    }
    mem_result = ablation.memory_impact(memory_configs)
    print("  Размер памяти → Скор:")
    for size, score in memory_configs.items():
        bar = "█" * int(score * 40)
        print(f"    {size:4d} entries: {score:.2f} {bar}")
    print(f"\n  Модель: {mem_result['formula']}")
    print(f"  R² = {mem_result['r_squared']:.3f}")
    print(f"  Интерпретация: {mem_result['interpretation']}")
    print(f"  Вывод: добавление памяти даёт убывающую отдачу (log-зависимость)")

    # --- Пример 4: Сводная таблица ---
    print("\n--- Пример 4: Сводная таблица аблиаций ---")
    components = ["tool_use", "memory", "cot", "prompt", "fine_tuning"]
    scores = [0.85, 0.70, 0.78, 0.82, 0.88]
    impacts = [baseline - s for s in scores]

    print(f"  {'Компонент':18s} {'Скор':>6s} {'Impact':>8s} {'Важность':>10s}")
    print(f"  {'-'*45}")
    for comp, score, impact in sorted(zip(components, scores, impacts), key=lambda x: -x[2]):
        importance = "Высокая" if impact > 0.15 else "Средняя" if impact > 0.08 else "Низкая"
        print(f"  {comp:18s} {score:.2f}  {impact:7.3f}  {importance}")

    print()


# ─────────────────────────────────────────────────────────────
# Запуск всех демонстраций
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  161 — Agent Evaluation: бенчмарки, метрики, режимы отказа")
    print("=" * 70 + "\n")

    demo_evaluation_metrics()
    demo_benchmark_tasks()
    demo_failure_modes()
    demo_ablation_studies()

    print("=" * 70)
    print("  Все 4 демонстрации завершены успешно!")
    print("=" * 70)
