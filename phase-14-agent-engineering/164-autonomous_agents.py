"""164 — Autonomous Agents: автономность, саморефлексия, восстановление ошибок

Темы:
  1. Goal-Driven Behavior (goal setting, progress tracking, termination criteria)
  2. Self-Reflection (output critique, quality assessment, improvement suggestions)
  3. Error Recovery (retry strategies, fallback plans, graceful degradation)
  4. Autonomous Workflows (long-running tasks, checkpointing, human-in-the-loop)

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


# =============================================================================
# 1. Goal-Driven Behavior
# =============================================================================

def demo_goal_driven():
    """Демонстрация поведения, управляемого целями."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: Goal-Driven Behavior")
    print("=" * 70)

    # --- 1.1 Goal Setting ---
    print("\n--- 1.1 Постановка целей ---")

    # Система постановки целей
    class Goal:
        """Одна цель агента."""
        def __init__(self, name, description, priority=1, deadline=None):
            self.name = name
            self.description = description
            self.priority = priority
            self.deadline = deadline
            self.status = "pending"  # pending, in_progress, completed, failed
            self.progress = 0  # 0-100%
            self.sub_goals = []

        def add_sub_goal(self, sub_goal):
            """Добавляет подцель."""
            self.sub_goals.append(sub_goal)

        def update_progress(self, progress):
            """Обновляет прогресс."""
            self.progress = min(100, max(0, progress))
            if self.progress >= 100:
                self.status = "completed"
            elif self.progress > 0:
                self.status = "in_progress"

        def __repr__(self):
            return f"Goal({self.name}: {self.progress}%)"

    # Создаём цели
    goals = [
        Goal("main_task", "Выполнить основную задачу", priority=3),
        Goal("data_collection", "Собрать данные", priority=2),
        Goal("analysis", "Проанализировать данные", priority=2),
        Goal("report", "Создать отчёт", priority=1)
    ]

    # Добавляем подцели
    goals[0].add_sub_goal(goals[1])
    goals[0].add_sub_goal(goals[2])
    goals[0].add_sub_goal(goals[3])

    # Демонстрация
    print("Постановка целей:")
    for goal in goals[:4]:
        print(f"  Цель: {goal.name}")
        print(f"    Описание: {goal.description}")
        print(f"    Приоритет: {goal.priority}")
        print(f"    Статус: {goal.status}")
        print()

    # --- 1.2 Progress Tracking ---
    print("\n--- 1.2 Отслеживание прогресса ---")

    # Система отслеживания
    class ProgressTracker:
        """Трекер прогресса выполнения целей."""
        def __init__(self):
            self.history = []
            self.metrics = {
                "total_goals": 0,
                "completed": 0,
                "failed": 0,
                "average_progress": 0
            }

        def update(self, goal, progress):
            """Обновляет прогресс цели."""
            goal.update_progress(progress)

            # Записываем в историю
            self.history.append({
                "goal": goal.name,
                "progress": progress,
                "timestamp": time.time()
            })

            # Обновляем метрики
            self._update_metrics()

        def _update_metrics(self):
            """Обновляет общие метрики."""
            if not self.history:
                return

            # Считаем завершённые цели
            goal_progress = {}
            for entry in self.history:
                goal_progress[entry["goal"]] = entry["progress"]

            self.metrics["total_goals"] = len(goal_progress)
            self.metrics["completed"] = sum(1 for p in goal_progress.values() if p >= 100)
            self.metrics["failed"] = sum(1 for p in goal_progress.values() if p < 0)
            self.metrics["average_progress"] = (
                sum(goal_progress.values()) / len(goal_progress)
                if goal_progress else 0
            )

        def get_status(self):
            """Возвращает текущий статус."""
            return self.metrics.copy()

        def get_eta(self, goal, rate):
            """Оценивает время до завершения."""
            remaining = 100 - goal.progress
            if rate <= 0:
                return float('inf')
            return remaining / rate  # в условных единицах

    # Демонстрация отслеживания
    tracker = ProgressTracker()
    main_goal = goals[0]

    # Симулируем прогресс
    print("Отслеживание прогресса:")
    progress_steps = [10, 25, 50, 75, 100]
    for step in progress_steps:
        tracker.update(main_goal, step)
        status = tracker.get_status()
        print(f"  Прогресс: {step}% | Средний: {status['average_progress']:.1f}%")

    # --- 1.3 Termination Criteria ---
    print("\n--- 1.3 Критерии завершения ---")

    # Различные критерии завершения
    class TerminationCriteria:
        """Критерии завершения работы агента."""
        def __init__(self):
            self.criteria = []

        def add_criterion(self, name, check_fn, description=""):
            """Добавляет критерий."""
            self.criteria.append({
                "name": name,
                "check": check_fn,
                "description": description
            })

        def should_stop(self, context):
            """Проверяет, нужно ли остановиться."""
            for criterion in self.criteria:
                if criterion["check"](context):
                    return True, criterion["name"]
            return False, None

    # Создаём критерии
    criteria = TerminationCriteria()

    # Критерий: достижение цели
    criteria.add_criterion(
        "goal_reached",
        lambda ctx: ctx.get("progress", 0) >= 100,
        "Цель достигнута (100%)"
    )

    # Критерий: превышение времени
    criteria.add_criterion(
        "timeout",
        lambda ctx: ctx.get("elapsed_time", 0) > ctx.get("max_time", 60),
        "Превышено максимальное время"
    )

    # Критерий: количество попыток
    criteria.add_criterion(
        "max_retries",
        lambda ctx: ctx.get("attempts", 0) >= ctx.get("max_attempts", 3),
        "Превышено количество попыток"
    )

    # Критерий: достижение минимального качества
    criteria.add_criterion(
        "min_quality",
        lambda ctx: ctx.get("quality_score", 0) >= ctx.get("min_quality", 0.8),
        "Достигнуто минимальное качество"
    )

    # Тестируем критерии
    print("Проверка критериев завершения:")

    contexts = [
        {"progress": 100, "elapsed_time": 30, "attempts": 1, "quality_score": 0.9},
        {"progress": 50, "elapsed_time": 70, "attempts": 1, "quality_score": 0.5},
        {"progress": 30, "elapsed_time": 30, "attempts": 4, "quality_score": 0.6}
    ]

    for i, ctx in enumerate(contexts):
        should_stop, reason = criteria.should_stop(ctx)
        print(f"\n  Контекст {i+1}: {ctx}")
        print(f"    Остановиться: {should_stop}")
        if should_stop:
            print(f"    Причина: {reason}")

    # --- 1.4 Goal Decomposition ---
    print("\n--- 1.4 Декомпозиция целей ---")

    # Разбиение сложной цели на подцели
    class GoalDecomposer:
        """Декомпозитор целей."""
        def __init__(self):
            self.decomposition_rules = {}

        def add_rule(self, goal_type, sub_goals):
            """Добавляет правило декомпозиции."""
            self.decomposition_rules[goal_type] = sub_goals

        def decompose(self, goal):
            """Декомпозирует цель."""
            goal_type = goal.get("type", "unknown")
            if goal_type in self.decomposition_rules:
                return self.decomposition_rules[goal_type]
            return [goal]

    # Создаём декомпозитор
    decomposer = GoalDecomposer()

    # Добавляем правила
    decomposer.add_rule("project", [
        {"type": "task", "name": "Планирование", "description": "Составить план"},
        {"type": "task", "name": "Реализация", "description": "Реализовать план"},
        {"type": "task", "name": "Тестирование", "description": "Протестировать результат"},
        {"type": "task", "name": "Деплой", "description": "Развернуть решение"}
    ])

    decomposer.add_rule("research", [
        {"type": "task", "name": "Сбор данных", "description": "Собрать информацию"},
        {"type": "task", "name": "Анализ", "description": "Проанализировать данные"},
        {"type": "task", "name": "Выводы", "description": "Сделать выводы"}
    ])

    # Декомпозируем цели
    print("Декомпозиция целей:")

    test_goals = [
        {"type": "project", "name": "Создание AI-ассистента"},
        {"type": "research", "name": "Исследование рынка"}
    ]

    for goal in test_goals:
        print(f"\n  Цель: {goal['name']}")
        sub_goals = decomposer.decompose(goal)
        print("  Подцели:")
        for i, sg in enumerate(sub_goals, 1):
            print(f"    {i}. {sg['name']}: {sg['description']}")


# =============================================================================
# 2. Self-Reflection
# =============================================================================

def demo_self_reflection():
    """Демонстрация саморефлексии агента."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: Self-Reflection")
    print("=" * 70)

    # --- 2.1 Output Critique ---
    print("\n--- 2.1 Критика вывода ---")

    # Система критики вывода
    class OutputCritic:
        """Критик, оценивающий качество вывода."""
        def __init__(self):
            self.criteria = []

        def add_criterion(self, name, weight, check_fn):
            """Добавляет критерий оценки."""
            self.criteria.append({
                "name": name,
                "weight": weight,
                "check": check_fn
            })

        def critique(self, output):
            """Оценивает вывод по критериям."""
            scores = {}
            total_weight = sum(c["weight"] for c in self.criteria)

            for criterion in self.criteria:
                score = criterion["check"](output)
                scores[criterion["name"]] = {
                    "score": score,
                    "weight": criterion["weight"],
                    "weighted": score * criterion["weight"]
                }

            # Общая оценка
            total_score = sum(s["weighted"] for s in scores.values()) / total_weight

            return {
                "scores": scores,
                "total_score": total_score,
                "grade": self._get_grade(total_score)
            }

        def _get_grade(self, score):
            """Определяет оценку по баллу."""
            if score >= 0.9:
                return "Отлично"
            elif score >= 0.7:
                return "Хорошо"
            elif score >= 0.5:
                return "Удовлетворительно"
            else:
                return "Неудовлетворительно"

    # Создаём критика
    critic = OutputCritic()

    # Добавляем критерии
    critic.add_criterion("точность", 0.3, lambda x: 0.9 if "Python" in x else 0.5)
    critic.add_criterion("полнота", 0.3, lambda x: min(1.0, len(x.split()) / 20))
    critic.add_criterion("чёткость", 0.2, lambda x: 0.8 if len(x) < 500 else 0.6)
    critic.add_criterion("примеры", 0.2, lambda x: 0.9 if "пример" in x.lower() or "```" in x else 0.4)

    # Тестируем
    outputs = [
        "Python — язык программирования. Пример: print('Hello')",
        "Python — высокоуровневый язык программирования общего назначения с динамической типизацией и автоматическим управлением памятью. Он广泛 используется в веб-разработке, науке о данных, искусственном интеллекте и автоматизации. Пример кода: def hello(): print('Hello, World!')",
        "Краткий ответ"
    ]

    print("Критика выводов:")
    for i, output in enumerate(outputs):
        print(f"\n  Вывод {i+1}: {output[:50]}...")
        result = critic.critique(output)
        print(f"    Общая оценка: {result['total_score']:.2f} ({result['grade']})")
        for name, score in result['scores'].items():
            print(f"      {name}: {score['score']:.2f} (вес: {score['weight']})")

    # --- 2.2 Quality Assessment ---
    print("\n--- 2.2 Оценка качества ---")

    # Метрики качества
    class QualityAssessor:
        """Оценщик качества работы агента."""
        def __init__(self):
            self.metrics_history = []

        def assess(self, output, expected=None):
            """Оценивает качество вывода."""
            metrics = {
                "length": len(output),
                "word_count": len(output.split()),
                "unique_words": len(set(output.lower().split())),
                "has_code": "```" in output or "def " in output or "class " in output,
                "has_examples": "пример" in output.lower() or "example" in output.lower()
            }

            # Если есть эталон, считаем точность
            if expected:
                output_words = set(output.lower().split())
                expected_words = set(expected.lower().split())
                intersection = output_words & expected_words
                union = output_words | expected_words
                metrics["jaccard_similarity"] = len(intersection) / len(union) if union else 0

            self.metrics_history.append(metrics)
            return metrics

        def get_average_metrics(self):
            """Средние метрики по всем оценкам."""
            if not self.metrics_history:
                return {}

            avg = {}
            keys = self.metrics_history[0].keys()
            for key in keys:
                values = [m[key] for m in self.metrics_history if isinstance(m[key], (int, float))]
                if values:
                    avg[key] = sum(values) / len(values)
            return avg

    # Демонстрация
    assessor = QualityAssessor()

    test_outputs = [
        ("Python — язык программирования.", "Python — язык программирования общего назначения."),
        ("```python\nprint('Hello')\n```", "Пример кода на Python."),
        ("Короткий текст", "Это должен был быть длинный текст с примерами.")
    ]

    print("Оценка качества:")
    for output, expected in test_outputs:
        metrics = assessor.assess(output, expected)
        print(f"\n  Вывод: {output[:40]}...")
        print(f"    Длина: {metrics['length']} символов")
        print(f"    Слов: {metrics['word_count']}")
        print(f"    Уникальных слов: {metrics['unique_words']}")
        if 'jaccard_similarity' in metrics:
            print(f"    Сходство с эталоном: {metrics['jaccard_similarity']:.2f}")

    # Средние метрики
    avg = assessor.get_average_metrics()
    print(f"\nСредние метрики:")
    for key, value in avg.items():
        print(f"  {key}: {value:.2f}")

    # --- 2.3 Improvement Suggestions ---
    print("\n--- 2.3 Предложения по улучшению ---")

    # Генератор предложений
    class ImprovementGenerator:
        """Генерирует предложения по улучшению."""
        def __init__(self):
            self.suggestion_rules = []

        def add_rule(self, condition, suggestion):
            """Добавляет правило генерации предложений."""
            self.suggestion_rules.append((condition, suggestion))

        def generate(self, metrics, critique_result):
            """Генерирует предложения на основе анализа."""
            suggestions = []

            # Проверяем правила
            for condition, suggestion in self.suggestion_rules:
                if condition(metrics, critique_result):
                    suggestions.append(suggestion)

            # Добавляем общие предложения на основе оценки
            total_score = critique_result.get("total_score", 0)
            if total_score < 0.7:
                suggestions.append("Рассмотрите возможность добавления примеров")
            if total_score < 0.5:
                suggestions.append("Увеличьте детализацию ответа")

            return suggestions

    # Создаём генератор
    generator = ImprovementGenerator()

    # Добавляем правила
    generator.add_rule(
        lambda m, c: m.get("word_count", 0) < 10,
        "Ответ слишком короткий. Добавьте больше деталей."
    )
    generator.add_rule(
        lambda m, c: m.get("unique_words", 0) < m.get("word_count", 0) * 0.5,
        "Слишком много повторений. Используйте синонимы."
    )
    generator.add_rule(
        lambda m, c: not m.get("has_code", False),
        "Добавьте пример кода для иллюстрации."
    )
    generator.add_rule(
        lambda m, c: c.get("scores", {}).get("полнота", {}).get("score", 1) < 0.6,
        "Ответ неполный. Раскройте тему более подробно."
    )

    # Генерируем предложения
    print("Предложения по улучшению:")

    test_cases = [
        ({"word_count": 5, "unique_words": 5, "has_code": False},
         {"total_score": 0.4, "scores": {"полнота": {"score": 0.3}}}),
        ({"word_count": 50, "unique_words": 20, "has_code": True},
         {"total_score": 0.8, "scores": {"полнота": {"score": 0.8}}})
    ]

    for metrics, critique in test_cases:
        suggestions = generator.generate(metrics, critique)
        print(f"\n  Метрики: {metrics}")
        print(f"  Оценка: {critique['total_score']}")
        if suggestions:
            print("  Предложения:")
            for s in suggestions:
                print(f"    - {s}")
        else:
            print("  Предложений нет")

    # --- 2.4 Reflection Loop ---
    print("\n--- 2.4 Цикл рефлексии ---")

    # Цикл саморефлексии
    class ReflectionLoop:
        """Цикл саморефлексии агента."""
        def __init__(self, max_iterations=5):
            self.max_iterations = max_iterations
            self.history = []

        def reflect(self, output, assessor, critic):
            """Выполняет цикл рефлексии."""
            current_output = output
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1

                # Оцениваем качество
                metrics = assessor.assess(current_output)
                critique = critic.critique(current_output)

                # Записываем в историю
                self.history.append({
                    "iteration": iteration,
                    "output": current_output[:100],
                    "score": critique["total_score"]
                })

                print(f"\n  Итерация {iteration}:")
                print(f"    Оценка: {critique['total_score']:.2f}")

                # Если оценка достаточна, останавливаемся
                if critique["total_score"] >= 0.85:
                    print(f"    Достаточное качество, остановка")
                    break

                # Эмулируем улучшение
                current_output = self._improve(current_output, critique)

            return current_output

        def _improve(self, output, critique):
            """Улучшает вывод на основе критики."""
            # Простая эмуляция улучшения
            if critique["total_score"] < 0.5:
                return output + ". Добавлены дополнительные детали и примеры."
            elif critique["total_score"] < 0.7:
                return output + " [улучшено]"
            return output

    # Демонстрация цикла
    print("Цикл саморефлексии:")
    loop = ReflectionLoop(max_iterations=3)

    assessor = QualityAssessor()
    critic = OutputCritic()
    critic.add_criterion("длина", 0.5, lambda x: min(1.0, len(x) / 100))
    critic.add_criterion("содержание", 0.5, lambda x: 0.6)

    initial_output = "Короткий ответ"
    final_output = loop.reflect(initial_output, assessor, critic)

    print(f"\n  Начальный вывод: {initial_output}")
    print(f"  Итоговый вывод: {final_output}")
    print(f"  Итераций: {len(loop.history)}")


# =============================================================================
# 3. Error Recovery
# =============================================================================

def demo_error_recovery():
    """Демонстрация восстановления ошибок."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: Error Recovery")
    print("=" * 70)

    # --- 3.1 Retry Strategies ---
    print("\n--- 3.1 Стратегии повторных попыток ---")

    # Различные стратегии повторных попыток
    class RetryStrategy:
        """Стратегия повторных попыток."""
        def __init__(self, name, max_retries, delay_fn):
            self.name = name
            self.max_retries = max_retries
            self.delay_fn = delay_fn

        def get_delay(self, attempt):
            """Вычисляет задержку для попытки."""
            return self.delay_fn(attempt)

    # Определяем стратегии
    strategies = {
        "exponential": RetryStrategy(
            "Экспоненциальная",
            max_retries=5,
            delay_fn=lambda attempt: min(30, 2 ** attempt)  # 1, 2, 4, 8, 16, 30
        ),
        "linear": RetryStrategy(
            "Линейная",
            max_retries=5,
            delay_fn=lambda attempt: attempt * 2  # 2, 4, 6, 8, 10
        ),
        "fixed": RetryStrategy(
            "Фиксированная",
            max_retries=3,
            delay_fn=lambda attempt: 5  # всегда 5
        ),
        "fibonacci": RetryStrategy(
            "Фибоначчи",
            max_retries=6,
            delay_fn=lambda attempt: [1, 1, 2, 3, 5, 8][min(attempt, 5)]
        )
    }

    # Демонстрация стратегий
    print("Стратегии повторных попыток:")
    for name, strategy in strategies.items():
        print(f"\n  {strategy.name}:")
        delays = [strategy.get_delay(i) for i in range(strategy.max_retries)]
        print(f"    Макс. попыток: {strategy.max_retries}")
        print(f"    Задержки: {delays}")

    # --- 3.2 Fallback Plans ---
    print("\n--- 3.2 Планы отката ---")

    # Система планов отката
    class FallbackPlan:
        """План отката при ошибке."""
        def __init__(self):
            self.plans = []

        def add_plan(self, error_type, fallback_fn, description=""):
            """Добавляет план отката."""
            self.plans.append({
                "error_type": error_type,
                "fallback": fallback_fn,
                "description": description
            })

        def execute(self, error_type, context):
            """Выполняет план отката."""
            for plan in self.plans:
                if plan["error_type"] == error_type:
                    print(f"    Выполняю план: {plan['description']}")
                    return plan["fallback"](context)
            return None

    # Создаём планы отката
    fallback = FallbackPlan()

    # План 1: Использование кэша
    fallback.add_plan(
        "network_error",
        lambda ctx: f"Данные из кэша: {ctx.get('cached_data', 'нет данных')}",
        "Использовать кэшированные данные"
    )

    # План 2: Упрощённый ответ
    fallback.add_plan(
        "api_error",
        lambda ctx: "Извините, сервис временно недоступен. Попробуйте позже.",
        "Вернуть упрощённый ответ"
    )

    # План 3: Альтернативный источник
    fallback.add_plan(
        "data_error",
        lambda ctx: f"Данные из альтернативного источника: альтернативные данные",
        "Использовать альтернативный источник"
    )

    # Тестируем
    print("Планы отката:")
    test_errors = ["network_error", "api_error", "data_error", "unknown_error"]
    context = {"cached_data": "Тестовые данные из кэша"}

    for error in test_errors:
        print(f"\n  Ошибка: {error}")
        result = fallback.execute(error, context)
        if result:
            print(f"    Результат: {result}")
        else:
            print(f"    План не найден, используется стандартная обработка")

    # --- 3.3 Graceful Degradation ---
    print("\n--- 3.3 Плавное снижение ---")

    # Система плавного снижения
    class GracefulDegradation:
        """Плавное снижение при ошибках."""
        def __init__(self):
            self.levels = []
            self.current_level = 0

        def add_level(self, name, functionality, quality):
            """Добавляет уровень функциональности."""
            self.levels.append({
                "name": name,
                "functionality": functionality,
                "quality": quality
            })

        def degrade(self):
            """Снижает уровень функциональности."""
            if self.current_level < len(self.levels) - 1:
                self.current_level += 1
                return self.get_current()
            return None

        def get_current(self):
            """Возвращает текущий уровень."""
            if self.current_level < len(self.levels):
                return self.levels[self.current_level]
            return None

        def get_full_functionality(self):
            """Возвращает полную функциональность."""
            return self.levels[0] if self.levels else None

    # Создаём систему снижения
    degradation = GracefulDegradation()

    # Добавляем уровни
    degradation.add_level("Полная функциональность", ["API", "Кэш", "ML"], "высокое")
    degradation.add_level("Базовая функциональность", ["API", "Кэш"], "среднее")
    degradation.add_level("Минимальная функциональность", ["Кэш"], "низкое")
    degradation.add_level("Автономный режим", [], "минимальное")

    # Демонстрация снижения
    print("Плавное снижение функциональности:")
    current = degradation.get_current()
    print(f"\n  Начальный уровень: {current['name']}")
    print(f"    Функциональность: {current['functionality']}")
    print(f"    Качество: {current['quality']}")

    for i in range(3):
        degraded = degradation.degrade()
        if degraded:
            print(f"\n  После снижения {i+1}:")
            print(f"    Уровень: {degraded['name']}")
            print(f"    Функциональность: {degraded['functionality']}")
            print(f"    Качество: {degraded['quality']}")

    # --- 3.4 Error Logging ---
    print("\n--- 3.4 Логирование ошибок ---")

    # Система логирования
    class ErrorLogger:
        """Логгер ошибок."""
        def __init__(self):
            self.errors = []
            self.statistics = collections.Counter()

        def log(self, error_type, message, context=None):
            """Логирует ошибку."""
            error_entry = {
                "type": error_type,
                "message": message,
                "context": context or {},
                "timestamp": time.time()
            }
            self.errors.append(error_entry)
            self.statistics[error_type] += 1

        def get_statistics(self):
            """Возвращает статистику ошибок."""
            return dict(self.statistics)

        def get_recent_errors(self, n=5):
            """Возвращает последние n ошибок."""
            return self.errors[-n:]

        def analyze_patterns(self):
            """Анализирует паттерны ошибок."""
            if not self.errors:
                return "Нет ошибок для анализа"

            # Анализируем частоту
            type_counts = self.statistics.most_common()
            total = sum(self.statistics.values())

            analysis = "Паттерны ошибок:\n"
            for error_type, count in type_counts:
                percentage = count / total * 100
                analysis += f"  {error_type}: {count} ({percentage:.1f}%)\n"

            return analysis

    # Демонстрация логирования
    logger = ErrorLogger()

    # Логируем ошибки
    errors = [
        ("network", "Таймаут соединения", {"url": "api.example.com"}),
        ("network", "Соединение разорвано", {"url": "api.example.com"}),
        ("api", "Неверный ответ API", {"status": 500}),
        ("data", "Невалидные данные", {"field": "user_id"}),
        ("network", "Таймаут соединения", {"url": "api.example.com"})
    ]

    print("Логирование ошибок:")
    for error_type, message, context in errors:
        logger.log(error_type, message, context)
        print(f"  [{error_type}] {message}")

    # Статистика
    print(f"\nСтатистика ошибок:")
    stats = logger.get_statistics()
    for error_type, count in stats.items():
        print(f"  {error_type}: {count}")

    # Анализ паттернов
    print(f"\n{logger.analyze_patterns()}")


# =============================================================================
# 4. Autonomous Workflows
# =============================================================================

def demo_autonomous_workflows():
    """Демонстрация автономных рабочих процессов."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Autonomous Workflows")
    print("=" * 70)

    # --- 4.1 Long-Running Tasks ---
    print("\n--- 4.1 Длительные задачи ---")

    # Менеджер длительных задач
    class LongRunningTask:
        """Длительная задача."""
        def __init__(self, name, estimated_duration):
            self.name = name
            self.estimated_duration = estimated_duration
            self.elapsed = 0
            self.status = "pending"
            self.checkpoints = []

        def start(self):
            """Начинает задачу."""
            self.status = "running"
            print(f"  Задача '{self.name}' начата")

        def update(self, delta):
            """Обновляет прогресс."""
            if self.status != "running":
                return

            self.elapsed += delta
            progress = min(100, (self.elapsed / self.estimated_duration) * 100)

            # Создаём чекпоинт каждые 25%
            if len(self.checkpoints) < progress // 25:
                self.checkpoints.append({
                    "progress": progress,
                    "timestamp": time.time()
                })
                print(f"    Чекпоинт: {progress:.0f}%")

            # Проверяем завершение
            if self.elapsed >= self.estimated_duration:
                self.status = "completed"
                print(f"  Задача '{self.name}' завершена")

        def get_progress(self):
            """Возвращает текущий прогресс."""
            if self.estimated_duration == 0:
                return 100
            return min(100, (self.elapsed / self.estimated_duration) * 100)

    # Демонстрация
    print("Длительные задачи:")

    tasks = [
        LongRunningTask("Обработка данных", 100),
        LongRunningTask("Обучение модели", 200),
        LongRunningTask("Генерация отчёта", 50)
    ]

    for task in tasks:
        task.start()

    # Симулируем выполнение
    print("\nСимуляция выполнения:")
    for step in range(10):
        for task in tasks:
            if task.status == "running":
                task.update(20)

    # Итоговое состояние
    print("\nИтоговое состояние:")
    for task in tasks:
        print(f"  {task.name}: {task.get_progress():.0f}% ({task.status})")

    # --- 4.2 Checkpointing ---
    print("\n--- 4.2 Чекпоинты ---")

    # Система чекпоинтов
    class CheckpointManager:
        """Менеджер чекпоинтов."""
        def __init__(self):
            self.checkpoints = {}
            self.current_checkpoint = None

        def save(self, task_id, state):
            """Сохраняет чекпоинт."""
            self.checkpoints[task_id] = {
                "state": state,
                "timestamp": time.time()
            }
            self.current_checkpoint = task_id
            print(f"    Чекпоинт сохранён: {task_id}")

        def load(self, task_id):
            """Загружает чекпоинт."""
            if task_id in self.checkpoints:
                return self.checkpoints[task_id]["state"]
            return None

        def list_checkpoints(self):
            """Список чекпоинтов."""
            return list(self.checkpoints.keys())

    # Демонстрация
    print("Система чекпоинтов:")
    checkpoint_mgr = CheckpointManager()

    # Сохраняем чекпоинты
    checkpoint_mgr.save("task_1", {"step": 1, "data": "начальные данные"})
    checkpoint_mgr.save("task_1", {"step": 2, "data": "обработанные данные"})
    checkpoint_mgr.save("task_2", {"step": 1, "data": "другие данные"})

    print(f"\n  Доступные чекпоинты: {checkpoint_mgr.list_checkpoints()}")

    # Загружаем чекпоинт
    state = checkpoint_mgr.load("task_1")
    if state:
        print(f"  Загружен чекпоинт task_1: {state}")

    # --- 4.3 Human-in-the-Loop ---
    print("\n--- 4.3 Человек в цикле ---")

    # Система взаимодействия с человеком
    class HumanInTheLoop:
        """Система взаимодействия с человеком."""
        def __init__(self):
            self.pending_approvals = []
            self.approved = []
            self.rejected = []

        def request_approval(self, task_id, description, options=None):
            """Запрашивает одобрение."""
            request = {
                "task_id": task_id,
                "description": description,
                "options": options or ["Одобрить", "Отклонить"],
                "timestamp": time.time()
            }
            self.pending_approvals.append(request)
            print(f"    Запрос одобрения: {description}")
            return request

        def respond(self, task_id, response):
            """Отвечает на запрос."""
            for i, request in enumerate(self.pending_approvals):
                if request["task_id"] == task_id:
                    self.pending_approvals.pop(i)
                    if response == "Одобрить":
                        self.approved.append(request)
                        print(f"    Одобрено: {request['description']}")
                    else:
                        self.rejected.append(request)
                        print(f"    Отклонено: {request['description']}")
                    return True
            return False

        def get_status(self):
            """Возвращает статус запросов."""
            return {
                "pending": len(self.pending_approvals),
                "approved": len(self.approved),
                "rejected": len(self.rejected)
            }

    # Демонстрация
    print("Взаимодействие с человеком:")
    hitl = HumanInTheLoop()

    # Запрашиваем одобрения
    hitl.request_approval("T1", "Удалить временные файлы?")
    hitl.request_approval("T2", "Отправить отчёт заказчику?")
    hitl.request_approval("T3", "Запустить автоматические тесты?")

    # Отвечаем на запросы
    hitl.respond("T1", "Одобрить")
    hitl.respond("T2", "Одобрить")
    hitl.respond("T3", "Отклонить")

    # Статус
    status = hitl.get_status()
    print(f"\n  Статус: {status}")

    # --- 4.4 Workflow Orchestration ---
    print("\n--- 4.4 Оркестрация рабочих процессов ---")

    # Оркестратор рабочих процессов
    class WorkflowOrchestrator:
        """Оркестратор рабочих процессов."""
        def __init__(self):
            self.workflows = {}
            self.running = []

        def define_workflow(self, name, steps):
            """Определяет рабочий процесс."""
            self.workflows[name] = {
                "steps": steps,
                "current_step": 0,
                "status": "defined"
            }

        def start_workflow(self, name):
            """Запускает рабочий процесс."""
            if name in self.workflows:
                self.workflows[name]["status"] = "running"
                self.running.append(name)
                print(f"  Запуск: {name}")
                return True
            return False

        def execute_step(self, name):
            """Выполняет текущий шаг."""
            if name not in self.workflows:
                return None

            workflow = self.workflows[name]
            if workflow["current_step"] >= len(workflow["steps"]):
                workflow["status"] = "completed"
                return None

            step = workflow["steps"][workflow["current_step"]]
            workflow["current_step"] += 1

            print(f"    Шаг {workflow['current_step']}: {step['name']}")

            # Проверяем, завершён ли процесс
            if workflow["current_step"] >= len(workflow["steps"]):
                workflow["status"] = "completed"
                print(f"  Процесс {name} завершён")

            return step

        def get_status(self):
            """Возвращает статус всех процессов."""
            return {
                name: {
                    "status": wf["status"],
                    "progress": f"{wf['current_step']}/{len(wf['steps'])}"
                }
                for name, wf in self.workflows.items()
            }

    # Создаём оркестратор
    orchestrator = WorkflowOrchestrator()

    # Определяем процессы
    orchestrator.define_workflow("ETL", [
        {"name": "Извлечение данных", "type": "extract"},
        {"name": "Трансформация", "type": "transform"},
        {"name": "Загрузка", "type": "load"}
    ])

    orchestrator.define_workflow("ML_Pipeline", [
        {"name": "Подготовка данных", "type": "preprocess"},
        {"name": "Обучение модели", "type": "train"},
        {"name": "Оценка качества", "type": "evaluate"},
        {"name": "Деплой", "type": "deploy"}
    ])

    # Запускаем процессы
    print("Оркестрация рабочих процессов:")
    orchestrator.start_workflow("ETL")
    orchestrator.start_workflow("ML_Pipeline")

    # Выполняем шаги
    print("\nВыполнение шагов:")
    for _ in range(5):
        for name in orchestrator.running:
            orchestrator.execute_step(name)

    # Статус
    print("\nИтоговый статус:")
    status = orchestrator.get_status()
    for name, info in status.items():
        print(f"  {name}: {info['status']} ({info['progress']})")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Модуль 164: Autonomous Agents")
    print("Тема: автономность, саморефлексия, восстановление ошибок\n")

    demo_goal_driven()
    demo_self_reflection()
    demo_error_recovery()
    demo_autonomous_workflows()

    print("\n" + "=" * 70)
    print("Все демонстрации модуля 164 завершены.")
    print("=" * 70)
