"""162 — Prompt Engineering for Agents: системные промпты, few-shot, динамический промптинг

Темы:
  1. System Prompt Design (role definition, constraints, output format, tool instructions)
  2. Few-Shot for Agents (example selection, demonstration format, chain examples)
  3. Dynamic Prompting (context injection, history management, template variables)
  4. Prompt Optimization (A/B testing prompts, failure-driven improvement)

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
# 1. System Prompt Design
# =============================================================================

def demo_system_prompt():
    """Демонстрация проектирования системных промптов для агентов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: System Prompt Design")
    print("=" * 70)

    # --- 1.1 Role Definition ---
    print("\n--- 1.1 Определение роли агента ---")

    # Системный промпт определяет поведение агента
    system_prompt = {
        "role": "Ты — финансовый аналитик-агент",
        "expertise": ["анализ данных", "прогнозирование", "рекомендации"],
        "constraints": [
            "Не давай инвестиционных советов",
            "Всегда указывай источники данных",
            "Оцени достоверность своих выводов"
        ],
        "tone": "Профессиональный, но понятный"
    }

    # Формируем итоговый системный промпт из компонентов
    prompt_parts = []
    prompt_parts.append(f"РОЛЬ: {system_prompt['role']}")
    prompt_parts.append(f"ЭКСПЕРТИЗА: {', '.join(system_prompt['expertise'])}")
    prompt_parts.append("ОГРАНИЧЕНИЯ:")
    for constraint in system_prompt['constraints']:
        prompt_parts.append(f"  - {constraint}")
    prompt_parts.append(f"ТОН ОБЩЕНИЯ: {system_prompt['tone']}")

    full_system_prompt = "\n".join(prompt_parts)
    print(full_system_prompt)

    # Подсчитываем количество токенов (приближённо)
    token_count = len(full_system_prompt.split())
    print(f"\nПриблизительное количество слов (токенов): {token_count}")

    # --- 1.2 Constraints ---
    print("\n--- 1.2 Ограничения агента ---")

    # Различные типы ограничений для агента
    constraints = {
        "scope": "Отвечай только на вопросы о финансах",
        "safety": "Не выполняй опасные операции без подтверждения",
        "format": "Всегда отвечай в формате JSON",
        "latency": "Давай ответы короче 200 слов"
    }

    # Формируем блок ограничений
    constraint_block = "ОГРАНИЧЕНИЯ АГЕНТА:\n"
    for name, value in constraints.items():
        constraint_block += f"  [{name.upper()}] {value}\n"

    print(constraint_block)

    # Проверяем соответствие формата
    sample_response = "{'recommendation': 'buy', 'confidence': 0.85}"
    is_json_format = sample_response.startswith('{')
    print(f"Формат JSON соблюдён: {is_json_format}")

    # --- 1.3 Output Format ---
    print("\n--- 1.3 Формат вывода ---")

    # Схема ожидаемого формата ответа
    output_schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "description": "Текст анализа"},
            "confidence": {"type": "number", "description": "Уверенность 0-1"},
            "recommendation": {"type": "string", "enum": ["buy", "hold", "sell"]},
            "risks": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["analysis", "confidence", "recommendation"]
    }

    print("Схема вывода:")
    print(json.dumps(output_schema, indent=2, ensure_ascii=False))

    # Пример ответа в нужном формате
    example_response = {
        "analysis": "Рост выручки на 15% за квартал",
        "confidence": 0.78,
        "recommendation": "hold",
        "risks": ["Волатильность рынка", "Геополитические риски"]
    }
    print("\nПример ответа агента:")
    print(json.dumps(example_response, indent=2, ensure_ascii=False))

    # --- 1.4 Tool Instructions ---
    print("\n--- 1.4 Инструкции по использованию инструментов ---")

    # Описания инструментов для агента
    tools = [
        {
            "name": "search_web",
            "description": "Поиск информации в интернете",
            "parameters": {"query": "str"},
            "usage": "Используй для актуальных данных"
        },
        {
            "name": "calculate",
            "description": "Математические вычисления",
            "parameters": {"expression": "str"},
            "usage": "Используй для точных расчётов"
        },
        {
            "name": "get_stock_price",
            "description": "Получение текущей цены акции",
            "parameters": {"ticker": "str"},
            "usage": "Используй для рыночных данных"
        }
    ]

    # Генерируем инструкции по инструментам
    tool_instructions = "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n"
    for tool in tools:
        tool_instructions += f"\n  {tool['name']}()\n"
        tool_instructions += f"    Описание: {tool['description']}\n"
        tool_instructions += f"    Параметры: {tool['parameters']}\n"
        tool_instructions += f"    Когда использовать: {tool['usage']}\n"

    print(tool_instructions)

    # Вычисляем общую стоимость промпта
    total_chars = len(full_system_prompt) + len(constraint_block) + len(tool_instructions)
    print(f"Общий размер системного промпта: {total_chars} символов")
    print(f"Приблизительная стоимость (запрос): {total_chars / 4} токенов")


# =============================================================================
# 2. Few-Shot for Agents
# =============================================================================

def demo_few_shot():
    """Демонстрация few-shot обучения для агентов."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: Few-Shot для агентов")
    print("=" * 70)

    # --- 2.1 Example Selection ---
    print("\n--- 2.1 Выбор примеров ---")

    # Пул примеров для few-shot обучения
    examples_pool = [
        {"input": "Какая погода в Москве?", "category": "weather", "quality": 0.9},
        {"input": "Сколько стоит акция Apple?", "category": "finance", "quality": 0.85},
        {"input": "Переведи текст на английский", "category": "translation", "quality": 0.8},
        {"input": "Напиши код на Python", "category": "coding", "quality": 0.95},
        {"input": "Суммаризируй статью", "category": "summarization", "quality": 0.75},
        {"input": "Какой маршрут до аэропорта?", "category": "navigation", "quality": 0.7},
    ]

    # Алгоритм выбора наиболее релевантных примеров
    def select_examples(query, pool, n=3):
        """Выбирает n наиболее подходящих примеров для query."""
        # Эмулируем поиск похожих примеров
        query_words = set(query.lower().split())
        scored = []
        for ex in pool:
            ex_words = set(ex["input"].lower().split())
            # Пересечение слов
            overlap = len(query_words & ex_words)
            # Учитываем качество примера
            score = overlap * 0.3 + ex["quality"] * 0.7
            scored.append((score, ex))
        # Сортируем по убыванию score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]

    test_query = "Какая погода сегодня?"
    selected = select_examples(test_query, examples_pool)
    print(f"Запрос: {test_query}")
    print("Выбранные примеры:")
    for i, ex in enumerate(selected):
        print(f"  {i+1}. [{ex['category']}] {ex['input']} (качество: {ex['quality']})")

    # --- 2.2 Demonstration Format ---
    print("\n--- 2.2 Формат демонстрации ---")

    # Формируем few-shot промпт из выбранных примеров
    few_shot_template = """Ты — умный агент. Отвечай на вопросы пользователя.

Пример 1:
Вопрос: Какая погода в Москве?
Действие: search_web("погода Москва")
Ответ: В Москве сейчас +15°C, облачно.

Пример 2:
Вопрос: Сколько стоит акция Apple?
Действие: get_stock_price("AAPL")
Ответ: Акция Apple торгуется по $185.50.

Премер 3:
Вопрос: Напиши Hello World на Python
Действие: generate_code("python", "hello world")
Ответ: print("Hello, World!")

Текущий вопрос: {query}
Действие:"""

    # Заполняем шаблон
    filled_prompt = few_shot_template.format(query=test_query)
    print("Сформированный few-shot промпт:")
    print(filled_prompt)

    # --- 2.3 Chain Examples ---
    print("\n--- 2.3 Цепочки примеров ---")

    # Примеры цепочки рассуждений (chain-of-thought)
    chain_examples = [
        {
            "step": "1. Понимание задачи",
            "reasoning": "Пользователь спрашивает о погоде — нужен актуальный прогноз",
            "action": "search_web('прогноз погоды')"
        },
        {
            "step": "2. Обработка данных",
            "reasoning": "Получены данные о погоде — нужно структурировать",
            "action": "format_weather_data(raw_data)"
        },
        {
            "step": "3. Формирование ответа",
            "reasoning": "Данные готовы — создаём понятный ответ",
            "action": "generate_response(weather_info)"
        }
    ]

    print("Цепочка рассуждений агента:")
    for example in chain_examples:
        print(f"\n  {example['step']}")
        print(f"    Рассуждение: {example['reasoning']}")
        print(f"    Действие: {example['action']}")

    # Вычисляем длину цепочки
    total_reasoning = sum(len(e['reasoning']) for e in chain_examples)
    print(f"\nСуммарная длина рассуждений: {total_reasoning} символов")

    # --- 2.4 Dynamic Example Selection ---
    print("\n--- 2.4 Динамический выбор примеров ---")

    # Кластеризация запросов для лучшего выбора
    query_clusters = {
        "weather": ["погода", "температура", "осадки", "прогноз"],
        "finance": ["акции", "цена", "рынок", "инвестиции"],
        "code": ["код", "программа", "функция", "алгоритм"],
        "translation": ["перевод", "текст", "язык", "английский"]
    }

    def find_cluster(query, clusters):
        """Находит кластер для запроса по ключевым словам."""
        query_lower = query.lower()
        scores = {}
        for cluster, keywords in clusters.items():
            # Подсчитываем совпадения с ключевыми словами
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[cluster] = score
        if scores:
            return max(scores, key=scores.get)
        return "general"

    # Тестируем кластеризацию
    test_queries = [
        "Какая погода в Питере?",
        "Сколько стоят акции Газпрома?",
        "Напиши функцию сортировки",
        "Переведи фразу на английский"
    ]

    print("Кластеризация запросов:")
    for query in test_queries:
        cluster = find_cluster(query, query_clusters)
        print(f"  '{query}' -> кластер: {cluster}")

    # Подсчитываем точность кластеризации
    correct = 0
    expected = ["weather", "finance", "code", "translation"]
    for query, exp in zip(test_queries, expected):
        found = find_cluster(query, query_clusters)
        if found == exp:
            correct += 1
    accuracy = correct / len(expected)
    print(f"\nТочность кластеризации: {accuracy:.1%}")


# =============================================================================
# 3. Dynamic Prompting
# =============================================================================

def demo_dynamic_prompting():
    """Демонстрация динамического промптинга."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: Dynamic Prompting")
    print("=" * 70)

    # --- 3.1 Context Injection ---
    print("\n--- 3.1 Инъекция контекста ---")

    # Класс для управления контекстом агента
    class ContextManager:
        def __init__(self):
            self.global_context = {}  # Глобальный контекст
            self.local_context = {}   # Локальный контекст
            self.history = []         # История взаимодействий

        def inject_context(self, key, value, scope="local"):
            """Инжектит контекст в промпт."""
            if scope == "global":
                self.global_context[key] = value
            else:
                self.local_context[key] = value

        def build_prompt(self, query):
            """Строит промпт с учётом контекста."""
            parts = []

            # Глобальный контекст
            if self.global_context:
                parts.append("ГЛОБАЛЬНЫЙ КОНТЕКСТ:")
                for k, v in self.global_context.items():
                    parts.append(f"  {k}: {v}")

            # Локальный контекст
            if self.local_context:
                parts.append("ЛОКАЛЬНЫЙ КОНТЕКСТ:")
                for k, v in self.local_context.items():
                    parts.append(f"  {k}: {v}")

            # История
            if self.history:
                parts.append("ИСТОРИЯ:")
                for h in self.history[-3:]:  # Последние 3 сообщения
                    parts.append(f"  {h}")

            # Запрос
            parts.append(f"ЗАПРОС: {query}")

            return "\n".join(parts)

        def add_to_history(self, message):
            """Добавляет сообщение в историю."""
            self.history.append(message)

    # Демонстрация работы
    cm = ContextManager()
    cm.inject_context("user_role", "разработчик", "global")
    cm.inject_context("language", "Python", "global")
    cm.inject_context("project", "AI Assistant", "local")

    cm.add_to_history("Пользователь: Помоги с функцией")
    cm.add_to_history("Агент: Какую функцию нужно написать?")

    prompt = cm.build_prompt("Напиши функцию для сортировки")
    print("Промпт с инъекцией контекста:")
    print(prompt)

    # --- 3.2 History Management ---
    print("\n--- 3.2 Управление историей ---")

    # Стратегии управления размером истории
    class HistoryManager:
        def __init__(self, max_tokens=500):
            self.history = []
            self.max_tokens = max_tokens

        def add_message(self, role, content):
            """Добавляет сообщение в историю."""
            self.history.append({
                "role": role,
                "content": content,
                "tokens": len(content.split())
            })

        def get_tokens(self):
            """Подсчитывает общее количество токенов."""
            return sum(msg["tokens"] for msg in self.history)

        def truncate_history(self, strategy="sliding_window"):
            """Обрезает историю по выбранной стратегии."""
            total = self.get_tokens()
            if total <= self.max_tokens:
                return self.history

            if strategy == "sliding_window":
                # Удаляем старые сообщения
                while self.get_tokens() > self.max_tokens and len(self.history) > 2:
                    self.history.pop(0)

            elif strategy == "summarize":
                # Эмулируем суммаризацию старых сообщений
                if len(self.history) > 4:
                    old = self.history[:len(self.history)//2]
                    summary_tokens = sum(m["tokens"] for m in old)
                    self.history = self.history[len(self.history)//2:]
                    # Добавляем суммаризацию
                    self.history.insert(0, {
                        "role": "system",
                        "content": f"[Суммаризация предыдущего контекста: {summary_tokens} токенов]",
                        "tokens": 15
                    })

            elif strategy == "keep_recent":
                # Оставляем только последние N сообщений
                self.history = self.history[-4:]

            return self.history

    # Демонстрация различных стратегий
    hm = HistoryManager(max_tokens=50)

    # Добавляем сообщения
    messages = [
        ("user", "Привет! Как дела?"),
        ("assistant", "Привет! У меня всё отлично, спасибо!"),
        ("user", "Помоги написать код на Python"),
        ("assistant", "Конечно! Какую задачу нужно решить?"),
        ("user", "Нужна функция для сортировки списка"),
        ("assistant", "Вот функция сортировки пузырьком..."),
    ]

    for role, content in messages:
        hm.add_message(role, content)

    print(f"Исходное количество сообщений: {len(hm.history)}")
    print(f"Общее количество токенов: {hm.get_tokens()}")

    # Тестируем стратегии
    strategies = ["sliding_window", "summarize", "keep_recent"]
    for strategy in strategies:
        hm_copy = HistoryManager(max_tokens=50)
        for role, content in messages:
            hm_copy.add_message(role, content)
        hm_copy.truncate_history(strategy)
        print(f"\nСтратегия '{strategy}':")
        print(f"  Осталось сообщений: {len(hm_copy.history)}")
        print(f"  Токенов: {hm_copy.get_tokens()}")

    # --- 3.3 Template Variables ---
    print("\n--- 3.3 Шаблонные переменные ---")

    # Система шаблонов с переменными
    class PromptTemplate:
        def __init__(self, template):
            self.template = template
            self.variables = {}

        def set_variable(self, name, value):
            """Устанавливает переменную шаблона."""
            self.variables[name] = value

        def render(self):
            """Рендерит шаблон с подстановкой переменных."""
            result = self.template
            for name, value in self.variables.items():
                # Заменяем {name} на значение
                result = result.replace("{" + name + "}", str(value))
            return result

        def get_missing_variables(self):
            """Находит незаполненные переменные."""
            # Ищем все {variable} в шаблоне
            pattern = r'\{(\w+)\}'
            found = set(re.findall(pattern, self.template))
            return found - set(self.variables.keys())

    # Примеры шаблонов
    templates = {
        "analysis": """
Ты — аналитик данных.
Проект: {project_name}
Пользователь: {user_role}
Язык программирования: {language}

Задача: {task_description}

Дай конкретные рекомендации.
""",
        "code_review": """
Проведи ревью кода:
Файл: {file_name}
Функция: {function_name}

Требования:
1. Читаемость: {readability_score}/10
2. Эффективность: {efficiency_score}/10
3. Безопасность: {security_score}/10
"""
    }

    # Рендерим шаблон анализа
    analysis_template = PromptTemplate(templates["analysis"])
    analysis_template.set_variable("project_name", "AI Assistant")
    analysis_template.set_variable("user_role", "Разработчик")
    analysis_template.set_variable("language", "Python")
    analysis_template.set_variable("task_description", "Оптимизировать время отклика")

    print("Шаблон анализа:")
    print(analysis_template.render())

    # Проверяем незаполненные переменные
    review_template = PromptTemplate(templates["code_review"])
    review_template.set_variable("file_name", "agent.py")
    review_template.set_variable("function_name", "process_query")

    missing = review_template.get_missing_variables()
    print(f"\nНезаполненные переменные: {missing}")

    # --- 3.4 Conditional Prompting ---
    print("\n--- 3.4 Условный промптинг ---")

    # Динамическое формирование промпта на основе условий
    class ConditionalPrompt:
        def __init__(self):
            self.conditions = []

        def add_condition(self, condition_fn, prompt_fragment):
            """Добавляет условный фрагмент промпта."""
            self.conditions.append((condition_fn, prompt_fragment))

        def build(self, context):
            """Строит промпт, учитывая условия."""
            fragments = []
            for condition_fn, fragment in self.conditions:
                if condition_fn(context):
                    fragments.append(fragment)
            return "\n\n".join(fragments)

    # Создаём условный промпт
    cp = ConditionalPrompt()

    # Условие: если пользователь новичок
    cp.add_condition(
        lambda ctx: ctx.get("experience", "senior") == "junior",
        "Используй простой язык. Объясняй базовые концепции."
    )

    # Условие: если нужен код
    cp.add_condition(
        lambda ctx: ctx.get("needs_code", False),
        "Всегда приводи примеры кода с комментариями."
    )

    # Условие: если критическая задача
    cp.add_condition(
        lambda ctx: ctx.get("priority", "normal") == "critical",
        "ВНИМАНИЕ: Критическая задача! Давай только проверенные решения."
    )

    # Тестируем разные контексты
    contexts = [
        {"experience": "junior", "needs_code": True},
        {"experience": "senior", "priority": "critical"},
        {"experience": "junior", "priority": "critical"}
    ]

    for i, ctx in enumerate(contexts):
        prompt = cp.build(ctx)
        print(f"\nКонтекст {i+1}: {ctx}")
        print(f"Сгенерированный промпт:\n{prompt}")


# =============================================================================
# 4. Prompt Optimization
# =============================================================================

def demo_prompt_optimization():
    """Демонстрация оптимизации промптов."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Prompt Optimization")
    print("=" * 70)

    # --- 4.1 A/B Testing ---
    print("\n--- 4.1 A/B тестирование промптов ---")

    # Эмуляция A/B тестирования
    class ABTestPrompts:
        def __init__(self):
            self.variants = {}  # Варианты промптов
            self.metrics = {}   # Метрики для каждого варианта

        def add_variant(self, name, prompt):
            """Добавляет вариант промпта."""
            self.variants[name] = prompt
            self.metrics[name] = {"success": 0, "failure": 0, "total": 0}

        def record_result(self, variant_name, success):
            """Записывает результат использования промпта."""
            if variant_name in self.metrics:
                self.metrics[variant_name]["total"] += 1
                if success:
                    self.metrics[variant_name]["success"] += 1
                else:
                    self.metrics[variant_name]["failure"] += 1

        def get_statistics(self):
            """Вычисляет статистику по вариантам."""
            stats = {}
            for name, metrics in self.metrics.items():
                total = metrics["total"]
                if total > 0:
                    success_rate = metrics["success"] / total
                    stats[name] = {
                        "total": total,
                        "success_rate": success_rate,
                        "std_error": math.sqrt(success_rate * (1 - success_rate) / total)
                    }
            return stats

    # Создаём A/B тест
    ab_test = ABTestPrompts()

    # Варианты промптов
    prompts = {
        "short": "Переведи текст.",
        "detailed": "Ты — профессиональный переводчик. Переведи следующий текст, сохраняя стиль и контекст.",
        "with_examples": "Переведи текст. Пример: 'Hello' -> 'Привет'. Сохраняй формат."
    }

    for name, prompt in prompts.items():
        ab_test.add_variant(name, prompt)

    # Эмулируем результаты тестирования
    random.seed(42)
    for _ in range(20):
        # Симулируем разные результаты для разных промптов
        for variant in prompts:
            # Короткий промпт — 60% успех
            if variant == "short":
                success = random.random() < 0.60
            # Детальный — 85% успех
            elif variant == "detailed":
                success = random.random() < 0.85
            # С примерами — 80% успех
            else:
                success = random.random() < 0.80
            ab_test.record_result(variant, success)

    # Выводим результаты
    stats = ab_test.get_statistics()
    print("Результаты A/B тестирования промптов:")
    print("-" * 50)
    for name, stat in stats.items():
        print(f"  {name}:")
        print(f"    Всего тестов: {stat['total']}")
        print(f"    Успешность: {stat['success_rate']:.1%}")
        print(f"    Стандартная ошибка: {stat['std_error']:.3f}")

    # Определяем лучший вариант
    best_variant = max(stats.items(), key=lambda x: x[1]['success_rate'])
    print(f"\nЛучший вариант: {best_variant[0]} ({best_variant[1]['success_rate']:.1%})")

    # --- 4.2 Failure Analysis ---
    print("\n--- 4.2 Анализ ошибок ---")

    # Анализ типичных ошибок в промптах
    failure_types = {
        "vague_instruction": {
            "count": 15,
            "description": "Неоднозначные инструкции",
            "example": "Сделай хорошо",
            "fix": "Уточни критерии успеха"
        },
        "missing_context": {
            "count": 12,
            "description": "Недостаточно контекста",
            "example": "Исправь ошибку",
            "fix": "Добавь спецификацию ошибки"
        },
        "wrong_format": {
            "count": 8,
            "description": "Неправильный формат вывода",
            "example": "Ответь в свободной форме",
            "fix": "Укажи формат: JSON, список, текст"
        },
        "over_constraint": {
            "count": 5,
            "description": "Слишком много ограничений",
            "example": "10+ правил одновременно",
            "fix": "Приоритизируй ограничения"
        }
    }

    print("Типичные ошибки промптов:")
    total_failures = sum(f["count"] for f in failure_types.values())
    for ftype, info in failure_types.items():
        percentage = info["count"] / total_failures * 100
        print(f"\n  [{ftype}] ({info['count']} случаев, {percentage:.1f}%)")
        print(f"    Описание: {info['description']}")
        print(f"    Пример: \"{info['example']}\"")
        print(f"    Исправление: {info['fix']}")

    # Приоритизация исправлений
    sorted_fixes = sorted(failure_types.items(), key=lambda x: x[1]['count'], reverse=True)
    print("\nПриоритет исправлений:")
    for i, (ftype, info) in enumerate(sorted_fixes, 1):
        print(f"  {i}. {info['fix']} ({info['count']} случаев)")

    # --- 4.3 Iterative Improvement ---
    print("\n--- 4.3 Итеративное улучшение ---")

    # Процесс итеративного улучшения промпта
    iterations = [
        {
            "version": 1,
            "prompt": "Ответь на вопрос",
            "score": 0.45,
            "issues": ["Слишком абстрактно", "Нет формата"]
        },
        {
            "version": 2,
            "prompt": "Ты — помощник. Ответь на вопрос пользователя кратко.",
            "score": 0.62,
            "issues": ["Нет структуры", "Непонятен контекст"]
        },
        {
            "version": 3,
            "prompt": "Ты — помощник по Python. Ответь на вопрос:\n"
                     "1. Кратко опиши проблему\n"
                     "2. Дай решение с кодом\n"
                     "3. Объясни почему это работает",
            "score": 0.78,
            "issues": ["Нет обработки ошибок"]
        },
        {
            "version": 4,
            "prompt": "Ты — помощник по Python.\n"
                     "Задача пользователя: {question}\n\n"
                     "Формат ответа:\n"
                     "1. Анализ проблемы\n"
                     "2. Решение (код на Python)\n"
                     "3. Объяснение\n"
                     "4. Возможные ошибки и как их избежать",
            "score": 0.91,
            "issues": []
        }
    ]

    print("Итеративное улучшение промпта:")
    print("-" * 50)
    prev_score = 0
    for iteration in iterations:
        improvement = iteration["score"] - prev_score
        print(f"\nВерсия {iteration['version']} (Score: {iteration['score']:.2f})")
        print(f"  Улучшение: +{improvement:.2f}")
        print(f"  Проблемы: {', '.join(iteration['issues']) if iteration['issues'] else 'Нет'}")
        prev_score = iteration["score"]

    # График улучшения
    print("\nГрафик улучшения:")
    for iteration in iterations:
        bar_length = int(iteration["score"] * 30)
        bar = "█" * bar_length
        print(f"  v{iteration['version']}: {bar} {iteration['score']:.2f}")

    # --- 4.4 Cost Analysis ---
    print("\n--- 4.4 Анализ стоимости ---")

    # Анализ стоимости промптов
    cost_analysis = {
        "short_prompt": {
            "tokens": 10,
            "success_rate": 0.60,
            "cost_per_1000": 0.002,
            "total_cost_1000_queries": 0.002
        },
        "detailed_prompt": {
            "tokens": 50,
            "success_rate": 0.85,
            "cost_per_1000": 0.01,
            "total_cost_1000_queries": 0.01
        },
        "with_examples": {
            "tokens": 100,
            "success_rate": 0.80,
            "cost_per_1000": 0.02,
            "total_cost_1000_queries": 0.02
        }
    }

    # Эффективность по стоимости
    print("Эффективность промптов (на 1000 запросов):")
    print("-" * 50)
    for name, data in cost_analysis.items():
        # Эффективность = успешность / стоимость
        efficiency = data["success_rate"] / data["total_cost_1000_queries"]
        print(f"\n  {name}:")
        print(f"    Токенов: {data['tokens']}")
        print(f"    Успешность: {data['success_rate']:.1%}")
        print(f"    Стоимость: ${data['total_cost_1000_queries']:.4f}")
        print(f"    Эффективность: {efficiency:.0f} (успех/$)")

    # Рекомендация
    best_efficiency = max(
        cost_analysis.items(),
        key=lambda x: x[1]["success_rate"] / x[1]["total_cost_1000_queries"]
    )
    print(f"\nЛучшая эффективность: {best_efficiency[0]}")
    print(f"  Эффективность: {best_efficiency[1]['success_rate'] / best_efficiency[1]['total_cost_1000_queries']:.0f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Модуль 162: Prompt Engineering for Agents")
    print("Тема: системные промпты, few-shot, динамический промптинг\n")

    demo_system_prompt()
    demo_few_shot()
    demo_dynamic_prompting()
    demo_prompt_optimization()

    print("\n" + "=" * 70)
    print("Все демонстрации модуля 162 завершены.")
    print("=" * 70)
