"""195 — LLM-Based Multi-Agent: CrewAI, AutoGen, multi-agent debate

Темы:
  1. LLM Agent Roles (specialized agents, role-based prompting, expertise simulation)
  2. Multi-Agent Debate (argumentation, fact-checking, consensus building)
  3. Workflow Orchestration (sequential, parallel, conditional agent flows)
  4. Agent Collaboration Patterns (review, pair programming, brainwriting)

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


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 1: Роли LLM-агентов
# ──────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """Симуляция LLM-агента с заданной ролью и областью экспертизы."""

    def __init__(self, name, role, expertise, knowledge_base):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.knowledge_base = knowledge_base  # {topic: fact}

    def build_system_prompt(self):
        """Формирует системный промпт на основе роли и экспертизы."""
        topics = ", ".join(self.expertise)
        return (
            f"Вы — {self.role}. Ваша экспертиза: {topics}. "
            f"Отвечайте профессионально и конкретно."
        )

    def respond(self, query):
        """Симулирует ответ агента на основе его базы знаний."""
        # Ищем релевантные факты из базы знаний
        relevant = {}
        for topic, fact in self.knowledge_base.items():
            # Простой поиск по ключевым словам
            keywords = topic.lower().split()
            query_lower = query.lower()
            if any(kw in query_lower for kw in keywords):
                relevant[topic] = fact

        if relevant:
            facts = "; ".join(relevant.values())
            confidence = min(0.95, 0.6 + 0.1 * len(relevant))
            return {
                "agent": self.name,
                "role": self.role,
                "response": facts,
                "confidence": round(confidence, 2),
                "sources": list(relevant.keys())
            }
        else:
            return {
                "agent": self.name,
                "role": self.role,
                "response": "У меня недостаточно данных по этому вопросу.",
                "confidence": 0.1,
                "sources": []
            }


def demo_llm_agent_roles():
    """Демонстрация ролей LLM-агентов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: Роли LLM-агентов (role-based prompting)")
    print("=" * 70)

    # Создаём агентов разных ролей
    agents = [
        LLMAgent(
            name="Алиса",
            role="исследователь данных",
            expertise=["статистика", "анализ данных", "машинное обучение"],
            knowledge_base={
                "статистика": "p-value < 0.05 обычно считается статистически значимым",
                "анализ данных": "EDA включает визуализацию распределений и корреляций",
                "машинное обучение": "Random Forest — ансамблевый метод, устойчивый к переобучению"
            }
        ),
        LLMAgent(
            name="Борис",
            role="архитектор ПО",
            expertise=["системный дизайн", "микросервисы", "базы данных"],
            knowledge_base={
                "системный дизайн": "Используйте принцип SRP и разделение ответственности",
                "микросервисы": "Каждый сервис должен иметь свою базу данных",
                "базы данных": "Нормализация до 3NF устраняет большинство аномалий"
            }
        ),
        LLMAgent(
            name="Вера",
            role="исследователь NLP",
            expertise=["обработка текста", "трансформеры", "эмбеддинги"],
            knowledge_base={
                "обработка текста": "Токенизация — первый этап NLP-пайплайна",
                "трансформеры": "Attention mechanism: Q*K^T / sqrt(d_k)",
                "эмбеддинги": "Word2Vec учитывает контекст через skip-gram или CBOW"
            }
        )
    ]

    # Пример 1: Системные промпты
    print("\n--- Пример 1: Системные промпты агентов ---")
    for agent in agents:
        prompt = agent.build_system_prompt()
        print(f"  {agent.name} ({agent.role}):")
        print(f"    {prompt}")
    print()

    # Пример 2: Специализированные запросы
    print("--- Пример 2: Специализированные запросы ---")
    queries = [
        "Что такое p-value в статистике?",
        "Как спроектировать микросервисную архитектуру?",
        "Как работает attention в трансформерах?"
    ]
    for q in queries:
        print(f"\n  Запрос: '{q}'")
        for agent in agents:
            result = agent.respond(q)
            print(f"    {result['agent']}: {result['response']}")
            print(f"    (уверенность: {result['confidence']})")
    print()

    # Пример 3: Эмуляция экспертизы через role-based prompting
    print("--- Пример 3: Эмуляция экспертизы ---")
    # Показываем, как роль определяет стиль ответа
    role_styles = {
        "исследователь данных": "На основании проведённого анализа данных...",
        "архитектор ПО": "С точки зрения архитектуры системы...",
        "исследователь NLP": "Согласно теории обработки естественного языка..."
    }
    for agent in agents:
        style = role_styles.get(agent.role, "...")
        print(f"  {agent.role}: '{style}' — так начинает ответ каждый раз")
    print()

    # Пример 4: Мультиагентный запрос — все агенты отвечают
    print("--- Пример 4: Мультиагентный запрос (все агенты) ---")
    query = "Как улучшить качество модели машинного обучения?"
    print(f"  Запрос: '{query}'")
    for agent in agents:
        result = agent.respond(query)
        print(f"  [{result['role']}] {result['agent']}: {result['response']}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 2: Мультиагентные дебаты
# ──────────────────────────────────────────────────────────────────────────────

class DebateEngine:
    """Движок мультиагентных дебатов с аргументацией и консенсусом."""

    def __init__(self, agents, max_rounds=3):
        self.agents = agents
        self.max_rounds = max_rounds
        self.history = []

    def generate_argument(self, agent, topic, stance, context=None):
        """Генерирует аргумент агента (симуляция)."""
        # Простая эвристика для генерации аргументов
        templates = {
            "pro": [
                f"Поддерживаю {topic}, потому что это повышает эффективность.",
                f"Данные показывают, что {topic} даёт преимущества.",
                f"Из практики: {topic} работает лучше альтернатив."
            ],
            "con": [
                f"Возражу: {topic} имеет серьёзные риски.",
                f"Статистика говорит об обратном — {topic} неэффективно.",
                f"Из опыта: проблемы с {topic} очевидны."
            ],
            "neutral": [
                f"Анализируя {topic}, вижу и плюсы, и минусы.",
                f"Нужны дополнительные данные по {topic}.",
                f"Баланс аргументов за и против {topic} примерно равен."
            ]
        }

        options = templates.get(stance, templates["neutral"])
        argument = random.choice(options)

        if context:
            argument += f" Контекст: {context}"

        return {
            "agent": agent.name,
            "stance": stance,
            "argument": argument,
            "round": len(self.history) // len(self.agents) + 1
        }

    def run_debate(self, topic):
        """Запускает раунд дебатов."""
        stances = ["pro", "con", "neutral"]
        print(f"\n  Тема дебатов: '{topic}'")
        print(f"  Количество раундов: {self.max_rounds}")
        print(f"  Участники: {', '.join(a.name for a in self.agents)}")
        print()

        for round_num in range(1, self.max_rounds + 1):
            print(f"  --- Раунд {round_num} ---")
            for agent in self.agents:
                stance = random.choice(stances)
                argument = self.generate_argument(agent, topic, stance)
                self.history.append(argument)
                print(f"    [{argument['stance'].upper():>7}] {argument['agent']}: "
                      f"{argument['argument']}")

        # Подсчёт голосов за/против
        votes = collections.Counter(a["stance"] for a in self.history)
        print(f"\n  Итоги: {dict(votes)}")
        return votes

    def find_consensus(self):
        """Ищет консенсус на основе истории дебатов."""
        if not self.history:
            return None

        # Анализируем позиции агентов
        agent_stances = collections.defaultdict(list)
        for entry in self.history:
            agent_stances[entry["agent"]].append(entry["stance"])

        # Определяем тенденцию каждого агента
        consensus = {}
        for agent_name, stances in agent_stances.items():
            pro_count = stances.count("pro")
            con_count = stances.count("con")
            if pro_count > con_count:
                consensus[agent_name] = "поддерживает"
            elif con_count > pro_count:
                consensus[agent_name] = "возражает"
            else:
                consensus[agent_name] = "нейтрален"

        return consensus


def demo_multi_agent_debate():
    """Демонстрация мультиагентных дебатов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: Мультиагентные дебаты")
    print("=" * 70)

    # Создаём агентов для дебатов
    debaters = [
        LLMAgent("Дмитрий", "оптимист", ["инновации"], {}),
        LLMAgent("Елена", "скептик", ["безопасность"], {}),
        LLMAgent("Фёдор", "аналитик", ["данные"], {})
    ]

    engine = DebateEngine(debaters, max_rounds=2)

    # Пример 1: Дебаты о внедрении AI
    print("\n--- Пример 1: Дебаты о внедрении AI в компанию ---")
    votes1 = engine.run_debate("внедрение AI в бизнес-процессы")

    # Пример 2: Поиск консенсуса
    print("\n--- Пример 2: Поиск консенсуса ---")
    consensus = engine.find_consensus()
    for agent, stance in consensus.items():
        print(f"  {agent}: {stance}")

    # Пример 3: Факт-чекинг аргументов
    print("\n--- Пример 3: Факт-чекинг аргументов ---")
    facts_db = {
        "AI": "Индекс AI-зрелости компаний вырос на 47% за 2023 год",
        "бизнес": "ROI от AI составляет в среднем 3.5x за 2 года"
    }

    for entry in engine.history:
        arg = entry["argument"]
        # Проверяем, содержит ли аргумент ключевые факты
        checked = False
        for keyword, fact in facts_db.items():
            if keyword.lower() in arg.lower():
                print(f"  [{entry['agent']}] Факт-чек: '{fact}'")
                checked = True
        if not checked:
            print(f"  [{entry['agent']}] Аргумент не содержит проверяемых фактов")

    # Пример 4: Взвешенная оценка аргументов
    print("\n--- Пример 4: Взвешенная оценка аргументов ---")
    # Каждый агент получает вес по своей экспертизе
    weights = {"оптимист": 0.3, "скептик": 0.4, "аналитик": 0.3}

    score = 0
    total_weight = 0
    for entry in engine.history:
        w = weights.get(entry["agent"], 0.33)
        if entry["stance"] == "pro":
            score += w
        elif entry["stance"] == "con":
            score -= w
        total_weight += w

    normalized = score / total_weight if total_weight else 0
    print(f"  Взвешенная оценка: {normalized:.2f} (от -1 до +1)")
    print(f"  Интерпретация: {'в целом за' if normalized > 0 else 'в целом против'}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 3: Оркестрация рабочих процессов
# ──────────────────────────────────────────────────────────────────────────────

class WorkflowOrchestrator:
    """Оркестратор рабочих процессов multi-agent системы."""

    def __init__(self):
        self.agents = {}
        self.results = {}

    def register_agent(self, name, role, capabilities):
        """Регистрирует агента в системе."""
        self.agents[name] = {
            "role": role,
            "capabilities": capabilities,
            "tasks_completed": 0
        }

    def sequential_flow(self, task, agent_sequence):
        """Последовательный поток: каждый агент обрабатывает результат предыдущего."""
        print(f"\n  Последовательный поток: {task}")
        print(f"  Цепочка: {' -> '.join(agent_sequence)}")

        current_output = task
        for agent_name in agent_sequence:
            agent = self.agents.get(agent_name)
            if agent:
                # Симулируем обработку
                processed = f"[{agent['role']}] Обработано: {current_output[:50]}..."
                current_output = processed
                print(f"    {agent_name} ({agent['role']}): {processed[:80]}")
                agent["tasks_completed"] += 1

        self.results[task] = current_output
        return current_output

    def parallel_flow(self, task, agent_names):
        """Параллельный поток: все агенты работают одновременно."""
        print(f"\n  Параллельный поток: {task}")
        print(f"  Агенты: {', '.join(agent_names)}")

        results = {}
        for agent_name in agent_names:
            agent = self.agents.get(agent_name)
            if agent:
                result = f"[{agent['role']}] Результат от {agent_name}"
                results[agent_name] = result
                agent["tasks_completed"] += 1
                print(f"    {agent_name}: готово")

        # Объединяем результаты
        combined = " | ".join(results.values())
        print(f"  Объединённый результат: {combined[:100]}")
        self.results[task] = combined
        return combined

    def conditional_flow(self, task, conditions):
        """Условный поток: выбор агента зависит от условия."""
        print(f"\n  Условный поток: {task}")

        for condition, agent_name in conditions:
            agent = self.agents.get(agent_name)
            if agent:
                print(f"  Условие '{condition}' -> {agent_name} ({agent['role']})")
                result = f"[{agent['role']}] Выполнено условие: {condition}"
                agent["tasks_completed"] += 1
                print(f"    Результат: {result[:80]}")
                return result

        print("  Ни одно условие не выполнено")
        return None


def demo_workflow_orchestration():
    """Демонстрация оркестрации рабочих процессов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: Оркестрация multi-agent процессов")
    print("=" * 70)

    orchestrator = WorkflowOrchestrator()

    # Регистрируем агентов
    orchestrator.register_agent("Аналитик", "аналитик данных", ["анализ", "визуализация"])
    orchestrator.register_agent("Разработчик", "backend-разработчик", ["код", "API"])
    orchestrator.register_agent("Тестер", "QA-инженер", ["тестирование", "баг-репорты"])
    orchestrator.register_agent("Менеджер", "проджект-менеджер", ["планирование", "коммуникация"])

    # Пример 1: Последовательный поток
    print("\n--- Пример 1: Последовательный поток ---")
    orchestrator.sequential_flow(
        "Разработка ML-пайплайна",
        ["Менеджер", "Аналитик", "Разработчик", "Тестер"]
    )

    # Пример 2: Параллельный поток
    print("\n--- Пример 2: Параллельный поток ---")
    orchestrator.parallel_flow(
        "Подготовка данных",
        ["Аналитик", "Разработчик"]
    )

    # Пример 3: Условный поток
    print("\n--- Пример 3: Условный поток ---")
    orchestrator.conditional_flow(
        "Обработка ошибки",
        [
            ("критическая ошибка", "Менеджер"),
            ("баг в коде", "Разработчик"),
            ("проблема с данными", "Аналитик")
        ]
    )

    # Пример 4: Статистика по агентам
    print("\n--- Пример 4: Статистика выполнения ---")
    for name, info in orchestrator.agents.items():
        print(f"  {name} ({info['role']}): "
              f"выполнено задач — {info['tasks_completed']}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 4: Паттерны сотрудничества агентов
# ──────────────────────────────────────────────────────────────────────────────

class CollaborationPatterns:
    """Паттерны сотрудничества multi-agent систем."""

    @staticmethod
    def review_pattern(author, reviewer, code):
        """Паттерн «Code Review»: один агент пишет, другой проверяет."""
        print(f"\n  [Code Review] Автор: {author}, Ревьюер: {reviewer}")

        # Автор создаёт код
        print(f"    {author} написал: {code[:60]}...")

        # Ревьюер проверяет
        issues = []
        if "TODO" in code:
            issues.append("Найден незавершённый TODO")
        if len(code) < 20:
            issues.append("Код слишком короткий для функции")
        if "print" in code and "error" not in code.lower():
            issues.append("Используется print вместо логирования")

        if issues:
            for issue in issues:
                print(f"    {reviewer} нашёл: {issue}")
            return {"status": "needs_revision", "issues": issues}
        else:
            print(f"    {reviewer}: Код одобрен!")
            return {"status": "approved", "issues": []}

    @staticmethod
    def pair_programming(agent1, agent2, task):
        """Паттерн «Pair Programming»: два агента работают вместе."""
        print(f"\n  [Pair Programming] {agent1} + {agent2}")
        print(f"    Задача: {task}")

        # Симуляция чередования ролей
        steps = [
            (agent1, "driver", "Пишу основную логику"),
            (agent2, "navigator", "Подсказываю оптимизацию"),
            (agent1, "driver", "Рефакторю по замечаниям"),
            (agent2, "navigator", "Проверяю edge cases")
        ]

        for agent, role, action in steps:
            print(f"    {agent} ({role}): {action}")

        return {
            "driver": agent1,
            "navigator": agent2,
            "steps_completed": len(steps)
        }

    @staticmethod
    def brainwriting(agents, topic, rounds=2):
        """Паттерн «Brainwriting»: анонимная генерация идей."""
        print(f"\n  [Brainwriting] Тема: {topic}")
        print(f"  Участники: {', '.join(agents)}")

        all_ideas = []
        for round_num in range(1, rounds + 1):
            print(f"\n    Раунд {round_num}:")
            for agent in agents:
                # Генерируем идею на основе хеша
                seed_val = hashlib.md5(
                    f"{agent}{topic}{round_num}".encode()
                ).hexdigest()
                random.seed(int(seed_val[:8], 16))

                ideas_pool = [
                    "Использовать агентную архитектуру",
                    "Внедрить автоматическое тестирование",
                    "Создать общий словарь терминов",
                    "Настроить мониторинг в реальном времени",
                    "Добавить механизм обратной связи",
                    "Реализовать кэширование результатов"
                ]
                idea = random.choice(ideas_pool)
                all_ideas.append({"author": "аноним", "idea": idea, "round": round_num})
                print(f"      Идея: {idea}")

        # Уникализируем идеи
        unique = list({i["idea"] for i in all_ideas})
        print(f"\n  Всего уникальных идей: {len(unique)}")
        random.seed(42)  # восстанавливаем seed
        return unique


def demo_collaboration_patterns():
    """Демонстрация паттернов сотрудничества."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Паттерны сотрудничества агентов")
    print("=" * 70)

    # Пример 1: Code Review
    print("\n--- Пример 1: Code Review ---")
    code_good = "def predict(x): return model(x)"
    code_bad = "print('TODO: implement this')"
    CollaborationPatterns.review_pattern("Автор", "Ревьюер", code_good)
    CollaborationPatterns.review_pattern("Автор", "Ревьюер", code_bad)

    # Пример 2: Pair Programming
    print("\n--- Пример 2: Pair Programming ---")
    CollaborationPatterns.pair_programming(
        "Разработчик-1", "Разработчик-2",
        "Реализовать API для классификации"
    )

    # Пример 3: Brainwriting
    print("\n--- Пример 3: Brainwriting ---")
    ideas = CollaborationPatterns.brainwriting(
        ["Участник_A", "Участник_B", "Участник_C"],
        "Улучшение multi-agent системы",
        rounds=2
    )

    # Пример 4: Сравнение паттернов
    print("\n--- Пример 4: Сравнение паттернов ---")
    patterns = {
        "Code Review": {"speed": "средняя", "качество": "высокое",
                        "масштабируемость": "хорошая"},
        "Pair Programming": {"speed": "низкая", "качество": "очень высокое",
                             "масштабируемость": "ограниченная"},
        "Brainwriting": {"speed": "высокая", "качество": "среднее",
                         "масштабируемость": "отличная"}
    }
    for name, props in patterns.items():
        print(f"\n  {name}:")
        for prop, value in props.items():
            print(f"    {prop}: {value}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_llm_agent_roles()
    demo_multi_agent_debate()
    demo_workflow_orchestration()
    demo_collaboration_patterns()
