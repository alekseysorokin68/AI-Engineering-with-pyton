"""163 — Agent Frameworks: LangChain, AutoGPT, CrewAI концепции

Темы:
  1. LangChain Concepts (chains, agents, tools, memory, callbacks)
  2. AutoGPT Patterns (goal-driven loop, web browsing, file operations)
  3. CrewAI Patterns (role-based agents, collaboration, task delegation)
  4. Framework Comparison (trade-offs, when to use what, custom vs framework)

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
# 1. LangChain Concepts
# =============================================================================

def demo_langchain():
    """Демонстрация концепций LangChain."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: LangChain Concepts")
    print("=" * 70)

    # --- 1.1 Chains ---
    print("\n--- 1.1 Цепочки (Chains) ---")

    # Эмуляция цепочки обработки
    class Chain:
        """Базовый класс цепочки обработки."""
        def __init__(self):
            self.steps = []

        def add_step(self, name, process_fn):
            """Добавляет шаг в цепочку."""
            self.steps.append({"name": name, "process": process_fn})

        def run(self, input_data):
            """Запускает цепочку обработки."""
            result = input_data
            print(f"  Входные данные: {result}")

            for step in self.steps:
                result = step["process"](result)
                print(f"  После '{step['name']}': {result}")

            return result

    # Создаём цепочку обработки текста
    text_chain = Chain()

    # Шаг 1: Удаление лишних пробелов
    text_chain.add_step("normalize", lambda x: " ".join(x.split()))

    # Шаг 2: Приведение к нижнему регистру
    text_chain.add_step("lowercase", lambda x: x.lower())

    # Шаг 3: Удаление знаков препинания
    text_chain.add_step("remove_punctuation", lambda x: re.sub(r'[^\w\s]', '', x))

    # Шаг 4: Токенизация
    text_chain.add_step("tokenize", lambda x: x.split())

    # Запускаем цепочку
    input_text = "  Привет,   Мир!   Как  дела?  "
    result = text_chain.run(input_text)
    print(f"  Результат: {result}")

    # --- 1.2 Agents ---
    print("\n--- 1.2 Агенты (Agents) ---")

    # Эмуляция агента с инструментами
    class Agent:
        """Агент с возможностью выбора инструментов."""
        def __init__(self, name):
            self.name = name
            self.tools = {}
            self.memory = []

        def register_tool(self, name, tool_fn, description=""):
            """Регистрирует инструмент."""
            self.tools[name] = {
                "function": tool_fn,
                "description": description
            }

        def think(self, query):
            """Агент анализирует запрос и выбирает инструмент."""
            # Простая логика выбора инструмента
            query_lower = query.lower()

            if any(word in query_lower for word in ["計算", "計算", "сколько", "посчитай"]):
                return "calculator"
            elif any(word in query_lower for word in ["найди", "поиск", "search"]):
                return "search"
            elif any(word in query_lower for word in ["запиши", "сохрани", "save"]):
                return "save"
            else:
                return "default"

        def run(self, query):
            """Выполняет запрос."""
            self.memory.append({"role": "user", "content": query})

            # Выбираем инструмент
            tool_name = self.think(query)
            print(f"  [{self.name}] Выбран инструмент: {tool_name}")

            # Выполняем инструмент
            if tool_name in self.tools:
                result = self.tools[tool_name]["function"](query)
            else:
                result = f"Обработка запроса: {query}"

            self.memory.append({"role": "assistant", "content": result})
            return result

    # Создаём агента
    assistant = Agent("Помощник")

    # Регистрируем инструменты
    assistant.register_tool(
        "calculator",
        lambda q: f"Результат вычисления: {sum(range(1, 11))}",
        "Математические вычисления"
    )
    assistant.register_tool(
        "search",
        lambda q: "Найдена информация: Python — язык программирования",
        "Поиск информации"
    )
    assistant.register_tool(
        "save",
        lambda q: "Данные сохранены в файл",
        "Сохранение данных"
    )

    # Тестируем агента
    queries = [
        "Посчитай сумму чисел от 1 до 10",
        "Найди информацию о Python",
        "Сохрани результат",
        "Расскажи анекдот"
    ]

    for query in queries:
        print(f"\n  Запрос: {query}")
        result = assistant.run(query)
        print(f"  Ответ: {result}")

    # --- 1.3 Tools ---
    print("\n--- 1.3 Инструменты (Tools) ---")

    # Библиотека инструментов
    class ToolKit:
        """Набор инструментов для агента."""
        def __init__(self):
            self.tools = {}

        def register(self, name, func, params=None):
            """Регистрирует инструмент."""
            self.tools[name] = {
                "function": func,
                "params": params or {},
                "usage_count": 0
            }

        def execute(self, name, **kwargs):
            """Выполняет инструмент."""
            if name not in self.tools:
                return f"Инструмент '{name}' не найден"

            tool = self.tools[name]
            tool["usage_count"] += 1

            # Проверяем параметры
            required = tool["params"].get("required", [])
            for param in required:
                if param not in kwargs:
                    return f"Отсутствует обязательный параметр: {param}"

            return tool["function"](**kwargs)

        def get_stats(self):
            """Возвращает статистику использования."""
            return {
                name: {
                    "usage_count": tool["usage_count"],
                    "description": tool["params"].get("description", "")
                }
                for name, tool in self.tools.items()
            }

    # Создаём набор инструментов
    toolkit = ToolKit()

    # Регистрируем инструменты
    toolkit.register(
        "web_search",
        lambda query, limit=5: f"Найдено {limit} результатов по запросу '{query}'",
        {"params": ["query", "limit"], "required": ["query"], "description": "Поиск в интернете"}
    )

    toolkit.register(
        "read_file",
        lambda path: f"Содержимое файла {path}: ...данные...",
        {"params": ["path"], "required": ["path"], "description": "Чтение файла"}
    )

    toolkit.register(
        "write_file",
        lambda path, content: f"Записано {len(content)} байт в {path}",
        {"params": ["path", "content"], "required": ["path", "content"], "description": "Запись в файл"}
    )

    # Используем инструменты
    print("Использование инструментов:")
    print(f"  1. {toolkit.execute('web_search', query='Python tutorials', limit=3)}")
    print(f"  2. {toolkit.execute('read_file', path='data.txt')}")
    print(f"  3. {toolkit.execute('write_file', path='output.txt', content='Hello')}")

    # Статистика
    print("\nСтатистика использования:")
    for name, stats in toolkit.get_stats().items():
        print(f"  {name}: {stats['usage_count']} вызовов")

    # --- 1.4 Memory ---
    print("\n--- 1.4 Память (Memory) ---")

    # Различные типы памяти
    class MemorySystem:
        """Система памяти агента."""
        def __init__(self, max_short_term=5):
            self.short_term = []    # Краткосрочная память
            self.long_term = {}     # Долгосрочная память
            self.semantic = {}      # Семантическая память
            self.max_short_term = max_short_term

        def add_short_term(self, message):
            """Добавляет в краткосрочную память."""
            self.short_term.append(message)
            if len(self.short_term) > self.max_short_term:
                # Эмулируем консолидацию в долгосрочную память
                old = self.short_term.pop(0)
                key = hashlib.md5(old.encode()).hexdigest()[:8]
                self.long_term[key] = old

        def add_semantic(self, key, value):
            """Добавляет в семантическую память."""
            self.semantic[key] = value

        def recall(self, query):
            """Вспоминает релевантную информацию."""
            results = []

            # Ищем в краткосрочной памяти
            for msg in self.short_term:
                if query.lower() in msg.lower():
                    results.append(("short_term", msg))

            # Ищем в долгосрочной памяти
            for key, msg in self.long_term.items():
                if query.lower() in msg.lower():
                    results.append(("long_term", msg))

            # Ищем в семантической памяти
            for key, value in self.semantic.items():
                if query.lower() in str(value).lower():
                    results.append(("semantic", f"{key}: {value}"))

            return results

    # Демонстрация работы памяти
    memory = MemorySystem(max_short_term=3)

    # Добавляем сообщения
    messages = [
        "Пользователь: Привет!",
        "Агент: Здравствуйте!",
        "Пользователь: Расскажи о Python",
        "Агент: Python — язык программирования",
        "Пользователь: Какие у него особенности?",
        "Агент: Простой синтаксис, большая экосистема"
    ]

    print("Добавление сообщений в память:")
    for msg in messages:
        memory.add_short_term(msg)
        print(f"  + {msg}")

    print(f"\nКраткосрочная память ({len(memory.short_term)} элементов):")
    for msg in memory.short_term:
        print(f"    {msg}")

    print(f"\nДолгосрочная память ({len(memory.long_term)} элементов):")
    for key, msg in list(memory.long_term.items())[:3]:
        print(f"    [{key}]: {msg}")

    # Добавляем семантические знания
    memory.add_semantic("python_created", "1991 год")
    memory.add_semantic("python_paradigm", "объектно-ориентированный")
    memory.add_semantic("python_use", "веб, данные, AI")

    # Ищем в памяти
    print("\nПоиск в памяти по запросу 'Python':")
    results = memory.recall("Python")
    for memory_type, content in results:
        print(f"  [{memory_type}] {content}")

    # --- 1.5 Callbacks ---
    print("\n--- 1.5 Callbacks ---")

    # Система колбэков
    class CallbackSystem:
        """Система колбэков для мониторинга."""
        def __init__(self):
            self.callbacks = {
                "on_start": [],
                "on_end": [],
                "on_error": [],
                "on_tool_use": []
            }

        def register(self, event, callback):
            """Регистрирует колбэк."""
            if event in self.callbacks:
                self.callbacks[event].append(callback)

        def trigger(self, event, **kwargs):
            """Вызывает все колбэки события."""
            results = []
            for callback in self.callbacks.get(event, []):
                result = callback(**kwargs)
                results.append(result)
            return results

    # Создаём систему колбэков
    callbacks = CallbackSystem()

    # Регистрируем колбэки
    def log_start(**kwargs):
        print(f"  [LOG] Начало выполнения: {kwargs.get('task', 'unknown')}")
        return "logged"

    def log_end(**kwargs):
        print(f"  [LOG] Завершение: {kwargs.get('task', 'unknown')}")
        return "logged"

    def log_error(**kwargs):
        print(f"  [ERROR] {kwargs.get('error', 'unknown error')}")
        return "error_logged"

    callbacks.register("on_start", log_start)
    callbacks.register("on_end", log_end)
    callbacks.register("on_error", log_error)

    # Тестируем колбэки
    print("\nТестирование колбэков:")
    callbacks.trigger("on_start", task="Обработка запроса")
    callbacks.trigger("on_end", task="Обработка запроса")

    # Эмуляция ошибки
    print("\nЭмуляция ошибки:")
    callbacks.trigger("on_error", error="Timeout при обращении к API")


# =============================================================================
# 2. AutoGPT Patterns
# =============================================================================

def demo_autogpt():
    """Демонстрация паттернов AutoGPT."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: AutoGPT Patterns")
    print("=" * 70)

    # --- 2.1 Goal-Driven Loop ---
    print("\n--- 2.1 Цикл, управляемый целями ---")

    # Эмуляция цикла AutoGPT
    class AutoGPTLoop:
        """Основной цикл AutoGPT."""
        def __init__(self, goals):
            self.goals = goals  # Список целей
            self.current_goal_idx = 0
            self.thoughts = []
            self.actions = []
            self.results = []

        def think(self):
            """Агент размышляет о текущей цели."""
            if self.current_goal_idx >= len(self.goals):
                return None

            current_goal = self.goals[self.current_goal_idx]
            thought = {
                "goal": current_goal,
                "plan": f"План достижения: {current_goal}",
                "reasoning": "Анализирую лучший подход..."
            }
            self.thoughts.append(thought)
            return thought

        def act(self, thought):
            """Агент выполняет действие."""
            action = {
                "type": "search",
                "target": thought["goal"],
                "details": f"Выполняю действие для: {thought['goal']}"
            }
            self.actions.append(action)
            return action

        def observe(self, action, result):
            """Агент наблюдает результат."""
            self.results.append({
                "action": action,
                "result": result
            })
            return result

        def should_continue(self):
            """Проверяет, нужно ли продолжать."""
            return self.current_goal_idx < len(self.goals)

        def run(self, max_iterations=10):
            """Запускает основной цикл."""
            iteration = 0
            while self.should_continue() and iteration < max_iterations:
                iteration += 1
                print(f"\n  --- Итерация {iteration} ---")

                # Размышление
                thought = self.think()
                if thought is None:
                    break
                print(f"  ЦЕЛЬ: {thought['goal']}")
                print(f"  ПЛАН: {thought['plan']}")

                # Действие
                action = self.act(thought)
                print(f"  ДЕЙСТВИЕ: {action['details']}")

                # Наблюдение (эмулируем результат)
                result = f"Результат для '{thought['goal']}': готово"
                self.observe(action, result)
                print(f"  РЕЗУЛЬТАТ: {result}")

                # Переход к следующей цели
                self.current_goal_idx += 1

            return self.results

    # Создаём цикл с целями
    goals = [
        "Найти информацию о курсе Python",
        "Составить план обучения",
        "Найти практические проекты",
        "Оценить время обучения"
    ]

    loop = AutoGPTLoop(goals)
    results = loop.run()

    print(f"\nВсего выполнено действий: {len(results)}")

    # --- 2.2 Web Browsing ---
    print("\n--- 2.2 Веб-браузинг ---")

    # Эмуляция веб-браузера
    class WebBrowser:
        """Эмуляция веб-браузера для агента."""
        def __init__(self):
            self.history = []
            self.current_page = None
            self.cache = {}

        def navigate(self, url):
            """Переходит на страницу."""
            # Эмулируем задержку сети
            time.sleep(0.01)

            # Проверяем кэш
            if url in self.cache:
                print(f"    [КЭШ] Страница {url} загружена из кэша")
                self.current_page = self.cache[url]
            else:
                # Эмулируем загрузку
                page_content = self._fetch_page(url)
                self.cache[url] = page_content
                self.current_page = page_content
                print(f"    [ЗАГРУЗКА] Страница {url} загружена")

            self.history.append(url)
            return self.current_page

        def _fetch_page(self, url):
            """Эмулирует загрузку страницы."""
            # Возвращаем тестовые данные
            return {
                "url": url,
                "title": f"Страница {url}",
                "content": f"Содержимое страницы {url}...",
                "links": [f"{url}/page{i}" for i in range(1, 4)]
            }

        def extract_links(self):
            """Извлекает ссылки со страницы."""
            if self.current_page:
                return self.current_page.get("links", [])
            return []

        def search(self, query):
            """Поиск по запросу."""
            # Эмулируем результаты поиска
            return [
                {"url": f"https://result{i}.com", "title": f"Результат {i} для '{query}'"}
                for i in range(1, 4)
            ]

    # Демонстрация работы браузера
    browser = WebBrowser()

    print("Поиск информации:")
    results = browser.search("Python教程")
    for r in results:
        print(f"  {r['title']}: {r['url']}")

    print("\nНавигация по результатам:")
    page = browser.navigate(results[0]["url"])
    print(f"  Загружена: {page['title']}")

    links = browser.extract_links()
    print(f"  Найдено ссылок: {len(links)}")

    # --- 2.3 File Operations ---
    print("\n--- 2.3 Файловые операции ---")

    # Эмуляция файловой системы
    class VirtualFileSystem:
        """Виртуальная файловая система для агента."""
        def __init__(self):
            self.files = {}
            self.permissions = {"read": True, "write": True, "execute": False}

        def create_file(self, path, content):
            """Создаёт файл."""
            if not self.permissions["write"]:
                return "Ошибка: нет прав на запись"

            self.files[path] = {
                "content": content,
                "created": time.time(),
                "modified": time.time()
            }
            return f"Файл {path} создан"

        def read_file(self, path):
            """Читает файл."""
            if not self.permissions["read"]:
                return "Ошибка: нет прав на чтение"

            if path in self.files:
                return self.files[path]["content"]
            return f"Ошибка: файл {path} не найден"

        def write_file(self, path, content):
            """Записывает в файл."""
            if not self.permissions["write"]:
                return "Ошибка: нет прав на запись"

            if path in self.files:
                self.files[path]["content"] = content
                self.files[path]["modified"] = time.time()
                return f"Файл {path} обновлён"
            else:
                return self.create_file(path, content)

        def list_files(self):
            """Список файлов."""
            return list(self.files.keys())

        def get_file_info(self, path):
            """Информация о файле."""
            if path in self.files:
                return {
                    "path": path,
                    "size": len(self.files[path]["content"]),
                    "created": self.files[path]["created"],
                    "modified": self.files[path]["modified"]
                }
            return None

    # Демонстрация работы с файлами
    fs = VirtualFileSystem()

    print("Файловые операции:")
    print(f"  1. {fs.create_file('data.txt', 'Данные для анализа')}")
    print(f"  2. {fs.create_file('config.json', '{\"key\": \"value\"}')}")
    print(f"  3. {fs.write_file('output.txt', 'Результат обработки')}")

    print("\nЧтение файлов:")
    for path in fs.list_files():
        content = fs.read_file(path)
        print(f"  {path}: {content[:30]}...")

    print("\nИнформация о файлах:")
    for path in fs.list_files():
        info = fs.get_file_info(path)
        print(f"  {path}: {info['size']} байт")

    # --- 2.4 Task Management ---
    print("\n--- 2.4 Управление задачами ---")

    # Система управления задачами
    class TaskManager:
        """Менеджер задач для агента."""
        def __init__(self):
            self.tasks = {}
            self.dependencies = {}

        def add_task(self, task_id, description, priority="normal"):
            """Добавляет задачу."""
            self.tasks[task_id] = {
                "description": description,
                "priority": priority,
                "status": "pending",
                "created": time.time()
            }

        def add_dependency(self, task_id, depends_on):
            """Добавляет зависимость задачи."""
            if task_id not in self.dependencies:
                self.dependencies[task_id] = []
            self.dependencies[task_id].append(depends_on)

        def complete_task(self, task_id):
            """Отмечает задачу как выполненную."""
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "completed"
                return f"Задача {task_id} выполнена"
            return f"Задача {task_id} не найдена"

        def get_ready_tasks(self):
            """Получает задачи, готовые к выполнению."""
            ready = []
            for task_id, task in self.tasks.items():
                if task["status"] != "pending":
                    continue

                # Проверяем зависимости
                deps = self.dependencies.get(task_id, [])
                all_deps_complete = all(
                    self.tasks.get(dep, {}).get("status") == "completed"
                    for dep in deps
                )

                if all_deps_complete:
                    ready.append(task_id)

            return ready

        def get_status_summary(self):
            """Возвращает сводку по статусам."""
            summary = collections.Counter(task["status"] for task in self.tasks.values())
            return dict(summary)

    # Создаём задачи
    tm = TaskManager()

    # Добавляем задачи
    tm.add_task("T1", "Собрать данные", "high")
    tm.add_task("T2", "Очистить данные", "normal")
    tm.add_task("T3", "Обработать данные", "normal")
    tm.add_task("T4", "Создать отчёт", "low")

    # Добавляем зависимости
    tm.add_dependency("T2", "T1")  # T2 зависит от T1
    tm.add_dependency("T3", "T2")  # T3 зависит от T2
    tm.add_dependency("T4", "T3")  # T4 зависит от T3

    print("Управление задачами:")
    print(f"  Всего задач: {len(tm.tasks)}")

    # Выполняем задачи по порядку
    iteration = 0
    while True:
        ready = tm.get_ready_tasks()
        if not ready:
            break

        iteration += 1
        print(f"\n  Итерация {iteration}:")
        for task_id in ready:
            print(f"    Выполняю: {task_id} - {tm.tasks[task_id]['description']}")
            tm.complete_task(task_id)

    print(f"\nИтоговая сводка: {tm.get_status_summary()}")


# =============================================================================
# 3. CrewAI Patterns
# =============================================================================

def demo_crewai():
    """Демонстрация паттернов CrewAI."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: CrewAI Patterns")
    print("=" * 70)

    # --- 3.1 Role-Based Agents ---
    print("\n--- 3.1 Агенты на основе ролей ---")

    # Определяем роли агентов
    class AgentRole:
        """Описание роли агента."""
        def __init__(self, name, role, goal, backstory):
            self.name = name
            self.role = role
            self.goal = goal
            self.backstory = backstory

        def __repr__(self):
            return f"Agent({self.name}: {self.role})"

    # Создаём роли
    roles = [
        AgentRole(
            name="Исследователь",
            role="Аналитик данных",
            goal="Найти и проанализировать информацию",
            backstory="Опытный аналитик с 10-летним стажем"
        ),
        AgentRole(
            name="Писатель",
            role="Контент-мейкер",
            goal="Создать качественный контент",
            backstory="Профессиональный писатель и редактор"
        ),
        AgentRole(
            name="Ревьюер",
            role="QA-специалист",
            goal="Обеспечить качество результата",
            backstory="Эксперт по контролю качества"
        )
    ]

    print("Определённые роли:")
    for role in roles:
        print(f"\n  {role.name}")
        print(f"    Роль: {role.role}")
        print(f"    Цель: {role.goal}")
        print(f"    Предыстория: {role.backstory}")

    # --- 3.2 Collaboration ---
    print("\n--- 3.2 Совместная работа ---")

    # Эмуляция совместной работы агентов
    class CollaborativeAgent:
        """Агент, работающий в команде."""
        def __init__(self, role):
            self.role = role
            self.tasks_completed = []
            self.messages = []

        def receive_task(self, task):
            """Получает задачу."""
            self.messages.append(f"Получена задача: {task}")
            return f"{self.role.name} получил задачу"

        def work_on_task(self, task):
            """Работает над задачей."""
            # Эмулируем работу
            result = f"{self.role.name} выполнил: {task}"
            self.tasks_completed.append(result)
            return result

        def send_to_agent(self, other_agent, message):
            """Отправляет сообщение другому агенту."""
            other_agent.messages.append(f"От {self.role.name}: {message}")
            return f"Сообщение отправлено {other_agent.role.name}"

    # Создаём команду
    team = [CollaborativeAgent(role) for role in roles]

    # Демонстрация совместной работы
    print("\nПроцесс совместной работы:")

    # Задача от исследователя
    print("\n1. Исследователь получает задачу:")
    team[0].receive_task("Исследовать рынок AI")
    result = team[0].work_on_task("Исследовать рынок AI")
    print(f"   {result}")

    # Исследователь отправляет данные писателю
    print("\n2. Исследователь отправляет данные писателю:")
    team[0].send_to_agent(team[1], "Данные для статьи: AI растёт на 40% в год")
    print(f"   Писатель получил: {team[1].messages[-1]}")

    # Писатель работает над контентом
    print("\n3. Писатель создаёт контент:")
    team[1].receive_task("Написать статью на основе данных")
    result = team[1].work_on_task("Написать статью")
    print(f"   {result}")

    # Писатель отправляет ревьюеру
    print("\n4. Писатель отправляет на ревью:")
    team[1].send_to_agent(team[2], "Статья готова, проверь качество")
    print(f"   Ревьюер получил: {team[2].messages[-1]}")

    # Ревьюер проверяет
    print("\n5. Ревьюер проверяет качество:")
    team[2].receive_task("Проверить статью на качество")
    result = team[2].work_on_task("Проверить качество статьи")
    print(f"   {result}")

    # --- 3.3 Task Delegation ---
    print("\n--- 3.3 Делегирование задач ---")

    # Система делегирования
    class TaskDelegationSystem:
        """Система делегирования задач."""
        def __init__(self):
            self.agents = {}
            self.task_queue = []
            self.completed_tasks = []

        def register_agent(self, agent_id, skills):
            """Регистрирует агента с его навыками."""
            self.agents[agent_id] = {
                "skills": skills,
                "workload": 0,
                "completed": 0
            }

        def add_task(self, task_id, required_skills, priority=1):
            """Добавляет задачу в очередь."""
            self.task_queue.append({
                "id": task_id,
                "required_skills": required_skills,
                "priority": priority,
                "status": "pending"
            })

        def find_best_agent(self, task):
            """Находит лучшего агента для задачи."""
            best_agent = None
            best_score = -1

            for agent_id, agent_info in self.agents.items():
                # Считаем совпадение навыков
                skill_match = len(
                    set(task["required_skills"]) & set(agent_info["skills"])
                )
                # Учитываем загрузку (меньше загрузка — лучше)
                workload_penalty = agent_info["workload"] * 0.1

                score = skill_match - workload_penalty

                if score > best_score:
                    best_score = score
                    best_agent = agent_id

            return best_agent

        def delegate_tasks(self):
            """Делегирует задачи агентам."""
            # Сортируем по приоритету
            self.task_queue.sort(key=lambda x: x["priority"], reverse=True)

            for task in self.task_queue:
                if task["status"] != "pending":
                    continue

                agent_id = self.find_best_agent(task)
                if agent_id:
                    task["status"] = "delegated"
                    task["assigned_to"] = agent_id
                    self.agents[agent_id]["workload"] += 1
                    print(f"  Задача {task['id']} -> Агент {agent_id}")

        def complete_task(self, task_id):
            """Отмечает задачу как выполненную."""
            for task in self.task_queue:
                if task["id"] == task_id:
                    task["status"] = "completed"
                    agent_id = task.get("assigned_to")
                    if agent_id and agent_id in self.agents:
                        self.agents[agent_id]["workload"] -= 1
                        self.agents[agent_id]["completed"] += 1
                    self.completed_tasks.append(task)
                    return True
            return False

    # Создаём систему делегирования
    delegation = TaskDelegationSystem()

    # Регистрируем агентов
    delegation.register_agent("agent_1", ["python", "数据分析", "статистика"])
    delegation.register_agent("agent_2", ["python", "機器学習", "нейросети"])
    delegation.register_agent("agent_3", ["python", "визуализация", "отчёты"])

    # Добавляем задачи
    delegation.add_task("T1", ["数据分析", "статистика"], priority=3)
    delegation.add_task("T2", ["нейросети", "python"], priority=2)
    delegation.add_task("T3", ["визуализация", "отчёты"], priority=1)
    delegation.add_task("T4", ["数据分析", "python"], priority=2)

    # Делегируем задачи
    print("Делегирование задач:")
    delegation.delegate_tasks()

    # Выполняем задачи
    print("\nВыполнение задач:")
    for task in delegation.task_queue:
        if task["status"] == "delegated":
            delegation.complete_task(task["id"])
            print(f"  Задача {task['id']} выполнена агентом {task['assigned_to']}")

    # Статистика
    print("\nСтатистика агентов:")
    for agent_id, info in delegation.agents.items():
        print(f"  {agent_id}: выполнено {info['completed']}, загрузка {info['workload']}")

    # --- 3.4 Communication Protocols ---
    print("\n--- 3.4 Протоколы коммуникации ---")

    # Различные протоколы общения
    class CommunicationProtocol:
        """Протокол коммуникации между агентами."""
        def __init__(self, protocol_type="broadcast"):
            self.protocol_type = protocol_type
            self.messages = []

        def send(self, sender, receiver, content):
            """Отправляет сообщение."""
            message = {
                "sender": sender,
                "receiver": receiver,
                "content": content,
                "timestamp": time.time()
            }
            self.messages.append(message)
            return message

        def broadcast(self, sender, content):
            """Отправляет сообщение всем."""
            message = {
                "sender": sender,
                "receiver": "all",
                "content": content,
                "timestamp": time.time()
            }
            self.messages.append(message)
            return message

        def get_messages(self, agent_id=None):
            """Получает сообщения для агента."""
            if agent_id is None:
                return self.messages
            return [
                msg for msg in self.messages
                if msg["receiver"] == agent_id or msg["receiver"] == "all"
            ]

    # Демонстрация протоколов
    print("Протоколы коммуникации:")

    # Прямая коммуникация
    protocol = CommunicationProtocol("direct")
    protocol.send("Agent_A", "Agent_B", "Привет! Нужна помощь с задачей")
    protocol.send("Agent_B", "Agent_A", "Конечно! Расскажи подробнее")
    print(f"\n  Прямая коммуникация ({len(protocol.messages)} сообщений):")
    for msg in protocol.messages:
        print(f"    {msg['sender']} -> {msg['receiver']}: {msg['content']}")

    # Широковещательная коммуникация
    broadcast_protocol = CommunicationProtocol("broadcast")
    broadcast_protocol.broadcast("Leader", "Внимание! Новая задача для всех!")
    print(f"\n  Широковещательная коммуникация:")
    print(f"    Сообщение от Leader: {broadcast_protocol.messages[0]['content']}")

    # Очередь сообщений
    print(f"\n  Сообщения для Agent_B:")
    messages_for_b = protocol.get_messages("Agent_B")
    for msg in messages_for_b:
        print(f"    от {msg['sender']}: {msg['content']}")


# =============================================================================
# 4. Framework Comparison
# =============================================================================

def demo_comparison():
    """Демонстрация сравнения фреймворков."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Framework Comparison")
    print("=" * 70)

    # --- 4.1 Feature Comparison ---
    print("\n--- 4.1 Сравнение возможностей ---")

    # Данные для сравнения
    frameworks = {
        "LangChain": {
            "maturity": 9,
            "flexibility": 8,
            "ease_of_use": 7,
            "community": 9,
            "documentation": 8,
            "performance": 7,
            "use_cases": ["RAG", "Chatbots", "Agents"],
            "complexity": "medium"
        },
        "AutoGPT": {
            "maturity": 6,
            "flexibility": 5,
            "ease_of_use": 6,
            "community": 7,
            "documentation": 5,
            "performance": 5,
            "use_cases": ["Autonomous tasks", "Research", "Automation"],
            "complexity": "high"
        },
        "CrewAI": {
            "maturity": 5,
            "flexibility": 7,
            "ease_of_use": 8,
            "community": 6,
            "documentation": 7,
            "performance": 6,
            "use_cases": ["Multi-agent", "Collaboration", "Complex workflows"],
            "complexity": "medium"
        }
    }

    # Выводим сравнение
    print("Сравнение возможностей фреймворков:")
    print("-" * 70)
    print(f"{'Фреймворк':<15} {'Зрелость':<10} {'Гибкость':<10} {'Простота':<10} {'Сообщество':<10}")
    print("-" * 70)

    for name, features in frameworks.items():
        print(f"{name:<15} "
              f"{features['maturity']}/10{'':<5} "
              f"{features['flexibility']}/10{'':<5} "
              f"{features['ease_of_use']}/10{'':<5} "
              f"{features['community']}/10")

    # --- 4.2 Trade-offs ---
    print("\n--- 4.2 Компромиссы ---")

    trade_offs = {
        "LangChain": {
            "pros": [
                "Большое сообщество",
                "Хорошая документация",
                "Множество интеграций"
            ],
            "cons": [
                "Может быть избыточным",
                "Сложность конфигурации",
                "Зависимости"
            ]
        },
        "AutoGPT": {
            "pros": [
                "Автономность",
                "Сложные задачи",
                "Интересные эксперименты"
            ],
            "cons": [
                "Непредсказуемость",
                "Высокое потребление ресурсов",
                "Сложность отладки"
            ]
        },
        "CrewAI": {
            "pros": [
                "Простота использования",
                "Хорошо для командной работы",
                "Чистый API"
            ],
            "cons": [
                "Молодой фреймворк",
                "Ограниченные возможности",
                "Меньше интеграций"
            ]
        }
    }

    for framework, info in trade_offs.items():
        print(f"\n{framework}:")
        print("  Плюсы:")
        for pro in info["pros"]:
            print(f"    + {pro}")
        print("  Минусы:")
        for con in info["cons"]:
            print(f"    - {con}")

    # --- 4.3 Use Case Selection ---
    print("\n--- 4.3 Выбор по应用场景 ---")

    use_cases = {
        "Простой чат-бот": "LangChain",
        "Анализ документов (RAG)": "LangChain",
        "Автономные исследования": "AutoGPT",
        "Мультиагентные системы": "CrewAI",
        "Сложные рабочие процессы": "CrewAI",
        "Прототипирование": "LangChain",
        "Долгосрочные задачи": "AutoGPT"
    }

    print("Рекомендации по выбору:")
    for use_case, recommendation in use_cases.items():
        print(f"  {use_case:<25} -> {recommendation}")

    # --- 4.4 Custom vs Framework ---
    print("\n--- 4.4 Своё решение vs Фреймворк ---")

    # Критерии для принятия решения
    criteria = {
        "time_to_market": {
            "question": "Важна ли скорость разработки?",
            "custom": "Медленно, но точно",
            "framework": "Быстро, но с ограничениями"
        },
        "maintainability": {
            "question": "Нужна ли поддержка долгосрочно?",
            "custom": "Полный контроль",
            "framework": "Зависимость от обновлений"
        },
        "performance": {
            "question": "Критична ли производительность?",
            "custom": "Оптимизировано под задачу",
            "framework": "Общие оптимизации"
        },
        "team_expertise": {
            "question": "Есть ли экспертиза в команде?",
            "custom": "Нужны глубокие знания",
            "framework": "Можно быстро начать"
        }
    }

    print("Критерии выбора:")
    print("-" * 50)
    for criterion, info in criteria.items():
        print(f"\n  {info['question']}")
        print(f"    Своё решение: {info['custom']}")
        print(f"    Фреймворк: {info['framework']}")

    # Рекомендация
    print("\nОбщая рекомендация:")
    print("  Начните с фреймворка (LangChain)")
    print("  Переходите на своё решение при необходимости")
    print("  Комбинируйте: фреймворк для прототипа, своё для продакшена")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Модуль 163: Agent Frameworks")
    print("Тема: LangChain, AutoGPT, CrewAI концепции\n")

    demo_langchain()
    demo_autogpt()
    demo_crewai()
    demo_comparison()

    print("\n" + "=" * 70)
    print("Все демонстрации модуля 163 завершены.")
    print("=" * 70)
