"""157 — Tool Integration: вызов функций, реестр инструментов, валидация параметров

Темы:
  1. Tool Definition (JSON schema, parameter types, required fields)
  2. Tool Registry (registration, lookup, dynamic tool loading)
  3. Function Calling (parsing LLM output, extracting tool calls, validation)
  4. Tool Execution (sandboxed execution, timeout, retry, error handling)

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
# 1. Tool Definition — определение инструментов через JSON Schema
# ============================================================

def demo_tool_definition():
    print("=" * 70)
    print("DEMO 1: Tool Definition — JSON Schema, типы параметров")
    print("=" * 70)

    # --- 1.1 Базовое определение инструмента ---
    print("\n--- 1.1 Базовое определение инструмента ---")

    def define_tool(name, description, parameters, returns):
        """Определение инструмента с JSON Schema параметрами."""
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": [k for k, v in parameters.items() if v.get("required", False)],
            },
            "returns": returns,
        }

    # Определение инструмента для поиска
    search_tool = define_tool(
        name="search_knowledge_base",
        description="Поиск информации в базе знаний по запросу",
        parameters={
            "query": {
                "type": "string",
                "description": "Поисковый запрос",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Максимальное количество результатов",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
            "category": {
                "type": "string",
                "description": "Категория поиска",
                "enum": ["all", "documents", "code", "images"],
                "default": "all",
            },
        },
        returns={"type": "array", "items": {"type": "object"}},
    )

    print(f"  Инструмент: {search_tool['name']}")
    print(f"  Описание: {search_tool['description']}")
    print(f"  Параметры:")
    for pname, pschema in search_tool["parameters"]["properties"].items():
        required = " [обязательный]" if pname in search_tool["parameters"]["required"] else ""
        print(f"    - {pname}: {pschema['type']}{required}")
        print(f"      {pschema['description']}")

    # --- 1.2 Валидация параметров ---
    print("\n--- 1.2 Валидация параметров по схеме ---")

    def validate_params(tool_def, params):
        """Валидация параметров инструмента по JSON Schema."""
        errors = []
        schema = tool_def["parameters"]

        # Проверка обязательных полей
        for req_field in schema.get("required", []):
            if req_field not in params:
                errors.append(f"Отсутствует обязательный параметр '{req_field}'")

        # Проверка типов и ограничений
        for pname, pvalue in params.items():
            if pname not in schema.get("properties", {}):
                errors.append(f"Неизвестный параметр '{pname}'")
                continue

            pschema = schema["properties"][pname]
            expected_type = pschema.get("type")

            # Проверка типа
            type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
            if expected_type in type_map:
                if not isinstance(pvalue, type_map[expected_type]):
                    errors.append(f"Параметр '{pname}': ожидался {expected_type}, "
                                 f"получен {type(pvalue).__name__}")
                    continue

            # Проверка enum
            if "enum" in pschema and pvalue not in pschema["enum"]:
                errors.append(f"Параметр '{pname}': значение '{pvalue}' "
                             f"не входит в {pschema['enum']}")

            # Проверка min/max
            if "minimum" in pschema and isinstance(pvalue, (int, float)):
                if pvalue < pschema["minimum"]:
                    errors.append(f"Параметр '{pname}': {pvalue} < minimum ({pschema['minimum']})")
            if "maximum" in pschema and isinstance(pvalue, (int, float)):
                if pvalue > pschema["maximum"]:
                    errors.append(f"Параметр '{pname}': {pvalue} > maximum ({pschema['maximum']})")

        return errors

    # Тесты валидации
    test_cases = [
        {"query": "python tutorial", "max_results": 10},                    # OK
        {"query": 123},                                                      # Неверный тип
        {"max_results": 5},                                                  # Нет query
        {"query": "test", "category": "videos"},                             # Неверный enum
        {"query": "test", "max_results": 100},                               # Превышен maximum
        {"query": "test", "max_results": 10, "category": "code", "extra": 1}, # Лишний параметр
    ]

    for i, params in enumerate(test_cases):
        errors = validate_params(search_tool, params)
        status = "OK" if not errors else f"ОШИБКИ: {errors}"
        print(f"  Тест {i+1}: {params}")
        print(f"    → {status}")

    # --- 1.3 Сложные типы (вложенные объекты, массивы) ---
    print("\n--- 1.3 Сложные типы: вложенные объекты и массивы ---")

    complex_tool = define_tool(
        name="analyze_code",
        description="Анализ кода с указанием правил и формата вывода",
        parameters={
            "code": {"type": "string", "description": "Исходный код", "required": True},
            "language": {"type": "string", "description": "Язык программирования", "required": True},
            "rules": {
                "type": "array",
                "description": "Список правил анализа",
                "items": {"type": "string"},
            },
            "options": {
                "type": "object",
                "description": "Дополнительные опции",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "fix_suggestions": {"type": "boolean", "default": True},
                },
            },
        },
        returns={"type": "object", "properties": {"issues": {"type": "array"}}},
    )

    # Печать структуры
    print(f"  Инструмент: {complex_tool['name']}")
    props = complex_tool["parameters"]["properties"]
    for pname, pschema in props.items():
        ptype = pschema["type"]
        if ptype == "array":
            item_type = pschema.get("items", {}).get("type", "?")
            print(f"    - {pname}: array<{item_type}>")
        elif ptype == "object":
            sub_props = list(pschema.get("properties", {}).keys())
            print(f"    - {pname}: object{{{', '.join(sub_props)}}}")
        else:
            print(f"    - {pname}: {ptype}")

    # --- 1.4 Генерация схемы из функции ---
    print("\n--- 1.4 Автоматическая генерация схемы из Python-функции ---")

    def generate_schema(func):
        """Генерация JSON Schema из сигнатуры Python-функции."""
        import inspect
        sig = inspect.signature(func)
        schema = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            prop = {"type": "string"}  # По умолчанию строка

            # Определение типа из аннотации
            if param.annotation != inspect.Parameter.empty:
                type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
                prop["type"] = type_map.get(param.annotation, "string")

            # Определение обязательности и значения по умолчанию
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
            else:
                prop["default"] = param.default

            schema["properties"][param_name] = prop

        return schema

    # Пример функции
    def add_numbers(a: int, b: int, precision: int = 2) -> float:
        """Сложение двух чисел с точностью."""
        return round(a + b, precision)

    generated_schema = generate_schema(add_numbers)
    print(f"  Функция: add_numbers(a: int, b: int, precision: int = 2)")
    print(f"  Сгенерированная схема:")
    print(f"    type: {generated_schema['type']}")
    print(f"    required: {generated_schema['required']}")
    for pname, pschema in generated_schema["properties"].items():
        default = f" = {pschema['default']}" if "default" in pschema else " [обязательный]"
        print(f"    {pname}: {pschema['type']}{default}")

    print()


# ============================================================
# 2. Tool Registry — реестр инструментов
# ============================================================

def demo_tool_registry():
    print("=" * 70)
    print("DEMO 2: Tool Registry — регистрация, поиск, динамическая загрузка")
    print("=" * 70)

    # --- 2.1 Базовый реестр ---
    print("\n--- 2.1 Базовый реестр инструментов ---")

    class ToolRegistry:
        """Реестр для хранения и поиска инструментов."""

        def __init__(self):
            self._tools = {}
            self._categories = collections.defaultdict(list)

        def register(self, name, func, description="", category="general",
                     schema=None):
            """Регистрация инструмента."""
            self._tools[name] = {
                "func": func,
                "description": description,
                "category": category,
                "schema": schema or {},
                "call_count": 0,
            }
            self._categories[category].append(name)

        def get(self, name):
            """Получение инструмента по имени."""
            if name not in self._tools:
                raise KeyError(f"Инструмент '{name}' не найден")
            return self._tools[name]["func"]

        def list_tools(self, category=None):
            """Список всех инструментов."""
            if category:
                names = self._categories.get(category, [])
                return [(n, self._tools[n]) for n in names]
            return list(self._tools.items())

        def search(self, query):
            """Поиск инструментов по описанию."""
            results = []
            query_lower = query.lower()
            for name, info in self._tools.items():
                if (query_lower in name.lower() or
                    query_lower in info["description"].lower()):
                    results.append((name, info))
            return results

    registry = ToolRegistry()

    # Регистрация инструментов
    registry.register("add", lambda a, b: a + b,
                     description="Сложение двух чисел", category="math")
    registry.register("multiply", lambda a, b: a * b,
                     description="Умножение двух чисел", category="math")
    registry.register("upper", lambda s: s.upper(),
                     description="Перевод строки в верхний регистр", category="text")
    registry.register("reverse", lambda s: s[::-1],
                     description="Разворот строки", category="text")
    registry.register("factorial", lambda n: math.factorial(n),
                     description="Вычисление факториала числа", category="math")

    # Список всех инструментов
    all_tools = registry.list_tools()
    print(f"  Всего инструментов: {len(all_tools)}")
    for name, info in all_tools:
        print(f"    - {name} [{info['category']}]: {info['description']}")

    # --- 2.2 Поиск и фильтрация ---
    print("\n--- 2.2 Поиск и фильтрация инструментов ---")

    # Поиск по описанию
    search_results = registry.search("чисел")
    print(f"  Поиск 'чисел': {[name for name, _ in search_results]}")

    # Фильтрация по категории
    math_tools = registry.list_tools(category="math")
    print(f"  Инструменты категории 'math': {[name for name, _ in math_tools]}")

    text_tools = registry.list_tools(category="text")
    print(f"  Инструменты категории 'text': {[name for name, _ in text_tools]}")

    # --- 2.3 Динамическая загрузка ---
    print("\n--- 2.3 Динамическая загрузка инструментов ---")

    # Определение инструментов как данных (можно загрузить из JSON)
    tool_definitions = [
        {
            "name": "sqrt",
            "description": "Квадратный корень числа",
            "category": "math",
            "func_name": "sqrt",
        },
        {
            "name": "log",
            "description": "Логарифм числа (по основанию 10)",
            "category": "math",
            "func_name": "log10",
        },
        {
            "name": "capitalize",
            "description": "Капитализация строки",
            "category": "text",
            "func_name": "capitalize",
        },
    ]

    # Маппинг имён функций
    func_map = {
        "sqrt": lambda x: math.sqrt(x),
        "log10": lambda x: math.log10(x) if x > 0 else float("-inf"),
        "capitalize": lambda s: s.capitalize(),
    }

    # Динамическая регистрация
    loaded_count = 0
    for defn in tool_definitions:
        func = func_map.get(defn["func_name"])
        if func:
            registry.register(
                name=defn["name"],
                func=func,
                description=defn["description"],
                category=defn["category"],
            )
            loaded_count += 1
            print(f"  Загружен: {defn['name']} ({defn['description']})")

    print(f"\n  Всего загружено: {loaded_count}")
    print(f"  Всего инструментов в реестре: {len(registry.list_tools())}")

    # --- 2.4 Вызов инструментов через реестр ---
    print("\n--- 2.4 Вызов инструментов через реестр ---")

    def call_tool(registry, tool_name, *args, **kwargs):
        """Вызов инструмента через реестр с подсчётом вызовов."""
        func = registry.get(tool_name)
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        registry._tools[tool_name]["call_count"] += 1
        return result, duration

    # Вызовы
    calls = [
        ("add", (3, 7), {}),
        ("multiply", (4, 5), {}),
        ("sqrt", (144,), {}),
        ("upper", ("hello world",), {}),
        ("factorial", (10,), {}),
        ("add", (100, 200), {}),
    ]

    for tool_name, args, kwargs in calls:
        result, duration = call_tool(registry, tool_name, *args, **kwargs)
        print(f"  {tool_name}{args} = {result} ({duration*1000:.2f}мс)")

    # Статистика вызовов
    print("\n  Статистика вызовов:")
    for name, info in sorted(registry.list_tools(),
                             key=lambda x: -x[1]["call_count"]):
        if info["call_count"] > 0:
            print(f"    {name}: {info['call_count']} вызовов")

    print()


# ============================================================
# 3. Function Calling — парсинг вывода LLM и вызов функций
# ============================================================

def demo_function_calling():
    print("=" * 70)
    print("DEMO 3: Function Calling — парсинг LLM output, извлечение tool calls")
    print("=" * 70)

    # --- 3.1 Парсинг JSON tool calls ---
    print("\n--- 3.1 Парсинг JSON tool calls из текста ---")

    def extract_tool_calls(text):
        """Извлечение tool calls из текста LLM."""
        calls = []

        # Паттерн 1: JSON-блоки
        json_pattern = r'```json\s*\n(.*?)\n\s*```'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "tool" in data:
                    calls.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "tool" in item:
                            calls.append(item)
            except json.JSONDecodeError:
                continue

        # Паттерн 2: Функциональный вызов
        func_pattern = r'(\w+)\(([^)]*)\)'
        for match in re.finditer(func_pattern, text):
            func_name = match.group(1)
            if func_name in ["print", "len", "range", "str", "int", "float"]:
                continue  # Пропуск встроенных функций
            args_str = match.group(2)
            calls.append({
                "tool": func_name,
                "args": [a.strip().strip("'\"") for a in args_str.split(",") if a.strip()],
            })

        return calls

    # Тестовый текст LLM
    llm_output = """
    Я помогу вам с этой задачей. Сначала выполню поиск.

    ```json
    {"tool": "search", "args": {"query": "python async", "max_results": 5}}
    ```

    Затем вычислю результат:
    calculate(expression="2**10 + 5**3")

    И сохраню в файл:
    ```json
    {"tool": "write_file", "args": {"path": "result.txt", "content": "done"}}
    ```
    """

    calls = extract_tool_calls(llm_output)
    print(f"  Найдено tool calls: {len(calls)}")
    for i, call in enumerate(calls, 1):
        tool = call.get("tool", "?")
        args = call.get("args", {})
        print(f"    {i}. {tool}({args})")

    # --- 3.2 Валидация tool calls ---
    print("\n--- 3.2 Валидация tool calls перед выполнением ---")

    # Реестр доступных инструментов
    available_tools = {
        "search": {"params": ["query", "max_results"]},
        "calculate": {"params": ["expression"]},
        "write_file": {"params": ["path", "content"]},
        "send_email": {"params": ["to", "subject", "body"]},
    }

    def validate_tool_call(call, available):
        """Валидация tool call."""
        errors = []
        tool_name = call.get("tool")

        # Проверка существования инструмента
        if tool_name not in available:
            errors.append(f"Инструмент '{tool_name}' не найден")
            return errors

        # Проверка параметров
        args = call.get("args", {})
        if isinstance(args, dict):
            required = available[tool_name]["params"]
            for param in required:
                if param not in args:
                    errors.append(f"Отсутствует параметр '{param}'")

            # Проверка лишних параметров
            for param in args:
                if param not in required:
                    errors.append(f"Неизвестный параметр '{param}'")

        return errors

    # Тесты валидации
    test_calls = [
        {"tool": "search", "args": {"query": "test", "max_results": 5}},       # OK
        {"tool": "search", "args": {}},                                         # Нет query
        {"tool": "unknown_tool", "args": {"x": 1}},                             # Неизвестный инструмент
        {"tool": "calculate", "args": {"expression": "1+1", "extra": "x"}},     # Лишний параметр
        {"tool": "write_file", "args": {"path": "f.txt"}},                      # Нет content
    ]

    for call in test_calls:
        errors = validate_tool_call(call, available_tools)
        status = "OK" if not errors else f"ОШИБКИ: {errors}"
        print(f"  {call['tool']}: {status}")

    # --- 3.3 Безопасный вызов функций ---
    print("\n--- 3.3 Безопасный вызов функций (sandbox) ---")

    # Ограниченный набор функций
    safe_functions = {
        "add": lambda a, b: a + b,
        "multiply": lambda a, b: a * b,
        "upper": lambda s: s.upper(),
        "reverse": lambda s: s[::-1],
        "len": lambda s: len(str(s)),
    }

    def safe_call(func_name, args):
        """Безопасный вызов функции из白листа."""
        if func_name not in safe_functions:
            return {"error": f"Функция '{func_name}' не разрешена"}

        try:
            # Преобразование аргументов
            converted_args = []
            for arg in args:
                try:
                    converted_args.append(int(arg))
                except ValueError:
                    try:
                        converted_args.append(float(arg))
                    except ValueError:
                        converted_args.append(arg.strip("'\""))

            result = safe_functions[func_name](*converted_args)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    # Тесты безопасного вызова
    safe_tests = [
        ("add", ["3", "5"]),
        ("multiply", ["4", "7"]),
        ("upper", ["hello"]),
        ("reverse", ["python"]),
        ("unknown", ["1", "2"]),
        ("add", ["abc", "3"]),
    ]

    for func_name, args in safe_tests:
        result = safe_call(func_name, args)
        print(f"  {func_name}({args}) → {result}")

    # --- 3.4 Форматирование tool call для LLM ---
    print("\n--- 3.4 Форматирование tool call для отправки в LLM ---")

    def format_tools_for_llm(tools):
        """Форматирование списка инструментов для промпта LLM."""
        prompt_parts = ["Доступные инструменты:\n"]

        for tool in tools:
            params_desc = []
            for pname, pschema in tool.get("parameters", {}).items():
                required = "[обязательный]" if pschema.get("required") else "[опциональный]"
                params_desc.append(f"    - {pname} ({pschema['type']}): "
                                  f"{pschema.get('description', '')} {required}")

            prompt_parts.append(f"## {tool['name']}")
            prompt_parts.append(f"Описание: {tool['description']}")
            if params_desc:
                prompt_parts.append("Параметры:")
                prompt_parts.extend(params_desc)
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    tools_for_llm = [
        {
            "name": "search",
            "description": "Поиск информации",
            "parameters": {
                "query": {"type": "string", "description": "Запрос", "required": True},
                "limit": {"type": "integer", "description": "Лимит", "required": False},
            },
        },
        {
            "name": "calculate",
            "description": "Вычисление выражения",
            "parameters": {
                "expression": {"type": "string", "description": "Выражение", "required": True},
            },
        },
    ]

    formatted = format_tools_for_llm(tools_for_llm)
    print(formatted)

    print()


# ============================================================
# 4. Tool Execution — выполнение инструментов с песочницей
# ============================================================

def demo_tool_execution():
    print("=" * 70)
    print("DEMO 4: Tool Execution — песочница, таймаут, повтор, ошибки")
    print("=" * 70)

    # --- 4.1 Песочница выполнения ---
    print("\n--- 4.1 Песочница выполнения (sandbox) ---")

    class ToolSandbox:
        """Песочница для безопасного выполнения инструментов."""

        ALLOWED_BUILTINS = {"abs", "round", "min", "max", "sum", "len", "str",
                           "int", "float", "bool", "list", "dict", "tuple"}

        def __init__(self):
            self.execution_log = []

        def execute(self, code, context=None):
            """Выполнение кода в ограниченном окружении."""
            # Ограниченное пространство имён
            safe_globals = {"__builtins__": {}}
            for name in self.ALLOWED_BUILTINS:
                safe_globals["__builtins__"][name] = __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)

            if context:
                safe_globals.update(context)

            try:
                result = eval(code, safe_globals)
                self.execution_log.append({"code": code, "result": result, "error": None})
                return {"success": True, "result": result}
            except Exception as e:
                self.execution_log.append({"code": code, "result": None, "error": str(e)})
                return {"success": False, "error": str(e)}

    sandbox = ToolSandbox()

    # Безопасные выражения
    safe_codes = [
        "abs(-42)",
        "round(3.14159, 2)",
        "max([1, 5, 3, 9, 2])",
        "sum([10, 20, 30])",
        "len([1, 2, 3, 4, 5])",
    ]

    for code in safe_codes:
        result = sandbox.execute(code)
        print(f"  {code} → {result}")

    # Опасные выражения (должны быть заблокированы)
    dangerous_codes = [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "exec('import os')",
    ]

    print("\n  Опасные выражения (должны быть заблокированы):")
    for code in dangerous_codes:
        result = sandbox.execute(code)
        status = "ЗАБЛОКИРОВАНО" if not result["success"] else "ПРОШЛО!"
        print(f"  {code[:40]}... → {status}")

    # --- 4.2 Таймаут выполнения ---
    print("\n--- 4.2 Таймаут выполнения ---")

    def execute_with_timeout(func, args, timeout_sec=1.0):
        """Выполнение функции с ограничением по времени."""
        import threading

        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        start = time.time()
        thread.start()
        thread.join(timeout=timeout_sec)
        duration = time.time() - start

        if thread.is_alive():
            return {"success": False, "error": "Таймаут", "duration": duration}

        if exception[0]:
            return {"success": False, "error": str(exception[0]), "duration": duration}

        return {"success": True, "result": result[0], "duration": duration}

    # Быстрые функции
    fast_func = lambda x: x * 2
    result = execute_with_timeout(fast_func, (5,), timeout_sec=1.0)
    print(f"  Быстрая функция (5*2): {result}")

    # Медленная функция (имитация)
    def slow_func(n):
        """Имитация медленной операции."""
        time.sleep(0.1)  # Небольшая задержка
        return sum(range(n))

    result = execute_with_timeout(slow_func, (1000,), timeout_sec=1.0)
    print(f"  Медленная функция (сумма 0..999): {result}")

    # Очень медленная функция (таймаут)
    def very_slow_func():
        time.sleep(2.0)
        return "done"

    result = execute_with_timeout(very_slow_func, (), timeout_sec=0.1)
    print(f"  Очень медленная функция (таймаут 0.1с): {result}")

    # --- 4.3 Retry с экспоненциальной задержкой ---
    print("\n--- 4.3 Retry с экспоненциальной задержкой ---")

    def retry_with_backoff(func, args, max_retries=3, base_delay=0.1):
        """Повторный вызов с экспоненциальной задержкой."""
        attempts = []
        for attempt in range(max_retries):
            start = time.time()
            try:
                result = func(*args)
                duration = time.time() - start
                attempts.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "result": result,
                    "duration": duration,
                })
                return {"success": True, "result": result, "attempts": attempts}
            except Exception as e:
                duration = time.time() - start
                delay = base_delay * (2 ** attempt)
                attempts.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e),
                    "duration": duration,
                    "next_delay": delay,
                })
                if attempt < max_retries - 1:
                    time.sleep(delay)

        return {"success": False, "error": "Все попытки исчерпаны", "attempts": attempts}

    # Функция, котораяucceeds на 3-й попытке
    attempt_counter = [0]
    def flaky_func(x):
        """Нестабильная функция."""
        attempt_counter[0] += 1
        if attempt_counter[0] % 3 != 0:
            raise ValueError(f"Ошибка на попытке {attempt_counter[0]}")
        return x * 10

    result = retry_with_backoff(flaky_func, (5,), max_retries=5, base_delay=0.05)
    print(f"  Результат: {result['success']}, значение: {result.get('result')}")
    print(f"  Попыток: {len(result['attempts'])}")
    for a in result["attempts"]:
        status = "OK" if a["success"] else f"ОШИБКА: {a['error']}"
        print(f"    Попытка {a['attempt']}: {status} ({a['duration']*1000:.1f}мс)")

    # --- 4.4 Цепочка инструментов (pipeline) ---
    print("\n--- 4.4 Цепочка инструментов (pipeline) ---")

    class ToolPipeline:
        """Цепочка инструментов для последовательной обработки."""

        def __init__(self):
            self.steps = []
            self.results = []

        def add_step(self, name, func):
            """Добавление шага в цепочку."""
            self.steps.append({"name": name, "func": func})

        def execute(self, initial_input):
            """Выполнение цепочки."""
            current = initial_input
            self.results = []

            for i, step in enumerate(self.steps):
                start = time.time()
                try:
                    current = step["func"](current)
                    duration = time.time() - start
                    self.results.append({
                        "step": i + 1,
                        "name": step["name"],
                        "input": str(current)[:50],
                        "output": str(current)[:50],
                        "success": True,
                        "duration": duration,
                    })
                except Exception as e:
                    duration = time.time() - start
                    self.results.append({
                        "step": i + 1,
                        "name": step["name"],
                        "error": str(e),
                        "success": False,
                        "duration": duration,
                    })
                    return {"success": False, "error": str(e), "results": self.results}

            return {"success": True, "final_output": current, "results": self.results}

    # Цепочка обработки текста
    pipeline = ToolPipeline()
    pipeline.add_step("upper", lambda s: s.upper())
    pipeline.add_step("replace_spaces", lambda s: s.replace(" ", "_"))
    pipeline.add_step("add_prefix", lambda s: f"processed_{s}")
    pipeline.add_step("hash", lambda s: hashlib.md5(s.encode()).hexdigest()[:10])

    initial = "hello world from tool pipeline"
    result = pipeline.execute(initial)

    print(f"  Вход: '{initial}'")
    print(f"  Выход: '{result.get('final_output', 'ОШИБКА')}'")
    print(f"  Шаги:")
    for r in result["results"]:
        status = "OK" if r["success"] else f"ОШИБКА: {r.get('error')}"
        print(f"    {r['step']}. {r['name']}: {status} ({r['duration']*1000:.2f}мс)")

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  УРОК 157: Tool Integration")
    print("  Вызов функций, реестр инструментов, валидация параметров")
    print("=" * 70 + "\n")

    demo_tool_definition()
    demo_tool_registry()
    demo_function_calling()
    demo_tool_execution()

    print("=" * 70)
    print("  Все демонстрации завершены!")
    print("=" * 70)
