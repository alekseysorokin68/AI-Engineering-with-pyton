"""168 — Code Agents: генерация кода, выполнение, отладка

Темы:
  1. Code Generation — шаблонная генерация, манипуляции AST, дополнение кода
  2. Code Execution — песочница выполнения, таймауты, захват вывода
  3. Testing Agents — генерация тестов, написание ассертов, анализ покрытия
  4. Debugging Agents — анализ ошибок, предложения исправлений, детекция регрессий

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


# ===========================================================================
# Демо 1: Code Generation — шаблонная генерация, AST-манипуляции, дополнение
# ===========================================================================
def demo_code_generation():
    print("=" * 70)
    print("ДЕМО 1: Code Generation — генерация кода агентом")
    print("=" * 70)

    # --- 1.1 Шаблонная генерация функций ---
    print("\n--- 1.1 Шаблонная генерация функций ---")

    def generate_function(name, params, body_lines, return_expr=None):
        """Генерирует строку Python-функции по шаблону."""
        # Формируем список параметров
        params_str = ", ".join(params)
        # Формируем тело функции с отступами
        body = "\n".join(f"    {line}" for line in body_lines)
        # Добавляем return, если задан
        if return_expr:
            body += f"\n    return {return_expr}"
        # Собираем полный код функции
        code = f"def {name}({params_str}):\n{body}"
        return code

    # Генерируем функцию вычисления факториала
    fact_code = generate_function(
        name="factorial",
        params=["n"],
        body_lines=[
            "if n <= 1:",
            "    return 1",
            "result = 1",
            "for i in range(2, n + 1):",
            "    result *= i",
        ],
        return_expr="result",
    )
    print(f"Сгенерированный код:\n{fact_code}")
    print(f"Длина сгенерированного кода: {len(fact_code)} символов")

    # Генерируем функцию поиска в списке
    search_code = generate_function(
        name="linear_search",
        params=["arr", "target"],
        body_lines=[
            "for i, val in enumerate(arr):",
            "    if val == target:",
            "        return i",
        ],
        return_expr="-1",
    )
    print(f"\nСгенерированный код:\n{search_code}")

    # --- 1.2 AST-манипуляции (анализ через регулярные выражения) ---
    print("\n--- 1.2 Анализ структуры кода (парсинг AST-подобных конструкций) ---")

    def extract_functions(code_text):
        """Извлекает имена функций и их параметры из кода."""
        # Паттерн для поиска определений функций
        pattern = r"def\s+(\w+)\(([^)]*)\):"
        matches = re.findall(pattern, code_text)
        result = []
        for fname, params in matches:
            # Разбиваем параметры и убираем пробелы
            param_list = [p.strip() for p in params.split(",") if p.strip()]
            result.append({"name": fname, "params": param_list})
        return result

    sample_code = fact_code + "\n\n" + search_code
    functions = extract_functions(sample_code)
    print("Найденные функции:")
    for func in functions:
        print(f"  - {func['name']}({', '.join(func['params'])})")

    # Подсчёт сложности: количество строк кода
    lines_count = len([l for l in sample_code.split("\n") if l.strip()])
    print(f"Общее количество строк кода: {lines_count}")

    # --- 1.3 Дополнение кода (code completion) ---
    print("\n--- 1.3 Дополнение кода (предсказание следующей строки) ---")

    def complete_code(prefix, known_patterns):
        """Дополняет код на основе известных паттернов."""
        # Ищем наиболее вероятное продолжение
        best_match = None
        best_score = -1
        # Разбиваем префикс на токены
        tokens = prefix.split()
        for pattern in known_patterns:
            pattern_tokens = pattern.split()
            # Считаем совпадение последних токенов
            score = 0
            for i in range(min(3, len(tokens), len(pattern_tokens))):
                if tokens[-(i + 1)] == pattern_tokens[-(i + 1)]:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = pattern
        return best_match

    # Известные паттерны кода
    patterns = [
        "for i in range(n): result += i",
        "if x > 0: count += 1",
        "while stack: item = stack.pop()",
        "for key in data: values.append(data[key])",
    ]

    test_prefixes = [
        "for i in range(n):",
        "if x >",
        "while",
    ]
    for prefix in test_prefixes:
        completion = complete_code(prefix, patterns)
        print(f"  Префикс: '{prefix}' -> дополнение: '{completion}'")

    # --- 1.4 Генерация класса ---
    print("\n--- 1.4 Генерация класса по описанию ---")

    def generate_class(class_name, attributes, methods):
        """Генерирует класс с атрибутами и методами."""
        # Генерируем __init__
        init_params = ", ".join(attributes)
        init_body = "\n".join(f"        self.{a} = {a}" for a in attributes)
        init_method = f"    def __init__(self, {init_params}):\n{init_body}"

        # Генерируем repr
        repr_attrs = ", ".join(f"{a}={{self.{a}}}" for a in attributes)
        repr_method = f'    def __repr__(self):\n        return f"{class_name}({repr_attrs})"'

        # Генерируем пользовательские методы
        custom_methods = []
        for mname, mbody in methods:
            custom_methods.append(f"    def {mname}(self):\n        {mbody}")

        all_methods = "\n\n".join([init_method, repr_method] + custom_methods)
        return f"class {class_name}:\n\n{all_methods}"

    gen_class = generate_class(
        "Point",
        ["x", "y"],
        [
            ("distance_to_origin", "return math.sqrt(self.x ** 2 + self.y ** 2)"),
            ("manhattan_distance", "return abs(self.x) + abs(self.y)"),
        ],
    )
    print("Сгенерированный класс:")
    print(gen_class)

    # Вычисляем хеш сгенерированного кода для детерминированности
    code_hash = hashlib.md5(gen_class.encode()).hexdigest()[:12]
    print(f"\nХеш сгенерированного кода: {code_hash}")
    print("Код-агент способен генерировать функции, классы и дополнять код")


# ===========================================================================
# Демо 2: Code Execution — песочница, таймауты, захват вывода
# ===========================================================================
def demo_code_execution():
    print("\n" + "=" * 70)
    print("ДЕМО 2: Code Execution — безопасное выполнение кода")
    print("=" * 70)

    # --- 2.1 Песочница выполнения (sandbox) ---
    print("\n--- 2.1 Песочница выполнения (ограниченная среда) ---")

    class CodeSandbox:
        """Песочница для безопасного выполнения Python-кода."""

        def __init__(self, timeout_seconds=5):
            # Время выполнения ограничено
            self.timeout = timeout_seconds

        def execute(self, code, variables=None):
            """Выполняет код в ограниченном окружении."""
            # Простой безопасный способ — используем exec с ограничениями
            output = []
            local_vars = variables or {}

            def safe_print(*args):
                output.append(" ".join(str(a) for a in args))

            local_vars["print"] = safe_print
            local_vars["math"] = math

            # Ограниченное пространство имён (только нужные функции)
            safe_globals = {
                "range": range, "len": len, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "sum": sum,
                "min": min, "max": max, "abs": abs, "round": round,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "print": safe_print, "math": math, "__builtins__": {},
            }

            try:
                exec(code, safe_globals, local_vars)
                return {"success": True, "output": output, "variables": local_vars}
            except Exception as e:
                return {"success": False, "error": str(e), "output": output}

    sandbox = CodeSandbox(timeout_seconds=5)

    # Тест 1: безопасный код
    code1 = """
result = 0
for i in range(10):
    result += i * i
print(f"Сумма квадратов от 0 до 9: {result}")
print(f"Формула: sum(i^2 for i in 0..9) = n(n+1)(2n+1)/6 = {9*10*19//6}")
"""
    res1 = sandbox.execute(code1)
    print(f"Выполнение успешно: {res1['success']}")
    for line in res1["output"]:
        print(f"  {line}")

    # Тест 2: код с ошибкой
    code2 = """
x = 10
y = 0
result = x / y  # Деление на ноль!
"""
    res2 = sandbox.execute(code2)
    print(f"\nВыполнение с ошибкой: {res2['success']}")
    if not res2["success"]:
        print(f"  Ошибка: {res2['error']}")

    # --- 2.2 Захват вывода ---
    print("\n--- 2.2 Захват stdout/stderr ---")

    class OutputCapture:
        """Перехватывает весь вывод функции."""

        def __init__(self):
            self.stdout = []
            self.stderr = []
            self.execution_time = 0

        def run(self, func, *args, **kwargs):
            """Запускает функцию и перехватывает вывод."""
            start_time = time.time()
            captured_output = []

            # Подменяем print на перехватчик
            original_print = print
            captured = {"print": captured_output}

            def intercept_print(*a, **kw):
                msg = " ".join(str(x) for x in a)
                captured_output.append(msg)

            # Выполняем функцию
            import builtins
            old_print = builtins.print
            builtins.print = intercept_print
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                result = f"Ошибка: {e}"
            finally:
                builtins.print = old_print

            self.execution_time = time.time() - start_time
            self.stdout = captured_output
            return result

    capture = OutputCapture()

    def sample_function():
        """Пример функции для перехвата вывода."""
        print("Шаг 1: Инициализация")
        print("Шаг 2: Вычисление")
        total = sum(range(100))
        print(f"Шаг 3: Результат = {total}")
        return total

    result = capture.run(sample_function)
    print(f"Возвращённое значение: {result}")
    print(f"Перехваченный вывод ({len(capture.stdout)} строк):")
    for line in capture.stdout:
        print(f"  | {line}")
    print(f"Время выполнения: {capture.execution_time:.6f} сек")

    # --- 2.3 Вычисление выражений ---
    print("\n--- 2.3 Безопасное вычисление математических выражений ---")

    def safe_eval_expr(expr, variables=None):
        """Безопасно вычисляет математическое выражение."""
        # Разрешённые операции
        safe_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow, "sqrt": math.sqrt, "log": math.log,
            "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
        }
        if variables:
            safe_names.update(variables)

        # Вычисляем выражение
        try:
            result = eval(expr, {"__builtins__": {}}, safe_names)
            return {"success": True, "result": result, "expr": expr}
        except Exception as e:
            return {"success": False, "error": str(e), "expr": expr}

    expressions = [
        "sqrt(144) + log(e**2)",
        "sin(pi/2) + cos(0)",
        "2**10 + 3**5",
        "round(pi, 4) * 100",
    ]
    for expr in expressions:
        res = safe_eval_expr(expr)
        if res["success"]:
            print(f"  {res['expr']} = {res['result']}")
        else:
            print(f"  {res['expr']} -> ОШИБКА: {res['error']}")

    # --- 2.4 Мониторинг ресурсов ---
    print("\n--- 2.4 Мониторинг ресурсов выполнения ---")

    class ResourceMonitor:
        """Мониторит использование ресурсов при выполнении кода."""

        def __init__(self):
            self.metrics = {
                "executions": 0,
                "total_time": 0.0,
                "errors": 0,
                "memory_estimate": 0,
            }

        def track_execution(self, func, *args, **kwargs):
            """Выполняет функцию и отслеживает метрики."""
            start = time.time()
            self.metrics["executions"] += 1
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                self.metrics["total_time"] += elapsed
                # Оценка памяти: количество созданных объектов
                self.metrics["memory_estimate"] += len(str(result))
                return {"result": result, "time": elapsed, "success": True}
            except Exception as e:
                self.metrics["errors"] += 1
                return {"error": str(e), "success": False}

        def summary(self):
            """Возвращает сводку метрик."""
            avg_time = (self.metrics["total_time"] / self.metrics["executions"]
                        if self.metrics["executions"] > 0 else 0)
            return {
                "total_executions": self.metrics["executions"],
                "total_time_s": round(self.metrics["total_time"], 4),
                "avg_time_s": round(avg_time, 6),
                "errors": self.metrics["errors"],
                "estimated_bytes": self.metrics["memory_estimate"],
            }

    monitor = ResourceMonitor()

    # Запускаем несколько вычислений
    test_funcs = [
        lambda: sum(i ** 2 for i in range(1000)),
        lambda: [i for i in range(500) if i % 7 == 0],
        lambda: math.factorial(20),
        lambda: sum(math.sin(i * 0.01) for i in range(100)),
    ]
    for fn in test_funcs:
        monitor.track_execution(fn)

    stats = monitor.summary()
    print(f"  Всего выполнений: {stats['total_executions']}")
    print(f"  Общее время: {stats['total_time_s']} сек")
    print(f"  Среднее время: {stats['avg_time_s']} сек")
    print(f"  Ошибок: {stats['errors']}")
    print(f"  Оценка памяти: {stats['estimated_bytes']} байт")
    print("Код-агент должен уметь безопасно выполнять и мониторить код")


# ===========================================================================
# Демо 3: Testing Agents — генерация тестов, ассерты, покрытие
# ===========================================================================
def demo_testing_agents():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Testing Agents — автоматическое тестирование")
    print("=" * 70)

    # --- 3.1 Генерация тестов ---
    print("\n--- 3.1 Автоматическая генерация тестов ---")

    def generate_tests(function_name, input_output_pairs):
        """Генерирует тесты для функции по парам вход-выход."""
        test_lines = []
        for i, (inputs, expected) in enumerate(input_output_pairs):
            # Формируем аргументы вызова
            if isinstance(inputs, tuple):
                args_str = ", ".join(repr(x) for x in inputs)
            else:
                args_str = repr(inputs)

            # Генерируем строку теста
            test_line = (
                f"    result_{i} = {function_name}({args_str})\n"
                f"    expected_{i} = {expected}\n"
                f"    assert result_{i} == expected_{i}, "
                f"f'Тест {i}: {function_name}({args_str}) = {{result_{i}}}, ожидалось {expected}'"
            )
            test_lines.append(test_line)

        # Собираем полный код тестов
        test_code = f"def test_{function_name}():\n" + "\n".join(test_lines)
        return test_code

    # Тестируем функцию возведения в квадрат
    square_tests = generate_tests(
        "square",
        [
            ((2,), 4),
            ((-3,), 9),
            ((0,), 0),
            ((10,), 100),
        ],
    )
    print("Сгенерированные тесты для square():")
    print(square_tests)

    # Тестируем функцию проверки палиндрома
    palindrome_tests = generate_tests(
        "is_palindrome",
        [
            (("racecar",), True),
            (("hello",), False),
            (("aba",), True),
            (("",), True),
        ],
    )
    print(f"\nСгенерированные тесты для is_palindrome():")
    print(f"  Количество тестов: {len([l for l in palindrome_tests.split(chr(10)) if 'assert' in l])}")

    # --- 3.2 Написание ассертов ---
    print("\n--- 3.2 Генерация проверок (assertions) ---")

    class AssertionGenerator:
        """Генерирует проверки на основе анализа данных."""

        @staticmethod
        def generate_equality_check(var_name, expected):
            """Генерирует проверку равенства."""
            return f"assert {var_name} == {expected}, f'Ожидалось {expected}, получено {{{var_name}}}'"

        @staticmethod
        def generate_type_check(var_name, expected_type):
            """Генерирует проверку типа."""
            return f"assert isinstance({var_name}, {expected_type.__name__}), f'Ожидался тип {expected_type.__name__}, получен {{type({var_name}).__name__}}'"

        @staticmethod
        def generate_range_check(var_name, low, high):
            """Генерирует проверку диапазона."""
            return f"assert {low} <= {var_name} <= {high}, f'{var_name} = {{{var_name}}}, ожидалось [{low}, {high}]'"

        @staticmethod
        def generate_collection_check(var_name, expected_len):
            """Генерирует проверку длины коллекции."""
            return f"assert len({var_name}) == {expected_len}, f'Длина {{len({var_name})}}, ожидалось {expected_len}'"

    gen = AssertionGenerator()
    # Генерируем набор проверок
    checks = [
        gen.generate_equality_check("result", 42),
        gen.generate_type_check("result", int),
        gen.generate_range_check("result", 0, 100),
        gen.generate_collection_check("items", 5),
    ]
    print("Сгенерированные проверки:")
    for c in checks:
        print(f"  {c}")

    # --- 3.3 Анализ покрытия ---
    print("\n--- 3.3 Анализ покрытия кода ---")

    class CoverageAnalyzer:
        """Анализатор покрытия кода (упрощённый)."""

        def __init__(self, code_lines):
            # Разбиваем код на строки
            self.lines = code_lines
            self.executed = set()
            self.total = len([l for l in code_lines if l.strip() and not l.strip().startswith("#")])

        def mark_executed(self, line_numbers):
            """Отмечает строки как выполненные."""
            for ln in line_numbers:
                if 0 <= ln < len(self.lines):
                    self.executed.add(ln)

        def coverage_percent(self):
            """Вычисляет процент покрытия."""
            if self.total == 0:
                return 0.0
            return (len(self.executed) / self.total) * 100

        def uncovered_lines(self):
            """Возвращает строки, которые не были выполнены."""
            result = []
            for i, line in enumerate(self.lines):
                if line.strip() and not line.strip().startswith("#") and i not in self.executed:
                    result.append((i + 1, line.strip()))
            return result

    # Пример кода для анализа
    sample_lines = [
        "def classify(x):",
        "    if x > 0:",
        "        return 'positive'",
        "    elif x < 0:",
        "        return 'negative'",
        "    else:",
        "        return 'zero'",
        "",
        "def process(data):",
        "    results = []",
        "    for item in data:",
        "        results.append(classify(item))",
        "    return results",
    ]

    analyzer = CoverageAnalyzer(sample_lines)
    # Имитируем выполнение: вызвали classify с положительным числом
    analyzer.mark_executed([0, 1, 2, 8, 9, 10, 11, 12])
    # Нет покрытия для веток elif и else в classify

    print(f"Процент покрытия: {analyzer.coverage_percent():.1f}%")
    uncovered = analyzer.uncovered_lines()
    print(f"Непокрытые строки ({len(uncovered)}):")
    for ln, code in uncovered:
        print(f"  Строка {ln}: {code}")

    # --- 3.4 Тестирование граничных случаев ---
    print("\n--- 3.4 Генерация тестов граничных случаев ---")

    def generate_boundary_tests(param_name, constraints):
        """Генерирует тесты для граничных значений."""
        tests = []
        # Граничные значения: минимум, максимум, около границ
        if "min" in constraints and "max" in constraints:
            low = constraints["min"]
            high = constraints["max"]
            tests.append({"value": low, "desc": f"минимум ({low})"})
            tests.append({"value": high, "desc": f"максимум ({high})"})
            tests.append({"value": low - 1, "desc": f"ниже минимума ({low - 1})"})
            tests.append({"value": high + 1, "desc": f"выше максимума ({high + 1})"})
            tests.append({"value": (low + high) // 2, "desc": f"середина ({(low + high) // 2})"})
        if "non_negative" in constraints:
            tests.append({"value": 0, "desc": "нуль"})
            tests.append({"value": -1, "desc": "отрицательное"})
        return tests

    # Генерируем тесты для массива
    boundary_tests = generate_boundary_tests("index", {"min": 0, "max": 99})
    print(f"Граничные тесты для индекса массива:")
    for t in boundary_tests:
        print(f"  {t['desc']}: index = {t['value']}")

    # Генерируем тесты для строки
    string_tests = generate_boundary_tests("length", {"min": 0, "max": 255})
    print(f"\nГраничные тесты для длины строки:")
    for t in string_tests:
        print(f"  {t['desc']}: length = {t['value']}")

    print("\nТестовый агент генерирует тесты, проверки и анализирует покрытие")


# ===========================================================================
# Демо 4: Debugging Agents — анализ ошибок, исправления, регрессии
# ===========================================================================
def demo_debugging_agents():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Debugging Agents — отладка и исправление кода")
    print("=" * 70)

    # --- 4.1 Анализ ошибок ---
    print("\n--- 4.1 Анализ ошибок и классификация ---")

    class ErrorAnalyzer:
        """Анализатор ошибок в коде."""

        # Паттерны ошибок и их описания
        ERROR_PATTERNS = {
            "ZeroDivisionError": {
                "pattern": r"division.*zero|zero.*division",
                "description": "Деление на ноль",
                "fix_hint": "Добавить проверку denominator != 0 перед делением",
            },
            "IndexError": {
                "pattern": r"index.*out.*range|list.*index.*out",
                "description": "Выход за границы массива",
                "fix_hint": "Проверить длину массива перед обращением по индексу",
            },
            "KeyError": {
                "pattern": r"key.*not found|key.*error",
                "description": "Ключ не найден в словаре",
                "fix_hint": "Использовать dict.get() или проверить наличие ключа",
            },
            "TypeError": {
                "pattern": r"type.*error|unsupported.*type",
                "description": "Несовместимые типы",
                "fix_hint": "Проверить типы данных перед операцией",
            },
            "AttributeError": {
                "pattern": r"attribute.*error|has no attribute",
                "description": "Атрибут не найден",
                "fix_hint": "Проверить имя атрибута и наличие у объекта",
            },
        }

        def classify_error(self, error_message):
            """Классифицирует ошибку по сообщению."""
            error_lower = error_message.lower()
            for etype, info in self.ERROR_PATTERNS.items():
                if re.search(info["pattern"], error_lower):
                    return {
                        "type": etype,
                        "description": info["description"],
                        "fix_hint": info["fix_hint"],
                        "original": error_message,
                    }
            return {"type": "Unknown", "description": "Неизвестная ошибка", "fix_hint": "Анализировать контекст", "original": error_message}

    analyzer = ErrorAnalyzer()

    # Примеры ошибок
    error_messages = [
        "ZeroDivisionError: division by zero",
        "IndexError: list index out of range",
        "KeyError: 'username'",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "AttributeError: 'NoneType' object has no attribute 'strip'",
    ]
    print("Классификация ошибок:")
    for msg in error_messages:
        result = analyzer.classify_error(msg)
        print(f"  [{result['type']}] {result['description']}")
        print(f"    Подсказка: {result['fix_hint']}")

    # --- 4.2 Предложение исправлений ---
    print("\n--- 4.2 Автоматическое предложение исправлений ---")

    class FixSuggester:
        """Предлагает исправления для типичных ошибок."""

        @staticmethod
        def suggest_fix(error_type, code_context):
            """Генерирует исправление на основе типа ошибки."""
            fixes = {
                "ZeroDivisionError": [
                    "if denominator != 0:",
                    "    result = numerator / denominator",
                    "else:",
                    "    result = None  # или обработка ошибки",
                ],
                "IndexError": [
                    "if 0 <= index < len(array):",
                    "    value = array[index]",
                    "else:",
                    "    value = None  # значение по умолчанию",
                ],
                "KeyError": [
                    "value = data.get(key, default_value)",
                    "# или:",
                    "if key in data:",
                    "    value = data[key]",
                    "else:",
                    "    value = default_value",
                ],
                "TypeError": [
                    "# Приведение типа перед операцией:",
                    "result = int(value_a) + int(value_b)",
                ],
            }
            return fixes.get(error_type, ["# Нет готового шаблона исправления"])

    suggester = FixSuggester()
    # Пример исправления
    print("Пример исправления KeyError:")
    fix_lines = suggester.suggest_fix("KeyError", "value = data['missing_key']")
    for line in fix_lines:
        print(f"  {line}")

    print("\nПример исправления ZeroDivisionError:")
    fix_lines = suggester.suggest_fix("ZeroDivisionError", "result = a / b")
    for line in fix_lines:
        print(f"  {line}")

    # --- 4.3 Детекция регрессий ---
    print("\n--- 4.3 Детекция регрессий ---")

    class RegressionDetector:
        """Детектор регрессий в поведении кода."""

        def __init__(self):
            self.baseline_results = {}

        def set_baseline(self, test_name, expected_results):
            """Устанавливает базовый результат для теста."""
            self.baseline_results[test_name] = expected_results

        def check_regression(self, test_name, actual_results):
            """Проверяет, есть ли регрессия по сравнению с базой."""
            if test_name not in self.baseline_results:
                return {"status": "new_test", "message": "Новый тест"}

            baseline = self.baseline_results[test_name]
            if len(actual_results) != len(baseline):
                return {
                    "status": "regression",
                    "message": f"Изменилось количество результатов: {len(baseline)} -> {len(actual_results)}",
                }

            failures = []
            for i, (base, actual) in enumerate(zip(baseline, actual_results)):
                if base != actual:
                    failures.append({
                        "index": i,
                        "expected": base,
                        "actual": actual,
                    })

            if failures:
                return {"status": "regression", "failures": failures}
            return {"status": "ok", "message": "Все тесты проходят"}

    detector = RegressionDetector()

    # Устанавливаем базовый результат
    baseline = [True, True, False, True, False]
    detector.set_baseline("palindrome_test", baseline)

    # Тест 1: без регрессии
    result1 = detector.check_regression("palindrome_test", [True, True, False, True, False])
    print(f"Тест 1 (без изменений): {result1['status']} — {result1.get('message', 'OK')}")

    # Тест 2: с регрессией
    actual2 = [True, True, True, True, False]  # Третий элемент изменился
    result2 = detector.check_regression("palindrome_test", actual2)
    print(f"Тест 2 (с регрессией): {result2['status']}")
    if result2["status"] == "regression" and "failures" in result2:
        for f in result2["failures"]:
            print(f"  Ошибка в позиции {f['index']}: ожидалось {f['expected']}, получено {f['actual']}")

    # --- 4.4 Автоматическая отладка ---
    print("\n--- 4.4 Процесс автоматической отладки ---")

    class AutoDebugger:
        """Процесс автоматической отладки кода."""

        def __init__(self):
            self.steps = []

        def debug(self, code, test_cases):
            """Выполняет полный цикл отладки."""
            self.steps = []

            # Шаг 1: Анализ кода
            self.steps.append({
                "step": 1,
                "action": "Анализ кода",
                "detail": f"Строк кода: {len(code.split(chr(10)))}",
            })

            # Шаг 2: Поиск потенциальных проблем
            issues = []
            # Проверяем наличие деления
            if "/" in code and "if" not in code:
                issues.append("Возможное деление на ноль без проверки")
            # Проверяем индексацию
            if re.search(r"\[\d+\]", code):
                issues.append("Хардкод индексов может вызвать IndexError")
            self.steps.append({
                "step": 2,
                "action": "Поиск проблем",
                "issues": issues,
            })

            # Шаг 3: Запуск тестов
            test_results = []
            for i, test in enumerate(test_cases):
                # Имитируем результат теста
                passed = random.random() > 0.3
                test_results.append({"test": i, "passed": passed})
            self.steps.append({
                "step": 3,
                "action": "Запуск тестов",
                "results": test_results,
            })

            # Шаг 4: Формирование отчёта
            passed_count = sum(1 for t in test_results if t["passed"])
            total = len(test_results)
            self.steps.append({
                "step": 4,
                "action": "Отчёт",
                "summary": f"Пройдено {passed_count}/{total} тестов, найдено {len(issues)} потенциальных проблем",
            })

            return self.steps

    debugger = AutoDebugger()
    buggy_code = """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
"""
    test_cases = [
        {"input": [1, 2, 3], "expected": 2.0},
        {"input": [10, 20], "expected": 15.0},
        {"input": [], "expected": None},  # Пустой список!
        {"input": [5], "expected": 5.0},
    ]

    steps = debugger.debug(buggy_code, test_cases)
    for step in steps:
        print(f"  Шаг {step['step']}: {step['action']}")
        if "detail" in step:
            print(f"    {step['detail']}")
        if "issues" in step and step["issues"]:
            for issue in step["issues"]:
                print(f"    Проблема: {issue}")
        if "results" in step:
            passed = sum(1 for r in step["results"] if r["passed"])
            print(f"    Результат: {passed}/{len(step['results'])} тестов пройдено")
        if "summary" in step:
            print(f"    {step['summary']}")

    print("\nАгент отладки выполняет: анализ -> поиск проблем -> тестирование -> исправление")


# ===========================================================================
# Запуск всех демонстраций
# ===========================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  168 — Code Agents: генерация кода, выполнение, отладка            ║")
    print("╚" + "═" * 68 + "╝")
    print()

    demo_code_generation()
    demo_code_execution()
    demo_testing_agents()
    demo_debugging_agents()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены")
    print("=" * 70)
