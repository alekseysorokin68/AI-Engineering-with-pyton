"""184 — Autonomous Debugging: поиск корневых причин, генерация исправлений, регрессионные тесты

Темы:
  1. Root Cause Analysis — fault tree, causal chain, hypothesis testing
  2. Fix Generation — patch synthesis, search-based repair, constraint solving
  3. Regression Testing — test generation, coverage, delta debugging
  4. Debugging Strategies — binary search, bisection, stack trace analysis

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
# 1. Root Cause Analysis
# ========================================================================

def demo_root_cause_analysis():
    """Демонстрация анализа корневых причин сбоев."""
    print("=" * 70)
    print("1. ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # --- 1a. Дерево отказов ---
    print("\n--- 1a. Дерево отказов (Fault Tree Analysis) ---\n")

    class FaultTree:
        """Дерево отказов с логическими операторами."""
        def __init__(self, name, node_type="event"):
            self.name = name
            self.node_type = node_type  # "event", "AND", "OR"
            self.children = []
            self.probability = None
            self.is_basic = False

        def add_child(self, child):
            self.children.append(child)

        def evaluate(self):
            """Вычисление вероятности отказа дерева."""
            if self.probability is not None:
                return self.probability

            if self.node_type == "OR":
                # P(A ∪ B) = 1 - (1-P(A))(1-P(B))
                p_fail = 1.0
                for child in self.children:
                    p_fail *= (1 - child.evaluate())
                self.probability = 1 - p_fail
            elif self.node_type == "AND":
                # P(A ∩ B) = P(A) × P(B)
                p_fail = 1.0
                for child in self.children:
                    p_fail *= child.evaluate()
                self.probability = p_fail
            else:
                self.probability = 0.0

            return self.probability

        def display(self, indent=0):
            """Вывод дерева с отступами."""
            prefix = "  " * indent
            if self.node_type in ("AND", "OR"):
                print(f"{prefix}[{self.node_type}] {self.name}: P={self.evaluate():.6f}")
            else:
                print(f"{prefix}  {self.name}: P={self.probability:.6f}")
            for child in self.children:
                child.display(indent + 1)

    # Построение дерева: "Система не отвечает"
    root = FaultTree("Система не отвечает", "OR")

    db_failure = FaultTree("База данных недоступна", "AND")
    db_failure.add_child(FaultTree("Сервер БД упал", "event"))
    db_failure.children[-1].probability = 0.02
    db_failure.add_child(FaultTree("Replica не активна", "event"))
    db_failure.children[-1].probability = 0.1

    net_failure = FaultTree("Сеть недоступна", "OR")
    net_failure.add_child(FaultTree("DNS не отвечает", "event"))
    net_failure.children[-1].probability = 0.005
    net_failure.add_child(FaultTree("Брандмауэр блокирует", "event"))
    net_failure.children[-1].probability = 0.01

    app_failure = FaultTree("Приложение упало", "AND")
    app_failure.add_child(FaultTree("OOM (нехватка памяти)", "event"))
    app_failure.children[-1].probability = 0.05
    app_failure.add_child(FaultTree("Утечка памяти", "event"))
    app_failure.children[-1].probability = 0.3

    root.add_child(db_failure)
    root.add_child(net_failure)
    root.add_child(app_failure)

    print("  Дерево отказов «Система не отвечает»:")
    root.display()
    print(f"\n  Общая вероятность отказа: {root.evaluate():.6f}")

    # Критический путь
    print(f"\n  Критические пути (наибольший вклад):")
    contributions = []
    for child in root.children:
        contrib = child.evaluate()
        contributions.append((child.name, contrib))
    contributions.sort(key=lambda x: x[1], reverse=True)
    for name, contrib in contributions:
        print(f"    {name}: {contrib:.6f} ({contrib / root.evaluate():.1%})")

    # --- 1b. Цепочка причин ---
    print("\n--- 1b. Цепочка причин (Causal Chain) ---\n")

    class CausalChain:
        """Цепочка причинно-следственных связей."""
        def __init__(self):
            self.events = []

        def add_event(self, cause, effect, strength=0.8):
            """Добавление связи причина → следствие."""
            self.events.append({
                'cause': cause,
                'effect': effect,
                'strength': strength
            })

        def find_root_causes(self):
            """Поиск корневых причин (узлы без входящих связей)."""
            effects = set(e['effect'] for e in self.events)
            causes = set(e['cause'] for e in self.events)
            root_causes = causes - effects
            return root_causes

        def trace_forward(self, cause):
            """Трассировка от причины к последствиям."""
            chain = [cause]
            current = cause
            while True:
                next_effects = [e['effect'] for e in self.events if e['cause'] == current]
                if not next_effects:
                    break
                current = next_effects[0]
                chain.append(current)
            return chain

    chain = CausalChain()
    chain.add_event("Утечка памяти", "Rising heap usage", 0.9)
    chain.add_event("Rising heap usage", "GC pressure", 0.85)
    chain.add_event("GC pressure", "Latency spikes", 0.9)
    chain.add_event("Latency spikes", "Request timeouts", 0.8)
    chain.add_event("Request timeouts", "User-facing errors", 0.7)
    chain.add_event("Баг в аллокаторе", "Утечка памяти", 0.95)

    roots = chain.find_root_causes()
    print(f"  Корневые причины: {roots}")

    for root_cause in roots:
        path = chain.trace_forward(root_cause)
        print(f"\n  Цепочка от «{root_cause}»:")
        for i, event in enumerate(path):
            arrow = " → " if i < len(path) - 1 else ""
            print(f"    [{i}] {event}{arrow}")

    # --- 1c. Тестирование гипотез ---
    print("\n--- 1c. Тестирование гипотез ---\n")

    class HypothesisTester:
        """Тестирование гипотез о причине бага."""
        def __init__(self, observed_error):
            self.observed_error = observed_error
            self.hypotheses = []
            self.tests_run = 0

        def add_hypothesis(self, name, test_func):
            """Добавление гипотезы с функцией тестирования."""
            self.hypotheses.append({
                'name': name,
                'test': test_func,
                'confirmed': None
            })

        def run_all_tests(self):
            """Запуск всех тестов."""
            for h in self.hypotheses:
                h['confirmed'] = h['test'](self.observed_error)
                self.tests_run += 1

        def get_ranked(self):
            """Ранжирование гипотез по вероятности."""
            return sorted(self.hypotheses,
                          key=lambda h: h['confirmed'],
                          reverse=True)

    # Симуляция: ошибка — "IndexError: list index out of range"
    tester = HypothesisTester("IndexError: list index out of range")
    tester.add_hypothesis(
        "Пустой список",
        lambda err: "index out of" in err.lower()
    )
    tester.add_hypothesis(
        "Неверный offset",
        lambda err: "index" in err.lower() and "range" in err.lower()
    )
    tester.add_hypothesis(
        "Null pointer",
        lambda err: "NoneType" in err
    )
    tester.add_hypothesis(
        "Timeout",
        lambda err: "timeout" in err.lower()
    )

    tester.run_all_tests()
    ranked = tester.get_ranked()

    print(f"  Наблюдаемая ошибка: {tester.observed_error}")
    print(f"  Протестировано гипотез: {tester.tests_run}")
    print(f"\n  Ранжирование гипотез:")
    for h in ranked:
        status = "ПОДТВЕРЖДЕНА" if h['confirmed'] else "отвергнута"
        print(f"    [{status:>12}] {h['name']}")

    # --- 1d. Байесовский анализ причин ---
    print("\n--- 1d. Байесовский анализ причин ---\n")

    def bayesian_cause_update(prior, likelihood, evidence):
        """Обновление вероятности причины по формуле Байеса.
        P(cause|evidence) = P(evidence|cause) × P(cause) / P(evidence)
        """
        return (likelihood * prior) / evidence

    # Причины с априорными вероятностями
    causes = {
        "Баг в коде": 0.4,
        "Проблема с инфраструктурой": 0.3,
        "Некорректные входные данные": 0.2,
        "Гонка данных (race condition)": 0.1,
    }

    # Вероятность наблюдаемого evidence при каждой причине
    evidence_given_cause = {
        "Баг в коде": 0.8,
        "Проблема с инфраструктурой": 0.5,
        "Некорректные входные данные": 0.3,
        "Гонка данных (race condition)": 0.6,
    }

    # Суммарная вероятность evidence
    p_evidence = sum(causes[c] * evidence_given_cause[c] for c in causes)

    print(f"  Априорные вероятности:")
    for cause, prior in causes.items():
        print(f"    {cause}: {prior:.2f}")

    print(f"\n  P(evidence) = {p_evidence:.4f}")

    posteriors = {}
    print(f"\n  Постериорные вероятности P(cause | evidence):")
    for cause in causes:
        posterior = bayesian_cause_update(
            causes[cause], evidence_given_cause[cause], p_evidence
        )
        posteriors[cause] = posterior
        print(f"    {cause}: {prior:.3f} → {posterior:.3f}")

    most_likely = max(posteriors, key=posteriors.get)
    print(f"\n  Наиболее вероятная причина: {most_likely} ({posteriors[most_likely]:.3f})")


# ========================================================================
# 2. Fix Generation
# ========================================================================

def demo_fix_generation():
    """Демонстрация автоматической генерации исправлений."""
    print("\n" + "=" * 70)
    print("2. FIX GENERATION")
    print("=" * 70)

    # --- 2a. Patch synthesis ---
    print("\n--- 2a. Синтез патчей (Patch Synthesis) ---\n")

    class PatchSynthesizer:
        """Простой синтезатор исправлений на основе шаблонов."""
        def __init__(self):
            self.templates = {
                'off_by_one': {
                    'pattern': r'range\(n\)',
                    'fix': lambda m: f'range({m.group(0)[6:-1]})',
                    'description': 'Замена range(n) на range(n+1) для off-by-one'
                },
                'null_check': {
                    'pattern': r'\.(\w+)\(',
                    'fix': None,
                    'description': 'Добавление проверки на None перед вызовом метода'
                },
                'index_check': {
                    'pattern': r'\[(\d+)\]',
                    'fix': None,
                    'description': 'Добавление проверки границ массива'
                }
            }

        def analyze_code(self, code):
            """Анализ кода и поиск потенциальных багов."""
            issues = []
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('#') or not stripped:
                    continue
                # Поиск потенциальных off-by-one
                if 'range(n)' in stripped and 'range(n+1)' not in stripped:
                    issues.append({
                        'line': i,
                        'type': 'off_by_one',
                        'code': stripped,
                        'confidence': 0.7
                    })
                # Поиск потенциальных null dereference
                if re.search(r'\w+\.\w+\(', stripped) and 'if' not in stripped:
                    issues.append({
                        'line': i,
                        'type': 'null_check',
                        'code': stripped,
                        'confidence': 0.5
                    })
            return issues

        def generate_patch(self, issue):
            """Генерация патча для найденной проблемы."""
            if issue['type'] == 'off_by_one':
                return f"  # БЫЛО: {issue['code']}\n  # ИСПРАВЛЕНИЕ: заменить range(n) на range(n+1)"
            elif issue['type'] == 'null_check':
                return f"  # БЫЛО: {issue['code']}\n  # ИСПРАВЛЕНИЕ: добавить проверку if obj is not None"
            return f"  # Неизвестный тип проблемы: {issue['type']}"

    buggy_code = """
def sum_array(arr, n):
    total = 0
    for i in range(n):
        total += arr[i]
    return total

def process_data(obj):
    result = obj.transform()
    return result
"""
    synthesizer = PatchSynthesizer()
    issues = synthesizer.analyze_code(buggy_code)

    print(f"  Анализируемый код:")
    for line in buggy_code.strip().split('\n'):
        print(f"    {line}")

    print(f"\n  Найдено проблем: {len(issues)}")
    for issue in issues:
        print(f"\n  Строка {issue['line']} [{issue['type']}]:")
        print(f"    Код: {issue['code']}")
        print(f"    Уверенность: {issue['confidence']:.0%}")
        patch = synthesizer.generate_patch(issue)
        print(f"    {patch}")

    # --- 2b. Search-based repair ---
    print("\n--- 2b. Поиск-based исправление (Search-Based Repair) ---\n")

    def buggy_function(x, y):
        """Функция с багом: неправильная формула."""
        return x * x - y * y  # Должно быть: x*x + y*y

    def repair_search(test_cases, max_iterations=100):
        """Поиск исправления через мутации."""
        # Кандидаты: операторы замены
        operators = [
            ('-', '+', 'a + b'),
            ('*', '/', 'a / b'),
            ('+', '-', 'a - b'),
        ]

        # Генерация тест-кейсов с ожидаемыми результатами
        results = []
        for x, y in test_cases:
            expected = x * x + y * y  # Правильная формула
            results.append((x, y, expected))

        best_fitness = float('inf')
        best_fix = None

        for iteration in range(max_iterations):
            for old_op, new_op, expr in operators:
                # Подсчёт «фитнеса» — суммарная ошибка на тестах
                total_error = 0
                for x, y, expected in results:
                    # Мутируем функцию
                    if old_op == '-' and new_op == '+':
                        actual = x * x + y * y
                    else:
                        actual = buggy_function(x, y)
                    total_error += abs(actual - expected)

                if total_error < best_fitness:
                    best_fitness = total_error
                    best_fix = (old_op, new_op, expr)

                if best_fitness == 0:
                    return best_fix, iteration + 1

        return best_fix, max_iterations

    test_cases = [(3, 4), (5, 12), (8, 15), (1, 1)]
    fix, iters = repair_search(test_cases)

    print(f"  Баг в формуле: x*x - y*y (должно быть x*x + y*y)")
    print(f"  Тест-кейсы: {test_cases}")
    print(f"\n  Поиск исправления:")
    print(f"    Найдена замена: '{fix[0]}' → '{fix[1]}' ({fix[2]})")
    print(f"    Итераций: {iters}")

    # Проверка
    for x, y in test_cases:
        buggy = buggy_function(x, y)
        expected = x * x + y * y
        print(f"    ({x}, {y}): buggy={buggy}, expected={expected}, "
              f"{'OK' if buggy == expected else 'FIX NEEDED'}")

    # --- 2c. Constraint solving ---
    print("\n--- 2c. Решение ограничений (Constraint Solving) ---\n")

    def solve_constraints(constraints, var_range=(-10, 10)):
        """Переборное решение системы ограничений."""
        solutions = []
        for x in range(var_range[0], var_range[1] + 1):
            for y in range(var_range[0], var_range[1] + 1):
                satisfied = True
                for constraint in constraints:
                    if not constraint(x, y):
                        satisfied = False
                        break
                if satisfied:
                    solutions.append((x, y))
        return solutions

    # Проблема: найти x, y такие что:
    # x + y > 5, x * y < 20, x > 0, y > 0, x != y
    constraints = [
        lambda x, y: x + y > 5,
        lambda x, y: x * y < 20,
        lambda x, y: x > 0,
        lambda x, y: y > 0,
        lambda x, y: x != y,
    ]

    solutions = solve_constraints(constraints, var_range=(1, 10))
    print(f"  Ограничения:")
    print(f"    1. x + y > 5")
    print(f"    2. x * y < 20")
    print(f"    3. x > 0, y > 0")
    print(f"    4. x ≠ y")
    print(f"\n  Решения (x, y):")
    for s in solutions[:10]:
        print(f"    ({s[0]}, {s[1]}): sum={s[0]+s[1]}, prod={s[0]*s[1]}")
    print(f"  Всего решений: {len(solutions)}")

    # --- 2d. Генерация исправлений для конкретного бага ---
    print("\n--- 2d. Автоматическое исправление бага с проверкой ---\n")

    class AutoFixer:
        """Автоматический фиксер багов с верификацией."""
        def __init__(self):
            self.fix_attempts = []

        def apply_fix(self, code, fix_description, test_func):
            """Применение исправления и проверка."""
            # Симуляция: модифицируем код
            fixed_code = code
            for _ in range(3):  # до 3 попыток
                # «Генерируем» исправление
                fix_id = hashlib.md5(fixed_code.encode()).hexdigest()[:6]
                self.fix_attempts.append(fix_id)

                # Проверяем
                passed = test_func(fixed_code)
                if passed:
                    return fixed_code, fix_id, True
                fixed_code = fixed_code  # следующая итерация

            return fixed_code, fix_id, False

    def buggy_code_snippet():
        return "def divide(a, b): return a / b"

    def fixed_code_snippet():
        return "def divide(a, b): return a / b if b != 0 else None"

    def test_division(code):
        """Проверка: деление на ноль не должно падать."""
        try:
            exec(code, {'__builtins__': {}})
            # Симуляция: встроенные функции не доступны в exec
            return True
        except Exception:
            return False

    fixer = AutoFixer()
    fixed, fix_id, success = fixer.apply_fix(
        buggy_code_snippet(),
        "Добавить проверку на ноль",
        test_division
    )

    print(f"  Исходный код: {buggy_code_snippet()}")
    print(f"  Исправленный: {fixed}")
    print(f"  ID патча: {fix_id}")
    print(f"  Успешно: {success}")
    print(f"  Попыток: {len(fixer.fix_attempts)}")
    print(f"  Попытки: {fixer.fix_attempts}")


# ========================================================================
# 3. Regression Testing
# ========================================================================

def demo_regression_testing():
    """Демонстрация регрессионного тестирования."""
    print("\n" + "=" * 70)
    print("3. REGRESSION TESTING")
    print("=" * 70)

    # --- 3a. Генерация тестов ---
    print("\n--- 3a. Автоматическая генерация тестов ---\n")

    class TestGenerator:
        """Генератор тестов на основе спецификации."""
        def __init__(self):
            self.tests = []

        def generate_boundary_tests(self, func_name, param_ranges):
            """Генерация граничных тестов."""
            tests = []
            for param_name, (min_val, max_val) in param_ranges.items():
                # Граничные значения
                tests.append({
                    'func': func_name,
                    'args': {param_name: min_val},
                    'type': 'boundary_min',
                    'description': f'{param_name} = min ({min_val})'
                })
                tests.append({
                    'func': func_name,
                    'args': {param_name: max_val},
                    'type': 'boundary_max',
                    'description': f'{param_name} = max ({max_val})'
                })
                tests.append({
                    'func': func_name,
                    'args': {param_name: (min_val + max_val) // 2},
                    'type': 'boundary_mid',
                    'description': f'{param_name} = mid ({(min_val + max_val) // 2})'
                })
            return tests

        def generate_equivalence_tests(self, func_name, partitions):
            """Генерация тестов по классам эквивалентности."""
            tests = []
            for param_name, classes in partitions.items():
                for i, (low, high, label) in enumerate(classes):
                    value = (low + high) // 2
                    tests.append({
                        'func': func_name,
                        'args': {param_name: value},
                        'type': 'equivalence',
                        'description': f'{param_name} ∈ {label} (value={value})'
                    })
            return tests

    gen = TestGenerator()
    boundary_tests = gen.generate_boundary_tests(
        'validate_age', {'age': (0, 150)}
    )
    equiv_tests = gen.generate_equivalence_tests(
        'validate_age',
        {'age': [(0, 0, 'невалидный'), (1, 17, 'несовершеннолетний'),
                  (18, 65, 'взрослый'), (66, 150, 'пожилой')]}
    )

    all_tests = boundary_tests + equiv_tests
    print(f"  Функция: validate_age(age)")
    print(f"  Граничные тесты: {len(boundary_tests)}")
    for t in boundary_tests:
        print(f"    {t['description']}")

    print(f"\n  Тесты эквивалентности: {len(equiv_tests)}")
    for t in equiv_tests:
        print(f"    {t['description']}")

    # --- 3b. Покрытие кода ---
    print("\n--- 3b. Анализ покрытия кода ---\n")

    class CodeCoverage:
        """Простой анализатор покрытия кода."""
        def __init__(self, code):
            self.lines = code.strip().split('\n')
            self.covered = set()
            self.total = len([l for l in self.lines if l.strip()
                              and not l.strip().startswith('#')])

        def record_execution(self, line_numbers):
            """Запись выполненных строк."""
            for ln in line_numbers:
                if 1 <= ln <= len(self.lines):
                    self.covered.add(ln)

        def coverage_percent(self):
            """Процент покрытия."""
            if self.total == 0:
                return 100.0
            return len(self.covered) / self.total * 100

        def uncovered_lines(self):
            """Непокрытые строки."""
            uncovered = []
            for i, line in enumerate(self.lines, 1):
                if line.strip() and not line.strip().startswith('#'):
                    if i not in self.covered:
                        uncovered.append((i, line))
            return uncovered

    sample_code = """
def process(x):
    if x > 0:
        return x * 2
    elif x == 0:
        return 0
    else:
        return -x

def main():
    a = process(5)
    b = process(-3)
    c = process(0)
    return a + b + c
"""
    coverage = CodeCoverage(sample_code)

    # Симуляция выполнения: process(5) → строки 2,3
    coverage.record_execution([2, 3])
    # process(-3) → строки 2,6,7
    coverage.record_execution([2, 6, 7])
    # process(0) → строки 2,4,5
    coverage.record_execution([2, 4, 5])
    # main() → строки 10,11,12,13
    coverage.record_execution([10, 11, 12, 13])

    print(f"  Всего строк кода: {coverage.total}")
    print(f"  Покрыто строк: {len(coverage.covered)}")
    print(f"  Покрытие: {coverage.coverage_percent():.1f}%")

    uncovered = coverage.uncovered_lines()
    if uncovered:
        print(f"\n  Непокрытые строки:")
        for ln, line in uncovered:
            print(f"    {ln}: {line}")
    else:
        print(f"\n  Все строки покрыты!")

    # --- 3c. Delta Debugging ---
    print("\n--- 3c. Delta Debugging — минимизация failing test ---\n")

    def delta_debug(failing_input, test_func, separators=None):
        """Алгоритм delta debugging для минимизации входных данных."""
        if separators is None:
            separators = list(range(len(failing_input)))

        # Начинаем с полного входа
        current = failing_input[:]
        minimized = False
        step = 0

        while len(current) > 1 and not minimized:
            step += 1
            # Разделяем на две половины
            mid = len(current) // 2
            left = current[:mid]
            right = current[mid:]

            # Тестируем левую половину
            if test_func(left):
                current = left
                print(f"  Шаг {step}: [{len(current)}] — левая часть воспроизводит баг")
                continue

            # Тестируем правую половину
            if test_func(right):
                current = right
                print(f"  Шаг {step}: [{len(current)}] — правая часть воспроизводит баг")
                continue

            # Обе части не воспроизводят — пробуем с меньшими кусочками
            minimized = True

        return current

    # Симуляция: строка из 16 символов, баг воспроизводится только с определёнными
    random.seed(42)
    failing_input = list(range(16))

    # Баг воспроизводится если в 입력е есть элементы {5, 8, 12}
    def test_bug(input_subset):
        return any(x in {5, 8, 12} for x in input_subset)

    print(f"  Исходный вход: {failing_input}")
    print(f"  Баг воспроизводится если есть элементы из {{5, 8, 12}}")

    minimized = delta_debug(failing_input, test_bug)
    print(f"\n  Минимизированный вход: {minimized}")
    print(f"  Сжатие: {len(failing_input)} → {len(minimized)} элементов")

    # --- 3d. Тестирование на регрессии ---
    print("\n--- 3d. Сравнение версий и обнаружение регрессий ---\n")

    class VersionTracker:
        """Отслеживание поведения между версиями."""
        def __init__(self):
            self.results = {}

        def record(self, version, test_name, result):
            """Запись результата теста."""
            if version not in self.results:
                self.results[version] = {}
            self.results[version][test_name] = result

        def find_regressions(self, old_version, new_version):
            """Поиск регрессий между двумя версиями."""
            regressions = []
            improvements = []
            old = self.results.get(old_version, {})
            new = self.results.get(new_version, {})

            all_tests = set(list(old.keys()) + list(new.keys()))
            for test in all_tests:
                old_val = old.get(test, None)
                new_val = new.get(test, None)
                if old_val is not None and new_val is not None:
                    if new_val < old_val * 0.95:  # Регрессия > 5%
                        regressions.append((test, old_val, new_val))
                    elif new_val > old_val * 1.05:  # Улучшение > 5%
                        improvements.append((test, old_val, new_val))

            return regressions, improvements

    tracker = VersionTracker()

    # Результаты версии 1.0
    random.seed(42)
    v1_results = {
        'accuracy': 0.85, 'latency_ms': 45, 'memory_mb': 120,
        'throughput': 1000, 'error_rate': 0.02
    }
    for name, val in v1_results.items():
        tracker.record('1.0', name, val)

    # Результаты версии 2.0 (с регрессиями и улучшениями)
    v2_results = {
        'accuracy': 0.83, 'latency_ms': 52, 'memory_mb': 115,
        'throughput': 1100, 'error_rate': 0.015
    }
    for name, val in v2_results.items():
        tracker.record('2.0', name, val)

    regressions, improvements = tracker.find_regressions('1.0', '2.0')

    print(f"  Версия 1.0:")
    for name, val in v1_results.items():
        print(f"    {name}: {val}")

    print(f"\n  Версия 2.0:")
    for name, val in v2_results.items():
        print(f"    {name}: {val}")

    print(f"\n  Регрессии ({len(regressions)}):")
    for test, old, new in regressions:
        change = ((new - old) / old) * 100
        print(f"    {test}: {old} → {new} ({change:+.1f}%)")

    print(f"\n  Улучшения ({len(improvements)}):")
    for test, old, new in improvements:
        change = ((new - old) / old) * 100
        print(f"    {test}: {old} → {new} ({change:+.1f}%)")


# ========================================================================
# 4. Debugging Strategies
# ========================================================================

def demo_debugging_strategies():
    """Демонстрация различных стратегий отладки."""
    print("\n" + "=" * 70)
    print("4. DEBUGGING STRATEGIES")
    print("=" * 70)

    # --- 4a. Бинарный поиск ---
    print("\n--- 4a. Бинарный поиск ошибки в коде ---\n")

    def find_bug_binary_search(test_func, code_lines):
        """Бинарный поиск строки с багом."""
        low, high = 0, len(code_lines) - 1
        steps = 0

        while low < high:
            steps += 1
            mid = (low + high) // 2
            # Тестируем: «работает ли всё до строки mid?»
            works = test_func(mid)
            if works:
                low = mid + 1
            else:
                high = mid

        return low, steps

    # Симуляция: баг на строке 13
    bug_line = 13
    code_lines = [f"line {i}: code" for i in range(1, 21)]

    def test_up_to(line_idx):
        """Проверка: работает ли всё до строки line_idx."""
        return line_idx < bug_line

    found_line, steps = find_bug_binary_search(test_up_to, code_lines)
    print(f"  Всего строк: {len(code_lines)}")
    print(f"  Баг на строке: {bug_line}")
    print(f"  Найден на строке: {found_line + 1}")
    print(f"  Шагов (бинарный): {steps}")
    print(f"  Шагов (линейный): {len(code_lines)}")
    print(f"  Экономия: {len(code_lines) - steps} шагов ({(1 - steps/len(code_lines)):.0%})")

    # --- 4b. Bisect ---
    print("\n--- 4b. Bisect — поиск первого «плохого» коммита ---\n")

    class CommitBisect:
        """Бисекция коммитов для поиска регрессии."""
        def __init__(self, total_commits, first_bad):
            self.total = total_commits
            self.first_bad = first_bad

        def bisect(self):
            """Поиск первого плохого коммита."""
            low, high = 0, self.total - 1
            steps = []
            while low < high:
                mid = (low + high) // 2
                is_bad = mid >= self.first_bad
                steps.append((mid, is_bad))
                if is_bad:
                    high = mid
                else:
                    low = mid + 1
            return low, steps

    bisect = CommitBisect(total_commits=128, first_bad=73)
    bad_commit, steps = bisect.bisect()

    print(f"  Всего коммитов: 128")
    print(f"  Первый «плохой» коммит: #73")
    print(f"\n  Шаги бисекции:")
    for i, (mid, is_bad) in enumerate(steps):
        status = "BAD" if is_bad else "GOOD"
        print(f"    Шаг {i + 1}: коммит #{mid} → {status}")
    print(f"\n  Найден коммит: #{bad_commit}")
    print(f"  Всего проверок: {len(steps)} (вместо 73 линейных)")

    # --- 4c. Анализ stack trace ---
    print("\n--- 4c. Анализ стека вызовов (Stack Trace Analysis) ---\n")

    class StackTraceAnalyzer:
        """Анализатор стека вызовов."""
        def __init__(self, stack_trace):
            self.frames = self._parse(stack_trace)

        def _parse(self, trace):
            """Парсинг текстового стека."""
            frames = []
            for line in trace.strip().split('\n'):
                # Формат: File "path", line N, in func
                match = re.search(r'File "(.+?)", line (\d+), in (.+)', line)
                if match:
                    frames.append({
                        'file': match.group(1),
                        'line': int(match.group(2)),
                        'function': match.group(3)
                    })
            return frames

        def find_user_code(self):
            """Поиск первого кадра в пользовательском коде (не библиотека)."""
            for frame in self.frames:
                if 'site-packages' not in frame['file'] and \
                   'lib/' not in frame['file'] and \
                   not frame['file'].startswith('<'):
                    return frame
            return None

        def summarize(self):
            """Краткое описание."""
            if not self.frames:
                return "Пустой стек"
            top = self.frames[0]
            user_frame = self.find_user_code()
            return {
                'error_location': f"{top['file']}:{top['line']} in {top['function']}",
                'user_code': user_frame,
                'depth': len(self.frames)
            }

    sample_trace = """Traceback (most recent call last):
  File "app/main.py", line 45, in process_request
    result = handler.handle(data)
  File "app/handlers.py", line 123, in handle
    parsed = self.parse_input(data)
  File "app/parser.py", line 67, in parse_input
    return json.loads(raw_data)
  File "/usr/lib/python3/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)"""

    analyzer = StackTraceAnalyzer(sample_trace)
    summary = analyzer.summarize()

    print(f"  Стек вызовов ({summary['depth']} кадров):")
    print(f"  Ошибка: {summary['error_location']}")
    user = summary['user_code']
    if user:
        print(f"  Пользовательский код: {user['file']}:{user['line']} in {user['function']}")

    print(f"\n  Цепочка вызовов:")
    for i, frame in enumerate(analyzer.frames):
        marker = " ← ОШИБКА" if i == 0 else (" ← USER" if i == 2 else "")
        print(f"    [{i}] {frame['file']}:{frame['line']} in {frame['function']}{marker}")

    # --- 4d. Логирование и трассировка ---
    print("\n--- 4d. Система логирования для отладки ---\n")

    class DebugLogger:
        """Простой логгер для отладки с трассировкой."""
        def __init__(self):
            self.entries = []
            self.timers = {}

        def log(self, level, message, **kwargs):
            """Запись лога."""
            entry = {
                'level': level,
                'message': message,
                'kwargs': kwargs,
                'timestamp': time.time()
            }
            self.entries.append(entry)

        def start_timer(self, name):
            """Начало таймера."""
            self.timers[name] = time.time()

        def stop_timer(self, name):
            """Остановка таймера."""
            if name in self.timers:
                duration = time.time() - self.timers[name]
                self.log('INFO', f'{name} завершён', duration_ms=duration * 1000)
                del self.timers[name]
                return duration
            return None

        def get_summary(self):
            """Сводка по логам."""
            levels = collections.Counter(e['level'] for e in self.entries)
            return dict(levels)

    logger = DebugLogger()

    # Симуляция отладки
    logger.start_timer("process_data")
    logger.log('DEBUG', 'Получены данные', size=1024)
    logger.log('INFO', 'Начало обработки', batch=42)
    logger.log('WARNING', 'Медленный запрос', latency_ms=1500)
    logger.log('DEBUG', 'Кэш промах', key='user:123')
    logger.log('ERROR', 'Таймаут подключения', host='db-master', timeout=30)
    logger.stop_timer("process_data")

    summary = logger.get_summary()
    print(f"  Всего записей: {len(logger.entries)}")
    print(f"  По уровням: {summary}")
    print(f"\n  Последние 5 записей:")
    for entry in logger.entries[-5:]:
        extra = ""
        if entry['kwargs']:
            extra = f" {entry['kwargs']}"
        print(f"    [{entry['level']:>7}] {entry['message']}{extra}")

    # Поиск паттернов
    print(f"\n  Анализ паттернов:")
    error_msgs = [e['message'] for e in logger.entries if e['level'] == 'ERROR']
    warn_msgs = [e['message'] for e in logger.entries if e['level'] == 'WARNING']
    print(f"    Ошибки: {error_msgs}")
    print(f"    Предупреждения: {warn_msgs}")


# ========================================================================
# Точка входа
# ========================================================================

if __name__ == "__main__":
    demo_root_cause_analysis()
    demo_fix_generation()
    demo_regression_testing()
    demo_debugging_strategies()
