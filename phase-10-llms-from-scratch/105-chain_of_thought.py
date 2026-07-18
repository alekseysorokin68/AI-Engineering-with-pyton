"""
Chain-of-Thought Reasoning from Scratch
========================================
Демонстрация основных техник CoT без внешних зависимостей.

Техники:
1. CoT prompting — пошаговое рассуждение
2. Self-consistency — голосование по нескольким путям рассуждения
3. Tree-of-Thought — древовидный поиск лучшего пути
"""

import random
import re
from typing import Optional

random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. CoT Prompting — пошаговое решение задачи
# ─────────────────────────────────────────────────────────────

class CoTSolver:
    """
    Имитация CoT prompting: задача разбивается на явные шаги,
    каждый шаг решается отдельно, затем собирается ответ.
    """

    def __init__(self):
        self.steps_log: list[str] = []

    def solve(self, problem: str) -> dict:
        """Решить задачу с пошаговым рассуждением."""
        self.steps_log = []
        steps = self._decompose(problem)
        results = []
        for i, step in enumerate(steps, 1):
            reasoning = self._reason_step(step, i, len(steps))
            results.append(reasoning)
            self.steps_log.append(f"Шаг {i}: {step} → {reasoning}")

        answer = self._synthesize(results)
        return {"steps": self.steps_log, "answer": answer}

    def _decompose(self, problem: str) -> list[str]:
        """Разложить задачу на подзадачи."""
        if "посчитай" in problem.lower() or "сколько" in problem.lower():
            numbers = [int(x) for x in re.findall(r'\d+', problem)]
            if len(numbers) >= 2:
                return [
                    f"Найти числа: {numbers}",
                    f"Выполнить операцию над {numbers}",
                    "Проверить результат",
                ]
        if "если" in problem.lower():
            return ["Выяснить условие", "Определить следствие", "Сформулировать вывод"]
        return ["Проанализировать задачу", "Найти решение", "Проверить ответ"]

    def _reason_step(self, step: str, idx: int, total: int) -> str:
        """Сгенерировать рассуждение для шага (детерминированная имитация)."""
        if "Найти числа" in step:
            numbers = re.findall(r'\d+', step)
            return f"Числа в задаче: {', '.join(numbers)}"
        if "операцию" in step:
            numbers = [int(x) for x in re.findall(r'\d+', step)]
            if len(numbers) >= 2:
                result = numbers[0] + numbers[1] if len(numbers) == 2 else sum(numbers)
                return f"{numbers} → результат: {result}"
        if "проверить" in step.lower():
            return "Результат выглядит корректно"
        if "условие" in step:
            return "Условие: выполняется"
        if "следствие" in step:
            return "Следствие: если условие истинно, результат = True"
        if "вывод" in step:
            return "Вывод: задача решена"
        return f"Анализ '{step}' завершён"

    def _synthesize(self, results: list[str]) -> str:
        """Собрать финальный ответ из промежуточных."""
        for r in reversed(results):
            match = re.search(r'результат:\s*(\S+)', r)
            if match:
                return f"Ответ: {match.group(1)}"
        return "Ответ: задача решена пошагово"


# ─────────────────────────────────────────────────────────────
# 2. Self-Consistency — голосование по путям рассуждения
# ─────────────────────────────────────────────────────────────

class SelfConsistency:
    """
    Запускает N параллельных CoT-рассуждений (с вариативностью),
    собирает ответы и выбирает наиболее частый (majority vote).
    """

    def __init__(self, n_paths: int = 5):
        self.n_paths = n_paths

    def solve(self, problem: str) -> dict:
        answers = []
        all_traces = []

        for i in range(self.n_paths):
            answer, trace = self._single_cot_path(problem, seed=i)
            answers.append(answer)
            all_traces.append(f"Путь {i+1}: {' → '.join(trace)}")

        counts = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1

        best_answer = max(counts, key=counts.get)
        confidence = counts[best_answer] / self.n_paths

        return {
            "all_answers": answers,
            "vote_counts": counts,
            "winner": best_answer,
            "confidence": confidence,
            "traces": all_traces,
        }

    def _single_cot_path(self, problem: str, seed: int) -> tuple:
        """Один путь CoT-рассуждения с вариативностью."""
        numbers = [int(x) for x in re.findall(r'\d+', problem)]

        # Вариативность: разные операции или порядки
        random.seed(42 + seed)
        variant = random.choice(["direct", "reverse", "grouped"])

        trace = [f"Рассматриваю числа {numbers} (режим: {variant})"]

        if len(numbers) >= 2:
            if variant == "direct":
                result = numbers[0] + numbers[1]
                trace.append(f"Складываю: {numbers[0]} + {numbers[1]} = {result}")
            elif variant == "reverse":
                result = numbers[-1] + numbers[-2]
                trace.append(f"Складываю с конца: {numbers[-1]} + {numbers[-2]} = {result}")
            else:
                mid = len(numbers) // 2
                left = sum(numbers[:mid])
                right = sum(numbers[mid:])
                result = left + right
                trace.append(f"Группирую: {left} + {right} = {result}")
        else:
            result = numbers[0] if numbers else 0
            trace.append(f"Одно число: {result}")

        return result, trace


# ─────────────────────────────────────────────────────────────
# 3. Tree-of-Thought (упрощённо) — древовидный поиск
# ─────────────────────────────────────────────────────────────

class TreeOfThought:
    """
    Генерирует несколько кандидатов-шагов, оценивает каждый,
    выбирает лучшую ветку, повторяет до финального ответа.

    Упрощённая версия: 2 уровня глубины, 3 кандидата на каждом.
    """

    def __init__(self, n_candidates: int = 3, depth: int = 2):
        self.n_candidates = n_candidates
        self.depth = depth

    def solve(self, problem: str) -> dict:
        numbers = [int(x) for x in re.findall(r'\d+', problem)]
        tree_log = []

        # Корень
        root = {"value": numbers, "score": 0, "path": ["начало"]}

        current_level = [root]
        for d in range(self.depth):
            next_level = []
            for node in current_level:
                candidates = self._generate_children(node)
                scored = [(c, self._evaluate(c)) for c in candidates]
                scored.sort(key=lambda x: x[1], reverse=True)
                best = scored[:self.n_candidates]
                for child, score in best:
                    child["score"] = score
                    next_level.append(child)
                    tree_log.append(
                        f"Уровень {d+1}: {node['value']} → {child['value']} "
                        f"(оценка: {score:.2f}, путь: {'→'.join(child['path'])})"
                    )
            current_level = next_level

        # Выбираем лучший лист
        best_leaf = max(current_level, key=lambda n: n["score"])
        final_answer = sum(best_leaf["value"]) if best_leaf["value"] else 0

        return {
            "best_path": best_leaf["path"],
            "best_value": best_leaf["value"],
            "best_score": best_leaf["score"],
            "final_answer": final_answer,
            "tree_log": tree_log,
        }

    def _generate_children(self, node: dict) -> list[dict]:
        """Сгенерировать кандидатов-шагов из текущего узла."""
        values = node["value"]
        children = []
        random.seed(hash(str(values)) % 2**32)

        if len(values) >= 2:
            # Вариант 1: сложить первые два
            new_vals = [values[0] + values[1]] + values[2:]
            children.append({"value": new_vals, "score": 0, "path": node["path"] + [f"сложить {values[0]}+{values[1]}"]})
            # Вариант 2: умножить первые два
            new_vals = [values[0] * values[1]] + values[2:]
            children.append({"value": new_vals, "score": 0, "path": node["path"] + [f"умножить {values[0]}*{values[1]}"]})
            # Вариант 3: вычесть
            new_vals = [values[0] - values[1]] + values[2:]
            children.append({"value": new_vals, "score": 0, "path": node["path"] + [f"вычесть {values[0]}-{values[1]}"]})
        else:
            children.append({"value": values[:], "score": 0, "path": node["path"] + ["без изменений"]})

        return children

    def _evaluate(self, node: dict) -> float:
        """
        Эвристика: предпочитаем положительные числа средней величины.
        (В реальной ToT здесь был бы LLM-valuator.)
        """
        vals = node["value"]
        if not vals:
            return 0.0
        total = sum(vals)
        # Штраф за отрицательные, бонус за умеренно большие
        score = total * 0.1
        if total < 0:
            score -= 5
        if total > 100:
            score -= 3
        return score


# ─────────────────────────────────────────────────────────────
# Демо
# ─────────────────────────────────────────────────────────────

def demo_cot():
    print("=" * 60)
    print("Демо 1: Chain-of-Thought — пошаговое решение")
    print("=" * 60)

    solver = CoTSolver()
    problems = [
        "Посчитай 23 + 47 + 15",
        "Если на столе 5 яблок и добавить 3, сколько будет?",
        "Сколько будет 12 * 3 + 7?",
    ]

    for problem in problems:
        result = solver.solve(problem)
        print(f"\nЗадача: {problem}")
        for step in result["steps"]:
            print(f"  {step}")
        print(f"  ★ {result['answer']}")


def demo_self_consistency():
    print("\n" + "=" * 60)
    print("Демо 2: Self-Consistency — голосование")
    print("=" * 60)

    sc = SelfConsistency(n_paths=5)
    problem = "Посчитай 10 + 20 + 30"
    result = sc.solve(problem)

    print(f"\nЗадача: {problem}")
    print(f"Путей рассуждения: {len(result['all_answers'])}")
    for trace in result["traces"]:
        print(f"  {trace}")
    print(f"\nГолоса: {result['vote_counts']}")
    print(f"Победитель: {result['winner']} "
          f"(уверенность: {result['confidence']:.0%})")


def demo_tot():
    print("\n" + "=" * 60)
    print("Демо 3: Tree-of-Thought — древовидный поиск")
    print("=" * 60)

    tot = TreeOfThought(n_candidates=3, depth=2)
    problem = "Посчитай 5 + 3 + 2"
    result = tot.solve(problem)

    print(f"\nЗадача: {problem}")
    print("Дерево рассуждений:")
    for entry in result["tree_log"]:
        print(f"  {entry}")
    print(f"\nЛучший путь: {' → '.join(result['best_path'])}")
    print(f"Промежуточное значение: {result['best_value']}")
    print(f"Итоговая оценка: {result['best_score']:.2f}")
    print(f"★ Ответ: {result['final_answer']}")


def demo_comparison():
    print("\n" + "=" * 60)
    print("Демо 4: Сравнение — обычный промпт vs CoT vs Self-Consistency")
    print("=" * 60)

    problem = "Посчитай 42 + 17 + 31"

    # Обычный промпт (прямой ответ)
    numbers = [int(x) for x in re.findall(r'\d+', problem)]
    direct_answer = sum(numbers)

    # CoT
    cot = CoTSolver()
    cot_result = cot.solve(problem)

    # Self-Consistency
    sc = SelfConsistency(n_paths=5)
    sc_result = sc.solve(problem)

    print(f"\nЗадача: {problem}")
    print(f"\n{'Метод':<25} {'Ответ':<10} {'Рассуждение'}")
    print("-" * 60)
    print(f"{'Обычный промпт':<25} {direct_answer:<10} нет (прямой ответ)")
    print(f"{'CoT (пошагово)':<25} {cot_result['answer'].split(': ')[-1] if ': ' in cot_result['answer'] else cot_result['answer']:<10} "
          f"{len(cot_result['steps'])} шагов")
    print(f"{'Self-Consistency':<25} {sc_result['winner']:<10} "
          f"голосование ({sc_result['confidence']:.0%} уверенность)")

    print("\nВывод:")
    print("  • Обычный промпт: быстрый, но хрупкий")
    print("  • CoT: прозрачный, легко проверять каждый шаг")
    print("  • Self-Consistency: устойчив к ошибкам одного пути")


# ─────────────────────────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_cot()
    demo_self_consistency()
    demo_tot()
    demo_comparison()
    print("\n" + "=" * 60)
    print("Все демонстрации завершены.")
    print("=" * 60)
