"""176 — Multi-Objectives: Парето-оптимальность, компромиссы, ограничения

Темы:
  1. Pareto Optimality (доминирующие решения, Парето-фронт, анализ компромиссов)
  2. Scalarization (взвешенная сумма, epsilon-constraint, целевое программирование)
  3. Constraint Satisfaction (жёсткие и мягкие ограничения, методы штрафов)
  4. Multi-Objective Optimization (NSGA концепции, сохранение разнообразия)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import itertools

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ПАРЕТО-ОПТИМАЛЬНОСТЬ
# ══════════════════════════════════════════════════════════════════════════════

def demo_pareto_optimality():
    """Парето-оптимальность: доминирование, фронт, компромиссы."""
    print("=" * 70)
    print("ДЕМО 1: PARETO OPTIMALITY — Парето-оптимальность")
    print("=" * 70)

    # --- 1a. Доминирование решений ---
    print("\n--- 1a. Доминирование решений ---")
    # Два критерия: минимизация f1, f2
    solutions = [
        {"id": 1, "f1": 2.0, "f2": 8.0},
        {"id": 2, "f1": 3.0, "f2": 5.0},
        {"id": 3, "f1": 4.0, "f2": 3.0},
        {"id": 4, "f1": 5.0, "f2": 2.0},
        {"id": 5, "f1": 3.0, "f2": 6.0},  # доминируется решением 2
        {"id": 6, "f1": 6.0, "f2": 7.0},  # доминируется многими
    ]

    def dominates(a, b):
        """Решение a доминирует b, если a не хуже по всем критериям
        и строго лучше хотя бы по одному."""
        better_or_equal = all(a[k] <= b[k] for k in ["f1", "f2"])
        strictly_better = any(a[k] < b[k] for k in ["f1", "f2"])
        return better_or_equal and strictly_better

    print("Анализ доминирования:")
    for sol in solutions:
        dom_by = [s["id"] for s in solutions if dominates(s, sol) and s["id"] != sol["id"]]
        dom_count = len(dom_by)
        print(f"  Решение {sol['id']}: f1={sol['f1']:.1f}, f2={sol['f2']:.1f}"
              f" -> доминируется {dom_count} решениями{': ' + str(dom_by) if dom_by else ''}")

    # --- 1b. Парето-фронт ---
    print("\n--- 1b. Парето-фронт ---")

    def pareto_front(solutions):
        """Нахождение Парето-фронта: решения, которые не доминируются."""
        front = []
        for s in solutions:
            is_dominated = any(dominates(other, s) for other in solutions if other["id"] != s["id"])
            if not is_dominated:
                front.append(s)
        return front

    front = pareto_front(solutions)
    print("Парето-фронт:")
    for s in sorted(front, key=lambda x: x["f1"]):
        print(f"  Решение {s['id']}: f1={s['f1']:.1f}, f2={s['f2']:.1f}")
    print(f"Размер Парето-фронта: {len(front)} из {len(solutions)}")

    # --- 1c. Анализ компромиссов на Парето-фронте ---
    print("\n--- 1c. Анализ компромиссов ---")
    front_sorted = sorted(front, key=lambda x: x["f1"])
    print("Компромиссы между решениями на Парето-фронте:")
    for i in range(len(front_sorted) - 1):
        a = front_sorted[i]
        b = front_sorted[i + 1]
        tradeoff_f1 = b["f1"] - a["f1"]
        tradeoff_f2 = a["f2"] - b["f2"]
        ratio = tradeoff_f2 / (tradeoff_f1 + 1e-12)
        print(f"  {a['id']} -> {b['id']}: "
              f"f1 +{tradeoff_f1:.1f} (ухудшение), f2 -{tradeoff_f2:.1f} (улучшение), "
              f"соотношение: {ratio:.2f}")

    # --- 1d. Расширение: случайное поколение и Парето-фронт ---
    print("\n--- 1d. Случайное поколение (100 решений) ---")
    random.seed(42)
    big_population = [
        {"id": i, "f1": random.uniform(0, 10), "f2": random.uniform(0, 10)}
        for i in range(100)
    ]
    big_front = pareto_front(big_population)
    big_front_sorted = sorted(big_front, key=lambda x: x["f1"])
    print(f"Парето-фронт из 100 решений: {len(big_front)} решений")
    print("Точки на фронте (каждая 3-я):")
    for s in big_front_sorted[::3]:
        print(f"  ({s['f1']:.2f}, {s['f2']:.2f})", end="")
    print()
    print(f"Диапазон f1: [{min(s['f1'] for s in big_front):.2f}, "
          f"{max(s['f1'] for s in big_front):.2f}]")
    print(f"Диапазон f2: [{min(s['f2'] for s in big_front):.2f}, "
          f"{max(s['f2'] for s in big_front):.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 2. СКАЛЯРИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def demo_scalarization():
    """Скаляризация: взвешенная сумма, epsilon-constraint, целевое программирование."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: SCALARIZATION — скаляризация")
    print("=" * 70)

    # --- 2a. Взвешенная сумма (Weighted Sum) ---
    print("\n--- 2a. Взвешенная сумма (Weighted Sum) ---")
    solutions = [
        {"id": 1, "f1": 1.0, "f2": 9.0},
        {"id": 2, "f1": 3.0, "f2": 5.0},
        {"id": 3, "f1": 5.0, "f2": 3.0},
        {"id": 4, "f1": 9.0, "f2": 1.0},
    ]

    # Разные веса
    weight_sets = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (0.9, 0.1)]
    print(f"{'Веса (w1, w2)':>15} | {'Лучшее решение':>15} | {'F = w1*f1 + w2*f2':>20}")
    print("-" * 60)

    for w1, w2 in weight_sets:
        best = None
        best_F = float("inf")
        for s in solutions:
            F = w1 * s["f1"] + w2 * s["f2"]
            if F < best_F:
                best_F = F
                best = s
        print(f"  ({w1:.1f}, {w2:.1f})    |     {best['id']:>10}    | {best_F:>18.2f}")

    # --- 2b. Epsilon-ограничение ---
    print("\n--- 2b. Epsilon-Constraint метод ---")
    # Минимизация f1 при ограничении f2 <= epsilon
    epsilons = [2.0, 4.0, 6.0, 8.0]
    print(f"{'Epsilon':>8} | {'Допустимые решения':>20} | {'Лучший f1':>10}")
    print("-" * 45)
    for eps in epsilons:
        feasible = [s for s in solutions if s["f2"] <= eps]
        if feasible:
            best = min(feasible, key=lambda s: s["f1"])
            ids = [str(s["id"]) for s in feasible]
            print(f"  {eps:6.1f}   | {', '.join(ids):>20}   | {best['f1']:>8.1f}")
        else:
            print(f"  {eps:6.1f}   | {'нет':>20}   | {'N/A':>10}")

    # --- 2c. Целевое программирование (Goal Programming) ---
    print("\n--- 2c. Целевое программирование ---")
    # Цели: f1* = 2.0, f2* = 4.0
    goal_f1, goal_f2 = 2.0, 4.0
    print(f"Целевые значения: f1*={goal_f1}, f2*={goal_f2}")
    print()

    deviations = []
    for s in solutions:
        d1 = s["f1"] - goal_f1  # положительное = превышение цели
        d2 = s["f2"] - goal_f2
        # Общее отклонение (взвешенное)
        total_dev = abs(d1) + abs(d2)
        deviations.append((s, d1, d2, total_dev))

    deviations.sort(key=lambda x: x[3])
    for s, d1, d2, total in deviations:
        print(f"  Решение {s['id']}: f1={s['f1']:.1f} (д1={d1:+.1f}), "
              f"f2={s['f2']:.1f} (д2={d2:+.1f}), сумма|отклонений|={total:.1f}")

    best_dev = deviations[0]
    print(f"\nЛучшее по целевому программированию: решение {best_dev[0]['id']}")

    # --- 2d. Конверт帕累托 фронта в скалярызацию ---
    print("\n--- 2d. Восстановление Парето-фронта через скаляризацию ---")
    # Показываем, что варьирование весов даёт разные точки фронта
    random.seed(42)
    large_solutions = [
        {"id": i, "f1": random.uniform(1, 10), "f2": random.uniform(1, 10)}
        for i in range(50)
    ]

    recovered = []
    for w1_pct in range(0, 101, 5):
        w1 = w1_pct / 100.0
        w2 = 1.0 - w1
        best = None
        best_F = float("inf")
        for s in large_solutions:
            F = w1 * s["f1"] + w2 * s["f2"]
            if F < best_F:
                best_F = F
                best = s
        if best and best["id"] not in [r["id"] for r in recovered]:
            recovered.append(best)

    print(f"Восстановлено {len(recovered)} уникальных точек из 21 варьирования весов")
    recovered_sorted = sorted(recovered, key=lambda x: x["f1"])
    print("Восстановленный фронт:")
    for s in recovered_sorted[:8]:
        print(f"  ({s['f1']:.2f}, {s['f2']:.2f})", end="")
    if len(recovered_sorted) > 8:
        print(" ...")
    else:
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. ОГРАНИЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def demo_constraint_satisfaction():
    """Ограничения: жёсткие и мягкие, методы штрафов."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: CONSTRAINT SATISFACTION — ограничения")
    print("=" * 70)

    # --- 3a. Жёсткие ограничения (Hard Constraints) ---
    print("\n--- 3a. Жёсткие ограничения (Hard Constraints) ---")
    # Задача: назначение задач агентам
    # Ограничения: каждый агент получает ≤ 2 задачи, каждая задача → 1 агент
    random.seed(42)
    n_agents = 3
    n_tasks = 6
    task_durations = [random.randint(1, 5) for _ in range(n_tasks)]
    agent_capacity = [8, 6, 7]  # максимальная нагрузка
    print(f"Агенты (вместимость): {agent_capacity}")
    print(f"Задачи (длительность): {task_durations}")

    # Жёсткое ограничение: суммарная длительность ≤ вместимость
    assignment = {a: [] for a in range(n_agents)}
    for t in range(n_tasks):
        # Назначаем задачу агенту с наименьшей загрузкой
        loads = [sum(task_durations[j] for j in assignment[a]) for a in range(n_agents)]
        best_agent = min(range(n_agents), key=lambda a: loads[a])
        assignment[best_agent].append(t)

    # Проверка ограничений
    violations = 0
    for a in range(n_agents):
        load = sum(task_durations[t] for t in assignment[a])
        status = "OK" if load <= agent_capacity[a] else "НАРУШЕНИЕ"
        if load > agent_capacity[a]:
            violations += 1
        print(f"  Агент {a}: задачи {assignment[a]}, загрузка={load}/{agent_capacity[a]} [{status}]")
    print(f"Нарушений жёстких ограничений: {violations}")

    # --- 3b. Мягкие ограничения (Soft Constraints) ---
    print("\n--- 3b. Мягкие ограничения (Soft Constraints) ---")
    # Мягкое ограничение: предпочтение равномерного распределения
    def soft_penalty(assignment, task_durations):
        """Штраф за неравномерное распределение."""
        loads = []
        for a in assignment:
            loads.append(sum(task_durations[t] for t in assignment[a]))
        mean_load = sum(loads) / len(loads)
        variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
        return variance  # чем меньше, тем равномернее

    # Попробуем 100 случайных назначений
    best_assignment = None
    best_penalty = float("inf")
    for _ in range(100):
        random.seed(random.randint(0, 10000))
        cand = {a: [] for a in range(n_agents)}
        for t in range(n_tasks):
            cand[random.randint(0, n_agents - 1)].append(t)
        pen = soft_penalty(cand, task_durations)
        if pen < best_penalty:
            best_penalty = pen
            best_assignment = dict(cand)

    loads = [sum(task_durations[t] for t in best_assignment[a]) for a in range(n_agents)]
    print(f"Лучшее назначение (soft penalty={best_penalty:.2f}):")
    for a in range(n_agents):
        print(f"  Агент {a}: задачи {best_assignment[a]}, загрузка={loads[a]}")

    # --- 3c. Метод штрафов (Penalty Method) ---
    print("\n--- 3c. Метод штрафов ---")
    # Оптимизация: x + y -> максимум, при x + y ≤ 5, x ≥ 0, y ≥ 0
    # Штраф: P = max(0, x + y - 5)^2
    random.seed(42)
    x, y = 3.0, 3.0  # начальная точка (нарушает ограничение x+y≤5)
    lr = 0.01
    lambda_pen = 10.0  # коэффициент штрафа

    print(f"Задача: max(x + y), при x + y ≤ 5, x ≥ 0, y ≥ 0")
    print(f"Начальная точка: ({x:.2f}, {y:.2f}), x+y={x+y:.2f}")
    print(f"{'Итерация':>8} | {'x':>6} | {'y':>6} | {'x+y':>6} | {'штраф':>6}")
    print("-" * 45)

    for iteration in range(20):
        # Целевая функция: f = x + y
        # Штраф: P = lambda * max(0, x + y - 5)^2
        violation = max(0, x + y - 5)
        penalty = lambda_pen * violation ** 2

        # Градиент целевой: df/dx = 1, df/dy = 1
        # Градиент штрафа: dP/dx = 2*lambda*violation, dP/dy = 2*lambda*violation
        grad_x = 1 - 2 * lambda_pen * violation
        grad_y = 1 - 2 * lambda_pen * violation

        x += lr * grad_x
        y += lr * grad_y
        x = max(0, x)  # ограничение x ≥ 0
        y = max(0, y)  # ограничение y ≥ 0

        if iteration % 4 == 0 or iteration == 19:
            print(f"  {iteration+1:6d}  | {x:6.2f} | {y:6.2f} | {x+y:6.2f} | {penalty:6.2f}")

    print(f"\nРезультат: x={x:.3f}, y={y:.3f}, x+y={x+y:.3f}")

    # --- 3d. Сравнение методов ограничений ---
    print("\n--- 3d. Сравнение методов ограничений ---")
    methods = {
        "Жёсткие": {"тип": "все или ничего", "гибкость": "низкая",
                     "применение": "логические ограничения"},
        "Мягкие": {"тип": "штраф", "гибкость": "высокая",
                    "применение": "предпочтения"},
        "Штрафные": {"тип": "градиентный штраф", "гибкость": "средняя",
                      "применение": "непрерывная оптимизация"},
    }
    print(f"{'Метод':>12} | {'Тип':>15} | {'Гибкость':>10} | {'Применение'}")
    print("-" * 65)
    for name, props in methods.items():
        print(f"  {name:>10} | {props['тип']:>15} | {props['гибкость']:>10} | {props['применение']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. МНОГОКРИТЕРИАЛЬНАЯ ОПТИМИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def demo_multi_objective_optimization():
    """Многокритериальная оптимизация: NSGA концепции, разнообразие."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: MULTI-OBJECTIVE OPTIMIZATION — NSGA концепции")
    print("=" * 70)

    # --- 4a. Non-dominated Sorting ---
    print("\n--- 4a. Non-dominated Sorting (сортировка по доминированию) ---")
    random.seed(42)
    population = [
        {"id": i, "f1": random.uniform(0, 10), "f2": random.uniform(0, 10)}
        for i in range(20)
    ]

    def dominates(a, b):
        """a доминирует b."""
        return all(a[k] <= b[k] for k in ["f1", "f2"]) and any(a[k] < b[k] for k in ["f1", "f2"])

    def non_dominated_sort(pop):
        """Сортировка по фронтам: фронт 0 — Парето-фронт."""
        fronts = []
        remaining = list(pop)
        while remaining:
            front = []
            for s in remaining:
                if not any(dominates(other, s) for other in remaining if other["id"] != s["id"]):
                    front.append(s)
            fronts.append(front)
            remaining = [s for s in remaining if s not in front]
        return fronts

    fronts = non_dominated_sort(population)
    print(f"Количество фронтов: {len(fronts)}")
    for i, front in enumerate(fronts):
        ids = [s["id"] for s in front]
        print(f"  Фронт {i}: {len(front)} решений (IDs: {ids})")

    # --- 4b. Crowding Distance (расстояние толпины) ---
    print("\n--- 4b. Crowding Distance (расстояние толпины) ---")

    def crowding_distance(front):
        """Расстояние толпины для сохранения разнообразия."""
        n = len(front)
        if n <= 2:
            return {s["id"]: float("inf") for s in front}

        distances = {s["id"]: 0.0 for s in front}
        for obj in ["f1", "f2"]:
            sorted_front = sorted(front, key=lambda s: s[obj])
            distances[sorted_front[0]["id"]] = float("inf")
            distances[sorted_front[-1]["id"]] = float("inf")
            obj_range = sorted_front[-1][obj] - sorted_front[0][obj]
            if obj_range == 0:
                continue
            for i in range(1, n - 1):
                distances[sorted_front[i]["id"]] += (
                    (sorted_front[i+1][obj] - sorted_front[i-1][obj]) / obj_range
                )
        return distances

    for i, front in enumerate(fronts[:3]):
        if len(front) < 2:
            continue
        cd = crowding_distance(front)
        print(f"  Фронт {i} (первые 5):")
        for s in sorted(front, key=lambda s: cd[s["id"]], reverse=True)[:5]:
            print(f"    ID={s['id']:2d}, f1={s['f1']:.2f}, f2={s['f2']:.2f}, "
                  f"crowding={cd[s['id']]:.3f}")

    # --- 4c. Tournament Selection для NSGA ---
    print("\n--- 4c. Tournament Selection (NSGA-style) ---")

    def nsga_tournament(pop, fronts, crowding):
        """Турнирный отбор с приоритетом фронта, затем расстояния толпины."""
        # Создаём словарь: id -> (фронт, crowding_distance)
        rank = {}
        for i, front in enumerate(fronts):
            for s in front:
                rank[s["id"]] = (i, crowding.get(s["id"], 0))

        def compare(a, b):
            """Сравнение: лучший фронт, затем большее расстояние толпины."""
            fa, ca = rank[a["id"]]
            fb, cb = rank[b["id"]]
            if fa != fb:
                return -1 if fa < fb else 1
            return -1 if ca > cb else 1

        # Турнир из 2
        winners = []
        for _ in range(10):
            pair = random.sample(pop, 2)
            if compare(pair[0], pair[1]) < 0:
                winners.append(pair[0])
            else:
                winners.append(pair[1])
        return winners

    front0 = fronts[0]
    cd0 = crowding_distance(front0)
    winners = nsga_tournament(population, fronts, cd0)
    print("10 турнирных победителей:")
    for w in winners:
        print(f"  ID={w['id']:2d}, f1={w['f1']:.2f}, f2={w['f2']:.2f}")

    # --- 4d. Эволюция NSGA-style за 5 поколений ---
    print("\n--- 4d. Эволюция NSGA-style (5 поколений) ---")
    random.seed(42)
    pop = [{"id": i, "f1": random.uniform(0, 10), "f2": random.uniform(0, 10)} for i in range(30)]

    def offspring_crossover(p1, p2):
        """Скрещивание: среднее по f1, f2 с шумом."""
        alpha = random.uniform(0.3, 0.7)
        c1 = {
            "id": -1,
            "f1": alpha * p1["f1"] + (1-alpha) * p2["f1"] + random.gauss(0, 0.5),
            "f2": alpha * p1["f2"] + (1-alpha) * p2["f2"] + random.gauss(0, 0.5),
        }
        c2 = {
            "id": -1,
            "f1": (1-alpha) * p1["f1"] + alpha * p2["f1"] + random.gauss(0, 0.5),
            "f2": (1-alpha) * p1["f2"] + alpha * p2["f2"] + random.gauss(0, 0.5),
        }
        c1["f1"] = max(0, c1["f1"])
        c1["f2"] = max(0, c1["f2"])
        c2["f1"] = max(0, c2["f1"])
        c2["f2"] = max(0, c2["f2"])
        return c1, c2

    next_id = len(pop)

    for gen in range(5):
        fronts = non_dominated_sort(pop)
        cd_all = {}
        for front in fronts:
            cd = crowding_distance(front)
            cd_all.update(cd)

        # Создаём потомков
        offspring = []
        for _ in range(15):
            # Турнирный отбор
            candidates = random.sample(pop, 2)
            r1 = fronts.index(next(f for f in fronts if candidates[0] in f))
            r2 = fronts.index(next(f for f in fronts if candidates[1] in f))
            if r1 < r2 or (r1 == r2 and cd_all.get(candidates[0]["id"], 0) > cd_all.get(candidates[1]["id"], 0)):
                p1 = candidates[0]
            else:
                p1 = candidates[1]

            candidates = random.sample(pop, 2)
            r1 = fronts.index(next(f for f in fronts if candidates[0] in f))
            r2 = fronts.index(next(f for f in fronts if candidates[1] in f))
            if r1 < r2 or (r1 == r2 and cd_all.get(candidates[0]["id"], 0) > cd_all.get(candidates[1]["id"], 0)):
                p2 = candidates[0]
            else:
                p2 = candidates[1]

            c1, c2 = offspring_crossover(p1, p2)
            c1["id"] = next_id
            c2["id"] = next_id + 1
            next_id += 2
            offspring.extend([c1, c2])

        # Объединяем родителей и потомков, берём лучших 30
        combined = pop + offspring
        combined_fronts = non_dominated_sort(combined)
        new_pop = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= 30:
                new_pop.extend(front)
            else:
                # Добавляем с наибольшим crowding distance
                cd = crowding_distance(front)
                remaining = 30 - len(new_pop)
                sorted_front = sorted(front, key=lambda s: cd[s["id"]], reverse=True)
                new_pop.extend(sorted_front[:remaining])
                break

        pop = new_pop
        front0_count = len([s for s in pop if not any(dominates(o, s) for o in pop if o["id"] != s["id"])])
        print(f"  Поколение {gen+1}: популяция={len(pop)}, фронт 0={front0_count}")

    # Финальный Парето-фронт
    final_front = [s for s in pop if not any(dominates(o, s) for o in pop if o["id"] != s["id"])]
    final_front.sort(key=lambda s: s["f1"])
    print(f"\nФинальный Парето-фронт: {len(final_front)} решений")
    print("Точки фронта:")
    for s in final_front:
        print(f"  ({s['f1']:.2f}, {s['f2']:.2f})", end="")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_pareto_optimality()
    demo_scalarization()
    demo_constraint_satisfaction()
    demo_multi_objective_optimization()
    print("\n" + "=" * 70)
    print("Все демо завершены: Multi-Objectives")
    print("=" * 70)
