"""
191 — Swarm Algorithms: ACO, PSO, ABC, алгоритм светлячков

Темы:
  1. ACO (оптимизация муравьиным количеством, обновление феромонов, испарение)
  2. PSO (рондо птичье стаи, обновление скорости, вес инерции)
  3. Artificial Bee Colony (назначные, наблюдатели, разведчики пчёлы)
  4. Firefly Algorithm (яркость, привлекательность, движение)

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
# 1. ACO — ОПТИМИЗАЦИЯ МУРАВЬИНЫМ КОЛИЧЕСТВОМ
# =============================================================================

def demo_aco():
    """Демонстрация ACO — феромоны, обновление, решение задачи коммивояжёра."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: ACO — ОПТИМИЗАЦИЯ МУРАВЬИНЫМ КОЛИЧЕСТВОМ")
    print("=" * 70)

    # --- 1.1 Задача коммивояжёра (TSP) ---
    print("\n--- 1.1 Задача коммивояжёра (TSP) ---")
    print("Найти кратчайший маршрут, проходящий через все города ровно по одному разу.\n")

    # Случайные города в 2D
    random.seed(42)
    n_cities = 8
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]

    print(f"  Города ({n_cities} штук):")
    for i, (x, y) in enumerate(cities):
        print(f"    Город {i}: ({x:.1f}, {y:.1f})")

    # Расчёт матрицы расстояний
    def distance_matrix(cities):
        n = len(cities)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = cities[i][0] - cities[j][0]
                    dy = cities[i][1] - cities[j][1]
                    dist[i][j] = math.sqrt(dx * dx + dy * dy)
        return dist

    dist = distance_matrix(cities)

    print("\n  Матрица расстояний (первые 4 города):")
    header = "       " + "".join(f"  {i:>5}" for i in range(4))
    print(header)
    for i in range(4):
        row = f"  {i:>4} " + "".join(f"  {dist[i][j]:>5.1f}" for j in range(4))
        print(row)

    # --- 1.2 Феромоны и эвристика ---
    print("\n--- 1.2 Модель феромонов и эвристики ---")
    print("Вероятность перехода: P(i→j) ∝ τ(i,j)^α × η(i,j)^β")
    print("  τ — феромон, η = 1/d — эвристика (обратная дистанция)")
    print("  α — вес феромона, β — вес эвристики\n")

    # Параметры ACO
    n_ants = 10
    n_iterations = 50
    alpha = 1.0  # вес феромона
    beta = 2.0   # вес эвристики
    rho = 0.5    # коэффициент испарения
    Q = 100.0    # константа феромона

    # Инициализация феромонов
    tau = [[0.1] * n_cities for _ in range(n_cities)]
    eta = [[0.0] * n_cities for _ in range(n_cities)]  # эвристика
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                eta[i][j] = 1.0 / dist[i][j]

    print("  Параметры:")
    print(f"    Муравьёв: {n_ants}, Итераций: {n_iterations}")
    print(f"    α={alpha}, β={beta}, ρ={rho}, Q={Q}")
    print(f"    Начальный феромон: 0.1 на всех рёбрах")

    # --- 1.3 Конструирование маршрутов ---
    print("\n--- 1.3 Конструирование маршрутов ---")

    def construct_route(dist, tau, eta, alpha, beta, n_cities):
        """Муравей строит маршрут."""
        visited = [False] * n_cities
        route = [random.randint(0, n_cities - 1)]
        visited[route[0]] = True

        for _ in range(n_cities - 1):
            current = route[-1]
            # Вычисляем вероятности для каждого непосещённого города
            probs = []
            candidates = []
            for j in range(n_cities):
                if not visited[j]:
                    prob = (tau[current][j] ** alpha) * (eta[current][j] ** beta)
                    probs.append(prob)
                    candidates.append(j)

            # Нормализация
            total = sum(probs)
            probs = [p / total for p in probs]

            # Рулетка: выбор следующего города
            r = random.random()
            cumsum = 0
            for idx, prob in enumerate(probs):
                cumsum += prob
                if r <= cumsum:
                    next_city = candidates[idx]
                    break
            else:
                next_city = candidates[-1]

            route.append(next_city)
            visited[next_city] = True

        return route

    def route_distance(route, dist):
        """Длина маршрута."""
        total = 0
        for i in range(len(route) - 1):
            total += dist[route[i]][route[i + 1]]
        total += dist[route[-1]][route[0]]  # возврат в начало
        return total

    # Пример одного маршрута
    random.seed(42)
    sample_route = construct_route(dist, tau, eta, alpha, beta, n_cities)
    sample_dist = route_distance(sample_route, dist)
    print(f"  Пример маршрута муравья: {' → '.join(map(str, sample_route))} → {sample_route[0]}")
    print(f"  Длина маршрута: {sample_dist:.2f}")

    # --- 1.4 Итеративный ACO ---
    print("\n--- 1.4 Итеративный ACO (обновление феромонов) ---")
    print("Алгоритм:")
    print("  1. Каждый муравей строит маршрут")
    print("  2. Обновляем феромоны: τ(i,j) ← (1-ρ)·τ(i,j) + Σ Δτ_k(i,j)")
    print("  3. Δτ_k = Q / L_k если муравей k прошёл ребро (i,j)\n")

    best_route = None
    best_distance = float('inf')
    history = []

    for iteration in range(n_iterations):
        # Все муравьи строят маршруты
        routes = []
        distances = []
        for _ in range(n_ants):
            route = construct_route(dist, tau, eta, alpha, beta, n_cities)
            d = route_distance(route, dist)
            routes.append(route)
            distances.append(d)

        # Ищем лучший маршрут в этой итерации
        iter_best_idx = min(range(n_ants), key=lambda i: distances[i])
        iter_best_dist = distances[iter_best_idx]

        if iter_best_dist < best_distance:
            best_distance = iter_best_dist
            best_route = routes[iter_best_idx][:]

        history.append(best_distance)

        # Испарение феромонов
        for i in range(n_cities):
            for j in range(n_cities):
                tau[i][j] *= (1 - rho)

        # Добавление нового феромона
        for k in range(n_ants):
            route = routes[k]
            d = distances[k]
            delta_tau = Q / d
            for i in range(len(route) - 1):
                tau[route[i]][route[i + 1]] += delta_tau
                tau[route[i + 1]][route[i]] += delta_tau  # симметричный граф
            # Замыкающее ребро
            tau[route[-1]][route[0]] += delta_tau
            tau[route[0]][route[-1]] += delta_tau

        if iteration in [0, 9, 24, 49]:
            print(f"  Итерация {iteration + 1:>3}: лучшая длина = {best_distance:.2f}")

    print(f"\n  Лучший найденный маршрут: {' → '.join(map(str, best_route))} → {best_route[0]}")
    print(f"  Длина: {best_distance:.2f}")

    # Сравнение с жадным (ближайший сосед)
    random.seed(42)
    start = 0
    greedy_route = [start]
    visited_greedy = {start}
    for _ in range(n_cities - 1):
        current = greedy_route[-1]
        best_next = None
        best_dist_next = float('inf')
        for j in range(n_cities):
            if j not in visited_greedy and dist[current][j] < best_dist_next:
                best_dist_next = dist[current][j]
                best_next = j
        greedy_route.append(best_next)
        visited_greedy.add(best_next)
    greedy_dist = route_distance(greedy_route, dist)

    print(f"\n  Сравнение с жадным алгоритмом (ближайший сосед):")
    print(f"    ACO:  {best_distance:.2f}")
    print(f"    Жадный: {greedy_dist:.2f}")
    print(f"    Улучшение: {(1 - best_distance / greedy_dist) * 100:.1f}%\n")


# =============================================================================
# 2. PSO — РОЙ ЧАСТИЦ
# =============================================================================

def demo_pso():
    """Демонстрация PSO — частицы, скорость, вес инерции, оптимизация функций."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: PSO — РОЙ ЧАСТИЦ")
    print("=" * 70)

    # --- 2.1 Функции для оптимизации ---
    print("\n--- 2.1 Тестовые функции ---")

    def sphere(x):
        """Сферическая функция: f(x) = Σx_i^2. Минимум = 0 в x = (0,...,0)."""
        return sum(xi * xi for xi in x)

    def rastrigin(x):
        """Функция Растригина: f(x) = Σ[x_i^2 - 10cos(2πx_i) + 10]."""
        n = len(x)
        return sum(xi * xi - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x)

    def rosenbrock(x):
        """Функция Розенброка: f(x) = Σ[100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]."""
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return total

    # Тестовые точки
    test_points = [
        [1.0, 1.0],
        [0.5, 0.5],
        [2.0, -1.0],
    ]

    print("  Sphere: f(x) = Σx_i²")
    for pt in test_points:
        print(f"    f({pt}) = {sphere(pt):.2f}")

    print("\n  Rastrigin: f(x) = Σ[x_i² - 10cos(2πx_i) + 10]")
    for pt in test_points:
        print(f"    f({pt}) = {rastrigin(pt):.2f}")

    print("\n  Rosenbrock: f(x) = Σ[100(x_{i+1}-x_i²)² + (1-x_i)²]")
    for pt in test_points:
        print(f"    f({pt}) = {rosenbrock(pt):.2f}")

    # --- 2.2 Алгоритм PSO ---
    print("\n--- 2.2 Алгоритм PSO ---")
    print("Формулы обновления:")
    print("  v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i - x_i(t)) + c2·r2·(gbest - x_i(t))")
    print("  x_i(t+1) = x_i(t) + v_i(t+1)")
    print("  w — вес инерции, c1 — когнитивный, c2 — социальный\n")

    def pso_optimize(func, dim, bounds, n_particles=30, n_iterations=100,
                     w=0.7, c1=1.5, c2=1.5):
        """Оптимизация методом роя частиц."""
        random.seed(42)

        # Инициализация частиц
        positions = []
        velocities = []
        pbest_positions = []
        pbest_values = []

        for _ in range(n_particles):
            pos = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
            vel = [random.uniform(-1, 1) for _ in range(dim)]
            positions.append(pos)
            velocities.append(vel)
            pbest_positions.append(pos[:])
            pbest_values.append(func(pos))

        # Глобальное лучшее
        gbest_idx = min(range(n_particles), key=lambda i: pbest_values[i])
        gbest_pos = pbest_positions[gbest_idx][:]
        gbest_val = pbest_values[gbest_idx]

        history = [gbest_val]

        for iteration in range(n_iterations):
            for i in range(n_particles):
                # Обновление скорости
                r1 = [random.random() for _ in range(dim)]
                r2 = [random.random() for _ in range(dim)]

                for d in range(dim):
                    cognitive = c1 * r1[d] * (pbest_positions[i][d] - positions[i][d])
                    social = c2 * r2[d] * (gbest_pos[d] - positions[i][d])
                    velocities[i][d] = w * velocities[i][d] + cognitive + social

                # Обновление позиции
                for d in range(dim):
                    positions[i][d] += velocities[i][d]
                    # Ограничение границами
                    positions[i][d] = max(bounds[d][0], min(bounds[d][1], positions[i][d]))

                # Оценка
                val = func(positions[i])

                # Обновление pbest
                if val < pbest_values[i]:
                    pbest_values[i] = val
                    pbest_positions[i] = positions[i][:]

            # Обновление gbest
            current_best = min(range(n_particles), key=lambda i: pbest_values[i])
            if pbest_values[current_best] < gbest_val:
                gbest_val = pbest_values[current_best]
                gbest_pos = pbest_positions[current_best][:]

            history.append(gbest_val)

        return gbest_pos, gbest_val, history

    # --- 2.3 Оптимизация Sphere ---
    print("--- 2.3 Оптимизация функции Sphere ---")
    dim = 3
    bounds = [(-10, 10)] * dim
    best_pos, best_val, history = pso_optimize(sphere, dim, bounds, n_particles=20, n_iterations=50)

    print(f"  Параметры: {20} частиц, 50 итераций, dim={dim}")
    print(f"  Найденный минимум: {best_val:.6f}")
    print(f"  Позиция: [{', '.join(f'{x:.4f}' for x in best_pos)}]")
    print(f"  История сходимости:")
    for step in [0, 9, 24, 49]:
        print(f"    Итерация {step + 1:>3}: gbest = {history[step]:.6f}")

    # --- 2.4 Влияние веса инерции ---
    print("\n--- 2.4 Влияние веса инерции w ---")
    print("w высокий → глобальный поиск (exploration)")
    print("w низкий → локальный поиск (exploitation)\n")

    w_values = [0.4, 0.7, 0.9]
    for w_val in w_values:
        best_pos, best_val, _ = pso_optimize(sphere, 3, bounds,
                                              n_particles=20, n_iterations=50, w=w_val)
        print(f"  w={w_val}: минимум = {best_val:.6f}, позиция = [{', '.join(f'{x:.4f}' for x in best_pos)}]")

    # Тест на Rastrigin
    print("\n--- 2.5 Оптимизация Rastrigin (мультимодальная) ---")
    bounds_rast = [(-5.12, 5.12)] * 3
    best_pos, best_val, history = pso_optimize(rastrigin, 3, bounds_rast,
                                                n_particles=30, n_iterations=100)

    print(f"  Параметры: 30 частиц, 100 итераций, dim=3")
    print(f"  Найденный минимум: {best_val:.6f}")
    print(f"  Позиция: [{', '.join(f'{x:.4f}' for x in best_pos)}]")
    print(f"  Ожидаемый глобальный минимум: 0.0 в (0, 0, 0)")

    # Сравнение с不同的 w
    print("\n  Сравнение разных w на Rastrigin:")
    for w_val in [0.4, 0.7, 0.9]:
        _, val, _ = pso_optimize(rastrigin, 3, bounds_rast,
                                  n_particles=30, n_iterations=100, w=w_val)
        print(f"    w={w_val}: минимум = {val:.4f}")
    print()


# =============================================================================
# 3. ABC — ИСКУССТВЕННАЯ ПЧЕЛИНАЯ КОЛОНИЯ
# =============================================================================

def demo_abc():
    """Демонстрация ABC —employed, onlooker, scout пчёлы."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: ABC — ИСКУССТВЕННАЯ ПЧЕЛИНАЯ КОЛОНИЯ")
    print("=" * 70)

    # --- 3.1 Структура колонии ---
    print("\n--- 3.1 Структура пчелиной колонии ---")
    print("Типы пчёл:")
    print("  Employed (назначные): исследуют свою пищу и делятся информацией")
    print("  Onlooker (наблюдатели): выбирают пищу на основе информации employed")
    print("  Scout (разведчики): ищут новую пищу случайно\n")

    # --- 3.2 Функция оценки ---
    print("--- 3.2 Функция оценки (fitness function) ---")

    def evaluate_solution(x):
        """Оцениваем решение (чем меньше, тем лучше — для минимизации)."""
        return sum(xi * xi for xi in x)

    def fitness_function(x):
        """Fitness = 1 / (1 + f(x)) если f(x) >= 0."""
        val = evaluate_solution(x)
        return 1.0 / (1.0 + val)

    # --- 3.3 Алгоритм ABC ---
    print("--- 3.3 Алгоритм ABC ---")
    print("Каждая итерация:")
    print("  1. Employed bees: модифицируют решение и сравнивают (greedy)")
    print("  2. Onlooker bees: выбирают решение пропорционально fitness")
    print("  3. Scout bees: заменяют solutions с limit попыток\n")

    def abc_optimize(func, dim, bounds, n_bees=20, n_iterations=100, limit=15):
        """Оптимизация методом искусственной пчелиной колонии."""
        random.seed(42)

        n_pop = n_bees  # каждая employed пчела = одна пища

        # Инициализация
        food_sources = []
        fitness_vals = []
        trial_counters = []

        for _ in range(n_pop):
            source = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            food_sources.append(source)
            fitness_vals.append(fitness_function(source))
            trial_counters.append(0)

        best_idx = max(range(n_pop), key=lambda i: fitness_vals[i])
        best_source = food_sources[best_idx][:]
        best_fitness = fitness_vals[best_idx]

        history = [best_fitness]

        for iteration in range(n_iterations):
            # === Employed bees phase ===
            for i in range(n_pop):
                # Выбираем соседнее решение
                j = random.randint(0, n_pop - 1)
                while j == i:
                    j = random.randint(0, n_pop - 1)

                # Модифицируем
                phi = random.uniform(-1, 1)
                new_source = food_sources[i][:]
                dim_choice = random.randint(0, dim - 1)
                new_source[dim_choice] = food_sources[i][dim_choice] + phi * (
                    food_sources[i][dim_choice] - food_sources[j][dim_choice]
                )
                # Ограничение границами
                new_source[dim_choice] = max(bounds[dim_choice][0],
                                             min(bounds[dim_choice][1], new_source[dim_choice]))

                new_fitness = fitness_function(new_source)

                # Greedy selection
                if new_fitness > fitness_vals[i]:
                    food_sources[i] = new_source
                    fitness_vals[i] = new_fitness
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # === Onlooker bees phase ===
            # Вычисляем вероятности выбора
            total_fitness = sum(fitness_vals)
            probabilities = [f / total_fitness for f in fitness_vals]

            for _ in range(n_pop):
                # Выбор по рулетке
                r = random.random()
                cumsum = 0
                selected = 0
                for idx, prob in enumerate(probabilities):
                    cumsum += prob
                    if r <= cumsum:
                        selected = idx
                        break

                # Модифицируем выбранное решение
                j = random.randint(0, n_pop - 1)
                while j == selected:
                    j = random.randint(0, n_pop - 1)

                phi = random.uniform(-1, 1)
                new_source = food_sources[selected][:]
                dim_choice = random.randint(0, dim - 1)
                new_source[dim_choice] = food_sources[selected][dim_choice] + phi * (
                    food_sources[selected][dim_choice] - food_sources[j][dim_choice]
                )
                new_source[dim_choice] = max(bounds[dim_choice][0],
                                             min(bounds[dim_choice][1], new_source[dim_choice]))

                new_fitness = fitness_function(new_source)

                if new_fitness > fitness_vals[selected]:
                    food_sources[selected] = new_source
                    fitness_vals[selected] = new_fitness
                    trial_counters[selected] = 0
                else:
                    trial_counters[selected] += 1

            # === Scout bees phase ===
            for i in range(n_pop):
                if trial_counters[i] > limit:
                    # Разведчик заменяет решение
                    food_sources[i] = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
                    fitness_vals[i] = fitness_function(food_sources[i])
                    trial_counters[i] = 0

            # Обновляем лучшее
            current_best = max(range(n_pop), key=lambda i: fitness_vals[i])
            if fitness_vals[current_best] > best_fitness:
                best_fitness = fitness_vals[current_best]
                best_source = food_sources[current_best][:]

            history.append(best_fitness)

            if iteration in [0, 24, 49, 99]:
                best_val = evaluate_solution(best_source)
                print(f"    Итерация {iteration + 1:>3}: лучшее решение = {best_val:.6f}, fitness = {best_fitness:.6f}")

        return best_source, best_fitness, history

    # --- 3.4 Запуск ABC ---
    print("--- 3.4 Оптимизация ABC (Sphere, dim=3) ---")
    bounds = [(-10, 10)] * 3
    best_source, best_fitness, history = abc_optimize(
        evaluate_solution, 3, bounds, n_bees=20, n_iterations=100, limit=15
    )

    best_val = evaluate_solution(best_source)
    print(f"\n  Результат:")
    print(f"    Лучшее решение: [{', '.join(f'{x:.4f}' for x in best_source)}]")
    print(f"    Значение функции: {best_val:.6f}")
    print(f"    Fitness: {best_fitness:.6f}")

    # Сравнение с PSO
    print("\n--- 3.5 Сравнение ABC и PSO ---")

    def pso_quick(func, dim, bounds, n_particles=20, n_iter=100, w=0.7, c1=1.5, c2=1.5):
        """Упрощённый PSO для сравнения."""
        random.seed(42)
        positions = [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(n_particles)]
        velocities = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_particles)]
        pbest = [p[:] for p in positions]
        pbest_val = [func(p) for p in positions]
        gbest_idx = min(range(n_particles), key=lambda i: pbest_val[i])
        gbest = pbest[gbest_idx][:]

        for _ in range(n_iter):
            for i in range(n_particles):
                r1 = [random.random() for _ in range(dim)]
                r2 = [random.random() for _ in range(dim)]
                for d in range(dim):
                    velocities[i][d] = (w * velocities[i][d] +
                                        c1 * r1[d] * (pbest[i][d] - positions[i][d]) +
                                        c2 * r2[d] * (gbest[d] - positions[i][d]))
                    positions[i][d] += velocities[i][d]
                    positions[i][d] = max(bounds[d][0], min(bounds[d][1], positions[i][d]))
                val = func(positions[i])
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = positions[i][:]
            gbest_idx = min(range(n_particles), key=lambda i: pbest_val[i])
            gbest = pbest[gbest_idx][:]

        return func(gbest)

    pso_result = pso_quick(evaluate_solution, 3, bounds)

    print(f"  Sphere (dim=3, 20 частиц/пчёл, 100 итераций):")
    print(f"    ABC:  {best_val:.6f}")
    print(f"    PSO:  {pso_result:.6f}")
    print(f"    Лучший: {'ABC' if best_val < pso_result else 'PSO'}")

    # На Rastrigin
    print("\n--- 3.6 Rastrigin — мультимодальная ---")
    bounds_rast = [(-5.12, 5.12)] * 3
    abc_source, abc_fit, _ = abc_optimize(
        lambda x: sum(xi * xi - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x),
        3, bounds_rast, n_bees=30, n_iterations=150, limit=20
    )
    abc_rast_val = sum(xi * xi - 10 * math.cos(2 * math.pi * xi) + 10 for xi in abc_source)

    print(f"\n  ABC результат на Rastrigin: {abc_rast_val:.4f}")
    print(f"  Ожидаемый минимум: 0.0 в (0, 0, 0)\n")


# =============================================================================
# 4. ALGORITM СВЕТЛЯЧКОВ (FIREFLY ALGORITHM)
# =============================================================================

def demo_firefly():
    """Демонстрация алгоритма светлячков — яркость, привлекательность, движение."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: ALGORITM СВЕТЛЯЧКОВ (FIREFLY ALGORITHM)")
    print("=" * 70)

    # --- 4.1 Концепция ---
    print("\n--- 4.1 Концепция алгоритма ---")
    print("Вдохновлён поведением светлячков ( firefly ):")
    print("  • Яркость пропорциональна качеству решения")
    print("  • Светлячки притягиваются к более ярким")
    print("  • Привлекательность убывает с расстоянием\n")

    # --- 4.2 Математическая модель ---
    print("--- 4.2 Математическая модель ---")
    print("Яркость: I(x) ∝ f(x) (quality of solution)")
    print("Привлекательность: β(r) = β₀ · exp(-γ · r²)")
    print("  β₀ — начальная привлекательность, γ — коэффициент поглощения")
    print("Движение: x_i(t+1) = x_i(t) + β(r_ij)·(x_j - x_i) + α·(rand - 0.5)")
    print("  α — коэффициент случайности\n")

    # --- 4.3 Алгоритм ---
    print("--- 4.3 Алгоритм светлячков ---")

    def firefly_optimize(func, dim, bounds, n_fireflies=25, n_iterations=100,
                         beta0=1.0, gamma=1.0, alpha=0.25):
        """Оптимизация методом светлячков."""
        random.seed(42)

        # Инициализация
        positions = []
        brightness = []

        for _ in range(n_fireflies):
            pos = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            positions.append(pos)
            brightness.append(func(pos))

        # История
        best_history = [min(brightness)]

        for iteration in range(n_iterations):
            # Сортируем по яркости (меньше = лучше для минимизации)
            sorted_idx = sorted(range(n_fireflies), key=lambda i: brightness[i])

            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if brightness[j] < brightness[i]:  # j более яркий (лучший)
                        # Расстояние
                        r_sq = sum((positions[i][d] - positions[j][d]) ** 2 for d in range(dim))

                        # Привлекательность
                        beta = beta0 * math.exp(-gamma * r_sq)

                        # Движение к более яркому
                        for d in range(dim):
                            rand_component = alpha * (random.random() - 0.5)
                            positions[i][d] += beta * (positions[j][d] - positions[i][d]) + rand_component
                            # Ограничение границами
                            positions[i][d] = max(bounds[d][0], min(bounds[d][1], positions[i][d]))

                        brightness[i] = func(positions[i])

            best_history.append(min(brightness))

            if iteration in [0, 24, 49, 99]:
                best_idx = min(range(n_fireflies), key=lambda i: brightness[i])
                print(f"    Итерация {iteration + 1:>3}: лучшая яркость = {brightness[best_idx]:.6f}")

        best_idx = min(range(n_fireflies), key=lambda i: brightness[i])
        return positions[best_idx], brightness[best_idx], best_history

    # --- 4.4 Оптимизация Sphere ---
    print("\n--- 4.4 Оптимизация Sphere (dim=3) ---")
    bounds = [(-10, 10)] * 3
    best_pos, best_bright, history = firefly_optimize(
        lambda x: sum(xi * xi for xi in x),
        3, bounds, n_fireflies=25, n_iterations=100,
        beta0=1.0, gamma=0.01, alpha=0.25
    )

    print(f"\n  Результат:")
    print(f"    Лучшая позиция: [{', '.join(f'{x:.4f}' for x in best_pos)}]")
    print(f"    Яркость (функция): {best_bright:.6f}")
    print(f"    Ожидаемый минимум: 0.0 в (0, 0, 0)")

    # --- 4.5 Влияние параметров ---
    print("\n--- 4.5 Влияние параметров ---")

    test_func = lambda x: sum(xi * xi for xi in x)

    print("  Влияние γ (коэффициент поглощения):")
    for gamma_val in [0.001, 0.01, 0.1, 1.0]:
        _, val, _ = firefly_optimize(test_func, 3, bounds,
                                      n_fireflies=20, n_iterations=50,
                                      beta0=1.0, gamma=gamma_val, alpha=0.25)
        print(f"    γ={gamma_val:<6}: минимум = {val:.6f}")

    print("\n  Влияние β₀ (начальная привлекательность):")
    for beta0_val in [0.5, 1.0, 2.0]:
        _, val, _ = firefly_optimize(test_func, 3, bounds,
                                      n_fireflies=20, n_iterations=50,
                                      beta0=beta0_val, gamma=0.01, alpha=0.25)
        print(f"    β₀={beta0_val:<4}: минимум = {val:.6f}")

    print("\n  Влияние α (случайный компонент):")
    for alpha_val in [0.05, 0.25, 0.5]:
        _, val, _ = firefly_optimize(test_func, 3, bounds,
                                      n_fireflies=20, n_iterations=50,
                                      beta0=1.0, gamma=0.01, alpha=alpha_val)
        print(f"    α={alpha_val:<4}: минимум = {val:.6f}")

    # --- 4.6 Сравнение алгоритмов ---
    print("\n--- 4.6 Сравнение всех алгоритмов (Sphere, dim=3, 100 итераций) ---")

    def pso_final(func, dim, bounds, n_particles=25, n_iter=100):
        random.seed(42)
        positions = [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(n_particles)]
        velocities = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_particles)]
        pbest = [p[:] for p in positions]
        pbest_val = [func(p) for p in positions]
        gbest_idx = min(range(n_particles), key=lambda i: pbest_val[i])
        gbest = pbest[gbest_idx][:]
        for _ in range(n_iter):
            for i in range(n_particles):
                r1 = [random.random() for _ in range(dim)]
                r2 = [random.random() for _ in range(dim)]
                for d in range(dim):
                    velocities[i][d] = (0.7 * velocities[i][d] +
                                        1.5 * r1[d] * (pbest[i][d] - positions[i][d]) +
                                        1.5 * r2[d] * (gbest[d] - positions[i][d]))
                    positions[i][d] += velocities[i][d]
                    positions[i][d] = max(bounds[d][0], min(bounds[d][1], positions[i][d]))
                val = func(positions[i])
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = positions[i][:]
            gbest_idx = min(range(n_particles), key=lambda i: pbest_val[i])
            gbest = pbest[gbest_idx][:]
        return func(gbest)

    def abc_final(func, dim, bounds, n_bees=25, n_iter=100, limit=15):
        random.seed(42)
        n_pop = n_bees
        food = [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(n_pop)]
        fit = [1.0 / (1.0 + func(s)) for s in food]
        trials = [0] * n_pop
        best_i = max(range(n_pop), key=lambda i: fit[i])
        best_val = fit[best_i]
        for _ in range(n_iter):
            for i in range(n_pop):
                j = random.randint(0, n_pop - 1)
                while j == i: j = random.randint(0, n_pop - 1)
                new = food[i][:]
                dc = random.randint(0, dim - 1)
                phi = random.uniform(-1, 1)
                new[dc] += phi * (food[i][dc] - food[j][dc])
                new[dc] = max(bounds[dc][0], min(bounds[dc][1], new[dc]))
                nf = 1.0 / (1.0 + func(new))
                if nf > fit[i]:
                    food[i] = new; fit[i] = nf; trials[i] = 0
                else: trials[i] += 1
            probs = [f / sum(fit) for f in fit]
            for _ in range(n_pop):
                r = random.random(); cs = 0; sel = 0
                for idx, p in enumerate(probs):
                    cs += p
                    if r <= cs: sel = idx; break
                j = random.randint(0, n_pop - 1)
                while j == sel: j = random.randint(0, n_pop - 1)
                new = food[sel][:]
                dc = random.randint(0, dim - 1)
                phi = random.uniform(-1, 1)
                new[dc] += phi * (food[sel][dc] - food[j][dc])
                new[dc] = max(bounds[dc][0], min(bounds[dc][1], new[dc]))
                nf = 1.0 / (1.0 + func(new))
                if nf > fit[sel]: food[sel] = new; fit[sel] = nf; trials[sel] = 0
                else: trials[sel] += 1
            for i in range(n_pop):
                if trials[i] > limit:
                    food[i] = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
                    fit[i] = 1.0 / (1.0 + func(food[i])); trials[i] = 0
            best_i = max(range(n_pop), key=lambda i: fit[i])
            if fit[best_i] > best_val: best_val = fit[best_i]
        best_i = max(range(n_pop), key=lambda i: fit[i])
        return func(food[best_i])

    sphere_func = lambda x: sum(xi * xi for xi in x)

    pso_result = pso_final(sphere_func, 3, bounds)
    abc_result = abc_final(sphere_func, 3, bounds)
    ff_result = best_bright  # уже вычислено выше

    print(f"\n  Результаты (Sphere, dim=3):")
    print(f"    PSO:     {pso_result:.6f}")
    print(f"    ABC:     {abc_result:.6f}")
    print(f"    Firefly: {ff_result:.6f}")

    results = [("PSO", pso_result), ("ABC", abc_result), ("Firefly", ff_result)]
    winner = min(results, key=lambda x: x[1])
    print(f"\n  Лучший алгоритм: {winner[0]} (минимум = {winner[1]:.6f})")
    print("  Замечание: для разных задач и размерностей победитель может отличаться.\n")


# =============================================================================
# ГЛАВНЫЙ БЛОК
# =============================================================================

if __name__ == "__main__":
    demo_aco()
    demo_pso()
    demo_abc()
    demo_firefly()
