"""182 — Swarm Intelligence: стигмергия, коллективное поведение, эмергентные паттерны

Темы:
  1. Stigmergy — изменение среды, феромонные следы, непрямая координация
  2. Ant Colony Optimization — обновление феромонов, испарение, поиск пути
  3. Particle Swarm Optimization — обновление скорости, личный/глобальный лучший
  4. Flocking & Swarming — правила разделения, выравнивания, слияния

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


# ──────────────────────────── 1. Stigmergy ────────────────────────────

def demo_stigmergy():
    """Изменение среды, феромонные следы, непрямая координация."""
    print("=" * 70)
    print("DEMO 1 — Stigmergy")
    print("=" * 70)

    # --- 1a. Что такое стигмергия ---
    print("\n--- 1a. Определение стигмергии ---")
    print("  Стигмергия — механизм координации, при котором агенты")
    print("  изменяют среду, а изменения среды влияют на поведение")
    print("  других агентов (непрямая коммуникация).")
    print()
    print("  Примеры:")
    print("  • Муравьи оставляют феромон → другие муравьи следуют за ним")
    print("  • Термиты складывают грязь → грязь привлекает больше термитов")
    print("  • Люди прокладывают тропки → тропки привлекают больше людей")

    # --- 1b. Простая модель стигмергии на сетке ---
    print("\n--- 1b. Стигмаргическая сетка (простая модель) ---")
    random.seed(42)
    grid_size = 15
    # Инициализируем сетку феромонов нулями
    pheromone_grid = [[0.0] * grid_size for _ in range(grid_size)]

    n_ants = 5
    n_steps = 30
    deposit_amount = 0.5
    evaporation_rate = 0.1

    # Начальные позиции муравьёв
    ant_positions = [(random.randint(0, grid_size - 1),
                      random.randint(0, grid_size - 1)) for _ in range(n_ants)]

    # Имитация перемещения муравьёв с депонированием феромонов
    for step in range(n_steps):
        # Испарение
        for r in range(grid_size):
            for c in range(grid_size):
                pheromone_grid[r][c] *= (1 - evaporation_rate)

        # Муравьи перемещаются и депонируют феромон
        for i in range(n_ants):
            r, c = ant_positions[i]
            # Депонируем феромон
            pheromone_grid[r][c] += deposit_amount
            # Случайное перемещение ( biases toward high-pheromone areas)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            # Вычисляем веса для каждого направления
            weights = []
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    weights.append(pheromone_grid[nr][nc] + 0.1)
                else:
                    weights.append(0)

            # Выбор направления
            total = sum(weights)
            if total > 0:
                probs = [w / total for w in weights]
                move_idx = random.choices(range(len(moves)), weights=probs)[0]
                dr, dc = moves[move_idx]
                ant_positions[i] = ((r + dr) % grid_size, (c + dc) % grid_size)

    # Выводим финальную карту феромонов
    max_pheromone = max(max(row) for row in pheromone_grid)
    symbols = " ·∘○●◉█"

    print(f"\n  Сетка {grid_size}×{grid_size}, {n_ants} муравьёв, {n_steps} шагов")
    print(f"  Испарение: {evaporation_rate}, Депонирование: {deposit_amount}")
    print(f"\n  Карта феромонов (чем темнее — тем больше феромона):")
    print("  +" + "-" * grid_size + "+")
    for r in range(grid_size):
        print("  |", end="")
        for c in range(grid_size):
            norm = pheromone_grid[r][c] / max_pheromone if max_pheromone > 0 else 0
            idx = min(int(norm * (len(symbols) - 1)), len(symbols) - 1)
            print(symbols[idx], end="")
        print("|")
    print("  +" + "-" * grid_size + "+")

    # --- 1c. Формирование маршрута через стигмергию ---
    print("\n--- 1c. Нахождение кратчайшего пути через стигмергию ---")
    # Простая задача: найти путь из точки A в точку B
    start = (0, 0)
    end = (4, 4)
    n_ants_path = 10
    n_iterations = 50
    grid_path_size = 5
    pheromone_path = [[0.1] * grid_path_size for _ in range(grid_path_size)]

    def find_path(start, end, pheromone, alpha=1.0):
        """Муравей ищет путь, следуя феромонам."""
        path = [start]
        current = start
        visited = {start}

        for _ in range(grid_path_size * 2):  # Максимум шагов
            if current == end:
                break
            r, c = current
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < grid_path_size and 0 <= nc < grid_path_size
                        and (nr, nc) not in visited):
                    candidates.append(((nr, nc), pheromone[nr][nc]))

            if not candidates:
                break

            # Выбор следующей клетки (пропорционально феромону)
            total = sum(p ** alpha for _, p in candidates)
            probs = [(p ** alpha) / total for _, p in candidates]
            idx = random.choices(range(len(candidates)), weights=probs)[0]
            next_pos = candidates[idx][0]
            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos

        return path

    best_path_length = float('inf')
    best_path = None

    for iteration in range(n_iterations):
        paths = []
        for _ in range(n_ants_path):
            path = find_path(start, end, pheromone_path)
            paths.append(path)

        # Обновление феромонов
        # Испарение
        for r in range(grid_path_size):
            for c in range(grid_path_size):
                pheromone_path[r][c] *= 0.8

        # Депонирование пропорционально качеству пути
        for path in paths:
            if path[-1] == end:
                path_len = len(path) - 1
                deposit = 1.0 / path_len
                for r, c in path:
                    pheromone_path[r][c] += deposit

                if path_len < best_path_length:
                    best_path_length = path_len
                    best_path = path

    print(f"  Сетка: {grid_path_size}×{grid_path_size}")
    print(f"  Начало: {start}, Конец: {end}")
    print(f"  Итераций: {n_iterations}, Муравьёв на итерацию: {n_ants_path}")
    print(f"  Лучший путь: {best_path}")
    print(f"  Длина пути: {best_path_length} шагов")

    # --- 1d. Сравнение с случайным поиском ---
    print("\n--- 1d. Сравнение: стигмаргический vs случайный поиск ---")
    random.seed(42)

    def random_path(start, end, max_steps=20):
        """Случайный поиск пути."""
        path = [start]
        current = start
        for _ in range(max_steps):
            if current == end:
                break
            r, c = current
            moves = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_path_size and 0 <= nc < grid_path_size:
                    moves.append((nr, nc))
            if moves:
                current = random.choice(moves)
                path.append(current)
        return path

    n_trials = 1000
    random_lengths = []
    for _ in range(n_trials):
        p = random_path(start, end)
        if p[-1] == end:
            random_lengths.append(len(p) - 1)

    avg_random = sum(random_lengths) / len(random_lengths) if random_lengths else float('inf')
    success_rate = len(random_lengths) / n_trials

    print(f"  Случайный поиск ({n_trials} попыток):")
    print(f"    Успешность: {success_rate:.1%}")
    print(f"    Средняя длина пути: {avg_random:.2f}")
    print(f"  Стигмаргический поиск:")
    print(f"    Длина лучшего пути: {best_path_length}")
    print(f"  → Стигмаргия находит {'короче' if best_path_length < avg_random else 'длиннее'} путь "
          f"(разница: {abs(best_path_length - avg_random):.2f})")


# ──────────────────────────── 2. Ant Colony Optimization ────────────────────────────

def demo_ant_colony_optimization():
    """Обновление феромонов, испарение, поиск пути в графе."""
    print("\n\n" + "=" * 70)
    print("DEMO 2 — Ant Colony Optimization (ACO)")
    print("=" * 70)

    # --- 2a. Граф задачи коммивояжёра ---
    print("\n--- 2a. Граф: задача коммивояжёра (TSP) ---")
    # 8 городов со случайными координатами
    random.seed(42)
    n_cities = 8
    cities = [(random.uniform(0, 100), random.uniform(0, 100))
              for _ in range(n_cities)]

    # Матрица расстояний
    def distance(c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    dist_matrix = [[distance(cities[i], cities[j])
                    for j in range(n_cities)]
                   for i in range(n_cities)]

    print(f"  Города ({n_cities} шт.):")
    for i, (x, y) in enumerate(cities):
        print(f"    Город {i}: ({x:.1f}, {y:.1f})")

    print(f"\n  Матрица расстояний (первые 4 города):")
    print(f"  {'':>8}", end="")
    for j in range(4):
        print(f"  {'Город ' + str(j):>10}", end="")
    print()
    for i in range(4):
        print(f"  Город {i}", end="")
        for j in range(4):
            print(f"{dist_matrix[i][j]:>10.1f}", end="")
        print()

    # --- 2b. Параметры ACO ---
    print("\n--- 2b. Параметры ACO ---")
    n_ants = 10
    n_iterations = 100
    alpha = 1.0    # Влияние феромона
    beta = 2.0     # Влияние расстояния (эвристика)
    rho = 0.5      # Скорость испарения
    Q = 100         # Константа депонирования
    initial_pheromone = 0.1

    print(f"  n_ants = {n_ants}")
    print(f"  n_iterations = {n_iterations}")
    print(f"  alpha (феромон) = {alpha}")
    print(f"  beta (эвристика) = {beta}")
    print(f"  rho (испарение) = {rho}")
    print(f"  Q (депонирование) = {Q}")

    # --- 2c. ACO: основной цикл ---
    print("\n--- 2c. Запуск ACO ---")
    pheromone = [[initial_pheromone] * n_cities for _ in range(n_cities)]
    best_tour = None
    best_length = float('inf')
    history = []

    for iteration in range(n_iterations):
        tours = []
        tour_lengths = []

        # Каждый муравей строит маршрут
        for _ in range(n_ants):
            visited = [False] * n_cities
            tour = [random.randint(0, n_cities - 1)]
            visited[tour[0]] = True

            for _ in range(n_cities - 1):
                current = tour[-1]
                # Вычисляем вероятности для каждого непосещённого города
                probs = []
                candidates = []
                for j in range(n_cities):
                    if not visited[j]:
                        tau = pheromone[current][j] ** alpha
                        eta = (1.0 / (dist_matrix[current][j] + 1e-10)) ** beta
                        probs.append(tau * eta)
                        candidates.append(j)

                if not candidates:
                    break

                # Нормализация и выбор
                total = sum(probs)
                probs = [p / total for p in probs]
                next_city = random.choices(candidates, weights=probs)[0]
                tour.append(next_city)
                visited[next_city] = True

            # Замыкаем маршрут (возврат в начальный город)
            tour.append(tour[0])

            # Вычисляем длину маршрута
            length = sum(dist_matrix[tour[i]][tour[i + 1]]
                         for i in range(len(tour) - 1))
            tours.append(tour)
            tour_lengths.append(length)

            if length < best_length:
                best_length = length
                best_tour = tour[:]

        # Испарение феромонов
        for i in range(n_cities):
            for j in range(n_cities):
                pheromone[i][j] *= (1 - rho)

        # Депонирование феромонов
        for tour, length in zip(tours, tour_lengths):
            deposit = Q / length
            for i in range(len(tour) - 1):
                pheromone[tour[i]][tour[i + 1]] += deposit
                pheromone[tour[i + 1]][tour[i]] += deposit

        history.append(best_length)

    print(f"  Найден лучший маршрут: {' → '.join(str(c) for c in best_tour)}")
    print(f"  Длина маршрута: {best_length:.2f}")

    # --- 2d. Сравнение с жадным и случайным ---
    print("\n--- 2d. Сравнение алгоритмов ---")

    def greedy_tsp(dist_matrix):
        """Жадный алгоритм (ближайший сосед)."""
        n = len(dist_matrix)
        visited = [False] * n
        tour = [0]
        visited[0] = True
        total = 0

        for _ in range(n - 1):
            current = tour[-1]
            best_next = -1
            best_dist = float('inf')
            for j in range(n):
                if not visited[j] and dist_matrix[current][j] < best_dist:
                    best_dist = dist_matrix[current][j]
                    best_next = j
            tour.append(best_next)
            visited[best_next] = True
            total += best_dist

        total += dist_matrix[tour[-1]][tour[0]]
        tour.append(tour[0])
        return tour, total

    def random_tsp(dist_matrix, n_trials=1000):
        """Случайные маршруты."""
        n = len(dist_matrix)
        best_length = float('inf')
        for _ in range(n_trials):
            tour = list(range(n))
            random.shuffle(tour)
            tour.append(tour[0])
            length = sum(dist_matrix[tour[i]][tour[i + 1]]
                         for i in range(len(tour) - 1))
            if length < best_length:
                best_length = length
        return best_length

    greedy_tour, greedy_length = greedy_tsp(dist_matrix)
    random_best = random_tsp(dist_matrix)

    print(f"  ACO:    {best_length:.2f}")
    print(f"  Жадный: {greedy_length:.2f}")
    print(f"  Лучший случайный: {random_best:.2f}")
    print(f"  Улучшение ACO vs жадный: {(1 - best_length / greedy_length) * 100:+.1f}%")
    print(f"  Улучшение ACO vs случайный: {(1 - best_length / random_best) * 100:+.1f}%")


# ──────────────────────────── 3. Particle Swarm Optimization ────────────────────────────

def demo_particle_swarm_optimization():
    """Обновление скорости, личный/глобальный лучший, оптимизация функций."""
    print("\n\n" + "=" * 70)
    print("DEMO 3 — Particle Swarm Optimization (PSO)")
    print("=" * 70)

    # --- 3a. Целевая функция ---
    print("\n--- 3a. Целевая функция: Розенброк ---")
    print("  f(x, y) = (1 - x)² + 100(y - x²)²")
    print("  Минимум в (1, 1), f(1,1) = 0")

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    print(f"  f(0, 0) = {rosenbrock(0, 0):.2f}")
    print(f"  f(1, 1) = {rosenbrock(1, 1):.2f}")
    print(f"  f(-1, -1) = {rosenbrock(-1, -1):.2f}")

    # --- 3b. Параметры PSO ---
    print("\n--- 3b. Параметры PSO ---")
    n_particles = 30
    n_iterations = 100
    w = 0.7     # Инерция
    c1 = 1.5    # Когнитивный коэффициент (личный опыт)
    c2 = 1.5    # Социальный коэффициент (опыт сообщества)
    bounds = (-5, 5)

    print(f"  n_particles = {n_particles}")
    print(f"  n_iterations = {n_iterations}")
    print(f"  w (инерция) = {w}")
    print(f"  c1 (когнитивный) = {c1}")
    print(f"  c2 (социальный) = {c2}")
    print(f"  Границы поиска: [{bounds[0]}, {bounds[1]}]")

    # --- 3c. Инициализация и основной цикл ---
    print("\n--- 3c. Запуск PSO ---")
    random.seed(42)

    # Инициализация частиц
    particles = []
    for _ in range(n_particles):
        x = random.uniform(*bounds)
        y = random.uniform(*bounds)
        vx = random.uniform(-1, 1)
        vy = random.uniform(-1, 1)
        particles.append({
            "pos": [x, y],
            "vel": [vx, vy],
            "best_pos": [x, y],
            "best_val": rosenbrock(x, y),
        })

    # Глобальный лучший
    global_best = min(particles, key=lambda p: p["best_val"])
    global_best_pos = global_best["best_pos"][:]
    global_best_val = global_best["best_val"]

    history = []

    for iteration in range(n_iterations):
        for p in particles:
            x, y = p["pos"]
            val = rosenbrock(x, y)

            # Обновление личного лучшего
            if val < p["best_val"]:
                p["best_val"] = val
                p["best_pos"] = [x, y]

            # Обновление глобального лучшего
            if val < global_best_val:
                global_best_val = val
                global_best_pos = [x, y]

        # Обновление скоростей и позиций
        for p in particles:
            for dim in range(2):
                r1 = random.random()
                r2 = random.random()
                # Формула PSO: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive = c1 * r1 * (p["best_pos"][dim] - p["pos"][dim])
                social = c2 * r2 * (global_best_pos[dim] - p["pos"][dim])
                p["vel"][dim] = w * p["vel"][dim] + cognitive + social

                # Обновление позиции
                p["pos"][dim] += p["vel"][dim]

                # Ограничение границами
                p["pos"][dim] = max(bounds[0], min(bounds[1], p["pos"][dim]))

        history.append(global_best_val)

    print(f"  Найден минимум: ({global_best_pos[0]:.6f}, {global_best_pos[1]:.6f})")
    print(f"  Значение функции: {global_best_val:.8f}")
    print(f"  Истинный минимум: (1.000000, 1.000000), f = 0.000000")

    # --- 3d. Сходимость ---
    print("\n--- 3d. Сходимость PSO ---")
    print(f"  Итерация │ Лучшее значение │ Улучшение")
    checkpoints = [0, 9, 19, 29, 49, 79, 99]
    for i in checkpoints:
        if i < len(history):
            impr = ""
            if i > 0:
                delta = history[i - 1] - history[i]
                impr = f" ({delta:+.4f})"
            print(f"  {i:>9} │ {history[i]:>16.6f} │ {impr}")

    print(f"\n  Итого: за {n_iterations} итераций значение уменьшилось "
          f"с {history[0]:.4f} до {history[-1]:.8f}")

    # --- 3e. Визуализация траекторий ---
    print("\n--- 3e. Пример траектории одной частицы ---")
    random.seed(42)
    single_particle = {
        "pos": [random.uniform(-3, 3), random.uniform(-3, 3)],
        "vel": [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
    }

    trajectory = [tuple(single_particle["pos"])]
    for _ in range(30):
        for dim in range(2):
            r1 = random.random()
            r2 = random.random()
            cognitive = c1 * r1 * (global_best_pos[dim] - single_particle["pos"][dim])
            social = c2 * r2 * (global_best_pos[dim] - single_particle["pos"][dim])
            single_particle["vel"][dim] = (w * single_particle["vel"][dim]
                                           + cognitive + social)
            single_particle["pos"][dim] += single_particle["vel"][dim]
            single_particle["pos"][dim] = max(bounds[0], min(bounds[1],
                                                             single_particle["pos"][dim]))
        trajectory.append(tuple(single_particle["pos"]))

    print(f"  Начало: ({trajectory[0][0]:.2f}, {trajectory[0][1]:.2f})")
    print(f"  Конец:  ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f})")
    print(f"  Промежуточные точки (каждая 5-я):")
    for i in range(0, len(trajectory), 5):
        x, y = trajectory[i]
        val = rosenbrock(x, y)
        print(f"    Шаг {i:>2}: ({x:>7.3f}, {y:>7.3f}), f = {val:.4f}")


# ──────────────────────────── 4. Flocking & Swarming ────────────────────────────

def demo_flocking_swarming():
    """Правила разделения, выравнивания, слияния (модель Рейнольдса)."""
    print("\n\n" + "=" * 70)
    print("DEMO 4 — Flocking & Swarming (Model of Reynolds)")
    print("=" * 70)

    # --- 4a. Три правила стаи ---
    print("\n--- 4a. Три правила стаи (Boids, Craig Reynolds, 1986) ---")
    print("  1. SEPARATION (Разделение): избегать столкновений с ближайшими соседями")
    print("  2. ALIGNMENT (Выравнивание): двигаться в направлении средней скорости соседей")
    print("  3. COHESION  (Слияние): двигаться к центру масс ближайших соседей")

    # --- 4b. Инициализация агентов ---
    print("\n--- 4b. Инициализация агентов ---")
    random.seed(42)
    n_agents = 20
    space_size = 50
    perception_radius = 10.0
    separation_radius = 3.0

    # Параметры весов правил
    w_separation = 1.5
    w_alignment = 1.0
    w_cohesion = 1.0

    # Инициализация позиций и скоростей
    agents = []
    for _ in range(n_agents):
        x = random.uniform(10, space_size - 10)
        y = random.uniform(10, space_size - 10)
        vx = random.uniform(-2, 2)
        vy = random.uniform(-2, 2)
        agents.append({"pos": [x, y], "vel": [vx, vy]})

    print(f"  Агентов: {n_agents}, Пространство: {space_size}×{space_size}")
    print(f"  Радиус восприятия: {perception_radius}")
    print(f"  Веса: разделение={w_separation}, выравнивание={w_alignment}, слияние={w_cohesion}")

    def distance(a, b):
        return math.sqrt((a["pos"][0] - b["pos"][0]) ** 2 +
                         (a["pos"][1] - b["pos"][1]) ** 2)

    def flocking_step(agents):
        """Один шаг симуляции стаи."""
        for i in range(len(agents)):
            sep = [0.0, 0.0]
            sep_count = 0
            ali = [0.0, 0.0]
            ali_count = 0
            coh = [0.0, 0.0]
            coh_count = 0

            for j in range(len(agents)):
                if i == j:
                    continue
                d = distance(agents[i], agents[j])

                if d < separation_radius:
                    # Разделение: убегаем от слишком близких соседей
                    dx = agents[i]["pos"][0] - agents[j]["pos"][0]
                    dy = agents[i]["pos"][1] - agents[j]["pos"][1]
                    if d > 0:
                        sep[0] += dx / d
                        sep[1] += dy / d
                    sep_count += 1

                if d < perception_radius:
                    # Выравнивание: берём скорость соседа
                    ali[0] += agents[j]["vel"][0]
                    ali[1] += agents[j]["vel"][1]
                    ali_count += 1

                    # Слияние: берём позицию соседа
                    coh[0] += agents[j]["pos"][0]
                    coh[1] += agents[j]["pos"][1]
                    coh_count += 1

            # Нормализация и применение весов
            if sep_count > 0:
                sep[0] = sep[0] / sep_count * w_separation
                sep[1] = sep[1] / sep_count * w_separation

            if ali_count > 0:
                ali[0] = ali[0] / ali_count - agents[i]["vel"][0]
                ali[1] = ali[1] / ali_count - agents[i]["vel"][1]
                ali[0] *= w_alignment
                ali[1] *= w_alignment

            if coh_count > 0:
                coh[0] = coh[0] / coh_count - agents[i]["pos"][0]
                coh[1] = coh[1] / coh_count - agents[i]["pos"][1]
                coh[0] *= w_cohesion
                coh[1] *= w_cohesion

            # Обновление скорости
            agents[i]["vel"][0] += sep[0] + ali[0] + coh[0]
            agents[i]["vel"][1] += sep[1] + ali[1] + coh[1]

            # Ограничение максимальной скорости
            speed = math.sqrt(agents[i]["vel"][0] ** 2 + agents[i]["vel"][1] ** 2)
            max_speed = 3.0
            if speed > max_speed:
                agents[i]["vel"][0] = agents[i]["vel"][0] / speed * max_speed
                agents[i]["vel"][1] = agents[i]["vel"][1] / speed * max_speed

            # Обновление позиции
            agents[i]["pos"][0] += agents[i]["vel"][0]
            agents[i]["pos"][1] += agents[i]["vel"][1]

            # Отражение от границ (не выходить за пределы)
            for dim in range(2):
                if agents[i]["pos"][dim] < 0:
                    agents[i]["pos"][dim] = 0
                    agents[i]["vel"][dim] *= -0.5
                elif agents[i]["pos"][dim] > space_size:
                    agents[i]["pos"][dim] = space_size
                    agents[i]["vel"][dim] *= -0.5

    # --- 4c. Запуск симуляции ---
    print("\n--- 4c. Симуляция (50 шагов) ---")
    n_steps = 50
    metrics_history = []

    for step in range(n_steps):
        flocking_step(agents)

        # Вычисляем метрики
        avg_speed = sum(math.sqrt(a["vel"][0] ** 2 + a["vel"][1] ** 2)
                        for a in agents) / len(agents)

        # Центр масс
        cx = sum(a["pos"][0] for a in agents) / len(agents)
        cy = sum(a["pos"][1] for a in agents) / len(agents)

        # Среднее расстояние до центра масс (мера компактности)
        avg_dist_to_center = sum(
            math.sqrt((a["pos"][0] - cx) ** 2 + (a["pos"][1] - cy) ** 2)
            for a in agents
        ) / len(agents)

        # Среднее косинусное сходство направлений
        speeds = [(a["vel"][0], a["vel"][1]) for a in agents]
        alignment_score = 0
        count = 0
        for i in range(len(speeds)):
            for j in range(i + 1, len(speeds)):
                dot = speeds[i][0] * speeds[j][0] + speeds[i][1] * speeds[j][1]
                mag1 = math.sqrt(speeds[i][0] ** 2 + speeds[i][1] ** 2)
                mag2 = math.sqrt(speeds[j][0] ** 2 + speeds[j][1] ** 2)
                if mag1 > 0 and mag2 > 0:
                    alignment_score += dot / (mag1 * mag2)
                    count += 1
        alignment_score = alignment_score / count if count > 0 else 0

        metrics_history.append({
            "step": step,
            "avg_speed": avg_speed,
            "center": (cx, cy),
            "compactness": avg_dist_to_center,
            "alignment": alignment_score,
        })

    print(f"  Начальные метрики (шаг 0):")
    m0 = metrics_history[0]
    print(f"    Средняя скорость: {m0['avg_speed']:.2f}")
    print(f"    Центр масс: ({m0['center'][0]:.1f}, {m0['center'][1]:.1f})")
    print(f"    Компактность: {m0['compactness']:.2f}")
    print(f"    Выравнивание: {m0['alignment']:.3f}")

    print(f"\n  Финальные метрики (шаг {n_steps}):")
    mf = metrics_history[-1]
    print(f"    Средняя скорость: {mf['avg_speed']:.2f}")
    print(f"    Центр масс: ({mf['center'][0]:.1f}, {mf['center'][1]:.1f})")
    print(f"    Компактность: {mf['compactness']:.2f}")
    print(f"    Выравнивание: {mf['alignment']:.3f}")

    # --- 4d. Эмергентное поведение ---
    print("\n--- 4d. Эмергентное поведение (выводы) ---")
    speed_change = mf["avg_speed"] - m0["avg_speed"]
    compact_change = mf["compactness"] - m0["compactness"]
    align_change = mf["alignment"] - m0["alignment"]

    print(f"  Изменение средней скорости: {speed_change:+.2f}")
    print(f"  Изменение компактности: {compact_change:+.2f}")
    print(f"  Изменение выравнивания: {align_change:+.3f}")

    print(f"\n  Наблюдения:")
    if compact_change < -1:
        print("  • Стая стала более компактной (слияние работает)")
    if align_change > 0.05:
        print("  • Направления движения выровнялись (выравнивание работает)")
    if speed_change < -0.1:
        print("  • Средняя скорость уменьшилась (стабилизация)")
    elif speed_change > 0.1:
        print("  • Средняя скорость увеличилась (разгон)")

    print(f"\n  Ключевая идея: из простых локальных правил (SEPARATION +")
    print(f"  ALIGNMENT + COHESION) возникает сложное коллективное поведение,")
    print(f"  не запрограммированное ни в одном отдельном агенте.")


if __name__ == "__main__":
    demo_stigmergy()
    demo_ant_colony_optimization()
    demo_particle_swarm_optimization()
    demo_flocking_swarming()
