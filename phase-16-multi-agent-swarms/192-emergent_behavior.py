"""192 — Emergent Behavior: самоорганизация, формирование паттернов, коллективный интеллект

Темы:
  1. Self-Organization (порядок, симметрия, фазовые переходы)
  2. Pattern Formation (клеточные автоматы, реакция-диффузия, паттерны Тьюринга)
  3. Collective Intelligence (мудрость толпы, коллективное решение задач)
  4. Stigmergy (средовая координация, феромонные следы, строительство)

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
# ДЕМО 1: Самоорганизация
# ============================================================

def demo_self_organization():
    """Показывает самоорганизацию: порядковые параметры, нарушение симметрии, фазовые переходы."""
    print("=" * 70)
    print("ДЕМО 1: САМООРГАНИЗАЦИЯ — порядковые параметры, симметрия, фазовые переходы")
    print("=" * 70)

    # --- 1a. Порядковый параметр (метательные метки) ---
    # Порядковый параметр измеряет степень упорядоченности системы.
    # Для ансамбля из N спинов: m = (1/N) * sum(s_i), где s_i ∈ {-1, +1}
    print("\n--- 1a. Порядковый параметр (метательные метки) ---")
    N = 20
    # Случайная конфигурация: каждый спин +1 или -1
    spins = [random.choice([-1, 1]) for _ in range(N)]
    m_random = sum(spins) / N
    print(f"  Случайная конфигурация из {N} спинов: {spins}")
    print(f"  Порядковый параметр m = (1/{N}) * Σs_i = {m_random:.4f}")
    print(f"  |m| = {abs(m_random):.4f} → {'случайное состояние (m≈0)' if abs(m_random) < 0.3 else 'упорядоченное состояние'}")

    # Упорядоченная конфигурация
    spins_ordered = [1] * N
    m_ordered = sum(spins_ordered) / N
    print(f"\n  Упорядоченная конфигурация: {spins_ordered[:8]}... (все +1)")
    print(f"  Порядковый параметр m = {m_ordered:.4f}")
    print(f"  |m| = {abs(m_ordered):.4f} → полностью упорядочено")

    # --- 1b. Нарушение симметрии ---
    # Изотропная система имеет симметрию вращения.
    # При фазовом переходе симметрия自发но нарушается.
    print("\n--- 1b. Нарушение симметрии ---")
    # Симметричная кривая (гауссова) и её «слом»
    x_values = [i * 0.2 - 5 for i in range(51)]
    # Два «пика» — двойной потенциал
    def double_well(x, a=1.0, b=0.1):
        """Двойной потенциал U(x) = -a*x^2 + b*x^4"""
        return -a * x ** 2 + b * x ** 4

    wells = [(x, double_well(x)) for x in x_values]
    # Нахождение минимумов
    min_wells = sorted(wells, key=lambda p: p[1])[:2]
    print(f"  Двойной потенциал U(x) = -x² + 0.1·x⁴")
    print(f"  Два минимума (нарушение симметрии):")
    for x, u in min_wells:
        print(f"    x = {x:+.2f}, U(x) = {u:.4f}")

    # Более того, если разбить на N=100 агентов и «бросить» в потенциал,
    # то примерно поровну упадут в каждый минимум
    N_agents = 1000
    left_count = 0
    right_count = 0
    for _ in range(N_agents):
        # Простая модель: агент спускается в ближайший минимум
        x_init = random.uniform(-3, 3)
        if x_init < 0:
            left_count += 1
        else:
            right_count += 1
    print(f"\n  {N_agents} агентов брошено в потенциал:")
    print(f"    Левый минимум: {left_count} агентов ({left_count/N_agents*100:.1f}%)")
    print(f"    Правый минимум: {right_count} агентов ({right_count/N_agents*100:.1f}%)")
    print(f"  → Симметрия нарушена自发но: агенты самоорганизовались в два кластера")

    # --- 1c. Фазовые переходы ---
    # Температура T управляет хаосом. При T_crit происходит фазовый переход.
    print("\n--- 1c. Фазовый переход (анизотропный обменный ферромагнетик) ---")
    T_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print("  Модель Изинга на 1D решётке (L=20, 1000 шагов MC)")
    print(f"  {'T':>6} | {'|m|':>8} | {'E/N':>8} | {'Состояние'}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")

    L = 20  # Размер решётки
    for T in T_values:
        # Инициализация: случайная решётка
        lattice = [random.choice([-1, 1]) for _ in range(L)]
        beta = 1.0 / T if T > 0 else 1e6

        # Метрополис-алгоритм
        for _ in range(1000):
            i = random.randint(0, L - 1)
            # Сумма соседей (периодические условия)
            neighbors = lattice[(i - 1) % L] + lattice[(i + 1) % L]
            s = lattice[i]
            dE = 2 * s * neighbors  # ΔE = 2·s_i·(сумма соседей)
            if dE <= 0 or random.random() < math.exp(-beta * dE):
                lattice[i] = -s

        m = abs(sum(lattice) / L)
        E = -sum(lattice[i] * lattice[(i + 1) % L] for i in range(L)) / L
        state = "ферромагнит" if m > 0.5 else ("критическая область" if m > 0.2 else "парамагнит")
        print(f"  {T:6.1f} | {m:8.4f} | {E:8.4f} | {state}")

    print("\n  Вывод: при понижении T происходит фазовый переход")
    print("  от парамагнитного (|m|≈0) к ферромагнитному (|m|>0)")

    # --- 1d. Коллективное поведение: «мудрость толпы» ---
    print("\n--- 1d. Самоорганизация через взаимодействие агентов ---")
    # Модель Вонгаса: каждый агент выбирает стратегию на основе邻居
    N = 50
    # Агенты: 0 или 1 (два «типа»)
    agents = [random.choice([0, 1]) for _ in range(N)]
    print(f"  Начальное состояние: 0×={sum(1 for a in agents if a==0)}, 1×={sum(1 for a in agents if a==1)}")

    # 20 шагов динамики: агенты копируют более успешного соседа
    for step in range(20):
        i = random.randint(0, N - 1)
        left = agents[(i - 1) % N]
        right = agents[(i + 1) % N]
        # Успех = количество одноимённых соседей
        left_score = (1 if agents[i] == left else 0) + (1 if agents[i] == left else 0)
        right_score = (1 if agents[i] == right else 0) + (1 if agents[i] == right else 0)
        if left_score > right_score:
            agents[i] = left
        elif right_score > left_score:
            agents[i] = right

    count_0 = sum(1 for a in agents if a == 0)
    count_1 = sum(1 for a in agents if a == 1)
    print(f"  После 20 шагов: 0×={count_0}, 1×={count_1}")
    print(f"  → Самоорганизация: доминирующий тип вытесняет меньшинство")


# ============================================================
# ДЕМО 2: Формирование паттернов
# ============================================================

def demo_pattern_formation():
    """Клеточные автоматы, реакция-диффузия, паттерны Тьюринга."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: ФОРМИРОВАНИЕ ПАТТЕРНОВ — клеточные автоматы, реакция-диффузия")
    print("=" * 70)

    # --- 2a. Клеточный автомат «Жизнь» Конвея ---
    print("\n--- 2a. Клеточный автомат «Жизнь» Конвея ---")
    size = 20
    # Инициализация решётки
    grid = [[0] * size for _ in range(size)]
    # Glider («планер») — знаменитый паттерн
    glider_coords = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
    for r, c in glider_coords:
        grid[r][c] = 1

    def count_neighbors(g, r, c, n):
        """Подсчёт соседей с циклическими условиями."""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = (r + dr) % n, (c + dc) % n
                count += g[nr][nc]
        return count

    def step_life(g, n):
        """Один шаг клеточного автомата."""
        new_g = [[0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                nb = count_neighbors(g, r, c, n)
                if g[r][c] == 1:
                    # Живая клетка выживает при 2 или 3 соседях
                    new_g[r][c] = 1 if nb in (2, 3) else 0
                else:
                    # Мёртвая клетка оживает при ровно 3 соседях
                    new_g[r][c] = 1 if nb == 3 else 0
        return new_g

    print("  Начальное состояние (Glider на 20×20):")
    # Выводим маленький фрагмент (верхний левый угол 10×10)
    for r in range(min(10, size)):
        row_str = ""
        for c in range(min(10, size)):
            row_str += "█" if grid[r][c] else "·"
        print(f"    {row_str}")

    # Запускаем на 5 шагов
    total_alive = [sum(sum(row) for row in grid)]
    for step_num in range(1, 6):
        grid = step_life(grid, size)
        alive = sum(sum(row) for row in grid)
        total_alive.append(alive)

    print(f"\n  Живых клеток по шагам: {total_alive}")
    print("  Glider перемещается по диагонали — паттерн сохраняется!")

    # --- 2b. Правило 110 (одномерный клеточный автомат) ---
    print("\n--- 2b. Одномерный клеточный автомат — Правило 110 ---")
    width = 60
    steps = 30
    # Начальное состояние: одна клетка в центре
    cells = [0] * width
    cells[width // 2] = 1

    # Правило 110: 01101110 в двоичном = 110
    rule = 110
    rule_binary = format(rule, '08b')
    print(f"  Правило {rule} → двоичное: {rule_binary}")

    # Построение таблицы переходов
    transitions = {}
    for i in range(8):
        # Тройка соседей → новый state
        pattern = format(i, '03b')
        new_state = int(rule_binary[7 - i])
        transitions[pattern] = new_state

    print(f"  Таблица переходов:")
    for pattern in ['111', '110', '101', '100', '011', '010', '001', '000']:
        print(f"    {pattern} → {transitions[pattern]}")

    # Визуализация
    print(f"\n  Эволюция (шаг 0 — начало):")
    # Показываем первые 4 шага
    all_states = [cells[:]]
    for step_num in range(steps):
        new_cells = [0] * width
        for i in range(1, width - 1):
            pattern = str(cells[i - 1]) + str(cells[i]) + str(cells[i + 1])
            new_cells[i] = transitions.get(pattern, 0)
        cells = new_cells
        all_states.append(cells[:])

    for step_num in range(min(8, steps)):
        row_str = ""
        for c in range(width):
            row_str += "█" if all_states[step_num][c] else "·"
        print(f"    Шаг {step_num:2d}: {row_str}")

    # --- 2c. Реакция-диффузия (модель Грея-Скотта) ---
    print("\n--- 2c. Реакция-диффузия (упрощённая модель) ---")
    # Упрощённая 1D модель: U и V — концентрации двух веществ
    # ∂U/∂t = Du·∇²U - U·V² + F·(1-U)
    # ∂V/∂t = Dv·∇²V + U·V² - (F+k)·V
    N = 50
    Du = 0.16  # Коэффициент диффузии U
    Dv = 0.08  # Коэффициент диффузии V
    F = 0.04   # Параметр корма
    k = 0.06   # Параметр скорости
    dt = 1.0   # Шаг времени

    # Начальное состояние:均匀 U=1, V=0, с «пятном» V
    U = [1.0] * N
    V = [0.0] * N
    # Добавляем начальное пятно в центре
    for i in range(N // 2 - 3, N // 2 + 3):
        V[i] = random.uniform(0.25, 0.5)
        U[i] = 0.5

    print(f"  Параметры: Du={Du}, Dv={Dv}, F={F}, k={k}")
    print(f"  Начальное состояние: U均匀≈1, V均匀≈0 + пятно в центре")

    # Эволюция (200 шагов)
    for _ in range(200):
        U_new = U[:]
        V_new = V[:]
        for i in range(1, N - 1):
            laplacian_u = U[i - 1] + U[i + 1] - 2 * U[i]
            laplacian_v = V[i - 1] + V[i + 1] - 2 * V[i]
            uvv = U[i] * V[i] * V[i]
            U_new[i] = U[i] + dt * (Du * laplacian_u - uvv + F * (1 - U[i]))
            V_new[i] = V[i] + dt * (Dv * laplacian_v + uvv - (F + k) * V[i])
        U = U_new
        V = V_new

    # Вывод концентрации V (индикатор паттерна)
    v_max = max(V)
    v_avg = sum(V) / len(V)
    print(f"  После 200 шагов:")
    print(f"  V_max = {v_max:.4f}, V_avg = {v_avg:.4f}")
    print("  Распределение V (·=0.01, ░=0.05, ▒=0.1, ▓=0.2, █=0.3+):")
    row_str = "    "
    for v in V:
        if v < 0.01:
            row_str += "·"
        elif v < 0.05:
            row_str += "░"
        elif v < 0.1:
            row_str += "▒"
        elif v < 0.2:
            row_str += "▓"
        else:
            row_str += "█"
    print(row_str)

    # --- 2d. Паттерны Тьюринга: волнистые полосы ---
    print("\n--- 2d. Паттерны Тьюринга: генерация волновых структур ---")
    # Параметрическое моделирование: суперпозиция волн
    N = 80
    wave1 = []
    wave2 = []
    combined = []
    for x in range(N):
        # Две интерферирующие волны с разными частотами
        w1 = math.sin(2 * math.pi * x / 15)
        w2 = 0.7 * math.sin(2 * math.pi * x / 23 + 1.2)
        # Нелинейная комбинация (имитация Тьюринга)
        c = w1 + w2 + 0.3 * w1 * w2
        wave1.append(w1)
        wave2.append(w2)
        combined.append(c)

    c_min = min(combined)
    c_max = max(combined)
    print(f"  Интерференция двух волн (λ₁=15, λ₂=23):")
    print("  Результат (нелинейная суперпозиция):")
    row_str = "    "
    for c in combined:
        norm = (c - c_min) / (c_max - c_min) if c_max > c_min else 0.5
        if norm < 0.2:
            row_str += "·"
        elif norm < 0.4:
            row_str += "░"
        elif norm < 0.6:
            row_str += "▒"
        elif norm < 0.8:
            row_str += "▓"
        else:
            row_str += "█"
    print(row_str)

    print(f"\n  Количество «пиков» (волнистых структур): ", end="")
    peaks = sum(1 for i in range(1, len(combined) - 1)
                if combined[i] > combined[i - 1] and combined[i] > combined[i + 1])
    print(f"{peaks}")
    print("  → Интерференция создаёт сложные паттерны, подобные Тьюринговым")


# ============================================================
# ДЕМО 3: Коллективный интеллект
# ============================================================

def demo_collective_intelligence():
    """Мудрость толпы, коллективное решение задач, агрегация мнений."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: КОЛЛЕКТИВНЫЙ ИНТЕЛЛЕКТ — мудрость толпы, агрегация мнений")
    print("=" * 70)

    # --- 3a. Задача «Угадай число быка» ---
    print("\n--- 3a. «Мудрость толпы»: угадай количество быков в стаде ---")
    # Галтон: каждое отдельное оценка неточна, но медиана толпы точна
    true_count = 789
    n_estimates = 100
    # Каждый «наблюдатель» делает ошибку ~ ±30%
    estimates = []
    for _ in range(n_estimates):
        # Логнормальное распределение ошибок
        error = random.gauss(0, 0.15)
        est = int(true_count * math.exp(error))
        estimates.append(est)

    # Агрегация: медиана
    estimates_sorted = sorted(estimates)
    median_est = estimates_sorted[len(estimates_sorted) // 2]
    mean_est = sum(estimates) / len(estimates)
    individual_errors = [abs(e - true_count) for e in estimates]
    median_error = abs(median_est - true_count)
    mean_error = abs(mean_est - true_count)
    avg_individual_error = sum(individual_errors) / len(individual_errors)

    print(f"  Истинное значение: {true_count}")
    print(f"  Количество оценок: {n_estimates}")
    print(f"  Несколько оценок: {estimates[:10]}...")
    print(f"  Средняя индивидуальная ошибка: {avg_individual_error:.1f}")
    print(f"  Среднее всех оценок: {mean_est:.1f} (ошибка: {mean_error:.1f})")
    print(f"  Медиана всех оценок: {median_est} (ошибка: {median_error})")
    print(f"  → Коллективная оценка точнее любой индивидуальной!")

    # --- 3b. Агрегация ранжирований (метод Кондорсе) ---
    print("\n--- 3b. Агрегация ранжирований (правило Кондорсе) ---")
    # 5 «кандидатов» (A, B, C, D, E)
    candidates = ['A', 'B', 'C', 'D', 'E']
    n_voters = 200

    # Генерация предпочтений из трёх «платформ»
    platforms = [
        ['A', 'B', 'C', 'D', 'E'],  # Платформа 1
        ['C', 'D', 'E', 'A', 'B'],  # Платформа 2
        ['B', 'E', 'A', 'D', 'C'],  # Платформа 3
    ]
    platform_weights = [0.4, 0.35, 0.25]

    all_rankings = []
    for _ in range(n_voters):
        # Выбор платформы
        r = random.random()
        cumsum = 0
        chosen = platforms[0]
        for p, w in zip(platforms, platform_weights):
            cumsum += w
            if r < cumsum:
                chosen = p
                break
        all_rankings.append(chosen[:])

    # Правило Кондорсе: для каждой пары кандидатов считаем, кто «побеждает»
    wins = {c: 0 for c in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            c1_wins = sum(1 for r in all_rankings if r.index(c1) < r.index(c2))
            c2_wins = n_voters - c1_wins
            if c1_wins > c2_wins:
                wins[c1] += 1
            else:
                wins[c2] += 1

    print(f"  {n_voters} избирателей, 3 платформы, 5 кандидатов")
    print(f"  Правило Кондорсе: для каждой пары — кто чаще предпочтительнее")
    print(f"  Количество «побед» в парных сравнениях:")
    for c in sorted(wins, key=wins.get, reverse=True):
        bar = "█" * wins[c]
        print(f"    Кандидат {c}: {wins[c]} побед {bar}")
    winner = max(wins, key=wins.get)
    print(f"  Победитель Кондорсе: {winner} (побеждает всех в парных сравнениях)")

    # --- 3c. Коллективное решение через «толпу» ---
    print("\n--- 3c. Коллективное решение: угадай вес предмета ---")
    true_weight = 347  # грамм
    n_guesses = 50
    guesses = [int(random.gauss(true_weight, 50)) for _ in range(n_guesses)]

    mean_guess = sum(guesses) / len(guesses)
    median_guess = sorted(guesses)[len(guesses) // 2]
    # Ближайший к среднему индивидуальный ответ
    closest_individual = min(guesses, key=lambda g: abs(g - true_weight))

    print(f"  Предмет весит: {true_weight} г")
    print(f"  {n_guesses} участников угадывают вес")
    print(f"  Несколько оценок: {guesses[:15]}...")
    print(f"  Лучшая индивидуальная оценка: {closest_individual} (ошибка {abs(closest_individual - true_weight)} г)")
    print(f"  Среднее толпы: {mean_guess:.1f} г (ошибка {abs(mean_guess - true_weight):.1f} г)")
    print(f"  Медиана толпы: {median_guess} г (ошибка {abs(median_guess - true_weight)} г)")

    # Разброс
    variance = sum((g - mean_guess) ** 2 for g in guesses) / len(guesses)
    std_dev = math.sqrt(variance)
    print(f"  Стандартное отклонение: {std_dev:.1f} г")
    print(f"  → Среднее толпы ({mean_guess:.0f}) ближе к истине ({true_weight})")
    print(f"     чем большинство индивидуальных оценок")

    # --- 3d. Краудсорсинг: коллективная классификация ---
    print("\n--- 3d. Коллективная классификация (множественная разметка) ---")
    n_items = 6
    n_annotators = 15
    # Истинные метки
    true_labels = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
    items = [f"image_{i}" for i in range(n_items)]

    # Каждый аннотатор «видит» изображение с шумом
    noisy_labels = []
    for a in range(n_annotators):
        annotator_labels = []
        for label in true_labels:
            if random.random() < 0.75:  # 75% точность
                annotator_labels.append(label)
            else:
                annotator_labels.append(random.choice(['cat', 'dog', 'bird']))
        noisy_labels.append(annotator_labels)

    # Голосование по каждому изображению
    results = []
    for i in range(n_items):
        votes = collections.Counter(al[i] for al in noisy_labels)
        majority_label = votes.most_common(1)[0][0]
        majority_count = votes.most_common(1)[0][1]
        confidence = majority_count / n_annotators
        correct = "✓" if majority_label == true_labels[i] else "✗"
        results.append((items[i], majority_label, confidence, correct))

    print(f"  {n_items} изображений, {n_annotators} аннотаторов (точность ~75%)")
    print(f"  {'Изображение':<12} | {'Голосование':<10} | {'Уверенность':<12} | {'Верно?'}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*5}")
    for item, label, conf, correct in results:
        print(f"  {item:<12} | {label:<10} | {conf:<12.2f} | {correct}")

    accuracy = sum(1 for _, _, _, c in results if c == '✓') / n_items * 100
    print(f"\n  Точность коллективной классификации: {accuracy:.0f}%")


# ============================================================
# ДЕМО 4: Стигмергия
# ============================================================

def demo_stigmergy():
    """Средовая координация, феромонные следы, строительство муравьями."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: СТИГМЕРГИЯ — средовая координация, феромоны, строительство")
    print("=" * 70)

    # --- 4a. Муравьиный алгоритм (ACO) ---
    print("\n--- 4a. Муравьиный алгоритм: решение задачи коммивояжёра ---")
    # 8 городов: задача TSP
    random.seed(42)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(8)]
    city_names = [chr(65 + i) for i in range(8)]

    def dist(c1, c2):
        """Расстояние между двумя городами."""
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    # Матрица расстояний
    n = len(cities)
    D = [[dist(cities[i], cities[j]) for j in range(n)] for i in range(n)]

    # Феромонная матрица
    tau = [[1.0] * n for _ in range(n)]
    alpha = 1.0   # Влияние феромона
    beta = 3.0    # Влияние расстояния
    rho = 0.5     # Испарение
    Q = 100.0     # Константа феромона

    best_len = float('inf')
    best_tour = None

    # 50 итераций, 10 муравьёв
    for iteration in range(50):
        ant_tours = []
        ant_lens = []

        for ant in range(10):
            visited = [False] * n
            tour = [random.randint(0, n - 1)]
            visited[tour[0]] = True

            for _ in range(n - 1):
                current = tour[-1]
                # Выбор следующего города
                probs = []
                for j in range(n):
                    if not visited[j]:
                        tau_ij = tau[current][j] ** alpha
                        eta_ij = (1.0 / max(D[current][j], 0.01)) ** beta
                        probs.append((j, tau_ij * eta_ij))
                    else:
                        probs.append((j, 0.0))

                total = sum(p for _, p in probs)
                if total == 0:
                    # Все города посещены — случайный выбор оставшегося
                    for j in range(n):
                        if not visited[j]:
                            tour.append(j)
                            visited[j] = True
                            break
                    continue

                # Рулетка
                r = random.random() * total
                cumsum = 0
                next_city = probs[0][0]
                for j, p in probs:
                    cumsum += p
                    if r <= cumsum:
                        next_city = j
                        break
                tour.append(next_city)
                visited[next_city] = True

            # Длина маршрута
            tour_len = sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))
            ant_tours.append(tour)
            ant_lens.append(tour_len)

        # Обновление феромона
        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1 - rho)  # Испарение

        # Добавление феромона лучшими муравьями
        for tour, tour_len in zip(ant_tours, ant_lens):
            dq = Q / tour_len
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                tau[a][b] += dq
                tau[b][a] += dq

        # Лучший маршрут
        min_idx = ant_lens.index(min(ant_lens))
        if ant_lens[min_idx] < best_len:
            best_len = ant_lens[min_idx]
            best_tour = ant_tours[min_idx][:]

    print(f"  Города: {', '.join(city_names)}")
    print(f"  Параметры: α={alpha}, β={beta}, ρ={rho}, Q={Q}")
    tour_str = " → ".join(city_names[c] for c in best_tour)
    tour_str += f" → {city_names[best_tour[0]]}"
    print(f"  Лучший маршрут (50 итераций): {tour_str}")
    print(f"  Длина маршрута: {best_len:.2f}")
    print(f"  → Муравьи нашли хороший маршрут через стигмергию (феромоны)")

    # --- 4b. Феромонные следы: навигация ---
    print("\n--- 4b. Феромонные следы: имитация навигации ---")
    # Простая сетка 10×10
    grid_size = 10
    pheromone = [[0.0] * grid_size for _ in range(grid_size)]
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)

    # Несколько «маршрутов» муравьёв от start до goal
    n_ants = 50
    routes = []
    for _ in range(n_ants):
        route = [start]
        pos = start
        while pos != goal:
            # Случайный шаг: вправо или вниз (упрощённо)
            options = []
            if pos[1] < grid_size - 1:
                options.append((pos[0], pos[1] + 1))  # Вправо
            if pos[0] < grid_size - 1:
                options.append((pos[0] + 1, pos[1]))  # Вниз
            next_pos = random.choice(options)
            route.append(next_pos)
            pos = next_pos
        routes.append(route)
        # Оставляем феромон на маршруте
        for cell in route:
            pheromone[cell[0]][cell[1]] += 1.0

    # Путь с наибольшим феромоном (аппроксимация)
    # Жадный путь: на каждом шаге идём в клетку с бóльшим феромоном
    path = [start]
    pos = start
    while pos != goal:
        options = []
        if pos[1] < grid_size - 1:
            options.append((pos[0], pos[1] + 1))
        if pos[0] < grid_size - 1:
            options.append((pos[0] + 1, pos[1]))
        # Выбираем с наибольшим феромоном
        best_next = max(options, key=lambda p: pheromone[p[0]][p[1]])
        path.append(best_next)
        pos = best_next

    print(f"  Сетка {grid_size}×{grid_size}, {n_ants} муравьёв")
    print(f"  Маршруты: от (0,0) до ({grid_size-1},{grid_size-1}), только ↓ и →")
    print(f"  Феромонная карта (μ = 0.0, ░ = 0.5, ▒ = 1.0, ▓ = 2.0, █ = 3.0+):")
    for r in range(grid_size):
        row_str = "    "
        for c in range(grid_size):
            v = pheromone[r][c]
            if v < 0.5:
                row_str += "μ"
            elif v < 1.0:
                row_str += "░"
            elif v < 2.0:
                row_str += "▒"
            elif v < 3.0:
                row_str += "▓"
            else:
                row_str += "█"
        print(row_str)
    print(f"  Жадный путь (по макс. феромону): длина {len(path) - 1} шагов")

    # --- 4c. Коллективное строительство ---
    print("\n--- 4c. Коллективное строительство: модель роевого строительства ---")
    # Модель: агенты добавляют «кирпичи» в структуру
    # Каждый агент: «если рядом N кирпичей → поставить свой»
    width = 20
    height = 10
    structure = [[0] * width for _ in range(height)]
    # Начальный «каркас» — нижний ряд
    structure[-1] = [1] * width

    n_builders = 10
    steps = 0
    target_height = 5  # Целевая высота

    while steps < 200:
        steps += 1
        for b in range(n_builders):
            # Случайная позиция
            r = random.randint(1, height - 1)
            c = random.randint(0, width - 1)
            if structure[r][c] == 1:
                continue

            # Подсчёт «опоры»: есть ли кирпич снизу или рядом
            support = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if structure[nr][nc] == 1:
                        support += 1

            # Правило: минимум 2 опоры
            if support >= 2:
                structure[r][c] = 1

        # Проверка высоты
        max_h = 0
        for r in range(height):
            if sum(structure[r]) > 0:
                max_h = height - r
                break
        if max_h >= target_height:
            break

    print(f"  Агентов-строителей: {n_builders}")
    print(f"  Правило: ставить кирпич, если ≥2 соседа уже заняты")
    print(f"  Результат после {steps} шагов:")
    for r in range(height):
        row_str = "    "
        for c in range(width):
            row_str += "█" if structure[r][c] else "·"
        print(row_str)

    bricks = sum(sum(row) for row in structure)
    print(f"  Всего кирпичей: {bricks}")
    print("  → Коллектив построил структуру без центрального планирования!")

    # --- 4d. Стигмергия: рой муравьёв ищет еду ---
    print("\n--- 4d. Стигмергия: рой муравьёв ищет еду ---")
    # Простая 1D модель
    n_cells = 20
    food_positions = {5: 10, 15: 10}  # {позиция: количество еды}
    pheromone = [0.0] * n_cells
    n_ants = 20
    ant_positions = [n_cells // 2] * n_ants  # Все начинают в центре

    # 50 шагов
    for step in range(50):
        for i in range(n_ants):
            # Муравей следует градиенту феромона + случайный блуждание
            pos = ant_positions[i]
            if random.random() < 0.7:  # 70% — следуем феромону
                left_p = pheromone[pos - 1] if pos > 0 else 0
                right_p = pheromone[pos + 1] if pos < n_cells - 1 else 0
                if left_p > right_p:
                    ant_positions[i] = pos - 1
                elif right_p > left_p:
                    ant_positions[i] = pos + 1
                else:
                    ant_positions[i] = pos + random.choice([-1, 1])
            else:  # 30% — случайное блуждание
                ant_positions[i] = max(0, min(n_cells - 1, pos + random.choice([-1, 1])))

            # Оставляем феромон
            ant_positions[i] = max(0, min(n_cells - 1, ant_positions[i]))
            pheromone[ant_positions[i]] += 0.5

            # Если нашли еду — забираем её
            p = ant_positions[i]
            if p in food_positions and food_positions[p] > 0:
                food_positions[p] -= 1
                pheromone[p] += 2.0  # Сильный феромон у еды

        # Испарение
        for j in range(n_cells):
            pheromone[j] *= 0.95

    print(f"  Начальное положение: все муравьи в центре (ячейка {n_cells // 2})")
    print(f"  Еда: ячейки 5 и 15 (по 10 единиц)")
    print(f"\n  Феромонная карта после 50 шагов:")
    max_pher = max(pheromone)
    row_str = "    "
    for p in pheromone:
        if max_pher > 0:
            norm = p / max_pher
        else:
            norm = 0
        if norm < 0.1:
            row_str += "·"
        elif norm < 0.3:
            row_str += "░"
        elif norm < 0.5:
            row_str += "▒"
        elif norm < 0.7:
            row_str += "▓"
        else:
            row_str += "█"
    print(row_str)

    # Оставшаяся еда
    total_food_left = sum(food_positions.values())
    print(f"\n  Оставшаяся еда: {total_food_left} из 20")
    print(f"  Муравьи нашли еду через стигмергию (феромонные следы)!")

    # Финальное положение муравьёв
    final_dist = collections.Counter(ant_positions)
    print(f"\n  Распределение муравьёв:")
    for pos in sorted(final_dist.keys()):
        print(f"    Ячейка {pos:2d}: {'█' * final_dist[pos]} ({final_dist[pos]} муравьёв)")


# ============================================================
# ЗАПУСК ВСЕХ ДЕМО
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  192 — Emergent Behavior: самоорганизация, паттерны, коллективный интеллект ║")
    print("╚" + "═" * 68 + "╝")

    demo_self_organization()
    demo_pattern_formation()
    demo_collective_intelligence()
    demo_stigmergy()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМО ЗАВЕРШЕНЫ: Emergent Behavior")
    print("=" * 70)
