"""175 — Self-Improvement: мета-обучение, эволюция промптов, самооптимизация

Темы:
  1. Meta-Learning (обучение обучению, стратегии адаптации few-shot)
  2. Prompt Evolution (генетические алгоритмы для промптов, мутация, селекция)
  3. Self-Optimization (отслеживание производительности, настройка параметров, авто-тюнинг)
  4. Reflection Loops (критика вывода, уточнение стратегий, синтез опыта)

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
# 1. МЕТА-ОБУЧЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def demo_meta_learning():
    """Мета-обучение: обучение обучению, few-shot адаптация."""
    print("=" * 70)
    print("ДЕМО 1: META-LEARNING — обучение обучению")
    print("=" * 70)

    # --- 1a. MAML-style: быстрая адаптация к новой задаче ---
    print("\n--- 1a. MAML-style: быстрая адаптация к новой задаче ---")
    # Базовая модель: y = w * x + b
    # Мета-обучение: инициализация, которая адаптируется за 1 шаг
    random.seed(42)
    w_meta, b_meta = 1.0, 0.5  # мета-параметры

    # Внутренний шаг (1 gradient step на новой задаче)
    def inner_step(w, b, data_x, data_y, lr_inner=0.1):
        """Один шаг градиентного спуска на данных задачи."""
        preds = [w * x + b for x in data_x]
        errors = [p - y for p, y in zip(preds, data_y)]
        n = len(data_x)
        grad_w = 2.0 / n * sum(e * x for e, x in zip(errors, data_x))
        grad_b = 2.0 / n * sum(errors)
        return w - lr_inner * grad_w, b - lr_inner * grad_b

    # 5 мета-задач
    tasks = []
    for i in range(5):
        true_w = random.uniform(0.5, 3.0)
        true_b = random.uniform(-1, 1)
        xs = [random.uniform(-2, 2) for _ in range(10)]
        ys = [true_w * x + true_b + random.gauss(0, 0.1) for x in xs]
        tasks.append((xs, ys, true_w, true_b))

    print(f"Начальные мета-параметры: w={w_meta:.3f}, b={b_meta:.3f}")
    for i, (xs, ys, tw, tb) in enumerate(tasks):
        w_adapted, b_adapted = inner_step(w_meta, b_meta, xs, ys)
        print(f"  Задача {i+1} (true: w={tw:.2f}, b={tb:.2f}): "
              f"адаптировано -> w={w_adapted:.3f}, b={b_adapted:.3f}")

    # --- 1b. Few-shot классификация: k-NN в мета-пространстве ---
    print("\n--- 1b. Few-shot классификация (k-NN) ---")
    # Задача: классифицировать точки 2D по расстоянию до эталонов
    random.seed(42)
    prototypes = {
        "A": [(1.0, 1.0), (1.2, 0.8)],   # 2 прототипа класса A
        "B": [(3.0, 3.0), (2.8, 3.2)],   # 2 прототипа класса B
        "C": [(0.5, 3.0), (0.8, 2.8)],   # 2 прототипа класса C
    }

    def dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    test_points = [(1.1, 0.9), (2.9, 3.1), (0.6, 2.9), (2.0, 2.0)]
    print("Прототипы (2-shot):")
    for cls, pts in prototypes.items():
        print(f"  Класс {cls}: {pts}")

    for pt in test_points:
        # Среднее расстояние до прототипов каждого класса
        best_cls = None
        best_dist = float("inf")
        for cls, pts in prototypes.items():
            avg_d = sum(dist(pt, p) for p in pts) / len(pts)
            if avg_d < best_dist:
                best_dist = avg_d
                best_cls = cls
        print(f"  Точка {pt} -> класс {best_cls} (dist={best_dist:.3f})")

    # --- 1c. Адаптивное обучение: learning rate scheduling ---
    print("\n--- 1c. Адаптивный learning rate (AdaGrad-style) ---")
    random.seed(42)
    w, b = 0.0, 0.0
    data = [(x, 2.0 * x + 1.0 + random.gauss(0, 0.5)) for x in [random.uniform(-3, 3) for _ in range(40)]]
    grad_sum_w, grad_sum_b = 1e-8, 1e-8  # накопленные квадраты градиентов
    lr_base = 0.5
    print(f"Итерация | MSE     | lr_w      | lr_b")
    print("-" * 50)
    for epoch in range(8):
        preds = [w * x + b for x, _ in data]
        errors = [p - y for p, (_, y) in zip(preds, data)]
        mse = sum(e * e for e in errors) / len(errors)
        n = len(data)
        gw = 2.0 / n * sum(e * x for e, (x, _) in zip(errors, data))
        gb = 2.0 / n * sum(errors)
        grad_sum_w += gw * gw
        grad_sum_b += gb * gb
        # Адаптивные learning rates
        lr_w = lr_base / math.sqrt(grad_sum_w)
        lr_b = lr_base / math.sqrt(grad_sum_b)
        w -= lr_w * gw
        b -= lr_b * gb
        print(f"  {epoch+1:4d}    | {mse:.4f}  | {lr_w:.6f}  | {lr_b:.6f}")

    # --- 1d. Мета-стратегия: выбор алгоритма по характеристикам задачи ---
    print("\n--- 1d. Мета-стратегия: выбор алгоритма ---")
    random.seed(42)

    def task_features(xs, ys):
        """Извлечение признаков задачи."""
        n = len(xs)
        mean_x = sum(xs) / n
        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        mean_y = sum(ys) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n
        # Линейная корреляция
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
        corr = cov / (math.sqrt(var_x * var_y) + 1e-12)
        return {"n": n, "var_x": var_x, "var_y": var_y, "corr": corr}

    def choose_algorithm(features):
        """Правило выбора: если корреляция высокая -> линейная регрессия,
        если мало данных -> константа, иначе -> k-NN."""
        if abs(features["corr"]) > 0.8:
            return "Линейная регрессия"
        elif features["n"] < 5:
            return "Константная модель"
        else:
            return "k-NN (k=3)"

    for i in range(4):
        n_samples = random.choice([3, 10, 30, 50])
        xs = [random.uniform(-5, 5) for _ in range(n_samples)]
        corr_type = random.choice(["linear", "noisy"])
        if corr_type == "linear":
            ys = [2.0 * x + 1.0 + random.gauss(0, 0.3) for x in xs]
        else:
            ys = [random.gauss(0, 5) for _ in xs]
        feat = task_features(xs, ys)
        algo = choose_algorithm(feat)
        print(f"  Задача {i+1}: n={feat['n']}, corr={feat['corr']:.3f} -> {algo}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ЭВОЛЮЦИЯ ПРОМПТОВ
# ══════════════════════════════════════════════════════════════════════════════

def demo_prompt_evolution():
    """Генетические алгоритмы для эволюции промптов."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: PROMPT EVOLUTION — эволюция промптов")
    print("=" * 70)

    # --- 2a. Генетическое представление промптов ---
    print("\n--- 2a. Генетическое представление промптов ---")
    # Промпт как набор модулей (гены)
    genes = {
        "роль": ["Ты эксперт.", "Ты помощник.", "Ты аналитик."],
        "формат": ["Ответь кратко.", "Дай подробный ответ.", "Используй списки."],
        "контекст": ["Для начинающих.", "Для профессионалов.", "Академический стиль."],
        "примеры": ["Без примеров.", "С 2 примерами.", "С аналогиями."],
    }

    def create_individual(gene_pool):
        """Создание особи (промпта) из набора генов."""
        return {gene: random.choice(options) for gene, options in gene_pool.items()}

    random.seed(42)
    population = [create_individual(genes) for _ in range(6)]
    print("Начальная популяция (6 особей):")
    for i, ind in enumerate(population):
        prompt = " | ".join(f"{k}: {v}" for k, v in ind.items())
        print(f"  [{i}] {prompt}")

    # --- 2b. Фитнес-функция ---
    print("\n--- 2b. Фитнес-функция промптов ---")

    def fitness(individual):
        """Оценка качества промпта по нескольким критериям."""
        score = 0.0
        # Критерий 1: наличие роли (важно)
        if "эксперт" in individual["роль"] or "аналитик" in individual["роль"]:
            score += 3.0
        # Критерий 2: формат ответа
        if "подробн" in individual["формат"]:
            score += 2.0
        if "списки" in individual["формат"]:
            score += 1.5
        # Критерий 3: контекст
        if "начинающ" in individual["контекст"]:
            score += 1.0
        if "профессионал" in individual["контекст"]:
            score += 2.0
        # Критерий 4: примеры
        if "примерами" in individual["примеры"]:
            score += 2.5
        elif "аналогиями" in individual["примеры"]:
            score += 2.0
        # Штраф за длину (слишком длинные промпты хуже)
        total_len = sum(len(v) for v in individual.values())
        if total_len > 80:
            score -= 1.0
        return score

    for i, ind in enumerate(population):
        f = fitness(ind)
        print(f"  [{i}] фитнес={f:.1f} | {ind['роль'][:15]}... | {ind['формат'][:20]}...")

    # --- 2c. Селекция и скрещивание ---
    print("\n--- 2c. Селекция и скрещивание (Tournament + Crossover) ---")

    def tournament_select(pop, fitnesses, k=3):
        """Турнирная селекция."""
        indices = random.sample(range(len(pop)), k)
        best = max(indices, key=lambda i: fitnesses[i])
        return pop[best]

    def crossover(parent1, parent2):
        """Скрещивание: каждый ген берётся от случайного родителя."""
        child = {}
        for gene in parent1:
            if random.random() < 0.5:
                child[gene] = parent1[gene]
            else:
                child[gene] = parent2[gene]
        return child

    def mutate(individual, gene_pool, mutation_rate=0.2):
        """Мутация: замена гена с вероятностью mutation_rate."""
        mutated = dict(individual)
        for gene in gene_pool:
            if random.random() < mutation_rate:
                mutated[gene] = random.choice(gene_pool[gene])
        return mutated

    random.seed(42)
    fitnesses = [fitness(ind) for ind in population]
    new_population = []
    for _ in range(len(population)):
        p1 = tournament_select(population, fitnesses)
        p2 = tournament_select(population, fitnesses)
        child = crossover(p1, p2)
        child = mutate(child, genes)
        new_population.append(child)

    new_fitnesses = [fitness(ind) for ind in new_population]
    print("Новое поколение после селекции/скрещивания/мутации:")
    for i, ind in enumerate(new_population):
        print(f"  [{i}] фитнес={new_fitnesses[i]:.1f} | {' | '.join(f'{k}: {v[:12]}' for k, v in ind.items())}")

    old_avg = sum(fitnesses) / len(fitnesses)
    new_avg = sum(new_fitnesses) / len(new_fitnesses)
    print(f"\nСредний фитнес: {old_avg:.2f} -> {new_avg:.2f} (изменение: {new_avg - old_avg:+.2f})")

    # --- 2d. Эволюция за 10 поколений ---
    print("\n--- 2d. Эволюция за 10 поколений ---")
    random.seed(42)
    pop = [create_individual(genes) for _ in range(20)]
    history = []

    for gen in range(10):
        fits = [fitness(ind) for ind in pop]
        best_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        history.append((avg_fit, best_fit))

        # Элитизм: сохраняем лучших 2
        elite_indices = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)[:2]
        elites = [pop[i] for i in elite_indices]

        # Новая популяция
        new_pop = list(elites)  # элитизм
        while len(new_pop) < 20:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)
            child = crossover(p1, p2)
            child = mutate(child, genes, mutation_rate=0.3)
            new_pop.append(child)
        pop = new_pop

        print(f"  Поколение {gen+1:2d}: avg={avg_fit:.2f}, best={best_fit:.1f}")

    print(f"\nПрогресс: начальный avg={history[0][0]:.2f} -> финальный avg={history[-1][0]:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. САМООПТИМИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def demo_self_optimization():
    """Самооптимизация: отслеживание, настройка параметров, авто-тюнинг."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: SELF-OPTIMIZATION — самооптимизация")
    print("=" * 70)

    # --- 3a. Отслеживание производительности ---
    print("\n--- 3a. Отслеживание производительности (Performance Tracking) ---")
    random.seed(42)
    # Метрики по эпохам
    metrics = {
        "loss": [],
        "accuracy": [],
        "latency_ms": [],
    }
    for epoch in range(12):
        loss = 2.0 * math.exp(-0.3 * epoch) + random.gauss(0, 0.05)
        acc = min(0.95, 0.5 + 0.4 * (1 - math.exp(-0.5 * epoch)) + random.gauss(0, 0.02))
        latency = 50 + 10 * (1 + 0.1 * epoch) + random.gauss(0, 2)
        metrics["loss"].append(loss)
        metrics["accuracy"].append(acc)
        metrics["latency_ms"].append(latency)

    print(f"{'Эпоха':>5} | {'Loss':>8} | {'Accuracy':>8} | {'Latency':>8}")
    print("-" * 40)
    for e in range(12):
        print(f"  {e+1:3d}  | {metrics['loss'][e]:8.4f} | {metrics['accuracy'][e]:8.4f} | {metrics['latency_ms'][e]:7.1f}ms")

    # --- 3b. Автоматическая настройка learning rate ---
    print("\n--- 3b. Автоматическая настройка learning rate ---")
    random.seed(42)

    def train_with_lr(lr, n_steps=50):
        """Обучение с фиксированным lr, возврат финального loss."""
        w = 0.0
        data = [(x, 3.0 * x + 1.0) for x in [random.uniform(-2, 2) for _ in range(20)]]
        for _ in range(n_steps):
            preds = [w * x for x, _ in data]
            errors = [p - y for p, (_, y) in zip(preds, data)]
            grad = 2.0 / len(data) * sum(e * x for e, (x, _) in zip(errors, data))
            w -= lr * grad
        # Финальный loss
        preds = [w * x for x, _ in data]
        errors = [p - y for p, (_, y) in zip(preds, data)]
        return sum(e * e for e in errors) / len(errors)

    # Поиск лучшего lr
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    print(f"{'lr':>8} | {'Final Loss':>10}")
    print("-" * 25)
    best_lr = None
    best_loss = float("inf")
    for lr in learning_rates:
        loss = train_with_lr(lr)
        print(f"  {lr:6.3f} | {loss:10.4f}")
        if loss < best_loss:
            best_loss = loss
            best_lr = lr

    print(f"\nЛучший lr: {best_lr} (loss={best_loss:.4f})")

    # --- 3c.贝叶с оптимизация (упрощённая) ---
    print("\n--- 3c. Байесовская оптимизация (упрощённая) ---")
    random.seed(42)
    # Целевая функция: f(x) = -(x-2)^2 + 5 (максимум в x=2)
    def objective(x):
        return -((x - 2.0) ** 2) + 5.0 + random.gauss(0, 0.1)

    # Простой surrogate: среднее наблюдений + неопределённость
    observations_x = []
    observations_y = []

    for iteration in range(8):
        if len(observations_x) < 3:
            # Исследование: случайная точка
            x_new = random.uniform(-2, 6)
        else:
            # Эксплуатация: точка с максимальным upper confidence bound
            best_x = None
            best_ucb = -float("inf")
            for _ in range(50):
                x_cand = random.uniform(-2, 6)
                # Среднее и стандартное отклонение по расстоянию
                dists = [abs(x_cand - xo) for xo in observations_x]
                weights = [1.0 / (d + 0.1) for d in dists]
                w_sum = sum(weights)
                mu = sum(w * y for w, y in zip(weights, observations_y)) / w_sum
                sigma = 1.0 / (sum(dists) / len(dists) + 0.1)
                ucb = mu + 1.5 * sigma
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_x = x_cand
            x_new = best_x

        y_new = objective(x_new)
        observations_x.append(x_new)
        observations_y.append(y_new)
        print(f"  Итерация {iteration+1}: x={x_new:.3f}, f(x)={y_new:.3f}")

    best_idx = max(range(len(observations_y)), key=lambda i: observations_y[i])
    print(f"\nЛучшая точка: x={observations_x[best_idx]:.3f}, f(x)={observations_y[best_idx]:.3f}")

    # --- 3d. Автоматический подбор гиперпараметров ---
    print("\n--- 3d. Автоматический подбор гиперпараметров ---")
    random.seed(42)

    def evaluate_config(config):
        """Оценка конфигурации: имитация метрики."""
        lr = config["lr"]
        batch = config["batch_size"]
        dropout = config["dropout"]
        # Имитация: оптимальные значения в середине диапазона
        score = 0.0
        score += -((lr - 0.01) ** 2) * 1000
        score += -((batch - 32) ** 2) / 100
        score += -((dropout - 0.3) ** 2) * 10
        score += random.gauss(0, 0.05)
        return score

    # Случайный поиск
    configs = []
    for _ in range(15):
        config = {
            "lr": 10 ** random.uniform(-4, -1),
            "batch_size": random.choice([8, 16, 32, 64, 128]),
            "dropout": random.uniform(0.1, 0.5),
        }
        score = evaluate_config(config)
        configs.append((score, config))

    configs.sort(reverse=True)
    print("Топ-5 конфигураций:")
    for i, (score, cfg) in enumerate(configs[:5]):
        print(f"  {i+1}. score={score:.4f} | lr={cfg['lr']:.5f}, batch={cfg['batch_size']}, dropout={cfg['dropout']:.3f}")

    print(f"\nЛучшая конфигурация: lr={configs[0][1]['lr']:.5f}, "
          f"batch={configs[0][1]['batch_size']}, dropout={configs[0][1]['dropout']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ПЕТЛИ РЕФЛЕКСИИ
# ══════════════════════════════════════════════════════════════════════════════

def demo_reflection_loops():
    """Петли рефлексии: критика вывода, уточнение стратегий, синтез опыта."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: REFLECTION LOOPS — петли рефлексии")
    print("=" * 70)

    # --- 4a. Критика вывода (Output Critique) ---
    print("\n--- 4a. Критика вывода (Output Critique) ---")
    # Симуляция: модель генерирует ответ, критик оценивает
    random.seed(42)
    outputs = [
        {"text": "Python — язык программирования.", "task": "объясни Python"},
        {"text": "Python很好用。", "task": "объясни Python"},
        {"text": "Python — интерпретируемый, динамически типизированный язык с автоматическим сборщиком мусора, подходящий для веб-разработки, анализа данных и ИИ.",
         "task": "объясни Python"},
    ]

    def critique(output, task):
        """Простая функция критики: оценка по критериям."""
        score = 0.0
        issues = []
        text = output["text"]
        # Критерий 1: релевантность (наличие ключевых слов)
        keywords = ["python", "программ", "язык"]
        kw_score = sum(1 for kw in keywords if kw.lower() in text.lower())
        score += kw_score * 2
        if kw_score < 2:
            issues.append("Мало релевантных ключевых слов")
        # Критерий 2: длина
        if len(text) < 20:
            score -= 2
            issues.append("Слишком короткий ответ")
        elif len(text) > 200:
            score -= 1
            issues.append("Слишком длинный ответ")
        else:
            score += 2
        # Критерий 3: структура
        if "." in text or "," in text:
            score += 1
        # Критерий 4: язык (должен быть русский)
        has_cyrillic = any("а" <= c <= "я" for c in text.lower())
        if has_cyrillic:
            score += 2
        else:
            issues.append("Нет кириллического текста")
            score -= 1
        return score, issues

    for i, out in enumerate(outputs):
        score, issues = critique(out, out["task"])
        print(f"  Вывод {i+1}: \"{out['text'][:50]}...\"")
        print(f"    Оценка: {score:.1f}, Проблемы: {issues if issues else 'нет'}")

    # --- 4b. Итеративное уточнение (Iterative Refinement) ---
    print("\n--- 4b. Итеративное уточнение стратегии ---")
    random.seed(42)
    # Стратегия как вектор параметров
    strategy = {"temperature": 0.7, "top_k": 50, "max_tokens": 100}

    def evaluate_strategy(strategy):
        """Оценка стратегии: имитация качества вывода."""
        t = strategy["temperature"]
        k = strategy["top_k"]
        m = strategy["max_tokens"]
        # Оптимум: t=0.5, k=40, m=150
        score = 0.0
        score += -((t - 0.5) ** 2) * 10
        score += -((k - 40) ** 2) / 100
        score += -((m - 150) ** 2) / 1000
        return score

    print(f"Начальная стратегия: {strategy}")
    print(f"Начальная оценка: {evaluate_strategy(strategy):.4f}")

    for iteration in range(5):
        # Предлагаем изменения
        candidates = []
        for param in strategy:
            for delta in [-0.1, 0.1]:
                new_strat = dict(strategy)
                new_strat[param] = strategy[param] + delta
                candidates.append((new_strat, param, delta))

        # Выбираем лучшее изменение
        best_candidate = None
        best_score = -float("inf")
        for cand, param, delta in candidates:
            score = evaluate_strategy(cand)
            if score > best_score:
                best_score = score
                best_candidate = (cand, param, delta)

        old_score = evaluate_strategy(strategy)
        strategy = best_candidate[0]
        param_changed = best_candidate[1]
        delta = best_candidate[2]
        print(f"  Итерация {iteration+1}: {param_changed} {delta:+.1f} -> оценка {old_score:.4f} -> {best_score:.4f}")

    print(f"\nФинальная стратегия: {strategy}")

    # --- 4c. Синтез опыта (Experience Synthesis) ---
    print("\n--- 4c. Синтез опыта из истории ---")
    random.seed(42)
    # История экспериментов
    experiments = [
        {"attempt": 1, "action": "lr=0.1", "result": "diverged", "metric": -10.0},
        {"attempt": 2, "action": "lr=0.01", "result": "converged_slow", "metric": 0.7},
        {"attempt": 3, "action": "lr=0.001", "result": "converged", "metric": 0.9},
        {"attempt": 4, "action": "lr=0.001, batch=64", "result": "converged", "metric": 0.92},
        {"attempt": 5, "action": "lr=0.001, batch=128", "result": "converged", "metric": 0.91},
        {"attempt": 6, "action": "lr=0.005, batch=32", "result": "converged_fast", "metric": 0.88},
    ]

    # Анализ паттернов
    print("История экспериментов:")
    for exp in experiments:
        print(f"  [{exp['attempt']}] {exp['action']:25s} -> {exp['result']:20s} metric={exp['metric']:.2f}")

    # Синтез: правила из опыта
    successful = [e for e in experiments if e["metric"] > 0.8]
    failed = [e for e in experiments if e["metric"] < 0]

    print("\nСинтезированные правила:")
    print(f"  1. Избегать lr > 0.05 (причина: расхождение в {len(failed)} случаях)")
    print(f"  2. Оптимальный lr ~ 0.001-0.005 ({len(successful)} успешных запуска)")
    print(f"  3. batch=32-64 даёт лучший баланс скорости и качества")

    # --- 4d. Цикл рефлексии (完整 loop) ---
    print("\n--- 4d. Полный цикл рефлексии ---")
    random.seed(42)
    performance_log = []

    for cycle in range(4):
        # Действие
        lr = 0.01 / (cycle + 1)
        # Результат
        metric = 0.5 + 0.1 * cycle + random.gauss(0, 0.03)
        performance_log.append({"cycle": cycle + 1, "lr": lr, "metric": metric})

        # Рефлексия: анализ тренда
        if len(performance_log) >= 2:
            prev = performance_log[-2]["metric"]
            curr = performance_log[-1]["metric"]
            trend = "улучшение" if curr > prev else "ухудшение"
        else:
            trend = "baseline"

        # Корректировка
        if trend == "ухудшение":
            lr = lr * 0.5  # уменьшаем lr
        else:
            lr = lr * 1.1  # увеличиваем lr

        print(f"  Цикл {cycle+1}: lr={performance_log[-1]['lr']:.5f}, "
              f"metric={metric:.4f}, тренд={trend}")

    print("\nИтог: цикл рефлексии позволил адаптировать lr к динамике обучения")


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_meta_learning()
    demo_prompt_evolution()
    demo_self_optimization()
    demo_reflection_loops()
    print("\n" + "=" * 70)
    print("Все демо завершены: Self-Improvement")
    print("=" * 70)
