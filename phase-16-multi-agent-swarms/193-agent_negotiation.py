"""193 — Agent Negotiation: торги, посредничество, контрактные сети

Темы:
  1. Bargaining (торги Нэша, чередование Рубинштейна, стратегии уступок)
  2. Mediation (третья сторона, арбитраж, разрешение конфликтов)
  3. Contract Net Protocol (объявление, подача заявок, награждение, исполнение)
  4. Multi-Issue Negotiation (компромиссы, пакетные сделки, Парето-улучшения)

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
# ДЕМО 1: Торги (Bargaining)
# ============================================================

def demo_bargaining():
    """Торги Нэша, чередование Рубинштейна, стратегии уступок."""
    print("=" * 70)
    print("ДЕМО 1: ТОРГИ (Bargaining) — Нэш, Рубинштейн, уступки")
    print("=" * 70)

    # --- 1a. Торги Нэша: двухсторонние торги ---
    print("\n--- 1a. Торги Нэша: двухсторонние торги ---")
    # Модель Нэша: делёж «пирога» размером 1
    # Каждый игрок требует долю x₁ и x₂. Сделка, если x₁ + x₂ ≤ 1
    # Точка Нэша: (x₁*, x₂*) = (0.5, 0.5) при равной силе

    # Расширение: сила переговоров определяется «угрозной альтернативой» (BATNA)
    batna_a = 0.2  # Лучшая альтернатива игрока A (минимальнаяacceptable доля)
    batna_b = 0.3  # Лучшая альтернатива игрока B

    # Функция переговоров Нэша: max (x₁ - batna₁)(x₂ - batna₂)
    # при x₁ + x₂ = 1
    # Решение: x₁ = (1 - batna₁ + batna₂) / 2
    nash_x = (1 - batna_a + batna_b) / 2
    nash_y = 1 - nash_x

    print(f"  Пирог: 1.0 (единый ресурс для дележа)")
    print(f"  BATNA игрока A: {batna_a} (минимумacceptable)")
    print(f"  BATNA игрока B: {batna_b}")
    print(f"  Решение Нэша:")
    print(f"    Доля A = (1 - {batna_a} + {batna_b}) / 2 = {nash_x:.4f}")
    print(f"    Доля B = 1 - {nash_x:.4f} = {nash_y:.4f}")
    print(f"    Выигрыш A: {nash_x:.4f} (выше BATNA {batna_a})")
    print(f"    Выигрыш B: {nash_y:.4f} (выше BATNA {batna_b})")

    # Проверка Парето-оптимальности
    print(f"\n  Проверка: сумма = {nash_x + nash_y:.4f} (= 1.0, Парето-оптимально)")

    # --- 1b. Чередование Рубинштейна ---
    print("\n--- 1b. Чередование Рубинштейна: альтернативные предложения ---")
    # Два агента по очереди делают предложения
    # Кто не соглашается — теряетδ (штраф за задержку)

    total_value = 100
    delta_a = 0.8  # Коэффициент обесценивания для A
    delta_b = 0.7  # Коэффициент обесценивания для B

    print(f"  Общая стоимость: {total_value}")
    print(f"  Коэффициент обесценивания A: δ_A = {delta_a} (теряет 20% за раунд)")
    print(f"  Коэффициент обесценивания B: δ_B = {delta_b} (теряет 30% за раунд)")

    # Раунды торга
    print(f"\n  {'Раунд':<8} | {'Предложение A':<16} | {'Предложение B':<16} | {'Состояние'}")
    print(f"  {'-'*8}-+-{'-'*16}-+-{'-'*16}-+-{'-'*15}")

    proposals = []
    for round_num in range(1, 8):
        # A предлагает: себе большую долю
        ask_a = total_value * (0.6 / (delta_a ** (round_num - 1)))
        ask_b = total_value - ask_a

        # B считает, что A завышает → B требует больше
        counter_b = total_value * (0.65 / (delta_b ** (round_num - 1)))
        counter_a = total_value - counter_b

        a_proposal = f"A={ask_a:.1f}, B={ask_b:.1f}"
        b_proposal = f"A={counter_a:.1f}, B={counter_b:.1f}"

        # Проверка: может ли B согласиться?
        if counter_a >= total_value * delta_b ** (round_num - 1) * 0.4:
            proposals.append((round_num, a_proposal, b_proposal, "B соглашается!"))
            break
        else:
            proposals.append((round_num, a_proposal, b_proposal, "отказ"))

    for rnd, prop_a, prop_b, status in proposals:
        print(f"  {rnd:<8} | {prop_a:<16} | {prop_b:<16} | {status}")

    # Аналитическое решение (бесконечные торги)
    if delta_a < 1 and delta_b < 1:
        # Формула Рубинштейна: x* = (1 - δ_B) / (1 - δ_A * δ_B)
        x_star = (1 - delta_b) / (1 - delta_a * delta_b)
        print(f"\n  Аналитическое решение (бесконечные торги):")
        print(f"  x* = (1 - δ_B) / (1 - δ_A · δ_B)")
        print(f"     = (1 - {delta_b}) / (1 - {delta_a} · {delta_b})")
        print(f"     = {x_star:.4f}")
        print(f"  A получает: {x_star * total_value:.2f}")
        print(f"  B получает: {(1 - x_star) * total_value:.2f}")

    # --- 1c. Стратегия уступок ---
    print("\n--- 1c. Стратегии уступок: Кондик, Болдуин-Лиски ---")
    # Кондик: уступки уменьшаются со временем
    # Baldwin-Liskov: линейные уступки

    print("  Стратегия Кондика ( diminishing concessions):")
    print(f"  {'Раунд':<8} | {'Запрос A':<12} | {'Запрос B':<12} | {'Разрыв'}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    a_demand = 80.0
    b_demand = 20.0
    for rnd in range(1, 8):
        gap = a_demand - b_demand
        print(f"  {rnd:<8} | {a_demand:<12.1f} | {b_demand:<12.1f} | {gap:<10.1f}")
        # Кондик: уступка пропорциональна оставшемуся разрыву
        concession = gap * 0.2
        a_demand -= concession
        b_demand += concession

    final_deal = (a_demand + b_demand) / 2
    print(f"\n  Итоговая сделка: A={final_deal:.2f}, B={100 - final_deal:.2f}")

    # --- 1d. Батна (BATNA) и переговорная сила ---
    print("\n--- 1d. BATNA и переговорная сила ---")
    # Модель: переговорная сила ∝ BATNA
    scenarios = [
        ("Квартира: продавец (низкий BATNA)", 0.1, 0.9),
        ("Квартира: покупатель (высокий BATNA)", 0.7, 0.3),
        ("Зарплата: начальник (сильная позиция)", 0.2, 0.8),
        ("Зарплата: программист (уникальные навыки)", 0.8, 0.2),
    ]

    print(f"  {'Сценарий':<40} | {'Доля A':<8} | {'Доля B':<8} | {'BATNA_A':<10} | {'BATNA_B':<10}")
    print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    for name, batna_a, batna_b in scenarios:
        # Решение Нэша
        deal_a = (1 - batna_a + batna_b) / 2
        deal_b = 1 - deal_a
        print(f"  {name:<40} | {deal_a:<8.3f} | {deal_b:<8.3f} | {batna_a:<10.2f} | {batna_b:<10.2f}")

    print(f"\n  Вывод: лучшая альтернатива (BATNA) определяет переговорную силу")
    print(f"  Кто имеет лучший BATNA — получает бóльшую долю")


# ============================================================
# ДЕМО 2: Медиация (Mediation)
# ============================================================

def demo_mediation():
    """Третья сторона, арбитраж, разрешение конфликтов."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: МЕДИАЦИЯ (Mediation) — третья сторона, арбитраж")
    print("=" * 70)

    # --- 2a. Модель медиации: третья сторона ---
    print("\n--- 2a. Модель медиации: влияние медиатора ---")
    # Три стороны: A, B и медиатор M
    # A и B не могут договориться → M предлагает компромисс

    # Позиции A и B
    pos_a = 80  # A хочет 80 из 100
    pos_b = 30  # B хочет 80 из 100 (т.е. B получит 100-30=70)

    # BATNA каждой стороны
    batna_a = 40
    batna_b = 50

    # Медиатор: находит «справедливый» компромисс
    # Метод: середина между позициями, ограниченная BATNA
    mediator_proposal = max(batna_a, min(pos_a, (pos_a + pos_b) / 2))

    print(f"  Позиция A: хочет {pos_a} из 100")
    print(f"  Позиция B: хочет {100 - pos_b} из 100 (B получит {pos_b})")
    print(f"  BATNA A: {batna_a}, BATNA B: {batna_b}")
    print(f"  Без медиации: сделка невозможна (A > B)")
    print(f"  Предложение медиатора: A получает {mediator_proposal:.0f}")
    print(f"  Выигрыш A: {mediator_proposal} (vs BATNA {batna_a})")
    print(f"  Выигрыш B: {100 - mediator_proposal:.0f} (vs BATNA {batna_b})")

    # --- 2b. Арбитраж: принудительное решение ---
    print("\n--- 2b. Арбитраж: принудительное решение ---")
    # Арбитр выносит решение, обязательное для обеих сторон

    disputes = [
        {"description": "Трудовой спор", "claim_a": 50000, "claim_b": 20000, "evidence_a": 0.7, "evidence_b": 0.3},
        {"description": "Спор о патенте", "claim_a": 1000000, "claim_b": 300000, "evidence_a": 0.4, "evidence_b": 0.6},
        {"description": "Страховой случай", "claim_a": 15000, "claim_b": 5000, "evidence_a": 0.8, "evidence_b": 0.2},
    ]

    print(f"  {'Спор':<20} | {'Претензия A':<14} | {'Претензия B':<14} | {'Вес A':<8} | {'Вес B':<8} | {'Решение'}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for d in disputes:
        # Решение арбитра: взвешенное среднее
        total = d["evidence_a"] + d["evidence_b"]
        award = d["claim_a"] * d["evidence_a"] / total + d["claim_b"] * d["evidence_b"] / total
        print(f"  {d['description']:<20} | ${d['claim_a']:<13,} | ${d['claim_b']:<13,} | {d['evidence_a']:<8.1f} | {d['evidence_b']:<8.1f} | ${award:,.0f}")

    # --- 2c. Конфликт интересов: модель игр ---
    print("\n--- 2c. Конфликт интересов: «Дилемма заключённого» с медиацией ---")
    # Стандартная дилемма: ( cooperate, defect )
    # Payoff matrix
    R = 3  # Награда за взаимное сотрудничество
    S = 0  # Страдание: ты сотрудничаешь, партнёр — нет
    T = 5  # Соблазн: ты предаёшь, партнёр — нет
    P = 1  # Наказание: оба предают

    print(f"  Матрица выигрышей:")
    print(f"  {'':>15} | {'B: cooperate':>14} | {'B: defect':>14}")
    print(f"  {'-'*15}-+-{'-'*14}-+-{'-'*14}")
    print(f"  {'A: cooperate':>15} | ({R}, {R}){'':<8} | ({S}, {T}){'':<8}")
    print(f"  {'A: defect':>15} | ({T}, {S}){'':<8} | ({P}, {P}){'':<8}")

    # Стратегии
    n_rounds = 20
    # A: тит-форт-тат (TFT)
    # B: всегда предаёт
    a_history = []
    b_history = []
    a_payoff = 0
    b_payoff = 0

    for r in range(n_rounds):
        if r == 0:
            a_move = "coop"
            b_move = "defect"
        else:
            # TFT: делать то же, что партнёр на предыдущем ходу
            a_move = b_history[-1]
            b_move = "defect"

        a_history.append(a_move)
        b_history.append(b_move)

        if a_move == "coop" and b_move == "coop":
            a_payoff += R
            b_payoff += R
        elif a_move == "coop" and b_move == "defect":
            a_payoff += S
            b_payoff += T
        elif a_move == "defect" and b_move == "coop":
            a_payoff += T
            b_payoff += S
        else:
            a_payoff += P
            b_payoff += P

    print(f"\n  {n_rounds} раундов: A использует TFT, B — всегда предаёт")
    print(f"  Стратегия A: {a_history[:10]}...")
    print(f"  Стратегия B: {b_history[:10]}...")
    print(f"  Итого: A = {a_payoff}, B = {b_payoff}")
    print(f"  → B выигрывает, но при кооперации оба получили бы больше")

    # С медиацией: медиатор обязывает кооперировать
    coop_payoff = R * n_rounds
    print(f"\n  С медиацией (обе стороны Coop): A = {coop_payoff}, B = {coop_payoff}")
    print(f"  Выигрыш медиации: A +{coop_payoff - a_payoff}, B +{coop_payoff - b_payoff}")

    # --- 2d. Мультиагентная медиация ---
    print("\n--- 2d. Мультиагентная медиация: разрешение споров ---")
    n_agents = 6
    agents = [f"Agent_{i}" for i in range(n_agents)]
    # Каждый агент претендует на ресурс
    resource_value = 100
    claims = {a: random.randint(10, 50) for a in agents}

    print(f"  Агенты: {agents}")
    print(f"  Претензии: {claims}")

    # Медиатор: пропорциональное распределение
    total_claims = sum(claims.values())
    allocations = {a: resource_value * claims[a] / total_claims for a in agents}

    print(f"\n  Распределение медиатором (пропорционально претензиям):")
    for a in sorted(allocations, key=allocations.get, reverse=True):
        bar = "█" * int(allocations[a] / 2)
        print(f"    {a}: {allocations[a]:.2f} ({claims[a]} претензий) {bar}")

    # Проверка: сумма = resource_value
    print(f"\n  Сумма: {sum(allocations.values()):.2f} (= {resource_value})")
    print(f"  → Медиатор справедливо распределил ресурс")


# ============================================================
# ДЕМО 3: Contract Net Protocol
# ============================================================

def demo_contract_net():
    """Объявление, подача заявок, награждение, исполнение."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: CONTRACT NET PROTOCOL — объявление, заявки, награждение")
    print("=" * 70)

    # --- 3a. Базовый CnP ---
    print("\n--- 3a. Contract Net Protocol: базовый цикл ---")
    # Менеджер (заказчик) → объявление задачи
    # Подрядчики (рабочие) → подача заявок
    # Менеджер → выбор лучшей заявки
    # Подрядчик → исполнение

    manager_task = {
        "id": "TASK-001",
        "description": "Обработка данных клиентов",
        "deadline": 300,
        "budget": 500,
        "requirements": ["CPU≥4", "RAM≥8GB"]
    }

    # Подрядчики
    contractors = [
        {"id": "C1", "capacity": "CPU=8, RAM=16GB", "cost_per_hour": 15, "reliability": 0.95},
        {"id": "C2", "capacity": "CPU=4, RAM=8GB", "cost_per_hour": 10, "reliability": 0.90},
        {"id": "C3", "capacity": "CPU=16, RAM=32GB", "cost_per_hour": 25, "reliability": 0.98},
        {"id": "C4", "capacity": "CPU=2, RAM=4GB", "cost_per_hour": 5, "reliability": 0.80},
    ]

    print(f"  Шаг 1 — Менеджер объявляет задачу:")
    print(f"    ID: {manager_task['id']}")
    print(f"    Описание: {manager_task['description']}")
    print(f"    Дедлайн: {manager_task['deadline']}с, Бюджет: ${manager_task['budget']}")
    print(f"    Требования: {manager_task['requirements']}")

    print(f"\n  Шаг 2 — Подрядчики подают заявки:")
    bids = []
    for c in contractors:
        # Проверка: подходит ли подрядчик?
        has_cpu = "CPU=8" in c["capacity"] or "CPU=16" in c["capacity"]
        has_ram = "RAM=8GB" in c["capacity"] or "RAM=16GB" in c["capacity"] or "RAM=32GB" in c["capacity"]

        if has_cpu and has_ram:
            # Расчёт стоимости (8 часов работы)
            hours = 8
            cost = c["cost_per_hour"] * hours
            if cost <= manager_task["budget"]:
                # Скоринг: надёжность / стоимость
                score = c["reliability"] / cost
                bids.append({
                    "contractor": c["id"],
                    "cost": cost,
                    "hours": hours,
                    "reliability": c["reliability"],
                    "score": score
                })
                print(f"    Заявка от {c['id']}: ${cost}, {hours}ч, надёжность={c['reliability']}")
        else:
            print(f"    {c['id']}: отклонён (не соответствует требованиям)")

    print(f"\n  Шаг 3 — Менеджер выбирает лучшую заявку:")
    best_bid = max(bids, key=lambda b: b["score"])
    print(f"    Победитель: {best_bid['contractor']}")
    print(f"    Стоимость: ${best_bid['cost']}")
    print(f"    Надёжность: {best_bid['reliability']}")
    print(f"    Скор (надёжность/стоимость): {best_bid['score']:.4f}")

    print(f"\n  Шаг 4 — Исполнение:")
    print(f"    {best_bid['contractor']} выполняет задачу за {best_bid['hours']}ч")
    success = random.random() < best_bid["reliability"]
    print(f"    Результат: {'УСПЕХ' if success else 'НЕУДАЧА'}")

    # --- 3b. Многозадачный CnP ---
    print("\n--- 3b. Многозадачный CnP: распределение 5 задач ---")
    tasks = [
        {"id": f"T{i}", "complexity": random.randint(1, 10), "deadline": random.randint(100, 500)}
        for i in range(5)
    ]

    print(f"  Задачи:")
    for t in tasks:
        print(f"    {t['id']}: сложность={t['complexity']}, дедлайн={t['deadline']}с")

    # Распределение задач (жадный алгоритм)
    assignments = {c["id"]: [] for c in contractors}
    contractor_load = {c["id"]: 0 for c in contractors}

    for task in sorted(tasks, key=lambda t: t["complexity"], reverse=True):
        # Выбираем наименее загруженного способного подрядчика
        best_c = None
        min_load = float('inf')
        for c in contractors:
            if contractor_load[c["id"]] + task["complexity"] <= 20:
                if contractor_load[c["id"]] < min_load:
                    min_load = contractor_load[c["id"]]
                    best_c = c["id"]
        if best_c:
            assignments[best_c].append(task["id"])
            contractor_load[best_c] += task["complexity"]

    print(f"\n  Распределение задач:")
    for c_id, tasks_list in assignments.items():
        if tasks_list:
            print(f"    {c_id}: {tasks_list} (загрузка: {contractor_load[c_id]})")

    # --- 3c. Конкуренция подрядчиков ---
    print("\n--- 3c. Конкуренция подрядчиков: аукцион ---")
    # Английский аукцион: цена растёт
    item = "Проект аналитики"
    start_price = 100
    price = start_price
    bidders = ["Alpha Corp", "Beta Inc", "Gamma LLC"]
    active_bidders = bidders[:]

    print(f"  Товар: {item}")
    print(f"  Начальная цена: ${start_price}")
    print(f"  Участники: {active_bidders}")

    print(f"\n  {'Раунд':<8} | {'Цена':<10} | {'Осталось':<15} | {'Снявшиеся'}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*15}-+-{'-'*15}")

    for rnd in range(1, 10):
        # Случайный подрядчик снимается
        if len(active_bidders) > 1 and random.random() < 0.3:
            quitter = random.choice(active_bidders)
            active_bidders.remove(quitter)
            print(f"  {rnd:<8} | ${price:<9} | {', '.join(active_bidders):<15} | {quitter}")
        else:
            print(f"  {rnd:<8} | ${price:<9} | {', '.join(active_bidders):<15} | —")

        price += random.randint(5, 20)

        if len(active_bidders) <= 1:
            break

    if active_bidders:
        print(f"\n  Победитель: {active_bidders[0]} за ${price}")

    # --- 3d. CnP с репутацией ---
    print("\n--- 3d. CnP с учётом репутации ---")
    contractor_reputation = {"C1": 0.92, "C2": 0.78, "C3": 0.97, "C4": 0.65}
    completed_tasks = {"C1": 45, "C2": 30, "C3": 60, "C4": 15}

    print(f"  Репутация подрядчиков:")
    for c_id in contractor_reputation:
        rep = contractor_reputation[c_id]
        tasks_done = completed_tasks[c_id]
        # Взвешенная репутация: репутация * log(задач + 1)
        weighted = rep * math.log(tasks_done + 1)
        print(f"    {c_id}: репутация={rep:.2f}, задач={tasks_done}, "
              f"взвешенная={weighted:.2f}")

    # Выбор подрядчика с учётом репутации
    best_c = max(contractor_reputation,
                 key=lambda c: contractor_reputation[c] * math.log(completed_tasks[c] + 1))
    print(f"\n  Лучший подрядчик (по взвешенной репутации): {best_c}")
    print(f"  → Учёт репутации предотвращает мошенничество и стимулирует качество")


# ============================================================
# ДЕМО 4: Multi-Issue Negotiation
# ============================================================

def demo_multi_issue():
    """Компромиссы, пакетные сделки, Парето-улучшения."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: MULTI-ISSUE NEGOTIATION — компромиссы, пакеты, Парето")
    print("=" * 70)

    # --- 4a. Двухсторонние торги по нескольким вопросам ---
    print("\n--- 4a. Торги по 3 вопросам: цена, срок, качество ---")
    # Вопросы: цена ($), срок (дни), качество (1-10)
    issues = ["Цена ($)", "Срок (дни)", "Качество (1-10)"]

    # Предпочтения стороны A (нормализованные веса)
    pref_a = {"Цена ($)": 0.5, "Срок (дни)": 0.2, "Качество (1-10)": 0.3}
    pref_b = {"Цена ($)": 0.3, "Срок (дни)": 0.5, "Качество (1-10)": 0.2}

    # Диапазоны
    ranges = {
        "Цена ($)": (50, 150),
        "Срок (дни)": (7, 30),
        "Качество (1-10)": (3, 10)
    }

    print(f"  Вопросы: {issues}")
    print(f"  Веса A: {pref_a}")
    print(f"  Веса B: {pref_b}")
    print(f"  Диапазоны: {ranges}")

    # Парето-оптимальные решения: 10 вариантов
    pareto_front = []
    for _ in range(100):
        # Случайная точка
        offer = {}
        for issue in issues:
            low, high = ranges[issue]
            offer[issue] = random.uniform(low, high)

        # Выигрыш A: нормализованный
        utility_a = sum(pref_a[i] * (offer[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0])
                       for i in issues)
        # Выигрыш B: обратный (A хочет дешевле, быстрее, качественнее)
        utility_b = sum(pref_b[i] * (1 - (offer[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]))
                       for i in issues)

        pareto_front.append((offer, utility_a, utility_b))

    # Фильтрация Парето-фронт
    pareto_optimal = []
    for i, (o_i, ua_i, ub_i) in enumerate(pareto_front):
        dominated = False
        for j, (o_j, ua_j, ub_j) in enumerate(pareto_front):
            if i != j and ua_j >= ua_i and ub_j >= ub_i and (ua_j > ua_i or ub_j > ub_i):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append((o_i, ua_i, ub_i))

    pareto_optimal.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Парето-оптимальные решения (топ-5):")
    print(f"  {'Вариант':<8} | {'Цена':<10} | {'Срок':<8} | {'Качество':<10} | {'U_A':<8} | {'U_B':<8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for idx, (offer, ua, ub) in enumerate(pareto_optimal[:5]):
        print(f"  {idx+1:<8} | {offer['Цена ($)']:<10.1f} | {offer['Срок (дни)']:<8.1f} | "
              f"{offer['Качество (1-10)']:<10.1f} | {ua:<8.3f} | {ub:<8.3f}")

    # --- 4b. Торги по одной стороне ---
    print("\n--- 4b. Неравномерная важность: асимметричные предпочтения ---")
    # У A цена важнее, у B — срок
    # Компромисс: A уступает по цене, B — по сроку

    best_offer = max(pareto_optimal, key=lambda x: x[1] + x[2])
    print(f"  A ценит цену (0.5), B ценит срок (0.5)")
    print(f"  Лучший компромисс (макс. сумма выигрышей):")
    for issue in issues:
        print(f"    {issue}: {best_offer[0][issue]:.1f}")
    print(f"  U(A) = {best_offer[1]:.3f}, U(B) = {best_offer[2]:.3f}")

    # --- 4c. Пакетная сделка ---
    print("\n--- 4c. Пакетная сделка (package deal) ---")
    # 4 опции для торга
    options = [
        {"name": "Базовый", "price": 100, "quality": 5, "delivery": 14},
        {"name": "Стандарт", "price": 200, "quality": 7, "delivery": 10},
        {"name": "Премиум", "price": 350, "quality": 9, "delivery": 7},
        {"name": "Экспресс", "price": 500, "quality": 10, "delivery": 3},
    ]

    # A: бюджет ограничен, хочет лучшее качество
    # B: хочет быструю доставку
    a_budget = 300
    b_delivery_need = 8

    print(f"  Бюджет A: ${a_budget}")
    print(f"  Нужная скорость B: ≤{b_delivery_need} дней")
    print(f"\n  {'Опция':<12} | {'Цена':<8} | {'Качество':<10} | {'Срок':<8} | {'A годится':<12} | {'B годится'}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}")

    viable = []
    for opt in options:
        a_ok = "✓" if opt["price"] <= a_budget else "✗"
        b_ok = "✓" if opt["delivery"] <= b_delivery_need else "✗"
        if a_ok == "✓" and b_ok == "✓":
            viable.append(opt)
        print(f"  {opt['name']:<12} | ${opt['price']:<7} | {opt['quality']:<10} | "
              f"{opt['delivery']:<8} | {a_ok:<12} | {b_ok}")

    print(f"\n  Возможные пакетные сделки: {[v['name'] for v in viable]}")
    if viable:
        best = max(viable, key=lambda o: o["quality"])
        print(f"  Лучший вариант: {best['name']} "
              f"(${best['price']}, качество={best['quality']}, срок={best['delivery']}д)")

    # --- 4d. Парето-улучшение через обмен ---
    print("\n--- 4d. Парето-улучшение: обмен уступками ---")
    # Начальная сделка: A получает 60, B получает 40
    current_a = 60
    current_b = 40
    print(f"  Начальная сделка: A={current_a}, B={current_b} (сумма={current_a + current_b})")

    # Парето-улучшение: A уступает 5, B уступает 5 → оба в выигрыше
    improvements = [
        (70, 50, "A жертвует 10, B жертвует 10 → оба получают больше"),
        (65, 45, "A жертвует 5, B жертвует 5"),
        (80, 30, "A получает больше, B меньше"),
        (55, 55, "Равный делёж"),
    ]

    print(f"\n  {'Вариант':<8} | {'A':<8} | {'B':<8} | {'Парето?':<10} | {'Комментарий'}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*40}")

    for new_a, new_b, comment in improvements:
        # Парето-улучшение: новый хуже текущего для кого-то
        pareto = "✓" if (new_a >= current_a and new_b >= current_b) else "✗"
        print(f"  {improvements.index((new_a, new_b, comment))+1:<8} | {new_a:<8} | {new_b:<8} | {pareto:<10} | {comment}")

    # Поиск лучшей Парето-точки
    best_improvement = max(improvements, key=lambda x: x[0] + x[1])
    print(f"\n  Лучшая точка: A={best_improvement[0]}, B={best_improvement[1]}")
    print(f"  → Через взаимные уступки можно улучшить положение обеих сторон")


# ============================================================
# ЗАПУСК ВСЕХ ДЕМО
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  193 — Agent Negotiation: торги, посредничество, контрактные сети        ║")
    print("╚" + "═" * 68 + "╝")

    demo_bargaining()
    demo_mediation()
    demo_contract_net()
    demo_multi_issue()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМО ЗАВЕРШЕНЫ: Agent Negotiation")
    print("=" * 70)
