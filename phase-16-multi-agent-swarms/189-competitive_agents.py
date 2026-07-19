"""
189 — Competitive Agents: теория игр, равновесие Нэша, аукционы

Темы:
  1. Основы теории игр (нормальная форма, матрица выигрышей, доминирующая стратегия)
  2. Равновесие Нэша (чистые/смешанные стратегии, лучший ответ, поиск равновесий)
  3. Аукционные механизмы (английский, голландский, закрытый, VCG)
  4. Конкурентные стратегии (минимакс, максимин, минимизация сожаления)

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
# 1. ОСНОВЫ ТЕОРИИ ИГР
# =============================================================================

def demo_game_theory_basics():
    """Демонстрация основ теории игр — нормальная форма, матрица выигрышей, доминирующая стратегия."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: ОСНОВЫ ТЕОРИИ ИГР")
    print("=" * 70)

    # --- 1.1 Нормальная форма игры ---
    print("\n--- 1.1 Нормальная форма игры ---")
    print("Игра 'Дилемма заключённого' — классическая модель конфликта")
    print("Два игрока: Каждый может КООПЕРИРОВАТЬ (C) или ПРЕДАТЬ (D)")
    print("Формат: (выигрыш_игрока1, выигрыш_игрока2)\n")

    # Матрица выигрышей: строки = игрок1, столбцы = игрок2
    # (Cooperate, Cooperate) = (-1, -1)  — оба сотрудничают, умеренная потеря
    # (Cooperate, Defect)    = (-3, 0)   — жертва и предатель
    # (Defect, Cooperate)    = (0, -3)   — предатель и жертва
    # (Defect, Defect)       = (-2, -2)  — оба предают, худший совместный результат

    payoff_matrix = {
        ("C", "C"): (-1, -1),
        ("C", "D"): (-3, 0),
        ("D", "C"): (0, -3),
        ("D", "D"): (-2, -2),
    }

    print("Матрица выигрышей (Дилемма заключённого):")
    print(f"  {'':>8} Игрок2: C    Игрок2: D")
    for a in ["C", "D"]:
        vals = [payoff_matrix[(a, b)] for b in ["C", "D"]]
        label = "Игрок1: " + a
        print(f"  {label:>8}   {str(vals[0]):>10}   {str(vals[1]):>10}")

    # --- 1.2 Доминирующая стратегия ---
    print("\n--- 1.2 Доминирующая стратегия ---")
    print("Стратeгия S доминирует над S', если S даёт не меньший выигрыш")
    print("при любой стратегии противника и строго больший при какой-то\n")

    # Проверяем доминирование для игрока 1
    for s1 in ["C", "D"]:
        for s2 in ["C", "D"]:
            p1_s1 = payoff_matrix[(s1, s2)][0]
            p1_s2 = payoff_matrix[(s2, s2)][0]  # s2 как стратегия игрока 1
            if s1 != s2:
                # Сравниваем s1 vs другую стратегию против всех ответов
                pass

    # Прямая проверка: D доминирует над C для игрока 1?
    dominates = True
    for opp in ["C", "D"]:
        pay_d = payoff_matrix[("D", opp)][0]
        pay_c = payoff_matrix[("C", opp)][0]
        print(f"  Если противник играет {opp}: D даёт {pay_d}, C даёт {pay_c}")
        if pay_d < pay_c:
            dominates = False

    if dominates:
        print("  => Стратегия D строго доминирует над C для игрока 1!")
        print("  => Rationalный игрок ВСЕГДА будет предавать")
    print()

    # --- 1.3 Симметричные и асимметричные игры ---
    print("--- 1.3 Симметричные и асимметричные игры ---")
    print("Симметричная игра: выигрыш не зависит от того, кто какой ролью играет")
    print("Т.е. payoff(s1, s2) = payoff(s2, s1) для всех s1, s2\n")

    # Проверка симметричности
    is_symmetric = True
    for (a, b), (p1, p2) in payoff_matrix.items():
        if payoff_matrix.get((b, a), (None, None)) != (p2, p1):
            is_symmetric = False
            break

    print(f"  Дилемма заключённого симметрична: {is_symmetric}")

    # Пример асимметричной игры: Охотник-Фермер
    print("\n  Пример асимметричной игры: Охотник и Фермер")
    hunter_farmer = {
        ("Hunt", "Farm"): (3, 2),
        ("Hunt", "Hunt"): (1, 1),
        ("Farm", "Farm"): (2, 3),
        ("Farm", "Hunt"): (2, 1),
    }
    print("  Формат: (выигрыш_Охотника, выигрыш_Фермера)")
    for (a, b), (p1, p2) in hunter_farmer.items():
        print(f"    Охотник={a}, Фермер={b}: ({p1}, {p2})")

    is_sym2 = True
    for (a, b), (p1, p2) in hunter_farmer.items():
        rev = hunter_farmer.get((b, a), (None, None))
        if rev != (p2, p1):
            is_sym2 = False
    print(f"  Охотник-Фермер симметрична: {is_sym2}\n")

    # --- 1.4 Концепция Nash- kvinna (предварительный обзор) ---
    print("--- 1.4 Эволюционно стабильная стратегия (ESS) ---")
    print("Стратегия s* — ESS, если ни одна мутантная стратегия s не может")
    print("захватить популяцию. Условие:")
    print("  E(s*, s*) > E(s, s*)  или  E(s*, s*) = E(s, s*) и E(s*, s) > E(s, s)\n")

    # Вычисляем средний выигрыш стратегии s* против s
    def expected_payoff(s1, s2, matrix):
        return matrix[(s1, s2)][0]

    # Проверка: является ли D ESS в дилемме заключённого?
    print("  Проверяем, является ли D (Предатель) ESS:")
    print(f"    E(D, D) = {expected_payoff('D', 'D', payoff_matrix)}")
    print(f"    E(C, D) = {expected_payoff('C', 'D', payoff_matrix)}")
    print(f"    E(D, C) = {expected_payoff('D', 'C', payoff_matrix)}")
    print(f"    E(C, C) = {expected_payoff('C', 'C', payoff_matrix)}")
    print(f"    E(D,D) > E(C,D)? {expected_payoff('D', 'D', payoff_matrix) > expected_payoff('C', 'D', payoff_matrix)}")
    print("  => D является ESS!\n")

    print("Итог: В дилемме заключённого доминирующая стратегия D —")
    print("  это единственное равновесие Нэша и ESS.")
    print("  Печальный парадокс: индивидуальная рациональность ведёт к collectively худшему результату.\n")


# =============================================================================
# 2. РАВНОВЕСИЕ НЭША
# =============================================================================

def demo_nash_equilibrium():
    """Демонстрация равновесий Нэша — чистые и смешанные стратегии, лучший ответ."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: РАВНОВЕСИЕ НЭША")
    print("=" * 70)

    # --- 2.1 Определение и поиск чистых равновесий ---
    print("\n--- 2.1 Поиск чистых равновесий Нэша ---")
    print("Равновесие Нэша: набор стратегий, где ни один игрок не может")
    print("улучшить свой выигрыш, односторонне изменив свою стратегию.\n")

    # Игра "Координатный выбор"
    # Два игрока выбирают A или B. Если координируются — большая награда.
    coord_payoffs = {
        ("A", "A"): (3, 3),
        ("A", "B"): (0, 1),
        ("B", "A"): (1, 0),
        ("B", "B"): (2, 2),
    }

    print("Игра 'Координатный выбор':")
    print(f"  {'':>8} Игрок2: A    Игрок2: B")
    for a in ["A", "B"]:
        vals = [coord_payoffs[(a, b)] for b in ["A", "B"]]
        print(f"  Игрок1:{a:>3}   {str(vals[0]):>10}   {str(vals[1]):>10}")

    # Поиск чистых равновесий: для каждой пары проверяем, что ни один не хочет отклоняться
    print("\n  Поиск Nash-равновесий:")
    nash_pure = []
    players = ["A", "B"]
    for s1 in players:
        for s2 in players:
            # Игрок 1: может ли улучшить, сменив s1?
            p1_current = coord_payoffs[(s1, s2)][0]
            p1_best_alt = max(coord_payoffs[(alt, s2)][0] for alt in players)
            # Игрок 2: может ли улучшить, сменив s2?
            p2_current = coord_payoffs[(s1, s2)][1]
            p2_best_alt = max(coord_payoffs[(s1, alt)][1] for alt in players)

            if p1_current >= p1_best_alt and p2_current >= p2_best_alt:
                nash_pure.append((s1, s2))
                print(f"    ({s1}, {s2}) — Nash! Выигрыш: {coord_payoffs[(s1, s2)]}")

    print(f"\n  Всего чистых равновесий: {len(nash_pure)}")

    # --- 2.2 Смешанные стратегии ---
    print("\n--- 2.2 Смешанные стратегии ---")
    print("В смешанной стратегии игрок выбирает вероятности для каждого действия.")
    print("Пусть p = P(игрок1 выбирает A), q = P(игрок2 выбирает A)\n")

    # Ожидаемый выигрыш игрока 1 при смешанных стратегиях
    # E1(p,q) = p*q*3 + p*(1-q)*0 + (1-p)*q*1 + (1-p)*(1-q)*2
    print("  Ожидаемый выигрыш игрока 1:")
    print("    E1(p,q) = 3pq + 0·p(1-q) + 1·(1-p)q + 2(1-p)(1-q)")
    print("            = 3pq + q - pq + 2 - 2p - 2q + 2pq")
    print("            = 4pq - p - q + 2\n")

    # Нахождение равновесия в смешанных стратегиях
    # Игрок 1 безразличен между A и B, когда:
    # E(A) = E(B) относительно q
    # E1(A, q) = 3q + 0(1-q) = 3q
    # E1(B, q) = 1q + 2(1-q) = q + 2 - 2q = 2 - q
    # 3q = 2 - q => 4q = 2 => q = 0.5
    print("  Игрок 1 безразличен при:")
    print("    E1(A,q) = 3q, E1(B,q) = 2-q")
    print("    3q = 2-q => q* = 0.5")
    print("  Симметрично: p* = 0.5")

    p_star = 0.5
    q_star = 0.5
    e1_mixed = 4 * p_star * q_star - p_star - q_star + 2
    print(f"  Ожидаемый выигрыш при (p*,q*) = ({p_star},{q_star}): {e1_mixed}")
    print("  => Смешанное равновесие: оба выбирают A и B с вероятностью 0.5\n")

    # --- 2.3 Лучший ответ (best response) ---
    print("--- 2.3 Функция лучшего ответа ---")
    print("Best response BR(s_-i) — стратегия, максимизирующая выигрыш")
    print("при данных стратегиях всех остальных игроков.\n")

    # Построение графика best response для координатной игры
    print("  Лучшие ответы для координатной игры:")
    print("  Для игрока 1:")
    # BR1(q) = A если 3q > 2-q, т.е. q > 0.5; B если q < 0.5; {A,B} если q=0.5
    for q_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        e_a = 3 * q_val
        e_b = 2 - q_val
        if e_a > e_b:
            br = "A"
        elif e_b > e_a:
            br = "B"
        else:
            br = "{A, B}"
        print(f"    q={q_val:.2f}: BR1 = {br}  (E(A)={e_a:.2f}, E(B)={e_b:.2f})")

    print("\n  Пересечение best response = Nash равновесие")
    print("  Точки пересечения: (A,A) и (B,B) — как нашли выше\n")

    # --- 2.4 Применение к реальным сценариям ---
    print("--- 2.4 Реальный сценарий: ценообразование Берtrand ---")
    print("Два продавца устанавливают цены p1, p2 для одинакового товара.")
    print("Потребители покупают у более дешёвого. При равных ценах — делят рынок.\n")

    # Дискретные цены: 10, 20, 30
    prices = [10, 20, 30]
    demand = 100  # общее число покупателей

    def bertrand_payoff(p1, p2, prices_list, dem):
        if p1 < p2:
            return p1 * dem
        elif p1 > p2:
            return 0
        else:
            return p1 * dem // 2

    print("  Дискретные цены: {10, 20, 30}, спрос: 100")
    print(f"  {'p1\\p2':>8}", end="")
    for p2 in prices:
        print(f"   {p2:>6}", end="")
    print()

    for p1 in prices:
        print(f"  {p1:>8}", end="")
        for p2 in prices:
            payoff = bertrand_payoff(p1, p2, prices, demand)
            print(f"   {payoff:>6}", end="")
        print()

    print("\n  Поиск Nash-равновесия:")
    bertrand_nash = []
    for p1 in prices:
        for p2 in prices:
            curr = bertrand_payoff(p1, p2, prices, demand)
            best_p1 = max(bertrand_payoff(p, p2, prices, demand) for p in prices)
            best_p2 = max(bertrand_payoff(p1, p, prices, demand) for p in prices)
            if curr >= best_p1 and curr >= best_p2:
                bertrand_nash.append((p1, p2))
                print(f"    (p1={p1}, p2={p2}) — Nash! Выигрыш: ({curr}, {curr})")

    print("  Вывод: в дискретном случае ценовой войны нет полного равновесия,")
    print("  но (10, 10) — ближайший к конкурентному результату.\n")


# =============================================================================
# 3. АУКЦИОННЫЕ МЕХАНИЗМЫ
# =============================================================================

def demo_auction_mechanisms():
    """Демонстрация аукционных механизмов — английский, голландский, закрытый, VCG."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: АУКЦИОННЫЕ МЕХАНИЗМЫ")
    print("=" * 70)

    # Установка оценок участников
    valuations = {
        "Алиса": 100,
        "Борис": 80,
        "Вика": 120,
        "Глеб": 60,
        "Даша": 90,
    }
    print(f"\nОценки участников (истинная стоимость для каждого):")
    for name, val in valuations.items():
        print(f"  {name}: {val}")

    # --- 3.1 Английский аукцион (открытый, восходящий) ---
    print("\n--- 3.1 Английский аукцион (открытый, восходящий) ---")
    print("Цена начинается низко и растёт. Участники делают ставки публично.")
    print("Последний оставшийся — побеждает, платит свою ставку.\n")

    # Симуляция английского аукциона
    current_price = 10
    step = 5
    active = dict(valuations)  # все активны
    history = []

    print("  Ход аукциона:")
    while len(active) > 1:
        # Кто готов платить текущую цену?
        remaining = {name: val for name, val in active.items() if val > current_price}
        dropped = set(active.keys()) - set(remaining.keys())
        for name in dropped:
            print(f"    Цена {current_price}: {name} выбывает (оценка {active[name]} < цена)")
            history.append((current_price, name, "выбыл"))
        active = remaining
        if len(active) > 1:
            current_price += step
            print(f"    Цена повышена до {current_price}")

    winner = list(active.keys())[0] if active else None
    print(f"\n  Победитель: {winner} (последний оставшийся)")
    print(f"  Цена покупки: {current_price}")
    print(f"  Выигрыш: оценка {valuations[winner]} - цена {current_price} = {valuations[winner] - current_price}")

    # --- 3.2 Голландский аукцион (открытый, нисходящий) ---
    print("\n--- 3.2 Голландский аукцион (открытый, нисходящий) ---")
    print("Цена начинается высоко и снижается. Первый, кто соглашается — побеждает.\n")

    max_val = max(valuations.values())
    current_price = max_val + 20  # стартовая цена
    decrease = 5
    winner_dutch = None

    print("  Ход аукциона:")
    while winner_dutch is None and current_price > 0:
        # Кто готов купить по текущей цене?
        bidders = [name for name, val in valuations.items() if val >= current_price]
        if bidders:
            # Первый по алфавиту побеждает (симуляция)
            winner_dutch = min(bidders)
            print(f"    Цена {current_price}: {winner_dutch} принимает! (оценка {valuations[winner_dutch]} >= цена)")
        else:
            print(f"    Цена {current_price}: нет желающих, снижаем...")
            current_price -= decrease

    print(f"\n  Победитель: {winner_dutch}")
    print(f"  Цена покупки: {current_price}")
    print(f"  Выигрыш: {valuations[winner_dutch] - current_price}")

    # --- 3.3 Закрытый аукцион (первая цена) ---
    print("\n--- 3.3 Закрытый аукцион (первая цена, sealed-bid) ---")
    print("Каждый участник делает.secretную ставку. Кто больше — побеждает.")
    print("Платит сумму своей ставки.\n")

    random.seed(42)
    # Рациональная стратегия: ставить ниже истинной оценки
    bids_sealed = {}
    for name, val in valuations.items():
        # Стратегия: ставим ~80% от оценки (shading)
        bid = int(val * random.uniform(0.6, 0.95))
        bids_sealed[name] = bid

    print("  Ставки участников (закрытый аукцион):")
    for name, bid in bids_sealed.items():
        print(f"    {name}: ставка {bid} (истинная оценка {valuations[name]})")

    winner_sealed = max(bids_sealed, key=bids_sealed.get)
    winning_bid = bids_sealed[winner_sealed]
    print(f"\n  Победитель: {winner_sealed} (ставка {winning_bid})")
    print(f"  Выигрыш: {valuations[winner_sealed]} - {winning_bid} = {valuations[winner_sealed] - winning_bid}")

    # --- 3.4 VCG аукцион (Викри-Кларк-Гровс) ---
    print("\n--- 3.4 VCG аукцион (Викри-Кларк-Гровс) ---")
    print("Каждый платит «вклад в ущерб другим». Целевой механизм:")
    print("  цена_i = max выигрыш_других_без_i - выигрыш_других_с_i\n")

    # VCG для одного товара: победитель — Вика (120)
    vcg_winner = "Вика"
    vcg_price_val = valuations[vcg_winner]

    # Выигрыш других без победителя
    others_without = sum(val for name, val in valuations.items() if name != vcg_winner)
    # Выигрыш других с победителем: второй по величине оценка определяет цену
    sorted_vals = sorted(valuations.values(), reverse=True)
    second_best = sorted_vals[1]
    # Без Вики лучший — Алиса (100), выигрыш = 100
    # С Викой лучший — Вика, выигрыш других = 0 (Вика забирает товар)
    # VCG цена = 100 - 0 = 100
    vcg_price = second_best

    print(f"  Победитель: {vcg_winner} (оценка {vcg_price_val})")
    print(f"  VCG цена = макс. выигрыш_других_без_Вики - выигрыш_других_с_Викой")
    print(f"           = {second_best} (лучшая оценка без Вики) - 0 = {vcg_price}")
    print(f"  Выигрыш Вики: {vcg_price_val - vcg_price}")

    # Доказательство, что VCG — truthful auction
    print("\n  Доказательство truthfulness VCG:")
    print("  Если Вика занижает ставку до 79 (ниже Алисы 100):")
    print("    Победитель = Алиса, цена = max(остальных без Алисы) = max(80, 70, 60, 90) = 90")
    print("    Выигрыш Вики при ставке 120 (честно):", 120 - vcg_price)
    print("    Выигрыш Вики при ставке 79 (лживо): 0 (не выиграла)")
    print("  => Честная стратегия — оптимальна!\n")

    # Сравнение механизмов
    print("--- Сравнение аукционных механизмов ---")
    comparison = [
        ("Английский", "Открытый", "Восходящий", "Последняя ставка", "Полная информация"),
        ("Голландский", "Открытый", "Нисходящий", "Первая ставка", "Мало информации"),
        ("Закрытый (1-я цена)", "Закрытый", "Скрытые ставки", "Своя ставка", "Риск winning curse"),
        ("VCG", "Закрытый", "Скрытые ставки", "Вклад в ущерб", "Truthful, сложный"),
    ]
    print(f"  {'Тип':<20} {'Формат':<10} {'Процесс':<12} {'Цена':<20} {'Особенность'}")
    for row in comparison:
        print(f"  {row[0]:<20} {row[1]:<10} {row[2]:<12} {row[3]:<20} {row[4]}")
    print()


# =============================================================================
# 4. КОНКУРЕНТНЫЕ СТРАТЕГИИ
# =============================================================================

def demo_competitive_strategies():
    """Демонстрация конкурентных стратегий — минимакс, максимин, минимизация сожаления."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: КОНКУРЕНТНЫЕ СТРАТЕГИИ")
    print("=" * 70)

    # --- 4.1 Минимакс стратегия ---
    print("\n--- 4.1 Минимакс стратегия ---")
    print("Минимакс: игрок minimizes свой максимальный потенциальный убыток.")
    print("  min_s max_s' U(s', s)  — minimize worst-case outcome\n")

    # Игра "Камень-Ножницы-Бумага" с числовыми выигрышами
    rps_payoffs = {
        #                   Камень   Ножницы  Бумага
        "Камень":   {"Камень": 0, "Ножницы": 1, "Бумага": -1},
        "Ножницы":  {"Камень": -1, "Ножницы": 0, "Бумага": 1},
        "Бумага":   {"Камень": 1, "Ножницы": -1, "Бумага": 0},
    }
    actions = ["Камень", "Ножницы", "Бумага"]

    print("  Матрица выигрыш (Камень-Ножницы-Бумага):")
    print(f"  {'':>12}", end="")
    for a in actions:
        print(f" {a:>10}", end="")
    print()
    for a1 in actions:
        print(f"  {a1:>12}", end="")
        for a2 in actions:
            print(f" {rps_payoffs[a1][a2]:>10}", end="")
        print()

    print("\n  Минимакс для игрока 1 (максимум при худшем противнике):")
    for a1 in actions:
        worst = max(-rps_payoffs[a2][a1] for a2 in actions)  # противник максимизирует свой выигрыш = минус выигрыш игрока 1
        worst_actual = min(rps_payoffs[a1][a2] for a2 in actions)
        print(f"    {a1}: worst-case выигрыш = {worst_actual}")

    minimax_val = max(min(rps_payoffs[a1][a2] for a2 in actions) for a1 in actions)
    print(f"  Минимакс значение: {minimax_val}")
    print("  => Все стратегии одинаковы — симметричная игра!\n")

    # --- 4.2 Максимин стратегия ---
    print("--- 4.2 Максимин стратегия ---")
    print("Максимин: игрок maximizes свой минимальный guaranteed выигрыш.")
    print("  max_s min_s' U(s, s')  — guarantee floor of payoff\n")

    print("  Максимин для каждой стратегии:")
    for a1 in actions:
        min_payoff = min(rps_payoffs[a1][a2] for a2 in actions)
        print(f"    min({a1}) = {min_payoff}")

    maximin_val = max(min(rps_payoffs[a1][a2] for a2 in actions) for a1 in actions)
    print(f"  Максимин значение: {maximin_val}")

    print("\n  Минимакс = Максимин = 0 (в нулевой суммовой игре)")
    print("  Это теорема фон Неймана: в нулевой суммовых играх minimax = maximin\n")

    # --- 4.3 Regret minimization (CFR-like) ---
    print("--- 4.3 Минимизация сожаления (Regret Minimization) ---")
    print("Сожаление: разница между полученным выигрышем и лучшим возможным.")
    print("  Regret(s, s') = max_a U(a, s') - U(s, s')\n")

    # Пример: 3x3 игра (не симметричная)
    game = {
        (0, 0): 6, (0, 1): 2, (0, 2): 8,
        (1, 0): 1, (1, 1): 5, (1, 2): 3,
        (2, 0): 7, (2, 1): 4, (2, 2): 2,
    }
    n_strategies = 3

    print("  Игра (выигрыш игрока 1):")
    for i in range(n_strategies):
        row = [game[(i, j)] for j in range(n_strategies)]
        print(f"    Стратегия {i}: {row}")

    # Итеративное CFR
    n_iterations = 100
    cumulative_regret = [0] * n_strategies
    strategy_sum = [0] * n_strategies
    strategy = [1.0 / n_strategies] * n_strategies  # начальная — равномерная

    for t in range(1, n_iterations + 1):
        # Средний выигрыш текущей стратегии
        avg_payoff = sum(strategy[i] * sum(game[(i, j)] for j in range(n_strategies)) / n_strategies for i in range(n_strategies))

        # Вычисляем regret для каждой стратегии
        for i in range(n_strategies):
            # Выигрыш стратегии i против равномерного противника
            payoff_i = sum(game[(i, j)] for j in range(n_strategies)) / n_strategies
            cumulative_regret[i] += payoff_i - avg_payoff

        # Обновляем стратегию на основе positive regrets
        total_pos_regret = sum(max(0, r) for r in cumulative_regret)
        if total_pos_regret > 0:
            strategy = [max(0, r) / total_pos_regret for r in cumulative_regret]
        else:
            strategy = [1.0 / n_strategies] * n_strategies

        # Накапливаем стратегию
        for i in range(n_strategies):
            strategy_sum[i] += strategy[i]

    # Нормализуем
    total = sum(strategy_sum)
    avg_strategy = [s / total for s in strategy_sum]

    print(f"\n  После {n_iterations} итераций CFR:")
    print(f"    Накопленные сожаления: {[f'{r:.2f}' for r in cumulative_regret]}")
    print(f"    Средняя стратегия: {[f'{s:.3f}' for s in avg_strategy]}")

    # Проверка: какая чистая стратегия лучше?
    best_single = max(range(n_strategies), key=lambda i: sum(game[(i, j)] for j in range(n_strategies)))
    print(f"    Лучшая чистая стратегия: {best_single} (сумма выигрышей: {sum(game[(best_single, j)] for j in range(n_strategies))})")

    expected = sum(avg_strategy[i] * sum(game[(i, j)] for j in range(n_strategies)) / n_strategies for i in range(n_strategies))
    print(f"    Ожидаемый выигрыш смешанной стратегии: {expected:.2f}")

    # --- 4.4 Применение в реальных сценариях ---
    print("\n--- 4.4 Реальный сценарий: кибербезопасность ---")
    print("Атакующий vs Защитник: классическая минимакс задача\n")

    # 3 атаки и 3 защиты
    attacks = ["DDoS", "SQL-инъекция", "Фишинг"]
    defenses = ["Файрвол", "WAF", "Обучение"]

    # Матрица: выигрыш атакующего (урон системе)
    cyber_payoffs = {
        ("DDoS", "Файрвол"): 2,       # файрвол смягчает DDoS
        ("DDoS", "WAF"): 1,           # WAF помогает
        ("DDoS", "Обучение"): 8,      # обучение не помогает от DDoS
        ("SQL-инъекция", "Файрвол"): 7,
        ("SQL-инъекция", "WAF"): 3,   # WAF блокирует SQL-инъекции
        ("SQL-инъекция", "Обучение"): 6,
        ("Фишинг", "Файрвол"): 5,
        ("Фишинг", "WAF"): 5,
        ("Фишинг", "Обучение"): 1,    # обучение блокирует фишинг
    }

    print("  Матрица урона (выигрыш атакующего):")
    print(f"  {'':>16}", end="")
    for d in defenses:
        print(f" {d:>12}", end="")
    print()
    for a in attacks:
        print(f"  {a:>16}", end="")
        for d in defenses:
            print(f" {cyber_payoffs[(a, d)]:>12}", end="")
        print()

    # Минимакс для атакующего
    print("\n  Минимакс для атакующего:")
    worst_cases = {}
    for a in attacks:
        worst = max(cyber_payoffs[(a, d)] for d in defenses)
        worst_cases[a] = worst
        print(f"    {a}: worst-case урон = {worst}")

    best_attack = min(worst_cases, key=worst_cases.get)
    print(f"  Оптимальная атака (minimax): {best_attack} (garantirovанный урон: {worst_cases[best_attack]})")

    # Максимин для защитника
    print("\n  Максимин для защитника:")
    for d in defenses:
        worst = max(cyber_payoffs[(a, d)] for a in attacks)
        print(f"    {d}: worst-case урон = {worst}")

    best_defense = min(defenses, key=lambda d: max(cyber_payoffs[(a, d)] for a in attacks))
    worst_for_best = max(cyber_payoffs[(a, best_defense)] for a in attacks)
    print(f"  Оптимальная защита (maximin): {best_defense} (worst-case урон: {worst_for_best})")

    print("\n  Нашли minimax решение:")
    print(f"    Атакующий выбирает: {best_attack} (garantirovannый урон: {worst_cases[best_attack]})")
    print(f"    Защитник выбирает: {best_defense} (worst-case урон: {worst_for_best})")
    print(f"    Это saddle point? {worst_cases[best_attack] == worst_for_best}")
    print()


# =============================================================================
# ГЛАВНЫЙ БЛОК
# =============================================================================

if __name__ == "__main__":
    demo_game_theory_basics()
    demo_nash_equilibrium()
    demo_auction_mechanisms()
    demo_competitive_strategies()
