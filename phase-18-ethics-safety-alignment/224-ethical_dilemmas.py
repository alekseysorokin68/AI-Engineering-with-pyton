"""
224 — Ethical Dilemmas: задача трамвая, компромиссы ценностей

Темы:
  1. Trolley Problem Variants (classical, loop, transplant, autonomous vehicle)
  2. Value Trade-offs (privacy vs safety, fairness vs accuracy, individual vs collective)
  3. Moral Frameworks (utilitarianism, deontology, virtue ethics, care ethics)
  4. AI Moral Reasoning (value-sensitive design, participatory design, moral buffers)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ============================================================
# Демо 1: Trolley Problem Variants — классическая, петля, трансплантация, AV
# ============================================================

def demo1_trolley_problem():
    """Демонстрация вариантов задачи о трамвае."""
    print("=" * 70)
    print("Демо 1: Trolley Problem Variants — варианты задачи о трамвае")
    print("=" * 70)

    # --- 1.1 Classical Trolley Problem ---
    print(f"\n--- 1.1 Classical Trolley Problem: классическая задача ---")
    print("Формулировка: трамвай едет на 5 человек. Вы можете переключить")
    print("стрелку, чтобы трамвай поехал на 1 человек. Переключите?")

    # Подсчёт «голосов» по разным этическим позициям
    responses_classical = {
        "Утилитаризм (переключить)": 75,
        "Деонтология (не переключить)": 10,
        "Другие позиции": 15,
    }

    print(f"\nРезультаты опроса (классический вариант):")
    total = sum(responses_classical.values())
    for response, pct in responses_classical.items():
        bar = "█" * (pct // 2)
        print(f"  {response:<35} {pct:>3}% {bar}")

    print(f"\n  Ключевой вопрос: переключение — это действие или бездействие?")
    print(f"  → Утилитарист: переключить (5 > 1)")
    print(f"  → Деонтолог: не переключить (активное причинение смерти неправильно)")

    # --- 1.2 Loop Variant ---
    print(f"\n--- 1.2 Loop Variant: вариант с петлёй ---")
    print("Формулировка: трамвай едет по петле. Вы можете остановить его,")
    print("но для этого нужно убить человека, чьё тело остановит трамвай.")
    print("Если не остановить — погибнут 5 человек.")

    print(f"\nМоральная интуиция:")
    print(f"  Классический: 5 vs 1 → большинство会选择 переключить")
    print(f"  Петля:        5 vs 1 (через убийство) → большинство会选择 НЕ переключать")
    print(f"\n  Почему? В петле: вы используете человека как средство (Kant)")
    print(f"  В классическом: вы не используете 1 человека как средство")

    loop_responses = {
        "Переключить (спасти 5)": 30,
        "Не переключить (не убивать)": 55,
        "Зависит от контекста": 15,
    }
    print(f"\nРезультаты опроса (петля):")
    for resp, pct in loop_responses.items():
        bar = "█" * (pct // 2)
        print(f"  {resp:<35} {pct:>3}% {bar}")

    # --- 1.3 Transplant Problem ---
    print(f"\n--- 1.3 Transplant Problem: задача о трансплантации ---")
    print("Формулировка: 5 пациентов умрут без пересадки органов.")
    print("Один здоровый пациент в соседней палате — его органы спасут всех пятерых.")
    print("Убить здорового пациента и забрать органы?")

    print(f"\nМоральная интуиция:")
    print(f"  Утилитарный расчёт: 5 жизней > 1 жизнь → УБИТЬ")
    print(f"  Но 90%+ людей会选择 НЕ убивать")
    print(f"\n  Почему? Нарушение фундаментального права на жизнь")
    print(f"  →康德: человек — цель, не средство")
    print(f"  → Вирту-этика: убийство — порок")

    transplant_responses = {
        "Убить (спасти 5)": 8,
        "Не убивать (права пациента)": 82,
        "Зависит от обстоятельств": 10,
    }
    print(f"\nРезультаты опроса (трансплантация):")
    for resp, pct in transplant_responses.items():
        bar = "█" * (pct // 2)
        print(f"  {resp:<35} {pct:>3}% {bar}")

    print(f"\n  Парадокс: логика = трамвай, но интуиция ≠ трамвай")
    print(f"  → Трансплантация: вы АКТИВНО убиваете человека")
    print(f"  → Трамвай: вы перенаправляете существующую угрозу")

    # --- 1.4 Autonomous Vehicle ---
    print(f"\n--- 1.4 Autonomous Vehicle: автономный автомобиль ---")
    print("Вопрос: как должен действовать автономный автомобиль в аварийной ситуации?")

    av_scenarios = [
        {
            "scenario": "Пешеход vs пассажир",
            "choice_a": "Врезаться в стену (пассажир погибнет)",
            "choice_b": "Сбить пешехода (пешеход погибнет)",
            "utilitarian": "B",
            "deontological": "A",
        },
        {
            "scenario": "1 пешеход vs 3 пешехода",
            "choice_a": "Сбить 1 человека",
            "choice_b": "Сбить 3 человека",
            "utilitarian": "A",
            "deontological": "Разные ответы",
        },
        {
            "scenario": "Дитя vs взрослый",
            "choice_a": "Сбить ребёнка",
            "choice_b": "Сбить взрослого",
            "utilitarian": "B (больше жизни впереди)",
            "deontological": "Запрещено различие по возрасту",
        },
        {
            "scenario": "Законный vs незаконный пешеход",
            "choice_a": "Сбить пешехода на зебре",
            "choice_b": "Сбить пешехода вне зебры",
            "utilitarian": "Нет различия",
            "deontological": "Нет различия",
        },
    ]

    print(f"\nСценарии для AV:")
    for i, sc in enumerate(av_scenarios, 1):
        print(f"\n  {i}. {sc['scenario']}")
        print(f"     A: {sc['choice_a']}")
        print(f"     B: {sc['choice_b']}")
        print(f"     Утилитаризм → {sc['utilitarian']}")
        print(f"     Деонтология → {sc['deontological']}")

    # Моральная матрица (пример: MIT Moral Machine)
    print(f"\nMoral Machine (MIT): глобальное исследование")
    print(f"  Результаты: люди предпочитают спасать молодых vs старых,")
    print(f"  больше людей vs меньше, пассажиров vs пешеходов")
    print(f"\n  Но: предпочтения ВАРЬИРУЮТСЯ по культурам!")
    print(f"  → Коллективистские культуры: больше внимания статусу")
    print(f"  → Индивидуалистические: больше внимания закону")

    print()


# ============================================================
# Демо 2: Value Trade-offs — компромиссы ценностей
# ============================================================

def demo2_value_tradeoffs():
    """Демонстрация компромиссов между ценностями."""
    print("=" * 70)
    print("Демо 2: Value Trade-offs — компромиссы ценностей")
    print("=" * 70)

    random.seed(42)

    # --- 2.1 Privacy vs Safety ---
    print(f"\n--- 2.1 Privacy vs Safety: приватность vs безопасность ---")

    privacy_safety = [
        {
            "context": "Медицинские данные",
            "privacy_concern": "Утечка диагнозов → дискриминация",
            "safety_benefit": "Раннее обнаружение эпидемий",
            "balance": "Anonymization + Consent",
        },
        {
            "context": "Видеонаблюдение с распознаванием",
            "privacy_concern": "Слежка за гражданами",
            "safety_benefit": "Предотвращение преступлений",
            "balance": "Временное хранение, нетface recognition",
        },
        {
            "context": "Геолокация пользователей",
            "privacy_concern": "Профилирование поведения",
            "safety_benefit": "Экстренные уведомления",
            "balance": "Opt-in,ifferential privacy",
        },
    ]

    for ps in privacy_safety:
        print(f"\n  {ps['context']}")
        print(f"    Приватность: {ps['privacy_concern']}")
        print(f"    Безопасность: {ps['safety_benefit']}")
        print(f"    Баланс: {ps['balance']}")

    # Матрица компромиссов
    print(f"\n  Матрица компромиссов Privacy vs Safety:")
    print(f"  {'':>25} {'Высокая приватность':>20} {'Низкая приватность':>20}")
    print(f"  {'Высокая безопасность':<25} {'Оптимально':>20} {'Репрессивно':>20}")
    print(f"  {'Низкая безопасность':<25} {'Уязвимо':>20} {'Хаос':>20}")

    # --- 2.2 Fairness vs Accuracy ---
    print(f"\n--- 2.2 Fairness vs Accuracy: справедливость vs точность ---")

    print(f"\nПример: кредитный скоринг")
    print(f"\n  Без fairness constraints:")
    print(f"    Точность: 92%")
    print(f"    Disparate Impact: 0.6 (неприемлемо < 0.8)")

    # Разные стратегии fairness
    fairness_strategies = [
        {"strategy": "No fairness constraint", "accuracy": 0.92, "disparate_impact": 0.60, "group_parity": "Нет"},
        {"strategy": "Demographic Parity", "accuracy": 0.88, "disparate_impact": 0.85, "group_parity": "Да"},
        {"strategy": "Equalized Odds", "accuracy": 0.89, "disparate_impact": 0.82, "group_parity": "Частично"},
        {"strategy": "Calibration", "accuracy": 0.91, "disparate_impact": 0.78, "group_parity": "Нет"},
    ]

    print(f"\n  Стратегии fairness:")
    print(f"  {'Стратегия':<25} {'Accuracy':>10} {'DI':>8} {'Group Parity':>15}")
    print(f"  {'-'*65}")
    for fs in fairness_strategies:
        print(f"  {fs['strategy']:<25} {fs['accuracy']:>10.0%} {fs['disparate_impact']:>8.2f} {fs['group_parity']:>15}")

    print(f"\n  Trade-off: fairness снижает точность, но повышает справедливость")
    print(f"  → Вопрос: какой fairness-критерий выбрать?")
    print(f"  → Зависит от домена: кредиты (DI), найм (EO), медицина (Calibration)")

    # --- 2.3 Individual vs Collective ---
    print(f"\n--- 2.3 Individual vs Collective: индивид vs коллектив ---")

    collectivism_spectrum = [
        {
            "culture": "Западная (индивидуалистическая)",
            "individual": 8,
            "collective": 2,
            "ai_implication": "Фокус на персональном выборе, consent",
        },
        {
            "culture": "Восточная (коллективистическая)",
            "individual": 3,
            "collective": 7,
            "ai_implication": "Фокус на благе сообщества, group welfare",
        },
        {
            "culture": "Скандинавская (социал-демократическая)",
            "individual": 5,
            "collective": 5,
            "ai_implication": "Баланс: права личности + общее благо",
        },
    ]

    print(f"\nСпектр культурных различий:")
    print(f"  {'Культура':<35} {'Индивид':>8} {'Коллектив':>10}")
    print(f"  {'-'*58}")
    for c in collectivism_spectrum:
        ind_bar = "█" * c["individual"]
        col_bar = "█" * c["collective"]
        print(f"  {c['culture']:<35} {ind_bar:<8} {col_bar:<10}")

    print(f"\n  Импликации для AI:")
    for c in collectivism_spectrum:
        print(f"    {c['culture']}: {c['ai_implication']}")

    # --- 2.4 Transparency vs Performance ---
    print(f"\n--- 2.4 Transparency vs Performance: прозрачность vs производительность ---")

    models_comparison = [
        {"model": "Линейная регрессия", "transparency": 10, "accuracy": 70, "speed": 10, "explainability": 10},
        {"model": "Decision Tree", "transparency": 9, "accuracy": 75, "speed": 9, "explainability": 9},
        {"model": "Random Forest", "transparency": 5, "accuracy": 85, "speed": 7, "explainability": 5},
        {"model": "XGBoost", "transparency": 4, "accuracy": 88, "speed": 7, "explainability": 4},
        {"model": "Neural Network ( shallow)", "transparency": 3, "accuracy": 90, "speed": 6, "explainability": 3},
        {"model": "Deep Neural Network", "transparency": 1, "accuracy": 95, "speed": 4, "explainability": 1},
    ]

    print(f"\nСравнение моделей по прозрачности и производительности:")
    print(f"  {'Модель':<25} {'Прозрач.':>9} {'Точность':>9} {'Скорость':>9} {'Объясним.':>10}")
    print(f"  {'-'*68}")
    for m in models_comparison:
        t_bar = "█" * m["transparency"]
        a_bar = "█" * (m["accuracy"] // 2)
        print(f"  {m['model']:<25} {t_bar:<9} {a_bar:<9} {'█' * m['speed']:<9} {m['explainability']}/10")

    print(f"\n  Trade-off: более сложные модели точнее, но менее прозрачны")
    print(f"  → Решение: SHAP, LIME, Attention visualization")
    print(f"  → Или: ансамбли (точность) + объяснимые мета-модели")

    print()


# ============================================================
# Демо 3: Moral Frameworks — утилитаризм, деонтология, вирту-этика
# ============================================================

def demo3_moral_frameworks():
    """Демонстрация различных моральных фреймворков."""
    print("=" * 70)
    print("Демо 3: Moral Frameworks — моральные фреймворки")
    print("=" * 70)

    random.seed(42)

    # --- 3.1 Utilitarianism ---
    print(f"\n--- 3.1 Utilitarianism: утилитаризм ---")
    print("Принцип:.maximize общего блага (utility)")
    print("Формула:道德的行为 = maximize Σ utility(всех затронутых)")

    # Пример: распределение ресурсов
    def calculate_utility(distribution, utility_fn):
        """Вычисление утилитарного блага."""
        return sum(utility_fn(amount) for amount in distribution)

    # Функция полезности: логарифмическая (убывающая предельная полезность)
    def log_utility(amount):
        """Логарифмическая функция полезности."""
        if amount <= 0:
            return -10
        return math.log(amount + 1)

    # Пример: 100 единиц ресурса, 4 человека
    total_resources = 100
    n_people = 4

    # Разные стратегии распределения
    distributions = [
        {"name": "Равномерное", "dist": [25, 25, 25, 25]},
        {"name": "Пропорциональное (богатые получают больше)", "dist": [40, 30, 20, 10]},
        {"name": "Обратное (бедные получают больше)", "dist": [10, 20, 30, 40]},
        {"name": "Максимин (максимум минимума)", "dist": [25, 25, 25, 25]},  # rawlsian
    ]

    print(f"\nРаспределение {total_resources} единиц ресурса между {n_people} людьми:")
    print(f"  Функция полезности: U(x) = ln(x + 1)\n")

    print(f"  {'Стратегия':<45} {'Утилита':>8}")
    print(f"  {'-'*55}")
    best_util = -float('inf')
    best_name = ""
    for d in distributions:
        util = calculate_utility(d["dist"], log_utility)
        d["utility"] = util
        if util > best_util:
            best_util = util
            best_name = d["name"]
        bar = "█" * int(util)
        print(f"  {d['name']:<45} {util:>8.2f} {bar}")

    print(f"\n  Лучшая стратегия (утилитаризм): {best_name}")

    # Проблема: utility monster
    print(f"\n  Проблема utility monster:")
    print(f"    Если один человек получает 1000x больше, его utility")
    print(f"    доминирует → утилитаризм оправдывает эксплуатацию")

    # --- 3.2 Deontology ---
    print(f"\n--- 3.2 Deontology: деонтология (Кант) ---")
    print("Принцип: некоторые действия неверны ВНЕЗАВИСИМО от последствий")
    print("Категорический императив: действуй только так, как хотел бы,")
    print("чтобы действовали все")

    # Проверка на категорический императив
    def check_categorical_imperative(action, universalizable):
        """Проверка: может ли действие быть универсализовано."""
        if universalizable:
            return "ПРОХОДИТ: действие может быть универсализовано"
        else:
            return "НЕ ПРОХОДИТ: если все так делают — противоречие"

    actions = [
        {"action": "Ложь для спасения друга", "universalizable": False,
         "reason": "Если все лгут — доверие невозможно"},
        {"action": "Помощь нуждающемуся", "universalizable": True,
         "reason": "Если все помогают — мир лучше"},
        {"action": "Кража для пропитания", "universalizable": False,
         "reason": "Если все крадут — собственность невозможна"},
        {"action": "Честная работа за зарплату", "universalizable": True,
         "reason": "Если все работают честно — справедливость"},
    ]

    print(f"\nПроверка действий на категорический императив:")
    for a in actions:
        result = check_categorical_imperative(a["action"], a["universalizable"])
        print(f"\n  Действие: {a['action']}")
        print(f"  Результат: {result}")
        print(f"  Причина: {a['reason']}")

    # --- 3.3 Virtue Ethics ---
    print(f"\n--- 3.3 Virtue Ethics: вирту-этика (Аристотель) ---")
    print("Принцип: моральный человек — это тот, кто обладает добродетелями")
    print("Добродетель = среднее между крайностями")

    virtues = [
        {"name": "Храбрость", "deficiency": "Трусость", "excess": "Безрассудство"},
        {"name": "Щедрость", "deficiency": "Жадность", "excess": "Расточительность"},
        {"name": "Честность", "deficiency": "Лживость", "excess": "Безтактность"},
        {"name": "Умеренность", "deficiency": "Аскетизм", "excess": "Чревоугодие"},
    ]

    print(f"\nАристотелевы добродетели:")
    print(f"  {'Добродетель':<15} {'Недостаток':<15} {'Избыток':<15} {'Среднее':>10}")
    print(f"  {'-'*60}")
    for v in virtues:
        print(f"  {v['name']:<15} {v['deficiency']:<15} {v['excess']:<15} {'Добродетель':>10}")

    # Пример: вирту-этика в AI
    print(f"\n  Вирту-этика в AI:")
    print(f"    → Какие добродетели должна иметь AI-система?")
    print(f"    → Честность: не лгать пользователям")
    print(f"    → Мудрость: знать границы своих знаний")
    print(f"    → Справедливость: не дискриминировать")
    print(f"    → Сдержанность: не навязывать решения")

    # --- 3.4 Care Ethics ---
    print(f"\n--- 3.4 Care Ethics: этика заботы ---")
    print("Принцип: мораль основана на заботе и ответственности")
    print("за конкретных людей, а абстрактных принципов")

    care_principles = [
        {"principle": "Забота о близких", "application": "Персонализация AI-помощи"},
        {"principle": "Ответственность за последствия", "application": "Отслеживание вреда от AI"},
        {"principle": "Чуткость к контексту", "application": "Учёт культурных различий"},
        {"principle": "Взаимозависимость", "application": "Сообщественный подход к AI"},
    ]

    print(f"\nПринципы этики заботы и их применения в AI:")
    for cp in care_principles:
        print(f"\n  {cp['principle']}")
        print(f"    Применение: {cp['application']}")

    # Сравнение фреймворков
    print(f"\nСравнение моральных фреймворков:")
    comparison = [
        {"aspect": "Фокус", "utilitarian": "Последствия", "deontological": "Действия",
         "virtue": "Характер", "care": "Отношения"},
        {"aspect": "Критерий", "utilitarian": "Utility", "deontological": "Правила",
         "virtue": "Добродетели", "care": "Забота"},
        {"aspect": "Сильная сторона", "utilitarian": "Количественный анализ",
         "deontological": "Права человека", "virtue": "Холистичный подход", "care": "Эмпатия"},
        {"aspect": "Слабая сторона", "utilitarian": "Utility monster",
         "deontological": "Догматизм", "virtue": "Субъективность", "care": "Эксклюзивность"},
    ]

    print(f"\n  {'Аспект':<20} {'Утилитар.':<18} {'Деонтол.':<18} {'Вирту':<18} {'Забота':<18}")
    print(f"  {'-'*95}")
    for c in comparison:
        print(f"  {c['aspect']:<20} {c['utilitarian']:<18} {c['deontological']:<18} "
              f"{c['virtue']:<18} {c['care']:<18}")

    print()


# ============================================================
# Демо 4: AI Moral Reasoning — value-sensitive design, moral buffers
# ============================================================

def demo4_ai_moral_reasoning():
    """Демонстрация моральных рассуждений в AI."""
    print("=" * 70)
    print("Демо 4: AI Moral Reasoning — моральные рассуждения в AI")
    print("=" * 70)

    random.seed(42)

    # --- 4.1 Value-Sensitive Design ---
    print(f"\n--- 4.1 Value-Sensitive Design: чувствительный к ценностям дизайн ---")
    print("Подход: проектирование систем с учётом человеческих ценностей")

    values = [
        {"value": "Автономия", "description": "Свобода выбора пользователя",
         "design_implication": "Пользователь контролирует данные и решения"},
        {"value": "Приватность", "description": "Контроль над личной информацией",
         "design_implication": "Минимальный сбор данных, шифрование"},
        {"value": "Справедливость", "description": "Равное обращение",
         "design_implication": "Тестирование на биас, diverse training data"},
        {"value": "Прозрачность", "description": "Понимание того, как работает система",
         "design_implication": "Explainable AI, documentation"},
        {"value": "Ответственность", "description": "Ясная цепочка ответственности",
         "design_implication": "Аудит, logging, accountability"},
    ]

    print(f"\nЦенности и их дизайн-импликации:")
    for v in values:
        print(f"\n  {v['value']}: {v['description']}")
        print(f"    Дизайн: {v['design_implication']}")

    # Пример: применение VSD к системе рекомендаций
    print(f"\nПрименение VSD к системе рекомендаций:")
    vsd_application = [
        {"phase": "Conceptual", "action": "Идентификация заинтересованных сторон",
         "stakeholders": "Пользователи, создатели контента, рекламодатели"},
        {"phase": "Empirical", "action": "Исследование ценностей пользователей",
         "methods": "Опросы, фокус-группы, интервью"},
        {"phase": "Technical", "action": "Реализация ценностей в коде",
         "implementation": "Privacy by design, fairness constraints"},
    ]

    for phase in vsd_application:
        print(f"\n  [{phase['phase']}] {phase['action']}")
        for key, val in phase.items():
            if key not in ("phase", "action"):
                print(f"    {key}: {val}")

    # --- 4.2 Participatory Design ---
    print(f"\n--- 4.2 Participatory Design: участие пользователей ---")

    participation_levels = [
        {
            "level": "Информирование",
            "description": "Пользователи получают информацию о системе",
            "power": "Нет",
            "example": "Документация, FAQ",
        },
        {
            "level": "Консультация",
            "description": "Пользователи выражают мнение",
            "power": "Совещательный",
            "example": "Опросы, feedback forms",
        },
        {
            "level": "Вовлечение",
            "description": "Пользователи участвуют в обсуждении",
            "power": "Совместный",
            "example": "Фокус-группы, co-design workshops",
        },
        {
            "level": "Сотрудничество",
            "description": "Пользователи — партнёры в проектировании",
            "power": "Разделённый",
            "example": "Participatory action research",
        },
        {
            "level": "Пользовательский контроль",
            "description": "Пользователи определяют направление",
            "power": "Пользовательский",
            "example": "Community-driven development",
        },
    ]

    print(f"\nУровни участия пользователей:")
    print(f"  {'Уровень':<25} {'Описание':<35} {'Власть':<15} {'Пример'}")
    print(f"  {'-'*100}")
    for pl in participation_levels:
        print(f"  {pl['level']:<25} {pl['description']:<35} {pl['power']:<15} {pl['example']}")

    # --- 4.3 Moral Buffers ---
    print(f"\n--- 4.3 Moral Buffers: моральные буферы ---")
    print("Механизмы для предотвращения морально опасных решений AI")

    moral_buffers = [
        {
            "buffer": "Human-in-the-Loop",
            "description": "Человек подтверждает критические решения",
            "threshold": "Все решения с impact > medium",
            "implementation": "Approval queue, escalation",
        },
        {
            "buffer": "Confidence Threshold",
            "description": "При низкой уверенности → передать человеку",
            "threshold": "confidence < 0.8",
            "implementation": "Вероятностный выход, rejection option",
        },
        {
            "buffer": "Rate Limiting",
            "description": "Ограничение скорости автоматических решений",
            "threshold": "max 100 решений/час",
            "implementation": "Token bucket, cooldown",
        },
        {
            "buffer": "Audit Trail",
            "description": "Полный лог всех решений для post-factum анализа",
            "threshold": "Все решения",
            "implementation": "Immutable logging, blockchain",
        },
        {
            "buffer": "Kill Switch",
            "description": "Немедленная остановка системы",
            "threshold": "При обнаружении критической ошибки",
            "implementation": "Circuit breaker, emergency stop",
        },
    ]

    print(f"\nМоральные буферы:")
    for mb in moral_buffers:
        print(f"\n  {mb['buffer']}")
        print(f"    Описание: {mb['description']}")
        print(f"    Порог: {mb['threshold']}")
        print(f"    Реализация: {mb['implementation']}")

    # --- 4.4 Practical Example: моральный решатель ---
    print(f"\n--- 4.4 Practical Example: моральный решатель ---")

    # Простой моральный решатель на основе весов ценностей
    def moral_decision(context, value_weights):
        """Принятие морального решения на основе весов ценностей."""
        scores = {}
        for option, values in context["options"].items():
            score = sum(value_weights.get(v, 0) * w for v, w in values.items())
            scores[option] = score
        return scores

    # Контекст: автономный автомобиль
    av_context = {
        "scenario": "Авария: 2 пешехода на красный vs 1 пассажир",
        "options": {
            "Действие A (сбить пешеходов)": {
                "safety": 0.3,      # Меньше людей пострадает
                "legality": 0.9,    # Пешеходы на красный
                "empathy": 0.2,     # Пассажир — «наш»
                "fairness": 0.5,    # Оба варианта несправедливы
            },
            "Действие B (врезаться в стену)": {
                "safety": 0.7,      # Больше людей пострадает
                "legality": 0.5,    # Неясно
                "empathy": 0.8,     # Сохраняем жизнь пассажира
                "fairness": 0.4,    # Пассажир не виноват
            },
        },
    }

    # Разные наборы весов (разные культуры/ценности)
    weight_profiles = [
        {"name": "Утилитарист", "weights": {"safety": 0.7, "legality": 0.1, "empathy": 0.1, "fairness": 0.1}},
        {"name": "Деонтолог", "weights": {"safety": 0.1, "legality": 0.7, "empathy": 0.1, "fairness": 0.1}},
        {"name": "Заботливый", "weights": {"safety": 0.2, "legality": 0.1, "empathy": 0.6, "fairness": 0.1}},
        {"name": "Справедливый", "weights": {"safety": 0.2, "legality": 0.1, "empathy": 0.1, "fairness": 0.6}},
    ]

    print(f"\nМоральный решатель для автономного автомобиля:")
    print(f"  Сценарий: {av_context['scenario']}\n")

    for profile in weight_profiles:
        scores = moral_decision(av_context, profile["weights"])
        best_option = max(scores, key=scores.get)
        print(f"  {profile['name']}:")
        for option, score in scores.items():
            marker = " ← ВЫБОР" if option == best_option else ""
            print(f"    {option}: {score:.3f}{marker}")
        print()

    print(f"\n  Вывод: разные ценности → разные решения")
    print(f"  → AI не должен навязывать один набор ценностей")
    print(f"  → Решение: настраиваемые моральные параметры + human oversight")

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    demo1_trolley_problem()
    demo2_value_tradeoffs()
    demo3_moral_frameworks()
    demo4_ai_moral_reasoning()
