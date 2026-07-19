"""216 — AI Ethics Fundamentals: принципы, фреймворки, ответственный AI

Темы:
  1. Ethical Principles — beneficence, non-maleficence, autonomy, justice
  2. AI Ethics Frameworks — EU guidelines, IEEE standards, corporate principles
  3. Stakeholder Analysis — impact assessment, affected parties, power dynamics
  4. Ethical Decision Making — stakeholder analysis, trade-offs, documentation

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# =============================================================================
# Демо 1: Этические принципы AI
# =============================================================================
def demo_ethical_principles():
    print("=" * 70)
    print("Демо 1: Этические принципы AI")
    print("=" * 70)

    # --- 1.1 Принцип благотворительности (beneficence) ---
    print("\n--- 1.1 Принцип благотворительности (beneficence) ---")

    # Модель «приносит пользу» — считаем полезные действия
    ai_actions = [
        {"action": "диагностика болезни", "benefit": 9, "risk": 2},
        {"action": "рекомендация фильма", "benefit": 3, "risk": 1},
        {"action": "автопилот автомобиля", "benefit": 8, "risk": 5},
        {"action": "фильтр спама", "benefit": 5, "risk": 0},
    ]

    # Формула net_benefit = benefit - risk
    print("Формула: net_benefit = benefit - risk")
    print("Максимизация net_benefit при ограничении risk <= threshold\n")

    risk_threshold = 3
    best_action = None
    best_score = -float("inf")

    for item in ai_actions:
        nb = item["benefit"] - item["risk"]
        meets_threshold = item["risk"] <= risk_threshold
        status = "✓ допустимо" if meets_threshold else "✗ превышен порог риска"
        print(f"  {item['action']}: benefit={item['benefit']}, risk={item['risk']}, "
              f"net_benefit={nb}, {status}")
        if meets_threshold and nb > best_score:
            best_score = nb
            best_action = item["action"]

    print(f"\n  Лучшее действие по принципу beneficence: {best_action} "
          f"(net_benefit={best_score})")

    # --- 1.2 Принцип недовреда (non-maleficence) ---
    print("\n--- 1.2 Принцип недовреда (non-maleficence) ---")

    # Проверка систем на потенциальный вред
    system_checks = {
        "система найма": {
            "historical_bias": True,
            "opacity": False,
            "scale": "large",
            "irreversibility": False,
        },
        "система медицинской диагностики": {
            "historical_bias": False,
            "opacity": True,
            "scale": "large",
            "irreversibility": True,
        },
        "рекомендательная система": {
            "historical_bias": True,
            "opacity": False,
            "scale": "large",
            "irreversibility": False,
        },
    }

    # Формула риска: risk = sum(factor_scores)
    risk_factors = {"historical_bias": 3, "opacity": 2, "scale": 2, "irreversibility": 4}

    print("Факторы риска:", risk_factors)
    print("risk_score = сумма активных факторов\n")

    for system_name, checks in system_checks.items():
        risk_score = sum(risk_factors[f] for f, v in checks.items() if v)
        risk_level = "высокий" if risk_score >= 6 else "средний" if risk_score >= 3 else "низкий"
        print(f"  {system_name}: risk_score={risk_score}, уровень={risk_level}")
        for factor, active in checks.items():
            if active:
                print(f"    - {factor}: +{risk_factors[factor]}")

    # --- 1.3 Принцип автономии (autonomy) ---
    print("\n--- 1.3 Принцип автономии (autonomy) ---")

    # Взаимодействие: уважает ли AI автономию пользователя?
    interactions = [
        {"type": "согласие", "explicit": True, "revocable": True, "informed": True},
        {"type": "настройка по умолчанию", "explicit": False, "revocable": True, "informed": False},
        {"type": "мониторинг", "explicit": True, "revocable": True, "informed": False},
        {"type": "принудительное вмешательство", "explicit": False, "revocable": False, "informed": False},
    ]

    # Каждый компонент автономии даёт 1 балл, максимум 3
    print("Метрика автономии: explicit + revocable + informed (макс 3)\n")

    for inter in interactions:
        score = sum(1 for k in ["explicit", "revocable", "informed"] if inter[k])
        quality = "высокая" if score == 3 else "средняя" if score >= 2 else "низкая"
        print(f"  {inter['type']}: автономия={score}/3, качество={quality}")
        print(f"    explicit={inter['explicit']}, revocable={inter['revocable']}, "
              f"informed={inter['informed']}")

    # --- 1.4 Принцип справедливости (justice) ---
    print("\n--- 1.4 Принцип справедливости (justice) ---")

    # Распределение ресурсов по группам
    groups = {
        "группа A (основная)": {"population": 500, "benefits_received": 400, "harms_received": 50},
        "группа B (меньшинство)": {"population": 150, "benefits_received": 40, "harms_received": 45},
        "группа C (привилегированная)": {"population": 50, "benefits_received": 45, "harms_received": 2},
    }

    print("Коэффициент справедливости: benefit_rate - harm_rate\n")
    for gname, gdata in groups.items():
        pop = gdata["population"]
        benefit_rate = gdata["benefits_received"] / pop
        harm_rate = gdata["harms_received"] / pop
        justice_score = benefit_rate - harm_rate
        print(f"  {gname}:")
        print(f"    benefit_rate = {gdata['benefits_received']}/{pop} = {benefit_rate:.3f}")
        print(f"    harm_rate = {gdata['harms_received']}/{pop} = {harm_rate:.3f}")
        print(f"    justice_score = {justice_score:.3f}")

    # Гини коэффициент по benefit_rate
    rates = []
    for gdata in groups.values():
        rates.append(gdata["benefits_received"] / gdata["population"])
    rates.sort()
    n = len(rates)
    numerator = sum((2 * (i + 1) - n - 1) * rates[i] for i in range(n))
    gini = numerator / (n * sum(rates)) if sum(rates) > 0 else 0
    print(f"\n  Коэффициент неравенства Гини: {gini:.4f} (0=идеальное равенство)")


# =============================================================================
# Демо 2: Фреймворки AI Ethics
# =============================================================================
def demo_ethics_frameworks():
    print("\n\n" + "=" * 70)
    print("Демо 2: Фреймворки AI Ethics")
    print("=" * 70)

    # --- 2.1 Руководства ЕС (EU AI Act) ---
    print("\n--- 2.1 Руководства ЕС (EU AI Act) ---")

    # Классификация систем AI по уровню риска
    eu_risk_levels = {
        "неприемлемый риск": {
            "examples": ["социальный рейтинг", "массовая слежка"],
            "requirements": ["полный запрет"],
        },
        "высокий риск": {
            "examples": ["кредитный скоринг", "медицинская диагностика", "найм"],
            "requirements": [
                "оценка воздействия",
                "прозрачность",
                "человеческий контроль",
                "мониторинг",
            ],
        },
        "ограниченный риск": {
            "examples": ["чат-бот", "распознавание эмоций"],
            "requirements": ["раскрытие факта AI", "прозрачность"],
        },
        "минимальный риск": {
            "examples": ["фильтр спама", "игровой AI"],
            "requirements": ["базовые требования"],
        },
    }

    print("Классификация EU AI Act:\n")
    for level, data in eu_risk_levels.items():
        print(f"  [{level}]")
        print(f"    Примеры: {', '.join(data['examples'])}")
        print(f"    Требования: {', '.join(data['requirements'])}")

    # Симуляция оценки конкретной системы
    system_name = "система оценки кредитоспособности"
    risk_factors_present = ["personal_data", "financial_impact", "automated_decision"]
    risk_score = len(risk_factors_present) * 2 + 1  # формула риска
    determined_level = "высокий риск" if risk_score >= 5 else "ограниченный риск"

    print(f"\n  Оценка системы '{system_name}':")
    print(f"    Факторы риска: {risk_factors_present}")
    print(f"    Risk score = {risk_score} → уровень: {determined_level}")

    # --- 2.2 IEEE Standards ---
    print("\n--- 2.2 IEEE Standards для этики AI ---")

    ieee_principles = {
        "устойчивость": {
            "описание": "AI не должен вредить окружающей среде",
            "метрики": ["carbon_footprint", "energy_efficiency"],
            "вес": 0.2,
        },
        "прозрачность": {
            "описание": "Решения AI должны быть объяснимы",
            "метрики": ["explainability_score", "documentation_completeness"],
            "вес": 0.3,
        },
        "подотчётность": {
            "описание": "Должны быть определены ответственные лица",
            "метрики": ["audit_trail_coverage", "liability_clarity"],
            "вес": 0.25,
        },
        "безопасность": {
            "описание": "Защита от вреда и злоупотреблений",
            "метрики": ["security_test_coverage", "incident_response_time"],
            "вес": 0.25,
        },
    }

    # Оценка системы по каждому принципу
    scores = {"устойчивость": 0.8, "прозрачность": 0.6, "подотчётность": 0.7, "безопасность": 0.9}

    print("Принципы IEEE и оценка:\n")
    weighted_total = 0
    for principle, data in ieee_principles.items():
        score = scores[principle]
        weighted = score * data["вес"]
        weighted_total += weighted
        print(f"  {principle}: score={score:.1f}, вес={data['вес']:.2f}, "
              f"взвешенный={weighted:.3f}")
        print(f"    Метрики: {', '.join(data['метрики'])}")

    print(f"\n  Итоговый IEEE compliance score: {weighted_total:.3f} / 1.0")

    # --- 2.3 Корпоративные принципы ---
    print("\n--- 2.3 Корпоративные принципы (Big Tech) ---")

    companies = {
        "Google": ["не навредить", "прозрачность", "справедливость", "конфиденциальность",
                    "ответственное использование"],
        "Microsoft": ["надёжность", "безопасность", "приватность", "прозрачность",
                       "справедливость", "инклюзивность"],
        "OpenAI": ["полезность", "безопасность", "прозрачность",
                    "содействие общей выгоде"],
    }

    # Находим общие принципы
    all_sets = [set(principles) for principles in companies.values()]
    common_principles = all_sets[0]
    for s in all_sets[1:]:
        common_principles = common_principles & s

    print("Корпоративные принципы и их пересечение:\n")
    for company, principles in companies.items():
        print(f"  {company}: {', '.join(principles)}")

    print(f"\n  Общие принципы всех компаний: {', '.join(common_principles) if common_principles else 'нет'}")

    # Jaccard similarity между парами
    print("\n  Коеффициент Жаккара между компаниями:")
    company_names = list(companies.keys())
    for i in range(len(company_names)):
        for j in range(i + 1, len(company_names)):
            set_a = set(companies[company_names[i]])
            set_b = set(companies[company_names[j]])
            jaccard = len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0
            print(f"    {company_names[i]} ∩ {company_names[j]}: "
                  f"{jaccard:.3f} ({len(set_a & set_b)} общих из {len(set_a | set_b)} уникальных)")

    # --- 2.4 Сравнение фреймворков ---
    print("\n--- 2.4 Сравнение фреймворков ---")

    frameworks = {
        "EU AI Act": {"age": 2024, "scope": "международный", "enforceability": 0.95,
                      "specificity": 0.90, "flexibility": 0.40},
        "IEEE P7000": {"age": 2021, "scope": "международный", "enforceability": 0.30,
                       "specificity": 0.70, "flexibility": 0.80},
        "NIST AI RMF": {"age": 2023, "scope": "американский", "enforceability": 0.20,
                        "specificity": 0.75, "flexibility": 0.85},
        "Corporates": {"age": 2020, "scope": "внутренний", "enforceability": 0.10,
                       "specificity": 0.50, "flexibility": 0.95},
    }

    print("Мультикритериальная оценка фреймворков:\n")
    weights = {"enforceability": 0.35, "specificity": 0.30, "flexibility": 0.35}
    for fname, fdata in frameworks.items():
        score = sum(fdata[k] * weights[k] for k in weights)
        print(f"  {fname}: composite={score:.3f}")
        for k in weights:
            print(f"    {k}: {fdata[k]:.2f} × {weights[k]:.2f} = {fdata[k] * weights[k]:.3f}")


# =============================================================================
# Демо 3: Анализ стейкхолдеров
# =============================================================================
def demo_stakeholder_analysis():
    print("\n\n" + "=" * 70)
    print("Демо 3: Анализ стейкхолдеров")
    print("=" * 70)

    # --- 3.1 Оценка воздействия (impact assessment) ---
    print("\n--- 3.1 Оценка воздействия (impact assessment) ---")

    stakeholders = [
        {"name": "пользователи", "influence": 0.3, "impact": 0.9, "interest": 0.8},
        {"name": "инвесторы", "influence": 0.8, "impact": 0.5, "interest": 0.7},
        {"name": "регуляторы", "influence": 0.9, "impact": 0.7, "interest": 0.6},
        {"name": "сотрудники", "influence": 0.5, "impact": 0.6, "interest": 0.9},
        {"name": "общество", "influence": 0.2, "impact": 0.8, "interest": 0.3},
    ]

    # Классификация по матрице influence × impact
    print("Матрица стейкхолдеров: influence × impact\n")
    for s in stakeholders:
        power = s["influence"] * s["impact"]
        quadrant = ""
        if s["influence"] >= 0.5 and s["impact"] >= 0.5:
            quadrant = "управляй (высокое влияние + высокое воздействие)"
        elif s["influence"] >= 0.5:
            quadrant = "информируй (высокое влияние)"
        elif s["impact"] >= 0.5:
            quadrant = "вовлеки (высокое воздействие)"
        else:
            quadrant = "мониторь (низкое)"

        print(f"  {s['name']}: influence={s['influence']}, impact={s['impact']}, "
              f"power={power:.2f}")
        print(f"    → {quadrant}")

    # --- 3.2 Затронутые стороны ---
    print("\n--- 3.2 Затронутые стороны (affected parties) ---")

    ai_system = "система автономного найма"
    affected = [
        {"group": "кандидаты", "count": 10000, "disproportionately_affected": True,
         "ability_to_opt_out": False, "representation_in_data": 0.3},
        {"group": "HR-специалисты", "count": 50, "disproportionately_affected": False,
         "ability_to_opt_out": True, "representation_in_data": 0.9},
        {"group": " работодатели", "count": 200, "disproportionately_affected": False,
         "ability_to_opt_out": True, "representation_in_data": 0.8},
        {"group": "нетрадиционные кандидаты", "count": 2000, "disproportionately_affected": True,
         "ability_to_opt_out": False, "representation_in_data": 0.1},
    ]

    # Формула уязвимости: vulnerability = disproportionately_affected × (1 - opt_out) × (1 - representation)
    print(f"Система: {ai_system}")
    print("Формула: vulnerability = disproportionate × (1 - opt_out) × (1 - representation)\n")

    for party in affected:
        vuln = (1.0 if party["disproportionately_affected"] else 0.0)
        vuln *= (1.0 - (1.0 if party["ability_to_opt_out"] else 0.0))
        vuln *= (1.0 - party["representation_in_data"])
        vuln_level = "высокая" if vuln >= 0.5 else "средняя" if vuln >= 0.2 else "низкая"
        print(f"  {party['group']} (n={party['count']}): vulnerability={vuln:.3f} → {vuln_level}")
        print(f"    representation={party['representation_in_data']}, "
              f"opt_out={party['ability_to_opt_out']}")

    # --- 3.3 Динамика силы ---
    print("\n--- 3.3 Динамика силы (power dynamics) ---")

    power_relations = [
        ("корпорация", "разработчик", 0.9, "прямая"),
        ("корпорация", "пользователь", 0.7, "асимметричная"),
        ("разработчик", "данные", 0.8, "прямая"),
        ("регулятор", "корпорация", 0.5, "ограниченная"),
        ("пользователь", "регулятор", 0.3, "косвенная"),
    ]

    print("Индекс асимметрии: power_ratio = power_source / power_target\n")
    for source, target, power, relation_type in power_relations:
        # Нормализуем power_ratio через sigmoid-like функцию
        asymmetry = power / (1.0 + (1.0 - power))  # смещённая формула
        print(f"  {source} → {target}: power={power:.1f}, тип={relation_type}")
        print(f"    asymmetry_index={asymmetry:.3f}")

    # --- 3.4 Документация этического воздействия ---
    print("\n--- 3.4 Документация этического воздействия ---")

    impact_doc = {
        "система": "AI-ассистент для медицинских рекомендаций",
        "дата_оценки": "2024-01-15",
        "версия": "2.1",
        "оценки": [
            {
                "категория": "прозрачность",
                "описание": "Пациенты не могут понять, как формируется рекомендация",
                "уровень_риска": "высокий",
                "меры": ["добавить объяснение", "предоставить альтернативы"],
            },
            {
                "категория": "справедливость",
                "описание": "Тестирование показало 15% разницу точности по демографиям",
                "уровень_риска": "высокий",
                "меры": ["ресемплинг данных", "калибровка по группам"],
            },
            {
                "категория": "безопасность",
                "описание": "Возможны ложные рекомендации при нетипичных симптомах",
                "уровень_риска": "средний",
                "меры": ["ограничение scope", "обязательная верификация врачом"],
            },
        ],
        "подпись_оценщика": hashlib.sha256("ethics_review_2024".encode()).hexdigest()[:16],
    }

    print(json.dumps(impact_doc, indent=2, ensure_ascii=False)[:800] + "...")

    # Считаем статистику
    risk_levels = [e["уровень_риска"] for e in impact_doc["оценки"]]
    risk_counter = collections.Counter(risk_levels)
    print(f"\n  Статистика: {dict(risk_counter)}")
    total_measures = sum(len(e["меры"]) for e in impact_doc["оценки"])
    print(f"  Всего мер: {total_measures} по {len(impact_doc['оценки'])} категориям")


# =============================================================================
# Демо 4: Этическое принятие решений
# =============================================================================
def demo_ethical_decision_making():
    print("\n\n" + "=" * 70)
    print("Демо 4: Этическое принятие решений")
    print("=" * 70)

    # --- 4.1 Анализ стейкхолдеров для решения ---
    print("\n--- 4.1 Анализ стейкхолдеров для решения ---")

    decision = "Запуск системы автоматического скрининга резюме"

    affected_groups = [
        {"group": "соискатели", "interest": 0.9, "power": 0.2, "concern": "дискриминация"},
        {"group": "HR-отдел", "interest": 0.8, "power": 0.6, "concern": "эффективность"},
        {"group": "юридический отдел", "interest": 0.5, "power": 0.7, "concern": "合规性"},
        {"group": "IT-отдел", "interest": 0.6, "power": 0.5, "concern": "надёжность"},
    ]

    print(f"Решение: {decision}\n")
    print("Матрица стейкхолдеров:")
    print(f"  {'Группа':<20} {'Интерес':>8} {'Власть':>8} {'Приоритет':>10}")
    print("  " + "-" * 50)

    priorities = []
    for g in affected_groups:
        # Приоритет = interest × power + interest × 0.3 (для учёта уязвимости)
        priority = g["interest"] * g["power"] + g["interest"] * 0.3
        priorities.append((g["group"], priority, g["concern"]))
        print(f"  {g['group']:<20} {g['interest']:>8.2f} {g['power']:>8.2f} {priority:>10.3f}")

    # Сортируем по приоритету
    priorities.sort(key=lambda x: x[1], reverse=True)
    print("\n  Приоритет рассмотрения:")
    for i, (group, priority, concern) in enumerate(priorities, 1):
        print(f"    {i}. {group} (приоритет={priority:.3f}, основная озабоченность: {concern})")

    # --- 4.2 Компромиссы (trade-offs) ---
    print("\n--- 4.2 Компромиссы (trade-offs) ---")

    options = [
        {
            "название": "полный запуск",
            "ценность": {"эффективность": 0.9, "справедливость": 0.3,
                          "прозрачность": 0.2, "безопасность": 0.4},
        },
        {
            "название": "ограниченный пилот",
            "ценность": {"эффективность": 0.5, "справедливость": 0.7,
                          "прозрачность": 0.7, "безопасность": 0.7},
        },
        {
            "название": "отложить запуск",
            "ценность": {"эффективность": 0.1, "справедливость": 0.9,
                          "прозрачность": 0.9, "безопасность": 0.9},
        },
    ]

    # Разные этические перспективы с весами
    perspectives = {
        "утилитарная": {"эффективность": 0.5, "справедливость": 0.2,
                         "прозрачность": 0.1, "безопасность": 0.2},
        "деонтологическая": {"эффективность": 0.1, "справедливость": 0.3,
                               "прозрачность": 0.3, "безопасность": 0.3},
        "этика добродетели": {"эффективность": 0.2, "справедливость": 0.3,
                               "прозрачность": 0.3, "безопасность": 0.2},
    }

    print("Веса по этическим перспективам:\n")
    for pname, weights in perspectives.items():
        print(f"  {pname}: {weights}")

    print("\nРезультаты для каждой перспективы:\n")
    for pname, weights in perspectives.items():
        print(f"  {pname}:")
        best_option = None
        best_score = -1
        for opt in options:
            score = sum(opt["ценность"][k] * weights[k] for k in weights)
            if score > best_score:
                best_score = score
                best_option = opt["название"]
            print(f"    {opt['название']}: {score:.3f}")
        print(f"    → лучший выбор: {best_option}\n")

    # --- 4.3 Документирование решений ---
    print("--- 4.3 Документирование решений ---")

    decision_record = {
        "проблема": "Выбор между скоростью запуска и тщательностью тестирования fairness",
        "контекст": "Система скрининга резюме, затрагивает 5000+ кандидатов в год",
        "рассмотренные_альтернативы": [
            "полный запуск (риск: дискриминация)",
            "ограниченный пилот (риск: задержка)",
            "отложить запуск (риск: потеря эффективности)",
        ],
        "критерии_решения": [
            "минимальный вред для уязвимых групп",
            "воспроизводимость и объяснимость",
            "соответствие нормативным требования",
        ],
        "выбранное_решение": "ограниченный пилот с мониторингом fairness",
        "обоснование": "Баланс между利益 и справедливостью; пилот позволяет выявить проблемы "
                       "до масштабирования",
        "результат": {
            "кандидатов_затронуто": 500,
            "обнаружено_проблем": 2,
            "принято_мер": 2,
            "fairness_before": 0.65,
            "fairness_after": 0.82,
        },
    }

    print(json.dumps(decision_record, indent=2, ensure_ascii=False)[:600] + "...")

    # Метрики документации
    completeness = sum(1 for k in decision_record if decision_record[k]) / len(decision_record)
    print(f"\n  Полнота документации: {completeness * 100:.0f}%")

    # --- 4.4 Этический аудит-чеклист ---
    print("\n--- 4.4 Этический аудит-чеклист ---")

    checklist = [
        ("выявление стейкхолдеров", True),
        ("оценка воздействия", True),
        ("проверка данных на смещения", True),
        ("тестирование fairness", False),
        ("планирование мониторинга", False),
        ("документирование решений", True),
        ("процесс обжалования", False),
        ("периодический пересмотр", False),
    ]

    completed = sum(1 for _, done in checklist if done)
    total = len(checklist)

    print(f"Прогресс аудита: {completed}/{total} ({completed / total * 100:.0f}%)\n")
    for item, done in checklist:
        status = "✓" if done else "✗"
        print(f"  [{status}] {item}")

    # Оценка риска незавершённых пунктов
    incomplete_items = [item for item, done in checklist if not done]
    risk_score = len(incomplete_items) / total * 10
    print(f"\n  Риск незавершённого аудита: {risk_score:.1f}/10")
    print(f"  Незавершённые пункты: {', '.join(incomplete_items)}")


# =============================================================================
# Точка входа
# =============================================================================
if __name__ == "__main__":
    demo_ethical_principles()
    demo_ethics_frameworks()
    demo_stakeholder_analysis()
    demo_ethical_decision_making()
