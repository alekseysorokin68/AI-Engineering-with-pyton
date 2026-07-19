"""227 — AI Regulation: EU AI Act, отраслевые правила, международные стандарты

Темы:
  1. EU AI Act — категории риска, запрещённое ИИ, требования прозрачности
  2. Sector Regulations — здравоохранение, финансы, автономный транспорт
  3. International Standards — ISO, NIST, IEEE, принципы OECD
  4. Compliance Implementation — документация, тестирование, отчётность

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# 1. EU AI Act — Европейский регламент ИИ
# ─────────────────────────────────────────────────────────────────────

def demo_eu_ai_act():
    print("=" * 70)
    print("DEMO 1 — EU AI Act: категории риска и требования регулирования")
    print("=" * 70)

    # --- 1.1 Категории риска EU AI Act ---
    print("\n--- 1.1 Четыре категории риска EU AI Act ---")
    risk_levels = {
        "Неприемлемый риск (PROHIBITED)": {
            "уровень": 4,
            "штраф": "35M€ или 7% глобальной выручки",
            "примеры": [
                "Социальное кредитование (Social scoring)",
                "Биометрическая идентификация на расстоянии",
                "Эксплуатация уязвимых групп",
                "Манипуляция поведением",
                "Предиктивная полицейская (на основе профилей)",
            ],
            "цвет": "🔴",
        },
        "Высокий риск (HIGH RISK)": {
            "уровень": 3,
            "штраф": "15M€ или 3% глобальной выручки",
            "примеры": [
                "Системы найма и оценки работников",
                "Кредитный скоринг",
                "Диагностические медицинские ИИ",
                "Образовательные оценочные системы",
                "Управление автономным транспортом",
            ],
            "цвет": "🟠",
        },
        "Ограниченный риск (LIMITED RISK)": {
            "уровень": 2,
            "штраф": "N/A — требования прозрачности",
            "примеры": [
                "Чат-боты (должны раскрывать свою природу)",
                "Генеративный контент (водяные знаки)",
                "Системы распознавания эмоций",
            ],
            "цвет": "🟡",
        },
        "Минимальный риск (MINIMAL RISK)": {
            "уровень": 1,
            "штраф": "Нет специальных требований",
            "примеры": [
                "Спам-фильтры",
                "Рекомендательные системы",
                "Игровые ИИ",
                "Виртуальные помощники",
            ],
            "цвет": "🟢",
        },
    }
    for level_name, info in risk_levels.items():
        print(f"\n  {info['цвет']} {level_name}")
        print(f"    Штрафы: {info['штраф']}")
        print(f"    Примеры:")
        for ex in info["примеры"]:
            print(f"      • {ex}")

    # --- 1.2 Запрещённые практики ---
    print("\n--- 1.2 Запрещённые ИИ-практики (Article 5) ---")
    prohibited = [
        ("Сублиминальные техники", "Манипуляция подсознанием людей"),
        ("Эксплуатация уязвимостей", "Возраст, инвалидность, соц. положение"),
        ("Социальное кредитование", "Оценка надёжности граждан государством"),
        ("Биометрическая категоризация", "Политические взгляды, религия, раса"),
        ("Предиктивная полиция", "Оценка риска преступления по профилю"),
        ("Сбор эмоций на работе", "Мониторинг эмоций сотрудников"),
        ("Скрапинг лиц из интернета", "Создание баз без согласия"),
    ]
    for practice, desc in prohibited:
        print(f"  ✗ {practice}: {desc}")

    # --- 1.3 Требования прозрачности ---
    print("\n--- 1.3 Требования прозрачности для генеративного ИИ ---")
    transparency_reqs = [
        ("Раскрытие ИИ-природы", "Пользователь должен знать, что взаимодействует с ИИ"),
        ("Водяные знаки", "Метки для контента, сгенерированного ИИ"),
        ("Документация训练数据", "Описание данных, использованных для обучения"),
        ("Раскрытие capabilities", "Ограничения и возможности системы"),
        ("Обработка жалоб", "Процедура рассмотрения ошибок"),
    ]
    print("  Согласно EU AI Act, провайдеры генеративного ИИ должны:")
    for req, desc in transparency_reqs:
        print(f"    ✓ {req}: {desc}")

    # --- 1.4 Штрафы и шкала ---
    print("\n--- 1.4 Штрафная шкала EU AI Act ---")
    violations = [
        ("Нарушение запрещённых практик", 35, "7%"),
        ("Нарушение требований HIGH RISK", 15, "3%"),
        ("Непредоставление информации", 7.5, "1%"),
        ("Нарушение требований к обучению", 7.5, "1%"),
        ("Невыполнение корректирующих мер", 3, "0.5%"),
    ]
    print("  {'Нарушение':<40s} {'Макс.штраф':>12s} {'% выручки':>10s}")
    for violation, fine_mln, pct in violations:
        bar = "█" * int(fine_mln * 2)
        print(f"    {violation:<40s} {fine_mln:>10.1f}M€ {pct:>8s}  {bar}")


# ─────────────────────────────────────────────────────────────────────
# 2. Sector Regulations — отраслевые правила
# ─────────────────────────────────────────────────────────────────────

def demo_sector_regulations():
    print("\n" + "=" * 70)
    print("DEMO 2 — Sector Regulations: здравоохранение, финансы, транспорт")
    print("=" * 70)

    # --- 2.1 Медицинское регулирование ---
    print("\n--- 2.1 Регулирование ИИ в здравоохранении ---")
    medical_ai = [
        ("Диагностический ИИ (FDA)", "Class II/III", "510(k) или PMA", "12-18 мес"),
        ("Медицинский чат-бот", "Class I", "510(k)", "6-12 мес"),
        ("ИИ для разработки лекарств", "Class III", "PMA", "24-36 мес"),
        ("ИИ-мониторинг пациентов", "Class II", "510(k)", "12-18 мес"),
    ]
    print("  {'Система':<30s} {'Класс':>10s} {'Процедура':>14s} {'Сроки':>10s}")
    for system, cls, procedure, timeline in medical_ai:
        print(f"    {system:<30s} {cls:>10s} {procedure:>14s} {timeline:>10s}")
    # Ключевые требования FDA
    print("\n  Требования FDA для ИИ/ML в медицине:")
    fda_reqs = [
        "Predetermined Change Control Plan (PCCP)",
        "Good Machine Learning Practice (GMLP)",
        "Клиническая валидация на репрезентативных данных",
        "Мониторинг post-market performance",
        "Управление переобучением (locked vs adaptive algorithms)",
    ]
    for req in fda_reqs:
        print(f"    • {req}")

    # --- 2.2 Финансовое регулирование ---
    print("\n--- 2.2 Регулирование ИИ в финансах ---")
    finance_regulations = [
        ("ECOA (США)", "Запрет дискриминации в кредитовании", "Объяснимость решений"),
        ("SR 11-7 (ФРС)", "Управление моделями", "Model Risk Management"),
        ("GDPR (ЕС)", "Право на объяснение", "Automated decision-making"),
        ("MiFID II (ЕС)", "Алгоритмическая торговля", "Контроль и мониторинг"),
        ("DORA (ЕС)", "Цифровая Operational Resilience", "Стресс-тестирование ИИ"),
    ]
    print("  {'Регулятор':<18s} {'Фокус':<35s} {'Требование':<30s}")
    for regulator, focus, requirement in finance_regulations:
        print(f"    {regulator:<18s} {focus:<35s} {requirement:<30s}")

    # Модель оценки bias в кредитном скоринге
    print("\n  Проверка bias в кредитном ИИ:")
    groups = [
        ("Основная группа", 0.72, 1000),
        ("Этническое меньшинство", 0.58, 500),
        ("Женщины", 0.68, 800),
        ("Молодёжь (<25)", 0.45, 300),
        ("Пожилые (>65)", 0.52, 400),
    ]
    base_rate = 0.72  # Основная группа
    print(f"  {'Группа':<25s} {'Approval':>10s} {'Disparity':>12s} {'Status':>10s}")
    for group, rate, count in groups:
        disparity = rate / base_rate
        status = "✓ OK" if disparity > 0.80 else "⚠ BIAS"
        print(f"    {group:<25s} {rate*100:>8.0f}%  {disparity:>10.2f}x  {status:>10s}")

    # --- 2.3 Автономный транспорт ---
    print("\n--- 2.3 Регулирование автономного транспорта ---")
    sae_levels = [
        (0, "Нет автоматизации", "Полный контроль водителя", "Не требует ИИ-регулирования"),
        (1, "Assisted", "Удержание полосы, круиз", "Базовые стандарты безопасности"),
        (2, "Partial", "Парковка, автопилот на шоссе", "ISO 26262 (ASIL-B)"),
        (3, "Conditional", "Водитель должен быть готов", "ISO 21448 (SOTIF)"),
        (4, "High", "Автономное в определённых зонах", "Полное ИИ-регулирование"),
        (5, "Full", "Полная автономность", "Максимальные требования"),
    ]
    print("  {'Уровень':>8s} {'Описание':<30s} {'Требования':<40s}")
    for level, desc, requirements, reg in sae_levels:
        print(f"    SAE {level}: {desc:<30s} {reg:<40s}")

    # --- 2.4 Сравнение регуляторных подходов ---
    print("\n--- 2.4 Сравнение регуляторных подходов по регионам ---")
    approaches = [
        ("ЕС (EU AI Act)", "Risk-based", "Обязательно", "Конфиденциально", "CE-маркировка"),
        ("США (Executive Order)", "Sector-specific", "Рекомендуется", "Отраслевые", "NIST Framework"),
        ("Китай", "Content control", "Обязательно", "Гос. контроль", "Регистрация"),
        ("Великобритания", "Pro-innovation", "Рекомендуется", "Саморегулирование", "Sandbox"),
        ("Канада", "Risk-based", "Обязательно", "PIPEDA", "Impact Assessment"),
    ]
    print("  {'Регион':<22s} {'Подход':<18s} {'Обязательность':>14s} {'Данные':<14s} {'Маркировка':<14s}")
    for region, approach, mandatory, data, marking in approaches:
        print(f"    {region:<22s} {approach:<18s} {mandatory:>14s} {data:<14s} {marking:<14s}")


# ─────────────────────────────────────────────────────────────────────
# 3. International Standards — международные стандарты
# ─────────────────────────────────────────────────────────────────────

def demo_international_standards():
    print("\n" + "=" * 70)
    print("DEMO 3 — International Standards: ISO, NIST, IEEE, OECD")
    print("=" * 70)

    # --- 3.1 ISO стандарты для ИИ ---
    print("\n--- 3.1 Ключевые ISO-стандарты для ИИ ---")
    iso_standards = [
        ("ISO/IEC 22989:2022", "AI Concepts & Terminology", "Основные определения ИИ"),
        ("ISO/IEC 23053:2023", "ML Framework", "Фреймворк систем машинного обучения"),
        ("ISO/IEC 42001:2023", "AI Management System", "Система управления ИИ (AIMS)"),
        ("ISO/IEC 24027:2021", "AI Bias in Systems", "Оценка и устранение bias"),
        ("ISO/IEC 25059:2023", "AI Quality Model", "Модель качества ИИ-систем"),
    ]
    print("  {'Стандарт':<22s} {'Название':<30s} {'Описание':<35s}")
    for code, name, desc in iso_standards:
        print(f"    {code:<22s} {name:<30s} {desc:<35s}")

    # --- 3.2 NIST AI RMF ---
    print("\n--- 3.2 NIST AI Risk Management Framework ---")
    nist_functions = [
        ("GOVERN", [
            "Establish AI risk management culture",
            "Define roles and responsibilities",
            "Implement oversight mechanisms",
            "Document AI system decisions",
        ]),
        ("MAP", [
            "Identify intended purpose and context",
            "Catalog AI system components",
            "Assess impact on stakeholders",
            "Document assumptions and limitations",
        ]),
        ("MEASURE", [
            "Implement performance metrics",
            "Monitor for bias and fairness",
            "Track reliability and robustness",
            "Evaluate explainability",
        ]),
        ("MANAGE", [
            "Respond to identified risks",
            "Implement risk treatment plans",
            "Communicate with stakeholders",
            "Continuously improve practices",
        ]),
    ]
    print("  Четыре основные функции NIST AI RMF:")
    for function, items in nist_functions:
        print(f"\n  {function}:")
        for item in items:
            print(f"    • {item}")

    # --- 3.3 IEEE Ethically Aligned Design ---
    print("\n--- 3.3 IEEE Ethically Aligned Design: ключевые принципы ---")
    ieee_principles = [
        ("Human Rights", "ИИ должен защищать права человека"),
        ("Well-being", "ИИ должен способствовать благополучию"),
        ("Accountability", "Прозрачная ответственность за решения ИИ"),
        ("Transparency", "Понятность процессов принятия решений"),
        ("Awareness of misuse", "Предотвращение злоупотреблений"),
        ("Sustainability", "Экологическая устойчивость ИИ"),
    ]
    for i, (principle, desc) in enumerate(ieee_principles, 1):
        print(f"  {i}. {principle}: {desc}")

    # --- 3.4 OECD AI Principles ---
    print("\n--- 3.4 OECD AI Principles (2019, 46 стран) ---")
    oecd_principles = [
        ("Инклюзивный рост", "Содействие экономическому росту и благосостоянию"),
        ("Человекоцентрированный", "Соответствие правам человека и демократическим ценностям"),
        ("Прозрачность", "Объяснимость решений ИИ"),
        ("Безопасность", "Надёжность и кибербезопасность"),
        ("Ответственность", "Организации должны нести ответственность за ИИ"),
    ]
    print("  Пять основных принципов:")
    for principle, desc in oecd_principles:
        print(f"    ✓ {principle}: {desc}")

    # Матрица соответствия стандартов
    print("\n--- Матрица соответствия стандартов ---")
    standards_matrix = [
        ("ISO 42001", [1, 1, 1, 0, 1]),
        ("NIST AI RMF", [1, 1, 1, 1, 1]),
        ("IEEE EAD", [1, 1, 1, 0, 1]),
        ("EU AI Act", [1, 1, 1, 1, 0]),
    ]
    principles_short = ["Human Rights", "Well-being", "Accountability", "Transparency", "Sustainability"]
    header = "  {'Стандарт':<16s}" + "".join(f" {p:>14s}" for p in principles_short)
    print(header)
    for standard, coverage in standards_matrix:
        row = f"    {standard:<16s}" + "".join(f" {'✓' if c else '·':>14s}" for c in coverage)
        print(row)


# ─────────────────────────────────────────────────────────────────────
# 4. Compliance Implementation — внедрение compliance
# ─────────────────────────────────────────────────────────────────────

def demo_compliance_implementation():
    print("\n" + "=" * 70)
    print("DEMO 4 — Compliance Implementation: документация, тестирование, отчётность")
    print("=" * 70)

    # --- 4.1 Требуемая документация ---
    print("\n--- 4.1 Требуемая документация для HIGH RISK ИИ ---")
    documentation = [
        ("System Card", "Описание системы, назначение, ограничения", "Обязательно"),
        ("Data Sheet", "Источники данных, процесс сбора, bias analysis", "Обязательно"),
        ("Model Card", "Архитектура, метрики, ограничения", "Обязательно"),
        ("Risk Assessment", "Оценка рисков и мер по снижению", "Обязательно"),
        ("Test Report", "Результаты тестирования на fairness/robustness", "Обязательно"),
        ("Monitoring Plan", "План мониторинга post-deployment", "Обязательно"),
        ("Incident Response", "Процедура реагирования на инциденты", "Рекомендуется"),
        ("User Guide", "Инструкция для пользователей", "Рекомендуется"),
    ]
    print("  {'Документ':<20s} {'Описание':<50s} {'Статус':>12s}")
    for doc, desc, status in documentation:
        marker = "✓" if status == "Обязательно" else "○"
        print(f"    {marker} {doc:<18s} {desc:<50s} {status:>12s}")

    # --- 4.2 Framework для тестирования ---
    print("\n--- 4.2 Фреймворк тестирования ИИ-системы ---")
    test_categories = [
        ("Fairness Testing", [
            "Demographic parity: P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)",
            "Equalized odds: P(Ŷ=1|A=0,Y=y) ≈ P(Ŷ=1|A=1,Y=y)",
            "Calibration: P(Y=1|Ŷ=s,A=0) ≈ P(Y=1|Ŷ=s,A=1)",
        ]),
        ("Robustness Testing", [
            "Adversarial examples (FGSM, PGD)",
            "Distribution shift detection",
            "Out-of-distribution robustness",
        ]),
        ("Explainability Testing", [
            "Feature importance consistency (SHAP/LIME)",
            "Counterfactual explanations",
            "Rule extraction from black-box models",
        ]),
        ("Security Testing", [
            "Data poisoning resistance",
            "Model extraction defense",
            "Prompt injection resistance (для LLM)",
        ]),
    ]
    for category, tests in test_categories:
        print(f"\n  {category}:")
        for test in tests:
            print(f"    • {test}")

    # --- 4.3 Compliance Score Calculator ---
    print("\n--- 4.3 Калькулятор Compliance Score ---")
    # Модель: Compliance = Σ(weight_i × completion_i) / Σ(weight_i)
    compliance_items = [
        ("Documentation complete", 0.15, 1.0),
        ("Fairness testing passed", 0.20, 0.85),
        ("Robustness testing passed", 0.15, 0.70),
        ("Explainability provided", 0.10, 0.90),
        ("Human oversight configured", 0.10, 0.80),
        ("Data governance established", 0.10, 0.95),
        ("Monitoring pipeline active", 0.10, 0.60),
        ("Incident response plan", 0.05, 0.40),
        ("User notification system", 0.05, 0.75),
    ]
    total_weight = sum(w for _, w, _ in compliance_items)
    weighted_sum = sum(w * c for _, w, c in compliance_items)
    compliance_score = weighted_sum / total_weight
    print(f"  Формула: Score = Σ(weight × completion) / Σ(weight)")
    print(f"\n  {'Компонент':<35s} {'Вес':>6s} {'Выполнение':>12s} {'Вклад':>8s}")
    for item, weight, completion in compliance_items:
        contribution = weight * completion
        bar = "█" * int(completion * 20)
        print(f"    {item:<35s} {weight:>5.0%} {completion*100:>10.0f}%  {contribution:>6.3f} {bar}")
    print(f"\n  ИТОГОВЫЙ Compliance Score: {compliance_score:.1%}")
    if compliance_score >= 0.9:
        status = "✓ СООТВЕТСТВУЕТ"
    elif compliance_score >= 0.7:
        status = "⚠ ЧАСТИЧНОЕ СООТВЕТСТВИЕ"
    else:
        status = "✗ НЕ СООТВЕТСТВУЕТ"
    print(f"  Статус: {status}")

    # --- 4.4 Процесс непрерывного compliance ---
    print("\n--- 4.4 Процесс непрерывного Compliance (после развёртывания) ---")
    continuous_process = [
        ("День 1-7", "Baseline assessment", "Установка baseline метрик"),
        ("День 8-30", "Initial monitoring", "Сбор данных мониторинга"),
        ("День 31-90", "Bias drift detection", "Обнаружение drift в fairness"),
        ("День 91-180", "Performance audit", "Аудит производительности"),
        ("День 181-365", "Full compliance review", "Полный пересмотр compliance"),
        ("Постоянно", "Incident response", "Реагирование на инциденты"),
        ("Квартально", "Stakeholder reporting", "Отчётность перед заинтересованными сторонами"),
    ]
    print("  {'Этап':<18s} {'Действие':<25s} {'Описание':<40s}")
    for timeline, action, description in continuous_process:
        print(f"    {timeline:<18s} {action:<25s} {description:<40s}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_eu_ai_act()
    demo_sector_regulations()
    demo_international_standards()
    demo_compliance_implementation()
