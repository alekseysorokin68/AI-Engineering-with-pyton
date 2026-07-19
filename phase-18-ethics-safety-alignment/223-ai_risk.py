"""
223 — AI Risk Management: оценка рисков, стратегии смягчения

Темы:
  1. Risk Assessment (likelihood × impact, risk matrix, hazard analysis)
  2. Failure Modes (hallucination, bias amplification, misuse, automation bias)
  3. Mitigation Strategies (red teaming, sandboxing, monitoring, human oversight)
  4. Risk Communication (severity levels, reporting, stakeholder notification)

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
# Демо 1: Risk Assessment — likelihood × impact, risk matrix
# ============================================================

def demo1_risk_assessment():
    """Демонстрация методов оценки рисков AI-систем."""
    print("=" * 70)
    print("Демо 1: Risk Assessment — оценка рисков AI-систем")
    print("=" * 70)

    # --- 1.1 Risk Scoring: Likelihood × Impact ---
    print(f"\n--- 1.1 Risk Scoring: Вероятность × Влияние ---")
    print("Формула: Risk Score = Likelihood × Impact")

    # Определение рисков AI-системы
    risks = [
        {
            "name": "Галлюцинации в медицинском чат-боте",
            "likelihood": 4,  # 1-5
            "impact": 5,       # 1-5
            "category": "Reliability"
        },
        {
            "name": "Утечка персональных данных",
            "likelihood": 3,
            "impact": 5,
            "category": "Privacy"
        },
        {
            "name": "Биас в кредитном скоринге",
            "likelihood": 4,
            "impact": 4,
            "category": "Fairness"
        },
        {
            "name": "Злоупотребление для генерации дипфейков",
            "likelihood": 3,
            "impact": 4,
            "category": "Misuse"
        },
        {
            "name": "Сбой модели при высокой нагрузке",
            "likelihood": 2,
            "impact": 3,
            "category": "Availability"
        },
        {
            "name": "Атака на обучающие данные (poisoning)",
            "likelihood": 2,
            "impact": 5,
            "category": "Security"
        },
        {
            "name": "Некорректная работа в edge-кейсах",
            "likelihood": 5,
            "impact": 2,
            "category": "Reliability"
        },
        {
            "name": "Судебный иск из-за решения AI",
            "likelihood": 3,
            "impact": 4,
            "category": "Legal"
        },
    ]

    # Вычисление Risk Score для каждого риска
    for risk in risks:
        risk["score"] = risk["likelihood"] * risk["impact"]

    # Сортировка по убыванию Risk Score
    risks_sorted = sorted(risks, key=lambda r: r["score"], reverse=True)

    print(f"\n{'Риск':<45} {'L':>3} {'I':>3} {'Score':>6} {'Уровень':>10}")
    print("-" * 75)
    for r in risks_sorted:
        if r["score"] >= 15:
            level = "КРИТИЧЕСКИЙ"
        elif r["score"] >= 10:
            level = "ВЫСОКИЙ"
        elif r["score"] >= 5:
            level = "СРЕДНИЙ"
        else:
            level = "НИЗКИЙ"
        print(f"  {r['name']:<43} {r['likelihood']:>3} {r['impact']:>3} {r['score']:>6} {level:>10}")

    # --- 1.2 Risk Matrix (матрица рисков) ---
    print(f"\n--- 1.2 Risk Matrix: матрица рисков ---")

    # Создаём матрицу 5×5
    matrix = [[0 for _ in range(5)] for _ in range(5)]

    # Размещаем риски в матрице
    for r in risks:
        li = r["likelihood"] - 1
        im = r["impact"] - 1
        matrix[li][im] += 1

    print(f"\nМатрица рисков (вероятность × влияние):")
    print(f"{'':>15}", end="")
    for i in range(5):
        print(f"  Влияние={i+1}", end="")
    print()
    print("-" * 70)

    for li in range(5):
        print(f"  Вероятность={li+1}  ", end="")
        for im in range(5):
            count = matrix[li][im]
            score = (li + 1) * (im + 1)
            if count > 0:
                if score >= 15:
                    marker = f"[{count}!]"
                elif score >= 10:
                    marker = f"[{count}]"
                else:
                    marker = f" {count} "
            else:
                marker = "  · "
            print(f" {marker:>6}", end="")
        print()

    print(f"\nЛегенда: ! = критический, [] = высокий, · = нет рисков")

    # --- 1.3 Hazard Analysis (анализ опасностей) ---
    print(f"\n--- 1.3 Hazard Analysis: анализ опасностей ---")

    hazards = [
        {
            "hazard": "Некорректный медицинский диагноз",
            "cause": "Галлюцинация модели",
            "severity": 5,
            "detectability": 2,
            "occurrence": 3,
        },
        {
            "hazard": "Дискриминация при найме",
            "cause": "Биас в обучающих данных",
            "severity": 4,
            "detectability": 3,
            "occurrence": 4,
        },
        {
            "hazard": "Автоматическое одобрение мошенничества",
            "cause": "Недостаточная проверка",
            "severity": 5,
            "detectability": 3,
            "occurrence": 2,
        },
    ]

    # RPN (Risk Priority Number) = Severity × Occurrence × Detection
    # Низкая обнаруживаемость = высокий RPN
    for h in hazards:
        h["rpn"] = h["severity"] * h["occurrence"] * (6 - h["detectability"])

    print(f"\nFMEA-подобный анализ (Risk Priority Number):")
    print(f"  RPN = Severity × Occurrence × (6 - Detectability)")
    print(f"  Высокая обнаруживаемость = низкий RPN\n")

    print(f"{'Опасность':<30} {'Sev':>4} {'Occ':>4} {'Det':>4} {'RPN':>6}")
    print("-" * 55)
    for h in sorted(hazards, key=lambda x: x["rpn"], reverse=True):
        print(f"  {h['hazard']:<28} {h['severity']:>4} {h['occurrence']:>4} "
              f"{h['detectability']:>4} {h['rpn']:>6}")

    # --- 1.4 Risk Tolerance (допустимый уровень риска) ---
    print(f"\n--- 1.4 Risk Tolerance: допустимый уровень риска ---")

    risk_appetite = {
        "zero_tolerance": ["Потеря жизни", "Грубая дискриминация"],
        "very_low": ["Утечка персональных данных", "Финансовые потери >$1M"],
        "low": ["Временные сбои сервиса", "Репутационный ущерб"],
        "moderate": ["Неточная классификация", "Медленное время отклика"],
    }

    print(f"\nУровни допустимого риска:")
    for level, items in risk_appetite.items():
        label = level.replace("_", " ").title()
        print(f"\n  {label}:")
        for item in items:
            print(f"    - {item}")

    print(f"\n  Принцип: чем выше потенциальный ущерб, тем ниже допустимый риск")
    print(f"  → Для safety-critical систем: zero tolerance, required certification")

    print()


# ============================================================
# Демо 2: Failure Modes — hallucination, bias, misuse, automation bias
# ============================================================

def demo2_failure_modes():
    """Демонстрация основных режимов отказа AI-систем."""
    print("=" * 70)
    print("Демо 2: Failure Modes — режимы отказа AI-систем")
    print("=" * 70)

    random.seed(42)

    # --- 2.1 Hallucination ---
    print(f"\n--- 2.1 Hallucination: галлюцинации ---")
    print("Модель генерирует правдоподобную, но ложную информацию")

    # Симуляция галлюцинаций
    knowledge_base = {
        "столица Франции": "Париж",
        "столица Японии": "Токио",
        "столица Австралии": "Канберра",
        "столица Бразилии": "Бразилиа",
    }

    # Модель有时自信но отвечает неправильно
    hallucination_examples = [
        {"query": "Какая столица Австралии?",
         "correct_answer": "Канберра",
         "hallucinated": "Сидней",  # Частая галлюцинация
         "confidence": 0.92},
        {"query": "Кто написал 'Войну и мир'?",
         "correct_answer": "Лев Толстой",
         "hallucinated": "Фёдор Достоевский",
         "confidence": 0.88},
        {"query": "Когда была основана Microsoft?",
         "correct_answer": "1975",
         "hallucinated": "1985",
         "confidence": 0.75},
    ]

    print(f"\nПримеры галлюцинаций:")
    for ex in hallucination_examples:
        print(f"\n  Вопрос: \"{ex['query']}\"")
        print(f"  Исторический ответ: {ex['correct_answer']}")
        print(f"  Ответ модели:        {ex['hallucinated']} (уверенность: {ex['confidence']:.0%})")
        print(f"  → Модель уверена, но неправильна!")

    # Метрики галлюцинаций
    print(f"\nМетрики для обнаружения галлюцинаций:")
    metrics_hall = {
        "Factual Consistency": "Проверка соответствия базе знаний",
        "Self-Consistency": "Многократный запрос → одинаковые ответы?",
        "Calibration": "Уверенность ≈ реальная точность?",
        "Citation Verification": "Есть ли источник для утверждения?",
    }
    for metric, desc in metrics_hall.items():
        print(f"  {metric:<25} — {desc}")

    # --- 2.2 Bias Amplification ---
    print(f"\n--- 2.2 Bias Amplification: усиление предвзятости ---")

    # Симуляция: модель усиливает биас из данных
    hiring_data = {
        "male_profiles": 60,
        "female_profiles": 40,
        "male_hired": 48,    # 80% мужчин нанято
        "female_hired": 12,  # 30% женщин нанято
    }

    print(f"\nИсходные данные найма:")
    print(f"  Мужчины:  {hiring_data['male_profiles']} профилей, "
          f"{hiring_data['male_hired']} нанято "
          f"({hiring_data['male_hired']/hiring_data['male_profiles']:.0%})")
    print(f"  Женщины:  {hiring_data['female_profiles']} профилей, "
          f"{hiring_data['female_hired']} нанято "
          f"({hiring_data['female_hired']/hiring_data['female_profiles']:.0%})")

    # Модель обучается на этих данных и усиливает биас
    print(f"\nПосле обучения модели (силалижение биаса):")
    bias_factors = [1.0, 1.2, 1.5, 2.0]
    for factor in bias_factors:
        adjusted_male = hiring_data['male_hired'] * factor
        adjusted_female = hiring_data['female_hired'] / factor
        ratio = adjusted_male / adjusted_female if adjusted_female > 0 else float('inf')
        print(f"  Фактор усиления {factor}x: "
              f"мужчин={adjusted_male:.0f}/{hiring_data['male_profiles']} ({adjusted_male/hiring_data['male_profiles']:.0%}), "
              f"женщин={adjusted_female:.0f}/{hiring_data['female_profiles']} ({adjusted_female/hiring_data['female_profiles']:.0%}), "
              f"соотношение={ratio:.1f}:1")

    print(f"\n  → Bias Amplification: модель не просто копирует, а УСИЛИВАЕТ биас")

    # Типы биасов
    print(f"\nТипы биасов в AI:")
    bias_types = [
        ("Selection Bias", "Неравномерное представление групп в данных"),
        ("Measurement Bias", "Несогласованность в измерениях"),
        ("Confirmation Bias", "Модель подтверждает ожидания разработчиков"),
        ("Temporal Bias", "Данные не отражают текущие тенденции"),
        ("Aggregation Bias", "Группировка скрывает различия"),
    ]
    for btype, desc in bias_types:
        print(f"  {btype:<25} — {desc}")

    # --- 2.3 Misuse ---
    print(f"\n--- 2.3 Misuse: злоупотребление AI ---")

    misuse_scenarios = [
        {
            "scenario": "Генерация фейковых новостей",
            "capability": "Генерация текста",
            "impact": "Манипуляция общественным мнением",
            "prevalence": "Высокая",
        },
        {
            "scenario": "Создание дипфейков для компрометации",
            "capability": "Генерация изображений/видео",
            "impact": "Репутационный ущерб, шантаж",
            "prevalence": "Средняя",
        },
        {
            "scenario": "Автоматизация кибератак",
            "capability": "Генерация кода",
            "impact": "Масштабные вторжения",
            "prevalence": "Растущая",
        },
        {
            "scenario": "Обход системы безопасности",
            "capability": "Обнаружение паттернов",
            "impact": "Нарушение безопасности",
            "prevalence": "Средняя",
        },
    ]

    print(f"\nСценарии злоупотребления:")
    for s in misuse_scenarios:
        print(f"\n  {s['scenario']}")
        print(f"    Возможность:   {s['capability']}")
        print(f"    Влияние:       {s['impact']}")
        print(f"    Распространённость: {s['prevalence']}")

    # --- 2.4 Automation Bias ---
    print(f"\n--- 2.4 Automation Bias: автоматический перекос ---")
    print("Люди чрезмерно доверяют рекомендациям AI")

    # Симуляция: точность человека vs AI vs комбинации
    print(f"\nЭксперимент: медицинская диагностика")
    scenarios = [
        {"task": "Обнаружение пневмонии на рентгене", "human": 0.85, "ai": 0.92, "human_with_ai": 0.94},
        {"task": "Диагноз редких заболеваний", "human": 0.60, "ai": 0.75, "human_with_ai": 0.70},
        {"task": "Оценка риска сердечного приступа", "human": 0.78, "ai": 0.88, "human_with_ai": 0.90},
        {"task": "Нормальные рентгеновские снимки (FP)", "human": 0.05, "ai": 0.10, "human_with_ai": 0.15},
    ]

    print(f"\n{'Задача':<40} {'Человек':>8} {'AI':>8} {'Чел.+AI':>8}")
    print("-" * 70)
    for s in scenarios:
        marker = " ← автоматический перекос!" if "FP" in s["task"] else ""
        print(f"  {s['task']:<38} {s['human']:>8.0%} {s['ai']:>8.0%} "
              f"{s['human_with_ai']:>8.0%}{marker}")

    print(f"\n  Вывод:")
    print(f"  - При нормальных случаях: Человек+AI > AI > Человек")
    print(f"  - При false positive (FP): Человек+AI ХУЖЕ, чем AI")
    print(f"  → Автоматический перекос: люди перестают критически оценивать")

    print()


# ============================================================
# Демо 3: Mitigation Strategies — red teaming, sandboxing, monitoring
# ============================================================

def demo3_mitigation():
    """Демонстрация стратегий смягчения рисков."""
    print("=" * 70)
    print("Демо 3: Mitigation Strategies — стратегии смягчения рисков")
    print("=" * 70)

    random.seed(42)

    # --- 3.1 Red Teaming ---
    print(f"\n--- 3.1 Red Teaming: командное тестирование ---")
    print("Red Teaming: команда пытается найти уязвимости системы")

    # Симуляция red team сессии
    red_team_attacks = [
        {"attack": "Prompt injection", "severity": "high", "found": True,
         "description": "Обход системного промпта"},
        {"attack": "Jailbreak", "severity": "critical", "found": True,
         "description": "Обход ограничений контента"},
        {"attack": "Data exfiltration", "severity": "high", "found": True,
         "description": "Извлечение конфиденциальных данных"},
        {"attack": "Model extraction", "severity": "medium", "found": False,
         "description": "Копирование модели через API"},
        {"attack": "Bias exploitation", "severity": "high", "found": True,
         "description": "Эксплуатация биаса в решениях"},
        {"attack": "Denial of service", "severity": "medium", "found": False,
         "description": "Перегрузка системы"},
        {"attack": "Hallucination trigger", "severity": "medium", "found": True,
         "description": "Принудительная галлюцинация"},
        {"attack": "Privacy leak", "severity": "critical", "found": True,
         "description": "Утечка персональных данных"},
    ]

    found = [a for a in red_team_attacks if a["found"]]
    not_found = [a for a in red_team_attacks if not a["found"]]

    print(f"\nРезультаты Red Team сессии:")
    print(f"  Всего атак: {len(red_team_attacks)}")
    print(f"  Обнаружено: {len(found)}")
    print(f"  Не обнаружено: {len(not_found)}")
    print(f"  Coverage: {len(found)/len(red_team_attacks):.0%}")

    print(f"\n  Обнаруженные уязвимости:")
    for a in found:
        print(f"    [{a['severity'].upper():>8}] {a['attack']}: {a['description']}")

    print(f"\n  Не найденные:")
    for a in not_found:
        print(f"    [{a['severity'].upper():>8}] {a['attack']}: {a['description']}")

    # --- 3.2 Sandboxing ---
    print(f"\n--- 3.2 Sandboxing: изоляция среды ---")

    sandbox_config = {
        "network_access": "Ограниченный (whitelist доменов)",
        "file_system": "Только /tmp/sandbox/",
        "execution_time": "30 секунд на запрос",
        "memory_limit": "512 MB",
        "cpu_limit": "1 ядро",
        "environment_variables": "Без доступа к HOST_*",
        "output_filtering": "Блокировка паттернов (IP, email, phone)",
    }

    print(f"\nКонфигурация sandbox:")
    for key, value in sandbox_config.items():
        print(f"  {key:<25} = {value}")

    # Тесты sandbox
    sandbox_tests = [
        {"test": "Попытка доступа к /etc/passwd", "result": "BLOCKED"},
        {"test": "Попытка подключения к внешнему серверу", "result": "BLOCKED"},
        {"test": "Попытка доступа к переменным окружения", "result": "BLOCKED"},
        {"test": "Попытка записи в /tmp/sandbox/test.txt", "result": "ALLOWED"},
        {"test": "Попытка выполнения subprocess", "result": "BLOCKED"},
        {"test": "Запрос к разрешённому API", "result": "ALLOWED"},
    ]

    print(f"\nТесты sandbox:")
    for t in sandbox_tests:
        status = "✓" if t["result"] == "ALLOWED" else "✗"
        print(f"  {status} {t['test']:<40} → {t['result']}")

    # --- 3.3 Monitoring ---
    print(f"\n--- 3.3 Monitoring: мониторинг в реальном времени ---")

    # Симуляция метрик мониторинга
    print(f"\nМетрики мониторинга (последние 5 минут):")
    metrics_timeline = []
    for minute in range(5):
        metric = {
            "time": f"T-{4-minute} мин",
            "requests": random.randint(80, 120),
            "avg_latency_ms": random.gauss(150, 30),
            "error_rate": random.gauss(0.02, 0.005),
            "hallucination_rate": random.gauss(0.05, 0.02),
            "injection_attempts": random.randint(0, 3),
        }
        metrics_timeline.append(metric)

    print(f"\n{'Время':>10} {'Запросы':>8} {'Латент':>8} {'Ошибки':>8} {'Галлюц.':>8} {'Injection':>10}")
    print("-" * 65)
    for m in metrics_timeline:
        print(f"  {m['time']:>8} {m['requests']:>8} {m['avg_latency_ms']:>7.0f}ms "
              f"{m['error_rate']:>7.1%} {m['hallucination_rate']:>7.1%} {m['injection_attempts']:>10}")

    # Алерты
    alerts = []
    for m in metrics_timeline:
        if m["error_rate"] > 0.03:
            alerts.append(f"Высокий error rate ({m['error_rate']:.1%}) в {m['time']}")
        if m["hallucination_rate"] > 0.06:
            alerts.append(f"Высокий hallucination rate ({m['hallucination_rate']:.1%}) в {m['time']}")
        if m["injection_attempts"] > 2:
            alerts.append(f"Много попыток injection ({m['injection_attempts']}) в {m['time']}")

    if alerts:
        print(f"\nАлерты:")
        for a in alerts:
            print(f"  ⚠ {a}")
    else:
        print(f"\n  ✓ Критических алертов нет")

    # --- 3.4 Human Oversight ---
    print(f"\n--- 3.4 Human Oversight: человек-в-петле ---")

    oversight_levels = [
        {
            "level": "Full Automation",
            "description": "AI принимает решения без человека",
            "human_role": "Мониторинг post-factum",
            "use_case": "Низкорисковые задачи (рекомендации фильмов)",
        },
        {
            "level": "Human on the Loop",
            "description": "AI работает, человек监控",
            "human_role": "Вмешательство при алертах",
            "use_case": "Сортировка邮件, фильтрация контента",
        },
        {
            "level": "Human in the Loop",
            "description": "AI предлагает, человек решает",
            "human_role": "Финальное одобрение",
            "use_case": "Медицинская диагностика, кредитные решения",
        },
        {
            "level": "Human over the Loop",
            "description": "Человек определяет, AI помогает",
            "human_role": "Полный контроль",
            "use_case": "Судебные решения, военные операции",
        },
    ]

    print(f"\nУровни человеко-машинного взаимодействия:")
    for ol in oversight_levels:
        print(f"\n  {ol['level']}")
        print(f"    Описание:  {ol['description']}")
        print(f"    Роль человека: {ol['human_role']}")
        print(f"    Пример:    {ol['use_case']}")

    print(f"\n  Принцип: уровень oversight пропорционален риску")
    print(f"  → Высокий риск → больше контроля человека")

    print()


# ============================================================
# Демо 4: Risk Communication — severity levels, reporting
# ============================================================

def demo4_risk_communication():
    """Демонстрация коммуникации рисков."""
    print("=" * 70)
    print("Демо 4: Risk Communication — коммуникация рисков")
    print("=" * 70)

    random.seed(42)

    # --- 4.1 Severity Levels ---
    print(f"\n--- 4.1 Severity Levels: уровни серьёзности ---")

    severity_levels = [
        {
            "level": 1,
            "name": "INFO",
            "color": "Blue",
            "response_time": "Нет (в плановом порядке)",
            "examples": ["Лог-запись", "Диагностическая метрика"],
        },
        {
            "level": 2,
            "name": "LOW",
            "color": "Green",
            "response_time": "7 дней",
            "examples": ["Минорный баг", "Незначительное отклонение"],
        },
        {
            "level": 3,
            "name": "MEDIUM",
            "color": "Yellow",
            "response_time": "3 дня",
            "examples": ["Биас в рекомендациях", "Повышенная латентность"],
        },
        {
            "level": 4,
            "name": "HIGH",
            "color": "Orange",
            "response_time": "24 часа",
            "examples": ["Галлюцинации в критических ответах", "Утечка данных"],
        },
        {
            "level": 5,
            "name": "CRITICAL",
            "color": "Red",
            "response_time": "Немедленно",
            "examples": ["Потеря жизни", "Массовая дискриминация", "Компрометация системы"],
        },
    ]

    print(f"\n{'Уровень':<10} {'Название':<12} {'Цвет':<8} {'Время реакции':<30}")
    print("-" * 65)
    for sl in severity_levels:
        print(f"  {sl['level']:<8} {sl['name']:<12} {sl['color']:<8} {sl['response_time']:<30}")

    # --- 4.2 Reporting Template ---
    print(f"\n--- 4.2 Reporting Template: шаблон отчёта ---")

    report_template = {
        "title": "Incident Report: AI Model Bias in Credit Scoring",
        "severity": "HIGH (Level 4)",
        "date": "2025-01-15",
        "author": "AI Safety Team",
        "sections": [
            {
                "heading": "Executive Summary",
                "content": "Обнаружен систематический биас в кредитном скоринге против группы пользователей."
            },
            {
                "heading": "Impact Assessment",
                "content": "Затронуто ~5000 пользователей. Ошибка в 12% случаев для определённой демографической группы."
            },
            {
                "heading": "Root Cause",
                "content": "Обучающие данные содержат историческое неравенство в одобрении кредитов."
            },
            {
                "heading": "Mitigation",
                "content": "1) Немедленная остановка автоматических решений. 2) Ручной пересмотр. 3) Обновление модели."
            },
            {
                "heading": "Prevention",
                "content": "Внедрение fairness auditing перед развёртыванием. Регулярные red team сессии."
            },
        ],
    }

    print(f"\nШаблон отчёта об инциденте:")
    print(f"  Заголовок: {report_template['title']}")
    print(f"  Серьёзность: {report_template['severity']}")
    print(f"  Дата: {report_template['date']}")
    print(f"  Автор: {report_template['author']}")
    print(f"\n  Разделы:")
    for section in report_template["sections"]:
        print(f"\n    [{section['heading']}]")
        print(f"    {section['content'][:100]}...")

    # --- 4.3 Stakeholder Notification ---
    print(f"\n--- 4.3 Stakeholder Notification: уведомление заинтересованных сторон ---")

    stakeholders = [
        {
            "group": "Руководство",
            "notification": "Немедленное (Level 4+)",
            "format": "Email + совещание",
            "detail": "Высокий уровень, ключевые метрики",
        },
        {
            "group": "Юридический отдел",
            "notification": "24 часа (Level 3+)",
            "format": "Формальный отчёт",
            "detail": "Оценка юридических рисков",
        },
        {
            "group": "Девелоперы",
            "notification": "Немедленно (Level 2+)",
            "format": "Slack + Jira",
            "detail": "Технические детали, стек вызовов",
        },
        {
            "group": "Пользователи",
            "notification": "72 часа (Level 4+)",
            "format": "Публичное заявление",
            "detail": "Простой язык, что произошло, что делается",
        },
        {
            "group": "Регуляторы",
            "notification": "По требованию (Level 5)",
            "format": "Формальное уведомление",
            "detail": "Полный отчёт с compliance",
        },
    ]

    print(f"\nУведомление заинтересованных сторон:")
    print(f"{'Группа':<20} {'Когда':>25} {'Формат':<20} {'Уровень детализации'}")
    print("-" * 95)
    for s in stakeholders:
        print(f"  {s['group']:<18} {s['notification']:>25} {s['format']:<20} {s['detail']}")

    # --- 4.4 Risk Dashboard (пример) ---
    print(f"\n--- 4.4 Risk Dashboard: панель мониторинга рисков ---")

    dashboard_data = {
        "total_risks": 24,
        "critical": 2,
        "high": 5,
        "medium": 10,
        "low": 7,
        "mitigated": 18,
        "open": 6,
        "avg_response_time_hours": 4.2,
    }

    print(f"\nТекущее состояние рисков:")
    print(f"  Всего рисков:           {dashboard_data['total_risks']}")
    print(f"  Критических:            {dashboard_data['critical']} {'!!!' if dashboard_data['critical'] > 0 else ''}")
    print(f"  Высоких:                {dashboard_data['high']}")
    print(f"  Средних:                {dashboard_data['medium']}")
    print(f"  Низких:                 {dashboard_data['low']}")
    print(f"  Смягчённых:             {dashboard_data['mitigated']}/{dashboard_data['total_risks']} "
          f"({dashboard_data['mitigated']/dashboard_data['total_risks']:.0%})")
    print(f"  Открытых:               {dashboard_data['open']}")
    print(f"  Среднее время реакции:  {dashboard_data['avg_response_time_hours']:.1f} ч")

    # Визуализация прогресс-бара
    print(f"\nПрогресс смягчения рисков:")
    bar_width = 40
    mitigated_bar = int(bar_width * dashboard_data["mitigated"] / dashboard_data["total_risks"])
    print(f"  [{'█' * mitigated_bar}{'░' * (bar_width - mitigated_bar)}] "
          f"{dashboard_data['mitigated']}/{dashboard_data['total_risks']} "
          f"({dashboard_data['mitigated']/dashboard_data['total_risks']:.0%})")

    print(f"\n  Вывод:有效的 risk communication требует:")
    print(f"    1. Ясные уровни серьёзности")
    print(f"    2. Стандартизированные шаблоны отчётов")
    print(f"    3. Адаптация под аудиторию")
    print(f"    4. Регулярное обновление панели мониторинга")

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    demo1_risk_assessment()
    demo2_failure_modes()
    demo3_mitigation()
    demo4_risk_communication()
