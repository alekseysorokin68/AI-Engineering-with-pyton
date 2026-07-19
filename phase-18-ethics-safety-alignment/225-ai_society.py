"""225 — AI & Society: влияние на рабочие места, демократию, концентрацию власти

Темы:
  1. Labor Impact — смещение рабочих мест, сдвиг навыков, создание новых ролей
  2. Democratic Impact — дипфейки, дезинформация, манипуляция
  3. Power Concentration — концентрация вычислений, монополии данных, дефицит талантов
  4. Digital Divide — неравенство доступа, глобальные различия, инклюзия

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
# 1. Labor Impact — влияние ИИ на рынок труда
# ─────────────────────────────────────────────────────────────────────

def demo_labor_impact():
    print("=" * 70)
    print("DEMO 1 — Labor Impact: влияние ИИ на рабочие места")
    print("=" * 70)

    # --- 1.1 Модель смещения рабочих мест ---
    print("\n--- 1.1 Прогноз смещения рабочих мест по отраслям ---")
    # Вероятность автоматизации по отраслям (данные OECD)
    sectors = {
        "Производство": 0.65,
        "Транспорт и логистика": 0.58,
        "Розничная торговля": 0.53,
        "Финансы и бухгалтерия": 0.47,
        "Здравоохранение": 0.22,
        "Образование": 0.18,
        "Творческие профессии": 0.15,
        "IT и инженерия": 0.12,
    }
    # Симулируем 1000 работников в каждой отрасли
    random.seed(42)
    results = {}
    for sector, prob in sectors.items():
        displaced = sum(1 for _ in range(1000) if random.random() < prob)
        results[sector] = displaced
        pct = displaced / 10
        print(f"  {sector:<30s}: {displaced:>4d}/1000 ({pct:5.1f}%) подвержены автоматизации")

    # Формула: P(displacement) = f(task_routine, tech_readiness, cost_saving)
    print("\n  Формула риска: R = α·routine_tasks + β·tech_readiness + γ·cost_saving")
    print("  Где α=0.4, β=0.35, γ=0.25 — весовые коэффициенты")

    # --- 1.2 Сдвиг навыков (skill transition) ---
    print("\n--- 1.2 Модель сдвига навыков ---")
    # Навыки, которые теряют и приобретают ценность
    skills_deprecated = [
        ("Рутинные вычисления", -0.45),
        ("Ручная сортировка данных", -0.62),
        ("Базовый кодинг", -0.30),
        ("Администрирование БД", -0.38),
    ]
    skills_emerging = [
        ("Промпт-инжиниринг", 0.72),
        ("AI-етика и безопасность", 0.68),
        ("Data storytelling", 0.55),
        ("Пrompt evaluation", 0.61),
    ]
    print("  Устаревающие навыки:")
    for skill, delta in skills_deprecated:
        bar = "█" * int(abs(delta) * 30)
        print(f"    {skill:<30s}: {delta:+.2f}  ▼{bar}")
    print("  Набирающие ценность:")
    for skill, delta in skills_emerging:
        bar = "█" * int(delta * 30)
        print(f"    {skill:<30s}: {delta:+.2f}  ▲{bar}")

    # --- 1.3 Создание новых рабочих мест ---
    print("\n--- 1.3 Прогноз создания новых профессий ---")
    new_jobs = [
        ("Инженер промптов", "Проектирование и оптимизация промптов для LLM"),
        ("AI-тренер данных", "Подготовка и аннотация данных для обучения моделей"),
        ("Специалист по AI-безопасности", "Тестирование на вредоносность, red-teaming"),
        ("AI-продюсер", "Интеграция AI-инструментов в бизнес-процессы"),
        ("Этический аудитор AI", "Проверка моделей на.bias, fairness, compliance"),
    ]
    # Модель: каждая новая роль = сумма 3+ старых ролей
    print("  Формула: new_role = f(deprecated_skills, market_demand, tech_capability)")
    for job, desc in new_jobs:
        print(f"    • {job}: {desc}")

    # --- 1.4 Экономическая модель перехода ---
    print("\n--- 1.4 Экономика переходного периода ---")
    # Модель Фибоначчи для Adoption curve
    adoption = [1, 2]
    for i in range(10):
        adoption.append(adoption[-1] + adoption[-2])
    max_val = max(adoption)
    normalized = [v / max_val for v in adoption]
    print("  Кривая внедрения ИИ (нормализованная):")
    for i, v in enumerate(normalized):
        bar = "▓" * int(v * 40)
        print(f"    Год {i+1:>2d}: {v:.3f} {bar}")
    # Формула переходных издержек
    print("\n  Переходные издержки: TC = Training_cost + Productivity_loss + Integration_cost")
    print("  NPV выгоды: Σ (ΔProductivity_t - TC_t) / (1 + r)^t, r = 0.08")


# ─────────────────────────────────────────────────────────────────────
# 2. Democratic Impact — влияние на демократию
# ─────────────────────────────────────────────────────────────────────

def demo_democratic_impact():
    print("\n" + "=" * 70)
    print("DEMO 2 — Democratic Impact: дипфейки, дезинформация, манипуляция")
    print("=" * 70)

    # --- 2.1 Генерация дипфейк-сценариев ---
    print("\n--- 2.1 Типология угроз дипфейков ---")
    deepfake_types = {
        "Face Swap": {
            "опасность": 0.85,
            "обнаружение": 0.45,
            "описание": "Замена лица на видео/фото",
        },
        "Voice Clone": {
            "опасность": 0.80,
            "обнаружение": 0.35,
            "описание": "Клонирование голоса для звонков",
        },
        "Lip Sync": {
            "опасность": 0.70,
            "обнаружение": 0.50,
            "описание": "Подгонка губ под ложную речь",
        },
        "Full Body Puppet": {
            "опасность": 0.90,
            "обнаружение": 0.25,
            "описание": "Полная генерация тела и движений",
        },
    }
    for dtype, info in deepfake_types.items():
        danger_bar = "▓" * int(info["опасность"] * 20)
        detect_bar = "░" * int(info["обнаружение"] * 20)
        gap = info["опасность"] - info["обнаружение"]
        print(f"  {dtype}:")
        print(f"    Опасность:    {danger_bar} {info['опасность']:.2f}")
        print(f"    Обнаружение:  {detect_bar} {info['обнаружение']:.2f}")
        print(f"    Пробел:       {gap:+.2f} ({info['описание']})")

    # --- 2.2 Модель распространения дезинформации ---
    print("\n--- 2.2 Модель распространения дезинформации (SIR) ---")
    # S — восприимчивые, I — инфицированные (поверили), R — выздоровевшие (критики)
    S, I, R = 10000, 1, 0
    beta = 0.3   # скорость заражения (вирусность)
    gamma = 0.1  # скорость выздоровления (фактчекинг)
    print(f"  Начало: S={S}, I={I}, R={R}")
    print(f"  Параметры: β={beta} (вирусность), γ={gamma} (фактчекинг)")
    for day in range(1, 11):
        new_infected = beta * S * I / (S + I + R)
        new_recovered = gamma * I
        S -= new_infected
        I += new_infected - new_recovered
        R += new_recovered
        print(f"  День {day:>2d}: S={S:>7.0f} I={I:>7.0f} R={R:>7.0f} "
              f"(вирусность: {new_infected:.0f}, фактчек: {new_recovered:.0f})")

    # --- 2.3 Анализ манипулятивных паттернов ---
    print("\n--- 2.3 Детектирование манипулятивных паттернов в тексте ---")
    manipulation_patterns = {
        "Appeal to fear": r"(если не|иначе|последствия|катастроф|ужас)",
        "Bandwagon": r"(все знают|каждый|большинство|единственный способ)",
        "False dilemma": r"(или.*или|только два|нет альтернатив)",
        "Ad hominem": r"(тупой|некомпетентн|не понима|не умеет)",
        "Straw man": r"(на самом деле.*хотят|по сути.*означает)",
    }
    test_texts = [
        "Все знают, что если не принять меры сейчас, последствия будут катастрофическими",
        "Единственный способ решить проблему — это либо принять закон, либо ждать катастрофы",
        "Этот эксперт некомпетентен и не понимает сути проблемы",
        "На самом деле они хотят лишь увеличить контроль над населением",
        "Только дурак не понимает важности этой технологии для будущего",
    ]
    for text in test_texts:
        print(f"\n  Текст: \"{text}\"")
        found = []
        for ptype, pattern in manipulation_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(ptype)
        if found:
            print(f"  ⚠ Обнаружены паттерны: {', '.join(found)}")
        else:
            print(f"  ✓ Манипулятивных паттернов не обнаружено")

    # --- 2.4 Метрика доверия к информации ---
    print("\n--- 2.4 Индекс доверия к источнику (Trust Score) ---")
    # Формула: Trust = w1·source_age + w2·fact_check_rate + w3·(1 - bias_score)
    sources = [
        ("Рекомендовано в соцсети", 0.1, 0.2, 0.8),
        ("Государственное СМИ", 0.7, 0.6, 0.3),
        ("Научный журнал", 0.9, 0.95, 0.1),
        ("Блог анонимного автора", 0.05, 0.1, 0.9),
        ("Fact-checking.org", 0.8, 0.9, 0.15),
    ]
    w1, w2, w3 = 0.3, 0.4, 0.3  # веса
    print(f"  Формула: Trust = {w1}·SourceAge + {w2}·FactCheck + {w3}·(1 - Bias)")
    for name, age, fcheck, bias in sources:
        trust = w1 * age + w2 * fcheck + w3 * (1 - bias)
        bar = "█" * int(trust * 30)
        level = "ВЫСОКИЙ" if trust > 0.6 else "СРЕДНИЙ" if trust > 0.35 else "НИЗКИЙ"
        print(f"    {name:<35s}: {trust:.3f} {bar} [{level}]")


# ─────────────────────────────────────────────────────────────────────
# 3. Power Concentration — концентрация власти
# ─────────────────────────────────────────────────────────────────────

def demo_power_concentration():
    print("\n" + "=" * 70)
    print("DEMO 3 — Power Concentration: концентрация вычислений и данных")
    print("=" * 70)

    # --- 3.1 Концентрация вычислительных мощностей ---
    print("\n--- 3.1 Распределение GPU-мощностей среди компаний ---")
    companies = {
        "Google": {"gpu_count": 2500000, "share_pct": 35.0, "tpu_v5": 900000},
        "Microsoft": {"gpu_count": 1800000, "share_pct": 25.0, "tpu_v5": 0},
        "Meta": {"gpu_count": 1200000, "share_pct": 16.7, "tpu_v5": 0},
        "Amazon (AWS)": {"gpu_count": 800000, "share_pct": 11.1, "tpu_v5": 0},
        "OpenAI": {"gpu_count": 500000, "share_pct": 6.9, "tpu_v5": 100000},
        "Другие": {"gpu_count": 400000, "share_pct": 5.3, "tpu_v5": 50000},
    }
    total_gpus = sum(c["gpu_count"] for c in companies.values())
    print(f"  Всего GPU: {total_gpus:,}")
    for company, info in companies.items():
        bar = "█" * int(info["share_pct"] * 0.8)
        print(f"    {company:<20s}: {info['gpu_count']:>10,} GPU ({info['share_pct']:>5.1f}%) {bar}")
    # Индекс Херфиндаля-Хиршмана (концентрация рынка)
    hhi = sum((c["share_pct"]) ** 2 for c in companies.values())
    print(f"\n  Индекс Херфиндаля-Хиршмана: HHI = {hhi:.1f}")
    print(f"  (HHI < 1500: конкурентный | 1500-2500: умеренная | >2500: высокая концентрация)")

    # --- 3.2 Монополия данных ---
    print("\n--- 3.2 Объём данных для обучения крупнейших моделей ---")
    models = [
        ("GPT-2 (2019)", 40, 1.5e9),
        ("GPT-3 (2020)", 570, 175e9),
        ("PaLM (2022)", 3000, 540e9),
        ("GPT-4 (2023)", 13000, 1800e9),
        ("Llama 3 (2024)", 15000, 405e9),
        ("GPT-5 estimate", 50000, 1800e9),
    ]
    print("  {'Модель':<25s} {'Данные (TB)':>12s} {'Параметры':>12s} {'Efficiency':>12s}")
    for name, data_tb, params in models:
        # Эффективность = параметры / объём данных (млрд/TB)
        efficiency = params / 1e9 / data_tb if data_tb > 0 else 0
        bar = "▓" * min(int(data_tb / 500), 40)
        print(f"    {name:<25s} {data_tb:>10.0f} TB {params/1e9:>10.0f}B  {efficiency:>8.2f}B/TB {bar}")

    # --- 3.3 Дефицит талантов ---
    print("\n--- 3.3 Глобальный дефицит AI-специалистов ---")
    talent_data = {
        "США": {"demand": 450000, "supply": 320000, "gap_pct": 28.9},
        "Китай": {"demand": 380000, "supply": 290000, "gap_pct": 23.7},
        "ЕС": {"demand": 220000, "supply": 145000, "gap_pct": 34.1},
        "Индия": {"demand": 180000, "supply": 120000, "gap_pct": 33.3},
        "Великобритания": {"demand": 85000, "supply": 52000, "gap_pct": 38.8},
    }
    print("  {'Страна':<20s} {'Спрос':>8s} {'Предложение':>12s} {'Дефицит':>8s}")
    for country, data in talent_data.items():
        gap = data["demand"] - data["supply"]
        bar = "▓" * int(data["gap_pct"] / 2)
        print(f"    {country:<20s} {data['demand']:>7,} {data['supply']:>10,} "
              f"{data['gap_pct']:>6.1f}% {bar}")

    # --- 3.4 Модель эскалации вычислений ---
    print("\n--- 3.4 Эскалация требований к вычислениям (Закон Гровера) ---")
    # Compute required grows exponentially with model size
    compute_baseline = 1e18  # FLOPs для GPT-2
    years = list(range(2019, 2031))
    growth_rate = 3.4  # рост в 3.4 раза в год (исторический тренд)
    print(f"  Базовый уровень (GPT-2, 2019): {compute_baseline:.1e} FLOPs")
    print(f"  Темп роста: ×{growth_rate} в год")
    for year in years:
        flops = compute_baseline * (growth_rate ** (year - 2019))
        exp = math.log10(flops)
        bar = "█" * min(int(exp - 17), 25)
        print(f"    {year}: {flops:.1e} FLOPs (10^{exp:.1f}) {bar}")


# ─────────────────────────────────────────────────────────────────────
# 4. Digital Divide — цифровое неравенство
# ─────────────────────────────────────────────────────────────────────

def demo_digital_divide():
    print("\n" + "=" * 70)
    print("DEMO 4 — Digital Divide: неравенство доступа к ИИ")
    print("=" * 70)

    # --- 4.1 Индекс доступа к ИИ по регионам ---
    print("\n--- 4.1 Индекс доступа к ИИ (AI Access Index) ---")
    regions = {
        "Северная Америка": {"internet_pct": 93, "ai_tools": 95, "education": 88, "infra": 92},
        "Западная Европа": {"internet_pct": 91, "ai_tools": 88, "education": 85, "infra": 87},
        "Восточная Азия": {"internet_pct": 78, "ai_tools": 82, "education": 72, "infra": 80},
        "Латинская Америка": {"internet_pct": 68, "ai_tools": 45, "education": 52, "infra": 55},
        "Юго-Восточная Азия": {"internet_pct": 62, "ai_tools": 38, "education": 45, "infra": 48},
    }
    weights = {"internet_pct": 0.25, "ai_tools": 0.30, "education": 0.25, "infra": 0.20}
    scores = {}
    print("  Формула: AI_Access = 0.25·Internet + 0.30·AI_Tools + 0.25·Education + 0.20·Infra")
    for region, metrics in regions.items():
        score = sum(metrics[k] * weights[k] for k in weights)
        scores[region] = score
        bar = "█" * int(score / 3)
        print(f"    {region:<25s}: {score:.1f}/100 {bar}")

    # --- 4.2 Глобальное неравенство (коэффициент Джини) ---
    print("\n--- 4.2 Коэффициент Джини доступа к ИИ ---")
    # Данные: % населения с доступом к AI-инструментам
    access_by_country = [
        ("США", 72), ("Китай", 45), ("Германия", 68), ("Индия", 18),
        ("Бразилия", 32), ("Нигерия", 8), ("Япония", 62), ("Индонезия", 15),
        ("Великобритания", 70), ("Эфиопия", 3), ("Корея", 58), ("Мексика", 28),
    ]
    values = sorted([v for _, v in access_by_country])
    n = len(values)
    cumulative = [sum(values[:i+1]) for i in range(n)]
    total = cumulative[-1]
    gini = 1 - 2 * sum(cumulative) / (n * total) if total > 0 else 0
    print(f"  Коэффициент Джини: {gini:.3f} (0 = полное равенство, 1 = полное неравенство)")
    print("\n  Распределение доступа:")
    for country, val in sorted(access_by_country, key=lambda x: x[1], reverse=True):
        bar = "█" * (val // 3)
        print(f"    {country:<20s}: {val:>3d}% {bar}")

    # --- 4.3 Языковое неравенство ---
    print("\n--- 4.3 Языковое покрытие ИИ-моделей ---")
    languages = {
        "Английский": {"speakers_mln": 1500, "ai_quality": 0.98, "models": 95},
        "Китайский": {"speakers_mln": 1100, "ai_quality": 0.92, "models": 85},
        "Испанский": {"speakers_mln": 550, "ai_quality": 0.85, "models": 70},
        "Арабский": {"speakers_mln": 420, "ai_quality": 0.72, "models": 55},
        "Хинди": {"speakers_mln": 600, "ai_quality": 0.68, "models": 45},
        "Суахили": {"speakers_mln": 100, "ai_quality": 0.35, "models": 15},
        "Бенгальский": {"speakers_mln": 230, "ai_quality": 0.42, "models": 20},
    }
    print("  {'Язык':<15s} {'Носителей (млн)':>15s} {'Качество ИИ':>12s} {'Покрытие':>10s}")
    for lang, info in languages.items():
        quality_bar = "▓" * int(info["ai_quality"] * 20)
        print(f"    {lang:<15s} {info['speakers_mln']:>13,}M "
              f"{info['ai_quality']:>10.2f}   {quality_bar}")

    # --- 4.4 Модель включения (inclusion metrics) ---
    print("\n--- 4.4 Метрики инклюзии ИИ-системы ---")
    # Модель: Inclusion = f(accessibility, language_support, cost_barrier, digital_literacy)
    ai_systems = [
        ("ChatGPT Plus", 0.85, 0.92, 0.60, 0.75),
        ("Claude", 0.88, 0.90, 0.65, 0.70),
        ("Голосовые ассистенты", 0.70, 0.80, 0.85, 0.55),
        ("Open-source LLMs", 0.60, 0.50, 0.95, 0.30),
        ("Government AI portals", 0.65, 0.55, 0.90, 0.40),
    ]
    w_inc = {"accessibility": 0.25, "language": 0.30, "cost": 0.25, "literacy": 0.20}
    print("  Формула: Inclusion = 0.25·Accessibility + 0.30·Language + 0.25·(1-Cost) + 0.20·Literacy")
    for name, acc, lang, cost, lit in ai_systems:
        score = (w_inc["accessibility"] * acc +
                 w_inc["language"] * lang +
                 w_inc["cost"] * cost +
                 w_inc["literacy"] * lit)
        bar = "█" * int(score * 30)
        print(f"    {name:<25s}: {score:.3f} {bar}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_labor_impact()
    demo_democratic_impact()
    demo_power_concentration()
    demo_digital_divide()
