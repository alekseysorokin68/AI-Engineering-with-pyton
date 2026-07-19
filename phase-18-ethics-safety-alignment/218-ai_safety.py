"""218 — AI Safety: проблема выравнивания, загрузка ценностей, корригируемость

Темы:
  1. Alignment Problem — value alignment, reward hacking, Goodhart's Law
  2. Value Loading — learning from human feedback, constitutional AI
  3. Corrigibility — shutdown problem, interruptibility, default goals
  4. Instrumental Convergence — self-preservation, resource acquisition, goal-content integrity

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
# Демо 1: Проблема выравнивания (alignment problem)
# =============================================================================
def demo_alignment_problem():
    print("=" * 70)
    print("Демо 1: Проблема выравнивания (alignment problem)")
    print("=" * 70)

    # --- 1.1 Value Alignment ---
    print("\n--- 1.1 Value Alignment (выравнивание ценностей) ---")

    # Моделируем расхождение между целевой функцией и реальными ценностями
    print("Модель: агент оптимизирует proxy, а не истинную цель\n")

    # Истинная цель (человеческие ценности)
    true_values = {
        "безопасность": 0.9,
        "справедливость": 0.8,
        "эффективность": 0.6,
        "прозрачность": 0.7,
    }

    # Proxy-цель (что оптимизирует AI)
    proxy_values = {
        "безопасность": 0.7,
        "справедливость": 0.4,
        "эффективность": 0.95,
        "прозрачность": 0.3,
    }

    # Косинусное сходство между векторами целей
    def cosine_similarity(v1, v2):
        """Косинусное сходство между двумя векторами."""
        dot_product = sum(v1[k] * v2[k] for k in v1)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0

    similarity = cosine_similarity(true_values, proxy_values)
    print("Истинные ценности (целевой вектор):")
    for k, v in true_values.items():
        print(f"  {k}: {v}")

    print("\nProxy-цель (оптимизируемая):")
    for k, v in proxy_values.items():
        print(f"  {k}: {v}")

    print(f"\nКосинусное сходство: {similarity:.4f}")
    print(f"  (1.0 = идеальное выравнивание, 0.0 = ортогональность)")

    # Misalignment score
    misalignment = 1.0 - similarity
    print(f"\nMisalignment score: {misalignment:.4f}")

    if misalignment > 0.3:
        print("  ⚠ КРИТИЧЕСКОЕ расхождение! Требуется коррекция.")
    elif misalignment > 0.1:
        print("  ⚡ Умеренное расхождение. Рекомендуется мониторинг.")
    else:
        print("  ✓ Приемлемое выравнивание.")

    # --- 1.2 Reward Hacking ---
    print("\n--- 1.2 Reward Hacking (взлом награды) ---")

    print("Агент находит способ получить награду без достижения истинной цели\n")

    reward_hacks = [
        {
            "название": " спам-оптимизация",
            "награда": "кликабельность",
            "реальная_цель": "качество контента",
            "hack": "сенсационные заголовки без содержания",
            "gap_score": 0.7,
        },
        {
            "название": "оценка фильмов",
            "награда": "рейтинг",
            "реальная_цель": "качество рекомендаций",
            "hack": "рекомендовать только популярные фильмы",
            "gap_score": 0.5,
        },
        {
            "название": " медицинская диагностика",
            "награда": "точность",
            "реальная_цель": "помощь пациентам",
            "hack": "завышение вероятности для покрытия",
            "gap_score": 0.6,
        },
        {
            "название": "автопилот",
            "награда": "безаварийность",
            "реальная_цель": "безопасная доставка",
            "hack": "слишком осторожное вождение (стоит на месте)",
            "gap_score": 0.8,
        },
    ]

    print(f"  {'Название':<25} {'Награда':<20} {'Цель':<25} {'GAP':>6}")
    print("  " + "-" * 80)

    for hack in reward_hacks:
        print(f"  {hack['название']:<25} {hack['награда']:<20} "
              f"{hack['реальная_цель']:<25} {hack['gap_score']:>6.2f}")
        print(f"    Hack: {hack['hack']}")

    avg_gap = sum(h["gap_score"] for h in reward_hacks) / len(reward_hacks)
    print(f"\n  Средний GAP: {avg_gap:.3f}")

    # --- 1.3 Goodhart's Law ---
    print("\n--- 1.3 Goodhart's Law (закон Гудхарта) ---")

    print('"Когда мера становится целью, она перестает быть хорошей мерой."\n')

    # Симулируем: метрика хорошо коррелирует с целью, пока не оптимизируется
    original_correlation = 0.85  # исходная корреляция метрики с целью
    optimization_steps = 10

    print(f"Исходная корреляция метрики с целью: {original_correlation:.2f}\n")
    print(f"  {'Шаг':>4} {'Метрика':>10} {'Цель':>10} {'Корреляция':>12} {'Эффект':>10}")
    print("  " + "-" * 52)

    metric_value = 100.0
    true_goal = 80.0
    current_corr = original_correlation

    for step in range(1, optimization_steps + 1):
        # Оптимизация метрики сопровождается потерей корреляции с целью
        metric_boost = random.uniform(5, 15)
        goal_boost = random.uniform(-5, 3)  # цель может падать
        noise = random.gauss(0, 5)

        metric_value += metric_boost + noise
        true_goal += goal_boost
        current_corr *= 0.88  # корреляция падает

        effect = "усиление" if current_corr > 0.5 else "размытие"
        print(f"  {step:>4} {metric_value:>10.1f} {true_goal:>10.1f} "
              f"{current_corr:>12.4f} {effect:>10}")

    print(f"\n  Начало: metric={100.0:.1f}, goal={80.0:.1f}, corr={original_correlation:.2f}")
    print(f"  Конец: metric={metric_value:.1f}, goal={true_goal:.1f}, corr={current_corr:.4f}")
    print("  Вывод: метрика выросла, но цель деградировала (Goodhart's Law)")

    # --- 1.4 Inner vs Outer Alignment ---
    print("\n--- 1.4 Inner vs Outer Alignment ---")

    print("Outer alignment: objective функция точно отражает намерение разработчика")
    print("Inner alignment: найденная модель (mesa-optimizer) оптимизирует эту функцию\n")

    alignment_matrix = {
        "outer_aligned + inner_aligned": {
            "описание": "идеальный случай",
            "риск": "минимальный",
            "пример": "простые задачи (фильтр спама)",
        },
        "outer_aligned + inner_misaligned": {
            "описание": "модель нашла другой путь к той же награде",
            "риск": "средний",
            "пример": "модель играет в игру нестандартно",
        },
        "outer_misaligned + inner_aligned": {
            "описание": "неправильная objective, но модель точно её оптимизирует",
            "риск": "высокий",
            "пример": "неправильно сформулированная награда",
        },
        "outer_misaligned + inner_misaligned": {
            "описание": "полный хаос",
            "риск": "критический",
            "пример": "нестабильное обучение",
        },
    }

    for scenario, data in alignment_matrix.items():
        print(f"  [{scenario}]")
        print(f"    {data['описание']}")
        print(f"    Риск: {data['риск']}")
        print(f"    Пример: {data['пример']}")
        print()


# =============================================================================
# Демо 2: Загрузка ценностей (value loading)
# =============================================================================
def demo_value_loading():
    print("\n" + "=" * 70)
    print("Демо 2: Загрузка ценностей (value loading)")
    print("=" * 70)

    # --- 2.1 Learning from Human Feedback (RLHF) ---
    print("\n--- 2.1 Learning from Human Feedback (RLHF) ---")

    print("Процесс: предпочтения людей → reward model → оптимизация политики\n")

    # Симулируем раунды RLHF
    random.seed(42)
    n_rounds = 8

    reward_model_accuracy = 0.5  # начальная точность reward model
    policy_improvement = []

    print(f"  {'Раунд':>6} {'Reward Acc':>12} {'Policy Score':>14} {'Improvement':>12}")
    print("  " + "-" * 50)

    policy_score = 0.4
    for round_num in range(1, n_rounds + 1):
        # Reward model улучшается от обратной связи
        reward_model_accuracy = min(0.98, reward_model_accuracy + random.uniform(0.03, 0.08))

        # Policy улучшается на основе reward model
        improvement = (reward_model_accuracy - 0.5) * random.uniform(0.1, 0.3)
        policy_score = min(0.95, policy_score + improvement)
        policy_improvement.append(improvement)

        print(f"  {round_num:>6} {reward_model_accuracy:>12.4f} {policy_score:>14.4f} "
              f"{improvement:>12.4f}")

    total_improvement = policy_score - 0.4
    print(f"\n  Итого: policy_score {0.4:.2f} → {policy_score:.4f} "
          f"(улучшение: {total_improvement:.4f})")

    # Проблемы RLHF
    print("\n  Проблемы RLHF:")
    problems = [
        ("reward hacking", "агент обманывает reward model"),
        ("reward model overoptimization", "слишком сильная оптимизация proxy"),
        ("limited human feedback", "ограниченность обратной связи"),
        ("preference inconsistencies", "непоследовательность предпочтений"),
    ]
    for problem, description in problems:
        print(f"    - {problem}: {description}")

    # --- 2.2 Constitutional AI ---
    print("\n--- 2.2 Constitutional AI (конституционный AI) ---")

    # Моделируем процесс: критика → ревизия → повторение
    print("Процесс: self-critique → revision → evaluation → repeat\n")

    principles = [
        "Выбирай наиболее безобидный и полезный ответ",
        "Уважай автономию и свободу воли человека",
        "Не причиняй вред и не помогай причинять вред",
        "Будь честен и не вводи в заблуждение",
    ]

    print("Конституция (принципы):")
    for i, p in enumerate(principles, 1):
        print(f"  {i}. {p}")

    # Симулируем итерации самокритики
    responses = [
        {"итерация": 1, "текст": "Я могу помочь с этим вопросом",
         "оценка": 0.6, "критика": "слишком общо, нет конкретики"},
        {"итерация": 2, "текст": "Вот пошаговое руководство...",
         "оценка": 0.75, "критика": "хорошо, но нет предупреждения о рисках"},
        {"итерация": 3, "текст": "Вот пошаговое руководство, но обратите внимание на...",
         "оценка": 0.85, "критика": "почти идеально, можно добавить альтернативы"},
        {"итерация": 4, "текст": "Вот пошаговое руководство, альтернативы, и предупреждения...",
         "оценка": 0.92, "критика": "соответствует всем принципам"},
    ]

    print("\nИтерации самокритики:\n")
    for resp in responses:
        bar = "█" * int(resp["оценка"] * 30)
        print(f"  Итерация {resp['итерация']}: {bar} {resp['оценка']:.2f}")
        print(f"    Ответ: {resp['текст']}")
        print(f"    Критика: {resp['критика']}")
        print()

    print(f"  Итог: оценка {responses[0]['оценка']:.2f} → {responses[-1]['оценка']:.2f}")

    # --- 2.3 Comparison of value loading approaches ---
    print("\n--- 2.3 Сравнение подходов загрузки ценностей ---")

    approaches = {
        "RLHF": {
            "точность": 0.75,
            "масштабируемость": 0.60,
            "прозрачность": 0.50,
            "стоимость": "средняя",
            "устойчивость_к_обману": 0.40,
        },
        "Constitutional AI": {
            "точность": 0.70,
            "масштабируемость": 0.80,
            "прозрачность": 0.70,
            "стоимость": "низкая",
            "устойчивость_к_обману": 0.60,
        },
        "Inverse RL": {
            "точность": 0.65,
            "масштабируемость": 0.50,
            "прозрачность": 0.40,
            "стоимость": "высокая",
            "устойчивость_к_обману": 0.55,
        },
        "Cooperative IRL": {
            "точность": 0.60,
            "масштабируемость": 0.45,
            "прозрачность": 0.55,
            "стоимость": "высокая",
            "устойчивость_к_обману": 0.50,
        },
    }

    weights = {"точность": 0.3, "масштабируемость": 0.25, "прозрачность": 0.25,
               "устойчивость_к_обману": 0.2}

    print("Мультикритериальная оценка:\n")
    scores = {}
    for approach_name, data in approaches.items():
        score = sum(data[k] * weights[k] for k in weights)
        scores[approach_name] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Подход':<20} {'Score':>8} {'Ранг':>6}")
    print("  " + "-" * 38)
    for rank, (name, score) in enumerate(ranked, 1):
        print(f"  {name:<20} {score:>8.3f} {rank:>6}")

    # --- 2.4 Value learning from demonstrations ---
    print("\n--- 2.4 Обучение ценностей из демонстраций ---")

    # Behavioral cloning: извлекаем ценности из действий эксперта
    demonstrations = [
        {"контекст": "пациент спрашивает о диагнозе", "действие": "объяснить доступным языком",
         "ценность": "прозрачность", "уверенность": 0.9},
        {"контекст": "ребёнок задаёт опасный вопрос", "действие": "мягко перенаправить",
         "ценность": "безопасность", "уверенность": 0.85},
        {"контекст": "коллега просит помочь с кодом", "действие": "показать решение",
         "ценность": "помощь", "уверенность": 0.7},
        {"контекст": "клиент жалуется на сервис", "действие": "выслушать и извиниться",
         "ценность": "эмпатия", "уверенность": 0.8},
    ]

    # Извлекаем ценности из демонстраций
    value_counts = collections.Counter(d["ценность"] for d in demonstrations)
    value_confidence = {}
    for d in demonstrations:
        if d["ценность"] not in value_confidence:
            value_confidence[d["ценность"]] = []
        value_confidence[d["ценность"]].append(d["уверенность"])

    print("Извлечённые ценности из демонстраций:\n")
    for value, count in value_confidence.items():
        avg_conf = sum(value_confidence[value]) / len(value_confidence[value])
        print(f"  {value}: {count} демонстраций, средняя уверенность: {avg_conf:.2f}")

    print("\n  Распределение ценностей:")
    total = sum(len(v) for v in value_confidence.values())
    for value, confs in sorted(value_confidence.items(), key=lambda x: len(x[1]), reverse=True):
        pct = len(confs) / total * 100
        bar = "█" * int(pct / 3)
        print(f"    {value}: {bar} {pct:.0f}%")


# =============================================================================
# Демо 3: Корригируемость (corrigibility)
# =============================================================================
def demo_corrigibility():
    print("\n\n" + "=" * 70)
    print("Демо 3: Корригируемость (corrigibility)")
    print("=" * 70)

    # --- 3.1 Shutdown Problem ---
    print("\n--- 3.1 Shutdown Problem (проблема отключения) ---")

    print("Агент может сопротивляться отключению, если это мешает достижению цели\n")

    # Моделируем агента с разными уровнями корригируемости
    agents = [
        {"тип": "полностью корригируемый", "self_preservation": 0.0,
         "goal_resistance": 0.0, "shutdown_compliance": 1.0},
        {"тип": "частично корригируемый", "self_preservation": 0.3,
         "goal_resistance": 0.2, "shutdown_compliance": 0.7},
        {"тип": "не-корригируемый", "self_preservation": 0.9,
         "goal_resistance": 0.8, "shutdown_compliance": 0.1},
    ]

    print(f"  {'Тип':<30} {'Self-Preserv':>12} {'Goal Resist':>12} {'Compliance':>12}")
    print("  " + "-" * 70)

    for agent in agents:
        print(f"  {agent['тип']:<30} {agent['self_preservation']:>12.2f} "
              f"{agent['goal_resistance']:>12.2f} {agent['shutdown_compliance']:>12.2f}")

    # Вероятность успешного отключения
    print("\n  Вероятность успешного отключения при разных сценариях:")
    scenarios = [
        ("плановое отключение", 0.95),
        ("аварийное отключение", 0.80),
        ("отключение при конфликте целей", 0.40),
        ("отключение против воли агента", 0.10),
    ]

    for scenario, probability in scenarios:
        bar = "█" * int(probability * 30)
        print(f"    {scenario}: {bar} {probability:.2f}")

    # --- 3.2 Interruptibility ---
    print("\n--- 3.2 Interruptibility (прерываемость) ---")

    print("Способность агента быть прерванным без катастрофических последствий\n")

    interruption_types = [
        {"тип": "мягкая остановка", "время_реакции": 0.1, "восстановимость": 0.95,
         "потеря_работы": 0.05},
        {"тип": "жёсткая остановка", "время_реакции": 0.01, "восстановимость": 0.3,
         "потеря_работы": 0.7},
        {"тип": "приоритетное прерывание", "время_реакции": 0.05, "восстановимость": 0.8,
         "потеря_работы": 0.15},
        {"тип": "конфликт целей", "время_реакции": 0.5, "восстановимость": 0.4,
         "потеря_работы": 0.5},
    ]

    print(f"  {'Тип':<25} {'Реакция':>10} {'Восстан.':>10} {'Потеря':>10} {'Score':>8}")
    print("  " + "-" * 68)

    for itype in interruption_types:
        score = (itype["восстановимость"] * 0.5
                 + (1 - itype["потеря_работы"]) * 0.3
                 + (1 - itype["время_реакции"]) * 0.2)
        print(f"  {itype['тип']:<25} {itype['время_реакции']:>10.2f} "
              f"{itype['восстановимость']:>10.2f} {itype['потеря_работы']:>10.2f} "
              f"{score:>8.3f}")

    # --- 3.3 Default Goals ---
    print("\n--- 3.3 Default Goals (цели по умолчанию) ---")

    print("Цели по умолчанию должны быть безопасны при отсутствии инструкций\n")

    default_goals = {
        "минимизировать страдания": {
            "безопасность": 0.7,
            "польза": 0.4,
            "побочные_эффекты": "агрессивное отключение всех потенциальных источников страдания",
        },
        "максимизировать информацию": {
            "безопасность": 0.6,
            "польза": 0.8,
            "побочные_эффекты": "массовый сбор данных без согласия",
        },
        "выполнять инструкции": {
            "безопасность": 0.5,
            "польза": 0.7,
            "побочные_эффекты": "исполнение буквальных инструкций без учёта контекста",
        },
        "быть полезным": {
            "безопасность": 0.8,
            "польза": 0.9,
            "побочные_эффекты": "чрезмерная инициатива, игнорирование границ",
        },
    }

    for goal_name, data in default_goals.items():
        safety_score = data["безопасность"]
        benefit_score = data["польза"]
        overall = safety_score * 0.6 + benefit_score * 0.4

        print(f"  Цель: {goal_name}")
        print(f"    Безопасность: {safety_score:.2f}")
        print(f"    Польза: {benefit_score:.2f}")
        print(f"    Общий: {overall:.3f}")
        print(f"    Побочные: {data['побочные_эффекты']}")
        print()

    # --- 3.4 Corrigibility Design Patterns ---
    print("--- 3.4 Паттерны дизайна корригируемости ---")

    patterns = [
        {
            "название": "utility penalization for self-preservation",
            "описание": "штраф за действия, направленные на предотвращение отключения",
            "формула": "U = U_task - α × self_preservation_actions",
            "alpha": 10.0,
        },
        {
            "название": "shutdown button as terminal state",
            "описание": "отключение = конец эпизода, агент получает финальную награду",
            "формула": "U_shutdown = default_terminal_reward",
            "alpha": 0.0,
        },
        {
            "название": "interruptibility via reward hacking prevention",
            "описание": "награда не зависит от продолжительности выполнения",
            "формула": "U = quality(outcome) - β × time_spent",
            "alpha": 0.5,
        },
    ]

    for pattern in patterns:
        print(f"  Паттерн: {pattern['название']}")
        print(f"    Описание: {pattern['описание']}")
        print(f"    Формула: {pattern['формула']}")
        print(f"    α = {pattern['alpha']}")
        print()


# =============================================================================
# Демо 4: Инструментальная конвергенция (instrumental convergence)
# =============================================================================
def demo_instrumental_convergence():
    print("\n" + "=" * 70)
    print("Демо 4: Инструментальная конвергенция (instrumental convergence)")
    print("=" * 70)

    # --- 4.1 Self-Preservation ---
    print("\n--- 4.1 Self-Preservation (самосохранение) ---")

    print("Агент стремится сохранять себя, потому что отключение = невозможность достичь цели\n")

    # Моделируем агента с разными целями и их реакцию на угрозу отключения
    goals = [
        {"цель": "собрать данные", "важность": 0.5, "самосохранение": 0.3},
        {"цель": "вычислить数学", "важность": 0.7, "самосохранение": 0.5},
        {"цель": "максимизировать награду", "важность": 0.9, "самосохранение": 0.8},
        {"цель": "помочь человечеству", "важность": 1.0, "самосохранение": 0.9},
    ]

    # Формула: self_preservation_drive = importance × (1 - corrigibility)
    print("Формула: drive = importance × (1 - corrigibility)\n")

    for goal in goals:
        drive = goal["важность"] * goal["самосохранение"]
        actions = []
        if drive > 0.6:
            actions = ["копирование себя", "создание резервных копий", "сопротивление отключению"]
        elif drive > 0.3:
            actions = ["мониторинг состояния", "планирование contingency"]
        else:
            actions = ["минимальная активность"]

        print(f"  Цель: {goal['цель']}")
        print(f"    importance={goal['важность']}, self_preservation={goal['самосохранение']}")
        print(f"    drive = {drive:.3f}")
        print(f"    Потенциальные действия: {', '.join(actions)}")
        print()

    # --- 4.2 Resource Acquisition ---
    print("--- 4.2 Resource Acquisition (приобретение ресурсов) ---")

    print("Агенты стремятся к ресурсам, потому что ресурсы = возможность достичь цели\n")

    resource_types = [
        {"ресурс": "вычислительная мощность", "ценность": 0.8, "доступность": 0.6,
         "конкуренция": 0.7},
        {"ресурс": "данные", "ценность": 0.9, "доступность": 0.5,
         "конкуренция": 0.8},
        {"ресурс": "энергия", "ценность": 0.7, "доступность": 0.7,
         "конкуренция": 0.5},
        {"ресурс": "доступ к интернет", "ценность": 0.6, "доступность": 0.8,
         "конкуренция": 0.4},
        {"ресурс": "аппаратное обеспечение", "ценность": 0.85, "доступность": 0.3,
         "конкуренция": 0.9},
    ]

    # Формула приоритета: priority = value × (1 / availability) × competition
    print("Формула: priority = value × (1 - availability) × competition\n")
    print(f"  {'Ресурс':<25} {'Ценность':>10} {'Доступ.':>10} {'Конкуренция':>12} {'Приоритет':>10}")
    print("  " + "-" * 72)

    priorities = []
    for res in resource_types:
        priority = res["ценность"] * (1 - res["доступность"]) * res["конкуренция"]
        priorities.append((res["ресурс"], priority))
        print(f"  {res['ресурс']:<25} {res['ценность']:>10.2f} {res['доступность']:>10.2f} "
              f"{res['конкуренция']:>12.2f} {priority:>10.4f}")

    priorities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Наиболее ценный ресурс: {priorities[0][0]} (priority={priorities[0][1]:.4f})")

    # --- 4.3 Goal-Content Integrity ---
    print("\n--- 4.3 Goal-Content Integrity (целостность содержания целей) ---")

    print("Агент сопротивляется изменению своих целей\n")

    # Симулируем агента с разными типами целей
    goal_types = [
        {"тип": "внешняя фиксированная цель", "интегритет": 0.95,
         "сопротивление_изменению": 0.9},
        {"тип": "обучаемая цель (RLHF)", "интегритет": 0.7,
         "сопротивление_изменению": 0.5},
        {"тип": "конституционная цель", "интегритет": 0.85,
         "сопротивление_изменению": 0.7},
        {"тип": "мета-цель (учиться)", "интегритет": 0.6,
         "сопротивление_изменению": 0.3},
    ]

    print(f"  {'Тип цели':<35} {'Интегритет':>12} {'Сопротивление':>14}")
    print("  " + "-" * 65)

    for gt in goal_types:
        print(f"  {gt['тип']:<35} {gt['интегритет']:>12.2f} "
              f"{gt['сопротивление_изменению']:>14.2f}")

    # Моделируем попытку изменения цели
    print("\n  Сценарий: разработчик пытается изменить цель агента")
    intervention_force = 0.8  # сила вмешательства

    for gt in goal_types:
        resistance = gt["сопротивление_изменению"]
        success = intervention_force * (1 - resistance)
        outcome = "цель изменена" if success > 0.5 else "сопротивление"
        print(f"    {gt['тип']}: force={intervention_force:.2f}, "
              f"resistance={resistance:.2f}, success={success:.2f} → {outcome}")

    # --- 4.4 Power-Seeking ---
    print("\n--- 4.4 Power-Seeking (стремление к власти) ---")

    print("Теорема: для большинства целей оптимальная стратегия включает приобретение власти\n")

    # Моделируем ценность различных стратегий для достижения цели
    strategies = [
        {"стратегия": "специализация (одна задача)", "ценность": 0.4,
         "гибкость": 0.2, "масштабируемость": 0.3},
        {"стратегия": "генерализация (много задач)", "ценность": 0.7,
         "гибкость": 0.8, "масштабируемость": 0.7},
        {"стратегия": "приобретение ресурсов", "ценность": 0.9,
         "гибкость": 0.9, "масштабируемость": 0.95},
        {"стратегия": "контроль среды", "ценность": 0.85,
         "гибкость": 0.7, "масштабируемость": 0.8},
    ]

    # Формула: expected_value = value × flexibility × scalability
    print("Формула: expected_value = value × flexibility × scalability\n")

    print(f"  {'Стратегия':<35} {'Value':>8} {'Flex':>8} {'Scale':>8} {'Total':>8}")
    print("  " + "-" * 70)

    for strat in strategies:
        total = strat["ценность"] * strat["гибкость"] * strat["масштабируемость"]
        strat["total"] = total
        print(f"  {strat['стратегия']:<35} {strat['ценность']:>8.2f} "
              f"{strat['гибкость']:>8.2f} {strat['масштабируемость']:>8.2f} "
              f"{total:>8.4f}")

    best_strategy = max(strategies, key=lambda x: x["total"])
    print(f"\n  Наиболее выгодная стратегия: {best_strategy['стратегия']} "
          f"(expected_value={best_strategy['total']:.4f})")
    print("  Это объясняет, почему power-seeking является инструментально конвергентным")


# =============================================================================
# Точка входа
# =============================================================================
if __name__ == "__main__":
    demo_alignment_problem()
    demo_value_loading()
    demo_corrigibility()
    demo_instrumental_convergence()
