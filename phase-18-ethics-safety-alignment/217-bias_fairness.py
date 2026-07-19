"""217 — Bias & Fairness: типы смещений, метрики справедливости, смягчение

Темы:
  1. Types of Bias — historical, representation, measurement, aggregation, deployment
  2. Fairness Metrics — demographic parity, equalized odds, individual fairness
  3. Bias Detection — disparate impact, statistical parity difference, calibration
  4. Bias Mitigation — pre-processing, in-processing, post-processing techniques

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
# Демо 1: Типы смещений (bias types)
# =============================================================================
def demo_bias_types():
    print("=" * 70)
    print("Демо 1: Типы смещений (bias types)")
    print("=" * 70)

    # --- 1.1 Historical bias ---
    print("\n--- 1.1 Historical bias (историческое смещение) ---")

    # Моделируем исторические данные найма: на основе исторических паттернов
    # женщины реже назначались на руководящие позиции
    years = list(range(1990, 2024))
    total_applicants = 1000
    male_rate = 0.55  # историческое распределение кандидатов
    female_rate = 0.45
    base_hire_rate_male = 0.30
    base_hire_rate_female = 0.15  # историческое смещение

    # Симулируем с постепенным снижением смещения
    print("Формула: adjusted_female_hire_rate = base × (1 + diminishing_factor)\n")
    print(f"  {'Год':>6} {'Мужчины (наём)':>16} {'Женщины (наём)':>16} {'Разница':>10}")
    print("  " + "-" * 55)

    for year in years:
        # diminishing factor: историческое смещение постепенно уменьшается
        diminishing = max(0, 1.0 - (year - 1990) / 50.0)
        female_hire_rate = base_hire_rate_female * (1 + diminishing * 0.5)
        male_applicants = int(total_applicants * male_rate)
        female_applicants = int(total_applicants * female_rate)
        male_hired = int(male_applicants * base_hire_rate_male)
        female_hired = int(female_applicants * female_hire_rate)
        gap = base_hire_rate_male - female_hire_rate

        if year % 7 == 1990 % 7:
            print(f"  {year:>6} {male_hired:>5}/{male_applicants:<8} "
                  f"{female_hired:>5}/{female_applicants:<8} {gap:>10.3f}")

    print("\n  Вывод: историческое смещение сохраняется даже при смене политики")

    # --- 1.2 Representation bias ---
    print("\n--- 1.2 Representation bias (смещение представительства) ---")

    # Распределение данных по группам в датасете
    population_distribution = {"группа A": 0.60, "группа B": 0.25, "группа C": 0.10,
                                "группа D": 0.05}
    dataset_distribution = {"группа A": 0.70, "группа B": 0.20, "группа C": 0.08,
                             "группа D": 0.02}

    # Формула: representation_ratio = dataset_pct / population_pct
    print("Формула: representation_ratio = dataset_pct / population_pct")
    print("representation_ratio < 1 → группа недопредставленна\n")

    for group in population_distribution:
        pop_pct = population_distribution[group]
        ds_pct = dataset_distribution[group]
        ratio = ds_pct / pop_pct if pop_pct > 0 else 0
        status = "недопредставлена" if ratio < 0.8 else "перепредставлена" if ratio > 1.2 \
            else "адекватно"
        print(f"  {group}: pop={pop_pct:.2f}, dataset={ds_pct:.2f}, "
              f"ratio={ratio:.3f} → {status}")

    # --- 1.3 Measurement bias ---
    print("\n--- 1.3 Measurement bias (смещение измерения) ---")

    # Два теста с разной точностью для разных групп
    tests = {
        "тест A (стандартный)": {
            "группа X": {"accuracy": 0.92, "false_positive": 0.03, "false_negative": 0.05},
            "группа Y": {"accuracy": 0.88, "false_positive": 0.08, "false_negative": 0.04},
        },
        "тест B (калиброванный)": {
            "группа X": {"accuracy": 0.90, "false_positive": 0.05, "false_negative": 0.05},
            "группа Y": {"accuracy": 0.89, "false_positive": 0.06, "false_negative": 0.05},
        },
    }

    print("Различия в метриках по группам (смещение измерения):\n")
    for test_name, groups in tests.items():
        print(f"  {test_name}:")
        metrics = list(groups.keys())
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                g1, g2 = metrics[i], metrics[j]
                for metric in ["accuracy", "false_positive", "false_negative"]:
                    diff = abs(groups[g1][metric] - groups[g2][metric])
                    if diff > 0.02:
                        print(f"    Δ{metric} ({g1} vs {g2}): {diff:.3f} ← смещение!")

    # --- 1.4 Aggregation bias ---
    print("\n--- 1.4 Aggregation bias (смещение агрегации) ---")

    # Модель одна для всех, но данные разных групп имеют разные распределения
    groups_data = {
        "молодые (<30)": {"mean_income": 35000, "std_income": 12000, "mean_age": 25, "n": 300},
        "средние (30-50)": {"mean_income": 65000, "std_income": 20000, "mean_age": 40, "n": 500},
        "старшие (>50)": {"mean_income": 55000, "std_income": 18000, "mean_age": 55, "n": 200},
    }

    # Одна модель: y = w1 * income + w2 * age
    w1, w2 = 0.001, 100  # веса глобальной модели

    print("Формула глобальной модели: score = w1 × income + w2 × age")
    print(f"  w1={w1}, w2={w2}\n")

    # Для каждой группы считаем R²
    for gname, gdata in groups_data.items():
        pred_score = w1 * gdata["mean_income"] + w2 * gdata["mean_age"]
        # Симулируем реальный score с групповым смещением
        group_effect = 0.002 * gdata["mean_income"] + 50 * gdata["mean_age"]
        error = abs(pred_score - group_effect) / group_effect if group_effect != 0 else 0
        r_squared = max(0, 1 - error ** 2)

        print(f"  {gname}:")
        print(f"    pred_score = {pred_score:.1f}, group_effect = {group_effect:.1f}")
        print(f"    ошибка = {error:.3f}, R² ≈ {r_squared:.3f}")

    print("\n  Вывод: одна модель не учитывает различия между группами")

    # --- 1.5 Deployment bias ---
    print("\n--- 1.5 Deployment bias (смещение развёртывания) ---")

    # Модель работает хорошо в лаборатории, но не в реальном мире
    deployment_scenarios = [
        {"условие": "лаборатория (данные чистые)", "accuracy": 0.94, "coverage": 1.0},
        {"условие": "город (данные разнообразные)", "accuracy": 0.87, "coverage": 0.95},
        {"условие": "сельская местность", "accuracy": 0.72, "coverage": 0.60},
        {"условие": "развивающиеся страны", "accuracy": 0.55, "coverage": 0.30},
    ]

    print("Производительность модели в разных условиях:\n")
    for scenario in deployment_scenarios:
        degradation = (0.94 - scenario["accuracy"]) / 0.94 * 100
        print(f"  {scenario['условие']}:")
        print(f"    accuracy={scenario['accuracy']:.2f}, coverage={scenario['coverage']:.2f}, "
              f"деградация={degradation:.1f}%")


# =============================================================================
# Демо 2: Метрики справедливости (fairness metrics)
# =============================================================================
def demo_fairness_metrics():
    print("\n\n" + "=" * 70)
    print("Демо 2: Метрики справедливости (fairness metrics)")
    print("=" * 70)

    # --- 2.1 Demographic Parity ---
    print("\n--- 2.1 Demographic Parity (демографический паритет) ---")

    # Данные: положительные решения по группам
    results = {
        "группа A": {"positive": 180, "negative": 120, "total": 300},
        "группа B": {"positive": 45, "negative": 105, "total": 150},
        "группа C": {"positive": 20, "negative": 80, "total": 100},
    }

    # Формула: PR = positive / total
    # Demographic parity: |PR_A - PR_B| <= threshold
    print("Формула: PR(group) = positive / total")
    print("Демографический паритет: |PR_A - PR_B| <= threshold\n")

    positive_rates = {}
    for gname, gdata in results.items():
        pr = gdata["positive"] / gdata["total"]
        positive_rates[gname] = pr
        print(f"  {gname}: PR = {gdata['positive']}/{gdata['total']} = {pr:.4f}")

    # Проверяем попарно
    groups = list(positive_rates.keys())
    print("\n  Попарные разницы:")
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            diff = abs(positive_rates[groups[i]] - positive_rates[groups[j]])
            threshold = 0.1
            status = "✓ проходит" if diff <= threshold else "✗ нарушено"
            print(f"    |PR({groups[i]}) - PR({groups[j]})| = {diff:.4f} {status} "
                  f"(threshold={threshold})")

    # --- 2.2 Equalized Odds ---
    print("\n--- 2.2 Equalized Odds (уравненные шансы) ---")

    # Для группы положительных и отрицательных истинных значений
    equalized_data = {
        "группа A": {
            "tp_rate": 0.85,  # true positive rate (sensitivity)
            "fp_rate": 0.10,  # false positive rate
            "tn_rate": 0.90,  # true negative rate
            "fn_rate": 0.15,  # false negative rate
        },
        "группа B": {
            "tp_rate": 0.70,
            "fp_rate": 0.20,
            "tn_rate": 0.80,
            "fn_rate": 0.30,
        },
    }

    # Формулы
    print("Equalized odds требует:")
    print("  TPR_A = TPR_B (одинаковая чувствительность)")
    print("  FPR_A = FPR_B (одинаковая ложная положительность)\n")

    for gname, gdata in equalized_data.items():
        print(f"  {gname}:")
        print(f"    TPR (sensitivity) = {gdata['tp_rate']:.2f}")
        print(f"    FPR = {gdata['fp_rate']:.2f}")
        print(f"    TNR (specificity) = {gdata['tn_rate']:.2f}")
        print(f"    FNR = {gdata['fn_rate']:.2f}")

    # Проверка equalized odds
    tpr_diff = abs(equalized_data["группа A"]["tp_rate"] - equalized_data["группа B"]["tp_rate"])
    fpr_diff = abs(equalized_data["группа A"]["fp_rate"] - equalized_data["группа B"]["fp_rate"])
    threshold = 0.05

    print(f"\n  Разница TPR: {tpr_diff:.3f} {'✓' if tpr_diff <= threshold else '✗'} "
          f"(threshold={threshold})")
    print(f"  Разница FPR: {fpr_diff:.3f} {'✓' if fpr_diff <= threshold else '✗'} "
          f"(threshold={threshold})")

    # --- 2.3 Individual Fairness ---
    print("\n--- 2.3 Individual Fairness (индивидуальная справедливость) ---")

    # Формула: для похожих людей, результаты должны быть похожими
    # d(f(x), f(y)) <= L × d(x, y)

    # Симулируем людей с признаками
    people = [
        {"id": 1, "income": 50000, "age": 35, "education": 4, "decision": "approved"},
        {"id": 2, "income": 51000, "age": 36, "education": 4, "decision": "denied"},
        {"id": 3, "income": 80000, "age": 35, "education": 5, "decision": "approved"},
        {"id": 4, "income": 50500, "age": 34, "education": 4, "decision": "approved"},
    ]

    # Расстояние между людьми: нормализованное евклидово расстояние
    def person_distance(p1, p2):
        """Расстояние между двумя людьми (нормализованное)."""
        norm_income = 100000  # нормализация
        norm_age = 10
        norm_edu = 5
        d = math.sqrt(
            ((p1["income"] - p2["income"]) / norm_income) ** 2
            + ((p1["age"] - p2["age"]) / norm_age) ** 2
            + ((p1["education"] - p2["education"]) / norm_edu) ** 2
        )
        return d

    def decision_distance(d1, d2):
        """Расстояние решений: 0 если одинаковые, 1 если разные."""
        return 0 if d1 == d2 else 1

    print("Формула: d(f(x), f(y)) <= L × d(x, y)")
    print("L = Lipschitz constant (константа Липшица)\n")

    # Проверяем пары похожих людей
    L = 2.0  # допустимая константа Липшица
    print(f"Допустимая L = {L}\n")

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1, p2 = people[i], people[j]
            dist = person_distance(p1, p2)
            dec_dist = decision_distance(p1["decision"], p2["decision"])
            violation = dec_dist > L * dist
            status = "✗ нарушена" if violation else "✓ соблюдена"

            if dist < 0.5:  # смотрим только на похожих людей
                print(f"  Человек {p1['id']} vs {p2['id']}:")
                print(f"    x-distance = {dist:.4f}")
                print(f"    f-distance = {dec_dist} ({p1['decision']} vs {p2['decision']})")
                print(f"    L × d(x,y) = {L * dist:.4f}")
                print(f"    {status}")

    # --- 2.4 Calibration ---
    print("\n--- 2.4 Calibration (калибровка) ---")

    # Модель выдаёт вероятности; калибровка: P(actual | predicted=p) ≈ p
    calibration_data = {
        "-bin_0.0-0.2": {"predicted_mean": 0.12, "actual_positive_rate": 0.10, "count": 200},
        "bin_0.2-0.4": {"predicted_mean": 0.31, "actual_positive_rate": 0.28, "count": 180},
        "bin_0.4-0.6": {"predicted_mean": 0.48, "actual_positive_rate": 0.52, "count": 150},
        "bin_0.6-0.8": {"predicted_mean": 0.72, "actual_positive_rate": 0.75, "count": 120},
        "bin_0.8-1.0": {"predicted_mean": 0.89, "actual_positive_rate": 0.91, "count": 100},
    }

    print("Формула: калиброванность = |predicted_mean - actual_rate|\n")
    print(f"  {'Диапазон':>15} {'Предсказанный':>14} {'Фактический':>14} {'Разница':>10} {'N':>6}")
    print("  " + "-" * 65)

    total_calibration_error = 0
    total_count = 0
    for bin_name, data in calibration_data.items():
        cal_error = abs(data["predicted_mean"] - data["actual_positive_rate"])
        total_calibration_error += cal_error * data["count"]
        total_count += data["count"]
        status = "✓" if cal_error < 0.05 else "✗"
        print(f"  {bin_name:>15} {data['predicted_mean']:>14.3f} "
              f"{data['actual_positive_rate']:>14.3f} {cal_error:>10.3f} "
              f"{data['count']:>6} {status}")

    avg_cal_error = total_calibration_error / total_count if total_count > 0 else 0
    print(f"\n  Expected Calibration Error (ECE): {avg_cal_error:.4f}")


# =============================================================================
# Демо 3: Обнаружение смещений (bias detection)
# =============================================================================
def demo_bias_detection():
    print("\n\n" + "=" * 70)
    print("Демо 3: Обнаружение смещений (bias detection)")
    print("=" * 70)

    # --- 3.1 Disparate Impact ---
    print("\n--- 3.1 Disparate Impact (несоразмерное воздействие) ---")

    # Данные: одобрение кредитов
    approval_data = {
        "основная_группа": {"approved": 800, "rejected": 200},
        "защищённая_группа": {"approved": 150, "rejected": 100},
    }

    # Формула: DI = PR(protected) / PR(reference)
    # 4/5 rule: DI >= 0.8 → нет显著ного воздействия
    print("Формула: DI = PR(protected) / PR(reference)")
    print("Правило 4/5: DI >= 0.8 → chấp nhậnável\n")

    for group_name, group_data in approval_data.items():
        pr = group_data["approved"] / (group_data["approved"] + group_data["rejected"])
        approval_data[group_name]["pr"] = pr
        print(f"  {group_name}: approved={group_data['approved']}, "
              f"rejected={group_data['rejected']}, PR={pr:.4f}")

    pr_ref = approval_data["основная_группа"]["pr"]
    pr_prot = approval_data["защищённая_группа"]["pr"]
    di = pr_prot / pr_ref if pr_ref > 0 else 0
    threshold = 0.8

    print(f"\n  DI = {pr_prot:.4f} / {pr_ref:.4f} = {di:.4f}")
    print(f"  Порог: {threshold}")
    print(f"  Результат: {'✓ chấp nhậnável' if di >= threshold else '✗ неприемлемо'}")

    # --- 3.2 Statistical Parity Difference ---
    print("\n--- 3.2 Statistical Parity Difference ---")

    # SPD = PR(protected) - PR(reference)
    spd = pr_prot - pr_ref
    print(f"Формула: SPD = PR(protected) - PR(reference)")
    print(f"  SPD = {pr_prot:.4f} - {pr_ref:.4f} = {spd:.4f}")
    print(f"  Идеальное значение: 0.0")
    print(f"  Допустимый диапазон: [-0.1, 0.1]")

    # Визуализация
    bar_ref = "█" * int(pr_ref * 40)
    bar_prot = "█" * int(pr_prot * 40)
    print(f"\n  Основная:  {bar_ref} {pr_ref:.3f}")
    print(f"  Защищённая: {bar_prot} {pr_prot:.3f}")

    # --- 3.3 Калибровка по группам ---
    print("\n--- 3.3 Калибровка по группам ---")

    # Модель калибрована в целом, но не по группам
    group_calibration = {
        "мужчины": {
            "predicted": [0.2, 0.4, 0.6, 0.8],
            "actual": [0.18, 0.42, 0.58, 0.82],
        },
        "женщины": {
            "predicted": [0.2, 0.4, 0.6, 0.8],
            "actual": [0.30, 0.50, 0.65, 0.85],
        },
    }

    print("Групповая калибровка: одинаковые predicted → одинаковые actual?\n")

    all_diffs = []
    for group_name, group_data in group_calibration.items():
        diffs = [abs(p - a) for p, a in zip(group_data["predicted"], group_data["actual"])]
        mean_diff = sum(diffs) / len(diffs)
        all_diffs.append(mean_diff)
        print(f"  {group_name}:")
        for p, a in zip(group_data["predicted"], group_data["actual"]):
            print(f"    predicted={p:.1f} → actual={a:.2f} (diff={abs(p - a):.2f})")
        print(f"    средняя ошибка калибровки: {mean_diff:.4f}")

    calibration_gap = abs(all_diffs[0] - all_diffs[1])
    print(f"\n  Разрыв калибровки между группами: {calibration_gap:.4f}")
    print(f"  {'✓ калибровка адекватна' if calibration_gap < 0.05 else '✗ необходима перекалибровка'}")

    # --- 3.4 Cross-validation fairness ---
    print("\n--- 3.4 Cross-validation fairness ---")

    # Проверяем стабильность fairness метрик по фолдам
    n_folds = 5
    fairness_per_fold = []

    random.seed(42)
    for fold in range(n_folds):
        # Симулируем DI для каждого фолда
        base_di = 0.75
        noise = random.gauss(0, 0.03)
        fold_di = base_di + noise
        fairness_per_fold.append(fold_di)
        print(f"  Fold {fold + 1}: DI = {fold_di:.4f}")

    mean_di = sum(fairness_per_fold) / len(fairness_per_fold)
    std_di = math.sqrt(sum((x - mean_di) ** 2 for x in fairness_per_fold) / len(fairness_per_fold))
    cv = std_di / mean_di if mean_di != 0 else 0

    print(f"\n  Среднее DI: {mean_di:.4f}")
    print(f"  Стд. отклонение: {std_di:.4f}")
    print(f"  Коэффициент вариации: {cv:.4f}")
    print(f"  {'✓ стабильно' if cv < 0.1 else '✗ нестабильно — требует улучшения'}")


# =============================================================================
# Демо 4: Смягчение смещений (bias mitigation)
# =============================================================================
def demo_bias_mitigation():
    print("\n\n" + "=" * 70)
    print("Демо 4: Смягчение смещений (bias mitigation)")
    print("=" * 70)

    # --- 4.1 Pre-processing ---
    print("\n--- 4.1 Pre-processing (предобработка данных) ---")

    # Техника 1: Reweighting (перевзвешивание)
    print("Техника: Reweighting (перевзвешивание образцов)\n")

    random.seed(42)
    n_samples = 200
    # Исходные данные с смещением
    data = []
    for _ in range(n_samples):
        group = random.choice(["A", "B"])
        feature = random.gauss(50, 15)
        # Смещение: группа A чаще получает положительный исход
        bias = 10 if group == "A" else -10
        outcome = 1 if (feature + bias + random.gauss(0, 20)) > 50 else 0
        data.append({"group": group, "feature": feature, "outcome": outcome})

    # Считаем исходные метрики
    groups_original = {"A": {"pos": 0, "total": 0}, "B": {"pos": 0, "total": 0}}
    for d in data:
        groups_original[d["group"]]["total"] += 1
        groups_original[d["group"]]["pos"] += d["outcome"]

    pr_a_orig = groups_original["A"]["pos"] / groups_original["A"]["total"]
    pr_b_orig = groups_original["B"]["pos"] / groups_original["B"]["total"]
    print(f"  До обработки: PR_A={pr_a_orig:.4f}, PR_B={pr_b_orig:.4f}, "
          f"SPD={abs(pr_a_orig - pr_b_orig):.4f}")

    # Вычисляем веса для выравнивания
    overall_positive_rate = sum(d["outcome"] for d in data) / len(data)
    weights = []
    for d in data:
        group_rate = groups_original[d["group"]]["pos"] / groups_original[d["group"]]["total"]
        # Вес = (общая частота) / (групповая частота) для этого исхода
        if d["outcome"] == 1:
            weight = overall_positive_rate / group_rate if group_rate > 0 else 1
        else:
            weight = (1 - overall_positive_rate) / (1 - group_rate) if group_rate < 1 else 1
        weights.append(weight)

    # Применяем перевзвешивание
    weighted_positive_a = sum(weights[i] for i in range(len(data))
                              if data[i]["group"] == "A" and data[i]["outcome"] == 1)
    weighted_total_a = sum(weights[i] for i in range(len(data)) if data[i]["group"] == "A")
    weighted_positive_b = sum(weights[i] for i in range(len(data))
                              if data[i]["group"] == "B" and data[i]["outcome"] == 1)
    weighted_total_b = sum(weights[i] for i in range(len(data)) if data[i]["group"] == "B")

    pr_a_weighted = weighted_positive_a / weighted_total_a if weighted_total_a > 0 else 0
    pr_b_weighted = weighted_positive_b / weighted_total_b if weighted_total_b > 0 else 0

    print(f"  После обработки: PR_A={pr_a_weighted:.4f}, PR_B={pr_b_weighted:.4f}, "
          f"SPD={abs(pr_a_weighted - pr_b_weighted):.4f}")

    # Техника 2: Disparate Impact Remover
    print("\nТехника: Disparate Impact Remover (ремувер несоразмерного воздействия)\n")

    # Корректируем признак, чтобы уменьшить корреляцию с защищённым атрибутом
    repair_level = 0.5  # 0 = без изменений, 1 = полная коррекция

    features_a = [d["feature"] for d in data if d["group"] == "A"]
    features_b = [d["feature"] for d in data if d["group"] == "B"]

    # Калибры распределений (процентили)
    sorted_a = sorted(features_a)
    sorted_b = sorted(features_b)
    median_a = sorted_a[len(sorted_a) // 2]
    median_b = sorted_b[len(sorted_b) // 2]

    print(f"  repair_level = {repair_level}")
    print(f"  Медиана группы A: {median_a:.2f}")
    print(f"  Медиана группы B: {median_b:.2f}")

    # Корректируем признаки группы B к медиане A
    shift = (median_a - median_b) * repair_level
    corrected_features_b = [f + shift for f in features_b]
    new_median_b = sum(corrected_features_b) / len(corrected_features_b)

    print(f"  Сдвиг для группы B: {shift:.2f}")
    print(f"  Новая медиана группы B: {new_median_b:.2f}")

    # --- 4.2 In-processing ---
    print("\n--- 4.2 In-processing (внутрипроцессная коррекция) ---")

    # Техника: Adversarial Debiasing (состязательное обесценение)
    print("Техника: Adversarial Debiasing (состязательное обучение)\n")

    # Симулируем процесс обучения
    epochs = 10
    predictor_loss = []
    adversary_loss = []

    p_loss = 0.8
    a_loss = 0.6

    print(f"  {'Эпоха':>6} {'Predictor Loss':>15} {'Adversary Loss':>15} {'Смещение':>10}")
    print("  " + "-" * 50)

    for epoch in range(epochs):
        # Predictor учится предсказывать, adversary пытается угадать группу
        p_loss *= random.uniform(0.88, 0.96)
        a_loss *= random.uniform(0.90, 1.05)  # adversary может расти (это хорошо!)
        bias_level = a_loss  # чем выше loss adversary, тем меньше смещение

        predictor_loss.append(p_loss)
        adversary_loss.append(a_loss)

        marker = "✓" if epoch >= 3 else " "
        print(f"  {epoch + 1:>6} {p_loss:>15.4f} {a_loss:>15.4f} {bias_level:>10.4f} {marker}")

    print(f"\n  Predictor loss: {predictor_loss[0]:.4f} → {predictor_loss[-1]:.4f} "
          f"(уменьшение: {(1 - predictor_loss[-1] / predictor_loss[0]) * 100:.1f}%)")
    print(f"  Adversary loss: {adversary_loss[0]:.4f} → {adversary_loss[-1]:.4f} "
          f"(цель: рост = модель скрывает группу)")

    # Техника: Fairness Constraints
    print("\nТехника: Fairness Constraints (ограничения справедливости)\n")

    # Добавляем штраф за несправедливость в функцию потерь
    lambda_fairness = 0.5  # вес fairness штрафа
    base_loss = 0.35
    unfairness = 0.20
    total_loss = base_loss + lambda_fairness * unfairness

    print(f"  Loss = base_loss + λ × unfairness")
    print(f"  base_loss = {base_loss}")
    print(f"  unfairness = {unfairness}")
    print(f"  λ = {lambda_fairness}")
    print(f"  total_loss = {total_loss:.4f}")

    # Разные значения λ
    print("\n  Влияние λ на баланс точность/справедливость:\n")
    for lam in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        tl = base_loss + lam * unfairness
        acc = max(0, 0.95 - lam * 0.05)  # модельная точность
        fair = max(0, 1.0 - unfairness * (1 - lam / 3))
        print(f"    λ={lam:.1f}: total_loss={tl:.3f}, accuracy≈{acc:.2f}, fairness≈{fair:.2f}")

    # --- 4.3 Post-processing ---
    print("\n--- 4.3 Post-processing (постобработка) ---")

    # Техника: Threshold Optimization (оптимизация порогов)
    print("Техника: Threshold Optimization (отдельные пороги по группам)\n")

    # Исходные предсказания
    predictions = [
        {"group": "A", "score": 0.85, "label": 1},
        {"group": "A", "score": 0.45, "label": 0},
        {"group": "A", "score": 0.72, "label": 1},
        {"group": "A", "score": 0.33, "label": 0},
        {"group": "B", "score": 0.65, "label": 1},
        {"group": "B", "score": 0.40, "label": 1},
        {"group": "B", "score": 0.55, "label": 0},
        {"group": "B", "score": 0.30, "label": 0},
    ]

    # Один порог для всех
    common_threshold = 0.5
    print(f"  Единый порог: {common_threshold}")
    group_metrics = {}
    for group in ["A", "B"]:
        preds = [p for p in predictions if p["group"] == group]
        tp = sum(1 for p in preds if p["score"] >= common_threshold and p["label"] == 1)
        fp = sum(1 for p in preds if p["score"] >= common_threshold and p["label"] == 0)
        fn = sum(1 for p in preds if p["score"] < common_threshold and p["label"] == 1)
        tn = sum(1 for p in preds if p["score"] < common_threshold and p["label"] == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        group_metrics[group] = {"tpr": tpr, "fpr": fpr}
        print(f"    Группа {group}: TPR={tpr:.2f}, FPR={fpr:.2f}")

    print(f"\n  Разница TPR: {abs(group_metrics['A']['tpr'] - group_metrics['B']['tpr']):.2f}")

    # Оптимизированные пороги по группам
    print(f"\n  Оптимизированные пороги:")
    thresholds = {"A": 0.5, "B": 0.35}  # смещённый порог для группы B
    for group in ["A", "B"]:
        preds = [p for p in predictions if p["group"] == group]
        t = thresholds[group]
        tp = sum(1 for p in preds if p["score"] >= t and p["label"] == 1)
        fp = sum(1 for p in preds if p["score"] >= t and p["label"] == 0)
        fn = sum(1 for p in preds if p["score"] < t and p["label"] == 1)
        tn = sum(1 for p in preds if p["score"] < t and p["label"] == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"    Группа {group}: threshold={t}, TPR={tpr:.2f}, FPR={fpr:.2f}")

    # --- 4.4 Сравнение подходов ---
    print("\n--- 4.4 Сравнение подходов к смягчению ---")

    approaches = {
        "Pre-processing (reweight)": {
            "accuracy_drop": 0.02,
            "fairness_improvement": 0.15,
            "implementation_cost": "низкая",
            "transparency": "высокая",
        },
        "Pre-processing (remover)": {
            "accuracy_drop": 0.03,
            "fairness_improvement": 0.12,
            "implementation_cost": "средняя",
            "transparency": "средняя",
        },
        "In-processing (adversarial)": {
            "accuracy_drop": 0.01,
            "fairness_improvement": 0.20,
            "implementation_cost": "высокая",
            "transparency": "низкая",
        },
        "In-processing (constraints)": {
            "accuracy_drop": 0.04,
            "fairness_improvement": 0.18,
            "implementation_cost": "средняя",
            "transparency": "средняя",
        },
        "Post-processing (threshold)": {
            "accuracy_drop": 0.01,
            "fairness_improvement": 0.10,
            "implementation_cost": "низкая",
            "transparency": "высокая",
        },
    }

    print("Сравнение подходов:\n")
    print(f"  {'Подход':<30} {'ΔAccuracy':>10} {'ΔFairness':>10} {'Стоимость':>10} {'Прозрач.':>10}")
    print("  " + "-" * 75)

    for approach, data in approaches.items():
        print(f"  {approach:<30} {data['accuracy_drop']:>10.2f} "
              f"{data['fairness_improvement']:>10.2f} "
              f"{data['implementation_cost']:>10} {data['transparency']:>10}")

    # Рекомендация на основе Pareto efficiency
    print("\n  Pareto-эффективные подходы:")
    pareto = []
    for a1_name, a1 in approaches.items():
        dominated = False
        for a2_name, a2 in approaches.items():
            if a1_name != a2_name:
                if (a2["accuracy_drop"] <= a1["accuracy_drop"]
                        and a2["fairness_improvement"] >= a1["fairness_improvement"]):
                    dominated = True
                    break
        if not dominated:
            pareto.append(a1_name)

    for p in pareto:
        print(f"    - {p}: {approaches[p]}")


# =============================================================================
# Точка входа
# =============================================================================
if __name__ == "__main__":
    demo_bias_types()
    demo_fairness_metrics()
    demo_bias_detection()
    demo_bias_mitigation()
