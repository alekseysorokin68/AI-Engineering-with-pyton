"""205 — ML Monitoring: дрейф данных, дрейф моделей, мониторинг производительности

Темы:
  1. Data Drift (обнаружение смещения распределений, PSI, KS-тест)
  2. Model Drift (дрейф концепции, деградация производительности, алертинг)
  3. Performance Monitoring (задержка, пропускная способность, ошибки, SLA)
  4. Monitoring Dashboards (ключевые метрики, визуализация, правила алертинга)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import statistics

random.seed(42)

# ============================================================================
# Демо 1: Data Drift — смещение распределений, PSI, KS-тест
# ============================================================================

def demo_data_drift():
    """Демонстрация обнаружения дрейфа данных: PSI, KS-тест, визуализация."""
    print("=" * 70)
    print("ДЕМО 1: Data Drift — обнаружение смещения распределений")
    print("=" * 70)

    # --- 1.1 Population Stability Index (PSI) ---
    print("\n--- 1.1 Population Stability Index (PSI) ---")
    # Формула: PSI = Σ (P_actual - P_ref) × ln(P_actual / P_ref)
    # PSI < 0.1 → нет дрейфа, 0.1-0.2 → умеренный, > 0.2 → сильный

    def compute_psi_manual(reference, current, n_bins=10):
        """Вычисление PSI между двумя выборками."""
        all_vals = sorted(reference + current)
        min_v, max_v = all_vals[0], all_vals[-1]
        # Границы бинов
        bin_width = (max_v - min_v) / n_bins
        edges = [min_v + i * bin_width for i in range(n_bins + 1)]

        def count_in_bins(data):
            counts = [0] * n_bins
            for v in data:
                idx = min(int((v - min_v) / bin_width), n_bins - 1)
                counts[idx] += 1
            return counts

        ref_counts = count_in_bins(reference)
        cur_counts = count_in_bins(current)

        eps = 1e-6
        ref_prop = [(c / len(reference)) + eps for c in ref_counts]
        cur_prop = [(c / len(current)) + eps for c in cur_counts]

        psi = sum((cur_prop[i] - ref_prop[i]) * math.log(cur_prop[i] / ref_prop[i])
                  for i in range(n_bins))
        return psi, ref_prop, cur_prop

    # Эталонное распределение (обучающие данные)
    random.seed(42)
    ref_data = [random.gauss(100, 20) for _ in range(1000)]
    # Текущее распределение без дрейфа
    random.seed(55)
    cur_clean = [random.gauss(100, 20) for _ in range(1000)]
    # Текущее распределение с дрейфом среднего
    random.seed(77)
    cur_drift = [random.gauss(115, 25) for _ in range(1000)]

    psi_clean, _, _ = compute_psi_manual(ref_data, cur_clean)
    psi_drift, _, _ = compute_psi_manual(ref_data, cur_drift)

    print(f"Эталон: N=1000, mean≈100, std≈20")
    print(f"Без дрейфа: PSI = {psi_clean:.4f} -> {'НОРМА' if psi_clean < 0.1 else 'ДРЕЙФ'}")
    print(f"С дрейфом:  PSI = {psi_drift:.4f} -> {'НОРМА' if psi_drift < 0.1 else 'ДРЕЙФ'}")
    print(f"Пороги: <0.1 (норма), 0.1-0.2 (умеренный), >0.2 (сильный)")

    # --- 1.2 Kolmogorov-Smirnov Test ---
    print("\n--- 1.2 Kolmogorov-Smirnov (KS) Test ---")
    # KS-статистика = max|CDF_1(x) - CDF_2(x)|

    def ks_test(sample1, sample2):
        """Вычисление KS-статистики и p-value (аппроксимация)."""
        sorted1 = sorted(sample1)
        sorted2 = sorted(sample2)
        n1, n2 = len(sorted1), len(sorted2)
        all_vals = sorted(set(sorted1 + sorted2))

        max_diff = 0
        for v in all_vals:
            cdf1 = sum(1 for x in sorted1 if x <= v) / n1
            cdf2 = sum(1 for x in sorted2 if x <= v) / n2
            diff = abs(cdf1 - cdf2)
            if diff > max_diff:
                max_diff = diff

        # Аппроксимация p-value через KS распределение
        # p-value ≈ 2 * exp(-2 * (sqrt(n_eff) * D)^2)
        n_eff = (n1 * n2) / (n1 + n2)
        approx_p = 2 * math.exp(-2 * (math.sqrt(n_eff) * max_diff) ** 2)
        return max_diff, approx_p

    random.seed(42)
    ks_a = [random.gauss(50, 10) for _ in range(500)]
    random.seed(55)
    ks_b_same = [random.gauss(50, 10) for _ in range(500)]
    random.seed(77)
    ks_b_diff = [random.gauss(60, 15) for _ in range(500)]

    ks_stat_same, p_same = ks_test(ks_a, ks_b_same)
    ks_stat_diff, p_diff = ks_test(ks_a, ks_b_diff)

    print(f"KS-тест (одинаковые распределения):")
    print(f"  KS = {ks_stat_same:.4f}, p-value ≈ {p_same:.4f}")
    print(f"  Решение: {'ДРЕЙФ' if p_same < 0.05 else 'НОРМА'} (α=0.05)")
    print(f"KS-тест (различные распределения):")
    print(f"  KS = {ks_stat_diff:.4f}, p-value ≈ {p_diff:.4f}")
    print(f"  Решение: {'ДРЕЙФ' if p_diff < 0.05 else 'НОРМА'} (α=0.05)")

    # --- 1.3 Квантильное сравнение (Quantile Comparison) ---
    print("\n--- 1.3 Квантильное сравнение распределений ---")

    def quantile(data, q):
        """Вычисление квантиля (q от 0 до 1)."""
        sorted_data = sorted(data)
        idx = q * (len(sorted_data) - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_data[lower]
        return sorted_data[lower] + (sorted_data[upper] - sorted_data[lower]) * (idx - lower)

    quantiles_to_check = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"{'Квантиль':>10} | {'Эталон':>10} | {'Без дрейфа':>10} | {'С дрейфом':>10} | {'Разница':>10}")
    print("-" * 60)
    for q in quantiles_to_check:
        ref_q = quantile(ref_data, q)
        clean_q = quantile(cur_clean, q)
        drift_q = quantile(cur_drift, q)
        diff = abs(drift_q - ref_q)
        print(f"{q:10.2f} | {ref_q:10.2f} | {clean_q:10.2f} | {drift_q:10.2f} | {diff:10.2f}")

    # --- 1.4 Multivariate Drift Detection ---
    print("\n--- 1.4 Многомерный дрейф ( корреляционная матрица) ---")
    random.seed(42)
    n = 500
    # Генерируем связанные признаки
    x1 = [random.gauss(0, 1) for _ in range(n)]
    x2 = [x1[i] * 0.7 + random.gauss(0, 0.7) for i in range(n)]  # коррелированный
    x3 = [random.gauss(0, 1) for _ in range(n)]  # некоррелированный

    def pearson_corr(a, b):
        """Коэффициент корреляции Пирсона."""
        n = len(a)
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        std_a = math.sqrt(sum((a[i] - mean_a) ** 2 for i in range(n)) / n)
        std_b = math.sqrt(sum((b[i] - mean_b) ** 2 for i in range(n)) / n)
        return cov / (std_a * std_b) if std_a * std_b > 0 else 0

    corr_matrix = [
        [pearson_corr(x1, x1), pearson_corr(x1, x2), pearson_corr(x1, x3)],
        [pearson_corr(x2, x1), pearson_corr(x2, x2), pearson_corr(x2, x3)],
        [pearson_corr(x3, x1), pearson_corr(x3, x2), pearson_corr(x3, x3)],
    ]
    feature_names = ["x1", "x2", "x3"]
    print("Корреляционная матрица (эталон):")
    header = f"{'':>5}" + "".join(f"{name:>8}" for name in feature_names)
    print(header)
    for i, name in enumerate(feature_names):
        row = "".join(f"{corr_matrix[i][j]:8.4f}" for j in range(3))
        print(f"{name:>5}{row}")

    # Дрейф: меняем корреляцию x1-x2
    random.seed(77)
    x1_drift = [random.gauss(0, 1) for _ in range(n)]
    x2_drift = [x1_drift[i] * 0.2 + random.gauss(0, 1) for i in range(n)]
    corr_changed = pearson_corr(x1_drift, x2_drift)
    print(f"\nКорреляция x1-x2: эталон={pearson_corr(x1, x2):.4f}, "
          f"текущая={corr_changed:.4f}")
    print(f"Изменение: {abs(corr_changed - pearson_corr(x1, x2)):.4f} "
          f"-> {'АНОМАЛИЯ' if abs(corr_changed - pearson_corr(x1, x2)) > 0.2 else 'норма'}")

    print("\n[OK] Data Drift — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 2: Model Drift — концептуальный дрейф, деградация, алертинг
# ============================================================================

def demo_model_drift():
    """Демонстрация дрейфа модели: концептуальный дрейф, деградация, алерты."""
    print("=" * 70)
    print("ДЕМО 2: Model Drift — концептуальный дрейф и деградация")
    print("=" * 70)

    # --- 2.1 Мониторинг метрик модели во времени ---
    print("\n--- 2.1 Мониторинг accuracy модели по дням ---")
    # Симуляция: модель обучена на данных до января, дрейф концепции в марте
    random.seed(42)
    model_metrics = []
    base_accuracy = 0.89
    for day in range(1, 31):
        # Постепенная деградация с дня 20
        if day < 20:
            drift = 0
        else:
            drift = -0.003 * (day - 20)  # линейное падение
        noise = random.gauss(0, 0.005)
        acc = base_accuracy + drift + noise
        acc = max(0, min(1, acc))
        model_metrics.append({"day": day, "accuracy": round(acc, 4)})

    # Вывод: только каждые 5 дней + аномальные
    print(f"{'День':>5} | {'Accuracy':>10} | {'Статус':>15}")
    print("-" * 35)
    for m in model_metrics:
        if m["day"] % 5 == 0 or m["accuracy"] < 0.85:
            status = "КРИТИЧНО" if m["accuracy"] < 0.85 else "норма"
            print(f"{m['day']:5d} | {m['accuracy']:10.4f} | {status:>15}")

    # --- 2.2 Running Window Alerting ---
    print("\n--- 2.2 Running Window Alerting (скользящее окно) ---")
    window_size = 7
    alert_threshold = 0.05  # падение > 5% от baseline

    baseline_acc = statistics.mean([m["accuracy"] for m in model_metrics[:7]])
    print(f"Baseline (дни 1-7): accuracy = {baseline_acc:.4f}")
    print(f"Порог алерта: падение > {alert_threshold*100:.0f}%")
    print()

    alerts = []
    for i in range(window_size, len(model_metrics)):
        window = model_metrics[i - window_size: i]
        window_mean = statistics.mean([m["accuracy"] for m in window])
        drop = baseline_acc - window_mean
        if drop > alert_threshold:
            alerts.append((model_metrics[i]["day"], drop))
            print(f"  ALERT: день {model_metrics[i]['day']}, "
                  f"скольз. среднее={window_mean:.4f}, падение={drop:.4f}")

    if not alerts:
        print("  Алертов нет.")
    else:
        print(f"\n  Всего алертов: {len(alerts)}")

    # --- 2.3 Prediction Distribution Monitoring ---
    print("\n--- 2.3 Мониторинг распределения предсказаний ---")
    # Если распределение P(y_pred) меняется — это индикатор дрейфа
    random.seed(42)
    predictions_early = [random.choice([0, 1]) for _ in range(1000)]
    # Позже: модель начинает предсказывать больше положительных
    random.seed(77)
    predictions_late = [random.choices([0, 1], weights=[0.3, 0.7])[0] for _ in range(1000)]

    pos_rate_early = sum(predictions_early) / len(predictions_early)
    pos_rate_late = sum(predictions_late) / len(predictions_late)

    print(f"Ранний период: positive rate = {pos_rate_early:.3f} "
          f"(ожидается ≈0.5)")
    print(f"Поздний период: positive rate = {pos_rate_late:.3f}")
    print(f"Смещение предсказаний: {abs(pos_rate_late - pos_rate_early):.3f}")
    print(f"Статус: {'ДРЕЙФ ПРЕДСКАЗАНИЙ' if abs(pos_rate_late - 0.5) > 0.15 else 'норма'}")

    # --- 2.4 Retraining Trigger Logic ---
    print("\n--- 2.4 Логика триггера переобучения ---")
    retraining_rules = [
        {"metric": "accuracy", "condition": "<", "threshold": 0.82, "window": "7d"},
        {"metric": "psi_score", "condition": ">", "threshold": 0.2, "window": "30d"},
        {"metric": "prediction_bias", "condition": ">", "threshold": 0.15, "window": "7d"},
        {"metric": "latency_p99_ms", "condition": ">", "threshold": 500, "window": "24h"},
    ]
    current_values = {
        "accuracy": 0.84,
        "psi_score": 0.25,
        "prediction_bias": 0.12,
        "latency_p99_ms": 350,
    }
    print("Правила переобучения:")
    triggers = []
    for rule in retraining_rules:
        val = current_values[rule["metric"]]
        triggered = False
        if rule["condition"] == "<" and val < rule["threshold"]:
            triggered = True
        elif rule["condition"] == ">" and val > rule["threshold"]:
            triggered = True
        status = "ТРИГГЕР" if triggered else "ок"
        if triggered:
            triggers.append(rule["metric"])
        print(f"  {rule['metric']:25s} {rule['condition']} {rule['threshold']:8} "
              f"(окно: {rule['window']:3s}) | текущее: {val:>8} -> [{status}]")

    print(f"\nРешение: {'ТРЕБУЕТСЯ ПЕРЕОБУЧЕНИЕ' if triggers else 'переобучение не требуется'}")
    if triggers:
        print(f"Причины: {', '.join(triggers)}")

    print("\n[OK] Model Drift — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 3: Performance Monitoring — задержка, throughput, ошибки, SLA
# ============================================================================

def demo_performance_monitoring():
    """Демонстрация мониторинга производительности: латентность, ошибки, SLA."""
    print("=" * 70)
    print("ДЕМО 3: Performance Monitoring — латентность, ошибки, SLA")
    print("=" * 70)

    # --- 3.1 Латентность инференса (percentiles) ---
    print("\n--- 3.1 Латентность инференса (перцентили) ---")
    random.seed(42)
    # Имитация латентности запросов (в миллисекундах)
    latencies = sorted([random.expovariate(1/50) + 10 for _ in range(1000)])

    percentiles = [50, 90, 95, 99, 99.9]
    print(f"Всего запросов: {len(latencies)}")
    print(f"{'Перцентиль':>12} | {'Задержка (мс)':>15} | {'Статус':>10}")
    print("-" * 45)
    for p in percentiles:
        idx = int(p / 100 * len(latencies))
        idx = min(idx, len(latencies) - 1)
        val = latencies[idx]
        status = "OK" if val < 200 else "WARN" if val < 500 else "КРИТИЧНО"
        print(f"{'p' + str(p):>12} | {val:15.2f} | {status:>10}")

    print(f"\nСредняя задержка: {statistics.mean(latencies):.2f} мс")
    print(f"Ст. отклонение: {statistics.stdev(latencies):.2f} мс")

    # --- 3.2 Throughput и RPS ---
    print("\n--- 3.2 Throughput и Requests Per Second (RPS) ---")
    # Симуляция RPS за минуту (60 секунд)
    random.seed(42)
    rps_timeline = []
    for sec in range(60):
        base_rps = 100
        # Пик нагрузки: секунды 20-30
        if 20 <= sec <= 30:
            base_rps = 300
        rps = base_rps + random.gauss(0, 10)
        rps_timeline.append(max(0, rps))

    print("RPS по интервалам (секунда, rps):")
    intervals = [(0, 19, "нагрузка-базовая"), (20, 30, "ПИК"), (31, 59, "нагрузка-базовая")]
    for start, end, label in intervals:
        segment = rps_timeline[start:end+1]
        avg_rps = statistics.mean(segment)
        max_rps = max(segment)
        print(f"  {label:20s} (сек {start:2d}-{end:2d}): "
              f"avg={avg_rps:.1f}, max={max_rps:.1f}")

    total_requests = sum(rps_timeline)
    print(f"\nВсего запросов за минуту: {total_requests:.0f}")
    print(f"Средний RPS: {total_requests/60:.1f}")

    # --- 3.3 Error Rate Monitoring ---
    print("\n--- 3.3 Мониторинг Error Rate ---")
    # Типы ошибок ML-сервиса
    error_types = {
        "timeout": 0, "model_error": 0, "validation_error": 0,
        "oom": 0, "success": 0,
    }
    random.seed(42)
    for _ in range(5000):
        r = random.random()
        if r < 0.02:
            error_types["timeout"] += 1
        elif r < 0.025:
            error_types["model_error"] += 1
        elif r < 0.03:
            error_types["validation_error"] += 1
        elif r < 0.031:
            error_types["oom"] += 1
        else:
            error_types["success"] += 1

    total = sum(error_types.values())
    error_rate = (total - error_types["success"]) / total * 100
    print(f"Всего запросов: {total}")
    print(f"Успешных: {error_types['success']} ({error_types['success']/total*100:.2f}%)")
    print(f"Ошибок: {total - error_types['success']} ({error_rate:.2f}%)")
    print(f"Детализация ошибок:")
    for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        if err_type != "success" and count > 0:
            pct = count / total * 100
            print(f"  {err_type:20s}: {count:5d} ({pct:.2f}%)")

    sla_target = 99.5  # 99.5% uptime
    actual_uptime = error_types["success"] / total * 100
    print(f"\nSLA: {sla_target}% uptime")
    print(f"Фактический uptime: {actual_uptime:.2f}%")
    print(f"SLA {'ВЫПОЛНЕН' if actual_uptime >= sla_target else 'НАРУШЕН'}")

    # --- 3.4 Мониторинг памяти и GPU ---
    print("\n--- 3.4 Мониторинг ресурсов (память, CPU) ---")
    resource_metrics = {
        "memory_mb": {"values": [], "limit": 8192, "alert_pct": 85},
        "cpu_percent": {"values": [], "limit": 100, "alert_pct": 90},
        "gpu_memory_mb": {"values": [], "limit": 16384, "alert_pct": 80},
        "gpu_utilization": {"values": [], "limit": 100, "alert_pct": 95},
    }
    random.seed(42)
    for _ in range(100):
        resource_metrics["memory_mb"]["values"].append(
            random.gauss(5000, 500))
        resource_metrics["cpu_percent"]["values"].append(
            random.gauss(60, 15))
        resource_metrics["gpu_memory_mb"]["values"].append(
            random.gauss(12000, 1000))
        resource_metrics["gpu_utilization"]["values"].append(
            random.gauss(70, 10))

    print(f"{'Ресурс':20s} | {'Среднее':>10} | {'Макс':>10} | {'Лимит':>10} | {'Статус':>10}")
    print("-" * 70)
    for name, info in resource_metrics.items():
        avg_val = statistics.mean(info["values"])
        max_val = max(info["values"])
        usage_pct = max_val / info["limit"] * 100
        status = "ОК" if usage_pct < info["alert_pct"] else "ВНИМАНИЕ"
        print(f"{name:20s} | {avg_val:10.1f} | {max_val:10.1f} | "
              f"{info['limit']:10d} | {status:>10}")

    print("\n[OK] Performance Monitoring — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 4: Monitoring Dashboards — ключевые метрики, визуализация, алерты
# ============================================================================

def demo_monitoring_dashboards():
    """Демонстрация дашбордов мониторинга: метрики, визуализация, алертинг."""
    print("=" * 70)
    print("ДЕМО 4: Monitoring Dashboards — ключевые метрики и алертинг")
    print("=" * 70)

    # --- 4.1 Сборка дашборда метрик ---
    print("\n--- 4.1 Текущий дашборд ML-сервиса ---")
    dashboard = {
        "service": "recommendation_model_v3",
        "timestamp": "2025-03-15T14:30:00Z",
        "metrics": {
            "accuracy": {"value": 0.876, "unit": "", "trend": "down", "threshold": 0.85},
            "latency_p50_ms": {"value": 42, "unit": "ms", "trend": "stable", "threshold": 100},
            "latency_p99_ms": {"value": 189, "unit": "ms", "trend": "up", "threshold": 500},
            "rps": {"value": 1250, "unit": "req/s", "trend": "stable", "threshold": None},
            "error_rate": {"value": 0.012, "unit": "%", "trend": "stable", "threshold": 0.05},
            "gpu_utilization": {"value": 78, "unit": "%", "trend": "up", "threshold": 90},
            "psi_score": {"value": 0.18, "unit": "", "trend": "up", "threshold": 0.2},
        }
    }

    print(f"Сервис: {dashboard['service']}")
    print(f"Время: {dashboard['timestamp']}")
    print(f"\n{'Метрика':25s} | {'Значение':>10} | {'Тренд':>8} | {'Порог':>8} | {'Статус':>10}")
    print("-" * 75)
    for name, info in dashboard["metrics"].items():
        threshold_str = str(info["threshold"]) if info["threshold"] else "—"
        if info["threshold"] is not None:
            if info["unit"] == "%":
                status = "OK" if info["value"] < info["threshold"] else "АЛЕРТ"
            elif info["trend"] == "up":
                status = "OK" if info["value"] < info["threshold"] else "АЛЕРТ"
            else:
                status = "OK" if info["value"] > info["threshold"] else "АЛЕРТ"
        else:
            status = "—"
        trend_arrow = {"up": "↑", "down": "↓", "stable": "→"}
        print(f"{name:25s} | {info['value']:10} | "
              f"{trend_arrow.get(info['trend'], '?'):>8} | {threshold_str:>8} | {status:>10}")

    # --- 4.2 Правила алертинга (Alerting Rules) ---
    print("\n--- 4.2 Правила алертинга (Alert Rules) ---")
    alert_rules = [
        {
            "name": "HighLatency",
            "condition": "latency_p99_ms > 500",
            "severity": "warning",
            "for": "5m",
            "description": "99-й перцентиль латентности превышает 500мс",
        },
        {
            "name": "ModelDrift",
            "condition": "psi_score > 0.2",
            "severity": "critical",
            "for": "1h",
            "description": "PSI модели превышает порог дрейфа",
        },
        {
            "name": "HighErrorRate",
            "condition": "error_rate > 5%",
            "severity": "critical",
            "for": "10m",
            "description": "Доля ошибок превышает 5%",
        },
        {
            "name": "GPUMemoryHigh",
            "condition": "gpu_memory_usage > 90%",
            "severity": "warning",
            "for": "15m",
            "description": "Использование GPU памяти критически высоко",
        },
    ]
    for rule in alert_rules:
        sev_marker = " !!!" if rule["severity"] == "critical" else " !"
        print(f"  [{rule['severity'].upper():8s}]{sev_marker} {rule['name']}")
        print(f"    Условие: {rule['condition']}")
        print(f"    Для: {rule['for']}, Описание: {rule['description']}")
        print()

    # --- 4.3 ASCII Sparkline для метрик ---
    print("--- 4.3 ASCII Sparkline (тренд метрики) ---")

    def sparkline(values, width=40):
        """Генерация ASCII sparkline из значений."""
        blocks = " ▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1
        # Усекаем/расширяем до width
        step = max(1, len(values) // width)
        sampled = values[::step][:width]
        result = ""
        for v in sampled:
            idx = int((v - mn) / rng * (len(blocks) - 1))
            result += blocks[idx]
        return result

    random.seed(42)
    # Генерируем 7 дней метрик
    metrics_over_time = {
        "accuracy": [0.89 + random.gauss(0, 0.01) - 0.001 * i for i in range(7)],
        "latency_p99": [120 + random.gauss(0, 20) + 5 * i for i in range(7)],
        "error_rate": [0.01 + random.gauss(0, 0.005) for _ in range(7)],
    }
    for metric_name, values in metrics_over_time.items():
        spark = sparkline(values)
        print(f"  {metric_name:20s}: {spark} [{values[0]:.3f} -> {values[-1]:.3f}]")

    # --- 4.4 Инцидент-трекинг ---
    print("\n--- 4.4 Инцидент-трекинг (Incident Log) ---")
    incidents = [
        {"id": "INC-2025-042", "severity": "P1", "status": "resolved",
         "service": "recommendation_model",
         "description": "PSI > 0.3, точность упала на 8%",
         "duration": "2h 15m", "root_cause": "новый тип пользователей"},
        {"id": "INC-2025-043", "severity": "P2", "status": "monitoring",
         "service": "embedding_service",
         "description": "p99 латентность > 1с при пике нагрузки",
         "duration": "45m", "root_cause": "нехватка GPU"},
        {"id": "INC-2025-044", "severity": "P3", "status": "resolved",
         "service": "feature_store",
         "description": "Stale features: online store не обновлялся 10 минут",
         "duration": "10m", "root_cause": "cron job завис"},
    ]
    for inc in incidents:
        print(f"\n  {inc['id']} [{inc['severity']}] — {inc['status'].upper()}")
        print(f"    Сервис: {inc['service']}")
        print(f"    Описание: {inc['description']}")
        print(f"    Длительность: {inc['duration']}")
        print(f"    Корневая причина: {inc['root_cause']}")

    print("\n[OK] Monitoring Dashboards — 4 подпримера выполнены.\n")


# ============================================================================
# Точка входа
# ============================================================================

if __name__ == "__main__":
    demo_data_drift()
    demo_model_drift()
    demo_performance_monitoring()
    demo_monitoring_dashboards()
