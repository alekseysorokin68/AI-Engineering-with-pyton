"""181 — Self-Monitoring: обнаружение аномалий, оценка качества, калибровка уверенности

Темы:
  1. Anomaly Detection — статистические тесты, сдвиг распределений, обнаружение новизны
  2. Quality Estimation — оценка выходов, самопроверка, тесты согласованности
  3. Confidence Calibration — temperature scaling, reliability diagrams, ECE
  4. Self-Diagnostics — health checks, отслеживание деградации, алерты

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


# ──────────────────────────── 1. Anomaly Detection ────────────────────────────

def demo_anomaly_detection():
    """Статистические тесты, сдвиг распределений, обнаружение новизны."""
    print("=" * 70)
    print("DEMO 1 — Anomaly Detection")
    print("=" * 70)

    # --- 1a. Z-score метод ---
    print("\n--- 1a. Z-score метод (обнаружение выбросов) ---")
    # Генерируем данные с несколькими аномалиями
    random.seed(42)
    normal_data = [random.gauss(100, 15) for _ in range(90)]
    # Добавляем аномалии
    anomalies = [250, 280, -50, 300]
    all_data = normal_data + anomalies

    mean = sum(all_data) / len(all_data)
    std = math.sqrt(sum((x - mean) ** 2 for x in all_data) / len(all_data))

    print(f"  Размер выборки: {len(all_data)}")
    mean_val = sum(all_data) / len(all_data)
    std_val = math.sqrt(sum((x - mean_val) ** 2 for x in all_data) / len(all_data))
    print(f"  Среднее: {mean_val:.2f}, Стд. отклонение: {std_val:.2f}")

    threshold = 2.5
    detected = []
    for i, x in enumerate(all_data):
        z = abs(x - mean_val) / std_val
        if z > threshold:
            detected.append((i, x, z))

    print(f"  Порог |z| > {threshold}")
    print(f"  Обнаружено аномалий: {len(detected)}")
    for idx, val, z in detected[:5]:
        print(f"    [{idx}] = {val:.1f}, z-score = {z:.2f}")
    if len(detected) > 5:
        print(f"    ... и ещё {len(detected) - 5}")

    # --- 1b. IQR метод (межквартильный размах) ---
    print("\n--- 1b. IQR метод (межквартильный размах) ---")
    sorted_data = sorted(all_data)
    n = len(sorted_data)
    q1 = sorted_data[n // 4]
    q3 = sorted_data[3 * n // 4]
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_anomalies = [x for x in all_data if x < lower_bound or x > upper_bound]

    print(f"  Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")
    print(f"  Границы: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Аномалий по IQR: {len(iqr_anomalies)}")
    for x in sorted(iqr_anomalies):
        print(f"    {x:.1f}")

    # --- 1c. Сдвиг распределения (Distribution Shift) ---
    print("\n--- 1c. Обнаружение сдвига распределения ---")
    # Базовое распределение и новое (сдвинутое)
    baseline = [random.gauss(50, 10) for _ in range(200)]
    incoming = [random.gauss(55, 12) for _ in range(200)]

    # KS-подобный статистик: максимальное расстояние CDF
    all_vals = sorted(set(baseline + incoming))

    def empirical_cdf(data, x):
        return sum(1 for v in data if v <= x) / len(data)

    max_diff = 0
    max_diff_x = 0
    for x in all_vals:
        d = abs(empirical_cdf(baseline, x) - empirical_cdf(incoming, x))
        if d > max_diff:
            max_diff = d
            max_diff_x = x

    # Эмпирическое значение критерия (приблизительный порог)
    ks_threshold = 3.0 * math.sqrt(2 / 200)  # Приблизительный порог для KS-теста
    shift_detected = max_diff > ks_threshold

    print(f"  Базовое распределение: N(50, 10²), n=200")
    print(f"  Входящее распределение: N(55, 12²), n=200")
    print(f"  KS-статистика: {max_diff:.4f} (порог: {ks_threshold:.4f})")
    print(f"  Сдвиг обнаружен: {'ДА' if shift_detected else 'НЕТ'}")

    # --- 1d. Обнаружение новизны (Novelty Detection) ---
    print("\n--- 1d. Обнаружение новизны (Novelty Detection) ---")
    # Метод: расстояние до k ближайших соседей
    def knn_novelty_score(point, reference, k=5):
        distances = [math.sqrt(sum((p - r) ** 2 for p, r in zip(point, ref)))
                     for ref in reference]
        distances.sort()
        return sum(distances[:k]) / k  # Среднее расстояние до k ближайших

    # 1D пример: тренировочные и тестовые точки
    train = [(random.gauss(10, 2),) for _ in range(100)]
    test_normal = [(random.gauss(10, 2),) for _ in range(20)]
    test_novel = [(random.gauss(30, 2),) for _ in range(5)]

    train_scores = [knn_novelty_score(t, train) for t in train]
    threshold_novelty = sorted(train_scores)[int(0.95 * len(train_scores))]

    normal_novel = sum(1 for t in test_normal if knn_novelty_score(t, train) > threshold_novelty)
    novel_detected = sum(1 for t in test_novel if knn_novelty_score(t, train) > threshold_novelty)

    print(f"  Тренировка: {len(train)} точек, k=5")
    print(f"  Порог (95-й перцентиль): {threshold_novelty:.4f}")
    print(f"  Нормальные тестовые: обнаружено новинок = {normal_novel}/{len(test_normal)}")
    print(f"  Новые тестовые:      обнаружено новинок = {novel_detected}/{len(test_novel)}")


# ──────────────────────────── 2. Quality Estimation ────────────────────────────

def demo_quality_estimation():
    """Оценка качества выходов, самопроверка, тесты согласованности."""
    print("\n\n" + "=" * 70)
    print("DEMO 2 — Quality Estimation")
    print("=" * 70)

    # --- 2a. Простая оценка качества ответа LLM ---
    print("\n--- 2a. Скоринг качества текстового ответа ---")
    def score_response(response, criteria):
        """Оценивает ответ по нескольким критериям (0-1 каждый)."""
        scores = {}
        text = response.lower()

        # Критерий 1: Длина (слишком короткий или длинный — плохо)
        word_count = len(text.split())
        if 20 <= word_count <= 200:
            scores["Длина"] = 1.0
        elif 10 <= word_count <= 300:
            scores["Длина"] = 0.7
        else:
            scores["Длина"] = 0.3

        # Критерий 2: Наличие структуры (абзацы, списки)
        has_structure = "\n" in response or any(c in response for c in ["- ", "1.", "2."])
        scores["Структура"] = 1.0 if has_structure else 0.3

        # Критерий 3: Ключевые слова
        if criteria:
            found = sum(1 for kw in criteria if kw.lower() in text)
            scores["Релевантность"] = found / len(criteria)
        else:
            scores["Релевантность"] = 0.5

        # Критерий 4: Отсутствие повторов (quality signal)
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        scores["Уникальность"] = min(unique_ratio * 1.5, 1.0)

        return scores

    response = """Машинное обучение — это область искусственного интеллекта,
которая позволяет компьютерам учиться на данных. Основные виды:
- Обучение с учителем (supervised learning)
- Обучение без учителя (unsupervised learning)
- Обучение с подкреплением (reinforcement learning)"""

    criteria = ["машинное обучение", "данные", "supervised", "reinforcement"]
    scores = score_response(response, criteria)

    print(f"  Ответ ({len(response.split())} слов):")
    for line in response.strip().split("\n")[:3]:
        print(f"    {line.strip()}")
    print(f"\n  Оценки по критериям:")
    total = 0
    for name, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {name:<16} {bar} {score:.2f}")
        total += score
    avg = total / len(scores)
    print(f"    {'Среднее':<16} {'█' * int(avg * 20)}{'░' * (20 - int(avg * 20))} {avg:.2f}")

    # --- 2b. Самопроверка через согласованность ---
    print("\n--- 2b. Тест согласованности (consistency check) ---")
    # Имитация: модель отвечает несколько раз на тот же вопрос
    random.seed(42)
    answers_per_question = 5
    question = "Сколько континентов на Земле?"
    # "Правильный" ответ: 7
    simulated_answers = []
    for _ in range(answers_per_question):
        # Иногда модель ошибается
        if random.random() < 0.15:
            simulated_answers.append(random.choice([5, 6, 8, 4]))
        else:
            simulated_answers.append(7)

    counts = collections.Counter(simulated_answers)
    most_common = counts.most_common(1)[0]
    consistency = most_common[1] / len(simulated_answers)

    print(f"  Вопрос: {question}")
    print(f"  Ответы {answers_per_question} запусков: {simulated_answers}")
    print(f"  Распределение: {dict(counts)}")
    print(f"  Наиболее частый ответ: {most_common[0]} ({most_common[1]}/{len(simulated_answers)})")
    print(f"  Согласованность: {consistency:.0%}")
    print(f"  → {'Высокая' if consistency > 0.8 else 'Низкая'} уверенность в ответе")

    # --- 2c. Self-evaluation через вероятности ---
    print("\n--- 2c. Самооценка через распределение вероятностей ---")
    # Модель выдаёт распределение вероятностей по классам
    class_probabilities = [0.65, 0.15, 0.10, 0.05, 0.05]
    class_names = ["Класс A", "Класс B", "Класс C", "Класс D", "Класс E"]

    entropy = -sum(p * math.log2(p) for p in class_probabilities if p > 0)
    max_entropy = math.log2(len(class_probabilities))
    confidence = 1 - entropy / max_entropy

    print(f"  Распределение вероятностей по классам:")
    for name, prob in zip(class_names, class_probabilities):
        bar = "█" * int(prob * 30)
        print(f"    {name}: {bar} {prob:.2f}")
    print(f"  Энтропия: {entropy:.3f} / {max_entropy:.3f} (макс.)")
    print(f"  Уверенность: {confidence:.2%}")
    print(f"  → Ответ: {class_names[0]} с {class_probabilities[0]:.0%} уверенностью")

    # --- 2d. Проверка детерминированности ---
    print("\n--- 2d. Проверка детерминированности ---")
    # При детерминированном выводе одинаковый input → одинаковый output
    def deterministic_hash(text, seed=42):
        """Детерминированное хэширование (имитация детерминированного вывода)."""
        random.seed(seed)
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def nondeterministic_hash(text):
        """Недетерминированный хэш (с случайным seed)."""
        return hashlib.md5(text.encode() + str(random.random()).encode()).hexdigest()[:8]

    test_input = "Пример запроса к модели"
    det_results = [deterministic_hash(test_input) for _ in range(5)]
    nondet_results = [nondeterministic_hash(test_input) for _ in range(5)]

    det_unique = len(set(det_results))
    nondet_unique = len(set(nondet_results))

    print(f"  Вход: '{test_input}'")
    print(f"  Детерминированный вывод (seed=42): {det_results[:3]}...")
    print(f"    Уникальных результатов: {det_unique}/5 → {'ДЕТЕРМИНИРОВАН' if det_unique == 1 else 'НЕДЕТЕРМИНИРОВАН'}")
    print(f"  Недетерминированный вывод: {nondet_results[:3]}...")
    print(f"    Уникальных результатов: {nondet_unique}/5 → {'ДЕТЕРМИНИРОВАН' if nondet_unique == 1 else 'НЕДЕТЕРМИНИРОВАН'}")


# ──────────────────────────── 3. Confidence Calibration ────────────────────────────

def demo_confidence_calibration():
    """Temperature scaling, reliability diagrams, ECE."""
    print("\n\n" + "=" * 70)
    print("DEMO 3 — Confidence Calibration")
    print("=" * 70)

    # --- 3a. Генерация калибровочных данных ---
    print("\n--- 3a. Данные: predicted confidence vs actual accuracy ---")
    random.seed(42)
    n_samples = 500
    # Модель переоценена: предсказывает высокую уверенность, но точность ниже
    predicted_confidence = []
    actual_correct = []
    for _ in range(n_samples):
        conf = random.uniform(0.3, 0.99)
        # Точность зависит от уверенности, но с noise и сдвигом
        accuracy_prob = conf * 0.75 + random.gauss(0, 0.1)
        accuracy_prob = max(0, min(1, accuracy_prob))
        predicted_confidence.append(conf)
        actual_correct.append(1 if random.random() < accuracy_prob else 0)

    # --- 3b. Binning для reliability diagram ---
    print("\n--- 3b. Reliability Diagram (бины по уверенности) ---")
    n_bins = 10
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = [(conf, correct) for conf, correct in
                  zip(predicted_confidence, actual_correct)
                  if low <= conf < high]
        if in_bin:
            avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
            avg_acc = sum(a for _, a in in_bin) / len(in_bin)
            bin_centers.append(avg_conf)
            bin_accuracies.append(avg_acc)
            bin_counts.append(len(in_bin))
        else:
            bin_centers.append((low + high) / 2)
            bin_accuracies.append(0)
            bin_counts.append(0)

    print(f"  {'Bin':<12}{'Уверенность':>14}{'Точность':>12}{'Кол-во':>10}{'Разница':>12}")
    for i in range(n_bins):
        if bin_counts[i] > 0:
            diff = bin_centers[i] - bin_accuracies[i]
            print(f"  [{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]"
                  f"{bin_centers[i]:>14.3f}{bin_accuracies[i]:>12.3f}"
                  f"{bin_counts[i]:>10}{diff:>+12.3f}")

    # --- 3c. Expected Calibration Error (ECE) ---
    print("\n--- 3c. Expected Calibration Error (ECE) ---")
    total_samples = sum(bin_counts)
    ece = 0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * abs(bin_centers[i] - bin_accuracies[i])

    print(f"  ECE = Σ (|confidence - accuracy| × bin_proportion)")
    print(f"  ECE = {ece:.4f}")
    print(f"  → {'Хорошая' if ece < 0.05 else 'Плохая'} калибровка (ECE {'< 0.05' if ece < 0.05 else '>= 0.05'})")

    # --- 3d. Temperature Scaling ---
    print("\n--- 3d. Temperature Scaling (калибровка) ---")
    # Temperature scaling: softmax(logits / T)
    # При T > 1 распределение становится более "мягким" (менее уверенным)
    print("  Temperature Scaling: p_i = exp(logit_i / T) / Σ exp(logit_j / T)")

    # Имитация logits для одного примера
    raw_logits = [2.1, 0.5, -0.3, -1.0]

    def softmax(logits, temperature=1.0):
        """Вычисляет softmax с temperature scaling."""
        scaled = [l / temperature for l in logits]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        sum_exp = sum(exps)
        return [e / sum_exp for e in exps]

    temperatures = [0.5, 1.0, 2.0, 5.0]
    print(f"\n  Логиты: {raw_logits}")
    print(f"\n  {'Temperature':<14}", end="")
    for i in range(len(raw_logits)):
        print(f"  {'P(class ' + str(i) + ')':>14}", end="")
    print(f"  {'Энтропия':>12}")

    for T in temperatures:
        probs = softmax(raw_logits, T)
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        print(f"  T={T:<12.1f}", end="")
        for p in probs:
            print(f"{p:>14.4f}", end="")
        print(f"{entropy:>12.4f}")

    print(f"\n  → T=1.0: исходное распределение (модель переоценена)")
    print(f"  → T>1.0: сглаженное (более калиброванное)")
    print(f"  → T<1.0: более острое (ещё менее калиброванное)")

    # Вычисляем оптимальную T через минимизацию NLL
    print("\n  Оптимизация T через Negative Log-Likelihood:")
    best_t = 1.0
    best_nll = float('inf')

    for t_int in range(10, 500):
        t = t_int / 100.0
        nll = 0
        for conf, correct in zip(predicted_confidence, actual_correct):
            # Конвертируем信心 в "logits" и считаем NLL
            if conf > 0.5:
                logit = math.log(conf / (1 - conf))
            else:
                logit = -2.0
            probs = softmax([logit, -logit], t)
            p_correct = probs[0] if correct == 1 else probs[1]
            nll -= math.log(max(p_correct, 1e-10))
        nll /= n_samples
        if nll < best_nll:
            best_nll = nll
            best_t = t

    print(f"  Оптимальная температура: T = {best_t:.2f}")
    print(f"  NLL при T=1.0:   {sum(-math.log(max(softmax([math.log(p/(1-p+1e-10)) if p > 0.01 else -2, -2], 1.0)[0 if c == 1 else 1], 1e-10)) for p, c in zip(predicted_confidence, actual_correct)) / n_samples:.4f}")
    print(f"  NLL при T={best_t:.2f}: {best_nll:.4f}")


# ──────────────────────────── 4. Self-Diagnostics ────────────────────────────

def demo_self_diagnostics():
    """Health checks, отслеживание деградации, алерты."""
    print("\n\n" + "=" * 70)
    print("DEMO 4 — Self-Diagnostics")
    print("=" * 70)

    # --- 4a. Health check системы ---
    print("\n--- 4a. Health Check ---")
    def health_check(services):
        """Проверяет состояние сервисов."""
        results = {}
        for name, check_fn in services.items():
            try:
                results[name] = ("OK", check_fn())
            except Exception as e:
                results[name] = ("FAIL", str(e))
        return results

    # Имитация проверок
    services = {
        "CPU": lambda: f"{random.uniform(20, 80):.1f}%",
        "RAM": lambda: f"{random.uniform(40, 90):.1f}%",
        "Disk": lambda: f"{random.uniform(30, 70):.1f}%",
        "Network": lambda: "Доступен",
        "Database": lambda: "Подключена",
    }

    results = health_check(services)
    for service, (status, detail) in results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {service:<12} {detail}")

    # --- 4b. Отслеживание метрик во времени ---
    print("\n--- 4b. Отслеживание метрик (Performance Tracking) ---")
    # Генерируем временной ряд метрик с постепенной деградацией
    random.seed(42)
    n_points = 50
    latency_values = []
    error_rates = []
    for i in range(n_points):
        # Латентность постепенно растёт
        base_latency = 50 + i * 0.5
        latency = base_latency + random.gauss(0, 5)
        latency_values.append(latency)

        # Error rate медленно увеличивается
        base_error = 0.01 + i * 0.0005
        error_rate = max(0, base_error + random.gauss(0, 0.005))
        error_rates.append(error_rate)

    # Скользящее среднее для обнаружения тренда
    window = 10
    smoothed_latency = []
    for i in range(len(latency_values)):
        start = max(0, i - window + 1)
        avg = sum(latency_values[start:i + 1]) / (i - start + 1)
        smoothed_latency.append(avg)

    print(f"  Мониторинг {n_points} замеров, окно скользящего среднего = {window}")
    print(f"\n  {'Замер':<10}{'Латентность':>14}{'Сглаженная':>14}{'Error Rate':>14}")
    for i in [0, 10, 20, 30, 40, 49]:
        print(f"  #{i:<9}{latency_values[i]:>13.1f}ms{smoothed_latency[i]:>13.1f}ms"
              f"{error_rates[i] * 100:>13.2f}%")

    # --- 4c. Обнаружение деградации ---
    print("\n--- 4c. Обнаружение деградации (Degradation Alert) ---")
    # Сравниваем последний и первый скользящие средние
    early_avg = sum(smoothed_latency[:10]) / 10
    late_avg = sum(smoothed_latency[-10:]) / 10
    degradation_pct = (late_avg - early_avg) / early_avg * 100

    print(f"  Средняя латентность (первые 10 замеров): {early_avg:.1f}ms")
    print(f"  Средняя латентность (последние 10 замеров): {late_avg:.1f}ms")
    print(f"  Деградация: {degradation_pct:+.1f}%")

    # Пороги алертов
    alert_thresholds = {
        "Info": 10,
        "Warning": 25,
        "Critical": 50,
    }

    alert_level = "OK"
    for level, threshold in sorted(alert_thresholds.items(), key=lambda x: x[1]):
        if degradation_pct > threshold:
            alert_level = level

    print(f"  Пороги: Info > {alert_thresholds['Info']}%, "
          f"Warning > {alert_thresholds['Warning']}%, "
          f"Critical > {alert_thresholds['Critical']}%")
    print(f"  → Уровень алерта: {alert_level}")

    # --- 4d. Автоматические рекомендации ---
    print("\n--- 4d. Автоматические рекомендации ---")
    def generate_recommendations(metrics):
        """Генерирует рекомендации на основе метрик."""
        recs = []
        if metrics["avg_latency"] > 100:
            recs.append("Увеличить таймауты или оптимизировать запросы к БД")
        if metrics["error_rate"] > 0.05:
            recs.append("Проверить логи ошибок, возможно перезапустить сервис")
        if metrics["latency_trend"] > 0.5:
            recs.append("Возможна утечка памяти — проверить heap dump")
        if metrics["cpu_usage"] > 80:
            recs.append("Масштабировать horizontally (добавить инстансы)")
        if not recs:
            recs.append("Система работает нормально, рекомендаций нет")
        return recs

    current_metrics = {
        "avg_latency": late_avg,
        "error_rate": error_rates[-1],
        "latency_trend": degradation_pct / 100,
        "cpu_usage": random.uniform(60, 95),
    }

    print(f"\n  Текущие метрики:")
    for k, v in current_metrics.items():
        print(f"    {k}: {v:.3f}")

    recommendations = generate_recommendations(current_metrics)
    print(f"\n  Рекомендации:")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec}")

    # Метрика состояния здоровья
    health_score = max(0, 100 - degradation_pct - current_metrics["error_rate"] * 1000)
    print(f"\n  Health Score: {health_score:.1f}/100")
    if health_score > 80:
        print("  Состояние: ОТЛИЧНО")
    elif health_score > 60:
        print("  Состояние: УДОВЛЕТВОРИТЕЛЬНО")
    else:
        print("  Состояние: ТРЕБУЕТ ВНИМАНИЯ")


if __name__ == "__main__":
    demo_anomaly_detection()
    demo_quality_estimation()
    demo_confidence_calibration()
    demo_self_diagnostics()
