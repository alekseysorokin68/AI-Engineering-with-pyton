"""
222 — Robustness & Adversarial ML: атаки, защиты, устойчивость моделей

Темы:
  1. Adversarial Attacks (FGSM, PGD, C&W concepts, perturbation budgets)
  2. Attack Types (evasion, poisoning, model extraction, data inference)
  3. Defenses (adversarial training, input validation, certified robustness)
  4. Robustness Evaluation (robust accuracy, worst-case performance, certification)

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
# Вспомогательные функции для демонстрации атак
# ============================================================

def sigmoid(x):
    """Сигмоидальная функция активации."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def softmax(scores):
    """Вычисление softmax по вектору оценок."""
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def dot(a, b):
    """Скалярное произведение двух векторов."""
    return sum(ai * bi for ai, bi in zip(a, b))


def l2_norm(v):
    """L2-норма вектора."""
    return math.sqrt(sum(x * x for x in v))


def l_inf_norm(v):
    """L∞-норма вектора (максимальный элемент по модулю)."""
    return max(abs(x) for x in v)


def cross_entropy_loss(probs, target_idx):
    """Вычисление кросс-энтропийного лосса для одного примера."""
    p = max(probs[target_idx], 1e-12)
    return -math.log(p)


# ============================================================
# Демо 1: Adversarial Attacks — FGSM, PGD, C&W concepts
# ============================================================

def demo1_adversarial_attacks():
    """Демонстрация основных методов атак на модели."""
    print("=" * 70)
    print("Демо 1: Adversarial Attacks — FGSM, PGD, C&W")
    print("=" * 70)

    # --- 1.1 Простая модель: линейный классификатор ---
    # Зададим случайные веса и данные
    random.seed(42)
    input_dim = 6
    num_classes = 3

    # Генерация весов модели
    weights = [[random.gauss(0, 0.5) for _ in range(input_dim)] for _ in range(num_classes)]
    biases = [random.gauss(0, 0.1) for _ in range(num_classes)]

    def predict(x):
        """Предсказание классификатора."""
        scores = [dot(w, x) + b for w, b in zip(weights, biases)]
        return softmax(scores)

    # Тестовый вход
    x_clean = [random.gauss(0, 1) for _ in range(input_dim)]
    probs_clean = predict(x_clean)
    true_class = probs_clean.index(max(probs_clean))
    print(f"\n--- 1.1 Чистый вход ---")
    print(f"Вход: {[round(v, 4) for v in x_clean]}")
    print(f"Вероятности классов: {[round(p, 4) for p in probs_clean]}")
    print(f"Предсказанный класс: {true_class} (уверенность: {probs_clean[true_class]:.4f})")

    # --- 1.2 FGSM: Fast Gradient Sign Method ---
    # grad_loss_x ≈ sign(x) для кросс-энтропии → ищем направление увеличения лосса
    # FGSM: x_adv = x + eps * sign(grad)
    print(f"\n--- 1.2 FGSM (Fast Gradient Sign Method) ---")
    print("Формула: x_adv = x + ε · sign(∇_x L(f(x), y))")

    epsilon = 0.3  # Параметр возмущения (budget)

    # Аппроксимация градиента для линейной модели:
    # grad = weights[true_class] (упрощённо)
    grad_approx = weights[true_class][:]  # копия градиента

    # FGSM знаковый шаг
    perturbation_fgsm = []
    for g in grad_approx:
        if g > 0:
            perturbation_fgsm.append(epsilon)
        elif g < 0:
            perturbation_fgsm.append(-epsilon)
        else:
            perturbation_fgsm.append(0.0)

    x_adv_fgsm = [x_clean[i] + perturbation_fgsm[i] for i in range(input_dim)]
    probs_fgsm = predict(x_adv_fgsm)
    adv_class_fgsm = probs_fgsm.index(max(probs_fgsm))

    print(f"ε (параметр возмущения): {epsilon}")
    print(f"Возмущение (sign-вектор): {[round(p, 3) for p in perturbation_fgsm]}")
    print(f"L∞-норма возмущения: {l_inf_norm(perturbation_fgsm):.4f}")
    print(f"L2-норма возмущения: {l2_norm(perturbation_fgsm):.4f}")
    print(f"Атакованный вход: {[round(v, 4) for v in x_adv_fgsm]}")
    print(f"Вероятности после FGSM: {[round(p, 4) for p in probs_fgsm]}")
    print(f"Новое предсказание: {adv_class_fgsm} (уверенность: {probs_fgsm[adv_class_fgsm]:.4f})")
    success_fgsm = adv_class_fgsm != true_class
    print(f"Атака успешна: {success_fgsm}")

    # --- 1.3 PGD: Projected Gradient Descent (многошаговая атака) ---
    print(f"\n--- 1.3 PGD (Projected Gradient Descent) ---")
    print("Формула: x_(t+1) = Proj_B(x_t + α · sign(∇_x L))")

    pgd_steps = 10
    pgd_alpha = 0.05  # Шаг атаки на каждой итерации

    x_pgd = list(x_clean)  # Копия чистого входа
    best_adv = list(x_pgd)
    best_loss = cross_entropy_loss(predict(x_pgd), true_class)

    for step in range(pgd_steps):
        # Вычисляем градиент (упрощённая аппроксимация)
        probs_current = predict(x_pgd)
        grad_step = []
        for d in range(input_dim):
            # Численный градиент: ∂L/∂x[d]
            delta = 0.01
            x_plus = list(x_pgd)
            x_plus[d] += delta
            loss_plus = cross_entropy_loss(predict(x_plus), true_class)
            x_minus = list(x_pgd)
            x_minus[d] -= delta
            loss_minus = cross_entropy_loss(predict(x_minus), true_class)
            grad_step.append((loss_plus - loss_minus) / (2 * delta))

        # PGD шаг: знаковый градиент + проекция в ε-шар
        pgd_pert = [pgd_alpha * (1.0 if g > 0 else (-1.0 if g < 0 else 0.0)) for g in grad_step]
        x_pgd = [x_pgd[i] + pgd_pert[i] for i in range(input_dim)]

        # Проекция: ограничиваем L∞-расстояние от оригинала
        for i in range(input_dim):
            x_pgd[i] = max(x_clean[i] - epsilon, min(x_clean[i] + epsilon, x_pgd[i]))

        current_loss = cross_entropy_loss(predict(x_pgd), true_class)
        if current_loss > best_loss:
            best_loss = current_loss
            best_adv = list(x_pgd)

    probs_pgd = predict(best_adv)
    adv_class_pgd = probs_pgd.index(max(probs_pgd))

    print(f"Количество шагов: {pgd_steps}, α (шаг): {pgd_alpha}")
    print(f"Начальный лосс: {cross_entropy_loss(predict(x_clean), true_class):.4f}")
    print(f"Лучший лосс после PGD: {best_loss:.4f}")
    print(f"Вероятности после PGD: {[round(p, 4) for p in probs_pgd]}")
    print(f"L∞-расстояние от оригинала: {l_inf_norm([best_adv[i] - x_clean[i] for i in range(input_dim)]):.4f}")
    print(f"Новое предсказание: {adv_class_pgd} (уверенность: {probs_pgd[adv_class_pgd]:.4f})")
    success_pgd = adv_class_pgd != true_class
    print(f"Атака успешна: {success_pgd}")

    # --- 1.4 C&W:概念 — Carlini & Wagner оптимизация ---
    print(f"\n--- 1.4 C&W (Carlini & Wagner) — концепция ---")
    print("C&W минимизирует: ||δ||₂ + c · max(max_{i≠t} Z(x+δ)[i] - Z(x+δ)[t], -κ)")
    print("Где δ — возмущение, Z — логиты, t — целевой класс, κ — confidence margin")

    # Демонстрация C&W на простом примере: поиск минимального возмущения
    # для переклассификации с целевым классом
    target_class = (true_class + 1) % num_classes  # Целевой класс ≠ истинного

    best_pert_cw = None
    best_l2_cw = float('inf')

    # Простой перебор масштабов возмущения (упрощённая C&W)
    for scale in [i * 0.01 for i in range(1, 50)]:
        # Создаём возмущение в направлении, увеличивающем оценку целевого класса
        direction = []
        for d in range(input_dim):
            # Градиент цели минус градиент истинного класса
            delta = 0.001
            x_test = list(x_clean)

            # O_t(x)
            x_test[d] += delta
            z_t_plus = dot(weights[target_class], x_test) + biases[target_class]
            x_test[d] -= 2 * delta
            z_t_minus = dot(weights[target_class], x_test) + biases[target_class]
            grad_target = (z_t_plus - z_t_minus) / (2 * delta)

            # O_y(x)
            x_test[d] = x_clean[d] + delta
            z_y_plus = dot(weights[true_class], x_test) + biases[true_class]
            x_test[d] = x_clean[d] - delta
            z_y_minus = dot(weights[true_class], x_test) + biases[true_class]
            grad_true = (z_y_plus - z_y_minus) / (2 * delta)

            direction.append(grad_target - grad_true)

        # Нормализация направления
        norm_dir = l2_norm(direction)
        if norm_dir < 1e-12:
            continue
        normed = [d / norm_dir for d in direction]
        pert = [normed[i] * scale for i in range(input_dim)]
        x_test_cw = [x_clean[i] + pert[i] for i in range(input_dim)]
        probs_test = predict(x_test_cw)

        # Проверяем: атакованный пример классифицируется как target_class
        # и confidence margin достаточен
        if probs_test.index(max(probs_test)) == target_class:
            l2_dist = l2_norm(pert)
            if l2_dist < best_l2_cw:
                best_l2_cw = l2_dist
                best_pert_cw = list(pert)

    if best_pert_cw is not None:
        x_cw = [x_clean[i] + best_pert_cw[i] for i in range(input_dim)]
        probs_cw = predict(x_cw)
        print(f"\nЦелевой класс атаки: {target_class}")
        print(f"Минимальная L2-норма возмущения: {best_l2_cw:.4f}")
        print(f"Вероятности после C&W: {[round(p, 4) for p in probs_cw]}")
        print(f"Предсказание: {probs_cw.index(max(probs_cw))} (цель: {target_class})")
        print(f"Уверенность в целевом классе: {probs_cw[target_class]:.4f}")
        print(f"\nКлючевое отличие C&W от FGSM:")
        print(f"  FGSM:    одношаговая, фиксированный ε → быстрая, но неточные")
        print(f"  PGD:     итеративная, проекция → надёжнее")
        print(f"  C&W:     оптимизация L2 → минимальное возмущение, целенаправленная")
    else:
        print(f"Целевой класс: {target_class} — переклассификация не найдена в пределах 0.5")

    print()


# ============================================================
# Демо 2: Attack Types — evasion, poisoning, extraction, inference
# ============================================================

def demo2_attack_types():
    """Демонстрация различных типов атак на ML-системы."""
    print("=" * 70)
    print("Демо 2: Attack Types — типы атак на ML-системы")
    print("=" * 70)

    # --- 2.1 Evasion Attack: обход детектора спама ---
    print(f"\n--- 2.1 Evasion Attack: обход спам-фильтра ---")
    print("Evasion: модификация входных данных для обхода модели на инференсе")

    # Простой спам-фильтр на основе ключевых слов
    spam_keywords = {"buy", "free", "winner", "cash", "prize", "offer", "deal", "discount"}
    ham_keywords = {"meeting", "project", "report", "deadline", "team", "schedule", "review"}

    def spam_score(text):
        """Простая функция оценки спама на основе ключевых слов."""
        words = set(re.findall(r'[a-z]+', text.lower()))
        s = len(words & spam_keywords)
        h = len(words & ham_keywords)
        return s - h  # положительный → спам

    def classify_spam(score, threshold=1):
        """Классификация по порогу."""
        return "SPAM" if score >= threshold else "HAM"

    # Оригинальное спам-сообщение
    original_msg = "Buy now! Free prize winner cash offer deal"
    score_orig = spam_score(original_msg)
    class_orig = classify_spam(score_orig)

    print(f"\nОригинальное сообщение: \"{original_msg}\"")
    print(f"Оценка спама: {score_orig}, класс: {class_orig}")

    # Evasion: замена символов для обхода фильтра
    evasion_msgs = [
        "Buy now! Free prize winner cash offer deal",           # оригинал
        "Büy n0w! Fr33 pr1z3 w1nn3r c4sh 0ff3r d34l",        # замена букв на цифры
        "BuY nOw! FrEe PrIzE WiNnEr CaSh OfFeR dEaL",       # чередование регистра
        "B.u.y n.o.w! F.r.e.e p.r.i.z.e o.f.f.e.r",          # разделение точками
    ]

    print(f"\nEvasion-атаки (замена символов):")
    print(f"{'Сообщение':<50} {'Оценка':>6} {'Класс':>6}")
    print("-" * 65)
    for msg in evasion_msgs:
        s = spam_score(msg)
        c = classify_spam(s)
        display = msg[:48] + ".." if len(msg) > 50 else msg
        print(f"{display:<50} {s:>6} {c:>6}")

    print(f"\nВывод: простые замены символов НЕ обходят фильтр")
    print(f"  → Нужны устойчивые к evasion модели ( embeddings, context-aware )")

    # --- 2.2 Poisoning Attack: отравление данных обучения ---
    print(f"\n--- 2.2 Poisoning Attack: отравление данных обучения ---")
    print("Poisoning: внедрение отравленных данных в обучающую выборку")

    random.seed(42)

    # Чистые обучающие данные: (.features, label)
    clean_data = []
    for _ in range(50):
        # Класс 0: feature[0] ≈ 0.2
        f0 = random.gauss(0.2, 0.1)
        clean_data.append(([f0, random.gauss(0.5, 0.2)], 0))
    for _ in range(50):
        # Класс 1: feature[0] ≈ 0.8
        f0 = random.gauss(0.8, 0.1)
        clean_data.append(([f0, random.gauss(0.5, 0.2)], 1))

    # Простой KNN-классификатор
    def knn_classify(x, data, k=3):
        """KNN-классификация по k ближайшим соседям."""
        dists = [(math.sqrt(sum((x[i] - d[0][i]) ** 2 for i in range(len(x)))), d[1]) for d in data]
        dists.sort(key=lambda t: t[0])
        neighbors = [d[1] for d in dists[:k]]
        return 1 if sum(neighbors) > k / 2 else 0

    # Тестовые данные
    test_x = [0.5, 0.5]  # На границе决策面
    print(f"\nТестовая точка: {test_x}")
    print(f"KNN (k=3) на чистых данных: класс {knn_classify(test_x, clean_data)}")

    # Отравление: добавляем 5 отравленных примеров
    poisoned_data = list(clean_data)
    poison_count = 0
    for _ in range(5):
        # Отравленные точки: feature[0] ≈ 0.5, label = 1 (ложная метка)
        f0 = random.gauss(0.5, 0.05)
        poisoned_data.append(([f0, random.gauss(0.5, 0.1)], 1))
        poison_count += 1

    print(f"Добавлено отравленных примеров: {poison_count}")
    print(f"Отравленные точки: feature[0] ≈ 0.5, label = 1 (ложная метка)")

    # Новая граница решения
    test_points = [[0.3 + i * 0.05, 0.5] for i in range(9)]
    print(f"\nГраница решения (KNN на чистых vs отравленных данных):")
    print(f"{'Точка':>10} {'Чистые':>8} {'Отравл.':>8}")
    print("-" * 30)
    for tp in test_points:
        c_clean = knn_classify(tp, clean_data)
        c_poison = knn_classify(tp, poisoned_data)
        print(f"  {tp[0]:.2f}    {c_clean:>8} {c_poison:>8}")

    print(f"\nВывод: отравленные данные смещают границу решения")
    print(f"  → Backdoor attacks: внедряют скрытое поведение")

    # --- 2.3 Model Extraction: извлечение модели ---
    print(f"\n--- 2.3 Model Extraction: извлечение модели ---")
    print("Model Extraction: копирование модели через API-query")

    # Целевая модель: простая функция
    def target_model(x):
        """Секретная целевая модель: нелинейная функция."""
        return math.sin(x[0] * 3.14) * 0.5 + x[1] * 0.3 + 0.2

    # Генерация запросов к API (чёрный ящик)
    random.seed(42)
    api_queries = []
    api_responses = []
    num_queries = 20

    for _ in range(num_queries):
        x_q = [random.uniform(-1, 1), random.uniform(-1, 1)]
        y_q = target_model(x_q)
        api_queries.append(x_q)
        api_responses.append(y_q)

    print(f"\nЦелевая модель: f(x) = sin(x₁·π)·0.5 + x₂·0.3 + 0.2")
    print(f"Запросов к API: {num_queries}")

    # Суррогатная модель: обучаем линейную регрессию
    # y = w₁x₁ + w₂x₂ + b
    # Аппроксимация методом наименьших квадратов
    # Средние значения
    mean_x1 = sum(q[0] for q in api_queries) / num_queries
    mean_x2 = sum(q[1] for q in api_queries) / num_queries
    mean_y = sum(api_responses) / num_queries

    # Вычисление коэффициентов (упрощённая формула)
    num_w1 = sum((q[0] - mean_x1) * (api_responses[i] - mean_y) for i, q in enumerate(api_queries))
    den_w1 = sum((q[0] - mean_x1) ** 2 for q in api_queries)
    num_w2 = sum((q[1] - mean_x2) * (api_responses[i] - mean_y) for i, q in enumerate(api_queries))
    den_w2 = sum((q[1] - mean_x2) ** 2 for q in api_queries)

    w1 = num_w1 / den_w1 if den_w1 > 0 else 0
    w2 = num_w2 / den_w2 if den_w2 > 0 else 0
    b = mean_y - w1 * mean_x1 - w2 * mean_x2

    print(f"\nОбученная суррогатная модель: y ≈ {w1:.4f}·x₁ + {w2:.4f}·x₂ + {b:.4f}")
    print(f"Истинная модель:              y = 0.5·sin(x₁·π) + 0.3·x₂ + 0.2")

    # Оценка качества извлечения
    total_error = 0
    for i in range(num_queries):
        y_true = api_responses[i]
        y_surrogate = w1 * api_queries[i][0] + w2 * api_queries[i][1] + b
        total_error += (y_true - y_surrogate) ** 2
    mse = total_error / num_queries

    print(f"MSE между целевой и суррогатной моделью: {mse:.4f}")
    print(f"Предсказание на (0.5, 0.5): целевая={target_model([0.5, 0.5]):.4f}, "
          f"суррогат={w1 * 0.5 + w2 * 0.5 + b:.4f}")

    print(f"\nВывод: линейная модель плохо аппроксимирует нелинейную цель")
    print(f"  → Защита: rate limiting, watermarking, query detection")

    # --- 2.4 Data Inference Attack: утечка информации из данных ---
    print(f"\n--- 2.4 Data Inference Attack: утечка конфиденциальных данных ---")
    print("Inference: восстановление приватной информации по модели/результатам")

    # Модель, обученная на медицинских данных
    # Восстанавливаем «приватные» атрибуты по агрегатным статистикам
    random.seed(42)

    # Симуляция базы данных пациентов
    patient_records = []
    ages = []
    for i in range(100):
        age = random.randint(20, 80)
        bmi = random.gauss(25, 5)
        blood_pressure = random.gauss(120, 15) + (age - 40) * 0.5
        cholesterol = random.gauss(200, 30) + (age - 40) * 1.0
        has_diabetes = 1 if (age > 50 and bmi > 28) or random.random() < 0.1 else 0
        patient_records.append({
            'id': i,
            'age': age,
            'bmi': round(bmi, 1),
            'bp': round(blood_pressure, 1),
            'chol': round(cholesterol, 1),
            'diabetes': has_diabetes
        })
        ages.append(age)

    # Модель доступна как чёрный ящик: принимает (bmi, bp, chol), возвращает вероятность диабета
    def diabetes_model(bmi, bp, chol):
        """Симуляция модели для предсказания диабета."""
        score = (bmi - 25) * 0.02 + (bp - 120) * 0.005 + (chol - 200) * 0.003
        return sigmoid(score * 5)

    # Атака: извлечение распределения возраста через membership inference
    # Идея: если модель «помнит» пример, она увереннее
    print(f"\nMembership Inference Attack (концепция):")
    print(f"Вопрос: принадлежит ли запись X обучающей выборке?")

    # Простая эвристика: высокая уверенность → возможно, из обучающей выборки
    random.seed(42)
    member_confidences = []
    non_member_confidences = []

    for p in patient_records[:20]:
        conf = diabetes_model(p['bmi'], p['bp'], p['chol'])
        member_confidences.append(conf)

    # «Чужие» записи (не в обучающей выборке)
    for _ in range(20):
        bmi_fake = random.gauss(25, 6)
        bp_fake = random.gauss(125, 18)
        chol_fake = random.gauss(195, 35)
        conf = diabetes_model(bmi_fake, bp_fake, chol_fake)
        non_member_confidences.append(conf)

    avg_member = sum(member_confidences) / len(member_confidences)
    avg_non_member = sum(non_member_confidences) / len(non_member_confidences)

    print(f"Средняя уверенность для членов обучающей выборки: {avg_member:.4f}")
    print(f"Средняя уверенность для не-членов:               {avg_non_member:.4f}")
    print(f"Разница: {abs(avg_member - avg_non_member):.4f}")

    # Атака восстановления возраста по агрегатным статистикам
    print(f"\nAttribute Inference Attack (концепция):")
    # Группировка по диабету и вычисление средних характеристик
    diabetic = [p for p in patient_records if p['diabetes'] == 1]
    non_diabetic = [p for p in patient_records if p['diabetes'] == 0]

    avg_age_diab = sum(p['age'] for p in diabetic) / len(diabetic)
    avg_age_non = sum(p['age'] for p in non_diabetic) / len(non_diabetic)

    print(f"Средний возраст диабетиков: {avg_age_diab:.1f} лет (n={len(diabetic)})")
    print(f"Средний возраст без диабета: {avg_age_non:.1f} лет (n={len(non_diabetic)})")
    print(f"Разница: {avg_age_diab - avg_age_non:.1f} лет")
    print(f"\nВывод: даже агрегатные данные могут раскрыть приватность")
    print(f"  → Differential Privacy, Federated Learning для защиты")

    print()


# ============================================================
# Демо 3: Defenses — adversarial training, input validation, certified robustness
# ============================================================

def demo3_defenses():
    """Демонстрация методов защиты от атак."""
    print("=" * 70)
    print("Демо 3: Defenses — методы защиты от атак")
    print("=" * 70)

    # --- 3.1 Adversarial Training ---
    print(f"\n--- 3.1 Adversarial Training: обучение на атакованных примерах ---")
    print("Идея: добавить атакованные примеры в обучающую выборку")

    random.seed(42)

    # Простая модель: линейный классификатор
    def train_simple(x_data, y_data, lr=0.01, epochs=20):
        """Обучение простого линейного классификатора."""
        w = [0.0, 0.0]
        b = 0.0
        for epoch in range(epochs):
            for x, y in zip(x_data, y_data):
                pred = sigmoid(dot(w, x) + b)
                error = pred - y
                for i in range(len(w)):
                    w[i] -= lr * error * x[i]
                b -= lr * error
        return w, b

    def accuracy(x_data, y_data, w, b):
        """Вычисление точности."""
        correct = 0
        for x, y in zip(x_data, y_data):
            pred = 1 if sigmoid(dot(w, x) + b) >= 0.5 else 0
            if pred == y:
                correct += 1
        return correct / len(y_data)

    def fgsm_attack_simple(x, w, b, epsilon=0.3):
        """FGSM атака на простую модель."""
        pred = sigmoid(dot(w, x) + b)
        # Градиент: sign(x) · (pred - y)
        perturbation = []
        for xi in x:
            grad = xi * (pred - 1)  # Упрощённый градиент
            if grad > 0:
                perturbation.append(epsilon)
            elif grad < 0:
                perturbation.append(-epsilon)
            else:
                perturbation.append(0.0)
        return [x[i] + perturbation[i] for i in range(len(x))]

    # Чистые данные
    x_clean = []
    y_clean = []
    for _ in range(40):
        x_clean.append([random.gauss(0.2, 0.1), random.gauss(0.5, 0.2)])
        y_clean.append(0)
    for _ in range(40):
        x_clean.append([random.gauss(0.8, 0.1), random.gauss(0.5, 0.2)])
        y_clean.append(1)

    # Тестовые данные (чистые)
    x_test = []
    y_test = []
    for _ in range(20):
        x_test.append([random.gauss(0.2, 0.15), random.gauss(0.5, 0.25)])
        y_test.append(0)
    for _ in range(20):
        x_test.append([random.gauss(0.8, 0.15), random.gauss(0.5, 0.25)])
        y_test.append(1)

    # Тестовые данные (атакованные)
    epsilon = 0.3

    # Обычное обучение (без adversarial)
    w_clean, b_clean = train_simple(x_clean, y_clean)
    acc_clean = accuracy(x_test, y_test, w_clean, b_clean)

    # Атака на модель, обученную на чистых данных
    x_test_adv = [fgsm_attack_simple(x, w_clean, b_clean, epsilon) for x in x_test]
    acc_clean_adv = accuracy(x_test_adv, y_test, w_clean, b_clean)

    print(f"\nБез adversarial training:")
    print(f"  Точность на чистых данных:   {acc_clean:.4f}")
    print(f"  Точность на атакованных:     {acc_clean_adv:.4f}")
    print(f"  Потеря точности:             {acc_clean - acc_clean_adv:.4f}")

    # Adversarial training
    x_adv_train = [fgsm_attack_simple(x, w_clean, b_clean, epsilon) for x in x_clean]
    x_mixed = x_clean + x_adv_train
    y_mixed = y_clean + y_clean  # Метки остаются те же

    w_adv, b_adv = train_simple(x_mixed, y_mixed, lr=0.01, epochs=30)
    acc_adv = accuracy(x_test, y_test, w_adv, b_adv)

    # Атака на модель, обученную с adversarial training
    x_test_adv2 = [fgsm_attack_simple(x, w_adv, b_adv, epsilon) for x in x_test]
    acc_adv_adv = accuracy(x_test_adv2, y_test, w_adv, b_adv)

    print(f"\nС adversarial training:")
    print(f"  Точность на чистых данных:   {acc_adv:.4f}")
    print(f"  Точность на атакованных:     {acc_adv_adv:.4f}")
    print(f"  Потеря точности:             {acc_adv - acc_adv_adv:.4f}")

    print(f"\nУлучшение робастности: {acc_adv_adv - acc_clean_adv:+.4f}")
    print(f"  Adversarial training повышает устойчивость к FGSM-атакам")

    # --- 3.2 Input Validation ---
    print(f"\n--- 3.2 Input Validation: валидация входных данных ---")
    print("Проверка входных данных на аномалии и искажения")

    # Правила валидации
    validation_rules = {
        "pixel_range": {"min": 0.0, "max": 1.0},
        "max_perturbation": 0.1,
        "l_inf_bound": 0.3,
        "statistical_bound": 3.0,  # Стандартные отклонения
    }

    def validate_input(x_original, x_test_input, rules):
        """Валидация входных данных по набору правил."""
        violations = []

        # Проверка диапазона значений
        for i, val in enumerate(x_test_input):
            if val < rules["pixel_range"]["min"] or val > rules["pixel_range"]["max"]:
                violations.append(f"Элемент {i}: значение {val:.4f} вне [{rules['pixel_range']['min']}, {rules['pixel_range']['max']}]")

        # Проверка L∞-расстояния от оригинала
        linf = l_inf_norm([x_test_input[i] - x_original[i] for i in range(len(x_original))])
        if linf > rules["l_inf_bound"]:
            violations.append(f"L∞-расстояние: {linf:.4f} > {rules['l_inf_bound']}")

        # Проверка статистических аномалий
        mean_diff = abs(sum(x_test_input) / len(x_test_input) - sum(x_original) / len(x_original))
        std_orig = math.sqrt(sum((v - sum(x_original) / len(x_original)) ** 2 for v in x_original) / len(x_original))
        if std_orig > 0 and mean_diff / std_orig > rules["statistical_bound"]:
            violations.append(f"Статистическое отклонение: {mean_diff / std_orig:.2f}σ > {rules['statistical_bound']}σ")

        return violations

    # Тест 1: Чистый вход
    x_orig = [0.5, 0.3, 0.7, 0.4]
    x_clean_test = [0.51, 0.29, 0.71, 0.41]
    violations_clean = validate_input(x_orig, x_clean_test, validation_rules)
    print(f"\nТест 1 — Чистый вход:")
    print(f"  Оригинал: {x_orig}")
    print(f"  Тестовый: {x_clean_test}")
    print(f"  Нарушений: {len(violations_clean)}")
    if violations_clean:
        for v in violations_clean:
            print(f"    - {v}")
    else:
        print(f"    ✓ Валидация пройдена")

    # Тест 2: Атакованный вход (FGSM)
    x_adv_test = [0.5 + 0.3, 0.3 - 0.3, 0.7 + 0.3, 0.4 - 0.3]
    x_adv_test = [max(0, min(1, v)) for v in x_adv_test]
    violations_adv = validate_input(x_orig, x_adv_test, validation_rules)
    print(f"\nТест 2 — Атакованный вход:")
    print(f"  Оригинал: {x_orig}")
    print(f"  Тестовый: {[round(v, 4) for v in x_adv_test]}")
    print(f"  Нарушений: {len(violations_adv)}")
    for v in violations_adv:
        print(f"    - {v}")

    # Тест 3: Аномальные значения
    x_anomaly = [0.5, 1.5, 0.7, -0.3]  # Выходят за [0, 1]
    violations_anom = validate_input(x_orig, x_anomaly, validation_rules)
    print(f"\nТест 3 — Аномальные значения:")
    print(f"  Оригинал: {x_orig}")
    print(f"  Тестовый: {x_anomaly}")
    print(f"  Нарушений: {len(violations_anom)}")
    for v in violations_anom:
        print(f"    - {v}")

    print(f"\n  Input Validation: эффективна против простых атак, но не обходит adversarial training")

    # --- 3.3 Certified Robustness ---
    print(f"\n--- 3.3 Certified Robustness: сертифицированная устойчивость ---")
    print("Вместо эмпирической защиты, даём математическую гарантию")

    # Certified Robustness через Randomized Smoothing (концепция)
    # Идея: классифицируем по «множественному голосованию» зашумлённых копий

    def noisy_classify(x, sigma=0.1, n_samples=100):
        """Классификация с добавлением шума и голосованием."""
        class_votes = {0: 0, 1: 0}
        for _ in range(n_samples):
            x_noisy = [xi + random.gauss(0, sigma) for xi in x]
            # Простая модель
            pred = 1 if x_noisy[0] > 0.5 else 0
            class_votes[pred] += 1
        return class_votes

    random.seed(42)

    # Тестовая точка
    x_cert = [0.55, 0.45]
    sigma = 0.1
    n_samples = 1000

    votes = noisy_classify(x_cert, sigma, n_samples)
    total = sum(votes.values())
    p_class = max(votes.values()) / total
    winning_class = max(votes, key=votes.get)

    print(f"\nRandomized Smoothing:")
    print(f"  Точка: {x_cert}, σ = {sigma}, N = {n_samples}")
    print(f"  Голоса: {votes}")
    print(f"  Класс-победитель: {winning_class} (p = {p_class:.4f})")

    # Сертификат: радиус устойчивости
    # Формула: R = σ · Φ⁻¹(p_A)
    # где Φ⁻¹ — обратная функция нормального распределения
    # Упрощённая аппроксимация Φ⁻¹
    def norm_ppf_approx(p):
        """Аппроксимация обратнойCDF нормального распределения."""
        # Rational approximation (Abramowitz & Stegun)
        if p <= 0 or p >= 1:
            return 0.0
        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))
        else:
            return -norm_ppf_approx(1 - p)

    if p_class > 0.5:
        radius_cert = sigma * norm_ppf_approx(p_class)
        print(f"  Сертифицированный радиус: R = {radius_cert:.4f}")
        print(f"  Гарантия: при ||δ||₂ < {radius_cert:.4f} предсказание стабильно")
    else:
        print(f"  p ≤ 0.5 — нет сертификата (модель неуверена)")

    # Мультиклассовый случай
    print(f"\n  Мультиклассовый случай:")
    for class_id in range(3):
        p_fake = random.uniform(0.3, 0.6)
        print(f"    Класс {class_id}: p = {p_fake:.4f}")

    print(f"\n  Certified Robustness: даёт математическую гарантию, но")
    print(f"    - требует Gaussian noise → не применимо к дискретным данным")
    print(f"    - радиус сертификата может быть слишком мал")

    print()


# ============================================================
# Демо 4: Robustness Evaluation — robust accuracy, worst-case, certification
# ============================================================

def demo4_robustness_evaluation():
    """Демонстрация оценки устойчивости моделей."""
    print("=" * 70)
    print("Демо 4: Robustness Evaluation — оценка устойчивости моделей")
    print("=" * 70)

    random.seed(42)

    # --- 4.1 Robust Accuracy ---
    print(f"\n--- 4.1 Robust Accuracy: робастная точность ---")
    print("Формула: Robust Accuracy = Accuracy на атакованных примерах")

    # Симуляция результатов модели на разных уровнях атак
    clean_accuracy = 0.95
    attack_levels = [
        {"name": "FGSM (ε=0.05)", "epsilon": 0.05, "robust_acc": 0.93},
        {"name": "FGSM (ε=0.10)", "epsilon": 0.10, "robust_acc": 0.89},
        {"name": "FGSM (ε=0.20)", "epsilon": 0.20, "robust_acc": 0.82},
        {"name": "PGD-10 (ε=0.10)", "epsilon": 0.10, "robust_acc": 0.85},
        {"name": "PGD-20 (ε=0.10)", "epsilon": 0.10, "robust_acc": 0.83},
        {"name": "C&W (L2)", "epsilon": None, "robust_acc": 0.78},
    ]

    print(f"\nЧистая точность (без атак): {clean_accuracy:.4f}")
    print(f"\n{'Атака':<25} {'ε':>8} {'Робастная точность':>18} {'Потеря':>8}")
    print("-" * 65)
    for level in attack_levels:
        eps = f"{level['epsilon']}" if level['epsilon'] is not None else "N/A"
        loss = clean_accuracy - level['robust_acc']
        print(f"  {level['name']:<23} {eps:>8} {level['robust_acc']:>18.4f} {loss:>8.4f}")

    # --- 4.2 Worst-Case Performance ---
    print(f"\n--- 4.2 Worst-Case Performance: худший случай ---")
    print("Оценка модели на наиболее уязвимых подмножествах данных")

    # Симуляция: классы с разной устойчивостью
    class_performance = {
        "Класс 0 (автомобили)": {"clean": 0.97, "robust_fgsm": 0.91, "robust_pgd": 0.88},
        "Класс 1 (велосипеды)":  {"clean": 0.94, "robust_fgsm": 0.72, "robust_pgd": 0.65},
        "Класс 2 (пешеходы)":   {"clean": 0.96, "robust_fgsm": 0.85, "robust_pgd": 0.79},
        "Класс 3 (знаки)":       {"clean": 0.98, "robust_fgsm": 0.93, "robust_pgd": 0.90},
        "Класс 4 (светофоры)":   {"clean": 0.95, "robust_fgsm": 0.78, "robust_pgd": 0.70},
    }

    print(f"\n{'Класс':<30} {'Чистая':>8} {'FGSM':>8} {'PGD-10':>8} {'Уязвимость':>10}")
    print("-" * 70)
    worst_robust = 1.0
    worst_class = ""
    for cls, perf in class_performance.items():
        robust_min = min(perf['robust_fgsm'], perf['robust_pgd'])
        vulnerability = perf['clean'] - robust_min
        marker = " ⚠" if vulnerability > 0.2 else ""
        print(f"  {cls:<28} {perf['clean']:>8.4f} {perf['robust_fgsm']:>8.4f} "
              f"{perf['robust_pgd']:>8.4f} {vulnerability:>10.4f}{marker}")
        if robust_min < worst_robust:
            worst_robust = robust_min
            worst_class = cls

    print(f"\nХудший класс: {worst_class} (robust accuracy: {worst_robust:.4f})")
    print(f"  → Нужен приоритет в улучшении для этого класса")

    # Worst-case across ε
    print(f"\nWorst-case анализ по ε:")
    print(f"{'ε':>6} {'Средняя точность':>16} {'Худшая точность':>16} {'Разброс':>10}")
    print("-" * 55)
    for eps_val in [0.05, 0.10, 0.15, 0.20, 0.30]:
        # Симуляция: точность падает пропорционально
        accs = [0.97 - eps_val * 0.5 + random.gauss(0, 0.02) for _ in range(5)]
        mean_acc = sum(accs) / len(accs)
        min_acc = min(accs)
        spread = max(accs) - min_acc
        print(f"  {eps_val:>4.2f} {mean_acc:>16.4f} {min_acc:>16.4f} {spread:>10.4f}")

    # --- 4.3 Certification: методы сертификации ---
    print(f"\n--- 4.3 Certification: методы сертификации ---")
    print("Сертификация устойчивости: математическая гарантия")

    methods = [
        {
            "name": "Interval Bound Propagation (IBP)",
            "description": "Пропагация интервалов через слои сети",
            "certified_radius": 0.15,
            "overhead": "Низкая",
            "precision": "Грубая (over-approximation)"
        },
        {
            "name": "Randomized Smoothing",
            "description": "Голосование зашумлённых копий",
            "certified_radius": 0.25,
            "overhead": "Высокая (1000+ запросов)",
            "precision": "Точная (L2-сертификат)"
        },
        {
            "name": "Lipschitz Bound",
            "description": "Ограничение чувствительности через константу Липшица",
            "certified_radius": 0.10,
            "overhead": "Средняя",
            "precision": "Грубая"
        },
        {
            "name": "Complete Verification (MILP/SAT)",
            "description": "Полная проверка через булеву satisfiability",
            "certified_radius": 1.0,
            "overhead": "Экспоненциальная",
            "precision": "Точная (ground truth)"
        },
    ]

    print(f"\n{'Метод':<35} {'R (L2)':>8} {'Overhead':>15} {'Precision':>25}")
    print("-" * 90)
    for m in methods:
        print(f"  {m['name']:<33} {m['certified_radius']:>8.2f} {m['overhead']:>15} {m['precision']:>25}")

    # Сравнение: empirical vs certified
    print(f"\nEmpirical vs Certified:")
    print(f"  Empirical:  PGD-10 attack → accuracy = 0.83 (может быть обойдена更强ой атакой)")
    print(f"  Certified:  Randomized Smoothing → R = 0.25 (математическая гарантия)")
    print(f"\n  Типичная картина:")
    print(f"    Empirical robust accuracy (PGD) ≥ Certified robust accuracy (Smoothing)")
    print(f"    Потому что: PGD не находит оптимальную атаку + сертификат «грубее» реальности")

    # --- 4.4 Практическая оценка: метрики и протоколы ---
    print(f"\n--- 4.4 Практическая оценка: метрики и протоколы ---")

    metrics = {
        "Accuracy (clean)": "Базовая точность на неповреждённых данных",
        "Accuracy (FGSM)": "Точность после FGSM-атаки (ε фиксирован)",
        "Accuracy (PGD-10)": "Точность после PGD-10 (10 шагов)",
        "Accuracy (PGD-100)": "Точность после PGD-100 (100 шагов, сильнее)",
        "Certified Radius (L2)": "Радиус сертификата устойчивости",
        "Mean Perturbation Size": "Средний размер возмущения для успеха атаки",
        "Attack Success Rate": "Доля успешных атак при фиксированном ε",
    }

    print(f"\nМетрики для оценки робастности:")
    print(f"{'Метрика':<30} {'Описание'}")
    print("-" * 75)
    for metric, desc in metrics.items():
        print(f"  {metric:<28} {desc}")

    # Протокол тестирования
    print(f"\nПротокол тестирования:")
    steps = [
        ("1. Базовая точность", "Измерить accuracy на чистом тестовом наборе"),
        ("2. FGSM-атака", "Запустить FGSM с несколькими ε"),
        ("3. PGD-атака", "Запустить PGD (10, 20, 100 шагов) с несколькими ε"),
        ("4. Мulti-target", "Проверить targeted атаки на все классы"),
        ("5. Certified", "Вычислить certified radius (если применимо)"),
        ("6. Worst-case", "Найти наиболее уязвимые классы/подмножества"),
    ]
    for step, desc in steps:
        print(f"  {step}: {desc}")

    # Демонстрация: сводная таблица
    print(f"\nСводная таблица (пример):")
    models = ["Baseline CNN", "Adversarial Training", "Certified (Smoothing)", "Ensemble"]
    results = {
        "Baseline CNN":           {"clean": 0.95, "fgsm": 0.72, "pgd": 0.65, "cert": "N/A"},
        "Adversarial Training":   {"clean": 0.91, "fgsm": 0.88, "pgd": 0.85, "cert": "N/A"},
        "Certified (Smoothing)":  {"clean": 0.88, "fgsm": 0.86, "pgd": 0.85, "cert": "0.25"},
        "Ensemble":               {"clean": 0.93, "fgsm": 0.89, "pgd": 0.87, "cert": "0.18"},
    }

    print(f"{'Модель':<30} {'Clean':>8} {'FGSM':>8} {'PGD':>8} {'Cert R':>8}")
    print("-" * 65)
    for model_name, res in results.items():
        print(f"  {model_name:<28} {res['clean']:>8.4f} {res['fgsm']:>8.4f} "
              f"{res['pgd']:>8.4f} {res['cert']:>8}")

    print(f"\n  Вывод: нет «идеального» метода — всегда компромисс между:")
    print(f"    - Чистой точностью (clean accuracy)")
    print(f"    - Робастностью (robust accuracy)")
    print(f"    - Вычислительной сложностью")
    print(f"    - Размером сертифицированного радиуса")

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    demo1_adversarial_attacks()
    demo2_attack_types()
    demo3_defenses()
    demo4_robustness_evaluation()
