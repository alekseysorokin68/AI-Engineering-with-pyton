"""
213 — Compliance & Governance: GDPR, аудит, объяснимость

Темы:
  1. Data Privacy (GDPR basics, consent, right to erasure, data minimization)
  2. Audit Trails (action logging, provenance tracking, compliance reporting)
  3. Explainability (SHAP/LIME concepts, feature importance, decision traces)
  4. Model Cards (documentation, intended use, limitations, fairness metrics)

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


# ============================================================
# Демо 1: Приватность данных — GDPR-базисы, согласие, право на удаление
# ============================================================
def demo_data_privacy():
    print("=" * 70)
    print("ДЕМО 1: Приватность данных — GDPR, согласие, право на удаление")
    print("=" * 70)

    # --- 1.1 Модель записи пользователя с метаданными согласия ---
    print("\n--- 1.1 Регистрация пользователя с согласием GDPR ---")

    # Каждый пользователь хранится как словарь с полями
    user_db = {}
    user_counter = 0

    def register_user(name, email, consent_purpose, consent_given):
        """Регистрирует пользователя и записывает факт согласия."""
        nonlocal user_counter
        user_counter += 1
        user_id = f"USR-{user_counter:04d}"

        # Хэшируем email для анонимизации в логах
        email_hash = hashlib.sha256(email.encode()).hexdigest()[:12]

        user_db[user_id] = {
            "name": name,
            "email": email,
            "email_hash": email_hash,
            "consent": {
                "purpose": consent_purpose,
                "given": consent_given,
                "timestamp": time.time(),
                "version": "GDPR-v2.1",
            },
            "data_fields": ["name", "email"],
            "created_at": time.time(),
        }

        # Проверяем: если согласие не дано — нельзя обрабатывать данные
        if not consent_given:
            print(f"  [ПРЕДУПРЕЖДЕНИЕ] Пользователь {user_id} НЕ дал согласие!")

        print(f"  Зарегистрирован: {user_id} | email_hash={email_hash} | "
              f"согласие={'ДА' if consent_given else 'НЕТ'}")
        return user_id

    # Регистрируем пользователей
    uid1 = register_user("Иван Петров", "ivan@example.com", "аналитика", True)
    uid2 = register_user("Мария Сидорова", "maria@example.com", "маркетинг", True)
    uid3 = register_user("Алексей Козлов", "alex@example.com", "реклама", False)

    print(f"\n  Всего пользователей в БД: {len(user_db)}")
    print(f"  Активных согласий: {sum(1 for u in user_db.values() if u['consent']['given'])}")

    # --- 1.2 Право на удаление (Right to Erasure / «Забыть меня») ---
    print("\n--- 1.2 Право на удаление (Right to Erasure) ---")

    deletion_log = []

    def erase_user(user_id):
        """Удаляет все данные пользователя и логирует операцию."""
        if user_id not in user_db:
            print(f"  ОШИБКА: Пользователь {user_id} не найден")
            return False

        user = user_db[user_id]
        # Сохраняем хэш для аудита без содержания данных
        audit_hash = user["email_hash"]
        fields_deleted = list(user["data_fields"])

        # Физическое удаление из БД
        del user_db[user_id]

        # Запись в лог удаления (без персональных данных!)
        deletion_record = {
            "user_id": user_id,
            "email_hash_for_audit": audit_hash,
            "fields_erased": fields_deleted,
            "timestamp": time.time(),
            "reason": "GDPR Art.17 — право на удаление",
        }
        deletion_log.append(deletion_record)

        print(f"  Удалён: {user_id} | удалённые поля: {fields_deleted}")
        print(f"  Аудит-хэш: {audit_hash} (персональные данные уничтожены)")
        return True

    # Удаляем пользователя, который не давал согласия
    erase_user(uid3)
    print(f"  Осталось пользователей: {len(user_db)}")
    print(f"  Записей в логе удалений: {len(deletion_log)}")

    # --- 1.3 Минимизация данных — хранение только необходимого ---
    print("\n--- 1.3 Минимизация данных (Data Minimization) ---")

    # Пример: для рекомендаций нам НЕ нужны имя и email
    def build_recommendation_profile(user_id):
        """Строит минимальный профиль для рекомендаций — без PII."""
        user = user_db.get(user_id)
        if not user:
            return None

        # Минимальный набор: только то, что нужно для алгоритма
        minimal_profile = {
            "user_hash": user["email_hash"],
            "interaction_score": random.randint(1, 100),
            "category_pref": random.choice(["техника", "еда", "спорт", "книги"]),
        }
        return minimal_profile

    for uid in user_db:
        profile = build_recommendation_profile(uid)
        print(f"  {uid}: {profile}")

    # Подсчитываем разницу в размере данных
    full_size = sum(len(json.dumps(u)) for u in user_db.values())
    minimal_size = sum(len(json.dumps(build_recommendation_profile(uid)))
                       for uid in user_db)
    reduction = (1 - minimal_size / full_size) * 100 if full_size else 0
    print(f"\n  Полный размер: ~{full_size} байт → Минимальный: ~{minimal_size} байт")
    print(f"  Сокращение: {reduction:.0f}%")

    # --- 1.4 Журнал обработки персональных данных ---
    print("\n--- 1.4 Реестр обработки персональных данных (ROPA) ---")

    processing_registry = []

    def log_processing(user_id, purpose, legal_basis, data_categories, retention_days):
        """Записывает операцию обработки в реестр."""
        entry = {
            "user_id": user_id,
            "purpose": purpose,
            "legal_basis": legal_basis,
            "data_categories": data_categories,
            "retention_days": retention_days,
            "timestamp": time.time(),
        }
        processing_registry.append(entry)
        print(f"  Обработка: {user_id} | цель: {purpose} | "
              f"основание: {legal_basis} | удержание: {retention_days} дн.")

    for uid in user_db:
        log_processing(uid, "персонализация", "согласие",
                       ["email", "имя"], retention_days=365)
        log_processing(uid, "аналитика", "легитимный интерес",
                       ["hash", "метрики"], retention_days=90)

    print(f"\n  Всего записей в реестре: {len(processing_registry)}")
    purposes = collections.Counter(e["purpose"] for e in processing_registry)
    print(f"  По целям: {dict(purposes)}")

    print("\n  === Итог Demo 1 ===")
    print("  GDPR требует: согласие, право на удаление, минимизацию, учёт обработки")
    print("  Каждая операция с данными должна иметь юридическое основание")


# ============================================================
# Демо 2: Аудит-трейлы — логирование действий, прослеживаемость
# ============================================================
def demo_audit_trails():
    print("\n" + "=" * 70)
    print("ДЕМО 2: Аудит-трейлы — логирование действий, прослеживаемость")
    print("=" * 70)

    # --- 2.1 Система логирования действий ---
    print("\n--- 2.1 Система логирования действий ---")

    audit_log = []

    def log_action(actor, action, resource, details=None):
        """Логирует действие с хэшем целостности."""
        entry = {
            "timestamp": time.time(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "details": details or {},
        }
        # Вычисляем хэш для обнаружения подделки
        entry_str = json.dumps(entry, sort_keys=True)
        entry["integrity_hash"] = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
        audit_log.append(entry)
        print(f"  [{actor}] {action} → {resource} | hash={entry['integrity_hash']}")
        return entry

    # Имитируем реальные действия в системе
    log_action("data_engineer", "upload", "dataset_v3.parquet",
               {"rows": 50000, "cols": 12})
    log_action("ml_engineer", "train", "model_xgboost_v2",
               {"algorithm": "XGBoost", "epochs": 100})
    log_action("ml_engineer", "evaluate", "model_xgboost_v2",
               {"accuracy": 0.94, "f1": 0.91})
    log_action("ml_ops", "deploy", "model_xgboost_v2",
               {"target": "production", " replicas": 3})
    log_action("analyst", "query", "predictions_table",
               {"filters": "date=today"})

    print(f"\n  Всего событий в журнале: {len(audit_log)}")

    # --- 2.2 Прослеживаемость (Provenance) — цепочка от данных до модели ---
    print("\n--- 2.2 Прослеживаемость (Provenance Tracking) ---")

    # Каждый артефакт ссылается на свои источники
    artifacts = {
        "raw_data": {
            "type": "dataset",
            "source": "api://weather-service/v2",
            "hash": hashlib.md5(b"weather_raw").hexdigest()[:8],
            "parents": [],
        },
        "clean_data": {
            "type": "dataset",
            "source": "pipeline/preprocessing",
            "hash": hashlib.md5(b"weather_clean").hexdigest()[:8],
            "parents": ["raw_data"],
        },
        "model_v1": {
            "type": "model",
            "source": "training/xgboost",
            "hash": hashlib.md5(b"model_v1_weights").hexdigest()[:8],
            "parents": ["clean_data"],
        },
        "model_v2": {
            "type": "model",
            "source": "training/xgboost_tuned",
            "hash": hashlib.md5(b"model_v2_weights").hexdigest()[:8],
            "parents": ["clean_data", "model_v1"],
        },
        "prediction": {
            "type": "output",
            "source": "serving/production",
            "hash": hashlib.md5(b"pred_batch_42").hexdigest()[:8],
            "parents": ["model_v2"],
        },
    }

    # Рекурсивно строим дерево происхождения
    def trace_provenance(artifact_id, depth=0):
        """Выводит цепочку происхождения артефакта."""
        art = artifacts[artifact_id]
        indent = "    " * depth
        print(f"{indent}├── {artifact_id} ({art['type']}) | hash={art['hash']}")
        for parent_id in art["parents"]:
            trace_provenance(parent_id, depth + 1)

    print("  Цепочка происхождения для prediction:")
    trace_provenance("prediction")

    # Проверяем целостность цепочки
    print("\n  Проверка целостности:")
    for art_id, art in artifacts.items():
        # Проверяем, что все родители существуют
        valid = all(p in artifacts for p in art["parents"])
        status = "OK" if valid else "БРОШЕН!"
        print(f"    {art_id}: {status}")

    # --- 2.3 Аудит-отчёт для комплаенса ---
    print("\n--- 2.3 Генерация аудит-отчёта ---")

    def generate_audit_report(log_entries):
        """Генерирует структурированный аудит-отчёт."""
        report = {
            "report_id": hashlib.sha256(str(time.time()).encode()).hexdigest()[:10],
            "period": "2024-Q4",
            "total_events": len(log_entries),
            "by_actor": {},
            "by_action": {},
            "integrity_status": "PASS",
        }

        # Подсчёт по.actorам
        for entry in log_entries:
            actor = entry["actor"]
            action = entry["action"]
            report["by_actor"][actor] = report["by_actor"].get(actor, 0) + 1
            report["by_action"][action] = report["by_action"].get(action, 0) + 1

        # Проверка целостности (упрощённо: все хэши должны быть 16 символов)
        for entry in log_entries:
            if len(entry.get("integrity_hash", "")) != 16:
                report["integrity_status"] = "FAIL"
                break

        return report

    report = generate_audit_report(audit_log)
    print(f"  Report ID: {report['report_id']}")
    print(f"  Период: {report['period']}")
    print(f"  Всего событий: {report['total_events']}")
    print(f"  По.actorам: {report['by_actor']}")
    print(f"  По действиям: {report['by_action']}")
    print(f"  Статус целостности: {report['integrity_status']}")

    # --- 2.4 Обнаружение аномалий в аудит-логе ---
    print("\n--- 2.4 Обнаружение аномалий в аудит-логе ---")

    def detect_anomalies(log_entries):
        """Ищет подозрительные паттерны в журнале."""
        anomalies = []

        # Правило 1: один actor melakukan больше 3 действий за сессию
        actor_counts = collections.Counter(e["actor"] for e in log_entries)
        for actor, count in actor_counts.items():
            if count > 3:
                anomalies.append({
                    "type": "high_activity",
                    "actor": actor,
                    "count": count,
                    "severity": "medium",
                })

        # Правило 2: deploy без предшествующего evaluate
        actions = [e["action"] for e in log_entries]
        has_deploy = "deploy" in actions
        has_evaluate = "evaluate" in actions
        if has_deploy and not has_evaluate:
            anomalies.append({
                "type": "deploy_without_eval",
                "severity": "high",
            })

        # Правило 3: необычное время (имитируем)
        night_actions = [e for e in log_entries
                         if 2 <= time.localtime(e["timestamp"]).tm_hour <= 5]
        if night_actions:
            anomalies.append({
                "type": "off_hours_activity",
                "count": len(night_actions),
                "severity": "low",
            })

        return anomalies

    anomalies = detect_anomalies(audit_log)
    print(f"  Обнаружено аномалий: {len(anomalies)}")
    for a in anomalies:
        print(f"    [{a['severity'].upper()}] {a['type']}: {a}")

    print("\n  === Итог Demo 2 ===")
    print("  Аудит-трейлы обеспечивают: прослеживаемость, целостность, обнаружение аномалий")
    print("  Каждое действие должно быть логировано и проверяемо")


# ============================================================
# Демо 3: Объяснимость — SHAP/LIME концепции, важность признаков
# ============================================================
def demo_explainability():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Объяснимость — SHAP/LIME концепции, важность признаков")
    print("=" * 70)

    # --- 3.1 Простая модель: взвешенная сумма (линейная) ---
    print("\n--- 3.1 Линейная модель и вклад признаков ---")

    # Наша "модель":预测 = w0 + w1*x1 + w2*x2 + w3*x3
    features = ["возраст", "доход", "кредит_рейтинг"]
    weights = [0.3, 0.5, 0.2]  # w1, w2, w3
    bias = 0.1  # w0

    def simple_linear_model(x):
        """Вычисляет предсказание линейной модели."""
        prediction = bias
        contributions = []
        for i, (feat, w) in enumerate(zip(features, weights)):
            contrib = w * x[i]
            contributions.append((feat, contrib))
            prediction += contrib
        return prediction, contributions

    # Пример: предсказание для клиента
    sample = [35, 50000, 720]  # возраст, доход, рейтинг
    pred, contribs = simple_linear_model(sample)

    print("  Модель: pred = {b} + {w1}*возраст + {w2}*доход + {w3}*рейтинг".format(
        b=bias, w1=weights[0], w2=weights[1], w3=weights[2]))
    print(f"  Вход: возраст={sample[0]}, доход={sample[1]}, рейтинг={sample[2]}")
    print(f"  Предсказание: {pred:.2f}")
    print("  Вклад каждого признака:")
    for feat, contrib in contribs:
        pct = contrib / pred * 100 if pred else 0
        bar = "█" * int(abs(pct) / 2)
        sign = "+" if contrib > 0 else "-"
        print(f"    {feat:>15}: {sign}{abs(contrib):.2f} ({pct:+.1f}%) {bar}")

    # --- 3.2 SHAP-значения (концепция Shapley) ---
    print("\n--- 3.2 SHAP-значения — теория игр в ML ---")

    # Для простоты: модель предсказывает, одобрят ли кредит
    # SHAP показывает, как каждый признак сдвигает предсказание от базового
    baseline_prediction = 0.5  # среднее по всем клиентам

    def compute_shap_values(x, baseline, weights, features):
        """
        Упрощённый расчёт SHAP для линейной модели.
        SHAP_i = w_i * (x_i - mean_i) для линейных моделей.
        """
        means = [30, 40000, 650]  # средние по population
        shap_values = {}
        for i, feat in enumerate(features):
            # SHAP = вклад признака выше/ниже среднего
            shap_values[feat] = weights[i] * (x[i] - means[i])

        return shap_values

    shaps = compute_shap_values(sample, baseline_prediction, weights, features)
    print(f"  Базовое предсказание (среднее): {baseline_prediction}")
    print("  SHAP-значения (сдвиг от базового):")
    total_shap = 0
    for feat, shap_val in sorted(shaps.items(), key=lambda t: abs(t[1]), reverse=True):
        total_shap += shap_val
        direction = "↑ повышает" if shap_val > 0 else "↓ понижает"
        bar_len = int(min(abs(shap_val) / 50, 20))
        bar = "█" * bar_len
        print(f"    {feat:>15}: {shap_val:+.1f}  {direction}  {bar}")

    final = baseline_prediction + total_shap
    print(f"\n  Итого: {baseline_prediction} + {total_shap:.1f} = {final:.2f}")

    # --- 3.3 LIME-подобная локальная интерпретация ---
    print("\n--- 3.3 LIME-подобная локальная интерпретация ---")

    # Генерируем "окрестность" вокруг точки и смотрим, как меняется предсказание
    def lime_local_explanation(x, model_fn, n_samples=200):
        """Генерирует локальную интерпретацию методом пертурбаций."""
        feature_importance = [0.0] * len(x)
        perturbation_scale = [5, 5000, 30]  # масштаб возмущения для каждого признака

        for _ in range(n_samples):
            # Генерируем случайную точку в окрестности
            x_perturbed = []
            for j in range(len(x)):
                noise = random.gauss(0, perturbation_scale[j])
                x_perturbed.append(x[j] + noise)

            pred_original, _ = model_fn(x)
            pred_perturbed, _ = model_fn(x_perturbed)

            # Вклад признака = как сильно возмущение изменило предсказание
            for j in range(len(x)):
                delta = pred_perturbed - pred_original
                # Нормируем на величину шума
                if abs(x_perturbed[j] - x[j]) > 1e-10:
                    feature_importance[j] += delta * (x[j] - x_perturbed[j])

        # Нормируем
        total = sum(abs(fi) for fi in feature_importance)
        if total > 0:
            feature_importance = [fi / total for fi in feature_importance]

        return feature_importance

    lime_importances = lime_local_explanation(sample, simple_linear_model)
    print("  LIME: локальная важность признаков (нормализованная):")
    for feat, imp in sorted(zip(features, lime_importances),
                            key=lambda t: abs(t[1]), reverse=True):
        bar_len = int(abs(imp) * 40)
        bar = "█" * bar_len
        print(f"    {feat:>15}: {imp:+.3f}  {bar}")

    # --- 3.4 Трассировка решений (Decision Trace) ---
    print("\n--- 3.4 Трассировка решений — пошаговый audit ---")

    def decision_trace(x, features, weights, bias, threshold=0.5):
        """Возвращает полную трассировку принятия решения."""
        trace = {
            "input": dict(zip(features, x)),
            "steps": [],
            "final_prediction": None,
            "decision": None,
        }

        cumulative = bias
        trace["steps"].append({
            "step": 0,
            "operation": "bias",
            "value": bias,
            "cumulative": cumulative,
            "explanation": f"Начинаем с базового значения {bias}",
        })

        for i, (feat, w) in enumerate(zip(features, weights)):
            contrib = w * x[i]
            cumulative += contrib
            trace["steps"].append({
                "step": i + 1,
                "operation": f"{feat} × {w}",
                "feature_value": x[i],
                "contribution": contrib,
                "cumulative": cumulative,
                "explanation": (f"{feat}={x[i]} × вес={w} → вклад {contrib:+.2f}, "
                                f"итого {cumulative:.2f}"),
            })

        trace["final_prediction"] = cumulative
        trace["decision"] = "ОДОБРЕНО" if cumulative > threshold else "ОТКЛОНЕНО"
        return trace

    trace = decision_trace(sample, features, weights, bias)
    print(f"  Входные данные: {trace['input']}")
    print("  Пошаговая трассировка:")
    for step in trace["steps"]:
        print(f"    Шаг {step['step']}: {step['explanation']}")
    print(f"\n  Финальное предсказание: {trace['final_prediction']:.2f}")
    print(f"  Решение: {trace['decision']} (порог = {0.5})")

    print("\n  === Итог Demo 3 ===")
    print("  Объяснимость требует: вклад признаков, SHAP-значения, локальные интерпретации")
    print("  Трассировка решений помогает понять ЛЮБОЕ предсказание модели")


# ============================================================
# Демо 4: Карточки моделей — документация, ограничения, справедливость
# ============================================================
def demo_model_cards():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Карточки моделей — документация, ограничения, справедливость")
    print("=" * 70)

    # --- 4.1 Генерация карточки модели ---
    print("\n--- 4.1 Генерация карточки модели (Model Card) ---")

    model_card = {
        "model_name": "CreditScorer v2.1",
        "version": "2.1.0",
        "date": "2024-12-01",
        "team": "ML Platform Team",
        "intended_use": {
            "primary": "Оценка кредитоспособности физических лиц",
            "out_of_scope": [
                "Юридические лица",
                "Решения о выдаче кредитов (только скоринг)",
                "Использование без человеческого контроля",
            ],
        },
        "training_data": {
            "source": "Внутренняя БД банка",
            "period": "2020-2024",
            "size_samples": 500000,
            "features": 24,
            "label": "дефолт_в_90_дней (0/1)",
            "preprocessing": "удаление дубликатов, импутация медианой",
        },
        "evaluation_results": {
            "test_size": 100000,
            "metrics": {
                "accuracy": 0.91,
                "precision": 0.87,
                "recall": 0.83,
                "f1": 0.85,
                "auc_roc": 0.94,
                "gini": 0.88,
            },
        },
        "limitations": [
            "Модель не учитывает неформальную занятость",
            "Хуже работает для возраста < 21 (мало данных)",
            "Не применима к регионам с высокой инфляцией",
        ],
        "ethical_considerations": [
            "Возможна дискриминация по возрасту (молодые клиенты)",
            "Необходима регулярная проверка fairness metrics",
        ],
    }

    # Выводим карточку в форматированном виде
    print("  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Модель: {model_card['model_name']:<38} │")
    print(f"  │  Версия: {model_card['version']:<38} │")
    print(f"  │  Дата:   {model_card['date']:<38} │")
    print(f"  │  Команда: {model_card['team']:<37} │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Назначение: {model_card['intended_use']['primary']:<33} │")
    print("  └─────────────────────────────────────────────────┘")

    # --- 4.2 Метрики справедливости (Fairness) ---
    print("\n--- 4.2 Метрики справедливости ---")

    # Имитируем результаты для разных групп
    groups = {
        "мужчины 25-40": {"tp": 4500, "fp": 500, "fn": 800, "tn": 4200},
        "женщины 25-40": {"tp": 4300, "fp": 600, "fn": 700, "tn": 4400},
        "мужчины 18-25": {"tp": 1800, "fp": 400, "fn": 600, "tn": 2200},
        "женщины 18-25": {"tp": 1900, "fp": 350, "fn": 550, "tn": 2200},
        "мужчины 50+":   {"tp": 2100, "fp": 300, "fn": 400, "tn": 2200},
        "женщины 50+":   {"tp": 2000, "fp": 320, "fn": 380, "tn": 2300},
    }

    def compute_group_metrics(confusion):
        """Вычисляет метрики для одной группы."""
        tp = confusion["tp"]
        fp = confusion["fp"]
        fn = confusion["fn"]
        tn = confusion["tn"]
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        approval_rate = (tp + fp) / total if total else 0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "approval_rate": approval_rate,
        }

    print(f"  {'Группа':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Approval%':>10}")
    print("  " + "-" * 62)

    group_metrics = {}
    for group_name, confusion in groups.items():
        metrics = compute_group_metrics(confusion)
        group_metrics[group_name] = metrics
        print(f"  {group_name:<20} {metrics['accuracy']:.3f} {metrics['precision']:.3f} "
              f"{metrics['recall']:.3f} {metrics['f1']:.3f} "
              f"{metrics['approval_rate']*100:.1f}%")

    # Disparate Impact: ratio approval rates
    print("\n  Disparate Impact Ratio (мин/макс approval_rate):")
    approval_rates = [m["approval_rate"] for m in group_metrics.values()]
    di_ratio = min(approval_rates) / max(approval_rates) if max(approval_rates) else 0
    print(f"    DI = {di_ratio:.3f} (норма: > 0.8)")
    if di_ratio >= 0.8:
        print("    Статус: ПРОХОДИТ порог справедливости")
    else:
        print("    Статус: НЕ ПРОХОДИТ — необходима корректировка!")

    # Equal Opportunity: разница в recall между группами
    print("\n  Equal Opportunity Difference (разница в recall):")
    recalls = [m["recall"] for m in group_metrics.values()]
    eo_diff = max(recalls) - min(recalls)
    print(f"    EOD = {eo_diff:.3f} (норма: < 0.1)")
    if eo_diff < 0.1:
        print("    Статус: ПРОХОДИТ")
    else:
        print("    Статус: НЕ ПРОХОДИТ — модели дают разные шансы группам")

    # --- 4.3 Версионирование и откат ---
    print("\n--- 4.3 Версионирование и откат модели ---")

    model_registry = [
        {
            "version": "v1.0",
            "date": "2024-06-01",
            "metrics": {"f1": 0.78, "auc": 0.85},
            "status": "deprecated",
            "notes": "Первая версия, baseline",
        },
        {
            "version": "v1.5",
            "date": "2024-08-15",
            "metrics": {"f1": 0.82, "auc": 0.89},
            "status": "deprecated",
            "notes": "Улучшение после feature engineering",
        },
        {
            "version": "v2.0",
            "date": "2024-10-01",
            "metrics": {"f1": 0.85, "auc": 0.93},
            "status": "deprecated",
            "notes": "Новый алгоритм (XGBoost)",
        },
        {
            "version": "v2.1",
            "date": "2024-12-01",
            "metrics": {"f1": 0.87, "auc": 0.94},
            "status": "production",
            "notes": "Тюнинг гиперпараметров, fairness check",
        },
    ]

    print("  Регистр моделей:")
    for m in model_registry:
        status_marker = " ★" if m["status"] == "production" else "  "
        print(f"    {status_marker} {m['version']} ({m['date']}) | "
              f"F1={m['metrics']['f1']:.2f} AUC={m['metrics']['auc']:.2f} | "
              f"{m['status']:>12} | {m['notes']}")

    # Функция "отката" на предыдущую версию
    def rollback_model(registry, target_version):
        """Откатывает модель на указанную версию."""
        for m in registry:
            if m["version"] == target_version and m["status"] != "production":
                # Снимаем текущий production
                for r in registry:
                    if r["status"] == "production":
                        r["status"] = "rolled_back"
                m["status"] = "production"
                print(f"\n  ОТКАТ: {target_version} снова в production!")
                print(f"  Причина: fairness metrics не прошли порог")
                return m
        print(f"  Версия {target_version} не найдена")
        return None

    rollback_model(model_registry, "v2.0")

    # --- 4.4 Экспорт карточки в JSON ---
    print("\n--- 4.4 Экспорт карточки в формат JSON ---")

    export_data = {
        "model_card": model_card,
        "fairness_report": {
            "group_metrics": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in group_metrics.items()
            },
            "disparate_impact": round(di_ratio, 4),
            "equal_opportunity_diff": round(eo_diff, 4),
        },
        "registry": model_registry,
    }

    # Выводим ключевые поля JSON
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    lines = json_str.split("\n")
    # Показываем первые 30 строк
    for line in lines[:30]:
        print(f"  {line}")
    if len(lines) > 30:
        print(f"  ... (ещё {len(lines) - 30} строк)")

    # Хэш всего документа для целостности
    doc_hash = hashlib.sha256(json_str.encode()).hexdigest()[:16]
    print(f"\n  Хэш документа: {doc_hash}")
    print(f"  Размер: {len(json_str)} символов")

    print("\n  === Итог Demo 4 ===")
    print("  Карточка модели = документация + fairness + ограничения + версии")
    print("  Без карточки модель нельзя выпускать в production!")


# ============================================================
# Точка входа
# ============================================================
if __name__ == "__main__":
    demo_data_privacy()
    demo_audit_trails()
    demo_explainability()
    demo_model_cards()
