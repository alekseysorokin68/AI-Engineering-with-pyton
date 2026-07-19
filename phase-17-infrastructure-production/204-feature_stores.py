"""204 — Feature Stores: инженерия признаков, serving, свежесть данных

Темы:
  1. Feature Engineering (пайплайны трансформаций, создание признаков, кодирование)
  2. Feature Store Architecture (offline store, online store, реестр)
  3. Feature Serving (real-time vs batch, свежесть данных, кэширование)
  4. Feature Monitoring (обнаружение дрейфа, отслеживание важности признаков)

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
# Демо 1: Feature Engineering — пайплайны трансформаций, создание признаков,
#         кодирование категорий
# ============================================================================

def demo_feature_engineering():
    """Демонстрация основ feature engineering: трансформации, кодирование,
    создание производных признаков."""
    print("=" * 70)
    print("ДЕМО 1: Feature Engineering — трансформации, кодирование признаков")
    print("=" * 70)

    # --- 1.1 Нормализация (Min-Max Scaling) ---
    print("\n--- 1.1 Нормализация (Min-Max Scaling) ---")
    # Формула: X_norm = (X - X_min) / (X_max - X_min)
    raw_values = [15, 22, 8, 42, 35, 10, 28, 5]
    print(f"Исходные значения: {raw_values}")

    min_val = min(raw_values)
    max_val = max(raw_values)
    normalized = [(x - min_val) / (max_val - min_val) for x in raw_values]
    print(f"Мин: {min_val}, Макс: {max_val}")
    for orig, norm in zip(raw_values, normalized):
        print(f"  {orig:3d} -> {norm:.4f}")

    # --- 1.2 Стандартизация (Z-score) ---
    print("\n--- 1.2 Стандартизация (Z-score Normalization) ---")
    # Формула: Z = (X - mean) / std
    data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 45]
    mean_val = statistics.mean(data)
    std_val = statistics.stdev(data)
    print(f"Данные: {data}")
    print(f"Среднее: {mean_val:.2f}, Ст.отклонение: {std_val:.2f}")
    z_scores = [(x - mean_val) / std_val for x in data]
    for orig, z in zip(data, z_scores):
        print(f"  {orig:3d} -> Z={z:+.4f}")
    print(f"Проверка: среднее Z={statistics.mean(z_scores):.6f} (≈0)")

    # --- 1.3 One-Hot Encoding для категорий ---
    print("\n--- 1.3 One-Hot Encoding категорий ---")
    categories = ["кот", "собака", "рыба", "кот", "собака", "собака"]
    unique_cats = sorted(set(categories))
    # Создаём словарь для кодирования
    one_hot_map = {cat: [0] * len(unique_cats) for cat in unique_cats}
    for i, cat in enumerate(unique_cats):
        one_hot_map[cat][i] = 1
    print(f"Категории: {categories}")
    print(f"Уникальные: {unique_cats}")
    print(f"Словарь кодирования:")
    for cat in unique_cats:
        print(f"  '{cat}' -> {one_hot_map[cat]}")
    # Применяем кодирование
    encoded = [one_hot_map[cat] for cat in categories]
    print(f"Закодированные: {encoded}")

    # --- 1.4 Биннинг (дискретизация непрерывных признаков) ---
    print("\n--- 1.4 Биннинг (дискретизация) ---")
    # Разбиваем возраст на группы: молодой, средний, пожилой
    ages = [18, 25, 32, 45, 55, 62, 71, 28, 39, 48]
    bins = [(0, 25, "молодой"), (25, 45, "средний"), (45, 100, "пожилой")]
    print(f"Возраст: {ages}")
    binned = []
    for age in ages:
        for low, high, label in bins:
            if low <= age < high:
                binned.append(label)
                break
        else:
            binned.append("неизвестно")
    print(f"Результат биннинга:")
    for age, label in zip(ages, binned):
        print(f"  {age:3d} -> {label}")

    print("\n[OK] Feature Engineering — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 2: Feature Store Architecture — offline store, online store, реестр
# ============================================================================

def demo_feature_store_architecture():
    """Демонстрация архитектуры feature store: хранилища, реестр, контроль версий."""
    print("=" * 70)
    print("ДЕМО 2: Feature Store Architecture — хранилища и реестр признаков")
    print("=" * 70)

    # --- 2.1 Offline Store — хранение партиций данных для обучения ---
    print("\n--- 2.1 Offline Store — партиции для обучения ---")
    # Имитация табличных данных с партиционированием по дате
    offline_store = {
        "2025-01-15": [
            {"user_id": "u001", "age": 28, "purchases": 12, "avg_check": 450.0},
            {"user_id": "u002", "age": 35, "purchases": 8, "avg_check": 320.0},
        ],
        "2025-01-16": [
            {"user_id": "u001", "age": 28, "purchases": 13, "avg_check": 460.0},
            {"user_id": "u003", "age": 42, "purchases": 20, "avg_check": 580.0},
        ],
        "2025-01-17": [
            {"user_id": "u002", "age": 35, "purchases": 9, "avg_check": 310.0},
            {"user_id": "u003", "age": 42, "purchases": 22, "avg_check": 590.0},
        ],
    }
    print("Партиции offline store (имитация Parquet/S3):")
    for date, records in offline_store.items():
        print(f"  Дата: {date}, записей: {len(records)}")
        for rec in records:
            print(f"    {rec}")

    # --- 2.2 Online Store — быстрый доступ для serving ---
    print("\n--- 2.2 Online Store — кэш для инференса ---")
    # Имитация Redis-like хранилища с TTL
    online_store = {}
    ttl_seconds = 300  # 5 минут жизни записи
    current_time = time.time()

    # Записываем в online store
    features_online = {"user_id": "u001", "age": 28, "purchases": 13,
                       "avg_check": 460.0, "last_updated": current_time}
    online_store["u001"] = features_online
    print(f"Online store (key: u001):")
    print(f"  Признаки: {json.dumps({k: v for k, v in features_online.items() if k != 'last_updated'}, indent=2)}")
    print(f"  TTL: {ttl_seconds} сек")
    print(f"  Возраст признака: {current_time - features_online['last_updated']:.1f} сек")

    # Проверка актуальности
    is_fresh = (current_time - features_online["last_updated"]) < ttl_seconds
    print(f"  Свежесть: {'актуально' if is_fresh else 'устарело'}")

    # --- 2.3 Реестр признаков (Feature Registry) ---
    print("\n--- 2.3 Реестр признаков (Feature Registry) ---")
    registry = {
        "user.age": {
            "type": "int",
            "owner": "team-analytics",
            "description": "Возраст пользователя на момент регистрации",
            "source": "user_profiles",
            "offline_type": "parquet",
            "online_type": "redis",
        },
        "user.purchases_count": {
            "type": "int",
            "owner": "team-commerce",
            "description": "Количество покупок за последние 30 дней",
            "source": "transactions",
            "offline_type": "parquet",
            "online_type": "redis",
        },
        "user.avg_check": {
            "type": "float",
            "owner": "team-analytics",
            "description": "Средний чек пользователя",
            "source": "transactions",
            "offline_type": "parquet",
            "online_type": "redis",
        },
    }
    print("Регистр признаков (Feature Group: user_features):")
    for name, meta in registry.items():
        print(f"\n  {name}:")
        print(f"    Тип: {meta['type']}, Владелец: {meta['owner']}")
        print(f"    Описание: {meta['description']}")
        print(f"    Источник: {meta['source']}")
        print(f"    Offline: {meta['offline_type']}, Online: {meta['online_type']}")

    # --- 2.4 Версионирование признаков ---
    print("\n--- 2.4 Версионирование признаков ---")
    feature_versions = [
        {"version": "v1", "formula": "purchases_count / days_since_reg",
         "accuracy": 0.82, "created": "2025-01-01"},
        {"version": "v2", "formula": "log(purchases_count + 1) / sqrt(days_since_reg)",
         "accuracy": 0.87, "created": "2025-01-10"},
        {"version": "v3", "formula": "exp(purchases_count / (days + 1)) - 1",
         "accuracy": 0.89, "created": "2025-01-20", "status": "active"},
    ]
    print("Версии признака 'purchase_rate':")
    for v in feature_versions:
        status = v.get("status", "archived")
        marker = " <-- ТЕКУЩАЯ" if status == "active" else ""
        print(f"  {v['version']}: {v['formula']}")
        print(f"    Точность: {v['accuracy']:.2f}, Дата: {v['created']}{marker}")

    print("\n[OK] Feature Store Architecture — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 3: Feature Serving — real-time vs batch, свежесть данных, кэширование
# ============================================================================

def demo_feature_serving():
    """Демонстрация serving признаков: batch vs real-time, кэширование, TTL."""
    print("=" * 70)
    print("ДЕМО 3: Feature Serving — real-time vs batch, кэширование")
    print("=" * 70)

    # --- 3.1 Batch Feature Generation ---
    print("\n--- 3.1 Batch Feature Generation ---")
    # Генерация признаков заранее для всех пользователей
    users_batch = [
        {"id": "u1", "total_orders": 45, "total_spent": 12500, "days_active": 365},
        {"id": "u2", "total_orders": 3, "total_spent": 800, "days_active": 14},
        {"id": "u3", "total_orders": 20, "total_spent": 5600, "days_active": 180},
    ]
    print("Батч-признаки (вычислены заранее):")
    batch_features = {}
    for u in users_batch:
        # Создаём производные признаки
        avg_order_value = u["total_spent"] / max(u["total_orders"], 1)
        orders_per_day = u["total_orders"] / max(u["days_active"], 1)
        spend_per_day = u["total_spent"] / max(u["days_active"], 1)
        batch_features[u["id"]] = {
            "avg_order_value": round(avg_order_value, 2),
            "orders_per_day": round(orders_per_day, 4),
            "spend_per_day": round(spend_per_day, 2),
        }
        print(f"  {u['id']}: avg_order={avg_order_value:.2f}, "
              f"orders/day={orders_per_day:.4f}, spend/day={spend_per_day:.2f}")

    # --- 3.2 Real-time Feature Computation ---
    print("\n--- 3.2 Real-time Feature Computation ---")
    # Вычисление признаков на лету по событию
    event = {"user_id": "u1", "event_type": "click", "timestamp": time.time(),
             "item_id": "item_42", "price": 299.0}
    print(f"Событие: {event['event_type']} на item={event['item_id']}, "
          f"цена={event['price']}")

    # Real-time признаки на основе контекста
    hour_of_day = time.localtime(event["timestamp"]).tm_hour
    is_weekend = time.localtime(event["timestamp"]).tm_wday >= 5
    user_features = batch_features.get(event["user_id"], {})
    realtime_features = {
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "user_avg_order": user_features.get("avg_order_value", 0),
        "price_vs_avg": event["price"] / max(user_features.get("avg_order_value", 1), 0.01),
    }
    print(f"Real-time признаки:")
    for k, v in realtime_features.items():
        print(f"  {k}: {v}")

    # --- 3.3 Проверка свежести данных (Freshness Check) ---
    print("\n--- 3.3 Проверка свежести данных (Freshness) ---")
    feature_sources = [
        {"name": "user_profiles", "last_update": time.time() - 60, "max_staleness": 300},
        {"name": "transactions", "last_update": time.time() - 500, "max_staleness": 300},
        {"name": "clickstream", "last_update": time.time() - 10, "max_staleness": 60},
    ]
    print("Проверка актуальности источников признаков:")
    for src in feature_sources:
        staleness = time.time() - src["last_update"]
        is_fresh = staleness < src["max_staleness"]
        status = "СВЕЖО" if is_fresh else "УСТАРЕЛО"
        print(f"  {src['name']}: staleness={staleness:.0f}с, "
              f"max={src['max_staleness']}с -> [{status}]")

    # --- 3.4 Кэширование признаков с TTL ---
    print("\n--- 3.4 Кэширование признаков с LRU-стратегией ---")
    # Простой LRU-кэш
    class FeatureCache:
        def __init__(self, max_size=5, ttl=10):
            self.cache = collections.OrderedDict()
            self.max_size = max_size
            self.ttl = ttl
            self.hits = 0
            self.misses = 0

        def get(self, key):
            if key in self.cache:
                val, ts = self.cache[key]
                if time.time() - ts < self.ttl:
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return val
                else:
                    del self.cache[key]  # устаревшая запись
            self.misses += 1
            return None

        def put(self, key, value):
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # удаляем самый старый
            self.cache[key] = (value, time.time())

    cache = FeatureCache(max_size=3, ttl=10)
    # Заполняем кэш
    for uid in ["u1", "u2", "u3", "u4"]:
        feats = batch_features.get(uid, {"avg_order_value": 0})
        cache.put(uid, feats)
    # Запрашиваем
    for uid in ["u2", "u4", "u1", "u5"]:
        result = cache.get(uid)
        status = f"найдено: {result}" if result else "промах (miss)"
        print(f"  get('{uid}'): {status}")
    print(f"Статистика кэша: hits={cache.hits}, misses={cache.misses}")

    print("\n[OK] Feature Serving — 4 подпримера выполнены.\n")


# ============================================================================
# Демо 4: Feature Monitoring — обнаружение дрейфа, важность признаков
# ============================================================================

def demo_feature_monitoring():
    """Демонстрация мониторинга признаков: дрейф распределений, важность."""
    print("=" * 70)
    print("ДЕМО 4: Feature Monitoring — дрейф данных и важность признаков")
    print("=" * 70)

    # --- 4.1 Population Stability Index (PSI) ---
    print("\n--- 4.1 Population Stability Index (PSI) ---")
    # Формула: PSI = sum((P_actual - P_expected) * ln(P_actual / P_expected))
    # Бакеты: делим диапазон на 5 частей
    def compute_psi(reference, current, n_bins=5):
        """Вычисление PSI между двумя выборками."""
        min_val = min(min(reference), min(current))
        max_val = max(max(reference), max(current))
        # Создаём бакеты
        edges = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
        # Подсчитываем пропорции
        ref_counts = [0] * n_bins
        cur_counts = [0] * n_bins
        for v in reference:
            for i in range(n_bins):
                if edges[i] <= v < edges[i + 1] or (i == n_bins - 1 and v == edges[-1]):
                    ref_counts[i] += 1
                    break
        for v in current:
            for i in range(n_bins):
                if edges[i] <= v < edges[i + 1] or (i == n_bins - 1 and v == edges[-1]):
                    cur_counts[i] += 1
                    break
        # Нормализуем (добавляем small epsilon чтобы избежать деления на 0)
        eps = 1e-6
        ref_prop = [(c / len(reference)) + eps for c in ref_counts]
        cur_prop = [(c / len(current)) + eps for c in cur_counts]
        # PSI = sum((A - E) * ln(A / E))
        psi = sum((cur_prop[i] - ref_prop[i]) * math.log(cur_prop[i] / ref_prop[i])
                  for i in range(n_bins))
        return psi

    # Эталонные данные (обучающая выборка)
    random.seed(42)
    reference_data = [random.gauss(50, 10) for _ in range(500)]
    # Текущие данные — без дрейфа
    current_clean = [random.gauss(50, 10) for _ in range(500)]
    # Текущие данные — с дрейфом
    random.seed(99)
    current_drift = [random.gauss(55, 15) for _ in range(500)]

    psi_clean = compute_psi(reference_data, current_clean)
    psi_drift = compute_psi(reference_data, current_drift)
    print(f"Эталон: N=500, mean≈50, std≈10")
    print(f"Текущие (без дрейфа): PSI = {psi_clean:.4f} -> {'НОРМА' if psi_clean < 0.1 else 'ДРЕЙФ'}")
    print(f"Текущие (с дрейфом): PSI = {psi_drift:.4f} -> {'НОРМА' if psi_drift < 0.1 else 'ДРЕЙФ'}")
    print(f"Пороговые значения: PSI < 0.1 (норма), 0.1-0.2 (умеренный), > 0.2 (сильный)")

    # --- 4.2 Kolmogorov-Smirnov Test ---
    print("\n--- 4.2 Kolmogorov-Smirnov (KS) Test ---")
    # KS-статистика: максимальная разность CDF двух выборок
    def empirical_cdf(data, point):
        """Эмпирическая CDF: доля значений <= point."""
        return sum(1 for x in data if x <= point) / len(data)

    def ks_statistic(sample1, sample2):
        """Вычисление KS-статистики между двумя выборками."""
        all_points = sorted(set(sample1 + sample2))
        max_diff = 0
        for p in all_points:
            cdf1 = empirical_cdf(sample1, p)
            cdf2 = empirical_cdf(sample2, p)
            diff = abs(cdf1 - cdf2)
            if diff > max_diff:
                max_diff = diff
        return max_diff

    random.seed(42)
    sample_a = [random.gauss(100, 15) for _ in range(200)]
    sample_b_clean = [random.gauss(100, 15) for _ in range(200)]
    random.seed(77)
    sample_b_drift = [random.gauss(110, 20) for _ in range(200)]

    ks_clean = ks_statistic(sample_a, sample_b_clean)
    ks_drift = ks_statistic(sample_a, sample_b_drift)
    print(f"KS-статистика (без дрейфа): {ks_clean:.4f}")
    print(f"KS-статистика (с дрейфом):  {ks_drift:.4f}")
    print(f"Критическое значение (α=0.05, N=200): ~0.097")
    print(f"Результат: {'ДРЕЙФ ОБНАРУжен' if ks_drift > 0.097 else 'Дрейф не обнаружен'}")

    # --- 4.3 Отслеживание важности признаков (Permutation Importance) ---
    print("\n--- 4.3 Permutation Importance ---")
    # Имитация модели: линейная комбинация признаков
    random.seed(42)
    n_samples = 300
    # Генерируем признаки
    features = {
        "age": [random.gauss(35, 10) for _ in range(n_samples)],
        "income": [random.gauss(50000, 15000) for _ in range(n_samples)],
        "hours_on_site": [random.gauss(10, 3) for _ in range(n_samples)],
    }
    # Целевая переменная: y = 0.5*age + 0.3*income/10000 + 0.1*hours + noise
    targets = []
    for i in range(n_samples):
        y = (0.5 * features["age"][i] +
             0.3 * features["income"][i] / 10000 +
             0.1 * features["hours_on_site"][i] +
             random.gauss(0, 0.5))
        targets.append(y)

    # Базовая ошибка модели (MSE)
    def predict(features, coeffs):
        """Линейное предсказание."""
        return [coeffs[0] * features["age"][i] +
                coeffs[1] * features["income"][i] / 10000 +
                coeffs[2] * features["hours_on_site"][i]
                for i in range(len(features["age"]))]

    def mse(preds, actuals):
        return sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(actuals)

    coeffs = [0.5, 0.3, 0.1]
    base_preds = predict(features, coeffs)
    base_error = mse(base_preds, targets)
    print(f"Базовая MSE модели: {base_error:.4f}")

    # Permutation importance: перемешиваем каждый признак по очереди
    importance = {}
    for feat_name in ["age", "income", "hours_on_site"]:
        # Копируем признак и перемешиваем
        shuffled = features[feat_name][:]
        random.shuffle(shuffled)
        permuted_features = {k: v[:] for k, v in features.items()}
        permuted_features[feat_name] = shuffled
        perm_preds = predict(permuted_features, coeffs)
        perm_error = mse(perm_preds, targets)
        importance[feat_name] = perm_error - base_error
        print(f"  Перемешан '{feat_name}': MSE={perm_error:.4f}, "
              f"importance={importance[feat_name]:.4f}")

    # Нормализуем важности
    total_imp = sum(max(v, 0) for v in importance.values())
    print(f"\nНормализованная важность признаков:")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
        pct = (max(imp, 0) / total_imp * 100) if total_imp > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"  {name:20s}: {imp:+.4f} ({pct:.1f}%) {bar}")

    # --- 4.4 Мониторинг статистик признаков во времени ---
    print("\n--- 4.4 Мониторинг статистик признаков по дням ---")
    random.seed(42)
    daily_stats = []
    for day in range(1, 8):
        # Постепенное смещение среднего (симуляция дрейфа)
        shift = day * 0.5 if day >= 5 else 0
        values = [random.gauss(50 + shift, 10) for _ in range(100)]
        daily_stats.append({
            "day": day,
            "mean": round(statistics.mean(values), 2),
            "std": round(statistics.stdev(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "count": len(values),
        })

    print(f"{'День':>5} | {'Среднее':>8} | {'Ст.откл':>8} | {'Мин':>8} | {'Макс':>8} | {'N':>4}")
    print("-" * 60)
    for s in daily_stats:
        marker = " <-- НАЧАЛО ДРЕЙФА" if s["day"] == 5 else ""
        print(f"{s['day']:5d} | {s['mean']:8.2f} | {s['std']:8.2f} | "
              f"{s['min']:8.2f} | {s['max']:8.2f} | {s['count']:4d}{marker}")

    # Простое обнаружение аномалии: среднее выходит за 2 стандартных отклонения
    baseline_means = [s["mean"] for s in daily_stats[:4]]
    baseline_avg = statistics.mean(baseline_means)
    baseline_std = statistics.stdev(baseline_means)
    print(f"\nБазовый среднее (дни 1-4): {baseline_avg:.2f}, std: {baseline_std:.2f}")
    print(f"Порог аномалии: [{baseline_avg - 2*baseline_std:.2f}, "
          f"{baseline_avg + 2*baseline_std:.2f}]")
    for s in daily_stats:
        is_anomaly = abs(s["mean"] - baseline_avg) > 2 * baseline_std
        if is_anomaly:
            print(f"  День {s['day']}: mean={s['mean']:.2f} -> АНОМАЛИЯ!")

    print("\n[OK] Feature Monitoring — 4 подпримера выполнены.\n")


# ============================================================================
# Точка входа
# ============================================================================

if __name__ == "__main__":
    demo_feature_engineering()
    demo_feature_store_architecture()
    demo_feature_serving()
    demo_feature_monitoring()
