"""211 — Cost Optimization: spot instances, auto-scaling, right-sizing

Темы:
  1. Выбор инстансов (CPU vs GPU, размер памяти, сравнение стоимости)
  2. Spot/Preemptible Instances (обработка прерываний, checkpointing, fallback)
  3. Auto-Scaling (метрики, предиктивный, по расписанию, cooldown)
  4. Cost Monitoring (тегирование, бюджеты, алерты, рекомендации)

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


# =============================================================================
# Демо 1: Выбор инстансов
# =============================================================================
def demo_instance_selection():
    """Демонстрация выбора инстансов: CPU vs GPU, memory sizing, cost comparison"""
    print("=" * 70)
    print("ДЕМО 1: ВЫБОР ИНСТАНСОВ — CPU vs GPU, размер памяти, стоимость")
    print("=" * 70)

    # --- 1.1 Каталог инстансов ---
    print("\n--- 1.1 Каталог инстансов (AWS-style) ---")

    instances = [
        {"name": "t3.medium", "type": "general", "vcpu": 2, "ram_gb": 4,
         "price_h": 0.0416, "gpu": 0, "gpu_mem": 0},
        {"name": "c5.4xlarge", "type": "compute", "vcpu": 16, "ram_gb": 32,
         "price_h": 0.68, "gpu": 0, "gpu_mem": 0},
        {"name": "r5.2xlarge", "type": "memory", "vcpu": 8, "ram_gb": 64,
         "price_h": 0.504, "gpu": 0, "gpu_mem": 0},
        {"name": "p3.2xlarge", "type": "gpu", "vcpu": 8, "ram_gb": 61,
         "price_h": 3.06, "gpu": 1, "gpu_mem": 16},
        {"name": "p3.8xlarge", "type": "gpu", "vcpu": 32, "ram_gb": 244,
         "price_h": 12.24, "gpu": 4, "gpu_mem": 64},
        {"name": "p4d.24xlarge", "type": "gpu", "vcpu": 96, "ram_gb": 1152,
         "price_h": 32.77, "gpu": 8, "gpu_mem": 320},
        {"name": "g5.xlarge", "type": "gpu", "vcpu": 4, "ram_gb": 16,
         "price_h": 1.006, "gpu": 1, "gpu_mem": 24},
        {"name": "g5.4xlarge", "type": "gpu", "vcpu": 16, "ram_gb": 64,
         "price_h": 2.036, "gpu": 1, "gpu_mem": 24},
    ]

    print(f"  {'Name':<15} {'Type':<10} {'vCPU':<6} {'RAM':<8} {'GPU':<5} {'GPU Mem':<9} {'$/hour':<8}")
    print("  " + "-" * 62)
    for inst in instances:
        gpu_str = str(inst["gpu"]) if inst["gpu"] > 0 else "-"
        gpu_mem_str = f"{inst['gpu_mem']} GB" if inst["gpu_mem"] > 0 else "-"
        print(f"  {inst['name']:<15} {inst['type']:<10} {inst['vcpu']:<6} "
              f"{inst['ram_gb']:<8} {gpu_str:<5} {gpu_mem_str:<9} ${inst['price_h']:.4f}")

    # --- 1.2 Стоимость за месяц ---
    print("\n--- 1.2 Стоимость за месяц (730 часов) ---")

    hours_per_month = 730
    print(f"  {'Name':<15} {'Часовая':<12} {'Месячная':<14} {'Годовая':<14}")
    print("  " + "-" * 55)
    for inst in instances:
        monthly = inst["price_h"] * hours_per_month
        yearly = monthly * 12
        print(f"  {inst['name']:<15} ${inst['price_h']:<11.4f} ${monthly:<13.2f} ${yearly:<13.2f}")

    # --- 1.3 Cost per vCPU ---
    print("\n--- 1.3 Стоимость за vCPU и за GB RAM ---")

    print(f"  {'Name':<15} {'$/vCPU/h':<12} {'$/GB/h':<12} {'$/vCPU/mo':<12} {'$/GB/mo':<12}")
    print("  " + "-" * 60)
    for inst in instances:
        per_vcpu = inst["price_h"] / inst["vcpu"] if inst["vcpu"] > 0 else 0
        per_gb = inst["price_h"] / inst["ram_gb"] if inst["ram_gb"] > 0 else 0
        print(f"  {inst['name']:<15} ${per_vcpu:<11.4f} ${per_gb:<11.4f} "
              f"${per_vcpu * hours_per_month:<11.2f} ${per_gb * hours_per_month:<11.2f}")

    # --- 1.4 Правило выбора ---
    print("\n--- 1.4 Правило выбора инстанса ---")

    workloads = [
        {"name": "ML Training (large model)", "ram_gb": 64, "gpu_needed": True,
         "gpu_mem_gb": 40, "priority": "gpu"},
        {"name": "Data preprocessing", "ram_gb": 256, "gpu_needed": False,
         "gpu_mem_gb": 0, "priority": "memory"},
        {"name": "Web API serving", "ram_gb": 8, "gpu_needed": False,
         "gpu_mem_gb": 0, "priority": "cpu"},
        {"name": "Inference (small model)", "ram_gb": 16, "gpu_needed": True,
         "gpu_mem_gb": 8, "priority": "gpu"},
        {"name": "Fine-tuning LLM", "ram_gb": 128, "gpu_needed": True,
         "gpu_mem_gb": 80, "priority": "gpu"},
    ]

    for wl in workloads:
        # Находим подходящий инстанс
        candidates = []
        for inst in instances:
            if inst["ram_gb"] >= wl["ram_gb"]:
                if wl["gpu_needed"]:
                    if inst["gpu"] > 0 and inst["gpu_mem"] >= wl["gpu_mem_gb"]:
                        candidates.append(inst)
                else:
                    candidates.append(inst)

        if candidates:
            best = min(candidates, key=lambda x: x["price_h"])
            gpu_req = ""
            if wl["gpu_needed"]:
                gpu_req = f", GPU ≥ {wl['gpu_mem_gb']} GB"
            print(f"\n  {wl['name']}:")
            print(f"    Требования: RAM ≥ {wl['ram_gb']} GB{gpu_req}")
            print(f"    → Рекомендация: {best['name']} (${best['price_h']:.4f}/h)")
        else:
            print(f"\n  {wl['name']}: нет подходящего инстанса в каталоге!")

    print("\n" + "=" * 70)
    print("ВЫВОД: Правильный выбор инстанса экономит 30-60% от стоимости")
    print("=" * 70)


# =============================================================================
# Демо 2: Spot/Preemptible Instances
# =============================================================================
def demo_spot_instances():
    """Демонстрация spot instances: прерывания, checkpointing, fallback"""
    print("\n" + "=" * 70)
    print("ДЕМО 2: SPOT/PREEMPTIBLE INSTANCES — прерывания, checkpointing, fallback")
    print("=" * 70)

    # --- 2.1 Экономия на spot ---
    print("\n--- 2.1 Экономия: On-Demand vs Spot ---")

    pricing = {
        "p3.2xlarge": {"on_demand": 3.06, "spot": 0.92},
        "g5.xlarge": {"on_demand": 1.006, "spot": 0.30},
        "c5.4xlarge": {"on_demand": 0.68, "spot": 0.20},
        "p4d.24xlarge": {"on_demand": 32.77, "spot": 9.83},
    }

    print(f"  {'Instance':<18} {'On-Demand':<12} {'Spot':<12} {'Экономия':<12}")
    print("  " + "-" * 52)
    for name, prices in pricing.items():
        savings = (1 - prices["spot"] / prices["on_demand"]) * 100
        print(f"  {name:<18} ${prices['on_demand']:<11.2f} ${prices['spot']:<11.2f} {savings:.0f}%")

    # --- 2.2 Модель прерываний ---
    print("\n--- 2.2 Модель прерываний Spot ---")

    # Симуляция spot instance lifecycle
    random.seed(42)

    class SpotInstanceSimulator:
        """Симулятор жизненного цикла spot instance"""

        def __init__(self, interruption_rate_per_hour=0.05):
            self.interruption_rate = interruption_rate_per_hour
            self.downtime_minutes = 5  # время на перезапуск
            self.checkpoint_interval_min = 10  # интервал checkpoint'ов

        def simulate_session(self, hours=24, checkpoint_enabled=True):
            """Симуляция сессии с возможными прерываниями"""
            total_minutes = hours * 60
            current_minute = 0
            interruptions = 0
            checkpoints_saved = 0
            checkpoints_lost = 0
            lost_work_minutes = 0

            while current_minute < total_minutes:
                # Вероятность прерывания в текущей минуте
                if random.random() < self.interruption_rate / 60:
                    interruptions += 1

                    # Проверяем, был ли checkpoint
                    last_checkpoint = (current_minute // self.checkpoint_interval_min) * self.checkpoint_interval_min
                    minutes_since_checkpoint = current_minute - last_checkpoint

                    if checkpoint_enabled and minutes_since_checkpoint <= self.checkpoint_interval_min:
                        # Есть checkpoint — теряем только منذ последнего checkpoint
                        lost_work_minutes += minutes_since_checkpoint
                        checkpoints_lost += 1
                    else:
                        # Нет checkpoint — теряем всю работу с последнего checkpoint
                        lost_work_minutes += self.checkpoint_interval_min
                        checkpoints_lost += 1

                    # Перезапуск
                    current_minute += self.downtime_minutes
                else:
                    current_minute += 1

                # Периодический checkpoint
                if checkpoint_enabled and current_minute % self.checkpoint_interval_min == 0:
                    checkpoints_saved += 1

            uptime_pct = (total_minutes - interruptions * self.downtime_minutes) / total_minutes * 100
            return {
                "interruptions": interruptions,
                "checkpoints_saved": checkpoints_saved,
                "checkpoints_lost": checkpoints_lost,
                "lost_work_minutes": lost_work_minutes,
                "uptime_pct": uptime_pct,
            }

    sim = SpotInstanceSimulator(interruption_rate_per_hour=0.05)

    print("  Симуляция 24ч работы spot instance:")
    print()

    for checkpoint in [False, True]:
        result = sim.simulate_session(hours=24, checkpoint_enabled=checkpoint)
        mode = "БЕЗ checkpoint" if not checkpoint else "С checkpoint"
        print(f"  {mode}:")
        print(f"    Прерываний: {result['interruptions']}")
        print(f"    Checkpoints создано: {result['checkpoints_saved']}")
        print(f"    Потеряно работы: {result['lost_work_minutes']} мин")
        print(f"    Uptime: {result['uptime_pct']:.1f}%")
        print()

    # --- 2.3 Checkpointing стратегии ---
    print("--- 2.3 Стратегии Checkpointing ---")

    strategies = [
        {
            "name": "Periodic (каждые N шагов)",
            "interval": "100 шагов",
            "overhead": "~2% времени",
            "loss": "до 100 шагов",
            "use_case": "Обучение моделей"
        },
        {
            "name": "Delta (инкрементальный)",
            "interval": "При накоплении изменений",
            "overhead": "~0.5% времени",
            "loss": "Минимальная",
            "use_case": "Долгие inference сессии"
        },
        {
            "name": "Event-driven",
            "interval": "При的重要 milestones",
            "overhead": "~1% времени",
            "loss": "Зависит от frequency",
            "use_case": "Pipeline с чёткими этапами"
        },
        {
            "name": "Async (отдельный поток)",
            "interval": "Фоновый процесс",
            "overhead": "~0% (не блокирует)",
            "loss": "До interval checkpoint'а",
            "use_case": "Высоконагруженные системы"
        },
    ]

    for s in strategies:
        print(f"\n  {s['name']}:")
        print(f"    Интервал: {s['interval']}")
        print(f"    Overhead: {s['overhead']}")
        print(f"    Потеря: {s['loss']}")
        print(f"    Use case: {s['use_case']}")

    # --- 2.4 Fallback стратегии ---
    print("\n--- 2.4 Fallback стратегии ---")

    fallbacks = [
        {"name": "On-Demand fallback", "cost_multiplier": "3.3x",
         "reliability": "100%", "use_when": "Критичные workload'ы"},
        {"name": "Multi-AZ spot", "cost_multiplier": "1.1x",
         "reliability": "95%", "use_when": "Распределённые задачи"},
        {"name": "Reserved + Spot mix", "cost_multiplier": "1.5x",
         "reliability": "99%", "use_when": "Предсказуемый baseline"},
        {"name": "Spot Fleet", "cost_multiplier": "1.2x",
         "reliability": "90%", "use_when": "Гибкие требования"},
    ]

    print(f"  {'Стратегия':<25} {'Стоимость':<12} {'Надёжность':<12} {'Когда использовать'}")
    print("  " + "-" * 75)
    for f in fallbacks:
        print(f"  {f['name']:<25} {f['cost_multiplier']:<12} {f['reliability']:<12} {f['use_when']}")

    # ROI расчёт
    print("\n  ROI: Spot + Checkpointing vs On-Demand")
    on_demand_cost = 3.06  # p3.2xlarge
    spot_cost = 0.92
    checkpoint_overhead = 0.02
    interruption_cost = 5 * 0.02  # ~5 прерываний/день × стоимость

    monthly_on_demand = on_demand_cost * 730
    monthly_spot = spot_cost * 730 * (1 + checkpoint_overhead) + interruption_cost * 30

    print(f"  On-Demand: ${monthly_on_demand:.0f}/мес")
    print(f"  Spot + Checkpointing: ${monthly_spot:.0f}/мес")
    print(f"  Экономия: ${monthly_on_demand - monthly_spot:.0f}/мес ({(1 - monthly_spot/monthly_on_demand)*100:.0f}%)")

    print("\n" + "=" * 70)
    print("ВЫВОД: Spot instances экономят 60-70%, checkpointing снижает риск потерь")
    print("=" * 70)


# =============================================================================
# Демо 3: Auto-Scaling
# =============================================================================
def demo_autoscaling():
    """Демонстрация auto-scaling: метрики, предиктивный, расписание, cooldown"""
    print("\n" + "=" * 70)
    print("ДЕМО 3: AUTO-SCALING — метрики, предиктивный, расписание, cooldown")
    print("=" * 70)

    # --- 3.1 Базовый auto-scaling ---
    print("\n--- 3.1 Базовый Auto-Scaling (metric-based) ---")

    class AutoScaler:
        """Модель auto-scaler на основе метрик"""

        def __init__(self, min_instances=1, max_instances=10,
                     target_utilization=70, cooldown_seconds=300):
            self.min_instances = min_instances
            self.max_instances = max_instances
            self.target_utilization = target_utilization
            self.cooldown = cooldown_seconds
            self.instances = min_instances
            self.last_scale_time = 0

        def evaluate(self, current_utilization, current_time):
            """Оценка необходимости масштабирования"""
            if current_time - self.last_scale_time < self.cooldown:
                return self.instances, "cooldown"

            desired = self.instances

            if current_utilization > self.target_utilization + 10:
                # Нужно добавить инстансы
                desired = math.ceil(self.instances * (current_utilization / self.target_utilization))
                desired = min(desired, self.max_instances)
            elif current_utilization < self.target_utilization - 20:
                # Можно убрать инстансы
                desired = max(self.min_instances, self.instances - 1)

            if desired != self.instances:
                self.instances = desired
                self.last_scale_time = current_time
                return desired, "scaled"
            return self.instances, "stable"

    scaler = AutoScaler(min_instances=2, max_instances=8, target_utilization=70)

    # Симуляция: нарастание нагрузки
    print(f"  Конфигурация: min=2, max=8, target=70%, cooldown=300s")
    print()

    workload = [30, 35, 45, 60, 75, 85, 95, 90, 70, 50, 35, 25]
    print(f"  {'Time':<6} {'Utilization':<14} {'Instances':<12} {'Status'}")
    print("  " + "-" * 45)

    for t, util in enumerate(workload):
        instances, status = scaler.evaluate(util, t * 60)
        util_bar = "█" * (util // 5)
        print(f"  {t*60:>5}s {util:>5}% {util_bar:<20} {instances:>3} {status}")

    # --- 3.2 Предиктивный auto-scaling ---
    print("\n--- 3.2 Предиктивный Auto-Scaling ---")

    # Модель: анализ паттернов нагрузки
    print("  Анализ паттерна нагрузки за неделю:")
    print()

    # Симуляция: рабочие часы vs ночь
    hourly_pattern = []
    for hour in range(24):
        if 9 <= hour <= 18:
            base = 70 + random.uniform(-5, 15)  # рабочие часы
        elif 6 <= hour <= 9 or 18 <= hour <= 22:
            base = 40 + random.uniform(-5, 10)  # утро/вечер
        else:
            base = 15 + random.uniform(-5, 10)  # ночь
        hourly_pattern.append(base)

    print("  Час | Нагрузка% | Bar")
    print("  " + "-" * 40)
    for hour, load in enumerate(hourly_pattern):
        bar = "█" * int(load / 3)
        print(f"  {hour:>3} | {load:>5.1f}%   | {bar}")

    # Предсказание: следующие 6 часов
    print("\n  Предсказание на следующие 6 часов (linear extrapolation):")
    recent_avg = sum(hourly_pattern[-6:]) / 6
    trend = (hourly_pattern[-1] - hourly_pattern[-6]) / 6

    for h in range(1, 7):
        predicted = recent_avg + trend * h
        recommended_instances = max(1, math.ceil(predicted / 25))  # 1 инстанс на 25%
        print(f"  +{h}h: predicted={predicted:.1f}%, instances={recommended_instances}")

    # --- 3.3 Scheduled scaling ---
    print("\n--- 3.3 Scheduled Scaling (по расписанию) ---")

    schedule = [
        {"time": "00:00", "action": "scale_down", "instances": 2, "reason": "ночь"},
        {"time": "06:00", "action": "scale_up", "instances": 4, "reason": "утренний пик"},
        {"time": "09:00", "action": "scale_up", "instances": 6, "reason": "рабочий день"},
        {"time": "12:00", "action": "maintain", "instances": 6, "reason": "обеденный пик"},
        {"time": "18:00", "action": "scale_down", "instances": 3, "reason": "вечер"},
        {"time": "22:00", "action": "scale_down", "instances": 2, "reason": "ночь"},
    ]

    print(f"  {'Time':<8} {'Action':<12} {'Instances':<12} {'Reason'}")
    print("  " + "-" * 50)
    for entry in schedule:
        print(f"  {entry['time']:<8} {entry['action']:<12} {entry['instances']:<12} {entry['reason']}")

    # --- 3.4 Cooldown и stability ---
    print("\n--- 3.4 Cooldown и Stability ---")

    print("  Cooldown предотвращает 'flapping':")
    print("  1. Scale up → ждём cooldown → проверяем метрику")
    print("  2. Если метрика всё высокая → scale up ещё")
    print("  3. Если метрика упала → ждём cooldown → scale down")

    # Модель cooldown
    cooldown_seconds = 300
    events = [
        (0, "scale_up", 60, "load=85%"),
        (180, "check", 60, "load=90% (всё high)"),
        (360, "cooldown", 60, "ожидание..."),
        (480, "scale_up", 60, "load=92%"),
        (600, "check", 60, "load=50% (низкая)"),
        (780, "cooldown", 60, "ожидание..."),
        (900, "scale_down", 60, "load=45%"),
    ]

    print("\n  Timeline:")
    for t, action, duration, note in events:
        print(f"  t={t:>4}s: {action:<12} {note}")

    print(f"\n  Без cooldown: 5+ scale операций за 15 минут")
    print(f"  С cooldown ({cooldown_seconds}s): 2-3 scale операции за 15 минут")

    print("\n" + "=" * 70)
    print("ВЫВОД: Auto-scaling + cooldown = стабильная стоимость + адаптивность")
    print("=" * 70)


# =============================================================================
# Демо 4: Cost Monitoring
# =============================================================================
def demo_cost_monitoring():
    """Демонстрация cost monitoring: тегирование, бюджеты, алерты"""
    print("\n" + "=" * 70)
    print("ДЕМО 4: COST MONITORING — тегирование, бюджеты, алерты")
    print("=" * 70)

    # --- 4.1 Тегирование ресурсов ---
    print("\n--- 4.1 Тегирование ресурсов ---")

    resources = [
        {"id": "i-001", "type": "EC2", "name": "ml-training-gpu",
         "tags": {"team": "ml", "project": "llm-finetune", "env": "prod", "cost-center": "research"}},
        {"id": "i-002", "type": "EC2", "name": "api-server",
         "tags": {"team": "backend", "project": "inference", "env": "prod", "cost-center": "production"}},
        {"id": "i-003", "type": "S3", "name": "model-artifacts",
         "tags": {"team": "ml", "project": "llm-finetune", "env": "prod", "cost-center": "storage"}},
        {"id": "r-001", "type": "RDS", "name": "user-db",
         "tags": {"team": "backend", "project": "inference", "env": "prod", "cost-center": "production"}},
    ]

    print("  Ресурсы с тегами:")
    for r in resources:
        tags_str = ", ".join(f"{k}={v}" for k, v in r["tags"].items())
        print(f"    {r['id']} ({r['type']}) {r['name']}: {tags_str}")

    # Группировка по тегам
    print("\n  Расходы по командам:")
    costs = {"ml": 2400, "backend": 800, "devops": 300}
    total = sum(costs.values())
    for team, cost in sorted(costs.items(), key=lambda x: -x[1]):
        pct = cost / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {team:<10} ${cost:>6} ({pct:>5.1f}%) {bar}")

    # --- 4.2 Бюджеты и алерты ---
    print("\n--- 4.2 Бюджеты и алерты ---")

    budgets = [
        {"name": "ML Training", "monthly_limit": 5000, "current_spend": 3200,
         "alert_thresholds": [50, 80, 100]},
        {"name": "API Serving", "monthly_limit": 2000, "current_spend": 1100,
         "alert_thresholds": [50, 80, 100]},
        {"name": "Storage", "monthly_limit": 500, "current_spend": 480,
         "alert_thresholds": [50, 80, 100]},
    ]

    for b in budgets:
        usage_pct = b["current_spend"] / b["monthly_limit"] * 100
        remaining = b["monthly_limit"] - b["current_spend"]

        print(f"\n  Бюджет: {b['name']}")
        print(f"    Лимит: ${b['monthly_limit']:,}")
        print(f"    Потрачено: ${b['current_spend']:,} ({usage_pct:.1f}%)")
        print(f"    Остаток: ${remaining:,}")

        # Проверка алертов
        for threshold in b["alert_thresholds"]:
            if usage_pct >= threshold:
                print(f"    ⚠ ALERT: {threshold}% threshold breached!")

        # Визуализация
        filled = int(usage_pct / 5)
        empty = 20 - filled
        bar = "█" * filled + "░" * empty
        print(f"    Прогресс: [{bar}] {usage_pct:.1f}%")

    # --- 4.3 Анализ стоимости по компонентам ---
    print("\n--- 4.3 Анализ стоимости по компонентам ---")

    cost_breakdown = {
        "GPU инстансы": {"cost": 8500, "pct": 55.2},
        "CPU инстансы": {"cost": 2100, "pct": 13.7},
        "Хранилище (S3)": {"cost": 800, "pct": 5.2},
        "Базы данных (RDS)": {"cost": 1200, "pct": 7.8},
        "Сеть (data transfer)": {"cost": 950, "pct": 6.2},
        "Load Balancers": {"cost": 350, "pct": 2.3},
        "Другое": {"cost": 1480, "pct": 9.6},
    }

    print(f"  {'Компонент':<25} {'Стоимость':<12} {'Доля':<8} {'Визуализация'}")
    print("  " + "-" * 65)
    for name, data in sorted(cost_breakdown.items(), key=lambda x: -x[1]["cost"]):
        bar = "█" * int(data["pct"] / 2)
        print(f"  {name:<25} ${data['cost']:<11,} {data['pct']:>5.1f}%  {bar}")

    total_cost = sum(d["cost"] for d in cost_breakdown.values())
    print(f"\n  ИТОГО: ${total_cost:,}/мес")

    # --- 4.4 Рекомендации по оптимизации ---
    print("\n--- 4.4 Рекомендации по оптимизации ---")

    recommendations = [
        {
            "action": "Перевод на Spot Instances",
            "savings": 3200,
            "effort": "Низкая",
            "risk": "Низкий",
            "details": "GPU инстансы → spot (70% savings)"
        },
        {
            "action": "Right-sizing GPU инстансов",
            "savings": 1500,
            "effort": "Средняя",
            "risk": "Средний",
            "details": "Используется 40% GPU memory → уменьшить тип"
        },
        {
            "action": "S3 Lifecycle Policies",
            "savings": 400,
            "effort": "Низкая",
            "risk": "Низкий",
            "details": "Старые артефакты → Glacier после 30 дней"
        },
        {
            "action": "Reserved Instances (1 год)",
            "savings": 1200,
            "effort": "Низкая",
            "risk": "Средний",
            "details": "Базовая нагрузка → RI (30% savings)"
        },
        {
            "action": "Auto-scaling оптимизация",
            "savings": 800,
            "effort": "Средняя",
            "risk": "Низкий",
            "details": "Уменьшить cooldown, улучшить метрики"
        },
    ]

    print(f"  {'Действие':<30} {'Экономия':<12} {'Сложность':<12} {'Риск'}")
    print("  " + "-" * 70)
    for rec in recommendations:
        print(f"  {rec['action']:<30} ${rec['savings']:<11,} {rec['effort']:<12} {rec['risk']}")

    total_savings = sum(r["savings"] for r in recommendations)
    print(f"\n  Потенциальная экономия: ${total_savings:,}/мес ({total_savings/total_cost*100:.0f}%)")
    print(f"  После оптимизации: ${total_cost - total_savings:,}/мес")

    # ROI расчёта
    print("\n  ROI внедрения:")
    implementation_hours = 40
    hourly_rate = 75
    implementation_cost = implementation_hours * hourly_rate
    monthly_savings = total_savings
    payback_months = implementation_cost / monthly_savings

    print(f"    Время внедрения: {implementation_hours} ч × ${hourly_rate}/ч = ${implementation_cost:,}")
    print(f"    Ежемесячная экономия: ${monthly_savings:,}")
    print(f"    Окупаемость: {payback_months:.1f} месяцев")

    print("\n" + "=" * 70)
    print("ВЫВОД: Cost monitoring + optimization = 30-50% снижение расходов")
    print("=" * 70)


# =============================================================================
# Запуск всех демонстраций
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("УРОК 211: COST OPTIMIZATION")
    print("Spot instances, auto-scaling, right-sizing")
    print("=" * 70)
    print()

    demo_instance_selection()
    demo_spot_instances()
    demo_autoscaling()
    demo_cost_monitoring()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 70)
