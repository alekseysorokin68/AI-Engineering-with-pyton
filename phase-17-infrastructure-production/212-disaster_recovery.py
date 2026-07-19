"""212 — Disaster Recovery: резервное копирование, переключение, хаос-инженерия

Темы:
  1. Стратегии резервного копирования (full, incremental, point-in-time recovery)
  2. Шаблоны переключения (active-passive, active-active, DNS failover)
  3. Хаос-инженерия (game day, fault injection, steady state hypothesis)
  4. RTO/RPO (recovery time objective, recovery point objective, планирование)

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
# Демо 1: Стратегии резервного копирования
# =============================================================================
def demo_backup_strategies():
    """Демонстрация стратегий backup: full, incremental, point-in-time recovery"""
    print("=" * 70)
    print("ДЕМО 1: СТРАТЕГИИ РЕЗЕРВНОГО КОПИРОВАНИЯ — full, incremental, PITR")
    print("=" * 70)

    # --- 1.1 Full vs Incremental vs Differential ---
    print("\n--- 1.1 Full vs Incremental vs Differential ---")

    # Модель файловой системы для backup
    class BackupSimulator:
        """Симулятор стратегий резервного копирования"""

        def __init__(self, total_size_gb=100):
            self.total_size = total_size_gb
            self.daily_change_pct = 5  # % данных меняется ежедневно
            self.backups = []

        def simulate_full_backup(self, days=7):
            """Модель: Full backup каждый день"""
            results = []
            for day in range(days):
                backup_size = self.total_size
                time_minutes = backup_size * 0.5  # ~0.5 мин/GB
                results.append({
                    "day": day,
                    "type": "full",
                    "size_gb": backup_size,
                    "time_min": time_minutes
                })
            return results

        def simulate_incremental(self, days=7):
            """Модель: Full + daily incremental"""
            results = []
            prev_backup = set()

            for day in range(days):
                if day == 0:
                    # Full backup
                    backup_size = self.total_size
                    type_ = "full"
                    prev_backup = set(range(int(self.total_size)))
                else:
                    # Incremental: только изменения
                    changed = int(self.total_size * self.daily_change_pct / 100)
                    backup_size = changed
                    type_ = "incremental"
                    prev_backup = set(range(changed))

                time_minutes = backup_size * 0.5
                results.append({
                    "day": day,
                    "type": type_,
                    "size_gb": backup_size,
                    "time_min": time_minutes
                })
            return results

        def simulate_differential(self, days=7):
            """Модель: Full + daily differential (от последнего full)"""
            results = []
            cumulative_changes = 0

            for day in range(days):
                if day == 0:
                    backup_size = self.total_size
                    type_ = "full"
                    cumulative_changes = 0
                else:
                    cumulative_changes += int(self.total_size * self.daily_change_pct / 100)
                    backup_size = cumulative_changes
                    type_ = "differential"

                time_minutes = backup_size * 0.5
                results.append({
                    "day": day,
                    "type": type_,
                    "size_gb": backup_size,
                    "time_min": time_minutes
                })
            return results

    sim = BackupSimulator(total_size_gb=100)

    # Full backup
    full_results = sim.simulate_full_backup(7)
    print("  Full Backup (ежедневно):")
    print(f"  {'День':<6} {'Тип':<12} {'Размер':<12} {'Время'}")
    print("  " + "-" * 40)
    for r in full_results:
        print(f"  {r['day']:<6} {r['type']:<12} {r['size_gb']:<10} GB  {r['time_min']:.1f} мин")

    total_full = sum(r['size_gb'] for r in full_results)
    print(f"  ИТОГО за 7 дней: {total_full} GB\n")

    # Incremental
    inc_results = sim.simulate_incremental(7)
    print("  Incremental Backup:")
    print(f"  {'День':<6} {'Тип':<14} {'Размер':<12} {'Время'}")
    print("  " + "-" * 42)
    for r in inc_results:
        print(f"  {r['day']:<6} {r['type']:<14} {r['size_gb']:<10} GB  {r['time_min']:.1f} мин")

    total_inc = sum(r['size_gb'] for r in inc_results)
    print(f"  ИТОГО за 7 дней: {total_inc} GB\n")

    # Differential
    diff_results = sim.simulate_differential(7)
    print("  Differential Backup:")
    print(f"  {'День':<6} {'Тип':<16} {'Размер':<12} {'Время'}")
    print("  " + "-" * 44)
    for r in diff_results:
        print(f"  {r['day']:<6} {r['type']:<16} {r['size_gb']:<10} GB  {r['time_min']:.1f} мин")

    total_diff = sum(r['size_gb'] for r in diff_results)
    print(f"  ИТОГО за 7 дней: {total_diff} GB")

    # --- 1.2 Recovery Time ---
    print("\n--- 1.2 Recovery Time (время восстановления) ---")

    print("  Full:    восстановить 1 файл → быстро (1 backup)")
    print("  Incremental: восстановить 1 файл → медленно (chain of backups)")
    print("  Differential: восстановить 1 файл → среднее (2 backups)\n")

    # Модель recovery time
    restore_speed_gb_min = 2  # скорость восстановления
    days_to_restore = 5  # нужно восстановить до 5-го дня

    # Full: берём бэкап дня 5
    full_restore = full_results[4]['size_gb'] / restore_speed_gb_min
    print(f"  Full (день 5): {full_results[4]['size_gb']} GB / {restore_speed_gb_min} GB/мин = {full_restore:.1f} мин")

    # Incremental: full + 4 incremental
    inc_restore_size = inc_results[0]['size_gb'] + sum(r['size_gb'] for r in inc_results[1:5])
    inc_restore = inc_restore_size / restore_speed_gb_min
    print(f"  Incremental (день 5): {inc_restore_size} GB / {restore_speed_gb_min} GB/мин = {inc_restore:.1f} мин")

    # Differential: full + differential
    diff_restore_size = diff_results[0]['size_gb'] + diff_results[4]['size_gb']
    diff_restore = diff_restore_size / restore_speed_gb_min
    print(f"  Differential (день 5): {diff_restore_size} GB / {restore_speed_gb_min} GB/мин = {diff_restore:.1f} мин")

    # --- 1.3 Point-in-Time Recovery ---
    print("\n--- 1.3 Point-in-Time Recovery (PITR) ---")

    print("  PITR = восстановление к конкретному моменту времени")
    print("  Требует: WAL (Write-Ahead Log) или transaction log\n")

    # Модель WAL
    wal_entries = [
        {"time": "10:00:00", "lsn": 1000, "op": "INSERT users", "size": 1024},
        {"time": "10:00:05", "lsn": 1001, "op": "UPDATE accounts", "size": 512},
        {"time": "10:00:10", "lsn": 1002, "op": "INSERT orders", "size": 2048},
        {"time": "10:00:15", "lsn": 1003, "op": "DELETE sessions", "size": 256},
        {"time": "10:00:20", "lsn": 1004, "op": "UPDATE inventory", "size": 1024},
    ]

    print("  WAL записи:")
    print(f"  {'Time':<12} {'LSN':<8} {'Operation':<20} {'Size'}")
    print("  " + "-" * 50)
    for entry in wal_entries:
        print(f"  {entry['time']:<12} {entry['lsn']:<8} {entry['op']:<20} {entry['size']} bytes")

    # Восстановление к 10:00:12
    target_time = "10:00:12"
    print(f"\n  Восстановление к {target_time}:")
    print("  1. Восстановить full backup")
    print("  2. Применить WAL до 10:00:12")
    print("  3. Пропустить записи после 10:00:12")

    applicable = [e for e in wal_entries if e["time"] <= target_time]
    print(f"  Применено записей: {len(applicable)}/{len(wal_entries)}")

    # --- 1.4 Сравнение стратегий ---
    print("\n--- 1.4 Сравнение стратегий ---")

    strategies = [
        {
            "name": "Full Daily",
            "storage": "Высокая",
            "backup_time": "Долгое",
            "recovery_time": "Быстрое",
            "complexity": "Низкая",
            "best_for": "Критичные данные, маленькие объёмы"
        },
        {
            "name": "Full + Incremental",
            "storage": "Низкая",
            "backup_time": "Быстрое",
            "recovery_time": "Долгое",
            "complexity": "Средняя",
            "best_for": "Большие объёмы, нечастое восстановление"
        },
        {
            "name": "Full + Differential",
            "storage": "Средняя",
            "backup_time": "Среднее",
            "recovery_time": "Среднее",
            "complexity": "Средняя",
            "best_for": "Баланс между скоростью и временем"
        },
        {
            "name": "PITR (WAL)",
            "storage": "Зависит от retention",
            "backup_time": "Непрерывно",
            "recovery_time": "Точное",
            "complexity": "Высокая",
            "best_for": "Транзакционные системы"
        },
    ]

    for s in strategies:
        print(f"\n  {s['name']}:")
        print(f"    Хранилище: {s['storage']}")
        print(f"    Время backup: {s['backup_time']}")
        print(f"    Время восстановления: {s['recovery_time']}")
        print(f"    Сложность: {s['complexity']}")
        print(f"    Best for: {s['best_for']}")

    print("\n" + "=" * 70)
    print("ВЫВОД: Выбор стратегии зависит от RTO/RPO и размера данных")
    print("=" * 70)


# =============================================================================
# Демо 2: Шаблоны переключения (Failover)
# =============================================================================
def demo_failover_patterns():
    """Демонстрация failover: active-passive, active-active, DNS failover"""
    print("\n" + "=" * 70)
    print("ДЕМО 2: ШАБЛОНЫ ПЕРЕКЛЮЧЕНИЯ — active-passive, active-active, DNS failover")
    print("=" * 70)

    # --- 2.1 Active-Passive ---
    print("\n--- 2.1 Active-Passive Failover ---")

    class ActivePassiveCluster:
        """Модель active-passive кластера"""

        def __init__(self):
            self.primary = {"status": "active", "health": 100, "name": "Primary"}
            self.secondary = {"status": "passive", "health": 100, "name": "Secondary"}
            self.failover_count = 0
            self.failback_count = 0

        def health_check(self):
            """Проверка здоровья primary"""
            return self.primary["health"] > 0

        def failover(self):
            """Переключение на secondary"""
            if self.primary["status"] == "active":
                self.primary["status"] = "passive"
                self.secondary["status"] = "active"
                self.failover_count += 1
                return True
            return False

        def failback(self):
            """Возврат на primary"""
            if self.secondary["status"] == "active" and self.primary["health"] > 50:
                self.secondary["status"] = "passive"
                self.primary["status"] = "active"
                self.failback_count += 1
                return True
            return False

    cluster = ActivePassiveCluster()

    # Симуляция
    events = [
        {"time": "10:00", "action": "health_check", "primary_health": 100},
        {"time": "10:05", "action": "primary_degradation", "primary_health": 50},
        {"time": "10:10", "action": "health_check", "primary_health": 50},
        {"time": "10:15", "action": "primary_failure", "primary_health": 0},
        {"time": "10:20", "action": "failover", "primary_health": 0},
        {"time": "10:30", "action": "primary_recovery", "primary_health": 80},
        {"time": "10:35", "action": "failback", "primary_health": 80},
    ]

    print("  Сценарий: Active-Passive с health checks\n")
    for event in events:
        cluster.primary["health"] = event["primary_health"]

        if event["action"] == "failover":
            cluster.failover()
        elif event["action"] == "failback":
            cluster.failback()

        active = "Primary" if cluster.primary["status"] == "active" else "Secondary"
        print(f"  {event['time']}: {event['action']:<20} "
              f"primary={event['primary_health']:>3}%  active → {active}")

    print(f"\n  Failovers: {cluster.failover_count}, Failbacks: {cluster.failback_count}")

    # --- 2.2 Active-Active ---
    print("\n--- 2.2 Active-Active Failover ---")

    print("  Active-Active: оба узла обрабатывают трафик\n")

    # Модель: распределение нагрузки
    nodes = [
        {"name": "Node-A", "capacity": 100, "load": 0, "status": "active"},
        {"name": "Node-B", "capacity": 100, "load": 0, "status": "active"},
    ]

    def distribute_traffic(nodes, total_traffic):
        """Распределение трафика между узлами"""
        active_nodes = [n for n in nodes if n["status"] == "active"]
        if not active_nodes:
            return 0

        per_node = total_traffic / len(active_nodes)
        for node in active_nodes:
            node["load"] = min(per_node, node["capacity"])
        return total_traffic

    # Сценарий: рост трафика
    traffic_levels = [50, 100, 120, 150, 180]
    print(f"  {'Traffic':<10} ", end="")
    for n in nodes:
        print(f"{n['name']:<12} ", end="")
    print("Status")
    print("  " + "-" * 55)

    for traffic in traffic_levels:
        distribute_traffic(nodes, traffic)
        statuses = " + ".join(f"{n['name']}={n['load']:.0f}/{n['capacity']}" for n in nodes)
        print(f"  {traffic:<10} ", end="")
        for n in nodes:
            print(f"{n['load']:>5.0f}/{n['capacity']:<5}    ", end="")
        print(f"  {statuses}")

    # Fail其中一个 узла
    print("\n  Node-A падает:")
    nodes[0]["status"] = "failed"
    distribute_traffic(nodes, 150)

    for n in nodes:
        status = "ACTIVE" if n["status"] == "active" else "FAILED"
        print(f"  {n['name']}: {status}, load={n['load']:.0f}/{n['capacity']}")

    print("\n  Преимущества Active-Active:")
    print("    + Нет простоя при отказе одного узла")
    print("    + Более равномерная загрузка")
    print("    - Сложность синхронизации данных")
    print("    - Возможны split-brain проблемы")

    # --- 2.3 DNS Failover ---
    print("\n--- 2.3 DNS Failover ---")

    print("  DNS Failover: маршрутизация через DNS records\n")

    # Модель DNS с health checks
    dns_records = [
        {"fqdn": "api.example.com", "type": "A", "ttl": 60,
         "records": [
             {"ip": "10.0.1.1", "weight": 100, "health": "healthy"},
             {"ip": "10.0.2.1", "weight": 100, "health": "healthy"},
         ]},
    ]

    print("  DNS Records:")
    for record in dns_records:
        print(f"    {record['fqdn']} (TTL={record['ttl']}s)")
        for r in record["records"]:
            status = "✓" if r["health"] == "healthy" else "✗"
            print(f"      {r['ip']:<15} weight={r['weight']} {status}")

    # Failover: один IP падает
    print("\n  Failover: 10.0.1.1 unhealthy → removed from DNS")
    dns_records[0]["records"][0]["health"] = "unhealthy"

    healthy = [r for r in dns_records[0]["records"] if r["health"] == "healthy"]
    print(f"  Осталось healthy records: {len(healthy)}")

    for r in healthy:
        print(f"    {r['ip']} → единственный target")

    # --- 2.4 Сравнение failover стратегий ---
    print("\n--- 2.4 Сравнение Failover стратегий ---")

    comparisons = [
        {"strategy": "Active-Passive", "rto": "30-60 сек", "complexity": "Низкая",
         "cost": "2x (1 idle)", "data_consistency": "Высокая"},
        {"strategy": "Active-Active", "rto": "0 (мгновенно)", "complexity": "Высокая",
         "cost": "2x (оба active)", "data_consistency": "Средняя (replication lag)"},
        {"strategy": "DNS Failover", "rto": "TTL + check (1-5 мин)", "complexity": "Средняя",
         "cost": "2x (оба active)", "data_consistency": "Зависит от backend"},
        {"strategy": "Load Balancer", "rto": "0 (мгновенно)", "complexity": "Средняя",
         "cost": "2x (оба в pool)", "data_consistency": "Высокая (health checks)"},
    ]

    print(f"  {'Стратегия':<18} {'RTO':<22} {'Сложность':<12} {'Стоимость':<16} {'Data Consistency'}")
    print("  " + "-" * 85)
    for c in comparisons:
        print(f"  {c['strategy']:<18} {c['rto']:<22} {c['complexity']:<12} "
              f"{c['cost']:<16} {c['data_consistency']}")

    print("\n" + "=" * 70)
    print("ВЫВОД: Active-Active для availability, Active-Passive для простоты")
    print("=" * 70)


# =============================================================================
# Демо 3: Хаос-инженерия
# =============================================================================
def demo_chaos_engineering():
    """Демонстрация chaos engineering: game day, fault injection, steady state"""
    print("\n" + "=" * 70)
    print("ДЕМО 3: ХАОС-ИНЖЕНЕРИЯ — game day, fault injection, steady state")
    print("=" * 70)

    # --- 3.1 Steady State Hypothesis ---
    print("\n--- 3.1 Steady State Hypothesis ---")

    print("  Steady State: система работает нормально при определённых метриках\n")

    # Модель steady state
    steady_state_metrics = {
        "error_rate": {"target": 0.01, "current": 0.005, "unit": "%"},
        "latency_p99": {"target": 200, "current": 150, "unit": "ms"},
        "throughput": {"target": 1000, "current": 1200, "unit": "req/s"},
        "availability": {"target": 99.9, "current": 99.95, "unit": "%"},
    }

    print("  Steady State Hypothesis:")
    print("  ЕСЛИ система в steady state")
    print("  ТОГДА она выдержит эксперимент\n")

    print(f"  {'Metric':<20} {'Target':<12} {'Current':<12} {'Status'}")
    print("  " + "-" * 50)
    for metric, values in steady_state_metrics.items():
        status = "✓ OK" if values["current"] <= values["target"] * 1.1 else "✗ FAIL"
        print(f"  {metric:<20} {values['target']:<12} {values['current']:<12} {status}")

    # --- 3.2 Fault Injection ---
    print("\n--- 3.2 Fault Injection ---")

    class ChaosExperiment:
        """Модель chaos experiment"""

        def __init__(self, name, fault_type, target, duration_sec):
            self.name = name
            self.fault_type = fault_type
            self.target = target
            self.duration = duration_sec
            self.results = []

        def inject_fault(self):
            """Инжекция fault в систему"""
            print(f"  Инжекция: {self.fault_type} → {self.target}")
            print(f"  Длительность: {self.duration} сек")

        def measure_impact(self):
            """Измерение воздействия"""
            # Модель: метрики во время fault
            baseline_latency = 100
            baseline_error_rate = 0.01

            if self.fault_type == "network_latency":
                latency_impact = baseline_latency * 3
                error_impact = baseline_error_rate * 2
            elif self.fault_type == "pod_kill":
                latency_impact = baseline_latency * 5
                error_impact = baseline_error_rate * 10
            elif self.fault_type == "disk_full":
                latency_impact = baseline_latency * 10
                error_impact = baseline_error_rate * 50
            else:
                latency_impact = baseline_latency * 2
                error_impact = baseline_error_rate * 5

            self.results = {
                "baseline_latency": baseline_latency,
                "impact_latency": latency_impact,
                "baseline_error": baseline_error_rate,
                "impact_error": error_impact,
                "recovery_time": self.duration * 2
            }
            return self.results

    experiments = [
        ChaosExperiment("Network partitions", "network_latency", "api-gateway", 60),
        ChaosExperiment("Pod killing", "pod_kill", "payment-service", 30),
        ChaosExperiment("Disk pressure", "disk_full", "data-store", 120),
        ChaosExperiment("CPU stress", "cpu_stress", "worker-pool", 90),
    ]

    for exp in experiments:
        print(f"\n  Эксперимент: {exp.name}")
        exp.inject_fault()
        results = exp.measure_impact()
        print(f"  Результаты:")
        print(f"    Latency: {results['baseline_latency']}ms → {results['impact_latency']}ms "
              f"({results['impact_latency']/results['baseline_latency']:.1f}x)")
        print(f"    Error rate: {results['baseline_error']}% → {results['impact_error']}% "
              f"({results['impact_error']/results['baseline_error']:.1f}x)")
        print(f"    Recovery time: {results['recovery_time']}s")

    # --- 3.3 Game Day ---
    print("\n--- 3.3 Game Day (Игровой день) ---")

    print("  Game Day: плановая тренировка по реагированию на инциденты\n")

    game_day_script = [
        {"step": 1, "time": "09:00", "action": "Объявление эксперимента",
         "description": "Уведомление команд, начало monitoring"},
        {"step": 2, "time": "09:15", "action": "Inject: Отключение primary DB",
         "description": "База данных перестаёт отвечать"},
        {"step": 3, "time": "09:16", "action": "Обнаружение проблемы",
         "description": "Алерт сработал через 60 сек"},
        {"step": 4, "time": "09:18", "action": "Начало реагирования",
         "description": "On-call инженер начал диагностику"},
        {"step": 5, "time": "09:25", "action": "Failover на replica",
         "description": "Реплика promoted до primary"},
        {"step": 6, "time": "09:28", "action": "Восстановление сервиса",
         "description": "API снова отвечает"},
        {"step": 7, "time": "09:30", "action": "Восстановление primary",
         "description": "Оригинальная DB восстановлена"},
        {"step": 8, "time": "09:35", "action": "Debrief",
         "description": "Обсуждение: что прошло хорошо, что улучшить"},
    ]

    print(f"  {'Step':<6} {'Time':<8} {'Action':<30} {'Description'}")
    print("  " + "-" * 75)
    for step in game_day_script:
        print(f"  {step['step']:<6} {step['time']:<8} {step['action']:<30} {step['description']}")

    # Метрики Game Day
    print("\n  Метрики Game Day:")
    game_metrics = {
        "MTTD (Mean Time to Detect)": "60 сек",
        "MTTR (Mean Time to Recover)": "12 мин",
        "Impact on Users": "2 мин downtime",
        "Data Loss": "0 (RPO=0)",
        "Team Coordination": "Хорошая",
    }
    for metric, value in game_metrics.items():
        print(f"    {metric}: {value}")

    # --- 3.4 Chaos Engineering Principles ---
    print("\n--- 3.4 Принципы Chaos Engineering ---")

    principles = [
        {
            "name": "1. Стабилизировать steady state",
            "description": "Определить нормальное поведение системы",
            "example": "Error rate < 0.1%, latency p99 < 200ms"
        },
        {
            "name": "2. Гипотеза",
            "description": "Что мы ожидаем при faults?",
            "example": "System выдержит loss of 1 replica без downtime"
        },
        {
            "name": "3. Эксперимент",
            "description": "Ввести реальные faults в production",
            "example": "Kill random pods, add network latency"
        },
        {
            "name": "4. Автоматизация",
            "description": "Запуск экспериментов непрерывно",
            "example": "CI/CD pipeline включает chaos experiments"
        },
        {
            "name": "5. Минимизация blast radius",
            "description": "Начинать с малого, расширять постепенно",
            "example": "1% трафика → 10% → 50% → 100%"
        },
    ]

    for p in principles:
        print(f"\n  {p['name']}:")
        print(f"    Описание: {p['description']}")
        print(f"    Пример: {p['example']}")

    print("\n" + "=" * 70)
    print("ВЫВОД: Chaos Engineering находит проблемы ДО того, как они находят вас")
    print("=" * 70)


# =============================================================================
# Демо 4: RTO/RPO
# =============================================================================
def demo_rto_rpo():
    """Демонстрация RTO/RPO: recovery time objective, recovery point objective"""
    print("\n" + "=" * 70)
    print("ДЕМО 4: RTO/RPO — recovery time objective, recovery point objective")
    print("=" * 70)

    # --- 4.1 Определения ---
    print("\n--- 4.1 Определения ---")

    print("  RTO (Recovery Time Objective):")
    print("    Максимально допустимое время простоя после катастрофы")
    print("    Сколько времени система может быть недоступна?\n")

    print("  RPO (Recovery Point Objective):")
    print("    Максимально допустимая потеря данных")
    print("    Сколько данных мы готовы потерять (в минутах/часах)?\n")

    # Модель: RTO/RPO trade-offs
    print("  Trade-off: чем ниже RTO/RPO → тем выше стоимость\n")

    rto_rpo_matrix = [
        {"rto": "24 часа", "rpo": "24 часа", "strategy": "Backup to tape",
         "cost": "1x", "availability": "99%"},
        {"rto": "4 часа", "rpo": "1 час", "strategy": "Daily backup + offsite",
         "cost": "2x", "availability": "99.5%"},
        {"rto": "1 час", "rpo": "15 минут", "strategy": "Incremental backup + replication",
         "cost": "5x", "availability": "99.9%"},
        {"rto": "15 минут", "rpo": "5 минут", "strategy": "Continuous replication",
         "cost": "10x", "availability": "99.95%"},
        {"rto": "1 минута", "rpo": "0 (zero)", "strategy": "Synchronous replication + auto-failover",
         "cost": "20x", "availability": "99.99%"},
    ]

    print(f"  {'RTO':<12} {'RPO':<12} {'Стратегия':<35} {'Стоимость':<10} {'Availability'}")
    print("  " + "-" * 85)
    for item in rto_rpo_matrix:
        print(f"  {item['rto']:<12} {item['rpo']:<12} {item['strategy']:<35} "
              f"{item['cost']:<10} {item['availability']}")

    # --- 4.2 Расчёт RTO/RPO ---
    print("\n--- 4.2 Расчёт RTO и RPO ---")

    # Модель: Recovery Time
    class RecoveryCalculator:
        """Калькулятор RTO/RPO"""

        def __init__(self):
            self.components = []

        def add_component(self, name, detection_time, assessment_time,
                         failover_time, validation_time, data_recovery_time=0):
            self.components.append({
                "name": name,
                "detection": detection_time,
                "assessment": assessment_time,
                "failover": failover_time,
                "validation": validation_time,
                "data_recovery": data_recovery_time
            })

        def calculate_rto(self):
            """Общий RTO = сумма всех этапов"""
            total = 0
            for comp in self.components:
                comp_total = (comp["detection"] + comp["assessment"] +
                            comp["failover"] + comp["validation"] + comp["data_recovery"])
                total += comp_total
            return total

        def calculate_rpo(self, replication_lag=0, backup_interval=0):
            """RPO = max(replication_lag, time_since_last_backup)"""
            return max(replication_lag, backup_interval)

    calc = RecoveryCalculator()

    # Пример: восстановление кластера
    calc.add_component("Load Balancer", detection_time=30, assessment_time=60,
                      failover_time=30, validation_time=60)
    calc.add_component("Application Server", detection_time=30, assessment_time=120,
                      failover_time=300, validation_time=180)
    calc.add_component("Database", detection_time=30, assessment_time=60,
                      failover_time=180, validation_time=120, data_recovery_time=300)

    print("  Компоненты и время восстановления:\n")
    print(f"  {'Component':<20} {'Detect':<10} {'Assess':<10} {'Failover':<12} {'Validate':<12} {'Data':<10} {'Total'}")
    print("  " + "-" * 85)

    total_rto = 0
    for comp in calc.components:
        comp_total = (comp["detection"] + comp["assessment"] +
                     comp["failover"] + comp["validation"] + comp["data_recovery"])
        total_rto += comp_total
        print(f"  {comp['name']:<20} {comp['detection']:<10} {comp['assessment']:<10} "
              f"{comp['failover']:<12} {comp['validation']:<12} {comp['data_recovery']:<10} {comp_total}")

    print(f"\n  Общий RTO: {total_rto} сек ({total_rto/60:.1f} мин)")

    # RPO расчёт
    print("\n  RPO Calculation:")
    scenarios = [
        {"name": "Synchronous replication", "lag": 0, "backup": 0},
        {"name": "Async replication (5s lag)", "lag": 5, "backup": 3600},
        {"name": "Daily backup only", "lag": 0, "backup": 86400},
        {"name": "Hourly backup", "lag": 0, "backup": 3600},
    ]

    for scenario in scenarios:
        rpo = max(scenario["lag"], scenario["backup"])
        print(f"  {scenario['name']:<35} → RPO = {rpo} сек ({rpo/60:.1f} мин)")

    # --- 4.3 Стоимость vs RTO/RPO ---
    print("\n--- 4.3 Стоимость vs RTO/RPO ---")

    # Модель: стоимость = f(RTO, RPO)
    def cost_model(rto_hours, rpo_hours, base_cost=10000):
        """Модель: стоимость растёт экспоненциально с уменьшением RTO/RPO"""
        # Чем меньше RTO/RPO → тем дороже
        rto_factor = math.exp(-rto_hours / 10)  # экспоненциальный рост
        rpo_factor = math.exp(-rpo_hours / 5)

        infrastructure_cost = base_cost * (1 + 1 / (rto_hours + 0.1)) * (1 + 1 / (rpo_hours + 0.1))
        return infrastructure_cost

    print("\n  Стоимость инфраструктуры vs RTO/RPO:")
    print(f"  {'RTO (hours)':<15} {'RPO (hours)':<15} {'Cost ($)':<12} {'vs Baseline'}")
    print("  " + "-" * 55)

    baseline_cost = cost_model(24, 24)
    for rto_h in [24, 4, 1, 0.25]:
        for rpo_h in [24, 1, 0.25]:
            cost = cost_model(rto_h, rpo_h)
            ratio = cost / baseline_cost
            print(f"  {rto_h:<15} {rpo_h:<15} ${cost:<11,.0f} {ratio:.1f}x")

    # --- 4.4 Business Impact ---
    print("\n--- 4.4 Business Impact Analysis ---")

    print("  Расчёт стоимости простоя (Downtime Cost per Hour):\n")

    businesses = [
        {"name": "E-commerce", "revenue_per_hour": 50000,
         "reputation_cost": 10000, "sla_penalty": 5000},
        {"name": "SaaS Platform", "revenue_per_hour": 25000,
         "reputation_cost": 15000, "sla_penalty": 10000},
        {"name": "Financial Trading", "revenue_per_hour": 500000,
         "reputation_cost": 100000, "sla_penalty": 50000},
        {"name": "Healthcare", "revenue_per_hour": 10000,
         "reputation_cost": 100000, "sla_penalty": 25000},
    ]

    print(f"  {'Business':<18} {'Revenue/hr':<14} {'Reputation':<14} {'SLA Penalty':<14} {'Total/hr'}")
    print("  " + "-" * 70)

    for biz in businesses:
        total = biz["revenue_per_hour"] + biz["reputation_cost"] + biz["sla_penalty"]
        print(f"  {biz['name']:<18} ${biz['revenue_per_hour']:<13,} ${biz['reputation_cost']:<13,} "
              f"${biz['sla_penalty']:<13,} ${total:,}")

    # Рекомендации
    print("\n  Рекомендации по RTO/RPO для разных бизнесов:")
    recommendations = [
        ("E-commerce", "RTO < 1 час, RPO < 15 мин", "Потеря продаж + репутация"),
        ("SaaS Platform", "RTO < 30 мин, RPO < 5 мин", "Клиенты уходят к конкурентам"),
        ("Financial Trading", "RTO < 1 мин, RPO = 0", "Потеря сделок = катастрофа"),
        ("Healthcare", "RTO < 5 мин, RPO < 1 мин", "Жизни людей в опасности"),
    ]

    for business, rto_rpo, reason in recommendations:
        print(f"    {business}: {rto_rpo} ({reason})")

    print("\n" + "=" * 70)
    print("ВЫВОД: RTO/RPO определяют стоимость DR; баланс между стоимостью и бизнес-риском")
    print("=" * 70)


# =============================================================================
# Запуск всех демонстраций
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("УРОК 212: DISASTER RECOVERY")
    print("Резервное копирование, переключение, хаос-инженерия")
    print("=" * 70)
    print()

    demo_backup_strategies()
    demo_failover_patterns()
    demo_chaos_engineering()
    demo_rto_rpo()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 70)
