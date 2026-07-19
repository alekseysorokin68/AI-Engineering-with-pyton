"""150 — Kubernetes Basics: поды, сервисы, деплойменты, масштабирование

Темы:
  1. Pod Concepts — контейнеры, лейблы, политика перезапуска, лимиты ресурсов
  2. Services — ClusterIP, NodePort, LoadBalancer, DNS-обнаружение
  3. Deployments — реплики, скатывания обновлений, откат
  4. Scaling — HPA, метрики ресурсов, политики масштабирования

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import sqlite3

random.seed(42)

# =============================================================================
# Демо 1: Pod Concepts — контейнеры, лейблы, политика перезапуска, лимиты
# =============================================================================

def demo_pod_concepts():
    print("=" * 70)
    print("ДЕМО 1: Pod Concepts — контейнеры, лейблы, restart policy, ресурсы")
    print("=" * 70)

    # --- 1.1 Модель контейнера в поде ---
    # Каждый под — это группа из одного или более контейнеров
    # с общим сетевым пространством и томами
    print("\n--- 1.1 Модель контейнера в поде ---")

    container = {
        "name": "web-app",
        "image": "nginx:1.25-alpine",
        "ports": [{"containerPort": 80, "protocol": "TCP"}],
        "env": [
            {"name": "APP_ENV", "value": "production"},
            {"name": "DB_HOST", "value": "postgres-svc"},
        ],
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
        },
    }

    # Под может содержать несколько контейнеров (sidecar, init)
    pod_spec = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "web-pod-abc123",
            "labels": {"app": "web", "env": "production", "tier": "frontend"},
        },
        "spec": {
            "containers": [container],
            "restartPolicy": "Always",
            "terminationGracePeriodSeconds": 30,
        },
    }

    print(f"Под: {pod_spec['metadata']['name']}")
    print(f"Лейблы: {pod_spec['metadata']['labels']}")
    print(f"Контейнеров: {len(pod_spec['spec']['containers'])}")
    print(f"Restart policy: {pod_spec['spec']['restartPolicy']}")
    print(f"CPU request: {container['resources']['requests']['cpu']}, "
          f"limit: {container['resources']['limits']['cpu']}")

    # --- 1.2 Лейблы и селекторы ---
    # Лейблы — ключ-значение, привязанные к объектам k8s
    # Селекторы фильтруют объекты по лейблам
    print("\n--- 1.2 Лейблы и селекторы ---")

    pods = [
        {"name": "web-0", "labels": {"app": "web", "version": "v1"}},
        {"name": "web-1", "labels": {"app": "web", "version": "v2"}},
        {"name": "api-0", "labels": {"app": "api", "version": "v1"}},
        {"name": "api-1", "labels": {"app": "api", "version": "v2"}},
        {"name": "db-0",   "labels": {"app": "db",  "version": "v1"}},
    ]

    # Селектор по app=web
    selector = {"app": "web"}
    matched = [p for p in pods if all(p["labels"].get(k) == v for k, v in selector.items())]
    print(f"Селектор {{app: web}} -> {[p['name'] for p in matched]}")

    # Сложный селектор
    selector2 = {"app": "api", "version": "v2"}
    matched2 = [p for p in pods if all(p["labels"].get(k) == v for k, v in selector2.items())]
    print(f"Селектор {{app: api, version: v2}} -> {[p['name'] for p in matched2]}")

    # Подсчёт по лейблам
    app_counts = collections.Counter(p["labels"]["app"] for p in pods)
    print(f"Распределение по app: {dict(app_counts)}")

    # --- 1.3 Restart policies ---
    # Always — перезапускать всегда (по умолчанию для Deployment)
    # OnFailure — только при ошибке (код != 0)
    # Never — не перезапускать
    print("\n--- 1.3 Restart policies ---")

    policies = {
        "Always":        "Для Deployment/ReplicaSet — контейнер перезапускается всегда",
        "OnFailure":     "Для Job — перезапуск только при crash (exit code != 0)",
        "Never":         "Для Job — ни один перезапуск",
    }

    for policy, desc in policies.items():
        print(f"  {policy:12s} -> {desc}")

    # Моделируем перезапуски пода
    restart_counts = {}
    for _ in range(20):
        pod_name = random.choice(["web-0", "web-1", "api-0"])
        restart_counts[pod_name] = restart_counts.get(pod_name, 0) + 1

    print("\nРестарты за наблюдаемый период:")
    for pod, count in sorted(restart_counts.items()):
        status = "Healthy" if count <= 2 else "CrashLoopBackOff"
        print(f"  {pod}: {count} рестартов -> {status}")

    # --- 1.4 Расчёт ресурсов кластера ---
    # requests = гарантированные ресурсы; limits = максимум
    print("\n--- 1.4 Расчёт ресурсов кластера ---")

    nodes = [
        {"name": "node-1", "cpu_capacity": 4000, "mem_capacity": 8192},  # в millicores и Mi
        {"name": "node-2", "cpu_capacity": 4000, "mem_capacity": 8192},
        {"name": "node-3", "cpu_capacity": 8000, "mem_capacity": 16384},
    ]

    workloads = [
        {"name": "web",   "replicas": 3, "cpu_req": 100, "cpu_lim": 500,  "mem_req": 128, "mem_lim": 256},
        {"name": "api",   "replicas": 2, "cpu_req": 200, "cpu_lim": 1000, "mem_req": 256, "mem_lim": 512},
        {"name": "worker","replicas": 4, "cpu_req": 150, "cpu_lim": 750,  "mem_req": 192, "mem_lim": 384},
    ]

    total_cpu_req = sum(w["cpu_req"] * w["replicas"] for w in workloads)
    total_mem_req = sum(w["mem_req"] * w["replicas"] for w in workloads)
    total_capacity_cpu = sum(n["cpu_capacity"] for n in nodes)
    total_capacity_mem = sum(n["mem_capacity"] for n in nodes)

    print(f"  Всего CPU request: {total_cpu_req}m / {total_capacity_cpu}m "
          f"({total_cpu_req/total_capacity_cpu*100:.1f}%)")
    print(f"  Всего MEM request: {total_mem_req}Mi / {total_capacity_mem}Mi "
          f"({total_mem_req/total_capacity_mem*100:.1f}%)")
    print("  Формула: utilization = sum(pod_requests) / sum(node_capacity) * 100%")
    print("  Рекомендуется держать utilization < 80% для буфера")


# =============================================================================
# Демо 2: Services — ClusterIP, NodePort, LoadBalancer, DNS
# =============================================================================

def demo_services():
    print("=" * 70)
    print("ДЕМО 2: Services — ClusterIP, NodePort, LoadBalancer, DNS")
    print("=" * 70)

    # --- 2.1 ClusterIP (по умолчанию) ---
    # Внутренний VIP, доступный только внутри кластера
    print("\n--- 2.1 ClusterIP ---")

    cluster_ip_svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "postgres-svc"},
        "spec": {
            "type": "ClusterIP",
            "selector": {"app": "postgres"},
            "ports": [{"port": 5432, "targetPort": 5432, "protocol": "TCP"}],
            "clusterIP": "10.96.0.100",
        },
    }

    print(f"Сервис: {cluster_ip_svc['metadata']['name']}")
    print(f"Тип: {cluster_ip_svc['spec']['type']}")
    print(f"ClusterIP: {cluster_ip_svc['spec']['clusterIP']}")
    print(f"Порт: {cluster_ip_svc['spec']['ports'][0]['port']} -> "
          f"targetPort: {cluster_ip_svc['spec']['ports'][0]['targetPort']}")
    print("Доступен только из кластера: curl postgres-svc:5432")

    # --- 2.2 NodePort ---
    # Открывает порт на КАЖДОМ 노де кластера
    print("\n--- 2.2 NodePort ---")

    node_port_svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "web-nodeport"},
        "spec": {
            "type": "NodePort",
            "selector": {"app": "web"},
            "ports": [{"port": 80, "targetPort": 8080, "nodePort": 30080}],
        },
    }

    node_ips = ["192.168.1.10", "192.168.1.11", "192.168.1.12"]
    print(f"Сервис: {node_port_svc['metadata']['name']}")
    print(f"NodePort: {node_port_svc['spec']['ports'][0]['nodePort']}")
    print("Доступ с любого ноды:")
    for ip in node_ips:
        print(f"  http://{ip}:30080")

    # --- 2.3 LoadBalancer ---
    # Обеспечивает внешний IP через облачный балансировщик
    print("\n--- 2.3 LoadBalancer ---")

    lb_svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "web-lb"},
        "spec": {
            "type": "LoadBalancer",
            "selector": {"app": "web"},
            "ports": [{"port": 443, "targetPort": 8443}],
            "loadBalancerIP": "34.120.55.88",
        },
    }

    print(f"Сервис: {lb_svc['metadata']['name']}")
    print(f"Внешний IP: {lb_svc['spec']['loadBalancerIP']}")
    print(f"Порт: {lb_svc['spec']['ports'][0]['port']} -> "
          f"targetPort: {lb_svc['spec']['ports'][0]['targetPort']}")
    print("Трафик: клиент -> LoadBalancer(34.120.55.88:443) -> Pod(:8443)")

    # --- 2.4 DNS-обнаружение ---
    # Формат: <service-name>.<namespace>.svc.cluster.local
    print("\n--- 2.4 DNS-обнаружение ---")

    services = [
        {"name": "postgres-svc", "namespace": "default", "port": 5432},
        {"name": "redis-svc",    "namespace": "cache",   "port": 6379},
        {"name": "api-svc",      "namespace": "backend", "port": 8080},
    ]

    for svc in services:
        fqdn = f"{svc['name']}.{svc['namespace']}.svc.cluster.local"
        short = svc["name"]
        print(f"  {fqdn}:{svc['port']}")
        print(f"    Короткое имя (в том же ns): {short}:{svc['port']}")
        print()

    # Симуляция DNS-резолва
    dns_cache = {}
    for svc in services:
        fqdn = f"{svc['name']}.{svc['namespace']}.svc.cluster.local"
        # Случайный IP для каждого сервиса
        ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        dns_cache[fqdn] = ip

    print("DNS-кэш (симуляция):")
    for name, ip in dns_cache.items():
        print(f"  {name} -> {ip}")


# =============================================================================
# Демо 3: Deployments — реплики, скатывания, откат
# =============================================================================

def demo_deployments():
    print("=" * 70)
    print("ДЕМО 3: Deployments — реплики, скатывания обновлений, откат")
    print("=" * 70)

    # --- 3.1 Определение Deployment ---
    print("\n--- 3.1 Определение Deployment ---")

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "web-deployment"},
        "spec": {
            "replicas": 3,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxSurge": 1,
                    "maxUnavailable": 0,
                },
            },
            "selector": {"matchLabels": {"app": "web"}},
            "template": {
                "metadata": {"labels": {"app": "web", "version": "v3"}},
                "spec": {
                    "containers": [{
                        "name": "web",
                        "image": "myapp:v3.1.0",
                        "ports": [{"containerPort": 8080}],
                    }],
                },
            },
        },
    }

    print(f"Deployment: {deployment['metadata']['name']}")
    print(f"Реплик: {deployment['spec']['replicas']}")
    print(f"Стратегия: {deployment['spec']['strategy']['type']}")
    print(f"maxSurge: {deployment['spec']['strategy']['rollingUpdate']['maxSurge']}")
    print(f"maxUnavailable: {deployment['spec']['strategy']['rollingUpdate']['maxUnavailable']}")
    print(f"Образ: {deployment['spec']['template']['spec']['containers'][0]['image']}")

    # --- 3.2 Скатывание обновлений (Rolling Update) ---
    # maxSurge=1: максимум на 1 под больше желаемого
    # maxUnavailable=0: ни один под не может быть недоступен
    print("\n--- 3.2 Скатывание обновлений (Rolling Update) ---")

    # Симуляция шагов rolling update
    steps = []
    desired = 3
    current = {"v2": 3, "v3": 0}
    max_surge = 1
    max_unavail = 0

    print(f"Начальное состояние: v2={current['v2']}, v3={current['v3']}")
    print(f"Целевое: v2=0, v3={desired}")
    print(f"maxSurge={max_surge}, maxUnavailable={max_unavail}")
    print()

    step_num = 0
    while current["v2"] > 0:
        step_num += 1
        # Создаём новую v3, если есть место для maxSurge
        total = current["v2"] + current["v3"]
        if total < desired + max_surge and current["v2"] > 0:
            current["v3"] += 1
            action = "Создан pod v3"
        else:
            # Удаляем старый v2
            current["v2"] -= 1
            action = "Удалён pod v2"

        total = current["v2"] + current["v3"]
        print(f"  Шаг {step_num}: {action} -> v2={current['v2']}, v3={current['v3']}, "
              f"всего={total}")
        steps.append(step_num)

    print(f"\nRolling update завершён за {len(steps)} шагов")

    # --- 3.3 Откат (Rollback) ---
    print("\n--- 3.3 Откат (Rollback) ---")

    revisions = [
        {"revision": 1, "image": "myapp:v1.0.0", "created": "2025-01-15"},
        {"revision": 2, "image": "myapp:v2.0.0", "created": "2025-02-20"},
        {"revision": 3, "image": "myapp:v3.0.0", "created": "2025-03-10"},
        {"revision": 4, "image": "myapp:v3.1.0", "created": "2025-03-25"},
    ]

    print("История ревизий:")
    for r in revisions:
        current_marker = " <-- текущая" if r["revision"] == 4 else ""
        print(f"  Ревизия {r['revision']}: {r['image']} ({r['created']}){current_marker}")

    # Откат к ревизии 2
    target_revision = 2
    target_image = next(r["image"] for r in revisions if r["revision"] == target_revision)
    print(f"\nОткат к ревизии {target_revision}: образ будет {target_image}")
    print("Процесс: создание новой ревизии 5 на базе ревизии 2")
    print("  -> Все поды обновятся до v2.0.0 через rolling update")

    # --- 3.4 ReplicaSet и история ---
    print("\n--- 3.4 ReplicaSet и история ---")

    # ReplicaSet — низкоуровневый объект, контролируемый Deployment
    replica_sets = [
        {"name": "web-deployment-7b5f4c8d9", "replicas": 0, "image": "myapp:v2.0.0"},
        {"name": "web-deployment-9c3e6a1b2", "replicas": 0, "image": "myapp:v3.0.0"},
        {"name": "web-deployment-4d8f2e5a7", "replicas": 3, "image": "myapp:v3.1.0"},
    ]

    print("ReplicaSet'ы (управляемые Deployment):")
    for rs in replica_sets:
        status = "активный" if rs["replicas"] > 0 else "масштабирован до 0"
        print(f"  {rs['name']}: {rs['replicas']} реплик, образ={rs['image']}, статус={status}")

    print("\nПорядок истории (historyLimit=5):")
    print("  Deployment хранит максимум 5 последних ReplicaSet")
    print("  Старые RS масштабируются до 0, но не удаляются")
    print("  Откат = масштабирование старого RS до нужного числа реплик")


# =============================================================================
# Демо 4: Scaling — HPA, метрики, политики
# =============================================================================

def demo_scaling():
    print("=" * 70)
    print("ДЕМО 4: Scaling — HPA, метрики ресурсов, политики масштабирования")
    print("=" * 70)

    # --- 4.1 HPA (Horizontal Pod Autoscaler) ---
    # Масштабирует количество подов на основе метрик
    print("\n--- 4.1 HPA (Horizontal Pod Autoscaler) ---")

    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": "web-hpa"},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": "web-deployment",
            },
            "minReplicas": 2,
            "maxReplicas": 10,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": 70},
                    },
                },
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {"type": "Utilization", "averageUtilization": 80},
                    },
                },
            ],
        },
    }

    print(f"HPA: {hpa['metadata']['name']}")
    print(f"Цель: {hpa['spec']['scaleTargetRef']['name']}")
    print(f"Реплики: {hpa['spec']['minReplicas']} - {hpa['spec']['maxReplicas']}")
    for m in hpa["spec"]["metrics"]:
        print(f"  Метрика: {m['resource']['name']}, "
              f"целевая утилизация: {m['resource']['target']['averageUtilization']}%")

    # --- 4.2 Формула масштабирования ---
    # desiredReplicas = ceil[currentReplicas * (currentMetricValue / targetMetricValue)]
    print("\n--- 4.2 Формула масштабирования HPA ---")

    print("Формула: desiredReplicas = ceil[currentReplicas × (currentValue / targetValue)]")
    print()

    current_replicas = 3
    target_cpu = 70

    scenarios = [
        {"current_cpu": 50, "label": "Низкая нагрузка"},
        {"current_cpu": 70, "label": "На целевом уровне"},
        {"current_cpu": 90, "label": "Высокая нагрузка"},
        {"current_cpu": 140, "label": "Экстремальная нагрузка"},
    ]

    for s in scenarios:
        desired = math.ceil(current_replicas * (s["current_cpu"] / target_cpu))
        desired = max(2, min(desired, 10))  # ограничения min/max
        change = desired - current_replicas
        action = "scale UP" if change > 0 else ("scale DOWN" if change < 0 else "stable")
        print(f"  {s['label']}: CPU={s['current_cpu']}%, "
              f"desired=ceil({current_replicas}×{s['current_cpu']}/{target_cpu})={desired}, "
              f"изменение={change:+d} -> {action}")

    # --- 4.3 Симуляция изменения нагрузки ---
    print("\n--- 4.3 Симуляция изменения нагрузки во времени ---")

    random.seed(42)
    replicas = 3
    min_r, max_r = 2, 10
    target_util = 70
    history = []

    # Симулируем 12 метрических выборок (по 15 сек = 3 мин)
    for tick in range(12):
        # Нагрузка растёт, затем падает
        if tick < 6:
            load = 30 + tick * 15 + random.randint(-5, 5)
        else:
            load = 90 - (tick - 6) * 12 + random.randint(-5, 5)
        load = max(10, min(load, 200))

        desired = math.ceil(replicas * (load / target_util))
        desired = max(min_r, min(desired, max_r))
        old_replicas = replicas
        replicas = desired

        history.append({
            "tick": tick + 1,
            "cpu_pct": load,
            "replicas": replicas,
            "change": replicas - old_replicas,
        })

    print(f"  {'tick':>4s} {'CPU%':>6s} {'replicas':>8s} {'change':>7s}")
    print(f"  {'----':>4s} {'------':>6s} {'--------':>8s} {'-------':>7s}")
    for h in history:
        arrow = "▲" if h["change"] > 0 else ("▼" if h["change"] < 0 else "=")
        print(f"  {h['tick']:4d} {h['cpu_pct']:6.0f} {h['replicas']:8d} {h['change']:+7d} {arrow}")

    # --- 4.4 Политики масштабирования ---
    # stabilizationWindowSeconds — окно стабилизации
    # policies — ограничения скорости изменения
    print("\n--- 4.4 Политики масштабирования ---")

    scaling_policies = [
        {"type": "Pods",    "value": 2, "periodSeconds": 60,
         "описание": "Максимум ±2 пода за 60 сек"},
        {"type": "Percent", "value": 50, "periodSeconds": 60,
         "описание": "Максимум ±50% подов за 60 сек"},
    ]

    behavior = {
        "scaleUp": {
            "stabilizationWindowSeconds": 60,
            "policies": scaling_policies,
            "selectPolicy": {"name": "Max"},
        },
        "scaleDown": {
            "stabilizationWindowSeconds": 300,
            "policies": [{"type": "Percent", "value": 10, "periodSeconds": 120}],
            "selectPolicy": {"name": "Min"},
        },
    }

    print("scaleUp:")
    print(f"  stabilizationWindow: {behavior['scaleUp']['stabilizationWindowSeconds']}с")
    for p in behavior["scaleUp"]["policies"]:
        if "описание" in p:
            print(f"  Политика: {p['описание']}")
        else:
            print(f"  Политика: ±{p['value']}% за {p['periodSeconds']}с")
    print(f"  selectPolicy: {behavior['scaleUp']['selectPolicy']['name']}")
    print()
    print("scaleDown:")
    print(f"  stabilizationWindow: {behavior['scaleDown']['stabilizationWindowSeconds']}с")
    for p in behavior["scaleDown"]["policies"]:
        if "описание" in p:
            print(f"  Политика: {p['описание']}")
        else:
            print(f"  Политика: ±{p['value']}% за {p['periodSeconds']}с")
    print(f"  selectPolicy: {behavior['scaleDown']['selectPolicy']['name']}")
    print()
    print("Примечание: scaleDown медленнее scaleUp, чтобы избежать флиппинга")


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    demo_pod_concepts()
    print()
    demo_services()
    print()
    demo_deployments()
    print()
    demo_scaling()
