"""
202 — Container Orchestration: Kubernetes продвинутый, операторы, service mesh

Темы:
  1. Kubernetes Advanced (StatefulSets, DaemonSets, Jobs, CronJobs)
  2. Operators (custom resources, reconciliation loops, operator pattern)
  3. Service Mesh (sidecar proxy, traffic management, mTLS)
  4. GitOps (ArgoCD concepts, declarative deployments, drift detection)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

# Фиксируем seed для воспроизводимости
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1: Kubernetes Advanced (StatefulSets, DaemonSets, Jobs, CronJobs)
# ─────────────────────────────────────────────────────────────────────────────
def demo_kubernetes_advanced():
    """Демонстрация продвинутых концепций Kubernetes."""
    print("=" * 70)
    print("DEMO 1: Kubernetes Advanced (StatefulSets, DaemonSets, Jobs, CronJobs)")
    print("=" * 70)

    # 1.1 StatefulSet — управление��态化 pod-ами
    print("\n1.1 StatefulSet — управление状态化 pod-ами:")
    print("   StatefulSet обеспечивает стабильные имена и хранилище для pod-ов\n")

    class StatefulSet:
        """Симуляция Kubernetes StatefulSet."""
        def __init__(self, name, replicas, image):
            self.name = name
            self.replicas = replicas
            self.image = image
            self.pods = []
            self._create_pods()

        def _create_pods(self):
            """Создание pod-ов со стабильными именами."""
            for i in range(self.replicas):
                pod = {
                    "name": f"{self.name}-{i}",
                    "hostname": f"{self.name}-{i}.headless-svc.default.svc.cluster.local",
                    "status": "Running",
                    "volume": f"pvc-{self.name}-{i}",
                    "ordinal": i,
                }
                self.pods.append(pod)

        def scale(self, new_replicas):
            """Масштабирование с сохранением имён."""
            old_count = len(self.pods)
            if new_replicas > old_count:
                for i in range(old_count, new_replicas):
                    pod = {
                        "name": f"{self.name}-{i}",
                        "hostname": f"{self.name}-{i}.headless-svc.default.svc.cluster.local",
                        "status": "Running",
                        "volume": f"pvc-{self.name}-{i}",
                        "ordinal": i,
                    }
                    self.pods.append(pod)
            else:
                self.pods = self.pods[:new_replicas]
            self.replicas = new_replicas

    sts = StatefulSet("mysql", 3, "mysql:8.0")
    print("   StatefulSet 'mysql' создан:")
    for pod in sts.pods:
        print(f"     - {pod['name']}: hostname={pod['hostname']}, volume={pod['volume']}")

    sts.scale(5)
    print(f"\n   Масштабирование до 5 реплик:")
    for pod in sts.pods:
        print(f"     - {pod['name']}: {pod['status']}")

    # 1.2 DaemonSet — один pod на каждый узел
    print("\n1.2 DaemonSet — один pod на каждый узел:")
    print("   DaemonSet гарантирует запуск pod-а на каждом (или 특정ных) узлах\n")

    class DaemonSet:
        """Симуляция DaemonSet."""
        def __init__(self, name, image, nodes):
            self.name = name
            self.image = image
            self.pods = {}
            for node in nodes:
                self.pods[node] = {
                    "name": f"{name}-{node}",
                    "node": node,
                    "status": "Running",
                    "image": image,
                }

        def get_status(self):
            """Получить статус pod-ов."""
            return {node: pod["status"] for node, pod in self.pods.items()}

    nodes = ["node-1", "node-2", "node-3", "node-4"]
    daemon = DaemonSet("fluentd", "fluentd:v1.16", nodes)
    print("   DaemonSet 'fluentd' на узлах:")
    for node, status in daemon.get_status().items():
        print(f"     - {node}: {status}")

    # 1.3 Job — одноразовые задачи
    print("\n1.3 Job — одноразовые задачи:")
    print("   Job создаёт pod-ы для выполнения задач и завершает их по окончании\n")

    class Job:
        """Симуляция Kubernetes Job."""
        def __init__(self, name, completions, parallelism):
            self.name = name
            self.completions = completions
            self.parallelism = parallelism
            self.succeeded = 0
            self.failed = 0
            self.active = 0

        def run(self):
            """Симуляция выполнения задач."""
            print(f"   Job '{self.name}': completions={self.completions}, parallelism={self.parallelism}")
            remaining = self.completions
            while remaining > 0:
                batch = min(remaining, self.parallelism)
                self.active = batch
                print(f"     Запуск {batch} pod-ов (осталось: {remaining})")
                # Симуляция выполнения
                for i in range(batch):
                    if random.random() < 0.9:  # 90% шанс успеха
                        self.succeeded += 1
                        remaining -= 1
                        print(f"       Pod-{self.succeeded}: SUCCESS")
                    else:
                        self.failed += 1
                        print(f"       Pod-{self.succeeded + self.failed}: FAILED (повтор)")
                self.active = 0
            print(f"   Итого: succeeded={self.succeeded}, failed={self.failed}")

    job = Job("data-processing", completions=5, parallelism=2)
    job.run()

    # 1.4 CronJob — запуск по расписанию
    print("\n1.4 CronJob — запуск по расписанию:")
    print("   CronJob запускает Job по cron-выражению\n")

    class CronJob:
        """Симуляция CronJob."""
        def __init__(self, name, schedule, job_template):
            self.name = name
            self.schedule = schedule
            self.job_template = job_template
            self.history = []

        def trigger(self, current_time):
            """Триггер запуска (в реальности проверяется cron)."""
            hour = current_time.tm_hour
            minute = current_time.tm_min
            print(f"   CronJob '{self.name}' проверяет расписание: {self.schedule}")
            print(f"   Текущее время: {hour:02d}:{minute:02d}")
            # Простая проверка: запуск каждые 6 часов
            if hour % 6 == 0:
                self.history.append({"time": f"{hour:02d}:{minute:02d}", "status": "completed"})
                print(f"   -> Job запущен и завершён")
            else:
                print(f"   -> Время ещё не пришло")

    cron = CronJob("backup-database", "0 */6 * * *", "backup-job")
    current = time.gmtime()
    cron.trigger(current)
    print(f"   История запусков: {cron.history}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2: Operators (custom resources, reconciliation loops)
# ─────────────────────────────────────────────────────────────────────────────
def demo_operators():
    """Демонстрация паттерна Operator в Kubernetes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Operators (Custom Resources, Reconciliation Loops)")
    print("=" * 70)

    # 2.1 Custom Resource Definition (CRD)
    print("\n2.1 Custom Resource Definition (CRD):")
    print("   CRD позволяет определять собственные типы ресурсов в Kubernetes\n")

    crd = {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {"name": "mlmodels.ai.example.com"},
        "spec": {
            "group": "ai.example.com",
            "versions": [{"name": "v1", "served": True, "storage": True}],
            "scope": "Namespaced",
            "names": {
                "plural": "mlmodels",
                "singular": "mlmodel",
                "kind": "MLModel",
                "shortNames": ["mlm"],
            },
        },
    }
    print("   CRD для ML моделей:")
    print(f"   apiVersion: {crd['apiVersion']}")
    print(f"   kind: {crd['kind']}")
    print(f"   name: {crd['metadata']['name']}")
    print(f"   names: {crd['spec']['names']}")

    # 2.2 Custom Resource (MLModel)
    print("\n2.2 Custom Resource — MLModel:")
    mlmodel_cr = {
        "apiVersion": "ai.example.com/v1",
        "kind": "MLModel",
        "metadata": {"name": "sentiment-analyzer", "namespace": "production"},
        "spec": {
            "modelType": "transformer",
            "framework": "pytorch",
            "version": "2.1.0",
            "replicas": 3,
            "resources": {"cpu": "2", "memory": "4Gi", "gpu": "1"},
            "endpoints": ["/predict", "/health"],
        },
    }
    print("   Custom Resource 'MLModel':")
    print(json.dumps(mlmodel_cr, indent=2))

    # 2.3 Reconciliation Loop (основа оператора)
    print("\n2.3 Reconciliation Loop (основа оператора):")
    print("   Оператор непрерывно сравнивает desired state с actual state\n")

    class Reconciler:
        """Симуляция reconciliation loop."""
        def __init__(self, desired_state):
            self.desired = desired_state
            self.actual = {"replicas": 0, "status": "None", "endpoints": []}
            self.iteration = 0

        def reconcile(self):
            """Один цикл reconciliation."""
            self.iteration += 1
            actions = []

            # Проверка количества реплик
            if self.actual["replicas"] != self.desired["replicas"]:
                diff = self.desired["replicas"] - self.actual["replicas"]
                action = "scale_up" if diff > 0 else "scale_down"
                self.actual["replicas"] = self.desired["replicas"]
                actions.append(f"{action}: {abs(diff)} replicas")

            # Проверка статуса
            if self.actual["status"] != self.desired.get("status", "Running"):
                self.actual["status"] = self.desired.get("status", "Running")
                actions.append(f"status_update: {self.actual['status']}")

            # Проверка endpoints
            desired_eps = set(self.desired.get("endpoints", []))
            actual_eps = set(self.actual["endpoints"])
            if desired_eps != actual_eps:
                self.actual["endpoints"] = list(desired_eps)
                actions.append("endpoints_synced")

            return actions

    desired = {"replicas": 3, "status": "Running", "endpoints": ["/predict", "/health"]}
    reconciler = Reconciler(desired)

    # Симуляция нескольких итераций
    for _ in range(3):
        actions = reconciler.reconcile()
        if actions:
            print(f"   Итерация {reconciler.iteration}: {actions}")
        else:
            print(f"   Итерация {reconciler.iteration}: нет изменений (converged)")

    print(f"\n   Actual state: {reconciler.actual}")
    print("   -> Reconciliation loop поддерживает desired state")

    # 2.4 Operator pattern — полный жизненный цикл
    print("\n2.4 Operator Pattern — жизненный цикл:")
    print("   1. Watch: отслеживание изменений в Custom Resources")
    print("   2. Reconcile: сравнение desired vs actual state")
    print("   3. Act: выполнение действий для достижения desired state")
    print("   4. Report: обновление status в Custom Resource\n")

    lifecycle_steps = [
        ("Watch",     "Получено событие CREATE для MLModel/sentiment-analyzer"),
        ("Reconcile", "desired=3 replicas, actual=0 replicas"),
        ("Act",       "Создаём 3 pod-а с образом pytorch/sentiment:v2.1.0"),
        ("Report",    "Обновляем status: readyReplicas=3"),
        ("Reconcile", "desired=3, actual=3 -> нет изменений"),
        ("Watch",     "Получено событие UPDATE: replicas=5"),
        ("Reconcile", "desired=5, actual=3"),
        ("Act",       "Масштабируем до 5 pod-ов"),
        ("Report",    "Обновляем status: readyReplicas=5"),
    ]

    for i, (step, description) in enumerate(lifecycle_steps):
        print(f"   [{i+1}] {step:10}: {description}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: Service Mesh (sidecar proxy, traffic management, mTLS)
# ─────────────────────────────────────────────────────────────────────────────
def demo_service_mesh():
    """Демонстрация концепций Service Mesh."""
    print("\n" + "=" * 70)
    print("DEMO 3: Service Mesh (Sidecar Proxy, Traffic Management, mTLS)")
    print("=" * 70)

    # 3.1 Sidecar Proxy (Envoy/Istio)
    print("\n3.1 Sidecar Proxy паттерн:")
    print("   Каждый pod получает sidecar-прокси для управления сетевым трафиком\n")

    class SidecarProxy:
        """Симуляция sidecar proxy."""
        def __init__(self, service_name):
            self.service = service_name
            self.inbound_rules = []
            self.outbound_rules = []
            self.metrics = {"requests": 0, "errors": 0, "latency_sum": 0}

        def add_inbound_rule(self, port, protocol, source):
            """Добавить правило входящего трафика."""
            self.inbound_rules.append({
                "port": port, "protocol": protocol, "source": source
            })

        def add_outbound_rule(self, destination, port):
            """Добавить правило исходящего трафика."""
            self.outbound_rules.append({"destination": destination, "port": port})

        def intercept_request(self, request):
            """Перехватить входящий запрос."""
            self.metrics["requests"] += 1
            latency = random.randint(5, 50)
            self.metrics["latency_sum"] += latency
            if random.random() < 0.05:
                self.metrics["errors"] += 1
                return {"status": 503, "error": "upstream unavailable"}
            return {"status": 200, "latency_ms": latency}

        def get_metrics(self):
            """Получить метрики прокси."""
            reqs = self.metrics["requests"]
            avg_latency = self.metrics["latency_sum"] / max(reqs, 1)
            error_rate = self.metrics["errors"] / max(reqs, 1) * 100
            return {"requests": reqs, "avg_latency_ms": round(avg_latency, 1),
                    "error_rate_%": round(error_rate, 2)}

    # Создание прокси для сервисов
    api_proxy = SidecarProxy("api-service")
    api_proxy.add_inbound_rule(8080, "HTTP", "ingress-gateway")
    api_proxy.add_outbound_rule("database-service", 5432)
    api_proxy.add_outbound_rule("cache-service", 6379)

    print("   API Service Sidecar Proxy:")
    print(f"     Inbound rules: {api_proxy.inbound_rules}")
    print(f"     Outbound rules: {api_proxy.outbound_rules}")

    # Симуляция запросов
    print("\n   Симуляция 20 запросов:")
    for _ in range(20):
        api_proxy.intercept_request({"path": "/api/predict"})
    metrics = api_proxy.get_metrics()
    print(f"     Метрики: {metrics}")

    # 3.2 Traffic Management
    print("\n3.2 Traffic Management (управление трафиком):")

    class TrafficManager:
        """Управление распределением трафика."""
        def __init__(self, service):
            self.service = service
            self.routes = {}

        def add_route(self, name, destination, weight):
            """Добавить маршрут с весом."""
            self.routes[name] = {"destination": destination, "weight": weight}

        def route_request(self, request):
            """Маршрутизация запроса на основе весов."""
            total = sum(r["weight"] for r in self.routes.values())
            rand = random.random() * total
            cumulative = 0
            for name, route in self.routes.items():
                cumulative += route["weight"]
                if rand <= cumulative:
                    return name, route["destination"]
            return list(self.routes.keys())[-1], list(self.routes.values())[-1]["destination"]

    tm = TrafficManager("api-service")
    tm.add_route("stable", {"version": "v1.0", "replicas": 10}, weight=80)
    tm.add_route("canary", {"version": "v1.1", "replicas": 2}, weight=15)
    tm.add_route("experimental", {"version": "v2.0-alpha", "replicas": 1}, weight=5)

    print("   Распределение трафика (canary deployment):")
    results = collections.Counter()
    for _ in range(100):
        route, dest = tm.route_request({})
        results[f"{route} ({dest['version']})"] += 1

    for route, count in results.most_common():
        print(f"     {route}: {count}%")

    # 3.3 mTLS (mutual TLS)
    print("\n3.3 mTLS (Mutual TLS) в Service Mesh:")
    print("   Автоматическое шифрование и аутентификация между сервисами\n")

    class MTLSManager:
        """Симуляция mTLS менеджера."""
        def __init__(self):
            self.certificates = {}

        def issue_certificate(self, service_name):
            """Выдать сертификат сервису."""
            cert_id = hashlib.sha256(service_name.encode()).hexdigest()[:16]
            self.certificates[service_name] = {
                "cert_id": cert_id,
                "issued_at": time.time(),
                "expires_in": "24h",
                "trusted_by": "Istio CA",
            }
            return self.certificates[service_name]

        def verify(self, client, server):
            """Проверить mTLS соединение."""
            client_cert = self.certificates.get(client)
            server_cert = self.certificates.get(server)
            if client_cert and server_cert:
                return {"status": "verified", "cipher": "AES-256-GCM"}
            return {"status": "failed", "error": "certificate not found"}

    mtls = MTLSManager()

    # Выдача сертификатов
    services = ["api-service", "database-service", "cache-service", "worker-service"]
    print("   Выдача сертификатов:")
    for svc in services:
        cert = mtls.issue_certificate(svc)
        print(f"     {svc}: cert_id={cert['cert_id']}, trusted_by={cert['trusted_by']}")

    # Проверка соединений
    print("\n   Проверка mTLS соединений:")
    connections = [
        ("api-service", "database-service"),
        ("api-service", "cache-service"),
        ("worker-service", "database-service"),
        ("external-client", "api-service"),
    ]
    for client, server in connections:
        result = mtls.verify(client, server)
        status = result["status"]
        detail = result.get("cipher", result.get("error", ""))
        print(f"     {client} -> {server}: {status} {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: GitOps (ArgoCD, declarative deployments, drift detection)
# ─────────────────────────────────────────────────────────────────────────────
def demo_gitops():
    """Демонстрация концепций GitOps."""
    print("\n" + "=" * 70)
    print("DEMO 4: GitOps (ArgoCD, Declarative Deployments, Drift Detection)")
    print("=" * 70)

    # 4.1 GitOps принципы
    print("\n4.1 Принципы GitOps:")
    print("   1. Декларативность: желаемое состояние описывается в конфигах")
    print("   2. Версионность: Git как source of truth")
    print("   3. Автоматизация: PR -> CI -> CD pipeline")
    print("   4. Агентность: software agents автоматически применяют изменения\n")

    # 4.2 ArgoCD Application
    print("4.2 ArgoCD Application — описание приложения:")

    argocd_app = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Application",
        "metadata": {
            "name": "ml-inference",
            "namespace": "argocd",
        },
        "spec": {
            "project": "default",
            "source": {
                "repoURL": "https://github.com/example/ml-infra.git",
                "targetRevision": "main",
                "path": "apps/ml-inference/overlays/production",
            },
            "destination": {
                "server": "https://kubernetes.default.svc",
                "namespace": "production",
            },
            "syncPolicy": {
                "automated": {"prune": True, "selfHeal": True},
                "syncOptions": ["CreateNamespace=true"],
            },
        },
    }
    print(json.dumps(argocd_app, indent=2))

    # 4.3 Drift Detection
    print("\n4.3 Drift Detection (обнаружение отклонений):")
    print("   ArgoCD непрерывно сравнивает Git state с_cluster state\n")

    class DriftDetector:
        """Симуляция обнаружения drift между Git и кластером."""
        def __init__(self):
            self.git_state = {}
            self.cluster_state = {}
            self.drifts = []

        def set_git_state(self, state):
            """Установить состояние из Git."""
            self.git_state = state.copy()

        def set_cluster_state(self, state):
            """Установить текущее состояние кластера."""
            self.cluster_state = state.copy()

        def detect_drift(self):
            """Обнаружить отклонения."""
            self.drifts = []
            all_keys = set(self.git_state.keys()) | set(self.cluster_state.keys())
            for key in all_keys:
                git_val = self.git_state.get(key)
                cluster_val = self.cluster_state.get(key)
                if git_val != cluster_val:
                    self.drifts.append({
                        "field": key,
                        "git": git_val,
                        "cluster": cluster_val,
                    })
            return self.drifts

        def auto_heal(self):
            """Автоматическое восстановление (self-heal)."""
            healed = []
            for drift in self.drifts:
                self.cluster_state[drift["field"]] = drift["git"]
                healed.append(drift["field"])
            self.drifts = []
            return healed

    detector = DriftDetector()
    # Git state (source of truth)
    detector.set_git_state({
        "replicas": 3,
        "image": "ml-model:v2.1.0",
        "cpu_limit": "2",
        "memory_limit": "4Gi",
    })
    # Cluster state (с отклонениями)
    detector.set_cluster_state({
        "replicas": 2,        # drift: было 3, стало 2
        "image": "ml-model:v2.1.0",
        "cpu_limit": "4",     # drift: было 2, стало 4
        "memory_limit": "4Gi",
    })

    print("   Git state:     ", detector.git_state)
    print("   Cluster state: ", detector.cluster_state)

    drifts = detector.detect_drift()
    print(f"\n   Обнаружено отклонений: {len(drifts)}")
    for drift in drifts:
        print(f"     - {drift['field']}: git={drift['git']}, cluster={drift['cluster']}")

    # Self-heal
    healed = detector.auto_heal()
    print(f"\n   Self-heal: восстановлены поля {healed}")
    print(f"   Cluster state после восстановления: {detector.cluster_state}")

    # 4.4 GitOps Pipeline
    print("\n4.4 GitOps Pipeline:")
    print("   Developer -> Git Push -> CI Build -> Image Push -> Git Update -> ArgoCD Sync\n")

    pipeline_steps = [
        ("1. Developer",    "git push --feature/auth-v2",            "Код обновлён"),
        ("2. CI Pipeline",  "build -> test -> scan -> push image",   "Image: auth:v2.0.1"),
        ("3. Git Update",   "kustomize edit set image auth:v2.0.1", "Манифест обновлён"),
        ("4. ArgoCD Detect","diff: auth image v2.0.0 -> v2.0.1",    "Drift обнаружен"),
        ("5. ArgoCD Sync",  "kubectl apply -f manifests/",          "Rolling update"),
        ("6. Health Check", "readiness probe passed",                "Deployment ready"),
    ]

    for step, command, result in pipeline_steps:
        print(f"   {step}")
        print(f"     Команда: {command}")
        print(f"     Результат: {result}")
        print()

    # Итого
    print("   Итого GitOps pipeline:")
    print("   +------------------+--------------------------------------+-------------------+")
    print("   | Этап             | Действие                             | Результат         |")
    print("   +------------------+--------------------------------------+-------------------+")
    for step, command, result in pipeline_steps:
        step_short = step.split(". ")[1] if ". " in step else step
        print(f"   | {step_short:16} | {command:36} | {result:17} |")
    print("   +------------------+--------------------------------------+-------------------+")


if __name__ == "__main__":
    demo_kubernetes_advanced()
    demo_operators()
    demo_service_mesh()
    demo_gitops()
