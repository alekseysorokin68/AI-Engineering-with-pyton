"""155 — Cloud Computing: AWS/GCP/Azure basics, serverless, CDN

Темы:
  1. Cloud Service Models (IaaS, PaaS, SaaS, FaaS)
  2. Compute (VMs, containers, serverless functions)
  3. Storage (object storage, block storage, CDN)
  4. Serverless Patterns (functions, API gateway, event triggers)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import datetime

random.seed(42)


# ==========================================================================
# Демо 1 — Модели облачных сервисов: IaaS, PaaS, SaaS, FaaS
# ==========================================================================
def demo_cloud_models():
    """Демонстрация моделей облачных сервисов и их характеристик."""

    print("=" * 70)
    print("Демо 1: Модели облачных сервисов (Cloud Service Models)")
    print("=" * 70)

    # --- Подпример 1: IaaS (Infrastructure as a Service) ---
    print("\n--- 1.1 IaaS — Infrastructure as a Service ---")

    class IaaSSimulator:
        """Симуляция IaaS: аренда виртуальных машин."""
        def __init__(self):
            self._instances = {}
            self._next_id = 1
            self._pricing = {
                "t2.micro": {"vcpu": 1, "ram_gb": 1, "price_hour": 0.0116},
                "t2.small": {"vcpu": 1, "ram_gb": 2, "price_hour": 0.023},
                "t2.medium": {"vcpu": 2, "ram_gb": 4, "price_hour": 0.0464},
                "m5.large": {"vcpu": 2, "ram_gb": 8, "price_hour": 0.096},
            }

        def launch_instance(self, instance_type, os_image="ubuntu-22.04"):
            """Запуск виртуальной машины."""
            if instance_type not in self._pricing:
                return None, f"Неизвестный тип: {instance_type}"

            spec = self._pricing[instance_type]
            instance_id = f"i-{self._next_id:08x}"
            self._next_id += 1

            instance = {
                "id": instance_id,
                "type": instance_type,
                "os": os_image,
                "vcpu": spec["vcpu"],
                "ram_gb": spec["ram_gb"],
                "price_hour": spec["price_hour"],
                "state": "running",
                "uptime_hours": 0,
            }
            self._instances[instance_id] = instance
            print(f"  Запущена VM: {instance_id}")
            print(f"    Тип: {instance_type}, vCPU: {spec['vcpu']}, RAM: {spec['ram_gb']}GB")
            print(f"    ОС: {os_image}, Цена: ${spec['price_hour']:.4f}/час")
            return instance_id, "OK"

        def calculate_cost(self, hours):
            """Расчёт стоимости при заданном времени работы."""
            total = 0
            print(f"\n  Расчёт стоимости за {hours} часов:")
            for iid, inst in self._instances.items():
                cost = inst["price_hour"] * hours
                total += cost
                print(f"    {iid} ({inst['type']}): ${cost:.4f}")
            print(f"  ИТОГО: ${total:.4f}")
            return total

        def stop_instance(self, instance_id):
            """Остановка виртуальной машины."""
            if instance_id in self._instances:
                self._instances[instance_id]["state"] = "stopped"
                print(f"  VM {instance_id} остановлена")
                return True
            return False

    iaas = IaaSSimulator()

    # Запуск нескольких VM
    iaas.launch_instance("t2.micro", "ubuntu-22.04")
    iaas.launch_instance("t2.medium", "amazon-linux-2")
    iaas.launch_instance("m5.large", "ubuntu-22.04")

    # Расчёт стоимости
    iaas.calculate_cost(24)  # 24 часа работы

    # Остановка одной VM
    iaas.stop_instance("i-00000001")

    # --- Подпример 2: PaaS (Platform as a Service) ---
    print("\n--- 1.2 PaaS — Platform as a Service ---")

    class PaaSSimulator:
        """Симуляция PaaS: развёртывание приложений без управления инфраструктурой."""
        def __init__(self):
            self._apps = {}
            self._next_id = 1

        def deploy_app(self, name, runtime, config=None):
            """Развёртывание приложения на платформе."""
            app_id = f"app-{self._next_id:04d}"
            self._next_id += 1

            self._apps[app_id] = {
                "name": name,
                "runtime": runtime,
                "config": config or {},
                "status": "deployed",
                "url": f"https://{name}.cloudapp.example.com",
                "scale": 1,
            }
            print(f"  Приложение '{name}' развёрнуто")
            print(f"    ID: {app_id}")
            print(f"    Runtime: {runtime}")
            print(f"    URL: {self._apps[app_id]['url']}")
            return app_id

        def scale_app(self, app_id, replicas):
            """Масштабирование приложения."""
            if app_id in self._apps:
                self._apps[app_id]["scale"] = replicas
                print(f"  {self._apps[app_id]['name']} масштабирован до {replicas} реплик")
                return True
            return False

        def get_metrics(self, app_id):
            """Получение метрик приложения."""
            if app_id in self._apps:
                app = self._apps[app_id]
                # Симуляция метрик
                metrics = {
                    "requests_per_sec": random.randint(10, 100),
                    "avg_latency_ms": random.randint(50, 200),
                    "error_rate": round(random.uniform(0, 5), 2),
                    "cpu_usage": round(random.uniform(20, 80), 1),
                }
                return metrics
            return None

    paas = PaaSSimulator()

    # Развёртывание приложений
    app1 = paas.deploy_app("web-api", "python-3.11", {"port": 8000})
    app2 = paas.deploy_app("worker", "nodejs-18", {"queue": "tasks"})

    # Масштабирование
    paas.scale_app(app1, 3)
    paas.scale_app(app2, 2)

    # Метрики
    metrics = paas.get_metrics(app1)
    print(f"\n  Метрики web-api:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # --- Подпример 3: SaaS (Software as a Service) ---
    print("\n--- 1.3 SaaS — Software as a Service ---")

    class SaaSSimulator:
        """Симуляция SaaS: использование готового ПО как сервиса."""
        def __init__(self):
            self._plans = {
                "free": {"users": 1, "storage_gb": 1, "features": ["basic"]},
                "pro": {"users": 10, "storage_gb": 100, "features": ["basic", "analytics", "api"]},
                "enterprise": {"users": 1000, "storage_gb": 10000, "features": ["basic", "analytics", "api", "sso", "audit"]},
            }
            self._subscriptions = {}

        def subscribe(self, org_name, plan):
            """Подписка на SaaS-план."""
            if plan not in self._plans:
                return False, "Неизвестный план"
            plan_data = self._plans[plan]
            self._subscriptions[org_name] = {
                "plan": plan,
                "users_used": 0,
                "storage_used_gb": 0,
                "features": plan_data["features"],
                "max_users": plan_data["users"],
                "max_storage_gb": plan_data["storage_gb"],
            }
            print(f"  Организация '{org_name}' подписана на план '{plan}'")
            print(f"    Макс. пользователей: {plan_data['users']}")
            print(f"    Макс. хранилище: {plan_data['storage_gb']}GB")
            print(f"    Функции: {', '.join(plan_data['features'])}")
            return True, "OK"

        def add_users(self, org_name, count):
            """Добавление пользователей."""
            if org_name not in self._subscriptions:
                return False
            sub = self._subscriptions[org_name]
            sub["users_used"] += count
            if sub["users_used"] > sub["max_users"]:
                print(f"  Лимит пользователей превышен! ({sub['users_used']}/{sub['max_users']})")
                return False
            print(f"  Добавлено {count} пользователей. Всего: {sub['users_used']}/{sub['max_users']}")
            return True

        def check_features(self, org_name, feature):
            """Проверка доступности функции."""
            if org_name not in self._subscriptions:
                return False
            sub = self._subscriptions[org_name]
            has_feature = feature in sub["features"]
            print(f"  Функция '{feature}' для '{org_name}': {'доступна' if has_feature else 'недоступна'}")
            return has_feature

    saas = SaaSSimulator()

    # Демонстрация подписок
    saas.subscribe("startup", "free")
    saas.subscribe("company", "pro")
    saas.subscribe("enterprise", "enterprise")

    # Добавление пользователей
    saas.add_users("company", 5)

    # Проверка функций
    print()
    saas.check_features("startup", "api")     # Недоступна для free
    saas.check_features("company", "api")     # Доступна для pro
    saas.check_features("enterprise", "sso")  # Доступна для enterprise

    # --- Подпример 4: FaaS (Function as a Service) ---
    print("\n--- 1.4 FaaS — Function as a Service (Serverless) ---")

    class FaaSSimulator:
        """Симуляция FaaS: запуск функций без управления серверами."""
        def __init__(self):
            self._functions = {}
            self._invocations = []

        def deploy_function(self, name, runtime, handler):
            """Развёртывание функции."""
            func_id = f"func-{hashlib.md5(name.encode()).hexdigest()[:8]}"
            self._functions[func_id] = {
                "name": name,
                "runtime": runtime,
                "handler": handler,
                "invocations": 0,
                "avg_duration_ms": 0,
            }
            print(f"  Функция '{name}' развёрнута")
            print(f"    ID: {func_id}")
            print(f"    Runtime: {runtime}, Handler: {handler}")
            return func_id

        def invoke(self, func_id, payload=None):
            """Вызов функции с имитацией выполнения."""
            if func_id not in self._functions:
                return None, "Функция не найдена"

            func = self._functions[func_id]
            start_time = time.time()
            # Имитация времени выполнения
            exec_time = random.uniform(10, 200)
            time.sleep(exec_time / 1000)

            invocation = {
                "function": func["name"],
                "timestamp": datetime.datetime.now().isoformat(),
                "duration_ms": exec_time,
                "status": "success",
                "payload": payload,
            }
            self._invocations.append(invocation)
            func["invocations"] += 1

            print(f"  Вызов {func['name']}: {exec_time:.1f}ms")
            return invocation, "OK"

        def get_billing(self):
            """Расчёт стоимости: $0.20 за 1M вызовов + $0.0000166667 за ГБ-сек."""
            total_invocations = len(self._invocations)
            total_duration = sum(i["duration_ms"] for i in self._invocations)

            # Тарифы (упрощённые)
            invocation_cost = (total_invocations / 1_000_000) * 0.20
            duration_cost = (total_duration / 1000) * 0.0000166667 * 128  # 128MB RAM

            print(f"\n  Расчёт стоимости FaaS:")
            print(f"    Всего вызовов: {total_invocations}")
            print(f"    Стоимость вызовов: ${invocation_cost:.6f}")
            print(f"    Стоимость времени: ${duration_cost:.6f}")
            print(f"    ИТОГО: ${invocation_cost + duration_cost:.6f}")

    faas = FaaSSimulator()

    # Развёртывание функций
    f1 = faas.deploy_function("resize-image", "python-3.11", "handler.resize")
    f2 = faas.deploy_function("process-payment", "nodejs-18", "handler.pay")

    # Вызовы функций
    faas.invoke(f1, {"image_url": "photo.jpg", "width": 800})
    faas.invoke(f1, {"image_url": "banner.png", "width": 1200})
    faas.invoke(f2, {"amount": 99.99, "currency": "USD"})

    # Расчёт стоимости
    faas.get_billing()

    # Сравнение моделей
    print("\n--- Сравнение облачных моделей ---")
    models = {
        "IaaS": "Вы управляетесь инфраструктурой (VM, сети, хранилище)",
        "PaaS": "Платформа управляет инфраструктурой, вы — кодом",
        "SaaS": "Всё готово, вы просто используете ПО",
        "FaaS": "Вы пишете функции, платформа масштабирует автоматически",
    }
    for model, desc in models.items():
        print(f"  {model:6s} — {desc}")


# ==========================================================================
# Демо 2 — Вычисления: VM, контейнеры, serverless
# ==========================================================================
def demo_compute():
    """Демонстрация различных моделей вычислений в облаке."""

    print("\n" + "=" * 70)
    print("Демо 2: Вычисления (Compute)")
    print("=" * 70)

    # --- Подпример 1: Виртуальные машины ---
    print("\n--- 2.1 Виртуальные машины (VMs) ---")

    class VirtualMachine:
        """Модель виртуальной машины."""
        def __init__(self, name, vcpu, ram_gb, disk_gb):
            self.name = name
            self.vcpu = vcpu
            self.ram_gb = ram_gb
            self.disk_gb = disk_gb
            self.state = "stopped"
            self.os = None
            self._processes = []

        def start(self, os_image="ubuntu-22.04"):
            """Запуск VM с выбором ОС."""
            self.os = os_image
            self.state = "running"
            print(f"  VM '{self.name}' запущена: {self.vcpu} vCPU, {self.ram_gb}GB RAM, ОС={os_image}")

        def stop(self):
            """Остановка VM."""
            self.state = "stopped"
            self._processes.clear()
            print(f"  VM '{self.name}' остановлена")

        def run_process(self, name, cpu_needed, ram_needed):
            """Запуск процесса на VM."""
            total_cpu = sum(p["cpu"] for p in self._processes)
            total_ram = sum(p["ram"] for p in self._processes)

            if total_cpu + cpu_needed > self.vcpu:
                print(f"  Недостаточно CPU для '{name}' (нужно {cpu_needed}, доступно {self.vcpu - total_cpu})")
                return False
            if total_ram + ram_needed > self.ram_gb:
                print(f"  Недостаточно RAM для '{name}' (нужно {ram_needed}GB, доступно {self.ram_gb - total_ram}GB)")
                return False

            self._processes.append({"name": name, "cpu": cpu_needed, "ram": ram_needed})
            print(f"  Процесс '{name}' запущен (CPU={cpu_needed}, RAM={ram_needed}GB)")
            return True

        def status(self):
            """Получение статуса VM."""
            cpu_used = sum(p["cpu"] for p in self._processes)
            ram_used = sum(p["ram"] for p in self._processes)
            return {
                "name": self.name,
                "state": self.state,
                "os": self.os,
                "cpu_used": f"{cpu_used}/{self.vcpu}",
                "ram_used": f"{ram_used}/{self.ram_gb}GB",
                "processes": len(self._processes),
            }

    vm = VirtualMachine("web-server", vcpu=2, ram_gb=4, disk_gb=50)
    vm.start("ubuntu-22.04")
    vm.run_process("nginx", 1, 0.5)
    vm.run_process("python-api", 1, 1.5)
    vm.run_process("redis", 0.5, 0.5)

    status = vm.status()
    print(f"\n  Статус VM: {json.dumps(status, indent=2)}")

    # --- Подпример 2: Контейнеры (Docker-подобная модель) ---
    print("\n--- 2.2 Контейнеры (Containers) ---")

    class Container:
        """Модель контейнера (аналог Docker)."""
        def __init__(self, name, image, cpu_limit=1.0, memory_limit_mb=512):
            self.name = name
            self.image = image
            self.cpu_limit = cpu_limit
            self.memory_limit_mb = memory_limit_mb
            self.state = "created"
            self._env_vars = {}
            self._ports = []

        def add_env(self, key, value):
            """Добавление переменной окружения."""
            self._env_vars[key] = value

        def add_port(self, host_port, container_port):
            """Маппинг портов."""
            self._ports.append({"host": host_port, "container": container_port})

        def start(self):
            """Запуск контейнера."""
            self.state = "running"
            print(f"  Контейнер '{self.name}' запущен (image={self.image})")
            print(f"    Лимиты: CPU={self.cpu_limit}, RAM={self.memory_limit_mb}MB")
            if self._env_vars:
                print(f"    ENV: {self._env_vars}")
            if self._ports:
                print(f"    Порты: {self._ports}")

        def stop(self):
            """Остановка контейнера."""
            self.state = "stopped"
            print(f"  Контейнер '{self.name}' остановлен")

        def exec_command(self, command):
            """Выполнение команды внутри контейнера."""
            if self.state != "running":
                return None, "Контейнер не запущен"
            print(f"  [{self.name}] $ {command}")
            # Имитация вывода
            return f"output_of_{command.replace(' ', '_')}", "OK"

    # Запуск нескольких контейнеров
    c1 = Container("app", "python:3.11-slim", cpu_limit=2.0, memory_limit_mb=1024)
    c1.add_env("DATABASE_URL", "postgres://db:5432/mydb")
    c1.add_env("REDIS_URL", "redis://cache:6379")
    c1.add_port(8000, 8000)
    c1.start()

    c2 = Container("db", "postgres:15", cpu_limit=1.0, memory_limit_mb=512)
    c2.add_env("POSTGRES_PASSWORD", "secret")
    c2.add_port(5432, 5432)
    c2.start()

    # Выполнение команд
    c1.exec_command("pip install fastapi")
    c1.exec_command("python main.py")

    # --- Подпример 3: Serverless Functions ---
    print("\n--- 2.3 Serverless Functions ---")

    class ServerlessPlatform:
        """Платформа serverless вычислений."""
        def __init__(self):
            self._functions = {}
            self._cold_starts = 0
            self._warm_invocations = 0

        def deploy(self, name, runtime, memory_mb=128, timeout_sec=30):
            """Развёртывание функции."""
            self._functions[name] = {
                "runtime": runtime,
                "memory_mb": memory_mb,
                "timeout_sec": timeout_sec,
                "state": "cold",  # холодный запуск
                "last_invocation": None,
                "invocations": 0,
            }
            print(f"  Функция '{name}' развёрнута ({runtime}, {memory_mb}MB, timeout={timeout_sec}s)")

        def invoke(self, name, payload=None):
            """Вызов функции с обработкой cold/warm start."""
            if name not in self._functions:
                return None, "Функция не найдена"

            func = self._functions[name]
            is_cold = func["state"] == "cold"

            if is_cold:
                self._cold_starts += 1
                startup_time = random.uniform(100, 500)  # cold start: 100-500ms
                func["state"] = "warm"
                print(f"  COLD START '{name}': {startup_time:.0f}ms (инициализация runtime)")
            else:
                self._warm_invocations += 1
                startup_time = random.uniform(1, 10)  # warm: 1-10ms
                print(f"  WARM INVOKE '{name}': {startup_time:.1f}ms")

            exec_time = random.uniform(5, 50)
            total_time = startup_time + exec_time

            func["invocations"] += 1
            func["last_invocation"] = datetime.datetime.now().isoformat()

            return {
                "function": name,
                "cold_start": is_cold,
                "startup_ms": startup_time,
                "execution_ms": exec_time,
                "total_ms": total_time,
            }, "OK"

        def get_stats(self):
            """Статистика платформы."""
            total = self._cold_starts + self._warm_invocations
            cold_rate = (self._cold_starts / total * 100) if total else 0
            print(f"\n  Статистика серверless:")
            print(f"    Cold starts: {self._cold_starts} ({cold_rate:.1f}%)")
            print(f"    Warm invokes: {self._warm_invocations} ({100 - cold_rate:.1f}%)")

    sp = ServerlessPlatform()
    sp.deploy("process-image", "python-3.11", memory_mb=256)
    sp.deploy("send-email", "nodejs-18", memory_mb=128)

    # Несколько вызовов — первый будет cold start
    for i in range(5):
        result, status = sp.invoke("process-image")
        print(f"    Invoke {i+1}: {result['total_ms']:.1f}ms")

    sp.get_stats()

    # --- Подпример 4: Auto Scaling ---
    print("\n--- 2.4 Auto Scaling ---")

    class AutoScaler:
        """Автоматический масштабировщик на основе метрик."""
        def __init__(self, min_instances=1, max_instances=10, target_cpu=70):
            self.min_instances = min_instances
            self.max_instances = max_instances
            self.target_cpu = target_cpu
            self.instances = min_instances
            self.scaling_events = []

        def evaluate(self, current_cpu):
            """Оценка необходимости масштабирования."""
            if current_cpu > self.target_cpu + 10:
                # Нужно масштабирование вверх
                new_count = min(self.instances + 1, self.max_instances)
                if new_count > self.instances:
                    self.instances = new_count
                    self.scaling_events.append({
                        "action": "scale_up",
                        "from": self.instances - 1,
                        "to": new_count,
                        "cpu": current_cpu,
                    })
                    print(f"  SCALE UP: {self.instances-1} → {new_count} (CPU={current_cpu}%)")
            elif current_cpu < self.target_cpu - 20:
                # Нужно масштабирование вниз
                new_count = max(self.instances - 1, self.min_instances)
                if new_count < self.instances:
                    self.instances = new_count
                    self.scaling_events.append({
                        "action": "scale_down",
                        "from": self.instances + 1,
                        "to": new_count,
                        "cpu": current_cpu,
                    })
                    print(f"  SCALE DOWN: {self.instances+1} → {new_count} (CPU={current_cpu}%)")
            else:
                print(f"  OK: instances={self.instances}, CPU={current_cpu}% (target={self.target_cpu}%)")

    scaler = AutoScaler(min_instances=2, max_instances=8, target_cpu=70)

    # Симуляция изменения нагрузки
    cpu_values = [45, 55, 75, 85, 90, 80, 60, 40, 30, 25]
    for cpu in cpu_values:
        scaler.evaluate(cpu)

    print(f"\n  Итого событий масштабирования: {len(scaler.scaling_events)}")
    print(f"  Текущее количество инстансов: {scaler.instances}")


# ==========================================================================
# Демо 3 — Хранилище: object storage, block storage, CDN
# ==========================================================================
def demo_storage():
    """Демонстрация различных типов облачного хранилища."""

    print("\n" + "=" * 70)
    print("Демо 3: Хранилище (Storage)")
    print("=" * 70)

    # --- Подпример 1: Object Storage (S3-подобное) ---
    print("\n--- 3.1 Object Storage (S3-подобное хранилище) ---")

    class ObjectStorage:
        """Имитация S3-подобного объектного хранилища."""
        def __init__(self, bucket_name):
            self.bucket = bucket_name
            self._objects = {}
            self._versioning = False

        def put_object(self, key, data, content_type="application/octet-stream"):
            """Загрузка объекта."""
            version = 1
            if key in self._objects:
                version = self._objects[key]["version"] + 1

            checksum = hashlib.sha256(data.encode()).hexdigest()
            self._objects[key] = {
                "key": key,
                "data": data,
                "size_bytes": len(data.encode()),
                "content_type": content_type,
                "version": version,
                "checksum": checksum,
                "last_modified": datetime.datetime.now().isoformat(),
            }
            print(f"  PUT: {key} (version={version}, size={len(data)} bytes)")
            print(f"    SHA-256: {checksum[:32]}...")
            return version

        def get_object(self, key):
            """Получение объекта."""
            if key not in self._objects:
                return None, "Object not found"
            obj = self._objects[key]
            return obj, "OK"

        def delete_object(self, key):
            """Удаление объекта."""
            if key in self._objects:
                del self._objects[key]
                print(f"  DELETE: {key}")
                return True
            return False

        def list_objects(self, prefix=""):
            """Список объектов с фильтрацией по префиксу."""
            if prefix:
                items = [(k, v) for k, v in self._objects.items() if k.startswith(prefix)]
            else:
                items = list(self._objects.items())
            print(f"  LIST (prefix='{prefix}'): {len(items)} объектов")
            return items

        def get_bucket_info(self):
            """Информация о бакете."""
            total_size = sum(obj["size_bytes"] for obj in self._objects.values())
            print(f"\n  Информация о бакете '{self.bucket}':")
            print(f"    Объектов: {len(self._objects)}")
            print(f"    Общий размер: {total_size} bytes")
            return total_size

    s3 = ObjectStorage("my-ai-datasets")

    # Загрузка данных
    s3.put_object("models/resnet50.pt", "model_weights_binary_data...", "application/octet-stream")
    s3.put_object("data/train.jsonl", '{"text": "example"}\n' * 100, "application/json")
    s3.put_object("images/sample.jpg", "jpeg_binary_data...", "image/jpeg")
    s3.put_object("models/bert.pt", "bert_weights_data...", "application/octet-stream")

    # Получение объекта
    obj, status = s3.get_object("models/resnet50.pt")
    if obj:
        print(f"\n  GET models/resnet50.pt:")
        print(f"    Размер: {obj['size_bytes']} bytes")
        print(f"    Тип: {obj['content_type']}")
        print(f"    Версия: {obj['version']}")

    # Список объектов
    s3.list_objects("models/")

    # Информация о бакете
    s3.get_bucket_info()

    # --- Подпример 2: Block Storage (EBS-подобное) ---
    print("\n--- 3.2 Block Storage (EBS-подобное хранилище) ---")

    class BlockStorage:
        """Имитация блочного хранилища (аналог EBS)."""
        def __init__(self, volume_id, size_gb, volume_type="gp3"):
            self.volume_id = volume_id
            self.size_gb = size_gb
            self.volume_type = volume_type
            self.iops = self._get_default_iops()
            self.throughput_mbps = self._get_default_throughput()
            self.attached_to = None

        def _get_default_iops(self):
            """IOPS по умолчанию для типа тома."""
            defaults = {"gp3": 3000, "io2": 10000, "st1": 500, "sc1": 250}
            return defaults.get(self.volume_type, 3000)

        def _get_default_throughput(self):
            """Пропускная способность по умолчанию."""
            defaults = {"gp3": 125, "io2": 1000, "st1": 500, "sc1": 250}
            return defaults.get(self.volume_type, 125)

        def attach(self, instance_id):
            """Примонтирование тома к инстансу."""
            self.attached_to = instance_id
            print(f"  Tom '{self.volume_id}' примонтирован к {instance_id}")

        def detach(self):
            """Отмонтирование тома."""
            self.attached_to = None
            print(f"  Tom '{self.volume_id}' отмонтирован")

        def create_snapshot(self):
            """Создание снимка тома."""
            snapshot_id = f"snap-{hashlib.md5(self.volume_id.encode()).hexdigest()[:8]}"
            print(f"  Снимок создан: {snapshot_id}")
            return snapshot_id

        def resize(self, new_size_gb):
            """Изменение размера тома."""
            old_size = self.size_gb
            self.size_gb = new_size_gb
            print(f"  Размер изменён: {old_size}GB → {new_size_gb}GB")
            return True

        def status(self):
            """Получение статуса тома."""
            return {
                "volume_id": self.volume_id,
                "size_gb": self.size_gb,
                "type": self.volume_type,
                "iops": self.iops,
                "throughput": f"{self.throughput_mbps} MB/s",
                "attached_to": self.attached_to or "unattached",
            }

    ebs = BlockStorage("vol-abc123", size_gb=100, volume_type="gp3")
    print(f"  Создан том: {ebs.volume_id}")
    print(f"    Размер: {ebs.size_gb}GB, Тип: {ebs.volume_type}")
    print(f"    IOPS: {ebs.iops}, Пропускная способность: {ebs.throughput_mbps} MB/s")

    ebs.attach("i-00000001")
    ebs.create_snapshot()
    ebs.resize(200)

    print(f"\n  Статус: {json.dumps(ebs.status(), indent=2)}")

    # --- Подпример 3: CDN (Content Delivery Network) ---
    print("\n--- 3.3 CDN — Content Delivery Network ---")

    class CDNDistribution:
        """Имитация CDN для доставки контента."""
        def __init__(self, origin_domain):
            self.origin = origin_domain
            self._edge_locations = {
                "us-east-1": {"cache_hit_rate": 0.85, "latency_ms": 5},
                "eu-west-1": {"cache_hit_rate": 0.82, "latency_ms": 8},
                "ap-south-1": {"cache_hit_rate": 0.78, "latency_ms": 12},
            }
            self._cached_objects = {}
            self._requests = []

        def request(self, path, user_location="us-east-1"):
            """Обработка запроса через CDN."""
            is_cached = path in self._cached_objects
            edge = self._edge_locations.get(user_location, {"latency_ms": 50})

            if is_cached:
                latency = edge["latency_ms"]
                source = "EDGE"
                print(f"  CDN HIT: {path} (loc={user_location}, latency={latency}ms)")
            else:
                latency = edge["latency_ms"] + 100  # +100ms за обращение к origin
                source = "ORIGIN"
                self._cached_objects[path] = True
                print(f"  CDN MISS: {path} (loc={user_location}, latency={latency}ms → кэшировано)")

            self._requests.append({
                "path": path,
                "location": user_location,
                "cached": is_cached,
                "latency_ms": latency,
                "source": source,
            })
            return {"latency_ms": latency, "source": source}

        def invalidate(self, path):
            """Инвалидация кэша для конкретного пути."""
            if path in self._cached_objects:
                del self._cached_objects[path]
                print(f"  Инвалидация кэша: {path}")

        def stats(self):
            """Статистика CDN."""
            total = len(self._requests)
            hits = sum(1 for r in self._requests if r["cached"])
            avg_latency = sum(r["latency_ms"] for r in self._requests) / total if total else 0

            print(f"\n  Статистика CDN:")
            print(f"    Всего запросов: {total}")
            print(f"    Cache hits: {hits} ({hits/total*100:.1f}%)")
            print(f"    Средняя латентность: {avg_latency:.1f}ms")
            return {"total": total, "hits": hits, "avg_latency": avg_latency}

    cdn = CDNDistribution("origin.myapp.com")

    # Запросы от пользователей из разных регионов
    cdn.request("/static/app.js", "us-east-1")
    cdn.request("/static/style.css", "eu-west-1")
    cdn.request("/static/app.js", "ap-south-1")  # кэширован
    cdn.request("/api/data", "us-east-1")  # не кэшируется (API)

    cdn.stats()

    # --- Подпример 4: Storage Tiers ---
    print("\n--- 3.4 Storage Tiers (уровни хранилища) ---")

    storage_tiers = {
        "Hot (S3 Standard)": {
            "description": "Частый доступ, высокая производительность",
            "cost_per_gb_month": 0.023,
            "retrieval_cost": 0.0,
            "min_duration": "0 дней",
            "use_case": "Активные данные, веб-приложения",
        },
        "Warm (S3 IA)": {
            "description": "Инфrequent access, ниже стоимость",
            "cost_per_gb_month": 0.0125,
            "retrieval_cost": 0.01,
            "min_duration": "30 дней",
            "use_case": "Бэкапы, данные для аналитики",
        },
        "Cold (S3 Glacier)": {
            "description": "Редкий доступ, низкая стоимость",
            "cost_per_gb_month": 0.004,
            "retrieval_cost": 0.03,
            "min_duration": "90 дней",
            "use_case": "Архив, долгосрочное хранение",
        },
        "Archive (Glacier Deep)": {
            "description": "Минимальный доступ, минимальная стоимость",
            "cost_per_gb_month": 0.00099,
            "retrieval_cost": 0.02,
            "min_duration": "180 дней",
            "use_case": "Соответствие регуляциям, архив на годы",
        },
    }

    print(f"  {'Уровень':<25} {'Цена/GB/мес':<14} {'Стоимость чтения':<18} {'Мин. хранение'}")
    print(f"  {'-'*80}")
    for tier, info in storage_tiers.items():
        print(f"  {tier:<25} ${info['cost_per_gb_month']:<13.4f} "
              f"${info['retrieval_cost']:<17.3f} {info['min_duration']}")

    # Расчёт стоимости для 1TB данных
    tb = 1024  # GB
    print(f"\n  Стоимость хранения 1TB данных:")
    for tier, info in storage_tiers.items():
        monthly = tb * info["cost_per_gb_month"]
        print(f"    {tier:<25} ${monthly:.2f}/мес")


# ==========================================================================
# Демо 4 — Serverless паттерны: functions, API gateway, event triggers
# ==========================================================================
def demo_serverless_patterns():
    """Демонстрация паттернов serverless архитектуры."""

    print("\n" + "=" * 70)
    print("Демо 4: Serverless Patterns")
    print("=" * 70)

    # --- Подпример 1: Function Composition ---
    print("\n--- 4.1 Function Composition (композиция функций) ---")

    class ServerlessOrchestrator:
        """Оркестратор serverless функций (аналог AWS Step Functions)."""
        def __init__(self):
            self._functions = {}
            self._workflows = {}
            self._executions = []

        def register_function(self, name, handler):
            """Регистрация функции."""
            self._functions[name] = handler
            print(f"  Функция зарегистрирована: {name}")

        def create_workflow(self, name, steps):
            """Создание workflow из последовательности шагов."""
            self._workflows[name] = steps
            print(f"  Workflow создан: {name} ({len(steps)} шагов)")

        def execute_workflow(self, name, initial_input):
            """Выполнение workflow."""
            if name not in self._workflows:
                return None, "Workflow не найден"

            steps = self._workflows[name]
            current_input = initial_input
            execution_log = []

            print(f"\n  Запуск workflow '{name}':")
            for i, step in enumerate(steps):
                func_name = step["function"]
                if func_name not in self._functions:
                    execution_log.append({"step": i, "function": func_name, "status": "error"})
                    continue

                # Выполнение функции
                start_time = time.time()
                result = self._functions[func_name](current_input)
                duration_ms = (time.time() - start_time) * 1000

                execution_log.append({
                    "step": i,
                    "function": func_name,
                    "input": str(current_input)[:50],
                    "output": str(result)[:50],
                    "duration_ms": round(duration_ms, 2),
                })
                current_input = result
                print(f"    Шаг {i+1}: {func_name} → {str(result)[:60]}")

            self._executions.append({
                "workflow": name,
                "log": execution_log,
                "status": "completed",
            })
            return current_input, "OK"

    # Регистрация функций
    orchestrator = ServerlessOrchestrator()

    def validate_input(data):
        """Функция валидации."""
        return {"valid": True, "data": data.get("text", "").strip()}

    def process_text(data):
        """Функция обработки текста."""
        words = data.get("data", "").split()
        return {"word_count": len(words), "char_count": len(data.get("data", ""))}

    def analyze_sentiment(data):
        """Функция анализа тональности."""
        text = data.get("data", "")
        # Простой анализ по ключевым словам
        positive = ["хорошо", "отлично", "прекрасно", "нравится"]
        negative = ["плохо", "ужасно", "отвратительно", "ненавижу"]
        score = sum(1 for w in positive if w in text) - sum(1 for w in negative if w in text)
        return {"sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral", "score": score}

    def save_result(data):
        """Функция сохранения результата."""
        return {"saved": True, "result": data}

    orchestrator.register_function("validate", validate_input)
    orchestrator.register_function("process", process_text)
    orchestrator.register_function("analyze", analyze_sentiment)
    orchestrator.register_function("save", save_result)

    # Создание и выполнение workflow
    orchestrator.create_workflow("text-pipeline", [
        {"function": "validate"},
        {"function": "process"},
        {"function": "analyze"},
        {"function": "save"},
    ])

    result, status = orchestrator.execute_workflow("text-pipeline", {"text": "Это отличный текст для анализа"})
    print(f"\n  Итоговый результат: {result}")

    # --- Подпример 2: API Gateway ---
    print("\n--- 4.2 API Gateway (шлюз API) ---")

    class APIGateway:
        """Имитация API Gateway с маршрутизацией и трансформацией."""
        def __init__(self):
            self._routes = {}
            self._middleware = []
            self._request_log = []

        def add_route(self, method, path, handler, auth_required=False):
            """Добавление маршрута."""
            route_key = f"{method} {path}"
            self._routes[route_key] = {
                "handler": handler,
                "auth_required": auth_required,
            }
            print(f"  Route: {route_key} (auth={auth_required})")

        def add_middleware(self, middleware_fn):
            """Добавление middleware."""
            self._middleware.append(middleware_fn)

        def handle_request(self, method, path, headers=None, body=None):
            """Обработка входящего запроса."""
            route_key = f"{method} {path}"

            # Логирование
            start_time = time.time()

            # Проверка маршрута
            if route_key not in self._routes:
                return {"statusCode": 404, "body": "Not Found"}

            route = self._routes[route_key]

            # Выполнение middleware
            request_data = {
                "method": method,
                "path": path,
                "headers": headers or {},
                "body": body,
            }

            for mw in self._middleware:
                request_data = mw(request_data)
                if request_data is None:
                    return {"statusCode": 403, "body": "Forbidden"}

            # Вызов обработчика
            try:
                response = route["handler"](request_data)
                duration_ms = (time.time() - start_time) * 1000

                self._request_log.append({
                    "method": method,
                    "path": path,
                    "status": response.get("statusCode", 200),
                    "duration_ms": round(duration_ms, 2),
                })

                print(f"  {method} {path} → {response.get('statusCode', 200)} ({duration_ms:.1f}ms)")
                return response
            except Exception as e:
                return {"statusCode": 500, "body": str(e)}

    def auth_middleware(request):
        """Middleware проверки аутентификации."""
        token = request["headers"].get("Authorization", "")
        if not token.startswith("Bearer "):
            print("    Middleware: отсутствует токен авторизации")
            return None  # блокируем запрос
        request["user"] = "authenticated_user"
        return request

    def logging_middleware(request):
        """Middleware логирования."""
        print(f"    Middleware: запрос {request['method']} {request['path']}")
        return request

    gateway = APIGateway()
    gateway.add_middleware(logging_middleware)
    gateway.add_middleware(auth_middleware)

    # Обработчики
    def get_users(request):
        return {"statusCode": 200, "body": {"users": ["alice", "bob"]}}

    def create_user(request):
        return {"statusCode": 201, "body": {"message": "User created"}}

    gateway.add_route("GET", "/api/users", get_users, auth_required=True)
    gateway.add_route("POST", "/api/users", create_user, auth_required=True)

    # Запросы
    print()
    gateway.handle_request("GET", "/api/users", {"Authorization": "Bearer token123"})
    gateway.handle_request("GET", "/api/users", {})  # без токена
    gateway.handle_request("POST", "/api/users", {"Authorization": "Bearer token123"}, {"name": "new_user"})

    # --- Подпример 3: Event-Driven Architecture ---
    print("\n--- 4.3 Event-Driven Architecture (событийно-ориентированная архитектура) ---")

    class EventBridge:
        """Шина событий (аналог AWS EventBridge)."""
        def __init__(self):
            self._rules = []  # rules: (pattern, target_function)
            self._events = []

        def add_rule(self, pattern, target_fn, description=""):
            """Добавление правила обработки событий."""
            self._rules.append({
                "pattern": pattern,
                "target": target_fn,
                "description": description,
            })
            print(f"  Правило: {pattern} → {description}")

        def put_event(self, event_type, detail):
            """Отправка события на шину."""
            event = {
                "type": event_type,
                "detail": detail,
                "timestamp": datetime.datetime.now().isoformat(),
                "id": hashlib.md5(json.dumps(detail).encode()).hexdigest()[:8],
            }
            self._events.append(event)

            # Поиск подходящих правил
            matched = []
            for rule in self._rules:
                if self._match_pattern(rule["pattern"], event_type):
                    matched.append(rule)

            print(f"\n  Событие: {event_type}")
            print(f"    Detail: {json.dumps(detail)[:60]}...")
            print(f"    Совпавшие правила: {len(matched)}")

            # Выполнение обработчиков
            for rule in matched:
                result = rule["target"](event)
                print(f"    → {rule['description']}: {result}")

            return event

        def _match_pattern(self, pattern, event_type):
            """Проверка соответствия шаблона типу события."""
            if pattern == "*":
                return True
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                return event_type.startswith(prefix)
            return pattern == event_type

    bridge = EventBridge()

    # Обработчики событий
    def on_order_created(event):
        detail = event["detail"]
        return f"Заказ #{detail.get('order_id')} создан, сумма: {detail.get('amount')}"

    def on_payment_received(event):
        detail = event["detail"]
        return f"Оплата получена: {detail.get('amount')} {detail.get('currency')}"

    def on_inventory_low(event):
        detail = event["detail"]
        return f"Низкий запас: {detail.get('product')}, осталось: {detail.get('quantity')}"

    # Регистрация правил
    bridge.add_rule("order.*", on_order_created, "Обработка заказов")
    bridge.add_rule("payment.received", on_payment_received, "Обработка оплат")
    bridge.add_rule("*", lambda e: "Логирование события", "Логирование всех событий")

    # Генерация событий
    bridge.put_event("order.created", {"order_id": 12345, "amount": 2990, "items": 3})
    bridge.put_event("payment.received", {"amount": 2990, "currency": "RUB", "method": "card"})
    bridge.put_event("inventory.low", {"product": "Ноутбук", "quantity": 2})

    # --- Подпример 4: Event Source Mapping ---
    print("\n--- 4.4 Event Source Mapping (маппинг источников событий) ---")

    class EventSourceMapping:
        """Маппинг источников событий на serverless функции."""
        def __init__(self):
            self._sources = {}
            self._buffers = collections.defaultdict(list)

        def register_source(self, source_type, source_config, target_function):
            """Регистрация источника событий."""
            source_id = f"src-{hashlib.md5(json.dumps(source_config).encode()).hexdigest()[:8]}"
            self._sources[source_id] = {
                "type": source_type,
                "config": source_config,
                "target": target_function,
                "events_processed": 0,
            }
            print(f"  Источник зарегистрирован: {source_type} → {source_id}")
            return source_id

        def push_event(self, source_id, event_data):
            """Отправка события через зарегистрированный источник."""
            if source_id not in self._sources:
                return None, "Источник не найден"

            source = self._sources[source_id]

            # Добавление в буфер (batch processing)
            self._buffers[source_id].append(event_data)

            # Обработка пакета (по умолчанию батч = 10 или по таймауту)
            batch_size = 10
            if len(self._buffers[source_id]) >= batch_size:
                batch = self._buffers[source_id][:batch_size]
                self._buffers[source_id] = self._buffers[source_id][batch_size:]

                # Вызов целевой функции с пакетом
                result = source["target"](batch)
                source["events_processed"] += len(batch)

                print(f"  Батч обработан: {len(batch)} событий → {result}")
                return result, "OK"

            return None, "Buffering"

        def flush(self):
            """Принудительная обработка буфера."""
            for source_id, buffer in self._buffers.items():
                if buffer:
                    source = self._sources[source_id]
                    batch = buffer.copy()
                    buffer.clear()

                    result = source["target"](batch)
                    source["events_processed"] += len(batch)

                    print(f"  Flush: {len(batch)} событий обработано → {result}")

    mapping = EventSourceMapping()

    def process_log_batch(batch):
        """Обработка батча логов."""
        levels = collections.Counter(log.get("level", "INFO") for log in batch)
        return f"Обработано {len(batch)} логов: {dict(levels)}"

    def process_metric_batch(batch):
        """Обработка батча метрик."""
        avg_value = sum(m.get("value", 0) for m in batch) / len(batch)
        return f"Обработано {len(batch)} метрик, среднее: {avg_value:.2f}"

    # Регистрация источников
    log_src = mapping.register_source(
        "kinesis",
        {"stream": "app-logs", "batch_size": 10},
        process_log_batch
    )
    metric_src = mapping.register_source(
        "kafka",
        {"topic": "metrics", "consumer_group": "processor"},
        process_metric_batch
    )

    # Отправка событий
    for i in range(12):
        level = random.choice(["INFO", "WARN", "ERROR", "INFO"])
        mapping.push_event(log_src, {"level": level, "message": f"Log {i}"})

    for i in range(8):
        mapping.push_event(metric_src, {"name": "cpu_usage", "value": random.uniform(30, 90)})

    # Flush остатков
    mapping.flush()

    # Статистика
    print(f"\n  Статистика обработки:")
    for sid, source in mapping._sources.items():
        print(f"    {sid}: {source['events_processed']} событий обработано")


# ==========================================================================
# Точка входа
# ==========================================================================
if __name__ == "__main__":
    print("УРОК 155: CLOUD COMPUTING BASICS")
    print("Темы: Cloud Service Models, Compute, Storage, Serverless Patterns")
    print("=" * 70)

    demo_cloud_models()
    demo_compute()
    demo_storage()
    demo_serverless_patterns()

    print("\n" + "=" * 70)
    print("Урок завершён. Все 4 демо выполнены успешно.")
    print("=" * 70)
