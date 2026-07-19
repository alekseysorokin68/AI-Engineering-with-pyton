"""149 — Infrastructure as Code: Terraform concepts, declarative configuration

Темы:
  1. Принципы IaC (декларативный vs императивный подход, управление состоянием)
  2. Определения ресурсов (провайдеры, ресурсы, источники данных)
  3. Модули и композиция (переиспользуемые модули, выходы переменных)
  4. Состояние и планирование (файл состояния, цикл plan/apply, обнаружение дрейфа)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import uuid

random.seed(42)

# =============================================================================
# Демо 1: Принципы IaC — декларативный vs императивный, управление состоянием
# =============================================================================

def demo_iac_principles():
    print("=" * 70)
    print("Демо 1: Принципы IaC — декларативный vs императивный, управление состоянием")
    print("=" * 70)

    # --- 1.1 Декларативный vs императивный подход ---
    print("\n--- 1.1 Декларативный vs императивный подход ---")

    # Императивный: ОПИСЫВАЕМ КАК делать (пошаговые инструкции)
    print("  Императивный подход (как делать):")
    imperative_steps = [
        "1. Создать виртуальную машину",
        "2. Установить Python 3.11",
        "3. Скопировать код приложения",
        "4. Настроить Nginx как reverse proxy",
        "5. Запустить приложение",
        "6. Настроить файрвол",
        "7. Настроить SSL-сертификат",
    ]
    for step in imperative_steps:
        print(f"    {step}")

    # Декларативный: ОПИСЫВАЕМ ЧТО хотим получить
    print("\n  Декларативный подход (что получить):")
    declarative_config = {
        "resource": "virtual_machine",
        "name": "web_server",
        "specs": {
            "os": "Ubuntu 22.04",
            "ram_gb": 4,
            "cpu": 2,
            "disk_gb": 50
        },
        "services": ["python3.11", "nginx"],
        "config": {
            "nginx": {"proxy_pass": "http://localhost:8000", "ssl": True},
            "firewall": {"ports": [80, 443, 22]}
        }
    }
    print(f"  Конфигурация:")
    print(f"  {json.dumps(declarative_config, indent=4, ensure_ascii=False)}")

    # Сравнение
    print("\n  Сравнение:")
    print(f"    Императивный: 7 шагов, контроль на каждом этапе")
    print(f"    Декларативный: 1 описание, система сама вычисляет план")
    print(f"    Идеомпотентность: повторный apply даёт тот же результат")

    # --- 1.2 Управление состоянием (state management) ---
    print("\n--- 1.2 Управление состоянием ---")

    class StateManager:
        """Менеджер состояния инфраструктуры.
        State file хранит текущее состояние всех ресурсов."""
        def __init__(self):
            self.state = {
                "version": "1.0",
                "resources": {},
                "metadata": {"last_modified": None, "serial": 0}
            }

        def add_resource(self, resource_type, name, attributes):
            """Добавление ресурса в состояние"""
            key = f"{resource_type}.{name}"
            self.state["resources"][key] = {
                "type": resource_type,
                "name": name,
                "attributes": attributes,
                "created_at": time.time()
            }
            self.state["metadata"]["serial"] += 1
            self.state["metadata"]["last_modified"] = time.time()

        def remove_resource(self, resource_type, name):
            """Удаление ресурса из состояния"""
            key = f"{resource_type}.{name}"
            if key in self.state["resources"]:
                del self.state["resources"][key]
                self.state["metadata"]["serial"] += 1

        def get_resource(self, resource_type, name):
            """Получение ресурса из состояния"""
            key = f"{resource_type}.{name}"
            return self.state["resources"].get(key)

        def resource_count(self):
            return len(self.state["resources"])

    sm = StateManager()

    # Добавление ресурсов
    sm.add_resource("aws_instance", "web", {
        "ami": "ami-0c55b159cbfafe1f0",
        "instance_type": "t3.medium",
        "subnet_id": "subnet-abc123",
        "status": "running"
    })

    sm.add_resource("aws_db_instance", "main_db", {
        "engine": "postgresql",
        "engine_version": "15.3",
        "instance_class": "db.t3.micro",
        "allocated_storage": 20,
        "status": "available"
    })

    sm.add_resource("aws_s3_bucket", "logs", {
        "bucket_name": "my-app-logs-2024",
        "versioning": True,
        "encryption": "AES256"
    })

    print(f"  Ресурсов в состоянии: {sm.resource_count()}")
    for key, res in sm.state["resources"].items():
        attrs = res["attributes"]
        status = attrs.get("status", "created")
        print(f"    {key}: {status}")
        for k, v in attrs.items():
            if k != "status":
                print(f"      {k}: {v}")

    # --- 1.3 Идеомпотентность ---
    print("\n--- 1.3 Идеомпотентность ---")

    def apply_terraform(desired_state, current_state):
        """Применение desired state к current state.
        Идеомпотентность: повторный вызов с тем же desired
        не изменяет current state."""
        changes = []

        for resource, desired in desired_state.items():
            current = current_state.get(resource)
            if current is None:
                changes.append(f"  + {resource}: создать ({desired})")
                current_state[resource] = desired
            elif current != desired:
                changes.append(f"  ~ {resource}: обновить {current} → {desired}")
                current_state[resource] = desired
            else:
                changes.append(f"  = {resource}: без изменений (идеомпотентно)")

        # Удаление ресурсов, которых нет в desired
        for resource in list(current_state.keys()):
            if resource not in desired_state:
                changes.append(f"  - {resource}: удалить")
                del current_state[resource]

        return changes

    # Первый apply
    desired = {"vpc": "10.0.0.0/16", "subnet": "10.0.1.0/24", "sg": "web-sg"}
    current = {}

    print("  Первый apply:")
    changes = apply_terraform(desired, current)
    for c in changes:
        print(f"    {c}")

    # Повторный apply (ничего не должно измениться — идеомпотентность)
    print("\n  Повторный apply (ideomponent):")
    changes = apply_terraform(desired, current)
    for c in changes:
        print(f"    {c}")

    # Изменение конфигурации
    desired_updated = {"vpc": "10.0.0.0/16", "subnet": "10.0.2.0/24", "sg": "web-sg", "igw": "igw-001"}
    print("\n  Apply с изменениями:")
    changes = apply_terraform(desired_updated, current)
    for c in changes:
        print(f"    {c}")

    # --- 1.4 Версионирование конфигурации ---
    print("\n--- 1.4 Версионирование конфигурации ---")

    class ConfigVersion:
        def __init__(self, version, config, message):
            self.version = version
            self.config = config.copy()
            self.message = message
            self.timestamp = time.time()
            self.hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

    versions = [
        ConfigVersion("1.0.0", {"instance_type": "t3.micro", "count": 1}, "Начальная конфигурация"),
        ConfigVersion("1.1.0", {"instance_type": "t3.medium", "count": 2}, "Масштабирование"),
        ConfigVersion("1.2.0", {"instance_type": "t3.large", "count": 3, "load_balancer": True}, "Добавлен LB"),
    ]

    print("  История изменений конфигурации:")
    for v in versions:
        print(f"\n    v{v.version} ({v.message})")
        print(f"      Hash: {v.hash}")
        for k, val in v.config.items():
            print(f"      {k}: {val}")

    # Сравнение версий
    print("\n  diff между v1.1.0 и v1.2.0:")
    old = versions[1].config
    new = versions[2].config
    all_keys = set(old.keys()) | set(new.keys())
    for key in sorted(all_keys):
        if key in old and key not in new:
            print(f"    - {key}: {old[key]}")
        elif key not in old and key in new:
            print(f"    + {key}: {new[key]}")
        elif old[key] != new[key]:
            print(f"    ~ {key}: {old[key]} → {new[key]}")


# =============================================================================
# Демо 2: Определения ресурсов — провайдеры, ресурсы, источники данных
# =============================================================================

def demo_resource_definitions():
    print("\n\n" + "=" * 70)
    print("Демо 2: Определения ресурсов — провайдеры, ресурсы, источники данных")
    print("=" * 70)

    # --- 2.1 Провайдеры ---
    print("\n--- 2.1 Провайдеры (providers) ---")

    class Provider:
        def __init__(self, name, region=None, credentials=None):
            self.name = name
            self.region = region
            self.credentials = credentials or {}
            self.resources = []

        def create_resource(self, resource_type, config):
            """Создание ресурса у провайдера"""
            resource = {
                "provider": self.name,
                "type": resource_type,
                "config": config,
                "id": f"{self.name}_{resource_type}_{str(uuid.uuid4())[:6]}"
            }
            self.resources.append(resource)
            return resource

    # Определяем провайдеры
    aws = Provider("aws", region="eu-west-1")
    gcp = Provider("gcp", region="europe-west1")
    azure = Provider("azure", region="West Europe")

    print(f"  Провайдеры:")
    for p in [aws, gcp, azure]:
        print(f"    {p.name}: регион={p.region}")

    # Создание ресурсов
    vm = aws.create_resource("instance", {
        "ami": "ami-0c55b159cbfafe1f0",
        "instance_type": "t3.medium",
        "key_name": "my-key"
    })
    print(f"\n  Создан ресурс: {vm['id']}")
    print(f"  Конфигурация: {json.dumps(vm['config'], indent=4)}")

    bucket = gcp.create_resource("bucket", {
        "name": "my-data-bucket",
        "location": "EU",
        "storage_class": "STANDARD"
    })
    print(f"\n  Создан ресурс: {bucket['id']}")

    # --- 2.2 Ресурсы и их зависимости ---
    print("\n--- 2.2 Ресурсы и зависимости ---")

    class ResourceGraph:
        """Граф зависимостей ресурсов — определяет порядок создания/удаления"""
        def __init__(self):
            self.resources = {}
            self.dependencies = collections.defaultdict(list)

        def add_resource(self, name, resource_type, config):
            self.resources[name] = {"type": resource_type, "config": config, "status": "planned"}

        def add_dependency(self, resource, depends_on):
            """Зависимость: resource зависит от depends_on"""
            self.dependencies[resource].append(depends_on)

        def topological_sort(self):
            """Топологическая сортировка — определяет порядок создания"""
            visited = set()
            order = []

            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                for dep in self.dependencies.get(node, []):
                    dfs(dep)
                order.append(node)

            for resource in self.resources:
                dfs(resource)
            return order

        def plan(self):
            """План создания ресурсов"""
            order = self.topological_sort()
            print("  План создания ресурсов (порядок по зависимостям):")
            for i, name in enumerate(order, 1):
                res = self.resources[name]
                deps = self.dependencies.get(name, [])
                dep_str = f" (зависит от: {', '.join(deps)})" if deps else ""
                print(f"    {i}. {res['type']}.{name}{dep_str}")
            return order

    rg = ResourceGraph()

    # Определяем ресурсы и зависимости
    rg.add_resource("vpc", "aws_vpc", {"cidr_block": "10.0.0.0/16"})
    rg.add_resource("subnet", "aws_subnet", {"cidr_block": "10.0.1.0/24"})
    rg.add_resource("igw", "aws_internet_gateway", {})
    rg.add_resource("sg", "aws_security_group", {"ingress": [80, 443]})
    rg.add_resource("instance", "aws_instance", {"ami": "ami-0c55b159cbfafe1f0"})
    rg.add_resource("eip", "aws_eip", {})

    # Зависимости
    rg.add_dependency("subnet", "vpc")
    rg.add_dependency("igw", "vpc")
    rg.add_dependency("sg", "vpc")
    rg.add_dependency("instance", "subnet")
    rg.add_dependency("instance", "sg")
    rg.add_dependency("eip", "instance")

    order = rg.plan()

    # --- 2.3 Источники данных (data sources) ---
    print("\n--- 2.3 Источники данных (data sources) ---")

    class DataSource:
        """Источник данных — чтение информации из внешнего источника
        (например, AMI Ubuntu最新版, существующий VPC)"""
        def __init__(self, name, source_type, query):
            self.name = name
            self.source_type = source_type
            self.query = query
            self.result = None

        def lookup(self):
            """Поиск данных по запросу"""
            # Имитация поиска
            self.result = {
                "id": f"data-{str(uuid.uuid4())[:8]}",
                "type": self.source_type,
                "query": self.query,
                "found": True
            }
            return self.result

    # Поиск AMI Ubuntu最新版
    ami_source = DataSource(
        "latest_ubuntu",
        "aws_ami",
        {"name": "ubuntu/images/hvm-ssd/ubuntu-*", "owners": ["099720109477"]}
    )
    ami_result = ami_source.lookup()
    print(f"  Источник данных: {ami_source.name}")
    print(f"    Тип: {ami_source.source_type}")
    print(f"    Запрос: {json.dumps(ami_source.query)}")
    print(f"    Результат: ID={ami_result['id']}")

    # Поиск существующего VPC
    vpc_source = DataSource(
        "existing_vpc",
        "aws_vpc",
        {"filter": {"tag:Name": "production-vpc"}}
    )
    vpc_result = vpc_source.lookup()
    print(f"\n  Источник данных: {vpc_source.name}")
    print(f"    Тип: {vpc_source.source_type}")
    print(f"    Результат: {vpc_result}")

    # --- 2.4 Outputs (выходы ресурсов) ---
    print("\n--- 2.4 Outputs (выходы ресурсов) ---")

    class ResourceOutput:
        """Выходы ресурсов — значения, которые можно передать другим ресурсам
        или модулям (например, IP-адрес созданной VM)"""
        def __init__(self):
            self.outputs = {}

        def add_output(self, name, value, description=""):
            self.outputs[name] = {"value": value, "description": description}

        def get(self, name):
            return self.outputs.get(name, {}).get("value")

        def summary(self):
            print("  Outputs:")
            for name, info in self.outputs.items():
                print(f"    {name}: {info['value']}")
                if info['description']:
                    print(f"      Описание: {info['description']}")

    outputs = ResourceOutput()
    outputs.add_output("vpc_id", "vpc-abc123", "ID созданной VPC")
    outputs.add_output("subnet_id", "subnet-def456", "ID подсети")
    outputs.add_output("instance_public_ip", "52.14.23.100", "Публичный IP инстанса")
    outputs.add_output("instance_private_ip", "10.0.1.50", "Приватный IP инстанса")
    outputs.add_output("security_group_id", "sg-ghi789", "ID группы безопасности")

    outputs.summary()

    # Использование output в другом ресурсе
    print(f"\n  Использование outputs:")
    print(f"    SSH: ssh ubuntu@{outputs.get('instance_public_ip')}")
    print(f"    Internal: {outputs.get('instance_private_ip')}:8000")


# =============================================================================
# Демо 3: Модули и композиция — переиспользуемые модули, выходы переменных
# =============================================================================

def demo_modules():
    print("\n\n" + "=" * 70)
    print("Демо 3: Модули и композиция — переиспользуемые модули, выходы переменных")
    print("=" * 70)

    # --- 3.1 Определение модуля ---
    print("\n--- 3.1 Определение модуля ---")

    class TerraformModule:
        def __init__(self, name, description=""):
            self.name = name
            self.description = description
            self.variables = {}
            self.resources = []
            self.outputs = {}

        def add_variable(self, name, default=None, description="", type_="string"):
            self.variables[name] = {
                "default": default,
                "description": description,
                "type": type_
            }

        def add_resource(self, resource_type, name, config_template):
            self.resources.append({
                "type": resource_type,
                "name": name,
                "config": config_template
            })

        def add_output(self, name, value_template, description=""):
            self.outputs[name] = {"value": value_template, "description": description}

        def instantiate(self, var_values):
            """Создание экземпляра модуля с переданными переменными"""
            # Подстановка переменных в шаблоны
            resolved_resources = []
            for res in self.resources:
                resolved_config = {}
                for key, val in res["config"].items():
                    if isinstance(val, str) and val.startswith("var."):
                        var_name = val[4:]
                        resolved_config[key] = var_values.get(var_name, self.variables[var_name]["default"])
                    else:
                        resolved_config[key] = val
                resolved_resources.append({
                    "type": res["type"],
                    "name": res["name"],
                    "config": resolved_config
                })

            resolved_outputs = {}
            for name, info in self.outputs.items():
                if isinstance(info["value"], str) and info["value"].startswith("var."):
                    var_name = info["value"][4:]
                    resolved_outputs[name] = var_values.get(var_name)
                else:
                    resolved_outputs[name] = info["value"]

            return {"resources": resolved_resources, "outputs": resolved_outputs}

    # Создаём модуль веб-сервера
    web_module = TerraformModule("web_server", "Модуль веб-сервера с VPC")
    web_module.add_variable("instance_type", "t3.micro", "Тип EC2 инстанса", "string")
    web_module.add_variable("ami_id", "ami-0c55b159cbfafe1f0", "ID AMI", "string")
    web_module.add_variable("subnet_cidr", "10.0.1.0/24", "CIDR подсети", "string")
    web_module.add_variable("enable_monitoring", True, "Включить мониторинг", "bool")

    web_module.add_resource("aws_vpc", "main", {"cidr_block": "10.0.0.0/16"})
    web_module.add_resource("aws_subnet", "public", {"cidr_block": "var.subnet_cidr"})
    web_module.add_resource("aws_instance", "web", {
        "ami": "var.ami_id",
        "instance_type": "var.instance_type"
    })

    web_module.add_output("instance_id", "aws_instance.web.id", "ID инстанса")
    web_module.add_output("public_ip", "aws_instance.web.public_ip", "Публичный IP")

    print(f"  Модуль: {web_module.name}")
    print(f"  Описание: {web_module.description}")
    print(f"  Переменные:")
    for name, var in web_module.variables.items():
        print(f"    {name} ({var['type']}): {var['description']} [по умолчанию: {var['default']}]")
    print(f"  Ресурсы: {len(web_module.resources)}")
    print(f"  Outputs: {list(web_module.outputs.keys())}")

    # --- 3.2 Инстанцирование модуля ---
    print("\n--- 3.2 Инстанцирование модуля ---")

    # Создаём два экземпляра модуля с разными параметрами
    staging_vars = {
        "instance_type": "t3.micro",
        "ami_id": "ami-0c55b159cbfafe1f0",
        "subnet_cidr": "10.0.1.0/24",
        "enable_monitoring": False
    }

    production_vars = {
        "instance_type": "t3.large",
        "ami_id": "ami-0c55b159cbfafe1f0",
        "subnet_cidr": "10.0.2.0/24",
        "enable_monitoring": True
    }

    staging = web_module.instantiate(staging_vars)
    production = web_module.instantiate(production_vars)

    print("  Staging:")
    for res in staging["resources"]:
        print(f"    {res['type']}.{res['name']}: {json.dumps(res['config'])}")

    print("\n  Production:")
    for res in production["resources"]:
        print(f"    {res['type']}.{res['name']}: {json.dumps(res['config'])}")

    # --- 3.3 Композиция модулей ---
    print("\n--- 3.3 Композиция модулей ---")

    class ModuleComposition:
        """Композиция — использование нескольких модулей вместе"""
        def __init__(self, name):
            self.name = name
            self.modules = []

        def add_module(self, module, variables, depends_on=None):
            self.modules.append({
                "module": module,
                "variables": variables,
                "depends_on": depends_on or []
            })

        def plan(self):
            print(f"  Композиция '{self.name}':")
            for i, m in enumerate(self.modules, 1):
                mod = m["module"]
                deps = m["depends_on"]
                dep_str = f" → зависит от: {', '.join(deps)}" if deps else ""
                print(f"    {i}. {mod.name}{dep_str}")
                print(f"       Переменные: {json.dumps(m['variables'], ensure_ascii=False)}")

    composition = ModuleComposition("production_stack")
    composition.add_module(web_module, staging_vars)
    composition.add_module(web_module, production_vars, depends_on=["module.web_staging"])
    composition.plan()

    # --- 3.4 Переиспользование модулей ---
    print("\n--- 3.4 Переиспользование модулей ---")

    # Модуль для базы данных
    db_module = TerraformModule("database", "Модуль PostgreSQL БД")
    db_module.add_variable("engine_version", "15.3", "Версия PostgreSQL")
    db_module.add_variable("instance_class", "db.t3.micro", "Класс инстанса")
    db_module.add_variable("allocated_storage", 20, "Объём хранилища (GB)")

    db_module.add_resource("aws_db_instance", "main", {
        "engine": "postgresql",
        "engine_version": "var.engine_version",
        "instance_class": "var.instance_class",
        "allocated_storage": "var.allocated_storage"
    })

    db_module.add_output("endpoint", "aws_db_instance.main.endpoint", "Endpoint БД")

    # Использование обоих модулей
    all_modules = {"web": web_module, "database": db_module}
    print("  Доступные модули:")
    for name, mod in all_modules.items():
        print(f"    {name}: {mod.description}")
        print(f"      Ресурсы: {[r['type'] for r in mod.resources]}")
        print(f"      Outputs: {list(mod.outputs.keys())}")

    # Экономия кода
    original_lines = len(web_module.resources) * 15 + len(db_module.resources) * 15
    module_lines = len(web_module.resources) * 15 + len(db_module.resources) * 15
    print(f"\n  Без модулей: ~{original_lines} строк конфигурации")
    print(f"  С модулями: ~{module_lines} строк + 2 модуля переиспользуются")
    print(f"  Экономия: модули переиспользуются N раз вместо копирования")


# =============================================================================
# Демо 4: Состояние и планирование — plan/apply, обнаружение дрейфа
# =============================================================================

def demo_state_planning():
    print("\n\n" + "=" * 70)
    print("Демо 4: Состояние и планирование — plan/apply, обнаружение дрейфа")
    print("=" * 70)

    # --- 4.1 Файл состояния ---
    print("\n--- 4.1 Файл состояния (state file) ---")

    class TerraformState:
        def __init__(self):
            self.state = {
                "version": 4,
                "terraform_version": "1.5.0",
                "serial": 1,
                "lineage": str(uuid.uuid4()),
                "resources": [],
                "outputs": {}
            }

        def add_resource(self, module, resource_type, name, attributes, depends_on=None):
            """Добавление ресурса в состояние"""
            resource = {
                "module": module,
                "mode": "managed",
                "type": resource_type,
                "name": name,
                "provider": f"provider[\"registry.terraform.io/hashicorp/aws\"]",
                "instances": [{
                    "attributes": attributes,
                    "depends_on": depends_on or []
                }]
            }
            self.state["resources"].append(resource)
            self.state["serial"] += 1

        def to_json(self):
            return json.dumps(self.state, indent=2, ensure_ascii=False)

        def resource_count(self):
            return len(self.state["resources"])

    ts = TerraformState()

    # Добавляем ресурсы
    ts.add_resource("module.vpc", "aws_vpc", "main", {
        "id": "vpc-abc123",
        "cidr_block": "10.0.0.0/16",
        "enable_dns_hostnames": True,
        "tags": {"Name": "production-vpc"}
    })

    ts.add_resource("module.vpc", "aws_subnet", "public", {
        "id": "subnet-def456",
        "vpc_id": "vpc-abc123",
        "cidr_block": "10.0.1.0/24",
        "availability_zone": "eu-west-1a"
    })

    ts.add_resource("module.ec2", "aws_instance", "web", {
        "id": "i-0abc123def456",
        "ami": "ami-0c55b159cbfafe1f0",
        "instance_type": "t3.medium",
        "private_ip": "10.0.1.50",
        "public_ip": "52.14.23.100",
        "subnet_id": "subnet-def456"
    })

    print(f"  Состояние: {ts.resource_count()} ресурсов")
    print(f"  Serial: {ts.state['serial']}")
    print(f"  Terraform version: {ts.state['terraform_version']}")
    print(f"\n  Содержимое state (сокращённо):")
    for res in ts.state["resources"]:
        attrs = res["instances"][0]["attributes"]
        print(f"    {res['module']}.{res['type']}.{res['name']}:")
        for k, v in list(attrs.items())[:3]:
            print(f"      {k}: {v}")

    # --- 4.2 Цикл plan/apply ---
    print("\n--- 4.2 Цикл plan/apply ---")

    class PlanEngine:
        def __init__(self):
            self.current_state = {}
            self.planned_changes = []

        def load_state(self, state_dict):
            self.current_state = state_dict.copy()

        def plan(self, desired_config):
            """Генерация плана: сравнение desired с current state"""
            self.planned_changes = []

            # Новые ресурсы
            for name, config in desired_config.items():
                if name not in self.current_state:
                    self.planned_changes.append({
                        "action": "create",
                        "resource": name,
                        "config": config
                    })
                elif self.current_state[name] != config:
                    # Обновление существующего ресурса
                    changes = {}
                    for k, v in config.items():
                        if self.current_state[name].get(k) != v:
                            changes[k] = {"old": self.current_state[name].get(k), "new": v}
                    self.planned_changes.append({
                        "action": "update",
                        "resource": name,
                        "changes": changes
                    })

            # Удалённые ресурсы
            for name in self.current_state:
                if name not in desired_config:
                    self.planned_changes.append({
                        "action": "destroy",
                        "resource": name
                    })

            return self.planned_changes

        def apply(self):
            """Применение плана"""
            for change in self.planned_changes:
                if change["action"] == "create":
                    self.current_state[change["resource"]] = change["config"]
                    print(f"    + {change['resource']}: создан")
                elif change["action"] == "update":
                    for k, v in change["changes"].items():
                        self.current_state[change["resource"]][k] = v["new"]
                        print(f"    ~ {change['resource']}.{k}: {v['old']} → {v['new']}")
                elif change["action"] == "destroy":
                    del self.current_state[change["resource"]]
                    print(f"    - {change['resource']}: удалён")

    engine = PlanEngine()

    # Начальное состояние
    engine.load_state({
        "vpc": {"cidr": "10.0.0.0/16", "enable_dns": True},
        "subnet": {"cidr": "10.0.1.0/24"},
        "instance": {"type": "t3.micro", "ami": "ami-0c55b159cbfafe1f0"}
    })

    # Желаемая конфигурация
    desired = {
        "vpc": {"cidr": "10.0.0.0/16", "enable_dns": True},       # без изменений
        "subnet": {"cidr": "10.0.2.0/24"},                         # обновление
        "instance": {"type": "t3.large", "ami": "ami-0c55b159cbfafe1f0"},  # обновление
        "load_balancer": {"type": "application", "scheme": "internet-facing"}  # новый
    }

    print("  План изменений:")
    changes = engine.plan(desired)
    for change in changes:
        if change["action"] == "create":
            print(f"    + {change['resource']}: будет создан")
        elif change["action"] == "update":
            print(f"    ~ {change['resource']}: будет обновлён")
            for k, v in change["changes"].items():
                print(f"        {k}: {v['old']} → {v['new']}")
        elif change["action"] == "destroy":
            print(f"    - {change['resource']}: будет удалён")

    print(f"\n  Итого: +{sum(1 for c in changes if c['action']=='create')} "
          f"~{sum(1 for c in changes if c['action']=='update')} "
          f"-{sum(1 for c in changes if c['action']=='destroy')}")

    print("\n  Применение (apply):")
    engine.apply()

    # --- 4.3 Обнаружение дрейфа ---
    print("\n--- 4.3 Обнаружение дрейфа (drift detection) ---")

    class DriftDetector:
        """Обнаружение дрейфа — сравнение состояния в state file
        с реальным состоянием инфраструктуры"""
        def __init__(self):
            self.state_file = {}     # что записано в state
            self.actual_state = {}   # что реально в облаке

        def load_state(self, state):
            self.state_file = state.copy()

        def set_actual(self, actual):
            self.actual_state = actual.copy()

        def detect(self):
            """Обнаружение расхождений между state и реальностью"""
            drifts = []

            all_resources = set(self.state_file.keys()) | set(self.actual_state.keys())

            for resource in all_resources:
                state_val = self.state_file.get(resource)
                actual_val = self.actual_state.get(resource)

                if state_val is None:
                    drifts.append({"resource": resource, "type": "external", "detail": "создан вне Terraform"})
                elif actual_val is None:
                    drifts.append({"resource": resource, "type": "missing", "detail": "удалён вручную"})
                elif state_val != actual_val:
                    changes = {}
                    if isinstance(state_val, dict) and isinstance(actual_val, dict):
                        for k in set(state_val.keys()) | set(actual_val.keys()):
                            if state_val.get(k) != actual_val.get(k):
                                changes[k] = {"state": state_val.get(k), "actual": actual_val.get(k)}
                    drifts.append({"resource": resource, "type": "modified", "changes": changes})

            return drifts

    dd = DriftDetector()

    # State file (ожидаемое состояние)
    dd.load_state({
        "instance_type": "t3.micro",
        "ami": "ami-0c55b159cbfafe1f0",
        "subnet": "10.0.1.0/24",
        "tags": {"Environment": "production", "Team": "backend"}
    })

    # Реальное состояние (кто-то изменил вручную)
    dd.set_actual({
        "instance_type": "t3.large",           # изменено вручную!
        "ami": "ami-0c55b159cbfafe1f0",
        "subnet": "10.0.1.0/24",
        "tags": {"Environment": "production", "Team": "backend"},
        "security_group": "sg-new123"           # добавлено вне Terraform!
    })

    print("  Сравнение state file vs реальность:")
    drifts = dd.detect()
    if drifts:
        print(f"\n  Обнаружен дрейф ({len(drifts)} расхождений):")
        for d in drifts:
            if d["type"] == "modified":
                print(f"    ~ {d['resource']}: изменён")
                for k, v in d["changes"].items():
                    print(f"        {k}: {v['state']} → {v['actual']}")
            elif d["type"] == "external":
                print(f"    ! {d['resource']}: {d['detail']}")
            elif d["type"] == "missing":
                print(f"    ✗ {d['resource']}: {d['detail']}")
    else:
        print("    Дрейф не обнаружен")

    # --- 4.4 Workspace management ---
    print("\n--- 4.4 Управление workspace ---")

    class TerraformWorkspace:
        """Workspace — изолированные окружения (dev, staging, prod)
        с разными значениями переменных, но общей конфигурацией"""
        def __init__(self, name):
            self.name = name
            self.variables = {}
            self.state = TerraformState()

        def set_variable(self, key, value):
            self.variables[key] = value

        def get_variable(self, key):
            return self.variables.get(key)

        def summary(self):
            print(f"\n  Workspace: {self.name}")
            print(f"  Переменные:")
            for k, v in self.variables.items():
                print(f"    {k} = {v}")

    # Создаём workspace-ы для разных окружений
    workspaces = {}

    # Dev
    dev = TerraformWorkspace("dev")
    dev.set_variable("instance_type", "t3.micro")
    dev.set_variable("instance_count", 1)
    dev.set_variable("enable_monitoring", False)
    dev.set_variable("domain", "dev.example.com")
    workspaces["dev"] = dev

    # Staging
    staging = TerraformWorkspace("staging")
    staging.set_variable("instance_type", "t3.small")
    staging.set_variable("instance_count", 2)
    staging.set_variable("enable_monitoring", True)
    staging.set_variable("domain", "staging.example.com")
    workspaces["staging"] = staging

    # Production
    prod = TerraformWorkspace("production")
    prod.set_variable("instance_type", "t3.large")
    prod.set_variable("instance_count", 3)
    prod.set_variable("enable_monitoring", True)
    prod.set_variable("domain", "example.com")
    workspaces["production"] = prod

    print("  Workspace-ы:")
    for name, ws in workspaces.items():
        ws.summary()

    # Сравнение workspace-ов
    print("\n  Сравнение конфигураций:")
    keys = ["instance_type", "instance_count", "enable_monitoring", "domain"]
    header = f"    {'Параметр':<25} {'dev':<12} {'staging':<12} {'production':<12}"
    print(header)
    print("    " + "-" * 61)
    for key in keys:
        dev_val = dev.get_variable(key)
        stag_val = staging.get_variable(key)
        prod_val = prod.get_variable(key)
        print(f"    {key:<25} {str(dev_val):<12} {str(stag_val):<12} {str(prod_val):<12}")


# =============================================================================
# Запуск всех демонстраций
# =============================================================================

if __name__ == "__main__":
    demo_iac_principles()
    demo_resource_definitions()
    demo_modules()
    demo_state_planning()
