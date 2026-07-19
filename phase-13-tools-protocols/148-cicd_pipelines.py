"""148 — CI/CD Pipelines: GitHub Actions, testing, deployment automation

Темы:
  1. Концепции пайплайнов (триггер, задача, шаг, артефакт)
  2. Стратегии тестирования (单元, интеграционные, End-to-End, параллельные)
  3. Сборка и развёртывание (build артефакты, deploy в staging/production)
  4. Оптимизация пайплайнов (кэширование, матричные сборки, условные шаги)

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
# Демо 1: Концепции пайплайнов — триггер, задача, шаг, артефакт
# =============================================================================

def demo_pipeline_concepts():
    print("=" * 70)
    print("Демо 1: Концепции пайплайнов — триггер, задача, шаг, артефакт")
    print("=" * 70)

    # --- 1.1 Моделирование пайплайна ---
    print("\n--- 1.1 Моделирование пайплайна CI/CD ---")

    class PipelineStep:
        def __init__(self, name, command, timeout=60, continue_on_error=False):
            self.name = name
            self.command = command
            self.timeout = timeout
            self.continue_on_error = continue_on_error
            self.status = "pending"
            self.duration = 0

        def execute(self):
            """Имитация выполнения шага"""
            self.status = "running"
            start = time.time()
            # Имитируем разное время выполнения
            self.duration = random.uniform(0.1, 2.0)
            time.sleep(0.01)  # минимальная задержка для реалистичности
            # Имитируем: шаги с 'fail' в имени падают
            if "fail" in self.name.lower():
                self.status = "failed"
                return False
            self.status = "success"
            return True

    class PipelineJob:
        def __init__(self, name, runs_on="ubuntu-latest"):
            self.name = name
            self.runs_on = runs_on
            self.steps = []
            self.status = "pending"

        def add_step(self, name, command, **kwargs):
            step = PipelineStep(name, command, **kwargs)
            self.steps.append(step)
            return self

        def execute(self):
            self.status = "running"
            for step in self.steps:
                print(f"    [{self.name}] Шаг: {step.name}")
                success = step.execute()
                if not success and not step.continue_on_error:
                    self.status = "failed"
                    print(f"    [{self.name}] ✗ Шаг '{step.name}' провалился")
                    return False
            self.status = "success"
            print(f"    [{self.name}] ✓ Все шаги выполнены успешно")
            return True

    class Pipeline:
        def __init__(self, name, trigger_type="push"):
            self.name = name
            self.trigger_type = trigger_type
            self.jobs = []
            self.status = "pending"
            self.artifacts = []

        def add_job(self, job):
            self.jobs.append(job)
            return self

        def add_artifact(self, name, path):
            self.artifacts.append({"name": name, "path": path})

        def run(self, event=None):
            """Запуск пайплайна по триггеру"""
            print(f"\n  ▶ Пайплай '{self.name}' запущен триггером: {event or self.trigger_type}")
            self.status = "running"
            all_success = True
            for job in self.jobs:
                print(f"\n  Задача '{job.name}':")
                success = job.execute()
                if not success:
                    all_success = False
                    break
            self.status = "success" if all_success else "failed"
            return self.status

    # Создаём пайплайн
    pipeline = Pipeline("CI/CD Pipeline", trigger_type="push to main")

    # Задача тестирования
    test_job = PipelineJob("test", runs_on="ubuntu-latest")
    test_job.add_step("Checkout", "git checkout .")
    test_job.add_step("Setup Python", "python -m pip install -r requirements.txt")
    test_job.add_step("Run tests", "pytest tests/ -v")
    pipeline.add_job(test_job)

    # Задача сборки
    build_job = PipelineJob("build", runs_on="ubuntu-latest")
    build_job.add_step("Build", "python -m build")
    build_job.add_step("Upload artifact", "upload-artifact dist/")
    pipeline.add_job(build_job)

    # Артефакты
    pipeline.add_artifact("dist", "./dist/")
    pipeline.add_artifact("test-report", "./test-results/")

    # Запуск пайплайна
    result = pipeline.run("push")
    print(f"\n  Результат пайплайна: {result}")
    print(f"  Артефакты: {[a['name'] for a in pipeline.artifacts]}")

    # --- 1.2 Триггеры ---
    print("\n--- 1.2 Типы триггеров ---")

    triggers = {
        "push":    {"описание": "Когда код отправлен в репозиторий",
                    "пример": "git push origin main"},
        "pull_request": {"описание": "Когда создан или обновлён PR",
                         "пример": "PR opened: feature → main"},
        "schedule": {"описание": "По расписанию (cron)",
                     "пример": "0 2 * * 1 (каждый понедельник в 2:00)"},
        "workflow_dispatch": {"описание": "Ручной запуск",
                              "пример": "Кнопка 'Run workflow' в GitHub"},
        "release":  {"описание": "При публикации релиза",
                     "пример": "GitHub Release v1.0.0"},
    }

    print("  Типы триггеров:")
    for tname, tinfo in triggers.items():
        print(f"\n    {tname}:")
        print(f"      Описание: {tinfo['описание']}")
        print(f"      Пример: {tinfo['пример']}")

    # --- 1.3 Артефакты ---
    print("\n--- 1.3 Артефакты ---")

    class ArtifactManager:
        def __init__(self):
            self.artifacts = {}

        def upload(self, name, content, metadata=None):
            artifact_id = str(uuid.uuid4())[:8]
            self.artifacts[name] = {
                "id": artifact_id,
                "content": content,
                "metadata": metadata or {},
                "uploaded_at": time.time(),
                "size_bytes": len(json.dumps(content))
            }
            return artifact_id

        def download(self, name):
            return self.artifacts.get(name)

        def list_all(self):
            return {k: {"id": v["id"], "size": v["size_bytes"]} for k, v in self.artifacts.items()}

    am = ArtifactManager()
    am.upload("app-binary", {"type": "binary", "path": "./dist/app"}, {"os": "linux", "arch": "x64"})
    am.upload("test-report", {"passed": 142, "failed": 3}, {"format": "json"})
    am.upload("coverage", {"percentage": 87.5}, {"format": "html"})

    print("  Артефакты:")
    for name, info in am.list_all().items():
        print(f"    {name}: id={info['id']}, размер={info['size']} байт")

    # --- 1.4 Матричная конфигурация ---
    print("\n--- 1.4 Матричная конфигурация ---")

    # Матричная сборка — запуск пайплайна на разных ОС/версиях Python
    matrix = {
        "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "python": ["3.10", "3.11", "3.12"]
    }

    # Генерация всех комбинаций
    combinations = []
    for os_val in matrix["os"]:
        for py_ver in matrix["python"]:
            combinations.append({"os": os_val, "python": py_ver})

    print(f"  Матрица: {len(matrix['os'])} ОС × {len(matrix['python'])} версий Python")
    print(f"  Всего комбинаций: {len(combinations)}\n")

    for combo in combinations:
        status = "✓" if random.random() > 0.1 else "✗"
        print(f"    {status} {combo['os']:<20} Python {combo['python']}")


# =============================================================================
# Демо 2: Стратегии тестирования
# =============================================================================

def demo_testing_strategies():
    print("\n\n" + "=" * 70)
    print("Демо 2: Стратегии тестирования")
    print("=" * 70)

    # --- 2.1 Unit-тесты ---
    print("\n--- 2.1 Unit-тесты ---")

    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    def divide(a, b):
        if b == 0:
            raise ValueError("Деление на ноль")
        return a / b

    class TestSuite:
        def __init__(self, name):
            self.name = name
            self.tests = []
            self.results = []

        def test(self, description):
            """Декоратор для регистрации теста"""
            def decorator(func):
                self.tests.append({"name": description, "func": func})
                return func
            return decorator

        def run(self):
            passed = 0
            failed = 0
            for t in self.tests:
                try:
                    t["func"]()
                    self.results.append({"name": t["name"], "status": "PASSED"})
                    passed += 1
                except AssertionError as e:
                    self.results.append({"name": t["name"], "status": "FAILED", "error": str(e)})
                    failed += 1
                except Exception as e:
                    self.results.append({"name": t["name"], "status": "ERROR", "error": str(e)})
                    failed += 1
            return passed, failed

    suite = TestSuite("Математические функции")

    @suite.test("add: положительные числа")
    def _():
        assert add(2, 3) == 5

    @suite.test("add: отрицательные числа")
    def _():
        assert add(-1, -1) == -2

    @suite.test("multiply: базовый случай")
    def _():
        assert multiply(3, 4) == 12

    @suite.test("divide: базовый случай")
    def _():
        assert abs(divide(10, 3) - 3.3333) < 0.01

    @suite.test("divide: деление на ноль")
    def _():
        try:
            divide(1, 0)
            assert False, "Должно быть исключение"
        except ValueError:
            pass  # ожидаемое поведение

    passed, failed = suite.run()
    print(f"  Тесты: {passed} пройдено, {failed} провалено\n")
    for r in suite.results:
        status = "✓" if r["status"] == "PASSED" else "✗"
        print(f"    {status} {r['name']}")

    # --- 2.2 Интеграционные тесты ---
    print("\n--- 2.2 Интеграционные тесты ---")

    class Database:
        def __init__(self):
            self.data = {}
            self.transaction_log = []

        def begin_transaction(self):
            self.transaction_log.append("BEGIN")

        def commit(self):
            self.transaction_log.append("COMMIT")

        def rollback(self):
            self.transaction_log.append("ROLLBACK")
            # Откатываем изменения
            self.data = {k: v for k, v in self.data.items() if "_rollback" not in str(v)}

        def insert(self, key, value):
            self.data[key] = value

        def query(self, key):
            return self.data.get(key)

    class Cache:
        def __init__(self, db):
            self.db = db
            self.cache = {}
            self.hits = 0
            self.misses = 0

        def get(self, key):
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            value = self.db.query(key)
            if value is not None:
                self.cache[key] = value
            return value

        def set(self, key, value):
            self.cache[key] = value
            self.db.insert(key, value)

        def invalidate(self, key):
            self.cache.pop(key, None)

        def hit_rate(self):
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0

    # Интеграционный тест: БД + кэш
    db = Database()
    cache = Cache(db)

    print("  Тестирование интеграции БД + кэш:")
    cache.set("user:1", {"name": "Alice", "role": "admin"})
    cache.set("user:2", {"name": "Bob", "role": "user"})

    # Первый запрос — промах кэша, чтение из БД
    result = cache.get("user:1")
    print(f"    Запрос user:1: {result} (кэш: miss)")

    # Второй запрос — попадание в кэш
    result = cache.get("user:1")
    print(f"    Запрос user:1: {result} (кэш: hit)")

    # Обновление через кэш
    cache.set("user:1", {"name": "Alice", "role": "superadmin"})
    cache.invalidate("user:1")

    result = cache.get("user:1")
    print(f"    После обновления: {result}")
    print(f"    Hit rate: {cache.hit_rate():.2%}")

    # --- 2.3 End-to-End тесты ---
    print("\n--- 2.3 End-to-End тесты ---")

    class E2ETestRunner:
        def __init__(self):
            self.scenarios = []
            self.results = []

        def scenario(self, name):
            def decorator(func):
                self.scenarios.append({"name": name, "func": func})
                return func
            return decorator

        def run_all(self):
            for scenario in self.scenarios:
                start = time.time()
                try:
                    scenario["func"]()
                    duration = time.time() - start
                    self.results.append({
                        "name": scenario["name"],
                        "status": "PASSED",
                        "duration": duration
                    })
                except Exception as e:
                    duration = time.time() - start
                    self.results.append({
                        "name": scenario["name"],
                        "status": "FAILED",
                        "duration": duration,
                        "error": str(e)
                    })

    e2e = E2ETestRunner()

    @e2e.scenario("Пользователь авторизуется и создаёт заказ")
    def _():
        # Имитация полного потока
        time.sleep(0.001)
        user = {"id": 1, "token": "abc123"}
        assert user["token"] is not None
        order = {"user_id": user["id"], "items": ["laptop"], "total": 1500}
        assert order["total"] > 0

    @e2e.scenario("Пользователь оплачивает заказ")
    def _():
        time.sleep(0.001)
        payment = {"status": "success", "amount": 1500}
        assert payment["status"] == "success"

    @e2e.scenario("Пользователь отменяет заказ")
    def _():
        time.sleep(0.001)
        cancellation = {"status": "cancelled", "refund": 1500}
        assert cancellation["refund"] > 0

    e2e.run_all()
    print("  Результаты E2E тестов:")
    for r in e2e.results:
        status = "✓" if r["status"] == "PASSED" else "✗"
        print(f"    {status} {r['name']} ({r['duration']*1000:.1f}ms)")

    # --- 2.4 Параллельное тестирование ---
    print("\n--- 2.4 Параллельное тестирование ---")

    def simulate_parallel_test(test_id, duration_range=(0.01, 0.05)):
        """Имитация одного теста"""
        duration = random.uniform(*duration_range)
        time.sleep(0.001)
        passed = random.random() > 0.05  # 95% проходят
        return {"id": test_id, "duration": duration, "status": "PASSED" if passed else "FAILED"}

    # Последовательный запуск
    random.seed(42)
    tests = [f"test_{i}" for i in range(10)]

    start = time.time()
    sequential_results = [simulate_parallel_test(t) for t in tests]
    sequential_time = time.time() - start

    # Параллельный запуск (симуляция — в реальности multiprocessing)
    start = time.time()
    # Имитируем: параллельный запуск занимает время самого долгого теста
    parallel_durations = [random.uniform(0.01, 0.05) for _ in tests]
    parallel_time = max(parallel_durations)
    parallel_results = [{"id": t, "status": "PASSED"} for t in tests]

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"  10 тестов:")
    print(f"    Последовательно: {sequential_time*1000:.1f}ms")
    print(f"    Параллельно (max): {parallel_time*1000:.1f}ms")
    print(f"    Ускорение: {speedup:.1f}x")


# =============================================================================
# Демо 3: Сборка и развёртывание
# =============================================================================

def demo_build_deploy():
    print("\n\n" + "=" * 70)
    print("Демо 3: Сборка и развёртывание")
    print("=" * 70)

    # --- 3.1 Build артефакты ---
    print("\n--- 3.1 Build артефакты ---")

    class BuildPipeline:
        def __init__(self):
            self.steps = []
            self.artifacts = []
            self.build_id = str(uuid.uuid4())[:8]

        def add_step(self, name, func):
            self.steps.append({"name": name, "func": func})

        def build(self):
            print(f"  Build ID: {self.build_id}")
            for step in self.steps:
                print(f"    → {step['name']}...", end=" ")
                result = step["func"]()
                if result:
                    self.artifacts.append(result)
                    print(f"✓ ({result['name']})")
                else:
                    print("✓")
            return self.artifacts

    def compile_step():
        time.sleep(0.001)
        return {"name": "app.bin", "size": 1024 * 512, "type": "binary"}

    def package_step():
        time.sleep(0.001)
        return {"name": "app.tar.gz", "size": 1024 * 256, "type": "archive"}

    def docker_build():
        time.sleep(0.001)
        return {"name": "app:latest", "size": 1024 * 1024 * 100, "type": "docker_image"}

    build = BuildPipeline()
    build.add_step("Compile", compile_step)
    build.add_step("Package", package_step)
    build.add_step("Docker Build", docker_build)

    artifacts = build.build()
    print(f"\n  Артефакты сборки: {len(artifacts)}")
    for a in artifacts:
        size_mb = a['size'] / (1024 * 1024) if a['size'] > 1024*1024 else a['size'] / 1024
        unit = "MB" if a['size'] > 1024*1024 else "KB"
        print(f"    {a['name']}: {size_mb:.1f} {unit} ({a['type']})")

    # --- 3.2 Развертывание в staging ---
    print("\n--- 3.2 Развертывание в staging ---")

    class Deployment:
        def __init__(self, environment):
            self.environment = environment
            self.status = "pending"
            self.instances = 0
            self.health_checks = []

        def deploy(self, artifact, num_instances=2):
            print(f"  Развертывание в {self.environment}:")
            print(f"    Артефакт: {artifact['name']}")

            # Создание инстансов
            self.instances = num_instances
            for i in range(num_instances):
                print(f"    Инстанс {i+1}: запуск...", end=" ")
                time.sleep(0.001)
                health = random.random() > 0.1  # 90% здоровы
                self.health_checks.append({"instance": i+1, "healthy": health})
                status = "✓ healthy" if health else "✗ unhealthy"
                print(status)

            # Проверка общего здоровья
            healthy = sum(1 for h in self.health_checks if h["healthy"])
            if healthy == self.instances:
                self.status = "deployed"
                print(f"    → Все {self.instances} инстансов здоровы. Статус: DEPLOYED")
            else:
                self.status = "failed"
                print(f"    → {healthy}/{self.instances} здоровы. Статус: FAILED")
            return self.status

    staging = Deployment("staging")
    result = staging.deploy(artifacts[2], num_instances=3)

    # --- 3.3 Blue-Green развертывание ---
    print("\n--- 3.3 Blue-Green развертывание ---")

    class BlueGreenDeployment:
        """Blue-Green: два идентичных окружения. Один активен (Blue),
        другой готов к переключению (Green). При деплое обновляем
        неактивное окружение, затем переключаем трафик."""
        def __init__(self):
            self.blue = {"status": "active", "version": "1.0.0", "traffic": 100}
            self.green = {"status": "idle", "version": "1.0.0", "traffic": 0}

        def deploy_green(self, new_version):
            """Деплой новой версии в Green"""
            self.green["version"] = new_version
            self.green["status"] = "ready"
            print(f"    Green обновлён до версии {new_version}")

        def switch_traffic(self):
            """Переключение трафика Blue → Green"""
            self.blue["traffic"] = 0
            self.green["traffic"] = 100
            self.blue["status"] = "idle"
            self.green["status"] = "active"
            print(f"    Трафик переключён: Blue={self.blue['traffic']}%, Green={self.green['traffic']}%")

        def rollback(self):
            """Откат: переключение обратно на Blue"""
            self.blue["traffic"] = 100
            self.green["traffic"] = 0
            self.blue["status"] = "active"
            self.green["status"] = "idle"
            print(f"    ОТКАТ: трафик вернулся на Blue")

    bg = BlueGreenDeployment()
    print(f"  Начальное состояние:")
    print(f"    Blue: {bg.blue}")
    print(f"    Green: {bg.green}")

    print(f"\n  Деплой v2.0.0 в Green:")
    bg.deploy_green("2.0.0")
    bg.switch_traffic()

    print(f"\n  После переключения:")
    print(f"    Blue: {bg.blue}")
    print(f"    Green: {bg.green}")

    print(f"\n  Обнаружена проблема — откат:")
    bg.rollback()
    print(f"    Blue: {bg.blue}")
    print(f"    Green: {bg.green}")

    # --- 3.4 Canary развертывание ---
    print("\n--- 3.4 Canary развертывание ---")

    class CanaryDeployment:
        """Canary: постепенное увеличение доли трафика на новую версию.
        Если метрики в норме — увеличиваем; если нет — откат."""
        def __init__(self):
            self.current_version = "1.0.0"
            self.canary_version = "1.1.0"
            self.canary_percentage = 0
            self.metrics = {"errors": 0, "requests": 0}

        def step_increase(self, increment=5):
            """Увеличение доли canary"""
            self.canary_percentage = min(100, self.canary_percentage + increment)
            # Имитация метрик
            new_requests = random.randint(10, 50)
            new_errors = random.randint(0, 2) if self.canary_percentage > 50 else 0
            self.metrics["requests"] += new_requests
            self.metrics["errors"] += new_errors
            error_rate = self.metrics["errors"] / self.metrics["requests"] * 100

            print(f"    Canary: {self.canary_percentage}% | "
                  f"Запросы: {self.metrics['requests']} | "
                  f"Ошибки: {error_rate:.1f}%")

            if error_rate > 5:
                print(f"    → Высокий уровень ошибок! Откат!")
                self.canary_percentage = 0
                return False
            return True

    canary = CanaryDeployment()
    print("  Поэтапное увеличение canary:")
    for step in range(5):
        print(f"\n  Шаг {step+1}:")
        ok = canary.step_increase(10)
        if not ok:
            break
    print(f"\n  Финальный статус: canary={canary.canary_percentage}%, "
          f"версия={canary.canary_version if canary.canary_percentage > 0 else canary.current_version}")


# =============================================================================
# Демо 4: Оптимизация пайплайнов
# =============================================================================

def demo_pipeline_optimization():
    print("\n\n" + "=" * 70)
    print("Демо 4: Оптимизация пайплайнов")
    print("=" * 70)

    # --- 4.1 Кэширование зависимостей ---
    print("\n--- 4.1 Кэширование зависимостей ---")

    class DependencyCache:
        def __init__(self):
            self.cache = {}
            self.hits = 0
            self.misses = 0

        def get_cache_key(self, requirements_hash):
            """Ключ кэша — хеш файла зависимостей"""
            return f"deps_{requirements_hash}"

        def check(self, requirements_hash):
            """Проверка наличия кэша"""
            key = self.get_cache_key(requirements_hash)
            if key in self.cache:
                self.hits += 1
                return True, self.cache[key]
            self.misses += 1
            return False, None

        def store(self, requirements_hash, install_time):
            """Сохранение в кэш"""
            key = self.get_cache_key(requirements_hash)
            self.cache[key] = {"install_time": install_time, "cached_at": time.time()}

    cache = DependencyCache()

    # Имитация: requirements.txt с фиксированным хешем
    reqs = "numpy==1.24.0\npandas==2.0.0\nscikit-learn==1.3.0"
    req_hash = hashlib.md5(reqs.encode()).hexdigest()[:12]

    print(f"  Хеш requirements: {req_hash}")

    # Первый запуск — кэша нет, установка ~3 сек
    cached, data = cache.check(req_hash)
    if not cached:
        install_time = random.uniform(2.5, 3.5)
        cache.store(req_hash, install_time)
        print(f"  Запуск 1: MISS → установка зависимостей ({install_time:.1f}с)")
    else:
        print(f"  Запуск 1: HIT → восстановление из кэша ({data['install_time']:.1f}с)")

    # Второй запуск — кэш есть
    cached, data = cache.check(req_hash)
    if cached:
        print(f"  Запуск 2: HIT → восстановление из кэша ({data['install_time'] * 0.1:.1f}с)")
    else:
        install_time = random.uniform(2.5, 3.5)
        cache.store(req_hash, install_time)
        print(f"  Запуск 2: MISS → установка зависимостей ({install_time:.1f}с)")

    print(f"\n  Статистика кэша: hits={cache.hits}, misses={cache.misses}")

    # --- 4.2 Матричные сборки ---
    print("\n--- 4.2 Матричные сборки ---")

    class MatrixBuild:
        def __init__(self):
            self.matrix = {}
            self.results = []

        def set_matrix(self, **kwargs):
            self.matrix = kwargs

        def generate_combinations(self):
            keys = list(self.matrix.keys())
            values = list(self.matrix.values())
            combos = [{}]
            for key, vals in zip(keys, values):
                new_combos = []
                for combo in combos:
                    for v in vals:
                        new_combo = {**combo, key: v}
                        new_combos.append(new_combo)
                combos = new_combos
            return combos

        def run(self):
            combos = self.generate_combinations()
            print(f"  Матрица: {', '.join(f'{k}={v}' for k, v in self.matrix.items())}")
            print(f"  Всего комбинаций: {len(combos)}\n")

            for i, combo in enumerate(combos):
                # Имитация: разное время сборки на разных ОС
                build_time = random.uniform(0.5, 3.0)
                if combo.get("os") == "windows-latest":
                    build_time *= 1.3  # Windows обычно медленнее
                passed = random.random() > 0.05
                self.results.append({**combo, "time": build_time, "status": "✓" if passed else "✗"})

            # Вывод результатов
            for r in self.results:
                print(f"    {r['status']} {r.get('os','?'):<20} Python {r.get('python','?')}: {r['time']:.1f}с")

    mb = MatrixBuild()
    mb.set_matrix(os=["ubuntu-latest", "windows-latest"], python=["3.10", "3.12"])
    mb.run()

    # --- 4.3 Условные шаги ---
    print("\n--- 4.3 Условные шаги ---")

    class ConditionalStep:
        def __init__(self, name, condition, action):
            self.name = name
            self.condition = condition
            self.action = action
            self.executed = False

        def try_execute(self, context):
            if self.condition(context):
                result = self.action(context)
                self.executed = True
                return result
            return None

    # Определяем условия
    def is_main_branch(ctx):
        return ctx.get("branch") == "main"

    def is_pr(ctx):
        return ctx.get("event") == "pull_request"

    def has_changes(ctx):
        return ctx.get("changed_files", 0) > 0

    steps = [
        ConditionalStep("Run tests", lambda ctx: True, lambda ctx: "Тесты выполнены"),
        ConditionalStep("Deploy to staging", is_main_branch, lambda ctx: "Deployed to staging"),
        ConditionalStep("Run integration tests", is_pr, lambda ctx: "Интеграционные тесты пройдены"),
        ConditionalStep("Build docs", has_changes, lambda ctx: "Документация обновлена"),
        ConditionalStep("Deploy to production", is_main_branch, lambda ctx: "Deployed to production"),
    ]

    contexts = [
        {"branch": "main", "event": "push", "changed_files": 5},
        {"branch": "feature/auth", "event": "pull_request", "changed_files": 3},
        {"branch": "develop", "event": "push", "changed_files": 0},
    ]

    for ctx in contexts:
        print(f"\n  Контекст: branch={ctx['branch']}, event={ctx['event']}, changed={ctx.get('changed_files', 0)}")
        for step in steps:
            result = step.try_execute(ctx)
            status = f"→ {result}" if result else "→ пропущен"
            print(f"    {step.name}: {status}")

    # --- 4.4 Оптимизация через параллелизм ---
    print("\n--- 4.4 Оптимизация через параллелизм ---")

    def simulate_step(name, duration):
        """Имитация шага пайплайна"""
        time.sleep(0.001)
        return {"name": name, "duration": duration}

    # Последовательный пайплайн
    sequential_steps = [
        ("Install deps", 2.0),
        ("Lint", 1.0),
        ("Unit tests", 3.0),
        ("Integration tests", 4.0),
        ("Build", 2.0),
        ("Deploy", 1.5),
    ]

    total_sequential = sum(d for _, d in sequential_steps)
    print(f"  Последовательный: {total_sequential:.1f}с")
    for name, dur in sequential_steps:
        bar = "█" * int(dur * 4)
        print(f"    {name:<25} {dur:.1f}с |{bar}|")

    # Оптимизированный (параллельные группы)
    print(f"\n  Оптимизированный (параллельные группы):")
    parallel_groups = [
        ("Install deps", 2.0),
        [("Lint", 1.0), ("Unit tests", 3.0)],  # параллельно
        ("Build", 2.0),
        [("Integration tests", 4.0), ("Deploy", 1.5)],  # параллельно
    ]

    total_optimized = 0
    for group in parallel_groups:
        if isinstance(group, list):
            max_dur = max(d for _, d in group)
            total_optimized += max_dur
            names = [f"{n} ({d}с)" for n, d in group]
            print(f"    Параллельно: {', '.join(names)} → {max_dur:.1f}с")
        else:
            name, dur = group
            total_optimized += dur
            print(f"    {name}: {dur:.1f}с")

    savings = (1 - total_optimized / total_sequential) * 10
    print(f"\n  Итого: {total_sequential:.1f}с → {total_optimized:.1f}с "
          f"(экономия {total_sequential - total_optimized:.1f}с, {savings:.0f}%)")


# =============================================================================
# Запуск всех демонстраций
# =============================================================================

if __name__ == "__main__":
    demo_pipeline_concepts()
    demo_testing_strategies()
    demo_build_deploy()
    demo_pipeline_optimization()
