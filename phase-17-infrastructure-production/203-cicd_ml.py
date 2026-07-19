"""
203 — CI/CD for ML: ML-пайплайны, версионирование моделей, трекинг экспериментов

Темы:
  1. ML Pipelines (data validation -> training -> evaluation -> deployment)
  2. Model Versioning (semantic versioning, model registry, lineage)
  3. Experiment Tracking (metrics logging, artifact storage, comparison)
  4. Automated Testing (data tests, model tests, integration tests)

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
# DEMO 1: ML Pipelines (data validation -> training -> evaluation -> deployment)
# ─────────────────────────────────────────────────────────────────────────────
def demo_ml_pipelines():
    """Демонстрация ML-пайплайнов."""
    print("=" * 70)
    print("DEMO 1: ML Pipelines (Data Validation -> Training -> Evaluation -> Deployment)")
    print("=" * 70)

    # 1.1 Определение пайплайна
    print("\n1.1 Определение ML-пайплайна:")
    print("   ML Pipeline = последовательность этапов с автоматизацией\n")

    class PipelineStep:
        """Один шаг пайплайна."""
        def __init__(self, name, function, dependencies=None):
            self.name = name
            self.function = function
            self.dependencies = dependencies or []
            self.status = "pending"
            self.output = None
            self.duration_ms = 0

        def run(self, context):
            """Выполнить шаг."""
            start = time.time()
            self.status = "running"
            try:
                self.output = self.function(context)
                self.status = "completed"
            except Exception as e:
                self.status = "failed"
                self.output = str(e)
            self.duration_ms = round((time.time() - start) * 1000, 2)
            return self.output

    class MLPipeline:
        """ML-пайплайн."""
        def __init__(self, name):
            self.name = name
            self.steps = []
            self.context = {}

        def add_step(self, step):
            """Добавить шаг в пайплайн."""
            self.steps.append(step)
            return self

        def run(self):
            """Выполнить все шаги по порядку."""
            print(f"   Запуск пайплайна '{self.name}':")
            for step in self.steps:
                print(f"     [{step.name}] Запуск...")
                step.run(self.context)
                if step.output:
                    self.context.update(step.output)
                status_icon = "OK" if step.status == "completed" else "FAIL"
                print(f"     [{step.name}] {status_icon} ({step.duration_ms}ms)")
            return self.context

    # 1.2 Определение шагов пайплайна
    def data_validation(context):
        """Валидация данных."""
        # Генерация синтетических данных
        n_samples = 1000
        data = {"features": [], "labels": []}
        for _ in range(n_samples):
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 1 if x1 + x2 > 0 else 0
            data["features"].append([x1, x2])
            data["labels"].append(label)

        # Валидация
        null_count = sum(1 for row in data["features"] if any(v is None for v in row))
        label_dist = collections.Counter(data["labels"])

        report = {
            "n_samples": n_samples,
            "n_features": 2,
            "null_values": null_count,
            "label_distribution": dict(label_dist),
            "data": data,
        }
        print(f"       Samples: {n_samples}, Features: 2")
        print(f"       Null values: {null_count}")
        print(f"       Label distribution: {dict(label_dist)}")
        return report

    def data_preprocessing(context):
        """Предобработка данных."""
        data = context["data"]
        # Простая нормализация
        features = data["features"]
        means = [0, 0]
        stds = [1, 1]
        for row in features:
            means[0] += row[0]
            means[1] += row[1]
        means = [m / len(features) for m in means]
        for row in features:
            stds[0] += (row[0] - means[0]) ** 2
            stds[1] += (row[1] - means[1]) ** 2
        stds = [math.sqrt(s / len(features)) for s in stds]

        normalized = [[(row[0] - means[0]) / stds[0], (row[1] - means[1]) / stds[1]]
                      for row in features]

        print(f"       Normalized {len(normalized)} samples")
        print(f"       Feature means: [{means[0]:.4f}, {means[1]:.4f}]")
        print(f"       Feature stds: [{stds[0]:.4f}, {stds[1]:.4f}]")
        return {"normalized_data": normalized, "means": means, "stds": stds}

    def training(context):
        """Обучение модели (упрощённая линейная модель)."""
        features = context["normalized_data"]
        labels = context["data"]["labels"]
        n_features = len(features[0])

        # Инициализация весов
        weights = [random.gauss(0, 0.1) for _ in range(n_features)]
        bias = 0.0
        lr = 0.01
        epochs = 50
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for i in range(len(features)):
                # Forward pass
                z = sum(w * x for w, x in zip(weights, features[i])) + bias
                pred = 1 / (1 + math.exp(-max(-500, min(500, z))))  # sigmoid
                error = labels[i] - pred

                # Backward pass
                for j in range(n_features):
                    weights[j] += lr * error * features[i][j]
                bias += lr * error

                # Loss
                total_loss += -labels[i] * math.log(pred + 1e-10) - (1 - labels[i]) * math.log(1 - pred + 1e-10)
                if (pred > 0.5) == labels[i]:
                    correct += 1

            avg_loss = total_loss / len(features)
            accuracy = correct / len(features)
            losses.append(avg_loss)
            if epoch % 10 == 0:
                print(f"       Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        print(f"       Final: loss={losses[-1]:.4f}, accuracy={correct/len(features):.4f}")
        return {"weights": weights, "bias": bias, "losses": losses}

    def evaluation(context):
        """Оценка модели."""
        weights = context["weights"]
        bias = context["bias"]
        features = context["normalized_data"]
        labels = context["data"]["labels"]

        predictions = []
        for i in range(len(features)):
            z = sum(w * x for w, x in zip(weights, features[i])) + bias
            pred = 1 / (1 + math.exp(-max(-500, min(500, z))))
            predictions.append(1 if pred > 0.5 else 0)

        # Метрики
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

        accuracy = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"       Accuracy:  {accuracy:.4f}")
        print(f"       Precision: {precision:.4f}")
        print(f"       Recall:    {recall:.4f}")
        print(f"       F1 Score:  {f1:.4f}")
        print(f"       Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def deployment(context):
        """Деплой модели."""
        # Сохранение модели как JSON
        model_artifact = {
            "model_type": "logistic_regression",
            "weights": context["weights"],
            "bias": context["bias"],
            "metrics": {
                "accuracy": context["accuracy"],
                "f1": context["f1"],
            },
            "version": "1.0.0",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Хэш артефакта для целостности
        artifact_hash = hashlib.sha256(json.dumps(model_artifact, sort_keys=True).encode()).hexdigest()
        print(f"       Model artifact saved: {len(json.dumps(model_artifact))} bytes")
        print(f"       Artifact hash: {artifact_hash[:16]}...")
        print(f"       Endpoint: POST /v1/models/sentiment/predict")
        return {"artifact_hash": artifact_hash, "model_artifact": model_artifact}

    # 1.3 Запуск пайплайна
    print("\n1.3 Запуск полного пайплайна:")
    pipeline = MLPipeline("sentiment-training-v1")
    pipeline.add_step(PipelineStep("data_validation", data_validation))
    pipeline.add_step(PipelineStep("preprocessing", data_preprocessing))
    pipeline.add_step(PipelineStep("training", training))
    pipeline.add_step(PipelineStep("evaluation", evaluation))
    pipeline.add_step(PipelineStep("deployment", deployment))

    result = pipeline.run()

    # 1.4 DAG (Directed Acyclic Graph) пайплайна
    print("\n1.4 DAG (Directed Acyclic Graph) пайплайна:")
    dag = {
        "data_validation": [],
        "preprocessing": ["data_validation"],
        "feature_engineering": ["preprocessing"],
        "training": ["feature_engineering"],
        "evaluation": ["training"],
        "deployment": ["evaluation"],
    }
    print("   Структура пайплайна:")
    for step, deps in dag.items():
        dep_str = f" <- {deps}" if deps else " (начальный)"
        print(f"     {step}{dep_str}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2: Model Versioning (semantic versioning, model registry, lineage)
# ─────────────────────────────────────────────────────────────────────────────
def demo_model_versioning():
    """Демонстрация версионирования моделей."""
    print("\n" + "=" * 70)
    print("DEMO 2: Model Versioning (Semantic Versioning, Registry, Lineage)")
    print("=" * 70)

    # 2.1 Semantic Versioning для ML моделей
    print("\n2.1 Semantic Versioning для ML моделей:")
    print("   Формат: MAJOR.MINOR.PATCH (например, 2.1.0)\n")
    print("   MAJOR: изменение формата данных/архитектуры (breaking)")
    print("   MINOR: новый функционал, улучшение метрик")
    print("   PATCH: исправление багов, переобучение\n")

    class SemanticVersion:
        """Управление семантическим версионированием."""
        def __init__(self, major=0, minor=0, patch=0):
            self.major = major
            self.minor = minor
            self.patch = patch

        def bump_major(self):
            """Увеличить MAJOR версию."""
            self.major += 1
            self.minor = 0
            self.patch = 0

        def bump_minor(self):
            """УCREASE MINOR версию."""
            self.minor += 1
            self.patch = 0

        def bump_patch(self):
            """УCREASE PATCH версию."""
            self.patch += 1

        def __str__(self):
            return f"{self.major}.{self.minor}.{self.patch}"

        def __eq__(self, other):
            return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

        def __lt__(self, other):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    # Симуляция жизненного цикла версий
    version = SemanticVersion(1, 0, 0)
    print(f"   Начальная версия: {version}")

    changes = [
        ("patch", "Исправлен баг в предобработке"),
        ("patch", "Оптимизирована скорость инференса"),
        ("minor", "Добавлена поддержка новых языков"),
        ("major", "Изменён формат входных данных (breaking)"),
        ("patch", "Исправлена утечка памяти"),
    ]

    for change_type, description in changes:
        if change_type == "major":
            version.bump_major()
        elif change_type == "minor":
            version.bump_minor()
        else:
            version.bump_patch()
        print(f"   {change_type:6}: v{version} — {description}")

    # 2.2 Model Registry
    print("\n2.2 Model Registry (реестр моделей):")

    class ModelRegistry:
        """Реестр моделей."""
        def __init__(self):
            self.models = {}
            self.artifacts = {}

        def register(self, model_name, version, metadata):
            """Зарегистрировать модель."""
            key = f"{model_name}:{version}"
            self.models[key] = {
                "name": model_name,
                "version": version,
                "metadata": metadata,
                "status": "registered",
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "artifact_hash": hashlib.sha256(json.dumps(metadata).encode()).hexdigest()[:16],
            }
            return self.models[key]

        def promote(self, model_name, version, stage):
            """Продвинуть модель в stage (staging -> production)."""
            key = f"{model_name}:{version}"
            if key in self.models:
                self.models[key]["status"] = stage
                print(f"   Продвижение {key} -> {stage}")
            return self.models.get(key)

        def list_models(self):
            """Список всех моделей."""
            return self.models

    registry = ModelRegistry()

    # Регистрация моделей
    models_to_register = [
        ("sentiment-analyzer", "1.0.0", {"accuracy": 0.85, "f1": 0.83, "framework": "pytorch"}),
        ("sentiment-analyzer", "1.1.0", {"accuracy": 0.87, "f1": 0.86, "framework": "pytorch"}),
        ("sentiment-analyzer", "2.0.0", {"accuracy": 0.91, "f1": 0.90, "framework": "pytorch", "breaking": True}),
        ("text-classifier", "1.0.0", {"accuracy": 0.82, "f1": 0.80, "framework": "tensorflow"}),
    ]

    print("   Регистрация моделей:")
    for name, version, meta in models_to_register:
        result = registry.register(name, version, meta)
        print(f"     {name}:{version} — accuracy={meta['accuracy']}, hash={result['artifact_hash']}")

    # Продвижение в production
    print("\n   Продвижение моделей:")
    registry.promote("sentiment-analyzer", "2.0.0", "production")
    registry.promote("text-classifier", "1.0.0", "staging")

    # 2.3 Model Lineage (происхождение)
    print("\n2.3 Model Lineage (происхождение модели):")

    class ModelLineage:
        """Отслеживание происхождения модели."""
        def __init__(self):
            self.lineage = {}

        def record(self, model_name, version, inputs, outputs, parameters):
            """Записать lineage для модели."""
            key = f"{model_name}:{version}"
            self.lineage[key] = {
                "inputs": inputs,
                "outputs": outputs,
                "parameters": parameters,
                "parent_model": None,
            }

        def set_parent(self, child_key, parent_key):
            """Установить родительскую модель."""
            if child_key in self.lineage:
                self.lineage[child_key]["parent_model"] = parent_key

        def trace(self, model_key):
            """Проследить lineage модели."""
            chain = []
            current = model_key
            while current:
                chain.append(current)
                current = self.lineage.get(current, {}).get("parent_model")
            return chain

    lineage = ModelLineage()
    lineage.record("sentiment-analyzer", "1.0.0",
                   inputs=["reviews_v1.csv"],
                   outputs=["sentiment-v1.0.0.pt"],
                   parameters={"lr": 0.001, "epochs": 100})
    lineage.record("sentiment-analyzer", "2.0.0",
                   inputs=["reviews_v2.csv"],
                   outputs=["sentiment-v2.0.0.pt"],
                   parameters={"lr": 0.0005, "epochs": 150})
    lineage.set_parent("sentiment-analyzer:2.0.0", "sentiment-analyzer:1.0.0")

    print("   Lineage для sentiment-analyzer:")
    chain = lineage.trace("sentiment-analyzer:2.0.0")
    for i, model in enumerate(chain):
        info = lineage.lineage[model]
        indent = "  " * i
        print(f"   {indent}{model}")
        print(f"   {indent}  inputs: {info['inputs']}")
        print(f"   {indent}  outputs: {info['outputs']}")
        print(f"   {indent}  params: {info['parameters']}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: Experiment Tracking (metrics logging, artifact storage, comparison)
# ─────────────────────────────────────────────────────────────────────────────
def demo_experiment_tracking():
    """Демонстрация трекинга экспериментов."""
    print("\n" + "=" * 70)
    print("DEMO 3: Experiment Tracking (Metrics Logging, Artifact Storage)")
    print("=" * 70)

    # 3.1 Эксперимент и прогоны
    print("\n3.1 Эксперимент и прогоны (runs):")

    class ExperimentTracker:
        """Трекер экспериментов."""
        def __init__(self, experiment_name):
            self.experiment_name = experiment_name
            self.runs = []
            self.current_run = None

        def start_run(self, run_name, params):
            """Начать новый run."""
            self.current_run = {
                "run_id": hashlib.md5(run_name.encode()).hexdigest()[:8],
                "name": run_name,
                "params": params,
                "metrics": [],
                "artifacts": [],
                "status": "running",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            return self.current_run

        def log_metric(self, key, value, step=None):
            """Записать метрику."""
            if self.current_run:
                self.current_run["metrics"].append({
                    "key": key, "value": value, "step": step
                })

        def log_artifact(self, name, path, size_bytes):
            """Записать артефакт."""
            if self.current_run:
                self.current_run["artifacts"].append({
                    "name": name, "path": path, "size": size_bytes
                })

        def end_run(self, status="completed"):
            """Завершить run."""
            if self.current_run:
                self.current_run["status"] = status
                self.current_run["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.runs.append(self.current_run)
                self.current_run = None

        def compare_runs(self):
            """Сравнить все run-ы."""
            return self.runs

    tracker = ExperimentTracker("sentiment-analysis")

    # Run 1: базовая модель
    run1 = tracker.start_run("baseline-logistic", {
        "model": "logistic_regression", "lr": 0.01, "epochs": 50
    })
    for epoch in range(0, 50, 10):
        tracker.log_metric("loss", 0.5 - epoch * 0.008, step=epoch)
        tracker.log_metric("accuracy", 0.7 + epoch * 0.005, step=epoch)
    tracker.log_artifact("model.pkl", "artifacts/baseline/model.pkl", 1024000)
    tracker.end_run()

    # Run 2: улучшенная модель
    run2 = tracker.start_run("improved-cnn", {
        "model": "cnn", "lr": 0.001, "epochs": 100, "hidden_dim": 128
    })
    for epoch in range(0, 100, 10):
        tracker.log_metric("loss", 0.4 - epoch * 0.003, step=epoch)
        tracker.log_metric("accuracy", 0.75 + epoch * 0.002, step=epoch)
    tracker.log_artifact("model.pt", "artifacts/cnn/model.pt", 52428800)
    tracker.end_run()

    print("   Зарегистрированные run-ы:")
    for run in tracker.runs:
        final_metrics = {}
        for m in run["metrics"]:
            final_metrics[m["key"]] = m["value"]
        print(f"     {run['name']} (id={run['run_id']}):")
        print(f"       Params: {run['params']}")
        print(f"       Final metrics: {final_metrics}")
        print(f"       Artifacts: {[a['name'] for a in run['artifacts']]}")

    # 3.2 Сравнение экспериментов
    print("\n3.2 Сравнение экспериментов:")

    runs = tracker.compare_runs()
    print("   +------------------+----------+--------+----------+----------+")
    print("   | Run              | Model    | LR     | Loss     | Accuracy |")
    print("   +------------------+----------+--------+----------+----------+")
    for run in runs:
        metrics = {m["key"]: m["value"] for m in run["metrics"]}
        # Берём последние значения
        losses = [m["value"] for m in run["metrics"] if m["key"] == "loss"]
        accs = [m["value"] for m in run["metrics"] if m["key"] == "accuracy"]
        final_loss = losses[-1] if losses else "N/A"
        final_acc = accs[-1] if accs else "N/A"
        model = run["params"].get("model", "N/A")
        lr = run["params"].get("lr", "N/A")
        print(f"   | {run['name']:16} | {model:8} | {lr} | {final_loss:8.4f} | {final_acc:8.4f} |")
    print("   +------------------+----------+--------+----------+----------+")

    # 3.3 Хэширование артефактов для целостности
    print("\n3.3 Хэширование артефактов для целостности:")

    def compute_artifact_hash(data):
        """Вычислить хэш артефакта."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    artifacts_data = [
        {"name": "model_v1.pkl", "weights": [0.1, 0.2, 0.3], "bias": 0.5},
        {"name": "model_v2.pkl", "weights": [0.15, 0.25, 0.35], "bias": 0.45},
        {"name": "tokenizer.json", "vocab_size": 30000, "special_tokens": ["[PAD]", "[UNK]"]},
    ]

    for artifact in artifacts_data:
        h = compute_artifact_hash(artifact)
        print(f"   {artifact['name']}: sha256={h[:24]}...")

    # 3.4 Автоматический выбор лучшей модели
    print("\n3.4 Автоматический выбор лучшей модели:")

    class ModelSelector:
        """Автоматический выбор лучшей модели."""
        def __init__(self, metric="accuracy", mode="max"):
            self.metric = metric
            self.mode = mode

        def select(self, runs):
            """Выбрать лучший run."""
            best = None
            best_value = None
            for run in runs:
                metrics = {m["key"]: m["value"] for m in run["metrics"]}
                # Берём последнее значение метрики
                metric_values = [m["value"] for m in run["metrics"] if m["key"] == self.metric]
                if metric_values:
                    value = metric_values[-1]
                    if best is None or (self.mode == "max" and value > best_value) or \
                       (self.mode == "min" and value < best_value):
                        best = run
                        best_value = value
            return best, best_value

    selector = ModelSelector(metric="accuracy", mode="max")
    best_run, best_acc = selector.select(runs)
    print(f"   Критерий: maximize(accuracy)")
    print(f"   Лучший run: {best_run['name']} (accuracy={best_acc:.4f})")
    print(f"   -> Автоматический промоут в staging")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: Automated Testing (data tests, model tests, integration tests)
# ─────────────────────────────────────────────────────────────────────────────
def demo_automated_testing():
    """Демонстрация автоматизированного тестирования ML."""
    print("\n" + "=" * 70)
    print("DEMO 4: Automated Testing (Data Tests, Model Tests, Integration Tests)")
    print("=" * 70)

    # 4.1 Data Tests (тесты данных)
    print("\n4.1 Data Tests (тесты качества данных):")

    class DataTester:
        """Тесты для проверки качества данных."""
        def __init__(self):
            self.results = []

        def test_schema(self, data, expected_columns):
            """Проверка схемы данных."""
            actual_columns = set(data.keys())
            expected = set(expected_columns)
            missing = expected - actual_columns
            extra = actual_columns - expected
            passed = len(missing) == 0
            self.results.append({
                "test": "schema_check",
                "passed": passed,
                "missing": list(missing),
                "extra": list(extra),
            })
            return passed

        def test_null_ratio(self, data, column, max_null_ratio=0.1):
            """Проверка доли пропусков."""
            values = data.get(column, [])
            if not values:
                self.results.append({"test": f"null_ratio_{column}", "passed": False})
                return False
            null_count = sum(1 for v in values if v is None)
            null_ratio = null_count / len(values)
            passed = null_ratio <= max_null_ratio
            self.results.append({
                "test": f"null_ratio_{column}",
                "passed": passed,
                "null_ratio": round(null_ratio, 4),
                "max_allowed": max_null_ratio,
            })
            return passed

        def test_value_range(self, data, column, min_val, max_val):
            """Проверка диапазона значений."""
            values = [v for v in data.get(column, []) if v is not None]
            if not values:
                self.results.append({"test": f"range_{column}", "passed": False})
                return False
            actual_min = min(values)
            actual_max = max(values)
            passed = actual_min >= min_val and actual_max <= max_val
            self.results.append({
                "test": f"range_{column}",
                "passed": passed,
                "actual_range": (actual_min, actual_max),
                "expected_range": (min_val, max_val),
            })
            return passed

        def test_unique_ratio(self, data, column, min_unique_ratio=0.5):
            """Проверка доли уникальных значений."""
            values = [v for v in data.get(column, []) if v is not None]
            if not values:
                self.results.append({"test": f"unique_{column}", "passed": False})
                return False
            unique_ratio = len(set(values)) / len(values)
            passed = unique_ratio >= min_unique_ratio
            self.results.append({
                "test": f"unique_{column}",
                "passed": passed,
                "unique_ratio": round(unique_ratio, 4),
            })
            return passed

        def print_results(self):
            """Вывести результаты."""
            for r in self.results:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"     [{status}] {r['test']}: {r}")

    tester = DataTester()

    # Тестовые данные
    sample_data = {
        "id": list(range(1, 1001)),
        "text": [f"review_{i}" for i in range(1, 1001)],
        "rating": [random.randint(1, 5) for _ in range(1000)],
        "price": [random.uniform(10, 1000) for _ in range(1000)],
        "category": [random.choice(["A", "B", "C", "D"]) for _ in range(1000)],
    }
    # Добавляем несколько пропусков
    for i in range(0, 1000, 100):
        sample_data["price"][i] = None

    print("   Запуск тестов качества данных:")
    tester.test_schema(sample_data, ["id", "text", "rating", "price", "category"])
    tester.test_null_ratio(sample_data, "price", max_null_ratio=0.15)
    tester.test_value_range(sample_data, "rating", 1, 5)
    tester.test_unique_ratio(sample_data, "category", min_unique_ratio=0.1)
    tester.print_results()

    # 4.2 Model Tests (тесты модели)
    print("\n4.2 Model Tests (тесты модели):")

    class ModelTester:
        """Тесты для проверки модели."""
        def __init__(self):
            self.results = []

        def test_prediction_range(self, predictions, min_val=0, max_val=1):
            """Проверка диапазона предсказаний."""
            out_of_range = [p for p in predictions if p < min_val or p > max_val]
            passed = len(out_of_range) == 0
            self.results.append({
                "test": "prediction_range",
                "passed": passed,
                "out_of_range_count": len(out_of_range),
            })
            return passed

        def test_prediction_distribution(self, predictions, expected_mean, tolerance=0.1):
            """Проверка распределения предсказаний."""
            mean = sum(predictions) / len(predictions)
            passed = abs(mean - expected_mean) <= tolerance
            self.results.append({
                "test": "prediction_distribution",
                "passed": passed,
                "actual_mean": round(mean, 4),
                "expected_mean": expected_mean,
            })
            return passed

        def test_latency(self, predict_fn, test_input, max_latency_ms=100):
            """Проверка латентности инференса."""
            start = time.time()
            for _ in range(100):
                predict_fn(test_input)
            elapsed_ms = (time.time() - start) * 1000 / 100
            passed = elapsed_ms <= max_latency_ms
            self.results.append({
                "test": "latency",
                "passed": passed,
                "actual_ms": round(elapsed_ms, 2),
                "max_ms": max_latency_ms,
            })
            return passed

        def test_model_determinism(self, predict_fn, test_input, n_runs=10):
            """Проверка детерминированности модели."""
            results = [predict_fn(test_input) for _ in range(n_runs)]
            all_same = all(r == results[0] for r in results)
            self.results.append({
                "test": "determinism",
                "passed": all_same,
                "unique_results": len(set(results)),
            })
            return all_same

    model_tester = ModelTester()

    # Симуляция предсказаний
    mock_predictions = [random.random() for _ in range(100)]
    mock_predict = lambda x: 0.85  # Детерминированная функция

    print("   Запуск тестов модели:")
    model_tester.test_prediction_range(mock_predictions, 0, 1)
    model_tester.test_prediction_distribution(mock_predictions, expected_mean=0.5, tolerance=0.15)
    model_tester.test_latency(mock_predict, test_input="test", max_latency_ms=10)
    model_tester.test_model_determinism(mock_predict, test_input="test")

    for r in model_tester.results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"     [{status}] {r['test']}: {r}")

    # 4.3 Integration Tests (интеграционные тесты)
    print("\n4.3 Integration Tests (интеграционные тесты):")

    class IntegrationTester:
        """Интеграционные тесты ML-системы."""
        def __init__(self):
            self.results = []

        def test_api_endpoint(self, url, expected_status=200):
            """Тест API endpoint."""
            # Симуляция HTTP-запроса
            status = 200 if random.random() < 0.95 else 500
            latency = random.randint(10, 200)
            passed = status == expected_status
            self.results.append({
                "test": f"api_{url}",
                "passed": passed,
                "status": status,
                "latency_ms": latency,
            })
            return passed

        def test_batch_prediction(self, predict_fn, batch_size, expected_throughput):
            """Тест batch-предсказаний."""
            start = time.time()
            for _ in range(batch_size):
                predict_fn([0.1, 0.2])
            elapsed = time.time() - start
            throughput = batch_size / elapsed
            passed = throughput >= expected_throughput
            self.results.append({
                "test": "batch_prediction",
                "passed": passed,
                "throughput": round(throughput, 1),
                "expected": expected_throughput,
            })
            return passed

        def test_model_loading(self, model_path):
            """Тест загрузки модели."""
            # Симуляция загрузки
            load_time = random.uniform(0.5, 2.0)
            loaded = random.random() < 0.99
            self.results.append({
                "test": "model_loading",
                "passed": loaded,
                "load_time_s": round(load_time, 2),
                "path": model_path,
            })
            return loaded

    integration_tester = IntegrationTester()

    print("   Запуск интеграционных тестов:")
    mock_predict = lambda x: 0.85
    integration_tester.test_api_endpoint("/v1/models/sentiment/predict")
    integration_tester.test_batch_prediction(mock_predict, batch_size=100, expected_throughput=50)
    integration_tester.test_model_loading("artifacts/sentiment-v2.0.0.pt")

    for r in integration_tester.results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"     [{status}] {r['test']}: {r}")

    # 4.4 Сводка тестирования
    print("\n4.4 Сводка тестирования:")
    all_results = (
        [(r["test"], r["passed"]) for r in tester.results] +
        [(r["test"], r["passed"]) for r in model_tester.results] +
        [(r["test"], r["passed"]) for r in integration_tester.results]
    )
    total = len(all_results)
    passed = sum(1 for _, p in all_results if p)
    failed = total - passed

    print(f"   Всего тестов: {total}")
    print(f"   Пройдено: {passed} ({passed/total*100:.1f}%)")
    print(f"   Провалено: {failed} ({failed/total*100:.1f}%)")

    if failed == 0:
        print("   -> Все тесты пройдены, модель готова к деплою!")
    else:
        print("   -> Есть проваленные тесты, деплой заблокирован")
        print("   Проваленные тесты:")
        for name, p in all_results:
            if not p:
                print(f"     - {name}")


if __name__ == "__main__":
    demo_ml_pipelines()
    demo_model_versioning()
    demo_experiment_tracking()
    demo_automated_testing()
