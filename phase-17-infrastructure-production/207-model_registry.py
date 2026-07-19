"""
207 — Model Registry & Governance: версионирование, lineage, рабочие процессы утверждения

Темы:
  1. Model Versioning (семантическое версионирование, переходы между этапами, откат)
  2. Model Lineage (происхождение данных, пайплайн обучения, зависимости)
  3. Governance Workflows (контрольные точки одобрения, чек-листы обзора, аудит-трейл)
  4. Model Catalog (поиск, метаданные, сравнение, статус деплоя)

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


# ──────────────────────────────────────────────────────────────────────
# Демо 1: Версионирование моделей
# ──────────────────────────────────────────────────────────────────────
def demo_model_versioning():
    """Семантическое версионирование, этапы жизненного цикла, откат."""
    print("=" * 70)
    print("ДЕМО 1: Версионирование моделей (Model Versioning)")
    print("=" * 70)

    # --- 1.1 Семантическое версионирование ---
    print("\n--- 1.1 Семантическое версионирование (MAJOR.MINOR.PATCH) ---")

    class SemanticVersion:
        """Представление семантической версии: MAJOR.MINOR.PATCH."""

        def __init__(self, major: int, minor: int, patch: int):
            self.major = major
            self.minor = minor
            self.patch = patch

        def __str__(self):
            return f"{self.major}.{self.minor}.{self.patch}"

        def __repr__(self):
            return f"SemanticVersion({self})"

        def bump_major(self):
            """MAJOR — несовместимые изменения API модели."""
            return SemanticVersion(self.major + 1, 0, 0)

        def bump_minor(self):
            """MINOR — новая функциональность, обратно совместимая."""
            return SemanticVersion(self.major, self.minor + 1, 0)

        def bump_patch(self):
            """PATCH — исправления багов, метрик, без изменения API."""
            return SemanticVersion(self.major, self.minor, self.patch + 1)

        def __lt__(self, other):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    v = SemanticVersion(1, 0, 0)
    print(f"  Начальная версия модели:  v{v}")

    v_minor = v.bump_minor()
    print(f"  После добавления фичи:  v{v_minor}")

    v_patch = v_minor.bump_patch()
    print(f"  После исправления бага:  v{v_patch}")

    v_major = v_patch.bump_major()
    print(f"  После смены архитектуры: v{v_major}")

    print(f"\n  Формула: MAJOR.MINOR.PATCH")
    print(f"  MAJOR = несовместимые изменения API")
    print(f"  MINOR  = обратно совместимая новая фича")
    print(f"  PATCH  = обратно совместимое исправление бага")

    # --- 1.2 Переходы между этапами жизненного цикла ---
    print("\n--- 1.2 Переходы между этапами жизненного цикла ---")

    VALID_STAGES = ["development", "staging", "production", "archived"]
    VALID_TRANSITIONS = {
        "development": ["staging"],
        "staging": ["development", "production"],
        "production": ["staging", "archived"],
        "archived": ["development"],
    }

    class ModelRecord:
        """Запись модели в реестре с историей переходов."""

        def __init__(self, name: str, version: str, stage: str = "development"):
            self.name = name
            self.version = version
            self.stage = stage
            self.transition_history = [(stage, time.time())]

        def transition(self, target_stage: str) -> bool:
            """Переход на другой этап с проверкой допустимости."""
            allowed = VALID_TRANSITIONS.get(self.stage, [])
            if target_stage not in allowed:
                print(f"  ❌ Переход {self.stage} → {target_stage} НЕ допустим!")
                print(f"     Допустимые переходы из {self.stage}: {allowed}")
                return False
            self.transition_history.append((target_stage, time.time()))
            old_stage = self.stage
            self.stage = target_stage
            print(f"  ✅ Переход выполнен: {old_stage} → {target_stage}")
            return True

    model = ModelRecord("bert-sentiment", "2.1.0", "development")
    print(f"  Модель: {model.name} v{model.version} [{model.stage}]")
    model.transition("staging")
    model.transition("production")

    # Попытка нелегального перехода
    model.transition("development")

    print(f"\n  История переходов ({len(model.transition_history)} записей):")
    for stage, ts in model.transition_history:
        print(f"    → {stage}")

    # --- 1.3 Откат к предыдущей версии ---
    print("\n--- 1.3 Откат к предыдущей версии (Rollback) ---")

    class VersionRegistry:
        """Реестр версий модели с поддержкой отката."""

        def __init__(self, model_name: str):
            self.model_name = model_name
            self.versions = []  # список (version, stage, metrics)
            self.current_index = -1

        def register(self, version: str, metrics: dict):
            """Регистрация новой версии."""
            stage = "development"
            self.versions.append({"version": version, "stage": stage, "metrics": metrics})
            self.current_index = len(self.versions) - 1
            print(f"  📦 Зарегистрирована v{version}: {metrics}")

        def promote(self, version: str, stage: str):
            """Продвижение версии на этап."""
            for v in self.versions:
                if v["version"] == version:
                    v["stage"] = stage
                    print(f"  🚀 v{version} продвинута в {stage}")
                    break

        def rollback(self) -> dict:
            """Откат к предыдущей production-версии."""
            production_versions = [
                i for i, v in enumerate(self.versions) if v["stage"] == "production"
            ]
            if len(production_versions) < 2:
                print("  ❌ Нет предыдущей production-версии для отката")
                return {}
            prev_idx = production_versions[-2]
            self.current_index = prev_idx
            prev = self.versions[prev_idx]
            print(f"  ⏪ Откат к v{prev['version']} ({prev['metrics']})")
            return prev

    registry = VersionRegistry("bert-sentiment")
    registry.register("1.0.0", {"accuracy": 0.82, "f1": 0.80})
    registry.register("1.1.0", {"accuracy": 0.85, "f1": 0.83})
    registry.register("2.0.0", {"accuracy": 0.83, "f1": 0.79})  # регрессия

    registry.promote("1.0.0", "production")
    registry.promote("1.1.0", "production")
    registry.promote("2.0.0", "production")

    print(f"\n  Текущая production-версия: v{registry.versions[registry.current_index]['version']}")
    print(f"  Метрики: {registry.versions[registry.current_index]['metrics']}")
    print(f"\n  Обнаружена регрессия! Выполняем откат...")
    rollback_info = registry.rollback()

    print(f"\n  Сводка версий в реестре:")
    for v in registry.versions:
        print(f"    v{v['version']:>6s}  [{v['stage']:>12s}]  acc={v['metrics']['accuracy']:.2f}")

    # --- 1.4 Сравнение версий по метрикам ---
    print("\n--- 1.4 Сравнение версий по метрикам ---")

    def compare_versions(v1: dict, v2: dict) -> dict:
        """Сравнение двух версий по всем метрикам."""
        result = {}
        all_keys = set(v1["metrics"].keys()) | set(v2["metrics"].keys())
        for key in all_keys:
            val1 = v1["metrics"].get(key, 0)
            val2 = v2["metrics"].get(key, 0)
            diff = val2 - val1
            pct = (diff / val1 * 100) if val1 != 0 else float("inf")
            direction = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            result[key] = {"v1": val1, "v2": val2, "diff": diff, "pct": pct, "direction": direction}
        return result

    older = {"version": "1.0.0", "metrics": {"accuracy": 0.82, "f1": 0.80, "latency_ms": 12.5}}
    newer = {"version": "1.1.0", "metrics": {"accuracy": 0.85, "f1": 0.83, "latency_ms": 11.2}}

    comp = compare_versions(older, newer)
    print(f"  Сравнение v{older['version']} vs v{newer['version']}:")
    print(f"  {'Метрика':<15} {'v1':>8} {'v2':>8} {'Δ':>8} {'%':>8} {'Тренд'}")
    print(f"  {'-'*65}")
    for metric, info in comp.items():
        print(f"  {metric:<15} {info['v1']:>8.3f} {info['v2']:>8.3f} "
              f"{info['diff']:>+8.3f} {info['pct']:>+7.1f}% {info['direction']}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 2: Линеалогия моделей (Model Lineage)
# ──────────────────────────────────────────────────────────────────────
def demo_model_lineage():
    """Происхождение данных, пайплайн обучения, зависимости."""
    print("=" * 70)
    print("ДЕМО 2: Линеалогия моделей (Model Lineage)")
    print("=" * 70)

    # --- 2.1 Происхождение данных (Data Provenance) ---
    print("\n--- 2.1 Происхождение данных (Data Provenance) ---")

    class DataSource:
        """Источник данных с отслеживанием происхождения."""

        def __init__(self, name: str, source_type: str, record_count: int, checksum: str):
            self.name = name
            self.source_type = source_type  # database, api, file, streaming
            self.record_count = record_count
            self.checksum = checksum
            self.created_at = time.time()

        def fingerprint(self) -> str:
            """Уникальный отпечаток источника для отслеживания."""
            raw = f"{self.name}:{self.source_type}:{self.record_count}:{self.checksum}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]

        def __repr__(self):
            return f"DataSource({self.name}, {self.source_type}, {self.record_count} records)"

    # Создание цепочки источников
    raw_data = DataSource("raw_reviews.csv", "file", 50000, "a1b2c3d4")
    cleaned = DataSource("cleaned_reviews.parquet", "file", 48500, "e5f6a7b8")
    augmented = DataSource("augmented_reviews.parquet", "file", 48500, "c9d0e1f2")

    print(f"  Источники данных:")
    for ds in [raw_data, cleaned, augmented]:
        print(f"    {ds.name:35s} [{ds.source_type:>8s}] {ds.record_count:>6d} записей  "
              f"fingerprint={ds.fingerprint()}")

    # Цепочка трансформаций
    print(f"\n  Цепочка трансформаций (lineage graph):")
    print(f"    {raw_data.name}")
    print(f"      ↓  Очистка: удаление дубликатов, стемминг")
    print(f"    {cleaned.name} (-{raw_data.record_count - cleaned.record_count} записей)")
    print(f"      ↓  Аугментация: back-translation, synonym replacement")
    print(f"    {augmented.name} (тот же объём)")

    # --- 2.2 Метаданные пайплайна обучения ---
    print("\n--- 2.2 Метаданные пайплайна обучения ---")

    training_run = {
        "run_id": "run-2024-01-15-001",
        "model_name": "sentiment-bert",
        "framework": "PyTorch",
        "framework_version": "2.1.0",
        "python_version": "3.11.7",
        "dataset": {
            "name": "augmented_reviews.parquet",
            "split": {"train": 38800, "val": 4850, "test": 4850},
            "features": ["text", "label"],
            "num_classes": 3,
        },
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 5,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "max_seq_length": 256,
            "optimizer": "AdamW",
            "scheduler": "linear_with_warmup",
        },
        "compute": {
            "gpus": 2,
            "gpu_type": "NVIDIA A100 40GB",
            "training_time_seconds": 1847,
            "peak_memory_gb": 14.2,
        },
        "results": {
            "accuracy": 0.912,
            "f1_macro": 0.897,
            "f1_per_class": [0.93, 0.87, 0.89],
            "confusion_matrix": [[1620, 30, 15], [45, 1560, 25], [20, 35, 1595]],
        },
    }

    print(f"  ID запуска:         {training_run['run_id']}")
    print(f"  Модель:             {training_run['model_name']}")
    print(f"  Фреймворк:          {training_run['framework']} {training_run['framework_version']}")
    print(f"  Python:             {training_run['python_version']}")
    print(f"  Датасет:            {training_run['dataset']['name']}")
    print(f"  Split:              train={training_run['dataset']['split']['train']}, "
          f"val={training_run['dataset']['split']['val']}, "
          f"test={training_run['dataset']['split']['test']}")
    print(f"  Гиперпараметры:")
    for k, v in training_run["hyperparameters"].items():
        print(f"    {k:<25s} = {v}")
    print(f"  Вычисления:")
    for k, v in training_run["compute"].items():
        print(f"    {k:<25s} = {v}")
    print(f"  Результаты:")
    for k, v in training_run["results"].items():
        if k != "confusion_matrix":
            print(f"    {k:<25s} = {v}")

    # --- 2.3 Зависимости (Dependencies) ---
    print("\n--- 2.3 Зависимости модели ---")

    dependencies = {
        "data_sources": [
            {"name": "reviews_db", "type": "PostgreSQL", "version": "15.4", "schema": "v2"},
            {"name": "sentiment_labels", "type": "API", "endpoint": "/api/v1/labels", "version": "1.3"},
        ],
        "preprocessing": [
            {"name": "tokenizers", "library": "HuggingFace", "version": "0.15.0"},
            {"name": "text_cleaner", "library": "custom", "version": "2.1.0", "hash": "deadbeef1234"},
        ],
        "base_model": {
            "name": "bert-base-uncased",
            "source": "HuggingFace Hub",
            "version": "1.0",
            "num_params": 110000000,
        },
        "training_infra": {
            "framework": "PyTorch 2.1.0",
            "distributed": "DDP",
            "experiment_tracker": "MLflow 2.9.0",
        },
    }

    print(f"  Источники данных:")
    for ds in dependencies["data_sources"]:
        print(f"    • {ds['name']} ({ds['type']}, v{ds['version']})")

    print(f"  Предобработка:")
    for pp in dependencies["preprocessing"]:
        print(f"    • {pp['name']} ({pp['library']}, v{pp['version']})")

    bm = dependencies["base_model"]
    print(f"  Базовая модель:")
    print(f"    • {bm['name']} ({bm['source']}, {bm['num_params']:,} параметров)")

    ti = dependencies["training_infra"]
    print(f"  Инфраструктура:")
    print(f"    • {ti['framework']} | {ti['distributed']} | {ti['experiment_tracker']}")

    # --- 2.4 Хэширование для проверки целостности ---
    print("\n--- 2.4 Проверка целостности артефактов (artifact integrity) ---")

    def compute_artifact_hash(data: dict) -> str:
        """Вычисление SHA-256 хэша артефакта для проверки целостности."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    artifacts = [
        {"name": "model_weights.bin", "size_mb": 440, "epoch": 5},
        {"name": "tokenizer.json", "size_mb": 2.3, "vocab_size": 30522},
        {"name": "config.json", "size_mb": 0.001, "hidden_size": 768},
        {"name": "training_config.yaml", "size_mb": 0.005, "lr": 2e-5},
    ]

    print(f"  Артефакты и их хэши:")
    for art in artifacts:
        h = compute_artifact_hash(art)
        print(f"    {art['name']:<30s} size={art['size_mb']:>7.1f} MB  sha256={h}")

    # Проверка: повторный хэш должен совпадать
    rehash = compute_artifact_hash(artifacts[0])
    print(f"\n  Повторная проверка model_weights.bin: {rehash}")
    print(f"  Целостность: {'✅ ПОДТВЕРЖДЕНА' if rehash == compute_artifact_hash(artifacts[0]) else '❌ НАРУШЕНА'}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 3: Процессы управления (Governance Workflows)
# ──────────────────────────────────────────────────────────────────────
def demo_governance_workflows():
    """Контрольные точки одобрения, чек-листы, аудит-трейл."""
    print("=" * 70)
    print("ДЕМО 3: Процессы управления (Governance Workflows)")
    print("=" * 70)

    # --- 3.1 Контрольные точки одобрения (Approval Gates) ---
    print("\n--- 3.1 Контрольные точки одобрения (Approval Gates) ---")

    class ApprovalGate:
        """Контрольная точка, требующая одобрения перед продолжением."""

        def __init__(self, name: str, required_approvers: list, checklist: list):
            self.name = name
            self.required_approvers = required_approvers
            self.checklist = checklist  # [(item, passed)]
            self.approvals = {}
            self.timestamp = None

        def check_item(self, item: str, passed: bool):
            """Отметка выполнения пункта чек-листа."""
            for i, (name, _) in enumerate(self.checklist):
                if name == item:
                    self.checklist[i] = (name, passed)
                    status = "✅" if passed else "❌"
                    print(f"    {status} {item}")
                    return

        def approve(self, approver: str) -> bool:
            """Одобрение контрольной точки."""
            if approver not in self.required_approvers:
                print(f"    ❌ {approver} не является требуемым одобрителем")
                return False
            all_passed = all(passed for _, passed in self.checklist)
            if not all_passed:
                print(f"    ❌ Не все пункты чек-листа выполнены!")
                return False
            self.approvals[approver] = time.time()
            print(f"    ✅ {approver} одобрил контрольную точку '{self.name}'")
            return True

        def is_approved(self) -> bool:
            """Проверка: все ли требуемые одобрения получены."""
            return all(a in self.approvals for a in self.required_approvers)

    # Создание контрольной точки для деплоя в production
    gate = ApprovalGate(
        name="Production Deploy",
        required_approvers=["ml-engineer", "tech-lead", "security-reviewer"],
        checklist=[
            ("Метрики выше baseline", False),
            ("Нет регрессии latency", False),
            ("Безопасность проверена", False),
            ("Документация обновлена", False),
        ],
    )

    print(f"  Контрольная точка: {gate.name}")
    print(f"  Требуемые одобрители: {gate.required_approvers}")
    print(f"  Чек-лист:")
    for item, status in gate.checklist:
        print(f"    {'✅' if status else '⬜'} {item}")

    # Выполнение пунктов
    print(f"\n  Выполнение пунктов чек-листа:")
    gate.check_item("Метрики выше baseline", True)
    gate.check_item("Нет регрессии latency", True)
    gate.check_item("Безопасность проверена", True)
    gate.check_item("Документация обновлена", True)

    # Попытка одобрения без полного набора
    print(f"\n  Одобрение:")
    gate.approve("ml-engineer")
    gate.approve("tech-lead")
    gate.approve("security-reviewer")
    print(f"  Контрольная точка одобрена: {gate.is_approved()}")

    # --- 3.2 Чек-лист обзора модели (Review Checklist) ---
    print("\n--- 3.2 Чек-лист обзора модели (Review Checklist) ---")

    review_checklist = {
        "Качество модели": [
            ("Accuracy >= baseline + 2%", True),
            ("F1-macro >= baseline", True),
            ("Нет деградации на подгруппах", True),
            ("ROC-AUC > 0.90", True),
        ],
        "Производительность": [
            ("Latency p99 < 100ms", True),
            ("Throughput > 1000 req/s", False),
            ("Память < 2GB", True),
        ],
        "Безопасность и этика": [
            ("Нет bias по полу/расе", True),
            ("Нет утечки PII", True),
            ("Adversarial robustness тест пройден", False),
        ],
        "Операционная готовность": [
            ("Мониторинг настроен", True),
            ("Rollback план написан", True),
            ("Документация API обновлена", True),
        ],
    }

    total_items = 0
    passed_items = 0

    for category, items in review_checklist.items():
        cat_passed = sum(1 for _, p in items if p)
        cat_total = len(items)
        total_items += cat_total
        passed_items += cat_passed
        status_icon = "✅" if cat_passed == cat_total else "⚠️"
        print(f"\n  {status_icon} {category} ({cat_passed}/{cat_total}):")
        for item, passed in items:
            icon = "✅" if passed else "❌"
            print(f"    {icon} {item}")

    print(f"\n  Итого: {passed_items}/{total_items} пунктов выполнено "
          f"({passed_items/total_items*100:.0f}%)")
    if passed_items == total_items:
        print(f"  Статус: ✅ Готово к деплою")
    else:
        print(f"  Статус: ⚠️ Требуется доработка ({total_items - passed_items} пунктов)")

    # --- 3.3 Аудит-трейл ---
    print("\n--- 3.3 Аудит-трейл (Audit Trail) ---")

    class AuditTrail:
        """Неизменяемый журнал действий для аудита."""

        def __init__(self):
            self.entries = []
            self._prev_hash = "0" * 64

        def log(self, action: str, actor: str, details: dict):
            """Добавление записи в аудит-трейл."""
            entry = {
                "index": len(self.entries),
                "timestamp": time.time(),
                "action": action,
                "actor": actor,
                "details": details,
                "prev_hash": self._prev_hash,
            }
            # Хэш цепочки (аналог блокчейна)
            entry_str = json.dumps(entry, sort_keys=True)
            entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
            self._prev_hash = entry["hash"]
            self.entries.append(entry)

        def verify(self) -> bool:
            """Проверка целостности аудит-трейла."""
            prev = "0" * 64
            for entry in self.entries:
                if entry["prev_hash"] != prev:
                    return False
                check = dict(entry)
                stored_hash = check.pop("hash")
                check_str = json.dumps(check, sort_keys=True)
                if hashlib.sha256(check_str.encode()).hexdigest() != stored_hash:
                    return False
                prev = stored_hash
            return True

    audit = AuditTrail()
    audit.log("model_registered", "ml-engineer", {"model": "bert-sentiment", "version": "2.0.0"})
    audit.log("stage_transition", "ml-engineer", {"from": "development", "to": "staging"})
    audit.log("review_submitted", "tech-lead", {"status": "approved", "notes": "All checks pass"})
    audit.log("deployed", "platform-engineer", {"env": "production", "replicas": 3})
    audit.log("metric_alert", "monitoring", {"metric": "latency_p99", "value": 105, "threshold": 100})
    audit.log("rollback_initiated", "on-call", {"from": "2.0.0", "to": "1.1.0"})

    print(f"  Записи аудит-трейла ({len(audit.entries)} записей):")
    for entry in audit.entries:
        ts = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
        print(f"    [{ts}] {entry['action']:<25s} by {entry['actor']:<20s} "
              f"hash={entry['hash'][:12]}...")

    print(f"\n  Целостность аудит-трейла: {'✅ ВЕРНА' if audit.verify() else '❌ НАРУШЕНА'}")

    # Симуляция подделки
    print(f"\n  Симуляция подделки записи...")
    original_action = audit.entries[2]["action"]
    audit.entries[2]["action"] = "review_REJECTED"  # подделка!
    print(f"  Целостность после подделки: {'✅ ВЕРНА' if audit.verify() else '❌ НАРУШЕНА'}")
    audit.entries[2]["action"] = original_action  # восстановление

    # --- 3.4 Процесс утверждения модели ---
    print("\n--- 3.4 Процесс утверждения модели (Model Approval Process) ---")

    approval_workflow = [
        {"stage": "1. Регистрация", "actor": "ML Engineer", "action": "Загрузка модели в реестр",
         "artifacts": ["model_weights", "config", "tokenizer"], "status": "done"},
        {"stage": "2. Автотесты", "actor": "CI/CD Pipeline", "action": "Запуск unit/integration тестов",
         "artifacts": ["test_report.json", "coverage.json"], "status": "done"},
        {"stage": "3. Оценка качества", "actor": "ML Engineer", "action": "Запуск eval на test set",
         "artifacts": ["metrics.json", "confusion_matrix.png"], "status": "done"},
        {"stage": "4. Ревью", "actor": "Tech Lead", "action": "Код-ревью + проверка метрик",
         "artifacts": ["review_comments.md"], "status": "done"},
        {"stage": "5. Безопасность", "actor": "Security Team", "action": "Пентест + bias audit",
         "artifacts": ["security_report.pdf", "bias_report.pdf"], "status": "done"},
        {"stage": "6. Стейджинг", "actor": "Platform Engineer", "action": "Деплой в staging + smoke test",
         "artifacts": ["deploy_log.txt", "smoke_results.json"], "status": "done"},
        {"stage": "7. Продакшн", "actor": "Release Manager", "action": "Канареечный деплой + мониторинг",
         "artifacts": ["canary_metrics.json"], "status": "pending"},
    ]

    print(f"  Workflow утверждения модели:\n")
    for step in approval_workflow:
        icon = "✅" if step["status"] == "done" else "⬜"
        print(f"    {icon} {step['stage']}")
        print(f"       Ответственный: {step['actor']}")
        print(f"       Действие:      {step['action']}")
        print(f"       Артефакты:     {', '.join(step['artifacts'])}")
        print()

    done = sum(1 for s in approval_workflow if s["status"] == "done")
    total = len(approval_workflow)
    print(f"  Прогресс: {done}/{total} ({done/total*100:.0f}%)")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 4: Каталог моделей (Model Catalog)
# ──────────────────────────────────────────────────────────────────────
def demo_model_catalog():
    """Поиск, метаданные, сравнение, статус деплоя."""
    print("=" * 70)
    print("ДЕМО 4: Каталог моделей (Model Catalog)")
    print("=" * 70)

    # --- 4.1 Регистрация и метаданные ---
    print("\n--- 4.1 Регистрация моделей и метаданные ---")

    class CatalogEntry:
        """Запись в каталоге моделей."""

        def __init__(self, name: str, version: str, task: str, framework: str,
                     metrics: dict, tags: list, description: str = ""):
            self.name = name
            self.version = version
            self.task = task
            self.framework = framework
            self.metrics = metrics
            self.tags = tags
            self.description = description
            self.created_at = time.time()
            self.stage = "registered"
            self.deployments = []

        def __repr__(self):
            return f"CatalogEntry({self.name} v{self.version})"

    catalog = [
        CatalogEntry("bert-sentiment", "2.1.0", "text-classification", "PyTorch",
                      {"accuracy": 0.912, "f1": 0.897, "latency_ms": 15.3},
                      ["nlp", "sentiment", "production"],
                      "Классификатор тональности текста на базе BERT"),
        CatalogEntry("resnet-image", "1.3.2", "image-classification", "TensorFlow",
                      {"accuracy": 0.945, "top5": 0.991, "latency_ms": 8.7},
                      ["cv", "image", "production"],
                      "Классификация изображений на ResNet-50"),
        CatalogEntry("gpt summarizer", "0.9.0", "summarization", "PyTorch",
                      {"rouge_l": 0.42, "latency_ms": 245.0},
                      ["nlp", "summarization", "staging"],
                      "Суммаризация текста, fine-tuned GPT-2"),
        CatalogEntry("xgboost-credit", "3.0.1", "binary-classification", "XGBoost",
                      {"auc": 0.967, "accuracy": 0.934, "latency_ms": 0.8},
                      ["tabular", "finance", "production"],
                      "Рейтинг кредитного скоринга"),
        CatalogEntry("yolo-detection", "2.0.0", "object-detection", "PyTorch",
                      {"map50": 0.78, "map50-95": 0.52, "latency_ms": 22.1},
                      ["cv", "detection", "development"],
                      "Детекция объектов YOLOv8"),
    ]

    print(f"  Каталог моделей ({len(catalog)} моделей):\n")
    print(f"  {'Имя':<22s} {'Версия':<8s} {'Задача':<22s} {'Метрики':<25s} {'Стадия'}")
    print(f"  {'-'*100}")
    for entry in catalog:
        metrics_str = ", ".join(f"{k}={v}" for k, v in entry.metrics.items())
        print(f"  {entry.name:<22s} {entry.version:<8s} {entry.task:<22s} "
              f"{metrics_str:<25s} {entry.stage}")

    # --- 4.2 Поиск по каталогу ---
    print("\n--- 4.2 Поиск по каталогу ---")

    def search_catalog(catalog: list, query: str = "", task: str = "",
                       tags: list = None, min_metric: dict = None) -> list:
        """Поиск моделей в каталоге по различным критериям."""
        results = catalog[:]

        if query:
            query_lower = query.lower()
            results = [e for e in results if query_lower in e.name.lower()
                       or query_lower in e.description.lower()]

        if task:
            results = [e for e in results if e.task == task]

        if tags:
            results = [e for e in results if all(t in e.tags for t in tags)]

        if min_metric:
            for metric_name, min_val in min_metric.items():
                results = [e for e in results if e.metrics.get(metric_name, 0) >= min_val]

        return results

    # Поиск по тексту
    print("  Поиск по запросу 'sentiment':")
    results = search_catalog(catalog, query="sentiment")
    for r in results:
        print(f"    → {r.name} v{r.version}")

    # Поиск по задаче
    print("\n  Поиск по задаче 'image-classification':")
    results = search_catalog(catalog, task="image-classification")
    for r in results:
        print(f"    → {r.name} v{r.version}")

    # Поиск по тегам
    print("\n  Поиск по тегам ['nlp', 'production']:")
    results = search_catalog(catalog, tags=["nlp", "production"])
    for r in results:
        print(f"    → {r.name} v{r.version}")

    # Поиск по минимальным метрикам
    print("\n  Поиск моделей с accuracy >= 0.93:")
    results = search_catalog(catalog, min_metric={"accuracy": 0.93})
    for r in results:
        acc = r.metrics.get("accuracy", "N/A")
        print(f"    → {r.name} v{r.version} (accuracy={acc})")

    # --- 4.3 Сравнение моделей ---
    print("\n--- 4.3 Сравнение моделей ---")

    def compare_models(catalog: list, names: list) -> dict:
        """Сравнение нескольких моделей по всем метрикам."""
        entries = [e for e in catalog if e.name in names]
        if len(entries) < 2:
            return {}

        # Собираем все уникальные метрики
        all_metrics = set()
        for e in entries:
            all_metrics.update(e.metrics.keys())

        comparison = {}
        for metric in sorted(all_metrics):
            values = {}
            for e in entries:
                values[e.name] = e.metrics.get(metric)
            # Находим лучшее значение (для latency — меньше лучше)
            non_none = {k: v for k, v in values.items() if v is not None}
            if non_none:
                if "latency" in metric:
                    best = min(non_none, key=non_none.get)
                else:
                    best = max(non_none, key=non_none.get)
                comparison[metric] = {"values": values, "best": best}
        return comparison

    comp = compare_models(catalog, ["bert-sentiment", "resnet-image", "xgboost-credit"])
    print(f"  Сравнение: bert-sentiment vs resnet-image vs xgboost-credit\n")
    for metric, info in comp.items():
        print(f"  {metric}:")
        for name, val in info["values"].items():
            marker = " ★ ЛУЧШЕ" if name == info["best"] else ""
            print(f"    {name:<22s} = {val}{marker}")
        print()

    # --- 4.4 Статус деплоя ---
    print("\n--- 4.4 Статус деплоя (Deployment Status) ---")

    class DeploymentManager:
        """Управление деплоями моделей."""

        def __init__(self):
            self.deployments = {}

        def deploy(self, model_name: str, version: str, environment: str, replicas: int):
            """Деплой модели в указанное окружение."""
            key = f"{model_name}@{version}"
            self.deployments[key] = {
                "model": model_name,
                "version": version,
                "environment": environment,
                "replicas": replicas,
                "status": "running",
                "deployed_at": time.time(),
                "health": "healthy",
                "traffic_pct": 100,
            }
            print(f"  🚀 Деплой: {model_name} v{version} → {environment} "
                  f"({replicas} реплик)")

        def set_canary(self, model_name: str, version: str, traffic_pct: int):
            """Настройка канареечного деплоя."""
            key = f"{model_name}@{version}"
            if key in self.deployments:
                self.deployments[key]["traffic_pct"] = traffic_pct
                self.deployments[key]["status"] = "canary"
                print(f"  🐤 Канарейка: {model_name} v{version} получает {traffic_pct}% трафика")

        def rollback(self, model_name: str):
            """Откат к предыдущей версии."""
            model_deps = {k: v for k, v in self.deployments.items()
                          if v["model"] == model_name and v["status"] == "running"}
            if model_deps:
                key = list(model_deps.keys())[0]
                self.deployments[key]["status"] = "rolled_back"
                print(f"  ⏪ Откат: {model_deps[key]['model']} v{model_deps[key]['version']}")

        def status_report(self):
            """Отчёт о статусе всех деплоев."""
            print(f"\n  {'Модель':<22s} {'Версия':<8s} {'Окружение':<15s} "
                  f"{'Реплики':>8s} {'Статус':<12s} {'Трафик'}")
            print(f"  {'-'*80}")
            for key, dep in self.deployments.items():
                print(f"  {dep['model']:<22s} {dep['version']:<8s} {dep['environment']:<15s} "
                      f"{dep['replicas']:>8d} {dep['status']:<12s} {dep['traffic_pct']}%")

    dm = DeploymentManager()
    dm.deploy("bert-sentiment", "2.0.0", "production", 3)
    dm.deploy("resnet-image", "1.3.2", "production", 2)
    dm.deploy("xgboost-credit", "3.0.1", "production", 5)

    # Канареечный деплой новой версии
    dm.deploy("bert-sentiment", "2.1.0", "production", 1)
    dm.set_canary("bert-sentiment", "2.1.0", 10)

    # Откат при проблемах
    print(f"\n  Обнаружена проблема! Выполняем откат bert-sentiment...")
    dm.rollback("bert-sentiment")
    dm.deploy("bert-sentiment", "2.0.0", "production", 3)  # восстановление

    dm.status_report()

    # Статистика каталога
    print(f"\n  Статистика каталога:")
    stages = collections.Counter(e.stage for e in catalog)
    tasks = collections.Counter(e.task for e in catalog)
    print(f"    По стадиям:  {dict(stages)}")
    print(f"    По задачам:  {dict(tasks)}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_model_versioning()
    demo_model_lineage()
    demo_governance_workflows()
    demo_model_catalog()
