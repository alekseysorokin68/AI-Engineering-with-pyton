"""
215 — Building ML Platforms: архитектура, инструменты, внедрение

Темы:
  1. Platform Architecture (layers, components, integration points)
  2. Tooling Landscape (orchestration, tracking, serving, monitoring)
  3. Platform Adoption (migration strategy, training, documentation)
  4. Platform Evolution (feedback loops, iteration, scaling the platform)

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


# ============================================================
# Демо 1: Архитектура ML-платформы — слои, компоненты
# ============================================================
def demo_platform_architecture():
    print("=" * 70)
    print("ДЕМО 1: Архитектура ML-платформы — слои, компоненты")
    print("=" * 70)

    # --- 1.1 Многослойная архитектура ---
    print("\n--- 1.1 Многослойная архитектура ML-платформы ---")

    platform_layers = {
        "Layer 4: Consumer API": {
            "description": "API для потребителей моделей (продукты, аналитики)",
            "components": [
                {"name": "Prediction API", "tech": "FastAPI + gRPC",
                 "purpose": "Синхронный инференс", "latency_p99": "50ms"},
                {"name": "Batch API", "tech": "Airflow + Spark",
                 "purpose": "Пакетный инференс", "throughput": "1M rows/hour"},
                {"name": "Feature Store API", "tech": "Feast",
                 "purpose": "Получение фичей в реальном времени",
                 "latency_p99": "10ms"},
                {"name": "Model Registry API", "tech": "MLflow",
                 "purpose": "Управление версиями моделей"},
            ],
        },
        "Layer 3: ML Services": {
            "description": "Сервисы для работы с моделями",
            "components": [
                {"name": "Training Service", "tech": "Kubeflow Pipelines",
                 "purpose": "Оркестрация обучения"},
                {"name": "Experiment Tracker", "tech": "MLflow Tracking",
                 "purpose": "Журнал экспериментов"},
                {"name": "Model Serving", "tech": "Seldon Core / Triton",
                 "purpose": "Деплой моделей"},
                {"name": "Monitoring", "tech": "Prometheus + Grafana",
                 "purpose": "Метрики и алерты"},
            ],
        },
        "Layer 2: Data Platform": {
            "description": "Инфраструктура для данных",
            "components": [
                {"name": "Data Lake", "tech": "S3 + Delta Lake",
                 "purpose": "Хранение сырых и обработанных данных"},
                {"name": "Stream Processing", "tech": "Kafka + Flink",
                 "purpose": "Обработка данных в реальном времени"},
                {"name": "Batch Processing", "tech": "Spark",
                 "purpose": "Пакетная обработка"},
                {"name": "Data Catalog", "tech": "Apache Atlas",
                 "purpose": "Метаданные и lineage"},
            ],
        },
        "Layer 1: Infrastructure": {
            "description": "Базовая инфраструктура",
            "components": [
                {"name": "Compute", "tech": "Kubernetes + GPU nodes",
                 "purpose": "Вычислительные ресурсы"},
                {"name": "Storage", "tech": "S3 + EFS + Redis",
                 "purpose": "Хранение данных и моделей"},
                {"name": "Networking", "tech": "Istio + Envoy",
                 "purpose": "Сетевые политики и балансировка"},
                {"name": "Security", "tech": "Vault + IAM",
                 "purpose": "Управление секретами и доступом"},
            ],
        },
    }

    for layer_name, layer_info in platform_layers.items():
        print(f"\n  ╔══ {layer_name} ══╗")
        print(f"  ║ {layer_info['description']}")
        print(f"  ║")
        for comp in layer_info["components"]:
            tech = comp.get("tech", "—")
            purpose = comp.get("purpose", "—")
            extra = ""
            if "latency_p99" in comp:
                extra = f" | p99: {comp['latency_p99']}"
            elif "throughput" in comp:
                extra = f" | throughput: {comp['throughput']}"
            print(f"  ║   📦 {comp['name']:<25} │ {tech:<25} │ {purpose}{extra}")
        print(f"  ╚{'═' * 50}╝")

    # --- 1.2 Связи между компонентами ---
    print("\n--- 1.2 Интеграционные точки (data flow) ---")

    integrations = [
        ("Данные", "Data Lake → Stream Processing", "Kafka Connect"),
        ("Данные", "Stream Processing → Feature Store", "Flink SQL"),
        ("Обучение", "Feature Store → Training Service", "Feast SDK"),
        ("Обучение", "Training Service → Experiment Tracker", "MLflow API"),
        ("Обучение", "Training Service → Model Registry", "MLflow Register"),
        ("Деплой", "Model Registry → Model Serving", "Seldon Deploy"),
        ("Инференс", "Consumer API → Model Serving", "gRPC/REST"),
        ("Инференс", "Feature Store → Model Serving", "Online Store"),
        ("Мониторинг", "Model Serving → Monitoring", "Prometheus metrics"),
        ("Мониторинг", "Monitoring → Training Service", "Drift alert → Retrain"),
    ]

    print(f"  {'Источник':<12} │ {'Путь':<35} │ {'Протокол/инструмент'}")
    print("  " + "-" * 75)
    for source, path, protocol in integrations:
        print(f"  {source:<12} │ {path:<35} │ {protocol}")

    # --- 1.3 Метрики платформы ---
    print("\n--- 1.3 Ключевые метрики платформы ---")

    platform_metrics = {
        "Время от идеи до модели": {"target": "< 1 неделя", "actual": "3 дня",
                                    "status": "GREEN"},
        "Время от модели до production": {"target": "< 1 день", "actual": "4 часа",
                                          "status": "GREEN"},
        "Uptime prediction API": {"target": "> 99.9%", "actual": "99.95%",
                                  "status": "GREEN"},
        "Latency p99 prediction": {"target": "< 100ms", "actual": "45ms",
                                   "status": "GREEN"},
        "Throughput (req/sec)": {"target": "> 1000", "actual": "2500",
                                 "status": "GREEN"},
        "Стоимость за prediction": {"target": "< $0.001", "actual": "$0.0005",
                                    "status": "GREEN"},
        "Время обучения модели": {"target": "< 2 часа", "actual": "45 мин",
                                  "status": "GREEN"},
        "Покрытие тестами": {"target": "> 80%", "actual": "72%",
                             "status": "YELLOW"},
    }

    print(f"  {'Метрика':<35} {'Цель':<18} {'Факт':<18} {'Статус'}")
    print("  " + "-" * 90)
    for metric, values in platform_metrics.items():
        status_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[values["status"]]
        print(f"  {metric:<35} {values['target']:<18} {values['actual']:<18} "
              f"{status_icon} {values['status']}")

    # --- 1.4 Диаграмма потока данных ---
    print("\n--- 1.4 Диаграмма потока данных (ASCII) ---")

    dataflow = """
  ┌─────────────────────────────────────────────────────────┐
  │                    CONSUMERS                            │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
  │  │ Продукт  │  │Аналитик │  │  Mobile  │              │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
  │       │              │              │                    │
  │       └──────────────┼──────────────┘                    │
  │                      ▼                                   │
  │              ┌──────────────┐                            │
  │              │Prediction API│ ← Feature Store            │
  │              └──────┬───────┘                            │
  │                     │                                    │
  │                     ▼                                    │
  │              ┌──────────────┐     ┌──────────────┐       │
  │              │Model Serving │────→│  Monitoring  │       │
  │              └──────────────┘     └──────┬───────┘       │
  │                                          │ drift alert   │
  │                     ┌────────────────────┘               │
  │                     ▼                                    │
  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐       │
  │  │Feature   │← │  Training    │→ │Model Registry│       │
  │  │Store     │  │  Service     │  └──────────────┘       │
  │  └────┬─────┘  └──────────────┘                         │
  │       │                                                  │
  │       ▼                                                  │
  │  ┌──────────────────────────────┐                        │
  │  │      DATA PLATFORM           │                        │
  │  │  Data Lake + Stream + Batch  │                        │
  │  └──────────────────────────────┘                        │
  └─────────────────────────────────────────────────────────┘"""
    print(dataflow)

    print("\n  === Итог Demo 1 ===")
    print("  ML-платформа = 4 слоя: инфра → данные → ML-сервисы → API")
    print("  Каждый слой имеет чёткие интеграционные точки")


# ============================================================
# Демо 2: Ландшафт инструментов — orchestration, tracking, serving
# ============================================================
def demo_tooling_landscape():
    print("\n" + "=" * 70)
    print("ДЕМО 2: Ландшафт инструментов — orchestration, tracking, serving")
    print("=" * 70)

    # --- 2.1 Категории инструментов ---
    print("\n--- 2.1 Категории ML-тулзов ---")

    tool_categories = {
        "Data Versioning": {
            "tools": [
                {"name": "DVC", "pros": "Git-подобный, простой",
                 "cons": "Медленно на больших данных",
                 "best_for": "Малые-средние проекты"},
                {"name": "LakeFS", "pros": "Git-подобный для data lakes",
                 "cons": "Стоимость",
                 "best_for": "Большие data lakes"},
                {"name": "Delta Lake", "pros": "ACID-транзакции, time travel",
                 "cons": "Требует Spark",
                 "best_for": "Spark-пайплайны"},
            ],
        },
        "Experiment Tracking": {
            "tools": [
                {"name": "MLflow", "pros": "Open-source, гибкий",
                 "cons": "Мало встроенных визуализаций",
                 "best_for": "Универсальный"},
                {"name": "Weights & Biases", "pros": "Отличные визуализации, team features",
                 "cons": "Коммерческий (ест free tier)",
                 "best_for": "Команды, research"},
                {"name": "Neptune.ai", "pros": "Хороший UI, интеграции",
                 "cons": "Дорогой для больших команд",
                 "best_for": "Enterprise"},
            ],
        },
        "Orchestration": {
            "tools": [
                {"name": "Airflow", "pros": "Зрелый, множество операторов",
                 "cons": "Сложный в настройке",
                 "best_for": "Сложные пайплайны"},
                {"name": "Prefect", "pros": "Современный, Python-native",
                 "cons": "Меньше сообщество",
                 "best_for": "Новые проекты"},
                {"name": "Kubeflow Pipelines", "pros": "Интеграция с K8s",
                 "cons": "Сложная настройка",
                 "best_for": "Kubernetes-среда"},
                {"name": "Dagster", "pros": "Software-defined assets",
                 "cons": "Относительно новый",
                 "best_for": "Data-centric пайплайны"},
            ],
        },
        "Model Serving": {
            "tools": [
                {"name": "Seldon Core", "pros": "ML-specific, A/B, canary",
                 "cons": "Сложный в управлении",
                 "best_for": "Сложные ML-сценарии"},
                {"name": "Triton (NVIDIA)", "pros": "GPU-оптимизированный",
                 "cons": "Привязка к NVIDIA",
                 "best_for": "GPU-инференс"},
                {"name": "BentoML", "pros": "Простой, Python-first",
                 "cons": "Менее масштабируемый",
                 "best_for": "Прототипы и малые сервисы"},
                {"name": "vLLM", "pros": "LLM-оптимизированный",
                 "cons": "Только для LLM",
                 "best_for": "LLM serving"},
            ],
        },
        "Monitoring": {
            "tools": [
                {"name": "Evidently", "pros": "ML-specific, beautiful reports",
                 "cons": "Только data/model quality",
                 "best_for": "Data drift мониторинг"},
                {"name": "Prometheus + Grafana", "pros": "Универсальный, зрелый",
                 "cons": "Нужна настройка",
                 "best_for": "Infrastructure metrics"},
                {"name": "Whylabs", "pros": "Облачный, автоматический",
                 "cons": "Коммерческий",
                 "best_for": "Managed мониторинг"},
            ],
        },
    }

    for category, info in tool_categories.items():
        print(f"\n  ┌── {category} ──┐")
        for tool in info["tools"]:
            print(f"  │  🔧 {tool['name']}")
            print(f"  │     + {tool['pros']}")
            print(f"  │     - {tool['cons']}")
            print(f"  │     → Лучше для: {tool['best_for']}")
        print(f"  └{'─' * 50}┘")

    # --- 2.2 Сравнение: выбор для разных сценариев ---
    print("\n--- 2.2 Рекомендации по выбору стека ---")

    scenarios = [
        {"scenario": "Стартап, 2-3 инженера",
         "stack": "MLflow + Airflow + BentoML + Evidently",
         "reasoning": "Простой стек, быстрый старт, открытый код"},
        {"scenario": "Команда 5-10, Kubernetes",
         "stack": "W&B + Kubeflow + Seldon + Prometheus/Grafana",
         "reasoning": "Зрелые инструменты, масштабируемость"},
        {"scenario": "Enterprise, compliance",
         "stack": "Neptune + Airflow + Triton + Whylabs",
         "reasoning": "Коммерческая поддержка, audit trail, security"},
        {"scenario": "LLM-проект",
         "stack": "W&B + vLLM + custom monitoring",
         "reasoning": "GPU-оптимизация, LLM-специфичные метрики"},
    ]

    for s in scenarios:
        print(f"\n  Сценарий: {s['scenario']}")
        print(f"  Рекомендованный стек: {s['stack']}")
        print(f"  Обоснование: {s['reasoning']}")

    # --- 2.3 Совместимость инструментов (матрица) ---
    print("\n--- 2.3 Совместимость инструментов ---")

    compatibility = {
        ("MLflow", "Airflow"):     "✅ Отлично — Airflow запускает MLflow runs",
        ("MLflow", "Seldon"):      "✅ Хорошо — MLflow registry → Seldon deploy",
        ("MLflow", "Kubeflow"):    "⚠️  Дублирование — оба track и orchestrate",
        ("W&B", "Kubeflow"):       "✅ Хорошо — W&B tracking + KFP orchestration",
        ("W&B", "Seldon"):         "✅ Хорошо — W&B artifact → Seldon model",
        ("Evidently", "Prometheus"): "✅ Отлично — Evidently → Prometheus metrics",
        ("DVC", "MLflow"):         "✅ Хорошо — DVC для данных, MLflow для моделей",
        ("Airflow", "Kubeflow"):   "⚠️  Выбрать одно — обе orchestrate",
    }

    print(f"  {'Инструмент 1':<15} {'Инструмент 2':<15} {'Совместимость'}")
    print("  " + "-" * 65)
    for (t1, t2), status in compatibility.items():
        print(f"  {t1:<15} {t2:<15} {status}")

    # --- 2.4 Примерная стоимость стека ---
    print("\n--- 2.4 Примерная стоимость стека (облако, команда 5 чел.) ---")

    cost_breakdown = {
        "Kubernetes кластер (3 node, GPU)": 2500,
        "S3 хранилище (1 TB)": 23,
        "MLflow (self-hosted, ECS)": 200,
        "Airflow (MWAA)": 500,
        "Prometheus + Grafana (self-hosted)": 100,
        "Мониторинг (Evidently Cloud)": 300,
        "W&B (Team plan, 5 seats)": 375,
        "Логи (CloudWatch)": 100,
        "Сеть и трафик": 50,
    }

    total = 0
    print(f"  {'Компонент':<45} {'Стоимость/мес':>12}")
    print("  " + "-" * 60)
    for component, cost in cost_breakdown.items():
        total += cost
        print(f"  {component:<45} ${cost:>10,}")
    print("  " + "-" * 60)
    print(f"  {'ИТОГО':<45} ${total:>10,}")

    # Стоимость за модель в месяц
    models_served = 10
    cost_per_model = total / models_served
    print(f"\n  Стоимость за модель: ${cost_per_model:,.0f}/мес "
          f"(при {models_served} моделях в production)")

    print("\n  === Итог Demo 2 ===")
    print("  Ландшафт ML-тулзов огромный — выбирайте по сценарию")
    print("  Стоимость = инфра + инструменты + время на настройку")


# ============================================================
# Демо 3: Внедрение платформы — миграция, обучение, документация
# ============================================================
def demo_platform_adoption():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Внедрение платформы — миграция, обучение, документация")
    print("=" * 70)

    # --- 3.1 Стратегия миграции ---
    print("\n--- 3.1 Стратегия миграции на ML-платформу ---")

    migration_phases = [
        {
            "phase": "Pilot",
            "duration": "4 недели",
            "goal": "Доказать концепцию на 1 проекте",
            "activities": [
                "Выбрать 1 проект с high visibility",
                "Настроить минимальный стек (tracking + serving)",
                "Задокументировать все шаги",
                "Собрать feedback от команды",
            ],
            "success_criteria": [
                "Модель задеплоена через платформу",
                "Время деплоя < 1 дня (vs 1 неделя раньше)",
                "Команда положительно откликается",
            ],
            "risks": [
                "Команда сопротивляется переменам",
                "Платформа не покрывает edge cases",
            ],
        },
        {
            "phase": "Early Adoption",
            "duration": "8 недель",
            "goal": "3-5 проектов используют платформу",
            "activities": [
                "Мигрировать 3-5 существующих моделей",
                "Добавить monitoring и alerting",
                "Провести обучение для 2-3 команд",
                "Создать internal docs и runbooks",
            ],
            "success_criteria": [
                "5+ моделей в production через платформу",
                "Monitoring покрывает все критические модели",
                "Среднее время деплоя < 4 часов",
            ],
            "risks": [
                "Проблемы с масштабированием",
                "Слишком много customization requests",
            ],
        },
        {
            "phase": "Broad Adoption",
            "duration": "12 недель",
            "goal": "Все ML-проекты используют платформу",
            "activities": [
                "Миграция всех оставшихся моделей",
                "Автоматизация onboarding новых проектов",
                "Внутренний training program",
                "Оптимизация стоимости",
            ],
            "success_criteria": [
                "90%+ моделей через платформу",
                "Время онбординга нового проекта < 1 дня",
                "ROI > 200%",
            ],
            "risks": [
                "Legacy системы не мигрируются",
                "Команды обходят платформу",
            ],
        },
    ]

    for phase_info in migration_phases:
        print(f"\n  ┌── Фаза: {phase_info['phase']} ({phase_info['duration']}) ──┐")
        print(f"  │ Цель: {phase_info['goal']}")
        print(f"  │")
        print(f"  │ Активности:")
        for act in phase_info["activities"]:
            print(f"  │   • {act}")
        print(f"  │")
        print(f"  │ Критерии успеха:")
        for crit in phase_info["success_criteria"]:
            print(f"  │   ✓ {crit}")
        print(f"  │")
        print(f"  │ Риски:")
        for risk in phase_info["risks"]:
            print(f"  │   ⚠ {risk}")
        print(f"  └{'─' * 55}┘")

    # --- 3.2 Программа обучения ---
    print("\n--- 3.2 Программа обучения команды ---")

    training_program = [
        {"module": "Основы платформы",
         "audience": "Все",
         "duration": "2 часа",
         "topics": ["Обзор архитектуры", "Как запустить训练", "Как задеплоить модель"],
         "format": "Workshop + Hands-on"},
        {"module": "Experiment Tracking",
         "audience": "DS + MLE",
         "duration": "3 часа",
         "topics": ["MLflow/W&B basics", "Logging параметров", "Сравнение экспериментов"],
         "format": "Workshop"},
        {"module": "Model Serving",
         "audience": "MLE + PE",
         "duration": "4 часа",
         "topics": ["Docker + K8s basics", "Деплой модели", "A/B тестирование"],
         "format": "Hands-on lab"},
        {"module": "Monitoring",
         "audience": "PE + MLE",
         "duration": "3 часа",
         "topics": ["Data drift", "Model performance", "Alerting"],
         "format": "Workshop + Lab"},
        {"module": "Incident Response",
         "audience": "Все",
         "duration": "2 часа",
         "topics": ["Runbooks", "Post-mortem", "Escalation"],
         "format": "Tabletop exercise"},
    ]

    print(f"  {'Модуль':<25} {'Аудитория':<15} {'Длит.':<8} {'Формат'}")
    print("  " + "-" * 70)
    for module in training_program:
        print(f"  {module['module']:<25} {module['audience']:<15} "
              f"{module['duration']:<8} {module['format']}")

    total_hours = sum(int(m["duration"].split()[0]) for m in training_program)
    print(f"\n  Общая длительность: {total_hours} часов")

    # --- 3.3 Checklist онбординга нового проекта ---
    print("\n--- 3.3 Checklist онбординга нового ML-проекта ---")

    onboarding_checklist = [
        {"step": 1, "task": "Создать проект в MLflow",
         "owner": "ML Engineer", "time": "15 мин",
         "status": "DONE"},
        {"step": 2, "task": "Настроить DVC репозиторий",
         "owner": "Data Engineer", "time": "30 мин",
         "status": "DONE"},
        {"step": 3, "task": "Определить входные данные и их схему",
         "owner": "Data Scientist", "time": "1 час",
         "status": "DONE"},
        {"step": 4, "task": "Создать training pipeline (Kubeflow/Airflow)",
         "owner": "ML Engineer", "time": "4 часа",
         "status": "TODO"},
        {"step": 5, "task": "Настроить Feature Store",
         "owner": "Data Engineer", "time": "2 часа",
         "status": "TODO"},
        {"step": 6, "task": "Написать model card",
         "owner": "Data Scientist", "time": "1 час",
         "status": "TODO"},
        {"step": 7, "task": "Настроить CI/CD (training + deploy)",
         "owner": "Platform Engineer", "time": "4 часа",
         "status": "TODO"},
        {"step": 8, "task": "Настроить monitoring (Grafana dashboard)",
         "owner": "Platform Engineer", "time": "2 часа",
         "status": "TODO"},
        {"step": 9, "task": "Написать runbook для инцидентов",
         "owner": "ML Engineer", "time": "1 час",
         "status": "TODO"},
        {"step": 10, "task": "Провести demo для stakeholders",
         "owner": "Data Scientist", "time": "30 мин",
         "status": "TODO"},
    ]

    completed = sum(1 for s in onboarding_checklist if s["status"] == "DONE")
    total_steps = len(onboarding_checklist)
    progress = completed / total_steps * 100

    print(f"  Прогресс: {completed}/{total_steps} ({progress:.0f}%)")
    print(f"  [{'█' * int(progress / 5)}{'░' * (20 - int(progress / 5))}]")

    for step in onboarding_checklist:
        status_icon = "✅" if step["status"] == "DONE" else "⬜"
        print(f"  {status_icon} {step['step']:>2}. {step['task']}")
        print(f"       Владелец: {step['owner']} | Время: {step['time']}")

    # --- 3.4 Метрики успешности внедрения ---
    print("\n--- 3.4 Метрики успешности внедрения ---")

    adoption_metrics = [
        {"metric": "Проектов на платформе", "before": "0/15",
         "after": "12/15", "target": "90%"},
        {"metric": "Среднее время деплоя", "before": "7 дней",
         "after": "4 часа", "target": "< 1 день"},
        {"metric": "Экспериментов задокументировано", "before": "20%",
         "after": "85%", "target": "80%"},
        {"metric": "Воспроизводимость результатов", "before": "30%",
         "after": "95%", "target": "90%"},
        {"metric": "Время онбординга (новый проект)", "before": "2 недели",
         "after": "1 день", "target": "< 2 дня"},
        {"metric": "Инцидентов (модель в production)", "before": "5/мес",
         "after": "1/мес", "target": "< 2/мес"},
    ]

    print(f"  {'Метрика':<35} {'До':<12} {'После':<12} {'Цель':<12}")
    print("  " + "-" * 75)
    for m in adoption_metrics:
        print(f"  {m['metric']:<35} {m['before']:<12} {m['after']:<12} "
              f"{m['target']:<12}")

    print("\n  === Итог Demo 3 ===")
    print("  Внедрение = Pilot → Early Adoption → Broad Adoption")
    print("  Успех = обучение + документация + метрики")


# ============================================================
# Демо 4: Эволюция платформы — feedback loops, масштабирование
# ============================================================
def demo_platform_evolution():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Эволюция платформы — feedback loops, масштабирование")
    print("=" * 70)

    # --- 4.1 Цикл обратной связи ---
    print("\n--- 4.1 Цикл обратной связи (Feedback Loop) ---")

    feedback_sources = {
        "Пользователи платформы": {
            "method": "Опросы, интервью",
            "frequency": "Ежеквартально",
            "data": [
                "Что работает хорошо?",
                "Что вызывает проблемы?",
                "Какие фичи нужны?",
                "Какой NPS у платформы?",
            ],
            "nps_score": 42,
            "response_rate": "65%",
        },
        "Метрики использования": {
            "method": "Автоматический сбор",
            "frequency": "Непрерывно",
            "data": [
                "Количество запусков обучения",
                "Количество деплоев",
                "Среднее время деплоя",
                "Количество инцидентов",
            ],
            "current": {
                "training_runs_per_week": 120,
                "deploys_per_week": 35,
                "avg_deploy_time_hours": 4,
                "incidents_per_month": 1,
            },
        },
        "Диагностика платформы": {
            "method": "Мониторинг инфры",
            "frequency": "Непрерывно",
            "data": [
                "Uptime компонентов",
                "Latency API",
                "Стоимость",
                "Использование ресурсов",
            ],
        },
    }

    for source, info in feedback_sources.items():
        print(f"\n  📊 {source}")
        print(f"    Метод: {info['method']}")
        print(f"    Частота: {info['frequency']}")
        print(f"    Типы данных:")
        for item in info["data"]:
            print(f"      • {item}")

    # --- 4.2 Приоритизация улучшений ---
    print("\n--- 4.2 Приоритизация улучшений (RICE) ---")

    improvements = [
        {"name": "Автоматический ретрейнинг по дрифту",
         "reach": 8, "impact": 9, "confidence": 0.8, "effort": 5,
         "votes": 12},
        {"name": "Встроенный feature store (Feast)",
         "reach": 7, "impact": 7, "confidence": 0.9, "effort": 8,
         "votes": 8},
        {"name": "Self-service UI для DS",
         "reach": 9, "impact": 5, "confidence": 0.7, "effort": 10,
         "votes": 15},
        {"name": "GPU-кластер для LLM",
         "reach": 4, "impact": 8, "confidence": 0.6, "effort": 7,
         "votes": 5},
        {"name": "Cost dashboard",
         "reach": 6, "impact": 4, "confidence": 0.9, "effort": 3,
         "votes": 7},
    ]

    # RICE = (Reach × Impact × Confidence) / Effort
    for imp in improvements:
        rice = (imp["reach"] * imp["impact"] * imp["confidence"]) / imp["effort"]
        imp["rice_score"] = rice

    # Сортируем по RICE
    improvements.sort(key=lambda x: x["rice_score"], reverse=True)

    print(f"  {'Улучшение':<35} {'R':>3} {'I':>3} {'C':>4} {'E':>3} "
          f"{'RICE':>6} {'Голоса':>6}")
    print("  " + "-" * 70)
    for imp in improvements:
        print(f"  {imp['name']:<35} {imp['reach']:>3} {imp['impact']:>3} "
              f"{imp['confidence']:>4.1f} {imp['effort']:>3} "
              f"{imp['rice_score']:>6.1f} {imp['votes']:>6}")

    # --- 4.3 Roadmap развития платформы ---
    print("\n--- 4.3 Roadmap развития (6 месяцев) ---")

    roadmap_quarters = {
        "Q1 2025": {
            "theme": "Stability & Basics",
            "items": [
                {"feature": "Стабилизация serving infrastructure",
                 "status": "done", "impact": "high"},
                {"feature": "Monitoring: data drift + model performance",
                 "status": "done", "impact": "high"},
                {"feature": "Документация и runbooks",
                 "status": "done", "impact": "medium"},
                {"feature": "CI/CD для training pipelines",
                 "status": "in_progress", "impact": "high"},
            ],
        },
        "Q2 2025": {
            "theme": "Automation & Self-Service",
            "items": [
                {"feature": "Feature store (Feast) интеграция",
                 "status": "planned", "impact": "high"},
                {"feature": "Self-service UI для DS",
                 "status": "planned", "impact": "medium"},
                {"feature": "Автоматический ретрейнинг по дрифту",
                 "status": "planned", "impact": "high"},
                {"feature": "Cost optimization dashboard",
                 "status": "planned", "impact": "medium"},
            ],
        },
        "Q3 2025": {
            "theme": "Scale & Advanced Features",
            "items": [
                {"feature": "GPU-кластер для LLM",
                 "status": "planned", "impact": "high"},
                {"feature": "Multi-model serving",
                 "status": "planned", "impact": "medium"},
                {"feature": "Advanced A/B testing framework",
                 "status": "planned", "impact": "medium"},
                {"feature": "Cost anomaly detection",
                 "status": "planned", "impact": "low"},
            ],
        },
    }

    for quarter, info in roadmap_quarters.items():
        print(f"\n  ┌── {quarter}: {info['theme']} ──┐")
        for item in info["items"]:
            status_icon = {"done": "✅", "in_progress": "🔄",
                           "planned": "📋"}[item["status"]]
            impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[item["impact"]]
            print(f"  │ {status_icon} {item['feature']:<45} {impact_icon}")
        print(f"  └{'─' * 55}┘")

    # --- 4.4 Метрики роста платформы ---
    print("\n--- 4.4 Метрики роста платформы (по месяцам) ---")

    monthly_growth = [
        {"month": "Июль 2024",  "users": 5,  "models": 2,  "deploys": 8,
         "uptime": 99.0, "cost": 1500},
        {"month": "Август 2024", "users": 8,  "models": 4,  "deploys": 15,
         "uptime": 99.5, "cost": 2000},
        {"month": "Сентябрь 2024", "users": 12, "models": 7, "deploys": 25,
         "uptime": 99.8, "cost": 2500},
        {"month": "Октябрь 2024", "users": 15, "models": 10, "deploys": 35,
         "uptime": 99.9, "cost": 2800},
        {"month": "Ноябрь 2024", "users": 18, "models": 12, "deploys": 42,
         "uptime": 99.95, "cost": 3000},
        {"month": "Декабрь 2024", "users": 22, "models": 14, "deploys": 50,
         "uptime": 99.95, "cost": 3100},
    ]

    print(f"  {'Месяц':<15} {'Пользователи':>12} {'Модели':>8} {'Деплои':>8} "
          f"{'Uptime':>8} {'Стоимость':>10}")
    print("  " + "-" * 65)
    for m in monthly_growth:
        print(f"  {m['month']:<15} {m['users']:>12} {m['models']:>8} "
              f"{m['deploys']:>8} {m['uptime']:>7.1f}% ${m['cost']:>8,}")

    # Тренды
    user_growth = (monthly_growth[-1]["users"] - monthly_growth[0]["users"]) / \
                  monthly_growth[0]["users"] * 100
    model_growth = (monthly_growth[-1]["models"] - monthly_growth[0]["models"]) / \
                   monthly_growth[0]["models"] * 100
    cost_per_user = monthly_growth[-1]["cost"] / monthly_growth[-1]["users"]
    cost_per_model = monthly_growth[-1]["cost"] / monthly_growth[-1]["models"]

    print(f"\n  Тренды:")
    print(f"    Рост пользователей: +{user_growth:.0f}% за 6 мес.")
    print(f"    Рост моделей: +{model_growth:.0f}% за 6 мес.")
    print(f"    Стоимость на пользователя: ${cost_per_user:,.0f}/мес")
    print(f"    Стоимость на модель: ${cost_per_model:,.0f}/мес")

    # Кривая зрелости
    print(f"\n  Зрелость платформы:")
    maturity = 2 + (monthly_growth[-1]["uptime"] - 99.0) / 0.5
    maturity = min(maturity, 4.0)
    print(f"    Текущий уровень: {maturity:.1f} / 4.0")
    bar_filled = int(maturity / 4 * 20)
    bar_empty = 20 - bar_filled
    print(f"    [{'█' * bar_filled}{'░' * bar_empty}]")

    print("\n  === Итог Demo 4 ===")
    print("  Платформа эволюционирует через feedback loops")
    print("  Каждый месяц: метрики → приоритизация → улучшение")


# ============================================================
# Точка входа
# ============================================================
if __name__ == "__main__":
    demo_platform_architecture()
    demo_tooling_landscape()
    demo_platform_adoption()
    demo_platform_evolution()
