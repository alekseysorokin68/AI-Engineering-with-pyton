"""
214 — Team & Process: agile для ML, зрелость MLOps

Темы:
  1. Agile for ML (sprints, backlog, experiments as deliverables)
  2. MLOps Maturity (levels 0-4, manual→auto, tooling progression)
  3. Team Roles (ML engineer, data scientist, platform engineer, domain expert)
  4. Knowledge Management (documentation, model cards, runbooks)

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
# Демо 1: Agile для ML — спринты, бэклог, эксперименты как деливераблы
# ============================================================
def demo_agile_ml():
    print("=" * 70)
    print("ДЕМО 1: Agile для ML — спринты, бэклог, эксперименты как деливераблы")
    print("=" * 70)

    # --- 1.1 Формирование спринта из ML-задач ---
    print("\n--- 1.1 Формирование ML-спринта ---")

    # Бэклог проекта: задачи с приоритетами и оценкой
    backlog = [
        {"id": "ML-001", "title": "Сбор данных о транзакциях",
         "type": "data", "priority": 1, "story_points": 8,
         "assignee": "data_engineer", "status": "done"},
        {"id": "ML-002", "title": "Feature engineering для скоринга",
         "type": "feature", "priority": 2, "story_points": 13,
         "assignee": "data_scientist", "status": "done"},
        {"id": "ML-003", "title": "Baseline модель (логистическая регрессия)",
         "type": "model", "priority": 3, "story_points": 8,
         "assignee": "ml_engineer", "status": "done"},
        {"id": "ML-004", "title": "Эксперимент: XGBoost с тюнингом",
         "type": "experiment", "priority": 4, "story_points": 13,
         "assignee": "ml_engineer", "status": "in_progress"},
        {"id": "ML-005", "title": "Эксперимент: нейросеть (MLP)",
         "type": "experiment", "priority": 5, "story_points": 21,
         "assignee": "data_scientist", "status": "in_progress"},
        {"id": "ML-006", "title": "A/B тестирование моделей",
         "type": "experiment", "priority": 6, "story_points": 8,
         "assignee": "ml_engineer", "status": "todo"},
        {"id": "ML-007", "title": "Деплой в staging",
         "type": "deploy", "priority": 7, "story_points": 5,
         "assignee": "platform_engineer", "status": "todo"},
        {"id": "ML-008", "title": "Мониторинг дрифта",
         "type": "monitor", "priority": 8, "story_points": 8,
         "assignee": "platform_engineer", "status": "todo"},
    ]

    # Формируем спринт容量: 40 story points
    SPRINT_CAPACITY = 40

    def plan_sprint(backlog, capacity):
        """Планирует спринт: берём задачи по приоритету."""
        sprint_items = []
        remaining = capacity

        # Сортируем по приоритету, берём только todo/in_progress
        candidates = [t for t in backlog
                      if t["status"] in ("todo", "in_progress")]
        candidates.sort(key=lambda t: t["priority"])

        for task in candidates:
            if task["story_points"] <= remaining:
                sprint_items.append(task)
                remaining -= task["story_points"]

        return sprint_items, remaining

    sprint, leftover = plan_sprint(backlog, SPRINT_CAPACITY)
    total_points = sum(t["story_points"] for t in sprint)

    print(f"  Ёмкость спринта: {SPRINT_CAPACITY} story points")
    print(f"  Набрано: {total_points} story points")
    print(f"  Остаток: {leftover} story points\n")

    print("  Содержимое спринта:")
    for t in sprint:
        type_icon = {"model": "🤖", "experiment": "🔬",
                     "deploy": "🚀", "monitor": "📊",
                     "data": "📦", "feature": "⚙️"}.get(t["type"], "📋")
        print(f"    {type_icon} {t['id']}: {t['title']}")
        print(f"       Тип: {t['type']} | Оценка: {t['story_points']} SP | "
              f"Ответственный: {t['assignee']}")

    # --- 1.2 Эксперимент как деливерабл ---
    print("\n--- 1.2 Эксперимент как деливерабл ---")

    # В ML ".done" означает не просто "написал код", а "получил результат"
    experiments = [
        {
            "id": "EXP-001",
            "task_id": "ML-003",
            "hypothesis": "Логистическая регрессия能达到 F1 > 0.75",
            "method": "LogisticRegression(C=1.0, max_iter=1000)",
            "features": ["age", "income", "credit_score", "debt_ratio"],
            "results": {"f1": 0.78, "precision": 0.81, "recall": 0.75, "auc": 0.85},
            "status": "completed",
            "conclusion": "Baseline установлен, F1=0.78 > порог 0.75",
            "artifacts": ["model_v1.pkl", "metrics_v1.json", "confusion_matrix.png"],
        },
        {
            "id": "EXP-002",
            "task_id": "ML-004",
            "hypothesis": "XGBoost с тюнингом превзойдёт baseline на 5%+",
            "method": "XGBClassifier(max_depth=6, n_estimators=200, lr=0.1)",
            "features": ["age", "income", "credit_score", "debt_ratio",
                         "loan_amount", "employment_years"],
            "results": {"f1": 0.85, "precision": 0.87, "recall": 0.83, "auc": 0.93},
            "status": "completed",
            "conclusion": "Гипотеза подтверждена: F1=0.85, +9% к baseline",
            "artifacts": ["xgb_model.pkl", "feature_importance.png",
                          "shap_values.json", "params.json"],
        },
        {
            "id": "EXP-003",
            "task_id": "ML-005",
            "hypothesis": "Нейросеть MLP достигнет F1 > 0.87",
            "method": "MLPClassifier(hidden=(128, 64), dropout=0.3)",
            "features": ["age", "income", "credit_score", "debt_ratio",
                         "loan_amount", "employment_years", "num_accounts"],
            "results": {"f1": 0.82, "precision": 0.84, "recall": 0.80, "auc": 0.90},
            "status": "completed",
            "conclusion": "Гипотеза ОПРОВЕРГНУТА: F1=0.82 < 0.87, MLP хуже XGBoost",
            "artifacts": ["mlp_model.pkl", "training_curves.png", "loss_log.csv"],
        },
    ]

    print("  Журнал экспериментов:")
    for exp in experiments:
        status_icon = "✅" if exp["status"] == "completed" else "🔄"
        print(f"\n    {status_icon} {exp['id']}: {exp['hypothesis']}")
        print(f"       Метод: {exp['method']}")
        print(f"       Результат: F1={exp['results']['f1']:.2f} | "
              f"AUC={exp['results']['auc']:.2f}")
        print(f"       Вывод: {exp['conclusion']}")
        print(f"       Артефакты: {', '.join(exp['artifacts'])}")

    # --- 1.3 Retrospective спринта ---
    print("\n--- 1.3 Retrospective спринта ---")

    retrospective = {
        "sprint_number": 12,
        "went_well": [
            "Эксперименты задокументированы с первого раза",
            "Baseline модель деплоена за 1 день",
            "Регулярные standup'ы помогли выявить блокер early",
        ],
        "to_improve": [
            "Feature engineering занял больше времени, чем планировалось",
            "MLP эксперимент был избыточным (нужно было проверить теорию)",
            "Нет автоматического сравнения метрик между экспериментами",
        ],
        "actions": [
            {"action": "Добавить автосравнение метрик в CI",
             "owner": "platform_engineer", "deadline": "спринт 13"},
            {"action": "Перед экспериментом проверять: есть ли основания?",
             "owner": "data_scientist", "deadline": "спринт 13"},
            {"action": "Оценка feature engineering: разбить на подзадачи",
             "owner": "scrum_master", "deadline": "спринт 13"},
        ],
    }

    print(f"  Спринт #{retrospective['sprint_number']}")
    print("\n  Что прошло хорошо:")
    for item in retrospective["went_well"]:
        print(f"    + {item}")
    print("\n  Что нужно улучшить:")
    for item in retrospective["to_improve"]:
        print(f"    - {item}")
    print("\n  План действий:")
    for act in retrospective["actions"]:
        print(f"    → {act['action']} ({act['owner']}, к {act['deadline']})")

    # --- 1.4 Velocity трекинг ---
    print("\n--- 1.4 Velocity трекинг — предсказание скорости команды ---")

    sprints = [
        {"sprint": 9,  "planned": 35, "completed": 30, "bugs": 2},
        {"sprint": 10, "planned": 40, "completed": 38, "bugs": 1},
        {"sprint": 11, "planned": 40, "completed": 35, "bugs": 3},
        {"sprint": 12, "planned": 40, "completed": 40, "bugs": 1},
    ]

    # Средняя velocity и тренд
    velocities = [s["completed"] for s in sprints]
    avg_velocity = sum(velocities) / len(velocities)
    trend = (velocities[-1] - velocities[0]) / len(velocities)

    print(f"  {'Спринт':>8} {'План':>6} {'Факт':>6} {'Баги':>5} {'Выполнение':>10}")
    print("  " + "-" * 40)
    for s in sprints:
        completion = s["completed"] / s["planned"] * 100
        print(f"  {s['sprint']:>8} {s['planned']:>6} {s['completed']:>6} "
              f"{s['bugs']:>5} {completion:>9.0f}%")

    print(f"\n  Средняя velocity: {avg_velocity:.1f} SP/спринт")
    print(f"  Тренд: {trend:+.1f} SP/спринт ({'↑ растёт' if trend > 0 else '↓ падает'})")
    print(f"  Рекомендация на спринт 13: {int(avg_velocity)} SP "
          f"(±{int(trend * 2)})")

    print("\n  === Итог Demo 1 ===")
    print("  Agile в ML: спринты = итерации, эксперименты = деливераблы")
    print("  Velocity помогает предсказывать и планировать")


# ============================================================
# Демо 2: Зрелость MLOps — уровни 0-4, переход от ручного к автоматическому
# ============================================================
def demo_mlops_maturity():
    print("\n" + "=" * 70)
    print("ДЕМО 2: Зрелость MLOps — уровни 0-4, ручное → автоматическое")
    print("=" * 70)

    # --- 2.1 Определение текущего уровня зрелости ---
    print("\n--- 2.1 Уровни зрелости MLOps ---")

    maturity_levels = {
        0: {
            "name": "Мануальный ML",
            "description": "Всё вручную, на ноутбуках",
            "training": "Jupyter notebook на ноутбуке",
            "deployment": "Скопировал файл модели",
            "monitoring": "Нет мониторинга",
            "retraining": "По запросу, вручную",
            "tools": ["Jupyter", "sklearn", "git"],
            "pain_points": ["Воспроизводимость", "Масштабирование", "Скорость"],
        },
        1: {
            "name": "Полуавтоматизация",
            "description": "Скрипты для обучения, но ручной деплой",
            "training": "Python-скрипты с параметрами",
            "deployment": "Дocker-контейнер, ручной запуск",
            "monitoring": "Базовые логи",
            "retraining": "Крон-задача раз в месяц",
            "tools": ["Docker", "MLflow", "Airflow", "GitHub"],
            "pain_points": ["Фрагментация", "Дублирование кода"],
        },
        2: {
            "name": "Автоматизация пайплайнов",
            "description": "CI/CD для ML, автоматический деплой",
            "training": "Пайплайны с параметрами",
            "deployment": "CI/CD → staging → production",
            "monitoring": "Метрики модели + данные",
            "retraining": "Автоматическое по расписанию",
            "tools": ["Kubeflow", "Seldon", "Prometheus", "Grafana"],
            "pain_points": ["Стоимость инфры", "Сложность настройки"],
        },
        3: {
            "name": "Проактивный мониторинг",
            "description": "Дрифт-детекция, авто-ретрейнинг",
            "training": "Гибкие пайплайны с A/B",
            "deployment": "Canary/Blue-green автоматически",
            "monitoring": "Дрифт-детекция + алерты",
            "retraining": "По триггеру (дрифт, расписание)",
            "tools": ["Seldon Alibi", "Evidently", "Kafka"],
            "pain_points": ["Стоимость", "Комплексность алертов"],
        },
        4: {
            "name": "Автономный ML",
            "description": "AutoML, самообслуживание, NL interfaces",
            "training": "AutoML + AutoFeature",
            "deployment": "One-click deploy, self-healing",
            "monitoring": "Предиктивный мониторинг",
            "retraining": "Автономное, с human-in-the-loop",
            "tools": ["Custom platform", "AutoML", "LLM interfaces"],
            "pain_points": ["Начальные инвестиции", "Культура"],
        },
    }

    for level, info in maturity_levels.items():
        print(f"\n  Уровень {level}: {info['name']}")
        print(f"    {info['description']}")
        print(f"    Обучение:    {info['training']}")
        print(f"    Деплой:      {info['deployment']}")
        print(f"    Мониторинг:  {info['monitoring']}")
        print(f"    Ретрейнинг:  {info['retraining']}")
        print(f"    Инструменты: {', '.join(info['tools'])}")
        print(f"    Проблемы:    {', '.join(info['pain_points'])}")

    # --- 2.2 Аудит текущего состояния команды ---
    print("\n--- 2.2 Аудит зрелости команды ---")

    team_capabilities = {
        "versioning_data":        1,  # 0=нет, 1=git, 2=DVC, 3=full lineage
        "versioning_models":      1,  # 0=нет, 1=pickle, 2=MLflow, 3=model registry
        "automated_training":     0,  # 0=ноутбук, 1=скрипт, 2=пайплайн, 3=AutoML
        "automated_deployment":   1,  # 0=копируем, 1=Docker, 2=CI/CD, 3=one-click
        "monitoring":             0,  # 0=нет, 1=логи, 2=метрики, 3=дрift
        "retraining":             0,  # 0=ручное, 1=по расписанию, 2=по триггеру, 3=auto
        "feature_store":          0,  # 0=нет, 1=таблица, 2=Feast, 3=online+offline
        "experiment_tracking":    1,  # 0=нет, 1=тетрадь, 2=MLflow, 3=полный
    }

    # Вычисляем общий уровень зрелости (среднее по всем аспектам)
    avg_maturity = sum(team_capabilities.values()) / len(team_capabilities)

    # Определяем общий уровень
    overall_level = math.floor(avg_maturity)

    print(f"  {'Аспект':<25} {'Уровень':>8} {'Описание':<30}")
    print("  " + "-" * 65)

    aspect_descriptions = {
        "versioning_data":      ["Нет", "Git", "DVC", "Полный lineage"],
        "versioning_models":    ["Нет", "Pickle файл", "MLflow", "Model Registry"],
        "automated_training":   ["Ноутбук", "Скрипт", "Пайплайн", "AutoML"],
        "automated_deployment": ["Копирование", "Docker", "CI/CD", "One-click"],
        "monitoring":           ["Нет", "Логи", "Метрики", "Дрифт-детекция"],
        "retraining":           ["Ручное", "По расписанию", "По триггеру", "Автономное"],
        "feature_store":        ["Нет", "Таблица", "Feast", "Online + Offline"],
        "experiment_tracking":  ["Нет", "Тетрадь", "MLflow", "Полный"],
    }

    for aspect, level in team_capabilities.items():
        desc = aspect_descriptions[aspect][level]
        bar = "█" * (level + 1) + "░" * (3 - level)
        print(f"  {aspect:<25} {bar} [{level}/3] {desc}")

    print(f"\n  Средний уровень зрелости: {avg_maturity:.2f} / 3.0")
    print(f"  Общий уровень MLOps: {overall_level} из 4")

    # Рекомендации по улучшению
    print("\n  Рекомендации по улучшению:")
    for aspect, level in team_capabilities.items():
        if level < 2:
            target = level_descriptions = aspect_descriptions[aspect]
            next_level = target[min(level + 1, 3)]
            print(f"    → {aspect}: следующий шаг = {next_level}")

    # --- 2.3 Roadmap перехода на следующий уровень ---
    print("\n--- 2.3 Roadmap перехода на уровень MLOps 2 ---")

    roadmap = [
        {"phase": 1, "weeks": "1-2", "task": "Настроить DVC для versioning данных",
         "effort": "8 SP", "owner": "data_engineer",
         "dependency": None},
        {"phase": 2, "weeks": "2-3", "task": "Внедрить MLflow для tracking экспериментов",
         "effort": "13 SP", "owner": "ml_engineer",
         "dependency": "DVC настроен"},
        {"phase": 3, "weeks": "3-4", "task": "Автоматизировать training pipeline (Airflow)",
         "effort": "21 SP", "owner": "platform_engineer",
         "dependency": "MLflow работает"},
        {"phase": 4, "weeks": "4-5", "task": "Настроить CI/CD для моделей (GitHub Actions)",
         "effort": "13 SP", "owner": "platform_engineer",
         "dependency": "Pipeline работает"},
        {"phase": 5, "weeks": "5-6", "task": "Базовый мониторинг (Grafana + Prometheus)",
         "effort": "8 SP", "owner": "platform_engineer",
         "dependency": "CI/CD настроен"},
        {"phase": 6, "weeks": "6-7", "task": "Документация и обучение команды",
         "effort": "5 SP", "owner": "scrum_master",
         "dependency": "Всё предыдущее"},
    ]

    print(f"  {'Фаза':>5} {'Недели':>7} {'Задача':<50} {'SP':>4} {'Dep':>5}")
    print("  " + "-" * 80)
    for item in roadmap:
        dep = item["dependency"][:10] if item["dependency"] else "—"
        print(f"  {item['phase']:>5} {item['weeks']:>7} {item['task']:<50} "
              f"{item['effort']:>4} {dep:>5}")

    total_sp = sum(int(item["effort"].split()[0]) for item in roadmap)
    print(f"\n  Общий объём: {total_sp} story points")
    print(f"  Длительность: {roadmap[-1]['weeks']} недель")

    # --- 2.4 Стоимость и ROI ---
    print("\n--- 2.4 Стоимость и ROI перехода ---")

    costs = {
        "Инфраструктура (облако)": 5000,
        "Лицензии (Databricks, etc.)": 8000,
        "Время команды (68 SP × ~2000 руб/SP)": 136000,
        "Обучение и документация": 15000,
    }

    benefits = {
        "Сокращение времени деплоя (7 дней → 1 день)": 40000,
        "Снижение количества багов (30% → 5%)": 30000,
        "Ускорение экспериментов (2 недели → 2 дня)": 25000,
        "Автоматический мониторинг (vs ручная проверка)": 20000,
    }

    total_cost = sum(costs.values())
    total_benefit = sum(benefits.values())
    roi = (total_benefit - total_cost) / total_cost * 100

    print("  Затраты:")
    for item, cost in costs.items():
        print(f"    {item}: {cost:,} руб.")
    print(f"    {'ИТОГО':>50}: {total_cost:,} руб.")

    print("\n  Выгоды:")
    for item, benefit in benefits.items():
        print(f"    {item}: {benefit:,} руб./год")
    print(f"    {'ИТОГО':>50}: {total_benefit:,} руб./год")

    print(f"\n  ROI: {roi:+.0f}% (окупаемость: {total_cost/total_benefit*12:.1f} мес.)")

    print("\n  === Итог Demo 2 ===")
    print("  MLOps зрелость = от ноутбука до автономного ML")
    print("  Каждый уровень снижает время от идеи до продакшена")


# ============================================================
# Демо 3: Роли в команде ML
# ============================================================
def demo_team_roles():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Роли в команде ML — ML engineer, data scientist, etc.")
    print("=" * 70)

    # --- 3.1 Определение ролей и их Responsibilities ---
    print("\n--- 3.1 Роли и их Responsibilities ---")

    roles = {
        "ML Engineer": {
            "short": "MLE",
            "focus": "Продуктивные ML-системы",
            "responsibilities": [
                "Дизайн и реализация ML-пайплайнов",
                "Оптимизация моделей для production",
                "Деплой и масштабирование моделей",
                "Мониторинг и техобслуживание",
            ],
            "skills": ["Python", "ML frameworks", "Docker", "K8s", "CI/CD"],
            "tools": ["PyTorch", "TensorFlow", "MLflow", "Kubeflow"],
            "typical_tasks": [
                "Реализовать inference pipeline",
                "Настроить autoscaling для модели",
                "Интегрировать модель с API",
            ],
        },
        "Data Scientist": {
            "short": "DS",
            "focus": "Исследования и моделирование",
            "responsibilities": [
                "Исследование данных (EDA)",
                "Проектирование признаков (feature engineering)",
                "Разработка и обучение моделей",
                "Анализ результатов и рекомендации",
            ],
            "skills": ["Python", "Статистика", "ML", "SQL", "Визуализация"],
            "tools": ["Jupyter", "sklearn", "XGBoost", "SHAP", "pandas"],
            "typical_tasks": [
                "Провести EDA нового датасета",
                "Сравнить 3 алгоритма на данных",
                "Подготовить отчёт с рекомендациями",
            ],
        },
        "Platform Engineer": {
            "short": "PE",
            "focus": "Инфраструктура для ML",
            "responsibilities": [
                "Настройка вычислительных кластеров",
                "Разработка внутренних платформ и тулзов",
                "Безопасность и compliance",
                "Оптимизация стоимости инфраструктуры",
            ],
            "skills": ["Kubernetes", "Terraform", "Cloud", "Networking", "Security"],
            "tools": ["K8s", "Terraform", "AWS/GCP", "Prometheus", "Grafana"],
            "typical_tasks": [
                "Настроить GPU-кластер для обучения",
                "Развернуть internal ML platform",
                "Оптимизировать стоимость облака",
            ],
        },
        "Domain Expert": {
            "short": "DE",
            "focus": "Экспертиза предметной области",
            "responsibilities": [
                "Определение бизнес-требований",
                "Валидация результатов модели",
                "Приоритизация задач",
                "Связь с бизнесом",
            ],
            "skills": ["Отраслевые знания", "Аналитика", "Коммуникация"],
            "tools": ["Excel", "Дашборды", "Отчёты"],
            "typical_tasks": [
                "Определить метрику успеха",
                "Проверить предсказания на исторических данных",
                "Объяснить бизнесу ограничения модели",
            ],
        },
    }

    for role_name, role_info in roles.items():
        print(f"\n  ╔══ {role_name} ({role_info['short']}) ══╗")
        print(f"  ║ Фокус: {role_info['focus']}")
        print(f"  ║")
        print(f"  ║ Responsibilities:")
        for resp in role_info["responsibilities"]:
            print(f"  ║   • {resp}")
        print(f"  ║")
        print(f"  ║ Навыки: {', '.join(role_info['skills'])}")
        print(f"  ║ Инструменты: {', '.join(role_info['tools'])}")
        print(f"  ║ Типичные задачи:")
        for task in role_info["typical_tasks"]:
            print(f"  ║   → {task}")
        print(f"  ╚{'═' * (len(role_name) + 15)}╝")

    # --- 3.2 Матрица взаимодействия ---
    print("\n--- 3.2 Матрица взаимодействия между ролями ---")

    interaction_matrix = {
        ("MLE", "DS"):    "Приём моделей от DS → адаптация для production",
        ("MLE", "PE"):    "Использование платформы → обратная связь о потребностях",
        ("DS", "DE"):     "Получение требований → представление результатов",
        ("DS", "PE"):     "Запрос вычислений → предоставление ресурсов",
        ("PE", "DE"):     "Настройка мониторинга → определение бизнес-метрик",
        ("MLE", "DE"):    "Демо системы → валидация результатов",
    }

    for (r1, r2), description in interaction_matrix.items():
        print(f"  {r1:>3} ←→ {r2:<3}: {description}")

    # --- 3.3 Распределение времени по ролям ---
    print("\n--- 3.3 Распределение времени в проекте (неделя) ---")

    weekly_time = {
        "ML Engineer":    {"код": 40, "встречи": 5, "документация": 5,
                          "исследование": 0, "инфраструктура": 10},
        "Data Scientist": {"код": 25, "встречи": 5, "документация": 5,
                          "исследование": 25, "инфраструктура": 0},
        "Platform Eng":   {"код": 30, "встречи": 5, "документация": 5,
                          "исследование": 0, "инфраструктура": 20},
        "Domain Expert":  {"код": 0, "встречи": 15, "документация": 10,
                          "исследование": 5, "инфраструктура": 0},
    }

    categories = ["код", "встречи", "документация", "исследование", "инфраструктура"]
    print(f"  {'Роль':<15}", end="")
    for cat in categories:
        print(f" {cat:>10}", end="")
    print(f" {'Итого':>7}")
    print("  " + "-" * 72)

    for role, times in weekly_time.items():
        total = sum(times.values())
        print(f"  {role:<15}", end="")
        for cat in categories:
            hours = times[cat]
            bar = "█" * (hours // 5) if hours > 0 else ""
            print(f" {hours:>3}ч {bar:<6}", end="")
        print(f" {total:>3}ч")

    # --- 3.4 RACI-матрица для ML-проекта ---
    print("\n--- 3.4 RACI-матрица (Responsible, Accountable, Consulted, Informed) ---")

    raci = [
        ("Определение бизнес-требований", "I", "I", "C", "R"),
        ("Сбор и очистка данных",         "C", "I", "R", "I"),
        ("Feature engineering",           "C", "R", "I", "I"),
        ("Обучение модели",               "C", "R", "I", "I"),
        ("Docker-контейнер модели",       "R", "I", "C", "I"),
        ("CI/CD пайплайн",               "C", "I", "R", "I"),
        ("Деплой в production",           "R", "I", "A", "I"),
        ("Мониторинг дрифта",             "R", "I", "A", "I"),
        ("Валидация бизнес-результатов",  "I", "C", "I", "R"),
        ("Решение о ретрейнинге",         "C", "A", "I", "R"),
    ]

    roles_short = ["MLE", "DS", "PE", "DE"]
    print(f"  {'Действие':<40} {'MLE':>4} {'DS':>4} {'PE':>4} {'DE':>4}")
    print("  " + "-" * 60)
    for action, *assignments in raci:
        print(f"  {action:<40}", end="")
        for a in assignments:
            print(f" {a:>4}", end="")
        print()

    print("  Легенда: R=Responsible (исполнитель), A=Accountable (отвечает), "
          "C=Consulted, I=Informed")

    print("\n  === Итог Demo 3 ===")
    print("  ML-команда = MLE + DS + PE + Domain Expert")
    print("  Каждая роль важна; без одной — система не работает")


# ============================================================
# Демо 4: Управление знаниями — документация, runbooks
# ============================================================
def demo_knowledge_management():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Управление знаниями — документация, runbooks")
    print("=" * 70)

    # --- 4.1 Структура ML-проекта ---
    print("\n--- 4.1 Стандартная структура ML-проекта ---")

    project_structure = {
        "ml-project/": {
            "README.md": "Обзор проекта,.quickstart, команды",
            "docs/": {
                "data_dictionary.md": "Описание всех полей данных",
                "model_card.md": "Карточка модели (см. Demo 4 в 213)",
                "architecture.md": "Диаграмма архитектуры",
                "api_spec.md": "Спецификация API (OpenAPI)",
            },
            "src/": {
                "data/": "Скрипты загрузки и очистки данных",
                "features/": "Feature engineering",
                "models/": "Обучение и инференс",
                "serving/": "API-сервер для модели",
                "monitoring/": "Мониторинг и алерты",
            },
            "configs/": {
                "training.yaml": "Параметры обучения",
                "serving.yaml": "Параметры инференса",
                "monitoring.yaml": "Пороги алертов",
            },
            "tests/": {
                "test_data.py": "Тесты данных (схема, распределения)",
                "test_model.py": "Тесты модели (latency, accuracy)",
                "test_api.py": "Интеграционные тесты API",
            },
            "notebooks/": "EDA и исследования (не для production!)",
            "experiments/": "Журнал экспериментов",
            "runbooks/": "Инструкции по инцидентам",
        }
    }

    def print_tree(tree, indent=""):
        for name, content in tree.items():
            if isinstance(content, dict):
                print(f"  {indent}📁 {name}")
                print_tree(content, indent + "  ")
            else:
                print(f"  {indent}📄 {name}: {content}")

    print_tree(project_structure)

    # --- 4.2 Runbook: что делать при инциденте ---
    print("\n--- 4.2 Runbook: Инцидент — модель деградировала ---")

    runbook = {
        "title": "Инцидент: Деградация модели в production",
        "severity_levels": {
            "P1": "Модель полностью не работает (latency > 5s или error rate > 10%)",
            "P2": "Метрики модели упали > 10% от baseline",
            "P3": "Метрики модели упали > 5% от baseline",
        },
        "steps": [
            {
                "step": 1,
                "action": "Определить масштаб проблемы",
                "commands": [
                    "Проверить Grafana dashboard: model_latency, error_rate",
                    "Посмотреть логи: kubectl logs -l app=model-server --tail=100",
                    "Проверить алерты в PagerDuty/Slack",
                ],
                "time_estimate": "5 мин",
            },
            {
                "step": 2,
                "action": "Проверить данные",
                "commands": [
                    "Сравнить распределение входных данных с training data",
                    "Проверить наличие data drift (Evidently report)",
                    "Проверить freshness данных (когда обновлялись?)",
                ],
                "time_estimate": "15 мин",
            },
            {
                "step": 3,
                "action": "Проверить модель",
                "commands": [
                    "Сравнить текущие метрики с baseline",
                    "Проверить, не изменились ли входные фичи",
                    "Запустить shadow scoring на тестовых данных",
                ],
                "time_estimate": "20 мин",
            },
            {
                "step": 4,
                "action": "Решение: откат или фикс",
                "commands": [
                    "Если P1: Откатить на предыдущую версию (kubectl rollout undo)",
                    "Если P2/P3: Ретрейнить модель с свежими данными",
                    "Уведомить stakeholders в Slack",
                ],
                "time_estimate": "30 мин",
            },
            {
                "step": 5,
                "action": "Post-mortem",
                "commands": [
                    "Заполнить шаблон post-mortem",
                    "Определить root cause",
                    "Добавить автоматическую проверку, чтобы предотвратить повторение",
                ],
                "time_estimate": "60 мин",
            },
        ],
    }

    print(f"  Runbook: {runbook['title']}")
    print("\n  Уровни серьёзности:")
    for level, desc in runbook["severity_levels"].items():
        print(f"    {level}: {desc}")

    print("\n  Пошаговая инструкция:")
    for step_info in runbook["steps"]:
        print(f"\n  Шаг {step_info['step']}: {step_info['action']} "
              f"(~{step_info['time_estimate']})")
        for cmd in step_info["commands"]:
            print(f"    $ {cmd}")

    # --- 4.3 Шаблон Post-Mortem ---
    print("\n--- 4.3 Шаблон Post-Mortem ---")

    post_mortem = {
        "incident_id": "INC-2024-0042",
        "date": "2024-12-15",
        "duration": "2 часа 15 минут",
        "severity": "P2",
        "summary": "Модель кредитного скоринга показала рост false positive на 15%",
        "timeline": [
            "14:00 — Алерт: accuracy упала с 0.91 до 0.82",
            "14:05 — Инженер начал investigation",
            "14:20 — Обнаружен дрифт в признаке income",
            "14:45 — Запущен ретрейнинг на свежих данных",
            "15:30 — Новая модель протестирована, accuracy=0.90",
            "16:00 — Деплой новой модели в production",
            "16:15 — Метрики восстановлены, алерт закрыт",
        ],
        "root_cause": "Внедрение нового источника данных income изменило распределение",
        "impact": "15% клиентов получили неверные скоринговые оценки",
        "action_items": [
            "Автоматический мониторинг data drift (due: спринт 13)",
            "Валидация нового источника перед подключением (due: спринт 12)",
            "Уведомление DS при изменении распределения > 5% (due: спринт 14)",
        ],
    }

    print(f"  Incident: {post_mortem['incident_id']}")
    print(f"  Дата: {post_mortem['date']} | Длительность: {post_mortem['duration']}")
    print(f"  Серьёзность: {post_mortem['severity']}")
    print(f"  Краткое описание: {post_mortem['summary']}")
    print(f"\n  Root cause: {post_mortem['root_cause']}")
    print(f"  Влияние: {post_mortem['impact']}")

    print("\n  Таймлайн:")
    for event in post_mortem["timeline"]:
        print(f"    {event}")

    print("\n  Action items:")
    for item in post_mortem["action_items"]:
        print(f"    ☐ {item}")

    # --- 4.4 Каталог знаний команды ---
    print("\n--- 4.4 Каталог знаний (Knowledge Base) ---")

    knowledge_base = [
        {"topic": "Как запустить обучение",
         "doc": "docs/run_training.md",
         "last_updated": "2024-12-01",
         "author": "ml_engineer",
         "tags": ["training", "quickstart"]},
        {"topic": "Как добавить новый признак",
         "doc": "docs/add_feature.md",
         "last_updated": "2024-11-20",
         "author": "data_scientist",
         "tags": ["features", "how-to"]},
        {"topic": "Как откатить модель",
         "doc": "runbooks/rollback_model.md",
         "last_updated": "2024-12-15",
         "author": "platform_engineer",
         "tags": ["incident", "rollback"]},
        {"topic": "Как настроить мониторинг",
         "doc": "docs/monitoring_setup.md",
         "last_updated": "2024-11-10",
         "author": "platform_engineer",
         "tags": ["monitoring", "grafana"]},
        {"topic": "Как провести A/B тест",
         "doc": "docs/ab_testing.md",
         "last_updated": "2024-10-25",
         "author": "data_scientist",
         "tags": ["experiment", "ab-test"]},
        {"topic": "Как обработать инцидент P1",
         "doc": "runbooks/incident_response.md",
         "last_updated": "2024-12-15",
         "author": "platform_engineer",
         "tags": ["incident", "runbook", "p1"]},
    ]

    # Поиск по тегам
    def search_knowledge(base, query):
        """Ищет статьи по ключевому слову в тегах и теме."""
        results = []
        query_lower = query.lower()
        for item in base:
            if (query_lower in item["topic"].lower() or
                any(query_lower in tag for tag in item["tags"])):
                results.append(item)
        return results

    print("  Каталог статей:")
    for item in knowledge_base:
        print(f"    📄 {item['topic']}")
        print(f"       Файл: {item['doc']} | Обновлено: {item['last_updated']} "
              f"| Автор: {item['author']}")
        print(f"       Теги: {', '.join(item['tags'])}")

    # Поиск
    search_terms = ["инцидент", "модель", "мониторинг"]
    for term in search_terms:
        results = search_knowledge(knowledge_base, term)
        titles = [r["topic"] for r in results]
        print(f"\n  Поиск '{term}': найдено {len(results)} — {', '.join(titles)}")

    print("\n  === Итог Demo 4 ===")
    print("  Управление знаниями: структура проекта + runbooks + post-mortem + каталог")
    print("  Хорошая документация = быстрое онбординг и fewer incidents")


# ============================================================
# Точка входа
# ============================================================
if __name__ == "__main__":
    demo_agile_ml()
    demo_mlops_maturity()
    demo_team_roles()
    demo_knowledge_management()
