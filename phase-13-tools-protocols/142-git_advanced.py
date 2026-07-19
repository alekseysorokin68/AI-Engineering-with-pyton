"""142 — Git Advanced: branching, merging, rebasing, workflows

Темы:
  1. Branching Strategies (feature branch, gitflow, trunk-based)
  2. Merging (fast-forward, 3-way merge, merge conflicts)
  3. Rebasing (interactive rebase, squash, clean history)
  4. Git Hooks & Automation (pre-commit, commit message conventions)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import os
import pathlib

random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Демон 1: Стратегии ветвления (feature branch, gitflow, trunk-based)
# ──────────────────────────────────────────────────────────────────────────────
def demo_branching_strategies():
    """Демонстрация стратегий ветвления в Git."""
    print("=" * 70)
    print("Демон 1: Стратегии ветвления (feature branch, gitflow, trunk-based)")
    print("=" * 70)

    # ── 1.1 Симуляция графа коммитов ──
    print("\n[1.1] Симуляция графа коммитов:")
    commits = []
    branches = {"main": [], "develop": [], "feature/login": [],
                "feature/api": [], "release/v1.0": []}

    commit_id = 1000
    messages = [
        ("main", "Инициализация проекта"),
        ("main", "Настройка CI/CD"),
        ("develop", "Создание ветки develop"),
        ("feature/login", "Ветка feature: авторизация"),
        ("feature/login", "Добавлен JWT-токен"),
        ("feature/login", "Форма логина"),
        ("feature/login", "Тесты авторизации"),
        ("develop", "Слияние feature/login"),
        ("feature/api", "Ветка feature: REST API"),
        ("feature/api", "CRUD операции"),
        ("feature/api", "Валидация входных данных"),
        ("release/v1.0", "Создание release ветки"),
        ("release/v1.0", "Исправления перед релизом"),
        ("main", "Слияние release/v1.0 → v1.0"),
        ("develop", "Обновление develop после релиза"),
    ]

    print("\n  Граф коммитов (упрощённая схема):")
    print("  main:     A ─── B ──────────────────────── M1 ────────")
    print("  develop:        └── C ──── M2 ────── M3 ──── N1 ────")
    print("  feature/login:        D ─── E ─── F ───┘")
    print("  feature/api:                                G ─── H ─── I ───┘")
    print("  release/v1.0:                                   J ─── K ───┘")
    print()

    for branch, msg in messages:
        commit_id += 1
        short_id = f"{commit_id:04x}"
        commits.append({"id": short_id, "branch": branch, "message": msg})
        print(f"  [{short_id}] ({branch:>16}) {msg}")

    # ── 1.2 Feature Branch Workflow ──
    print("\n[1.2] Feature Branch Workflow (простой):")
    print("  main ───── A ──── B ──── C ──── D ──── E ─────")
    print("                \\                    /")
    print("                 feature-x ─ F ── G")
    print()
    print("  Правила:")
    print("    1. Каждая фича — отдельная ветка от main")
    print("    2. Работа ведётся только в feature-ветке")
    print("    3. После ревью — слияние в main через PR/MR")
    print("    4. Feature-ветка удаляется после слияния")

    # ── 1.3 Gitflow Workflow ──
    print("\n[1.3] Gitflow Workflow:")
    print("  main:     A ──────── M1 ──────────── M2 ────────")
    print("  develop:      ──── C ──── D ──── M3 ──── N1 ────")
    print("  feature:          E ──── F ───┘    H ──── I ───┘")
    print("  release:                               J ──── K ───┘")
    print("  hotfix:                                              L ──┘")
    print()
    print("  Ветки:")
    gitflow_branches = [
        ("main", "Стабильные релизы (production)"),
        ("develop", "Интеграция фич (интеграционная ветка)"),
        ("feature/*", "Разработка новых возможностей"),
        ("release/*", "Подготовка к релизу (багфиксы, документация)"),
        ("hotfix/*", "Экспресс-исправления в production"),
    ]
    for name, purpose in gitflow_branches:
        print(f"    {name:<18} → {purpose}")

    # ── 1.4 Trunk-Based Development ──
    print("\n[1.4] Trunk-Based Development:")
    print("  main:     A ── B ── C ── D ── E ── F ── G ── H ──")
    print("               /     |     \\       /     |")
    print("              f1     f2     f3    f4     f5")
    print()
    print("  Характеристики:")
    trunk_features = [
        "Короткоживущие ветки (< 1-2 дня)",
        "Частая интеграция в main/trunk",
        "Feature flags для незавершённых фич",
        "Контролируемая сложность (trunk never broken)",
        "Подходит для CI/CD с частыми деплоями",
    ]
    for feat in trunk_features:
        print(f"    • {feat}")

    # ── 1.5 Сравнение стратегий ──
    print("\n[1.5] Сравнение стратегий:")
    strategies = [
        ("Feature Branch", "Маленькие команды", "Простая", "Средняя", "Низкая"),
        ("Gitflow", "Выпуск релизов", "Сложная", "Высокая", "Высокая"),
        ("Trunk-Based", "CI/CD, DevOps", "Средняя", "Высокая", "Низкая"),
    ]
    print("  ┌──────────────────┬──────────────────┬────────────┬──────────┬──────────┐")
    print("  │ Стратегия        │ Когда использовать│ Сложность  │ Гибкость │ Скорость │")
    print("  ├──────────────────┼──────────────────┼────────────┼──────────┼──────────┤")
    for name, use, complexity, flexibility, speed in strategies:
        print(f"  │ {name:<16} │ {use:<16} │ {complexity:<10} │ {flexibility:<8} │ {speed:<8} │")
    print("  └──────────────────┴──────────────────┴────────────┴──────────┴──────────┘")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 2: Слияние (fast-forward, 3-way merge, конфликты)
# ──────────────────────────────────────────────────────────────────────────────
def demo_merging():
    """Демонстрация стратегий слияния в Git."""
    print("=" * 70)
    print("Демон 2: Слияние (fast-forward, 3-way merge, конфликты)")
    print("=" * 70)

    # ── 2.1 Fast-Forward Merge ──
    print("\n[2.1] Fast-Forward Merge:")
    print("  До слияния:")
    print("    main:     A ──── B")
    print("              \\")
    print("  feature:     └── C ──── D")
    print()
    print("  После слияния (HEAD просто перемещается):")
    print("    main:     A ──── B ──── C ──── D  ← main (HEAD)")
    print("  feature:     └── C ──── D")
    print()
    print("  Когда возможно: нет коммитов в main после ответвления")
    print("  Команда: git merge --ff-only feature")
    print()

    # ── 2.2 Симуляция Fast-Forward ──
    print("  Симуляция fast-forward:")
    history_main = ["A001", "A002"]
    history_feature = ["A001", "A002", "B001", "B002", "B003"]
    print(f"    main до:     {history_main}")
    print(f"    feature:     {history_feature}")
    # Fast-forward: просто перемещаем указатель
    history_main = history_feature.copy()
    print(f"    main после:  {history_main}")
    print("    HEAD перемещён на B003 — новый коммит создан не был")

    # ── 2.3 Three-Way Merge ──
    print("\n[2.3] Three-Way Merge:")
    print("  До слияния:")
    print("    main:     A ──── B ──── E")
    print("              \\")
    print("  feature:     └── C ──── D")
    print()
    print("  После слияния (создаётся merge-коммит):")
    print("    main:     A ──── B ──── E ──── M")
    print("              \\                 /")
    print("  feature:     └── C ──── D ──┘")
    print()
    print("  Команда: git merge feature")
    print("  Merge-коммит M имеет двух родителей (E и D)")
    print("  Алгоритм: находит общий базовый коммит (A), "
          "сравнивает A→E и A→D")

    # ── 2.4 Merge Conflicts ──
    print("\n[2.4] Разрешение конфликтов слияния:")
    print("  Конфликт возникает, когда обе ветки изменили одну строку:")
    print()
    print("  === HEAD (main) ===")
    print("  def process(data):")
    print("      result = data * 2      ← изменено в main")
    print("      return result")
    print()
    print("  === feature ===")
    print("  def process(data):")
    print("      result = data * 3      ← изменено в feature")
    print("      return result")
    print()
    print("  Файл после слияния (неразрешённый конфликт):")
    print("  def process(data):")
    print("  <<<<<<< HEAD")
    print("      result = data * 2")
    print("  =======")
    print("      result = data * 3")
    print("  >>>>>>> feature")
    print("      return result")

    # ── 2.5 Стратегии разрешения конфликтов ──
    print("\n[2.5] Стратегии разрешения конфликтов:")
    strategies = [
        ("Ручное редактирование", "Открыть файл, выбрать нужные строки, "
         "удалить маркеры <<<<<<<"),
        ("git checkout --ours file", "Принять версию из текущей ветки"),
        ("git checkout --theirs file", "Принять версию из вливаемой ветки"),
        ("git mergetool", "Открыть графический инструмент "
         "(meld, kdiff3, vscode)"),
        ("git rebase --onto", "Перебазировать ветку, избегая конфликта"),
    ]
    for name, desc in strategies:
        print(f"    • {name}:")
        print(f"      {desc}")

    # ── 2.6 Симуляция разрешения конфликта ──
    print("\n[2.6] Симуляция разрешения конфликта:")
    # Имитируем конфликтные версии
    ours = "    result = data * 2"
    theirs = "    result = data * 3"
    # Разрешение: взять среднее (для демонстрации)
    print(f"  Наша версия:    {ours.strip()}")
    print(f"  Их версия:      {theirs.strip()}")
    # Симуляция: берёмOURS (самый простой путь)
    print(f"  Решение (--ours): {ours.strip()}")
    print(f"  Итоговый код:")
    print("  def process(data):")
    print(f"  {ours}")
    print("      return result")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 3: Rebasing (interactive rebase, squash, чистая история)
# ──────────────────────────────────────────────────────────────────────────────
def demo_rebasing():
    """Демонстрация rebasing в Git."""
    print("=" * 70)
    print("Демон 3: Rebasing (interactive rebase, squash, чистая история)")
    print("=" * 70)

    # ── 3.1 Основы rebase ──
    print("\n[3.1] Основы rebase:")
    print("  До rebase:")
    print("    main:     A ──── B ──── C ──── D")
    print("              \\")
    print("  feature:     └── E ──── F")
    print()
    print("  git rebase main (перебазировать feature на D):")
    print("    main:     A ──── B ──── C ──── D")
    print("                                \\")
    print("  feature:                          └── E' ──── F'")
    print()
    print("  E' и F' — это ПЕРЕСОЗДАННЫЕ коммиты с новыми хешами!")
    print("  Результат: линейная история без merge-коммитов")
    print()
    print("  Важно: НЕ делайте rebase уже общих коммитов (pushed commits)")
    print("  Правило: rebase = до push, merge = после push")

    # ── 3.2 Interactive Rebase ──
    print("\n[3.2] Interactive Rebase (git rebase -i HEAD~5):")
    print("  pick a1b2c3d Добавлена форма входа")
    print("  pick d4e5f6a Фикс опечатки в форме")
    print("  pick b7c8d9e Ещё одна правка")
    print("  pick e0f1a2b Рефакторинг валидации")
    print("  pick c3d4e5f Тесты для формы")
    print()
    print("  Возможные действия:")
    actions = [
        ("pick", "Оставить коммит как есть"),
        ("reword", "Изменить сообщение коммита"),
        ("edit", "Остановиться для редактирования содержимого"),
        ("squash", "Объединить с предыдущим коммитом"),
        ("fixup", "Объединить, отбросив сообщение этого коммита"),
        ("drop", "Удалить коммит"),
    ]
    for cmd, desc in actions:
        print(f"    {cmd:<8} — {desc}")

    # ── 3.3 Squash Commits ──
    print("\n[3.3] Squash коммитов (до и после):")
    print("  До squash:")
    squash_before = [
        ("a1b2c3d", "Добавлена форма входа"),
        ("d4e5f6a", "Фикс опечатки"),
        ("b7c8d9e", "Ещё одна правка формы"),
        ("e0f1a2b", "Валидация формы"),
        ("c3d4e5f", "Тесты формы входа"),
    ]
    for cid, msg in squash_before:
        print(f"    {cid} {msg}")

    print("\n  git rebase -i HEAD~5 → squash все в один:")
    squash_after = [("f9a8b7c", "Добавлена форма входа с валидацией и тестами")]
    for cid, msg in squash_after:
        print(f"    {cid} {msg}")

    print("\n  Результат: 5 коммитов → 1 чистый коммит")

    # ── 3.4 Rebase vs Merge ──
    print("\n[3.4] Rebase vs Merge:")
    comparison = [
        ("Rebase", "Переписывает историю", "Линейная", "Легко читать",
         "Нельзя для общих коммитов"),
        ("Merge", "Сохраняет историю", "С графом", "Полная картина",
         "Merge-коммиты загрязняют лог"),
    ]
    print("  ┌───────────┬────────────────────┬────────────┬───────────┬──────────────────┐")
    print("  │ Операция   │ Эффект              │ История    │ Читаемость│ Безопасность     │")
    print("  ├───────────┼────────────────────┼────────────┼───────────┼──────────────────┤")
    for name, effect, hist, read, safe in comparison:
        print(f"  │ {name:<9} │ {effect:<18} │ {hist:<10} │ {read:<9} │ {safe:<16} │")
    print("  └───────────┴────────────────────┴────────────┴───────────┴──────────────────┘")

    # ── 3.5 Симуляция rebase ──
    print("\n[3.5] Симуляция rebase (пересчёт хешей):")
    original_commits = [
        ("abc1234", "Добавление функции calculate()"),
        ("def5678", "Исправление бага"),
        ("ghi9012", "Рефакторинг calculate()"),
    ]
    print("  Исходная история feature:")
    for cid, msg in original_commits:
        print(f"    {cid} {msg}")

    # Симулируем пересчёт хешей (реальные хешы изменятся)
    rebased_commits = []
    for cid, msg in original_commits:
        new_hash = hashlib.md5(f"rebased-{cid}".encode()).hexdigest()[:7]
        rebased_commits.append((new_hash, msg))

    print("\n  После git rebase main:")
    for cid, msg in rebased_commits:
        print(f"    {cid} (пересоздан) {msg}")

    print("  ВНИМАНИЕ: хеш изменился! Это НЕОБРАТИМОЕ изменение истории")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 4: Git Hooks & автоматизация (pre-commit, соглашения)
# ──────────────────────────────────────────────────────────────────────────────
def demo_git_hooks():
    """Демонстрация Git hooks и автоматизации."""
    print("=" * 70)
    print("Демон 4: Git Hooks & автоматизация (pre-commit, соглашения)")
    print("=" * 70)

    # ── 4.1 Типы Git Hooks ──
    print("\n[4.1] Типы Git Hooks:")
    hooks = {
        "Client-side (локальные)": [
            ("pre-commit", "Перед добавлением в индекс (git add)"),
            ("prepare-commit-msg", "Перед открытием редактора сообщения"),
            ("commit-msg", "После ввода сообщения коммита"),
            ("pre-push", "Перед отправкой в удалённый репозиторий"),
            ("post-checkout", "После переключения ветки (git checkout)"),
        ],
        "Server-side (на сервере)": [
            ("pre-receive", "Перед принятием push"),
            ("post-receive", "После принятия push"),
            ("pre-rebase", "Перед перебазированием"),
        ],
    }
    for side, hook_list in hooks.items():
        print(f"\n  {side}:")
        for hook, desc in hook_list:
            print(f"    {hook:<20} — {desc}")

    # ── 4.2 Пример pre-commit хука ──
    print("\n[4.2] Пример pre-commit хука (.git/hooks/pre-commit):")
    hook_example = """\
#!/bin/bash
# Pre-commit hook: проверка стиля кода и секретов

echo "🔍 Проверка перед коммитом..."

# Проверяем, что нет секретов
if git diff --cached --name-only | xargs grep -l "password\\|secret\\|api_key" 2>/dev/null; then
    echo "❌ ОШИБКА: Обнаружены潜在ные секреты в файлах!"
    echo "   Убедитесь, что ключи вынесены в переменные окружения."
    exit 1
fi

# Запуск линтера
if ! ruff check .; then
    echo "❌ Линтер ruff обнаружил ошибки. Исправьте перед коммитом."
    exit 1
fi

# Запуск форматтера
ruff format --check .

echo "✅ Все проверки пройдены!"
"""
    for line in hook_example.strip().split("\n"):
        print(f"    {line}")

    # ── 4.3 Commit Message Conventions ──
    print("\n[4.3] Conventional Commits:")
    print("  Формат: <type>(<scope>): <description>")
    print()
    commit_types = [
        ("feat", "Новая фича"),
        ("fix", "Исправление бага"),
        ("docs", "Изменения в документации"),
        ("style", "Форматирование (не влияет на логику)"),
        ("refactor", "Рефакторинг (без добавления фич/багфиксов)"),
        ("test", "Добавление/изменение тестов"),
        ("chops", "Сборка, CI/CD, инструменты"),
        ("perf", "Улучшение производительности"),
        ("ci", "Настройка CI"),
        ("build", "Изменения системы сборки"),
    ]
    for t, desc in commit_types:
        print(f"    {t:<8} — {desc}")

    print("\n  Примеры:")
    examples = [
        "feat(auth): добавлена двухфакторная аутентификация",
        "fix(api): исправлена обработка 404 ошибки",
        "docs(readme): добавлен раздел установки",
        "refactor(core): вынесен парсер в отдельный модуль",
        "test(auth): добавлены unit-тесты для login()",
        "chore(deps): обновлены зависимости",
    ]
    for ex in examples:
        print(f"    • {ex}")

    # ── 4.4 Конфигурация hook-менеджеров ──
    print("\n[4.4] Инструменты управления хуками:")
    hook_managers = [
        ("pre-commit", "pip install pre-commit",
         ".pre-commit-config.yaml",
         "Универсальный менеджер, множество встроенных хуков"),
        ("husky", "npm install husky",
         ".husky/pre-commit",
         "Для JS/Node.js проектов"),
        ("lefthook", "brew install lefthook",
         "lefthook.yml",
         "Быстрый, на Go, параллельный запуск"),
    ]
    for name, install, config, desc in hook_managers:
        print(f"    {name}:")
        print(f"      Установка: {install}")
        print(f"      Конфиг: {config}")
        print(f"      Особенность: {desc}")

    # ── 4.5 Пример .pre-commit-config.yaml ──
    print("\n[4.5] Пример .pre-commit-config.yaml:")
    config = """\
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace    # убирает пробелы в конце строк
      - id: end-of-file-fixer      # добавляет перенос строки в конце
      - id: check-yaml             # проверяет синтаксис YAML
      - id: check-added-large-files  # запрещает файлы > 1MB

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff                   # линтер
      - id: ruff-format            # форматтер

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks               # поиск секретов
"""
    for line in config.strip().split("\n"):
        print(f"    {line}")

    # ── 4.6 Симуляция проверки commit-msg ──
    print("\n[4.6] Симуляция проверки commit-msg hook:")
    test_messages = [
        ("feat(auth): добавлена регистрация", True, "Валидный Conventional Commit"),
        ("fix: исправлен баг", True, "Валидно (scope опционален)"),
        ("исправил баг", False, "Нет типа (feat/fix/...)"),
        ("Feat: опечатка в регистре", False, "Тип должен быть в нижнем регистре"),
        ("feat(auth): добавлена регистрация\n\n подробное описание\n\n BREAKING CHANGE: API изменено",
         True, "С body и BREAKING CHANGE"),
    ]

    valid_types = {t for t, _ in commit_types}

    for msg, expected_valid, reason in test_messages:
        first_line = msg.split("\n")[0]
        match = re.match(r"^(\w+)(?:\(.+\))?:\s+.+", first_line)
        if match:
            commit_type = match.group(1)
            is_valid = commit_type in valid_types
        else:
            is_valid = False

        status = "✓ Допущен" if is_valid else "✗ ОТКЛОНЁН"
        print(f"\n  Сообщение: \"{first_line}\"")
        print(f"    Ожидание: {expected_valid}")
        print(f"    Результат: {status}")
        print(f"    Причина: {reason}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_branching_strategies()
    demo_merging()
    demo_rebasing()
    demo_git_hooks()
    print("=" * 70)
    print("Конец урока 142: Git Advanced")
    print("=" * 70)
