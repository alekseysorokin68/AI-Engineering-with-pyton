"""141 — Python Packaging: pip, pyproject.toml, wheel, virtual environments

Темы:
  1. Package Structure (src layout, __init__.py, modules)
  2. pyproject.toml (metadata, dependencies, build system)
  3. Building Distributions (sdist, wheel, versioning)
  4. Virtual Environments (venv, dependency isolation, requirements.txt)

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
# Демон 1: Структура пакета (src layout, __init__.py, модули)
# ──────────────────────────────────────────────────────────────────────────────
def demo_package_structure():
    """Демонстрация структуры Python-пакета и имён модулей."""
    print("=" * 70)
    print("Демон 1: Структура пакета (src layout, __init__.py, модули)")
    print("=" * 70)

    # ── 1.1 Имитация файловой структуры пакета ──
    # src layout: исходники лежат внутри src/ чтобы избежать
    # случайного импорта локальной папки вместо установленного пакета
    package_tree = {
        "my_package": {
            "__init__.py": "# версия пакета\n__version__ = '0.1.0'\n",
            "core.py": "# основная логика\ndef compute(): pass\n",
            "utils.py": "# вспомогательные функции\ndef helper(): pass\n",
            "subpkg": {
                "__init__.py": "# подпакет\n",
                "transforms.py": "# трансформации\ndef transform(): pass\n",
            },
        }
    }

    print("\n[1.1] Типичная структура src-layout пакета:")
    print("  my_package/")
    print("  ├── src/")
    print("  │   └── my_package/")
    print("  │       ├── __init__.py    ← помечает каталог как пакет")
    print("  │       ├── core.py        ← основной модуль")
    print("  │       ├── utils.py       ← утилиты")
    print("  │       └── subpkg/        ← подпакет")
    print("  │           ├── __init__.py")
    print("  │           └── transforms.py")
    print("  ├── pyproject.toml")
    print("  └── tests/")

    # ── 1.2 Зачем нужен src layout ──
    print("\n[1.2] Преимущества src layout:")
    advantages = [
        ("Изоляция исходников", "Импорт всегда через установленный пакет, "
         "а не через sys.path к корню репозитория"),
        ("Явная зависимость", "Пакет нельзя использовать без установки "
         "(pip install -e .) — ловим ошибки на раннем этапе"),
        ("Совместимость", "Стандарт PyPA, поддерживается setuptools, "
         "hatchling, flit, poetry и другими сборщиками"),
        ("Чистый тест", "Тесты работают против установленной версии, "
         "а не против случайных файлов в cwd"),
    ]
    for title, desc in advantages:
        print(f"    • {title}: {desc}")

    # ── 1.3 __init__.py и импорт ──
    print("\n[1.3] Роль __init__.py:")
    print("  __init__.py определяет публичное API пакета.")
    print("  Пример содержимого __init__.py:")
    print('    from .core import compute')
    print('    from .utils import helper')
    print("    __all__ = ['compute', 'helper']  # экспорт через from pkg import *")
    print("    __version__ = '0.1.0'")

    # ── 1.4 Симуляция импорта модуля ──
    # Имитируем, что модуль core.py содержит полезную функцию
    core_module_source = package_tree["my_package"]["core.py"]
    print("\n[1.4] Имитация импорта модуля:")
    print(f"  Исходник core.py: {core_module_source.strip()}")
    # Симулируем динамический импорт через exec
    ns = {}
    exec("def compute(x): return x ** 2 + 1", ns)
    result = ns["compute"](7)
    print(f"  Динамический импорт и вызов compute(7) = {result}")
    print("  Формула: compute(x) = x² + 1, compute(7) = 49 + 1 = 50")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 2: pyproject.toml (метаданные, зависимости, build system)
# ──────────────────────────────────────────────────────────────────────────────
def demo_pyproject_toml():
    """Демонстрация формата pyproject.toml и его парсинг."""
    print("=" * 70)
    print("Демон 2: pyproject.toml (метаданные, зависимости, build system)")
    print("=" * 70)

    # ── 2.1 Структура pyproject.toml ──
    pyproject_content = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ai-ml-toolkit"
version = "0.3.0"
description = "Набор утилит для AI/ML-инженерии"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Иван Иванов", email = "ivan@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "requests>=2.28",
    "rich>=13.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1"]
docs = ["sphinx>=5.0"]

[project.scripts]
ai-toolkit = "ai_ml_toolkit.cli:main"
"""

    print("\n[2.1] Полный пример pyproject.toml:")
    for line in pyproject_content.strip().split("\n"):
        print(f"  {line}")

    # ── 2.2 Парсинг pyproject.toml вручную ──
    print("\n[2.2] Парсинг полей pyproject.toml:")
    toml_dict = {}
    current_section = None
    for line in pyproject_content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            toml_dict[current_section] = {}
        elif "=" in line and current_section:
            key, _, value = line.partition("=")
            key = key.strip().strip('"')
            value = value.strip().strip('"').strip("'")
            # Простая обработка списков
            if value.startswith("["):
                value = [v.strip().strip('"').strip("'")
                         for v in value[1:-1].split(",") if v.strip()]
            toml_dict[current_section][key] = value

    for section, fields in toml_dict.items():
        print(f"  [{section}]")
        for k, v in fields.items():
            print(f"    {k} = {v}")

    # ── 2.3 Формирование строки зависимостей ──
    print("\n[2.3] Извлечение зависимостей и формирование requirements:")
    deps = toml_dict.get("project", {}).get("dependencies", [])
    if isinstance(deps, str):
        deps = [deps]
    print(f"  Основные зависимости ({len(deps)} шт.):")
    for dep in deps:
        name = re.split(r"[>=<~!]", dep)[0]
        spec = dep[len(name):]
        print(f"    • {name} → спецификация: {spec or '(любая версия)'}")

    # ── 2.4 Проверка версионного диапазона ──
    print("\n[2.4] Проверка версионных спецификаций (симуляция):")
    version_checks = [
        ("requests", "2.30.0", ">=2.28"),
        ("requests", "2.27.0", ">=2.28"),
        ("rich", "13.5.0", ">=13.0"),
        ("rich", "12.0.0", ">=13.0"),
    ]
    for pkg, installed, spec in version_checks:
        # Простая проверка: извлекаем минимальную версию и сравниваем
        min_ver = spec.lstrip(">=<!~")
        ok = installed >= min_ver
        status = "✓ Удовлетворяет" if ok else "✗ НЕ удовлетворяет"
        print(f"    {pkg}=={installed} {spec} → {status}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 3: Сборка дистрибутивов (sdist, wheel, версионирование)
# ──────────────────────────────────────────────────────────────────────────────
def demo_building_distributions():
    """Демонстрация сборки sdist/wheel и управление версиями."""
    print("=" * 70)
    print("Демон 3: Сборка дистрибутивов (sdist, wheel, версионирование)")
    print("=" * 70)

    # ── 3.1 Форматы дистрибутивов ──
    print("\n[3.1] Типы дистрибутивов:")
    print("  ┌─────────┬───────────────────────────────────────────────────┐")
    print("  │  Формат  │  Описание                                         │")
    print("  ├─────────┼───────────────────────────────────────────────────┤")
    print("  │  sdist   │  Исходный код (.tar.gz). Нужен Python и           │")
    print("  │         │  build-зависимости для сборки на целевой машине.   │")
    print("  ├─────────┼───────────────────────────────────────────────────┤")
    print("  │  wheel   │  Бинарный пакет (.whl). Готовый к установке,      │")
    print("  │         │  без этапа компиляции. Быстрее и безопаснее.      │")
    print("  ├─────────┼───────────────────────────────────────────────────┤")
    print("  │  wheel   │  Для C-расширений: содержит скомпилированные     │")
    print("  │ (binary) │  .so/.pyd файлы для конкретной платформы.        │")
    print("  └─────────┴───────────────────────────────────────────────────┘")

    # ── 3.2 Именование wheel-файлов ──
    print("\n[3.2] Формат имени wheel-файла:")
    print("  {distribution}-{version}(-{build tag})?-{python tag}-"
          "{abi tag}-{platform tag}.whl")
    wheel_names = [
        ("ai_ml_toolkit", "0.3.0", "py3", "none", "any",
         "Pure Python пакет"),
        ("ai_ml_toolkit", "0.3.0", "cp311", "cp311", "manylinux_2_17_x86_64",
         "CPython 3.11, Linux x86_64"),
        ("ai_ml_toolkit", "0.3.0", "cp310", "cp310", "win_amd64",
         "CPython 3.10, Windows x64"),
    ]
    for name, ver, py, abi, plat, desc in wheel_names:
        whl = f"{name}-{ver}-{py}-{abi}-{plat}.whl"
        print(f"  {whl}")
        print(f"    └─ {desc}")

    # ── 3.3 Управление версиями (CalVer vs SemVer) ──
    print("\n[3.3] Стратегии версионирования:")
    print("  SemVer (Semantic Versioning): MAJOR.MINOR.PATCH")
    print("    MAJOR — несовместимые изменения API")
    print("    MINOR — обратно-совместимые новые возможности")
    print("    PATCH — обратно-совместимые исправления багов")
    print()
    semver_sequence = [
        (0, 1, 0, "Первая публикация"),
        (0, 2, 0, "Добавлена функция validate()"),
        (0, 2, 1, "Исправлен баг в parse()"),
        (1, 0, 0, "Стабильный API, готов к production"),
        (1, 1, 0, "Добавлена поддержка Python 3.12"),
        (2, 0, 0, "Удалены deprecated-методы"),
    ]
    for major, minor, patch, desc in semver_sequence:
        print(f"    v{major}.{minor}.{patch} — {desc}")

    print("\n  CalVer (Calendar Versioning): YYYY.MM.PATCH")
    print("  Примеры:")
    calver_examples = [
        ("2025.1.0", "Первый релиз 2025 года"),
        ("2025.6.3", "Шестой релиз, третий патч"),
    ]
    for ver, desc in calver_examples:
        print(f"    {ver} — {desc}")

    # ── 3.4 Симуляция сборки ──
    print("\n[3.4] Симуляция процесса сборки:")
    build_steps = [
        ("Сборка sdist", "python -m build --sdist", "ai_ml_toolkit-0.3.0.tar.gz"),
        ("Сборка wheel", "python -m build --wheel", "ai_ml_toolkit-0.3.0-py3-none-any.whl"),
        ("Проверка", "twine check dist/*", "PASSED"),
        ("Загрузка", "twine upload dist/*", "Upload success"),
    ]
    for step_name, command, result in build_steps:
        # Симулируем хеш-сумму для уникальности артефактов
        artifact_hash = hashlib.md5(result.encode()).hexdigest()[:8]
        print(f"  Шаг: {step_name}")
        print(f"    Команда: {command}")
        print(f"    Результат: {result} (хеш: {artifact_hash})")
        print()

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 4: Виртуальные окружения (venv, изоляция, requirements.txt)
# ──────────────────────────────────────────────────────────────────────────────
def demo_virtual_environments():
    """Демонстрация venv, изоляции зависимостей и requirements.txt."""
    print("=" * 70)
    print("Демон 4: Виртуальные окружения (venv, изоляция, requirements.txt)")
    print("=" * 70)

    # ── 4.1 Структура venv ──
    print("\n[4.1] Структура виртуального окружения venv:")
    venv_structure = """\
  .venv/
  ├── pyvenv.cfg          ← конфигурация (python-home, version)
  ├── Include/             ← заголовочные файлы (Windows)
  ├── Lib/
  │   └── site-packages/  ← установленные пакеты
  │       ├── pip/
  │       ├── setuptools/
  │       └── ...         ← ваши зависимости
  └── Scripts/ (Windows) / bin/ (Linux)
      ├── python.exe
      ├── pip.exe
      └── activate
  """
    for line in venv_structure.strip().split("\n"):
        print(f"  {line}")

    # ── 4.2 Создание и активация ──
    print("\n[4.2] Команды создания и использования venv:")
    commands = [
        ("Создание окружения", "python -m venv .venv"),
        ("Активация (Linux/Mac)", "source .venv/bin/activate"),
        ("Активация (Windows)", ".venv\\Scripts\\Activate.ps1"),
        ("Деактивация", "deactivate"),
        ("Установка пакета", "pip install requests"),
        ("Сохранение зависимостей", "pip freeze > requirements.txt"),
        ("Установка из файла", "pip install -r requirements.txt"),
        ("Просмотр установленных", "pip list"),
    ]
    for desc, cmd in commands:
        print(f"  {desc}:")
        print(f"    $ {cmd}")

    # ── 4.3 Формат requirements.txt ──
    print("\n[4.3] Пример requirements.txt:")
    requirements = """\
# Файл с зависимостями (сгенерирован pip freeze)
# Основные
requests==2.31.0
rich==13.7.0
click==8.1.7

# Точная фиксация версий (рекомендуется для production)
numpy==1.26.4
pandas==2.2.0

# Диапазон версий (для разработки)
pytest>=7.0,<8.0
ruff>=0.1.0

# Ссылка на git-репозиторий
# git+https://github.com/user/repo.git@main
"""
    for line in requirements.strip().split("\n"):
        print(f"  {line}")

    # ── 4.4 Сравнение инструментов управления окружениями ──
    print("\n[4.4] Сравнение инструментов управления окружениями:")
    tools_data = [
        ("venv", "Стандартная библиотека", "Простота", "Нет.lock-файла"),
        ("virtualenv", "pip install virtualenv", "Быстрее venv", "Кеширование"),
        ("poetry", "pip install poetry", "pyproject.toml + lock", "Сложнее"),
        ("pipenv", "pip install pipenv", "Pipfile + Pipfile.lock", "Дебаты"),
        ("pdm", "pip install pdm", "pyproject.toml + lock", "Молодой"),
        ("uv", "curl -LsSf .../uv.sh", "Молниеносный", "Новый"),
    ]
    print("  ┌───────────┬───────────────────────┬───────────────────┬──────────────────┐")
    print("  │ Инструмент│ Установка              │ Особенность        │ Недостаток        │")
    print("  ├───────────┼───────────────────────┼───────────────────┼──────────────────┤")
    for tool, install, feature, drawback in tools_data:
        print(f"  │ {tool:<9} │ {install:<21} │ {feature:<17} │ {drawback:<16} │")
    print("  └───────────┴───────────────────────┴───────────────────┴──────────────────┘")

    # ── 4.5 Симуляция изоляции ──
    print("\n[4.5] Симуляция изоляции зависимостей:")
    print("  Проект A требует: requests==2.28, rich==13.0")
    print("  Проект B требует: requests==2.31, rich==12.0")

    # Моделируем конфликт версий
    project_a_deps = {"requests": "2.28.0", "rich": "13.0.0"}
    project_b_deps = {"requests": "2.31.0", "rich": "12.0.0"}

    print(f"\n  Без venv: конфликт! requests не может быть одновременно 2.28 и 2.31")
    print(f"  С venv: каждое окружение изолировано:")
    print(f"    .venv_a/: {json.dumps(project_a_deps, indent=0)}")
    print(f"    .venv_b/: {json.dumps(project_b_deps, indent=0)}")

    # ── 4.6 Проверка целостности зависимостей ──
    print("\n[4.6] Проверка целостности requirements.txt (симуляция):")
    req_lines = [l.strip() for l in requirements.strip().split("\n")
                 if l.strip() and not l.strip().startswith("#")]

    total_packages = 0
    for line in req_lines:
        pkg_name = re.split(r"[>=<~!]", line)[0].strip()
        if not pkg_name:
            continue
        total_packages += 1
        # Проверяем, что спецификация валидна
        spec = line[len(pkg_name):]
        if spec and not re.match(r"^[>=<~!]+[\d.]+$", spec):
            print(f"  ⚠ {line}: нестандартная спецификация")
        else:
            print(f"  ✓ {line}")

    print(f"\n  Итого пакетов для установки: {total_packages}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_package_structure()
    demo_pyproject_toml()
    demo_building_distributions()
    demo_virtual_environments()
    print("=" * 70)
    print("Конец урока 141: Python Packaging")
    print("=" * 70)
