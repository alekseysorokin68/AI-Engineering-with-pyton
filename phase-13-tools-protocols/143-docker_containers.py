"""143 — Docker & Containerization: images, layers, multi-stage builds

Темы:
  1. Container Concepts (images, containers, layers, union filesystem)
  2. Dockerfile Patterns (multi-stage build, caching, slim images)
  3. Image Optimization (layer ordering, .dockerignore, security scanning)
  4. Container Orchestration Basics (docker-compose, networking, volumes)

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
# Демон 1: Концепции контейнеров (образы, контейнеры, слои, union FS)
# ──────────────────────────────────────────────────────────────────────────────
def demo_container_concepts():
    """Демонстрация основных концепций Docker: образы, слои, контейнеры."""
    print("=" * 70)
    print("Демон 1: Концепции контейнеров (образы, контейнеры, слои, union FS)")
    print("=" * 70)

    # ── 1.1 Архитектура Docker ──
    print("\n[1.1] Архитектура Docker:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Docker CLI  →  Docker daemon (dockerd)             │")
    print("  │                    │                                 │")
    print("  │              ┌─────┴─────┐                          │")
    print("  │              │ containerd│ → runc (OCI runtime)     │")
    print("  │              └───────────┘                          │")
    print("  │                                                     │")
    print("  │  Образы хранятся в: /var/lib/docker/                │")
    print("  │  Контейнеры: overlay2 (union filesystem)           │")
    print("  └─────────────────────────────────────────────────────┘")

    # ── 1.2 Слои образа ──
    print("\n[1.2] Слои образа (Layer Architecture):")
    layers = [
        ("Layer 4: COPY app.py /app/", "8.2 KB", "Копирование кода"),
        ("Layer 3: RUN pip install -r req.txt", "45.3 MB", "Установка зависимостей"),
        ("Layer 2: RUN apt-get install python3", "120.7 MB", "Системные пакеты"),
        ("Layer 1: FROM python:3.11-slim", "152.4 MB", "Базовый образ"),
    ]
    print("  Каждая инструкция Dockerfile создаёт ОДИН слой:")
    print()
    for i, (layer, size, desc) in enumerate(layers):
        bar_len = min(int(float(size.split()[0]) / 10), 30)
        bar = "█" * bar_len
        print(f"    {layer}")
        print(f"      │ {size:<10} │ {bar} │ {desc}")

    print()
    print("  Итого: 326.6 MB (с кешем слоёв пересоздание app.py = 8.2 KB)")
    print("  Формула экономии: ΔSize = Size(Layer4) = 8.2 KB (вместо 326.6 MB)")

    # ── 1.3 Union Filesystem ──
    print("\n[1.3] Union Filesystem (overlay2):")
    print("  overlay2 объединяет слои в единую файловую систему:")
    print()
    print("  ┌───────────────────────────────────────┐")
    print("  │  Контейнер видит единый корень /      │")
    print("  │                                       │")
    print("  │  ┌─── Top Layer (read-write) ───┐     │")
    print("  │  │  Изменения контейнера        │     │")
    print("  │  └──────────────────────────────┘     │")
    print("  │  ┌─── Lower Layer 1 (read-only) ─┐    │")
    print("  │  │  COPY app.py /app/             │    │")
    print("  │  └────────────────────────────────┘    │")
    print("  │  ┌─── Lower Layer 2 (read-only) ─┐    │")
    print("  │  │  pip install dependencies      │    │")
    print("  │  └────────────────────────────────┘    │")
    print("  │  ┌─── Lower Layer 3 (read-only) ─┐    │")
    print("  │  │  apt-get install packages      │    │")
    print("  │  └────────────────────────────────┘    │")
    print("  │  ┌─── Base Layer (read-only) ────┐     │")
    print("  │  │  FROM python:3.11-slim        │     │")
    print("  │  └───────────────────────────────┘     │")
    print("  └───────────────────────────────────────┘")
    print()
    print("  Copy-on-Write (CoW): изменение файла копирует его в top layer")

    # ── 1.4 Образ vs Контейнер ──
    print("\n[1.4] Разница между образом (image) и контейнером (container):")
    print("  ┌─────────────────────┬──────────────────────────────────────┐")
    print("  │      Образ           │      Контейнер                       │")
    print("  ├─────────────────────┼──────────────────────────────────────┤")
    print("  │  Read-only шаблон    │  Экземпляр образа с read-write слоем│")
    print("  │  Несколько слоёв     │  Один read-write слой поверх        │")
    print("  │  Не изменяется       │  Состояние меняется при работе       │")
    print("  │  Хранится в registry │  Запускается на хосте                │")
    print("  │  docker image pull   │  docker run                          │")
    print("  └─────────────────────┴──────────────────────────────────────┘")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 2: Паттерны Dockerfile (multi-stage, кеширование, slim)
# ──────────────────────────────────────────────────────────────────────────────
def demo_dockerfile_patterns():
    """Демонстрация паттернов написания Dockerfile."""
    print("=" * 70)
    print("Демон 2: Паттерны Dockerfile (multi-stage, кеширование, slim)")
    print("=" * 70)

    # ── 2.1 Базовый Dockerfile ──
    print("\n[2.1] Базовый Dockerfile (простой):")
    basic_dockerfile = """\
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
"""
    for line in basic_dockerfile.strip().split("\n"):
        print(f"  {line}")

    # ── 2.2 Multi-Stage Build ──
    print("\n[2.2] Multi-Stage Build (сборка + запуск):")
    multistage = """\
# === Stage 1: Сборка ===
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY . .
RUN python -m py_compile app.py  # проверка синтаксиса

# === Stage 2: Запуск ===
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /app /app
EXPOSE 8000
CMD ["python", "app.py"]
"""
    for line in multistage.strip().split("\n"):
        print(f"  {line}")

    print("\n  Экономия:")
    print("    Stage 1 (builder):  890 MB  ← компиляторы, исходники, кеш pip")
    print("    Stage 2 (runtime):  210 MB  ← только бинарники и зависимости")
    print("    Итого в registry:   210 MB  ← экономия 680 MB (76%)!")

    # ── 2.3 Кеширование слоёв ──
    print("\n[2.3] Кеширование слоёв и оптимизация порядка инструкций:")
    print("  Правило: менее изменяемые инструкции → ВЫШЕ в Dockerfile")
    print()
    print("  ❌ ПЛОХО (кеш ломается часто):")
    bad_dockerfile = """\
COPY . .                         # ← меняется при КАЖДОМ коммите
RUN pip install -r requirements.txt  # ← пересоздаётся каждый раз!"""
    for line in bad_dockerfile.strip().split("\n"):
        print(f"    {line}")

    print()
    print("  ✓ ХОРОШО (кеш используется эффективно):")
    good_dockerfile = """\
COPY requirements.txt .          # ← меняется РЕДКО
RUN pip install -r requirements.txt  # ← кешируется!
COPY . .                         # ← кеш слоя pip сохраняется"""
    for line in good_dockerfile.strip().split("\n"):
        print(f"    {line}")

    # ── 2.4 Slim образы ──
    print("\n[2.4] Выбор базового образа:")
    base_images = [
        ("python:3.11", "890 MB", "Полный Debian + Python", "Разработка"),
        ("python:3.11-slim", "152 MB", "Минимальный Debian + Python", "Production"),
        ("python:3.11-alpine", "52 MB", "Alpine Linux", "Минимальный размер"),
        ("python:3.11-slim-bookworm", "145 MB", "Debian Bookworm slim", "Стабильный"),
        ("gcr.io/distroless/python3", "48 MB", "Без shell", "Максимальная безопасность"),
    ]
    print("  ┌──────────────────────────────────┬──────────┬──────────────────────┬────────────────┐")
    print("  │ Образ                              │ Размер   │ Описание              │ Когда использовать│")
    print("  ├──────────────────────────────────┼──────────┼──────────────────────┼────────────────┤")
    for img, size, desc, use in base_images:
        print(f"  │ {img:<32} │ {size:<8} │ {desc:<20} │ {use:<16} │")
    print("  └──────────────────────────────────┴──────────┴──────────────────────┴────────────────┘")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 3: Оптимизация образов (порядок слоёв, .dockerignore, безопасность)
# ──────────────────────────────────────────────────────────────────────────────
def demo_image_optimization():
    """Демонстрация оптимизации Docker-образов."""
    print("=" * 70)
    print("Демон 3: Оптимизация образов (порядок слоёв, .dockerignore, безопасность)")
    print("=" * 70)

    # ── 3.1 .dockerignore ──
    print("\n[3.1] Файл .dockerignore:")
    dockerignore = """\
# Игнорируем ненужные файлы при сборке контекста
.git
.gitignore
.env
.env.*
*.pyc
__pycache__/
.mypy_cache/
.pytest_cache/
.venv/
venv/
node_modules/
*.egg-info/
dist/
build/
.vscode/
.idea/
*.md
!README.md
Dockerfile
docker-compose*.yml
tests/
docs/
"""
    for line in dockerignore.strip().split("\n"):
        print(f"    {line}")

    # Симуляция расчёта размера контекста
    print("\n  Симуляция расчёта размера контекста:")
    files = [
        (".git/", "45.2 MB", True),
        ("src/", "2.3 MB", False),
        ("requirements.txt", "0.5 KB", False),
        ("tests/", "1.1 MB", True),
        ("docs/", "3.4 MB", True),
        (".venv/", "120.5 MB", True),
        ("app.py", "12.0 KB", False),
        ("Dockerfile", "0.8 KB", True),
    ]
    total_before = 0
    total_after = 0
    for fname, size_str, ignored in files:
        # Парсим размер
        size_val = float(re.search(r"[\d.]+", size_str).group())
        if "MB" in size_str:
            size_val *= 1024 * 1024
        elif "KB" in size_str:
            size_val *= 1024

        total_before += size_val
        if not ignored:
            total_after += size_val
        status = "✗ игнорируется" if ignored else "✓ в контексте"
        print(f"    {fname:<25} {size_str:<12} {status}")

    total_before_mb = total_before / (1024 * 1024)
    total_after_mb = total_after / (1024 * 1024)
    print(f"\n  Контекст до .dockerignore: {total_before_mb:.1f} MB")
    print(f"  Контекст после .dockerignore: {total_after_mb:.1f} MB")
    print(f"  Экономия: {total_before_mb - total_after_mb:.1f} MB "
          f"({(1 - total_after/total_before)*100:.0f}%)")

    # ── 3.2 Порядок слоёв ──
    print("\n[3.2] Оптимальный порядок слоёв:")
    print("  Принцип: частые изменения → вниз, редкие → вверх")
    print()
    optimization_steps = [
        ("1. FROM", "Базовый образ (меняется раз в квартал)"),
        ("2. RUN apt-get update && install", "Системные зависимости (редко)"),
        ("3. COPY requirements.txt", "Список зависимостей (иногда)"),
        ("4. RUN pip install", "Python-пакеты (иногда)"),
        ("5. COPY . .", "Исходный код (часто)"),
        ("6. RUN python -m compileall", "Компиляция (часто)"),
        ("7. EXPOSE / CMD", "Метаданные (почти всегда)"),
    ]
    for step, desc in optimization_steps:
        print(f"    {step:<35} ← {desc}")

    # ── 3.3 Безопасность образов ──
    print("\n[3.3] Безопасность Docker-образов:")
    security_practices = [
        ("Не запускать от root",
         "USER non-root в Dockerfile",
         "Снижение привилегий, CVE-защита"),
        ("Сканирование CVE",
         "docker scout cves image:tag",
         "Поиск известных уязвимостей"),
        ("Минимальный образ",
         "Использовать slim/alpine/distroless",
         "Меньше поверхность атаки"),
        ("Регулярное обновление",
         "docker pull python:3.11-slim",
         "Патчи безопасности"),
        ("Без секретов в образе",
         "multi-stage build, --mount=type=secret",
         "Нет ключей в слоях"),
    ]
    for name, tool, reason in security_practices:
        print(f"    • {name}:")
        print(f"      Инструмент: {tool}")
        print(f"      Причина: {reason}")

    # ── 3.4 Симуляция анализа образа ──
    print("\n[3.4] Симуляция анализа образа (docker history):")
    image_layers = [
        ("sha256:a1b2c3", "152 MB", "FROM python:3.11-slim"),
        ("sha256:d4e5f6", "45 MB", "RUN apt-get update && apt-get install -y curl"),
        ("sha256:g7h8i9", "23 MB", "COPY requirements.txt /app/"),
        ("sha256:j0k1l2", "89 MB", "RUN pip install --no-cache-dir -r /app/requirements.txt"),
        ("sha256:m3n4o5", "0.01 MB", "COPY . /app/"),
        ("sha256:p6q7r8", "0 KB", "EXPOSE 8000"),
        ("sha256:s9t0u1", "0 KB", 'CMD ["python", "/app/main.py"]'),
    ]
    print("  IMAGE          CREATED BY                                      SIZE")
    print("  " + "-" * 68)
    for layer_id, size, cmd in image_layers:
        print(f"  {layer_id}  {cmd:<48} {size}")

    # Считаем размер слоя с pip install
    pip_size_mb = 89
    print(f"\n  ⚠ Самый большой слой: pip install ({pip_size_mb} MB)")
    print("  Рекомендация: использовать кеширование:")
    print("    RUN --mount=type=cache,target=/root/.cache/pip \\")
    print("        pip install -r requirements.txt")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Демон 4: Основы оркестрации (docker-compose, сеть, тома)
# ──────────────────────────────────────────────────────────────────────────────
def demo_container_orchestration():
    """Демонстрация docker-compose, сетей и томов."""
    print("=" * 70)
    print("Демон 4: Основы оркестрации (docker-compose, networking, volumes)")
    print("=" * 70)

    # ── 4.1 docker-compose.yml ──
    print("\n[4.1] Пример docker-compose.yml (ML-приложение):")
    compose_file = """\
version: '3.9'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/ml_data
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    volumes:
      - model_data:/app/models
      - ./src:/app/src  # hot-reload для разработки
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: ml_data
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d ml_data"]
      interval: 5s
      timeout: 5s
      retries: 5

  cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  model_data:
  pgdata:
"""
    for line in compose_file.strip().split("\n"):
        print(f"  {line}")

    # ── 4.2 Сети в Docker ──
    print("\n[4.2] Сети (Networking) в Docker:")
    print("  Типы сетей:")
    networks = [
        ("bridge", "По умолчанию. Контейнеры в изолированной сети. "
         "Контейнеры доступны друг другу по имени сервиса."),
        ("host", "Контейнер использует сеть хоста (без изоляции). "
         "Быстрее, но менее безопасно."),
        ("none", "Без сетевого доступа. Полная изоляция."),
        ("overlay", "Для кластера Docker Swarm. Межхостовая связь."),
    ]
    for net_type, desc in networks:
        print(f"    {net_type:<8} — {desc}")

    print("\n  Разрешение имён (DNS):")
    print("    web → db          (POSTGRES_HOST=db)")
    print("    web → cache       (REDIS_HOST=cache)")
    print("    docker-compose автоматически создаёт DNS-записи!")

    # ── 4.3 Тома (Volumes) ──
    print("\n[4.3] Тома (Volumes) — персистентные данные:")
    volume_types = [
        ("Named Volume", "Данные хранятся в /var/lib/docker/volumes/",
         "Рекомендуется для production (базы данных, модели)"),
        ("Bind Mount", "Монтирование директории хоста в контейнер",
         "Для разработки (hot-reload кода)"),
        ("tmpfs", "Временная файловая система в RAM",
         "Для секретов, промежуточных данных"),
    ]
    for vtype, location, use in volume_types:
        print(f"    {vtype}:")
        print(f"      Расположение: {location}")
        print(f"      Использование: {use}")

    print("\n  Примеры монтирования:")
    mount_examples = [
        ("volumes:", "model_data:/app/models", "Named volume"),
        ("volumes:", "./src:/app/src", "Bind mount (разработка)"),
        ("volumes:", "./data:/app/data:ro", "Read-only bind mount"),
        ("tmpfs:", "/tmp", "RAM-диск для временных файлов"),
    ]
    for key, value, desc in mount_examples:
        print(f"    {key} {value}  # {desc}")

    # ── 4.4 Симуляция запуска docker-compose ──
    print("\n[4.4] Симуляция запуска docker-compose up:")
    services = [
        ("web", "Создан", "Запущен", "healthy"),
        ("db", "Создан", "Запущен", "healthy"),
        ("cache", "Создан", "Запущен", "healthy"),
    ]
    print("  Контейнеры:")
    for name, created, started, health in services:
        container_id = hashlib.md5(name.encode()).hexdigest()[:12]
        print(f"    {name}  {container_id}  {started}  {health}")

    # ── 4.5 Полезные команды docker-compose ──
    print("\n[4.5] Полезные команды docker-compose:")
    commands = [
        ("docker-compose up -d", "Запуск всех сервисов в фоне"),
        ("docker-compose down", "Остановка и удаление контейнеров"),
        ("docker-compose logs -f web", "Логи конкретного сервиса"),
        ("docker-compose exec web bash", "Войти в контейнер"),
        ("docker-compose ps", "Список запущенных сервисов"),
        ("docker-compose build --no-cache", "Пересобрать образы без кеша"),
        ("docker-compose pull", "Обновить образы из registry"),
        ("docker-compose restart web", "Перезапустить один сервис"),
    ]
    for cmd, desc in commands:
        print(f"    $ {cmd:<42} # {desc}")

    # ── 4.6 Переменные окружения ──
    print("\n[4.6] Управление переменными окружения:")
    print("  Способы передачи конфигурации:")
    env_methods = [
        ("environment:", "Прямое указание в compose-файле"),
        ("env_file:", "Ссылка на .env файл"),
        ("${VAR}", "Подстановка из переменных хоста"),
        ("Docker secrets", "Безопасная передача через Swarm"),
    ]
    for method, desc in env_methods:
        print(f"    {method:<20} — {desc}")

    print("\n  Пример .env файла:")
    env_example = """\
DATABASE_URL=postgres://user:pass@db:5432/ml_data
REDIS_URL=redis://cache:6379
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO
"""
    for line in env_example.strip().split("\n"):
        print(f"    {line}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_container_concepts()
    demo_dockerfile_patterns()
    demo_image_optimization()
    demo_container_orchestration()
    print("=" * 70)
    print("Конец урока 143: Docker & Containerization")
    print("=" * 70)
