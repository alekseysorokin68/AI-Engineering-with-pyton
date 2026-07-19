"""152 — Database Fundamentals: SQL, индексация, оптимизация запросов, ORM

Темы:
  1. SQL Basics — SELECT, JOIN, GROUP BY, подзапросы
  2. Indexing — B-tree, hash, составные индексы, планы запросов
  3. Connection Pooling — размер пула, таймауты, retry-логика
  4. ORM Patterns — маппинг, ленивая загрузка, миграции

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import sqlite3

random.seed(42)

# =============================================================================
# Демо 1: SQL Basics — SELECT, JOIN, GROUP BY, подзапросы
# =============================================================================

def demo_sql_basics():
    print("=" * 70)
    print("ДЕМО 1: SQL Basics — SELECT, JOIN, GROUP BY, подзапросы")
    print("=" * 70)

    # Создаём временную БД в памяти
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # Создаём таблицы
    cur.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            department TEXT,
            salary REAL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT,
            amount REAL,
            order_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Вставка данных
    users = [
        (1, "Иванов И.И.",   "ivanov@mail.ru",   "Engineering", 150000),
        (2, "Петрова А.С.",  "petrova@mail.ru",  "Marketing",   120000),
        (3, "Сидоров К.М.",  "sidorov@mail.ru",  "Engineering", 180000),
        (4, "Козлова Е.В.",  "kozlova@mail.ru",  "HR",          110000),
        (5, "Новиков Д.А.",  "novikov@mail.ru",  "Engineering", 160000),
        (6, "Морозова О.П.", "morozova@mail.ru", "Marketing",   130000),
    ]
    cur.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users)

    orders = [
        (1, 1, "Ноутбук",      120000, "2025-01-15"),
        (2, 1, "Монитор",       45000, "2025-02-10"),
        (3, 2, "Ноутбук",      120000, "2025-01-20"),
        (4, 3, "Сервер",       350000, "2025-03-01"),
        (5, 3, "Ноутбук",      120000, "2025-03-15"),
        (6, 4, "Стул",          35000, "2025-02-28"),
        (7, 5, "Ноутбук",      120000, "2025-01-25"),
        (8, 5, "Клавиатура",    15000, "2025-02-15"),
        (9, 6, "Монитор",       45000, "2025-03-10"),
        (10, 2, "Стул",         35000, "2025-03-20"),
    ]
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    # --- 1.1 SELECT и WHERE ---
    print("\n--- 1.1 SELECT и WHERE ---")

    cur.execute("SELECT name, department, salary FROM users WHERE salary > 130000")
    rows = cur.fetchall()
    print("Пользователи с зарплатой > 130000:")
    print(f"  {'Имя':>16s}  {'Отдел':>12s}  {'Зарплата':>10s}")
    for name, dept, salary in rows:
        print(f"  {name:>16s}  {dept:>12s}  {salary:>10.0f}")

    # --- 1.2 JOIN ---
    print("\n--- 1.2 JOIN (INNER, LEFT) ---")

    cur.execute("""
        SELECT u.name, u.department, o.product, o.amount
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        ORDER BY u.name, o.amount DESC
    """)
    rows = cur.fetchall()
    print("INNER JOIN: только пользователи с заказами")
    for name, dept, product, amount in rows:
        print(f"  {name:>16s}  {dept:>12s}  {product:14s}  {amount:>10.0f}₽")

    # LEFT JOIN: все пользователи, даже без заказов
    cur.execute("""
        SELECT u.name, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.id
    """)
    rows = cur.fetchall()
    print("\nLEFT JOIN: количество заказов по пользователям")
    for name, count in rows:
        print(f"  {name:>16s}: {count} заказов")

    # --- 1.3 GROUP BY и агрегатные функции ---
    print("\n--- 1.3 GROUP BY и агрегатные функции ---")

    cur.execute("""
        SELECT u.department,
               COUNT(DISTINCT u.id) as employees,
               ROUND(SUM(o.amount), 0) as total_spent,
               ROUND(AVG(o.amount), 0) as avg_order,
               COUNT(o.id) as orders
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.department
        HAVING total_spent > 0
        ORDER BY total_spent DESC
    """)
    rows = cur.fetchall()
    print("Статистика по отделам:")
    print(f"  {'Отдел':>12s}  {'Сотр.':>5s}  {'Потрачено':>12s}  {'Средний':>10s}  {'Заказов':>7s}")
    for dept, emps, total, avg, orders in rows:
        print(f"  {dept:>12s}  {emps:>5d}  {total:>12.0f}₽  {avg:>10.0f}₽  {orders:>7d}")

    # --- 1.4 Подзапросы ---
    print("\n--- 1.4 Подзапросы ---")

    # Подзапрос в WHERE: пользователи, которые потратили больше среднего
    cur.execute("""
        SELECT name, department, email
        FROM users
        WHERE id IN (
            SELECT user_id
            FROM orders
            GROUP BY user_id
            HAVING SUM(amount) > (
                SELECT AVG(total) FROM (
                    SELECT SUM(amount) as total
                    FROM orders
                    GROUP BY user_id
                )
            )
        )
    """)
    rows = cur.fetchall()
    print("Пользователи, потратившие больше среднего:")
    for name, dept, email in rows:
        print(f"  {name} ({dept}) — {email}")

    conn.close()


# =============================================================================
# Демо 2: Indexing — B-tree, hash, составные индексы, планы запросов
# =============================================================================

def demo_indexing():
    print("=" * 70)
    print("ДЕМО 2: Indexing — B-tree, hash, составные индексы, планы запросов")
    print("=" * 70)

    # Создаём БД с большим объёмом данных
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            in_stock INTEGER DEFAULT 1,
            created_at TEXT
        )
    """)

    # Генерируем 10000 товаров
    categories = ["Electronics", "Books", "Clothing", "Food", "Tools"]
    random.seed(42)
    products = []
    for i in range(1, 10001):
        cat = random.choice(categories)
        price = round(random.uniform(10, 5000), 2)
        products.append((
            i,
            f"Product_{i:05d}",
            cat,
            price,
            random.randint(0, 1),
            f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        ))

    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", products)
    conn.commit()

    # --- 2.1 B-tree индекс ---
    # Балансированное дерево для диапазонных иequality-запросов
    print("\n--- 2.1 B-tree индекс ---")

    print("Пример B-tree (упрощённый) для поля category:")
    print("  [Electronics | Books]")
    print("  /           |         \\")
    print("[Clothing] [Food] [Tools]")
    print()
    print("B-tree поддерживает: =, <, >, <=, >=, BETWEEN, LIKE 'abc%'")
    print("Не поддерживает: LIKE '%abc' ( wildcards в начале)")

    # Создаём индекс
    cur.execute("CREATE INDEX idx_category ON products(category)")
    conn.commit()

    # Анализ плана запроса
    cur.execute("EXPLAIN QUERY PLAN SELECT * FROM products WHERE category = 'Electronics'")
    plan = cur.fetchall()
    print(f"\nПлан запроса (с индексом): {plan[0][3]}")

    # Сравнение: запрос по неиндексированному полю
    cur.execute("EXPLAIN QUERY PLAN SELECT * FROM products WHERE price > 1000")
    plan = cur.fetchall()
    print(f"План запроса (без индекса): {plan[0][3]}")

    # --- 2.2 Hash-индекс ---
    print("\n--- 2.2 Hash-индекс ---")

    print("Hash-индекс: O(1) для точных совпадений (=)")
    print("Не поддерживает: диапазоны, ORDER BY, частичные совпадения")
    print("Используется редко в modern БД (B-tree быстрее на практике)")

    # Симуляция hash-таблицы
    hash_index = {}
    for row in cur.execute("SELECT id, category FROM products"):
        cat = row[1]
        if cat not in hash_index:
            hash_index[cat] = []
        hash_index[cat].append(row[0])

    print("\nHash-индекс по category:")
    for cat, ids in hash_index.items():
        print(f"  {cat:14s}: {len(ids)} записей, hash bucket -> [{ids[0]}..{ids[-1]}]")

    # Поиск через hash: O(1)
    target = "Electronics"
    result_count = len(hash_index.get(target, []))
    print(f"\nПоиск '{target}' через hash: {result_count} результатов (O(1))")

    # --- 2.3 Составные индексы ---
    # Порядок колонок важен: (category, price) != (price, category)
    print("\n--- 2.3 Составные индексы ---")

    # Составной индекс
    cur.execute("CREATE INDEX idx_cat_price ON products(category, price)")
    conn.commit()

    # Запрос, использующий оба поля составного индекса
    cur.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM products
        WHERE category = 'Books' AND price < 100
    """)
    plan = cur.fetchall()
    print(f"Запрос category='Books' AND price<100: {plan[0][3]}")

    # Запрос, использующий только первую колонку
    cur.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM products WHERE category = 'Books'
    """)
    plan = cur.fetchall()
    print(f"Запрос category='Books': {plan[0][3]}")

    print("\nПравило: колонки в составном индексе идут от самого 'узкого' к самому 'широкому'")
    print("  (category, price) эффективен для:")
    print("    WHERE category = ?")
    print("    WHERE category = ? AND price > ?")
    print("    ORDER BY category, price")
    print("  НЕ эффективен для: WHERE price > ? (без category)")

    # --- 2.4 Анализ плана запроса (EXPLAIN) ---
    print("\n--- 2.4 Анализ плана запроса (EXPLAIN) ---")

    # Запрос без индекса
    cur.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM products
        WHERE name = 'Product_05000'
    """)
    plan = cur.fetchall()
    print(f"Поиск по name (без индекса):")
    for row in plan:
        print(f"  {row[3]}")

    # Создаём индекс для name
    cur.execute("CREATE INDEX idx_name ON products(name)")
    conn.commit()

    cur.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM products
        WHERE name = 'Product_05000'
    """)
    plan = cur.fetchall()
    print(f"\nПоиск по name (с индексом):")
    for row in plan:
        print(f"  {row[3]}")

    # Запрос с сортировкой
    cur.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM products
        WHERE category = 'Electronics'
        ORDER BY price
        LIMIT 10
    """)
    plan = cur.fetchall()
    print(f"\nЗапрос с сортировкой (с составным индексом):")
    for row in plan:
        print(f"  {row[3]}")

    # Бенчмарк: время поиска
    print("\nБенчмарк поиска (10000 записей):")
    iterations = 1000

    # Поиск по category (с индексом)
    start = time.perf_counter()
    for _ in range(iterations):
        cur.execute("SELECT * FROM products WHERE category = 'Electronics'")
        cur.fetchall()
    indexed_time = (time.perf_counter() - start) / iterations * 1000

    # Поиск по name (с индексом)
    start = time.perf_counter()
    for _ in range(iterations):
        cur.execute("SELECT * FROM products WHERE name = 'Product_05000'")
        cur.fetchall()
    name_time = (time.perf_counter() - start) / iterations * 1000

    print(f"  category (индекс, ~2000 записей): {indexed_time:.4f}мс")
    print(f"  name (уникальный индекс, 1 запись): {name_time:.4f}мс")

    conn.close()


# =============================================================================
# Демо 3: Connection Pooling — размер пула, таймауты, retry
# =============================================================================

def demo_connection_pooling():
    print("=" * 70)
    print("ДЕМО 3: Connection Pooling — размер пула, таймауты, retry")
    print("=" * 70)

    # --- 3.1 Концепция пула соединений ---
    print("\n--- 3.1 Концепция пула соединений ---")

    print("Без пула:")
    print("  Запрос -> Открыть TCP -> SSL handshake -> Auth -> Query -> Закрыть")
    print("  Время: ~50-200ms только на установку соединения")
    print()
    print("С пулом:")
    print("  Запрос -> Взять из пула (1-5ms) -> Query -> Вернуть в пул")
    print("  Соединения переиспользуются, не тратятся на создание/закрытие")

    # --- 3.2 Модель пула соединений ---
    print("\n--- 3.2 Модель пула соединений ---")

    class ConnectionPool:
        """Простая модель пула соединений"""

        def __init__(self, min_size=5, max_size=20, timeout=30):
            self.min_size = min_size
            self.max_size = max_size
            self.timeout = timeout
            self.available = min_size  # доступные соединения
            self.in_use = 0
            self.total_created = 0
            self.wait_count = 0
            self.timeout_count = 0

        def acquire(self):
            """Получить соединение из пула"""
            if self.available > 0:
                self.available -= 1
                self.in_use += 1
                return True
            elif self.total_created < self.max_size:
                # Создаём новое соединение
                self.total_created += 1
                self.in_use += 1
                return True
            else:
                # Пул исчерпан, ждём или таймаут
                self.wait_count += 1
                self.timeout_count += 1
                return False

        def release(self):
            """Вернуть соединение в пул"""
            self.in_use -= 1
            self.available += 1

        def status(self):
            return {
                "available": self.available,
                "in_use": self.in_use,
                "total": self.available + self.in_use,
                "waiters": self.wait_count,
            }

    pool = ConnectionPool(min_size=5, max_size=10, timeout=30)
    print(f"Пул: min={pool.min_size}, max={pool.max_size}, timeout={pool.timeout}с")

    # Симуляция нагрузки
    print("\nСимуляция нагрузки (50 запросов):")
    for i in range(50):
        if not pool.acquire():
            print(f"  Запрос {i+1}: ОШИБКА — пул исчерпан (таймаут)")
        else:
            # Симулируем выполнение запроса
            if random.random() < 0.3:
                pool.release()

    status = pool.status()
    print(f"\nСтатус пула: available={status['available']}, "
          f"in_use={status['in_use']}, total={status['total']}")
    print(f"Ожиданий: {status['waiters']}")

    # --- 3.3 Retry-логика ---
    print("\n--- 3.3 Retry-логика ---")

    def retry_with_backoff(func, max_retries=3, base_delay=0.1):
        """Экспоненциальный backoff с jitter"""
        attempts = []
        for attempt in range(max_retries):
            try:
                result = func()
                attempts.append({"attempt": attempt + 1, "status": "OK"})
                return result, attempts
            except Exception as e:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.05)
                attempts.append({
                    "attempt": attempt + 1,
                    "status": "FAIL",
                    "error": str(e),
                    "retry_in": f"{delay:.3f}с",
                })
                time.sleep(delay)
        return None, attempts

    # Функция, которая иногда падает
    fail_counter = 0
    def flaky_query():
        nonlocal fail_counter
        fail_counter += 1
        if fail_counter <= 2:
            raise ConnectionError("connection refused")
        return {"rows": 42}

    result, attempts = retry_with_backoff(flaky_query, max_retries=5, base_delay=0.01)
    print("Retry-попытки:")
    for a in attempts:
        if a["status"] == "OK":
            print(f"  Попытка {a['attempt']}: {a['status']}")
        else:
            print(f"  Попытка {a['attempt']}: {a['status']} ({a['error']}) -> retry через {a['retry_in']}")
    print(f"Результат: {result}")

    # --- 3.4 Конфигурация пула для продакшена ---
    print("\n--- 3.4 Рекомендации по конфигурации пула ---")

    configs = [
        {"workload": "Web (CPU-bound)",   "pool_size": 10,  "timeout": 30,  "max_overflow": 5},
        {"workload": "Web (IO-bound)",     "pool_size": 20,  "timeout": 60,  "max_overflow": 10},
        {"workload": "Data pipeline",      "pool_size": 5,   "timeout": 300, "max_overflow": 2},
        {"workload": "High-traffic API",   "pool_size": 50,  "timeout": 10,  "max_overflow": 20},
    ]

    print(f"  {'Workload':>20s}  {'pool_size':>9s}  {'timeout':>7s}  {'max_overflow':>12s}")
    print(f"  {'-'*20:>20s}  {'-'*9:>9s}  {'-'*7:>7s}  {'-'*12:>12s}")
    for c in configs:
        print(f"  {c['workload']:>20s}  {c['pool_size']:>9d}  {c['timeout']:>7d}с  {c['max_overflow']:>12d}")

    print("\nФормула: pool_size = (target_rps × avg_query_time) / concurrency_factor")
    print("  target_rps: целевая пропускная способность")
    print("  avg_query_time: среднее время запроса (в секундах)")
    print("  concurrency_factor: обычно 2-3x для буфера")


# =============================================================================
# Демо 4: ORM Patterns — маппинг, ленивая загрузка, миграции
# =============================================================================

def demo_orm_patterns():
    print("=" * 70)
    print("ДЕМО 4: ORM Patterns — маппинг, ленивая загрузка, миграции")
    print("=" * 70)

    # --- 4.1 Маппинг объектов на таблицы ---
    print("\n--- 4.1 Маппинг объектов на таблицы (ORM Mapping) ---")

    # Простейшая ORM-модель (без внешних библиотек)
    class ORMModel:
        """Базовый класс модели с маппингом на SQL"""
        _table = ""
        _columns = {}

        @classmethod
        def create_table_sql(cls):
            cols = []
            for name, dtype in cls._columns.items():
                constraint = "PRIMARY KEY" if name == "id" else ""
                cols.append(f"    {name} {dtype} {constraint}".strip())
            return f"CREATE TABLE IF NOT EXISTS {cls._table} (\n" + ",\n".join(cols) + "\n)"

        @classmethod
        def insert_sql(cls, data):
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            return f"INSERT INTO {cls._table} ({cols}) VALUES ({placeholders})", list(data.values())

    class User(ORMModel):
        _table = "orm_users"
        _columns = {
            "id": "INTEGER",
            "name": "TEXT",
            "email": "TEXT",
            "department": "TEXT",
        }

    class Article(ORMModel):
        _table = "orm_articles"
        _columns = {
            "id": "INTEGER",
            "title": "TEXT",
            "author_id": "INTEGER",
            "content": "TEXT",
            "published": "INTEGER",
        }

    print(f"Модель User: таблица={User._table}")
    print(f"  Колонки: {list(User._columns.keys())}")
    print(f"  SQL:\n{User.create_table_sql()}")

    print(f"Модель Article: таблица={Article._table}")
    print(f"  Колонки: {list(Article._columns.keys())}")

    # Создаём таблицы
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(User.create_table_sql())
    cur.executescript(Article.create_table_sql())

    # Вставка данных через ORM-маппинг
    users_data = [
        {"id": 1, "name": "Иван", "email": "ivan@example.com", "department": "Engineering"},
        {"id": 2, "name": "Анна", "email": "anna@example.com", "department": "Marketing"},
        {"id": 3, "name": "Олег", "email": "oleg@example.com", "department": "Engineering"},
    ]

    for data in users_data:
        sql, params = User.insert_sql(data)
        cur.execute(sql, params)

    articles_data = [
        {"id": 1, "title": "Введение в ML", "author_id": 1, "content": "Содержание...", "published": 1},
        {"id": 2, "title": "SEO-оптимизация", "author_id": 2, "content": "Содержание...", "published": 1},
        {"id": 3, "title": "Kubernetes 101", "author_id": 1, "content": "Содержание...", "published": 0},
    ]

    for data in articles_data:
        sql, params = Article.insert_sql(data)
        cur.execute(sql, params)

    conn.commit()
    print("Данные успешно вставлены через ORM-маппинг")

    # --- 4.2 Ленивая загрузка (Lazy Loading) ---
    print("\n--- 4.2 Ленивая загрузка (Lazy Loading) ---")

    print("Eager loading: загружает все связанные данные сразу")
    print("  SELECT u.*, a.* FROM users u JOIN articles a ON u.id = a.author_id")
    print("  -> Все данные загружены одной командой, но может быть избыточно")
    print()
    print("Lazy loading: загружает связанные данные по требованию")
    print("  1. Загружаем user")
    print("  2. Только при обращении к user.articles -> запрос к БД")
    print("  -> Экономит память, но может вызвать N+1 запросов")

    # Симуляция N+1 проблемы
    cur.execute("SELECT id, name FROM orm_users")
    all_users = cur.fetchall()

    print("\nN+1 проблема (ленивая загрузка):")
    total_queries = 1
    print(f"  Запрос 1: SELECT id, name FROM orm_users -> {len(all_users)} пользователей")

    for user_id, name in all_users:
        cur.execute("SELECT title FROM orm_articles WHERE author_id = ?", (user_id,))
        articles = cur.fetchall()
        total_queries += 1
        titles = [a[0] for a in articles]
        print(f"  Запрос {total_queries}: SELECT ... WHERE author_id = {user_id} -> {titles}")

    print(f"\n  Всего запросов: {total_queries} (проблема N+1!)")
    print("  Решение: JOIN или subquery для загрузки одним запросом")

    # Решение через JOIN
    cur.execute("""
        SELECT u.name, GROUP_CONCAT(a.title) as articles
        FROM orm_users u
        LEFT JOIN orm_articles a ON u.id = a.author_id
        GROUP BY u.id
    """)
    rows = cur.fetchall()
    print("\nРешение (JOIN + GROUP_CONCAT): 1 запрос вместо N+1")
    for name, arts in rows:
        print(f"  {name}: {arts or '(нет статей)'}")

    # --- 4.3 Миграции ---
    print("\n--- 4.3 Миграции (Schema Migrations) ---")

    # Версионирование схемы
    migrations = [
        {
            "version": 1,
            "description": "Создание таблиц users и articles",
            "up": [
                "CREATE TABLE users_v (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
            ],
            "down": ["DROP TABLE users_v"],
        },
        {
            "version": 2,
            "description": "Добавление колонки department",
            "up": [
                "ALTER TABLE users_v ADD COLUMN department TEXT DEFAULT 'Unknown'",
            ],
            "down": ["ALTER TABLE users_v DROP COLUMN department"],
        },
        {
            "version": 3,
            "description": "Создание индекса по email",
            "up": [
                "CREATE INDEX idx_email ON users_v(email)",
            ],
            "down": ["DROP INDEX idx_email"],
        },
    ]

    print("История миграций:")
    for m in migrations:
        print(f"  v{m['version']}: {m['description']}")
        for cmd in m["up"]:
            print(f"    UP:   {cmd}")
        for cmd in m["down"]:
            print(f"    DOWN: {cmd}")

    # Применяем миграции
    cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    cur.execute("CREATE TABLE IF NOT EXISTS users_v (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")

    for m in migrations:
        cur.execute("SELECT MAX(version) FROM schema_version")
        current = cur.fetchone()[0] or 0
        if m["version"] > current:
            for cmd in m["up"]:
                try:
                    cur.execute(cmd)
                except sqlite3.OperationalError as e:
                    print(f"  Предупреждение: {e}")
            cur.execute("INSERT INTO schema_version VALUES (?)", (m["version"],))
            print(f"  Миграция v{m['version']} применена успешно")

    conn.commit()

    # --- 4.4 Паттерны ORM: relationship, serializer ---
    print("\n--- 4.4 Паттерны ORM: relationship, serializer ---")

    # Связь один-ко-многим
    print("Паттерн один-ко-многим (User has many Articles):")
    cur.execute("""
        SELECT u.id, u.name, u.department, COUNT(a.id) as article_count
        FROM orm_users u
        LEFT JOIN orm_articles a ON u.id = a.author_id
        GROUP BY u.id
    """)
    for uid, name, dept, count in cur.fetchall():
        print(f"  User({name}, {dept}) -> {count} articles")

    # Serializer: конвертация модели в JSON
    print("\nПаттерн Serializer (модель -> JSON):")

    def serialize(model_class, row, columns):
        """Конвертирует строку БД в словарь"""
        return dict(zip(columns, row))

    cur.execute("SELECT id, name, email, department FROM orm_users")
    for row in cur.fetchall():
        user_dict = serialize(User, row, User._columns.keys())
        print(f"  {json.dumps(user_dict, ensure_ascii=False)}")

    # Фильтрация через ORM
    print("\nФильтрация:")
    cur.execute("SELECT name, department FROM orm_users WHERE department = ?", ("Engineering",))
    engineers = cur.fetchall()
    print(f"  Engineers: {[name for name, _ in engineers]}")

    conn.close()


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    demo_sql_basics()
    print()
    demo_indexing()
    print()
    demo_connection_pooling()
    print()
    demo_orm_patterns()
