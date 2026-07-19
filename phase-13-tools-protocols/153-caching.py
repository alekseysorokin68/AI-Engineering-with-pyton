"""153 — Caching Strategies: Redis concepts, cache invalidation, TTL

Темы:
  1. Cache Patterns (cache-aside, read-through, write-through, write-behind)
  2. Eviction Policies (LRU, LFU, TTL, random)
  3. Cache Invalidation (time-based, event-based, versioning)
  4. Distributed Caching (consistent hashing, replication, hotspots)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import datetime

random.seed(42)


# ==========================================================================
# Демо 1 — Паттерны кэширования: cache-aside, read-through, write-through, write-behind
# ==========================================================================
def demo_cache_patterns():
    """Демонстрация четырёх основных паттернов кэширования."""

    print("=" * 70)
    print("Демо 1: Паттерны кэширования")
    print("=" * 70)

    # --- Подпример 1: Cache-Aside (ленивое кэширование) ---
    # Приложение сначала ищет данные в кэше, потом в БД
    # При записи: сначала пишем в БД, потом инвалидируем кэш
    print("\n--- 1.1 Cache-Aside (ленивое кэширование) ---")

    # Имитация «базы данных» — источник правды
    database = {
        "user:1": {"name": "Алиса", "email": "alice@example.com", "score": 95},
        "user:2": {"name": "Борис", "email": "boris@example.com", "score": 87},
        "user:3": {"name": "Вера", "email": "vera@example.com",  "score": 72},
    }
    # Кэш — промежуточное хранилище
    cache = {}

    def db_read(key):
        """Чтение из «базы данных» (медленная операция)."""
        time.sleep(0.01)  # имитация задержки сети/диска
        return database.get(key)

    def cache_aside_read(key):
        """
        Паттерн Cache-Aside:
        1. Проверяем кэш
        2. Если нет — читаем из БД
        3. Записываем в кэш
        """
        # Шаг 1: ищем в кэше
        if key in cache:
            print(f"  Cache-HIT для '{key}': {cache[key]}")
            return cache[key]
        # Шаг 2: кэш-промах — читаем из БД
        value = db_read(key)
        if value is not None:
            cache[key] = value  # Шаг 3: заполняем кэш
            print(f"  Cache-MISS для '{key}' → загружено из БД → кэшировано")
        return value

    # Попытка чтения — кэш пуст, данные загрузятся из БД
    result = cache_aside_read("user:1")
    print(f"  Результат: {result}")

    # Повторное чтение — данные уже в кэше (быстрее!)
    result = cache_aside_read("user:1")
    print(f"  Результат: {result}")

    # Инвалидация при записи — обновляем БД и удаляем из кэша
    def cache_aside_write(key, value):
        """Запись с инвалидацией кэша."""
        database[key] = value
        cache.pop(key, None)  # инвалидация — удаляем устаревшую запись
        print(f"  Запись '{key}' в БД + инвалидация кэша")

    cache_aside_write("user:1", {"name": "Алиса", "email": "alice@new.com", "score": 98})
    result = cache_aside_read("user:1")
    print(f"  Результат после обновления: {result}")

    # --- Подпример 2: Read-Through (кэш сам загружает данные) ---
    print("\n--- 1.2 Read-Through (сквозное чтение) ---")

    class ReadThroughCache:
        """Кэш с автоматической загрузкой при промахе."""
        def __init__(self, loader):
            self._store = {}
            self._loader = loader  # функция-загрузчик из БД
            self._misses = 0
            self._hits = 0

        def get(self, key):
            if key in self._store:
                self._hits += 1
                return self._store[key]
            # Кэш сам загружает данные через loader
            value = self._loader(key)
            self._store[key] = value
            self._misses += 1
            return value

        def stats(self):
            total = self._hits + self._misses
            rate = (self._hits / total * 100) if total else 0
            return {"hits": self._hits, "misses": self._misses, "hit_rate": rate}

    def slow_loader(key):
        """Имитация загрузки данных из БД."""
        return database.get(key, {"error": "not found"})

    rtc = ReadThroughCache(slow_loader)
    # Первый доступ — промах, данные загружаются автоматически
    for uid in ["user:1", "user:2", "user:1", "user:3", "user:1", "user:2"]:
        val = rtc.get(uid)
        print(f"  get('{uid}') → имя={val.get('name', 'N/A')}")

    stats = rtc.stats()
    print(f"  Статистика: hit={stats['hits']}, miss={stats['misses']}, "
          f"hit_rate={stats['hit_rate']:.1f}%")

    # --- Подпример 3: Write-Through (синхронная запись в кэш и БД) ---
    print("\n--- 1.3 Write-Through (сквозная запись) ---")

    class WriteThroughCache:
        """Кэш, записывающий данные одновременно в кэш и БД."""
        def __init__(self):
            self._cache = {}
            self._db = {}
            self._write_count = 0

        def put(self, key, value):
            # Синхронная запись: кэш и БД обновляются одновременно
            self._cache[key] = value
            self._db[key] = value
            self._write_count += 1
            print(f"  PUT '{key}': кэш и БД обновлены одновременно")

        def get(self, key):
            # Чтение всегда из кэша (быстро)
            return self._cache.get(key)

    wtc = WriteThroughCache()
    wtc.put("item:1", {"price": 100, "name": "Ноутбук"})
    wtc.put("item:2", {"price": 50, "name": "Мышь"})
    print(f"  Чтение item:1 → {wtc.get('item:1')}")
    print(f"  Всего записей: {wtc._write_count}")

    # --- Подпример 4: Write-Behind (отложенная запись в БД) ---
    print("\n--- 1.4 Write-Behind (отложенная запись) ---")

    class WriteBehindCache:
        """Кэш с отложенной синхронизацией с БД."""
        def __init__(self):
            self._cache = {}
            self._pending = []  # очередь на запись в БД

        def put(self, key, value):
            # Быстрая запись только в кэш
            self._cache[key] = value
            self._pending.append((key, value, datetime.datetime.now()))
            print(f"  PUT '{key}': записано в кэш (БД обновится позже)")

        def flush_to_db(self):
            """Фоновая синхронизация: записываем все накопленные данные."""
            print(f"  FLUSH: синхронизация {len(self._pending)} записей в БД...")
            for key, value, ts in self._pending:
                print(f"    → '{key}' = {value} (записано в {ts.strftime('%H:%M:%S')})")
            self._pending.clear()
            print("  БД синхронизирована")

    wb = WriteBehindCache()
    wb.put("sensor:temp", 23.5)
    wb.put("sensor:hum", 65)
    wb.put("sensor:press", 760)
    print(f"  Записей в очереди: {len(wb._pending)}")
    wb.flush_to_db()
    print(f"  Записей в очереди после flush: {len(wb._pending)}")

    # Итоговая сводка паттернов
    print("\n--- Сравнение паттернов ---")
    patterns = {
        "Cache-Aside":   "Приложение управляет кэшем; гибкость, но больше кода",
        "Read-Through":  "Кэш сам загружает данные; проще для приложения",
        "Write-Through": "Синхронная запись; консистентность, но медленнее",
        "Write-Behind":  "Отложенная запись; быстро, но риск потери данных",
    }
    for name, desc in patterns.items():
        print(f"  {name:16s} — {desc}")


# ==========================================================================
# Демо 2 — Политики вытеснения: LRU, LFU, TTL, random
# ==========================================================================
def demo_eviction_policies():
    """Демонстрация стратегий вытеснения элементов из кэша при нехватке места."""

    print("\n" + "=" * 70)
    print("Демо 2: Политики вытеснения (Eviction Policies)")
    print("=" * 70)

    # --- Подпример 1: LRU (Least Recently Used) ---
    print("\n--- 2.1 LRU — Least Recently Used (наименее недавно использованный) ---")

    class LRUCache:
        """Кэш с политикой вытеснения LRU."""
        def __init__(self, capacity):
            self._capacity = capacity
            self._order = collections.OrderedDict()  # ключ → значение, с порядком доступа

        def get(self, key):
            if key in self._order:
                # Перемещаем в конец — помечаем как «недавно использованный»
                self._order.move_to_end(key)
                return self._order[key]
            return None

        def put(self, key, value):
            if key in self._order:
                self._order.move_to_end(key)
            self._order[key] = value
            if len(self._order) > self._capacity:
                # Вытесняем самый «старый» элемент (первая позиция)
                evicted_key, evicted_val = self._order.popitem(last=False)
                print(f"  LRU вытеснил: {evicted_key}={evicted_val}")
            print(f"  PUT {key}={value} | кэш: {list(self._order.keys())}")

    print("  Ёмкость кэша = 3 элемента")
    lru = LRUCache(3)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    # Обращаемся к 'a' — он перемещается в конец
    print(f"  GET a={lru.get('a')} (обновлён порядок)")
    # Добавляем 'd' — вытесняется 'b' (самый старый)
    lru.put("d", 4)

    # --- Подпример 2: LFU (Least Frequently Used) ---
    print("\n--- 2.2 LFU — Least Frequently Used (наименее частый) ---")

    class LFUCache:
        """Кэш с политикой вытеснения LFU."""
        def __init__(self, capacity):
            self._capacity = capacity
            self._store = {}      # ключ → значение
            self._freq = {}       # ключ → частота обращений
            self._min_freq = 0

        def get(self, key):
            if key not in self._store:
                return None
            self._freq[key] = self._freq.get(key, 0) + 1
            # Обновляем минимальную частоту
            self._min_freq = min(self._freq.values()) if self._freq else 0
            return self._store[key]

        def put(self, key, value):
            if self._capacity <= 0:
                return
            if key in self._store:
                self._store[key] = value
                self._freq[key] = self._freq.get(key, 0) + 1
                return
            if len(self._store) >= self._capacity:
                # Вытесняем элемент с минимальной частотой
                min_keys = [k for k, f in self._freq.items() if f == self._min_freq]
                evicted_key = min_keys[0]
                del self._store[evicted_key]
                del self._freq[evicted_key]
                print(f"  LFU вытеснил: {evicted_key} (частота={self._min_freq})")
            self._store[key] = value
            self._freq[key] = 1
            self._min_freq = 1
            print(f"  PUT {key}={value} | частоты: {dict(self._freq)}")

    print("  Ёмкость кэша = 3 элемента")
    lfu = LFUCache(3)
    lfu.put("x", 10)
    lfu.put("y", 20)
    lfu.put("z", 30)
    # Часто обращаемся к 'x'
    for _ in range(5):
        lfu.get("x")
    lfu.get("y")  # y обращались 1 раз
    # Добавляем новый — вытеснится y (минимальная частота)
    lfu.put("w", 40)

    # --- Подпример 3: TTL (Time-To-Live) ---
    print("\n--- 2.3 TTL — Time-To-Live (время жизни) ---")

    class TTLCache:
        """Кэш с автоматическим удалением устаревших записей."""
        def __init__(self, default_ttl=2.0):
            self._store = {}  # ключ → (значение, время_создания)
            self._default_ttl = default_ttl

        def put(self, key, value, ttl=None):
            self._store[key] = (value, time.time(), ttl or self._default_ttl)

        def get(self, key):
            if key not in self._store:
                return None
            value, created, ttl = self._store[key]
            age = time.time() - created
            if age > ttl:
                # Запись устарела — удаляем
                del self._store[key]
                print(f"  TTL: '{key}' устарел (возраст={age:.1f}s > TTL={ttl}s)")
                return None
            print(f"  TTL: '{key}' валиден (возраст={age:.2f}s < TTL={ttl}s)")
            return value

        def cleanup(self):
            """Очистка всех устаревших записей."""
            now = time.time()
            expired = [k for k, (v, c, t) in self._store.items() if now - c > t]
            for k in expired:
                del self._store[k]
            print(f"  Cleanup: удалено {len(expired)} устаревших записей")

    ttl_cache = TTLCache(default_ttl=0.1)  # TTL = 0.1 секунды
    ttl_cache.put("session:abc", "user_data")
    ttl_cache.put("temp_key", "temporary", ttl=0.05)
    time.sleep(0.02)
    print(f"  Чтение session:abc: {ttl_cache.get('session:abc')}")
    print(f"  Чтение temp_key: {ttl_cache.get('temp_key')}")  # может быть устаревшим
    time.sleep(0.1)
    print(f"  После ожидания:")
    print(f"  session:abc: {ttl_cache.get('session:abc')}")
    ttl_cache.cleanup()
    print(f"  Осталось записей: {len(ttl_cache._store)}")

    # --- Подпример 4: Random Eviction ---
    print("\n--- 2.4 Random Eviction (случайное вытеснение) ---")

    class RandomCache:
        """Кэш со случайным вытеснением — просто и эффективно."""
        def __init__(self, capacity):
            self._capacity = capacity
            self._store = {}

        def put(self, key, value):
            if len(self._store) >= self._capacity:
                # Выбираем случайный ключ для вытеснения
                all_keys = list(self._store.keys())
                evicted_key = random.choice(all_keys)
                del self._store[evicted_key]
                print(f"  Random вытеснил: {evicted_key}")
            self._store[key] = value

        def get(self, key):
            return self._store.get(key)

    print("  Ёмкость кэша = 4 элемента")
    rc = RandomCache(4)
    for i in range(8):
        rc.put(f"item:{i}", i * 10)
    print(f"  Финальное состояние: {dict(rc._store)}")

    # Формула: вероятность вытеснения каждого элемента = 1/N (равномерная)
    print("\n--- Формулы ---")
    print("  LRU: вытесняется элемент с максимальным давлением последнего доступа")
    print("  LFU: вытесняется элемент с минимальной частотой обращений")
    print("  TTL: вытесняется при age(key) > TTL")
    print("  Random: P(вытеснения) = 1/N для каждого элемента")


# ==========================================================================
# Демо 3 — Инвалидация кэша: time-based, event-based, versioning
# ==========================================================================
def demo_cache_invalidation():
    """Демонстрация стратегий инвалидации кэша — когда удалять устаревшие данные."""

    print("\n" + "=" * 70)
    print("Демо 3: Инвалидация кэша (Cache Invalidation)")
    print("=" * 70)

    # --- Подпример 1: Time-based invalidation ---
    print("\n--- 3.1 Time-based инвалидация (по времени) ---")

    class TimeBasedCache:
        """Кэш с фиксированным TTL для каждого ключа."""
        def __init__(self):
            self._store = {}  # ключ → (значение, timestamp)
            self._ttl_map = {}  # ключ → ttl в секундах

        def put(self, key, value, ttl=5.0):
            self._store[key] = (value, time.time())
            self._ttl_map[key] = ttl

        def get(self, key):
            if key not in self._store:
                return None, "MISS"
            value, ts = self._store[key]
            age = time.time() - ts
            ttl = self._ttl_map.get(key, 5.0)
            if age > ttl:
                del self._store[key]
                del self._ttl_map[key]
                return None, f"EXPIRED (age={age:.2f}s > ttl={ttl}s)"
            remaining = ttl - age
            return value, f"HIT (осталось {remaining:.2f}s)"

    tbc = TimeBasedCache()
    tbc.put("news:top", "Главные новости дня", ttl=0.15)
    tbc.put("config:db", "postgresql://localhost", ttl=10.0)  # долгоживущая

    # Сразу после записи — данные валидны
    val, status = tbc.get("news:top")
    print(f"  news:top → {status}: {val}")

    # После ожидания — запись устарела
    time.sleep(0.2)
    val, status = tbc.get("news:top")
    print(f"  news:top (после ожидания) → {status}")

    # Конфигурация всё ещё жива
    val, status = tbc.get("config:db")
    print(f"  config:db → {status}")

    # --- Подпример 2: Event-based invalidation ---
    print("\n--- 3.2 Event-based инвалидация (по событиям) ---")

    class EventDrivenCache:
        """Кэш, инвалидируемый событиями (например, обновление данных в БД)."""
        def __init__(self):
            self._store = {}
            self._subscribers = collections.defaultdict(list)

        def put(self, key, value):
            self._store[key] = value

        def get(self, key):
            return self._store.get(key)

        def invalidate(self, key):
            """Инвалидация конкретного ключа."""
            if key in self._store:
                del self._store[key]
                # Уведомляем подписчиков об инвалидации
                for callback in self._subscribers.get(key, []):
                    callback(key)
                print(f"  ИНВАЛИДАЦИЯ: '{key}' удалён из кэша")

        def subscribe(self, key, callback):
            """Подписка на события инвалидации."""
            self._subscribers[key].append(callback)

    edc = EventDrivenCache()
    edc.put("user:5", {"name": "Елена", "role": "admin"})
    edc.put("user:5:permissions", ["read", "write", "delete"])

    # Подписываемся на инвалидацию
    def on_user_invalidated(key):
        print(f"  → Обработчик: перезагружаем данные для '{key}'")

    edc.subscribe("user:5", on_user_invalidated)
    edc.subscribe("user:5:permissions", on_user_invalidated)

    print(f"  До инвалидации: user:5 = {edc.get('user:5')}")
    edc.invalidate("user:5")
    edc.invalidate("user:5:permissions")
    print(f"  После инвалидации: user:5 = {edc.get('user:5')}")

    # --- Подпример 3: Version-based invalidation ---
    print("\n--- 3.3 Version-based инвалидация (по версии) ---")

    class VersionedCache:
        """Кэш с версионированием — клиенты получают актуальную версию."""
        def __init__(self):
            self._versions = {}  # ключ → (значение, версия)
            self._global_version = 0

        def put(self, key, value):
            self._global_version += 1
            self._versions[key] = (value, self._global_version)

        def get(self, key, min_version=0):
            """Получение данных с проверкой версии."""
            if key not in self._versions:
                return None, 0
            value, version = self._versions[key]
            if version < min_version:
                return None, version  # данные устарели
            return value, version

        def bump_version(self):
            """Глобальное обновление версии — все данные считаются устаревшими."""
            self._global_version += 1
            print(f"  Версия обновлена до {self._global_version}")

    vc = VersionedCache()
    vc.put("product:1", {"name": "Телефон", "price": 29990})
    print(f"  Версия product:1: v{vc.get('product:1')[1]}")

    # Клиент с версией 0 получает актуальные данные
    val, ver = vc.get("product:1", min_version=0)
    print(f"  Клиент v0 запрашивает → v{ver}: {val}")

    # Обновляем цену — появляется новая версия
    vc.put("product:1", {"name": "Телефон", "price": 27990})
    val, ver = vc.get("product:1", min_version=0)
    print(f"  После обновления → v{ver}: {val}")

    # Глобальное обновление версии
    vc.bump_version()
    val, ver = vc.get("product:1", min_version=2)
    if val is None:
        print(f"  Клиент v2 запрашивает → данные устарели (v{ver} < v2)")
    else:
        print(f"  Клиент v2 запрашивает → v{ver}: {val}")

    # --- Подпример 4: Cache stampede prevention ---
    print("\n--- 3.4 Защита от Cache Stampede (лавины запросов) ---")

    class StampedeProtectedCache:
        """Кэш с защитой от лавинного обновления (mutex lock)."""
        def __init__(self):
            self._store = {}
            self._locks = {}  # ключ → True если кто-то уже загружает

        def get(self, key, loader=None):
            if key in self._store:
                return self._store[key]
            if key in self._locks:
                print(f"  [{key}] Заблокирован — другой поток загружает")
                return None
            # Первый «победитель» загружает данные
            self._locks[key] = True
            try:
                if loader:
                    value = loader(key)
                    self._store[key] = value
                    print(f"  [{key}] Загружено и закэшировано")
                    return value
            finally:
                self._locks.pop(key, None)

    spc = StampedeProtectedCache()
    def expensive_load(key):
        """Дорогая загрузка из БД (имитация)."""
        time.sleep(0.05)
        return f"data_for_{key}"

    # 3 «клиента» запрашивают один и тот же ключ одновременно
    for i in range(3):
        result = spc.get("hot_key", loader=expensive_load)
        if result:
            print(f"  Клиент {i+1} получил: {result}")
        else:
            print(f"  Клиент {i+1}: данные загружаются...")

    print("\n--- Формула вероятности stampede ---")
    print("  P(stampede) = λ^k / k! * e^(-λ), где λ — средний RPS на ключ")
    print("  Решение: мьютекс / singleflight / jitter на TTL")


# ==========================================================================
# Демо 4 — Распределённое кэширование: consistent hashing, replication, hotspots
# ==========================================================================
def demo_distributed_caching():
    """Демонстрация принципов распределённого кэширования."""

    print("\n" + "=" * 70)
    print("Демо 4: Распределённое кэширование")
    print("=" * 70)

    # --- Подпример 1: Consistent Hashing ---
    print("\n--- 4.1 Consistent Hashing (последовательное хэширование) ---")

    class ConsistentHashRing:
        """Кольцо хэширования для распределения данных по узлам."""
        def __init__(self, nodes, virtual_nodes=100):
            self._ring = {}  # hash → node_name
            self._sorted_keys = []
            self._virtual_nodes = virtual_nodes

            for node in nodes:
                self.add_node(node)

        def _hash(self, key):
            """Хэш-функция для размещения на кольце."""
            h = hashlib.md5(key.encode()).hexdigest()
            return int(h, 16)

        def add_node(self, node):
            """Добавление узла с виртуальными нодами."""
            for i in range(self._virtual_nodes):
                virtual_key = f"{node}:v{i}"
                h = self._hash(virtual_key)
                self._ring[h] = node
                self._sorted_keys.append(h)
            self._sorted_keys.sort()

        def remove_node(self, node):
            """Удаление узла из кольца."""
            for i in range(self._virtual_nodes):
                virtual_key = f"{node}:v{i}"
                h = self._hash(virtual_key)
                if h in self._ring:
                    del self._ring[h]
            self._sorted_keys = sorted(k for k in self._sorted_keys if k in self._ring)

        def get_node(self, key):
            """Определение узла для заданного ключа."""
            if not self._ring:
                return None
            h = self._hash(key)
            # Бинарный поиск первого узла >= хэша ключа
            for ring_key in self._sorted_keys:
                if ring_key >= h:
                    return self._ring[ring_key]
            # Если ключ «за» последним узлом — возвращаем первый
            return self._ring[self._sorted_keys[0]]

    # Создаём кольцо с 3 узлами
    ring = ConsistentHashRing(["node-A", "node-B", "node-C"], virtual_nodes=50)

    # Распределяем 10 ключей по узлам
    print("  Распределение 10 ключей по 3 узлам:")
    distribution = collections.Counter()
    for i in range(10):
        key = f"user:{i}"
        node = ring.get_node(key)
        distribution[node] += 1
        print(f"    {key} → {node}")

    print(f"  Распределение: {dict(distribution)}")

    # Демонстрация: при добавлении узла меняется только ~1/N ключей
    print("\n  Добавляем node-D в кольцо...")
    ring.add_node("node-D")
    changes = 0
    for i in range(100):
        key = f"key:{i}"
        old_node = ring.get_node(key)  # уже с node-D, но покажем принцип
        if old_node == "node-D":
            changes += 1
    print(f"  ~{changes} из 100 ключей будут перенесены (≈25% при 4 узлах)")

    # --- Подпример 2: Replication ---
    print("\n--- 4.2 Репликация (Replication) ---")

    class ReplicatedCache:
        """Кэш с репликацией данных на несколько узлов."""
        def __init__(self, num_replicas=3):
            self._primary = {}  # основное хранилище
            self._replicas = [{} for _ in range(num_replicas)]
            self._num_replicas = num_replicas

        def put(self, key, value):
            # Запись в.primary
            self._primary[key] = value
            # Репликация на все реплики
            for i, replica in enumerate(self._replicas):
                replica[key] = value
            print(f"  PUT '{key}': записано в primary + {self._num_replicas} реплик")

        def get(self, key, allow_stale=False):
            """Чтение с проверкой консистентности."""
            if key in self._primary:
                return self._primary[key], "primary"
            # Если primary недоступен, читаем из реплик
            for i, replica in enumerate(self._replicas):
                if key in replica:
                    return replica.get(key), f"replica-{i}"
            return None, "not_found"

        def simulate_primary_failure(self):
            """Имитация отказа primary — данные берутся из реплики."""
            self._primary.clear()
            print("  ⚠ PRIMARY ОТКАЗАЛ — переключение на реплику")

    rc = ReplicatedCache(num_replicas=2)
    rc.put("critical:data", "важные данные")
    rc.put("session:xyz", "данные сессии")

    # Чтение из primary
    val, source = rc.get("critical:data")
    print(f"  Чтение: {val} (источник: {source})")

    # Имитация отказа primary
    rc.simulate_primary_failure()
    val, source = rc.get("critical:data")
    print(f"  Чтение после отказа: {val} (источник: {source})")

    # --- Подпример 3: Hotspot Detection ---
    print("\n--- 4.3 Обнаружение горячих клавиш (Hotspot Detection) ---")

    class HotspotDetector:
        """Детектор горячих клавиш с автоматическим кэшированием."""
        def __init__(self, threshold=10, window=5.0):
            self._access_log = collections.defaultdict(list)  # ключ → [timestamps]
            self._hot_cache = {}  # кэш для горячих клавиш
            self._threshold = threshold  # порог обращений за window
            self._window = window

        def access(self, key):
            now = time.time()
            self._access_log[key].append(now)
            # Удаляем старые записи
            self._access_log[key] = [
                t for t in self._access_log[key] if now - t < self._window
            ]
            freq = len(self._access_log[key])
            is_hot = freq >= self._threshold
            if is_hot and key not in self._hot_cache:
                self._hot_cache[key] = True
                print(f"  🔥 HOTSPOT: '{key}' — {freq} обращений за {self._window}с")
            return is_hot, freq

        def get_hot_keys(self):
            """Возвращает список горячих клавиш."""
            return list(self._hot_cache.keys())

    hd = HotspotDetector(threshold=3, window=10.0)
    # Имитация нормальной нагрузки
    for _ in range(2):
        hd.access("normal:key")
    # Имитация горячей клавиши
    for _ in range(5):
        is_hot, freq = hd.access("hot:product:1")
        if is_hot:
            print(f"  → product:1 кэширован (частота={freq})")

    print(f"  Горячие ключи: {hd.get_hot_keys()}")

    # --- Подпример 4: Distributed Lock ---
    print("\n--- 4.4 Распределённая блокировка (Distributed Lock) ---")

    class DistributedLock:
        """Имитация распределённой блокировки (аналог Redis SETNX)."""
        def __init__(self):
            self._locks = {}  # ключ → (holder_id, expiry_time)

        def acquire(self, key, holder_id, ttl=1.0):
            """Попытка захватить блокировку."""
            now = time.time()
            # Проверяем, не занята ли блокировка
            if key in self._locks:
                current_holder, expiry = self._locks[key]
                if now < expiry:
                    print(f"  LOCK '{key}' занят {current_holder} до {expiry:.1f}")
                    return False
                else:
                    print(f"  LOCK '{key}' устарел — освобождаем")

            # Захватываем блокировку
            self._locks[key] = (holder_id, now + ttl)
            print(f"  LOCK '{key}' захвачен {holder_id} (TTL={ttl}s)")
            return True

        def release(self, key, holder_id):
            """Освобождение блокировки (только если你是 владельцем)."""
            if key in self._locks:
                current_holder, _ = self._locks[key]
                if current_holder == holder_id:
                    del self._locks[key]
                    print(f"  UNLOCK '{key}' освобождён {holder_id}")
                    return True
                else:
                    print(f"  UNLOCK '{key}' — ошибка: {holder_id} не владелец")
            return False

        def is_locked(self, key):
            """Проверка статуса блокировки."""
            if key not in self._locks:
                return False
            _, expiry = self._locks[key]
            if time.time() >= expiry:
                del self._locks[key]
                return False
            return True

    dl = DistributedLock()
    # Два клиента пытаются захватить одну блокировку
    print("  Клиент A пытается захватить блокировку...")
    dl.acquire("resource:db_migrate", "client-A", ttl=0.2)
    print("  Клиент B пытается захватить ту же блокировку...")
    dl.acquire("resource:db_migrate", "client-B", ttl=0.2)

    # Ожидаем истечения блокировки
    time.sleep(0.3)
    print("  После истечения TTL:")
    dl.acquire("resource:db_migrate", "client-B", ttl=0.2)

    # Итоговая сводка
    print("\n--- Ключевые концепции ---")
    concepts = [
        "Consistent Hashing: при добавлении/удалении узла перехэшируется ~1/N ключей",
        "Репликация: чтение с fallback, запись — quorum или async",
        "Hotspot: автоматическое кэширование при превышении порога частоты",
        "Distributed Lock: SETNX + TTL для мьютекса в распределённой системе",
    ]
    for c in concepts:
        print(f"  • {c}")


# ==========================================================================
# Точка входа
# ==========================================================================
if __name__ == "__main__":
    print("УРОК 153: CACHING STRATEGIES")
    print("Темы: Cache Patterns, Eviction Policies, Cache Invalidation, Distributed Caching")
    print("=" * 70)

    demo_cache_patterns()
    demo_eviction_policies()
    demo_cache_invalidation()
    demo_distributed_caching()

    print("\n" + "=" * 70)
    print("Урок завершён. Все 4 демо выполнены успешно.")
    print("=" * 70)
