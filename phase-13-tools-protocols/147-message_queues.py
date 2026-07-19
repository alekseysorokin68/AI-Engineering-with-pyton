"""147 — Message Queues: RabbitMQ, Kafka concepts, async patterns

Темы:
  1. Основы очередей (производитель, потребитель, брокер, подтверждение)
  2. Паттерн Pub/Sub (топики, маршрутизация, fanout)
  3. Потоковая обработка событий (концепции Kafka: топики, партиции, смещения)
  4. Асинхронные паттерны (очереди задач, повторные попытки, мёртвая очередь)

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
# Демо 1: Основы очередей — производитель, потребитель, брокер, подтверждение
# =============================================================================

def demo_queue_basics():
    print("=" * 70)
    print("Демо 1: Основы очередей — производитель, потребитель, брокер, подтверждение")
    print("=" * 70)

    # --- 1.1 Моделирование простой очереди сообщений ---
    print("\n--- 1.1 Простая очередь сообщений ---")

    # Очередь — это структура данных, в которой сообщения хранятся
    # до тех пор, пока потребитель их не обработает
    queue = []

    # Производитель (producer) добавляет сообщения в очередь
    messages = ["order_001", "order_002", "order_003", "order_004"]
    for msg in messages:
        queue.append(msg)
        print(f"  [Производитель] Отправлено: {msg}")

    # Потребитель (consumer) извлекает сообщения из очереди
    print()
    while queue:
        msg = queue.pop(0)  # FIFO — первый вошёл, первый вышел
        print(f"  [Потребитель] Обработано: {msg}")

    print(f"  Очередь пуста: {len(queue) == 0}")

    # --- 1.2 Моделирование брокера с маршрутизацией ---
    print("\n--- 1.2 Брокер сообщений с маршрутизацией ---")

    # Брокер (broker) — это промежуточное звено между производителями
    # и потребителями. Он хранит сообщения и маршрутизирует их.
    class MessageBroker:
        def __init__(self):
            self.queues = collections.defaultdict(list)
            self.consumers = collections.defaultdict(list)

        def declare_queue(self, queue_name):
            """Объявляем очередь"""
            self.queues[queue_name] = []

        def bind_consumer(self, queue_name, consumer_id):
            """Привязываем потребителя к очереди"""
            self.consumers[queue_name].append(consumer_id)
            print(f"  [Брокер] Потребитель {consumer_id} привязан к очереди '{queue_name}'")

        def publish(self, queue_name, message):
            """Публикуем сообщение в очередь"""
            self.queues[queue_name].append(message)
            print(f"  [Брокер] Сообщение '{message}' добавлено в '{queue_name}'")

        def consume(self, queue_name):
            """Потребляем сообщение из очереди"""
            if self.queues[queue_name]:
                msg = self.queues[queue_name].pop(0)
                return msg
            return None

    broker = MessageBroker()
    broker.declare_queue("orders")
    broker.declare_queue("notifications")

    broker.bind_consumer("orders", "worker_1")
    broker.bind_consumer("notifications", "notifier_1")

    broker.publish("orders", {"id": 1, "item": "laptop"})
    broker.publish("orders", {"id": 2, "item": "mouse"})
    broker.publish("notifications", {"type": "email", "to": "user@mail.com"})

    # Потребление сообщений
    for queue_name in ["orders", "notifications"]:
        print()
        while True:
            msg = broker.consume(queue_name)
            if msg is None:
                break
            print(f"  [Потребитель] Из '{queue_name}': {msg}")

    # --- 1.3 Подтверждение доставки (acknowledgment) ---
    print("\n--- 1.3 Подтверждение доставки (ACK/NACK) ---")

    # Подтверждение гарантирует, что сообщение обработано успешно.
    # Если потребитель не отправил ACK, сообщение возвращается в очередь.
    class AckQueue:
        def __init__(self):
            self.messages = []
            self.pending = {}  # сообщения, ожидающие подтверждения

        def enqueue(self, msg):
            self.messages.append({"id": str(uuid.uuid4())[:8], "data": msg, "retries": 0})

        def dequeue(self):
            if self.messages:
                msg = self.messages.pop(0)
                self.pending[msg["id"]] = msg
                return msg
            return None

        def ack(self, msg_id):
            """Подтверждение успешной обработки"""
            if msg_id in self.pending:
                del self.pending[msg_id]
                return True
            return False

        def nack(self, msg_id):
            """Отклонение — сообщение возвращается в очередь"""
            if msg_id in self.pending:
                msg = self.pending.pop(msg_id)
                msg["retries"] += 1
                if msg["retries"] < 3:
                    self.messages.append(msg)
                    return "requeued"
                return "dead_letter"
            return None

    ack_queue = AckQueue()
    ack_queue.enqueue("process_payment")
    ack_queue.enqueue("send_email")
    ack_queue.enqueue("update_inventory")

    # Обработка с успешным ACK
    msg1 = ack_queue.dequeue()
    print(f"  Извлечено: {msg1['data']} (id={msg1['id']})")
    ack_queue.ack(msg1["id"])
    print(f"  ACK отправлен для {msg1['id']} — сообщение удалено")

    # Обработка с NACK (имитируем ошибку)
    msg2 = ack_queue.dequeue()
    print(f"\n  Извлечено: {msg2['data']} (id={msg2['id']})")
    result = ack_queue.nack(msg2["id"])
    print(f"  NACK отправлен — результат: {result} (попытка {msg2['retries']+1})")

    # Повторная обработка после NACK
    msg2_retry = ack_queue.dequeue()
    print(f"\n  Повторно извлечено: {msg2_retry['data']} (id={msg2_retry['id']})")
    ack_queue.ack(msg2_retry["id"])
    print(f"  ACK отправлен — обработка завершена")

    # --- 1.4 Приоритетная очередь ---
    print("\n--- 1.4 Приоритетная очередь ---")

    # Приоритетная очередь позволяет обрабатывать важные сообщения первыми
    import heapq

    class PriorityMsg:
        def __init__(self, priority, data):
            self.priority = priority
            self.data = data

        def __lt__(self, other):
            return self.priority < other.priority  # меньше = выше приоритет

        def __repr__(self):
            return f"P{self.priority}:{self.data}"

    priority_queue = []
    heapq.heappush(priority_queue, PriorityMsg(3, "low_priority_task"))
    heapq.heappush(priority_queue, PriorityMsg(1, "critical_alert"))
    heapq.heappush(priority_queue, PriorityMsg(2, "normal_task"))
    heapq.heappush(priority_queue, PriorityMsg(1, "emergency_shutdown"))

    print("  Сообщения в приоритетной очереди (P1 = критический):")
    while priority_queue:
        msg = heapq.heappop(priority_queue)
        print(f"    → {msg}")


# =============================================================================
# Демо 2: Паттерн Pub/Sub — топики, маршрутизация, fanout
# =============================================================================

def demo_pubsub():
    print("\n\n" + "=" * 70)
    print("Демо 2: Паттерн Pub/Sub — топики, маршрутизация, fanout")
    print("=" * 70)

    # --- 2.1 Простая шина событий (Event Bus) ---
    print("\n--- 2.1 Простая шина событий (Event Bus) ---")

    class EventBus:
        def __init__(self):
            self.subscribers = collections.defaultdict(list)
            self.event_log = []

        def subscribe(self, event_type, callback):
            """Подписка на тип события"""
            self.subscribers[event_type].append(callback)

        def publish(self, event_type, data):
            """Публикация события всем подписчикам"""
            self.event_log.append({"type": event_type, "data": data})
            for callback in self.subscribers[event_type]:
                callback(event_type, data)

    bus = EventBus()

    # Подписчики (consumers)
    def logger(event_type, data):
        print(f"    [Логгер] Записано: {event_type} → {data}")

    def email_notifier(event_type, data):
        print(f"    [Email] Отправлено уведомление: {data['message']}")

    def analytics(event_type, data):
        print(f"    [Аналитика] Событие '{event_type}' с размером данных: {len(str(data))} байт")

    bus.subscribe("order.created", logger)
    bus.subscribe("order.created", email_notifier)
    bus.subscribe("order.created", analytics)
    bus.subscribe("payment.processed", logger)

    # Публикация событий — все подписчики на "order.created" получают событие
    print("\n  Публикация 'order.created':")
    bus.publish("order.created", {"id": 42, "total": 1500, "message": "Заказ оформлен"})

    print("\n  Публикация 'payment.processed':")
    bus.publish("payment.processed", {"order_id": 42, "status": "success"})

    print(f"\n  Всего событий в логе: {len(bus.event_log)}")

    # --- 2.2 Маршрутизация по топикам (topic routing) ---
    print("\n--- 2.2 Маршрутизация по топикам ---")

    class TopicRouter:
        def __init__(self):
            self.routes = {}  # паттерн → список подписчиков
            self.messages_routed = 0

        def subscribe(self, pattern, handler):
            """Подписка по шаблону топика (с поддержкой * и #)"""
            self.routes[pattern] = self.routes.get(pattern, [])
            self.routes[pattern].append(handler)

        def publish(self, topic, message):
            """Публикация в топик с маршрутизацией по паттернам"""
            for pattern, handlers in self.routes.items():
                if self._match(pattern, topic):
                    for handler in handlers:
                        handler(topic, message)
                        self.messages_routed += 1

        def _match(self, pattern, topic):
            """Простое сопоставление паттернов: * — одно слово, # — любое кол-во"""
            p_parts = pattern.split(".")
            t_parts = topic.split(".")
            pi, ti = 0, 0
            while pi < len(p_parts) and ti < len(t_parts):
                if p_parts[pi] == "#":
                    return True  # # совпадает с оставшимися частями
                if p_parts[pi] == "*" or p_parts[pi] == t_parts[ti]:
                    pi += 1
                    ti += 1
                else:
                    return False
            return pi == len(p_parts) and ti == len(t_parts)

    router = TopicRouter()

    def log_handler(topic, msg):
        print(f"    [Лог] {topic}: {msg}")

    def alert_handler(topic, msg):
        print(f"    [Алерт] {topic}: {msg}")

    def debug_handler(topic, msg):
        print(f"    [Debug] {topic}: {msg}")

    # Подписка на разные паттерны
    router.subscribe("orders.#", log_handler)       # все события orders
    router.subscribe("orders.created", alert_handler) # только создание
    router.subscribe("*.error", debug_handler)        # все ошибки

    print("\n  Маршрутизация сообщений:")
    router.publish("orders.created", "Новый заказ #100")
    router.publish("orders.shipped", "Заказ #100 отправлен")
    router.publish("payments.error", "Ошибка оплаты")
    router.publish("users.login", "Вход пользователя")  # не маршрутизируется

    print(f"\n  Всего маршрутизировано: {router.messages_routed} сообщений")

    # --- 2.3 Fanout — рассылка всем подписчикам ---
    print("\n--- 2.3 Fanout — рассылка всем подписчикам ---")

    class FanoutExchange:
        """Обменник типа fanout — отправляет копию каждого сообщения
        всем привязанным очередям (как радиовещание)"""
        def __init__(self):
            self.queues = {}  # имя_очереди → список сообщений

        def declare_queue(self, name):
            self.queues[name] = []

        def bind(self, queue_name):
            """Привязка очереди к exchange"""
            if queue_name not in self.queues:
                self.declare_queue(queue_name)

        def publish(self, message):
            """Отправить сообщение во все привязанные очереди"""
            for qname in self.queues:
                self.queues[qname].append(message.copy())

    exchange = FanoutExchange()
    exchange.declare_queue("audit_log")
    exchange.declare_queue("real_time_dashboard")
    exchange.declare_queue("analytics_stream")

    # Публикация — сообщение попадает во ВСЕ очереди
    exchange.publish({"event": "user_signup", "user_id": "u_001", "plan": "pro"})
    exchange.publish({"event": "file_uploaded", "size_mb": 42})

    print("  Содержимое очередей после fanout-рассылки:")
    for qname, msgs in exchange.queues.items():
        print(f"    [{qname}]: {len(msgs)} сообщений — {msgs[0]['event']}")

    # --- 2.4 Dead Letter Exchange (DLX) ---
    print("\n--- 2.4 Dead Letter Exchange (DLX) ---")

    class DLXSystem:
        """Система с Dead Letter Exchange — сообщения, которые не удалось
        обработать после N попыток, попадают в отдельную 'мёртвую' очередь"""
        def __init__(self, max_retries=2):
            self.main_queue = []
            self.dlq = []  # dead letter queue
            self.max_retries = max_retries
            self.processed = []
            self.failed = []

        def enqueue(self, msg):
            self.main_queue.append({"data": msg, "retries": 0})

        def process(self):
            while self.main_queue:
                item = self.main_queue.pop(0)
                # Имитируем: нечётные id вызывают ошибку
                if item["data"]["id"] % 2 == 1 and item["retries"] == 0:
                    item["retries"] += 1
                    if item["retries"] < self.max_retries:
                        self.main_queue.append(item)
                        print(f"    Ошибка обработки {item['data']}, повтор (попытка {item['retries']})")
                    else:
                        self.dlq.append(item)
                        self.failed.append(item["data"])
                        print(f"    {item['data']} → Dead Letter Queue (все попытки исчерпаны)")
                else:
                    self.processed.append(item["data"])
                    print(f"    Успешно: {item['data']}")

    dlx = DLXSystem(max_retries=2)
    for i in range(1, 7):
        dlx.enqueue({"id": i, "task": f"job_{i}"})

    print("  Обработка с DLX:")
    dlx.process()
    print(f"\n  Успешно обработано: {len(dlx.processed)}")
    print(f"  В DLQ: {len(dlx.dlq)}")


# =============================================================================
# Демо 3: Потоковая обработка событий — концепции Kafka
# =============================================================================

def demo_event_streaming():
    print("\n\n" + "=" * 70)
    print("Демо 3: Потоковая обработка событий — концепции Kafka")
    print("=" * 70)

    # --- 3.1 Моделирование Kafka-топика с партициями ---
    print("\n--- 3.1 Kafka-топик с партициями ---")

    class KafkaPartition:
        def __init__(self, partition_id):
            self.id = partition_id
            self.records = []  # список записей
            self.offset = 0    # текущий смещение (offset)

        def append(self, key, value):
            """Добавление записи в партицию"""
            record = {
                "offset": self.offset,
                "key": key,
                "value": value,
                "timestamp": time.time()
            }
            self.records.append(record)
            self.offset += 1
            return record

        def read(self, start_offset=0):
            """Чтение записей с указанного смещения"""
            return [r for r in self.records if r["offset"] >= start_offset]

    class KafkaTopic:
        def __init__(self, name, num_partitions=3):
            self.name = name
            self.partitions = [KafkaPartition(i) for i in range(num_partitions)]

        def produce(self, key, value):
            """Производитель отправляет сообщение — партиция выбирается по хешу ключа"""
            partition_idx = hash(key) % len(self.partitions)
            record = self.partitions[partition_idx].append(key, value)
            return partition_idx, record

    topic = KafkaTopic("user_events", num_partitions=3)

    # Генерация событий
    events = [
        ("user_001", "page_view"),
        ("user_002", "add_to_cart"),
        ("user_001", "checkout"),
        ("user_003", "login"),
        ("user_002", "purchase"),
        ("user_004", "logout"),
    ]

    print(f"  Топик: {topic.name}, партиций: {len(topic.partitions)}")
    for key, value in events:
        pidx, rec = topic.produce(key, value)
        print(f"    → Партиция {pidx}: [{rec['offset']}] key={key}, value={value}")

    print(f"\n  Размеры партиций: {[len(p.records) for p in topic.partitions]}")

    # --- 3.2 Consumer Groups и смещения (offsets) ---
    print("\n--- 3.2 Consumer Groups и смещения (offsets) ---")

    class ConsumerGroup:
        """Группа потребителей — каждая партиция назначается только одному
        потребителю в группе (load balancing)"""
        def __init__(self, group_id, topic):
            self.group_id = group_id
            self.topic = topic
            self.offsets = {i: 0 for i in range(len(topic.partitions))}
            self.committed = {i: 0 for i in range(len(topic.partitions))}

        def poll(self, consumer_id, max_messages=2):
            """Потребитель запрашивает сообщения из назначенных партиций"""
            consumed = []
            for pidx in range(len(self.topic.partitions)):
                partition = self.topic.partitions[pidx]
                offset = self.offsets[pidx]
                records = partition.read(offset)[:max_messages]
                for rec in records:
                    consumed.append((pidx, rec))
                    self.offsets[pidx] = rec["offset"] + 1
            return consumed

        def commit(self):
            """Подтверждение обработки (commit offset)"""
            self.committed = self.offsets.copy()
            return self.committed.copy()

    cgroup = ConsumerGroup("analytics_group", topic)

    print(f"  Consumer Group: {cgroup.group_id}")
    print("  Шаг 1 — потребление:")
    msgs = cgroup.poll("worker_1", max_messages=1)
    for pidx, rec in msgs:
        print(f"    Партиция {pidx}: [{rec['offset']}] {rec['key']} → {rec['value']}")

    print(f"\n  Текущие offsets: {cgroup.offsets}")
    print("  Шаг 2 — commit:")
    committed = cgroup.commit()
    print(f"  Закоммиченные offsets: {committed}")

    print("\n  Шаг 3 — потребление оставшегося:")
    msgs = cgroup.poll("worker_1", max_messages=10)
    for pidx, rec in msgs:
        print(f"    Партиция {pidx}: [{rec['offset']}] {rec['key']} → {rec['value']}")

    # --- 3.3 Replication и ISR (In-Sync Replicas) ---
    print("\n--- 3.3 Replication и ISR ---")

    class Replica:
        def __init__(self, replica_id, is_leader=False):
            self.id = replica_id
            self.is_leader = is_leader
            self.records = []
            self.lag = 0  # отставание от лидера

        def replicate(self, record):
            """Репликация записи с лидера"""
            self.records.append(record)

    class KafkaPartitionWithReplication:
        def __init__(self, partition_id, replication_factor=3):
            self.id = partition_id
            self.leader = Replica(0, is_leader=True)
            self.replicas = [Replica(i) for i in range(1, replication_factor)]
            self.isr = [self.leader] + self.replicas  # все in-sync

        def produce(self, key, value):
            record = {"key": key, "value": value, "offset": len(self.leader.records)}
            # Запись сначала идёт к лидеру
            self.leader.replicate(record)
            # Затем реплицируется на все ISR
            for replica in self.replicas:
                replica.replicate(record)
            return record

        def simulate_lag(self, replica_idx, lag_amount):
            """Имитация отставания реплики"""
            self.replicas[replica_idx].lag = lag_amount
            # Удаляем последние записи, чтобы имитировать отставание
            self.replicas[replica_idx].records = \
                self.replicas[replica_idx].records[:-lag_amount] if lag_amount > 0 else \
                self.replicas[replica_idx].records

    partition_r = KafkaPartitionWithReplication(0, replication_factor=3)

    for i in range(5):
        partition_r.produce(f"key_{i}", f"value_{i}")

    print(f"  Лидер (Replica 0): {len(partition_r.leader.records)} записей")
    for rep in partition_r.replicas:
        print(f"  Реплика {rep.id}: {len(rep.records)} записей, lag={rep.lag}")

    # Имитируем отставание реплики 1
    partition_r.simulate_lag(1, 2)
    print(f"\n  После отставания реплики 1 на 2 записи:")
    for rep in partition_r.replicas:
        print(f"  Реплика {rep.id}: {len(rep.records)} записей, lag={rep.lag}")

    # --- 3.4 Log Compaction ---
    print("\n--- 3.4 Log Compaction ---")

    # Log compaction сохраняет только последнюю версию для каждого ключа
    def log_compaction(records):
        """Компактация лога — удаление дубликатов, сохранение последнего значения"""
        latest = {}
        for rec in records:
            latest[rec["key"]] = rec["value"]
        return latest

    # Имитируем лог с обновлениями
    raw_log = [
        {"key": "user_001", "value": "active"},
        {"key": "user_002", "value": "active"},
        {"key": "user_001", "value": "premium"},    # обновление
        {"key": "user_003", "value": "active"},
        {"key": "user_002", "value": "suspended"},  # обновление
        {"key": "user_001", "value": "enterprise"}, # обновление
    ]

    print(f"  Исходный лог ({len(raw_log)} записей):")
    for r in raw_log:
        print(f"    [{r['key']}] = {r['value']}")

    compacted = log_compaction(raw_log)
    print(f"\n  После компактации ({len(compacted)} записей — только последнее значение):")
    for k, v in compacted.items():
        print(f"    [{k}] = {v}")

    savings = (1 - len(compacted) / len(raw_log)) * 100
    print(f"  Экономия места: {savings:.1f}%")


# =============================================================================
# Демо 4: Асинхронные паттерны — очереди задач, retry, dead letter queue
# =============================================================================

def demo_async_patterns():
    print("\n\n" + "=" * 70)
    print("Демо 4: Асинхронные паттерны — очереди задач, retry, dead letter queue")
    print("=" * 70)

    # --- 4.1 Work Queue с round-robin ---
    print("\n--- 4.1 Work Queue с round-robin ---")

    class WorkQueue:
        """Очередь задач с round-robin распределением между потребителями.
        Задачи распределяются по очереди: первая задача → первый воркер,
        вторая → второй, третья → первый, и т.д."""
        def __init__(self):
            self.tasks = []
            self.workers = {}
            self.current_worker = 0

        def add_task(self, task):
            self.tasks.append({"id": str(uuid.uuid4())[:6], "data": task, "assigned_to": None})

        def add_worker(self, name):
            self.workers[name] = []

        def dispatch(self):
            """Распределение задач по воркерам (round-robin)"""
            worker_names = list(self.workers.keys())
            while self.tasks:
                task = self.tasks.pop(0)
                worker = worker_names[self.current_worker % len(worker_names)]
                task["assigned_to"] = worker
                self.workers[worker].append(task)
                self.current_worker += 1
                print(f"    Задача {task['data']} → {worker}")

    wq = WorkQueue()
    wq.add_worker("worker_A")
    wq.add_worker("worker_B")

    tasks = ["train_model", "evaluate", "preprocess", "predict", "save_checkpoint", "log_metrics"]
    for t in tasks:
        wq.add_task(t)

    print("  Распределение задач (round-robin):")
    wq.dispatch()
    print(f"\n  Нагрузка воркеров:")
    for name, assigned in wq.workers.items():
        print(f"    {name}: {len(assigned)} задач — {[t['data'] for t in assigned]}")

    # --- 4.2 Retry с экспоненциальной задержкой ---
    print("\n--- 4.2 Retry с экспоненциальной задержкой ---")

    def exponential_backoff(attempt, base_delay=0.1, max_delay=5.0):
        """Формула экспоненциальной задержки:
        delay = min(base_delay * 2^attempt, max_delay)"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        return delay

    print("  Формула: delay = min(base * 2^attempt, max_delay)")
    print(f"  base_delay=0.1, max_delay=5.0")
    print()

    for attempt in range(6):
        delay = exponential_backoff(attempt)
        print(f"    Попытка {attempt}: задержка = {delay:.3f} сек "
              f"(0.1 × 2^{attempt} = {0.1 * (2**attempt):.3f})")

    # Симуляция retry с jitter
    print("\n  С jitter (случайный разброс ±20%):")
    random.seed(42)
    for attempt in range(4):
        base = exponential_backoff(attempt)
        jitter = base * random.uniform(0.8, 1.2)
        print(f"    Попытка {attempt}: базовая={base:.3f}, с jitter={jitter:.3f}")

    # --- 4.3 Circuit Breaker ---
    print("\n--- 4.3 Circuit Breaker (прерыватель цепи) ---")

    class CircuitBreaker:
        """Паттерн Circuit Breaker — если количество ошибок превышает порог,
        цепь разрывается и запросы не отправляются (экономия ресурсов).
        После timeout происходит полупробное восстановление."""
        CLOSED = "CLOSED"      # нормальная работа
        OPEN = "OPEN"          # цепь разорвана, запросы не проходят
        HALF_OPEN = "HALF_OPEN"  # тестовое восстановление

        def __init__(self, failure_threshold=3, recovery_timeout=5):
            self.state = self.CLOSED
            self.failure_count = 0
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.last_failure_time = None

        def call(self, func, *args):
            if self.state == self.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    print(f"    Circuit Breaker: HALF_OPEN (тестовое восстановление)")
                else:
                    print(f"    Circuit Breaker: OPEN — запрос отклонён")
                    return None

            try:
                result = func(*args)
                if self.state == self.HALF_OPEN:
                    self.state = self.CLOSED
                    self.failure_count = 0
                    print(f"    Circuit Breaker: CLOSED (восстановлен)")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = self.OPEN
                    print(f"    Circuit Breaker: OPEN после {self.failure_count} ошибок")
                return None

    def unreliable_service(x):
        """Имитация ненадёжного сервиса — падает на нечётных входах"""
        if x % 2 == 1:
            raise ValueError(f"Ошибка сервиса для {x}")
        return x * 10

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.01)
    test_inputs = [1, 3, 5, 7, 2, 4, 6]  # нечётные вызовут ошибки

    print("  Тест Circuit Breaker (порог ошибок = 3):")
    for x in test_inputs:
        result = cb.call(unreliable_service, x)
        status = f"→ {result}" if result is not None else "→ отклонено"
        print(f"    Вход {x}: {status} [состояние: {cb.state}]")

    # --- 4.4 Sharded Queue (шардированная очередь) ---
    print("\n--- 4.4 Sharded Queue (шардированная очередь) ---")

    # Шардирование — распределение сообщений по нескольким очередям
    # для горизонтального масштабирования
    class ShardedQueue:
        def __init__(self, num_shards=4):
            self.shards = [[] for _ in range(num_shards)]
            self.num_shards = num_shards

        def _get_shard(self, key):
            """Определяем шард по хешу ключа"""
            return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_shards

        def enqueue(self, key, value):
            shard_idx = self._get_shard(key)
            self.shards[shard_idx].append({"key": key, "value": value})
            return shard_idx

        def stats(self):
            return {f"shard_{i}": len(q) for i, q in enumerate(self.shards)}

    sq = ShardedQueue(num_shards=4)
    users = [f"user_{i:03d}" for i in range(16)]

    print(f"  Шардов: {sq.num_shards}, пользователей: {len(users)}")
    for user in users:
        shard = sq.enqueue(user, {"action": "update"})
        print(f"    {user} → shard {shard}")

    print(f"\n  Распределение по шардам:")
    for shard, count in sq.stats().items():
        bar = "█" * count
        print(f"    {shard}: {count:2d} сообщений |{bar}|")


# =============================================================================
# Запуск всех демонстраций
# =============================================================================

if __name__ == "__main__":
    demo_queue_basics()
    demo_pubsub()
    demo_event_streaming()
    demo_async_patterns()
