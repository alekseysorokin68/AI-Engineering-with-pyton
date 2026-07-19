"""
201 — Distributed Systems Fundamentals: теорема CAP, модели консистентности, толерантность к ошибкам

Темы:
  1. CAP Theorem (consistency, availability, partition tolerance, trade-offs)
  2. Consistency Models (strong, eventual, causal, monotonic reads)
  3. Fault Tolerance (replication, quorum, heartbeat, failure detection)
  4. Distributed Consensus (Raft basics, leader election, log replication)

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
# DEMO 1: Теорема CAP
# ─────────────────────────────────────────────────────────────────────────────
def demo_cap_theorem():
    """Демонстрация теоремы CAP и компромиссов между свойствами."""
    print("=" * 70)
    print("DEMO 1: Theorem CAP (Consistency, Availability, Partition Tolerance)")
    print("=" * 70)

    # 1.1 Моделирование системы с выбором между C, A, P
    class CAPSystem:
        """Система, которая может поддерживать только 2 из 3 свойств CAP."""
        def __init__(self, name, consistency, availability, partition_tolerance):
            self.name = name
            self.consistency = consistency
            self.availability = availability
            self.partition_tolerance = partition_tolerance

        def describe(self):
            """Описание выбранного компромисса."""
            props = []
            if self.consistency:
                props.append("Consistency")
            if self.availability:
                props.append("Availability")
            if self.partition_tolerance:
                props.append("Partition Tolerance")
            return f"{self.name}: {', '.join(props)}"

    systems = [
        CAPSystem("CA System (Traditional RDBMS)", True, True, False),
        CAPSystem("CP System (MongoDB, HBase)", True, False, True),
        CAPSystem("AP System (Cassandra, DynamoDB)", False, True, True),
    ]

    print("\n1.1 Компромиссы CAP:")
    print("   Теорема CAP: распределённая система может гарантировать только 2 из 3:")
    print("   - Consistency (консистентность): все узлы видят одни данные")
    print("   - Availability (доступность): каждый запрос получает ответ")
    print("   - Partition Tolerance (толерантность к разбиениям): система работает при сетевых сбоях\n")

    for system in systems:
        print(f"   {system.describe()}")

    # 1.2 Формула вероятности консистентности при разбиении
    print("\n1.2 Вероятность консистентности при сетевом разбиении:")
    n_nodes = 5
    p_link = 0.8  # Вероятность связи между двумя узлами
    # Для простоты: P(все связаны) ≈ p_link^(n*(n-1)/2)
    p_connected = p_link ** (n_nodes * (n_nodes - 1) // 2)
    print(f"   Узлов: {n_nodes}, вероятность связи между узлами: {p_link}")
    print(f"   P(все узлы связаны) ≈ {p_link}^({n_nodes}*{n_nodes-1}/2) = {p_connected:.6f}")
    print(f"   -> При {n_nodes} узлах и p={p_link} система часто будет разорвана")

    # 1.3 Симуляция разбиения сети
    print("\n1.3 Симуляция сетевого разбиения:")
    nodes = list(range(n_nodes))
    random.shuffle(nodes)
    partition_point = len(nodes) // 2
    partition_a = nodes[:partition_point]
    partition_b = nodes[partition_point:]

    print(f"   Узлы: {nodes}")
    print(f"   Разбиение A: {partition_a}")
    print(f"   Разбиение B: {partition_b}")

    data_a = {node: f"data_from_A_{node}" for node in partition_a}
    data_b = {node: f"data_from_B_{node}" for node in partition_b}
    print(f"   Данные в Partition A: {data_a}")
    print(f"   Данные в Partition B: {data_b}")
    print("   -> При восстановлении связи необходимо разрешение конфликтов (CRDT, last-write-wins)")

    # 1.4 Таблица решений CAP
    print("\n1.4 Таблица решений CAP:")
    print("   +----------------+----------------+----------------+------------------+")
    print("   | Свойство       | CA (RDBMS)     | CP (MongoDB)   | AP (Cassandra)   |")
    print("   +----------------+----------------+----------------+------------------+")
    print("   | Консистентность| Высокая        | Высокая        | Базовая          |")
    print("   | Доступность    | Высокая        | Снижается      | Высокая          |")
    print("   | Разбиения      | Не толерантна  | Толерантна     | Толерантна       |")
    print("   | Использование  | Транзакции     | Хранилища      | IoT, Big Data    |")
    print("   +----------------+----------------+----------------+------------------+")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2: Модели консистентности
# ─────────────────────────────────────────────────────────────────────────────
def demo_consistency_models():
    """Демонстрация различных моделей консистентности."""
    print("\n" + "=" * 70)
    print("DEMO 2: Consistency Models (Strong, Eventual, Causal, Monotonic)")
    print("=" * 70)

    # 2.1 Сильная консистентность (Linearizability)
    class StrongConsistency:
        """Симуляция системы с сильной консистентностью."""
        def __init__(self):
            self.data = {}
            self.version = 0

        def write(self, key, value):
            """Запись блокирует чтение до подтверждения."""
            self.version += 1
            self.data[key] = (value, self.version)
            return True

        def read(self, key):
            """Чтение всегда возвращает последнюю запись."""
            if key in self.data:
                return self.data[key]
            return None

    store = StrongConsistency()
    print("\n2.1 Сильная консистентность (Linearizability):")
    print("   Гарантия: каждое чтение видит последнюю запись\n")

    store.write("x", 1)
    store.write("x", 2)
    store.write("y", 10)

    result_x = store.read("x")
    result_y = store.read("y")
    print(f"   Записали x=1, затем x=2, затем y=10")
    print(f"   Чтение x: {result_x} (ожидается: (2, 3))")
    print(f"   Чтение y: {result_y} (ожидается: (10, 3))")
    print("   -> Все чтения видят последние записи (линейнаялизуемость)")

    # 2.2 Конечная консистентность (Eventual Consistency)
    class EventualConsistency:
        """Симуляция системы с конечной консистентностью."""
        def __init__(self, n_replicas):
            self.replicas = [{} for _ in range(n_replicas)]
            self.pending = []

        def write(self, key, value):
            """Запись в главную реплику."""
            self.replicas[0][key] = value
            self.pending.append((key, value, 0))
            return True

        def propagate(self):
            """Асинхронная репликация."""
            new_pending = []
            for key, value, source in self.pending:
                for i in range(len(self.replicas)):
                    if i != source and random.random() < 0.5:
                        self.replicas[i][key] = value
            self.pending = new_pending

        def read(self, replica_idx):
            """Чтение из конкретной реплики."""
            return self.replicas[replica_idx].copy()

    print("\n2.2 Конечная консистентность (Eventual Consistency):")
    print("   Гарантия: при отсутствии новых записей, все реплики в конечном итоге сойдутся\n")

    ec = EventualConsistency(3)
    ec.write("x", 1)
    ec.write("x", 2)

    print("   До репликации:")
    for i in range(3):
        print(f"     Реплика {i}: {ec.read(i)}")

    # Повторяем репликацию несколько раз для сходимости
    for _ in range(10):
        ec.propagate()

    print("\n   После репликации (10 итераций):")
    for i in range(3):
        print(f"     Реплика {i}: {ec.read(i)}")
    print("   -> Реплики постепенно сходятся (eventual consistency)")

    # 2.3 Каузальная консистентность
    class CausalConsistency:
        """Симуляция каузальной консистентности."""
        def __init__(self):
            self.data = {}
            self.vector_clock = collections.defaultdict(int)

        def write(self, key, value, client_id):
            """Запись с векторными часами."""
            self.vector_clock[client_id] += 1
            self.data[key] = (value, dict(self.vector_clock))

        def read(self, key, client_clock):
            """Чтение с проверкой каузальности."""
            if key not in self.data:
                return None
            value, write_clock = self.data[key]
            causal = all(write_clock.get(k, 0) >= client_clock.get(k, 0)
                        for k in client_clock)
            return value if causal else "STALE"

    print("\n2.3 Каузальная консистентность (Causal Consistency):")
    print("   Гарантия: каузально связанные операции видят одинаковый порядок\n")

    cc = CausalConsistency()
    cc.write("x", 1, "client1")
    val = cc.read("x", dict(cc.vector_clock))
    cc.write("y", 100, "client2")

    print(f"   Клиент1 пишет x=1")
    print(f"   Клиент2 читает x={val} и пишет y=100 (каузально связано)")
    print(f"   Векторные часы: {dict(cc.vector_clock)}")
    print("   -> Операции поддерживают каузальный порядок (happens-before)")

    # 2.4 Монотонные чтения
    class MonotonicReads:
        """Симуляция монотонных чтений."""
        def __init__(self):
            self.data = {}
            self.version = 0
            self.session_last_read = {}

        def write(self, key, value):
            self.version += 1
            self.data[key] = (value, self.version)

        def read(self, key, session_id):
            """Чтение не может вернуть более старое значение."""
            if key not in self.data:
                return None
            value, version = self.data[key]
            last = self.session_last_read.get(session_id, 0)
            if version >= last:
                self.session_last_read[session_id] = version
                return value
            return "VIOLATION"

    print("\n2.4 Монотонные чтения (Monotonic Reads):")
    print("   Гарантия: последующее чтение не вернёт более старое значение\n")

    mr = MonotonicReads()
    mr.write("x", 1)
    read1 = mr.read("x", "session1")
    mr.write("x", 2)
    read2 = mr.read("x", "session1")

    print(f"   Сессия 1: чтение x={read1}, затем x={read2}")
    print(f"   Порядок версий: {read1} (v1) -> {read2} (v2)")
    print("   -> Чтения монотонны (значение не уменьшается)")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: Толерантность к ошибкам
# ─────────────────────────────────────────────────────────────────────────────
def demo_fault_tolerance():
    """Демонстрация толерантности к ошибкам."""
    print("\n" + "=" * 70)
    print("DEMO 3: Fault Tolerance (Replication, Quorum, Heartbeat)")
    print("=" * 70)

    # 3.1 Репликация данных
    class ReplicationManager:
        """Управление репликацией данных."""
        def __init__(self, n_replicas):
            self.replicas = [{} for _ in range(n_replicas)]
            self.alive = [True] * n_replicas

        def write(self, key, value, write_quorum):
            """Запись с кворумом."""
            written = 0
            for i in range(len(self.replicas)):
                if self.alive[i]:
                    self.replicas[i][key] = value
                    written += 1
            return written >= write_quorum

        def read(self, key, read_quorum):
            """Чтение с кворумом."""
            values = []
            for i in range(len(self.replicas)):
                if self.alive[i] and key in self.replicas[i]:
                    values.append(self.replicas[i][key])
            if len(values) >= read_quorum:
                counter = collections.Counter(values)
                return counter.most_common(1)[0][0]
            return None

        def kill(self, replica_idx):
            """Симуляция отказа узла."""
            self.alive[replica_idx] = False

    print("\n3.1 Репликация с кворумом:")
    print("   Формула: W + R > N (для консистентных чтений)")
    print("   Где W = write quorum, R = read quorum, N = число реплик\n")

    rm = ReplicationManager(5)
    write_ok = rm.write("x", 42, write_quorum=3)
    print(f"   Запись x=42 с W=3: {'успех' if write_ok else 'неудача'}")

    read_val = rm.read("x", read_quorum=3)
    print(f"   Чтение x с R=3: {read_val}")

    rm.kill(0)
    rm.kill(1)
    print(f"\n   Убили узлы 0 и 1 (осталось 3 из 5)")
    read_val = rm.read("x", read_quorum=3)
    print(f"   Чтение x с R=3: {read_val} (всё работает при N-W < выживших)")

    # 3.2 Quorum (кворум)
    print("\n3.2 Расчёт кворума:")
    N = 7
    W = 4
    R = 4

    print(f"   N={N}, W={W}, R={R}")
    symbol = ">" if W + R > N else "<="
    print(f"   W + R = {W + R} {symbol} N={N}")
    if W + R > N:
        print("   -> Гарантия консистентности (кворум достигнут)")
    else:
        print("   -> Нет гарантии консистентности (кворум не достигнут)")

    f_max = N - W
    print(f"   Макс. число отказов без потери записи: {f_max} узлов")
    print(f"   Макс. число отказов без потери чтения: {N - R} узлов")

    # 3.3 Heartbeat и обнаружение отказов
    class HeartbeatDetector:
        """Детектор отказов через heartbeat."""
        def __init__(self, nodes, timeout=3):
            self.nodes = {node: {"last_heartbeat": 0, "alive": True} for node in nodes}
            self.timeout = timeout
            self.time = 0

        def heartbeat(self, node):
            """Обновить heartbeat узла."""
            self.nodes[node]["last_heartbeat"] = self.time

        def tick(self):
            """Увеличить время и проверить узлы."""
            self.time += 1
            failed = []
            for node, info in self.nodes.items():
                if info["alive"] and (self.time - info["last_heartbeat"]) > self.timeout:
                    info["alive"] = False
                    failed.append(node)
            return failed

    print("\n3.3 Heartbeat и обнаружение отказов:")
    detector = HeartbeatDetector(["A", "B", "C", "D"], timeout=3)

    for node in ["A", "B", "C", "D"]:
        detector.heartbeat(node)

    print("   Инициализация: все узлы живы")
    for t in range(6):
        # Узел B перестаёт отвечать после t=1
        if t >= 2:
            detector.heartbeat("A")
            detector.heartbeat("C")
            detector.heartbeat("D")
        else:
            for node in ["A", "B", "C", "D"]:
                detector.heartbeat(node)

        failed = detector.tick()
        status = {node: ("alive" if info["alive"] else "DEAD")
                  for node, info in detector.nodes.items()}
        fail_str = f" (обнаружен отказ: {failed})" if failed else ""
        print(f"   t={t}: {status}{fail_str}")

    # 3.4 Симуляция failover
    print("\n3.4 Failover (переключение при отказе):")
    cluster_nodes = ["Primary", "Secondary-1", "Secondary-2", "Secondary-3"]
    active = cluster_nodes[0]
    print(f"   Начальное состояние: active = {active}")

    failure_sequence = [True, False, False, True]
    for i in range(len(failure_sequence)):
        failed = failure_sequence[i]
        if failed and i < len(cluster_nodes) and active == cluster_nodes[i]:
            for j in range(i + 1, len(cluster_nodes)):
                if not failure_sequence[j]:
                    active = cluster_nodes[j]
                    print(f"   t={i}: {cluster_nodes[i]} отказал -> failover к {active}")
                    break
            else:
                print(f"   t={i}: {cluster_nodes[i]} отказал -> нет доступных узлов!")
                break
        elif not failed:
            print(f"   t={i}: {cluster_nodes[i]} восстановлен (replication in progress)")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: Распределённый консенсус (Raft)
# ─────────────────────────────────────────────────────────────────────────────
def demo_distributed_consensus():
    """Демонстрация распределённого консенсуса (Raft)."""
    print("\n" + "=" * 70)
    print("DEMO 4: Distributed Consensus (Raft Basics)")
    print("=" * 70)

    # 4.1 Симуляция выборов лидера (Leader Election)
    class RaftNode:
        """Упрощённый узел Raft."""
        def __init__(self, node_id):
            self.id = node_id
            self.state = "follower"
            self.current_term = 0
            self.voted_for = None
            self.log = []

        def start_election(self):
            """Начать выборы."""
            self.current_term += 1
            self.state = "candidate"
            self.voted_for = self.id
            return self.current_term

        def receive_vote(self, term):
            """Проголосовать за кандидата."""
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
                self.state = "follower"
                return True
            return False

        def become_leader(self):
            """Стать лидером."""
            self.state = "leader"

    print("\n4.1 Выборы лидера (Leader Election):")
    raft_nodes = [RaftNode(i) for i in range(5)]
    print(f"   Узлы: {[n.id for n in raft_nodes]}")

    candidate = raft_nodes[2]
    term = candidate.start_election()
    print(f"\n   Узел {candidate.id} инициирует выборы (term={term})")

    votes = 1  # Голос за себя
    for node in raft_nodes:
        if node.id != candidate.id:
            if node.receive_vote(term):
                votes += 1
                print(f"   Узел {node.id} голосует за узел {candidate.id}")

    majority = len(raft_nodes) // 2 + 1
    print(f"\n   Голосов: {votes}/{len(raft_nodes)} (нужно {majority} для большинства)")
    if votes >= majority:
        candidate.become_leader()
        print(f"   -> Узел {candidate.id} становится лидером!")
    else:
        print("   -> Выборы не состоялись (мало голосов)")

    # 4.2 Репликация лога
    class RaftLog:
        """Симуляция лога Raft."""
        def __init__(self):
            self.entries = []

        def append(self, term, command):
            """Добавить запись в лог."""
            self.entries.append({"term": term, "command": command, "index": len(self.entries)})

        def replicate_from(self, source_log, match_index):
            """Реплицировать лог из источника."""
            replicated = 0
            for i in range(match_index, len(source_log.entries)):
                self.entries.append(source_log.entries[i].copy())
                replicated += 1
            return replicated

    print("\n4.2 Репликация лога (Log Replication):")
    leader_log = RaftLog()
    follower_log = RaftLog()

    commands = ["SET x=1", "SET y=2", "SET x=3"]
    for cmd in commands:
        leader_log.append(term=1, command=cmd)

    print(f"   Лог лидера: {[e['command'] for e in leader_log.entries]}")

    replicated = follower_log.replicate_from(leader_log, match_index=0)
    print(f"   Реплицировано {replicated} записей на follower")
    print(f"   Лог follower: {[e['command'] for e in follower_log.entries]}")

    # 4.3 Коммит через большинство
    print("\n4.3 Коммит через большинство:")
    n_replicas = 5
    commit_quorum = n_replicas // 2 + 1

    print(f"   Узлов: {n_replicas}, commit quorum: {commit_quorum}")
    print(f"   Запись коммитится, когда {commit_quorum} узлов подтвердили")

    committed = 0
    for i in range(n_replicas):
        if random.random() < 0.7:
            committed += 1
            print(f"   Узел {i}: подтверждено")
        else:
            print(f"   Узел {i}: ошибка репликации")

    if committed >= commit_quorum:
        print(f"   -> Коммит успешен ({committed}/{n_replicas} >= {commit_quorum})")
    else:
        print(f"   -> Коммит не удался ({committed}/{n_replicas} < {commit_quorum})")

    # 4.4 Сравнение консенсус-протоколов
    print("\n4.4 Сравнение протоколов консенсуса:")
    protocols = [
        ("Raft",   "Leader-based",  "Простота",            "Leader bottleneck"),
        ("Paxos",  "Бессистемный",  "Гибкость",            "Сложность реализации"),
        ("ZAB",    "Leader-based",  "Ordering",            "Специфичен для ZooKeeper"),
        ("EPaxos", "Бессистемный",  "Parallel commits",    "Сложность"),
    ]

    print("   +----------+----------------+------------------+--------------------+")
    print("   | Протокол | Тип            | Преимущество      | Недостаток         |")
    print("   +----------+----------------+------------------+--------------------+")
    for name, ptype, advantage, disadvantage in protocols:
        print(f"   | {name:8} | {ptype:14} | {advantage:16} | {disadvantage:17} |")
    print("   +----------+----------------+------------------+--------------------+")


if __name__ == "__main__":
    demo_cap_theorem()
    demo_consistency_models()
    demo_fault_tolerance()
    demo_distributed_consensus()
