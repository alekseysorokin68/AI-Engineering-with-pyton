"""197 — Communication Networks: топология, gossip протоколы, распространение информации

Темы:
  1. Network Topologies (ring, star, mesh, small-world, scale-free)
  2. Gossip Protocols (push, pull, push-pull, rumor spreading)
  3. Information Cascades (herding, information vs influence cascades)
  4. Network Resilience (node failure, cascading failures, robustness)

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


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 1: Топологии сетей
# ──────────────────────────────────────────────────────────────────────────────

class NetworkTopology:
    """Построение различных топологий сетей."""

    def __init__(self, n):
        self.n = n  # количество узлов
        self.adj = collections.defaultdict(set)  # список смежности

    def add_edge(self, u, v):
        """Добавляет ребро между узлами."""
        self.adj[u].add(v)
        self.adj[v].add(u)

    def ring(self):
        """Кольцевая топология: каждый узел связан с двумя соседями."""
        for i in range(self.n):
            self.add_edge(i, (i + 1) % self.n)
        return self

    def star(self):
        """Звёздная топология: центральный узел связан со всеми."""
        for i in range(1, self.n):
            self.add_edge(0, i)
        return self

    def mesh(self):
        """Полная.mesh-топология: каждый узел связан со всеми."""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.add_edge(i, j)
        return self

    def small_world(self, k=4, rewire_prob=0.1):
        """Small-world топология (модель Уоттса-Строгаца)."""
        # Начинаем с кольца с k-связями
        for i in range(self.n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % self.n
                self.add_edge(i, neighbor)

        # Переключаем рёбра с вероятностью rewire_prob
        for i in range(self.n):
            neighbors = list(self.adj[i])
            for neighbor in neighbors:
                if random.random() < rewire_prob:
                    self.adj[i].discard(neighbor)
                    self.adj[neighbor].discard(i)
                    # Выбираем случайный новый узел
                    new_neighbor = random.randint(0, self.n - 1)
                    while new_neighbor == i or new_neighbor in self.adj[i]:
                        new_neighbor = random.randint(0, self.n - 1)
                    self.add_edge(i, new_neighbor)
        return self

    def scale_free(self, m=2):
        """Барабаши-Альберт модель (scale-free, безмасштабная)."""
        # Начинаем с полного графа из m+1 узлов
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                self.add_edge(i, j)

        # Добавляем узлы с приоритетом присоединения
        for new_node in range(m + 1, self.n):
            # Вычисляем вероятности присоединения (preferential attachment)
            degrees = {i: len(self.adj[i]) for i in range(new_node)}
            total_degree = sum(degrees.values())
            if total_degree == 0:
                continue

            targets = set()
            while len(targets) < m:
                # Вероятность пропорциональна степени
                r = random.random() * total_degree
                cumulative = 0
                for node, deg in degrees.items():
                    cumulative += deg
                    if cumulative >= r:
                        targets.add(node)
                        break

            for target in targets:
                self.add_edge(new_node, target)
        return self

    def stats(self):
        """Вычисляет статистику топологии."""
        degrees = [len(self.adj[i]) for i in range(self.n)]
        avg_degree = sum(degrees) / self.n if self.n > 0 else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        density = sum(degrees) / (self.n * (self.n - 1)) if self.n > 1 else 0

        return {
            "nodes": self.n,
            "edges": sum(degrees) // 2,
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
            "min_degree": min_degree,
            "density": round(density, 4)
        }


def demo_network_topologies():
    """Демонстрация топологий сетей."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: Топологии сетей")
    print("=" * 70)

    n = 12  # количество узлов

    # Пример 1: Кольцевая топология
    print("\n--- Пример 1: Кольцевая топология ---")
    ring = NetworkTopology(n).ring()
    stats = ring.stats()
    print(f"  Узлов: {stats['nodes']}, Рёбер: {stats['edges']}")
    print(f"  Средняя степень: {stats['avg_degree']}")
    print(f"  Макс. степень: {stats['max_degree']}")
    print(f"  Плотность: {stats['density']}")
    print(f"  Соседи узла 0: {sorted(ring.adj[0])}")

    # Пример 2: Звёздная топология
    print("\n--- Пример 2: Звёздная топология ---")
    star = NetworkTopology(n).star()
    stats = star.stats()
    print(f"  Узлов: {stats['nodes']}, Рёбер: {stats['edges']}")
    print(f"  Средняя степень: {stats['avg_degree']}")
    print(f"  Центральный узел (0): степень = {len(star.adj[0])}")
    print(f"  Листовые узлы: степень = {stats['min_degree']}")

    # Пример 3: Small-world топология
    print("\n--- Пример 3: Small-world топология ---")
    random.seed(42)
    sw = NetworkTopology(n).small_world(k=4, rewire_prob=0.2)
    stats = sw.stats()
    print(f"  Узлов: {stats['nodes']}, Рёбер: {stats['edges']}")
    print(f"  Средняя степень: {stats['avg_degree']}")
    print(f"  Диапазон степеней: {stats['min_degree']}-{stats['max_degree']}")

    # Пример 4: Scale-free топология
    print("\n--- Пример 4: Scale-free (Барабаши-Альберт) ---")
    random.seed(42)
    sf = NetworkTopology(n).scale_free(m=2)
    stats = sf.stats()
    print(f"  Узлов: {stats['nodes']}, Рёбер: {stats['edges']}")
    print(f"  Средняя степень: {stats['avg_degree']}")
    print(f"  Макс. степень (хаб): {stats['max_degree']}")

    # Сравнение топологий
    print("\n--- Сравнение топологий ---")
    print(f"  {'Топология':<15} {'Рёбра':>6} {'Ср.степень':>10} {'Плотность':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10}")
    for name, topo in [("Кольцо", ring), ("Звезда", star),
                       ("Small-world", sw), ("Scale-free", sf)]:
        s = topo.stats()
        print(f"  {name:<15} {s['edges']:>6} {s['avg_degree']:>10.2f} "
              f"{s['density']:>10.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 2: Gossip-протоколы
# ──────────────────────────────────────────────────────────────────────────────

class GossipProtocol:
    """Реализация gossip-протоколов распространения информации."""

    def __init__(self, network):
        self.network = network  # NetworkTopology
        self.nodes = {}
        self.messages_delivered = 0
        self.total_rounds = 0

    def init_nodes(self):
        """Инициализирует узлы сети."""
        for i in range(self.network.n):
            self.nodes[i] = {
                "id": i,
                "knows": set(),
                "buffer": []
            }

    def inject_rumor(self, node_id, rumor):
        """Внедряет слух в один узел."""
        if node_id in self.nodes:
            self.nodes[node_id]["knows"].add(rumor)
            self.nodes[node_id]["buffer"].append(rumor)

    def push_gossip(self, rounds=10):
        """Push-протокол: отправитель отправляет известную информацию."""
        self.total_rounds = 0
        for _ in range(rounds):
            self.total_rounds += 1
            for node_id in range(self.network.n):
                node = self.nodes[node_id]
                if node["buffer"]:
                    # Выбираем случайного соседа
                    neighbors = list(self.network.adj[node_id])
                    if neighbors:
                        target_id = random.choice(neighbors)
                        target = self.nodes[target_id]
                        # Отправляем все сообщения из буфера
                        for msg in node["buffer"]:
                            if msg not in target["knows"]:
                                target["knows"].add(msg)
                                target["buffer"].append(msg)
                                self.messages_delivered += 1
                    node["buffer"] = []

    def pull_gossip(self, rounds=10):
        """Pull-протокол: получатель запрашивает информацию."""
        self.total_rounds = 0
        for _ in range(rounds):
            self.total_rounds += 1
            for node_id in range(self.network.n):
                node = self.nodes[node_id]
                neighbors = list(self.network.adj[node_id])
                if neighbors:
                    # Запрашиваем информацию у случайного соседа
                    target_id = random.choice(neighbors)
                    target = self.nodes[target_id]
                    for msg in target["knows"]:
                        if msg not in node["knows"]:
                            node["knows"].add(msg)
                            node["buffer"].append(msg)
                            self.messages_delivered += 1

    def push_pull_gossip(self, rounds=10):
        """Push-Pull протокол: комбинация обоих подходов."""
        self.total_rounds = 0
        for _ in range(rounds):
            self.total_rounds += 1
            for node_id in range(self.network.n):
                node = self.nodes[node_id]
                neighbors = list(self.network.adj[node_id])
                if neighbors:
                    target_id = random.choice(neighbors)
                    target = self.nodes[target_id]

                    # Push: отправляем свои сообщения
                    for msg in node["buffer"]:
                        if msg not in target["knows"]:
                            target["knows"].add(msg)
                            target["buffer"].append(msg)
                            self.messages_delivered += 1

                    # Pull: запрашиваем сообщения
                    for msg in target["knows"]:
                        if msg not in node["knows"]:
                            node["knows"].add(msg)
                            node["buffer"].append(msg)
                            self.messages_delivered += 1

                node["buffer"] = []

    def get_coverage(self):
        """Вычисляет долю узлов, знающих хотя бы один слух."""
        all_rumors = set()
        for node in self.nodes.values():
            all_rumors.update(node["knows"])

        if not all_rumors:
            return 0

        covered = sum(1 for node in self.nodes.values() if node["knows"])
        return round(covered / self.network.n, 3)

    def get_rumor_coverage(self):
        """Покрытие для каждого слуха отдельно."""
        if not self.nodes:
            return {}
        all_rumors = set()
        for node in self.nodes.values():
            all_rumors.update(node["knows"])

        coverage = {}
        for rumor in all_rumors:
            count = sum(1 for node in self.nodes.values() if rumor in node["knows"])
            coverage[rumor] = round(count / self.network.n, 3)
        return coverage


def demo_gossip_protocols():
    """Демонстрация gossip-протоколов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: Gossip-протоколы")
    print("=" * 70)

    # Пример 1: Push-протокол
    print("\n--- Пример 1: Push-протокол ---")
    random.seed(42)
    net = NetworkTopology(10).ring()
    gossip = GossipProtocol(net)
    gossip.init_nodes()
    gossip.inject_rumor(0, "rumor_A")

    for step in range(1, 6):
        gossip.push_gossip(rounds=1)
        coverage = gossip.get_coverage()
        print(f"  Раунд {step}: покрытие = {coverage:.1%}, "
              f"доставлено сообщений = {gossip.messages_delivered}")

    # Пример 2: Pull-протокол
    print("\n--- Пример 2: Pull-протокол ---")
    random.seed(42)
    gossip2 = GossipProtocol(net)
    gossip2.init_nodes()
    gossip2.inject_rumor(0, "rumor_B")

    for step in range(1, 6):
        gossip2.pull_gossip(rounds=1)
        coverage = gossip2.get_coverage()
        print(f"  Раунд {step}: покрытие = {coverage:.1%}, "
              f"доставлено сообщений = {gossip2.messages_delivered}")

    # Пример 3: Push-Pull протокол
    print("\n--- Пример 3: Push-Pull протокол ---")
    random.seed(42)
    gossip3 = GossipProtocol(net)
    gossip3.init_nodes()
    gossip3.inject_rumor(0, "rumor_C")

    for step in range(1, 6):
        gossip3.push_pull_gossip(rounds=1)
        coverage = gossip3.get_coverage()
        print(f"  Раунд {step}: покрытие = {coverage:.1%}, "
              f"доставлено сообщений = {gossip3.messages_delivered}")

    # Пример 4: Сравнение протоколов
    print("\n--- Пример 4: Сравнение протоколов ---")
    protocols = {
        "Push": (gossip, "rumor_A"),
        "Pull": (gossip2, "rumor_B"),
        "Push-Pull": (gossip3, "rumor_C")
    }
    print(f"  {'Протокол':<12} {'Финальное покрытие':>18} {'Сообщений':>10}")
    print(f"  {'-'*12} {'-'*18} {'-'*10}")
    for name, (gp, _) in protocols.items():
        print(f"  {name:<12} {gp.get_coverage():>17.1%} {gp.messages_delivered:>10}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 3: Каскады информации
# ──────────────────────────────────────────────────────────────────────────────

class InformationCascade:
    """Модель информационных каскадов."""

    def __init__(self, n, threshold=0.5):
        self.n = n
        self.threshold = threshold  # порог для принятия решения
        self.states = [0] * n  # 0 = не принял, 1 = принял
        self.history = []
        self.connections = collections.defaultdict(set)

    def add_connection(self, i, j):
        """Добавляет связь между агентами."""
        self.connections[i].add(j)
        self.connections[j].add(i)

    def herding_cascade(self, initial_adopters):
        """Каскад выпаса (herding): агенты копируют соседей."""
        print(f"\n  Каскад выпаса (порог = {self.threshold}):")
        print(f"  Начальные adopters: {initial_adopters}")

        for agent in initial_adopters:
            self.states[agent] = 1

        self.history.append(self.states.copy())

        for round_num in range(1, 10):
            changed = False
            new_states = self.states.copy()

            for i in range(self.n):
                if self.states[i] == 0:
                    # Считаем долю принявших среди соседей
                    neighbors = self.connections[i]
                    if neighbors:
                        adopted = sum(1 for j in neighbors if self.states[j] == 1)
                        ratio = adopted / len(neighbors)
                        if ratio >= self.threshold:
                            new_states[i] = 1
                            changed = True

            self.states = new_states
            self.history.append(self.states.copy())
            total = sum(self.states)
            print(f"  Раунд {round_num}: приняли = {total}/{self.n} "
                  f"({total/self.n:.1%})")

            if not changed:
                print(f"  Каскад завершился на раунде {round_num}")
                break

    def information_vs_influence(self):
        """Различие информационных и Influence-каскадов."""
        print("\n  Информационные vs Influence-каскады:")
        print("  " + "=" * 55)
        print("  | Тип каскада       | Механизм              |")
        print("  |-------------------|-----------------------|")
        print("  | Информационный    | Передача фактов       |")
        print("  | Influence         | Социальное давление   |")
        print("  " + "=" * 55)

        # Пример: информационный каскад
        print("\n  Информационный каскад (достоверность):")
        info_spread = [0.1, 0.3, 0.5, 0.7, 0.85, 0.92, 0.95]
        for i, p in enumerate(info_spread):
            bar = "#" * int(p * 40)
            print(f"    Раунд {i+1}: {p:.0%} [{bar}]")

        # Пример: influence каскад
        print("\n  Influence каскад (пороговый):")
        inf_spread = [0.02, 0.02, 0.02, 0.02, 0.15, 0.4, 0.75, 0.95]
        for i, p in enumerate(inf_spread):
            bar = "#" * int(p * 40)
            print(f"    Раунд {i+1}: {p:.0%} [{bar}]")

        print("\n  Вывод: информационные каскады растут плавно,")
        print("  influence — имеют точку перегиба (tipping point)")


def demo_information_cascades():
    """Демонстрация информационных каскадов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: Информационные каскады")
    print("=" * 70)

    # Пример 1: Каскад выпаса с низким порогом
    print("\n--- Пример 1: Каскад выпаса (низкий порог) ---")
    random.seed(42)
    cascade1 = InformationCascade(10, threshold=0.3)
    # Создаём линейную сеть
    for i in range(9):
        cascade1.add_connection(i, i + 1)
    cascade1.herding_cascade(initial_adopters=[0, 9])

    # Пример 2: Каскад выпаса с высоким порогом
    print("\n--- Пример 2: Каскад выпаса (высокий порог) ---")
    random.seed(42)
    cascade2 = InformationCascade(10, threshold=0.7)
    for i in range(9):
        cascade2.add_connection(i, i + 1)
    cascade2.herding_cascade(initial_adopters=[0, 9])

    # Пример 3: Влияние структуры сети на каскад
    print("\n--- Пример 3: Влияние структуры сети ---")
    for topo_name, topo_fn in [("Кольцо", lambda n: NetworkTopology(n).ring()),
                                ("Звезда", lambda n: NetworkTopology(n).star())]:
        random.seed(42)
        net = topo_fn(8)
        cascade = InformationCascade(8, threshold=0.5)
        cascade.connections = net.adj.copy()
        cascade.herding_cascade(initial_adopters=[0])
        total = sum(cascade.states)
        print(f"  {topo_name}: итого приняли = {total}/8")

    # Пример 4: Информационные vs Influence каскады
    print("\n--- Пример 4: Информационные vs Influence каскады ---")
    InformationCascade(0).information_vs_influence()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 4: Устойчивость сетей
# ──────────────────────────────────────────────────────────────────────────────

class NetworkResilience:
    """Анализ устойчивости сетей к отказам."""

    def __init__(self, network):
        self.network = network
        self.original_adj = {k: set(v) for k, v in network.adj.items()}

    def remove_node(self, node_id):
        """Удаляет узел из сети."""
        if node_id in self.network.adj:
            neighbors = list(self.network.adj[node_id])
            for neighbor in neighbors:
                self.network.adj[neighbor].discard(node_id)
            del self.network.adj[node_id]

    def restore(self):
        """Восстанавливает сеть."""
        self.network.adj = {k: set(v) for k, v in self.original_adj.items()}

    def random_failure(self, num_failures):
        """Случайный отказ узлов."""
        self.restore()
        nodes = list(range(self.network.n))
        random.shuffle(nodes)
        failed = nodes[:num_failures]

        for node in failed:
            self.remove_node(node)

        return self._measure_connectivity()

    def targeted_attack(self, num_failures):
        """Целевая атака на узлы с наибольшей степенью."""
        self.restore()
        # Сортируем по степени (убывание)
        nodes_by_degree = sorted(
            range(self.network.n),
            key=lambda x: len(self.network.adj.get(x, set())),
            reverse=True
        )
        failed = nodes_by_degree[:num_failures]

        for node in failed:
            self.remove_node(node)

        return self._measure_connectivity()

    def _measure_connectivity(self):
        """Измеряет связность сети."""
        remaining = [i for i in range(self.network.n) if i in self.network.adj]
        if not remaining:
            return {"components": 0, "largest_component": 0,
                    "connectivity": 0, "remaining_nodes": 0}

        # BFS для поиска компонент связности
        visited = set()
        components = []

        for start in remaining:
            if start in visited:
                continue
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in self.network.adj.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

        largest = max(components, key=len) if components else []

        return {
            "components": len(components),
            "largest_component": len(largest),
            "connectivity": round(len(largest) / self.network.n, 3)
            if self.network.n > 0 else 0,
            "remaining_nodes": len(remaining)
        }

    def cascading_failure(self, initial_failures, load_threshold=0.8):
        """Каскадные отказы: перегрузка при перераспределении трафика."""
        self.restore()
        failed_nodes = set(initial_failures)

        for node in initial_failures:
            self.remove_node(node)

        print(f"\n  Каскадные отказы (порог нагрузки = {load_threshold}):")
        print(f"  Начальные отказы: {initial_failures}")

        cascade_round = 0
        while True:
            cascade_round += 1
            new_failures = []

            # Пересчитываем нагрузку на оставшихся узлах
            for node in list(self.network.adj.keys()):
                if node in failed_nodes:
                    continue
                # Нагрузка = доля активных соседей от исходной
                original_degree = len(self.original_adj.get(node, set()))
                current_degree = len(self.network.adj.get(node, set()))
                if original_degree > 0:
                    load = current_degree / original_degree
                    # Узел выходит из строя, если нагрузка выше порога
                    # (симуляция: при потере соседей трафик перераспределяется)
                    if load < load_threshold and current_degree > 0:
                        # Упрощённая модель: узел отказывает с вероятностью
                        # обратно пропорциональной нагрузке
                        failure_prob = 1 - load
                        if random.random() < failure_prob:
                            new_failures.append(node)

            if not new_failures:
                break

            for node in new_failures:
                self.remove_node(node)
                failed_nodes.add(node)

            stats = self._measure_connectivity()
            print(f"  Раунд {cascade_round}: отказов = {len(new_failures)}, "
                  f"осталось = {stats['remaining_nodes']}, "
                  f"компонент = {stats['components']}")

            if stats['remaining_nodes'] == 0:
                print("  Полный отказ сети!")
                break

        self.restore()
        return failed_nodes


def demo_network_resilience():
    """Демонстрация устойчивости сетей."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Устойчивость сетей")
    print("=" * 70)

    # Пример 1: Случайный отказ vs целевая атака
    print("\n--- Пример 1: Случайный отказ vs целевая атака ---")
    random.seed(42)
    net = NetworkTopology(12).scale_free(m=2)
    resilience = NetworkResilience(net)

    print(f"  Сеть: {net.n} узлов, {net.stats()['edges']} рёбер")
    print(f"\n  {'Отказов':>8} {'Случайный':>10} {'Целевая':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10}")

    for num_failures in [1, 2, 3, 4, 5]:
        random.seed(42)
        rand_stats = resilience.random_failure(num_failures)
        random.seed(42)
        targ_stats = resilience.targeted_attack(num_failures)
        print(f"  {num_failures:>8} {rand_stats['connectivity']:>9.1%} "
              f"{targ_stats['connectivity']:>9.1%}")

    # Пример 2: Каскадные отказы
    print("\n--- Пример 2: Каскадные отказы ---")
    random.seed(42)
    net2 = NetworkTopology(10).small_world(k=4, rewire_prob=0.2)
    resilience2 = NetworkResilience(net2)
    resilience2.cascading_failure(initial_failures=[0, 1])

    # Пример 3: Восстановление после отказов
    print("\n--- Пример 3: Восстановление после отказов ---")
    random.seed(42)
    net3 = NetworkTopology(8).ring()
    resilience3 = NetworkResilience(net3)

    print(f"  Исходная сеть: {net3.stats()['edges']} рёбер")
    resilience3.targeted_attack(2)
    print(f"  После атаки (2 узла): {net3.stats()['edges']} рёбер")
    resilience3.restore()
    print(f"  После восстановления: {net3.stats()['edges']} рёбер")

    # Пример 4: Сравнение устойчивости топологий
    print("\n--- Пример 4: Сравнение устойчивости топологий ---")
    topologies = {
        "Кольцо": NetworkTopology(10).ring(),
        "Звезда": NetworkTopology(10).star(),
        "Small-world": NetworkTopology(10).small_world(k=4),
        "Scale-free": NetworkTopology(10).scale_free(m=2)
    }

    print(f"  {'Топология':<15} {'Случайный(2)':>12} {'Целевой(2)':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")

    for name, topo in topologies.items():
        res = NetworkResilience(topo)
        random.seed(42)
        rand = res.random_failure(2)
        random.seed(42)
        targ = res.targeted_attack(2)
        print(f"  {name:<15} {rand['connectivity']:>11.1%} "
              f"{targ['connectivity']:>11.1%}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_network_topologies()
    demo_gossip_protocols()
    demo_information_cascades()
    demo_network_resilience()
