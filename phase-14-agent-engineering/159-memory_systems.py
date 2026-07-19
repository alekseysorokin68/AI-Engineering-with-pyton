"""159 — Memory Systems: краткосрочная, долгосрочная, эпизодическая, семантическая память

Темы:
  1. Short-Term Memory — буфер, рабочая память, управление контекстным окном
  2. Long-Term Memory — хранение, извлечение, скоринг релевантности
  3. Episodic Memory — опытный плеистор, хранение траекторий, извлечение паттернов
  4. Semantic Memory — граф знаний, извлечение сущностей, хранение фактов

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


# ─────────────────────────────────────────────────────────────
# 1. Short-Term Memory — буфер, рабочая память, контекстное окно
# ─────────────────────────────────────────────────────────────

class ShortTermMemory:
    """Краткосрочная память агента: буфер фиксированного размера с eviction-политикой."""

    def __init__(self, capacity: int = 10):
        # Максимальный размер буфера
        self.capacity = capacity
        # Основное хранилище — список кортежей (время_создания, сообщение)
        self.buffer = []
        # Рабочая память — промежуточная область для текущей задачи
        self.working_memory = {}

    def add(self, message: str, timestamp: float = None):
        """Добавить сообщение в краткосрочную память."""
        if timestamp is None:
            timestamp = time.time()
        self.buffer.append((timestamp, message))
        # Если буфер переполнен — удаляем самое старое сообщение (FIFO eviction)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_recent(self, n: int = 5) -> list:
        """Получить последние n сообщений из буфера."""
        return [msg for _, msg in self.buffer[-n:]]

    def set_working(self, key: str, value):
        """Сохранить значение в рабочую память (текущий контекст задачи)."""
        self.working_memory[key] = value

    def get_working(self, key: str, default=None):
        """Получить значение из рабочей памяти."""
        return self.working_memory.get(key, default)

    def clear_working(self):
        """Очистить рабочую память (смена задачи)."""
        self.working_memory.clear()

    def token_estimate(self) -> int:
        """Приблизительная оценка количества токенов в буфере.
        Эвристика: ~1.3 токена на слово (для английского текста)."""
        total_words = sum(len(msg.split()) for _, msg in self.buffer)
        return int(total_words * 1.3)


def context_window_management(messages: list, max_tokens: int = 50) -> dict:
    """Управление контекстным окном: подбор сообщений до лимита токенов.

    Алгоритм: системный промпт всегда включён,
    затем добавляем сообщения от новых к старым.
    """
    # Системный промпт — всегда в контексте
    system_prompt = "You are a helpful assistant."
    system_tokens = int(len(system_prompt.split()) * 1.3)
    remaining = max_tokens - system_tokens

    selected = []
    used_tokens = 0

    # Идём от самого нового сообщения к самому старому
    for msg in reversed(messages):
        msg_tokens = int(len(msg.split()) * 1.3)
        if used_tokens + msg_tokens <= remaining:
            selected.insert(0, msg)
            used_tokens += msg_tokens
        else:
            break

    return {
        "system_prompt": system_prompt,
        "selected_messages": selected,
        "total_tokens": system_tokens + used_tokens,
        "budget": max_tokens,
        "utilization": round((system_tokens + used_tokens) / max_tokens, 2),
    }


def demo_short_term_memory():
    """Демонстрация: краткосрочная память, рабочая память, контекстное окно."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: КРАТКОСРОЧНАЯ ПАМЯТЬ (Short-Term Memory)")
    print("=" * 70)

    # --- Пример 1: Буфер с eviction ---
    print("\n--- Пример 1: Буфер с FIFO eviction ---")
    stm = ShortTermMemory(capacity=5)
    for i in range(8):
        stm.add(f"Сообщение пользователя #{i}")
    print(f"  Добавлено 8 сообщений в буфер размера 5")
    print(f"  Текущий буфер ({len(stm.buffer)} шт.):")
    for ts, msg in stm.buffer:
        print(f"    - {msg}")
    print(f"  Оценка токенов: {stm.token_estimate()}")

    # --- Пример 2: Рабочая память ---
    print("\n--- Пример 2: Рабочая память (working memory) ---")
    stm.set_working("current_task", "анализ данных продаж")
    stm.set_working("iteration", 3)
    stm.set_working("context", {"columns": ["date", "revenue", "region"]})
    print(f"  Текущая задача: {stm.get_working('current_task')}")
    print(f"  Итерация: {stm.get_working('iteration')}")
    print(f"  Контекст: {stm.get_working('context')}")
    stm.clear_working()
    print(f"  После очистки рабочей памяти: {stm.working_memory}")

    # --- Пример 3: Управление контекстным окном ---
    print("\n--- Пример 3: Управление контекстным окном ---")
    history = [
        "Привет! Как дела?",
        "Хочу узнать про Python",
        "Расскажи про декораторы",
        "А что такое замыкания?",
        "Как применять декораторы на практике?",
        "Покажи пример с @lru_cache",
    ]
    result = context_window_management(history, max_tokens=30)
    print(f"  Всего сообщений в истории: {len(history)}")
    print(f"  Лимит токенов: {result['budget']}")
    print(f"  Выбрано сообщений: {len(result['selected_messages'])}")
    for msg in result['selected_messages']:
        print(f"    - {msg}")
    print(f"  Использовано токенов: {result['total_tokens']} ({result['utilization'] * 100}%)")

    # --- Пример 4: Стратегия sliding window ---
    print("\n--- Пример 4: Стратегия sliding window ---")
    window_size = 4
    full_history = [f"turn_{i}" for i in range(10)]
    # Sliding window: берём последние window_size сообщений
    window = full_history[-window_size:]
    # Добавляем summary первых сообщений (компрессия)
    summary = f"[Summary of turns 0-{len(full_history) - window_size - 1}]"
    compressed = [summary] + window
    print(f"  Полная история: {len(full_history)} сообщений")
    print(f"  Окно (window_size={window_size}): {window}")
    print(f"  Сжатая версия ({len(compressed)} элементов):")
    for item in compressed:
        print(f"    - {item}")
    compression_ratio = len(compressed) / len(full_history)
    print(f"  Коэффициент сжатия: {compression_ratio:.1%}")

    print()


# ─────────────────────────────────────────────────────────────
# 2. Long-Term Memory — хранение, извлечение, скоринг релевантности
# ─────────────────────────────────────────────────────────────

class LongTermMemory:
    """Долгосрочная память агента: векторное хранилище с скорингом релевантности."""

    def __init__(self):
        # Хранилище: список словарей с текстом, эмбеддингом и метаданными
        self.entries = []

    def _embed(self, text: str) -> list:
        """Простая эмбеддинг-функция: хэш-баг на основе символов.
        В реальном агенте здесь был бы нейросетевой энкодер."""
        random.seed(hashlib.md5(text.encode()).hexdigest())
        return [random.random() for _ in range(8)]

    def store(self, text: str, metadata: dict = None):
        """Сохранить факт в долгосрочную память."""
        embedding = self._embed(text)
        entry = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "access_count": 0,
            "strength": 1.0,  # Сила воспоминания (для forgetting curve)
        }
        self.entries.append(entry)

    def cosine_similarity(self, a: list, b: list) -> float:
        """Косинусное сходство между двумя векторами.
        sim(A,B) = (A·B) / (||A|| × ||B||)"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Извлечь наиболее релевантные воспоминания по запросу."""
        query_emb = self._embed(query)
        scored = []
        for entry in self.entries:
            sim = self.cosine_similarity(query_emb, entry["embedding"])
            # Итоговый скор: сходство × сила воспоминания × (1 + log(1 + accesses))
            recency_boost = 1 + math.log1p(entry["access_count"])
            final_score = sim * entry["strength"] * recency_boost
            scored.append((final_score, entry))
        # Сортируем по убыванию скоринга
        scored.sort(key=lambda x: x[0], reverse=True)
        # Увеличиваем счётчик обращений для найденных
        for _, entry in scored[:top_k]:
            entry["access_count"] += 1
        return scored[:top_k]

    def forgetting_curve(self, entry: dict, t: float) -> float:
        """Кривая забывания Эббингауза: R = e^(-t / S)
        R — прочность воспоминания, t — время, S — стабильность."""
        S = 5.0 + entry["access_count"] * 2.0  # Чем больше обращений — тем стабильнее
        R = math.exp(-t / S)
        return R

    def consolidate(self, text: str, boost: float = 0.2):
        """Консолидация: усиление воспоминания через повторение."""
        for entry in self.entries:
            if text.lower() in entry["text"].lower():
                entry["strength"] = min(1.0, entry["strength"] + boost)
                entry["access_count"] += 1


def demo_long_term_memory():
    """Демонстрация: долгосрочная память, извлечение, скоринг релевантности."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: ДОЛГОСРОЧНАЯ ПАМЯТЬ (Long-Term Memory)")
    print("=" * 70)

    ltm = LongTermMemory()

    # --- Пример 1: Хранение фактов ---
    print("\n--- Пример 1: Хранение фактов ---")
    facts = [
        "Python — интерпретируемый язык программирования",
        "Декоратор — функция, принимающая функцию",
        "Transformer использует механизм внимания",
        "PyTorch — фреймворк для глубокого обучения",
        "RAG — Retrieval-Augmented Generation",
        "Байесовская оптимизация для гиперпараметров",
    ]
    for fact in facts:
        ltm.store(fact, metadata={"category": "knowledge"})
    print(f"  Сохранено {len(ltm.entries)} фактов в долгосрочной памяти")
    for e in ltm.entries:
        print(f"    - {e['text'][:50]}... (сила: {e['strength']:.2f})")

    # --- Пример 2: Извлечение по запросу ---
    print("\n--- Пример 2: Извлечение по запросу (retrieve) ---")
    query = "механизм внимания в нейросетях"
    results = ltm.retrieve(query, top_k=3)
    print(f"  Запрос: '{query}'")
    print(f"  Топ-{len(results)} результатов:")
    for rank, (score, entry) in enumerate(results, 1):
        print(f"    {rank}. [скор: {score:.4f}] {entry['text'][:60]}")

    # --- Пример 3: Кривая забывания ---
    print("\n--- Пример 3: Кривая забывания Эббингауза ---")
    print("  Формула: R = e^(-t / S), где S = 5.0 + access_count * 2.0")
    test_entry = ltm.entries[0]
    print(f"  Факт: '{test_entry['text'][:40]}...'")
    print(f"  Текущая сила: {test_entry['strength']:.2f}, обращений: {test_entry['access_count']}")
    for t in [0, 1, 3, 5, 10, 20]:
        R = ltm.forgetting_curve(test_entry, t)
        print(f"    t={t:2d}: R = {R:.4f} {'█' * int(R * 30)}")

    # --- Пример 4: Консолидация памяти ---
    print("\n--- Пример 4: Консолидация памяти (повторение) ---")
    ltm.consolidate("декоратор")
    ltm.consolidate("декоратор")
    print("  После 2-х повторений темы 'декоратор':")
    for e in ltm.entries:
        if "декоратор" in e["text"].lower():
            print(f"    - {e['text'][:50]}...")
            print(f"      Сила: {e['strength']:.2f}, обращений: {e['access_count']}")

    print()


# ─────────────────────────────────────────────────────────────
# 3. Episodic Memory — опытный плеистор, траектории, паттерны
# ─────────────────────────────────────────────────────────────

class EpisodicMemory:
    """Эпизодическая память: хранение и воспроизведение прошлых эпизодов (опыта)."""

    def __init__(self):
        # Эпизоды — последовательности (state, action, reward)
        self.episodes = []
        self.patterns = []

    def record_episode(self, episode_id: str, trajectory: list):
        """Записать эпизод: список шагов (state, action, reward)."""
        total_reward = sum(step["reward"] for step in trajectory)
        self.episodes.append({
            "id": episode_id,
            "trajectory": trajectory,
            "total_reward": total_reward,
            "length": len(trajectory),
        })

    def replay(self, episode_id: str) -> dict:
        """Воспроизвести эпизод по ID."""
        for ep in self.episodes:
            if ep["id"] == episode_id:
                return ep
        return None

    def extract_patterns(self, min_reward_threshold: float = 0.5):
        """Извлечь паттерны из успешных эпизодов.
        Паттерн = частая последовательность действий."""
        # Отбираем эпизоды с высокой наградой
        successful = [ep for ep in self.episodes if ep["total_reward"] >= min_reward_threshold]
        # Подсчитываем частоту пар действий (2-grams)
        action_pairs = collections.Counter()
        for ep in successful:
            actions = [step["action"] for step in ep["trajectory"]]
            for i in range(len(actions) - 1):
                pair = (actions[i], actions[i + 1])
                action_pairs[pair] += 1
        # Сохраняем паттерны с частотой >= 2
        self.patterns = [(pair, count) for pair, count in action_pairs.items() if count >= 2]
        return self.patterns

    def similarity(self, ep1: dict, ep2: dict) -> float:
        """Схожесть двух эпизодов: Jaccard-индекс по действиям."""
        actions1 = set(step["action"] for step in ep1["trajectory"])
        actions2 = set(step["action"] for step in ep2["trajectory"])
        if not actions1 and not actions2:
            return 1.0
        intersection = actions1 & actions2
        union = actions1 | actions2
        return len(intersection) / len(union) if union else 0.0

    def get_best_episode(self) -> dict:
        """Получить лучший эпизод по суммарной награде."""
        if not self.episodes:
            return None
        return max(self.episodes, key=lambda ep: ep["total_reward"])


def demo_episodic_memory():
    """Демонстрация: эпизодическая память, плеистор, извлечение паттернов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ (Episodic Memory)")
    print("=" * 70)

    em = EpisodicMemory()

    # --- Пример 1: Запись эпизодов ---
    print("\n--- Пример 1: Запись эпизодов (experience replay) ---")
    # Эпизод 1: успешное решение задачи
    em.record_episode("ep_001", [
        {"state": "start", "action": "analyze", "reward": 0.2},
        {"state": "analyzed", "action": "plan", "reward": 0.3},
        {"state": "planned", "action": "execute", "reward": 0.5},
    ])
    # Эпизод 2: неуспешное решение
    em.record_episode("ep_002", [
        {"state": "start", "action": "execute", "reward": 0.1},
        {"state": "error", "action": "retry", "reward": 0.0},
        {"state": "error", "action": "abort", "reward": -0.2},
    ])
    # Эпизод 3: успешное решение
    em.record_episode("ep_003", [
        {"state": "start", "action": "analyze", "reward": 0.2},
        {"state": "analyzed", "action": "plan", "reward": 0.3},
        {"state": "planned", "action": "execute", "reward": 0.5},
    ])
    print(f"  Записано {len(em.episodes)} эпизодов:")
    for ep in em.episodes:
        print(f"    {ep['id']}: {ep['length']} шагов, награда={ep['total_reward']:.2f}")

    # --- Пример 2: Воспроизведение эпизода ---
    print("\n--- Пример 2: Воспроизведение эпизода (replay) ---")
    replayed = em.replay("ep_001")
    print(f"  Эпизод {replayed['id']}:")
    for i, step in enumerate(replayed['trajectory']):
        print(f"    Шаг {i + 1}: state={step['state']}, action={step['action']}, reward={step['reward']}")

    # --- Пример 3: Извлечение паттернов ---
    print("\n--- Пример 3: Извлечение паттернов из успешных эпизодов ---")
    patterns = em.extract_patterns(min_reward_threshold=0.5)
    print(f"  Найдено паттернов (частые пары действий):")
    for (a1, a2), count in sorted(patterns, key=lambda x: -x[1]):
        print(f"    '{a1}' → '{a2}': встречается {count} раз(а)")

    # --- Пример 4: Схожесть эпизодов ---
    print("\n--- Пример 4: Схожесть эпизодов (Jaccard similarity) ---")
    ep1 = em.replay("ep_001")
    ep2 = em.replay("ep_002")
    ep3 = em.replay("ep_003")
    sim_12 = em.similarity(ep1, ep2)
    sim_13 = em.similarity(ep1, ep3)
    print(f"  Jaccard similarity(A, B) = |A ∩ B| / |A ∪ B|")
    print(f"  sim(ep_001, ep_002) = {sim_12:.4f} (успешный vs неуспешный)")
    print(f"  sim(ep_001, ep_003) = {sim_13:.4f} (успешный vs успешный)")
    best = em.get_best_episode()
    print(f"  Лучший эпизод: {best['id']} (награда={best['total_reward']:.2f})")

    print()


# ─────────────────────────────────────────────────────────────
# 4. Semantic Memory — граф знаний, извлечение сущностей, факты
# ─────────────────────────────────────────────────────────────

class SemanticMemory:
    """Семантическая память: граф знаний (entity-relation-entity)."""

    def __init__(self):
        # Граф знаний: {entity: {relation: [target_entities]}}
        self.knowledge = collections.defaultdict(lambda: collections.defaultdict(list))
        # Индекс фактов
        self.facts = []

    def add_fact(self, subject: str, relation: str, obj: str):
        """Добавить факт: (предмет, отношение, объект)."""
        self.knowledge[subject][relation].append(obj)
        self.knowledge[obj][f"inverse_{relation}"].append(subject)
        self.facts.append({"subject": subject, "relation": relation, "object": obj})

    def query(self, entity: str, relation: str = None) -> dict:
        """Запросить знания об entity."""
        if entity not in self.knowledge:
            return {}
        if relation:
            return {relation: self.knowledge[entity].get(relation, [])}
        return dict(self.knowledge[entity])

    def extract_entities(self, text: str) -> list:
        """Простое извлечение сущностей: заглавные слова как именованные сущности.
        (В реальном агенте — NER модель.)"""
        # Простая эвристика: слова с заглавной буквы в начале предложения
        words = text.split()
        entities = []
        for w in words:
            cleaned = re.sub(r'[^a-zA-Zа-яА-Я]', '', w)
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                entities.append(cleaned)
        return list(set(entities))

    def shortest_path(self, start: str, end: str, max_depth: int = 3) -> list:
        """Поиск кратчайшего пути между двумя сущностями (BFS)."""
        if start == end:
            return [start]
        visited = {start}
        queue = [(start, [start])]
        for _ in range(max_depth):
            next_queue = []
            for current, path in queue:
                for relation, targets in self.knowledge[current].items():
                    for target in targets:
                        if target == end:
                            return path + [target]
                        if target not in visited:
                            visited.add(target)
                            next_queue.append((target, path + [target]))
            queue = next_queue
        return []  # Путь не найден

    def count_facts(self) -> dict:
        """Подсчёт фактов по категориям отношений."""
        relation_counts = collections.Counter()
        for fact in self.facts:
            relation_counts[fact["relation"]] += 1
        return dict(relation_counts)

    def similarity_score(self, entity1: str, entity2: str) -> float:
        """Подобие двух сущностей: доля общих отношений / целых отношений."""
        rels1 = set(self.knowledge.get(entity1, {}).keys())
        rels2 = set(self.knowledge.get(entity2, {}).keys())
        if not rels1 and not rels2:
            return 0.0
        intersection = rels1 & rels2
        union = rels1 | rels2
        return len(intersection) / len(union) if union else 0.0


def demo_semantic_memory():
    """Демонстрация: граф знаний, извлечение сущностей, хранение фактов."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: СЕМАНТИЧЕСКАЯ ПАМЯТЬ (Semantic Memory)")
    print("=" * 70)

    sm = SemanticMemory()

    # --- Пример 1: Построение графа знаний ---
    print("\n--- Пример 1: Построение графа знаний ---")
    facts = [
        ("Python", "is_a", "язык программирования"),
        ("Python", "has_feature", "декораторы"),
        ("Python", "has_feature", "list comprehension"),
        ("Декоратор", "is_a", "функция"),
        ("Декоратор", "used_for", "модификация поведения"),
        ("Transformer", "is_a", "архитектура нейросети"),
        ("Transformer", "uses", "механизм внимания"),
        ("Внимание", "is_a", "механизм"),
        ("Внимание", "compute_with", "softmax"),
        ("GPT", "is_a", "языковая модель"),
        ("GPT", "based_on", "Transformer"),
    ]
    for subj, rel, obj in facts:
        sm.add_fact(subj, rel, obj)
    print(f"  Построен граф из {len(facts)} фактов")
    print(f"  Количество сущностей: {len(sm.knowledge)}")

    # --- Пример 2: Запрос знаний ---
    print("\n--- Пример 2: Запрос знаний (query) ---")
    query_entity = "Python"
    knowledge = sm.query(query_entity)
    print(f"  Запрос: '{query_entity}'")
    for rel, targets in knowledge.items():
        print(f"    {rel}: {', '.join(targets)}")

    # --- Пример 3: Извлечение сущностей из текста ---
    print("\n--- Пример 3: Извлечение сущностей из текста (NER) ---")
    text = "Трансформер использует Внимание для обработки последовательностей."
    entities = sm.extract_entities(text)
    print(f"  Текст: '{text}'")
    print(f"  Извлечённые сущности: {entities}")

    # --- Пример 4: Поиск пути и подобие ---
    print("\n--- Пример 4: Поиск пути между сущностями ---")
    path1 = sm.shortest_path("GPT", "softmax")
    path2 = sm.shortest_path("Python", "Transformer")
    print(f"  Путь от 'GPT' до 'softmax': {' → '.join(path1) if path1 else 'не найден'}")
    print(f"  Путь от 'Python' до 'Transformer': {'→'.join(path2) if path2 else 'не найден'}")

    sim_score = sm.similarity_score("Python", "Transformer")
    print(f"\n  Подобие Python ↔ Transformer: {sim_score:.4f}")
    print(f"  (Доля общих отношений от общего числа отношений)")

    relation_stats = sm.count_facts()
    print(f"\n  Статистика фактов по типам отношений:")
    for rel, count in sorted(relation_stats.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {count}")

    print()


# ─────────────────────────────────────────────────────────────
# Запуск всех демонстраций
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  159 — Memory Systems: краткосрочная, долгосрочная, эпизодическая, семантическая память")
    print("=" * 70 + "\n")

    demo_short_term_memory()
    demo_long_term_memory()
    demo_episodic_memory()
    demo_semantic_memory()

    print("=" * 70)
    print("  Все 4 демонстрации завершены успешно!")
    print("=" * 70)
