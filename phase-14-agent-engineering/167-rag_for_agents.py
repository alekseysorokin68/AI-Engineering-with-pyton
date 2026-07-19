"""167 — RAG for Agents: retrieval-augmented generation, базы знаний для агентов

Темы:
  1. Knowledge Base Design (стратегии чанкинга, метаданные, индексация)
  2. Retrieval Strategies (семантический поиск, гибридный поиск, reranking)
  3. RAG Pipeline (query → retrieve → augment → generate)
  4. RAG Quality (faithfulness, relevance, source attribution)

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
# Демо 1: Дизайн базы знаний
# ============================================================
def demo_knowledge_base_design():
    """Демонстрация дизайна базы знаний для RAG."""
    print("=" * 70)
    print("ДЕМО 1: ДИЗАЙН БАЗЫ ЗНАНИЙ (Knowledge Base Design)")
    print("=" * 70)

    # --- 1.1 Стратегии чанкинга ---
    print("\n[1.1] Стратегии чанкинга (Chunking Strategies)")
    print("-" * 50)

    class TextChunker:
        """Разби текста на чанки разными стратегиями."""

        def __init__(self):
            pass

        def fixed_size_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
            """Разбиение на фиксированные чанки с перекрытием."""
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append({
                    "text": chunk,
                    "start": start,
                    "end": end,
                    "size": len(chunk)
                })
                start += chunk_size - overlap
            return chunks

        def sentence_chunks(self, text: str, max_sentences: int = 3) -> list:
            """Разбиение по предложениям."""
            # Простое разбиение по символам окончания предложения
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_size = 0

            for sentence in sentences:
                current_chunk.append(sentence)
                current_size += len(sentence)

                if len(current_chunk) >= max_sentences:
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "sentence_count": len(current_chunk),
                        "size": current_size
                    })
                    current_chunk = []
                    current_size = 0

            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "sentence_count": len(current_chunk),
                    "size": current_size
                })

            return chunks

        def paragraph_chunks(self, text: str) -> list:
            """Разбиение по абзацам."""
            paragraphs = text.split("\n\n")
            return [
                {
                    "text": p.strip(),
                    "size": len(p.strip())
                }
                for p in paragraphs
                if p.strip()
            ]

        def semantic_chunks(self, text: str, min_chunk_size: int = 100) -> list:
            """Семантическое разбиение по тематическим блокам."""
            # Ищем границы разделителей
            separators = [". ", ".\n", "\n\n", "! ", "? "]
            chunks = []
            current_chunk = ""

            for i, char in enumerate(text):
                current_chunk += char

                # Проверяем, является ли текущая позиция границей предложения
                if char in ".!?" and i < len(text) - 1 and text[i + 1] == " ":
                    if len(current_chunk) >= min_chunk_size:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "size": len(current_chunk.strip()),
                            "type": "semantic"
                        })
                        current_chunk = ""

            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "size": len(current_chunk.strip()),
                    "type": "semantic"
                })

            return chunks

    chunker = TextChunker()

    # Демонстрационный текст
    sample_text = (
        "Python — высокоуровневый язык программирования. Он создался в 1991 году Гвидо ван Россумом. "
        "Python известен своей простотой и читаемостью кода. Язык широко используется в data science и AI. "
        "Библиотеки NumPy и Pandas сделали Python стандартом для анализа данных. "
        "Машинное обучение также активно развивается на Python. "
        "Библиотеки TensorFlow и PyTorch предоставляют инструменты для создания нейронных сетей. "
        "Flask и Django используются для веб-разработки. Python подходит для автоматизации задач."
    )

    print(f"  Исходный текст ({len(sample_text)} символов)")
    print(f"  \"{sample_text[:100]}...\"\n")

    # Фиксированные чанки
    fixed_chunks = chunker.fixed_size_chunks(sample_text, chunk_size=150, overlap=30)
    print(f"  Стратегия 1: Фиксированные чанки (размер=150, перекрытие=30)")
    print(f"    Количество чанков: {len(fixed_chunks)}")
    for i, chunk in enumerate(fixed_chunks[:3]):
        print(f"    Чанк {i+1} ({chunk['size']} символов): \"{chunk['text'][:50]}...\"")

    # Предложения
    sentence_chunks = chunker.sentence_chunks(sample_text, max_sentences=2)
    print(f"\n  Стратегия 2: По предложениям (макс. 2 предложения)")
    print(f"    Количество чанков: {len(sentence_chunks)}")
    for i, chunk in enumerate(sentence_chunks[:3]):
        print(f"    Чанк {i+1} ({chunk['sentence_count']} предложений): \"{chunk['text'][:60]}...\"")

    # Абзацы
    paragraph_chunks = chunker.paragraph_chunks(sample_text)
    print(f"\n  Стратегия 3: По абзацам")
    print(f"    Количество чанков: {len(paragraph_chunks)}")

    # Семантические чанки
    semantic_chunks = chunker.semantic_chunks(sample_text, min_chunk_size=80)
    print(f"\n  Стратегия 4: Семантические (мин. 80 символов)")
    print(f"    Количество чанков: {len(semantic_chunks)}")
    for i, chunk in enumerate(semantic_chunks[:3]):
        print(f"    Чанк {i+1} ({chunk['size']} символов): \"{chunk['text'][:60]}...\"")

    # --- 1.2 Метаданные ---
    print("\n[1.2] Метаданные чанков (Metadata)")
    print("-" * 50)

    class ChunkMetadata:
        """Управление метаданными чанков."""

        def __init__(self):
            self.metadata_fields = [
                "source", "title", "section", "page",
                "created_at", "updated_at", "author",
                "tags", "category", "importance"
            ]

        def enrich_chunk(self, chunk: dict, source_info: dict) -> dict:
            """Добавляет метаданные к чанку."""
            enriched = chunk.copy()
            enriched["metadata"] = {
                "source": source_info.get("filename", "unknown"),
                "title": source_info.get("title", "Untitled"),
                "section": source_info.get("section", "General"),
                "page": source_info.get("page", 1),
                "created_at": source_info.get("created_at", time.strftime("%Y-%m-%d")),
                "author": source_info.get("author", "Unknown"),
                "tags": source_info.get("tags", []),
                "category": self._categorize(chunk.get("text", "")),
                "importance": self._calculate_importance(chunk.get("text", "")),
                "char_count": len(chunk.get("text", "")),
                "word_count": len(chunk.get("text", "").split())
            }
            return enriched

        def _categorize(self, text: str) -> str:
            """Простая категоризация по ключевым словам."""
            text_lower = text.lower()
            if any(word in text_lower for word in ["python", "код", "программа", "функция"]):
                return "programming"
            elif any(word in text_lower for word in ["ai", "ml", "нейронн", "модел"]):
                return "ai_ml"
            elif any(word in text_lower for word in ["data", "данные", "анализ"]):
                return "data_science"
            return "general"

        def _calculate_importance(self, text: str) -> float:
            """Рассчитывает важность чанка (0-1)."""
            importance = 0.5  # Базовая важность

            # Длина текста влияет на важность
            if len(text) > 200:
                importance += 0.1
            if len(text) > 500:
                importance += 0.1

            # Ключевые слова повышают важность
            important_terms = ["важно", "критично", "основн", "ключев", "вывод"]
            for term in important_terms:
                if term in text.lower():
                    importance += 0.05

            return min(importance, 1.0)

        def filter_by_metadata(self, chunks: list, filters: dict) -> list:
            """Фильтрует чанки по метаданным."""
            result = chunks

            for key, value in filters.items():
                if key == "category":
                    result = [c for c in result if c.get("metadata", {}).get("category") == value]
                elif key == "min_importance":
                    result = [c for c in result
                             if c.get("metadata", {}).get("importance", 0) >= value]
                elif key == "tags":
                    result = [c for c in result
                             if any(tag in c.get("metadata", {}).get("tags", [])
                                   for tag in value)]

            return result

    metadata_mgr = ChunkMetadata()

    # Обогащаем чанки метаданными
    source_info = {
        "filename": "python_guide.md",
        "title": "Руководство по Python",
        "section": "Введение",
        "page": 1,
        "author": "Алексей Иванов",
        "tags": ["python", "programming", "tutorial"]
    }

    print("  Обогащение чанков метаданными:")
    enriched_chunks = []
    for i, chunk in enumerate(semantic_chunks[:4]):
        enriched = metadata_mgr.enrich_chunk(chunk, source_info)
        enriched_chunks.append(enriched)
        meta = enriched["metadata"]
        print(f"\n  Чанк {i+1}:")
        print(f"    Источник: {meta['source']}")
        print(f"    Категория: {meta['category']}")
        print(f"    Важность: {meta['importance']:.2f}")
        print(f"    Слов: {meta['word_count']}")
        print(f"    Теги: {meta['tags']}")

    # Фильтрация по метаданным
    print("\n  Фильтрация по категории 'programming':")
    filtered = metadata_mgr.filter_by_metadata(enriched_chunks, {"category": "programming"})
    print(f"    Найдено: {len(filtered)} чанков")

    print("\n  Фильтрация по важности >= 0.6:")
    filtered = metadata_mgr.filter_by_metadata(enriched_chunks, {"min_importance": 0.6})
    print(f"    Найдено: {len(filtered)} чанков")

    # --- 1.3 Индексация ---
    print("\n[1.3] Индексация (Indexing)")
    print("-" * 50)

    class SimpleIndex:
        """Простая инвертированная индексация для демонстрации."""

        def __init__(self):
            # Инвертированный индекс: слово -> список ID чанков
            self.inverted_index = collections.defaultdict(set)
            # Хранилище чанков
            self.chunks = {}
            self.chunk_counter = 0

        def add_chunk(self, chunk: dict) -> int:
            """Добавляет чанк в индекс."""
            self.chunk_counter += 1
            chunk_id = self.chunk_counter
            self.chunks[chunk_id] = chunk

            # Токенизация и индексация
            text = chunk.get("text", "").lower()
            words = re.findall(r'\b\w+\b', text)

            for word in words:
                self.inverted_index[word].add(chunk_id)

            return chunk_id

        def search(self, query: str, top_k: int = 5) -> list:
            """Поиск по индексу (простой TF-подобный)."""
            query_words = re.findall(r'\b\w+\b', query.lower())

            # Подсчёт частоты совпадений
            scores = collections.Counter()

            for word in query_words:
                if word in self.inverted_index:
                    for chunk_id in self.inverted_index[word]:
                        scores[chunk_id] += 1

            # Сортировка по скору
            results = []
            for chunk_id, score in scores.most_common(top_k):
                chunk = self.chunks[chunk_id]
                results.append({
                    "chunk_id": chunk_id,
                    "score": score,
                    "text_preview": chunk.get("text", "")[:100],
                    "metadata": chunk.get("metadata", {})
                })

            return results

        def get_index_stats(self) -> dict:
            """Возвращает статистику индекса."""
            return {
                "total_chunks": len(self.chunks),
                "unique_words": len(self.inverted_index),
                "avg_words_per_chunk": round(
                    sum(len(re.findall(r'\b\w+\b', c.get("text", "").lower()))
                        for c in self.chunks.values()) / max(len(self.chunks), 1), 1
                )
            }

    index = SimpleIndex()

    # Добавляем чанки в индекс
    print("  Индексация чанков:")
    for i, chunk in enumerate(enriched_chunks):
        chunk_id = index.add_chunk(chunk)
        print(f"    Чанк {chunk_id}: {len(chunk.get('text', '').split())} слов")

    stats = index.get_index_stats()
    print(f"\n  Статистика индекса:")
    print(f"    Всего чанков: {stats['total_chunks']}")
    print(f"    Уникальных слов: {stats['unique_words']}")
    print(f"    Среднее слов/чанк: {stats['avg_words_per_chunk']}")

    # Поиск
    print("\n  Поиск по запросу 'Python программирование':")
    results = index.search("Python программирование", top_k=3)
    for i, r in enumerate(results):
        print(f"    {i+1}. Чанк {r['chunk_id']} ( скор: {r['score']}): "
              f"\"{r['text_preview'][:60]}...\"")

    # --- 1.4 Векторная индексация (эмуляция) ---
    print("\n[1.4] Векторная индексация (эмуляция)")
    print("-" * 50)

    class VectorIndexEmulator:
        """Эмуляция векторной индексации для демонстрации."""

        def __init__(self, dimension: int = 64):
            self.dimension = dimension
            self.vectors = {}  # chunk_id -> vector
            self.chunks = {}

        def text_to_vector(self, text: str) -> list:
            """Простое хеширование текста в вектор (не настоящий embedding)."""
            # Используем хеш для создания детерминированного вектора
            text_hash = hashlib.md5(text.encode()).hexdigest()

            # Создаём вектор из хеша
            vector = []
            for i in range(self.dimension):
                # Берём байты из хеша и нормализуем
                byte_val = int(text_hash[i % len(text_hash)], 16)
                vector.append((byte_val / 15.0) * 2 - 1)  # Нормализуем к [-1, 1]

            return vector

        def cosine_similarity(self, vec1: list, vec2: list) -> float:
            """Вычисляет косинусное сходство между векторами."""
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a ** 2 for a in vec1))
            norm2 = math.sqrt(sum(b ** 2 for b in vec2))

            if norm1 == 0 or norm2 == 0:
                return 0

            return dot_product / (norm1 * norm2)

        def add_chunk(self, chunk_id: int, text: str):
            """Добавляет чанк с его вектором."""
            vector = self.text_to_vector(text)
            self.vectors[chunk_id] = vector
            self.chunks[chunk_id] = text

        def search(self, query: str, top_k: int = 5) -> list:
            """Поиск по векторному сходству."""
            query_vector = self.text_to_vector(query)

            # Вычисляем сходство со всеми чанками
            similarities = []
            for chunk_id, vector in self.vectors.items():
                sim = self.cosine_similarity(query_vector, vector)
                similarities.append({
                    "chunk_id": chunk_id,
                    "similarity": round(sim, 4),
                    "text_preview": self.chunks[chunk_id][:80]
                })

            # Сортируем по убыванию сходства
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            return similarities[:top_k]

    vec_index = VectorIndexEmulator(dimension=32)

    # Добавляем чанки
    print("  Индексация чанков векторами:")
    for i, chunk in enumerate(enriched_chunks):
        vec_index.add_chunk(i + 1, chunk.get("text", ""))
        print(f"    Чанк {i+1}: вектор размерности {vec_index.dimension}")

    # Векторный поиск
    print("\n  Векторный поиск: 'нейронные сети'")
    results = vec_index.search("нейронные сети", top_k=3)
    for i, r in enumerate(results):
        print(f"    {i+1}. Чанк {r['chunk_id']} (сходство: {r['similarity']:.4f})")
        print(f"       \"{r['text_preview']}...\"")

    # Объяснение сходства
    print("\n  Формула косинусного сходства:")
    print("    similarity(A, B) = (A · B) / (||A|| × ||B||)")
    print("    где A · B = sum(a_i × b_i), ||A|| = sqrt(sum(a_i²))")


# ============================================================
# Демо 2: Стратегии поиска
# ============================================================
def demo_retrieval_strategies():
    """Демонстрация стратегий поиска для RAG."""
    print("\n" + "=" * 70)
    print("ДЕМО 2: СТРАТЕГИИ ПОИСКА (Retrieval Strategies)")
    print("=" * 70)

    # --- 2.1 Семантический поиск ---
    print("\n[2.1] Семантический поиск (Semantic Search)")
    print("-" * 50)

    class SemanticSearchEngine:
        """Движок семантического поиска (эмуляция)."""

        def __init__(self):
            self.documents = []
            # Словарь синонимов для эмуляции семантики
            self.synonyms = {
                "машинное обучение": ["ml", "machine learning", "обучение моделей"],
                "нейронная сеть": ["neural network", "нейросеть", "deep learning"],
                "python": ["питон", "py"],
                "данные": ["data", "датасет", "информация"],
                "алгоритм": ["algorithm", "метод", "процедура"]
            }

        def add_document(self, doc_id: str, text: str, metadata: dict = None):
            """Добавляет документ в базу."""
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "tokens": self._tokenize(text)
            })

        def _tokenize(self, text: str) -> list:
            """Простая токенизация."""
            return re.findall(r'\b\w+\b', text.lower())

        def _expand_query(self, query: str) -> list:
            """Расширяет запрос синонимами."""
            tokens = self._tokenize(query)
            expanded = set(tokens)

            for token in tokens:
                for key, synonyms in self.synonyms.items():
                    if token in key or token in synonyms:
                        expanded.update(key.split())
                        expanded.update(synonyms)

            return list(expanded)

        def _calculate_relevance(self, query_tokens: list, doc_tokens: list) -> float:
            """Вычисляет релевантность на основе пересечения токенов."""
            query_set = set(query_tokens)
            doc_set = set(doc_tokens)

            intersection = query_set & doc_set
            union = query_set | doc_set

            if not union:
                return 0

            # Jaccard similarity
            jaccard = len(intersection) / len(union)

            # Дополнительный вес за точные совпадения
            exact_matches = sum(1 for t in query_tokens if t in doc_set)

            return jaccard + (exact_matches * 0.1)

        def search(self, query: str, top_k: int = 5) -> list:
            """Выполняет семантический поиск."""
            # Расширяем запрос
            expanded_tokens = self._expand_query(query)
            print(f"    Расширенный запрос: {expanded_tokens}")

            # Ищем совпадения
            results = []
            for doc in self.documents:
                relevance = self._calculate_relevance(expanded_tokens, doc["tokens"])
                results.append({
                    "id": doc["id"],
                    "text_preview": doc["text"][:80],
                    "relevance": round(relevance, 4),
                    "metadata": doc["metadata"]
                })

            # Сортируем по релевантности
            results.sort(key=lambda x: x["relevance"], reverse=True)

            return results[:top_k]

    search_engine = SemanticSearchEngine()

    # Добавляем документы
    documents = [
        ("doc1", "Python — язык программирования для data science и AI"),
        ("doc2", "Нейронные сети используют методы машинного обучения"),
        ("doc3", "Алгоритмы сортировки важны для обработки данных"),
        ("doc4", "TensorFlow и PyTorch — фреймворки для deep learning"),
        ("doc5", "Анализ данных помогает принимать решения")
    ]

    print("  Индексация документов:")
    for doc_id, text in documents:
        search_engine.add_document(doc_id, text, {"category": "tech"})
        print(f"    {doc_id}: \"{text[:50]}...\"")

    # Поиск
    print("\n  Семантический поиск: 'нейросети и обучение'")
    results = search_engine.search("нейросети и обучение", top_k=3)
    for i, r in enumerate(results):
        print(f"    {i+1}. {r['id']} (релевантность: {r['relevance']:.4f})")
        print(f"       \"{r['text_preview']}...\"")

    # --- 2.2 Гибридный поиск ---
    print("\n[2.2] Гибридный поиск (Hybrid Search)")
    print("-" * 50)

    class HybridSearchEngine:
        """Гибридный поиск: ключевые слова + семантика."""

        def __init__(self, keyword_weight: float = 0.5, semantic_weight: float = 0.5):
            self.keyword_weight = keyword_weight
            self.semantic_weight = semantic_weight
            self.documents = []

        def add_document(self, doc_id: str, text: str, vector: list = None):
            """Добавляет документ с текстом и опциональным вектором."""
            self.documents.append({
                "id": doc_id,
                "text": text,
                "tokens": set(re.findall(r'\b\w+\b', text.lower())),
                "vector": vector
            })

        def keyword_search(self, query: str) -> dict:
            """Поиск по ключевым словам (BM25-подобный)."""
            query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
            scores = {}

            for doc in self.documents:
                # Простой TF-IDF подсчёт
                matches = query_tokens & doc["tokens"]
                if matches:
                    # IDF: логарифм общего числа документов / число документов с термом
                    doc_counts = [sum(1 for d in self.documents if t in d["tokens"]) for t in matches]
                    idf = math.log(len(self.documents) / max(max(doc_counts), 1))

                    # TF: количество совпадений
                    tf = len(matches)

                    scores[doc["id"]] = tf * idf

            return scores

        def semantic_search(self, query: str) -> dict:
            """Семантический поиск по векторам."""
            scores = {}

            if not any(doc["vector"] for doc in self.documents):
                return scores

            # Генерируем вектор запроса (эмуляция)
            query_hash = hashlib.md5(query.encode()).hexdigest()
            query_vector = [int(c, 16) / 15 for c in query_hash[:32]]

            for doc in self.documents:
                if doc["vector"]:
                    # Косинусное сходство
                    dot = sum(a * b for a, b in zip(query_vector, doc["vector"]))
                    norm1 = math.sqrt(sum(a ** 2 for a in query_vector))
                    norm2 = math.sqrt(sum(b ** 2 for b in doc["vector"]))

                    if norm1 > 0 and norm2 > 0:
                        scores[doc["id"]] = dot / (norm1 * norm2)

            return scores

        def hybrid_search(self, query: str, top_k: int = 5) -> list:
            """Гибридный поиск с комбинацией методов."""
            keyword_scores = self.keyword_search(query)
            semantic_scores = self.semantic_search(query)

            # Нормализуем скоры
            max_keyword = max(keyword_scores.values()) if keyword_scores else 1
            max_semantic = max(semantic_scores.values()) if semantic_scores else 1

            combined_scores = {}
            for doc in self.documents:
                doc_id = doc["id"]
                kw_score = keyword_scores.get(doc_id, 0) / max_keyword
                sem_score = semantic_scores.get(doc_id, 0) / max_semantic

                combined = (kw_score * self.keyword_weight +
                           sem_score * self.semantic_weight)

                combined_scores[doc_id] = {
                    "combined": round(combined, 4),
                    "keyword": round(kw_score, 4),
                    "semantic": round(sem_score, 4)
                }

            # Сортируем по комбинированному скору
            results = sorted(combined_scores.items(),
                           key=lambda x: x[1]["combined"], reverse=True)

            return [
                {
                    "id": doc_id,
                    "text_preview": next(d["text"] for d in self.documents if d["id"] == doc_id)[:60],
                    **scores
                }
                for doc_id, scores in results[:top_k]
            ]

    hybrid_search = HybridSearchEngine(keyword_weight=0.6, semantic_weight=0.4)

    # Добавляем документы с векторами
    print("  Индексация документов с векторами:")
    for doc_id, text in documents:
        # Генерируем простой вектор для демонстрации
        vec_hash = hashlib.md5(text.encode()).hexdigest()
        vector = [int(c, 16) / 15 for c in vec_hash[:32]]
        hybrid_search.add_document(doc_id, text, vector)
        print(f"    {doc_id}: вектор из {len(vector)} измерений")

    # Гибридный поиск
    print("\n  Гибридный поиск: 'анализ данных'")
    results = hybrid_search.hybrid_search("анализ данных", top_k=3)
    for i, r in enumerate(results):
        print(f"    {i+1}. {r['id']}")
        print(f"       Комбинированный: {r['combined']}")
        print(f"       Ключевые слова: {r['keyword']}")
        print(f"       Семантика: {r['semantic']}")
        print(f"       \"{r['text_preview']}...\"")

    # --- 2.3 Reranking ---
    print("\n[2.3] Reranking (Переранжирование)")
    print("-" * 50)

    class Reranker:
        """Переранжирование результатов поиска."""

        def __init__(self):
            # Веса для различных факторов
            self.weights = {
                "relevance": 0.4,
                "freshness": 0.2,
                "authority": 0.2,
                "diversity": 0.2
            }

        def calculate_freshness(self, doc: dict) -> float:
            """Рассчитывает свежесть документа."""
            created = doc.get("created_at", "2024-01-01")
            # Простая эвристика: чем новее, тем выше
            try:
                days_ago = (time.time() - time.mktime(time.strptime(created, "%Y-%m-%d"))) / 86400
                return max(0, 1 - days_ago / 365)  # Убывает за год
            except:
                return 0.5

        def calculate_authority(self, doc: dict) -> float:
            """Рассчитывает авторитетность источника."""
            trusted_sources = {
                "docs.python.org": 0.9,
                "arxiv.org": 0.95,
                "github.com": 0.8,
                "stackoverflow.com": 0.7
            }

            source = doc.get("source", "")
            return trusted_sources.get(source, 0.5)

        def calculate_diversity(self, ranked_docs: list, candidate: dict) -> float:
            """Рассчитывает разнообразие (штраф за схожесть)."""
            if not ranked_docs:
                return 1.0

            candidate_tokens = set(candidate.get("tokens", []))

            max_similarity = 0
            for ranked in ranked_docs:
                ranked_tokens = set(ranked.get("tokens", []))
                if candidate_tokens and ranked_tokens:
                    similarity = len(candidate_tokens & ranked_tokens) / len(candidate_tokens | ranked_tokens)
                    max_similarity = max(max_similarity, similarity)

            return 1 - max_similarity

        def rerank(self, query: str, candidates: list) -> list:
            """Переранжирует кандидатов по нескольким факторам."""
            query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
            ranked = []

            for candidate in candidates:
                # Релевантность
                doc_tokens = set(re.findall(r'\b\w+\b', candidate.get("text", "").lower()))
                relevance = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)

                # Свежесть
                freshness = self.calculate_freshness(candidate)

                # Авторитетность
                authority = self.calculate_authority(candidate)

                # Разнообразие
                diversity = self.calculate_diversity(ranked, candidate)

                # Итоговый скор
                final_score = (
                    relevance * self.weights["relevance"] +
                    freshness * self.weights["freshness"] +
                    authority * self.weights["authority"] +
                    diversity * self.weights["diversity"]
                )

                candidate["rerank_score"] = round(final_score, 4)
                candidate["relevance"] = round(relevance, 4)
                candidate["freshness"] = round(freshness, 4)
                candidate["authority"] = round(authority, 4)
                candidate["diversity"] = round(diversity, 4)

                ranked.append(candidate)

            # Сортируем по итоговому скору
            ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

            return ranked

    reranker = Reranker()

    # Кандидаты для переранжирования
    candidates = [
        {"id": "doc1", "text": "Python для data science", "source": "docs.python.org", "created_at": "2024-01-15"},
        {"id": "doc2", "text": "Анализ данных в Python", "source": "github.com", "created_at": "2023-11-20"},
        {"id": "doc3", "text": "Машинное обучение на Python", "source": "arxiv.org", "created_at": "2024-02-01"},
        {"id": "doc4", "text": "Data science руководство", "source": "stackoverflow.com", "created_at": "2023-08-10"},
        {"id": "doc5", "text": "Python библиотеки для анализа", "source": "docs.python.org", "created_at": "2024-01-20"}
    ]

    print("  Переранжирование для запроса 'Python data science':")
    print(f"  Веса: relevance={reranker.weights['relevance']}, "
          f"freshness={reranker.weights['freshness']}, "
          f"authority={reranker.weights['authority']}, "
          f"diversity={reranker.weights['diversity']}")

    reranked = reranker.rerank("Python data science", candidates)

    print("\n  Результаты переранжирования:")
    for i, doc in enumerate(reranked):
        print(f"\n  {i+1}. {doc['id']}: \"{doc['text']}\"")
        print(f"     Итоговый скор: {doc['rerank_score']}")
        print(f"     Компоненты: relevance={doc['relevance']}, "
              f"freshness={doc['freshness']}, "
              f"authority={doc['authority']}, "
              f"diversity={doc['diversity']}")

    # --- 2.4 Фильтрация и постобработка ---
    print("\n[2.4] Фильтрация и постобработка")
    print("-" * 50)

    class ResultPostProcessor:
        """Постобработка результатов поиска."""

        def __init__(self):
            # Пороги фильтрации
            self.min_relevance = 0.1
            self.min_score = 0.05
            self.max_results = 10

        def filter_by_threshold(self, results: list) -> list:
            """Фильтрует по порогу релевантности."""
            return [r for r in results if r.get("rerank_score", 0) >= self.min_score]

        def deduplicate(self, results: list) -> list:
            """Удаляет дубликаты по содержимому."""
            seen = set()
            unique = []

            for r in results:
                # Создаём нормализованный ключ
                text_normalized = re.sub(r'\s+', ' ', r.get("text", "").lower().strip())
                text_hash = hashlib.md5(text_normalized.encode()).hexdigest()

                if text_hash not in seen:
                    seen.add(text_hash)
                    unique.append(r)

            return unique

        def diversify(self, results: list, max_per_source: int = 2) -> list:
            """Обеспечивает разнообразие источников."""
            source_counts = collections.Counter()
            diversified = []

            for r in results:
                source = r.get("source", "unknown")
                if source_counts[source] < max_per_source:
                    diversified.append(r)
                    source_counts[source] += 1

            return diversified

        def add_context(self, results: list, query: str) -> list:
            """Добавляет контекст к результатам."""
            query_tokens = set(re.findall(r'\b\w+\b', query.lower()))

            for r in results:
                text = r.get("text", "")
                text_tokens = set(re.findall(r'\b\w+\b', text.lower()))

                # Подсвечиваем совпадающие токены
                highlighted = text
                for token in query_tokens:
                    if token in text.lower():
                        highlighted = re.sub(
                            rf'\b({re.escape(token)})\b',
                            r'**\1**',
                            highlighted,
                            flags=re.IGNORECASE
                        )

                r["highlighted_text"] = highlighted
                r["overlap_tokens"] = list(query_tokens & text_tokens)

            return results

    processor = ResultPostProcessor()

    # Демонстрация постобработки
    print("  Постобработка результатов:")

    # Создаём тестовые результаты
    test_results = [
        {"id": "r1", "text": "Python для data science", "source": "docs", "rerank_score": 0.8},
        {"id": "r2", "text": "Python для data science", "source": "blog", "rerank_score": 0.75},
        {"id": "r3", "text": "Data science в Python", "source": "docs", "rerank_score": 0.6},
        {"id": "r4", "text": "Анализ данных", "source": "tutorial", "rerank_score": 0.3},
        {"id": "r5", "text": "Python библиотеки", "source": "docs", "rerank_score": 0.2},
        {"id": "r6", "text": "Машинное обучение", "source": "research", "rerank_score": 0.15}
    ]

    print(f"\n  Исходных результатов: {len(test_results)}")

    # Фильтрация
    filtered = processor.filter_by_threshold(test_results)
    print(f"  После фильтрации: {len(filtered)}")

    # Дедупликация
    deduped = processor.deduplicate(filtered)
    print(f"  После дедупликации: {len(deduped)}")

    # Разнообразие
    diversified = processor.diversify(deduped, max_per_source=2)
    print(f"  После обеспечения разнообразия: {len(diversified)}")

    # Добавление контекста
    enriched = processor.add_context(diversified, "Python data science")

    print("\n  Финальные результаты:")
    for r in enriched:
        print(f"    {r['id']}: \"{r.get('highlighted_text', r['text'])}\"")
        print(f"      Совпадающие токены: {r.get('overlap_tokens', [])}")


# ============================================================
# Демо 3: RAG Pipeline
# ============================================================
def demo_rag_pipeline():
    """Демонстрация полного RAG pipeline."""
    print("\n" + "=" * 70)
    print("ДЕМО 3: RAG PIPELINE")
    print("=" * 70)

    # --- 3.1 Query Processing ---
    print("\n[3.1] Обработка запроса (Query Processing)")
    print("-" * 50)

    class QueryProcessor:
        """Обработка и улучшение запроса."""

        def __init__(self):
            # Шаблоны для определения типа запроса
            self.query_types = {
                "factual": ["что", "какой", "кто", "где", "когда", "what", "which", "who"],
                "explanatory": ["объясни", "опиши", "почему", "explain", "describe", "why"],
                "procedural": ["как", "каким образом", "пошагово", "how", "steps"]
            }

        def analyze_query(self, query: str) -> dict:
            """Анализирует тип и намерение запроса."""
            query_lower = query.lower()
            words = set(re.findall(r'\b\w+\b', query_lower))

            # Определяем тип запроса
            detected_types = []
            for qtype, keywords in self.query_types.items():
                if words & set(keywords):
                    detected_types.append(qtype)

            # Извлекаем ключевые слова (исключая стоп-слова)
            stop_words = {"что", "это", "как", "и", "или", "но", "в", "на", "для",
                         "what", "is", "the", "a", "an", "and", "or", "for", "in"}
            keywords = words - stop_words

            return {
                "original": query,
                "keywords": list(keywords),
                "query_type": detected_types[0] if detected_types else "general",
                "word_count": len(words),
                "complexity": "simple" if len(words) < 8 else "complex"
            }

        def rewrite_query(self, query: str, context: dict = None) -> list:
            """Генерирует варианты переписанного запроса."""
            analysis = self.analyze_query(query)
            rewrites = []

            # Вариант 1: Добавление контекста
            if context:
                rewrite1 = f"{query} ({', '.join(context.get('topics', [])[:2])})"
                rewrites.append(rewrite1)

            # Вариант 2: Уточнение по типу
            if analysis["query_type"] == "factual":
                rewrite2 = f"факты о {query}"
                rewrites.append(rewrite2)
            elif analysis["query_type"] == "explanatory":
                rewrite2 = f"объяснение: {query}"
                rewrites.append(rewrite2)

            # Вариант 3: Ключевые слова
            if analysis["keywords"]:
                rewrite3 = " ".join(analysis["keywords"][:5])
                rewrites.append(rewrite3)

            return rewrites

    query_processor = QueryProcessor()

    # Тестовые запросы
    queries = [
        "Что такое машинное обучение?",
        "Объясни, как работают нейронные сети",
        "Как реализовать алгоритм кластеризации на Python?"
    ]

    print("  Анализ запросов:")
    for query in queries:
        analysis = query_processor.analyze_query(query)
        print(f"\n  Запрос: \"{query}\"")
        print(f"    Тип: {analysis['query_type']}")
        print(f"    Ключевые слова: {analysis['keywords'][:5]}")
        print(f"    Сложность: {analysis['complexity']}")

        rewrites = query_processor.rewrite_query(query, {"topics": ["AI", "Python"]})
        if rewrites:
            print(f"    Варианты переписывания:")
            for r in rewrites:
                print(f"      - \"{r}\"")

    # --- 3.2 Retrieval ---
    print("\n[3.2] Извлечение документов (Retrieval)")
    print("-" * 50)

    class DocumentRetriever:
        """Извлечение релевантных документов."""

        def __init__(self):
            self.documents = []
            self.index = {}  # Простой индекс

        def add_documents(self, docs: list):
            """Добавляет документы в базу."""
            for doc in docs:
                self.documents.append(doc)
                # Индексируем по ключевым словам
                tokens = set(re.findall(r'\b\w+\b', doc.get("text", "").lower()))
                for token in tokens:
                    if token not in self.index:
                        self.index[token] = []
                    self.index[token].append(len(self.documents) - 1)

        def retrieve(self, query: str, top_k: int = 5) -> list:
            """Извлекает релевантные документы."""
            query_tokens = set(re.findall(r'\b\w+\b', query.lower()))

            # Подсчёт релевантности
            doc_scores = collections.Counter()
            for token in query_tokens:
                if token in self.index:
                    for doc_idx in self.index[token]:
                        doc_scores[doc_idx] += 1

            # Сортировка и выбор top-k
            top_docs = doc_scores.most_common(top_k)

            results = []
            for doc_idx, score in top_docs:
                doc = self.documents[doc_idx].copy()
                doc["relevance_score"] = score / max(len(query_tokens), 1)
                doc["retrieval_rank"] = len(results) + 1
                results.append(doc)

            return results

    retriever = DocumentRetriever()

    # Документы для извлечения
    knowledge_docs = [
        {"id": "d1", "text": "Машинное обучение — область AI, которая позволяет компьютерам учиться на данных",
         "source": "wiki", "title": "Введение в ML"},
        {"id": "d2", "text": "Нейронные сети вдохновлены строением мозга человека",
         "source": "article", "title": "Нейронные сети"},
        {"id": "d3", "text": "Python — популярный язык для data science и анализа данных",
         "source": "docs", "title": "Python для DS"},
        {"id": "d4", "text": "TensorFlow и PyTorch — основные фреймворки для deep learning",
         "source": "docs", "title": "Фреймворки ML"},
        {"id": "d5", "text": "Алгоритмы сортировки важны для эффективной обработки данных",
         "source": "textbook", "title": "Алгоритмы"}
    ]

    print("  Индексация документов:")
    retriever.add_documents(knowledge_docs)
    print(f"    Добавлено документов: {len(knowledge_docs)}")
    print(f"    Размер индекса: {len(retriever.index)} терминов")

    # Извлечение
    print("\n  Извлечение для запроса 'машинное обучение Python':")
    results = retriever.retrieve("машинное обучение Python", top_k=3)
    for r in results:
        print(f"    {r['retrieval_rank']}. {r['id']} (релевантность: {r['relevance_score']:.2f})")
        print(f"       \"{r['text'][:60]}...\"")
        print(f"       Источник: {r['source']}")

    # --- 3.3 Augmentation ---
    print("\n[3.3] Аугментация контекста (Context Augmentation)")
    print("-" * 50)

    class ContextAugmentor:
        """Аугментация контекста для генерации ответа."""

        def __init__(self):
            self.max_context_tokens = 2000
            self.separators = {
                "document": "\n\n---\n\n",
                "section": "\n\n",
                "paragraph": "\n"
            }

        def augment_context(self, query: str, documents: list,
                           strategy: str = "concatenate") -> dict:
            """Аугментирует контекст из документов."""
            # Формируем промпт с контекстом
            context_parts = []
            total_chars = 0

            for doc in documents:
                text = doc.get("text", "")
                source = doc.get("source", "unknown")
                title = doc.get("title", "Untitled")

                # Форматируем документ
                doc_text = f"[{source}] {title}:\n{text}"
                context_parts.append(doc_text)
                total_chars += len(doc_text)

            # Объединяем по стратегии
            if strategy == "concatenate":
                full_context = self.separators["document"].join(context_parts)
            elif strategy == "ranked":
                # Сортируем по релевантности (предполагаем, что они уже отсортированы)
                full_context = self.separators["document"].join(context_parts)
            else:
                full_context = self.separators["section"].join(context_parts)

            # Формируем финальный промпт
            augmented_prompt = f"""Контекст из базы знаний:
{full_context}

Вопрос: {query}

На основе предоставленного контекста, дай подробный ответ. Если информации недостаточно, скажи об этом."""

            return {
                "prompt": augmented_prompt,
                "context_length": len(full_context),
                "documents_used": len(documents),
                "strategy": strategy,
                "token_estimate": len(augmented_prompt.split()) * 1.3  # Грубая оценка
            }

    augmentor = ContextAugmentor()

    # Аугментация контекста
    print("  Аугментация контекста для запроса 'что такое ML':")
    relevant_docs = results[:2]  # Берём топ-2 документа

    augmentation = augmentor.augment_context(
        "что такое машинное обучение",
        relevant_docs,
        strategy="concatenate"
    )

    print(f"\n  Стратегия: {augmentation['strategy']}")
    print(f"  Документов использовано: {augmentation['documents_used']}")
    print(f"  Длина контекста: {augmentation['context_length']} символов")
    print(f"  Оценка токенов: {augmentation['token_estimate']:.0f}")

    print(f"\n  Сформированный промпт (фрагмент):")
    print(f"  {augmentation['prompt'][:200]}...")

    # --- 3.4 Generation ---
    print("\n[3.4] Генерация ответа (Response Generation)")
    print("-" * 50)

    class ResponseGenerator:
        """Генерация ответа на основе контекста."""

        def __init__(self):
            self.response_templates = {
                "factual": "На основе имеющейся информации: {answer}. {sources}",
                "explanatory": "Давайте разберём это подробнее. {answer} {sources}",
                "procedural": "Вот как это работает: {answer} {sources}"
            }

        def generate(self, query: str, context: dict, query_type: str = "factual") -> dict:
            """Генерирует ответ (эмуляция)."""
            # В реальности здесь был бы вызов LLM
            # Для демонстрации формируем ответ на основе контекста

            # Извлекаем информацию из контекста
            documents = context.get("documents", [])
            relevant_info = [doc.get("text", "") for doc in documents[:2]]

            # Формируем ответ
            answer_parts = []
            for info in relevant_info:
                # Берём первое предложение как ключевую информацию
                first_sentence = re.split(r'[.!?]', info)[0]
                if first_sentence:
                    answer_parts.append(first_sentence.strip())

            answer = ". ".join(answer_parts)

            # Формируем список источников
            sources = []
            for doc in documents[:3]:
                sources.append(f"[{doc.get('source', 'source')}] {doc.get('title', 'doc')}")

            # Применяем шаблон
            template = self.response_templates.get(query_type,
                                                  self.response_templates["factual"])

            response_text = template.format(
                answer=answer,
                sources="Источники: " + ", ".join(sources) if sources else ""
            )

            return {
                "response": response_text,
                "query_type": query_type,
                "sources": sources,
                "confidence": 0.85,  # Эмуляция
                "tokens_used": len(response_text.split()) * 1.5
            }

    generator = ResponseGenerator()

    # Генерация ответа
    print("  Генерация ответа:")
    context = {
        "documents": relevant_docs,
        "query": "что такое машинное обучение"
    }

    response = generator.generate(
        "что такое машинное обучение",
        context,
        query_type="explanatory"
    )

    print(f"\n  Ответ:")
    print(f"  {response['response']}")
    print(f"\n  Мета:")
    print(f"    Тип запроса: {response['query_type']}")
    print(f"    Уверенность: {response['confidence']}")
    print(f"    Источники: {response['sources']}")

    # --- 3.5 Полный pipeline ---
    print("\n[3.5] Полный RAG Pipeline")
    print("-" * 50)

    class RAGPipeline:
        """Полный RAG pipeline: query → retrieve → augment → generate."""

        def __init__(self):
            self.query_processor = QueryProcessor()
            self.retriever = DocumentRetriever()
            self.augmentor = ContextAugmentor()
            self.generator = ResponseGenerator()
            self.metrics = {"queries_processed": 0, "avg_latency": 0}

        def initialize(self, documents: list):
            """Инициализирует базу знаний."""
            self.retriever.add_documents(documents)
            print(f"  RAG Pipeline инициализирован: {len(documents)} документов")

        def process_query(self, query: str) -> dict:
            """Обрабатывает запрос через весь pipeline."""
            start_time = time.time()

            # Шаг 1: Обработка запроса
            query_analysis = self.query_processor.analyze_query(query)

            # Шаг 2: Извлечение документов
            retrieved = self.retriever.retrieve(query, top_k=3)

            # Шаг 3: Аугментация контекста
            augmentation = self.augmentor.augment_context(
                query, retrieved, strategy="concatenate"
            )

            # Шаг 4: Генерация ответа
            context = {
                "documents": retrieved,
                "augmented_prompt": augmentation["prompt"]
            }
            response = self.generator.generate(
                query, context, query_type=query_analysis["query_type"]
            )

            # Метрики
            latency = (time.time() - start_time) * 1000
            self.metrics["queries_processed"] += 1
            self.metrics["avg_latency"] = (
                (self.metrics["avg_latency"] * (self.metrics["queries_processed"] - 1) + latency)
                / self.metrics["queries_processed"]
            )

            return {
                "query": query,
                "analysis": query_analysis,
                "retrieved_docs": len(retrieved),
                "augmented": augmentation,
                "response": response,
                "latency_ms": round(latency, 2)
            }

    pipeline = RAGPipeline()
    pipeline.initialize(knowledge_docs)

    # Обработка запросов
    print("\n  Обработка тестовых запросов:")
    test_queries = [
        "Что такое нейронные сети?",
        "Какие фреймворки используются для ML?",
        "Почему Python популярен в data science?"
    ]

    for query in test_queries:
        print(f"\n  Запрос: \"{query}\"")
        result = pipeline.process_query(query)
        print(f"    Тип: {result['analysis']['query_type']}")
        print(f"    Извлечено документов: {result['retrieved_docs']}")
        print(f"    Ответ: {result['response']['response'][:80]}...")
        print(f"    Задержка: {result['latency_ms']} мс")

    print(f"\n  Метрики pipeline:")
    print(f"    Всего обработано: {pipeline.metrics['queries_processed']}")
    print(f"    Средняя задержка: {pipeline.metrics['avg_latency']:.2f} мс")


# ============================================================
# Демо 4: Качество RAG
# ============================================================
def demo_rag_quality():
    """Демонстрация метрик и улучшения качества RAG."""
    print("\n" + "=" * 70)
    print("ДЕМО 4: КАЧЕСТВО RAG (RAG Quality)")
    print("=" * 70)

    # --- 4.1 Faithfulness (Верность источнику) ---
    print("\n[4.1] Faithfulness (Верность источникам)")
    print("-" * 50)

    class FaithfulnessEvaluator:
        """Оценка верности ответа источникам."""

        def __init__(self):
            # Паттерны, указывающие на галлюцинации
            self.hallucination_patterns = [
                r"я\s+(думаю|считаю|полагаю)",  # Субъективные утверждения
                r"возможно",                       # Неуверенность
                r"наверное",
                r"точно\s+не\s+знаю",
                r"казнится",                       # Опечатки
            ]

        def evaluate(self, response: str, sources: list) -> dict:
            """Оценивает верность ответа источникам."""
            response_lower = response.lower()

            # Извлекаем ключевые утверждения из ответа
            sentences = re.split(r'[.!?]', response)
            claims = [s.strip() for s in sentences if len(s.strip()) > 10]

            # Проверяем каждое утверждение на наличие в источниках
            source_text = " ".join(sources).lower()
            supported_claims = 0
            unsupported_claims = 0

            for claim in claims:
                claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
                source_words = set(re.findall(r'\b\w+\b', source_text))

                # Если хотя бы 30% слов из утверждения есть в источниках
                overlap = len(claim_words & source_words) / max(len(claim_words), 1)
                if overlap >= 0.3:
                    supported_claims += 1
                else:
                    unsupported_claims += 1

            # Проверка на галлюцинации
            hallucination_count = 0
            for pattern in self.hallucination_patterns:
                if re.search(pattern, response_lower):
                    hallucination_count += 1

            # Итоговая оценка
            total_claims = len(claims)
            faithfulness_score = supported_claims / max(total_claims, 1)

            return {
                "faithfulness_score": round(faithfulness_score, 3),
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
                "hallucination_indicators": hallucination_count,
                "verdict": "HIGH" if faithfulness_score > 0.7
                          else "MEDIUM" if faithfulness_score > 0.4
                          else "LOW"
            }

    faithfulness_eval = FaithfulnessEvaluator()

    # Тестовые ответы и источники
    test_cases = [
        {
            "response": "Python — язык программирования, созданный в 1991 году. Он широко используется в AI.",
            "sources": ["Python создан в 1991 году Гвидо ван Россумом", "Python используется в AI и data science"],
            "label": "Хороший ответ (верный)"
        },
        {
            "response": "Я думаю, что Python был создан в 2005 году. Возможно, его создал Гейтс.",
            "sources": ["Python создан в 1991 году Гвидо ван Россумом"],
            "label": "Плохой ответ (галлюцинации)"
        },
        {
            "response": "Нейронные сети работают как мозг. Они используют веса и активации для обучения.",
            "sources": ["Нейронные сети вдохновлены биологическими нейронами", "Веса определяют силу связи"],
            "label": "Средний ответ (частично верный)"
        }
    ]

    print("  Оценка верности ответов:")
    for i, case in enumerate(test_cases):
        result = faithfulness_eval.evaluate(case["response"], case["sources"])
        print(f"\n  Тест {i+1}: {case['label']}")
        print(f"    Ответ: \"{case['response'][:60]}...\"")
        print(f"    Оценка верности: {result['faithfulness_score']}")
        print(f"    Подтверждённых утверждений: {result['supported_claims']}/{result['total_claims']}")
        print(f"    Индикаторы галлюцинаций: {result['hallucination_indicators']}")
        print(f"    Вердикт: {result['verdict']}")

    # --- 4.2 Relevance (Релевантность) ---
    print("\n[4.2] Relevance (Релевантность)")
    print("-" * 50)

    class RelevanceEvaluator:
        """Оценка релевантности ответа запросу."""

        def __init__(self):
            # Стоп-слова для фильтрации
            self.stop_words = {
                "что", "это", "как", "и", "или", "но", "в", "на", "для",
                "is", "the", "a", "an", "and", "or", "for", "in", "of"
            }

        def evaluate(self, query: str, response: str) -> dict:
            """Оценивает релевантность ответа."""
            query_words = set(re.findall(r'\b\w+\b', query.lower())) - self.stop_words
            response_words = set(re.findall(r'\b\w+\b', response.lower())) - self.stop_words

            if not query_words:
                return {"relevance_score": 0, "verdict": "LOW"}

            # Jaccard similarity
            intersection = query_words & response_words
            union = query_words | response_words
            jaccard = len(intersection) / max(len(union), 1)

            # Взвешенное перекрытие (слова запроса важнее)
            query_coverage = len(intersection) / max(len(query_words), 1)
            response_coverage = len(intersection) / max(len(response_words), 1)

            # Итоговый скор (среднее с весами)
            relevance_score = (jaccard * 0.3 +
                             query_coverage * 0.5 +
                             response_coverage * 0.2)

            return {
                "relevance_score": round(relevance_score, 3),
                "query_coverage": round(query_coverage, 3),
                "response_coverage": round(response_coverage, 3),
                "jaccard_similarity": round(jaccard, 3),
                "matching_words": list(intersection),
                "verdict": "HIGH" if relevance_score > 0.6
                          else "MEDIUM" if relevance_score > 0.3
                          else "LOW"
            }

    relevance_eval = RelevanceEvaluator()

    # Тестовые случаи
    relevance_tests = [
        ("что такое машинное обучение", "Машинное обучение — область AI, которая позволяет системам учиться"),
        ("как работает Python", "Python — интерпретируемый язык программирования"),
        ("преимущества нейронных сетей", "Нейронные сети могут обучаться на данных")
    ]

    print("  Оценка релевантности:")
    for query, response in relevance_tests:
        result = relevance_eval.evaluate(query, response)
        print(f"\n  Запрос: \"{query}\"")
        print(f"  Ответ: \"{response[:50]}...\"")
        print(f"    Релевантность: {result['relevance_score']}")
        print(f"    Покрытие запроса: {result['query_coverage']}")
        print(f"    Совпадающие слова: {result['matching_words']}")
        print(f"    Вердикт: {result['verdict']}")

    # --- 4.3 Source Attribution (Атрибуция источников) ---
    print("\n[4.3] Source Attribution (Атрибуция источников)")
    print("-" * 50)

    class SourceAttributor:
        """Атрибуция утверждений источникам."""

        def __init__(self):
            # Паттерны для определения цитирования
            self.citation_patterns = [
                r"\[(.+?)\]",
                r"источник:\s*(.+?)(?:\.|$)",
                r"согласно\s+(.+?)(?:,|\.)",
                r"по данным\s+(.+?)(?:,|\.)"
            ]

        def extract_claims(self, response: str) -> list:
            """Извлекает утверждения из ответа."""
            sentences = re.split(r'[.!?]', response)
            claims = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:  # Игнорируем слишком короткие
                    claims.append({
                        "text": sentence,
                        "words": set(re.findall(r'\b\w+\b', sentence.lower()))
                    })

            return claims

        def attribute_to_sources(self, claims: list, sources: dict) -> list:
            """Атрибутирует утверждения источникам."""
            attributed = []

            for claim in claims:
                best_match = None
                best_score = 0

                for source_id, source_text in sources.items():
                    source_words = set(re.findall(r'\b\w+\b', source_text.lower()))
                    claim_words = claim["words"]

                    if claim_words and source_words:
                        overlap = len(claim_words & source_words) / max(len(claim_words), 1)
                        if overlap > best_score:
                            best_score = overlap
                            best_match = source_id

                attributed.append({
                    "claim": claim["text"],
                    "source": best_match,
                    "confidence": round(best_score, 3),
                    "is_attributed": best_score > 0.3
                })

            return attributed

        def evaluate_attribution(self, attributed_claims: list) -> dict:
            """Оценивает качество атрибуции."""
            total = len(attributed_claims)
            attributed = sum(1 for c in attributed_claims if c["is_attributed"])
            avg_confidence = sum(c["confidence"] for c in attributed_claims) / max(total, 1)

            return {
                "total_claims": total,
                "attributed_claims": attributed,
                "attribution_rate": round(attributed / max(total, 1), 3),
                "avg_confidence": round(avg_confidence, 3),
                "verdict": "GOOD" if attributed / max(total, 1) > 0.7
                          else "PARTIAL" if attributed / max(total, 1) > 0.4
                          else "POOR"
            }

    attributor = SourceAttributor()

    # Тестовый ответ с источниками
    test_response = """
    Python был создан в 1991 году. Язык широко используется в data science.
    Нейронные сети вдохновлены работой мозга. TensorFlow — популярный фреймворк.
    """

    test_sources = {
        "wiki_python": "Python — язык программирования, созданный Гвидо ван Россумом в 1991 году",
        "wiki_ml": "Машинное обучение использует нейронные сети для обучения",
        "docs_tf": "TensorFlow — открытый фреймворк для машинного обучения от Google"
    }

    print("  Атрибуция утверждений:")
    print(f"\n  Ответ:\n  {test_response.strip()}")

    # Извлекаем утверждения
    claims = attributor.extract_claims(test_response)
    print(f"\n  Извлечено утверждений: {len(claims)}")
    for c in claims:
        print(f"    - \"{c['text'][:50]}...\"")

    # Атрибуция к источникам
    attributed = attributor.attribute_to_sources(claims, test_sources)
    print("\n  Атрибуция к источникам:")
    for a in attributed:
        status = "✓" if a["is_attributed"] else "?"
        print(f"    {status} \"{a['claim'][:40]}...\" → {a['source']} "
              f"(уверенность: {a['confidence']})")

    # Оценка атрибуции
    eval_result = attributor.evaluate_attribution(attributed)
    print(f"\n  Оценка атрибуции:")
    print(f"    Всего утверждений: {eval_result['total_claims']}")
    print(f"    Атрибутировано: {eval_result['attributed_claims']}")
    print(f"    Процент атрибуции: {eval_result['attribution_rate'] * 100}%")
    print(f"    Средняя уверенность: {eval_result['avg_confidence']}")
    print(f"    Вердикт: {eval_result['verdict']}")

    # --- 4.4 Комплексная оценка качества ---
    print("\n[4.4] Комплексная оценка качества RAG")
    print("-" * 50)

    class RAGQualityEvaluator:
        """Комплексная оценка качества RAG pipeline."""

        def __init__(self):
            self.faithfulness_eval = FaithfulnessEvaluator()
            self.relevance_eval = RelevanceEvaluator()
            self.attributor = SourceAttributor()
            self.weights = {
                "faithfulness": 0.4,
                "relevance": 0.35,
                "attribution": 0.25
            }

        def evaluate(self, query: str, response: str, sources: list) -> dict:
            """Комплексная оценка."""
            # 1. Faithfulness
            faithfulness = self.faithfulness_eval.evaluate(response, sources)

            # 2. Relevance
            relevance = self.relevance_eval.evaluate(query, response)

            # 3. Attribution
            source_dict = {f"src_{i}": s for i, s in enumerate(sources)}
            claims = self.attributor.extract_claims(response)
            attributed = self.attributor.attribute_to_sources(claims, source_dict)
            attribution = self.attributor.evaluate_attribution(attributed)

            # Итоговый скор
            overall_score = (
                faithfulness["faithfulness_score"] * self.weights["faithfulness"] +
                relevance["relevance_score"] * self.weights["relevance"] +
                attribution["attribution_rate"] * self.weights["attribution"]
            )

            return {
                "overall_score": round(overall_score, 3),
                "components": {
                    "faithfulness": faithfulness["faithfulness_score"],
                    "relevance": relevance["relevance_score"],
                    "attribution": attribution["attribution_rate"]
                },
                "details": {
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "attribution": attribution
                },
                "verdict": "EXCELLENT" if overall_score > 0.8
                          else "GOOD" if overall_score > 0.6
                          else "FAIR" if overall_score > 0.4
                          else "POOR"
            }

        def compare_responses(self, query: str, responses: list, sources: list) -> dict:
            """Сравнивает качество разных ответов."""
            evaluations = []

            for i, response in enumerate(responses):
                eval_result = self.evaluate(query, response, sources)
                evaluations.append({
                    "response_id": i + 1,
                    "response_preview": response[:50],
                    "score": eval_result["overall_score"],
                    "verdict": eval_result["verdict"]
                })

            # Находим лучший ответ
            best = max(evaluations, key=lambda x: x["score"])

            return {
                "evaluations": evaluations,
                "best_response": best,
                "score_difference": round(
                    max(e["score"] for e in evaluations) -
                    min(e["score"] for e in evaluations), 3
                )
            }

    quality_evaluator = RAGQualityEvaluator()

    # Тестовые ответы для сравнения
    query = "Что такое Python и где он используется?"
    test_responses = [
        "Python — язык программирования, созданный в 1991 году. Он широко используется в data science, AI и веб-разработке.",
        "Я не уверен, но Python — это что-то про программирование. Возможно, его используют где-то.",
        "Python [wiki] — высокоуровневый язык программирования [docs], популярный в machine learning [arxiv]."
    ]

    sources = [
        "Python — язык программирования, созданный в 1991 году",
        "Python используется в data science, AI и веб-разработке",
        "Python популярен благодаря простоте и читаемости"
    ]

    print("  Сравнение ответов:")
    print(f"  Запрос: \"{query}\"")

    comparison = quality_evaluator.compare_responses(query, test_responses, sources)

    for i, eval_result in enumerate(comparison["evaluations"]):
        print(f"\n  Ответ {eval_result['response_id']}:")
        print(f"    \"{eval_result['response_preview']}...\"")
        print(f"    Оценка: {eval_result['score']}")
        print(f"    Вердикт: {eval_result['verdict']}")

    print(f"\n  Лучший ответ: №{comparison['best_response']['response_id']}")
    print(f"  Разброс оценок: {comparison['score_difference']}")

    # Детальный разбор лучшего ответа
    print("\n" + "-" * 50)
    print("  Детальный разбор лучшего ответа:")
    best_idx = comparison["best_response"]["response_id"] - 1
    detailed = quality_evaluator.evaluate(query, test_responses[best_idx], sources)

    print(f"\n  Общая оценка: {detailed['overall_score']}")
    print(f"  Компоненты:")
    for comp, score in detailed["components"].items():
        print(f"    {comp}: {score}")

    print(f"\n  Формула итоговой оценки:")
    print(f"    overall = faithfulness × {quality_evaluator.weights['faithfulness']}")
    print(f"           + relevance × {quality_evaluator.weights['relevance']}")
    print(f"           + attribution × {quality_evaluator.weights['attribution']}")

    # Рекомендации по улучшению
    print("\n  Рекомендации по улучшению:")
    if detailed["components"]["faithfulness"] < 0.6:
        print("    - Улучшитьfaithfulness: добавить больше контекста из источников")
    if detailed["components"]["relevance"] < 0.5:
        print("    - Улучшитьrelevance: уточнить запрос или переформулировать ответ")
    if detailed["components"]["attribution"] < 0.5:
        print("    - Улучшитьattribution: добавить явные ссылки на источники")
    if detailed["overall_score"] > 0.7:
        print("    - Качество хорошее, можно оптимизировать скорость")


# ============================================================
# Главная функция
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Модуль 167: RAG for Agents")
    print("Retrieval-Augmented Generation, базы знаний для агентов")
    print("=" * 70)

    demo_knowledge_base_design()
    demo_retrieval_strategies()
    demo_rag_pipeline()
    demo_rag_quality()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены!")
    print("=" * 70)