"""
RAG (Retrieval-Augmented Generation) — основы с нуля.

Самодостаточный файл: векторный поиск, retrieval, генерация с контекстом.
Без внешних зависимостей (numpy, torch, transformers, faiss, chromadb).
"""

import math
import re
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

random.seed(42)


# ============================================================
# 1. Векторное пространство документов (TF-IDF + Cosine)
# ============================================================

class SimpleTokenizer:
    """Простой токенизатор: нижний регистр + разбиение по словам."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


class TFIDFVectorizer:
    """Минимальный TF-IDF векторизатор на чистом Python."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.docs_tokens: List[List[str]] = []

    def fit(self, documents: List[str]) -> None:
        self.docs_tokens = [SimpleTokenizer.tokenize(doc) for doc in documents]
        n_docs = len(documents)

        # Подсчёт文档频率
        df: Dict[str, int] = Counter()
        for tokens in self.docs_tokens:
            unique = set(tokens)
            for t in unique:
                df[t] += 1

        # IDF: log(N / df) + 1 (сглаживание)
        all_words = set()
        for tokens in self.docs_tokens:
            all_words.update(tokens)

        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        self.idf = {}
        for word in self.vocab:
            self.idf[word] = math.log((n_docs + 1) / (df.get(word, 0) + 1)) + 1

    def transform(self, text: str) -> List[float]:
        tokens = SimpleTokenizer.tokenize(text)
        tf = Counter(tokens)
        n_terms = len(tokens) if tokens else 1

        vec = [0.0] * len(self.vocab)
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf_val = count / n_terms
                vec[idx] = tf_val * self.idf.get(word, 1.0)
        return vec

    def transform_batch(self, documents: List[str]) -> List[List[float]]:
        return [self.transform(doc) for doc in documents]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Косинусное сходство между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# 2. Индекс документов
# ============================================================

class DocumentIndex:
    """Индекс документов с TF-IDF векторизацией и поиском."""

    def __init__(self):
        self.vectorizer = TFIDFVectorizer()
        self.documents: List[Dict] = []
        self.vectors: List[List[float]] = []

    def add_documents(self, documents: List[Dict]) -> None:
        """Добавить документы в индекс. Каждый документ — {'id': ..., 'text': ..., 'metadata': ...}."""
        self.documents.extend(documents)
        texts = [doc['text'] for doc in documents]

        if len(self.vectorizer.vocab) == 0:
            self.vectorizer.fit(texts)
        else:
            # Переобучаем на расширенном корпусе
            all_texts = [doc['text'] for doc in self.documents]
            self.vectorizer.fit(all_texts)

        self.vectors = self.vectorizer.transform_batch([doc['text'] for doc in self.documents])

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Поиск наиболее релевантных документов."""
        query_vec = self.vectorizer.transform(query)

        scores = []
        for i, doc_vec in enumerate(self.vectors):
            sim = cosine_similarity(query_vec, doc_vec)
            scores.append((self.documents[i], sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 3. RAG: Retrieval + Generation
# ============================================================

class SimpleLLM:
    """Заглушка LLM: шаблонная генерация на основе retrieved контекста."""

    def generate(self, query: str, context_docs: List[Dict]) -> str:
        """
        Генерирует ответ, основываясь на query и найденных документах.
        В реальном RAG здесь был бы вызов GPT/Claude/local LLM.
        """
        if not context_docs:
            return f"К сожалению, релевантных документов по запросу '{query}' не найдено."

        # Извлекаем ключевые предложения из документов
        key_sentences = []
        for doc in context_docs[:3]:
            sentences = SimpleTokenizer.split_sentences(doc['text'])
            # Берём самое длинное предложение (обычно самое информативное)
            if sentences:
                best = max(sentences, key=len)
                key_sentences.append(best)

        # Формируем ответ
        header = f"На основе найденных документов:\n"
        body = "\n".join(f"  • {s}" for s in key_sentences)
        footer = f"\n\nВопрос: {query}"

        return header + body + footer


class RAGPipeline:
    """Полный RAG pipeline: индексация → retrieval → generation."""

    def __init__(self):
        self.index = DocumentIndex()
        self.llm = SimpleLLM()

    def add_documents(self, documents: List[Dict]) -> None:
        self.index.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        return self.index.search(query, top_k)

    def generate(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k)
        context_docs = [doc for doc, _ in results]
        return self.llm.generate(query, context_docs)


# ============================================================
# 4. Метрики оценки качества retrieval
# ============================================================

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Доля релевантных документов среди top-k найденных."""
    retrieved_at_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Какую долю всех релевантных документов удалось найти."""
    retrieved_at_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank — на каком месте первый релевантный документ."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# ДЕМОНСТРАЦИЯ
# ============================================================

def demo_1_indexing():
    """Демо 1: Индексация документов."""
    print("=" * 70)
    print("ДЕМО 1: Индексация документов (TF-IDF векторизация)")
    print("=" * 70)

    documents = [
        {
            "id": "doc_1",
            "text": "Python — высокоуровневый язык программирования. Он широко используется в Data Science и машинном обучении.",
            "metadata": {"topic": "programming", "language": "ru"}
        },
        {
            "id": "doc_2",
            "text": "Нейронные сети — вычислительные модели, вдохновлённые структурой мозга. Они лежат в основе deep learning.",
            "metadata": {"topic": "ml", "language": "ru"}
        },
        {
            "id": "doc_3",
            "text": "PostgreSQL — объектно-реляционная система управления базами данных. Поддерживает JSON и полнотекстовый поиск.",
            "metadata": {"topic": "databases", "language": "ru"}
        },
        {
            "id": "doc_4",
            "text": "Трансформеры — архитектура нейронных сетей на основе механизма внимания. GPT и BERT построены на трансформерах.",
            "metadata": {"topic": "ml", "language": "ru"}
        },
        {
            "id": "doc_5",
            "text": "Docker — платформа для контейнеризации приложений. Упрощает развёртывание и масштабирование сервисов.",
            "metadata": {"topic": "devops", "language": "ru"}
        },
        {
            "id": "doc_6",
            "text": "RAG — метод генерации ответов, combining retrieval из базы знаний и генеративной модели. Позволяет работать с актуальными данными.",
            "metadata": {"topic": "ml", "language": "ru"}
        },
        {
            "id": "doc_7",
            "text": "Векторные базы данных (Pinecone, Milvus) хранят эмбеддинги и поддерживают приближённый поиск ближайших соседей.",
            "metadata": {"topic": "databases", "language": "ru"}
        },
    ]

    index = DocumentIndex()
    index.add_documents(documents)

    print(f"\nПроиндексировано документов: {len(index.documents)}")
    print(f"Размер словаря (vocab): {len(index.vectorizer.vocab)}")
    print(f"Количество векторов: {len(index.vectors)}")

    # Показываем ненулевые компоненты TF-IDF для первого документа
    doc_vec = index.vectors[0]
    vocab = index.vectorizer.vocab
    non_zero = [(w, doc_vec[idx]) for w, idx in vocab.items() if doc_vec[idx] > 0]
    non_zero.sort(key=lambda x: x[1], reverse=True)

    print(f"\nТоп-5 TF-IDF токенов для doc_1:")
    for word, score in non_zero[:5]:
        print(f"  {word:20s} → {score:.4f}")

    print()


def demo_2_semantic_search():
    """Демо 2: Поиск по семантике."""
    print("=" * 70)
    print("ДЕМО 2: Поиск по семантике (cosine similarity)")
    print("=" * 70)

    documents = [
        {"id": "doc_1", "text": "Python — язык программирования для data science и анализа данных.", "metadata": {}},
        {"id": "doc_2", "text": "Нейронные сети используются для классификации изображений и обработки естественного языка.", "metadata": {}},
        {"id": "doc_3", "text": "PostgreSQL поддерживает JSONB и полнотекстовый поиск по документам.", "metadata": {}},
        {"id": "doc_4", "text": "Трансформеры — архитектура для обработки последовательностей на основе self-attention.", "metadata": {}},
        {"id": "doc_5", "text": "Docker позволяет упаковать приложение в контейнер для деплоя.", "metadata": {}},
        {"id": "doc_6", "text": "RAG combining retrieval из knowledge base с генеративной моделью для ответов на вопросы.", "metadata": {}},
        {"id": "doc_7", "text": "Векторные индексы ускоряют поиск похожих документов в миллионах записей.", "metadata": {}},
    ]

    index = DocumentIndex()
    index.add_documents(documents)

    queries = [
        "машинное обучение и нейросети",
        "базы данных и SQL",
        "контейнеры и деплой",
        "поиск информации по документам",
    ]

    for query in queries:
        print(f"\n🔍 Запрос: \"{query}\"")
        print("-" * 50)
        results = index.search(query, top_k=3)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"  {rank}. [{score:.4f}] {doc['text'][:80]}...")
    print()


def demo_3_generation():
    """Демо 3: Генерация с контекстом."""
    print("=" * 70)
    print("ДЕМО 3: Генерация с контекстом (RAG pipeline)")
    print("=" * 70)

    documents = [
        {"id": "doc_1", "text": "Python поддерживает множественные парадигмы: ООП, функциональное программирование. Используется в AI.", "metadata": {}},
        {"id": "doc_2", "text": "Нейронные сети состоят из слоёв: входной, скрытые, выходной. Обучаются методом обратного распространения ошибки.", "metadata": {}},
        {"id": "doc_3", "text": "PostgreSQL — надёжная БД с поддержкой транзакций, индексов и расширений (PostGIS, pg_trgm).", "metadata": {}},
        {"id": "doc_4", "text": "Трансформеры заменили RNN/LSTM для обработки текста. Self-attention позволяет учитывать весь контекст.", "metadata": {}},
        {"id": "doc_5", "text": "Docker и Kubernetes — стандарт для оркестрации контейнеров в продакшене.", "metadata": {}},
        {"id": "doc_6", "text": "RAG — Retrieval-Augmented Generation. Метод: сначала ищем релевантные документы, затем генерируем ответ с их учётом.", "metadata": {}},
    ]

    pipeline = RAGPipeline()
    pipeline.add_documents(documents)

    questions = [
        "Что такое RAG и как он работает?",
        "Как устроены нейронные сети?",
        "Какие БД подходят для аналитики?",
    ]

    for question in questions:
        print(f"\n❓ Вопрос: \"{question}\"")
        print("-" * 50)

        # Показываем retrieval результаты
        results = pipeline.retrieve(question, top_k=3)
        print("  Retrieved documents:")
        for rank, (doc, score) in enumerate(results, 1):
            print(f"    {rank}. [{score:.4f}] {doc['text'][:60]}...")

        # Генерация
        answer = pipeline.generate(question, top_k=3)
        print(f"\n  💡 Ответ:\n{answer}")
    print()


def demo_4_evaluation():
    """Демо 4: Оценка качества retrieval."""
    print("=" * 70)
    print("ДЕМО 4: Оценка качества retrieval")
    print("=" * 70)

    # Создаём индекс с размеченными документами
    documents = [
        {"id": "ml_1", "text": "Нейронные сети — основа глубокого обучения. CNN для изображений, RNN для последовательностей.", "metadata": {"topic": "ml"}},
        {"id": "ml_2", "text": "Трансформеры используют self-attention механизм для обработки последовательностей.", "metadata": {"topic": "ml"}},
        {"id": "ml_3", "text": "RAG combines retrieval и generation для ответов на вопросы по базе знаний.", "metadata": {"topic": "ml"}},
        {"id": "db_1", "text": "PostgreSQL поддерживает JSON, полнотекстовый поиск и расширения.", "metadata": {"topic": "db"}},
        {"id": "db_2", "text": "Redis — in-memory хранилище для кэширования и очередей сообщений.", "metadata": {"topic": "db"}},
        {"id": "dev_1", "text": "Docker контейнеризируют приложения, Kubernetes оркестрирует их.", "metadata": {"topic": "devops"}},
        {"id": "dev_2", "text": "CI/CD pipeline автоматизирует тестирование и деплой кода.", "metadata": {"topic": "devops"}},
    ]

    index = DocumentIndex()
    index.add_documents(documents)

    # Тестовые запросы с ожидаемыми релевантными документами
    test_cases = [
        {
            "query": "нейронные сети и deep learning",
            "relevant": ["ml_1", "ml_2"],
        },
        {
            "query": "базы данных для хранения",
            "relevant": ["db_1", "db_2"],
        },
        {
            "query": "контейнеры и деплой",
            "relevant": ["dev_1", "dev_2"],
        },
        {
            "query": "поиск и генерация ответов",
            "relevant": ["ml_3", "db_1"],  # pg_trgm для поиска + RAG
        },
    ]

    print("\nОценка retrieval по 4 тестовым запросам:\n")

    all_precisions = []
    all_recalls = []
    all_mrrs = []

    for tc in test_cases:
        query = tc["query"]
        relevant = tc["relevant"]

        results = index.search(query, top_k=5)
        retrieved_ids = [doc['id'] for doc, _ in results]

        p = precision_at_k(retrieved_ids, relevant, k=3)
        r = recall_at_k(retrieved_ids, relevant, k=5)
        m = mrr(retrieved_ids, relevant)

        all_precisions.append(p)
        all_recalls.append(r)
        all_mrrs.append(m)

        print(f"  Запрос: \"{query}\"")
        print(f"    Retrieved: {retrieved_ids[:5]}")
        print(f"    Relevant:  {relevant}")
        print(f"    P@3={p:.3f}  R@5={r:.3f}  MRR={m:.3f}")
        print()

    print("Итоговые метрики (macro-averaged):")
    print(f"  Precision@3: {sum(all_precisions)/len(all_precisions):.3f}")
    print(f"  Recall@5:    {sum(all_recalls)/len(all_recalls):.3f}")
    print(f"  MRR:         {sum(all_mrrs)/len(all_mrrs):.3f}")
    print()


# ============================================================
# Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RAG (Retrieval-Augmented Generation) — основы с нуля")
    print("  Без внешних зависимостей: чистый Python")
    print("=" * 70 + "\n")

    demo_1_indexing()
    demo_2_semantic_search()
    demo_3_generation()
    demo_4_evaluation()

    print("=" * 70)
    print("  Все демо завершены!")
    print("=" * 70)
