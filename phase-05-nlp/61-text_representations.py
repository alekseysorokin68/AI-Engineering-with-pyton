"""
Основы представления текста: Bag of Words, TF-IDF, N-grams.

Демо:
  1. Bag of Words — подсчёт слов
  2. TF-IDF — важность слов
  3. N-grams (биограммы, триграммы)
  4. Сравнение методов
"""

import math
import random
import re
from collections import Counter

random.seed(42)


# ── Утилиты ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё]+", text.lower())


def simple_stem(word: str) -> str:
    """Грубый стеммер: отсекает типичные окончания."""
    for suffix in ("tion", "sion", "ing", "ment", "ness", "ous", "able",
                   "edly", "ing", "ed", "ly", "er", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def preprocess(text: str) -> list[str]:
    return [simple_stem(w) for w in tokenize(text)]


# ── Демо 1: Bag of Words ────────────────────────────────────────────────────

def demo_bow():
    print("=" * 60)
    print("Демо 1: Bag of Words")
    print("=" * 60)

    corpus = [
        "Машинное обучение — это раздел искусственного интеллекта",
        "Искусственный интеллект изменяет мир технологий",
        "Обучение на данных — ключ к интеллектуальным системам",
    ]

    processed = [preprocess(doc) for doc in corpus]

    # Собираем словарь
    vocab = sorted({w for doc in processed for w in doc})
    word2idx = {w: i for i, w in enumerate(vocab)}

    print(f"\nКорпус ({len(corpus)} документа):")
    for i, doc in enumerate(corpus):
        print(f"  [{i}] {doc}")

    print(f"\nСловарь ({len(vocab)} уникальных токенов):")
    print(f"  {vocab}")

    # Строим матрицу BoW
    print("\nМатрица BoW (строка = документ, столбец = слово из словаря):")
    print(f"  {'Слово':<20}", end="")
    for doc_id in range(len(corpus)):
        print(f"  Doc{doc_id}", end="")
    print()

    bow_matrix = []
    for doc_id, tokens in enumerate(processed):
        counts = Counter(tokens)
        row = [counts.get(w, 0) for w in vocab]
        bow_matrix.append(row)
        print(f"  {vocab[0]:<20}", end="")
        # Выводим только первые 5 слов, чтобы таблица не была слишком широкой
        sample = min(5, len(vocab))
        for j in range(sample):
            print(f"  {counts.get(vocab[j], 0):>4}", end="")
        if len(vocab) > sample:
            print("  ...", end="")
        print()

    # Показываем полную матрицу для наглядности
    print("\nПолная матрица BoW:")
    header = "         " + "".join(f"{w:>6}" for w in vocab)
    print(header)
    for doc_id, tokens in enumerate(processed):
        counts = Counter(tokens)
        row_str = f"Doc{doc_id:<3}  " + "".join(
            f"{counts.get(w, 0):>6}" for w in vocab
        )
        print(row_str)

    # Пример: поисковый запрос
    query = "интеллект"
    query_tokens = preprocess(query)
    print(f"\nЗапрос: '{query}' → токены: {query_tokens}")

    # Косинусное сходство (простая версия)
    for doc_id, tokens in enumerate(processed):
        doc_counts = Counter(tokens)
        sim = sum(doc_counts.get(q, 0) for q in query_tokens)
        print(f"  Doc{doc_id}: совпадение = {sim}")


# ── Демо 2: TF-IDF ──────────────────────────────────────────────────────────

def demo_tfidf():
    print("\n" + "=" * 60)
    print("Демо 2: TF-IDF")
    print("=" * 60)

    corpus = [
        "кот сидит на коврике",
        "собака играет во дворе",
        "кот и собака дружат",
        "на коврике лежит кот",
    ]

    processed = [preprocess(doc) for doc in corpus]
    all_words = sorted({w for doc in processed for w in doc})

    print(f"\nКорпус ({len(corpus)} документа):")
    for i, doc in enumerate(corpus):
        print(f"  [{i}] {doc}")

    # TF: term frequency (нормализованная)
    def compute_tf(tokens):
        counts = Counter(tokens)
        total = len(tokens)
        return {w: c / total for w, c in counts.items()}

    # IDF: inverse document frequency (сглаженная)
    n_docs = len(corpus)
    doc_freq = Counter()
    for tokens in processed:
        doc_freq.update(set(tokens))

    def compute_idf(word):
        return math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1

    # TF-IDF
    print("\n--- TF (Term Frequency) ---")
    for doc_id, tokens in enumerate(processed):
        tf = compute_tf(tokens)
        top = sorted(tf.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{w}={v:.3f}" for w, v in top)
        print(f"  Doc{doc_id}: {top_str}")

    print("\n--- IDF (Inverse Document Frequency) ---")
    for w in all_words:
        idf = compute_idf(w)
        print(f"  {w:<20} IDF = {idf:.4f}  (в {doc_freq[w]}/{n_docs} доках)")

    print("\n--- TF-IDF Матрица (топ-3 слова по TF-IDF для каждого дока) ---")
    for doc_id, tokens in enumerate(processed):
        tf = compute_tf(tokens)
        tfidf = {w: tf_val * compute_idf(w) for w, tf_val in tf.items()}
        top = sorted(tfidf.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{w}={v:.4f}" for w, v in top)
        print(f"  Doc{doc_id}: {top_str}")

    # Косинусное сходство по TF-IDF
    print("\n--- Косинусное сходство (TF-IDF) ---")

    def doc_tfidf_vec(tokens):
        tf = compute_tf(tokens)
        return {w: tf_val * compute_idf(w) for w, tf_val in tf.items()}

    def cosine_sim(v1, v2):
        keys = set(v1) & set(v2)
        dot = sum(v1[k] * v2[k] for k in keys)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    vecs = [doc_tfidf_vec(tokens) for tokens in processed]
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            sim = cosine_sim(vecs[i], vecs[j])
            print(f"  Doc{i} ↔ Doc{j}: cosine = {sim:.4f}")


# ── Демо 3: N-grams ─────────────────────────────────────────────────────────

def demo_ngrams():
    print("\n" + "=" * 60)
    print("Демо 3: N-grams (биограммы, триграммы)")
    print("=" * 60)

    text = "искусственный интеллект меняет мир технологий каждый день"
    tokens = text.split()

    print(f"\nИсходный текст: '{text}'")
    print(f"Токены: {tokens}")

    def get_ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    for n, label in [(1, "Униграммы"), (2, "Биограммы"), (3, "Триграммы")]:
        ngrams = get_ngrams(tokens, n)
        counts = Counter(ngrams)
        print(f"\n--- {label} (n={n}) ---")
        print(f"  Всего: {len(ngrams)} | Уникальных: {len(counts)}")
        print(f"  Топ-5:")
        for ng, cnt in counts.most_common(5):
            print(f"    {ng}: {cnt}")

    # Демонстрация: n-grams для генерации текста (простая модель)
    print("\n--- Простая языковая модель на биограммах ---")
    text2 = (
        "машинное обучение это обучение на данных "
        "машинное обучение требует данных "
        "данные ключ к обучению"
    )
    tokens2 = text2.split()
    bigrams = get_ngrams(tokens2, 2)
    trigrams = get_ngrams(tokens2, 3)

    # Строим словарь: первое слово → следующее слово
    bigram_model = {}
    for a, b in bigrams:
        bigram_model.setdefault(a, []).append(b)

    print("  Модель биограмм (слово → возможные продолжения):")
    for word in sorted(bigram_model):
        continuations = bigram_model[word]
        print(f"    '{word}' → {continuations}")

    # Генерация случайного предложения
    random.seed(42)
    start = random.choice(list(bigram_model.keys()))
    generated = [start]
    for _ in range(5):
        last = generated[-1]
        if last in bigram_model:
            next_word = random.choice(bigram_model[last])
            generated.append(next_word)
        else:
            break
    print(f"\n  Сгенерированное предложение: {' '.join(generated)}")

    # Характеристики N-grams как признаков
    print("\n--- Характеристики N-grams как признаков ---")
    vocab = set(tokens)
    vocab_bi = set(f"{a} {b}" for a, b in get_ngrams(tokens, 2))
    vocab_tri = set(f"{a} {b} {c}" for a, b, c in get_ngrams(tokens, 3))
    print(f"  Униграммы: {len(vocab)} признаков")
    print(f"  Биограммы: {len(vocab_bi)} признаков")
    print(f"  Триграммы: {len(vocab_tri)} признаков")
    print("  Проблема: с ростом n размер словаря экспоненциально растёт!")


# ── Демо 4: Сравнение методов ───────────────────────────────────────────────

def demo_comparison():
    print("\n" + "=" * 60)
    print("Демо 4: Сравнение методов")
    print("=" * 60)

    corpus = [
        "кошка сидит на подоконнике",
        "собака бегает по парку",
        "на подоконнике лежит кошка",
        "в парке играют дети",
    ]

    processed = [preprocess(doc) for doc in corpus]
    all_words = sorted({w for doc in processed for w in doc})
    n_docs = len(corpus)

    # DF
    doc_freq = Counter()
    for tokens in processed:
        doc_freq.update(set(tokens))

    def compute_idf(word):
        return math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1

    print(f"\nКорпус:")
    for i, doc in enumerate(corpus):
        print(f"  [{i}] {doc}")

    # Метод 1: BoW
    print("\n--- Метод 1: Bag of Words ---")
    print("  Преимущества:")
    print("    + Простая реализация")
    print("    + Быстрый подсчёт")
    print("    + Хорошо для классификации")
    print("  Недостатки:")
    print("    - Не учитывает порядок слов")
    print("    - Не учитывает важность слов")
    print("    - Высокая размерность для больших корпусов")

    # Метод 2: TF-IDF
    print("\n--- Метод 2: TF-IDF ---")
    print("  Преимущества:")
    print("    + Учитывает важность слов в документе")
    print("    + Штрафует частые общие слова")
    print("    + Стандарт в информационном поиске")
    print("  Недостатки:")
    print("    - Не учитывает семантику")
    print("    - Все ещё не учитывает порядок слов")
    print("    - Высокая размерность")

    # Метод 3: N-grams
    print("\n--- Метод 3: N-grams ---")
    print("  Преимущества:")
    print("    + Учитывает локальный контекст")
    print("    + Может уловить порядок слов")
    print("    + Хорош для детекции языка, спама")
    print("  Недостатки:")
    print("    - Экспоненциальный рост размерности")
    print("    - Разреженные матрицы")
    print("    - Сильное переобучение при больших n")

    # Сравнительная таблица
    print("\n--- Сравнительная таблица ---")
    print(f"  {'Критерий':<25} {'BoW':<12} {'TF-IDF':<12} {'N-grams':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Простота реализации':<25} {'★★★★★':<12} {'★★★★☆':<12} {'★★★☆☆':<12}")
    print(f"  {'Учёт важности слов':<25} {'★☆☆☆☆':<12} {'★★★★★':<12} {'★☆☆☆☆':<12}")
    print(f"  {'Учёт контекста':<25} {'★☆☆☆☆':<12} {'★☆☆☆☆':<12} {'★★★☆☆':<12}")
    print(f"  {'Размерность':<25} {'Средняя':<12} {'Средняя':<12} {'Большая':<12}")
    print(f"  {'Применение':<25} {'Классиф.':<12} {'Поиск':<12} {'Язык/спам':<12}")

    # Пример: один и тот же текст, разные представления
    print("\n--- Демонстрация на одном тексте ---")
    example = "искусственный интеллект учится на данных"
    tokens = preprocess(example)
    print(f"  Текст: '{example}'")
    print(f"  Токены: {tokens}")

    # BoW
    bow = Counter(tokens)
    print(f"  BoW:   {dict(bow)}")

    # TF-IDF (однодокументный корпус)
    df_single = Counter(set(tokens))
    tfidf = {w: (c / len(tokens)) * (math.log(2 / (1 + 1)) + 1)
             for w, c in bow.items()}
    print(f"  TF-IDF: {dict(sorted(tfidf.items(), key=lambda x: -x[1]))}")

    # N-grams
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    print(f"  Биограммы: {bigrams}")


# ── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_bow()
    demo_tfidf()
    demo_ngrams()
    demo_comparison()
    print("\n" + "=" * 60)
    print("Готово! Все методы представления текста продемонстрированы.")
    print("=" * 60)
