"""
Основы Word Embeddings: Skip-gram, обучение эмбеддингов, косинусное сходство, аналогии.

Демо:
  1. Skip-gram — генерация пар слов
  2. Обучение эмбеддингов
  3. Косинусное сходство между словами
  4. Аналогии (king - man + woman = queen)
"""

import math
import random
import re
from collections import Counter, defaultdict

random.seed(42)


# ── Утилиты ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё]+", text.lower())


# ── Skip-gram: генерация пар слов ────────────────────────────────────────────

def generate_skipgram_pairs(tokens: list[str], window_size: int = 2) -> list[tuple[str, str]]:
    """Генерирует пары (target, context) для Skip-gram модели."""
    pairs = []
    for i, target in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((target, tokens[j]))
    return pairs


# ── Word Embedding Model ─────────────────────────────────────────────────────

class SimpleWord2Vec:
    """Упрощённая Skip-gram модель с отрицательным семплированием."""

    def __init__(self, vocab_size: int, embedding_dim: int = 10, learning_rate: float = 0.05):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.embeddings: list[list[float]] = []
        self.output_weights: list[list[float]] = []

    def build_vocab(self, tokens: list[str]):
        word_counts = Counter(tokens)
        vocab = [w for w, _ in word_counts.most_common(self.vocab_size)]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)
        self.embeddings = [[random.gauss(0, 0.1) for _ in range(self.embedding_dim)]
                           for _ in range(self.vocab_size)]
        self.output_weights = [[random.gauss(0, 0.1) for _ in range(self.embedding_dim)]
                                for _ in range(self.vocab_size)]

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))

    def train_pair(self, target_idx: int, context_idx: int, negative_indices: list[int]):
        """Обучение на одной паре (target, context) с отрицательным семплированием."""
        target_vec = self.embeddings[target_idx]

        # Positive sample
        pos_dot = sum(target_vec[k] * self.output_weights[context_idx][k]
                      for k in range(self.embedding_dim))
        pos_sig = self._sigmoid(pos_dot)

        # Градиент для positive
        error = self.lr * (1.0 - pos_sig)
        for k in range(self.embedding_dim):
            grad = error * self.output_weights[context_idx][k]
            self.embeddings[target_idx][k] += grad
            self.output_weights[context_idx][k] += error * target_vec[k]

        # Negative samples
        for neg_idx in negative_indices:
            neg_dot = sum(target_vec[k] * self.output_weights[neg_idx][k]
                          for k in range(self.embedding_dim))
            neg_sig = self._sigmoid(neg_dot)
            error = self.lr * (0.0 - neg_sig)
            for k in range(self.embedding_dim):
                grad = error * self.output_weights[neg_idx][k]
                self.embeddings[target_idx][k] += grad
                self.output_weights[neg_idx][k] += error * target_vec[k]

    def train(self, tokens: list[str], epochs: int = 5, window_size: int = 2, neg_samples: int = 3):
        self.build_vocab(tokens)
        pairs = generate_skipgram_pairs(tokens, window_size)
        word_indices = list(range(self.vocab_size))

        for epoch in range(epochs):
            random.shuffle(pairs)
            for target, context in pairs:
                if target not in self.word2idx or context not in self.word2idx:
                    continue
                t_idx = self.word2idx[target]
                c_idx = self.word2idx[context]
                negs = random.sample([i for i in word_indices if i != c_idx],
                                     min(neg_samples, self.vocab_size - 1))
                self.train_pair(t_idx, c_idx, negs)

    def get_embedding(self, word: str) -> list[float] | None:
        if word in self.word2idx:
            return self.embeddings[self.word2idx[word]]
        return None

    def get_vector(self, word: str) -> list[float] | None:
        """Возвращает вектор слова (из основных эмбеддингов)."""
        return self.get_embedding(word)


# ── Косинусное сходство ──────────────────────────────────────────────────────

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ── Аналогии ─────────────────────────────────────────────────────────────────

def most_similar(word: str, model: SimpleWord2Vec, top_n: int = 5) -> list[tuple[str, float]]:
    vec = model.get_embedding(word)
    if vec is None:
        return []
    sims = []
    for other_word, idx in model.word2idx.items():
        if other_word == word:
            continue
        other_vec = model.embeddings[idx]
        sim = cosine_similarity(vec, other_vec)
        sims.append((other_word, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


def analogy(a: str, b: str, c: str, model: SimpleWord2Vec, top_n: int = 3) -> list[tuple[str, float]]:
    """Решает аналогию: a - b + c = ? (например king - man + woman = queen)"""
    va = model.get_embedding(a)
    vb = model.get_embedding(b)
    vc = model.get_embedding(c)
    if va is None or vb is None or vc is None:
        return []
    target = [va[k] - vb[k] + vc[k] for k in range(model.embedding_dim)]
    sims = []
    exclude = {a, b, c}
    for word, idx in model.word2idx.items():
        if word in exclude:
            continue
        sim = cosine_similarity(target, model.embeddings[idx])
        sims.append((word, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  ДЕМОНСТРАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def demo_skipgram():
    print("=" * 65)
    print("Демо 1: Skip-gram — генерация пар слов")
    print("=" * 65)

    text = "the cat sat on the mat the dog sat on the rug"
    tokens = tokenize(text)
    print(f"\nТекст:   {text}")
    print(f"Токены:  {tokens}")

    pairs = generate_skipgram_pairs(tokens, window_size=2)
    print(f"\nПар (window=2): {len(pairs)}")
    print("\nПримеры пар (target → context):")
    for i, (t, c) in enumerate(pairs[:10]):
        print(f"  {i + 1:2d}. {t:>6s} → {c}")

    print(f"\nУникальных слов: {len(set(tokens))}")
    print(f"Всего пар:        {len(pairs)}")


def demo_training():
    print("\n" + "=" * 65)
    print("Демо 2: Обучение эмбеддингов")
    print("=" * 65)

    corpus = """
    the king ruled the kingdom with wisdom and justice
    the queen ruled the kingdom with grace and kindness
    the man walked through the city
    the woman walked through the city
    the king is a man and the queen is a woman
    a boy played in the garden with a ball
    a girl played in the garden with a doll
    the cat sat on the mat
    the dog lay on the rug
    the boy and the girl are friends
    the king and the queen are royalty
    a man and a woman are married
    """

    tokens = tokenize(corpus)
    print(f"\nРазмер корпуса: {len(tokens)} слов, {len(set(tokens))} уникальных")
    print(f"Слова: {sorted(set(tokens))}")

    model = SimpleWord2Vec(vocab_size=50, embedding_dim=10, learning_rate=0.05)
    model.train(tokens, epochs=5, window_size=2, neg_samples=3)

    print(f"\nРазмер словаря: {model.vocab_size}")
    print(f"Размер эмбеддинга: {model.embedding_dim}")
    print("\nПримеры эмбеддингов (первые 5 компонент):")
    for word in ["king", "queen", "man", "woman", "boy", "girl"]:
        vec = model.get_embedding(word)
        if vec:
            short = [f"{v:+.3f}" for v in vec[:5]]
            print(f"  {word:>6s}: [{', '.join(short)}, ...]")

    return model


def demo_cosine(model: SimpleWord2Vec):
    print("\n" + "=" * 65)
    print("Демо 3: Косинусное сходство")
    print("=" * 65)

    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("king", "man"),
        ("cat", "dog"),
        ("king", "cat"),
    ]

    print("\nПопарное сходство:\n")
    for w1, w2 in pairs:
        v1 = model.get_embedding(w1)
        v2 = model.get_embedding(w2)
        if v1 and v2:
            sim = cosine_similarity(v1, v2)
            bar = "█" * int((sim + 1) / 2 * 30)
            print(f"  {w1:>6s} ↔ {w2:<6s}  sim={sim:+.4f}  {bar}")

    print("\nБлижайшие слова для 'king':")
    for word, sim in most_similar("king", model, top_n=5):
        print(f"  {word:>8s}: {sim:+.4f}")

    print("\nБлижайшие слова для 'woman':")
    for word, sim in most_similar("woman", model, top_n=5):
        print(f"  {word:>8s}: {sim:+.4f}")


def demo_analogies(model: SimpleWord2Vec):
    print("\n" + "=" * 65)
    print("Демо 4: Аналогии (king - man + woman = queen)")
    print("=" * 65)

    analogies = [
        ("king", "man", "woman", "king − man + woman = ?"),
        ("queen", "woman", "man", "queen − woman + man = ?"),
        ("boy", "man", "woman", "boy − man + woman = ?"),
    ]

    for a, b, c, desc in analogies:
        print(f"\n  {desc}")
        results = analogy(a, b, c, model, top_n=3)
        if results:
            for word, sim in results:
                marker = " ← ожидается" if (
                    (a == "king" and word == "queen") or
                    (a == "queen" and word == "king") or
                    (a == "boy" and word == "girl")
                ) else ""
                print(f"    {word:>8s}: {sim:+.4f}{marker}")
        else:
            print("    (слова не найдены в словаре)")

    print("\n--- Поиск ближайших по векторной арифметике ---")
    king = model.get_embedding("king")
    man = model.get_embedding("man")
    woman = model.get_embedding("woman")
    if king and man and woman:
        diff = [king[i] - man[i] + woman[i] for i in range(model.embedding_dim)]
        sims = []
        for word, idx in model.word2idx.items():
            if word in {"king", "man", "woman"}:
                continue
            sim = cosine_similarity(diff, model.embeddings[idx])
            sims.append((word, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  king − man + woman → топ-5:")
        for word, sim in sims[:5]:
            print(f"    {word:>8s}: {sim:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("NLP — Основы Word Embeddings (Skip-gram)\n")

    demo_skipgram()
    model = demo_training()
    demo_cosine(model)
    demo_analogies(model)

    print("\n" + "=" * 65)
    print("Итого:")
    print("  - Skip-gram генерирует контекстные пары слов")
    print("  - Эмбеддинги обучаются через негативное семплирование")
    print("  - Косинусное сходство измеряет близость слов")
    print("  - Векторная арифметика решает аналогии")
    print("  - По мере обучения семантически близкие слова")
    print("    оказываются рядом в векторном пространстве")
    print("=" * 65)
