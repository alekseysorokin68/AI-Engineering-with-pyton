"""
Pre-training для LLM: Next Token Prediction & Masked Language Modeling
=====================================================================

Самодостаточный скрипт без внешних зависимостей (numpy/torch/transformers).
Покрывает:
  1. Next Token Prediction (GPT-style)
  2. Masked Language Modeling (BERT-style)
  3. Training loop для pre-training
  4. Сравнение подходов

Запуск: python 102-pretraining.py
"""

import random
import math
import copy

random.seed(42)


# =============================================================================
# 1. Утилиты
# =============================================================================

def softmax(x):
    max_x = max(x)
    e = [math.exp(v - max_x) for v in x]
    s = sum(e)
    return [v / s for v in e]


def cross_entropy(probs, target_idx):
    p = max(probs[target_idx], 1e-12)
    return -math.log(p)


def one_hot(idx, size):
    v = [0.0] * size
    v[idx] = 1.0
    return v


def argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])


def top_k(probs, k=3):
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:k]
    return [(i, p) for i, p in indexed]


# =============================================================================
# 2. Токенизация (простой word-level)
# =============================================================================

def build_vocab(texts):
    tokens = set()
    for t in texts:
        tokens.update(t.split())
    vocab = sorted(tokens)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


# =============================================================================
# 3. Mini-Embedding + Linear (без torch)
# =============================================================================

class MiniEmbedding:
    """Слой эмбеддингов: vocabulary_size -> embed_dim."""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Xavier-like инициализация
        scale = math.sqrt(2.0 / (vocab_size + embed_dim))
        self.W = [[random.gauss(0, scale) for _ in range(embed_dim)]
                   for _ in range(vocab_size)]

    def forward(self, idx):
        return self.W[idx][:]

    def backward(self, idx, grad_out, lr=0.01):
        for d in range(self.embed_dim):
            self.W[idx][d] -= lr * grad_out[d]


class MiniLinear:
    """Линейный слой: in_dim -> out_dim."""

    def __init__(self, in_dim, out_dim):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.W = [[random.gauss(0, scale) for _ in range(out_dim)]
                   for _ in range(in_dim)]
        self.b = [0.0] * out_dim

    def forward(self, x):
        out = self.b[:]
        for i in range(len(x)):
            for j in range(len(self.b)):
                out[j] += x[i] * self.W[i][j]
        return out

    def backward(self, x, grad_out, lr=0.01):
        grad_in = [0.0] * len(x)
        for i in range(len(x)):
            for j in range(len(grad_out)):
                self.W[i][j] -= lr * x[i] * grad_out[j]
                grad_in[i] += self.W[i][j] * grad_out[j]
        for j in range(len(grad_out)):
            self.b[j] -= lr * grad_out[j]
        return grad_in


# =============================================================================
# 4. Next Token Prediction (GPT-style)
# =============================================================================

class NextTokenPredictor:
    """
    Минимальная модель для предсказания следующего токена.
    Architecture: Embedding -> Linear -> Softmax
    """

    def __init__(self, vocab_size, embed_dim):
        self.embedding = MiniEmbedding(vocab_size, embed_dim)
        self.projection = MiniLinear(embed_dim, vocab_size)

    def forward(self, idx):
        emb = self.embedding.forward(idx)
        logits = self.projection.forward(emb)
        probs = softmax(logits)
        return probs

    def train_step(self, input_idx, target_idx, lr=0.05):
        """Один шаг обучения: forward + backward (упрощённый)."""
        emb = self.embedding.forward(input_idx)
        logits = self.projection.forward(emb)
        probs = softmax(logits)

        loss = cross_entropy(probs, target_idx)

        # Gradient of cross-entropy w.r.t. logits
        grad_logits = [probs[j] - (1.0 if j == target_idx else 0.0)
                       for j in range(len(probs))]

        # Backward through projection
        grad_emb = self.projection.backward(emb, grad_logits, lr)

        # Backward through embedding
        self.embedding.backward(input_idx, grad_emb, lr)

        return loss, probs

    def generate(self, start_idx, n_tokens, idx2word):
        """Авторегрессивная генерация текста."""
        current_idx = start_idx
        generated = []
        for _ in range(n_tokens):
            probs = self.forward(current_idx)
            next_idx = argmax(probs)
            generated.append((idx2word[next_idx], probs[next_idx]))
            current_idx = next_idx
        return generated


# =============================================================================
# 5. Masked Language Model (BERT-style)
# =============================================================================

class MaskedLanguageModel:
    """
    Минимальная модель для Masked LM.
    Architecture: Embedding -> Linear -> Softmax (для предсказания замаскированного токена)
    """

    def __init__(self, vocab_size, embed_dim, mask_token_idx):
        self.vocab_size = vocab_size
        self.mask_token_idx = mask_token_idx
        self.embedding = MiniEmbedding(vocab_size, embed_dim)
        self.projection = MiniLinear(embed_dim, vocab_size)

    def forward(self, seq_indices, mask_positions):
        """
        Принимает последовательность индексов и позиции масок.
        Для простоты: усредняем эмбеддинги всех токенов (BOW-like).
        """
        n = len(seq_indices)
        avg_emb = [0.0] * self.embedding.embed_dim
        for idx in seq_indices:
            emb = self.embedding.forward(idx)
            for d in range(len(avg_emb)):
                avg_emb[d] += emb[d] / n

        logits = self.projection.forward(avg_emb)
        probs = softmax(logits)
        return probs

    def train_step(self, seq_indices, mask_positions, target_indices, lr=0.05):
        """Обучение на замаскированной последовательности."""
        n = len(seq_indices)
        avg_emb = [0.0] * self.embedding.embed_dim
        for idx in seq_indices:
            emb = self.embedding.forward(idx)
            for d in range(len(avg_emb)):
                avg_emb[d] += emb[d] / n

        logits = self.projection.forward(avg_emb)
        probs = softmax(logits)

        # Суммарный loss по всем замаскированным позициям
        total_loss = 0.0
        for target_idx in target_indices:
            total_loss += cross_entropy(probs, target_idx)
        loss = total_loss / len(target_indices) if target_indices else 0.0

        # Backprop
        grad_logits = [0.0] * self.vocab_size
        for target_idx in target_indices:
            for j in range(self.vocab_size):
                grad_logits[j] += (probs[j] - (1.0 if j == target_idx else 0.0))
        for j in range(self.vocab_size):
            grad_logits[j] /= len(target_indices)

        grad_emb = self.projection.backward(avg_emb, grad_logits, lr)

        # Distribute gradient back (simplified)
        for idx in seq_indices:
            scaled_grad = [g / n for g in grad_emb]
            self.embedding.backward(idx, scaled_grad, lr)

        return loss, probs


def create_mlm_batch(sequence, mask_prob, mask_token_idx, vocab_size):
    """
    Создаёт замаскированную версию последовательности.
    С вероятностью mask_prob заменяем токен на [MASK].
    """
    masked_seq = sequence[:]
    mask_positions = []
    target_indices = []
    for i in range(len(sequence)):
        if random.random() < mask_prob:
            mask_positions.append(i)
            target_indices.append(sequence[i])
            masked_seq[i] = mask_token_idx
    return masked_seq, mask_positions, target_indices


# =============================================================================
# 6. Training Loop
# =============================================================================

def train_next_token(model, data, word2idx, epochs=30, lr=0.05, verbose=True):
    """Training loop для Next Token Prediction."""
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(data)
        for input_idx, target_idx in data:
            loss, _ = model.train_step(input_idx, target_idx, lr)
            epoch_loss += loss
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
    return losses


def train_masked_lm(model, sequences, mask_prob, mask_token_idx,
                     epochs=30, lr=0.05, verbose=True):
    """Training loop для Masked Language Model."""
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(sequences)
        for seq in sequences:
            if len(seq) < 2:
                continue
            masked_seq, mask_pos, targets = create_mlm_batch(
                seq, mask_prob, mask_token_idx, model.vocab_size)
            if not targets:
                continue
            loss, _ = model.train_step(masked_seq, mask_pos, targets, lr)
            epoch_loss += loss
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
    return losses


# =============================================================================
# 7. Демонстрации
# =============================================================================

def demo_next_token_prediction():
    """Демо 1: Next Token Prediction (GPT-style)"""
    print("=" * 70)
    print("ДЕМО 1: NEXT TOKEN PREDICTION (GPT-style)")
    print("=" * 70)

    # Корпус для обучения
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat chased the dog",
        "the dog sat on the mat",
        "the cat ran in the park",
    ]

    word2idx, idx2word = build_vocab(texts)
    vocab_size = len(word2idx)
    embed_dim = 8

    print(f"\n  Vocabulary ({vocab_size} токенов): {list(word2idx.keys())}")
    print(f"  Embedding dim: {embed_dim}")

    # Создаём пары (input, target) для next token prediction
    train_data = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            train_data.append((word2idx[tokens[i]], word2idx[tokens[i + 1]]))

    print(f"  Training pairs: {len(train_data)}")

    # Обучение
    print("\n  --- Обучение ---")
    model = NextTokenPredictor(vocab_size, embed_dim)
    losses = train_next_token(model, train_data, word2idx, epochs=50, lr=0.05)

    # Демонстрация предсказаний
    print("\n  --- Предсказания после обучения ---")
    test_words = ["the", "cat", "dog"]
    for word in test_words:
        if word in word2idx:
            idx = word2idx[word]
            probs = model.forward(idx)
            top = top_k(probs, 3)
            top_words = [(idx2word[i], f"{p:.3f}") for i, p in top]
            print(f"    '{word}' -> top-3: {top_words}")

    # Генерация текста
    print("\n  --- Генерация текста ---")
    start_word = "the"
    if start_word in word2idx:
        gen = model.generate(word2idx[start_word], 5, idx2word)
        sequence = [start_word] + [w for w, _ in gen]
        print(f"    Старт: '{start_word}'")
        print(f"    Результат: '{' '.join(sequence)}'")
        print(f"    Пословные вероятности:")
        for w, p in gen:
            print(f"      -> '{w}' (p={p:.3f})")

    print(f"\n  Финальный loss: {losses[-1]:.4f}")
    print(f"  Loss уменьшился: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return model, word2idx, idx2word


def demo_masked_lm():
    """Демо 2: Masked Language Modeling (BERT-style)"""
    print("\n" + "=" * 70)
    print("ДЕМО 2: MASKED LANGUAGE MODELING (BERT-style)")
    print("=" * 70)

    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat chased the dog",
        "the dog sat on the mat",
        "the cat ran in the park",
        "a dog is sleeping on the mat",
        "the cat is playing in the park",
    ]

    word2idx, idx2word = build_vocab(texts)
    vocab_size = len(word2idx)
    embed_dim = 8
    mask_token_idx = word2idx.get("[MASK]", vocab_size - 1)

    print(f"\n  Vocabulary ({vocab_size} токенов): {list(word2idx.keys())}")
    print(f"  Mask token: [MASK] (idx={mask_token_idx})")
    print(f"  Mask probability: 0.3")

    # Конвертируем тексты в индексы
    sequences = [[word2idx[w] for w in text.split()] for text in texts]

    # Обучение
    print("\n  --- Обучение ---")
    model = MaskedLanguageModel(vocab_size, embed_dim, mask_token_idx)
    mask_prob = 0.3
    losses = train_masked_lm(model, sequences, mask_prob, mask_token_idx,
                             epochs=50, lr=0.05)

    # Демонстрация предсказаний
    print("\n  --- Предсказания замаскированных токенов ---")
    test_phrases = [
        ("the [MASK] sat on the mat", 1),
        ("the cat [MASK] on the mat", 2),
        ("the [MASK] ran in the park", 1),
    ]
    for phrase, mask_pos in test_phrases:
        tokens = phrase.split()
        seq_idx = []
        positions = []
        for i, t in enumerate(tokens):
            if t == "[MASK]":
                seq_idx.append(mask_token_idx)
                positions.append(i)
            else:
                seq_idx.append(word2idx[t])

        probs = model.forward(seq_idx, positions)
        top = top_k(probs, 3)
        top_words = [(idx2word[i], f"{p:.3f}") for i, p in top]
        print(f"    '{phrase}' -> top-3: {top_words}")

    print(f"\n  Финальный loss: {losses[-1]:.4f}")
    print(f"  Loss уменьшился: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return model


def demo_training_loop():
    """Демо 3: Подробный training loop с метриками"""
    print("\n" + "=" * 70)
    print("ДЕМО 3: TRAINING LOOP ДЛЯ PRE-TRAINING")
    print("=" * 70)

    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat chased the dog",
        "the dog sat on the mat",
        "the cat ran in the park",
        "a big dog is here",
        "the small cat sat there",
        "dogs and cats are friends",
    ]

    word2idx, idx2word = build_vocab(texts)
    vocab_size = len(word2idx)
    embed_dim = 8

    print(f"\n  Vocabulary: {vocab_size} токенов")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Корпус: {len(texts)} предложений")

    # Создаём training data
    train_data = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            train_data.append((word2idx[tokens[i]], word2idx[tokens[i + 1]]))

    print(f"  Training pairs: {len(train_data)}")

    # Training loop с деталями
    print("\n  --- Training Loop (Next Token Prediction) ---")
    model = NextTokenPredictor(vocab_size, embed_dim)
    lr = 0.05
    epochs = 50
    batch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(train_data)
        batch_count = 0

        for input_idx, target_idx in train_data:
            loss, probs = model.train_step(input_idx, target_idx, lr)
            epoch_loss += loss
            batch_count += 1

            # Прогноз accuracy
            predicted = argmax(probs)
            correct = 1 if predicted == target_idx else 0
            batch_losses.append((loss, correct))

        avg_loss = epoch_loss / len(train_data)

        # Считаем accuracy за эпоху
        epoch_correct = sum(c for _, c in batch_losses[-len(train_data):])
        accuracy = epoch_correct / len(train_data)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {accuracy:.1%}")

    # Learning rate scheduling (простой)
    print("\n  --- Learning Rate Scheduling (Step Decay) ---")
    model2 = NextTokenPredictor(vocab_size, embed_dim)
    lr = 0.1
    decay_rate = 0.95
    for epoch in range(50):
        if (epoch + 1) % 15 == 0:
            lr *= decay_rate
        epoch_loss = 0.0
        random.shuffle(train_data)
        for input_idx, target_idx in train_data:
            loss, _ = model2.train_step(input_idx, target_idx, lr)
            epoch_loss += loss
        avg_loss = epoch_loss / len(train_data)
        if (epoch + 1) % 15 == 0:
            print(f"    Epoch {epoch+1:3d} | LR: {lr:.4f} | Loss: {avg_loss:.4f}")

    print("\n  Training loop завершён.")
    return model


def demo_comparison():
    """Демо 4: Сравнение подходов Next Token Prediction vs Masked LM"""
    print("\n" + "=" * 70)
    print("ДЕМО 4: СРАВНЕНИЕ ПОДХОДОВ")
    print("=" * 70)

    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat chased the dog",
        "the dog sat on the mat",
        "the cat ran in the park",
    ]

    word2idx, idx2word = build_vocab(texts)
    vocab_size = len(word2idx)
    embed_dim = 8

    print(f"\n  Vocabulary: {vocab_size} токенов | Embedding dim: {embed_dim}")
    print(f"  Корпус: {len(texts)} предложений")

    # --- Next Token Prediction ---
    print("\n  [A] NEXT TOKEN PREDICTION (GPT-style)")
    print("  " + "-" * 45)
    print("  Каждый токен предсказывает СЛЕДУЮЩИЙ токен.")
    print("  Левостороннее внимание: context = всё что СЛЕВА.\n")

    train_data = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            train_data.append((word2idx[tokens[i]], word2idx[tokens[i + 1]]))

    model_gpt = NextTokenPredictor(vocab_size, embed_dim)
    gpt_losses = train_next_token(model_gpt, train_data, word2idx,
                                  epochs=50, lr=0.05, verbose=False)

    # Предсказание "the cat" -> ?
    if "the" in word2idx and "cat" in word2idx:
        the_idx = word2idx["the"]
        cat_idx = word2idx["cat"]

        # "the" -> top-3
        probs_the = model_gpt.forward(the_idx)
        top_the = top_k(probs_the, 3)
        print(f"  После 'the':     top-3 = {[(idx2word[i], f'{p:.2f}') for i, p in top_the]}")

        # "cat" -> top-3
        probs_cat = model_gpt.forward(cat_idx)
        top_cat = top_k(probs_cat, 3)
        print(f"  После 'cat':     top-3 = {[(idx2word[i], f'{p:.2f}') for i, p in top_cat]}")

    print(f"  Финальный loss: {gpt_losses[-1]:.4f}")

    # --- Masked LM ---
    print("\n  [B] MASKED LANGUAGE MODELING (BERT-style)")
    print("  " + "-" * 45)
    print("  Замаскированный токен предсказывает СЕБЯ по контексту.")
    print("  Двусторонний контекст: context = всё предложение.\n")

    mask_token_idx = vocab_size - 1  # Используем последний индекс как [MASK]
    sequences = [[word2idx[w] for w in text.split()] for text in texts]
    mask_prob = 0.3

    model_bert = MaskedLanguageModel(vocab_size, embed_dim, mask_token_idx)
    bert_losses = train_masked_lm(model_bert, sequences, mask_prob, mask_token_idx,
                                  epochs=50, lr=0.05, verbose=False)

    # Предсказание "the [MASK] on the mat" -> ?
    test_seq = ["the", "[MASK]", "on", "the", "mat"]
    seq_idx = [word2idx.get(w, mask_token_idx) for w in test_seq]
    mask_pos = [1]
    probs_bert = model_bert.forward(seq_idx, mask_pos)
    top_bert = top_k(probs_bert, 3)
    print(f"  'the [MASK] on the mat':     top-3 = {[(idx2word[i], f'{p:.2f}') for i, p in top_bert]}")

    test_seq2 = ["the", "cat", "[MASK]", "the", "dog"]
    seq_idx2 = [word2idx.get(w, mask_token_idx) for w in test_seq2]
    mask_pos2 = [2]
    probs_bert2 = model_bert.forward(seq_idx2, mask_pos2)
    top_bert2 = top_k(probs_bert2, 3)
    print(f"  'the cat [MASK] the dog':    top-3 = {[(idx2word[i], f'{p:.2f}') for i, p in top_bert2]}")

    print(f"  Финальный loss: {bert_losses[-1]:.4f}")

    # --- Сводная таблица ---
    print("\n  " + "=" * 60)
    print("  СВОДНАЯ ТАБЛИЦА")
    print("  " + "=" * 60)
    print(f"  {'Параметр':<30} {'GPT (NTP)':<15} {'BERT (MLM)':<15}")
    print("  " + "-" * 60)
    print(f"  {'Архитектура':<30} {'Эмб+Линейн':<15} {'Эмб+Линейн':<15}")
    print(f"  {'Направление контекста':<30} {'Левосторонн':<15} {'Двусторонн':<15}")
    print(f"  {'Тип предсказания':<30} {'Следующий':<15} {'Замаскиров.':<15}")
    print(f"  {'Финальный loss':<30} {gpt_losses[-1]:<15.4f} {bert_losses[-1]:<15.4f}")
    print(f"  {'Training data efficiency':<30} {'1 pair/sent':<15} {'Batches/sent':<15}")
    print("  " + "=" * 60)

    print("\n  Ключевые различия:")
    print("  1. GPT предсказывает ТОЛЬКО следующий токен (авторегрессивно)")
    print("  2. BERT предсказывает ЗАМАСКИРОВАННЫЙ токен по обоим контекстам")
    print("  3. GPT лучше для генерации текста")
    print("  4. BERT лучше для понимания контекста (NLU задачи)")
    print("  5. GPT: loss = cross-entropy на каждом токене")
    print("  6. BERT: loss = cross-entropy только на замаскированных токенах")

    return model_gpt, model_bert


# =============================================================================
# 8. Основная программа
# =============================================================================

def main():
    print()
    print("Pre-training для LLM: Next Token Prediction & Masked LM")
    print("Python implementation (no external dependencies)")
    print("random.seed(42) установлен")

    # Демо 1
    model_gpt, w2i, i2w = demo_next_token_prediction()

    # Демо 2
    model_bert = demo_masked_lm()

    # Демо 3
    model_loop = demo_training_loop()

    # Демо 4
    model_gpt2, model_bert2 = demo_comparison()

    print("\n" + "=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 70)
    print("\nКлючевые концепции:")
    print("  1. Next Token Prediction: предсказание следующего токена (GPT)")
    print("  2. Masked LM: предсказание замаскированного токена (BERT)")
    print("  3. Training loop: forward -> loss -> backward -> update")
    print("  4. GPT для генерации, BERT для понимания")


if __name__ == "__main__":
    main()
