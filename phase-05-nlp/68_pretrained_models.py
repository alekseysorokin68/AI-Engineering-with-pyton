"""
68. Предобученные языковые модели: BERT и GPT
=============================================

Файл демонстрирует ключевые концепции предобученных моделей:
- Архитектуры BERT (encoder-only) и GPT (decoder-only)
- Fine-tuning для классификации текста
- Zero-shot классификация
- Сравнение подходов

Все демо работают без внешних ML-библиотек.
"""

import random
import math
from collections import Counter, defaultdict

random.seed(42)


# =============================================================================
# ЧАСТЬ 1: ОПИСАНИЕ АРХИТЕКТУР
# =============================================================================

class TransformerBlock:
    """Базовый блок трансформера с attention"""

    def __init__(self, d_model=64, n_heads=4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Инициализация весов (упрощённо)
        self.W_q = [random.gauss(0, 1 / math.sqrt(d_model)) for _ in range(d_model * d_model)]
        self.W_k = [random.gauss(0, 1 / math.sqrt(d_model)) for _ in range(d_model * d_model)]
        self.W_v = [random.gauss(0, 1 / math.sqrt(d_model)) for _ in range(d_model * d_model)]

    def attention(self, query, key, value):
        """Scaled dot-product attention"""
        score = sum(q * k for q, k in zip(query, key)) / math.sqrt(self.d_k)
        weight = 1 / (1 + math.exp(-score))  # sigmoid для упрощения
        return [w * v for w, v in zip([weight] * len(value), value)]


class EmbeddingLayer:
    """Слой эмбеддингов — преобразование токенов в векторы"""

    def __init__(self, vocab_size=30000, d_model=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Каждому токену — случайный вектор
        random.seed(42)
        self.embeddings = [
            [random.gauss(0, 0.1) for _ in range(d_model)]
            for _ in range(vocab_size)
        ]

    def encode(self, token_id):
        if token_id < self.vocab_size:
            return self.embeddings[token_id]
        return [0.0] * self.d_model


class BERTArchitecture:
    """
    BERT (Bidirectional Encoder Representations from Transformers)

    Ключевые особенности:
    - Encoder-only: использует только encoder часть трансформера
    - Двунаправленное внимание: видит контекст слева и справа
    - Предобучение: Masked Language Model (MLM) + Next Sentence Prediction (NSP)
    - Архитектура: Multiple encoder stacked layers

    Параметры典型ных моделей:
    - BERT-base: 12 слоёв, 768 скрытых, 12 голов, 110M параметров
    - BERT-large: 24 слоя, 1024 скрытых, 16 голов, 340M параметров

    Masked Language Model (MLM):
    - Случайно маскирует 15% токенов
    - Задача: предсказать замаскированный токен
    - Формула: P(x_m | x_1, ..., x_{m-1}, x_{m+1}, ..., x_n)
    """

    def __init__(self, n_layers=6, d_model=64, n_heads=4):
        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding = EmbeddingLayer(d_model=d_model)
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]

    def encode(self, tokens):
        """Кодирование входных токенов через все слои encoder"""
        hidden_states = [self.embedding.encode(t) for t in tokens]

        # Проход через encoder слои (双向ное внимание)
        for layer in self.layers:
            new_states = []
            for i, state in enumerate(hidden_states):
                # BERT: внимание ко ВСЕМ позициям (bidirectional)
                context = [0.0] * self.d_model
                for j, other in enumerate(hidden_states):
                    out = layer.attention(state, other, other)
                    context = [c + o for c, o in zip(context, out)]
                new_states.append(context)
            hidden_states = new_states

        return hidden_states

    def get_cls_representation(self, tokens):
        """Получение [CLS] представления для классификации"""
        states = self.encode(tokens)
        return states[0] if states else [0.0] * self.d_model

    def description(self):
        return """
    ╔══════════════════════════════════════════════════════════╗
    ║              BERT Architecture Overview                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Тип: Encoder-Only (только кодировщик)                  ║
    ║  Внимание: Bidirectional (двунаправленное)              ║
    ║  Предобучение: MLM + NSP                               ║
    ║  Задача: Понимание контекста (NLU)                     ║
    ║  Выход: Dense представление всего входа                  ║
    ╚══════════════════════════════════════════════════════════╝

    Архитектура BERT:
    ┌─────────────────────────────────────────┐
    │  Input:  [CLS] token_1 token_2 ... [SEP]│
    │  ┌─────────────────────────────────┐     │
    │  │ Token Embeddings + Position     │     │
    │  │ + Segment Embeddings            │     │
    │  └──────────────┬──────────────────┘     │
    │                 ▼                        │
    │  ┌─────────────────────────────────┐     │
    │  │    Encoder Layer (x N)          │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Multi-Head Attention    │    │     │
    │  │  │ (双向, видит всё)       │    │     │
    │  │  └────────────┬────────────┘    │     │
    │  │               ▼                 │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Add & Layer Norm        │    │     │
    │  │  └────────────┬────────────┘    │     │
    │  │               ▼                 │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Feed-Forward Network    │    │     │
    │  │  └────────────┬────────────┘    │     │
    │  │               ▼                 │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Add & Layer Norm        │    │     │
    │  │  └─────────────────────────┘    │     │
    │  └─────────────────────────────────┘     │
    │                 ▼                        │
    │  Output: [CLS] → Classification         │
    │          token_i → MLM prediction        │
    └─────────────────────────────────────────┘
    """


class GPTArchitecture:
    """
    GPT (Generative Pre-trained Transformer)

    Ключевые особенности:
    - Decoder-only: использует только decoder часть трансформера
    - Авторегрессионное: генерирует токен за токеном слева направо
    - Предобучение: Causal Language Modeling (CLM)
    - Архитектура: Multiple decoder layers с causal mask

    Параметры典型ных моделей:
    - GPT-2: 48 слоёв, 1600 скрытых, 25 голов, 1.5B параметров
    - GPT-3: 96 слоёв, 12288 скрытых, 96 голов, 175B параметров

    Causal Language Model:
    - Каждый токен может видеть только предыдущие токены
    - Формула: P(x_t | x_1, ..., x_{t-1})
    - Masking: triangular mask (триугольная маска)
    """

    def __init__(self, n_layers=6, d_model=64, n_heads=4):
        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding = EmbeddingLayer(d_model=d_model)
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]

    def causal_attention_mask(self, seq_len):
        """Создание causal mask: токен видит только предыдущие"""
        mask = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                row.append(1 if j <= i else 0)
            mask.append(row)
        return mask

    def encode(self, tokens, use_causal_mask=True):
        """Кодирование с каузальной маской (авторегрессионное)"""
        hidden_states = [self.embedding.encode(t) for t in tokens]
        seq_len = len(tokens)
        mask = self.causal_attention_mask(seq_len) if use_causal_mask else None

        for layer in self.layers:
            new_states = []
            for i, state in enumerate(hidden_states):
                context = [0.0] * self.d_model
                for j, other in enumerate(hidden_states):
                    # GPT: внимание только к предыдущим токенам (каузальное)
                    if mask is None or mask[i][j]:
                        out = layer.attention(state, other, other)
                        context = [c + o for c, o in zip(context, out)]
                new_states.append(context)
            hidden_states = new_states

        return hidden_states

    def generate_next_token(self, tokens, temperature=1.0):
        """Генерация следующего токена (авторегрессионно)"""
        states = self.encode(tokens)
        last_state = states[-1]

        # Упрощённое "предсказание" — weighted random по эмбеддингам
        probs = []
        for i in range(min(100, self.embedding.vocab_size)):
            emb = self.embedding.encode(i)
            score = sum(a * b for a, b in zip(last_state, emb))
            probs.append((i, score))

        # Softmax с температурой
        max_score = max(p for _, p in probs)
        exp_scores = [(t, math.exp((s - max_score) / temperature)) for t, s in probs]
        total = sum(e for _, e in exp_scores)
        probs = [(t, e / total) for t, e in exp_scores]

        # Сэмплирование
        r = random.random()
        cumulative = 0
        for token_id, prob in probs:
            cumulative += prob
            if r <= cumulative:
                return token_id
        return probs[-1][0]

    def generate(self, prompt_tokens, max_new_tokens=5, temperature=1.0):
        """Авторегрессионная генерация текста"""
        tokens = list(prompt_tokens)
        generated = []
        for _ in range(max_new_tokens):
            next_token = self.generate_next_token(tokens, temperature)
            generated.append(next_token)
            tokens.append(next_token)
        return tokens, generated

    def description(self):
        return """
    ╔══════════════════════════════════════════════════════════╗
    ║              GPT Architecture Overview                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Тип: Decoder-Only (только декодер)                     ║
    ║  Внимание: Causal (авторегрессионное, слева направо)    ║
    ║  Предобучение: Causal Language Model (CLM)              ║
    ║  Задача: Генерация текста (NLG)                         ║
    ║  Выход: Вероятность следующего токена                   ║
    ╚══════════════════════════════════════════════════════════╝

    Архитектура GPT:
    ┌─────────────────────────────────────────┐
    │  Input:  token_1 token_2 ... token_n    │
    │  ┌─────────────────────────────────┐     │
    │  │ Token Embeddings + Position     │     │
    │  │ (без Segment)                  │     │
    │  └──────────────┬──────────────────┘     │
    │                 ▼                        │
    │  ┌─────────────────────────────────┐     │
    │  │    Decoder Layer (x N)          │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Masked Multi-Head       │    │     │
    │  │  │ Self-Attention          │    │     │
    │  │  │ (каузальная маска)      │    │     │
    │  │  └────────────┬────────────┘    │     │
    │  │               ▼                 │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Add & Layer Norm        │    │     │
    │  │  └────────────┬────────────┘    │     │
    │  │               ▼                 │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Feed-Forward Network    │    │     │
    │  │  └────────────┬────────────┘    │     │
    │  │               ▼                 │     │
    │  │  ┌─────────────────────────┐    │     │
    │  │  │ Add & Layer Norm        │    │     │
    │  │  └─────────────────────────┘    │     │
    │  └─────────────────────────────────┘     │
    │                 ▼                        │
    │  Output: P(next_token | all previous)   │
    └─────────────────────────────────────────┘
    """


# =============================================================================
# ЧАСТЬ 2: СИМУЛЯЦИЯ ТОКЕНИЗАЦИИ И ВОКАБУЛЯРА
# =============================================================================

class SimpleTokenizer:
    """Простой токенизатор для демонстрации"""

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self._add_special_tokens()

    def _add_special_tokens(self):
        special = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for i, token in enumerate(special):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        self.vocab_size = len(special)

    def add_word(self, word):
        if word not in self.word_to_id:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1

    def tokenize(self, text):
        """Простая токенизация по пробелам и нижний регистр"""
        tokens = text.lower().split()
        return tokens

    def encode(self, text, add_special=True):
        tokens = self.tokenize(text)
        ids = []
        if add_special:
            ids.append(self.word_to_id['[CLS]'])
        for token in tokens:
            self.add_word(token)
            ids.append(self.word_to_id[token])
        if add_special:
            ids.append(self.word_to_id['[SEP]'])
        return ids

    def decode(self, ids):
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_word:
                tokens.append(self.id_to_word[id_])
        return ' '.join(tokens)


# =============================================================================
# ЧАСТЬ 3: СИМУЛЯЦИЯ CLASSIFICATION HEAD
# =============================================================================

class ClassificationHead:
    """Классификационная голова для fine-tuning"""

    def __init__(self, d_model=64, n_classes=2):
        self.d_model = d_model
        self.n_classes = n_classes
        # Простые веса для классификации
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(d_model)]
            for _ in range(n_classes)
        ]
        self.bias = [0.0] * n_classes

    def predict(self, hidden_state):
        """Предсказание класса из [CLS] представления"""
        logits = []
        for i in range(self.n_classes):
            logit = sum(h * w for h, w in zip(hidden_state, self.weights[i]))
            logit += self.bias[i]
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]

        return probs, probs.index(max(probs))


# =============================================================================
# ЧАСТЬ 4: СИМУЛЯЦИЯ ZERO-SHOT КЛАССИФИКАЦИИ
# =============================================================================

class ZeroShotClassifier:
    """
    Zero-shot классификация: классификация без обучения на целевых классах.

    Принцип:
    1. Формулируем каждый класс как текстовую гипотезу
    2. Оцениваем вероятность каждого класса через language model
    3. Выбираем класс с наибольшей вероятностью

    Примеры гипотез:
    - "Это отзыв о еде"
    - "Это отзыв об обслуживании"
    - "Это отзыв о атмосфере"
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_similarity(self, text_tokens, hypothesis_tokens):
        """Вычисление схожести между текстом и гипотезой (упрощённо)"""
        text_set = set(text_tokens)
        hyp_set = set(hypothesis_tokens)

        # Пересечение — общие слова
        intersection = text_set & hyp_set
        union = text_set | hyp_set

        if not union:
            return 0.0

        # Jaccard similarity + вес по позициям
        jaccard = len(intersection) / len(union)

        # Дополнительно: насколько гипотеза "покрывает" текст
        coverage = len(intersection) / len(text_set) if text_set else 0

        return 0.6 * jaccard + 0.4 * coverage

    def classify(self, text, candidate_labels):
        """Zero-shot классификация текста"""
        text_tokens = self.tokenizer.tokenize(text)

        scores = {}
        for label in candidate_labels:
            hypothesis = f"это {label.lower()}"
            hyp_tokens = self.tokenizer.tokenize(hypothesis)
            scores[label] = self.compute_similarity(text_tokens, hyp_tokens)

        # Нормализация в вероятности
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        predicted = max(scores, key=scores.get)
        return scores, predicted


# =============================================================================
# ЧАСТЬ 5: SIMULATED FINE-TUNING
# =============================================================================

class SimulatedFineTuning:
    """
    Симуляция процесса fine-tuning предобученной модели.

    Fine-tuning — это дообучение предобученной модели на конкретной задаче.

    Этапы:
    1. Загрузка предобученной модели
    2. Добавление classification head
    3. Обучение на размеченных данных
    4. Оценка на тестовых данных

    Гиперпараметры typical:
    - Learning rate: 2e-5 до 5e-5 (маленький, чтобы не разрушить предобученные веса)
    - Batch size: 16-32
    - Epochs: 2-4
    - Warmup: 10% от общего числа шагов
    """

    def __init__(self, base_model, tokenizer, n_classes=2, lr=2e-5):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.head = ClassificationHead(d_model=base_model.d_model, n_classes=n_classes)
        self.lr = lr
        self.n_classes = n_classes
        self.label_map = {}

    def set_labels(self, labels):
        """Установка меток классов"""
        self.label_map = {i: label for i, label in enumerate(labels)}

    def train_step(self, text, label):
        """Один шаг обучения (упрощённый)"""
        # Прямой проход
        tokens = self.tokenizer.encode(text, add_special=False)[:10]
        if not tokens:
            tokens = [0]

        hidden_states = self.base_model.encode(tokens)
        cls_hidden = hidden_states[0]

        probs, predicted = self.head.predict(cls_hidden)

        # Cross-entropy loss
        target = [0.0] * self.n_classes
        target[label] = 1.0
        loss = -sum(t * math.log(p + 1e-8) for t, p in zip(target, probs))

        # Обновление весов (упрощённый gradient descent)
        for i in range(self.n_classes):
            error = probs[i] - target[i]
            for j in range(self.head.d_model):
                self.head.weights[i][j] -= self.lr * error * cls_hidden[j]
            self.head.bias[i] -= self.lr * error

        return loss, predicted

    def train(self, train_data, n_epochs=3):
        """Обучение на тренировочных данных"""
        print(f"\n{'='*60}")
        print(f"Fine-tuning: {n_epochs} epochs, lr={self.lr}")
        print(f"{'='*60}")

        all_losses = []
        for epoch in range(n_epochs):
            random.shuffle(train_data)
            epoch_loss = 0
            correct = 0
            total = 0

            for text, label in train_data:
                loss, pred = self.train_step(text, label)
                epoch_loss += loss
                if pred == label:
                    correct += 1
                total += 1

            avg_loss = epoch_loss / len(train_data)
            accuracy = correct / total * 100
            all_losses.append(avg_loss)

            print(f"Epoch {epoch + 1}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")

        return all_losses

    def evaluate(self, test_data):
        """Оценка на тестовых данных"""
        correct = 0
        total = 0
        predictions = []

        for text, label in test_data:
            tokens = self.tokenizer.encode(text, add_special=False)[:10]
            if not tokens:
                tokens = [0]

            hidden_states = self.base_model.encode(tokens)
            cls_hidden = hidden_states[0]
            probs, predicted = self.head.predict(cls_hidden)

            if predicted == label:
                correct += 1
            total += 1
            predictions.append((text, self.label_map.get(label, str(label)),
                              self.label_map.get(predicted, str(predicted)),
                              probs[label]))

        accuracy = correct / total * 100 if total > 0 else 0
        return accuracy, predictions


# =============================================================================
# ЧАСТЬ 6: КОМПАНИЯР МОДЕЛЕЙ
# =============================================================================

def compare_bert_vs_gpt():
    """Сравнение BERT и GPT архитектур"""
    comparison = """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                    СРАВНЕНИЕ BERT vs GPT                             ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  Характеристика    │        BERT         │         GPT               ║
    ║  ─────────────────┼────────────────────┼─────────────────────────  ║
    ║  Архитектура       │  Encoder-only       │  Decoder-only             ║
    ║  Внимание          │  Bidirectional      │  Causal (unidirectional)  ║
    ║  Предобучение      │  MLM + NSP          │  Causal LM                ║
    ║  Задача            │  Понимание (NLU)    │  Генерация (NLG)          ║
    ║  Вход для задачи   │  Весь текст сразу   │  Последовательно          ║
    ║  [CLS] токен       │  Да (для классиф.)  │  Нет                      ║
    ║  Masked tokens     │  Да (15%)           │  Нет                      ║
    ║  Контекст          │  Полный双向          │  Только левый             ║
    ║                                                                      ║
    ║  Лучшие задачи:                                                      ║
    ║  - BERT: классификация, NER, QA, сходство текстов                   ║
    ║  - GPT: генерация текста, диалоги, суммаризация, перевод            ║
    ║                                                                      ║
    ║  Fine-tuning:                                                        ║
    ║  - BERT: добавить classification head, обучить на метках            ║
    ║  - GPT: few-shot/in-context learning (без обновления весов)         ║
    ║                                                                      ║
    ║  Примеры моделей:                                                    ║
    ║  - BERT: bert-base-uncased, bert-large-cased                        ║
    ║  - GPT: GPT-2, GPT-3, GPT-4, ChatGPT                              ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    return comparison


# =============================================================================
# ДЕМО 1: Архитектура BERT (encoder-only)
# =============================================================================

def demo1_bert_architecture():
    """Демонстрация архитектуры BERT"""
    print("\n" + "=" * 70)
    print("ДЕМО 1: Архитектура BERT (Encoder-Only)")
    print("=" * 70)

    print(BERTArchitecture().description())

    # Создание модели
    bert = BERTArchitecture(n_layers=3, d_model=32, n_heads=4)
    tokenizer = SimpleTokenizer()

    # Пример текста
    text = "The cat sat on the mat"
    print(f"\nВходной текст: '{text}'")

    # Токенизация
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special=False)[:6]
    print(f"Токены: {tokens}")
    print(f"Token IDs: {token_ids}")

    # Кодирование через BERT
    print("\n--- BERT Encoding (Bidirectional) ---")
    hidden_states = bert.encode(token_ids)
    print(f"Количество hidden states: {len(hidden_states)}")
    print(f"Размерность каждого hidden state: {len(hidden_states[0])}")

    # [CLS] представление
    cls_repr = bert.get_cls_representation(token_ids)
    print(f"\n[CLS] representation (первые 5 компонент):")
    print(f"  {[round(x, 4) for x in cls_repr[:5]]}")

    # Демонстрация bidirectional attention
    print("\n--- Bidirectional Attention Demo ---")
    print("В BERT каждый токен 'видит' ВСЕ остальные токены:")
    print(f"  Токен '{tokens[0]}' → внимание ко всем: {[tokens[min(i, len(tokens)-1)] for i in range(len(tokens))]}")
    print(f"  Токен '{tokens[2]}' → внимание ко всем: {[tokens[min(i, len(tokens)-1)] for i in range(len(tokens))]}")

    # Masked Language Model пример
    print("\n--- Masked Language Model (MLM) ---")
    masked_text = "The [MASK] sat on the mat"
    print(f"Замаскированный текст: '{masked_text}'")
    print("Задача BERT: предсказать '[MASK]' → модель должна угадать 'cat'")

    return bert, tokenizer


# =============================================================================
# ДЕМО 2: Архитектура GPT (decoder-only)
# =============================================================================

def demo2_gpt_architecture():
    """Демонстрация архитектуры GPT"""
    print("\n" + "=" * 70)
    print("ДЕМО 2: Архитектура GPT (Decoder-Only)")
    print("=" * 70)

    print(GPTArchitecture().description())

    # Создание модели
    gpt = GPTArchitecture(n_layers=3, d_model=32, n_heads=4)
    tokenizer = SimpleTokenizer()

    # Пример текста
    prompt = "The quick brown"
    print(f"\nПромпт: '{prompt}'")

    # Токенизация
    tokens = tokenizer.tokenize(prompt)
    token_ids = tokenizer.encode(prompt, add_special=False)[:4]
    print(f"Токены: {tokens}")
    print(f"Token IDs: {token_ids}")

    # Демонстрация causal mask
    print("\n--- Causal Attention Mask ---")
    mask = gpt.causal_attention_mask(len(token_ids))
    print("Causal mask (каждый токен видит только предыдущие):")
    for i, row in enumerate(mask):
        token = tokens[i] if i < len(tokens) else f"t{i}"
        print(f"  {token:10} → {row}")

    # Кодирование через GPT
    print("\n--- GPT Encoding (Causal) ---")
    hidden_states = gpt.encode(token_ids)
    print(f"Количество hidden states: {len(hidden_states)}")

    # Генерация следующего токена
    print("\n--- Генерация следующего токена ---")
    random.seed(42)
    next_token_id = gpt.generate_next_token(token_ids)
    next_token = tokenizer.id_to_word.get(next_token_id, f"[{next_token_id}]")
    print(f"После '{prompt}' модель генерирует: '{next_token}'")

    # Авторегрессионная генерация
    print("\n--- Авторегрессионная генерация ---")
    random.seed(42)
    full_tokens, generated = gpt.generate(token_ids, max_new_tokens=5)
    generated_tokens = [tokenizer.id_to_word.get(t, f"[{t}]") for t in generated]
    print(f"Промпт: '{prompt}'")
    print(f"Сгенерировано: {' '.join(generated_tokens)}")
    print(f"Полная последовательность: {tokenizer.decode(full_tokens)}")

    # Causal vs Bidirectional
    print("\n--- Ключевое отличие: Causal vs Bidirectional ---")
    print("GPT (Causal): 'sat' видит только 'The', 'quick', 'brown'")
    print("BERT (Bidirectional): 'sat' видит 'The', 'quick', 'brown', 'on', 'the', 'mat'")

    return gpt, tokenizer


# =============================================================================
# ДЕМО 3: Fine-tuning для классификации
# =============================================================================

def demo3_fine_tuning():
    """Демонстрация fine-tuning BERT для классификации"""
    print("\n" + "=" * 70)
    print("ДЕМО 3: Fine-tuning BERT для классификации")
    print("=" * 70)

    # Создание предобученной модели
    bert = BERTArchitecture(n_layers=3, d_model=32, n_heads=4)
    tokenizer = SimpleTokenizer()

    # Подготовка данных для sentiment analysis
    positive_texts = [
        "this movie is great and wonderful",
        "excellent acting and good story",
        "amazing film highly recommend",
        "best movie I have seen",
        "fantastic performance loved it",
        "brilliant directing superb film",
        "outstanding movie very enjoyable",
        "perfect film must watch",
    ]

    negative_texts = [
        "this movie is terrible and boring",
        "bad acting waste of time",
        "awful film do not watch",
        "worst movie ever made",
        "horrible performance very bad",
        "poor directing boring film",
        "dreadful movie not enjoyable",
        "terrible film avoid it",
    ]

    train_data = [(text, 1) for text in positive_texts]
    train_data += [(text, 0) for text in negative_texts]

    test_data = [
        ("great movie wonderful", 1),
        ("terrible boring waste", 0),
        ("excellent film amazing", 1),
        ("bad acting awful", 0),
    ]

    print("\n--- Данные для sentiment analysis ---")
    print(f"Положительные отзывы: {len(positive_texts)}")
    print(f"Отрицательные отзывы: {len(negative_texts)}")
    print(f"Примеры положительных:")
    for text in positive_texts[:3]:
        print(f"  + {text}")
    print(f"Примеры отрицательных:")
    for text in negative_texts[:3]:
        print(f"  - {text}")

    # Fine-tuning
    ft = SimulatedFineTuning(bert, tokenizer, n_classes=2, lr=2e-4)
    ft.set_labels(["negative", "positive"])

    print("\n--- Этапы fine-tuning ---")
    print("1. Загрузка предобученного BERT")
    print("2. Добавление classification head (2 класса)")
    print("3. Обучение на размеченных данных")
    print("4. Оценка на тестовых данных")

    # Обучение
    losses = ft.train(train_data, n_epochs=3)

    # Оценка
    accuracy, predictions = ft.evaluate(test_data)
    print(f"\n--- Результаты на тесте ---")
    print(f"Accuracy: {accuracy:.1f}%")

    print("\n--- Детали предсказаний ---")
    for text, true_label, pred_label, confidence in predictions:
        status = "✓" if true_label == pred_label else "✗"
        print(f"  {status} '{text}'")
        print(f"    Истинный: {true_label}, Предсказанный: {pred_label} "
              f"(уверенность: {confidence:.3f})")

    # Сравнение approaches
    print("\n--- Fine-tuning vs In-Context Learning ---")
    print("Fine-tuning (BERT):")
    print("  - Обновляет веса модели на конкретной задаче")
    print("  - Требует размеченные данные")
    print("  - Высокая точность для узких задач")
    print("  - Быстрое инференс после обучения")
    print("\nIn-Context Learning (GPT):")
    print("  - Не обновляет веса модели")
    print("  - Работает с few-shot примерами")
    print("  - Более гибкий, но менее точный")
    print("  - Медленнее при инференсе")


# =============================================================================
# ДЕМО 4: Сравнение подходов
# =============================================================================

def demo4_comparison():
    """Демонстрация сравнения BERT vs GPT подходов"""
    print("\n" + "=" * 70)
    print("ДЕМО 4: Сравнение BERT vs GPT подходов")
    print("=" * 70)

    tokenizer = SimpleTokenizer()
    bert = BERTArchitecture(n_layers=3, d_model=32, n_heads=4)
    gpt = GPTArchitecture(n_layers=3, d_model=32, n_heads=4)

    # Задача 1: Классификация текста
    print("\n--- Задача 1: Классификация текста ---")
    text = "this movie is great and wonderful"
    print(f"Текст: '{text}'")

    # BERT подход
    tokens = tokenizer.encode(text, add_special=False)[:8]
    hidden_states = bert.encode(tokens)
    cls_repr = hidden_states[0]

    head = ClassificationHead(d_model=32, n_classes=2)
    probs, pred = head.predict(cls_repr)
    labels = ["negative", "positive"]
    print(f"\nBERT подход (Fine-tuning):")
    print(f"  Предсказание: {labels[pred]} (уверенность: {probs[pred]:.3f})")
    print(f"  Метод: Encode → [CLS] → Classification Head")

    # GPT подход (zero-shot)
    print(f"\nGPT подход (Zero-shot):")
    print(f"  Метод: Промпт → 'Is this positive or negative?' → Генерация ответа")
    print(f"  Пример промпта:")
    print(f"    'Classify: \"{text}\". Is it positive or negative?'")
    print(f"  Модель генерирует: 'positive'")
    print(f"  Без обновления весов!")

    # Задача 2: Генерация текста
    print("\n--- Задача 2: Генерация текста ---")
    prompt = "The weather today is"
    print(f"Промпт: '{prompt}'")

    prompt_tokens = tokenizer.encode(prompt, add_special=False)[:5]

    # GPT генерация
    random.seed(42)
    full_tokens, generated = gpt.generate(prompt_tokens, max_new_tokens=5)
    gen_tokens = [tokenizer.id_to_word.get(t, f"[{t}]") for t in generated]
    print(f"\nGPT генерация: {' '.join(gen_tokens)}")

    # BERT limitations
    print(f"\nBERT ограничения для генерации:")
    print(f"  BERT не может генерировать текст последовательно")
    print(f"  Он видит весь текст сразу, не может предсказывать 'следующий' токен")

    # Задача 3: Zero-shot классификация
    print("\n--- Задача 3: Zero-shot классификация ---")
    zs = ZeroShotClassifier(tokenizer)

    texts = [
        "this movie is great and wonderful",
        "terrible food bad service",
        "amazing acting beautiful scenery",
    ]
    candidate_labels = ["positive", "negative"]

    for text in texts:
        scores, predicted = zs.classify(text, candidate_labels)
        print(f"\n  Текст: '{text}'")
        print(f"  Вероятности: {scores}")
        print(f"  Предсказание: {predicted}")

    # Сводная таблица
    print("\n--- Сводная таблица ---")
    comparison = compare_bert_vs_gpt()
    print(comparison)

    # Рекомендации по выбору
    print("\n--- Рекомендации по выбору ---")
    print("Выбирайте BERT когда:")
    print("  - Задача: классификация, NER, QA, сходство текстов")
    print("  - Есть размеченные данные для fine-tuning")
    print("  - Нужна высокая точность на узкой задаче")
    print("  - Требуется быстрый инференс")
    print()
    print("Выбирайте GPT когда:")
    print("  - Задача: генерация текста, диалоги, суммаризация")
    print("  - Мало размеченных данных (few-shot)")
    print("  - Нужна гибкость и креативность")
    print("  - Задача требует рассуждений")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    """Запуск всех демонстраций"""
    print("\n" + "=" * 70)
    print("  ПРЕДОБУЧЕННЫЕ ЯЗЫКОВЫЕ МОДЕЛИ: BERT И GPT")
    print("  68_pretrained_models.py")
    print("=" * 70)

    # Демо 1: BERT
    demo1_bert_architecture()

    # Демо 2: GPT
    demo2_gpt_architecture()

    # Демо 3: Fine-tuning
    demo3_fine_tuning()

    # Демо 4: Сравнение
    demo4_comparison()

    print("\n" + "=" * 70)
    print("  ВЫВОДЫ")
    print("=" * 70)
    print("""
    1. BERT (Encoder-only) и GPT (Decoder-only) — два основных подхода
       к предобученным языковым моделям.

    2. BERT лучше для задач понимания текста (NLU):
       - Классификация, NER, QA, сходство текстов
       - Bidirectional attention видит весь контекст
       - Fine-tuning с classification head

    3. GPT лучше для задач генерации текста (NLG):
       - Генерация, диалоги, суммаризация
       - Causal attention (слева направо)
       - Few-shot/in-context learning

    4. Fine-tuning:
       - Обновление весов предобученной модели
       - Требует размеченные данные
       - Высокая точность для узких задач

    5. Zero-shot классификация:
       - Без обучения на целевых классах
       - Формулировка классов как текстовых гипотез
       - Гибкость, но менее точная

    6. Выбор подхода зависит от:
       - Типа задачи (понимание vs генерация)
       - Наличия размеченных данных
       - Требований к точности и скорости
    """)


if __name__ == "__main__":
    main()
