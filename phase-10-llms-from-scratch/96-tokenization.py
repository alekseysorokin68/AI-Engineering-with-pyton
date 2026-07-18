"""
Токенизация — основы для LLM
=============================
BPE (Byte Pair Encoding), WordPiece (упрощённо), токенизация текста, словарь токенов.

Без внешних зависимостей (tiktoken, transformers, sentencepiece).
"""

import random
from collections import Counter, defaultdict

random.seed(42)


# =============================================================================
# 1. BPE (Byte Pair Encoding) — обучение словаря
# =============================================================================

class BPETokenizer:
    """Минимальная реализация BPE."""

    def __init__(self):
        self.merges = []  # список пар合并规则: [(old, new), ...]
        self.vocab = set()

    @staticmethod
    def _get_stats(word_freqs: dict[str, int]) -> Counter:
        """Подсчёт частот всех пар символов."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(pair: tuple[str, str], word_freqs: dict[str, int]) -> dict[str, int]:
        """Объединение самой частой пары во всех словах."""
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus: str, num_merges: int = 10):
        """Обучение BPE на корпусе текста."""
        # Шаг 1: разбиваем текст на слова и считаем частоты
        words = corpus.lower().split()
        word_freq = Counter(words)

        # Каждое слово → символы, разделённые пробелами, с </w> в конце
        word_freqs = {}
        for word, freq in word_freq.items():
            symbols = list(word) + ["</w>"]
            word_freqs[" ".join(symbols)] = freq

        print(f"  Начальный словарь ({len(word_freqs)} слов):")
        for w, f in sorted(word_freqs.items())[:5]:
            print(f"    {w}  ×{f}")
        print(f"  ... и ещё {len(word_freqs) - 5} слов\n")

        # Шаг 2: итеративное слияние
        self.vocab = set()
        for symbols in word_freqs:
            for s in symbols.split():
                self.vocab.add(s)

        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best = pairs.most_common(1)[0]
            pair, freq = best
            merged = "".join(pair)

            self.merges.append((pair, merged))
            self.vocab.add(merged)
            word_freqs = self._merge_pair(pair, word_freqs)

            print(f"  Merge #{i + 1}: {pair} -> '{merged}'  (freq={freq})")

        print(f"\n  Итого мержей: {len(self.merges)}")
        print(f"  Размер словаря: {len(self.vocab)}")

    def tokenize(self, text: str) -> list[str]:
        """Токенизация текста обученными мержами."""
        tokens = list(text.lower()) + ["</w>"]

        for pair, merged in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = merged
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens


# =============================================================================
# 2. WordPiece — упрощённая реализация
# =============================================================================

class WordPieceTokenizer:
    """
    Упрощённый WordPiece (как в BERT).
    Использует жадный longest-match-first подход.
    """

    def __init__(self, unk_token="[UNK]"):
        self.vocab: dict[str, int] = {}
        self.unk_token = unk_token

    def build_vocab(self, corpus: str, max_vocab_size: int = 50):
        """Построение словаря по корпусу."""
        words = corpus.lower().split()
        word_freq = Counter(words)

        # Начальный словарь — все отдельные символы + целые слова
        all_chars = set()
        for word in word_freq:
            all_chars.update(word)

        vocab = list(all_chars)
        vocab.append("[UNK]")
        vocab.append("[PAD]")
        vocab.append("[CLS]")
        vocab.append("[SEP]")

        # Добавляем слова по частоте (включая подслова с ##)
        for word, freq in word_freq.most_common(max_vocab_size):
            if word not in vocab:
                # Разбиваем на подслова и добавляем их в словарь
                for i in range(len(word)):
                    for j in range(i + 1, len(word) + 1):
                        substr = word[i:j]
                        if i > 0:
                            substr = "##" + substr
                        if substr not in vocab:
                            vocab.append(substr)
                if word not in vocab:
                    vocab.append(word)

        self.vocab = {token: idx for idx, token in enumerate(vocab)}

    @staticmethod
    def _tokenize_word(word: str) -> str:
        """Разбиение слова на подслова."""
        if len(word) == 0:
            return ""
        if len(word) == 1:
            return word
        # Упрощённый: просто по символам
        return "".join([c + "##" for c in word[:-1]] + [word[-1]])

    def tokenize(self, text: str) -> list[str]:
        """Токенизация текста с longest-match-first стратегией."""
        words = text.lower().split()
        all_tokens = ["[CLS]"]

        for word in words:
            chars = list(word)
            start = 0
            while start < len(chars):
                end = len(chars)
                found = False
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        all_tokens.append(substr)
                        found = True
                        break
                    end -= 1
                if not found:
                    all_tokens.append(self.unk_token)
                start = end

        all_tokens.append("[SEP]")
        return all_tokens

    def get_vocab_size(self) -> int:
        return len(self.vocab)


# =============================================================================
# 3. Токенизация текста — утилиты
# =============================================================================

def simple_tokenize(text: str) -> list[str]:
    """Простая токенизация по пробелам и знакам препинания."""
    import re
    # Разбиваем по не-алфавитным символам, сохраняя разделители
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


def char_tokenize(text: str) -> list[str]:
    """Посимвольная токенизация."""
    return list(text)


def word_tokenize(text: str) -> list[str]:
    """Токенизация по словам."""
    return text.split()


def build_vocab_from_tokens(token_lists: list[list[str]],
                            min_freq: int = 1) -> dict[str, int]:
    """Построение словаря (token -> index) из списка списков токенов."""
    freq = Counter()
    for tokens in token_lists:
        freq.update(tokens)

    vocab = {"[PAD]": 0, "[UNK]": 1}
    idx = 2
    for token, count in freq.most_common():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Кодирование токенов в индексы."""
    unk_idx = vocab.get("[UNK]", 0)
    return [vocab.get(t, unk_idx) for t in tokens]


def decode(indices: list[int], idx_to_token: dict[int, str]) -> str:
    """Декодирование индексов обратно в токены."""
    return " ".join(idx_to_token.get(i, "?") for i in indices)


# =============================================================================
# 4. Специальные токены
# =============================================================================

class SpecialTokens:
    """Специальные токены, используемые в LLM-токенизаторах."""

    PAD = "[PAD]"      # Заполнитель (padding)
    UNK = "[UNK]"      # Неизвестный токен
    CLS = "[CLS]"      # Классификационный (начало последовательности)
    SEP = "[SEP]"      # Разделитель
    MASK = "[MASK]"    # Маскированный токен (для MLM)
    BOS = "[BOS]"      # Beginning of Sequence
    EOS = "[EOS]"      # End of Sequence

    @staticmethod
    def add_special_tokens(tokens: list[str],
                           cls_token: bool = True,
                           sep_token: bool = True) -> list[str]:
        """Добавление специальных токенов к списку."""
        result = []
        if cls_token:
            result.append(SpecialTokens.CLS)
        result.extend(tokens)
        if sep_token:
            result.append(SpecialTokens.SEP)
        return result

    @staticmethod
    def create_attention_mask(token_ids: list[int],
                              pad_id: int = 0) -> list[int]:
        """Создание маски внимания (1 для реальных токенов, 0 для pad)."""
        return [1 if t != pad_id else 0 for t in token_ids]


# =============================================================================
# ДЕМО 1: BPE — обучение словаря
# =============================================================================

print("=" * 60)
print("ДЕМО 1: BPE — обучение словаря")
print("=" * 60)

corpus_bpe = (
    "low low low low low "
    "lower lower lower "
    "newest newest newest newest "
    "widest widest widest "
    "new new new new new "
    "low low low low"
)

print(f"Корпус: \"{corpus_bpe}\"\n")

bpe = BPETokenizer()
bpe.train(corpus_bpe, num_merges=15)

print(f"\nИтоговый словарь BPE: {sorted(bpe.vocab)}")


# =============================================================================
# ДЕМО 2: Токенизация текста
# =============================================================================

print("\n" + "=" * 60)
print("ДЕМО 2: Токенизация текста")
print("=" * 60)

sample_text = "The transformer model uses attention mechanism to process sequences."

print(f"Исходный текст: \"{sample_text}\"\n")

# Простая токенизация
tokens_simple = simple_tokenize(sample_text)
print(f"По словам и знакам: {tokens_simple}")

# Посимвольная токенизация
tokens_char = char_tokenize(sample_text)
print(f"Посимвольная:      {tokens_char}")

# Токенизация по словам
tokens_word = word_tokenize(sample_text)
print(f"По словам:         {tokens_word}")

# BPE токенизация (обучим на этом корпусе)
corpus_english = (
    "the transformer model uses attention "
    "the model processes sequences with attention "
    "attention is the mechanism of transformers "
    "the model uses attention mechanisms "
    "transformers process sequences using attention "
    "the attention mechanism processes the model "
    "a new model uses a new mechanism "
)

bpe2 = BPETokenizer()
bpe2.train(corpus_english, num_merges=20)

print(f"\nBPE токенизация: {bpe2.tokenize(sample_text.lower())}")

# Построение словаря из токенов
all_token_lists = [tokens_simple, tokens_word]
vocab = build_vocab_from_tokens(all_token_lists)
print(f"\nСловарь (token -> index):")
for tok, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print(f"  {tok:12s} -> {idx}")

encoded = encode(tokens_simple, vocab)
print(f"\nКодирование: {tokens_simple}")
print(f"Индексы:     {encoded}")

idx_to_token = {v: k for k, v in vocab.items()}
decoded = decode(encoded, idx_to_token)
print(f"Декодирование: {decoded}")


# =============================================================================
# ДЕМО 3: Сравнение BPE vs WordPiece
# =============================================================================

print("\n" + "=" * 60)
print("ДЕМО 3: Сравнение BPE vs WordPiece")
print("=" * 60)

test_sentences = [
    "the transformers use attention",
    "tokenization is important for models",
    "understanding subwords helps models",
]

corpus_wp = (
    "the transformer model uses attention mechanism "
    "tokenization is important for models "
    "understanding subwords helps models "
    "subword tokenization splits words "
    "transformers process sequences "
    "the model uses attention "
    "tokenization splits words into subwords "
    "important mechanisms use attention "
    "a model understands subword tokens "
)

# Обучаем BPE
bpe3 = BPETokenizer()
bpe3.train(corpus_wp, num_merges=20)

# Обучаем WordPiece
wp = WordPieceTokenizer()
wp.build_vocab(corpus_wp, max_vocab_size=40)

print(f"\n{'='*60}")
print(f"{'Текст':<45} | {'BPE':<15} | {'WordPiece':<15}")
print(f"{'='*60}")

for sent in test_sentences:
    bpe_tokens = bpe3.tokenize(sent.lower())
    wp_tokens = wp.tokenize(sent.lower())

    print(f"\n\"{sent}\"")
    print(f"  BPE:       {bpe_tokens}")
    print(f"  WordPiece: {wp_tokens}")
    print(f"  Длина BPE:       {len(bpe_tokens)}")
    print(f"  Длина WordPiece: {len(wp_tokens)}")


# =============================================================================
# ДЕМО 4: Специальные токены
# =============================================================================

print("\n" + "=" * 60)
print("ДЕМО 4: Специальные токены")
print("=" * 60)

text = "The model attends to all positions"

print(f"Исходный текст: \"{text}\"\n")

# Токенизация
tokens = word_tokenize(text)
print(f"Токены: {tokens}")

# Добавление специальных токенов
tokens_with_special = SpecialTokens.add_special_tokens(tokens)
print(f"Специальные токены: {tokens_with_special}")

# Создание словаря с special tokens
vocab_with_special = build_vocab_from_tokens([tokens])
all_tokens_list = list(vocab_with_special.keys()) + [
    SpecialTokens.PAD, SpecialTokens.CLS, SpecialTokens.SEP,
    SpecialTokens.MASK, SpecialTokens.BOS, SpecialTokens.EOS
]
full_vocab = {t: i for i, t in enumerate(all_tokens_list)}

print(f"\nПолный словарь с специальными токенами:")
for tok, idx in sorted(full_vocab.items(), key=lambda x: x[1]):
    print(f"  [{idx:2d}] {tok}")

# Кодирование
encoded_special = encode(tokens_with_special, full_vocab)
print(f"\nКодирование с special tokens:")
print(f"  Токены: {tokens_with_special}")
print(f"  Индексы: {encoded_special}")

# Attention mask
pad_idx = full_vocab.get(SpecialTokens.PAD, 0)
pad_token = full_vocab[SpecialTokens.PAD]
tokens_padded = [pad_token, pad_token] + tokens_with_special
print(f"\nPadded токены: {tokens_padded}")

mask = SpecialTokens.create_attention_mask(tokens_padded, pad_id=pad_idx)
print(f"Attention mask: {mask}")

# Декодирование
idx_to_tok = {v: k for k, v in full_vocab.items()}
decoded_back = decode(encoded_special, idx_to_tok)
print(f"\nДекодирование: \"{decoded_back}\"")

# WordPiece demo
print(f"\n--- WordPiece с special tokens ---")
wp2 = WordPieceTokenizer()
wp2.build_vocab(corpus_wp, max_vocab_size=30)
wp_tokens = wp2.tokenize(text.lower())
print(f"WordPiece токены: {wp_tokens}")

wp_tokens_special = SpecialTokens.add_special_tokens(wp_tokens)
print(f"С special tokens: {wp_tokens_special}")


# =============================================================================
# Итоги
# =============================================================================

print("\n" + "=" * 60)
print("ИТОГИ")
print("=" * 60)

print("""
1. BPE (Byte Pair Encoding):
   - Итеративно находит самые частые пары символов и объединяет их
   - Обучение: жадное, O(n × merges)
   - Используется в GPT-2, GPT-3, ChatGPT

2. WordPiece:
   - Longest-match-first стратегия
   - Использует префикс ## для подслов
   - Используется в BERT, DistilBERT

3. Токенизация текста:
   - По словам (пробелы)
   - Посимвольная
   - По regex-паттернам (знаки препинания отдельно)
   - Подсловная (BPE, WordPiece, SentencePiece)

4. Специальные токены:
   - [PAD] — заполнитель для батчей
   - [UNK] — неизвестные слова
   - [CLS] — начало (классификация)
   - [SEP] — разделитель
   - [MASK] — маскирование (обучение)
   - [BOS]/[EOS] — начало/конец последовательности

5. Словарь (vocab):
   - token -> index (encode)
   - index -> token (decode)
   - Чем больше словарь, тем лучше покрытие, но больше памяти
""")
