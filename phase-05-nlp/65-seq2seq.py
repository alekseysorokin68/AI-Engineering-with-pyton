"""
65 — Sequence-to-Sequence модели (Seq2Seq)
==========================================
Encoder-Decoder архитектура, Teacher Forcing, Beam Search Decoding.
Реализация на чистом Python (numpy-подобная логика через встроенные списки).
Без внешних ML-библиотек.

Демонстрации:
  1. Encoder — сжатие последовательности в вектор контекста
  2. Decoder — генерация последовательности по контексту
  3. Teacher Forcing vs Greedy Decoding
  4. Beam Search Decoding
"""

import random
import math
import copy

random.seed(42)

# ============================================================================
#  ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ============================================================================

def sigmoid(x: float) -> float:
    """Сигмоида."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def tanh(x: float) -> float:
    """Гиперболический тангенс."""
    return math.tanh(x)


def softmax(vals: list[float]) -> list[float]:
    """Softmax по списку значений."""
    max_v = max(vals)
    exps = [math.exp(v - max_v) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    """Умножение матрицы на вектор."""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    """Поэлементное сложение."""
    return [a[i] + b[i] for i in range(len(a))]


def make_matrix(rows: int, cols: int, scale: float = 0.5) -> list[list[float]]:
    """Создание матрицы со случайными весами."""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def make_vector(size: int, scale: float = 0.5) -> list[float]:
    """Создание вектора со случайными весами."""
    return [random.gauss(0, scale) for _ in range(size)]


def one_hot(idx: int, size: int) -> list[float]:
    """One-hot вектор."""
    v = [0.0] * size
    v[idx] = 1.0
    return v


def argmax(vals: list[float]) -> int:
    """Индекс максимума."""
    best = 0
    for i in range(1, len(vals)):
        if vals[i] > vals[best]:
            best = i
    return best


# ============================================================================
#  RNN-ЯЧЕЙКА (Building Block для Encoder/Decoder)
# ============================================================================

class RNNCell:
    """
    Простая RNN-ячейка:
        h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Веса вход→скрытое
        self.W_ih = make_matrix(hidden_size, input_size, 1.0 / math.sqrt(input_size))
        # Веса скрытое→скрытое
        self.W_hh = make_matrix(hidden_size, hidden_size, 1.0 / math.sqrt(hidden_size))
        self.b_h = [0.0] * hidden_size

    def forward(self, x: list[float], h_prev: list[float]) -> list[float]:
        """Один шаг RNN."""
        ih = mat_vec_mul(self.W_ih, x)
        hh = mat_vec_mul(self.W_hh, h_prev)
        h_t = [tanh(ih[i] + hh[i] + self.b_h[i]) for i in range(self.hidden_size)]
        return h_t

    def init_hidden(self) -> list[float]:
        """Начальный скрытый 상태."""
        return [0.0] * self.hidden_size


# ============================================================================
#  ENCODER
# ============================================================================

class Encoder:
    """
    Encoder: обрабатывает входную последовательность и возвращает
    контекстный вектор — финальное скрытое состояние RNN.

    Архитектура:
        x_1, x_2, ..., x_T  →  [RNN]  →  h_T (= context vector)
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding-матрица: vocab_size × embed_size
        self.embedding = make_matrix(vocab_size, embed_size, 1.0 / math.sqrt(embed_size))

        # RNN-ячейка
        self.cell = RNNCell(embed_size, hidden_size)

    def embed(self, token_idx: int) -> list[float]:
        """Получить вложение токена."""
        return list(self.embedding[token_idx])

    def encode(self, sequence: list[int]) -> tuple[list[float], list[list[float]]]:
        """
        Encode входную последовательность.

        Возвращает:
            context: финальное скрытое состояние (context vector)
            all_hidden: все промежуточные скрытые состояния (для Attention, если нужно)
        """
        h = self.cell.init_hidden()
        all_hidden = [list(h)]

        for token_idx in sequence:
            x_emb = self.embed(token_idx)
            h = self.cell.forward(x_emb, h)
            all_hidden.append(list(h))

        context = h
        return context, all_hidden


# ============================================================================
#  DECODER
# ============================================================================

class Decoder:
    """
    Decoder: генерирует выходную последовательность по контекстному вектору.

    Архитектура:
        h_0 = context
        y_t = Softmax(W_out * h_t)
        h_{t+1} = RNN(x_t, h_t)
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding-матрица
        self.embedding = make_matrix(vocab_size, embed_size, 1.0 / math.sqrt(embed_size))

        # RNN-ячейка
        self.cell = RNNCell(embed_size, hidden_size)

        # Выходная проекция: hidden_size → vocab_size
        self.W_out = make_matrix(vocab_size, hidden_size, 1.0 / math.sqrt(hidden_size))
        self.b_out = [0.0] * vocab_size

        # Токен начала последовательности (<sos>)
        self.sos_token = 0

    def embed(self, token_idx: int) -> list[float]:
        """Получить вложение токена."""
        return list(self.embedding[token_idx])

    def project(self, h: list[float]) -> list[float]:
        """Проекция скрытого состояния на словарь."""
        logits = mat_vec_mul(self.W_out, h)
        logits = [logits[i] + self.b_out[i] for i in range(self.vocab_size)]
        return softmax(logits)

    def decode_step(self, token_idx: int, h_prev: list[float]) -> tuple[list[float], list[float]]:
        """Один шаг декодирования."""
        x_emb = self.embed(token_idx)
        h_new = self.cell.forward(x_emb, h_prev)
        probs = self.project(h_new)
        return probs, h_new

    def generate_greedy(self, context: list[float], max_len: int = 20) -> tuple[list[int], list[list[float]]]:
        """Жадная генерация (Greedy Decoding)."""
        h = list(context)
        token = self.sos_token
        output = []
        probs_history = []

        for _ in range(max_len):
            probs, h = self.decode_step(token, h)
            probs_history.append(probs)
            token = argmax(probs)
            if token == 1:  # <eos>
                break
            output.append(token)

        return output, probs_history

    def decode_with_teacher(self, context: list[float], target: list[int],
                            max_len: int = 20) -> tuple[list[int], list[list[float]]]:
        """
        Декодирование с Teacher Forcing.
        На каждом шаге подаём правильный токен из target вместо предсказанного.
        """
        h = list(context)
        token = self.sos_token
        output = []
        probs_history = []

        for t in range(min(len(target), max_len)):
            probs, h = self.decode_step(token, h)
            probs_history.append(probs)
            predicted = argmax(probs)
            output.append(predicted)
            # Teacher forcing: подаём реальный токен из target
            token = target[t]

        return output, probs_history


# ============================================================================
#  SEQ2Seq МОДЕЛЬ
# ============================================================================

class Seq2Seq:
    """
    Seq2Seq = Encoder + Decoder.

    Процесс:
        1. Encoder читает входную последовательность → context vector
        2. Decoder генерирует выходную последовательность по context
    """

    def __init__(self, src_vocab: int, tgt_vocab: int, embed_size: int, hidden_size: int):
        self.encoder = Encoder(src_vocab, embed_size, hidden_size)
        self.decoder = Decoder(tgt_vocab, embed_size, hidden_size)

    def forward_greedy(self, src: list[int], max_len: int = 20) -> tuple[list[int], list[float]]:
        """Полный forward pass с greedy decoding."""
        context, _ = self.encoder.encode(src)
        output, probs = self.decoder.generate_greedy(context, max_len)

        # Средняя вероятность по шагам
        avg_probs = []
        for p in probs:
            max_p = max(p)
            avg_probs.append(max_p)

        return output, avg_probs

    def forward_teacher(self, src: list[int], tgt: list[int],
                        max_len: int = 20) -> tuple[list[int], list[float]]:
        """Полный forward pass с teacher forcing."""
        context, _ = self.encoder.encode(src)
        output, probs = self.decoder.decode_with_teacher(context, tgt, max_len)

        avg_probs = []
        for p in probs:
            max_p = max(p)
            avg_probs.append(max_p)

        return output, avg_probs


# ============================================================================
#  BEAM SEARCH
# ============================================================================

class BeamState:
    """Состояние в Beam Search."""

    def __init__(self, sequence: list[int], log_prob: float, hidden: list[float]):
        self.sequence = sequence
        self.log_prob = log_prob
        self.hidden = hidden

    def __lt__(self, other):
        return self.log_prob < other.log_prob


def beam_search(decoder: Decoder, context: list[float], beam_width: int = 3,
                max_len: int = 15) -> list[list[int]]:
    """
    Beam Search Decoding.

    На каждом шаге расширяем top-k кандидатов и выбираем top-k по log-probability.
    Возвращает список (sequence, log_prob), отсортированный по log_prob.

    Args:
        decoder: обученный декодер
        context: контекстный вектор от encoder
        beam_width: ширина луча (k)
        max_len: максимальная длина генерации

    Returns:
        Список кандидатов [(tokens, log_prob)] в порядке убывания
    """
    # Начальное состояние: <sos>
    h0 = list(context)
    init_token = decoder.sos_token

    # Начальные кандидаты
    probs0, h0_new = decoder.decode_step(init_token, h0)
    log_probs0 = [math.log(max(p, 1e-10)) for p in probs0]

    # Берём top-k стартовых токенов (исключая <sos>=0 и <eos>=1)
    candidates = []
    for token_idx in range(decoder.vocab_size):
        if token_idx == decoder.sos_token or token_idx == 1:
            continue
        candidates.append(BeamState(
            sequence=[token_idx],
            log_prob=log_probs0[token_idx],
            hidden=list(h0_new)
        ))

    # Сортируем и оставляем top-k
    candidates.sort(reverse=True)
    candidates = candidates[:beam_width]

    completed = []

    for step in range(max_len):
        all_new = []

        for beam in candidates:
            # Если последний токен — <eos>, не расширяем
            if beam.sequence[-1] == 1:
                completed.append(beam)
                continue

            # Расширяем этот луч
            probs, h_new = decoder.decode_step(beam.sequence[-1], beam.hidden)
            log_probs = [math.log(max(p, 1e-10)) for p in probs]

            for token_idx in range(decoder.vocab_size):
                if token_idx == decoder.sos_token:
                    continue

                new_seq = beam.sequence + [token_idx]
                new_log_prob = beam.log_prob + log_probs[token_idx]

                all_new.append(BeamState(
                    sequence=new_seq,
                    log_prob=new_log_prob,
                    hidden=list(h_new)
                ))

        if not all_new:
            break

        # Сортируем и оставляем top-k
        all_new.sort(reverse=True)
        candidates = all_new[:beam_width]

        # Добавляем завершённые
        candidates_completed = [c for c in candidates if c.sequence[-1] == 1]
        completed.extend(candidates_completed)

    # Добавляем оставшиеся кандидаты
    completed.extend(candidates)

    # Сортируем по log-prob
    completed.sort(reverse=True)

    # Нормализуем по длине для честного сравнения
    results = []
    for state in completed:
        seq_len = len(state.sequence)
        normalized_log_prob = state.log_prob / max(seq_len, 1)
        results.append((state.sequence, normalized_log_prob))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============================================================================
#  ДЕМОНСТРАЦИИ
# ============================================================================

def demo_1_encoder():
    """Демо 1: Encoder — сжатие последовательности в контекстный вектор."""
    print("=" * 70)
    print("ДЕМО 1: ENCODER — СЖАТИЕ ПОСЛЕДОВАТЕЛЬНОСТИ")
    print("=" * 70)

    # Словарь: <sos>=0, <eos>=1, русские буквы как токены
    vocab_tokens = ["<sos>", "<eos>"] + list("приветмир")
    vocab_size = len(vocab_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab_tokens)}

    print(f"\nСловарь ({vocab_size} токенов):")
    for t, i in token_to_idx.items():
        print(f"  {t!r:>10} → {i}")

    # Создаём encoder
    random.seed(42)
    embed_size = 8
    hidden_size = 16
    encoder = Encoder(vocab_size, embed_size, hidden_size)

    # Входная последовательность: "привет"
    input_text = "привет"
    input_seq = [token_to_idx[ch] for ch in input_text]

    print(f"\nВходная последовательность: '{input_text}'")
    print(f"Токены: {input_seq}")

    context, all_hidden = encoder.encode(input_seq)

    print(f"\nРазмерность контекстного вектора: {len(context)}")
    print(f"\nКонтекстный вектор (финальное скрытое состояние):")
    for i, v in enumerate(context):
        print(f"  h[{i:2d}] = {v:+.4f}")

    print(f"\nЭнтропия контекстного вектора (мера разнообразия):")
    abs_vals = [abs(v) for v in context]
    total = sum(abs_vals)
    entropy = 0.0
    for v in abs_vals:
        p = v / total if total > 0 else 0
        if p > 0:
            entropy -= p * math.log2(p)
    print(f"  {entropy:.4f} бит")

    print(f"\nПромежуточные скрытые состояния ({len(all_hidden)} шагов):")
    for step, h in enumerate(all_hidden):
        print(f"  Шаг {step}: mean={sum(h)/len(h):+.4f}, "
              f"std={(sum((x-sum(h)/len(h))**2 for x in h)/len(h))**0.5:.4f}")

    print("\n[OK] Encoder успешно сжал последовательность в контекстный вектор\n")


def demo_2_decoder():
    """Демо 2: Decoder — генерация последовательности по контексту."""
    print("=" * 70)
    print("ДЕМО 2: DECODER — ГЕНЕРАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТИ")
    print("=" * 70)

    vocab_tokens = ["<sos>", "<eos>"] + list("приветмир")
    vocab_size = len(vocab_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab_tokens)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    random.seed(42)
    embed_size = 8
    hidden_size = 16
    decoder = Decoder(vocab_size, embed_size, hidden_size)

    # Случайный контекстный вектор
    random.seed(42)
    context = make_vector(hidden_size, 0.3)

    print(f"\nСловарь декодера: {vocab_tokens}")
    print(f"Контекстный вектор (сгенерирован случайно для демонстрации):")
    for i, v in enumerate(context):
        print(f"  [{i:2d}] = {v:+.4f}")

    print("\n--- Жадная генерация (Greedy Decoding) ---")
    output, probs = decoder.generate_greedy(context, max_len=10)
    output_text = "".join(idx_to_token.get(t, "?") for t in output)

    print(f"Сгенерированная последовательность: {output}")
    print(f"Текст: '{output_text}'")
    print(f"Длина: {len(output)} токенов")
    # probs — список векторов вероятностей, берём max на каждом шаге
    step_confidences = [max(p) for p in probs]
    print(f"Средняя уверенность: {sum(step_confidences)/len(step_confidences):.4f}")

    print("\n--- Детализация по шагам ---")
    for step, p in enumerate(probs):
        top3 = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:3]
        top3_str = ", ".join(f"{idx_to_token[i]}={p[i]:.3f}" for i in top3)
        predicted_token = idx_to_token[output[step]] if step < len(output) else "<done>"
        print(f"  Шаг {step}: [{top3_str}]  →  предсказано: '{predicted_token}'")

    # Попробуем с другим контекстом
    print("\n--- Разные контексты → разные выходы ---")
    for seed_val in [1, 7, 13, 42, 99]:
        random.seed(seed_val)
        ctx = make_vector(hidden_size, 0.3)
        out, _ = decoder.generate_greedy(ctx, max_len=8)
        text = "".join(idx_to_token.get(t, "?") for t in out)
        print(f"  seed={seed_val:3d} → '{text}'")

    print("\n[OK] Decoder генерирует последовательности по контекстному вектору\n")


def demo_3_teacher_forcing():
    """Демо 3: Teacher Forcing vs Greedy Decoding."""
    print("=" * 70)
    print("ДЕМО 3: TEACHER FORCING vs GREEDY DECODING")
    print("=" * 70)

    vocab_tokens = ["<sos>", "<eos>"] + list("приветмир")
    vocab_size = len(vocab_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab_tokens)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    random.seed(42)
    embed_size = 8
    hidden_size = 16
    model = Seq2Seq(vocab_size, vocab_size, embed_size, hidden_size)

    # Входная последовательность
    src_text = "привет"
    src_seq = [token_to_idx[ch] for ch in src_text]

    # Целевая последовательность
    tgt_text = "мир"
    tgt_seq = [token_to_idx[ch] for ch in tgt_text]

    print(f"\nВход:  '{src_text}' → {src_seq}")
    print(f"Цель:  '{tgt_text}' → {tgt_seq}")
    print(f"\nПримечание: модель случайная (не обучена),")
    print(f"поэтому качество не важно — мы сравниваем стратегии.\n")

    # Greedy decoding
    print("--- Greedy Decoding ---")
    greedy_out, greedy_probs = model.forward_greedy(src_seq, max_len=10)
    greedy_text = "".join(idx_to_token.get(t, "?") for t in greedy_out)
    print(f"Выход: {greedy_out}")
    print(f"Текст: '{greedy_text}'")
    print(f"Средняя уверенность: {sum(greedy_probs)/max(len(greedy_probs),1):.4f}")
    print(f"Точность (совпадение с целью): "
          f"{sum(1 for a,b in zip(greedy_out, tgt_seq) if a==b)/max(len(tgt_seq),1)*100:.1f}%")

    # Teacher forcing
    print("\n--- Teacher Forcing ---")
    tf_out, tf_probs = model.forward_teacher(src_seq, tgt_seq, max_len=10)
    tf_text = "".join(idx_to_token.get(t, "?") for t in tf_out)
    print(f"Выход: {tf_out}")
    print(f"Текст: '{tf_text}'")
    print(f"Средняя уверенность: {sum(tf_probs)/max(len(tf_probs),1):.4f}")
    print(f"Точность (совпадение с целью): "
          f"{sum(1 for a,b in zip(tf_out, tgt_seq) if a==b)/max(len(tgt_seq),1)*100:.1f}%")

    # Сравнение по шагам
    print("\n--- Пошаговое сравнение ---")
    print(f"{'Шаг':>4} | {'Greedy':>15} | {'Teacher Forcing':>15} | {'Цель':>10}")
    print("-" * 55)
    for step in range(min(len(greedy_out), len(tf_out), len(tgt_seq))):
        g_tok = idx_to_token[greedy_out[step]]
        t_tok = idx_to_token[tf_out[step]]
        tgt_tok = idx_to_token[tgt_seq[step]]
        match_g = "✓" if greedy_out[step] == tgt_seq[step] else "✗"
        match_t = "✓" if tf_out[step] == tgt_seq[step] else "✗"
        print(f"  {step:2d}  | {g_tok:>6} {match_g:>7} | {t_tok:>6} {match_t:>7}  | {tgt_tok:>5}")

    # Почему Teacher Forcing лучше
    print("\n--- Ключевые различия ---")
    print("  Greedy Decoding:")
    print("    - На каждом шаге использует собственное предыдущее предсказание")
    print("    - Ошибки накапливаются (error propagation)")
    print("    - Используется при инференсе (inference)")
    print()
    print("  Teacher Forcing:")
    print("    - На каждом шаге подаёт правильный токен из target")
    print("    - Предотвращает каскад ошибок")
    print("    - Используется при обучении (training)")

    print("\n[OK] Сравнение Teacher Forcing и Greedy Decoding завершено\n")


def demo_4_beam_search():
    """Демо 4: Beam Search Decoding."""
    print("=" * 70)
    print("ДЕМО 4: BEAM SEARCH DECODING")
    print("=" * 70)

    vocab_tokens = ["<sos>", "<eos>"] + list("приветмир")
    vocab_size = len(vocab_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab_tokens)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    random.seed(42)
    embed_size = 8
    hidden_size = 16
    decoder = Decoder(vocab_size, embed_size, hidden_size)

    # Контекстный вектор
    random.seed(42)
    context = make_vector(hidden_size, 0.3)

    print(f"\nСловарь: {vocab_tokens}")
    print(f"Контекстный вектор зафиксирован (seed=42)\n")

    # Разная ширина луча
    for beam_width in [1, 2, 3, 5]:
        print(f"--- Beam Search (k={beam_width}) ---")
        results = beam_search(decoder, context, beam_width=beam_width, max_len=8)

        print(f"  Найдено кандидатов: {len(results)}")
        for rank, (seq, score) in enumerate(results[:5]):
            text = "".join(idx_to_token.get(t, "?") for t in seq)
            seq_str = " ".join(idx_to_token.get(t, "?") for t in seq)
            print(f"  Ранг {rank+1}: score={score:+.4f}  seq=[{seq_str}]  text='{text}'")
        print()

    # Сравнение Greedy vs Beam Search
    print("--- Сравнение: Greedy (k=1) vs Beam Search (k=5) ---")
    print()

    # Greedy
    output_greedy, probs_greedy = decoder.generate_greedy(context, max_len=8)
    text_greedy = "".join(idx_to_token.get(t, "?") for t in output_greedy)
    greedy_conf = [max(p) for p in probs_greedy]
    score_greedy = sum(math.log(max(c, 1e-10)) for c in greedy_conf) / max(len(greedy_conf), 1)
    print(f"  Greedy:  '{text_greedy}' (score={score_greedy:+.4f})")

    # Beam
    results_beam = beam_search(decoder, context, beam_width=5, max_len=8)
    if results_beam:
        best_seq, best_score = results_beam[0]
        text_beam = "".join(idx_to_token.get(t, "?") for t in best_seq)
        print(f"  Beam k=5: '{text_beam}' (score={best_score:+.4f})")

    # Анализ разных beam width
    print("\n--- Влияние ширины луча на количество кандидатов ---")
    print(f"{'k':>4} | {'Кандидатов':>10} | {'Лучший score':>12}")
    print("-" * 35)
    for k in [1, 2, 3, 5, 8, 10]:
        results = beam_search(decoder, context, beam_width=k, max_len=6)
        if results:
            best_score = results[0][1]
            print(f"  {k:2d} | {len(results):>10} | {best_score:>+12.4f}")
        else:
            print(f"  {k:2d} | {'0':>10} | {'N/A':>12}")

    # Пример поиска лучшего варианта
    print("\n--- Топ-3 кандидата при Beam Search (k=5) ---")
    results = beam_search(decoder, context, beam_width=5, max_len=8)
    for rank, (seq, score) in enumerate(results[:3]):
        text = "".join(idx_to_token.get(t, "?") for t in seq)
        print(f"  {rank+1}. '{text}' (score={score:+.4f}, длина={len(seq)})")

    print("\n--- Принцип работы Beam Search ---")
    print("  1. Начинаем с <sos> и вычисляем P(next_token) для всех токенов")
    print("  2. Выбираем top-k наиболее вероятных токенов")
    print("  3. Для каждого кандидата вычисляем P(next_token | текущий луч)")
    print("  4. Снова выбираем top-k среди ВСЕХ расширений")
    print("  5. Повторяем, пока не встретим <eos> или не достигнем max_len")
    print("  6. Возвращаем k лучших последовательностей")

    print("\n[OK] Beam Search демонстрация завершена\n")


# ============================================================================
#  ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       SEQUENCE-TO-SEQUENCE МОДЕЛИ (Seq2Seq)                        ║")
    print("║       Encoder-Decoder | Teacher Forcing | Beam Search               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    demo_1_encoder()
    print()

    demo_2_decoder()
    print()

    demo_3_teacher_forcing()
    print()

    demo_4_beam_search()
    print()

    print("=" * 70)
    print("ИТОГОВОЕ РЕЗЮМЕ")
    print("=" * 70)
    print("""
Архитектура Seq2Seq:
┌─────────────┐     context vector     ┌─────────────┐
│   Encoder   │ ─────────────────────→  │   Decoder   │
│  (RNN/GRU)  │                         │  (RNN/GRU)  │
└──────┬──────┘                         └──────┬──────┘
       │                                       │
  x_1, x_2, ..., x_T                    y_1, y_2, ..., y_S

Encoder:
  - Обрабатывает входную последовательность токенов
  - RNN обновляет скрытое состояние на каждом шаге
  - Финальное скрытое состояние = контекстный вектор

Decoder:
  - Генерирует выходную последовательность по контексту
  - На каждом шаге: embedding → RNN → softmax → токен

Teacher Forcing:
  - При обучении: на каждом шаге подаём правильный токен
  - Предотвращает каскад ошибок (error propagation)
  - При инференсе: используем Greedy или Beam Search

Beam Search:
  - Расширяет k наиболее вероятных кандидатов на каждом шаге
  - k=1 эквивалентен Greedy Decoding
  - k>1 позволяет найти более вероятную последовательность
  - Нормализация по длине для честного сравнения
""")


if __name__ == "__main__":
    main()
