"""
110 — LLM Evaluation: основы оценки качества языковых моделей

Темы:
  1. Perplexity — измерение неуверенности модели
  2. BLEU score — качество машинного перевода
  3. Human evaluation metrics — faithfulness, relevance, coherence
  4. Automated benchmarks — MMLU, HellaSwag, ARC, TruthfulQA

Самодостаточный файл — не требует numpy, torch, transformers, nltk, rouge-score.
"""

import math
import random
import re
from collections import Counter
from typing import Dict, List, Tuple


random.seed(42)


# ============================================================================
# 1. PERPLEXITY
# ============================================================================

def softmax(logits: List[float]) -> List[float]:
    """Вычисление softmax по списку логитов."""
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    return [e / total for e in exp_logits]


def log_softmax(logits: List[float]) -> List[float]:
    """Log-softmax по списку логитов."""
    s = softmax(logits)
    return [math.log(p + 1e-12) for p in s]


def cross_entropy(probs: List[float], target_idx: int) -> float:
    """Перекрёстная энтропия для одного токена."""
    return -math.log(probs[target_idx] + 1e-12)


def perplexity_from_probs(token_probs: List[float]) -> float:
    """
    Perplexity = exp(средняя negative log-probability).
    token_probs — список вероятностей истинного токена на каждом шаге.
    """
    if not token_probs:
        return 0.0
    log_probs = [math.log(p + 1e-12) for p in token_probs]
    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_log_prob)


def perplexity_from_logits(
    all_logits: List[List[float]],
    targets: List[int]
) -> float:
    """
    Perplexity из сырых логитов и целевых индексов токенов.
    all_logits[i] — вектор логитов для i-го токена.
    targets[i]    — индекс истинного токена.
    """
    total_ce = 0.0
    for logits, tgt in zip(all_logits, targets):
        probs = softmax(logits)
        total_ce += cross_entropy(probs, tgt)
    avg_ce = total_ce / len(targets)
    return math.exp(avg_ce)


def demo_perplexity():
    """Демо 1: Perplexity — измерение неуверенности модели."""
    print("=" * 70)
    print("DEMO 1: Perplexity — измерение неуверенности модели")
    print("=" * 70)

    vocab = ["the", "cat", "sat", "on", "mat", "dog", "happy", "ran", "big", "is"]
    vocab_size = len(vocab)

    # --- Простой пример: две假想 модели ---
    # Модель A: уверена в правильных токенах
    # Модель B: неуверена — распределение равномерное

    print("\n--- Формула ---")
    print("  Perplexity = exp( -1/N * Σ log P(token_i | context) )")
    print("  Низкая perplexity  → модель уверена, предсказывает хорошо")
    print("  Высокая perplexity → модель \" удивлена \" текстом")

    # Пример 1: Простая вероятность
    print("\n--- Пример 1: Одиночное предсказание ---")
    token_probs = [0.9, 0.05, 0.03, 0.01, 0.01]
    ppl = perplexity_from_probs(token_probs)
    print(f"  Вероятности токенов: {token_probs}")
    print(f"  Perplexity = {ppl:.4f}")
    print(f"  Интерпретация: модель \" удивлена \" в среднем ~{ppl:.1f}-мя вариантами")

    # Пример 2: Сравнение двух моделей
    print("\n--- Пример 2: Сравнение моделей ---")
    random.seed(42)

    targets = [random.randint(0, vocab_size - 1) for _ in range(8)]
    target_words = [vocab[t] for t in targets]
    print(f"  Целевая последовательность: {' '.join(target_words)}")

    # Модель A: распределение с пиком на правильном токене
    logits_a = []
    for tgt in targets:
        row = [random.uniform(-2, 0) for _ in range(vocab_size)]
        row[tgt] = random.uniform(2, 4)  # правильный токен имеет высокий логит
        logits_a.append(row)

    # Модель B: примерно равномерное распределение
    logits_b = []
    for _ in range(len(targets)):
        row = [random.uniform(-0.5, 0.5) for _ in range(vocab_size)]
        logits_b.append(row)

    ppl_a = perplexity_from_logits(logits_a, targets)
    ppl_b = perplexity_from_logits(logits_b, targets)

    print(f"\n  Модель A (уверенная): perplexity = {ppl_a:.4f}")
    print(f"  Модель B (равномерная): perplexity = {ppl_b:.4f}")
    print(f"  → Модель A лучше: perplexity в {ppl_b / ppl_a:.1f}x ниже")

    # Пример 3: Perplexity по шагам
    print("\n--- Пример 3: Perplexity по шагам ---")
    for i in range(len(targets)):
        prob_a = softmax(logits_a[i])[targets[i]]
        prob_b = softmax(logits_b[i])[targets[i]]
        print(
            f"  Шаг {i}: '{vocab[targets[i]]}' | "
            f"P(A)={prob_a:.4f} P(B)={prob_b:.4f} | "
            f"CE(A)={-math.log(prob_a + 1e-12):.4f} CE(B)={-math.log(prob_b + 1e-12):.4f}"
        )

    # Пример 4: Связь perplexity с cross-entropy
    print("\n--- Пример 4: Связь Perplexity и Cross-Entropy ---")
    for ppl_val in [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        ce = math.log(ppl_val)
        bits_per_token = ce / math.log(2)
        print(
            f"  PPL={ppl_val:>6.1f} → CE={ce:.4f} → "
            f"bits/token={bits_per_token:.2f}"
        )
    print("  Примечание: perplexity = 2^(bits_per_token)")

    print()


# ============================================================================
# 2. BLEU SCORE
# ============================================================================

def compute_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Извлечение n-грамм из списка токенов."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def clipped_count(
    candidate_ngrams: List[Tuple[str, ...]],
    reference_ngrams: Counter
) -> int:
    """Подсчёт clipped n-грамм (не больше, чем в референсе)."""
    candidate_counts = Counter(candidate_ngrams)
    clip = 0
    for ngram, count in candidate_counts.items():
        clip += min(count, reference_ngrams.get(ngram, 0))
    return clip


def brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """Штраф за краткость (Brevity Penalty)."""
    if candidate_len >= reference_len:
        return 1.0
    if candidate_len == 0:
        return 0.0
    return math.exp(1 - reference_len / candidate_len)


def bleu_score(
    candidate: List[str],
    references: List[List[str]],
    max_n: int = 4,
    weights: List[float] = None
) -> Dict[str, float]:
    """
    Вычисление BLEU-счёта.

    candidate   — список токенов кандидата
    references  — список списков токенов референсов
    max_n       — максимальный порядок n-грамм (по умолчанию 4)
    weights     — веса для каждого n (по умолчанию равные)

    Возвращает словарь с общим BLEU и precision для каждого n.
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Длина лучшего референса (по близости к кандидату)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - len(candidate)), rl))

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = compute_ngrams(candidate, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue

        # Объединение n-грамм из всех референсов
        ref_combined = Counter()
        for ref in references:
            ref_combined.update(compute_ngrams(ref, n))

        clipped = clipped_count(cand_ngrams, ref_combined)
        precisions.append(clipped / len(cand_ngrams))

    # Геометрическое среднее precision (с log-sum-exp для численной стабильности)
    log_avg = 0.0
    for p, w in zip(precisions, weights):
        if p == 0:
            log_avg = -float('inf')
            break
        log_avg += w * math.log(p)

    bp = brevity_penalty(len(candidate), closest_ref_len)
    bleu = bp * math.exp(log_avg)

    result = {"bleu": bleu}
    for n, p in enumerate(precisions, 1):
        result[f"precision_{n}"] = p
    result["brevity_penalty"] = bp
    result["candidate_len"] = len(candidate)
    result["closest_ref_len"] = closest_ref_len

    return result


def demo_bleu():
    """Демо 2: BLEU score — качество машинного перевода."""
    print("=" * 70)
    print("DEMO 2: BLEU Score — качество машинного перевода")
    print("=" * 70)

    print("\n--- Формула ---")
    print("  BLEU = BP × exp( Σ wₙ × log(precisionₙ) )")
    print("  BP = Brevity Penalty (штраф за слишком короткий перевод)")
    print("  precisionₙ = доля n-грамм кандидата, встречающихся в референсе")
    print("  По умолчанию: n = 1..4, веса равные (0.25)")

    # Пример 1: Разные кандидаты для одного референса
    print("\n--- Пример 1: Разные кандидаты ---")
    reference = "the cat is sitting on the mat".split()
    candidates = [
        "the cat is sitting on the mat".split(),       # идеальный
        "the cat sat on the mat".split(),              # хорошее
        "the the the the the the".split(),             # повтор
        "a cat on a mat".split(),                      # короткое
        "the dog is running in the park".split(),      # неправильное
    ]
    labels = [
        " идеальный кандидат",
        " хорошее (другая форма)",
        " повтор (bad precision)",
        " короткое (штраф BP)",
        " неправильное",
    ]

    for cand, label in zip(candidates, labels):
        result = bleu_score(cand, [reference])
        print(
            f"  {label:40s} | "
            f"BLEU={result['bleu']:.4f} "
            f"p1={result['precision_1']:.3f} "
            f"p4={result['precision_4']:.3f} "
            f"BP={result['brevity_penalty']:.3f}"
        )

    # Пример 2: Влияние числа референсов
    print("\n--- Пример 2: Несколько референсов ---")
    references_list = [
        "the cat is sitting on the mat".split(),
        "the cat sits on the mat".split(),
        "a cat is on the mat".split(),
    ]
    candidate = "the cat is on the mat".split()

    for n_refs in [1, 2, 3]:
        result = bleu_score(candidate, references_list[:n_refs])
        print(
            f"  Референсов={n_refs} | "
            f"BLEU={result['bleu']:.4f} "
            f"p1={result['precision_1']:.3f} "
            f"p4={result['precision_4']:.3f}"
        )
    print("  → Больше референсов обычно даёт более справедливую оценку")

    # Пример 3: BLEU для разных языков
    print("\n--- Пример 3: Перевод с русского на английский ---")
    ru_en_pairs = [
        (
            "машинное обучение это Subset of artificial intelligence",
            ["machine learning is a subset of artificial intelligence"],
            "точный перевод",
        ),
        (
            "машинное обучение это Subset of artificial intelligence",
            ["machine learning is part of artificial intelligence"],
            "хороший перевод",
        ),
        (
            "машинное обучение это Subset of artificial intelligence",
            ["AI includes machine learning methods"],
            "свободный перевод",
        ),
    ]
    candidate_ru = "машинное обучение это Subset of artificial intelligence".split()
    for cand_str, ref_strs, desc in ru_en_pairs:
        cand_tokens = cand_str.split()
        ref_tokens = [r.split() for r in ref_strs]
        result = bleu_score(cand_tokens, ref_tokens)
        print(f"  {desc:20s} | BLEU={result['bleu']:.4f}")

    # Пример 4: Пороговые значения BLEU
    print("\n--- Пример 4: Эталонные значения BLEU ---")
    print("  BLEU = 0.0  — нет совпадений n-грамм")
    print("  BLEU = 0.3  — минимально приемлемо")
    print("  BLEU = 0.5  — хороший перевод")
    print("  BLEU = 0.7  — очень高质量 перевод")
    print("  BLEU = 1.0  — точное совпадение с референсом")
    print("  Примечание: BLEU > 0.6 редко встретится для естественного языка")

    print()


# ============================================================================
# 3. HUMAN EVALUATION METRICS
# ============================================================================

def normalize_text(text: str) -> str:
    """Нормализация текста для сравнения."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def token_overlap(hypothesis: str, reference: str) -> float:
    """Jaccard similarity по токенам."""
    h_tokens = set(normalize_text(hypothesis).split())
    r_tokens = set(normalize_text(reference).split())
    if not h_tokens or not r_tokens:
        return 0.0
    intersection = h_tokens & r_tokens
    union = h_tokens | r_tokens
    return len(intersection) / len(union)


def faithfulness_score(
    generated: str,
    source: str,
    knowledge_base: List[str] = None
) -> Dict[str, float]:
    """
    Faithfulness — насколько ответ соответствует исходному тексту/знанию.

    Метрики:
    - factual_overlap: доля токенов из generated, найденных в source
    - source_coverage: доля токенов из source, покрытых generated
    - hallucination_rate: доля \"выдуманных\" токенов
    """
    gen_tokens = set(normalize_text(generated).split())
    src_tokens = set(normalize_text(source).split())

    if not gen_tokens:
        return {
            "factual_overlap": 0.0,
            "source_coverage": 0.0,
            "hallucination_rate": 1.0,
        }

    supported = gen_tokens & src_tokens
    hallucinated = gen_tokens - src_tokens

    factual_overlap = len(supported) / len(gen_tokens) if gen_tokens else 0.0
    source_coverage = len(supported) / len(src_tokens) if src_tokens else 0.0
    hallucination_rate = len(hallucinated) / len(gen_tokens) if gen_tokens else 0.0

    return {
        "factual_overlap": factual_overlap,
        "source_coverage": source_coverage,
        "hallucination_rate": hallucination_rate,
    }


def relevance_score(
    generated: str,
    query: str,
    keywords: List[str] = None
) -> Dict[str, float]:
    """
    Relevance — насколько ответ релевантен запросу.

    Метрики:
    - query_overlap: перекрытие токенов запроса и ответа
    - keyword_coverage: покрытие ключевых слов
    - length_ratio: отношение длин (контроль краткости/многословности)
    """
    gen_tokens = set(normalize_text(generated).split())
    query_tokens = set(normalize_text(query).split())

    query_overlap = token_overlap(generated, query)

    if keywords:
        all_kw_tokens = set()
        for kw in keywords:
            all_kw_tokens.update(normalize_text(kw).split())
        covered = gen_tokens & all_kw_tokens
        keyword_coverage = len(covered) / len(all_kw_tokens) if all_kw_tokens else 0.0
    else:
        keyword_coverage = 0.0

    gen_len = len(normalize_text(generated).split())
    query_len = len(normalize_text(query).split())
    length_ratio = gen_len / query_len if query_len > 0 else 0.0

    return {
        "query_overlap": query_overlap,
        "keyword_coverage": keyword_coverage,
        "length_ratio": length_ratio,
    }


def coherence_score(text: str) -> Dict[str, float]:
    """
    Coherence — связность текста (на основе простых эвристик).

    Метрики:
    - sentence_count: количество предложений
    - avg_sentence_length: средняя длина предложения
    - connective_density: плотность связующих слов
    - repetition_rate: доля повторяющихся слов
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = normalize_text(text).split()

    connectives = {
        "however", "therefore", "moreover", "furthermore", "additionally",
        "consequently", "thus", "hence", "meanwhile", "nevertheless",
        "because", "although", "while", "since", "although",
    }
    connective_count = sum(1 for w in words if w in connectives)
    connective_density = connective_count / len(words) if words else 0.0

    word_counts = Counter(words)
    repeated = sum(c - 1 for c in word_counts.values() if c > 1)
    repetition_rate = repeated / len(words) if words else 0.0

    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        "connective_density": connective_density,
        "repetition_rate": repetition_rate,
    }


def demo_human_eval():
    """Демо 3: Faithfulness и Relevance — метрики human evaluation."""
    print("=" * 70)
    print("DEMO 3: Faithfulness и Relevance — метрики human evaluation")
    print("=" * 70)

    # --- Faithfulness ---
    print("\n--- Faithfulness (соответствие исходному тексту) ---")
    source = (
        "Python created by Guido van Rossum released in 1991 "
        "interpreted language known for readability used in web development "
        "data science and machine learning"
    )

    generated_good = (
        "Python was created by Guido van Rossum and released in 1991. "
        "It is an interpreted language known for its readability and is "
        "widely used in web development and data science."
    )
    generated_halluc = (
        "Python was created by James Gosling and released in 1985. "
        "It is a compiled language known for its speed and is primarily "
        "used in enterprise software and mobile development."
    )
    generated_partial = (
        "Python is a popular programming language. "
        "It was created in the 1990s and is used for many things."
    )

    print(f"\n  Источник: \"{source[:70]}...\"")

    for gen, label in [
        (generated_good, "Хороший ответ"),
        (generated_halluc, "С галлюцинациями"),
        (generated_partial, "Частичный ответ"),
    ]:
        faith = faithfulness_score(gen, source)
        print(f"\n  {label}:")
        print(f"    Ответ: \"{gen[:70]}...\"")
        print(f"    Factual Overlap:  {faith['factual_overlap']:.3f}")
        print(f"    Source Coverage:  {faith['source_coverage']:.3f}")
        print(f"    Hallucination:   {faith['hallucination_rate']:.3f}")

    # --- Relevance ---
    print("\n--- Relevance (релевантность запросу) ---")
    query = "What are the main features of Python programming language?"
    keywords = ["python", "features", "programming", "language", "dynamic", "typing"]

    responses = [
        (
            "Python is a dynamic, interpreted programming language with features "
            "like automatic memory management, dynamic typing, and extensive libraries.",
            "Релевантный ответ",
        ),
        (
            "Java is a statically typed compiled language used in enterprise.",
            "Нерелевантный ответ",
        ),
        (
            "Python was created in the 1990s. It is used in many fields.",
            "Частично релевантный",
        ),
    ]

    print(f"\n  Запрос: \"{query}\"")
    print(f"  Ключевые слова: {keywords}")

    for resp, label in responses:
        rel = relevance_score(resp, query, keywords)
        print(f"\n  {label}:")
        print(f"    Ответ: \"{resp[:65]}...\"")
        print(f"    Query Overlap:     {rel['query_overlap']:.3f}")
        print(f"    Keyword Coverage:  {rel['keyword_coverage']:.3f}")
        print(f"    Length Ratio:      {rel['length_ratio']:.2f}")

    # --- Coherence ---
    print("\n--- Coherence (связность текста) ---")
    texts = [
        (
            "Machine learning is a subset of AI. However, not all AI is machine "
            "learning. Deep learning is a further subset. Therefore, the field "
            "is hierarchical. Moreover, each level has its own applications.",
            "Связный текст",
        ),
        (
            "cat dog the. Running is happy. Big the mat on sat. The is.",
            "Несвязный текст",
        ),
    ]

    for text, label in texts:
        coh = coherence_score(text)
        print(f"\n  {label}:")
        print(f"    Предложений: {coh['sentence_count']}")
        print(f"    Слов в предложении (ср.): {coh['avg_sentence_length']:.1f}")
        print(f"    Плотность связок: {coh['connective_density']:.3f}")
        print(f"    Коэф. повторений: {coh['repetition_rate']:.3f}")

    # --- Сводная таблица ---
    print("\n--- Типичные шкалы оценки ---")
    print("  Faithfulness: 0 (полные галлюцинации) → 1 (все факты подтверждены)")
    print("  Relevance:    0 (о теме) → 1 (точно отвечает на вопрос)")
    print("  Coherence:    0 (рассыпанный текст) → 1 (связный, логичный)")
    print("  Стандартная практика: 5-балльная шкала (1-5) для каждой метрики")

    print()


# ============================================================================
# 4. AUTOMATED BENCHMARKS
# ============================================================================

class SimpleBenchmark:
    """Простой фреймворк для автоматических бенчмарков."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.examples: List[Dict] = []
        self.results: List[Dict] = []

    def add_example(
        self,
        question: str,
        choices: List[str],
        correct_idx: int,
        category: str = "general"
    ):
        self.examples.append({
            "question": question,
            "choices": choices,
            "correct_idx": correct_idx,
            "category": category,
        })

    def evaluate(self, model_fn) -> Dict:
        """
        model_fn(question, choices) -> int (индекс выбранного ответа).
        """
        self.results = []
        correct = 0
        by_category: Dict[str, Dict] = {}

        for ex in self.examples:
            prediction = model_fn(ex["question"], ex["choices"])
            is_correct = prediction == ex["correct_idx"]
            if is_correct:
                correct += 1

            cat = ex["category"]
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if is_correct:
                by_category[cat]["correct"] += 1

            self.results.append({
                "question": ex["question"][:50],
                "predicted": prediction,
                "correct": ex["correct_idx"],
                "is_correct": is_correct,
                "category": cat,
            })

        accuracy = correct / len(self.examples) if self.examples else 0
        category_scores = {}
        for cat, stats in by_category.items():
            category_scores[cat] = stats["correct"] / stats["total"]

        return {
            "benchmark": self.name,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(self.examples),
            "categories": category_scores,
        }


def demo_benchmarks():
    """Демо 4: Автоматические бенчмарки (MMLU, HellaSwag, ARC, TruthfulQA)."""
    print("=" * 70)
    print("DEMO 4: Автоматические бенчмарки")
    print("=" * 70)

    print("\n--- Популярные бенчмарки для LLM ---")
    benchmarks_info = {
        "MMLU": "57 предметов (математика, история, право, ...) — 4-балльный выбор",
        "HellaSwag": "Завершение предложения — проверка \"модального здравого смысла\"",
        "ARC": "Научные вопросы — разделение на Easy / Challenge",
        "TruthfulQA": "817 вопросов — устойчивость к мифам и ложным стереотипам",
        "GSM8K": "Арифметические задачи с решением в несколько шагов",
        "HumanEval": "Генерация кода — 164 задачи на Python",
    }
    for name, desc in benchmarks_info.items():
        print(f"  {name:12s} — {desc}")

    # --- MMLU-style benchmark ---
    print("\n--- MMLU-стиль бенчмарк (общие знания) ---")
    mmlu = SimpleBenchmark("MMLU-mini", "Общие знания (5 предметов)")

    mmlu_examples = [
        ("What is the capital of France?",
         ["London", "Berlin", "Paris", "Madrid"], 2, "geography"),
        ("Which element has atomic number 1?",
         ["Helium", "Hydrogen", "Lithium", "Carbon"], 1, "science"),
        ("Who wrote '1984'?",
         ["Aldous Huxley", "George Orwell", "Ray Bradbury", "H.G. Wells"], 1, "literature"),
        ("What is the derivative of x^2?",
         ["x", "2x", "2x^2", "x^2"], 1, "math"),
        ("The Battle of Hastings took place in which year?",
         ["1066", "1215", "1492", "1776"], 0, "history"),
        ("What is the speed of light approximately?",
         ["300,000 km/s", "150,000 km/s", "300,000 m/s", "3,000 km/s"], 0, "science"),
        ("Which planet is known as the Red Planet?",
         ["Venus", "Mars", "Jupiter", "Saturn"], 1, "geography"),
        ("What is the square root of 144?",
         ["11", "12", "13", "14"], 1, "math"),
        ("Who painted the Mona Lisa?",
         ["Michelangelo", "Leonardo da Vinci", "Raphael", "Donatello"], 1, "art"),
        ("What is H2O commonly known as?",
         ["Hydrogen peroxide", "Water", "Salt", "Carbon dioxide"], 1, "science"),
    ]

    for q, choices, correct, cat in mmlu_examples:
        mmlu.add_example(q, choices, correct, cat)

    # --- HellaSwag-style benchmark ---
    print("\n--- HellaSwag-стиль бенчмарк (здравый смысл) ---")
    hella = SimpleBenchmark("HellaSwag-mini", "Завершение по здравому смыслу")

    hella_examples = [
        ("A person is cooking dinner. They pick up a knife and start chopping vegetables.",
         ["The vegetables are cooked on the stove.", "The vegetables are placed in a salad.",
          "The vegetables are thrown away.", "The vegetables are painted on a canvas."], 1, "everyday"),
        ("The dog ran towards the ball. The dog",
         ["picked up the ball in its mouth.", "read the newspaper carefully.",
          "started singing a song.", "began solving math problems."], 0, "everyday"),
        ("She opened the umbrella because",
         ["it was raining outside.", "the sun was too bright.",
          "she wanted to fly.", "the umbrella was broken."], 0, "causal"),
        ("The car stopped at the red light because",
         ["red lights indicate stopping.", "the driver was tired.",
          "the car ran out of gas.", "the road ended."], 0, "causal"),
    ]

    for q, choices, correct, cat in hella_examples:
        hella.add_example(q, choices, correct, cat)

    # --- Модели для демонстрации ---
    def random_model(question: str, choices: List[str]) -> int:
        """Случайная модель (baseline)."""
        return random.randint(0, len(choices) - 1)

    def rule_based_model(question: str, choices: List[str]) -> int:
        """Правиловая модель (чуть лучше случайной)."""
        q_lower = question.lower()
        # Простые эвристики
        if "capital" in q_lower:
            return 2  # Paris
        if "atomic number" in q_lower or "element" in q_lower:
            return 1  # Hydrogen
        if "speed of light" in q_lower:
            return 0
        if "red planet" in q_lower:
            return 1
        if "square root" in q_lower:
            return 1
        # HellaSwag: первый ответ обычно правдоподобнее
        if any(kw in q_lower for kw in ["because", "started", "ran"]):
            return 0
        return random.randint(0, len(choices) - 1)

    # Запуск оценки
    models = [
        ("Случайная модель", random_model),
        ("Правиловая модель", rule_based_model),
    ]

    all_results = {}
    for model_name, model_fn in models:
        random.seed(42)  # для воспроизводимости

        mmlu_result = mmlu.evaluate(model_fn)
        hella_result = hella.evaluate(model_fn)

        print(f"\n  === {model_name} ===")
        print(f"  MMLU: accuracy = {mmlu_result['accuracy']:.3f} "
              f"({mmlu_result['correct']}/{mmlu_result['total']})")
        for cat, score in mmlu_result["categories"].items():
            print(f"    {cat:15s}: {score:.3f}")

        print(f"  HellaSwag: accuracy = {hella_result['accuracy']:.3f} "
              f"({hella_result['correct']}/{hella_result['total']})")
        for cat, score in hella_result["categories"].items():
            print(f"    {cat:15s}: {score:.3f}")

        all_results[model_name] = {
            "MMLU": mmlu_result["accuracy"],
            "HellaSwag": hella_result["accuracy"],
        }

    # --- Сравнение моделей ---
    print("\n--- Сравнение моделей ---")
    print(f"  {'Модель':25s} | {'MMLU':>8s} | {'HellaSwag':>10s} | {'Среднее':>8s}")
    print("  " + "-" * 60)
    for model_name, scores in all_results.items():
        avg = sum(scores.values()) / len(scores)
        print(
            f"  {model_name:25s} | {scores['MMLU']:>8.3f} | "
            f"{scores['HellaSwag']:>10.3f} | {avg:>8.3f}"
        )

    # --- Метрики оценки моделей ---
    print("\n--- Дополнительные метрики автоматической оценки ---")
    metrics = {
        "Accuracy": "Доля правильных ответов (0-1)",
        "Pass@k": "Вероятность хотя бы одного правильного ответа за k попыток",
        "F1": "Баланс precision и recall для мультиклассовой классификации",
        "ROUGE-L": "Длина наибольшей общей подпоследовательности (для суммаризации)",
        "Exact Match": "Точное совпадение ответа (для QA)",
        "Win Rate": "Доля побед в попарном сравнении с другой моделью",
        "ELO Rating": "Рейтинг в стиле шахмат (Chatbot Arena)",
    }
    for metric, desc in metrics.items():
        print(f"  {metric:15s} — {desc}")

    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  110 — LLM Evaluation: основы оценки качества языковых моделей")
    print("=" * 70)
    print()

    demo_perplexity()
    demo_bleu()
    demo_human_eval()
    demo_benchmarks()

    print("=" * 70)
    print("  Итого: основные методы оценки LLM")
    print("=" * 70)
    print()
    print("  1. Perplexity    — внутренняя метрика модели (ниже = лучше)")
    print("  2. BLEU Score    — автоматическая оценка перевода (выше = лучше)")
    print("  3. Human Eval    — faithfulness, relevance, coherence (0-1 или 1-5)")
    print("  4. Benchmarks    — MMLU, HellaSwag, ARC, TruthfulQA (accuracy)")
    print()
    print("  Ключевые идеи:")
    print("  • Нет одной идеальной метрики — используйте комбинацию")
    print("  • Автоматические метрики не заменяют human evaluation")
    print("  • Perplexity измеряет \"удивление\" модели, а не качество ответа")
    print("  • BLEU не учитывает семантическое сходство")
    print("  • Benchmarks подвержены \"data contamination\" (утечке данных)")
    print()
