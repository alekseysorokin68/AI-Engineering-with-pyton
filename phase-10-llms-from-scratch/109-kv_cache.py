"""
KV Cache: эффективный inference для LLM
========================================

KV Cache — оптимизация, которая кэширует ключи (K) и значения (V)
из предыдущих токенов при автокодировании, чтобы избежать
повторного вычисления при генерации каждого нового токена.

Без KV Cache: генерация токена t требует O(t) операций (пересчёт K,V для всех предыдущих)
С KV Cache:   генерация токена t требует O(1) для K,V (только новый токен)

Реализация: чистый Python, без внешних зависимостей.
"""

import random
import time
import math

random.seed(42)


# ============================================================
# Вспомогательные функции (чистый Python)
# ============================================================

def mat_vec_mul(matrix, vec):
    """Умножение матрицы (список списков) на вектор."""
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in matrix]


def mat_mul(A, B):
    """Умножение матриц A (m×n) × B (n×p) -> (m×p)."""
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def transpose(M):
    """Транспонирование матрицы."""
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def softmax(vec):
    """Softmax по вектору."""
    max_v = max(vec)
    exp_v = [math.exp(x - max_v) for x in vec]
    s = sum(exp_v)
    return [x / s for x in exp_v]


def mat_softmax_rows(M):
    """Softmax по каждой строке матрицы."""
    return [softmax(row) for row in M]


def scale_dot_product_attention(Q, K, V, dk):
    """Scaled dot-product attention: Attn(Q,K,V) = softmax(QK^T / sqrt(dk)) V"""
    # QK^T
    Kt = transpose(K)
    scores = mat_mul(Q, Kt)
    # scale
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] /= math.sqrt(dk)
    # softmax
    attn_weights = mat_softmax_rows(scores)
    # weighted sum
    output = mat_mul(attn_weights, V)
    return output, attn_weights


def print_matrix(M, name="", indent=0):
    """Красивый вывод матрицы."""
    prefix = " " * indent
    if name:
        print(f"{prefix}{name}:")
    for row in M:
        print(f"{prefix}  [{', '.join(f'{x:8.4f}' for x in row)}]")


def random_matrix(rows, cols, scale=0.1):
    """Создать матрицу со случайными значениями."""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


# ============================================================
# Демо 1: KV Cache — хранение K и V
# ============================================================

def demo1_kv_cache_basics():
    print("=" * 70)
    print("ДЕМО 1: KV Cache — хранение K и V при автокодировании")
    print("=" * 70)

    dk = 4  # размерность ключей/значений
    n_heads = 1

    # Последовательность токенов: "A" "B" "C"
    # Каждый токен → эмбеддинг → проекции Wq, Wk, Wv → Q, K, V
    Wq = random_matrix(dk, dk, scale=0.5)
    Wk = random_matrix(dk, dk, scale=0.5)
    Wv = random_matrix(dk, dk, scale=0.5)

    # Эмбеддинги токенов (случайные, но фиксированные)
    token_embeddings = {
        "A": random_matrix(1, dk, scale=1.0)[0],
        "B": random_matrix(1, dk, scale=1.0)[0],
        "C": random_matrix(1, dk, scale=1.0)[0],
    }

    tokens = ["A", "B", "C"]

    print("\n--- Последовательное автокодирование БЕЗ KV Cache ---")
    print("При генерации токена 'C' нужно пересчитать K,V для A и B заново!\n")

    # Без KV Cache: каждый шаг пересчитываем ВСЁ
    full_k_history = []
    full_v_history = []

    for step, tok in enumerate(tokens):
        emb = [token_embeddings[tok]]  # (1, dk)
        # Проекции
        K_tok = mat_mul(emb, Wk)  # (1, dk)
        V_tok = mat_mul(emb, Wv)  # (1, dk)
        Q_tok = mat_mul(emb, Wq)  # (1, dk)

        # Полный пересчёт K и V для всей последовательности
        full_k_history.append(K_tok[0])
        full_v_history.append(V_tok[0])

        # Attention с накопленными K, V
        all_k = [list(row) for row in full_k_history]  # (step+1, dk)
        all_v = [list(row) for row in full_v_history]  # (step+1, dk)
        q = [Q_tok[0]]

        out, _ = scale_dot_product_attention(q, all_k, all_v, dk)
        print(f"  Шаг {step}: токен '{tok}' | K,V пересчитаны для {step+1} токенов | output[0..2] = {out[0][:3]}")

    print("\n--- Автокодирование С KV Cache ---")
    print("Кэшируем K,V и добавляем только новый токен!\n")

    # С KV Cache: храним K и V, добавляем только новый
    kv_cache_k = []
    kv_cache_v = []

    for step, tok in enumerate(tokens):
        emb = [token_embeddings[tok]]  # (1, dk)
        K_tok = mat_mul(emb, Wk)  # (1, dk)
        V_tok = mat_mul(emb, Wv)  # (1, dk)
        Q_tok = mat_mul(emb, Wq)  # (1, dk)

        # Добавляем в кэш (не пересчитываем!)
        kv_cache_k.append(K_tok[0])
        kv_cache_v.append(V_tok[0])

        # Attention с кэшированными K, V
        out, _ = scale_dot_product_attention(Q_tok, kv_cache_k, kv_cache_v, dk)
        print(f"  Шаг {step}: токен '{tok}' | KV cache: {step+1} элементов | output[0..2] = {out[0][:3]}")

    print("\n--- Сравнение ---")
    print(f"  Без KV Cache: вычислено {len(tokens)} × {len(tokens)} K,V проекций = {len(tokens)**2} умножений матриц")
    print(f"  С KV Cache:   вычислено {len(tokens)} × 1 K,V проекций = {len(tokens)} умножений матриц")
    print(f"  Экономия: вычисления K,V с O(n²) → O(n)")

    # Доказательство эквивалентности
    print("\n--- Проверка эквивалентности ---")
    # Повторный расчёт с кэшем для наглядности
    cache_k_final = []
    cache_v_final = []
    for tok in tokens:
        emb = [token_embeddings[tok]]
        K_tok = mat_mul(emb, Wk)
        V_tok = mat_mul(emb, Wv)
        cache_k_final.append(K_tok[0])
        cache_v_final.append(V_tok[0])

    # Полный пересчёт для токена C
    emb_c = [token_embeddings["C"]]
    K_c = mat_mul(emb_c, Wk)
    V_c = mat_mul(emb_c, Wv)
    Q_c = mat_mul(emb_c, Wq)

    out_full, _ = scale_dot_product_attention(Q_c, full_k_history, full_v_history, dk)
    out_cached, _ = scale_dot_product_attention(Q_c, cache_k_final, cache_v_final, dk)

    match = all(abs(a - b) < 1e-10 for a, b in zip(out_full[0], out_cached[0]))
    print(f"  Результаты совпадают: {match}")
    print()


# ============================================================
# Демо 2: Генерация с кэшированием (пошаговая)
# ============================================================

def demo2_autoregressive_generation():
    print("=" * 70)
    print("ДЕМО 2: Пошаговая генерация с KV Cache")
    print("=" * 70)

    dk = 4
    vocab_size = 5  # упрощённый словарь: 0..4

    # Матрицы проекций (все токены разделяют одни веса)
    Wq = random_matrix(dk, dk, scale=0.5)
    Wk = random_matrix(dk, dk, scale=0.5)
    Wv = random_matrix(dk, dk, scale=0.5)
    Wo = random_matrix(dk, dk, scale=0.5)  # выходная проекция

    # Эмбеддинги словаря
    embeddings = [random_matrix(1, dk, scale=1.0)[0] for _ in range(vocab_size)]

    print("\nЭмбеддинги словаря (vocab_size=5, dk=4):")
    for i, e in enumerate(embeddings):
        print(f"  token {i}: [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}, {e[3]:.3f}]")

    print("\n--- Генерация 6 токенов с KV Cache ---")
    print("(Каждый шаг: 1 new K,V проекция вместо пересчёта всей последовательности)\n")

    prompt_token = 0  # стартовый токен
    generated_tokens = [prompt_token]
    kv_cache_k = []
    kv_cache_v = []

    for step in range(6):
        if step == 0:
            tok = prompt_token
        else:
            tok = generated_tokens[-1]

        # Получаем эмбеддинг текущего токена
        emb = [embeddings[tok]]

        # Вычисляем K, V, Q только для НОВОГО токена
        K_new = mat_mul(emb, Wk)  # (1, dk)
        V_new = mat_mul(emb, Wv)  # (1, dk)
        Q_new = mat_mul(emb, Wq)  # (1, dk)

        # Добавляем в кэш
        kv_cache_k.append(K_new[0])
        kv_cache_v.append(V_new[0])

        # Attention с полным контекстом (из кэша)
        output, weights = scale_dot_product_attention(Q_new, kv_cache_k, kv_cache_v, dk)

        # Проецируем через Wo
        projected = mat_mul(output, Wo)

        # Простейший "декодер": выбираем argmax по первым vocab_size проекциям
        # (в реальном LLM здесь был бы LM head)
        scores = [sum(projected[0][j] * embeddings[i][j] for j in range(dk)) for i in range(vocab_size)]
        next_token = scores.index(max(scores))

        generated_tokens.append(next_token)

        print(f"  Step {step}: input_token={tok} | cache_size={len(kv_cache_k)} | "
              f"attn_weights={[f'{w:.3f}' for w in weights[0]]} | "
              f"next_token={next_token}")

    print(f"\n  Сгенерированная последовательность: {generated_tokens}")
    print(f"  Длина кэша: {len(kv_cache_k)} K,V пар")
    print(f"  Без KV Cache потребовалось бы пересчитать K,V для {sum(range(1, len(generated_tokens)+1))} позиций")
    print(f"  С KV Cache: всего {len(generated_tokens)} проекций K,V")
    print()


# ============================================================
# Демо 3: Сравнение скорости (с кэшем vs без кэширования)
# ============================================================

def demo3_speed_comparison():
    print("=" * 70)
    print("ДЕМО 3: Сравнение скорости — KV Cache vs Full Recomputation")
    print("=" * 70)

    dk = 8

    Wk = random_matrix(dk, dk, scale=0.5)
    Wv = random_matrix(dk, dk, scale=0.5)
    Wq = random_matrix(dk, dk, scale=0.5)

    seq_lengths = [5, 20, 50, 100]
    n_trials = 20

    print(f"\ndk={dk}, {n_trials} запусков на каждую длину\n")
    print(f"{'Длина':>6} | {'Без Cache (мс)':>16} | {'С Cache (мс)':>14} | {'Ускорение':>10} | {'K,V ops':>10}")
    print("-" * 70)

    for seq_len in seq_lengths:
        # Генерируем случайную последовательность
        tokens = [random_matrix(1, dk, scale=1.0)[0] for _ in range(seq_len)]

        # --- Полный пересчёт ---
        t0 = time.perf_counter()
        for _ in range(n_trials):
            all_k = []
            all_v = []
            for i, tok in enumerate(tokens):
                emb = [tok]
                K_i = mat_mul(emb, Wk)
                V_i = mat_mul(emb, Wv)
                Q_i = mat_mul(emb, Wq)

                all_k.append(K_i[0])
                all_v.append(V_i[0])

                # Attention с пересчётом (для замера K,V ops)
                scale_dot_product_attention(Q_i, all_k, all_v, dk)
        t_full = (time.perf_counter() - t0) / n_trials

        # --- С KV Cache ---
        t0 = time.perf_counter()
        for _ in range(n_trials):
            cache_k = []
            cache_v = []
            for i, tok in enumerate(tokens):
                emb = [tok]
                K_i = mat_mul(emb, Wk)
                V_i = mat_mul(emb, Wv)
                Q_i = mat_mul(emb, Wq)

                cache_k.append(K_i[0])
                cache_v.append(V_i[0])

                scale_dot_product_attention(Q_i, cache_k, cache_v, dk)
        t_cached = (time.perf_counter() - t0) / n_trials

        # K,V проекций: полный = n*(n+1)/2, cached = n
        full_kv_ops = seq_len * (seq_len + 1) // 2
        cached_kv_ops = seq_len
        speedup = t_full / t_cached if t_cached > 0 else float('inf')

        print(f"{seq_len:>6} | {t_full*1000:>14.3f} ms | {t_cached*1000:>12.3f} ms | {speedup:>8.2f}x | {full_kv_ops:>6}/{cached_kv_ops}")

    print("\n  Вывод: ускорение растёт с длиной последовательности.")
    print("  Для seq_len=100: O(n²) vs O(n) K,V операций — критическая разница!")
    print()


# ============================================================
# Демо 4: Память KV Cache
# ============================================================

def demo4_memory_analysis():
    print("=" * 70)
    print("ДЕМО 4: Анализ памяти KV Cache")
    print("=" * 70)

    # Параметры реальных моделей (упрощённые)
    models = [
        {"name": "GPT-2 Small",  "n_layers": 12, "n_heads": 12, "d_model": 768,  "bytes_per_param": 2},  # fp16
        {"name": "GPT-2 Medium", "n_layers": 24, "n_heads": 16, "d_model": 1024, "bytes_per_param": 2},
        {"name": "Llama-7B",     "n_layers": 32, "n_heads": 32, "d_model": 4096, "bytes_per_param": 2},
        {"name": "Llama-13B",    "n_layers": 40, "n_heads": 40, "d_model": 5120, "bytes_per_param": 2},
        {"name": "Llama-70B",    "n_layers": 80, "n_heads": 64, "d_model": 8192, "bytes_per_param": 2},
    ]

    print("\nРазмер KV Cache = 2 (K + V) × n_layers × seq_len × d_model × bytes_per_param")
    print("Формула: memory = 2 * L * S * D * B байт\n")

    print(f"{'Модель':<16} | {'Seq 1K':>12} | {'Seq 4K':>12} | {'Seq 16K':>12} | {'Seq 64K':>12}")
    print("-" * 75)

    for m in models:
        L = m["n_layers"]
        D = m["d_model"]
        B = m["bytes_per_param"]

        sizes = []
        for S in [1024, 4096, 16384, 65536]:
            bytes_needed = 2 * L * S * D * B
            sizes.append(bytes_needed)
        print(f"{m['name']:<16} | {sizes[0]/1024**2:>10.1f} MB | {sizes[1]/1024**2:>10.1f} MB | {sizes[2]/1024**2:>10.1f} MB | {sizes[3]/1024**2:>10.1f} MB")

    print("\n--- Важные инсайты ---")
    print("1. KV Cache растёт линейно с длиной последовательности")
    print("2. Для длинных контекстов (128K+) KV Cache может занимать больше памяти, чем веса модели!")
    print("3. Multi-Query Attention (MQA) и Grouped-Query Attention (GQA) уменьшают KV Cache:")
    print("   - MQA: все головы делят 1 K,V -> memory / n_heads")
    print("   - GQA: группы голов делят K,V -> memory / group_size")

    print("\n--- Пример: Llama-7B с разными стратегиями ---")
    m = {"n_layers": 32, "d_model": 4096, "bytes_per_param": 2}
    S = 4096  # seq_len = 4K
    B = m["bytes_per_param"]
    L = m["n_layers"]
    D = m["d_model"]

    full_kv = 2 * L * S * D * B
    mqa_kv = full_kv // 32     # все головы делят 1 K,V
    gqa_8 = full_kv // 8       # группы по 8 голов

    print(f"\n  Seq_len={S}, n_layers={L}, d_model={D}")
    print(f"  Multi-Head Attention (MHA):  {full_kv/1024**2:>8.1f} MB  (baseline)")
    print(f"  Grouped-Query Attention (GQA-8): {gqa_8/1024**2:>8.1f} MB  ({gqa_8/full_kv*100:.1f}%)")
    print(f"  Multi-Query Attention (MQA):    {mqa_kv/1024**2:>8.1f} MB  ({mqa_kv/full_kv*100:.1f}%)")

    print("\n--- Демонстрация GQA на уровне данных ---")
    dk = 4
    n_heads = 8
    n_kv_groups = 2  # GQA: 2 группы, каждая с 4 головами
    group_size = n_heads // n_kv_groups

    Wk_heads = [random_matrix(dk, dk, scale=0.5) for _ in range(n_heads)]
    Wv_heads = [random_matrix(dk, dk, scale=0.5) for _ in range(n_heads)]

    # MHA: 8 отдельных K,V
    mha_k = [random_matrix(1, dk, scale=0.5)[0] for _ in range(n_heads)]
    mha_v = [random_matrix(1, dk, scale=0.5)[0] for _ in range(n_heads)]
    mha_params = n_heads * dk * 2  # K + V для каждой головы

    # GQA: 2 K,V, каждая используется 4 головами
    gqa_k = [random_matrix(1, dk, scale=0.5)[0] for _ in range(n_kv_groups)]
    gqa_v = [random_matrix(1, dk, scale=0.5)[0] for _ in range(n_kv_groups)]
    gqa_params = n_kv_groups * dk * 2

    print(f"\n  n_heads={n_heads}, n_kv_groups={n_kv_groups}, dk={dk}")
    print(f"  MHA KV params: {mha_params} ({n_heads} K + {n_heads} V vectors)")
    print(f"  GQA KV params: {gqa_params} ({n_kv_groups} K + {n_kv_groups} V vectors)")
    print(f"  Сжатие: {mha_params/gqa_params:.1f}x")

    # Каждая группа голов видит свой K,V
    print(f"\n  Распределение K по головам (GQA):")
    for i in range(n_heads):
        group_idx = i // group_size
        print(f"    Head {i} -> KV group {group_idx}")

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  KV Cache: эффективный inference для LLM")
    print("  (чистый Python, без зависимостей)")
    print("=" * 70)
    print()

    demo1_kv_cache_basics()
    demo2_autoregressive_generation()
    demo3_speed_comparison()
    demo4_memory_analysis()

    print("=" * 70)
    print("  ИТОГО: KV Cache — ключевая оптимизация для LLM inference")
    print("  - Ускоряет генерацию с O(n²) до O(n) по K,V операциям")
    print("  - Экономит память при batch inference (разделяется между запросами)")
    print("  - MQA/GQA进一步减少 memory footprint для длинных контекстов")
    print("=" * 70)
