"""
Multi-Head Attention с нуля на Python
Параллельные головы внимания без внешних зависимостей
"""

import math
import random

random.seed(42)

# ─── Вспомогательные функции ────────────────────────────────────────────────

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def random_matrix(rows, cols, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def matmul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = zeros(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            s = 0.0
            for k in range(cols_A):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def transpose(A):
    return [list(row) for row in zip(*A)]


def softmax_row(row):
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_matrix(M):
    return [softmax_row(row) for row in M]


def scale_matrix(M, factor):
    return [[v * factor for v in row] for row in M]


def print_matrix(M, name="", indent=0):
    prefix = " " * indent
    if name:
        print(f"{prefix}{name}:")
    for row in M:
        vals = "  ".join(f"{v:+.4f}" for v in row)
        print(f"{prefix}  [{vals}]")
    print()


def print_vector(v, name="", indent=0):
    prefix = " " * indent
    if name:
        print(f"{prefix}{name}:")
    vals = "  ".join(f"{v_:+.4f}" for v_ in v)
    print(f"{prefix}  [{vals}]\n")


def visualize_attention(weights, tokens_q, tokens_kv):
    max_w = max(max(row) for row in weights)
    col_w = 10
    header = " " * 14
    for t in tokens_kv:
        header += f"{t:>{col_w}}"
    print(header)
    print(" " * 14 + "-" * (col_w * len(tokens_kv)))
    for i, row in enumerate(weights):
        line = f"  {tokens_q[i]:>10} |"
        for val in row:
            filled = int((val / max_w) * 8) if max_w > 0 else 0
            bar = "█" * filled + "░" * (8 - filled)
            line += f" {bar} {val:.3f}"
        print(line)
    print()


# ─── Single-Head Attention ──────────────────────────────────────────────────

def single_head_attention(Q, K, V):
    d_k = len(K[0])
    K_T = transpose(K)
    scores = matmul(Q, K_T)
    scale = 1.0 / math.sqrt(d_k)
    scaled = scale_matrix(scores, scale)
    weights = softmax_matrix(scaled)
    output = matmul(weights, V)
    return weights, output


# ─── Multi-Head Attention ──────────────────────────────────────────────────

def split_heads(matrix, n_heads):
    """Разбивает матрицу (seq_len, d_model) на n_heads частей по последнему измерению.
    Возвращает список из n_heads матриц (seq_len, d_head)."""
    seq_len = len(matrix)
    d_model = len(matrix[0])
    d_head = d_model // n_heads
    heads = []
    for h in range(n_heads):
        head = []
        for i in range(seq_len):
            row = matrix[i][h * d_head:(h + 1) * d_head]
            head.append(row)
        heads.append(head)
    return heads


def concat_heads(heads):
    """Склеивает список матриц (seq_len, d_head) обратно в (seq_len, d_model)."""
    seq_len = len(heads[0])
    result = []
    for i in range(seq_len):
        row = []
        for head in heads:
            row.extend(head[i])
        result.append(row)
    return result


def scaled_dot_product_attention(Q, K, V):
    """Scaled Dot-Product Attention."""
    d_k = len(K[0])
    K_T = transpose(K)
    scores = matmul(Q, K_T)
    scale = 1.0 / math.sqrt(d_k)
    scaled = scale_matrix(scores, scale)
    weights = softmax_matrix(scaled)
    output = matmul(weights, V)
    return weights, output


def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads):
    """Multi-Head Attention: проекция → разбиение → параллельные attention → конкатенация → проекция."""
    seq_len = len(X)
    d_model = len(X[0])

    # 1. Линейные проекции Q, K, V
    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)

    # 2. Разбиение на головы
    Q_heads = split_heads(Q, n_heads)
    K_heads = split_heads(K, n_heads)
    V_heads = split_heads(V, n_heads)

    # 3. Параллельные attention для каждой головы
    head_outputs = []
    head_weights = []
    for h in range(n_heads):
        weights, output = scaled_dot_product_attention(
            Q_heads[h], K_heads[h], V_heads[h]
        )
        head_weights.append(weights)
        head_outputs.append(output)

    # 4. Конкатенация результатов
    concat = concat_heads(head_outputs)

    # 5. Финальная линейная проекция
    output = matmul(concat, W_O)

    return {
        "Q": Q, "K": K, "V": V,
        "Q_heads": Q_heads, "K_heads": K_heads, "V_heads": V_heads,
        "head_weights": head_weights,
        "head_outputs": head_outputs,
        "concat": concat,
        "output": output,
    }


# ─── Демонстрации ───────────────────────────────────────────────────────────

def demo1():
    print("=" * 70)
    print("  ДЕМО 1: Разбиение на головы (Split Heads)")
    print("=" * 70)
    print()
    print("Multi-Head Attention начинается с разбиения Q, K, V на несколько")
    print("независимых 'голов', каждая работает со своей подмножеством измерений.")
    print()

    seq_len = 3
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads

    print(f"  seq_len  = {seq_len}")
    print(f"  d_model  = {d_model}")
    print(f"  n_heads  = {n_heads}")
    print(f"  d_head   = {d_model} / {n_heads} = {d_head}")
    print()

    X = random_matrix(seq_len, d_model, scale=1.0)
    print_matrix(X, "Входная матрица X (seq_len x d_model)")

    W_Q = random_matrix(d_model, d_model, scale=0.5)
    Q = matmul(X, W_Q)

    print_matrix(Q, "Q после проекции (seq_len x d_model)")
    print(f"Разбиваем Q на {n_heads} голов по {d_head} измерений каждая:")
    print()

    Q_heads = split_heads(Q, n_heads)
    for h, head in enumerate(Q_heads):
        print_matrix(head, f"  Голова {h} (seq_len x d_head={d_head})")

    print("Ключевая идея:")
    print(f"  Голова 0 видит измерения [0..{d_head - 1}] — один 'аспект' внимания")
    print(f"  Голова 1 видит измерения [{d_head}..{d_model - 1}] — другой 'аспект'")
    print()

    print("Обратная операция — конкатенация:")
    recombined = concat_heads(Q_heads)
    print_matrix(recombined, "  Q_heads → concat → обратно в (seq_len x d_model)")
    print(f"  Совпадает с оригинальным Q? {recombined == Q}")
    print()


def demo2():
    print("=" * 70)
    print("  ДЕМО 2: Параллельный Attention")
    print("=" * 70)
    print()
    print("Каждая голова выполняет Attention независимо — параллельно.")
    print("Это позволяет модели关注 разные 'аспекты' отношений.")
    print()

    seq_len = 4
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads
    tokens = ["The", "cat", "sat", "on"]

    X = random_matrix(seq_len, d_model, scale=1.0)
    W_Q = random_matrix(d_model, d_model, scale=0.5)
    W_K = random_matrix(d_model, d_model, scale=0.5)
    W_V = random_matrix(d_model, d_model, scale=0.5)

    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)

    Q_heads = split_heads(Q, n_heads)
    K_heads = split_heads(K, n_heads)
    V_heads = split_heads(V, n_heads)

    print(f"Токены: {tokens}")
    print(f"Голов: {n_heads}, d_head = {d_head}")
    print()

    for h in range(n_heads):
        print(f"--- ГОЛОВА {h} ---")
        print(f"  Q_{h} (d_head = {d_head}):")
        print_matrix(Q_heads[h], indent=4)
        print(f"  K_{h} (d_head = {d_head}):")
        print_matrix(K_heads[h], indent=4)

        weights, output = scaled_dot_product_attention(
            Q_heads[h], K_heads[h], V_heads[h]
        )
        print(f"  Attention weights для головы {h}:")
        visualize_attention(weights, tokens, tokens)
        print_matrix(output, f"  Output головы {h}")
        print()

    print("Наблюдение: разные головы produces different attention patterns!")
    print("  Голова 0 может фокусироваться на синтаксисе (subj → verb)")
    print("  Голова 1 может фокусироваться на семантике (кто действует)")
    print()


def demo3():
    print("=" * 70)
    print("  ДЕМО 3: Конкатенация и проекция")
    print("=" * 70)
    print()
    print("После параллельных attention, результаты склеиваются и проецируются.")
    print()

    seq_len = 3
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads
    tokens = ["I", "love", "AI"]

    X = random_matrix(seq_len, d_model, scale=1.0)
    W_Q = random_matrix(d_model, d_model, scale=0.5)
    W_K = random_matrix(d_model, d_model, scale=0.5)
    W_V = random_matrix(d_model, d_model, scale=0.5)
    W_O = random_matrix(d_model, d_model, scale=0.5)

    result = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)

    print("Шаг 1: Входные проекции")
    print_matrix(result["Q"], "  Q (все головы)")
    print()

    print("Шаг 2: Разбиение на головы")
    for h in range(n_heads):
        print_matrix(result["Q_heads"][h], f"  Q[{h}]", indent=2)
    print()

    print("Шаг 3: Параллельные Attention")
    for h in range(n_heads):
        print(f"  Голова {h}:")
        visualize_attention(result["head_weights"][h], tokens, tokens)
    print()

    print("Шаг 4: Конкатенация результатов голов")
    print(f"  Каждая голова: ({seq_len}, {d_head})")
    print(f"  После concat: ({seq_len}, {d_model})")
    print_matrix(result["concat"], "  Concat(H_0, H_1)")
    print()

    print("Шаг 5: Финальная проекция W_O")
    print(f"  W_O: ({d_model}, {d_model})")
    print_matrix(result["output"], "  Final output")

    print("Итого Multi-Head Attention:")
    print("  MultiHead(Q, K, V) = Concat(head_0, ..., head_h) @ W_O")
    print("  где head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)")
    print()


def demo4():
    print("=" * 70)
    print("  ДЕМО 4: Single-Head vs Multi-Head")
    print("=" * 70)
    print()
    print("Сравним: одна большая голова vs несколько маленьких.")
    print()

    seq_len = 4
    d_model = 8
    tokens = ["A", "B", "C", "D"]

    # Общие входные данные
    X = random_matrix(seq_len, d_model, scale=1.0)

    # --- Single Head ---
    print("─" * 70)
    print("  SINGLE-HEAD ATTENTION (одна голова, d_k = d_model)")
    print("─" * 70)
    print()

    W_Q_1 = random_matrix(d_model, d_model, scale=0.5)
    W_K_1 = random_matrix(d_model, d_model, scale=0.5)
    W_V_1 = random_matrix(d_model, d_model, scale=0.5)

    Q1 = matmul(X, W_Q_1)
    K1 = matmul(X, W_K_1)
    V1 = matmul(X, W_V_1)

    weights_1, output_1 = single_head_attention(Q1, K1, V1)

    print("  Attention weights:")
    visualize_attention(weights_1, tokens, tokens)
    print_matrix(output_1, "  Output")
    print(f"  Размерность одной головы: ({seq_len}, {d_model})")
    print()

    # --- Multi Head ---
    print("─" * 70)
    print(f"  MULTI-HEAD ATTENTION ({2} головы, d_head = {d_model // 2})")
    print("─" * 70)
    print()

    n_heads = 2
    d_head = d_model // n_heads
    W_Q_M = random_matrix(d_model, d_model, scale=0.5)
    W_K_M = random_matrix(d_model, d_model, scale=0.5)
    W_V_M = random_matrix(d_model, d_model, scale=0.5)
    W_O_M = random_matrix(d_model, d_model, scale=0.5)

    result = multi_head_attention(X, W_Q_M, W_K_M, W_V_M, W_O_M, n_heads)

    for h in range(n_heads):
        print(f"  Голова {h} (d_head = {d_head}):")
        visualize_attention(result["head_weights"][h], tokens, tokens)

    print_matrix(result["output"], "  Final output")
    print()

    # --- Анализ ---
    print("─" * 70)
    print("  СРАВНЕНИЕ")
    print("─" * 70)
    print()
    print("  Single-Head:")
    print(f"    - Параметров: Q({d_model}x{d_model}) + K + V = {3 * d_model * d_model}")
    print(f"    - Одна таблица attention weights — один 'фокус'")
    print(f"    - Выход: ({seq_len}, {d_model})")
    print()
    print("  Multi-Head (2 головы):")
    print(f"    - Параметров: Q+K+V({d_model}x{d_model}) + O({d_model}x{d_model}) = {4 * d_model * d_model}")
    print(f"    - 2 таблицы attention weights — разные 'аспекты'")
    print(f"    - Каждая голова: ({seq_len}, {d_head})")
    print(f"    - Выход: ({seq_len}, {d_model})")
    print()
    print("  Преимущества Multi-Head:")
    print("    1. Разные головы learning different patterns (синтаксис, семантика, позиции)")
    print("    2. Параллельные вычисления — эффективно на GPU")
    print("    3. Большая 'capacity' при том же d_model")
    print("    4. Стабильнее градиенты (каждая голова — отдельный путь)")
    print()


# ─── Запуск ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()

    print("=" * 70)
    print("  Multi-Head Attention реализован с нуля!")
    print("  Без numpy, torch, transformers — только Python + math + random")
    print("=" * 70)
