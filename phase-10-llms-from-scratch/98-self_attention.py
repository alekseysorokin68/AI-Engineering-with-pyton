"""
Self-Attention с нуля на Python
Scaled Dot-Product Attention без внешних зависимостей
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


def attention_weights_bar(weights, tokens_q, tokens_kv):
    header = " " * 14 + "  ".join(f"{t:>7}" for t in tokens_kv)
    print(header)
    for i, row in enumerate(weights):
        bars = ""
        for val in row:
            filled = int(val * 40)
            bars += f"{'█' * filled}{'░' * (40 - filled)}  "
        print(f"  {tokens_q[i]:>10} | {bars}{val:.4f}")


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


# ─── Scaled Dot-Product Attention ───────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V):
    d_k = len(K[0])
    K_T = transpose(K)
    scores = matmul(Q, K_T)
    scale_factor = 1.0 / math.sqrt(d_k)
    scaled_scores = scale_matrix(scores, scale_factor)
    weights = softmax_matrix(scaled_scores)
    output = matmul(weights, V)
    return scaled_scores, weights, output


# ─── Демонстрации ───────────────────────────────────────────────────────────

def demo1():
    print("=" * 70)
    print("  ДЕМО 1: Q, K, V проекции")
    print("=" * 70)
    print()
    print("Self-attention начинается с трёх проекций входных эмбеддингов:")
    print("  Q (Query)  — 'что я ищу'")
    print("  K (Key)    — 'что я могу предложить'")
    print("  V (Value)  — 'что я передаю'")
    print()

    d_model = 4
    n_tokens = 3
    d_k = 3

    print(f"Размерность модели d_model = {d_model}")
    print(f"Количество токенов n_tokens = {n_tokens}")
    print(f"Размерность ключей d_k = {d_k}")
    print()

    X = random_matrix(n_tokens, d_model, scale=1.0)
    print_matrix(X, "Входные эмбеддинги X")

    W_Q = random_matrix(d_model, d_k, scale=0.5)
    W_K = random_matrix(d_model, d_k, scale=0.5)
    W_V = random_matrix(d_model, d_k, scale=0.5)

    print_matrix(W_Q, "Матрица весов W_Q (Query projection)")
    print_matrix(W_K, "Матрица весов W_K (Key projection)")
    print_matrix(W_V, "Матрица весов W_V (Value projection)")

    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)

    print_matrix(Q, "Q = X @ W_Q (Query матрица)")
    print_matrix(K, "K = X @ W_K (Key матрица)")
    print_matrix(V, "V = X @ W_V (Value матрица)")

    return Q, K, V


def demo2(Q, K, V):
    print("=" * 70)
    print("  ДЕМО 2: Attention scores")
    print("=" * 70)
    print()
    print("Attention scores = Q @ K^T — показывает насколько каждый query")
    print("соответствует каждому key.")
    print()

    tokens = ["I", "love", "cats"]
    K_T = transpose(K)
    raw_scores = matmul(Q, K_T)

    print_matrix(raw_scores, "Сырые scores (Q @ K^T)")
    print("Значения могут быть большими — поэтому我们需要 нормализацию.")
    print()

    d_k = len(K[0])
    scale = 1.0 / math.sqrt(d_k)
    scaled = scale_matrix(raw_scores, scale)

    print(f"Масштабирование: scores / sqrt(d_k) = scores / sqrt({d_k}) = scores * {scale:.4f}")
    print()
    print_matrix(scaled, "Scaled scores")

    print("Интерпретация для токена 'love' (строка 1):")
    print(f"  → scores[1] = [{', '.join(f'{v:.4f}' for v in scaled[1])}]")
    print(f"  → Наибольший score указывает на наиболее релевантный токен")
    print()

    return scaled


def demo3(scaled_scores):
    print("=" * 70)
    print("  ДЕМО 3: Softmax normalization")
    print("=" * 70)
    print()
    print("Softmax превращает scores в вероятности (сумма = 1.0).")
    print("Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))")
    print()

    tokens = ["I", "love", "cats"]

    print("Процесс softmax для строки 'love':")
    row = scaled_scores[1]
    print(f"  Вход: [{', '.join(f'{v:.4f}' for v in row)}]")
    max_val = max(row)
    print(f"  max = {max_val:.4f} (вычитаем для численной стабильности)")

    shifted = [v - max_val for v in row]
    print(f"  После вычитания max: [{', '.join(f'{v:.4f}' for v in shifted)}]")

    exps = [math.exp(v) for v in shifted]
    print(f"  exp: [{', '.join(f'{v:.4f}' for v in exps)}]")

    total = sum(exps)
    probs = [e / total for e in exps]
    print(f"  sum(exp) = {total:.4f}")
    print(f"  softmax: [{', '.join(f'{v:.4f}' for v in probs)}]")
    print(f"  Сумма = {sum(probs):.4f} (должна быть ~1.0)")
    print()

    weights = softmax_matrix(scaled_scores)
    print_matrix(weights, "Матрица attention weights (все строки)")
    print("Каждая строка — распределение внимания для соответствующего токена.")
    print()

    return weights


def demo4(Q, K, V, weights):
    print("=" * 70)
    print("  ДЕМО 4: Визуализация attention weights")
    print("=" * 70)
    print()

    tokens_q = ["I", "love", "cats"]
    tokens_kv = ["I", "love", "cats"]

    print("Attention weights: кто на кого смотрит")
    print("(чем ярче — тем сильнее внимание)")
    print()

    visualize_attention(weights, tokens_q, tokens_kv)

    print("Горизонтальные полоски показывают силу внимания:")
    print("  █ = высокое внимание")
    print("  ░ = низкое внимание")
    print()

    print("Выход self-attention (V, взвешенное по weights):")
    output = matmul(weights, V)
    print_matrix(output, "Output = weights @ V")

    print("Ключевой инсайт:")
    print("  Каждый выходной вектор = взвешенная сумма V-векторов всех токенов,")
    print("  где веса определяются совместимостью Q и K.")
    print()

    print("Итого self-attention в одну строку:")
    print("  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V")
    print()

    print("=" * 70)
    print("  Сравнение: attention weights для разных токенов")
    print("=" * 70)
    print()

    for i, tok in enumerate(tokens_q):
        print(f"  '{tok}' смотрит на:")
        for j, kv_tok in enumerate(tokens_kv):
            bar_len = int(weights[i][j] * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"    {kv_tok:>6}: {bar} {weights[i][j]:.4f}")
        print()


# ─── Запуск ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Q, K, V = demo1()
    scaled = demo2(Q, K, V)
    weights = demo3(scaled)
    demo4(Q, K, V, weights)

    print("=" * 70)
    print("  Self-Attention реализован с нуля!")
    print("  Без numpy, torch, transformers — только Python + math + random")
    print("=" * 70)
