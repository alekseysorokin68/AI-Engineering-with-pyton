"""
Transformer Block — From Scratch (NumPy/Torch Free)
====================================================
Реализация блока трансформера на чистом Python с random.

Компоненты:
  1. Multi-Head Self-Attention
  2. Feed-Forward Network
  3. Layer Normalization
  4. Residual Connections
  5. Masking (causal / padding)
"""

import math
import random

random.seed(42)

# ──────────────────────────────────────────────
#  Вспомогательные матричные операции
# ──────────────────────────────────────────────

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def ones(rows, cols):
    return [[1.0] * cols for _ in range(rows)]

def rand_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def matmul(A, B):
    """A: (m×n), B: (n×p) -> (m×p)"""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = zeros(m, p)
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def transpose(A):
    return [list(row) for row in zip(*A)]

def scale_matrix(A, s):
    return [[v * s for v in row] for row in A]

def add_matrices(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]

def subtract_matrices(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]

def softmax_rows(A):
    """Softmax по каждой строке."""
    result = []
    for row in A:
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result

def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm: x -> gamma * (x - mean) / sqrt(var + eps) + beta"""
    rows, cols = len(x), len(x[0])
    out = zeros(rows, cols)
    for i in range(rows):
        mean = sum(x[i]) / cols
        var = sum((v - mean) ** 2 for v in x[i]) / cols
        for j in range(cols):
            out[i][j] = gamma[j] * (x[i][j] - mean) / math.sqrt(var + eps) + beta[j]
    return out

def relu_matrix(A):
    return [[max(0.0, v) for v in row] for row in A]

def gelu_approx(x):
    """GELU-аппроксимация через tanh."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_matrix(A):
    return [[gelu_approx(v) for v in row] for row in A]

def linear_forward(x, W, bias):
    """Линейный слой: x @ W + bias"""
    out = matmul(x, W)
    rows, cols = len(out), len(out[0])
    for i in range(rows):
        for j in range(cols):
            out[i][j] += bias[j]
    return out

def print_matrix(M, name="", max_width=90):
    """Красивый вывод матрицы."""
    if name:
        print(f"  {name}:")
    for row in M:
        vals = "  ".join(f"{v:7.4f}" for v in row)
        # обрезаем длинные строки
        if len(vals) > max_width:
            vals = vals[:max_width - 3] + "..."
        print(f"    [{vals}]")

def print_vec(v, name=""):
    if name:
        print(f"  {name}: [{', '.join(f'{x:.4f}' for x in v)}]")


# ══════════════════════════════════════════════
#  1. MULTI-HEAD SELF-ATTENTION
# ══════════════════════════════════════════════

class MultiHeadSelfAttention:
    """Multi-Head Self-Attention без маскирования."""

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Инициализация весов Q, K, V и выходного проекционного слоя
        scale = math.sqrt(2.0 / d_model)
        self.W_q = rand_matrix(d_model, d_model, scale)
        self.W_k = rand_matrix(d_model, d_model, scale)
        self.W_v = rand_matrix(d_model, d_model, scale)
        self.W_o = rand_matrix(d_model, d_model, scale)
        self.b_q = [0.0] * d_model
        self.b_k = [0.0] * d_model
        self.b_v = [0.0] * d_model
        self.b_o = [0.0] * d_model

    def forward(self, x, mask=None):
        """
        x: (seq_len, d_model)
        mask: (seq_len, seq_len) — None или матрица (True/False/float)
        """
        seq_len = len(x)

        # Линейные проекции
        Q = linear_forward(x, self.W_q, self.b_q)
        K = linear_forward(x, self.W_k, self.b_k)
        V = linear_forward(x, self.W_v, self.b_v)

        # Разбиение на головы: каждая голова получает d_k размерности
        # (seq_len, n_heads, d_k)
        heads = []
        for h in range(self.n_heads):
            # Извлекаем срез [h*d_k : (h+1)*d_k] для каждой головы
            Q_h = [[Q[i][h * self.d_k + j] for j in range(self.d_k)] for i in range(seq_len)]
            K_h = [[K[i][h * self.d_k + j] for j in range(self.d_k)] for i in range(seq_len)]
            V_h = [[V[i][h * self.d_k + j] for j in range(self.d_k)] for i in range(seq_len)]

            # Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
            Kt = transpose(K_h)                          # (d_k, seq_len)
            scores = matmul(Q_h, Kt)                     # (seq_len, seq_len)
            scale = 1.0 / math.sqrt(self.d_k)
            scores = scale_matrix(scores, scale)

            # Применение маски (если есть): заменяем masked позиции на -inf
            if mask is not None:
                for i in range(seq_len):
                    for j in range(seq_len):
                        if mask[i][j]:
                            scores[i][j] = -1e9

            # Softmax
            attn = softmax_rows(scores)                  # (seq_len, seq_len)

            # Взвешенная сумма значений
            head_out = matmul(attn, V_h)                 # (seq_len, d_k)
            heads.append(head_out)

        # Склеиваем головы обратно: (seq_len, d_model)
        concat = zeros(seq_len, self.d_model)
        for i in range(seq_len):
            for h in range(self.n_heads):
                for j in range(self.d_k):
                    concat[i][h * self.d_k + j] = heads[h][i][j]

        # Выходная проекция
        output = linear_forward(concat, self.W_o, self.b_o)
        return output


class CausalMaskedSelfAttention(MultiHeadSelfAttention):
    """Multi-Head Self-Attention с causal (autoregressive) маской."""

    def forward(self, x):
        seq_len = len(x)
        # Нижнетреугольная маска: токен i может смотреть только на j <= i
        mask = [[False] * seq_len for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                mask[i][j] = (j > i)  # True = замаскировать (future positions)
        return super().forward(x, mask=mask)


# ══════════════════════════════════════════════
#  2. FEED-FORWARD NETWORK
# ══════════════════════════════════════════════

class FeedForward:
    """Position-wise Feed-Forward Network: Linear -> GELU -> Linear."""

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        scale1 = math.sqrt(2.0 / d_model)
        scale2 = math.sqrt(2.0 / d_ff)
        self.W1 = rand_matrix(d_model, d_ff, scale1)
        self.b1 = [0.0] * d_ff
        self.W2 = rand_matrix(d_ff, d_model, scale2)
        self.b2 = [0.0] * d_model

    def forward(self, x):
        """x: (seq_len, d_model)"""
        h = linear_forward(x, self.W1, self.b1)  # (seq_len, d_ff)
        h = gelu_matrix(h)                       # nonlinear activation
        out = linear_forward(h, self.W2, self.b2) # (seq_len, d_model)
        return out


# ══════════════════════════════════════════════
#  3. TRANSFORMER BLOCK
# ══════════════════════════════════════════════

class TransformerBlock:
    """
    Полный блок трансформера:
      x -> LayerNorm -> MultiHeadSelfAttention -> + x (residual)
        -> LayerNorm -> FeedForward -> + prev (residual)

    Pre-Norm архитектура (как в GPT-2 / modern transformers).
    """

    def __init__(self, d_model, n_heads, d_ff, use_causal_mask=False):
        self.d_model = d_model

        if use_causal_mask:
            self.attention = CausalMaskedSelfAttention(d_model, n_heads)
        else:
            self.attention = MultiHeadSelfAttention(d_model, n_heads)

        self.ffn = FeedForward(d_model, d_ff)

        # Параметры LayerNorm (два: для attention и для FFN)
        self.gamma1 = [1.0] * d_model
        self.beta1 = [0.0] * d_model
        self.gamma2 = [1.0] * d_model
        self.beta2 = [0.0] * d_model

    def forward(self, x, return_intermediates=False):
        """
        x: (seq_len, d_model)
        """
        intermediates = {}

        # === Sub-layer 1: Multi-Head Self-Attention + Residual ===
        normed1 = layernorm_forward(x, self.gamma1, self.beta1)
        attn_out = self.attention.forward(normed1)
        h1 = add_matrices(x, attn_out)  # residual connection

        # === Sub-layer 2: Feed-Forward + Residual ===
        normed2 = layernorm_forward(h1, self.gamma2, self.beta2)
        ffn_out = self.ffn.forward(normed2)
        output = add_matrices(h1, ffn_out)  # residual connection

        if return_intermediates:
            intermediates['normed1'] = normed1
            intermediates['attn_out'] = attn_out
            intermediates['h1'] = h1
            intermediates['normed2'] = normed2
            intermediates['ffn_out'] = ffn_out
            return output, intermediates

        return output


# ══════════════════════════════════════════════
#  ДЕМОНСТРАЦИИ
# ══════════════════════════════════════════════

def demo_self_attention_residual():
    """Демо 1: Self-Attention + Residual Connection."""
    print("=" * 70)
    print("  ДЕМО 1: Multi-Head Self-Attention + Residual Connection")
    print("=" * 70)

    d_model = 8
    n_heads = 2
    seq_len = 4

    # Входные эмбеддинги (имитация токенов)
    x = [
        [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.4, -0.1],
        [0.1,  0.4, -0.5, 0.7, 0.3, -0.1, 0.2, 0.9],
        [-0.3, 0.6, 0.2, -0.4, 0.5, 0.8, -0.7, 0.1],
        [0.7, 0.2, -0.1, 0.3, -0.6, 0.4, 0.5, -0.3],
    ]

    print(f"\n  Вход (seq_len={seq_len}, d_model={d_model}):")
    print_matrix(x, "x")

    mha = MultiHeadSelfAttention(d_model, n_heads)
    attn_output = mha.forward(x)

    print(f"\n  Multi-Head Attention Output (n_heads={n_heads}):")
    print_matrix(attn_output, "attn_out")

    # Residual connection
    residual_out = add_matrices(x, attn_output)
    print(f"\n  After Residual Connection (x + attn_out):")
    print_matrix(residual_out, "output")

    # Проверка: residual должен сохранять масштаб входа
    norm_x = math.sqrt(sum(v**2 for row in x for v in row))
    norm_out = math.sqrt(sum(v**2 for row in residual_out for v in row))
    print(f"\n  L2 норма входа:      {norm_x:.4f}")
    print(f"  L2 норма residual:   {norm_out:.4f}")
    print(f"  Ratio:               {norm_out / norm_x:.4f}")
    print("  (Residual сохраняет информацию входа —_RATIO ≈ 1.0 ожидаем при малых весах)\n")


def demo_feedforward():
    """Демо 2: Feed-Forward Network."""
    print("=" * 70)
    print("  ДЕМО 2: Feed-Forward Network (FFN)")
    print("=" * 70)

    d_model = 8
    d_ff = 16
    seq_len = 3

    x = [
        [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.4, -0.1],
        [0.1,  0.4, -0.5, 0.7, 0.3, -0.1, 0.2, 0.9],
        [-0.3, 0.6, 0.2, -0.4, 0.5, 0.8, -0.7, 0.1],
    ]

    print(f"\n  Вход (seq_len={seq_len}, d_model={d_model}):")
    print_matrix(x, "x")

    ffn = FeedForward(d_model, d_ff)

    # Показываем слои FFN
    h = linear_forward(x, ffn.W1, ffn.b1)
    print(f"\n  After Linear 1 (d_ff={d_ff}):")
    print_matrix(h, "h (pre-activation)")

    h_gelu = gelu_matrix(h)
    print(f"\n  After GELU activation:")
    print_matrix(h_gelu, "h (post-GELU)")

    # Показываем sparse-подобную природу GELU (некоторые значения ≈ 0)
    n_near_zero = sum(1 for row in h_gelu for v in row if abs(v) < 0.01)
    n_total = d_ff * seq_len
    print(f"\n  GELU sparse-подобие: {n_near_zero}/{n_total} значений ≈ 0 (≈ {100*n_near_zero/n_total:.0f}%)")

    out = ffn.forward(x)
    print(f"\n  FFN Output (d_model={d_model}):")
    print_matrix(out, "output")
    print()


def demo_layernorm():
    """Демо 3: Layer Normalization."""
    print("=" * 70)
    print("  ДЕМО 3: Layer Normalization")
    print("=" * 70)

    d_model = 8
    seq_len = 3

    x = [
        [10.0, 2.0, -5.0, 3.0, 8.0, -2.0, 1.0, 0.5],
        [0.1,  0.2,  0.3, 0.4, 0.5,  0.6, 0.7, 0.8],
        [-3.0, 5.0,  2.0, 1.0, 0.0, -1.0, 4.0, 7.0],
    ]

    print(f"\n  Вход (seq_len={seq_len}, d_model={d_model}):")
    print_matrix(x, "x")

    # Без нормализации
    norms_before = []
    for row in x:
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        norms_before.append((mean, math.sqrt(var)))
    print(f"\n  Статистики ДО нормализации:")
    for i, (m, s) in enumerate(norms_before):
        print(f"    Токен {i}: mean={m:8.4f}, std={s:8.4f}")

    # С gamma=1, beta=0 (стандартный LayerNorm)
    gamma = [1.0] * d_model
    beta = [0.0] * d_model
    normed = layernorm_forward(x, gamma, beta)

    print(f"\n  После LayerNorm (gamma=1, beta=0):")
    print_matrix(normed, "normed")

    norms_after = []
    for row in normed:
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        norms_after.append((mean, math.sqrt(var)))
    print(f"\n  Статистики ПОСЛЕ нормализации:")
    for i, (m, s) in enumerate(norms_after):
        print(f"    Токен {i}: mean={m:12.8f}, std={s:12.8f}")

    # С обучаемыми gamma и beta
    gamma2 = [2.0, 1.5, 1.0, 0.5, 2.0, 1.0, 0.8, 1.2]
    beta2 = [0.1, -0.1, 0.0, 0.5, -0.2, 0.3, 0.0, 0.1]
    normed_custom = layernorm_forward(x, gamma2, beta2)
    print(f"\n  С кастомными gamma={gamma2[:3]}..., beta={beta2[:3]}...:")
    print_matrix(normed_custom, "normed_custom")
    print()


def demo_full_transformer_block():
    """Демо 4: Полный Transformer Block."""
    print("=" * 70)
    print("  ДЕМО 4: Полный Transformer Block (Pre-Norm Architecture)")
    print("=" * 70)

    d_model = 16
    n_heads = 4
    d_ff = 32
    seq_len = 5

    # Имитация эмбеддингов 5 токенов
    x = [
        [random.gauss(0, 0.5) for _ in range(d_model)] for _ in range(seq_len)
    ]

    print(f"\n  Архитектура:")
    print(f"    d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, seq_len={seq_len}")
    print(f"    Параметры: ~{d_model * d_model * 4 + d_model * d_ff * 2 + d_model * 4:,}")
    print(f"    (Q,K,V,O проекции + FFN два слоя + 2x LayerNorm gamma/beta)")

    print(f"\n  Вход (нормализованный для наглядности):")
    # Нормализуем вход для красивого вывода
    for i in range(seq_len):
        mean = sum(x[i]) / d_model
        var = sum((v - mean) ** 2 for v in x[i]) / d_model
        x[i] = [(v - mean) / math.sqrt(var + 1e-5) for v in x[i]]
    print_matrix(x, "x")

    block = TransformerBlock(d_model, n_heads, d_ff, use_causal_mask=True)
    output, intermediates = block.forward(x, return_intermediates=True)

    print(f"\n  --- Внутренний поток данных ---")

    print(f"\n  [1] LayerNorm → Multi-Head Self-Attention:")
    print(f"      normed1 нормализует вход для стабильности")
    print(f"      attention применяет causal маску (нижнетреугольная)")
    print_matrix(intermediates['normed1'][:2], "normed1 (first 2 tokens)")

    print(f"\n  [2] Residual Connection (x + attn_out):")
    print(f"      gradient flow: напрямую через skip connection")
    print_matrix(intermediates['h1'][:2], "h1 (first 2 tokens)")

    print(f"\n  [3] LayerNorm → FFN (GELU):")
    print(f"      FFN расширяет размерность: {d_model} → {d_ff} → {d_model}")
    print_matrix(intermediates['ffn_out'][:2], "ffn_out (first 2 tokens)")

    print(f"\n  [4] Residual Connection (h1 + ffn_out) → Final Output:")
    print_matrix(output[:2], "output (first 2 tokens)")

    # Анализ распределения
    all_vals = [v for row in output for v in row]
    mean_val = sum(all_vals) / len(all_vals)
    std_val = math.sqrt(sum((v - mean_val) ** 2 for v in all_vals) / len(all_vals))
    print(f"\n  Output stats: mean={mean_val:.6f}, std={std_val:.4f}")
    print(f"  (Распределение близко к N(0,1) благодаря LayerNorm — хорошо для обучения)")

    # Causal mask demo
    print(f"\n  --- Causal Mask Demo ---")
    print(f"  Causal mask (True = masked/future):")
    seq_demo = 6
    for i in range(seq_demo):
        row_str = "".join(["0" if j <= i else "1" for j in range(seq_demo)])
        print(f"    Токен {i}: [{row_str}]  (видит позиции 0..{i})")

    print(f"\n  Без causal mask любой токен видит ВСЕ остальные:")
    print(f"    Токен 0 видит: [0,1,2,3,4,5]  ← утечка будущего!")
    print(f"    Токен 0 видит: [0,1,2,3,4,5]  ← утечка будущего!")
    print()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  TRANSFORMER BLOCK — FROM SCRATCH (Pure Python)                    ║")
    print("║  random.seed(42) | No NumPy / No PyTorch / No Transformers        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    demo_self_attention_residual()
    demo_feedforward()
    demo_layernorm()
    demo_full_transformer_block()

    print("=" * 70)
    print("  Все демонстрации завершены.")
    print("  Компоненты: Multi-Head Attention, Feed-Forward (GELU),")
    print("  Layer Normalization, Residual Connections, Causal Masking")
    print("=" * 70)
