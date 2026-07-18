"""
Attention Mechanisms — Scaled Dot-Product, Multi-Head, Self-Attention
with visualization of attention weights.
"""

import numpy as np
import random

random.seed(42)
np.random.seed(42)


# ============================================================
# 1. Scaled Dot-Product Attention
# ============================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: numpy arrays (..., seq_len, d_k)
    Returns: output (..., seq_len, d_v), attention_weights (..., seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = np.matmul(weights, V)
    return output, weights


# ============================================================
# 2. Multi-Head Attention
# ============================================================

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, X):
        """X: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)"""
        batch, seq_len, _ = X.shape
        X = X.reshape(batch, seq_len, self.num_heads, self.d_k)
        return X.transpose(0, 2, 1, 3)

    def forward(self, X, mask=None):
        """X: (batch, seq_len, d_model) -> output same shape, attention_weights"""
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output, weights = scaled_dot_product_attention(Q, K, V, mask)

        batch, _, seq_len, d_k = attn_output.shape
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        output = attn_output @ self.W_O

        return output, weights


# ============================================================
# 3. Self-Attention (single layer, for a sentence)
# ============================================================

def simple_self_attention(embeddings, d_k=None):
    """
    embeddings: (seq_len, d_embed) numpy array
    Returns: context (seq_len, d_k), weights (seq_len, seq_len)
    """
    seq_len, d_embed = embeddings.shape
    if d_k is None:
        d_k = d_embed

    W_Q = np.random.randn(d_embed, d_k) * 0.1
    W_K = np.random.randn(d_embed, d_k) * 0.1
    W_V = np.random.randn(d_embed, d_k) * 0.1

    Q = embeddings @ W_Q
    K = embeddings @ W_K
    V = embeddings @ W_V

    scores = Q @ K.T / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context = weights @ V

    return context, weights


# ============================================================
# 4. Visualization helpers (text-based)
# ============================================================

def print_matrix(matrix, row_labels=None, col_labels=None, title="", fmt=".3f"):
    """Print a matrix as a formatted table."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    n_rows, n_cols = matrix.shape

    if col_labels:
        header = " " * 8
        for label in col_labels:
            header += f"{label:>10}"
        print(header)

    for i in range(n_rows):
        if row_labels:
            row_str = f"  {row_labels[i]:>8}"
        else:
            row_str = f"  {i:>8}"
        for j in range(n_cols):
            row_str += f"{matrix[i, j]:>10{fmt}}"
        print(row_str)
    print()


def print_attention_bar(matrix, tokens_i, tokens_j, title=""):
    """Print attention weights as horizontal bars."""
    if title:
        print(f"\n--- {title} ---")

    for i, token_i in enumerate(tokens_i):
        print(f"\n  '{token_i}' attends to:")
        for j, token_j in enumerate(tokens_j):
            bar_len = int(matrix[i, j] * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"    '{token_j}': {bar} {matrix[i, j]:.4f}")


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================
# Demo 1: Scaled Dot-Product Attention
# ============================================================

def demo_scaled_dot_product():
    print("\n" + "#" * 60)
    print("# Demo 1: Scaled Dot-Product Attention")
    print("#" * 60)

    seq_len, d_k = 5, 8
    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nInput shapes:")
    print(f"  Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    print(f"\nAttention weights (row-normalized, each row sums to 1):")
    w = weights[0]
    row_sums = np.sum(w, axis=-1)
    print(f"  Row sums: {row_sums}")

    print_matrix(w, title="Attention Weight Matrix",
                 col_labels=[f"K{i}" for i in range(seq_len)])

    print("\nWith causal mask (lower triangular):")
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print_matrix(weights_masked[0], title="Masked Attention Weights",
                 col_labels=[f"K{i}" for i in range(seq_len)])

    zero_upper = np.sum(weights_masked[0][np.triu_indices(seq_len, k=1)])
    print(f"  Sum of upper-triangle weights: {zero_upper:.10f} (should be ~0)")

    print("\n[OK] Scaled dot-product attention works correctly.")


# ============================================================
# Demo 2: Multi-Head Attention
# ============================================================

def demo_multi_head():
    print("\n" + "#" * 60)
    print("# Demo 2: Multi-Head Attention")
    print("#" * 60)

    batch, seq_len, d_model, num_heads = 1, 4, 16, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    X = np.random.randn(batch, seq_len, d_model)
    print(f"\nInput shape: {X.shape} (batch={batch}, seq={seq_len}, d_model={d_model})")
    print(f"Number of heads: {num_heads}, d_k per head: {d_model // num_heads}")

    output, weights = mha.forward(X)

    print(f"Output shape: {output.shape}")
    print(f"Per-head attention weights shape: {weights.shape}")

    print(f"\nPer-head attention (head 0):")
    print_matrix(weights[0, 0], title="Head 0 Attention",
                 col_labels=[f"pos{i}" for i in range(seq_len)])

    print(f"Per-head attention (head 1):")
    print_matrix(weights[0, 1], title="Head 1 Attention",
                 col_labels=[f"pos{i}" for i in range(seq_len)])

    print("\nEach head learns different attention patterns!")

    print("\nWith mask:")
    mask = np.tril(np.ones((seq_len, seq_len)))
    output_m, weights_m = mha.forward(X, mask=mask)
    print(f"Masked output shape: {output_m.shape}")
    upper_sum = np.sum(weights_m[0, 0][np.triu_indices(seq_len, k=1)])
    print(f"  Upper-triangle sum (head 0): {upper_sum:.10f}")

    print("\n[OK] Multi-head attention works correctly.")


# ============================================================
# Demo 3: Self-Attention on a Sentence
# ============================================================

def demo_self_attention_sentence():
    print("\n" + "#" * 60)
    print("# Demo 3: Self-Attention on a Sentence")
    print("#" * 60)

    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    d_embed = 6

    np.random.seed(42)
    vocab = {t: i for i, t in enumerate(tokens)}
    embeddings = np.random.randn(len(tokens), d_embed) * 0.3

    print(f"\nSentence: '{' '.join(tokens)}'")
    print(f"Embedding dim: {d_embed}")
    print(f"Number of tokens: {len(tokens)}")

    np.random.seed(42)
    context, weights = simple_self_attention(embeddings)

    print(f"\nAttention weight matrix ({len(tokens)}x{len(tokens)}):")
    print_matrix(weights, row_labels=tokens, col_labels=tokens,
                 title="Self-Attention Weights")

    print("Which token attends most to which:")
    for i, t in enumerate(tokens):
        most_attended = tokens[np.argmax(weights[i])]
        print(f"  '{t}' -> most attends to '{most_attended}' (weight={weights[i, np.argmax(weights[i])]:.4f})")

    print("\nToken similarity patterns (top-2 attended):")
    for i, t in enumerate(tokens):
        top2_idx = np.argsort(weights[i])[-2:][::-1]
        pairs = [(tokens[j], weights[i, j]) for j in top2_idx]
        print(f"  '{t}': {pairs[0][0]} ({pairs[0][1]:.3f}), {pairs[1][0]} ({pairs[1][1]:.3f})")

    print("\n[OK] Self-attention on sentence works correctly.")


# ============================================================
# Demo 4: Visualization of Attention Matrix
# ============================================================

def demo_visualization():
    print("\n" + "#" * 60)
    print("# Demo 4: Visualization of Attention Matrix")
    print("#" * 60)

    np.random.seed(42)
    tokens = ["I", "love", "deep", "learning", "!"]
    d_embed = 8

    embeddings = np.random.randn(len(tokens), d_embed) * 0.3

    context, weights = simple_self_attention(embeddings)

    print(f"\nSentence: '{' '.join(tokens)}'")
    print_matrix(weights, row_labels=tokens, col_labels=tokens,
                 title="Full Attention Matrix")

    print("\n--- Text-based heatmap ---")
    symbols = [" ", "░", "▒", "▓", "█"]
    print(f"\n  Legend: {' '.join(symbols)} = increasing weight")
    print()

    max_w = np.max(weights)
    for i, t_i in enumerate(tokens):
        row_str = f"  {t_i:>8} |"
        for j, t_j in enumerate(tokens):
            val = weights[i, j] / max_w
            idx = min(int(val * (len(symbols) - 1)), len(symbols) - 1)
            row_str += f" {symbols[idx]} "
        print(row_str)

    print(f"\n  Column labels:")
    print(f"  {'':>8} |", end="")
    for t in tokens:
        print(f" {t[0]:>2}", end="")
    print()

    print("\n--- Attention distribution per token ---")
    print_attention_bar(weights, tokens, tokens,
                        title="Horizontal bar chart")

    print("\n--- Comparison: two different random seeds ---")
    np.random.seed(0)
    emb_a = np.random.randn(len(tokens), d_embed) * 0.3
    _, w_a = simple_self_attention(emb_a)

    np.random.seed(1)
    emb_b = np.random.randn(len(tokens), d_embed) * 0.3
    _, w_b = simple_self_attention(emb_b)

    print("\n  Seed=0 attention (row 0, token 'I'):")
    for j, t in enumerate(tokens):
        bar_len = int(w_a[0, j] * 30)
        print(f"    {t:>8}: {'█' * bar_len} {w_a[0, j]:.4f}")

    print("\n  Seed=1 attention (row 0, token 'I'):")
    for j, t in enumerate(tokens):
        bar_len = int(w_b[0, j] * 30)
        print(f"    {t:>8}: {'█' * bar_len} {w_b[0, j]:.4f}")

    print("\nDifferent initializations lead to different attention patterns.")
    print("\n[OK] Visualization demo complete.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Attention Mechanisms — Complete Implementation")
    print("  Scaled Dot-Product | Multi-Head | Self-Attention")
    print("=" * 60)

    demo_scaled_dot_product()
    demo_multi_head()
    demo_self_attention_sentence()
    demo_visualization()

    print("\n" + "=" * 60)
    print("  All demos completed successfully.")
    print("=" * 60)
