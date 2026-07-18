"""
Transformer Architecture — from scratch (no PyTorch/TensorFlow)

Implements:
  1. Sinusoidal Positional Encoding
  2. Multi-Head Self-Attention
  3. Transformer Encoder Block
  4. Transformer Decoder Block
  5. Full Encoder-Decoder Transformer
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────
# Matrix utilities (pure Python)
# ─────────────────────────────────────────────

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def rand_matrix(rows, cols, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def matmul(A, B):
    """(m×k) @ (k×n) -> (m×n)"""
    m, k1 = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k1 == k2, f"Shape mismatch: ({m}×{k1}) @ ({k2}×{n})"
    C = zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k1):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def transpose(M):
    return [list(row) for row in zip(*M)]


def softmax(M):
    """Row-wise softmax."""
    result = []
    for row in M:
        max_val = max(row)
        exps = [math.exp(v - max_val) for v in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result


def matmul_attention(Q, K, V):
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V"""
    dk = len(Q[0])
    scale = math.sqrt(dk)
    KT = transpose(K)
    scores = matmul(Q, KT)
    scaled = [[v / scale for v in row] for row in scores]
    weights = softmax(scaled)
    return matmul(weights, V)


def linear_transform(X, W):
    """X @ W — simple linear projection."""
    return matmul(X, W)


def relu_matrix(M):
    return [[max(0.0, v) for v in row] for row in M]


def add_matrices(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def layer_norm(X, eps=1e-6):
    """Per-row (sample-wise) layer normalization."""
    result = []
    for row in X:
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        std = math.sqrt(var + eps)
        result.append([(v - mean) / std for v in row])
    return result


# ─────────────────────────────────────────────
# 1. Sinusoidal Positional Encoding
# ─────────────────────────────────────────────

def sinusoidal_encoding(max_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos][i] = math.sin(angle)
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(angle)
    return pe


def add_positional_encoding(X, PE):
    """X (seq_len × d_model) + PE (seq_len × d_model)"""
    seq_len = len(X)
    return add_matrices(X, PE[:seq_len])


# ─────────────────────────────────────────────
# 2. Multi-Head Self-Attention
# ─────────────────────────────────────────────

def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads):
    seq_len = len(X)
    d_model = len(X[0])
    d_k = d_model // n_heads
    heads = []

    for h in range(n_heads):
        # Each head gets its own slice of the weight matrices
        offset_q = h * d_k
        # Extract sub-matrices for this head
        Qh = [[row[offset_q + j] for j in range(d_k)] for row in matmul(X, W_q)]
        Kh = [[row[offset_q + j] for j in range(d_k)] for row in matmul(X, W_k)]
        Vh = [[row[offset_q + j] for j in range(d_k)] for row in matmul(X, W_v)]
        heads.append(matmul_attention(Qh, Kh, Vh))

    # Concatenate heads: each head is (seq_len × d_k) -> concat along d_k
    concatenated = zeros(seq_len, d_model)
    for i in range(seq_len):
        pos = 0
        for h in range(n_heads):
            for j in range(d_k):
                concatenated[i][pos] = heads[h][i][j]
                pos += 1

    return linear_transform(concatenated, W_o)


# ─────────────────────────────────────────────
# 3. Transformer Encoder Block
# ─────────────────────────────────────────────

class TransformerEncoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        rng = random.Random(seed)
        scale_q = math.sqrt(2.0 / d_model)
        scale_ff = math.sqrt(2.0 / d_ff)

        self.W_q = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]

        self.W_ff1 = [[rng.gauss(0, scale_ff) for _ in range(d_ff)] for _ in range(d_model)]
        self.b_ff1 = [0.0] * d_ff
        self.W_ff2 = [[rng.gauss(0, scale_ff) for _ in range(d_model)] for _ in range(d_ff)]
        self.b_ff2 = [0.0] * d_model

        self.n_heads = n_heads

    def forward(self, X):
        # Multi-head self-attention + residual + layer norm
        attn_out = multi_head_attention(X, self.W_q, self.W_k, self.W_v, self.W_o, self.n_heads)
        X1 = layer_norm(add_matrices(X, attn_out))

        # Feed-forward + residual + layer norm
        ff_hidden = matmul(X1, self.W_ff1)
        ff_hidden = [[v + b for v, b in zip(row, self.b_ff1)] for row in ff_hidden]
        ff_hidden = relu_matrix(ff_hidden)
        ff_out = matmul(ff_hidden, self.W_ff2)
        ff_out = [[v + b for v, b in zip(row, self.b_ff2)] for row in ff_out]
        X2 = layer_norm(add_matrices(X1, ff_out))

        return X2


# ─────────────────────────────────────────────
# 4. Transformer Decoder Block
# ─────────────────────────────────────────────

class TransformerDecoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        rng = random.Random(seed)
        scale_q = math.sqrt(2.0 / d_model)
        scale_ff = math.sqrt(2.0 / d_ff)

        # Masked self-attention
        self.W_q1 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k1 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v1 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o1 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]

        # Cross-attention
        self.W_q2 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k2 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v2 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o2 = [[rng.gauss(0, scale_q) for _ in range(d_model)] for _ in range(d_model)]

        # Feed-forward
        self.W_ff1 = [[rng.gauss(0, scale_ff) for _ in range(d_ff)] for _ in range(d_model)]
        self.b_ff1 = [0.0] * d_ff
        self.W_ff2 = [[rng.gauss(0, scale_ff) for _ in range(d_model)] for _ in range(d_ff)]
        self.b_ff2 = [0.0] * d_model

        self.n_heads = n_heads

    def masked_self_attention(self, X):
        """Self-attention with causal (look-ahead) mask."""
        seq_len = len(X)
        d_model = len(X[0])
        d_k = d_model // self.n_heads
        heads = []

        for h in range(self.n_heads):
            offset = h * d_k
            Qh = [[row[offset + j] for j in range(d_k)] for row in matmul(X, self.W_q1)]
            Kh = [[row[offset + j] for j in range(d_k)] for row in matmul(X, self.W_k1)]
            Vh = [[row[offset + j] for j in range(d_k)] for row in matmul(X, self.W_v1)]

            # Compute attention scores with mask
            KT = transpose(Kh)
            scores = matmul(Qh, KT)
            scale = math.sqrt(d_k)
            scaled = [[v / scale for v in row] for row in scores]

            # Apply causal mask: position i can only attend to positions <= i
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    scaled[i][j] = float('-inf')

            weights = softmax(scaled)
            heads.append(matmul(weights, Vh))

        concatenated = zeros(seq_len, d_model)
        for i in range(seq_len):
            pos = 0
            for h in range(self.n_heads):
                for j in range(d_k):
                    concatenated[i][pos] = heads[h][i][j]
                    pos += 1

        return linear_transform(concatenated, self.W_o1)

    def cross_attention(self, X, context):
        """Cross-attention: Q from decoder, K/V from encoder output."""
        seq_len = len(X)
        d_model = len(X[0])
        d_k = d_model // self.n_heads
        heads = []

        for h in range(self.n_heads):
            offset = h * d_k
            Qh = [[row[offset + j] for j in range(d_k)] for row in matmul(X, self.W_q2)]
            Kh = [[row[offset + j] for j in range(d_k)] for row in matmul(context, self.W_k2)]
            Vh = [[row[offset + j] for j in range(d_k)] for row in matmul(context, self.W_v2)]
            heads.append(matmul_attention(Qh, Kh, Vh))

        concatenated = zeros(seq_len, d_model)
        for i in range(seq_len):
            pos = 0
            for h in range(self.n_heads):
                for j in range(d_k):
                    concatenated[i][pos] = heads[h][i][j]
                    pos += 1

        return linear_transform(concatenated, self.W_o2)

    def forward(self, X, encoder_output):
        # Masked self-attention + residual + layer norm
        attn1 = self.masked_self_attention(X)
        X1 = layer_norm(add_matrices(X, attn1))

        # Cross-attention + residual + layer norm
        attn2 = self.cross_attention(X1, encoder_output)
        X2 = layer_norm(add_matrices(X1, attn2))

        # Feed-forward + residual + layer norm
        ff_hidden = matmul(X2, self.W_ff1)
        ff_hidden = [[v + b for v, b in zip(row, self.b_ff1)] for row in ff_hidden]
        ff_hidden = relu_matrix(ff_hidden)
        ff_out = matmul(ff_hidden, self.W_ff2)
        ff_out = [[v + b for v, b in zip(row, self.b_ff2)] for row in ff_out]
        X3 = layer_norm(add_matrices(X2, ff_out))

        return X3


# ─────────────────────────────────────────────
# 5. Full Transformer
# ─────────────────────────────────────────────

class Transformer:
    def __init__(self, src_vocab, tgt_vocab, d_model=32, n_heads=4,
                 d_ff=64, n_enc_layers=2, n_dec_layers=2):
        self.d_model = d_model
        self.src_emb = rand_matrix(src_vocab, d_model, scale=math.sqrt(2.0 / d_model))
        self.tgt_emb = rand_matrix(tgt_vocab, d_model, scale=math.sqrt(2.0 / d_model))

        enc_seeds = list(range(100, 100 + n_enc_layers * 10, 10))
        dec_seeds = list(range(200, 200 + n_dec_layers * 10, 10))

        self.encoder_layers = [
            TransformerEncoderBlock(d_model, n_heads, d_ff, seed=s)
            for s in enc_seeds
        ]
        self.decoder_layers = [
            TransformerDecoderBlock(d_model, n_heads, d_ff, seed=s)
            for s in dec_seeds
        ]

    def encode(self, src_indices):
        """Encode source token indices -> encoder output."""
        seq_len = len(src_indices)
        # Embedding + positional encoding
        embedded = [self.src_emb[idx][:] for idx in src_indices]
        PE = sinusoidal_encoding(seq_len + 50, self.d_model)
        X = add_positional_encoding(embedded, PE)

        for layer in self.encoder_layers:
            X = layer.forward(X)
        return X

    def decode(self, tgt_indices, encoder_output):
        """Decode target token indices given encoder output."""
        seq_len = len(tgt_indices)
        embedded = [self.tgt_emb[idx][:] for idx in tgt_indices]
        PE = sinusoidal_encoding(seq_len + 50, self.d_model)
        X = add_positional_encoding(embedded, PE)

        for layer in self.decoder_layers:
            X = layer.forward(X, encoder_output)
        return X

    def forward(self, src_indices, tgt_indices):
        """Full encoder-decoder forward pass."""
        enc_out = self.encode(src_indices)
        dec_out = self.decode(tgt_indices, enc_out)
        # Project to target vocabulary
        logits = matmul(dec_out, transpose(self.tgt_emb))
        return logits


# ─────────────────────────────────────────────
# Demo 1: Positional Encoding
# ─────────────────────────────────────────────

def demo_positional_encoding():
    print("=" * 60)
    print("DEMO 1: Sinusoidal Positional Encoding")
    print("=" * 60)

    max_len = 10
    d_model = 8
    PE = sinusoidal_encoding(max_len, d_model)

    print(f"\nShape: ({max_len}, {d_model})")
    print("\nFirst 4 positions (first 6 dimensions):")
    print(f"  {'pos':>3s}  {'dim0':>8s} {'dim1':>8s} {'dim2':>8s} {'dim3':>8s} {'dim4':>8s} {'dim5':>8s}")
    for pos in range(4):
        vals = [f"{PE[pos][d]:.4f}" for d in range(6)]
        print(f"  {pos:3d}  {' '.join(vals)}")

    # Verify properties
    print("\nProperty checks:")
    # PE(0, 0) = sin(0) = 0
    print(f"  PE(0, 0) = {PE[0][0]:.6f}  (expected: 0.0)")
    # PE(0, 1) = cos(0) = 1
    print(f"  PE(0, 1) = {PE[0][1]:.6f}  (expected: 1.0)")
    # PE(pos, 0) = sin(pos) — check monotonicity in first half-cycle
    print(f"  PE(1, 0) = {PE[1][0]:.6f}  (expected: sin(1) ≈ 0.8415)")
    print(f"  PE(2, 0) = {PE[2][0]:.6f}  (expected: sin(2) ≈ 0.9093)")

    # Show effect on embeddings
    print("\nDemonstration: adding PE to random embeddings")
    random.seed(42)
    embeddings = rand_matrix(4, d_model, scale=0.1)
    encoded = add_positional_encoding(embeddings, PE)

    norm_before = [math.sqrt(sum(v ** 2 for v in row)) for row in embeddings]
    norm_after = [math.sqrt(sum(v ** 2 for v in row)) for row in encoded]
    print(f"  Norm before PE: {[f'{n:.4f}' for n in norm_before]}")
    print(f"  Norm after PE:  {[f'{n:.4f}' for n in norm_after]}")
    print("  → PE adds positional signal without destroying magnitude")


# ─────────────────────────────────────────────
# Demo 2: Encoder Block
# ─────────────────────────────────────────────

def demo_encoder_block():
    print("\n" + "=" * 60)
    print("DEMO 2: Transformer Encoder Block")
    print("=" * 60)

    seq_len = 4
    d_model = 16
    n_heads = 4
    d_ff = 32

    random.seed(42)
    block = TransformerEncoderBlock(d_model, n_heads, d_ff, seed=42)

    # Input: random sequence
    X = rand_matrix(seq_len, d_model, scale=0.1)
    print(f"\nInput shape: ({seq_len}, {d_model})")
    print(f"Configuration: n_heads={n_heads}, d_ff={d_ff}")

    X_norm = layer_norm(X)
    norms_before = [math.sqrt(sum(v ** 2 for v in row)) for row in X_norm]

    output = block.forward(X)
    norms_after = [math.sqrt(sum(v ** 2 for v in row)) for row in output]

    print(f"\nOutput shape: ({len(output)}, {len(output[0])})")
    print(f"Input norms (after LN):  {[f'{n:.4f}' for n in norms_before]}")
    print(f"Output norms (after LN): {[f'{n:.4f}' for n in norms_after]}")

    # Verify residual connection preserved scale
    max_ratio = max(o / b if b > 0 else 0 for o, b in zip(norms_after, norms_before))
    print(f"  Max norm ratio (out/in): {max_ratio:.4f}")
    print("  → Residual connections keep norm in reasonable range")

    # Show internal structure
    print("\nInternal flow:")
    print("  1. Multi-head self-attention (4 heads × dim 4)")
    print("  2. Residual add + LayerNorm")
    print("  3. Feed-forward (16 → 32 → 16, ReLU)")
    print("  4. Residual add + LayerNorm")


# ─────────────────────────────────────────────
# Demo 3: Decoder Block
# ─────────────────────────────────────────────

def demo_decoder_block():
    print("\n" + "=" * 60)
    print("DEMO 3: Transformer Decoder Block")
    print("=" * 60)

    src_len = 4
    tgt_len = 3
    d_model = 16
    n_heads = 4
    d_ff = 32

    random.seed(42)

    # First create encoder output
    enc_block = TransformerEncoderBlock(d_model, n_heads, d_ff, seed=10)
    src = rand_matrix(src_len, d_model, scale=0.1)
    encoder_output = enc_block.forward(src)
    print(f"Encoder output shape: ({src_len}, {d_model})")

    # Decoder block
    dec_block = TransformerDecoderBlock(d_model, n_heads, d_ff, seed=42)
    tgt = rand_matrix(tgt_len, d_model, scale=0.1)
    print(f"Decoder input shape:  ({tgt_len}, {d_model})")

    output = dec_block.forward(tgt, encoder_output)
    print(f"Decoder output shape: ({len(output)}, {len(output[0])})")

    print("\nInternal flow:")
    print("  1. Masked multi-head self-attention (causal mask)")
    print("  2. Residual add + LayerNorm")
    print("  3. Cross-attention (Q=decoder, K/V=encoder)")
    print("  4. Residual add + LayerNorm")
    print("  5. Feed-forward (16 → 32 → 16, ReLU)")
    print("  6. Residual add + LayerNorm")

    # Demonstrate causal masking
    print("\nCausal mask demonstration (tgt_len=3):")
    mask = zeros(tgt_len, tgt_len)
    for i in range(tgt_len):
        for j in range(tgt_len):
            mask[i][j] = 1 if j <= i else 0
    for i in range(tgt_len):
        print(f"  Position {i} can attend to: {[j for j in range(tgt_len) if mask[i][j]]}")


# ─────────────────────────────────────────────
# Demo 4: Full Transformer
# ─────────────────────────────────────────────

def demo_full_transformer():
    print("\n" + "=" * 60)
    print("DEMO 4: Full Encoder-Decoder Transformer")
    print("=" * 60)

    src_vocab = 10
    tgt_vocab = 10
    d_model = 16
    n_heads = 4
    d_ff = 32

    random.seed(42)
    model = Transformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_enc_layers=2,
        n_dec_layers=2,
    )

    print(f"\nVocabulary: src={src_vocab}, tgt={tgt_vocab}")
    print(f"Model dimensions: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"Layers: 2 encoder + 2 decoder")

    # Source: "I love cats" (indices)
    src = [1, 4, 7]
    # Target shifted right: "<bos> the cats" (indices)
    tgt = [0, 3, 7]

    print(f"\nSource tokens (indices): {src}")
    print(f"Target tokens (indices): {tgt}")

    logits = model.forward(src, tgt)
    print(f"\nOutput logits shape: ({len(logits)}, {len(logits[0])})")

    # Convert to probabilities
    probs = softmax(logits)
    print(f"\nOutput probability shape: ({len(probs)}, {len(probs[0])})")

    # Show predictions
    print("\nPredictions for each target position:")
    for i, (prob_row, tgt_idx) in enumerate(zip(probs, tgt)):
        pred_idx = prob_row.index(max(prob_row))
        conf = max(prob_row)
        print(f"  Position {i}: input={tgt_idx}, predicted={pred_idx} (conf={conf:.4f})")

    # Compare with target
    correct = sum(1 for p_row, t in zip(probs, tgt) if p_row.index(max(p_row)) == t)
    print(f"\nExact match: {correct}/{len(tgt)} positions")

    # Parameter count estimate
    param_count = 0
    # Embeddings
    param_count += src_vocab * d_model + tgt_vocab * d_model
    # Encoder layers
    for _ in range(2):
        param_count += 4 * d_model * d_model  # Q, K, V, O
        param_count += d_model * d_ff + d_ff  # FF1
        param_count += d_ff * d_model + d_model  # FF2
    # Decoder layers
    for _ in range(2):
        param_count += 4 * d_model * d_model  # Self-attn Q,K,V,O
        param_count += 4 * d_model * d_model  # Cross-attn Q,K,V,O
        param_count += d_model * d_ff + d_ff  # FF1
        param_count += d_ff * d_model + d_model  # FF2

    print(f"\nEstimated parameters: ~{param_count:,}")

    # Verify forward pass consistency
    random.seed(42)
    model2 = Transformer(
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_enc_layers=2, n_dec_layers=2,
    )
    logits2 = model2.forward(src, tgt)
    all_match = all(
        abs(a - b) < 1e-10
        for row1, row2 in zip(logits, logits2)
        for a, b in zip(row1, row2)
    )
    print(f"\nDeterministic (same seed): {all_match}")


# ─────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo_positional_encoding()
    demo_encoder_block()
    demo_decoder_block()
    demo_full_transformer()
    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)
