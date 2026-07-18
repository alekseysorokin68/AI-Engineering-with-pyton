"""
101 — GPT Architecture from Scratch (Pure Python)
==================================================
Decoder-only Transformer with causal masking and text generation.
No external dependencies — only the standard library.
"""

import math
import random

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

random.seed(42)


def softmax(vec):
    """Numerically-stable softmax over a list of floats."""
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


def mat_vec(mat, vec):
    """Matrix-vector multiplication: mat (rows×cols) × vec (cols,) → (rows,)."""
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


def mat_mul(a, b):
    """Matrix multiplication: a (m×k) × b (k×n) → (m×n)."""
    k = len(a[0])
    n = len(b[0])
    result = [[0.0] * n for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(n):
            result[i][j] = sum(a[i][k2] * b[k2][j] for k2 in range(k))
    return result


def transpose(mat):
    """Transpose a matrix."""
    return [list(row) for row in zip(*mat)]


def add_matrices(a, b):
    """Element-wise addition of two matrices."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization: x, gamma, beta are vectors of same length."""
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [gamma[i] * ((x[i] - mean) / std) + beta[i] for i in range(len(x))]


def gelu(x):
    """GELU activation (approximate)."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def top_k_indices(logits, k):
    """Return indices of top-k values in logits."""
    indexed = list(enumerate(logits))
    indexed.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in indexed[:k]]


# ──────────────────────────────────────────────
#  Tokenizer (word-level, minimal)
# ──────────────────────────────────────────────

class SimpleTokenizer:
    """Minimal word-level tokenizer with special tokens."""

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, vocab=None):
        if vocab is None:
            vocab = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.vocab = list(vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

    def encode(self, text):
        return [self.token_to_id.get(w, self.token_to_id[self.UNK]) for w in text.split()]

    def decode(self, ids):
        return " ".join(self.id_to_token.get(i, self.UNK) for i in ids if i not in (
            self.token_to_id[self.PAD], self.token_to_id[self.BOS], self.token_to_id[self.EOS]
        ))

    def __len__(self):
        return len(self.vocab)


# ──────────────────────────────────────────────
#  Causal Mask
# ──────────────────────────────────────────────

def create_causal_mask(seq_len):
    """
    Upper-triangular mask: mask[i][j] = -inf if j > i, else 0.
    Ensures each position can only attend to itself and earlier positions.
    """
    NEG_INF = float("-inf")
    mask = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            row.append(NEG_INF if j > i else 0.0)
        mask.append(row)
    return mask


# ──────────────────────────────────────────────
#  Scaled Dot-Product Attention
# ──────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: lists of vectors (each a list of floats).
    mask: optional 2-D matrix (seq_q × seq_k) with 0 or -inf.
    Returns: (output_vectors, attention_weights).
    """
    d_k = len(Q[0])
    seq_q = len(Q)
    seq_k = len(K)

    # QK^T / sqrt(d_k)
    scores = []
    for i in range(seq_q):
        row = []
        for j in range(seq_k):
            s = sum(Q[i][t] * K[j][t] for t in range(d_k)) / math.sqrt(d_k)
            if mask is not None:
                s += mask[i][j]
            row.append(s)
        scores.append(row)

    # Softmax per query row
    attn_weights = [softmax(row) for row in scores]

    # Weighted sum of V
    d_v = len(V[0])
    output = []
    for i in range(seq_q):
        out_vec = [0.0] * d_v
        for j in range(seq_k):
            for t in range(d_v):
                out_vec[t] += attn_weights[i][j] * V[j][t]
        output.append(out_vec)

    return output, attn_weights


# ──────────────────────────────────────────────
#  Multi-Head Attention
# ──────────────────────────────────────────────

class MultiHeadAttention:
    """Multi-head self-attention with causal masking."""

    def __init__(self, d_model, n_heads, seed=42):
        rng = random.Random(seed)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        # Projection matrices (d_model × d_model)
        self.W_q = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_model)]

    def split_heads(self, x):
        """Split a list of vectors into n_heads groups."""
        heads = [[] for _ in range(self.n_heads)]
        for vec in x:
            for h in range(self.n_heads):
                start = h * self.d_k
                heads[h].append(vec[start:start + self.d_k])
        return heads

    def merge_heads(self, heads):
        """Merge n_heads groups back into a list of vectors."""
        merged = []
        for i in range(len(heads[0])):
            vec = []
            for h in range(self.n_heads):
                vec.extend(heads[h][i])
            merged.append(vec)
        return merged

    def forward(self, x, mask=None):
        """x: list of vectors. Returns list of output vectors."""
        Q = [mat_vec(self.W_q, vec) for vec in x]
        K = [mat_vec(self.W_k, vec) for vec in x]
        V = [mat_vec(self.W_v, vec) for vec in x]

        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)

        all_heads = []
        for h in range(self.n_heads):
            out, _ = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], mask)
            all_heads.append(out)

        merged = self.merge_heads(all_heads)
        output = [mat_vec(self.W_o, vec) for vec in merged]
        return output


# ──────────────────────────────────────────────
#  Feed-Forward Network
# ──────────────────────────────────────────────

class FeedForward:
    """Position-wise feed-forward: d_model → 4*d_model → d_model with GELU."""

    def __init__(self, d_model, seed=42):
        rng = random.Random(seed)
        d_ff = 4 * d_model
        self.W1 = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(d_ff)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[rng.gauss(0, 0.02) for _ in range(d_ff)] for _ in range(d_model)]
        self.b2 = [0.0] * d_model

    def forward(self, x):
        """x: list of vectors. Returns list of vectors."""
        output = []
        for vec in x:
            # Layer 1
            h = [mat_vec(self.W1, vec)[i] + self.b1[i] for i in range(len(self.b1))]
            h = [gelu(v) for v in h]
            # Layer 2
            out = [mat_vec(self.W2, h)[i] + self.b2[i] for i in range(len(self.b2))]
            output.append(out)
        return output


# ──────────────────────────────────────────────
#  Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock:
    """Single decoder block: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, d_model, n_heads, seed=42):
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, seed + 1)
        rng = random.Random(seed + 2)
        self.ln1_gamma = [1.0] * d_model
        self.ln1_beta = [0.0] * d_model
        self.ln2_gamma = [1.0] * d_model
        self.ln2_beta = [0.0] * d_model

    def forward(self, x, mask=None):
        # Sub-layer 1: Multi-Head Attention + residual
        normed = [layer_norm(v, self.ln1_gamma, self.ln1_beta) for v in x]
        attn_out = self.attn.forward(normed, mask)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[0]))] for i in range(len(x))]

        # Sub-layer 2: Feed-Forward + residual
        normed = [layer_norm(v, self.ln2_gamma, self.ln2_beta) for v in x]
        ffn_out = self.ffn.forward(normed)
        x = [[x[i][j] + ffn_out[i][j] for j in range(len(x[0]))] for i in range(len(x))]

        return x


# ──────────────────────────────────────────────
#  GPT Model (Decoder-only Transformer)
# ──────────────────────────────────────────────

class GPT:
    """
    Minimal GPT: token + positional embeddings, N transformer blocks, LM head.
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=128, seed=42):
        rng = random.Random(seed)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding: vocab_size × d_model
        self.token_emb = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(vocab_size)]

        # Positional embedding: max_seq_len × d_model
        self.pos_emb = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(max_seq_len)]

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, seed=seed + i * 10) for i in range(n_layers)]

        # Final layer norm
        self.ln_gamma = [1.0] * d_model
        self.ln_beta = [0.0] * d_model

        # LM head (weight-tied with token embedding is ideal, but we keep it separate for simplicity)
        self.lm_head = [[rng.gauss(0, 0.02) for _ in range(d_model)] for _ in range(vocab_size)]

    def forward(self, token_ids, mask=None):
        """
        token_ids: list of int (token indices).
        Returns logits: list of vectors (seq_len × vocab_size).
        """
        seq_len = len(token_ids)

        # Embeddings
        x = []
        for i in range(seq_len):
            te = self.token_emb[token_ids[i]]
            pe = self.pos_emb[i]
            combined = [te[j] + pe[j] for j in range(self.d_model)]
            x.append(combined)

        # Build causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len)

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer norm
        x = [layer_norm(v, self.ln_gamma, self.ln_beta) for v in x]

        # LM head → logits
        logits = [mat_vec(self.lm_head, vec) for vec in x]
        return logits


# ──────────────────────────────────────────────
#  Text Generation
# ──────────────────────────────────────────────

def greedy_decode(model, tokenizer, prompt_ids, max_new_tokens=10):
    """Greedy: always pick the token with highest logit."""
    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model.forward(generated)
        next_logits = logits[-1]
        next_id = max(range(len(next_logits)), key=lambda i: next_logits[i])
        generated.append(next_id)
        if next_id == tokenizer.token_to_id[tokenizer.EOS]:
            break
    return generated


def sample_decode(model, tokenizer, prompt_ids, max_new_tokens=10, temperature=1.0, top_k=0):
    """Sampling: temperature scaling + optional top-k filtering."""
    generated = list(prompt_ids)
    rng = random.Random(42)
    for _ in range(max_new_tokens):
        logits = model.forward(generated)
        next_logits = logits[-1]

        # Temperature scaling
        if temperature != 1.0:
            next_logits = [v / temperature for v in next_logits]

        # Top-k filtering
        if top_k > 0:
            indices = top_k_indices(next_logits, top_k)
            filtered = [float("-inf")] * len(next_logits)
            for idx in indices:
                filtered[idx] = next_logits[idx]
            next_logits = filtered

        probs = softmax(next_logits)
        # Sample from distribution
        r = rng.random()
        cum = 0.0
        next_id = len(probs) - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                next_id = i
                break
        generated.append(next_id)
        if next_id == tokenizer.token_to_id[tokenizer.EOS]:
            break
    return generated


# ──────────────────────────────────────────────
#  Datasets
# ──────────────────────────────────────────────

def build_vocab_and_data():
    """Build a small vocabulary and training data for demos."""
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends",
        "the cat and the dog played",
        "a big cat sat down",
        "the small dog ran fast",
        "cats like to play",
        "dogs like to run",
        "the cat chased the dog",
        "the dog chased the cat",
    ]
    all_words = set()
    for s in sentences:
        all_words.update(s.split())

    vocab = [SimpleTokenizer.PAD, SimpleTokenizer.UNK, SimpleTokenizer.BOS, SimpleTokenizer.EOS]
    vocab.extend(sorted(all_words))
    tokenizer = SimpleTokenizer(vocab)
    return tokenizer, sentences


# ──────────────────────────────────────────────
#  DEMOS
# ──────────────────────────────────────────────

def demo_1_architecture():
    print("=" * 65)
    print("DEMO 1: GPT Architecture (Decoder-only Transformer)")
    print("=" * 65)

    tokenizer, _ = build_vocab_and_data()
    vocab_size = len(tokenizer)
    d_model = 32
    n_heads = 4
    n_layers = 2

    model = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, max_seq_len=64, seed=42)

    print(f"\n  Vocab size   : {vocab_size}")
    print(f"  d_model      : {d_model}")
    print(f"  n_heads      : {n_heads}")
    print(f"  n_layers     : {n_layers}")
    print(f"  d_k (per head): {d_model // n_heads}")
    print(f"  d_ff (FFN)   : {4 * d_model}")
    print(f"  Max seq len  : {model.max_seq_len}")

    prompt = "the cat sat"
    ids = tokenizer.encode(prompt)
    logits = model.forward(ids)

    print(f"\n  Input tokens : {ids}")
    print(f"  Input text   : \"{prompt}\"")
    print(f"  Output shape : {len(logits)} positions × {len(logits[0])} vocab logits")
    print(f"  Last position logits (first 8): {[round(v, 4) for v in logits[-1][:8]]}")

    # Show architecture diagram
    print(f"""
  Architecture Diagram:
  ┌─────────────────────────────────────────┐
  │  Input: token_ids [{len(ids)} tokens]              │
  ├─────────────────────────────────────────┤
  │  Token Embedding  ({vocab_size} × {d_model})           │
  │       +                                 │
  │  Position Embedding ({model.max_seq_len} × {d_model})          │
  ├─────────────────────────────────────────┤
  │  ┌─── Transformer Block ×{n_layers} ──────────┐  │
  │  │  LayerNorm → Multi-Head Attention   │  │
  │  │  ({n_heads} heads × {d_model // n_heads} d_k) + Residual      │  │
  │  │  LayerNorm → Feed-Forward (GELU)    │  │
  │  │  ({d_model} → {4 * d_model} → {d_model}) + Residual         │  │
  │  └─────────────────────────────────────┘  │
  ├─────────────────────────────────────────┤
  │  Final LayerNorm                        │
  │  LM Head ({d_model} → {vocab_size})                 │
  ├─────────────────────────────────────────┤
  │  Output: logits [{len(logits)} × {len(logits[0])}]                │
  └─────────────────────────────────────────┘
""")
    print("  [OK] Architecture demo complete.\n")


def demo_2_causal_mask():
    print("=" * 65)
    print("DEMO 2: Causal Mask (Upper-Triangular)")
    print("=" * 65)

    seq_len = 6
    mask = create_causal_mask(seq_len)

    tokens = ["<BOS>", "the", "cat", "sat", "on", "the"]
    print(f"\n  Sequence: {tokens}")
    print(f"  Mask shape: {seq_len} × {seq_len}")
    print(f"\n  Causal mask (0 = attend, -inf = masked):\n")

    # Header
    print("  " + " " * 12 + "".join(f"{t:>8}" for t in tokens))
    print("  " + " " * 12 + "─" * (8 * seq_len))

    for i in range(seq_len):
        row_str = f"  {tokens[i]:>8}  │ "
        for j in range(seq_len):
            if mask[i][j] == 0.0:
                row_str += f"{'0':>8}"
            else:
                row_str += f"{'-∞':>8}"
        print(row_str)

    print(f"""
  Key observations:
  • Position 0 (<BOS>) can only see itself
  • Each position can attend to all previous positions + itself
  • Future positions are blocked (upper triangle = -inf)
  • This prevents information leakage during training
  • Total masked positions: {sum(1 for i in range(seq_len) for j in range(seq_len) if mask[i][j] == float('-inf'))}
  • Total visible positions: {sum(1 for i in range(seq_len) for j in range(seq_len) if mask[i][j] == 0.0)}
""")
    print("  [OK] Causal mask demo complete.\n")


def demo_3_greedy():
    print("=" * 65)
    print("DEMO 3: Greedy Decoding")
    print("=" * 65)

    tokenizer, sentences = build_vocab_and_data()
    model = GPT(vocab_size=len(tokenizer), d_model=32, n_heads=4, n_layers=2,
                max_seq_len=64, seed=42)

    prompt = "the cat"
    prompt_ids = tokenizer.encode(prompt)

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Token IDs: {prompt_ids}")
    print(f"  Strategy: Always pick highest-probability token (argmax)")
    print()

    generated = greedy_decode(model, tokenizer, prompt_ids, max_new_tokens=8)
    text = tokenizer.decode(generated)
    print(f"  Generated IDs : {generated}")
    print(f"  Generated text: \"{text}\"")
    print(f"  New tokens    : {len(generated) - len(prompt_ids)}")
    print()
    print("  Step-by-step greedy selection:")
    for step in range(len(prompt_ids), len(generated)):
        # Re-run to show logits at each step
        partial = generated[:step]
        logits = model.forward(partial)
        next_logits = logits[-1]
        next_id = generated[step]
        probs = softmax(next_logits)
        top_idx = max(range(len(next_logits)), key=lambda i: next_logits[i])
        print(f"    Step {step - len(prompt_ids) + 1}: "
              f"token_id={next_id:>3} "
              f"({tokenizer.id_to_token[next_id]:>8})  "
              f"prob={probs[next_id]:.4f}  "
              f"top_choice_id={top_idx}  "
              f"({tokenizer.id_to_token[top_idx]:>8})")

    print("\n  Note: Greedy decoding is deterministic — same input → same output.")
    print("  It can produce repetitive text and miss diverse phrasings.\n")
    print("  [OK] Greedy decoding demo complete.\n")


def demo_4_temperature():
    print("=" * 65)
    print("DEMO 4: Temperature Sampling")
    print("=" * 65)

    tokenizer, _ = build_vocab_and_data()
    model = GPT(vocab_size=len(tokenizer), d_model=32, n_heads=4, n_layers=2,
                max_seq_len=64, seed=42)

    prompt = "the dog"
    prompt_ids = tokenizer.encode(prompt)

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Token IDs: {prompt_ids}")
    print()

    temperatures = [0.1, 0.5, 1.0, 2.0]

    for temp in temperatures:
        print(f"  ── Temperature = {temp} ──")
        random.seed(42)
        results = []
        for trial in range(3):
            gen = sample_decode(model, tokenizer, prompt_ids, max_new_tokens=8,
                                temperature=temp, top_k=0)
            text = tokenizer.decode(gen)
            results.append(text)
            print(f"    Trial {trial + 1}: \"{text}\"")

        unique = len(set(results))
        print(f"    Diversity: {unique}/3 unique outputs")
        print()

    # Top-k demo
    print("  ── Top-k Sampling (k=3, temperature=1.0) ──")
    for trial in range(3):
        random.seed(42 + trial)
        gen = sample_decode(model, tokenizer, prompt_ids, max_new_tokens=8,
                            temperature=1.0, top_k=3)
        text = tokenizer.decode(gen)
        print(f"    Trial {trial + 1}: \"{text}\"")

    print("""
  Temperature effects:
  • T < 1.0 → sharper distribution, more deterministic
  • T = 1.0 → original probabilities, balanced
  • T > 1.0 → flatter distribution, more random
  • T → 0   → equivalent to greedy decoding
  • T → ∞   → uniform random sampling

  Top-k sampling:
  • Only considers the k most probable tokens
  • Filters out low-probability tokens
  • Combines with temperature for controlled generation
""")
    print("  [OK] Temperature sampling demo complete.\n")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║       101 — GPT Architecture from Scratch (Pure Python)     ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    demo_1_architecture()
    demo_2_causal_mask()
    demo_3_greedy()
    demo_4_temperature()

    print("=" * 65)
    print("All demos complete!")
    print("=" * 65)
