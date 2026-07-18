"""
97 — Embeddings & Positional Encoding from Scratch

Basics of token embeddings and positional encoding (learned & sinusoidal)
implemented in pure Python + math (no torch/transformers).
"""

import math
import random

random.seed(42)


# ─── helpers ───────────────────────────────────────────────────────────

def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def mat_to_str(mat, label="", cols=None):
    lines = [f"  {label}({len(mat)}x{len(mat[0])}):"] if label else []
    for row in mat:
        truncated = row[:cols] if cols else row
        s = "  ".join(f"{v:7.4f}" for v in truncated)
        suffix = " ..." if cols and len(row) > cols else ""
        lines.append(f"    [{s}{suffix}]")
    return "\n".join(lines)

def vec_to_str(vec, label="", width=7):
    s = "  ".join(f"{v:7.4f}" for v in vec)
    return f"  {label}[{s}]" if label else f"  [{s}]"

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def matmul_vec(mat, vec):
    return [dot(row, vec) for row in mat]

def softmax(vec):
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]

def cosine_sim(a, b):
    da = math.sqrt(dot(a, a))
    db = math.sqrt(dot(b, b))
    if da == 0 or db == 0:
        return 0.0
    return dot(a, b) / (da * db)


# ═══════════════════════════════════════════════════════════════════════
# Demo 1 — Token Embeddings
# ═══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Demo 1: Token Embeddings — Lookup Table")
print("=" * 70)

vocab = ["<pad>", "<unk>", "the", "cat", "sat", "on", "mat"]
embed_dim = 4

# Fixed embedding matrix (模拟 pre-trained / learned weights)
random.seed(42)
embedding_matrix = [
    [random.gauss(0, 0.5) for _ in range(embed_dim)]
    for _ in vocab
]

token_to_id = {tok: i for i, tok in enumerate(vocab)}

print(f"\nVocabulary ({len(vocab)} tokens, dim={embed_dim}):")
for tok, idx in token_to_id.items():
    print(f"  {idx}: '{tok}' -> {vec_to_str(embedding_matrix[idx])}")

print("\n--- Embedding lookup ---")
sentence = ["the", "cat", "sat", "on", "mat"]
print(f"  Sentence: {sentence}")
print(f"  Token IDs: {[token_to_id[t] for t in sentence]}")

embeddings = [embedding_matrix[token_to_id[t]] for t in sentence]
for tok, emb in zip(sentence, embeddings):
    print(f"  '{tok}' -> {vec_to_str(emb)}")

print("\n--- What embedding means geometrically ---")
cos_cat_cat = cosine_sim(embeddings[1], embeddings[1])
cos_cat_sat = cosine_sim(embeddings[1], embeddings[2])
cos_cat_mat = cosine_sim(embeddings[1], embeddings[4])
print(f"  cosine(cat, cat) = {cos_cat_cat:.4f}")
print(f"  cosine(cat, sat) = {cos_cat_sat:.4f}")
print(f"  cosine(cat, mat) = {cos_cat_mat:.4f}")
print("  (Random init — similarities are arbitrary; training would organize the space.)")


# ═══════════════════════════════════════════════════════════════════════
# Demo 2 — Learned Positional Embeddings
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 2: Learned Positional Embeddings")
print("=" * 70)

max_len = 8

random.seed(42)
pos_emb_matrix = [
    [random.gauss(0, 0.3) for _ in range(embed_dim)]
    for _ in range(max_len)
]

print(f"\nLearned positional embedding matrix ({max_len} positions, dim={embed_dim}):")
for i in range(max_len):
    print(f"  pos={i}: {vec_to_str(pos_emb_matrix[i])}")

print("\n--- Combining token + positional embeddings (additive) ---")
for i, (tok, emb) in enumerate(zip(sentence, embeddings)):
    pos_emb = pos_emb_matrix[i]
    combined = [t + p for t, p in zip(emb, pos_emb)]
    print(f"  pos={i}, '{tok}':")
    print(f"    token:  {vec_to_str(emb)}")
    print(f"    pos:    {vec_to_str(pos_emb)}")
    print(f"    sum:    {vec_to_str(combined)}")

print("\n--- Key insight ---")
print("  Same word at different positions gets DIFFERENT representations.")
print("  'cat' at pos=1 vs 'cat' at pos=3 would produce different vectors.")


# ═══════════════════════════════════════════════════════════════════════
# Demo 3 — Sinusoidal Positional Encoding
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 3: Sinusoidal Positional Encoding (Transformer original)")
print("=" * 70)

def sinusoidal_encoding(pos, d_model):
    """Compute PE(pos, 2i) and PE(pos, 2i+1) per the original paper."""
    enc = []
    for i in range(d_model):
        angle = pos / (10000 ** (2 * (i // 2) / d_model))
        if i % 2 == 0:
            enc.append(math.sin(angle))
        else:
            enc.append(math.cos(angle))
    return enc

print(f"\nSinusoidal PE (dim={embed_dim}):")
for i in range(max_len):
    pe = sinusoidal_encoding(i, embed_dim)
    print(f"  pos={i}: {vec_to_str(pe)}")

print("\n--- Adding sinusoidal PE to token embeddings ---")
for i, (tok, emb) in enumerate(zip(sentence, embeddings)):
    pe = sinusoidal_encoding(i, embed_dim)
    combined = [t + p for t, p in zip(emb, pe)]
    print(f"  pos={i}, '{tok}': token + PE = {vec_to_str(combined)}")

print("\n--- Dot-product similarity between positions (should show patterns) ---")
for i in range(min(6, max_len)):
    for j in range(i + 1, min(6, max_len)):
        pe_i = sinusoidal_encoding(i, embed_dim)
        pe_j = sinusoidal_encoding(j, embed_dim)
        sim = cosine_sim(pe_i, pe_j)
        print(f"  cos(pos={i}, pos={j}) = {sim:+.4f}")

print("\n  Close positions have higher similarity — relative positions are captured!")
print("  This is FIXED (no learned params) — works for sequences longer than seen in training.")


# ═══════════════════════════════════════════════════════════════════════
# Demo 4 — Learned vs Sinusoidal Comparison
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 4: Learned vs Sinusoidal — Comparison")
print("=" * 70)

print("\n--- Same token at different positions ---")
word = "cat"
word_id = token_to_id[word]
token_emb = embedding_matrix[word_id]

print(f"\n  Token: '{word}' (id={word_id})")
print(f"  Token embedding: {vec_to_str(token_emb)}\n")

for pos in range(5):
    learned_pe = pos_emb_matrix[pos]
    learned_combined = [t + p for t, p in zip(token_emb, learned_pe)]

    sinus_pe = sinusoidal_encoding(pos, embed_dim)
    sinus_combined = [t + p for t, p in zip(token_emb, sinus_pe)]

    sim_learn = cosine_sim(learned_combined, sinus_combined)
    print(f"  pos={pos}:")
    print(f"    Learned:   {vec_to_str(learned_combined)}")
    print(f"    Sinusoidal:{vec_to_str(sinus_combined)}")
    print(f"    Cosine sim: {sim_learn:+.4f}")

print("\n--- Transferability test ---")
print("  Can positional encoding generalize beyond max_len?")

print("\n  Learned approach — index out of bounds:")
try:
    _ = pos_emb_matrix[max_len]  # would crash
    print("    OK")
except IndexError:
    print(f"    CRASH! max_len={max_len}, no position={max_len} embedding.")
    print("    Solution: clip to max_len-1 or extend the matrix.")

print("\n  Sinusoidal approach — arbitrary position:")
far_pos = 1000
pe_far = sinusoidal_encoding(far_pos, embed_dim)
print(f"    pos={far_pos}: {vec_to_str(pe_far)}")
print(f"    Still works! No learned parameters needed.")

print("\n--- Summary table ---")
print("  ┌────────────────────┬────────────────────┬────────────────────┐")
print("  │ Property           │ Learned            │ Sinusoidal         │")
print("  ├────────────────────┼────────────────────┼────────────────────┤")
print("  │ Parameters         │ max_len × d_model  │ None (fixed)       │")
print("  │ Flexibility        │ Arbitrary patterns │ Regular, smooth    │")
print("  │ Generalization     │ Limited to max_len │ Any length         │")
print("  │ Used in            │ BERT, GPT-2        │ Original Transformer│")
print("  │ Relative positions │ Not captured       │ Encoded via angles │")
print("  └────────────────────┴────────────────────┴────────────────────┘")

print("\n--- Modern note: RoPE (Rotary Position Embeddings) ---")
print("  GPT-4, LLaMA, Mistral use RoPE which rotates Q/K vectors by position.")
print("  It encodes relative positions in attention scores — a middle ground.")
print("  Position is embedded via complex multiplication: q·e^{iθ}, k·e^{iθ}")
print("  attention(q, k) depends on (pos_q - pos_k), not absolute position.")

print("\nDone.")
