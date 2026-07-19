"""
126 — Vision Transformers (ViT): patch embeddings, positional encoding for images, classification

Темы:
  1. Image Patching (divide image into patches, flatten, linear projection)
  2. Patch Embeddings (learnable projections, positional encoding for 2D grids)
  3. ViT Architecture (patch tokens + CLS token, transformer encoder, classification head)
  4. ViT Variants (DeiT data efficiency, Swin shifted windows, comparison with CNN)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ─────────────────────────── helper utilities ───────────────────────────

def _mat_vec_mul(mat, vec):
    """Multiply matrix (list of lists) by vector (list)."""
    return [sum(r[j] * vec[j] for j in range(len(vec))) for r in mat]


def _softmax(v):
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    return [e / s for e in exps]


def _layernorm(x, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((xi - mean) ** 2 for xi in x) / len(x)
    return [(xi - mean) / math.sqrt(var + eps) for xi in x]


def _relu(v):
    return [max(0.0, x) for x in v]


def _xavier_init(rows, cols):
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


# ─────────────────────────── DEMO 1: Image Patching ───────────────────────────

def demo_image_patching():
    """
    Divide an image into patches, flatten them, and project to embeddings.
    ViT first splits an image of size HxW into N = (H/P)*(W/P) patches of size PxP.
    """
    print("=" * 70)
    print("DEMO 1: Image Patching — divide image into patches, flatten, project")
    print("=" * 70)

    # 1a) Synthetic 8x8 grayscale image (values 0..255)
    H, W = 8, 8
    image = [[(i * W + j + 37) % 256 for j in range(W)] for i in range(H)]
    print(f"\n[1a] Synthetic {H}x{W} image (top-left 4x4):")
    for row in image[:4]:
        print("  ", [f"{v:3d}" for v in row[:4]])

    # 1b) Patch extraction with patch size P=2
    P = 2
    n_patches_h = H // P
    n_patches_w = W // P
    n_patches = n_patches_h * n_patches_w

    patches = []
    for ph in range(n_patches_h):
        for pw in range(n_patches_w):
            patch = []
            for i in range(ph * P, (ph + 1) * P):
                for j in range(pw * P, (pw + 1) * P):
                    patch.append(image[i][j])
            patches.append(patch)

    print(f"\n[1b] Patch size P={P} → {n_patches_h}x{n_patches_w} = {n_patches} patches")
    print(f"  Each patch is {P}x{P} = {P * P} pixels")
    print(f"  First patch (flattened): {patches[0]}")
    print(f"  Second patch (flattened): {patches[1]}")

    # 1c) Flattened patch dimensions
    patch_dim = P * P  # flattened size
    print(f"\n[1c] Flattened patch dimension: {P}×{P} = {patch_dim}")
    print(f"  Image → sequence of {n_patches} vectors, each of dim {patch_dim}")

    # 1d) Linear projection: map patch_dim → embed_dim
    embed_dim = 8
    W_proj = _xavier_init(embed_dim, patch_dim)
    projected = [_mat_vec_mul(W_proj, patch) for patch in patches]

    print(f"\n[1d] Linear projection: {patch_dim} → {embed_dim} (embed_dim={embed_dim})")
    print(f"  Projection matrix shape: {embed_dim}×{patch_dim}")
    print(f"  Patch 0 projected (first 4 dims): {[round(v, 4) for v in projected[0][:4]]}")
    print(f"  Patch 1 projected (first 4 dims): {[round(v, 4) for v in projected[1][:4]]}")

    print("\n  FORMULA: z_p = x_patch * W_proj + b_proj")
    print(f"  Sequence length N = {n_patches}, Embed dim D = {embed_dim}")
    print(f"  Output tensor shape: ({n_patches}, {embed_dim})")


# ─────────────────────────── DEMO 2: Patch Embeddings ───────────────────────────

def demo_patch_embeddings():
    """
    Add positional encoding to patch embeddings.
    ViT uses learnable 1D positional embeddings added to the patch token sequence.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Patch Embeddings — positional encoding for 2D image grids")
    print("=" * 70)

    # 2a) Parameters
    n_patches = 16  # 4x4 image with P=2
    embed_dim = 8

    # Generate patch embeddings (simulated random)
    random.seed(42)
    patch_embeds = [[random.gauss(0, 0.5) for _ in range(embed_dim)] for _ in range(n_patches)]
    print(f"\n[2a] {n_patches} patch embeddings (dim={embed_dim}):")
    for i in range(min(4, n_patches)):
        print(f"  Patch {i:2d}: {[round(v, 3) for v in patch_embeds[i][:4]]}...")

    # 2b) 2D sinusoidal positional encoding (for comparison)
    def sinusoidal_2d(pos_h, pos_w, d):
        """Generate 2D sinusoidal positional encoding."""
        enc = []
        for i in range(d // 2):
            angle_h = pos_h / (10000 ** (2 * i / d))
            angle_w = pos_w / (10000 ** (2 * i / d))
            enc.append(math.sin(angle_h))
            enc.append(math.cos(angle_w))
        return enc[:d]

    pos_enc_sin = [sinusoidal_2d(i // 4, i % 4, embed_dim) for i in range(n_patches)]
    print(f"\n[2b] 2D sinusoidal positional encoding (4×4 grid):")
    print(f"  Pos (0,0) first 4: {[round(v, 3) for v in pos_enc_sin[0][:4]]}")
    print(f"  Pos (1,0) first 4: {[round(v, 3) for v in pos_enc_sin[4][:4]]}")

    # 2c) Learnable positional embeddings (random init, same shape)
    pos_embeds_learnable = [[random.gauss(0, 0.02) for _ in range(embed_dim)] for _ in range(n_patches)]
    print(f"\n[2c] Learnable positional embeddings (init σ=0.02):")
    print(f"  Shape: ({n_patches}, {embed_dim}) — {n_patches * embed_dim} parameters")
    for i in range(min(4, n_patches)):
        print(f"  Pos {i:2d}: {[round(v, 4) for v in pos_embeds_learnable[i][:4]]}...")

    # 2d) Final embeddings = patch_embed + pos_embed
    final_embeds = []
    for i in range(n_patches):
        combined = [patch_embeds[i][d] + pos_embeds_learnable[i][d] for d in range(embed_dim)]
        final_embeds.append(combined)

    print(f"\n[2d] Final patch embeddings (patch + positional):")
    print(f"  Formula: z_i = patch_embed_i + pos_embed_i")
    for i in range(min(4, n_patches)):
        print(f"  Token {i:2d}: {[round(v, 4) for v in final_embeds[i][:4]]}...")

    print(f"\n  Total learnable positional params: {n_patches * embed_dim}")
    print(f"  Sequence shape fed to transformer: ({n_patches}, {embed_dim})")


# ─────────────────────────── DEMO 3: ViT Architecture ───────────────────────────

def demo_vit_architecture():
    """
    Full ViT forward pass: CLS token, transformer encoder, classification head.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: ViT Architecture — CLS token + transformer encoder + classifier")
    print("=" * 70)

    embed_dim = 8
    n_patches = 9  # 3x3 grid, P=2 from 6x6 image
    n_heads = 2
    head_dim = embed_dim // n_heads
    n_layers = 2
    n_classes = 3

    # 3a) Prepend CLS token
    random.seed(42)
    cls_token = [random.gauss(0, 0.1) for _ in range(embed_dim)]
    patch_embeds = [[random.gauss(0, 0.5) for _ in range(embed_dim)] for _ in range(n_patches)]
    tokens = [cls_token] + patch_embeds  # sequence: 1 + N tokens

    print(f"\n[3a] CLS token prepended:")
    print(f"  CLS token: {[round(v, 4) for v in cls_token[:4]]}...")
    print(f"  Total sequence length: 1 (CLS) + {n_patches} (patches) = {len(tokens)} tokens")

    # 3b) Multi-head self-attention (simplified)
    def self_attention_single_head(q, k, v, head_dim):
        """Scaled dot-product attention for one head."""
        score = sum(q[i] * k[i] for i in range(head_dim)) / math.sqrt(head_dim)
        return score, v  # returning raw score for demo

    # QKV projections
    W_q = _xavier_init(embed_dim, embed_dim)
    W_k = _xavier_init(embed_dim, embed_dim)
    W_v = _xavier_init(embed_dim, embed_dim)

    Q = [_mat_vec_mul(W_q, t) for t in tokens]
    K = [_mat_vec_mul(W_k, t) for t in tokens]
    V = [_mat_vec_mul(W_v, t) for t in tokens]

    # Attention from CLS to all tokens
    attn_scores = []
    for i in range(len(tokens)):
        score = sum(Q[0][j] * K[i][j] for j in range(embed_dim)) / math.sqrt(embed_dim)
        attn_scores.append(score)
    attn_weights = _softmax(attn_scores)

    print(f"\n[3b] Self-attention (CLS → all tokens):")
    print(f"  Attention weights from CLS:")
    for i in range(min(5, len(tokens))):
        label = "CLS" if i == 0 else f"P{i - 1:2d}"
        print(f"    → {label}: {attn_weights[i]:.4f}")
    print(f"  Sum of weights: {sum(attn_weights):.6f}")

    # 3c) Transformer block: attention + FFN + residual + layernorm
    def transformer_block(token_seq, embed_dim):
        """One transformer block: LN → Attn → Residual → LN → FFN → Residual."""
        # LayerNorm + Attention
        normed = [_layernorm(t) for t in token_seq]
        # Simplified: use mean of attended tokens
        attn_out = []
        for t in normed:
            new_t = [t[j] * 0.9 + random.gauss(0, 0.01) for j in range(embed_dim)]
            attn_out.append(new_t)
        # Residual
        after_attn = [[token_seq[i][j] + attn_out[i][j] for j in range(embed_dim)]
                       for i in range(len(token_seq))]
        # FFN (2-layer MLP)
        ffn_hidden = embed_dim * 4
        W1 = _xavier_init(ffn_hidden, embed_dim)
        W2 = _xavier_init(embed_dim, ffn_hidden)
        ffn_out = []
        for t in after_attn:
            h = _relu(_mat_vec_mul(W1, t))
            o = _mat_vec_mul(W2, h)
            ffn_out.append(o)
        # Residual
        output = [[after_attn[i][j] + ffn_out[i][j] for j in range(embed_dim)]
                  for i in range(len(token_seq))]
        return output

    random.seed(42)
    hidden = tokens
    for layer in range(n_layers):
        hidden = transformer_block(hidden, embed_dim)
        print(f"  Layer {layer + 1} output CLS: {[round(v, 4) for v in hidden[0][:4]]}...")

    # 3d) Classification head: take CLS token, project to n_classes
    W_cls = _xavier_init(n_classes, embed_dim)
    logits = _mat_vec_mul(W_cls, hidden[0])
    probs = _softmax(logits)

    print(f"\n[3d] Classification head:")
    print(f"  CLS token → Linear({embed_dim} → {n_classes}) → softmax")
    print(f"  Logits: {[round(v, 4) for v in logits]}")
    print(f"  Probabilities: {[round(v, 4) for v in probs]}")
    print(f"  Predicted class: {probs.index(max(probs))}")
    print(f"\n  ViT summary:")
    print(f"    Input: ({n_patches + 1}, {embed_dim}) = CLS + {n_patches} patch tokens")
    print(f"    Encoder: {n_layers} transformer blocks, {n_heads} heads")
    print(f"    Output: {n_classes}-way classification from CLS token")


# ─────────────────────────── DEMO 4: ViT Variants ───────────────────────────

def demo_vit_variants():
    """
    DeiT (data-efficient), Swin Transformer (shifted windows), comparison with CNN.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: ViT Variants — DeiT, Swin, CNN comparison")
    print("=" * 70)

    # 4a) DeiT: distillation token
    embed_dim = 8
    random.seed(42)
    cls_token = [random.gauss(0, 0.1) for _ in range(embed_dim)]
    distill_token = [random.gauss(0, 0.1) for _ in range(embed_dim)]
    patch_tokens = [[random.gauss(0, 0.5) for _ in range(embed_dim)] for _ in range(9)]

    tokens_deit = [cls_token, distill_token] + patch_tokens
    print(f"\n[4a] DeiT (Data-efficient Image Transformer):")
    print(f"  Adds distillation token alongside CLS token")
    print(f"  Sequence: 1 CLS + 1 distill + {len(patch_tokens)} patches = {len(tokens_deit)} tokens")
    print(f"  During training: both CLS and distill predict labels")
    print(f"  During inference: only distill token used (student behavior)")
    print(f"  CLS token: {[round(v, 4) for v in cls_token[:4]]}...")
    print(f"  Distill token: {[round(v, 4) for v in distill_token[:4]]}...")
    print(f"  Train with: 0.5 * CE(cls_pred, label) + 0.5 * CE(distill_pred, label)")

    # 4b) Swin Transformer: shifted window attention
    H, W = 8, 8
    P = 4  # initial patch size
    n_patches_h = H // P
    n_patches_w = W // P
    window_size = 2  # 2x2 windows

    print(f"\n[4b] Swin Transformer (Shifted Window Attention):")
    print(f"  Image: {H}×{W}, Patch size: {P}×{P} → {n_patches_h}×{n_patches_w} patches")

    # Window partitioning
    windows = []
    for wh in range(n_patches_h // window_size):
        for ww in range(n_patches_w // window_size):
            win = []
            for i in range(window_size):
                for j in range(window_size):
                    win.append((wh * window_size + i, ww * window_size + j))
            windows.append(win)

    print(f"  Window size: {window_size}×{window_size}")
    print(f"  Number of windows: {len(windows)}")
    for i, w in enumerate(windows[:3]):
        print(f"  Window {i}: {w}")

    # Shifted windows
    print(f"\n  Shifted windows (offset by W//2 = {window_size // 2}):")
    shift = window_size // 2
    shifted = []
    for wh in range(n_patches_h):
        for ww in range(n_patches_w):
            new_h = (wh + shift) % n_patches_h
            new_w = (ww + shift) % n_patches_w
            shifted.append(((wh, ww), (new_h, new_w)))
    print(f"  First 5 patch reassignments:")
    for orig, shifted_pos in shifted[:5]:
        print(f"    {orig} → {shifted_pos}")

    # 4c) CNN vs ViT comparison
    print(f"\n[4c] CNN vs ViT Architecture Comparison:")

    configs = {
        "CNN (ResNet-50)": {
            "params_M": 25.6,
            "image_size": 224,
            "inductive_bias": "Local receptive field, weight sharing",
            "data_efficiency": "High (inductive bias helps)",
            "scalability": "Plateaus at scale",
            "receptive_field": "Grows with depth (local → global)",
        },
        "ViT-B/16": {
            "params_M": 86.6,
            "image_size": 224,
            "inductive_bias": "Minimal (only patch projection)",
            "data_efficiency": "Low (needs large datasets or distillation)",
            "scalability": "Scales well with data and compute",
            "receptive_field": "Global from layer 1 (self-attention)",
        },
        "Swin-B": {
            "params_M": 87.8,
            "image_size": 224,
            "inductive_bias": "Windowed attention + shifting",
            "data_efficiency": "Moderate",
            "scalability": "Hierarchical, good for dense tasks",
            "receptive_field": "Local windows → shifted → hierarchical",
        },
    }

    for name, cfg in configs.items():
        print(f"\n  {name}:")
        print(f"    Parameters: {cfg['params_M']}M")
        print(f"    Inductive bias: {cfg['inductive_bias']}")
        print(f"    Data efficiency: {cfg['data_efficiency']}")
        print(f"    Receptive field: {cfg['receptive_field']}")

    # 4d) Patch size and sequence length
    print(f"\n[4d] Patch Size Impact on Sequence Length (224×224 image):")
    img_size = 224
    for patch_size in [4, 8, 14, 16, 32]:
        n = (img_size // patch_size) ** 2
        print(f"  P={patch_size:2d}: {(img_size // patch_size)}×{(img_size // patch_size)} = {n:5d} patches "
              f"(seq_len = {n + 1} with CLS)")

    print(f"\n  Key insight: smaller patches → longer sequences → more compute")
    print(f"  P=16 is standard: 14×14 = 196 patches (+1 CLS = 197 tokens)")


# ─────────────────────────── Main ───────────────────────────

if __name__ == "__main__":
    demo_image_patching()
    demo_patch_embeddings()
    demo_vit_architecture()
    demo_vit_variants()
    print("\n" + "=" * 70)
    print("All Vision Transformer demos complete!")
    print("=" * 70)
