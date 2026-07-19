"""
127 — CLIP & Vision-Language Alignment: contrastive learning, zero-shot classification

Темы:
  1. Contrastive Learning (image-text pairs, similarity matrix, InfoNCE loss)
  2. CLIP Architecture (image encoder + text encoder, shared embedding space)
  3. Zero-Shot Classification (text prompts as classifiers, template ensemble)
  4. Contrastive Pre-training (batch construction, temperature learning, hard negatives)

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

def _normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-8:
        return v
    return [x / norm for x in v]


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _softmax(v, temperature=1.0):
    scaled = [x / temperature for x in v]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def _cross_entropy(probs, target_idx):
    """Cross-entropy loss for a single sample."""
    return -math.log(max(probs[target_idx], 1e-8))


def _xavier_init(rows, cols):
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def _mat_vec_mul(mat, vec):
    return [sum(r[j] * vec[j] for j in range(len(vec))) for r in mat]


# ─────────────────────────── DEMO 1: Contrastive Learning ───────────────────────────

def demo_contrastive_learning():
    """
    Image-text pairs, similarity matrix, InfoNCE loss.
    CLIP learns by matching correct image-text pairs and pushing apart mismatches.
    """
    print("=" * 70)
    print("DEMO 1: Contrastive Learning — similarity matrix, InfoNCE loss")
    print("=" * 70)

    embed_dim = 6
    batch_size = 4

    # 1a) Create image-text pairs
    random.seed(42)
    images = [[random.gauss(0, 1) for _ in range(embed_dim)] for _ in range(batch_size)]
    texts = [[random.gauss(0, 1) for _ in range(embed_dim)] for _ in range(batch_size)]

    # Normalize to unit vectors (CLIP uses L2 normalization)
    images_norm = [_normalize(img) for img in images]
    texts_norm = [_normalize(txt) for txt in texts]

    descriptions = [
        "a photo of a cat",
        "a photo of a dog",
        "a landscape with mountains",
        "a photo of a car",
    ]

    print(f"\n[1a] Batch of {batch_size} image-text pairs:")
    for i in range(batch_size):
        print(f"  Pair {i}: '{descriptions[i]}'")
        print(f"    Image embedding (first 3): {[round(v, 3) for v in images_norm[i][:3]]}")
        print(f"    Text embedding  (first 3): {[round(v, 3) for v in texts_norm[i][:3]]}")

    # 1b) Compute similarity matrix
    similarity = []
    for i in range(batch_size):
        row = []
        for j in range(batch_size):
            sim = _dot(images_norm[i], texts_norm[j])
            row.append(round(sim, 4))
        similarity.append(row)

    print(f"\n[1b] Similarity matrix S[i,j] = cos_sim(image_i, text_j):")
    print(f"  {'':>8s}", end="")
    for j in range(batch_size):
        print(f"  txt_{j:2d}", end="")
    print()
    for i in range(batch_size):
        print(f"  img_{i:2d}", end="")
        for j in range(batch_size):
            print(f"  {similarity[i][j]:+.4f}", end="")
        print()
    print(f"  Diagonal (correct pairs): {[similarity[i][i] for i in range(batch_size)]}")

    # 1c) InfoNCE loss
    temperature = 0.07  # CLIP default
    scale = 1.0 / temperature

    print(f"\n[1c] InfoNCE Loss (temperature τ={temperature}):")
    total_loss = 0.0
    for i in range(batch_size):
        logits = [similarity[i][j] * scale for j in range(batch_size)]
        probs = _softmax(logits)
        loss = _cross_entropy(probs, i)
        total_loss += loss
        print(f"  Image {i}: logits = {[round(l, 3) for l in logits]}")
        print(f"           probs  = {[round(p, 4) for p in probs]}")
        print(f"           loss   = -log({probs[i]:.4f}) = {loss:.4f}")

    avg_loss = total_loss / batch_size
    print(f"  Average InfoNCE loss: {avg_loss:.4f}")

    # 1d) Symmetric loss (image-to-text + text-to-image)
    total_loss_t2i = 0.0
    for j in range(batch_size):
        logits = [similarity[i][j] * scale for i in range(batch_size)]
        probs = _softmax(logits)
        loss = _cross_entropy(probs, j)
        total_loss_t2i += loss

    avg_loss_t2i = total_loss_t2i / batch_size
    symmetric_loss = (avg_loss + avg_loss_t2i) / 2

    print(f"\n[1d] Symmetric CLIP Loss:")
    print(f"  Image→Text loss: {avg_loss:.4f}")
    print(f"  Text→Image loss: {avg_loss_t2i:.4f}")
    print(f"  L_clip = (L_i2t + L_t2i) / 2 = {symmetric_loss:.4f}")
    print(f"\n  InfoNCE formula: L = -log(exp(s_ii/τ) / Σ_j exp(s_ij/τ))")
    print(f"  Both directions ensure symmetric alignment")


# ─────────────────────────── DEMO 2: CLIP Architecture ───────────────────────────

def demo_clip_architecture():
    """
    Image encoder + text encoder producing shared embeddings.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: CLIP Architecture — dual encoders, shared embedding space")
    print("=" * 70)

    embed_dim = 8
    img_dim = 12  # flattened image patch
    txt_vocab = 10  # simulated vocab size

    # 2a) Image encoder (simplified ViT-like)
    random.seed(42)
    W_img_proj = _xavier_init(embed_dim, img_dim)
    # Simulate averaged patch embeddings as image representation
    image_patches = [[random.gauss(0, 0.5) for _ in range(img_dim)] for _ in range(4)]
    image_embed = [0.0] * embed_dim
    for patch in image_patches:
        projected = _mat_vec_mul(W_img_proj, patch)
        image_embed = [image_embed[d] + projected[d] for d in range(embed_dim)]
    image_embed = _normalize(image_embed)

    print(f"\n[2a] Image Encoder (ViT-like):")
    print(f"  Input: 4 patches × {img_dim} features")
    print(f"  Linear projection: {img_dim} → {embed_dim}")
    print(f"  Global average pooling → L2 normalize")
    print(f"  Image embedding (first 4): {[round(v, 4) for v in image_embed[:4]]}")

    # 2b) Text encoder (simplified)
    W_txt_emb = _xavier_init(embed_dim, txt_vocab)
    # Simulate bag-of-words then project
    token_ids = [3, 7, 2, 1, 0]  # simulated tokens
    text_bow = [0.0] * txt_vocab
    for tid in token_ids:
        if tid < txt_vocab:
            text_bow[tid] += 1.0
    # Normalize bag-of-words
    bow_norm = math.sqrt(sum(x * x for x in text_bow))
    text_bow = [x / bow_norm for x in text_bow]

    text_embed = _normalize(_mat_vec_mul(W_txt_emb, text_bow))

    print(f"\n[2b] Text Encoder (Transformer-like):")
    print(f"  Input tokens: {token_ids}")
    print(f"  Token embeddings → pool → project to {embed_dim}d")
    print(f"  Text embedding (first 4): {[round(v, 4) for v in text_embed[:4]]}")

    # 2c) Shared embedding space
    sim = _dot(image_embed, text_embed)
    print(f"\n[2c] Shared Embedding Space:")
    print(f"  Both encoders output {embed_dim}-dimensional unit vectors")
    print(f"  Cosine similarity: image · text = {sim:.4f}")
    print(f"  During training: correct pairs should have high similarity")
    print(f"  The dot product in shared space IS the scoring function")

    # 2d) Scaling: CLIP contrastive pairs in batch
    batch_size = 32  # CLIP uses 32768 in full training
    n_positives = batch_size  # 1 correct per image
    n_negatives = batch_size * (batch_size - 1)  # all wrong pairs
    total_pairs = batch_size * batch_size

    print(f"\n[2d] Scaling Analysis:")
    print(f"  Batch size: {batch_size}")
    print(f"  Positive pairs: {n_positives}")
    print(f"  Negative pairs: {n_negatives}")
    print(f"  Total pairs: {total_pairs}")
    print(f"\n  CLIP training details:")
    print(f"    - Batch size: 32,768 (49,152 in latest)")
    print(f"    - Embedding dim: 512 (ViT-L/14) or 768 (ViT-H-14)")
    print(f"    - Temperature τ: learnable, init 0.07")
    print(f"    - Image encoder: ViT or ResNet")
    print(f"    - Text encoder: Transformer (63M params)")


# ─────────────────────────── DEMO 3: Zero-Shot Classification ───────────────────────────

def demo_zero_shot_classification():
    """
    Use text prompts as classifiers with template ensemble.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Zero-Shot Classification — text prompts as classifiers")
    print("=" * 70)

    embed_dim = 6

    # 3a) Image embedding and class names
    random.seed(42)
    image_embed = _normalize([random.gauss(0, 1) for _ in range(embed_dim)])
    class_names = ["cat", "dog", "bird", "fish"]
    class_embeds = {name: _normalize([random.gauss(0, 1) for _ in range(embed_dim)])
                    for name in class_names}

    print(f"\n[3a] Zero-shot classification setup:")
    print(f"  Image embedding: {[round(v, 3) for v in image_embed[:4]]}...")
    print(f"  Classes: {class_names}")
    for name in class_names:
        sim = _dot(image_embed, class_embeds[name])
        print(f"    cos_sim(image, '{name}') = {sim:.4f}")

    # 3b) Text prompt templates
    templates = [
        "a photo of a {}",
        "a blurry photo of a {}",
        "a painting of a {}",
        "a drawing of a {}",
        "a close-up of a {}",
    ]

    print(f"\n[3b] Prompt templates (ensemble of {len(templates)}):")
    for t in templates:
        print(f"  '{t}'")

    # 3c) Template ensemble: average similarity across templates
    print(f"\n[3c] Template ensemble results:")

    # Simulate different embeddings for each template (in real CLIP, each template gives different embedding)
    random.seed(123)
    template_class_embeds = {}
    for template in templates:
        for name in class_names:
            key = (template, name)
            # Each template produces a slightly different embedding
            base = class_embeds[name]
            noise = [random.gauss(0, 0.15) for _ in range(embed_dim)]
            template_class_embeds[key] = _normalize([base[i] + noise[i] for i in range(embed_dim)])

    avg_sims = {}
    for name in class_names:
        sims = []
        for template in templates:
            sim = _dot(image_embed, template_class_embeds[(template, name)])
            sims.append(sim)
        avg_sim = sum(sims) / len(sims)
        avg_sims[name] = avg_sim
        print(f"  '{name}': individual sims = {[round(s, 4) for s in sims]}")
        print(f"         averaged = {avg_sim:.4f}")

    probs = _softmax(list(avg_sims.values()), temperature=0.01)
    print(f"\n  Softmax probs: {dict(zip(class_names, [round(p, 4) for p in probs]))}")
    print(f"  Predicted: {class_names[probs.index(max(probs))]}")

    # 3d) Prompt engineering for better accuracy
    print(f"\n[3d] Prompt Engineering Impact:")
    print(f"  Good prompts: 'a photo of a {class_names[0]}' → specific, natural")
    print(f"  Bad prompts:  '{class_names[0]}' → too vague, no visual context")
    print(f"  Ensembling: average over {len(templates)} templates improves robustness")
    print(f"\n  Real CLIP performance (ImageNet zero-shot):")
    print(f"    ViT-L/14: 76.2% top-1 accuracy")
    print(f"    ViT-H-14: 78.0% top-1 accuracy")
    print(f"    Ensemble of prompts: +1-2% boost")
    print(f"\n  Key insight: CLIP never saw ImageNet training data!")
    print(f"  Zero-shot = no fine-tuning, just text prompts as classifiers")


# ─────────────────────────── DEMO 4: Contrastive Pre-training ───────────────────────────

def demo_contrastive_pretraining():
    """
    Batch construction, temperature learning, hard negatives.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Contrastive Pre-training — batches, temperature, hard negatives")
    print("=" * 70)

    embed_dim = 6

    # 4a) Batch construction from web data
    print(f"\n[4a] CLIP Batch Construction:")
    print(f"  Source: 400M image-text pairs from internet (WIT dataset)")
    print(f"  Batch construction:")
    print(f"    1. Sample N image-text pairs from web")
    print(f"    2. Each pair (image_i, text_i) is a positive match")
    print(f"    3. All other pairs in batch are negatives")
    print(f"    4. Total negatives per sample: N-1")

    batch_size = 8
    print(f"\n  For batch_size={batch_size}:")
    print(f"    Positive pairs: {batch_size}")
    print(f"    Negative pairs per image: {batch_size - 1}")
    print(f"    Total loss computed over: {batch_size} forward passes")

    # 4b) Temperature learning
    random.seed(42)
    print(f"\n[4b] Temperature Learning:")
    print(f"  Temperature τ controls sharpness of similarity distribution")

    temperatures = [0.01, 0.05, 0.07, 0.1, 0.5, 1.0]
    sim = 0.8  # fixed similarity for demo

    for temp in temperatures:
        logits = [sim / temp, 0.3 / temp, -0.1 / temp, 0.5 / temp]
        probs = _softmax(logits)
        print(f"  τ={temp:.2f}: logits={[round(l, 2) for l in logits]}, "
              f"max_prob={max(probs):.4f}")

    print(f"\n  Small τ → sharp distribution (confident)")
    print(f"  Large τ → soft distribution (uncertain)")
    print(f"  CLIP learns τ as a parameter (init 0.07)")
    print(f"  Final learned τ ≈ 0.01 (very sharp)")

    # 4c) Hard negatives
    print(f"\n[4c] Hard Negatives:")
    print(f"  Hard negatives = semantically similar but incorrect pairs")
    print(f"  Example: image of 'golden retriever' paired with text 'a brown dog'")
    print(f"  These are harder to distinguish from true positives")

    # Simulate easy vs hard negatives
    random.seed(42)
    positive_sim = 0.95
    easy_neg_sim = 0.2
    hard_neg_sim = 0.7

    print(f"\n  Similarity scores:")
    print(f"    Positive (correct pair):    {positive_sim:.2f}")
    print(f"    Easy negative (unrelated):  {easy_neg_sim:.2f}")
    print(f"    Hard negative (similar):    {hard_neg_sim:.2f}")
    print(f"\n  Hard negatives push the model to learn finer distinctions")

    # 4d) Training dynamics
    print(f"\n[4d] CLIP Training Dynamics:")
    print(f"  Epoch 1-10:   Learning basic visual-text alignment")
    print(f"  Epoch 10-50:  Refining fine-grained distinctions")
    print(f"  Epoch 50-100: Optimizing temperature and edge cases")

    # Simulate loss curve
    print(f"\n  Simulated training loss curve:")
    for epoch in [1, 5, 10, 25, 50, 100]:
        loss = 3.5 * math.exp(-epoch / 20) + 0.8 + random.gauss(0, 0.05)
        print(f"    Epoch {epoch:3d}: loss = {loss:.3f}")

    print(f"\n  CLIP training details:")
    print(f"    - 256 V100 GPUs")
    print(f"    - 32,768 batch size")
    print(f"    - Adam optimizer, lr=5e-4 → cosine decay")
    print(f"    - 32 epochs on 400M pairs")
    print(f"    - Total compute: ~1.8e23 FLOPs")
    print(f"\n  Key result: CLIP learns transferable representations")
    print(f"  without task-specific labels — just image-text alignment!")


# ─────────────────────────── Main ───────────────────────────

if __name__ == "__main__":
    demo_contrastive_learning()
    demo_clip_architecture()
    demo_zero_shot_classification()
    demo_contrastive_pretraining()
    print("\n" + "=" * 70)
    print("All CLIP & Vision-Language Alignment demos complete!")
    print("=" * 70)
