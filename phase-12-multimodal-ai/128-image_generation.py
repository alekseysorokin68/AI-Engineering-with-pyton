"""
128 — Image Generation: autoregressive, VQ-VAE, discrete image tokens

Темы:
  1. Autoregressive Image Generation (pixel-by-patch, raster scan order)
  2. VQ-VAE (encoder, vector quantization, decoder, codebook learning)
  3. Discrete Token Representation (image tokens, vocabulary, token prediction)
  4. Generation Quality (FID concept, mode coverage, diversity vs fidelity)

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

def _softmax(v, temperature=1.0):
    scaled = [x / temperature for x in v]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def _categorical_sample(probs):
    """Sample index from probability distribution."""
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1


def _euclidean_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _xavier_init(rows, cols):
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def _mat_vec_mul(mat, vec):
    return [sum(r[j] * vec[j] for j in range(len(vec))) for r in mat]


def _relu(v):
    return [max(0.0, x) for x in v]


# ─────────────────────────── DEMO 1: Autoregressive Image Generation ───────────────────────────

def demo_autoregressive_generation():
    """
    Generate images pixel-by-patch in raster scan order.
    Each patch is predicted conditioned on all previous patches.
    """
    print("=" * 70)
    print("DEMO 1: Autoregressive Image Generation — raster scan order")
    print("=" * 70)

    H, W = 8, 8
    P = 2  # patch size
    vocab_size = 16  # simulated pixel intensity levels

    # 1a) Raster scan ordering
    n_patches_h = H // P
    n_patches_w = W // P
    order = []
    for ph in range(n_patches_h):
        for pw in range(n_patches_w):
            order.append((ph, pw))

    print(f"\n[1a] Raster scan ordering ({n_patches_h}×{n_patches_w} patches):")
    print(f"  Image: {H}×{W}, Patch size: {P}×{P}")
    print(f"  Total patches: {len(order)}")
    print("  Scan order:")
    for i, (ph, pw) in enumerate(order):
        print(f"    {i:2d}: patch ({ph},{pw}) → pixels [{ph*P}:{(ph+1)*P}, {pw*P}:{(pw+1)*P}]")

    # 1b) Autoregressive prediction (simplified)
    print(f"\n[1b] Autoregressive prediction:")
    print(f"  p(patch_i | patch_1, ..., patch_{i-1})")

    # Simulate logits for each patch (in reality, from transformer)
    random.seed(42)
    generated = []
    for i, (ph, pw) in enumerate(order):
        # Simulate logits: bias towards values seen before
        logits = [random.gauss(0, 1) for _ in range(vocab_size)]
        # Add context: previous patches influence next
        if generated:
            context_mean = sum(generated) / len(generated)
            logits[int(context_mean)] += 2.0  # bias towards average

        probs = _softmax(logits)
        token = _categorical_sample(probs)
        generated.append(token)

        if i < 4:
            print(f"  Patch ({ph},{pw}): sampled token={token}, prob={probs[token]:.4f}")

    print(f"  Generated {len(generated)} tokens: {generated}")

    # 1c) Reshape to image grid
    grid = []
    idx = 0
    for ph in range(n_patches_h):
        row = []
        for pw in range(n_patches_w):
            row.append(generated[idx])
            idx += 1
        grid.append(row)

    print(f"\n[1c] Generated image grid (patch tokens):")
    for row in grid:
        print(f"  {[f'{v:3d}' for v in row]}")

    # 1d) Key properties
    print(f"\n[1d] Autoregressive Generation Properties:")
    print(f"  ✅ Exact likelihood computation (chain rule)")
    print(f"  ✅ No mode collapse (each position has unique distribution)")
    print(f"  ❌ Slow generation: O(N) sequential steps")
    print(f"  ❌ No parallelism within a sequence")
    print(f"\n  Generation steps: {len(order)} sequential forward passes")
    print(f"  Complexity: O(N × d²) where N = patches, d = hidden dim")
    print(f"\n  Examples: PixelCNN, ImageGPT, Parti, Make-A-Scene")


# ─────────────────────────── DEMO 2: VQ-VAE ───────────────────────────

def demo_vq_vae():
    """
    Vector Quantized VAE: encoder, codebook, decoder, training.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: VQ-VAE — encoder, vector quantization, codebook learning")
    print("=" * 70)

    input_dim = 8
    latent_dim = 4
    codebook_size = 8  # K = 8 codes

    # 2a) Encoder
    random.seed(42)
    W_enc = _xavier_init(latent_dim, input_dim)
    b_enc = [0.0] * latent_dim

    input_image = [random.gauss(0, 1) for _ in range(input_dim)]
    z_e = _mat_vec_mul(W_enc, input_image)  # encoder output

    print(f"\n[2a] Encoder: input_dim={input_dim} → latent_dim={latent_dim}")
    print(f"  Input: {[round(v, 3) for v in input_image[:4]]}...")
    print(f"  Encoder output z_e: {[round(v, 4) for v in z_e]}")

    # 2b) Vector Quantization (codebook lookup)
    codebook = [[random.gauss(0, 0.5) for _ in range(latent_dim)] for _ in range(codebook_size)]

    # Find nearest code
    distances = [_euclidean_dist(z_e, codebook[k]) for k in range(codebook_size)]
    nearest_idx = distances.index(min(distances))
    z_q = codebook[nearest_idx]  # quantized vector

    print(f"\n[2b] Vector Quantization (codebook size K={codebook_size}):")
    print(f"  Codebook entries:")
    for k in range(codebook_size):
        marker = " ← nearest" if k == nearest_idx else ""
        print(f"    e[{k}]: {[round(v, 3) for v in codebook[k]]}{marker}")
    print(f"  Distances: {[round(d, 4) for d in distances]}")
    print(f"  Nearest code: e[{nearest_idx}], distance={distances[nearest_idx]:.4f}")
    print(f"  Quantized z_q: {[round(v, 4) for v in z_q]}")

    # 2c) Decoder
    W_dec = _xavier_init(input_dim, latent_dim)
    decoded = _mat_vec_mul(W_dec, z_q)

    print(f"\n[2c] Decoder: latent_dim={latent_dim} → output_dim={input_dim}")
    print(f"  Quantized input z_q: {[round(v, 4) for v in z_q]}")
    print(f"  Decoder output: {[round(v, 4) for v in decoded[:4]]}...")
    print(f"  Original input: {[round(v, 4) for v in input_image[:4]]}...")

    # Reconstruction loss
    recon_loss = sum((decoded[i] - input_image[i]) ** 2 for i in range(input_dim)) / input_dim
    print(f"  Reconstruction loss (MSE): {recon_loss:.4f}")

    # 2d) VQ-VAE losses
    print(f"\n[2d] VQ-VAE Loss Components:")
    print(f"  1. Reconstruction loss: ||x - decode(z_q)||² = {recon_loss:.4f}")
    print(f"  2. Codebook loss: ||z_e - sg(z_q)||² (update codebook toward encoder)")
    print(f"  3. Commitment loss: ||sg(z_e) - z_q||² (encoder toward codebook)")
    print(f"\n  sg = stop gradient (blocks gradient flow)")
    print(f"  Total: L = L_recon + L_codebook + β * L_commitment")
    print(f"  β typically = 0.25")

    # Demonstrate codebook usage
    print(f"\n  Codebook utilization demo:")
    random.seed(42)
    test_inputs = [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(12)]
    code_counts = [0] * codebook_size
    for inp in test_inputs:
        z = _mat_vec_mul(W_enc, inp)
        dists = [_euclidean_dist(z, codebook[k]) for k in range(codebook_size)]
        code_counts[dists.index(min(dists))] += 1

    for k in range(codebook_size):
        bar = "█" * code_counts[k]
        print(f"    Code {k}: {code_counts[k]:2d} uses {bar}")


# ─────────────────────────── DEMO 3: Discrete Token Representation ───────────────────────────

def demo_discrete_tokens():
    """
    Image as discrete tokens, vocabulary, token prediction.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Discrete Token Representation — image tokens, vocabulary")
    print("=" * 70)

    # 3a) VQ-VAE codebook as vocabulary
    vocab_size = 16
    codebook_dim = 4

    random.seed(42)
    codebook = [[random.gauss(0, 1) for _ in range(codebook_dim)] for _ in range(vocab_size)]

    print(f"\n[3a] Codebook as Token Vocabulary:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Code dimension: {codebook_dim}")
    print(f"  Each code represents a visual pattern:")
    for i in range(min(8, vocab_size)):
        # Describe what each code "looks like"
        pattern = "bright" if sum(codebook[i]) > 0 else "dark"
        print(f"    Token {i:2d} ({pattern}): {[round(v, 2) for v in codebook[i]]}")

    # 3b) Image tokenization
    H, W = 8, 8
    downsample = 4  # VQ-VAE downsamples by 4x
    grid_h = H // downsample
    grid_w = W // downsample
    n_tokens = grid_h * grid_w

    # Simulate encoding an image
    image = [[(i * W + j + 42) % 256 / 255.0 for j in range(W)] for i in range(H)]
    tokens = []
    for gi in range(grid_h):
        for gj in range(grid_w):
            # Average patch values
            patch_vals = []
            for di in range(downsample):
                for dj in range(downsample):
                    patch_vals.append(image[gi * downsample + di][gj * downsample + dj])
            patch_mean = sum(patch_vals) / len(patch_vals)
            # Find nearest code
            best_dist = float('inf')
            best_k = 0
            for k in range(vocab_size):
                d = abs(patch_mean - codebook[k][0])
                if d < best_dist:
                    best_dist = d
                    best_k = k
            tokens.append(best_k)

    print(f"\n[3b] Image Tokenization ({H}×{W} → {grid_h}×{grid_w} grid):")
    print(f"  Downsample factor: {downsample}x")
    print(f"  Number of tokens: {n_tokens}")
    print(f"  Token grid:")
    idx = 0
    for gi in range(grid_h):
        row = []
        for gj in range(grid_w):
            row.append(f"{tokens[idx]:2d}")
            idx += 1
        print(f"    [{', '.join(row)}]")

    # 3c) Token sequence for autoregressive model
    print(f"\n[3c] Token Sequence for Autoregressive Model:")
    print(f"  Flatten grid: {tokens}")
    print(f"  Sequence length: {len(tokens)}")
    print(f"  Each token is an index into vocabulary of size {vocab_size}")
    print(f"\n  Autoregressive model learns: p(t_i | t_1, ..., t_{{i-1}})")
    print(f"  Generates one token at a time, left-to-right, top-to-bottom")

    # 3d) Scaling and compression
    print(f"\n[3d] Scaling Analysis:")
    for img_size in [64, 128, 256, 512]:
        for downsample in [4, 8, 16]:
            tokens_count = (img_size // downsample) ** 2
            compression = (img_size * img_size) / tokens_count
            print(f"  {img_size:3d}×{img_size:3d}, downsample={downsample:2d}x → "
                  f"{tokens_count:5d} tokens (compress {compression:.0f}x)")

    print(f"\n  DALL-E 2: 256×256 image → 32×32 = 1024 tokens")
    print(f"  Parti: 512×512 → 64×64 = 4096 tokens")
    print(f"\n  Key advantage: discrete tokens enable:")
    print(f"    - Transformer architecture (same as text)")
    print(f"    - Shared vocabulary across modalities")
    print(f"    - Autoregressive generation with exact likelihood")


# ─────────────────────────── DEMO 4: Generation Quality ───────────────────────────

def demo_generation_quality():
    """
    FID concept, mode coverage, diversity vs fidelity tradeoff.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Generation Quality — FID, mode coverage, diversity vs fidelity")
    print("=" * 70)

    # 4a) FID (Fréchet Inception Distance)
    print(f"\n[4a] Fréchet Inception Distance (FID):")
    print(f"  FID measures distance between real and generated image distributions")
    print(f"  Formula: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{{1/2}})")
    print(f"  Lower FID = better quality")

    # Simulate real vs generated features
    random.seed(42)
    n_real = 100
    n_gen = 100
    feat_dim = 4

    real_features = [[random.gauss(0.5, 0.3) for _ in range(feat_dim)] for _ in range(n_real)]
    gen_features = [[random.gauss(0.5, 0.35) for _ in range(feat_dim)] for _ in range(n_gen)]

    # Compute means
    real_mean = [sum(f[i] for f in real_features) / n_real for i in range(feat_dim)]
    gen_mean = [sum(f[i] for f in gen_features) / n_gen for i in range(feat_dim)]

    # Compute variance
    real_var = [sum((f[i] - real_mean[i]) ** 2 for f in real_features) / n_real for i in range(feat_dim)]
    gen_var = [sum((f[i] - gen_mean[i]) ** 2 for f in gen_features) / n_gen for i in range(feat_dim)]

    # Simplified FID (Euclidean between means + variance difference)
    mean_diff = sum((real_mean[i] - gen_mean[i]) ** 2 for i in range(feat_dim))
    var_diff = sum(abs(real_var[i] - gen_var[i]) for i in range(feat_dim))
    fid = mean_diff + var_diff

    print(f"\n  Real features:  mean={[round(m, 3) for m in real_mean]}")
    print(f"  Gen features:   mean={[round(m, 3) for m in gen_mean]}")
    print(f"  Mean distance²: {mean_diff:.4f}")
    print(f"  Variance diff:  {var_diff:.4f}")
    print(f"  FID (simplified): {fid:.4f}")

    # 4b) Mode coverage
    print(f"\n[4b] Mode Coverage:")
    n_modes = 5
    samples_per_mode = 20

    print(f"  {n_modes} modes in real distribution")
    print(f"  Each mode has {samples_per_mode} samples")

    # Real distribution: balanced across modes
    real_modes = list(range(n_modes)) * samples_per_mode

    # Good generator: covers all modes
    good_gen = []
    for m in range(n_modes):
        good_gen.extend([m] * (samples_per_mode + random.randint(-2, 2)))

    # Bad generator: mode collapse
    bad_gen = [0] * (n_modes * samples_per_mode)  # only mode 0

    # Compute mode coverage
    real_mode_set = set(real_modes)
    good_mode_set = set(good_gen)
    bad_mode_set = set(bad_gen)

    print(f"\n  Real distribution: modes = {sorted(real_mode_set)}")
    print(f"  Good generator:    modes = {sorted(good_mode_set)} (coverage: {len(good_mode_set)/n_modes:.0%})")
    print(f"  Bad generator:     modes = {sorted(bad_mode_set)} (coverage: {len(bad_mode_set)/n_modes:.0%})")

    # 4c) Diversity vs Fidelity tradeoff
    print(f"\n[4c] Diversity vs Fidelity Tradeoff:")
    print(f"  ┌─────────────────┬────────────┬────────────┬──────────────┐")
    print(f"  │ Method          │ FID (↓)    │ Diversity  │ Mode Collapse│")
    print(f"  ├─────────────────┼────────────┼────────────┼──────────────┤")

    methods = [
        ("GAN (original)",  "high",     "low",      "severe"),
        ("GAN (WGAN-GP)",   "medium",   "medium",   "moderate"),
        ("VAE",             "high",     "high",     "none"),
        ("VQ-VAE + AR",     "low",      "medium",   "rare"),
        ("Diffusion",       "very low", "high",     "none"),
        ("DALL-E 2",        "very low", "high",     "none"),
    ]

    for method, fid, diversity, collapse in methods:
        print(f"  │ {method:<15s} │ {fid:<10s} │ {diversity:<10s} │ {collapse:<12s} │")
    print(f"  └─────────────────┴────────────┴────────────┴──────────────┘")

    # 4d) Generation quality metrics
    print(f"\n[4d] Generation Quality Metrics:")
    print(f"  FID (Fréchet Inception Distance):")
    print(f"    - Extract features from Inception-v3")
    print(f"    - Compare real vs generated feature distributions")
    print(f"    - Lower = better, state-of-art: ~1-5 on ImageNet")
    print(f"\n  IS (Inception Score):")
    print(f"    - Measures quality AND diversity")
    print(f"    - Higher = better, but less reliable than FID")
    print(f"\n  Precision / Recall:")
    print(f"    - Precision: quality of generated samples")
    print(f"    - Recall: coverage of real distribution")
    print(f"    - Can trade off independently")

    print(f"\n  Practical tips:")
    print(f"    - Use FID-50K (50,000 samples for reliable measurement)")
    print(f"    - Report FID at multiple resolutions")
    print(f"    - Compare against same evaluation protocol")
    print(f"    - Human evaluation still important for perceptual quality")


# ─────────────────────────── Main ───────────────────────────

if __name__ == "__main__":
    demo_autoregressive_generation()
    demo_vq_vae()
    demo_discrete_tokens()
    demo_generation_quality()
    print("\n" + "=" * 70)
    print("All Image Generation demos complete!")
    print("=" * 70)
