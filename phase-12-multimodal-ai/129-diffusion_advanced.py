"""
129 — Diffusion Models Advanced: DDIM, classifier-free guidance, ControlNet concepts

Темы:
  1. DDIM Sampling (deterministic sampling, fewer steps, interpolation in latent space)
  2. Classifier-Free Guidance (conditional vs unconditional, guidance scale)
  3. Noise Schedules (linear, cosine, signal-to-noise ratio)
  4. Conditioning Mechanisms (text conditioning, image conditioning, ControlNet idea)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ---------------------------------------------------------------------------
# 1. DDIM Sampling — deterministic denoising with fewer steps
# ---------------------------------------------------------------------------
def demo_ddim_sampling():
    """Denoising Diffusion Implicit Models — skip steps, stay deterministic."""
    print("=" * 70)
    print("DEMO 1: DDIM Sampling — deterministic, fewer-step denoising")
    print("=" * 70)

    # --- 1a. Linear beta schedule + cumulative alphas ---
    T = 50  # full diffusion steps (compact for demo)
    beta_start, beta_end = 0.0001, 0.02
    betas = [beta_start + (beta_end - beta_start) * t / (T - 1) for t in range(T)]
    alphas = [1.0 - b for b in betas]
    alpha_bar = [alphas[0]]
    for a in alphas[1:]:
        alpha_bar.append(alpha_bar[-1] * a)

    print("\n--- 1a. Alpha-bar schedule (first 5 of {} steps) ---".format(T))
    for i in range(5):
        print(f"  t={i}: beta={betas[i]:.6f}  alpha={alphas[i]:.6f}  "
              f"alpha_bar={alpha_bar[i]:.6f}")

    # --- 1b. Forward diffusion: add noise to a clean signal ---
    x0 = 0.8  # clean scalar value
    t = 20
    eps = random.gauss(0, 1)  # sampled noise
    xt = math.sqrt(alpha_bar[t]) * x0 + math.sqrt(1 - alpha_bar[t]) * eps
    print(f"\n--- 1b. Forward diffusion at t={t} ---")
    print(f"  x0={x0}, eps={eps:.4f}, alpha_bar[{t}]={alpha_bar[t]:.6f}")
    print(f"  xt = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*eps")
    print(f"  xt = {xt:.6f}")

    # --- 1c. DDIM update formula (eta=0 → deterministic) ---
    eta = 0.0  # eta=0 is fully deterministic DDIM
    t_prev = t - 5
    eps_pred = random.gauss(0, 1)  # pretend model predicts noise

    # DDIM formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * predicted_x0
    #   + sqrt(1 - alpha_bar_{t-1} - sigma^2) * eps_pred + sigma * noise
    predicted_x0 = (xt - math.sqrt(1 - alpha_bar[t]) * eps_pred) / math.sqrt(alpha_bar[t])
    sigma = eta * math.sqrt((1 - alpha_bar[t_prev]) / (1 - alpha_bar[t])) * \
            math.sqrt(1 - alpha_bar[t] / alpha_bar[t_prev])
    noise_term = random.gauss(0, 1) if sigma > 0 else 0
    x_prev = (math.sqrt(alpha_bar[t_prev]) * predicted_x0 +
              math.sqrt(max(0, 1 - alpha_bar[t_prev] - sigma**2)) * eps_pred +
              sigma * noise_term)

    print(f"\n--- 1c. DDIM step t={t} → t'={t_prev} (eta={eta}) ---")
    print(f"  eps_pred={eps_pred:.4f}")
    print(f"  predicted_x0 = {predicted_x0:.6f}")
    print(f"  sigma={sigma:.6f}  (deterministic when eta=0)")
    print(f"  x_prev (denoised) = {x_prev:.6f}")

    # --- 1d. Interpolation in latent space between two noise samples ---
    eps_a = [random.gauss(0, 1) for _ in range(6)]
    eps_b = [random.gauss(0, 1) for _ in range(6)]
    alpha_val = 0.3  # interpolation factor
    interpolated = [alpha_val * a + (1 - alpha_val) * b for a, b in zip(eps_a, eps_b)]

    print(f"\n--- 1d. Latent interpolation (alpha={alpha_val}) ---")
    print(f"  eps_a (first 6): {[f'{v:.3f}' for v in eps_a]}")
    print(f"  eps_b (first 6): {[f'{v:.3f}' for v in eps_b]}")
    print(f"  interpolated:    {[f'{v:.3f}' for v in interpolated]}")
    print("  DDIM enables smooth interpolation because mapping is deterministic!")


# ---------------------------------------------------------------------------
# 2. Classifier-Free Guidance — conditional + unconditional predictions
# ---------------------------------------------------------------------------
def demo_classifier_free_guidance():
    """Guide generation with a conditional model using guidance scale w."""
    print("\n" + "=" * 70)
    print("DEMO 2: Classifier-Free Guidance — conditional vs unconditional")
    print("=" * 70)

    T = 50
    beta_start, beta_end = 0.0001, 0.02
    betas = [beta_start + (beta_end - beta_start) * t / (T - 1) for t in range(T)]
    alphas = [1.0 - b for b in betas]
    alpha_bar = [alphas[0]]
    for a in alphas[1:]:
        alpha_bar.append(alpha_bar[-1] * a)

    # Simulate a "model" that returns different eps for conditioned vs unconditioned
    def model_unconditional(x_t, t):
        random.seed(hashlib.md5(f"uncond_{t}".encode()).hexdigest()[:8], version=1)
        return random.gauss(0, 1)

    def model_conditional(x_t, t, class_label):
        random.seed(hashlib.md5(f"cond_{class_label}_{t}".encode()).hexdigest()[:8], version=1)
        return random.gauss(0.5, 0.5)  # biased toward class

    t = 25
    x_t = 0.3
    class_label = "cat"
    w = 7.5  # guidance scale

    # --- 2a. Unconditional prediction ---
    eps_uncond = model_unconditional(x_t, t)
    print(f"\n--- 2a. Unconditional noise prediction at t={t} ---")
    print(f"  eps_uncond = {eps_uncond:.4f}")

    # --- 2b. Conditional prediction ---
    eps_cond = model_conditional(x_t, t, class_label)
    print(f"\n--- 2b. Conditional noise prediction (class='{class_label}') ---")
    print(f"  eps_cond = {eps_cond:.4f}")

    # --- 2c. CFG formula: eps_guided = eps_uncond + w * (eps_cond - eps_uncond) ---
    eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
    print(f"\n--- 2c. Classifier-Free Guidance (w={w}) ---")
    print(f"  Formula: eps_guided = eps_uncond + w * (eps_cond - eps_uncond)")
    print(f"  eps_guided = {eps_uncond:.4f} + {w} * ({eps_cond:.4f} - {eps_uncond:.4f})")
    print(f"  eps_guided = {eps_guided:.4f}")

    # --- 2d. Effect of varying guidance scale ---
    print(f"\n--- 2d. Guidance scale sweep ---")
    print(f"  {'w':>6}  {'eps_guided':>12}  {'distance from uncond':>22}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*22}")
    for w_test in [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]:
        eps_g = eps_uncond + w_test * (eps_cond - eps_uncond)
        dist = abs(eps_g - eps_uncond)
        print(f"  {w_test:6.1f}  {eps_g:12.4f}  {dist:22.4f}")
    print("  Higher w → stronger class adherence but less diversity.")


# ---------------------------------------------------------------------------
# 3. Noise Schedules — linear, cosine, SNR
# ---------------------------------------------------------------------------
def demo_noise_schedules():
    """Compare linear and cosine noise schedules and their SNR curves."""
    print("\n" + "=" * 70)
    print("DEMO 3: Noise Schedules — linear, cosine, signal-to-noise ratio")
    print("=" * 70)

    T = 50

    # --- 3a. Linear schedule ---
    def linear_alpha_bar(t, T):
        return 1.0 - t / T

    # --- 3b. Cosine schedule (Nichol & Dhariwal) ---
    def cosine_alpha_bar(t, T, s=0.008):
        return math.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2

    print(f"\n--- 3a & 3b. Linear vs Cosine alpha_bar schedule (T={T}) ---")
    print(f"  {'t':>4}  {'linear':>10}  {'cosine':>10}  {'diff':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    for t in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
        ab_lin = linear_alpha_bar(t, T)
        ab_cos = cosine_alpha_bar(t, T)
        print(f"  {t:4d}  {ab_lin:10.6f}  {ab_cos:10.6f}  {ab_cos - ab_lin:+10.6f}")

    # --- 3c. Signal-to-noise ratio (SNR) ---
    print(f"\n--- 3c. Signal-to-Noise Ratio: SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)) ---")
    print(f"  {'t':>4}  {'SNR_lin':>12}  {'SNR_cos':>12}  {'SNR_lin dB':>12}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}")
    for t in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        ab_l = linear_alpha_bar(t, T)
        ab_c = cosine_alpha_bar(t, T)
        snr_l = ab_l / max(1e-8, 1 - ab_l)
        snr_c = ab_c / max(1e-8, 1 - ab_c)
        snr_db = 10 * math.log10(max(1e-10, snr_l))
        print(f"  {t:4d}  {snr_l:12.4f}  {snr_c:12.4f}  {snr_db:12.2f}")

    # --- 3d. Why cosine schedule matters ---
    print(f"\n--- 3d. Why cosine schedule matters ---")
    print("  Linear schedule drops signal too fast at high t → model must learn")
    print("  heavy denoising at low-SNR regime. Cosine schedule keeps signal")
    print("  longer, giving the model more balanced learning across all noise levels.")
    snr_ratio = (cosine_alpha_bar(40, T) / max(1e-8, 1 - cosine_alpha_bar(40, T))) / \
                max(1e-10, (linear_alpha_bar(40, T) / max(1e-8, 1 - linear_alpha_bar(40, T))))
    print(f"  At t=40: cosine SNR / linear SNR = {snr_ratio:.2f}x")


# ---------------------------------------------------------------------------
# 4. Conditioning Mechanisms — text, image, ControlNet
# ---------------------------------------------------------------------------
def demo_conditioning_mechanisms():
    """How diffusion models accept text, image, and structural conditioning."""
    print("\n" + "=" * 70)
    print("DEMO 4: Conditioning Mechanisms — text, image, ControlNet")
    print("=" * 70)

    # --- 4a. Text conditioning via cross-attention ---
    def simple_text_encoder(text):
        """Simulate text embedding by hashing tokens."""
        tokens = text.lower().split()
        embedding = [0.0] * 4
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            for i in range(4):
                embedding[i] += ((h >> (8 * i)) & 0xFF) / 255.0
        norm = math.sqrt(sum(v**2 for v in embedding)) or 1.0
        return [v / norm for v in embedding]

    prompt = "a beautiful sunset over the ocean"
    text_emb = simple_text_encoder(prompt)
    print(f"\n--- 4a. Text conditioning (cross-attention) ---")
    print(f"  Prompt: '{prompt}'")
    print(f"  Simulated embedding (dim=4): {[f'{v:.4f}' for v in text_emb]}")
    print(f"  This embedding is projected into K, V of cross-attention layers.")

    # --- 4b. Image conditioning via concat ---
    def simulate_image_conditioning(noisy_img, clean_condition):
        """Simulate concatenating a conditioning image channel-wise."""
        result = [0.5 * n + 0.5 * c for n, c in zip(noisy_img, clean_condition)]
        return result

    noisy = [random.gauss(0, 1) for _ in range(8)]
    clean = [random.gauss(0.5, 0.1) for _ in range(8)]
    conditioned = simulate_image_conditioning(noisy, clean)
    print(f"\n--- 4b. Image conditioning (channel concatenation) ---")
    print(f"  Noisy input (8 values): {[f'{v:.3f}' for v in noisy]}")
    print(f"  Clean condition:        {[f'{v:.3f}' for v in clean]}")
    print(f"  Conditioned (blended):  {[f'{v:.3f}' for v in conditioned]}")
    print(f"  Formula: x_conditioned = 0.5 * x_noisy + 0.5 * x_condition")

    # --- 4c. ControlNet concept: zero-conv + skip connections ---
    def zero_conv(x, weight):
        """Simulate zero-initialized convolution (starts outputting zeros)."""
        return [x_i * weight for x_i in x]

    def controlnet_forward(x, control_weight):
        """Simulate ControlNet's architecture: copy of encoder + zero conv."""
        encoder_out = [x_i * 0.8 for x_i in x]  # encoder features
        control_out = zero_conv(encoder_out, control_weight)
        skip = [e + c for e, c in zip(encoder_out, control_out)]
        return skip

    x_input = [random.gauss(0, 1) for _ in range(6)]
    print(f"\n--- 4c. ControlNet architecture concept ---")
    print(f"  Input features (6): {[f'{v:.3f}' for v in x_input]}")
    for cw in [0.0, 0.1, 0.5, 1.0]:
        out = controlnet_forward(x_input, cw)
        print(f"  zero_conv_weight={cw:.1f} → output: {[f'{v:.3f}' for v in out]}")
    print("  Zero-conv weight=0 → ControlNet starts as identity (safe training).")
    print("  Gradually learns to condition via skip connections to UNet.")

    # --- 4d. Conditioning strength comparison ---
    print(f"\n--- 4d. Conditioning strength trade-offs ---")
    print(f"  {'Method':<25} {'Strength':<12} {'Flexibility':<15} {'Use case'}")
    print(f"  {'-'*25} {'-'*12} {'-'*15} {'-'*30}")
    methods = [
        ("Cross-attention (text)", "High", "Very flexible", "Text-to-image"),
        ("Channel concat (image)", "Very high", "Fixed resolution", "Inpainting, img2img"),
        ("ControlNet (edge/pose)", "Tunable", "Spatial control", "Structural guidance"),
        ("IP-Adapter (image)", "Medium", "Style transfer", "Visual prompting"),
    ]
    for method, strength, flex, use in methods:
        print(f"  {method:<25} {strength:<12} {flex:<15} {use}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_ddim_sampling()
    demo_classifier_free_guidance()
    demo_noise_schedules()
    demo_conditioning_mechanisms()
    print("\n" + "=" * 70)
    print("All 4 demos completed — 129-diffusion_advanced.py")
    print("=" * 70)
