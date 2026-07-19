"""138 — Efficient Multimodal: pruning, distillation, streaming for multiple modalities

Темы:
  1. Vision Encoder Efficiency (token pruning, early exit, resolution scaling)
  2. Cross-Modal Efficiency (attention bottlenecks, feature compression)
  3. Streaming Multimodal (incremental processing, memory management)
  4. Hardware Optimization (GPU memory, mixed precision, tensor parallelism)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# ---------------------------------------------------------------------------
# Demo 1 — Vision Encoder Efficiency
# ---------------------------------------------------------------------------

def demo_vision_encoder_efficiency():
    """
    Shows token pruning, early exit, and resolution scaling for vision encoders.
    """
    print("=" * 70)
    print("DEMO 1 — Vision Encoder Efficiency")
    print("=" * 70)

    # --- 1a. Token Pruning ---
    print("\n--- 1a. Token Pruning ---")
    n_tokens = 197  # ViT-B/16 on 224x224 → 196 patches + 1 cls
    attention_scores = [random.random() for _ in range(n_tokens)]

    # keep top-k tokens by attention score
    k = 50
    ranked = sorted(range(n_tokens), key=lambda i: attention_scores[i], reverse=True)
    pruned_tokens = ranked[:k]
    compression = k / n_tokens
    print(f"  Original tokens:  {n_tokens}")
    print(f"  After pruning:    {k}")
    print(f"  Compression:      {compression:.1%}")
    print(f"  FLOPs saved:      ~{(1 - compression)*100:.0f}% of self-attention")

    # --- 1b. Early Exit ---
    print("\n--- 1b. Early Exit from Transformer Layers ---")
    n_layers = 12
    layer_confidences = []
    for layer_idx in range(n_layers):
        # confidence grows with depth, with some noise
        base = 0.4 + 0.05 * layer_idx
        conf = min(1.0, base + random.gauss(0, 0.02))
        layer_confidences.append(conf)

    threshold = 0.90
    exit_layer = n_layers  # default: exit after last layer
    for i, c in enumerate(layer_confidences):
        if c >= threshold:
            exit_layer = i + 1
            break

    print(f"  Confidence per layer: {[f'{c:.2f}' for c in layer_confidences]}")
    print(f"  Threshold: {threshold}")
    print(f"  Early exit at layer: {exit_layer}/{n_layers}")
    print(f"  Layers skipped: {n_layers - exit_layer} "
          f"({(n_layers - exit_layer)/n_layers:.0%} savings)")

    # --- 1c. Resolution Scaling ---
    print("\n--- 1c. Dynamic Resolution Scaling ---")
    resolutions = [
        (112, 112, "thumbnail scan"),
        (224, 224, "standard"),
        (336, 336, "high-res detail"),
        (448, 448, "ultra high-res"),
    ]
    base_flops = 4.2e9  # ViT-B at 224
    for w, h, label in resolutions:
        scale = (w * h) / (224 * 224)
        flops = base_flops * scale
        print(f"  {w}x{h} ({label:>20s}): scale={scale:.2f}x, FLOPs={flops:.2e}")

    print("\n  Strategy: start at low-res, upsample only ambiguous regions.")
    savings = 1.0 - (112*112 + 224*224*0.3) / (224*224)
    print(f"  Estimated savings vs full 224: ~{savings:.0%}")

    # --- 1d. Combined efficiency gains ---
    print("\n--- 1d. Combined Efficiency Gains ---")
    pruning_gain = 1 - compression
    early_exit_gain = 1 - exit_layer / n_layers
    resolution_gain = savings
    combined = 1 - (1 - pruning_gain) * (1 - early_exit_gain) * (1 - resolution_gain)
    print(f"  Token pruning:       {pruning_gain:.0%}")
    print(f"  Early exit:          {early_exit_gain:.0%}")
    print(f"  Resolution scaling:  {resolution_gain:.0%}")
    print(f"  Combined speedup:    {combined:.0%} FLOPs reduction")
    print(f"  Estimated throughput: ~{1/(1-combined):.1f}x faster inference")


# ---------------------------------------------------------------------------
# Demo 2 — Cross-Modal Efficiency
# ---------------------------------------------------------------------------

def demo_cross_modal_efficiency():
    """
    Shows attention bottlenecks and feature compression for cross-modal models.
    """
    print("=" * 70)
    print("DEMO 2 — Cross-Modal Efficiency")
    print("=" * 70)

    # --- 2a. Cross-Attention Complexity ---
    print("\n--- 2a. Cross-Attention Complexity ---")
    n_visual = 256  # visual tokens
    n_text = 77     # text tokens
    d_model = 768
    n_heads = 12

    # full cross-attention: O(Nv × Nt × d)
    full_flops = n_visual * n_text * d_model
    print(f"  Visual tokens: {n_visual}, Text tokens: {n_text}, d_model: {d_model}")
    print(f"  Full cross-attention FLOPs: {full_flops:,}")

    # bottleneck: compress visual tokens to match text count
    bottleneck_k = 32
    bottleneck_flops = bottleneck_k * n_text * d_model + n_visual * bottleneck_k * d_model
    print(f"  With bottleneck (k={bottleneck_k}): {bottleneck_flops:,}")
    print(f"  Speedup: {full_flops / bottleneck_flops:.1f}x")

    # --- 2b. Feature Compression via Quantization ---
    print("\n--- 2b. Feature Compression via Quantization ---")
    bits_options = [32, 16, 8, 4, 2, 1]
    original_size_mb = 10.0  # 10 MB per feature map
    for bits in bits_options:
        compressed = original_size_mb * bits / 32
        ratio = bits / 32
        print(f"  {bits:2d}-bit: {compressed:.2f} MB ({ratio:.0%} of original)")

    # --- 2c. Sparse Cross-Modal Attention ---
    print("\n--- 2c. Sparse Cross-Modal Attention ---")
    total_pairs = n_visual * n_text
    sparsity_levels = [0.5, 0.7, 0.9, 0.95, 0.99]
    print(f"  Total visual-text pairs: {total_pairs}")
    for sparsity in sparsity_levels:
        active = int(total_pairs * (1 - sparsity))
        flops_saved = total_pairs * d_model * sparsity
        print(f"  Sparsity {sparsity:.0%}: {active:,} active pairs, "
              f"~{flops_saved/1e6:.1f}M FLOPs saved")

    # --- 2d. Feature Distillation ---
    print("\n--- 2d. Cross-Modal Feature Distillation ---")
    teacher_params = 1.3e9  # 1.3B param teacher
    student_configs = [
        ("small", 120e6),
        ("tiny", 44e6),
        ("micro", 12e6),
    ]
    print(f"  Teacher model: {teacher_params/1e6:.0f}M parameters")
    for name, params in student_configs:
        compression_ratio = teacher_params / params
        # estimate accuracy retention based on empirical curve
        acc = 100 * (1 - 0.3 * math.exp(-params / 50e6))
        print(f"  Student ({name:>6s}): {params/1e6:5.0f}M params, "
              f"compression={compression_ratio:.0f}x, est. accuracy={acc:.1f}%")


# ---------------------------------------------------------------------------
# Demo 3 — Streaming Multimodal
# ---------------------------------------------------------------------------

def demo_streaming_multimodal():
    """
    Shows incremental processing and memory management for streaming inputs.
    """
    print("=" * 70)
    print("DEMO 3 — Streaming Multimodal")
    print("=" * 70)

    # --- 3a. Incremental Frame Processing ---
    print("\n--- 3a. Incremental Frame Processing ---")
    total_frames = 150  # 5-second video at 30fps
    fps = 30
    window_size = 8     # process every 8th frame
    processed = list(range(0, total_frames, window_size))
    print(f"  Total frames: {total_frames} ({total_frames/fps:.1f}s video)")
    print(f"  Processing window: every {window_size}th frame")
    print(f"  Frames processed: {len(processed)}/{total_frames} "
          f"({len(processed)/total_frames:.0%})")

    # adaptive: more frames in motion-heavy segments
    motion_scores = [random.random() for _ in range(total_frames // window_size)]
    adaptive_frames = []
    for i, score in enumerate(motion_scores):
        base_frame = i * window_size
        if score > 0.8:
            adaptive_frames.extend([base_frame, base_frame + 2, base_frame + 4])
        elif score > 0.5:
            adaptive_frames.extend([base_frame, base_frame + 3])
        else:
            adaptive_frames.append(base_frame)
    print(f"  Adaptive frame selection: {len(adaptive_frames)} frames "
          f"({len(adaptive_frames)/total_frames:.0%})")

    # --- 3b. Sliding Window Memory ---
    print("\n--- 3b. Sliding Window Memory Management ---")
    max_context_tokens = 4096
    token_sizes = {"image": 257, "text": 77, "audio": 128}
    window = []
    history = []

    for modality, size in [("image", 257), ("text", 77), ("image", 257),
                           ("text", 77), ("audio", 128), ("image", 257)]:
        entry = {"modality": modality, "tokens": size}
        window.append(entry)
        total = sum(e["tokens"] for e in window)
        while total > max_context_tokens and len(window) > 1:
            evicted = window.pop(0)
            history.append(evicted)
            total = sum(e["tokens"] for e in window)
        print(f"  + {modality:>5s} ({size:3d} tok) → window={len(window)}, "
              f"total={total}, evicted={len(history)}")

    # --- 3c. Incremental Token Generation ---
    print("\n--- 3c. Incremental Token Generation (Streaming) ---")
    full_response = "The image shows a mountain landscape with snow-capped peaks."
    tokens = full_response.split()
    buffer = ""
    chunk_size = 3
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    print(f"  Full response: \"{full_response}\"")
    print(f"  Total tokens: {len(tokens)}, chunk size: {chunk_size}")
    print("  Streaming output:")
    for idx, chunk in enumerate(chunks):
        buffer += (" " if buffer else "") + " ".join(chunk)
        # simulate latency
        print(f"    Chunk {idx+1}: \"{buffer}\"")

    # --- 3d. KV-Cache Eviction ---
    print("\n--- 3d. KV-Cache Eviction Strategies ---")
    n_cached = 2048
    budget = 1024
    strategies = {
        "sliding window": list(range(n_cached)),  # keep last budget
        "token importance": sorted(range(n_cached),
                                   key=lambda i: random.random(), reverse=True),
    }
    for name, ranked in strategies.items():
        kept = ranked[:budget]
        evicted = n_cached - len(kept)
        print(f"  {name:>20s}: kept={len(kept)}, evicted={evicted}, "
              f"savings={evicted/n_cached:.0%}")

    # H2O (heavy-hitter oracle) — keep recent + high-attention
    recent = n_cached // 2
    h2o_indices = list(range(n_cached - recent, n_cached))
    h2o_scores = {i: random.random() for i in range(n_cached - recent)}
    top_important = sorted(h2o_scores, key=h2o_scores.get, reverse=True)
    h2o_indices.extend(top_important[:budget - recent])
    print(f"  {'H2O (recent+important)':>20s}: kept={len(h2o_indices)}, "
          f"evicted={n_cached - len(h2o_indices)}, savings={1-len(h2o_indices)/n_cached:.0%}")


# ---------------------------------------------------------------------------
# Demo 4 — Hardware Optimization
# ---------------------------------------------------------------------------

def demo_hardware_optimization():
    """
    Shows GPU memory management, mixed precision, and tensor parallelism.
    """
    print("=" * 70)
    print("DEMO 4 — Hardware Optimization")
    print("=" * 70)

    # --- 4a. GPU Memory Estimation ---
    print("\n--- 4a. GPU Memory Estimation ---")
    model_params = 7e9  # 7B parameter model
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
    gpu_memory_gb = 24

    print(f"  Model: {model_params/1e9:.0f}B parameters, GPU: {gpu_memory_gb}GB")
    for dtype, bpb in bytes_per_param.items():
        weight_gb = model_params * bpb / 1e9
        overhead = 1.2  # optimizer states, activations
        total = weight_gb * overhead
        fits = "FITS" if total <= gpu_memory_gb else "OOM"
        print(f"  {dtype:>4s}: weights={weight_gb:.1f}GB, "
              f"estimated total={total:.1f}GB → {fits}")

    # --- 4b. Mixed Precision Speedup ---
    print("\n--- 4b. Mixed Precision (FP16/BF16) Speedup ---")
    base_tflops = 31.2  # A100 FP32 TFLOPS
    fp16_tflops = 312    # A100 FP16 TFLOPS
    bf16_tflops = 312    # same throughput as FP16

    matmul_sizes = [1024, 2048, 4096, 8192]
    print(f"  GPU: NVIDIA A100 80GB")
    print(f"  FP32 TFLOPS: {base_tflops}, FP16 TFLOPS: {fp16_tflops}")
    for n in matmul_sizes:
        flops_fp32 = 2 * n**3  # standard matmul FLOPs
        time_fp32 = flops_fp32 / (base_tflops * 1e12)
        time_fp16 = flops_fp32 / (fp16_tflops * 1e12)
        speedup = time_fp32 / time_fp16
        print(f"  {n:5d}x{n:<5d}: FP32={time_fp32*1000:.2f}ms, "
              f"FP16={time_fp16*1000:.2f}ms, speedup={speedup:.1f}x")

    # --- 4c. Tensor Parallelism ---
    print("\n--- 4c. Tensor Parallelism ---")
    n_gpus_options = [1, 2, 4, 8]
    base_latency_ms = 500
    comm_overhead = 0.05  # 5% communication overhead per GPU

    print(f"  Base latency (1 GPU): {base_latency_ms}ms")
    for n_gpus in n_gpus_options:
        ideal = base_latency_ms / n_gpus
        comm = comm_overhead * (n_gpus - 1) * base_latency_ms
        actual = ideal + comm
        efficiency = base_latency_ms / (actual * n_gpus)
        print(f"  {n_gpus} GPUs: ideal={ideal:.1f}ms, "
              f"+comm={comm:.1f}ms, actual={actual:.1f}ms, "
              f"efficiency={efficiency:.0%}")

    # --- 4d. Quantization Impact ---
    print("\n--- 4d. Quantization Impact Analysis ---")
    model_size_gb = 14.0  # 7B at fp16
    benchmarks = [
        ("Wikitext perplexity", 8.5, [8.5, 8.6, 9.1, 12.3, 18.7]),
        ("MMLU accuracy (%)", 65.0, [65.0, 64.8, 63.2, 59.1, 52.3]),
    ]
    dtypes = ["fp16", "int8", "int4", "int3", "int2"]
    sizes = [model_size_gb * x for x in [1.0, 0.5, 0.25, 0.1875, 0.125]]

    header = f"  {'dtype':>5s} {'size':>7s}"
    for name, _, _ in benchmarks:
        header += f" {name:>22s}"
    print(header)

    for i, dtype in enumerate(dtypes):
        row = f"  {dtype:>5s} {sizes[i]:5.1f}GB"
        for _, baseline, values in benchmarks:
            val = values[i]
            delta = val - baseline
            sign = "+" if delta > 0 else ""
            row += f" {val:8.1f} ({sign}{delta:.1f})"
        print(row)

    print(f"\n  Recommendation: INT4 for inference, INT8 for fine-tuning")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_vision_encoder_efficiency()
    demo_cross_modal_efficiency()
    demo_streaming_multimodal()
    demo_hardware_optimization()
