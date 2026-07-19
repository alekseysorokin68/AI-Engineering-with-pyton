"""115 — Inference Optimization: спекулятивное декодирование, непрерывный батчинг, paged attention

Темы:
  1. Speculative Decoding (draft model, verify, reject sampling)
  2. Continuous Batching (iteration-level scheduling, Inflight batching)
  3. Paged Attention (virtual memory for KV cache, block management)
  4. Quantization Effects (speed vs quality tradeoffs)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import time
import collections

random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Demo 1 — Speculative Decoding
# ──────────────────────────────────────────────────────────────────────────────

def demo_speculative_decoding():
    """Speculative decoding with draft model, verification, and rejection sampling."""
    print("=" * 70)
    print("DEMO 1 — Speculative Decoding: draft model, verify, reject sampling")
    print("=" * 70)

    # --- Sub-example 1: Draft-then-verify pipeline ---
    print("\n[1.1] Draft-then-verify pipeline")
    draft_probs = [0.6, 0.55, 0.7, 0.4, 0.65]  # draft model confidence
    target_probs = [0.8, 0.3, 0.9, 0.7, 0.5]   # target model confidence
    tokens = ["The", " cat", " sat", " on", " the"]
    gamma = 3  # draft length

    print(f"  Draft model proposes {gamma} tokens, target model verifies all")
    accepted = 0
    for i in range(gamma):
        draft_t = draft_probs[i]
        target_t = target_probs[i]
        # Acceptance criterion: min(1, p_target / p_draft)
        accept_prob = min(1.0, target_t / draft_t) if draft_t > 0 else 0
        accepted_this = random.random() < accept_prob
        status = "ACCEPTED" if accepted_this else f"REJECTED (draft={draft_t:.2f}, target={target_t:.2f})"
        print(f"  Token '{tokens[i]}': draft_p={draft_t:.2f} target_p={target_t:.2f} "
              f"accept_prob={accept_prob:.2f} → {status}")
        if accepted_this:
            accepted += 1
        else:
            break
    print(f"  Accepted {accepted}/{gamma} drafted tokens")

    # --- Sub-example 2: Acceptance rate calculation ---
    print("\n[1.2] Theoretical acceptance rate")
    all_draft = [0.7, 0.8, 0.6, 0.9, 0.5, 0.75, 0.85, 0.4, 0.65, 0.7]
    all_target = [0.8, 0.7, 0.7, 0.95, 0.6, 0.8, 0.9, 0.5, 0.7, 0.75]
    total_accepted = 0
    for d, t in zip(all_draft, all_target):
        acc_prob = min(1.0, t / d) if d > 0 else 0
        if random.random() < acc_prob:
            total_accepted += 1
    accept_rate = total_accepted / len(all_draft)
    avg_speedup = 1 / (1 - accept_rate * (1 - 1/len(all_draft))) if accept_rate < 1 else len(all_draft)
    print(f"  Draft tokens: {len(all_draft)}")
    print(f"  Accepted: {total_accepted}/{len(all_draft)} = {accept_rate:.1%}")
    print(f"  Expected speedup: ~{1 + accept_rate * (len(all_draft)-1):.1f}x")

    # --- Sub-example 3: Memory/compute tradeoff ---
    print("\n[1.3] Compute savings from speculative decoding")
    target_flops_per_token = 1e12
    draft_flops_per_token = 5e10  # 20x smaller model
    gamma = 4
    accept_rate = 0.7

    # Standard decoding: gamma forward passes of target model
    standard_flops = gamma * target_flops_per_token
    # Speculative: 1 target pass + gamma draft passes
    speculative_flops = target_flops_per_token + gamma * draft_flops_per_token
    # After rejection, re-run target from rejection point
    expected_reject_pos = gamma * (1 - accept_rate)
    extra_target = expected_reject_pos * target_flops_per_token

    total_spec_flops = speculative_flops + extra_target
    savings = (1 - total_spec_flops / standard_flops) * 100
    print(f"  Standard ({gamma} target passes): {standard_flops:.0e} FLOPS")
    print(f"  Speculative: 1 target + {gamma} draft + ~{expected_reject_pos:.1f} target re-runs")
    print(f"  Speculative total: {total_spec_flops:.0e} FLOPS")
    print(f"  Compute savings: {savings:.1f}%")

    # --- Sub-example 4: Rejection sampling distribution ---
    print("\n[1.4] Rejection sampling detail")
    q = [0.3, 0.25, 0.2, 0.15, 0.1]   # draft distribution
    p = [0.2, 0.3, 0.25, 0.15, 0.1]   # target distribution
    print(f"  Draft distribution q: {q}")
    print(f"  Target distribution p: {p}")
    print(f"  Acceptance ratios p/q:")
    for i in range(len(q)):
        ratio = min(1.0, p[i] / q[i]) if q[i] > 0 else 0
        print(f"    Token {i}: p={p[i]:.2f}/q={q[i]:.2f} = {ratio:.2f} "
              f"→ accept with prob {ratio:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 2 — Continuous Batching
# ──────────────────────────────────────────────────────────────────────────────

def demo_continuous_batching():
    """Iteration-level scheduling and inflight batching."""
    print("\n" + "=" * 70)
    print("DEMO 2 — Continuous Batching: iteration-level scheduling, Inflight batching")
    print("=" * 70)

    # --- Sub-example 1: Static vs continuous batching ---
    print("\n[2.1] Static vs Continuous Batching comparison")
    # Static: wait for longest sequence
    sequences = [
        {"id": "A", "total_tokens": 10, "remaining": 10},
        {"id": "B", "total_tokens": 5,  "remaining": 5},
        {"id": "C", "total_tokens": 8,  "remaining": 8},
    ]
    print("  Static batching (wait for all to finish):")
    static_time = max(s["total_tokens"] for s in sequences)
    print(f"    Batch runs for {static_time} iterations (longest sequence)")
    for s in sequences:
        idle = static_time - s["total_tokens"]
        print(f"    Seq {s['id']}: done at iter {s['total_tokens']}, "
              f"idles {idle} iterations ({100*idle/static_time:.0f}% waste)")

    # Continuous: slots free up immediately
    print("\n  Continuous batching (iteration-level scheduling):")
    inflight = [dict(s) for s in sequences]
    iteration = 0
    total_tokens_generated = 0
    while inflight:
        iteration += 1
        finished = []
        for seq in inflight:
            seq["remaining"] -= 1
            total_tokens_generated += 1
            if seq["remaining"] <= 0:
                finished.append(seq)
        for f in finished:
            inflight.remove(f)
            print(f"    Iter {iteration}: Seq {f['id']} finished")
    print(f"    Continuous batch: {iteration} iterations total, 0 idle time")

    # --- Sub-example 2: Inflight request scheduling ---
    print("\n[2.2] Inflight request scheduling simulation")
    max_batch = 4
    arrivals = [
        {"id": "R1", "arrival": 0, "gen_tokens": 5, "remaining": 5},
        {"id": "R2", "arrival": 1, "gen_tokens": 3, "remaining": 3},
        {"id": "R3", "arrival": 2, "gen_tokens": 4, "remaining": 4},
        {"id": "R4", "arrival": 3, "gen_tokens": 2, "remaining": 2},
        {"id": "R5", "arrival": 4, "gen_tokens": 6, "remaining": 6},
    ]
    inflight_queue = []
    completed = []

    for tick in range(12):
        # Add arriving requests
        for r in arrivals:
            if r["arrival"] == tick and r["remaining"] > 0:
                inflight_queue.append(r)
                print(f"  Tick {tick}: {r['id']} arrived → inflight ({len(inflight_queue)} active)")

        # Process one iteration for each inflight request
        still_active = []
        for r in inflight_queue:
            r["remaining"] -= 1
            if r["remaining"] <= 0:
                completed.append(r)
                print(f"  Tick {tick}: {r['id']} completed")
            else:
                still_active.append(r)
        inflight_queue = still_active

        if not inflight_queue and tick > 5:
            break

    print(f"  All {len(completed)} requests completed in {tick+1} ticks")

    # --- Sub-example 3: Batching efficiency ---
    print("\n[2.3] Batching efficiency metrics")
    gpu_total_iters = 20
    active_iters = 15
    batch_sizes = [4, 4, 4, 3, 0]
    avg_batch = sum(batch_sizes) / len(batch_sizes)
    gpu_util = active_iters / gpu_total_iters
    print(f"  Total iterations: {gpu_total_iters}")
    print(f"  Active iterations: {active_iters}")
    print(f"  GPU utilization: {gpu_util:.0%}")
    print(f"  Average batch size: {avg_batch:.1f}")
    print(f"  Batches processed: {len(batch_sizes)}")
    for i, bs in enumerate(batch_sizes):
        bar = "█" * bs + "░" * (4 - bs)
        print(f"    Iter {i}: [{bar}] batch_size={bs}")

    # --- Sub-example 4: Priority scheduling ---
    print("\n[2.4] Priority-based request scheduling")
    priority_requests = [
        {"id": "P1", "priority": 1, "tokens": 3, "remaining": 3},
        {"id": "P2", "priority": 3, "tokens": 2, "remaining": 2},
        {"id": "P3", "priority": 2, "tokens": 4, "remaining": 4},
    ]
    print(f"  Max concurrent: 2")
    for tick in range(6):
        # Sort by priority (lower = higher priority)
        active = sorted([r for r in priority_requests if r["remaining"] > 0],
                       key=lambda x: x["priority"])[:2]
        tick_info = []
        for r in active:
            r["remaining"] -= 1
            tick_info.append(f"{r['id']}(p{r['priority']})")
            if r["remaining"] == 0:
                tick_info[-1] += "✓"
        print(f"  Tick {tick}: processing {', '.join(tick_info)}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 3 — Paged Attention
# ──────────────────────────────────────────────────────────────────────────────

def demo_paged_attention():
    """Virtual memory for KV cache, block management."""
    print("\n" + "=" * 70)
    print("DEMO 3 — Paged Attention: virtual memory for KV cache, block management")
    print("=" * 70)

    # --- Sub-example 1: KV cache fragmentation problem ---
    print("\n[3.1] KV Cache fragmentation without paging")
    seq_lengths = [128, 256, 64, 512, 192]
    max_seq = max(seq_lengths)
    total_allocated = len(seq_lengths) * max_seq
    total_used = sum(seq_lengths)
    fragmentation = (1 - total_used / total_allocated) * 100
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Max length: {max_seq}")
    print(f"  Pre-allocated (naive): {total_allocated} slots")
    print(f"  Actually used: {total_used} slots")
    print(f"  Memory waste: {fragmentation:.1f}%")
    print(f"  Wasted slots: {total_allocated - total_used}")

    # --- Sub-example 2: Paged allocation ---
    print("\n[3.2] Paged KV Cache allocation")
    block_size = 16  # tokens per block
    seq_lengths = [128, 256, 64, 512, 192]
    blocks_needed = []
    for sl in seq_lengths:
        blocks = math.ceil(sl / block_size)
        blocks_needed.append(blocks)
        waste = blocks * block_size - sl
        print(f"  Seq len {sl:4d}: needs {blocks:2d} blocks, waste={waste} tokens")
    total_blocks = sum(blocks_needed)
    total_paged = total_blocks * block_size
    print(f"  Total blocks: {total_blocks}, total slots: {total_paged}")
    print(f"  Total waste: {total_paged - sum(seq_lengths)} "
          f"({(1-sum(seq_lengths)/total_paged)*100:.1f}%)")
    print(f"  vs naive waste: {(max(seq_lengths)*len(seq_lengths)-sum(seq_lengths))} "
          f"({(1-sum(seq_lengths)/(max(seq_lengths)*len(seq_lengths)))*100:.1f}%)")

    # --- Sub-example 3: Block table mapping ---
    print("\n[3.3] Block table (virtual → physical mapping)")
    physical_blocks = 20
    block_table = {}
    for seq_id, sl in enumerate(["A", "B", "C"], 1):
        blocks = random.sample(range(physical_blocks), 2)
        block_table[seq_id] = {"blocks": blocks, "seq_len": [128, 256, 64][seq_id-1]}
    print(f"  Physical blocks available: {physical_blocks}")
    print(f"  Block tables:")
    for seq_id, info in block_table.items():
        print(f"    Seq {seq_id}: blocks={info['blocks']}, seq_len={info['seq_len']}")
    print(f"  → Physical memory is non-contiguous, like virtual memory pages")

    # --- Sub-example 4: Copy-on-write for shared prefixes ---
    print("\n[3.4] Copy-on-Write for shared KV cache prefixes")
    shared_prefix = ["<system>", "You", "are", "a", "helpful", "assistant", "about"]
    conversations = [
        ["Tell", "me", "about", "AI"],
        ["What", "is", "Python?"],
        ["Explain", "deep", "learning"],
    ]
    # Shared prefix blocks
    prefix_blocks = [0, 1]
    print(f"  Shared prefix: {shared_prefix}")
    print(f"  Prefix blocks: {prefix_blocks} (shared, read-only)")
    for i, conv in enumerate(conversations, 1):
        new_blocks = random.sample(range(2, 20), 2)
        print(f"  Conv {i}: prefix blocks={prefix_blocks} + new blocks={new_blocks}")
        print(f"    → Only {len(new_blocks)} blocks allocated (not {len(prefix_blocks)+len(new_blocks)})")
    print(f"  Memory saved by CoW: ~{len(prefix_blocks) * len(conversations) * 16} tokens")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 4 — Quantization Effects
# ──────────────────────────────────────────────────────────────────────────────

def demo_quantization_effects():
    """Speed vs quality tradeoffs in quantization."""
    print("\n" + "=" * 70)
    print("DEMO 4 — Quantization Effects: speed vs quality tradeoffs")
    print("=" * 70)

    # --- Sub-example 1: Precision levels ---
    print("\n[4.1] Model precision levels comparison")
    precisions = {
        "FP32": {"bits": 32, "size_gb": 64.0, "speed_tflops": 1.0,  "quality": 100},
        "FP16": {"bits": 16, "size_gb": 32.0, "speed_tflops": 2.0,  "quality": 99.9},
        "INT8": {"bits": 8,  "size_gb": 16.0, "speed_tflops": 3.5,  "quality": 99.5},
        "INT4": {"bits": 4,  "size_gb": 8.0,  "speed_tflops": 6.0,  "quality": 98.0},
        "INT2": {"bits": 2,  "size_gb": 4.0,  "speed_tflops": 10.0, "quality": 85.0},
    }
    print(f"  {'Precision':<10} {'Bits':>5} {'Size(GB)':>10} {'Speed':>8} {'Quality':>8}")
    print(f"  {'-'*45}")
    for name, info in precisions.items():
        print(f"  {name:<10} {info['bits']:>5} {info['size_gb']:>10.1f} "
              f"{info['speed_tflops']:>7.1f}x {info['quality']:>7.1f}%")

    # --- Sub-example 2: Quantization impact on perplexity ---
    print("\n[4.2] Perplexity degradation by quantization level")
    base_perplexity = 5.2
    for name, info in precisions.items():
        degradation = (100 - info["quality"]) / 100 * 2.0  # rough perplexity increase
        quant_ppl = base_perplexity + degradation
        print(f"  {name:<8}: perplexity {quant_ppl:.2f} "
              f"(Δ={degradation:+.2f}, quality={info['quality']:.1f}%)")

    # --- Sub-example 3: Throughput vs memory ---
    print("\n[4.3] Throughput vs Memory tradeoff")
    model_size_fp16 = 32  # GB for 70B model
    gpu_memory = 48  # A100 48GB

    for name, info in precisions.items():
        model_size = model_size_fp16 * (info["bits"] / 16)
        fits = "YES" if model_size <= gpu_memory else "NO"
        throughput = info["speed_tflops"]
        print(f"  {name:<8}: model={model_size:.0f}GB fits_48GB={fits:<3} "
              f"throughput={throughput:.1f}x")

    # --- Sub-example 4: Mixed-precision strategy ---
    print("\n[4.4] Mixed-precision quantization strategy")
    layers = [
        ("attention.q_proj", "INT8", 0.8),
        ("attention.k_proj", "INT8", 0.8),
        ("attention.v_proj", "INT4", 1.5),
        ("attention.o_proj", "INT4", 1.5),
        ("mlp.gate_proj", "INT4", 1.5),
        ("mlp.up_proj", "INT4", 1.5),
        ("mlp.down_proj", "INT8", 0.8),
        ("embed_tokens", "FP16", 1.0),
        ("lm_head", "FP16", 1.0),
    ]
    total_params = 70e9
    total_speedup = 0
    print(f"  Layer-level quantization for 70B model:")
    for layer_name, precision, speedup in layers:
        params_frac = 1 / len(layers)
        total_speedup += params_frac * speedup
        print(f"    {layer_name:<20}: {precision:<4} (speedup={speedup:.1f}x)")
    print(f"  Overall speedup: {total_speedup:.2f}x vs FP16")
    print(f"  Memory reduction: ~{(1 - total_speedup/2) * 100:.0f}%")


if __name__ == "__main__":
    demo_speculative_decoding()
    demo_continuous_batching()
    demo_paged_attention()
    demo_quantization_effects()
