"""
120 — Fine-tuning at Scale: распределённое обучение, DeepSpeed, efficient training

Темы:
  1. Data Preparation (format conversion, quality filtering, deduplication)
  2. Training Strategies (full fine-tuning, adapter, progressive)
  3. Distributed Concepts (data parallelism, model parallelism, pipeline)
  4. Hyperparameter Optimization (learning rate, batch size, warmup)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import time
import collections
import statistics

random.seed(42)


# ─────────────────────────────────────────────────────────────
# DEMO 1 — Data Preparation: format conversion, quality filtering, deduplication
# ─────────────────────────────────────────────────────────────
def demo_data_preparation():
    print("=" * 70)
    print("DEMO 1 — Data Preparation: format conversion, quality filtering, deduplication")
    print("=" * 70)

    # --- 1.1 Format conversion: raw text → instruction-response pairs ---
    print("\n--- 1.1 Format conversion: raw text → instruction-response pairs ---")
    raw_samples = [
        "The capital of France is Paris. It is located in the north-central part of the country.",
        "Python was created by Guido van Rossum in 1991. It emphasizes code readability.",
        "The capital of France is Paris. It is in the north-central part.",
        "Machine learning is a subset of artificial intelligence. It enables systems to learn from data.",
        "Python was created by Guido van Rossum in 1991. It focuses on readability.",
        "Deep learning uses neural networks with many layers. It excels at image recognition.",
        "Short.",
        "",
        "The capital of France is Paris. It is located in the north-central part of the country.",
    ]

    def to_instruction_format(text):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) < 2:
            return None
        return {
            "instruction": f"Explain the following concept",
            "input": sentences[0],
            "output": " ".join(sentences[1:]),
            "source_text": text
        }

    converted = []
    for t in raw_samples:
        pair = to_instruction_format(t)
        if pair:
            converted.append(pair)
    print(f"  Raw samples:       {len(raw_samples)}")
    print(f"  Converted pairs:   {len(converted)}")
    print(f"  Conversion rate:   {len(converted)/len(raw_samples)*100:.1f}%")
    print(f"  Example output:    {json.dumps(converted[0], indent=2)[:120]}...")

    # --- 1.2 Quality filtering: length, language, coherence heuristics ---
    print("\n--- 1.2 Quality filtering: length, language, coherence heuristics ---")

    def compute_quality_score(sample):
        text = sample.get("output", "")
        words = text.split()
        # Length score: penalize too short or too long
        word_count = len(words)
        if word_count < 3:
            length_score = 0.1
        elif word_count > 200:
            length_score = 0.5
        else:
            length_score = min(1.0, word_count / 30)
        # Vocabulary diversity
        if word_count > 0:
            unique_ratio = len(set(w.lower() for w in words)) / word_count
        else:
            unique_ratio = 0.0
        # Sentence structure: average sentence length
        sentences = re.split(r'[.!?]', text)
        avg_sent_len = statistics.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        structure_score = min(1.0, avg_sent_len / 15)
        # Combined
        score = 0.4 * length_score + 0.3 * unique_ratio + 0.3 * structure_score
        return round(score, 3)

    scored = [(s, compute_quality_score(s)) for s in converted]
    scored.sort(key=lambda x: x[1], reverse=True)

    for i, (s, score) in enumerate(scored[:5]):
        print(f"  Sample {i}: score={score:.3f}, words={len(s['output'].split())}")

    threshold = 0.3
    filtered = [s for s, sc in scored if sc >= threshold]
    print(f"\n  Quality threshold: {threshold}")
    print(f"  Before filtering:  {len(converted)}")
    print(f"  After filtering:   {len(filtered)}")
    print(f"  Removed:           {len(converted) - len(filtered)}")

    # --- 1.3 Deduplication: exact + fuzzy ---
    print("\n--- 1.3 Deduplication: exact hash + minhash-style fuzzy ---")

    def exact_hash(text):
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def fuzzy_signature(text, shingle_size=3):
        words = text.lower().split()
        shingles = set()
        for i in range(max(1, len(words) - shingle_size + 1)):
            shingle = " ".join(words[i:i + shingle_size])
            shingles.add(hashlib.md5(shingle.encode()).hexdigest()[:8])
        return shingles

    def jaccard_similarity(set_a, set_b):
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # Exact dedup
    seen_hashes = set()
    exact_deduped = []
    for s in filtered:
        h = exact_hash(s["output"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            exact_deduped.append(s)
    print(f"  After exact dedup: {len(exact_deduped)}")

    # Fuzzy dedup (Jaccard >= 0.7)
    fuzzy_threshold = 0.7
    sigs = [(s, fuzzy_signature(s["output"])) for s in exact_deduped]
    keep = [True] * len(sigs)
    comparisons = 0
    for i in range(len(sigs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(sigs)):
            if not keep[j]:
                continue
            comparisons += 1
            sim = jaccard_similarity(sigs[i][1], sigs[j][1])
            if sim >= fuzzy_threshold:
                keep[j] = False
    final_deduped = [s for s, k in zip(exact_deduped, keep) if k]
    print(f"  Fuzzy comparisons: {comparisons}")
    print(f"  After fuzzy dedup: {len(final_deduped)}")
    print(f"  Total pipeline:    {len(raw_samples)} → {len(final_deduped)} samples")
    print(f"  Data reduction:    {(1 - len(final_deduped)/len(raw_samples))*100:.1f}%")

    # --- 1.4 Tokenization analysis (character-level simulation) ---
    print("\n--- 1.4 Tokenization analysis: BPE-like vocabulary construction ---")

    def build_bpe_vocab(text, num_merges=20):
        # Character-level init
        chars = list(text)
        vocab = {}
        for c in chars:
            vocab[c] = vocab.get(c, 0) + 1
        merges = []
        for _ in range(num_merges):
            if len(chars) < 2:
                break
            pairs = collections.Counter()
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i + 1])] += 1
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0]
            token = best_pair[0][0] + best_pair[0][1]
            count = best_pair[1]
            merges.append((best_pair[0], count))
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair[0]:
                    new_chars.append(token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            vocab[token] = count
        return vocab, merges

    sample_text = " ".join(s["output"] for s in final_deduped[:3])
    vocab, merges = build_bpe_vocab(sample_text, num_merges=15)
    print(f"  Input text length:       {len(sample_text)} chars")
    print(f"  Initial vocab size:      {len(set(sample_text))}")
    print(f"  After 15 BPE merges:     {len(vocab)} tokens")
    print(f"  Top 5 merges:")
    for merged_token, count in merges[:5]:
        print(f"    '{merged_token[0]}' + '{merged_token[1]}' → '{merged_token[0]+merged_token[1]}' (count={count})")

    print("\n" + "=" * 70)
    print("DEMO 1 COMPLETE — Data preparation pipeline demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# DEMO 2 — Training Strategies: full fine-tuning, adapter, progressive
# ─────────────────────────────────────────────────────────────
def demo_training_strategies():
    print("=" * 70)
    print("DEMO 2 — Training Strategies: full fine-tuning, adapter, progressive")
    print("=" * 70)

    # --- 2.1 Full fine-tuning: parameter count & memory estimation ---
    print("\n--- 2.1 Full fine-tuning: parameter count & memory estimation ---")

    def estimate_model_params(layers, hidden, heads, vocab, ff_mult=4):
        attn_params = layers * (4 * hidden * hidden + 4 * hidden)  # Q, K, V, O + biases
        ff_params = layers * (2 * hidden * (ff_mult * hidden) + hidden * (ff_mult * hidden) + hidden)
        embed_params = vocab * hidden
        ln_params = layers * 4 * hidden  # 2 LayerNorms per layer
        total = attn_params + ff_params + embed_params + ln_params
        return {
            "attention": attn_params,
            "feed_forward": ff_params,
            "embeddings": embed_params,
            "layer_norm": ln_params,
            "total": total,
            "total_b": total * 4  # fp32
        }

    configs = [
        ("GPT-2 Small",  12, 768,  12, 50257),
        ("GPT-2 Medium", 24, 1024, 16, 50257),
        ("LLaMA-7B",     32, 4096, 32, 32000),
    ]

    for name, layers, hidden, heads, vocab in configs:
        p = estimate_model_params(layers, hidden, heads, vocab)
        print(f"\n  {name} ({p['total']/1e6:.1f}M params):")
        print(f"    Attention:     {p['attention']/1e6:>10.1f}M  ({p['attention']/p['total']*100:5.1f}%)")
        print(f"    Feed-forward:  {p['feed_forward']/1e6:>10.1f}M  ({p['feed_forward']/p['total']*100:5.1f}%)")
        print(f"    Embeddings:    {p['embeddings']/1e6:>10.1f}M  ({p['embeddings']/p['total']*100:5.1f}%)")
        print(f"    Memory (fp32): {p['total_b']/1e9:.2f} GB")
        print(f"    Memory (fp16): {p['total_b']/2/1e9:.2f} GB")

    # --- 2.2 Adapter modules: parameter-efficient fine-tuning ---
    print("\n--- 2.2 Adapter modules: parameter-efficient fine-tuning ---")

    class MockAdapter:
        def __init__(self, hidden_size, bottleneck_size):
            self.down_proj = [[0.0] * bottleneck_size for _ in range(hidden_size)]
            self.up_proj = [[0.0] * hidden_size for _ in range(bottleneck_size)]
            self.down_params = hidden_size * bottleneck_size
            self.up_params = bottleneck_size * hidden_size
            self.total_params = self.down_params + self.up_params
            self.relu_count = bottleneck_size

        def forward(self, x):
            # Down-project → ReLU → Up-project → residual
            reduced = [0.0] * len(self.down_proj[0])
            for i in range(len(self.down_proj[0])):
                for j in range(len(x)):
                    reduced[i] += x[j] * self.down_proj[j][i]
            activated = [max(0, v) for v in reduced]
            output = [0.0] * len(self.up_proj[0])
            for i in range(len(self.up_proj[0])):
                for j in range(len(activated)):
                    output[i] += activated[j] * self.up_proj[j][i]
            return [x[i] + output[i] for i in range(len(x))]

    hidden = 768
    bottleneck = 64
    adapter = MockAdapter(hidden, bottleneck)
    full_ft_params = estimate_model_params(12, 768, 12, 50257)["total"]
    adapter_total = adapter.total_params * 12  # one adapter per layer
    ratio = adapter_total / full_ft_params * 100

    print(f"  Hidden size:          {hidden}")
    print(f"  Bottleneck size:      {bottleneck}")
    print(f"  Adapter per layer:    {adapter.total_params:,} params")
    print(f"  Total (12 adapters):  {adapter_total:,} params")
    print(f"  Full model:           {full_ft_params:,} params")
    print(f"  Trainable fraction:   {ratio:.3f}%")
    print(f"  Memory savings:       {(1 - ratio/100)*100:.2f}%")

    # --- 2.3 Progressive training: curriculum learning simulation ---
    print("\n--- 2.3 Progressive training: curriculum learning simulation ---")

    def simulate_curriculum_stages(n_samples=1000):
        random.seed(42)
        difficulties = []
        for _ in range(n_samples):
            word_count = random.randint(5, 500)
            vocab_complexity = random.random()
            noise_level = random.random()
            difficulty = 0.3 * min(word_count / 200, 1.0) + 0.4 * vocab_complexity + 0.3 * noise_level
            difficulties.append(difficulty)
        difficulties.sort()
        return difficulties

    all_diffs = simulate_curriculum_stages(1000)
    stages = [
        ("Easy (epoch 1-2)",   0, 200),
        ("Medium (epoch 3-4)", 200, 600),
        ("Hard (epoch 5-6)",   600, 900),
        ("Full (epoch 7-8)",   0, 1000),
    ]

    print("  Stage            | Samples | Avg Difficulty | Loss (simulated)")
    print("  " + "-" * 60)
    for name, start, end in stages:
        stage_diffs = all_diffs[start:end]
        avg_diff = statistics.mean(stage_diffs) if stage_diffs else 0
        # Simulated loss: harder data → higher loss
        simulated_loss = 2.0 * math.exp(-0.003 * (end - start)) + 0.5 * avg_diff
        print(f"  {name:<18}| {len(stage_diffs):>7} | {avg_diff:.3f}            | {simulated_loss:.4f}")

    # --- 2.4 Memory optimization: gradient checkpointing & mixed precision ---
    print("\n--- 2.4 Memory optimization: gradient checkpointing & mixed precision ---")

    def memory_analysis(model_params_m, seq_len, batch_size, layers):
        # Standard fp32
        param_mem = model_params_m * 1e6 * 4
        grad_mem = model_params_m * 1e6 * 4
        optimizer_mem = model_params_m * 1e6 * 8  # Adam: 2 states
        activation_mem = layers * batch_size * seq_len * 768 * 4 * 3  # simplified
        total_fp32 = (param_mem + grad_mem + optimizer_mem + activation_mem) / 1e9

        # fp16 + mixed precision
        param_mem_16 = model_params_m * 1e6 * 2
        grad_mem_16 = model_params_m * 1e6 * 2
        optimizer_mem_16 = model_params_m * 1e6 * 8  # master weights stay fp32
        activation_mem_16 = activation_mem / 2
        total_fp16 = (param_mem_16 + grad_mem_16 + optimizer_mem_16 + activation_mem_16) / 1e9

        # With gradient checkpointing
        activation_checkpointed = activation_mem_16 / layers * 2  # ~2x reduction
        total_ckpt = (param_mem_16 + grad_mem_16 + optimizer_mem_16 + activation_checkpointed) / 1e9

        return {
            "fp32": round(total_fp32, 3),
            "fp16": round(total_fp16, 3),
            "fp16_ckpt": round(total_ckpt, 3),
        }

    print(f"  Model: 7B params, seq_len=2048, batch=8, layers=32")
    mem = memory_analysis(7000, 2048, 8, 32)
    print(f"  fp32 (standard):     {mem['fp32']:>8.3f} GB")
    print(f"  fp16 (mixed prec.):  {mem['fp16']:>8.3f} GB")
    print(f"  fp16 + checkpoint:   {mem['fp16_ckpt']:>8.3f} GB")
    print(f"  Savings fp16:        {(1 - mem['fp16']/mem['fp32'])*100:.1f}%")
    print(f"  Savings ckpt:        {(1 - mem['fp16_ckpt']/mem['fp32'])*100:.1f}%")

    print("\n" + "=" * 70)
    print("DEMO 2 COMPLETE — Training strategies demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# DEMO 3 — Distributed Concepts: data parallelism, model parallelism, pipeline
# ─────────────────────────────────────────────────────────────
def demo_distributed_concepts():
    print("=" * 70)
    print("DEMO 3 — Distributed Concepts: data parallelism, model parallelism, pipeline")
    print("=" * 70)

    # --- 3.1 Data parallelism: gradient aggregation simulation ---
    print("\n--- 3.1 Data parallelism: gradient aggregation across GPUs ---")

    def simulate_data_parallel(n_gpus=4, batch_size=8, lr=0.001):
        random.seed(42)
        # Simulate gradient vectors from different GPUs
        param_dim = 16
        master_weights = [random.gauss(0, 0.1) for _ in range(param_dim)]
        all_gpu_grads = []
        for gpu in range(n_gpus):
            grads = [random.gauss(0, 0.01) + 0.001 * math.sin(gpu) for _ in range(param_dim)]
            all_gpu_grads.append(grads)

        # All-reduce: average gradients
        avg_grads = []
        for i in range(param_dim):
            avg_grads.append(sum(all_gpu_grads[g][i] for g in range(n_gpus)) / n_gpus)

        # Weight update: w = w - lr * grad
        updated_weights = [master_weights[i] - lr * avg_grads[i] for i in range(param_dim)]

        # Compute norm before and after
        grad_norm_before = math.sqrt(sum(g ** 2 for g in all_gpu_grads[0]))
        grad_norm_after = math.sqrt(sum(g ** 2 for g in avg_grads))
        weight_change = math.sqrt(sum((updated_weights[i] - master_weights[i]) ** 2 for i in range(param_dim)))

        print(f"  GPUs:              {n_gpus}")
        print(f"  Batch per GPU:     {batch_size}")
        print(f"  Effective batch:   {n_gpus * batch_size}")
        print(f"  Grad norm (GPU 0): {grad_norm_before:.6f}")
        print(f"  Grad norm (avg):   {grad_norm_after:.6f}")
        print(f"  Noise reduction:   {(1 - grad_norm_after/grad_norm_before)*100:.1f}%")
        print(f"  Weight change norm:{weight_change:.8f}")
        return updated_weights

    final_w = simulate_data_parallel()

    # --- 3.2 Model parallelism: tensor-parallel split simulation ---
    print("\n--- 3.2 Model parallelism: tensor-parallel split simulation ---")

    def simulate_tensor_parallel(matrix_rows=64, matrix_cols=64, n_gpus=4):
        random.seed(42)
        matrix = [[random.gauss(0, 1) for _ in range(matrix_cols)] for _ in range(matrix_rows)]
        vector = [random.gauss(0, 1) for _ in range(matrix_cols)]

        # Naive: full matmul on one GPU
        start = time.time()
        result_full = [sum(matrix[i][j] * vector[j] for j in range(matrix_cols)) for i in range(matrix_rows)]
        naive_time = time.time() - start

        # Tensor parallel: split rows across GPUs
        rows_per_gpu = matrix_rows // n_gpus
        start = time.time()
        partial_results = []
        for g in range(n_gpus):
            start_row = g * rows_per_gpu
            end_row = start_row + rows_per_gpu
            partial = [sum(matrix[i][j] * vector[j] for j in range(matrix_cols)) for i in range(start_row, end_row)]
            partial_results.extend(partial)
        tp_time = time.time() - start

        # Verify correctness
        matches = all(abs(result_full[i] - partial_results[i]) < 1e-10 for i in range(matrix_rows))
        mem_full = matrix_rows * matrix_cols * 8 / 1e6
        mem_per_gpu = rows_per_gpu * matrix_cols * 8 / 1e6

        print(f"  Matrix size:       {matrix_rows} x {matrix_cols}")
        print(f"  GPUs:              {n_gpus}")
        print(f"  Rows per GPU:      {rows_per_gpu}")
        print(f"  Full memory:       {mem_full:.2f} MB")
        print(f"  Memory per GPU:    {mem_per_gpu:.2f} MB")
        print(f"  Memory reduction:  {(1 - mem_per_gpu/mem_full)*100:.1f}%")
        print(f"  Results match:     {matches}")
        print(f"  Output sample:     [{result_full[0]:.4f}, {result_full[1]:.4f}, ...]")
        return result_full

    result = simulate_tensor_parallel()

    # --- 3.3 Pipeline parallelism: micro-batch scheduling ---
    print("\n--- 3.3 Pipeline parallelism: micro-batch scheduling ---")

    def simulate_pipeline_parallel(n_stages=4, n_microbatches=8, stage_times=None):
        if stage_times is None:
            stage_times = [1.0, 1.2, 0.8, 1.1]  # per-stage compute time

        # Naive sequential pipeline
        naive_time = n_microbatches * sum(stage_times)

        # Pipeline parallel: 1F1B schedule
        warmup_microbatches = min(n_microbatches, n_stages)
        # Steady state: after warmup, each step processes one microbatch
        total_steps = warmup_microbatches + (n_microbatches - warmup_microbatches)
        max_stage_time = max(stage_times)
        pipeline_time = (n_stages + n_microbatches - 1) * max_stage_time

        # Bubble ratio
        ideal_time = n_microbatches * max_stage_time
        bubble_ratio = (pipeline_time - ideal_time) / pipeline_time * 100

        print(f"  Stages:            {n_stages}")
        print(f"  Micro-batches:     {n_microbatches}")
        print(f"  Stage times:       {stage_times}")
        print(f"  Naive sequential:  {naive_time:.1f} time units")
        print(f"  Pipeline (1F1B):   {pipeline_time:.1f} time units")
        print(f"  Speedup:           {naive_time/pipeline_time:.2f}x")
        print(f"  Bubble ratio:      {bubble_ratio:.1f}%")
        print(f"  Throughput:        {n_microbatches/pipeline_time:.2f} batches/time-unit")

        # Schedule visualization (compact)
        print(f"\n  1F1B Schedule (simplified):")
        max_bars = 40
        for stage in range(n_stages):
            line = f"    Stage {stage}: "
            chars = []
            for mb in range(n_microbatches):
                start_step = mb + stage
                if start_step < total_steps:
                    chars.append("█")
                else:
                    chars.append("·")
            line += "".join(chars[:max_bars])
            print(line)

    simulate_pipeline_parallel()

    # --- 3.4 Communication patterns: all-reduce cost analysis ---
    print("\n--- 3.4 Communication patterns: all-reduce cost analysis ---")

    def analyze_allreduce(n_gpus, param_bytes, bandwidth_gbps=100):
        """Ring all-reduce cost analysis."""
        # Ring all-reduce: 2*(n-1)/n * data_size / bandwidth
        data_gb = param_bytes / 1e9
        ring_cost = 2 * (n_gpus - 1) / n_gpus * data_gb / bandwidth_gbps

        # Tree all-reduce: log2(n) * data_size / bandwidth
        import math as _m
        tree_cost = _m.ceil(_m.log2(n_gpus)) * data_gb / bandwidth_gbps

        # Naive gather + broadcast: (n-1)/n * data_size / bandwidth * 2
        naive_cost = 2 * (n_gpus - 1) / n_gpus * data_gb / bandwidth_gbps

        # Computation: fp32 multiply-add per param
        flops_per_param = 2
        compute_time = param_bytes / 4 * flops_per_param / 1e12  # assuming 1 TFLOP/s

        return {
            "ring": round(ring_cost * 1000, 2),  # ms
            "tree": round(tree_cost * 1000, 2),
            "naive": round(naive_cost * 1000, 2),
            "compute_ms": round(compute_time * 1000, 2),
        }

    print(f"  Bandwidth: 100 Gbps, Model: 7B params (fp32)")
    param_bytes_7b = 7e9 * 4
    for ng in [2, 4, 8, 16, 32]:
        costs = analyze_allreduce(ng, param_bytes_7b)
        comm_compute_ratio = costs["ring"] / costs["compute_ms"] if costs["compute_ms"] > 0 else 0
        print(f"  {ng:>2} GPUs: ring={costs['ring']:>8.2f}ms  "
              f"tree={costs['tree']:>8.2f}ms  "
              f"comm/compute={comm_compute_ratio:.2f}x")

    print("\n" + "=" * 70)
    print("DEMO 3 COMPLETE — Distributed concepts demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# DEMO 4 — Hyperparameter Optimization: learning rate, batch size, warmup
# ─────────────────────────────────────────────────────────────
def demo_hyperparameter_optimization():
    print("=" * 70)
    print("DEMO 4 — Hyperparameter Optimization: learning rate, batch size, warmup")
    print("=" * 70)

    # --- 4.1 Learning rate schedules ---
    print("\n--- 4.1 Learning rate schedules: cosine, linear, warmup ---")

    def cosine_schedule(step, total_steps, max_lr, min_lr, warmup_steps):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    def linear_schedule(step, total_steps, max_lr, min_lr, warmup_steps):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr - (max_lr - min_lr) * progress

    total_steps = 1000
    max_lr = 3e-4
    min_lr = 1e-5
    warmup_steps = 100

    checkpoints = [0, 50, 100, 200, 500, 750, 1000]
    print(f"  Total steps: {total_steps}, Max LR: {max_lr}, Warmup: {warmup_steps} steps")
    print(f"\n  {'Step':>6} | {'Cosine LR':>12} | {'Linear LR':>12} | {'Ratio':>8}")
    print(f"  {'-'*48}")
    for step in checkpoints:
        cos_lr = cosine_schedule(step, total_steps, max_lr, min_lr, warmup_steps)
        lin_lr = linear_schedule(step, total_steps, max_lr, min_lr, warmup_steps)
        ratio = cos_lr / lin_lr if lin_lr > 0 else float('inf')
        print(f"  {step:>6} | {cos_lr:>12.6f} | {lin_lr:>12.6f} | {ratio:>8.3f}")

    # --- 4.2 Batch size scaling & linear scaling rule ---
    print("\n--- 4.2 Batch size scaling: linear scaling rule ---")

    def linear_scaling_rule(base_lr, base_batch, new_batch, smooth=0.01):
        """
        lr_new = lr_base * (new_batch / base_batch) * smooth_factor
        smooth_factor accounts for diminishing returns at large batches
        """
        raw_scale = new_batch / base_batch
        smooth_scale = raw_scale ** smooth * math.log(1 + raw_scale) / raw_scale
        return base_lr * raw_scale * smooth_scale

    base_lr = 3e-4
    base_batch = 256
    batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    print(f"  Base: lr={base_lr}, batch={base_batch}")
    print(f"\n  {'Batch':>6} | {'Scale':>8} | {'Raw LR':>12} | {'Smoothed LR':>12} | {'Efficiency':>10}")
    print(f"  {'-'*58}")
    for bs in batch_sizes:
        raw_lr = base_lr * (bs / base_batch)
        smooth_lr = linear_scaling_rule(base_lr, base_batch, bs)
        # Efficiency: diminishing returns model
        efficiency = 1 - math.exp(-0.001 * bs)  # saturates at 1.0
        print(f"  {bs:>6} | {bs/base_batch:>8.2f} | {raw_lr:>12.6f} | {smooth_lr:>12.6f} | {efficiency:>10.3f}")

    # --- 4.3 Warmup strategies ---
    print("\n--- 4.3 Warmup strategies: linear, gradual, no warmup ---")

    def simulate_training_with_warmup(warmup_strategy, total_steps=500, base_lr=3e-4):
        random.seed(42)
        losses = []
        lr_history = []
        for step in range(total_steps):
            # LR schedule
            if warmup_strategy == "none":
                lr = base_lr
            elif warmup_strategy == "linear_10pct":
                warmup = int(0.1 * total_steps)
                lr = base_lr * min(1.0, step / warmup) if step < warmup else base_lr
            elif warmup_strategy == "gradual_20pct":
                warmup = int(0.2 * total_steps)
                lr = base_lr * (step / warmup) ** 0.5 if step < warmup else base_lr
            elif warmup_strategy == "cosine_restart":
                warmup = 50
                if step < warmup:
                    lr = base_lr * step / warmup
                else:
                    cycle_len = 200
                    cycle_step = (step - warmup) % cycle_len
                    lr = base_lr * (0.5 + 0.5 * math.cos(math.pi * cycle_step / cycle_len))
            else:
                lr = base_lr

            lr_history.append(lr)
            # Simulated loss: decreases with step, noisy early
            noise = random.gauss(0, 0.1 * max(0.1, 1 - step / total_steps))
            base_loss = 3.0 * math.exp(-0.005 * step) + 0.5
            # Large LR → instability
            instability = max(0, (lr - base_lr * 1.5) * 0.01) if lr > base_lr else 0
            loss = base_loss + noise + instability
            losses.append(loss)

        return losses, lr_history

    strategies = ["none", "linear_10pct", "gradual_20pct", "cosine_restart"]
    print(f"  {'Strategy':<20} | {'Final Loss':>10} | {'Min Loss':>10} | {'Stability':>10}")
    print(f"  {'-'*60}")
    for strat in strategies:
        losses, lrs = simulate_training_with_warmup(strat)
        final_loss = statistics.mean(losses[-50:])
        min_loss = min(losses)
        stability = 1 - statistics.stdev(losses[-100:]) / statistics.mean(losses[-100:])
        print(f"  {strat:<20} | {final_loss:>10.4f} | {min_loss:>10.4f} | {stability:>10.4f}")

    # --- 4.4 Bayesian optimization: hyperparameter search simulation ---
    print("\n--- 4.4 Bayesian optimization: hyperparameter search ---")

    def objective_function(lr, batch_size, dropout):
        """Simulated objective: lower is better (validation loss)."""
        # Known optimum around lr=2e-4, batch=512, dropout=0.1
        lr_loss = 2.0 * (math.log(lr / 2e-4)) ** 2
        batch_loss = 0.5 * ((batch_size - 512) / 512) ** 2
        dropout_loss = 1.0 * (dropout - 0.1) ** 2
        interaction = -0.1 * lr * batch_size * dropout  # some interaction
        return lr_loss + batch_loss + dropout_loss + interaction + 1.5

    def random_search(n_trials=30):
        random.seed(42)
        best_loss = float('inf')
        results = []
        for trial in range(n_trials):
            lr = 10 ** random.uniform(-5, -2)
            batch_size = random.choice([64, 128, 256, 512, 1024, 2048])
            dropout = random.uniform(0, 0.5)
            loss = objective_function(lr, batch_size, dropout)
            results.append((trial, lr, batch_size, dropout, loss))
            if loss < best_loss:
                best_loss = loss
        return results

    def bayesian_search(n_trials=30):
        random.seed(42)
        # Initial random samples
        results = []
        for trial in range(min(5, n_trials)):
            lr = 10 ** random.uniform(-5, -2)
            batch_size = random.choice([64, 128, 256, 512, 1024, 2048])
            dropout = random.uniform(0, 0.5)
            loss = objective_function(lr, batch_size, dropout)
            results.append((trial, lr, batch_size, dropout, loss))

        # "Bayesian" refinement: sample near best
        for trial in range(5, n_trials):
            # Find current best
            best = min(results, key=lambda x: x[4])
            # Perturb around best
            lr = best[1] * 10 ** random.gauss(0, 0.5)
            lr = max(1e-5, min(1e-2, lr))
            batch_size = random.choice([64, 128, 256, 512, 1024, 2048])
            dropout = max(0, min(0.5, best[3] + random.gauss(0, 0.05)))
            loss = objective_function(lr, batch_size, dropout)
            results.append((trial, lr, batch_size, dropout, loss))
        return results

    rs_results = random_search(30)
    bs_results = bayesian_search(30)

    print(f"\n  Random Search (30 trials):")
    rs_best = min(rs_results, key=lambda x: x[4])
    rs_convergence = [min(r[4] for r in rs_results[:i+1]) for i in range(len(rs_results))]
    print(f"    Best loss:    {rs_best[4]:.4f}")
    print(f"    Best params:  lr={rs_best[1]:.6f}, batch={rs_best[2]}, dropout={rs_best[3]:.3f}")

    print(f"\n  Bayesian Search (30 trials):")
    bs_best = min(bs_results, key=lambda x: x[4])
    bs_convergence = [min(r[4] for r in bs_results[:i+1]) for i in range(len(bs_results))]
    print(f"    Best loss:    {bs_best[4]:.4f}")
    print(f"    Best params:  lr={bs_best[1]:.6f}, batch={bs_best[2]}, dropout={bs_best[3]:.3f}")

    # Convergence comparison
    print(f"\n  Convergence (best-so-far at trial checkpoints):")
    for t in [0, 9, 19, 29]:
        if t < len(rs_convergence) and t < len(bs_convergence):
            print(f"    Trial {t+1:>2}: RS={rs_convergence[t]:.4f}, BS={bs_convergence[t]:.4f}, "
                  f"BS advantage={((rs_convergence[t] - bs_convergence[t])/rs_convergence[t])*100:.1f}%")

    # True optimum
    true_opt = objective_function(2e-4, 512, 0.1)
    print(f"\n  True optimum: {true_opt:.4f} (lr=2e-4, batch=512, dropout=0.1)")
    print(f"  RS gap:       {rs_best[4] - true_opt:.4f}")
    print(f"  BS gap:       {bs_best[4] - true_opt:.4f}")

    print("\n" + "=" * 70)
    print("DEMO 4 COMPLETE — Hyperparameter optimization demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_data_preparation()
    demo_training_strategies()
    demo_distributed_concepts()
    demo_hyperparameter_optimization()
