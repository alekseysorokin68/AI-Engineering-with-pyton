"""114 — LLM Serving: батчинг, стриминг, балансировка нагрузки

Темы:
  1. Request Batching (dynamic batching, padding, max batch size)
  2. Streaming Responses (token-by-token, SSE format, chunked output)
  3. Load Balancing (round-robin, least-connections, weighted)
  4. Rate Limiting (token bucket, sliding window, retry with backoff)

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
# Demo 1 — Request Batching
# ──────────────────────────────────────────────────────────────────────────────

def demo_request_batching():
    """Dynamic batching of LLM inference requests with padding."""
    print("=" * 70)
    print("DEMO 1 — Request Batching: dynamic batching, padding, max batch size")
    print("=" * 70)

    # --- Sub-example 1: Token sequences and padding ---
    print("\n[1.1] Token sequences and zero-padding to uniform length")
    requests = [
        {"id": "r1", "tokens": [101, 2023, 3045, 102],          "prompt": "Hello world"},
        {"id": "r2", "tokens": [101, 2023, 3045, 102, 4056, 789, 102], "prompt": "What is AI?"},
        {"id": "r3", "tokens": [101, 5023, 102],                 "prompt": "Hi"},
        {"id": "r4", "tokens": [101, 6789, 3045, 2345, 102],     "prompt": "Tell me more"},
    ]
    max_len = max(len(r["tokens"]) for r in requests)
    print(f"  Max sequence length in batch: {max_len}")

    padded = []
    for r in requests:
        pad_len = max_len - len(r["tokens"])
        padded_tokens = r["tokens"] + [0] * pad_len
        attention_mask = [1] * len(r["tokens"]) + [0] * pad_len
        padded.append({"id": r["id"], "tokens": padded_tokens,
                       "attention_mask": attention_mask, "orig_len": len(r["tokens"])})
        print(f"  {r['id']}: {r['tokens']} + [{pad_len} pads]  mask={attention_mask}")

    print(f"  Padded batch shape: ({len(padded)}, {max_len})")
    wasted = sum(p["attention_mask"].count(0) for p in padded)
    total = len(padded) * max_len
    print(f"  Padding waste: {wasted}/{total} tokens ({100*wasted/total:.1f}%)")

    # --- Sub-example 2: Dynamic batching by similarity ---
    print("\n[1.2] Dynamic batching — group requests by similar length")
    all_requests = [
        {"id": f"r{i}", "tokens": [101] + [random.randint(100, 30000)
              for _ in range(random.randint(3, 20))] + [102]}
        for i in range(12)
    ]
    max_batch_size = 4
    max_len_tolerance = 5

    # Sort by length, then group into batches
    all_requests.sort(key=lambda r: len(r["tokens"]))
    batches = []
    current_batch = [all_requests[0]]
    for req in all_requests[1:]:
        if (len(current_batch) < max_batch_size and
                len(req["tokens"]) - len(current_batch[0]["tokens"]) <= max_len_tolerance):
            current_batch.append(req)
        else:
            batches.append(current_batch)
            current_batch = [req]
    batches.append(current_batch)

    for i, batch in enumerate(batches):
        lengths = [len(r["tokens"]) for r in batch]
        batch_max = max(lengths)
        waste = sum(batch_max - l for l in lengths)
        ids = [r["id"] for r in batch]
        print(f"  Batch {i+1}: ids={ids} lengths={lengths} "
              f"max={batch_max} waste={waste}")

    # --- Sub-example 3: Throughput calculation ---
    print("\n[1.3] Throughput comparison: individual vs batched inference")
    single_latency_ms = 150  # per-request latency
    batch_size = 8
    batch_latency_ms = 280   # batch overhead + compute

    single_throughput = 1000 / single_latency_ms  # tokens/sec equivalent
    batch_throughput = (batch_size * 1000) / batch_latency_ms
    speedup = batch_throughput / single_throughput
    print(f"  Single request: {single_latency_ms}ms → {single_throughput:.1f} req/s")
    print(f"  Batch of {batch_size}: {batch_latency_ms}ms → {batch_throughput:.1f} req/s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  GPU utilization improvement: {speedup*100 - 100:.0f}% more throughput")

    # --- Sub-example 4: Prefill vs decode batching ---
    print("\n[1.4] Prefill vs decode phase batching tradeoff")
    prefill_tokens = 512
    decode_tokens = 128
    flops_per_token = 1e12  # hypothetical

    prefill_flops = prefill_tokens * flops_per_token
    decode_flops = decode_tokens * flops_per_token
    ratio = prefill_flops / decode_flops
    print(f"  Prefill phase: {prefill_tokens} tokens × {flops_per_token:.0e} FLOPS = {prefill_flops:.0e} FLOPS")
    print(f"  Decode phase:  {decode_tokens} tokens × {flops_per_token:.0e} FLOPS = {decode_flops:.0e} FLOPS")
    print(f"  Prefill/Decode ratio: {ratio:.1f}x")
    print(f"  → Prefill is compute-bound, decode is memory-bound")
    print(f"  → Dynamic batching helps more for decode-heavy workloads")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 2 — Streaming Responses
# ──────────────────────────────────────────────────────────────────────────────

def demo_streaming_responses():
    """Token-by-token streaming with SSE format."""
    print("\n" + "=" * 70)
    print("DEMO 2 — Streaming Responses: token-by-token, SSE format")
    print("=" * 70)

    # --- Sub-example 1: Token-by-token generation ---
    print("\n[2.1] Token-by-token generation simulation")
    vocabulary = ["The", " answer", " is", ":", " 42", ".", " Let", " me",
                  " explain", " this", " in", " detail", ".", " First", ",",
                  " we", " need", " to", " understand", " the", " context", "."]
    random.seed(42)
    generated = []
    ttfb_ms = 35  # time to first token
    tpot_ms = 12  # time per output token

    print(f"  TTFB (time to first token): {ttfb_ms}ms")
    print(f"  TPOT (time per output token): {tpot_ms}ms")
    print(f"  Generated tokens:")
    for i, token in enumerate(vocabulary[:10]):
        generated.append(token)
        cumulative_ms = ttfb_ms + i * tpot_ms
        print(f"    [{cumulative_ms:4d}ms] +'{token}' → '{''.join(generated)}'")

    total_ms = ttfb_ms + len(vocabulary[:10]) * tpot_ms
    print(f"  Total time for 10 tokens: {total_ms}ms")

    # --- Sub-example 2: SSE (Server-Sent Events) format ---
    print("\n[2.2] SSE (Server-Sent Events) format")
    tokens = ["Hello", ",", " world", "!"]
    sse_lines = []
    for i, token in enumerate(tokens):
        event = {
            "id": i,
            "event": "token",
            "data": json.dumps({
                "token": token,
                "index": i,
                "finish_reason": "stop" if i == len(tokens) - 1 else None
            })
        }
        line = f"event: {event['event']}\ndata: {event['data']}\n"
        sse_lines.append(line)
        print(f"  SSE event {i}:")
        for l in line.strip().split("\n"):
            print(f"    {l}")

    # --- Sub-example 3: Chunked transfer encoding ---
    print("\n[2.3] Chunked HTTP transfer simulation")
    full_response = "The capital of France is Paris."
    chunk_size = 12
    chunks = [full_response[i:i+chunk_size] for i in range(0, len(full_response), chunk_size)]
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        print(f"  Chunk {i}: [{chunk_len:2d} bytes] \"{chunk}\"")
    print(f"  Total chunks: {len(chunks)}, total bytes: {len(full_response)}")

    # --- Sub-example 4: Streaming latency analysis ---
    print("\n[2.4] Streaming vs non-streaming latency perception")
    response_tokens = 200
    ttfb = 300  # ms
    tpot = 8    # ms
    total_generation = ttfb + response_tokens * tpot
    first_token_perceived = ttfb
    non_stream_total = total_generation

    print(f"  Response length: {response_tokens} tokens")
    print(f"  TTFB: {ttfb}ms")
    print(f"  TPOT: {tpot}ms")
    print(f"  Non-streaming total wait: {non_stream_total}ms")
    print(f"  Streaming perceived first content: {first_token_perceived}ms")
    print(f"  Perceived latency reduction: {(1 - first_token_perceived/non_stream_total)*100:.0f}%")
    print(f"  User sees first token after {first_token_perceived}ms instead of {non_stream_total}ms")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 3 — Load Balancing
# ──────────────────────────────────────────────────────────────────────────────

def demo_load_balancing():
    """Load balancing strategies for LLM inference endpoints."""
    print("\n" + "=" * 70)
    print("DEMO 3 — Load Balancing: round-robin, least-connections, weighted")
    print("=" * 70)

    # --- Sub-example 1: Round-robin ---
    print("\n[3.1] Round-Robin Load Balancing")
    servers = ["GPU-A100-1", "GPU-A100-2", "GPU-H100-1"]
    requests = [f"req_{i}" for i in range(9)]
    rr_index = 0
    assignments = []
    for req in requests:
        server = servers[rr_index % len(servers)]
        assignments.append((req, server))
        rr_index += 1
    for req, server in assignments:
        print(f"  {req:8s} → {server}")
    counts = collections.Counter(s for _, s in assignments)
    print(f"  Distribution: {dict(counts)}")

    # --- Sub-example 2: Weighted round-robin ---
    print("\n[3.2] Weighted Round-Robin (GPU capacity weighted)")
    weighted_servers = [
        {"name": "A100-40GB", "weight": 1, "current_load": 0.0},
        {"name": "A100-80GB", "weight": 2, "current_load": 0.0},
        {"name": "H100-80GB", "weight": 3, "current_load": 0.0},
    ]
    for i in range(12):
        # Pick server with lowest load/weight ratio
        best = min(weighted_servers, key=lambda s: s["current_load"] / s["weight"])
        best["current_load"] += 1
        print(f"  req_{i:2d} → {best['name']:12s} (load={best['current_load']}, "
              f"load/weight={best['current_load']/best['weight']:.2f})")

    # --- Sub-example 3: Least connections ---
    print("\n[3.3] Least Connections strategy")
    conn_servers = {"A100-1": 3, "A100-2": 1, "H100-1": 5}
    print(f"  Initial connections: {conn_servers}")
    for i in range(6):
        chosen = min(conn_servers, key=conn_servers.get)
        conn_servers[chosen] += 1
        print(f"  req_{i}: chose {chosen} (now {conn_servers[chosen]} conns)")

    # --- Sub-example 4: Consistent hashing ---
    print("\n[3.4] Consistent Hashing for sticky sessions")
    num_virtual_nodes = 60
    hash_ring = []
    server_names = ["S1", "S2", "S3"]
    for sn in server_names:
        for vn in range(num_virtual_nodes):
            h = int(hashlib.md5(f"{sn}#{vn}".encode()).hexdigest()[:8], 16)
            hash_ring.append((h, sn))
    hash_ring.sort()

    def lookup(key):
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        for ring_h, sn in hash_ring:
            if ring_h >= h:
                return sn
        return hash_ring[0][1]

    session_keys = ["user_1001", "user_1002", "user_1003", "user_1004", "user_1005"]
    for key in session_keys:
        server = lookup(key)
        print(f"  {key:14s} hash→ server {server}")

    # Show what happens when a server is removed
    print("\n  After removing S2 (failover):")
    hash_ring_no_s2 = [(h, sn) for h, sn in hash_ring if sn != "S2"]
    def lookup_no_s2(key):
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        for ring_h, sn in hash_ring_no_s2:
            if ring_h >= h:
                return sn
        return hash_ring_no_s2[0][1]

    for key in session_keys:
        old_server = lookup(key)
        new_server = lookup_no_s2(key)
        moved = "MOVED" if old_server != new_server else "same"
        print(f"  {key:14s}: {old_server}→{new_server}  [{moved}]")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 4 — Rate Limiting
# ──────────────────────────────────────────────────────────────────────────────

def demo_rate_limiting():
    """Token bucket, sliding window, and retry with backoff."""
    print("\n" + "=" * 70)
    print("DEMO 4 — Rate Limiting: token bucket, sliding window, backoff")
    print("=" * 70)

    # --- Sub-example 1: Token bucket algorithm ---
    print("\n[4.1] Token Bucket Algorithm")
    bucket_capacity = 10
    refill_rate = 2  # tokens per second
    tokens = bucket_capacity

    events = [
        (0.0, "request", 3),   # consume 3 tokens
        (0.5, "request", 2),   # consume 2
        (1.0, "refill"),       # refill: +2 tokens (1s × 2/s)
        (1.5, "request", 5),   # consume 5
        (2.0, "refill"),       # refill: +2
        (2.5, "request", 8),   # try to consume 8 — should fail
    ]
    last_refill = 0.0
    for t, action, *args in events:
        if action == "refill":
            elapsed = t - last_refill
            tokens = min(bucket_capacity, tokens + int(elapsed * refill_rate))
            last_refill = t
            print(f"  t={t:.1f}s REFILL → tokens={tokens}/{bucket_capacity}")
        else:
            amount = args[0]
            if tokens >= amount:
                tokens -= amount
                print(f"  t={t:.1f}s ALLOW  ({amount} tokens) → tokens={tokens}/{bucket_capacity}")
            else:
                print(f"  t={t:.1f}s DENY   (need {amount}, have {tokens}) → tokens={tokens}/{bucket_capacity}")

    # --- Sub-example 2: Sliding window log ---
    print("\n[4.2] Sliding Window Log Rate Limiter")
    window_sec = 5
    max_requests = 3
    request_log = []

    test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 4.5, 5.5]
    for t in test_times:
        request_log = [rt for rt in request_log if t - rt <= window_sec]
        if len(request_log) < max_requests:
            request_log.append(t)
            print(f"  t={t:.1f}s ALLOW  (window has {len(request_log)}/{max_requests})")
        else:
            oldest = request_log[0]
            retry_after = oldest + window_sec - t
            print(f"  t={t:.1f}s DENY   (window full, retry after {retry_after:.1f}s)")

    # --- Sub-example 3: Exponential backoff ---
    print("\n[4.3] Exponential Backoff with Jitter")
    max_retries = 5
    base_delay = 1.0
    for attempt in range(max_retries):
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, delay * 0.5)
        total_delay = delay + jitter
        print(f"  Attempt {attempt+1}: delay={delay:.2f}s + jitter={jitter:.2f}s "
              f"= total {total_delay:.2f}s")

    # --- Sub-example 4: Multi-tier rate limiting ---
    print("\n[4.4] Multi-tier Rate Limiting (per-user + global)")
    tiers = {
        "free":    {"rpm": 10,  "tpm": 10000,  "concurrent": 2},
        "pro":     {"rpm": 100, "tpm": 100000, "concurrent": 10},
        "enterprise": {"rpm": 1000, "tpm": 1000000, "concurrent": 50},
    }
    users = [
        ("alice", "free", 15),
        ("bob", "pro", 80),
        ("carol", "enterprise", 500),
    ]
    global_rpm_limit = 200
    global_usage = 0

    for name, tier, requested_rpm in users:
        limits = tiers[tier]
        user_allowed = min(requested_rpm, limits["rpm"])
        if global_usage + user_allowed > global_rpm_limit:
            user_allowed = max(0, global_rpm_limit - global_usage)
        global_usage += user_allowed
        print(f"  {name:8s} ({tier:12s}): requested={requested_rpm:4d}rpm "
              f"limit={limits['rpm']:5d}rpm allowed={user_allowed:4d}rpm "
              f"(global remaining={global_rpm_limit - global_usage})")


if __name__ == "__main__":
    demo_request_batching()
    demo_streaming_responses()
    demo_load_balancing()
    demo_rate_limiting()
