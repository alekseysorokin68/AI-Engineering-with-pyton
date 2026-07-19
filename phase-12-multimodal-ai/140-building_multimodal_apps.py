"""140 — Building Multimodal Apps: end-to-end pipelines, API design, deployment

Темы:
  1. Multimodal Pipeline Design (input processing, model orchestration, output)
  2. API Design for Multimodal (file upload, streaming, error handling)
  3. Production Considerations (caching, batching, cost management)
  4. Real-World Applications (document AI, visual search, accessibility)

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
# Demo 1 — Multimodal Pipeline Design
# ---------------------------------------------------------------------------

def demo_pipeline_design():
    """
    Shows input processing, model orchestration, and output formatting.
    """
    print("=" * 70)
    print("DEMO 1 — Multimodal Pipeline Design")
    print("=" * 70)

    # --- 1a. Input Processing Pipeline ---
    print("\n--- 1a. Input Processing Pipeline ---")
    inputs = {
        "image": {"format": "JPEG", "size": (1920, 1080), "channels": 3,
                  "file_size_mb": 2.4},
        "text": {"format": "UTF-8", "length": 256, "language": "en"},
        "audio": {"format": "WAV", "duration_s": 12.5, "sample_rate": 16000},
    }
    processing_steps = {
        "image": [
            ("resize", "scale to 224x224", 0.5),
            ("normalize", "mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]", 0.01),
            ("augment", "random horizontal flip (train only)", 0.02),
        ],
        "text": [
            ("tokenize", "BPE tokenizer, vocab=50257", 0.05),
            ("truncate", "max_length=77", 0.01),
            ("pad", "right-pad to batch max length", 0.005),
        ],
        "audio": [
            ("resample", "to 16kHz mono", 0.1),
            ("mel_spectrogram", "n_mels=80, hop_length=160", 0.3),
            ("normalize", "per-channel mean/std", 0.02),
        ],
    }

    for modality, info in inputs.items():
        print(f"\n  {modality.upper()} INPUT: {info}")
        print(f"  Processing steps:")
        for step, desc, latency in processing_steps[modality]:
            print(f"    {step:>20s}: {desc} ({latency*1000:.0f}ms)")

    # --- 1b. Model Orchestration ---
    print("\n--- 1b. Model Orchestration (Ensemble/Chain) ---")
    pipeline_stages = [
        {"name": "Vision Encoder", "model": "ViT-L/14", "params": "304M",
         "input": "image", "output": "visual_features [B,257,1024]"},
        {"name": "Text Encoder", "model": "RoBERTa-large", "params": "355M",
         "input": "text", "output": "text_features [B,77,1024]"},
        {"name": "Cross-Attention", "model": "FusionNet", "params": "120M",
         "input": "visual+text", "output": "fused [B,257,1024]"},
        {"name": "Decoder", "model": "GPT-2 Medium", "params": "345M",
         "input": "fused", "output": "generated_text"},
    ]
    for i, stage in enumerate(pipeline_stages, 1):
        arrow = " → " if i < len(pipeline_stages) else ""
        print(f"  Stage {i}: {stage['name']} ({stage['model']}, {stage['params']} params)")
        print(f"          Input:  {stage['input']}")
        print(f"          Output: {stage['output']}")
        if i < len(pipeline_stages):
            print(f"          ↓")

    total_params = sum(int(s["params"].replace("M", "")) for s in pipeline_stages)
    print(f"\n  Total pipeline parameters: {total_params}M")

    # --- 1c. Output Formatting ---
    print("\n--- 1c. Output Formatting ---")
    raw_output = {
        "text_logits": [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.2],
        "confidence": 0.82,
        "attention_weights": [0.1, 0.15, 0.25, 0.3, 0.12, 0.05, 0.03],
    }
    # post-processing
    tokens = ["A", "beautiful", "sunset", "over", "the", "mountains", "."]
    text_output = []
    for token, logit in zip(tokens, raw_output["text_logits"]):
        prob = math.exp(logit) / sum(math.exp(l) for l in raw_output["text_logits"])
        text_output.append({"token": token, "probability": prob})

    print(f"  Generated text: {' '.join(t['token'] for t in text_output)}")
    print(f"  Confidence: {raw_output['confidence']:.2%}")
    print(f"  Token probabilities:")
    for t in text_output[:5]:
        bar = "█" * int(t["probability"] * 40)
        print(f"    {t['token']:>12s}: {t['probability']:.3f} {bar}")

    # --- 1d. Pipeline Latency Breakdown ---
    print("\n--- 1d. Pipeline Latency Breakdown ---")
    latency_breakdown = [
        ("Preprocessing", 15),
        ("Vision Encoder", 45),
        ("Text Encoder", 20),
        ("Cross-Attention", 30),
        ("Decoder (10 tokens)", 80),
        ("Postprocessing", 10),
    ]
    total_ms = sum(ms for _, ms in latency_breakdown)
    print(f"  {'Stage':>25s} {'Latency':>10s} {'%':>6s} {'Bar'}")
    for stage, ms in latency_breakdown:
        pct = ms / total_ms
        bar = "█" * int(pct * 50)
        print(f"  {stage:>25s} {ms:>7d}ms {pct:>6.0%} {bar}")
    print(f"  {'TOTAL':>25s} {total_ms:>7d}ms 100%")
    print(f"  Throughput: {1000/total_ms:.1f} requests/sec")


# ---------------------------------------------------------------------------
# Demo 2 — API Design for Multimodal
# ---------------------------------------------------------------------------

def demo_api_design():
    """
    Shows file upload handling, streaming, and error handling patterns.
    """
    print("=" * 70)
    print("DEMO 2 — API Design for Multimodal")
    print("=" * 70)

    # --- 2a. File Upload Handling ---
    print("\n--- 2a. File Upload Handling ---")
    file_types = {
        "image/jpeg": {"max_size_mb": 10, "accepted": True,
                       "preprocess": "decode → resize → normalize"},
        "image/png":  {"max_size_mb": 10, "accepted": True,
                       "preprocess": "decode → alpha strip → resize → normalize"},
        "video/mp4":  {"max_size_mb": 100, "accepted": True,
                       "preprocess": "extract frames → sample at 1fps"},
        "audio/wav":  {"max_size_mb": 25, "accepted": True,
                       "preprocess": "resample → mel spectrogram"},
        "text/plain":{"max_size_mb": 1, "accepted": True,
                       "preprocess": "tokenize → truncate to 77 tokens"},
        "application/pdf": {"max_size_mb": 20, "accepted": True,
                            "preprocess": "OCR → page extraction → chunk"},
    }
    print(f"  {'Content-Type':>22s} {'Max Size':>10s} {'Accepted':>10s}")
    for ct, info in file_types.items():
        status = "✓" if info["accepted"] else "✗"
        print(f"  {ct:>22s} {info['max_size_mb']:>7d}MB {status:>10s}")

    # --- 2b. API Endpoint Design ---
    print("\n--- 2b. API Endpoint Design ---")
    endpoints = [
        {"method": "POST", "path": "/v1/analyze",
         "description": "Analyze image + text",
         "request_type": "multipart/form-data",
         "response": "JSON with analysis results"},
        {"method": "POST", "path": "/v1/generate",
         "description": "Generate text from image",
         "request_type": "multipart/form-data",
         "response": "SSE stream of tokens"},
        {"method": "GET", "path": "/v1/models",
         "description": "List available models",
         "request_type": "none",
         "response": "JSON model list"},
        {"method": "GET", "path": "/v1/health",
         "description": "Health check",
         "request_type": "none",
         "response": "JSON status"},
    ]
    for ep in endpoints:
        print(f"\n  {ep['method']:>4s} {ep['path']}")
        print(f"       {ep['description']}")
        print(f"       Request:  {ep['request_type']}")
        print(f"       Response: {ep['response']}")

    # --- 2c. Streaming Response ---
    print("\n--- 2c. Streaming Response (SSE) ---")
    tokens_stream = [
        "The", " image", " shows", " a", " mountain", " landscape",
        " with", " snow", "-capped", " peaks", " under", " a", " clear",
        " blue", " sky", ".", " There", " are", " pine", " trees",
        " in", " the", " foreground", "."
    ]
    print(f"  Simulated streaming ({len(tokens_stream)} tokens):")
    buffer = ""
    for i, token in enumerate(tokens_stream):
        buffer += token
        # simulate token-by-token streaming
        if i % 8 == 7 or i == len(tokens_stream) - 1:
            print(f"    [{i+1:>2d}] \"{buffer}\"")

    # --- 2d. Error Handling ---
    print("\n--- 2d. Error Handling Patterns ---")
    error_cases = [
        {"code": 400, "type": "BadRequest", "message": "Unsupported file type: .bmp",
         "retry": False, "action": "Return error, suggest supported formats"},
        {"code": 413, "type": "PayloadTooLarge",
         "message": "File size 50MB exceeds limit of 10MB",
         "retry": False, "action": "Return error with max size info"},
        {"code": 429, "type": "RateLimited",
         "message": "Rate limit exceeded: 100 req/min",
         "retry": True, "action": "Exponential backoff, retry after 60s"},
        {"code": 503, "type": "ServiceUnavailable",
         "message": "Model loading, please retry",
         "retry": True, "action": "Retry with 5s delay, max 3 retries"},
        {"code": 500, "type": "InternalServerError",
         "message": "CUDA out of memory",
         "retry": True, "action": "Retry on different GPU, alert ops"},
    ]
    for err in error_cases:
        retry_str = "RETRY" if err["retry"] else "NO RETRY"
        print(f"  {err['code']} {err['type']:>22s}: {retry_str}")
        print(f"     Message: \"{err['message']}\"")
        print(f"     Action:  {err['action']}")


# ---------------------------------------------------------------------------
# Demo 3 — Production Considerations
# ---------------------------------------------------------------------------

def demo_production_considerations():
    """
    Shows caching, batching, and cost management for multimodal APIs.
    """
    print("=" * 70)
    print("DEMO 3 — Production Considerations")
    print("=" * 70)

    # --- 3a. Response Caching ---
    print("\n--- 3a. Response Caching ---")
    requests = [
        {"hash": hashlib.md5(b"image1+text_a").hexdigest()[:12],
         "image": "photo_001.jpg", "text": "a dog playing"},
        {"hash": hashlib.md5(b"image1+text_a").hexdigest()[:12],
         "image": "photo_001.jpg", "text": "a dog playing"},  # duplicate
        {"hash": hashlib.md5(b"image2+text_b").hexdigest()[:12],
         "image": "photo_002.jpg", "text": "a cat sleeping"},
        {"hash": hashlib.md5(b"image1+text_a").hexdigest()[:12],
         "image": "photo_001.jpg", "text": "a dog playing"},  # duplicate
    ]

    cache = {}
    cache_hits = 0
    for req in requests:
        is_hit = req["hash"] in cache
        if is_hit:
            cache_hits += 1
            result = cache[req["hash"]]
            print(f"  CACHE HIT  [{req['hash']}] → {result}")
        else:
            result = f"analysis({req['image']}, {req['text']})"
            cache[req["hash"]] = result
            print(f"  CACHE MISS [{req['hash']}] → computing {result}")

    hit_rate = cache_hits / len(requests)
    print(f"\n  Cache hit rate: {cache_hits}/{len(requests)} = {hit_rate:.0%}")
    print(f"  Cost savings: ~${cache_hits * 0.002:.3f} per batch")

    # --- 3b. Request Batching ---
    print("\n--- 3b. Request Batching ---")
    batch_configs = [
        {"batch_size": 1,  "throughput_rps": 10,  "latency_ms": 100,
         "gpu_util": 0.15},
        {"batch_size": 4,  "throughput_rps": 32,  "latency_ms": 125,
         "gpu_util": 0.45},
        {"batch_size": 16, "throughput_rps": 80,  "latency_ms": 200,
         "gpu_util": 0.75},
        {"batch_size": 32, "throughput_rps": 120, "latency_ms": 267,
         "gpu_util": 0.88},
        {"batch_size": 64, "throughput_rps": 140, "latency_ms": 457,
         "gpu_util": 0.92},
    ]
    print(f"  {'Batch':>6s} {'RPS':>6s} {'Latency':>10s} {'GPU%':>6s} {'Efficiency'}")
    for cfg in batch_configs:
        efficiency = cfg["throughput_rps"] / (cfg["latency_ms"] / 1000)
        print(f"  {cfg['batch_size']:>6d} {cfg['throughput_rps']:>6d} "
              f"{cfg['latency_ms']:>7d}ms {cfg['gpu_util']:>6.0%} "
              f"{efficiency:>8.0f}")

    # --- 3c. Cost Management ---
    print("\n--- 3c. Cost Management ---")
    pricing = {
        "vision_model": {"cost_per_1k_images": 0.05, "daily_volume": 100000},
        "text_model":   {"cost_per_1k_tokens": 0.002, "daily_volume": 5000000},
        "decoder":      {"cost_per_1k_tokens": 0.006, "daily_volume": 2000000},
    }
    total_daily = 0
    for model, info in pricing.items():
        unit_key = [k for k in info if k.startswith("cost_per_1k")][0]
        unit_name = unit_key.replace("cost_per_1k_", "")
        cost = info["daily_volume"] / 1000 * info[unit_key]
        total_daily += cost
        print(f"  {model:>20s}: ${cost:>7.2f}/day ({info['daily_volume']:>10,} {unit_name})")

    print(f"\n  {'Total daily cost':>20s}: ${total_daily:>8.2f}")
    print(f"  {'Monthly estimate':>20s}: ${total_daily * 30:>8.2f}")

    # --- 3d. Auto-scaling Rules ---
    print("\n--- 3d. Auto-scaling Rules ---")
    scaling_rules = [
        {"metric": "GPU utilization", "scale_up": ">80%", "scale_down": "<30%",
         "cooldown": "5 min", "min_instances": 2, "max_instances": 20},
        {"metric": "Request queue", "scale_up": ">100", "scale_down": "<10",
         "cooldown": "2 min", "min_instances": 2, "max_instances": 20},
        {"metric": "P95 latency", "scale_up": ">500ms", "scale_down": "<100ms",
         "cooldown": "5 min", "min_instances": 2, "max_instances": 20},
    ]
    for rule in scaling_rules:
        print(f"\n  Metric: {rule['metric']}")
        print(f"    Scale UP:   {rule['scale_up']}  (cooldown: {rule['cooldown']})")
        print(f"    Scale DOWN: {rule['scale_down']} (cooldown: {rule['cooldown']})")
        print(f"    Instances:  {rule['min_instances']}-{rule['max_instances']}")


# ---------------------------------------------------------------------------
# Demo 4 — Real-World Applications
# ---------------------------------------------------------------------------

def demo_real_world_applications():
    """
    Shows document AI, visual search, and accessibility applications.
    """
    print("=" * 70)
    print("DEMO 4 — Real-World Applications")
    print("=" * 70)

    # --- 4a. Document AI ---
    print("\n--- 4a. Document AI Pipeline ---")
    doc_pipeline = {
        "input": "scanned invoice (PDF)",
        "stages": [
            ("OCR", "Tesseract/EasyOCR → raw text + bounding boxes", 0.05),
            ("Layout Analysis", "detect tables, headers, key-value pairs", 0.02),
            ("Entity Extraction", "invoice#, date, total, vendor name", 0.01),
            ("Validation", "check against PO, verify totals", 0.005),
            ("Export", "structured JSON → ERP system", 0.001),
        ],
        "accuracy": {"ocr": 0.98, "layout": 0.95, "entity": 0.92, "end_to_end": 0.88},
    }
    print(f"  Input: {doc_pipeline['input']}")
    print(f"  Pipeline stages:")
    for stage, desc, latency in doc_pipeline["stages"]:
        print(f"    {stage:>20s}: {desc} ({latency*1000:.0f}ms)")
    print(f"\n  Accuracy:")
    for metric, value in doc_pipeline["accuracy"].items():
        print(f"    {metric:>20s}: {value:.0%}")

    # --- 4b. Visual Search ---
    print("\n--- 4b. Visual Search System ---")
    catalog_size = 1_000_000
    embedding_dim = 512
    index_type = "HNSW"

    # simulate search
    query_features = [random.random() for _ in range(embedding_dim)]
    results = []
    for rank in range(5):
        similarity = random.uniform(0.7, 0.99)
        item_id = random.randint(1, catalog_size)
        results.append((rank + 1, item_id, similarity))

    print(f"  Catalog: {catalog_size:,} items, embedding dim: {embedding_dim}")
    print(f"  Index: {index_type} (memory: {catalog_size * embedding_dim * 4 / 1e6:.0f}MB)")
    print(f"\n  Top-5 results for query:")
    for rank, item_id, sim in results:
        bar = "█" * int(sim * 40)
        print(f"    #{rank} item_{item_id:06d}: similarity={sim:.4f} {bar}")

    # --- 4c. Accessibility ---
    print("\n--- 4c. Accessibility Applications ---")
    accessibility_apps = [
        {"name": "Image Description (Alt Text)",
         "input": "image", "output": "text description",
         "accuracy": 0.89, "latency_ms": 120,
         "use_cases": ["screen readers", "search indexing", "social media"]},
        {"name": "Scene Understanding",
         "input": "image/video", "output": "structured scene graph",
         "accuracy": 0.82, "latency_ms": 250,
         "use_cases": ["blind navigation", "AR assistance", "robotics"]},
        {"name": "Document Reading",
         "input": "document image", "output": "spoken text",
         "accuracy": 0.95, "latency_ms": 180,
         "use_cases": ["dyslexia support", "visual impairment", "multilingual"]},
    ]
    for app in accessibility_apps:
        print(f"\n  {app['name']}")
        print(f"    Input:  {app['input']}")
        print(f"    Output: {app['output']}")
        print(f"    Accuracy: {app['accuracy']:.0%}, Latency: {app['latency_ms']}ms")
        print(f"    Use cases: {', '.join(app['use_cases'])}")

    # --- 4d. End-to-End Metrics ---
    print("\n--- 4d. End-to-End Application Metrics ---")
    metrics = {
        "Latency (P50)": "120ms",
        "Latency (P99)": "450ms",
        "Throughput": "1,200 req/sec",
        "Accuracy (human eval)": "91.2%",
        "Cost per request": "$0.0023",
        "Monthly active users": "50,000",
        "Uptime (30d)": "99.97%",
        "Error rate": "0.08%",
    }
    for metric, value in metrics.items():
        print(f"  {metric:>30s}: {value}")

    print("\n  Key takeaways:")
    print("    1. Start with pre-trained multimodal models (CLIP, LLaVA)")
    print("    2. Use caching and batching to reduce costs 3-5x")
    print("    3. Implement streaming for better UX (time-to-first-token)")
    print("    4. Monitor accuracy, latency, and cost continuously")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_pipeline_design()
    demo_api_design()
    demo_production_considerations()
    demo_real_world_applications()
