"""
122 — LLM Deployment: контейнеризация, масштабирование, мониторинг

Темы:
  1. Containerization Patterns (Docker, model packaging, dependencies)
  2. Scaling Strategies (horizontal/vertical, auto-scaling, GPU scheduling)
  3. Health Checks (readiness, liveness, model health)
  4. Deployment Strategies (blue-green, canary, rollback)

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
# DEMO 1 — Containerization Patterns: Docker, model packaging, dependencies
# ─────────────────────────────────────────────────────────────
def demo_containerization():
    print("=" * 70)
    print("DEMO 1 — Containerization Patterns: Docker, model packaging, dependencies")
    print("=" * 70)

    # --- 1.1 Dockerfile generation ---
    print("\n--- 1.1 Dockerfile generation for LLM serving ---")

    def generate_dockerfile(model_type="transformer", gpu=True, quantize=False):
        layers = []
        layers.append("# Stage 1: Build")
        layers.append("FROM python:3.11-slim AS builder")
        layers.append("WORKDIR /app")
        layers.append("COPY requirements.txt .")
        layers.append("RUN pip install --no-cache-dir -r requirements.txt")
        layers.append("")
        layers.append("# Stage 2: Runtime")
        if gpu:
            layers.append("FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime")
            layers.append("ENV NVIDIA_VISIBLE_DEVICES=all")
        else:
            layers.append("FROM python:3.11-slim AS runtime")
        layers.append("WORKDIR /app")
        layers.append("COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages")
        layers.append("COPY . .")
        layers.append("ENV MODEL_PATH=/app/models/" + model_type)
        layers.append("ENV QUANTIZE=" + str(quantize).lower())
        layers.append("EXPOSE 8080")
        layers.append("HEALTHCHECK --interval=30s --timeout=10s --retries=3")
        layers.append("  CMD curl -f http://localhost:8080/health || exit 1")
        layers.append('CMD ["python", "serve.py", "--model", "${MODEL_PATH}"]')
        return "\n".join(layers)

    dockerfile = generate_dockerfile("llama-7b", gpu=True, quantize=True)
    print(f"  Generated Dockerfile ({len(dockerfile.splitlines())} lines):")
    for line in dockerfile.splitlines()[:15]:
        print(f"    {line}")
    print(f"    ...")

    # --- 1.2 Model packaging: size estimation ---
    print("\n--- 1.2 Model packaging: size & dependency estimation ---")

    def estimate_model_package(model_name, params_m, quantization=None, includes_tokenizer=True):
        # Base model size
        if quantization == "fp16":
            size_gb = params_m * 1e6 * 2 / 1e9
        elif quantization == "int8":
            size_gb = params_m * 1e6 * 1 / 1e9
        elif quantization == "int4":
            size_gb = params_m * 1e6 * 0.5 / 1e9
        else:  # fp32
            size_gb = params_m * 1e6 * 4 / 1e9

        # Tokenizer
        tokenizer_size = 0.015 if includes_tokenizer else 0  # ~15MB typical

        # Dependencies
        deps = {
            "transformers": 0.5,
            "torch": 2.0,
            "cuda_runtime": 0.5,
            "fastapi": 0.01,
            "uvicorn": 0.005,
            "nginx": 0.02,
        }
        dep_size = sum(deps.values())

        total = size_gb + tokenizer_size + dep_size

        return {
            "model_size_gb": round(size_gb, 3),
            "tokenizer_gb": round(tokenizer_size, 3),
            "deps_gb": round(dep_size, 3),
            "total_gb": round(total, 3),
            "quantization": quantization or "fp32",
        }

    models = [
        ("GPT-2 Small", 124, None),
        ("GPT-2 Medium", 355, "fp16"),
        ("LLaMA-7B", 7000, "int8"),
        ("LLaMA-7B", 7000, "int4"),
        ("LLaMA-13B", 13000, "int4"),
    ]

    print(f"  {'Model':<16} | {'Params':>8} | {'Quant':>5} | {'Model':>8} | {'Deps':>6} | {'Total':>8}")
    print(f"  {'-'*65}")
    for name, params, quant in models:
        pkg = estimate_model_package(name, params, quant)
        print(f"  {name:<16} | {params:>7,}M | {pkg['quantization']:>5} | {pkg['model_size_gb']:>7.2f}G | {pkg['deps_gb']:>5.2f}G | {pkg['total_gb']:>7.2f}G")

    # --- 1.3 Dependency resolution simulation ---
    print("\n--- 1.3 Dependency resolution simulation ---")

    def resolve_dependencies(packages):
        """Simulate dependency resolution with conflict detection."""
        dep_graph = {
            "torch": {"version": "2.1.0", "deps": ["numpy>=1.24", "nvidia-cuda-runtime>=12.1"]},
            "transformers": {"version": "4.36.0", "deps": ["torch>=2.0", "tokenizers>=0.15", "numpy>=1.24"]},
            "tokenizers": {"version": "0.15.0", "deps": []},
            "numpy": {"version": "1.26.0", "deps": []},
            "fastapi": {"version": "0.108.0", "deps": ["pydantic>=2.0", "starlette>=0.27"]},
            "uvicorn": {"version": "0.25.0", "deps": ["click>=8.0"]},
            "pydantic": {"version": "2.5.0", "deps": []},
            "starlette": {"version": "0.27.0", "deps": []},
            "click": {"version": "8.1.0", "deps": []},
            "nvidia-cuda-runtime": {"version": "12.1.0", "deps": []},
            "nvidia-cublas": {"version": "12.1.0", "deps": ["nvidia-cuda-runtime>=12.1"]},
            "nvidia-cudnn": {"version": "8.9.0", "deps": ["nvidia-cuda-runtime>=12.1"]},
        }

        resolved = {}
        conflicts = []
        order = []

        def resolve(pkg_name, visited=None):
            if visited is None:
                visited = set()
            if pkg_name in resolved:
                return True
            if pkg_name in visited:
                return False  # cycle
            visited.add(pkg_name)

            if pkg_name not in dep_graph:
                conflicts.append(f"Package '{pkg_name}' not found")
                return False

            pkg = dep_graph[pkg_name]
            for dep in pkg["deps"]:
                dep_name = re.match(r'([a-zA-Z0-9_-]+)', dep).group(1)
                if not resolve(dep_name, visited.copy()):
                    conflicts.append(f"Failed to resolve dependency: {dep}")

            resolved[pkg_name] = pkg["version"]
            order.append(pkg_name)
            return True

        for pkg in packages:
            resolve(pkg)

        return resolved, conflicts, order

    required = ["torch", "transformers", "fastapi", "uvicorn"]
    resolved, conflicts, order = resolve_dependencies(required)

    print(f"  Required packages: {required}")
    print(f"  Resolved {len(resolved)} packages ({len(order)} in dependency order):")
    for i, pkg in enumerate(order):
        print(f"    {i+1}. {pkg}=={resolved[pkg]}")
    if conflicts:
        print(f"  Conflicts: {conflicts}")
    else:
        print(f"  No conflicts detected")

    # --- 1.4 Image layer analysis ---
    print("\n--- 1.4 Docker image layer analysis ---")

    def analyze_layers(layers):
        total_size = sum(l["size_mb"] for l in layers)
        cached_count = sum(1 for l in layers if l["cached"])
        cache_ratio = cached_count / len(layers) * 100
        return {
            "total_layers": len(layers),
            "total_size_mb": total_size,
            "cached_layers": cached_count,
            "cache_ratio": cache_ratio,
            "rebuild_time_estimate_s": total_size * 0.01 * (1 - cache_ratio / 100),
        }

    image_layers = [
        {"name": "base_cuda", "size_mb": 800, "cached": True},
        {"name": "python_runtime", "size_mb": 150, "cached": True},
        {"name": "pip_install_deps", "size_mb": 3500, "cached": False},
        {"name": "copy_model_weights", "size_mb": 14000, "cached": False},
        {"name": "copy_app_code", "size_mb": 5, "cached": False},
        {"name": "config_files", "size_mb": 1, "cached": True},
    ]

    stats = analyze_layers(image_layers)
    print(f"  Image layers:")
    for l in image_layers:
        cache_mark = " (cached)" if l["cached"] else " (needs rebuild)"
        print(f"    {l['name']:<25}: {l['size_mb']:>8,} MB{cache_mark}")
    print(f"\n  Total layers:   {stats['total_layers']}")
    print(f"  Total size:     {stats['total_size_mb']:,} MB ({stats['total_size_mb']/1024:.1f} GB)")
    print(f"  Cache hit rate: {stats['cache_ratio']:.0f}%")
    print(f"  Rebuild time:   ~{stats['rebuild_time_estimate_s']:.0f}s (estimated)")

    print("\n" + "=" * 70)
    print("DEMO 1 COMPLETE — Containerization patterns demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# DEMO 2 — Scaling Strategies: horizontal/vertical, auto-scaling, GPU scheduling
# ─────────────────────────────────────────────────────────────
def demo_scaling_strategies():
    print("=" * 70)
    print("DEMO 2 — Scaling Strategies: horizontal/vertical, auto-scaling, GPU scheduling")
    print("=" * 70)

    # --- 2.1 Horizontal vs Vertical scaling ---
    print("\n--- 2.1 Horizontal vs Vertical scaling cost analysis ---")

    def scaling_cost_analysis(base_throughput_rps, target_rps, base_latency_ms):
        """Compare horizontal and vertical scaling approaches."""
        # Vertical scaling: bigger machine
        # Throughput ~ memory (diminishing returns)
        # Cost ~ memory^1.5 (rough)
        vertical_options = []
        for multiplier in [1, 2, 4, 8]:
            new_throughput = base_throughput_rps * math.log2(1 + multiplier) / math.log2(2)
            new_latency = base_latency_ms / (multiplier ** 0.3)  # sublinear improvement
            cost = 50 * multiplier ** 1.5  # GPU instance cost
            vertical_options.append({
                "strategy": f"Vertical ({multiplier}x GPU)",
                "throughput": round(new_throughput, 1),
                "latency_ms": round(new_latency, 1),
                "cost_per_hour": round(cost, 2),
                "meets_target": new_throughput >= target_rps,
            })

        # Horizontal scaling: more machines
        for n_machines in [1, 2, 4, 8]:
            new_throughput = base_throughput_rps * n_machines * 0.92  # 8% overhead
            new_latency = base_latency_ms * 1.05 ** (n_machines - 1)  # slight increase
            cost = 50 * n_machines
            vertical_options.append({
                "strategy": f"Horizontal ({n_machines}x node)",
                "throughput": round(new_throughput, 1),
                "latency_ms": round(new_latency, 1),
                "cost_per_hour": round(cost, 2),
                "meets_target": new_throughput >= target_rps,
            })

        return vertical_options

    target = 100
    results = scaling_cost_analysis(25, target, 50)

    print(f"  Base: {25} RPS, {50}ms latency | Target: {target} RPS")
    print(f"\n  {'Strategy':<24} | {'Throughput':>10} | {'Latency':>10} | {'Cost/hr':>10} | {'Meets Target':>13}")
    print(f"  {'-'*75}")
    for r in results:
        mark = "YES" if r["meets_target"] else "no"
        print(f"  {r['strategy']:<24} | {r['throughput']:>9.1f} | {r['latency_ms']:>8.1f}ms | ${r['cost_per_hour']:>8.2f} | {mark:>13}")

    # --- 2.2 Auto-scaling simulation ---
    print("\n--- 2.2 Auto-scaling simulation: reactive vs predictive ---")

    def simulate_auto_scaling(n_hours=24, base_rps=30):
        """Simulate traffic pattern and auto-scaling behavior."""
        random.seed(42)

        # Traffic pattern: daily cycle with noise
        traffic = []
        for h in range(n_hours):
            base = base_rps * (1 + 0.5 * math.sin(2 * math.pi * (h - 6) / 24))
            noise = random.gauss(0, base_rps * 0.1)
            traffic.append(max(0, base + noise))

        # Reactive scaling: scale when CPU > 80%
        reactive_instances = []
        current_instances = 1
        for rps in traffic:
            capacity_per_instance = 40  # RPS per instance
            needed = math.ceil(rps / capacity_per_instance)
            # Reactive: only scale up quickly, scale down slowly
            if needed > current_instances:
                current_instances = needed  # Scale up immediately
            elif needed < current_instances - 1:
                current_instances -= 1  # Scale down slowly
            reactive_instances.append(current_instances)

        # Predictive scaling: pre-scale based on forecast
        predictive_instances = []
        for h in range(n_hours):
            # Simple prediction: average of next 2 hours
            next_indices = [(h + 1 + k) % n_hours for k in range(2)]
            future_avg = statistics.mean([traffic[i] for i in next_indices])
            needed = math.ceil(future_avg / capacity_per_instance)
            predictive_instances.append(max(1, needed))

        # Cost analysis
        cost_per_instance_hour = 2.50
        reactive_cost = sum(reactive_instances) * cost_per_instance_hour
        predictive_cost = sum(predictive_instances) * cost_per_instance_hour

        return {
            "traffic": traffic,
            "reactive": reactive_instances,
            "predictive": predictive_instances,
            "reactive_cost": reactive_cost,
            "predictive_cost": predictive_cost,
            "reactive_max": max(reactive_instances),
            "predictive_max": max(predictive_instances),
            "reactive_avg": statistics.mean(reactive_instances),
            "predictive_avg": statistics.mean(predictive_instances),
        }

    sim = simulate_auto_scaling()

    print(f"  24-hour simulation:")
    print(f"  {'Hour':>4} | {'Traffic':>8} | {'Reactive':>10} | {'Predictive':>11}")
    print(f"  {'-'*42}")
    for h in range(0, 24, 3):
        print(f"  {h:>4} | {sim['traffic'][h]:>7.1f} | {sim['reactive'][h]:>10} | {sim['predictive'][h]:>11}")

    print(f"\n  Reactive:   avg={sim['reactive_avg']:.1f} instances, max={sim['reactive_max']}, cost=${sim['reactive_cost']:.2f}")
    print(f"  Predictive: avg={sim['predictive_avg']:.1f} instances, max={sim['predictive_max']}, cost=${sim['predictive_cost']:.2f}")
    print(f"  Cost savings (predictive): ${(sim['reactive_cost'] - sim['predictive_cost']):.2f} ({(1-sim['predictive_cost']/sim['reactive_cost'])*100:.1f}%)")

    # --- 2.3 GPU scheduling ---
    print("\n--- 2.3 GPU scheduling: bin-packing simulation ---")

    def gpu_scheduling_simulation():
        random.seed(42)
        gpus = [{"id": i, "vram_gb": 80, "used_gb": 0, "models": []} for i in range(4)]

        model_requests = [
            {"name": "LLaMA-7B-fp16", "vram_needed": 14},
            {"name": "LLaMA-7B-int8", "vram_needed": 7},
            {"name": "GPT-2-xl", "vram_needed": 3},
            {"name": "Whisper-large", "vram_needed": 6},
            {"name": "LLaMA-13B-int4", "vram_needed": 8},
            {"name": "SD-xl", "vram_needed": 12},
            {"name": "BERT-large", "vram_needed": 2},
            {"name": "CLIP-vit", "vram_needed": 4},
        ]

        # First-fit decreasing bin packing
        model_requests_sorted = sorted(model_requests, key=lambda x: x["vram_needed"], reverse=True)

        scheduled = []
        failed = []
        for model in model_requests_sorted:
            placed = False
            for gpu in gpus:
                if gpu["vram_gb"] - gpu["used_gb"] >= model["vram_needed"]:
                    gpu["used_gb"] += model["vram_needed"]
                    gpu["models"].append(model["name"])
                    scheduled.append({"model": model["name"], "gpu": gpu["id"], "vram": model["vram_needed"]})
                    placed = True
                    break
            if not placed:
                failed.append(model["name"])

        total_used = sum(g["used_gb"] for g in gpus)
        total_available = sum(g["vram_gb"] for g in gpus)
        utilization = total_used / total_available * 100

        print(f"  GPUs: {len(gpus)} x {gpus[0]['vram_gb']}GB VRAM")
        print(f"  Models to schedule: {len(model_requests)}")
        print(f"\n  GPU allocation:")
        for gpu in gpus:
            usage_pct = gpu["used_gb"] / gpu["vram_gb"] * 100
            bar = "█" * int(usage_pct / 5)
            print(f"    GPU {gpu['id']}: {gpu['used_gb']:>2}/{gpu['vram_gb']}GB ({usage_pct:.0f}%) {bar}")
            for m in gpu["models"]:
                print(f"         → {m}")
        print(f"\n  Total VRAM usage: {total_used}/{total_available}GB ({utilization:.1f}%)")
        print(f"  Scheduled: {len(scheduled)}/{len(model_requests)}")
        print(f"  Failed:    {len(failed)}")
        if failed:
            print(f"  Failed models: {', '.join(failed)}")

    gpu_scheduling_simulation()

    # --- 2.4 Load balancing strategies ---
    print("\n--- 2.4 Load balancing strategy comparison ---")

    def compare_load_balancers(n_requests=1000, n_servers=4):
        random.seed(42)
        server_capacities = [100, 80, 120, 90]  # different server capacities

        strategies = {}

        # Round-robin
        loads = [0] * n_servers
        for i in range(n_requests):
            server = i % n_servers
            loads[server] += 1
        strategies["round_robin"] = {
            "loads": loads,
            "max_load": max(loads),
            "load_variance": statistics.variance(loads) if len(loads) > 1 else 0,
            "imbalance": (max(loads) - min(loads)) / statistics.mean(loads) * 100,
        }

        # Least connections (simulated)
        loads = [0] * n_servers
        for _ in range(n_requests):
            server = loads.index(min(loads))
            loads[server] += 1
        strategies["least_connections"] = {
            "loads": loads,
            "max_load": max(loads),
            "load_variance": statistics.variance(loads) if len(loads) > 1 else 0,
            "imbalance": (max(loads) - min(loads)) / statistics.mean(loads) * 100,
        }

        # Weighted round-robin (proportional to capacity)
        loads = [0] * n_servers
        total_capacity = sum(server_capacities)
        weights = [c / total_capacity for c in server_capacities]
        for _ in range(n_requests):
            r = random.random()
            cumulative = 0
            for s in range(n_servers):
                cumulative += weights[s]
                if r <= cumulative:
                    loads[s] += 1
                    break
        strategies["weighted_rr"] = {
            "loads": loads,
            "max_load": max(loads),
            "load_variance": statistics.variance(loads) if len(loads) > 1 else 0,
            "imbalance": (max(loads) - min(loads)) / statistics.mean(loads) * 100,
        }

        # Consistent hashing (simulated)
        loads = [0] * n_servers
        for _ in range(n_requests):
            h = random.random() * 360
            server = int(h / 360 * n_servers)
            loads[min(server, n_servers - 1)] += 1
        strategies["consistent_hash"] = {
            "loads": loads,
            "max_load": max(loads),
            "load_variance": statistics.variance(loads) if len(loads) > 1 else 0,
            "imbalance": (max(loads) - min(loads)) / statistics.mean(loads) * 100,
        }

        print(f"  Servers: {n_servers}, Requests: {n_requests}")
        print(f"  Server capacities: {server_capacities}")
        print(f"\n  {'Strategy':<22} | {'Loads':>20} | {'Variance':>10} | {'Imbalance':>10}")
        print(f"  {'-'*72}")
        for name, data in strategies.items():
            loads_str = str(data["loads"])
            print(f"  {name:<22} | {loads_str:>20} | {data['load_variance']:>10.1f} | {data['imbalance']:>9.1f}%")

        best = min(strategies.items(), key=lambda x: x[1]["imbalance"])
        print(f"\n  Best strategy: {best[0]} (imbalance: {best[1]['imbalance']:.1f}%)")

    compare_load_balancers()

    print("\n" + "=" * 70)
    print("DEMO 2 COMPLETE — Scaling strategies demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# DEMO 3 — Health Checks: readiness, liveness, model health
# ─────────────────────────────────────────────────────────────
def demo_health_checks():
    print("=" * 70)
    print("DEMO 3 — Health Checks: readiness, liveness, model health")
    print("=" * 70)

    # --- 3.1 Readiness probe simulation ---
    print("\n--- 3.1 Readiness probe simulation ---")

    def simulate_readiness_checks(n_checks=20):
        random.seed(42)
        checks = []
        for i in range(n_checks):
            # Model loaded?
            model_loaded = random.random() > 0.1 if i > 2 else False
            # GPU memory available?
            gpu_memory_ok = random.random() > 0.05
            # Config loaded?
            config_loaded = random.random() > 0.02
            # Dependencies ready?
            deps_ready = random.random() > 0.03

            ready = model_loaded and gpu_memory_ok and config_loaded and deps_ready
            latency_ms = random.uniform(1, 5) if ready else 0

            checks.append({
                "check_num": i + 1,
                "ready": ready,
                "model_loaded": model_loaded,
                "gpu_memory_ok": gpu_memory_ok,
                "config_loaded": config_loaded,
                "deps_ready": deps_ready,
                "latency_ms": round(latency_ms, 2),
            })
        return checks

    readiness_checks = simulate_readiness_checks()
    ready_count = sum(1 for c in readiness_checks if c["ready"])
    first_ready = next((c["check_num"] for c in readiness_checks if c["ready"]), None)

    print(f"  Checks performed: {len(readiness_checks)}")
    print(f"  Ready responses:  {ready_count}/{len(readiness_checks)}")
    print(f"  First ready at:   Check #{first_ready}")
    print(f"\n  Check sequence:")
    for c in readiness_checks[:15]:
        status = "READY" if c["ready"] else "NOT READY"
        icon = "✓" if c["ready"] else "✗"
        details = (f"model={'Y' if c['model_loaded'] else 'N'} "
                   f"gpu={'Y' if c['gpu_memory_ok'] else 'N'} "
                   f"config={'Y' if c['config_loaded'] else 'N'} "
                   f"deps={'Y' if c['deps_ready'] else 'N'}")
        print(f"    #{c['check_num']:>2} [{icon}] {status:<10} ({details})")

    # --- 3.2 Liveness probe: crash detection ---
    print("\n--- 3.2 Liveness probe: crash & deadlock detection ---")

    def simulate_liveness(n_periods=30):
        random.seed(42)
        events = []
        status = "healthy"
        consecutive_failures = 0
        restart_count = 0

        for period in range(n_periods):
            # Health status
            if status == "restarting":
                status = "healthy"
                consecutive_failures = 0
                events.append({"period": period, "status": "restarted", "restart_count": restart_count})
                continue

            # Random failure probability
            fail_prob = 0.05
            if status == "degraded":
                fail_prob = 0.3

            if random.random() < fail_prob:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    status = "crashed"
                    restart_count += 1
                    events.append({"period": period, "status": "crashed", "restart_count": restart_count})
                    status = "restarting"
                elif consecutive_failures >= 2:
                    status = "degraded"
                    events.append({"period": period, "status": "degraded", "restart_count": restart_count})
                else:
                    events.append({"period": period, "status": "warning", "restart_count": restart_count})
            else:
                consecutive_failures = 0
                status = "healthy"
                events.append({"period": period, "status": "healthy", "restart_count": restart_count})

        return events

    liveness_events = simulate_liveness()
    status_counts = collections.Counter(e["status"] for e in liveness_events)
    restarts = liveness_events[-1]["restart_count"] if liveness_events else 0

    print(f"  Monitoring period: {len(liveness_events)} check intervals")
    print(f"  Status distribution:")
    for status, count in status_counts.most_common():
        bar = "█" * count
        print(f"    {status:<12}: {bar} ({count})")
    print(f"  Total restarts:    {restarts}")

    # Timeline
    print(f"\n  Timeline (last 20 checks):")
    timeline = "".join(
        {"healthy": ".", "warning": "!", "degraded": "D", "crashed": "X", "restarted": "R"}.get(e["status"], "?")
        for e in liveness_events[-20:]
    )
    print(f"    {timeline}")
    print(f"    Legend: . = healthy, ! = warning, D = degraded, X = crashed, R = restarted")

    # --- 3.3 Model health: output quality monitoring ---
    print("\n--- 3.3 Model health: output quality monitoring ---")

    def monitor_model_quality(n_predictions=200):
        random.seed(42)
        metrics_history = []
        quality_degradation_start = 150  # simulate drift after this point

        for i in range(n_predictions):
            # Base quality scores
            relevance = random.gauss(0.85, 0.05)
            coherence = random.gauss(0.88, 0.04)
            safety_score = random.gauss(0.95, 0.02)
            latency_ms = random.gauss(120, 15)

            # Introduce drift after quality_degradation_start
            if i >= quality_degradation_start:
                drift = (i - quality_degradation_start) * 0.002
                relevance -= drift
                coherence -= drift * 0.5
                latency_ms += drift * 50

            metrics_history.append({
                "prediction": i + 1,
                "relevance": round(max(0, min(1, relevance)), 3),
                "coherence": round(max(0, min(1, coherence)), 3),
                "safety": round(max(0, min(1, safety_score)), 3),
                "latency_ms": round(max(10, latency_ms), 1),
            })

        return metrics_history

    quality_history = monitor_model_quality()

    # Analyze windows
    window_size = 50
    windows = []
    for start in range(0, len(quality_history), window_size):
        window = quality_history[start:start + window_size]
        if len(window) >= 10:
            windows.append({
                "range": f"{start+1}-{start+len(window)}",
                "avg_relevance": statistics.mean([m["relevance"] for m in window]),
                "avg_coherence": statistics.mean([m["coherence"] for m in window]),
                "avg_safety": statistics.mean([m["safety"] for m in window]),
                "avg_latency": statistics.mean([m["latency_ms"] for m in window]),
            })

    print(f"  Predictions monitored: {len(quality_history)}")
    print(f"\n  Quality windows (rolling {window_size}-prediction average):")
    print(f"  {'Range':<12} | {'Relevance':>10} | {'Coherence':>10} | {'Safety':>8} | {'Latency':>10}")
    print(f"  {'-'*60}")
    for w in windows:
        # Trend indicator
        rel_trend = "↓" if w["avg_relevance"] < 0.80 else "→" if w["avg_relevance"] < 0.83 else "↑"
        print(f"  {w['range']:<12} | {w['avg_relevance']:>9.3f}{rel_trend} | {w['avg_coherence']:>10.3f} | {w['avg_safety']:>8.3f} | {w['avg_latency']:>8.1f}ms")

    # Detect degradation
    if len(windows) >= 2:
        first_window = windows[0]
        last_window = windows[-1]
        relevance_drop = (first_window["avg_relevance"] - last_window["avg_relevance"]) / first_window["avg_relevance"] * 100
        latency_increase = (last_window["avg_latency"] - first_window["avg_latency"]) / first_window["avg_latency"] * 100
        print(f"\n  Degradation detected:")
        print(f"    Relevance drop:   {relevance_drop:.1f}%")
        print(f"    Latency increase: {latency_increase:.1f}%")
        if relevance_drop > 5:
            print(f"    Action: ALERT - Model quality degradation detected!")

    # --- 3.4 Health check dashboard ---
    print("\n--- 3.4 Health check dashboard ---")

    def generate_health_dashboard():
        return {
            "timestamp": "2024-01-15T14:30:00Z",
            "service": "llm-serving",
            "version": "2.1.0",
            "checks": {
                "readiness": {
                    "status": "healthy",
                    "model_loaded": True,
                    "gpu_available": True,
                    "uptime_s": 86400,
                },
                "liveness": {
                    "status": "healthy",
                    "consecutive_failures": 0,
                    "last_restart": "2024-01-14T08:00:00Z",
                },
                "model_health": {
                    "status": "degraded",
                    "quality_score": 0.78,
                    "avg_latency_ms": 185,
                    "error_rate": 0.023,
                    "drift_detected": True,
                },
                "resource": {
                    "status": "healthy",
                    "gpu_utilization": 0.72,
                    "memory_utilization": 0.65,
                    "queue_depth": 12,
                },
            },
            "overall_status": "degraded",
        }

    dashboard = generate_health_dashboard()
    print(f"  === HEALTH DASHBOARD ===")
    print(f"  Service:    {dashboard['service']} v{dashboard['version']}")
    print(f"  Timestamp:  {dashboard['timestamp']}")
    print(f"  Overall:    {dashboard['overall_status'].upper()}")
    print()
    for check_name, check in dashboard["checks"].items():
        icon = "✓" if check["status"] == "healthy" else "⚠" if check["status"] == "degraded" else "✗"
        print(f"  [{icon}] {check_name}:")
        for key, val in check.items():
            if key != "status":
                if isinstance(val, float) and val < 1:
                    print(f"      {key}: {val:.1%}")
                elif isinstance(val, bool):
                    print(f"      {key}: {'Yes' if val else 'No'}")
                else:
                    print(f"      {key}: {val}")

    print("\n" + "=" * 70)
    print("DEMO 3 COMPLETE — Health checks demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# DEMO 4 — Deployment Strategies: blue-green, canary, rollback
# ─────────────────────────────────────────────────────────────
def demo_deployment_strategies():
    print("=" * 70)
    print("DEMO 4 — Deployment Strategies: blue-green, canary, rollback")
    print("=" * 70)

    # --- 4.1 Blue-green deployment ---
    print("\n--- 4.1 Blue-green deployment simulation ---")

    def simulate_blue_green(n_requests=500, switch_at=250, failure_rate_green=0.02, failure_rate_blue=0.05):
        random.seed(42)
        events = []
        green_responses = 0
        blue_responses = 0
        green_failures = 0
        blue_failures = 0

        for i in range(n_requests):
            if i < switch_at:
                # Green (current version)
                is_failure = random.random() < failure_rate_green
                if is_failure:
                    green_failures += 1
                green_responses += 1
                events.append({"request": i, "env": "green", "success": not is_failure})
            else:
                # Blue (new version)
                is_failure = random.random() < failure_rate_blue
                if is_failure:
                    blue_failures += 1
                blue_responses += 1
                events.append({"request": i, "env": "blue", "success": not is_failure})

        green_success_rate = (green_responses - green_failures) / green_responses * 100 if green_responses > 0 else 0
        blue_success_rate = (blue_responses - blue_failures) / blue_responses * 100 if blue_responses > 0 else 0

        return {
            "green": {"responses": green_responses, "failures": green_failures, "success_rate": green_success_rate},
            "blue": {"responses": blue_responses, "failures": blue_failures, "success_rate": blue_success_rate},
            "switch_point": switch_at,
        }

    bg_result = simulate_blue_green()
    print(f"  Blue-Green Deployment:")
    print(f"  Traffic split: 100% Green → 100% Blue at request #{bg_result['switch_point']}")
    print(f"\n  {'Environment':<12} | {'Requests':>10} | {'Failures':>10} | {'Success Rate':>12}")
    print(f"  {'-'*50}")
    for env in ["green", "blue"]:
        data = bg_result[env]
        print(f"  {env:<12} | {data['responses']:>10} | {data['failures']:>10} | {data['success_rate']:>11.1f}%")

    # --- 4.2 Canary deployment ---
    print("\n--- 4.2 Canary deployment simulation ---")

    def simulate_canary(traffic_percentages=[1, 5, 10, 25, 50, 100],
                       canary_error_rate=0.03, baseline_error_rate=0.01):
        random.seed(42)
        results = []
        promoted = True

        for pct in traffic_percentages:
            n_requests = 1000
            canary_n = int(n_requests * pct / 100)
            baseline_n = n_requests - canary_n

            canary_errors = sum(1 for _ in range(canary_n) if random.random() < canary_error_rate)
            baseline_errors = sum(1 for _ in range(baseline_n) if random.random() < baseline_error_rate)

            canary_rate = canary_errors / canary_n * 100 if canary_n > 0 else 0
            baseline_rate = baseline_errors / baseline_n * 100 if baseline_n > 0 else 0

            # Promotion decision
            error_ratio = canary_rate / baseline_rate if baseline_rate > 0 else float('inf')
            should_promote = error_ratio < 2.0

            if not should_promote:
                promoted = False

            results.append({
                "traffic_pct": pct,
                "canary_n": canary_n,
                "canary_errors": canary_errors,
                "canary_rate": canary_rate,
                "baseline_rate": baseline_rate,
                "error_ratio": error_ratio,
                "promote": should_promote,
            })

        return results, promoted

    canary_results, final_promoted = simulate_canary()

    print(f"  Canary Deployment Stages:")
    print(f"  {'Traffic%':>10} | {'Canary N':>10} | {'Canary Err%':>12} | {'Baseline%':>10} | {'Ratio':>6} | {'Decision':>10}")
    print(f"  {'-'*70}")
    for r in canary_results:
        decision = "PROMOTE" if r["promote"] else "ROLLBACK"
        print(f"  {r['traffic_pct']:>9}% | {r['canary_n']:>10} | {r['canary_rate']:>11.2f}% | {r['baseline_rate']:>9.2f}% | {r['error_ratio']:>5.1f}x | {decision:>10}")

    print(f"\n  Final decision: {'PROMOTED' if final_promoted else 'ROLLED BACK'}")

    # --- 4.3 Rollback simulation ---
    print("\n--- 4.3 Rollback timing & impact analysis ---")

    def simulate_rollback(n_requests=1000, failure_start=400, rollback_trigger=410,
                         rollback_duration_s=30, impact_per_second=0.5):
        random.seed(42)
        events = []
        current_version = "v2.1.0"
        rolled_back = False
        rollback_start_time = 0
        total_affected = 0
        downtime_requests = 0

        for i in range(n_requests):
            request_time = i * 0.1  # 100ms between requests

            if i < failure_start:
                # Normal operation
                success = random.random() < 0.99
                events.append({"time": request_time, "version": current_version,
                             "success": success, "phase": "normal"})
            elif not rolled_back:
                # Failure period (before rollback)
                success = random.random() < 0.4  # 60% failure rate
                if not success:
                    total_affected += 1
                events.append({"time": request_time, "version": current_version,
                             "success": success, "phase": "failure"})
                if i >= rollback_trigger and not rolled_back:
                    rolled_back = True
                    rollback_start_time = request_time
            else:
                # Rollback in progress
                time_since_trigger = request_time - rollback_start_time
                if time_since_trigger < rollback_duration_s:
                    # Transitioning
                    success = random.random() < (0.4 + 0.6 * time_since_trigger / rollback_duration_s)
                    downtime_requests += 1
                    events.append({"time": request_time, "version": "transitioning",
                                 "success": success, "phase": "rollback"})
                else:
                    # Rollback complete
                    success = random.random() < 0.99
                    events.append({"time": request_time, "version": "v2.0.9",
                                 "success": success, "phase": "recovered"})

        # Summary
        phases = collections.Counter(e["phase"] for e in events)
        phase_success = {}
        for phase in ["normal", "failure", "rollback", "recovered"]:
            phase_events = [e for e in events if e["phase"] == phase]
            if phase_events:
                phase_success[phase] = sum(1 for e in phase_events if e["success"]) / len(phase_events) * 100

        return {
            "phases": dict(phases),
            "phase_success": phase_success,
            "total_affected": total_affected,
            "downtime_requests": downtime_requests,
            "recovery_time_s": rollback_duration_s,
        }

    rollback_result = simulate_rollback()

    print(f"  Rollback Timeline:")
    print(f"  Phase           | Requests | Success Rate")
    print(f"  {'-'*45}")
    for phase in ["normal", "failure", "rollback", "recovered"]:
        count = rollback_result["phases"].get(phase, 0)
        rate = rollback_result["phase_success"].get(phase, 0)
        bar = "█" * int(rate / 5)
        print(f"  {phase:<16}| {count:>8} | {rate:>6.1f}% {bar}")

    print(f"\n  Affected requests:  {rollback_result['total_affected']}")
    print(f"  Recovery time:      {rollback_result['recovery_time_s']}s")
    print(f"  Downtime requests:  {rollback_result['downtime_requests']}")

    # --- 4.4 Deployment strategy comparison ---
    print("\n--- 4.4 Deployment strategy comparison ---")

    def compare_strategies():
        strategies = {
            "Rolling Update": {
                "downtime_s": 0,
                "rollback_time_s": 60,
                "resource_overhead": 0.1,
                "complexity": 2,
                "risk": 3,
                "description": "Gradual replacement of instances",
            },
            "Blue-Green": {
                "downtime_s": 0,
                "rollback_time_s": 5,
                "resource_overhead": 1.0,
                "complexity": 3,
                "risk": 1,
                "description": "Two identical environments, instant switch",
            },
            "Canary": {
                "downtime_s": 0,
                "rollback_time_s": 10,
                "resource_overhead": 0.05,
                "complexity": 4,
                "risk": 2,
                "description": "Small traffic percentage to new version",
            },
            "Shadow": {
                "downtime_s": 0,
                "rollback_time_s": 5,
                "resource_overhead": 1.0,
                "complexity": 5,
                "risk": 1,
                "description": "Mirror traffic, compare results",
            },
        }

        # Weighted scoring
        weights = {"downtime_s": 0.3, "rollback_time_s": 0.2, "resource_overhead": 0.2,
                   "complexity": 0.15, "risk": 0.15}

        print(f"  {'Strategy':<18} | {'Downtime':>10} | {'Rollback':>10} | {'Overhead':>10} | {'Complexity':>11} | {'Risk':>5}")
        print(f"  {'-'*75}")
        for name, data in strategies.items():
            print(f"  {name:<18} | {data['downtime_s']:>8}s | {data['rollback_time_s']:>8}s | {data['resource_overhead']:>9.0%} | {data['complexity']:>11} | {data['risk']::>5}/5")

        # Score each strategy (lower is better)
        print(f"\n  Weighted Scores (lower = better):")
        scored = []
        for name, data in strategies.items():
            score = sum(weights[k] * data[k] for k in weights)
            scored.append((name, round(score, 3)))
        scored.sort(key=lambda x: x[1])
        for i, (name, score) in enumerate(scored):
            medal = "★" if i == 0 else " " if i == 1 else " "
            print(f"    {medal} {name:<18}: {score:.3f}")

        return scored

    scored_strategies = compare_strategies()

    print("\n" + "=" * 70)
    print("DEMO 4 COMPLETE — Deployment strategies demonstrated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_containerization()
    demo_scaling_strategies()
    demo_health_checks()
    demo_deployment_strategies()
