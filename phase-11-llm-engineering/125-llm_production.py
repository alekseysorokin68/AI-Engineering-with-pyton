"""125 — LLM Production Patterns: A/B тестирование, откат, мониторинг в продакшене

Темы:
  1. Production Architecture (API gateway, load balancing, circuit breakers)
  2. A/B Testing (traffic splitting, statistical significance, winner selection)
  3. Incident Response (rollback procedures, escalation, post-mortem)
  4. Quality Monitoring (drift detection, user feedback, continuous evaluation)

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
import datetime

random.seed(42)


# ---------------------------------------------------------------------------
# Demo 1: Production Architecture
# ---------------------------------------------------------------------------
def demo_production_architecture():
    """API gateway, load balancing, circuit breakers."""
    print("=" * 70)
    print("Demo 1 — Production Architecture: gateway, LB, circuit breakers")
    print("=" * 70)

    # 1.1 API Gateway request routing
    print("\n--- 1.1 API Gateway Routing ---")

    class APIGateway:
        def __init__(self):
            self.routes = {}
            self.middleware = []

        def route(self, path, method="POST"):
            self.routes[path] = method

        def add_middleware(self, name):
            self.middleware.append(name)

        def process(self, path):
            applied = list(self.middleware)
            endpoint = self.routes.get(path, "404 NOT FOUND")
            return {"path": path, "middleware": applied, "endpoint": endpoint}

    gw = APIGateway()
    gw.add_middleware("rate_limiter")
    gw.add_middleware("auth_validator")
    gw.add_middleware("request_logger")
    gw.route("/v1/chat/completions", "POST")
    gw.route("/v1/embeddings", "POST")

    endpoints = ["/v1/chat/completions", "/v1/embeddings", "/v1/unknown"]
    for ep in endpoints:
        result = gw.process(ep)
        print(f"  {ep:<30s}  middleware: {', '.join(result['middleware'])}")
        print(f"  {'':30s}  endpoint:  {result['endpoint']}")

    # 1.2 Load balancer (weighted round-robin)
    print("\n--- 1.2 Weighted Round-Robin Load Balancer ---")

    class WeightedRoundRobin:
        def __init__(self, backends):
            self.backends = backends  # list of (name, weight)
            self.current_weights = [0] * len(backends)

        def next(self):
            total_weight = sum(w for _, w in self.backends)
            # Add effective weight
            for i, (_, w) in enumerate(self.backends):
                self.current_weights[i] += w

            # Select highest
            best = self.current_weights.index(max(self.current_weights))
            selected = self.backends[best][0]

            # Subtract total weight
            self.current_weights[best] -= total_weight
            return selected

    backends = [("llm-1 (A100)", 5), ("llm-2 (A100)", 5), ("llm-3 (V100)", 3)]
    lb = WeightedRoundRobin(backends)

    distribution = collections.Counter()
    for _ in range(13):
        selected = lb.next()
        distribution[selected] += 1

    print(f"  Backends: {[(n, w) for n, w in backends]}")
    print(f"  Distribution over 13 requests:")
    for name, count in distribution.most_common():
        bar = "#" * (count * 3)
        print(f"    {name:<20s}  {count:>2} requests  {bar}")

    # 1.3 Circuit breaker
    print("\n--- 1.3 Circuit Breaker ---")

    class CircuitBreaker:
        def __init__(self, failure_threshold=3, recovery_timeout=5):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failures = 0
            self.state = "CLOSED"
            self.last_failure_time = 0

        def call(self, success):
            now = time.time()
            if self.state == "OPEN":
                if now - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    print(f"    -> HALF_OPEN: attempting probe")
                else:
                    print(f"    -> OPEN: request blocked (circuit breaker)")
                    return False

            if success:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                    print(f"    -> CLOSED: probe succeeded, circuit recovered")
                elif self.state == "CLOSED":
                    self.failures = max(0, self.failures - 1)
                return True
            else:
                self.failures += 1
                self.last_failure_time = now
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    print(f"    -> OPEN: {self.failures} failures reached threshold")
                return False

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    outcomes = [False, False, False, True, True, False, False, True, True, True]
    print(f"  Threshold: {cb.failure_threshold} failures, Recovery: {cb.recovery_timeout}s")
    for i, success in enumerate(outcomes):
        result = cb.call(success)
        status = "SUCCESS" if result else "FAIL"
        print(f"  [{i+1:>2}] call={'OK' if success else 'ERR':<4s} -> {status}  state={cb.state}")

    # 1.4 Request queue with priority
    print("\n--- 1.4 Priority Request Queue ---")

    class PriorityLLMQueue:
        def __init__(self):
            self.queues = collections.defaultdict(list)

        def enqueue(self, request, priority=0):
            self.queues[priority].append(request)

        def dequeue(self):
            for p in sorted(self.queues.keys(), reverse=True):
                if self.queues[p]:
                    return self.queues[p].pop(0), p
            return None, None

        def size(self):
            return sum(len(q) for q in self.queues.values())

    pq = PriorityLLMQueue()
    pq.enqueue("batch_embed_10k", priority=0)
    pq.enqueue("user_chat_critical", priority=3)
    pq.enqueue("user_chat_normal", priority=1)
    pq.enqueue("batch_embed_5k", priority=0)
    pq.enqueue("user_chat_high", priority=2)

    print(f"  Queue size: {pq.size()}")
    print(f"  Processing order (priority -> request):")
    while pq.size() > 0:
        req, pri = pq.dequeue()
        print(f"    priority={pri}  -> {req}")

    print()


# ---------------------------------------------------------------------------
# Demo 2: A/B Testing
# ---------------------------------------------------------------------------
def demo_ab_testing():
    """Traffic splitting, statistical significance, winner selection."""
    print("=" * 70)
    print("Demo 2 — A/B Testing: traffic split, significance, winner")
    print("=" * 70)

    # 2.1 Traffic splitting
    print("\n--- 2.1 Traffic Splitting ---")

    class TrafficSplitter:
        def __init__(self, variants):
            """variants: list of (name, weight)"""
            self.variants = variants
            self.total_weight = sum(w for _, w in variants)

        def route(self, user_id):
            h = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
            bucket = h % self.total_weight
            cumulative = 0
            for name, weight in self.variants:
                cumulative += weight
                if bucket < cumulative:
                    return name
            return self.variants[-1][0]

    splitter = TrafficSplitter([("control", 50), ("variant_a", 30), ("variant_b", 20)])
    distribution = collections.Counter()
    for uid in range(1000):
        variant = splitter.route(uid)
        distribution[variant] += 1

    print(f"  Config: {splitter.variants}")
    print(f"  Result over 1000 users:")
    for name, count in distribution.most_common():
        actual_pct = count / 1000 * 100
        print(f"    {name:<12s}  {count:>4} users ({actual_pct:.1f}%)")

    # 2.2 Statistical significance (z-test for proportions)
    print("\n--- 2.2 Statistical Significance (Z-Test) ---")

    def proportion_z_test(successes_a, n_a, successes_b, n_b):
        """Two-proportion z-test. Returns z-stat and p-value."""
        p_a = successes_a / n_a
        p_b = successes_b / n_b
        p_pool = (successes_a + successes_b) / (n_a + n_b)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        if se == 0:
            return 0, 1.0
        z = (p_b - p_a) / se
        # Two-tailed p-value approximation
        p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return z, p_val

    # Simulated A/B test results
    random.seed(42)
    n_per_group = 500
    control_conversions = sum(random.random() < 0.12 for _ in range(n_per_group))
    variant_conversions = sum(random.random() < 0.15 for _ in range(n_per_group))

    z, p_val = proportion_z_test(control_conversions, n_per_group,
                                  variant_conversions, n_per_group)
    alpha = 0.05

    print(f"  Control:  {control_conversions}/{n_per_group} conversions ({control_conversions/n_per_group*100:.1f}%)")
    print(f"  Variant:  {variant_conversions}/{n_per_group} conversions ({variant_conversions/n_per_group*100:.1f}%)")
    print(f"  Z-statistic: {z:.4f}")
    print(f"  P-value: {p_val:.4f}")
    print(f"  Alpha: {alpha}")
    print(f"  -> {'SIGNIFICANT: variant is better' if p_val < alpha and z > 0 else 'NOT significant: no winner yet'}")

    # 2.3 Sample size calculator
    print("\n--- 2.3 Minimum Sample Size Calculator ---")

    def min_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
        """Minimum sample size per variant for a two-proportion z-test."""
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2
        # Z-values for common alpha/power
        z_alpha = 1.96   # two-sided alpha=0.05
        z_beta = 0.842   # power=0.80
        n = ((z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
              z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde ** 2)
        return math.ceil(n)

    for baseline, mde in [(0.10, 0.02), (0.10, 0.05), (0.20, 0.03), (0.05, 0.01)]:
        n = min_sample_size(baseline, mde)
        print(f"  baseline={baseline:.0%}, MDE={mde:.0%}  ->  n={n:>5} per variant (total {2*n})")

    # 2.4 Sequential testing (early stopping)
    print("\n--- 2.4 Sequential Testing (Early Stopping Rules) ---")

    def sequential_test(control_data, variant_data, alpha=0.05):
        """O'Brien-Fleming style: check significance at interim looks."""
        n = min(len(control_data), len(variant_data))
        look_points = [n // 4, n // 2, 3 * n // 4, n]
        alpha_spent = 0
        alpha_per_look = alpha / len(look_points)  # Bonferroni-like

        for look in look_points:
            c_data = control_data[:look]
            v_data = variant_data[:look]
            c_conv = sum(c_data)
            v_conv = sum(v_data)
            z, p = proportion_z_test(c_conv, look, v_conv, look)
            reject = p < alpha_per_look
            print(f"    Look at n={look:>4}: ctrl={c_conv}/{look} ({c_conv/look:.1%}), "
                  f"var={v_conv}/{look} ({v_conv/look:.1%}), p={p:.4f} "
                  f"{'-> STOP: significant' if reject else ''}")
            if reject:
                return True, look
        return False, n

    random.seed(42)
    ctrl_data = [1 if random.random() < 0.12 else 0 for _ in range(200)]
    var_data = [1 if random.random() < 0.18 else 0 for _ in range(200)]
    stopped, at_n = sequential_test(ctrl_data, var_data)
    print(f"  Result: {'Early stop at n=' + str(at_n) if stopped else 'Full sample needed (n=' + str(at_n) + ')'}")

    print()


# ---------------------------------------------------------------------------
# Demo 3: Incident Response
# ---------------------------------------------------------------------------
def demo_incident_response():
    """Rollback procedures, escalation, post-mortem."""
    print("=" * 70)
    print("Demo 3 — Incident Response: rollback, escalation, post-mortem")
    print("=" * 70)

    # 3.1 Version management and rollback
    print("\n--- 3.1 Version Management & Rollback ---")

    class ModelRegistry:
        def __init__(self):
            self.deployments = []
            self.active = None

        def deploy(self, model_name, version, config):
            entry = {
                "model": model_name,
                "version": version,
                "config": config,
                "deployed_at": datetime.datetime.now().isoformat(),
                "status": "active",
            }
            self.deployments.append(entry)
            self.active = entry
            return entry

        def rollback(self):
            if len(self.deployments) < 2:
                return None
            self.active["status"] = "rolled_back"
            prev = self.deployments[-2]
            prev["status"] = "restored"
            self.active = prev
            return prev

        def list_versions(self):
            return self.deployments

    registry = ModelRegistry()
    registry.deploy("gpt-4-finetuned", "v1.0", {"temperature": 0.7, "max_tokens": 1000})
    registry.deploy("gpt-4-finetuned", "v1.1", {"temperature": 0.3, "max_tokens": 2000})
    registry.deploy("gpt-4-finetuned", "v2.0", {"temperature": 0.9, "max_tokens": 500})

    print("  Deployment history:")
    for d in registry.list_versions():
        status_marker = " <-- ACTIVE" if d == registry.active else ""
        print(f"    {d['version']:<8s}  {d['config']}  [{d['status']}]{status_marker}")

    print(f"\n  Current active: {registry.active['version']}")
    rolled = registry.rollback()
    print(f"  After rollback: {registry.active['version']} (restored)")
    print(f"  Rollback config: {rolled['config']}")

    # 3.2 Incident timeline
    print("\n--- 3.2 Incident Timeline ---")

    class Incident:
        def __init__(self, title, severity):
            self.title = title
            self.severity = severity
            self.timeline = []
            self.status = "open"

        def add_event(self, time_str, event):
            self.timeline.append({"time": time_str, "event": event})

        def close(self, resolution):
            self.status = "resolved"
            self.add_event("now", f"RESOLVED: {resolution}")

    inc = Incident("LLM API returning 500 errors", "P1")
    inc.add_event("14:00", "Alert triggered: error rate > 5%")
    inc.add_event("14:02", "Oncall engineer paged")
    inc.add_event("14:05", "Root cause identified: model endpoint OOM")
    inc.add_event("14:07", "Rollback to v1.1 initiated")
    inc.add_event("14:10", "Rollback complete, error rate dropping")
    inc.close("v1.1 restored, error rate back to normal")

    print(f"  Incident: {inc.title} [{inc.severity}]")
    for event in inc.timeline:
        print(f"    {event['time']:<8s}  {event['event']}")

    # 3.3 Escalation matrix
    print("\n--- 3.3 Escalation Matrix ---")

    escalation_matrix = [
        {"level": "L1", "severity": "P3/P4", "response": "next business day", "team": "oncall eng"},
        {"level": "L2", "severity": "P2", "response": "30 min", "team": "senior oncall + eng lead"},
        {"level": "L3", "severity": "P1", "response": "5 min", "team": "CTO + eng lead + oncall"},
        {"level": "L4", "severity": "P0 (data loss)", "response": "immediate", "team": "exec team + legal"},
    ]

    print(f"  {'Level':<8s}  {'Severity':<20s}  {'Response Time':<20s}  {'Team'}")
    print(f"  {'-'*8}  {'-'*20}  {'-'*20}  {'-'*35}")
    for level in escalation_matrix:
        print(f"  {level['level']:<8s}  {level['severity']:<20s}  {level['response']:<20s}  {level['team']}")

    # 3.4 Post-mortem template
    print("\n--- 3.4 Post-Mortem Summary ---")

    postmortem = {
        "title": "LLM API v2.0 OOM causing 500 errors",
        "severity": "P1",
        "duration_min": 10,
        "impact": "~500 requests failed, ~200 users affected",
        "timeline": [
            "14:00 - Error rate spike detected",
            "14:07 - Rollback initiated",
            "14:10 - Service restored",
        ],
        "root_cause": "v2.0 config set max_tokens=2000 causing GPU OOM under load",
        "action_items": [
            "Add OOM guard in deployment pipeline",
            "Load test all config changes before production deploy",
            "Implement automatic rollback on error rate > 10%",
        ],
    }

    print(f"  Title: {postmortem['title']}")
    print(f"  Severity: {postmortem['severity']}, Duration: {postmortem['duration_min']} min")
    print(f"  Impact: {postmortem['impact']}")
    print(f"  Root cause: {postmortem['root_cause']}")
    print(f"  Action items:")
    for i, item in enumerate(postmortem['action_items'], 1):
        print(f"    {i}. {item}")

    print()


# ---------------------------------------------------------------------------
# Demo 4: Quality Monitoring
# ---------------------------------------------------------------------------
def demo_quality_monitoring():
    """Drift detection, user feedback, continuous evaluation."""
    print("=" * 70)
    print("Demo 4 — Quality Monitoring: drift, feedback, continuous eval")
    print("=" * 70)

    # 4.1 Response quality scoring
    print("\n--- 4.1 Response Quality Scoring ---")

    def quality_score(response_len, has_structure, relevance_keywords, hallucination_flags):
        """Heuristic quality score 0-100."""
        len_score = min(30, response_len * 0.03)     # up to 30 pts for length
        struct_score = min(20, has_structure * 10)     # up to 20 pts for structure
        rel_score = min(30, len(relevance_keywords) * 6)  # up to 30 pts
        hall_penalty = hallucination_flags * 15        # -15 per flag
        return max(0, min(100, len_score + struct_score + rel_score - hall_penalty))

    responses = [
        {"len": 200, "structure": 1, "relevance": ["key", "concept", "example"], "halluc": 0},
        {"len": 50, "structure": 0, "relevance": [], "halluc": 1},
        {"len": 500, "structure": 2, "relevance": ["a", "b", "c", "d", "e"], "halluc": 0},
        {"len": 150, "structure": 1, "relevance": ["keyword"], "halluc": 2},
    ]

    for i, r in enumerate(responses):
        score = quality_score(r["len"], r["structure"], r["relevance"], r["halluc"])
        print(f"  Response {i+1}: len={r['len']}, struct={r['structure']}, "
              f"relevance={len(r['relevance'])}, halluc={r['halluc']}  -> score={score:.0f}/100")

    # 4.2 User feedback aggregation
    print("\n--- 4.2 User Feedback Aggregation ---")

    class FeedbackTracker:
        def __init__(self):
            self.ratings = []
            self.comments = []

        def add(self, rating, comment=""):
            self.ratings.append(rating)
            if comment:
                self.comments.append(comment)

        def summary(self):
            if not self.ratings:
                return {}
            return {
                "count": len(self.ratings),
                "mean": round(statistics.mean(self.ratings), 2),
                "median": statistics.median(self.ratings),
                "std": round(statistics.stdev(self.ratings), 2) if len(self.ratings) > 1 else 0,
                "distribution": dict(collections.Counter(self.ratings)),
                "satisfaction_pct": round(sum(1 for r in self.ratings if r >= 4) / len(self.ratings) * 100, 1),
            }

    random.seed(42)
    ft = FeedbackTracker()
    for _ in range(100):
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
        ft.add(rating)

    s = ft.summary()
    print(f"  Total feedback: {s['count']}")
    print(f"  Mean rating: {s['mean']}/5  (std={s['std']})")
    print(f"  Satisfaction (>=4): {s['satisfaction_pct']}%")
    print(f"  Distribution:")
    for rating in sorted(s["distribution"].keys()):
        count = s["distribution"][rating]
        bar = "#" * count
        print(f"    {rating} stars: {count:>3}  {bar}")

    # 4.3 Continuous evaluation pipeline
    print("\n--- 4.3 Continuous Evaluation Pipeline ---")

    class EvalPipeline:
        def __init__(self):
            self.metrics = collections.defaultdict(list)

        def evaluate(self, model_version, test_cases):
            results = {"correct": 0, "total": len(test_cases)}
            for input_text, expected, got in test_cases:
                # Simple exact/contains match
                if expected.lower() in got.lower():
                    results["correct"] += 1

            accuracy = results["correct"] / results["total"] * 100
            self.metrics[model_version].append(accuracy)
            return accuracy

        def trend(self, model_version):
            data = self.metrics.get(model_version, [])
            if len(data) < 2:
                return "stable"
            recent = statistics.mean(data[-3:])
            earlier = statistics.mean(data[:3])
            diff = recent - earlier
            if diff > 2:
                return f"improving (+{diff:.1f})"
            elif diff < -2:
                return f"degrading ({diff:.1f})"
            return "stable"

    pipeline = EvalPipeline()

    # Simulate evaluation runs
    test_cases = [
        ("capital of France", "paris", "Paris is the capital of France"),
        ("2+2", "4", "2+2 equals 4"),
        ("largest planet", "jupiter", "Jupiter is the largest planet"),
        ("boiling point of water", "100", "Water boils at 100 degrees Celsius"),
    ]

    print(f"  Model: gpt-4-finetuned, Running 5 evaluation rounds:")
    for run in range(5):
        # Simulate slight variation in accuracy
        base_acc = 75 + run * 2 + random.gauss(0, 3)
        accuracy = max(50, min(100, base_acc))
        pipeline.metrics["v2.0"].append(accuracy)
        bar = "#" * int(accuracy / 2)
        print(f"    Run {run+1}: accuracy={accuracy:.1f}%  {bar}")

    trend = pipeline.trend("v2.0")
    print(f"  Trend: {trend}")

    # 4.4 SLA monitoring
    print("\n--- 4.4 SLA Monitoring Dashboard ---")

    sla_targets = {
        "availability": {"target": 99.9, "actual": 99.95, "unit": "%"},
        "latency_p50": {"target": 200, "actual": 150, "unit": "ms", "lower_better": True},
        "latency_p99": {"target": 2000, "actual": 1800, "unit": "ms", "lower_better": True},
        "error_rate": {"target": 1.0, "actual": 0.3, "unit": "%", "lower_better": True},
        "throughput": {"target": 100, "actual": 120, "unit": "req/s"},
    }

    print(f"  {'Metric':<20s}  {'Target':>8s}  {'Actual':>8s}  {'Status':>10s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}")
    for metric, data in sla_targets.items():
        lower_better = data.get("lower_better", False)
        if lower_better:
            met = data["actual"] <= data["target"]
        else:
            met = data["actual"] >= data["target"]
        status = "MET" if met else "BREACHED"
        target_str = f"{data['target']}{data['unit']}"
        actual_str = f"{data['actual']}{data['unit']}"
        print(f"  {metric:<20s}  {target_str:>8s}  {actual_str:>8s}  {status:>10s}")

    # 4.5 Alert aggregation for quality
    print("\n--- 4.5 Quality Alert Aggregation ---")

    quality_alerts = [
        {"metric": "response_quality", "value": 72, "threshold": 80, "direction": "below"},
        {"metric": "user_satisfaction", "value": 3.8, "threshold": 4.0, "direction": "below"},
        {"metric": "hallucination_rate", "value": 0.08, "threshold": 0.05, "direction": "above"},
        {"metric": "response_quality", "value": 75, "threshold": 80, "direction": "below"},
        {"metric": "hallucination_rate", "value": 0.09, "threshold": 0.05, "direction": "above"},
    ]

    # Group by metric
    grouped = collections.defaultdict(list)
    for a in quality_alerts:
        grouped[a["metric"]].append(a)

    print(f"  Total quality alerts: {len(quality_alerts)}")
    print(f"  Unique metrics affected: {len(grouped)}")
    for metric, alerts in grouped.items():
        values = [a["value"] for a in alerts]
        print(f"    {metric:<25s}  {len(alerts)} alerts, values: {values}")

    # Overall quality health
    all_met = all(
        (a["value"] <= a["threshold"] if a["direction"] == "below" else a["value"] >= a["threshold"])
        for a in quality_alerts
    )
    print(f"  Overall quality health: {'ALL METRICS PASSING' if all_met else 'ATTENTION NEEDED'}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_production_architecture()
    demo_ab_testing()
    demo_incident_response()
    demo_quality_monitoring()
