"""123 — LLM Monitoring: наблюдаемость, логирование, обнаружение дрейфа

Темы:
  1. Logging Patterns (structured logs, request tracing, latency tracking)
  2. Metrics Collection (token usage, latency percentiles, error rates)
  3. Drift Detection (data drift, concept drift, statistical tests)
  4. Alerting Rules (threshold-based, anomaly detection, escalation)

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
# Demo 1: Logging Patterns
# ---------------------------------------------------------------------------
def demo_logging_patterns():
    """Structured logs, request tracing, latency tracking."""
    print("=" * 70)
    print("Demo 1 — Logging Patterns: structured logs, tracing, latency")
    print("=" * 70)

    # 1.1 Structured JSON log entry
    print("\n--- 1.1 Structured JSON Log Entry ---")

    def make_log_entry(level, message, **extra):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message,
            "service": "llm-gateway",
            "trace_id": hashlib.md5(message.encode()).hexdigest()[:12],
        }
        entry.update(extra)
        return entry

    log = make_log_entry("INFO", "request completed", model="gpt-4", tokens=342, latency_ms=1823)
    print(json.dumps(log, indent=2))

    # 1.2 Request tracing — chain of spans
    print("\n--- 1.2 Request Tracing (Span Chain) ---")

    class Span:
        def __init__(self, name, parent_id=None):
            self.span_id = hashlib.md5(name.encode()).hexdigest()[:8]
            self.parent_id = parent_id
            self.name = name
            self.start = time.time()
            self.end = None
            self.attributes = {}

        def finish(self):
            self.end = time.time()

        @property
        def duration_ms(self):
            return round((self.end - self.start) * 1000, 1) if self.end else 0

    root = Span("http POST /chat")
    token_span = Span("tokenize", parent_id=root.span_id)
    token_span.start -= 0.012
    token_span.finish()
    llm_span = Span("llm.generate", parent_id=root.span_id)
    llm_span.start = token_span.end
    llm_span.finish()
    root.end = llm_span.end

    for sp in [root, token_span, llm_span]:
        parent_info = f" -> parent={sp.parent_id}" if sp.parent_id else ""
        print(f"  span={sp.name:<20s}  id={sp.span_id}  dur={sp.duration_ms}ms{parent_info}")

    # 1.3 Latency tracking with rolling window
    print("\n--- 1.3 Rolling Latency Window ---")

    class LatencyTracker:
        def __init__(self, window=10):
            self.window = window
            self.latencies = collections.deque(maxlen=window)

        def record(self, latency_ms):
            self.latencies.append(latency_ms)

        def stats(self):
            if not self.latencies:
                return {}
            data = sorted(self.latencies)
            return {
                "count": len(data),
                "mean": round(statistics.mean(data), 1),
                "p50": round(statistics.median(data), 1),
                "p95": round(data[int(len(data) * 0.95)], 1),
                "max": max(data),
            }

    tracker = LatencyTracker(window=8)
    sample_latencies = [120, 135, 98, 210, 145, 300, 110, 125, 180, 95]
    for i, lat in enumerate(sample_latencies):
        tracker.record(lat)
        if (i + 1) % 4 == 0:
            s = tracker.stats()
            print(f"  After {i+1:>2} requests: mean={s['mean']}ms  p50={s['p50']}ms  p95={s['p95']}ms")

    # 1.4 Log aggregation and deduplication
    print("\n--- 1.4 Log Deduplication (Hash-Based) ---")

    seen_hashes = set()
    raw_logs = [
        {"msg": "rate limit hit", "code": 429},
        {"msg": "rate limit hit", "code": 429},
        {"msg": "timeout", "code": 504},
        {"msg": "rate limit hit", "code": 429},
        {"msg": "timeout", "code": 504},
        {"msg": "new error: auth failed", "code": 401},
    ]
    unique_count = 0
    for log_entry in raw_logs:
        h = hashlib.md5(json.dumps(log_entry, sort_keys=True).encode()).hexdigest()[:10]
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_count += 1
            print(f"  NEW  [{log_entry['code']}] {log_entry['msg']}")
        else:
            print(f"  DUP  [{log_entry['code']}] {log_entry['msg']}")
    print(f"  Unique: {unique_count}/{len(raw_logs)}  (suppressed {len(raw_logs) - unique_count} duplicates)")

    print()


# ---------------------------------------------------------------------------
# Demo 2: Metrics Collection
# ---------------------------------------------------------------------------
def demo_metrics_collection():
    """Token usage, latency percentiles, error rates."""
    print("=" * 70)
    print("Demo 2 — Metrics Collection: tokens, percentiles, error rates")
    print("=" * 70)

    # 2.1 Token usage tracking per model
    print("\n--- 2.1 Token Usage per Model ---")

    class TokenMeter:
        def __init__(self):
            self.data = collections.defaultdict(lambda: {"prompt": 0, "completion": 0, "requests": 0})

        def log(self, model, prompt_tokens, completion_tokens):
            self.data[model]["prompt"] += prompt_tokens
            self.data[model]["completion"] += completion_tokens
            self.data[model]["requests"] += 1

        def summary(self):
            result = {}
            for model, d in self.data.items():
                total = d["prompt"] + d["completion"]
                result[model] = {
                    "requests": d["requests"],
                    "total_tokens": total,
                    "avg_per_req": round(total / d["requests"], 1),
                    "prompt_ratio": round(d["prompt"] / total * 100, 1),
                }
            return result

    meter = TokenMeter()
    for _ in range(20):
        model = random.choice(["gpt-4", "gpt-3.5-turbo", "claude-3"])
        prompt_t = random.randint(50, 500)
        comp_t = random.randint(20, 300)
        meter.log(model, prompt_t, comp_t)

    for model, stats in meter.summary().items():
        print(f"  {model:<18s}  reqs={stats['requests']:>3}  tokens={stats['total_tokens']:>6}  "
              f"avg={stats['avg_per_req']:>6.1f}  prompt%={stats['prompt_ratio']:.1f}%")

    # 2.2 Percentile computation (without numpy)
    print("\n--- 2.2 Latency Percentiles (Pure Python) ---")

    def percentile(data, p):
        """Linear interpolation percentile."""
        sorted_d = sorted(data)
        k = (len(sorted_d) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_d[int(k)]
        return sorted_d[f] * (c - k) + sorted_d[c] * (k - f)

    latencies = [random.gauss(150, 40) for _ in range(200)]
    latencies = [max(10, l) for l in latencies]  # clamp negatives

    for p in [50, 75, 90, 95, 99]:
        val = percentile(latencies, p)
        print(f"  p{p:<3d} = {val:>7.1f} ms")

    # 2.3 Error rate over time windows
    print("\n--- 2.3 Error Rate (5-Minute Windows) ---")

    random.seed(42)
    windows = []
    for w in range(6):
        total_req = random.randint(80, 120)
        errors = random.choices([0, 1], weights=[0.95, 0.05], k=total_req).count(1)
        rate = errors / total_req * 100
        windows.append({"window": w, "total": total_req, "errors": errors, "rate": round(rate, 2)})

    for w in windows:
        bar = "#" * int(w["rate"] * 3)
        print(f"  W{w['window']}: {w['total']:>3} reqs, {w['errors']:>2} err, rate={w['rate']:>5.2f}%  {bar}")

    # 2.4 Composite health score
    print("\n--- 2.4 Composite LLM Health Score ---")

    def health_score(error_rate_pct, avg_latency_ms, p99_latency_ms, token_budget_pct):
        # Each metric contributes 0-100, weighted
        err_score = max(0, 100 - error_rate_pct * 100)        # 0% error = 100
        lat_score = max(0, 100 - (avg_latency_ms - 100) / 4)  # 100ms = 100, 500ms = 0
        p99_score = max(0, 100 - (p99_latency_ms - 200) / 8)
        bud_score = max(0, 100 - (token_budget_pct - 80) * 5)
        score = 0.4 * err_score + 0.3 * lat_score + 0.15 * p99_score + 0.15 * bud_score
        return round(score, 1)

    configs = [
        {"name": "healthy", "err": 0.01, "lat": 150, "p99": 350, "bud": 60},
        {"name": "degraded", "err": 0.05, "lat": 300, "p99": 800, "bud": 85},
        {"name": "critical", "err": 0.15, "lat": 500, "p99": 1500, "bud": 98},
    ]
    for c in configs:
        s = health_score(c["err"], c["lat"], c["p99"], c["bud"])
        status = "HEALTHY" if s > 70 else ("WARNING" if s > 40 else "CRITICAL")
        print(f"  {c['name']:<12s}  score={s:>5.1f}  -> {status}")

    print()


# ---------------------------------------------------------------------------
# Demo 3: Drift Detection
# ---------------------------------------------------------------------------
def demo_drift_detection():
    """Data drift, concept drift, statistical tests."""
    print("=" * 70)
    print("Demo 3 — Drift Detection: data drift, concept drift, stats tests")
    print("=" * 70)

    # 3.1 Distribution comparison via KL divergence
    print("\n--- 3.1 KL Divergence for Token Length Distribution ---")

    def kl_divergence(p, q, bins=10):
        """Approximate KL(p || q) over binned histograms."""
        min_val = min(min(p), min(q))
        max_val = max(max(p), max(q))
        edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]

        def hist(data):
            counts = [0] * bins
            for v in data:
                for j in range(bins):
                    if edges[j] <= v < edges[j + 1]:
                        counts[j] += 1
                        break
                else:
                    counts[-1] += 1
            total = max(sum(counts), 1)
            return [c / total + 1e-10 for c in counts]

        p_hist = hist(p)
        q_hist = hist(q)
        return sum(pi * math.log(pi / qi) for pi, qi in zip(p_hist, q_hist))

    baseline_lengths = [random.gauss(150, 30) for _ in range(200)]
    current_lengths = [random.gauss(165, 35) for _ in range(200)]  # slightly shifted
    drifted_lengths = [random.gauss(200, 50) for _ in range(200)]  # significantly shifted

    kl_1 = kl_divergence(baseline_lengths, current_lengths)
    kl_2 = kl_divergence(baseline_lengths, drifted_lengths)
    print(f"  KL(baseline || current)   = {kl_1:.4f}  (small drift)")
    print(f"  KL(baseline || drifted)   = {kl_2:.4f}  (significant drift)")
    threshold = 0.05
    print(f"  Threshold: {threshold}  -> current={'DRIFT' if kl_1 > threshold else 'OK'}  "
          f"drifted={'DRIFT' if kl_2 > threshold else 'OK'}")

    # 3.2 PSI (Population Stability Index)
    print("\n--- 3.2 Population Stability Index (PSI) ---")

    def psi(reference, current, bins=10):
        min_val = min(min(reference), min(current))
        max_val = max(max(reference), max(current))
        edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]

        def proportions(data):
            counts = [0] * bins
            for v in data:
                for j in range(bins):
                    if edges[j] <= v < edges[j + 1]:
                        counts[j] += 1
                        break
                else:
                    counts[-1] += 1
            total = max(len(data), 1)
            return [max(c / total, 0.001) for c in counts]

        ref_p = proportions(reference)
        cur_p = proportions(current)
        return sum((cur - ref) * math.log(cur / ref) for ref, cur in zip(ref_p, cur_p))

    ref_data = [random.gauss(50, 10) for _ in range(500)]
    small_shift = [random.gauss(52, 11) for _ in range(500)]
    big_shift = [random.gauss(65, 15) for _ in range(500)]

    psi_small = psi(ref_data, small_shift)
    psi_big = psi(ref_data, big_shift)
    print(f"  PSI (small shift)  = {psi_small:.4f}  {'< 0.1 -> no drift' if psi_small < 0.1 else '> 0.1 -> drift detected'}")
    print(f"  PSI (large shift)  = {psi_big:.4f}  {'< 0.1 -> no drift' if psi_big < 0.1 else '> 0.2 -> significant drift'}")

    # 3.3 Concept drift via performance window comparison
    print("\n--- 3.3 Concept Drift (Rolling Performance) ---")

    def rolling_accuracy(n=60):
        """Simulate accuracy that degrades mid-stream (concept drift)."""
        accs = []
        for i in range(n):
            if i < 30:
                accs.append(random.gauss(0.88, 0.03))
            else:
                # gradual degradation
                degradation = (i - 30) * 0.008
                accs.append(random.gauss(0.88 - degradation, 0.03))
        return [max(0, min(1, a)) for a in accs]

    accs = rolling_accuracy()
    window = 10
    print(f"  Window-based accuracy (window={window}):")
    for i in range(0, len(accs), window):
        chunk = accs[i:i + window]
        avg = statistics.mean(chunk)
        bar = "#" * int(avg * 40)
        print(f"    t={i:>2}-{i+len(chunk)-1:<2}  acc={avg:.3f}  {bar}")

    # Detect drift: compare first window vs last window
    first_w = accs[:window]
    last_w = accs[-window:]
    diff = statistics.mean(first_w) - statistics.mean(last_w)
    print(f"  Drift magnitude (first - last window): {diff:.4f}")
    print(f"  -> {'CONCEPT DRIFT DETECTED' if diff > 0.05 else 'No significant drift'}")

    # 3.4 Chi-squared test for categorical drift
    print("\n--- 3.4 Chi-Squared Test (Categorical Drift) ---")

    def chi_squared(observed, expected):
        return sum((o - e) ** 2 / e for o, e in zip(observed, expected))

    def chi_squared_p_value(chi2, df):
        """Approximate p-value via Wilson-Hilferty (df >= 1)."""
        if df <= 0:
            return 1.0
        z = (chi2 / df) ** (1/3) - (1 - 2 / (9 * df))
        z /= math.sqrt(2 / (9 * df))
        # Standard normal CDF approximation
        return 0.5 * (1 + math.erf(-z / math.sqrt(2)))

    # Baseline model usage distribution
    baseline = {"gpt-4": 400, "gpt-3.5": 350, "claude": 250}
    # Current month — shift toward cheaper models
    current = {"gpt-4": 300, "gpt-3.5": 450, "claude": 250}

    obs = list(current.values())
    exp = list(baseline.values())
    chi2 = chi_squared(obs, exp)
    df = len(obs) - 1
    p = chi_squared_p_value(chi2, df)
    print(f"  Baseline: {baseline}")
    print(f"  Current:  {current}")
    print(f"  Chi-squared = {chi2:.2f}, df = {df}, p-value = {p:.4f}")
    print(f"  -> {'Distribution shift DETECTED (p < 0.05)' if p < 0.05 else 'No significant shift'}")

    print()


# ---------------------------------------------------------------------------
# Demo 4: Alerting Rules
# ---------------------------------------------------------------------------
def demo_alerting_rules():
    """Threshold-based, anomaly detection, escalation."""
    print("=" * 70)
    print("Demo 4 — Alerting Rules: thresholds, anomalies, escalation")
    print("=" * 70)

    # 4.1 Threshold-based alerting
    print("\n--- 4.1 Threshold-Based Alerting ---")

    class AlertRule:
        def __init__(self, name, metric, threshold, direction="above", severity="warning"):
            self.name = name
            self.metric = metric
            self.threshold = threshold
            self.direction = direction
            self.severity = severity

        def check(self, value):
            if self.direction == "above" and value > self.threshold:
                return self.severity
            elif self.direction == "below" and value < self.threshold:
                return self.severity
            return None

    rules = [
        AlertRule("high_error_rate", "error_rate", 0.05, "above", "critical"),
        AlertRule("high_latency_p99", "p99_latency_ms", 2000, "above", "warning"),
        AlertRule("low_throughput", "req_per_sec", 10, "below", "warning"),
        AlertRule("budget_exceeded", "daily_cost_usd", 500, "above", "critical"),
    ]

    # Simulated current metrics
    current_metrics = {
        "error_rate": 0.07,
        "p99_latency_ms": 1800,
        "req_per_sec": 25,
        "daily_cost_usd": 520,
    }

    for rule in rules:
        val = current_metrics.get(rule.metric, 0)
        result = rule.check(val)
        status = f"ALERT [{result.upper()}]" if result else "OK"
        print(f"  {rule.name:<25s}  {rule.metric}={val:<10}  threshold={rule.threshold}  -> {status}")

    # 4.2 Anomaly detection via Z-score
    print("\n--- 4.2 Anomaly Detection (Z-Score) ---")

    def detect_anomalies(data, window=10, z_thresh=2.0):
        anomalies = []
        for i in range(window, len(data)):
            hist = data[i - window:i]
            mean = statistics.mean(hist)
            stdev = statistics.stdev(hist) if len(hist) > 1 else 1e-10
            z = (data[i] - mean) / stdev
            if abs(z) > z_thresh:
                anomalies.append((i, data[i], round(z, 2)))
        return anomalies

    # Simulate request latency with injected anomalies
    random.seed(42)
    lat_stream = [random.gauss(150, 20) for _ in range(50)]
    lat_stream[20] = 450  # spike
    lat_stream[35] = 40   # dip
    lat_stream[42] = 500  # spike

    anomalies = detect_anomalies(lat_stream, window=10, z_thresh=2.0)
    print(f"  Stream length: {len(lat_stream)}")
    print(f"  Anomalies found: {len(anomalies)}")
    for idx, val, z in anomalies:
        print(f"    index={idx:>2}  value={val:>6.1f}ms  z-score={z:>+.2f}")

    # 4.3 Escalation policy
    print("\n--- 4.3 Escalation Policy ---")

    class EscalationPolicy:
        def __init__(self):
            self.levels = [
                {"severity": "info", "notify": "slack #ops", "timeout_min": 0},
                {"severity": "warning", "notify": "slack #ops + oncall pager", "timeout_min": 5},
                {"severity": "critical", "notify": "pager + phone call + slack #incidents", "timeout_min": 1},
            ]

        def route(self, severity):
            for level in self.levels:
                if level["severity"] == severity:
                    return level
            return self.levels[0]

    policy = EscalationPolicy()
    test_alerts = ["info", "warning", "critical"]
    for sev in test_alerts:
        route = policy.route(sev)
        print(f"  {sev.upper():<10s} -> notify: {route['notify']:<45s}  ack timeout: {route['timeout_min']}min")

    # 4.4 Alert correlation (grouping related alerts)
    print("\n--- 4.4 Alert Correlation & Grouping ---")

    alerts_raw = [
        {"time": "10:01", "rule": "high_error_rate", "service": "llm-api"},
        {"time": "10:01", "rule": "high_latency_p99", "service": "llm-api"},
        {"time": "10:02", "rule": "high_error_rate", "service": "llm-api"},
        {"time": "10:05", "rule": "low_throughput", "service": "tokenizer"},
        {"time": "10:05", "rule": "high_error_rate", "service": "tokenizer"},
    ]

    # Group by service
    grouped = collections.defaultdict(list)
    for a in alerts_raw:
        grouped[a["service"]].append(a)

    print(f"  Raw alerts: {len(alerts_raw)}")
    print(f"  Correlated groups: {len(grouped)}")
    for svc, alerts in grouped.items():
        rules_set = list({a["rule"] for a in alerts})
        print(f"    [{svc}] {len(alerts)} alerts, {len(rules_set)} unique rules: {', '.join(rules_set)}")

    # 4.5 Cool-down (suppress repeat alerts)
    print("\n--- 4.5 Cool-Down Suppression ---")

    class CoolDownSuppressor:
        def __init__(self, cooldown_sec=60):
            self.cooldown = cooldown_sec
            self.last_fired = {}

        def should_fire(self, rule_key, now_sec):
            last = self.last_fired.get(rule_key, -float("inf"))
            if now_sec - last >= self.cooldown:
                self.last_fired[rule_key] = now_sec
                return True
            return False

    suppressor = CoolDownSuppressor(cooldown_sec=30)
    events = [
        (0, "high_latency"), (5, "high_latency"), (10, "high_latency"),
        (35, "high_latency"), (40, "high_latency"),
        (0, "high_errors"), (15, "high_errors"),
    ]
    for t, rule in events:
        fired = suppressor.should_fire(rule, t)
        print(f"  t={t:>3}s  rule={rule:<15s}  -> {'FIRED' if fired else 'suppressed (cooldown)'}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_logging_patterns()
    demo_metrics_collection()
    demo_drift_detection()
    demo_alerting_rules()
