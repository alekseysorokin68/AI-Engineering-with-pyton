"""124 — Cost Optimization: подсчёт токенов, кэширование, маршрутизация

Темы:
  1. Token Cost Analysis (pricing models, cost estimation, budget tracking)
  2. Caching Strategies (semantic cache, exact match, prefix cache)
  3. Model Routing (complexity detection, model selection, fallback chains)
  4. Distillation (knowledge transfer, student-teacher, size reduction)

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
# Demo 1: Token Cost Analysis
# ---------------------------------------------------------------------------
def demo_token_cost_analysis():
    """Pricing models, cost estimation, budget tracking."""
    print("=" * 70)
    print("Demo 1 — Token Cost Analysis: pricing, estimation, budgets")
    print("=" * 70)

    # 1.1 Pricing model definition and comparison
    print("\n--- 1.1 Pricing Model Comparison ---")

    pricing = {
        "gpt-4":         {"input": 0.030, "output": 0.060},
        "gpt-4-turbo":   {"input": 0.010, "output": 0.030},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }

    print(f"  {'Model':<20s}  {'Input $/1K tok':>14s}  {'Output $/1K tok':>14s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}")
    for model, prices in pricing.items():
        print(f"  {model:<20s}  ${prices['input']:>12.4f}  ${prices['output']:>12.4f}")

    # 1.2 Cost estimation for a conversation
    print("\n--- 1.2 Cost Estimation per Request ---")

    def estimate_cost(model, prompt_tokens, completion_tokens):
        p = pricing[model]
        input_cost = (prompt_tokens / 1000) * p["input"]
        output_cost = (completion_tokens / 1000) * p["output"]
        return input_cost + output_cost

    requests = [
        ("gpt-4", 1200, 300),
        ("gpt-3.5-turbo", 1200, 300),
        ("claude-3-sonnet", 1200, 300),
    ]
    for model, pt, ct in requests:
        cost = estimate_cost(model, pt, ct)
        print(f"  {model:<20s}  prompt={pt:>4}  completion={ct:>4}  cost=${cost:.6f}")

    # Savings calculation
    gpt4_cost = estimate_cost("gpt-4", 1200, 300)
    gpt35_cost = estimate_cost("gpt-3.5-turbo", 1200, 300)
    savings_pct = (1 - gpt35_cost / gpt4_cost) * 100
    print(f"  -> Switching gpt-4 -> gpt-3.5-turbo saves {savings_pct:.1f}% per request")

    # 1.3 Budget tracking with daily limits
    print("\n--- 1.3 Daily Budget Tracker ---")

    class BudgetTracker:
        def __init__(self, daily_limit_usd):
            self.daily_limit = daily_limit_usd
            self.spent = 0.0
            self.history = []

        def charge(self, model, prompt_t, completion_t, timestamp=None):
            cost = estimate_cost(model, prompt_t, completion_t)
            self.spent += cost
            self.history.append({"model": model, "cost": cost, "cumulative": self.spent})
            return cost

        def remaining(self):
            return max(0, self.daily_limit - self.spent)

        def utilization(self):
            return min(100, self.spent / self.daily_limit * 100)

        def projection(self, hours_elapsed, total_hours=24):
            if hours_elapsed <= 0:
                return 0
            rate = self.spent / hours_elapsed
            return rate * total_hours

    tracker = BudgetTracker(daily_limit_usd=50.0)
    random.seed(42)
    for i in range(15):
        model = random.choice(["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
        pt = random.randint(200, 2000)
        ct = random.randint(50, 500)
        tracker.charge(model, pt, ct)

    proj_24h = tracker.projection(hours_elapsed=8, total_hours=24)
    print(f"  Daily limit: ${tracker.daily_limit:.2f}")
    print(f"  Spent so far: ${tracker.spent:.4f}")
    print(f"  Remaining: ${tracker.remaining():.4f}")
    print(f"  Utilization: {tracker.utilization():.2f}%")
    print(f"  24h projection: ${proj_24h:.4f}  -> {'OVER BUDGET' if proj_24h > tracker.daily_limit else 'within budget'}")

    # 1.4 Cost breakdown by category
    print("\n--- 1.4 Cost Breakdown by Task Category ---")

    categories = collections.defaultdict(float)
    for entry in tracker.history:
        # Assign category based on token count heuristic
        model = entry["model"]
        if "gpt-4" in model and entry["cost"] > 0.001:
            categories["complex_reasoning"] += entry["cost"]
        else:
            categories["simple_chat"] += entry["cost"]

    total = sum(categories.values())
    print(f"  {'Category':<25s}  {'Cost':>10s}  {'Share':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}")
    for cat, cost in sorted(categories.items(), key=lambda x: -x[1]):
        share = cost / total * 100 if total > 0 else 0
        bar = "#" * int(share / 2)
        print(f"  {cat:<25s}  ${cost:>8.4f}  {share:>6.1f}%  {bar}")

    print()


# ---------------------------------------------------------------------------
# Demo 2: Caching Strategies
# ---------------------------------------------------------------------------
def demo_caching_strategies():
    """Semantic cache, exact match, prefix cache."""
    print("=" * 70)
    print("Demo 2 — Caching Strategies: exact match, prefix, semantic")
    print("=" * 70)

    # 2.1 Exact-match cache
    print("\n--- 2.1 Exact-Match Cache ---")

    class ExactMatchCache:
        def __init__(self, max_size=100):
            self.cache = {}
            self.max_size = max_size
            self.hits = 0
            self.misses = 0

        def _key(self, prompt, model):
            return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:16]

        def get(self, prompt, model):
            key = self._key(prompt, model)
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

        def put(self, prompt, model, response):
            if len(self.cache) >= self.max_size:
                # evict oldest (FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            key = self._key(prompt, model)
            self.cache[key] = response

        def hit_rate(self):
            total = self.hits + self.misses
            return self.hits / total * 100 if total > 0 else 0

    ec = ExactMatchCache(max_size=50)
    prompts = [
        ("What is Python?", "gpt-4"),
        ("What is Python?", "gpt-4"),          # duplicate
        ("Explain recursion", "gpt-4"),
        ("What is Python?", "gpt-4"),          # duplicate
        ("What is machine learning?", "gpt-3.5"),
        ("Explain recursion", "gpt-4"),        # duplicate
        ("What is Python?", "gpt-4"),          # duplicate
    ]
    for prompt, model in prompts:
        cached = ec.get(prompt, model)
        if cached:
            print(f"  HIT   '{prompt[:35]:<35s}' -> {cached[:40]}")
        else:
            # Simulate LLM call
            response = f"Response to '{prompt[:20]}...'"
            ec.put(prompt, model, response)
            print(f"  MISS  '{prompt[:35]:<35s}' -> cached new response")
    print(f"  Hit rate: {ec.hit_rate():.1f}% ({ec.hits} hits / {ec.hits + ec.misses} total)")

    # 2.2 Prefix cache (for conversation continuations)
    print("\n--- 2.2 Prefix Cache (Conversation Prefix) ---")

    class PrefixCache:
        def __init__(self):
            self.prefixes = {}  # prefix -> cached response

        def find_longest_prefix(self, messages):
            """Find longest cached prefix for a message sequence."""
            best_len = 0
            for plen in range(len(messages), 0, -1):
                prefix_key = json.dumps(messages[:plen])
                if prefix_key in self.prefixes:
                    best_len = plen
                    break
            return best_len

        def cache_prefix(self, messages, response):
            prefix_key = json.dumps(messages)
            self.prefixes[prefix_key] = response

    pc = PrefixCache()
    # Simulate a growing conversation
    conv = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    pc.cache_prefix(conv, "Python is a programming language.")

    # New message adds to conversation
    conv_new = conv + [{"role": "user", "content": "How do I install it?"}]
    prefix_len = pc.find_longest_prefix(conv_new)
    print(f"  Cached conversation turns: 3")
    print(f"  New message sequence length: {len(conv_new)}")
    print(f"  Longest cached prefix: {prefix_len} turns ({prefix_len/len(conv_new)*100:.0f}% of input)")
    print(f"  Tokens saved by prefix reuse: ~{prefix_len * 40} tokens (estimated)")

    # 2.3 Semantic cache (similarity-based)
    print("\n--- 2.3 Semantic Cache (Similarity-Based) ---")

    def simple_hash_tokens(text):
        """Crude token approximation: split on whitespace + punctuation."""
        return re.findall(r"\w+", text.lower())

    def jaccard_similarity(set_a, set_b):
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    class SemanticCache:
        def __init__(self, threshold=0.7):
            self.threshold = threshold
            self.entries = []  # list of (tokens_set, response, original_text)

        def query(self, text):
            tokens = set(simple_hash_tokens(text))
            best_sim = 0
            best_resp = None
            for entry_tokens, resp, orig in self.entries:
                sim = jaccard_similarity(tokens, entry_tokens)
                if sim > best_sim:
                    best_sim = sim
                    best_resp = resp
            if best_sim >= self.threshold:
                return best_resp, best_sim
            return None, 0

        def store(self, text, response):
            tokens = set(simple_hash_tokens(text))
            self.entries.append((tokens, response, text))

    sc = SemanticCache(threshold=0.5)
    sc.store("What is the capital of France?", "Paris")
    sc.store("How does photosynthesis work?", "Plants use sunlight to convert CO2 and water into glucose.")

    queries = [
        "What is the capital of France?",        # exact
        "What's the capital city of France?",    # similar
        "Tell me about photosynthesis",          # similar to #2
        "How do black holes form?",              # new
    ]
    for q in queries:
        resp, sim = sc.query(q)
        status = f"SIMILARITY={sim:.2f} -> '{resp[:40]}'" if resp else "MISS"
        print(f"  '{q[:45]:<45s}'  {status}")

    # 2.4 Cache cost savings calculator
    print("\n--- 2.4 Cache Cost Savings Calculator ---")

    def compute_savings(total_requests, cache_hit_rate, avg_cost_per_request):
        total_cost_no_cache = total_requests * avg_cost_per_request
        cache_requests = int(total_requests * cache_hit_rate / 100)
        llm_requests = total_requests - cache_requests
        cache_overhead = cache_requests * avg_cost_per_request * 0.001  # 0.1% of cost to look up
        total_cost_cached = llm_requests * avg_cost_per_request + cache_overhead
        saved = total_cost_no_cache - total_cost_cached
        return {
            "total_requests": total_requests,
            "cached_requests": cache_requests,
            "cost_no_cache": round(total_cost_no_cache, 2),
            "cost_with_cache": round(total_cost_cached, 2),
            "savings_usd": round(saved, 2),
            "savings_pct": round(saved / total_cost_no_cache * 100, 1) if total_cost_no_cache > 0 else 0,
        }

    for hit_rate in [30, 50, 70]:
        result = compute_savings(total_requests=10000, cache_hit_rate=hit_rate, avg_cost_per_request=0.005)
        print(f"  Hit rate {hit_rate}%: {result['cached_requests']} cached, "
              f"${result['cost_no_cache']:.2f} -> ${result['cost_with_cache']:.2f} "
              f"(save ${result['savings_usd']:.2f}, {result['savings_pct']}%)")

    print()


# ---------------------------------------------------------------------------
# Demo 3: Model Routing
# ---------------------------------------------------------------------------
def demo_model_routing():
    """Complexity detection, model selection, fallback chains."""
    print("=" * 70)
    print("Demo 3 — Model Routing: complexity, selection, fallbacks")
    print("=" * 70)

    # 3.1 Complexity scoring
    print("\n--- 3.1 Request Complexity Scoring ---")

    def complexity_score(prompt):
        """Estimate request complexity from prompt features."""
        score = 0
        # Length factor
        word_count = len(prompt.split())
        score += min(30, word_count * 0.3)

        # Keyword heuristics
        complex_keywords = ["analyze", "compare", "reason", "prove", "implement", "architect",
                            "debug", "optimize", "trade-off", "evaluate"]
        simple_keywords = ["what is", "define", "list", "name", "translate", "hello"]

        text_lower = prompt.lower()
        for kw in complex_keywords:
            if kw in text_lower:
                score += 10
        for kw in simple_keywords:
            if kw in text_lower:
                score -= 5

        # Question mark count (multiple questions = complex)
        score += prompt.count("?") * 3

        # Code block detection
        if "```" in prompt or "def " in prompt or "class " in prompt:
            score += 15

        return max(0, min(100, score))

    test_prompts = [
        "Hello!",
        "What is Python?",
        "List the planets in our solar system.",
        "Implement a binary search tree in Python with insert, delete, and search operations.",
        "Compare and contrast REST and GraphQL. Analyze trade-offs for a real-time chat application.",
        "def fibonacci(n): ... optimize this for large n",
    ]
    for p in test_prompts:
        score = complexity_score(p)
        tier = "S" if score > 50 else ("M" if score > 25 else "E")
        print(f"  [{tier}] score={score:>3}  '{p[:55]}{'...' if len(p) > 55 else ''}'")

    # 3.2 Model selection based on complexity
    print("\n--- 3.2 Model Selection Strategy ---")

    model_tiers = {
        "S": {"model": "gpt-4", "cost_per_1k": 0.06, "quality": 0.98},
        "M": {"model": "gpt-4-turbo", "cost_per_1k": 0.03, "quality": 0.95},
        "E": {"model": "gpt-3.5-turbo", "cost_per_1k": 0.0015, "quality": 0.85},
    }

    def select_model(prompt):
        score = complexity_score(prompt)
        if score > 50:
            tier = "S"
        elif score > 25:
            tier = "M"
        else:
            tier = "E"
        info = model_tiers[tier]
        return tier, info

    print(f"  {'Prompt':<50s}  {'Tier':>4}  {'Model':<18s}  {'Est. Cost':>9s}")
    print(f"  {'-'*50}  {'-'*4}  {'-'*18}  {'-'*9}")
    for p in test_prompts:
        tier, info = select_model(p)
        est_cost = info["cost_per_1k"] * len(p.split()) * 0.001  # rough
        print(f"  {p[:50]:<50s}  {tier:>4}  {info['model']:<18s}  ${est_cost:.6f}")

    # 3.3 Fallback chains
    print("\n--- 3.3 Fallback Chain ---")

    class FallbackRouter:
        def __init__(self, chain):
            self.chain = chain  # list of (model, max_retries)

        def route(self, prompt, simulate_failures=None):
            """Returns (model_used, attempts, success, error_msg)."""
            simulate_failures = simulate_failures or {}
            for model, max_retries in self.chain:
                fail_prob = simulate_failures.get(model, 0)
                for attempt in range(1, max_retries + 1):
                    # Simulate call
                    if random.random() < fail_prob:
                        continue  # retry same model
                    return model, attempt, True, None
                # All retries exhausted for this model
            return None, 0, False, "all models exhausted"

    router = FallbackRouter([
        ("gpt-4", 1),
        ("gpt-4-turbo", 2),
        ("gpt-3.5-turbo", 3),
    ])

    random.seed(42)
    scenarios = [
        {"desc": "all healthy", "failures": {}},
        {"desc": "gpt-4 down", "failures": {"gpt-4": 1.0}},
        {"desc": "gpt-4 + turbo down", "failures": {"gpt-4": 1.0, "gpt-4-turbo": 1.0}},
    ]
    for sc in scenarios:
        model, attempts, success, err = router.route("test prompt", sc["failures"])
        status = f"OK -> {model} (attempt {attempts})" if success else f"FAILED: {err}"
        print(f"  {sc['desc']:<25s}  {status}")

    # 3.4 Routing cost analysis
    print("\n--- 3.4 Routing Cost Impact ---")

    def analyze_routing_impact(n_requests, naive_model, routing_distribution):
        """Compare naive (single model) vs routed approach."""
        naive_info = model_tiers.get(naive_model, {"cost_per_1k": 0.05})
        naive_cost = n_requests * naive_info["cost_per_1k"]

        routed_cost = 0
        for tier, fraction in routing_distribution.items():
            info = model_tiers[tier]
            routed_cost += n_requests * fraction * info["cost_per_1k"]

        savings = naive_cost - routed_cost
        return {
            "naive_cost": round(naive_cost, 2),
            "routed_cost": round(routed_cost, 2),
            "savings": round(savings, 2),
            "pct": round(savings / naive_cost * 100, 1) if naive_cost > 0 else 0,
        }

    # 70% simple, 20% medium, 10% complex (typical distribution)
    dist = {"E": 0.70, "M": 0.20, "S": 0.10}
    result = analyze_routing_impact(10000, "gpt-4", dist)
    print(f"  Naive (all gpt-4):  ${result['naive_cost']:>8.2f}")
    print(f"  Routed (70%E/20%M/10%S): ${result['routed_cost']:>8.2f}")
    print(f"  Savings: ${result['savings']:.2f} ({result['pct']}%)")

    print()


# ---------------------------------------------------------------------------
# Demo 4: Distillation
# ---------------------------------------------------------------------------
def demo_distillation():
    """Knowledge transfer, student-teacher, size reduction."""
    print("=" * 70)
    print("Demo 4 — Distillation: knowledge transfer, student-teacher")
    print("=" * 70)

    # 4.1 Knowledge distillation simulation
    print("\n--- 4.1 Knowledge Distillation (Soft Labels) ---")

    def softmax(logits, temperature=1.0):
        scaled = [x / temperature for x in logits]
        max_val = max(scaled)
        exps = [math.exp(x - max_val) for x in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    def hard_label(logits):
        probs = softmax(logits)
        idx = probs.index(max(probs))
        return [1.0 if i == idx else 0.0 for i in range(len(probs))]

    # Teacher model outputs (logits for 4 classes)
    teacher_logits = [5.0, 2.0, 0.5, -1.0]
    temperatures = [1.0, 2.0, 5.0]

    print(f"  Teacher logits: {teacher_logits}")
    print(f"  Hard labels:    {hard_label(teacher_logits)}")
    print()
    for T in temperatures:
        soft = softmax(teacher_logits, temperature=T)
        print(f"  T={T:<4.1f} soft labels: [{', '.join(f'{p:.4f}' for p in soft)}]")

    # 4.2 Training data generation from teacher
    print("\n--- 4.2 Synthetic Training Data from Teacher ---")

    def generate_teacher_data(n_samples=50):
        """Simulate teacher generating soft-label training data."""
        data = []
        categories = ["code", "math", "creative", "factual"]
        for _ in range(n_samples):
            # Random input features
            word_count = random.randint(5, 200)
            has_code = random.random() > 0.7
            has_numbers = random.random() > 0.6
            question_marks = random.randint(0, 3)

            # Teacher "inference"
            logits = [
                2.0 if has_code else -1.0,        # code
                1.5 if has_numbers else 0.0,       # math
                1.0 + word_count * 0.01,           # creative
                0.5,                                # factual
            ]
            soft = softmax(logits, temperature=2.0)
            data.append({
                "features": {"words": word_count, "code": has_code, "numbers": has_numbers},
                "soft_labels": soft,
            })
        return data

    teacher_data = generate_teacher_data(50)
    # Aggregate soft labels
    avg_soft = [0.0] * 4
    for d in teacher_data:
        for i in range(4):
            avg_soft[i] += d["soft_labels"][i]
    avg_soft = [x / len(teacher_data) for x in avg_soft]
    categories = ["code", "math", "creative", "factual"]
    print(f"  Generated {len(teacher_data)} samples from teacher")
    print(f"  Average soft labels across dataset:")
    for cat, val in zip(categories, avg_soft):
        bar = "#" * int(val * 40)
        print(f"    {cat:<10s}  {val:.4f}  {bar}")

    # 4.3 Student vs Teacher comparison
    print("\n--- 4.3 Student-Teacher Accuracy Comparison ---")

    def simulate_accuracy(n_test=100, teacher_strength=0.92, student_strength=0.86, gap=0.06):
        """Simulate accuracy with distillation gap."""
        teacher_correct = 0
        student_correct = 0
        student_with_distill = 0

        for _ in range(n_test):
            difficulty = random.random()
            # Teacher gets hard examples right more often
            t_correct = random.random() < teacher_strength - difficulty * 0.15
            if t_correct:
                teacher_correct += 1

            # Student without distillation
            s_correct = random.random() < student_strength - difficulty * 0.2
            if s_correct:
                student_correct += 1

            # Student with distillation (closes the gap)
            sd_correct = random.random() < (teacher_strength - gap) - difficulty * 0.17
            if sd_correct:
                student_with_distill += 1

        return {
            "teacher": teacher_correct / n_test * 100,
            "student_baseline": student_correct / n_test * 100,
            "student_distilled": student_with_distill / n_test * 100,
        }

    random.seed(42)
    acc = simulate_accuracy(n_test=1000)
    print(f"  Teacher accuracy:        {acc['teacher']:.1f}%")
    print(f"  Student (baseline):      {acc['student_baseline']:.1f}%")
    print(f"  Student (distilled):     {acc['student_distilled']:.1f}%")
    gap_before = acc['teacher'] - acc['student_baseline']
    gap_after = acc['teacher'] - acc['student_distilled']
    print(f"  Accuracy gap:            {gap_before:.1f}% -> {gap_after:.1f}%  (reduced by {gap_before - gap_after:.1f}%)")

    # 4.4 Model size vs performance trade-off
    print("\n--- 4.4 Size-Performance Trade-Off ---")

    models = [
        {"name": "gpt-4 (teacher)", "params_b": 1800, "accuracy": 95.2, "latency_ms": 800, "cost_per_1k": 0.06},
        {"name": "gpt-3.5 (baseline)", "params_b": 175, "accuracy": 87.5, "latency_ms": 200, "cost_per_1k": 0.0015},
        {"name": "distilled-7B", "params_b": 7, "accuracy": 82.1, "latency_ms": 50, "cost_per_1k": 0.0003},
        {"name": "distilled-3B", "params_b": 3, "accuracy": 78.5, "latency_ms": 25, "cost_per_1k": 0.0001},
    ]

    print(f"  {'Model':<25s}  {'Params':>8s}  {'Acc':>6s}  {'Latency':>8s}  {'Cost/1K':>9s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*9}")
    for m in models:
        size_str = f"{m['params_b']}B" if m['params_b'] >= 1 else f"{m['params_b']*1000:.0f}M"
        print(f"  {m['name']:<25s}  {size_str:>8s}  {m['accuracy']:>5.1f}%  {m['latency_ms']:>6d}ms  ${m['cost_per_1k']:.4f}")

    # Efficiency score: accuracy per dollar
    print()
    print(f"  Efficiency (accuracy / cost_per_1k):")
    for m in models:
        efficiency = m["accuracy"] / (m["cost_per_1k"] * 1000)
        bar = "#" * min(50, int(efficiency))
        print(f"    {m['name']:<25s}  {efficiency:>8.2f}  {bar}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_token_cost_analysis()
    demo_caching_strategies()
    demo_model_routing()
    demo_distillation()
