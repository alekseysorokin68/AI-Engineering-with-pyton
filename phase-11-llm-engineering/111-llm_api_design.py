"""111 — LLM API Design: системные промпты, форматы сообщений, temperature, top-p, top-k

Темы:
  1. Message Formats (system/user/assistant roles, message arrays)
  2. Temperature & Sampling (temperature scaling, top-p nucleus, top-k)
  3. System Prompt Engineering (role setting, constraints, output format)
  4. API Response Handling (usage tracking, token counting, retry logic)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib

random.seed(42)


# ---------------------------------------------------------------------------
# Демо 1: Message Formats — системные промпты, форматы сообщений
# ---------------------------------------------------------------------------
def demo_message_formats():
    print("=" * 70)
    print("DEMO 1: Message Formats — форматы сообщений для LLM API")
    print("=" * 70)

    # 1.1 Простейший single-turn диалог
    messages_single = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    print("\n--- 1.1 Single-turn messages ---")
    for i, msg in enumerate(messages_single):
        print(f"  [{i}] role={msg['role']:10s} content={msg['content']}")

    # 1.2 Multi-turn диалог с system prompt
    messages_multi = [
        {"role": "system", "content": "You are a helpful math tutor. Answer concisely."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4"},
        {"role": "user", "content": "And 3 × 4?"},
    ]
    print("\n--- 1.2 Multi-turn dialog with system prompt ---")
    for i, msg in enumerate(messages_multi):
        print(f"  [{i}] role={msg['role']:10s} content={msg['content']}")

    # 1.3 Функция подсчёта токенов (по символам)
    def estimate_tokens(text):
        words = text.split()
        chars_per_token = 4.0
        return max(1, int(len(text) / chars_per_token))

    total_tokens = sum(estimate_tokens(m["content"]) for m in messages_multi)
    print(f"\n  Estimated total input tokens: {total_tokens}")

    # 1.4 Сериализация в JSON — стандартный формат передачи
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages_multi,
        "temperature": 0.7,
    }
    json_str = json.dumps(payload, indent=2)
    print(f"\n--- 1.4 API payload (JSON, {len(json_str)} chars) ---")
    print(json_str[:300] + ("..." if len(json_str) > 300 else ""))


# ---------------------------------------------------------------------------
# Демо 2: Temperature & Sampling — temperature, top-p, top-k
# ---------------------------------------------------------------------------
def demo_temperature_sampling():
    print("\n" + "=" * 70)
    print("DEMO 2: Temperature & Sampling — temperature scaling, top-p, top-k")
    print("=" * 70)

    # Логиты: распределение перед softmax
    vocab = ["cat", "dog", "fish", "bird", "snake", "rabbit"]
    logits = [3.0, 2.5, 1.0, 0.5, 0.1, -1.0]

    def softmax(l, temp):
        scaled = [x / max(temp, 0.01) for x in l]
        max_s = max(scaled)
        exps = [math.exp(x - max_s) for x in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    # 2.1 Влияние temperature на распределение
    print("\n--- 2.1 Temperature scaling effect ---")
    for temp in [0.2, 0.7, 1.0, 1.5, 2.0]:
        probs = softmax(logits, temp)
        top_idx = probs.index(max(probs))
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        print(f"  T={temp:.1f}: top=\"{vocab[top_idx]}\" p={max(probs):.3f}  "
              f"entropy={entropy:.3f}  probs={[f'{p:.3f}' for p in probs]}")

    # 2.2 Top-p (nucleus sampling)
    print("\n--- 2.2 Top-p (nucleus sampling) ---")
    probs = softmax(logits, 1.0)
    sorted_pairs = sorted(zip(vocab, probs), key=lambda x: -x[1])

    for threshold in [0.5, 0.8, 0.95, 1.0]:
        cumulative = 0.0
        nucleus = []
        for word, p in sorted_pairs:
            cumulative += p
            nucleus.append((word, p, cumulative))
            if cumulative >= threshold:
                break
        nucleus_words = [w for w, p, _ in nucleus]
        total_in = sum(p for _, p, _ in nucleus)
        print(f"  top_p={threshold:.2f}: nucleus={nucleus_words} "
              f"(covers {total_in:.1%} of probability)")

    # 2.3 Top-k sampling
    print("\n--- 2.3 Top-k sampling ---")
    for k in [1, 2, 3, 5]:
        top_k_pairs = sorted_pairs[:k]
        total_prob = sum(p for _, p in top_k_pairs)
        renorm = [p / total_prob for _, p in top_k_pairs]
        print(f"  top_k={k}: words={[w for w, _ in top_k_pairs]} "
              f"probs={[f'{p:.3f}' for p in renorm]}")

    # 2.4 Совмещённый сэмплинг: top-k + temperature + top-p
    print("\n--- 2.4 Combined: top-k(3) + T=0.5 + top-p(0.9) ---")
    random.seed(42)
    k = 3
    temp = 0.5
    top_p_threshold = 0.9

    top_k_words = [w for w, _ in sorted_pairs[:k]]
    top_k_logits = [logits[vocab.index(w)] for w in top_k_words]
    probs = softmax(top_k_logits, temp)

    cumulative = 0.0
    filtered = []
    for w, p in zip(top_k_words, probs):
        cumulative += p
        if cumulative <= top_p_threshold:
            filtered.append((w, p))
        else:
            break

    total_fp = sum(p for _, p in filtered)
    final_probs = [p / total_fp for _, p in filtered]
    sampled = random.choices([w for w, _ in filtered], weights=final_probs, k=5)
    print(f"  After filtering: {[w for w, _ in filtered]}")
    print(f"  5 samples: {sampled}")


# ---------------------------------------------------------------------------
# Демо 3: System Prompt Engineering — конструирование промптов
# ---------------------------------------------------------------------------
def demo_system_prompts():
    print("\n" + "=" * 70)
    print("DEMO 3: System Prompt Engineering — role, constraints, output format")
    print("=" * 70)

    # 3.1 Role-based system prompt
    system_role = (
        "You are a senior Python engineer with 10 years of experience. "
        "You always write clean, well-structured code."
    )
    print(f"\n--- 3.1 Role prompt ---\n  \"{system_role}\"")

    # 3.2 Constraint prompt — ограничения
    system_constraints = (
        "Rules:\n"
        "1. Never use external libraries.\n"
        "2. Keep answers under 50 words.\n"
        "3. Use numbered lists.\n"
        "4. If unsure, say 'I don't know'."
    )
    print(f"\n--- 3.2 Constraints prompt ---\n{system_constraints}")

    # 3.3 Output format prompt — задание формата вывода
    system_format = (
        "Always respond in this exact JSON format:\n"
        '{"answer": "...", "confidence": 0.0-1.0, "sources": ["..."]}'
    )
    print(f"--- 3.3 Output format prompt ---\n  \"{system_format}\"")

    # 3.4 Симуляция: как system prompt влияет на структуру ответа
    print("\n--- 3.4 Simulated response with format constraint ---")
    simulated_response = {
        "answer": "Python uses dynamic typing",
        "confidence": 0.95,
        "sources": ["Python docs", "PEP 484"]
    }
    print(f"  System prompt enforces structure:")
    print(f"  {json.dumps(simulated_response, indent=4)}")

    # Метрика: длина системного промпта в «токенах»
    combined = system_role + "\n" + system_constraints + "\n" + system_format
    est_tokens = len(combined.split()) * 1.3  # ~1.3 tokens per word avg
    print(f"\n  System prompt total: {len(combined)} chars, ~{int(est_tokens)} tokens")


# ---------------------------------------------------------------------------
# Демо 4: API Response Handling — usage tracking, retry logic
# ---------------------------------------------------------------------------
def demo_api_response():
    print("\n" + "=" * 70)
    print("DEMO 4: API Response Handling — usage, retry, token counting")
    print("=" * 70)

    # 4.1 Симуляция API-ответа
    fake_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The quick brown fox jumps over the lazy dog."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 10,
            "total_tokens": 35
        }
    }
    print("\n--- 4.1 Simulated API response ---")
    print(f"  id: {fake_response['id']}")
    print(f"  finish_reason: {fake_response['choices'][0]['finish_reason']}")
    print(f"  usage: {fake_response['usage']}")

    # 4.2 Token usage tracking
    print("\n--- 4.2 Token usage tracking ---")
    class UsageTracker:
        def __init__(self):
            self.total_prompt = 0
            self.total_completion = 0
            self.calls = []

        def record(self, usage):
            self.total_prompt += usage["prompt_tokens"]
            self.total_completion += usage["completion_tokens"]
            self.calls.append(usage)

        def summary(self):
            total = self.total_prompt + self.total_completion
            cost = self.total_prompt * 0.0000005 + self.total_completion * 0.0000015
            return {
                "calls": len(self.calls),
                "prompt_tokens": self.total_prompt,
                "completion_tokens": self.total_completion,
                "total_tokens": total,
                "estimated_cost_usd": round(cost, 6)
            }

    tracker = UsageTracker()
    tracker.record({"prompt_tokens": 25, "completion_tokens": 10})
    tracker.record({"prompt_tokens": 30, "completion_tokens": 15})
    tracker.record({"prompt_tokens": 20, "completion_tokens": 8})
    print(f"  After 3 calls: {json.dumps(tracker.summary(), indent=4)}")

    # 4.3 Retry logic с exponential backoff
    print("\n--- 4.3 Retry with exponential backoff ---")
    def simulate_retry(max_retries=4):
        random.seed(42)
        for attempt in range(max_retries):
            will_fail = random.random() < 0.6
            backoff = min(2 ** attempt + random.random(), 30)
            status = "FAIL" if will_fail else "OK"
            print(f"  Attempt {attempt+1}: {status}  "
                  f"backoff={backoff:.2f}s")
            if not will_fail:
                return {"success": True, "attempts": attempt + 1}
        return {"success": False, "attempts": max_retries}

    result = simulate_retry()
    print(f"  Result: {json.dumps(result)}")

    # 4.4 Token counting: characters vs words
    print("\n--- 4.4 Token estimation methods ---")
    sample = "The quick brown fox jumps over the lazy dog"
    char_based = len(sample) / 4.0
    word_based = len(sample.split()) * 1.3
    hash_based = int(hashlib.md5(sample.encode()).hexdigest()[:8], 16) % 100
    print(f"  Text: \"{sample}\"")
    print(f"  Chars: {len(sample)}")
    print(f"  Char-based estimate: {char_based:.1f} tokens")
    print(f"  Word-based estimate: {word_based:.1f} tokens")
    print(f"  Hash-based (for demo): {hash_based} (pseudo)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_message_formats()
    demo_temperature_sampling()
    demo_system_prompts()
    demo_api_response()
    print("\n" + "=" * 70)
    print("All demos complete: 111-llm_api_design.py")
    print("=" * 70)
