"""112 — Token Management: контекстные окна, стратегии обрезки, sliding window

Темы:
  1. Token Counting (character-based estimation, BPE simulation)
  2. Context Window Management (sliding window, truncation strategies)
  3. Conversation Compression (summarization, message pruning)
  4. Smart Truncation (importance-based, recency bias)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib

random.seed(42)


# ---------------------------------------------------------------------------
# Демо 1: Token Counting — подсчёт токенов
# ---------------------------------------------------------------------------
def demo_token_counting():
    print("=" * 70)
    print("DEMO 1: Token Counting — character-based, BPE simulation")
    print("=" * 70)

    # 1.1 Character-based estimation
    text = "The quick brown fox jumps over the lazy dog. It was a sunny day."
    char_count = len(text)
    word_count = len(text.split())
    char_based_estimate = char_count / 4.0
    word_based_estimate = word_count * 1.3

    print(f"\n--- 1.1 Character-based estimation ---")
    print(f"  Text: \"{text}\"")
    print(f"  Characters: {char_count}")
    print(f"  Words: {word_count}")
    print(f"  Token estimate (chars/4): {char_based_estimate:.1f}")
    print(f"  Token estimate (words×1.3): {word_based_estimate:.1f}")
    print(f"  Formula: tokens ≈ chars / 4.0 (English average)")

    # 1.2 BPE simulation — симуляция Byte-Pair Encoding
    print("\n--- 1.2 BPE simulation ---")

    def simple_bpe_encode(text, merges):
        tokens = list(text)
        for a, b in merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens = tokens[:i] + [a + b] + tokens[i + 2:]
                else:
                    i += 1
        return tokens

    merges = [("t", "h"), ("e", " "), ("qu", "i"), ("th", "e"), ("er", " "), (" th", "e")]
    bpe_tokens = simple_bpe_encode("the quick", merges)
    print(f"  Input: \"the quick\"")
    print(f"  Merges applied: {merges[:3]}...")
    print(f"  BPE tokens: {bpe_tokens}")
    print(f"  Token count: {len(bpe_tokens)}")

    # 1.3 Vocabulary size vs token count relationship
    print("\n--- 1.3 Vocabulary size effect ---")
    vocab_sizes = [1000, 5000, 10000, 30000, 50000]
    sample_words = 100
    for vocab_size in vocab_sizes:
        # Larger vocab → more unique subwords, fewer tokens per word
        tokens_per_word = 3.5 - 0.6 * math.log10(vocab_size / 1000)
        tokens_per_word = max(1.0, tokens_per_word)
        total_tokens = int(sample_words * tokens_per_word)
        print(f"  Vocab={vocab_size:>6}: ~{tokens_per_word:.2f} tokens/word → "
              f"{total_tokens} tokens for {sample_words} words")

    # 1.4 Multi-language token efficiency
    print("\n--- 1.4 Multi-language token efficiency ---")
    languages = {
        "English": "Hello, how are you doing today?",
        "Chinese": "你好，你今天过得怎么样？",
        "Russian": "Привет, как у тебя дела сегодня?",
        "Japanese": "こんにちは、今日はどうですか？",
    }
    for lang, text in languages.items():
        chars = len(text)
        english_ratio = len(text.encode("utf-8")) / len(text.encode("ascii", errors="replace")) if text.isascii() else len(text.encode("utf-8")) / (chars * 2)
        est_tokens = len(text.encode("utf-8")) / 3.5  # rough byte-level
        print(f"  {lang:>10}: {chars} chars, ~{est_tokens:.0f} tokens "
              f"(UTF-8 bytes/3.5)")


# ---------------------------------------------------------------------------
# Демо 2: Context Window Management — слайдинг окно
# ---------------------------------------------------------------------------
def demo_context_window():
    print("\n" + "=" * 70)
    print("DEMO 2: Context Window Management — sliding window, truncation")
    print("=" * 70)

    def estimate_tokens(msg):
        return max(1, len(msg["content"]) // 4)

    # Создаём симуляцию длинного разговора
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming language and its history."},
        {"role": "assistant", "content": "Python was created by Guido van Rossum and released in 1991."},
        {"role": "user", "content": "What are its main features?"},
        {"role": "assistant", "content": "Python features include dynamic typing, garbage collection, and a large standard library with modules for web, data, and automation."},
        {"role": "user", "content": "How does it compare to Java?"},
        {"role": "assistant", "content": "Python is interpreted while Java is compiled. Python uses dynamic typing, Java uses static. Python is often more concise."},
        {"role": "user", "content": "What about performance?"},
        {"role": "assistant", "content": "Java is generally faster due to JIT compilation. Python is slower but great for rapid prototyping and scripting."},
        {"role": "user", "content": "Tell me about the GIL."},
        {"role": "assistant", "content": "The Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecode simultaneously."},
        {"role": "user", "content": "How do I use async in Python?"},
        {"role": "assistant", "content": "Use async/await syntax with asyncio module. Define coroutines with async def and run with asyncio.run()."},
    ]

    # 2.1 Sliding window truncation
    print("\n--- 2.1 Sliding window (keep last N messages) ---")
    window_sizes = [3, 5, 7]
    for wsize in window_sizes:
        truncated = conversation[-wsize:]
        tokens = sum(estimate_tokens(m) for m in truncated)
        roles = [m["role"][0] for m in truncated]
        print(f"  Window={wsize}: tokens={tokens}, roles={''.join(roles)}")

    # 2.2 System-priority truncation — system prompt always kept
    print("\n--- 2.2 System-priority truncation ---")
    max_tokens = 120
    system_msg = conversation[0]
    system_tokens = estimate_tokens(system_msg)
    remaining_budget = max_tokens - system_tokens

    kept = [system_msg]
    for msg in reversed(conversation[1:]):
        msg_tokens = estimate_tokens(msg)
        if remaining_budget - msg_tokens >= 0:
            kept.insert(1, msg)
            remaining_budget -= msg_tokens

    kept_roles = [m["role"][0] for m in kept]
    kept_tokens = sum(estimate_tokens(m) for m in kept)
    print(f"  Budget: {max_tokens} tokens")
    print(f"  Kept: {len(kept)} messages, {kept_tokens} tokens")
    print(f"  Roles: {''.join(kept_roles)}")

    # 2.3 Summary-based truncation
    print("\n--- 2.3 Summary-based truncation ---")
    summary = ("Conversation about Python: discussed history (Guido, 1991), "
               "features (dynamic typing, stdlib), comparison with Java "
               "(interpreted vs compiled, speed), GIL limitations, and async/await.")
    summary_tokens = estimate_tokens({"content": summary})
    print(f"  Summary: \"{summary[:80]}...\"")
    print(f"  Summary tokens: ~{summary_tokens}")
    print(f"  Original conversation tokens: {sum(estimate_tokens(m) for m in conversation)}")
    print(f"  Compression ratio: {sum(estimate_tokens(m) for m in conversation) / summary_tokens:.1f}x")

    # 2.4 Token budget allocation
    print("\n--- 2.4 Token budget allocation ---")
    total_budget = 200
    allocations = {
        "system_prompt": 0.10,
        "conversation_history": 0.50,
        "current_query": 0.15,
        "response_space": 0.25,
    }
    for name, pct in allocations.items():
        tokens = int(total_budget * pct)
        print(f"  {name:25s}: {pct:.0%} = {tokens} tokens")
    print(f"  {'TOTAL':25s}: 100% = {total_budget} tokens")


# ---------------------------------------------------------------------------
# Демо 3: Conversation Compression — сжатие разговора
# ---------------------------------------------------------------------------
def demo_conversation_compression():
    print("\n" + "=" * 70)
    print("DEMO 3: Conversation Compression — summarization, pruning")
    print("=" * 70)

    # 3.1 Message pruning — удаление по возрасту
    print("\n--- 3.1 Message pruning by age ---")
    messages = [
        {"role": "user", "content": "Hello!", "age": 10},
        {"role": "assistant", "content": "Hi there!", "age": 9},
        {"role": "user", "content": "How's the weather?", "age": 8},
        {"role": "assistant", "content": "It's sunny.", "age": 7},
        {"role": "user", "content": "Thanks!", "age": 6},
        {"role": "assistant", "content": "You're welcome!", "age": 5},
        {"role": "user", "content": "Bye!", "age": 4},
    ]
    threshold = 6
    pruned = [m for m in messages if m["age"] <= threshold]
    print(f"  Messages before pruning: {len(messages)}")
    print(f"  Messages after pruning (age <= {threshold}): {len(pruned)}")
    print(f"  Removed: {len(messages) - len(pruned)} old messages")

    # 3.2 Importance-based pruning
    print("\n--- 3.2 Importance-based pruning ---")

    def score_importance(msg):
        keywords = ["important", "key", "remember", "critical", "never forget"]
        score = 1.0
        content_lower = msg["content"].lower()
        for kw in keywords:
            if kw in content_lower:
                score += 2.0
        if msg["role"] == "system":
            score += 3.0
        if msg["role"] == "assistant" and len(msg["content"]) > 50:
            score += 0.5
        return round(score, 2)

    scored_msgs = [{"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": "Remember this: my project is about climate change."},
                   {"role": "assistant", "content": "Got it, your project focuses on climate change."},
                   {"role": "user", "content": "What's 2+2?"},
                   {"role": "assistant", "content": "4."},
                   {"role": "user", "content": "This is important: the deadline is Friday!"},
                   {"role": "assistant", "content": "Understood, deadline is Friday. I'll keep that in mind."}]

    scored = [(msg, score_importance(msg)) for msg in scored_msgs]
    scored.sort(key=lambda x: x[1], reverse=True)
    print("  Importance scores:")
    for msg, score in scored:
        print(f"    [{score:4.1f}] {msg['role']:10s}: {msg['content'][:60]}")

    top_n = 4
    kept = sorted(scored[:top_n], key=lambda x: scored_msgs.index(x[0]))
    print(f"\n  Keeping top {top_n} by importance:")
    for msg, score in kept:
        print(f"    [{score:4.1f}] {msg['role']}: {msg['content'][:60]}")

    # 3.3 Content deduplication
    print("\n--- 3.3 Content deduplication ---")
    raw_msgs = ["I love cats", "I love cats", "Dogs are great", "I love cats", "Dogs are great"]
    seen = set()
    unique = []
    for msg in raw_msgs:
        fingerprint = hashlib.md5(msg.lower().encode()).hexdigest()[:8]
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(msg)
    print(f"  Raw: {raw_msgs}")
    print(f"  After dedup: {unique}")
    print(f"  Removed {len(raw_msgs) - len(unique)} duplicate(s)")

    # 3.4 Compression summary
    print("\n--- 3.4 Compression summary ---")
    sample_conv = [
        {"role": "user", "content": "Tell me about Python history."},
        {"role": "assistant", "content": "Python was created by Guido van Rossum in 1991."},
        {"role": "user", "content": "What are its main features?"},
        {"role": "assistant", "content": "Dynamic typing, garbage collection, large standard library."},
        {"role": "user", "content": "How does it compare to Java?"},
        {"role": "assistant", "content": "Python is interpreted, Java is compiled. Python is more concise."},
        {"role": "user", "content": "Tell me about the GIL."},
        {"role": "assistant", "content": "The Global Interpreter Lock prevents true multithreading."},
    ]
    original_tokens = sum(len(m["content"]) // 4 for m in sample_conv)
    compressed = "Summary: user asked about Python history, features, Java comparison, GIL, and async."
    compressed_tokens = len(compressed) // 4
    ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0
    print(f"  Original: {original_tokens} tokens")
    print(f"  Compressed: {compressed_tokens} tokens")
    print(f"  Ratio: {ratio:.1f}x compression")


# ---------------------------------------------------------------------------
# Демо 4: Smart Truncation — importance + recency bias
# ---------------------------------------------------------------------------
def demo_smart_truncation():
    print("\n" + "=" * 70)
    print("DEMO 4: Smart Truncation — importance-based, recency bias")
    print("=" * 70)

    # 4.1 Recency bias scoring
    print("\n--- 4.1 Recency bias scoring ---")

    def recency_score(index, total, decay=0.85):
        return round(decay ** (total - 1 - index), 3)

    total = 8
    for i in range(total):
        score = recency_score(i, total)
        bar = "█" * int(score * 30)
        print(f"  msg[{i}]: {score:.3f} {bar}")

    # 4.2 Combined scoring: importance + recency
    print("\n--- 4.2 Combined importance + recency ---")

    def combined_score(msg, index, total, importance_weight=0.4, recency_weight=0.6):
        imp = score_importance(msg)
        rec = recency_score(index, total)
        return round(importance_weight * imp + recency_weight * rec, 3)

    messages = [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "Tell me about classes."},
        {"role": "assistant", "content": "Classes are blueprints for objects using class keyword."},
        {"role": "user", "content": "Show inheritance."},
        {"role": "assistant", "content": "class Dog(Animal): pass — Dog inherits from Animal."},
        {"role": "user", "content": "What about decorators?"},
        {"role": "assistant", "content": "@decorator syntax modifies function behavior."},
        {"role": "user", "content": "Remember: always use type hints!"},
    ]

    def score_importance(msg):
        score = 1.0
        if msg["role"] == "system":
            score += 3.0
        if any(kw in msg["content"].lower() for kw in ["remember", "always", "important"]):
            score += 2.0
        if len(msg["content"]) > 40:
            score += 0.5
        return score

    combined = [(msg, combined_score(msg, i, len(messages))) for i, msg in enumerate(messages)]
    combined.sort(key=lambda x: -x[1])
    print("  Scores (importance + recency):")
    for msg, score in combined[:5]:
        print(f"    {score:.3f} | {msg['role']:10s} | {msg['content'][:50]}")

    # 4.3 Budget-aware smart truncation
    print("\n--- 4.3 Budget-aware smart truncation ---")
    budget = 100
    system_msg = messages[0]
    system_tokens = len(system_msg["content"]) // 4
    remaining = budget - system_tokens

    scored_msgs = [(messages[i], combined_score(messages[i], i, len(messages)))
                   for i in range(1, len(messages))]
    scored_msgs.sort(key=lambda x: -x[1])

    selected = [system_msg]
    used = system_tokens
    for msg, score in scored_msgs:
        tok = len(msg["content"]) // 4
        if used + tok <= budget:
            selected.append(msg)
            used += tok

    selected.sort(key=lambda x: messages.index(x))
    print(f"  Budget: {budget} tokens")
    print(f"  System prompt: {system_tokens} tokens (always kept)")
    print(f"  Selected {len(selected) - 1} messages within {remaining} remaining tokens:")
    for msg in selected:
        tok = len(msg["content"]) // 4
        print(f"    {msg['role']:10s} [{tok:3d} tokens]: {msg['content'][:50]}")

    # 4.4 Truncation strategy comparison
    print("\n--- 4.4 Strategy comparison ---")
    strategies = {
        "Keep-first": lambda msgs: msgs[:4],
        "Keep-last": lambda msgs: msgs[-4:],
        "System+Last": lambda msgs: [msgs[0]] + msgs[-3:],
        "Smart": lambda msgs: [msgs[0]] + [m for m, _ in scored_msgs[:3]],
    }

    all_msgs = messages
    for name, strategy in strategies.items():
        result = strategy(all_msgs)
        total_tok = sum(len(m["content"]) // 4 for m in result)
        print(f"  {name:15s}: {len(result)} msgs, {total_tok} tokens")
        for m in result:
            print(f"    {m['role']:10s}: {m['content'][:50]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_token_counting()
    demo_context_window()
    demo_conversation_compression()
    demo_smart_truncation()
    print("\n" + "=" * 70)
    print("All demos complete: 112-token_management.py")
    print("=" * 70)
