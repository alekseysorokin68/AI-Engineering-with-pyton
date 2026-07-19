"""119 — LLM Memory: история разговоров, продвинутый RAG, долгосрочная память

Темы:
  1. Conversation Memory (buffer, summary, window-based)
  2. Long-term Memory (entity extraction, knowledge graph simulation)
  3. Memory Retrieval (similarity search, relevance ranking)
  4. Memory Consolidation (compression, deduplication, decay)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import datetime
import collections

random.seed(42)


# =============================================================================
# 1. Conversation Memory — buffer, summary, window-based
# =============================================================================

def demo_conversation_memory():
    print("=" * 70)
    print("DEMO 1: Conversation Memory — buffer, summary, window-based")
    print("=" * 70)

    # --- 1a. Full buffer memory ---
    class BufferMemory:
        def __init__(self, max_tokens=500):
            self.messages = []
            self.max_tokens = max_tokens

        def add(self, role, content):
            tokens_est = len(content.split())
            self.messages.append({"role": role, "content": content, "tokens": tokens_est})

        def get_context(self):
            total = sum(m["tokens"] for m in self.messages)
            if total <= self.max_tokens:
                return self.messages
            # Trim from oldest
            kept = []
            running = 0
            for m in reversed(self.messages):
                running += m["tokens"]
                if running > self.max_tokens:
                    break
                kept.insert(0, m)
            return kept

        def stats(self):
            total = sum(m["tokens"] for m in self.messages)
            return {"messages": len(self.messages), "total_tokens": total, "within_limit": total <= self.max_tokens}

    print("--- Buffer Memory ---")
    buf = BufferMemory(max_tokens=30)
    buf.add("user", "Hello, I need help with Python")
    buf.add("assistant", "Sure! What specific topic?")
    buf.add("user", "How do I read files?")
    buf.add("assistant", "Use open('file.txt').read() to read a file in Python.")
    buf.add("user", "And write files?")
    buf.add("assistant", "Use with open('file.txt', 'w') as f: f.write('content').")
    stats = buf.stats()
    print(f"  Messages: {stats['messages']}, Tokens: {stats['total_tokens']}, Within limit: {stats['within_limit']}")
    ctx = buf.get_context()
    print(f"  Context returned: {len(ctx)} messages")
    for m in ctx:
        print(f"    [{m['role']}] {m['content'][:50]}... ({m['tokens']} tok)")

    # --- 1b. Summary memory ---
    def summarize_conversation(messages):
        summaries = []
        for m in messages:
            words = m["content"].split()[:8]
            summary = " ".join(words)
            summaries.append(f"{m['role']}: {summary}")
        combined = " | ".join(summaries)
        # Simple extractive summary
        key_phrases = []
        for m in messages:
            if any(w in m["content"].lower() for w in ["help", "how", "what", "why", "important"]):
                words = m["content"].split()[:5]
                key_phrases.append(" ".join(words))
        return {
            "full_summary": combined[:200],
            "key_phrases": key_phrases,
            "msg_count": len(messages),
            "compression_ratio": round(len(combined) / sum(len(m["content"]) for m in messages), 2)
        }

    print("\n--- Summary Memory ---")
    summary = summarize_conversation(buf.messages)
    print(f"  Message count: {summary['msg_count']}")
    print(f"  Compression ratio: {summary['compression_ratio']}")
    print(f"  Key phrases: {summary['key_phrases']}")
    print(f"  Summary: {summary['full_summary'][:100]}...")

    # --- 1c. Window-based memory ---
    def window_memory(messages, window_size=3):
        if len(messages) <= window_size:
            return messages
        # Keep first message + last window_size messages
        first = messages[0]
        recent = messages[-window_size:]
        dropped = len(messages) - window_size - 1
        return [first, {"role": "system", "content": f"[{dropped} earlier messages omitted]"}] + recent

    print("\n--- Window-Based Memory ---")
    all_msgs = [{"role": "user", "content": f"Message {i}: question about topic {i % 5}"} for i in range(10)]
    windowed = window_memory(all_msgs, window_size=3)
    print(f"  Total messages: {len(all_msgs)}, Window size: 3")
    print(f"  Windowed context: {len(windowed)} messages")
    for m in windowed:
        print(f"    [{m['role']}] {m['content'][:50]}")

    # --- 1d. Token budget management ---
    def manage_token_budget(messages, budget):
        total_tokens = sum(len(m["content"].split()) for m in messages)
        if total_tokens <= budget:
            return messages, budget - total_tokens
        # Priority: system > user > assistant, recent > old
        priority = {"system": 3, "user": 2, "assistant": 1}
        scored = [(i, priority.get(m["role"], 0) * (i + 1) / len(messages)) for i, m in enumerate(messages)]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = set()
        remaining = budget
        for idx, _ in scored:
            tokens = len(messages[idx]["content"].split())
            if tokens <= remaining:
                selected.add(idx)
                remaining -= tokens
        kept = [messages[i] for i in sorted(selected)]
        return kept, remaining

    print("\n--- Token Budget Management ---")
    budget_msgs = [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"{'question ' * (i+1)}topic {i}"}
                   for i in range(8)]
    budget = 30
    kept, remaining = manage_token_budget(budget_msgs, budget)
    total_kept = sum(len(m["content"].split()) for m in kept)
    print(f"  Input: {len(budget_msgs)} messages, budget: {budget} tokens")
    print(f"  Kept: {len(kept)} messages ({total_kept} tokens), remaining: {remaining}")
    for m in kept:
        print(f"    [{m['role']}] {m['content'][:40]}...")

    print()


# =============================================================================
# 2. Long-term Memory — entity extraction, knowledge graph simulation
# =============================================================================

def demo_long_term_memory():
    print("=" * 70)
    print("DEMO 2: Long-term Memory — entity extraction, knowledge graph simulation")
    print("=" * 70)

    # --- 2a. Entity extraction ---
    def extract_entities(text):
        patterns = {
            "person": r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b',
            "date": r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b',
            "email": r'\b[\w.]+@[\w.]+\.\w+\b',
            "number": r'\b\d+(?:\.\d+)?\b',
            "organization": r'\b(?:Google|Apple|Microsoft|OpenAI|Meta)\b'
        }
        entities = {}
        for etype, pattern in patterns.items():
            found = re.findall(pattern, text)
            if found:
                entities[etype] = list(set(found))
        return entities

    text = "On 2024-01-15, John Smith from Google met with Sarah Jones at Apple. " \
           "They discussed 3 projects and emailed john@google.com about 150 units."
    print("--- Entity Extraction ---")
    entities = extract_entities(text)
    print(f"  Text: {text[:70]}...")
    for etype, vals in entities.items():
        print(f"  {etype}: {vals}")

    # --- 2b. Knowledge graph simulation ---
    class KnowledgeGraph:
        def __init__(self):
            self.triples = []  # (subject, relation, object)
            self.entity_index = collections.defaultdict(list)
            self.relation_index = collections.defaultdict(list)

        def add(self, subject, relation, obj):
            triple = (subject, relation, obj)
            self.triples.append(triple)
            self.entity_index[subject].append(triple)
            self.entity_index[obj].append(triple)
            self.relation_index[relation].append(triple)

        def query_subject(self, entity):
            return self.entity_index.get(entity, [])

        def query_relation(self, relation):
            return self.relation_index.get(relation, [])

        def find_path(self, start, end, max_hops=3):
            if start == end:
                return [start]
            queue = [(start, [start])]
            visited = {start}
            while queue:
                node, path = queue.pop(0)
                if len(path) > max_hops:
                    continue
                for s, r, o in self.entity_index[node]:
                    next_node = o if s == node else s
                    if next_node == end:
                        return path + [f"--({r})-->", next_node]
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [f"--({r})-->", next_node]))
            return None

    print("\n--- Knowledge Graph ---")
    kg = KnowledgeGraph()
    kg.add("Alice", "works_at", "Google")
    kg.add("Alice", "knows", "Bob")
    kg.add("Bob", "works_at", "Meta")
    kg.add("Alice", "lives_in", "New York")
    kg.add("Bob", "lives_in", "San Francisco")
    kg.add("Google", "headquarters_in", "Mountain View")

    print(f"  Total triples: {len(kg.triples)}")
    print(f"  Triples about Alice:")
    for s, r, o in kg.query_subject("Alice"):
        print(f"    {s} --[{r}]--> {o}")

    print(f"\n  'works_at' relations:")
    for s, r, o in kg.query_relation("works_at"):
        print(f"    {s} -> {o}")

    path = kg.find_path("Alice", "Mountain View")
    print(f"\n  Path Alice -> Mountain View: {' '.join(path) if path else 'no path found'}")

    # --- 2c. Temporal memory with decay ---
    def memory_with_decay(entries, decay_rate=0.1):
        now = datetime.datetime(2024, 6, 15, 12, 0, 0)
        scored = []
        for entry in entries:
            entry_time = datetime.datetime.strptime(entry["timestamp"], "%Y-%m-%d")
            age_days = (now - entry_time).days
            relevance = math.exp(-decay_rate * age_days)
            access_boost = 1.0 + 0.1 * entry.get("access_count", 0)
            final_score = round(relevance * access_boost, 3)
            scored.append({**entry, "age_days": age_days, "relevance": final_score})
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored

    print("\n--- Memory with Temporal Decay ---")
    mem_entries = [
        {"content": "Python basics", "timestamp": "2024-01-10", "access_count": 5},
        {"content": "ML pipeline", "timestamp": "2024-05-20", "access_count": 2},
        {"content": "Docker setup", "timestamp": "2024-06-01", "access_count": 1},
        {"content": "API design", "timestamp": "2024-03-15", "access_count": 8},
    ]
    scored = memory_with_decay(mem_entries)
    for s in scored:
        print(f"  {s['content']:20s} age={s['age_days']:>3}d access={s['access_count']} relevance={s['relevance']}")

    # --- 2d. Entity co-occurrence matrix ---
    def cooccurrence_matrix(documents, window=2):
        entity_docs = []
        for doc in documents:
            ents = extract_entities(doc)
            flat = [e for vals in ents.values() for e in vals]
            entity_docs.append(flat)
        # Build co-occurrence within window
        all_entities = sorted(set(e for doc in entity_docs for e in doc))
        n = len(all_entities)
        idx = {e: i for i, e in enumerate(all_entities)}
        matrix = [[0] * n for _ in range(n)]
        for doc in entity_docs:
            for i in range(len(doc)):
                for j in range(i + 1, min(i + window + 1, len(doc))):
                    a, b = idx[doc[i]], idx[doc[j]]
                    matrix[a][b] += 1
                    matrix[b][a] += 1
        return all_entities, matrix

    docs = [
        "John Smith from Google met Sarah Jones at Apple.",
        "Sarah Jones joined Microsoft last year.",
        "John Smith published a paper with 42 references."
    ]
    print("\n--- Entity Co-occurrence Matrix ---")
    ents, co_matrix = cooccurrence_matrix(docs, window=2)
    print(f"  Entities: {ents}")
    print("  Co-occurrence matrix:")
    header = "        " + "  ".join([e[:8].ljust(8) for e in ents])
    print(header)
    for i, row in enumerate(co_matrix):
        print(f"  {ents[i][:8]:8s} {row}")

    print()


# =============================================================================
# 3. Memory Retrieval — similarity search, relevance ranking
# =============================================================================

def demo_memory_retrieval():
    print("=" * 70)
    print("DEMO 3: Memory Retrieval — similarity search, relevance ranking")
    print("=" * 70)

    # --- 3a. TF-IDF-like similarity ---
    def tfidf_similarity(query, documents):
        # Tokenize
        query_tokens = query.lower().split()
        doc_tokens = [doc.lower().split() for doc in documents]
        # Term frequency
        all_tokens = set(query_tokens)
        for dt in doc_tokens:
            all_tokens.update(dt)
        # IDF
        n_docs = len(documents)
        idf = {}
        for token in all_tokens:
            df = sum(1 for dt in doc_tokens if token in dt)
            idf[token] = math.log((1 + n_docs) / (1 + df)) + 1
        # TF-IDF for query
        tf_query = collections.Counter(query_tokens)
        q_vec = {t: tf * idf.get(t, 0) for t, tf in tf_query.items()}
        # TF-IDF for each doc and cosine similarity
        scores = []
        for dt in doc_tokens:
            tf_doc = collections.Counter(dt)
            d_vec = {t: tf * idf.get(t, 0) for t, tf in tf_doc.items()}
            # Cosine similarity
            common = set(q_vec.keys()) & set(d_vec.keys())
            dot = sum(q_vec[t] * d_vec[t] for t in common)
            norm_q = math.sqrt(sum(v**2 for v in q_vec.values())) or 1e-10
            norm_d = math.sqrt(sum(v**2 for v in d_vec.values())) or 1e-10
            scores.append(round(dot / (norm_q * norm_d), 4))
        return scores

    print("--- TF-IDF Similarity Search ---")
    docs = [
        "Python machine learning deep learning neural networks",
        "JavaScript web development frontend backend",
        "Python data science pandas numpy data analysis",
        "Docker containers deployment DevOps",
        "Machine learning model training optimization"
    ]
    query = "Python machine learning"
    scores = tfidf_similarity(query, docs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"  Query: \"{query}\"")
    for idx, score in ranked:
        print(f"  [{score:.4f}] doc_{idx}: {docs[idx][:50]}")

    # --- 3b. BM25 ranking ---
    def bm25_score(query, documents, k1=1.5, b=0.75):
        query_tokens = query.lower().split()
        doc_tokens = [doc.lower().split() for doc in documents]
        avg_dl = sum(len(dt) for dt in doc_tokens) / len(documents)
        # IDF
        n_docs = len(documents)
        idf = {}
        all_tokens = set(query_tokens)
        for dt in doc_tokens:
            all_tokens.update(dt)
        for token in all_tokens:
            df = sum(1 for dt in doc_tokens if token in dt)
            idf[token] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        # BM25 per doc
        scores = []
        for dt in doc_tokens:
            dl = len(dt)
            tf = collections.Counter(dt)
            score = 0
            for qt in query_tokens:
                if qt in tf:
                    term_freq = tf[qt]
                    numerator = term_freq * (k1 + 1)
                    denominator = term_freq + k1 * (1 - b + b * dl / avg_dl)
                    score += idf.get(qt, 0) * numerator / denominator
            scores.append(round(score, 4))
        return scores

    print("\n--- BM25 Ranking ---")
    query2 = "deep learning neural"
    scores2 = bm25_score(query2, docs)
    ranked2 = sorted(enumerate(scores2), key=lambda x: x[1], reverse=True)
    print(f"  Query: \"{query2}\"")
    for idx, score in ranked2:
        print(f"  [{score:.4f}] doc_{idx}: {docs[idx][:50]}")

    # --- 3c. Reciprocal Rank Fusion ---
    def reciprocal_rank_fusion(rankings, k=60):
        scores = collections.defaultdict(float)
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Reciprocal Rank Fusion (RRF) ---")
    ranking_1 = [0, 2, 4, 1, 3]
    ranking_2 = [2, 0, 4, 3, 1]
    ranking_3 = [0, 1, 2, 4, 3]
    rrf = reciprocal_rank_fusion([ranking_1, ranking_2, ranking_3])
    print(f"  Rankings:")
    print(f"    TF-IDF:  {ranking_1}")
    print(f"    BM25:    {ranking_2}")
    print(f"    Keyword: {ranking_3}")
    print(f"  Fused ranking (RRF):")
    for doc_id, score in rrf:
        print(f"    doc_{doc_id}: score={score:.4f} -> {docs[doc_id][:40]}")

    # --- 3d. Relevance feedback loop ---
    def relevance_feedback(query, documents, initial_scores, feedback_docs, alpha=0.75):
        # Rocchio-style relevance feedback
        # Expand query with terms from relevant docs
        feedback_tokens = []
        for doc_id in feedback_docs:
            feedback_tokens.extend(documents[doc_id].lower().split())
        fb_counter = collections.Counter(feedback_tokens)
        top_terms = [t for t, _ in fb_counter.most_common(3)]
        expanded_query = query.lower().split() + top_terms
        # Re-rank with expanded query
        new_scores = tfidf_similarity(" ".join(expanded_query), documents)
        # Blend
        blended = [round(alpha * old + (1 - alpha) * new, 4)
                   for old, new in zip(initial_scores, new_scores)]
        return blended, expanded_query

    print("\n--- Relevance Feedback (Rocchio) ---")
    initial = bm25_score("data analysis", docs)
    print(f"  Initial scores: {[round(s, 4) for s in initial]}")
    print(f"  User marks doc_0 and doc_2 as relevant")
    blended, expanded = relevance_feedback("data analysis", docs, initial, [0, 2])
    print(f"  Expanded query tokens: {expanded[:10]}...")
    ranked_fb = sorted(enumerate(blended), key=lambda x: x[1], reverse=True)
    print(f"  Re-ranked:")
    for idx, score in ranked_fb:
        print(f"    [{score:.4f}] doc_{idx}: {docs[idx][:50]}")

    print()


# =============================================================================
# 4. Memory Consolidation — compression, deduplication, decay
# =============================================================================

def demo_memory_consolidation():
    print("=" * 70)
    print("DEMO 4: Memory Consolidation — compression, deduplication, decay")
    print("=" * 70)

    # --- 4a. Text compression ---
    def compress_messages(messages, compression_ratio=0.5):
        compressed = []
        for msg in messages:
            words = msg["content"].split()
            target_len = max(1, int(len(words) * compression_ratio))
            # Keep first and last words + important middle ones
            if len(words) <= target_len:
                compressed.append(msg["content"])
                continue
            important = words[:2]
            middle_target = target_len - 4
            middle = words[2:-2]
            if middle_target > 0 and middle:
                step = max(1, len(middle) // middle_target)
                important.extend(middle[::step][:middle_target])
            important.extend(words[-2:])
            compressed.append(" ".join(important))
        return compressed

    print("--- Message Compression ---")
    messages = [
        {"role": "user", "content": "I need help understanding how neural networks work in deep learning applications"},
        {"role": "assistant", "content": "Neural networks are computational models inspired by biological neurons that learn patterns from data"},
        {"role": "user", "content": "Can you explain the backpropagation algorithm used for training these networks"},
    ]
    compressed = compress_messages(messages, compression_ratio=0.5)
    for orig, comp in zip(messages, compressed):
        orig_len = len(orig["content"].split())
        comp_len = len(comp.split())
        print(f"  [{orig['role']}] Original ({orig_len} words): {orig['content'][:60]}...")
        print(f"  Compressed ({comp_len} words): {comp[:60]}...")
        print(f"  Ratio: {comp_len/orig_len:.2f}")
        print()

    # --- 4b. Deduplication ---
    def deduplicate_memories(memories, threshold=0.8):
        def similarity(a, b):
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        clusters = []
        assigned = [False] * len(memories)
        for i in range(len(memories)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, len(memories)):
                if not assigned[j] and similarity(memories[i]["content"], memories[j]["content"]) >= threshold:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters

    print("--- Deduplication ---")
    memories = [
        {"content": "Python is great for machine learning"},
        {"content": "Python is excellent for machine learning tasks"},
        {"content": "Docker containers help with deployment"},
        {"content": "Python is great for machine learning and AI"},
        {"content": "Containers in Docker aid deployment processes"},
    ]
    clusters = deduplicate_memories(memories, threshold=0.5)
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}:")
        for idx in cluster:
            print(f"    [{idx}] {memories[idx]['content']}")
    print(f"  Original: {len(memories)} memories -> {len(clusters)} unique clusters")

    # --- 4c. Memory decay and forgetting ---
    def apply_decay(memories, half_life_days=30):
        now = datetime.datetime(2024, 7, 15, 12, 0, 0)
        decayed = []
        for mem in memories:
            mem_time = datetime.datetime.strptime(mem["timestamp"], "%Y-%m-%d")
            age_days = (now - mem_time).days
            strength = math.exp(-0.693 * age_days / half_life_days)
            access_modifier = 1.0 + 0.05 * mem.get("access_count", 0)
            final_strength = min(1.0, strength * access_modifier)
            keep = final_strength > 0.1
            decayed.append({
                **mem,
                "age_days": age_days,
                "decay_strength": round(final_strength, 3),
                "keep": keep
            })
        return decayed

    print("\n--- Memory Decay (Half-life = 30 days) ---")
    decay_mems = [
        {"content": "Lesson 1: Python basics", "timestamp": "2024-01-01", "access_count": 20},
        {"content": "Lesson 2: Data structures", "timestamp": "2024-03-01", "access_count": 10},
        {"content": "Lesson 3: ML algorithms", "timestamp": "2024-06-01", "access_count": 3},
        {"content": "Lesson 4: Deep learning", "timestamp": "2024-07-10", "access_count": 1},
    ]
    decayed = apply_decay(decay_mems, half_life_days=30)
    for d in decayed:
        status = "KEEP" if d["keep"] else "FORGET"
        print(f"  {d['content']:30s} age={d['age_days']:>3}d access={d['access_count']:>2} "
              f"strength={d['decay_strength']:.3f} -> {status}")
    kept = sum(1 for d in decayed if d["keep"])
    print(f"  Result: {kept}/{len(decayed)} memories retained")

    # --- 4d. Memory consolidation pipeline ---
    def consolidate(raw_messages):
        # Step 1: Deduplicate
        contents = [{"content": m["content"], "role": m["role"]} for m in raw_messages]
        unique_contents = list({m["content"]: m for m in contents}.values())

        # Step 2: Compress
        compressed = compress_messages(unique_contents, compression_ratio=0.6)

        # Step 3: Extract key facts
        key_facts = []
        for msg in unique_contents:
            words = msg["content"].split()
            # Simple keyword extraction
            keywords = [w for w in words if len(w) > 5 and w[0].isupper()]
            if keywords:
                key_facts.append({"role": msg["role"], "keywords": keywords, "content": msg["content"][:80]})

        # Step 4: Build summary
        all_keywords = [kw for kf in key_facts for kw in kf["keywords"]]
        keyword_freq = collections.Counter(all_keywords).most_common(5)

        return {
            "original_count": len(raw_messages),
            "after_dedup": len(unique_contents),
            "compressed_count": len(compressed),
            "key_facts": key_facts,
            "top_keywords": keyword_freq
        }

    print("\n--- Memory Consolidation Pipeline ---")
    raw = [
        {"role": "user", "content": "How do I optimize Python performance?"},
        {"role": "assistant", "content": "Use Python profiling tools and optimize critical loops"},
        {"role": "user", "content": "How to optimize Python speed?"},
        {"role": "assistant", "content": "Consider using C extensions, Cython, or profiling tools"},
        {"role": "user", "content": "Tell me about Machine Learning pipelines"},
    ]
    result = consolidate(raw)
    print(f"  Original: {result['original_count']} messages")
    print(f"  After dedup: {result['after_dedup']}")
    print(f"  After compression: {result['compressed_count']}")
    print(f"  Top keywords: {result['top_keywords']}")
    for kf in result["key_facts"]:
        print(f"    [{kf['role']}] {kf['content'][:50]}... | keywords: {kf['keywords']}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_conversation_memory()
    demo_long_term_memory()
    demo_memory_retrieval()
    demo_memory_consolidation()
