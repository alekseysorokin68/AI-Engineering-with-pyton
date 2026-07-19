"""
137 — Multimodal Benchmarks: evaluation metrics, VQA, captioning, retrieval

Темы:
  1. VQA Metrics (accuracy, exact match, VQA evaluation protocol)
  2. Image Captioning Metrics (BLEU, CIDEr, METEOR concepts)
  3. Retrieval Metrics (Recall@K, MAP, cross-modal retrieval)
  4. Multimodal Benchmarks (VQAv2, GQA, COCO, ImageNet)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)

def demo_vqa_metrics():
    """
    Section 1: VQA Metrics
    Accuracy, exact match, VQA evaluation protocol
    """
    print("=" * 70)
    print("DEMO 1: VQA Metrics")
    print("=" * 70)
    
    # 1.1 VQA Accuracy computation
    print("\n[1.1] VQA Accuracy Computation")
    print("-" * 40)
    
    # VQA uses soft accuracy: min(#humans_who_said_answer / 3, 1)
    def vqa_accuracy(predicted, ground_truth_answers):
        """
        predicted: model's answer string
        ground_truth_answers: list of human annotator answers
        """
        # Count how many annotators gave this answer
        pred_lower = predicted.lower().strip()
        count = sum(1 for a in ground_truth_answers if a.lower().strip() == pred_lower)
        
        # VQA accuracy: min(count/3, 1)
        accuracy = min(count / 3.0, 1.0)
        return accuracy
    
    # Example predictions and ground truth
    examples = [
        {
            'question': 'What color is the car?',
            'predicted': 'red',
            'ground_truth': ['red', 'red', 'red', 'red', 'red', 'red',
                           'red', 'red', 'red', 'red']  # 10 annotators
        },
        {
            'question': 'How many people are there?',
            'predicted': '2',
            'ground_truth': ['2', '2', '2', '2', '3', '2',
                           '2', '3', '2', '2']  # Mixed answers
        },
        {
            'question': 'Is this a dog?',
            'predicted': 'yes',
            'ground_truth': ['yes', 'yes', 'yes', 'no', 'yes', 'yes',
                           'yes', 'no', 'yes', 'yes']  # Mostly yes
        },
        {
            'question': 'What is the man doing?',
            'predicted': 'running',
            'ground_truth': ['walking', 'jogging', 'running', 'walking',
                           'jogging', 'walking', 'running', 'jogging',
                           'walking', 'running']  # Diverse answers
        },
    ]
    
    total_accuracy = 0
    print("VQA Evaluation Protocol:")
    print("Formula: accuracy = min(#humans_who_said_answer / 3, 1)")
    print("-" * 50)
    
    for ex in examples:
        acc = vqa_accuracy(ex['predicted'], ex['ground_truth'])
        total_accuracy += acc
        
        # Count answer distribution
        answer_counts = collections.Counter(a.lower() for a in ex['ground_truth'])
        top_answers = answer_counts.most_common(3)
        
        print(f"\nQ: {ex['question']}")
        print(f"  Predicted: '{ex['predicted']}'")
        print(f"  Ground truth answers: {ex['ground_truth'][:5]}...")
        print(f"  Answer distribution: {dict(top_answers)}")
        print(f"  VQA Accuracy: {acc:.3f}")
    
    mean_accuracy = total_accuracy / len(examples)
    print(f"\nMean VQA Accuracy: {mean_accuracy:.3f}")
    
    # 1.2 Exact Match vs Soft Accuracy
    print("\n[1.2] Exact Match vs Soft Accuracy")
    print("-" * 40)
    
    def exact_match(predicted, ground_truth):
        return 1.0 if predicted.lower().strip() == ground_truth.lower().strip() else 0.0
    
    def normalize_answer(answer):
        # Lowercase, remove articles, punctuation, extra whitespace
        answer = answer.lower()
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = ' '.join(answer.split())
        return answer
    
    def soft_match(predicted, ground_truth):
        pred_norm = normalize_answer(predicted)
        gt_norm = normalize_answer(ground_truth)
        return 1.0 if pred_norm == gt_norm else 0.0
    
    comparison_examples = [
        ('a dog', 'dog'),           # Articles
        ('The cat', 'cat'),         # Articles
        ('running!', 'running'),    # Punctuation
        ('a big red car', 'big red car'),  # Articles
        ('is jumping', 'jumping'),  # Verb tense
    ]
    
    print("Exact Match vs Normalized Match:")
    print(f"{'Predicted':<20} {'Ground Truth':<20} {'Exact':>8} {'Soft':>8}")
    print("-" * 60)
    
    for pred, gt in comparison_examples:
        em = exact_match(pred, gt)
        sm = soft_match(pred, gt)
        print(f"{pred:<20} {gt:<20} {em:>8.0f} {sm:>8.0f}")
    
    print("\nNormalization steps: lowercase -> remove articles -> remove punctuation")
    
    # 1.3 Answer type analysis
    print("\n[1.3] Answer Type Analysis")
    print("-" * 40)
    
    # Categorize questions by answer type
    answer_types = {
        'yes/no': {'examples': ['yes', 'no'], 'count': 45000},
        'number': {'examples': ['1', '2', '3', '4', '5'], 'count': 25000},
        'other': {'examples': ['blue', 'running', 'dog', 'table'], 'count': 30000},
    }
    
    total_q = sum(t['count'] for t in answer_types.values())
    
    print("Answer Type Distribution (VQAv2):")
    print(f"{'Type':<15} {'Count':>10} {'Percentage':>12} {'Examples'}")
    print("-" * 60)
    
    for atype, info in answer_types.items():
        pct = info['count'] / total_q * 100
        examples = ', '.join(info['examples'][:3])
        print(f"{atype:<15} {info['count']:>10,} {pct:>11.1f}% {examples}")
    
    print(f"\nTotal questions: {total_q:,}")
    print("\nNote: yes/no questions are easiest, 'other' is most challenging")
    
    # 1.4 Confidence calibration
    print("\n[1.4] Confidence Calibration")
    print("-" * 40)
    
    # Simulate model confidence vs accuracy
    random.seed(42)
    num_bins = 5
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    calibration_data = []
    for i in range(num_bins):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        # Simulate: accuracy slightly lower than confidence (overconfident)
        accuracy = bin_center * random.uniform(0.7, 0.95)
        count = random.randint(500, 2000)
        calibration_data.append({
            'confidence': bin_center,
            'accuracy': accuracy,
            'count': count
        })
    
    print("Confidence Calibration Analysis:")
    print(f"{'Confidence Bin':>15} {'Accuracy':>10} {'Count':>8} {'Gap':>8}")
    print("-" * 45)
    
    ece = 0  # Expected Calibration Error
    total_samples = sum(d['count'] for d in calibration_data)
    
    for data in calibration_data:
        gap = abs(data['confidence'] - data['accuracy'])
        ece += (data['count'] / total_samples) * gap
        print(f"{data['confidence']:>14.1%} {data['accuracy']:>10.3f} "
              f"{data['count']:>8} {gap:>8.3f}")
    
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("Formula: ECE = Σ (|n_b/N| × |acc_b - conf_b|)")
    print("Lower ECE = better calibrated model")
    
    print()

def demo_captioning_metrics():
    """
    Section 2: Image Captioning Metrics
    BLEU, CIDEr, METEOR concepts
    """
    print("=" * 70)
    print("DEMO 2: Image Captioning Metrics")
    print("=" * 70)
    
    # 2.1 BLEU Score computation
    print("\n[2.1] BLEU Score Computation")
    print("-" * 40)
    
    def compute_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def compute_bleu(candidate, references, max_n=4):
        """
        Compute BLEU score with brevity penalty.
        """
        bp = 1.0
        precisions = []
        
        # Brevity penalty
        cand_len = len(candidate)
        ref_lens = [len(ref) for ref in references]
        closest_ref = min(ref_lens, key=lambda x: abs(x - cand_len))
        
        if cand_len < closest_ref:
            bp = math.exp(1 - closest_ref / cand_len) if cand_len > 0 else 0
        
        # Precision for each n-gram
        for n in range(1, max_n + 1):
            cand_ngrams = compute_ngrams(candidate, n)
            if not cand_ngrams:
                precisions.append(0)
                continue
            
            # Count matching n-grams
            ref_ngram_counts = collections.Counter()
            for ref in references:
                ref_ngrams = compute_ngrams(ref, n)
                ref_ngram_counts.update(ref_ngrams)
            
            clipped = 0
            cand_counts = collections.Counter(cand_ngrams)
            for ngram, count in cand_counts.items():
                clipped += min(count, ref_ngram_counts.get(ngram, 0))
            
            precision = clipped / len(cand_ngrams) if cand_ngrams else 0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            score = 0
        else:
            log_avg = sum(math.log(p) for p in precisions) / len(precisions)
            score = bp * math.exp(log_avg)
        
        return score, bp, precisions
    
    # Example captions
    candidate = ['a', 'cat', 'sitting', 'on', 'a', 'red', 'chair']
    references = [
        ['a', 'cat', 'is', 'sitting', 'on', 'a', 'red', 'chair'],
        ['the', 'cat', 'sits', 'on', 'a', 'red', 'chair'],
        ['a', 'cat', 'on', 'a', 'red', 'chair'],
    ]
    
    bleu_score, bp, precisions = compute_bleu(candidate, references)
    
    print("Candidate: 'a cat sitting on a red chair'")
    print("References:")
    for i, ref in enumerate(references):
        print(f"  {i+1}. '{' '.join(ref)}'")
    
    print(f"\nBLEU-4 Score: {bleu_score:.4f}")
    print(f"Brevity Penalty: {bp:.4f}")
    print(f"Precisions (1-gram to 4-gram): {[f'{p:.3f}' for p in precisions]}")
    print("Formula: BLEU = BP × exp(Σ w_n × log(p_n))")
    
    # 2.2 CIDEr Score concept
    print("\n[2.2] CIDEr Score Concept")
    print("-" * 40)
    
    def compute_tf_idf_ngrams(caption, corpus, n=1):
        """Compute TF-IDF weighted n-grams."""
        # Term frequency in caption
        ngrams = compute_ngrams(caption, n)
        tf = collections.Counter(ngrams)
        
        # Document frequency in corpus
        df = collections.Counter()
        for doc in corpus:
            doc_ngrams = set(compute_ngrams(doc, n))
            for ng in doc_ngrams:
                df[ng] += 1
        
        # TF-IDF weights
        num_docs = len(corpus)
        tfidf = {}
        for ngram, count in tf.items():
            idf = math.log(num_docs / (df.get(ngram, 0) + 1))
            tfidf[ngram] = count * idf
        
        return tfidf
    
    # Corpus of reference captions
    corpus = [
        ['a', 'cat', 'is', 'sitting', 'on', 'a', 'red', 'chair'],
        ['the', 'cat', 'sits', 'on', 'a', 'red', 'chair'],
        ['a', 'cat', 'on', 'a', 'red', 'chair'],
        ['a', 'dog', 'lying', 'on', 'a', 'blue', 'couch'],
        ['two', 'cats', 'playing', 'in', 'the', 'garden'],
    ]
    
    # Compute CIDEr for candidate
    cider_score = 0
    ngram_weights = {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}
    
    print("CIDEr Score Computation:")
    print("Uses TF-IDF weighted n-gram matching against corpus")
    
    for n in range(1, 5):
        cand_tfidf = compute_tf_idf_ngrams(candidate, corpus, n)
        
        # Average reference TF-IDF
        ref_tfidfs = [compute_tf_idf_ngrams(ref, corpus, n) for ref in references]
        
        # Cosine similarity
        all_ngrams = set(cand_tfidf.keys())
        for rtf in ref_tfidfs:
            all_ngrams.update(rtf.keys())
        
        # Vector representation
        cand_vec = [cand_tfidf.get(ng, 0) for ng in all_ngrams]
        ref_vecs = [[rtf.get(ng, 0) for ng in all_ngrams] for rtf in ref_tfidfs]
        
        # Average cosine similarity
        sims = []
        for rv in ref_vecs:
            dot = sum(c * r for c, r in zip(cand_vec, rv))
            norm_c = math.sqrt(sum(c**2 for c in cand_vec))
            norm_r = math.sqrt(sum(r**2 for r in rv))
            sim = dot / (norm_c * norm_r) if norm_c * norm_r > 0 else 0
            sims.append(sim)
        
        avg_sim = sum(sims) / len(sims)
        cider_score += ngram_weights[n] * avg_sim
        
        print(f"  {n}-gram CIDEr: {avg_sim:.4f}")
    
    print(f"\nCIDEr Score: {cider_score:.4f}")
    print("Formula: CIDEr = (1/N) Σ_n g^n · (cosine_sim between TF-IDF vectors)")
    
    # 2.3 METEOR Score concept
    print("\n[2.3] METEOR Score Concept")
    print("-" * 40)
    
    # Simplified METEOR with stemming and synonyms
    def simple_stem(word):
        """Very basic stemmer for demonstration."""
        suffixes = ['ing', 'ed', 's', 'ly', 'er', 'est']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word
    
    # Simple synonym groups
    synonyms = {
        'cat': ['feline', 'kitten'],
        'dog': ['puppy', 'canine'],
        'big': ['large', 'huge'],
        'small': ['little', 'tiny'],
        'run': ['sprint', 'jog'],
    }
    
    candidate_meteor = ['a', 'cat', 'running', 'on', 'the', 'floor']
    reference_meteor = ['a', 'feline', 'running', 'on', 'the', 'ground']
    
    # Build synonym-expanded reference
    ref_expanded = set()
    for word in reference_meteor:
        ref_expanded.add(word)
        ref_expanded.add(simple_stem(word))
        for syn_group, syns in synonyms.items():
            if word == syn_group or word in syns:
                ref_expanded.add(syn_group)
                ref_expanded.update(syns)
    
    # Compute matches
    matches = 0
    match_details = []
    for word in candidate_meteor:
        stemmed = simple_stem(word)
        if word in ref_expanded or stemmed in ref_expanded:
            matches += 1
            match_details.append(f"'{word}' matched")
    
    precision = matches / len(candidate_meteor)
    recall = matches / len(reference_meteor)
    f_mean = (precision * recall) / (0.5 * precision + 0.5 * recall) if (precision + recall) > 0 else 0
    
    print("METEOR Components:")
    print(f"  Candidate: {candidate_meteor}")
    print(f"  Reference: {reference_meteor}")
    print(f"\n  Matches found: {matches}")
    for detail in match_details[:4]:
        print(f"    - {detail}")
    
    print(f"\n  Precision: {matches}/{len(candidate_meteor)} = {precision:.3f}")
    print(f"  Recall: {matches}/{len(reference_meteor)} = {recall:.3f}")
    print(f"  F-mean: {f_mean:.3f}")
    print("Formula: METEOR = F_mean × (1 - penalty)")
    print("METEOR uses stemming, synonyms, and exact matching")
    
    # 2.4 Caption evaluation protocol
    print("\n[2.4] Caption Evaluation Protocol")
    print("-" * 40)
    
    # Multiple metrics comparison
    metrics_summary = {
        'BLEU-1': {'score': 0.72, 'range': (0, 1), 'focus': 'precision'},
        'BLEU-4': {'score': 0.35, 'range': (0, 1), 'focus': 'precision'},
        'METEOR': {'score': 0.28, 'range': (0, 1), 'focus': 'recall'},
        'CIDEr': {'score': 1.15, 'range': (0, 10), 'focus': 'consensus'},
        'SPICE': {'score': 0.21, 'range': (0, 1), 'focus': 'semantic'},
    }
    
    print("Standard Caption Evaluation Metrics:")
    print(f"{'Metric':<10} {'Score':>8} {'Range':>12} {'Focus':>12}")
    print("-" * 45)
    
    for metric, info in metrics_summary.items():
        range_str = f"[{info['range'][0]}, {info['range'][1]}]"
        print(f"{metric:<10} {info['score']:>8.2f} {range_str:>12} {info['focus']:>12}")
    
    print("\nNote: Each metric captures different aspects of caption quality")
    print("BLEU: n-gram precision, METEOR: recall + synonyms, CIDEr: consensus")
    
    print()

def demo_retrieval_metrics():
    """
    Section 3: Retrieval Metrics
    Recall@K, MAP, cross-modal retrieval
    """
    print("=" * 70)
    print("DEMO 3: Retrieval Metrics")
    print("=" * 70)
    
    # 3.1 Recall@K computation
    print("\n[3.1] Recall@K Computation")
    print("-" * 40)
    
    def compute_recall_at_k(ranked_list, ground_truth, k):
        """
        ranked_list: list of retrieved item indices (ranked)
        ground_truth: set of relevant item indices
        """
        retrieved_at_k = ranked_list[:k]
        relevant_retrieved = sum(1 for item in retrieved_at_k if item in ground_truth)
        return relevant_retrieved / len(ground_truth) if ground_truth else 0
    
    # Simulated retrieval results
    num_items = 20
    relevant_items = {2, 5, 8, 12, 17}  # 5 relevant items
    
    # Ranked retrieval list (simulated similarities)
    random.seed(42)
    similarities = [(i, random.random()) for i in range(num_items)]
    similarities.sort(key=lambda x: -x[1])
    ranked_list = [item[0] for item in similarities]
    
    print(f"Total items: {num_items}")
    print(f"Relevant items: {sorted(relevant_items)}")
    print(f"Top 10 retrieved: {ranked_list[:10]}")
    
    print("\nRecall@K Results:")
    for k in [1, 2, 5, 10]:
        recall = compute_recall_at_k(ranked_list, relevant_items, k)
        relevant_in_k = sum(1 for item in ranked_list[:k] if item in relevant_items)
        print(f"  Recall@{k}: {recall:.3f} ({relevant_in_k}/{len(relevant_items)} relevant found)")
    
    print("\nFormula: Recall@K = |relevant ∩ top-K| / |relevant|")
    
    # 3.2 Mean Average Precision (MAP)
    print("\n[3.2] Mean Average Precision (MAP)")
    print("-" * 40)
    
    def compute_ap(ranked_list, ground_truth):
        """Compute Average Precision for a single query."""
        score = 0
        num_relevant = 0
        
        for i, item in enumerate(ranked_list):
            if item in ground_truth:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                score += precision_at_i
        
        return score / len(ground_truth) if ground_truth else 0
    
    # Multiple queries
    queries = [
        {'id': 'q1', 'relevant': {1, 4, 7, 15}},
        {'id': 'q2', 'relevant': {0, 3, 8, 12, 19}},
        {'id': 'q3', 'relevant': {2, 6}},
        {'id': 'q4', 'relevant': {5, 10, 14}},
    ]
    
    print("Mean Average Precision (MAP) Computation:")
    print("-" * 50)
    
    ap_scores = []
    for query in queries:
        # Generate ranked list for this query
        random.seed(hash(query['id']))
        sims = [(i, random.random()) for i in range(num_items)]
        sims.sort(key=lambda x: -x[1])
        ranked = [s[0] for s in sims]
        
        ap = compute_ap(ranked, query['relevant'])
        ap_scores.append(ap)
        
        print(f"\n{query['id']}: relevant={sorted(query['relevant'])}")
        print(f"  Top 5: {ranked[:5]}")
        print(f"  AP = {ap:.4f}")
    
    map_score = sum(ap_scores) / len(ap_scores)
    print(f"\nMAP = {map_score:.4f}")
    print("Formula: MAP = (1/Q) × Σ_q AP(q)")
    print("Formula: AP(q) = (1/|rel|) × Σ_k Precision@k × rel(k)")
    
    # 3.3 Cross-modal retrieval evaluation
    print("\n[3.3] Cross-Modal Retrieval Evaluation")
    print("-" * 40)
    
    # Simulated cross-modal retrieval results
    num_queries = 100
    num_retrieved = 50
    
    random.seed(42)
    
    # Image-to-Text retrieval
    i2t_recall = {k: 0 for k in [1, 5, 10, 25, 50]}
    
    for _ in range(num_queries):
        # Random number of relevant items (1-5)
        num_rel = random.randint(1, 5)
        rel_items = set(random.sample(range(num_retrieved), num_rel))
        
        # Random ranking
        ranking = list(range(num_retrieved))
        random.shuffle(ranking)
        
        for k in i2t_recall:
            retrieved = set(ranking[:k])
            i2t_recall[k] += len(retrieved & rel_items) / num_rel
    
    # Normalize
    for k in i2t_recall:
        i2t_recall[k] /= num_queries
    
    # Text-to-Image retrieval (simulated)
    t2i_recall = {k: 0 for k in [1, 5, 10, 25, 50]}
    
    for _ in range(num_queries):
        num_rel = random.randint(1, 5)
        rel_items = set(random.sample(range(num_retrieved), num_rel))
        ranking = list(range(num_retrieved))
        random.shuffle(ranking)
        
        for k in t2i_recall:
            retrieved = set(ranking[:k])
            t2i_recall[k] += len(retrieved & rel_items) / num_rel
    
    for k in t2i_recall:
        t2i_recall[k] /= num_queries
    
    print(f"Cross-Modal Retrieval Results ({num_queries} queries):")
    print(f"{'K':>5} {'I2T Recall':>12} {'T2I Recall':>12} {'Mean':>10}")
    print("-" * 42)
    
    for k in [1, 5, 10, 25, 50]:
        mean_r = (i2t_recall[k] + t2i_recall[k]) / 2
        print(f"{k:>5} {i2t_recall[k]:>12.3f} {t2i_recall[k]:>12.3f} {mean_r:>10.3f}")
    
    print("\nFormula: Recall@K measures fraction of relevant items in top-K")
    print("I2T: given image, retrieve text; T2I: given text, retrieve image")
    
    # 3.4 Mean Reciprocal Rank (MRR)
    print("\n[3.4] Mean Reciprocal Rank (MRR)")
    print("-" * 40)
    
    def compute_mrr(ranked_list, ground_truth):
        """Compute Mean Reciprocal Rank."""
        for i, item in enumerate(ranked_list):
            if item in ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    # Example queries with their rankings
    mrr_examples = [
        {'relevant': {3}, 'ranking': [0, 1, 2, 3, 4, 5]},      # Rank 4
        {'relevant': {1}, 'ranking': [0, 1, 2, 3, 4, 5]},      # Rank 2
        {'relevant': {5}, 'ranking': [0, 1, 2, 3, 4, 5]},      # Rank 6
        {'relevant': {0}, 'ranking': [0, 1, 2, 3, 4, 5]},      # Rank 1
        {'relevant': {2}, 'ranking': [0, 1, 2, 3, 4, 5]},      # Rank 3
    ]
    
    print("Mean Reciprocal Rank (MRR):")
    print("-" * 40)
    
    rr_scores = []
    for i, ex in enumerate(mrr_examples):
        rr = compute_mrr(ex['ranking'], ex['relevant'])
        rr_scores.append(rr)
        
        rank = ex['ranking'].index(list(ex['relevant'])[0]) + 1
        print(f"  Query {i+1}: first relevant at rank {rank}, RR = 1/{rank} = {rr:.3f}")
    
    mrr = sum(rr_scores) / len(rr_scores)
    print(f"\nMRR = {mrr:.3f}")
    print("Formula: MRR = (1/Q) × Σ_q (1/rank_of_first_relevant)")
    print("Focuses on position of first relevant result")
    
    print()

def demo_multimodal_benchmarks():
    """
    Section 4: Multimodal Benchmarks
    VQAv2, GQA, COCO, ImageNet
    """
    print("=" * 70)
    print("DEMO 4: Multimodal Benchmarks")
    print("=" * 70)
    
    # 4.1 VQAv2 Dataset overview
    print("\n[4.1] VQAv2 Dataset Overview")
    print("-" * 40)
    
    vqa_stats = {
        'train': {'images': 82783, 'questions': 443757, 'answers': 4437570},
        'val': {'images': 40504, 'questions': 214354, 'answers': 2143540},
        'test-dev': {'images': 40504, 'questions': 214354, 'answers': 2143540},
        'test-standard': {'images': 40504, 'questions': 214354, 'answers': 2143540},
    }
    
    print("VQAv2 Dataset Statistics:")
    print(f"{'Split':<15} {'Images':>10} {'Questions':>12} {'Answers':>12}")
    print("-" * 50)
    
    for split, stats in vqa_stats.items():
        print(f"{split:<15} {stats['images']:>10,} {stats['questions']:>12,} {stats['answers']:>12,}")
    
    total_images = sum(s['images'] for s in vqa_stats.values())
    total_questions = sum(s['questions'] for s in vqa_stats.values())
    print(f"\nTotal: {total_images:,} images, {total_questions:,} questions")
    
    print("\nAnswer Type Distribution:")
    answer_dist = {'yes/no': 44.7, 'number': 14.6, 'other': 40.7}
    for atype, pct in answer_dist.items():
        bar = '#' * int(pct / 2)
        print(f"  {atype:>8}: {pct:5.1f}% {bar}")
    
    # 4.2 GQA Dataset overview
    print("\n[4.2] GQA Dataset Overview")
    print("-" * 40)
    
    gqa_stats = {
        'train': {'images': 82226, 'questions': 22235162},
        'val': {'images': 9871, 'questions': 1015760},
        'testdev': {'images': 997, 'questions': 49822},
    }
    
    print("GQA Dataset Statistics:")
    print(f"{'Split':<10} {'Images':>10} {'Questions':>15}")
    print("-" * 37)
    
    for split, stats in gqa_stats.items():
        print(f"{split:<10} {stats['images']:>10,} {stats['questions']:>15,}")
    
    print("\nGQA Characteristics:")
    print("  - Compositional reasoning questions")
    print("  - Scene graph grounded")
    print("  - Single correct answer per question")
    print("  - Question types: yes/no, open, number")
    
    question_types = {'yes/no': 30, 'open': 60, 'number': 10}
    print("\nQuestion Types:")
    for qtype, pct in question_types.items():
        print(f"    {qtype}: {pct}%")
    
    # 4.3 COCO Captions overview
    print("\n[4.3] COCO Captions Overview")
    print("-" * 40)
    
    coco_stats = {
        'train': {'images': 118287, 'captions': 591753},
        'val': {'images': 5000, 'captions': 25000},
        'test': {'images': 40775, 'captions': 202650},  # test-dev subset
    }
    
    print("COCO Captions Statistics:")
    print(f"{'Split':<10} {'Images':>10} {'Captions':>10} {'Captions/Image':>15}")
    print("-" * 47)
    
    for split, stats in coco_stats.items():
        caps_per_img = stats['captions'] / stats['images']
        print(f"{split:<10} {stats['images']:>10,} {stats['captions']:>10,} {caps_per_img:>15.1f}")
    
    print("\nCaption Statistics:")
    random.seed(42)
    caption_lengths = [random.randint(8, 25) for _ in range(1000)]
    avg_len = sum(caption_lengths) / len(caption_lengths)
    vocab_size = random.randint(8000, 10000)
    
    print(f"  Average caption length: {avg_len:.1f} words")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Captions per image: 5")
    
    # 4.4 Benchmark comparison
    print("\n[4.4] Benchmark Comparison")
    print("-" * 40)
    
    benchmarks = {
        'VQAv2': {
            'task': 'Visual Question Answering',
            'size': '1.1M questions',
            'metrics': ['VQA Accuracy'],
            'best_model': '82.49%',
            'difficulty': 'Medium'
        },
        'GQA': {
            'task': 'Compositional Reasoning',
            'size': '22M questions',
            'metrics': ['Accuracy'],
            'best_model': '64.61%',
            'difficulty': 'Hard'
        },
        'COCO Captioning': {
            'task': 'Image Captioning',
            'size': '123K images',
            'metrics': ['BLEU-4', 'CIDEr', 'METEOR'],
            'best_model': 'B@4: 40.0, C: 141.6',
            'difficulty': 'Medium'
        },
        'ImageNet': {
            'task': 'Image Classification',
            'size': '1.2M images, 1000 classes',
            'metrics': ['Top-1 Accuracy', 'Top-5 Accuracy'],
            'best_model': '91.0% / 99.0%',
            'difficulty': 'Medium'
        },
        'Flickr30k': {
            'task': 'Image Retrieval',
            'size': '31K images',
            'metrics': ['Recall@1', 'Recall@5', 'Recall@10'],
            'best_model': 'R@1: 95.6%',
            'difficulty': 'Easy'
        },
    }
    
    print("Multimodal Benchmarks Comparison:")
    print("=" * 70)
    
    for name, info in benchmarks.items():
        print(f"\n{name}:")
        print(f"  Task: {info['task']}")
        print(f"  Size: {info['size']}")
        print(f"  Metrics: {', '.join(info['metrics'])}")
        print(f"  Best Model: {info['best_model']}")
        print(f"  Difficulty: {info['difficulty']}")
    
    # Leaderboard simulation
    print("\n" + "=" * 70)
    print("Simulated VQAv2 Leaderboard (2024):")
    print("-" * 50)
    
    leaderboard = [
        ('GPT-4V (Oracle)', 82.49),
        ('PaLI-X', 81.97),
        ('Flamingo', 80.45),
        ('BLIP-2', 79.14),
        ('LLaVA-1.5', 78.02),
        ('CoCa', 77.30),
    ]
    
    print(f"{'Rank':<6} {'Model':<20} {'Score':>10}")
    print("-" * 38)
    
    for rank, (model, score) in enumerate(leaderboard, 1):
        bar = '#' * int(score / 2)
        print(f"{rank:<6} {model:<20} {score:>9.2f}%")
    
    print("\nNote: Scores are approximate and change with new submissions")
    
    print()

if __name__ == "__main__":
    demo_vqa_metrics()
    demo_captioning_metrics()
    demo_retrieval_metrics()
    demo_multimodal_benchmarks()