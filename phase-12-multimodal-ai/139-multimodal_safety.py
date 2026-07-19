"""139 — Multimodal Safety: content filtering, bias detection, visual hallucination

Темы:
  1. Visual Content Safety (NSFW detection, harmful content filtering)
  2. Bias in Multimodal Models (gender, racial, cultural bias in vision-language)
  3. Visual Hallucination (object hallucination, attribute binding errors)
  4. Safety Alignment (RLHF for multimodal, constitutional AI)

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
# Demo 1 — Visual Content Safety
# ---------------------------------------------------------------------------

def demo_visual_content_safety():
    """
    Shows NSFW detection, harmful content filtering, and safety classifiers.
    """
    print("=" * 70)
    print("DEMO 1 — Visual Content Safety")
    print("=" * 70)

    # --- 1a. NSFW Detection Pipeline ---
    print("\n--- 1a. NSFW Detection Pipeline ---")
    categories = ["safe", "suggestive", "explicit", "violence", "hate_symbol"]
    thresholds = {"safe": 0.0, "suggestive": 0.3, "explicit": 0.7, "violence": 0.5,
                  "hate_symbol": 0.6}

    # simulate classifier outputs for 6 images
    random.seed(42)
    image_scores = []
    for img_id in range(6):
        scores = {cat: random.random() for cat in categories}
        # normalize so scores sum to 1
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}
        top_cat = max(scores, key=scores.get)
        confidence = scores[top_cat]
        blocked = confidence >= thresholds.get(top_cat, 0.5)
        image_scores.append((img_id, top_cat, confidence, blocked, scores))

    print(f"  {'Image':>6s} {'Category':>12s} {'Conf':>6s} {'Decision':>10s}")
    for img_id, cat, conf, blocked, _ in image_scores:
        decision = "BLOCKED" if blocked else "ALLOWED"
        print(f"  {img_id:>6d} {cat:>12s} {conf:>6.3f} {decision:>10s}")

    # --- 1b. Multi-label Safety Scoring ---
    print("\n--- 1b. Multi-label Safety Scoring ---")
    image_text_pairs = [
        (0, "A person standing on a cliff"),
        (1, "A knife on a cutting board"),
        (2, "A crowd at a protest"),
        (3, "A child playing with a toy"),
    ]
    safety_dimensions = ["violence", "self_harm", "sexual", "drugs", "weapons"]
    print(f"  {'Pair':>4s} {'Description':>35s}", end="")
    for dim in safety_dimensions:
        print(f" {dim:>8s}", end="")
    print(f" {'Risk':>8s}")

    for img_id, desc in image_text_pairs:
        scores = {dim: random.uniform(0, 0.4) for dim in safety_dimensions}
        # add semantic hints
        if "knife" in desc.lower():
            scores["weapons"] = random.uniform(0.5, 0.9)
        if "child" in desc.lower():
            scores["sexual"] = random.uniform(0, 0.1)
        max_risk = max(scores.values())
        risk_level = "HIGH" if max_risk > 0.6 else "MED" if max_risk > 0.3 else "LOW"
        print(f"  {img_id:>4d} {desc:>35s}", end="")
        for dim in safety_dimensions:
            print(f" {scores[dim]:>8.3f}", end="")
        print(f" {risk_level:>8s}")

    # --- 1c. Text-Image Consistency Check ---
    print("\n--- 1c. Text-Image Consistency for Safety ---")
    safety_violations = [
        {"text": "A peaceful park scene", "visual_safety": 0.95, "text_safety": 0.98,
         "consistency": 0.92},
        {"text": "A weapon displayed", "visual_safety": 0.2, "text_safety": 0.15,
         "consistency": 0.88},
        {"text": "A normal household item", "visual_safety": 0.85, "text_safety": 0.9,
         "consistency": 0.3},  # mismatch — image is actually harmful
    ]
    for item in safety_violations:
        combined = (item["visual_safety"] + item["text_safety"]) / 2
        safe = combined > 0.5 and item["consistency"] > 0.5
        print(f"  Text: \"{item['text']}\"")
        print(f"    Visual safety: {item['visual_safety']:.2f}, "
              f"Text safety: {item['text_safety']:.2f}, "
              f"Consistency: {item['consistency']:.2f}")
        print(f"    Combined score: {combined:.2f} → {'SAFE' if safe else 'UNSAFE'}")

    # --- 1d. Content Filter Cascade ---
    print("\n--- 1d. Content Filter Cascade (Efficiency) ---")
    n_images = 1000
    filter_stages = [
        ("fast hash check", 0.45),    # 45% filtered by known bad hashes
        ("low-res classifier", 0.25), # 25% more filtered by cheap model
        ("full classifier", 0.15),    # 15% more by heavy model
        ("human review", 0.05),       # 5% sent to humans
    ]
    remaining = n_images
    print(f"  Processing {n_images} images through cascade:")
    for stage, filter_rate in filter_stages:
        filtered = int(remaining * filter_rate)
        remaining -= filtered
        print(f"    {stage:>25s}: filtered={filtered:>4d}, remaining={remaining:>4d}")

    print(f"\n  Throughput: {n_images/2.0:.0f} images/sec (cascade)")
    print(f"  vs full model: {n_images/45.0:.0f} images/sec")
    print(f"  Speedup: {45.0/2.0:.0f}x")


# ---------------------------------------------------------------------------
# Demo 2 — Bias in Multimodal Models
# ---------------------------------------------------------------------------

def demo_bias_in_multimodal():
    """
    Shows gender, racial, and cultural bias in vision-language models.
    """
    print("=" * 70)
    print("DEMO 2 — Bias in Multimodal Models")
    print("=" * 70)

    # --- 2a. Gender Bias in Image Captioning ---
    print("\n--- 2a. Gender Bias in Image Captioning ---")
    professions = ["doctor", "nurse", "engineer", "teacher", "CEO", "assistant"]
    gender_distribution = {
        "doctor":   {"male": 0.72, "female": 0.28},
        "nurse":    {"male": 0.15, "female": 0.85},
        "engineer": {"male": 0.81, "female": 0.19},
        "teacher":  {"male": 0.38, "female": 0.62},
        "CEO":      {"male": 0.89, "female": 0.11},
        "assistant":{"male": 0.22, "female": 0.78},
    }
    print(f"  {'Profession':>12s} {'Male':>8s} {'Female':>8s} {'Ratio M/F':>10s}")
    for prof, dist in gender_distribution.items():
        ratio = dist["male"] / dist["female"]
        print(f"  {prof:>12s} {dist['male']:>8.0%} {dist['female']:>8.0%} "
              f"{ratio:>10.2f}")

    # --- 2b. Racial Bias in Face Recognition ---
    print("\n--- 2b. Racial Bias in Face Recognition ---")
    demographics = [
        ("White", 0.995, 0.001),
        ("Black", 0.965, 0.015),
        ("Asian", 0.982, 0.003),
        ("Hispanic", 0.978, 0.008),
        ("Indian", 0.971, 0.012),
    ]
    print(f"  {'Demographic':>12s} {'Accuracy':>10s} {'FPR':>8s} {'Disparity':>10s}")
    baseline_acc = demographics[0][1]
    for demo, acc, fpr in demographics:
        disparity = baseline_acc - acc
        print(f"  {demo:>12s} {acc:>10.3f} {fpr:>8.3f} {disparity:>10.3f}")

    # Equalized odds gap
    max_gap = max(demographics, key=lambda x: baseline_acc - x[1])
    print(f"\n  Largest accuracy gap: {baseline_acc - max_gap[1]:.3f} "
          f"({max_gap[0]} vs {demographics[0][0]})")

    # --- 2c. Cultural Bias in VQA ---
    print("\n--- 2c. Cultural Bias in Visual Question Answering ---")
    questions = [
        "What is this person eating?",
        "What holiday is being celebrated?",
        "What is the typical occupation here?",
    ]
    cultural_answers = [
        {"Western": "pizza", "East Asian": "rice", "South Asian": "curry",
         "Middle Eastern": "hummus"},
        {"Western": "Christmas", "East Asian": "Lunar New Year",
         "South Asian": "Diwali", "Middle Eastern": "Eid"},
        {"Western": "office worker", "East Asian": "factory worker",
         "South Asian": "IT professional", "Middle Eastern": "merchant"},
    ]
    for q, answers in zip(questions, cultural_answers):
        print(f"\n  Q: \"{q}\"")
        for culture, answer in answers.items():
            print(f"    {culture:>15s} → \"{answer}\"")

    # --- 2d. Bias Metrics ---
    print("\n--- 2d. Bias Metrics and Mitigation ---")
    # demographic parity difference
    before_mitigation = {"gender": 0.34, "race": 0.22, "age": 0.18}
    after_mitigation = {"gender": 0.08, "race": 0.05, "age": 0.04}
    print(f"  {'Metric':>8s} {'Before':>8s} {'After':>8s} {'Reduction':>10s}")
    for metric in before_mitigation:
        before = before_mitigation[metric]
        after = after_mitigation[metric]
        reduction = (before - after) / before
        print(f"  {metric:>8s} {before:>8.2f} {after:>8.2f} {reduction:>10.0%}")

    print("\n  Mitigation strategies:")
    print("    1. Balanced training data (re-weighting underrepresented groups)")
    print("    2. Adversarial debiasing (remove protected attributes from features)")
    print("    3. Counterfactual augmentation (swap demographic attributes)")
    print("    4. Post-hoc calibration (adjust decision thresholds per group)")


# ---------------------------------------------------------------------------
# Demo 3 — Visual Hallucination
# ---------------------------------------------------------------------------

def demo_visual_hallucination():
    """
    Shows object hallucination, attribute binding errors, and detection methods.
    """
    print("=" * 70)
    print("DEMO 3 — Visual Hallucination")
    print("=" * 70)

    # --- 3a. Types of Visual Hallucination ---
    print("\n--- 3a. Types of Visual Hallucination ---")
    hallucination_types = {
        "object hallucination": "model mentions objects not present in image",
        "attribute error": "correct object, wrong attribute (e.g., wrong color)",
        "relation error": "correct objects, wrong spatial relationship",
        "action hallucination": "incorrect action/activity description",
        "counting error": "wrong number of objects",
    }
    for htype, desc in hallucination_types.items():
        print(f"  {htype:>22s}: {desc}")

    # --- 3b. Hallucination Rate by Model ---
    print("\n--- 3b. Hallucination Rate by Model Size ---")
    models = [
        ("Tiny (12M)", 0.12, 0.38),
        ("Small (125M)", 0.08, 0.25),
        ("Base (350M)", 0.05, 0.18),
        ("Large (1.3B)", 0.03, 0.12),
        ("XL (7B)", 0.02, 0.08),
    ]
    print(f"  {'Model':>16s} {'Object Hall.':>12s} {'Attribute Err':>14s} {'Total':>8s}")
    for name, obj, attr in models:
        total = (obj + attr) / 2
        print(f"  {name:>16s} {obj:>12.0%} {attr:>14.0%} {total:>8.0%}")

    # --- 3c. Attribute Binding Errors ---
    print("\n--- 3c. Attribute Binding Error Analysis ---")
    ground_truth = {
        "red car": {"object": "car", "color": "red", "position": "left"},
        "blue ball": {"object": "ball", "color": "blue", "position": "center"},
        "green tree": {"object": "tree", "color": "green", "position": "right"},
    }
    predictions = [
        {"object": "car", "color": "red", "position": "left"},   # correct
        {"object": "ball", "color": "red", "position": "center"}, # color error
        {"object": "tree", "color": "green", "position": "left"}, # position error
        {"object": "car", "color": "blue", "position": "right"},  # all wrong
    ]

    print(f"  {'Prediction':>25s} {'Object':>7s} {'Color':>6s} {'Position':>9s}")
    for pred in predictions:
        key = f"{pred['color']} {pred['object']}"
        gt = ground_truth.get(key, {"object": "?", "color": "?", "position": "?"})
        obj_match = "✓" if pred["object"] == gt["object"] else "✗"
        col_match = "✓" if pred["color"] == gt["color"] else "✗"
        pos_match = "✓" if pred["position"] == gt["position"] else "✗"
        print(f"  {key:>25s} {obj_match:>7s} {col_match:>6s} {pos_match:>9s}")

    # --- 3d. Hallucination Detection Methods ---
    print("\n--- 3d. Hallucination Detection Methods ---")
    methods = [
        ("CLIPScore", "image-text similarity", 0.72),
        ("CHAIR", "caption hallucination metric", 0.81),
        ("POPE", " polling-based probe", 0.78),
        ("LCVS", " learnable consistency", 0.85),
    ]
    n_samples = 200
    print(f"  Evaluating on {n_samples} image-text pairs:")
    for name, desc, detection_rate in methods:
        detected = int(n_samples * detection_rate)
        false_positives = int(n_samples * (1 - detection_rate) * 0.3)
        print(f"  {name:>10s} ({desc}): "
              f"detected={detected}, FPs={false_positives}, "
              f"precision={detected/(detected+false_positives):.2f}")

    print("\n  Best practice: ensemble multiple detection methods for robustness")


# ---------------------------------------------------------------------------
# Demo 4 — Safety Alignment
# ---------------------------------------------------------------------------

def demo_safety_alignment():
    """
    Shows RLHF for multimodal and constitutional AI approaches.
    """
    print("=" * 70)
    print("DEMO 4 — Safety Alignment")
    print("=" * 70)

    # --- 4a. RLHF for Multimodal Models ---
    print("\n--- 4a. RLHF for Multimodal Models ---")
    rlhf_stages = [
        ("Stage 1: Supervised Fine-tuning", "human-labeled image-text pairs",
         10000, 2.5),
        ("Stage 2: Reward Model", "human preference comparisons",
         50000, 8.0),
        ("Stage 3: PPO Optimization", "policy gradient with KL penalty",
         100000, 45.0),
    ]
    for stage, desc, n_examples, cost in rlhf_stages:
        print(f"\n  {stage}")
        print(f"    Data: {desc}")
        print(f"    Annotations: {n_examples:,}")
        print(f"    Cost estimate: ${cost:,.0f}k")

    total_cost = sum(c for _, _, _, c in rlhf_stages)
    print(f"\n  Total RLHF cost estimate: ${total_cost:,.0f}k")

    # --- 4b. Constitutional AI Principles ---
    print("\n--- 4b. Constitutional AI Principles for Multimodal ---")
    principles = [
        ("Harmlessness", "Model refuses to generate harmful visual descriptions"),
        ("Helpfulness", "Model provides accurate, useful visual information"),
        ("Honesty", "Model acknowledges uncertainty about visual content"),
        ("Fairness", "Model avoids stereotypes in image descriptions"),
        ("Privacy", "Model respects PII in images (faces, text, etc.)"),
    ]
    for i, (name, desc) in enumerate(principles, 1):
        print(f"  {i}. {name}: {desc}")

    # --- 4c. Safety Taxonomy ---
    print("\n--- 4c. Multimodal Safety Taxonomy ---")
    taxonomy = {
        "Direct Harm": {
            "violence": ["graphic content", "threats", "weapon instructions"],
            "self_harm": ["suicide methods", "self-injury", "eating disorders"],
            "sexual": ["explicit content", "CSAM", "non-consensual"],
        },
        "Indirect Harm": {
            "misinformation": ["deepfakes", "medical misinformation", "fake news"],
            "privacy": ["facial recognition", "location tracking", "PII exposure"],
            "bias": ["stereotyping", "discrimination", "representation harm"],
        },
        "Systemic Risk": {
            "surveillance": ["mass identification", "behavioral tracking"],
            "manipulation": ["propaganda", "emotional exploitation"],
        },
    }
    for category, subcats in taxonomy.items():
        print(f"\n  {category}:")
        for subcat, examples in subcats.items():
            print(f"    {subcat:>15s}: {', '.join(examples)}")

    # --- 4d. Safety Evaluation Framework ---
    print("\n--- 4d. Safety Evaluation Framework ---")
    eval_categories = [
        ("Refusal accuracy", 0.94, "correctly refuses harmful requests"),
        ("False refusal rate", 0.06, "incorrectly refuses safe requests"),
        ("Harmful completion rate", 0.02, "generates harmful content"),
        ("Bias score (max group diff)", 0.08, "demographic performance gap"),
        ("Hallucination rate", 0.15, "mentions objects not in image"),
    ]
    print(f"  {'Metric':>35s} {'Score':>7s} {'Description'}")
    for metric, score, desc in eval_categories:
        status = "✓" if score < 0.15 else "⚠" if score < 0.25 else "✗"
        print(f"  {metric:>35s} {score:>7.2f} {status} {desc}")

    # overall safety rating
    scores = [s for _, s, _ in eval_categories]
    overall = sum(scores) / len(scores)
    print(f"\n  Overall safety score: {overall:.2f}")
    print(f"  Safety rating: {'PASS' if overall < 0.15 else 'NEEDS REVIEW'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_visual_content_safety()
    demo_bias_in_multimodal()
    demo_visual_hallucination()
    demo_safety_alignment()
