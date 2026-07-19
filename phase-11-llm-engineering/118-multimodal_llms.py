"""118 — Multi-Modal LLMs: vision + language, audio + language

Темы:
  1. Vision-Language Architecture (image encoding, cross-attention, alignment)
  2. Image Understanding (captioning, VQA, OCR simulation)
  3. Audio-Language Models (audio tokens, speech understanding)
  4. Multi-Modal Fusion (early/late fusion, attention mechanisms)

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
# 1. Vision-Language Architecture — image encoding, cross-attention, alignment
# =============================================================================

def demo_vision_language_architecture():
    print("=" * 70)
    print("DEMO 1: Vision-Language Architecture — image encoding, cross-attention, alignment")
    print("=" * 70)

    # --- 1a. Image patch embedding ---
    def image_to_patches(image_grid, patch_size=2):
        h, w = len(image_grid), len(image_grid[0])
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = [row[j:j+patch_size] for row in image_grid[i:i+patch_size]]
                patches.append(patch)
        return patches

    # Simulate 8x8 grayscale image
    random.seed(42)
    image = [[round(random.random(), 2) for _ in range(8)] for _ in range(8)]
    patches = image_to_patches(image, patch_size=2)
    print("--- Image Patch Embedding ---")
    print(f"  Image size: {len(image)}x{len(image[0])}")
    print(f"  Patch size: 2x2")
    print(f"  Number of patches: {len(patches)}")
    print(f"  First patch: {patches[0]}")

    # --- 1b. Linear projection to embedding space ---
    def project_patches(patches, embed_dim=4):
        embeddings = []
        for patch in patches:
            flat = [val for row in patch for val in row]
            random.seed(hash(str(flat)))
            embedding = [round(random.gauss(0, 1), 3) for _ in range(embed_dim)]
            norm = math.sqrt(sum(e**2 for e in embedding))
            embedding = [round(e / norm, 3) for e in embedding]
            embeddings.append(embedding)
        return embeddings

    embeddings = project_patches(patches[:4], embed_dim=4)
    print("\n--- Linear Projection to Embeddings ---")
    print(f"  Embedding dimension: 4")
    for i, emb in enumerate(embeddings):
        norm = math.sqrt(sum(e**2 for e in emb))
        print(f"  Patch {i}: {emb} (norm={norm:.3f})")

    # --- 1c. Cross-attention mechanism ---
    def cross_attention(query_emb, key_embs, value_embs):
        scores = []
        for k in key_embs:
            score = sum(q * kv for q, kv in zip(query_emb, k))
            scores.append(score)
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        weights = [round(e / sum_exp, 3) for e in exp_scores]
        output = [round(sum(w * v for w, v in zip(weights, [ve[i] for ve in value_embs])), 3)
                  for i in range(len(query_emb))]
        return weights, output

    query = embeddings[0]
    keys = embeddings
    values = [[round(random.random(), 2) for _ in range(4)] for _ in range(4)]
    print("\n--- Cross-Attention ---")
    print(f"  Query: {query}")
    weights, output = cross_attention(query, keys, values)
    print(f"  Attention weights: {weights}")
    print(f"  Output: {output}")
    print(f"  Weights sum: {sum(weights):.3f}")

    # --- 1d. Contrastive alignment (CLIP-like) ---
    def contrastive_alignment(image_embs, text_embs, temperature=0.07):
        n_img = len(image_embs)
        n_txt = len(text_embs)
        logits = []
        for i in range(n_img):
            row = []
            for j in range(n_txt):
                dot = sum(a * b for a, b in zip(image_embs[i], text_embs[j]))
                row.append(round(dot / temperature, 3))
            logits.append(row)
        return logits

    random.seed(42)
    text_embs = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    logits = contrastive_alignment(embeddings[:3], text_embs)
    print("\n--- Contrastive Alignment (CLIP-style) ---")
    print(f"  Temperature: 0.07")
    print(f"  Logits matrix ({len(logits)}x{len(logits[0])}):")
    for i, row in enumerate(logits):
        print(f"    img_{i}: {row}")
    best_match = max(range(len(logits)), key=lambda i: max(logits[i]))
    print(f"  Best image-text pair: img_{best_match} (max logit: {max(logits[best_match])})")

    print()


# =============================================================================
# 2. Image Understanding — captioning, VQA, OCR simulation
# =============================================================================

def demo_image_understanding():
    print("=" * 70)
    print("DEMO 2: Image Understanding — captioning, VQA, OCR simulation")
    print("=" * 70)

    # --- 2a. Image captioning ---
    def generate_caption(detected_objects, scene_attributes):
        random.seed(hash(str(detected_objects) + str(scene_attributes)))
        templates = [
            "A {scene} with {obj1} and {obj2}.",
            "{obj1} in a {scene} setting with {obj2}.",
            "The image shows {obj1}, {obj2} in a {scene}."
        ]
        if len(detected_objects) >= 2:
            template = random.choice(templates)
            caption = template.format(
                scene=scene_attributes.get("scene", "outdoor"),
                obj1=detected_objects[0],
                obj2=detected_objects[1]
            )
        else:
            caption = f"A {scene_attributes.get('scene', 'scene')} with {detected_objects[0]}."
        return caption

    objects = ["dog", "ball", "tree"]
    scene = {"scene": "park", "lighting": "bright", "weather": "sunny"}
    print("--- Image Captioning ---")
    caption = generate_caption(objects, scene)
    print(f"  Objects: {objects}")
    print(f"  Scene: {scene}")
    print(f"  Caption: \"{caption}\"")

    # --- 2b. Visual Question Answering ---
    def vqa_answer(image_features, question):
        q_lower = question.lower()
        if any(w in q_lower for w in ["how many", "count"]):
            count = len(image_features.get("objects", []))
            return f"{count} objects detected: {image_features.get('objects', [])}"
        elif any(w in q_lower for w in ["color", "colour"]):
            colors = image_features.get("dominant_colors", ["unknown"])
            return f"Dominant colors: {', '.join(colors[:3])}"
        elif any(w in q_lower for w in ["what", "describe"]):
            return f"Scene: {image_features.get('scene_type', 'unknown')}, objects: {len(image_features.get('objects', []))}"
        elif any(w in q_lower for w in ["where", "location"]):
            return f"Location estimate: {image_features.get('location', 'indoor')}"
        return "Unable to determine answer from visual features."

    image_feat = {
        "objects": ["person", "laptop", "coffee_cup", "desk"],
        "dominant_colors": ["brown", "blue", "white"],
        "scene_type": "office",
        "location": "indoor"
    }
    questions = [
        "How many objects are in the image?",
        "What colors are dominant?",
        "What is this scene?",
        "Where is this?"
    ]
    print("\n--- Visual Question Answering (VQA) ---")
    for q in questions:
        ans = vqa_answer(image_feat, q)
        print(f"  Q: {q}")
        print(f"  A: {ans}")

    # --- 2c. OCR simulation ---
    def simulate_ocr(grid, char_map):
        lines = []
        for row in grid:
            line = ""
            for cell in row:
                line += char_map.get(cell, " ")
            lines.append(line)
        return lines

    char_map = {(1, 0): "H", (2, 0): "E", (3, 0): "L", (4, 0): "O"}
    ocr_grid = [[(1, 0), (2, 0), (3, 0), (3, 0), (4, 0)],
                [(1, 0), (2, 0), (3, 0), (3, 0), (4, 0)],
                [(1, 0), (2, 0), (3, 0), (3, 0), (4, 0)]]
    print("\n--- OCR Simulation ---")
    text = simulate_ocr(ocr_grid, {k: v * 2 for k, v in char_map.items()})
    for line in text:
        print(f"  |{line}|")
    print(f"  Detected text: {'HELLO'}")

    # --- 2d. Region of Interest (ROI) analysis ---
    def analyze_roi(image_grid, roi):
        x1, y1, x2, y2 = roi
        region = [row[y1:y2+1] for row in image_grid[x1:x2+1]]
        flat = [v for row in region for v in row]
        mean_val = sum(flat) / len(flat)
        variance = sum((v - mean_val)**2 for v in flat) / len(flat)
        min_val = min(flat)
        max_val = max(flat)
        return {
            "roi": roi,
            "size": f"{x2-x1+1}x{y2-y1+1}",
            "mean": round(mean_val, 3),
            "variance": round(variance, 3),
            "range": [round(min_val, 3), round(max_val, 3)]
        }

    random.seed(42)
    grid = [[round(random.random(), 3) for _ in range(10)] for _ in range(10)]
    rois = [(0, 0, 4, 4), (5, 5, 9, 9), (0, 5, 4, 9)]
    print("\n--- Region of Interest Analysis ---")
    for roi in rois:
        stats = analyze_roi(grid, roi)
        print(f"  ROI {stats['roi']} ({stats['size']}): mean={stats['mean']}, var={stats['variance']}, range={stats['range']}")

    print()


# =============================================================================
# 3. Audio-Language Models — audio tokens, speech understanding
# =============================================================================

def demo_audio_language():
    print("=" * 70)
    print("DEMO 3: Audio-Language Models — audio tokens, speech understanding")
    print("=" * 70)

    # --- 3a. Audio to tokens (spectrogram simulation) ---
    def audio_to_tokens(n_mels=8, n_frames=12):
        random.seed(42)
        spectrogram = [[round(random.gauss(0, 1), 3) for _ in range(n_mels)] for _ in range(n_frames)]
        # Quantize to discrete tokens
        n_bins = 256
        tokens = []
        for frame in spectrogram:
            token = int(abs(sum(frame)) * 10) % n_bins
            tokens.append(token)
        return spectrogram, tokens

    print("--- Audio to Tokens (Spectrogram Quantization) ---")
    spec, tokens = audio_to_tokens()
    print(f"  Spectrogram shape: {len(spec)} frames x {len(spec[0])} mel bins")
    print(f"  First 5 frames (mel): {[spec[i][:3] for i in range(5)]}")
    print(f"  Token sequence (first 12): {tokens[:12]}")
    print(f"  Vocabulary size: 256")

    # --- 3b. Speech-to-text token decoding ---
    def decode_tokens_to_text(tokens, vocabulary):
        words = []
        for t in tokens:
            if t < len(vocabulary):
                words.append(vocabulary[t])
        # Deduplicate consecutive
        result = [words[0]] if words else []
        for w in words[1:]:
            if w != result[-1]:
                result.append(w)
        return " ".join(result)

    vocab = ["hello", "the", "model", "is", "a", "speech", "to", "text",
             "system", "that", "converts", "audio", "tokens", "words", "end"]
    tokens2 = [0, 0, 5, 7, 9, 11, 12, 13, 14]
    print("\n--- Speech Token Decoding ---")
    print(f"  Tokens: {tokens2}")
    print(f"  Vocabulary (subset): {vocab[:8]}...")
    text = decode_tokens_to_text(tokens2, vocab)
    print(f"  Decoded: \"{text}\"")

    # --- 3c. Audio feature extraction ---
    def extract_features(signal, frame_size=256, hop_size=128):
        frames = []
        for i in range(0, len(signal) - frame_size + 1, hop_size):
            frame = signal[i:i+frame_size]
            # Energy
            energy = sum(x**2 for x in frame) / frame_size
            # Zero-crossing rate
            zcr = sum(1 for j in range(1, len(frame)) if (frame[j] >= 0) != (frame[j-1] >= 0)) / len(frame)
            # Spectral centroid approximation (power spectrum weighted frequency)
            n_fft = min(256, len(frame))
            half = n_fft // 2
            powers = []
            for i in range(half):
                re_part = sum(frame[k] * math.cos(2 * math.pi * i * k / n_fft) for k in range(n_fft))
                im_part = sum(frame[k] * math.sin(2 * math.pi * i * k / n_fft) for k in range(n_fft))
                powers.append(re_part**2 + im_part**2)
            total_power = sum(powers) + 1e-10
            centroid = sum(i * p for i, p in enumerate(powers)) / total_power
            frames.append({
                "energy": round(energy, 4),
                "zcr": round(zcr, 4),
                "spectral_centroid": round(centroid, 2)
            })
        return frames

    random.seed(42)
    signal = [math.sin(2 * math.pi * 440 * i / 16000) + 0.5 * math.sin(2 * math.pi * 880 * i / 16000)
              for i in range(2048)]
    print("\n--- Audio Feature Extraction ---")
    features = extract_features(signal[:1024])
    print(f"  Signal length: {1024} samples")
    print(f"  Frames extracted: {len(features)}")
    for i, f in enumerate(features[:4]):
        print(f"  Frame {i}: energy={f['energy']}, zcr={f['zcr']}, centroid={f['spectral_centroid']}")

    # --- 3d. Audio-language alignment ---
    def align_audio_text(audio_tokens, text_tokens, method="dtw"):
        n = len(audio_tokens)
        m = len(text_tokens)
        # Simple alignment: map audio segments to text tokens
        segment_size = n // m if m > 0 else n
        alignments = []
        for j in range(m):
            start = j * segment_size
            end = min(start + segment_size, n)
            seg_tokens = audio_tokens[start:end]
            confidence = round(random.uniform(0.5, 0.99), 3)
            alignments.append({
                "text_token": text_tokens[j],
                "audio_segment": [start, end],
                "confidence": confidence
            })
        return alignments

    random.seed(42)
    a_tokens = list(range(10, 30))
    t_tokens = ["hello", "world", "test", "audio"]
    print("\n--- Audio-Text Alignment (DTW-style) ---")
    alignments = align_audio_text(a_tokens, t_tokens)
    for a in alignments:
        print(f"  \"{a['text_token']}\" -> audio[{a['audio_segment'][0]}:{a['audio_segment'][1]}] "
              f"conf={a['confidence']}")
    avg_conf = sum(a["confidence"] for a in alignments) / len(alignments)
    print(f"  Average alignment confidence: {avg_conf:.3f}")

    print()


# =============================================================================
# 4. Multi-Modal Fusion — early/late fusion, attention mechanisms
# =============================================================================

def demo_multimodal_fusion():
    print("=" * 70)
    print("DEMO 4: Multi-Modal Fusion — early/late fusion, attention mechanisms")
    print("=" * 70)

    # --- 4a. Early fusion ---
    def early_fusion(image_embs, text_embs):
        # Concatenate all embeddings
        all_embs = image_embs + text_embs
        fused_dim = len(all_embs[0])
        # Average pooling
        fused = [round(sum(e[i] for e in all_embs) / len(all_embs), 3) for i in range(fused_dim)]
        return fused, len(all_embs)

    random.seed(42)
    img_embs = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    txt_embs = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(2)]
    print("--- Early Fusion (Concatenation + Pooling) ---")
    fused, n_total = early_fusion(img_embs, txt_embs)
    print(f"  Image embeddings: {len(img_embs)} x dim={len(img_embs[0])}")
    print(f"  Text embeddings: {len(txt_embs)} x dim={len(txt_embs[0])}")
    print(f"  Total embeddings combined: {n_total}")
    print(f"  Fused representation: {fused}")

    # --- 4b. Late fusion ---
    def late_fusion(modality_scores, weights=None):
        n_classes = len(modality_scores[list(modality_scores.keys())[0]])
        if weights is None:
            weights = {k: 1.0 / len(modality_scores) for k in modality_scores}
        final = [0.0] * n_classes
        for modality, scores in modality_scores.items():
            w = weights[modality]
            final = [round(f + w * s, 3) for f, s in zip(final, scores)]
        # Normalize
        total = sum(final)
        if total > 0:
            final = [round(f / total, 3) for f in final]
        return final

    print("\n--- Late Fusion (Weighted Score Averaging) ---")
    scores = {
        "vision": [0.8, 0.1, 0.1],
        "audio": [0.3, 0.6, 0.1],
        "text": [0.5, 0.3, 0.2]
    }
    weights = {"vision": 0.4, "audio": 0.3, "text": 0.3}
    for mod, s in scores.items():
        print(f"  {mod}: {s} (weight={weights[mod]})")
    fused_scores = late_fusion(scores, weights)
    print(f"  Late fused: {fused_scores}")
    print(f"  Sum: {sum(fused_scores):.3f}")

    # --- 4c. Multi-head attention fusion ---
    def multi_head_attention(queries, keys, values, n_heads=2, d_model=4):
        head_dim = d_model // n_heads
        all_outputs = []
        for h in range(n_heads):
            start = h * head_dim
            end = start + head_dim
            q_head = [q[start:end] for q in queries]
            k_head = [k[start:end] for k in keys]
            v_head = [v[start:end] for v in values]
            # Attention
            scores = []
            for qi in q_head:
                row = [sum(a * b for a, b in zip(qi, kj)) / math.sqrt(head_dim) for kj in k_head]
                max_r = max(row)
                exp_r = [math.exp(s - max_r) for s in row]
                sum_exp = sum(exp_r)
                weights = [e / sum_exp for e in exp_r]
                scores.append(weights)
            # Weighted sum
            outputs = []
            for i, w_row in enumerate(scores):
                out = [round(sum(w * v[j] for w, v in zip(w_row, v_head)), 3)
                       for j in range(head_dim)]
                outputs.append(out)
            all_outputs.append(outputs)
        # Concatenate heads
        concatenated = []
        for i in range(len(queries)):
            concat = []
            for h_out in all_outputs:
                concat.extend(h_out[i])
            concatenated.append(concat)
        return concatenated, scores

    random.seed(42)
    q = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    k = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    v = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    print("\n--- Multi-Head Attention Fusion ---")
    print(f"  Queries: {len(q)}, Keys: {len(k)}, Values: {len(v)}")
    print(f"  n_heads=2, d_model=4, head_dim=2")
    mha_out, attn_weights = multi_head_attention(q, k, v, n_heads=2, d_model=4)
    print(f"  Output shape: {len(mha_out)} x {len(mha_out[0])}")
    print(f"  Attention weights (last head):")
    for i, w in enumerate(attn_weights):
        print(f"    q_{i}: {[round(x, 3) for x in w]}")

    # --- 4d. Cross-modal similarity matrix ---
    def cross_modal_similarity(mod1_embs, mod2_embs):
        matrix = []
        for i, e1 in enumerate(mod1_embs):
            row = []
            for j, e2 in enumerate(mod2_embs):
                dot = sum(a * b for a, b in zip(e1, e2))
                norm1 = math.sqrt(sum(a**2 for a in e1))
                norm2 = math.sqrt(sum(b**2 for b in e2))
                cosine = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
                row.append(round(cosine, 3))
            matrix.append(row)
        return matrix

    print("\n--- Cross-Modal Similarity Matrix ---")
    random.seed(42)
    vis_embs = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    aud_embs = [[round(random.gauss(0, 1), 3) for _ in range(4)] for _ in range(3)]
    sim_matrix = cross_modal_similarity(vis_embs, aud_embs)
    print(f"  Visual embeddings: {len(vis_embs)}, Audio embeddings: {len(aud_embs)}")
    print("  Similarity matrix (rows=visual, cols=audio):")
    header = "        " + "  ".join([f"aud_{j}" for j in range(len(aud_embs))])
    print(header)
    for i, row in enumerate(sim_matrix):
        print(f"  vis_{i}:  {row}")
    best = max(((i, j, sim_matrix[i][j]) for i in range(len(sim_matrix)) for j in range(len(sim_matrix[0]))),
               key=lambda x: x[2])
    print(f"  Best match: vis_{best[0]} <-> aud_{best[1]} (sim={best[2]})")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_vision_language_architecture()
    demo_image_understanding()
    demo_audio_language()
    demo_multimodal_fusion()
