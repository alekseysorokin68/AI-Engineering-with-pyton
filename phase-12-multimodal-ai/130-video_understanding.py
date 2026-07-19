"""
130 — Video Understanding: temporal modeling, action recognition, video QA

Темы:
  1. Video as Frame Sequences (temporal dimension, frame sampling strategies)
  2. Temporal Modeling (3D convolutions concept, temporal attention, time-series pooling)
  3. Action Recognition (spatial + temporal features, two-stream approach)
  4. Video QA (question answering over video, temporal grounding)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ---------------------------------------------------------------------------
# 1. Video as Frame Sequences — frame sampling and temporal structure
# ---------------------------------------------------------------------------
def demo_frame_sequences():
    """Video = ordered frames; sampling strategies affect what models see."""
    print("=" * 70)
    print("DEMO 1: Video as Frame Sequences — temporal dimension & sampling")
    print("=" * 70)

    total_frames = 120  # e.g. 4 seconds at 30fps

    # --- 1a. Uniform sampling: pick N evenly-spaced frames ---
    n_samples = 8
    step = total_frames / n_samples
    uniform_frames = [int(i * step) for i in range(n_samples)]
    print(f"\n--- 1a. Uniform sampling: {n_samples} frames from {total_frames} ---")
    print(f"  Step = {total_frames}/{n_samples} = {step:.1f}")
    print(f"  Selected frames: {uniform_frames}")

    # --- 1b. Random temporal jitter (data augmentation) ---
    jitter_range = 2
    jittered = [f + random.randint(-jitter_range, jitter_range) for f in uniform_frames]
    jittered = [max(0, min(total_frames - 1, f)) for f in jittered]
    print(f"\n--- 1b. Random temporal jitter (±{jitter_range}) ---")
    print(f"  Original: {uniform_frames}")
    print(f"  Jittered: {jittered}")

    # --- 1c. Keyframe-based sampling (detect scene changes) ---
    # Simulate scene change scores (higher = more likely a scene boundary)
    scene_scores = [random.random() for _ in range(total_frames)]
    # Inject a scene change
    scene_scores[35] = 0.95
    scene_scores[72] = 0.88
    threshold = 0.85
    keyframes = [i for i, s in enumerate(scene_scores) if s > threshold]
    print(f"\n--- 1c. Keyframe sampling (threshold={threshold}) ---")
    print(f"  Detected {len(keyframes)} keyframes at: {keyframes}")
    print(f"  Keyframes capture scene transitions → more informative frames.")

    # --- 1d. Dense vs sparse sampling trade-offs ---
    print(f"\n--- 1d. Sampling strategy comparison ---")
    print(f"  {'Strategy':<25} {'Frames':<10} {'Temp. info':<15} {'Cost':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*15} {'-'*10}")
    strategies = [
        ("Single frame", 1, "None", "Very low"),
        ("Uniform (4)", 4, "Low", "Low"),
        ("Uniform (16)", 16, "Medium", "Medium"),
        ("Dense (every)", total_frames, "Full", "Very high"),
        ("Keyframe-based", len(keyframes), "Adaptive", "Medium"),
    ]
    for strat, frames, tinfo, cost in strategies:
        print(f"  {strat:<25} {frames:<10} {tinfo:<15} {cost:<10}")


# ---------------------------------------------------------------------------
# 2. Temporal Modeling — 3D convs, attention, pooling
# ---------------------------------------------------------------------------
def demo_temporal_modeling():
    """Methods to aggregate temporal information across video frames."""
    print("\n" + "=" * 70)
    print("DEMO 2: Temporal Modeling — 3D convolutions, attention, pooling")
    print("=" * 70)

    n_frames = 8
    feat_dim = 6
    random.seed(42)
    features = [[random.gauss(0, 1) for _ in range(feat_dim)] for _ in range(n_frames)]

    # --- 2a. Temporal average pooling (simplest baseline) ---
    avg_pooled = [0.0] * feat_dim
    for frame in features:
        for d in range(feat_dim):
            avg_pooled[d] += frame[d]
    avg_pooled = [v / n_frames for v in avg_pooled]

    print(f"\n--- 2a. Temporal Average Pooling ---")
    print(f"  Input: {n_frames} frames × {feat_dim} features")
    print(f"  Pooled: {[f'{v:.4f}' for v in avg_pooled]}")
    print(f"  Formula: h = (1/T) * sum_t(f_t)")
    print(f"  Pro: simple, no params. Con: loses all temporal order.")

    # --- 2b. 3D Convolution concept (kernel_size along time) ---
    def conv1d_temporal(features, kernel_size=3):
        """Simulate 1D temporal convolution along the time axis."""
        output = []
        for t in range(len(features) - kernel_size + 1):
            window = features[t:t + kernel_size]
            # Simple "convolution" with random weights
            out_feat = []
            for d in range(len(features[0])):
                val = sum(window[k][d] * (1.0 / kernel_size) for k in range(kernel_size))
                out_feat.append(val)
            output.append(out_feat)
        return output

    conv_out = conv1d_temporal(features, kernel_size=3)
    print(f"\n--- 2b. 3D Convolution (kernel_size=3 over time) ---")
    print(f"  Input:  {n_frames} frames × {feat_dim} features")
    print(f"  Output: {len(conv_out)} frames × {feat_dim} features")
    print(f"  3D conv: kernel slides over (time × height × width)")
    print(f"  Captures local temporal patterns (motion, short actions)")

    # --- 2c. Temporal self-attention ---
    def scaled_dot_product_attention(Q, K, V):
        """Compute self-attention over temporal sequence."""
        d_k = len(Q[0])
        scores = []
        for i in range(len(Q)):
            row = []
            for j in range(len(K)):
                dot = sum(Q[i][k] * K[j][k] for k in range(d_k))
                row.append(dot / math.sqrt(d_k))
            scores.append(row)

        # Softmax per row
        attn_weights = []
        for row in scores:
            max_val = max(row)
            exp_row = [math.exp(s - max_val) for s in row]
            total = sum(exp_row)
            attn_weights.append([e / total for e in exp_row])

        # Weighted sum of values
        output = []
        for i in range(len(Q)):
            out = [0.0] * len(V[0])
            for j in range(len(V)):
                for d in range(len(V[0])):
                    out[d] += attn_weights[i][j] * V[j][d]
            output.append(out)
        return output, attn_weights

    Q = [features[i][:3] for i in range(n_frames)]  # dim=3 for speed
    K = [features[i][:3] for i in range(n_frames)]
    V = features[:][:]

    attn_out, attn_w = scaled_dot_product_attention(Q, K, V)
    print(f"\n--- 2c. Temporal Self-Attention ---")
    print(f"  Each frame attends to all other frames.")
    print(f"  Attention weight at t=0 → t=3: {attn_w[0][3]:.4f}")
    print(f"  Attention weight at t=0 → t=0 (self): {attn_w[0][0]:.4f}")
    print(f"  Pro: captures long-range dependencies. Con: O(T²) cost.")

    # --- 2d. Max pooling over time (picks strongest activation) ---
    max_pooled = [-float('inf')] * feat_dim
    max_frame_idx = [0] * feat_dim
    for t, frame in enumerate(features):
        for d in range(feat_dim):
            if frame[d] > max_pooled[d]:
                max_pooled[d] = frame[d]
                max_frame_idx[d] = t

    print(f"\n--- 2d. Temporal Max Pooling ---")
    print(f"  Pooled: {[f'{v:.4f}' for v in max_pooled]}")
    print(f"  Winning frame per dim: {max_frame_idx}")
    print(f"  Pro: captures most salient moment. Con: single-frame information.")


# ---------------------------------------------------------------------------
# 3. Action Recognition — two-stream and spatial+temporal features
# ---------------------------------------------------------------------------
def demo_action_recognition():
    """Recognize actions by combining spatial (appearance) and temporal (motion)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Action Recognition — spatial + temporal, two-stream")
    print("=" * 70)

    actions = ["running", "jumping", "sitting", "eating", "waving"]
    random.seed(42)

    # --- 3a. Simulated spatial features (what objects are present) ---
    def extract_spatial_features(action):
        """Simulate CNN features from a single frame (appearance)."""
        random.seed(hashlib.md5(f"spatial_{action}".encode()).hexdigest()[:8], version=1)
        return [random.gauss(0, 1) for _ in range(5)]

    print(f"\n--- 3a. Spatial features (single frame appearance) ---")
    spatial_feats = {}
    for action in actions:
        feats = extract_spatial_features(action)
        spatial_feats[action] = feats
        print(f"  {action:<12}: {[f'{v:+.3f}' for v in feats]}")
    print("  Spatial stream sees: person, objects, background, pose.")

    # --- 3b. Simulated temporal features (optical flow / motion) ---
    def extract_temporal_features(action):
        """Simulate features from optical flow or 3D conv (motion)."""
        random.seed(hashlib.md5(f"temporal_{action}".encode()).hexdigest()[:8], version=1)
        return [random.gauss(0, 1) for _ in range(5)]

    print(f"\n--- 3b. Temporal features (motion / optical flow) ---")
    temporal_feats = {}
    for action in actions:
        feats = extract_temporal_features(action)
        temporal_feats[action] = feats
        print(f"  {action:<12}: {[f'{v:+.3f}' for v in feats]}")
    print("  Temporal stream sees: motion direction, speed, trajectory.")

    # --- 3c. Two-stream fusion (late fusion via concatenation) ---
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a)) or 1e-8
        nb = math.sqrt(sum(x**2 for x in b)) or 1e-8
        return dot / (na * nb)

    test_action = "running"
    test_spatial = spatial_feats[test_action]
    test_temporal = temporal_feats[test_action]

    print(f"\n--- 3c. Two-stream late fusion for '{test_action}' ---")
    print(f"  Spatial cosine similarities:")
    for action in actions:
        sim = cosine_sim(test_spatial, spatial_feats[action])
        print(f"    vs {action:<12}: {sim:+.4f}")
    print(f"  Temporal cosine similarities:")
    for action in actions:
        sim = cosine_sim(test_temporal, temporal_feats[action])
        print(f"    vs {action:<12}: {sim:+.4f}")

    # --- 3d. Final fused prediction ---
    alpha = 0.5  # fusion weight
    print(f"\n--- 3d. Fused prediction (spatial weight={alpha}, temporal={1-alpha}) ---")
    scores = []
    for action in actions:
        s_sim = cosine_sim(test_spatial, spatial_feats[action])
        t_sim = cosine_sim(test_temporal, temporal_feats[action])
        fused = alpha * s_sim + (1 - alpha) * t_sim
        scores.append((action, fused))
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Action':<12} {'Spatial':>10} {'Temporal':>10} {'Fused':>10} {'Rank':>6}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for rank, (action, fused) in enumerate(scores, 1):
        s_sim = cosine_sim(test_spatial, spatial_feats[action])
        t_sim = cosine_sim(test_temporal, temporal_feats[action])
        print(f"  {action:<12} {s_sim:10.4f} {t_sim:10.4f} {fused:10.4f} {rank:6d}")
    print(f"  Predicted: {scores[0][0]} (highest fused score)")


# ---------------------------------------------------------------------------
# 4. Video QA — question answering over video content
# ---------------------------------------------------------------------------
def demo_video_qa():
    """Answer questions about video content by grounding temporal evidence."""
    print("\n" + "=" * 70)
    print("DEMO 4: Video QA — temporal grounding and question answering")
    print("=" * 70)

    random.seed(42)

    # Simulated video: each frame has detected objects + actions
    n_frames = 10
    frame_data = []
    objects_pool = ["person", "ball", "cup", "chair", "dog", "book"]
    actions_pool = ["walking", "throwing", "sitting", "reading", "running"]

    for i in range(n_frames):
        objs = random.sample(objects_pool, k=random.randint(1, 3))
        act = random.choice(actions_pool)
        frame_data.append({"frame": i, "objects": objs, "action": act})

    print(f"\n--- 4a. Video content representation (10 frames) ---")
    print(f"  {'Frame':>5}  {'Objects':<30}  {'Action':<12}")
    print(f"  {'-'*5}  {'-'*30}  {'-'*12}")
    for fd in frame_data:
        print(f"  {fd['frame']:5d}  {str(fd['objects']):<30}  {fd['action']:<12}")

    # --- 4b. Simple question parser ---
    def parse_video_question(question):
        """Extract intent and entities from a video question."""
        q = question.lower()
        words = [re.sub(r'[^a-z]', '', w) for w in q.split()]
        if "what" in q and "action" in q:
            return "temporal_action", None
        elif "when" in q:
            entities = [w for w in words if w in objects_pool]
            return "temporal_localize", entities[0] if entities else None
        elif "how many" in q and "frame" in q:
            entity = [w for w in words if w in objects_pool]
            return "count_frames", entity[0] if entity else None
        elif "where" in q or "which" in q:
            return "spatial_localize", None
        return "unknown", None

    questions = [
        "What action is happening in the video?",
        "When does the ball appear?",
        "How many frames contain a person?",
    ]

    # --- 4c. Temporal action localization ---
    print(f"\n--- 4b-c. Video QA processing ---")
    for q in questions:
        intent, entity = parse_video_question(q)
        print(f"\n  Q: '{q}'")
        print(f"  Intent: {intent}, Entity: {entity}")

        if intent == "temporal_action":
            action_counts = collections.Counter(fd["action"] for fd in frame_data)
            top_action, top_count = action_counts.most_common(1)[0]
            print(f"  A: The dominant action is '{top_action}' ({top_count}/{n_frames} frames)")

        elif intent == "temporal_localize" and entity:
            matching = [fd["frame"] for fd in frame_data if entity in fd["objects"]]
            if matching:
                print(f"  A: '{entity}' appears in frames {matching}")
                print(f"     First appearance: frame {matching[0]}, last: frame {matching[-1]}")
            else:
                print(f"  A: '{entity}' not found in any frame")

        elif intent == "count_frames" and entity:
            count = sum(1 for fd in frame_data if entity in fd["objects"])
            print(f"  A: '{entity}' appears in {count} out of {n_frames} frames")

    # --- 4d. Attention-based frame importance ---
    print(f"\n--- 4d. Frame importance via attention scoring ---")
    # Score frames by how many unique objects they contain
    all_objects = set()
    frame_scores = []
    for fd in frame_data:
        new = len(set(fd["objects"]) - all_objects)
        all_objects.update(fd["objects"])
        frame_scores.append((fd["frame"], new))

    print(f"  {'Frame':>5}  {'New objects':>11}  {'Importance':>11}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*11}")
    for frame, score in frame_scores:
        bar = "█" * max(0, score * 3)
        print(f"  {frame:5d}  {score:11d}  {bar}")
    important = [f for f, s in frame_scores if s > 0]
    print(f"  Key frames (introducing new objects): {important}")
    print(f"  → These frames would receive highest attention weight in a Video QA model.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_frame_sequences()
    demo_temporal_modeling()
    demo_action_recognition()
    demo_video_qa()
    print("\n" + "=" * 70)
    print("All 4 demos completed — 130-video_understanding.py")
    print("=" * 70)
