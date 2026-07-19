"""
136 — Cross-Modal Transfer: fine-tuning multimodal models, adapters, few-shot learning

Темы:
  1. Cross-Modal Alignment (shared representations, projection layers)
  2. Multimodal Adapters (visual adapters, cross-attention tuning)
  3. Few-Shot Multimodal Learning (prompt tuning, in-context learning with images)
  4. Knowledge Distillation (teacher-student across modalities)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)

def demo_cross_modal_alignment():
    """
    Section 1: Cross-Modal Alignment
    Shared representations and projection layers
    """
    print("=" * 70)
    print("DEMO 1: Cross-Modal Alignment")
    print("=" * 70)
    
    # 1.1 Shared embedding space simulation
    print("\n[1.1] Shared Embedding Space Simulation")
    print("-" * 40)
    
    # Simulate image and text embeddings in shared space
    embedding_dim = 8
    
    def normalize(vec):
        norm = math.sqrt(sum(v**2 for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec
    
    def cosine_similarity(a, b):
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai**2 for ai in a))
        norm_b = math.sqrt(sum(bi**2 for bi in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
    
    # Create aligned embeddings (image-text pairs should be close)
    random.seed(42)
    
    # Base concepts
    cat_visual = normalize([random.gauss(0.8, 0.1) for _ in range(embedding_dim)])
    cat_text = normalize([v + random.gauss(0, 0.15) for v in cat_visual])
    
    dog_visual = normalize([random.gauss(-0.5, 0.1) for _ in range(embedding_dim)])
    dog_text = normalize([v + random.gauss(0, 0.15) for v in dog_visual])
    
    car_visual = normalize([random.gauss(0.3, 0.2) for _ in range(embedding_dim)])
    car_text = normalize([v + random.gauss(0, 0.15) for v in car_visual])
    
    print("Cross-Modal Similarity Matrix:")
    print(f"{'':>12} {'cat_img':>10} {'cat_txt':>10} {'dog_img':>10} {'dog_txt':>10} {'car_img':>10}")
    
    all_visual = [cat_visual, dog_visual, car_visual]
    all_text = [cat_text, dog_text, car_text]
    names = ['cat_img', 'cat_txt', 'dog_img', 'dog_txt', 'car_img', 'car_txt']
    all_vecs = all_visual + all_text
    
    for i, name_i in enumerate(names):
        print(f"{name_i:>12}", end='')
        for j, name_j in enumerate(names):
            sim = cosine_similarity(all_vecs[i], all_vecs[j])
            print(f"{sim:>10.3f}", end='')
        print()
    
    print("\nExpected: same concept (visual↔text) should have high similarity")
    
    # 1.2 Projection layer simulation
    print("\n[1.2] Projection Layer Simulation")
    print("-" * 40)
    
    # Visual encoder output -> shared space
    visual_dim = 12
    shared_dim = 8
    
    # Random projection matrix
    random.seed(42)
    proj_matrix = [[random.gauss(0, 1/math.sqrt(visual_dim)) 
                     for _ in range(shared_dim)] 
                    for _ in range(visual_dim)]
    
    def project(vec, matrix):
        result = []
        for j in range(len(matrix[0])):
            val = sum(vec[i] * matrix[i][j] for i in range(len(vec)))
            result.append(val)
        return normalize(result)
    
    # Original visual feature
    visual_feature = [random.gauss(0, 1) for _ in range(visual_dim)]
    projected = project(visual_feature, proj_matrix)
    
    print(f"Original visual feature dim: {len(visual_feature)}")
    print(f"Projected to shared space dim: {len(projected)}")
    print(f"Original (first 4): {[f'{v:.3f}' for v in visual_feature[:4]]}")
    print(f"Projected (first 4): {[f'{v:.3f}' for v in projected[:4]]}")
    print("Formula: shared = normalize(W_visual @ visual_feature + b)")
    
    # 1.3 Contrastive loss computation
    print("\n[1.3] Contrastive Loss Computation (InfoNCE)")
    print("-" * 40)
    
    batch_size = 4
    temperature = 0.07
    
    # Create batch of image-text pairs
    random.seed(42)
    image_embeddings = [normalize([random.gauss(0, 1) for _ in range(embedding_dim)]) 
                        for _ in range(batch_size)]
    text_embeddings = [normalize([random.gauss(0, 1) for _ in range(embedding_dim)]) 
                       for _ in range(batch_size)]
    
    # Make positive pairs more similar
    for i in range(batch_size):
        text_embeddings[i] = normalize([image_embeddings[i][j] + random.gauss(0, 0.3) 
                                        for j in range(embedding_dim)])
    
    # Compute similarity matrix
    sim_matrix = []
    for i in range(batch_size):
        row = []
        for j in range(batch_size):
            sim = cosine_similarity(image_embeddings[i], text_embeddings[j])
            row.append(sim / temperature)
        sim_matrix.append(row)
    
    # InfoNCE loss for images
    total_loss = 0
    for i in range(batch_size):
        # Positive pair is at index i
        exp_sims = [math.exp(sim_matrix[i][j]) for j in range(batch_size)]
        sum_exp = sum(exp_sims)
        loss_i = -math.log(exp_sims[i] / sum_exp)
        total_loss += loss_i
    
    avg_loss = total_loss / batch_size
    
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print(f"Similarity matrix (scaled by 1/T):")
    for i, row in enumerate(sim_matrix):
        print(f"  Image {i}: {[f'{s:.2f}' for s in row]}")
    print(f"\nInfoNCE Loss: {avg_loss:.4f}")
    print("Formula: L = -log(exp(s_ii/T) / Σ_j exp(s_ij/T))")
    
    # 1.4 Alignment metrics
    print("\n[1.4] Alignment Metrics")
    print("-" * 40)
    
    # Retrieval accuracy
    def retrieval_accuracy(query_embs, key_embs, top_k=1):
        correct = 0
        for i, query in enumerate(query_embs):
            sims = [cosine_similarity(query, key) for key in key_embs]
            ranked = sorted(range(len(sims)), key=lambda j: -sims[j])
            if i in ranked[:top_k]:
                correct += 1
        return correct / len(query_embs)
    
    r1 = retrieval_accuracy(image_embeddings, text_embeddings, top_k=1)
    r5 = retrieval_accuracy(image_embeddings, text_embeddings, top_k=min(5, batch_size))
    
    print(f"Image-to-Text Retrieval:")
    print(f"  Recall@1: {r1:.3f}")
    print(f"  Recall@{min(5, batch_size)}: {r5:.3f}")
    print(f"\nText-to-Image Retrieval:")
    r1_t2i = retrieval_accuracy(text_embeddings, image_embeddings, top_k=1)
    print(f"  Recall@1: {r1_t2i:.3f}")
    print("Formula: Recall@K = |{queries with correct in top-K}| / |queries|")
    
    print()

def demo_multimodal_adapters():
    """
    Section 2: Multimodal Adapters
    Visual adapters, cross-attention tuning
    """
    print("=" * 70)
    print("DEMO 2: Multimodal Adapters")
    print("=" * 70)
    
    # 2.1 Adapter architecture simulation
    print("\n[2.1] Adapter Architecture Simulation")
    print("-" * 40)
    
    # Bottleneck adapter: down-project -> nonlinearity -> up-project
    input_dim = 64
    adapter_dim = 8
    
    random.seed(42)
    W_down = [[random.gauss(0, 1/math.sqrt(input_dim)) for _ in range(adapter_dim)]
              for _ in range(input_dim)]
    W_up = [[random.gauss(0, 1/math.sqrt(adapter_dim)) for _ in range(input_dim)]
            for _ in range(adapter_dim)]
    
    def relu(vec):
        return [max(0, v) for v in vec]
    
    def adapter_forward(x, W_d, W_u):
        # Down projection
        hidden = []
        for j in range(len(W_d[0])):
            val = sum(x[i] * W_d[i][j] for i in range(len(x)))
            hidden.append(val)
        
        # ReLU
        hidden = relu(hidden)
        
        # Up projection
        output = []
        for j in range(len(W_u[0])):
            val = sum(hidden[i] * W_u[i][j] for i in range(len(hidden)))
            output.append(val)
        
        return output
    
    # Original input
    x = [random.gauss(0, 1) for _ in range(input_dim)]
    adapter_out = adapter_forward(x, W_down, W_up)
    
    # Residual connection
    adapted = [x[i] + adapter_out[i] for i in range(input_dim)]
    
    # Count parameters
    adapter_params = input_dim * adapter_dim + adapter_dim * input_dim
    original_params = input_dim * input_dim  #假设全连接层
    
    print(f"Adapter Configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Bottleneck dim: {adapter_dim}")
    print(f"  Adapter parameters: {adapter_params}")
    print(f"  Original layer parameters: {original_params}")
    print(f"  Parameter efficiency: {adapter_params/original_params*100:.1f}%")
    print("Formula: adapter(x) = x + W_up · ReLU(W_down · x)")
    
    # 2.2 Cross-attention mechanism
    print("\n[2.2] Cross-Attention Mechanism")
    print("-" * 40)
    
    # Simulate cross-attention: text attending to image features
    seq_len_text = 4
    seq_len_image = 6
    d_model = 8
    
    random.seed(42)
    
    # Text queries
    text_features = [[random.gauss(0, 1) for _ in range(d_model)] 
                     for _ in range(seq_len_text)]
    # Image keys and values
    image_keys = [[random.gauss(0, 1) for _ in range(d_model)] 
                  for _ in range(seq_len_image)]
    image_values = [[random.gauss(0, 1) for _ in range(d_model)] 
                    for _ in range(seq_len_image)]
    
    def dot_product_attention(query, keys, values, temperature=1.0):
        scores = []
        for key in keys:
            score = sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
            scores.append(score / temperature)
        
        # Softmax
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        weights = [e / sum_exp for e in exp_scores]
        
        # Weighted sum of values
        output = [0.0] * len(values[0])
        for w, v in zip(weights, values):
            for i in range(len(output)):
                output[i] += w * v[i]
        
        return output, weights
    
    print("Cross-Attention: Text Query → Image Key/Value")
    print(f"Text sequence length: {seq_len_text}")
    print(f"Image sequence length: {seq_len_image}")
    
    for i, (query, text_feat) in enumerate(zip(range(seq_len_text), text_features)):
        output, weights = dot_product_attention(text_feat, image_keys, image_values)
        print(f"\nText token {i}:")
        print(f"  Attention weights: {[f'{w:.3f}' for w in weights]}")
        print(f"  Output (first 4): {[f'{v:.3f}' for v in output[:4]]}")
    
    print("\nFormula: CrossAttn(Q, K, V) = softmax(QK^T / √d) · V")
    
    # 2.3 Adapter placement strategies
    print("\n[2.3] Adapter Placement Strategies")
    print("-" * 40)
    
    # Simulate transformer layers
    num_layers = 6
    strategies = {
        'parallel': 'Adapter runs alongside original layer',
        'serial_after': 'Adapter after original layer (post-norm)',
        'serial_before': 'Adapter before original layer (pre-norm)',
        'bottleneck_attn': 'Adapter only in attention blocks',
        'bottleneck_ffn': 'Adapter only in feed-forward blocks',
    }
    
    print("Adapter Placement Strategies in Transformer:")
    print("-" * 50)
    
    for name, desc in strategies.items():
        if 'parallel' in name:
            params_per_layer = 2 * 64 * 8  # Down + Up
            overhead = "Low"
        elif 'serial' in name:
            params_per_layer = 2 * 64 * 8
            overhead = "Medium"
        else:
            params_per_layer = 64 * 8  # Half the parameters
            overhead = "Very Low"
        
        total_params = params_per_layer * num_layers
        print(f"\n{name}:")
        print(f"  Description: {desc}")
        print(f"  Parameters per layer: {params_per_layer}")
        print(f"  Total adapter params: {total_params}")
        print(f"  Inference overhead: {overhead}")
    
    # 2.4 Multi-adapter fusion
    print("\n[2.4] Multi-Adapter Fusion")
    print("-" * 40)
    
    # Multiple task-specific adapters
    tasks = ['VQA', 'Captioning', 'Retrieval']
    adapter_outputs = {}
    
    random.seed(42)
    for task in tasks:
        adapter_outputs[task] = [random.gauss(0, 1) for _ in range(8)]
    
    # Fusion strategies
    def mean_fusion(outputs):
        fused = [0.0] * len(outputs[0])
        for out in outputs:
            for i in range(len(fused)):
                fused[i] += out[i] / len(outputs)
        return fused
    
    def attention_fusion(outputs, query):
        weights = []
        for out in outputs:
            sim = sum(q * o for q, o in zip(query, out))
            weights.append(sim)
        
        max_w = max(weights)
        exp_w = [math.exp(w - max_w) for w in weights]
        sum_exp = sum(exp_w)
        weights = [e / sum_exp for e in exp_w]
        
        fused = [0.0] * len(outputs[0])
        for w, out in zip(weights, outputs):
            for i in range(len(fused)):
                fused[i] += w * out[i]
        
        return fused, weights
    
    query = [random.gauss(0, 1) for _ in range(8)]
    
    print("Multi-Adapter Fusion Strategies:")
    print("-" * 40)
    
    # Mean fusion
    mean_fused = mean_fusion(list(adapter_outputs.values()))
    print("\n1. Mean Fusion:")
    print(f"   Output (first 4): {[f'{v:.3f}' for v in mean_fused[:4]]}")
    
    # Attention fusion
    attn_fused, attn_weights = attention_fusion(list(adapter_outputs.values()), query)
    print("\n2. Attention Fusion:")
    print(f"   Weights: {dict(zip(tasks, [f'{w:.3f}' for w in attn_weights]))}")
    print(f"   Output (first 4): {[f'{v:.3f}' for v in attn_fused[:4]]}")
    
    print("\nFormula: mean_fused = (1/N) Σ adapter_i(x)")
    print("Formula: attn_fused = Σ softmax(q·adapter_i(x)/T) · adapter_i(x)")
    
    print()

def demo_few_shot_multimodal():
    """
    Section 3: Few-Shot Multimodal Learning
    Prompt tuning, in-context learning with images
    """
    print("=" * 70)
    print("DEMO 3: Few-Shot Multimodal Learning")
    print("=" * 70)
    
    # 3.1 Visual prompt tuning
    print("\n[3.1] Visual Prompt Tuning")
    print("-" * 40)
    
    # Learnable visual prompts prepended to image tokens
    num_image_tokens = 10
    num_learnable_prompts = 3
    token_dim = 8
    
    random.seed(42)
    
    # Fixed image tokens
    image_tokens = [[random.gauss(0, 1) for _ in range(token_dim)] 
                    for _ in range(num_image_tokens)]
    
    # Learnable prompts (initialized randomly)
    visual_prompts = [[random.gauss(0, 0.02) for _ in range(token_dim)] 
                      for _ in range(num_learnable_prompts)]
    
    # Simulated gradients and update
    learning_rate = 0.1
    gradient_scale = 0.5
    
    print(f"Image tokens: {num_image_tokens}")
    print(f"Learnable visual prompts: {num_learnable_prompts}")
    print(f"Token dimension: {token_dim}")
    print(f"\nInitial prompts (first 2 dims):")
    for i, p in enumerate(visual_prompts):
        print(f"  Prompt {i}: {[f'{v:.4f}' for v in p[:2]]}")
    
    # Simulate training update
    for epoch in range(3):
        for i in range(num_learnable_prompts):
            grad = [random.gauss(0, gradient_scale) for _ in range(token_dim)]
            visual_prompts[i] = [p - learning_rate * g 
                                for p, g in zip(visual_prompts[i], grad)]
    
    print(f"\nAfter 3 epochs of training:")
    for i, p in enumerate(visual_prompts):
        print(f"  Prompt {i}: {[f'{v:.4f}' for v in p[:2]]}")
    
    # Final sequence: [prompts | image_tokens]
    full_sequence = visual_prompts + image_tokens
    print(f"\nFull sequence length: {len(full_sequence)} ({num_learnable_prompts} prompts + {num_image_tokens} image tokens)")
    print("Formula: sequence = [learnable_prompts; image_tokens]")
    
    # 3.2 In-context learning with examples
    print("\n[3.2] In-Context Learning with Examples")
    print("-" * 40)
    
    # Few-shot examples: image-text pairs
    few_shot_examples = [
        {'image_features': [0.8, 0.2, 0.1, 0.9], 'text': 'a red apple', 'label': 'fruit'},
        {'image_features': [0.1, 0.7, 0.3, 0.2], 'text': 'a blue sky', 'label': 'nature'},
        {'image_features': [0.9, 0.8, 0.1, 0.1], 'text': 'an orange carrot', 'label': 'vegetable'},
    ]
    
    # Query image
    query_features = [0.7, 0.3, 0.1, 0.8]
    
    def compute_similarity(a, b):
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai**2 for ai in a))
        norm_b = math.sqrt(sum(bi**2 for bi in b))
        return dot / (norm_a * norm_b)
    
    print("Few-Shot Examples:")
    for i, ex in enumerate(few_shot_examples):
        print(f"  Example {i+1}: {ex['text']} -> {ex['label']}")
        print(f"    Features: {[f'{v:.2f}' for v in ex['image_features']]}")
    
    print(f"\nQuery image features: {[f'{v:.2f}' for v in query_features]}")
    
    # Nearest neighbor classification
    similarities = []
    for ex in few_shot_examples:
        sim = compute_similarity(query_features, ex['image_features'])
        similarities.append((sim, ex['label'], ex['text']))
    
    similarities.sort(reverse=True)
    
    print("\nSimilarity ranking:")
    for sim, label, text in similarities:
        print(f"  {text}: similarity = {sim:.3f}")
    
    predicted = similarities[0][1]
    print(f"\nPredicted class (1-NN): {predicted}")
    print("Formula: predict = argmax_x similarity(query, example_x)")
    
    # 3.3 Prompt engineering for VLMs
    print("\n[3.3] Prompt Engineering for VLMs")
    print("-" * 40)
    
    # Different prompt templates
    templates = {
        'zero_shot': "What is in this image?",
        'one_shot': "Image: [cat photo]\nAnswer: a cat\n\nImage: [query]\nAnswer:",
        'chain_of_thought': "Let's analyze this image step by step.\n1. First, I see...",
        'task_specific': "Classify the main object in this image as one of: {classes}",
    }
    
    # Simulated confidence scores for each template
    random.seed(42)
    print("Prompt Template Comparison:")
    print("-" * 40)
    
    for name, template in templates.items():
        # Simulate model confidence
        confidence = random.uniform(0.6, 0.95)
        latency = random.uniform(10, 100)
        
        print(f"\n{name}:")
        print(f"  Template: {template[:50]}...")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Latency: {latency:.0f}ms")
    
    print("\nBest template varies by task complexity and model capability")
    
    # 3.4 Meta-learning for multimodal tasks
    print("\n[3.4] Meta-Learning for Multimodal Tasks")
    print("-" * 40)
    
    # Simulate MAML-style meta-learning
    meta_lr = 0.01
    inner_lr = 0.005
    num_tasks = 5
    
    random.seed(42)
    
    # Meta-parameters (shared initialization)
    meta_params = [random.gauss(0, 0.1) for _ in range(4)]
    
    print("MAML-style Meta-Learning for Multimodal Tasks")
    print(f"Meta LR: {meta_lr}, Inner LR: {inner_lr}")
    print(f"Number of tasks: {num_tasks}")
    
    task_performances = []
    
    for task_id in range(num_tasks):
        # Task-specific data
        support_set_size = random.randint(1, 5)
        
        # Inner loop adaptation
        adapted_params = list(meta_params)
        for _ in range(3):  # 3 gradient steps
            grad = [random.gauss(0, 0.1) for _ in range(4)]
            adapted_params = [p - inner_lr * g 
                             for p, g in zip(adapted_params, grad)]
        
        # Evaluate on query set
        performance = random.uniform(0.6, 0.9)
        task_performances.append(performance)
        
        print(f"\nTask {task_id + 1}:")
        print(f"  Support set size: {support_set_size}")
        print(f"  Adapted params (first 2): {[f'{p:.4f}' for p in adapted_params[:2]]}")
        print(f"  Performance: {performance:.3f}")
    
    avg_performance = sum(task_performances) / len(task_performances)
    print(f"\nAverage meta-learned performance: {avg_performance:.3f}")
    print("Formula: θ* = θ - α∇_θ Σ_T L_T(f_θ'_T, query_T)")
    
    print()

def demo_knowledge_distillation():
    """
    Section 4: Knowledge Distillation
    Teacher-student across modalities
    """
    print("=" * 70)
    print("DEMO 4: Knowledge Distillation")
    print("=" * 70)
    
    # 4.1 Teacher-student architecture
    print("\n[4.1] Teacher-Student Architecture")
    print("-" * 40)
    
    # Teacher: large multimodal model
    teacher_dim = 64
    student_dim = 16
    
    random.seed(42)
    
    # Teacher embeddings
    teacher_visual = [random.gauss(0, 1) for _ in range(teacher_dim)]
    teacher_text = [random.gauss(0, 1) for _ in range(teacher_dim)]
    
    # Student embeddings (smaller)
    student_visual = [random.gauss(0, 1) for _ in range(student_dim)]
    student_text = [random.gauss(0, 1) for _ in range(student_dim)]
    
    # Projection layers for distillation
    teacher_to_student_V = [[random.gauss(0, 1/math.sqrt(teacher_dim)) 
                             for _ in range(student_dim)]
                            for _ in range(teacher_dim)]
    teacher_to_student_T = [[random.gauss(0, 1/math.sqrt(teacher_dim)) 
                             for _ in range(student_dim)]
                            for _ in range(teacher_dim)]
    
    def project_down(vec, matrix):
        result = []
        for j in range(len(matrix[0])):
            val = sum(vec[i] * matrix[i][j] for i in range(len(vec)))
            result.append(val)
        return result
    
    def cosine_sim(a, b):
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai**2 for ai in a))
        norm_b = math.sqrt(sum(bi**2 for bi in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
    
    # Project teacher to student space
    teacher_V_proj = project_down(teacher_visual, teacher_to_student_V)
    teacher_T_proj = project_down(teacher_text, teacher_to_student_T)
    
    print("Teacher-Student Configuration:")
    print(f"  Teacher embedding dim: {teacher_dim}")
    print(f"  Student embedding dim: {student_dim}")
    print(f"  Compression ratio: {teacher_dim/student_dim:.1f}x")
    print(f"\nCross-modal similarities (after projection):")
    print(f"  Teacher V-T: {cosine_sim(teacher_visual, teacher_text):.3f}")
    print(f"  Student V-T: {cosine_sim(student_visual, student_text):.3f}")
    print(f"  Teacher proj vs Student: {cosine_sim(teacher_V_proj, student_visual):.3f}")
    
    print("\nFormula: student_loss = L_task + α·L_distill")
    
    # 4.2 Distillation loss components
    print("\n[4.2] Distillation Loss Components")
    print("-" * 40)
    
    # Soft targets from teacher
    num_classes = 4
    temperature = 3.0
    
    # Teacher logits
    teacher_logits = [random.gauss(0, 2) for _ in range(num_classes)]
    
    # Student logits
    student_logits = [random.gauss(0, 1) for _ in range(num_classes)]
    
    def softmax(logits, temp):
        scaled = [l / temp for l in logits]
        max_l = max(scaled)
        exp_l = [math.exp(l - max_l) for l in scaled]
        sum_exp = sum(exp_l)
        return [e / sum_exp for e in exp_l]
    
    def kl_divergence(p, q):
        kl = 0
        for pi, qi in zip(p, q):
            if pi > 0 and qi > 0:
                kl += pi * math.log(pi / qi)
        return kl
    
    teacher_probs = softmax(teacher_logits, temperature)
    student_probs = softmax(student_logits, temperature)
    
    # Hard label (ground truth)
    true_label = 2
    hard_target = [1.0 if i == true_label else 0.0 for i in range(num_classes)]
    
    # Loss components
    alpha = 0.7  # Distillation weight
    
    # Cross-entropy with hard labels
    ce_loss = -math.log(student_probs[true_label] + 1e-8)
    
    # KL divergence with soft targets
    kl_loss = kl_divergence(teacher_probs, student_probs)
    
    # Combined loss
    total_loss = (1 - alpha) * ce_loss + alpha * (temperature ** 2) * kl_loss
    
    print(f"Temperature: {temperature}")
    print(f"Distillation weight (α): {alpha}")
    print(f"\nTeacher logits: {[f'{l:.2f}' for l in teacher_logits]}")
    print(f"Teacher probs: {[f'{p:.3f}' for p in teacher_probs]}")
    print(f"\nStudent logits: {[f'{l:.2f}' for l in student_logits]}")
    print(f"Student probs: {[f'{p:.3f}' for p in student_probs]}")
    print(f"\nLoss Components:")
    print(f"  CE (hard labels): {ce_loss:.4f}")
    print(f"  KL (soft targets): {kl_loss:.4f}")
    print(f"  Total loss: {total_loss:.4f}")
    print("Formula: L = (1-α)·CE(y, ŷ) + α·T²·KL(p_teacher || p_student)")
    
    # 4.3 Feature-based distillation
    print("\n[4.3] Feature-Based Distillation")
    print("-" * 40)
    
    # Intermediate features from teacher and student
    teacher_features = {
        'layer1': [random.gauss(0, 1) for _ in range(32)],
        'layer2': [random.gauss(0, 1) for _ in range(32)],
        'layer3': [random.gauss(0, 1) for _ in range(32)],
    }
    
    student_features = {
        'layer1': [random.gauss(0, 1) for _ in range(8)],
        'layer2': [random.gauss(0, 1) for _ in range(8)],
        'layer3': [random.gauss(0, 1) for _ in range(8)],
    }
    
    # Feature projection matrices
    feature_projs = {}
    for layer in teacher_features:
        feature_projs[layer] = [[random.gauss(0, 1/math.sqrt(32)) 
                                 for _ in range(8)]
                                for _ in range(32)]
    
    def feature_loss(teacher_feat, student_feat, proj_matrix):
        projected = project_down(teacher_feat, proj_matrix)
        # MSE loss
        mse = sum((p - s)**2 for p, s in zip(projected, student_feat)) / len(student_feat)
        return mse
    
    print("Feature Distillation (per layer):")
    total_feature_loss = 0
    
    for layer in teacher_features:
        loss = feature_loss(teacher_features[layer], 
                           student_features[layer], 
                           feature_projs[layer])
        total_feature_loss += loss
        print(f"  {layer}: MSE = {loss:.4f}")
    
    print(f"\nTotal feature loss: {total_feature_loss:.4f}")
    print("Formula: L_feat = Σ_l MSE(Proj(T_l), S_l)")
    
    # 4.4 Modality-specific distillation
    print("\n[4.4] Modality-Specific Distillation")
    print("-" * 40)
    
    # Distill visual and textual knowledge separately
    distillation_config = {
        'visual': {
            'teacher_dim': 512,
            'student_dim': 128,
            'loss_weight': 1.0,
            'description': 'Visual backbone distillation'
        },
        'textual': {
            'teacher_dim': 768,
            'student_dim': 192,
            'loss_weight': 0.8,
            'description': 'Text encoder distillation'
        },
        'fusion': {
            'teacher_dim': 1024,
            'student_dim': 256,
            'loss_weight': 1.2,
            'description': 'Multimodal fusion distillation'
        }
    }
    
    print("Modality-Specific Distillation Configuration:")
    print("-" * 50)
    
    total_params_saved = 0
    for modality, config in distillation_config.items():
        teacher_params = config['teacher_dim'] ** 2  # Approximate
        student_params = config['student_dim'] ** 2
        saved = teacher_params - student_params
        total_params_saved += saved
        
        compression = teacher_params / student_params
        
        print(f"\n{modality.upper()} ({config['description']}):")
        print(f"  Teacher dim: {config['teacher_dim']}")
        print(f"  Student dim: {config['student_dim']}")
        print(f"  Compression: {compression:.1f}x")
        print(f"  Loss weight: {config['loss_weight']}")
        print(f"  Params saved: {saved:,}")
    
    print(f"\nTotal parameters saved: {total_params_saved:,}")
    print("Formula: L_total = Σ_m w_m · L_distill(modality_m)")
    
    print()

if __name__ == "__main__":
    demo_cross_modal_alignment()
    demo_multimodal_adapters()
    demo_few_shot_multimodal()
    demo_knowledge_distillation()