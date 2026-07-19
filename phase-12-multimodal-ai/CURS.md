# Phase 12: Multimodal AI

> Мультимодальные модели — от vision transformers до production приложений.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 126 | [Vision Transformers](#урок-126-vision-transformers) | [Код](126-vision_transformers.py) |
| 127 | [CLIP & Alignment](#урок-127-clip--vision-language-alignment) | [Код](127-clip_alignment.py) |
| 128 | [Image Generation](#урок-128-image-generation) | [Код](128-image_generation.py) |
| 129 | [Diffusion Advanced](#урок-129-diffusion-models-advanced) | [Код](129-diffusion_advanced.py) |
| 130 | [Video Understanding](#урок-130-video-understanding) | [Код](130-video_understanding.py) |
| 131 | [Audio & Speech](#урок-131-audio--speech-models) | [Код](131-audio_speech_models.py) |
| 132 | [Document Understanding](#урок-132-document-understanding) | [Код](132-document_understanding.py) |
| 133 | [3D Vision](#урок-133-3d-vision) | [Код](133-3d_vision.py) |
| 134 | [Multimodal RAG](#урок-134-multimodal-rag) | [Код](134-multimodal_rag.py) |
| 135 | [Multimodal Agents](#урок-135-multimodal-agents) | [Код](135-multimodal_agents.py) |
| 136 | [Cross-Modal Transfer](#урок-136-cross-modal-transfer) | [Код](136-cross_modal_transfer.py) |
| 137 | [Multimodal Benchmarks](#урок-137-multimodal-benchmarks) | [Код](137-multimodal_benchmarks.py) |
| 138 | [Efficient Multimodal](#урок-138-efficient-multimodal) | [Код](138-efficient_multimodal.py) |
| 139 | [Multimodal Safety](#урок-139-multimodal-safety) | [Код](139-multimodal_safety.py) |
| 140 | [Building Multimodal Apps](#урок-140-building-multimodal-apps) | [Код](140-building_multimodal_apps.py) |

---

## Урок 126: Vision Transformers

### Image Patching

```
Image (H × W × C) → N patches (P × P × C)
N = (H/P) × (W/P)
Патч → Flatten → Linear Projection → d_model-dimensional vector
```

### Patch Embedding

```
z_0 = [CLS] + patch_embeddings + positional_embeddings
z_0: (N+1) × d_model
```

### ViT Forward Pass

```
z_0 → [Transformer Encoder × L] → CLS token → MLP Head → Class logits
```

---

## Урок 127: CLIP & Vision-Language Alignment

### Contrastive Learning

```
L = -1/N × Σ log( exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ) )

sim(I, T) = I · T / (|I| × |T|)   (косинусное сходство)
```

### Zero-Shot Classification

```
prompts = ["a photo of a {class}" for class in classes]
text_embeddings = encode(prompts)
image_embedding = encode(image)
probs = softmax(image_embedding · text_embeddings / τ)
```

---

## Урок 128: Image Generation

### Autoregressive Image Generation

```
Raster scan: left→right, top→bottom
P(x_t | x_{<t}) — предсказание следующего патча
```

### VQ-VAE

```
Image → Encoder → z_e → Quantize (codebook lookup) → z_q → Decoder → Image重建
Loss = Reconstruction + β × ||z_e - sg(z_q)||²
```

---

## Урок 129: Diffusion Models Advanced

### DDIM Sampling

```
x_{t-1} = √(ᾱ_{t-1}) × predicted_x0 + √(1-ᾱ_{t-1}) × ε_θ
Deterministic: одинаковый noise → одинаковый результат
```

### Classifier-Free Guidance

```
ε_guided = ε_uncond + w × (ε_cond - ε_uncond)
w = 1.0  — без guidance
w = 7.5  — стандартное значение
```

### Noise Schedules

```
Linear:    β_t = β_start + t × (β_end - β_start) / T
Cosine:    α̅_t = cos²(π/2 × (t + s) / (T + s))
```

---

## Урок 130: Video Understanding

### Frame Sampling

```
Uniform:   каждый N-й кадр
Keyframe:  по изменению сцены
Strided:   с фиксированным stride
```

### Temporal Modeling

```
3D Conv:     时空 свёртки (T × H × W)
Attention:    attention по временной оси
Two-Stream:   spatial (RGB) + optical flow
```

---

## Урок 131: Audio & Speech Models

### Audio Tokenization

```
Audio Waveform → Mel Spectrogram → Audio Tokens (codec)
Sample Rate: 16kHz, 80 мел-фильтров
```

### Whisper Architecture

```
Audio → Encoder (Conv1D + Transformer) → Decoder (Transformer + Cross-Attention)
Multitask: language detection, transcription, translation
```

### Decoding

```
CTC:        P(y|x) = Π P(y_t|x)  (независимые предсказания)
Attention:  P(y_t|x, y_{<t})  (авторегрессия)
```

---

## Урок 132: Document Understanding

### Layout Analysis

```
Page → Region Detection → Reading Order → Structured Document
Regions: text, image, table, figure, header, footer
```

### OCR Pipeline

```
Image → Text Detection (bounding boxes) → Text Recognition → Post-processing
```

### Table Extraction

```
Table Image → Grid Detection → Cell Identification → Structured Output (JSON/CSV)
```

---

## Урок 133: 3D Vision

### Point Cloud

```
N points × (x, y, z, [features])
Voxelization: 3D grid → occupancy
```

### Depth Estimation

```
Monocular:  Image → Depth Map (relative depth)
Stereo:     Disparity = f × B / Z  (f=focal, B=baseline, Z=depth)
```

### Point Cloud Processing

```
Sampling:    Farthest Point Sampling (FPS)
Grouping:    KNN → local neighborhoods
Features:    PointNet: MLP + max pooling
```

---

## Урок 134: Multimodal RAG

### Cross-Modal Retrieval

```
Text query → text_embedding → search image_embeddings → Top-K images
Image query → image_embedding → search text_embeddings → Top-K texts
```

### Multimodal Indexing

```
Hybrid Index: text_index + image_index
Fusion: α × text_score + (1-α) × image_score
```

---

## Урок 135: Multimodal Agents

### Vision-Based Tool Use

```
Screenshot → Element Detection → Grounding (x, y coordinates) → Action
```

### GUI Automation

```
Observe (screenshot) → Think (plan action) → Act (click/type/scroll) → Observe
```

### Embodied AI

```
Perception (camera) → Planning (goal) → Action (motor commands) → Environment
```

---

## Урок 136: Cross-Modal Transfer

### Multimodal Adapters

```
Frozen backbone + small adapter modules
Visual Adapter:  image_features → adapter → adapted_features
Cross-Attention: text queries attend to image features
```

### Few-Shot Multimodal

```
Prompt: "This is a photo of [class]."
In-context: include K image-label examples
```

---

## Урок 137: Multimodal Benchmarks

### VQA Metrics

```
Accuracy = min(#humans who said answer / 3, 1)
Soft Accuracy:允许部分 совпадение
```

### Captioning Metrics

```
BLEU:   precision of n-grams
CIDEr:  TF-IDF weighted n-gram similarity
METEOR: unigram matching with stemming and synonyms
```

### Retrieval Metrics

```
Recall@K:  доля queries, где верхний K содержит правильный ответ
MAP:       Mean Average Precision по всем запросам
```

---

## Урок 138: Efficient Multimodal

### Vision Encoder Efficiency

```
Token Pruning:    удаление неважных visual tokens
Early Exit:       выход на промежуточных слоях
Resolution Scaling: динамическое разрешение
```

### Cross-Modal Efficiency

```
Attention Bottleneck:  сжатие visual tokens перед cross-attention
Feature Compression:   PCA, quantization
```

---

## Урок 139: Multimodal Safety

### Visual Hallucination

```
Object Hallucination:  модель "видит" несуществующие объекты
Attribute Binding:     неправильное связывание атрибутов
Counting Errors:       неправильный подсчёт объектов
```

### Bias Detection

```
Gender Bias:   ассоциации профессий с полом
Racial Bias:   распределение по расам в генерации
Cultural Bias: западноцентричные представления
```

---

## Урок 140: Building Multimodal Apps

### Pipeline Design

```
Input → Preprocess → Model → Postprocess → Output
         ↓              ↓            ↓
     resize/crop    inference    filter/parse
     normalize      batch        format
```

### Production Considerations

```
Caching:      результаты моделей (invalidate by input hash)
Batching:     группировка запросов по типу модальности
Cost:         routing по стоимости (text vs image vs video)
```
