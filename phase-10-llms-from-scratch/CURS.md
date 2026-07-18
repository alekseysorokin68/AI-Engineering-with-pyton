# Phase 10: LLMs from Scratch

> Языковые модели с нуля — от токенизации до GPT.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 96 | [Tokenization](#урок-96-tokenization) | [Код](96-tokenization.py) |
| 97 | [Embeddings](#урок-97-embeddings--positional-encoding) | [Код](97-embeddings.py) |
| 98 | [Self-Attention](#урок-98-self-attention) | [Код](98-self_attention.py) |
| 99 | [Multi-Head Attention](#урок-99-multi-head-attention) | [Код](99_multi_head_attention.py) |
| 100 | [Transformer Block](#урок-100-transformer-block) | [Код](100-transformer_block.py) |
| 101 | [GPT Architecture](#урок-101-gpt-architecture) | [Код](101-gpt_architecture.py) |
| 102 | [Pre-training](#урок-102-pre-training) | [Код](102-pretraining.py) |
| 103 | [Fine-tuning LLMs](#урок-103-fine-tuning-llms) | [Код](103-fine_tuning.py) |
| 104 | [Prompting](#урок-104-prompting--in-context-learning) | [Код](104-prompting.py) |
| 105 | [Chain-of-Thought](#урок-105-chain-of-thought) | [Код](105-chain_of_thought.py) |
| 106 | [RAG](#урок-106-rag) | [Код](106-rag.py) |
| 107 | [LoRA](#урок-107-lora) | [Код](107-lora.py) |
| 108 | [Quantization](#урок-108-quantization) | [Код](108-quantization.py) |
| 109 | [KV Cache](#урок-109-kv-cache) | [Код](109-kv_cache.py) |
| 110 | [LLM Evaluation](#урок-110-llm-evaluation) | [Код](110-llm_evaluation.py) |

---

## Урок 96: Tokenization

### BPE (Byte Pair Encoding)

```
1. Начинаем с символов
2. Находим самую частую пару
3. Объединяем в новый токен
4. Повторяем до нужного размера словаря
```

### WordPiece

```
Как BPE, но использует score = frequency / (freq_left × freq_right)
Разбивает на ##subwords для OOV слов
```

---

## Урок 97: Embeddings & Positional Encoding

### Token Embedding

```
token_id → lookup table → d_model-dimensional vector
```

### Sinusoidal Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

---

## Урок 98: Self-Attention

### Scaled Dot-Product

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

---

## Урок 99: Multi-Head Attention

```
head_i = Attention(X × W_i^Q, X × W_i^K, X × W_i^V)
MultiHead = Concat(head_1, ..., head_h) × W^O
```

---

## Урок 100: Transformer Block

```
Input → LayerNorm → Self-Attention → Residual → LayerNorm → FFN → Residual
```

---

## Урок 101: GPT Architecture

### Decoder-only Transformer

```
Input → [Causal Attention → FFN → LayerNorm] × N → LM Head → Logits
```

### Causal Mask

```
mask[i][j] = -∞ если j > i (future tokens)
```

---

## Урок 102: Pre-training

| Подход | Задача | Пример |
|---|---|---|
| Next Token Prediction | Предсказать следующий токен | GPT |
| Masked LM | Предсказать замаскированные токены | BERT |

---

## Урок 103: Fine-tuning LLMs

### SFT (Supervised Fine-Tuning)

```
Формат: System → User → Assistant
Loss: cross-entropy на ответах
```

---

## Урок 104: Prompting

| Техника | Описание |
|---|---|
| Zero-shot | Задача без примеров |
| Few-shot | С примерами |
| Chain-of-thought | С пошаговым рассуждением |

---

## Урок 105: Chain-of-Thought

### Методы

| Метод | Описание |
|---|---|
| CoT | Пошаговое решение |
| Self-consistency | Голосование по нескольким путям |
| Tree-of-Thought | Дерево поиска решений |

---

## Урок 106: RAG

### Pipeline

```
Query → Retriever (TF-IDF/Embeddings) → Top-K документов → LLM + Context → Answer
```

---

## Урок 107: LoRA

### Low-Rank Adaptation

```
ΔW = B × A, где B: (d × r), A: (r × d), r << d

W_new = W_frozen + α × B × A

Экономия: 10-100x меньше параметров
```

---

## Урок 108: Quantization

| Формат | Размер | Точность |
|---|---|---|
| FP32 | 4 bytes | 100% |
| FP16 | 2 bytes | ~99.9% |
| INT8 | 1 byte | ~99.5% |
| INT4 | 0.5 bytes | ~98% |

---

## Урок 109: KV Cache

```
Без кэша: пересчитываем K,V для всех токенов → O(n²)
С кэшем: добавляем только новый K,V → O(n)
```

### Размер KV Cache

```
2 × n_layers × seq_len × d_model × bytes_per_param
```

---

## Урок 110: LLM Evaluation

| Метрика | Что измеряет |
|---|---|
| Perplexity | Неуверенность модели (чем меньше, тем лучше) |
| BLEU | Качество перевода |
| Faithfulness | Соответствие контексту |
| MMLU | Знания по всем предметам |
