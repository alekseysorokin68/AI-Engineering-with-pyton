# Phase 5: NLP

> Обработка естественного языка — от токенизации до трансформеров.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 61 | [Text Representations](#урок-61-text-representations) | [Код](61-text_representations.py) |
| 62 | [Word Embeddings](#урок-62-word-embeddings) | [Код](62-word_embeddings.py) |
| 63 | [RNN](#урок-63-recurrent-neural-networks) | [Код](63-rnn.py) |
| 64 | [LSTM & GRU](#урок-64-lstm--gru) | [Код](64-lstm_gru.py) |
| 65 | [Sequence-to-Sequence](#урок-65-sequence-to-sequence) | [Код](65-seq2seq.py) |
| 66 | [Attention Mechanism](#урок-66-attention-mechanism) | [Код](66-attention.py) |
| 67 | [Transformer Architecture](#урок-67-transformer-architecture) | [Код](67-transformer.py) |
| 68 | [Pre-trained Language Models](#урок-68-pre-trained-language-models) | [Код](68_pretrained_models.py) |

---

## Урок 61: Text Representations

### Методы

| Метод | Описание |
|---|---|
| Bag of Words | Подсчёт частот слов |
| TF-IDF | Важность слова = TF × IDF |
| N-grams | Подсчёт последовательностей слов |

---

## Урок 62: Word Embeddings

### Word2Vec (Skip-gram)

```
Обучаем модель предсказывать контекст по слову
Результат: векторные представления слов

cosine(a, b) = a·b / (||a|| × ||b||)
```

---

## Урок 63: RNN

### SimpleRNN

```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
```

### Проблема

Затухание градиентов через длинные последовательности.

---

## Урок 64: LSTM & GRU

### LSTM

```
f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)    (forget gate)
i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)    (input gate)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(...)       (cell state)
o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)    (output gate)
h_t = o_t ⊙ tanh(c_t)
```

### GRU (упрощённый LSTM)

```
z_t = sigmoid(W_z × [h_{t-1}, x_t] + b_z)    (update gate)
r_t = sigmoid(W_r × [h_{t-1}, x_t] + b_r)    (reset gate)
h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ tanh(...)
```

---

## Урок 65: Sequence-to-Sequence

### Encoder-Decoder

```
Encoder: x_1, x_2, ..., x_n → context vector
Decoder: context → y_1, y_2, ..., y_m
```

### Teacher Forcing

Подавать правильный токен на каждом шаге декодера при обучении.

---

## Урок 66: Attention Mechanism

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

### Multi-Head Attention

```
head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
MultiHead = Concat(head_1, ..., head_h) × W^O
```

---

## Урок 67: Transformer Architecture

### Encoder

```
Input → Positional Encoding → [Self-Attention → Add & Norm → FFN → Add & Norm] × N
```

### Decoder

```
Input → Positional Encoding → [Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm] × N
```

### Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

---

## Урок 68: Pre-trained Language Models

### BERT vs GPT

| | BERT | GPT |
|---|---|---|
| Архитектура | Encoder-only | Decoder-only |
| Внимание | Bidirectional | Causal (left-to-right) |
| Задача | Понимание текста | Генерация текста |
| Предобучение | Masked LM | Next token prediction |
