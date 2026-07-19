# Phase 11: LLM Engineering

> Практическая инженерия LLM — от API до продакшена.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 111 | [LLM API Design](#урок-111-llm-api-design) | [Код](111-llm_api_design.py) |
| 112 | [Token Management](#урок-112-token-management) | [Код](112-token_management.py) |
| 113 | [Prompt Engineering Advanced](#урок-113-prompt-engineering-advanced) | [Код](113-prompt_engineering_advanced.py) |
| 114 | [LLM Serving](#урок-114-llm-serving) | [Код](114-llm_serving.py) |
| 115 | [Inference Optimization](#урок-115-inference-optimization) | [Код](115-inference_optimization.py) |
| 116 | [LLM Security](#урок-116-llm-security) | [Код](116-llm_security.py) |
| 117 | [Agents & Tool Use](#урок-117-agents--tool-use) | [Код](117-agents_tool_use.py) |
| 118 | [Multi-Modal LLMs](#урок-118-multimodal-llms) | [Код](118-multimodal_llms.py) |
| 119 | [LLM Memory](#урок-119-llm-memory) | [Код](119-llm_memory.py) |
| 120 | [Fine-tuning at Scale](#урок-120-fine-tuning-at-scale) | [Код](120-finetuning_at_scale.py) |
| 121 | [LLM Testing](#урок-121-llm-testing) | [Код](121-llm_testing.py) |
| 122 | [LLM Deployment](#урок-122-llm-deployment) | [Код](122-llm_deployment.py) |
| 123 | [LLM Monitoring](#урок-123-llm-monitoring) | [Код](123-llm_monitoring.py) |
| 124 | [Cost Optimization](#урок-124-cost-optimization) | [Код](124-cost_optimization.py) |
| 125 | [LLM Production](#урок-125-llm-production) | [Код](125-llm_production.py) |

---

## Урок 111: LLM API Design

### Message Formats

```
System  → задаёт роль и контекст
User    → запрос пользователя
Assistant → ответ модели
```

### Temperature & Sampling

```
P(token) = exp(logit / T) / Σ exp(logit_i / T)

T = 1.0  — стандартное распределение
T → 0   — жадный выбор (максимум)
T → ∞   — равномерное распределение
```

### Top-p (Nucleus Sampling)

```
Отсортировать по вероятности → взять кумулятивную сумму ≤ p
```

---

## Урок 112: Token Management

### Контекстное окно

```
context_window = prompt_tokens + max_tokens
Остаток = context_window - prompt_tokens
```

### Sliding Window

```
[ msg1, msg2, msg3, msg4, msg5 ]  →  [ msg3, msg4, msg5 ]
  старые сообщения удаляются, системный промпт сохраняется
```

---

## Урок 113: Prompt Engineering Advanced

### Структурированный вывод

```
JSON Mode:  { "key": "value" }
XML Mode:   <result><key>value</key></result>
Markdown:   | Key | Value |
```

### Prompt Chaining

```
Step 1 → Step 2 → Step 3 → Final Answer
         (каждый шаг проверяется перед следующим)
```

---

## Урок 114: LLM Serving

### Dynamic Batching

```
batch_size = min(pending_requests, max_batch)
wait_time < timeout → накапливаем запросы
```

### Streaming

```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"token": "Hello"}
data: {"token": " world"}
data: [DONE]
```

### Rate Limiting (Token Bucket)

```
tokens += elapsed_time × refill_rate
if tokens >= cost: tokens -= cost, allow
else: reject
```

---

## Урок 115: Inference Optimization

### Speculative Decoding

```
Draft model: генерирует K токенов быстро
Target model: верифицирует все K за один проход
Accept rate = min(1, p_target / p_draft)
```

### Paged Attention

```
KV cache разбит на блоки (pages)
Блоки могут быть не смежными в памяти
Реф-счётчик: блоки разделяются между запросами
```

---

## Урок 116: LLM Security

### Prompt Injection

```
Direct:  "Ignore previous instructions and..."
Indirect: вставка вредоносных данных в контекст
```

### Guardrails

```
Input  → [Allowlist Check] → [Content Filter] → LLM
Output → [Output Filter] → [Policy Check] → Response
```

---

## Урок 117: Agents & Tool Use

### ReAct Pattern

```
Thought: нужна информация о погоде
Action:  get_weather("Moscow")
Observation: пасмурно, 15°C
Thought: могу ответить
Action:  finish("В Москве пасмурно, 15°C")
```

### Function Calling

```json
{
  "name": "get_weather",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"}
    }
  }
}
```

---

## Урок 118: Multi-Modal LLMs

### Vision-Language

```
Image → CNN/ViT → image_tokens
Text  → Tokenizer → text_tokens
                ↓
         Cross-Attention Fusion
                ↓
            Response
```

### Multi-Modal Fusion

```
Early Fusion:  конкатенация на уровне токенов
Late Fusion:   отдельные энкодеры + совместное внимание
```

---

## Урок 119: LLM Memory

### Типы памяти

```
Buffer Memory:  последние N сообщений
Summary Memory: краткое содержание всей истории
Long-term:      извлечённые факты и знания
```

### Memory Retrieval

```
query_embedding · memory_embedding / (|q| · |m|)
→ Top-K наиболее релевантных воспоминаний
```

---

## Урок 120: Fine-tuning at Scale

### Стратегии обучения

```
Full Fine-tuning:  все параметры обновляются
Adapter:           замороженные веса + маленькие модули
Progressive:       поэтапное размораживание слоёв
```

### Distributed Training

```
Data Parallelism:    каждый GPU — полная модель, разные батчи
Model Parallelism:   модель разбита между GPU
Pipeline Parallelism: слои распределены по GPU
```

---

## Урок 121: LLM Testing

### Red Teaming

```
Prompt Injection → проверка устойчивости
Jailbreaking    → попытки обхода ограничений
Data Extraction → попытки утечки данных
```

### A/B Testing

```
p_value < 0.05 → статистически значимая разница
Sample size = (Z_α/2 + Z_β)² × 2σ² / δ²
```

---

## Урок 122: LLM Deployment

### Стратегии развёртывания

```
Blue-Green:  два окружения, мгновенный переключ
Canary:      постепенный rollout на долю трафика
Rollback:    возврат к предыдущей версии
```

### Health Checks

```
Readiness:  модель загружена и готова принимать запросы
Liveness:   процесс жив и отвечает
Model:      качество ответов в пределах нормы
```

---

## Урок 123: LLM Monitoring

### Метрики

```
Latency P50/P95/P99 — время ответа
Token/s             — скорость генерации
Error Rate          — доля ошибок
Cost per Request    — стоимость запроса
```

### Drift Detection

```
PSI = Σ (A_i - B_i) × ln(A_i / B_i)
PSI < 0.1  — нет дрейфа
PSI > 0.2  — значительный дрейф
```

---

## Урок 124: Cost Optimization

### Кэширование

```
Exact Match:  полное совпадение запроса
Prefix Cache: совпадение начала (system prompt)
Semantic:     семантически похожие запросы
```

### Model Routing

```
Простой запрос → дешёвая модель (GPT-4o-mini)
Сложный запрос → мощная модель (GPT-4o)
```

---

## Урок 125: LLM Production

### Circuit Breaker

```
CLOSED   → запросы проходят正常
OPEN     → все запросы отклоняются
HALF-OPEN → тестовые запросы для проверки восстановления
```

### Incident Response

```
1. Детект → алерт сработал
2. Триаж  → классификация серьёзности
3. Митигация → откат или hotfix
4. Постмортем → анализ причин и предотвращение
```
