# Phase 14: Agent Engineering

> Инженерия AI-агентов — от архитектуры до продакшена.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 156 | [Agent Architecture](#урок-156-agent-architecture) | [Код](156-agent_architecture.py) |
| 157 | [Tool Integration](#урок-157-tool-integration) | [Код](157-tool_integration.py) |
| 158 | [Planning & Reasoning](#урок-158-planning--reasoning) | [Код](158-planning_reasoning.py) |
| 159 | [Memory Systems](#урок-159-memory-systems) | [Код](159-memory_systems.py) |
| 160 | [Multi-Agent Systems](#урок-160-multi-agent-systems) | [Код](160-multi_agent_systems.py) |
| 161 | [Agent Evaluation](#урок-161-agent-evaluation) | [Код](161-agent_evaluation.py) |
| 162 | [Prompt Engineering for Agents](#урок-162-prompt-engineering-for-agents) | [Код](162-prompt_engineering_agents.py) |
| 163 | [Agent Frameworks](#урок-163-agent-frameworks) | [Код](163-agent_frameworks.py) |
| 164 | [Autonomous Agents](#урок-164-autonomous-agents) | [Код](164-autonomous_agents.py) |
| 165 | [Agent Security](#урок-165-agent-security) | [Код](165-agent_security.py) |
| 166 | [Agent Deployment](#урок-166-agent-deployment) | [Код](166-agent_deployment.py) |
| 167 | [RAG for Agents](#урок-167-rag-for-agents) | [Код](167-rag_for_agents.py) |
| 168 | [Code Agents](#урок-168-code-agents) | [Код](168-code_agents.py) |
| 169 | [Browser Agents](#урок-169-browser-agents) | [Код](169-browser_agents.py) |
| 170 | [Agent Use Cases](#урок-170-agent-use-cases) | [Код](170-agent_use_cases.py) |

---

## Урок 156: Agent Architecture

### Agent Loop

```
while not done:
    observation = env.observe()
    thought = llm.think(observation, memory)
    action = agent.decide(thought)
    result = env.execute(action)
    memory.update(observation, thought, action, result)
```

### State Machine

```
IDLE → THINKING → ACTING → OBSERVING → THINKING (цикл)
  ↓                    ↓
ERROR ←──────────── FAILURE
  ↓
TERMINATED
```

---

## Урок 157: Tool Integration

### Tool Schema

```json
{
  "name": "search_web",
  "description": "Поиск в интернете",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Поисковый запрос"}
    },
    "required": ["query"]
  }
}
```

### Function Calling Pipeline

```
LLM Output → Parse JSON → Validate Schema → Execute Tool → Format Result → LLM
```

---

## Урок 158: Planning & Reasoning

### ReAct

```
Thought: Нужно найти population of France
Action:  search("population of France 2024")
Observation: 67.8 million
Thought: Могу ответить
Action:  finish("Population of France: 67.8 million")
```

### Plan-and-Execute

```
Plan:  [step1, step2, step3]
Execute: step1 → result1 → step2(result1) → result2 → ...
Replan: если step_i провалился → новый план
```

---

## Урок 159: Memory Systems

### Типы памяти

```
Short-Term:   последние N сообщений (buffer)
Long-Term:    извлечённые факты (检索)
Episodic:     траектории действий (опыт)
Semantic:     граф знаний (факты и связи)
```

### Retrieval Score

```
relevance(query, memory) = cosine_sim(emb_q, emb_m) × recency × importance
```

---

## Урок 160: Multi-Agent Systems

### Паттерны кооперации

```
Sequential:   A → B → C (конвейер)
Parallel:     A, B, C одновременно → aggregator
Hierarchical: Manager → Worker1, Worker2, Worker3
Peer-to-Peer: A ↔ B ↔ C (равные)
```

### Communication

```
Message Passing:  агент отправляет сообщение другому
Blackboard:       общая область памяти
Pub/Sub:          подписка на события
```

---

## Урок 161: Agent Evaluation

### Метрики

```
Success Rate:     доля решённых задач
Step Efficiency:  среднее число шагов на задачу
Cost:             стоимость (токены × цена)
Latency:          время решения
```

### Failure Modes

```
Hallucination:    выдуманные инструменты/результаты
Tool Misuse:      неправильный выбор инструмента
Infinite Loop:    зацикливание на одной задаче
Goal Deviation:   отклонение от первоначальной цели
```

---

## Урок 162: Prompt Engineering for Agents

### System Prompt Structure

```
1. Role:      "Ты — AI-ассистент для анализа данных"
2. Tools:     описание доступных инструментов
3. Constraints: ограничения (не используй X, всегда делай Y)
4. Format:    формат вывода (JSON, markdown, specific schema)
```

### Dynamic Prompting

```
Template: "Контекст: {context}\nИстория: {history}\nЗадача: {task}"
Variables заполняются на каждом шаге
```

---

## Урок 163: Agent Frameworks

### LangChain

```
Chain:    Prompt → LLM → OutputParser
Agent:    Prompt + Tools → LLM → Action/Observation loop
Memory:   ConversationBufferMemory, ConversationSummaryMemory
```

### AutoGPT

```
Goal → Think → Browse/Write/Execute → Reflect → Think → ...
```

### CrewAI

```
Agent:    role + goal + backstory + tools
Task:     description + agent + expected_output
Crew:     agents + tasks + process (sequential/hierarchical)
```

---

## Урок 164: Autonomous Agents

### Goal-Driven

```
Goal: "Создай отчёт о продажах за Q1"
Subgoals: [собрать данные, проанализировать, написать, оформить]
Progress: 0% → 25% → 50% → 75% → 100%
```

### Self-Reflection

```
Output → Critique → "Ответ неполный, добавь источники" → Regenerate
```

### Error Recovery

```
Error → Classify → Retry (другой инструмент) → Fallback (ручной ввод)
```

---

## Урок 165: Agent Security

### Sandboxing

```
Agent Code → Restricted Runtime
  - No network access (или whitelist)
  - No filesystem write (или temp dir)
  - CPU/memory limits
  - Timeout
```

### Permission Model

```
Tool-Level:  search (allow), exec (deny), read (allow)
Role-Based:  admin (all), user (read-only), guest (limited)
Escalation:  agent → human approval → execute
```

---

## Урок 166: Agent Deployment

### Cost Management

```
Budget: $10/day
Routing: simple → cheap model, complex → expensive model
Caching: exact match + semantic cache
```

### Monitoring

```
Metrics: latency_p95, success_rate, cost_per_task, tokens_used
Alerts:  cost > budget, success_rate < 80%, latency > 30s
```

---

## Урок 167: RAG for Agents

### RAG Pipeline

```
Query → Retriever (Top-K docs) → Context Injection → LLM → Answer
                                      ↓
                              Source Attribution
```

### Chunking Strategies

```
Fixed Size:  512 токенов с overlap
Semantic:    по абзацам/предложениям
Recursive:   по заголовкам → абзацам → предложениям
```

---

## Урок 168: Code Agents

### Code Generation Pipeline

```
Task → Generate Code → Execute in Sandbox → Check Output → Fix Errors → Iterate
```

### Debugging Loop

```
Error Message → Analyze Stack Trace → Locate Bug → Suggest Fix → Apply & Retest
```

---

## Урок 169: Browser Agents

### Web Interaction

```
GET URL → Parse HTML → Extract Links/Forms → Submit Form → Parse Response
```

### Session Management

```
Cookies:    сохраняются между запросами
Headers:    User-Agent, Authorization
Rate Limit: delay between requests
```

---

## Урок 170: Agent Use Cases

### Customer Support

```
Ticket → Classify (topic, urgency) → Route → Generate Response → Escalate if needed
```

### Research Agent

```
Question → Search → Read → Summarize → Cite Sources → Generate Report
```

### Data Analysis

```
Dataset → Profile → Clean → Analyze → Visualize → Report Insights
```
