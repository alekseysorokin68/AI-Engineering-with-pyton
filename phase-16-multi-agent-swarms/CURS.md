# Phase 16: Multi-Agent & Swarms

> Мультагентные системы и роевой интеллект.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 186 | [Multi-Agent Fundamentals](#урок-186-multi-agent-fundamentals) | [Код](186-multi_agent_fundamentals.py) |
| 187 | [Agent Communication](#урок-187-agent-communication) | [Код](187-agent_communication.py) |
| 188 | [Cooperative Strategies](#урок-188-cooperative-strategies) | [Код](188-cooperative_strategies.py) |
| 189 | [Competitive Agents](#урок-189-competitive-agents) | [Код](189-competitive_agents.py) |
| 190 | [Agent Organizations](#урок-190-agent-organizations) | [Код](190-agent_organizations.py) |
| 191 | [Swarm Algorithms](#урок-191-swarm-algorithms) | [Код](191-swarm_algorithms.py) |
| 192 | [Emergent Behavior](#урок-192-emergent-behavior) | [Код](192-emergent_behavior.py) |
| 193 | [Agent Negotiation](#урок-193-agent-negotiation) | [Код](193-agent_negotiation.py) |
| 194 | [MARL](#урок-194-multi-agent-reinforcement-learning) | [Код](194-marl.py) |
| 195 | [LLM Multi-Agent](#урок-195-llm-based-multi-agent) | [Код](195_llm_multi_agent.py) |
| 196 | [Simulation & Testing](#урок-196-simulation--testing) | [Код](196-simulation_testing.py) |
| 197 | [Communication Networks](#урок-197-communication-networks) | [Код](197-communication_networks.py) |
| 198 | [Task Distribution](#урок-198-task-distribution) | [Код](198-task_distribution.py) |
| 199 | [Scaling Agents](#урок-199-scaling-multi-agent-systems) | [Код](199-scaling_agents.py) |
| 200 | [Production Multi-Agent](#урок-200-production-multi-agent) | [Код](200-production_multi_agent.py) |

---

## Урок 186: Multi-Agent Fundamentals

### Таксономия агентов

```
Cooperative:      общая цель, координация
Competitive:      противоположные интересы
Mixed:            кооперативные подгруппы + конкуренция
Centralized:      один центральный координатор
Decentralized:    равные агенты, peer-to-peer
```

### FIPA ACL Message

```
(inform
  :sender agent1
  :receiver agent2
  :content (price widget 100)
  :language FIPA-SL
  :ontology trade)
```

---

## Урок 187: Agent Communication

### Речевые акты

```
inform:   передать информацию
request:  попросить действие
propose:  предложить сделку
agree:    согласиться на предложение
refuse:   отказаться
confirm:  подтвердить
```

### Протоколы

```
Request-Inform:  A просит → B выполняет → B сообщает результат
Contract-Net:    объявление → подача заявок → присуждение → выполнение
Auction:         аукцион → ставки → победитель
```

---

## Урок 188: Cooperative Strategies

### Shapley Value

```
φ_i = Σ_{S⊆N\{i}} |S|!(|N|-|S|-1)!/|N|! × (v(S∪{i}) - v(S))
Справедливое распределение выигрыша по вкладу
```

### Coalition Formation

```
v({A,B}) = 10, v({A}) = 4, v({B}) = 5
Синергия: 10 - 4 - 5 = 1 > 0 → выгодно объединиться
```

---

## Урок 189: Competitive Agents

### Nash Equilibrium

```
Все игроки выбирают лучшую стратегию, учитывая других
Ни один игрок не может улучшить свой результат, изменив стратегию
```

### Типы аукционов

```
English:      открытое повышение ставок (самая высокая побеждает)
Dutch:        открытое снижение цены (первый поднявший руку)
Sealed-Bid:   закрытые ставки (первый/второй ценой)
VCG:         each платит externalities другим
```

---

## Урок 190: Agent Organizations

### Типы структур

```
Hierarchy:    CEO → VP → Manager → Worker (цепочка команд)
Flat:         все агенты равны (peer-to-peer)
Matrix:       dual reporting (function + project)
Network:      гибкие связи, временные коалиции
```

### Role Assignment

```
Capability Matching: назначить агенту задачу по компетенции
Workload Balancing:  распределить нагрузку равномерно
Role Evolution:      роли меняются со временем
```

---

## Урок 191: Swarm Algorithms

### ACO (Ant Colony)

```
P(edge) = τ^α × η^β / Σ(τ^α × η^β)
τ: феромон (опыт)
η: эвристика (1/distance)
Update: τ = (1-ρ) × τ + Δτ
```

### PSO (Particle Swarm)

```
v = w × v + c₁ × r₁ × (pbest - x) + c₂ × r₂ × (gbest - x)
x = x + v
```

### ABC (Artificial Bee Colony)

```
Employed:  источник食物 → вспомогательный поиск рядом
Onlooker:  выбирает источник по вероятности → улучшает
Scout:     если source исчерпан → новый случайный
```

---

## Урок 192: Emergent Behavior

### Self-Organization

```
Order Parameter: φ = 1/N × Σ s_i (средняя ориентация)
Phase Transition: при критическом параметре → порядок
```

### Cellular Automata

```
Conway's Game of Life:
  Alive + 2-3 neighbors → survive
  Dead + 3 neighbors → born
  Otherwise → die
```

---

## Урок 193: Agent Negotiation

### Nash Bargaining

```
max (u₁ - d₁) × (u₂ - d₂)
d: disagreement point (лучшая альтернатива)
```

### Rubinstein Alternation

```
Предложение A → B: принять или отклонить → B предлагает → A: принять или отклонить
discount factor δ: чем больше, тем терпеливее
```

---

## Урок 194: MARL

### Cooperative MARL

```
QMIX: Q_tot = f(Q₁, Q₂, ..., Qₙ)
f: monotonic (не必须要全连接)
Training: centralized, execution: decentralized
```

### Self-Play

```
Agent vs Agent → оба улучшаются
AlphaGo: policy network + value network через self-play
```

---

## Урок 195: LLM Multi-Agent

### Multi-Agent Debate

```
Agent A: аргумент → Agent B: контраргумент → Agent C: оценка → Consensus
```

### Workflow Patterns

```
Sequential:  Agent1 → Agent2 → Agent3
Parallel:    Agent1, Agent2, Agent3 одновременно → aggregator
Conditional: if complex → specialist, else → generalist
```

---

## Урок 196: Simulation & Testing

### Agent-Based Model

```
Environment: сетка/граф
Agents: правила поведения
Rules: переходы состояний
Emergence: макро-паттерны из микро-правил
```

### Testing

```
Unit:     каждый агент изолированно
Integration: взаимодействие пар агентов
Scenario:  полные сценарии
Stress:    много агентов, экстремальные условия
```

---

## Урок 197: Communication Networks

### Топологии

```
Ring:     каждая нода связана с 2 соседями
Star:     центральная нода + листья
Mesh:     полная связность
Small-World: кластеры + случайные связи
Scale-Free:  power-law degree distribution
```

### Gossip Protocols

```
Push:      отправляю случайному соседу
Pull:      запрашиваю у случайного соседа
Push-Pull: обмен с邻居ом
```

---

## Урок 198: Task Distribution

### Task Allocation

```
Hungarian: оптимальное назначение за O(n³)
Auction:   агенты предлагают цену → лучший выигрывает
Greedy:    назначить наиболее доступному агенту
```

### Work Stealing

```
Idle Agent: проверяет очередь соседа → забирает задачу
Work Sharing: busy agent отправляет задачу idle agent
```

---

## Урок 199: Scaling Multi-Agent

### Hierarchical Control

```
Strategic Layer:  глобальные цели
Tactical Layer:  планирование подцелей
Operational Layer: выполнение действий
```

### Scalability

```
Partitioning: разбить агентов по подсетям
Caching:      кэшировать результаты вычислений
Async:        асинхронные сообщения (не блокирующие)
```

---

## Урок 200: Production Multi-Agent

### Deployment

```
Service Mesh:  управление сервисами (Istio, Linkerd)
Message Broker: RabbitMQ, Kafka для асинхронных сообщений
Event Bus:     pub/sub для событийной архитектуры
```

### Monitoring

```
Agent Metrics:   latency, throughput, error rate
Communication:   message volume, latency, failures
System:          CPU, memory, network
```

### Debugging

```
Message Tracing: логировать все сообщения
State Inspection: inspect agent state at any point
Replay:          воспроизвести историю для отладки
```
