# Phase 15: Autonomous Systems

> Автономные системы — от архитектуры до продакшена.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 171 | [Autonomous Architecture](#урок-171-autonomous-system-architecture) | [Код](171-autonomous_architecture.py) |
| 172 | [Self-Learning](#урок-172-self-learning-systems) | [Код](172-self_learning.py) |
| 173 | [Adaptive Behavior](#урок-173-adaptive-behavior) | [Код](173-adaptive_behavior.py) |
| 174 | [World Models](#урок-174-world-models) | [Код](174-world_models.py) |
| 175 | [Self-Improvement](#урок-175-self-improvement) | [Код](175-self_improvement.py) |
| 176 | [Multi-Objectives](#урок-176-multi-objectives) | [Код](176-multi_objectives.py) |
| 177 | [Resource-Aware](#урок-177-resource-aware-agents) | [Код](177-resource_aware.py) |
| 178 | [Long-Horizon](#урок-178-long-horizon-autonomy) | [Код](178-long_horizon.py) |
| 179 | [Human-AI Teaming](#урок-179-human-ai-teaming) | [Код](179-human_ai_teaming.py) |
| 180 | [Autonomous Decisions](#урок-180-autonomous-decision-making) | [Код](180-autonomous_decisions.py) |
| 181 | [Self-Monitoring](#урок-181-self-monitoring) | [Код](181-self_monitoring.py) |
| 182 | [Swarm Intelligence](#урок-182-swarm-intelligence) | [Код](182-swarm_intelligence.py) |
| 183 | [Continual Learning](#урок-183-continual-learning) | [Код](183-continual_learning.py) |
| 184 | [Autonomous Debugging](#урок-184-autonomous-debugging) | [Код](184-autonomous_debugging.py) |
| 185 | [Production Autonomy](#урок-185-production-autonomy) | [Код](185-production_autonomy.py) |

---

## Урок 171: Autonomous System Architecture

### Уровни автономии (SAE)

```
L0:  Ручное управление
L1:  Помощь водителю (ACC)
L2:  Частичная автоматизация (lane + cruise)
L3:  Условная автоматизация (переход в определённых условиях)
L4:  Высокая автоматизация (без водителя, ограниченная зона)
L5:  Полная автоматизация (везде, всегда)
```

### Архитектура принятия решений

```
Perception → World Model → Planner → Executor → Environment
     ↑                                              ↓
     └──────────── Observation ←──────────────────┘
```

---

## Урок 172: Self-Learning Systems

### Онлайн-обучение

```
New Data → Update Model → Evaluate → Keep/Revert
  w = w - lr × ∇L(x_new, y_new)
```

### Experience Replay

```
Buffer: [exp1, exp2, ..., expN]
Sample: batch_size случайных переходов
Priority: P(i) ∝ |δ_i|^α  (TD-error)
```

### Intrinsic Motivation

```
Reward = extrinsic + β × intrinsic
intrinsic = prediction_error + count_bonus + curiosity
```

---

## Урок 173: Adaptive Behavior

### Behavior Tree

```
Selector:    попробовать детей по очереди (первый успешный)
Sequence:    выполнить всех детей (первый провал → FAIL)
Decorator:   модифицировать результат ребёнка
```

### GOAP

```
Goal:  [has_gold = true]
State: [has_pickaxe = true, at_mine = true]
Plan:  [mine_ore → smelt_bar → buy_gold]
A*:    cost(plan) = sum(action_costs)
```

### Utility System

```
score(eat) = urgency(hunger) × feasibility(has_food)
score(sleep) = urgency(fatigue) × feasibility(has_bed)
Best action = argmax(scores)
```

---

## Урок 174: World Models

### Transition Model

```
P(s_{t+1} | s_t, a_t) — модель динамики среды
Predicted: s_{t+1} = f(s_t, a_t)
```

### MCTS (Monte Carlo Tree Search)

```
Select → Expand → Simulate → Backpropagate
UCB1 = Q(s,a) + c × sqrt(ln(N(s)) / N(s,a))
```

### Dreamer-Style Planning

```
Experience → Train World Model → Generate Imagined Trajectories → Optimize Policy
```

---

## Урок 175: Self-Improvement

### Prompt Evolution (GA)

```
Population: [prompt_1, prompt_2, ..., prompt_N]
Fitness: score(prompt_i) на задачах
Selection: турнирная селекция
Mutation: случайные изменения
Crossover: комбинация лучших
```

### Meta-Learning (MAML)

```
Fast Adaptation: θ' = θ - α × ∇L_task(θ)
Meta-Update:     θ = θ - β × ∇Σ L_task_i(θ')
```

---

## Урок 176: Multi-Objectives

### Pareto Optimality

```
Решение A доминирует B, если:
  f(A) ≥ f(B) по всем целям И строго > хотя бы по одной
Pareto Front: множество недоминирующих решений
```

### Weighted Sum

```
min F = w₁ × f₁(x) + w₂ × f₂(x) + ... + wₙ × fₙ(x)
Разные веса → разные точки на Pareto Front
```

---

## Урок 177: Resource-Aware Agents

### Token Budget

```
Budget: 10000 токенов
Used:   3500
Remaining: 6500
Strategy: если остаток < 30% → краткие ответы
```

### Lazy Evaluation

```
Если значение уже вычислено → вернуть кэш
Если依赖 не нужны → пропустить вычисление
Если результат не нужен → defer вычисление
```

---

## Урок 178: Long-Horizon Autonomy

### Task DAG

```
A → B → D → F
A → C → E → F
Параллельно: B и C могут выполняться одновременно
```

### Checkpointing

```
Step 1: ✓ → checkpoint
Step 2: ✓ → checkpoint
Step 3: FAIL → загрузить checkpoint → retry step 3
```

### Retry Strategies

```
Exponential Backoff: delay = min(base × 2^attempt, max_delay)
Jitter: delay += random(0, jitter)
Circuit Breaker: failures > threshold → open → после timeout → half-open
```

---

## Урок 179: Human-AI Teaming

### Handoff Protocol

```
Confidence > 0.9  → AI отвечает автономно
0.7 < Conf < 0.9  → AI отвечает с оговоркой
Conf < 0.7        → handoff человеку
```

### Trust Calibration

```
Calibration Error = |confidence - accuracy|
Хорошо: confidence ≈ accuracy для каждого бина
Плохо: confidence 0.9, accuracy 0.5 (overconfident)
```

---

## Урок 180: Autonomous Decision Making

### Expected Utility

```
EU(action) = Σ P(outcome_i) × U(outcome_i)
Best: argmax(EU)
```

### Minimax Regret

```
Regret(a, s) = U(best_for_s) - U(a, s)
Minimax Regret = min_a max_s Regret(a, s)
```

### Bayesian Decision

```
Posterior ∝ Prior × Likelihood
EVPI = E[max_a U(a,θ)] - max_a E[U(a,θ)]
```

---

## Урок 181: Self-Monitoring

### Anomaly Detection

```
Z-score: z = (x - μ) / σ
Anomaly: |z| > 3
CUSUM: кумулятивная сумма отклонений
```

### Confidence Calibration

```
Expected Calibration Error (ECE) = Σ |bin_acc - bin_conf| × bin_size
Temperature Scaling: logits = logits / T
```

---

## Урок 182: Swarm Intelligence

### Ant Colony Optimization

```
P(choose edge) = τ^α × η^β / Σ(τ^α × η^β)
τ: феромон (опыт)
η: эвристика (1/distance)
Evaporation: τ = (1-ρ) × τ + Δτ
```

### Particle Swarm

```
v = w × v + c₁ × r₁ × (pbest - x) + c₂ × r₂ × (gbest - x)
x = x + v
```

### Flocking Rules

```
Separation:  отдалиться от соседей (избежать столкновений)
Alignment:   ориентироваться по направлению соседей
Cohesion:    двигаться к центру группы
```

---

## Урок 183: Continual Learning

### Catastrophic Forgetting

```
Train on Task A → Train on Task B → Performance on A drops!
Решение: replay buffer, regularization, progressive networks
```

### EWC (Elastic Weight Consolidation)

```
L_total = L_task + λ/2 × Σ F_i × (θ_i - θ*_i)²
F: Fisher Information (важность параметра)
θ*: оптимальные параметры на предыдущей задаче
```

---

## Урок 184: Autonomous Debugging

### Root Cause Analysis

```
Error → Stack Trace → Fault Tree → Hypothesis → Test → Fix
```

### Delta Debugging

```
Input: [1, 2, 3, 4, 5, 6] → crash
Bisect: [1, 2, 3] → no crash
       [4, 5, 6] → crash
       [4, 5] → crash
       [4] → crash → minimal failing input
```

---

## Урок 185: Production Autonomy

### SLA

```
Availability: 99.9% = 8.76 hours downtime/year
Latency P99:  < 500ms
Error Rate:   < 0.1%
```

### Graceful Degradation

```
Full:       все фичи
Degraded:   основные фичи, упрощённый ответ
Fallback:   rule-based ответ
Down:       сообщение об ошибке
```

### Circuit Breaker States

```
CLOSED → нормальная работа
OPEN → все запросы отклоняются (timeout)
HALF-OPEN → тестовый запрос для проверки восстановления
```
