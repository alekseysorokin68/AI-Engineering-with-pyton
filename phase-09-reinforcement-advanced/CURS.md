# Phase 9: Reinforcement Learning (Advanced)

> Продвинутый RL — от Model-Based до Multi-Agent.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 91 | [Model-Based RL](#урок-91-model-based-rl) | [Код](91-model_based_rl.py) |
| 92 | [Monte Carlo Methods](#урок-92-monte-carlo-methods) | [Код](92-monte_carlo.py) |
| 93 | [Temporal Difference Learning](#урок-93-temporal-difference-learning) | [Код](93-temporal_difference.py) |
| 94 | [SARSA vs Q-Learning](#урок-94-sarsa-vs-q-learning) | [Код](94-sarsa_vs_qlearning.py) |
| 95 | [Multi-Agent RL](#урок-95-multi-agent-rl) | [Код](95_multi_agent_rl.py) |

---

## Урок 91: Model-Based RL

### Подход

```
1. Учись модели среды: T(s'|s,a)
2. Планируй через модель
3. Обновляй Q-значения реальными наблюдениями

Dyna-Q: реальные + симулированные переходы
```

---

## Урок 92: Monte Carlo Methods

### MC Policy Evaluation

```
1. Собери полный эпизод
2. Для каждого (s, t): G_t = Σ r_{t+1:T}
3. V(s) = среднее G_t для всех посещений s
```

### First-visit vs Every-visit

| Метод | Когда считать |
|---|---|
| First-visit | Только первое посещение s в эпизоде |
| Every-visit | Каждое посещение s в эпизоде |

---

## Урок 93: Temporal Difference Learning

### TD(0)

```
V(s) = V(s) + α × [r + γ × V(s') - V(s)]

Бутстраппинг: используем оценку V(s') вместо полного возврата
```

### TD(λ)

```
e(s) = γ × λ × e(s) + 1    (eligibility trace)
V(s) = V(s) + α × δ × e(s)  (обновление через trace)
```

---

## Урок 94: SARSA vs Q-Learning

### SARSA (on-policy)

```
Q(s,a) = Q(s,a) + α × [r + γ × Q(s',a') - Q(s,a)]

a' — действие, которое МЫ ВЫБРАЛИ
```

### Q-Learning (off-policy)

```
Q(s,a) = Q(s,a) + α × [r + γ × max_a'(Q(s',a')) - Q(s,a)]

max — лучшее действие, даже если мы его не выбирали
```

---

## Урок 95: Multi-Agent RL

### Типы

| Тип | Описание |
|---|---|
| Cooperative | Агенты работают вместе |
| Competitive | Агенты соревнуются |
| Communication | Агенты обмениваются информацией |
