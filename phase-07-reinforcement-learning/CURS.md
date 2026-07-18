# Phase 7: Reinforcement Learning

> Обучение с подкреплением — от Q-learning до RLHF.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 76 | [Introduction to RL](#урок-76-introduction-to-rl) | [Код](76-introduction_rl.py) |
| 77 | [Multi-Armed Bandits](#урок-77-multi-armed-bandits) | [Код](77-multi_armed_bandits.py) |
| 78 | [Tabular Q-Learning](#урок-78-tabular-q-learning) | [Код](78-q_learning.py) |
| 79 | [Deep Q-Networks](#урок-79-deep-q-networks) | [Код](79-dqn.py) |
| 80 | [Policy Gradient](#урок-80-policy-gradient) | [Код](80-policy_gradient.py) |
| 81 | [Actor-Critic](#урок-81-actor-critic) | [Код](81-actor_critic.py) |
| 82 | [RLHF](#урок-82-rlhf) | [Код](82-rlhf.py) |

---

## Урок 76: Introduction to RL

### Основы

| Концепция | Описание |
|---|---|
| Среда (Environment) | Мир, в котором действует агент |
| Состояние (State) | Текущая ситуация |
| Действие (Action) | Выбор агента |
| Вознаграждение (Reward) | Обратная связь от среды |
| Политика (Policy) | Стратегия выбора действий |

---

## Урок 77: Multi-Armed Bandits

### Стратегии

| Стратегия | Как работает |
|---|---|
| Epsilon-greedy | С вероятностью ε — случайно, иначе — лучшее |
| UCB | Балансирует exploration/exploitation через уверенность |
| Thompson Sampling | Байесовский подход, автоматический баланс |

---

## Урок 78: Tabular Q-Learning

### Q-обновление

```
Q(s,a) = Q(s,a) + α × [r + γ × max_a'(Q(s',a')) - Q(s,a)]
```

α — learning rate, γ — discount factor

---

## Урок 79: Deep Q-Networks

### DQN

```
Нейросеть аппроксирует Q-функцию: Q(s) → [Q(s,a1), Q(s,a2), ...]

Приёмы:
- Experience Replay: буфер для повторного использования
- Target Network: стабилизация обучения
- Epsilon Decay: уменьшение exploration со временем
```

---

## Урок 80: Policy Gradient

### REINFORCE

```
∇J(θ) = Σ ∇log π(a|s; θ) × G_t

G_t = сумма вознаграждений от шага t до конца эпизода
```

### Baseline

```
Advantage = G_t - baseline

Baseline = среднее вознаграждение → снижает дисперсию
```

---

## Урок 81: Actor-Critic

### A2C (Advantage Actor-Critic)

```
Actor: политика π(a|s; θ) → вероятности действий
Critic: ценность V(s; φ) → оценка состояния

Advantage = r + γ × V(s') - V(s)
Actor обновляется через advantage
Critic обновляется через TD error
```

---

## Урок 82: RLHF

### PPO (Proximal Policy Optimization)

```
ratio = π_new(a|s) / π_old(a|s)
clipped surrogate = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

### RLHF пайплайн

```
1. Предобучение (SFT) → базовая модель
2. Reward model → обучение на human preferences
3. PPO + KL penalty → оптимизация политики
```

### KL penalty

```
loss = -reward + β × KL(π_new || π_ref)

β контролирует отклонение от базовой модели
```
