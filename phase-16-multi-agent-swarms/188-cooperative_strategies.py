"""
188 — Cooperative Strategies: совместное планирование, общие цели, коалиции

Темы:
  1. Joint Planning — построение общего плана, слияние планов, разрешение конфликтов
  2. Shared Goals — распределение целей, стратегии обязательств, согласование целей
  3. Coalition Formation — характеристическая функция, цена Шапли, ядро
  4. Cooperation Incentives — репутация, взаимность, наказание

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time
import itertools

random.seed(42)


# ─────────────────────────── Демо 1: Совместное планирование ───────────────────────────

def demo_joint_planning():
    """Демонстрация построения общего плана, слияния планов, разрешения конфликтов."""
    print("=" * 70)
    print("ДЕМО 1: Joint Planning — построение общего плана, слияние планов")
    print("=" * 70)

    # 1.1 Индивидуальные планы агентов
    print("Индивидуальные планы агентов:")
    print("-" * 50)

    plan_A = [
        {"time": 1, "action": "подготовить_материалы", "resource": "мастерская"},
        {"time": 2, "action": "собрать_каркас", "resource": "мастерская"},
        {"time": 3, "action": "установить_каркас", "resource": "площадка"},
        {"time": 4, "action": "отделка", "resource": "площадка"},
    ]

    plan_B = [
        {"time": 1, "action": "закупить_компоненты", "resource": "склад"},
        {"time": 2, "action": "смонтировать_проводку", "resource": "площадка"},
        {"time": 3, "action": "установить_оборудование", "resource": "площадка"},
        {"time": 4, "action": "протестировать", "resource": "лаборатория"},
    ]

    print("  План А (строитель):")
    for step in plan_A:
        print(f"    t={step['time']}: {step['action']} (ресурс: {step['resource']})")

    print("\n  План B (электрик):")
    for step in plan_B:
        print(f"    t={step['time']}: {step['action']} (ресурс: {step['resource']})")

    # 1.2 Обнаружение конфликтов
    print("\nОбнаружение конфликтов:")
    print("-" * 50)

    def detect_conflicts(plan_a, plan_b):
        """Обнаружение временных и ресурсных конфликтов."""
        conflicts = []
        for step_a in plan_a:
            for step_b in plan_b:
                # Временной конфликт — оба в одно время на одной площадке
                if step_a["time"] == step_b["time"] and step_a["resource"] == step_b["resource"]:
                    conflicts.append({
                        "type": "resource_conflict",
                        "time": step_a["time"],
                        "resource": step_a["resource"],
                        "actions": [step_a["action"], step_b["action"]],
                        "agents": ["А", "Б"],
                    })
        return conflicts

    conflicts = detect_conflicts(plan_A, plan_B)
    if conflicts:
        for i, c in enumerate(conflicts, 1):
            print(f"  Конфликт {i}: время={c['time']}, ресурс={c['resource']}")
            print(f"    А: {c['actions'][0]}, Б: {c['actions'][1]}")
    else:
        print("  Конфликты не обнаружены")

    # 1.3 Слияние планов (plan merging)
    print("\nСлияние планов (Plan Merging):")
    print("-" * 50)

    def merge_plans(plan_a, plan_b):
        """Простое слияние планов с переназначением ресурсов."""
        merged = []
        all_steps = []
        for step in plan_a:
            all_steps.append({**step, "agent": "А"})
        for step in plan_b:
            all_steps.append({**step, "agent": "Б"})

        # Сортировка по времени
        all_steps.sort(key=lambda s: s["time"])

        # Переназначение ресурсов при конфликтах
        used_resources = {}
        for step in all_steps:
            time_key = step["time"]
            resource = step["resource"]
            if time_key in used_resources and resource in used_resources[time_key]:
                # Конфликт — переназначаем ресурс
                alternative = f"{resource}_{step['agent']}"
                step["resource"] = alternative
                step["note"] = f"переназначен с {resource}"
            if time_key not in used_resources:
                used_resources[time_key] = set()
            used_resources[time_key].add(step["resource"])
            merged.append(step)
        return merged

    merged = merge_plans(plan_A, plan_B)
    print("  Объединённый план:")
    for step in merged:
        note = f" ({step.get('note', '')})" if step.get('note') else ""
        print(f"    t={step['time']} агент={step['agent']}: {step['action']} [{step['resource']}]{note}")

    # 1.4 Упорядоченное выполнение (sequential execution)
    print("\nУпорядоченное выполнение (sequential):")
    print("-" * 50)

    def sequential_execution(merged_plan):
        """Выполнение плана с проверкой предусловий."""
        completed = []  # список кортежей (time, resource)
        execution_log = []
        for step in merged_plan:
            time_slot = step["time"]
            action = step["action"]
            resource = step["resource"]

            # Проверка: все действия на этом ресурсе завершены
            prereqs_ok = True
            for prev_time, prev_resource in completed:
                if prev_resource == resource and prev_time >= time_slot:
                    prereqs_ok = False
                    break

            if prereqs_ok:
                completed.append((time_slot, resource))
                execution_log.append(f"  t={time_slot}: ✓ {step['agent']} выполняет '{action}' [{resource}]")
            else:
                execution_log.append(f"  t={time_slot}: ✗ {step['agent']} БЛОКИРОВАН '{action}' [{resource}]")

        return execution_log

    exec_log = sequential_execution(merged)
    for entry in exec_log:
        print(entry)

    print()


# ─────────────────────────── Демо 2: Общие цели ───────────────────────────

def demo_shared_goals():
    """Демонстрация распределения целей, стратегий обязательств, согласования целей."""
    print("=" * 70)
    print("ДЕМО 2: Shared Goals — распределение целей, стратегии обязательств")
    print("=" * 70)

    # 2.1 Распределение целей (goal allocation)
    print("Распределение целей (Goal Allocation):")
    print("-" * 50)

    class GoalAllocation:
        def __init__(self, agents, goals):
            self.agents = agents
            self.goals = goals
            self.allocation = {}

        def allocate_by_capability(self, capabilities):
            """Распределение по способностям агентов."""
            for agent in self.agents:
                best_goal = None
                best_score = -1
                for goal in self.goals:
                    if goal not in self.allocation.values():
                        score = capabilities.get(agent, {}).get(goal, 0)
                        if score > best_score:
                            best_score = score
                            best_goal = goal
                if best_goal:
                    self.allocation[agent] = best_goal
            return self.allocation

    goals = ["исследование", "сбор_данных", "анализ", "отчёт"]
    agents = ["Учёный_1", "Учёный_2", "Учёный_3", "Учёный_4"]

    # Способности агентов (оценка от 0 до 1)
    capabilities = {
        "Учёный_1": {"исследование": 0.9, "сбор_данных": 0.3, "анализ": 0.5, "отчёт": 0.4},
        "Учёный_2": {"исследование": 0.4, "сбор_данных": 0.8, "анализ": 0.6, "отчёт": 0.5},
        "Учёный_3": {"исследование": 0.3, "сбор_данных": 0.4, "анализ": 0.9, "отчёт": 0.7},
        "Учёный_4": {"исследование": 0.5, "сбор_данных": 0.6, "анализ": 0.4, "отчёт": 0.9},
    }

    allocator = GoalAllocation(agents, goals)
    allocation = allocator.allocate_by_capability(capabilities)
    print("  Распределение по способностям:")
    for agent, goal in allocation.items():
        score = capabilities[agent][goal]
        print(f"    {agent} → {goal} (оценка: {score:.2f})")

    # 2.2 Стратегии обязательств (commitment strategies)
    print("\nСтратегии обязательств (Commitment Strategies):")
    print("-" * 50)

    commitments = {
        "blind_commitment": "безусловное обязательство — агент следует плану до конца",
        "single-minded": "однозначное обязательство — агент следует плану, пока цель достижима",
        "open-minded": "открытое обязательство — агент пересматривает план при изменении обстоятельств",
        "leveling": "уровневое обязательство — агент стремится к лучшему из доступных планов",
    }
    for strategy, desc in commitments.items():
        print(f"  {strategy:22s} — {desc}")

    # 2.3 Модель обязательства
    print("\nМодель обязательства агента:")
    print("-" * 50)

    class Commitment:
        def __init__(self, agent, goal, strategy="single-minded"):
            self.agent = agent
            self.goal = goal
            self.strategy = strategy
            self.status = "committed"
            self.effort = 0
            self.progress = 0

        def step(self, progress_delta):
            self.effort += 1
            self.progress = min(1.0, self.progress + progress_delta)
            if self.progress >= 1.0:
                self.status = "achieved"
            elif self.effort > 10 and self.progress < 0.3:
                if self.strategy == "blind_commitment":
                    self.status = "persisting"
                elif self.strategy == "single-minded":
                    self.status = "reconsidering"
                elif self.strategy == "open-minded":
                    self.status = "abandoning"
            return self.status

    for strategy in ["blind_commitment", "single-minded", "open-minded"]:
        commitment = Commitment("Агент_X", "найти_решение", strategy)
        print(f"\n  Стратегия: {strategy}")
        for i in range(5):
            progress_delta = random.uniform(0.05, 0.15)
            status = commitment.step(progress_delta)
            print(f"    Шаг {i+1}: прогресс={commitment.progress:.2f}, статус={status}")

    # 2.4 Согласование целей (goal reconciliation)
    print("\nСогласование целей (Goal Reconciliation):")
    print("-" * 50)

    agent_goals = {
        "Агент_1": {"цель": "максимизировать_качество", "приоритет": 0.9},
        "Агент_2": {"цель": "минимизировать_стоимость", "приоритет": 0.7},
        "Агент_3": {"цель": "максимизировать_скорость", "приоритет": 0.8},
    }

    def reconcile_goals(goals_dict):
        """Простое согласование целей через взвешенное голосование."""
        weights = {a: g["приоритет"] for a, g in goals_dict.items()}
        total_weight = sum(weights.values())
        consensus = {}
        for a, g in goals_dict.items():
            weight = weights[a] / total_weight
            consensus[g["цель"]] = round(weight, 3)
        return consensus

    consensus = reconcile_goals(agent_goals)
    print("  Цели агентов:")
    for agent, info in agent_goals.items():
        print(f"    {agent}: {info['цель']} (приоритет={info['приоритет']})")
    print("  Веса целей в консенсусе:")
    for goal, weight in consensus.items():
        print(f"    {goal}: {weight}")

    print()


# ─────────────────────────── Демо 3: Формирование коалиций ───────────────────────────

def demo_coalition_formation():
    """Демонстрация характеристической функции, цены Шапли, ядра."""
    print("=" * 70)
    print("ДЕМО 3: Coalition Formation — характеристическая функция, цена Шапли")
    print("=" * 70)

    # 3.1 Характеристическая функция (characteristic function)
    print("Характеристическая функция v(S) — выигрыш коалиции:")
    print("-" * 50)

    # Агенты: A, B, C, D
    agents = ["A", "B", "C", "D"]
    # Характеристическая функция: v(S) = выигрыш подмножества агентов
    v = {
        frozenset(): 0,
        frozenset(["A"]): 10,
        frozenset(["B"]): 8,
        frozenset(["C"]): 12,
        frozenset(["D"]): 6,
        frozenset(["A", "B"]): 25,
        frozenset(["A", "C"]): 28,
        frozenset(["A", "D"]): 18,
        frozenset(["B", "C"]): 22,
        frozenset(["B", "D"]): 14,
        frozenset(["C", "D"]): 20,
        frozenset(["A", "B", "C"]): 45,
        frozenset(["A", "B", "D"]): 35,
        frozenset(["A", "C", "D"]): 40,
        frozenset(["B", "C", "D"]): 38,
        frozenset(["A", "B", "C", "D"]): 60,
    }

    print("  Коалиция          Выигрыш v(S)")
    for coalition in sorted(v.keys(), key=lambda s: (len(s), s)):
        name = "{" + ",".join(sorted(coalition)) + "}" if coalition else "{}"
        print(f"  {name:20s} {v[coalition]}")

    # 3.2 Цена Шапли (Shapley Value)
    print("\nЦена Шапли (Shapley Value):")
    print("-" * 50)

    def shapley_value(agent, agents_list, v_func):
        """Вычисление цены Шапли для агента."""
        n = len(agents_list)
        phi = 0

        # Перебор ВСЕХ перестановок всех агентов
        for perm in itertools.permutations(agents_list):
            # Позиция агента в перестановке
            pos = perm.index(agent)
            # Коалиция агентов ДО него (в порядке перестановки)
            coalition_before = frozenset(perm[:pos])
            # Коалиция с агентом
            coalition_with = coalition_before | frozenset([agent])
            # предельный вклад
            marginal = v_func.get(coalition_with, 0) - v_func.get(coalition_before, 0)
            phi += marginal

        return phi / math.factorial(n)

    shapley_values = {}
    for agent in agents:
        phi = shapley_value(agent, agents, v)
        shapley_values[agent] = phi
        print(f"  φ({agent}) = {phi:.2f}")

    total = sum(shapley_values.values())
    print(f"\n  Сумма Шапли: {total:.2f}")
    print(f"  v(N) = {v[frozenset(agents)]}")
    print(f"  Эффективность: {'ДА' if abs(total - v[frozenset(agents)]) < 0.01 else 'НЕТ'}")

    # 3.3 Ядро (Core)
    print("\nЯдро (Core) — набор стабильных распределений:")
    print("-" * 50)

    def is_in_core(distribution, agents_list, v_func):
        """Проверка: находится ли распределение в ядре?"""
        n = len(agents_list)
        total_value = v_func.get(frozenset(agents_list), 0)

        # Проверка эффективности
        if abs(sum(distribution.values()) - total_value) > 0.01:
            return False, "неэффективно"

        # Проверка суб-additivity для всех коалиций
        for size in range(1, n):
            for coalition in itertools.combinations(agents_list, size):
                coalition_set = frozenset(coalition)
                coalition_value = v_func.get(coalition_set, 0)
                distribution_value = sum(distribution[a] for a in coalition)
                if distribution_value < coalition_value - 0.01:
                    return False, f"коалиция {coalition} получает {distribution_value:.2f} < v(S)={coalition_value}"

        return True, "в ядре"

    # Тестирование распределений
    distributions = [
        {a: shapley_values[a] for a in agents},  # Шапли
        {a: v[frozenset(agents)] / len(agents) for a in agents},  # Равное
        {"A": 30, "B": 10, "C": 15, "D": 5},  # Произвольное
    ]

    for dist in distributions:
        in_core, reason = is_in_core(dist, agents, v)
        status = "ЯДРО" if in_core else f"ВНЕ ЯДРА ({reason})"
        total = sum(dist.values())
        print(f"  Распределение: { {a: round(dist[a], 2) for a in agents} }")
        print(f"    Сумма={total:.2f}, Статус: {status}")

    # 3.4 Простой аукцион коалиций
    print("\nАукцион коалиций (всеобщий аукцион):")
    print("-" * 50)

    def grand_auction(v_func, agents_list):
        """Простой аукцион: каждый агент назначает цену, лучшая коалиция побеждает."""
        best_coalition = None
        best_value = 0
        bids = {}

        for size in range(2, len(agents_list) + 1):
            for coalition in itertools.combinations(agents_list, size):
                coalition_set = frozenset(coalition)
                value = v_func.get(coalition_set, 0)
                # Среднее предложение на агента
                per_agent = value / len(coalition)
                bids[coalition] = {"value": value, "per_agent": per_agent}
                if value > best_value:
                    best_value = value
                    best_coalition = coalition

        return best_coalition, best_value, bids

    best_coalition, best_value, bids = grand_auction(v, agents)
    print(f"  Лучшая коалиция: {{{','.join(best_coalition)}}} = {best_value}")
    print(f"  На агента: {best_value/len(best_coalition):.2f}")
    print("\n  Все предложения:")
    for coalition in sorted(bids.keys()):
        info = bids[coalition]
        name = "{" + ",".join(coalition) + "}"
        print(f"    {name:15s}: v(S)={info['value']:3d}, на_агента={info['per_agent']:.2f}")

    print()


# ─────────────────────────── Демо 4: Стимулы к сотрудничеству ───────────────────────────

def demo_cooperation_incentives():
    """Демонстрация репутации, взаимности, наказания."""
    print("=" * 70)
    print("ДЕМО 4: Cooperation Incentives — репутация, взаимность, наказание")
    print("=" * 70)

    # 4.1 Система репутации
    print("Система репутации (Reputation):")
    print("-" * 50)

    class ReputationSystem:
        def __init__(self, agents, decay=0.95):
            self.agents = agents
            self.reputation = {a: 0.5 for a in agents}  # начальная репутация [0, 1]
            self.history = {a: [] for a in agents}
            self.decay = decay

        def update(self, agent, action, reward):
            """Обновление репутации на основе действия."""
            self.history[agent].append({"action": action, "reward": reward})
            # Экспоненциальное скользящее среднее
            old_rep = self.reputation[agent]
            self.reputation[agent] = old_rep * self.decay + reward * (1 - self.decay)

        def get_trust_score(self, agent):
            """Оценка доверия на основе репутации."""
            return self.reputation[agent]

        def report(self):
            for agent in self.agents:
                rep = self.reputation[agent]
                actions = len(self.history[agent])
                print(f"  {agent:10s}: репутация={rep:.3f}, действий={actions}")

    rep_system = ReputationSystem(["Агент_1", "Агент_2", "Агент_3"])

    # Симуляция действий
    actions = [
        ("Агент_1", "cooperate", 0.8),
        ("Агент_1", "cooperate", 0.9),
        ("Агент_2", "defect", 0.2),
        ("Агент_2", "cooperate", 0.7),
        ("Агент_3", "cooperate", 0.85),
        ("Агент_3", "cooperate", 0.75),
        ("Агент_1", "cooperate", 0.95),
        ("Агент_2", "defect", 0.1),
    ]
    print("  История действий:")
    for agent, action, reward in actions:
        rep_system.update(agent, action, reward)
        print(f"    {agent}: {action}, награда={reward:.2f}")

    print("\n  Текущая репутация:")
    rep_system.report()

    # 4.2 Принцип взаимности (Tit-for-Tat)
    print("\nПринцип взаимности (Tit-for-Tat):")
    print("-" * 50)

    def tit_for_tat(opponent_history):
        """Тактика 'око за око': повторяем предыдущее действие оппонента."""
        if not opponent_history:
            return "cooperate"
        return opponent_history[-1]

    def simulate_prisoners_dilemma(strategy_a, strategy_b, rounds=8):
        """Симуляция 'дилеммы заключённого'."""
        history_a = []
        history_b = []
        scores = {"A": 0, "B": 0}

        for rnd in range(1, rounds + 1):
            action_a = strategy_a(history_b)
            action_b = strategy_b(history_a)

            # Выигрыш: (cooperate, cooperate) = (3, 3)
            # (cooperate, defect) = (0, 5)
            # (defect, cooperate) = (5, 0)
            # (defect, defect) = (1, 1)
            if action_a == "cooperate" and action_b == "cooperate":
                scores["A"] += 3
                scores["B"] += 3
            elif action_a == "cooperate" and action_b == "defect":
                scores["A"] += 0
                scores["B"] += 5
            elif action_a == "defect" and action_b == "cooperate":
                scores["A"] += 5
                scores["B"] += 0
            else:
                scores["A"] += 1
                scores["B"] += 1

            history_a.append(action_a)
            history_b.append(action_b)
            print(f"    Раунд {rnd}: A={action_a:10s} B={action_b:10s} | Счёт: A={scores['A']}, B={scores['B']}")

        return scores

    def always_cooperate(_):
        return "cooperate"

    def always_defect(_):
        return "defect"

    def random_strategy(_):
        return random.choice(["cooperate", "defect"])

    print("  Tit-for-Tat vs Always-Cooperate:")
    scores = simulate_prisoners_dilemma(tit_for_tat, always_cooperate)
    print(f"  Итог: A={scores['A']}, B={scores['B']}")

    print("\n  Tit-for-Tat vs Always-Defect:")
    scores = simulate_prisoners_dilemma(tit_for_tat, always_defect)
    print(f"  Итог: A={scores['A']}, B={scores['B']}")

    # 4.3 Механизм наказания (punishment)
    print("\nМеханизм наказания (Punishment):")
    print("-" * 50)

    class PunishmentMechanism:
        def __init__(self, agents, fine_rate=0.3):
            self.agents = agents
            self.fine_rate = fine_rate
            self.violations = {a: 0 for a in agents}
            self.balance = {a: 100.0 for a in agents}  # начальный баланс

        def cooperate(self, agent, gain):
            """Сотрудничество: агент получает выгоду."""
            self.balance[agent] += gain
            return f"{agent} сотрудничает: +{gain:.2f}"

        def defect(self, agent, gain):
            """Нарушение: агент получает выгоду, но штрафуется."""
            self.balance[agent] += gain
            self.violations[agent] += 1
            fine = gain * self.fine_rate * self.violations[agent]
            self.balance[agent] -= fine
            return f"{agent} нарушает: +{gain:.2f}, штраф={fine:.2f} (нарушений: {self.violations[agent]})"

        def report(self):
            print("\n  Балансы:")
            for agent in self.agents:
                print(f"    {agent}: {self.balance[agent]:.2f} (нарушений: {self.violations[agent]})")

    punish = PunishmentMechanism(["Честный", "Нарушитель_1", "Нарушитель_2"])
    actions_punish = [
        ("Честный", "cooperate", 10),
        ("Нарушитель_1", "defect", 15),
        ("Нарушитель_2", "defect", 12),
        ("Честный", "cooperate", 10),
        ("Нарушитель_1", "defect", 15),
        ("Нарушитель_2", "cooperate", 8),
        ("Честный", "cooperate", 10),
        ("Нарушитель_1", "defect", 15),
    ]
    for agent, action, gain in actions_punish:
        if action == "cooperate":
            result = punish.cooperate(agent, gain)
        else:
            result = punish.defect(agent, gain)
        print(f"  {result}")
    punish.report()

    # 4.4 Механизм вознаграждения (reward mechanism)
    print("\nМеханизм вознаграждения (Reward Mechanism):")
    print("-" * 50)

    class RewardMechanism:
        def __init__(self, agents, base_reward=5.0):
            self.agents = agents
            self.base_reward = base_reward
            self.scores = {a: 0 for a in agents}
            self.bonuses = {a: 0 for a in agents}

        def evaluate(self, agent, quality):
            """Оценка качества работы."""
            self.scores[agent] += quality
            # Бонус за высокое качество
            if quality > 0.8:
                bonus = quality * 2
                self.bonuses[agent] += bonus
                return f"{agent}: качество={quality:.2f}, бонус=+{bonus:.2f}"
            return f"{agent}: качество={quality:.2f}, без бонуса"

        def report(self):
            print("\n  Итоги:")
            for agent in self.agents:
                total = self.scores[agent] + self.bonuses[agent]
                print(f"    {agent}: балл={self.scores[agent]:.2f}, бонус={self.bonuses[agent]:.2f}, итого={total:.2f}")

    reward = RewardMechanism(["Рабочий_1", "Рабочий_2", "Рабочий_3"])
    evaluations = [
        ("Рабочий_1", 0.9),
        ("Рабочий_2", 0.6),
        ("Рабочий_3", 0.85),
        ("Рабочий_1", 0.95),
        ("Рабочий_2", 0.7),
        ("Рабочий_3", 0.8),
    ]
    for agent, quality in evaluations:
        result = reward.evaluate(agent, quality)
        print(f"  {result}")
    reward.report()

    print()


if __name__ == "__main__":
    demo_joint_planning()
    demo_shared_goals()
    demo_coalition_formation()
    demo_cooperation_incentives()
