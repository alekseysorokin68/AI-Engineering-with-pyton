"""
190 — Agent Organizations: иерархии, команды, сети, роли

Темы:
  1. Организационные структуры (иерархия, плоская, матричная, сетевая)
  2. Назначение ролей (matching возможностей, балансировка нагрузки, эволюция ролей)
  3. Формирование команд (состав команды, дополняемые навыки, синергия)
  4. Организационное обучение (обмен знаниями, адаптация, реструктуризация)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# =============================================================================
# 1. ОРГАНИЗАЦИОННЫЕ СТРУКТУРЫ
# =============================================================================

def demo_organizational_structures():
    """Демонстрация организационных структур — иерархия, плоская, матричная, сетевая."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: ОРГАНИЗАЦИОННЫЕ СТРУКТУРЫ")
    print("=" * 70)

    # --- 1.1 Иерархическая структура ---
    print("\n--- 1.1 Иерархическая структура ---")
    print("Классическая пирамида: каждый подчиняется одному руководителю.\n")

    # Дерево агентов
    hierarchy = {
        "CEO": {"subordinates": ["CTO", "CPO", "CFO"], "level": 0},
        "CTO": {"subordinates": ["DevLead", "QA_Lead"], "level": 1},
        "CPO": {"subordinates": ["UX_Lead", "Data_Lead"], "level": 1},
        "CFO": {"subordinates": ["Accountant"], "level": 1},
        "DevLead": {"subordinates": ["Dev1", "Dev2"], "level": 2},
        "QA_Lead": {"subordinates": ["QA1"], "level": 2},
        "UX_Lead": {"subordinates": [], "level": 2},
        "Data_Lead": {"subordinates": [], "level": 2},
        "Accountant": {"subordinates": [], "level": 2},
        "Dev1": {"subordinates": [], "level": 3},
        "Dev2": {"subordinates": [], "level": 3},
        "QA1": {"subordinates": [], "level": 3},
    }

    # Визуализация дерева
    def print_tree(node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        children = hierarchy[node]["subordinates"]
        print(f"{prefix}{connector}{node}")
        for i, child in enumerate(children):
            extension = "    " if is_last else "│   "
            print_tree(child, prefix + extension, i == len(children) - 1)

    print("  Организационное дерево:")
    print("  CEO")
    for i, child in enumerate(hierarchy["CEO"]["subordinates"]):
        print_tree(child, "  ", i == len(hierarchy["CEO"]["subordinates"]) - 1)

    # Метрики иерархии
    total_agents = len(hierarchy)
    max_depth = max(info["level"] for info in hierarchy.values())
    spans = [len(info["subordinates"]) for info in hierarchy.values()]
    avg_span = sum(spans) / len(spans)
    non_leaf = sum(1 for info in hierarchy.values() if info["subordinates"])

    print(f"\n  Метрики иерархии:")
    print(f"    Всего агентов: {total_agents}")
    print(f"    Глубина дерева: {max_depth}")
    print(f"    Средний span of control: {avg_span:.1f}")
    print(f"    Руководителей (нет leaf): {non_leaf}")

    # Ширина communication pathway
    # В иерархии: message проходит через ceiling(level_diff) узлов
    print(f"    Communication path length (Dev1→QA1): ceil(|3-3|/2) + 2 = 4 узла (до общего руководителя)\n")

    # --- 1.2 Плоская структура ---
    print("--- 1.2 Плоская структура ---")
    print("Минимум уровней управления. Все агенты примерно равны.\n")

    flat_team = [
        {"name": "Agent_A", "skills": ["code", "review"], "load": 0.7},
        {"name": "Agent_B", "skills": ["test", "deploy"], "load": 0.5},
        {"name": "Agent_C", "skills": ["design", "code"], "load": 0.8},
        {"name": "Agent_D", "skills": ["data", "ml"], "load": 0.6},
        {"name": "Agent_E", "skills": ["ops", "monitor"], "load": 0.4},
    ]

    print("  Плоская команда (все агенты на одном уровне):")
    for agent in flat_team:
        print(f"    {agent['name']}: навыки={agent['skills']}, загрузка={agent['load']:.0%}")

    avg_load = sum(a["load"] for a in flat_team) / len(flat_team)
    max_load = max(a["load"] for a in flat_team)
    min_load = min(a["load"] for a in flat_team)
    load_variance = sum((a["load"] - avg_load) ** 2 for a in flat_team) / len(flat_team)

    print(f"\n  Метрики плоской структуры:")
    print(f"    Средняя загрузка: {avg_load:.1%}")
    print(f"    Макс/мин загрузка: {max_load:.1%} / {min_load:.1%}")
    print(f"    Дисперсия нагрузки: {load_variance:.4f}")
    print(f"    Communication path: O(1) — все связаны напрямую\n")

    # --- 1.3 Матричная структура ---
    print("--- 1.3 Матричная структура ---")
    print("Двойное подчинение: агенты имеют функционального и проектного руководителя.\n")

    # Функциональные департаменты
    functional = {
        "Engineering": ["Dev1", "Dev2", "Dev3"],
        "Data": ["Data1", "Data2"],
        "Design": ["Design1"],
    }
    # Проектные команды
    projects = {
        "Project_Alpha": ["Dev1", "Data1", "Design1"],
        "Project_Beta": ["Dev2", "Dev3", "Data2"],
    }

    print("  Функциональные департаменты:")
    for dept, members in functional.items():
        print(f"    {dept}: {members}")

    print("\n  Проектные команды:")
    for proj, members in projects.items():
        print(f"    {proj}: {members}")

    # Матрица назначений (кто в каком проекте и департаменте)
    assignments = {}
    for dept, members in functional.items():
        for member in members:
            assignments[member] = {"dept": dept, "projects": []}
    for proj, members in projects.items():
        for member in members:
            if member in assignments:
                assignments[member]["projects"].append(proj)

    print("\n  Матрица назначений:")
    for agent, info in assignments.items():
        projs = ", ".join(info["projects"]) if info["projects"] else "нет"
        print(f"    {agent}: {info['dept']} → проекты: {projs}")

    # Конфликт интересов: Dev1 одновременно в Alpha и Engineering
    print("\n  Потенциальные конфликты:")
    for agent, info in assignments.items():
        if len(info["projects"]) > 1:
            print(f"    {agent}:双重 подчинение ({info['projects']}) — нужен приоритет!")
    print()

    # --- 1.4 Сетевая структура ---
    print("--- 1.4 Сетевая структура ---")
    print("Децентрализованная: агенты связаны по потребности, нет фиксированной иерархии.\n")

    # Граф связей агентов
    network_edges = [
        ("Agent_1", "Agent_2"), ("Agent_1", "Agent_3"),
        ("Agent_2", "Agent_3"), ("Agent_2", "Agent_4"),
        ("Agent_3", "Agent_5"), ("Agent_4", "Agent_5"),
        ("Agent_4", "Agent_6"), ("Agent_5", "Agent_6"),
        ("Agent_6", "Agent_1"),  # кольцевая связь
    ]

    # Строим adjacency list
    adj = collections.defaultdict(list)
    for a, b in network_edges:
        adj[a].append(b)
        adj[b].append(a)

    print("  Граф связей:")
    for node in sorted(adj.keys()):
        neighbors = ", ".join(sorted(adj[node]))
        print(f"    {node} → [{neighbors}]")

    # Центральность по degree
    print("\n  Degree centrality (количество связей):")
    n_nodes = len(adj)
    for node in sorted(adj.keys(), key=lambda x: -len(adj[x])):
        centrality = len(adj[node]) / (n_nodes - 1)
        print(f"    {node}: degree={len(adj[node])}, centrality={centrality:.2f}")

    # BFS для поиска shortest path
    def bfs_shortest(adj, start, end):
        visited = {start}
        queue = collections.deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == end:
                return dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1

    # Diameter графа
    max_dist = 0
    for node1 in adj:
        for node2 in adj:
            if node1 < node2:
                d = bfs_shortest(adj, node1, node2)
                max_dist = max(max_dist, d)

    print(f"\n  Diameter графа (максимальный shortest path): {max_dist}")
    print("  => Сетевая структура: быстрый обмен информацией, но сложная координация\n")


# =============================================================================
# 2. НАЗНАЧЕНИЕ РОЛЕЙ
# =============================================================================

def demo_role_assignment():
    """Демонстрация назначения ролей — matching возможностей, балансировка нагрузки."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: НАЗНАЧЕНИЕ РОЛЕЙ")
    print("=" * 70)

    # --- 2.1 Matching возможностей ---
    print("\n--- 2.1 Matching возможностей (capability matching) ---")
    print("Назначение ролей на основе навыков агентов и требований задач.\n")

    agents = {
        "Alice": {"code": 0.9, "test": 0.7, "design": 0.3, "ml": 0.8},
        "Bob": {"code": 0.6, "test": 0.9, "design": 0.4, "ml": 0.5},
        "Carol": {"code": 0.7, "test": 0.5, "design": 0.9, "ml": 0.4},
        "Dave": {"code": 0.5, "test": 0.6, "design": 0.2, "ml": 0.9},
    }

    tasks = {
        "Backend_API": {"code": 0.8, "test": 0.5},
        "Frontend_UI": {"code": 0.6, "design": 0.8},
        "ML_Pipeline": {"code": 0.7, "ml": 0.9},
        "QA_Suite": {"test": 0.9, "code": 0.3},
    }

    print("  Навыки агентов:")
    for agent, skills in agents.items():
        skill_str = ", ".join(f"{k}={v:.1f}" for k, v in skills.items())
        print(f"    {agent}: {skill_str}")

    print("\n  Требования задач:")
    for task, reqs in tasks.items():
        req_str = ", ".join(f"{k}>={v}" for k, v in reqs.items())
        print(f"    {task}: {req_str}")

    # Жадное назначение: для каждой задачи выбираем агента с лучшим matching score
    def match_score(agent_skills, task_reqs):
        """Вычисляем score matching: среднее соответствия навыков требованиям."""
        score = 0
        for skill, req in task_reqs.items():
            agent_level = agent_skills.get(skill, 0)
            score += min(agent_level / req, 1.0)  # нормализуем до 1.0
        return score / len(task_reqs) if task_reqs else 0

    print("\n  Матрица scores (агент × задача):")
    print(f"  {'':>12}", end="")
    for task in tasks:
        print(f" {task:>14}", end="")
    print()
    for agent in agents:
        print(f"  {agent:>12}", end="")
        for task in tasks:
            score = match_score(agents[agent], tasks[task])
            print(f" {score:>14.2f}", end="")
        print()

    # Жадное назначение
    assignments = {}
    used_agents = set()
    for task in tasks:
        best_agent = None
        best_score = -1
        for agent in agents:
            if agent not in used_agents:
                score = match_score(agents[agent], tasks[task])
                if score > best_score:
                    best_score = score
                    best_agent = agent
        assignments[task] = (best_agent, best_score)
        used_agents.add(best_agent)

    print("\n  Жадное назначение:")
    for task, (agent, score) in assignments.items():
        print(f"    {task} → {agent} (score={score:.2f})")

    # --- 2.2 Балансировка нагрузки ---
    print("\n--- 2.2 Балансировка нагрузки ---")
    print("Распределение задач для минимизации дисбаланса загрузки.\n")

    agent_workload = {"Alice": 2, "Bob": 1, "Carol": 3, "Dave": 1}
    task_complexity = {"Task_A": 3, "Task_B": 2, "Task_C": 4, "Task_D": 1, "Task_E": 2}

    print("  Текущая загрузка агентов:")
    for agent, load in agent_workload.items():
        print(f"    {agent}: {load} единиц")

    print("  Сложность новых задач:")
    for task, complexity in task_complexity.items():
        print(f"    {task}: {complexity} единиц")

    # Балансировка: назначаем задачи агенту с минимальной текущей загрузкой
    balanced = {}
    workload = dict(agent_workload)
    for task, complexity in sorted(task_complexity.items(), key=lambda x: -x[1]):
        best_agent = min(workload, key=workload.get)
        balanced[task] = best_agent
        workload[best_agent] += complexity

    print("\n  Балансированное назначение:")
    for task, agent in balanced.items():
        print(f"    {task} → {agent} (задача={task_complexity[task]})")

    print("\n  Итоговая загрузка:")
    for agent, load in sorted(workload.items(), key=lambda x: -x[1]):
        bar = "█" * load
        print(f"    {agent}: {load} {bar}")

    final_loads = list(workload.values())
    load_range = max(final_loads) - min(final_loads)
    print(f"\n  Диапазон нагрузки: {load_range} (min={min(final_loads)}, max={max(final_loads)})")
    print(f"  Коэффициент вариации: {math.sqrt(sum((l - sum(final_loads)/len(final_loads))**2 for l in final_loads) / len(final_loads)) / (sum(final_loads)/len(final_loads)):.2f}\n")

    # --- 2.3 Эволюция ролей ---
    print("--- 2.3 Эволюция ролей (role evolution) ---")
    print("Роли меняются со временем на основе производительности.\n")

    agent_history = {
        "Agent_X": [
            {"role": "Junior Dev", "performance": 0.6, "tasks_completed": 10},
            {"role": "Mid Dev", "performance": 0.75, "tasks_completed": 25},
            {"role": "Senior Dev", "performance": 0.85, "tasks_completed": 40},
        ],
        "Agent_Y": [
            {"role": "Data Analyst", "performance": 0.7, "tasks_completed": 15},
            {"role": "ML Engineer", "performance": 0.8, "tasks_completed": 20},
            {"role": "ML Lead", "performance": 0.9, "tasks_completed": 35},
        ],
    }

    print("  История эволюции ролей:")
    for agent, history in agent_history.items():
        print(f"\n    {agent}:")
        for step, entry in enumerate(history):
            arrow = " → " if step > 0 else "   "
            print(f"    {arrow}{entry['role']} (perf={entry['performance']:.2f}, tasks={entry['tasks_completed']})")

    # Правила повышения
    print("\n  Правила повышения:")
    print("    Если performance > 0.8 И tasks_completed > 30 → следующая роль")
    print("    Если performance < 0.5 → понижение до предыдущей роли")

    # Проверяем текущий статус
    for agent, history in agent_history.items():
        latest = history[-1]
        if latest["performance"] > 0.8 and latest["tasks_completed"] > 30:
            print(f"    {agent}: кандидат на следующую роль (performance={latest['performance']:.2f})")

    # --- 2.4 Оптимальное назначение (Hungarian-like) ---
    print("\n--- 2.4 Оптимальное назначение (жадный Hungarian) ---")
    print("Назначение 1:1 для max total match score.\n")

    agents_list = list(agents.keys())
    tasks_list = list(tasks.keys())
    n = min(len(agents_list), len(tasks_list))

    # Жадное назначение с максимальным score
    remaining_agents = list(agents_list)
    remaining_tasks = list(tasks_list)
    optimal = {}

    for _ in range(n):
        best_score = -1
        best_pair = None
        for a in remaining_agents:
            for t in remaining_tasks:
                s = match_score(agents[a], tasks[t])
                if s > best_score:
                    best_score = s
                    best_pair = (a, t)
        if best_pair:
            optimal[best_pair[1]] = (best_pair[0], best_score)
            remaining_agents.remove(best_pair[0])
            remaining_tasks.remove(best_pair[1])

    total_score = sum(s for _, s in optimal.values())
    print("  Оптимальное 1:1 назначение:")
    for task, (agent, score) in optimal.items():
        print(f"    {task} → {agent} (score={score:.2f})")
    print(f"  Суммарный score: {total_score:.2f}\n")


# =============================================================================
# 3. ФОРМИРОВАНИЕ КОМАНД
# =============================================================================

def demo_team_formation():
    """Демонстрация формирования команд — состав, дополняемые навыки, синергия."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: ФОРМИРОВАНИЕ КОМАНД")
    print("=" * 70)

    # --- 3.1 Состав команды ---
    print("\n--- 3.1 Состав команды (team composition) ---")
    print("Формирование команды с балансом навыков и опыта.\n")

    candidate_pool = {
        "Alice": {"skills": {"frontend": 9, "backend": 7, "devops": 3}, "exp": 5, "collab": 0.8},
        "Bob": {"skills": {"frontend": 4, "backend": 9, "devops": 8}, "exp": 8, "collab": 0.6},
        "Carol": {"skills": {"frontend": 7, "backend": 5, "devops": 6}, "exp": 3, "collab": 0.9},
        "Dave": {"skills": {"frontend": 8, "backend": 6, "devops": 4}, "exp": 4, "collab": 0.7},
        "Eve": {"skills": {"frontend": 3, "backend": 8, "devops": 9}, "exp": 7, "collab": 0.5},
        "Frank": {"skills": {"frontend": 6, "backend": 7, "devops": 5}, "exp": 6, "collab": 0.8},
    }

    team_requirements = {
        "frontend": 8,  # минимум 8 по frontend
        "backend": 7,   # минимум 7 по backend
        "devops": 6,    # минимум 6 по devops
    }

    print("  Кандидаты:")
    for name, info in candidate_pool.items():
        skills_str = ", ".join(f"{k}={v}" for k, v in info["skills"].items())
        print(f"    {name}: {skills_str} | exp={info['exp']} | collab={info['collab']:.1f}")

    print(f"\n  Требования команды: {team_requirements}")

    # Формируем команду: жадный выбор покрывающих требования
    def team_coverage(team, reqs):
        """Проверяет, покрывает ли команда требования."""
        for skill, min_level in reqs.items():
            max_level = max(candidate_pool[agent]["skills"].get(skill, 0) for agent in team)
            if max_level < min_level:
                return False
        return True

    # Жадный алгоритм: добавляем агента, который больше всего улучшает покрытие
    team = []
    remaining = list(candidate_pool.keys())

    for _ in range(4):  # команда из 4 человек
        best_agent = None
        best_improvement = -1
        for agent in remaining:
            test_team = team + [agent]
            # Считаем покрытие: для каждого навыка — max level
            coverage = 0
            for skill, req_level in team_requirements.items():
                max_level = max(candidate_pool[a]["skills"].get(skill, 0) for a in test_team)
                coverage += min(max_level / req_level, 1.0)
            improvement = coverage / len(team_requirements)
            if improvement > best_improvement:
                best_improvement = improvement
                best_agent = agent
        team.append(best_agent)
        remaining.remove(best_agent)

    print(f"\n  Сформированная команда:")
    team_skills = collections.defaultdict(int)
    for agent in team:
        for skill, level in candidate_pool[agent]["skills"].items():
            team_skills[skill] = max(team_skills[skill], level)
        print(f"    {agent}: {candidate_pool[agent]['skills']}")

    print(f"\n  Покрытие требований:")
    for skill, req in team_requirements.items():
        achieved = team_skills[skill]
        status = "OK" if achieved >= req else "FAIL"
        print(f"    {skill}: {achieved}/{req} [{status}]")

    # --- 3.2 Дополняемые навыки ---
    print("\n--- 3.2 Дополняемые навыки (complementary skills) ---")
    print("Ищем пары агентов с максимальным complementarity score.\n")

    def complementarity(agent1, agent2):
        """Score дополнительности: разница навыков越高, тем лучше."""
        skills1 = candidate_pool[agent1]["skills"]
        skills2 = candidate_pool[agent2]["skills"]
        all_skills = set(skills1.keys()) | set(skills2.keys())
        score = 0
        for skill in all_skills:
            diff = abs(skills1.get(skill, 0) - skills2.get(skill, 0))
            score += diff
        return score

    print("  Топ-5 пар по дополнительности:")
    pairs = []
    agents_list = list(candidate_pool.keys())
    for i in range(len(agents_list)):
        for j in range(i + 1, len(agents_list)):
            score = complementarity(agents_list[i], agents_list[j])
            pairs.append((agents_list[i], agents_list[j], score))

    pairs.sort(key=lambda x: -x[2])
    for a1, a2, score in pairs[:5]:
        print(f"    {a1} + {a2}: complementarity={score}")

    # --- 3.3 Синергия команды ---
    print("\n--- 3.3 Синергия команды (team synergy) ---")
    print("Синергия = команда产出 > sum(individual outputs)\n")

    # Модель: производительность команды зависит от совместимости
    def team_synergy(team_agents):
        """Вычисляем synergy score."""
        total = 0
        n = len(team_agents)
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = team_agents[i], team_agents[j]
                # Synergy = complementarity × average collaboration
                comp = complementarity(a1, a2)
                collab = (candidate_pool[a1]["collab"] + candidate_pool[a2]["collab"]) / 2
                total += comp * collab
        # Нормализуем на количество пар
        n_pairs = n * (n - 1) / 2
        return total / n_pairs if n_pairs > 0 else 0

    # Сравниваем разные команды
    teams_to_compare = [
        ("Команда A", ["Alice", "Bob", "Carol"]),
        ("Команда B", ["Alice", "Eve", "Frank"]),
        ("Команда C", ["Bob", "Dave", "Eve"]),
    ]

    print("  Сравнение synergy разных команд:")
    best_synergy = -1
    best_team_name = ""
    for team_name, members in teams_to_compare:
        syn = team_synergy(members)
        bar = "█" * int(syn * 2)
        print(f"    {team_name}: synergy={syn:.2f} {bar}")
        if syn > best_synergy:
            best_synergy = syn
            best_team_name = team_name

    print(f"\n  Лучшая команда по synergy: {best_team_name} (score={best_synergy:.2f})")

    # --- 3.4 Оптимальная команда для проекта ---
    print("\n--- 3.4 Оптимальная команда для проекта ---")

    project = {
        "name": "E-commerce Platform",
        "requirements": {"frontend": 8, "backend": 7, "devops": 6},
        "budget": 3,  # максимум 3 агента
    }

    print(f"  Проект: {project['name']}")
    print(f"  Требования: {project['requirements']}")
    print(f"  Бюджет: {project['budget']} агента\n")

    # Перебор всех комбинаций из 3 агентов
    import itertools
    best_combo = None
    best_total = -1

    for combo in itertools.combinations(candidate_pool.keys(), project["budget"]):
        # Оцениваем покрытие + synergy
        coverage = 0
        for skill, req in project["requirements"].items():
            max_level = max(candidate_pool[a]["skills"].get(skill, 0) for a in combo)
            coverage += min(max_level / req, 1.0)
        coverage /= len(project["requirements"])

        syn = team_synergy(list(combo))
        total = coverage * 0.7 + syn * 0.3  # взвешенная сумма

        if total > best_total:
            best_total = total
            best_combo = combo

    print(f"  Оптимальная команда:")
    for agent in best_combo:
        print(f"    {agent}: {candidate_pool[agent]['skills']}")
    print(f"  Total score: {best_total:.2f} (coverage × 0.7 + synergy × 0.3)\n")


# =============================================================================
# 4. ОРГАНИЗАЦИОННОЕ ОБУЧЕНИЕ
# =============================================================================

def demo_organizational_learning():
    """Демонстрация организационного обучения — обмен знаниями, адаптация, реструктуризация."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: ОРГАНИЗАЦИОННОЕ ОБУЧЕНИЕ")
    print("=" * 70)

    # --- 4.1 Обмен знаниями ---
    print("\n--- 4.1 Обмен знаниями (knowledge sharing) ---")
    print("Распространение навыков между агентами через взаимодействие.\n")

    # Модель: агенты имеют уровни навыков, обмен усредняет
    agents_skills = {
        "Agent_1": {"code": 0.9, "test": 0.3},
        "Agent_2": {"code": 0.4, "test": 0.8},
        "Agent_3": {"code": 0.6, "test": 0.5},
    }

    print("  Начальные навыки:")
    for agent, skills in agents_skills.items():
        print(f"    {agent}: {skills}")

    # Симуляция обмена: каждый шаг агенты делятся навыками
    def knowledge_sharing_step(agents, sharing_rate=0.1):
        """Один шаг обмена знаниями."""
        new_skills = {}
        for agent in agents:
            new_skills[agent] = dict(agents[agent])

        for agent in agents:
            for other in agents:
                if agent != other:
                    for skill in agents[agent]:
                        # Агент учится у другого с sharing_rate
                        current = agents[agent][skill]
                        other_level = agents[other][skill]
                        new_skills[agent][skill] = current + sharing_rate * (other_level - current)
        return new_skills

    print("\n  Симуляция обмена (10 шагов, rate=0.1):")
    for step in range(1, 11):
        agents_skills = knowledge_sharing_step(agents_skills, 0.1)
        if step in [1, 5, 10]:
            print(f"\n  Шаг {step}:")
            for agent, skills in agents_skills.items():
                skill_str = ", ".join(f"{k}={v:.3f}" for k, v in skills.items())
                print(f"    {agent}: {skill_str}")

    print("\n  Итог: все агенты выровнялись по навыкам (конвергенция)")

    # --- 4.2 Адаптация к изменениям ---
    print("\n--- 4.2 Адаптация к изменениям ---")
    print("Организация реагирует на изменения внешней среды.\n")

    class AdaptiveOrganization:
        def __init__(self, agents, environment):
            self.agents = agents
            self.env = environment
            self.adaptation_rate = 0.2
            self.history = []

        def assess_fitness(self):
            """Оцениваем пригодность каждого агента к текущей среде."""
            fitness = {}
            for agent, skills in self.agents.items():
                # Fitness = насколько навыки соответствуют потребностям среды
                score = 0
                for skill, importance in self.env.items():
                    agent_level = skills.get(skill, 0)
                    score += agent_level * importance
                fitness[agent] = score
            return fitness

        def adapt(self):
            """Адаптируем навыки агентов к среде."""
            fitness = self.assess_fitness()
            for agent in self.agents:
                for skill in self.env:
                    current = self.agents[agent].get(skill, 0)
                    importance = self.env[skill]
                    # Увеличиваем навыки, важные для среды
                    self.agents[agent][skill] = current + self.adaptation_rate * importance * (1 - current)
            return fitness

    env_v1 = {"code": 0.5, "test": 0.3, "deploy": 0.2}
    org = AdaptiveOrganization(
        agents_skills,
        env_v1
    )

    print("  Среда v1 (code=0.5, test=0.3, deploy=0.2):")
    for step in range(1, 6):
        fitness = org.adapt()
        print(f"    Шаг {step}: fitness = {', '.join(f'{k}={v:.2f}' for k, v in fitness.items())}")

    # Среда меняется!
    print("\n  Среда меняется! (code=0.2, test=0.5, deploy=0.3)")
    org.env = {"code": 0.2, "test": 0.5, "deploy": 0.3}
    for step in range(1, 6):
        fitness = org.adapt()
        print(f"    Шаг {step}: fitness = {', '.join(f'{k}={v:.2f}' for k, v in fitness.items())}")

    print("  => Организация адаптировалась к новой среде\n")

    # --- 4.3 Реструктуризация ---
    print("--- 4.3 Реструктуризация (restructuring) ---")
    print("Изменение структуры организации для повышения эффективности.\n")

    # Текущая структура: иерархия
    current_structure = {
        "Manager": {"team": ["Dev1", "Dev2", "Dev3"], "overhead": 0.3},
        "Dev1": {"skills": ["backend"], "productivity": 0.8},
        "Dev2": {"skills": ["backend", "frontend"], "productivity": 0.7},
        "Dev3": {"skills": ["frontend"], "productivity": 0.9},
    }

    # Эффективность = суммарная производительность - overhead
    def org_efficiency(structure):
        total_prod = sum(info.get("productivity", 0) for info in structure.values() if "productivity" in info)
        overhead = sum(info.get("overhead", 0) for info in structure.values() if "overhead" in info)
        return total_prod - overhead

    print(f"  Текущая структура (иерархия):")
    for name, info in current_structure.items():
        print(f"    {name}: {info}")
    print(f"  Эффективность: {org_efficiency(current_structure):.2f}")

    # Вариант 1: плоская структура (без менеджера)
    flat_structure = {
        "Dev1": {"skills": ["backend"], "productivity": 0.8},
        "Dev2": {"skills": ["backend", "frontend"], "productivity": 0.7},
        "Dev3": {"skills": ["frontend"], "productivity": 0.9},
    }
    print(f"\n  Вариант 1 (плоская):")
    print(f"  Эффективность: {org_efficiency(flat_structure):.2f}")

    # Вариант 2: команда с техлидом
    team_structure = {
        "TechLead": {"skills": ["backend", "frontend"], "productivity": 0.6, "overhead": 0.1},
        "Dev1": {"skills": ["backend"], "productivity": 0.85},
        "Dev2": {"skills": ["backend"], "productivity": 0.8},
        "Dev3": {"skills": ["frontend"], "productivity": 0.95},
    }
    print(f"\n  Вариант 2 (команда с TechLead):")
    print(f"  Эффективность: {org_efficiency(team_structure):.2f}")

    variants = [
        ("Иерархия", current_structure),
        ("Плоская", flat_structure),
        ("Команда с TechLead", team_structure),
    ]
    best = max(variants, key=lambda x: org_efficiency(x[1]))
    print(f"\n  Лучшая структура: {best[0]} (efficiency={org_efficiency(best[1]):.2f})")

    # --- 4.4 Организационная память ---
    print("\n--- 4.4 Организационная память ---")
    print("Накопление и использование опыта организации.\n")

    org_memory = {
        "lessons": [
            {"project": "Alpha", "lesson": "Слишком длинные спринты снижают мораль", "impact": -0.2},
            {"project": "Beta", "lesson": "Автоматизация тестов экономит 30% времени", "impact": 0.3},
            {"project": "Gamma", "lesson": "Парный programming улучшает качество", "impact": 0.15},
        ],
        "best_practices": [
            "Делай code review для каждого PR",
            "Деплой через feature flags",
            "Мониторинг в production",
        ],
        "anti_patterns": [
            "Big bang releases",
            "No automated tests",
            "Hero culture",
        ],
    }

    print("  Уроки проектов:")
    for lesson in org_memory["lessons"]:
        sign = "+" if lesson["impact"] > 0 else ""
        print(f"    [{lesson['project']}] {lesson['lesson']} (impact: {sign}{lesson['impact']:.0%})")

    print("\n  Best practices:")
    for bp in org_memory["best_practices"]:
        print(f"    ✓ {bp}")

    print("\n  Anti-patterns:")
    for ap in org_memory["anti_patterns"]:
        print(f"    ✗ {ap}")

    # Индекс полезности памяти
    total_impact = sum(l["impact"] for l in org_memory["lessons"])
    print(f"\n  Суммарный impact уроков: {total_impact:+.2f}")
    print("  => Организация накопила положительный опыт, готова к новым проектам\n")


# =============================================================================
# ГЛАВНЫЙ БЛОК
# =============================================================================

if __name__ == "__main__":
    demo_organizational_structures()
    demo_role_assignment()
    demo_team_formation()
    demo_organizational_learning()
