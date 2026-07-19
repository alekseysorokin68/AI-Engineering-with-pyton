"""196 — Simulation & Testing: агентное моделирование, социальная симуляция

Темы:
  1. Agent-Based Modeling (agents, environment, rules, emergence)
  2. Simulation Design (parameters, scenarios, metrics, visualization)
  3. Testing Multi-Agent Systems (integration tests, scenario tests, stress tests)
  4. Analysis & Validation (statistical analysis, sensitivity analysis, calibration)

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


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 1: Агентное моделирование (ABM)
# ──────────────────────────────────────────────────────────────────────────────

class Agent:
    """Агент в ABM-модели — обладает состоянием и правилами поведения."""

    def __init__(self, agent_id, x, y, state="normal"):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.state = state
        self.energy = random.randint(50, 100)
        self.neighbors = []
        self.history = [state]

    def move(self, grid_size):
        """Случайное движение агента на сетке."""
        dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.x = max(0, min(grid_size - 1, self.x + dx))
        self.y = max(0, min(grid_size - 1, self.y + dy))
        self.energy -= 1

    def interact(self, other_agent):
        """Взаимодействие между агентами (заражение / обмен энергией)."""
        if self.state == "infected" and other_agent.state == "normal":
            # Вероятность заражения зависит от расстояния
            dist = math.sqrt((self.x - other_agent.x)**2 + (self.y - other_agent.y)**2)
            prob = max(0, 1 - dist / 5)
            if random.random() < prob:
                other_agent.state = "infected"
                return True
        elif self.state == "normal" and other_agent.state == "infected":
            dist = math.sqrt((self.x - other_agent.x)**2 + (self.y - other_agent.y)**2)
            prob = max(0, 1 - dist / 5)
            if random.random() < prob:
                self.state = "infected"
                return True
        return False


class Environment:
    """Среда для агентного моделирования."""

    def __init__(self, size=20, num_agents=50):
        self.size = size
        self.agents = []
        self.grid = [[[] for _ in range(size)] for _ in range(size)]
        self.tick = 0

        # Создаём агентов
        for i in range(num_agents):
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            state = "infected" if i < 3 else "normal"  # 3 заражённых стартовых
            agent = Agent(i, x, y, state)
            self.agents.append(agent)
            self.grid[x][y].append(i)

    def update(self):
        """Один шаг симуляции."""
        self.tick += 1

        # Перемещаем агентов
        for agent in self.agents:
            old_x, old_y = agent.x, agent.y
            agent.move(self.size)
            if (old_x, old_y) != (agent.x, agent.y):
                self.grid[old_x][old_y] = [
                    i for i in self.grid[old_x][old_y] if i != agent.agent_id
                ]
                self.grid[agent.x][agent.y].append(agent.agent_id)

        # Взаимодействие между агентами на одной клетке
        infections = 0
        for x in range(self.size):
            for y in range(self.size):
                cell_agents = self.grid[x][y]
                for i in range(len(cell_agents)):
                    for j in range(i + 1, len(cell_agents)):
                        a1 = self.agents[cell_agents[i]]
                        a2 = self.agents[cell_agents[j]]
                        if a1.interact(a2):
                            infections += 1

        return infections

    def get_stats(self):
        """Собирает статистику по симуляции."""
        states = collections.Counter(a.state for a in self.agents)
        avg_energy = sum(a.energy for a in self.agents) / len(self.agents)
        return {
            "tick": self.tick,
            "total_agents": len(self.agents),
            "states": dict(states),
            "avg_energy": round(avg_energy, 2),
            "infected_ratio": round(states.get("infected", 0) / len(self.agents), 3)
        }


def demo_agent_based_modeling():
    """Демонстрация агентного моделирования."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: Агентное моделирование (ABM)")
    print("=" * 70)

    # Пример 1: Создание среды и агентов
    print("\n--- Пример 1: Инициализация среды ---")
    env = Environment(size=10, num_agents=20)
    stats = env.get_stats()
    print(f"  Размер сетки: {env.size}x{env.size}")
    print(f"  Количество агентов: {stats['total_agents']}")
    print(f"  Начальное состояние: {stats['states']}")
    print(f"  Начальный уровень заражения: {stats['infected_ratio']}")

    # Пример 2: Динамика распространения
    print("\n--- Пример 2: Динамика распространения ---")
    env2 = Environment(size=15, num_agents=30)
    print(f"  Тик | Заражено | Здоровых | Энергия")
    print(f"  ----|----------|----------|--------")
    for _ in range(8):
        env2.update()
        s = env2.get_stats()
        print(f"  {s['tick']:4d} | {s['states'].get('infected', 0):8d} | "
              f"{s['states'].get('normal', 0):8d} | {s['avg_energy']:.1f}")

    # Пример 3: Emergence —Emergentные паттерны
    print("\n--- Пример 3: Emergence — возникающие паттерны ---")
    env3 = Environment(size=12, num_agents=25)
    cluster_history = []
    for _ in range(10):
        env3.update()
        # Подсчитываем кластеры заражённых
        infected = [(a.x, a.y) for a in env3.agents if a.state == "infected"]
        if len(infected) >= 2:
            # Среднее расстояние между заражёнными
            total_dist = 0
            count = 0
            for i in range(len(infected)):
                for j in range(i + 1, len(infected)):
                    d = math.sqrt((infected[i][0] - infected[j][0])**2 +
                                  (infected[i][1] - infected[j][1])**2)
                    total_dist += d
                    count += 1
            avg_dist = total_dist / count if count > 0 else 0
        else:
            avg_dist = 0
        cluster_history.append(avg_dist)
        print(f"  Тик {env3.tick}: среднее расстояние между заражёнными = {avg_dist:.2f}")

    # Пример 4: Энергетика агентов
    print("\n--- Пример 4: Энергетика агентов ---")
    env4 = Environment(size=10, num_agents=15)
    energies = [a.energy for a in env4.agents]
    print(f"  Начальные энергии: {energies}")
    print(f"  Мин: {min(energies)}, Макс: {max(energies)}, "
          f"Средняя: {sum(energies)/len(energies):.1f}")
    for _ in range(5):
        env4.update()
    final_energies = [a.energy for a in env4.agents]
    print(f"  После 5 тиков: {final_energies}")
    print(f"  Мин: {min(final_energies)}, Макс: {max(final_energies)}, "
          f"Средняя: {sum(final_energies)/len(final_energies):.1f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 2: Дизайн симуляций
# ──────────────────────────────────────────────────────────────────────────────

class SimulationDesign:
    """Дизайн симуляций с параметрами, сценариями и метриками."""

    def __init__(self, name):
        self.name = name
        self.parameters = {}
        self.scenarios = {}
        self.results = {}

    def add_parameter(self, name, default, min_val, max_val, description=""):
        """Добавляет параметр симуляции."""
        self.parameters[name] = {
            "default": default,
            "min": min_val,
            "max": max_val,
            "current": default,
            "description": description
        }

    def add_scenario(self, name, overrides):
        """Добавляет сценарий (набор параметров)."""
        self.scenarios[name] = overrides

    def run_scenario(self, scenario_name, steps=10):
        """Запускает сценарий и собирает метрики."""
        if scenario_name not in self.scenarios:
            print(f"  Сценарий '{scenario_name}' не найден!")
            return None

        # Применяем параметры сценария
        params = {k: v["default"] for k, v in self.parameters.items()}
        params.update(self.scenarios[scenario_name])

        # Симуляция (линейная модель роста с шумом)
        trajectory = []
        value = params.get("initial_value", 100)
        growth_rate = params.get("growth_rate", 0.05)
        noise = params.get("noise", 0.1)
        capacity = params.get("capacity", 1000)

        for step in range(steps):
            # Логистическая модель: dN/dt = r * N * (1 - N/K)
            delta = growth_rate * value * (1 - value / capacity)
            delta += random.gauss(0, noise * value)
            value = max(0, min(capacity * 1.5, value + delta))
            trajectory.append(round(value, 2))

        self.results[scenario_name] = {
            "parameters": params,
            "trajectory": trajectory,
            "final_value": trajectory[-1],
            "peak": max(trajectory),
            "min": min(trajectory)
        }
        return self.results[scenario_name]

    def compare_scenarios(self):
        """Сравнивает результаты всех сценариев."""
        if not self.results:
            print("  Нет результатов для сравнения!")
            return

        print(f"\n  {'Сценарий':<20} {'Финал':>8} {'Пик':>8} {'Мин':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        for name, result in self.results.items():
            print(f"  {name:<20} {result['final_value']:>8.2f} "
                  f"{result['peak']:>8.2f} {result['min']:>8.2f}")


def demo_simulation_design():
    """Демонстрация дизайна симуляций."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: Дизайн симуляций")
    print("=" * 70)

    # Пример 1: Определение параметров
    print("\n--- Пример 1: Определение параметров ---")
    sim = SimulationDesign("Рост популяции")
    sim.add_parameter("initial_value", 100, 10, 500, "Начальное значение")
    sim.add_parameter("growth_rate", 0.05, 0.01, 0.2, "Скорость роста")
    sim.add_parameter("noise", 0.1, 0.0, 0.5, "Уровень шума")
    sim.add_parameter("capacity", 1000, 100, 5000, "Вместимость среды")

    for name, param in sim.parameters.items():
        print(f"  {name}: {param['default']} "
              f"(диапазон: {param['min']}-{param['max']}) — {param['description']}")

    # Пример 2: Определение сценариев
    print("\n--- Пример 2: Определение сценариев ---")
    sim.add_scenario("оптимистичный", {
        "growth_rate": 0.1, "noise": 0.05, "capacity": 2000
    })
    sim.add_scenario("пессимистичный", {
        "growth_rate": 0.02, "noise": 0.2, "capacity": 500
    })
    sim.add_scenario("нейтральный", {
        "growth_rate": 0.05, "noise": 0.1, "capacity": 1000
    })

    for name, params in sim.scenarios.items():
        print(f"  {name}: {params}")

    # Пример 3: Запуск сценариев
    print("\n--- Пример 3: Запуск сценариев ---")
    for scenario_name in sim.scenarios:
        result = sim.run_scenario(scenario_name, steps=15)
        print(f"\n  {scenario_name}:")
        print(f"    Траектория: {result['trajectory'][:5]}...")
        print(f"    Финальное значение: {result['final_value']:.2f}")

    # Пример 4: Сравнение сценариев
    print("\n--- Пример 4: Сравнение сценариев ---")
    sim.compare_scenarios()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 3: Тестирование multi-agent систем
# ──────────────────────────────────────────────────────────────────────────────

class MultiAgentTester:
    """Тестирование multi-agent систем."""

    def __init__(self):
        self.test_results = []

    def integration_test(self, agents, interaction_fn, expected_states):
        """Интеграционный тест: проверяет взаимодействие агентов."""
        test_name = "integration_test"
        passed = True
        details = []

        # Запускаем взаимодействие
        actual_states = []
        for agent in agents:
            interaction_fn(agent)
            actual_states.append(agent.get("state", "unknown"))

        # Проверяем ожидаемые состояния
        for i, (actual, expected) in enumerate(zip(actual_states, expected_states)):
            if actual != expected:
                passed = False
                details.append(f"  Агент {i}: ожидалось '{expected}', получено '{actual}'")

        result = {
            "test": test_name,
            "passed": passed,
            "agents_tested": len(agents),
            "details": details
        }
        self.test_results.append(result)
        return result

    def scenario_test(self, scenario_fn, scenario_params, assertions):
        """Тест сценария: проверяет сценарий при заданных параметрах."""
        test_name = f"scenario_test_{len(self.test_results)}"
        passed = True
        details = []

        # Запускаем сценарий
        result = scenario_fn(scenario_params)

        # Проверяем утверждения
        for assertion in assertions:
            try:
                check = assertion["check"](result)
                if not check:
                    passed = False
                    details.append(
                        f"  Утверждение не выполнено: {assertion.get('description', 'N/A')}"
                    )
            except Exception as e:
                passed = False
                details.append(f"  Ошибка в утверждении: {e}")

        test_result = {
            "test": test_name,
            "passed": passed,
            "details": details
        }
        self.test_results.append(test_result)
        return test_result

    def stress_test(self, system_fn, load_levels):
        """Стресс-тест: проверяет систему при возрастающей нагрузке."""
        print("\n  Стресс-тест:")
        results = []

        for load in load_levels:
            start_time = time.time()
            try:
                system_fn(load)
                elapsed = time.time() - start_time
                results.append({"load": load, "time": elapsed, "status": "ok"})
                print(f"    Нагрузка {load}: OK ({elapsed:.3f}с)")
            except Exception as e:
                elapsed = time.time() - start_time
                results.append({"load": load, "time": elapsed, "status": f"error: {e}"})
                print(f"    Нагрузка {load}: ОШИБКА ({e})")

        return results

    def summary(self):
        """Выводит сводку по тестам."""
        total = len(self.test_results)
        passed = sum(1 for t in self.test_results if t["passed"])
        failed = total - passed

        print(f"\n  Сводка тестов: {passed}/{total} пройдено, {failed} провалено")
        for t in self.test_results:
            status = "OK" if t["passed"] else "FAIL"
            print(f"    [{status}] {t['test']}")


def demo_testing_multi_agent():
    """Демонстрация тестирования multi-agent систем."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: Тестирование multi-agent систем")
    print("=" * 70)

    tester = MultiAgentTester()

    # Пример 1: Интеграционный тест
    print("\n--- Пример 1: Интеграционный тест ---")
    agents = [
        {"id": 0, "state": "idle"},
        {"id": 1, "state": "idle"},
        {"id": 2, "state": "idle"}
    ]

    def activate_agent(agent):
        agent["state"] = "active"

    expected = ["active", "active", "active"]
    result = tester.integration_test(agents, activate_agent, expected)
    print(f"  Результат: {'ПРОЙДЕН' if result['passed'] else 'ПРОВАЛЕН'}")
    print(f"  Агентов протестировано: {result['agents_tested']}")

    # Пример 2: Тест сценария
    print("\n--- Пример 2: Тест сценария ---")
    def epidemic_scenario(params):
        """Простая модель эпидемии."""
        population = params["population"]
        initial_infected = params["initial_infected"]
        transmission_rate = params["transmission_rate"]
        steps = params["steps"]

        infected = initial_infected
        history = [infected]
        for _ in range(steps):
            new_infected = infected * transmission_rate
            infected = min(population, infected + new_infected)
            history.append(round(infected, 1))

        return {
            "population": population,
            "final_infected": history[-1],
            "peak_infected": max(history),
            "history": history
        }

    assertions = [
        {
            "check": lambda r: r["final_infected"] <= r["population"],
            "description": "Заражённые не превышают популяцию"
        },
        {
            "check": lambda r: r["final_infected"] > 0,
            "description": "Заражение произошло"
        }
    ]

    result = tester.scenario_test(epidemic_scenario, {
        "population": 1000,
        "initial_infected": 10,
        "transmission_rate": 0.3,
        "steps": 20
    }, assertions)
    print(f"  Результат: {'ПРОЙДЕН' if result['passed'] else 'ПРОВАЛЕН'}")
    for d in result["details"]:
        print(f"    {d}")

    # Пример 3: Стресс-тест
    print("\n--- Пример 3: Стресс-тест ---")
    def process_agents(num_agents):
        """Обработка N агентов."""
        agents = [{"id": i, "state": "idle"} for i in range(num_agents)]
        for agent in agents:
            agent["state"] = "active"
            # Простая обработка
            _ = sum(j**2 for j in range(min(100, num_agents)))
        return agents

    tester.stress_test(process_agents, [10, 50, 100, 500])

    # Пример 4: Сводка
    print("\n--- Пример 4: Сводка тестов ---")
    tester.summary()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Раздел 4: Анализ и валидация
# ──────────────────────────────────────────────────────────────────────────────

class AnalysisValidator:
    """Статистический анализ и валидация симуляций."""

    @staticmethod
    def mean(values):
        """Среднее значение."""
        return sum(values) / len(values) if values else 0

    @staticmethod
    def variance(values):
        """Дисперсия."""
        m = AnalysisValidator.mean(values)
        return sum((x - m) ** 2 for x in values) / len(values) if values else 0

    @staticmethod
    def std_dev(values):
        """Стандартное отклонение."""
        return math.sqrt(AnalysisValidator.variance(values))

    @staticmethod
    def confidence_interval(values, confidence=0.95):
        """Доверительный интервал (z-интервал)."""
        n = len(values)
        if n < 2:
            return (0, 0)
        m = AnalysisValidator.mean(values)
        se = AnalysisValidator.std_dev(values) / math.sqrt(n)
        # Z-значение для 95% = 1.96
        z = 1.96 if confidence == 0.95 else 1.645
        return (round(m - z * se, 4), round(m + z * se, 4))

    @staticmethod
    def sensitivity_analysis(base_params, param_name, range_vals, sim_fn):
        """Анализ чувствительности: как изменение параметра влияет на результат."""
        print(f"\n  Анализ чувствительности для '{param_name}':")
        results = []
        for val in range_vals:
            params = base_params.copy()
            params[param_name] = val
            result = sim_fn(params)
            results.append({"param": val, "result": result})
            print(f"    {param_name}={val}: результат={result:.4f}")

        # Вычисляем чувствительность
        if len(results) >= 2:
            r_vals = [r["result"] for r in results]
            p_vals = [r["param"] for r in results]
            sensitivity = (r_vals[-1] - r_vals[0]) / (p_vals[-1] - p_vals[0])
            print(f"  Чувствительность: {sensitivity:.4f}")
            return results, sensitivity
        return results, 0

    @staticmethod
    def calibration(measured, simulated, metric="mae"):
        """Калибровка модели: сравнение измеренных и смоделированных данных."""
        if len(measured) != len(simulated):
            print("  Ошибка: измеренные и смоделированные данные разной длины!")
            return None

        errors = [m - s for m, s in zip(measured, simulated)]
        abs_errors = [abs(e) for e in errors]
        sq_errors = [e**2 for e in errors]

        mae = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
        mape = sum(abs(e / m) for e, m in zip(errors, measured) if m != 0) / len(measured) * 100

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "max_error": round(max(abs_errors), 4),
            "correlation": AnalysisValidator._correlation(measured, simulated)
        }

    @staticmethod
    def _correlation(x, y):
        """Корреляция Пирсона."""
        n = len(x)
        if n < 2:
            return 0
        mx, my = sum(x)/n, sum(y)/n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mx)**2 for xi in x))
        den_y = math.sqrt(sum((yi - my)**2 for yi in y))
        if den_x * den_y == 0:
            return 0
        return round(num / (den_x * den_y), 4)


def demo_analysis_validation():
    """Демонстрация анализа и валидации."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Анализ и валидация симуляций")
    print("=" * 70)

    av = AnalysisValidator()

    # Пример 1: Статистический анализ
    print("\n--- Пример 1: Статистический анализ ---")
    # Генерируем данные симуляции (нормальное распределение через Бокса-Мюллера)
    random.seed(42)
    data = []
    for _ in range(100):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(max(u1, 0.001))) * math.cos(2 * math.pi * u2)
        data.append(50 + 10 * z)  # N(50, 10)

    print(f"  Размер выборки: {len(data)}")
    print(f"  Среднее: {av.mean(data):.2f}")
    print(f"  Дисперсия: {av.variance(data):.2f}")
    print(f"  Стандартное отклонение: {av.std_dev(data):.2f}")
    ci = av.confidence_interval(data)
    print(f"  95% доверительный интервал: ({ci[0]:.2f}, {ci[1]:.2f})")

    # Пример 2: Анализ чувствительности
    print("\n--- Пример 2: Анализ чувствительности ---")
    def simple_growth(params):
        """Простая модель роста."""
        return params["rate"] * params["initial"] * params["time"]

    base = {"initial": 100, "time": 10, "rate": 0.1}
    rate_range = [0.01, 0.05, 0.1, 0.15, 0.2]
    av.sensitivity_analysis(base, "rate", rate_range, simple_growth)

    # Пример 3: Калибровка модели
    print("\n--- Пример 3: Калидация модели ---")
    # Реальные данные (измеренные)
    measured = [100, 120, 145, 175, 210, 250, 295, 340]
    # Смоделированные данные
    simulated = [100, 118, 140, 168, 200, 238, 282, 330]

    calibration = av.calibration(measured, simulated)
    print(f"  MAE (средняя абсолютная ошибка): {calibration['mae']}")
    print(f"  RMSE (корень из среднеквадратичной ошибки): {calibration['rmse']}")
    print(f"  MAPE (процентная ошибка): {calibration['mape']}%")
    print(f"  Максимальная ошибка: {calibration['max_error']}")
    print(f"  Корреляция Пирсона: {calibration['correlation']}")

    # Пример 4: Визуализация расхождения
    print("\n--- Пример 4: Расхождение модели и данных ---")
    for i, (m, s) in enumerate(zip(measured, simulated)):
        diff = m - s
        bar = "+" * abs(diff) if diff > 0 else "-" * abs(diff)
        print(f"  Точка {i+1}: измер={m:4d}, модель={s:4d}, "
              f"разница={diff:+4d} [{bar}]")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_agent_based_modeling()
    demo_simulation_design()
    demo_testing_multi_agent()
    demo_analysis_validation()
