"""156 — Agent Architecture: цикл агента, восприятие-действие, управление состоянием

Темы:
  1. Agent Loop (Observe → Think → Act → Observe cycle, main loop implementation)
  2. Agent State (state machine, transitions, persistence)
  3. Agent Configuration (system prompt, tools, constraints, temperature)
  4. Agent Lifecycle (initialization, execution, termination, error handling)

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

# ============================================================
# 1. Agent Loop — цикл агента восприятия-действия
# ============================================================

def demo_agent_loop():
    print("=" * 70)
    print("DEMO 1: Agent Loop — цикл агента Observe → Think → Act")
    print("=" * 70)

    # --- 1.1 Простой цикл агента ---
    print("\n--- 1.1 Простой цикл агента (3 итерации) ---")

    class SimpleAgent:
        """Простейший агент с циклом Observe-Think-Act."""

        def __init__(self):
            self.step = 0
            self.memory = []  # история наблюдений

        def observe(self, environment):
            """Наблюдение — сбор информации из среды."""
            observation = {
                "step": self.step,
                "state": environment.get("state", "unknown"),
                "value": environment.get("value", 0),
            }
            self.memory.append(observation)
            return observation

        def think(self, observation):
            """Мысль — принятие решения на основе наблюдения."""
            # Простая логика: если значение > 5, увеличить, иначе уменьшить
            if observation["value"] > 5:
                action = "increase"
                reasoning = f"Значение {observation['value']} > 5, нужно увеличить"
            else:
                action = "decrease"
                reasoning = f"Значение {observation['value']} <= 5, нужно уменьшить"
            return {"action": action, "reasoning": reasoning}

        def act(self, decision, environment):
            """Действие — выполнение решения."""
            if decision["action"] == "increase":
                environment["value"] += 3
            else:
                environment["value"] -= 1
            environment["state"] = "modified"
            return environment

    # Запуск цикла агента
    env = {"state": "initial", "value": 3}
    agent = SimpleAgent()

    for i in range(3):
        obs = agent.observe(env)
        decision = agent.think(obs)
        env = agent.act(decision, env)
        print(f"  Итерация {i+1}: наблюдение={obs}, решение={decision['action']}, "
              f"новое значение={env['value']}")

    # --- 1.2 Цикл с ограничением итераций ---
    print("\n--- 1.2 Цикл с ограничением итераций (max_steps) ---")

    def agent_loop_with_limit(agent_fn, env, max_steps=5):
        """Цикл агента с ограничением на количество шагов."""
        history = []
        for step in range(max_steps):
            observation = {"step": step, "env_state": env.copy()}
            decision = agent_fn(observation)
            env = decision["apply"](env)
            history.append({
                "step": step,
                "observation": observation,
                "decision": decision["name"],
            })
            # Условие остановки
            if decision.get("stop", False):
                print(f"  Агент остановился на шаге {step+1}: {decision['name']}")
                break
        return history

    def simple_decider(obs):
        """Простая функция принятия решений."""
        val = obs["env_state"].get("counter", 0)
        if val >= 3:
            return {"name": "stop", "stop": True,
                    "apply": lambda e: e}
        return {
            "name": "increment",
            "apply": lambda e: {**e, "counter": e.get("counter", 0) + 1},
        }

    history = agent_loop_with_limit(simple_decider, {"counter": 0}, max_steps=10)
    for h in history:
        print(f"  Шаг {h['step']+1}: действие={h['decision']}")

    # --- 1.3 Агент с памятью ---
    print("\n--- 1.3 Агент с короткой и долгосрочной памятью ---")

    class AgentWithMemory:
        """Агент с разделением памяти на рабочую и долгосрочную."""

        def __init__(self, capacity_short=3):
            self.short_memory = collections.deque(maxlen=capacity_short)
            self.long_memory = []

        def store(self, observation):
            """Сохранение наблюдения в память."""
            self.short_memory.append(observation)
            # Если накопилось достаточно — переносим в долгосрочную
            if len(self.short_memory) == self.short_memory.maxlen:
                summary = sum(self.short_memory) / len(self.short_memory)
                self.long_memory.append(round(summary, 2))
                self.short_memory.clear()

        def recall(self):
            """Получение информации из памяти."""
            return {
                "short": list(self.short_memory),
                "long": self.long_memory.copy(),
            }

    agent_mem = AgentWithMemory(capacity_short=3)
    values = [10, 20, 30, 40, 50, 60, 70]
    for v in values:
        agent_mem.store(v)
        mem = agent_mem.recall()
        print(f"  Значение {v}: краткоср={mem['short']}, долгоср={mem['long']}")

    # --- 1.4 Цикл с обратной связью ---
    print("\n--- 1.4 Цикл с обратной связью (reward signal) ---")

    def agent_loop_with_reward(env, steps=6):
        """Цикл агента с наградой за каждый шаг."""
        total_reward = 0
        trajectory = []
        for step in range(steps):
            # Наблюдение
            state = env["position"]
            # Решение: выбрать направление
            action = random.choice(["left", "right"])
            # Действие
            if action == "right":
                env["position"] += 1
            else:
                env["position"] -= 1
            # Награда: чем ближе к цели, тем лучше
            reward = -abs(env["target"] - env["position"])
            total_reward += reward
            trajectory.append({
                "step": step + 1, "action": action,
                "pos": env["position"], "reward": reward,
            })
        return trajectory, total_reward

    env_reward = {"position": 0, "target": 5}
    traj, total_r = agent_loop_with_reward(env_reward)
    for t in traj:
        print(f"  Шаг {t['step']}: действие={t['action']}, "
              f"позиция={t['pos']}, награда={t['reward']}")
    print(f"  Итоговая награда: {total_r}")

    print()


# ============================================================
# 2. Agent State — конечный автомат агента
# ============================================================

def demo_agent_state():
    print("=" * 70)
    print("DEMO 2: Agent State — конечный автомат и переходы")
    print("=" * 70)

    # --- 2.1 Конечный автомат агента ---
    print("\n--- 2.1 Конечный автомат агента ---")

    class AgentStateMachine:
        """Конечный автомат для управления состоянием агента."""

        # Допустимые переходы: (текущее_состояние) → {следующее_состояние}
        TRANSITIONS = {
            "idle": {"start": "observing"},
            "observing": {"data_ready": "thinking", "timeout": "idle"},
            "thinking": {"plan_ready": "acting", "uncertain": "observing"},
            "acting": {"done": "idle", "error": "thinking", "retry": "observing"},
        }

        def __init__(self):
            self.state = "idle"
            self.history = []

        def transition(self, event):
            """Переход в новое состояние по событию."""
            self.history.append({"from": self.state, "event": event})
            if event in self.TRANSITIONS.get(self.state, {}):
                self.state = self.TRANSITIONS[self.state][event]
                return True
            return False  # Недопустимый переход

    sm = AgentStateMachine()
    events = ["start", "data_ready", "plan_ready", "done",
              "start", "data_ready", "uncertain", "data_ready",
              "plan_ready", "error", "retry", "data_ready",
              "plan_ready", "done"]

    for event in events:
        old_state = sm.state
        success = sm.transition(event)
        status = "OK" if success else "ОШИБКА перехода"
        print(f"  Событие '{event}': {old_state} → {sm.state} [{status}]")

    # --- 2.2 Персистентное состояние ---
    print("\n--- 2.2 Сериализация и восстановление состояния ---")

    def serialize_state(agent_state):
        """Сериализация состояния агента в JSON."""
        return json.dumps(agent_state, ensure_ascii=False, indent=2)

    def deserialize_state(data):
        """Восстановление состояния из JSON."""
        return json.loads(data)

    state = {
        "agent_id": "agent-001",
        "current_step": 5,
        "memory": [10, 20, 30],
        "config": {"temperature": 0.7, "max_tokens": 1000},
        "history": ["step1", "step2", "step3"],
    }

    serialized = serialize_state(state)
    print(f"  Сериализованное состояние ({len(serialized)} байт):")
    for line in serialized.split("\n")[:8]:
        print(f"    {line}")

    restored = deserialize_state(serialized)
    print(f"  Восстановлено: agent_id={restored['agent_id']}, "
          f"шаг={restored['current_step']}, память={restored['memory']}")

    # --- 2.3 Проверка целостности через хеш ---
    print("\n--- 2.3 Проверка целостности состояния (хеш) ---")

    def state_hash(state_dict):
        """Вычисление хеша состояния для проверки целостности."""
        state_str = json.dumps(state_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    original_hash = state_hash(state)
    print(f"  Хеш оригинала: {original_hash}")

    # Модифицируем состояние
    modified_state = state.copy()
    modified_state["current_step"] = 6
    modified_hash = state_hash(modified_state)
    print(f"  Хеш после изменения: {modified_hash}")
    print(f"  Хеши совпадают: {original_hash == modified_hash}")

    # Восстанавливаем и проверяем
    restored_state = json.loads(serialized)
    restored_hash = state_hash(restored_state)
    print(f"  Хеш восстановленного: {restored_hash}")
    print(f"  Хеш совпадает с оригиналом: {original_hash == restored_hash}")

    # --- 2.4 Журнал переходов ---
    print("\n--- 2.4 Журнал переходов (audit log) ---")

    class StateAuditLog:
        """Журнал для отслеживания всех изменений состояния."""

        def __init__(self):
            self.entries = []

        def log(self, old_state, new_state, action, reason):
            """Запись перехода в журнал."""
            self.entries.append({
                "timestamp": len(self.entries),
                "old": old_state,
                "new": new_state,
                "action": action,
                "reason": reason,
                "hash": state_hash({"old": old_state, "new": new_state}),
            })

        def get_transitions(self, state_name):
            """Получение всех переходов из указанного состояния."""
            return [e for e in self.entries if e["old"] == state_name]

    audit = StateAuditLog()
    transitions_log = [
        ("idle", "observing", "start", "Запуск агента"),
        ("observing", "thinking", "data_ready", "Данные получены"),
        ("thinking", "acting", "plan_ready", "План составлен"),
        ("acting", "idle", "done", "Задача выполнена"),
        ("idle", "observing", "start", "Новая задача"),
        ("observing", "thinking", "data_ready", "Данные получены"),
    ]

    for old, new, action, reason in transitions_log:
        audit.log(old, new, action, reason)

    print("  Журнал переходов:")
    for entry in audit.entries:
        print(f"    [{entry['timestamp']}] {entry['old']} → {entry['new']} "
              f"({entry['action']}): {entry['reason']}")

    observing_transitions = audit.get_transitions("observing")
    print(f"\n  Переходы из состояния 'observing': {len(observing_transitions)}")
    for t in observing_transitions:
        print(f"    → {t['new']} ({t['reason']})")

    print()


# ============================================================
# 3. Agent Configuration — конфигурация агента
# ============================================================

def demo_agent_configuration():
    print("=" * 70)
    print("DEMO 3: Agent Configuration — системный промпт, инструменты, ограничения")
    print("=" * 70)

    # --- 3.1 Системный промпт ---
    print("\n--- 3.1 Системный промпт и его влияние ---")

    class AgentConfig:
        """Конфигурация агента с системным промптом и параметрами."""

        def __init__(self, system_prompt, temperature=0.7,
                     max_tokens=1000, tools=None):
            self.system_prompt = system_prompt
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.tools = tools or []

        def build_context(self, user_message):
            """Формирование контекста для LLM."""
            return {
                "system": self.system_prompt,
                "user": user_message,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "available_tools": [t["name"] for t in self.tools],
            }

        def validate(self):
            """Валидация конфигурации."""
            errors = []
            if not self.system_prompt:
                errors.append("Системный промпт не задан")
            if not (0.0 <= self.temperature <= 2.0):
                errors.append(f"Temperature {self.temperature} вне диапазона [0, 2]")
            if self.max_tokens <= 0:
                errors.append(f"max_tokens должен быть > 0, получено {self.max_tokens}")
            return errors

    # Конфигурация для разных задач
    configs = {
        "coder": AgentConfig(
            system_prompt="Ты — опытный программист. Пиши чистый код.",
            temperature=0.2,
            max_tokens=2000,
        ),
        "creative": AgentConfig(
            system_prompt="Ты — креативный писатель. Создавай яркие образы.",
            temperature=0.9,
            max_tokens=1500,
        ),
        "analyst": AgentConfig(
            system_prompt="Ты — аналитик. Давай точные, структурированные ответы.",
            temperature=0.3,
            max_tokens=1000,
        ),
    }

    for name, cfg in configs.items():
        ctx = cfg.build_context("Привет!")
        errors = cfg.validate()
        status = "валидна" if not errors else f"ошибки: {errors}"
        print(f"  {name}: temp={cfg.temperature}, tokens={cfg.max_tokens}, "
              f"инструменты={ctx['available_tools']}, статус={status}")

    # --- 3.2 Реестр инструментов ---
    print("\n--- 3.2 Реестр инструментов агента ---")

    def make_tool(name, description, parameters):
        """Создание описания инструмента."""
        return {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    tools = [
        make_tool("search", "Поиск информации в базе знаний",
                  {"query": {"type": "string", "required": True},
                   "limit": {"type": "integer", "default": 5}}),
        make_tool("calculate", "Вычисление математического выражения",
                  {"expression": {"type": "string", "required": True}}),
        make_tool("write_file", "Запись содержимого в файл",
                  {"path": {"type": "string", "required": True},
                   "content": {"type": "string", "required": True}}),
    ]

    print(f"  Зарегистрировано инструментов: {len(tools)}")
    for tool in tools:
        params = ", ".join(f"{k}({v['type']})" for k, v in tool["parameters"].items())
        print(f"    - {tool['name']}: {tool['description']} [{params}]")

    # --- 3.3 Ограничения агента ---
    print("\n--- 3.3 Ограничения (constraints) и guardrails ---")

    class AgentConstraints:
        """Ограничения на поведение агента."""

        def __init__(self):
            self.max_tool_calls = 10
            self.max_retries = 3
            self.forbidden_actions = ["delete_all", "execute_code"]
            self.required_fields = ["action", "reasoning"]

        def check(self, action):
            """Проверка действия на соответствие ограничениям."""
            violations = []

            # Проверка запрещённых действий
            if action.get("type") in self.forbidden_actions:
                violations.append(f"Действие '{action['type']}' запрещено")

            # Проверка обязательных полей
            for field in self.required_fields:
                if field not in action:
                    violations.append(f"Отсутствует обязательное поле '{field}'")

            # Проверка длины рассуждения
            reasoning = action.get("reasoning", "")
            if len(reasoning) > 500:
                violations.append(f"Рассуждение слишком длинное ({len(reasoning)} > 500)")

            return violations

    constraints = AgentConstraints()
    test_actions = [
        {"type": "search", "reasoning": "Ищу информацию"},
        {"type": "delete_all", "reasoning": "Удаляю данные"},
        {"type": "calculate", "reasoning": "Вычисляю"},
        {"type": "write", "reasoning": "x" * 501},
    ]

    for action in test_actions:
        violations = constraints.check(action)
        status = "OK" if not violations else f"нарушения: {violations}"
        print(f"  Действие {action.get('type', '?')}: {status}")

    # --- 3.4 Температура и сэмплирование ---
    print("\n--- 3.4 Температура и сэмплирование (top-k) ---")

    def sample_with_temperature(logits, temperature, top_k=None):
        """Сэмплирование с температурой и top-k фильтрацией."""
        # Нормализация логитов
        max_logit = max(logits)
        scaled = [(logit - max_logit) / max(temperature, 0.01) for logit in logits]
        # Преобразование в вероятности (softmax)
        exp_vals = [math.exp(s) for s in scaled]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]

        # Top-k фильтрация
        if top_k is not None:
            indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:top_k]
            total = sum(p for _, p in indexed)
            probs = [p / total if i in [idx for idx, _ in indexed] else 0
                     for i, p in enumerate(probs)]

        # Сэмплирование
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i, probs
        return len(probs) - 1, probs

    # Пример: 5 токенов с разными температурами
    logits = [2.0, 1.0, 0.5, 3.0, 0.1]
    tokens = ["кот", "пёс", "рыба", "птица", "змея"]

    print(f"  Логиты: {logits}")
    print(f"  Токены: {tokens}")

    for temp in [0.1, 0.7, 1.5]:
        print(f"\n  Температура = {temp}:")
        counts = [0] * len(tokens)
        for _ in range(100):
            idx, _ = sample_with_temperature(logits, temp)
            counts[idx] += 1
        for i, (tok, cnt) in enumerate(zip(tokens, counts)):
            bar = "█" * (cnt // 2)
            print(f"    {tok}: {cnt:3d}/100 {bar}")

    print()


# ============================================================
# 4. Agent Lifecycle — жизненный цикл агента
# ============================================================

def demo_agent_lifecycle():
    print("=" * 70)
    print("DEMO 4: Agent Lifecycle — инициализация, выполнение, завершение")
    print("=" * 70)

    # --- 4.1 Инициализация агента ---
    print("\n--- 4.1 Инициализация агента (setup phase) ---")

    class Agent:
        """Агент с полным жизненным циклом."""

        def __init__(self, name, config=None):
            self.name = name
            self.config = config or {}
            self.state = "uninitialized"
            self.initialized_at = None
            self.errors = []

        def initialize(self):
            """Инициализация агента — настройка ресурсов."""
            self.state = "initializing"
            print(f"  [{self.name}] Инициализация...")

            # Проверка конфигурации
            if "model" not in self.config:
                self.errors.append("Не задана модель в конфигурации")
                self.state = "error"
                return False

            # Имитация загрузки ресурсов
            self.initialized_at = time.time()
            self.state = "ready"
            print(f"  [{self.name}] Готов! Модель: {self.config['model']}")
            return True

        def execute(self, task):
            """Выполнение задачи."""
            if self.state != "ready":
                print(f"  [{self.name}] Невозможно выполнить: состояние={self.state}")
                return None

            self.state = "running"
            print(f"  [{self.name}] Выполняю задачу: {task}")

            # Имитация обработки
            result = f"Результат для '{task}'"
            self.state = "ready"
            return result

        def shutdown(self):
            """Корректное завершение работы агента."""
            self.state = "shutting_down"
            print(f"  [{self.name}] Завершение работы...")
            self.state = "terminated"
            print(f"  [{self.name}] Агент завершён.")

    # Демонстрация жизненного цикла
    agent = Agent("assistant", {"model": "mimo-v2", "temperature": 0.7})
    print(f"  Начальное состояние: {agent.state}")

    agent.initialize()
    print(f"  После инициализации: {agent.state}")

    result = agent.execute("Написать функцию сортировки")
    print(f"  Результат: {result}")

    agent.shutdown()
    print(f"  После завершения: {agent.state}")

    # --- 4.2 Обработка ошибок ---
    print("\n--- 4.2 Обработка ошибок и восстановление ---")

    class ResilientAgent:
        """Агент с обработкой ошибок и автоматическим восстановлением."""

        def __init__(self, name, max_retries=3):
            self.name = name
            self.max_retries = max_retries
            self.state = "idle"
            self.error_log = []

        def safe_execute(self, task_fn, *args):
            """Безопасное выполнение с повторными попытками."""
            for attempt in range(self.max_retries):
                try:
                    self.state = "running"
                    result = task_fn(*args)
                    self.state = "idle"
                    return {"success": True, "result": result, "attempts": attempt + 1}
                except Exception as e:
                    self.error_log.append({
                        "attempt": attempt + 1,
                        "error": str(e),
                    })
                    print(f"  [{self.name}] Попытка {attempt+1} не удалась: {e}")
                    if attempt < self.max_retries - 1:
                        self.state = "retrying"
                        wait_time = 2 ** attempt  # Экспоненциальная задержка
                        print(f"  [{self.name}] Повтор через {wait_time}с...")

            self.state = "failed"
            return {"success": False, "error": "Все попытки исчерпаны",
                    "attempts": self.max_retries}

    def unstable_task(x):
        """Задача, которая иногда завершается ошибкой."""
        r = random.random()
        if r < 0.6:  # 60% вероятность ошибки
            raise ValueError(f"Случайная ошибка (p={r:.2f})")
        return x * 2

    resilient = ResilientAgent("resilient", max_retries=3)
    result = resilient.safe_execute(unstable_task, 5)
    print(f"  Результат: {result}")

    # --- 4.3 Метрики жизненного цикла ---
    print("\n--- 4.3 Метрики жизненного цикла ---")

    class AgentMetrics:
        """Сбор метрик жизненного цикла агента."""

        def __init__(self):
            self.start_time = None
            self.task_count = 0
            self.total_time = 0.0
            self.error_count = 0

        def start(self):
            """Начало отсчёта."""
            self.start_time = time.time()

        def record_task(self, duration, success):
            """Запись метрики задачи."""
            self.task_count += 1
            self.total_time += duration
            if not success:
                self.error_count += 1

        def summary(self):
            """Итоговая сводка."""
            uptime = time.time() - self.start_time if self.start_time else 0
            avg_time = self.total_time / max(self.task_count, 1)
            success_rate = ((self.task_count - self.error_count) /
                          max(self.task_count, 1) * 100)
            return {
                "uptime": round(uptime, 3),
                "tasks": self.task_count,
                "avg_time": round(avg_time, 3),
                "errors": self.error_count,
                "success_rate": round(success_rate, 1),
            }

    metrics = AgentMetrics()
    metrics.start()

    # Имитация выполнения задач
    task_durations = [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.12]
    task_successes = [True, True, False, True, True, True, False]

    for dur, success in zip(task_durations, task_successes):
        metrics.record_task(dur, success)

    summary = metrics.summary()
    print(f"  Аптайм: {summary['uptime']}с")
    print(f"  Всего задач: {summary['tasks']}")
    print(f"  Среднее время: {summary['avg_time']}с")
    print(f"  Ошибки: {summary['errors']}")
    print(f"  Успешность: {summary['success_rate']}%")

    # --- 4.4 Graceful Shutdown ---
    print("\n--- 4.4 Graceful Shutdown (корректное завершение) ---")

    class ManagedAgent:
        """Агент с управляемым завершением."""

        def __init__(self, name):
            self.name = name
            self.state = "running"
            self.cleanup_hooks = []

        def register_cleanup(self, hook_fn, description):
            """Регистрация функции очистки."""
            self.cleanup_hooks.append({"fn": hook_fn, "desc": description})

        def shutdown(self):
            """Корректное завершение с очисткой ресурсов."""
            print(f"  [{self.name}] Начало завершения...")
            self.state = "stopping"

            # Выполнение хуков очистки в обратном порядке
            for hook in reversed(self.cleanup_hooks):
                try:
                    hook["fn"]()
                    print(f"    Очищено: {hook['desc']}")
                except Exception as e:
                    print(f"    Ошибка при очистке '{hook['desc']}': {e}")

            self.state = "stopped"
            print(f"  [{self.name}] Завершён. Состояние: {self.state}")

    managed = ManagedAgent("managed-agent")
    managed.register_cleanup(lambda: print("    → Соединения закрыты"),
                           "сетевые соединения")
    managed.register_cleanup(lambda: print("    → Кэш очищен"),
                           "кэш в памяти")
    managed.register_cleanup(lambda: print("    → Файлы удалены"),
                           "временные файлы")
    managed.register_cleanup(lambda: print("    → Логи сохранены"),
                           "лог-файлы")

    managed.shutdown()

    print()


# ============================================================
# Запуск всех демонстраций
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  УРОК 156: Agent Architecture")
    print("  Цикл агента, восприятие-действие, управление состоянием")
    print("=" * 70 + "\n")

    demo_agent_loop()
    demo_agent_state()
    demo_agent_configuration()
    demo_agent_lifecycle()

    print("=" * 70)
    print("  Все демонстрации завершены!")
    print("=" * 70)
