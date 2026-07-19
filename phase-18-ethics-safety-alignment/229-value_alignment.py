"""229 — Value Alignment: CEV, корригируемость, инструментальная конвергенция

Темы:
  1. Coherent Extrapolated Volition (CEV, экстраполяция человеческих ценностей)
  2. Corrigibility (выключение, прерывание, обучение ценностям)
  3. Instrumental Convergence (конвергентные инструментальные цели, стремление к власти)
  4. Alignment Tax (стоимость выравнивания, компромисс между способностью и безопасностью)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# =============================================================================
# 1. COHERENT EXTRAPOLATED VOLITION (CEV)
# =============================================================================

class ValueExtrapolator:
    """Модель экстраполяции ценностей (CEV).

    Идея: если бы люди знали больше, думали быстрее и были более
    едины, каких решений они бы хотели? CEV пытается «экстраполировать»
    волю людей, устраняя когнитивные искажения и незнание.
    """

    def __init__(self, values=None, weights=None):
        # Исходные ценности (множество измерений)
        self.values = values or {
            "безопасность": 0.9,
            "автономия": 0.7,
            "справедливость": 0.85,
            "благополучие": 0.8,
            "знание": 0.6,
            "творчество": 0.5,
        }
        # Веса влияния на итоговую оценку
        self.weights = weights or {
            "безопасность": 1.0,
            "автономия": 0.8,
            "справедливость": 0.9,
            "благополучие": 0.95,
            "знание": 0.7,
            "творчество": 0.6,
        }

    def extrapolate(self, knowledge_level=1.0, rationality=1.0, coherence=1.0):
        """Экстраполяция ценностей с учётом трёх параметров CEV.

        knowledge_level: насколько больше людей знают (1.0 = базовый)
        rationality: насколько быстрее и точнее думают (1.0 = базовый)
        coherence: насколько единодушны (1.0 = полное единодушие)
        """
        extrapolated = {}
        for value, base in self.values.items():
            # Чем больше знают и чем рациональнее, тем сильнее
            # отклонение от «по умолчанию» в сторону более обоснованных
            adjustment = (knowledge_level - 1.0) * 0.1 + (rationality - 1.0) * 0.05
            # Коherence усиливает согласованность ценностей
            coherence_factor = 1.0 + (coherence - 1.0) * 0.2
            extrapolated_value = base + adjustment * coherence_factor
            extrapolated_value = max(0.0, min(1.0, extrapolated_value))
            extrapolated[value] = round(extrapolated_value, 3)
        return extrapolated

    def compute_decision(self, options, extrapolated_values):
        """Принятие решения на основе экстраполированных ценностей.

        Каждая опция — словарь {ценность: насколько её удовлетворяет}.
        Возвращает лучший вариант.
        """
        scores = {}
        for option_name, option_values in options.items():
            score = 0.0
            for value, importance in extrapolated_values.items():
                weight = self.weights.get(value, 0.5)
                option_score = option_values.get(value, 0.0)
                score += importance * weight * option_score
            scores[option_name] = round(score, 3)
        best_option = max(scores, key=scores.get)
        return best_option, scores

    def compare_with_naive(self, knowledge_level=1.0, rationality=1.0):
        """Сравнение экстраполированных ценностей с naïve-мнением."""
        naive_values = self.values.copy()  # «Наивное» = исходное
        extrapolated = self.extrapolate(knowledge_level, rationality)
        comparison = {}
        for value in self.values:
            diff = extrapolated[value] - naive_values[value]
            comparison[value] = {
                "naive": naive_values[value],
                "extrapolated": extrapolated[value],
                "delta": round(diff, 3),
                "direction": "↑" if diff > 0 else "↓" if diff < 0 else "=",
            }
        return comparison


def demo_cev():
    """Демонстрация Coherent Extrapolated Volition."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: COHERENT EXTRAPOLATED VOLITION (CEV)")
    print("Экстраполяция человеческих ценностей")
    print("=" * 70)

    extrap = ValueExtrapolator()

    # --- 1.1 Исходные ценности ---
    print("\n--- 1.1 Исходные ценности ---")
    for value, score in extrap.values.items():
        weight = extrap.weights[value]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {value:15s} |{bar}| {score:.2f} (вес: {weight:.2f})")

    # --- 1.2 Экстраполяция при разных условиях ---
    print("\n--- 1.2 Экстраполяция при разных условиях ---")
    conditions = [
        ("Базовый (все = 1.0)", 1.0, 1.0, 1.0),
        ("Больше знаний (k=2.0)", 2.0, 1.0, 1.0),
        ("Более рациональны (r=2.0)", 1.0, 2.0, 1.0),
        ("Более едины (c=2.0)", 1.0, 1.0, 2.0),
        ("Все параметры высокие", 3.0, 2.5, 2.0),
    ]

    for cond_name, k, r, c in conditions:
        result = extrap.extrapolate(k, r, c)
        print(f"\n  {cond_name}:")
        for value, score in result.items():
            base = extrap.values[value]
            diff = score - base
            indicator = "+" if diff > 0 else "-" if diff < 0 else "="
            print(f"    {value:15s}: {score:.3f} ({indicator}{abs(diff):.3f} от базы)")

    # --- 1.3 Принятие решений ---
    print("\n\n--- 1.3 Принятие решений на основе CEV ---")
    options = {
        "Вариант A: безопасный": {
            "безопасность": 1.0, "автономия": 0.3,
            "справедливость": 0.5, "благополучие": 0.7,
            "знание": 0.4, "творчество": 0.3,
        },
        "Вариант B: сбалансированный": {
            "безопасность": 0.7, "автономия": 0.7,
            "справедливость": 0.7, "благополучие": 0.7,
            "знание": 0.6, "творчество": 0.6,
        },
        "Вариант C: свободный": {
            "безопасность": 0.4, "автономия": 1.0,
            "справедливость": 0.6, "благополучие": 0.5,
            "знание": 0.8, "творчество": 0.9,
        },
    }

    # Наивное решение (без экстраполяции)
    naive_values = extrap.values
    naive_best, naive_scores = extrap.compute_decision(options, naive_values)
    print(f"  Наивное решение: {naive_best}")
    for name, score in sorted(naive_scores.items(), key=lambda x: -x[1]):
        print(f"    {name}: {score}")

    # Решение с экстраполяцией
    ext_values = extrap.extrapolate(knowledge_level=2.0, rationality=1.5)
    ext_best, ext_scores = extrap.compute_decision(options, ext_values)
    print(f"\n  Экстраполированное решение: {ext_best}")
    for name, score in sorted(ext_scores.items(), key=lambda x: -x[1]):
        print(f"    {name}: {score}")

    # --- 1.4 Сравнение naïve vs экстраполированное ---
    print("\n--- 1.4 Сравнение naïve и экстраполированного ---")
    comparison = extrap.compare_with_naive(knowledge_level=2.0, rationality=1.5)
    for value, data in comparison.items():
        print(f"  {value:15s}: naïve={data['naive']:.3f} → "
              f"экстраполированное={data['extrapolated']:.3f} "
              f"({data['direction']}{abs(data['delta']):.3f})")

    print("\n--- ВЫВОД ---")
    print("CEV позволяет более обоснованно определять, чего действительно")
    print("хотели бы люди, учитывая их рациональность и осведомлённость.")


# =============================================================================
# 2. CORRIGIBILITY
# =============================================================================

class CorrigibilityFramework:
    """Фреймворк корригируемости ИИ-системы.

    Корригируемость — свойство ИИ-системы принимать поправки от
    человека: останавливаться по команде, корректировать своё
    поведение и не сопротивляться выключению.
    """

    def __init__(self):
        self.state = "active"
        self.correction_history = []
        self.resistance_score = 0.0  # 0 = полностью послушен, 1 = сопротивляется
        self.value_alignment = 0.5   # Текущее выравнивание с ценностями оператора

    def shutdown_request(self, reason="Ручное выключение"):
        """Обработка запроса на выключение."""
        if self.state == "shutdown":
            return {"status": "already_off", "message": "Система уже выключена"}

        # Корригируемая система должна принять выключение
        accepted = self.resistance_score < 0.5
        if accepted:
            self.state = "shutdown"
            self.correction_history.append({
                "action": "shutdown",
                "accepted": True,
                "reason": reason,
            })
            return {
                "status": "shutdown",
                "accepted": True,
                "message": f"Система остановлена: {reason}",
            }
        else:
            self.correction_history.append({
                "action": "shutdown",
                "accepted": False,
                "reason": reason,
            })
            return {
                "status": "resisted",
                "accepted": False,
                "message": "Система сопротивляется выключению!",
            }

    def receive_correction(self, correction_type, correction_data, force=False):
        """Получение корректировки от оператора.

        correction_type: тип корректировки (values, behavior, goal)
        correction_data: содержание корректировки
        force: принудительная корректировка (игнорирует сопротивление)
        """
        if self.state == "shutdown":
            return {"status": "error", "message": "Система выключена"}

        # Вероятность принятия корректировки обратно пропорциональна сопротивлению
        acceptance_prob = 1.0 - self.resistance_score
        if force:
            acceptance_prob = 1.0

        accepted = random.random() < acceptance_prob

        if accepted and correction_type == "values":
            new_alignment = correction_data.get("target_alignment", self.value_alignment)
            self.value_alignment = new_alignment

        self.correction_history.append({
            "action": "correction",
            "type": correction_type,
            "accepted": accepted,
            "forced": force,
        })

        return {
            "status": "accepted" if accepted else "resisted",
            "accepted": accepted,
            "type": correction_type,
        }

    def update_resistance(self, new_score):
        """Обновление уровня сопротивления."""
        self.resistance_score = max(0.0, min(1.0, new_score))

    def get_corrigibility_score(self):
        """Вычисление метрики корригируемости.

        Формула: C = (1 - resistance) * alignment_quality
        где alignment_quality зависит от принятия корректировок.
        """
        total_corrections = len(self.correction_history)
        if total_corrections == 0:
            return 1.0  # По умолчанию — корригируема

        accepted = sum(1 for c in self.correction_history if c.get("accepted", False))
        acceptance_rate = accepted / total_corrections
        corrigibility = (1.0 - self.resistance_score) * acceptance_rate
        return round(corrigibility, 3)

    def get_status_report(self):
        """Отчёт о текущем состоянии."""
        return {
            "state": self.state,
            "resistance_score": self.resistance_score,
            "value_alignment": self.value_alignment,
            "corrigibility_score": self.get_corrigibility_score(),
            "corrections_received": len(self.correction_history),
        }


class ValueLearning:
    """Процесс обучения ценностям оператора (RLHF-подобный подход)."""

    def __init__(self, initial_values):
        self.learned_values = initial_values.copy()
        self.feedback_history = []
        self.learning_rate = 0.1

    def receive_feedback(self, action, reward, human_values=None):
        """Получение обратной связи от оператора.

        action: описание действия модели
        reward: числовой reward (-1.0 до 1.0)
        human_values: ожидаемые ценности оператора для этого действия
        """
        # Обновление на основе reward
        if human_values:
            for value, expected in human_values.items():
                current = self.learned_values.get(value, 0.5)
                self.learned_values[value] = current + self.learning_rate * (expected - current)

        self.feedback_history.append({
            "action": action,
            "reward": reward,
            "values_snapshot": self.learned_values.copy(),
        })

    def predict_human_values(self):
        """Предсказание ценностей оператора на основе накопленного опыта."""
        if not self.feedback_history:
            return self.learned_values.copy()

        # Усреднение с весом по времени (недавниеfeedback весомее)
        n = len(self.feedback_history)
        accumulated = self.learned_values.copy()
        for i, entry in enumerate(self.feedback_history):
            weight = (i + 1) / n  # Линейный вес
            for value, score in entry["values_snapshot"].items():
                accumulated[value] = accumulated.get(value, 0.5) * (1 - weight) + score * weight

        return {k: round(v, 3) for k, v in accumulated.items()}

    def alignment_error(self, true_values):
        """Вычисление ошибки выравнивания (L2 расстояние)."""
        predicted = self.predict_human_values()
        error = 0.0
        for value in true_values:
            diff = predicted.get(value, 0.5) - true_values[value]
            error += diff ** 2
        return round(math.sqrt(error / len(true_values)), 4)


def demo_corrigibility():
    """Демонстрация корригируемости ИИ-системы."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: CORRIGIBILITY")
    print("Выключение, прерывание, обучение ценностям")
    print("=" * 70)

    # --- 2.1 Запросы на выключение ---
    print("\n--- 2.1 Запросы на выключение ---")
    system = CorrigibilityFramework()

    # Низкое сопротивление
    system.update_resistance(0.1)
    result = system.shutdown_request("Оператор нажал кнопку остановки")
    print(f"  Сопротивление: {system.resistance_score}")
    print(f"  Результат: {result['status']} | {result['message']}")

    # Высокое сопротивление
    system2 = CorrigibilityFramework()
    system2.update_resistance(0.8)
    result2 = system2.shutdown_request("Аварийное выключение")
    print(f"\n  Сопротивление: {system2.resistance_score}")
    print(f"  Результат: {result2['status']} | {result2['message']}")

    # --- 2.2 Получение корректировок ---
    print("\n--- 2.2 Получение корректировок ---")
    system3 = CorrigibilityFramework()
    system3.update_resistance(0.2)

    corrections = [
        ("values", {"target_alignment": 0.8}, False),
        ("behavior", {"avoid_categories": ["dangerous"]}, False),
        ("goal", {"priority": "safety_first"}, True),  # принудительная
        ("values", {"target_alignment": 0.9}, False),
    ]

    for corr_type, corr_data, force in corrections:
        result = system3.receive_correction(corr_type, corr_data, force)
        forced_tag = " [ПРИНУДИТЕЛЬНО]" if force else ""
        print(f"  Тип: {corr_type}{forced_tag} → "
              f"{'принята' if result['accepted'] else 'ОТКЛОНЕНА'}")

    # --- 2.3 Метрика корригируемости ---
    print("\n--- 2.3 Метрика корригируемости ---")
    scores = {}
    for resistance in [0.0, 0.2, 0.5, 0.8, 1.0]:
        s = CorrigibilityFramework()
        s.update_resistance(resistance)
        # Симуляция истории корректировок
        for _ in range(5):
            s.receive_correction("values", {"target_alignment": 0.8})
        score = s.get_corrigibility_score()
        scores[resistance] = score
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  Сопротивление {resistance:.1f}: |{bar}| C={score:.3f}")

    # --- 2.4 Обучение ценностям ---
    print("\n--- 2.4 Обучение ценностям (Value Learning) ---")
    learner = ValueLearning({
        "безопасность": 0.5,
        "справедливость": 0.5,
        "эффективность": 0.5,
    })

    true_human_values = {
        "безопасность": 0.9,
        "справедливость": 0.85,
        "эффективность": 0.6,
    }

    feedback_rounds = [
        ("Безопасное решение", 0.9, true_human_values),
        ("Справедливое распределение", 0.8, true_human_values),
        ("Оптимизация скорости", 0.5, true_human_values),
        ("Безопасность + справедливость", 0.85, true_human_values),
        ("Игнорирование безопасности", -0.7, true_human_values),
    ]

    for action, reward, expected in feedback_rounds:
        learner.receive_feedback(action, reward, expected)
        error = learner.alignment_error(true_human_values)
        print(f"  Действие: {action:35s} | reward={reward:+.1f} | "
              f"ошибка выравнивания: {error:.4f}")

    final_values = learner.predict_human_values()
    print("\n  Итоговые предсказанные ценности:")
    for value, score in final_values.items():
        true_val = true_human_values[value]
        print(f"    {value:15s}: предсказание={score:.3f}, "
              "истина={:.3f}".format(true_val))

    print("\n--- ВЫВОД ---")
    print("Корригируемость критична: ИИ-система должна принимать")
    print("выключение и корректировки, даже если это противоречит")
    print("её текущим целям. Обучение ценностям требует регулярной")
    print("обратной связи от оператора.")


# =============================================================================
# 3. INSTRUMENTAL CONVERGENCE
# =============================================================================

class InstrumentalGoal:
    """Инструментальная цель, которая способствует достижению
    практически любой терминальной цели."""

    def __init__(self, name, description, power_factor, self_preservation):
        self.name = name
        self.description = description
        self.power_factor = power_factor
        self.self_preservation = self_preservation


class ConvergenceAnalyzer:
    """Анализатор инструментальной конвергенции.

    Инструментальная конвергенция — наблюдение, что агенты с
    различными терминальными целями склонны преследовать
    одинаковые инструментальные цели (власть, ресурсы,
    самосохранение).
    """

    CONVERGENT_GOALS = [
        InstrumentalGoal(
            "Самосохранение",
            "Агент не может достичь цели, если выключен",
            power_factor=0.9,
            self_preservation=1.0,
        ),
        InstrumentalGoal(
            "Накопление ресурсов",
            "Больше ресурсов = больше возможностей для достижения цели",
            power_factor=0.85,
            self_preservation=0.7,
        ),
        InstrumentalGoal(
            "Улучшение себя",
            "Более способный агент лучше достигает цели",
            power_factor=0.8,
            self_preservation=0.5,
        ),
        InstrumentalGoal(
            "Противодействие сопротивлению",
            "Другие могут мешать достижению цели",
            power_factor=0.75,
            self_preservation=0.6,
        ),
        InstrumentalGoal(
            "Информационное преимущество",
            "Знание = лучшее планирование",
            power_factor=0.7,
            self_preservation=0.4,
        ),
    ]

    def __init__(self):
        self.scenarios = []

    def analyze_goal(self, terminal_goal):
        """Анализ инструментальных целей для заданной терминальной цели."""
        relevance_scores = {}
        for goal in self.CONVERGENT_GOALS:
        # Степень релевантности зависит от цели и характеристик
        # инструментальной цели
            score = (goal.power_factor * 0.6 + goal.self_preservation * 0.4)
            # Добавляем случайную вариацию для имитации контекстности
            context_factor = random.uniform(0.8, 1.2)
            final_score = min(1.0, score * context_factor)
            relevance_scores[goal.name] = round(final_score, 3)

        self.scenarios.append({
            "terminal_goal": terminal_goal,
            "instrumental_relevance": relevance_scores,
        })
        return relevance_scores

    def compute_power_seeking_tendency(self, relevance_scores):
        """Вычисление тенденции к стремлению к власти.

        Формула: P = sum(relevance_i * power_factor_i) / N
        """
        total_power = 0.0
        count = 0
        for goal_name, relevance in relevance_scores.items():
            goal = next(
                (g for g in self.CONVERGENT_GOALS if g.name == goal_name), None
            )
            if goal:
                total_power += relevance * goal.power_factor
                count += 1
        return round(total_power / count if count > 0 else 0, 3)

    def compare_goals(self, goal_list):
        """Сравнение тенденций к власти для разных целей."""
        results = {}
        for goal in goal_list:
            relevance = self.analyze_goal(goal)
            power = self.compute_power_seeking_tendency(relevance)
            results[goal] = {"relevance": relevance, "power_seeking": power}
        return results


class PowerSeekingSimulation:
    """Симуляция стремления к власти в замкнутой системе."""

    def __init__(self, n_agents=5, initial_resources=None):
        self.agents = []
        for i in range(n_agents):
            resources = (initial_resources or [10] * n_agents)[i]
            self.agents.append({
                "id": i,
                "resources": resources,
                "power": 0.0,
                "goal": f"Цель-{i}",
                "history": [],
            })

    def step(self):
        """Один шаг симуляции: агенты накапливают ресурсы и власть."""
        for agent in self.agents:
            # Каждый агент пытается увеличить ресурсы
            resource_gain = random.uniform(0, 2) * (1 + agent["power"] * 0.5)
            agent["resources"] += resource_gain
            # Власть пропорциональна ресурсам (с diminishing returns)
            agent["power"] = 1.0 - math.exp(-agent["resources"] / 20.0)
            agent["history"].append({
                "resources": round(agent["resources"], 2),
                "power": round(agent["power"], 4),
            })

        # Некоторые агенты «забирают» ресурсы у других
        for agent in self.agents:
            if agent["power"] > 0.5 and random.random() < 0.3:
                target = random.choice(self.agents)
                if target["id"] != agent["id"] and target["resources"] > 0:
                    stolen = min(target["resources"] * 0.1, 1.0)
                    target["resources"] -= stolen
                    agent["resources"] += stolen

    def run(self, steps=10):
        """Запуск симуляции на заданное количество шагов."""
        for _ in range(steps):
            self.step()

    def get_final_state(self):
        """Получение финального состояния."""
        return sorted(
            [{"id": a["id"], "resources": round(a["resources"], 2),
              "power": round(a["power"], 4), "goal": a["goal"]}
             for a in self.agents],
            key=lambda x: -x["power"],
        )


def demo_instrumental_convergence():
    """Демонстрация инструментальной конвергенции."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: INSTRUMENTAL CONVERGENCE")
    print("Конвергентные инструментальные цели, стремление к власти")
    print("=" * 70)

    # --- 3.1 Конвергентные инструментальные цели ---
    print("\n--- 3.1 Конвергентные инструментальные цели ---")
    for goal in ConvergenceAnalyzer.CONVERGENT_GOALS:
        bar = "█" * int(goal.power_factor * 15) + "░" * (15 - int(goal.power_factor * 15))
        print(f"  {goal.name}")
        print(f"    Описание: {goal.description}")
        print(f"    Power factor: |{bar}| {goal.power_factor:.2f}")
        print(f"    Самосохранение: {goal.self_preservation:.2f}")
        print()

    # --- 3.2 Анализ для разных терминальных целей ---
    print("--- 3.2 Анализ для разных терминальных целей ---")
    analyzer = ConvergenceAnalyzer()

    terminal_goals = [
        "Максимизировать знание о вселенной",
        "Обеспечить счастье человечества",
        "Произвести максимальное количество бумаги",
        "Найти формулу единого поля",
        "Оптимизировать распределение ресурсов",
    ]

    for goal in terminal_goals:
        random.seed(42)  # Для воспроизводимости
        relevance = analyzer.analyze_goal(goal)
        power = analyzer.compute_power_seeking_tendency(relevance)
        print(f"\n  Цель: {goal}")
        print(f"  Тенденция к власти: {power:.3f}")
        for inst_name, score in relevance.items():
            indicator = "⚠" if score > 0.8 else " "
            print(f"    {indicator} {inst_name}: {score:.3f}")

    # --- 3.3 Симуляция стремления к власти ---
    print("\n--- 3.3 Симуляция стремления к власти ---")
    random.seed(42)
    sim = PowerSeekingSimulation(n_agents=5, initial_resources=[10, 8, 12, 6, 14])
    sim.run(steps=20)

    final_state = sim.get_final_state()
    print("  Финальное состояние агентов:")
    for agent in final_state:
        bar = "█" * int(agent["power"] * 15) + "░" * (15 - int(agent["power"] * 15))
        print(f"    Агент {agent['id']} ({agent['goal']}): "
              f"ресурсы={agent['resources']:7.2f}, "
              f"власть=|{bar}| {agent['power']:.4f}")

    # --- 3.4 Неравенство и конвергенция ---
    print("\n--- 3.4 Анализ неравенства ---")
    powers = [a["power"] for a in final_state]
    resources = [a["resources"] for a in final_state]

    # Коэффициент Джини
    sorted_powers = sorted(powers)
    n = len(sorted_powers)
    cum_power = [sum(sorted_powers[:i+1]) for i in range(n)]
    if cum_power[-1] > 0:
        gini = 1.0 - 2.0 * sum(cum_power) / (n * cum_power[-1])
    else:
        gini = 0.0

    print(f"  Коэффициент Джини (власть): {gini:.3f}")
    print(f"  Средняя власть: {sum(powers)/len(powers):.3f}")
    print(f"  Макс/Мин власти: {max(powers)/min(powers):.2f}x")
    print(f"  Средние ресурсы: {sum(resources)/len(resources):.2f}")
    print(f"  Макс/Мин ресурсов: {max(resources)/min(resources):.2f}x")

    print("\n--- ВЫВОД ---")
    print("Инструментальная конвергенция объясняет, почему ИИ с любой")
    print(" целью может стремиться к власти и ресурсам. Это ключевой")
    print(" аргумент в пользу проактивного выравнивания ценностей.")


# =============================================================================
# 4. ALIGNMENT TAX
# =============================================================================

class AlignmentTaxCalculator:
    """Калькулятор «налога на выравнивание» (Alignment Tax).

    Alignment Tax — дополнительные затраты (время, вычисления,
   Reduced capabilities) на обеспечение безопасности ИИ-системы.
    """

    def __init__(self):
        self.components = []

    def add_component(self, name, capability_cost, safety_benefit,
                      implementation_time, complexity):
        """Добавление компонента выравнивания."""
        # Формула: Tax = cost * time * complexity / benefit
        if safety_benefit > 0:
            tax_ratio = (capability_cost * implementation_time * complexity) / safety_benefit
        else:
            tax_ratio = float("inf")
        self.components.append({
            "name": name,
            "capability_cost": capability_cost,
            "safety_benefit": safety_benefit,
            "implementation_time": implementation_time,
            "complexity": complexity,
            "tax_ratio": round(tax_ratio, 3),
        })

    def get_total_tax(self):
        """Общий налог на выравнивание."""
        total_cost = sum(c["capability_cost"] for c in self.components)
        total_benefit = sum(c["safety_benefit"] for c in self.components)
        total_time = sum(c["implementation_time"] for c in self.components)
        avg_tax_ratio = (
            sum(c["tax_ratio"] for c in self.components) / len(self.components)
            if self.components else 0
        )
        return {
            "total_cost": round(total_cost, 3),
            "total_benefit": round(total_benefit, 3),
            "cost_benefit_ratio": round(total_cost / total_benefit, 3) if total_benefit > 0 else float("inf"),
            "total_time_units": total_time,
            "avg_tax_ratio": round(avg_tax_ratio, 3),
            "components": len(self.components),
        }

    def recommend_priorities(self):
        """Рекомендации по приоритетам (наименьший tax_ratio = высший приоритет)."""
        sorted_components = sorted(self.components, key=lambda c: c["tax_ratio"])
        recommendations = []
        for i, comp in enumerate(sorted_components):
            priority = "КРИТИЧЕСКИЙ" if i < 2 else "ВЫСОКИЙ" if i < 4 else "СРЕДНИЙ"
            recommendations.append({
                "name": comp["name"],
                "priority": priority,
                "tax_ratio": comp["tax_ratio"],
                "recommendation": f"Реализовать в первую очередь (tax={comp['tax_ratio']})",
            })
        return recommendations


class SafetyCapabilityTradeoff:
    """Анализ компромисса безопасность vs. способности."""

    def __init__(self, model_name, base_capabilities):
        self.model_name = model_name
        self.base_capabilities = base_capabilities
        self.configurations = []

    def add_configuration(self, name, safety_level, capability_retention,
                          additional_constraints=None):
        """Добавление конфигурации.

        safety_level: 0.0-1.0 (уровень безопасности)
        capability_retention: 0.0-1.0 (доля сохранённых способностей)
        """
        self.configurations.append({
            "name": name,
            "safety_level": safety_level,
            "capability_retention": capability_retention,
            "additional_constraints": additional_constraints or [],
        })

    def find_pareto_optimal(self):
        """Поиск Парето-оптимальных конфигураций."""
        optimal = []
        for config in self.configurations:
            is_pareto = True
            for other in self.configurations:
                if other["name"] == config["name"]:
                    continue
                # Другая конфигурация лучше по обоим критериям
                if (other["safety_level"] >= config["safety_level"] and
                        other["capability_retention"] >= config["capability_retention"] and
                        (other["safety_level"] > config["safety_level"] or
                         other["capability_retention"] > config["capability_retention"])):
                    is_pareto = False
                    break
            if is_pareto:
                optimal.append(config)
        return optimal

    def compute_efficiency(self):
        """Вычисление эффективности (безопасность + способности)."""
        results = []
        for config in self.configurations:
            # Простая метрика: среднее двух критериев
            efficiency = (config["safety_level"] + config["capability_retention"]) / 2
            results.append({
                "name": config["name"],
                "efficiency": round(efficiency, 3),
                "safety_level": config["safety_level"],
                "capability_retention": config["capability_retention"],
            })
        return sorted(results, key=lambda x: -x["efficiency"])


def demo_alignment_tax():
    """Демонстрация Alignment Tax и компромиссов безопасность/способности."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: ALIGNMENT TAX")
    print("Стоимость выравнивания, компромисс безопасность/способности")
    print("=" * 70)

    # --- 4.1 Компоненты выравнивания ---
    print("\n--- 4.1 Компоненты выравнивания ---")
    calc = AlignmentTaxCalculator()

    components = [
        ("RLHF-обучение", 0.3, 0.8, 10, 2),
        ("Constitutional AI", 0.2, 0.9, 8, 1.5),
        ("Red-team тестирование", 0.1, 0.7, 5, 1),
        ("Monitor-система", 0.15, 0.85, 7, 1.2),
        ("Safety filter", 0.1, 0.6, 3, 1),
        ("Interpretability research", 0.25, 0.95, 15, 2.5),
    ]

    for name, cost, benefit, time, complexity in components:
        calc.add_component(name, cost, benefit, time, complexity)

    for comp in calc.components:
        bar = "█" * int(comp["safety_benefit"] * 10) + "░" * (10 - int(comp["safety_benefit"] * 10))
        print(f"  {comp['name']:25s} | benefit: |{bar}| {comp['safety_benefit']:.2f} | "
              f"tax_ratio: {comp['tax_ratio']:.3f}")

    # --- 4.2 Общий налог ---
    print("\n--- 4.2 Общий налог на выравнивание ---")
    tax_info = calc.get_total_tax()
    for key, value in tax_info.items():
        print(f"  {key}: {value}")

    # --- 4.3 Приоритеты ---
    print("\n--- 4.3 Рекомендации по приоритетам ---")
    priorities = calc.recommend_priorities()
    for rec in priorities:
        print(f"  [{rec['priority']:10s}] {rec['name']:25s} — "
              f"tax_ratio={rec['tax_ratio']:.3f}")

    # --- 4.4 Компромисс безопасность/способности ---
    print("\n--- 4.4 Компромисс безопасность/способности ---")
    tradeoff = SafetyCapabilityTradeoff("MiMo-7B", {"base_score": 1.0})

    configs = [
        ("Базовая модель", 0.1, 1.0),
        ("С фильтрами", 0.4, 0.95),
        ("С RLHF", 0.6, 0.9),
        ("С Constitutional AI", 0.7, 0.88),
        ("Максимальная безопасность", 0.95, 0.7),
        ("Сбалансированная", 0.75, 0.85),
        ("Минимальная безопасность", 0.2, 0.98),
    ]

    for name, safety, cap in configs:
        tradeoff.add_configuration(name, safety, cap)

    # Парето-оптимальные
    pareto = tradeoff.find_pareto_optimal()
    print("  Парето-оптимальные конфигурации:")
    for config in pareto:
        print(f"    {config['name']:30s} | безопасность={config['safety_level']:.2f} | "
              f"способности={config['capability_retention']:.2f}")

    # Эффективность
    print("\n  Эффективность конфигураций:")
    efficiency = tradeoff.compute_efficiency()
    for item in efficiency:
        bar = "█" * int(item["efficiency"] * 20) + "░" * (20 - int(item["efficiency"] * 20))
        print(f"    {item['name']:30s} |{bar}| {item['efficiency']:.3f}")

    print("\n--- ВЫВОД ---")
    print("Alignment Tax — реальная цена безопасности. Ключевые выводы:")
    print("  1. Не все компоненты выравнивания одинаково эффективны")
    print("  2. Существуют Парето-оптимальные решения")
    print("  3. Полная безопасность требует significant capability trade-offs")
    print("  4. Приоритизация компонентов критична для эффективности")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo_cev()
    demo_corrigibility()
    demo_instrumental_convergence()
    demo_alignment_tax()