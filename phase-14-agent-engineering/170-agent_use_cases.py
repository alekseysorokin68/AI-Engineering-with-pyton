"""170 — Agent Use Cases: поддержка клиентов, исследования, анализ данных

Темы:
  1. Customer Support Agent — классификация тикетов, генерация ответов, эскалация
  2. Research Agent — сбор информации, синтез, генерация отчётов
  3. Data Analysis Agent — исследование данных, визуализация, генерация выводов
  4. Creative Agent — генерация контента, мозговой штурм, итерации

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


# ===========================================================================
# Демо 1: Customer Support Agent — тикеты, ответы, эскалация
# ===========================================================================
def demo_customer_support():
    print("=" * 70)
    print("ДЕМО 1: Customer Support Agent — агент поддержки клиентов")
    print("=" * 70)

    # --- 1.1 Классификация тикетов ---
    print("\n--- 1.1 Классификация тикетов ---")

    class TicketClassifier:
        """Классификатор обращений клиентов."""

        # Ключевые слова для каждой категории
        CATEGORIES = {
            "billing": ["оплата", "платёж", "счёт", "деньги", "возврат", "списание", "тариф", "подписка"],
            "technical": ["ошибка", "баг", "не работает", "вылетает", "тормозит", "глюк", "сбой", "обновление"],
            "account": ["пароль", "вход", "регистрация", "аккаунт", "профиль", "удалить", "блокировка"],
            "feature": ["функция", "возможность", "добавить", "улучшить", "предложение", "интеграция"],
            "security": ["взлом", "утечка", "безопасность", "подозрительно", "фишинг", "спам"],
        }

        def classify(self, text):
            """Классифицирует обращение по тексту."""
            text_lower = text.lower()
            scores = {}
            for category, keywords in self.CATEGORIES.items():
                # Подсчёт совпадений ключевых слов
                score = sum(1 for kw in keywords if kw in text_lower)
                scores[category] = score

            # Выбираем категорию с максимальным score
            if max(scores.values()) == 0:
                return {"category": "general", "confidence": 0.0, "scores": scores}

            best_category = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[best_category] / total if total > 0 else 0
            return {
                "category": best_category,
                "confidence": round(confidence, 2),
                "scores": scores,
            }

    classifier = TicketClassifier()

    # Примеры обращений
    tickets = [
        "Мой платёж не прошёл, списались деньги дважды",
        "Приложение вылетает при запуске, ошибка 500",
        "Не могу войти в аккаунт, забыл пароль",
        "Было бы хорошо добавить интеграцию с Telegram",
        "Получил подозрительное письмо от вашего имени",
    ]

    print("Классификация обращений:")
    for i, ticket in enumerate(tickets):
        result = classifier.classify(ticket)
        print(f"\n  Тикет {i + 1}: '{ticket[:50]}...'")
        print(f"    Категория: {result['category']}")
        print(f"    Уверенность: {result['confidence']:.0%}")

    # --- 1.2 Генерация ответов ---
    print("\n--- 1.2 Генерация ответов ---")

    class ResponseGenerator:
        """Генератор ответов для клиентов."""

        TEMPLATES = {
            "billing": [
                "Здравствуйте! Мы обнаружили проблему с вашим платежом.",
                "Ваш запрос на возврат принят. Средства вернутся в течение 3-5 рабочих дней.",
                "Для уточнения по счёту, пожалуйста, предоставьте номер заказа.",
            ],
            "technical": [
                "Спасибо за обращение! Мы работаем над исправлением проблемы.",
                "Пожалуйста, попробуйте очистить кэш и перезапустить приложение.",
                "Мы зафиксировали баг и включили исправление в следующее обновление.",
            ],
            "account": [
                "Для восстановления доступа перейдите по ссылке 'Забыли пароль'.",
                "Ваш аккаунт успешно разблокирован. Рекомендуем сменить пароль.",
                "Для удаления аккаунта, пожалуйста, подтвердите личность.",
            ],
            "feature": [
                "Спасибо за предложение! Мы передадим его в продуктовую команду.",
                "Эта функция запланирована на следующий квартал.",
                "Ваше предложение очень ценно. Мы рассмотрим его на ближайшем совещании.",
            ],
            "security": [
                "Срочно: рекомендуем сменить пароль и включить двухфакторную аутентификацию.",
                "Мы проверим подозрительную активность в вашем аккаунте.",
                "Не переходите по ссылкам в подозрительных письмах. Мы уже расследуем инцидент.",
            ],
        }

        def generate(self, category, context=None):
            """Генерирует ответ для заданной категории."""
            templates = self.TEMPLATES.get(category, self.TEMPLATES["technical"])
            # Выбираем шаблон на основе контекста
            if context:
                # Простая эвристика: если упоминается срочность
                urgency = any(w in context.lower() for w in ["срочно", "немедленно", "проблема"])
                if urgency:
                    return templates[0]  # Первый шаблон обычно более формальный
            return random.choice(templates)

    generator = ResponseGenerator()

    print("Генерация ответов по категориям:")
    categories = ["billing", "technical", "account", "feature", "security"]
    for cat in categories:
        response = generator.generate(cat, "это срочно, помогите!")
        print(f"\n  [{cat.upper()}]")
        print(f"  Ответ: {response}")

    # --- 1.3 Эскалация ---
    print("\n--- 1.3 Система эскалации ---")

    class EscalationManager:
        """Менеджер эскалации обращений."""

        LEVELS = {
            1: {"name": "L1 — Первый уровень", "response_time": "24 часа", "skills": ["базовая поддержка"]},
            2: {"name": "L2 — Техническая поддержка", "response_time": "12 часов", "skills": ["углублённая диагностика"]},
            3: {"name": "L3 — Экспертный уровень", "response_time": "4 часа", "skills": ["разработка", "инфраструктура"]},
            4: {"name": "L4 — Менеджмент", "response_time": "2 часа", "skills": ["стратегические решения"]},
        }

        def __init__(self):
            self.tickets = []

        def create_ticket(self, text, category, priority="normal"):
            """Создаёт тикет с определённым приоритетом."""
            ticket_id = hashlib.md5(text.encode()).hexdigest()[:8]
            level = self._determine_level(category, priority)
            ticket = {
                "id": ticket_id,
                "text": text,
                "category": category,
                "priority": priority,
                "level": level,
                "status": "open",
                "escalations": 0,
            }
            self.tickets.append(ticket)
            return ticket

        def _determine_level(self, category, priority):
            """Определяет уровень поддержки."""
            if category == "security" or priority == "critical":
                return 3
            if priority == "high":
                return 2
            return 1

        def escalate(self, ticket_id, reason):
            """Эскалирует тикет на уровень выше."""
            for ticket in self.tickets:
                if ticket["id"] == ticket_id:
                    if ticket["level"] < 4:
                        ticket["level"] += 1
                        ticket["escalations"] += 1
                        return {
                            "new_level": ticket["level"],
                            "level_info": self.LEVELS[ticket["level"]],
                            "reason": reason,
                        }
                    return {"error": "Тикет уже на максимальном уровне"}
            return {"error": "Тикет не найден"}

    escalation = EscalationManager()

    # Создаём тикеты
    ticket1 = escalation.create_ticket("Ошибка при оплате", "billing", "normal")
    ticket2 = escalation.create_ticket("Взлом аккаунта!", "security", "critical")
    ticket3 = escalation.create_ticket("Не работает функция экспорта", "technical", "high")

    print("Созданные тикеты:")
    for t in escalation.tickets:
        level_info = escalation.LEVELS[t["level"]]
        print(f"  [{t['id']}] {t['category']} ({t['priority']}) -> {level_info['name']}")

    # Эскалация тикета
    result = escalation.escalate(ticket1["id"], "Клиент недоволен")
    print(f"\nЭскалация тикета {ticket1['id']}:")
    print(f"  Новый уровень: {result['level_info']['name']}")
    print(f"  Время ответа: {result['level_info']['response_time']}")

    # --- 1.4 Метрики поддержки ---
    print("\n--- 1.4 Метрики服务质量 поддержки ---")

    class SupportMetrics:
        """Метрики качества работы поддержки."""

        def __init__(self):
            self.tickets = []

        def add_ticket(self, category, resolution_time_hours, satisfaction, escalated):
            """Добавляет данные по тикету."""
            self.tickets.append({
                "category": category,
                "resolution_time": resolution_time_hours,
                "satisfaction": satisfaction,
                "escalated": escalated,
            })

        def calculate_metrics(self):
            """Вычисляет агрегированные метрики."""
            if not self.tickets:
                return {}

            total = len(self.tickets)
            avg_time = sum(t["resolution_time"] for t in self.tickets) / total
            avg_satisfaction = sum(t["satisfaction"] for t in self.tickets) / total
            escalation_rate = sum(1 for t in self.tickets if t["escalated"]) / total

            # Метрики по категориям
            by_category = collections.defaultdict(list)
            for t in self.tickets:
                by_category[t["category"]].append(t["resolution_time"])

            category_avg = {cat: sum(times) / len(times) for cat, times in by_category.items()}

            return {
                "total_tickets": total,
                "avg_resolution_time_hours": round(avg_time, 2),
                "avg_satisfaction": round(avg_satisfaction, 2),
                "escalation_rate": round(escalation_rate, 2),
                "category_avg_time": {k: round(v, 2) for k, v in category_avg.items()},
            }

    metrics = SupportMetrics()
    # Добавляем данные
    random.seed(42)
    for _ in range(20):
        cat = random.choice(["billing", "technical", "account", "feature"])
        time_h = random.uniform(1, 48)
        sat = random.uniform(3, 5)
        escalated = random.random() < 0.3
        metrics.add_ticket(cat, time_h, sat, escalated)

    result = metrics.calculate_metrics()
    print("Метрики поддержки:")
    print(f"  Всего тикетов: {result['total_tickets']}")
    print(f"  Среднее время решения: {result['avg_resolution_time_hours']} ч")
    print(f"  Средняя удовлетворённость: {result['avg_satisfaction']:.2f}/5.0")
    print(f"  Процент эскалаций: {result['escalation_rate']:.0%}")
    print("  Среднее время по категориям:")
    for cat, t in result["category_avg_time"].items():
        print(f"    {cat}: {t} ч")


# ===========================================================================
# Демо 2: Research Agent — сбор, синтез, отчёты
# ===========================================================================
def demo_research_agent():
    print("\n" + "=" * 70)
    print("ДЕМО 2: Research Agent — агент исследований")
    print("=" * 70)

    # --- 2.1 Сбор информации ---
    print("\n--- 2.1 Сбор информации из источников ---")

    class ResearchCollector:
        """Сборщик информации из различных источников."""

        def __init__(self):
            self.sources = []
            self.findings = []

        def add_source(self, name, url, reliability=0.8):
            """Добавляет источник."""
            self.sources.append({
                "name": name,
                "url": url,
                "reliability": reliability,
                "accessed": False,
            })

        def collect_finding(self, source_name, content, confidence=0.9):
            """Записывает находку из источника."""
            self.findings.append({
                "source": source_name,
                "content": content,
                "confidence": confidence,
                "timestamp": time.time(),
            })

        def get_findings_by_source(self):
            """Группирует находки по источнику."""
            by_source = collections.defaultdict(list)
            for f in self.findings:
                by_source[f["source"]].append(f)
            return dict(by_source)

        def summary(self):
            """Возвращает сводку по сбору."""
            by_source = self.get_findings_by_source()
            return {
                "total_sources": len(self.sources),
                "total_findings": len(self.findings),
                "sources_used": list(by_source.keys()),
                "avg_confidence": sum(f["confidence"] for f in self.findings) / max(1, len(self.findings)),
            }

    collector = ResearchCollector()

    # Добавляем источники
    sources_data = [
        ("arXiv论文", "https://arxiv.org/abs/2024.12345", 0.95),
        ("TechCrunch", "https://techcrunch.com/article", 0.75),
        ("GitHub Repo", "https://github.com/project", 0.85),
        ("Reddit讨论", "https://reddit.com/r/ai", 0.6),
    ]
    for name, url, rel in sources_data:
        collector.add_source(name, url, rel)

    # Собираем находки
    findings_data = [
        ("arXiv论文", "Модель достигает 95% точности на benchmark", 0.92),
        ("arXiv论文", "Использует архитектуру трансформер с 1B параметрами", 0.95),
        ("GitHub Repo", "Проект имеет 5000 звёзд, активно развивается", 0.88),
        ("TechCrunch", "Компания привлекла $50M инвестиций", 0.7),
        ("Reddit讨论", "Сообщество отмечает простоту API", 0.65),
    ]
    for source, content, conf in findings_data:
        collector.collect_finding(source, content, conf)

    print("Собранные источники:")
    for s in collector.sources:
        print(f"  {s['name']}: reliability={s['reliability']}")

    summary = collector.summary()
    print(f"\nИтого: {summary['total_findings']} находок из {len(summary['sources_used'])} источников")
    print(f"Средняя достоверность: {summary['avg_confidence']:.2f}")

    # --- 2.2 Синтез информации ---
    print("\n--- 2.2 Синтез и обобщение ---")

    class ResearchSynthesizer:
        """Синтезатор информации из нескольких источников."""

        def __init__(self):
            self.conclusions = []
            self.conflicts = []

        def synthesize(self, findings):
            """Синтезирует выводы из находок."""
            # Группируем по ключевым темам
            themes = collections.defaultdict(list)
            for f in findings:
                # Простое определение темы по ключевым словам
                content = f["content"].lower()
                if any(w in content for w in ["модель", "точность", "benchmark", "архитектура"]):
                    themes["производительность"].append(f)
                elif any(w in content for w in ["инвестиции", "компания", "рынок"]):
                    themes["бизнес"].append(f)
                elif any(w in content for w in ["проект", "github", "сообщество"]):
                    themes["экосистема"].append(f)
                else:
                    themes["прочее"].append(f)

            # Формируем выводы по темам
            for theme, theme_findings in themes.items():
                avg_confidence = sum(f["confidence"] for f in theme_findings) / len(theme_findings)
                self.conclusions.append({
                    "theme": theme,
                    "findings_count": len(theme_findings),
                    "avg_confidence": avg_confidence,
                    "key_points": [f["content"][:60] for f in theme_findings],
                })

            return self.conclusions

        def find_conflicts(self, findings):
            """Находит противоречия между источниками."""
            # Простая эвристика: ищем противоположные утверждения
            self.conflicts = []
            contents = [f["content"] for f in findings]
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    # Проверяем наличие противоположных слов
                    opposites = [("хорош", "плох"), ("высок", "низк"), ("быстр", "медлен")]
                    for pos, neg in opposites:
                        if (pos in contents[i].lower() and neg in contents[j].lower()) or \
                           (neg in contents[i].lower() and pos in contents[j].lower()):
                            self.conflicts.append({
                                "source1": findings[i]["source"],
                                "source2": findings[j]["source"],
                                "topic": f"Противоречие: {pos}/{neg}",
                            })
            return self.conflicts

    synthesizer = ResearchSynthesizer()
    conclusions = synthesizer.synthesize(collector.findings)

    print("Синтез по темам:")
    for c in conclusions:
        print(f"\n  Тема: {c['theme']}")
        print(f"  Найдено находок: {c['findings_count']}")
        print(f"  Средняя достоверность: {c['avg_confidence']:.2f}")
        for kp in c["key_points"]:
            print(f"    - {kp}")

    # --- 2.3 Генерация отчётов ---
    print("\n--- 2.3 Генерация отчёта ---")

    class ReportGenerator:
        """Генератор исследовательских отчётов."""

        def generate(self, topic, findings, conclusions):
            """Генерирует структурированный отчёт."""
            report = []
            report.append(f"# Отчёт: {topic}")
            report.append(f"\nДата: {time.strftime('%Y-%m-%d')}")
            report.append(f"Источников: {len(set(f['source'] for f in findings))}")
            report.append(f"Найдок: {len(findings)}")

            report.append("\n## Основные выводы")
            for i, c in enumerate(conclusions, 1):
                report.append(f"\n### {i}. {c['theme'].title()}")
                report.append(f"Найдено {c['findings_count']} подтверждений")
                for kp in c["key_points"]:
                    report.append(f"- {kp}")

            report.append("\n## Рекомендации")
            report.append("1. Продолжить мониторинг за developments")
            report.append("2. Провести собственное тестирование")
            report.append("3. Оценить практическое применение")

            return "\n".join(report)

    report_gen = ReportGenerator()
    report = report_gen.generate("Обзор AI-агентов", collector.findings, conclusions)
    print("Сгенерированный отчёт (первые 15 строк):")
    for line in report.split("\n")[:15]:
        print(f"  {line}")

    # --- 2.4 Оценка достоверности ---
    print("\n--- 2.4 Оценка достоверности источников ---")

    class CredibilityScorer:
        """Оценщик достоверности информации."""

        def __init__(self):
            self.source_scores = {}

        def score_source(self, source_name, factors):
            """Оценивает источник по факторам."""
            # Факторы:权威性, актуальность, независимость, репутация
            weights = {"authority": 0.3, "recency": 0.2, "independence": 0.25, "reputation": 0.25}
            score = sum(factors.get(f, 0.5) * w for f, w in weights.items())
            self.source_scores[source_name] = {
                "total": round(score, 3),
                "factors": factors,
            }
            return score

        def compare_sources(self):
            """Сравнивает источники по достоверности."""
            return sorted(
                self.source_scores.items(),
                key=lambda x: x[1]["total"],
                reverse=True,
            )

    scorer = CredibilityScorer()
    # Оцениваем источники
    scores_data = [
        ("arXiv论文", {"authority": 0.95, "recency": 0.8, "independence": 0.9, "reputation": 0.9}),
        ("TechCrunch", {"authority": 0.7, "recency": 0.9, "independence": 0.6, "reputation": 0.75}),
        ("GitHub Repo", {"authority": 0.85, "recency": 0.85, "independence": 0.8, "reputation": 0.8}),
        ("Reddit讨论", {"authority": 0.4, "recency": 0.7, "independence": 0.7, "reputation": 0.5}),
    ]
    for name, factors in scores_data:
        scorer.score_source(name, factors)

    print("Рейтинг источников по достоверности:")
    for i, (name, data) in enumerate(scorer.compare_sources(), 1):
        print(f"  {i}. {name}: {data['total']:.3f}")
        factors_str = ", ".join(f"{k}={v:.2f}" for k, v in data["factors"].items())
        print(f"     Факторы: {factors_str}")


# ===========================================================================
# Демо 3: Data Analysis Agent — исследование, визуализация, выводы
# ===========================================================================
def demo_data_analysis():
    print("\n" + "=" * 70)
    print("ДЕМО 3: Data Analysis Agent — агент анализа данных")
    print("=" * 70)

    # --- 3.1 Исследование данных ---
    print("\n--- 3.1 Исследование данных (Exploratory Data Analysis) ---")

    class DataExplorer:
        """Исследователь данных."""

        def __init__(self, data):
            self.data = data
            self.n = len(data)

        def basic_stats(self):
            """Вычисляет базовые статистики."""
            sorted_data = sorted(self.data)
            mean = sum(self.data) / self.n
            variance = sum((x - mean) ** 2 for x in self.data) / self.n
            std_dev = math.sqrt(variance)

            # Медиана
            if self.n % 2 == 0:
                median = (sorted_data[self.n // 2 - 1] + sorted_data[self.n // 2]) / 2
            else:
                median = sorted_data[self.n // 2]

            # Квартили
            q1_idx = self.n // 4
            q3_idx = (3 * self.n) // 4
            q1 = sorted_data[q1_idx]
            q3 = sorted_data[q3_idx]
            iqr = q3 - q1

            return {
                "count": self.n,
                "mean": round(mean, 4),
                "std": round(std_dev, 4),
                "min": min(self.data),
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": max(self.data),
                "iqr": round(iqr, 4),
                "range": max(self.data) - min(self.data),
            }

        def detect_outliers(self):
            """Обнаруживает выбросы методом IQR."""
            stats = self.basic_stats()
            lower = stats["q1"] - 1.5 * stats["iqr"]
            upper = stats["q3"] + 1.5 * stats["iqr"]
            outliers = [x for x in self.data if x < lower or x > upper]
            return {
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "outliers": outliers,
                "outlier_count": len(outliers),
            }

    # Генерируем данные
    random.seed(42)
    data = [random.gauss(100, 15) for _ in range(50)]
    data.append(200)  # Выброс
    data.append(10)   # Выброс

    explorer = DataExplorer(data)
    stats = explorer.basic_stats()
    print("Базовые статистики:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    outliers = explorer.detect_outliers()
    print(f"\nОбнаружено выбросов: {outliers['outlier_count']}")
    print(f"  Границы: [{outliers['lower_bound']}, {outliers['upper_bound']}]")
    if outliers["outliers"]:
        print(f"  Выбросы: {outliers['outliers']}")

    # --- 3.2 Предложения визуализации ---
    print("\n--- 3.2 Предложения визуализации ---")

    class VisualizationAdvisor:
        """Советник по визуализации данных."""

        RECOMMENDATIONS = {
            "single_numeric": {
                "chart": "Гистограмма (Histogram)",
                "reason": "Показывает распределение одной числовой переменной",
                "code": "plt.hist(data, bins=30)",
            },
            "two_numeric": {
                "chart": "Диаграмма рассеяния (Scatter Plot)",
                "reason": "Показывает связь между двумя числовыми переменными",
                "code": "plt.scatter(x, y)",
            },
            "categorical_numeric": {
                "chart": "Box Plot",
                "reason": "Сравнивает распределение числовой переменной по категориям",
                "code": "plt.boxplot([group1, group2, group3])",
            },
            "time_series": {
                "chart": "Линейный график (Line Chart)",
                "reason": "Показывает тенденции во времени",
                "code": "plt.plot(dates, values)",
            },
            "proportions": {
                "chart": "Круговая диаграмма (Pie Chart)",
                "reason": "Показывает доли от целого (до 5 категорий)",
                "code": "plt.pie(sizes, labels=labels)",
            },
        }

        def advise(self, data_type, description):
            """Рекомендует тип визуализации."""
            if data_type in self.RECOMMENDATIONS:
                rec = self.RECOMMENDATIONS[data_type]
                return {
                    "recommended": rec["chart"],
                    "reason": rec["reason"],
                    "example_code": rec["code"],
                    "description": description,
                }
            return {"error": f"Неизвестный тип данных: {data_type}"}

    advisor = VisualizationAdvisor()
    print("Рекомендации по визуализации:")
    scenarios = [
        ("single_numeric", "Распределение возраста пользователей"),
        ("two_numeric", "Зависимость выручки от рекламных расходов"),
        ("categorical_numeric", "Сравнение конверсии по каналам"),
        ("time_series", "Динамика активных пользователей"),
        ("proportions", "Доля рынка по компаниям"),
    ]
    for data_type, desc in scenarios:
        rec = advisor.advise(data_type, desc)
        print(f"\n  {desc}")
        print(f"    Тип: {data_type}")
        print(f"    Рекомендация: {rec['recommended']}")
        print(f"    Причина: {rec['reason']}")
        print(f"    Пример: {rec['example_code']}")

    # --- 3.3 Генерация выводов ---
    print("\n--- 3.3 Автоматическая генерация выводов ---")

    class InsightGenerator:
        """Генератор аналитических выводов."""

        def analyze_correlation(self, x_data, y_data):
            """Анализирует корреляцию между двумя наборами данных."""
            n = len(x_data)
            mean_x = sum(x_data) / n
            mean_y = sum(y_data) / n

            # Вычисляем коэффициент корреляции Пирсона
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_data, y_data))
            denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_data))
            denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_data))

            if denom_x == 0 or denom_y == 0:
                correlation = 0
            else:
                correlation = numerator / (denom_x * denom_y)

            # Интерпретация
            if abs(correlation) > 0.7:
                strength = "сильная"
            elif abs(correlation) > 0.4:
                strength = "умеренная"
            else:
                strength = "слабая"

            direction = "положительная" if correlation > 0 else "отрицательная"

            return {
                "correlation": round(correlation, 4),
                "strength": strength,
                "direction": direction,
                "interpretation": f"{strength} {direction} корреляция (r={correlation:.3f})",
            }

        def generate_insights(self, data, column_name):
            """Генерирует выводы по данным."""
            explorer = DataExplorer(data)
            stats = explorer.basic_stats()
            outliers = explorer.detect_outliers()

            insights = []
            # Анализ нормальности (эмпирическое правило)
            within_1std = sum(1 for x in data if abs(x - stats["mean"]) <= stats["std"])
            pct_within_1std = within_1std / len(data) * 100

            if 60 <= pct_within_1std <= 80:
                insights.append(f"Распределение {column_name} приблизительно нормальное ({pct_within_1std:.1f}% в пределах 1σ)")
            else:
                insights.append(f"Распределение {column_name} не является нормальным ({pct_within_1std:.1f}% в пределах 1σ)")

            if outliers["outlier_count"] > 0:
                insights.append(f"Обнаружено {outliers['outlier_count']} выбросов (вне [{outliers['lower_bound']}, {outliers['upper_bound']}])")

            if stats["range"] > 3 * stats["std"]:
                insights.append(f"Высокая вариативность: размах ({stats['range']:.2f}) > 3σ ({3 * stats['std']:.2f})")

            return insights

    generator = InsightGenerator()

    # Корреляция
    random.seed(42)
    x = [i * 2 + random.gauss(0, 5) for i in range(30)]
    y = [xi * 1.5 + random.gauss(0, 10) for xi in x]

    corr_result = generator.analyze_correlation(x, y)
    print(f"Анализ корреляции X и Y:")
    print(f"  Коэффициент Пирсона: {corr_result['correlation']}")
    print(f"  Интерпретация: {corr_result['interpretation']}")

    # Автоматические выводы
    insights = generator.generate_insights(data, "значения")
    print(f"\nАвтоматические выводы:")
    for insight in insights:
        print(f"  • {insight}")

    # --- 3.4 Рекомендации по обработке ---
    print("\n--- 3.4 Рекомендации по обработке данных ---")

    class DataProcessor:
        """Рекомендации по обработке данных."""

        @staticmethod
        def recommend_cleaning(data):
            """Рекомендует методы очистки."""
            recommendations = []
            explorer = DataExplorer(data)
            outliers = explorer.detect_outliers()

            if outliers["outlier_count"] > 0:
                recommendations.append({
                    "issue": f"{outliers['outlier_count']} выбросов",
                    "solution": "Замена на медиану или IQR-фильтрация",
                })

            # Проверка пропусков (в данных с None)
            missing = sum(1 for x in data if x is None)
            if missing > 0:
                recommendations.append({
                    "issue": f"{missing} пропущенных значений",
                    "solution": "Импутация средним/медианой или удаление",
                })

            return recommendations

        @staticmethod
        def recommend_features(data, target_corr=0.5):
            """Рекомендует инженерию признаков."""
            recommendations = []
            explorer = DataExplorer(data)
            stats = explorer.basic_stats()

            # Рекомендация нормализации
            if stats["std"] > 0:
                cv = stats["std"] / abs(stats["mean"]) if stats["mean"] != 0 else 0
                if cv > 0.5:
                    recommendations.append({
                        "feature": "Нормализация",
                        "reason": f"Высокий коэффициент вариации ({cv:.2f})",
                        "method": "StandardScaler или MinMaxScaler",
                    })

            # Рекомендация бинаризации
            if len(set(data)) > 10:
                recommendations.append({
                    "feature": "Бинаризация",
                    "reason": "Много уникальных значений",
                    "method": "Разбить на квантили",
                })

            return recommendations

    processor = DataProcessor()
    cleaning_recs = processor.recommend_cleaning(data)
    print("Рекомендации по очистке:")
    for rec in cleaning_recs:
        print(f"  Проблема: {rec['issue']}")
        print(f"  Решение: {rec['solution']}")

    feature_recs = processor.recommend_features(data)
    print("\nРекомендации по инженерии признаков:")
    for rec in feature_recs:
        print(f"  {rec['feature']}: {rec['reason']}")
        print(f"    Метод: {rec['method']}")


# ===========================================================================
# Демо 4: Creative Agent — контент, мозговой штурм, итерации
# ===========================================================================
def demo_creative_agent():
    print("\n" + "=" * 70)
    print("ДЕМО 4: Creative Agent — агент творческих задач")
    print("=" * 70)

    # --- 4.1 Генерация контента ---
    print("\n--- 4.1 Генерация контента ---")

    class ContentGenerator:
        """Генератор контента разных форматов."""

        TEMPLATES = {
            "email": {
                "structure": ["Тема: {topic}", "", "Уважаемый клиент!", "", "{body}", "", "С уважением, {sender}"],
                "formal_tone": True,
            },
            "social_post": {
                "structure": ["{hook}", "", "{body}", "", "{hashtags}", "", "{cta}"],
                "formal_tone": False,
            },
            "article": {
                "structure": ["# {title}", "", "## Введение", "{intro}", "", "## Основная часть", "{body}", "", "## Заключение", "{conclusion}"],
                "formal_tone": True,
            },
        }

        def generate(self, format_type, topic, details):
            """Генерирует контент в заданном формате."""
            template = self.TEMPLATES.get(format_type)
            if not template:
                return f"Неизвестный формат: {format_type}"

            # Заполняем шаблон
            structure = template["structure"]
            filled = []
            for line in structure:
                for key, value in details.items():
                    line = line.replace(f"{{{key}}}", value)
                filled.append(line)

            return "\n".join(filled)

    generator = ContentGenerator()

    # Генерируем email
    email = generator.generate(
        "email",
        "Обновление сервиса",
        {
            "topic": "Плановое обновление 15.07",
            "body": "Мы обновим систему 15 июля. Ожидается кратковременная недоступность.",
            "sender": "Команда поддержки",
        },
    )
    print("Сгенерированный email:")
    for line in email.split("\n"):
        print(f"  {line}")

    # Генерируем пост для соцсетей
    post = generator.generate(
        "social_post",
        "Новый продукт",
        {
            "hook": "Знакомьтесь: наш новый AI-ассистент!",
            "body": "Он понимает контекст и помогает решать задачи на русском языке.",
            "hashtags": "#AI #Нейросети #ИскусственныйИнтеллект",
            "cta": "Попробуйте бесплатно: example.com/ai",
        },
    )
    print("\nСгенерированный пост:")
    for line in post.split("\n"):
        print(f"  {line}")

    # --- 4.2 Мозговой штурм ---
    print("\n--- 4.2 Мозговой штурм (Brainstorming) ---")

    class BrainstormEngine:
        """Движок мозгового штурма."""

        def __init__(self):
            self.ideas = []
            self.connections = []

        def add_idea(self, text, category, rating=None):
            """Добавляет идею."""
            self.ideas.append({
                "id": len(self.ideas),
                "text": text,
                "category": category,
                "rating": rating or random.randint(1, 10),
            })

        def connect_ideas(self, id1, id2, strength=0.5):
            """Связывает две идеи."""
            self.connections.append({"from": id1, "to": id2, "strength": strength})

        def generate_variations(self, idea_id, n=3):
            """Генерирует вариации идеи."""
            if idea_id >= len(self.ideas):
                return []
            base = self.ideas[idea_id]
            variations = []
            prefixes = ["Улучшенная", "Альтернативная", "Минимальная", "Расширенная", "Простая"]
            suffixes = ["для бизнеса", "для образования", "для развлечений", "с ИИ", "с геймификацией"]

            for i in range(n):
                prefix = prefixes[i % len(prefixes)]
                suffix = suffixes[i % len(suffixes)]
                variations.append({
                    "original": base["text"],
                    "variation": f"{prefix} версия: {base['text']} ({suffix})",
                })
            return variations

        def top_ideas(self, n=5):
            """Возвращает лучшие идеи по рейтингу."""
            return sorted(self.ideas, key=lambda x: x["rating"], reverse=True)[:n]

    brainstorm = BrainstormEngine()

    # Добавляем идеи
    ideas_data = [
        ("Чат-бот для поддержки", "продукт", 8),
        ("Голосовой ассистент", "продукт", 9),
        ("Автоматизация отчётов", "инструмент", 7),
        ("Образовательная платформа", "образование", 8),
        ("Анализатор данных", "инструмент", 6),
    ]
    for text, cat, rating in ideas_data:
        brainstorm.add_idea(text, cat, rating)

    # Связываем идеи
    brainstorm.connect_ideas(0, 3, 0.7)
    brainstorm.connect_ideas(1, 0, 0.8)
    brainstorm.connect_ideas(2, 4, 0.6)

    print("Все идеи:")
    for idea in brainstorm.ideas:
        print(f"  [{idea['id']}] {idea['text']} (рейтинг: {idea['rating']})")

    print("\nЛучшие идеи:")
    for idea in brainstorm.top_ideas(3):
        print(f"  {idea['text']}: {idea['rating']}/10")

    # Вариации идеи
    variations = brainstorm.generate_variations(1, n=3)
    print(f"\nВариации идеи '{brainstorm.ideas[1]['text']}':")
    for v in variations:
        print(f"  - {v['variation']}")

    # --- 4.3 Итеративное улучшение ---
    print("\n--- 4.3 Итеративное улучшение контента ---")

    class IterativeImprover:
        """Итеративный улучшатель контента."""

        def __init__(self):
            self.iterations = []

        def improve(self, text, criteria):
            """Одна итерация улучшения."""
            improved = text
            changes = []

            # Проверяем длину
            if len(text) < 50:
                improved += ". " + criteria.get("expansion", "Добавляем больше деталей.")
                changes.append("Расширение текста")

            # Проверяем наличие цифр
            if not re.search(r"\d", text):
                number = random.randint(1, 100)
                improved = f"Около {number}% " + improved.lower()
                changes.append("Добавление конкретики")

            # Проверяем структуру
            if "\n" not in improved:
                improved = improved.replace(". ", ".\n- ", 1)
                changes.append("Добавление структуры")

            self.iterations.append({
                "original": text,
                "improved": improved,
                "changes": changes,
            })
            return improved

        def get_history(self):
            """Возвращает историю итераций."""
            return self.iterations

    improver = IterativeImprover()
    original_text = "Наш продукт помогает бизнесу."

    print(f"Исходный текст: '{original_text}'")
    current = original_text

    for i in range(3):
        current = improver.improve(current, {"expansion": "Он использует передовые технологии ИИ."})
        print(f"\nИтерация {i + 1}:")
        print(f"  Текст: {current}")
        changes = improver.iterations[-1]["changes"]
        if changes:
            print(f"  Изменения: {', '.join(changes)}")

    # --- 4.4 Оценка качества ---
    print("\n--- 4.4 Оценка качества контента ---")

    class QualityEvaluator:
        """Оценщик качества контента."""

        def evaluate(self, text):
            """Оценивает текст по нескольким критериям."""
            scores = {}

            # Длина текста
            word_count = len(text.split())
            if word_count < 10:
                scores["length"] = 3
            elif word_count < 50:
                scores["length"] = 7
            else:
                scores["length"] = 9

            # Структура (наличие заголовков, списков)
            structure_score = 5
            if "\n" in text:
                structure_score += 2
            if "-" in text or "*" in text:
                structure_score += 1
            if re.search(r"#{1,3}\s", text):
                structure_score += 2
            scores["structure"] = min(10, structure_score)

            # Уникальность (разнообразие слов)
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(1, len(words))
            scores["uniqueness"] = round(unique_ratio * 10, 1)

            # Общий балл
            scores["overall"] = round(sum(scores.values()) / len(scores), 1)

            return scores

    evaluator = QualityEvaluator()

    # Оцениваем разные тексты
    texts_to_evaluate = [
        "Короткий",
        "Это хороший текст, который содержит достаточно слов для оценки.",
        "# Заголовок\n\nОсновной текст с деталями.\n\n- Пункт 1\n- Пункт 2",
    ]

    print("Оценка качества текстов:")
    for text in texts_to_evaluate:
        scores = evaluator.evaluate(text)
        print(f"\n  Текст: '{text[:50]}...'")
        for criterion, score in scores.items():
            print(f"    {criterion}: {score}/10")


# ===========================================================================
# Запуск всех демонстраций
# ===========================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  170 — Agent Use Cases: поддержка, исследования, анализ           ║")
    print("╚" + "═" * 68 + "╝")
    print()

    demo_customer_support()
    demo_research_agent()
    demo_data_analysis()
    demo_creative_agent()

    print("\n" + "=" * 70)
    print("Все демонстрации завершены")
    print("=" * 70)
