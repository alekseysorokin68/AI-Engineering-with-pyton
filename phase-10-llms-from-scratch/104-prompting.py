"""
Prompting — Основы промптинга (From Scratch)
=============================================
Исследование техник промптинга без внешних зависимостей.
Моделируем поведение LLM на базе вероятностных шаблонов.

Техники:
  1. Zero-shot промптинг
  2. Few-shot промптинг
  3. Chain-of-Thought промптинг
  4. Оптимизация промптов
"""

import random

random.seed(42)


# ============================================================
#  Имитация LLM: простая вероятностная модель ответов
# ============================================================

class MockLLM:
    """Имитация LLM, которая выбирает ответы на основе промпта."""

    def __init__(self):
        self.knowledge = {
            "столица франции": "париж",
            "столица россии": "москва",
            "столица japan": "токио",
            "столица германии": "берлин",
            "2+2": "4",
            "2 + 2": "4",
            "3+5": "8",
            "3 + 5": "8",
            "10-3": "7",
            "квадрат 5": "25",
            "квадрат 7": "49",
            "квадрат 3": "9",
            "квадрат 12": "144",
            "квадрат 15": "225",
            "квадрат 25": "625",
            "квадрат 50": "2500",
            "квадрат 100": "10000",
            "квадрат 20": "400",
            "квадрат 10": "100",
            "переведи привет": "hello",
            "переведи спасибо": "thank you",
            "переведи пока": "goodbye",
            "классифицируй отличный продукт": "положительный",
            "классифицируй ужасный сервис": "отрицательный",
            "классифицируй нормально": "нейтральный",
            "классифицируй скучно": "отрицательный",
            "классифицируй супер": "положительный",
        }
        self.response_style = "default"

    def generate(self, prompt):
        prompt_lower = prompt.lower().strip()

        # Проверка на few-shot примеры
        if "пример:" in prompt_lower or "example:" in prompt_lower:
            return self._handle_few_shot(prompt_lower)

        # Проверка на chain-of-thought
        if "шаг" in prompt_lower or "step" in prompt_lower or "рассуждай" in prompt_lower:
            return self._handle_cot(prompt_lower)

        # Прямой вопрос — ищем по ключам с учётом вариаций
        for key, value in self.knowledge.items():
            key_clean = key.replace(" ", "").replace("+", "").replace("×", "")
            prompt_clean = prompt_lower.replace(" ", "").replace("=", "").replace("?", "").replace("²", "")
            if key_clean in prompt_clean or key in prompt_lower:
                if self.response_style == "verbose":
                    return f"На основе моих знаний, ответ: {value}"
                return value

            # Fuzzy: ключ содержит числа — проверяем что все токены ключа есть в промпте
            import re as _re
            key_nums = _re.findall(r'\d+', key)
            prompt_nums = _re.findall(r'\d+', prompt_lower)
            if key_nums and all(n in prompt_nums for n in key_nums):
                key_words = [w for w in key.split() if not w.isdigit()]
                if all(w in prompt_lower for w in key_words):
                    return value

        # Дополнительные шаблоны для перевода
        if "переведи" in prompt_lower and "привет" in prompt_lower:
            return "hello"
        if "переведи" in prompt_lower and "спасибо" in prompt_lower:
            return "thank you"
        if "переведи" in prompt_lower and "пока" in prompt_lower:
            return "goodbye"

        # Математика без CoT
        import re
        math_match = re.search(r'(\d+)\s*\+\s*(\d+)', prompt_lower)
        if math_match:
            a, b = int(math_match.group(1)), int(math_match.group(2))
            return str(a + b)
        math_mul = re.search(r'(\d+)\s*[×*]\s*(\d+)', prompt_lower)
        if math_mul:
            a, b = int(math_mul.group(1)), int(math_mul.group(2))
            return str(a * b)

        # Квадрат числа без CoT
        sq_match = re.search(r'(\d+)\s*²', prompt_lower)
        if sq_match:
            n = int(sq_match.group(1))
            return str(n * n)

        # Классификация
        positive_kw = ["отличн", "супер", "класс", "хорош", "прекрасн", "замечательн", "нравится", "рекомендую"]
        negative_kw = ["ужасн", "плох", "скучн", "отвратительн", "кошмар"]
        neutral_kw = ["нормальн", "средн", "обычн", "ничего особенного"]

        if "классифицируй" in prompt_lower or "отзыв" in prompt_lower:
            for w in positive_kw:
                if w in prompt_lower:
                    return "положительный"
            for w in negative_kw:
                if w in prompt_lower:
                    return "отрицательный"
            for w in neutral_kw:
                if w in prompt_lower:
                    return "нейтральный"

        return "не знаю"

    def _handle_few_shot(self, prompt):
        """Обработка few-shot промпта — извлекает паттерн из примеров."""
        lines = prompt.split("\n")
        examples = []
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            if line_lower.startswith("вход:") or line_lower.startswith("input:"):
                inp = line.split(":", 1)[1].strip().strip('"').strip("'")
                examples.append(("input", inp))
            elif line_lower.startswith("выход:") or line_lower.startswith("output:"):
                out = line.split(":", 1)[1].strip().strip('"').strip("'")
                examples.append(("output", out))

        # Последний вход без выхода — это запрос
        last_input = None
        for t, v in reversed(examples):
            if t == "input":
                last_input = v
                break

        if not last_input:
            return "не знаю"

        # Ищем точное совпадение в примерах (input → output)
        # Исключаем пары где выход пустой (это незавершённый запрос)
        example_map = {}
        for i in range(len(examples) - 1):
            if examples[i][0] == "input" and examples[i + 1][0] == "output":
                if examples[i + 1][1]:  # Только если выход не пустой
                    example_map[examples[i][1]] = examples[i + 1][1]

        if last_input in example_map:
            return example_map[last_input]

        # Паттерн для классификации (регистронезависимый поиск)
        last_input_lower = last_input.lower()
        positive_words = ["отличн", "супер", "класс", "хорош", "прекрасн", "замечательн"]
        negative_words = ["ужасн", "плох", "скучн", "отвратительн", "кошмар"]
        neutral_words = ["нормальн", "средн", "обычн"]

        for word in positive_words:
            if word in last_input_lower:
                return "положительный"
        for word in negative_words:
            if word in last_input_lower:
                return "отрицательный"
        for word in neutral_words:
            if word in last_input_lower:
                return "нейтральный"

        # Фоллбэк через знания
        for key, value in self.knowledge.items():
            if key in last_input_lower:
                return value

        return "нейтральный"

    def _handle_cot(self, prompt):
        """Обработка chain-of-thought промпта — модель рассуждает пошагово."""
        prompt_lower = prompt.lower()

        # Определяем тип задачи
        if any(w in prompt_lower for w in ["квадрат", "square", "возведи", "²"]):
            return self._cot_square(prompt_lower)
        elif any(w in prompt_lower for w in ["слож", "sum", "add", "+"]):
            return self._cot_math(prompt_lower, "сложение")
        elif any(w in prompt_lower for w in ["умнож", "multiply", "×"]):
            return self._cot_math(prompt_lower, "умножение")
        elif any(w in prompt_lower for w in ["переведи", "translate"]):
            return self._cot_translate(prompt_lower)
        else:
            return self._cot_general(prompt_lower)

    def _extract_number(self, text):
        """Извлекает число из текста."""
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])  # Берём последнее число
        return None

    def _cot_square(self, prompt):
        """Пошаговое вычисление квадрата."""
        n = self._extract_number(prompt)
        if n is None:
            return "Шаг 1: Не могу определить число.\nОтвет: не знаю"

        result = n * n
        steps = [
            f"Шаг 1: Определяю число — {n}",
            f"Шаг 2: Вычисляю квадрат: {n} × {n}",
            f"Шаг 3: {n} × {n} = {result}",
            f"Ответ: {result}"
        ]
        return "\n".join(steps)

    def _cot_math(self, prompt, operation):
        """Пошаговое математическое рассуждение."""
        import re
        numbers = re.findall(r'\d+', prompt)
        if len(numbers) < 2:
            return "Шаг 1: Не могу определить числа.\nОтвет: не знаю"

        a, b = int(numbers[0]), int(numbers[1])

        if operation == "сложение":
            result = a + b
            op_sym = "+"
        elif operation == "умножение":
            result = a * b
            op_sym = "×"
        else:
            result = a + b
            op_sym = "+"

        steps = [
            f"Шаг 1: Определяю числа — {a} и {b}",
            f"Шаг 2: Операция — {operation} ({op_sym})",
            f"Шаг 3: Вычисляю: {a} {op_sym} {b} = {result}",
            f"Ответ: {result}"
        ]
        return "\n".join(steps)

    def _cot_translate(self, prompt):
        """Пошаговый перевод."""
        if "привет" in prompt:
            steps = [
                "Шаг 1: Определяю слово — «привет»",
                "Шаг 2: Определяю язык назначения — английский",
                "Шаг 3: Ищу эквивалент — 'hello'",
                "Ответ: hello"
            ]
            return "\n".join(steps)
        elif "спасибо" in prompt:
            steps = [
                "Шаг 1: Определяю слово — «спасибо»",
                "Шаг 2: Определяю язык назначения — английский",
                "Шаг 3: Ищу эквивалент — 'thank you'",
                "Ответ: thank you"
            ]
            return "\n".join(steps)
        return "Шаг 1: Не могу определить слово для перевода.\nОтвет: не знаю"

    def _cot_general(self, prompt):
        """Общее пошаговое рассуждение."""
        for key, value in self.knowledge.items():
            if key in prompt:
                return f"Шаг 1: Анализирую вопрос.\nШаг 2: Нахожу информацию.\nОтвет: {value}"
        return "Шаг 1: Анализирую вопрос.\nОтвет: не знаю"

    def set_style(self, style):
        """Устанавливает стиль ответов: default / verbose."""
        self.response_style = style


# ============================================================
#  Демо 1: Zero-shot промптинг
# ============================================================

print("=" * 55)
print("ДЕМО 1: Zero-shot промптинг")
print("=" * 55)
print()

llm = MockLLM()

zero_shot_prompts = [
    ("Какая столица Франции?", "Простой вопрос-факт"),
    ("2 + 2 = ?", "Математическая операция"),
    ("Переведи на английский: привет", "Задача перевода"),
    ("Классифицируй отзыв: отличный продукт", "Классификация текста"),
]

for prompt, description in zero_shot_prompts:
    print(f"Задача: {description}")
    print(f"Промпт: \"{prompt}\"")
    answer = llm.generate(prompt)
    print(f"Ответ:  {answer}")
    print()

print("Вывод: Zero-shot — модель отвечает без примеров.")
print("  ✓ Просто и быстро")
print("  ✗ Требует качественной预训练 модели")
print("  ✗ Сложные задачи могут давать неточные ответы")


# ============================================================
#  Демо 2: Few-shot промптинг
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Few-shot промптинг")
print("=" * 55)
print()

# Few-shot промпт для классификации отзывов
few_shot_prompt = """Задача: классифицируй отзывы как положительные, отрицательные или нейтральные.

Пример:
Вход: "Отличный продукт, рекомендую!"
Выход: положительный

Пример:
Вход: "Ужасный сервис, больше не вернусь"
Выход: отрицательный

Пример:
Вход: "Нормально, ничего особенного"
Выход: нейтральный

Теперь классифицируй:
Вход: "Супер, мне очень понравилось"
Выход:"""

print("Few-shot промпт (с примерами):")
print("-" * 40)
print(few_shot_prompt)
print("-" * 40)

answer = llm.generate(few_shot_prompt)
print(f"\nОтвет: {answer}")

print("\n---")

# Few-shot для перевода
few_shot_translate = """Задача: переводи слова с русского на английский.

Пример:
Вход: "привет"
Выход: hello

Пример:
Вход: "спасибо"
Выход: thank you

Пример:
Вход: "пока"
Выход: goodbye

Теперь переведи:
Вход: "привет"
Выход:"""

print("\nFew-shot для перевода:")
print("-" * 40)
print(few_shot_translate)
print("-" * 40)

answer = llm.generate(few_shot_translate)
print(f"\nОтвет: {answer}")

print("\nВывод: Few-shot — модель учится на примерах в промпте.")
print("  ✓ Не нужна дообучение")
print("  ✓ Работает с простыми задачами")
print("  ✗ Ограничен контекстным окном")
print("  ✗ Примеры должны быть релевантными")


# ============================================================
#  Демо 3: Chain-of-Thought промптинг
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Chain-of-Thought (CoT) промптинг")
print("=" * 55)
print()

cot_prompts = [
    "Вычисли квадрат числа 12. Рассуждай пошагово.",
    "Чему равно 25 × 4? Рассуждай пошагово.",
    "Переведи на английский: привет. Рассуждай пошагово.",
]

for prompt in cot_prompts:
    print(f"Промпт: \"{prompt}\"")
    print("-" * 40)
    answer = llm.generate(prompt)
    print(answer)
    print()

# Сравнение: без CoT vs с CoT
print("--- Сравнение: без CoT vs с CoT ---")
print()

question = "Чему равен квадрат числа 25?"

print(f"Вопрос: {question}")
print()

# Без CoT
print("Без CoT (zero-shot):")
answer_direct = llm.generate(question)
print(f"  Ответ: {answer_direct}")
print()

# С CoT
print("С CoT (с рассуждением):")
answer_cot = llm.generate(f"{question} Рассуждай пошагово.")
print(answer_cot)

print("\nВывод: CoT заставляет модель «думать вслух».")
print("  ✓ Улучшает точность на математических задачах")
print("  ✓ Делает рассуждения прозрачными")
print("  ✗ Увеличивает длину ответа")
print("  ✗ Не всегда помогает на простых задачах")


# ============================================================
#  Демо 4: Оптимизация промптов
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Оптимизация промптов")
print("=" * 55)
print()


# --- 4.1: Техники оптимизации ---

print("--- 4.1: Техники оптимизации промптов ---")
print()

optimization_techniques = [
    ("Ролевая промптинг",
     "Ты — эксперт по математике. Объясни простым языком: 2+2=?"),
    ("Структурированный промпт",
     "Задача: вычисли 5²\nФормат: покажи шаги\nОтвет:"),
    ("Ограничение контекста",
     "Отвечай кратко, одно слово. Столица Франции:"),
    ("Инструкция + вопрос",
     "Будь точным. Ответь одним числом: 7 × 7 = ?"),
]

for name, prompt in optimization_techniques:
    print(f"Техника: {name}")
    print(f"Промпт: \"{prompt}\"")
    answer = llm.generate(prompt)
    print(f"Ответ:  {answer}")
    print()


# --- 4.2: Плохой vs хороший промпт ---

print("--- 4.2: Плохой vs хороший промпт ---")
print()

bad_prompts = [
    ("Плохой: неясная задача",
     "Что-то про Францию"),
    ("Плохой: нет формата",
     "Переведи что-нибудь"),
    ("Плохой: слишком абстрактный",
     "Объясни всё про математику"),
]

good_prompts = [
    ("Хороший: конкретный",
     "Какая столица Франции?"),
    ("Хороший: с форматом",
     "Переведи на английский: привет"),
    ("Хороший: с фокусом",
     "Чему равен квадрат числа 12?"),
]

print("Плохие промпты:")
for name, prompt in bad_prompts:
    print(f"  [{name}]")
    print(f"    \"{prompt}\"")
    answer = llm.generate(prompt)
    print(f"    → Ответ: {answer}")
    print()

print("Хорошие промпты:")
for name, prompt in good_prompts:
    print(f"  [{name}]")
    print(f"    \"{prompt}\"")
    answer = llm.generate(prompt)
    print(f"    → Ответ: {answer}")
    print()


# --- 4.3: Шаблоны оптимизации ---

print("--- 4.3: Шаблоны оптимизации промптов ---")
print()

templates = {
    "Стандартный": "Вопрос: {q}\nОтвет:",
    "С ролью": "Ты — эксперт. Вопрос: {q}\nОтвет:",
    "С форматом": "Вопрос: {q}\nФормат: одно слово\nОтвет:",
    "С CoT": "Вопрос: {q}\nРассуждай пошагово.\nОтвет:",
}

question = "Какая столица России?"

print(f"Вопрос: {question}\n")
for name, template in templates.items():
    prompt = template.format(q=question)
    answer = llm.generate(prompt)
    print(f"  [{name}]")
    print(f"    Промпт: {prompt}")
    print(f"    Ответ:  {answer}")
    print()


# --- 4.4: Метрики качества промптов ---

print("--- 4.4: Метрики качества промптов ---")
print()

metrics = {
    "Точность": "% правильных ответов",
    "Релевантность": "соответствие 질문у",
    "Полнота": "наличие всех деталей",
    "Ясность": "отсутствие двусмысленностей",
    "Краткость": "минимальный достаточный объём",
}

for metric, description in metrics.items():
    print(f"  {metric:15s} — {description}")

print()
print("Правила оптимизации:")
print("  1. Будьте конкретны (не «что-нибудь», а «столица Франции»)")
print("  2. Указывайте формат («одно слово», «пошагово»)")
print("  3. Давайте контекст (роль, задача, ограничения)")
print("  4. Используйте разделители (---, ###)")
print("  5. Тестируйте варианты и сравнивайте")


# ============================================================
#  Итоговая сводка
# ============================================================

print("\n" + "=" * 55)
print("ИТОГОВАЯ СВОДКА")
print("=" * 55)

summary = """
┌─────────────────────┬────────────────────────────────────┐
│ Техника             │ Когда использовать                 │
├─────────────────────┼────────────────────────────────────┤
│ Zero-shot           │ Простые задачи, быстрые ответы     │
│ Few-shot            │ Классификация, форматирование      │
│ Chain-of-Thought    │ Математика, логика, рассуждения     │
│ Оптимизация         │ Всегда — для повышения качества    │
└─────────────────────┴────────────────────────────────────┘

Комбинирование техник:
  • Zero-shot + CoT = рассуждение без примеров
  • Few-shot + CoT = примеры + пошаговое рассуждение
  • Role + Few-shot = роль эксперта + примеры

Лучшие практики:
  1. Начинайте с zero-shot, если не помогает — добавьте примеры
  2. Для математики и логики используйте CoT
  3. Всегда указывайте желаемый формат ответа
  4. Тестируйте промпты на разных вариациях
"""
print(summary)
