"""
Transfer Learning — основы и сравнение подходов.
Демонстрация на упрощённых моделях без внешних зависимостей.
"""

import random
import math
import time

random.seed(42)

# ============================================================
# Мини-фреймворк для имитации нейросетей
# ============================================================

def relu(x):
    """ReLU активация"""
    return max(0.0, x)


def relu_deriv(x):
    """Производная ReLU"""
    return 1.0 if x > 0 else 0.0


class LinearLayer:
    """Полносвязный слой с Xavier-инициализацией"""
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        # Xavier инициализация
        limit = math.sqrt(6.0 / (in_size + out_size))
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(out_size)]
            for _ in range(in_size)
        ]
        self.biases = [0.0] * out_size
        self.frozen = False

    def forward(self, x):
        """Прямой проход: x (вход) -> выход слоя"""
        output = []
        for j in range(self.out_size):
            s = self.biases[j]
            for i in range(self.in_size):
                s += x[i] * self.weights[i][j]
            output.append(s)
        return output

    def get_param_count(self):
        return self.in_size * self.out_size + self.out_size

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False


class SimpleNetwork:
    """Простая нейросеть из последовательности слоёв"""
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        """Прямой проход через все слои с ReLU"""
        for layer in self.layers:
            x = layer.forward(x)
            # Применяем ReLU ко всем слоям кроме последнего
            if layer != self.layers[-1]:
                x = [relu(v) for v in x]
        return x

    def total_params(self):
        return sum(l.get_param_count() for l in self.layers)

    def frozen_params(self):
        return sum(l.get_param_count() for l in self.layers if l.frozen)

    def trainable_params(self):
        return self.total_params() - self.frozen_params()

    def freeze_all(self):
        for l in self.layers:
            l.freeze()

    def unfreeze_last(self, n=1):
        """Разморозить последние n слоёв"""
        for l in self.layers[-n:]:
            l.unfreeze()


class PretrainedBackbone(SimpleNetwork):
    """Имитация предобученной модели (backbone)"""
    def __init__(self):
        # Backbone: вход 784 (28x28) -> 512 -> 256 -> 128
        super().__init__([784, 512, 256, 128])


class ClassifierHead:
    """Классификационная голова (новый слой)"""
    def __init__(self, in_size, out_size):
        self.layer = LinearLayer(in_size, out_size)

    def forward(self, x):
        return self.layer.forward(x)


# ============================================================
# Генерация «данных» (симуляция признаков изображений)
# ============================================================

def generate_synthetic_data(n_samples=200, feature_dim=784, n_classes=10):
    """Генерация синтетических данных, имитирующих MNIST-подобные признаки"""
    data = []
    for _ in range(n_samples):
        label = random.randint(0, n_classes - 1)
        # Каждый класс имеет «центр» — свой характерный шумовой паттерн
        features = []
        for i in range(feature_dim):
            base = math.sin(label * 0.7 + i * 0.01) * 0.5
            features.append(base + random.gauss(0, 0.3))
        data.append((features, label))
    return data


def simulate_training_time(params, frozen_pct, epochs):
    """
    Симуляция времени обучения (упрощённая модель).
    Учитывает: количество параметров, долю замороженных, эпохи.
    """
    trainable = params * (1 - frozen_pct)
    # Время пропорционально trainable параметрам × эпохи
    base_time = trainable * epochs * 0.0000001
    # Overhead для backward pass
    overhead = trainable * epochs * 0.00000005
    return base_time + overhead + random.uniform(0.01, 0.05)


def accuracy_approximation(n_classes, method, frozen_pct):
    """
    Симуляция точности в зависимости от метода transfer learning.
    Основана на реальных трендах из литературы.
    """
    random.seed(42 + hash(method) % 1000)
    if method == "feature_extraction":
        # Feature extraction: хорошо при малом объёме данных
        base = 0.82 + frozen_pct * 0.05
    elif method == "fine_tuning":
        # Fine-tuning: лучше при достаточных данных
        base = 0.85 + frozen_pct * 0.03
    elif method == "gradual_unfreezing":
        # Posterior unfreezing: компромисс
        base = 0.84 + frozen_pct * 0.04
    else:
        base = 0.70

    noise = random.gauss(0, 0.008)
    return min(0.99, max(0.5, base + noise))


# ============================================================
# ДЕМО 1: Feature Extraction — использование предобученных признаков
# ============================================================

def demo_feature_extraction():
    print("=" * 65)
    print("ДЕМО 1: Feature Extraction — предобученные признаки")
    print("=" * 65)

    backbone = PretrainedBackbone()

    print(f"\n[Backbone] Архитектура: 784 -> 512 -> 256 -> 128")
    print(f"[Backbone] Всего параметров: {backbone.total_params():,}")
    print(f"[Backbone] Все слои заморожены: да (предобучены на ImageNet)")

    # Замораживаем backbone
    backbone.freeze_all()

    # Создаём голову классификации
    classifier = ClassifierHead(in_size=128, out_size=10)
    print(f"\n[Head] Новый классификатор: 128 -> 10")
    print(f"[Head] Параметров головы: {classifier.layer.get_param_count():,}")

    # Симуляция forward pass
    print(f"\n--- Forward Pass ---")
    sample_data = generate_synthetic_data(n_samples=5, feature_dim=784)

    for i, (features, label) in enumerate(sample_data[:3]):
        # Backbone (без градиентов — только inference)
        backbone_out = backbone.forward(features)
        # Классификатор
        logits = classifier.forward(backbone_out)
        pred = max(range(len(logits)), key=lambda j: logits[j])
        status = "✓" if pred == label else "✗"
        print(f"  Пример {i+1}: backbone_output[:3]={[f'{v:.3f}' for v in backbone_out[:3]]}, "
              f"label={label}, pred={pred} {status}")

    # Подсчёт
    frozen_pct = 1.0
    print(f"\n--- Итоги Feature Extraction ---")
    print(f"  Замороженных параметров: {backbone.frozen_params():,} ({frozen_pct*100:.0f}%)")
    print(f"  Обучаемых параметров:    {classifier.layer.get_param_count():,}")
    print(f"  Преимущества: быстрое обучение, не портит предобученные веса")
    print(f"  Ограничения: качество ограничено качеством backbone")
    print()


# ============================================================
# ДЕМО 2: Fine-tuning — дообучение последних слоёв
# ============================================================

def demo_fine_tuning():
    print("=" * 65)
    print("ДЕМО 2: Fine-tuning — дообучение последних слоёв")
    print("=" * 65)

    # Стратегии fine-tuning
    strategies = [
        ("Только голова", 0, "Заморожен backbone, обучаем только классификатор"),
        ("Последний слой backbone + голова", 1,
         "Размораживаем последний слой backbone"),
        ("Последние 2 слоя backbone + голова", 2,
         "Размораживаем последние 2 слоя backbone"),
        ("Все слои (full fine-tune)", 3,
         "Обучаем всю сеть с малым learning rate"),
    ]

    print("\nСтратегии fine-tuning:\n")
    for name, unfreeze_n, desc in strategies:
        backbone = PretrainedBackbone()
        backbone.freeze_all()
        backbone.unfreeze_last(unfreeze_n)
        classifier = ClassifierHead(128, 10)

        total = backbone.total_params() + classifier.layer.get_param_count()
        trainable = backbone.trainable_params() + classifier.layer.get_param_count()
        pct = trainable / total * 100

        print(f"  [{name}]")
        print(f"    Описание: {desc}")
        print(f"    Всего параметров:      {total:,}")
        print(f"    Обучаемых параметров:  {trainable:,} ({pct:.1f}%)")
        print()

    # Демонстрация поэтапного размораживания
    print("--- Поэтапное размораживание (Gradual Unfreezing) ---\n")
    backbone = PretrainedBackbone()
    backbone.freeze_all()

    stages = [
        (0, "Шаг 1: Обучаем только голову классификации"),
        (1, "Шаг 2: Размораживаем последний слой (256->128)"),
        (2, "Шаг 3: Размораживаем предпоследний слой (512->256)"),
        (3, "Шаг 4: Полный fine-tune (все слои)"),
    ]

    for unfreeze_n, stage_desc in stages:
        if unfreeze_n == 0:
            backbone.freeze_all()
        else:
            backbone.unfreeze_last(unfreeze_n)
        classifier = ClassifierHead(128, 10)
        trainable = backbone.trainable_params() + classifier.layer.get_param_count()
        print(f"  {stage_desc}")
        print(f"    Обучаемых параметров: {trainable:,}")
    print()


# ============================================================
# ДЕМО 3: Сравнение времени обучения
# ============================================================

def demo_training_time_comparison():
    print("=" * 65)
    print("ДЕМО 3: Сравнение времени обучения")
    print("=" * 65)

    backbone = PretrainedBackbone()
    total_params = backbone.total_params() + ClassifierHead(128, 10).layer.get_param_count()

    methods = [
        ("Обучение с нуля", 0.0, 50),
        ("Feature Extraction", 1.0, 30),
        ("Fine-tune: голова", 0.92, 30),
        ("Fine-tune: +1 слой", 0.78, 30),
        ("Fine-tune: +2 слоя", 0.60, 30),
        ("Full Fine-tune", 0.0, 25),
    ]

    print(f"\nОбщее число параметров модели: {total_params:,}")
    print(f"Эпохи: указаны для каждого метода\n")
    print(f"{'Метод':<25} {'Заморозка':>10} {'Эпохи':>7} {'≈Время (с)':>12}")
    print("-" * 60)

    base_time = None
    for name, frozen_pct, epochs in methods:
        t = simulate_training_time(total_params, frozen_pct, epochs)
        if name == "Обучение с нуля":
            base_time = t
        speedup = base_time / t if base_time and t > 0 else 1
        print(f"{name:<25} {frozen_pct*100:>8.0f}% {epochs:>7} {t:>10.4f}с  "
              f"(×{speedup:.1f})")

    print(f"\n--- Выводы ---")
    print(f"  • Feature Extraction — самый быстрый (только голова обучается)")
    print(f"  • Чем больше слоёв разморожено, тем дольше обучение")
    print(f"  • Полный fine-tune: лучшее качество, но максимальное время")
    print(f"  • Стратегия: начинай с Feature Extraction, затем постепенно")
    print(f"    размораживай слои, если качество недостаточно")
    print()


# ============================================================
# ДЕМО 4: Когда Transfer Learning помогает
# ============================================================

def demo_when_transfer_learning_helps():
    print("=" * 65)
    print("ДЕМО 4: Когда Transfer Learning помогает")
    print("=" * 65)

    scenarios = [
        {
            "name": "Мало данных + похожая задача",
            "data_size": "100 изображений",
            "recommendation": "Feature Extraction (заморозить backbone)",
            "reason": "Мало данных — fine-tuning переобучит backbone. "
                      "Предобученные признаки универсальны между задачами.",
            "expected_acc": "0.82-0.88",
        },
        {
            "name": "Мало данных + отличающаяся задача",
            "data_size": "500 изображений",
            "recommendation": "Feature Extraction + простой классификатор",
            "reason": "Задачи отличаются — предобученные признаки могут не подходить. "
                      "Но всё лучше, чем обучение с нуля.",
            "expected_acc": "0.65-0.75",
        },
        {
            "name": "Много данных + похожая задача",
            "data_size": "50,000 изображений",
            "recommendation": "Fine-tuning последних 2-3 слоёв",
            "reason": "Достаточно данных для дообучения. "
                      "Начальная инициализация ускоряет сходимость.",
            "expected_acc": "0.90-0.95",
        },
        {
            "name": "Много данных + отличающаяся задача",
            "data_size": "100,000 изображений",
            "recommendation": "Fine-tuning с малым LR или обучение с нуля",
            "reason": "Много данных + сложная задача = полный fine-tune "
                      "или полное обучение. Предобучение даёт лишь стартовую точку.",
            "expected_acc": "0.92-0.97",
        },
    ]

    print("\nСценарии и рекомендации:\n")
    for i, s in enumerate(scenarios, 1):
        print(f"  {i}. {s['name']}")
        print(f"     Данные: {s['data_size']}")
        print(f"     Рекомендация: {s['recommendation']}")
        print(f"     Обоснование: {s['reason']}")
        print(f"     Ожидаемая точность: {s['expected_acc']}")
        print()

    # Сравнение: Transfer Learning vs обучение с нуля
    print("--- Сравнение: Transfer Learning vs обучение с нуля ---\n")

    sizes = [50, 100, 500, 1000, 5000, 50000]
    print(f"{'Размер данных':>15} | {'С нуля':>12} | {'Transfer':>12} | {'Разница':>10}")
    print("-" * 55)

    for n in sizes:
        # Модель: accuracy растёт с размером данных
        random.seed(42 + n)
        from_scratch = 0.3 + 0.5 * (1 - math.exp(-n / 3000)) + random.gauss(0, 0.01)
        transfer = 0.6 + 0.35 * (1 - math.exp(-n / 2000)) + random.gauss(0, 0.01)
        diff = transfer - from_scratch
        marker = " ← transfer лучше" if diff > 0.05 else ""
        print(f"{n:>15} | {from_scratch:>11.1%} | {transfer:>11.1%} | "
              f"{diff:>+9.1%}{marker}")

    print(f"\n--- Ключевые принципы ---")
    print(f"  1. Transfer learning эффективен когда исходная и целевая задачи")
    print(f"     используют низкоуровневые признаки (кромы, текстуры)")
    print(f"  2. Чем меньше данных — тем больше слоёв стоит заморозить")
    print(f"  3. Чем больше данных — тем больше можно дообучить")
    print(f"  4. Learning rate для fine-tuning должен быть в 10-100× меньше")
    print(f"     чем для обучения с нуля (чтобы не разрушить предобученные веса)")
    print(f"  5. Domain adaptation: при большом расхождении доменов")
    print(f"     предобучение может навредить — в этом случае обучение с нуля лучше")
    print()


# ============================================================
# Запуск всех демо
# ============================================================

if __name__ == "__main__":
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║        TRANSFER LEARNING — ОСНОВЫ И СРАВНЕНИЕ ПОДХОДОВ         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    demo_feature_extraction()
    demo_fine_tuning()
    demo_training_time_comparison()
    demo_when_transfer_learning_helps()

    print("=" * 65)
    print("Резюме: Transfer Learning")
    print("=" * 65)
    print("""
  Transfer Learning — техника, позволяющая использовать знания,
  полученные на одной задаче, для решения другой.

  Три основных подхода:
  ┌─────────────────────┬────────────────────┬──────────────────────┐
  │ Подход              │ Что обучаем        │ Когда использовать   │
  ├─────────────────────┼────────────────────┼──────────────────────┤
  │ Feature Extraction  │ Только голову      │ Мало данных,         │
  │                     │                    │ быстрый прототип     │
  ├─────────────────────┼────────────────────┼──────────────────────┤
  │ Fine-tuning         │ Голову + последние │ Средний объём данных,│
  │                     │ слои backbone      │ нужна точность       │
  ├─────────────────────┼────────────────────┼──────────────────────┤
  │ Full Fine-tune      │ Всю сеть           │ Много данных,        │
  │                     │                    │ максимальная точность│
  └─────────────────────┴────────────────────┴──────────────────────┘

  Эмпирическое правило:
    • < 1000 изображений → Feature Extraction
    • 1000-10000 → Fine-tuning последних слоёв
    • > 10000 → Full Fine-tuning
""")
