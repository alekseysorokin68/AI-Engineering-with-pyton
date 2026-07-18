import math
import random

random.seed(42)


# ============================================================
#  Простая модель для fine-tuning
# ============================================================

class SimpleLinear:
    def __init__(self, in_dim, out_dim):
        self.W = [[random.gauss(0, 0.1) for _ in range(out_dim)] for _ in range(in_dim)]
        self.b = [0.0] * out_dim

    def forward(self, x):
        return [sum(self.W[i][j] * x[i] for i in range(len(x))) + self.b[j]
                for j in range(len(self[0]))]

    def __getitem__(self, idx):
        return self.W


# ============================================================
#  Демо 1: SFT — Supervised Fine-Tuning
# ============================================================

print("=" * 55)
print("ДЕМО 1: SFT — Supervised Fine-Tuning")
print("=" * 55)

# Простая задача: классификация на основе признаков
train_data = [
    ([1.0, 0.5], 0),
    ([0.8, 0.3], 0),
    ([0.2, 0.8], 1),
    ([0.1, 0.9], 1),
]

print("\nДатасет для fine-tuning:")
for x, y in train_data:
    print(f"  {x} → класс {y}")

print("\nSFT: обучаем модель на размеченных данных")
print("  Формат: input → label")
print("  Loss: cross-entropy между предсказанием и меткой")


# ============================================================
#  Демо 2: Формат инструкций
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Формат инструкций")
print("=" * 55)

instructions = [
    {"instruction": "Переведи на английский", "input": "Привет мир", "output": "Hello world"},
    {"instruction": "Суммируй текст", "input": "Длинный текст...", "output": "Краткое резюме"},
    {"instruction": "Классифицируй", "input": "Это спам", "output": "spam"},
]

print("\nФормат инструкций:")
for inst in instructions:
    print(f"\n  System: {inst['instruction']}")
    print(f"  User: {inst['input']}")
    print(f"  Assistant: {inst['output']}")


# ============================================================
#  Демо 3: Training loop для SFT
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Training loop для SFT")
print("=" * 55)

def cross_entropy_loss(pred, target):
    pred = max(1e-7, min(1 - 1e-7, pred))
    return -(target * math.log(pred) + (1 - target) * math.log(1 - pred))

# Простая модель: sigmoid(w*x + b)
w, b = 0.0, 0.0
lr = 0.1

print("\nОбучение на 4 примерах:")
for epoch in range(100):
    total_loss = 0
    for x_val, y_val in train_data:
        z = w * x_val[0] + b * x_val[1]
        pred = 1 / (1 + math.exp(-z))
        loss = cross_entropy_loss(pred, y_val)
        total_loss += loss

        dw = (pred - y_val) * x_val[0]
        db = (pred - y_val) * x_val[1]
        w -= lr * dw
        b -= lr * db

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss={total_loss/len(train_data):.4f}, w={w:.4f}, b={b:.4f}")


# ============================================================
#  Демо 4: До и после fine-tuning
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: До и после fine-tuning")
print("=" * 55)

# До fine-tuning (случайные веса)
w_before = 0.5
b_before = 0.5

print("\nДо fine-tuning:")
for x_val, y_val in train_data:
    z = w_before * x_val[0] + b_before * x_val[1]
    pred = 1 / (1 + math.exp(-z))
    print(f"  {x_val} → pred={pred:.4f}, true={y_val}")

print(f"\nПосле fine-tuning (100 эпох):")
for x_val, y_val in train_data:
    z = w * x_val[0] + b * x_val[1]
    pred = 1 / (1 + math.exp(-z))
    print(f"  {x_val} → pred={pred:.4f}, true={y_val}")

print(f"\nВеса: w={w:.4f}, b={b:.4f}")
