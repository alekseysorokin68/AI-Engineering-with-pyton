"""
44 — Оптимизаторы для нейросетей с нуля
=========================================
Реализация ключевых оптимизаторов без NumPy/PyTorch:
  1. SGD (+ momentum)
  2. RMSProp
  3. Adam
  4. AdamW (Decoupled Weight Decay)
  5. Сравнение скорости сходимости

Все вычисления на чистом Python + random.
"""

import random

random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# 1. Линейная модель: y = w*x + b  (задача регрессии)
# ──────────────────────────────────────────────────────────────────────

def make_data(n=80, noise=0.3):
    """Генерирует данные y = 2*x + 1 + шум."""
    random.seed(42)
    xs = [random.uniform(-3, 3) for _ in range(n)]
    ys = [2.0 * x + 1.0 + random.gauss(0, noise) for x in xs]
    return xs, ys


def predict(xs, w, b):
    return [w * x + b for x in xs]


def mse_loss(ys_true, ys_pred):
    n = len(ys_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(ys_true, ys_pred)) / n


def grad_mse(xs, ys, w, b):
    """Градиенты MSE по w и b."""
    n = len(xs)
    preds = predict(xs, w, b)
    dw = sum(-2 * x * (y - p) for x, y, p in zip(xs, ys, preds)) / n
    db = sum(-2 * (y - p) for y, p in zip(ys, preds)) / n
    return dw, db


# ──────────────────────────────────────────────────────────────────────
# 2. Классы оптимизаторов
# ──────────────────────────────────────────────────────────────────────

class SGD:
    """Стохастический градиентный спуск (с опциональным momentum)."""

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.vw = 0.0
        self.vb = 0.0

    def step(self, dw, db):
        self.vw = self.momentum * self.vw + dw
        self.vb = self.momentum * self.vb + db
        w_update = -self.lr * self.vw
        b_update = -self.lr * self.vb
        return w_update, b_update

    def name(self):
        if self.momentum > 0:
            return f"SGD+Momentum(γ={self.momentum})"
        return "SGD"


class RMSProp:
    """Root Mean Square Propagation."""

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sw = 0.0
        self.sb = 0.0

    def step(self, dw, db):
        self.sw = self.beta * self.sw + (1 - self.beta) * dw ** 2
        self.sb = self.beta * self.sb + (1 - self.beta) * db ** 2
        w_update = -self.lr * dw / (self.sw ** 0.5 + self.eps)
        b_update = -self.lr * db / (self.sb ** 0.5 + self.eps)
        return w_update, b_update

    def name(self):
        return "RMSProp"


class Adam:
    """Adaptive Moment Estimation."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mw = 0.0
        self.mb = 0.0
        self.vw = 0.0
        self.vb = 0.0
        self.t = 0

    def step(self, dw, db):
        self.t += 1
        # 1st moment (mean)
        self.mw = self.beta1 * self.mw + (1 - self.beta1) * dw
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        # 2nd moment (uncentered variance)
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * dw ** 2
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db ** 2
        # bias correction
        mw_hat = self.mw / (1 - self.beta1 ** self.t)
        mb_hat = self.mb / (1 - self.beta1 ** self.t)
        vw_hat = self.vw / (1 - self.beta2 ** self.t)
        vb_hat = self.vb / (1 - self.beta2 ** self.t)
        # update
        w_update = -self.lr * mw_hat / (vw_hat ** 0.5 + self.eps)
        b_update = -self.lr * mb_hat / (vb_hat ** 0.5 + self.eps)
        return w_update, b_update

    def name(self):
        return "Adam"


class AdamW(Adam):
    """Adam с декаплексированным weight decay (L2-регуляризация отдельно от адаптивного lr)."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.wd = weight_decay

    def step(self, dw, db):
        # L2 weight decay напрямую на весах (не через градиент)
        w_update, b_update = super().step(dw, db)
        w_update -= self.lr * self.wd  # weight decay
        return w_update, b_update

    def name(self):
        return f"AdamW(wd={self.wd})"


# ──────────────────────────────────────────────────────────────────────
# 3. Тренировочный цикл
# ──────────────────────────────────────────────────────────────────────

def train(xs, ys, optimizer, epochs=300, w_init=0.0, b_init=0.0):
    """Возвращает историю loss и финальные (w, b)."""
    w, b = w_init, b_init
    history = []
    for epoch in range(epochs):
        preds = predict(xs, w, b)
        loss = mse_loss(ys, preds)
        history.append(loss)
        dw, db = grad_mse(xs, ys, w, b)
        w_up, b_up = optimizer.step(dw, db)
        w += w_up
        b += b_up
    return history, w, b


def run_comparison(optimizers, epochs=300, title=""):
    """Запускает несколько оптимизаторов и возвращает {name: history}."""
    xs, ys = make_data()
    results = {}
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for opt in optimizers:
        hist, w, b = train(xs, ys, opt, epochs)
        results[opt.name()] = hist
        print(f"  {opt.name():30s}  loss[0]={hist[0]:.4f}  loss[{epochs-1}]={hist[-1]:.6f}  "
              f"w={w:.4f} b={b:.4f}")
    return results


def print_bar_chart(labels, values, width=50):
    """Простая ASCII-гистограмма."""
    max_val = max(values) if values else 1
    for label, val in zip(labels, values):
        bar_len = int(val / max_val * width) if max_val > 0 else 0
        print(f"  {label:30s} | {'█' * bar_len} {val:.6f}")


# ──────────────────────────────────────────────────────────────────────
# 4. Демонстрации
# ──────────────────────────────────────────────────────────────────────

def demo1():
    """Демо 1: SGD vs SGD+Momentum — ускорение сходимости."""
    print("\n" + "=" * 60)
    print("  ДЕМО 1: SGD vs SGD + Momentum")
    print("=" * 60)
    print("  Momentum (γ) накапливает «импульс» градиента:")
    print("  v_t = γ·v_{t-1} + ∇L")
    print("  w   = w - lr · v_t")
    print("  Эффект: ускорение по плоским направлениям, гашение колебаний.\n")

    xs, ys = make_data()
    optimizers = [
        SGD(lr=0.05, momentum=0.0),      # vanilla
        SGD(lr=0.05, momentum=0.5),
        SGD(lr=0.05, momentum=0.9),
        SGD(lr=0.05, momentum=0.99),
    ]
    results = run_comparison(optimizers, epochs=200, title="SGD vs Momentum")

    # Сравнение финальных loss
    print("\n  Финальные loss (200 эпох):")
    final_losses = [results[k][-1] for k in results]
    print_bar_chart(list(results.keys()), final_losses)

    # Время достижения loss < 0.5
    print("\n  Эпоха, на которой loss впервые < 0.5:")
    for name, hist in results.items():
        threshold_epoch = next((i for i, v in enumerate(hist) if v < 0.5), None)
        if threshold_epoch is not None:
            print(f"    {name:30s}  → эпоха {threshold_epoch}")
        else:
            print(f"    {name:30s}  → не достигнуто за 200 эпох")


def demo2():
    """Демо 2: Adam — адаптивные learning rates."""
    print("\n" + "=" * 60)
    print("  ДЕМО 2: Adam — Адаптивные Learning Rates")
    print("=" * 60)
    print("  Adam = Momentum (1-й момент) + RMSProp (2-й момент)")
    print("  m_t = β₁·m_{t-1} + (1-β₁)·∇L       ← средний градиент")
    print("  v_t = β₂·v_{t-1} + (1-β₂)·(∇L)²     ← средний квадрат")
    print("  m̂ = m / (1-β₁ᵗ),  v̂ = v / (1-β₂ᵗ)   ← bias correction")
    print("  w = w - lr · m̂ / (√v̂ + ε)\n")

    xs, ys = make_data()
    optimizers = [
        SGD(lr=0.05, momentum=0.0),
        Adam(lr=0.05),
        Adam(lr=0.1),
        Adam(lr=0.01),
    ]
    results = run_comparison(optimizers, epochs=200, title="SGD vs Adam (разные lr)")

    print("\n  Финальные loss:")
    final_losses = [results[k][-1] for k in results]
    print_bar_chart(list(results.keys()), final_losses)

    # Показываем адаптивные lr для каждого параметра
    print("\n  Адаптивные learning rates — эффективный lr для w и b:")
    opt = Adam(lr=0.1)
    random.seed(42)
    xs_demo, ys_demo = make_data()
    w, b = 0.0, 0.0
    for step_i in range(1, 11):
        dw, db = grad_mse(xs_demo, ys_demo, w, b)
        w_up, b_up = opt.step(dw, db)
        # effective lr = |update / gradient|
        eff_lr_w = abs(w_up / dw) if abs(dw) > 1e-12 else 0
        eff_lr_b = abs(b_up / db) if abs(db) > 1e-12 else 0
        w += w_up
        b += b_up
        if step_i in [1, 3, 5, 10]:
            print(f"    step {step_i:2d}: eff_lr_w={eff_lr_w:.6f}  eff_lr_b={eff_lr_b:.6f}")


def demo3():
    """Демо 3: Сравнение всех оптимизаторов."""
    print("\n" + "=" * 60)
    print("  ДЕМО 3: Сравнение ВСЕХ оптимизаторов")
    print("=" * 60)

    xs, ys = make_data()
    optimizers = [
        SGD(lr=0.05, momentum=0.0),
        SGD(lr=0.05, momentum=0.9),
        RMSProp(lr=0.05),
        Adam(lr=0.05),
        AdamW(lr=0.05, weight_decay=0.01),
    ]
    results = run_comparison(optimizers, epochs=300, title="Все оптимизаторы (lr=0.05)")

    # Таблица: loss каждые 50 эпох
    print("\n  Loss по эпохам:")
    header = f"  {'Оптимизатор':30s}"
    for ep in range(0, 301, 50):
        header += f"  ep{ep:3d}"
    print(header)
    print("  " + "-" * (30 + 7 * 7))
    for name, hist in results.items():
        row = f"  {name:30s}"
        for ep in range(0, 301, 50):
            row += f"  {hist[min(ep, len(hist)-1)]:.4f}"
        print(row)

    # Итоговый ranking
    print("\n  Рейтинг (финальный loss, чем меньше → тем лучше):")
    ranked = sorted(results.items(), key=lambda kv: kv[1][-1])
    for i, (name, hist) in enumerate(ranked, 1):
        print(f"    {i}. {name:30s}  loss = {hist[-1]:.6f}")

    # AdamW: эффект weight decay
    print("\n  AdamW: влияние weight decay:")
    xs2, ys2 = make_data()
    for wd in [0.0, 0.001, 0.01, 0.1]:
        opt = AdamW(lr=0.05, weight_decay=wd)
        hist, w, b = train(xs2, ys2, opt, epochs=300)
        print(f"    wd={wd:<6}  final loss={hist[-1]:.6f}  w={w:.4f}  b={b:.4f}")


def demo4():
    """Демо 4: Влияние learning rate."""
    print("\n" + "=" * 60)
    print("  ДЕМО 4: Влияние Learning Rate")
    print("=" * 60)
    print("  Слишком большой lr → расходимость")
    print("  Слишком маленький lr → медленная сходимость")
    print("  Оптимальный lr    → быстрая и стабильная сходимость\n")

    xs, ys = make_data()
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    print("  Adam с разными learning rates:")
    print(f"  {'lr':>8s}  {'loss[0]':>10s}  {'loss[100]':>10s}  {'loss[299]':>10s}  {'статус':>15s}")
    print("  " + "-" * 65)

    for lr in lrs:
        opt = Adam(lr=lr)
        hist, w, b = train(xs, ys, opt, epochs=300)
        diverged = any(v > 1e6 for v in hist[10:])
        status = "РАСХОДИМОСТЬ" if diverged else "OK"
        print(f"  {lr:>8.3f}  {hist[0]:>10.4f}  {hist[min(99,len(hist)-1)]:>10.4f}  "
              f"{hist[-1]:>10.4f}  {status:>15s}")

    # SGD с разными lr
    print("\n  SGD с разными learning rates:")
    print(f"  {'lr':>8s}  {'loss[0]':>10s}  {'loss[100]':>10s}  {'loss[299]':>10s}  {'статус':>15s}")
    print("  " + "-" * 65)

    for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        opt = SGD(lr=lr)
        hist, w, b = train(xs, ys, opt, epochs=300)
        diverged = any(v > 1e6 for v in hist[10:])
        status = "РАСХОДИМОСТЬ" if diverged else "OK"
        print(f"  {lr:>8.3f}  {hist[0]:>10.4f}  {hist[min(99,len(hist)-1)]:>10.4f}  "
              f"{hist[-1]:>10.4f}  {status:>15s}")

    # Безопасная зона
    print("\n  Безопасная зона lr (Adam):")
    xs3, ys3 = make_data()
    for lr in [0.005, 0.01, 0.02, 0.03, 0.05]:
        opt = Adam(lr=lr)
        hist, w, b = train(xs3, ys3, opt, epochs=300)
        print(f"    lr={lr:.3f}  loss={hist[-1]:.6f}  w={w:.4f}  b={b:.4f}")


def demo_summary():
    """Сводная таблица всех оптимизаторов."""
    print("\n" + "=" * 60)
    print("  СВОДНАЯ ТАБЛИЦА: Все оптимизаторы")
    print("=" * 60)
    print()
    print("  Формула          │ Ключевая идея                    │ Параметры")
    print("  ─────────────────┼──────────────────────────────────┼─────────────────────")
    print("  SGD              │ w = w - lr · ∇L                  │ lr")
    print("  SGD+Momentum     │ v = γv + ∇L; w = w - lr·v       │ lr, γ (momentum)")
    print("  RMSProp          │ v = βv + (1-β)(∇L)²             │ lr, β (decay), ε")
    print("                   │ w = w - lr·∇L/√v                │")
    print("  Adam             │ m = β₁m + (1-β₁)∇L (1-й момент) │ lr, β₁, β₂, ε")
    print("                   │ v = β₂v + (1-β₂)(∇L)² (2-й м.) │")
    print("                   │ bias-corrected: m̂, v̂            │")
    print("  AdamW            │ Adam + отдельный weight decay    │ Adam + λ (wd)")
    print()
    print("  Рекомендации:")
    print("    • SGD+Momentum  — хорош для компьютерного зрения (ResNet и т.д.)")
    print("    • Adam          — хорош для NLP, GAN, рекомендаций")
    print("    • AdamW         — предпочтителен когда нужна L2-регуляризация")
    print("    • lr            — начинайте с 3e-4 (Adam) или 1e-2 (SGD)")
    print()


# ──────────────────────────────────────────────────────────────────────
# 5. Запуск
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ОПТИМИЗАТОРЫ ДЛЯ НЕЙРОСЕТЕЙ — с нуля на Python")
    print("  Модель: y = w·x + b  (линейная регрессия, MSE loss)")
    print("  Все вычисления на чистом Python, random.seed(42)")
    print("=" * 60)

    demo1()
    demo2()
    demo3()
    demo4()
    demo_summary()

    print("\n" + "=" * 60)
    print("  Готово! Все 4 демонстрации выполнены.")
    print("=" * 60)
