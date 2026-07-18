"""
43 — Функции потерь (Loss Functions) с нуля на Python
=====================================================
Без numpy / sklearn / torch. Только random и math.

Содержание:
  1. MSE (Mean Squared Error)
  2. Cross-Entropy (binary + categorical)
  3. Huber Loss
  4. Contrastive Loss
  5. Сравнительная таблица свойств
  Демо 1–4
"""

import math
import random

random.seed(42)

EPS = 1e-12  # защита от log(0)

# ──────────────────────────────────────────────
# 1. MSE — Mean Squared Error
# ──────────────────────────────────────────────

def mse(y_true, y_pred):
    """Среднеквадратичная ошибка."""
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def mse_grad(y_true, y_pred):
    """Градиент MSE по y_pred: 2*(yp - yt)/n."""
    n = len(y_true)
    return [2.0 * (yp - yt) / n for yt, yp in zip(y_true, y_pred)]


# MAE (для сравнения)
def mae(y_true, y_pred):
    """Средняя абсолютная ошибка."""
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def mae_grad(y_true, y_pred):
    """Градиент MAE по y_pred: sign(yp - yt)/n."""
    n = len(y_true)
    return [math.copysign(1.0, yp - yt) / n for yt, yp in zip(y_true, y_pred)]


# ──────────────────────────────────────────────
# 2. Cross-Entropy Loss
# ──────────────────────────────────────────────

def binary_cross_entropy(y_true, y_prob):
    """
    Binary Cross-Entropy.
    y_true — список из 0/1
    y_prob — список вероятностей [0, 1]
    L = -1/N Σ [ y*log(p) + (1-y)*log(1-p) ]
    """
    n = len(y_true)
    loss = 0.0
    for yt, yp in zip(y_true, y_prob):
        p = max(EPS, min(1.0 - EPS, yp))  # clamp
        loss += -(yt * math.log(p) + (1.0 - yt) * math.log(1.0 - p))
    return loss / n


def categorical_cross_entropy(y_true_idx, y_probs):
    """
    Categorical Cross-Entropy (hard labels).
    y_true_idx — список целевых классов (int)
    y_probs    — список списков вероятностей, sum每行 ≈ 1
    L = -1/N Σ log( p[ correct_class ] )
    """
    n = len(y_true_idx)
    loss = 0.0
    for cls, probs in zip(y_true_idx, y_probs):
        p = max(EPS, probs[cls])
        loss += -math.log(p)
    return loss / n


# ──────────────────────────────────────────────
# 3. Huber Loss (smooth L1)
# ──────────────────────────────────────────────

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss: квадратичная при малых ошибках, линейная при больших.
    L_δ(a) = 0.5*a²           если |a| ≤ δ
             δ*(|a| - 0.5*δ)   иначе
    """
    n = len(y_true)
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        a = abs(yt - yp)
        if a <= delta:
            total += 0.5 * a * a
        else:
            total += delta * (a - 0.5 * delta)
    return total / n


def huber_grad(y_true, y_pred, delta=1.0):
    """Градиент Huber Loss по y_pred."""
    n = len(y_true)
    grads = []
    for yt, yp in zip(y_true, y_pred):
        diff = yp - yt
        a = abs(diff)
        if a <= delta:
            grads.append(diff)
        else:
            grads.append(delta * math.copysign(1.0, diff))
    return [g / n for g in grads]


# ──────────────────────────────────────────────
# 4. Contrastive Loss
# ──────────────────────────────────────────────

def contrastive_loss(y_true, dist, margin=1.0):
    """
    Contrastive Loss для пар примеров.
    y_true — 1 (похожи) или 0 (не похожи)
    dist   — расстояние между парами
    L = y * d² + (1-y) * max(0, margin - d)²
    """
    n = len(y_true)
    total = 0.0
    for yt, d in zip(y_true, dist):
        if yt == 1:
            total += d * d
        else:
            val = max(0.0, margin - d)
            total += val * val
    return total / n


# ──────────────────────────────────────────────
# 5. Сравнительная таблица свойств
# ──────────────────────────────────────────────

def print_comparison_table():
    print("=" * 78)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА СВОЙСТВ ФУНКЦИЙ ПОТЕРЬ")
    print("=" * 78)
    header = (
        f"{'Функция':<25} {'Чувств.к':<12} {'Robust':<10} "
        f"{'Гладкая':<10} {'Диапазон':<12} {'Применение'}"
    )
    print(header)
    print("-" * 78)

    rows = [
        ("MSE",               "высокая",   "нет",  "да",  "[0, +∞)",   "регрессия"),
        ("MAE",               "одинаковая","да",   "нет", "[0, +∞)",   "регрессия (выбросы)"),
        ("Huber (δ=1)",       "смешанная", "да",   "да",  "[0, +∞)",   "регрессия (компромисс)"),
        ("Binary CE",         "высокая",   "нет",  "да",  "[0, +∞)",   "бинарная классиф."),
        ("Categorical CE",    "высокая",   "нет",  "да",  "[0, +∞)",   "многокласс. классиф."),
        ("Contrastive",       "смешанная", "частично","да","[0, +∞)",  "сиамские сети"),
    ]
    for name, sens, robust, smooth, rng, use in rows:
        print(f"{name:<25} {sens:<12} {robust:<10} {smooth:<10} {rng:<12} {use}")
    print()


# ──────────────────────────────────────────────
# ДЕМО 1: MSE vs MAE
# ──────────────────────────────────────────────

def demo1_mse_vs_mae():
    print("=" * 78)
    print("ДЕМО 1: MSE vs MAE — как разные потери реагируют на выбросы")
    print("=" * 78)

    y_true = [3.0, 5.0, 2.5, 7.0, 4.0]
    y_pred_normal = [3.1, 4.8, 2.6, 6.9, 4.1]  # хорошие предсказания
    y_pred_outlier = [3.1, 4.8, 2.6, 15.0, 4.1]  # один выброс: 15 вместо 7

    print(f"\nИстинные значения:     {y_true}")
    print(f"Хорошие предсказания: {y_pred_normal}")
    print(f"С выбросом (7→15):    {y_pred_outlier}")

    print("\n--- Без выброса ---")
    print(f"  MSE = {mse(y_true, y_pred_normal):.4f}")
    print(f"  MAE = {mae(y_true, y_pred_normal):.4f}")

    print("\n--- С выбросом ---")
    print(f"  MSE = {mse(y_true, y_pred_outlier):.4f}  (увеличение: "
          f"{mse(y_true, y_pred_outlier)/mse(y_true, y_pred_normal):.1f}x)")
    print(f"  MAE = {mae(y_true, y_pred_outlier):.4f}  (увеличение: "
          f"{mae(y_true, y_pred_outlier)/mae(y_true, y_pred_normal):.1f}x)")

    print("\n→ MSE сильно наказывает за выброс (квадратичная штрафность).")
    print("→ MAE одинаково реагирует на все ошибки (линейная штрафность).")

    # Градиенты
    print("\n--- Градиенты при y_true=7.0, разные y_pred ---")
    for yp in [7.0, 7.5, 10.0, 15.0]:
        g_mse = 2.0 * (yp - 7.0)
        g_mae = math.copysign(1.0, yp - 7.0)
        print(f"  y_pred={yp:>5.1f}  ∂MSE/∂yp = {g_mse:>6.1f}   ∂MAE/∂yp = {g_mae:>5.1f}")

    print("→ Градиент MSE растёт пропорционально ошибке — может вызвать"
          " взрыв градиентов при больших отклонениях.\n")


# ──────────────────────────────────────────────
# ДЕМО 2: Binary vs Categorical Cross-Entropy
# ──────────────────────────────────────────────

def demo2_cross_entropy():
    print("=" * 78)
    print("ДЕМО 2: Binary vs Categorical Cross-Entropy")
    print("=" * 78)

    # --- Binary ---
    print("\n--- Binary Cross-Entropy ---")
    y_true_b = [1, 0, 1, 1, 0]
    y_prob_b = [0.9, 0.1, 0.8, 0.3, 0.95]

    print(f"y_true: {y_true_b}")
    print(f"y_prob: {y_prob_b}")
    for yt, yp in zip(y_true_b, y_prob_b):
        p = max(EPS, min(1.0 - EPS, yp))
        loss = -(yt * math.log(p) + (1.0 - yt) * math.log(1.0 - p))
        print(f"  y={yt}, p={yp:.2f}  →  L = -[{yt}·log({p:.4f}) + {1-yt}·log({1-p:.4f})]"
              f" = {loss:.4f}")

    print(f"\n  Средняя BCE = {binary_cross_entropy(y_true_b, y_prob_b):.4f}")

    # Идеальное предсказание
    y_perf = [1.0, 0.0, 1.0, 1.0, 0.0]
    print(f"  Идеальное предсказание → BCE = {binary_cross_entropy(y_true_b, y_perf):.4f}")
    print("  → Чем ближе вероятность к истине, тем меньше потеря.")

    # Худшее предсказание
    y_worst = [0.05, 0.95, 0.05, 0.05, 0.95]
    print(f"  Худшее предсказание   → BCE = {binary_cross_entropy(y_true_b, y_worst):.4f}")

    # --- Categorical ---
    print("\n--- Categorical Cross-Entropy ---")
    classes = ["кот", "собака", "птица"]
    y_true_c = [0, 2, 1, 0]           # индексы классов
    y_probs_c = [
        [0.7, 0.2, 0.1],  # предсказали кот (верно)
        [0.1, 0.1, 0.8],  # предсказали птицу (верно)
        [0.2, 0.5, 0.3],  # предсказали собаку (верно)
        [0.4, 0.4, 0.2],  # предсказали кот (верно, но неуверенно)
    ]

    print(f"\nКлассы: {classes}")
    print(f"y_true_idx: {y_true_c}")
    print("Предсказания (вероятности по классам):")
    for i, (cls, probs) in enumerate(zip(y_true_c, y_probs_c)):
        p = max(EPS, probs[cls])
        loss_i = -math.log(p)
        print(f"  sample {i}: true={classes[cls]:<8} probs={[f'{x:.2f}' for x in probs]}"
              f"  → -log({p:.4f}) = {loss_i:.4f}")

    print(f"\n  Средняя CCE = {categorical_cross_entropy(y_true_c, y_probs_c):.4f}")

    # Неправильные предсказания
    y_wrong_c = [2, 0, 1, 2]
    print(f"\n--- Неправильные предсказания ---")
    print(f"y_true:      {y_true_c}")
    print(f"y_pred_idx:  {y_wrong_c}")
    print(f"  CCE = {categorical_cross_entropy(y_wrong_c, y_probs_c):.4f}  "
          "(значительно выше)\n")


# ──────────────────────────────────────────────
# ДЕМО 3: Huber Loss — компромисс MSE и MAE
# ──────────────────────────────────────────────

def demo3_huber():
    print("=" * 78)
    print("ДЕМО 3: Huber Loss — компромисс MSE и MAE")
    print("=" * 78)

    y_true = [3.0, 5.0, 2.5, 7.0]
    y_pred = [3.1, 4.8, 2.6, 15.0]  # выброс в последнем элементе

    print(f"\ny_true: {y_true}")
    errors = [round(yp - yt, 1) for yt, yp in zip(y_true, y_pred)]
    print(f"y_pred: {y_pred}  (ошибки: {errors})")

    print("\n--- Сравнение при разных delta ---")
    print(f"{'delta':<10} {'MSE':<12} {'Huber':<12} {'MAE':<12}")
    print("-" * 46)

    mse_val = mse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    print(f"{'---':<10} {mse_val:<12.4f} {'---':<12} {mae_val:<12.4f}")

    for delta in [0.5, 1.0, 2.0, 5.0]:
        h = huber_loss(y_true, y_pred, delta=delta)
        print(f"{delta:<10.1f} {'---':<12} {h:<12.4f} {'---':<12}")

    print("\n--- Поведение Huber при разных ошибках (delta=1.0) ---")
    print(f"{'Ошибка |a|':<14} {'MSE':<10} {'Huber':<10} {'MAE':<10} {'Тип Huber'}")
    print("-" * 64)
    for a in [0.1, 0.5, 1.0, 2.0, 5.0]:
        mse_v = a * a
        h_v = 0.5 * a * a if a <= 1.0 else 1.0 * (a - 0.5)
        mae_v = a
        kind = "квадратичная" if a <= 1.0 else "линейная"
        print(f"{a:<14.1f} {mse_v:<10.2f} {h_v:<10.2f} {mae_v:<10.2f} {kind}")

    print("\n--- Градиенты Huber ---")
    print("При |error| ≤ δ: градиент линейный (как MSE)")
    print("При |error| > δ: градиент постоянный (как MAE)")
    print()

    y_t2 = [5.0]
    y_p2 = [5.0]
    print(f"{'y_pred':<10} {'error':<10} {'∂MSE':<10} {'∂Huber':<10} {'∂MAE':<10}")
    print("-" * 50)
    for offset in [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]:
        yp = 5.0 + offset
        err = yp - 5.0
        g_mse = 2.0 * err
        a = abs(err)
        if a <= 1.0:
            g_huber = err
        else:
            g_huber = math.copysign(1.0, err)
        g_mae = math.copysign(1.0, err)
        print(f"{yp:<10.1f} {err:<10.1f} {g_mse:<10.1f} {g_huber:<10.1f} {g_mae:<10.1f}")

    print("\n→ Huber = 'умный компромисс':")
    print("  • Маленькие ошибки → квадратичный штраф (точность)")
    print("  • Большие ошибки   → линейный штраф   (робастность)\n")


# ──────────────────────────────────────────────
# ДЕМО 4: Как loss влияет на обучение
# ──────────────────────────────────────────────

def demo4_loss_impact():
    print("=" * 78)
    print("ДЕМО 4: Как выбор loss влияет на процесс обучения (градиентный спуск)")
    print("=" * 78)

    # Простая задача: предсказать y = 2x + 1
    random.seed(42)
    n_samples = 20
    data = [(x, 2.0 * x + 1.0 + random.gauss(0, 0.3)) for x in range(n_samples)]
    # Добавим выброс
    data.append((10, 35.0))  # вместо 21.0

    # Линейная модель: y = w*x + b
    def train_loop(loss_type, lr=0.001, epochs=200, delta=1.0):
        w, b = 0.0, 0.0
        losses = []
        for epoch in range(epochs):
            # Forward
            preds = [w * x + b for x, _ in data]
            y_true = [y for _, y in data]

            # Loss
            if loss_type == "mse":
                L = mse(y_true, preds)
            elif loss_type == "mae":
                L = mae(y_true, preds)
            else:
                L = huber_loss(y_true, preds, delta)

            losses.append(L)

            # Backward — градиенты по w и b
            n = len(data)
            dw, db = 0.0, 0.0
            for (x, yt), yp in zip(data, preds):
                diff = yp - yt
                if loss_type == "mse":
                    dw += 2 * diff * x
                    db += 2 * diff
                elif loss_type == "mae":
                    s = math.copysign(1.0, diff) if diff != 0 else 0
                    dw += s * x
                    db += s
                else:  # huber
                    a = abs(diff)
                    if a <= delta:
                        dw += diff * x
                        db += diff
                    else:
                        dw += delta * math.copysign(1.0, diff) * x
                        db += delta * math.copysign(1.0, diff)
            dw /= n
            db /= n

            w -= lr * dw
            b -= lr * db

        return w, b, losses

    print("\nЗадача: y = 2x + 1 + шум + выброс (x=10, y=35)")
    print("Модель: y = w*x + b, обучение градиентным спуском\n")

    results = {}
    for loss_type in ["mse", "huber", "mae"]:
        w, b, losses = train_loop(loss_type)
        results[loss_type] = (w, b, losses)

    print(f"{'Loss':<12} {'w (→2.0)':<12} {'b (→1.0)':<12} {'Финальный loss':<16} {'Предсказ. x=10'}")
    print("-" * 68)
    for lt in ["mse", "huber", "mae"]:
        w, b, ls = results[lt]
        pred_10 = w * 10 + b
        true_10 = 21.0
        print(f"{lt:<12} {w:<12.4f} {b:<12.4f} {ls[-1]:<16.4f} {pred_10:<14.1f} (true=21)")

    print("\n--- Эволюция loss по эпохам (каждые 50) ---")
    print(f"{'Эпоха':<10}", end="")
    for lt in ["mse", "huber", "mae"]:
        print(f" {lt:<14}", end="")
    print()
    print("-" * 52)
    for epoch in range(0, 200, 50):
        print(f"{epoch:<10}", end="")
        for lt in ["mse", "huber", "mae"]:
            _, _, ls = results[lt]
            print(f" {ls[epoch]:<14.4f}", end="")
        print()

    print("\n--- Анализ влияния выброса ---")
    w_mse, _, _ = results["mse"]
    w_huber, _, _ = results["huber"]
    w_mae, _, _ = results["mae"]

    print(f"  Истинный w = 2.0000")
    print(f"  MSE:    w = {w_mse:.4f}  (отклонение: {abs(w_mse-2):.4f}) — тянется к выбросу")
    print(f"  Huber:  w = {w_huber:.4f}  (отклонение: {abs(w_huber-2):.4f}) — баланс")
    print(f"  MAE:    w = {w_mae:.4f}  (отклонение: {abs(w_mae-2):.4f}) — игнорирует выброс")

    print("\n→ ВЫВОДЫ:")
    print("  1. MSE чувствителен к выбросам — модель 'подстраивается' под аномалии.")
    print("  2. MAE робастна — выброс почти не влияет на модель.")
    print("  3. Huber — компромисс: устойчив к выбросам, но сохраняет чувствительность.")
    print("  4. Выбор loss — это выбор баланса между точностью и устойчивостью.\n")


# ══════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         ФУНКЦИИ ПОТЕРЬ (LOSS FUNCTIONS) — ОСНОВЫ С НУЛЯ          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    print_comparison_table()
    demo1_mse_vs_mae()
    demo2_cross_entropy()
    demo3_huber()
    demo4_loss_impact()

    print("=" * 78)
    print("КРАТКАЯ СВОДКА")
    print("=" * 78)
    print("""
  Функция потерь определяет, ЧТО именно модель минимизирует.

  ┌──────────────────┬────────────────────────────────────────────────┐
  │ Задача           │ Типичный loss                                  │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ Регрессия        │ MSE / MAE / Huber / Log-Cosh                   │
  │ Бинарная класиф. │ Binary Cross-Entropy (Logistic Loss)           │
  │ Многоклассовая   │ Categorical Cross-Entropy (Softmax + NLL)      │
  │ Сиамские сети    │ Contrastive / Triplet Loss                     │
  │ Сегментация      │ Dice Loss / Focal Loss / BCE + Dice            │
  │ Генеративные     │ KL-Divergence / Reconstruction Loss            │
  └──────────────────┴────────────────────────────────────────────────┘

  Ключевые интуиции:
  • MSE → штрафует большие ошибки сильнее (квадрат) → чувствителен к выбросам
  • MAE → штрафует все ошибки одинаково (линейно) → робастна, но не гладкая
  • Huber → MSE при малых ошибках, MAE при больших → лучшее из двух миров
  • CE → наказывает за уверенное неправильное предсказание экспоненциально
  • Contrastive → разносит разные классы, сближает одинаковые

  Выбор loss — это часть архитектуры модели. Неправильный loss может
  привести к тому, что модель оптимизирует не то, что вам нужно.
""")
