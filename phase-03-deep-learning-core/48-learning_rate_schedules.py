"""
48. Learning Rate Schedules — расписания скорости обучения с нуля.

Реализованы:
  1. Step Decay
  2. Exponential Decay
  3. Cosine Annealing
  4. Warmup + Decay
  5. One Cycle Policy

Зависимости: только стандартная библиотека Python.
"""

import math
import random

random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# 1. STEP DECAY
# ──────────────────────────────────────────────────────────────────────

def step_decay_schedule(initial_lr, drop_factor, drop_every):
    """
    Каждые drop_every эпох умножаем lr на drop_factor.

    Args:
        initial_lr: начальная скорость обучения
        drop_factor: множитель (0 < drop_factor <= 1)
        drop_every: каждые сколько эпох уменьшать

    Returns:
        функция(epoch) -> lr
    """
    def schedule(epoch):
        return initial_lr * (drop_factor ** (epoch // drop_every))
    return schedule


# ──────────────────────────────────────────────────────────────────────
# 2. EXPONENTIAL DECAY
# ──────────────────────────────────────────────────────────────────────

def exponential_decay_schedule(initial_lr, decay_rate):
    """
    lr(epoch) = initial_lr * decay_rate^epoch

    Args:
        initial_lr: начальная скорость обучения
        decay_rate: коэффициент затухания (0 < decay_rate < 1)

    Returns:
        функция(epoch) -> lr
    """
    def schedule(epoch):
        return initial_lr * (decay_rate ** epoch)
    return schedule


# ──────────────────────────────────────────────────────────────────────
# 3. COSINE ANNEALING
# ──────────────────────────────────────────────────────────────────────

def cosine_annealing_schedule(initial_lr, min_lr, total_epochs):
    """
    Cosine Annealing — плавное уменьшение lr по косинусоиде.

    lr(t) = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t / T))

    Args:
        initial_lr: начальная скорость обучения
        min_lr: минимальная скорость обучения
        total_epochs: общее количество эпох

    Returns:
        функция(epoch) -> lr
    """
    def schedule(epoch):
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
    return schedule


# ──────────────────────────────────────────────────────────────────────
# 4. WARMUP + DECAY
# ──────────────────────────────────────────────────────────────────────

def warmup_decay_schedule(initial_lr, warmup_epochs, total_epochs, decay_type="linear"):
    """
    Warmup: lr линейно растёт от 0 до initial_lr за warmup_epochs.
    После warmup: linear decay до ~0 или cosine decay.

    Args:
        initial_lr: целевая скорость обучения
        warmup_epochs: количество эпох warmup
        total_epochs: общее количество эпох
        decay_type: "linear" или "cosine" после warmup

    Returns:
        функция(epoch) -> lr
    """
    train_epochs = total_epochs - warmup_epochs

    def schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(train_epochs - 1, 1)
            if decay_type == "linear":
                return initial_lr * (1.0 - progress)
            elif decay_type == "cosine":
                return initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                return initial_lr * (1.0 - progress)
    return schedule


# ──────────────────────────────────────────────────────────────────────
# 5. ONE CYCLE POLICY
# ──────────────────────────────────────────────────────────────────────

def one_cycle_schedule(max_lr, total_epochs, step_ratio=0.3, div_factor=25, final_div_factor=1e4):
    """
    One Cycle Policy (Leslie Smith):
      - Фаза 1 (step_ratio доля): lr растёт от max_lr/div_factor до max_lr
      - Фаза 2 (остаток): lr убывает от max_lr до max_lr/(div_factor * final_div_factor)

    Args:
        max_lr: максимальная скорость обучения
        total_epochs: общее количество эпох
        step_ratio: доля эпох на фазу роста (по умолчанию 0.3)
        div_factor: делитель для начального lr (по умолчанию 25)
        final_div_factor: делитель для финального lr (по умолчанию 10000)

    Returns:
        функция(epoch) -> lr
    """
    initial_lr = max_lr / div_factor
    final_lr = max_lr / (div_factor * final_div_factor)
    up_epochs = int(total_epochs * step_ratio)
    down_epochs = total_epochs - up_epochs

    def schedule(epoch):
        if epoch < up_epochs:
            progress = epoch / max(up_epochs - 1, 1)
            return initial_lr + (max_lr - initial_lr) * progress
        else:
            progress = (epoch - up_epochs) / max(down_epochs - 1, 1)
            return max_lr - (max_lr - final_lr) * progress
    return schedule


# ──────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────────────────────────────

def simulate_training(schedule_fn, epochs, loss_start=2.5, lr_impact=0.4):
    """
    Упрощённая симуляция обучения.
    На каждой эпохе loss уменьшается пропорционально текущему lr.

    Returns:
        список (epoch, lr, loss)
    """
    results = []
    loss = loss_start
    for epoch in range(epochs):
        lr = schedule_fn(epoch)
        improvement = lr * lr_impact * (1 + random.gauss(0, 0.05))
        loss = max(0.01, loss - improvement + random.gauss(0, 0.02))
        results.append((epoch, lr, loss))
    return results


def print_table(headers, rows, col_widths=None):
    """Красивый вывод таблицы."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
                      for i, h in enumerate(headers)]

    header_line = "".join(str(h).center(col_widths[i]) for i, h in enumerate(headers))
    separator = "".join("-" * col_widths[i] for i in range(len(headers)))

    print(f"  {header_line}")
    print(f"  {separator}")
    for row in rows:
        line = "".join(str(row[i]).center(col_widths[i]) for i in range(len(headers)))
        print(f"  {line}")


def bar(value, max_value, width=30):
    """ASCII-бар для визуализации."""
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return "#" * filled + "." * (width - filled)


# ══════════════════════════════════════════════════════════════════════
#  ДЕМО 1: Все расписания
# ══════════════════════════════════════════════════════════════════════

def demo_all_schedules():
    print("=" * 70)
    print("  ДЕМО 1: Все расписания learning rate")
    print("=" * 70)

    epochs = 20
    initial_lr = 0.1

    schedules = {
        "Step Decay":         step_decay_schedule(initial_lr, 0.5, 5),
        "Exponential Decay":  exponential_decay_schedule(initial_lr, 0.85),
        "Cosine Annealing":   cosine_annealing_schedule(initial_lr, 0.001, epochs),
        "Warmup + Linear":    warmup_decay_schedule(initial_lr, 5, epochs, "linear"),
        "One Cycle":          one_cycle_schedule(initial_lr, epochs),
    }

    for name, sched in schedules.items():
        print(f"\n  >>> {name}")
        print(f"      Epoch 0:  lr = {sched(0):.6f}")
        print(f"      Epoch {epochs // 2}:  lr = {sched(epochs // 2):.6f}")
        print(f"      Epoch {epochs - 1}:  lr = {sched(epochs - 1):.6f}")

    # Таблица сравнения
    print("\n  Сравнение lr на ключевых эпохах:")
    headers = ["Epoch", "Step", "Exp", "Cosine", "Warmup", "OneCycle"]
    rows = []
    for epoch in [0, 4, 9, 14, 19]:
        row = [epoch]
        for name, sched in schedules.items():
            row.append(f"{sched(epoch):.5f}")
        rows.append(row)
    print_table(headers, rows)
    print()


# ══════════════════════════════════════════════════════════════════════
#  ДЕМО 2: Визуализация lr по эпохам
# ══════════════════════════════════════════════════════════════════════

def demo_visualization():
    print("=" * 70)
    print("  ДЕМО 2: Визуализация learning rate по эпохам")
    print("=" * 70)

    epochs = 30
    initial_lr = 0.1

    schedules = {
        "Step Decay":        step_decay_schedule(initial_lr, 0.5, 10),
        "Exponential Decay": exponential_decay_schedule(initial_lr, 0.88),
        "Cosine Annealing":  cosine_annealing_schedule(initial_lr, 0.001, epochs),
        "Warmup + Cosine":   warmup_decay_schedule(initial_lr, 5, epochs, "cosine"),
        "One Cycle":         one_cycle_schedule(initial_lr, epochs),
    }

    max_lr = initial_lr

    for name, sched in schedules.items():
        print(f"\n  {name}:")
        for epoch in range(epochs):
            lr = sched(epoch)
            print(f"    Epoch {epoch:2d} | lr={lr:.6f} | {bar(lr, max_lr, 35)}")
    print()


# ══════════════════════════════════════════════════════════════════════
#  ДЕМО 3: Сравнение на практике (симуляция обучения)
# ══════════════════════════════════════════════════════════════════════

def demo_practical_comparison():
    print("=" * 70)
    print("  ДЕМО 3: Сравнение расписаний на практике (симуляция)")
    print("=" * 70)

    epochs = 50
    initial_lr = 0.1

    schedules = {
        "Constant LR":       lambda epoch: initial_lr,
        "Step Decay":        step_decay_schedule(initial_lr, 0.5, 15),
        "Exponential Decay": exponential_decay_schedule(initial_lr, 0.92),
        "Cosine Annealing":  cosine_annealing_schedule(initial_lr, 0.001, epochs),
        "Warmup + Decay":    warmup_decay_schedule(initial_lr, 5, epochs, "cosine"),
        "One Cycle":         one_cycle_schedule(initial_lr, epochs),
    }

    results = {}
    for name, sched in schedules.items():
        random.seed(42)  # Одинаковые условия для честного сравнения
        sim = simulate_training(sched, epochs, loss_start=3.0, lr_impact=0.35)
        results[name] = sim

    # Вывод итоговых результатов
    print("\n  Итоговые результаты после 50 эпох:")
    print("  " + "-" * 50)
    headers = ["Расписание", "Финальный Loss", "Мин. Loss", "Эпоха мин."]

    rows = []
    for name, sim in results.items():
        final_loss = sim[-1][2]
        min_loss = min(s[2] for s in sim)
        min_epoch = [s[0] for s in sim if s[2] == min_loss][0]
        rows.append([name, f"{final_loss:.4f}", f"{min_loss:.4f}", min_epoch])

    rows.sort(key=lambda r: float(r[1]))
    print_table(headers, rows)

    # График финальных loss (ASCII)
    print("\n  Финальные loss (чем ниже — тем лучше):")
    best_final = min(float(r[1]) for r in rows)
    worst_final = max(float(r[1]) for r in rows)
    for r in rows:
        val = float(r[1])
        normalized = (worst_final - val) / (worst_final - best_final + 1e-9)
        print(f"    {r[0]:22s} | {bar(normalized, 1.0, 40)} | {val:.4f}")

    # Лучшее расписание
    best_name = rows[0][0]
    print(f"\n  Лучшее расписание: {best_name} (loss={rows[0][1]})")
    print()


# ══════════════════════════════════════════════════════════════════════
#  ДЕМО 4: Warmup — старт с малого lr
# ══════════════════════════════════════════════════════════════════════

def demo_warmup():
    print("=" * 70)
    print("  ДЕМО 4: Warmup — старт с малого learning rate")
    print("=" * 70)

    epochs = 40
    initial_lr = 0.1

    warmup_schedules = {
        "Без Warmup":         warmup_decay_schedule(initial_lr, 0, epochs, "cosine"),
        "Warmup 5 эпох":     warmup_decay_schedule(initial_lr, 5, epochs, "cosine"),
        "Warmup 10 эпох":    warmup_decay_schedule(initial_lr, 10, epochs, "cosine"),
        "Warmup 5 (линейн.)": warmup_decay_schedule(initial_lr, 5, epochs, "linear"),
    }

    print("\n  lr на первых 15 эпохах:")
    headers = ["Epoch", "Без WU", "WU=5", "WU=10", "WU=5 lin"]
    rows = []
    for epoch in range(15):
        row = [epoch]
        for name, sched in warmup_schedules.items():
            row.append(f"{sched(epoch):.5f}")
        rows.append(row)
    print_table(headers, rows)

    # Визуализация warmup-фазы
    print("\n  Warmup-фаза (первые 10 эпох):")
    for epoch in range(10):
        vals = []
        for name, sched in warmup_schedules.items():
            vals.append((name, sched(epoch)))
        print(f"    Epoch {epoch:2d}:", end="")
        for name, lr in vals:
            print(f"  {name[:12]:>12s}={lr:.5f}", end="")
        print()

    # Сравнение стабильности
    print("\n  Влияние warmup на стабильность градиентов:")
    print("  (Симуляция: большой lr без warmup может вызвать 'взрыв' градиентов)")
    print()

    # Демонстрация нестабильности
    print("  Сценарий: lr=0.1 без warmup на нестабильном датасете")
    random.seed(42)
    loss = 5.0
    for epoch in range(10):
        lr = initial_lr
        gradient_noise = random.gauss(0, 1.5)  # Высокий шум
        if epoch < 2:
            improvement = lr * gradient_noise
            loss_before = loss
            loss = max(0.01, loss - improvement)
            status = "ВЗРЫВ!" if abs(loss - loss_before) > 1.0 else "ok"
        else:
            improvement = lr * 0.3 + random.gauss(0, 0.1)
            loss = max(0.01, loss - improvement)
            status = "ok"
        print(f"    Epoch {epoch}: loss={loss:.4f} [{status}]")

    print("\n  Сценарий: warmup 3 эпохи — lr растёт постепенно")
    random.seed(42)
    loss = 5.0
    warmup_epochs = 3
    for epoch in range(10):
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            lr = initial_lr * 0.5
        gradient_noise = random.gauss(0, 1.5)
        improvement = lr * 0.3 + random.gauss(0, 0.1)
        loss = max(0.01, loss - improvement)
        print(f"    Epoch {epoch}: lr={lr:.4f} loss={loss:.4f}")

    print("\n  Warmup защищает от нестабильности в начале обучения!")
    print("  Рекомендация: используйте warmup для больших батчей и трансформеров.")
    print()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  48. Learning Rate Schedules — расписания lr с нуля на Python")
    print("#" * 70 + "\n")

    demo_all_schedules()
    demo_visualization()
    demo_practical_comparison()
    demo_warmup()

    print("=" * 70)
    print("  Итоги:")
    print("  - Step Decay: просто и предсказуемо, но грубо")
    print("  - Exponential Decay: плавное затухание, хорошее базовое расписание")
    print("  - Cosine Annealing: лучший баланс, используется в SOTA моделях")
    print("  - Warmup + Decay: защита от нестабильности в начале обучения")
    print("  - One Cycle Policy: агрессивное обучение, быстрая сходимость")
    print("  - Выбор расписания зависит от задачи, модели и данных")
    print("=" * 70)
    print()
