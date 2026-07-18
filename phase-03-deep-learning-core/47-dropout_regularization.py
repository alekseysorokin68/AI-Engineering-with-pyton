"""
47 — Dropout & Regularization methods from scratch.

Demonstrates:
  1. Dropout (random neuron deactivation)
  2. L1 / L2 weight regularization
  3. Early stopping
  4. Comparison of all methods on the same toy dataset

No external libraries — only stdlib (random, math, copy).
"""

import random
import math
import copy

random.seed(42)

# ---------------------------------------------------------------------------
# Helper: sigmoid, MSE, derivative
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def sigmoid_deriv(out):
    return out * (1.0 - out)


def mse_loss(y_true, y_pred):
    n = len(y_true)
    return sum(((y_true[i][0] if isinstance(y_true[i], list) else y_true[i]) - y_pred[i]) ** 2 for i in range(n)) / n


# ---------------------------------------------------------------------------
# Simple 2-layer neural network (pure Python)
# ---------------------------------------------------------------------------

class NeuralNet:
    """Fully-connected net: input_size -> hidden_size -> output_size."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.lr = lr
        self.w_ih = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                      for _ in range(input_size)]
        self.b_h = [0.0] * hidden_size
        self.w_ho = [[random.uniform(-0.5, 0.5) for _ in range(output_size)]
                     for _ in range(hidden_size)]
        self.b_o = [0.0] * output_size

    def forward(self, x, dropout_mask=None):
        # hidden
        h_raw = [self.b_h[j] for j in range(len(self.b_h))]
        for i in range(len(x)):
            for j in range(len(self.b_h)):
                h_raw[j] += x[i] * self.w_ih[i][j]
        h_out = [sigmoid(v) for v in h_raw]

        # apply dropout mask
        if dropout_mask is not None:
            h_out = [h_out[j] * dropout_mask[j] for j in range(len(h_out))]

        # output
        o_raw = [self.b_o[k] for k in range(len(self.b_o))]
        for j in range(len(h_out)):
            for k in range(len(self.b_o)):
                o_raw[k] += h_out[j] * self.w_ho[j][k]
        o_out = [sigmoid(v) for v in o_raw]
        return h_out, o_out

    def train_step(self, x, y_true, dropout_rate=0.0, l1=0.0, l2=0.0):
        # dropout mask
        mask = None
        if dropout_rate > 0:
            mask = [0.0 if random.random() < dropout_rate else 1.0
                    for _ in range(len(self.b_h))]

        h_out, o_out = self.forward(x, mask)

        # output layer deltas
        o_err = [(y_true[k] - o_out[k]) * sigmoid_deriv(o_out[k])
                 for k in range(len(o_out))]

        # hidden layer deltas
        h_err = [0.0] * len(h_out)
        for j in range(len(h_out)):
            for k in range(len(o_err)):
                h_err[j] += o_err[k] * self.w_ho[j][k]
            if mask is None or mask[j] != 0:
                h_err[j] *= sigmoid_deriv(h_out[j])
            else:
                h_err[j] = 0.0

        # update hidden->output weights
        for j in range(len(h_out)):
            for k in range(len(o_err)):
                grad = o_err[k] * h_out[j]
                # L2 penalty on gradient
                grad -= l2 * self.w_ho[j][k]
                # L1 penalty (subgradient)
                grad -= l1 * (1.0 if self.w_ho[j][k] > 0 else -1.0)
                self.w_ho[j][k] += self.lr * grad

        # update input->hidden weights
        for i in range(len(x)):
            for j in range(len(h_err)):
                grad = h_err[j] * x[i]
                grad -= l2 * self.w_ih[i][j]
                grad -= l1 * (1.0 if self.w_ih[i][j] > 0 else -1.0)
                self.w_ih[i][j] += self.lr * grad

        # biases
        for k in range(len(o_err)):
            self.b_o[k] += self.lr * o_err[k]
        for j in range(len(h_err)):
            self.b_h[j] += self.lr * h_err[j]

        return o_out

    def predict(self, x):
        _, o = self.forward(x, dropout_mask=None)
        return o

    def copy(self):
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Synthetic dataset: XOR-like with noise
# ---------------------------------------------------------------------------

def make_dataset(n=200):
    xs, ys = [], []
    for _ in range(n):
        x1 = random.random()
        x2 = random.random()
        label = 1.0 if (x1 > 0.5) ^ (x2 > 0.5) else 0.0
        # add noise
        x1 += random.gauss(0, 0.05)
        x2 += random.gauss(0, 0.05)
        xs.append([x1, x2])
        ys.append([label])
    return xs, ys


def train_network(net, xs, ys, epochs, dropout_rate=0.0, l1=0.0, l2=0.0,
                  early_stop_patience=None, xs_val=None, ys_val=None):
    history = []
    best_val = float('inf')
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        indices = list(range(len(xs)))
        random.shuffle(indices)
        for i in indices:
            net.train_step(xs[i], ys[i], dropout_rate=dropout_rate,
                           l1=l1, l2=l2)
        preds = [net.predict(xs[i])[0] for i in range(len(xs))]
        loss = mse_loss(ys, preds)
        history.append(loss)

        # early stopping check
        if early_stop_patience is not None and xs_val is not None:
            val_preds = [net.predict(xs_val[i])[0] for i in range(len(xs_val))]
            val_loss = mse_loss(ys_val, val_preds)
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_weights = net.copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    # restore best weights
                    net.w_ih = best_weights.w_ih
                    net.b_h = best_weights.b_h
                    net.w_ho = best_weights.w_ho
                    net.b_o = best_weights.b_o
                    history.append(f"Early stop at epoch {epoch + 1}")
                    break

    return history


def evaluate(net, xs, ys):
    correct = 0
    total = len(xs)
    for i in range(total):
        pred = net.predict(xs[i])[0]
        predicted_class = 1 if pred > 0.5 else 0
        actual_class = int(ys[i][0])
        if predicted_class == actual_class:
            correct += 1
    return correct / total * 100


# ===========================================================================
# Demo 1 — Dropout: random neuron deactivation
# ===========================================================================

def demo_dropout():
    print("=" * 60)
    print("DEMO 1: Dropout — случайное обнуление нейронов")
    print("=" * 60)

    xs, ys = make_dataset(200)
    xs_train, ys_train = xs[:160], ys[:160]
    xs_test, ys_test = xs[160:], ys[160:]

    print("\nПараметры:")
    print("  Датасет: 200 сэмплов (160 train / 20 val / 20 test)")
    print("  Архитектура: 2 -> 16 -> 1")
    print("  Эпохи: 300")
    print()

    # No dropout
    random.seed(42)
    net_no_drop = NeuralNet(2, 16, 1, lr=0.5)
    train_network(net_no_drop, xs_train, ys_train, epochs=300)
    acc_no_drop = evaluate(net_no_drop, xs_test, ys_test)

    # With dropout 0.5
    random.seed(42)
    net_drop = NeuralNet(2, 16, 1, lr=0.5)
    train_network(net_drop, xs_train, ys_train, epochs=300, dropout_rate=0.5)
    acc_drop = evaluate(net_drop, xs_test, ys_test)

    # Show sample forward passes with dropout
    random.seed(7)
    print("Пример forward pass (dropout=0.5):")
    sample_x = [0.7, 0.3]
    hidden_no_drop, out_no_drop = net_no_drop.forward(sample_x)
    mask_ex = [0.0 if random.random() < 0.5 else 1.0 for _ in range(16)]
    hidden_drop, out_drop = net_drop.forward(sample_x, dropout_mask=mask_ex)

    active = [j for j, m in enumerate(mask_ex) if m == 1.0]
    inactive = [j for j, m in enumerate(mask_ex) if m == 0.0]
    print(f"  Вход: {sample_x}")
    print(f"  Активные нейроны ({len(active)}): {active[:8]}...")
    print(f"  Обнулённые ({len(inactive)}): {inactive[:8]}...")
    print(f"  Выход без dropout: {out_no_drop[0]:.4f}")
    print(f"  Выход с dropout:   {out_drop[0]:.4f}")

    print(f"\n  Точность без dropout: {acc_no_drop:.1f}%")
    print(f"  Точность с dropout:  {acc_drop:.1f}%")
    print(f"\n  Dropout помогает сети обобщаться, выключая случайные")
    print(f"  нейроны при обучении — это предотвращает overfitting.\n")


# ===========================================================================
# Demo 2 — L1 vs L2 Regularization
# ===========================================================================

def demo_l1_l2():
    print("=" * 60)
    print("DEMO 2: L1 vs L2 регуляризация")
    print("=" * 60)

    xs, ys = make_dataset(200)
    xs_train, ys_train = xs[:160], ys[:160]
    xs_val, ys_val = xs[160:180], ys[160:180]
    xs_test, ys_test = xs[180:], ys[180:]

    configs = [
        ("Без регуляризации", 0.0, 0.0),
        ("L2 (lambda=0.0001)", 0.0, 0.0001),
        ("L2 (lambda=0.001)",  0.0, 0.001),
        ("L1 (lambda=0.0001)", 0.0001, 0.0),
        ("L1 (lambda=0.001)",  0.001, 0.0),
        ("L1+L2 (0.0005 ea)", 0.0005, 0.0005),
    ]

    print("\nПараметры:")
    print("  Архитектура: 2 -> 16 -> 1")
    print("  Эпохи: 300, lr=0.5\n")

    print(f"{'Метод':<22} {'Loss (train)':<14} {'Loss (val)':<14} {'Acc (test)':<12} {'% ненулевых':<14}")
    print("-" * 78)

    for name, l1, l2 in configs:
        random.seed(42)
        net = NeuralNet(2, 16, 1, lr=0.5)
        train_history = train_network(net, xs_train, ys_train, epochs=300,
                                      l1=l1, l2=l2)
        # final train loss
        preds_train = [net.predict(xs_train[i])[0] for i in range(len(xs_train))]
        train_loss = mse_loss(ys_train, preds_train)

        # val loss
        preds_val = [net.predict(xs_val[i])[0] for i in range(len(xs_val))]
        val_loss = mse_loss(ys_val, preds_val)

        # test accuracy
        acc = evaluate(net, xs_test, ys_test)

        # count non-zero weights
        total_w = 0
        non_zero = 0
        for row in net.w_ih:
            for w in row:
                total_w += 1
                if abs(w) > 1e-6:
                    non_zero += 1
        for row in net.w_ho:
            for w in row:
                total_w += 1
                if abs(w) > 1e-6:
                    non_zero += 1

        sparse_pct = (1 - non_zero / total_w) * 100 if total_w > 0 else 0
        print(f"  {name:<20} {train_loss:<14.6f} {val_loss:<14.6f} {acc:<12.1f} {sparse_pct:.0f}% разреж.")

    print(f"\n  L1 регуляризация разреживает веса (sparsity) —")
    print(f"  many weights become exactly 0.")
    print(f"  L2 регуляризация штрафует большие веса —")
    print(f"  weights stay small but rarely exactly 0.")
    print(f"  Комбинация L1+L2 даёт компромисс.\n")


# ===========================================================================
# Demo 3 — Early Stopping
# ===========================================================================

def demo_early_stopping():
    print("=" * 60)
    print("DEMO 3: Early Stopping")
    print("=" * 60)

    xs, ys = make_dataset(200)
    xs_train, ys_train = xs[:120], ys[:120]
    xs_val, ys_val = xs[120:160], ys[120:160]
    xs_test, ys_test = xs[160:], ys[160:]

    print("\nПараметры:")
    print("  Архитектура: 2 -> 16 -> 1")
    print("  Макс эпохи: 1000")
    print("  Patience: 50 эпох без улучшения на val\n")

    # Without early stopping
    random.seed(42)
    net_full = NeuralNet(2, 16, 1, lr=0.5)
    hist_full = train_network(net_full, xs_train, ys_train, epochs=1000)
    full_epochs = len(hist_full)
    acc_full = evaluate(net_full, xs_test, ys_test)

    preds_val_full = [net_full.predict(xs_val[i])[0] for i in range(len(xs_val))]
    val_loss_full = mse_loss(ys_val, preds_val_full)

    # With early stopping
    random.seed(42)
    net_es = NeuralNet(2, 16, 1, lr=0.5)
    hist_es = train_network(net_es, xs_train, ys_train, epochs=1000,
                            early_stop_patience=50, xs_val=xs_val, ys_val=ys_val)

    # Check how many epochs actually ran (last element might be a string)
    es_epochs = len(hist_es)
    if isinstance(hist_es[-1], str):
        es_epochs = len(hist_es) - 1
        stop_msg = hist_es[-1]
    else:
        stop_msg = "completed all epochs"

    preds_val_es = [net_es.predict(xs_val[i])[0] for i in range(len(xs_val))]
    val_loss_es = mse_loss(ys_val, preds_val_es)
    acc_es = evaluate(net_es, xs_test, ys_test)

    # Final train losses
    preds_tr_full = [net_full.predict(xs_train[i])[0] for i in range(len(xs_train))]
    loss_tr_full = mse_loss(ys_train, preds_tr_full)
    preds_tr_es = [net_es.predict(xs_train[i])[0] for i in range(len(xs_train))]
    loss_tr_es = mse_loss(ys_train, preds_tr_es)

    print(f"{'Метод':<22} {'Эпохи':<10} {'Train Loss':<14} {'Val Loss':<14} {'Test Acc':<10}")
    print("-" * 72)
    print(f"  {'Без early stop':<20} {full_epochs:<10} {loss_tr_full:<14.6f} {val_loss_full:<14.6f} {acc_full:<10.1f}%")
    print(f"  {'С early stop':<20} {es_epochs:<10} {loss_tr_es:<14.6f} {val_loss_es:<14.6f} {acc_es:<10.1f}%")

    print(f"\n  {stop_msg}")
    print(f"\n  Early stopping предотвращает overfitting, останавливая")
    print(f"  обучение когда val loss перестаёт уменьшаться.")
    print(f"  Экономия: {full_epochs - es_epochs} эпох ({(1 - es_epochs/full_epochs)*100:.0f}% времени).\n")


# ===========================================================================
# Demo 4 — Сравнение всех методов
# ===========================================================================

def demo_comparison():
    print("=" * 60)
    print("DEMO 4: Сравнение всех методов регуляризации")
    print("=" * 60)

    xs, ys = make_dataset(300)
    xs_train, ys_train = xs[:200], ys[:200]
    xs_val, ys_val = xs[200:250], ys[200:250]
    xs_test, ys_test = xs[250:], ys[250:]

    print("\nПараметры:")
    print("  Датасет: 300 сэмплов (200 train / 50 val / 50 test)")
    print("  Архитектура: 2 -> 16 -> 1")
    print("  Эпохи: 500, lr=0.5\n")

    methods = [
        ("Базовая модель",      dict(dropout_rate=0.0, l1=0.0, l2=0.0, early_stop_patience=None)),
        ("Dropout 0.3",         dict(dropout_rate=0.3, l1=0.0, l2=0.0, early_stop_patience=None)),
        ("Dropout 0.5",         dict(dropout_rate=0.5, l1=0.0, l2=0.0, early_stop_patience=None)),
        ("L2 (0.0005)",         dict(dropout_rate=0.0, l1=0.0, l2=0.0005, early_stop_patience=None)),
        ("L1 (0.0005)",         dict(dropout_rate=0.0, l1=0.0005, l2=0.0, early_stop_patience=None)),
        ("Early Stop (pat=50)", dict(dropout_rate=0.0, l1=0.0, l2=0.0, early_stop_patience=50,
                                    xs_val=xs_val, ys_val=ys_val)),
        ("L2 + Dropout 0.3",   dict(dropout_rate=0.3, l1=0.0, l2=0.0005, early_stop_patience=None)),
        ("L2 + Early Stop",     dict(dropout_rate=0.0, l1=0.0, l2=0.0005, early_stop_patience=50,
                                    xs_val=xs_val, ys_val=ys_val)),
        ("Всё вместе",          dict(dropout_rate=0.3, l1=0.0002, l2=0.0003, early_stop_patience=50,
                                    xs_val=xs_val, ys_val=ys_val)),
    ]

    print(f"{'Метод':<24} {'Train Loss':<14} {'Val Loss':<14} {'Test Acc':<10} {'Эпохи':<8}")
    print("-" * 72)

    results = []
    for name, kwargs in methods:
        random.seed(42)
        net = NeuralNet(2, 16, 1, lr=0.5)

        xs_v = kwargs.pop("xs_val", None)
        ys_v = kwargs.pop("ys_val", None)

        hist = train_network(net, xs_train, ys_train, epochs=500,
                             xs_val=xs_v, ys_val=ys_v, **kwargs)

        epochs_ran = len(hist)
        if isinstance(hist[-1], str):
            epochs_ran = len(hist) - 1

        preds_tr = [net.predict(xs_train[i])[0] for i in range(len(xs_train))]
        train_loss = mse_loss(ys_train, preds_tr)

        preds_v = [net.predict(xs_val[i])[0] for i in range(len(xs_val))]
        val_loss = mse_loss(ys_val, preds_v)

        acc = evaluate(net, xs_test, ys_test)
        results.append((name, train_loss, val_loss, acc, epochs_ran))

        print(f"  {name:<22} {train_loss:<14.6f} {val_loss:<14.6f} {acc:<10.1f}% {epochs_ran:<8}")

    # Summary
    print("\n" + "-" * 72)
    best_acc = max(results, key=lambda r: r[3])
    best_val = min(results, key=lambda r: r[2])
    least_epochs = min(results, key=lambda r: r[4])

    print(f"\n  Лучшая test точность:  {best_acc[0]} ({best_acc[3]:.1f}%)")
    print(f"  Низший val loss:       {best_val[0]} ({best_val[2]:.6f})")
    print(f"  Меньше всего эпох:     {least_epochs[0]} ({least_epochs[4]})")

    print("\n  ВЫВОДЫ:")
    print("  1. Dropout эффективен при переобучении — снижает val loss")
    print("  2. L2 регуляризация стабилизирует обучение, удерживая веса малыми")
    print("  3. Early stopping — простой и надёжный метод контроля переобучения")
    print("  4. Комбинация методов обычно даёт лучший результат")
    print("  5. Подбирать гиперпараметры регуляризации на валидационной выборке!\n")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  РЕГУЛЯРИЗАЦИЯ НЕЙРОННЫХ СЕТЕЙ — ИЗ СЛЕДСТВ В КОД")
    print("=" * 60 + "\n")

    demo_dropout()
    demo_l1_l2()
    demo_early_stopping()
    demo_comparison()

    print("=" * 60)
    print("  Все демо завершены!")
    print("=" * 60 + "\n")
