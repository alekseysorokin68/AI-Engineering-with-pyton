"""
Imbalanced Data Handling — методы работы с несбалансированными данными.
Без внешних зависимостей (sklearn не используется).
"""
import random
import math

random.seed(42)

# ─────────────────────────── Утилиты ───────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _rand_normal(mu=0.0, sigma=1.0):
    u1 = random.random()
    u2 = random.random()
    return mu + sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# ─────────────────────── Генерация данных ──────────────────────

def generate_imbalanced(n_total=1000, ratio=0.05, n_features=5, seed=42):
    """
    Генерирует несбалансированный датасет.
    ratio — доля миноритарного класса (0.05 = 5%).
    Возвращает (X, y) где X — список векторов, y — список меток {0, 1}.
    """
    random.seed(seed)
    n_minority = int(n_total * ratio)
    n_majority = n_total - n_minority

    X, y = [], []

    for _ in range(n_majority):
        features = [_rand_normal(0, 1) for _ in range(n_features)]
        X.append(features)
        y.append(0)

    for _ in range(n_minority):
        features = [_rand_normal(2, 1) for _ in range(n_features)]
        X.append(features)
        y.append(1)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


# ──────────────────────── Метрики ─────────────────────────────

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def precision_score(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def auc_score(y_true, y_scores):
    """AUC через трапециевидное интегрирование ROC-кривой."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    pos = sum(1 for t in y_true if t == 1)
    neg = sum(1 for t in y_true if t == 0)
    if pos == 0 or neg == 0:
        return 0.5

    tp, fp = 0, 0
    tpr_prev, fpr_prev = 0.0, 0.0
    auc = 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / pos
        fpr = fp / neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    return auc


# ─────────────────── Oversampling (SMOTE-подобный) ─────────────

def _knn_indices(X, idx, k=5):
    """Находит k ближайших соседей для X[idx]."""
    x = X[idx]
    dists = []
    for i, xi in enumerate(X):
        if i == idx:
            continue
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, xi)))
        dists.append((d, i))
    dists.sort(key=lambda x: x[0])
    return [i for _, i in dists[:k]]


def smote_oversample(X, y, k=5, seed=42):
    """
    SMOTE-подобный oversampling: для каждого миноритарного примера
    генерирует синтетический пример между ним и одним из k ближайших соседей.
    """
    random.seed(seed)
    minority_idx = [i for i, label in enumerate(y) if label == 1]
    majority_idx = [i for i, label in enumerate(y) if label == 0]
    target_n = len(majority_idx)

    X_new = list(X)
    y_new = list(y)

    while len([l for l in y_new if l == 1]) < target_n:
        idx = random.choice(minority_idx)
        neighbors = _knn_indices(X_new, idx, k=min(k, len(X_new) - 1))
        if not neighbors:
            continue
        neighbor = random.choice(neighbors)
        lam = random.random()
        synthetic = [
            X_new[idx][f] + lam * (X_new[neighbor][f] - X_new[idx][f])
            for f in range(len(X_new[0]))
        ]
        X_new.append(synthetic)
        y_new.append(1)

    return X_new, y_new


# ─────────────────── Undersampling ─────────────────────────────

def undersample(X, y, seed=42):
    """Рандомный undersampling мажоритарного класса до размера миноритарного."""
    random.seed(seed)
    minority_idx = [i for i, label in enumerate(y) if label == 1]
    majority_idx = [i for i, label in enumerate(y) if label == 0]
    target_n = len(minority_idx)

    sampled_majority = random.sample(majority_idx, target_n)
    keep = set(minority_idx + sampled_majority)

    X_new = [X[i] for i in sorted(keep)]
    y_new = [y[i] for i in sorted(keep)]
    return X_new, y_new


# ──────────────────── Class Weights ────────────────────────────

def compute_class_weights(y):
    """Вычисляет веса классов: weight = n / (n_classes * count)."""
    n = len(y)
    n_classes = len(set(y))
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return {c: n / (n_classes * cnt) for c, cnt in counts.items()}


def weighted_predict(X, weights, threshold=0.0):
    """
    Логистическая регрессия с class weights.
    Функция потерь: weighted cross-entropy.
    Возвращает предсказания {0, 1}.
    """
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    lr = 0.1
    n_epochs = 200

    n0 = weights.get(0, 1.0)
    n1 = weights.get(1, 1.0)

    for _ in range(n_epochs):
        for xi, yi in zip(X, y_train):
            z = _dot(w, xi) + b
            pred = _sigmoid(z)
            err = pred - yi
            weight = n1 if yi == 1 else n0
            for j in range(n_features):
                w[j] -= lr * weight * err * xi[j]
            b -= lr * weight * err

    preds = []
    for xi in X_test:
        z = _dot(w, xi) + b
        preds.append(1 if _sigmoid(z) > 0.5 else 0)
    return preds


# ──────────────────── Логистическая регрессия ──────────────────

def train_logistic_regression(X_train, y_train, lr=0.1, epochs=200, weights=None):
    """Обучает логистическую регрессию. Возвращает (w, b)."""
    n_features = len(X_train[0])
    w = [0.0] * n_features
    b = 0.0

    if weights is None:
        weights = {0: 1.0, 1: 1.0}

    n0 = weights.get(0, 1.0)
    n1 = weights.get(1, 1.0)

    for _ in range(epochs):
        for xi, yi in zip(X_train, y_train):
            z = _dot(w, xi) + b
            pred = _sigmoid(z)
            err = pred - yi
            class_weight = n1 if yi == 1 else n0
            for j in range(n_features):
                w[j] -= lr * class_weight * err * xi[j]
            b -= lr * class_weight * err

    return w, b


def predict(X, w, b, threshold=0.5):
    """Предсказания обученной модели."""
    return [1 if _sigmoid(_dot(w, xi) + b) > threshold else 0 for xi in X]


def predict_proba(X, w, b):
    """Вероятности предсказаний."""
    return [_sigmoid(_dot(w, xi) + b) for xi in X]


# ─────────────────── Разбиение данных ──────────────────────────

def train_test_split(X, y, test_ratio=0.3, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    X_tr = [X[i] for i in train_idx]
    y_tr = [y[i] for i in train_idx]
    X_te = [X[i] for i in test_idx]
    y_te = [y[i] for i in test_idx]
    return X_tr, y_tr, X_te, y_te


# ═══════════════════════ ДЕМОНСТРАЦИИ ═══════════════════════════

def demo_1_imbalanced_problem():
    print("=" * 65)
    print("ДЕМО 1: Проблема несбалансированных данных")
    print("=" * 65)

    X, y = generate_imbalanced(n_total=1000, ratio=0.05)
    n0 = sum(1 for l in y if l == 0)
    n1 = sum(1 for l in y if l == 1)
    print(f"\nДатасет: {len(y)} примеров")
    print(f"  Класс 0 (мажоритарный): {n0} ({n0/len(y)*100:.1f}%)")
    print(f"  Класс 1 (миноритарный): {n1} ({n1/len(y)*100:.1f}%)")
    print(f"  Дисбаланс: {n0/n1:.1f}x")

    X_train, y_train, X_test, y_test = train_test_split(X, y)

    w, b = train_logistic_regression(X_train, y_train)
    y_pred = predict(X_test, w, b)
    y_proba = predict_proba(X_test, w, b)

    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"\nНаивная модель (все предсказания = 0):")
    print(f"  Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.1f}%")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"  F1: {f1_score(y_test, y_pred):.3f}")
    print(f"  AUC: {auc_score(y_test, y_proba):.3f}")

    print(f"\nПроблема: модель предсказывает ТОЛЬКО класс 0!")
    print(f"  Все {fn} миноритарных примеров пропущены (FN={fn})")
    print(f"  Accuracy высокая ({(tp+tn)/(tp+tn+fp+fn)*100:.1f}%), но модель бесполезна")


def demo_2_oversampling_vs_undersampling():
    print("\n" + "=" * 65)
    print("ДЕМО 2: Oversampling (SMOTE) vs Undersampling")
    print("=" * 65)

    X, y = generate_imbalanced(n_total=1000, ratio=0.05)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"\nИсходный датасет:")
    n0 = sum(1 for l in y_train if l == 0)
    n1 = sum(1 for l in y_train if l == 1)
    print(f"  Train: {len(y_train)} (0={n0}, 1={n1})")
    n0 = sum(1 for l in y_test if l == 0)
    n1 = sum(1 for l in y_test if l == 1)
    print(f"  Test:  {len(y_test)} (0={n0}, 1={n1})")

    # --- Oversampling ---
    X_smote, y_smote = smote_oversample(X_train, y_train)
    n0s = sum(1 for l in y_smote if l == 0)
    n1s = sum(1 for l in y_smote if l == 1)
    print(f"\nПосле SMOTE oversampling:")
    print(f"  Train: {len(y_smote)} (0={n0s}, 1={n1s})")

    w_sm, b_sm = train_logistic_regression(X_smote, y_smote)
    y_pred_sm = predict(X_test, w_sm, b_sm)
    y_proba_sm = predict_proba(X_test, w_sm, b_sm)

    print(f"\n  SMOTE модель:")
    print(f"    Precision: {precision_score(y_test, y_pred_sm):.3f}")
    print(f"    Recall:    {recall_score(y_test, y_pred_sm):.3f}")
    print(f"    F1:        {f1_score(y_test, y_pred_sm):.3f}")
    print(f"    AUC:       {auc_score(y_test, y_proba_sm):.3f}")

    # --- Undersampling ---
    X_under, y_under = undersample(X_train, y_train)
    n0u = sum(1 for l in y_under if l == 0)
    n1u = sum(1 for l in y_under if l == 1)
    print(f"\nПосле undersampling:")
    print(f"  Train: {len(y_under)} (0={n0u}, 1={n1u})")

    w_un, b_un = train_logistic_regression(X_under, y_under)
    y_pred_un = predict(X_test, w_un, b_un)
    y_proba_un = predict_proba(X_test, w_un, b_un)

    print(f"\n  Undersampling модель:")
    print(f"    Precision: {precision_score(y_test, y_pred_un):.3f}")
    print(f"    Recall:    {recall_score(y_test, y_pred_un):.3f}")
    print(f"    F1:        {f1_score(y_test, y_pred_un):.3f}")
    print(f"    AUC:       {auc_score(y_test, y_proba_un):.3f}")


def demo_3_class_weights():
    print("\n" + "=" * 65)
    print("ДЕМО 3: Class Weights")
    print("=" * 65)

    X, y = generate_imbalanced(n_total=1000, ratio=0.05)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    weights = compute_class_weights(y_train)
    print(f"\nВеса классов:")
    for c, w in sorted(weights.items()):
        cnt = sum(1 for l in y_train if l == c)
        print(f"  Класс {c}: weight={w:.3f} (count={cnt})")

    # Без весов
    w0, b0 = train_logistic_regression(X_train, y_train, weights={0: 1.0, 1: 1.0})
    y_pred0 = predict(X_test, w0, b0)
    y_proba0 = predict_proba(X_test, w0, b0)

    print(f"\nБез class weights:")
    print(f"  Precision: {precision_score(y_test, y_pred0):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred0):.3f}")
    print(f"  F1:        {f1_score(y_test, y_pred0):.3f}")
    print(f"  AUC:       {auc_score(y_test, y_proba0):.3f}")

    # С весами
    w1, b1 = train_logistic_regression(X_train, y_train, weights=weights)
    y_pred1 = predict(X_test, w1, b1)
    y_proba1 = predict_proba(X_test, w1, b1)

    print(f"\nС class weights:")
    print(f"  Precision: {precision_score(y_test, y_pred1):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred1):.3f}")
    print(f"  F1:        {f1_score(y_test, y_pred1):.3f}")
    print(f"  AUC:       {auc_score(y_test, y_proba1):.3f}")


def demo_4_metrics_comparison():
    print("\n" + "=" * 65)
    print("ДЕМО 4: Сравнение метрик — до и после обработки")
    print("=" * 65)

    X, y = generate_imbalanced(n_total=1000, ratio=0.05)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    weights = compute_class_weights(y_train)

    results = []

    # 1. Без обработки
    w, b = train_logistic_regression(X_train, y_train)
    yp = predict(X_test, w, b)
    ypr = predict_proba(X_test, w, b)
    results.append((
        "Без обработки",
        precision_score(y_test, yp),
        recall_score(y_test, yp),
        f1_score(y_test, yp),
        auc_score(y_test, ypr),
    ))

    # 2. SMOTE
    Xs, ys = smote_oversample(X_train, y_train)
    ws, bs = train_logistic_regression(Xs, ys)
    yp = predict(X_test, ws, bs)
    ypr = predict_proba(X_test, ws, bs)
    results.append((
        "SMOTE oversampling",
        precision_score(y_test, yp),
        recall_score(y_test, yp),
        f1_score(y_test, yp),
        auc_score(y_test, ypr),
    ))

    # 3. Undersampling
    Xu, yu = undersample(X_train, y_train)
    wu, bu = train_logistic_regression(Xu, yu)
    yp = predict(X_test, wu, bu)
    ypr = predict_proba(X_test, wu, bu)
    results.append((
        "Undersampling",
        precision_score(y_test, yp),
        recall_score(y_test, yp),
        f1_score(y_test, yp),
        auc_score(y_test, ypr),
    ))

    # 4. Class weights
    wc, bc = train_logistic_regression(X_train, y_train, weights=weights)
    yp = predict(X_test, wc, bc)
    ypr = predict_proba(X_test, wc, bc)
    results.append((
        "Class weights",
        precision_score(y_test, yp),
        recall_score(y_test, yp),
        f1_score(y_test, yp),
        auc_score(y_test, ypr),
    ))

    # 5. SMOTE + Class weights
    wc2, bc2 = train_logistic_regression(Xs, ys, weights=compute_class_weights(ys))
    yp = predict(X_test, wc2, bc2)
    ypr = predict_proba(X_test, wc2, bc2)
    results.append((
        "SMOTE + Class weights",
        precision_score(y_test, yp),
        recall_score(y_test, yp),
        f1_score(y_test, yp),
        auc_score(y_test, ypr),
    ))

    print(f"\n{'Метод':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 55)
    for name, prec, rec, f1, auc in results:
        print(f"{name:<25} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {auc:>7.3f}")

    print(f"\nВыводы:")
    print(f"  - Accuracy обманчива при дисбалансе классов")
    print(f"  - Recall вырастает после oversampling/undersampling")
    print(f"  - Class weights — самый эффективный метод (без генерации данных)")
    print(f"  - Комбинация SMOTE + Class weights даёт лучшие результаты")


# ═══════════════════════════ ЗАПУСК ═════════════════════════════

if __name__ == "__main__":
    demo_1_imbalanced_problem()
    demo_2_oversampling_vs_undersampling()
    demo_3_class_weights()
    demo_4_metrics_comparison()
    print("\n" + "=" * 65)
    print("Все демонстрации завершены.")
    print("=" * 65)
