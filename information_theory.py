import math
import random

random.seed(42)


# ============================================================
#  Информационное содержание и энтропия
# ============================================================

def information_content(p, base=2):
    if p <= 0:
        return float('inf')
    if p >= 1:
        return 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )


# ============================================================
#  Перекрёстная энтропия и KL-расхождение
# ============================================================

def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)


# ============================================================
#  Softmax и cross-entropy loss
# ============================================================

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])


# ============================================================
#  Взаимная информация
# ============================================================

def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi


# ============================================================
#  Label smoothing
# ============================================================

def label_smoothing(true_class, num_classes, epsilon=0.1):
    smooth = [epsilon / num_classes] * num_classes
    smooth[true_class] += (1 - epsilon)
    return smooth


# ============================================================
#  Демо 1: Информационное содержание
# ============================================================

print("=" * 55)
print("ДЕМО 1: Информационное содержание (Surprise)")
print("=" * 55)

events = [
    ("Монетка орёл", 0.5),
    ("Выпало 6 на кубике", 1/6),
    ("1 из 1000", 0.001),
    ("Наверняка", 1.0),
]

for name, prob in events:
    info = information_content(prob, base=2)
    print(f"  {name:<25} P={prob:<8.3f} surprise={info:.2f} бит")


# ============================================================
#  Демо 2: Энтропия
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Энтропия")
print("=" * 55)

distributions = [
    ("Честная монета", [0.5, 0.5]),
    ("Смещённая (99%)", [0.99, 0.01]),
    ("Честный кубик", [1/6] * 6),
    ("Один класс (100%)", [1.0]),
    ("Три класса (равные)", [1/3, 1/3, 1/3]),
]

for name, probs in distributions:
    h = entropy(probs, base=2)
    h_nat = entropy(probs, base=math.e)
    print(f"  {name:<25} H={h:.4f} бит  ({h_nat:.4f} нат)")


# ============================================================
#  Демо 3: Cross-entropy и KL-расхождение
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Cross-entropy и KL-расхождение")
print("=" * 55)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

h_true = entropy(true_dist, base=2)
ce_good = cross_entropy(true_dist, good_model, base=2)
ce_bad = cross_entropy(true_dist, bad_model, base=2)
kl_good = kl_divergence(true_dist, good_model, base=2)
kl_bad = kl_divergence(true_dist, bad_model, base=2)

print(f"\nИстинное:  {true_dist}")
print(f"Хорошая:   {good_model}")
print(f"Плохая:    {bad_model}")
print(f"\nЭнтропия истинного:     {h_true:.4f} бит")
print(f"CE (хорошая модель):    {ce_good:.4f} бит")
print(f"CE (плохая модель):     {ce_bad:.4f} бит")
print(f"KL (хорошая):           {kl_good:.4f} бит")
print(f"KL (плохая):            {kl_bad:.4f} бит")
print(f"\nПроверка: CE = H + KL")
print(f"  Хорошая: {h_true:.4f} + {kl_good:.4f} = {h_true + kl_good:.4f} ≈ {ce_good:.4f}")
print(f"  Плохая:  {h_true:.4f} + {kl_bad:.4f} = {h_true + kl_bad:.4f} ≈ {ce_bad:.4f}")


# ============================================================
#  Демо 4: Cross-entropy loss для классификации
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Cross-entropy loss для классификации")
print("=" * 55)

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"\nЛогиты:     {logits}")
print(f"Softmax:    {[f'{p:.4f}' for p in probs]}")
print(f"Правильный: класс {true_class}")
print(f"Loss:       {loss:.4f} нат")
print(f"Perplexity: {math.exp(loss):.2f}")


# ============================================================
#  Демо 5: Cross-entropy = negative log-likelihood
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: CE = NLL (проверка)")
print("=" * 55)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"\nCross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Разница:                 {abs(ce_loss - nll):.2e}")


# ============================================================
#  Демо 6: Взаимная информация
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Взаимная информация (Mutual Information)")
print("=" * 55)

# Независимые переменные
independent = [[0.25, 0.25], [0.25, 0.25]]

# Зависимые переменные
dependent = [[0.45, 0.05], [0.05, 0.45]]

# Полностью зависимые
perfect = [[0.5, 0.0], [0.0, 0.5]]

mi_ind = mutual_information(independent, base=2)
mi_dep = mutual_information(dependent, base=2)
mi_perf = mutual_information(perfect, base=2)

print(f"\nНезависимые X,Y:   MI = {mi_ind:.4f} бит  (знаем X → ничего о Y)")
print(f"Зависимые X,Y:     MI = {mi_dep:.4f} бит  (знаем X → больше о Y)")
print(f"Полностью dependent: MI = {mi_perf:.4f} бит  (знаем X → всё о Y)")


# ============================================================
#  Демо 7: Label smoothing
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Label smoothing")
print("=" * 55)

num_classes = 4
true_class = 2

hard = [0.0] * num_classes
hard[true_class] = 1.0

soft = label_smoothing(true_class, num_classes, epsilon=0.1)

print(f"\nHard target: {hard}  (энтропия = {entropy(hard, base=2):.4f})")
print(f"Soft target: {[f'{x:.3f}' for x in soft]}  (энтропия = {entropy(soft, base=2):.4f})")
print(f"\nLabel smoothing увеличивает энтропию целевого распределения → регуляризация")


# ============================================================
#  Демо 8: Почему CE — лучшая loss для классификации
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Три взгляда на cross-entropy")
print("=" * 55)

logits = [5.0, 2.0, 0.5]
true_class = 1

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"\nЛогиты: {logits}, правильный класс: {true_class}")
print(f"Softmax: {[f'{p:.4f}' for p in probs]}")
print(f"\n1. Теория информации:CE = {loss:.4f} нат (среднее удивление)")
print(f"2. Максимальное правдоподобие: NLL = {loss:.4f} (максимизируем P(data))")
print(f"3. Градиент: dL/dz_i = softmax(z_i) - one_hot(i)")
