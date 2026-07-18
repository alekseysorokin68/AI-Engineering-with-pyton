import math
import random

random.seed(42)


# ============================================================
#  Базовые функции
# ============================================================

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b


# ============================================================
#  PMF и PDF — распределения с нуля
# ============================================================

def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


# ============================================================
#  Мат. ожидание и дисперсия
# ============================================================

def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


# ============================================================
#  Семплирование
# ============================================================

def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples


# ============================================================
#  Softmax и Log-softmax
# ============================================================

def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


# ============================================================
#  Центральная предельная теорема
# ============================================================

def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages


# ============================================================
#  Демо 1: Условная вероятность
# ============================================================

print("=" * 55)
print("ДЕМО 1: Условная вероятность")
print("=" * 55)

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}  (ожидается 1/3 ≈ 0.3333)")


# ============================================================
#  Демо 2: PMF распределения
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: PMF — Bernoulli, Poisson, Categorical")
print("=" * 55)

# Bernoulli
p = 0.3
print(f"\nBernoulli(p={p}):")
for k in [0, 1]:
    print(f"  P(X={k}) = {bernoulli_pmf(k, p):.4f}")

# Poisson
lam = 3.0
print(f"\nPoisson(λ={lam}):")
for k in range(7):
    print(f"  P(X={k}) = {poisson_pmf(k, lam):.4f}")

# Categorical
probs = [0.7, 0.2, 0.1]
labels = ["кошка", "собака", "птица"]
print(f"\nCategorical:")
for i, (label, prob) in enumerate(zip(labels, probs)):
    print(f"  P({label}) = {prob:.4f}")


# ============================================================
#  Демо 3: Нормальное распределение (PDF)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Нормальное распределение — PDF")
print("=" * 55)

mu, sigma = 0, 1
print(f"\nNormal(μ={mu}, σ={sigma}):")
print(f"  P(X=0) = {normal_pdf(0, mu, sigma):.4f}  (максимум)")
print(f"  P(X=1) = {normal_pdf(1, mu, sigma):.4f}")
print(f"  P(X=2) = {normal_pdf(2, mu, sigma):.4f}")
print(f"  P(X=3) = {normal_pdf(3, mu, sigma):.4f}  (почти 0)")

# Правило 3 сигм
print(f"\nПравило 3 сигм (μ=0, σ=1):")
print(f"  68% данных: P(-1 < X < 1) ≈ 0.6827")
print(f"  95% данных: P(-2 < X < 2) ≈ 0.9545")
print(f"  99.7% данных: P(-3 < X < 3) ≈ 0.9973")


# ============================================================
#  Демо 4: Мат. ожидание и дисперсия
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Мат. ожидание и дисперсия")
print("=" * 55)

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Кубик: E[X] = {mu:.4f}, Var(X) = {var:.4f}, σ = {var**0.5:.4f}")

# Бернулли
bern_values = [0, 1]
p = 0.3
bern_probs = [1-p, p]
mu_b = expected_value(bern_values, bern_probs)
var_b = variance(bern_values, bern_probs)
print(f"Bernoulli(p=0.3): E[X] = {mu_b:.4f}, Var(X) = {var_b:.4f}")


# ============================================================
#  Демо 5: Семплирование
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Семплирование из распределений")
print("=" * 55)

# Bernoulli
samples_b = sample_bernoulli(0.3, 1000)
print(f"\nBernoulli(0.3), 1000 сэмплов:")
print(f"  Среднее = {sum(samples_b)/len(samples_b):.3f}  (ожидается 0.3)")

# Categorical
samples_c = sample_categorical([0.7, 0.2, 0.1], 1000)
print(f"\nCategorical([0.7, 0.2, 0.1]), 1000 сэмплов:")
for i, label in enumerate(labels):
    count = samples_c.count(i)
    print(f"  {label}: {count/len(samples_c):.3f}  (ожидается {[0.7, 0.2, 0.1][i]:.3f})")

# Normal
samples_n = sample_normal(0, 1, 10000)
mean_n = sum(samples_n) / len(samples_n)
var_n = sum((x - mean_n)**2 for x in samples_n) / len(samples_n)
print(f"\nNormal(0,1), 10000 сэмплов (Box-Muller):")
print(f"  Среднее = {mean_n:.4f}  (ожидается 0)")
print(f"  Дисперсия = {var_n:.4f}  (ожидается 1)")


# ============================================================
#  Демо 6: Softmax и Cross-entropy
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Softmax и Cross-entropy loss")
print("=" * 55)

logits = [2.0, 1.0, 0.1]
probs = softmax(logits)
log_probs = log_softmax(logits)

print(f"\nЛогиты:  {logits}")
print(f"Softmax: {[f'{p:.4f}' for p in probs]}")
print(f"Сумма:   {sum(probs):.6f}  (должна быть 1.0)")
print(f"Log-softmax: {[f'{lp:.4f}' for lp in log_probs]}")

# Cross-entropy для каждого класса
print(f"\nCross-entropy loss (правильный класс = i):")
for i in range(3):
    loss = cross_entropy_loss(logits, i)
    print(f"  класс {i}: L = -log_softmax[{i}] = {loss:.4f}")


# ============================================================
#  Демо 7: Центральная предельная теорема
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Центральная предельная теорема")
print("=" * 55)

# Равномерное распределение
uniform_avg = demonstrate_clt(lambda: random.random(), n_samples=30, n_averages=10000)
mean_u = sum(uniform_avg) / len(uniform_avg)
var_u = sum((x - mean_u)**2 for x in uniform_avg) / len(uniform_avg)
print(f"\nРавномерное U(0,1):")
print(f"  Среднее 30 сэмплов, 10000 усреднений:")
print(f"    E[X̄] = {mean_u:.4f}  (ожидается 0.5)")
print(f"    Var(X̄) = {var_u:.6f}  (ожидается ~0.0083 = 1/(12*30))")

# Экспоненциальное распределение
exp_avg = demonstrate_clt(lambda: -math.log(random.random()), n_samples=30, n_averages=10000)
mean_e = sum(exp_avg) / len(exp_avg)
var_e = sum((x - mean_e)**2 for x in exp_avg) / len(exp_avg)
print(f"\nЭкспоненциальное Exp(1):")
print(f"  Среднее 30 сэмплов, 10000 усреднений:")
print(f"    E[X̄] = {mean_e:.4f}  (ожидается 1.0)")
print(f"    Var(X̄) = {var_e:.6f}  (ожидается ~0.0333 = 1/30)")


# ============================================================
#  Демо 8: Логарифмические вероятности
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Почему лог-вероятности важны")
print("=" * 55)

# Произведение вероятностей vs сумма логарифмов
probs = [0.01, 0.003, 0.02, 0.05, 0.1]
product = 1
log_sum = 0
for p in probs:
    product *= p
    log_sum += math.log(p)

print(f"Вероятности: {probs}")
print(f"Произведение P:  {product:.2e}  (underflow!)")
print(f"Сумма log P:     {log_sum:.4f}")
print(f"exp(log сумма):  {math.exp(log_sum):.2e}  (то же самое, но численно стабильно)")

# Длинная последовательность
print(f"\n50 слов, P=0.01 каждое:")
product_50 = 0.01 ** 50
log_sum_50 = 50 * math.log(0.01)
print(f"  P(предложение) = {product_50:.2e}  (underflow!)")
print(f"  log P = {log_sum_50:.4f}  (конечное число)")
