import math
import random

random.seed(42)


# ============================================================
#  Базовые функции
# ============================================================

def softmax(logits, temperature=1.0):
    scaled = [z / temperature for z in logits]
    max_l = max(scaled)
    exps = [math.exp(z - max_l) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def sample_from_probs(probs):
    r = random.random()
    cumsum = 0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1

def mean(x):
    return sum(x) / len(x)

def std(x):
    m = mean(x)
    return math.sqrt(sum((xi - m)**2 for xi in x) / len(x))


# ============================================================
#  Inverse CDF
# ============================================================

def sample_exponential(lam):
    u = random.random()
    return -math.log(u) / lam

def sample_uniform(a, b):
    return a + (b - a) * random.random()


# ============================================================
#  Rejection Sampling
# ============================================================

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x


# ============================================================
#  Importance Sampling
# ============================================================

def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n


# ============================================================
#  Monte Carlo Pi
# ============================================================

def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n


# ============================================================
#  Metropolis-Hastings
# ============================================================

def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples


# ============================================================
#  Gibbs Sampling
# ============================================================

def gibbs_sampling_2d(cond_x_given_y, cond_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = cond_x_given_y(y)
        y = cond_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples


# ============================================================
#  Temperature Sampling
# ============================================================

def temperature_sample(logits, temperature):
    probs = softmax(logits, temperature)
    return sample_from_probs(probs)


# ============================================================
#  Top-k Sampling
# ============================================================

def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]


# ============================================================
#  Top-p (Nucleus) Sampling
# ============================================================

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]


# ============================================================
#  Reparameterization Trick
# ============================================================

def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon


# ============================================================
#  Gumbel-Softmax
# ============================================================

def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(max(p, 1e-10)) + gumbel_sample() for p in logits]
    return softmax(gumbels, temperature)


# ============================================================
#  Демо 1: Inverse CDF — Exponential
# ============================================================

print("=" * 55)
print("ДЕМО 1: Inverse CDF — Exponential")
print("=" * 55)

lam = 2.0
samples = [sample_exponential(lam) for _ in range(10000)]
print(f"\nExponential(λ={lam})")
print(f"  Теоретическое среднее: {1/lam:.4f}")
print(f"  Сэмплированное:        {mean(samples):.4f}")
print(f"  Теоретическое σ:       {1/lam:.4f}")
print(f"  Сэмплированное σ:      {std(samples):.4f}")


# ============================================================
#  Демо 2: Rejection Sampling
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Rejection Sampling")
print("=" * 55)

# Сэмплируем из полукруга
def target_circle(x):
    if -1 <= x <= 1:
        return math.sqrt(max(0, 1 - x**2))
    return 0

def proposal_uniform():
    return random.uniform(-1, 1)

def proposal_pdf(x):
    return 0.5 if -1 <= x <= 1 else 0

M = 1.0  # max of target / max of proposal

accepted = []
attempts = 0
while len(accepted) < 5000:
    x = proposal_uniform()
    u = random.random()
    attempts += 1
    if u < target_circle(x) / (M * proposal_pdf(x)):
        accepted.append(x)

acceptance_rate = len(accepted) / attempts
print(f"\nПолукруг (rejection sampling)")
print(f"  Сэмплов: {len(accepted)}")
print(f"  Попыток: {attempts}")
print(f"  Acceptance rate: {acceptance_rate:.4f}")
print(f"  Среднее: {mean(accepted):.4f} (ожидается ~0)")
print(f"  σ: {std(accepted):.4f}")


# ============================================================
#  Демо 3: Importance Sampling
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Importance Sampling")
print("=" * 55)

# Оцениваем E[X²] где X ~ N(0,1), сэмплируя из Uniform(-3,3)
f = lambda x: x**2
target = lambda x: math.exp(-x**2/2) / math.sqrt(2*math.pi)
proposal = lambda: random.uniform(-3, 3)
proposal_pdf_fn = lambda x: 1/6 if -3 <= x <= 3 else 0

estimates = [importance_sampling_estimate(f, target, proposal_pdf_fn, proposal, 1000)
             for _ in range(100)]

true_value = 1.0  # E[X²] для N(0,1) = 1
print(f"\nE[X²] где X ~ N(0,1)")
print(f"  Истинное: {true_value}")
print(f"  IS оценка: {mean(estimates):.4f} ± {std(estimates):.4f}")


# ============================================================
#  Демо 4: Monte Carlo Pi
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Monte Carlo оценка π")
print("=" * 55)

for n in [1000, 10000, 100000, 1000000]:
    pi_est = monte_carlo_pi(n)
    error = abs(pi_est - math.pi)
    print(f"  N={n:>8}: π ≈ {pi_est:.6f}  (ошибка: {error:.6f})")


# ============================================================
#  Демо 5: Metropolis-Hastings
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Metropolis-Hastings MCMC")
print("=" * 55)

# Сэмплируем из смеси двух гауссиан
def target_log_pdf(x):
    # 0.5 * N(-3, 1) + 0.5 * N(3, 1)
    log_p1 = -0.5 * (x + 3)**2
    log_p2 = -0.5 * (x - 3)**2
    log_sum = max(log_p1, log_p2) + math.log(1 + math.exp(-abs(log_p1 - log_p2)))
    return log_sum

def proposal_sample(x):
    return x + random.gauss(0, 0.5)

def proposal_log_pdf(x_from, x_to):
    return -0.5 * ((x_to - x_from) / 0.5)**2

samples_mh = metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, 0.0, 5000, 1000)

print(f"\nСмесь N(-3,1) + N(3,1)")
print(f"  Сэмплов: {len(samples_mh)}")
print(f"  Среднее: {mean(samples_mh):.2f} (ожидается ~0)")
print(f"  Мин: {min(samples_mh):.2f}, Макс: {max(samples_mh):.2f}")
print(f"  < 0: {sum(1 for s in samples_mh if s < 0)/len(samples_mh):.0%} (ожидается ~50%)")


# ============================================================
#  Демо 6: Gibbs Sampling
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Gibbs Sampling 2D")
print("=" * 55)

# 2D нормальное распределение с корреляцией
def cond_x_given_y(y):
    return random.gauss(0.5 * y, 0.866)  # μ = 0.5y, σ = √(1-0.25)

def cond_y_given_x(x):
    return random.gauss(0.5 * x, 0.866)

samples_gibbs = gibbs_sampling_2d(cond_x_given_y, cond_y_given_x, 0, 0, 5000, 1000)
xs = [s[0] for s in samples_gibbs]
ys = [s[1] for s in samples_gibbs]

print(f"\n2D нормальное с корреляцией 0.5")
print(f"  Сэмплов: {len(samples_gibbs)}")
print(f"  Среднее X: {mean(xs):.3f}, Среднее Y: {mean(ys):.3f}")
print(f"  Corr(X,Y): {sum((x-mean(xs))*(y-mean(ys)) for x,y in samples_gibbs)/len(samples_gibbs)/(std(xs)*std(ys)):.3f}")


# ============================================================
#  Демо 7: Temperature Sampling
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Temperature Sampling")
print("=" * 55)

logits = [3.0, 2.0, 1.0, 0.5, 0.1]
tokens = ["кот", "пёс", "рыба", "птица", "змея"]

print(f"\nЛогиты: {logits}")
print(f"Токены: {tokens}\n")

for T in [0.3, 0.7, 1.0, 1.5, 2.0]:
    probs = softmax(logits, T)
    sampled_counts = [0] * len(logits)
    for _ in range(1000):
        idx = temperature_sample(logits, T)
        sampled_counts[idx] += 1
    print(f"  T={T:.1f}: {[f'{t}:{c/10:.0%}' for t, c in zip(tokens, sampled_counts)]}")


# ============================================================
#  Демо 8: Top-k vs Top-p
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Top-k vs Top-p Sampling")
print("=" * 55)

logits_long = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]
tokens_long = [f"t{i}" for i in range(10)]

print(f"\nЛогиты: {logits_long}")

# Top-k
print(f"\nTop-k (k=3):")
for _ in range(3):
    idx = top_k_sample(logits_long, 3)
    print(f"  → {tokens_long[idx]}")

# Top-p
print(f"\nTop-p (p=0.8):")
for _ in range(3):
    idx = top_p_sample(logits_long, 0.8)
    print(f"  → {tokens_long[idx]}")


# ============================================================
#  Демо 9: Reparameterization Trick
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 9: Reparameterization Trick")
print("=" * 55)

mu, sigma = 2.0, 0.5
print(f"\nμ = {mu}, σ = {sigma}")
print(f"\nСтандартный сэмпл: z ~ N({mu}, {sigma}²)")
print(f"  → Градиент d/dμ: ??? (не дифференцируемо)")

print(f"\nTrick: z = μ + σ × ε, ε ~ N(0,1)")
samples_rp = [reparam_sample(mu, sigma) for _ in range(1000)]
print(f"  Среднее: {mean(samples_rp):.3f} (ожидается {mu})")
print(f"  σ: {std(samples_rp):.3f} (ожидается {sigma})")
print(f"  Градиент d/dμ = 1, d/dσ = ε  ← дифференцируемо!")


# ============================================================
#  Демо 10: Gumbel-Softmax
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 10: Gumbel-Softmax")
print("=" * 55)

probs = [0.7, 0.2, 0.1]
print(f"\nВероятности: {probs}")

for tau in [0.1, 0.5, 1.0, 2.0]:
    gs_samples = [gumbel_softmax(probs, tau) for _ in range(1000)]
    avg = [sum(s[i] for s in gs_samples)/1000 for i in range(3)]
    print(f"  τ={tau:.1f}: среднее = {[f'{a:.3f}' for a in avg]}")

print(f"\n  τ → 0: approaching [1, 0, 0] (hard categorical)")
print(f"  τ → ∞: approaching [0.33, 0.33, 0.33] (uniform)")
