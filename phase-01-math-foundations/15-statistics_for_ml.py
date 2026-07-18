import math
import random

random.seed(42)


# ============================================================
#  Описательная статистика
# ============================================================

def mean(x):
    return sum(x) / len(x)

def median(x):
    s = sorted(x)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def mode(x):
    from collections import Counter
    counts = Counter(x)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]

def variance(x, population=True):
    m = mean(x)
    n = len(x)
    denom = n if population else n - 1
    return sum((xi - m) ** 2 for xi in x) / denom

def std(x, population=True):
    return math.sqrt(variance(x, population))

def percentile(x, p):
    s = sorted(x)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])

def iqr(x):
    return percentile(x, 75) - percentile(x, 25)


# ============================================================
#  Корреляция
# ============================================================

def pearson_correlation(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)

def spearman_correlation(x, y):
    def rank(data):
        sorted_idx = sorted(range(len(data)), key=lambda i: data[i])
        ranks = [0] * len(data)
        for rank_val, idx in enumerate(sorted_idx, 1):
            ranks[idx] = rank_val
        return ranks

    rx = rank(x)
    ry = rank(y)
    return pearson_correlation(rx, ry)

def covariance_matrix(data):
    n = len(data)
    d = len(data[0])
    means = [mean([data[i][j] for i in range(n)]) for j in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum((data[k][i] - means[i]) * (data[k][j] - means[j])
                           for k in range(n)) / n
    return cov


# ============================================================
#  t-test
# ============================================================

def t_test_one_sample(x, mu0):
    n = len(x)
    mx = mean(x)
    s = std(x, population=False)
    t = (mx - mu0) / (s / math.sqrt(n))
    return t, n - 1

def t_test_two_sample(x, y):
    n1, n2 = len(x), len(y)
    mx, my = mean(x), mean(y)
    v1, v2 = variance(x, population=False), variance(y, population=False)
    se = math.sqrt(v1/n1 + v2/n2)
    t = (mx - my) / se if se > 0 else 0
    return t

def t_test_paired(x, y):
    diffs = [xi - yi for xi, yi in zip(x, y)]
    return t_test_one_sample(diffs, 0)


# ============================================================
#  Chi-squared test
# ============================================================

def chi_squared_test(observed, expected):
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return chi2


# ============================================================
#  Bootstrap
# ============================================================

def bootstrap_ci(data, stat_fn, n_bootstrap=10000, ci=0.95):
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    alpha = (1 - ci) / 2
    lower = stats[int(alpha * n_bootstrap)]
    upper = stats[int((1 - alpha) * n_bootstrap)]
    return lower, upper, mean(stats)


# ============================================================
#  Cohen's d
# ============================================================

def cohens_d(x, y):
    mx, my = mean(x), mean(y)
    nx, ny = len(x), len(y)
    vx, vy = variance(x, population=False), variance(y, population=False)
    pooled_std = math.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    if pooled_std == 0:
        return 0
    return (mx - my) / pooled_std


# ============================================================
#  A/B тест симуляция
# ============================================================

def simulate_ab_test(n_samples=1000, true_diff=0.02, alpha=0.05):
    """Симуляция A/B теста с проверкой Type I и Type II ошибок"""
    # H0: разницы нет (обе группы = 0.5)
    # H1: группа B лучше на true_diff

    false_positives = 0
    true_positives = 0
    n_trials = 500

    for _ in range(n_trials):
        # Группа A: accuracy ~ 0.5
        a_scores = [1 if random.random() < 0.5 else 0 for _ in range(n_samples)]
        # Группа B: accuracy ~ 0.5 + true_diff
        b_scores = [1 if random.random() < (0.5 + true_diff) else 0 for _ in range(n_samples)]

        a_mean = mean(a_scores)
        b_mean = mean(b_scores)

        # Простой z-test
        p_a = mean(a_scores)
        p_b = mean(b_scores)
        p_pool = (sum(a_scores) + sum(b_scores)) / (2 * n_samples)
        se = math.sqrt(2 * p_pool * (1 - p_pool) / n_samples)
        z = (b_mean - a_mean) / se if se > 0 else 0

        # Приблизительный p-value (используем порог z > 1.96)
        if abs(z) > 1.96:
            if true_diff == 0:
                false_positives += 1
            else:
                true_positives += 1

    if true_diff == 0:
        return false_positives / n_trials, "Type I error rate"
    else:
        return true_positives / n_trials, "Power (1 - Type II)"


# ============================================================
#  Демо 1: Описательная статистика
# ============================================================

print("=" * 55)
print("ДЕМО 1: Описательная статистика")
print("=" * 55)

data = [2, 4, 4, 4, 5, 5, 7, 9, 10, 100]
print(f"\nДанные: {data}")
print(f"  Среднее:    {mean(data):.2f}")
print(f"  Медиана:    {median(data):.2f}")
print(f"  Мода:       {mode(data)}")
print(f"  σ (pop):    {std(data, population=True):.2f}")
print(f"  σ (sample): {std(data, population=False):.2f}")
print(f"  Дисперсия:  {variance(data):.2f}")
print(f"  Q1:         {percentile(data, 25):.2f}")
print(f"  Q3:         {percentile(data, 75):.2f}")
print(f"  IQR:        {iqr(data):.2f}")
print(f"\n→ Среднее >> медианы (14.5 vs 5.0) — правый хвост (выброс 100)")


# ============================================================
#  Демо 2: Корреляция
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Корреляция")
print("=" * 55)

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_lin = [2, 4, 5, 8, 9, 11, 13, 16, 17, 20]
y_nonlin = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

print(f"\nx:    {x}")
print(f"y_lin:    {y_lin} (линейная)")
print(f"y_nonlin: {y_nonlin[:5]}... (x², нелинейная)")
print(f"\nPearson(x, y_lin):    {pearson_correlation(x, y_lin):.4f}")
print(f"Pearson(x, y_nonlin): {pearson_correlation(x, y_nonlin):.4f}")
print(f"Spearman(x, y_lin):   {spearman_correlation(x, y_lin):.4f}")
print(f"Spearman(x, y_nonlin):{spearman_correlation(x, y_nonlin):.4f}")
print(f"\n→ Pearson ловит линейную, Spearman — монотонную (x²)")


# ============================================================
#  Демо 3: Ковариационная матрица
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Ковариационная матрица")
print("=" * 55)

data_2d = [[1, 2], [2, 4], [3, 5], [4, 7], [5, 8]]
cov = covariance_matrix(data_2d)

print(f"\nДанные: {data_2d}")
print(f"Ковариационная матрица:")
print(f"  [{cov[0][0]:6.2f}  {cov[0][1]:6.2f}]")
print(f"  [{cov[1][0]:6.2f}  {cov[1][1]:6.2f}]")
print(f"\nCov(x,y) = {cov[0][1]:.2f} > 0 — положительная корреляция")


# ============================================================
#  Демо 4: t-test
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: t-test")
print("=" * 55)

# Одновыборочный t-test: среднее = 5?
sample = [4.8, 5.1, 4.9, 5.2, 5.0, 4.7, 5.3, 4.8, 5.1, 4.9]
t1, df1 = t_test_one_sample(sample, 5.0)
print(f"\nОдновыборочный: H0: μ = 5.0")
print(f"  Данные: {sample}")
print(f"  t = {t1:.4f}, df = {df1}")
print(f"  (|t| < 2 → не отвергаем H0)")

# Двухвыборочный: разные группы?
group_a = [85, 88, 90, 87, 92, 89, 86, 91]
group_b = [78, 82, 80, 79, 83, 81, 77, 84]
t2 = t_test_two_sample(group_a, group_b)
print(f"\nДвухвыборочный: H0: μ_A = μ_B")
print(f"  A: {group_a}, среднее = {mean(group_a):.1f}")
print(f"  B: {group_b}, среднее = {mean(group_b):.1f}")
print(f"  t = {t2:.4f}")
print(f"  (|t| >> 2 → отвергаем H0, группы разные)")


# ============================================================
#  Демо 5: Chi-squared test
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Chi-squared test")
print("=" * 55)

observed = [120, 80]
expected = [100, 100]
chi2 = chi_squared_test(observed, expected)

print(f"\nНаблюдаемые: {observed}")
print(f"Ожидаемые:   {expected}")
print(f"χ² = {chi2:.2f}")
print(f"→ (120-100)²/100 + (80-100)²/100 = 4 + 4 = 8")


# ============================================================
#  Демо 6: Bootstrap
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Bootstrap confidence interval")
print("=" * 55)

accuracies = [0.82, 0.85, 0.88, 0.84, 0.87, 0.83, 0.86, 0.89, 0.85, 0.88]
lower, upper, boot_mean = bootstrap_ci(accuracies, mean, n_bootstrap=5000)

print(f"\nТочечная оценка: {mean(accuracies):.4f}")
print(f"Bootstrap mean:  {boot_mean:.4f}")
print(f"95% CI:          [{lower:.4f}, {upper:.4f}]")
print(f"Ширина:          {upper - lower:.4f}")


# ============================================================
#  Демо 7: Cohen's d
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Cohen's d (размер эффекта)")
print("=" * 55)

model_a = [0.85, 0.87, 0.86, 0.88, 0.84, 0.87, 0.86, 0.85]
model_b = [0.89, 0.91, 0.90, 0.92, 0.88, 0.91, 0.90, 0.89]

d = cohens_d(model_a, model_b)
print(f"\nModel A: среднее = {mean(model_a):.4f}, σ = {std(model_a, False):.4f}")
print(f"Model B: среднее = {mean(model_b):.4f}, σ = {std(model_b, False):.4f}")
print(f"Cohen's d = {d:.4f}")

if abs(d) < 0.2:
    print("→ Маленький эффект")
elif abs(d) < 0.5:
    print("→ Средний эффект")
elif abs(d) < 0.8:
    print("→ Заметный эффект")
else:
    print("→ Большой эффект")


# ============================================================
#  Демо 8: Статистическая vs практическая значимость
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Статистическая vs практическая значимость")
print("=" * 55)

n_samples_list = [100, 1000, 10000, 100000]
true_diff = 0.001  # 0.1% разница

print(f"\nИстинная разница: {true_diff*100:.1f}%")
print(f"\n{'n':>10} {'p-value':>12} {'Significant?':>15}")
print("-" * 40)

for n in n_samples_list:
    a = [1 if random.random() < 0.5 else 0 for _ in range(n)]
    b = [1 if random.random() < (0.5 + true_diff) else 0 for _ in range(n)]

    p_a = mean(a)
    p_b = mean(b)
    p_pool = (sum(a) + sum(b)) / (2 * n)
    se = math.sqrt(2 * p_pool * (1 - p_pool) / n) if p_pool > 0 and p_pool < 1 else 1e-10
    z = (p_b - p_a) / se
    sig = "Да" if abs(z) > 1.96 else "Нет"

    print(f"  {n:>8}  {abs(z):>10.2f}  {sig:>15}")

print(f"\n→ При большом n даже 0.1% разница становится 'значимой'")


# ============================================================
#  Демо 9: A/B тест симуляция
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 9: A/B тест — Type I и Power")
print("=" * 55)

# H0 верна (разницы нет)
rate_i, label_i = simulate_ab_test(n_samples=500, true_diff=0)
print(f"\nH0 верна (разницы нет):  {label_i} = {rate_i:.2%} (ожидается ~5%)")

# H1 верна (разница 2%)
rate_ii, label_ii = simulate_ab_test(n_samples=500, true_diff=0.02)
print(f"H1 верна (разница 2%):   {label_ii} = {rate_ii:.2%}")

# Больше данных
rate_iii, label_iii = simulate_ab_test(n_samples=2000, true_diff=0.02)
print(f"H1 верна (n=2000):       {label_iii} = {rate_iii:.2%}")
print(f"\n→ Больше данных = больше Power (ловит реальные разницы)")


# ============================================================
#  Демо 10: Доверительный интервал — интерпретация
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 10: Доверительный интервал")
print("=" * 55)

print(f"\n95% CI = x̄ ± 1.96 × (s/√n)")
print(f"\nИнтерпретация:")
print(f"  Если повторить эксперимент 100 раз,")
print(f"  ~95 интервалов содержат истинное среднее.")
print(f"\nНЕ означает:")
print(f"  'Вероятность что среднее в интервале = 95%'")
print(f"\nПример:")
print(f"  x̄ = 85.3, s = 5.2, n = 30")
x_bar, s, n = 85.3, 5.2, 30
ci_low = x_bar - 1.96 * s / math.sqrt(n)
ci_high = x_bar + 1.96 * s / math.sqrt(n)
print(f"  95% CI = [{ci_low:.2f}, {ci_high:.2f}]")
