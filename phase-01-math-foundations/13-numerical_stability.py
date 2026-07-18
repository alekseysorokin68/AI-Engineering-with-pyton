import math
import struct

random_seed = 42


# ============================================================
#  Stable Softmax
# ============================================================

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ============================================================
#  Log-Sum-Exp
# ============================================================

def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))


# ============================================================
#  Stable Cross-Entropy
# ============================================================

def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob


# ============================================================
#  Gradient Checking
# ============================================================

def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    all_ok = True
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        if rel_error >= tolerance:
            all_ok = False
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")
    return all_ok


# ============================================================
#  Gradient Clipping
# ============================================================

def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

def clip_by_value(gradients, max_val):
    return [max(-max_val, min(max_val, g)) for g in gradients]


# ============================================================
#  NaN/Inf Detection
# ============================================================

def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"  WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True


# ============================================================
#  Mixed Precision Simulation
# ============================================================

def float32_to_float16(x):
    try:
        packed = struct.pack('f', x)
        f32 = struct.unpack('f', packed)[0]
        packed16 = struct.pack('e', f32)
        return struct.unpack('e', packed16)[0]
    except OverflowError:
        return float('inf') if x > 0 else float('-inf')

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]


# ============================================================
#  Catastrophic Cancellation — Welford's Algorithm
# ============================================================

def variance_naive(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean)**2 for x in data) / n

def variance_welford(data):
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    return M2 / n if n > 1 else 0.0


# ============================================================
#  Демо 1: Floating point precision
# ============================================================

print("=" * 55)
print("ДЕМО 1: Floating point precision")
print("=" * 55)

print(f"\n0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Разница: {(0.1 + 0.2) - 0.3:.2e}")

# Machine epsilon
eps = 1.0
while 1.0 + eps != 1.0:
    eps /= 2
eps *= 2
print(f"\nMachine epsilon (float32): ~{eps:.2e}")
print(f"numpy.finfo:               ~1.19e-07")


# ============================================================
#  Демо 2: Overflow/Underflow
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Overflow и Underflow")
print("=" * 55)

print(f"\nexp(88.7)  = {math.exp(88.7):.2e}  (предел float32)")
try:
    print(f"exp(89.0)  = {math.exp(89.0):.2e}  (overflow!)")
except OverflowError:
    print(f"exp(89.0)  = inf  (overflow!)")

print(f"exp(-87.3) = {math.exp(-87.3):.2e}  (предел underflow)")
print(f"exp(-104)  = {math.exp(-104):.2e}  (underflow!)")
print(f"log(0.0)   = -inf")
print(f"log(1e-300)= {math.log(1e-300):.2f}")


# ============================================================
#  Демо 3: Stable Softmax
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Stable Softmax")
print("=" * 55)

safe = [2.0, 1.0, 0.1]
print(f"\nБезопасные логиты: {safe}")
print(f"  Naive:  {[f'{p:.4f}' for p in softmax_naive(safe)]}")
print(f"  Stable: {[f'{p:.4f}' for p in softmax_stable(safe)]}")

dangerous = [100.0, 101.0, 102.0]
print(f"\nОпасные логиты: {dangerous}")
print(f"  Naive:  ← overflow → NaN")
print(f"  Stable: {[f'{p:.4f}' for p in softmax_stable(dangerous)]}")

extreme = [1000.0, 1001.0, 1002.0]
print(f"\nЭкстремальные: {extreme}")
print(f"  Stable: {[f'{p:.4f}' for p in softmax_stable(extreme)]}")


# ============================================================
#  Демо 4: Log-Sum-Exp
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Log-Sum-Exp")
print("=" * 55)

safe_vals = [1.0, 2.0, 3.0]
print(f"\nБезопасные: {safe_vals}")
print(f"  Naive:  {logsumexp_naive(safe_vals):.6f}")
print(f"  Stable: {logsumexp_stable(safe_vals):.6f}")

large_vals = [500.0, 501.0, 502.0]
print(f"\nБольшие: {large_vals}")
print(f"  Naive:  → overflow → inf")
print(f"  Stable: {logsumexp_stable(large_vals):.6f}")

negative_vals = [-1000.0, -1001.0, -1002.0]
print(f"\nОтрицательные: {negative_vals}")
print(f"  Naive:  → underflow → -inf")
print(f"  Stable: {logsumexp_stable(negative_vals):.6f}")


# ============================================================
#  Демо 5: Stable Cross-Entropy
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Stable Cross-Entropy")
print("=" * 55)

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"\nЛогиты: {logits}, правильный: {true_class}")
print(f"  Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"  Stable: {cross_entropy_stable(true_class, logits):.6f}")

large_logits = [100.0, 105.0, 99.0]
print(f"\nБольшие логиты: {large_logits}")
print(f"  Naive:  → overflow → NaN")
print(f"  Stable: {cross_entropy_stable(true_class, large_logits):.6f}")


# ============================================================
#  Демо 6: Gradient Checking
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Gradient Checking")
print("=" * 55)

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
print(f"\nf(x,y) = x² + 3xy + y³")
print(f"Точка: ({point[0]}, {point[1]})")
print(f"f = {f(point):.2f}")

analytical = f_grad(point)
numerical = numerical_gradient(f, point)

print(f"\nГрадиент:")
ok = check_gradient(analytical, numerical)
print(f"\nРезультат: {'✓ все ок' if ok else '✗ есть ошибки'}")


# ============================================================
#  Демо 7: Gradient Clipping
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Gradient Clipping")
print("=" * 55)

grads = [10.0, 20.0, 30.0]
print(f"\nИсходные градиенты: {grads}")
print(f"  ||grad|| = {math.sqrt(sum(g**2 for g in grads)):.2f}")

clipped_norm = clip_by_norm(grads, max_norm=5.0)
print(f"\nClip by norm (max=5.0):")
print(f"  {clipped_norm}")
print(f"  ||grad|| = {math.sqrt(sum(g**2 for g in clipped_norm)):.2f}")

clipped_val = clip_by_value(grads, max_val=15.0)
print(f"\nClip by value (max=15.0):")
print(f"  {clipped_val}")


# ============================================================
#  Демо 8: NaN/Inf Detection
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: NaN/Inf Detection")
print("=" * 55)

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])


# ============================================================
#  Демо 9: Catastrophic Cancellation
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 9: Catastrophic Cancellation")
print("=" * 55)

data = [1000000.0, 1000001.0, 1000002.0]
true_var = 2.0 / 3.0

print(f"\nДанные: {data}")
print(f"Истинная дисперсия: {true_var:.6f}")

var_naive = variance_naive(data)
var_welford = variance_welford(data)

print(f"\nNaive formula (E[x²] - E[x]²): {var_naive:.6f}")
print(f"Welford algorithm:              {var_welford:.6f}")
print(f"\nОшибка naive:   {abs(var_naive - true_var):.2e}")
print(f"Ошибка Welford: {abs(var_welford - true_var):.2e}")


# ============================================================
#  Демо 10: Mixed Precision Simulation
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 10: Mixed Precision Simulation")
print("=" * 55)

values = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 60000.0, 100000.0]
print(f"\n{'Значение':>12} {'float32':>12} {'float16':>12} {'bfloat16':>12}")
print("-" * 52)

for v in values:
    f16 = float32_to_float16(v)
    bf16 = simulate_bfloat16(v)
    f16_str = f"{f16:.4g}" if not math.isinf(f16) else "inf"
    bf16_str = f"{bf16:.4g}" if not math.isinf(bf16) else "inf"
    print(f"{v:>12.1f} {v:>12.4g} {f16_str:>12} {bf16_str:>12}")

print(f"\nfloat16: max=65504, precision ~3 digits")
print(f"bfloat16: max=3.4e38, precision ~2 digits")
print(f"→ bfloat16 лучше для обучения (range > precision)")
