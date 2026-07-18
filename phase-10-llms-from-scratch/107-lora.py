"""
LoRA: Low-Rank Adaptation of Large Language Models
===================================================

Basics of LoRA — Low-rank adaptation for efficient fine-tuning.

Covers:
1. Low-rank factorization of weight matrices
2. Freezing base weights, training only LoRA parameters
3. Comparison: full fine-tuning vs LoRA

All implementations are self-contained (no numpy/torch/transformers/peft).
"""

import random
import math

random.seed(42)


# ============================================================
# Matrix operations (pure Python)
# ============================================================

def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def mat_create(rows, cols, fill=0.0):
    return [[fill] * cols for _ in range(rows)]


def mat_random(rows, cols, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_mul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = mat_zeros(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            s = 0.0
            for k in range(cols_A):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_transpose(A):
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def mat_frobenius_norm(A):
    s = 0.0
    for row in A:
        for val in row:
            s += val * val
    return math.sqrt(s)


def mat_fill(rows, cols, value):
    return [[value] * cols for _ in range(rows)]


def mat_copy(A):
    return [row[:] for row in A]


def mat_num_params(A):
    return len(A) * len(A[0])


def mat_shape(A):
    return (len(A), len(A[0]))


def mat_diag(d):
    n = len(d)
    M = mat_zeros(n, n)
    for i in range(n):
        M[i][i] = d[i]
    return M


def svd_approx(A, r):
    """
    Simplified SVD approximation using power iteration.
    Returns U_r (m x r), S_r (r x r), Vt_r (r x n) such that
    A ≈ U_r @ S_r @ Vt_r
    """
    m = len(A)
    n = len(A[0])
    r = min(r, min(m, n))

    # Compute A^T A (n x n) for right singular vectors
    At = mat_transpose(A)
    AtA = mat_mul(At, A)

    # Power iteration to find top-r eigenvectors
    V_cols = []
    for _ in range(r):
        # Random start
        v = [random.gauss(0, 1) for _ in range(n)]
        v_norm = math.sqrt(sum(x*x for x in v))
        v = [x / v_norm for x in v]

        eigenvalue = 0.0
        for _ in range(200):
            # w = AtA @ v
            w = [0.0] * n
            for i in range(n):
                for j in range(n):
                    w[i] += AtA[i][j] * v[j]

            eigenvalue = math.sqrt(sum(x*x for x in w))
            if eigenvalue < 1e-12:
                break
            v = [x / eigenvalue for x in w]

        V_cols.append((eigenvalue, v))

        # Deflate: AtA = AtA - eigenvalue * v * v^T
        for i in range(n):
            for j in range(n):
                AtA[i][j] -= eigenvalue * v[i] * v[j]

    # Sort by eigenvalue descending
    V_cols.sort(key=lambda x: -x[0])
    Vt_r = mat_zeros(r, n)
    S_r = mat_zeros(r, r)
    for k in range(r):
        S_r[k][k] = math.sqrt(max(V_cols[k][0], 0))
        Vt_r[k] = V_cols[k][1]

    # U = A @ V @ S^{-1}
    U_r = mat_zeros(m, r)
    for k in range(r):
        s_val = S_r[k][k]
        if s_val < 1e-12:
            continue
        # u_k = A @ v_k / s_k
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i][j] * Vt_r[k][j]
            U_r[i][k] = s / s_val

    return U_r, S_r, Vt_r


# ============================================================
# Demo 1: Low-rank factorization
# ============================================================

def demo_low_rank_factorization():
    print("=" * 60)
    print("DEMO 1: Low-rank factorization of weight matrices")
    print("=" * 60)
    print()
    print("Idea: any matrix W (d x d) with rank r can be factorized as")
    print("  W = A @ B")
    print("where A is (d x r) and B is (r x d), with r << d.")
    print("This saves memory: d^2 → 2*d*r parameters.")
    print()

    d = 8
    r = 2

    # Create a low-rank matrix
    A_true = mat_random(d, r, scale=1.0)
    B_true = mat_random(r, d, scale=1.0)
    W = mat_mul(A_true, B_true)

    print(f"Original matrix W: {d}x{d} = {d*d} parameters")
    print(f"Target rank: r = {r}")
    print(f"Factorized: A ({d}x{r}) @ B ({r}x{d}) = {2*d*r} parameters")
    print(f"Memory savings: {d*d} → {2*d*r} = {d*d / (2*d*r):.1f}x reduction")
    print()

    # Factorize W
    U_r, S_r, Vt_r = svd_approx(W, r)

    # Reconstruct
    W_approx = mat_mul(mat_mul(U_r, S_r), Vt_r)

    # Compute error
    err = mat_frobenius_norm(mat_sub(W, W_approx))
    frob_norm = mat_frobenius_norm(W)
    rel_err = err / (frob_norm + 1e-12)

    print("Original matrix W:")
    for i in range(d):
        row_str = " ".join(f"{W[i][j]:7.3f}" for j in range(d))
        print(f"  [{row_str}]")
    print()

    print(f"Reconstructed W ≈ U @ S @ Vt (rank-{r}):")
    for i in range(d):
        row_str = " ".join(f"{W_approx[i][j]:7.3f}" for j in range(d))
        print(f"  [{row_str}]")
    print()

    print(f"Frobenius norm of W:          {frob_norm:.4f}")
    print(f"Approximation error:          {err:.6f}")
    print(f"Relative error:               {rel_err:.6f}")
    print()

    # Show compression ratios for different ranks
    print("Compression at different ranks:")
    print(f"  {'Rank':<8} {'Params':<12} {'Compression':<15} {'Rel.Error':<12}")
    for rr in [1, 2, 3, 4]:
        U_r, S_r, Vt_r = svd_approx(W, rr)
        W_r = mat_mul(mat_mul(U_r, S_r), Vt_r)
        e = mat_frobenius_norm(mat_sub(W, W_r)) / (frob_norm + 1e-12)
        params = 2 * d * rr
        comp = d * d / params
        print(f"  {rr:<8} {params:<12} {comp:<15.2f}x {e:<12.6f}")
    print()


# ============================================================
# Demo 2: LoRA — adding adapters
# ============================================================

def demo_lora_adapters():
    print("=" * 60)
    print("DEMO 2: LoRA — adding low-rank adapters to frozen weights")
    print("=" * 60)
    print()
    print("LoRA: W' = W + alpha * A @ B")
    print("  W  — frozen pretrained weight (d x d)")
    print("  A  — low-rank adapter (d x r), random init")
    print("  B  — low-rank adapter (r x d), initialized to zeros")
    print("  alpha — scaling factor")
    print("  Only A and B are trained. W stays frozen.")
    print()

    d = 8
    r = 2
    alpha = 1.0

    # Frozen pretrained weights
    W = mat_random(d, d, scale=0.5)
    print(f"Frozen weight W: {d}x{d}")
    print(f"  Frobenius norm: {mat_frobenius_norm(W):.4f}")
    print()

    # LoRA adapter: A (random init), B (zeros init)
    A = mat_random(d, r, scale=0.01)
    B = mat_zeros(r, d)

    print(f"LoRA adapter A: {d}x{r} (random init, scale=0.01)")
    print(f"LoRA adapter B: {r}x{d} (zeros init)")
    print(f"  Initial ΔW = alpha * A @ B:")
    dW = mat_scale(mat_mul(A, B), alpha)
    print(f"  Frobenius norm of ΔW: {mat_frobenius_norm(dW):.6f}")
    print()

    # Simulate training: update A to learn a task
    print("Simulating training of LoRA adapter (5 steps)...")
    learning_rate = 0.1
    for step in range(5):
        # Simulate gradient: pretend we need ΔW to approximate target
        target_dW = mat_scale(W, 0.1)  # target: 10% of W
        error = mat_sub(target_dW, dW)
        grad_norm = mat_frobenius_norm(error)

        # Update A (simple gradient descent)
        for i in range(d):
            for j in range(r):
                s = 0.0
                for k in range(d):
                    s += error[i][k] * B[j][k]
                A[i][j] += learning_rate * s

        # Update B
        for i in range(r):
            for j in range(d):
                s = 0.0
                for k in range(d):
                    s += error[k][j] * A[k][i]
                B[i][j] += learning_rate * s

        dW = mat_scale(mat_mul(A, B), alpha)
        print(f"  Step {step+1}: ||error|| = {grad_norm:.4f}, "
              f"||ΔW|| = {mat_frobenius_norm(dW):.4f}")
    print()

    # Final state
    W_new = mat_add(W, dW)
    print(f"Updated weight W' = W + alpha * A @ B")
    print(f"  Frobenius norm of W:  {mat_frobenius_norm(W):.4f}")
    print(f"  Frobenius norm of ΔW: {mat_frobenius_norm(dW):.4f}")
    print(f"  Frobenius norm of W': {mat_frobenius_norm(W_new):.4f}")
    print()

    # Show that B initialized to zeros gives zero change
    B_zero = mat_zeros(r, d)
    dW_zero = mat_scale(mat_mul(A, B_zero), alpha)
    print(f"Key insight: B initialized to zeros → ΔW = 0 at start")
    print(f"  ||A @ B_zero|| = {mat_frobenius_norm(dW_zero):.6f}")
    print(f"This ensures the model starts from the pretrained behavior.")
    print()


# ============================================================
# Demo 3: Number of parameters
# ============================================================

def demo_parameter_count():
    print("=" * 60)
    print("DEMO 3: Parameter count — full fine-tuning vs LoRA")
    print("=" * 60)
    print()
    print("For a weight matrix of size d × d:")
    print("  Full fine-tuning: d² trainable parameters")
    print("  LoRA (rank r):    2 × d × r trainable parameters")
    print()

    # Compare for typical transformer hidden sizes
    configs = [
        ("GPT-2 small",  768,  12),
        ("GPT-2 medium", 1024, 16),
        ("GPT-2 large",  1280, 20),
        ("BERT-base",    768,  12),
        ("BERT-large",   1024, 16),
        ("LLaMA-7B",     4096, 32),
        ("LLaMA-13B",    5120, 40),
        ("LLaMA-65B",    8192, 64),
    ]

    r = 8  # typical LoRA rank

    print(f"  {'Model':<16} {'d':<8} {'Full FT':<15} {'LoRA r=8':<15} {'Ratio':<10}")
    print("  " + "-" * 64)
    for name, d, _ in configs:
        full = d * d
        lora = 2 * d * r
        ratio = full / lora
        print(f"  {name:<16} {d:<8} {full:>12,}   {lora:>12,}   {ratio:>7.1f}x")
    print()

    # Realistic: a single attention layer has 4 weight matrices (Q, K, V, O)
    d = 4096
    num_heads = 32
    head_dim = d // num_heads
    r = 8

    qkv_o_full = 4 * d * d
    qkv_o_lora = 4 * 2 * d * r

    print(f"Example: LLaMA-7B attention layer (d={d}, 4 matrices)")
    print(f"  Full fine-tuning: {qkv_o_full:>12,} params")
    print(f"  LoRA (r={r}):      {qkv_o_lora:>12,} params")
    print(f"  Ratio:             {qkv_o_full / qkv_o_lora:>12.1f}x")
    print()

    # FFN layer
    d_ff = 4 * d
    ffn_full = 2 * d * d_ff  # two weight matrices in FFN
    ffn_lora = 2 * 2 * d * r

    print(f"LLaMA-7B FFN layer (d={d}, d_ff={d_ff})")
    print(f"  Full fine-tuning: {ffn_full:>12,} params")
    print(f"  LoRA (r={r}):      {ffn_lora:>12,} params")
    print(f"  Ratio:             {ffn_full / ffn_lora:>12.1f}x")
    print()

    # Total for 32 layers
    n_layers = 32
    total_full = n_layers * (qkv_o_full + ffn_full)
    total_lora = n_layers * (qkv_o_lora + ffn_lora)

    print(f"Total across {n_layers} layers:")
    print(f"  Full fine-tuning: {total_full:>14,} params = {total_full/1e6:.1f}M")
    print(f"  LoRA (r={r}):      {total_lora:>14,} params = {total_lora/1e6:.1f}M")
    print(f"  Ratio:             {total_full / total_lora:>14.1f}x")
    print(f"  Savings:           {(1 - total_lora/total_full)*100:.1f}%")
    print()


# ============================================================
# Demo 4: Comparison — LoRA vs full fine-tuning
# ============================================================

def demo_comparison():
    print("=" * 60)
    print("DEMO 4: LoRA vs full fine-tuning — training simulation")
    print("=" * 60)
    print()
    print("Simulating a simple classification task:")
    print("  - Input: d-dimensional vector")
    print("  - Task: learn a weight matrix W_task")
    print("  - Compare: full fine-tuning vs LoRA")
    print()

    d = 16
    r = 4
    n_samples = 32
    alpha = 1.0
    epochs = 20
    lr = 0.01

    # Generate synthetic task
    random.seed(42)
    W_target = mat_random(d, d, scale=0.1)  # target weight to learn
    X = [mat_random(1, d, scale=1.0)[0] for _ in range(n_samples)]

    def forward(x, W):
        """Simple linear forward: y = W @ x"""
        result = [0.0] * d
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += W[i][j] * x[j]
            result[i] = s
        return result

    def loss(pred, target):
        s = 0.0
        for i in range(len(pred)):
            s += (pred[i] - target[i]) ** 2
        return s / len(pred)

    # Generate target outputs
    Y = [forward(x, W_target) for x in X]

    # --- Full fine-tuning ---
    random.seed(42)
    W_ft = mat_random(d, d, scale=0.01)
    losses_ft = []

    for epoch in range(epochs):
        total_loss = 0.0
        for idx in range(n_samples):
            x = X[idx]
            y_true = Y[idx]
            y_pred = forward(x, W_ft)
            l = loss(y_pred, y_true)
            total_loss += l

            # Compute gradient: dL/dW = 2/n * (y_pred - y_true) @ x^T
            for i in range(d):
                for j in range(d):
                    W_ft[i][j] -= lr * 2.0 / n_samples * (y_pred[i] - y_true[i]) * x[j]

        losses_ft.append(total_loss / n_samples)

    # --- LoRA ---
    random.seed(42)
    W_base = mat_random(d, d, scale=0.01)  # frozen base
    A_lora = mat_random(d, r, scale=0.01)
    B_lora = mat_zeros(r, d)
    losses_lora = []

    for epoch in range(epochs):
        total_loss = 0.0
        for idx in range(n_samples):
            x = X[idx]
            y_true = Y[idx]

            # W = W_base + alpha * A @ B
            dW = mat_scale(mat_mul(A_lora, B_lora), alpha)
            W_lora = mat_add(W_base, dW)
            y_pred = forward(x, W_lora)
            l = loss(y_pred, y_true)
            total_loss += l

            # Gradient w.r.t. A and B only (W_base frozen)
            error = [y_pred[i] - y_true[i] for i in range(d)]

            # dL/dA = 2/n * error @ (B @ x)^T
            Bx = [0.0] * r
            for i in range(r):
                for j in range(d):
                    Bx[i] += B_lora[i][j] * x[j]

            for i in range(d):
                for j in range(r):
                    A_lora[i][j] -= lr * 2.0 / n_samples * error[i] * Bx[j]

            # dL/dB = 2/n * (A^T @ error) @ x^T
            for i in range(r):
                At_e = 0.0
                for k in range(d):
                    At_e += A_lora[k][i] * error[k]
                for j in range(d):
                    B_lora[i][j] -= lr * 2.0 / n_samples * At_e * x[j]

        losses_lora.append(total_loss / n_samples)

    # --- Results ---
    print(f"Model dimension: d = {d}")
    print(f"LoRA rank: r = {r}")
    print(f"Samples: {n_samples}, Epochs: {epochs}, LR: {lr}")
    print()

    print("Training loss over epochs:")
    print(f"  {'Epoch':<8} {'Full FT':<15} {'LoRA':<15}")
    print("  " + "-" * 38)
    for e in range(0, epochs, 5):
        print(f"  {e+1:<8} {losses_ft[e]:<15.6f} {losses_lora[e]:<15.6f}")
    print(f"  {epochs:<8} {losses_ft[-1]:<15.6f} {losses_lora[-1]:<15.6f}")
    print()

    # Parameter comparison
    full_params = d * d
    lora_params = 2 * d * r
    print(f"Parameter count:")
    print(f"  Full fine-tuning: {full_params:>8} trainable params")
    print(f"  LoRA (r={r}):      {lora_params:>8} trainable params")
    print(f"  Ratio:            {full_params / lora_params:.1f}x fewer params")
    print()

    # Final quality
    print(f"Final loss:")
    print(f"  Full fine-tuning: {losses_ft[-1]:.6f}")
    print(f"  LoRA:             {losses_lora[-1]:.6f}")
    print()

    # Quality achieved at parameter budget of LoRA
    print(f"Key insight: LoRA achieves comparable loss with")
    print(f"  {full_params / lora_params:.1f}x fewer trainable parameters!")
    print()
    print("Advantages of LoRA:")
    print("  1. Drastically fewer parameters to train and store")
    print("  2. Original weights remain frozen (no catastrophic forgetting)")
    print("  3. Easy to switch adapters for different tasks")
    print("  4. Inference cost is the same as original model (A @ B can be merged)")
    print("  5. Lower memory footprint during training")
    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print(" LoRA: Low-Rank Adaptation of Large Language Models")
    print("=" * 60)
    print()
    print("LoRA freezes the pretrained model weights and injects")
    print("trainable low-rank decomposition matrices into each")
    print("layer of the Transformer architecture.")
    print()

    demo_low_rank_factorization()
    print()
    demo_lora_adapters()
    print()
    demo_parameter_count()
    print()
    demo_comparison()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("LoRA key ideas:")
    print("  1. Low-rank factorization: W ≈ A @ B (r << d)")
    print("  2. Freeze base weights W, only train A and B")
    print("  3. Dramatic parameter reduction (10-100x)")
    print("  4. Competitive with full fine-tuning")
    print("  5. Plug-and-play: swap adapters for different tasks")
    print()
