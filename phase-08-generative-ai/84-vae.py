"""
84 - Variational Autoencoder (VAE)
==================================
Самодостаточная реализация VAE на чистом Python (numpy/torch/tensorflow НЕ используются).

Ключевые компоненты:
  1. Reparameterization trick — z = mu + sigma * epsilon
  2. ELBO loss = Reconstruction loss + KL divergence
  3. Генерация новых данных из latent space (sampling)

Демонстрации:
  Demo 1: Reparameterization trick
  Demo 2: ELBO loss computation
  Demo 3: Генерация данных из latent space
  Demo 4: Сравнение с обычным Autoencoder
"""

import math
import random

random.seed(42)

# ============================================================
# Математические утилиты (纯 Python, без зависимостей)
# ============================================================

def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def mat_random(rows, cols, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def mat_mul(A, B):
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k == k2
    result = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            result[i][j] = s
    return result

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def vec_to_col(v):
    return [[x] for x in v]

def col_to_vec(M):
    return [row[0] for row in M]

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def sigmoid_vec(v):
    return [sigmoid(x) for x in v]

def tanh_vec(v):
    return [math.tanh(x) for x in v]

def elementwise_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def elementwise_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def vec_scale(v, s):
    return [x * s for x in v]

def vec_sum(v):
    return sum(v)

def vec_concat(*vecs):
    r = []
    for v in vecs:
        r.extend(v)
    return r

def format_vec(v, decimals=4):
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in v) + "]"

def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
# Linear слой с сохранением activations для backward
# ============================================================

class LinearLayer:
    def __init__(self, in_dim, out_dim):
        scale = math.sqrt(2.0 / in_dim)
        self.W = mat_random(in_dim, out_dim, scale)
        self.b = [0.0] * out_dim
        self.dW = mat_zeros(in_dim, out_dim)
        self.db = [0.0] * out_dim

    def forward(self, x):
        self._input = list(x)
        out = mat_mul([x], self.W)[0]
        return elementwise_add(out, self.b)

    def backward(self, grad_out):
        x = self._input
        x_col = vec_to_col(x)
        g_row = [grad_out]
        dW = mat_mul(x_col, g_row)
        for i in range(len(self.dW)):
            for j in range(len(self.dW[0])):
                self.dW[i][j] += dW[i][j]
        for j in range(len(self.b)):
            self.db[j] += grad_out[j]
        Wt = mat_transpose(self.W)
        return mat_mul([grad_out], Wt)[0]

    def zero_grad(self):
        for i in range(len(self.dW)):
            for j in range(len(self.dW[0])):
                self.dW[i][j] = 0.0
        self.db = [0.0] * len(self.b)

    def update(self, lr):
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] -= lr * self.dW[i][j]
        for j in range(len(self.b)):
            self.b[j] -= lr * self.db[j]


# ============================================================
# VAE
# ============================================================

class VAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.enc1 = LinearLayer(input_dim, hidden_dim)
        self.enc_mu = LinearLayer(hidden_dim, latent_dim)
        self.enc_logvar = LinearLayer(hidden_dim, latent_dim)
        self.dec1 = LinearLayer(latent_dim, hidden_dim)
        self.dec_out = LinearLayer(hidden_dim, input_dim)

    def forward(self, x):
        # Encoder
        h1_pre = self.enc1.forward(x)
        self._enc_h1 = tanh_vec(h1_pre)
        self._mu = self.enc_mu.forward(self._enc_h1)
        self._log_var = self.enc_logvar.forward(self._enc_h1)
        # Reparameterize
        self._epsilon = [random.gauss(0, 1) for _ in range(self.latent_dim)]
        self._sigma = [math.exp(0.5 * lv) for lv in self._log_var]
        self._z = [m + s * e for m, s, e in zip(self._mu, self._sigma, self._epsilon)]
        # Decoder
        h2_pre = self.dec1.forward(self._z)
        self._dec_h1 = tanh_vec(h2_pre)
        logits = self.dec_out.forward(self._dec_h1)
        recon = sigmoid_vec(logits)
        self._recon = recon
        return recon, self._mu, self._log_var, self._z

    def backward(self, x):
        """
        Backpropagation for ELBO = BCE + KL.

        ELBO gradients:
          d(BCE)/d(logits_i) = recon_i - x_i  (since sigmoid + BCE is numerically stable)
          d(KL)/d(mu_i) = mu_i
          d(KL)/d(log_var_i) = 0.5 * (exp(log_var_i) - 1)
        """
        recon = self._recon
        mu = self._mu
        log_var = self._log_var
        z = self._z
        sigma = self._sigma

        # --- Decoder backward ---
        # d(BCE)/d(logit) = sigmoid(logit) - x = recon - x
        grad_logits = elementwise_sub(recon, x)

        # Through dec_out linear layer
        grad_dec_h1 = self.dec_out.backward(grad_logits)

        # Through tanh: d/dh tanh(h) = 1 - tanh(h)^2
        # We need the pre-activation, but we stored the post-activation (dec_h1)
        # Reconstruct: h_post = tanh(h_pre), so grad_h_pre = grad * (1 - h_post^2)
        grad_dec_h1_pre = [g * (1.0 - h * h) for g, h in zip(grad_dec_h1, self._dec_h1)]

        # Through dec1 linear layer
        grad_z = self.dec1.backward(grad_dec_h1_pre)

        # --- KL divergence gradients ---
        # d(KL)/d(mu) = mu,  d(KL)/d(log_var) = 0.5 * (exp(log_var) - 1)
        grad_mu_kl = list(mu)
        grad_logvar_kl = [0.5 * (math.exp(lv) - 1) for lv in log_var]

        # --- Reparameterization gradient ---
        # z = mu + sigma * epsilon
        # dz/dmu = 1,  dz/dsigma = epsilon,  dsigma/dlogvar = 0.5 * sigma
        # So: dL/dmu = grad_z + grad_mu_kl
        #     dL/dlogvar = grad_z * epsilon * 0.5 * sigma + grad_logvar_kl
        grad_mu_total = elementwise_add(grad_z, grad_mu_kl)
        grad_logvar_total = elementwise_add(
            [g * e * 0.5 * s for g, e, s in zip(grad_z, self._epsilon, sigma)],
            grad_logvar_kl
        )

        # --- Encoder backward ---
        grad_enc_h1_mu = self.enc_mu.backward(grad_mu_total)
        grad_enc_h1_logvar = self.enc_logvar.backward(grad_logvar_total)
        grad_enc_h1 = elementwise_add(grad_enc_h1_mu, grad_enc_h1_logvar)

        # Through tanh
        grad_enc_h1_pre = [g * (1.0 - h * h) for g, h in zip(grad_enc_h1, self._enc_h1)]
        self.enc1.backward(grad_enc_h1_pre)

    def all_layers(self):
        return [self.enc1, self.enc_mu, self.enc_logvar, self.dec1, self.dec_out]

    def zero_grads(self):
        for layer in self.all_layers():
            layer.zero_grad()

    def update(self, lr):
        for layer in self.all_layers():
            layer.update(lr)

    def generate(self, z=None):
        if z is None:
            z = [random.gauss(0, 1) for _ in range(self.latent_dim)]
        h_pre = self.dec1.forward(z)
        h = tanh_vec(h_pre)
        logits = self.dec_out.forward(h)
        return sigmoid_vec(logits)


# ============================================================
# Обычный Autoencoder
# ============================================================

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.enc1 = LinearLayer(input_dim, hidden_dim)
        self.enc_out = LinearLayer(hidden_dim, latent_dim)
        self.dec1 = LinearLayer(latent_dim, hidden_dim)
        self.dec_out = LinearLayer(hidden_dim, input_dim)

    def forward(self, x):
        h_pre = self.enc1.forward(x)
        self._h1 = tanh_vec(h_pre)
        z_pre = self.enc_out.forward(self._h1)
        self._z = tanh_vec(z_pre)
        h2_pre = self.dec1.forward(self._z)
        self._h2 = tanh_vec(h2_pre)
        logits = self.dec_out.forward(self._h2)
        self._recon = sigmoid_vec(logits)
        return self._recon, self._z

    def backward(self, x):
        recon = self._recon
        grad_logits = elementwise_sub(recon, x)
        grad_h2 = self.dec_out.backward(grad_logits)
        grad_h2_pre = [g * (1.0 - h * h) for g, h in zip(grad_h2, self._h2)]
        grad_z = self.dec1.backward(grad_h2_pre)
        grad_h1 = self.enc_out.backward(grad_z)
        grad_h1_pre = [g * (1.0 - h * h) for g, h in zip(grad_h1, self._h1)]
        self.enc1.backward(grad_h1_pre)

    def zero_grads(self):
        for layer in [self.enc1, self.enc_out, self.dec1, self.dec_out]:
            layer.zero_grad()

    def update(self, lr):
        for layer in [self.enc1, self.enc_out, self.dec1, self.dec_out]:
            layer.update(lr)

    def encode(self, x):
        _, z = self.forward(x)
        return z

    def decode(self, z):
        h_pre = self.dec1.forward(z)
        h = tanh_vec(h_pre)
        logits = self.dec_out.forward(h)
        return sigmoid_vec(logits)

    def generate(self, z=None):
        if z is None:
            z = [random.gauss(0, 1) for _ in range(self.latent_dim)]
        return self.decode(z)


# ============================================================
# Loss functions
# ============================================================

def bce_loss(x, recon):
    loss = 0.0
    for xi, ri in zip(x, recon):
        ri = max(min(ri, 1 - 1e-7), 1e-7)
        loss -= xi * math.log(ri) + (1 - xi) * math.log(1 - ri)
    return loss

def mse_loss(x, recon):
    return sum((xi - ri) ** 2 for xi, ri in zip(x, recon)) / len(x)

def kl_divergence(mu, log_var):
    kl = 0.0
    for m, lv in zip(mu, log_var):
        kl += 1.0 + lv - m * m - math.exp(lv)
    return -0.5 * kl

def elbo_loss(x, recon, mu, log_var):
    recon_l = bce_loss(x, recon)
    kl = kl_divergence(mu, log_var)
    return recon_l + kl, recon_l, kl


# ============================================================
# Demo 1: Reparameterization Trick
# ============================================================

def demo_reparameterization():
    print_separator("Demo 1: Reparameterization Trick")
    print()
    print("Ключевая идея VAE: вместо детерминированного кодирования")
    print("  x -> z (фиксированное),")
    print("  мы кодируем распределение: x -> (mu, sigma) -> z = mu + sigma * epsilon")
    print()
    print("  где epsilon ~ N(0, 1) — шум из стандартного нормального распределения.")
    print("  Это позволяет дифференцировать по параметрам (mu, sigma) через")
    print("  случайный узел (stochastic node).")
    print()

    input_dim = 4
    latent_dim = 2
    model = VAE(input_dim, 8, latent_dim)

    x = [0.5, 0.3, 0.8, 0.1]
    recon, mu, log_var, z = model.forward(x)

    print(f"  Вход x:               {format_vec(x)}")
    print(f"  Encoded mu:            {format_vec(mu)}")
    print(f"  Encoded log_var:       {format_vec(log_var)}")
    print()

    # Многократное семплирование
    print("  Многократное семплирование z из одного x (разные epsilon):")
    print("  " + "-" * 55)
    random.seed(42)
    for i in range(5):
        sigma_s = [math.exp(0.5 * lv) for lv in log_var]
        eps_s = [random.gauss(0, 1) for _ in range(len(mu))]
        z_sample = [m + s * e for m, s, e in zip(mu, sigma_s, eps_s)]
        recon_sample = model.generate(z_sample)
        print(f"  z[{i}] = {format_vec(z_sample)}  ->  recon = {format_vec(recon_sample)}")
    print()

    sigma = [math.exp(0.5 * lv) for lv in log_var]
    print(f"  Sigma (std):           {format_vec(sigma)}")
    print(f"  Значение sigma показывает неопределённость кодирования.")
    print(f"  Маленький sigma -> z точно определён; большой -> много вариантов.")
    print()


# ============================================================
# Demo 2: ELBO Loss
# ============================================================

def demo_elbo():
    print_separator("Demo 2: ELBO (Evidence Lower Bound) Loss")
    print()
    print("ELBO = Reconstruction Loss + KL Divergence")
    print("  ELBO  = -E_q[log p(x|z)] + KL(q(z|x) || p(z))")
    print()
    print("  Reconstruction loss (BCE): мера насколько хорошо decoder восстанавливает x")
    print("  KL divergence: штраф за отклонение q(z|x) от prior p(z) = N(0,I)")
    print()

    input_dim = 4
    latent_dim = 2
    model = VAE(input_dim, 8, latent_dim)

    x = [0.5, 0.3, 0.8, 0.1]
    recon, mu, log_var, z = model.forward(x)
    total, recon_l, kl = elbo_loss(x, recon, mu, log_var)

    print(f"  Вход x:        {format_vec(x)}")
    print(f"  Реконструкция:  {format_vec(recon)}")
    print(f"  mu:             {format_vec(mu)}")
    print(f"  log_var:        {format_vec(log_var)}")
    print()

    print(f"  BCE loss (reconstruction):  {recon_l:.6f}")
    print(f"  KL divergence:              {kl:.6f}")
    print(f"  Total ELBO loss:            {total:.6f}")
    print()

    print("  KL divergence >= 0 всегда (неравенство Гиббса).")
    print(f"  KL = 0 только если q(z|x) = N(0,I) (prior совпадает с posteriором).")
    print()

    print("  Декомпозиция KL по измерениям:")
    for i in range(latent_dim):
        kl_i = kl_divergence([mu[i]], [log_var[i]])
        print(f"    z[{i}]: mu={mu[i]:.4f}, log_var={log_var[i]:.4f}, KL_i={kl_i:.6f}")
    print()


# ============================================================
# Demo 3: Генерация данных из Latent Space
# ============================================================

def demo_generation():
    print_separator("Demo 3: Генерация данных из Latent Space")
    print()
    print("Обученный VAE позволяет генерировать новые данные:")
    print("  1. Берём случайный z ~ N(0, I)")
    print("  2. Прогоняем через decoder: x_new = decode(z)")
    print()

    input_dim = 4
    latent_dim = 2
    model = VAE(input_dim, 16, latent_dim)

    # Синтетический датасет: 3 кластера
    data = []
    for _ in range(10):
        data.append([0.8 + random.gauss(0, 0.05), 0.2 + random.gauss(0, 0.05),
                      0.1 + random.gauss(0, 0.05), 0.9 + random.gauss(0, 0.05)])
    for _ in range(10):
        data.append([0.1 + random.gauss(0, 0.05), 0.8 + random.gauss(0, 0.05),
                      0.9 + random.gauss(0, 0.05), 0.2 + random.gauss(0, 0.05)])
    for _ in range(10):
        data.append([0.5 + random.gauss(0, 0.05), 0.5 + random.gauss(0, 0.05),
                      0.5 + random.gauss(0, 0.05), 0.5 + random.gauss(0, 0.05)])

    print("  Датасет: 3 кластера в 4-мерном пространстве")
    print("  Кластер 1: (~0.8, ~0.2, ~0.1, ~0.9)")
    print("  Кластер 2: (~0.1, ~0.8, ~0.9, ~0.2)")
    print("  Кластер 3: (~0.5, ~0.5, ~0.5, ~0.5)")
    print()

    # Обучение
    lr = 0.02
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(data)
        for xi in data:
            x_c = [max(min(v, 0.999), 0.001) for v in xi]
            recon, mu, log_var, z = model.forward(x_c)
            loss, _, _ = elbo_loss(x_c, recon, mu, log_var)
            total_loss += loss
            model.zero_grads()
            model.backward(x_c)
            model.update(lr)
        if (epoch + 1) % 25 == 0:
            avg = total_loss / len(data)
            print(f"  Epoch {epoch+1:3d}: avg ELBO loss = {avg:.4f}")
    print()

    # Генерация
    print("  Генерация 5 новых точек из latent space:")
    print("  " + "-" * 55)
    for i in range(5):
        z_new = [random.gauss(0, 1) for _ in range(latent_dim)]
        x_gen = model.generate(z_new)
        print(f"  z = {format_vec(z_new)}  ->  x = {format_vec(x_gen)}")
    print()

    # Интерполяция
    print("  Интерполяция в latent space (от кластера 1 к кластера 2):")
    print("  " + "-" * 55)
    x1 = data[0]
    x2 = data[12]
    mu1, _ = model.forward(x1)[1], None
    mu1 = model._mu
    model.forward(x2)
    mu2 = model._mu
    for alpha_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = [mu1[i] * (1 - alpha_val) + mu2[i] * alpha_val for i in range(latent_dim)]
        x_interp = model.generate(z_interp)
        print(f"  alpha={alpha_val:.2f}: x = {format_vec(x_interp)}")
    print()


# ============================================================
# Demo 4: Сравнение VAE с обычным Autoencoder
# ============================================================

def demo_comparison():
    print_separator("Demo 4: Сравнение VAE с обычным Autoencoder")
    print()
    print("  Обычный AE:  x -> encoder -> z (точный) -> decoder -> x'")
    print("  VAE:         x -> encoder -> (mu, sigma) -> z (шум) -> decoder -> x'")
    print()
    print("  Различия:")
    print("  1. AE: детерминированный; VAE: стохастический")
    print("  2. AE: latent space может быть разорван; VAE: continuous latent space")
    print("  3. AE: нельзя генерировать; VAE: можно сэмплить из N(0,I)")
    print("  4. AE: minimizes MSE; VAE: maximizes ELBO (= reconstruction + KL)")
    print()

    input_dim = 4
    latent_dim = 2
    vae = VAE(input_dim, 16, latent_dim)
    ae = Autoencoder(input_dim, 16, latent_dim)

    # Данные
    data = []
    for _ in range(15):
        data.append([0.8 + random.gauss(0, 0.1), 0.2 + random.gauss(0, 0.1),
                      0.1 + random.gauss(0, 0.1), 0.9 + random.gauss(0, 0.1)])
    for _ in range(15):
        data.append([0.1 + random.gauss(0, 0.1), 0.8 + random.gauss(0, 0.1),
                      0.9 + random.gauss(0, 0.1), 0.2 + random.gauss(0, 0.1)])

    # Обучение
    lr = 0.02
    epochs = 80
    for epoch in range(epochs):
        vae_total = 0.0
        ae_total = 0.0
        random.shuffle(data)
        for xi in data:
            x_c = [max(min(v, 0.999), 0.001) for v in xi]
            # VAE
            recon_v, mu, lv, z = vae.forward(x_c)
            vae_l, _, _ = elbo_loss(x_c, recon_v, mu, lv)
            vae_total += vae_l
            vae.zero_grads()
            vae.backward(x_c)
            vae.update(lr)
            # AE
            recon_a, _ = ae.forward(x_c)
            ae_l = mse_loss(x_c, recon_a)
            ae_total += ae_l
            ae.zero_grads()
            ae.backward(x_c)
            ae.update(lr)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: VAE ELBO={vae_total/len(data):.4f}  |  AE MSE={ae_total/len(data):.4f}")
    print()

    # Восстановление
    print("  Сравнение восстановления (reconstruction):")
    print("  " + "-" * 58)
    test_x = data[0]
    x_c = [max(min(v, 0.999), 0.001) for v in test_x]

    recon_v, _, _, _ = vae.forward(x_c)
    recon_a, _ = ae.forward(x_c)

    print(f"  Оригинал:  {format_vec(test_x)}")
    print(f"  VAE recon:  {format_vec(recon_v)}")
    print(f"  AE recon:   {format_vec(recon_a)}")
    vae_err = math.sqrt(sum((a-b)**2 for a, b in zip(test_x, recon_v)) / len(test_x))
    ae_err = math.sqrt(sum((a-b)**2 for a, b in zip(test_x, recon_a)) / len(test_x))
    print(f"  VAE RMSE:   {vae_err:.6f}")
    print(f"  AE RMSE:    {ae_err:.6f}")
    print()

    # Генерация
    print("  Генерация из latent space:")
    print("  " + "-" * 58)
    print("  VAE (z ~ N(0,I)):")
    for i in range(3):
        z_new = [random.gauss(0, 1) for _ in range(latent_dim)]
        x_gen = vae.generate(z_new)
        print(f"    [{i}] z={format_vec(z_new)} -> x={format_vec(x_gen)}")
    print()
    print("  AE (z ~ N(0,I)):")
    for i in range(3):
        z_new = [random.gauss(0, 1) for _ in range(latent_dim)]
        x_gen = ae.generate(z_new)
        print(f"    [{i}] z={format_vec(z_new)} -> x={format_vec(x_gen)}")
    print()

    print("  Наблюдение: AE не обучался на данных из N(0,I), поэтому")
    print("  его генерация выдаёт бессмысленные значения.")
    print("  VAE обучался с KL-штрафом, поэтому latent space непрерывен")
    print("  и генерация из N(0,I) осмысленна.")
    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  84 - Variational Autoencoder (VAE)")
    print("  Реализация на чистом Python")
    print("=" * 60)

    demo_reparameterization()
    demo_elbo()
    demo_generation()
    demo_comparison()

    print_separator("Итоги")
    print()
    print("  VAE — это генеративная модель, которая:")
    print("  1. Кодирует x -> (mu, log_var) (распределение в latent space)")
    print("  2. Использует reparameterization trick для дифференцируемости")
    print("  3. Обучается через ELBO = Reconstruction + KL divergence")
    print("  4. Может генерировать новые данные, сэмплируя z ~ N(0,I)")
    print()
    print("  Преимущества над обычным AE:")
    print("  - Непрерывный, структурированный latent space")
    print("  - Возможность генерации новых данных")
    print("  - Интерполяция между объектами")
    print("  - Вывод вероятностных представлений")
    print()
