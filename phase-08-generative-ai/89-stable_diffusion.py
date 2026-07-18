"""
89-stable_diffusion.py
Архитектура Stable Diffusion — от текста до изображения.

Stable Diffusion — латентная диффузионная модель:
  1. VAE сжимает изображение в латентное пространство (512x512 -> 64x64x4)
  2. UNet предсказывает шум в латентном пространстве на каждом шаге
  3. Text Encoder (CLIP) кодирует текстовое описание в эмбеддинг
  4. Pipeline: текст -> текстовый эмбеддинг -> итеративное удаление шума из латента -> декод в изображение

Все вычисления на чистом Python (без numpy/torch/tensorflow).
"""

import random
import math


random.seed(42)


# ============================================================
#  УТИЛИТЫ
# ============================================================

def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def mat_fill(rows, cols, val=0.0):
    return [[val] * cols for _ in range(rows)]

def vec_zeros(n):
    return [0.0] * n

def vec_fill(n, val=0.0):
    return [val] * n

def mat_matmul(A, B):
    """A: (m, k), B: (k, n) -> (m, n)"""
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C

def mat_add(A, B):
    m, n = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_scale(a, s):
    return [x * s for x in a]

def relu(x):
    return [max(0.0, v) for v in x]

def softmax(vec):
    mx = max(vec)
    exps = [math.exp(v - mx) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]

def gelu(x):
    """GELU-активация (приближение)."""
    return [0.5 * v * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3))) for v in x]

def layernorm(x, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((v - mean)**2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]

def cosine_schedule(t, T):
    """Косинусное расписание噪声调度."""
    return math.cos((t / T) * math.pi / 2) ** 2

def linear_interpolate(a, b, t):
    return [a_i * (1 - t) + b_i * t for a_i, b_i in zip(a, b)]

def randn_vec(n, rng):
    """Генерация вектора нормального распределения (Box-Muller)."""
    result = []
    for i in range(n):
        u1 = max(rng.random(), 1e-10)
        u2 = rng.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        result.append(z)
    return result

def randn_matrix(rows, cols, rng):
    return [randn_vec(cols, rng) for _ in range(rows)]

def matrix_to_flat(A):
    return [A[i][j] for i in range(len(A)) for j in range(len(A[0]))]

def flat_to_matrix(flat, rows, cols):
    M = []
    idx = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(flat[idx])
            idx += 1
        M.append(row)
    return M

def downsample_2d(mat, factor=2):
    """Простой max-pooling 2D."""
    rows = len(mat)
    cols = len(mat[0])
    new_rows = rows // factor
    new_cols = cols // factor
    result = mat_zeros(new_rows, new_cols)
    for i in range(new_rows):
        for j in range(new_cols):
            max_val = -1e18
            for di in range(factor):
                for dj in range(factor):
                    val = mat[i * factor + di][j * factor + dj]
                    if val > max_val:
                        max_val = val
            result[i][j] = max_val
    return result

def upsample_2d(mat, factor=2):
    """Простой билинейный апсемплинг."""
    rows = len(mat)
    cols = len(mat[0])
    new_rows = rows * factor
    new_cols = cols * factor
    result = mat_zeros(new_rows, new_cols)
    for i in range(new_rows):
        for j in range(new_cols):
            src_i = i / factor
            src_j = j / factor
            i0 = int(src_i)
            j0 = int(src_j)
            i1 = min(i0 + 1, rows - 1)
            j1 = min(j0 + 1, cols - 1)
            di = src_i - i0
            dj = src_j - j0
            val = (mat[i0][j0] * (1 - di) * (1 - dj) +
                   mat[i1][j0] * di * (1 - dj) +
                   mat[i0][j1] * (1 - di) * dj +
                   mat[i1][j1] * di * dj)
            result[i][j] = val
    return result

def matrix_flatten(A):
    """Сглаживание матрицы в вектор."""
    return [A[i][j] for i in range(len(A)) for j in range(len(A[0]))]

def normalize_image(mat):
    """Нормализация матрицы значений в диапазон [0, 255]."""
    flat = matrix_flatten(mat)
    mn = min(flat)
    mx = max(flat)
    rng = mx - mn if abs(mx - mn) > 1e-10 else 1.0
    return [[int((mat[i][j] - mn) / rng * 255) for j in range(len(mat[0]))] for i in range(len(mat))]

def print_matrix_ascii(mat, title="", width=40):
    """Вывод матрицы в виде ASCII-арта."""
    if title:
        print(f"\n  {title}")
    rows = len(mat)
    cols = len(mat[0])
    normalized = normalize_image(mat)
    chars = " .:-=+*#%@"
    for i in range(rows):
        line = "  "
        for j in range(cols):
            idx = normalized[i][j] * (len(chars) - 1) // 255
            line += chars[idx] * 2
        print(line)


# ============================================================
#  1. VAE (Variational Autoencoder)
# ============================================================

class SimpleLinear:
    """Полносвязный слой с Xavier-инициализацией."""
    def __init__(self, in_f, out_f, rng):
        self.weight = mat_zeros(out_f, in_f)
        self.bias = vec_zeros(out_f)
        limit = math.sqrt(6.0 / (in_f + out_f))
        for i in range(out_f):
            for j in range(in_f):
                self.weight[i][j] = (rng.random() * 2 - 1) * limit
            self.bias[i] = (rng.random() * 2 - 1) * 0.01

    def forward(self, x):
        out = []
        for i in range(len(self.weight)):
            s = self.bias[i]
            for j in range(len(x)):
                s += self.weight[i][j] * x[j]
            out.append(s)
        return out


class ConvLayer2D:
    """Упрощённая 2D-свёртка (ядро 3x3, stride=1, padding=1)."""
    def __init__(self, rng):
        self.kernel = mat_zeros(3, 3)
        for i in range(3):
            for j in range(3):
                self.kernel[i][j] = (rng.random() * 2 - 1) * 0.3
        self.bias = (rng.random() * 2 - 1) * 0.01

    def forward(self, mat):
        rows = len(mat)
        cols = len(mat[0])
        result = mat_zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                s = self.bias
                for ki in range(-1, 2):
                    for kj in range(-1, 2):
                        ni, nj = i + ki, j + kj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            s += self.kernel[ki + 1][kj + 1] * mat[ni][nj]
                result[i][j] = s
        return result


class VAEEncoder:
    """VAE Encoder: изображение -> (mean, logvar) в latent space."""
    def __init__(self, latent_dim, rng):
        self.latent_dim = latent_dim
        self.conv1 = ConvLayer2D(rng)
        self.conv2 = ConvLayer2D(rng)
        self.fc_mean = SimpleLinear(16, latent_dim, rng)
        self.fc_logvar = SimpleLinear(16, latent_dim, rng)

    def forward(self, img):
        """img: матрица 8x8 -> (mean, logvar) векторы размера latent_dim."""
        h = relu(matrix_flatten(self.conv1.forward(img)))
        h = relu(matrix_flatten(self.conv2.forward(img)))
        h = h[:16] if len(h) > 16 else h + [0.0] * (16 - len(h))
        mean = self.fc_mean.forward(h)
        logvar = self.fc_logvar.forward(h)
        return mean, logvar

    def reparameterize(self, mean, logvar, rng):
        """Репараметризация: z = mean + eps * exp(0.5 * logvar)."""
        eps = randn_vec(self.latent_dim, rng)
        std = [math.exp(0.5 * lv) for lv in logvar]
        z = [mean[i] + eps[i] * std[i] for i in range(self.latent_dim)]
        return z


class VAEDecoder:
    """VAE Decoder: latent vector -> изображение."""
    def __init__(self, latent_dim, rng):
        self.latent_dim = latent_dim
        self.fc = SimpleLinear(latent_dim, 16, rng)
        self.conv1 = ConvLayer2D(rng)
        self.conv2 = ConvLayer2D(rng)

    def forward(self, z):
        """z: latent vector -> матрица 4x4."""
        h = gelu(self.fc.forward(z))
        mat = flat_to_matrix(h[:16], 4, 4)
        mat = relu(matrix_flatten(self.conv1.forward(mat)))
        mat = flat_to_matrix(mat[:16], 4, 4)
        out = self.conv2.forward(mat)
        return out


class VAE:
    """Полный VAE: encoder -> reparameterize -> decoder."""
    def __init__(self, latent_dim=4, rng=None):
        if rng is None:
            rng = random.Random(42)
        self.encoder = VAEEncoder(latent_dim, rng)
        self.decoder = VAEDecoder(latent_dim, rng)
        self.latent_dim = latent_dim

    def encode(self, img, rng):
        mean, logvar = self.encoder.forward(img)
        z = self.encoder.reparameterize(mean, logvar, rng)
        return z, mean, logvar

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, img, rng):
        z, mean, logvar = self.encode(img, rng)
        recon = self.decode(z)
        return recon, z, mean, logvar

    def kl_divergence(self, mean, logvar):
        """KL-расстояние: 0.5 * sum(mean^2 + exp(logvar) - logvar - 1)."""
        kl = 0.0
        for i in range(len(mean)):
            kl += mean[i]**2 + math.exp(logvar[i]) - logvar[i] - 1
        return 0.5 * kl


# ============================================================
#  2. UNet — предсказание шума
# ============================================================

class SelfAttention1D:
    """Упрощённый Self-Attention для 1D-последовательности."""
    def __init__(self, dim, rng):
        self.dim = dim
        self.Wq = SimpleLinear(dim, dim, rng)
        self.Wk = SimpleLinear(dim, dim, rng)
        self.Wv = SimpleLinear(dim, dim, rng)
        self.Wo = SimpleLinear(dim, dim, rng)

    def forward(self, x):
        """x: список векторов [seq_len, dim]."""
        seq = len(x)
        Q = [self.Wq.forward(v) for v in x]
        K = [self.Wk.forward(v) for v in x]
        V = [self.Wv.forward(v) for v in x]

        scores = mat_zeros(seq, seq)
        scale = math.sqrt(self.dim)
        for i in range(seq):
            for j in range(seq):
                scores[i][j] = vec_dot(Q[i], K[j]) / scale

        attn = []
        for i in range(seq):
            row = softmax(scores[i])
            out = vec_zeros(self.dim)
            for j in range(seq):
                out = vec_add(out, vec_scale(V[j], row[j]))
            attn.append(out)

        return [self.Wo.forward(v) for v in attn]


class ResBlock:
    """Residual Block с Conv + time modulation + skip connection."""
    def __init__(self, rng):
        self.conv1 = ConvLayer2D(rng)
        self.conv2 = ConvLayer2D(rng)

    def forward(self, mat, time_emb=None):
        rows = len(mat)
        cols = len(mat[0])
        n = rows * cols

        # Conv + ReLU
        h = self.conv1.forward(mat)
        h = relu(matrix_flatten(h)[:n])
        h = flat_to_matrix(h, rows, cols)

        # Time modulation (broadcast to spatial dims)
        if time_emb is not None:
            h_flat = matrix_flatten(h)
            te_len = len(time_emb)
            te = time_emb + [0.0] * (n - te_len) if n > te_len else time_emb[:n]
            h_flat = vec_add(h_flat[:n], te[:n])
            h = flat_to_matrix(h_flat, rows, cols)

        # Conv + ReLU
        h = self.conv2.forward(h)
        h = relu(matrix_flatten(h)[:n])
        h = flat_to_matrix(h, rows, cols)

        # Skip connection
        skip_flat = matrix_flatten(mat)[:n]
        h_flat = matrix_flatten(h)[:n]
        h_flat = vec_add(h_flat[:n], skip_flat[:n])
        return flat_to_matrix(h_flat[:n], rows, cols)


class TimeEmbedding:
    """Sinusoidal time embedding (как в Transformer)."""
    def __init__(self, dim, rng):
        self.dim = dim
        self.fc1 = SimpleLinear(dim, dim, rng)
        self.fc2 = SimpleLinear(dim, dim, rng)

    def forward(self, t):
        """t: scalar -> вектор dim."""
        half = self.dim // 2
        freqs = [math.exp(-math.log(10000.0) * i / half) for i in range(half)]
        emb = []
        for i in range(half):
            emb.append(math.sin(t * freqs[i]))
            emb.append(math.cos(t * freqs[i]))
        emb = emb[:self.dim]
        if len(emb) < self.dim:
            emb += [0.0] * (self.dim - len(emb))
        h = gelu(self.fc1.forward(emb))
        return self.fc2.forward(h)


class UNet:
    """UNet для предсказания шума в латентном пространстве.

    Архитектура:
      - Time embedding: timestep -> вектор
      - Encoder: 3 ResBlock + downsampling (8x8 -> 4x4 -> 2x2)
      - Bottleneck: ResBlock + SelfAttention
      - Decoder: 2 ResBlock + upsampling (2x2 -> 4x4 -> 8x8)
      - Output: Conv -> предсказание шума (матрица 8x8)
    """
    def __init__(self, rng=None):
        if rng is None:
            rng = random.Random(42)
        self.time_emb = TimeEmbedding(16, rng)

        # Encoder: 8x8 -> 4x4 -> 2x2
        self.enc1 = ResBlock(rng)
        self.enc2 = ResBlock(rng)
        self.enc3 = ResBlock(rng)

        # Bottleneck at 2x2
        self.bottleneck = ResBlock(rng)
        self.attention = SelfAttention1D(4, rng)

        # Decoder: 2x2 -> 4x4 -> 8x8
        self.dec2 = ResBlock(rng)
        self.dec1 = ResBlock(rng)

        # Output projection
        self.out_conv = ConvLayer2D(rng)

    def forward(self, noisy_latent, timestep, rng=None):
        """
        noisy_latent: матрица 8x8 (шумный латент)
        timestep: float (текущий шаг диффузионного процесса)
        Returns: предсказание шума (матрица 8x8)
        """
        t_emb = self.time_emb.forward(timestep)

        # Encoder path: 8x8 -> 4x4 -> 2x2
        h1 = self.enc1.forward(noisy_latent, t_emb)
        h1_small = downsample_2d(h1, 2)

        h2 = self.enc2.forward(h1_small, t_emb)
        h2_small = downsample_2d(h2, 2)

        h3 = self.enc3.forward(h2_small, t_emb)

        # Bottleneck: 2x2 + attention
        bn = self.bottleneck.forward(h3, t_emb)
        bn_flat = matrix_flatten(bn)[:4]
        seq = [bn_flat]
        attn_out = self.attention.forward(seq)
        bn = flat_to_matrix(attn_out[0][:4], 2, 2)

        # Decoder path: 2x2 -> 4x4 -> 8x8
        d2 = upsample_2d(bn, 2)
        d2 = self.dec2.forward(d2, t_emb)

        d1 = upsample_2d(d2, 2)
        d1 = self.dec1.forward(d1, t_emb)

        # Output conv
        out = self.out_conv.forward(d1)

        return out

    def predict_noise(self, noisy_latent, timestep, rng=None):
        """Обёртка для предсказания шума."""
        return self.forward(noisy_latent, timestep, rng)


# ============================================================
#  3. Text Encoder (упрощённый CLIP)
# ============================================================

class TextEncoder:
    """Упрощённый текстовый энкодер (стиль CLIP).

    Слово -> Embedding -> Self-Attention -> Mean Pooling -> Project -> text_emb.
    """
    def __init__(self, vocab_size=128, embed_dim=8, rng=None):
        if rng is None:
            rng = random.Random(42)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Словарь эмбеддингов (vocab_size x embed_dim)
        self.embedding = randn_matrix(vocab_size, embed_dim, rng)
        for i in range(vocab_size):
            for j in range(embed_dim):
                self.embedding[i][j] *= 0.02

        self.attention = SelfAttention1D(embed_dim, rng)
        self.projection = SimpleLinear(embed_dim, embed_dim, rng)

    def tokenize(self, text):
        """Простая токенизация: каждый символ -> ASCII код % vocab_size."""
        return [ord(c) % self.vocab_size for c in text.lower() if c.isalnum()]

    def encode_tokens(self, token_ids):
        """Кодирование токенов в эмбеддинги."""
        return [self.embedding[tid][:] for tid in token_ids]

    def forward(self, text, rng=None):
        """
        text: строка
        Returns: вектор эмбеддинга размера embed_dim.
        """
        if rng is None:
            rng = random.Random(42)
        tokens = self.tokenize(text)
        if not tokens:
            tokens = [0]

        embs = self.encode_tokens(tokens)
        attn_out = self.attention.forward(embs)
        pooled = vec_zeros(self.embed_dim)
        for v in attn_out:
            pooled = vec_add(pooled, v)
        pooled = vec_scale(pooled, 1.0 / len(attn_out))

        out = self.projection.forward(pooled)
        norm = math.sqrt(vec_dot(out, out)) + 1e-8
        return vec_scale(out, 1.0 / norm)

    def compute_similarity(self, text1, text2):
        """Косинусное сходство между двумя текстами."""
        emb1 = self.forward(text1)
        emb2 = self.forward(text2)
        return vec_dot(emb1, emb2)


# ============================================================
#  4. Diffusion Pipeline
# ============================================================

class DiffusionPipeline:
    """Полный Stable Diffusion Pipeline.

    Шаги:
      1. Text Encoder кодирует промпт
      2. VAE Encoder кодирует референсное изображение (или начинаем с шума)
      3. Forward diffusion: добавляем шум к латенту
      4. Reverse diffusion: UNet итеративно удаляет шум (с conditioning на текст)
      5. VAE Decoder декодирует чистый латент в изображение
    """
    def __init__(self, rng=None):
        if rng is None:
            rng = random.Random(42)
        self.rng = rng
        self.vae = VAE(latent_dim=4, rng=rng)
        self.unet = UNet(rng=rng)
        self.text_encoder = TextEncoder(vocab_size=128, embed_dim=8, rng=rng)
        self.num_steps = 5  # Количество шагов диффузионного процесса

    def add_noise(self, latent, timestep, T):
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * eps."""
        alpha_t = cosine_schedule(timestep, T)
        sqrt_alpha = math.sqrt(alpha_t)
        sqrt_one_minus_alpha = math.sqrt(1 - alpha_t)
        noise = randn_matrix(len(latent), len(latent[0]), self.rng)
        noisy = mat_add(mat_scale(latent, sqrt_alpha), mat_scale(noise, sqrt_one_minus_alpha))
        return noisy, noise

    def denoise_step(self, noisy_latent, timestep, T):
        """Один шаг обратной диффузионного процесса (DDPM-style)."""
        pred_noise = self.unet.predict_noise(noisy_latent, timestep, self.rng)

        alpha_t = cosine_schedule(timestep, T)
        alpha_prev = cosine_schedule(max(timestep - 1, 0), T)
        sqrt_alpha = math.sqrt(max(alpha_t, 1e-8))
        sqrt_one_minus_alpha = math.sqrt(max(1 - alpha_t, 1e-8))

        # DDPM: x_{t-1} = (x_t - (1-alpha_t)/sqrt(1-alpha_t) * eps) / sqrt(alpha_t) + sigma * z
        coef = (1 - alpha_t) / max(sqrt_one_minus_alpha, 1e-8)
        denoised = mat_scale(
            mat_add(noisy_latent, mat_scale(pred_noise, -coef)),
            1.0 / max(sqrt_alpha, 1e-8)
        )

        sigma = math.sqrt(max((1 - alpha_prev) / max(1 - alpha_t, 1e-8), 0)) * 0.05
        noise_for_next = randn_matrix(len(noisy_latent), len(noisy_latent[0]), self.rng)
        denoised = mat_add(denoised, mat_scale(noise_for_next, sigma))

        return denoised, pred_noise

    def generate(self, prompt, seed=None):
        """Полная генерация изображения из текста."""
        if seed is not None:
            self.rng = random.Random(seed)

        # 1. Кодирование текста
        text_emb = self.text_encoder.forward(prompt, self.rng)

        # 2. Создание начального шумного латента (64x64 -> упрощённо 8x8)
        latent = randn_matrix(8, 8, self.rng)

        # 3. Условная генерация: текстовый эмбеддинг влияет на начальный шум
        text_bias = text_emb[:min(len(text_emb), 8)]
        for i in range(min(len(text_bias), 8)):
            latent[i % 8][0] += text_bias[i] * 0.5

        print(f"\n  [Pipeline] Текстовый эмбеддинг (первые 8 компонент):")
        print(f"    {[f'{v:+.4f}' for v in text_emb[:8]]}")
        print(f"\n  [Pipeline] Начальный шумный латент (8x8):")
        for row in latent[:4]:
            print(f"    {[f'{v:+.3f}' for v in row[:4]]} ...")

        # 4. Итеративное удаление шума
        print(f"\n  [Pipeline] Запуск диффузионного процесса ({self.num_steps} шагов)...")
        for t in range(self.num_steps, 0, -1):
            latent, pred_noise = self.denoise_step(latent, t, self.num_steps)
            flat_noise = matrix_flatten(pred_noise)
            noise_energy = sum(v**2 for v in flat_noise) / len(flat_noise)
            print(f"    Шаг {self.num_steps - t + 1}/{self.num_steps} (t={t}): "
                  f"энергия предсказ. шума = {noise_energy:.4f}")

        # 5. Декодирование из изображения
        # Latent — матрица 8x8, а VAE декодер ожидает вектор dim=4
        # Усредняем 2x2 блоки, чтобы получить 4 элемента
        latent_vec = []
        for bi in range(4):
            for bj in range(4):
                block_sum = 0.0
                for di in range(2):
                    for dj in range(2):
                        block_sum += latent[bi * 2 + di][bj * 2 + dj]
                latent_vec.append(block_sum / 4.0)
        image = self.vae.decode(latent_vec[:self.vae.latent_dim])

        print(f"\n  [Pipeline] Итоговый латент (после denoising):")
        for row in latent[:4]:
            print(f"    {[f'{v:+.3f}' for v in row[:4]]} ...")

        return image, latent, text_emb


# ============================================================
#  ДЕМОНСТРАЦИИ
# ============================================================

def demo_vae_latent_space():
    """Демо 1: VAE — сжатие изображения в латентное пространство."""
    print("=" * 60)
    print("  ДЕМО 1: VAE — сжатие в латентное пространство")
    print("=" * 60)

    rng = random.Random(42)
    vae = VAE(latent_dim=4, rng=rng)

    # Создаём тестовое изображение 8x8 с простым паттерном
    img = mat_zeros(8, 8)
    for i in range(8):
        for j in range(8):
            img[i][j] = math.sin(i * 0.5) * math.cos(j * 0.5) + (rng.random() * 0.2)

    print("\n  Оригинальное изображение (8x8):")
    for row in img:
        print(f"    {[f'{v:+.3f}' for v in row]}")

    # Encoder
    z, mean, logvar = vae.encode(img, rng)

    print(f"\n  --- Encoder -> Latent Space ---")
    print(f"  Latent vector (dim={len(z)}): {[f'{v:+.4f}' for v in z]}")
    print(f"  Mean:  {[f'{v:+.4f}' for v in mean]}")
    print(f"  LogVar:{[f'{v:+.4f}' for v in logvar]}")

    # KL-дивергенция
    kl = vae.kl_divergence(mean, logvar)
    print(f"  KL-дивергенция: {kl:.4f}")
    print(f"  (KL → 0 означает, что posterior близок к стандартному N(0,1))")

    # Decoder
    recon = vae.decode(z)

    print(f"\n  --- Decoder -> Реконструкция (4x4) ---")
    for row in recon:
        print(f"    {[f'{v:+.3f}' for v in row]}")

    # Апсемплинг до 8x8 для сравнения
    recon_8x8 = upsample_2d(recon, 2)
    print(f"\n  Реконструкция (апсемплинг до 8x8):")
    for row in recon_8x8:
        print(f"    {[f'{v:+.3f}' for v in row]}")

    # MSE reconstruction error
    mse = 0.0
    for i in range(8):
        for j in range(8):
            mse += (img[i][j] - recon_8x8[i][j]) ** 2
    mse /= 64
    print(f"\n  Reconstruction MSE: {mse:.4f}")
    print(f"  (Без обучения веса случайные, поэтому MSE высокий — это нормально)")

    print()


def demo_unet_noise_prediction():
    """Демо 2: UNet — предсказание шума."""
    print("=" * 60)
    print("  ДЕМО 2: UNet — предсказание шума в латентном пространстве")
    print("=" * 60)

    rng = random.Random(42)
    unet = UNet(rng=rng)

    # Создаём шумный латент
    noisy_latent = randn_matrix(8, 8, rng)

    print("\n  Шумный латент (8x8, первые 4 строки):")
    for row in noisy_latent[:4]:
        print(f"    {[f'{v:+.3f}' for v in row[:4]]} ...")

    # Предсказание шума на разных timestep
    print(f"\n  --- Предсказание шума на разных шагах ---")
    timesteps = [5, 4, 3, 2, 1]
    for t in timesteps:
        pred_noise = unet.predict_noise(noisy_latent, t, rng)
        flat = matrix_flatten(pred_noise)
        energy = sum(v**2 for v in flat) / len(flat)
        mean_val = sum(flat) / len(flat)
        print(f"  t={t}: среднее={mean_val:+.4f}, энергия(MSE)={energy:.4f}")

    # Архитектура UNet
    print(f"\n  --- Архитектура UNet ---")
    print(f"  Input:  8x8 (шумный латент)")
    print(f"  Enc1:   ResBlock(8x8) + Downsample -> 4x4")
    print(f"  Enc2:   ResBlock(4x4) + Downsample -> 2x2")
    print(f"  Enc3:   ResBlock(2x2)")
    print(f"  BN:     ResBlock + SelfAttention")
    print(f"  Dec2:   Upsample + ResBlock -> 4x4")
    print(f"  Dec1:   Upsample + ResBlock -> 8x8")
    print(f"  Output: Conv -> 8x8 (предсказание шума)")

    print(f"\n  Time Embedding:")
    print(f"  Sinusoidal -> FC(GELU) -> FC -> вектор для модуляции ResBlock")

    print()


def demo_text_embedding():
    """Демо 3: Text Encoder — эмбеддинги текста."""
    print("=" * 60)
    print("  ДЕМО 3: Text Encoder (упрощённый CLIP)")
    print("=" * 60)

    rng = random.Random(42)
    encoder = TextEncoder(vocab_size=128, embed_dim=8, rng=rng)

    # Тестовые промпты
    prompts = [
        "a red cat",
        "a blue dog",
        "the sun rises",
        "a red dog",
    ]

    print("\n  Токенизация:")
    for p in prompts:
        tokens = encoder.tokenize(p)
        print(f"  '{p}' -> {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

    print(f"\n  Эмбеддинги (нормализованные, dim=8):")
    embeddings = {}
    for p in prompts:
        emb = encoder.forward(p, rng)
        embeddings[p] = emb
        print(f"  '{p}': {[f'{v:+.4f}' for v in emb]}")

    print(f"\n  Косинусное сходство:")
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            sim = encoder.compute_similarity(prompts[i], prompts[j])
            print(f"  sim('{prompts[i]}', '{prompts[j]}') = {sim:+.4f}")

    print(f"\n  Архитектура Text Encoder:")
    print(f"  Text -> Tokenize -> Embedding(128x8)")
    print(f"       -> SelfAttention(8)")
    print(f"       -> Mean Pooling")
    print(f"       -> Linear Projection(8->8)")
    print(f"       -> L2 Normalize")

    print()


def demo_full_pipeline():
    """Демо 4: Полный Stable Diffusion Pipeline."""
    print("=" * 60)
    print("  ДЕМО 4: Полный Stable Diffusion Pipeline")
    print("=" * 60)

    pipeline = DiffusionPipeline(rng=random.Random(42))

    print("\n  Архитектура Stable Diffusion:")
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  Text Prompt -> [Text Encoder (CLIP)] -> text_emb   │")
    print("  │                                         ↓            │")
    print("  │  Noise z_T -> [UNet + Text Condition] -> z_{T-1}    │")
    print("  │                    ↑ timestep, text_emb              │")
    print("  │              ... (T steps) ...                       │")
    print("  │                    ↓                                 │")
    print("  │              z_0 (clean latent)                      │")
    print("  │                    ↓                                 │")
    print("  │         [VAE Decoder] -> Generated Image            │")
    print("  └──────────────────────────────────────────────────────┘")

    prompts = [
        "a red cat sitting on blue sofa",
        "sunrise over mountain peaks",
    ]

    for prompt in prompts:
        print(f"\n  {'─' * 50}")
        print(f"  Промпт: \"{prompt}\"")
        image, latent, text_emb = pipeline.generate(prompt, seed=42)

        print(f"\n  Сгенерированное изображение (ASCII):")
        print_matrix_ascii(image, f"Image for: \"{prompt}\"")

    print(f"\n  Итоговый латент 8x8 (нормализованный):")
    print_matrix_ascii(latent, "Final Latent")

    print(f"\n  Ключевые компоненты Stable Diffusion:")
    print(f"  1. VAE: сжатие 512x512x3 -> 64x64x4 (в ~48 раз меньше)")
    print(f"  2. UNet: предсказание шума с cross-attention на текст")
    print(f"  3. Text Encoder: CLIP ViT-L/14, 77 токенов, 768-dim")
    print(f"  4. Scheduler: DDPM/DDIM/PLMS (мы используем DDPM-style)")
    print(f"  5. CFG: Classifier-Free Guidance для усиления влияния промпта")

    print()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║" + " Stable Diffusion — Архитектура".center(58) + "║")
    print("║" + " VAE + UNet + Text Encoder + Pipeline".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    demo_vae_latent_space()
    demo_unet_noise_prediction()
    demo_text_embedding()
    demo_full_pipeline()

    print("=" * 60)
    print("  Все демонстрации завершены.")
    print("=" * 60)
