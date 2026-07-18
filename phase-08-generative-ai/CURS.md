# Phase 8: Generative AI

> Генеративный AI — от VAE до Diffusion Models.

[Вернуться к README](../README.md)

## Оглавление

| # | Урок | Код |
|---|------|-----|
| 83 | [Autoencoders](#урок-83-autoencoders) | [Код](83-autoencoders.py) |
| 84 | [Variational Autoencoders](#урок-84-variational-autoencoders) | [Код](84-vae.py) |
| 85 | [Generative Adversarial Networks](#урок-85-generative-adversarial-networks) | [Код](85-gan.py) |
| 86 | [GAN Training & Stability](#урок-86-gan-training--stability) | [Код](86-gan_stability.py) |
| 87 | [Diffusion Models](#урок-87-diffusion-models) | [Код](87-diffusion_models.py) |
| 88 | [Denoising Diffusion](#урок-88-denoising-diffusion) | [Код](88-denoising_diffusion.py) |
| 89 | [Stable Diffusion](#урок-89-stable-diffusion) | [Код](89-stable_diffusion.py) |
| 90 | [Text-to-Image](#урок-90-text-to-image) | [Код](90_text_to_image.py) |

---

## Урок 83: Autoencoders

### Архитектура

```
Input → Encoder → Latent Space → Decoder → Reconstruction

Loss = ||Input - Reconstruction||²
```

### Latent Space

Сжатое представление данных. Интерполяция в latent space → плавные переходы.

---

## Урок 84: Variational Autoencoders

### Reparameterization Trick

```
z = μ + σ × ε,  ε ~ N(0, 1)

Позволяет дифференцировать стохастический узел
```

### ELBO Loss

```
ELBO = Reconstruction + KL(q(z|x) || N(0, I))

KL >= 0, минимизируем = приближаем к нормали
```

---

## Урок 85: GAN

### Архитектура

```
Generator:     z (шум) → fake data
Discriminator: data → real/fake

Minimax game:
  min_G max_D [E[log(D(x))] + E[log(1 - D(G(z)))]]
```

---

## Урок 86: GAN Training & Stability

### Методы

| Метод | Описание |
|---|---|
| Label smoothing | Мягкие метки (0.9 вместо 1.0) |
| WGAN | Wasserstein distance вместо BCE |
| Gradient penalty | Штраф за большие градиенты критика |

---

## Урок 87: Diffusion Models

### Forward Process

```
x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

Добавляем шум step by step до чистого шума
```

### Reverse Process

```
x_{t-1} = μ_θ(x_t, t) + σ_t × ε

Нейросеть учится убирать шум
```

---

## Урок 88: Denoising Diffusion

### UNet

```
Input → [Downsample → ResBlock → Attention] × N
     → Bottleneck
     → [Upsample → ResBlock → Skip Connection] × N
```

### Sampling

| Алгоритм | Шаги | Скорость |
|---|---|---|
| DDPM | T (все) | Медленно |
| DDIM | <T (выборочно) | Быстро |

### CFG (Classifier-Free Guidance)

```
ε_guided = ε_uncond + w × (ε_cond - ε_uncond)

w > 1: усиливает условие
```

---

## Урок 89: Stable Diffusion

### Архитектура

```
Text → CLIP Encoder → Text Embeddings
Noisy Image → UNet (с attention) → Noise Prediction
Latent ↔ VAE Encoder/Decoder ↔ Pixel Space
```

---

## Урок 90: Text-to-Image

### Pipeline

```
Prompt → Text Embedding → Latent Noise → Denoise Loop → Image
```

### CLIP

```
similarity = cos(text_embedding, image_embedding)

Чем выше → тем лучше совпадение
```
