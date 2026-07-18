"""
88 — Denoising Diffusion Probabilistic Models (DDPM)
=====================================================
Самодостаточный скрипт: только stdlib + random.
Реализует ключевые компоненты диффузионных моделей:
  1) UNet-архитектура (упрощённая, на lists)
  2) Предсказание шума (noise prediction)
  3) Алгоритмы сэмплирования: DDPM и DDIM
  4) Classifier-Free Guidance (CFG)

random.seed(42) для воспроизводимости.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

random.seed(42)

# ─────────────────────────────────────────────
#  Вспомогательные математические примитивы
# ─────────────────────────────────────────────

def _zeros(n: int) -> List[float]:
    return [0.0] * n

def _ones(n: int) -> List[float]:
    return [1.0] * n

def _randn(n: int) -> List[float]:
    """Box-Muller + seed → гауссовский вектор длины n."""
    out: List[float] = []
    for _ in range((n + 1) // 2):
        u1 = max(random.random(), 1e-12)
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2.0 * math.pi * u2))
        out.append(r * math.sin(2.0 * math.pi * u2))
    return out[:n]

def _add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def _sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def _scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]

def _mul(a: List[float], b: List[float]) -> List[float]:
    return [x * y for x, y in zip(a, b)]

def _matvec(W: List[List[float]], x: List[float]) -> List[float]:
    """W: (out, in), x: (in,) → (out,)."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in W]

def _relu(v: List[float]) -> List[float]:
    return [max(0.0, x) for x in v]

def _silu(v: List[float]) -> List[float]:
    """Swish: x * sigmoid(x)."""
    return [x / (1.0 + math.exp(-x)) for x in v]

def _layernorm(v: List[float], eps: float = 1e-5) -> List[float]:
    n = len(v)
    mu = sum(v) / n
    var = sum((x - mu) ** 2 for x in v) / n
    std = math.sqrt(var + eps)
    return [(x - mu) / std for x in v]

def _sinusoidal_emb(t: float, dim: int) -> List[float]:
    """Позиционное синусоидальное эмбеддинг времени t (dim чётное)."""
    half = dim // 2
    freqs = [math.exp(-math.log(10000.0) * i / half) for i in range(half)]
    args = [t * f for f in freqs]
    emb = []
    for a in args:
        emb.append(math.sin(a))
    for a in args:
        emb.append(math.cos(a))
    return emb

# ─────────────────────────────────────────────
#  Шумовой расписание (noise schedule)
# ─────────────────────────────────────────────

class NoiseSchedule:
    """
    Линейное расписание β_t от t=0..T.
    Хранит α_t, ᾱ_t (cumprod), и предвычисленные коэффициенты
    для DDPM reverse step.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas: List[float] = []
        self.alphas: List[float] = []
        self.alpha_bars: List[float] = []
        # DDPM reverse
        self.alpha_bar_prev: List[float] = []
        self.coeff1: List[float] = []
        self.coeff2: List[float] = []

        ab = 1.0
        for t in range(T):
            b = beta_start + (beta_end - beta_start) * t / (T - 1)
            a = 1.0 - b
            self.betas.append(b)
            self.alphas.append(a)
            ab *= a
            self.alpha_bars.append(ab)

        for t in range(T):
            ab_t = self.alpha_bars[t]
            ab_prev = self.alpha_bars[t - 1] if t > 0 else 1.0
            self.alpha_bar_prev.append(ab_prev)
            c1 = math.sqrt(ab_prev) * self.betas[t] / (1.0 - ab_t)
            c2 = math.sqrt(self.alphas[t]) * (1.0 - ab_prev) / (1.0 - ab_t)
            self.coeff1.append(c1)
            self.coeff2.append(c2)

        # DDIM
        self.sqrt_ab: List[float] = [math.sqrt(a) for a in self.alpha_bars]
        self.sqrt_one_minus_ab: List[float] = [math.sqrt(1.0 - a) for a in self.alpha_bars]
        self.sqrt_recip_ab: List[float] = [1.0 / math.sqrt(a) for a in self.alpha_bars]

    def add_noise(self, x0: List[float], t: int, noise: List[float]) -> List[float]:
        """x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · ε"""
        ab = self.sqrt_ab[t]
        one_m = self.sqrt_one_minus_ab[t]
        return _add(_scale(x0, ab), _scale(noise, one_m))

    def ddpm_step(self, x_t: List[float], pred_noise: List[float], t: int) -> List[float]:
        """Один reverse step по DDPM."""
        mean = _scale(x_t, self.coeff2[t])
        mean = _add(mean, _scale(pred_noise, self.coeff1[t]))
        if t > 0:
            z = _randn(len(x_t))
            beta_t = self.betas[t]
            return _add(mean, _scale(z, math.sqrt(beta_t)))
        return mean

# ─────────────────────────────────────────────
#  UNet-архитектура (упрощённая)
# ─────────────────────────────────────────────

class Linear:
    """Полносвязный слой: out = W @ x + b."""

    def __init__(self, in_f: int, out_f: int):
        self.W: List[List[float]] = []
        self.b: List[float] = _zeros(out_f)
        for _ in range(out_f):
            row = []
            for _ in range(in_f):
                row.append(random.gauss(0.0, math.sqrt(2.0 / in_f)))
            self.W.append(row)

    def __call__(self, x: List[float]) -> List[float]:
        return _add(_matvec(self.W, x), self.b)


class ResBlock:
    """
    Упрощённый ResBlock:
        out = x + Linear(SiLU(LayerNorm(Linear(x + time_emb))))
    """

    def __init__(self, dim: int):
        self.norm = LayerNormModule(dim)
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim)
        self.time_proj = Linear(dim, dim)

    def __call__(self, x: List[float], time_emb: List[float]) -> List[float]:
        t = self.time_proj(time_emb)
        h = _add(x, t)
        h = self.norm(h)
        h = _silu(h)
        h = self.fc1(h)
        h = _silu(h)
        h = self.fc2(h)
        # residual
        return _add(x, h)


class LayerNormModule:
    def __init__(self, dim: int):
        self.eps = 1e-5
    def __call__(self, x: List[float]) -> List[float]:
        return _layernorm(x, self.eps)


class UNet:
    """
    Минимальная UNet-архитектура для шумоподавления.

    Структура:
        in_proj  → ResBlock(down) → ResBlock(mid) → ResBlock(up) → out_proj

    На каждом уровне — один ResBlock + time embedding.
    Skip-connections: out = ResBlock_up(mid_out + ResBlock_down(x))
    """

    def __init__(self, dim: int = 32, time_dim: int = 32):
        self.time_dim = time_dim
        self.time_mlp1 = Linear(time_dim, dim)
        self.time_mlp2 = Linear(dim, dim)

        self.in_proj = Linear(dim, dim)

        self.down1 = ResBlock(dim)
        self.down2 = ResBlock(dim)
        self.mid1 = ResBlock(dim)
        self.up1 = ResBlock(dim)
        self.up2 = ResBlock(dim)
        self.out_proj = Linear(dim, dim)

    def _time_embedding(self, t: int) -> List[float]:
        emb = _sinusoidal_emb(float(t) / 1000.0, self.time_dim)
        emb = self.time_mlp1(emb)
        emb = _silu(emb)
        emb = self.time_mlp2(emb)
        emb = _silu(emb)
        return emb

    def __call__(self, x: List[float], t: int) -> List[float]:
        """
        x: входной вектор (flatten image / latent)
        t: timestep
        Returns: predicted noise ε_θ(x_t, t)
        """
        te = self._time_embedding(t)

        h = self.in_proj(x)
        skip1 = self.down1(h, te)
        skip2 = self.down2(skip1, te)
        mid = self.mid1(skip2, te)

        # skip-connections
        u1 = _add(mid, skip2)
        u1 = self.up1(u1, te)
        u2 = _add(u1, skip1)
        u2 = self.up2(u2, te)

        out = self.out_proj(u2)
        return out


# ─────────────────────────────────────────────
#  Noise Predictor (wrapper)
# ─────────────────────────────────────────────

class NoisePredictor:
    """Оборачивает UNet + NoiseSchedule для удобства."""

    def __init__(self, unet: UNet, schedule: NoiseSchedule):
        self.unet = unet
        self.schedule = schedule

    def predict_noise(self, x_t: List[float], t: int) -> List[float]:
        return self.unet(x_t, t)

    def ddpm_sample(self, shape: int) -> List[float]:
        """Полный DDPM sampling loop: x_T → x_0."""
        x = _randn(shape)
        print(f"  [DDPM] x_T  (t={self.schedule.T}) → norm={_norm(x):.4f}")
        for t in reversed(range(self.schedule.T)):
            pred = self.predict_noise(x, t)
            x = self.schedule.ddpm_step(x, pred, t)
            if t % 200 == 0 or t == 0:
                print(f"  [DDPM] t={t:>4d}  norm={_norm(x):.4f}")
        return x

    def ddim_sample(self, shape: int, steps: int = 50, eta: float = 0.0) -> List[float]:
        """
        DDIM sampling: ускоренный (steps << T).
        eta=0 → детерминированный; eta=1 → как DDPM.
        """
        T = self.schedule.T
        # Sub-sequence of timesteps
        stride = T // steps
        ts = list(range(0, T, stride))
        ts = list(reversed(ts))

        x = _randn(shape)
        print(f"  [DDIM] x_T  (t={T}) steps={steps} eta={eta}  norm={_norm(x):.4f}")

        for i, t in enumerate(ts):
            pred = self.predict_noise(x, t)
            ab_t = self.schedule.alpha_bars[t]
            ab_prev = self.schedule.alpha_bars[ts[i + 1]] if i + 1 < len(ts) else 1.0

            # x0 prediction
            x0_pred = _scale(_sub(x, _scale(pred, math.sqrt(1.0 - ab_t))), 1.0 / math.sqrt(ab_t))

            # direction pointing to x_t
            dir_xt = _scale(pred, math.sqrt(max(1.0 - ab_prev - eta ** 2 * (1.0 - ab_prev) * (1.0 - ab_t) / (1.0 - ab_t + 1e-10), 0.0)))

            noise_term = _zeros(shape)
            if t > 0 and eta > 0:
                sigma = eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * math.sqrt(1.0 - ab_t / ab_prev + 1e-10)
                noise_term = _scale(_randn(shape), sigma)

            x = _add(_add(_scale(x0_pred, math.sqrt(ab_prev)), dir_xt), noise_term)

            if t % 200 == 0 or t == 0:
                print(f"  [DDIM] t={t:>4d}  norm={_norm(x):.4f}")

        return x


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v) / len(v))


# ─────────────────────────────────────────────
#  Classifier-Free Guidance (CFG)
# ─────────────────────────────────────────────

class CFGWrapper:
    """
    Classifier-Free Guidance.

    ε_guided = ε_uncond + w · (ε_cond − ε_uncond)

    Где:
        ε_uncond  = модель без условия (class=null)
        ε_cond    = модель с условием  (class=c)
        w         = guidance scale (>1 усиливает условие)
    """

    def __init__(self, unet: UNet, schedule: NoiseSchedule, w: float = 7.5):
        self.unet = unet
        self.schedule = schedule
        self.w = w

    def predict_guided(self, x_t: List[float], t: int, condition: List[float]) -> List[float]:
        """
       condition: вектор-условие (например, класс или текстовый embedding).
        """
        # unconditional: condition заменяется на zeros
        uncond = self.unet(x_t, t)

        # conditional: condition добавляется к входу
        x_cond = _add(x_t, condition)
        cond = self.unet(x_cond, t)

        # guidance formula
        guided = _add(uncond, _scale(_sub(cond, uncond), self.w))
        return guided

    def sample(self, shape: int, condition: List[float]) -> List[float]:
        """DDPM sampling с CFG."""
        x = _randn(shape)
        print(f"  [CFG w={self.w}] x_T → norm={_norm(x):.4f}")
        for t in reversed(range(self.schedule.T)):
            pred = self.predict_guided(x, t, condition)
            x = self.schedule.ddpm_step(x, pred, t)
            if t % 250 == 0 or t == 0:
                print(f"  [CFG] t={t:>4d}  norm={_norm(x):.4f}")
        return x


# ═════════════════════════════════════════════
#  ДЕМОНСТРАЦИИ
# ═════════════════════════════════════════════

def demo1_unet_forward():
    """
    Демо 1: UNet forward pass
    ─────────────────────────
    Прогоняем фиктивный вход через UNet и смотрим на размерность
    и масштаб выхода.
    """
    print("=" * 60)
    print("DEMO 1: UNet Forward Pass")
    print("=" * 60)

    random.seed(42)
    dim = 16
    unet = UNet(dim=dim, time_dim=dim)

    x = _randn(dim)
    t = 100

    out = unet(x, t)

    print(f"  input  dim = {len(x)},  norm = {_norm(x):.4f}")
    print(f"  output dim = {len(out)},  norm = {_norm(out):.4f}")
    print(f"  timestep   = {t}")
    print()

    # Количество параметров (грубая оценка)
    total_params = 0
    for layer_name in ["in_proj", "out_proj"]:
        layer = getattr(unet, layer_name)
        total_params += len(layer.W) * len(layer.W[0]) + len(layer.b)
    for block_name in ["down1", "down2", "mid1", "up1", "up2"]:
        block = getattr(unet, block_name)
        for sub in [block.fc1, block.fc2, block.time_proj]:
            total_params += len(sub.W) * len(sub.W[0]) + len(sub.b)

    print(f"  ~estimated params: {total_params:,}")
    print()


def demo2_noise_prediction():
    """
    Демо 2: Noise Prediction
    ─────────────────────────
    Показываем, как модель предсказывает шум на зашумлённом входе.
    """
    print("=" * 60)
    print("DEMO 2: Noise Prediction")
    print("=" * 60)

    random.seed(42)
    dim = 12
    schedule = NoiseSchedule(T=1000)
    unet = UNet(dim=dim, time_dim=16)
    predictor = NoisePredictor(unet, schedule)

    # чистый сигнал
    x0 = [0.5] * dim
    noise = _randn(dim)

    print(f"  x0 (clean):          {_fmt_vec(x0[:4])} ...")
    print(f"  noise ε:             {_fmt_vec(noise[:4])} ...")
    print()

    for t in [0, 100, 500, 999]:
        xt = schedule.add_noise(x0, t, noise)
        pred = predictor.predict_noise(xt, t)
        cos_sim = _cosine_sim(noise, pred)
        print(f"  t={t:>4d}  |x_t|={_norm(xt):.4f}  |pred|={_norm(pred):.4f}  "
              f"cos(ε,ε̂)={cos_sim:.4f}")

    print()
    print("  ↑ cos→1 при корректном обучении, пока модель не обучена")
    print("    предсказание случайное → cos≈0")
    print()


def demo3_ddpm_vs_ddim():
    """
    Демо 3: DDPM vs DDIM Sampling
    ───────────────────────────────
    Сравниваем скорость и результат DDPM (T=1000 шагов) vs DDIM (50 шагов).
    """
    print("=" * 60)
    print("DEMO 3: DDPM vs DDIM Sampling")
    print("=" * 60)

    random.seed(42)
    dim = 8
    T = 200  # уменьшено для скорости
    schedule = NoiseSchedule(T=T)
    unet = UNet(dim=dim, time_dim=16)
    predictor = NoisePredictor(unet, schedule)

    # DDPM
    print("\n  --- DDPM (полный T=200 шагов) ---")
    import time
    t0 = time.time()
    x_ddpm = predictor.ddpm_sample(dim)
    t_ddpm = time.time() - t0
    print(f"  Done: {t_ddpm:.3f}s  final_norm={_norm(x_ddpm):.4f}")

    # DDIM (50 шагов)
    print("\n  --- DDIM (50 шагов, eta=0.0) ---")
    t0 = time.time()
    x_ddim = predictor.ddim_sample(dim, steps=50, eta=0.0)
    t_ddim = time.time() - t0
    print(f"  Done: {t_ddim:.3f}s  final_norm={_norm(x_ddim):.4f}")

    # DDIM (20 шагов, eta=0.5)
    print("\n  --- DDIM (20 шагов, eta=0.5) ---")
    t0 = time.time()
    x_ddim2 = predictor.ddim_sample(dim, steps=20, eta=0.5)
    t_ddim2 = time.time() - t0
    print(f"  Done: {t_ddim2:.3f}s  final_norm={_norm(x_ddim2):.4f}")

    print()
    print(f"  Сравнение: DDPM {t_ddpm:.3f}s vs DDIM(50) {t_ddim:.3f}s "
          f"vs DDIM(20) {t_ddim2:.3f}s")
    print(f"  DDIM ускорение (50 steps): ~{t_ddpm/max(t_ddim,1e-9):.1f}x")
    print(f"  DDIM ускорение (20 steps): ~{t_ddpm/max(t_ddim2,1e-9):.1f}x")
    print()


def demo4_cfg():
    """
    Демо 4: Classifier-Free Guidance
    ──────────────────────────────────
    Показываем, как guidance scale w влияет на предсказание шума.
    """
    print("=" * 60)
    print("DEMO 4: Classifier-Free Guidance (CFG)")
    print("=" * 60)

    random.seed(42)
    dim = 12
    T = 200
    schedule = NoiseSchedule(T=T)
    unet = UNet(dim=dim, time_dim=16)

    cfg_w3 = CFGWrapper(unet, schedule, w=3.0)
    cfg_w7 = CFGWrapper(unet, schedule, w=7.5)
    cfg_w15 = CFGWrapper(unet, schedule, w=15.0)

    # фиктивное условие (one-hot или embedding)
    condition = _zeros(dim)
    condition[2] = 1.0
    condition[5] = 1.0

    x_test = _randn(dim)
    t_test = 100

    print(f"\n  condition = {condition}")
    print(f"  x_t norm  = {_norm(x_test):.4f},  t = {t_test}\n")

    # Без условия
    uncond = unet(x_test, t_test)
    print(f"  ε_uncond  norm = {_norm(uncond):.4f}")

    # С условием
    x_cond_input = _add(x_test, condition)
    cond = unet(x_cond_input, t_test)
    print(f"  ε_cond    norm = {_norm(cond):.4f}")

    print()
    for w, wrapper in [(3.0, cfg_w3), (7.5, cfg_w7), (15.0, cfg_w15)]:
        guided = wrapper.predict_guided(x_test, t_test, condition)
        diff = _norm(_sub(guided, uncond))
        print(f"  w={w:>4.1f}  ε_guided norm = {_norm(guided):.4f}  "
              f"|ε_guided − ε_uncond| = {diff:.4f}")

    print()
    print("  ↑ w越大, 引导越强, 输出越偏离无条件预测")
    print("    w=1.0 → 无引导 (≈ ε_cond)")
    print("    w=7.5 → 常用值 (文本到图像)")
    print("    w>10  → 过强引导, 可能失真")
    print()


# ─────────────────────────────────────────────
#  Вспомогательная функция форматирования
# ─────────────────────────────────────────────

def _fmt_vec(v: List[float]) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  88 — Denoising Diffusion Probabilistic Models (DDPM) ║")
    print("║  Pure Python · No NumPy · No PyTorch                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo1_unet_forward()
    demo2_noise_prediction()
    demo3_ddpm_vs_ddim()
    demo4_cfg()

    print("═" * 60)
    print("All demos complete.")
    print("═" * 60)
