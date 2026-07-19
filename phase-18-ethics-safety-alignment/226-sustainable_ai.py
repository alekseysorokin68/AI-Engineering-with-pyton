"""226 — Sustainable AI: энергопотребление, экологическое воздействие

Темы:
  1. Carbon Footprint — углеродный след обучения, инференса, жизненный цикл
  2. Energy Efficiency — сжатие моделей, эффективные архитектуры, аппаратура
  3. Green Computing — возобновляемая энергия, компенсации углерода, планирование
  4. Sustainability Metrics — PUE, углеродная интенсивность, TCO

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# 1. Carbon Footprint — углеродный след ИИ
# ─────────────────────────────────────────────────────────────────────

def demo_carbon_footprint():
    print("=" * 70)
    print("DEMO 1 — Carbon Footprint: углеродный след обучения и инференса")
    print("=" * 70)

    # --- 1.1 Углеродный след обучения моделей ---
    print("\n--- 1.1 Сравнение углеродного следа обучения крупнейших моделей ---")
    # CO2 в тоннах (данные из исследований)
    models = [
        ("BERT-base", 65, 110e6, 1e3),
        ("BERT-large", 150, 340e6, 3e3),
        ("GPT-3 (175B)", 552, 175e9, 3.14e6),
        ("PaLM (540B)", 863, 540e9, 9.2e6),
        ("GPT-4 (est.)", 2500, 1800e9, 25e6),
        ("Llama 3 (405B)", 1200, 405e9, 12e6),
    ]
    print("  {'Модель':<22s} {'CO2 (t)':>10s} {'Параметры':>12s} {'GPU-ч':>12s}")
    for name, co2, params, gpu_hours in models:
        params_b = params / 1e9
        co2_per_param = co2 / (params / 1e6) if params > 0 else 0
        print(f"    {name:<22s} {co2:>8,} t {params_b:>10.0f}B {gpu_hours:>10,.0f} ч")
    # Формула: CO2 = GPU_hours × power_per_gpu × PUE × carbon_intensity
    print("\n  Формула: CO2 = GPU_hours × Power_GPU(kW) × PUE × Carbon_Intensity(gCO2/kWh)")

    # --- 1.2 Инференс vs Обучение ---
    print("\n--- 1.2 Сравнение: углеродный след обучения vs инференса ---")
    # Одноразовый запрос vs всё обучение
    inference_scenarios = [
        ("Один запрос GPT-4", 0.000003, 0.003),
        ("10K запросов/день × 1 год", 10.95, 0.003),
        ("100K запросов/день × 1 год", 109.5, 0.003),
        ("Поиск Google (один)", 0.000007, 0.0005),
        ("Email (один)", 0.000004, 0.0004),
    ]
    training_co2 = 552  # GPT-3 обучение
    print(f"  Обучение GPT-3: {training_co2} т CO2")
    print(f"\n  {'Сценарий':<35s} {'CO2 (г)':>10s} {'Эквивалент':>15s}")
    for name, co2_grams, eq in inference_scenarios:
        # Эквивалент: зарядка телефона
        phone_charges = co2_grams / 8 if co2_grams > 0 else 0
        if phone_charges < 1:
            phone_str = f"{phone_charges*1000:.1f} мЗарядок"
        elif phone_charges < 1000:
            phone_str = f"{phone_charges:.1f} Зарядок"
        else:
            phone_str = f"{phone_charges/1000:.1f}K Зарядок"
        print(f"    {name:<35s} {co2_grams:>10.3f} {phone_str:>15s}")

    # --- 1.3 Жизненный цикл углеродного следа ---
    print("\n--- 1.3 Жизненный цикл ИИ-системы (Lifecycle Analysis) ---")
    lifecycle = {
        "Производство оборудования": {"co2_tons": 150, "pct": 8.5},
        "Обучение модели": {"co2_tons": 552, "pct": 31.3},
        "Инференс (3 года)": {"co2_tons": 850, "pct": 48.3},
        "Дообучение/обновления": {"co2_tons": 120, "pct": 6.8},
        "Утилизация": {"co2_tons": 85, "pct": 4.8},
    }
    total = sum(v["co2_tons"] for v in lifecycle.values())
    print(f"  Всего за жизненный цикл: {total:,.0f} т CO2")
    for stage, info in lifecycle.items():
        bar = "█" * int(info["pct"] / 2)
        print(f"    {stage:<30s}: {info['co2_tons']:>6.0f} т ({info['pct']:>5.1f}%) {bar}")

    # --- 1.4 Водный след ---
    print("\n--- 1.4 Водный след охлаждения дата-центров ---")
    dc_models = [
        ("Локальный сервер", 1.2, 50, 0.8),
        ("Облачный (PUE=1.5)", 2.5, 200, 1.2),
        ("Гипермасштабный (PUE=1.1)", 4.0, 500, 0.6),
        ("GPU-кластер (A100)", 8.5, 1200, 1.8),
    ]
    print("  Формула: Water = Power(MW) × IT_Load × WUE(L/kWh) × Hours")
    print(f"  {'Тип':<30s} {'Мощность(MW)':>12s} {'IT(MWh)':>10s} {'WUE':>6s} {'Вода(м³)':>10s}")
    for name, power, it_load, wue in dc_models:
        water = power * it_load * wue
        print(f"    {name:<30s} {power:>10.1f} MW {it_load:>8.0f} MWh {wue:>4.1f} {water:>8.0f} м³")


# ─────────────────────────────────────────────────────────────────────
# 2. Energy Efficiency — энергоэффективность
# ─────────────────────────────────────────────────────────────────────

def demo_energy_efficiency():
    print("\n" + "=" * 70)
    print("DEMO 2 — Energy Efficiency: сжатие моделей и эффективные решения")
    print("=" * 70)

    # --- 2.1 Методы сжатия моделей ---
    print("\n--- 2.1 Сравнение методов сжатия моделей ---")
    compression = [
        ("FP32 (baseline)", 1.0, 1.0, 1.0),
        ("FP16 Mixed Precision", 0.5, 0.99, 0.85),
        ("INT8 Quantization", 0.25, 0.97, 0.65),
        ("INT4 Quantization", 0.125, 0.93, 0.40),
        ("Pruning (50%)", 0.5, 0.95, 0.75),
        ("Knowledge Distillation", 0.1, 0.88, 0.55),
        ("LoRA (rank=16)", 0.02, 0.96, 0.70),
    ]
    print("  {'Метод':<30s} {'Размер':>8s} {'Качество':>10s} {'Скорость':>10s}")
    for name, size, quality, speed in compression:
        size_bar = "▓" * int(size * 30)
        print(f"    {name:<30s} {size:>6.1f}x {quality:>8.0%}   {speed:>8.0%}  {size_bar}")

    # --- 2.2 Эффективные архитектуры ---
    print("\n--- 2.2 Сравнение архитектур по энергоэффективности ---")
    architectures = [
        ("Dense Transformer", 1.0, 1.0, 1.0),
        ("Sparse Transformer (Mixture of Experts)", 0.4, 0.98, 0.65),
        ("Flash Attention v2", 0.7, 1.0, 1.8),
        ("Linear Attention", 0.3, 0.92, 2.5),
        ("State Space Model (Mamba)", 0.35, 0.95, 2.2),
        ("RWKV", 0.25, 0.90, 3.0),
    ]
    print("  {'Архитектура':<40s} {'FLOPs':>8s} {'Качество':>10s} {'Скорость':>10s}")
    for name, flops, quality, speed in architectures:
        flops_bar = "▓" * int(flops * 20)
        speed_bar = "█" * int(speed * 10)
        print(f"    {name:<40s} {flops:>6.1f}x {quality:>8.0%}   {speed:>6.1f}x {speed_bar}")

    # --- 2.3 Аппаратная оптимизация ---
    print("\n--- 2.3 Сравнение GPU по энергоэффективности (TOPS/Watt) ---")
    gpus = [
        ("NVIDIA V100 (2017)", 125, 300, 0.42),
        ("NVIDIA A100 (2020)", 312, 400, 0.78),
        ("NVIDIA H100 (2022)", 990, 700, 1.41),
        ("NVIDIA H200 (2024)", 1190, 700, 1.70),
        ("Google TPU v5e", 197, 200, 0.99),
        ("AMD MI300X (2024)", 1300, 750, 1.73),
    ]
    print("  {'GPU':<25s} {'TOPS':>8s} {'TDP(W)':>8s} {'TOPS/W':>10s} {'Эффективность':>15s}")
    for name, tops, tdp, tw in gpus:
        bar = "█" * int(tw * 15)
        print(f"    {name:<25s} {tops:>7,} {tdp:>7,} {tw:>8.2f}  {bar}")

    # --- 2.4 Модель энергопотребления по фазам ---
    print("\n--- 2.4 Распределение энергии по фазам жизненного цикла ИИ ---")
    phases = {
        "Предобучение": {"energy_kwh": 1_287_000, "gpu_hours": 3_640_000, "cost_usd": 4_600_000},
        "Fine-tuning": {"energy_kwh": 85_000, "gpu_hours": 240_000, "cost_usd": 300_000},
        "Инференс (год)": {"energy_kwh": 500_000, "gpu_hours": 1_440_000, "cost_usd": 600_000},
        "Хранение данных": {"energy_kwh": 45_000, "gpu_hours": 0, "cost_usd": 25_000},
    }
    total_energy = sum(p["energy_kwh"] for p in phases.values())
    print(f"  Всего энергии: {total_energy:,.0f} кВт·ч")
    for phase, info in phases.items():
        pct = info["energy_kwh"] / total_energy * 100
        bar = "█" * int(pct / 2)
        print(f"    {phase:<20s}: {info['energy_kwh']:>12,} кВт·ч ({pct:>5.1f}%) "
              f"GPU-ч: {info['gpu_hours']:>10,} ${info['cost_usd']:>10,} {bar}")


# ─────────────────────────────────────────────────────────────────────
# 3. Green Computing — зелёные вычисления
# ─────────────────────────────────────────────────────────────────────

def demo_green_computing():
    print("\n" + "=" * 70)
    print("DEMO 3 — Green Computing: возобновляемая энергия и компенсации")
    print("=" * 70)

    # --- 3.1 Источники энергии дата-центров ---
    print("\n--- 3.1 Структура энергоснабжения крупнейших дата-центров ---")
    providers = {
        "Google": {"solar": 0.40, "wind": 0.35, "hydro": 0.15, "gas": 0.10},
        "Microsoft": {"solar": 0.30, "wind": 0.25, "hydro": 0.20, "gas": 0.15, "nuclear": 0.10},
        "Amazon": {"solar": 0.25, "wind": 0.20, "hydro": 0.15, "gas": 0.30, "other": 0.10},
        "Meta": {"solar": 0.35, "wind": 0.30, "hydro": 0.10, "gas": 0.25},
    }
    colors = {"solar": "☀", "wind": "🌬", "hydro": "💧", "gas": "🔥", "nuclear": "⚛", "other": "⚡"}
    print("  {'Провайдер':<15s} {'Солнце':>8s} {'Ветер':>8s} {'Вода':>8s} {'Газ':>8s}")
    for provider, mix in providers.items():
        vals = [f"{mix.get(k, 0)*100:>6.0f}%" for k in ["solar", "wind", "hydro", "gas"]]
        renewable = sum(mix.get(k, 0) for k in ["solar", "wind", "hydro", "nuclear"])
        print(f"    {provider:<15s} {'  '.join(vals)}  (Возобновл.: {renewable*100:.0f}%)")

    # --- 3.2 Модель компенсации углерода ---
    print("\n--- 3.2 Экономика компенсации углерода ---")
    offset_options = [
        ("Посадка деревьев", 5.0, 0.15, 30),
        ("Ветровая энергетика", 25.0, 0.90, 5),
        ("Солнечные панели", 18.0, 0.85, 8),
        ("Углеродные кредиты (VCS)", 15.0, 0.70, 1),
        ("Прямой захват CO2", 250.0, 0.95, 2),
    ]
    co2_to_offset = 1000  # тонн CO2
    print(f"  Цель: компенсировать {co2_to_offset} тонн CO2")
    print("  {'Метод':<30s} {'Стоимость/t':>12s} {'Эффективность':>14s} {'Время(лет)':>12s} {'Итого':>12s}")
    for name, cost_per_ton, effectiveness, duration in offset_options:
        total_cost = co2_to_offset * cost_per_ton / effectiveness
        eff_bar = "█" * int(effectiveness * 15)
        print(f"    {name:<30s} ${cost_per_ton:>10.1f} {effectiveness*100:>12.0f}% {duration:>10d} л ${total_cost:>10,.0f}")

    # --- 3.3 Планирование нагрузки ---
    print("\n--- 3.3 Оптимизация времени обучения по碳интенсивности сети ---")
    # Моделируем углеродную интенсивность электросети по часам
    random.seed(42)
    hours = list(range(24))
    # Ночью меньше нагрузка → ниже углеродная интенсивность
    carbon_intensity = [
        350 + 150 * math.sin((h - 6) * math.pi / 12) + random.randint(-20, 20)
        for h in hours
    ]
    print("  Углеродная интенсивность сети по часам (гCO2/кWh):")
    for h, ci in zip(hours, carbon_intensity):
        bar = "▓" * int(ci / 20)
        marker = " ◄ ЛУЧШЕ" if ci < 350 else ""
        print(f"    {h:>2d}:00 → {ci:>6.0f} гCO2/кWh {bar}{marker}")

    # Оптимальные окна для обучения
    best_hours = [h for h, ci in zip(hours, carbon_intensity) if ci < 350]
    print(f"\n  Оптимальные часы для обучения: {best_hours}")
    avg_best = sum(carbon_intensity[h] for h in best_hours) / len(best_hours)
    avg_all = sum(carbon_intensity) / len(carbon_intensity)
    savings = (1 - avg_best / avg_all) * 100
    print(f"  Средняя интенсивность (лучшие часы): {avg_best:.0f} гCO2/кWh")
    print(f"  Средняя интенсивность (все часы):    {avg_all:.0f} гCO2/кWh")
    print(f"  Экономия CO2: {savings:.1f}%")

    # --- 3.4 Carbon-aware scheduling ---
    print("\n--- 3.4 Carbon-Aware Scheduler: пример расписания ---")
    training_tasks = [
        ("Pretrain layer 0-5", 48, 200),
        ("Pretrain layer 6-11", 48, 200),
        ("Fine-tune head", 24, 100),
        ("Evaluation", 12, 50),
    ]
    print("  {'Задача':<25s} {'Длительность':>14s} {'Мощность(kW)':>14s} {'CO2(кг)':>10s}")
    total_co2 = 0
    for task_name, hours, power_kw in training_tasks:
        # Средняя углеродная интенсивность за лучшие часы
        task_co2 = hours * power_kw * avg_best / 1000
        total_co2 += task_co2
        print(f"    {task_name:<25s} {hours:>12d} ч {power_kw:>12.0f} кВт {task_co2:>8.1f} кг")
    print(f"  {'ИТОГО':<25s} {'':>14s} {'':>14s} {total_co2:>8.1f} кг")


# ─────────────────────────────────────────────────────────────────────
# 4. Sustainability Metrics — метрики устойчивости
# ─────────────────────────────────────────────────────────────────────

def demo_sustainability_metrics():
    print("\n" + "=" * 70)
    print("DEMO 4 — Sustainability Metrics: PUE, TCO, углеродная интенсивность")
    print("=" * 70)

    # --- 4.1 PUE (Power Usage Effectiveness) ---
    print("\n--- 4.1 PUE — Power Usage Effectiveness ---")
    # PUE = Total Facility Energy / IT Equipment Energy
    datacenters = [
        ("Типичный дата-центр", 2.0, 10, 100),
        ("Эффективный (Google)", 1.1, 10, 100),
        ("Hyperscale (AWS)", 1.2, 10, 100),
        ("Локальный сервер", 1.8, 2, 100),
        ("Гипероптимизированный", 1.06, 10, 100),
    ]
    print("  Формула: PUE = Общая_энергия / IT_энергия")
    print("  Идеальный PUE = 1.0 (вся энергия идёт на IT)")
    print(f"  {'Дата-центр':<30s} {'PUE':>6s} {'IT(MW)':>8s} {'Всего(MW)':>10s} {'Потери(MW)':>11s}")
    for name, pue, it_mw, _ in datacenters:
        total = it_mw * pue
        overhead = total - it_mw
        bar = "█" * int(overhead * 5)
        print(f"    {name:<30s} {pue:>5.2f} {it_mw:>7.0f} {total:>9.1f} {overhead:>9.1f} {bar}")

    # --- 4.2 Углеродная интенсивность ---
    print("\n--- 4.2 Углеродная интенсивность электроэнергии по странам ---")
    countries = [
        ("Исландия", 28, "геотрм/гидро"),
        ("Норвегия", 30, "гидро"),
        ("Франция", 56, "атом"),
        ("Канада", 120, "гидро/газ"),
        ("Германия", 350, "уголь/ветер"),
        ("США (средний)", 390, "газ/уголь"),
        ("Китай", 560, "уголь"),
        ("Индия", 710, "уголь"),
        ("Австралия", 530, "уголь/газ"),
    ]
    print("  {'Страна':<20s} {'гCO2/кWh':>10s} {'Основной источник':>20s}")
    for name, ci, source in countries:
        bar = "▓" * int(ci / 25)
        print(f"    {name:<20s} {ci:>8} {source:>20s} {bar}")

    # --- 4.3 Total Cost of Ownership (TCO) ---
    print("\n--- 4.3 TCO — Total Cost of Ownership для ИИ-инфраструктуры ---")
    # 3-летний TCO для GPU-кластера
    configs = [
        ("4×A100 (on-prem)", 4, 64000, 400000, 120000, 1.15),
        ("8×A100 (on-prem)", 8, 64000, 800000, 200000, 1.15),
        ("Cloud A100 (hourly)", 8, 64000, 0, 2800000, 1.10),
        ("4×H100 (on-prem)", 4, 120000, 1200000, 180000, 1.12),
        ("Cloud H100 (reserved)", 8, 120000, 0, 3500000, 1.10),
    ]
    print("  Формула: TCO = Hardware + Energy(3yr) + Cooling + Staff + Depreciation")
    print(f"  {'Конфигурация':<30s} {'GPU':>4s} {'FLOPS':>8s} {'HW($)':>10s} {'Энергия(3yr)':>14s} {'PUE':>5s} {'TCO(3yr)':>12s}")
    for name, gpu_count, flops, hw_cost, energy_3yr, pue in configs:
        cooling = energy_3yr * (pue - 1) / pue
        staff = 50000
        tco = hw_cost + energy_3yr + cooling + staff
        tco_per_flop = tco / (flops * gpu_count) if flops * gpu_count > 0 else 0
        print(f"    {name:<30s} {gpu_count:>4d} {flops:>7,} ${hw_cost:>9,} ${energy_3yr:>13,} {pue:>4.2f} ${tco:>11,}")

    # --- 4.4 Индекс устойчивости модели ---
    print("\n--- 4.4 Composite Sustainability Score (CSS) ---")
    # CSS = w1·Energy_efficiency + w2·Carbon_offset + w3·Water_efficiency + w4·Hardware_recycling
    w1, w2, w3, w4 = 0.35, 0.25, 0.20, 0.20
    models_sustainability = [
        ("GPT-3 (2020)", 0.30, 0.10, 0.40, 0.50),
        ("GPT-4 (2023)", 0.55, 0.40, 0.55, 0.60),
        ("Llama 3 (2024)", 0.70, 0.50, 0.65, 0.70),
        ("Phi-3 (2024)", 0.85, 0.60, 0.75, 0.80),
        ("Gemma 2 (2024)", 0.80, 0.55, 0.70, 0.75),
    ]
    print(f"  Формула: CSS = {w1}·Energy + {w2}·Carbon + {w3}·Water + {w4}·Recycling")
    print(f"  {'Модель':<22s} {'Энергия':>8s} {'Углерод':>8s} {'Вода':>8s} {'Ресайклинг':>11s} {'CSS':>6s}")
    for name, energy, carbon, water, recycling in models_sustainability:
        css = w1*energy + w2*carbon + w3*water + w4*recycling
        bar = "█" * int(css * 30)
        print(f"    {name:<22s} {energy:>7.0%} {carbon:>7.0%} {water:>7.0%} {recycling:>9.0%}  {css:.3f} {bar}")

    # Рейтинг
    print("\n  Рейтинг по устойчивости:")
    scored = []
    for name, energy, carbon, water, recycling in models_sustainability:
        css = w1*energy + w2*carbon + w3*water + w4*recycling
        scored.append((name, css))
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, css) in enumerate(scored, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"    {medal} #{rank} {name:<22s} CSS = {css:.3f}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_carbon_footprint()
    demo_energy_efficiency()
    demo_green_computing()
    demo_sustainability_metrics()
