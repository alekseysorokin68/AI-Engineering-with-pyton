"""
209 — Edge Deployment: оптимизация моделей, ONNX, мобильный инференс

Темы:
  1. Model Optimization (прунинг, квантизация, дистилляция знаний)
  2. ONNX Format (экспорт моделей, runtime, кросс-платформенный деплой)
  3. Mobile Inference (TFLite, Core ML концепции, оптимизация латентности)
  4. Edge Patterns (оффлайн инференс, федеративное обучение, обновления моделей)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections
import time

random.seed(42)


# ──────────────────────────────────────────────────────────────────────
# Демо 1: Оптимизация моделей (Model Optimization)
# ──────────────────────────────────────────────────────────────────────
def demo_model_optimization():
    """Прунинг, квантизация, дистилляция знаний."""
    print("=" * 70)
    print("ДЕМО 1: Оптимизация моделей (Model Optimization)")
    print("=" * 70)

    # --- 1.1 Прунинг (Pruning) ---
    print("\n--- 1.1 Прунинг (Pruning) — удаление незначимых весов ---")

    class SimpleModel:
        """Упрощённая модель: матрица весов + вектор смещения."""

        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            random.seed(42)
            self.w1 = [[random.gauss(0, 0.5) for _ in range(hidden_size)]
                        for _ in range(input_size)]
            self.b1 = [random.gauss(0, 0.1) for _ in range(hidden_size)]
            self.w2 = [[random.gauss(0, 0.5) for _ in range(output_size)]
                        for _ in range(hidden_size)]
            self.b2 = [random.gauss(0, 0.1) for _ in range(output_size)]

        def count_params(self) -> int:
            """Подсчёт общего числа параметров."""
            return (len(self.w1) * len(self.w1[0]) + len(self.b1) +
                    len(self.w2) * len(self.w2[0]) + len(self.b2))

        def prune_magnitude(self, sparsity: float) -> dict:
            """Прунинг по величине весов: обнуляем smallest |w| параметров."""
            # Собираем все веса (без biases для простоты)
            all_weights = []
            for row in self.w1:
                all_weights.extend(row)
            for row in self.w2:
                all_weights.extend(row)

            total = len(all_weights)
            n_prune = int(total * sparsity)

            # Находим порог (n_prune-е по величине)
            sorted_abs = sorted([abs(w) for w in all_weights])
            threshold = sorted_abs[n_prune] if n_prune < total else sorted_abs[-1]

            # Обнуляем веса ниже порога
            pruned_count = 0
            for i in range(len(self.w1)):
                for j in range(len(self.w1[i])):
                    if abs(self.w1[i][j]) < threshold:
                        self.w1[i][j] = 0.0
                        pruned_count += 1
            for i in range(len(self.w2)):
                for j in range(len(self.w2[i])):
                    if abs(self.w2[i][j]) < threshold:
                        self.w2[i][j] = 0.0
                        pruned_count += 1

            return {
                "original_params": total,
                "pruned_params": pruned_count,
                "remaining_params": total - pruned_count,
                "sparsity_achieved": pruned_count / total,
                "threshold": threshold,
            }

    model = SimpleModel(input_size=64, hidden_size=32, output_size=10)
    original_params = model.count_params()
    print(f"  Исходная модель:")
    print(f"    Архитектура: 64 → 32 → 10")
    print(f"    Параметров:  {original_params:,}")
    print(f"    Размер (FP32): {original_params * 4 / 1024:.1f} KB")

    # Разные уровни прунинга
    for sparsity in [0.3, 0.5, 0.7, 0.9]:
        model_copy = SimpleModel(input_size=64, hidden_size=32, output_size=10)
        result = model_copy.prune_magnitude(sparsity)
        remaining = result["remaining_params"]
        size_fp32 = remaining * 4 / 1024
        print(f"\n  Прунинг {sparsity*100:.0f}%:")
        print(f"    Обнулено:        {result['pruned_params']:,} весов")
        print(f"    Осталось:        {result['remaining_params']:,} параметров")
        print(f"    Реальная разреж.: {result['sparsity_achieved']*100:.1f}%")
        print(f"    Размер (FP32):   {size_fp32:.1f} KB")
        print(f"    Формула: new_size = original × (1 - sparsity) = "
              f"{original_params} × {1-sparsity:.1f} = {int(original_params*(1-sparsity))}")

    # --- 1.2 Квантизация (Quantization) ---
    print("\n--- 1.2 Квантизация (Quantization) — снижение точности чисел ---")

    def quantize_fp32_to_int8(values: list) -> tuple:
        """Квантизация FP32 → INT8 (affine quantization)."""
        min_val = min(values)
        max_val = max(values)

        # Вычисление scale и zero_point
        scale = (max_val - min_val) / 255.0  # 256 уровней для INT8
        zero_point = round(-min_val / scale)
        zero_point = max(0, min(255, zero_point))  # ограничиваем диапазон

        # Квантизация
        quantized = [round(v / scale + zero_point) for v in values]
        quantized = [max(0, min(255, q)) for q in quantized]

        # Деквантизация (для оценки ошибки)
        dequantized = [(q - zero_point) * scale for q in quantized]

        # Вычисление ошибки
        errors = [abs(v - dq) for v, dq in zip(values, dequantized)]
        mse = sum(e**2 for e in errors) / len(errors)
        max_error = max(errors)

        return {
            "scale": scale,
            "zero_point": zero_point,
            "quantized": quantized,
            "dequantized": dequantized,
            "mse": mse,
            "max_error": max_error,
            "compression_ratio": 4.0,  # FP32 (4 байта) → INT8 (1 байт)
        }

    # Симуляция весов модели
    random.seed(42)
    weights = [random.gauss(0, 0.5) for _ in range(1000)]

    result = quantize_fp32_to_int8(weights)

    print(f"  Исходные данные: {len(weights)} значений (FP32)")
    print(f"  Диапазон: [{min(weights):.4f}, {max(weights):.4f}]")
    print(f"\n  Параметры квантизации:")
    print(f"    Scale:     {result['scale']:.6f}")
    print(f"    Zero-point: {result['zero_point']}")
    print(f"\n  Качество:")
    print(f"    MSE:       {result['mse']:.8f}")
    print(f"    Max error: {result['max_error']:.6f}")
    print(f"    Сжатие:    {result['compression_ratio']:.0f}x (FP32 → INT8)")
    print(f"\n  Формула: q = round(x / scale + zero_point)")
    print(f"           deq = (q - zero_point) × scale")
    print(f"           scale = (max - min) / 255")

    # Примеры значений
    print(f"\n  Примеры квантизации:")
    print(f"  {'Исходное (FP32)':>15s} {'Квантиз. (INT8)':>16s} {'Деквантиз.':>12s} {'Ошибка':>10s}")
    for i in range(5):
        orig = weights[i]
        quant = result["quantized"][i]
        deq = result["dequantized"][i]
        err = abs(orig - deq)
        print(f"  {orig:>15.6f} {quant:>16d} {deq:>12.6f} {err:>10.6f}")

    # --- 1.3 Дистилляция знаний (Knowledge Distillation) ---
    print("\n--- 1.3 Дистилляция знаний (Knowledge Distillation) ---")

    def softmax(values: list, temperature: float = 1.0) -> list:
        """Softmax с температурой."""
        scaled = [v / temperature for v in values]
        max_v = max(scaled)
        exps = [math.exp(v - max_v) for v in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    def cross_entropy(predicted: list, target: list) -> float:
        """Кросс-энтропия между двумя распределениями."""
        loss = 0.0
        for p, t in zip(predicted, target):
            if p > 0:
                loss -= t * math.log(p + 1e-10)
        return loss

    # Учитель ( большая модель): 10 классов, "мягкие" вероятности
    teacher_logits = [2.1, 0.3, -0.5, 1.8, 0.1, -1.2, 0.8, -0.3, 1.5, -0.7]

    # Студент (малая модель): тоже 10 классов, но менее уверенные
    student_logits = [1.5, 0.5, 0.2, 1.2, 0.3, -0.5, 0.6, 0.1, 1.0, -0.3]

    # Ground truth: one-hot
    true_label = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # класс 3

    print(f"  Учитель (большая модель): 10 классов")
    print(f"  Студент (малая модель):  10 классов")
    print(f"  Истинный класс: 3")

    # Разные температуры
    for temp in [1.0, 2.0, 5.0]:
        teacher_probs = softmax(teacher_logits, temp)
        student_probs = softmax(student_logits, temp)

        # Soft targets от учителя
        print(f"\n  Температура T={temp:.0f}:")
        print(f"    Учитель soft-targets: {[f'{p:.3f}' for p in teacher_probs[:5]]}...")
        print(f"    Стudent  predictions: {[f'{p:.3f}' for p in student_probs[:5]]}...")

        # Hard loss (交叉entropy с ground truth)
        hard_loss = cross_entropy(student_probs, true_label)

        # Soft loss (交叉entropy с soft targets учителя)
        teacher_soft = softmax(teacher_logits, temp)
        soft_loss = cross_entropy(student_probs, teacher_soft)

        # Combined loss: α × soft_loss + (1-α) × hard_loss
        alpha = 0.7
        combined = alpha * soft_loss + (1 - alpha) * hard_loss

        print(f"    Hard loss (CE с truth):      {hard_loss:.4f}")
        print(f"    Soft loss (CE с учителем):    {soft_loss:.4f}")
        print(f"    Combined (α={alpha}):          {combined:.4f}")
        print(f"    Формула: L = α·L_soft(T) + (1-α)·L_hard")

    # Сравнение размеров
    print(f"\n  Сравнение моделей:")
    print(f"    {'Модель':<25s} {'Параметры':>12s} {'Размер':>10s} {'Точность'}")
    print(f"    {'-'*65}")
    teacher_params = 110000000
    student_params = 6000000
    print(f"    {'Учитель (BERT-base)':<25s} {teacher_params:>12,} {teacher_params*4/1e6:>9.0f} MB  91.2%")
    print(f"    {'Студент (BERT-tiny)':<25s} {student_params:>12,} {student_params*4/1e6:>9.1f} MB  87.5%")
    print(f"    {'Студент (дистиллир.)':<25s} {student_params:>12,} {student_params*4/1e6:>9.1f} MB  89.8%")
    print(f"\n  Выигрыш: {teacher_params/student_params:.1f}x меньше параметров, "
          f"+2.3% к точности после дистилляции")

    # --- 1.4 Оценка trade-off ---
    print("\n--- 1.4 Trade-off: точность vs размер vs скорость ---")

    models_comparison = [
        {"name": "FP32 (baseline)", "size_mb": 440, "accuracy": 91.2, "latency_ms": 15.3},
        {"name": "FP16", "size_mb": 220, "accuracy": 91.1, "latency_ms": 11.2},
        {"name": "INT8 (QAT)", "size_mb": 110, "accuracy": 90.5, "latency_ms": 7.8},
        {"name": "INT8 (PTQ)", "size_mb": 110, "accuracy": 89.8, "latency_ms": 8.1},
        {"name": "50% pruning + INT8", "size_mb": 55, "accuracy": 88.9, "latency_ms": 5.2},
        {"name": "Дистиллированный", "size_mb": 24, "accuracy": 89.8, "latency_ms": 3.1},
        {"name": "70% pruning + INT8", "size_mb": 33, "accuracy": 87.2, "latency_ms": 3.8},
    ]

    print(f"  {'Метод':<25s} {'Размер':>8s} {'Точность':>10s} {'Латент.':>10s} {'Сжатие'}")
    print(f"  {'-'*70}")
    baseline = models_comparison[0]
    for m in models_comparison:
        compression = baseline["size_mb"] / m["size_mb"]
        acc_drop = baseline["accuracy"] - m["accuracy"]
        speedup = baseline["latency_ms"] / m["latency_ms"]
        print(f"  {m['name']:<25s} {m['size_mb']:>6d} MB {m['accuracy']:>8.1f}% "
              f"{m['latency_ms']:>8.1f} ms {compression:>5.1f}x")

    print(f"\n  Метрики эффективности:")
    print(f"    Масштабируемость = точность / размер")
    for m in models_comparison:
        efficiency = m["accuracy"] / m["size_mb"]
        print(f"    {m['name']:<25s} efficiency = {m['accuracy']}/{m['size_mb']} = {efficiency:.3f}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 2: ONNX Format
# ──────────────────────────────────────────────────────────────────────
def demo_onnx_format():
    """Экспорт моделей, runtime, кросс-платформенный деплой."""
    print("=" * 70)
    print("ДЕМО 2: ONNX Format — кросс-платформенный формат моделей")
    print("=" * 70)

    # --- 2.1 Структура ONNX-модели ---
    print("\n--- 2.1 Структура ONNX-модели ---")

    # Симуляция ONNX-модели как вычислительного графа
    onnx_model = {
        "ir_version": 8,
        "opset_imports": [{"domain": "", "version": 17}],
        "producer_name": "pytorch",
        "producer_version": "2.1.0",
        "graph": {
            "name": "sentiment_model",
            "inputs": [
                {"name": "input_ids", "type": "tensor(int64)", "shape": [1, 128]},
                {"name": "attention_mask", "type": "tensor(int64)", "shape": [1, 128]},
            ],
            "outputs": [
                {"name": "logits", "type": "tensor(float)", "shape": [1, 3]},
            ],
            "nodes": [
                {"op_type": "MatMul", "name": "encoder.layer.0.qkv", "inputs": ["input_ids", "W_qkv"], "outputs": ["qkv_out"]},
                {"op_type": "Softmax", "name": "encoder.layer.0.attn", "inputs": ["qkv_out"], "outputs": ["attn_out"]},
                {"op_type": "MatMul", "name": "encoder.layer.0.ffn", "inputs": ["attn_out", "W_ffn"], "outputs": ["ffn_out"]},
                {"op_type": "Relu", "name": "encoder.layer.0.activation", "inputs": ["ffn_out"], "outputs": ["relu_out"]},
                {"op_type": "MatMul", "name": "classifier", "inputs": ["relu_out", "W_cls"], "outputs": ["logits"]},
            ],
            "initializers": [
                {"name": "W_qkv", "shape": [768, 2304], "data_type": "FLOAT"},
                {"name": "W_ffn", "shape": [768, 3072], "data_type": "FLOAT"},
                {"name": "W_cls", "shape": [768, 3], "data_type": "FLOAT"},
            ],
        },
    }

    graph = onnx_model["graph"]
    print(f"  ONNX модель: {graph['name']}")
    print(f"  IR version: {onnx_model['ir_version']}")
    print(f"  Opset: {onnx_model['opset_imports'][0]['version']}")
    print(f"  Producer: {onnx_model['producer_name']} {onnx_model['producer_version']}")

    print(f"\n  Входы:")
    for inp in graph["inputs"]:
        print(f"    {inp['name']:<25s} тип={inp['type']:<20s} форма={inp['shape']}")

    print(f"\n  Выходы:")
    for out in graph["outputs"]:
        print(f"    {out['name']:<25s} тип={out['type']:<20s} форма={out['shape']}")

    print(f"\n  Узлы вычислительного графа ({len(graph['nodes'])} узлов):")
    for node in graph["nodes"]:
        print(f"    {node['op_type']:<12s} name={node['name']}")
        print(f"      входы: {node['inputs']}")
        print(f"      выходы: {node['outputs']}")

    print(f"\n  Параметры (initializers): {len(graph['initializers'])} тензоров")
    total_params = 0
    for init in graph["initializers"]:
        params = 1
        for dim in init["shape"]:
            params *= dim
        total_params += params
        print(f"    {init['name']:<15s} форма={init['shape']}  параметров={params:,}")
    print(f"    Всего параметров: {total_params:,}")

    # --- 2.2 Конвертация между форматами ---
    print("\n--- 2.2 Конвертация между форматами ---")

    conversion_paths = [
        ("PyTorch", "ONNX", "torch.onnx.export()", "Прямой экспорт через API"),
        ("TensorFlow", "ONNX", "tf2onnx.convert()", "Через tf2onnx"),
        ("Keras", "ONNX", "tf2onnx (через SavedModel)", "Сначала SavedModel → ONNX"),
        ("sklearn", "ONNX", "skl2onnx.convert_sklearn()", "Через skl2onnx"),
        ("ONNX", "TFLite", "onnx-tf + TFLite converter", "ONNX → TF → TFLite"),
        ("ONNX", "Core ML", "coremltools (через ONNX)", "ONNX → CoreML"),
        ("PyTorch", "TorchScript", "torch.jit.trace/script()", "JIT-компиляция"),
        ("PyTorch", "TorchScript→ONNX", "через ONNX", "Indirect путь"),
    ]

    print(f"  Пути конвертации:")
    print(f"  {'Источник':<12s} → {'Цель':<12s} {'Метод':<35s} {'Замечание'}")
    print(f"  {'-'*95}")
    for src, dst, method, note in conversion_paths:
        print(f"  {src:<12s} → {dst:<12s} {method:<35s} {note}")

    # --- 2.3 ONNX Runtime ---
    print("\n--- 2.3 ONNX Runtime — выполнение моделей ---")

    class ONNXRuntimeSimulator:
        """Симуляция ONNX Runtime с различными оптимизациями."""

        def __init__(self, model_info: dict):
            self.model_info = model_info
            self.optimizations = []
            self.execution_stats = {}

        def add_optimization(self, name: str, speedup: float, description: str):
            """Добавление оптимизации."""
            self.optimizations.append({
                "name": name,
                "speedup": speedup,
                "description": description,
            })

        def execute(self, input_data: list) -> dict:
            """Симуляция выполнения модели."""
            base_latency_ms = 15.3

            # Применяем оптимизации последовательно
            current_latency = base_latency_ms
            applied = []
            for opt in self.optimizations:
                prev = current_latency
                current_latency /= opt["speedup"]
                applied.append({
                    "name": opt["name"],
                    "latency_before": prev,
                    "latency_after": current_latency,
                    "improvement": (prev - current_latency) / prev * 100,
                })

            self.execution_stats = {
                "base_latency_ms": base_latency_ms,
                "final_latency_ms": current_latency,
                "total_speedup": base_latency_ms / current_latency,
                "optimizations_applied": applied,
            }
            return self.execution_stats

    runtime = ONNXRuntimeSimulator({"name": "sentiment_model.onnx"})

    # Добавляем оптимизации
    runtime.add_optimization("Graph Optimization", 1.3, "Устранение冗余 узлов")
    runtime.add_optimization("Constant Folding", 1.15, "Вычисление констант при компиляции")
    runtime.add_optimization("Layer Fusion", 1.25, "Объединение MatMul + Bias + Activation")
    runtime.add_optimization("Memory Planning", 1.1, "Оптимизация использования памяти")

    # Запуск
    test_input = [random.randint(0, 30000) for _ in range(128)]
    stats = runtime.execute(test_input)

    print(f"  ONNX Runtime Execution Provider: CPU")
    print(f"  Базовая латентность: {stats['base_latency_ms']:.1f} ms")
    print(f"\n  Применённые оптимизации:")
    for opt in stats["optimizations_applied"]:
        print(f"    {opt['name']:<25s} {opt['latency_before']:>6.1f}ms → "
              f"{opt['latency_after']:>6.1f}ms (-{opt['improvement']:.1f}%)")

    print(f"\n  Итоговая латентность: {stats['final_latency_ms']:.1f} ms")
    print(f"  Общий ускорение: {stats['total_speedup']:.2f}x")

    # --- 2.4 Кросс-платформенный деплой ---
    print("\n--- 2.4 Кросс-платформенный деплой ---")

    platforms = [
        {"name": "Linux (x86_64)", "ep": "CPUExecutionProvider", "formats": ["ONNX", "PyTorch", "TF"]},
        {"name": "Linux (ARM64)", "ep": "CPUExecutionProvider", "formats": ["ONNX", "TFLite"]},
        {"name": "Windows (x86_64)", "ep": "CPUExecutionProvider", "formats": ["ONNX", "PyTorch"]},
        {"name": "macOS (Apple Silicon)", "ep": "CoreMLExecutionProvider", "formats": ["ONNX", "CoreML"]},
        {"name": "Android (ARM64)", "ep": "NNAPIExecutionProvider", "formats": ["ONNX", "TFLite"]},
        {"name": "iOS (ARM64)", "ep": "CoreMLExecutionProvider", "formats": ["ONNX", "CoreML"]},
        {"name": "NVIDIA GPU", "ep": "CUDAExecutionProvider", "formats": ["ONNX", "PyTorch", "TensorRT"]},
        {"name": "WebAssembly", "ep": "WASMExecutionProvider", "formats": ["ONNX"]},
    ]

    print(f"  Поддержка платформ:")
    print(f"  {'Платформа':<25s} {'Execution Provider':<30s} {'Форматы'}")
    print(f"  {'-'*85}")
    for p in platforms:
        formats_str = ", ".join(p["formats"])
        print(f"  {p['name']:<25s} {p['ep']:<30s} {formats_str}")

    print(f"\n  Преимущество ONNX: одна модель → {len(platforms)} платформ")
    print(f"  Без перекомпиляции на каждой платформе")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 3: Mobile Inference
# ──────────────────────────────────────────────────────────────────────
def demo_mobile_inference():
    """TFLite, Core ML концепции, оптимизация латентности."""
    print("=" * 70)
    print("ДЕМО 3: Mobile Inference — мобильный инференс")
    print("=" * 70)

    # --- 3.1 TFLite (TensorFlow Lite) ---
    print("\n--- 3.1 TFLite (TensorFlow Lite) ---")

    class TFLiteModelSimulator:
        """Симуляция TFLite-модели с метаданными."""

        def __init__(self, name: str, quantized: bool = False):
            self.name = name
            self.quantized = quantized
            self.input_shape = [1, 224, 224, 3]
            self.output_shape = [1, 1001]
            self.delegates = []

        def get_size_mb(self) -> float:
            """Размер модели в MB."""
            if self.quantized:
                return 4.2  # INT8 квантизованная
            return 16.8  # FP32

        def estimate_latency(self, device: str) -> float:
            """Оценка латентности на устройстве."""
            base = 45.0 if not self.quantized else 25.0
            device_factors = {
                "pixel_7": 1.0,
                "iphone_14": 0.8,
                "samsung_s23": 0.9,
                "raspberry_pi_4": 4.5,
                "jetson_nano": 0.6,
            }
            return base * device_factors.get(device, 1.0)

        def add_delegate(self, delegate: str):
            """Добавление делегата (hardware acceleration)."""
            self.delegates.append(delegate)

    # Сравнение FP32 vs INT8
    model_fp32 = TFLiteModelSimulator("mobilenet_v2", quantized=False)
    model_int8 = TFLiteModelSimulator("mobilenet_v2_int8", quantized=True)

    print(f"  MobileNetV2 для мобильного инференса:")
    print(f"\n  {'Параметр':<25s} {'FP32':>12s} {'INT8':>12s} {'Экономия'}")
    print(f"  {'-'*65}")
    print(f"  {'Размер модели':<25s} {model_fp32.get_size_mb():>10.1f} MB "
          f"{model_int8.get_size_mb():>10.1f} MB "
          f"{model_fp32.get_size_mb()/model_int8.get_size_mb():.1f}x")
    print(f"  {'Входная форма':<25s} {str(model_fp32.input_shape):>12s} {str(model_int8.input_shape):>12s}")
    print(f"  {'Выходная форма':<25s} {str(model_fp32.output_shape):>12s} {str(model_int8.output_shape):>12s}")

    # Латентность на разных устройствах
    print(f"\n  Латентность (мс) на устройствах:")
    devices = ["pixel_7", "iphone_14", "samsung_s23", "raspberry_pi_4", "jetson_nano"]
    print(f"  {'Устройство':<20s} {'FP32':>8s} {'INT8':>8s} {'Ускорение'}")
    print(f"  {'-'*50}")
    for device in devices:
        lat_fp32 = model_fp32.estimate_latency(device)
        lat_int8 = model_int8.estimate_latency(device)
        speedup = lat_fp32 / lat_int8
        print(f"  {device:<20s} {lat_fp32:>7.1f} {lat_int8:>7.1f} {speedup:>6.2f}x")

    # --- 3.2 Core ML (Apple) ---
    print("\n--- 3.2 Core ML (Apple ecosystem) ---")

    coreml_model = {
        "name": "SentimentClassifier",
        "version": "1.0",
        "platform": "iOS 16+ / macOS 13+",
        "neural_engine": True,
        "gpu_support": True,
        "compute_units": ["CPU", "GPU", "NeuralEngine"],
        "input_type": "MLMultiArray (Int32, [1, 128])",
        "output_type": "MLMultiArray (Float32, [1, 3])",
        "model_size_mb": 8.5,
        "estimated_latency_ms": {
            "iPhone 14 Pro (Neural Engine)": 3.2,
            "iPhone 14 Pro (GPU)": 5.1,
            "iPhone 14 Pro (CPU)": 12.4,
            "MacBook Pro M2 (GPU)": 2.8,
            "MacBook Pro M2 (CPU)": 8.7,
        },
    }

    print(f"  Core ML модель: {coreml_model['name']}")
    print(f"  Платформа: {coreml_model['platform']}")
    print(f"  Neural Engine: {'Да' if coreml_model['neural_engine'] else 'Нет'}")
    print(f"  Размер: {coreml_model['model_size_mb']} MB")
    print(f"  Compute Units: {', '.join(coreml_model['compute_units'])}")

    print(f"\n  Латентность по compute units:")
    for device, latency in coreml_model["estimated_latency_ms"].items():
        print(f"    {device:<40s} {latency:>6.1f} ms")

    # --- 3.3 Оптимизация латентности ---
    print("\n--- 3.3 Оптимизация латентности на мобильных устройствах ---")

    optimization_techniques = [
        {
            "name": "Квантизация (INT8)",
            "impact": "Размер: 4x↓, Latency: 2-3x↑",
            "tradeoff": "Потеря 0.5-1% точности",
            "implementation": "post-training quantization / QAT",
        },
        {
            "name": "Прунинг (50-70%)",
            "impact": "Размер: 2-3x↓, Latency: 1.5-2x↑",
            "tradeoff": "Потеря 1-3% точности",
            "implementation": "magnitude pruning + fine-tuning",
        },
        {
            "name": "Кэширование промежуточных результатов",
            "impact": "Latency: 1.3-1.5x↑ для повторных входов",
            "tradeoff": "Увеличение потребления памяти",
            "implementation": "LRU cache для эмбеддингов",
        },
        {
            "name": "Batching (групповой инференс)",
            "impact": "Throughput: 2-4x↑, Latency: стабильная",
            "tradeoff": "Увеличение задержки для первого запроса",
            "implementation": "dynamic batching с таймаутом",
        },
        {
            "name": "Предварительная загрузка модели",
            "impact": "First-inference latency: 50-100x↑",
            "tradeoff": "Увеличение потребления RAM на все время",
            "implementation": "model warmup при старте приложения",
        },
        {
            "name": "Hardware-specific оптимизации",
            "impact": "Latency: 1.5-3x↑",
            "tradeoff": "Платформозависимый код",
            "implementation": "NNAPI (Android), CoreML (iOS), Hexagon (Qualcomm)",
        },
    ]

    print(f"  Техники оптимизации мобильного инференса:\n")
    for i, tech in enumerate(optimization_techniques, 1):
        print(f"  {i}. {tech['name']}")
        print(f"     Влияние:     {tech['impact']}")
        print(f"     Trade-off:   {tech['tradeoff']}")
        print(f"     Реализация:  {tech['implementation']}")
        print()

    # --- 3.4 Бенчмарки ---
    print("\n--- 3.4 Бенчмарки мобильного инференса ---")

    benchmarks = [
        {"model": "MobileNetV2", "params": "3.4M", "imagenet_top1": 71.8,
         "iphone14_ms": 5.1, "pixel7_ms": 8.3, "size_int8_mb": 3.4},
        {"model": "EfficientNet-Lite0", "params": "4.7M", "imagenet_top1": 75.1,
         "iphone14_ms": 7.2, "pixel7_ms": 11.5, "size_int8_mb": 4.7},
        {"model": "MobileNetV3-Small", "params": "2.5M", "imagenet_top1": 67.5,
         "iphone14_ms": 3.8, "pixel7_ms": 6.1, "size_int8_mb": 2.5},
        {"model": "NASNetMobile", "params": "5.3M", "imagenet_top1": 74.0,
         "iphone14_ms": 9.5, "pixel7_ms": 14.2, "size_int8_mb": 5.3},
        {"model": "SqueezeNet", "params": "1.2M", "imagenet_top1": 57.5,
         "iphone14_ms": 2.1, "pixel7_ms": 3.8, "size_int8_mb": 1.2},
    ]

    print(f"  {'Модель':<20s} {'Параметры':>10s} {'Top-1':>8s} {'iPhone':>8s} {'Pixel':>8s} {'Размер'}")
    print(f"  {'-'*70}")
    for b in benchmarks:
        print(f"  {b['model']:<20s} {b['params']:>10s} {b['imagenet_top1']:>7.1f}% "
              f"{b['iphone14_ms']:>6.1f}ms {b['pixel7_ms']:>6.1f}ms {b['size_int8_mb']:>4.1f}MB")

    print(f"\n  Trade-off точность vs скорость:")
    for b in benchmarks:
        efficiency = b["imagenet_top1"] / b["iphone14_ms"]
        print(f"    {b['model']:<20s} efficiency = {b['imagenet_top1']}/{b['iphone14_ms']} = "
              f"{efficiency:.1f} (accuracy/ms)")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Демо 4: Edge Patterns
# ──────────────────────────────────────────────────────────────────────
def demo_edge_patterns():
    """Оффлайн инференс, федеративное обучение, обновления моделей."""
    print("=" * 70)
    print("ДЕМО 4: Edge Patterns — паттерны работы на edge-устройствах")
    print("=" * 70)

    # --- 4.1 Оффлайн инференс ---
    print("\n--- 4.1 Оффлайн инференс (Offline Inference) ---")

    class EdgeDevice:
        """Симуляция edge-устройства с оффлайн-инференсом."""

        def __init__(self, device_id: str, model_version: str, storage_mb: float):
            self.device_id = device_id
            self.model_version = model_version
            self.storage_mb = storage_mb
            self.model_size_mb = 8.5
            self.cache = {}  # кэш предсказаний
            self.pending_sync = []  # очередь на синхронизацию

        def infer(self, input_data: dict) -> dict:
            """Выполнение инференса на устройстве (без сети)."""
            input_hash = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()

            # Проверка кэша
            if input_hash in self.cache:
                return {"prediction": self.cache[input_hash], "cached": True}

            # "Инференс" — упрощённая логика
            features = input_data.get("features", [0.5])
            score = sum(features) / len(features)
            prediction = "positive" if score > 0.5 else "negative"

            self.cache[input_hash] = prediction

            # Добавляем в очередь синхронизации
            self.pending_sync.append({
                "input": input_data,
                "prediction": prediction,
                "timestamp": time.time(),
            })

            return {"prediction": prediction, "cached": False}

        def sync_with_server(self) -> dict:
            """Синхронизация с сервером (когда есть сеть)."""
            synced = len(self.pending_sync)
            data = list(self.pending_sync)
            self.pending_sync = []
            return {"synced_records": synced, "data": data}

        def storage_usage(self) -> dict:
            """Использование хранилища."""
            cache_size = len(self.cache) * 0.001  # ~1KB на запись
            return {
                "model": self.model_size_mb,
                "cache": cache_size,
                "total": self.model_size_mb + cache_size,
                "available": self.storage_mb - self.model_size_mb - cache_size,
            }

    # Симуляция работы устройства
    device = EdgeDevice("edge-001", "2.1.0", storage_mb=64.0)

    # Оффлайн-инференс (нет сети)
    print(f"  Устройство: {device.device_id}")
    print(f"  Модель: v{device.model_version} ({device.model_size_mb} MB)")
    print(f"  Сеть: ❌ ОТКЛЮЧЕНА\n")

    test_inputs = [
        {"features": [0.8, 0.7, 0.9], "text": "Great product!"},
        {"features": [0.2, 0.1, 0.3], "text": "Terrible experience"},
        {"features": [0.6, 0.5, 0.7], "text": "Pretty good"},
        {"features": [0.8, 0.7, 0.9], "text": "Great product!"},  # дубликат
    ]

    print(f"  Оффлайн-инференс:")
    for inp in test_inputs:
        result = device.infer(inp)
        cache_status = "📦 из кэша" if result["cached"] else "🆕 вычислено"
        print(f"    {inp['text']:<25s} → {result['prediction']:<10s} {cache_status}")

    print(f"\n  Кэш: {len(device.cache)} уникальных предсказаний")
    print(f"  Ожидает синхронизации: {len(device.pending_sync)} записей")

    # Восстановление сети
    print(f"\n  Сеть: ✅ ВОССТАНОВЛЕНА")
    sync_result = device.sync_with_server()
    print(f"  Синхронизировано: {sync_result['synced_records']} записей")

    storage = device.storage_usage()
    print(f"\n  Использование хранилища:")
    for k, v in storage.items():
        print(f"    {k:<15s} = {v:.2f} MB")

    # --- 4.2 Федеративное обучение ---
    print("\n--- 4.2 Федеративное обучение (Federated Learning) ---")

    class FederatedLearning:
        """Симуляция федеративного обучения."""

        def __init__(self, num_clients: int, model_size: int):
            self.num_clients = num_clients
            self.global_model = [random.gauss(0, 0.1) for _ in range(model_size)]
            self.rounds = []

        def local_training(self, client_id: int, local_data_size: int,
                           epochs: int = 3, lr: float = 0.01) -> dict:
            """Локальное обучение на устройстве."""
            # Симуляция: градиентный спуск на локальных данных
            updates = []
            for param in self.global_model:
                gradient = random.gauss(0, 0.05) * (local_data_size / 1000)
                updated = param - lr * gradient
                updates.append(updated)

            # Вычисление локальной метрики
            loss = sum(abs(u - p) for u, p in zip(updates, self.global_model)) / len(updates)

            return {
                "client_id": client_id,
                "updates": updates,
                "data_size": local_data_size,
                "local_loss": loss,
            }

        def aggregate(self, client_updates: list) -> dict:
            """Агрегация обновлений (FedAvg)."""
            # Взвешенное среднее по размеру данных
            total_data = sum(cu["data_size"] for cu in client_updates)

            new_model = [0.0] * len(self.global_model)
            for cu in client_updates:
                weight = cu["data_size"] / total_data
                for i in range(len(self.global_model)):
                    new_model[i] += cu["updates"][i] * weight

            # Вычисление изменений
            delta = sum(abs(n - o) for n, o in zip(new_model, self.global_model)) / len(new_model)

            self.global_model = new_model
            return {"aggregated_clients": len(client_updates), "total_data": total_data,
                    "model_delta": delta}

    fl = FederatedLearning(num_clients=5, model_size=20)

    print(f"  Федеративное обучение: {fl.num_clients} клиентов")
    print(f"  Размер модели: {len(fl.global_model)} параметров")

    for round_num in range(3):
        print(f"\n  Раунд {round_num + 1}:")

        # Каждый клиент обучается локально
        updates = []
        for client_id in range(fl.num_clients):
            data_size = random.randint(100, 1000)
            update = fl.local_training(client_id, data_size)
            updates.append(update)
            print(f"    Клиент {client_id}: {data_size} образцов, "
                  f"локальный loss={update['local_loss']:.4f}")

        # Агрегация
        agg_result = fl.aggregate(updates)
        print(f"  → Агрегация (FedAvg): {agg_result['aggregated_clients']} клиентов, "
              f"{agg_result['total_data']} образцов, Δ={agg_result['model_delta']:.6f}")

    print(f"\n  Преимущества федеративного обучения:")
    print(f"    ✅ Данные не покидают устройство (приватность)")
    print(f"    ✅ Снижение сетевого трафика (отправляются только градиенты)")
    print(f"    ✅ Соответствие GDPR/CCPA")
    print(f"    ✅ Обучение на свежих данных с устройств")

    # --- 4.3 Обновления моделей на edge ---
    print("\n--- 4.3 Обновления моделей на edge (OTA Updates) ---")

    class OTAUpdateManager:
        """Управление обновлениями моделей на edge-устройствах."""

        def __init__(self):
            self.devices = {}
            self.update_history = []

        def register_device(self, device_id: str, current_version: str,
                           bandwidth_mbps: float = 10.0):
            """Регистрация устройства."""
            self.devices[device_id] = {
                "current_version": current_version,
                "bandwidth_mbps": bandwidth_mbps,
                "status": "online",
                "battery_pct": random.randint(20, 100),
            }

        def check_updates(self, latest_version: str) -> list:
            """Проверка доступных обновлений."""
            updates_needed = []
            for device_id, info in self.devices.items():
                if info["current_version"] != latest_version and info["status"] == "online":
                    updates_needed.append({
                        "device_id": device_id,
                        "current": info["current_version"],
                        "target": latest_version,
                        "bandwidth_mbps": info["bandwidth_mbps"],
                    })
            return updates_needed

        def calculate_update_strategy(self, update_info: dict, model_size_mb: float) -> dict:
            """Расчёт стратегии обновления."""
            bandwidth = update_info["bandwidth_mbps"]
            # Время загрузки = размер (MB) × 8 (бит) / пропускная способность (Мбит/с)
            download_time_sec = model_size_mb * 8 / bandwidth

            # Стратегия
            if download_time_sec < 30:
                strategy = "immediate"
                description = "Немедленное обновление"
            elif download_time_sec < 300:
                strategy = "wifi_only"
                description = "Обновление только по WiFi"
            else:
                strategy = "background"
                description = "Фоновое обновление с приоритетом"

            return {
                "download_time_sec": download_time_sec,
                "strategy": strategy,
                "description": description,
            }

    ota = OTAUpdateManager()

    # Регистрация устройств
    devices_config = [
        ("phone-001", "2.0.0", 50.0),
        ("phone-002", "2.0.0", 15.0),
        ("tablet-001", "1.9.0", 100.0),
        ("iot-001", "1.8.0", 2.0),
        ("iot-002", "2.0.0", 5.0),
    ]

    for device_id, version, bandwidth in devices_config:
        ota.register_device(device_id, version, bandwidth)

    latest_version = "2.1.0"
    model_size_mb = 8.5

    print(f"  Последняя версия модели: v{latest_version}")
    print(f"  Размер модели: {model_size_mb} MB\n")

    updates = ota.check_updates(latest_version)
    print(f"  Устройства, требующие обновления: {len(updates)}\n")
    print(f"  {'Устройство':<15s} {'Текущая':>10s} {'Целевая':>10s} {'Стратегия':<20s} {'Время загр.'}")
    print(f"  {'-'*80}")

    for update in updates:
        strategy = ota.calculate_update_strategy(update, model_size_mb)
        print(f"  {update['device_id']:<15s} v{update['current']:<9s} v{update['target']:<9s} "
              f"{strategy['description']:<20s} {strategy['download_time_sec']:>6.1f}с")

    # --- 4.4 A/B тестирование на edge ---
    print("\n--- 4.4 A/B тестирование на edge-устройствах ---")

    class EdgeABTest:
        """A/B тестирование моделей на edge-устройствах."""

        def __init__(self, experiment_name: str, traffic_split: dict):
            self.experiment_name = experiment_name
            self.traffic_split = traffic_split  # {"control": 0.5, "treatment": 0.5}
            self.results = {"control": [], "treatment": []}

        def assign_variant(self, device_id: str) -> str:
            """Определение варианта для устройства (детерминированное)."""
            h = int(hashlib.md5(device_id.encode()).hexdigest(), 16)
            assignment = h % 100 / 100.0

            cumulative = 0.0
            for variant, weight in self.traffic_split.items():
                cumulative += weight
                if assignment < cumulative:
                    return variant
            return list(self.traffic_split.keys())[-1]

        def record_metric(self, variant: str, metric_name: str, value: float):
            """Запись метрики для варианта."""
            self.results[variant].append({"metric": metric_name, "value": value})

        def analyze(self) -> dict:
            """Анализ результатов A/B теста."""
            analysis = {}
            for variant, records in self.results.items():
                if not records:
                    continue
                values = [r["value"] for r in records]
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                std_val = math.sqrt(variance)
                analysis[variant] = {
                    "count": len(values),
                    "mean": mean_val,
                    "std": std_val,
                    "min": min(values),
                    "max": max(values),
                }
            return analysis

    ab_test = EdgeABTest(
        "Sentiment Model v2.1 vs v2.0",
        traffic_split={"control": 0.5, "treatment": 0.5}
    )

    # Симуляция: 100 устройств
    print(f"  Эксперимент: {ab_test.experiment_name}")
    print(f"  Разделение трафика: {ab_test.traffic_split}\n")

    variant_counts = collections.Counter()
    for i in range(100):
        device_id = f"device-{i:03d}"
        variant = ab_test.assign_variant(device_id)
        variant_counts[variant] += 1

        # Симуляция метрик
        if variant == "control":
            latency = random.gauss(15.3, 2.0)
            accuracy = random.gauss(0.897, 0.02)
        else:
            latency = random.gauss(12.1, 1.8)  # treatment быстрее
            accuracy = random.gauss(0.912, 0.02)  # treatment точнее

        ab_test.record_metric(variant, "latency_ms", latency)
        ab_test.record_metric(variant, "accuracy", accuracy)

    print(f"  Распределение устройств:")
    for variant, count in variant_counts.items():
        print(f"    {variant}: {count} устройств ({count}%)")

    analysis = ab_test.analyze()

    print(f"\n  Результаты:")
    for variant, stats in analysis.items():
        print(f"\n  {variant.upper()}:")
        print(f"    Количество:   {stats['count']}")
        print(f"    Latency:      {stats['mean']:.1f} ± {stats['std']:.1f} ms "
              f"(min={stats['min']:.1f}, max={stats['max']:.1f})")

    # Простая статистическая значимость (Z-test)
    if "control" in analysis and "treatment" in analysis:
        ctrl = analysis["control"]
        treat = analysis["treatment"]

        # Z-тест для latency
        ctrl_mean = ctrl["mean"]
        treat_mean = treat["mean"]
        se = math.sqrt(ctrl["std"]**2 / ctrl["count"] + treat["std"]**2 / treat["count"])
        z_score = (ctrl_mean - treat_mean) / se if se > 0 else 0

        # Приблизительное p-value
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))

        print(f"\n  Статистический анализ (Z-тест для latency):")
        print(f"    Control mean:  {ctrl_mean:.2f} ms")
        print(f"    Treatment mean: {treat_mean:.2f} ms")
        print(f"    Разница:       {ctrl_mean - treat_mean:+.2f} ms")
        print(f"    Z-score:       {z_score:.3f}")
        print(f"    p-value:       {p_value:.4f}")
        print(f"    Стат. значимость: {'✅ ДА (p < 0.05)' if p_value < 0.05 else '❌ НЕТ (p >= 0.05)'}")

        if treat_mean < ctrl_mean:
            improvement = (ctrl_mean - treat_mean) / ctrl_mean * 100
            print(f"\n  Вывод: Treatment модель быстрее на {improvement:.1f}%")
            print(f"  Рекомендация: {'🚀 Развёрнуть treatment на все устройства' if p_value < 0.05 else '⏳ Продолжить тест'}")

    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_model_optimization()
    demo_onnx_format()
    demo_mobile_inference()
    demo_edge_patterns()
