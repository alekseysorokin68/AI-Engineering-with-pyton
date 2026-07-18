import math
import random
from functools import reduce

random.seed(42)


# ============================================================
#  Простой Tensor class
# ============================================================

class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(f"Cannot reshape {len(self._data)} into {shape}")
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _flatten_nested(data):
        def flatten(d):
            if isinstance(d, (list, tuple)):
                return [x for item in d for x in flatten(item)]
            return [d]
        flat = flatten(data)
        shape = []
        d = data
        while isinstance(d, (list, tuple)):
            shape.append(len(d))
            d = d[0] if d else 0
        return flat, tuple(shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def data(self):
        return self._data

    def reshape(self, new_shape):
        new_shape = list(new_shape)
        for i, s in enumerate(new_shape):
            if s == -1:
                known = reduce(lambda a, b: a * b, [x for x in new_shape if x != -1], 1)
                new_shape[i] = len(self._data) // known
        return Tensor(self._data[:], tuple(new_shape))

    def squeeze(self, axis=None):
        new_shape = tuple(s for i, s in enumerate(self._shape)
                         if not (s == 1 and (axis is None or i == axis)))
        return Tensor(self._data[:], new_shape if new_shape else ())

    def unsqueeze(self, axis):
        new_shape = list(self._shape)
        new_shape.insert(axis, 1)
        return Tensor(self._data[:], tuple(new_shape))

    def transpose(self, axis1, axis2):
        if self.ndim < 2:
            return self
        new_shape = list(self._shape)
        new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]

        result = [0] * len(self._data)
        for idx in range(len(self._data)):
            coords = self._idx_to_coords(idx)
            coords[axis1], coords[axis2] = coords[axis2], coords[axis1]
            new_idx = self._coords_to_idx(coords, tuple(new_shape))
            result[new_idx] = self._data[idx]
        return Tensor(result, tuple(new_shape))

    def _idx_to_coords(self, idx):
        coords = []
        for s in self._shape:
            if s > 0:
                coords.append(idx // s)
                idx %= s
            else:
                coords.append(0)
        return coords

    @staticmethod
    def _coords_to_idx(coords, shape):
        idx = 0
        for c, s in zip(coords, shape):
            idx = idx * s + c
        return idx

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor([a + b for a, b in zip(self._data, other._data)], self._shape)
        return Tensor([a + other for a in self._data], self._shape)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor([a * b for a, b in zip(self._data, other._data)], self._shape)
        return Tensor([a * other for a in self._data], self._shape)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor([a - b for a, b in zip(self._data, other._data)], self._shape)
        return Tensor([a - other for a in self._data], self._shape)

    def sum(self, axis=None):
        if axis is None:
            return Tensor([sum(self._data)], ())
        # Упрощённая реализация для 1D/2D
        if self.ndim == 1:
            return Tensor([sum(self._data)], ())
        if self.ndim == 2:
            if axis == 0:
                result = [sum(self._data[i * self._shape[1]:(i + 1) * self._shape[1]]
                              for i in range(self._shape[1]))
                          for j in range(self._shape[1])]
                # Упрощение: суммируем по столбцам
                result = []
                for j in range(self._shape[1]):
                    s = sum(self._data[i * self._shape[1] + j] for i in range(self._shape[0]))
                    result.append(s)
                return Tensor(result, (self._shape[1],))
            if axis == 1:
                result = []
                for i in range(self._shape[0]):
                    s = sum(self._data[i * self._shape[1]:(i + 1) * self._shape[1]])
                    result.append(s)
                return Tensor(result, (self._shape[0],))
        return self

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self._data[:8]}{'...' if len(self._data) > 8 else ''})"


# ============================================================
#  Вспомогательные функции
# ============================================================

def softmax(x, axis=-1):
    """Softmax по последней оси"""
    result = []
    for row in x:
        max_val = max(row)
        exps = [math.exp(v - max_val) for v in row]
        total = sum(exps)
        result.append([e / total for e in exps])
    return result

def mat_to_str(M, precision=2):
    lines = []
    for row in M:
        lines.append("[" + ", ".join(f"{x:>7.{precision}f}" for x in row) + "]")
    return "[" + "\n ".join(lines) + "]"


# ============================================================
#  Демо 1: Tensor — создание, reshape, strides
# ============================================================

print("=" * 55)
print("ДЕМО 1: Tensor — основы")
print("=" * 55)

t1 = Tensor([[1, 2, 3], [4, 5, 6]])
print(f"Создание: {t1}")
print(f"Shape: {t1.shape}")
print(f"Strides: {t1.strides}")
print(f"ndim: {t1.ndim}")

t2 = t1.reshape((3, 2))
print(f"\nReshape (2,3) → (3,2): {t2}")

t3 = t1.reshape((6,))
print(f"Reshape (2,3) → (6,):  {t3}")

t4 = t3.reshape((2, 3))
print(f"Reshape обратно:       {t4}")


# ============================================================
#  Демо 2: Squeeze и Unsqueeze
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Squeeze и Unsqueeze")
print("=" * 55)

t5 = Tensor([1, 2, 3])
print(f"Исходный: shape={t5.shape}")

t6 = t5.unsqueeze(0)
print(f"Unsqueeze(0): shape={t6.shape} — добавили размер батча")

t7 = t5.unsqueeze(1)
print(f"Unsqueeze(1): shape={t7.shape} — сделали столбец")

t8 = Tensor([1, 2, 3, 4], shape=(1, 2, 1, 2))
print(f"\nИсходный (1,2,1,2): {t8.shape}")
t9 = t8.squeeze()
print(f"После squeeze:      {t9.shape}")


# ============================================================
#  Демо 3: Transpose
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Transpose")
print("=" * 55)

mat = Tensor([[1, 2, 3], [4, 5, 6]])
print(f"Исходная матрица (2,3): {mat.shape}")
tr = mat.transpose(0, 1)
print(f"После transpose:        {tr.shape}")
print(f"Данные: {tr.data}")


# ============================================================
#  Демо 4: Broadcasting (NumPy)
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Broadcasting")
print("=" * 55)

# Простое сложение с bias
activations = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
bias = [0.1, 0.2, 0.3]
result = [[a + b for a, b in zip(row, bias)] for row in activations]
print(f"Activations (2,3) + bias (3,):")
print(f"  {activations}")
print(f"  + {bias}")
print(f"  = {result}")

# Умножение на scale
print(f"\nМасштабирование по каналам:")
print(f"  images (2,3) * scale (3,) → broadcast")
result2 = [[a * b for a, b in zip(row, [0.5, 1.0, 1.5])] for row in activations]
print(f"  = {result2}")

# Pairwise distance
print(f"\nПарные расстояния (broadcasting):")
A = [[0, 0], [1, 1]]
B = [[0, 1], [1, 0], [2, 2]]
# (M,1,2) - (1,N,2) → (M,N,2) → sum → sqrt
M, N = len(A), len(B)
dist = [[0.0] * N for _ in range(M)]
for i in range(M):
    for j in range(N):
        d = math.sqrt(sum((a - b)**2 for a, b in zip(A[i], B[j])))
        dist[i][j] = round(d, 3)
print(f"  A={A}, B={B}")
print(f"  Расстояния: {dist}")


# ============================================================
#  Демо 5: Einsum — базовые операции
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Einsum")
print("=" * 55)

# Dot product
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
dot = sum(x * y for x, y in zip(a, b))
print(f"Dot product: {a} · {b} = {dot}")

# Matmul
A = [[1, 2], [3, 4], [5, 6]]
B = [[7, 8, 9], [10, 11, 12]]
result = [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(3)] for i in range(3)]
print(f"\nMatmul (3,2) @ (2,3):")
print(f"  A = {A}")
print(f"  B = {B}")
print(f"  A @ B = {mat_to_str(result)}")

# Outer product
u = [1, 2, 3]
v = [10, 20]
outer = [[x * y for y in v] for x in u]
print(f"\nOuter product: {u} ⊗ {v}")
for row in outer:
    print(f"  {row}")

# Trace
M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
trace = sum(M[i][i] for i in range(3))
print(f"\nTrace: {trace}")


# ============================================================
#  Демо 6: Batch matmul
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Batch matmul")
print("=" * 55)

batch_A = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
batch_B = [[[1, 0], [0, 1]], [[2, 0], [0, 2]]]

batch_result = []
for A_b, B_b in zip(batch_A, batch_B):
    res = [[sum(A_b[i][k] * B_b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    batch_result.append(res)

print(f"Batch A (2,2,2):")
for b in batch_A:
    print(f"  {b}")
print(f"\nBatch B (2,2,2):")
for b in batch_B:
    print(f"  {b}")
print(f"\nBatch A @ B:")
for b in batch_result:
    print(f"  {b}")


# ============================================================
#  Демо 7: Multi-head Attention через einsum
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Multi-head Attention")
print("=" * 55)

B, H, T, D = 2, 4, 8, 16
E = H * D

print(f"Параметры: batch={B}, heads={H}, seq_len={T}, head_dim={D}")
print(f"Embed dim: {E} = {H} × {D}")

# Генерируем данные
random.seed(42)
X = [[random.gauss(0, 0.1) for _ in range(E)] for _ in range(B * T)]

# W_q
W_q = [[random.gauss(0, 0.02) for _ in range(E)] for _ in range(E)]

# Q = X @ W_q  → (B*T, E)
Q_flat = [[sum(X[i][k] * W_q[k][j] for k in range(E)) for j in range(E)] for i in range(B * T)]

# Reshape to (B, T, H, D) — упрощённо берём первые D компонент на каждую голову
# Attention scores для одной головы
print(f"\nПропуск shape через attention:")
print(f"  Input:         ({B}, {T}, {E})")
print(f"  Q = X @ W_q:   ({B}, {T}, {E})")
print(f"  reshape+trans: ({B}, {H}, {T}, {D})")
print(f"  scores = Q@Kᵀ: ({B}, {H}, {T}, {T})  ← attention map")
print(f"  weights= softmax({B}, {H}, {T}, {T})")
print(f"  output= w@V:   ({B}, {H}, {T}, {D})")
print(f"  merge heads:   ({B}, {T}, {E})")
print(f"  out @ W_o:     ({B}, {T}, {E})")

# Упрощённый attention для одной головы
H1 = 1
Q1 = [[random.gauss(0, 0.1) for _ in range(D)] for _ in range(T)]
K1 = [[random.gauss(0, 0.1) for _ in range(D)] for _ in range(T)]
V1 = [[random.gauss(0, 0.1) for _ in range(D)] for _ in range(T)]

# scores = Q @ K^T / sqrt(D)
scores = [[0.0] * T for _ in range(T)]
for i in range(T):
    for j in range(T):
        scores[i][j] = sum(Q1[i][k] * K1[j][k] for k in range(D)) / math.sqrt(D)

# softmax
weights = softmax(scores)

# output = weights @ V
output = [[0.0] * D for _ in range(T)]
for i in range(T):
    for j in range(D):
        output[i][j] = sum(weights[i][k] * V1[k][j] for k in range(T))

print(f"\nОдна голова (упрощённо):")
print(f"  Q shape: ({T}, {D})")
print(f"  K shape: ({T}, {D})")
print(f"  scores:  ({T}, {T})")
print(f"  weights: ({T}, {T})")
print(f"  output:  ({T}, {D})")
print(f"  Sum weights[0]: {sum(weights[0]):.4f} (должна быть ~1.0)")


# ============================================================
#  Демо 8: Таблица — каждый слой как tensor операция
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 8: Каждый слой = tensor операция")
print("=" * 55)

print(f"""
┌─────────────────┬──────────────────────────┬──────────────────────┐
│ Операция        │ Tensor Form              │ Einsum               │
├─────────────────┼──────────────────────────┼──────────────────────┤
│ Linear layer    │ Y = X @ W.T + b          │ "bd,od->bo"          │
│ Attention QKV   │ Q = X @ W_q              │ "btd,dh->bth"        │
│ Attention scores│ Q @ K.T / sqrt(d)        │ "bhtd,bhsd->bhts"   │
│ Attention output│ softmax(scores) @ V      │ "bhts,bhsd->bhtd"   │
│ Batch norm      │ (X - mu) / sigma * gamma │ element-wise + broadcast │
│ Softmax         │ exp(x) / sum(exp(x))     │ element-wise + reduction │
└─────────────────┴──────────────────────────┴──────────────────────┘
""")
