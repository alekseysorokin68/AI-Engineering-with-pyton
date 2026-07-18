"""
53 - Convolution from scratch
Pure Python 2D convolution without numpy/PIL/cv2/torch.
"""

import random

random.seed(42)


# ── helpers ──────────────────────────────────────────────────────────────

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def make_matrix(rows, cols, lo=0.0, hi=255.0):
    return [[round(random.uniform(lo, hi), 1) for _ in range(cols)] for _ in range(rows)]


def print_matrix(m, title="", decimals=1):
    if title:
        print(title)
    for row in m:
        print("  ".join(f"{v:7.{decimals}f}" for v in row))
    print()


# ── padding ──────────────────────────────────────────────────────────────

def pad_zero(mat, pad):
    """Zero-padding around a 2D matrix."""
    if pad == 0:
        return [row[:] for row in mat]
    rows, cols = len(mat), len(mat[0])
    out = zeros(rows + 2 * pad, cols + 2 * pad)
    for r in range(rows):
        for c in range(cols):
            out[r + pad][c + pad] = mat[r][c]
    return out


def pad_reflect(mat, pad):
    """Reflect-padding around a 2D matrix."""
    if pad == 0:
        return [row[:] for row in mat]
    rows, cols = len(mat), len(mat[0])
    out = zeros(rows + 2 * pad, cols + 2 * pad)
    for r in range(rows + 2 * pad):
        for c in range(cols + 2 * pad):
            # map to original coordinates via reflection
            rr = r - pad
            cc = c - pad
            # reflect within [0, rows-1]
            if rr < 0:
                rr = -rr
            if rr >= rows:
                rr = 2 * (rows - 1) - rr
            if cc < 0:
                cc = -cc
            if cc >= cols:
                cc = 2 * (cols - 1) - cc
            rr = max(0, min(rr, rows - 1))
            cc = max(0, min(cc, cols - 1))
            out[r][c] = mat[rr][cc]
    return out


# ── 2D convolution ──────────────────────────────────────────────────────

def conv2d(mat, kernel, stride=1, padding=0, pad_mode="zero"):
    """
    Pure-Python 2D convolution.

    mat:    input matrix (H x W)
    kernel: filter matrix (kH x kW) — NOT flipped (correlation convention,
            same as deep-learning frameworks).
    stride: step between kernel placements
    padding: number of cells to pad on each side
    pad_mode: 'zero' or 'reflect'

    Returns output matrix.
    """
    if pad_mode == "reflect":
        padded = pad_reflect(mat, padding)
    else:
        padded = pad_zero(mat, padding)

    ph, pw = len(padded), len(padded[0])
    kh, kw = len(kernel), len(kernel[0])

    out_h = (ph - kh) // stride + 1
    out_w = (pw - kw) // stride + 1

    out = zeros(out_h, out_w)
    for r in range(out_h):
        for c in range(out_w):
            s = 0.0
            for kr in range(kh):
                for kc in range(kw):
                    s += padded[r * stride + kr][c * stride + kc] * kernel[kr][kc]
            out[r][c] = round(s, 4)
    return out


# ── filter definitions ──────────────────────────────────────────────────

FILTERS = {
    "identity": [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ],
    "sharpen": [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    "blur": [
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ],
    "edge_detect": [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
}


# ══════════════════════════════════════════════════════════════════════════
# DEMOS
# ══════════════════════════════════════════════════════════════════════════

def demo_identity():
    print("=" * 60)
    print("DEMO 1: Convolution with identity kernel")
    print("=" * 60)
    img = make_matrix(5, 5, 0, 20)
    print_matrix(img, "Input (5x5):")
    out = conv2d(img, FILTERS["identity"], stride=1, padding=0)
    print_matrix(out, "After identity kernel (should equal input):")

    match = all(
        abs(img[r + 1][c + 1] - out[r][c]) < 1e-6
        for r in range(len(out))
        for c in range(len(out[0]))
    )
    print(f"  Input == Output: {match}\n")


def demo_filters():
    print("=" * 60)
    print("DEMO 2: Different filters")
    print("=" * 60)
    img = make_matrix(6, 6, 0, 30)
    print_matrix(img, "Input (6x6):")

    for name, kernel in FILTERS.items():
        if name == "identity":
            continue
        out = conv2d(img, kernel, stride=1, padding=1, pad_mode="zero")
        print_matrix(out, f"Filter: {name} (with zero-padding=1):")


def demo_padding():
    print("=" * 60)
    print("DEMO 3: Influence of padding")
    print("=" * 60)
    img = make_matrix(5, 5, 0, 25)
    kernel = FILTERS["sharpen"]
    print_matrix(img, "Input (5x5):")

    for p in [0, 1, 2]:
        out_no_ref = conv2d(img, kernel, stride=1, padding=p, pad_mode="zero")
        out_ref = conv2d(img, kernel, stride=1, padding=p, pad_mode="reflect")
        print(f"--- padding={p}, zero-pad  -> output {len(out_no_ref)}x{len(out_no_ref[0])} ---")
        print_matrix(out_no_ref)
        print(f"--- padding={p}, reflect-pad -> output {len(out_ref)}x{len(out_ref[0])} ---")
        print_matrix(out_ref)


def demo_stride():
    print("=" * 60)
    print("DEMO 4: Influence of stride")
    print("=" * 60)
    img = make_matrix(8, 8, 0, 40)
    kernel = FILTERS["blur"]
    print_matrix(img, "Input (8x8):")

    for s in [1, 2, 3, 4]:
        out = conv2d(img, kernel, stride=s, padding=0)
        print(f"--- stride={s} -> output {len(out)}x{len(out[0])} ---")
        print_matrix(out)


# ── main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_identity()
    demo_filters()
    demo_padding()
    demo_stride()
    print("Done.")
