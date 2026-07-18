"""
52 - Image Representations
===========================
Basics of representing and manipulating images in pure Python.
No external libraries (numpy, PIL, cv2, torch) — only stdlib.

Topics:
  - Image as a 2D pixel matrix (grayscale)
  - Brightness, contrast, inversion
  - Histogram
  - Box blur and edge detection (Sobel-like)
"""

import random

random.seed(42)

# ============================================================
# 1. Core image representation
# ============================================================

def create_gradient(width, height):
    """Create a grayscale gradient image (left-to-right)."""
    return [[int(x / max(width - 1, 1) * 255) for x in range(width)]
            for _ in range(height)]


def create_checkerboard(width, height, block=4):
    """Create a checkerboard pattern."""
    return [
        [255 if (x // block + y // block) % 2 == 0 else 0
         for x in range(width)]
        for y in range(height)
    ]


def create_random_image(width, height):
    """Create a random grayscale image (seeded)."""
    return [[random.randint(0, 255) for _ in range(width)]
            for _ in range(height)]


def create_circle(width, height, radius=None):
    """Create an image with a white circle on black background."""
    if radius is None:
        radius = min(width, height) // 3
    cx, cy = width // 2, height // 2
    return [
        [255 if ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2 else 0
         for x in range(width)]
        for y in range(height)
    ]


def print_ascii(image, scale=1):
    """Print an image as ASCII art."""
    chars = " .:-=+*#%@"
    h = len(image)
    w = len(image[0]) if h > 0 else 0
    for y in range(0, h, max(1, scale)):
        row = ""
        for x in range(0, w, max(1, scale)):
            v = image[y][x]
            idx = min(v * (len(chars) - 1) // 255, len(chars) - 1)
            row += chars[idx]
        print(row)


# ============================================================
# 2. Pixel operations
# ============================================================

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))


def adjust_brightness(image, delta):
    """Add delta to every pixel."""
    return [[clamp(p + delta) for p in row] for row in image]


def adjust_contrast(image, factor):
    """Multiply pixel deviation from 128 by factor."""
    return [[clamp(int(128 + (p - 128) * factor)) for p in row]
            for row in image]


def invert(image):
    """Invert all pixels: 255 - p."""
    return [[255 - p for p in row] for row in image]


def threshold(image, t=128):
    """Binarize: pixels >= t become 255, else 0."""
    return [[255 if p >= t else 0 for p in row] for row in image]


# ============================================================
# 3. Histogram
# ============================================================

def histogram(image, bins=256):
    """Return a list of counts for each intensity level."""
    hist = [0] * bins
    step = 256 // bins
    for row in image:
        for p in row:
            idx = min(p // step, bins - 1)
            hist[idx] += 1
    return hist


def print_histogram(hist, width=50):
    """Print a horizontal histogram bar chart."""
    max_val = max(hist) if hist else 1
    step = 256 // len(hist)
    for i, count in enumerate(hist):
        if count == 0:
            continue
        bar_len = int(count / max_val * width)
        label = f"{i * step:3d}-{(i + 1) * step - 1:3d}"
        print(f"  {label} | {'#' * bar_len} ({count})")


# ============================================================
# 4. Filters
# ============================================================

def get_pixel(image, y, x):
    h = len(image)
    w = len(image[0]) if h > 0 else 0
    if 0 <= y < h and 0 <= x < w:
        return image[y][x]
    return 0  # zero padding


def box_blur(image, radius=1):
    """Apply a box blur (average filter)."""
    h = len(image)
    w = len(image[0]) if h > 0 else 0
    result = [[0] * w for _ in range(h)]
    size = (2 * radius + 1) ** 2
    for y in range(h):
        for x in range(w):
            total = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    total += get_pixel(image, y + dy, x + dx)
            result[y][x] = total // size
    return result


def edge_detect_sobel(image):
    """Sobel-like edge detection using horizontal and vertical kernels."""
    h = len(image)
    w = len(image[0]) if h > 0 else 0
    result = [[0] * w for _ in range(h)]

    gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            sx = sy = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    p = get_pixel(image, y + dy, x + dx)
                    sx += p * gx[dy + 1][dx + 1]
                    sy += p * gy[dy + 1][dx + 1]
            mag = int((sx ** 2 + sy ** 2) ** 0.5)
            result[y][x] = clamp(mag)
    return result


def sharpen(image):
    """Sharpen filter using an unsharp mask kernel."""
    h = len(image)
    w = len(image[0]) if h > 0 else 0
    result = [[0] * w for _ in range(h)]
    kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            total = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    total += get_pixel(image, y + dy, x + dx) * kernel[dy + 1][dx + 1]
            result[y][x] = clamp(total)
    return result


# ============================================================
# DEMO 1: Create and visualize images
# ============================================================

def demo1():
    print("=" * 55)
    print("DEMO 1: Creating and visualizing images")
    print("=" * 55)

    print("\n--- Gradient (16x8) ---")
    grad = create_gradient(16, 8)
    print_ascii(grad)

    print("\n--- Checkerboard (16x8, block=2) ---")
    checker = create_checkerboard(16, 8, block=2)
    print_ascii(checker)

    print("\n--- Circle (20x12) ---")
    circ = create_circle(20, 12)
    print_ascii(circ)

    print("\n--- Random image (12x6) ---")
    rand_img = create_random_image(12, 6)
    print_ascii(rand_img)


# ============================================================
# DEMO 2: Pixel operations
# ============================================================

def demo2():
    print("\n" + "=" * 55)
    print("DEMO 2: Pixel operations")
    print("=" * 55)

    img = create_gradient(20, 10)
    print("\nOriginal (20x10 gradient):")
    print_ascii(img)

    bright = adjust_brightness(img, 80)
    print("\nBrightness +80:")
    print_ascii(bright)

    contrast = adjust_contrast(img, 2.0)
    print("\nContrast x2.0:")
    print_ascii(contrast)

    inv = invert(img)
    print("\nInverted:")
    print_ascii(inv)

    binary = threshold(img, 128)
    print("\nThreshold (t=128):")
    print_ascii(binary)


# ============================================================
# DEMO 3: Histogram
# ============================================================

def demo3():
    print("\n" + "=" * 55)
    print("DEMO 3: Histogram")
    print("=" * 55)

    # Small 4-bin histogram for readability
    img = create_gradient(20, 10)
    print("\nGradient image histogram (4 bins):")
    hist = histogram(img, bins=4)
    print_histogram(hist, width=40)

    # Random image with more bins
    rand_img = create_random_image(30, 15)
    print("\nRandom image histogram (8 bins):")
    hist2 = histogram(rand_img, bins=8)
    print_histogram(hist2, width=40)

    # Show pixel distribution stats
    flat = [p for row in rand_img for p in row]
    mean_val = sum(flat) / len(flat)
    min_val = min(flat)
    max_val = max(flat)
    print(f"\nStats: mean={mean_val:.1f}, min={min_val}, max={max_val}, "
          f"pixels={len(flat)}")


# ============================================================
# DEMO 4: Filters
# ============================================================

def demo4():
    print("\n" + "=" * 55)
    print("DEMO 4: Filters (blur, edge detection, sharpen)")
    print("=" * 55)

    # Use a circle image — good for showing filter effects
    img = create_circle(20, 12, radius=4)
    print("\nOriginal circle (20x12):")
    print_ascii(img)

    blurred = box_blur(img, radius=1)
    print("\nBox blur (radius=1):")
    print_ascii(blurred)

    edges = edge_detect_sobel(img)
    print("\nSobel edge detection:")
    print_ascii(edges)

    sharp = sharpen(img)
    print("\nSharpen filter:")
    print_ascii(sharp)

    # Show filter on random image
    rand_img = create_random_image(16, 8)
    print("\nRandom image:")
    print_ascii(rand_img)

    rand_blur = box_blur(rand_img, radius=1)
    print("\nRandom image after box blur:")
    print_ascii(rand_blur)

    rand_edges = edge_detect_sobel(rand_img)
    print("\nRandom image edges:")
    print_ascii(rand_edges)


# ============================================================
# Run all demos
# ============================================================

if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
    print("\n" + "=" * 55)
    print("All demos complete.")
    print("=" * 55)
