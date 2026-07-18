"""
58 - Data Augmentation
======================
Методы аугментации изображений на чистом Python (без внешних зависимостей).

Аугментации:
  1. Горизонтальный / вертикальный флип
  2. Поворот (на произвольный угол)
  3. Изменение яркости / контраста
  4. Обрезка (crop)
  5. Гауссов шум

Демо:
  Демо 1 — Базовые аугментации
  Демо 2 — Параметры аугментаций
  Демо 3 — Влияние на классификацию
  Демо 4 — Аугментация vs увеличение данных
"""

import random
import math
import copy

random.seed(42)

# ============================================================
#  Реализация «мини-изображения» — список строк-пикселей
# ============================================================

class Image:
    """Простейшее изображение: ширина x высота пикселей, RGB 0-255."""

    def __init__(self, width, height, pixels=None):
        self.width = width
        self.height = height
        # pixels[h][w] = (r, g, b)
        if pixels is not None:
            self.pixels = copy.deepcopy(pixels)
        else:
            self.pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    def get(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[y][x]
        return (0, 0, 0)

    def set(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def clone(self):
        return Image(self.width, self.height, self.pixels)

    def to_grid(self):
        """Возвращает строку-представление для print()."""
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                r, g, b = self.get(x, y)
                avg = (r + g + b) // 3
                if avg > 170:
                    row += "█"
                elif avg > 100:
                    row += "▓"
                elif avg > 50:
                    row += "░"
                else:
                    row += " "
            lines.append(row)
        return "\n".join(lines)

    def stats(self):
        """Средняя яркость, мин, макс."""
        vals = []
        for y in range(self.height):
            for x in range(self.width):
                r, g, b = self.get(x, y)
                vals.append((r + g + b) / 3)
        return sum(vals) / len(vals), min(vals), max(vals)


def make_test_image(w=12, h=10):
    """Создаёт тестовое изображение-паттерн (градиент + прямоугольники)."""
    img = Image(w, h)
    for y in range(h):
        for x in range(w):
            r = int(50 + 180 * (x / max(w - 1, 1)))
            g = int(30 + 160 * (y / max(h - 1, 1)))
            b = int(200 - 120 * ((x + y) / max(w + h - 2, 1)))
            img.set(x, y, (r, g, b))
    # центральный прямоугольник
    for y in range(h // 3, h * 2 // 3):
        for x in range(w // 4, w * 3 // 4):
            img.set(x, y, (255, 255, 100))
    return img


# ============================================================
#  1. Горизонтальный / вертикальный флип
# ============================================================

def flip_horizontal(img):
    """Отражение по горизонтали (лево <-> право)."""
    out = Image(img.width, img.height)
    for y in range(img.height):
        for x in range(img.width):
            out.set(x, y, img.get(img.width - 1 - x, y))
    return out


def flip_vertical(img):
    """Отражение по вертикали (верх <-> низ)."""
    out = Image(img.width, img.height)
    for y in range(img.height):
        for x in range(img.width):
            out.set(x, y, img.get(x, img.height - 1 - y))
    return out


def flip_both(img):
    """Отражение по обоим осям (180°)."""
    out = Image(img.width, img.height)
    for y in range(img.height):
        for x in range(img.width):
            out.set(x, y, img.get(img.width - 1 - x, img.height - 1 - y))
    return out


# ============================================================
#  2. Поворот
# ============================================================

def rotate_90(img):
    """Поворот на 90° по часовой стрелке (матрица transposed)."""
    out = Image(img.height, img.width)  # ширина/высота меняются местами
    for y in range(img.height):
        for x in range(img.width):
            out.set(img.height - 1 - y, x, img.get(x, y))
    return out


def rotate_270(img):
    """Поворот на 270° (90° против часовой)."""
    out = Image(img.height, img.width)
    for y in range(img.height):
        for x in range(img.width):
            out.set(y, img.width - 1 - x, img.get(x, y))
    return out


def rotate_angle(img, angle_deg):
    """
    Поворот на произвольный угол (nearest-neighbour).
    Новый размер = bounding-box исходного изображения.
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    w, h = img.width, img.height
    cx, cy = w / 2.0, h / 2.0
    new_w = int(abs(w * cos_a) + abs(h * sin_a)) + 1
    new_h = int(abs(w * sin_a) + abs(h * cos_a)) + 1
    ncx, ncy = new_w / 2.0, new_h / 2.0
    out = Image(new_w, new_h)
    for y in range(new_h):
        for x in range(new_w):
            src_x = int(cos_a * (x - ncx) + sin_a * (y - ncy) + cx)
            src_y = int(-sin_a * (x - ncx) + cos_a * (y - ncy) + cy)
            if 0 <= src_x < w and 0 <= src_y < h:
                out.set(x, y, img.get(src_x, src_y))
            else:
                out.set(x, y, (30, 30, 30))
    return out


# ============================================================
#  3. Яркость / контраст
# ============================================================

def clamp(v):
    return max(0, min(255, v))


def adjust_brightness(img, factor):
    """
    factor > 1.0 — ярче, < 1.0 — темнее.
    Каждый канал умножается на factor.
    """
    out = img.clone()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = out.get(x, y)
            out.set(x, y, (clamp(int(r * factor)),
                           clamp(int(g * factor)),
                           clamp(int(b * factor))))
    return out


def adjust_contrast(img, factor):
    """
    factor > 1.0 — больше контраст, < 1.0 — меньше.
    Каждый пиксель сдвигается к среднему значению.
    """
    avg_brightness = 0
    count = img.width * img.height
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = img.get(x, y)
            avg_brightness += (r + g + b) / 3.0
    avg_brightness /= count
    out = img.clone()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = out.get(x, y)
            nr = clamp(int(avg_brightness + factor * (r - avg_brightness)))
            ng = clamp(int(avg_brightness + factor * (g - avg_brightness)))
            nb = clamp(int(avg_brightness + factor * (b - avg_brightness)))
            out.set(x, y, (nr, ng, nb))
    return out


def adjust_brightness_contrast(img, brightness=1.0, contrast=1.0):
    """Комбинация яркости и контраста за один проход."""
    avg_brightness = 0
    count = img.width * img.height
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = img.get(x, y)
            avg_brightness += (r + g + b) / 3.0
    avg_brightness /= count
    out = img.clone()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = out.get(x, y)
            nr = clamp(int(avg_brightness + contrast * (brightness * r - avg_brightness)))
            ng = clamp(int(avg_brightness + contrast * (brightness * g - avg_brightness)))
            nb = clamp(int(avg_brightness + contrast * (brightness * b - avg_brightness)))
            out.set(x, y, (nr, ng, nb))
    return out


# ============================================================
#  4. Обрезка (crop)
# ============================================================

def crop(img, x, y, w, h):
    """Вырезает прямоугольную область из изображения."""
    out = Image(w, h)
    for dy in range(h):
        for dx in range(w):
            out.set(dx, dy, img.get(x + dx, y + dy))
    return out


def random_crop(img, crop_w, crop_h, seed=None):
    """Случайная обрезка с фиксированным seed."""
    if seed is not None:
        random.seed(seed)
    if crop_w > img.width:
        crop_w = img.width
    if crop_h > img.height:
        crop_h = img.height
    max_x = img.width - crop_w
    max_y = img.height - crop_h
    cx = random.randint(0, max_x)
    cy = random.randint(0, max_y)
    return crop(img, cx, cy, crop_w, crop_h), (cx, cy)


def center_crop(img, crop_w, crop_h):
    """Центральная обрезка."""
    cx = (img.width - crop_w) // 2
    cy = (img.height - crop_h) // 2
    return crop(img, cx, cy, crop_w, crop_h)


# ============================================================
#  5. Гауссов шум
# ============================================================

def gaussian_noise(img, sigma=25):
    """
    Добавляет гауссов шум к каждому каналу.
    sigma — стандартное отклонение.
    """
    out = img.clone()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = out.get(x, y)
            # Box-Muller для гауссова шума
            u1 = random.random()
            u2 = random.random()
            while u1 == 0:
                u1 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            nr = clamp(int(r + sigma * z))
            ng = clamp(int(g + sigma * z))
            nb = clamp(int(b + sigma * z))
            out.set(x, y, (nr, ng, nb))
    return out


# ============================================================
#  Пайплайн аугментаций
# ============================================================

class AugmentationPipeline:
    """Последовательное применение нескольких аугментаций."""

    def __init__(self):
        self.transforms = []

    def add(self, func, **kwargs):
        self.transforms.append((func, kwargs))
        return self

    def apply(self, img):
        out = img.clone()
        for func, kwargs in self.transforms:
            out = func(out, **kwargs)
        return out


# ============================================================
#  Демо 1: Базовые аугментации
# ============================================================

def demo1():
    print("=" * 60)
    print("Демо 1: Базовые аугментации")
    print("=" * 60)

    img = make_test_image(12, 10)
    print("\nОригинал:")
    print(img.to_grid())

    print("\n--- Горизонтальный флип ---")
    h_flip = flip_horizontal(img)
    print(h_flip.to_grid())

    print("\n--- Вертикальный флип ---")
    v_flip = flip_vertical(img)
    print(v_flip.to_grid())

    print("\n--- Поворот 90° ---")
    rot90 = rotate_90(img)
    print(rot90.to_grid())

    print("\n--- Поворот 270° ---")
    rot270 = rotate_270(img)
    print(rot270.to_grid())

    print("\n--- Поворот 30° (nearest-neighbour) ---")
    rot30 = rotate_angle(img, 30)
    print(rot30.to_grid())

    print("\n--- Яркость x1.5 ---")
    bright = adjust_brightness(img, 1.5)
    print(bright.to_grid())

    print("\n--- Контраст x2.0 ---")
    contrast = adjust_contrast(img, 2.0)
    print(contrast.to_grid())

    print("\n--- Центральный crop 6x5 ---")
    cr = center_crop(img, 6, 5)
    print(cr.to_grid())

    print("\n--- Случайный crop (seed=42) ---")
    rc, (cx, cy) = random_crop(img, 8, 6, seed=42)
    print(f"Позиция: x={cx}, y={cy}")
    print(rc.to_grid())

    print("\n--- Гауссов шум (sigma=30) ---")
    random.seed(42)
    noisy = gaussian_noise(img, sigma=30)
    print(noisy.to_grid())

    print("\n--- Статистики ---")
    for name, im in [("Оригинал", img), ("Яркость x1.5", bright),
                     ("Контраст x2.0", contrast), ("Шум σ=30", noisy)]:
        avg, mn, mx = im.stats()
        print(f"  {name:20s}: avg={avg:6.1f}  min={mn:4.0f}  max={mx:4.0f}")


# ============================================================
#  Демо 2: Параметры аугментаций
# ============================================================

def demo2():
    print("\n" + "=" * 60)
    print("Демо 2: Параметры аугментаций")
    print("=" * 60)

    img = make_test_image(10, 8)

    # --- Яркость ---
    print("\n--- Яркость: factor = 0.5, 1.0, 2.0 ---")
    for f in [0.5, 1.0, 2.0]:
        aug = adjust_brightness(img, f)
        avg, mn, mx = aug.stats()
        print(f"  factor={f:.1f}  avg={avg:.1f}  min={mn:.0f}  max={mx:.0f}")

    # --- Контраст ---
    print("\n--- Контраст: factor = 0.3, 1.0, 3.0 ---")
    for f in [0.3, 1.0, 3.0]:
        aug = adjust_contrast(img, f)
        avg, mn, mx = aug.stats()
        print(f"  factor={f:.1f}  avg={avg:.1f}  min={mn:.0f}  max={mx:.0f}")

    # --- Поворот ---
    print("\n--- Поворот: угол = 0, 30, 45, 60, 90 ---")
    for angle in [0, 30, 45, 60, 90]:
        aug = rotate_angle(img, angle)
        print(f"  angle={angle:3d}°  size={aug.width}x{aug.height}")

    # --- Шум ---
    print("\n--- Гауссов шум: sigma = 10, 30, 50, 100 ---")
    for s in [10, 30, 50, 100]:
        random.seed(42)
        aug = gaussian_noise(img, sigma=s)
        avg, mn, mx = aug.stats()
        print(f"  sigma={s:3d}  avg={avg:.1f}  min={mn:.0f}  max={mx:.0f}")

    # --- Crop ---
    print("\n--- Crop: размеры ---")
    sizes = [(8, 6), (6, 4), (4, 3)]
    for cw, ch in sizes:
        cr, _ = random_crop(img, cw, ch, seed=42)
        print(f"  crop={cw}x{ch}  -> output={cr.width}x{cr.height}")


# ============================================================
#  Демо 3: Влияние на классификацию
# ============================================================

def demo3():
    print("\n" + "=" * 60)
    print("Демо 3: Влияние аугментаций на классификацию")
    print("=" * 60)

    def simple_classifier(img):
        """
        Простая эвристика-классификатор:
        считает долю светлых пикселей (>150) и доля синего канала.
        Возвращает «класс»: A / B / C / D.
        """
        total = 0
        bright_count = 0
        blue_sum = 0
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.get(x, y)
                total += 1
                if (r + g + b) / 3 > 150:
                    bright_count += 1
                blue_sum += b
        bright_ratio = bright_count / total
        blue_ratio = blue_sum / (total * 255)
        if bright_ratio > 0.3:
            return "A (bright)"
        elif blue_ratio > 0.45:
            return "B (blue)"
        elif bright_ratio > 0.1:
            return "C (medium)"
        else:
            return "D (dark)"

    img = make_test_image(12, 10)
    original_class = simple_classifier(img)
    print(f"\nОригинал -> класс: {original_class}")
    print(f"  яркость={img.stats()[0]:.1f}")

    augments = [
        ("Гориз. флип", flip_horizontal(img)),
        ("Поворот 45°", rotate_angle(img, 45)),
        ("Яркость x1.3", adjust_brightness(img, 1.3)),
        ("Контраст x1.5", adjust_contrast(img, 1.5)),
        ("Шум σ=20", gaussian_noise(img, sigma=20)),
        ("Crop 8x6", center_crop(img, 8, 6)),
    ]

    stable = 0
    changed = 0
    for name, aug in augments:
        cls = simple_classifier(aug)
        avg, _, _ = aug.stats()
        status = "OK" if cls == original_class else "ИЗМЕНЁН"
        if cls == original_class:
            stable += 1
        else:
            changed += 1
        print(f"  {name:20s} -> {cls:15s}  [{status}]  avg={avg:.1f}")

    print(f"\nИтого: стабильных={stable}, изменений={changed} из {len(augments)}")
    print("Вывод: простые аугментации (флип, контраст) сохраняют класс;")
    print("       сильные (поворот, шум, crop) могут изменить.")


# ============================================================
#  Демо 4: Аугментация vs увеличение данных
# ============================================================

def demo4():
    print("\n" + "=" * 60)
    print("Демо 4: Аугментация vs увеличение данных")
    print("=" * 60)

    img = make_test_image(10, 8)

    # --- Вариант 1: просто повторяем оригинал ---
    print("\n--- Вариант 1: Без аугментации (5 копий) ---")
    dataset_no_aug = [img.clone() for _ in range(5)]
    print(f"  Размер датасета: {len(dataset_no_aug)}")
    print(f"  Уникальных изображений: {len(set(id(i) for i in dataset_no_aug))} (все уникальные объекты, но одинаковые пиксели)")

    # --- Вариант 2: аугментация ---
    print("\n--- Вариант 2: С аугментацией (5 различных) ---")
    pipeline = AugmentationPipeline()
    pipeline.add(flip_horizontal)

    augmented = [
        img.clone(),
        flip_horizontal(img),
        flip_vertical(img),
        rotate_90(img),
        adjust_brightness(img, 1.2),
    ]
    names = ["Оригинал", "Флип H", "Флип V", "Поворот 90°", "Яркость x1.2"]

    for name, a in zip(names, augmented):
        avg, _, _ = a.stats()
        print(f"  {name:20s}  avg={avg:.1f}  size={a.width}x{a.height}")

    # --- Вариант 3: комбинированные аугментации ---
    print("\n--- Вариант 3: Комбинированные аугментации (6 шт.) ---")
    random.seed(42)
    combos = []
    combo_names = []
    for i, (angle, bright, do_flip) in enumerate([
        (0, 1.0, False),
        (30, 1.0, False),
        (0, 1.3, False),
        (0, 1.0, True),
        (15, 1.2, False),
        (45, 0.8, True),
    ]):
        tmp = img.clone()
        if do_flip:
            tmp = flip_horizontal(tmp)
        if angle != 0:
            tmp = rotate_angle(tmp, angle)
        if bright != 1.0:
            tmp = adjust_brightness(tmp, bright)
        combos.append(tmp)
        fname = ""
        if do_flip:
            fname += "F"
        if angle != 0:
            fname += f"+R{angle}°"
        if bright != 1.0:
            fname += f"+B{bright}"
        if not fname:
            fname = "orig"
        combo_names.append(fname)

    for name, a in zip(combo_names, combos):
        avg, _, _ = a.stats()
        print(f"  {name:20s}  avg={avg:.1f}  size={a.width}x{a.height}")

    # --- Сравнение ---
    print("\n--- Сравнение ---")
    print(f"  Без аугментации: 5 изображений, все пиксели идентичны")
    print(f"  Простая аугментация: 5 изображений, различаются по отражениям/яркости")
    print(f"  Комбинированная: 6 изображений, максимальное разнообразие")
    print()

    # мера разнообразия — средняя попарная разность пикселей
    def pixel_diff(a, b):
        total = 0
        count = 0
        w = min(a.width, b.width)
        h = min(a.height, b.height)
        for y in range(h):
            for x in range(w):
                ra, ga, ba = a.get(x, y)
                rb, gb, bb = b.get(x, y)
                total += abs(ra - rb) + abs(ga - gb) + abs(ba - bb)
                count += 1
        return total / max(count, 1) / 3.0

    def avg_pairwise_diff(images):
        total = 0
        n = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                total += pixel_diff(images[i], images[j])
                n += 1
        return total / max(n, 1)

    print(f"  Средняя попарная разность пикселей:")
    d1 = avg_pairwise_diff(dataset_no_aug[:5])
    d2 = avg_pairwise_diff(augmented)
    d3 = avg_pairwise_diff(combos)
    print(f"    Без аугментации:      {d1:.1f}")
    print(f"    Простая аугментация:  {d2:.1f}")
    print(f"    Комбинированная:      {d3:.1f}")

    print("\nВывод:")
    print("  Аугментация — это способ увеличить разнообразие данных БЕЗ")
    print("  сбора новых образцов. Разнообразие = лучше обобщение модели.")
    print("  Комбинированные аугментации дают наибольшее разнообразие.")


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  58 — Data Augmentation (чистый Python, без зависимостей)║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo1()
    demo2()
    demo3()
    demo4()

    print("\n" + "=" * 60)
    print("Все демонстрации завершены.")
    print("=" * 60)
