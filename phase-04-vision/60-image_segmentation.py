"""
Phase 04 — Vision: Основы сегментации изображений
==================================================
Демонстрация ключевых концепций image segmentation без внешних зависимостей.

Реализовано:
  1. Пороговая сегментация (thresholding) — метод Оцу
  2. Кластеризация пикселей (K-Means)
  3. Connected components (связные компоненты)
  4. Метрика IoU для масок сегментации
"""

import random

random.seed(42)


# =============================================================================
# Вспомогательные функции: работа с изображениями как списками списков
# =============================================================================

def create_image(height, width, fill=0):
    """Создаёт изображение (height x width), заполненное fill."""
    return [[fill] * width for _ in range(height)]


def image_from_values(values, height, width):
    """Создаёт изображение из плоского списка значений [0..255]."""
    img = []
    idx = 0
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(values[idx])
            idx += 1
        img.append(row)
    return img


def get_pixel(img, r, c):
    """Возвращает значение пикселя (grayscale 0-255)."""
    return img[r][c]


def set_pixel(img, r, c, value):
    """Устанавливает значение пикселя."""
    img[r][c] = value


def image_shape(img):
    """Возвращает (height, width)."""
    return len(img), len(img[0])


def flatten_image(img):
    """Возвращает плоский список всех значений пикселей."""
    return [v for row in img for v in row]


# =============================================================================
# 1. Пороговая сегментация (Thresholding) — метод Оцу
# =============================================================================

def compute_histogram(values, num_bins=256):
    """
    Вычисляет гистограмму значений [0..255].

    Возвращает список длиной num_bins, где hist[i] — количество пикселей
    со значением i (или попадающим в бин i).
    """
    hist = [0] * num_bins
    for v in values:
        idx = min(int(v), num_bins - 1)
        hist[idx] += 1
    return hist


def otsu_threshold(values):
    """
    Находит оптимальный порог методом Оцу.

    Алгоритм:
      1. Строим гистограмму яркостей.
      2. Для каждого возможного порога t:
         - вычисляем веса классов w0, w1
         - вычисляем средние классов mu0, mu1
         - вычисляем межклассовую дисперсию sigma2 = w0 * w1 * (mu0 - mu1)^2
      3. Порог t* — тот, при котором sigma2 максимален.

    Возвращает:
      threshold — оптимальный порог
      histogram — гистограмма
    """
    hist = compute_histogram(values, num_bins=256)
    total = sum(hist)

    best_thresh = 0
    best_variance = 0.0

    for t in range(256):
        w0 = sum(hist[:t])
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue

        sum0 = sum(i * hist[i] for i in range(t))
        sum1 = sum(i * hist[i] for i in range(t, 256))

        mu0 = sum0 / w0
        mu1 = sum1 / w1

        variance = w0 * w1 * (mu0 - mu1) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = t

    return best_thresh, hist


def apply_threshold(img, threshold):
    """
    Бинарная сегментация: пиксели >= threshold → 255, иначе → 0.

    Возвращает бинарную маску (0 или 255).
    """
    h, w = image_shape(img)
    mask = create_image(h, w, 0)
    for r in range(h):
        for c in range(w):
            if get_pixel(img, r, c) >= threshold:
                set_pixel(mask, r, c, 255)
    return mask


def apply_threshold_multi(img, thresholds):
    """
    Мультиуровневая сегментация по нескольким порогам.

    thresholds — отсортированный список порогов.
    Возвращает маску с labels 0, 1, 2, ..., len(thresholds).
    """
    h, w = image_shape(img)
    labels = create_image(h, w, 0)
    for r in range(h):
        for c in range(w):
            val = get_pixel(img, r, c)
            label = 0
            for t in thresholds:
                if val >= t:
                    label += 1
            labels[r][c] = label
    return labels


# =============================================================================
# 2. Кластеризация пикселей (K-Means)
# =============================================================================

def kmeans_init_centroids(values, k, rng):
    """
    Инициализация центроидов методом k-means++ (упрощённый).

    values — плоский список интенсивностей пикселей
    k      — число кластеров
    rng    — экземпляр random.Random для воспроизводимости

    Возвращает список k центроидов (значения яркости).
    """
    centroids = [rng.choice(values)]

    for _ in range(1, k):
        distances = []
        for v in values:
            min_dist = min((v - c) ** 2 for c in centroids)
            distances.append(min_dist)

        total_dist = sum(distances)
        if total_dist == 0:
            centroids.append(rng.choice(values))
            continue

        probs = [d / total_dist for d in distances]
        cumprobs = []
        cumsum = 0.0
        for p in probs:
            cumsum += p
            cumprobs.append(cumsum)

        r_val = rng.random()
        for i, cp in enumerate(cumprobs):
            if cp >= r_val:
                centroids.append(values[i])
                break

    return sorted(centroids)


def kmeans_segmentation(img, k, max_iter=50, tol=1e-4):
    """
    K-Means кластеризация пикселей для сегментации.

    img      — изображение (список списков, grayscale)
    k        — число кластеров
    max_iter — максимальное число итераций
    tol      — порог сходимости (изменение центроидов)

    Возвращает:
      labels   — маска кластеров (0..k-1)
      centroids — финальные центроиды
    """
    h, w = image_shape(img)
    values = flatten_image(img)
    rng = random.Random(42)
    centroids = kmeans_init_centroids(values, k, rng)

    labels_flat = [0] * len(values)

    for iteration in range(max_iter):
        # Шаг 1: назначение каждого пикселя ближайшему центроиду
        for i, v in enumerate(values):
            best_c = 0
            best_dist = float('inf')
            for ci, c in enumerate(centroids):
                dist = (v - c) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_c = ci
            labels_flat[i] = best_c

        # Шаг 2: обновление центроидов
        new_centroids = [0.0] * k
        counts = [0] * k
        for i, v in enumerate(values):
            c = labels_flat[i]
            new_centroids[c] += v
            counts[c] += 1

        for ci in range(k):
            if counts[ci] > 0:
                new_centroids[ci] /= counts[ci]
            else:
                new_centroids[ci] = centroids[ci]

        # Проверка сходимости
        shift = sum((new_centroids[i] - centroids[i]) ** 2 for i in range(k)) ** 0.5
        centroids = new_centroids

        if shift < tol:
            break

    # Переводим плоский labels обратно в 2D
    labels = create_image(h, w, 0)
    idx = 0
    for r in range(h):
        for c in range(w):
            labels[r][c] = labels_flat[idx]
            idx += 1

    return labels, centroids


def labels_to_visualization(labels, centroids):
    """
    Converts cluster labels to grayscale visualization.

    Каждый кластер получает яркость, пропорциональную его центроиду.
    """
    h, w = image_shape(labels)
    vis = create_image(h, w, 0)
    for r in range(h):
        for c in range(w):
            label = labels[r][c]
            vis[r][c] = int(max(0, min(255, centroids[label])))
    return vis


# =============================================================================
# 3. Connected Components (связные компоненты)
# =============================================================================

def connected_components(mask, connectivity=4):
    """
    Находит связные компоненты в бинарной маске.

    mask         — бинарное изображение (0 = фон, ненулевое = объект)
    connectivity — 4 (NSEW) или 8 (включая диагонали)

    Возвращает:
      num_components — число компонент
      labels         — маска с номерами компонент (1, 2, ..., num_components)
    """
    h, w = image_shape(mask)
    labels = create_image(h, w, 0)
    current_label = 0

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            if get_pixel(mask, r, c) != 0 and get_pixel(labels, r, c) == 0:
                current_label += 1
                # BFS
                queue = [(r, c)]
                set_pixel(labels, r, c, current_label)
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in neighbors:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if get_pixel(mask, nr, nc) != 0 and get_pixel(labels, nr, nc) == 0:
                                set_pixel(labels, nr, nc, current_label)
                                queue.append((nr, nc))

    return current_label, labels


def component_stats(labels, num_components):
    """
    Считает статистику по каждой компоненте.

    Возвращает список словарей:
      {label, area, bbox: (min_r, min_c, max_r, max_c), centroid: (cr, cc)}
    """
    h, w = image_shape(labels)
    stats = []
    for lbl in range(1, num_components + 1):
        min_r, min_c = h, w
        max_r, max_c = -1, -1
        sum_r, sum_c = 0, 0
        area = 0
        for r in range(h):
            for c in range(w):
                if get_pixel(labels, r, c) == lbl:
                    area += 1
                    sum_r += r
                    sum_c += c
                    min_r = min(min_r, r)
                    min_c = min(min_c, c)
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)
        centroid_r = sum_r / area if area > 0 else 0
        centroid_c = sum_c / area if area > 0 else 0
        stats.append({
            'label': lbl,
            'area': area,
            'bbox': (min_r, min_c, max_r, max_c),
            'centroid': (centroid_r, centroid_c),
        })
    return stats


def relabel_components(labels, num_components):
    """
    Перенумеровывает компоненты по убыванию площади.

    Возвращает (new_labels, new_num, sorted_stats).
    """
    stats = component_stats(labels, num_components)
    sorted_stats = sorted(stats, key=lambda s: s['area'], reverse=True)

    h, w = image_shape(labels)
    old_to_new = {}
    for new_idx, s in enumerate(sorted_stats, start=1):
        old_to_new[s['label']] = new_idx

    new_labels = create_image(h, w, 0)
    for r in range(h):
        for c in range(w):
            old_lbl = get_pixel(labels, r, c)
            if old_lbl in old_to_new:
                set_pixel(new_labels, r, c, old_to_new[old_lbl])

    return new_labels, num_components, sorted_stats


# =============================================================================
# 4. Метрика IoU для сегментации
# =============================================================================

def compute_segmentation_iou(mask_a, mask_b):
    """
    Вычисляет IoU (Intersection over Union) для двух бинарных масок.

    mask_a, mask_b — бинарные изображения (0 и ненулевое = позитив)

    IoU = |A ∩ B| / |A ∪ B|

    Возвращает:
      iou         — значение IoU [0, 1]
      intersection — число пикселей пересечения
      union        — число пикселей объединения
    """
    h, w = image_shape(mask_a)
    intersection = 0
    union = 0

    for r in range(h):
        for c in range(w):
            a = get_pixel(mask_a, r, c) != 0
            b = get_pixel(mask_b, r, c) != 0
            if a and b:
                intersection += 1
            if a or b:
                union += 1

    iou = intersection / union if union > 0 else 0.0
    return iou, intersection, union


def compute_pixel_accuracy(pred_mask, gt_mask):
    """Вычисляет pixel accuracy — долю правильно размеченных пикселей."""
    h, w = image_shape(pred_mask)
    correct = 0
    total = h * w
    for r in range(h):
        for c in range(w):
            if get_pixel(pred_mask, r, c) == get_pixel(gt_mask, r, c):
                correct += 1
    return correct / total if total > 0 else 0.0


def dice_coefficient(mask_a, mask_b):
    """
    Dice coefficient (F1 для масок).

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    h, w = image_shape(mask_a)
    intersection = 0
    count_a = 0
    count_b = 0

    for r in range(h):
        for c in range(w):
            a = get_pixel(mask_a, r, c) != 0
            b = get_pixel(mask_b, r, c) != 0
            if a:
                count_a += 1
            if b:
                count_b += 1
            if a and b:
                intersection += 1

    denom = count_a + count_b
    return 2.0 * intersection / denom if denom > 0 else 0.0


def mean_iou_per_class(pred_labels, gt_labels, num_classes):
    """
    mIoU — средний IoU по классам (включая фон).

    pred_labels, gt_labels — целочисленные маски (0, 1, 2, ...)
    num_classes            — общее число классов (включая фон)
    """
    h, w = image_shape(pred_labels)
    ious = []

    for cls in range(num_classes):
        intersection = 0
        union = 0
        for r in range(h):
            for c in range(w):
                pred_pos = get_pixel(pred_labels, r, c) == cls
                gt_pos = get_pixel(gt_labels, r, c) == cls
                if pred_pos and gt_pos:
                    intersection += 1
                if pred_pos or gt_pos:
                    union += 1
        iou = intersection / union if union > 0 else 1.0
        ious.append(iou)

    return sum(ious) / len(ious) if ious else 0.0, ious


# =============================================================================
# Генерация синтетических изображений
# =============================================================================

def create_synthetic_scene(height=64, width=64):
    """
    Создаёт синтетическое изображение с чёткими областями.

    Фон — тёмный (30), три объекта:
      - Прямоугольник сверху-слева (яркость 200)
      - Круг в центре (яркость 120)
      - Прямоугольник снизу-справа (яркость 240)
    """
    img = create_image(height, width, 30)

    # Прямоугольник 1: яркость 200
    for r in range(5, 20):
        for c in range(5, 25):
            set_pixel(img, r, c, 200)

    # Круг в центре: яркость 120
    center_r, center_c, radius = 32, 32, 12
    for r in range(height):
        for c in range(width):
            dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
            if dist <= radius:
                set_pixel(img, r, c, 120)

    # Прямоугольник 2: яркость 240
    for r in range(45, 60):
        for c in range(40, 60):
            set_pixel(img, r, c, 240)

    return img


def create_gradient_image(height=32, width=32):
    """Создаёт изображение с горизонтальным градиентом 0-255."""
    img = create_image(height, width, 0)
    for r in range(height):
        for c in range(width):
            val = int(c / (width - 1) * 255)
            set_pixel(img, r, c, val)
    return img


def create_ground_truth_mask(height=64, width=64):
    """
    Создаёт ground truth маску для синтетического сцены.

    0 = фон, 1 = прямоугольник, 2 = круг, 3 = прямоугольник 2
    """
    mask = create_image(height, width, 0)

    for r in range(5, 20):
        for c in range(5, 25):
            mask[r][c] = 1

    center_r, center_c, radius = 32, 32, 12
    for r in range(height):
        for c in range(width):
            dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
            if dist <= radius:
                mask[r][c] = 2

    for r in range(45, 60):
        for c in range(40, 60):
            mask[r][c] = 3

    return mask


def create_noisy_image(base_img, noise_level=15):
    """Добавляет шум к изображению."""
    h, w = image_shape(base_img)
    noisy = create_image(h, w, 0)
    rng = random.Random(42)
    for r in range(h):
        for c in range(w):
            val = get_pixel(base_img, r, c) + rng.randint(-noise_level, noise_level)
            noisy[r][c] = max(0, min(255, val))
    return noisy


def print_image_ascii(img, scale=1, label=""):
    """Печатает изображение в виде ASCII-арта."""
    h, w = image_shape(img)
    if label:
        print(f"  {label}")
    chars = " .:-=+*#%@"
    step_h = max(1, h // (20 * scale))
    step_w = max(1, w // (40 * scale))
    for r in range(0, h, step_h):
        line = "  "
        for c in range(0, w, step_w):
            val = get_pixel(img, r, c)
            idx = int(val / 255 * (len(chars) - 1))
            line += chars[idx] * scale
        print(line)


# =============================================================================
# Демо 1: Пороговая сегментация (Otsu)
# =============================================================================

def demo_thresholding():
    print("=" * 65)
    print("Демо 1: Пороговая сегментация (метод Оцу)")
    print("=" * 65)
    print()

    img = create_synthetic_scene(64, 64)
    values = flatten_image(img)
    threshold, hist = otsu_threshold(values)

    print(f"  Изображение: 64x64 синтетическая сцена")
    print(f"  Объекты: прямоугольник (200), круг (120), прямоугольник (240)")
    print(f"  Фон: 30")
    print(f"  Диапазон яркостей: [{min(values)}, {max(values)}]")
    print(f"  Средняя яркость: {sum(values)/len(values):.1f}")
    print(f"  Оптимальный порог Оцу: {threshold}")
    print()

    # Бинарная сегментация
    binary_mask = apply_threshold(img, threshold)
    values_mask = flatten_image(binary_mask)
    num_fg = sum(1 for v in values_mask if v > 0)
    num_bg = sum(1 for v in values_mask if v == 0)
    print(f"  Бинарная маска:")
    print(f"    Фон (0)   : {num_bg} пикселей ({num_bg/len(values_mask)*100:.1f}%)")
    print(f"    Объект (1): {num_fg} пикселей ({num_fg/len(values_mask)*100:.1f}%)")
    print()

    print_image_ascii(img, scale=1, label="Исходное изображение:")
    print()
    print_image_ascii(binary_mask, scale=1, label=f"Бинарная маска (порог={threshold}):")
    print()

    # Мультиуровневая сегментация
    thresholds = [80, 160, 220]
    multi_mask = apply_threshold_multi(img, thresholds)
    multi_values = flatten_image(multi_mask)
    print(f"  Мультиуровневая сегментация (пороги: {thresholds}):")
    for label_val in range(len(thresholds) + 1):
        count = sum(1 for v in multi_values if v == label_val)
        pct = count / len(multi_values) * 100
        print(f"    Класс {label_val}: {count} пикселей ({pct:.1f}%)")
    print()

    # Влияние шума
    noisy = create_noisy_image(img, noise_level=15)
    noisy_thresh, _ = otsu_threshold(flatten_image(noisy))
    noisy_mask = apply_threshold(noisy, noisy_thresh)
    gt_mask_binary = create_image(64, 64, 0)
    for r in range(64):
        for c in range(64):
            if get_pixel(img, r, c) > 80:
                gt_mask_binary[r][c] = 255
    iou_clean, _, _ = compute_segmentation_iou(binary_mask, gt_mask_binary)
    iou_noisy, _, _ = compute_segmentation_iou(noisy_mask, gt_mask_binary)
    print(f"  Влияние шума:")
    print(f"    Без шума : IoU с GT = {iou_clean:.4f}")
    print(f"    С шумом  : IoU с GT = {iou_noisy:.4f}")
    print(f"    Разница  : {abs(iou_clean - iou_noisy):.4f}")
    print()


# =============================================================================
# Демо 2: K-Means для сегментации
# =============================================================================

def demo_kmeans():
    print("=" * 65)
    print("Демо 2: K-Means кластеризация для сегментации")
    print("=" * 65)
    print()

    img = create_synthetic_scene(64, 64)
    values = flatten_image(img)

    print(f"  Изображение: 64x64 синтетическая сцена (4 уровня яркости)")
    print(f"  Уникальные значения: {sorted(set(values))}")
    print()

    # K=3 кластера
    print(f"  --- K=3 ---")
    labels_3, centroids_3 = kmeans_segmentation(img, k=3, max_iter=50)
    print(f"  Центроиды: {[f'{c:.1f}' for c in centroids_3]}")

    flat_labels = flatten_image(labels_3)
    for k_val in range(3):
        count = sum(1 for v in flat_labels if v == k_val)
        pct = count / len(flat_labels) * 100
        print(f"    Кластер {k_val}: {count} пикселей ({pct:.1f}%) — центроид {centroids_3[k_val]:.1f}")
    print()

    # K=4 кластера (правильное число)
    print(f"  --- K=4 (правильное число классов) ---")
    labels_4, centroids_4 = kmeans_segmentation(img, k=4, max_iter=50)
    print(f"  Центроиды: {[f'{c:.1f}' for c in centroids_4]}")

    flat_labels_4 = flatten_image(labels_4)
    for k_val in range(4):
        count = sum(1 for v in flat_labels_4 if v == k_val)
        pct = count / len(flat_labels_4) * 100
        print(f"    Кластер {k_val}: {count} пикселей ({pct:.1f}%) — центроид {centroids_4[k_val]:.1f}")
    print()

    # Сравнение с ground truth
    gt_mask = create_ground_truth_mask(64, 64)
    gt_flat = flatten_image(gt_mask)

    # Конвертируем GT в бинарную маску (объект vs фон)
    gt_binary = create_image(64, 64, 0)
    for r in range(64):
        for c in range(64):
            if get_pixel(gt_mask, r, c) > 0:
                gt_binary[r][c] = 255

    # K=2 для бинарной сегментации
    labels_2, centroids_2 = kmeans_segmentation(img, k=2, max_iter=50)
    pred_binary = create_image(64, 64, 0)
    for r in range(64):
        for c in range(64):
            if get_pixel(labels_2, r, c) == 1:
                pred_binary[r][c] = 255

    iou_k2, inter, union = compute_segmentation_iou(pred_binary, gt_binary)
    acc_k2 = compute_pixel_accuracy(labels_2, gt_mask)
    dice_k2 = dice_coefficient(pred_binary, gt_binary)

    print(f"  Бинарная сегментация (K=2) vs Ground Truth:")
    print(f"    IoU      : {iou_k2:.4f}")
    print(f"    Dice     : {dice_k2:.4f}")
    print(f"    Accuracy : {acc_k2:.4f}")
    print()

    # Визуализация
    vis_3 = labels_to_visualization(labels_3, centroids_3)
    print_image_ascii(img, scale=1, label="Исходное изображение:")
    print()
    print_image_ascii(vis_3, scale=1, label="K-Means (K=3) результат:")
    print()

    # Сравнение разных K
    print(f"  Сравнение разных K:")
    print(f"    {'K':>3}  {'IoU vs GT':>10}  {'Dice':>8}  {'Accuracy':>10}")
    for k_val in [2, 3, 4, 5]:
        lbls, cents = kmeans_segmentation(img, k=k_val, max_iter=50)
        # Конвертируем в бинарную (кластер с наибольшим центроидом = объект)
        sorted_cents = sorted(range(k_val), key=lambda i: cents[i])
        fg_label = sorted_cents[-1]
        pred_b = create_image(64, 64, 0)
        for r in range(64):
            for c in range(64):
                if lbls[r][c] == fg_label:
                    pred_b[r][c] = 255
        iou_val, _, _ = compute_segmentation_iou(pred_b, gt_binary)
        dice_val = dice_coefficient(pred_b, gt_binary)
        acc_val = compute_pixel_accuracy(lbls, gt_mask)
        print(f"    {k_val:>3}  {iou_val:>10.4f}  {dice_val:>8.4f}  {acc_val:>10.4f}")
    print()


# =============================================================================
# Демо 3: Connected Components
# =============================================================================

def demo_connected_components():
    print("=" * 65)
    print("Демо 3: Connected Components (связные компоненты)")
    print("=" * 65)
    print()

    # Бинарная маска с несколькими объектами
    mask = create_image(16, 16, 0)

    # Объект 1: прямоугольник
    for r in range(1, 5):
        for c in range(1, 6):
            mask[r][c] = 255

    # Объект 2: точка
    mask[3][10] = 255

    # Объект 3: L-образная фигура
    for r in range(8, 12):
        mask[r][3] = 255
    for c in range(3, 8):
        mask[11][c] = 255

    # Объект 4: диагональ (связна при connectivity=8, нет при connectivity=4)
    for i in range(4):
        mask[2 + i][11 + i] = 255

    print("  Бинарная маска 16x16 с 4 объектами:")
    print_image_ascii(mask, scale=1)
    print()

    # Connectivity = 4
    num_cc4, labels4 = connected_components(mask, connectivity=4)
    stats4 = component_stats(labels4, num_cc4)
    print(f"  Connectivity = 4 (NSEW):")
    print(f"    Найдено компонент: {num_cc4}")
    for s in stats4:
        print(f"    Компонента {s['label']}: площадь={s['area']}, "
              f"bbox={s['bbox']}, центр=({s['centroid'][0]:.0f}, {s['centroid'][1]:.0f})")
    print()

    # Connectivity = 8
    num_cc8, labels8 = connected_components(mask, connectivity=8)
    stats8 = component_stats(labels8, num_cc8)
    print(f"  Connectivity = 8 (включая диагонали):")
    print(f"    Найдено компонент: {num_cc8}")
    for s in stats8:
        print(f"    Компонента {s['label']}: площадь={s['area']}, "
              f"bbox={s['bbox']}, центр=({s['centroid'][0]:.0f}, {s['centroid'][1]:.0f})")
    print()

    print(f"  Разница: при connectivity=4 найдено {num_cc4} компонент, "
          f"при connectivity=8 — {num_cc8}")
    print(f"  Диагональные пиксели объединяются в одну компоненту при connectivity=8")
    print()

    # Сортировка по площади
    sorted_labels, _, sorted_stats = relabel_components(labels8, num_cc8)
    print(f"  Сортировка по убыванию площади:")
    for s in sorted_stats:
        print(f"    Компонента {s['label']}: площадь={s['area']}")
    print()

    # Применение к сегментации
    print(f"  Применение к результатам K-Means:")
    img = create_synthetic_scene(64, 64)
    gt_mask = create_ground_truth_mask(64, 64)
    gt_binary = create_image(64, 64, 0)
    for r in range(64):
        for c in range(64):
            if get_pixel(gt_mask, r, c) > 0:
                gt_binary[r][c] = 255

    num_objects, obj_labels = connected_components(gt_binary, connectivity=8)
    obj_stats = component_stats(obj_labels, num_objects)
    print(f"    Найдено объектов в GT: {num_objects}")
    for s in obj_stats:
        print(f"    Объект {s['label']}: площадь={s['area']}, "
              f"bbox={s['bbox']}")
    print()


# =============================================================================
# Демо 4: Сравнение методов сегментации
# =============================================================================

def demo_comparison():
    print("=" * 65)
    print("Демо 4: Сравнение методов сегментации")
    print("=" * 65)
    print()

    img = create_synthetic_scene(64, 64)
    noisy_img = create_noisy_image(img, noise_level=20)
    gt_mask = create_ground_truth_mask(64, 64)
    values = flatten_image(img)

    # Ground truth бинарная маска
    gt_binary = create_image(64, 64, 0)
    for r in range(64):
        for c in range(64):
            if get_pixel(gt_mask, r, c) > 0:
                gt_binary[r][c] = 255

    print(f"  Изображение: 64x64 с 3 объектами + шум (noise_level=20)")
    print()

    results = []

    # --- Метод 1: Порог Оцу ---
    thresh, _ = otsu_threshold(values)
    bin_otsu = apply_threshold(img, thresh)
    iou, _, _ = compute_segmentation_iou(bin_otsu, gt_binary)
    acc = compute_pixel_accuracy(bin_otsu, gt_binary)
    dice = dice_coefficient(bin_otsu, gt_binary)
    results.append(("Otsu (чистое)", iou, dice, acc))

    # Оцу на шумном
    noisy_thresh, _ = otsu_threshold(flatten_image(noisy_img))
    bin_otsu_noisy = apply_threshold(noisy_img, noisy_thresh)
    iou_n, _, _ = compute_segmentation_iou(bin_otsu_noisy, gt_binary)
    acc_n = compute_pixel_accuracy(bin_otsu_noisy, gt_binary)
    dice_n = dice_coefficient(bin_otsu_noisy, gt_binary)
    results.append(("Otsu (шум)", iou_n, dice_n, acc_n))

    # --- Метод 2: K-Means ---
    for k_val in [2, 3, 4]:
        labels, centroids = kmeans_segmentation(img, k=k_val, max_iter=50)
        sorted_c = sorted(range(k_val), key=lambda i: centroids[i])
        fg_label = sorted_c[-1]
        pred_b = create_image(64, 64, 0)
        for r in range(64):
            for c in range(64):
                if labels[r][c] == fg_label:
                    pred_b[r][c] = 255
        iou, _, _ = compute_segmentation_iou(pred_b, gt_binary)
        acc = compute_pixel_accuracy(labels, gt_mask)
        dice = dice_coefficient(pred_b, gt_binary)
        results.append((f"K-Means (K={k_val})", iou, dice, acc))

    # --- Метод 3: Connected Components на GT ---
    num_cc, cc_labels = connected_components(gt_binary, connectivity=8)
    cc_binary = create_image(64, 64, 0)
    for r in range(64):
        for c in range(64):
            if get_pixel(cc_labels, r, c) > 0:
                cc_binary[r][c] = 255
    iou, _, _ = compute_segmentation_iou(cc_binary, gt_binary)
    acc = compute_pixel_accuracy(cc_binary, gt_binary)
    dice = dice_coefficient(cc_binary, gt_binary)
    results.append(("CC на GT", iou, dice, acc))

    print(f"  Сравнительная таблица:")
    print(f"  {'Метод':<20} {'IoU':>8} {'Dice':>8} {'Accuracy':>10}")
    print(f"  {'-'*48}")
    for name, iou_val, dice_val, acc_val in results:
        print(f"  {name:<20} {iou_val:>8.4f} {dice_val:>8.4f} {acc_val:>10.4f}")
    print()

    # Детальный анализ IoU по классам
    print(f"  IoU по классам (K-Means K=4 vs GT):")
    labels_4, centroids_4 = kmeans_segmentation(img, k=4, max_iter=50)

    # Маппинг кластеров к классам GT (жадный)
    gt_classes = sorted(set(flatten_image(gt_mask)))
    k_classes = sorted(range(4), key=lambda i: centroids_4[i])

    mapping = {}
    for gt_cls in gt_classes:
        if gt_cls == 0:
            continue
        best_k = 0
        best_iou = -1
        for k_idx in k_classes:
            if k_idx in mapping.values():
                continue
            gt_c = create_image(64, 64, 0)
            pred_c = create_image(64, 64, 0)
            for r in range(64):
                for c in range(64):
                    if get_pixel(gt_mask, r, c) == gt_cls:
                        gt_c[r][c] = 255
                    if get_pixel(labels_4, r, c) == k_idx:
                        pred_c[r][c] = 255
            iou_val, _, _ = compute_segmentation_iou(pred_c, gt_c)
            if iou_val > best_iou:
                best_iou = iou_val
                best_k = k_idx
        mapping[gt_cls] = best_k

    for gt_cls in gt_classes:
        if gt_cls == 0:
            continue
        k_idx = mapping.get(gt_cls, 0)
        gt_c = create_image(64, 64, 0)
        pred_c = create_image(64, 64, 0)
        for r in range(64):
            for c in range(64):
                if get_pixel(gt_mask, r, c) == gt_cls:
                    gt_c[r][c] = 255
                if get_pixel(labels_4, r, c) == k_idx:
                    pred_c[r][c] = 255
        iou_val, _, _ = compute_segmentation_iou(pred_c, gt_c)
        name = {1: "Прямоугольник", 2: "Круг", 3: "Прямоугольник 2"}.get(gt_cls, f"Class {gt_cls}")
        print(f"    Класс {gt_cls} ({name}): IoU = {iou_val:.4f}")
    print()

    # Вывод оценки шума
    print(f"  Влияние шума на сегментацию:")
    iou_clean_otsu = results[0][1]
    iou_noisy_otsu = results[1][1]
    print(f"    Otsu (чистое) : IoU = {iou_clean_otsu:.4f}")
    print(f"    Otsu (шум)    : IoU = {iou_noisy_otsu:.4f}")
    print(f"    Шум снизил IoU на {abs(iou_clean_otsu - iou_noisy_otsu):.4f}")
    print()


# =============================================================================
# Запуск всех демо
# =============================================================================

if __name__ == "__main__":
    demo_thresholding()
    demo_kmeans()
    demo_connected_components()
    demo_comparison()
    print("Все демо завершены.")
