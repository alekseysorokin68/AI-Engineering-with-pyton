"""
Phase 04 — Vision: Основы детекции объектов
==========================================
Демонстрация ключевых концепций object detection без внешних зависимостей.

Реализовано:
  1. IoU (Intersection over Union)
  2. Non-Maximum Suppression (NMS)
  3. Anchor boxes
  4. Метрика mAP (mean Average Precision)
"""

import random

random.seed(42)


# =============================================================================
# 1. IoU — Intersection over Union
# =============================================================================

def compute_iou(box_a, box_b):
    """
    Вычисляет IoU между двумя прямоугольниками.

    Формат бокса: [x1, y1, x2, y2] — (левый верхний, правый нижний)

    IoU = площадь пересечения / площадь объединения
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# =============================================================================
# 2. Non-Maximum Suppression (NMS)
# =============================================================================

def nms(boxes, scores, iou_threshold=0.5):
    """
    Подавление немаксимумов — убирает дублирующиеся боксы.

    Алгоритм:
      1. Сортируем боксы по убыванию confidence.
      2. Берём бокс с максимальным score, добавляем в результат.
      3. Убираем все боксы, у которых IoU с выбранным >= порога.
      4. Повторяем, пока боксы не закончатся.

    Вход:
      boxes          — список боксов [[x1,y1,x2,y2], ...]
      scores         — список уверенностей [float, ...]
      iou_threshold  — порог IoU для подавления

    Выход:
      list of indices — индексы оставшихся боксов
    """
    if not boxes:
        return []

    indices = list(range(len(scores)))
    indices.sort(key=lambda i: scores[i], reverse=True)

    keep = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        remaining = []
        for idx in indices:
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold:
                remaining.append(idx)
        indices = remaining

    return keep


# =============================================================================
# 3. Anchor boxes
# =============================================================================

def generate_anchors(feature_map_h, feature_map_w, stride, scales, ratios):
    """
    Генерирует anchor boxes для одного уровня feature map.

    Для каждой ячейки (i, j) feature map создаётся несколько anchors
    разных масштабов и соотношений сторон.

    Вход:
      feature_map_h, feature_map_w — размеры feature map
      stride   — коэффициент уменьшения (pixel stride)
      scales   — список масштабов (относительно базового размера)
      ratios   — список соотношений сторон (width / height)

    Выход:
      list of [x1, y1, x2, y2] — все anchor boxes
    """
    anchors = []
    base_size = stride

    for i in range(feature_map_h):
        for j in range(feature_map_w):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for scale in scales:
                for ratio in ratios:
                    w = base_size * scale * (ratio ** 0.5)
                    h = base_size * scale * (1.0 / ratio ** 0.5)
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    anchors.append([x1, y1, x2, y2])

    return anchors


# =============================================================================
# 4. Метрика mAP (mean Average Precision)
# =============================================================================

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Вычисляет Average Precision (AP) для одного класса.

    predictions   — список (confidence, box), отсортированный по confidence ↓
    ground_truths — список box для одного класса
    iou_threshold — порог IoU для определения TP/FP

    Возвращает:
      ap        — average precision
      precisions — массив precision для кривой PR
      recalls    — массив recall для кривой PR
    """
    tp = [0] * len(predictions)
    fp = [0] * len(predictions)
    matched = set()

    for i, (conf, pred_box) in enumerate(predictions):
        best_iou = 0.0
        best_gt = -1
        for j, gt_box in enumerate(ground_truths):
            if j in matched:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            matched.add(best_gt)
        else:
            fp[i] = 1

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    total_gt = len(ground_truths) if ground_truths else 1

    for i in range(len(predictions)):
        tp_cumsum += tp[i]
        fp_cumsum += fp[i]
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # Вычисляем AP как площадь под PR-кривой (метод 11 точек)
    ap = 0.0
    for t in range(11):
        t_recall = t / 10.0
        max_precision = 0.0
        for i in range(len(recalls)):
            if recalls[i] >= t_recall:
                max_precision = max(max_precision, precisions[i])
        ap += max_precision / 11.0

    return ap, precisions, recalls


def compute_map(detections, ground_truths_per_class, iou_threshold=0.5):
    """
    Вычисляет mAP (mean Average Precision) по всем классам.

    detections — dict: {class_id: [(confidence, box), ...]}
    ground_truths_per_class — dict: {class_id: [box, ...]}
    """
    aps = {}
    for cls_id in sorted(set(list(detections.keys()) + list(ground_truths_per_class.keys()))):
        preds = sorted(detections.get(cls_id, []), key=lambda x: x[0], reverse=True)
        gts = ground_truths_per_class.get(cls_id, [])
        if not gts:
            continue
        ap, _, _ = compute_ap(preds, gts, iou_threshold)
        aps[cls_id] = ap

    map_value = sum(aps.values()) / len(aps) if aps else 0.0
    return map_value, aps


# =============================================================================
# Демо 1: Вычисление IoU
# =============================================================================

def demo_iou():
    print("=" * 65)
    print("Демо 1: Вычисление IoU (Intersection over Union)")
    print("=" * 65)

    box_a = [50, 50, 150, 150]
    box_b = [100, 100, 200, 200]
    iou = compute_iou(box_a, box_b)
    print(f"  Бокс A: {box_a}")
    print(f"  Бокс B: {box_b}")
    print(f"  IoU    : {iou:.4f}")
    print()

    box_c = [0, 0, 100, 100]
    box_d = [200, 200, 300, 300]
    iou2 = compute_iou(box_c, box_d)
    print(f"  Бокс C: {box_c}")
    print(f"  Бокс D: {box_d}")
    print(f"  IoU    : {iou2:.4f} (нет пересечения)")
    print()

    box_e = [10, 10, 90, 90]
    box_f = [10, 10, 90, 90]
    iou3 = compute_iou(box_e, box_f)
    print(f"  Бокс E: {box_e}")
    print(f"  Бокс F: {box_f}")
    print(f"  IoU    : {iou3:.4f} (полное совпадение)")
    print()


# =============================================================================
# Демо 2: Non-Maximum Suppression (NMS)
# =============================================================================

def demo_nms():
    print("=" * 65)
    print("Демо 2: Non-Maximum Suppression (NMS)")
    print("=" * 65)

    boxes = [
        [100, 100, 210, 210],
        [105, 105, 215, 215],
        [200, 200, 320, 320],
        [210, 205, 330, 325],
        [50, 50, 100, 100],
    ]
    scores = [0.95, 0.80, 0.70, 0.85, 0.60]

    print("  Входные боксы (x1, y1, x2, y2) + confidence:")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"    [{i}] {b}  conf={s:.2f}")
    print()

    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"  NMS (iou_threshold=0.5) — оставлены индексы: {keep}")
    for idx in keep:
        print(f"    [{idx}] {boxes[idx]}  conf={scores[idx]:.2f}")
    print()

    keep_strict = nms(boxes, scores, iou_threshold=0.3)
    print(f"  NMS (iou_threshold=0.3) — оставлены индексы: {keep_strict}")
    for idx in keep_strict:
        print(f"    [{idx}] {boxes[idx]}  conf={scores[idx]:.2f}")
    print()


# =============================================================================
# Демо 3: Anchor boxes
# =============================================================================

def demo_anchors():
    print("=" * 65)
    print("Демо 3: Anchor boxes")
    print("=" * 65)

    feature_map_h, feature_map_w = 4, 4
    stride = 32
    scales = [0.5, 1.0, 2.0]
    ratios = [0.5, 1.0, 2.0]

    anchors = generate_anchors(feature_map_h, feature_map_w, stride, scales, ratios)

    total = feature_map_h * feature_map_w * len(scales) * len(ratios)
    print(f"  Feature map  : {feature_map_h}x{feature_map_w}")
    print(f"  Stride       : {stride}")
    print(f"  Scales       : {scales}")
    print(f"  Ratios       : {ratios}")
    print(f"  Всего anchors: {total}")
    print()

    print("  Примеры anchors для ячейки (0, 0):")
    for i in range(len(scales) * len(ratios)):
        a = anchors[i]
        w = a[2] - a[0]
        h = a[3] - a[1]
        s = scales[i // len(ratios)]
        r = ratios[i % len(ratios)]
        print(f"    scale={s}, ratio={r}  →  [{a[0]:.1f}, {a[1]:.1f}, {a[2]:.1f}, {a[3]:.1f}]  ({w:.1f}x{h:.1f})")
    print()

    print(f"  Пример anchors для ячейки (2, 3):")
    offset = (2 * feature_map_w + 3) * len(scales) * len(ratios)
    for i in range(len(scales) * len(ratios)):
        a = anchors[offset + i]
        w = a[2] - a[0]
        h = a[3] - a[1]
        s = scales[i // len(ratios)]
        r = ratios[i % len(ratios)]
        print(f"    scale={s}, ratio={r}  →  [{a[0]:.1f}, {a[1]:.1f}, {a[2]:.1f}, {a[3]:.1f}]  ({w:.1f}x{h:.1f})")
    print()


# =============================================================================
# Демо 4: Метрика mAP
# =============================================================================

def demo_map():
    print("=" * 65)
    print("Демо 4: Метрика mAP (mean Average Precision)")
    print("=" * 65)

    # Класс 0 (кошки)
    gt_cats = [
        [50, 50, 150, 150],
        [200, 100, 350, 300],
    ]
    det_cats = [
        (0.95, [55, 55, 145, 145]),
        (0.85, [210, 110, 340, 290]),
        (0.70, [300, 300, 400, 400]),
        (0.60, [60, 60, 140, 140]),
    ]

    # Класс 1 (собаки)
    gt_dogs = [
        [100, 200, 250, 350],
    ]
    det_dogs = [
        (0.90, [105, 205, 245, 345]),
        (0.50, [400, 400, 500, 500]),
    ]

    detections = {0: det_cats, 1: det_dogs}
    gts = {0: gt_cats, 1: gt_dogs}

    print("  Класс 0 (кошки):")
    print(f"    Ground truth: {len(gt_cats)} объектов")
    print(f"    Детекции   : {len(det_cats)}")
    print()

    ap_cats, _, _ = compute_ap(sorted(det_cats, key=lambda x: x[0], reverse=True), gt_cats)
    print(f"    AP (кошки)   = {ap_cats:.4f}")
    print()

    print("  Класс 1 (собаки):")
    print(f"    Ground truth: {len(gt_dogs)} объектов")
    print(f"    Детекции   : {len(det_dogs)}")
    print()

    ap_dogs, _, _ = compute_ap(sorted(det_dogs, key=lambda x: x[0], reverse=True), gt_dogs)
    print(f"    AP (собаки)  = {ap_dogs:.4f}")
    print()

    map_val, aps = compute_map(detections, gts)
    print(f"  AP по классам: {aps}")
    print(f"  mAP@0.5       = {map_val:.4f}")
    print()

    # Демонстрация precision-recall кривой для класса «кошки»
    print("  Precision-Recall кривая (класс 0 — кошки):")
    preds_sorted = sorted(det_cats, key=lambda x: x[0], reverse=True)
    ap, precisions, recalls = compute_ap(preds_sorted, gt_cats)
    print(f"    {'Det':>3}  {'Conf':>5}  {'TP/FP':>5}  {'Prec':>6}  {'Recall':>6}")
    tp_cum = 0
    fp_cum = 0
    matched = set()
    for i, (conf, pred_box) in enumerate(preds_sorted):
        best_iou = 0.0
        best_gt = -1
        for j, gt_box in enumerate(gt_cats):
            if j in matched:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        is_tp = best_iou >= 0.5 and best_gt >= 0
        if is_tp:
            tp_cum += 1
            matched.add(best_gt)
            tag = "TP"
        else:
            fp_cum += 1
            tag = "FP"
        prec = tp_cum / (tp_cum + fp_cum)
        rec = tp_cum / len(gt_cats)
        print(f"    {i:>3}  {conf:>5.2f}  {tag:>5}  {prec:>6.3f}  {rec:>6.3f}")
    print()


# =============================================================================
# Запуск всех демо
# =============================================================================

if __name__ == "__main__":
    demo_iou()
    demo_nms()
    demo_anchors()
    demo_map()
    print("Все демо завершены.")
