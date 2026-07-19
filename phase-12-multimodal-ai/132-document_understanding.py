"""
132 — Document Understanding: layout analysis, OCR pipeline, table extraction

Темы:
  1. Document Layout Analysis (page segmentation, region detection, reading order)
  2. OCR Pipeline (text detection, recognition, post-processing)
  3. Table Extraction (grid detection, cell identification, structured output)
  4. Document QA (visual question answering over documents)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)

# =============================================================================
# Demo 1: Document Layout Analysis
# =============================================================================
def demo_layout_analysis():
    print("=" * 70)
    print("DEMO 1: Document Layout Analysis — page segmentation, region detection")
    print("=" * 70)

    # --- Sub-example 1: Page segmentation into regions ---
    print("\n[1.1] Page segmentation — dividing a page into semantic regions")
    print("-" * 60)

    def segment_page(page_width, page_height, seed=42):
        """Simulate page segmentation into layout regions."""
        rng = random.Random(seed)
        regions = []
        region_types = ["header", "text", "image", "table", "footer", "sidebar"]
        y_pos = 0
        while y_pos < page_height * 0.9:
            rtype = rng.choice(region_types[:4])  # skip sidebar/footer for simplicity
            h = rng.randint(30, 120)
            if rtype == "text":
                # Text can span full width or partial
                x1 = rng.choice([0, 40, 80])
                x2 = page_width - rng.choice([0, 40, 80])
            elif rtype == "image":
                x1 = rng.randint(20, 100)
                x2 = min(x1 + rng.randint(100, page_width - 100), page_width)
            else:
                x1 = 0
                x2 = page_width
            regions.append({
                "type": rtype,
                "bbox": (x1, y_pos, x2, min(y_pos + h, page_height)),
                "confidence": round(rng.uniform(0.7, 0.99), 3)
            })
            y_pos += h + rng.randint(5, 15)
        return regions

    page_w, page_h = 595, 842  # A4 at 72 DPI
    regions = segment_page(page_w, page_h)
    print(f"Page size: {page_w} x {page_h} pts (A4 at 72 DPI)")
    print(f"Detected {len(regions)} regions:")
    for i, r in enumerate(regions):
        print(f"  Region {i}: {r['type']:8s} bbox={r['bbox']}  conf={r['confidence']:.3f}")

    # --- Sub-example 2: Reading order detection ---
    print("\n[1.2] Reading order — topological sort of layout regions")
    print("-" * 60)

    def detect_reading_order(regions):
        """Sort regions by reading order: top-to-bottom, left-to-right.
        
        Reading order algorithm:
        1. Sort regions by y-center (top to bottom)
        2. For regions at similar y-position, sort by x-center (left to right)
        3. Apply tolerance: regions within 10px vertical overlap are 'same row'
        """
        def sort_key(r):
            bbox = r["bbox"]
            y_center = (bbox[1] + bbox[3]) / 2
            x_center = (bbox[0] + bbox[2]) / 2
            # Quantize y to handle rows (10px tolerance)
            row = y_center // 10
            return (row, x_center)
        
        sorted_regions = sorted(regions, key=sort_key)
        return sorted_regions

    ordered = detect_reading_order(regions)
    print("Reading order (reading direction: top→bottom, left→right):")
    for i, r in enumerate(ordered):
        bbox = r["bbox"]
        print(f"  {i+1}. [{r['type']:8s}] y={bbox[1]:3d}-{bbox[3]:3d}  x={bbox[0]:3d}-{bbox[2]:3d}")

    # --- Sub-example 3: Region overlap detection ---
    print("\n[1.3] Region overlap detection — finding intersecting regions")
    print("-" * 60)

    def bbox_overlap(b1, b2):
        """Compute intersection area of two bounding boxes (x1,y1,x2,y2)."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        if ix1 < ix2 and iy1 < iy2:
            return (ix2 - ix1) * (iy2 - iy1)
        return 0

    def find_overlaps(regions):
        """Find all pairs of overlapping regions."""
        overlaps = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                area = bbox_overlap(regions[i]["bbox"], regions[j]["bbox"])
                if area > 0:
                    b1 = regions[i]["bbox"]
                    b2 = regions[j]["bbox"]
                    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                    iou = area / (a1 + a2 - area)
                    overlaps.append((i, j, area, iou))
        return overlaps

    overlaps = find_overlaps(regions)
    if overlaps:
        print(f"Found {len(overlaps)} overlapping region pairs:")
        for i, j, area, iou in overlaps[:5]:
            print(f"  Regions {i} & {j}: overlap={area}px²  IoU={iou:.4f}")
    else:
        print("No overlapping regions detected (clean segmentation)")

    # --- Sub-example 4: Document structure hierarchy ---
    print("\n[1.4] Document structure — hierarchical region grouping")
    print("-" * 60)

    def build_structure_tree(regions):
        """Build a simple tree structure from nested regions.
        
        Parent-child relationship: region A contains region B if
        A.bbox fully contains B.bbox.
        """
        tree = {"type": "page", "children": []}
        # Sort by area descending for containment check
        indexed = list(enumerate(regions))
        indexed.sort(key=lambda x: (x[1]["bbox"][2]-x[1]["bbox"][0]) * (x[1]["bbox"][3]-x[1]["bbox"][1]), reverse=True)
        
        placed = {}
        for idx, r in indexed:
            node = {"type": r["type"], "id": idx, "children": []}
            placed[idx] = node
        
        for idx, r in indexed:
            bbox = r["bbox"]
            best_parent = None
            for other_idx, other_r in indexed:
                if other_idx == idx:
                    continue
                obbox = other_r["bbox"]
                # Check if other contains this
                if (obbox[0] <= bbox[0] and obbox[1] <= bbox[1] and
                    obbox[2] >= bbox[2] and obbox[3] >= bbox[3]):
                    if best_parent is None or other_idx < best_parent:
                        best_parent = other_idx
            if best_parent is not None:
                placed[best_parent]["children"].append(placed[idx])
            else:
                tree["children"].append(placed[idx])
        return tree

    tree = build_structure_tree(regions)
    def print_tree(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}├── {node['type']} (id={node.get('id', 'root')})")
        for child in node["children"]:
            print_tree(child, indent + 1)
    print("Document structure tree:")
    print_tree(tree)

    print("\n" + "=" * 70)
    print("Layout analysis: segmented page → detected reading order → grouped hierarchy")
    print("=" * 70)


# =============================================================================
# Demo 2: OCR Pipeline
# =============================================================================
def demo_ocr_pipeline():
    print("\n" + "=" * 70)
    print("DEMO 2: OCR Pipeline — text detection, recognition, post-processing")
    print("=" * 70)

    # --- Sub-example 1: Text detection via Connected Component Analysis ---
    print("\n[2.1] Text detection — connected component analysis on binary image")
    print("-" * 60)

    def create_binary_text_image(seed=42):
        """Create a simple binary image with 'text' regions as connected components."""
        rng = random.Random(seed)
        h, w = 50, 120
        image = [[0] * w for _ in range(h)]
        # Simulate text lines as horizontal blobs
        lines = [
            (5, 10, 3),   # line 1: y, height, y_offset
            (20, 12, 5),
            (38, 8, 4),
        ]
        for ly, lh, yoff in lines:
            x = 0
            while x < w - 10:
                word_len = rng.randint(8, 25)
                for dy in range(lh):
                    for dx in range(word_len):
                        if 0 <= ly + yoff + dy < h and 0 <= x + dx < w:
                            # Add some noise
                            if rng.random() > 0.05:
                                image[ly + yoff + dy][x + dx] = 1
                x += word_len + rng.randint(3, 8)
        return image

    def find_connected_components(image):
        """Flood-fill connected component labeling."""
        h = len(image)
        w = len(image[0])
        labels = [[0] * w for _ in range(h)]
        current_label = 0
        
        def flood_fill(r, c, label):
            if r < 0 or r >= h or c < 0 or c >= w:
                return
            if image[r][c] == 0 or labels[r][c] != 0:
                return
            labels[r][c] = label
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                flood_fill(r + dr, c + dc, label)
        
        components = []
        for r in range(h):
            for c in range(w):
                if image[r][c] == 1 and labels[r][c] == 0:
                    current_label += 1
                    flood_fill(r, c, current_label)
                    # Find bounding box
                    min_r, min_c, max_r, max_c = r, c, r, c
                    for rr in range(h):
                        for cc in range(w):
                            if labels[rr][cc] == current_label:
                                min_r, min_c = min(min_r, rr), min(min_c, cc)
                                max_r, max_c = max(max_r, rr), max(max_c, cc)
                    components.append({
                        "label": current_label,
                        "bbox": (min_c, min_r, max_c, max_r),
                        "area": sum(1 for rr in range(h) for cc in range(w) if labels[rr][cc] == current_label)
                    })
        return components

    image = create_binary_text_image()
    components = find_connected_components(image)
    print(f"Binary image: {len(image)}x{len(image[0])} pixels")
    print(f"Found {len(components)} connected components (potential text regions):")
    for comp in components[:6]:
        bbox = comp["bbox"]
        print(f"  Component {comp['label']}: bbox=({bbox[0]:3d},{bbox[1]:3d})-({bbox[2]:3d},{bbox[3]:3d})  area={comp['area']:4d}px")
    print(f"  ... ({len(components)} total)")

    # --- Sub-example 2: Character recognition simulation ---
    print("\n[2.2] Character recognition — template matching simulation")
    print("-" * 60)

    def simulate_ocr_recognition(components, seed=42):
        """Simulate character recognition using template similarity."""
        rng = random.Random(seed)
        # Simplified template features for characters
        char_templates = {
            'A': [0.8, 0.3, 0.9, 0.1], 'B': [0.7, 0.6, 0.8, 0.2],
            'C': [0.6, 0.2, 0.5, 0.1], 'D': [0.9, 0.4, 0.8, 0.3],
            'E': [0.5, 0.5, 0.7, 0.2], 'H': [0.8, 0.8, 0.9, 0.1],
            'I': [0.2, 0.9, 0.3, 0.1], 'L': [0.7, 0.1, 0.6, 0.1],
            'O': [0.8, 0.8, 0.9, 0.9], 'T': [0.9, 0.5, 0.8, 0.1],
        }
        chars = list(char_templates.keys())
        recognized = []
        for comp in components[:8]:
            # Generate feature vector
            feat = [rng.uniform(0, 1) for _ in range(4)]
            # Find best matching template
            best_char = rng.choice(chars)
            best_score = 0
            for ch, tmpl in char_templates.items():
                # Cosine-like similarity (dot product for unit-ish vectors)
                dot = sum(f * t for f, t in zip(feat, tmpl))
                norm1 = math.sqrt(sum(f*f for f in feat)) + 1e-8
                norm2 = math.sqrt(sum(t*t for t in tmpl)) + 1e-8
                sim = dot / (norm1 * norm2)
                if sim > best_score:
                    best_score = sim
                    best_char = ch
            recognized.append((best_char, best_score))
        return recognized

    recognized = simulate_ocr_recognition(components)
    print("Template matching results (cosine similarity):")
    for ch, score in recognized:
        print(f"  Character '{ch}' — similarity={score:.4f}  {'✓ HIGH' if score > 0.7 else '○ LOW'}")

    # --- Sub-example 3: OCR post-processing with language model ---
    print("\n[2.3] Post-processing — spell correction and confidence filtering")
    print("-" * 60)

    def ocr_post_processing(raw_text, confidence_scores, seed=42):
        """Apply post-processing to OCR output.
        
        Steps:
        1. Remove low-confidence characters
        2. Merge split words
        3. Apply dictionary-based correction
        """
        rng = random.Random(seed)
        # Common English words for dictionary check
        dictionary = {"the", "and", "for", "are", "but", "not", "you", "all",
                      "can", "had", "her", "was", "one", "our", "out", "day",
                      "get", "has", "him", "his", "how", "its", "may", "new",
                      "now", "old", "see", "way", "who", "did", "got", "let",
                      "say", "too", "use", "hello", "world", "test"}
        
        # Step 1: Filter by confidence
        filtered = ""
        for ch, conf in zip(raw_text, confidence_scores):
            if conf > 0.5:
                filtered += ch
            else:
                filtered += " "  # Replace low-confidence with space
        
        # Step 2: Merge split words (multiple spaces)
        filtered = re.sub(r'\s+', ' ', filtered).strip()
        
        # Step 3: Simple dictionary correction
        words = filtered.split()
        corrected = []
        corrections = []
        for word in words:
            w_lower = word.lower()
            if w_lower in dictionary:
                corrected.append(word)
            else:
                # Try removing one character (OCR insertion error)
                found = False
                for i in range(len(word)):
                    candidate = word[:i] + word[i+1:]
                    if candidate.lower() in dictionary:
                        corrections.append((word, candidate))
                        corrected.append(candidate)
                        found = True
                        break
                if not found:
                    corrected.append(word)
        
        return " ".join(corrected), corrections

    raw_ocr = "Teehhlloo WWoorlldd"  # Simulated noisy OCR output
    confidences = [random.uniform(0.3, 1.0) for _ in raw_ocr]
    corrected, corrections = ocr_post_processing(raw_ocr, confidences)
    
    print(f"Raw OCR output:      '{raw_ocr}'")
    print(f"Confidence scores:   {[f'{c:.2f}' for c in confidences]}")
    print(f"After correction:    '{corrected}'")
    if corrections:
        print(f"Corrections applied: {corrections}")

    # --- Sub-example 4: OCR confidence estimation ---
    print("\n[2.4] Confidence estimation — character-level uncertainty")
    print("-" * 60)

    def estimate_confidence(char_probs, seed=42):
        """Estimate OCR confidence from character probability distributions.
        
        Confidence = product of max probabilities per character.
        Entropy measures uncertainty: H = -sum(p * log(p)).
        """
        rng = random.Random(seed)
        text = ""
        confidences = []
        entropies = []
        
        for probs in char_probs:
            # probs is a dict of char -> probability
            max_char = max(probs, key=probs.get)
            max_prob = probs[max_char]
            text += max_char
            confidences.append(max_prob)
            
            # Compute entropy
            entropy = 0
            for p in probs.values():
                if p > 0:
                    entropy -= p * math.log2(p)
            entropies.append(entropy)
        
        # Overall confidence
        overall_conf = math.prod(confidences) ** (1 / len(confidences))  # Geometric mean
        avg_entropy = sum(entropies) / len(entropies)
        
        return text, confidences, entropies, overall_conf, avg_entropy

    # Simulated character probability distributions
    char_probs_list = [
        {'H': 0.85, 'B': 0.08, 'R': 0.04, 'K': 0.03},
        {'e': 0.92, 'a': 0.05, 'o': 0.02, 'i': 0.01},
        {'l': 0.78, '1': 0.12, 'i': 0.06, 't': 0.04},
        {'l': 0.88, '1': 0.07, 'i': 0.03, 't': 0.02},
        {'o': 0.91, '0': 0.06, 'a': 0.02, 'e': 0.01},
    ]
    
    text, confs, ents, overall, avg_ent = estimate_confidence(char_probs_list)
    print(f"Recognized: '{text}'")
    print(f"Per-char confidence: {[f'{c:.3f}' for c in confs]}")
    print(f"Per-char entropy:    {[f'{e:.3f}' for e in ents]} bits")
    print(f"Overall confidence:  {overall:.4f} (geometric mean)")
    print(f"Avg entropy:         {avg_ent:.4f} bits")
    print(f"Interpretation: {'HIGH confidence' if overall > 0.8 else 'MEDIUM confidence' if overall > 0.6 else 'LOW confidence'}")

    print("\n" + "=" * 70)
    print("OCR pipeline: detection → recognition → post-processing → confidence")
    print("=" * 70)


# =============================================================================
# Demo 3: Table Extraction
# =============================================================================
def demo_table_extraction():
    print("\n" + "=" * 70)
    print("DEMO 3: Table Extraction — grid detection, cell identification, structured output")
    print("=" * 70)

    # --- Sub-example 1: Table structure detection via line analysis ---
    print("\n[3.1] Table structure detection — horizontal and vertical line analysis")
    print("-" * 60)

    def generate_table_lines(seed=42):
        """Generate simulated table lines for structure detection."""
        rng = random.Random(seed)
        rows, cols = 4, 3
        x_start, y_start = 50, 30
        cell_w, cell_h = 100, 25
        
        h_lines = []  # Horizontal lines
        v_lines = []  # Vertical lines
        
        for r in range(rows + 1):
            y = y_start + r * cell_h
            noise = [rng.uniform(-1, 1) for _ in range(cols + 1)]
            h_lines.append([(x_start + c * cell_w + noise[c], y + noise[c]) for c in range(cols + 1)])
        
        for c in range(cols + 1):
            x = x_start + c * cell_w
            noise = [rng.uniform(-1, 1) for _ in range(rows + 1)]
            v_lines.append([(x + noise[r], y_start + r * cell_h + noise[r]) for r in range(rows + 1)])
        
        return h_lines, v_lines, rows, cols

    h_lines, v_lines, nrows, ncols = generate_table_lines()
    print(f"Table grid: {nrows} rows x {ncols} cols")
    print(f"Horizontal lines detected: {len(h_lines)}")
    print(f"Vertical lines detected:   {len(v_lines)}")
    
    # Find intersections
    intersections = []
    for hi, hline in enumerate(h_lines):
        for vi, vline in enumerate(v_lines):
            # Approximate intersection
            hx = hline[min(vi, len(hline)-1)][0]
            vy = vline[min(hi, len(vline)-1)][1]
            intersections.append((hx, vy))
    
    print(f"Grid intersections: {len(intersections)} ({len(h_lines)} x {len(v_lines)})")
    print(f"Grid structure: {nrows}x{ncols} = {nrows*ncols} cells")

    # --- Sub-example 2: Cell identification and content extraction ---
    print("\n[3.2] Cell identification — extracting structured content from grid")
    print("-" * 60)

    def extract_table_cells(h_lines, v_lines, content_seed=42):
        """Extract cell contents from table grid."""
        rng = random.Random(content_seed)
        sample_data = [
            ["Name", "Score", "Grade"],
            ["Alice", "92", "A"],
            ["Bob", "78", "B+"],
            ["Carol", "85", "A-"],
        ]
        
        cells = []
        for r in range(len(sample_data)):
            for c in range(len(sample_data[r])):
                # Approximate cell bbox
                x1 = v_lines[c][0][0] if c < len(v_lines) else v_lines[-1][0][0]
                x2 = v_lines[c+1][0][0] if c+1 < len(v_lines) else v_lines[-1][0][0] + 100
                y1 = h_lines[r][0][1] if r < len(h_lines) else h_lines[-1][0][1]
                y2 = h_lines[r+1][0][1] if r+1 < len(h_lines) else h_lines[-1][0][1] + 25
                
                cells.append({
                    "row": r,
                    "col": c,
                    "content": sample_data[r][c],
                    "bbox": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                    "is_header": r == 0
                })
        return cells, sample_data

    cells, raw_data = extract_table_cells(h_lines, v_lines)
    print("Extracted table cells:")
    for cell in cells:
        marker = "[H]" if cell["is_header"] else "   "
        print(f"  {marker} Row {cell['row']}, Col {cell['col']}: '{cell['content']}'  bbox={cell['bbox']}")

    # --- Sub-example 3: Table to structured output ---
    print("\n[3.3] Structured output — converting cells to JSON/CSV")
    print("-" * 60)

    def cells_to_structured(cells, format_type="json"):
        """Convert extracted cells to structured format."""
        # Group by rows
        rows = collections.defaultdict(list)
        for cell in cells:
            rows[cell["row"]].append(cell)
        
        # Sort rows and build table
        table = []
        for r in sorted(rows.keys()):
            row_cells = sorted(rows[r], key=lambda c: c["col"])
            table.append([c["content"] for c in row_cells])
        
        if format_type == "json":
            if len(table) > 1:
                headers = table[0]
                data = []
                for row in table[1:]:
                    data.append(dict(zip(headers, row)))
                return json.dumps({"headers": headers, "data": data, "row_count": len(data)}, indent=2)
            return json.dumps({"table": table}, indent=2)
        elif format_type == "csv":
            return "\n".join([",".join(row) for row in table])
        return table

    json_output = cells_to_structured(cells, "json")
    print("JSON output:")
    print(json_output)

    csv_output = cells_to_structured(cells, "csv")
    print("\nCSV output:")
    print(csv_output)

    # --- Sub-example 4: Table quality assessment ---
    print("\n[3.4] Table quality — assessing extraction accuracy")
    print("-" * 60)

    def assess_table_quality(cells, ground_truth=None, seed=42):
        """Assess table extraction quality metrics.
        
        Metrics:
        - Cell accuracy: % of correctly extracted cells
        - Structure accuracy: % of correctly identified rows/cols
        - Content accuracy: Levenshtein similarity of content
        """
        rng = random.Random(seed)
        
        # Simulate ground truth
        gt_data = [
            ["Name", "Score", "Grade"],
            ["Alice", "92", "A"],
            ["Bob", "78", "B+"],
            ["Carol", "85", "A-"],
        ]
        
        # Extracted data
        ext_data = []
        rows = collections.defaultdict(list)
        for cell in cells:
            rows[cell["row"]].append(cell)
        for r in sorted(rows.keys()):
            row_cells = sorted(rows[r], key=lambda c: c["col"])
            ext_data.append([c["content"] for c in row_cells])
        
        # Compute metrics
        gt_flat = [item for row in gt_data for item in row]
        ext_flat = [item for row in ext_data for item in row]
        
        # Cell-level accuracy
        cell_matches = sum(1 for g, e in zip(gt_flat, ext_flat) if g == e)
        cell_accuracy = cell_matches / max(len(gt_flat), 1)
        
        # Row/Col structure
        structure_match = (len(gt_data) == len(ext_data) and 
                          all(len(g) == len(e) for g, e in zip(gt_data, ext_data)))
        
        # Content similarity (Levenshtein-like)
        def simple_similarity(s1, s2):
            """Simple character-level similarity."""
            if not s1 or not s2:
                return 0.0
            matches = sum(1 for a, b in zip(s1, s2) if a == b)
            return matches / max(len(s1), len(s2))
        
        content_sims = [simple_similarity(g, e) for g, e in zip(gt_flat, ext_flat)]
        avg_content_sim = sum(content_sims) / len(content_sims) if content_sims else 0
        
        return {
            "cell_accuracy": cell_accuracy,
            "structure_correct": structure_match,
            "avg_content_similarity": avg_content_sim,
            "total_cells_gt": len(gt_flat),
            "total_cells_ext": len(ext_flat),
            "cells_matched": cell_matches
        }

    metrics = assess_table_quality(cells)
    print("Table extraction quality metrics:")
    print(f"  Cell accuracy:        {metrics['cell_accuracy']:.1%} ({metrics['cells_matched']}/{metrics['total_cells_gt']})")
    print(f"  Structure correct:    {metrics['structure_correct']}")
    print(f"  Content similarity:   {metrics['avg_content_similarity']:.3f}")
    
    # Overall score
    score = (metrics['cell_accuracy'] * 0.4 + 
             (1.0 if metrics['structure_correct'] else 0.0) * 0.3 + 
             metrics['avg_content_similarity'] * 0.3)
    print(f"  Overall score:        {score:.3f} ({'EXCELLENT' if score > 0.9 else 'GOOD' if score > 0.7 else 'NEEDS REVIEW'})")

    print("\n" + "=" * 70)
    print("Table extraction: lines → grid → cells → structured output → quality")
    print("=" * 70)


# =============================================================================
# Demo 4: Document QA
# =============================================================================
def demo_document_qa():
    print("\n" + "=" * 70)
    print("DEMO 4: Document QA — visual question answering over documents")
    print("=" * 70)

    # --- Sub-example 1: Document representation for QA ---
    print("\n[4.1] Document representation — encoding regions for QA")
    print("-" * 60)

    def create_document_representation(seed=42):
        """Create a structured document representation for QA processing."""
        rng = random.Random(seed)
        
        document = {
            "title": "Quarterly Financial Report Q3 2024",
            "sections": [
                {
                    "id": "s1",
                    "type": "header",
                    "content": "Executive Summary",
                    "bbox": (50, 50, 545, 80),
                    "embedding_hash": hashlib.md5(b"executive_summary").hexdigest()[:8]
                },
                {
                    "id": "s2",
                    "type": "text",
                    "content": "Revenue increased by 15% compared to Q2 2024, reaching $4.2M. Net profit margin improved to 12.5%.",
                    "bbox": (50, 85, 545, 150),
                    "embedding_hash": hashlib.md5(b"revenue_text").hexdigest()[:8]
                },
                {
                    "id": "s3",
                    "type": "table",
                    "content": {"headers": ["Metric", "Q2", "Q3", "Change"], 
                               "data": [{"Metric": "Revenue", "Q2": "$3.65M", "Q3": "$4.2M", "Change": "+15%"},
                                       {"Metric": "Costs", "Q2": "$2.9M", "Q3": "$3.1M", "Change": "+6.9%"},
                                       {"Metric": "Profit", "Q2": "$0.75M", "Q3": "$0.525M", "Change": "-30%"}]},
                    "bbox": (50, 155, 545, 280),
                    "embedding_hash": hashlib.md5(b"financial_table").hexdigest()[:8]
                },
                {
                    "id": "s4",
                    "type": "text",
                    "content": "Operating expenses were primarily driven by R&D investments and marketing expansion.",
                    "bbox": (50, 285, 545, 340),
                    "embedding_hash": hashlib.md5(b"expenses_text").hexdigest()[:8]
                },
                {
                    "id": "s5",
                    "type": "image",
                    "content": "Revenue growth chart showing upward trend from Jan to Sep 2024",
                    "bbox": (50, 345, 350, 450),
                    "embedding_hash": hashlib.md5(b"revenue_chart").hexdigest()[:8]
                },
            ]
        }
        return document

    doc = create_document_representation()
    print(f"Document: '{doc['title']}'")
    print(f"Total sections: {len(doc['sections'])}")
    for sec in doc["sections"]:
        content_preview = sec["content"][:50] + "..." if isinstance(sec["content"], str) else str(sec["content"])[:50]
        print(f"  [{sec['type']:6s}] id={sec['id']} bbox={sec['bbox']} content='{content_preview}'")

    # --- Sub-example 2: Question understanding ---
    print("\n[4.2] Question understanding — parsing user questions about documents")
    print("-" * 60)

    def parse_question(question):
        """Parse and classify a question about a document.
        
        Question types:
        - FACTOID: asks for specific value (what, when, who)
        - COMPARISON: asks to compare (compare, difference, more/less)
        - AGGREGATION: asks for summary (total, average, summarize)
        - VISUAL: asks about layout/structure (where, how many, which section)
        """
        q = question.lower().strip()
        
        # Determine question type
        if any(w in q for w in ["compare", "difference", "more", "less", "higher", "lower"]):
            qtype = "COMPARISON"
        elif any(w in q for w in ["total", "average", "summarize", "overall", "summary"]):
            qtype = "AGGREGATION"
        elif any(w in q for w in ["where", "how many", "which section", "position", "locate"]):
            qtype = "VISUAL"
        else:
            qtype = "FACTOID"
        
        # Extract key entities
        entities = []
        if "revenue" in q:
            entities.append("revenue")
        if "profit" in q:
            entities.append("profit")
        if "cost" in q or "expense" in q:
            entities.append("costs")
        if "q2" in q or "second quarter" in q:
            entities.append("Q2")
        if "q3" in q or "third quarter" in q:
            entities.append("Q3")
        if "chart" in q or "graph" in q or "image" in q:
            entities.append("visual")
        if "table" in q:
            entities.append("table")
        
        # Determine target section type
        section_type = "any"
        if any(w in q for w in ["table", "data", "numbers", "metrics"]):
            section_type = "table"
        elif any(w in q for w in ["chart", "graph", "image", "figure"]):
            section_type = "image"
        elif any(w in q for w in ["title", "header", "section"]):
            section_type = "header"
        
        return {
            "question": question,
            "type": qtype,
            "entities": entities,
            "target_section": section_type
        }

    questions = [
        "What was the revenue in Q3?",
        "Compare Q2 and Q3 profits",
        "Where is the revenue chart located?",
        "Summarize the document",
    ]
    
    for q in questions:
        parsed = parse_question(q)
        print(f"Q: '{q}'")
        print(f"   Type: {parsed['type']}, Entities: {parsed['entities']}, Target: {parsed['target_section']}")

    # --- Sub-example 3: Answer extraction ---
    print("\n[4.3] Answer extraction — finding answers in document content")
    print("-" * 60)

    def extract_answer(doc, parsed_question):
        """Extract answer from document based on parsed question."""
        qtype = parsed_question["type"]
        entities = parsed_question["entities"]
        target_type = parsed_question["target_section"]
        
        # Find relevant sections
        relevant = []
        for sec in doc["sections"]:
            score = 0
            content_str = str(sec["content"]).lower()
            for ent in entities:
                if ent.lower() in content_str:
                    score += 2
            if target_type == sec["type"] or target_type == "any":
                score += 1
            if score > 0:
                relevant.append((score, sec))
        
        relevant.sort(key=lambda x: x[0], reverse=True)
        
        if not relevant:
            return {"answer": "Information not found in document", "confidence": 0.0, "source": None}
        
        best_score, best_sec = relevant[0]
        
        # Extract answer based on question type
        if qtype == "FACTOID":
            if isinstance(best_sec["content"], dict) and "data" in best_sec["content"]:
                # Table data
                for row in best_sec["content"]["data"]:
                    for ent in entities:
                        if ent.lower() in str(row).lower():
                            answer = str(row)
                            return {"answer": answer, "confidence": 0.9, "source": best_sec["id"]}
            answer = best_sec["content"][:100] if isinstance(best_sec["content"], str) else str(best_sec["content"])[:100]
            return {"answer": answer, "confidence": min(best_score / 5, 1.0), "source": best_sec["id"]}
        
        elif qtype == "COMPARISON":
            if len(relevant) >= 2:
                s1 = relevant[0][1]
                s2 = relevant[1][1]
                return {
                    "answer": f"Comparing sections: {s1['id']} vs {s2['id']}",
                    "confidence": 0.85,
                    "source": f"{s1['id']},{s2['id']}"
                }
            return {"answer": "Insufficient data for comparison", "confidence": 0.3, "source": best_sec["id"]}
        
        elif qtype == "VISUAL":
            return {
                "answer": f"Located at position {best_sec['bbox']}",
                "confidence": 0.95,
                "source": best_sec["id"]
            }
        
        else:  # AGGREGATION
            return {
                "answer": f"Document contains {len(doc['sections'])} sections",
                "confidence": 0.7,
                "source": "full_document"
            }

    test_questions = [
        ("What was the revenue in Q3?", "FACTOID"),
        ("Compare Q2 and Q3 profits", "COMPARISON"),
        ("Where is the revenue chart?", "VISUAL"),
    ]
    
    for q_text, expected_type in test_questions:
        parsed = parse_question(q_text)
        answer = extract_answer(doc, parsed)
        print(f"Q: '{q_text}'")
        print(f"   Answer: {answer['answer']}")
        print(f"   Confidence: {answer['confidence']:.2f} | Source: {answer['source']}")

    # --- Sub-example 4: Multi-modal QA fusion ---
    print("\n[4.4] Multi-modal QA — fusing text, table, and visual answers")
    print("-" * 60)

    def multimodal_qa_fusion(doc, question, seed=42):
        """Fuse answers from multiple modalities (text, table, image).
        
        Fusion strategy:
        1. Extract answers from each modality
        2. Compute confidence scores
        3. Weighted fusion based on relevance
        """
        rng = random.Random(seed)
        parsed = parse_question(question)
        
        # Collect answers from different modalities
        modality_answers = []
        
        for sec in doc["sections"]:
            content_str = str(sec["content"]).lower()
            question_entities = [e.lower() for e in parsed["entities"]]
            relevance = sum(1 for e in question_entities if e in content_str)
            
            if relevance > 0:
                # Simulate modality-specific confidence
                if sec["type"] == "text":
                    confidence = 0.7 + rng.uniform(0, 0.2)
                elif sec["type"] == "table":
                    confidence = 0.8 + rng.uniform(0, 0.15)  # Tables often more precise
                elif sec["type"] == "image":
                    confidence = 0.6 + rng.uniform(0, 0.25)  # Images may be less precise for factual QA
                else:
                    confidence = 0.5 + rng.uniform(0, 0.3)
                
                modality_answers.append({
                    "modality": sec["type"],
                    "section_id": sec["id"],
                    "relevance": relevance,
                    "confidence": confidence,
                    "content": sec["content"]
                })
        
        # Weighted fusion
        if not modality_answers:
            return {"fused_answer": "No relevant information found", "final_confidence": 0.0,
                    "modalities_used": [], "num_sources": 0}
        
        total_weight = sum(a["confidence"] * a["relevance"] for a in modality_answers)
        weighted_content = ""
        for a in sorted(modality_answers, key=lambda x: x["confidence"] * x["relevance"], reverse=True):
            content_str = str(a["content"])[:60]
            weighted_content += f"[{a['modality']}] {content_str}...\n"
        
        final_confidence = total_weight / (sum(a["relevance"] for a in modality_answers) + 1e-8)
        
        return {
            "fused_answer": weighted_content.strip(),
            "final_confidence": min(final_confidence, 1.0),
            "modalities_used": [a["modality"] for a in modality_answers],
            "num_sources": len(modality_answers)
        }

    # Test multimodal fusion
    fusion_questions = [
        "What is the revenue information?",
        "Show me the financial data",
    ]
    
    for q in fusion_questions:
        result = multimodal_qa_fusion(doc, q)
        print(f"Q: '{q}'")
        print(f"   Modalities used: {result['modalities_used']}")
        print(f"   Sources: {result['num_sources']}")
        print(f"   Final confidence: {result['final_confidence']:.3f}")
        print(f"   Answer preview: {result['fused_answer'][:80]}...")

    print("\n" + "=" * 70)
    print("Document QA: parse question → extract from sections → fuse modalities → answer")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    demo_layout_analysis()
    demo_ocr_pipeline()
    demo_table_extraction()
    demo_document_qa()
