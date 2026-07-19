"""
135 — Multimodal Agents: vision-based tool use, GUI automation, embodied AI

Темы:
  1. Vision-Based Tool Use (screenshot understanding, element grounding)
  2. GUI Automation (click coordinates, form filling, navigation)
  3. Embodied AI (robot perception, action planning, sim-to-real)
  4. Multimodal Planning (visual reasoning, spatial understanding)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)

def demo_vision_tool_use():
    """
    Section 1: Vision-Based Tool Use
    Screenshot understanding and element grounding
    """
    print("=" * 70)
    print("DEMO 1: Vision-Based Tool Use")
    print("=" * 70)
    
    # 1.1 Screenshot pixel analysis
    print("\n[1.1] Screenshot Pixel Analysis")
    print("-" * 40)
    
    # Simulated 8x8 grayscale screenshot (values 0-255)
    random.seed(42)
    screenshot = []
    for row in range(8):
        pixel_row = []
        for col in range(8):
            if row < 3 and col < 4:  # Button region
                pixel_row.append(random.randint(80, 120))
            elif row >= 5 and 1 <= col <= 6:  # Text region
                pixel_row.append(random.randint(200, 240))
            else:
                pixel_row.append(random.randint(240, 255))  # Background
        screenshot.append(pixel_row)
    
    # Compute mean intensity per region
    button_pixels = [screenshot[r][c] for r in range(3) for c in range(4)]
    text_pixels = [screenshot[r][c] for r in range(5, 8) for c in range(1, 7)]
    bg_pixels = [screenshot[r][c] for r in range(8) for c in range(8)]
    
    button_mean = sum(button_pixels) / len(button_pixels)
    text_mean = sum(text_pixels) / len(text_pixels)
    
    print(f"Screenshot: 8x8 pixels simulated")
    print(f"Button region mean intensity: {button_mean:.1f}")
    print(f"Text region mean intensity: {text_mean:.1f}")
    print(f"Contrast ratio: {text_mean / button_mean:.2f}x")
    print("Formula: Contrast = text_mean / button_mean")
    
    # 1.2 Element detection via thresholding
    print("\n[1.2] Element Detection via Thresholding")
    print("-" * 40)
    
    threshold = 128
    detected_elements = []
    
    for row in range(8):
        for col in range(8):
            if screenshot[row][col] < threshold:
                detected_elements.append({
                    'x': col, 'y': row,
                    'intensity': screenshot[row][col],
                    'type': 'interactive' if screenshot[row][col] < 100 else 'border'
                })
    
    print(f"Threshold: {threshold}")
    print(f"Detected {len(detected_elements)} elements below threshold")
    for elem in detected_elements[:4]:
        print(f"  Element at ({elem['x']},{elem['y']}): intensity={elem['intensity']}, type={elem['type']}")
    print("Formula: element_detected = pixel_value < threshold")
    
    # 1.3 Color histogram for UI classification
    print("\n[1.3] Color Histogram for UI Classification")
    print("-" * 40)
    
    # Simulate RGB screenshot histogram
    random.seed(42)
    color_bins = {'dark': 0, 'medium': 0, 'bright': 0}
    all_pixels = []
    
    for _ in range(64):
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        all_pixels.append(luminance)
        
        if luminance < 85:
            color_bins['dark'] += 1
        elif luminance < 170:
            color_bins['medium'] += 1
        else:
            color_bins['bright'] += 1
    
    print("Luminance formula: L = 0.299*R + 0.587*G + 0.114*B")
    print("Color distribution in screenshot:")
    for cat, count in color_bins.items():
        pct = count / 64 * 100
        print(f"  {cat}: {count} pixels ({pct:.1f}%)")
    
    ui_type = 'dark_theme' if color_bins['dark'] > color_bins['bright'] else 'light_theme'
    print(f"Classified UI theme: {ui_type}")
    
    # 1.4 Bounding box prediction
    print("\n[1.4] Bounding Box Prediction")
    print("-" * 40)
    
    # Find bounding box of interactive elements
    if detected_elements:
        min_x = min(e['x'] for e in detected_elements)
        max_x = max(e['x'] for e in detected_elements)
        min_y = min(e['y'] for e in detected_elements)
        max_y = max(e['y'] for e in detected_elements)
        
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        area = bbox_width * bbox_height
        
        print(f"Bounding box: ({min_x},{min_y}) to ({max_x},{max_y})")
        print(f"Dimensions: {bbox_width}x{bbox_height}")
        print(f"Center: ({center_x:.1f}, {center_y:.1f})")
        print(f"Area: {area} pixels²")
        print(f"Formula: center = ((x_min + x_max)/2, (y_min + y_max)/2)")
    
    print()

def demo_gui_automation():
    """
    Section 2: GUI Automation
    Click coordinates, form filling, navigation
    """
    print("=" * 70)
    print("DEMO 2: GUI Automation")
    print("=" * 70)
    
    # 2.1 Coordinate mapping (screen to relative)
    print("\n[2.1] Coordinate Mapping (Screen to Relative)")
    print("-" * 40)
    
    screen_width = 1920
    screen_height = 1080
    
    # Simulated UI elements with screen coordinates
    elements = [
        {'name': 'Login Button', 'screen_x': 960, 'screen_y': 720},
        {'name': 'Email Field', 'screen_x': 800, 'screen_y': 400},
        {'name': 'Password Field', 'screen_x': 800, 'screen_y': 500},
        {'name': 'Submit Link', 'screen_x': 1100, 'screen_y': 650},
    ]
    
    for elem in elements:
        rel_x = elem['screen_x'] / screen_width
        rel_y = elem['screen_y'] / screen_height
        print(f"{elem['name']}:")
        print(f"  Screen: ({elem['screen_x']}, {elem['screen_y']})")
        print(f"  Relative: ({rel_x:.3f}, {rel_y:.3f})")
    
    print(f"\nFormula: relative = screen_coord / screen_dimension")
    print(f"Screen: {screen_width}x{screen_height}")
    
    # 2.2 Form filling simulation
    print("\n[2.2] Form Filling Simulation")
    print("-" * 40)
    
    form_data = {
        'username': 'demo_user',
        'email': 'user@example.com',
        'password': 's3cur3_p@ss',
        'confirm_password': 's3cur3_p@ss'
    }
    
    field_validations = {
        'username': lambda x: len(x) >= 3 and x.isalnum(),
        'email': lambda x: '@' in x and '.' in x.split('@')[-1],
        'password': lambda x: len(x) >= 8 and any(c.isdigit() for c in x),
        'confirm_password': lambda x: x == form_data.get('password', '')
    }
    
    print("Form filling sequence:")
    for field, value in form_data.items():
        is_valid = field_validations[field](value)
        masked = '*' * len(value) if 'password' in field else value
        print(f"  1. Click on '{field}' field")
        print(f"  2. Type: {masked}")
        print(f"  3. Validation: {'PASS' if is_valid else 'FAIL'}")
    
    all_valid = all(field_validations[f](v) for f, v in form_data.items())
    print(f"\nOverall form validity: {'VALID' if all_valid else 'INVALID'}")
    
    # 2.3 Navigation path planning
    print("\n[2.3] Navigation Path Planning")
    print("-" * 40)
    
    pages = {
        'home': {'links': ['about', 'products', 'contact']},
        'about': {'links': ['team', 'careers', 'home']},
        'products': {'links': ['product_a', 'product_b', 'home']},
        'contact': {'links': ['support', 'sales', 'home']},
        'product_a': {'links': ['buy_a', 'products']},
        'buy_a': {'links': ['checkout', 'products']},
        'checkout': {'links': ['payment', 'cart']},
    }
    
    def find_path(graph, start, goal, visited=None):
        if visited is None:
            visited = set()
        if start == goal:
            return [start]
        visited.add(start)
        for next_page in graph.get(start, {}).get('links', []):
            if next_page not in visited:
                path = find_path(graph, next_page, goal, visited.copy())
                if path:
                    return [start] + path
        return None
    
    source, target = 'home', 'checkout'
    path = find_path(pages, source, target)
    
    print(f"Navigation: {source} -> {target}")
    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Steps required: {len(path) - 1}")
    print(f"Formula: BFS/pathfinding on page graph")
    
    # 2.4 Click timing simulation
    print("\n[2.4] Click Timing Simulation")
    print("-" * 40)
    
    random.seed(42)
    num_clicks = 10
    click_intervals = []
    
    prev_time = 0
    for i in range(num_clicks):
        interval = random.expovariate(1.0 / 200)  # Mean 200ms
        click_time = prev_time + interval
        click_intervals.append(interval)
        prev_time = click_time
    
    mean_interval = sum(click_intervals) / len(click_intervals)
    std_interval = math.sqrt(sum((x - mean_interval)**2 for x in click_intervals) / len(click_intervals))
    
    print(f"Simulated {num_clicks} clicks")
    print(f"Mean interval: {mean_interval:.1f}ms")
    print(f"Std deviation: {std_interval:.1f}ms")
    print(f"Min interval: {min(click_intervals):.1f}ms")
    print(f"Max interval: {max(click_intervals):.1f}ms")
    print("Formula: interval ~ Exponential(mean=200ms)")
    print(f"Coefficient of variation: {std_interval/mean_interval:.2f}")
    
    print()

def demo_embodied_ai():
    """
    Section 3: Embodied AI
    Robot perception, action planning, sim-to-real
    """
    print("=" * 70)
    print("DEMO 3: Embodied AI")
    print("=" * 70)
    
    # 3.1 Robot state representation
    print("\n[3.1] Robot State Representation")
    print("-" * 40)
    
    # 3-DOF robot arm state
    joint_angles = [0.5, -1.2, 0.8]  # radians
    link_lengths = [1.0, 0.8, 0.5]
    
    # Forward kinematics - compute end effector position
    x, y = 0, 0
    cumulative_angle = 0
    
    print("Forward Kinematics:")
    print("Joint angles (radians):", [f"{a:.2f}" for a in joint_angles])
    print("Link lengths:", link_lengths)
    
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        cumulative_angle += angle
        x += length * math.cos(cumulative_angle)
        y += length * math.sin(cumulative_angle)
        print(f"  Link {i+1}: angle={cumulative_angle:.2f}, "
              f"end=({x:.3f}, {y:.3f})")
    
    print(f"\nEnd effector position: ({x:.3f}, {y:.3f})")
    print(f"Reach (distance from origin): {math.sqrt(x**2 + y**2):.3f}")
    print("Formula: x = Σ L_i * cos(Σ θ_j), y = Σ L_i * sin(Σ θ_j)")
    
    # 3.2 Obstacle detection and avoidance
    print("\n[3.2] Obstacle Detection and Avoidance")
    print("-" * 40)
    
    obstacles = [
        {'center': (2.0, 1.5), 'radius': 0.5, 'type': 'static'},
        {'center': (3.0, 0.5), 'radius': 0.3, 'type': 'dynamic'},
        {'center': (1.5, 2.0), 'radius': 0.4, 'type': 'static'},
    ]
    
    robot_pos = (0, 0)
    goal_pos = (4.0, 2.0)
    safety_margin = 0.2
    
    def distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    print(f"Robot at: {robot_pos}")
    print(f"Goal at: {goal_pos}")
    print(f"Safety margin: {safety_margin}")
    print(f"\nObstacle analysis:")
    
    for i, obs in enumerate(obstacles):
        dist = distance(robot_pos, obs['center'])
        safe_dist = obs['radius'] + safety_margin
        blocked = dist < safe_dist
        print(f"  Obstacle {i+1}: center={obs['center']}, r={obs['radius']}")
        print(f"    Distance from robot: {dist:.3f}")
        print(f"    Safe distance needed: {safe_dist:.3f}")
        print(f"    Status: {'BLOCKED' if blocked else 'SAFE'}")
    
    print("Formula: collision = dist(robot, obstacle) < radius + margin")
    
    # 3.3 Action space discretization
    print("\n[3.3] Action Space Discretization")
    print("-" * 40)
    
    # Discrete action space for robot movement
    num_actions = 8  # 8 directions
    action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    action_angles = [i * (2 * math.pi / num_actions) for i in range(num_actions)]
    
    # Q-values for each action at current state
    random.seed(42)
    q_values = [random.uniform(-1, 1) for _ in range(num_actions)]
    
    print(f"Action space: {num_actions} discrete directions")
    print("Action mapping:")
    for name, angle, q in zip(action_names, action_angles, q_values):
        print(f"  {name}: angle={math.degrees(angle):.0f}°, Q-value={q:.3f}")
    
    best_action_idx = max(range(num_actions), key=lambda i: q_values[i])
    print(f"\nBest action (greedy): {action_names[best_action_idx]} "
          f"(Q={q_values[best_action_idx]:.3f})")
    
    # Softmax action selection
    temperature = 0.5
    exp_q = [math.exp(q / temperature) for q in q_values]
    sum_exp = sum(exp_q)
    probs = [e / sum_exp for e in exp_q]
    
    print(f"\nSoftmax action distribution (T={temperature}):")
    for name, prob in zip(action_names, probs):
        bar = '#' * int(prob * 40)
        print(f"  {name}: {prob:.3f} {bar}")
    print("Formula: P(a) = exp(Q(a)/T) / Σ exp(Q(a')/T)")
    
    # 3.4 Sim-to-real gap modeling
    print("\n[3.4] Sim-to-Real Gap Modeling")
    print("-" * 40)
    
    # Simulated vs real performance metrics
    metrics = {
        'success_rate': {'sim': 0.92, 'real': 0.71},
        'avg_reward': {'sim': 85.3, 'real': 62.1},
        'completion_time': {'sim': 12.5, 'real': 18.3},
        'energy_usage': {'sim': 100.0, 'real': 135.2},
    }
    
    print("Sim-to-Real Performance Comparison:")
    print(f"{'Metric':<20} {'Sim':>10} {'Real':>10} {'Gap':>10} {'Ratio':>10}")
    print("-" * 60)
    
    for metric_name, values in metrics.items():
        sim_val = values['sim']
        real_val = values['real']
        gap = real_val - sim_val
        ratio = real_val / sim_val if sim_val != 0 else float('inf')
        print(f"{metric_name:<20} {sim_val:>10.1f} {real_val:>10.1f} "
              f"{gap:>+10.1f} {ratio:>10.2f}")
    
    # Domain randomization effect
    print("\nDomain Randomization Impact:")
    random.seed(42)
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    real_performance = []
    
    for noise in noise_levels:
        # Simulate: more randomization improves real-world transfer
        base_perf = 0.71
        improvement = 0.15 * (1 - math.exp(-3 * noise))
        perf = base_perf + improvement
        real_performance.append(perf)
        print(f"  Noise σ={noise:.1f}: Real perf = {perf:.3f}")
    
    print("Formula: perf_real = base + max_improvement * (1 - e^(-k*noise))")
    
    print()

def demo_multimodal_planning():
    """
    Section 4: Multimodal Planning
    Visual reasoning, spatial understanding
    """
    print("=" * 70)
    print("DEMO 4: Multimodal Planning")
    print("=" * 70)
    
    # 4.1 Visual scene graph generation
    print("\n[4.1] Visual Scene Graph Generation")
    print("-" * 40)
    
    # Simulated scene objects with attributes
    scene_objects = [
        {'id': 0, 'name': 'table', 'attributes': ['wooden', 'brown', 'large'],
         'bbox': (100, 200, 400, 150)},
        {'id': 1, 'name': 'cup', 'attributes': ['ceramic', 'white', 'small'],
         'bbox': (250, 180, 50, 60)},
        {'id': 2, 'name': 'book', 'attributes': ['hardcover', 'blue', 'medium'],
         'bbox': (300, 190, 80, 20)},
        {'id': 3, 'name': 'laptop', 'attributes': ['silver', 'open', 'medium'],
         'bbox': (150, 160, 200, 120)},
    ]
    
    # Spatial relationships
    relationships = [
        (1, 0, 'on'),    # cup on table
        (2, 0, 'on'),    # book on table
        (3, 0, 'on'),    # laptop on table
        (1, 3, 'next_to'),# cup next to laptop
    ]
    
    print("Scene Objects:")
    for obj in scene_objects:
        print(f"  [{obj['id']}] {obj['name']}: {', '.join(obj['attributes'])}")
        print(f"      BBox: {obj['bbox']}")
    
    print("\nRelationships:")
    for subj_id, obj_id, rel in relationships:
        subj = scene_objects[subj_id]['name']
        obj = scene_objects[obj_id]['name']
        print(f"  {subj} --[{rel}]--> {obj}")
    
    print(f"\nTotal: {len(scene_objects)} objects, {len(relationships)} relationships")
    print("Formula: Scene Graph = (Objects, Attributes, Relationships)")
    
    # 4.2 Spatial reasoning queries
    print("\n[4.2] Spatial Reasoning Queries")
    print("-" * 40)
    
    def bbox_center(bbox):
        return (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
    
    def bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0]+box1[2], box2[0]+box2[2])
        y2 = min(box1[1]+box1[3], box2[1]+box2[3])
        
        intersection = max(0, x2-x1) * max(0, y2-y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # Query: What is left of the cup?
    cup_bbox = scene_objects[1]['bbox']
    cup_center = bbox_center(cup_bbox)
    
    print(f"Query: What is left of the cup (center: {cup_center})?")
    for obj in scene_objects:
        if obj['name'] == 'cup':
            continue
        obj_center = bbox_center(obj['bbox'])
        if obj_center[0] < cup_center[0]:
            print(f"  Answer: {obj['name']} (center: {obj_center})")
    
    # IoU calculations
    print("\nBounding Box IoU (Intersection over Union):")
    print("Formula: IoU = Area(Intersection) / Area(Union)")
    
    for i in range(len(scene_objects)):
        for j in range(i+1, len(scene_objects)):
            iou = bbox_iou(scene_objects[i]['bbox'], scene_objects[j]['bbox'])
            if iou > 0:
                print(f"  {scene_objects[i]['name']} ∩ {scene_objects[j]['name']}: "
                      f"IoU = {iou:.3f}")
    
    # 4.3 Visual question answering pipeline
    print("\n[4.3] Visual Question Answering Pipeline")
    print("-" * 40)
    
    questions = [
        {'q': 'How many objects are on the table?', 'type': 'count'},
        {'q': 'What color is the cup?', 'type': 'attribute'},
        {'q': 'Is the laptop open?', 'type': 'yes_no'},
        {'q': 'Which object is largest?', 'type': 'comparison'},
    ]
    
    answers = []
    objects_on_table = [r[0] for r in relationships if r[1] == 0 and r[2] == 'on']
    
    for question in questions:
        if question['type'] == 'count':
            answer = str(len(objects_on_table))
        elif question['type'] == 'attribute':
            cup = scene_objects[1]
            answer = cup['attributes'][1]  # 'white'
        elif question['type'] == 'yes_no':
            laptop = scene_objects[3]
            answer = 'yes' if 'open' in laptop['attributes'] else 'no'
        elif question['type'] == 'comparison':
            largest = max(scene_objects, key=lambda o: o['bbox'][2] * o['bbox'][3])
            answer = largest['name']
        
        answers.append(answer)
        print(f"Q: {question['q']}")
        print(f"A: {answer}")
    
    print("\nVQA Pipeline: Question Encoding → Visual Features → Fusion → Answer")
    
    # 4.4 Attention visualization for spatial understanding
    print("\n[4.4] Attention Visualization for Spatial Understanding")
    print("-" * 40)
    
    # Simulated attention weights between question words and image regions
    question_words = ['what', 'color', 'is', 'the', 'cup']
    image_regions = ['table', 'cup', 'book', 'laptop', 'background']
    
    random.seed(42)
    attention_matrix = []
    for word in question_words:
        row = [random.random() for _ in range(len(image_regions))]
        total = sum(row)
        row = [v / total for v in row]
        attention_matrix.append(row)
    
    print("Question-Image Attention Matrix:")
    print(f"{'':>12}", end='')
    for region in image_regions:
        print(f"{region:>10}", end='')
    print()
    
    for word, row in zip(question_words, attention_matrix):
        print(f"{word:>12}", end='')
        for val in row:
            print(f"{val:>10.3f}", end='')
        print()
    
    # Find highest attention for each word
    print("\nHighest attention per word:")
    for word, row in zip(question_words, attention_matrix):
        best_idx = max(range(len(row)), key=lambda i: row[i])
        print(f"  '{word}' → {image_regions[best_idx]} ({row[best_idx]:.3f})")
    
    print("\nFormula: attention(q, v) = softmax(q^T W_k v / sqrt(d))")
    
    print()

if __name__ == "__main__":
    demo_vision_tool_use()
    demo_gui_automation()
    demo_embodied_ai()
    demo_multimodal_planning()