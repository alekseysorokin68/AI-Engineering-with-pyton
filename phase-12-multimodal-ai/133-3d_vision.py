"""
133 — 3D Vision: point clouds, depth estimation, 3D understanding

Темы:
  1. Point Cloud Representation (3D coordinates, voxelization, octree)
  2. Depth Estimation (monocular depth, stereo vision concepts)
  3. Point Cloud Processing (sampling, grouping, feature extraction)
  4. 3D Object Detection (bounding boxes in 3D, evaluation metrics)

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
# Demo 1: Point Cloud Representation
# =============================================================================
def demo_point_cloud_representation():
    print("=" * 70)
    print("DEMO 1: Point Cloud Representation — 3D coordinates, voxelization, octree")
    print("=" * 70)

    # --- Sub-example 1: Basic point cloud operations ---
    print("\n[1.1] Point cloud basics — 3D coordinates and operations")
    print("-" * 60)

    def generate_point_cloud(num_points=100, seed=42):
        """Generate a synthetic 3D point cloud."""
        rng = random.Random(seed)
        points = []
        for _ in range(num_points):
            x = rng.uniform(-5, 5)
            y = rng.uniform(-5, 5)
            # Simulate a sphere surface with noise
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(0, math.pi)
            r = 3 + rng.gauss(0, 0.2)
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            points.append((x, y, z))
        return points

    def point_cloud_stats(points):
        """Compute point cloud statistics."""
        n = len(points)
        if n == 0:
            return {}
        
        # Compute centroid
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        cz = sum(p[2] for p in points) / n
        
        # Compute bounds
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        min_z = min(p[2] for p in points)
        max_z = max(p[2] for p in points)
        
        # Compute average distance from centroid
        avg_dist = sum(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) for p in points) / n
        
        # Compute density (points per unit volume)
        vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        density = n / vol if vol > 0 else 0
        
        return {
            "num_points": n,
            "centroid": (round(cx, 3), round(cy, 3), round(cz, 3)),
            "bounds": {
                "x": (round(min_x, 3), round(max_x, 3)),
                "y": (round(min_y, 3), round(max_y, 3)),
                "z": (round(min_z, 3), round(max_z, 3)),
            },
            "avg_distance": round(avg_dist, 3),
            "density": round(density, 3)
        }

    pc = generate_point_cloud(200)
    stats = point_cloud_stats(pc)
    print(f"Point cloud with {stats['num_points']} points")
    print(f"Centroid: {stats['centroid']}")
    print(f"Bounds:")
    for axis, (lo, hi) in stats['bounds'].items():
        print(f"  {axis}: [{lo:.3f}, {hi:.3f}] (range={hi-lo:.3f})")
    print(f"Average distance from centroid: {stats['avg_distance']:.3f}")
    print(f"Point density: {stats['density']:.3f} pts/unit³")

    # --- Sub-example 2: Voxelization ---
    print("\n[1.2] Voxelization — discretizing 3D space into voxels")
    print("-" * 60)

    def voxelize(points, voxel_size=1.0):
        """Convert point cloud to voxel grid.
        
        Voxelization formula:
        - For each point (x, y, z), compute voxel index:
          ix = floor((x - min_x) / voxel_size)
          iy = floor((y - min_y) / voxel_size)
          iz = floor((z - min_z) / voxel_size)
        """
        if not points:
            return {}, {}
        
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        min_z = min(p[2] for p in points)
        
        voxels = collections.defaultdict(list)
        for p in points:
            ix = int(math.floor((p[0] - min_x) / voxel_size))
            iy = int(math.floor((p[1] - min_y) / voxel_size))
            iz = int(math.floor((p[2] - min_z) / voxel_size))
            voxels[(ix, iy, iz)].append(p)
        
        # Compute voxel occupancy stats
        occupancy = {}
        for key, pts in voxels.items():
            # Centroid of points in this voxel
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)
            occupancy[key] = {
                "count": len(pts),
                "centroid": (round(cx, 3), round(cy, 3), round(cz, 3))
            }
        
        return voxels, occupancy

    pc_small = generate_point_cloud(100, seed=123)
    voxels, occ = voxelize(pc_small, voxel_size=1.0)
    print(f"Input: {len(pc_small)} points, voxel_size=1.0")
    print(f"Occupied voxels: {len(voxels)}")
    
    total_volume = len(voxels) * 1.0**3
    print(f"Total occupied volume: {total_volume:.1f} cubic units")
    
    # Show voxel occupancy distribution
    counts = [v["count"] for v in occ.values()]
    if counts:
        print(f"Points per voxel: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
    
    # Show sample voxels
    print("Sample voxels:")
    for i, (key, info) in enumerate(list(occ.items())[:5]):
        print(f"  Voxel {key}: {info['count']} points, centroid={info['centroid']}")

    # --- Sub-example 3: Octree representation ---
    print("\n[1.3] Octree — hierarchical spatial decomposition")
    print("-" * 60)

    class OctreeNode:
        """Simple octree node for point cloud compression."""
        def __init__(self, center, size, points, depth=0, max_depth=4, max_points=8):
            self.center = center
            self.size = size
            self.points = points
            self.depth = depth
            self.children = []
            self.is_leaf = True
            
            if depth < max_depth and len(points) > max_points:
                self._subdivide()
        
        def _subdivide(self):
            """Subdivide into 8 children."""
            half = self.size / 2
            offsets = [(dx, dy, dz) for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]]
            
            child_points = [[] for _ in range(8)]
            for p in self.points:
                # Determine which octant
                idx = 0
                if p[0] > self.center[0]: idx += 1
                if p[1] > self.center[1]: idx += 2
                if p[2] > self.center[2]: idx += 4
                child_points[idx].append(p)
            
            for i, (dx, dy, dz) in enumerate(offsets):
                if child_points[i]:
                    child_center = (
                        self.center[0] + dx * half/2,
                        self.center[1] + dy * half/2,
                        self.center[2] + dz * half/2
                    )
                    child = OctreeNode(child_center, half, child_points[i], 
                                      self.depth + 1)
                    self.children.append(child)
            
            if self.children:
                self.is_leaf = False
                self.points = []  # Internal nodes don't store points
        
        def count_nodes(self):
            """Count total nodes in subtree."""
            count = 1
            for child in self.children:
                count += child.count_nodes()
            return count
        
        def count_leaves(self):
            """Count leaf nodes."""
            if self.is_leaf:
                return 1
            count = 0
            for child in self.children:
                count += child.count_leaves()
            return count

    def build_octree(points, max_depth=4, max_points=8):
        """Build octree from point cloud."""
        if not points:
            return None
        
        # Compute bounding box
        min_coords = [min(p[i] for p in points) for i in range(3)]
        max_coords = [max(p[i] for p in points) for i in range(3)]
        center = tuple((min_coords[i] + max_coords[i]) / 2 for i in range(3))
        size = max(max_coords[i] - min_coords[i] for i in range(3))
        
        return OctreeNode(center, size, points, max_depth=max_depth, max_points=max_points)

    pc_octree = generate_point_cloud(150, seed=456)
    octree = build_octree(pc_octree, max_depth=3, max_points=10)
    
    print(f"Input: {len(pc_octree)} points")
    print(f"Octree stats:")
    print(f"  Total nodes: {octree.count_nodes()}")
    print(f"  Leaf nodes:  {octree.count_leaves()}")
    print(f"  Compression: {len(pc_octree)} points → {octree.count_leaves()} leaves")
    
    # Compute compression ratio
    naive_size = len(pc_octree) * 3 * 4  # 3 floats * 4 bytes each
    octree_size = octree.count_nodes() * (3 + 1) * 4  # center + size
    compression = naive_size / max(octree_size, 1)
    print(f"  Naive storage: {naive_size} bytes")
    print(f"  Octree storage: {octree_size} bytes")
    print(f"  Compression ratio: {compression:.2f}x")

    # --- Sub-example 4: Point cloud encoding for neural networks ---
    print("\n[1.4] Point encoding — feature encoding for 3D deep learning")
    print("-" * 60)

    def encode_points_floater(points, num_freqs=10):
        """Encode 3D points using positional encoding (similar to NeRF).
        
        Positional encoding formula:
        PE(x) = [sin(2^0 * π * x), cos(2^0 * π * x), ..., sin(2^(L-1) * π * x), cos(2^(L-1) * π * x)]
        
        This maps each coordinate to a higher-dimensional space, helping
        neural networks learn high-frequency functions.
        """
        encoded = []
        for p in points[:5]:  # Show first 5 points
            pe = []
            for freq in range(num_freqs):
                for coord in p:
                    pe.append(math.sin((2 ** freq) * math.pi * coord))
                    pe.append(math.cos((2 ** freq) * math.pi * coord))
            encoded.append(pe)
        
        return encoded, len(encoded[0])

    pc_encode = [(1.5, 2.3, -0.8), (0.0, 1.0, 0.5), (-2.1, 0.5, 3.2)]
    encoded, feat_dim = encode_points_floater(pc_encode, num_freqs=4)
    
    print(f"Input: {len(pc_encode)} points with 3 coordinates each")
    print(f"Positional encoding: {feat_dim} features per point")
    print(f"\nEncoding formula: PE(x) = [sin(2^k * π * x), cos(2^k * π * x)] for k=0..{3}")
    print(f"\nSample encoding (first point {pc_encode[0]}):")
    for i in range(min(8, feat_dim)):
        print(f"  PE[{i}] = {encoded[0][i]:.4f}")

    print("\n" + "=" * 70)
    print("Point cloud representation: coordinates → voxels → octree → encoding")
    print("=" * 70)


# =============================================================================
# Demo 2: Depth Estimation
# =============================================================================
def demo_depth_estimation():
    print("\n" + "=" * 70)
    print("DEMO 2: Depth Estimation — monocular depth, stereo vision concepts")
    print("=" * 70)

    # --- Sub-example 1: Depth map representation ---
    print("\n[2.1] Depth map — representing distance as 2D image")
    print("-" * 60)

    def generate_depth_map(width=20, height=15, seed=42):
        """Generate a synthetic depth map.
        
        Depth map: each pixel stores distance from camera (Z coordinate).
        Typical range: 0.1m to 100m for indoor/outdoor scenes.
        """
        rng = random.Random(seed)
        depth_map = []
        
        for y in range(height):
            row = []
            for x in range(width):
                # Simulate a room with walls
                # Floor: closer at bottom, farther at top
                floor_depth = 2.0 + (y / height) * 8.0
                
                # Add wall objects
                if 5 <= x <= 8 and 3 <= y <= 7:
                    depth = 3.0  # Close object (table)
                elif 12 <= x <= 15 and 2 <= y <= 10:
                    depth = 5.0  # Medium object (chair)
                elif x >= 17:
                    depth = 10.0  # Far wall
                else:
                    depth = floor_depth + rng.gauss(0, 0.2)
                
                row.append(max(0.1, depth))
            depth_map.append(row)
        
        return depth_map

    depth_map = generate_depth_map(20, 15)
    
    print(f"Depth map: {len(depth_map[0])}x{len(depth_map)} pixels")
    print(f"Depth range: [{min(min(row) for row in depth_map):.2f}, {max(max(row) for row in depth_map):.2f}] meters")
    
    # Statistics
    all_depths = [d for row in depth_map for d in row]
    mean_depth = sum(all_depths) / len(all_depths)
    std_depth = math.sqrt(sum((d - mean_depth)**2 for d in all_depths) / len(all_depths))
    print(f"Mean depth: {mean_depth:.2f}m, Std: {std_depth:.2f}m")
    
    # Visualize as ASCII
    print("\nDepth map visualization (darker = closer):")
    for row in depth_map[:8]:
        line = ""
        for d in row:
            # Normalize to 0-9
            normalized = int((d - 1) / 9 * 9)
            normalized = max(0, min(9, normalized))
            line += str(normalized)
        print(f"  {line}")

    # --- Sub-example 2: Monocular depth estimation concepts ---
    print("\n[2.2] Monocular depth cues — estimating depth from single image")
    print("-" * 60)

    def compute_depth_cues(image_features, seed=42):
        """Compute depth cues from image features.
        
        Monocular depth cues:
        1. Texture gradient: denser texture = farther away
        2. Size constancy: known object size → distance estimate
        3. Height in field: lower = closer (for ground plane)
        4. Aerial perspective: hazier = farther
        """
        rng = random.Random(seed)
        cues = {}
        
        # Texture gradient
        texture_density = image_features.get("texture_density", 0.5)
        cues["texture_depth"] = 1.0 / (texture_density + 0.1)  # Inverse relationship
        
        # Size constancy (assume car height = 1.5m)
        apparent_size = image_features.get("object_height_px", 100)
        image_height = image_features.get("image_height", 480)
        focal_length = image_features.get("focal_length", 500)
        real_size = 1.5  # meters
        
        # depth = (real_size * focal_length) / apparent_size
        cues["size_depth"] = (real_size * focal_length) / max(apparent_size, 1)
        
        # Height in field of view
        y_position = image_features.get("y_position", 0.5)
        # Assume horizon at y=0.3, ground plane below
        cues["height_depth"] = max(0.5, (y_position - 0.3) * 20)
        
        # Aerial perspective (haze level)
        haze = image_features.get("haze_level", 0.2)
        cues["aerial_depth"] = 5.0 * haze  # More haze = farther
        
        # Combined estimate (weighted average)
        weights = [0.2, 0.4, 0.3, 0.1]
        depths = [cues["texture_depth"], cues["size_depth"], 
                  cues["height_depth"], cues["aerial_depth"]]
        cues["combined_depth"] = sum(w * d for w, d in zip(weights, depths))
        
        return cues

    # Example features
    features = {
        "texture_density": 0.3,
        "object_height_px": 80,
        "image_height": 480,
        "focal_length": 500,
        "y_position": 0.6,
        "haze_level": 0.15
    }
    
    cues = compute_depth_cues(features)
    print("Monocular depth cues:")
    for cue_name, depth in cues.items():
        print(f"  {cue_name:20s}: {depth:.3f}m")
    
    print(f"\nInterpretation:")
    print(f"  - Texture suggests ~{cues['texture_depth']:.1f}m depth")
    print(f"  - Object size suggests ~{cues['size_depth']:.1f}m depth")
    print(f"  - Position in image suggests ~{cues['height_depth']:.1f}m depth")
    print(f"  - Combined estimate: ~{cues['combined_depth']:.1f}m")

    # --- Sub-example 3: Stereo vision fundamentals ---
    print("\n[2.3] Stereo vision — depth from disparity")
    print("-" * 60)

    def stereo_depth_estimation(disparity_map, baseline, focal_length):
        """Compute depth from stereo disparity.
        
        Stereo depth formula:
        Z = (f * B) / d
        
        where:
        - Z = depth (distance from camera)
        - f = focal length (pixels)
        - B = baseline (distance between cameras, meters)
        - d = disparity (difference in pixel position, pixels)
        """
        depth_map = []
        for row in disparity_map:
            depth_row = []
            for d in row:
                if d > 0:
                    z = (focal_length * baseline) / d
                    depth_row.append(z)
                else:
                    depth_row.append(float('inf'))  # No disparity = infinite depth
            depth_map.append(depth_row)
        return depth_map

    # Simulate disparity map
    rng = random.Random(42)
    width, height = 10, 8
    disparity_map = []
    for y in range(height):
        row = []
        for x in range(width):
            # Disparity decreases with distance
            base_disparity = 30 - x * 2 - y * 1
            noise = rng.gauss(0, 0.5)
            row.append(max(0, base_disparity + noise))
        disparity_map.append(row)
    
    # Stereo parameters
    baseline = 0.12  # 12cm baseline (typical stereo camera)
    focal_length = 640  # pixels
    
    stereo_depth = stereo_depth_estimation(disparity_map, baseline, focal_length)
    
    print(f"Stereo parameters:")
    print(f"  Baseline: {baseline*100:.0f}cm")
    print(f"  Focal length: {focal_length}px")
    print(f"  Formula: Z = (f × B) / d = ({focal_length} × {baseline}) / d")
    
    print(f"\nDisparity → Depth mapping:")
    print(f"  Disparity (px) → Depth (m)")
    for d_val in [5, 10, 15, 20, 25, 30]:
        z = (focal_length * baseline) / d_val
        print(f"  {d_val:6d}px → {z:.2f}m")

    # --- Sub-example 4: Depth refinement with edge detection ---
    print("\n[2.4] Depth refinement — edge-aware depth filtering")
    print("-" * 60)

    def edge_aware_depth_filter(depth_map, edge_map, sigma_s=0.1, sigma_r=0.1):
        """Apply edge-aware bilateral filtering to depth map.
        
        Bilateral filter formula:
        filtered(p) = Σ G_s(p-q) * G_r(I(p)-I(q)) * depth(q) / W
        
        where:
        - G_s = spatial Gaussian (distance between pixels)
        - G_r = range Gaussian (depth difference)
        - W = normalization factor
        """
        h = len(depth_map)
        w = len(depth_map[0])
        filtered = [[0.0] * w for _ in range(h)]
        
        for y in range(h):
            for x in range(w):
                weighted_sum = 0
                weight_sum = 0
                
                # 3x3 kernel
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            # Spatial weight
                            spatial_dist = math.sqrt(dx**2 + dy**2)
                            spatial_weight = math.exp(-spatial_dist**2 / (2 * sigma_s**2))
                            
                            # Range weight (depth difference)
                            depth_diff = abs(depth_map[y][x] - depth_map[ny][nx])
                            range_weight = math.exp(-depth_diff**2 / (2 * sigma_r**2))
                            
                            # Edge-aware: reduce weight if edge exists
                            edge_weight = 1.0 - edge_map[ny][nx] * 0.5
                            
                            weight = spatial_weight * range_weight * edge_weight
                            weighted_sum += weight * depth_map[ny][nx]
                            weight_sum += weight
                
                filtered[y][x] = weighted_sum / max(weight_sum, 1e-8)
        
        return filtered

    # Generate edge map (simulated)
    edge_map = [[0.0] * 20 for _ in range(15)]
    for y in range(15):
        for x in range(20):
            if random.random() < 0.1:  # 10% edges
                edge_map[y][x] = 1.0
    
    filtered_depth = edge_aware_depth_filter(depth_map, edge_map)
    
    print("Edge-aware bilateral filtering:")
    print(f"  σ_s (spatial) = 0.1, σ_r (range) = 0.1")
    
    # Compute smoothing effect
    orig_var = sum((d - mean_depth)**2 for d in all_depths) / len(all_depths)
    filtered_depths = [d for row in filtered_depth for d in row]
    filtered_mean = sum(filtered_depths) / len(filtered_depths)
    filtered_var = sum((d - filtered_mean)**2 for d in filtered_depths) / len(filtered_depths)
    
    print(f"  Original variance: {orig_var:.4f}")
    print(f"  Filtered variance: {filtered_var:.4f}")
    print(f"  Smoothing effect: {(1 - filtered_var/orig_var)*100:.1f}% variance reduction")
    print(f"  Edge preservation: Edges reduce filter weight by 50%")

    print("\n" + "=" * 70)
    print("Depth estimation: monocular cues → stereo disparity → refinement")
    print("=" * 70)


# =============================================================================
# Demo 3: Point Cloud Processing
# =============================================================================
def demo_point_cloud_processing():
    print("\n" + "=" * 70)
    print("DEMO 3: Point Cloud Processing — sampling, grouping, feature extraction")
    print("=" * 70)

    # --- Sub-example 1: Farthest Point Sampling ---
    print("\n[3.1] Farthest Point Sampling (FPS) — selecting representative points")
    print("-" * 60)

    def farthest_point_sampling(points, num_samples, seed=42):
        """Select points using Farthest Point Sampling.
        
        Algorithm:
        1. Start with random point
        2. Iteratively select point farthest from all selected points
        3. Repeat until num_samples points selected
        
        This ensures uniform coverage of the point cloud.
        """
        rng = random.Random(seed)
        n = len(points)
        if num_samples >= n:
            return list(range(n))
        
        # Initialize with random point
        selected = [rng.randint(0, n - 1)]
        
        # Distance from each point to nearest selected point
        min_distances = [float('inf')] * n
        
        for _ in range(num_samples - 1):
            # Update distances
            last_selected = points[selected[-1]]
            for i in range(n):
                if i not in selected:
                    dist = math.sqrt(sum((a - b)**2 for a, b in zip(points[i], last_selected)))
                    min_distances[i] = min(min_distances[i], dist)
            
            # Select farthest point
            best_idx = max(range(n), key=lambda i: min_distances[i] if i not in selected else -1)
            selected.append(best_idx)
        
        return selected

    def generate_3d_points(n=200, seed=42):
        """Generate random 3D points."""
        rng = random.Random(seed)
        return [(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(n)]

    points = generate_3d_points(200)
    num_samples = 20
    
    selected_indices = farthest_point_sampling(points, num_samples)
    selected_points = [points[i] for i in selected_indices]
    
    print(f"Input: {len(points)} points")
    print(f"Sampled: {num_samples} points using FPS")
    
    # Compute coverage metrics
    def compute_coverage(all_points, sample_points):
        """Compute coverage: average distance to nearest sample."""
        total_dist = 0
        for p in all_points:
            min_dist = min(math.sqrt(sum((a-b)**2 for a,b in zip(p, sp))) for sp in sample_points)
            total_dist += min_dist
        return total_dist / len(all_points)
    
    coverage = compute_coverage(points, selected_points)
    print(f"Average distance to nearest sample: {coverage:.3f}")
    
    # Compare with random sampling
    rng = random.Random(123)
    random_indices = rng.sample(range(len(points)), num_samples)
    random_points = [points[i] for i in random_indices]
    random_coverage = compute_coverage(points, random_points)
    print(f"Random sampling coverage: {random_coverage:.3f}")
    print(f"FPS improvement: {(1 - coverage/random_coverage)*100:.1f}% better coverage")

    # --- Sub-example 2: K-Nearest Neighbors grouping ---
    print("\n[3.2] K-Nearest Neighbors — local neighborhood extraction")
    print("-" * 60)

    def knn_search(query_point, points, k=5):
        """Find K nearest neighbors using brute force.
        
        Distance metric: Euclidean distance
        d(p, q) = sqrt(Σ(p_i - q_i)²)
        """
        distances = []
        for i, p in enumerate(points):
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(query_point, p)))
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])
        return distances[:k]

    points_knn = generate_3d_points(100, seed=789)
    query = (0.0, 0.0, 0.0)
    k = 5
    
    neighbors = knn_search(query, points_knn, k)
    
    print(f"Query point: {query}")
    print(f"K={k} nearest neighbors:")
    for dist, idx in neighbors:
        p = points_knn[idx]
        print(f"  Neighbor {idx}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  dist={dist:.3f}")
    
    # Compute neighborhood statistics
    neighbor_dists = [d for d, _ in neighbors]
    avg_dist = sum(neighbor_dists) / len(neighbor_dists)
    max_dist = max(neighbor_dists)
    print(f"\nNeighborhood statistics:")
    print(f"  Average distance: {avg_dist:.3f}")
    print(f"  Max distance: {max_dist:.3f}")
    print(f"  Neighborhood radius: {max_dist:.3f}")

    # --- Sub-example 3: Feature extraction ---
    print("\n[3.3] Local feature extraction — geometric features for each point")
    print("-" * 60)

    def extract_local_features(points, point_idx, k=10):
        """Extract geometric features for a point using its neighborhood.
        
        Features:
        1. Centroid offset: distance from point to neighborhood centroid
        2. Surface normal: via PCA on local points
        3. Linearity: ratio of eigenvalues (λ1/λ2)
        4. Planarity: ratio of eigenvalues ((λ2-λ3)/λ1)
        5. Sphericity: ratio of eigenvalues (λ3/λ1)
        """
        # Get k nearest neighbors
        neighbors = knn_search(points[point_idx], points, k + 1)  # +1 for self
        neighbor_points = [points[idx] for _, idx in neighbors if idx != point_idx][:k]
        
        if len(neighbor_points) < 3:
            return {"error": "insufficient neighbors"}
        
        # Compute centroid
        n = len(neighbor_points)
        centroid = tuple(sum(p[i] for p in neighbor_points) / n for i in range(3))
        
        # Compute covariance matrix
        cov = [[0.0] * 3 for _ in range(3)]
        for p in neighbor_points:
            diff = [p[i] - centroid[i] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    cov[i][j] += diff[i] * diff[j]
        for i in range(3):
            for j in range(3):
                cov[i][j] /= n
        
        # Simple eigenvalue estimation (3x3 symmetric matrix)
        # For demo purposes, use diagonal dominance as proxy
        eigenvalues = [cov[i][i] for i in range(3)]
        eigenvalues.sort(reverse=True)
        
        # Compute features
        centroid_offset = math.sqrt(sum((points[point_idx][i] - centroid[i])**2 for i in range(3)))
        
        lambda1, lambda2, lambda3 = eigenvalues
        linearity = lambda1 / max(lambda2, 1e-8)
        planarity = (lambda2 - lambda3) / max(lambda1, 1e-8)
        sphericity = lambda3 / max(lambda1, 1e-8)
        
        # Surface normal (simplified: direction of smallest variance)
        normal = [0, 0, 1]  # Placeholder
        
        return {
            "centroid_offset": round(centroid_offset, 4),
            "eigenvalues": [round(e, 4) for e in eigenvalues],
            "linearity": round(linearity, 4),
            "planarity": round(planarity, 4),
            "sphericity": round(sphericity, 4),
            "surface_normal": normal
        }

    # Extract features for a sample point
    sample_idx = 0
    features = extract_local_features(points_knn, sample_idx, k=10)
    
    print(f"Features for point {sample_idx}: {points_knn[sample_idx]}")
    print(f"  Centroid offset: {features['centroid_offset']}")
    print(f"  Eigenvalues: {features['eigenvalues']}")
    print(f"  Linearity:   {features['linearity']:.4f} (1=line, >1=elongated)")
    print(f"  Planarity:   {features['planarity']:.4f} (1=planar)")
    print(f"  Sphericity:  {features['sphericity']:.4f} (1=spherical)")
    print(f"  Interpretation: {'Linear' if features['linearity'] > 2 else 'Planar' if features['planarity'] > 0.5 else 'Scattered'}")

    # --- Sub-example 4: Point cloud downsampling ---
    print("\n[3.4] Downsampling strategies — reducing point cloud density")
    print("-" * 60)

    def voxel_grid_downsample(points, voxel_size=1.0):
        """Downsample using voxel grid averaging.
        
        All points within a voxel are replaced by their centroid.
        """
        min_coords = [min(p[i] for p in points) for i in range(3)]
        
        voxels = collections.defaultdict(list)
        for p in points:
            key = tuple(int(math.floor((p[i] - min_coords[i]) / voxel_size)) for i in range(3))
            voxels[key].append(p)
        
        downsampled = []
        for key, pts in voxels.items():
            centroid = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
            downsampled.append(centroid)
        
        return downsampled, len(voxels)

    points_down = generate_3d_points(500, seed=321)
    
    # Different voxel sizes
    voxel_sizes = [0.5, 1.0, 2.0]
    print(f"Input: {len(points_down)} points")
    print(f"\nVoxel grid downsampling:")
    for vs in voxel_sizes:
        downsampled, num_voxels = voxel_grid_downsample(points_down, vs)
        reduction = (1 - len(downsampled) / len(points_down)) * 100
        print(f"  Voxel size {vs}: {len(downsampled)} points ({reduction:.1f}% reduction), {num_voxels} voxels")

    print("\n" + "=" * 70)
    print("Point cloud processing: FPS → KNN → features → downsampling")
    print("=" * 70)


# =============================================================================
# Demo 4: 3D Object Detection
# =============================================================================
def demo_3d_object_detection():
    print("\n" + "=" * 70)
    print("DEMO 4: 3D Object Detection — bounding boxes in 3D, evaluation metrics")
    print("=" * 70)

    # --- Sub-example 1: 3D bounding box representation ---
    print("\n[4.1] 3D bounding box — axis-aligned and oriented boxes")
    print("-" * 60)

    class BBox3D:
        """3D bounding box representation."""
        def __init__(self, center, dimensions, rotation=0.0, label="object"):
            """
            center: (x, y, z) center of box
            dimensions: (l, w, h) length, width, height
            rotation: rotation around z-axis in radians
            """
            self.center = center
            self.dimensions = dimensions
            self.rotation = rotation
            self.label = label
        
        def volume(self):
            """Compute volume: V = l × w × h"""
            return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        
        def corners(self):
            """Compute 8 corners of the box."""
            l, w, h = [d/2 for d in self.dimensions]
            # Axis-aligned corners relative to center
            corners_local = [
                (-l, -w, -h), (l, -w, -h), (l, w, -h), (-l, w, -h),
                (-l, -w, h), (l, -w, h), (l, w, h), (-l, w, h)
            ]
            
            # Apply rotation around z-axis
            cos_r = math.cos(self.rotation)
            sin_r = math.sin(self.rotation)
            
            corners_rotated = []
            for c in corners_local:
                x_rot = c[0] * cos_r - c[1] * sin_r
                y_rot = c[0] * sin_r + c[1] * cos_r
                corners_rotated.append((
                    x_rot + self.center[0],
                    y_rot + self.center[1],
                    c[2] + self.center[2]
                ))
            return corners_rotated

    # Create example boxes
    boxes = [
        BBox3D(center=(1.0, 2.0, 0.5), dimensions=(2.0, 1.0, 1.5), rotation=0.3, label="car"),
        BBox3D(center=(5.0, 3.0, 1.0), dimensions=(0.5, 0.5, 2.0), rotation=0.0, label="pedestrian"),
        BBox3D(center=(3.0, 1.0, 0.8), dimensions=(3.0, 1.5, 1.2), rotation=-0.5, label="truck"),
    ]
    
    print("3D Bounding boxes:")
    for box in boxes:
        corners = box.corners()
        print(f"\n  {box.label}:")
        print(f"    Center: {box.center}")
        print(f"    Dimensions (l,w,h): {box.dimensions}")
        print(f"    Rotation: {math.degrees(box.rotation):.1f}°")
        print(f"    Volume: {box.volume():.2f} m³")
        print(f"    Corner 0: ({corners[0][0]:.2f}, {corners[0][1]:.2f}, {corners[0][2]:.2f})")

    # --- Sub-example 2: 3D IoU computation ---
    print("\n[4.2] 3D Intersection over Union (IoU) — measuring box overlap")
    print("-" * 60)

    def compute_axis_aligned_iou(box1, box2):
        """Compute IoU for axis-aligned 3D boxes.
        
        IoU = Intersection Volume / Union Volume
        
        For axis-aligned boxes:
        1. Compute overlap along each axis
        2. Intersection = product of overlaps
        3. Union = V1 + V2 - Intersection
        """
        # For axis-aligned boxes (rotation=0)
        c1, d1 = box1.center, box1.dimensions
        c2, d2 = box2.center, box2.dimensions
        
        # Compute overlap on each axis
        overlaps = []
        for i in range(3):
            lo1 = c1[i] - d1[i]/2
            hi1 = c1[i] + d1[i]/2
            lo2 = c2[i] - d2[i]/2
            hi2 = c2[i] + d2[i]/2
            
            overlap = max(0, min(hi1, hi2) - max(lo1, lo2))
            overlaps.append(overlap)
        
        intersection = overlaps[0] * overlaps[1] * overlaps[2]
        union = box1.volume() + box2.volume() - intersection
        
        return intersection / max(union, 1e-8)

    # Test IoU computation
    box_a = BBox3D(center=(0, 0, 0), dimensions=(2, 2, 2), label="A")
    box_b = BBox3D(center=(1, 0, 0), dimensions=(2, 2, 2), label="B")
    box_c = BBox3D(center=(5, 5, 5), dimensions=(2, 2, 2), label="C")
    
    iou_ab = compute_axis_aligned_iou(box_a, box_b)
    iou_ac = compute_axis_aligned_iou(box_a, box_c)
    
    print(f"Box A: center={box_a.center}, size={box_a.dimensions}")
    print(f"Box B: center={box_b.center}, size={box_b.dimensions}")
    print(f"Box C: center={box_c.center}, size={box_c.dimensions}")
    print(f"\nIoU(A, B) = {iou_ab:.4f} (overlapping)")
    print(f"IoU(A, C) = {iou_ac:.4f} (no overlap)")
    
    # Demonstrate IoU range
    print(f"\nIoU interpretation:")
    print(f"  0.0: No overlap")
    print(f"  0.5: Moderate overlap")
    print(f"  1.0: Perfect match")

    # --- Sub-example 3: Detection evaluation metrics ---
    print("\n[4.3] Detection metrics — AP, precision, recall")
    print("-" * 60)

    def compute_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
        """Compute 3D detection metrics.
        
        Metrics:
        - True Positive (TP): prediction matches ground truth (IoU > threshold)
        - False Positive (FP): prediction doesn't match any GT
        - False Negative (FN): ground truth not matched by any prediction
        
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        AP = Average Precision (area under PR curve)
        """
        # Match predictions to ground truths using greedy IoU matching
        matched_gt = set()
        tp_list = []
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(ground_truths):
                if j not in matched_gt:
                    iou = compute_axis_aligned_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_list.append(1)  # True Positive
                matched_gt.add(best_gt_idx)
            else:
                tp_list.append(0)  # False Positive
        
        # Compute cumulative metrics
        tp_cumsum = []
        fp_cumsum = []
        tp_count = 0
        fp_count = 0
        
        for tp in tp_list:
            if tp == 1:
                tp_count += 1
            else:
                fp_count += 1
            tp_cumsum.append(tp_count)
            fp_cumsum.append(fp_count)
        
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(tp_cumsum, fp_cumsum)]
        recalls = [tp / max(len(ground_truths), 1) for tp in tp_cumsum]
        
        # Compute AP (area under PR curve)
        ap = 0
        for i in range(len(precisions)):
            if i == 0:
                ap += precisions[i] * recalls[i]
            else:
                ap += precisions[i] * (recalls[i] - recalls[i-1])
        
        return {
            "precision": precisions[-1] if precisions else 0,
            "recall": recalls[-1] if recalls else 0,
            "ap": ap,
            "tp": tp_count,
            "fp": fp_count,
            "fn": len(ground_truths) - len(matched_gt)
        }

    # Example detection results
    gt_boxes = [
        BBox3D(center=(0, 0, 0), dimensions=(2, 1, 1), label="car"),
        BBox3D(center=(5, 0, 0), dimensions=(2, 1, 1), label="car"),
        BBox3D(center=(10, 0, 0), dimensions=(2, 1, 1), label="car"),
    ]
    
    pred_boxes = [
        BBox3D(center=(0.1, 0.1, 0), dimensions=(2, 1, 1), label="car"),   # Match
        BBox3D(center=(5.2, -0.1, 0), dimensions=(2, 1, 1), label="car"),  # Match
        BBox3D(center=(15, 0, 0), dimensions=(2, 1, 1), label="car"),      # No match (FP)
    ]
    
    metrics = compute_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5)
    
    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"Predicted boxes: {len(pred_boxes)}")
    print(f"IoU threshold: 0.5")
    print(f"\nDetection results:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  AP@0.5:    {metrics['ap']:.3f}")

    # --- Sub-example 4: 3D NMS (Non-Maximum Suppression) ---
    print("\n[4.4] 3D Non-Maximum Suppression — removing duplicate detections")
    print("-" * 60)

    def nms_3d(boxes, scores, iou_threshold=0.3):
        """Apply Non-Maximum Suppression in 3D.
        
        Algorithm:
        1. Sort boxes by confidence score (descending)
        2. Select box with highest score
        3. Remove all boxes with IoU > threshold
        4. Repeat until no boxes remain
        """
        # Sort by score
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        keep = []
        suppressed = set()
        
        for i in indices:
            if i in suppressed:
                continue
            
            keep.append(i)
            
            # Suppress overlapping boxes
            for j in indices:
                if j in suppressed or j == i:
                    continue
                iou = compute_axis_aligned_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    suppressed.add(j)
        
        return keep

    # Example: multiple overlapping detections
    nms_boxes = [
        BBox3D(center=(0, 0, 0), dimensions=(2, 1, 1), label="car"),
        BBox3D(center=(0.2, 0.1, 0), dimensions=(2, 1, 1), label="car"),
        BBox3D(center=(0.1, -0.1, 0), dimensions=(2, 1, 1), label="car"),
        BBox3D(center=(5, 0, 0), dimensions=(2, 1, 1), label="car"),
        BBox3D(center=(5.1, 0.2, 0), dimensions=(2, 1, 1), label="car"),
    ]
    nms_scores = [0.95, 0.88, 0.82, 0.91, 0.75]
    
    print(f"Input: {len(nms_boxes)} boxes with scores: {[f'{s:.2f}' for s in nms_scores]}")
    print(f"IoU threshold: 0.3")
    
    keep_indices = nms_3d(nms_boxes, nms_scores, iou_threshold=0.3)
    
    print(f"\nAfter NMS: {len(keep_indices)} boxes kept")
    print(f"Suppressed: {len(nms_boxes) - len(keep_indices)} duplicates removed")
    print(f"\nKept detections:")
    for idx in keep_indices:
        box = nms_boxes[idx]
        score = nms_scores[idx]
        print(f"  Box {idx}: center={box.center}, score={score:.2f}")

    print("\n" + "=" * 70)
    print("3D detection: boxes → IoU → metrics → NMS → final detections")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    demo_point_cloud_representation()
    demo_depth_estimation()
    demo_point_cloud_processing()
    demo_3d_object_detection()
