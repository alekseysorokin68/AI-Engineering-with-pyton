"""
134 — Multimodal RAG: document parsing, image retrieval, cross-modal search

Темы:
  1. Multimodal Document Parsing (text + image extraction, layout-aware parsing)
  2. Image Embeddings (CLIP embeddings, visual features for retrieval)
  3. Cross-Modal Search (text-to-image, image-to-text retrieval)
  4. Multimodal Indexing (hybrid indexes, relevance fusion)

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
# Demo 1: Multimodal Document Parsing
# =============================================================================
def demo_multimodal_parsing():
    print("=" * 70)
    print("DEMO 1: Multimodal Document Parsing — text + image extraction")
    print("=" * 70)

    # --- Sub-example 1: Document structure extraction ---
    print("\n[1.1] Document structure — parsing mixed content documents")
    print("-" * 60)

    def parse_multimodal_document(raw_content, seed=42):
        """Parse a document containing text, images, and tables.
        
        Layout-aware parsing extracts:
        - Text blocks with metadata
        - Image regions with captions
        - Tables with structure
        - Reading order preservation
        """
        rng = random.Random(seed)
        
        # Simulate parsed document elements
        elements = [
            {
                "type": "heading",
                "content": "Project Report: AI Assistant",
                "level": 1,
                "page": 1,
                "position": (50, 50)
            },
            {
                "type": "text",
                "content": "This report describes the development of an AI assistant system. The system uses natural language processing and computer vision capabilities.",
                "page": 1,
                "position": (50, 100),
                "word_count": 23
            },
            {
                "type": "image",
                "caption": "Figure 1: System architecture diagram",
                "dimensions": (400, 300),
                "page": 1,
                "position": (100, 200),
                "hash": hashlib.md5(b"architecture_diagram").hexdigest()[:8]
            },
            {
                "type": "text",
                "content": "The architecture consists of three main components: input processing, inference engine, and response generation.",
                "page": 1,
                "position": (50, 520),
                "word_count": 16
            },
            {
                "type": "table",
                "caption": "Table 1: Performance metrics",
                "headers": ["Metric", "Value", "Unit"],
                "rows": [
                    ["Latency", "45", "ms"],
                    ["Accuracy", "94.2", "%"],
                    ["Memory", "2.1", "GB"]
                ],
                "page": 2,
                "position": (50, 100)
            },
            {
                "type": "image",
                "caption": "Figure 2: Performance over time",
                "dimensions": (500, 250),
                "page": 2,
                "position": (50, 300),
                "hash": hashlib.md5(b"performance_chart").hexdigest()[:8]
            },
        ]
        
        # Compute reading order
        for i, elem in enumerate(elements):
            elem["reading_order"] = i
        
        return elements

    doc_elements = parse_multimodal_document(None)
    
    print(f"Parsed document: {len(doc_elements)} elements")
    print(f"Element types: {collections.Counter(e['type'] for e in doc_elements)}")
    
    print("\nDocument structure:")
    for elem in doc_elements:
        if elem["type"] == "image":
            print(f"  [{elem['type']:6s}] {elem['caption']} ({elem['dimensions'][0]}x{elem['dimensions'][1]})")
        elif elem["type"] == "table":
            print(f"  [{elem['type']:6s}] {elem['caption']} ({len(elem['rows'])} rows)")
        else:
            preview = elem["content"][:50] + "..."
            print(f"  [{elem['type']:6s}] {preview}")

    # --- Sub-example 2: Content chunking for RAG ---
    print("\n[1.2] Content chunking — splitting documents for retrieval")
    print("-" * 60)

    def chunk_document(elements, chunk_size=100, overlap=20):
        """Chunk document elements for RAG indexing.
        
        Chunking strategies:
        - Fixed-size: split text into fixed word chunks
        - Semantic: split at sentence boundaries
        - Element-based: keep each element as a chunk
        """
        chunks = []
        chunk_id = 0
        
        for elem in elements:
            if elem["type"] == "text":
                words = elem["content"].split()
                
                # Split into chunks with overlap
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_words = words[start:end]
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "type": "text",
                        "content": " ".join(chunk_words),
                        "page": elem["page"],
                        "word_count": len(chunk_words),
                        "metadata": {
                            "start_pos": start,
                            "end_pos": end,
                            "has_overlap": start > 0
                        }
                    })
                    chunk_id += 1
                    start += chunk_size - overlap
            
            elif elem["type"] == "image":
                chunks.append({
                    "chunk_id": chunk_id,
                    "type": "image",
                    "content": elem["caption"],
                    "page": elem["page"],
                    "dimensions": elem["dimensions"],
                    "hash": elem["hash"]
                })
                chunk_id += 1
            
            elif elem["type"] == "table":
                # Serialize table to text
                table_text = elem["caption"] + ": "
                table_text += " | ".join(elem["headers"]) + ". "
                for row in elem["rows"]:
                    table_text += " | ".join(row) + ". "
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "type": "table",
                    "content": table_text.strip(),
                    "page": elem["page"],
                    "headers": elem["headers"],
                    "num_rows": len(elem["rows"])
                })
                chunk_id += 1
        
        return chunks

    chunks = chunk_document(doc_elements, chunk_size=20, overlap=5)
    
    print(f"Document chunked into {len(chunks)} chunks")
    print(f"Chunk types: {collections.Counter(c['type'] for c in chunks)}")
    
    print("\nSample chunks:")
    for chunk in chunks[:4]:
        preview = chunk['content'][:60] + "..."
        print(f"  Chunk {chunk['chunk_id']}: [{chunk['type']:6s}] {preview}")

    # --- Sub-example 3: Image-text association ---
    print("\n[1.3] Image-text association — linking visuals with context")
    print("-" * 60)

    def associate_image_text(elements, context_window=2):
        """Associate images with nearby text for multimodal retrieval.
        
        Strategy: For each image, collect text from surrounding elements.
        """
        associations = []
        
        for i, elem in enumerate(elements):
            if elem["type"] == "image":
                # Collect context from nearby elements
                context_texts = []
                for j in range(max(0, i - context_window), min(len(elements), i + context_window + 1)):
                    if elements[j]["type"] == "text":
                        context_texts.append(elements[j]["content"])
                
                # Create combined representation
                combined_text = elem["caption"] + " " + " ".join(context_texts)
                
                associations.append({
                    "image_hash": elem["hash"],
                    "caption": elem["caption"],
                    "context": context_texts,
                    "combined_text": combined_text[:200],
                    "context_length": len(" ".join(context_texts).split())
                })
        
        return associations

    associations = associate_image_text(doc_elements)
    
    print(f"Image-text associations: {len(associations)}")
    for assoc in associations:
        print(f"\n  Image: {assoc['caption']}")
        print(f"  Context words: {assoc['context_length']}")
        print(f"  Combined text preview: {assoc['combined_text'][:80]}...")

    # --- Sub-example 4: Metadata extraction ---
    print("\n[1.4] Metadata extraction — enriching document content")
    print("-" * 60)

    def extract_metadata(elements):
        """Extract rich metadata from document elements."""
        metadata = {
            "total_elements": len(elements),
            "pages": set(),
            "total_words": 0,
            "total_images": 0,
            "total_tables": 0,
            "elements_by_type": collections.defaultdict(int),
            "page_distribution": collections.defaultdict(int)
        }
        
        for elem in elements:
            metadata["pages"].add(elem["page"])
            metadata["elements_by_type"][elem["type"]] += 1
            metadata["page_distribution"][elem["page"]] += 1
            
            if elem["type"] == "text":
                metadata["total_words"] += elem.get("word_count", 0)
            elif elem["type"] == "image":
                metadata["total_images"] += 1
            elif elem["type"] == "table":
                metadata["total_tables"] += 1
        
        metadata["pages"] = sorted(metadata["pages"])
        metadata["avg_elements_per_page"] = len(elements) / max(len(metadata["pages"]), 1)
        
        return metadata

    metadata = extract_metadata(doc_elements)
    
    print("Document metadata:")
    print(f"  Total elements: {metadata['total_elements']}")
    print(f"  Pages: {metadata['pages']}")
    print(f"  Total words: {metadata['total_words']}")
    print(f"  Images: {metadata['total_images']}")
    print(f"  Tables: {metadata['total_tables']}")
    print(f"  Avg elements/page: {metadata['avg_elements_per_page']:.1f}")
    print(f"  Elements by type: {dict(metadata['elements_by_type'])}")

    print("\n" + "=" * 70)
    print("Multimodal parsing: extract → chunk → associate → enrich metadata")
    print("=" * 70)


# =============================================================================
# Demo 2: Image Embeddings
# =============================================================================
def demo_image_embeddings():
    print("\n" + "=" * 70)
    print("DEMO 2: Image Embeddings — CLIP embeddings, visual features")
    print("=" * 70)

    # --- Sub-example 1: Simulated CLIP-like embedding ---
    print("\n[2.1] Image embedding generation — CLIP-style feature extraction")
    print("-" * 60)

    def generate_image_embedding(image_id, seed=42):
        """Simulate CLIP-like image embedding.
        
        CLIP embeddings:
        - Dimension: 512 or 768
        - Normalized to unit hypersphere
        - Trained with contrastive loss on image-text pairs
        """
        rng = random.Random(seed + hash(image_id) % 1000)
        dim = 512
        
        # Generate random embedding
        embedding = [rng.gauss(0, 1) for _ in range(dim)]
        
        # Normalize to unit vector
        norm = math.sqrt(sum(x**2 for x in embedding))
        embedding = [x / norm for x in embedding]
        
        return embedding

    def cosine_similarity(v1, v2):
        """Compute cosine similarity between two vectors.
        
        cos(a, b) = (a · b) / (||a|| × ||b||)
        """
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a**2 for a in v1))
        norm2 = math.sqrt(sum(b**2 for b in v2))
        return dot / (norm1 * norm2)

    # Generate embeddings for sample images
    image_ids = ["photo_001", "diagram_002", "chart_003", "photo_004"]
    embeddings = {img_id: generate_image_embedding(img_id) for img_id in image_ids}
    
    print(f"Generated embeddings for {len(image_ids)} images")
    print(f"Embedding dimension: {len(embeddings[image_ids[0]])}")
    
    # Show embedding statistics
    for img_id, emb in embeddings.items():
        norm = math.sqrt(sum(x**2 for x in emb))
        mean_val = sum(emb) / len(emb)
        std_val = math.sqrt(sum((x - mean_val)**2 for x in emb) / len(emb))
        print(f"  {img_id}: norm={norm:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")

    # --- Sub-example 2: Visual similarity computation ---
    print("\n[2.2] Visual similarity — finding similar images")
    print("-" * 60)

    def compute_similarity_matrix(embeddings_dict):
        """Compute pairwise similarity matrix."""
        ids = list(embeddings_dict.keys())
        n = len(ids)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                matrix[i][j] = cosine_similarity(
                    embeddings_dict[ids[i]], 
                    embeddings_dict[ids[j]]
                )
        
        return ids, matrix

    ids, sim_matrix = compute_similarity_matrix(embeddings)
    
    print("Pairwise cosine similarity matrix:")
    print(f"{'':12s}", end="")
    for img_id in ids:
        print(f"{img_id:12s}", end="")
    print()
    
    for i, img_id in enumerate(ids):
        print(f"{img_id:12s}", end="")
        for j in range(len(ids)):
            print(f"{sim_matrix[i][j]:12.4f}", end="")
        print()

    # Find most similar pair (excluding self)
    max_sim = -1
    max_pair = None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim_matrix[i][j] > max_sim:
                max_sim = sim_matrix[i][j]
                max_pair = (ids[i], ids[j])
    
    print(f"\nMost similar pair: {max_pair[0]} ↔ {max_pair[1]} (similarity={max_sim:.4f})")

    # --- Sub-example 3: Embedding compression ---
    print("\n[2.3] Embedding compression — reducing storage requirements")
    print("-" * 60)

    def compress_embedding(embedding, compression_ratio=0.5):
        """Compress embedding using dimensionality reduction.
        
        Simple PCA-like compression:
        1. Keep top-k dimensions (by variance)
        2. Or use random projection
        """
        dim = len(embedding)
        compressed_dim = int(dim * compression_ratio)
        
        # Random projection (simplified)
        rng = random.Random(42)
        projection_matrix = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(compressed_dim)]
        
        # Project
        compressed = []
        for row in projection_matrix:
            val = sum(a * b for a, b in zip(embedding, row))
            compressed.append(val)
        
        # Normalize
        norm = math.sqrt(sum(x**2 for x in compressed))
        if norm > 0:
            compressed = [x / norm for x in compressed]
        
        return compressed

    original_emb = embeddings["photo_001"]
    compressed_emb = compress_embedding(original_emb, compression_ratio=0.25)
    
    print(f"Original embedding: {len(original_emb)} dimensions")
    print(f"Compressed embedding: {len(compressed_emb)} dimensions")
    print(f"Compression ratio: {len(compressed_emb)/len(original_emb)*100:.1f}%")
    
    # Compute storage savings
    original_bytes = len(original_emb) * 4  # float32
    compressed_bytes = len(compressed_emb) * 4
    print(f"Original storage: {original_bytes} bytes")
    print(f"Compressed storage: {compressed_bytes} bytes")
    print(f"Space savings: {(1 - compressed_bytes/original_bytes)*100:.1f}%")
    
    # Check similarity preservation
    sim_original = cosine_similarity(original_emb, embeddings["photo_004"])
    # For compressed, we need to compress both
    compressed_004 = compress_embedding(embeddings["photo_004"], compression_ratio=0.25)
    sim_compressed = cosine_similarity(compressed_emb, compressed_004)
    print(f"Similarity preservation: {sim_compressed/sim_original*100:.1f}%")

    # --- Sub-example 4: Multi-scale feature extraction ---
    print("\n[2.4] Multi-scale features — capturing different visual aspects")
    print("-" * 60)

    def extract_multiscale_features(image_embedding, num_scales=4):
        """Extract multi-scale features from image embedding.
        
        Multi-scale analysis:
        - Global: entire image embedding
        - Regional: split embedding into quadrants
        - Local: fine-grained patches
        - Texture: frequency domain features
        """
        dim = len(image_embedding)
        features = {}
        
        # Global features
        features["global"] = {
            "mean": sum(image_embedding) / dim,
            "std": math.sqrt(sum((x - sum(image_embedding)/dim)**2 for x in image_embedding) / dim),
            "l2_norm": math.sqrt(sum(x**2 for x in image_embedding))
        }
        
        # Regional features (split into quadrants)
        quadrant_size = dim // 4
        for i, name in enumerate(["top_left", "top_right", "bottom_left", "bottom_right"]):
            start = i * quadrant_size
            end = start + quadrant_size
            quadrant = image_embedding[start:end]
            features[name] = {
                "mean": sum(quadrant) / len(quadrant),
                "energy": sum(x**2 for x in quadrant)
            }
        
        # Texture features (simplified frequency analysis)
        features["texture"] = {
            "high_freq_energy": sum(image_embedding[i]**2 for i in range(0, dim, 2)),
            "low_freq_energy": sum(image_embedding[i]**2 for i in range(1, dim, 2)),
            "texture_ratio": sum(image_embedding[i]**2 for i in range(0, dim, 2)) / 
                            max(sum(image_embedding[i]**2 for i in range(1, dim, 2)), 1e-8)
        }
        
        return features

    features = extract_multiscale_features(embeddings["diagram_002"])
    
    print("Multi-scale features for 'diagram_002':")
    print(f"\n  Global features:")
    for key, val in features["global"].items():
        print(f"    {key}: {val:.4f}")
    
    print(f"\n  Regional features:")
    for name in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        print(f"    {name}: mean={features[name]['mean']:.4f}, energy={features[name]['energy']:.4f}")
    
    print(f"\n  Texture features:")
    for key, val in features["texture"].items():
        print(f"    {key}: {val:.4f}")

    print("\n" + "=" * 70)
    print("Image embeddings: CLIP features → similarity → compression → multi-scale")
    print("=" * 70)


# =============================================================================
# Demo 3: Cross-Modal Search
# =============================================================================
def demo_cross_modal_search():
    print("\n" + "=" * 70)
    print("DEMO 3: Cross-Modal Search — text-to-image, image-to-text retrieval")
    print("=" * 70)

    def cosine_similarity(v1, v2):
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a**2 for a in v1))
        norm2 = math.sqrt(sum(b**2 for b in v2))
        return dot / (norm1 * norm2)

    # --- Sub-example 1: Text embedding generation ---
    print("\n[3.1] Text embedding — encoding queries for cross-modal search")
    print("-" * 60)

    def generate_text_embedding(text, seed=42):
        """Simulate CLIP text embedding.
        
        CLIP text encoder:
        - Tokenizes input text
        - Processes through transformer
        - Outputs 512-dim embedding
        - Normalized to unit hypersphere
        """
        # Simple hash-based embedding (simulated)
        rng = random.Random(seed + hash(text) % 1000)
        dim = 512
        
        # Generate embedding influenced by text content
        embedding = []
        for i in range(dim):
            # Add some structure based on character values
            char_influence = sum(ord(c) for c in text[:10]) / 1000
            embedding.append(rng.gauss(char_influence, 1))
        
        # Normalize
        norm = math.sqrt(sum(x**2 for x in embedding))
        embedding = [x / norm for x in embedding]
        
        return embedding

    # Sample queries
    queries = [
        "a photo of a cat",
        "a diagram showing system architecture",
        "a chart with performance metrics",
        "a landscape photo of mountains"
    ]
    
    text_embeddings = {q: generate_text_embedding(q) for q in queries}
    
    print(f"Generated embeddings for {len(queries)} text queries")
    print(f"Embedding dimension: {len(text_embeddings[queries[0]])}")
    
    for q in queries:
        emb = text_embeddings[q]
        print(f"  '{q[:40]}...' → norm={math.sqrt(sum(x**2 for x in emb)):.4f}")

    # Generate image embeddings for cross-modal search
    def generate_image_embedding(image_id, seed=42):
        """Simulate CLIP image embedding."""
        rng = random.Random(seed + hash(image_id) % 1000)
        dim = 512
        embedding = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x**2 for x in embedding))
        embedding = [x / norm for x in embedding]
        return embedding

    image_ids = ["photo_001", "diagram_002", "chart_003", "photo_004"]
    image_embeddings = {img_id: generate_image_embedding(img_id) for img_id in image_ids}

    # --- Sub-example 2: Text-to-image retrieval ---
    print("\n[3.2] Text-to-image retrieval — finding images matching text query")
    print("-" * 60)

    def text_to_image_search(query_embedding, image_embeddings, top_k=3):
        """Search images by text query.
        
        Cross-modal retrieval:
        1. Encode text query with text encoder
        2. Compute similarity with all image embeddings
        3. Return top-k most similar images
        """
        similarities = []
        for img_id, img_emb in image_embeddings.items():
            sim = cosine_similarity(query_embedding, img_emb)
            similarities.append((img_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    # Use previously generated image embeddings
    query = "a photo of a cat"
    query_emb = text_embeddings[query]
    
    results = text_to_image_search(query_emb, image_embeddings, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Top {len(results)} matches:")
    for rank, (img_id, sim) in enumerate(results, 1):
        print(f"  {rank}. {img_id} (similarity={sim:.4f})")

    # --- Sub-example 3: Image-to-text retrieval ---
    print("\n[3.3] Image-to-text retrieval — finding text describing an image")
    print("-" * 60)

    def image_to_text_search(image_embedding, text_embeddings, top_k=3):
        """Search text captions by image.
        
        Reverse cross-modal retrieval:
        1. Encode image with image encoder
        2. Compute similarity with all text embeddings
        3. Return top-k most similar text queries
        """
        similarities = []
        for text, text_emb in text_embeddings.items():
            sim = cosine_similarity(image_embedding, text_emb)
            similarities.append((text, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # Search for text matching an image
    target_image = "diagram_002"
    image_emb = image_embeddings[target_image]
    
    text_results = image_to_text_search(image_emb, text_embeddings, top_k=3)
    
    print(f"Query image: '{target_image}'")
    print(f"Top {len(text_results)} matching texts:")
    for rank, (text, sim) in enumerate(text_results, 1):
        print(f"  {rank}. '{text[:50]}...' (similarity={sim:.4f})")

    # --- Sub-example 4: Multi-query fusion ---
    print("\n[3.4] Multi-query fusion — combining multiple search queries")
    print("-" * 60)

    def multi_query_fusion(queries, embeddings_dict, top_k=3, fusion_method="average"):
        """Fuse results from multiple queries.
        
        Fusion methods:
        - average: average similarity scores
        - max: take maximum similarity
        - weighted: apply query-specific weights
        """
        # Collect similarities from all queries
        all_similarities = collections.defaultdict(list)
        
        for query in queries:
            query_emb = generate_text_embedding(query)
            for item_id, item_emb in embeddings_dict.items():
                sim = cosine_similarity(query_emb, item_emb)
                all_similarities[item_id].append(sim)
        
        # Apply fusion
        fused_scores = {}
        for item_id, sims in all_similarities.items():
            if fusion_method == "average":
                fused_scores[item_id] = sum(sims) / len(sims)
            elif fusion_method == "max":
                fused_scores[item_id] = max(sims)
            elif fusion_method == "weighted":
                # Weight later queries more
                weights = [1.0 / (i + 1) for i in range(len(sims))]
                total_weight = sum(weights)
                fused_scores[item_id] = sum(w * s for w, s in zip(weights, sims)) / total_weight
        
        # Sort and return top-k
        results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # Multi-query search
    multi_queries = [
        "a technical diagram",
        "system architecture",
        "flowchart or blueprint"
    ]
    
    fusion_results = multi_query_fusion(multi_queries, image_embeddings, top_k=3, fusion_method="average")
    
    print(f"Queries: {multi_queries}")
    print(f"Fusion method: average")
    print(f"\nTop {len(fusion_results)} results:")
    for rank, (img_id, score) in enumerate(fusion_results, 1):
        print(f"  {rank}. {img_id} (fused score={score:.4f})")

    print("\n" + "=" * 70)
    print("Cross-modal search: text→image, image→text, multi-query fusion")
    print("=" * 70)


# =============================================================================
# Demo 4: Multimodal Indexing
# =============================================================================
def demo_multimodal_indexing():
    print("\n" + "=" * 70)
    print("DEMO 4: Multimodal Indexing — hybrid indexes, relevance fusion")
    print("=" * 70)

    def generate_text_embedding(text, seed=42):
        """Simulate CLIP text embedding."""
        rng = random.Random(seed + hash(text) % 1000)
        dim = 512
        embedding = []
        for i in range(dim):
            char_influence = sum(ord(c) for c in text[:10]) / 1000
            embedding.append(rng.gauss(char_influence, 1))
        norm = math.sqrt(sum(x**2 for x in embedding))
        embedding = [x / norm for x in embedding]
        return embedding

    def generate_image_embedding(image_id, seed=42):
        """Simulate CLIP image embedding."""
        rng = random.Random(seed + hash(image_id) % 1000)
        dim = 512
        embedding = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x**2 for x in embedding))
        embedding = [x / norm for x in embedding]
        return embedding

    def cosine_similarity(v1, v2):
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a**2 for a in v1))
        norm2 = math.sqrt(sum(b**2 for b in v2))
        return dot / (norm1 * norm2)

    # --- Sub-example 1: Hybrid index structure ---
    print("\n[4.1] Hybrid index — combining text and image indexes")
    print("-" * 60)

    class HybridIndex:
        """Hybrid index supporting text, image, and cross-modal search."""
        
        def __init__(self):
            self.text_index = {}  # text_id -> {text, embedding, metadata}
            self.image_index = {}  # image_id -> {embedding, metadata}
            self.cross_modal_links = {}  # links between text and images
        
        def add_text(self, text_id, text, embedding, metadata=None):
            """Add text document to index."""
            self.text_index[text_id] = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {}
            }
        
        def add_image(self, image_id, embedding, metadata=None):
            """Add image to index."""
            self.image_index[image_id] = {
                "embedding": embedding,
                "metadata": metadata or {}
            }
        
        def link(self, text_id, image_id, relevance_score=1.0):
            """Create link between text and image."""
            if text_id not in self.cross_modal_links:
                self.cross_modal_links[text_id] = []
            self.cross_modal_links[text_id].append({
                "image_id": image_id,
                "score": relevance_score
            })
        
        def stats(self):
            """Get index statistics."""
            return {
                "text_count": len(self.text_index),
                "image_count": len(self.image_index),
                "link_count": sum(len(v) for v in self.cross_modal_links.values())
            }

    # Build hybrid index
    index = HybridIndex()
    
    # Add text documents
    text_docs = [
        ("doc1", "Machine learning algorithms for image classification"),
        ("doc2", "Natural language processing techniques"),
        ("doc3", "Computer vision applications in healthcare"),
    ]
    
    for text_id, text in text_docs:
        emb = generate_text_embedding(text)
        index.add_text(text_id, text, emb, {"type": "document"})
    
    # Add images
    for img_id in ["img1", "img2", "img3"]:
        index.add_image(img_id, generate_image_embedding(img_id))
    
    # Create cross-modal links
    index.link("doc1", "img1", 0.9)
    index.link("doc1", "img2", 0.7)
    index.link("doc3", "img3", 0.85)
    
    stats = index.stats()
    print("Hybrid index created:")
    print(f"  Text documents: {stats['text_count']}")
    print(f"  Images: {stats['image_count']}")
    print(f"  Cross-modal links: {stats['link_count']}")

    # --- Sub-example 2: Relevance scoring ---
    print("\n[4.2] Relevance scoring — ranking results across modalities")
    print("-" * 60)

    def compute_relevance_score(query_embedding, item_embedding, item_metadata, weights=None):
        """Compute weighted relevance score.
        
        Score = w_visual * visual_sim + w_text * text_sim + w_meta * meta_score
        
        where:
        - visual_sim: cosine similarity of embeddings
        - text_sim: text matching score (if available)
        - meta_score: metadata-based score (freshness, popularity, etc.)
        """
        if weights is None:
            weights = {"visual": 0.6, "text": 0.3, "meta": 0.1}
        
        # Visual similarity
        visual_sim = cosine_similarity(query_embedding, item_embedding)
        
        # Text similarity (simplified)
        text_sim = random.uniform(0.5, 0.9)  # Simulated
        
        # Metadata score
        meta_score = item_metadata.get("relevance", 0.5)
        
        # Weighted combination
        total_score = (
            weights["visual"] * visual_sim +
            weights["text"] * text_sim +
            weights["meta"] * meta_score
        )
        
        return {
            "total": total_score,
            "visual": visual_sim,
            "text": text_sim,
            "meta": meta_score
        }

    # Score items for a query
    query_text = "image classification algorithms"
    query_emb = generate_text_embedding(query_text)
    
    print(f"Query: '{query_text}'")
    print(f"Weights: visual=0.6, text=0.3, meta=0.1")
    print("\nRelevance scores:")
    
    for text_id, text_data in index.text_index.items():
        scores = compute_relevance_score(query_emb, text_data["embedding"], text_data["metadata"])
        print(f"  {text_id}: total={scores['total']:.4f} "
              f"(visual={scores['visual']:.3f}, text={scores['text']:.3f}, meta={scores['meta']:.3f})")

    # --- Sub-example 3: Index optimization ---
    print("\n[4.3] Index optimization — improving search efficiency")
    print("-" * 60)

    def optimize_embedding_storage(embeddings_dict, target_dim=64):
        """Optimize embedding storage using quantization.
        
        Scalar quantization:
        1. Find min/max values
        2. Scale to integer range [0, 255]
        3. Store as uint8 instead of float32
        
        Storage reduction: 4x (float32 → uint8)
        """
        # Collect all values
        all_values = []
        for emb in embeddings_dict.values():
            all_values.extend(emb)
        
        min_val = min(all_values)
        max_val = max(all_values)
        val_range = max_val - min_val
        
        # Quantize each embedding
        quantized = {}
        for emb_id, emb in embeddings_dict.items():
            q_emb = []
            for val in emb:
                # Scale to [0, 255]
                q_val = int((val - min_val) / val_range * 255)
                q_val = max(0, min(255, q_val))
                q_emb.append(q_val)
            quantized[emb_id] = q_emb
        
        # Compute size reduction
        original_size = len(embeddings_dict) * len(list(embeddings_dict.values())[0]) * 4  # float32
        quantized_size = len(quantized) * len(list(quantized.values())[0])  # uint8
        
        return quantized, {
            "original_size": original_size,
            "quantized_size": quantized_size,
            "compression_ratio": original_size / quantized_size,
            "min_val": min_val,
            "max_val": max_val
        }

    # Create sample embeddings for optimization demo
    sample_embeddings = {f"doc_{i}": generate_text_embedding(f"document {i}") for i in range(10)}
    quantized_index, opt_stats = optimize_embedding_storage(sample_embeddings)
    
    print(f"Embedding quantization (float32 → uint8):")
    print(f"  Original size: {opt_stats['original_size']} bytes")
    print(f"  Quantized size: {opt_stats['quantized_size']} bytes")
    print(f"  Compression ratio: {opt_stats['compression_ratio']:.1f}x")
    print(f"  Value range: [{opt_stats['min_val']:.3f}, {opt_stats['max_val']:.3f}]")

    # --- Sub-example 4: Distributed indexing concepts ---
    print("\n[4.4] Distributed indexing — sharding and replication")
    print("-" * 60)

    def simulate_distributed_index(num_shards=4, num_replicas=2):
        """Simulate distributed index architecture.
        
        Sharding strategies:
        - Hash-based: shard_id = hash(item_id) % num_shards
        - Range-based: partition by ID ranges
        - Geographic: partition by region
        
        Replication: each shard has replicas for fault tolerance
        """
        # Simulate items
        items = [f"item_{i:04d}" for i in range(100)]
        
        # Hash-based sharding
        shards = collections.defaultdict(list)
        for item in items:
            shard_id = hash(item) % num_shards
            shards[shard_id].append(item)
        
        # Compute shard statistics
        shard_stats = {}
        for shard_id, shard_items in shards.items():
            shard_stats[shard_id] = {
                "item_count": len(shard_items),
                "replicas": num_replicas,
                "storage_bytes": len(shard_items) * 512  # Assume 512 bytes per item
            }
        
        # Total storage with replication
        total_items = sum(s["item_count"] for s in shard_stats.values())
        total_storage = sum(s["storage_bytes"] * s["replicas"] for s in shard_stats.values())
        
        return {
            "num_shards": num_shards,
            "num_replicas": num_replicas,
            "items_per_shard": shard_stats,
            "total_items": total_items,
            "total_storage": total_storage
        }

    dist_index = simulate_distributed_index(num_shards=4, num_replicas=2)
    
    print(f"Distributed index architecture:")
    print(f"  Shards: {dist_index['num_shards']}")
    print(f"  Replicas per shard: {dist_index['num_replicas']}")
    print(f"  Total items: {dist_index['total_items']}")
    
    print(f"\nShard distribution:")
    for shard_id, stats in dist_index['items_per_shard'].items():
        print(f"  Shard {shard_id}: {stats['item_count']} items, {stats['storage_bytes']*stats['replicas']} bytes (with replication)")
    
    print(f"\nTotal storage: {dist_index['total_storage']} bytes")
    print(f"Avg items/shard: {dist_index['total_items']/dist_index['num_shards']:.1f}")

    print("\n" + "=" * 70)
    print("Multimodal indexing: hybrid structure → relevance → optimization → distribution")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    demo_multimodal_parsing()
    demo_image_embeddings()
    demo_cross_modal_search()
    demo_multimodal_indexing()
