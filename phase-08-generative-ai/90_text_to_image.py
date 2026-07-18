"""
Phase 08: Generative AI — Text-to-Image Generation Basics
==========================================================
Core concepts:
1. CLIP — text-image alignment
2. Image generation pipeline
3. Prompt engineering fundamentals
4. Comparison of generation methods

All implementations are self-contained (no external ML libraries).
"""

import random
import math

random.seed(42)

# =============================================================================
# Part 1: CLIP — Text-Image Alignment (Simplified)
# =============================================================================

class SimpleTokenizer:
    """Простой токенизатор для текста."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        
    def tokenize(self, text):
        """Токенизация текста в числовые ID."""
        words = text.lower().split()
        ids = []
        for word in words:
            # Простой хеш для маппинга слов в ID
            word_id = hash(word) % self.vocab_size
            ids.append(word_id)
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
        return ids


class SimpleTextEncoder:
    """Упрощённый текстовый энкодер (имитация CLIP text encoder)."""
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        # Случайные веса эмбеддингов
        random.seed(42)
        self.embeddings = [[random.gauss(0, 0.1) for _ in range(embedding_dim)] 
                          for _ in range(1000)]
        
    def encode(self, token_ids):
        """Кодирование токенов в вектор фиксированной размерности."""
        if not token_ids:
            return [0.0] * self.embedding_dim
            
        # Усреднение эмбеддингов токенов
        result = [0.0] * self.embedding_dim
        for token_id in token_ids:
            emb = self.embeddings[token_id % len(self.embeddings)]
            for i in range(self.embedding_dim):
                result[i] += emb[i]
                
        # Нормализация
        norm = math.sqrt(sum(x*x for x in result))
        if norm > 0:
            result = [x / norm for x in result]
            
        return result


class SimpleImageEncoder:
    """Упрощённый энкодер изображений (имитация CLIP image encoder)."""
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        random.seed(123)
        self.weights = [[random.gauss(0, 0.05) for _ in range(embedding_dim)]
                       for _ in range(256)]
        
    def encode(self, image_features):
        """Кодирование признаков изображения в вектор."""
        if not image_features:
            return [0.0] * self.embedding_dim
            
        result = [0.0] * self.embedding_dim
        for feature_idx in image_features:
            weight = self.weights[feature_idx % len(self.weights)]
            for i in range(self.embedding_dim):
                result[i] += weight[i]
                
        # Нормализация
        norm = math.sqrt(sum(x*x for x in result))
        if norm > 0:
            result = [x / norm for x in result]
            
        return result


class CLIPModel:
    """Упрощённая модель CLIP для выравнивания текст-изображение."""
    
    def __init__(self):
        self.text_encoder = SimpleTextEncoder()
        self.image_encoder = SimpleImageEncoder()
        
    def cosine_similarity(self, vec1, vec2):
        """Вычисление косинусного сходства между векторами."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a*a for a in vec1))
        norm2 = math.sqrt(sum(b*b for b in vec2))
        return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
        
    def get_text_embedding(self, text):
        """Получение эмбеддинга текста."""
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        return self.text_encoder.encode(tokens)
        
    def get_image_embedding(self, image_features):
        """Получение эмбеддинга изображения."""
        return self.image_encoder.encode(image_features)
        
    def compute_similarity(self, text, image_features):
        """Вычисление сходства между текстом и изображением."""
        text_emb = self.get_text_embedding(text)
        image_emb = self.get_image_embedding(image_features)
        return self.cosine_similarity(text_emb, image_emb)


def demo_clip_alignment():
    """Демонстрация CLIP — выравнивание текст-изображение."""
    print("\n" + "="*70)
    print("DEMO 1: CLIP — Text-Image Alignment")
    print("="*70)
    
    model = CLIPModel()
    
    # Тестовые тексты и "изображения" (псевдо-признаки)
    texts = [
        "a photo of a cat",
        "a photo of a dog", 
        "a landscape with mountains",
        "a red sports car",
        "a blue ocean sunset"
    ]
    
    # Симуляция изображений как наборов признаков (индексы пикселей/областей)
    images = {
        "cat_image": [12, 45, 67, 89, 123, 200],      # Кошка
        "dog_image": [15, 48, 70, 92, 125, 203],      # Собака
        "mountains": [30, 60, 90, 150, 180, 220],     # Горы
        "sports_car": [5, 25, 50, 100, 150, 250],     # Спорткар
        "ocean_sunset": [10, 35, 75, 120, 170, 240]   # Закат
    }
    
    print("\nТестовые изображения:")
    print("-" * 40)
    for img_name in images:
        print(f"  • {img_name}")
    
    print("\nРезультаты CLIP similarity:")
    print("-" * 40)
    
    # Вычисляем сходство для всех комбинаций
    results = []
    for text in texts:
        similarities = {}
        for img_name, img_features in images.items():
            sim = model.compute_similarity(text, img_features)
            similarities[img_name] = sim
        results.append((text, similarities))
        
    for text, sims in results:
        print(f"\nText: \"{text}\"")
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        for img_name, sim in sorted_sims[:3]:
            print(f"  → {img_name}: {sim:.4f}")
            
    # Показать лучшие пары
    print("\n" + "-"*40)
    print("Лучшие текст-изображение пары:")
    print("-"*40)
    
    all_pairs = []
    for text, sims in results:
        for img_name, sim in sims.items():
            all_pairs.append((text, img_name, sim))
            
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    for text, img_name, sim in all_pairs[:5]:
        print(f"  {sim:.4f} | \"{text}\" ↔ {img_name}")


# =============================================================================
# Part 2: Image Generation Pipeline (Simplified Diffusion)
# =============================================================================

class SimpleDiffusionProcess:
    """Упрощённый процесс диффузии для генерации изображений."""
    
    def __init__(self, image_size=32, num_timesteps=50):
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        random.seed(42)
        
    def add_noise(self, image, noise_level):
        """Добавление шума к изображению."""
        noisy = []
        for pixel in image:
            noise = random.gauss(0, noise_level)
            noisy.append(max(0, min(255, pixel + noise)))
        return noisy
        
    def denoise_step(self, noisy_image, timestep, text_condition=None):
        """Один шаг денойзинга (упрощённый)."""
        denoised = []
        # Коэффициент шума зависит от timestep
        noise_factor = 1.0 - (timestep / self.num_timesteps)
        
        for pixel in noisy_image:
            # Упрощённый шаг диффузии
            if text_condition and "bright" in text_condition.lower():
                target = pixel * 0.3 + 200 * 0.7
            elif text_condition and "dark" in text_condition.lower():
                target = pixel * 0.3 + 50 * 0.7
            else:
                target = pixel * 0.5 + 128 * 0.5
                
            denoised.append(max(0, min(255, int(target * noise_factor + pixel * (1-noise_factor)))))
        return denoised
        
    def generate_from_text(self, text_prompt, seed=None):
        """Генерация изображения из текста."""
        if seed is not None:
            random.seed(seed)
            
        # Начинаем с чистого шума
        image = [random.randint(0, 255) for _ in range(self.image_size * self.image_size)]
        
        print(f"\n  Генерация для: \"{text_prompt}\"")
        print(f"  Размер изображения: {self.image_size}x{self.image_size}")
        print(f"  Шагов диффузии: {self.num_timesteps}")
        
        # Процесс диффузии (от шума к изображению)
        for t in range(self.num_timesteps - 1, -1, -1):
            image = self.denoise_step(image, t, text_prompt)
            
        # Статистика сгенерированного изображения
        avg_intensity = sum(image) / len(image)
        min_val = min(image)
        max_val = max(image)
        
        return {
            "image": image,
            "avg_intensity": avg_intensity,
            "min": min_val,
            "max": max_val,
            "text_prompt": text_prompt
        }


class TextPromptEncoder:
    """Кодирование текстовых промптов для условия генерации."""
    
    def __init__(self):
        self.style_keywords = {
            "photorealistic": ["photo", "realistic", "real", "photograph"],
            "artistic": ["painting", "artistic", "art", "oil", "canvas"],
            "cartoon": ["cartoon", "anime", "illustration", "drawing"],
            "sketch": ["sketch", "pencil", "line art", "doodle"]
        }
        
        self.subject_keywords = {
            "animal": ["cat", "dog", "bird", "horse", "fish"],
            "landscape": ["mountain", "ocean", "forest", "desert", "sky"],
            "person": ["person", "human", "man", "woman", "child"],
            "object": ["car", "house", "tree", "flower", "building"]
        }
        
    def encode_prompt(self, prompt):
        """Кодирование промпта в структурированное представление."""
        words = prompt.lower().split()
        
        # Определение стиля
        style = "unknown"
        for style_name, keywords in self.style_keywords.items():
            if any(kw in words for kw in keywords):
                style = style_name
                break
                
        # Определение субъекта
        subject = "unknown"
        for subject_name, keywords in self.subject_keywords.items():
            if any(kw in words for kw in keywords):
                subject = subject_name
                break
                
        # Извлечение модификаторов (цвета, настроение и т.д.)
        modifiers = []
        color_words = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
        mood_words = ["happy", "sad", "dramatic", "peaceful", "vibrant", "muted"]
        
        for word in words:
            if word in color_words:
                modifiers.append(f"color:{word}")
            if word in mood_words:
                modifiers.append(f"mood:{word}")
                
        return {
            "style": style,
            "subject": subject,
            "modifiers": modifiers,
            "word_count": len(words)
        }


def demo_generation_pipeline():
    """Демонстрация пайплайна генерации из шума + текст."""
    print("\n" + "="*70)
    print("DEMO 2: Image Generation Pipeline (Simplified Diffusion)")
    print("="*70)
    
    # Создаём процесс диффузии
    diffusion = SimpleDiffusionProcess(image_size=16, num_timesteps=30)
    
    # Тестовые промпты
    prompts = [
        "a bright sunny day in a green forest",
        "a dark night sky with stars",
        "a colorful abstract painting",
        "a realistic photo of a mountain landscape"
    ]
    
    generated_images = []
    for i, prompt in enumerate(prompts):
        print(f"\n--- Изображение {i+1} ---")
        result = diffusion.generate_from_text(prompt, seed=42 + i)
        generated_images.append(result)
        
        print(f"  Средняя яркость: {result['avg_intensity']:.1f}")
        print(f"  Диапазон пикселей: [{result['min']:.0f}, {result['max']:.0f}]")
        
    # Анализ результатов
    print("\n" + "-"*40)
    print("Сравнение сгенерированных изображений:")
    print("-"*40)
    
    for img in generated_images:
        brightness = "светлое" if img['avg_intensity'] > 150 else "тёмное"
        print(f"  {img['text_prompt'][:30]:30} → {brightness}, avg={img['avg_intensity']:.0f}")
        
    # Демонстрация влияния seed
    print("\n" + "-"*40)
    print("Влияние seed на генерацию:")
    print("-"*40)
    
    base_prompt = "a beautiful landscape"
    for seed in [42, 123, 456]:
        random.seed(seed)
        result = diffusion.generate_from_text(base_prompt, seed=seed)
        print(f"  Seed {seed}: avg_intensity={result['avg_intensity']:.1f}")


# =============================================================================
# Part 3: Prompt Engineering Basics
# =============================================================================

class PromptEngineer:
    """Инструменты для разработки промптов для генерации изображений."""
    
    def __init__(self):
        self.positive_modifiers = [
            "highly detailed", "professional", "beautiful", "stunning",
            "masterpiece", "best quality", "8k", "hdr", "ultra realistic"
        ]
        
        self.negative_modifiers = [
            "blurry", "low quality", "deformed", "ugly", "duplicate",
            "error", "extra fingers", "poorly drawn", "bad anatomy"
        ]
        
    def enhance_prompt(self, base_prompt, style=None, quality_boost=True):
        """Улучшение промпта модификаторами."""
        enhanced = base_prompt
        
        if style:
            style_map = {
                "photo": "professional photograph, realistic, detailed",
                "art": "digital art, illustration, artistic",
                "anime": "anime style, vibrant colors, detailed",
                "oil_painting": "oil painting, classical, rich colors",
                "cinematic": "cinematic lighting, dramatic, movie still"
            }
            enhanced += f", {style_map.get(style, style)}"
            
        if quality_boost:
            random.seed(hash(base_prompt) % 1000)
            selected = random.sample(self.positive_modifiers, min(3, len(self.positive_modifiers)))
            enhanced += ", " + ", ".join(selected)
            
        return enhanced
        
    def create_negative_prompt(self, issues=None):
        """Создание негативного промпта."""
        negative = list(self.negative_modifiers[:5])
        
        if issues:
            for issue in issues:
                if issue not in negative:
                    negative.append(issue)
                    
        return ", ".join(negative)
        
    def analyze_prompt_structure(self, prompt):
        """Анализ структуры промпта."""
        words = prompt.split()
        
        # Определение компонентов
        components = {
            "subject": [],
            "modifiers": [],
            "quality_terms": [],
            "composition": []
        }
        
        quality_terms = ["detailed", "high quality", "4k", "8k", "professional", "realistic"]
        composition_terms = ["closeup", "wide angle", "portrait", "landscape", "overhead", "aerial"]
        
        for word in words:
            word_lower = word.lower()
            if word_lower in quality_terms:
                components["quality_terms"].append(word)
            elif word_lower in composition_terms:
                components["composition"].append(word)
            else:
                # Простая эвристика
                if len(word) > 3:
                    components["subject"].append(word)
                    
        return {
            "word_count": len(words),
            "components": components,
            "has_quality_terms": len(components["quality_terms"]) > 0,
            "has_composition": len(components["composition"]) > 0
        }


def demo_prompt_engineering():
    """Демонстрация основ prompt engineering."""
    print("\n" + "="*70)
    print("DEMO 3: Prompt Engineering")
    print("="*70)
    
    engineer = PromptEngineer()
    
    # Базовые промпты для улучшения
    base_prompts = [
        "a cat sitting on a windowsill",
        "a futuristic city at night",
        "portrait of an old fisherman",
        "a magical forest with glowing mushrooms"
    ]
    
    styles = ["photo", "art", "anime", "cinematic"]
    
    print("\nУлучшение промптов:")
    print("-"*40)
    
    for i, (prompt, style) in enumerate(zip(base_prompts, styles)):
        enhanced = engineer.enhance_prompt(prompt, style)
        print(f"\n{i+1}. Базовый: \"{prompt}\"")
        print(f"   Стиль: {style}")
        print(f"   Улучшенный: \"{enhanced}\"")
        
    # Демонстрация негативных промптов
    print("\n" + "-"*40)
    print("Негативные промпты:")
    print("-"*40)
    
    issues_list = [
        None,
        ["extra fingers", "bad hands"],
        ["blurry", "low resolution", "artifacts"]
    ]
    
    for issues in issues_list:
        neg_prompt = engineer.create_negative_prompt(issues)
        print(f"\n  Проблемы: {issues}")
        print(f"  Негативный промпт: \"{neg_prompt}\"")
        
    # Анализ структуры промптов
    print("\n" + "-"*40)
    print("Анализ структуры промптов:")
    print("-"*40)
    
    test_prompts = [
        "a beautiful detailed landscape photograph",
        "closeup portrait, high quality, professional lighting",
        "anime girl with blue hair, vibrant colors, masterpiece"
    ]
    
    for prompt in test_prompts:
        analysis = engineer.analyze_prompt_structure(prompt)
        print(f"\n  Промпт: \"{prompt}\"")
        print(f"  Слов: {analysis['word_count']}")
        print(f"  Качество: {analysis['has_quality_terms']}")
        print(f"  Композиция: {analysis['has_composition']}")
        print(f"  Компоненты: {analysis['components']}")


# =============================================================================
# Part 4: Comparison of Generation Methods
# =============================================================================

class GenerationMethodComparator:
    """Сравнение различных методов генерации изображений."""
    
    def __init__(self):
        self.methods = {
            "GAN": {
                "full_name": "Generative Adversarial Network",
                "speed": 0.1,      # секунд на изображение
                "quality": 7,       # из 10
                "diversity": 5,
                "controllability": 3,
                "training_data": "10K-100K images",
                "memory": "2-8 GB",
                "pros": ["Fast inference", "Good quality", "Well-understood"],
                "cons": ["Mode collapse", "Training instability", "Limited control"]
            },
            "VAE": {
                "full_name": "Variational Autoencoder",
                "speed": 0.05,
                "quality": 6,
                "diversity": 8,
                "controllability": 6,
                "training_data": "10K-100K images",
                "memory": "1-4 GB",
                "pros": ["Fast", "Smooth latent space", "Good for interpolation"],
                "cons": ["Blurry outputs", "Lower quality than GANs"]
            },
            "Diffusion": {
                "full_name": "Diffusion Models (DDPM, Stable Diffusion)",
                "speed": 2.0,
                "quality": 9,
                "diversity": 9,
                "controllability": 8,
                "training_data": "100K-1B images",
                "memory": "4-16 GB",
                "pros": ["State-of-art quality", "High diversity", "Text-conditional"],
                "cons": ["Slow inference", "High compute cost"]
            },
            "CLIP+GAN": {
                "full_name": "CLIP-Guided Generation",
                "speed": 1.5,
                "quality": 8,
                "diversity": 7,
                "controllability": 9,
                "training_data": "400M image-text pairs",
                "memory": "8-12 GB",
                "pros": ["Text-guided", "Controllable", "Flexible"],
                "cons": ["Requires CLIP model", "May be unstable"]
            }
        }
        
    def compare_all(self):
        """Сравнение всех методов."""
        print("\nСравнительная таблица методов генерации:")
        print("="*90)
        print(f"{'Метод':<12} {'Качество':<10} {'Скорость':<12} {'Разнообразие':<14} {'Управляемость':<14}")
        print("-"*90)
        
        for name, props in self.methods.items():
            print(f"{name:<12} {props['quality']}/10{'':>5} {props['speed']:.1f}s{'':>8} "
                  f"{props['diversity']}/10{'':>7} {props['controllability']}/10")
                  
    def get_recommendation(self, requirements):
        """Рекомендация метода на основе требований."""
        best_method = None
        best_score = -1
        
        for name, props in self.methods.items():
            score = 0
            
            if requirements.get("speed_priority"):
                score += (10 - props["speed"] * 5) * requirements["speed_priority"]
            if requirements.get("quality_priority"):
                score += props["quality"] * requirements["quality_priority"]
            if requirements.get("controllability_priority"):
                score += props["controllability"] * requirements["controllability_priority"]
            if requirements.get("diversity_priority"):
                score += props["diversity"] * requirements["diversity_priority"]
                
            if score > best_score:
                best_score = score
                best_method = name
                
        return best_method, best_score


def demo_method_comparison():
    """Демонстрация сравнения методов генерации."""
    print("\n" + "="*70)
    print("DEMO 4: Comparison of Generation Methods")
    print("="*70)
    
    comparator = GenerationMethodComparator()
    
    # Сравнительная таблица
    comparator.compare_all()
    
    # Детали каждого метода
    print("\n" + "-"*40)
    print("Детали методов:")
    print("-"*40)
    
    for name, props in comparator.methods.items():
        print(f"\n{name} ({props['full_name']}):")
        print(f"  Память: {props['memory']}")
        print(f"  Обучение: {props['training_data']}")
        print(f"  Плюсы: {', '.join(props['pros'][:3])}")
        print(f"  Минусы: {', '.join(props['cons'][:3])}")
        
    # Рекомендации по сценариям
    print("\n" + "-"*40)
    print("Рекомендации по сценариям:")
    print("-"*40)
    
    scenarios = [
        {"name": "Быстрая генерация", "speed_priority": 2, "quality_priority": 1},
        {"name": "Высокое качество", "quality_priority": 2, "diversity_priority": 1},
        {"name": "Текстовое управление", "controllability_priority": 2, "quality_priority": 1},
        {"name": "Разнообразные Outputs", "diversity_priority": 2, "quality_priority": 1}
    ]
    
    for scenario in scenarios:
        name = scenario.pop("name")
        method, score = comparator.get_recommendation(scenario)
        print(f"\n  {name}:")
        print(f"    → Рекомендация: {method} (score: {score:.1f})")
        
    # Эволюция методов
    print("\n" + "-"*40)
    print("Эволюция методов генерации изображений:")
    print("-"*40)
    
    timeline = [
        ("2014", "GAN (Goodfellow)"),
        ("2015", "DCGAN — стабильная архитектура"),
        ("2016", "Variational Autoencoders"),
        ("2017", "ProGAN — генерация лиц"),
        ("2018", "BigGAN — масштабная генерация"),
        ("2020", "DDPM — диффузионные модели"),
        ("2021", "CLIP — связь текст-изображение"),
        ("2022", "DALL-E 2, Stable Diffusion"),
        ("2023", "SDXL, DALL-E 3, Midjourney v5"),
        ("2024", "Sora (видео), Flux, improved SD")
    ]
    
    for year, event in timeline:
        print(f"  {year}: {event}")


# =============================================================================
# Part 5: Interactive Demo — Combined Pipeline
# =============================================================================

def demo_combined_pipeline():
    """Комбинированная демонстрация всех концепций."""
    print("\n" + "="*70)
    print("BONUS: Combined Text-to-Image Pipeline Demo")
    print("="*70)
    
    # Шаг 1: Prompt Engineering
    print("\n[Step 1] Prompt Engineering")
    engineer = PromptEngineer()
    raw_prompt = "a red dragon flying over a medieval castle"
    enhanced = engineer.enhance_prompt(raw_prompt, style="art")
    negative = engineer.create_negative_prompt(["blurry", "low quality"])
    
    print(f"  Raw prompt: \"{raw_prompt}\"")
    print(f"  Enhanced: \"{enhanced}\"")
    print(f"  Negative: \"{negative}\"")
    
    # Шаг 2: CLIP Encoding
    print("\n[Step 2] CLIP Text Encoding")
    model = CLIPModel()
    text_emb = model.get_text_embedding(enhanced)
    print(f"  Text embedding dim: {len(text_emb)}")
    print(f"  Embedding stats: mean={sum(text_emb)/len(text_emb):.4f}, "
          f"std={math.sqrt(sum(x*x for x in text_emb)/len(text_emb)):.4f}")
    
    # Шаг 3: Diffusion Generation
    print("\n[Step 3] Diffusion Generation")
    diffusion = SimpleDiffusionProcess(image_size=8, num_timesteps=20)
    result = diffusion.generate_from_text(enhanced, seed=42)
    
    print(f"  Generated image stats:")
    print(f"    Average intensity: {result['avg_intensity']:.1f}")
    print(f"    Dynamic range: [{result['min']:.0f}, {result['max']:.0f}]")
    
    # Шаг 4: Quality Assessment
    print("\n[Step 4] Quality Assessment")
    analysis = engineer.analyze_prompt_structure(enhanced)
    print(f"  Prompt analysis:")
    print(f"    Words: {analysis['word_count']}")
    print(f"    Quality terms: {analysis['has_quality_terms']}")
    print(f"    Composition: {analysis['has_composition']}")
    print(f"    Components: {analysis['components']}")
    
    # Рекомендация
    print("\n[Step 5] Method Recommendation")
    comparator = GenerationMethodComparator()
    method, score = comparator.get_recommendation({
        "quality_priority": 2,
        "controllability_priority": 1.5
    })
    print(f"  Best method for this prompt: {method}")
    print(f"  Confidence score: {score:.1f}/20")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Phase 08: Generative AI — Text-to-Image Generation")
    print("="*70)
    
    # Запуск всех демонстраций
    demo_clip_alignment()
    demo_generation_pipeline()
    demo_prompt_engineering()
    demo_method_comparison()
    demo_combined_pipeline()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
