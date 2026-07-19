"""113 — Prompt Engineering Advanced: инструкции, структурированный вывод, JSON mode

Темы:
  1. Instruction Patterns (delimiters, role-based, step-by-step)
  2. Structured Output (JSON schemas, XML, markdown tables)
  3. Output Parsers (regex, keyword extraction, format validation)
  4. Prompt Chaining (multi-step pipelines, error recovery)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib

random.seed(42)


# ---------------------------------------------------------------------------
# Демо 1: Instruction Patterns — разграничители, роли, пошаговые инструкции
# ---------------------------------------------------------------------------
def demo_instruction_patterns():
    print("=" * 70)
    print("DEMO 1: Instruction Patterns — delimiters, roles, step-by-step")
    print("=" * 70)

    # 1.1 Delimiter-based instructions
    print("\n--- 1.1 Delimiter-based instructions ---")
    prompt = """Analyze the following text and identify the sentiment.

<text>
The product arrived on time and works perfectly. However, the packaging was damaged.
</text>

<instructions>
1. Identify positive and negative sentiments.
2. Classify overall sentiment as positive/negative/neutral.
3. Provide a confidence score 0-1.
</instructions>"""

    print(f"  Prompt ({len(prompt.split())} words):")
    print(f"  Uses delimiters: <text>...</text>, <instructions>...</instructions>")
    print(f"  Delimiter count: {prompt.count('<') + prompt.count('>')}")
    print(f"  Lines: {len(prompt.splitlines())}")

    # 1.2 Role-based instructions
    print("\n--- 1.2 Role-based instructions ---")
    role_prompt = """<role>
You are a senior data scientist with expertise in NLP.
You think step by step and justify your reasoning.
</role>

<task>
Classify the following review as spam or ham.
</task>

<output_format>
{"classification": "spam"|"ham", "confidence": 0.0-1.0, "reasoning": "..."}
</output_format>"""

    print(f"  Role prompt ({len(role_prompt.split())} words):")
    for tag in ["role", "task", "output_format"]:
        start = role_prompt.find(f"<{tag}>")
        end = role_prompt.find(f"</{tag}>")
        print(f"  Section <{tag}>: {end - start} chars")

    # 1.3 Step-by-step instructions (Chain of Thought)
    print("\n--- 1.3 Step-by-step (Chain of Thought) ---")
    cot_prompt = """Solve this step by step:

Step 1: Read the problem carefully.
Step 2: Identify the key variables.
Step 3: Set up equations.
Step 4: Solve each equation.
Step 5: Verify your answer.

Question: If 3x + 7 = 22, what is x?"""
    steps = re.findall(r"Step \d+: (.+)", cot_prompt)
    for i, step in enumerate(steps, 1):
        print(f"  Step {i}: {step}")

    # 1.4 Few-shot with examples
    print("\n--- 1.4 Few-shot instruction pattern ---")
    few_shot = """Classify these reviews:

Example 1:
Review: "Amazing product, fast delivery!"
Label: positive

Example 2:
Review: "Broke after 2 days, terrible quality."
Label: negative

Example 3:
Review: "It's okay, nothing special."
Label: neutral

Classify:
Review: "Love it! Best purchase ever." """
    examples = re.findall(r"Example \d+:\nReview: \"(.+?)\"\nLabel: (\w+)", few_shot)
    print(f"  Few-shot examples provided: {len(examples)}")
    for i, (rev, lab) in enumerate(examples, 1):
        print(f"  Example {i}: review=\"{rev}\" → {lab}")


# ---------------------------------------------------------------------------
# Демо 2: Structured Output — JSON, XML, markdown
# ---------------------------------------------------------------------------
def demo_structured_output():
    print("\n" + "=" * 70)
    print("DEMO 2: Structured Output — JSON schemas, XML, markdown tables")
    print("=" * 70)

    # 2.1 JSON schema enforcement
    print("\n--- 2.1 JSON schema definition ---")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "skills": {"type": "array", "items": {"type": "string"}},
            "rating": {"type": "number", "minimum": 0, "maximum": 5}
        },
        "required": ["name", "age", "skills"]
    }
    print(f"  Schema: {json.dumps(schema, indent=4)}")

    # Validate simulated output
    simulated = {"name": "Alice", "age": 30, "skills": ["Python", "ML"], "rating": 4.5}
    valid = all(k in simulated for k in schema["required"])
    print(f"\n  Simulated output: {json.dumps(simulated)}")
    print(f"  Valid: {valid}")

    # 2.2 XML-structured output
    print("\n--- 2.2 XML-structured output ---")
    xml_output = """<analysis>
  <sentiment>positive</sentiment>
  <confidence>0.87</confidence>
  <keywords>
    <keyword>amazing</keyword>
    <keyword>fast</keyword>
    <keyword>delivery</keyword>
  </keywords>
</analysis>"""
    print(f"  XML response:")
    for line in xml_output.strip().splitlines():
        print(f"  {line}")

    # Parse keywords
    keywords = re.findall(r"<keyword>(.+?)</keyword>", xml_output)
    sentiment = re.search(r"<sentiment>(.+?)</sentiment>", xml_output)
    print(f"\n  Parsed keywords: {keywords}")
    print(f"  Parsed sentiment: {sentiment.group(1) if sentiment else 'N/A'}")

    # 2.3 Markdown table output
    print("\n--- 2.3 Markdown table output ---")
    table = """| Model     | Params | MMLU  | Speed  |
|-----------|--------|-------|--------|
| GPT-4o    | 200B   | 88.7  | Fast   |
| Claude 3  | 175B   | 86.8  | Medium |
| Gemini    | 156B   | 83.7  | Fast   |"""
    print(table)

    # Parse table
    rows = re.findall(r"\| (.+?) \|", table)
    print(f"\n  Parsed {len(rows)} cells from markdown table")

    # 2.4 JSON mode validation
    print("\n--- 2.4 JSON mode validation ---")

    def validate_json_mode(text):
        try:
            parsed = json.loads(text)
            return {"valid": True, "keys": list(parsed.keys()), "data": parsed}
        except json.JSONDecodeError as e:
            return {"valid": False, "error": str(e)}

    json_candidates = [
        '{"answer": "Python", "confidence": 0.95}',
        '{"answer": "Python", "confidence": 0.95',  # missing }
        'The answer is Python with high confidence',  # not JSON
        '{"items": [1, 2, 3], "count": 3}',
    ]
    for candidate in json_candidates:
        result = validate_json_mode(candidate)
        status = "OK" if result["valid"] else "FAIL"
        print(f"  [{status}] {candidate[:50]:50s} → {result.get('keys', result.get('error'))}")


# ---------------------------------------------------------------------------
# Демо 3: Output Parsers — regex, keywords, format validation
# ---------------------------------------------------------------------------
def demo_output_parsers():
    print("\n" + "=" * 70)
    print("DEMO 3: Output Parsers — regex, keyword extraction, validation")
    print("=" * 70)

    # 3.1 Regex-based extraction
    print("\n--- 3.1 Regex extraction ---")
    llm_response = """Based on the analysis, here are my findings:

Confidence: 0.85
Sentiment: positive
Score: 4.2/5.0

The model detected 3 key themes in the text.
Date: 2025-01-15
Priority: HIGH"""

    patterns = {
        "confidence": r"Confidence:\s*([\d.]+)",
        "sentiment": r"Sentiment:\s*(\w+)",
        "score": r"Score:\s*([\d.]+)/([\d.]+)",
        "themes": r"(\d+)\s*key theme",
        "date": r"Date:\s*(\d{4}-\d{2}-\d{2})",
        "priority": r"Priority:\s*(\w+)",
    }

    extracted = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, llm_response)
        if match:
            extracted[field] = match.group(1) if len(match.groups()) == 1 else match.groups()

    for k, v in extracted.items():
        print(f"  {k:12s}: {v}")

    # 3.2 Keyword extraction
    print("\n--- 3.2 Keyword extraction ---")

    def extract_keywords(text, top_n=5):
        words = re.findall(r"\b[a-z]+\b", text.lower())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                     "at", "to", "for", "of", "and", "or", "it", "this", "that"}
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        sorted_kw = sorted(freq.items(), key=lambda x: -x[1])
        return sorted_kw[:top_n]

    sample_text = ("Natural language processing enables computers to understand "
                   "human language. The language model processes text using "
                   "transformer architecture for language understanding tasks.")
    keywords = extract_keywords(sample_text)
    print(f"  Text: \"{sample_text[:70]}...\"")
    print(f"  Keywords: {keywords}")

    # 3.3 Format validation
    print("\n--- 3.3 Format validation ---")

    def validate_format(text, expected_format):
        validators = {
            "json": lambda t: bool(json.loads(t)),
            "email": lambda t: bool(re.match(r"^[\w.+-]+@[\w-]+\.[\w.]+$", t)),
            "url": lambda t: bool(re.match(r"^https?://", t)),
            "date": lambda t: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", t)),
            "number": lambda t: bool(re.match(r"^-?\d+\.?\d*$", t.strip())),
        }
        validator = validators.get(expected_format)
        if not validator:
            return {"valid": False, "error": f"Unknown format: {expected_format}"}
        return {"valid": validator(text), "format": expected_format}

    test_cases = [
        ('{"key": "value"}', "json"),
        ("user@example.com", "email"),
        ("https://example.com", "url"),
        ("2025-01-15", "date"),
        ("42.5", "number"),
        ("not-a-date", "date"),
        ("invalid@@@", "email"),
    ]
    for text, fmt in test_cases:
        result = validate_format(text, fmt)
        status = "OK" if result["valid"] else "FAIL"
        print(f"  [{status}] {fmt:8s}: \"{text}\"")

    # 3.4 Multi-format extraction
    print("\n--- 3.4 Multi-format extraction ---")
    complex_response = """{
  "task": "classification",
  "results": [
    {"label": "spam", "confidence": 0.92},
    {"label": "ham", "confidence": 0.08}
  ],
  "metadata": {
    "model": "gpt-4o",
    "latency_ms": 150
  }
}"""

    parsed = json.loads(complex_response)
    print(f"  Task: {parsed['task']}")
    for r in parsed["results"]:
        print(f"  {r['label']:5s}: {r['confidence']:.2f}")
    print(f"  Model: {parsed['metadata']['model']}")
    print(f"  Latency: {parsed['metadata']['latency_ms']}ms")


# ---------------------------------------------------------------------------
# Демо 4: Prompt Chaining — multi-step pipelines, error recovery
# ---------------------------------------------------------------------------
def demo_prompt_chaining():
    print("\n" + "=" * 70)
    print("DEMO 4: Prompt Chaining — multi-step pipelines, error recovery")
    print("=" * 70)

    # 4.1 Two-step chain: extract → classify
    print("\n--- 4.1 Two-step chain: extract → classify ---")

    def step_extract(text):
        entities = re.findall(r"\b[A-Z][a-z]+\b", text)
        return {"entities": list(set(entities)), "word_count": len(text.split())}

    def step_classify(extracted):
        if extracted["word_count"] > 20:
            return {"type": "long", "confidence": 0.8}
        elif extracted["word_count"] > 5:
            return {"type": "medium", "confidence": 0.9}
        return {"type": "short", "confidence": 0.7}

    raw_text = "Apple announced a new iPhone model yesterday in California. The event was held at their headquarters."
    step1 = step_extract(raw_text)
    step2 = step_classify(step1)
    print(f"  Input: \"{raw_text[:60]}...\"")
    print(f"  Step 1 (extract): {step1}")
    print(f"  Step 2 (classify): {step2}")

    # 4.2 Three-step chain: analyze → plan → execute
    print("\n--- 4.2 Three-step chain: analyze → plan → execute ---")

    def step_analyze(data):
        issues = []
        for item in data:
            if item.get("score", 0) < 0.5:
                issues.append({"item": item["name"], "severity": "high"})
            elif item.get("score", 0) < 0.8:
                issues.append({"item": item["name"], "severity": "medium"})
        return {"issues": issues, "total": len(data)}

    def step_plan(analysis):
        plan = []
        for issue in analysis["issues"]:
            if issue["severity"] == "high":
                plan.append(f"Fix {issue['item']} immediately")
            else:
                plan.append(f"Schedule {issue['item']} for review")
        return {"actions": plan, "priority_order": len(analysis["issues"])}

    def step_execute(plan):
        results = []
        for i, action in enumerate(plan["actions"]):
            success = random.random() > 0.2
            results.append({"action": action, "success": success})
        return results

    data = [
        {"name": "API endpoint", "score": 0.3},
        {"name": "Auth module", "score": 0.6},
        {"name": "Database", "score": 0.9},
        {"name": "UI components", "score": 0.4},
    ]

    analysis = step_analyze(data)
    plan = step_plan(analysis)
    execution = step_execute(plan)

    print(f"  Input: {len(data)} items")
    print(f"  Analysis: {analysis}")
    print(f"  Plan: {plan}")
    print(f"  Execution:")
    for r in execution:
        status = "OK" if r["success"] else "FAIL"
        print(f"    [{status}] {r['action']}")

    # 4.3 Error recovery in chains
    print("\n--- 4.3 Error recovery ---")

    def robust_parse(text):
        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            fixed = re.sub(r",\s*}", "}", text)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                return json.loads(fixed), "auto_fixed"
            except json.JSONDecodeError:
                return None, "parse_failed"

    broken_jsons = [
        '{"key": "value",}',           # trailing comma
        '{"a": 1, "b": [1, 2, 3,}',    # broken array
        '{"nested": {"x": "y"}}',       # valid
    ]
    for j in broken_jsons:
        result, error = robust_parse(j)
        status = "OK" if result else "FAIL"
        print(f"  [{status}] {j:35s} → {result or error}")

    # 4.4 Pipeline with retry
    print("\n--- 4.4 Pipeline with retry ---")

    def pipeline_with_retry(input_data, max_retries=3):
        steps_completed = []
        for attempt in range(max_retries):
            try:
                # Step 1: validate
                if not input_data:
                    raise ValueError("Empty input")
                steps_completed.append("validate")

                # Step 2: process
                processed = {k: str(v).upper() for k, v in input_data.items()}
                steps_completed.append("process")

                # Step 3: output
                output = json.dumps(processed, sort_keys=True)
                steps_completed.append("output")

                return {"success": True, "output": output, "attempts": attempt + 1,
                        "steps": steps_completed}
            except Exception as e:
                steps_completed.append(f"error:{e}")

        return {"success": False, "attempts": max_retries, "steps": steps_completed}

    result = pipeline_with_retry({"name": "test", "value": 42})
    print(f"  Result: {json.dumps(result, indent=4)}")

    # Failed case
    result_fail = pipeline_with_retry({})
    print(f"\n  Failed case: {json.dumps(result_fail, indent=4)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_instruction_patterns()
    demo_structured_output()
    demo_output_parsers()
    demo_prompt_chaining()
    print("\n" + "=" * 70)
    print("All demos complete: 113-prompt_engineering_advanced.py")
    print("=" * 70)
