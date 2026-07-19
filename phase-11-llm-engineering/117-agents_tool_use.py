"""117 — Agents & Tool Use: вызов функций, ReAct, планирование

Темы:
  1. Function Calling (JSON schema, parameter validation, tool registry)
  2. ReAct Pattern (Reason + Act loop, observation formatting)
  3. Tool Orchestration (sequential/parallel, error handling, retries)
  4. Planning Strategies (task decomposition, goal tracking)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import datetime
import collections

random.seed(42)


# =============================================================================
# 1. Function Calling — JSON schema, parameter validation, tool registry
# =============================================================================

def demo_function_calling():
    print("=" * 70)
    print("DEMO 1: Function Calling — JSON schema, parameter validation, tool registry")
    print("=" * 70)

    # --- 1a. Tool registry with JSON schemas ---
    tool_registry = {
        "get_weather": {
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["city"]
            }
        },
        "calculate": {
            "description": "Perform arithmetic",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        },
        "search": {
            "description": "Search knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["query"]
            }
        }
    }
    print("--- Tool Registry ---")
    for name, spec in tool_registry.items():
        req = spec["parameters"].get("required", [])
        print(f"  {name}: {spec['description']} | required: {req}")

    # --- 1b. Parameter validation ---
    def validate_params(tool_name, params):
        schema = tool_registry.get(tool_name, {}).get("parameters", {})
        errors = []
        required = schema.get("required", [])
        for field in required:
            if field not in params:
                errors.append(f"Missing required parameter: '{field}'")
        for key, val in params.items():
            if key in schema.get("properties", {}):
                prop = schema["properties"][key]
                if prop["type"] == "string" and not isinstance(val, str):
                    errors.append(f"'{key}' must be string, got {type(val).__name__}")
                elif prop["type"] == "integer" and not isinstance(val, int):
                    errors.append(f"'{key}' must be integer, got {type(val).__name__}")
                if "enum" in prop and val not in prop["enum"]:
                    errors.append(f"'{key}' must be one of {prop['enum']}, got '{val}'")
                if "minimum" in prop and isinstance(val, (int, float)) and val < prop["minimum"]:
                    errors.append(f"'{key}' must be >= {prop['minimum']}, got {val}")
        return errors

    test_cases = [
        ("get_weather", {"city": "Moscow", "units": "celsius"}),
        ("get_weather", {"units": "fahrenheit"}),  # missing city
        ("calculate", {"expression": "2+2"}),
        ("search", {"query": "AI", "top_k": 15}),  # out of range
    ]
    print("\n--- Parameter Validation ---")
    for tool, params in test_cases:
        errs = validate_params(tool, params)
        status = "OK" if not errs else f"ERRORS: {errs}"
        print(f"  {tool}({params}) -> {status}")

    # --- 1c. Simulated function calling ---
    def execute_tool(tool_name, params):
        if tool_name == "get_weather":
            random.seed(hashlib.md5(params["city"].encode()).hexdigest())
            temp = random.randint(-10, 35)
            return {"city": params["city"], "temp": temp, "units": params.get("units", "celsius")}
        elif tool_name == "calculate":
            result = eval(params["expression"])  # safe for demo
            return {"expression": params["expression"], "result": result}
        elif tool_name == "search":
            results = [f"doc_{i}" for i in range(params.get("top_k", 3))]
            return {"query": params["query"], "results": results}
        return {"error": "unknown tool"}

    print("\n--- Tool Execution ---")
    call = {"name": "get_weather", "arguments": {"city": "Berlin"}}
    result = execute_tool(call["name"], call["arguments"])
    print(f"  Call: {call['name']}({json.dumps(call['arguments'])})")
    print(f"  Result: {json.dumps(result)}")

    # --- 1d. LLM-style function call parsing ---
    def parse_function_call(llm_output):
        match = re.search(r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}', llm_output)
        if match:
            return {"name": match.group(1), "arguments": json.loads(match.group(2))}
        return None

    raw = 'I need weather info. {"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    parsed = parse_function_call(raw)
    print(f"\n--- LLM Output Parsing ---")
    print(f"  Raw: {raw[:70]}...")
    print(f"  Parsed: {json.dumps(parsed)}")

    print()


# =============================================================================
# 2. ReAct Pattern — Reason + Act loop, observation formatting
# =============================================================================

def demo_react_pattern():
    print("=" * 70)
    print("DEMO 2: ReAct Pattern — Reason + Act loop, observation formatting")
    print("=" * 70)

    # --- 2a. ReAct trace structure ---
    react_trace = {
        "thought_1": "The user wants the weather in Paris. I should use get_weather.",
        "action_1": {"name": "get_weather", "args": {"city": "Paris"}},
        "observation_1": {"temp": 18, "condition": "cloudy"},
        "thought_2": "Got temperature 18°C. I'll also check tomorrow's forecast.",
        "action_2": {"name": "get_forecast", "args": {"city": "Paris", "days": 1}},
        "observation_2": {"tomorrow_temp": 22, "condition": "sunny"},
        "thought_3": "I have all the info. Current: 18°C cloudy. Tomorrow: 22°C sunny.",
        "answer": "Paris: 18°C cloudy today, 22°C sunny tomorrow."
    }
    print("--- ReAct Trace Structure ---")
    for k, v in react_trace.items():
        print(f"  {k}: {json.dumps(v) if isinstance(v, dict) else v}")

    # --- 2b. Simulated ReAct loop ---
    def react_agent(question, max_steps=5):
        trace = []
        env_state = {"context": question, "results": {}}
        step = 0
        keywords_to_tools = {
            "weather": "get_weather",
            "search": "web_search",
            "calculate": "calculator",
            "summarize": "summarizer"
        }

        while step < max_steps:
            step += 1
            # Reason
            thought = f"Step {step}: Analyzing '{env_state['context'][:40]}...'"
            # Decide action
            action = None
            for kw, tool in keywords_to_tools.items():
                if kw in env_state["context"].lower():
                    action = {"name": tool, "args": {"input": env_state["context"][:50]}}
                    break
            if action is None:
                # Final answer
                trace.append({"step": step, "thought": thought, "answer": f"Processed: {question[:60]}"})
                break
            # Observe
            obs = {"status": "success", "data": f"result_for_{action['name']}"}
            env_state["results"][action["name"]] = obs
            trace.append({"step": step, "thought": thought, "action": action, "observation": obs})

        return trace

    print("\n--- Simulated ReAct Loop ---")
    trace = react_agent("Search for latest weather in Moscow and summarize it")
    for entry in trace:
        print(f"  Step {entry['step']}:")
        for k in ["thought", "action", "observation", "answer"]:
            if k in entry:
                print(f"    {k}: {json.dumps(entry[k]) if isinstance(entry[k], dict) else entry[k]}")

    # --- 2c. Observation formatting ---
    def format_observation(raw_obs):
        if isinstance(raw_obs, dict):
            lines = [f"  {k}: {v}" for k, v in raw_obs.items()]
            return "Observation:\n" + "\n".join(lines)
        return f"Observation: {raw_obs}"

    observations = [
        {"temperature": 25, "humidity": 60, "wind": "10 km/h"},
        "Search returned 5 results",
        {"error": "timeout", "retry_in": 5}
    ]
    print("\n--- Observation Formatting ---")
    for obs in observations:
        print(f"  {format_observation(obs)}")

    # --- 2d. ReAct with error recovery ---
    def react_with_recovery(question):
        steps = []
        attempts = 0
        max_retries = 3
        success = False

        while not success and attempts < max_retries:
            attempts += 1
            if attempts <= 2:
                steps.append({
                    "attempt": attempts,
                    "thought": f"Trying tool call (attempt {attempts})",
                    "action": "get_data",
                    "result": "ERROR: timeout" if attempts == 1 else "ERROR: 500"
                })
            else:
                steps.append({
                    "attempt": attempts,
                    "thought": "Switching to fallback strategy",
                    "action": "use_cache",
                    "result": "SUCCESS: cached data retrieved"
                })
                success = True
        return steps

    print("\n--- ReAct with Error Recovery ---")
    steps = react_with_recovery("Get real-time data")
    for s in steps:
        print(f"  Attempt {s['attempt']}: {s['thought']} -> {s['result']}")

    print()


# =============================================================================
# 3. Tool Orchestration — sequential/parallel, error handling, retries
# =============================================================================

def demo_tool_orchestration():
    print("=" * 70)
    print("DEMO 3: Tool Orchestration — sequential/parallel, error handling, retries")
    print("=" * 70)

    # --- 3a. Sequential orchestration ---
    def sequential_pipeline(steps, data):
        results = []
        current = data
        for step in steps:
            random.seed(hash(step["name"]))
            if step["name"] == "tokenize":
                tokens = current.split()[:5]
                current = tokens
                results.append({"step": "tokenize", "output": tokens})
            elif step["name"] == "embed":
                embeddings = [round(random.gauss(0, 0.5), 4) for _ in current]
                current = embeddings
                results.append({"step": "embed", "output_length": len(embeddings)})
            elif step["name"] == "classify":
                scores = {c: round(random.random(), 3) for c in ["positive", "negative", "neutral"]}
                current = scores
                results.append({"step": "classify", "output": scores})
        return results

    pipeline = [
        {"name": "tokenize", "params": {"max_tokens": 5}},
        {"name": "embed", "params": {"dim": 8}},
        {"name": "classify", "params": {"labels": ["positive", "negative", "neutral"]}}
    ]
    print("--- Sequential Pipeline ---")
    seq_results = sequential_pipeline(pipeline, "This is a great demonstration of orchestration")
    for r in seq_results:
        print(f"  {r['step']}: {json.dumps(r, ensure_ascii=False)}")

    # --- 3b. Parallel orchestration ---
    def parallel_execute(tools, data):
        results = []
        for tool in tools:
            random.seed(hash(tool["name"] + str(random.randint(0, 1000))))
            latency = round(random.uniform(0.1, 1.0), 2)
            results.append({
                "tool": tool["name"],
                "latency_s": latency,
                "result": f"output_of_{tool['name']}"
            })
        results.sort(key=lambda x: x["latency_s"])
        total = sum(r["latency_s"] for r in results)
        parallel_time = max(r["latency_s"] for r in results)
        return results, total, parallel_time

    print("\n--- Parallel Execution ---")
    tools = [
        {"name": "sentiment_analysis"},
        {"name": "entity_extraction"},
        {"name": "topic_classification"},
        {"name": "summarization"}
    ]
    results, seq_time, par_time = parallel_execute(tools, "some input")
    for r in results:
        print(f"  {r['tool']}: {r['latency_s']}s -> {r['result']}")
    print(f"  Sequential: {seq_time:.2f}s | Parallel: {par_time:.2f}s | Speedup: {seq_time/par_time:.1f}x")

    # --- 3c. Error handling with retries ---
    def execute_with_retry(tool_func, args, max_retries=3, backoff=1.0):
        history = []
        for attempt in range(1, max_retries + 1):
            try:
                result = tool_func(args)
                history.append({"attempt": attempt, "status": "success"})
                return result, history
            except Exception as e:
                wait = backoff * (2 ** (attempt - 1))
                history.append({"attempt": attempt, "status": "error", "msg": str(e), "wait": wait})
        return None, history

    def flaky_tool(args):
        r = random.random()
        if r < 0.6:
            raise ConnectionError("Service unavailable")
        return {"data": "ok"}

    print("\n--- Retry with Exponential Backoff ---")
    random.seed(42)
    result, history = execute_with_retry(flaky_tool, {"q": "test"})
    for h in history:
        print(f"  Attempt {h['attempt']}: {h['status']}" + (f" - retry in {h.get('wait', 0)}s" if "wait" in h else ""))
    print(f"  Final result: {result}")

    # --- 3d. Tool composition with dependency graph ---
    def topological_sort(graph):
        in_degree = {n: 0 for n in graph}
        for n, deps in graph.items():
            for d in deps:
                in_degree[n] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for n, deps in graph.items():
                if node in deps:
                    in_degree[n] -= 1
                    if in_degree[n] == 0:
                        queue.append(n)
        return order

    deps = {
        "fetch_data": ["parse_input"],
        "parse_input": [],
        "enrich": ["fetch_data", "parse_input"],
        "validate": ["enrich"],
        "output": ["validate"]
    }
    print("\n--- Dependency Graph & Topological Order ---")
    order = topological_sort(deps)
    print(f"  Dependencies: {json.dumps(deps)}")
    print(f"  Execution order: {' -> '.join(order)}")

    print()


# =============================================================================
# 4. Planning Strategies — task decomposition, goal tracking
# =============================================================================

def demo_planning_strategies():
    print("=" * 70)
    print("DEMO 4: Planning Strategies — task decomposition, goal tracking")
    print("=" * 70)

    # --- 4a. Task decomposition ---
    def decompose_task(goal, depth=0, max_depth=3):
        random.seed(hash(goal) + depth)
        if depth >= max_depth or len(goal.split()) <= 2:
            return {"task": goal, "subtasks": [], "estimated_time": round(random.uniform(1, 10), 1)}
        n_sub = random.randint(2, 3)
        subtasks = []
        words = goal.split()
        for i in range(n_sub):
            sub = " ".join(words[i % len(words):(i + 1) % len(words) + 1] or words[:2])
            subtasks.append(decompose_task(sub, depth + 1, max_depth))
        return {"task": goal, "subtasks": subtasks, "estimated_time": round(sum(s["estimated_time"] for s in subtasks), 1)}

    print("--- Task Decomposition Tree ---")
    tree = decompose_task("Build a search engine with ranking")

    def print_tree(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}├─ {node['task']} (~{node['estimated_time']}s)")
        for sub in node["subtasks"]:
            print_tree(sub, indent + 1)

    print_tree(tree)

    # --- 4b. Goal tracking ---
    class GoalTracker:
        def __init__(self, goal):
            self.goal = goal
            self.subgoals = []
            self.completed = set()

        def add_subgoal(self, name, depends_on=None):
            self.subgoals.append({
                "name": name,
                "status": "pending",
                "depends_on": depends_on or []
            })

        def complete(self, name):
            self.completed.add(name)
            for sg in self.subgoals:
                if sg["name"] == name:
                    sg["status"] = "done"

        def progress(self):
            done = sum(1 for s in self.subgoals if s["status"] == "done")
            return done / max(len(self.subgoals), 1)

        def status_report(self):
            report = []
            for sg in self.subgoals:
                deps_met = all(d in self.completed for d in sg["depends_on"])
                if sg["status"] == "pending" and deps_met:
                    sg["status"] = "ready"
                report.append(f"  [{sg['status']:>7}] {sg['name']} (deps: {sg['depends_on']})")
            return report

    tracker = GoalTracker("Deploy ML model")
    tracker.add_subgoal("collect_data")
    tracker.add_subgoal("preprocess", ["collect_data"])
    tracker.add_subgoal("train_model", ["preprocess"])
    tracker.add_subgoal("evaluate", ["train_model"])
    tracker.add_subgoal("deploy", ["evaluate"])

    print("\n--- Goal Tracking ---")
    print("  Progress:", tracker.progress())
    for line in tracker.status_report():
        print(line)

    tracker.complete("collect_data")
    tracker.complete("preprocess")
    print("\n  After completing collect_data and preprocess:")
    for line in tracker.status_report():
        print(line)
    print("  Progress:", tracker.progress())

    # --- 4c. Plan refinement ---
    def refine_plan(plan, new_evidence):
        refined = []
        for step in plan:
            if new_evidence.get(step["name"]) == "failed":
                refined.append({
                    **step,
                    "status": "refined",
                    "alternative": f"retry_{step['name']}_v2",
                    "reason": "original failed"
                })
            else:
                refined.append({**step, "status": "kept"})
        return refined

    plan = [
        {"name": "data_collection", "method": "api"},
        {"name": "data_validation", "method": "schema_check"},
        {"name": "training", "method": "fine_tune"}
    ]
    evidence = {"data_collection": "failed", "data_validation": "ok", "training": "ok"}
    print("\n--- Plan Refinement ---")
    print("  Original plan:", [s["name"] for s in plan])
    refined = refine_plan(plan, evidence)
    for step in refined:
        status = step.get("status", "unknown")
        alt = step.get("alternative", "")
        print(f"  {step['name']}: {status}" + (f" -> use {alt}" if alt else ""))

    # --- 4d. Hierarchical task network ---
    def htn_plan(methods, current_state, goal):
        plan = []
        queue = [(goal, current_state)]
        visited = set()
        while queue:
            task, state = queue.pop(0)
            if task in visited:
                continue
            visited.add(task)
            if task in methods:
                method = methods[task]
                new_state = {**state, **method.get("precondition", {})}
                plan.append({"task": task, "method": method["name"], "prereqs": list(method.get("precondition", {}).keys())})
                for sub in method.get("subtasks", []):
                    queue.append((sub, new_state))
            else:
                plan.append({"task": task, "method": "primitive", "prereqs": []})
        return plan

    methods = {
        "serve_food": {
            "name": "order_and_serve",
            "precondition": {"kitchen_ready": True},
            "subtasks": ["prepare_ingredients", "cook_meal", "plate_dish"]
        },
        "prepare_ingredients": {
            "name": "mise_en_place",
            "precondition": {"ingredients_available": True},
            "subtasks": ["chop_vegetables", "measure_spices"]
        },
        "cook_meal": {
            "name": "stove烹饪",
            "precondition": {"stove_hot": True},
            "subtasks": ["heat_pan", "cook_protein"]
        }
    }

    print("\n--- Hierarchical Task Network ---")
    htn = htn_plan(methods, {"kitchen_ready": True}, "serve_food")
    for item in htn:
        print(f"  {item['task']}: method={item['method']}, prereqs={item['prereqs']}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_function_calling()
    demo_react_pattern()
    demo_tool_orchestration()
    demo_planning_strategies()
