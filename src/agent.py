"""The agent: its environment (tools), its inputs and its loop.

Environment E0 is the upstream MedAgentBench v2 tool set with its mechanical defects fixed (`tools.py`, `tools.FIXES`).
Environment E1 is E0 and the institutional tools (`institution.py`).
The system prompt describes a general way of working, the same for every task and arm; the tool descriptions carry
each tool's capability and use, and the tools do the raw computation. The prompt frames each rule as a standing order
whose actions apply when its criteria are met; the step that settles the criteria ends when every criterion has its
answer and records. The prompt scopes the work to one rule, the task's instructions or the protocol of its topic, and
names the unmet outcome (the chart stays unchanged); the review checks against that rule.
Instruction level L0 is the upstream user message: the original instruction and the original `context`.
Instruction level L2 is one goal sentence per task family (`GOALS`): the clinical topic and the patient, with no codes,
thresholds, steps, time or answer format. In L2 the agent gets no `context`; E1 holds that information.

The original-harness baseline (arm "original") is the upstream benchmark unchanged: upstream prompt, tools and loop.
Arm "original-corrected" is that baseline given the corrected task of arms A to D (`run.py`).

The loop follows the upstream agent (Responses API, one tool call at a time, ends at `finish`) with three changes that
hold in arms A to D:
- Tool outputs reach the agent as compact JSON (`serialized`); upstream sends the Python repr.
- A tool error (invalid arguments, a refused write) goes back to the agent as the tool output; upstream ends the run.
- On the first consecutive response without a tool call the agent gets one reminder; on the second the run stops.
- The first `finish` does not end the run: its output is the review (`review`), the agent's successful chart changes
  and a request to check them against the task; the next `finish` ends the run. A run that ends without the second
  `finish` keeps the first answer.
"""
import json

from clinical_agent import institution, tools as toolset

SYSTEM_PROMPT = (
    "You work as a clinician in a hospital's electronic health record, through the tools you are given. The user "
    "message sets one task on one topic. One rule governs it: the task's own instructions, or the hospital's protocol "
    "for that topic; other protocols that a search returns belong to other reviews.\n"
    "1. Read the rule. List each criterion and the action it triggers, including what the rule says a missing record "
    "means.\n"
    "2. Settle each criterion from the chart, with the codes, sources and times the rule names: met when the records "
    "show it, otherwise unmet. This step is complete when every criterion is marked met or unmet with its records.\n"
    "3. Act on each met criterion exactly as the rule states; an unmet criterion leaves the chart unchanged. Then "
    "finish with your answer."
)
REMINDER = "Continue the task with a tool call. When you have the answer, call `finish` with it."
REVIEW = ("Before the task ends, check your work against the task's rule. {changes}\nEach change carries out, once, an "
          "action of that rule whose criterion the records showed met, and each such action is done. Then call "
          "`finish` with your final answer.")

# The topic names the protocol of the family in the protocol library (`institution.PROTOCOLS`).
GOAL = "Review {topic} for patient {patient} and act according to the hospital's protocols."
GOALS = {"task1": "renal mass CT Abdomen follow-up", "task2": "DVT prophylaxis", "task3": "heart rate",
         "task4": "urinary catheter", "task5": "malignant neoplasm of left kidney follow-up",
         "task6": "thyroid function", "task7": "QTc interval", "task8": "opioid orders and naloxone",
         "task9": "influenza vaccination", "task10": "COVID-19 vaccination"}

ARMS = {"A": {"instruction": "L0", "environment": "E0"}, "B": {"instruction": "L0", "environment": "E1"},
        "C": {"instruction": "L2", "environment": "E0"}, "D": {"instruction": "L2", "environment": "E1"},
        "original": {"instruction": "L0", "environment": "original"},
        "original-corrected": {"instruction": "L0", "environment": "original"}}
# The original-harness baseline (`ORIGINAL_MAX_STEPS` in run.py): the upstream system prompt of the benchmark run
# (`scripts/collect_agent_responses.py`), the upstream tools, the original task and the upstream loop, which ends at
# the first tool error or the first response without a tool call.
ORIGINAL_PROMPT = (toolset.UPSTREAM / "src/prompts/system.txt").read_text(encoding="utf-8")


def tools(environment, endpoint, task):
    """{tool name: Tool} for one FHIR endpoint (http://host:port/fhir)."""
    if environment == "original":
        return toolset.original(endpoint)
    if environment == "E0":
        return toolset.e0(endpoint)
    if environment == "E1":
        return institution.e1(endpoint, task)
    raise ValueError("Environment must be E0 or E1")


def user_message(task, level):
    """L0: the upstream user message (`MedAgent.create_user_message`). L2: the goal sentence of the task family."""
    if level == "L2":
        goal = GOAL.format(topic=GOALS[task["id"].split("_")[0]], patient=task["eval_MRN"])
        return f"<instruction>\n{goal}\n</instruction>\n"
    if level != "L0":
        raise ValueError("Instruction level must be L0 or L2")
    content = f"<instruction>\n{task['instruction']}\n</instruction>\n"
    if task.get("context"):
        content += f"<context>\n{task['context']}\n</context>\n"
    return content


def text_of(item):
    return "\n".join(p.get("text", "") for p in item.get("content") or [] if isinstance(p, dict))


def serialized(result):
    """The tool output as the agent reads it in arms A to D: text as it is, anything else as compact JSON (upstream and
    the original harness: the Python repr)."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)


def call_tool(tool, arguments):
    """(arguments, ok, output) for the JSON `arguments` of a call. The output of a failed call is the error text that
    the agent reads."""
    try:
        args = json.loads(arguments)
    except ValueError as error:
        return arguments, False, f"Invalid JSON arguments: {error}"
    if tool is None:
        return args, False, "Unknown tool"
    try:
        return args, True, tool(tool.input_schema.model_validate(args))
    except Exception as error:
        return args, False, f"{type(error).__name__}: {error}"


def review(trace):
    """The review message: the successful chart changes of the attempt so far, as the agent called them."""
    changes = [f"- {x['name']}: {serialized(x['arguments'])}" for x in trace
               if x["type"] == "tool_call" and x["ok"] and x["name"] in toolset.WRITES]
    listed = ("The changes you made to the chart:\n" + "\n".join(changes)) if changes \
        else "You made no changes to the chart."
    return REVIEW.format(changes=listed)


def run(client, task, arm, endpoint, *, model, effort, max_steps):
    """One attempt. Returns {"status", "value", "trace", "calls", "steps"}.

    `trace` items: message, tool_call (with `ok` and `output`), reminder. `calls` holds one record per model call:
    returned model, service tier, status and usage (with `cost`, the cost that OpenRouter reports). Status is completed, no_tool_call,
    step_limit, agent_error or tool_error (original harness: the first failed tool call ends the run)."""
    original = ARMS[arm]["environment"] == "original"
    registry = tools(ARMS[arm]["environment"], endpoint, task)
    schemas = [t.json_schema() for t in registry.values()]
    messages = [{"role": "system", "content": ORIGINAL_PROMPT if original else SYSTEM_PROMPT},
                {"role": "user", "content": user_message(task, ARMS[arm]["instruction"])}]
    trace, calls = [], []
    first_answer = None  # the value of the first `finish`, which starts the review (arms A to D)

    def end(status, step, value=None, **extra):
        value = first_answer if value is None else value
        return {"status": status, "value": [] if value is None else value, "trace": trace, "calls": calls,
                "steps": step, **extra}

    step, empty = 0, 0
    try:
        for step in range(1, max_steps + 1):
            response = client.responses.create(model=model["name"], input=messages, tools=schemas,
                                               parallel_tool_calls=False, reasoning={"effort": effort},
                                               max_output_tokens=8192, service_tier=model["service_tier"])
            data = response.model_dump(mode="json")
            calls.append({k: data.get(k) for k in ("model", "service_tier", "status", "usage")})
            used_tool = False
            for output in response.output:
                item = output if isinstance(output, dict) else output.model_dump()
                if item.get("type") == "message":
                    content = text_of(item)
                    messages.append({"role": item.get("role", "assistant"), "content": content})
                    trace.append({"type": "message", "content": content})
                    continue
                if item.get("type") != "function_call":
                    continue
                used_tool = True
                messages.append(item)
                args, ok, result = call_tool(registry.get(item["name"]), item["arguments"])
                # The first `finish` keeps its answer as its output in the trace (`run.attempt_value` reads the last
                # one) and shows the agent the review instead.
                reviewing = ok and item["name"] == "finish" and not original and first_answer is None
                shown = review(trace) if reviewing else result
                trace.append({"type": "tool_call", "name": item["name"], "arguments": args, "ok": ok,
                              "output": result, **({"review": shown} if reviewing else {})})
                messages.append({"type": "function_call_output", "call_id": item["call_id"],
                                 "output": str(shown) if original else serialized(shown)})
                if reviewing:
                    first_answer = result
                elif ok and item["name"] == "finish":
                    return end("completed", step, result)
                if original and not ok:
                    return end("tool_error", step)
            empty = 0 if used_tool else empty + 1
            if original and empty:
                return end("no_tool_call", step)
            if empty == 1:
                messages.append({"role": "user", "content": REMINDER})
                trace.append({"type": "reminder", "content": REMINDER})
            elif empty == 2:
                return end("no_tool_call", step)
        return end("step_limit", step)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        return end("agent_error", step, error_type=type(error).__name__, error=str(error))
