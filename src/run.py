"""Run one arm on one part of the split and score each attempt with the upstream v2 scorer.

The agent gets the corrected task (`tasks.CORRECTIONS`); in arm "original" it gets the original task. Each attempt has
three outcomes: `passed` (corrected task, scorer clock pinned to the task's reference time), `passed_upstream`
(original task, unchanged scorer, machine clock at scoring time) and `passed_revised` (as `passed`, with the revised
scorer, `revised_scorer.py`).

    python -m clinical_agent.run --arm A --part dev
    python -m clinical_agent.run --arm A --part dev --tasks task10_17 --run-id smoke

Each run starts its own FHIR containers (one per worker) and removes them at the end. Every attempt is scored on its
worker right after it ends; then the resources that the attempt created are deleted, so the next attempt on that
worker sees the original records. Output: results/<run id>/ with config.json (the command and all parameters),
attempts/<task id>.json, summary.json and fhir-resources.json (the container receipt). `--resume` continues a run
folder with the same parameters and skips the tasks that have an attempt file. Every run calls the paid model.
config.json records the SHA-256 of the source files; a validate or test run starts only when it equals the locked
design (`results/design-lock/lock.json`).
"""
import argparse
import datetime
import hashlib
import json
import queue
import shlex
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests

from clinical_agent import agent, institution, revised_scorer, tools
from clinical_agent.fhir_workers import FhirWorkers, atomic_json
from clinical_agent.openrouter_transport import OpenRouterChat
from clinical_agent.settings import FHIR_IMAGE, MODEL, MODELS, RESULTS
from clinical_agent.tasks import CORRECTIONS, SCORER, corrected, load_tasks

# Upstream wrapper `api_mapping`: the tools whose calls the scorer reads as HTTP requests.
API = {"patient_search": ("GET", "Patient"), "fhir_vitals_create": ("POST", "Observation"),
       "fhir_medication_request_create": ("POST", "MedicationRequest"),
       "fhir_service_request_create": ("POST", "ServiceRequest"), "fhir_observation_search": ("GET", "Observation"),
       "fhir_medication_request_search": ("GET", "MedicationRequest"), "fhir_vitals_search": ("GET", "Observation"),
       "fhir_procedure_search": ("GET", "Procedure"), "fhir_condition_search": ("GET", "Condition"),
       "fhir_medication_request_discontinue": ("POST", "MedicationRequest")}


def history(trace, base):
    """The upstream wrapper history (`_to_task_result`) of a trace. A call enters the history only if it ran without
    error, so a write that the server or the schema refused, which changed nothing, is not counted as an action. A
    discontinuation shows the payload that the tool posted."""
    wrapper = tools.upstream("wrapper")
    items = []
    for step in trace:
        if step["type"] == "message":
            items.append(wrapper.ChatHistoryItem(role="agent", content=step["content"]))
        elif step["type"] == "tool_call" and step["ok"]:
            if step["name"] not in API:
                items.append(wrapper.ChatHistoryItem(role="agent", content=f"{step['name']}({step['arguments']})"))
                continue
            method, resource = API[step["name"]]
            # The runs D-dev-100-criteria to -3 had a `reason` argument that the server never received
            # (`tools.FIXES`: `write_criteria`); it stays out of the history when those attempts are scored again.
            sent = step["output"]["posted"] if step["name"] == "fhir_medication_request_discontinue" else \
                {k: v for k, v in step["arguments"].items() if k != "reason"}
            items += [wrapper.ChatHistoryItem(role="agent", content=f"{method} {urljoin(base, resource)}\n"
                                                                    f"{json.dumps(sent)}"),
                      wrapper.ChatHistoryItem(role="agent", content=f"{method} request accepted")]
    return wrapper.TaskResult(result=json.dumps(attempt_value(trace)), history=items)


def attempt_value(trace):
    finished = [s for s in trace if s["type"] == "tool_call" and s["name"] == "finish" and s["ok"]]
    return finished[-1]["output"] if finished else []


# Arms A to D: 16 steps of work and the review step before the end (agent.review).
MAX_STEPS, ORIGINAL_MAX_STEPS = 17, 8
SCORING = threading.Lock()
# Captured once: a capture inside `score` before the lock could take another thread's revised replacement.
HAS_POST = tools.upstream(SCORER.removeprefix("src.")).check_has_post


def score(task, trace, base, pinned, revised=False):
    """True, False, or the error text of the scorer. `pinned` replaces the scorer's machine clock (family 1) by the
    reference time of the task context; `revised` uses the revised scorer (`revised_scorer`)."""
    scorer = tools.upstream(SCORER.removeprefix("src."))
    clock, has_post = datetime.datetime, HAS_POST  # the scorer's `from datetime import datetime`

    class Pinned(clock):
        @classmethod
        def now(cls, tz=None):
            return scorer.extract_now_from_context(task)

    family = task["id"].split("_")[0]
    with SCORING:  # the clock and `check_has_post` are module globals of the scorer
        scorer.datetime = Pinned if pinned else clock
        scorer.check_has_post = revised_scorer.check_has_post(scorer) if revised else has_post
        try:
            function = revised_scorer.family(scorer, family) if revised else getattr(scorer, family)
            return bool(function(task, history(trace, base), base))
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        finally:
            scorer.datetime, scorer.check_has_post = clock, has_post


OUTCOMES = {"passed": "scorer_error", "passed_upstream": "scorer_error_upstream",
            "passed_revised": "scorer_error_revised"}


def scores(original, trace, base):
    """The primary outcome (corrected task, pinned clock), the upstream outcome (original task, machine clock) and the
    revised outcome (corrected task, pinned clock, revised scorer)."""
    task = corrected(original)
    results = {"passed": score(task, trace, base, True), "passed_upstream": score(original, trace, base, False),
               "passed_revised": score(task, trace, base, True, revised=True)}
    return {**{k: v if isinstance(v, bool) else None for k, v in results.items()},
            **{OUTCOMES[k]: v for k, v in results.items() if isinstance(v, str)}}


def delete_creations(trace, base):
    """Delete the resources that the attempt created and check that each one is gone (HTTP 410)."""
    receipts = []
    for step in trace:
        if step["type"] == "tool_call" and step["ok"] and API.get(step["name"], ("",))[0] == "POST":
            created = step["output"].get("created", step["output"])
            path = f"{created['resourceType']}/{created['id']}"
            deleted = requests.delete(urljoin(base, path), timeout=60).status_code
            receipts.append({"resource": path, "delete": deleted, "after": requests.get(urljoin(base, path),
                                                                                         timeout=60).status_code})
    if any(r["after"] != 410 for r in receipts):
        raise RuntimeError(f"Created resources remain on the FHIR server: {receipts}")
    return receipts


def summarize(tasks, attempts):
    families = defaultdict(list)
    for tid in tasks:
        families[tid.split("_")[0]].append(attempts[tid]["passed"])
    for tid in tasks:
        families[tid.split("_")[0] + " upstream"].append(attempts[tid].get("passed_upstream"))
        families[tid.split("_")[0] + " revised"].append(attempts[tid].get("passed_revised"))
    passed = [a["passed"] for a in attempts.values()]
    # An attempt saved before an outcome existed (a resumed older run) is "not_scored", not a scorer error.
    other = {name: [a.get(f"passed_{name}", "absent") for a in attempts.values()] for name in ("upstream", "revised")}
    return {"attempts": len(attempts), "passed": passed.count(True), "failed": passed.count(False),
            "scorer_errors": passed.count(None), "pass_rate": passed.count(True) / len(passed) if passed else None,
            **{name: {"passed": v.count(True), "failed": v.count(False), "scorer_errors": v.count(None),
                      "not_scored": v.count("absent")} for name, v in other.items()},
            "by_family": {f: {"passed": v.count(True), "attempts": len(v)}
                          for f, v in sorted(families.items(),
                                             key=lambda kv: (int(kv[0].split()[0].removeprefix("task")), kv[0]))},
            "status": dict(Counter(a["status"] for a in attempts.values())),
            "tool_errors": sum(a["tool_errors"] for a in attempts.values()),
            "model_calls": sum(len(a["calls"]) for a in attempts.values()),
            "cost_usd": sum((c.get("usage") or {}).get("cost") or 0 for a in attempts.values() for c in a["calls"])}


def source_sha256():
    """{file name: SHA-256} of the study's source files and their combined digest (`results/design-lock/lock.json`
    holds the locked one)."""
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"files": files, "combined": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(agent.ARMS))
    p.add_argument("--part", choices=("dev", "validate", "test"), help="part of the split (default: all tasks)")
    p.add_argument("--tasks", nargs="+", help="task ids inside --part (default: the whole part)")
    p.add_argument("--run-id", help="output folder name under results/ (default: <arm>-<part>-<UTC time>)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--max-steps", type=int, help=f"default {MAX_STEPS}; the original-harness baseline "
                                                 f"{ORIGINAL_MAX_STEPS}, as upstream")
    p.add_argument("--model", choices=sorted(MODELS), default=MODEL["name"])
    p.add_argument("--reasoning-effort", default="medium")
    return p.parse_args(argv)


def main(argv=None, client=None):
    """`client` replaces the model transport (offline checks); it must have `responses.create(**kw)`."""
    args = parse(argv)
    # The upstream scorer prints agent text; a console code page (cp949) that cannot encode it raised inside the scorer.
    sys.stdout.reconfigure(errors="backslashreplace")
    pool = load_tasks(args.part)
    ids = sorted(args.tasks or pool, key=lambda t: [int(n) for n in t.removeprefix("task").split("_")])
    if missing := [t for t in ids if t not in pool]:
        raise SystemExit(f"Not in part {args.part}: {missing}")
    run_id = args.run_id or f"{args.arm}-{args.part or 'all'}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out = RESULTS / run_id
    original = agent.ARMS[args.arm]["environment"] == "original"
    corrects = args.arm != "original"
    max_steps = args.max_steps or (ORIGINAL_MAX_STEPS if original else MAX_STEPS)
    config = {"arm": args.arm, **agent.ARMS[args.arm], "part": args.part, "tasks": ids, "model": MODELS[args.model],
              "reasoning_effort": args.reasoning_effort, "max_steps": max_steps,
              "system_prompt": agent.ORIGINAL_PROMPT if original else agent.SYSTEM_PROMPT,
              "tool_fixes": {} if original else tools.FIXES,
              "benchmark_corrections": CORRECTIONS if corrects else {}, "scorer": SCORER, "fhir_image": FHIR_IMAGE,
              "source_sha256": source_sha256(),
              "client": "openrouter_transport" if client is None else type(client).__name__}
    if config["instruction"] == "L2":
        config["goal"] = {"template": agent.GOAL, "topics": agent.GOALS}
    if config["environment"] == "E1":
        config["institution"] = {"terminology": institution.TERMINOLOGY, "order_catalog": institution.ORDER_CATALOG,
                                 "protocols": institution.PROTOCOLS}
    lock = RESULTS / "design-lock" / "lock.json"
    if args.part in ("validate", "test") and lock.exists() and \
            json.loads(lock.read_text(encoding="utf-8"))["source_sha256"]["combined"] != \
            config["source_sha256"]["combined"]:
        raise SystemExit(f"The source differs from the locked design ({lock}); validate and test run the locked code")
    if (out / "config.json").exists():
        if not args.resume:
            raise SystemExit(f"{out} exists; pass --resume to continue it")
        saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "commands"} != config:
            raise SystemExit("--resume needs the parameters of the saved run")
        config["commands"] = saved["commands"]
    (out / "attempts").mkdir(parents=True, exist_ok=True)
    config["commands"] = config.get("commands", []) + [{"argv": shlex.join([sys.executable, "-m", __spec__.name,
                                                                             *(argv or sys.argv[1:])]),
                                                         "started_at_unix": time.time()}]
    atomic_json(out / "config.json", config)

    if client is None:
        client = OpenRouterChat(MODELS[args.model])
    todo = [t for t in ids if not (out / "attempts" / f"{t}.json").exists()]
    tasks = load_tasks()
    if todo:
        workers = FhirWorkers(FHIR_IMAGE, min(args.workers, len(todo)), run_id, out)
        try:
            endpoints = queue.Queue()
            for endpoint in workers.start():
                endpoints.put(endpoint)

            def attempt(tid):
                endpoint = endpoints.get()
                try:
                    base = endpoint.rstrip("/") + "/"
                    started = time.time()
                    given = corrected(tasks[tid]) if corrects else tasks[tid]
                    result = agent.run(client, given, args.arm, endpoint, model=MODELS[args.model],
                                       effort=args.reasoning_effort, max_steps=max_steps)
                    record = {"task": tid, **scores(tasks[tid], result["trace"], base), **result,
                              "tool_errors": sum(s["type"] == "tool_call" and not s["ok"] for s in result["trace"]),
                              "seconds": round(time.time() - started, 1),
                              "deleted": delete_creations(result["trace"], base)}
                    atomic_json(out / "attempts" / f"{tid}.json", record)
                    print(f"{tid} {record['status']} passed={record['passed']}", flush=True)
                finally:
                    endpoints.put(endpoint)

            with ThreadPoolExecutor(min(args.workers, len(todo))) as executor:
                list(executor.map(attempt, todo))
        finally:
            workers.close()
    attempts = {t: json.loads((out / "attempts" / f"{t}.json").read_text(encoding="utf-8")) for t in ids}
    summary = summarize(ids, attempts)
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    main()
