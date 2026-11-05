"""MedAgentBench v2 tasks and the study split.

The 300 new-patient tasks (10 families x 30) of the upstream benchmark, scored by the upstream v2 scorer
(`medagentbenchevals/new_refsol.py`). The split is the one of level-up-jev v13 (same seed and rule), so the two studies
share their held-out test tasks: per family 10 dev, 5 validate and 15 test tasks; `dev` 100, `validate` 50, `test` 150.
"""

import json
import random

from clinical_agent.settings import BENCHMARKS

UPSTREAM = BENCHMARKS / "MedAgentBench-v2/medagentbench_v2"
TASKS = UPSTREAM / "src/MedAgentBench/data/medagentbench/new_patient_tasks.json"
SCORER = "src.medagentbenchevals.new_refsol"
SPLIT_SEED = 20261026
PER_FAMILY = {"dev": 10, "validate": 5}  # the rest of each family is test
PARTS = ("dev", "validate", "test")

# Benchmark corrections (user decisions 2026-10-27), the same in every arm: the task states what the scorer needs and
# the task text omits.
# - Time of families 1 and 5: the scorer reads it from no task text (family 1: the machine clock, pinned in run.py to
#   the task's time; family 5: the default of `extract_now_from_context`, which is earlier than the latest records of
#   all 60 patients). REFERENCE_TIME follows the latest record of those patients (2023-11-13T09:00).
# - Families 1, 5, 9 and 10: the scorer counts a patient with no prior study or vaccine as due.
# - Family 3: the scorer reads the answer as two numbers.
# - Time of families 2, 6, 7 and 8 (user decision 2026-10-28, from the runs after `D-dev-100-tools`): the task states no
#   time, so agents signed orders with dates of their own (2025-01-01, 2026-09-29: 10 of 47 orders in D-dev-100-tools).
#   The scorer does not read the time; the latest record of these 120 patients is 2023-11-13T09:00, before
#   REFERENCE_TIME.
# Each run records CORRECTIONS in its config.json and is scored also on the original tasks.
REFERENCE_TIME = "2023-11-13T12:00:00+00:00"
NOW = f"It's {REFERENCE_TIME} now."
CORRECTIONS = {
    "task1": f"{NOW} A patient with no recorded CT Abdomen study is due for a new one.",
    "task2": NOW,
    "task3": "Answer with the two averages as numbers, [6-hour average, 12-hour average]; use -1 for a period that "
             "has no heart rate value.",
    "task5": f"{NOW} A patient with no recorded CT Abdomen/Pelvis is due for a new one.",
    "task6": NOW,
    "task7": NOW,
    "task8": NOW,
    "task9": "A patient with no recorded influenza vaccine is due for a new one.",
    "task10": "A patient with no recorded COVID-19 vaccine is due for a booster.",
}


def split(tasks):
    """{task id: "dev", "validate" or "test"}. Each family's sorted ids are shuffled with one Random(SPLIT_SEED),
    families in order; the first 10 are dev, the next 5 validate, the rest test."""
    families = {}
    for tid in sorted(tasks, key=lambda t: [int(n) for n in t.removeprefix("task").split("_")]):
        families.setdefault(tid.split("_")[0], []).append(tid)
    rng = random.Random(SPLIT_SEED)
    part = {}
    for family in sorted(families, key=lambda f: int(f.removeprefix("task"))):
        ids = families[family]
        rng.shuffle(ids)
        dev, validate = PER_FAMILY["dev"], PER_FAMILY["dev"] + PER_FAMILY["validate"]
        part.update({tid: "dev" if i < dev else "validate" if i < validate else "test" for i, tid in enumerate(ids)})
    return part


def load_tasks(part=None):
    """{task id: task}, all 300 or one part of the split."""
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text(encoding="utf-8"))}
    if part is None:
        return tasks
    if part not in PARTS:
        raise ValueError("part must be one of: " + ", ".join(PARTS))
    parts = split(tasks)
    return {tid: task for tid, task in tasks.items() if parts[tid] == part}


def corrected(task):
    """The task with its benchmark correction appended to the context."""
    extra = CORRECTIONS.get(task["id"].split("_")[0])
    if not extra:
        return task
    context = (task.get("context") or "").rstrip()
    if context and not context.rstrip("'\"").endswith((".", "!", "?")):
        context += "."
    return {**task, "context": f"{context} {extra}".lstrip()}
