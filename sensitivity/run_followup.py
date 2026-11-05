"""Run one arm under a follow-up sensitivity condition (`results/followup-lock/plan.json`).

The locked source is not edited: this script replaces, in memory, either the terminology entries and the terminology
search description of `clinical_agent.institution` (S5_no_record_locations: each entry without its "Records:"
sentence, the resource type and search field that the locked design added) or the goal sentence of
`clinical_agent.agent` (S6_goal_paraphrase: one reworded goal per task family), and then calls
`clinical_agent.run.main` with the remaining arguments. `followup.json` (next to this script) holds the variant
entries and goals. The run's config.json records what the agent saw; this script adds sensitivity.json with the
condition and the SHA-256 of this script and of followup.json. A test run starts only when both equal the locked
digests in results/followup-lock/plan.json.

    python sensitivity/run_followup.py --condition S5_no_record_locations --arm D --part test \
        --run-id D-test-150-s5-rep1 --workers 6
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from clinical_agent import agent, institution, run
from clinical_agent.fhir_workers import atomic_json
from clinical_agent.settings import RESULTS

HERE = Path(__file__).parent
VARIANTS = HERE / "followup.json"
CONDITIONS = ("S5_no_record_locations", "S6_goal_paraphrase")


def apply(condition, variants):
    """Replace the terminology entries or the goal sentence for the condition."""
    v = variants[condition]
    if condition == "S5_no_record_locations":
        institution.TERMINOLOGY = v["terminology"]
        institution.DESCRIPTIONS["terminology_search"] = v["terminology_description"]
    else:
        agent.GOAL, agent.GOALS = v["template"], v["topics"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    args, rest = p.parse_known_args()
    digests = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__), VARIANTS)}
    if "test" in rest:
        lock = json.loads((RESULTS / "followup-lock" / "plan.json").read_text(encoding="utf-8"))["sha256"]
        if {k.split("/")[-1]: v for k, v in lock.items() if k.startswith("sensitivity/")} != digests:
            raise SystemExit("followup.json or this script differs from results/followup-lock/plan.json; test runs "
                             "the locked files")
    apply(args.condition, json.loads(VARIANTS.read_text(encoding="utf-8")))
    run.main(rest)
    run_id = rest[rest.index("--run-id") + 1]
    atomic_json(RESULTS / run_id / "sensitivity.json", {"condition": args.condition, "argv": sys.argv,
                                                        "sha256": digests})


if __name__ == "__main__":
    main()
