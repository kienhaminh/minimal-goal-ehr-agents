"""Run one arm under a sensitivity condition of the institutional libraries (`results/sensitivity-lock/plan.json`).

The locked source is not edited: this script replaces the library entries of `clinical_agent.institution` in memory
and then calls `clinical_agent.run.main` with the remaining arguments. `libraries.json` (next to this script) holds
the variant entries. Each condition's entries are shuffled in one fixed order (seed 1), so that the target entries do
not always come first when search scores tie. The run's config.json records the entries the agent saw; this script
adds sensitivity.json with the condition and the SHA-256 of this script and of libraries.json. A test run starts only
when both equal the locked digests in results/sensitivity-lock/plan.json.

    python sensitivity/run_sensitivity.py --condition S2_realistic --arm D --part dev --run-id D-dev-100-s2 --workers 6
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from clinical_agent import institution, run
from clinical_agent.fhir_workers import atomic_json
from clinical_agent.settings import RESULTS

HERE = Path(__file__).parent
LIBRARIES = HERE / "libraries.json"
CONDITIONS = ("S1_paraphrase", "S2_realistic", "S3_stress", "S4_no_protocols")


def shuffled(entries):
    entries = list(entries)
    random.Random(1).shuffle(entries)
    return entries


def apply(condition, lib):
    """Replace the library entries of `institution` for the condition."""
    if condition == "S4_no_protocols":
        locked_e1 = institution.e1
        institution.PROTOCOLS = []
        institution.e1 = lambda endpoint, task: {k: v for k, v in locked_e1(endpoint, task).items()
                                                 if k != "protocol_search"}
        return
    protocols, terminology, catalog = lib["paraphrased_protocols"], [], []
    if condition in ("S2_realistic", "S3_stress"):
        protocols += lib["unrelated_protocols"]
        terminology += lib["unrelated_terminology"]
        catalog += lib["unrelated_catalog"]
    if condition == "S3_stress":
        protocols += lib["near_miss_protocols"]
        terminology += lib["near_miss_terminology"]
        catalog += lib["near_miss_catalog"]
    institution.PROTOCOLS = shuffled(protocols)
    institution.TERMINOLOGY = shuffled(institution.TERMINOLOGY + terminology) if terminology else institution.TERMINOLOGY
    institution.ORDER_CATALOG = shuffled(institution.ORDER_CATALOG + catalog) if catalog else institution.ORDER_CATALOG


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    args, rest = p.parse_known_args()
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), LIBRARIES)}
    lock = json.loads((RESULTS / "sensitivity-lock" / "plan.json").read_text(encoding="utf-8")).get("sha256", {})
    if "test" in rest and {k.split("/")[-1]: v for k, v in lock.items()} | digests != \
            {k.split("/")[-1]: v for k, v in lock.items()}:
        raise SystemExit("libraries.json or this script differs from results/sensitivity-lock/plan.json; test runs the "
                         "locked files")
    apply(args.condition, json.loads(LIBRARIES.read_text(encoding="utf-8")))
    run.main(rest)
    run_id = rest[rest.index("--run-id") + 1]
    atomic_json(RESULTS / run_id / "sensitivity.json", {
        "condition": args.condition, "argv": sys.argv,
        "sha256": digests})


if __name__ == "__main__":
    main()
