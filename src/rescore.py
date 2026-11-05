"""Score the saved attempts of finished runs again, offline, with the upstream and the revised scorer (`run.scores`).

    python -m clinical_agent.rescore --runs A-dev-100 D-dev-100-v2 --out rescore-revised-scorer

No model is called. One fresh FHIR container serves all runs: each run deleted the resources its attempts created, so
the container holds the records that the attempts saw. Each run folder gets rescore.json (the outcomes of every task,
the outcomes saved by the run, and the summary with the three outcomes). results/<out>/ gets config.json (the command),
comparison.json (per run: passes under each scorer and the tasks whose outcome changed) and the container receipt.
"""
import argparse
import json
import shlex
import sys
import time

from clinical_agent import revised_scorer, run
from clinical_agent.fhir_workers import FhirWorkers, atomic_json
from clinical_agent.settings import FHIR_IMAGE, RESULTS
from clinical_agent.tasks import load_tasks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders under results/")
    p.add_argument("--out", required=True, help="folder under results/ for the comparison and the container receipt")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")
    out = RESULTS / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    atomic_json(out / "config.json", {"argv": shlex.join([sys.executable, "-m", __spec__.name,
                                                          *(argv or sys.argv[1:])]),
                                      "started_at_unix": time.time(), "runs": args.runs,
                                      "revised_scorer": revised_scorer.__doc__})
    tasks = load_tasks()
    comparison = {}
    workers = FhirWorkers(FHIR_IMAGE, 1, args.out, out)
    try:
        base = workers.start()[0].rstrip("/") + "/"
        for run_id in args.runs:
            folder = RESULTS / run_id
            ids = json.loads((folder / "config.json").read_text(encoding="utf-8"))["tasks"]
            attempts = {t: json.loads((folder / "attempts" / f"{t}.json").read_text(encoding="utf-8")) for t in ids}
            rescored = {t: run.scores(tasks[t], a["trace"], base) for t, a in attempts.items()}
            saved = {t: {k: a.get(k) for k in ("passed", "passed_upstream")} for t, a in attempts.items()}
            kept = {t: {k: v for k, v in a.items() if not k.startswith("scorer_error")} for t, a in attempts.items()}
            summary = run.summarize(ids, {t: {**kept[t], **rescored[t]} for t in ids})
            atomic_json(folder / "rescore.json", {"tasks": {t: {**rescored[t], "saved": saved[t]} for t in ids},
                                                  "summary": summary})
            changed = {t: r["passed_revised"] for t, r in rescored.items()
                       if r["passed"] is not None and r["passed_revised"] is not None
                       and r["passed"] != r["passed_revised"]}
            comparison[run_id] = {
                "attempts": len(ids),
                **{k: sum(r[k] is True for r in rescored.values()) for k in run.OUTCOMES},
                "saved_passed": sum(s["passed"] is True for s in saved.values()),
                # passed_upstream of family 1 reads the machine clock, so it can differ from the saved value
                "not_reproduced": {t: [k for k in saved[t] if rescored[t][k] != saved[t][k]] for t in ids
                                   if any(rescored[t][k] != saved[t][k] for k in saved[t])},
                "revised_gains": sorted(t for t, v in changed.items() if v),
                "revised_losses": sorted(t for t, v in changed.items() if not v),
                "scorer_errors": {f"{t} {k}": e for t, r in rescored.items() for k, e in r.items()
                                  if k.startswith("scorer_error")}}
            print(run_id, json.dumps(comparison[run_id]), flush=True)
    finally:
        workers.close()
    atomic_json(out / "comparison.json", comparison)


if __name__ == "__main__":
    main()
