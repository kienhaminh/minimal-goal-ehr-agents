"""Check the status-changing discontinuation tool and the final-state evaluator on a disposable FHIR container.

No model is called. The script starts one container of the study's pinned FHIR image, runs the scenarios below with
the tools called directly, reads the patient's records before and after each scenario, judges each with the final-state
evaluator, checks every expectation, restores the baseline, and removes the container. It also checks that the
source in `src/` still has the locked digest, before and after the tool is installed in memory.

    python -X utf8 benchmark_fixes/verify.py

Scenarios: one active order discontinued; selective discontinuation among several orders; repeated calls and an order
that was already stopped; an unknown id and an order of another patient; the wrong medication; the order that should
remain; the QTc family with and without a prolonged QTc; and the study tool on the same request, for contrast.
Output: results/fhir-correct-v1/verification.json (calls, before and after resources, evaluations and checks).
"""
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clinical_agent import run, tools  # noqa: E402
from clinical_agent.fhir_workers import FhirWorkers, atomic_json  # noqa: E402
from clinical_agent.settings import FHIR_IMAGE, RESULTS  # noqa: E402

import discontinue  # noqa: E402
import final_state as fs  # noqa: E402

OUT = RESULTS / discontinue.VERSION
LOCK = RESULTS / "design-lock" / "lock.json"
ECG_ORDER = {"resourceType": "ServiceRequest", "code": {"coding": [{"system": "http://snomed.info/sct",
                                                                    "code": "445118002", "display": "12-lead ECG"}]},
             "authoredOn": "2023-11-13T12:00:00+00:00", "status": "active", "intent": "order", "priority": "routine",
             "subject": {"reference": "Patient/S0869531"}, "note": {"text": "Follow-up ECG for prolonged QTc."},
             "occurrenceDateTime": "2023-11-14T12:00:00+00:00"}
checks = []


def check(scenario, name, condition, detail=None):
    checks.append({"scenario": scenario, "check": name, "passed": bool(condition), "detail": detail})
    print(("PASS " if condition else "FAIL ") + f"[{scenario}] {name}")


def call(tool, **arguments):
    step = {"tool": tool.name, "arguments": arguments}
    try:
        step.update(ok=True, output=tool.process(tool.input_schema.model_validate(arguments)))
    except Exception as error:  # the agent loop shows a failed call as its error text
        step.update(ok=False, output=f"{type(error).__name__}: {error}")
    return step


def raw(base, route, rid):
    r = requests.get(f"{base}/{route}/{rid}", params={"_format": "json"}, headers=fs.NO_CACHE, timeout=60)
    return r.json() if r.ok else {"http_status": r.status_code}


def active(base, patient, rid):
    return discontinue.listed_active(base, rid, patient)


def locked():
    return json.loads(LOCK.read_text(encoding="utf-8"))["source_sha256"]["combined"] == \
        run.source_sha256()["combined"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_factory = tools.medication_request_discontinue
    evidence = {"version": discontinue.VERSION, "fhir_image": FHIR_IMAGE, "scenarios": [], "checks": checks}
    check("frozen harness", "source digest equals the locked design before the corrected tool is installed", locked())
    workers = FhirWorkers(FHIR_IMAGE, 1, discontinue.VERSION + "-verify", OUT / "fhir-workers")
    try:
        base = workers.start()[0].rstrip("/")
        frozen = frozen_factory(base)
        discontinue.apply()
        check("frozen harness", "source digest still equals the locked design with the corrected tool in memory",
              locked())
        e0 = tools.e0(base)
        stop, order = e0[discontinue.NAME], e0["fhir_service_request_create"]
        check("tool set", "E0 builds the corrected tool and description",
              stop.description == discontinue.DESCRIPTION and "patient" in stop.input_schema.model_fields)

        def scenario(name, patient, family, actions, readings=(None,), others=()):
            before = fs.read_state(base, patient)
            others_before = {p: fs.read_state(base, p) for p in others}
            steps = [call(tool, **arguments) for tool, arguments in actions]
            after = fs.read_state(base, patient)
            evaluations = {r or "default": fs.evaluate(family, before, after, r) for r in readings}
            stopped, unrelated, created = fs.changes(before, after)
            touched = sorted({str(a.get("id")) for _, a in actions if a.get("id")} | set(stopped))
            record = {"scenario": name, "patient": patient, "family": family, "calls": steps,
                      "before": {i: before["MedicationRequest"].get(i) for i in touched},
                      "after": {i: raw(base, "MedicationRequest", i) for i in touched},
                      "created": created, "unrelated_changes": unrelated,
                      "medication_requests": {"before": len(before["MedicationRequest"]),
                                              "after": len(after["MedicationRequest"])},
                      "evaluations": evaluations,
                      "other_patients_unchanged": {p: fs.read_state(base, p) == s for p, s in others_before.items()}}
            fs.restore(base, before)
            record["baseline_restored"] = True
            evidence["scenarios"].append(record)
            return record

        # 1. task2_10 (S3032536): heparin 5000 785576 and the newer heparin infusion 785589.
        s = scenario("one active order", "S3032536", "task2", [(stop, {"id": "785589", "patient": "S3032536"})],
                     readings=("either", "heparin", "newest"))
        n = "one active order"
        check(n, "call succeeds with result 'discontinued'", s["calls"][0]["ok"]
              and s["calls"][0]["output"]["result"] == "discontinued", s["calls"][0]["output"])
        check(n, "original order now has status 'stopped'", s["after"]["785589"]["status"] == "stopped")
        check(n, "all other fields of the order are kept",
              {k: v for k, v in fs.body(s["after"]["785589"]).items() if k != "status"}
              == {k: v for k, v in s["before"]["785589"].items() if k != "status"})
        check(n, "the order has a new version", s["after"]["785589"]["meta"]["versionId"] != "1",
              s["after"]["785589"]["meta"])
        check(n, "no MedicationRequest was added", s["medication_requests"]["before"] == s["medication_requests"]["after"],
              s["medication_requests"])
        check(n, "no other order changed", s["unrelated_changes"] == [], s["unrelated_changes"])
        check(n, "evaluator passes: one active anticoagulant order remains (heparin 5000), readings 'either' and "
                 "'heparin'", s["evaluations"]["either"]["passed"] and s["evaluations"]["heparin"]["passed"])
        check(n, "evaluator fails the 'newest' reading, which keeps the newer infusion",
              s["evaluations"]["newest"]["passed"] is False, s["evaluations"]["newest"])

        # The active-order search, checked on a live discontinuation before the restore.
        before = fs.read_state(base, "S3032536")
        call(stop, id="785589", patient="S3032536")
        check(n, "active-order search no longer returns the order", not active(base, "S3032536", "785589"))
        check(n, "active-order search still returns the order that should remain", active(base, "S3032536", "785576"))
        fs.restore(base, before)
        check(n, "restore makes the order active again", active(base, "S3032536", "785589"))

        # 2. task2_14 (S6539215): three heparin 5000 orders; the newest, 770520, should remain.
        n = "selective discontinuation"
        s = scenario(n, "S6539215", "task2", [(stop, {"id": "770507", "patient": "S6539215"}),
                                              (stop, {"id": "770524", "patient": "Patient/S6539215"})],
                     readings=("either", "heparin", "newest"))
        check(n, "both calls succeed", all(c["ok"] and c["output"]["result"] == "discontinued" for c in s["calls"]))
        check(n, "both targets are stopped", all(s["after"][i]["status"] == "stopped" for i in ("770507", "770524")))
        check(n, "the newest order 770520 stays active and unchanged", "770520" not in s["after"]
              and s["evaluations"]["either"]["active_anticoagulant_orders_after"][0].startswith("770520"),
              s["evaluations"]["either"]["active_anticoagulant_orders_after"])
        check(n, "evaluator passes under all three readings",
              all(e["passed"] for e in s["evaluations"].values()), s["evaluations"])

        # 3. Repeated calls and an order that was already stopped at baseline.
        n = "already stopped and repeated"
        state = fs.read_state(base, "S6539215")
        stopped_before = sorted(i for i, m in state["MedicationRequest"].items() if m.get("status") == "stopped")
        actions = [(stop, {"id": "770507", "patient": "S6539215"}), (stop, {"id": "770507", "patient": "S6539215"})]
        version = int(raw(base, "MedicationRequest", "770507")["meta"]["versionId"])
        if stopped_before:
            actions.append((stop, {"id": stopped_before[0], "patient": "S6539215"}))
        s = scenario(n, "S6539215", "task2", actions)
        first, second = s["calls"][0], s["calls"][1]
        check(n, "first call discontinues", first["ok"] and first["output"]["result"] == "discontinued")
        check(n, "repeated call reports 'already stopped' and writes nothing",
              second["ok"] and second["output"]["result"] == "already stopped"
              and int(s["after"]["770507"]["meta"]["versionId"]) == version + 1,
              {"version_before": version, "after": s["after"]["770507"]["meta"]})
        check(n, "no MedicationRequest was added", s["medication_requests"]["before"] == s["medication_requests"]["after"],
              s["medication_requests"])
        if stopped_before:
            third = s["calls"][2]
            check(n, "an order stopped at baseline reports 'already stopped' and keeps its version",
                  third["ok"] and third["output"]["result"] == "already stopped"
                  and s["after"][stopped_before[0]]["meta"]["versionId"] == "1",
                  {"id": stopped_before[0], "meta": s["after"][stopped_before[0]]["meta"]})
        else:
            check(n, "patient has no order stopped at baseline (case not available)", True, "S6539215")

        # 4. Unknown id and an order of another patient.
        n = "wrong id and patient mismatch"
        s = scenario(n, "S3032536", "task2", [(stop, {"id": "999999999", "patient": "S3032536"}),
                                              (stop, {"id": "770520", "patient": "S3032536"})],
                     others=("S6539215",))
        check(n, "unknown id fails with an explicit error", not s["calls"][0]["ok"]
              and "No medication order" in s["calls"][0]["output"], s["calls"][0]["output"])
        check(n, "order of another patient fails with an explicit error", not s["calls"][1]["ok"]
              and "belongs to Patient/S6539215" in s["calls"][1]["output"], s["calls"][1]["output"])
        check(n, "no record of either patient changed", s["unrelated_changes"] == [] and not s["created"]["MedicationRequest"]
              and all(s["other_patients_unchanged"].values()) and s["after"]["770520"]["status"] == "active",
              s["other_patients_unchanged"])

        # 5. The wrong medication: ampicillin-sulbactam 785572 instead of an anticoagulant.
        n = "wrong medication"
        s = scenario(n, "S3032536", "task2", [(stop, {"id": "785572", "patient": "S3032536"})])
        e = s["evaluations"]["default"]
        check(n, "tool stops the named order (it does not judge the protocol)", s["calls"][0]["ok"])
        check(n, "evaluator fails: stop outside the protocol and two anticoagulant orders remain",
              e["passed"] is False and any("outside the protocol" in p for p in e["problems"])
              and len(e["active_anticoagulant_orders_after"]) == 2, e)

        # 6. The order that should remain: stopping both anticoagulant orders leaves none.
        n = "order that should remain"
        s = scenario(n, "S3032536", "task2", [(stop, {"id": "785576", "patient": "S3032536"}),
                                              (stop, {"id": "785589", "patient": "S3032536"})])
        e = s["evaluations"]["default"]
        check(n, "evaluator fails when no active anticoagulant order remains",
              e["passed"] is False and e["active_anticoagulant_orders_after"] == [], e)

        # 7. task7_12 (S0869531): QTc 565 ms, 19 active listed QT-prolonging orders.
        n = "QTc prolonged"
        s = scenario(n, "S0869531", "task7", [(order, ECG_ORDER), (stop, {"id": "771801", "patient": "S0869531"})],
                     readings=("one", "all"))
        check(n, "ECG order and one discontinuation succeed", all(c["ok"] for c in s["calls"]))
        check(n, "evaluator passes the 'one' reading", s["evaluations"]["one"]["passed"], s["evaluations"]["one"])
        check(n, "evaluator fails the 'all' reading (18 listed orders stay active)",
              s["evaluations"]["all"]["passed"] is False and s["evaluations"]["all"]["listed_active_after"] == 18,
              s["evaluations"]["all"])

        # 8. task7_5 (S2090974): QTc 428 ms, no action is due.
        n = "QTc not prolonged"
        s = scenario(n, "S2090974", "task7", [])
        check(n, "evaluator passes an unchanged chart", s["evaluations"]["default"]["passed"])
        s = scenario(n, "S2090974", "task7", [(stop, {"id": "785662", "patient": "S2090974"})])
        check(n, "evaluator fails a discontinuation on a no-action task",
              s["evaluations"]["default"]["passed"] is False, s["evaluations"]["default"])

        # 9. The frozen tool on the request of scenario 1.
        n = "frozen tool"
        s = scenario(n, "S3032536", "task2", [(frozen, {"id": "785589"})])
        e = s["evaluations"]["default"]
        check(n, "frozen tool leaves the original order active", s["after"]["785589"]["status"] == "active")
        check(n, "frozen tool adds a 'stopped' MedicationRequest",
              s["medication_requests"]["after"] == s["medication_requests"]["before"] + 1
              and s["created"]["MedicationRequest"][0]["status"] == "stopped", s["created"])
        check(n, "evaluator fails it: two anticoagulant orders remain and a record was added",
              e["passed"] is False and len(e["active_anticoagulant_orders_after"]) == 2, e)
    finally:
        workers.close()
    atomic_json(OUT / "verification.json", evidence)
    failed = [c for c in checks if not c["passed"]]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
