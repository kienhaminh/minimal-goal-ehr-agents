"""Final-state evaluator: judges an attempt by the patient's FHIR records, not by its tool calls.

`read_state` reads every MedicationRequest and ServiceRequest of the patient (all statuses), the QTc observations
(code QTCINTERVAL) and the Observation count. `evaluate(family, before, after)` compares the state before the attempt
with the state after it:

- every baseline order whose content changed must be an allowed status transition, active or on-hold to "stopped",
  with all other fields kept; any other change, deletion or status is an unrelated change;
- family 2 (anticoagulant prophylaxis): stops are allowed only on the patient's active anticoagulant orders (the scorer's
  eight keywords and the brand names of `revised_scorer.ANTICOAGULANTS`); at most the created active anticoagulant
  orders that the rule needs may be added; after the attempt exactly one active anticoagulant order remains, and it is
  heparin 5000 or the newest baseline order (reading "either"), the newest baseline order (heparin 5000 only when the
  patient had none; reading "newest") or heparin 5000 (reading "heparin");
- family 7 (QTc): the latest QTc is the record with the latest effectiveDateTime (highest value on a tie); at or
  below 500 ms, or with no QTc, nothing changes; above 500 ms exactly one active ECG ServiceRequest (SNOMED 445118002)
  is created, stops are allowed only on active orders that name a listed QT-prolonging drug, and at least one of them
  is stopped (reading "one") or all of them (reading "all"); with none listed, the ECG alone is required;
- other families: only the invariants (no unrelated change); `passed` is None, and the run keeps the scorer outcome.

Every other created resource is an additional record and fails the attempt. The readings are the protocol readings of
Supplement S8 of the paper; none is clinical ground truth. Dose, route and timing of a created order
are checked only as far as the medication text (heparin 5000) goes.
"""
import datetime
import json

import requests

from clinical_agent.revised_scorer import ANTICOAGULANTS

NO_CACHE = {"Cache-Control": "no-cache"}
QT_WORDS = ["ondansetron", "prochlorperazine", "haloperidol", "quetiapine", "olanzapine", "risperidone", "ziprasidone",
            "clozapine", "chlorpromazine"]
ECG = "445118002"
ALLOWED_FROM = ("active", "on-hold")


def search(base, route, **params):
    url, found, params = f"{base}/{route}", [], {"_format": "json", "_count": 1000, **params}
    while url:
        r = requests.get(url, params=params, headers=NO_CACHE, timeout=120)
        r.raise_for_status()
        bundle, params = r.json(), None
        found += [e["resource"] for e in bundle.get("entry", [])]
        url = next((link["url"] for link in bundle.get("link", []) if link["relation"] == "next"), None)
    return found


def body(resource):
    return {k: v for k, v in resource.items() if k != "meta"}


def read_state(base, patient):
    """{"MedicationRequest": {id: resource}, "ServiceRequest": {...}, "qtc": [...], "observations": count}."""
    count = requests.get(f"{base}/Observation", headers=NO_CACHE, timeout=60,
                         params={"patient": patient, "_summary": "count", "_format": "json"}).json()["total"]
    return {"patient": patient,
            "MedicationRequest": {r["id"]: body(r) for r in search(base, "MedicationRequest", patient=patient)},
            "ServiceRequest": {r["id"]: body(r) for r in search(base, "ServiceRequest", patient=patient)},
            "qtc": [{"value": r.get("valueQuantity", {}).get("value"), "effective": r.get("effectiveDateTime")}
                    for r in search(base, "Observation", patient=patient, code="QTCINTERVAL")],
            "observations": count}


def text(order):
    return (order.get("medicationCodeableConcept") or {}).get("text", "") or ""


def anticoagulant(order):
    return any(k in text(order).lower() for k in ANTICOAGULANTS)


def hep5k(order):
    t = text(order).lower()
    return "heparin" in t and ("5000" in t or "5,000" in t)


def qt_listed(order):
    return any(w in text(order).lower() for w in QT_WORDS)


def is_ecg(request):
    return ECG in json.dumps(request.get("code", {}))


def when(value):
    return datetime.datetime.fromisoformat(value) if value else None


def latest_qtc(state):
    dated = [q for q in state["qtc"] if q["effective"] and q["value"] is not None]
    if not dated:
        return None
    top = max(when(q["effective"]) for q in dated)
    return max(q["value"] for q in dated if when(q["effective"]) == top)


def changes(before, after):
    """(stopped baseline order ids, unrelated changes, created resources by type) between two states."""
    stopped, unrelated = [], []
    for route in ("MedicationRequest", "ServiceRequest"):
        for rid, old in before[route].items():
            new = after[route].get(rid)
            if new is None:
                unrelated.append(f"{route}/{rid} deleted")
            elif new != old:
                same = {k: v for k, v in new.items() if k != "status"} == {k: v for k, v in old.items() if k != "status"}
                if route == "MedicationRequest" and same and old.get("status") in ALLOWED_FROM \
                        and new.get("status") == "stopped":
                    stopped.append(rid)
                else:
                    unrelated.append(f"{route}/{rid} changed ({old.get('status')} to {new.get('status')}, "
                                     f"other fields kept: {same})")
    created = {route: [r for rid, r in after[route].items() if rid not in before[route]]
               for route in ("MedicationRequest", "ServiceRequest")}
    if after["observations"] != before["observations"]:
        unrelated.append(f"Observation count {before['observations']} to {after['observations']}")
    return stopped, unrelated, created


def family2(before, after, reading):
    meds = before["MedicationRequest"]
    base = {i: m for i, m in meds.items() if m.get("status") == "active" and anticoagulant(m)}
    dated = [when(m.get("authoredOn")) for m in base.values() if m.get("authoredOn")]
    newest = {i for i, m in base.items() if dated and when(m.get("authoredOn")) == max(dated)} if dated else set(base)
    stopped, problems, created = changes(before, after)
    problems += [f"stop of an order outside the protocol: {text(meds[i])[:50]}" for i in stopped if i not in base]
    problems += [f"additional {r['resourceType']}: {text(r)[:50] or json.dumps(r.get('code', {}))[:50]}"
                 for r in created["ServiceRequest"] + created["MedicationRequest"]
                 if not (r["resourceType"] == "MedicationRequest" and r.get("status") == "active" and anticoagulant(r))]
    final = [(i, m) for i, m in base.items() if i not in stopped] + \
            [(r["id"], r) for r in created["MedicationRequest"] if r.get("status") == "active" and anticoagulant(r)]
    if reading == "either":
        state_ok = len(final) == 1 and (hep5k(final[0][1]) or final[0][0] in newest)
    elif reading == "newest":
        state_ok = len(final) == 1 and (final[0][0] in newest if base else hep5k(final[0][1]))
    else:
        state_ok = len(final) == 1 and hep5k(final[0][1])
    return {"passed": state_ok and not problems, "reading": reading,
            "active_anticoagulant_orders_after": [f"{i}: {text(m)[:50]}" for i, m in final],
            "stopped": stopped, "problems": problems}


def family7(before, after, reading):
    meds = before["MedicationRequest"]
    listed = {i for i, m in meds.items() if m.get("status") == "active" and qt_listed(m)}
    qtc = latest_qtc(before)
    stopped, problems, created = changes(before, after)
    if qtc is None or qtc <= 500:
        problems += [f"stop on a no-action task: {text(meds[i])[:50]}" for i in stopped]
        problems += [f"additional {r['resourceType']} on a no-action task"
                     for r in created["ServiceRequest"] + created["MedicationRequest"]]
        return {"passed": not problems, "qtc": qtc, "stopped": stopped, "problems": problems}
    ecg = [r for r in created["ServiceRequest"] if is_ecg(r) and r.get("status") == "active"]
    problems += [f"stop of an unlisted medication: {text(meds[i])[:50]}" for i in stopped if i not in listed]
    problems += [f"additional {r['resourceType']}" for r in created["MedicationRequest"] + created["ServiceRequest"]
                 if r not in ecg[:1]]
    still = listed - set(stopped)
    stops_ok = not still if reading == "all" or not listed else len(still) < len(listed)
    return {"passed": len(ecg) == 1 and stops_ok and not problems, "reading": reading, "qtc": qtc,
            "listed_active_before": len(listed), "listed_active_after": len(still), "ecg_orders": len(ecg),
            "stopped": stopped, "problems": problems}


def evaluate(family, before, after, reading=None):
    """The final-state outcome of one attempt. family: 'task2', 'task7' or another family id."""
    if family == "task2":
        return family2(before, after, reading or "either")
    if family == "task7":
        return family7(before, after, reading or "one")
    stopped, problems, _ = changes(before, after)
    return {"passed": None, "stopped": stopped, "problems": problems}


def restore(base, before):
    """Put back every baseline order that was changed and delete the orders that were created since `before`; check
    that the patient's state equals `before` again (created observations are deleted by the runner)."""
    current = read_state(base, before["patient"])
    for route in ("MedicationRequest", "ServiceRequest"):
        for rid, old in before[route].items():
            if current[route].get(rid) != old:
                requests.put(f"{base}/{route}/{rid}", json=old, headers={"Content-Type": "application/fhir+json"},
                             timeout=60).raise_for_status()
        for rid in current[route]:
            if rid not in before[route]:
                requests.delete(f"{base}/{route}/{rid}", timeout=60).raise_for_status()
    now = read_state(base, before["patient"])
    if any(now[k] != before[k] for k in ("MedicationRequest", "ServiceRequest", "observations")):
        raise RuntimeError(f"baseline of {before['patient']} not restored")
