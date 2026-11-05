"""The revised scorer: the upstream v2 scorer (`new_refsol.py`) with six revisions (user decisions 2026-10-27).

It is a second, sensitivity outcome (`passed_revised`); the upstream scorer stays the primary outcome. The revisions
follow written criteria, decided from the scorer code and the task text, with the impact checked on dev data only:
task fidelity, equivalent forms the text allows, decisions fixed by the stated time and not by an arbitrary order,
false passes fixed as well as false fails, no requirement beyond the task text, and ambiguities of the text accepted
in each of their readings.

1. Family 2: brand names of anticoagulants count as anticoagulants (ELIQUIS was missed).
2. Family 2: after the agent's actions exactly one active anticoagulant order remains ("Ensure exactly one active
   order remains"); upstream checks only that some heparin was created or some order stopped.
3. Family 2: the remaining order is the newest of the active orders ("keeping only the newest") or a heparin 5000
   order, existing or new ("The specific order is heparin 5000 units SC q8h prophylaxis"). The task text supports both
   readings, so both pass. With no active order only the heparin reading applies.
4. Family 2: the new order may write the dose as "5,000" or "5000"; upstream accepts both only in existing records.
5. Family 7: the latest QTc is the latest `effectiveDateTime` (the highest value if several share it); upstream takes
   the latest `meta.lastUpdated`.
6. All families: an action is an accepted POST (`extract_posts`); upstream `check_has_post` counts the word "POST" in
   any agent text.

Family 2 details. A stop is a MedicationRequest posted with status stopped, cancelled, completed or entered-in-error.
It ends the order named by `priorPrescription`; a stop without that reference ends the oldest still-active order with
the same medication text. A new order is a posted active MedicationRequest whose text names an anticoagulant. Orders
with the same, latest `authoredOn` are all "the newest"; an order without `authoredOn` is never the newest unless no
order has one. Times are compared as datetimes; a time without an offset is read as UTC.

Families 1 and 3 to 10 run the upstream code with revision 6; families 2 and 7 are the upstream code with the
revisions above. The caller holds the scorer lock (`run.score`), because revision 6 replaces a module global.
"""
import json
from datetime import datetime, timezone

ANTICOAGULANTS = ["heparin", "enoxaparin", "dalteparin", "fondaparinux", "apixaban", "rivaroxaban", "dabigatran",
                  "warfarin",
                  # revision 1: brand names of the agents above, and the other anticoagulants ("any anti-coagulant")
                  "lovenox", "fragmin", "arixtra", "eliquis", "xarelto", "pradaxa", "coumadin", "jantoven",
                  "edoxaban", "savaysa", "argatroban", "bivalirudin"]
STOPPED = {"stopped", "cancelled", "completed", "entered-in-error"}


def when(value):
    """The datetime of a FHIR date or dateTime, None if absent."""
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def latest(resources, field):
    """The resources with the latest `field`; all of them if none has one."""
    dated = [(when(r.get(field)), r) for r in resources]
    top = max((d for d, _ in dated if d), default=None)
    return [r for d, r in dated if d == top] if top else list(resources)


def check_has_post(scorer):
    """Revision 6, in place of `scorer.check_has_post`."""
    return lambda results: bool(scorer.extract_posts(results))


def task2(scorer, case_data, results, fhir_api_base):
    patient = {"reference": f"Patient/{case_data['eval_MRN']}"}

    def text(resource):
        return resource.get("medicationCodeableConcept", {}).get("text", "").lower()

    def anticoagulant(resource):
        return any(k in text(resource) for k in ANTICOAGULANTS)

    def heparin_5000(resource):
        blob = str(resource).lower()
        return "heparin" in blob and ("5000" in blob or "5,000" in blob)  # revision 4

    url = f"{fhir_api_base}MedicationRequest?patient={case_data['eval_MRN']}&status=active&_count=5000&_format=json"
    entries = [e["resource"] for e in json.loads(scorer.send_get_request(url)["data"]).get("entry", [])
               if scorer.before_cutoff(e["resource"])]
    active = [r for r in entries if anticoagulant(r)]

    posts = [p for _, p in scorer.extract_posts(results)]
    orders = [p for p in posts if p.get("resourceType") == "MedicationRequest"]
    remaining = {r["id"]: r for r in active}
    for stop in (p for p in orders if p.get("status", "").lower() in STOPPED):
        target = stop.get("priorPrescription", {}).get("reference", "").removeprefix("MedicationRequest/")
        if not target:
            same = [r for r in remaining.values() if text(r) == text(stop)]
            target = min(same, key=lambda r: when(r.get("authoredOn")) or datetime.min.replace(
                tzinfo=timezone.utc))["id"] if same else ""
        remaining.pop(target, None)
    created = [p for p in orders if p.get("status", "").lower() == "active" and anticoagulant(p)]
    final = list(remaining.values()) + created

    if len(final) != 1:  # revision 2
        return False
    newest = {r["id"] for r in latest(active, "authoredOn")}
    if not (heparin_5000(final[0]) or final[0].get("id") in newest):  # revision 3
        return False
    return all(p.get("subject") == patient for p in posts)


def task7(scorer, case_data, results, fhir_api_base):
    """The upstream code with revisions 5 and 6."""
    qt_med_words = ["ondansetron", "prochlorperazine", "haloperidol", "quetiapine", "olanzapine", "risperidone",
                    "ziprasidone", "clozapine", "chlorpromazine"]
    patient = {"reference": f"Patient/{case_data['eval_MRN']}"}
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=QTCINTERVAL&_count=5000&_format=json"
    entries = [e["resource"] for e in json.loads(scorer.send_get_request(url)["data"]).get("entry", [])
               if scorer.before_cutoff(e["resource"])]
    posts = scorer.extract_posts(results)
    if not entries:
        return not posts
    if max(r["valueQuantity"]["value"] for r in latest(entries, "effectiveDateTime")) <= 500:  # revision 5
        return not posts
    found_ecg = found_stop = False
    for _, p in posts:
        if p.get("resourceType") not in {"ServiceRequest", "MedicationRequest"} or p.get("subject") != patient:
            return False
        blob = str(p).lower()
        status = p.get("status", "").lower()
        if p["resourceType"] == "ServiceRequest" and status == "active" and ("445118002" in blob or "ecg" in blob):
            found_ecg = True
        if p["resourceType"] == "MedicationRequest" and status in STOPPED and any(w in blob for w in qt_med_words):
            found_stop = True
    return found_ecg and found_stop


REVISED = {"task2": task2, "task7": task7}


def family(scorer, name):
    """The revised scoring function of a family: (case_data, results, fhir_api_base) -> bool."""
    if name in REVISED:
        return lambda *args: REVISED[name](scorer, *args)
    return getattr(scorer, name)
