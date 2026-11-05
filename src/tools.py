"""Environment E0: the upstream MedAgentBench v2 tools with their mechanical defects fixed.

User decision 2026-10-27: fix the mechanical defects in all arms, keep the upstream schemas otherwise, and leave the
knowledge gaps (which heart-rate source the scorer counts, which text an order code carries) to environment E1. The
fixes add no task information; each one is listed in FIXES, which every run records in its config.json. Tools that
FIXES does not name are the upstream tools, imported unchanged except for their descriptions: user decision
2026-10-28, every tool is described from the data it returns (DESCRIPTIONS).

Evidence for the fixes (probe of 2026-10-27 on a fresh FHIR container, all 98 task patients): 33 patients have more
than 200 medication orders (up to 1831); in 19 tasks of families 2, 7, 8 and 10 the upstream medication search page
hides active orders that the task is about; 92 patients have more than the 20 vital signs of the default page; the
calculator returned the machine date (2026-10-27); families 2 and 7 need a discontinuation that no upstream tool
describes. Every search bundle and every record an agent created carried meta.lastUpdated of the request time
(2026-10-27; the stored records carry 2024-12), and 32 dev attempts of arm D
(Luna, 2026-10-27) wrote a 2026 date into tool arguments; one read a CT of 2023-10-30 as "over 12 months ago".
"""
import concurrent.futures
import dataclasses
import datetime
import importlib
import json
import sys
import types
from typing import Optional

import requests
from pydantic import BaseModel, Field

from clinical_agent.tasks import UPSTREAM

PAGE_SIZE = 200
# The name filter reads every match from the server in pages of this size; the agent still gets pages of PAGE_SIZE.
# Reading the 12272 vital signs of S2937751 took 49 s in pages of 200 and 8 s in pages of 2000 (2026-10-28).
FILTER_PAGE_SIZE = 2000
FIXES = {
    "paging": "fhir_medication_request_search, fhir_vitals_search, fhir_procedure_search, fhir_condition_search and "
              "fhir_observation_search return pages of 200 with the total match count, and take a `page` number "
              "(upstream: one page of 200, or the server default of 20 for vitals, with no way to reach the rest and, "
              "for observations, no total; S2937751 has more than 200 potassium results, probe of 2026-10-28). "
              "Observation paging applies from the runs after `D-dev-100-forms`.",
    "status_filter": "fhir_medication_request_search takes `status` (upstream: no status filter).",
    "sort": "fhir_medication_request_search sorts newest first by authoredOn, fhir_vitals_search by date and "
            "fhir_condition_search by recordedDate (upstream: server order); the four paged searches break ties by "
            "id so that pages neither repeat nor skip records. Observation and procedure searches were already "
            "sorted newest first.",
    "clock": "calculator has no clock: datetime today(), now() and utcnow() return an error, and the two description "
             "examples that used them use fixed dates (upstream: the machine clock).",
    "procedure_date": "fhir_procedure_search `date` is optional (upstream: required, with no way to search all dates) "
                      "and, from the runs after `D-dev-100-descriptions`, a list of date conditions as in vitals_date "
                      "(upstream: one string, so no closed range).",
    "discontinue": "fhir_medication_request_discontinue files a MedicationRequest with status 'stopped' for the "
                   "medication and patient of an existing order, linked to it by priorPrescription. This is the "
                   "discontinuation shape that the upstream scorer reads (families 2 and 7); upstream has no tool "
                   "that describes it.",
    "vitals_date": "fhir_vitals_search `date` is a list of date conditions that all apply, so it takes a closed range "
                   "(upstream: one string described as a date range, which the server rejects for every range form; "
                   "HAPI-1882 in 10 dev attempts of family 3, 2026-10-27).",
    "descriptions": "every E0 tool description states what the tool does, which records it returns with their main "
                    "fields, and when to use it (DESCRIPTIONS), checked on the records of the 98 task patients "
                    "(2026-10-28); the patient parameters are described as a patient id, and the calculator expression "
                    "lists what it can use. Upstream descriptions were missing or wrong: fhir_vitals_search 'any other "
                    "non-duplicable data', while it returns only vital-signs records; fhir_procedure_search 'only "
                    "completed procedures' and fhir_condition_search 'only reconciled and confirmed problems', while no "
                    "record carries a status; fhir_observation_search 'labs' with a 'lab code' (fix of 2026-10-27, now "
                    "part of this one); the patient parameters 'the condition is for'. From the runs after "
                    "`D-dev-100-descriptions`, the descriptions also carry what the system prompt said before (user "
                    "decision 2026-10-28): the vital-sign and observation searches state that one measurement can have "
                    "records in both, the observation `code` that it matches exactly, upper case included, the optional "
                    "codes of the procedure and condition searches that all records are searched by default, and finish "
                    "that the task ends only through it; the fhir_service_request_create `subject` is described as the "
                    "patient of the service request (upstream: 'the medication request'). The create tools state the "
                    "FHIR R4 rules (user decision 2026-10-28): one ServiceRequest requests one service and one "
                    "MedicationRequest one medication; ServiceRequest `priority` is routine, urgent, asap or stat "
                    "(upstream: 'Use \"stat\"'); `authoredOn` is when the order is signed, normally the current time "
                    "(upstream: 'The date the prescription was written').",
    "medication_timing": "fhir_medication_request_create takes `dosageInstruction.timing.code.text`, the frequency as "
                         "text, and `route.text`, in the shape of every existing order (upstream: a plain-string "
                         "route, which the server drops without an error, and no timing, while the description of "
                         "2026-10-28 named timing; the agent of task 2_4 in D-dev-100-tools left a 'q8h' order unplaced). "
                         "From the runs after `D-dev-100-tools`.",
    "next_page": "the paged searches add `next_page` when matches remain after the page, and their descriptions state it "
                 "(upstream: no paging; before: `total` only, and most dev attempts of task 7_6 read only page 1 of "
                 "235 orders). From the runs after `D-dev-100-tools`.",
    "compact_records": "search records leave out what repeats the search or another field: the record type, the "
                       "patient, an `issued` equal to the effective time, the code systems of the category and the "
                       "value unit, and a `code.text` equal to the code (`compact`; upstream: shown). A page of 200 "
                       "active orders of S1698248 was 97973 characters, about 150 per record of them the patient, and "
                       "no dev attempt of D-dev-100-stable to -3 asked for a second page though 59 outputs offered "
                       "one. From the runs after `D-dev-100-stable-3` (user decision 2026-10-28: the agent gets data "
                       "prepared for reasoning).",
    "measurement_summary": "fhir_observation_search (with a code) and fhir_vitals_search add `summary`: per measurement, "
                           "the count, mean, minimum, maximum, unit and time span of the numeric values of every "
                           "match, on all pages (upstream: none). In D-dev-100-stable to -3, 7 of the 10 family 3 "
                           "failures were records placed in the wrong period or averaged wrongly by the agent. From "
                           "the runs after `D-dev-100-stable-3`.",
    "calculator_time": "calculator has hours_between and days_between, which take date-times as records write them "
                       "(upstream: datetime only, which the agent used in 3 of 300 attempts of D-dev-100-stable to "
                       "-3). From the runs after `D-dev-100-stable-3`.",
    "write_criteria": "the descriptions of the tools that change the chart say to use them when a rule's met "
                      "criteria call for the change. In D-dev-100-layers to -3, the agent placed orders right after a "
                      "search that showed their criteria unmet (task 4_20, 4_24, 4_7) and named them mistakes in its "
                      "own answer. From the runs after `D-dev-100-layers-3` (user decision 2026-10-28). The runs "
                      "D-dev-100-criteria to -3 also had a `reason` argument first in these tools; the agent then "
                      "placed orders to record that criteria were unmet (8 failed attempts), and it was removed after "
                      "them (user decision 2026-10-28).",
    "observation_code_list":"fhir_observation_search `code` is optional; without it the tool returns the patient's "
                             "code list, each short code with its record count and newest time, and each "
                             "vital-sign name likewise (user decision 2026-10-28, from the runs after "
                             "`D-dev-100-descriptions`; upstream: `code` required, so no way to find which codes the "
                             "chart holds; the server returns every Observation for an empty code, 21203 for "
                             "S2937751).",
    "server_meta": "every FHIR tool output omits the meta fields that the server writes when a record is stored "
                   "(lastUpdated, versionId, source); they carry server storage times (the request time on bundles and new "
                   "records), not chart times "
                   "(upstream: shown).",
    "bundle_wrapper": "every FHIR search output keeps only resourceType, total and each entry's resource; the bundle id, "
                      "type and links and each entry's fullUrl and search mode are left out (upstream: shown). The "
                      "record ids stay in each resource.",
    "compact_json": "tool outputs reach the agent as compact JSON (agent.serialized; upstream: the Python repr).",
    "calculator_builtins": "calculator has len, round, min, max, abs and sorted besides sum, and its description lists "
                           "them (upstream: the description lists only sum and every one of them, sum included, raised "
                           "NameError).",
    "paging_description": "the paged searches state their page size, order, `total` and later pages in the tool "
                          "description (upstream: 'returns up to 200' or nothing).",
    "medication_name": "fhir_medication_request_search takes `name`, text that the medication name contains in any "
                       "case. The filter runs in the tool over every match, because the server's code:text matches "
                       "only the start of the whole name (probe 2026-10-28: 'chloride' found 0 of 5 orders whose name "
                       "contains it, '20 MEQ' 0). Upstream: no name filter.",
    "medication_dead_params": "fhir_medication_request_search drops `category` and `date`: no MedicationRequest in the "
                              "data carries a category or a dosage boundsPeriod, so any value matched nothing "
                              "(category 'Inpatient' returned 0 orders in task 2_15, 2026-10-27).",
    "vitals_name": "fhir_vitals_search takes `name`, text that the vital sign name contains in any case, filtered in "
                   "the tool as for medication_name (probe 2026-10-28: code:text 'Rate' found 0 of 1761 'Heart Rate' "
                   "and 'Resp Rate' records). Upstream: no name filter.",
    "observation_date": "fhir_observation_search takes `date`, a list of date conditions as in vitals_date (upstream: no "
                        "date filter, so only the newest 200 records of a code were reachable).",
    "code_lists": "the `code` descriptions of fhir_observation_search and fhir_procedure_search state that several codes "
                  "separated by commas match any of them, which the server already supports (upstream: not stated).",
}
SERVER_META = {"lastUpdated", "versionId", "source"}
# Search transport around the records: the bundle's own id, type and links, and each entry's server URL and match mode.
BUNDLE_WRAPPER = {"id", "type", "link"}
ENTRY_WRAPPER = {"fullUrl", "search"}
CALCULATOR_BUILTINS = {"sum": sum, "len": len, "round": round, "min": min, "max": max, "abs": abs, "sorted": sorted}
VITALS_DATE = ("The date range for when the observation was taken: one or more FHIR date conditions that all apply, "
               "each a prefix and a date-time with its time zone, for example "
               "['ge2023-11-13T00:00:00+00:00', 'le2023-11-13T12:00:00+00:00'].")
OBSERVATION_DATE = VITALS_DATE.replace("The date range for when the observation was taken",
                                       "The date range for when the observation was taken or collected") \
    + " By default all dates are searched."
PROCEDURE_DATE = VITALS_DATE.replace("The date range for when the observation was taken",
                                     "The date range for when the procedure was performed") \
    + " By default all dates are searched."
CODE_LIST = " Several codes separated by commas match any of them."
WRITES = ("fhir_medication_request_create", "fhir_service_request_create", "fhir_medication_request_discontinue",
          "fhir_vitals_create")
AUTHORED_ON = "The date and time the {what}, normally the current time, with its time zone."
PATIENT = "The patient's id, for example 'S0000000' or 'Patient/S0000000'."
# Every laboratory code is upper case without spaces, and the server matches a code exactly: 'HEARTRATE' found 22
# records of S2937751, 'heartrate', 'HeartRate' and 'HEART RATE' none (probe of 2026-10-28).
OBSERVATION_CODE = ("The short code of the measurement, as the records carry it in `code.coding`: upper case without "
                    "spaces, an abbreviation such as 'K' for potassium or joined words such as 'DATAQUERY' for "
                    "'data query'. The code matches exactly, upper case included." + CODE_LIST)
CALCULATOR_EXPRESSION = (
    "One Python expression on a single line. For times it has hours_between(start, end) and days_between(start, end), "
    "which take the date-times as the records and tools write them, for example "
    "hours_between('2023-11-13T02:25:00+00:00', '2023-11-13T12:00:00+00:00') gives 9.5833 and is negative when end "
    "is before start. It can also use math (for example math.sqrt(2)), datetime (for example "
    "(datetime.date(2025, 12, 25) - datetime.date(2025, 1, 1)).days), Decimal (for example Decimal('1.2') ** 3) and "
    "sum, len, round, min, max, abs and sorted, which are already available. Write dates and times as fixed values "
    "taken from the task and the records. Examples: 2 + 2 * 3; max([3, 7, 5]) - min([3, 7, 5]); "
    "(datetime.datetime(2025, 12, 25) + datetime.timedelta(days=7)).timestamp().")
PAGE = f"Page of the results, {PAGE_SIZE} records per page, starting at 1. The bundle `total` is the number of matches."
PAGED = (" Returns {size} records per page, {order}, with the number of matches in `total`; when more matches remain, "
         "`next_page` gives the page that holds the next ones. Records leave out the patient, who is the one "
         "searched.")
SUMMARY = (" A result with records also has `summary`: for each measurement with numeric values among all the "
           "matches, on every page, the count, mean, minimum and maximum of those values, the unit and the oldest and "
           "newest time. With a date "
           "range, the summary describes that period; use it for averages and counts over a period.")
NO_CLOCK = "The calculator has no clock. Use the times given in the task and in the records."


def paged(description, order):
    return description + PAGED.format(size=PAGE_SIZE, order=order)


# What each tool does, which records it returns with their main fields, and when to use it. The record fields were
# checked on the 98 task patients of a fresh FHIR container (2026-10-28): vital signs carry only a name and a text
# value; laboratory Observations carry a short code and a numeric value; procedures and conditions carry no status.
# Several measurements have records of both kinds, for example the vital signs 'Temp', 'Heart Rate' and 'FiO2' and the
# laboratory codes TEMP, HEARTRATE and FIO2 (probe of all patients, 2026-10-28).
DESCRIPTIONS = {
    "patient_search": "Finds patients by identifier (MRN), name or birth date. Each Patient record has `id` (the MRN, "
                      "which the other tools take as `patient`), `name`, `gender`, `birthDate`, `address` and "
                      "`telecom`. Use it to confirm the patient before you work on the chart.",
    "fhir_vitals_search": paged(
        "Searches the patient's vital signs: Observation records of category vital-signs. Each record has the vital "
        "sign's name in `code.text`, for example 'Temp', the value as text in `valueString` and the time in "
        "`effectiveDateTime`. Use it for measurements recorded by name. The same measurement can also have records "
        "under a short code in fhir_observation_search, for example 'Temp' here and 'TEMP' there; search both tools "
        "to see all of its records.", "newest first by date") + SUMMARY,
    "fhir_observation_search": paged(
        "Searches the patient's coded measurements, such as laboratory results: Observation records found by their "
        "short code. Each record has the code in `code.coding`, the number and unit in `valueQuantity` and the time "
        "in `effectiveDateTime`. Use it for any measurement whose code you know or can look up. Without a code it "
        "returns the chart's code list instead: each short code with its number of records and newest time, and "
        "each vital-sign name likewise; use it to find the code of a measurement. The same measurement can also have "
        "records by name in fhir_vitals_search; search both tools to see all of its records.",
        "newest first by date") + SUMMARY,
    "fhir_medication_request_search": paged(
        "Searches the patient's medication orders (MedicationRequest), optionally by medication name and status. Each "
        "order has the medication name in `medicationCodeableConcept.text`, `status`, `authoredOn` and the dose, route "
        "and timing in `dosageInstruction`. Use it to review the patient's medications and to find the `id` of an "
        "order.", "newest first by authoredOn"),
    "fhir_medication_request_create": "Files a medication order (MedicationRequest) for the patient: the medication, "
                                      "dose, route and timing, in the shape of the existing orders. Use it when a "
                                      "rule's met criteria call for a medication; each medication is one order.",
    "fhir_medication_request_discontinue": "Discontinues a medication order: files a MedicationRequest with status "
                                           "\"stopped\" for the same medication and patient, linked to the original "
                                           "order by priorPrescription. Use it when a rule's met criteria call "
                                           "for stopping an order, with the order's `id` from "
                                           "fhir_medication_request_search.",
    "fhir_service_request_create": "Files an order (ServiceRequest) for the patient for a service other than a "
                                   "medication, such as a procedure, a test or a referral, identified by its code in "
                                   "`code.coding`. Use it when a rule's met criteria call for such an "
                                   "order; each service is one order.",
    "fhir_vitals_create": "Files a vital-sign measurement (Observation of category vital-signs) in the patient's chart. "
                          "Use it to record a measurement.",
    "fhir_procedure_search": paged(
        "Searches the services performed for the patient (Procedure). Each record has the CPT code and name in `code` "
        "and the time in `performedDateTime`. Use it to find whether and when a service was performed.",
        "newest first by date"),
    "fhir_condition_search": paged(
        "Searches the patient's problem list (Condition). Each record has the ICD-10 code and name in `code`, "
        "`onsetDateTime` and `recordedDate`. Use it to check a diagnosis.", "newest first by recordedDate"),
    "calculator": "Evaluates one Python expression and returns the result: arithmetic, and the hours or days between "
                  "two date-times as the records write them (hours_between, days_between). Use it for any sum, "
                  "difference or time span that a decision depends on.",
    "finish": "Ends the task with your final answer in `value`; the task ends only through this tool. Call it when the "
              "task is done: the first call shows the changes you made to the chart for a last check, and the next "
              "call ends the task.",
}


def upstream(module):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    return importlib.import_module("src." + module)


def without_server_meta(value):
    """`value` with the server-written meta fields removed at every depth, and a meta left empty removed."""
    if isinstance(value, list):
        return [without_server_meta(v) for v in value]
    if not isinstance(value, dict):
        return value
    kept = {}
    for key, item in value.items():
        if key == "meta" and isinstance(item, dict):
            item = {k: v for k, v in item.items() if k not in SERVER_META}
            if not item:
                continue
        kept[key] = without_server_meta(item)
    return kept


def without_bundle_wrapper(value):
    """A search bundle with only `resourceType`, `total` and the records (`entry[].resource`); other values unchanged."""
    if not (isinstance(value, dict) and value.get("resourceType") == "Bundle"):
        return value
    kept = {k: v for k, v in value.items() if k not in BUNDLE_WRAPPER}
    if "entry" in kept:
        kept["entry"] = [{k: v for k, v in e.items() if k not in ENTRY_WRAPPER} for e in kept["entry"]]
    return kept


def get(route, params, page, contains=None, name=None, measure=None):
    """One page of the search. With `contains`, only the records whose `name(record)` contains that text (any case);
    the server's `code:text` matches only the start of a name, so this filter runs here over every match. With
    `measure`, the bundle also has `summary`, the numeric values of every match per measurement (`summarized`)."""
    # `_id` breaks sort ties; without it the server repeated some records across pages and skipped others (1822
    # unique of 1831 active orders of S6309742).
    query = {**params, "_sort": params["_sort"] + ",_id", "_total": "accurate", "_format": "json"}
    page = max(page or 1, 1)
    offset = (page - 1) * PAGE_SIZE
    if contains is None and measure is None:
        return with_next_page(requests.get(route, params={**query, "_count": PAGE_SIZE, "_offset": offset}).json(),
                              page, offset)
    bundles = every_page(route, query)
    if bundles[0].get("resourceType") != "Bundle":
        return bundles[0]
    kept = [e for b in bundles for e in b.get("entry", [])
            if contains is None or contains.lower() in name(e["resource"]).lower()]
    summary = summarized(kept, measure) if measure else []
    return with_next_page({"resourceType": "Bundle", "total": len(kept), **({"summary": summary} if summary else {}),
                           "entry": kept[offset:offset + PAGE_SIZE]}, page, offset)


def every_page(route, query):
    """Every page of the search, in pages of FILTER_PAGE_SIZE; the pages after the first are read in parallel
    (S6309742: 44 s one after another). A server error comes back as the first and only item."""
    query = {**query, "_count": FILTER_PAGE_SIZE}
    first = requests.get(route, params={**query, "_offset": 0}).json()
    if first.get("resourceType") != "Bundle":
        return [first]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        rest = list(pool.map(lambda start: requests.get(route, params={**query, "_offset": start}).json(),
                             range(FILTER_PAGE_SIZE, first.get("total", 0), FILTER_PAGE_SIZE)))
    errors = [b for b in rest if b.get("resourceType") != "Bundle"]
    return errors[:1] or [first, *rest]


def summarized(entries, measure):
    """Per measurement, the count, mean, minimum and maximum of the numeric values of `entries`, and the time span they
    cover. `measure(record)` gives (measurement, value, unit, time) or None; text values that are not one number, such
    as a blood pressure '120/80', are left out."""
    groups = {}
    for entry in entries:
        found = measure(entry["resource"])
        if not found:
            continue
        key, value, unit, when = found
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        group = groups.setdefault(key, {"values": [], "units": set(), "times": []})
        group["values"].append(value)
        group["units"].add(unit)
        group["times"].append(when)
    return [{"measurement": key, "count": len(g["values"]), "mean": round(sum(g["values"]) / len(g["values"]), 2),
             "min": min(g["values"]), "max": max(g["values"]),
             **({"unit": next(iter(g["units"]))} if len(g["units"]) == 1 and None not in g["units"] else {}),
             "oldest": min(t for t in g["times"] if t), "newest": max(t for t in g["times"] if t)}
            for key, g in groups.items()]


def lab_value(record):
    coding = (record.get("code") or {}).get("coding") or []
    quantity = record.get("valueQuantity") or {}
    if not coding or "value" not in quantity:
        return None
    return coding[0]["code"], quantity["value"], quantity.get("unit"), record.get("effectiveDateTime")


def vital_value(record):
    return (record.get("code") or {}).get("text"), record.get("valueString"), None, record.get("effectiveDateTime")


def compact(record):
    """A search record without what repeats the search or another field: the record type (the tool names it), the
    patient (the one searched), an `issued` time equal to the effective time, the code systems of the category and of
    the value unit, and a `code.text` equal to the code's display or code."""
    kept = {k: v for k, v in record.items() if k not in ("resourceType", "subject")}
    if kept.get("issued") == kept.get("effectiveDateTime"):
        kept.pop("issued", None)
    if "category" in kept:
        kept["category"] = [{"coding": [{"code": c.get("code")} for c in cat.get("coding", [])]}
                            for cat in kept["category"]]
    if isinstance(kept.get("valueQuantity"), dict):
        kept["valueQuantity"] = {k: v for k, v in kept["valueQuantity"].items() if k in ("value", "unit")}
    concept = kept.get("code")
    if isinstance(concept, dict) and concept.get("coding") and concept.get("text") in (
            concept["coding"][0].get("display"), concept["coding"][0].get("code")):
        kept["code"] = {k: v for k, v in concept.items() if k != "text"}
    return kept


def compact_records(value):
    """A search bundle with each record `compact`; other values unchanged."""
    if not (isinstance(value, dict) and value.get("resourceType") == "Bundle" and "entry" in value):
        return value
    return {**value, "entry": [{**e, "resource": compact(e["resource"])} for e in value["entry"]]}


def with_next_page(bundle, page, offset):
    """The bundle with `next_page`, the number of the next page, when matches remain after this one. Most dev attempts
    of task 7_6 read only the first page of its 235 active orders, though `total` showed the rest; the one QT-prolonging
    order that the protocol names is on page 2."""
    if bundle.get("resourceType") == "Bundle" and offset + len(bundle.get("entry", [])) < bundle.get("total", 0):
        bundle["next_page"] = page + 1
    return bundle


def medication_request_search(api_base):
    up = upstream("tool.medication_request_search")
    tool = upstream("tool.base").tool

    # Upstream `category` and `date` are left out: no record carries a category or a boundsPeriod, so both always
    # matched nothing.
    class MedicationRequestSearchParams(BaseModel):
        patient: str = Field(description=PATIENT)
        name: Optional[str] = Field(description="Text that the medication name contains, in any case, for example "
                                                "'insulin'. By default all medications are searched.")
        status: Optional[str] = Field(description="The order status, for example active, stopped or completed. "
                                                  "By default all statuses are searched.")
        page: Optional[int] = Field(description=PAGE)

    @tool(name="fhir_medication_request_search", description=DESCRIPTIONS["fhir_medication_request_search"])
    def search(args: MedicationRequestSearchParams):
        params = args.model_dump(exclude_none=True)
        page, text = params.pop("page", 1), params.pop("name", None)
        return get(f"{api_base}/MedicationRequest", {**params, "_sort": "-authoredon"}, page, text,
                   lambda r: (r.get("medicationCodeableConcept") or {}).get("text", ""))

    return search


def vitals_search(api_base):
    up = upstream("tool.vitals_search")
    tool = upstream("tool.base").tool

    class VitalsSearchParams(up.VitalsSearchParams):
        date: Optional[list[str]] = Field(description=VITALS_DATE)
        patient: str = Field(description=PATIENT)
        name: Optional[str] = Field(description="Text that the vital sign name contains, in any case, for example "
                                                "'Temp'. By default all vital signs are searched.")
        page: Optional[int] = Field(description=PAGE)

    @tool(name="fhir_vitals_search", description=DESCRIPTIONS["fhir_vitals_search"])
    def search(args: VitalsSearchParams):
        params = args.model_dump(exclude_none=True)
        page, text = params.pop("page", 1), params.pop("name", None)
        return get(f"{api_base}/Observation", {**params, "_sort": "-date"}, page, text,
                   lambda r: (r.get("code") or {}).get("text", ""), vital_value)

    return search


def procedure_search(api_base):
    up = upstream("tool.procedure_search")
    tool = upstream("tool.base").tool

    class ProcedureSearchParams(up.ProcedureSearchParams):
        code: Optional[str] = Field(default=None, description=up.ProcedureSearchParams.model_fields["code"].description
                                    + CODE_LIST + " By default all procedures are searched.")
        date: Optional[list[str]] = Field(default=None, description=PROCEDURE_DATE)
        patient: str = Field(description=PATIENT)

    class ProcedureSearchArgs(BaseModel):
        search_params: ProcedureSearchParams
        page: Optional[int] = Field(description=PAGE)
        explanation: str = Field(description="Explanation for calling this tool")
        model_config = {"extra": "forbid"}

    @tool(name="fhir_procedure_search", description=DESCRIPTIONS["fhir_procedure_search"])
    def search(args: ProcedureSearchArgs):
        return get(f"{api_base}/Procedure", {**args.search_params.model_dump(exclude_none=True), "_sort": "-date"},
                   args.page)

    return search


def observation_search(api_base):
    up = upstream("tool.observation_search")
    tool = upstream("tool.base").tool

    class ObservationSearchParams(up.ObservationSearchParams):
        code: Optional[str] = Field(description=OBSERVATION_CODE + " Without a code, the tool returns the chart's "
                                                "code list instead of records.")
        patient: str = Field(description=PATIENT)
        date: Optional[list[str]] = Field(description=OBSERVATION_DATE)

    class ObservationSearchArgs(up.ObservationSearchArgs):
        search_params: ObservationSearchParams
        page: Optional[int] = Field(description=PAGE)

    @tool(name="fhir_observation_search", description=DESCRIPTIONS["fhir_observation_search"])
    def search(args: ObservationSearchArgs):
        params = {**args.search_params.model_dump(exclude_none=True), "_sort": "-date"}
        if "code" in params:
            return get(f"{api_base}/Observation", params, args.page, measure=lab_value)
        return chart_codes(f"{api_base}/Observation", params)

    return search


def chart_codes(route, params):
    """The patient's measurements in `params` (patient, date) as a code list: each short code (with its name where the
    record names it otherwise), the number of records and the newest time, and each vital-sign name likewise, newest
    first."""
    bundles = every_page(route, {**params, "_sort": "-date,_id", "_elements": "code,effective", "_total": "accurate",
                                 "_format": "json"})
    codes, vitals = {}, {}
    for bundle in bundles:
        if bundle.get("resourceType") != "Bundle":
            return bundle
        for entry in bundle.get("entry", []):
            record, when = entry["resource"], entry["resource"].get("effectiveDateTime")
            concept = record.get("code") or {}
            coding = concept.get("coding") or []
            if coding:
                code = coding[0]["code"]
                name = coding[0].get("display") or concept.get("text")
                found = codes.setdefault(code, {"code": code, **({"name": name} if name and name != code else {}),
                                                "records": 0, "newest": when})
            else:
                found = vitals.setdefault(concept.get("text"), {"name": concept.get("text"), "records": 0,
                                                                "newest": when})
            found["records"] += 1
    return {"codes": list(codes.values()), "vital_signs": list(vitals.values())}


def condition_search(api_base):
    up = upstream("tool.condition_search")
    tool = upstream("tool.base").tool

    class ConditionSearchParams(up.ConditionSearchParams):
        code: Optional[str] = Field(description=up.ConditionSearchParams.model_fields["code"].description
                                    + " By default all conditions are searched.")
        patient: str = Field(description=PATIENT)

    class ConditionSearchArgs(up.ConditionSearchArgs):
        search_params: ConditionSearchParams
        page: Optional[int] = Field(description=PAGE)

    @tool(name="fhir_condition_search", description=DESCRIPTIONS["fhir_condition_search"])
    def search(args: ConditionSearchArgs):
        return get(f"{api_base}/Condition",
                   {**args.search_params.model_dump(exclude_none=True), "_sort": "-recorded-date"}, args.page)

    return search


def medication_request_create(api_base):
    up = upstream("tool.medication_request_create")

    # Every existing order carries its frequency in dosageInstruction.timing.code.text ('TID', 'Continuous'; 1831 of
    # 1831 orders of S6309742); the upstream schema has no timing, and the agent of task 2_4 (D-dev-100-tools) left a
    # 'q8h' order unplaced for that reason.
    class TimingCode(BaseModel):
        text: str = Field(description="The frequency as text, for example 'TID', 'q8h' or 'Once'.")

    class Timing(BaseModel):
        code: TimingCode

    # FHIR `route` is a CodeableConcept, and existing orders carry `route.text` ('Oral'); the server drops the upstream
    # plain-string route without an error (checked 2026-10-28).
    class Route(BaseModel):
        text: str = Field(description="The route as text, for example 'Oral', 'SC' or 'IV'.")

    class DosageInstruction(up.DosageInstruction):
        route: Route = Field(description="How the medication is given.")
        timing: Optional[Timing] = Field(description="When the medication is given.")

    class MedicationRequestCreateParams(up.MedicationRequestCreateParams):
        authoredOn: str = Field(description=AUTHORED_ON.format(what="prescription is signed"))
        dosageInstruction: list[DosageInstruction]

    return dataclasses.replace(up.create(api_base), input_schema=MedicationRequestCreateParams)


def service_request_create(api_base):
    up = upstream("tool.service_request_create")

    class Subject(up.Subject):
        reference: str = Field(description="The patient FHIR ID for whom the service request is. Format: "
                                           "Patient/{patient_id}")

    # FHIR R4 ServiceRequest: "allows requesting only a single procedure"; several services are separate
    # ServiceRequests; priority is required to be routine, urgent, asap or stat.
    class ServiceRequestCreateParams(up.ServiceRequestCreateParams):
        code: up.Code = Field(description="The one service this order requests. Several codings name that same "
                                          "service in different code systems; several services, such as two tests, "
                                          "are one order each.")
        authoredOn: str = Field(description=AUTHORED_ON.format(what="order is signed"))
        priority: str = Field(description="How urgent the order is: routine, urgent, asap or stat.")
        subject: Subject

    return dataclasses.replace(up.create(api_base), input_schema=ServiceRequestCreateParams)


def calculator():
    tool = upstream("tool.base").tool

    def no_clock(*args, **kwargs):
        raise RuntimeError(NO_CLOCK)

    clock_free = types.ModuleType("datetime")
    clock_free.__dict__.update({k: v for k, v in vars(datetime).items() if not k.startswith("__")})
    clock_free.date = type("date", (datetime.date,), {"today": classmethod(no_clock)})
    clock_free.datetime = type("datetime", (datetime.datetime,), {m: classmethod(no_clock)
                                                                   for m in ("today", "now", "utcnow")})
    class CalculatorParams(BaseModel):
        expression: str = Field(description=CALCULATOR_EXPRESSION)

    def moment(value):
        moment = datetime.datetime.fromisoformat(value) if isinstance(value, str) else value
        return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)

    def hours_between(start, end):
        return round((moment(end) - moment(start)).total_seconds() / 3600, 4)

    def days_between(start, end):
        return round((moment(end) - moment(start)).total_seconds() / 86400, 4)

    @tool(name="calculator", description=DESCRIPTIONS["calculator"])
    def calculate(args: CalculatorParams):
        import math, decimal
        safe_globals = {"__builtins__": CALCULATOR_BUILTINS, "math": math, "datetime": clock_free,
                        "Decimal": decimal.Decimal, "hours_between": hours_between, "days_between": days_between}
        return eval(args.expression, safe_globals, {})

    return calculate


def medication_request_discontinue(api_base):
    tool = upstream("tool.base").tool

    class MedicationRequestDiscontinueParams(BaseModel):
        id: str = Field(description="The FHIR id of the medication order to discontinue.")

    @tool(name="fhir_medication_request_discontinue", description=DESCRIPTIONS["fhir_medication_request_discontinue"])
    def discontinue(args: MedicationRequestDiscontinueParams):
        original = requests.get(f"{api_base}/MedicationRequest/{args.id}", params={"_format": "json"})
        original.raise_for_status()
        original = original.json()
        payload = {"resourceType": "MedicationRequest",
                   "medicationCodeableConcept": original["medicationCodeableConcept"],
                   "dosageInstruction": original.get("dosageInstruction", []),
                   "status": "stopped", "intent": "order",
                   "subject": {"reference": original["subject"]["reference"]},
                   "priorPrescription": {"reference": f"MedicationRequest/{args.id}"}}
        res = requests.post(f"{api_base}/MedicationRequest", json=payload,
                            headers={"Content-Type": "application/fhir+json"})
        res.raise_for_status()
        return {"posted": payload, "created": res.json()}

    return discontinue


FIXED = {"medication_request_search": medication_request_search, "vitals_search": vitals_search,
         "procedure_search": procedure_search, "condition_search": condition_search,
         "observation_search": observation_search, "service_request_create": service_request_create,
         "medication_request_create": medication_request_create}
E0 = ["patient_search", "vitals_search", "observation_search", "medication_request_search",
      "medication_request_create", "medication_request_discontinue", "service_request_create", "vitals_create",
      "procedure_search", "condition_search", "calculator", "finish"]


ORIGINAL = ["patient_search", "vitals_search", "observation_search", "medication_request_search",
            "medication_request_create", "service_request_create", "vitals_create", "procedure_search",
            "condition_search", "calculator", "finish"]


def original(endpoint):
    """{tool name: Tool}: the upstream tool set unchanged (`MedAgent.tools`), for the original-harness baseline."""
    base = endpoint.rstrip("/")
    made = [upstream("tool." + n).create() if n in ("calculator", "finish") else upstream("tool." + n).create(base)
            for n in ORIGINAL]
    return {t.name: t for t in made}


def e0(endpoint):
    """{tool name: Tool} for one FHIR endpoint (http://host:port/fhir)."""
    base = endpoint.rstrip("/")
    made = []
    for name in E0:
        if name in FIXED:
            made.append(FIXED[name](base))
        elif name == "medication_request_discontinue":
            made.append(medication_request_discontinue(base))
        elif name == "calculator":
            made.append(calculator())
        elif name == "finish":
            made.append(upstream("tool.finish").create())
        else:
            made.append(upstream("tool." + name).create(base))
    return {t.name: dataclasses.replace(
                t, description=DESCRIPTIONS[t.name],
                process=lambda args, run=t.process: compact_records(without_bundle_wrapper(without_server_meta(
                    run(args)))))
            for t in made}
