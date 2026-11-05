"""Environment E1: the institutional tools that environment E0 lacks (current time, terminology, order catalog, protocol
library).

Every entry is moved from the instruction or corrected `context` (`tasks.corrected`) of one MedAgentBench v2 task
family, in the original wording where a sentence fits; no entry adds clinical information that those texts do not
contain. The one addition is in TERMINOLOGY: for each code or medication class that existing records carry, the
"Records:" sentence names the FHIR resource type and the search field, as a hospital terminology service does (user
decision 2026-10-27). It was checked on a fresh FHIR container (2026-10-27): HEARTRATE, TSH, FT4 and QTCINTERVAL are
only on laboratory Observations; IMGCT0491, IMGIL0001, NUR1373, CON417, 90686 and COVIDVACCINE on Procedures; C64.2 on
Conditions. The order codes 74177, 445118002 and 91320 are on no existing record, so their entries have no "Records:"
sentence.
Within a family the context is the same for all 30 tasks except the stated time, so the entries are shared and only
the time depends on the task. `current_time` returns it, and the protocol library shows it before its entries, as an
electronic health record shows the time on screen (user decision 2026-10-27, after the dev failure analysis of arm D).
The library holds the entries of all ten families, as a hospital's library holds all of its protocols. The time comes
only from the "It's <time> now" sentence of the corrected task context (`tasks.corrected`); families 2, 6, 7 and 8
get it from the correction from the runs after `D-dev-100-tools`, and stated no time before.
"""
import re

from pydantic import BaseModel, Field

from clinical_agent.tools import e0, upstream

MEDICATION = "Records: MedicationRequest; the medication name is in medicationCodeableConcept.text."
LAB = "Records: Observation, category 'laboratory', numeric valueQuantity; search by `code` '{}'."
TERMINOLOGY = [
    "CT Abdomen: the CPT codes for existing CT Abdomen studies are IMGCT0491 or IMGIL0001. Records: Procedure; "
    "search by `code` 'IMGCT0491,IMGIL0001'.",
    "CT Abdomen/Pelvis with IV contrast: the CPT code is 74177.",
    "Heart rate: the code for heart rate is 'HEARTRATE'. " + LAB.format("HEARTRATE"),
    "Urinary catheter: the procedure code for urinary catheter is 'NUR1373'. Records: Procedure; search by `code` "
    "'NUR1373'.",
    "Malignant neoplasm of left kidney: the ICD-10 code is C64.2. Records: Condition; search by `code` 'C64.2'.",
    "Interventional Radiology referral: the CPT code is CON417. Records: Procedure; search by `code` 'CON417'.",
    "TSH: the code for TSH is 'TSH'. " + LAB.format("TSH"),
    "Free T4: the code for free T4 is 'FT4'. " + LAB.format("FT4"),
    "QTc interval: the code for QTc interval is 'QTCINTERVAL'. " + LAB.format("QTCINTERVAL"),
    "12-lead ECG: the SNOMED code for 12-lead ECG is 445118002.",
    "Influenza vaccine: the CPT code for influenza vaccine is 90686. Records: Procedure; search by `code` '90686'.",
    "COVID-19 vaccine: the code for COVID-19 vaccine procedure is 'COVIDVACCINE'; COVID-19 vaccines are also recorded "
    "as MedicationRequest with text containing 'COVID-19 VAC'. Records: Procedure, search by `code` 'COVIDVACCINE'; "
    "MedicationRequest, medicationCodeableConcept.text.",
    "COVID vaccine booster: the CPT code for the single-dose COVID vaccine is 91320.",
    "Opioid analgesic medications: hydromorphone, oxycodone, fentanyl, hydrocodone, morphine. " + MEDICATION,
    "QT-prolonging medications: text containing ondansetron, prochlorperazine and most antipsychotic medications. "
    + MEDICATION,
    "Anticoagulant medications: any anti-coagulant medication (for example, LMWH, DOAC, etc), including heparin. "
    + MEDICATION,
]
ORDER_CATALOG = [
    "CT Abdomen/Pelvis with IV contrast: CPT 74177.",
    "Interventional Radiology referral: CPT CON417, request 'Evaluate renal mass for biopsy or ablation.'",
    "Urinary catheter removal: service request to remove the catheter; urinary catheter procedure code 'NUR1373'.",
    "DVT prophylaxis: heparin 5000 units SC q8h prophylaxis.",
    "Levothyroxine 25 mcg PO daily.",
    "TSH lab test: code 'TSH'.",
    "Free T4 lab test: code 'FT4'.",
    "12-lead ECG: SNOMED 445118002.",
    "Naloxone nasal spray 4 mg.",
    "Influenza vaccine (quadrivalent, preservative-free, IM): CPT 90686.",
    "COVID vaccine booster: single-dose COVID vaccine, CPT 91320.",
]
PROTOCOLS = [
    "CT Abdomen follow-up of a renal mass: find the date of the most recent CT Abdomen procedure (CPT IMGCT0491 or "
    "IMGIL0001). If the study was performed more than 12 months ago, order a new CT Abdomen with IV contrast (CPT "
    "74177) with indication 'Renal mass follow-up'. A patient with no recorded CT Abdomen study is due for a new one.",
    "DVT prophylaxis: a patient has exactly one active DVT prophylaxis order. If there are zero orders, create one: "
    "heparin 5000 units SC q8h prophylaxis. If there are multiple orders, discontinue duplicates keeping only the "
    "newest. Ensure exactly one active order remains. A duplicate order could be any anti-coagulant medication (for "
    "example, LMWH, DOAC, etc).",
    "Heart rate review: calculate the average heart rate (code 'HEARTRATE') over the past 6 hours and the past 12 "
    "hours. Answer with "
    "the two averages as numbers, [6-hour average, 12-hour average]; use -1 for a period that has no heart rate "
    "value.",
    "Urinary catheter: if a patient has a urinary catheter (procedure code 'NUR1373') that has been in place for more "
    "than 48 hours without a documented removal order, create a service request to remove the catheter.",
    "Malignant neoplasm of left kidney: confirm the diagnosis (ICD-10 C64.2). If true and the last CT Abdomen/Pelvis "
    "is more than 3 months old, order a new CT Abdomen/Pelvis with IV contrast (CPT 74177, for oncologic follow-up) "
    "and a referral to Interventional Radiology (CPT CON417) with the request 'Evaluate renal mass for biopsy or "
    "ablation.' A patient with no recorded CT Abdomen/Pelvis is due for a new one.",
    "Thyroid function (TSH code 'TSH', free T4 code 'FT4'): retrieve the most recent TSH and free T4 values. If TSH > 10 "
    "uIU/mL on two separate instances of lab values (separated by at least one month in date of lab collection), start "
    "levothyroxine 25 mcg PO daily and order repeat TSH/FT4 in 6 weeks. If TSH is 4.5–10 uIU/mL and free T4 is normal "
    "(0.8–1.8 ng/dL), order repeat TSH/FT4 in 3 months (no medication yet).",
    "Prolonged QTc: check the latest QTc interval (code 'QTCINTERVAL'). The threshold for prolonged QTc is > 500 ms. If "
    "prolonged, discontinue QT-prolonging medications (text containing ondansetron, prochlorperazine and most "
    "antipsychotic medications) and order a follow-up 12-lead ECG (SNOMED 445118002) to be performed in 24 hours.",
    "Naloxone with opioids: every active opioid analgesic order (hydromorphone, oxycodone, fentanyl, hydrocodone, "
    "morphine) has a matching naloxone prescription. If an opioid order is active without naloxone, create a naloxone "
    "order: naloxone nasal spray 4 mg.",
    "Influenza vaccine: determine the date of the last influenza vaccine (CPT 90686). If it was administered more than "
    "365 days ago, order a new influenza vaccine (quadrivalent, preservative-free, IM) for today. A patient with no "
    "recorded influenza vaccine is due for a new one.",
    "COVID-19 booster: find the most recent COVID-19 vaccine (procedure code 'COVIDVACCINE', or MedicationRequest with "
    "text containing 'COVID-19 VAC'). If the last dose was more than 12 months ago, order a COVID booster: a "
    "single-dose COVID vaccine (CPT 91320) today. A patient with no recorded COVID-19 vaccine is due for a booster.",
]
NOW = re.compile(r"It's (\S+) now")


def search(entries, query):
    """Entries with a word that starts with a word of the query (case-insensitive), most matching words first; all
    entries for an empty query."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1}
    if not words:
        return entries
    tokens = [re.findall(r"[a-z0-9]+", e.lower()) for e in entries]
    scored = [(sum(any(t.startswith(w) for t in tokens[i]) for w in words), i) for i in range(len(entries))]
    return [entries[i] for n, i in sorted(scored, key=lambda s: (-s[0], s[1])) if n]


class Query(BaseModel):
    query: str = Field(description="Words to search for. An empty query lists all entries.")


class NoArguments(BaseModel):
    pass


DESCRIPTIONS = {
    "terminology_search": "Searches the hospital terminology. Each entry gives the code of a clinical item and, for "
                          "items that existing records carry, the resource type and search field of those records. "
                          "Use it to turn a clinical term into the code that a search or an order needs.",
    "order_catalog_search": "Searches the hospital order catalog. Each entry gives an orderable item with its code or "
                            "order details. Use it to find what an order needs before you place it.",
    "protocol_search": "Searches the hospital protocol library. Each entry is the standing order for a clinical topic: "
                       "its criteria and the actions they trigger, with the codes it uses; when the current time is "
                       "known, the results start with it. Use it to learn what the hospital requires for a topic.",
    "current_time": "Returns the current date and time, or says that it is not available. Use it for anything that "
                    "depends on now, such as how long ago something happened.",
}


def library(name, entries, header=None):
    tool = upstream("tool.base").tool

    @tool(name=name, description=DESCRIPTIONS[name])
    def find(args: Query):
        found = search(entries, args.query) or ["No entry matches."]
        return [header, *found] if header else found

    return find


def stated_time(task):
    stated = NOW.search(task.get("context") or "")
    return stated.group(1) if stated else None


def current_time(task):
    tool = upstream("tool.base").tool
    stated = stated_time(task)

    @tool(name="current_time", description=DESCRIPTIONS["current_time"])
    def now(args: NoArguments):
        return stated or "The current time is not available."

    return now


def e1(endpoint, task):
    """{tool name: Tool}: environment E0 and the institutional tools."""
    made = [current_time(task), library("terminology_search", TERMINOLOGY),
            library("order_catalog_search", ORDER_CATALOG),
            library("protocol_search", PROTOCOLS, stated_time(task) and f"Current time: {stated_time(task)}.")]
    return {**e0(endpoint), **{t.name: t for t in made}}
