"""A medication discontinuation tool that changes the status of the original order.

The study tool `fhir_medication_request_discontinue` (`src/tools.py`) discontinues an order the way the MedAgentBench v2
scorer reads it: it posts a new MedicationRequest with status "stopped" linked by `priorPrescription`, and the original
order stays active. This module gives the FHIR R4 form. `medication_request_discontinue` reads the order, checks its
patient and status, updates the same resource to status "stopped" with a conditional PUT (If-Match on its version),
keeps every other field, and returns success only when a new read shows the order stopped, unchanged otherwise, and
absent from the patient's active orders. An order that is already stopped is left as it is, so a repeated call writes
nothing. No run reported in the paper used this tool.

`apply()` installs the tool in memory, in place of the study tool, for both environments (both build it through
`tools.medication_request_discontinue`); the locked source files in `src/` are not edited.
"""
import requests
from pydantic import BaseModel, Field

from clinical_agent import tools

VERSION = "fhir-correct-v1"
NAME = "fhir_medication_request_discontinue"
NO_CACHE = {"Cache-Control": "no-cache"}
FHIR_JSON = {"Content-Type": "application/fhir+json"}
STOPPABLE = ("active", "on-hold")
DESCRIPTION = ("Discontinues a medication order: sets the status of the existing order (MedicationRequest) to "
               "\"stopped\", keeps its other fields, and confirms that it no longer appears among the patient's active "
               "orders. Use it when a rule's met criteria call for stopping an order, with the order's `id` from "
               "fhir_medication_request_search and the patient's id. An order that is already stopped stays as it is.")
FIX = ("fhir_medication_request_discontinue (fhir-correct-v1) updates the original order to status 'stopped' with a "
       "conditional PUT, after checking its patient and status, and verifies the stored result and the active-order "
       "search; an already stopped order is left unchanged (study tool: posts a new 'stopped' MedicationRequest "
       "linked by priorPrescription and leaves the original active).")


def without_meta(resource):
    return {k: v for k, v in resource.items() if k != "meta"}


def read(api_base, order_id):
    r = requests.get(f"{api_base}/MedicationRequest/{order_id}", params={"_format": "json"}, headers=NO_CACHE,
                     timeout=60)
    if r.status_code in (404, 410):
        raise ValueError(f"No medication order has id {order_id}.")
    r.raise_for_status()
    return r.json()


def listed_active(api_base, order_id, patient):
    r = requests.get(f"{api_base}/MedicationRequest", headers=NO_CACHE, timeout=60,
                     params={"_id": order_id, "patient": patient, "status": "active", "_format": "json"})
    r.raise_for_status()
    return bool(r.json().get("entry"))


def medication_request_discontinue(api_base):
    tool = tools.upstream("tool.base").tool

    class MedicationRequestDiscontinueParams(BaseModel):
        id: str = Field(description="The FHIR id of the medication order to discontinue.")
        patient: str = Field(description=tools.PATIENT)

    @tool(name=NAME, description=DESCRIPTION)
    def discontinue(args: MedicationRequestDiscontinueParams):
        order_id = args.id.strip().removeprefix("MedicationRequest/")
        patient = args.patient.strip().removeprefix("Patient/")
        order = read(api_base, order_id)
        subject = order.get("subject", {}).get("reference", "")
        if subject.removeprefix("Patient/") != patient:
            raise ValueError(f"Medication order {order_id} belongs to {subject or 'no patient'}, not Patient/{patient}.")
        text = order.get("medicationCodeableConcept", {}).get("text")
        if order.get("status") == "stopped":
            return {"result": "already stopped", "id": order_id, "status": "stopped", "medication": text}
        if order.get("status") not in STOPPABLE:
            raise ValueError(f"Medication order {order_id} has status '{order.get('status')}'; an active or on-hold "
                             f"order can be discontinued.")
        body = {**without_meta(order), "status": "stopped"}
        put = requests.put(f"{api_base}/MedicationRequest/{order_id}", json=body, timeout=60,
                           headers={**FHIR_JSON, "If-Match": f'W/"{order["meta"]["versionId"]}"'})
        if put.status_code == 412:
            raise RuntimeError(f"Medication order {order_id} changed while it was being discontinued; read it again.")
        put.raise_for_status()
        stored = read(api_base, order_id)
        kept = {k: v for k, v in without_meta(stored).items() if k != "status"} == \
            {k: v for k, v in without_meta(order).items() if k != "status"}
        if stored.get("status") != "stopped" or not kept or listed_active(api_base, order_id, patient):
            raise RuntimeError(f"Medication order {order_id} was not verified as discontinued: stored status "
                               f"'{stored.get('status')}', other fields kept: {kept}.")
        return {"result": "discontinued", "id": order_id, "status": "stopped", "medication": text}

    return discontinue


def apply():
    """Install the tool and its description in memory, and record the change in `tools.FIXES`."""
    tools.medication_request_discontinue = medication_request_discontinue
    tools.DESCRIPTIONS[NAME] = DESCRIPTION
    tools.FIXES["discontinue"] = FIX
