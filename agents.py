"""
Step 2: Severity Agent
Step 3: Routing Logic
Step 4: Tool Examples (lab checker, vitals scorer, composite scorer)
Step 5: Tool Node (executes tools inside the graph)
"""

import json
import re
import logging
import requests
from typing import List

from state import TriageState

log = logging.getLogger("MediRoute.agents")

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"


def llm_query(prompt: str, system: str = "") -> tuple:
    """Call Ollama. Falls back to rule engine if unavailable."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("response", ""), "ollama"
    except Exception as e:
        log.debug("Ollama unavailable: %s", e)
        return "", "rule_based"


# ─────────────────────────────────────────────────────────────
# STEP 2 — SEVERITY AGENT
# ─────────────────────────────────────────────────────────────
CRITICAL_KEYWORDS = [
    "cardiac arrest", "chest pain", "crushing chest", "heart attack",
    "stroke", "facial droop", "not breathing", "airway obstruction",
    "anaphylaxis", "unresponsive", "unconscious", "seizure",
    "severe trauma", "gunshot", "stab", "septic shock",
    "pulmonary embolism", "aortic dissection", "eclampsia", "meningitis",
]
HIGH_KEYWORDS = [
    "shortness of breath", "difficulty breathing", "dyspnoea",
    "fever above 39", "temp 39", "temp 40", "high fever",
    "severe abdominal pain", "vomiting blood", "altered mental",
    "confusion", "severe pain", "pain 8", "pain 9", "pain 10",
    "worst headache", "thunderclap", "fracture",
]
MODERATE_KEYWORDS = [
    "moderate pain", "pain 4", "pain 5", "pain 6", "pain 7",
    "nausea", "vomiting", "diarrhoea", "diarrhea", "uti",
    "fever", "rash", "sprain", "back pain", "ear pain",
]


def severity_agent(state: TriageState) -> TriageState:
    """STEP 2 — Reads symptom_notes, produces symptom_severity + symptom_flags."""
    log.info("[SeverityAgent] patient=%s", state["patient_id"])
    notes = state["symptom_notes"].lower()

    flags = []
    for kw in CRITICAL_KEYWORDS:
        if kw in notes:
            flags.append(kw)
    if flags:
        rule_sev = "CRITICAL"
    else:
        for kw in HIGH_KEYWORDS:
            if kw in notes:
                flags.append(kw)
        rule_sev = "HIGH" if flags else ""
        if not flags:
            for kw in MODERATE_KEYWORDS:
                if kw in notes:
                    flags.append(kw)
            rule_sev = "MODERATE" if flags else "LOW"
    if not flags:
        flags = ["no red-flag symptoms detected"]

    system = (
        "You are a clinical triage nurse AI. "
        "Classify severity as CRITICAL/HIGH/MODERATE/LOW. "
        'Return ONLY valid JSON: {"severity":"HIGH","flags":["flag1"]}'
    )
    raw, engine = llm_query(f"Symptom notes: {state['symptom_notes']}", system)
    severity = rule_sev

    if raw:
        try:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                p = json.loads(m.group())
                llm_sev = p.get("severity", "").upper()
                order = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
                if order.get(llm_sev, 0) > order.get(severity, 0):
                    severity = llm_sev
                flags = list(set(flags + p.get("flags", [])))
        except Exception:
            pass

    log.info("[SeverityAgent] severity=%s", severity)
    return {
        **state,
        "symptom_severity": severity,
        "symptom_flags":    flags,
        "llm_used":         engine,
        "audit_trail": state["audit_trail"] + [{
            "agent": "SeverityAgent",
            "severity": severity,
            "flags": flags,
            "engine": engine,
        }],
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 — ROUTING LOGIC
# ─────────────────────────────────────────────────────────────
def routing_logic(state: TriageState) -> str:
    """
    STEP 3 — Conditional edge function.
    Returns node name to jump to based on severity.
    """
    sev = state.get("symptom_severity", "LOW")
    next_node = state.get("next_agent", "")

    if next_node and next_node != "done":
        log.info("[Router] Directed → %s", next_node)
        return next_node

    if sev == "CRITICAL":
        log.info("[Router] CRITICAL → supervisor fast path")
        return "supervisor_agent"
    else:
        log.info("[Router] %s → tool_node", sev)
        return "tool_node"


# ─────────────────────────────────────────────────────────────
# STEP 4 — TOOL EXAMPLES
# ─────────────────────────────────────────────────────────────
LAB_THRESHOLDS = {
    "glucose":     (60,   250,  "mg/dL"),
    "sodium":      (130,  150,  "mEq/L"),
    "potassium":   (3.0,  5.5,  "mEq/L"),
    "creatinine":  (None, 2.0,  "mg/dL"),
    "haemoglobin": (7.0,  None, "g/dL"),
    "hemoglobin":  (7.0,  None, "g/dL"),
    "wbc":         (2.0,  20.0, "K/uL"),
    "platelets":   (50,   None, "K/uL"),
    "lactate":     (None, 2.5,  "mmol/L"),
    "ph":          (7.25, 7.55, ""),
    "o2_sat":      (90,   None, "%"),
    "spo2":        (90,   None, "%"),
    "heart_rate":  (40,   130,  "bpm"),
    "systolic_bp": (80,   200,  "mmHg"),
    "troponin":    (None, 0.04, "ng/mL"),
    "inr":         (None, 3.5,  ""),
    "bun":         (None, 50,   "mg/dL"),
}

HIGH_RISK_CONDITIONS = [
    "diabetes", "hypertension", "heart disease", "cardiac",
    "copd", "asthma", "renal failure", "kidney disease",
    "cancer", "immunocompromised", "hiv", "transplant",
    "pregnancy", "pregnant", "anticoagulant", "warfarin",
]


def tool_check_labs(lab_values: dict) -> tuple:
    """Tool: Lab Checker — scans labs against clinical thresholds."""
    if not lab_values:
        return "NORMAL", ["no lab values provided"]

    alerts, critical = [], False
    for key, val in lab_values.items():
        k = key.lower().replace(" ", "_")
        try:
            v = float(str(val).replace(",", ""))
        except Exception:
            continue
        if k in LAB_THRESHOLDS:
            lo, hi, unit = LAB_THRESHOLDS[k]
            if lo and v < lo:
                lvl = "CRITICAL" if v < lo * 0.85 else "HIGH"
                alerts.append(f"{key}={v}{unit} LOW [{lvl}] (min {lo})")
                if lvl == "CRITICAL":
                    critical = True
            elif hi and v > hi:
                lvl = "CRITICAL" if v > hi * 1.15 else "HIGH"
                alerts.append(f"{key}={v}{unit} HIGH [{lvl}] (max {hi})")
                if lvl == "CRITICAL":
                    critical = True

    if not alerts:
        return "NORMAL", ["all lab values within range"]
    return ("CRITICAL" if critical else "HIGH"), alerts


def tool_score_history(patient_history: dict) -> tuple:
    """Tool: History Risk Scorer — scores comorbidity burden."""
    score, flags = 0, []
    text = json.dumps(patient_history).lower()

    for cond in HIGH_RISK_CONDITIONS:
        if cond in text:
            flags.append(cond)
            score += 15

    age = patient_history.get("age", patient_history.get("Age", 0))
    if isinstance(age, (int, float)):
        if age >= 75:
            score += 25
            flags.append(f"age {int(age)} - high risk")
        elif age >= 60:
            score += 15
            flags.append(f"age {int(age)} - elevated risk")
        elif age <= 2:
            score += 20
            flags.append(f"age {int(age)} - infant")

    return min(score, 100), flags


def tool_composite_score(sym_sev: str, hist_risk: int, lab_sev: str) -> tuple:
    """Tool: Composite Priority Calculator — combines all signals."""
    sev_w = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}.get(sym_sev, 1)
    lab_w = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "NORMAL": 1}.get(lab_sev, 1)
    hist_w = hist_risk / 100
    composite = sev_w * 0.5 + lab_w * 0.3 + hist_w * 0.2 * 4

    if composite >= 3.5 or sev_w == 4 or lab_w == 4:
        p, c = "P1", min(0.95, 0.75 + composite * 0.05)
        action = "Immediate physician. Resuscitation bay. Do not delay."
    elif composite >= 2.5:
        p, c = "P2", min(0.90, 0.70 + composite * 0.04)
        action = "Physician within 15 minutes. Continuous monitoring."
    elif composite >= 1.8:
        p, c = "P3", min(0.85, 0.65 + composite * 0.04)
        action = "Nurse assessment within 30 minutes. Routine vitals."
    else:
        p, c = "P4", min(0.80, 0.60 + composite * 0.04)
        action = "Non-urgent. Waiting room. Reassess if deteriorates."

    reasoning = (
        f"Symptom={sym_sev}({sev_w}/4) | Lab={lab_sev}({lab_w}/4) | "
        f"History={hist_risk}/100 | Composite={composite:.2f} → {p}"
    )
    return p, round(c, 2), reasoning, action


# ─────────────────────────────────────────────────────────────
# STEP 5 — TOOL NODE
# ─────────────────────────────────────────────────────────────
def tool_node(state: TriageState) -> TriageState:
    """STEP 5 — Executes all tools and writes results back to state."""
    log.info("[ToolNode] patient=%s", state["patient_id"])

    lab_sev, lab_alerts = tool_check_labs(state["lab_values"])
    hist_risk, hist_flags = tool_score_history(state["patient_history"])

    log.info("[ToolNode] lab=%s hist_risk=%d", lab_sev, hist_risk)
    return {
        **state,
        "lab_severity":       lab_sev,
        "lab_alerts":         lab_alerts,
        "history_risk_score": hist_risk,
        "history_flags":      hist_flags,
        "audit_trail": state["audit_trail"] + [{
            "agent":        "ToolNode",
            "lab_severity": lab_sev,
            "lab_alerts":   lab_alerts[:3],
            "history_risk": hist_risk,
            "hist_flags":   hist_flags[:3],
        }],
    }
