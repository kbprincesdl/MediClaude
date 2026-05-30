"""
Step 7: Recommendation Agent
Step 8: Supervisor Agent
Step 9: Memory (ChromaDB case store + recall)
"""

import json
import re
import logging
from typing import List

import chromadb

from agents import llm_query, tool_composite_score

log = logging.getLogger("MediRoute.supervisor")

MEMORY_DIR        = "./chroma_db"
MEMORY_COLLECTION = "triage_memory"


# ─────────────────────────────────────────────────────────────
# STEP 9 — MEMORY
# ─────────────────────────────────────────────────────────────
def _memory_col():
    try:
        client = chromadb.PersistentClient(path=MEMORY_DIR)
        return client.get_or_create_collection(MEMORY_COLLECTION)
    except Exception:
        client = chromadb.Client()
        return client.get_or_create_collection(MEMORY_COLLECTION)


def memory_store(state: dict):
    """Saves completed triage case to ChromaDB for future recall."""
    try:
        col = _memory_col()
        doc = (
            f"Patient: {state['patient_id']} | "
            f"Symptoms: {state['symptom_notes'][:200]} | "
            f"Priority: {state['triage_priority']} | "
            f"Severity: {state['symptom_severity']} | "
            f"Reasoning: {state['reasoning'][:200]}"
        )
        col.upsert(
            documents=[doc],
            ids=[state["run_id"]],
            metadatas=[{
                "patient_id": state["patient_id"],
                "priority":   state["triage_priority"],
                "confidence": str(state["confidence"]),
                "timestamp":  state["timestamp"],
            }],
        )
        log.info("[Memory] Stored  run_id=%s", state["run_id"])
    except Exception as e:
        log.warning("[Memory] Store failed: %s", e)


def memory_recall(symptom_notes: str, n: int = 2) -> List[str]:
    """Retrieves similar past cases from ChromaDB."""
    try:
        col = _memory_col()
        if col.count() == 0:
            return []
        results = col.query(query_texts=[symptom_notes], n_results=min(n, col.count()))
        return results.get("documents", [[]])[0]
    except Exception as e:
        log.warning("[Memory] Recall failed: %s", e)
        return []


# ─────────────────────────────────────────────────────────────
# STEP 7 — RECOMMENDATION AGENT
# ─────────────────────────────────────────────────────────────
def recommendation_agent(state: dict) -> dict:
    """STEP 7 — Generates structured clinical recommendation using RAG + signals."""
    log.info("[RecommendationAgent] patient=%s", state["patient_id"])

    rag_ctx   = state.get("rag_context", "")
    sim_cases = memory_recall(state["symptom_notes"])
    past_ctx  = "\n".join(f"- {c}" for c in sim_cases) if sim_cases else "No prior cases."

    system = (
        "You are a senior emergency nurse. Using the patient signals and "
        "retrieved protocols, write a concise clinical recommendation (3-5 sentences). "
        "Focus on immediate actions, tests to order, and monitoring priorities."
    )
    prompt = (
        f"Symptom severity: {state.get('symptom_severity')}\n"
        f"Symptom flags: {', '.join(state.get('symptom_flags', []))}\n"
        f"Lab severity: {state.get('lab_severity')}\n"
        f"Lab alerts: {', '.join(state.get('lab_alerts', []))}\n"
        f"History risk: {state.get('history_risk_score')}/100\n"
        f"History conditions: {', '.join(state.get('history_flags', []))}\n\n"
        f"Relevant protocols:\n{rag_ctx[:600]}\n\n"
        f"Similar cases:\n{past_ctx}\n\n"
        "Write a focused clinical recommendation:"
    )

    raw, engine = llm_query(prompt, system)
    recommendation = raw.strip() if raw else _rule_recommendation(state)

    log.info("[RecommendationAgent] %d chars via %s", len(recommendation), engine)
    return {
        **state,
        "recommendation": recommendation,
        "audit_trail": state["audit_trail"] + [{
            "agent":      "RecommendationAgent",
            "engine":     engine,
            "rec_length": len(recommendation),
        }],
    }


def _rule_recommendation(state: dict) -> str:
    """Rule-based recommendation when LLM is unavailable."""
    sev   = state.get("symptom_severity", "LOW")
    flags = state.get("symptom_flags", [])
    labs  = state.get("lab_alerts", [])
    hist  = state.get("history_flags", [])
    lines = []

    if sev == "CRITICAL":
        lines.append("CRITICAL presentation — activate emergency protocol immediately.")
        lines.append("Ensure airway, breathing, circulation. IV access x2. Continuous monitoring.")
    elif sev == "HIGH":
        lines.append("High acuity — rapid physician assessment required.")
        lines.append("Establish IV access, baseline vitals, 12-lead ECG if chest pain.")
    elif sev == "MODERATE":
        lines.append("Moderate acuity — nurse-led assessment and analgesia as indicated.")
    else:
        lines.append("Low acuity — standard waiting room protocol.")

    if flags:
        lines.append(f"Red flags: {', '.join(flags[:3])}.")
    if labs:
        lines.append(f"Abnormal labs: {', '.join(labs[:2])}.")
    if hist:
        lines.append(f"High-risk comorbidities: {', '.join(hist[:3])}.")

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────
# STEP 8 — SUPERVISOR AGENT
# ─────────────────────────────────────────────────────────────
def supervisor_agent(state: dict) -> dict:
    """STEP 8 — Final decision-maker. Assigns P1-P4, confidence, reasoning."""
    log.info("[SupervisorAgent] patient=%s", state["patient_id"])

    sym_sev   = state.get("symptom_severity", "LOW")
    hist_risk = state.get("history_risk_score", 0)
    lab_sev   = state.get("lab_severity", "NORMAL")

    priority, confidence, reasoning, action = tool_composite_score(
        sym_sev, hist_risk, lab_sev
    )

    system = (
        "You are the Chief Medical Officer AI. "
        "Review all triage signals and make the final P1/P2/P3/P4 decision. "
        "Return ONLY JSON: "
        '{"priority":"P2","confidence":0.88,"reasoning":"...","action":"..."}'
    )
    prompt = (
        f"Symptom severity : {sym_sev}\n"
        f"Symptom flags    : {', '.join(state.get('symptom_flags', []))}\n"
        f"History risk     : {hist_risk}/100\n"
        f"History flags    : {', '.join(state.get('history_flags', []))}\n"
        f"Lab severity     : {lab_sev}\n"
        f"Lab alerts       : {', '.join(state.get('lab_alerts', []))}\n"
        f"Recommendation   : {state.get('recommendation', '')[:200]}\n"
        f"Rule priority    : {priority} (confidence {confidence:.2f})\n\n"
        "Make the final triage decision."
    )

    raw, engine = llm_query(prompt, system)

    if raw:
        try:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                llm_p = parsed.get("priority", "").upper()
                order = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}
                if order.get(llm_p, 0) > order.get(priority, 0):
                    priority = llm_p
                confidence = max(confidence, float(parsed.get("confidence", confidence)))
                if parsed.get("reasoning"):
                    reasoning += f" | LLM: {parsed['reasoning']}"
                if parsed.get("action"):
                    action = parsed["action"]
        except Exception:
            pass

    rec = state.get("recommendation", "")
    if rec:
        reasoning += f" | Rec: {rec[:120]}"

    log.info("[SupervisorAgent] FINAL priority=%s confidence=%.2f", priority, confidence)

    updated = {
        **state,
        "triage_priority":    priority,
        "confidence":         confidence,
        "reasoning":          reasoning,
        "recommended_action": action,
        "next_agent":         "done",
        "llm_used":           state.get("llm_used", engine),
        "audit_trail": state["audit_trail"] + [{
            "agent":      "SupervisorAgent",
            "priority":   priority,
            "confidence": confidence,
            "action":     action,
            "engine":     engine,
        }],
    }

    memory_store(updated)
    return updated
