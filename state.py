"""
Step 1: Define State
Shared data bag passed between all agents in the LangGraph pipeline.
"""

from typing import TypedDict, List, Annotated
import operator


class TriageState(TypedDict):
    # Raw inputs
    patient_id:       str
    symptom_notes:    str
    patient_history:  dict
    lab_values:       dict

    # Agent 1 - Severity Agent
    symptom_severity: str        # CRITICAL / HIGH / MODERATE / LOW
    symptom_flags:    List[str]  # red-flag keywords found

    # Agent 2 - History Reviewer
    history_risk_score: int      # 0-100
    history_flags:      List[str]

    # Agent 3 - Lab Interpreter
    lab_alerts:   List[str]
    lab_severity: str            # CRITICAL / HIGH / MODERATE / NORMAL

    # Agent 4 - RAG context
    rag_context:  str

    # Agent 5 - Recommendation
    recommendation: str

    # Supervisor final decision
    triage_priority:    str      # P1 / P2 / P3 / P4
    confidence:         float    # 0.0 - 1.0
    reasoning:          str
    recommended_action: str

    # Routing and meta
    next_agent: str
    llm_used:   str
    run_id:     str
    timestamp:  str

    # Append-only audit trail
    audit_trail: Annotated[List[dict], operator.add]
