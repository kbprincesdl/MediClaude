"""
Step 11: LangGraph Assembly
FastAPI: REST endpoints for testing

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import uuid
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from state  import TriageState
from agents import severity_agent, routing_logic, tool_node
from rag    import rag_agent, ingest_protocols
from tools  import recommendation_agent, supervisor_agent, memory_recall

log = logging.getLogger("MediRoute.app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────
# STEP 11 — LANGGRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────
def build_graph():
    """
    Assembles the full multi-agent LangGraph pipeline.

    Flow:
        severity_agent
            ↓ (conditional via routing_logic)
        ┌── tool_node → rag_agent → recommendation_agent ──┐
        │                                                   ↓
        └──────────────── supervisor_agent (fast path) ─────┘
                                   ↓
                                  END
    """
    g = StateGraph(TriageState)

    g.add_node("severity_agent",       severity_agent)
    g.add_node("tool_node",            tool_node)
    g.add_node("rag_agent",            rag_agent)
    g.add_node("recommendation_agent", recommendation_agent)
    g.add_node("supervisor_agent",     supervisor_agent)

    g.set_entry_point("severity_agent")

    g.add_conditional_edges(
        "severity_agent",
        routing_logic,
        {
            "tool_node":        "tool_node",
            "supervisor_agent": "supervisor_agent",  # CRITICAL fast path
        },
    )

    g.add_edge("tool_node",            "rag_agent")
    g.add_edge("rag_agent",            "recommendation_agent")
    g.add_edge("recommendation_agent", "supervisor_agent")
    g.add_edge("supervisor_agent",     END)

    return g.compile()


_GRAPH = build_graph()
log.info("LangGraph compiled successfully")


def run_triage(patient_id, symptom_notes, patient_history, lab_values) -> dict:
    initial: TriageState = {
        "patient_id":           patient_id,
        "symptom_notes":        symptom_notes,
        "patient_history":      patient_history,
        "lab_values":           lab_values,
        "symptom_severity":     "",
        "symptom_flags":        [],
        "history_risk_score":   0,
        "history_flags":        [],
        "lab_alerts":           [],
        "lab_severity":         "NORMAL",
        "rag_context":          "",
        "recommendation":       "",
        "triage_priority":      "",
        "confidence":           0.0,
        "reasoning":            "",
        "recommended_action":   "",
        "next_agent":           "",
        "llm_used":             "rule_based",
        "audit_trail":          [],
        "timestamp":            datetime.now().isoformat(),
        "run_id":               str(uuid.uuid4()),
    }
    final = _GRAPH.invoke(initial)
    return {
        "patient_id":         final["patient_id"],
        "run_id":             final["run_id"],
        "timestamp":          final["timestamp"],
        "triage_priority":    final["triage_priority"],
        "confidence":         final["confidence"],
        "reasoning":          final["reasoning"],
        "recommended_action": final["recommended_action"],
        "recommendation":     final["recommendation"],
        "symptom_severity":   final["symptom_severity"],
        "symptom_flags":      final["symptom_flags"],
        "lab_severity":       final["lab_severity"],
        "lab_alerts":         final["lab_alerts"],
        "history_risk":       final["history_risk_score"],
        "history_flags":      final["history_flags"],
        "llm_used":           final["llm_used"],
        "audit_trail":        final["audit_trail"],
    }


# ─────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting MediRoute — ingesting medical protocols...")
    ingest_protocols()
    log.info("Ready at http://localhost:8000/docs")
    yield


app = FastAPI(
    title="MediRoute — AI Triage System",
    description=(
        "Multi-agent LangGraph triage system for hospital emergency departments.\n\n"
        "Agents: SeverityAgent → ToolNode → RAGAgent → RecommendationAgent → SupervisorAgent\n\n"
        "LLM: Local Ollama (llama3/mistral) with rule-based fallback — no cloud calls."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PatientHistory(BaseModel):
    age:         Optional[int]  = Field(None,  example=67)
    sex:         Optional[str]  = Field(None,  example="M")
    conditions:  Optional[list] = Field(default_factory=list, example=["hypertension","diabetes"])
    medications: Optional[list] = Field(default_factory=list, example=["aspirin"])
    allergies:   Optional[list] = Field(default_factory=list, example=["penicillin"])


class TriageRequest(BaseModel):
    patient_id:      str            = Field(..., example="PT-001")
    symptom_notes:   str            = Field(..., example="67-year-old male with crushing chest pain radiating to left arm. Diaphoresis. BP 90/60.")
    patient_history: PatientHistory = Field(default_factory=PatientHistory)
    lab_values:      dict           = Field(default_factory=dict, example={"troponin": 0.18, "glucose": 310})


class QuickTriageRequest(BaseModel):
    symptom_notes: str = Field(..., example="Patient with severe chest pain and shortness of breath")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html><head><title>MediRoute</title>
    <style>
      body{font-family:system-ui;background:#0f172a;color:#e2e8f0;
           display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
      .box{max-width:560px;text-align:center;padding:40px}
      h1{color:#38bdf8;font-size:2.2rem;margin-bottom:8px}
      p{color:#94a3b8;line-height:1.7}
      .badge{display:inline-block;padding:4px 12px;border-radius:999px;font-size:13px;margin:4px}
      .p1{background:#7f1d1d;color:#fca5a5}
      .p2{background:#78350f;color:#fcd34d}
      .p3{background:#1e3a5f;color:#93c5fd}
      .p4{background:#14532d;color:#86efac}
      a{color:#38bdf8;text-decoration:none;font-weight:600}
    </style></head>
    <body><div class="box">
      <h1>🏥 MediRoute</h1>
      <p>Multi-Agent AI Triage System for Hospital Emergency Departments</p>
      <p>
        <span class="badge p1">P1 Immediate</span>
        <span class="badge p2">P2 Urgent</span>
        <span class="badge p3">P3 Less Urgent</span>
        <span class="badge p4">P4 Non-Urgent</span>
      </p>
      <p style="margin-top:24px">
        <a href="/docs">📖 Swagger UI (Interactive API)</a><br><br>
        <a href="/redoc">📚 ReDoc Documentation</a><br><br>
        <a href="/demo">🧪 Run Demo Patients</a>
      </p>
    </div></body></html>
    """


@app.post("/triage", tags=["Triage"])
def triage_patient(req: TriageRequest):
    """Full triage — runs all 5 agents with complete patient data."""
    try:
        return run_triage(
            patient_id      = req.patient_id,
            symptom_notes   = req.symptom_notes,
            patient_history = req.patient_history.model_dump(),
            lab_values      = req.lab_values,
        )
    except Exception as e:
        log.error("Triage error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/triage/quick", tags=["Triage"])
def quick_triage(req: QuickTriageRequest):
    """Quick triage — symptom text only, no history or labs required."""
    result = run_triage(
        patient_id      = f"QUICK-{str(uuid.uuid4())[:8].upper()}",
        symptom_notes   = req.symptom_notes,
        patient_history = {},
        lab_values      = {},
    )
    return {
        "patient_id":         result["patient_id"],
        "triage_priority":    result["triage_priority"],
        "confidence":         result["confidence"],
        "symptom_severity":   result["symptom_severity"],
        "symptom_flags":      result["symptom_flags"],
        "recommended_action": result["recommended_action"],
    }


@app.get("/patients/similar", tags=["Memory"])
def similar_cases(symptom_notes: str):
    """Memory recall — returns similar past cases (Step 9 demo)."""
    cases = memory_recall(symptom_notes, n=3)
    return {"similar_cases": cases}


@app.get("/demo", tags=["System"])
def run_demo():
    """Runs 3 demo patients — great for quick testing."""
    cases = [
        {
            "patient_id":      "DEMO-P1",
            "symptom_notes":   "67-year-old with crushing chest pain and diaphoresis. BP 90/60.",
            "patient_history": {"age": 67, "conditions": ["hypertension","diabetes"]},
            "lab_values":      {"troponin": 0.18, "glucose": 310, "systolic_bp": 88},
        },
        {
            "patient_id":      "DEMO-P3",
            "symptom_notes":   "28-year-old with moderate back pain 5/10 for 2 days.",
            "patient_history": {"age": 28, "conditions": []},
            "lab_values":      {},
        },
        {
            "patient_id":      "DEMO-P4",
            "symptom_notes":   "22-year-old with sore throat and runny nose for 3 days.",
            "patient_history": {"age": 22, "conditions": []},
            "lab_values":      {"wbc": 9.5, "glucose": 90},
        },
    ]
    return {"results": [
        {k: run_triage(**c)[k] for k in
         ["patient_id","triage_priority","confidence","symptom_severity","recommended_action"]}
        for c in cases
    ]}


@app.get("/health", tags=["System"])
def health():
    return {
        "status":  "ok",
        "service": "MediRoute AI Triage",
        "version": "1.0.0",
        "agents":  ["SeverityAgent","ToolNode","RAGAgent","RecommendationAgent","SupervisorAgent"],
        "llm":     "Ollama (llama3) with rule-based fallback",
        "memory":  "ChromaDB local",
    }
