"""
Step 6:  Medical RAG Agent
Step 10: Chunk → ChromaDB Storage → Retriever
"""

import logging
import re
from pathlib import Path
from typing import List

import chromadb

log = logging.getLogger("MediRoute.rag")

CHROMA_DIR    = "./chroma_db"
COLLECTION    = "medical_guidelines"
PDF_PATH      = "./protocols/medical_guidelines.pdf"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

EMBEDDED_PROTOCOLS = """
=== EMERGENCY TRIAGE PROTOCOLS (Manchester Triage System) ===

PRIORITY 1 - IMMEDIATE (Red) - Target: 0 minutes
Conditions: Cardiac arrest, respiratory failure, major haemorrhage,
unconscious patient, anaphylaxis, severe burns >40%, status epilepticus.
Action: Resuscitation bay. Physician present immediately. Continuous monitoring.
Call crash team if cardiac arrest. Airway management priority.

PRIORITY 2 - URGENT (Orange) - Target: 10 minutes
Conditions: Chest pain with diaphoresis, stroke symptoms (FAST positive),
severe dyspnoea (SpO2 <92%), altered consciousness, severe pain (8-10/10),
uncontrolled bleeding, suspected sepsis, eclampsia.
Action: Physician review within 10-15 minutes. IV access. ECG for chest pain.
Continuous monitoring. Blood cultures if sepsis suspected.

PRIORITY 3 - LESS URGENT (Yellow) - Target: 30 minutes
Conditions: Moderate pain (4-7/10), stable asthma, mild head injury,
vomiting without dehydration, urinary symptoms, minor wounds.
Action: Nurse-led assessment. Vitals every 30 minutes. Analgesia as per protocol.

PRIORITY 4 - NON-URGENT (Green) - Target: 60-120 minutes
Conditions: Minor injuries, cold/flu symptoms, minor lacerations,
stable chronic complaints, medication requests.
Action: Waiting room. Nurse assessment within 60 minutes.

=== CARDIAC EMERGENCIES ===
STEMI criteria: Chest pain + ST elevation >1mm in 2 contiguous leads.
Protocol: Activate cath lab within 90 minutes door-to-balloon. Aspirin 300mg stat.
Troponin >0.04 ng/mL: High sensitivity. Serial ECGs. Cardiology consult.
Heart rate <40 or >130: Immediate physician review. 12-lead ECG.

=== SEPSIS PROTOCOL (Sepsis-6 Bundle) ===
Criteria: SOFA score >= 2 from suspected infection.
Lactate >2.0 mmol/L: Concern. Lactate >4.0 mmol/L: Septic shock.
Within 1 hour: Blood cultures x2, IV antibiotics, IV fluids 30ml/kg,
oxygen to maintain SpO2 >94%, urine output monitoring.

=== STROKE PROTOCOL (FAST) ===
Face drooping + Arm weakness + Speech difficulty + Time to call.
CT head without contrast within 25 minutes of arrival.
Thrombolysis window: 4.5 hours from symptom onset if eligible.
BP target: <185/110 mmHg before thrombolysis.

=== DIABETIC EMERGENCIES ===
DKA: Glucose >250 + Ketones + pH <7.3 + Bicarbonate <18.
Protocol: IV fluid resuscitation, insulin infusion, electrolyte replacement.
Potassium must be >3.5 before insulin. Monitor every hour.
Hypoglycaemia: Glucose <60 mg/dL - 50ml 50% dextrose IV or glucagon 1mg IM.

=== LAB CRITICAL VALUES ===
Potassium <3.0 or >6.0 mEq/L: Immediate physician notification.
Sodium <120 or >160 mEq/L: Emergent correction.
Glucose <40 or >500 mg/dL: Critical - treat immediately.
Haemoglobin <7.0 g/dL: Transfusion threshold.
Troponin >0.04 ng/mL: Rule out ACS. Serial measurements.
Lactate >4.0 mmol/L: Septic shock protocol.
INR >5.0: Bleeding risk - hold anticoagulant, consider reversal.
Creatinine >3.0 mg/dL: Acute kidney injury - nephrology consult.

=== RESPIRATORY EMERGENCIES ===
SpO2 <90%: High-flow oxygen 15L/min via non-rebreather mask.
SpO2 <85%: Consider intubation. Anaesthesia consult immediately.
COPD exacerbation: Controlled oxygen (target SpO2 88-92%). Salbutamol nebuliser.

=== ELDERLY PATIENT CONSIDERATIONS ===
Age >75: Lower threshold for imaging. Atypical presentations common.
Altered cognition in elderly: Rule out sepsis, metabolic cause, stroke first.
Falls in elderly: Full trauma assessment. Anticoagulant status critical.

=== PAEDIATRIC CONSIDERATIONS ===
All paediatric patients: Weight-based dosing. Paediatric team alert.
Age <2 with fever: Full sepsis workup mandatory.
"""


# ─────────────────────────────────────────────────────────────
# STEP 10a — CHUNK
# ─────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    log.info("Chunked into %d pieces", len(chunks))
    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 10b — CHROMADB STORAGE
# ─────────────────────────────────────────────────────────────
def get_chroma_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_or_create_collection(COLLECTION)
    except Exception as e:
        log.warning("PersistentClient failed (%s), using in-memory", e)
        client = chromadb.Client()
        return client.get_or_create_collection(COLLECTION)


def load_pdf_text(path: str) -> str:
    try:
        import PyPDF2
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        log.info("Loaded PDF: %s (%d chars)", path, len(text))
        return text
    except Exception as e:
        log.debug("PDF load skipped: %s", e)
        return ""


def ingest_protocols():
    """Ingests medical guidelines into ChromaDB (runs once on startup)."""
    col = get_chroma_collection()
    if col.count() > 0:
        log.info("ChromaDB already has %d chunks — skipping ingest", col.count())
        return

    source_text = ""
    if Path(PDF_PATH).exists():
        source_text = load_pdf_text(PDF_PATH)
    if not source_text:
        log.info("Using embedded protocol text")
        source_text = EMBEDDED_PROTOCOLS

    chunks    = chunk_text(source_text)
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "medical_guidelines", "chunk_index": i}
                 for i in range(len(chunks))]
    try:
        col.add(documents=chunks, ids=ids, metadatas=metadatas)
        log.info("Ingested %d protocol chunks", len(chunks))
    except Exception as e:
        log.warning("ChromaDB ingest error: %s", e)


# ─────────────────────────────────────────────────────────────
# STEP 10c — RETRIEVER
# ─────────────────────────────────────────────────────────────
def retrieve_protocols(query: str, n: int = 3) -> str:
    """Queries ChromaDB for relevant protocol text."""
    try:
        col = get_chroma_collection()
        if col.count() == 0:
            ingest_protocols()
        results = col.query(query_texts=[query], n_results=min(n, col.count()))
        docs = results.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs) if docs else "No protocol found."
    except Exception as e:
        log.warning("Retrieval failed: %s — keyword fallback", e)
        query_lower = query.lower()
        sections = EMBEDDED_PROTOCOLS.split("===")
        relevant = []
        for sec in sections:
            words = set(query_lower.split())
            if len(words & set(sec.lower().split())) >= 2:
                relevant.append(sec.strip())
        return "\n\n".join(relevant[:3]) if relevant else EMBEDDED_PROTOCOLS[:800]


# ─────────────────────────────────────────────────────────────
# STEP 6 — MEDICAL RAG AGENT
# ─────────────────────────────────────────────────────────────
def rag_agent(state) -> dict:
    """STEP 6 — Retrieves relevant protocol text from ChromaDB."""
    log.info("[RAGAgent] patient=%s", state["patient_id"])

    query_parts = [state["symptom_notes"]]
    if state.get("symptom_flags"):
        query_parts.append(" ".join(state["symptom_flags"][:3]))
    if state.get("lab_alerts"):
        query_parts.append(" ".join(state["lab_alerts"][:2]))

    query   = " ".join(query_parts)[:300]
    context = retrieve_protocols(query)

    log.info("[RAGAgent] retrieved %d chars", len(context))
    return {
        **state,
        "rag_context": context,
        "audit_trail": state["audit_trail"] + [{
            "agent":       "RAGAgent",
            "query":       query[:100],
            "context_len": len(context),
        }],
    }
