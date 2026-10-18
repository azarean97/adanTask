"""
Standard Baseline RAG System Configuration.
Contains API settings, model endpoints, file paths, and default prompts.
All configurations and comments are in English.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent


def _load_env_file():
    """Read KEY=VALUE lines from a .env file (project folder or adan_tmp/) into os.environ."""
    for env_file in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

# API Endpoints and Authentication
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.gapgpt.app/v1")
API_KEY = os.getenv("API_KEY", "")                # set in adan_tmp/.env (see .env.example)

# Model Definitions
# Uses lightweight gpt-4o-mini for cost efficiency (< $1 budget)
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Uses standard text-embedding-3-large embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Data and Storage Paths
PDF_PATH = BASE_DIR / "ckd_evaluation_classification_stratification.pdf"
ARFF_PATH = BASE_DIR / "Chronic_Kidney_Disease" / "chronic_kidney_disease.arff"

# Dedicated directory for saving all pipeline and module outputs
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard Prompt Templates
DEFAULT_PROMPTS = {
    "vlm_prompt": """You are an OCR and clinical document extraction model analyzing a medical guideline page.
Extract all visible tables, key clinical facts, and relationships in clean JSON format:
{
  "page_summary": "Short summary of the page content",
  "elements": [
    {
      "type": "table|chart|flowchart|boxed_summary",
      "title": "Title of section or table",
      "clinical_facts": ["Discrete medical fact or threshold"],
      "relationships": [
        {"from": "Entity A", "relation": "defines|progresses_to|requires", "to": "Entity B"}
      ]
    }
  ]
}
Return only valid JSON without markdown wrapping.""",

    "multihop_prompt": """You are a multi-hop query generator in a clinical RAG system.
Given the patient's primary symptoms and initial retrieved findings (Hop 1), generate a targeted Hop 2 search query to retrieve specific clinical guideline treatment recommendations and complications management.

Patient Case:
{patient_text}

Hop 1 Initial Staging Findings:
{hop1_findings}

Return JSON:
{
  "intermediate_reasoning": "Brief explanation of what information is missing",
  "hop2_query": "Targeted search query for second retrieval pass",
  "key_risk_factors": ["risk factor 1", "risk factor 2"]
}""",

    "actor_prompt": """You are a Clinical Decision Support Agent (Actor) in a multi-hop RAG framework.
Synthesize the patient case with the retrieved KDIGO guidelines and reference in-context exemplars to formulate an evidence-grounded diagnosis and action plan.

### Clinical Target Directive & Question:
Given the patient's labs, retrieved KDIGO guidelines, and in-context reference exemplars from the patient dataset:
1. Determine if the patient meets diagnostic criteria for Chronic Kidney Disease (CKD vs NOT_CKD).
2. Assign the precise KDIGO GFR stage (Stage 1 to Stage 5) and GFR category (G1 to G5).
3. Evaluate key risk factors (Hypertension, Diabetes, Albuminuria) and synthesize an evidence-grounded clinical action plan.

### Patient Record Under Evaluation:
{patient_text}

### In-Context Few-Shot Exemplars (from UCI CKD Patient Dataset):
{few_shot_exemplars}

### Retrieved KDIGO Guideline Evidence:
{retrieved_evidence}

### Multi-Hop Reasoning Context:
{multihop_context}

Return valid JSON:
{
  "predicted_class": "CKD or NOT_CKD",
  "ckd_stage": "Stage 1|Stage 2|Stage 3|Stage 4|Stage 5|Not Applicable",
  "gfr_category": "e.g. G1 (>=90), G2 (60-89), G3 (30-59), G4 (15-29), G5 (<15)",
  "confidence": 0.0 to 1.0,
  "clinical_rationale": "Clear reasoning connecting patient labs, few-shot exemplars, and guideline criteria",
  "clinical_action_plan": ["Action item 1", "Action item 2"],
  "evidence_citations": ["Chunk-ID or guideline citation"]
}""",

    "critic_prompt": """You are the Clinical Verification Critic in an Actor-Critic architecture.
Strictly audit the Actor's proposed diagnosis against the retrieved guideline excerpts. Verify citations and claims.

Patient Record:
{patient_text}

Proposed Actor Diagnosis:
{actor_decision}

Retrieved Evidence Excerpts:
{retrieved_evidence}

Return valid JSON:
{
  "verdict": "accept|revise|insufficient_evidence",
  "evidence_support_score": 0.0 to 1.0,
  "staging_correct": true or false,
  "unsupported_claims": ["claim 1 if any"],
  "contradictions": ["contradiction if any"],
  "revision_instructions": "Specific guidance if verdict is revise, otherwise empty string",
  "citation_audit": [
    {"claim": "statement", "is_supported": true or false}
  ]
}"""
}

# In-memory runtime prompt store
CURRENT_PROMPTS = dict(DEFAULT_PROMPTS)
