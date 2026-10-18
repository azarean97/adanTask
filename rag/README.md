# Standard Baseline RAG: Clinical Multi-Hop Decision Support (Chronic Kidney Disease)

A lightweight, clean, modular, and educational implementation of a **Standard Baseline RAG Workflow** applied to Chronic Kidney Disease (CKD).

This implementation uses exclusively the medical data in `adan_tmp/`:
1. **Clinical Practice Guideline (PDF):** `ckd_evaluation_classification_stratification.pdf` (The KDIGO and K/DOQI guidelines for CKD evaluation, classification, and management).
2. **Clinical Patient Records:** `Chronic_Kidney_Disease/chronic_kidney_disease.arff` (The UCI benchmark dataset containing 400 patient records with lab features such as serum creatinine, albuminuria, blood pressure, diabetes, hemoglobin, and urea).
3. **API & Models:** Connected to `https://api.gapgpt.app/v1` using `gpt-4o-mini` (cost-efficient LLM & Vision) and `text-embedding-3-large` (dense semantic vector embeddings).

---

## 🚫 Exclusion of Custom Project Novelties

To ensure a **standard, canonical baseline RAG** workflow, all domain-specific custom novelties from the parent project have been excluded:
- ❌ **No Rule-Based Expert System:** No boolean Excel matrix or hardcoded epidemiological prior candidates.
- ❌ **No UMLS CUI Ontology Tree Distance Decay:** No offline SNOMED/UMLS graph traversal or decay weights.
- ❌ **No Age-Calibrated Z-Score Laboratory Normalization:** Standard raw clinical values are used directly.
- ❌ **No Multi-Lens Consensus Voting with Reverse Probability Logits:** Standard direct LLM reasoning.
- ❌ **No Dynamic Few-Shot Case Samplers from Excel:** Prompts rely solely on retrieved evidence.
- ❌ **No Taxonomy Fuzzy Tree Guardrail Enforcers:** Standard structured JSON validation.

---

## 🏗️ Standard Baseline RAG Architecture

The workflow consists of standard canonical stages:

```
[Clinical PDF Guideline] ──> [1. VLM Document Parser] ──> [Chunks, Tables, Clinical Facts & Triples]
                                                                        │
                                      ┌─────────────────────────────────┴─────────────────────────────────┐
                                      ▼                                                                   ▼
                       [2. Dense Vector Embeddings]                                         [2. Relational Knowledge Graph]
                       (text-embedding-3-large + BM25)                                       (Standard entity & relation edges)
                                      │                                                                   │
                                      └─────────────────────────────────┬─────────────────────────────────┘
                                                                        ▼
                                                       [3. Multi-Hop Retrieval]
                                                       - Hop 1: Staging & threshold search
                                                       - Intermediate reasoning: Query synthesis
                                                       - Hop 2: Complications & clinical action plan search
                                                       - Reciprocal Rank Fusion (RRF)
                                                                        ▼
                                                       [4. Actor (Primary Answerer)]
                                                       (Diagnostic staging, rationale & citations)
                                                                        ▼
                                                       [5. Critic (Audit & Verification)]
                                                       (Evidence support score, citation verification)
                                                                        │
                                                    ┌───────────────────┴───────────────────┐
                                                    ▼                                       ▼
                                            [Verdict: Accept]                       [Verdict: Revise]
                                                    │                                       │
                                                    │                                [Actor Revisions]
                                                    ▼                                       ▼
                                                    └───────────────────┬───────────────────┘
                                                                        ▼
                                                       [6. Final Verified Decision]
                                                                        │
                                                                        ▼
                                                     [Saved to adan_tmp/outputs/*.json]
```

---

## 📁 Project File Structure

- [config.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/config.py): API connection settings, model specifications, output paths, and editable default prompts.
- [vlm_parser.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/vlm_parser.py): Converts scanned guideline PDF pages into images and extracts tables and facts using standard VLM vision (`gpt-4o-mini`).
- [embeddings.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/embeddings.py): Dense vector embedding generator using `text-embedding-3-large` and pure Python cosine similarity.
- [knowledge_graph.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/knowledge_graph.py): Relational graph representation for disease stages, biomarkers, and clinical actions with standard BFS path traversal.
- [retrievers.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/retrievers.py): Standard multi-retrievers (Okapi BM25, Dense Vector, Knowledge Graph) combined via Reciprocal Rank Fusion (RRF).
- [patient_data.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/patient_data.py): Clinical dataset loader for `chronic_kidney_disease.arff`, calculating standard MDRD eGFR and providing benchmark clinical cases.
- [pipeline.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/pipeline.py): End-to-end decision engine orchestrating Hop 1, Hop 2, Actor, Critic, and saving all artifacts into `outputs/`.
- [app.py](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/app.py): Flask REST API serving endpoints for the interactive web application.
- [outputs/](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/outputs/): Dedicated directory storing all generated structured JSON results.
- [web/](file:///Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp/web/): High-contrast dark glassmorphic web interface (HTML5, Vanilla CSS, JavaScript).

---

## 🚀 Running the Application

To run the web application locally:

```bash
cd /Users/alizarean/Documents/mainProjectS/arzeghar/PID/adan_tmp
.venv/bin/python app.py
```

Open your browser and navigate to:
👉 **`http://localhost:5001`**

---

## 🎛️ Interactive Web Interface Features

1. **Patient Case Selection & Custom Lab Adjustment:**
   - Choose from pre-configured benchmark patients (e.g. Stage 3 CKD with diabetes and hypertension, Stage 4 severe CKD, healthy control).
   - Adjust patient age, blood pressure, creatinine, or albuminuria in real time with live eGFR preview.
2. **Step-by-Step Transparency Tabs:**
   - **★ Final Decision:** Primary diagnostic card, verified confidence, clinical rationale, action plan, and cited chunks.
   - **1. VLM Parsing:** Side-by-side view of the original scanned KDIGO PDF page alongside the extracted structured JSON.
   - **2. Knowledge Graph:** Interactive visual canvas rendering relational graph nodes and edges.
   - **3. Multi-Hop Retrieval:** Breakdown of Hop 1 query, intermediate reasoning, Hop 2 query, and the RRF rank fusion matrix.
   - **4. Actor (Answerer):** Initial proposed diagnosis prior to audit.
   - **5. Critic (Audit):** Evidence support score bar, verdict (accept/revise), unsupported claims check, and revision guidance.
   - **📁 Outputs Directory:** Real-time explorer allowing you to browse and inspect all saved JSON files in `adan_tmp/outputs/`.
3. **Dynamic Prompt Editor:**
   - Click **"Edit System Prompts"** in the top navigation to view or modify any of the system prompts (`VLM`, `Multi-Hop`, `Actor`, `Critic`) on the fly without server restarts.
