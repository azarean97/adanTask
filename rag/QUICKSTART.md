# Mini-PID-RAG: Quickstart Guide

A lightweight, baseline Multi-Hop Clinical RAG system for Chronic Kidney Disease (CKD) staging and action planning based on the KDIGO guideline and UCI CKD patient dataset.

---

## 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- An active Internet connection for the OpenAI-compatible API (`https://api.gapgpt.app/v1`)

---

## 2. Installation

Navigate to the project directory and create a virtual environment:

```bash
cd adan_tmp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Configuration

API key and endpoints are pre-configured in `config.py`. You can optionally override them via environment variables:

```bash
export API_BASE_URL="https://api.gapgpt.app/v1"
export API_KEY="sk-your-key-here"
export LLM_MODEL="gpt-4o-mini"
export EMBEDDING_MODEL="text-embedding-3-large"
```

---

## 4. How to Run

### Option A: Launch Interactive Web Application
```bash
python app.py
```
Then open your browser at: **http://127.0.0.1:5001**

The web interface features:
- **Interactive Patient Testing:** Choose from benchmark cases or enter custom lab values.
- **Multi-Hop Toggle:** Switch Multi-Hop retrieval ON or OFF.
- **In-Context Few-Shot Exemplars:** Real patient archetypes loaded from UCI CKD dataset.
- **VLM Guideline Viewer:** High-res scanned KDIGO tables + extracted triples.
- **Full Transparency:** View the exact executed LLM prompt directly on screen.

### Option B: Run Standalone Python Pipeline
```bash
python -c "
from pipeline import pipeline
from patient_data import patient_db
patient = patient_db.get_benchmark_cases()[0]
res = pipeline.run(patient, enable_multihop=True)
print('Decision:', res['final_output'])
"
```

All execution outputs, embeddings, and step traces are saved automatically to `adan_tmp/outputs/`.
