"""
Standard Baseline RAG Web Application Backend (Flask).
Provides clean REST API endpoints for the interactive web interface:
- Running pipeline on benchmark or custom patients
- Inspecting step-by-step outputs (VLM, KG, Multi-hop, Retrievals, Actor, Critic)
- Viewing and downloading all generated outputs from the outputs directory
- Dynamically editing and updating prompts
All comments and code are strictly in English.
"""

import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pymupdf

from config import BASE_DIR, CURRENT_PROMPTS, DEFAULT_PROMPTS, PDF_PATH, OUTPUTS_DIR, VLM_MODEL, LLM_MODEL, EMBEDDING_MODEL
from pipeline import pipeline
from patient_data import patient_db, calculate_egfr, estimate_ckd_stage
from knowledge_graph import kg_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

# Ensure pipeline is initialized at startup
try:
    pipeline.initialize()
except Exception as e:
    logger.error(f"Startup initialization note: {e}")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    """Return system readiness and model configurations."""
    return jsonify({
        "status": "ready" if pipeline.is_initialized else "initializing",
        "models": {
            "vlm": VLM_MODEL,
            "llm": LLM_MODEL,
            "embeddings": EMBEDDING_MODEL
        },
        "total_chunks": len(pipeline.chunks),
        "total_kg_nodes": len(kg_instance.nodes),
        "total_patients": len(patient_db.patients)
    })

@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    """Get or dynamically update system prompts."""
    if request.method == "POST":
        data = request.json or {}
        new_prompts = data.get("prompts", {})
        for k, v in new_prompts.items():
            if k in CURRENT_PROMPTS:
                CURRENT_PROMPTS[k] = v
        return jsonify({"message": "Prompts updated successfully", "prompts": CURRENT_PROMPTS})
    return jsonify({
        "prompts": CURRENT_PROMPTS,
        "default_prompts": DEFAULT_PROMPTS
    })

@app.route("/api/config/reset", methods=["POST"])
def reset_config():
    """Reset all prompts to system defaults."""
    global CURRENT_PROMPTS
    CURRENT_PROMPTS = dict(DEFAULT_PROMPTS)
    return jsonify({"message": "Prompts reset to defaults", "prompts": CURRENT_PROMPTS})

@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Return benchmark clinical cases and total patient count."""
    benchmarks = patient_db.get_benchmark_cases()
    return jsonify({
        "benchmarks": benchmarks,
        "sample_count": len(patient_db.patients)
    })

@app.route("/api/page_image/<int:page_num>")
def get_page_image(page_num):
    """Render and serve a specific page from the guideline PDF."""
    try:
        cache_img = OUTPUTS_DIR / f"page_{page_num}.png"
        if not cache_img.exists():
            doc = pymupdf.open(str(PDF_PATH))
            if 1 <= page_num <= len(doc):
                pix = doc[page_num - 1].get_pixmap(dpi=150)
                pix.save(str(cache_img))
            else:
                return jsonify({"error": "Page number out of range"}), 404
        return send_file(str(cache_img), mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/graph", methods=["GET"])
def get_graph():
    """Return knowledge graph nodes and links for frontend rendering."""
    return jsonify(kg_instance.to_web_dict())

@app.route("/api/outputs", methods=["GET"])
def list_outputs():
    """List all saved output files in the outputs directory."""
    files = []
    for f in OUTPUTS_DIR.glob("*.json"):
        files.append({"name": f.name, "size_bytes": f.stat().st_size})
    return jsonify({"outputs": files})

@app.route("/api/outputs/<filename>")
def get_output_file(filename):
    """Retrieve the content of a specific output file."""
    target = OUTPUTS_DIR / filename
    if target.exists() and target.is_file():
        return send_file(str(target), mimetype="application/json")
    return jsonify({"error": "File not found"}), 404

@app.route("/api/vlm/data", methods=["GET"])
def get_vlm_data():
    """Return parsed VLM guideline pages and extracted facts/triples."""
    from vlm_parser import get_parsed_guideline_corpus
    try:
        pages = get_parsed_guideline_corpus()
        return jsonify({"pages": pages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/initial_result", methods=["GET"])
def get_initial_result():
    """Return pre-computed verified pipeline trace for instant display on page load."""
    trace_file = OUTPUTS_DIR / "full_pipeline_trace.json"
    if trace_file.exists():
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            logger.warning(f"Error loading initial trace: {e}")
    return jsonify({"status": "no_previous_run"})

@app.route("/api/ask", methods=["POST"])
def ask_assistant():
    """Handle natural language user inquiries routed across documents, structured database, or calculation tools."""
    try:
        body = request.json or {}
        query = body.get("query", "").strip()
        if not query:
            return jsonify({"error": "Empty query provided"}), 400
        result = pipeline.answer_assistant_query(query)
        return jsonify(result)
    except Exception as e:
        logger.exception("Assistant query failed")
        return jsonify({"error": str(e)}), 500

@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Execute full baseline RAG workflow on a patient record."""
    try:
        body = request.json or {}
        patient_data = body.get("patient")
        enable_multihop = bool(body.get("enable_multihop", True))
        clinical_query = body.get("clinical_query")

        if not patient_data:
            age = float(body.get("age", 55))
            sc = float(body.get("sc", 1.8))
            al = int(body.get("al", 2))
            bp = float(body.get("bp", 130))
            htn = body.get("htn", "yes")
            dm = body.get("dm", "yes")
            hemo = float(body.get("hemo", 10.5))
            egfr = calculate_egfr(sc, age)

            patient_data = {
                "patient_id": "CUSTOM-PATIENT",
                "age": age,
                "bp": bp,
                "sc": sc,
                "al": al,
                "htn": htn,
                "dm": dm,
                "hemo": hemo,
                "egfr": egfr,
                "calculated_stage": estimate_ckd_stage(egfr, al),
                "clinical_text": (
                    f"Patient ID: CUSTOM-PATIENT\n"
                    f"Age: {age} years\n"
                    f"Blood Pressure: {bp} mm/Hg\n"
                    f"Serum Creatinine: {sc} mg/dL (Estimated eGFR: {egfr} mL/min/1.73m2)\n"
                    f"Urine Albumin: {al}+\n"
                    f"Hemoglobin: {hemo} g/dL\n"
                    f"Hypertension: {htn}\n"
                    f"Diabetes Mellitus: {dm}"
                )
            }

        custom_prompts = body.get("prompts")
        weights = body.get("weights")

        result = pipeline.run(
            patient=patient_data,
            custom_prompts=custom_prompts,
            retrieval_weights=weights,
            enable_multihop=enable_multihop,
            clinical_query=clinical_query
        )

        return jsonify(result)
    except Exception as e:
        logger.exception("Pipeline execution failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print(f"Starting Standard Baseline RAG server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
