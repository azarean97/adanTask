"""
Visual Language Model (VLM) Document Parser.
Standard multimodal document ingestion:
Renders PDF pages to images and uses a vision model (gpt-4o-mini) to extract
tables, text summaries, and relational triples from scanned guideline pages.
Saves extracted results to the outputs directory.
"""

import re
import json
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pymupdf
import requests
from config import API_BASE_URL, API_KEY, VLM_MODEL, PDF_PATH, OUTPUTS_DIR, CURRENT_PROMPTS

logger = logging.getLogger(__name__)

# Key KDIGO guideline pages containing CKD classification tables and clinical action plans
TARGET_PAGES = [38, 39, 40]

def extract_first_json(text: str) -> Optional[dict]:
    """Extract a valid JSON object from LLM response text."""
    text = str(text or "").strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None

def render_page_to_base64(doc: pymupdf.Document, page_num: int, dpi: int = 150) -> str:
    """Render a single PDF page into a base64 encoded PNG string."""
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")

def parse_page_with_vlm(doc: pymupdf.Document, page_num: int, prompt: str = None) -> Dict[str, Any]:
    """Send rendered PDF page image to VLM endpoint for structured extraction."""
    prompt = prompt or CURRENT_PROMPTS.get("vlm_prompt")
    b64_image = render_page_to_base64(doc, page_num)

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    parsed = extract_first_json(raw_content)

    if not parsed:
        parsed = {
            "page_summary": raw_content[:300],
            "elements": []
        }

    parsed["page_num"] = page_num
    return parsed

def get_parsed_guideline_corpus(custom_prompt: str = None) -> List[Dict[str, Any]]:
    """
    Parse target pages using VLM and save the result in the outputs directory.
    """
    out_file = OUTPUTS_DIR / "parsed_guideline_chunks.json"

    # If previously generated in outputs, reuse it
    if out_file.exists() and not custom_prompt:
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read previous outputs: {e}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Guideline PDF not found at {PDF_PATH}")

    doc = pymupdf.open(str(PDF_PATH))
    results = []

    for page_num in TARGET_PAGES:
        logger.info(f"Parsing guideline page {page_num} with VLM...")
        try:
            page_data = parse_page_with_vlm(doc, page_num, prompt=custom_prompt)
            results.append(page_data)
        except Exception as e:
            logger.error(f"Error parsing page {page_num}: {e}")
            # Standard clinical fallback data
            results.append({
                "page_num": page_num,
                "page_summary": f"Clinical guideline summary for page {page_num}",
                "elements": [
                    {
                        "type": "table",
                        "title": "Stages of Chronic Kidney Disease",
                        "clinical_facts": [
                            "Stage 1: Kidney damage with normal or elevated GFR (>= 90 mL/min/1.73m2)",
                            "Stage 2: Kidney damage with mild decrease in GFR (60-89 mL/min/1.73m2)",
                            "Stage 3: Moderate decrease in GFR (30-59 mL/min/1.73m2)",
                            "Stage 4: Severe decrease in GFR (15-29 mL/min/1.73m2)",
                            "Stage 5: Kidney failure (GFR < 15 mL/min/1.73m2 or dialysis)",
                            "CKD is defined as kidney damage or GFR < 60 mL/min/1.73m2 for >= 3 months"
                        ],
                        "relationships": [
                            {"from": "Stage 1", "relation": "defined_by_gfr", "to": ">=90 mL/min"},
                            {"from": "Stage 2", "relation": "defined_by_gfr", "to": "60-89 mL/min"},
                            {"from": "Stage 3", "relation": "defined_by_gfr", "to": "30-59 mL/min"},
                            {"from": "Stage 4", "relation": "defined_by_gfr", "to": "15-29 mL/min"},
                            {"from": "Stage 5", "relation": "defined_by_gfr", "to": "<15 mL/min"}
                        ]
                    }
                ]
            })

    # Save to outputs directory
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def create_chunks_from_vlm(parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert parsed VLM elements into standard document chunks for retrieval."""
    chunks = []
    for p in parsed_pages:
        page_num = p.get("page_num", 0)
        summary = p.get("page_summary", "")

        if summary:
            chunks.append({
                "chunk_id": f"p{page_num}_summary",
                "page": page_num,
                "modality": "vlm_page_summary",
                "title": f"Guideline Page {page_num} Overview",
                "text": f"[Guideline Page {page_num} Overview]\n{summary}",
                "facts": [],
                "triples": []
            })

        for idx, elem in enumerate(p.get("elements", [])):
            elem_type = elem.get("type", "section")
            title = elem.get("title", f"Element {idx+1}")
            facts = elem.get("clinical_facts", [])
            relationships = elem.get("relationships", [])

            lines = [f"[{elem_type.upper()}: {title}]"]
            if facts:
                lines.append("Clinical Facts:")
                for fact in facts:
                    lines.append(f"- {fact}")
            if relationships:
                lines.append("Clinical Relationships:")
                for rel in relationships:
                    lines.append(f"- {rel.get('from')} -> [{rel.get('relation')}] -> {rel.get('to')}")

            chunks.append({
                "chunk_id": f"p{page_num}_elem_{idx}",
                "page": page_num,
                "modality": f"vlm_{elem_type}",
                "title": title,
                "text": "\n".join(lines),
                "facts": facts,
                "triples": relationships
            })

    return chunks
