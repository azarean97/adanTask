"""
Patient Clinical Data Loader and Feature Extractor.
Parses the UCI Chronic Kidney Disease dataset (chronic_kidney_disease.arff),
calculates estimated GFR (eGFR) using standard MDRD formula,
and provides benchmark patient records for evaluation.
All comments and code are strictly in English.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from config import ARFF_PATH

def calculate_egfr(sc: float, age: float, is_female: bool = False) -> float:
    """Calculate estimated GFR (mL/min/1.73 m2) using clinical MDRD formula approximation."""
    try:
        if sc <= 0 or age <= 0:
            return 90.0
        gender_mult = 0.742 if is_female else 1.0
        egfr = 175.0 * (sc ** -1.154) * (age ** -0.203) * gender_mult
        return round(egfr, 1)
    except Exception:
        return 60.0

def estimate_ckd_stage(egfr: float, albumin: int = 0) -> str:
    """Determine CKD stage based on eGFR and albuminuria damage marker."""
    if egfr >= 90:
        return "Stage 1 (Normal/High GFR with Kidney Damage)" if albumin > 0 else "Normal Kidney Function (G1)"
    elif 60 <= egfr < 90:
        return "Stage 2 (Mild decrease in GFR with Kidney Damage)" if albumin > 0 else "Mildly Decreased (G2)"
    elif 30 <= egfr < 60:
        return "Stage 3 (Moderate decrease in GFR)"
    elif 15 <= egfr < 30:
        return "Stage 4 (Severe decrease in GFR)"
    else:
        return "Stage 5 (Kidney Failure / Dialysis)"

class PatientDataset:
    """Loader and manager for Chronic Kidney Disease clinical cases."""
    def __init__(self, arff_path: Path = ARFF_PATH):
        self.arff_path = Path(arff_path)
        self.patients: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self.arff_path.exists():
            return

        with open(self.arff_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        data_started = False
        attrs = [
            "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
            "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc",
            "rbcc", "htn", "dm", "cad", "appet", "pe", "ane", "class"
        ]

        pid = 1
        for line in lines:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@data"):
                data_started = True
                continue
            if not data_started:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(attrs):
                continue

            record = {"patient_id": f"PAT-{pid:03d}"}
            for k, val in zip(attrs, parts):
                if val == "?":
                    record[k] = None
                else:
                    try:
                        record[k] = float(val) if ("." in val or k in ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "al", "su"]) else val
                    except ValueError:
                        record[k] = val

            sc_val = record.get("sc")
            age_val = record.get("age") or 50.0
            al_val = int(record.get("al") or 0) if record.get("al") is not None else 0

            if sc_val is not None:
                record["egfr"] = calculate_egfr(float(sc_val), float(age_val))
                record["calculated_stage"] = estimate_ckd_stage(record["egfr"], al_val)
            else:
                record["egfr"] = None
                record["calculated_stage"] = "Unknown"

            summary_parts = [
                f"Patient ID: {record['patient_id']}",
                f"Age: {record.get('age', 'N/A')} years",
                f"Blood Pressure: {record.get('bp', 'N/A')} mm/Hg",
                f"Serum Creatinine: {record.get('sc', 'N/A')} mg/dL (Estimated eGFR: {record['egfr']} mL/min/1.73m2)",
                f"Urine Albumin: {record.get('al', '0')}+",
                f"Blood Urea: {record.get('bu', 'N/A')} mg/dL",
                f"Hemoglobin: {record.get('hemo', 'N/A')} g/dL",
                f"Hypertension: {record.get('htn', 'no')}",
                f"Diabetes Mellitus: {record.get('dm', 'no')}",
                f"Pedal Edema: {record.get('pe', 'no')}",
                f"Anemia: {record.get('ane', 'no')}",
                f"True Clinical Classification: {record.get('class', 'unknown')}"
            ]
            record["clinical_text"] = "\n".join(summary_parts)
            self.patients.append(record)
            pid += 1

    def get_benchmark_cases(self) -> List[Dict[str, Any]]:
        """Return 5 representative clinical benchmark cases across different stages."""
        if not self.patients:
            return []

        st3 = next((p for p in self.patients if p.get("egfr") and 30 <= p["egfr"] < 60 and p.get("htn") == "yes"), self.patients[2])
        st4 = next((p for p in self.patients if p.get("egfr") and 15 <= p["egfr"] < 30), self.patients[3])
        st1_2 = next((p for p in self.patients if p.get("egfr") and p["egfr"] >= 60 and (p.get("al") or 0) >= 1), self.patients[0])
        st5 = next((p for p in self.patients if p.get("egfr") and p["egfr"] < 15), self.patients[8])
        ctrl = next((p for p in self.patients if p.get("class") == "notckd"), self.patients[-1])

        st3["case_label"] = "Case A: Moderate CKD (Stage 3 + Hypertension + Diabetes)"
        st4["case_label"] = "Case B: Severe CKD (Stage 4 + Acidosis/Edema)"
        st1_2["case_label"] = "Case C: Early CKD (Stage 1/2 with Albuminuria)"
        st5["case_label"] = "Case D: Critical Stage 5 (Kidney Failure / Dialysis Indication)"
        ctrl["case_label"] = "Case E: Non-CKD Control Patient"

        return [st3, st4, st1_2, st5, ctrl]

    def get_few_shot_exemplars(self, exclude_patient_id: Optional[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """
        Return representative few-shot exemplar patients from the dataset for in-context learning.
        Selects distinct clinical archetypes: early CKD, established moderate CKD, and non-CKD control.
        """
        if not self.patients:
            return []

        exemplars = []

        # Archetype 1: Early CKD (eGFR >= 60 with persistent albuminuria / kidney damage marker)
        ex1 = next((p for p in self.patients if p.get("patient_id") != exclude_patient_id and p.get("egfr") and p["egfr"] >= 60 and (p.get("al") or 0) >= 1), None)
        if ex1:
            exemplars.append({
                "exemplar_id": 1,
                "patient_id": ex1["patient_id"],
                "age": ex1.get("age"),
                "sc": ex1.get("sc"),
                "al": ex1.get("al"),
                "egfr": ex1.get("egfr"),
                "htn": ex1.get("htn"),
                "dm": ex1.get("dm"),
                "summary": f"Age {ex1.get('age')}y, SC {ex1.get('sc')} mg/dL (eGFR {ex1.get('egfr')}), Albumin {ex1.get('al')}+, HTN: {ex1.get('htn')}",
                "gold_class": "CKD",
                "gold_stage": "Stage 2 (Mild decrease in GFR with kidney damage)",
                "clinical_insight": "Presence of albuminuria (kidney damage marker) establishes CKD even when GFR is only mildly decreased (60-89 mL/min)."
            })

        # Archetype 2: Moderate Stage 3 CKD with hypertension and diabetes
        ex2 = next((p for p in self.patients if p.get("patient_id") != exclude_patient_id and p.get("egfr") and 30 <= p["egfr"] < 60 and p.get("htn") == "yes"), None)
        if ex2:
            exemplars.append({
                "exemplar_id": 2,
                "patient_id": ex2["patient_id"],
                "age": ex2.get("age"),
                "sc": ex2.get("sc"),
                "al": ex2.get("al"),
                "egfr": ex2.get("egfr"),
                "htn": ex2.get("htn"),
                "dm": ex2.get("dm"),
                "summary": f"Age {ex2.get('age')}y, SC {ex2.get('sc')} mg/dL (eGFR {ex2.get('egfr')}), Albumin {ex2.get('al')}+, HTN: {ex2.get('htn')}, DM: {ex2.get('dm')}",
                "gold_class": "CKD",
                "gold_stage": "Stage 3 (Moderate decrease in GFR, 30-59 mL/min)",
                "clinical_insight": "eGFR 30-59 mL/min meets Stage 3 criteria regardless of markers; requires aggressive BP control (<130/80) and glycemic optimization."
            })

        # Archetype 3: Non-CKD Normal Control
        ex3 = next((p for p in self.patients if p.get("patient_id") != exclude_patient_id and p.get("class") == "notckd" and (p.get("al") or 0) == 0), None)
        if ex3:
            exemplars.append({
                "exemplar_id": 3,
                "patient_id": ex3["patient_id"],
                "age": ex3.get("age"),
                "sc": ex3.get("sc"),
                "al": ex3.get("al"),
                "egfr": ex3.get("egfr"),
                "htn": ex3.get("htn"),
                "dm": ex3.get("dm"),
                "summary": f"Age {ex3.get('age')}y, SC {ex3.get('sc')} mg/dL (eGFR {ex3.get('egfr')}), Albumin 0, HTN: {ex3.get('htn')}, DM: {ex3.get('dm')}",
                "gold_class": "NOT_CKD",
                "gold_stage": "Not Applicable (Preserved Renal Function)",
                "clinical_insight": "No proteinuria, normal/mildly reduced age-appropriate GFR without damage markers; patient does not meet diagnostic threshold for CKD."
            })

        return exemplars[:count]

    def get_database_summary(self) -> Dict[str, Any]:
        """Return high-level statistics and aggregations over the 400 patient records."""
        total = len(self.patients)
        if total == 0:
            return {}

        ckd_count = sum(1 for p in self.patients if p.get("class") == "ckd")
        notckd_count = sum(1 for p in self.patients if p.get("class") == "notckd")
        htn_count = sum(1 for p in self.patients if p.get("htn") == "yes")
        dm_count = sum(1 for p in self.patients if p.get("dm") == "yes")
        both_htn_dm = sum(1 for p in self.patients if p.get("htn") == "yes" and p.get("dm") == "yes")

        sc_values = [p["sc"] for p in self.patients if p.get("sc") is not None]
        avg_sc = round(sum(sc_values) / len(sc_values), 2) if sc_values else 0.0

        egfr_values = [p["egfr"] for p in self.patients if p.get("egfr") is not None]
        avg_egfr = round(sum(egfr_values) / len(egfr_values), 1) if egfr_values else 0.0

        return {
            "total_patients": total,
            "ckd_patients": ckd_count,
            "non_ckd_patients": notckd_count,
            "hypertension_patients": htn_count,
            "diabetes_patients": dm_count,
            "both_hypertension_and_diabetes": both_htn_dm,
            "average_serum_creatinine": avg_sc,
            "average_egfr": avg_egfr
        }

patient_db = PatientDataset()
