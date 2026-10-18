"""Quick test script to verify Mini-PID-RAG pipeline execution."""
import json
from pipeline import pipeline
from patient_data import patient_db

def test():
    print("Initializing pipeline...")
    pipeline.initialize()
    print("Getting benchmark cases...")
    cases = patient_db.get_benchmark_cases()
    if not cases:
        print("No cases loaded!")
        return

    test_case = cases[0]  # Case A: Stage 3 CKD
    print(f"Running pipeline on: {test_case.get('case_label')}")
    result = pipeline.run(test_case)

    print("\n=== EXECUTION RESULT ===")
    print("Hop 1 Query:", result["step3_multihop_retrieval"]["hop1_query"])
    print("Hop 2 Query:", result["step3_multihop_retrieval"]["hop2_query"])
    print("\nActor Proposed Stage:", result["step4_actor"].get("ckd_stage"))
    print("Critic Verdict:", result["step5_critic"].get("verdict"), "(Support Score:", result["step5_critic"].get("evidence_support_score"), ")")
    print("\nFinal Output Decision:")
    print(json.dumps(result["final_output"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test()
