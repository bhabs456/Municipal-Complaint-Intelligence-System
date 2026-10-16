"""Batch 3 Automated Validation Script
Municipal Complaint Intelligence System

Executes independent invariant checks against all Batch 3 model artifacts and evaluation deliverables.
Exits with code 0 on complete validation pass, code 1 on any invariant failure.
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.taxonomy_rules import APPROVED_TAXONOMY

BATCH_3_DATA_DIR = BACKEND_DIR / "app/data/processed/batch_3"
BATCH_3_MODELS_DIR = BACKEND_DIR / "app/ml/saved_models/batch_3"
RAW_CSV_PATH = BACKEND_DIR / "app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv"
EXPECTED_RAW_SHA = "d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac"


def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_batch_3():
    print("=" * 70)
    print("STARTING INDEPENDENT BATCH 3 VALIDATION SUITE")
    print("=" * 70)

    # 1. Raw File Checksum
    raw_sha = compute_sha256(RAW_CSV_PATH)
    assert raw_sha == EXPECTED_RAW_SHA, f"FAIL: Raw file altered! {raw_sha} != {EXPECTED_RAW_SHA}"
    print("✓ Check 1: Raw CSV SHA-256 untouched and verified.")

    # 2. Task 1 Model Artifact & Inference
    t1_model_path = BATCH_3_MODELS_DIR / "task1_civic_classifier.joblib"
    assert t1_model_path.exists(), "FAIL: task1_civic_classifier.joblib missing"
    t1_pipe = joblib.load(t1_model_path)
    sample_text = ["Pothole on 100ft road causing traffic block", "Need passport verification status"]
    preds = t1_pipe.predict(sample_text)
    probs = t1_pipe.predict_proba(sample_text)
    assert len(preds) == 2, "FAIL: Task 1 inference failed"
    assert probs.shape == (2, 2), "FAIL: Task 1 probabilities shape invalid"
    print("✓ Check 2: Task 1 model loaded and verified inference.")

    # 3. Task 2 Model Artifact & Inference
    t2_model_path = BATCH_3_MODELS_DIR / "task2_category_classifier.joblib"
    assert t2_model_path.exists(), "FAIL: task2_category_classifier.joblib missing"
    t2_pipe = joblib.load(t2_model_path)
    cat_preds = t2_pipe.predict(sample_text)
    cat_probs = t2_pipe.predict_proba(sample_text)
    assert len(cat_preds) == 2, "FAIL: Task 2 inference failed"
    assert cat_probs.shape[1] == 6, f"FAIL: Expected 6 categories, got {cat_probs.shape[1]}"
    print("✓ Check 3: Task 2 model loaded and verified 6-category inference.")

    # 4. Task 3 Hierarchical Bundle & Structural Invariants
    t3_model_path = BATCH_3_MODELS_DIR / "task3_subcategory_classifier_bundle.joblib"
    assert t3_model_path.exists(), "FAIL: task3_subcategory_classifier_bundle.joblib missing"
    t3_bundle = joblib.load(t3_model_path)
    assert t3_bundle["type"] == "category_aware_hierarchical", "FAIL: Invalid Task 3 bundle type"
    assert len(t3_bundle["branches"]) == 6, f"FAIL: Expected 6 branches, got {len(t3_bundle['branches'])}"
    for cat, branch in t3_bundle["branches"].items():
        sub_preds = branch.predict(["Sample complaint"])
        sub_probs = branch.predict_proba(["Sample complaint"])
        assert len(sub_preds) == 1, f"FAIL: Branch {cat} inference failed"
        for sub in branch.classes_:
            expected_dept = APPROVED_TAXONOMY.get((cat, sub))
            assert expected_dept is not None, f"FAIL: Taxonomy violation in branch {cat} -> {sub}"
    print("✓ Check 4: Task 3 category-aware hierarchical bundle loaded and verified 100% taxonomy conformity.")

    # 5. Deterministic Department Mapping
    assert len(APPROVED_TAXONOMY) == 24, "FAIL: Approved taxonomy count != 24"
    unique_depts = set(APPROVED_TAXONOMY.values())
    assert len(unique_depts) == 9, f"FAIL: Expected 9 departments, got {len(unique_depts)}"
    print("✓ Check 5: Deterministic subcategory-to-department mapping verified (exactly 9 departments).")

    # 6. Confidence Configuration
    conf_path = BATCH_3_MODELS_DIR / "confidence_config.json"
    assert conf_path.exists(), "FAIL: confidence_config.json missing"
    with open(conf_path) as f:
        conf_cfg = json.load(f)
    assert "task1_civic_confidence_threshold" in conf_cfg, "FAIL: Task 1 threshold missing"
    assert "task2_category_confidence_threshold" in conf_cfg, "FAIL: Task 2 threshold missing"
    assert "task3_subcategory_confidence_threshold" in conf_cfg, "FAIL: Task 3 threshold missing"
    print(f"✓ Check 6: Operational confidence thresholds verified ({conf_cfg['task1_civic_confidence_threshold']}, {conf_cfg['task2_category_confidence_threshold']}, {conf_cfg['task3_subcategory_confidence_threshold']}).")

    # 7. Model Metadata File
    meta_path = BATCH_3_MODELS_DIR / "model_metadata.json"
    assert meta_path.exists(), "FAIL: model_metadata.json missing"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["task1"]["test_macro_f1"] > 0.85, "FAIL: Task 1 test macro-F1 too low"
    assert meta["task2"]["test_macro_f1"] > 0.75, "FAIL: Task 2 test macro-F1 too low"
    assert meta["task3"]["test_accuracy_e2e"] > 0.65, "FAIL: Task 3 test e2e accuracy too low"
    print(f"✓ Check 7: Model metadata verified (T1 Macro-F1: {meta['task1']['test_macro_f1']:.4f}, T2 Macro-F1: {meta['task2']['test_macro_f1']:.4f}, T3 E2E Acc: {meta['task3']['test_accuracy_e2e']*100:.2f}%).")

    # 8. Evaluation CSVs & Manifests
    required_csvs = [
        "model_comparison_task1.csv",
        "model_comparison_task2.csv",
        "model_comparison_task3.csv",
        "per_class_metrics_task1_test.csv",
        "per_class_metrics_task2_test.csv",
        "per_class_metrics_task3_test_e2e.csv",
        "per_class_metrics_department_test.csv",
        "confidence_calibration_report.csv",
        "misclassifications_task1.csv",
        "misclassifications_task2.csv",
        "misclassifications_task3.csv",
    ]
    for csv_name in required_csvs:
        p = BATCH_3_DATA_DIR / csv_name
        assert p.exists() and p.stat().st_size > 0, f"FAIL: {csv_name} missing or empty"
    print(f"✓ Check 8: All {len(required_csvs)} evaluation CSVs and audit manifests verified present and non-empty.")

    # 9. Confusion Matrices JSON
    cm_path = BATCH_3_DATA_DIR / "confusion_matrices.json"
    assert cm_path.exists(), "FAIL: confusion_matrices.json missing"
    with open(cm_path) as f:
        cms = json.load(f)
    assert "task1_civic_test" in cms, "FAIL: Task 1 CM missing"
    assert "task2_category_test" in cms, "FAIL: Task 2 CM missing"
    assert "task3_subcategory_test_e2e" in cms, "FAIL: Task 3 CM missing"
    assert "department_assignment_test" in cms, "FAIL: Dept CM missing"
    print("✓ Check 9: Full confusion matrices for all 4 stages verified.")

    # 10. Audit Report
    report_path = BACKEND_DIR / "app/data/reports/batch_3_validation_report.md"
    assert report_path.exists() and report_path.stat().st_size > 1000, "FAIL: Validation report missing or incomplete"
    print("✓ Check 10: Final Batch 3 Markdown audit report verified.")

    print("=" * 70)
    print("ALL 10 BATCH 3 VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = validate_batch_3()
    sys.exit(0 if success else 1)
