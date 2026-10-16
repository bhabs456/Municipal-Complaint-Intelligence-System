"""Independent Validation Script for Experiment v4
Municipal Complaint Intelligence System (MCIS)
"""
import hashlib
import json
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.taxonomy_rules import APPROVED_TAXONOMY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ValidateV4")

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def run_validation():
    logger.info("==================================================================")
    logger.info("Validating Experiment v4 Artifacts & Taxonomy")
    logger.info("==================================================================")
    errors = []

    # 1. Raw CSV Checksum
    raw_path = BACKEND_DIR / "app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv"
    if not raw_path.exists():
        errors.append("Raw CSV not found!")
    else:
        raw_sha = compute_sha256(raw_path)
        if raw_sha != "d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac":
            errors.append(f"Raw CSV modified! {raw_sha}")
        else:
            logger.info("✓ [Invariant 1] Raw CSV bitwise immutable.")

    # 2. Taxonomy Integrity
    cats = set(k[0] for k in APPROVED_TAXONOMY.keys())
    subcats = set(k[1] for k in APPROVED_TAXONOMY.keys())
    depts = set(APPROVED_TAXONOMY.values())
    if len(cats) != 6 or len(subcats) != 24 or len(depts) != 9:
        errors.append("Taxonomy corrupted!")
    else:
        logger.info(f"✓ [Invariant 2] Locked taxonomy confirmed: 6 cats, 24 subcats, 9 depts.")

    # 3. Deliverables Existence
    reqs = [
        "experiment_config.json",
        "dataset_manifest.json",
        "training_report.md",
        "evaluation_report.md",
        "per_class_metrics.csv",
        "prediction_transition_counts.csv",
        "row_by_row_test_predictions.csv",
        "task2_category_classifier_v4.joblib",
        "task3_subcategory_classifier_bundle_v4.joblib",
    ]
    for rf in reqs:
        if not (CURRENT_DIR / rf).exists():
            errors.append(f"Missing deliverable: {rf}")
    if not errors:
        logger.info(f"✓ [Invariant 3] All required deliverables exist in {CURRENT_DIR.name}.")

    # 4. Model Load & Predict
    try:
        t2 = joblib.load(CURRENT_DIR / "task2_category_classifier_v4.joblib")
        p = t2.predict(["Potholes and broken asphalt on Ring Road"])[0]
        if p != "Roads & Footpaths":
            errors.append(f"Unexpected prediction: {p}")
        logger.info(f"✓ [Invariant 4] Task 2 model loads and predicts correctly (query -> '{p}').")
    except Exception as e:
        errors.append(f"Failed to load Task 2 model: {e}")

    try:
        t3 = joblib.load(CURRENT_DIR / "task3_subcategory_classifier_bundle_v4.joblib")
        branches = t3["branches"]
        if len(branches) != 6:
            errors.append(f"Branches count mismatch: {len(branches)}")
        logger.info(f"✓ [Invariant 5] Task 3 hierarchical bundle verified (6 branches).")
    except Exception as e:
        errors.append(f"Failed to load Task 3 bundle: {e}")

    if errors:
        logger.error(f"Validation FAILED with {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)
    else:
        logger.info("==================================================================")
        logger.info("ALL VALIDATION INVARIANTS PASSED SUCCESSFULLY!")
        logger.info("==================================================================")

if __name__ == "__main__":
    run_validation()
