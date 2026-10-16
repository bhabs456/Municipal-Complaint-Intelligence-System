"""Standalone Automated Validation Suite for Dataset Reduction v1
Municipal Complaint Intelligence System (MCIS)
Verifies all 10 non-negotiable invariants, checksums, distributions, and leakage controls.
"""

import hashlib
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetReductionValidate")

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent.parent
RAW_CSV_PATH = DATA_DIR / "raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv"
TAXONOMY_PATH = DATA_DIR / "processed/final_taxonomy_mapping.json"
EXPECTED_RAW_SHA = "d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac"


def compute_sha256(filepath: Path) -> str:
    s = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            s.update(chunk)
    return s.hexdigest()


def run_validation():
    logger.info("=" * 80)
    logger.info("RUNNING DATASET REDUCTION V1 VALIDATION SUITE")
    logger.info("=" * 80)

    # 1. Raw CSV Checksum
    raw_sha = compute_sha256(RAW_CSV_PATH)
    assert raw_sha == EXPECTED_RAW_SHA, f"Raw CSV altered: {raw_sha}"
    logger.info("[PASS] Check 1: Raw CSV SHA-256 untouched and verified.")

    # 2. Manifest and Files existence
    manifest_path = CURRENT_DIR / "dataset_manifest.json"
    assert manifest_path.exists(), "Manifest missing!"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info("[PASS] Check 2: Manifest exists and is readable.")

    # 3. Audit reconciliation
    audit_path = CURRENT_DIR / "dataset_reduction_audit.csv"
    assert audit_path.exists(), "Audit CSV missing!"
    df_audit = pd.read_csv(audit_path)
    assert len(df_audit) == 16071, f"Audit row count mismatch: {len(df_audit)}"
    disp_counts = df_audit["disposition"].value_counts().to_dict()
    assert disp_counts.get("IN_SCOPE", 0) == 13142
    assert disp_counts.get("OUT_OF_SCOPE", 0) == 2564
    assert disp_counts.get("REVIEW_REQUIRED", 0) == 365
    assert disp_counts.get("MAPPING_ERROR", 0) == 0
    assert sum(disp_counts.values()) == 16071
    logger.info(f"[PASS] Check 3: Audit dispositions reconcile exactly to 16,071 ({disp_counts}).")

    # 4. Scope filtered dataset
    scope_path = CURRENT_DIR / "scope_filtered_dataset.csv"
    assert scope_path.exists(), "Scope filtered CSV missing!"
    df_scope = pd.read_csv(scope_path)
    assert len(df_scope) == 13142, f"Scope filtered row count mismatch: {len(df_scope)}"
    logger.info(f"[PASS] Check 4: Scope filtered dataset verified (13,142 records).")

    # 5. Training dataset
    train_path = CURRENT_DIR / "training_dataset.csv"
    assert train_path.exists(), "Training CSV missing!"
    df_train = pd.read_csv(train_path)
    assert len(df_train) == 8191, f"Training dataset total row count mismatch: {len(df_train)}"
    train_split_counts = df_train["split"].value_counts().to_dict()
    assert train_split_counts.get("train", 0) == 5585, f"Train split count mismatch: {train_split_counts}"
    assert train_split_counts.get("val", 0) == 1317, f"Val split count mismatch: {train_split_counts}"
    assert train_split_counts.get("test", 0) == 1289, f"Test split count mismatch: {train_split_counts}"
    logger.info(f"[PASS] Check 5: Training dataset verified ({train_split_counts}).")

    # 6. Taxonomy conformance
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        tax_raw = json.load(f)
    approved_subcats = set()
    for entry in tax_raw["taxonomy"]:
        approved_subcats.add(entry["subcategory"])

    scope_subcats = set(df_scope["target_subcategory"].dropna().unique())
    train_subcats = set(df_train["target_subcategory"].dropna().unique())
    assert scope_subcats.issubset(approved_subcats), f"Unapproved subcategory in scope: {scope_subcats - approved_subcats}"
    assert train_subcats.issubset(approved_subcats), f"Unapproved subcategory in train: {train_subcats - approved_subcats}"
    assert len(scope_subcats) == 24, f"Expected 24 subcategories in scope, found {len(scope_subcats)}"
    assert len(train_subcats) == 24, f"Expected 24 subcategories in train, found {len(train_subcats)}"
    logger.info(f"[PASS] Check 6: All 24 subcategories strictly conform to approved taxonomy.")

    # 7. Distribution reports
    cat_dist_path = CURRENT_DIR / "category_distribution.csv"
    sub_dist_path = CURRENT_DIR / "subcategory_distribution.csv"
    assert cat_dist_path.exists() and sub_dist_path.exists()
    df_cat_dist = pd.read_csv(cat_dist_path)
    df_sub_dist = pd.read_csv(sub_dist_path)
    assert len(df_cat_dist) == 6, f"Expected 6 categories, got {len(df_cat_dist)}"
    assert len(df_sub_dist) == 24, f"Expected 24 subcategories, got {len(df_sub_dist)}"
    logger.info(f"[PASS] Check 7: Distribution tables complete (6 categories, 24 subcategories).")

    # 8. Unmapped labels report
    unmapped_path = CURRENT_DIR / "unmapped_labels.csv"
    assert unmapped_path.exists()
    df_unmapped = pd.read_csv(unmapped_path)
    assert len(df_unmapped) > 0
    assert df_unmapped["frequency"].sum() == (2564 + 365)
    logger.info(f"[PASS] Check 8: Unmapped labels report verified ({len(df_unmapped)} distinct pairs, sum={df_unmapped['frequency'].sum()}).")

    # 9. Duplicate leakage check across splits
    for name, df_check in [("scope_filtered_dataset", df_scope), ("training_dataset", df_train)]:
        splits_per_group = df_check.groupby("duplicate_group_id")["split"].nunique()
        leakage = (splits_per_group > 1).sum()
        assert leakage == 0, f"Duplicate leakage in {name}: {leakage} groups cross splits!"
    logger.info(f"[PASS] Check 9: Zero duplicate group leakage across train/val/test splits.")

    # 10. Traceability of source row IDs
    assert df_scope["source_row_id"].nunique() == len(df_scope)
    assert set(df_train["source_row_id"]).issubset(set(df_scope["source_row_id"]))
    logger.info(f"[PASS] Check 10: Source row IDs strictly preserved and traceable.")

    logger.info("=" * 80)
    logger.info("ALL 10 VALIDATION CHECKS PASSED SUCCESSFULLY (PASS)")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
