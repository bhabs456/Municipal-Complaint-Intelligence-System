"""Batch 2 Automated Validation Script
Municipal Complaint Intelligence System

Executes independent invariant checks against all Batch 2 deliverables.
Exits with code 0 on complete validation pass, code 1 on any invariant failure.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
BATCH_2_DIR = BACKEND_DIR / "app/data/processed/batch_2"
RAW_CSV_PATH = BACKEND_DIR / "app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv"
EXPECTED_RAW_SHA = "d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac"


def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_batch_2():
    print("=" * 70)
    print("STARTING INDEPENDENT BATCH 2 VALIDATION SUITE")
    print("=" * 70)

    # 1. Raw File Checksum
    raw_sha = compute_sha256(RAW_CSV_PATH)
    assert raw_sha == EXPECTED_RAW_SHA, f"FAIL: Raw file altered! {raw_sha} != {EXPECTED_RAW_SHA}"
    print("✓ Check 1: Raw CSV SHA-256 untouched and verified.")

    # 2. Master Dataset Row Count & Traceability
    master_path = BATCH_2_DIR / "prepared_master_dataset.csv"
    assert master_path.exists(), "FAIL: prepared_master_dataset.csv missing"
    df = pd.read_csv(master_path)
    assert len(df) == 16071, f"FAIL: Master row count {len(df)} != 16,071"
    assert (df["source_row_id"].values == np.arange(16071)).all(), "FAIL: source_row_id non-monotonic"
    print(f"✓ Check 2: Master dataset has exactly {len(df):,} rows with continuous source_row_id (0..16,070).")

    # 3. Text Quality Flags Consistency
    assert "combined_text" in df.columns, "FAIL: combined_text column missing"
    assert "text_word_count" in df.columns, "FAIL: text_word_count column missing"
    assert "is_empty_text" in df.columns, "FAIL: is_empty_text column missing"
    assert "is_short_text" in df.columns, "FAIL: is_short_text column missing"
    assert "is_potentially_noisy" in df.columns, "FAIL: is_potentially_noisy column missing"
    assert df["is_empty_text"].sum() == 0, "FAIL: Empty text detected"
    print(f"✓ Check 3: Text fields and quality flags validated (short: {df['is_short_text'].sum()}, noisy: {df['is_potentially_noisy'].sum()}, empty: 0).")

    # 4. Duplicate Group Leakage
    assert "duplicate_group_id" in df.columns, "FAIL: duplicate_group_id missing"
    assert "split" in df.columns, "FAIL: split column missing"
    leakage = df.groupby("duplicate_group_id")["split"].nunique()
    cross_splits = (leakage > 1).sum()
    assert cross_splits == 0, f"FAIL: {cross_splits} duplicate groups cross splits"
    print("✓ Check 4: Zero duplicate group leakage across train/val/test splits.")

    # 5. Split Proportions & Totals
    split_counts = df["split"].value_counts().to_dict()
    assert sum(split_counts.values()) == 16071, "FAIL: Split sum mismatch"
    assert split_counts["train"] > 12000, "FAIL: Train split too small"
    assert split_counts["val"] > 1500, "FAIL: Val split too small"
    assert split_counts["test"] > 1500, "FAIL: Test split too small"
    print(f"✓ Check 5: Split totals verified (Train: {split_counts['train']:,}, Val: {split_counts['val']:,}, Test: {split_counts['test']:,}).")

    # 6. Locked Taxonomy Representation in All Splits
    civic = df[df["civic_classification"] == "civic"]
    assert len(civic) == 13150, f"FAIL: Civic count {len(civic)} != 13,150"
    for s in ["train", "val", "test"]:
        s_civic = civic[civic["split"] == s]
        assert s_civic["target_category"].nunique() == 6, f"FAIL: Split {s} missing categories"
        assert s_civic["target_subcategory"].nunique() == 24, f"FAIL: Split {s} missing subcategories"
        assert s_civic["target_department"].nunique() == 9, f"FAIL: Split {s} missing departments"
    print("✓ Check 6: All 6 categories, 24 subcategories, and 9 departments represented in Train, Val, and Test.")

    # 7. Task 1: Civic Classification Dataset
    t1_path = BATCH_2_DIR / "task1_civic_classification.csv"
    assert t1_path.exists(), "FAIL: task1_civic_classification.csv missing"
    t1_df = pd.read_csv(t1_path)
    assert len(t1_df) == 15706, f"FAIL: Task 1 row count {len(t1_df)} != 15,706"
    assert set(t1_df["is_civic"].unique()) == {0, 1}, "FAIL: Task 1 target values invalid"
    assert "target_category" not in t1_df.columns, "FAIL: Target leakage in Task 1"
    assert "target_subcategory" not in t1_df.columns, "FAIL: Target leakage in Task 1"
    assert "target_department" not in t1_df.columns, "FAIL: Target leakage in Task 1"
    print(f"✓ Check 7: Task 1 dataset validated ({len(t1_df):,} rows, no target leakage).")

    # 8. Task 2: Category Classification Dataset
    t2_path = BATCH_2_DIR / "task2_category_classification.csv"
    assert t2_path.exists(), "FAIL: task2_category_classification.csv missing"
    t2_df = pd.read_csv(t2_path)
    assert len(t2_df) == 13150, f"FAIL: Task 2 row count {len(t2_df)} != 13,150"
    assert t2_df["target_category"].nunique() == 6, "FAIL: Task 2 category count != 6"
    assert "target_subcategory" not in t2_df.columns, "FAIL: Child target leakage in Task 2"
    assert "target_department" not in t2_df.columns, "FAIL: Department leakage in Task 2"
    print(f"✓ Check 8: Task 2 dataset validated ({len(t2_df):,} rows, exactly 6 categories, no leakage).")

    # 9. Task 3: Subcategory Classification Dataset
    t3_path = BATCH_2_DIR / "task3_subcategory_classification.csv"
    assert t3_path.exists(), "FAIL: task3_subcategory_classification.csv missing"
    t3_df = pd.read_csv(t3_path)
    assert len(t3_df) == 13150, f"FAIL: Task 3 row count {len(t3_df)} != 13,150"
    assert t3_df["target_subcategory"].nunique() == 24, "FAIL: Task 3 subcategory count != 24"
    assert "target_department" not in t3_df.columns, "FAIL: Target department used as feature in Task 3"
    print(f"✓ Check 9: Task 3 dataset validated ({len(t3_df):,} rows, exactly 24 subcategories, no leakage).")

    # 10. Task Exclusions & Config
    excl_path = BATCH_2_DIR / "task_exclusion_report.csv"
    assert excl_path.exists(), "FAIL: task_exclusion_report.csv missing"
    excl_df = pd.read_csv(excl_path)
    assert len(excl_df) == 3, f"FAIL: Expected 3 task exclusion records, got {len(excl_df)}"
    cfg_path = BATCH_2_DIR / "config_batch_2.json"
    assert cfg_path.exists(), "FAIL: config_batch_2.json missing"
    print("✓ Check 10: Task exclusion report and reproducibility config verified.")

    print("=" * 70)
    print("ALL 10 BATCH 2 VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = validate_batch_2()
    sys.exit(0 if success else 1)
