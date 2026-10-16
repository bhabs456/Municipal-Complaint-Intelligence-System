# Batch 2 Validation & Audit Report: Phases 2.6 – 2.10
## Municipal Complaint Intelligence System

**Execution Timestamp:** 2026-10-16 15:58:22 UTC  
**Random Seed:** 42 (reproducible `train_test_split`)  
**Status:** VALIDATION PASSED (All 10 invariants strictly satisfied)

---

## 1. Executive Summary & Master Reconciliation

| Metric | Count / Value | Validation Check |
| :--- | :--- | :--- |
| **Raw Input Dataset SHA-256** | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` | **UNTOUCHED & IDENTICAL** |
| **Master Input Rows (from Batch 1)** | **16,071** | Verified 100% matched |
| **Master Output Rows (Batch 2)** | **16,071** | **100.0% RECONCILED (0 rows lost)** |
| **Traceability Identifier** | `source_row_id` (0 to 16,070) | Continuous & 100% unique |
| **Unique Duplicate Text Groups** | **15,477** groups | Derived from `(clean_title, clean_description)` |
| **Duplicate Leakage Across Splits** | **0 groups** | **STRICT ZERO LEAKAGE VERIFIED** |
| **Locked Categories** | **6** | Exactly 6 represented (all in train/val/test) |
| **Locked Subcategories** | **24** | Exactly 24 represented (all in train/val/test) |
| **Locked Departments** | **9** | Exactly 9 represented |

---

## 2. Phase 2.6: Text Preparation & Quality Flagging

Combined text field `combined_text` was constructed conservatively from `clean_title` and `clean_description`:
- Preserves all numbers (e.g. `4th cross`, `80 feet road`, `ward 150`, `3 days`).
- Preserves negation tokens (`not`, `never`, `no`, `without`, `cannot`).
- Preserves location terminology (`road`, `street`, `cross`, `main`, `layout`, `nagar`, `circle`, `halli`).
- Standardizes punctuation boundary between title and description.
- Preserves original raw fields intact without destructive overwriting.

### Text Quality Metrics & Flags (N = 16,071)
| Quality Metric / Flag | Count | Percentage | Action Taken |
| :--- | :--- | :--- | :--- |
| `text_word_count` (Median) | **35 words** | — | Min: 3, Median: 35, Max: 718 |
| `text_char_count` (Median) | **213 chars** | — | Min: 18, Median: 213, Max: 3,892 |
| `is_empty_text` | **0** | **0.00%** | Zero complaints have empty text |
| `is_short_text` (<6 words or <25 chars) | **378** | **2.35%** | **Flagged, NOT deleted** (preserved in master dataset) |
| `is_potentially_noisy` (repetition / junk ratio) | **78** | **0.49%** | **Flagged, NOT deleted** (preserved in master dataset) |

---

## 3. Phase 2.7: Class Distribution & Imbalance Analysis

### 3.1 Civic Domain Distribution (Master Dataset: N = 16,071)
| Domain Class | Count | Percentage | Task Eligibility |
| :--- | :--- | :--- | :--- |
| **Civic (In-Scope)** | **13,150** | **81.82%** | Eligible for Tasks 1, 2, and 3 |
| **Non-Civic (External Jurisdictions)** | **2,556** | **15.90%** | Eligible as negative class for Task 1; Excluded from Tasks 2 & 3 |
| **Uncertain (Generic Labels)** | **365** | **2.27%** | Excluded from supervised training (requires manual review) |
| **Total** | **16,071** | **100.00%** | 100.0% coverage |

### 3.2 Locked Category Distribution (Civic Records: N = 13,150)
| Category | Record Count | % of Civic Dataset |
| :--- | :--- | :--- |
| Roads & Footpaths | 5,173 | 39.34% |
| Waste & Sanitation | 4,683 | 35.61% |
| Street Lighting & Electricity | 1,762 | 13.40% |
| Sewage & Drainage | 689 | 5.24% |
| Pollution | 502 | 3.82% |
| Water Supply | 341 | 2.59% |

### 3.3 Locked Department Distribution (Civic Records: N = 13,150)
| Department | Record Count | % of Civic Dataset |
| :--- | :--- | :--- |
| Roads Department | 4,947 | 37.62% |
| Waste Management Department | 3,870 | 29.43% |
| Street Lighting Department | 1,506 | 11.45% |
| Sanitation Department | 813 | 6.18% |
| Drainage Department | 586 | 4.46% |
| Pollution Control Department | 502 | 3.82% |
| Water Supply Department | 341 | 2.59% |
| Sewerage Department | 329 | 2.50% |
| Electrical Department | 256 | 1.95% |

### 3.4 Locked Subcategory Distribution (24 Subcategories: N = 13,150)
| Subcategory | Record Count | % of Civic Dataset |
| :--- | :--- | :--- |
| Garbage Dumping | 3,379 | 25.70% |
| Potholes & Road Damage | 2,354 | 17.90% |
| Road Repair / Construction | 1,948 | 14.81% |
| Street Light Not Working / Repair | 1,342 | 10.21% |
| Street / Public Area Cleaning | 813 | 6.18% |
| Footpath Damage / Repair | 645 | 4.90% |
| Garbage Collection / Removal | 386 | 2.94% |
| Drain Blockage / Desilting | 270 | 2.05% |
| Noise Pollution | 264 | 2.01% |
| Water Supply Interruption | 243 | 1.85% |
| Road Waterlogging | 226 | 1.72% |
| Sewer Line Repair / Maintenance | 199 | 1.51% |
| Electricity Supply | 194 | 1.48% |
| Street Light Installation | 164 | 1.25% |
| Air Pollution | 132 | 1.00% |
| Sewer Blockage / Overflow | 130 | 0.99% |
| Waste Disposal & Segregation | 105 | 0.80% |
| Storm Water Drainage / Waterlogging | 90 | 0.68% |
| Water Pollution | 83 | 0.63% |
| Electrical Pole / Transformer Issues | 62 | 0.47% |
| Water Leakage | 56 | 0.43% |
| Land / Waste Pollution | 23 | 0.17% |
| Water Pipeline Damage / Maintenance | 22 | 0.17% |
| New Water Connection / Pipeline | 20 | 0.15% |

### 3.5 Class Imbalance & Rare Class Strategy Recommendations
- **Rarest Subcategories**:
  - `New Water Connection / Pipeline`: 20 records (0.15%)
  - `Water Pipeline Damage / Maintenance`: 22 records (0.17%)
  - `Land / Waste Pollution`: 23 records (0.17%)
  - `Water Leakage`: 56 records (0.43%)
  - `Electrical Pole / Transformer Issues`: 62 records (0.47%)
- **Top Subcategories**:
  - `Garbage Dumping`: 3,379 records (25.70%)
  - `Potholes & Road Damage`: 2,354 records (17.90%)
  - `Road Repair / Construction`: 1,948 records (14.81%)
- **Recommended Model Training Strategies (for Phase 2.11+)**:
  1. **Class Weighting (`compute_class_weight='balanced'`)**: Adjust loss function inversely proportional to class frequencies to penalize errors on rare classes without altering data distribution.
  2. **Hierarchical / Category-Aware Modeling (Task 3)**: Condition subcategory prediction on the predicted category. This reduces the classification space from 24 classes to 4–6 subcategories per branch, vastly improving precision on rare water/pollution classes.
  3. **No Resampling Applied in Batch 2**: As mandated, raw distributions are preserved without synthetic sampling (SMOTE) or downsampling in the dataset artifacts.

---

## 4. Phase 2.8: Train / Validation / Test Split Strategy

### Split Protocol
- **Target Ratio**: 80% Train, 10% Validation, 10% Test.
- **Grouping Unit**: `duplicate_group_id` (15,477 unique clusters of identical complaints).
- **Stratification Unit**: 26 fine-grained strata (24 civic subcategories + non-civic + uncertain).
- **Random Seed**: Fixed at `42` for exact reproducibility.

### Split Counts & Proportions
| Partition | Groups Count | % of Groups | Total Records | % of Records |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | **12,381** | **80.00%** | **12,884** | **80.17%** |
| **Validation** | **1,548** | **10.00%** | **1,608** | **10.01%** |
| **Test** | **1,548** | **10.00%** | **1,579** | **9.83%** |
| **Total** | **15,477** | **100.00%** | **16,071** | **100.00%** |

### Duplicate Leakage Verification
- Number of duplicate groups spanning multiple splits: **0 groups (0.00%)**.
- Every duplicate or near-duplicate cluster is contained entirely within a single split partition.

### Temporal & Geographic Leakage Considerations
- The dataset spans 43 months (January 2019 to July 2022) across 198 municipal wards.
- Splitting purely by time would introduce severe non-stationary shift (COVID-19 lockdowns during 2020–2021 drastically suppressed road/street complaints and altered municipal patterns).
- Stratified group splitting guarantees that all 24 subcategories and 6 categories are represented in both Validation and Test sets, while sampling across all wards and seasons evenly.

---

## 5. Phase 2.9: Task-Specific ML Datasets & Leakage Prevention

| Task | Eligible Rows | Train Split | Val Split | Test Split | Excluded Rows | Target Variable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Civic vs. Non-Civic** | **15,706** | 12,593 | 1,570 | 1,543 | 365 (Uncertain) | `is_civic` (1 / 0) |
| **Task 2: 6-Category Classification** | **13,150** | 10,544 | 1,315 | 1,291 | 2,921 (Non-civic + Uncertain) | `target_category` |
| **Task 3: Category-Aware Subcategory** | **13,150** | 10,544 | 1,315 | 1,291 | 2,921 (Non-civic + Uncertain) | `target_subcategory` |

### Task-Specific Exclusion Report
| Task | Eligible | Excluded | Exclusion Justification |
| :--- | :--- | :--- | :--- |
| **Task 1: Civic vs. Non-Civic Classification** | **15,706** | **365** | Generic labels ('Others', 'Others - Others', blank) lacking verified ground truth label |
| **Task 2: 6-Category Classification** | **13,150** | **2,921** | Non-civic complaints (2,556) belong to external agencies outside municipal jurisdiction; uncertain complaints (365) lack ground truth |
| **Task 3: Hierarchical Category-Aware Subcategory Classification** | **13,150** | **2,921** | Non-civic complaints (2,556) outside municipal jurisdiction; uncertain complaints (365) lack ground truth |

### Data Leakage Safeguards
1. **Target Feature Isolation**: `target_category`, `target_subcategory`, and `target_department` are strictly removed from feature columns.
2. **Deterministic Department Mapping**: `target_department` is NEVER an input feature. It is resolved deterministically from `(category, subcategory)` via `final_taxonomy_mapping.json`.
3. **Post-Outcome Feature Removal**: `complaint_status_title` (status at ingestion) and `comment_count` (community engagement) are omitted from ML training sets.
4. **Zero Preprocessing Leakage**: No TF-IDF, CountVectorizer, or learned transformation was fitted on the dataset. Vectorizers will be fit solely on the training splits during Phase 2.11+.

---

## 6. Phase 2.10: Invariant Validation Matrix

| Check # | Invariant Rule | Expected | Observed | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Raw source CSV checksum unaltered | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` | Identical | **PASS** |
| 2 | Master record count maintained | 16,071 | 16,071 | **PASS** |
| 3 | `source_row_id` uniqueness and continuity | Monotonic 0..16,070 | Monotonic 0..16,070 | **PASS** |
| 4 | Zero duplicate leakage across splits | 0 groups | 0 groups | **PASS** |
| 5 | Locked categories represented across all splits | 6 in train/val/test | 6 in train/val/test | **PASS** |
| 6 | Locked subcategories represented across all splits | 24 in train/val/test | 24 in train/val/test | **PASS** |
| 7 | Non-civic & uncertain preserved in master dataset | 2,921 rows | 2,921 rows preserved | **PASS** |
| 8 | Task-specific exclusions fully justified & logged | 100% documented | 100% documented | **PASS** |
| 9 | Feature sets free of target & post-outcome leakage | 0 leaking features | 0 leaking features | **PASS** |
| 10 | Pipeline fully reproducible via fixed seed (42) | Deterministic | Verified deterministic | **PASS** |

---

## 7. Deliverables Manifest (Batch 2)

All artifacts are generated in [`backend/app/data/processed/batch_2/`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/batch_2):
1. `prepared_master_dataset.csv` (16,071 rows, 40 columns)
2. `class_distribution_civic.csv`
3. `class_distribution_category.csv`
4. `class_distribution_subcategory.csv`
5. `class_distribution_department.csv`
6. `split_assignments.csv` (16,071 rows: `source_row_id`, `duplicate_group_id`, `split`)
7. `task1_civic_classification.csv` (15,706 rows)
8. `task2_category_classification.csv` (13,150 rows)
9. `task3_subcategory_classification.csv` (13,150 rows)
10. `task_exclusion_report.csv` (Task exclusion manifests)
11. `config_batch_2.json` (Configuration parameters & seed 42)
12. `transformation_log_batch_2.jsonl` (Auditable event logs)
13. `batch_2_validation_report.md` (This document)
