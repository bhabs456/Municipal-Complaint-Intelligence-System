# Batch 1 Validation Report: Safe Dataset Preparation (Phases 2.2 – 2.5)

**Project:** Municipal Complaint Intelligence System  
**Dataset:** Bengaluru / I Change My City Civic Complaints (16,071 records)  
**Execution Timestamp:** 2026-10-16 15:47:29 UTC  
**Status:** VALIDATION PASSED (All invariants strictly satisfied)

---

## 1. Executive Summary & Core Reconciliations

| Metric | Value | Invariant Check |
| :--- | :--- | :--- |
| **Input Raw Dataset Path** | `backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv` | Verified present |
| **Raw File SHA-256 (Pre & Post)** | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` | **UNTOUCHED / IDENTICAL** |
| **Input Row Count** | **16,071** | Pre-condition verified |
| **Cleaned Output Row Count** | **16,071** | **100.0% RECONCILED (0 rows lost)** |
| **Traceability Identifier** | `source_row_id` (0 to 16,070) | Continuous & 1:1 aligned |
| **Distinct Raw (Category, Subcategory) Pairs** | **231** | 100% processed |
| **Civic Complaints (Normalized & In-Scope)** | **13,150** (81.82%) | Bound to locked taxonomy |
| **Non-Civic Complaints (Out of Scope / Deferred)** | **2,556** (15.90%) | Preserved with reasons |
| **Uncertain Complaints (Review Queue)** | **365** (2.27%) | Preserved for manual review |
| **Exact Duplicate Rows (All 17 cols identical)** | **59** (33 repeated occurrences) | **Flagged, NOT deleted** |
| **Duplicate Text Pairs (Title + Desc identical)** | **834** (586 repeated occurrences) | **Flagged, NOT deleted** |
| **Approved Locked Categories** | **6** | Exactly 6 represented |
| **Approved Locked Subcategories** | **24** | Exactly 24 represented |
| **Approved Locked Departments** | **9** | Exactly 9 represented |

---

## 2. Invariant Validation Matrix

| Check # | Invariant Rule | Expected | Observed | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Input row count == Output row count | 16,071 | 16071 | **PASS** |
| 2 | Original raw file untouched | SHA-256 match | Identical | **PASS** |
| 3 | All original 17 columns preserved | 17 columns | 17 columns present | **PASS** |
| 4 | Stable row identifier `source_row_id` | 0 to 16,070 monotonic | Monotonic 0..16,070 | **PASS** |
| 5 | Locked taxonomy category count | Exactly 6 | 6 | **PASS** |
| 6 | Locked taxonomy subcategory count | Exactly 24 | 24 | **PASS** |
| 7 | Locked taxonomy department count | Exactly 9 | 9 | **PASS** |
| 8 | Subcategory to department 1:1 integrity | Exact match per taxonomy | 100% match | **PASS** |
| 9 | Non-civic records null taxonomy integrity | target_category is NULL | 100% NULL | **PASS** |
| 10 | No silent row drops | 0 dropped | 0 dropped | **PASS** |

---

## 3. Civic Classification Breakdown

| Classification | Count | Percentage | Description / Handling Policy |
| :--- | :--- | :--- | :--- |
| **Civic** | **13,150** | **81.82%** | Directly matches municipal service delivery within the locked 6 categories. Ready for Phase 2.6 text processing and model training. |
| **Non-Civic** | **2,556** | **15.90%** | State-level or external agency jurisdiction (e.g., Police traffic fines, BMTC bus routes, voter cards, commercial trade licenses). Preserved with explicit audit reasons. |
| **Uncertain** | **365** | **2.27%** | Generic labels (`Others`, `Others - Others`, blank) lacking category context. Preserved in `unmapped_review_queue.csv` for human review. |
| **Total** | **16,071** | **100.00%** | Full dataset coverage. |

---

## 4. Locked Taxonomy Distribution (Civic Complaints: N = 13,150)

### 4.1 By Category (6 Categories)
| Target Category | Record Count | % of Civic Dataset |
| :--- | :--- | :--- |
| Roads & Footpaths | 5,173 | 39.34% |
| Waste & Sanitation | 4,683 | 35.61% |
| Street Lighting & Electricity | 1,762 | 13.40% |
| Sewage & Drainage | 689 | 5.24% |
| Pollution | 502 | 3.82% |
| Water Supply | 341 | 2.59% |

### 4.2 By Department (9 Departments)
| Target Department | Record Count | % of Civic Dataset |
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

### 4.3 By Subcategory (24 Subcategories)
| Target Subcategory | Record Count | % of Civic Dataset |
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

---

## 5. Duplicate Records & Data Quality Flags

| Quality Flag | Flagged Count | Percentage | Action Taken |
| :--- | :--- | :--- | :--- |
| `is_exact_duplicate_row` | **59** | **0.37%** | Flagged as exact duplicates across all 17 raw fields. Preserved in dataset for full auditability. |
| `is_duplicate_text_pair` | **834** | **5.19%** | Flagged as duplicate complaint text (title + description identical). Preserved for deduplication modeling. |
| `has_missing_location` | **0** | **0.00%** | Flagged where latitude or longitude is null or zero. |
| `has_missing_category` | **32** | **0.20%** | Flagged where category or subcategory was blank in raw data. |

---

## 6. Generated Deliverables Inventory

1. `backend/app/data/processed/cleaned_working_dataset.csv` (16,071 rows, 26 columns)
2. `backend/app/data/processed/raw_to_standard_subcategory_mapping.csv` (231 distinct raw pairs)
3. `backend/app/data/processed/final_taxonomy_mapping.csv` & `final_taxonomy_mapping.json` (Locked 6/24/9 mapping)
4. `backend/app/data/processed/civic_classification.csv` (16,071 rows with audit reasons)
5. `backend/app/data/processed/unmapped_review_queue.csv` (365 review records)
6. `backend/app/data/processed/transformation_log.jsonl` (Auditable transformation event log)
7. `backend/app/data/reports/batch_1_validation_report.md` (This document)

---

## 7. Next Steps (Phase 2.6+)
- Data preparation for Batch 1 is officially complete and validated.
- Proceed to Phase 2.6 (Feature Engineering & Text Preprocessing) using `cleaned_working_dataset.csv` filtered by `civic_classification == 'civic'`.
