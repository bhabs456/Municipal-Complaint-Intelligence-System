# Dataset Reduction and Quality Audit Report (v1)
## Municipal Complaint Intelligence System (MCIS)

**Execution Date:** 2026-10-16 16:44:00 UTC  
**Scope Version:** `dataset_reduction_v1`  
**Status:** VALIDATED & READY FOR RETRAINING  

---

## 1. Executive Summary & Objective

This report details the execution of the **Scope-Based Dataset Reduction and Training Data Preparation** task. Following the successful completion of Stage 2 (Phases 2.1–2.18), the objective of this reduction is to produce a refined, high-integrity training dataset that:
1. **Focuses strictly on municipal public works and services** defined by the locked taxonomy (6 categories, 24 subcategories, 9 departments).
2. **Explicitly excludes non-municipal jurisdictions**, primarily traffic and road-safety enforcement (Traffic Police) and land/lake encroachment enforcement (Revenue/Forest/Encroachment Directorate).
3. **Mitigates severe majority-class imbalance** (reducing subcategory skew from 168.95:1 down to 37.50:1) while retaining 100% of all distinct minority-class examples.
4. **Guarantees zero data leakage**: test (1,289 rows) and validation (1,317 rows) splits remain completely untouched, ensuring honest, unbiased evaluation.

---

## 2. Integrity Verification & Authoritative Source

* **Raw Dataset Path:** [`backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv`](backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv)
* **Raw CSV SHA-256:** `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` (**Verified Untouched**)
* **Authoritative Source Dataset:** [`backend/app/data/processed/batch_2/prepared_master_dataset.csv`](backend/app/data/processed/batch_2/prepared_master_dataset.csv)
* **Total Records Ingested:** 16,071
* **Locked Taxonomy Source:** [`backend/app/data/processed/final_taxonomy_mapping.json`](backend/app/data/processed/final_taxonomy_mapping.json)

---

## 3. Record-Count Reconciliation & Scope Dispositions

Every record in the 16,071-row master dataset was audited and assigned one of four definitive dispositions:

| Disposition | Count | Percentage | Definition & Policy |
| :--- | :---: | :---: | :--- |
| **`IN_SCOPE`** | **13,142** | **81.77%** | Confidently maps to the locked 6 categories and 24 subcategories of municipal civil service. |
| **`OUT_OF_SCOPE`** | **2,564** | **15.95%** | External agency jurisdictions: Traffic Police (967), Animal Husbandry (832), Crime/Patrolling (132), Transport/BMTC (46), Land Encroachment (49), Lake Encroachment (8), Certificates (53), etc. |
| **`REVIEW_REQUIRED`** | **365** | **2.27%** | Completely generic/ambiguous labels (`Others`: 346, `NaN`: 18, `Pension`: 1) lacking verified ground truth. Kept in review queue; excluded from ML training. |
| **`MAPPING_ERROR`** | **0** | **0.00%** | Zero contradictory or invalid taxonomy pairings discovered. |
| **TOTAL** | **16,071** | **100.00%** | **Exact 100% reconciliation across all source records.** |

### Scope Exclusions Detail:
* **Encroachment Enforcement (57 records):**
  * `Community Infrastructure -> Government Land/Property Encroachment`: 49 records (Out of scope).
  * `Lakes -> Removal of Encroachment In Lakes`: 8 records (Excluded from Water Pollution per Constraint 4).
* **Traffic & Road Safety Enforcement (967 records):**
  * Traffic congestion (210), parking violations (343), signal jumping (13), signal maintenance (33), traffic signboards (16), speed breakers/cameras (352). Excluded to avoid confusing municipal civil engineering with police law enforcement.

---

## 4. Class Distribution & Imbalance Audit

### 4.1 Six Approved Categories Distribution

| Category | In-Scope Count | In-Scope % | Train (Reduced) Count | Train (Reduced) % |
| :--- | :---: | :---: | :---: | :---: |
| **Roads & Footpaths** | 5,173 | 39.36% | 1,862 | 33.34% |
| **Waste & Sanitation** | 4,683 | 35.63% | 1,585 | 28.38% |
| **Street Lighting & Electricity** | 1,762 | 13.41% | 933 | 16.71% |
| **Sewage & Drainage** | 689 | 5.24% | 538 | 9.63% |
| **Pollution** | 494 | 3.76% | 396 | 7.09% |
| **Water Supply** | 341 | 2.59% | 271 | 4.85% |
| **Total** | **13,142** | **100.00%** | **5,585** | **100.00%** |

### 4.2 Twenty-Four Approved Subcategories Distribution & Reduction Summary

* **Original In-Scope Subcategory Range:** Min = 20 (`New Water Connection / Pipeline`), Max = 3,379 (`Garbage Dumping`).
* **Original Imbalance Ratio:** $3379 / 20 = 168.95:1$.
* **Training Partition Soft Ceiling (N <= 600):**
  * Top majority classes capped at 600 records in train: `Garbage Dumping` (from 2,574 unique to 600), `Potholes & Road Damage` (from 1,814 to 600), `Road Repair / Construction` (from 1,532 to 600), `Street Light Not Working / Repair` (from 995 to 600).
  * All remaining 20 subcategories: **100% retained** without any downsampling.
* **Reduced Training Imbalance Ratio:** $600 / 16 = 37.50:1$ (**77.8% reduction in majority skew!**).

---

## 5. Partition Strategy & Zero-Leakage Guarantee

To maintain scientific rigor and prevent evaluation distortion:
1. **Validation Split (`val`, 1,317 rows)**: 100% preserved as-is.
2. **Test Split (`test`, 1,289 rows)**: 100% preserved as-is.
3. **Training Split (`train`, 5,585 rows)**: Quality-controlled, deduplicated, and majority-damped.
4. **Zero Duplicate Leakage**: All 12,633 unique `duplicate_group_id` clusters remain strictly confined to their respective splits. Zero clusters cross train, validation, or test boundaries.

---

## 6. Generated Versioned Artifacts

All artifacts are persisted under [`backend/app/data/processed/dataset_reduction_v1/`](backend/app/data/processed/dataset_reduction_v1/):

1. [`scope_filtered_dataset.csv`](scope_filtered_dataset.csv) (13,142 rows): Master in-scope dataset across all splits.
2. [`training_dataset.csv`](training_dataset.csv) (8,191 rows: train=5,585, val=1,317, test=1,289): Retraining-ready dataset.
3. [`dataset_reduction_audit.csv`](dataset_reduction_audit.csv) (16,071 rows): Record-level disposition and quality flags.
4. [`category_distribution.csv`](category_distribution.csv): 6-category frequency tables and percentages.
5. [`subcategory_distribution.csv`](subcategory_distribution.csv): 24-subcategory distribution including departments.
6. [`unmapped_labels.csv`](unmapped_labels.csv): Detailed inventory of out-of-scope and review-required labels.
7. [`dataset_manifest.json`](dataset_manifest.json): Cryptographic provenance, checksums, and parameters.
8. [`dataset_reduction_validate.py`](dataset_reduction_validate.py): Automated validation suite.

---

## 7. Recommendation & Retraining Readiness Verdict

**Verdict: FULLY READY FOR MODEL RETRAINING**

The reduced dataset resolves majority-class over-representation without sacrificing minority-class recall or corrupting evaluation splits. Retraining on `training_dataset.csv` is expected to improve minority class F1 scores, reduce misclassifications into `Garbage Dumping` and `Potholes`, and provide superior real-world generalization across municipal departments.
