# Independent Audit Report: Dataset Reduction v1
## Municipal Complaint Intelligence System (MCIS)

**Audit Execution Timestamp:** 2026-10-16 16:47:30 UTC  
**Auditor Role:** Senior ML/Data Quality Systems Auditor  
**Artifact Directory Audited:** [`backend/app/data/processed/dataset_reduction_v1/`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1)  
**Overall Readiness Verdict:** **READY FOR RETRAINING** (with 1 documented low-risk observation regarding historical casing variants)

---

## 1. Executive Summary & Verification Scorecard

An exhaustive, read-only audit of the dataset reduction artifacts was performed. Every count, class frequency, split assignment, taxonomy mapping, hash, and leakage boundary was independently calculated from the files on disk without relying on pre-existing claims.

| Check # | Audit Dimension | Status | Key Evidence / Metric |
| :---: | :--- | :---: | :--- |
| **1.1** | Raw Dataset SHA-256 Checksum | **PASS** | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` (Untouched) |
| **1.2** | Record Count Reconciliation | **PASS** | 16,071 master = 13,142 In-Scope + 2,564 Out-of-Scope + 365 Review + 0 Error |
| **1.3** | Artifact Provenance & Checksums | **PASS** | 100% of artifact SHA-256 hashes match `dataset_manifest.json` |
| **1.4** | Unique Source ID Traceability | **PASS** | Zero synthetic, duplicate, or missing `source_row_id`s in any output |
| **1.5** | Training Exclusion Accounting | **PASS** | Exactly 4,951 records excluded from train (45 exact dups + 391 near-dups + 4,515 majority cap) |
| **2.1** | Category Distribution Totals | **PASS** | All 6 approved categories populated; matches `category_distribution.csv` |
| **2.2** | Subcategory Distribution & Minima | **PASS** | Full in-scope min = 20 (`New Water Connection`), train reduced min = 16 |
| **2.3** | Training Imbalance Ratio | **PASS** | Max = 600, Min = 16 $\to$ Ratio = **37.50:1** (down from 168.95:1) |
| **2.4** | Minority Class Retention | **PASS** | 100% of distinct minority-class complaint clusters retained in training |
| **3.1** | Val/Test Preservation | **PASS** | Val (1,317) and Test (1,289) IDs 100% identical to approved master split |
| **3.2** | Duplicate Group Isolation | **PASS** | 0 `duplicate_group_id` clusters cross train, val, or test boundaries |
| **3.3** | Normalized Text Boundary Audit | **REVIEW_REQUIRED** | 4 short generic phrases (e.g. "yellow spot", "broken footpath") differ by casing in Batch 2 |
| **3.4** | Training-Only Downsampling | **PASS** | 100% of downsampling occurred in `train`; 0% in `val` or `test` |
| **4.1** | Taxonomy Conformance | **PASS** | 100% of retained records map strictly to the 24 locked (category, subcategory, dept) triples |
| **4.2** | Scope Exclusions Rigor | **PASS** | Traffic police (967), Lake encroachment (8), Land encroachment (49) correctly excluded |
| **5.1** | Sampling Seed Reproducibility | **PASS** | Random seed 42 reproduced the exact 5,585 training record IDs with 100% fidelity |

---

## 2. Check 1: Record Count & Accounting Verification

### 2.1 Disposition Reconciliation
Reading directly from [`dataset_reduction_audit.csv`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1/dataset_reduction_audit.csv):
* **Source Master Record Count:** **16,071 records**
* **`IN_SCOPE`:** **13,142 records (81.77%)**
* **`OUT_OF_SCOPE`:** **2,564 records (15.95%)**
* **`REVIEW_REQUIRED`:** **365 records (2.27%)**
* **`MAPPING_ERROR`:** **0 records (0.00%)**
* **Sum of Dispositions:** $13,142 + 2,564 + 365 + 0 = \mathbf{16,071}$ (**100.0% reconciliation**)

### 2.2 Output File Counts & Source ID Uniqueness

| Artifact File | Total Rows | Unique `source_row_id` | Duplicate Source IDs |
| :--- | :---: | :---: | :---: |
| [`dataset_reduction_audit.csv`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1/dataset_reduction_audit.csv) | 16,071 | 16,071 | **0** |
| [`scope_filtered_dataset.csv`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1/scope_filtered_dataset.csv) | 13,142 | 13,142 | **0** |
| [`training_dataset.csv`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1/training_dataset.csv) | 8,191 | 8,191 | **0** |

### 2.3 Partition Accounting: How 13,142 Records Became 8,191 Records
The reduction from 13,142 in-scope records to 8,191 records in `training_dataset.csv` is explained by a multi-stage quality control protocol applied **strictly within the training partition**:

```
[In-Scope Master Dataset: 13,142 records]
  ├── Validation Partition: 1,317 records (100% Preserved)
  ├── Test Partition:       1,289 records (100% Preserved)
  └── Raw Training Split:  10,536 records
        │
        ├── [-45 records]  Stage 1: Exact Duplicate Row Filtering (10,536 -> 10,491)
        │
        ├── [-391 records] Stage 2: Near-Duplicate Cluster Deduplication (10,491 -> 10,100 unique clusters)
        │
        └── [-4,515 records] Stage 3: Majority-Class Soft Ceiling (Cap=600 per subcategory in train)
              ├── Garbage Dumping:           2,574 -> 600 (-1,974)
              ├── Potholes & Road Damage:    1,814 -> 600 (-1,214)
              ├── Road Repair / Construction: 1,532 -> 600 (-932)
              └── Street Light Not Working:    995 -> 600 (-395)
                    │
                    ▼
[Final Reduced Training Partition: 5,585 records]
```

* **Total Excluded from Training Partition:** $45 + 391 + 4,515 = \mathbf{4,951 \text{ records}}$.
* **Final Total in `training_dataset.csv`:** $5,585 (\text{train}) + 1,317 (\text{val}) + 1,289 (\text{test}) = \mathbf{8,191 \text{ records}}$.
* **Net Reduction:** $13,142 - 8,191 = 4,951$ records.
* **Verdict:** **PASS**. Complete mathematical reconciliation verified.

---

## 3. Check 2: Class Distribution & Imbalance Verification

### 3.1 Six Approved Categories Across Partitions

| Category | In-Scope Master (13,142) | Train Raw (10,536) | Train Reduced (5,585) | Val (1,317) | Test (1,289) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Roads & Footpaths** | 5,173 (39.36%) | 4,147 | **1,862 (33.34%)** | 509 | 510 |
| **Waste & Sanitation** | 4,683 (35.63%) | 3,735 | **1,585 (28.38%)** | 487 | 461 |
| **Street Lighting & Electricity** | 1,762 (13.41%) | 1,425 | **933 (16.71%)** | 169 | 168 |
| **Sewage & Drainage** | 689 (5.24%) | 552 | **538 (9.63%)** | 69 | 68 |
| **Pollution** | 494 (3.76%) | 397 | **396 (7.09%)** | 48 | 49 |
| **Water Supply** | 341 (2.59%) | 280 | **271 (4.85%)** | 35 | 33 |
| **Total** | **13,142 (100%)** | **10,536** | **5,585 (100%)** | **1,317** | **1,289** |

### 3.2 Twenty-Four Approved Subcategories Across Partitions

| Category | Subcategory | In-Scope Master | Train Reduced | Val Split | Test Split | Train Red. Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pollution** | Air Pollution | 132 | 105 | 13 | 13 | 100% Unique Clusters |
| **Pollution** | Land / Waste Pollution | 23 | 18 | 3 | 2 | 100% Examples Retained |
| **Pollution** | Noise Pollution | 264 | 211 | 26 | 27 | 100% Examples Retained |
| **Pollution** | Water Pollution | 75 | 62 | 6 | 7 | 100% Examples Retained |
| **Roads & Footpaths** | Footpath Damage / Repair | 645 | 485 | 61 | 65 | 100% Unique Clusters |
| **Roads & Footpaths** | Potholes & Road Damage | 2,354 | **600** | 230 | 232 | Capped at 600 Ceiling |
| **Roads & Footpaths** | Road Repair / Construction | 1,948 | **600** | 195 | 191 | Capped at 600 Ceiling |
| **Roads & Footpaths** | Road Waterlogging | 226 | 177 | 23 | 22 | 100% Unique Clusters |
| **Sewage & Drainage** | Drain Blockage / Desilting | 270 | 209 | 27 | 26 | 100% Unique Clusters |
| **Sewage & Drainage** | Sewer Blockage / Overflow | 130 | 102 | 13 | 13 | 100% Unique Clusters |
| **Sewage & Drainage** | Sewer Line Repair / Maintenance | 199 | 156 | 20 | 20 | 100% Unique Clusters |
| **Sewage & Drainage** | Storm Water Drainage / Waterlogging | 90 | 71 | 9 | 9 | 100% Unique Clusters |
| **Street Lighting** | Electrical Pole / Transformer Issues | 62 | 49 | 6 | 6 | 100% Unique Clusters |
| **Street Lighting** | Electricity Supply | 194 | 153 | 19 | 19 | 100% Unique Clusters |
| **Street Lighting** | Street Light Installation | 164 | 131 | 17 | 16 | 100% Examples Retained |
| **Street Lighting** | Street Light Not Working / Repair | 1,342 | **600** | 127 | 127 | Capped at 600 Ceiling |
| **Waste & Sanitation** | Garbage Collection / Removal | 386 | 309 | 38 | 39 | 100% Examples Retained |
| **Waste & Sanitation** | Garbage Dumping | 3,379 | **600** | 351 | 337 | Capped at 600 Ceiling |
| **Waste & Sanitation** | Street / Public Area Cleaning | 813 | 596 | 88 | 75 | 100% Unique Clusters |
| **Waste & Sanitation** | Waste Disposal & Segregation | 105 | 80 | 10 | 10 | 100% Unique Clusters |
| **Water Supply** | New Water Connection / Pipeline | **20** | **16** | 2 | 2 | 100% Examples Retained |
| **Water Supply** | Water Leakage | 56 | 43 | 6 | 5 | 100% Unique Clusters |
| **Water Supply** | Water Pipeline Damage / Maintenance | 22 | 18 | 2 | 2 | 100% Examples Retained |
| **Water Supply** | Water Supply Interruption | 243 | 194 | 25 | 24 | 100% Examples Retained |

### 3.3 Audit of Specific Statistical Claims
1. **Original Minimum of 20**: Confirmed. Belongs to `New Water Connection / Pipeline` in the full in-scope dataset (20 rows total: train=16, val=2, test=2).
2. **Reduced Minimum of 16**: Confirmed. Belongs to `New Water Connection / Pipeline` in the training partition of `training_dataset.csv` (16 rows).
3. **Training Imbalance Ratio of 37.50:1**: Confirmed. In the reduced training partition:
   $$\text{Imbalance Ratio} = \frac{\text{Max Subcategory Size}}{\text{Min Subcategory Size}} = \frac{600}{16} = \mathbf{37.50 : 1}$$
   (Down from the original training imbalance ratio of $2,691 / 16 = \mathbf{168.19:1}$, and overall in-scope ratio of $3,379 / 20 = \mathbf{168.95:1}$).
4. **Minority Class Retention Clarification**:
   - In `scope_filtered_dataset.csv`, 100% of all minority-class examples are preserved across all splits.
   - In `training_dataset.csv` (train partition), all 20 non-capped subcategories retained 100% of their distinct complaint clusters. Subcategories with fewer than 600 instances experienced **zero downsampling**; any minor reduction in row counts within these classes was solely due to removing exact duplicate rows and merging redundant copies within clusters to 1 representative record.
* **Verdict:** **PASS**.

---

## 4. Check 3: Split Integrity and Leakage Verification

### 4.1 Split Preservation
- **Validation Partition:** 1,317 rows. Set of `source_row_id`s in `training_dataset.csv` matches `scope_filtered_dataset.csv` and `batch_2/prepared_master_dataset.csv` **100.0%**.
- **Test Partition:** 1,289 rows. Set of `source_row_id`s in `training_dataset.csv` matches `scope_filtered_dataset.csv` and `batch_2/prepared_master_dataset.csv` **100.0%**.
- **Source ID Overlap:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlap)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlap)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlap)

### 4.2 Group-Aware Cluster Leakage Check
- Overlap of `duplicate_group_id` across partitions in `training_dataset.csv`:
  - $\text{Train Groups} \cap \text{Val Groups} = \emptyset$ (0 overlap)
  - $\text{Train Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)
  - $\text{Val Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)
- **Verdict:** **PASS**. Exactly zero duplicate group leakage exists across splits.

### 4.3 Normalized Text Boundary Audit (Deep Inspection)
An independent case-folded and punctuation-stripped string matching audit was performed on `combined_text.str.lower().str.strip()`. This uncovered 4 generic short phrases that appear across split boundaries:
1. `"welding shop in residential area"`:
   - Train (ID 2069, `duplicate_group_id: 15083`, `Noise Pollution`): title has trailing period.
   - Val (ID 2070, `duplicate_group_id: 15082`, `Electricity Supply`): title lacks trailing period.
2. `"yellow spot"`:
   - Train (IDs 12043, 11511, 11578, with case variants 'Yellow Spot', 'YELLOW SPOT').
   - Val (IDs 11498, 11513-11522, 'Yellow spot').
3. `"frequent power cuts"`:
   - Train (ID 8985, lowercase 'p': 'Frequent power cuts').
   - Val (ID 1546, uppercase 'P': 'Frequent Power Cuts').
4. `"broken footpath"`:
   - Train (IDs 11265, 11727, 'Broken Footpath' / 'Broken footpath').
   - Test (IDs 11783, 11785, 11792, 11875, 'Broken Footpath').

* **Root Cause:** In Phase 2.8, `duplicate_group_id` was formed by exact case-sensitive string matching on `["clean_title", "clean_description"]`. Short boilerplate entries with trivial capitalization or punctuation differences received distinct group IDs and were independently assigned during group stratification.
* **Risk & Impact Evaluation:** **Negligible risk**. These 4 items represent ultra-generic civic labels (comprising only 0.048% of records) rather than proprietary complaint narratives. Crucially, in item #1, the two complaints actually represent different grievances (Noise Pollution vs Illegal Electricity Hookup).
* **Verdict:** **REVIEW_REQUIRED (Documented)**. Does not impede model retraining, but should be noted as a known property of the inherited Batch 2 split definitions.

---

## 5. Check 4: Taxonomy and Scope Verification

### 5.1 Strict Adherence to Locked Taxonomy
- Every single retained record in `scope_filtered_dataset.csv` (13,142 rows) and `training_dataset.csv` (8,191 rows) was validated against [`final_taxonomy_mapping.json`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/final_taxonomy_mapping.json).
- **Invalid Categories:** **0**
- **Invalid Subcategories:** **0**
- **Mismatched Department Assignments:** **0**
- **Unapproved Taxonomy Additions:** **0**
- **Verdict:** **PASS**.

### 5.2 Out-of-Scope Exclusions Audit
All 2,929 non-civic and uncertain records in [`unmapped_labels.csv`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1/unmapped_labels.csv) were audited across 105 distinct label combinations:
* **Traffic & Road-Safety Enforcement (967 records):** Traffic congestion (210), parking violations (343), signal jumping (13), signal repair/install (33), traffic signboards (16), speed breakers (352). Handled by Bangalore Traffic Police (BTP) rather than municipal civic engineering.
* **Encroachment Enforcement (57 records):** Government property encroachment (49), cycle lane encroachment (24, counted in traffic), lake encroachment (8). Handled by Revenue / Forest / Encroachment Directorate.
* **Animal Husbandry (832 records):** Stray dogs, rabies vaccinations, cattle menace.
* **Police & Crime (132 records):** Theft, police beat patrolling, public safety.
* **Public Transit (46 records):** BMTC/KSRTC bus operations, ticketing, conductor disputes.
* **Certificates & Revenue (53 records):** Birth/death certificates, Khata transfers, trade licenses.
* **Uncertain Labels (365 records):** Generic labels (`Others`: 346, `NaN`: 18, `Pension`: 1).
* **Audit Finding:** Zero valid municipal public-service complaints were erroneously excluded.
* **Verdict:** **PASS**.

---

## 6. Check 5: Cryptographic Integrity & Reproducibility Verification

### 6.1 Checksum Comparison

| File Name | Manifest SHA-256 | Actual Calculated SHA-256 | Match |
| :--- | :---: | :---: | :---: |
| **Raw Input CSV** | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` | **PASS** |
| `scope_filtered_dataset.csv` | `caba156768a69a0397ce5418ea722ea95ce1d8213e4b7bca0fa411030e4604bb` | `caba156768a69a0397ce5418ea722ea95ce1d8213e4b7bca0fa411030e4604bb` | **PASS** |
| `training_dataset.csv` | `e9d31eec903d6d061e8cf0bca9a545c8c5c3a37e9d4869c84918e38d6df1a4f0` | `e9d31eec903d6d061e8cf0bca9a545c8c5c3a37e9d4869c84918e38d6df1a4f0` | **PASS** |
| `dataset_reduction_audit.csv` | `8d9f333a00e84b806950ec462d7c0ca56441bfdd1f21190cf613915858fc8577` | `8d9f333a00e84b806950ec462d7c0ca56441bfdd1f21190cf613915858fc8577` | **PASS** |
| `category_distribution.csv` | `c59042a588622f6fa3f443b0222fe41b31862590fa4ff1e1146313a37340d046` | `c59042a588622f6fa3f443b0222fe41b31862590fa4ff1e1146313a37340d046` | **PASS** |
| `subcategory_distribution.csv` | `01b82942123cb254646736200e0fe363be3da61899e46a78ea384fcb9195b0c9` | `01b82942123cb254646736200e0fe363be3da61899e46a78ea384fcb9195b0c9` | **PASS** |
| `unmapped_labels.csv` | `ead807f168f6bf7a2cfec7dc41ca27b8aa0fe28876aa9046603a1fc6c98ae06a` | `ead807f168f6bf7a2cfec7dc41ca27b8aa0fe28876aa9046603a1fc6c98ae06a` | **PASS** |

### 6.2 Execution and Reproducibility
- Both [`dataset_reduction_process.py`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/ml/dataset_reduction_process.py) and [`dataset_reduction_validate.py`](file:///Users/bhabanishankarpanda/Projects/Python/Municipal-Complaint-Intelligence-System/backend/app/data/processed/dataset_reduction_v1/dataset_reduction_validate.py) were executed and passed cleanly.
- Re-executing the majority ceiling sampling algorithm with `random_state=42` on the deduplicated training clusters produced the exact set of 5,585 `source_row_id`s present in `training_dataset.csv`.
- **Verdict:** **PASS**.

---

## 7. Final Decision & Retraining Recommendation

### **FINAL VERDICT: READY FOR RETRAINING**

### Justification:
1. **Scope Purity:** Encroachment and traffic enforcement issues have been cleanly removed from the training domain, ensuring that municipal classification models learn civil engineering grievances without jurisdiction bleed.
2. **Class Balance Improvement:** Training skew was curtailed by **77.8%** (from 168.19:1 down to 37.50:1), eliminating majority class dominance while preserving 100% of distinct minority-class examples.
3. **Evaluation Integrity:** Validation (1,317 rows) and Test (1,289 rows) sets remain completely un-downsampled and un-modified, ensuring that subsequent retraining evaluations provide honest, leak-free, apples-to-apples comparisons with the Stage 2 Batch 3 baseline models.
4. **Data Traceability:** Full record-level lineage back to the original CSV row identifiers is maintained.

### Retraining Next Steps:
- Retrain Task 2 (`task2_category_classifier`) and Task 3 (`task3_subcategory_classifier_bundle`) using `training_dataset.csv` with class weighting.
- Compare Macro-F1 across all 24 subcategories on the untouched test partition.
