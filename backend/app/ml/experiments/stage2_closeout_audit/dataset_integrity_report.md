# MCIS Stage 2 — Dataset Integrity & Reconciliation Audit Report

**Audit Timestamp**: 2026-10-16T18:30:57Z  

---

## 1. Raw Dataset Verification
- **Path**: `backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv`
- **SHA-256 Checksum**: `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac`
- **File Size**: 8,377,752 bytes
- **Row Count**: 16,071 records
- **Column Count**: 17 columns
- **Encoding**: Latin-1 / ISO-8859-1 (handles byte `0x92` quotes safely)
- **Status**: **PASS — BITWISE IMMUTABLE**.

---

## 2. In-Scope Dataset Reconciliation (Reduction v1)
- **Scope Filtered Dataset**: `backend/app/data/processed/dataset_reduction_v1/scope_filtered_dataset.csv`
- **Total In-Scope Complaints**: **13,142 records**
- **Out-of-Scope (Deferred)**: **2,564 records** (encroachments, transport, police matters)
- **Review Queue (Ambiguous)**: **365 records** (blank/others)
- **Reconciliation**: $13,142 + 2,564 + 365 = 16,071$ (**100.0% row conservation**).

---

## 3. Training Partitioning & Class Balancing Audit

| Dataset Version | Partition | Record Count | Imbalance Ratio | Majority Class Strategy |
| :--- | :--- | :---: | :---: | :--- |
| **Reduction v1** | Train | 5,585 | 37.5 : 1 | Hard cap at 600 samples for top 4 classes |
| **Reduction v1** | Validation | 1,317 | — | 10% stratified holdout |
| **Reduction v1** | Test | 1,289 | — | 10% stratified holdout |
| **Reduction v1 Total** | Combined | **8,191** | — | Fixed benchmark partition |
| **Reduction v2** | Train | 7,224 | 46.5 : 1 | Soft-cap formula: $\min(N, 600 + 10\sqrt{N})$ |
| **Reduction v2** | Validation | 1,317 | — | Identical validation holdout |
| **Reduction v2** | Test | 1,289 | — | Identical test holdout |
| **Reduction v2 Total** | Combined | **9,830** | — | Fixed benchmark partition |

---

## 4. Grouped Split Leakage Verification
- **Methodology**: Evaluated `duplicate_group_id` intersection across train, validation, and test splits.
- **Train $\cap$ Validation Leakage**: **0 duplicate groups**
- **Train $\cap$ Test Leakage**: **0 duplicate groups**
- **Validation $\cap$ Test Leakage**: **0 duplicate groups**
- **Conclusion**: Strict zero-leakage guarantee satisfied.
