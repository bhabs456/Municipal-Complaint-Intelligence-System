# MCIS Stage 2 — Artifact Integrity Report: Subcategory Improvement v1

**Audit Timestamp**: 2026-10-16T18:19:01Z  
**Scope**: Cryptographic verification of all inputs and generated artifacts  

---

## 1. Raw Data Invariant Check
- **File**: `backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv`
- **Expected SHA-256**: `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac`
- **Computed SHA-256**: `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac`
- **Verification**: **PASS — BITWISE IMMUTABLE**.

---

## 2. Taxonomy Schema Invariant Check
- **Categories**: 6 verified.
- **Subcategories**: 24 verified.
- **Departments**: 9 verified.
- **Deterministic Department Mapping**: 100% verified.

---

## 3. Cryptographic Artifact Registry

| Component | File Path | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| Raw CSV | `backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv` | 8,377,752 B | `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac` |
| Dataset Reduction v2 | `backend/app/ml/experiments/dataset_reduction_v2/training_dataset_v2.csv` | 16,433,264 B | `f8910f82f9d33010f0c17d3b8300c28b6dd34045616d50af5812aea469e8bc3e` |
| Saved Bundle Model | `backend/app/ml/experiments/subcategory_improvement_v1/saved_models/best_subcategory_model_bundle.joblib` | 19,917,307 B | `4ce1aa59d2b5518123c1df484234d173aed4993a41b494f86e8edf6d8ebe5042` |
