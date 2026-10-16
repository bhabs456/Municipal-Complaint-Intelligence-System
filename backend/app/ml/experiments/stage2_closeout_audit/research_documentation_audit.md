# MCIS Stage 2 — Research Documentation & Repository Audit

**Audit Timestamp**: 2026-10-16T18:30:57Z  

---

## 1. Standalone Research Paper Status
- **Audit Finding**: **NOT PRESENT / BLOCKED**.
- **Inspection Detail**: The repository was searched for `.tex`, `.pdf`, and standalone manuscript files. No formal academic manuscript currently exists in the repository.
- **Current Documentation Sources**:
  - `README.md` (project introduction)
  - `PROGRESS.md` (authoritative step-by-step engineering and research log)
  - `backend/app/data/reports/` (Phase 2.1–2.18 data audit and validation reports)
  - `backend/app/ml/experiments/` (experimental audit reports for v1, v2, v3, v4, final audit, and subcategory improvement)

---

## 2. Claim Verification Against Empirical Data

| Documented Claim in Repository | Verified Empirical Reality | Audit Verdict |
| :--- | :--- | :---: |
| **Raw Dataset Size**: 16,071 records | Verified bitwise: exactly 16,071 rows, 17 columns | **ACCURATE** |
| **Raw SHA-256 Checksum**: `d951dbb4...` | SHA-256 matches expected hash bit-for-bit | **ACCURATE** |
| **Taxonomy**: 6 categories, 24 subcategories, 9 departments | Exactly 6 / 24 / 9 enforced across code and seed data | **ACCURATE** |
| **Batch 3 E2E Subcategory Accuracy**: 71.37% | Reproduced exactly: 71.37% on 1,289 test records | **ACCURATE** |
| **Reduction v2 E2E Subcategory Accuracy**: 69.74% | Reproduced exactly: 69.74% on 1,289 test records | **ACCURATE** |
| **Reduction v2 Department Accuracy**: 85.34% | Reproduced exactly: 85.34% (1,100 / 1,289 records) | **ACCURATE** |
| **Experiment v3 Macro-F1**: 0.4991 | Reproduced exactly: 0.4991 on 1,289 test records | **ACCURATE** |
| **80%–85% Accuracy Target**: Claimed vs. Achieved | The report explicitly states the target was **NOT REACHED** (achieved 69.90%) | **ACCURATE & TRANSPARENT** |
| **Theoretical Cascade Ceiling**: ~89% | Mathematically proven: $88.91\%$ Category Accuracy caps E2E pipeline | **ACCURATE** |

---

## 3. Repository Hygiene & Safety Check
- **API Keys & Credentials**: Verified **0 hardcoded secrets** in tracked files.
- **Personal Data (PII)**: Examples displayed in audit reports have phone numbers, emails, and street numbers masked (`[PHONE]`, `[EMAIL]`, `#[NUM]`).
- **Dependencies**: Verified lockfiles and requirements.txt (`scikit-learn`, `pandas`, `fastapi`, `sqlalchemy`, `psycopg2-binary`, `shapely`).

---

## 4. Proposed Additions Prior to Research Paper Submission
When drafting the formal academic manuscript:
1. **Add Scholarly Citations & DOIs**:
   - Reference foundational literature on municipal complaint classification (e.g., BBMP Sahaaya portal studies, CPGRAMS studies).
   - Cite Scikit-Learn Platt scaling calibration (Niculescu-Mizil & Caruana, 2005).
   - Cite BallTree spatial indexing (Omohundro, 1989) and Haversine geospatial metrics.
2. **Explicit Discussion of Lexical Ambiguity**:
   - Include the error analysis breakdown showing that 29% of test errors stem from mutual confusion between `Potholes & Road Damage` and `Road Repair / Construction`.
3. **Distinguish Oracle vs. End-to-End Performance**:
   - Always present Oracle accuracy (77.4%–78.9%) alongside End-to-End accuracy (69.7%–71.4%) to illustrate the cascading error boundary.
