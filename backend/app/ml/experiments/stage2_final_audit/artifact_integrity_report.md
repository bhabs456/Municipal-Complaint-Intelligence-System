# MCIS Stage 2 — Artifact & Pipeline Integrity Report

**Audit Timestamp**: 2026-10-16T23:35:00+05:30  
**Scope**: Read-Only Cryptographic and Schema Integrity Audit  

---

## 1. Raw Dataset Invariant Verification
- **Target File**: `backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv`
- **Expected SHA-256**: `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac`
- **Computed SHA-256**: `d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac`
- **Integrity Status**: **PASS — BITWISE IMMUTABLE**. The original raw dataset has never been altered, rewritten, or renamed.

---

## 2. Taxonomy Schema Invariant Verification
- **Locked Hierarchy**:
  - **Categories**: Exactly 6 (`Roads & Footpaths`, `Waste & Sanitation`, `Street Lighting & Electricity`, `Sewage & Drainage`, `Pollution`, `Water Supply`)
  - **Subcategories**: Exactly 24 standardized municipal subcategories
  - **Departments**: Exactly 9 municipal operational departments
- **Department Mapping Invariant**:
  - All 24 subcategories map 1:1 and deterministically to their respective municipal departments.
  - Verification: 100% of test predictions across all 5 models map deterministically to the approved department without fallback failures.

---

## 3. Test Set Splitting & Membership Audit
- **Common Test Size**: Exactly **1,289 records**
- **Row Alignment**: `source_row_id` ordering is 100% identical across:
  - `dataset_reduction_v1/error_analysis/row_by_row_test_predictions.csv`
  - `dataset_reduction_v2/row_by_row_test_predictions.csv`
  - `algorithm_v3_logistic_regression/row_by_row_test_predictions.csv`
  - `algorithm_v4_complement_nb/row_by_row_test_predictions.csv`
- **Ground Truth Consistency**: 0 mismatched category, subcategory, or department target labels.

---

## 4. Model Artifact Cryptographic Registry

| Experiment | Artifact Description | File Path | File Size | SHA-256 Checksum | Verification Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| Batch 3 | Task 2 LinearSVC | `backend/app/ml/saved_models/batch_3/task2_category_classifier.joblib` | 7,984,272 B | `390935d24961590385208a18489d8ce5bb3971b2361f3cfccc03152de2218888` | Verified & Intact |
| Batch 3 | Task 3 Bundle | `backend/app/ml/saved_models/batch_3/task3_subcategory_classifier_bundle.joblib` | 18,556,299 B | `9ba0b63f698801c2adf467add3ad0fec7d9a3c56daa3e23c6f9ea536c5a26f5a` | Verified & Intact |
| Batch 3 | Task 3 Global | `backend/app/ml/saved_models/batch_3/task3_global_baseline.joblib` | 28,486,836 B | `ebecfd95d0dee0cde6915ba4e76b708174d98937ab21d6fc79f131cb49e10828` | Verified & Intact |
| Batch 3 | Task 1 Civic | `backend/app/ml/saved_models/batch_3/task1_civic_classifier.joblib` | 2,717,169 B | `8e04e4c5481a5a20dfab00d5efd8125aa92bad144233dd7cac663bc5a0a5300c` | Verified & Intact |
| Reduction v1 | Task 2 LinearSVC | `backend/app/ml/experiments/dataset_reduction_v1/task2_category_classifier_v1.joblib` | 4,920,927 B | `f9e2f258ebb8e9abcc4848ff4647b2314594ae910cb46e3fcff04025cb41b22a` | Verified & Intact |
| Reduction v1 | Task 3 Bundle | `backend/app/ml/experiments/dataset_reduction_v1/task3_subcategory_classifier_bundle_v1.joblib` | 12,552,686 B | `7e5a1244481d6eb4a3557561d88e4d8b6dabe8ff7090ec8038d3761238b8ca32` | Verified & Intact |
| Reduction v2 | Task 2 LinearSVC | `backend/app/ml/experiments/dataset_reduction_v2/task2_category_classifier_v2.joblib` | 6,017,328 B | `62827429dcc086780a4601e2db163b9b8096c39336e681a718501a45dcdd86f7` | Verified & Intact |
| Reduction v2 | Task 3 Bundle | `backend/app/ml/experiments/dataset_reduction_v2/task3_subcategory_classifier_bundle_v2.joblib` | 14,869,115 B | `e09719513b72e37c2b8cb7e02aa87b1138f87def03813c2bd808229632860647` | Verified & Intact |
| Exp v3 | Task 2 LogisticReg | `backend/app/ml/experiments/algorithm_v3_logistic_regression/task2_category_classifier_v3.joblib` | 2,577,388 B | `0e712a07ade588c4666915828e5263df878ee4f6afb5f2d3a929de8df688f592` | Verified & Intact |
| Exp v3 | Task 3 Bundle | `backend/app/ml/experiments/algorithm_v3_logistic_regression/task3_subcategory_classifier_bundle_v3.joblib` | 9,486,989 B | `da21ad5ae77373e65660380929e07c476e1ffed926429e8529226652f4ad8442` | Verified & Intact |
| Exp v4 | Task 2 CompNB | `backend/app/ml/experiments/algorithm_v4_complement_nb/task2_category_classifier_v4.joblib` | 4,582,017 B | `9a50f0497d08c466ab8be4be5489be2339b1a365f227aaf5d7349f6cc28a6d5e` | Verified & Intact |
| Exp v4 | Task 3 Bundle | `backend/app/ml/experiments/algorithm_v4_complement_nb/task3_subcategory_classifier_bundle_v4.joblib` | 16,199,117 B | `41728dbb98468af19cce696c96c2fcf468787c778ca43ea91a07421121ea934f` | Verified & Intact |

---

## 5. Inference Invariant Audit
All 5 model pipelines were instantiated and verified with raw string input from the test set.
- Fresh inference against saved predictions: **100.0% Exact Match** (1,289 / 1,289 rows identical across all 5 models).
- Deterministic department lookup: **100.0% Exact Match** (1,289 / 1,289 rows identical across all 5 models).
- Conclusion: The reported metrics are 100% reproducible directly from the saved model artifacts.
