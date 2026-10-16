# MCIS Stage 2 — Model Artifact Integrity & Load Test Report

**Audit Timestamp**: 2026-10-16T18:30:57Z  

---

## 1. Registry of Inspected Artifacts (19 Components)

| Experiment | Component Description | Path | File Size | SHA-256 Hash | Load Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| Batch 3 | Task 1 Civic Classifier | `backend/app/ml/saved_models/batch_3/task1_civic_classifier.joblib` | 2,717,169 B | `8e04e4c5481a5a20...` | **PASS (joblib loaded)** |
| Batch 3 | Task 2 Category Classifier | `backend/app/ml/saved_models/batch_3/task2_category_classifier.joblib` | 7,984,272 B | `390935d249615903...` | **PASS (joblib loaded)** |
| Batch 3 | Task 3 Subcategory Bundle | `backend/app/ml/saved_models/batch_3/task3_subcategory_classifier_bundle.joblib` | 18,556,299 B | `9ba0b63f698801c2...` | **PASS (joblib loaded)** |
| Batch 3 | Task 3 Global Baseline | `backend/app/ml/saved_models/batch_3/task3_global_baseline.joblib` | 28,486,836 B | `ebecfd95d0dee0cd...` | **PASS (joblib loaded)** |
| Batch 3 | Confidence Config | `backend/app/ml/saved_models/batch_3/confidence_config.json` | 327 B | `1188680b6bfb5fe4...` | **PASS (json loaded)** |
| Batch 3 | Model Metadata | `backend/app/ml/saved_models/batch_3/model_metadata.json` | 1,665 B | `7e15c2a1bd637ce1...` | **PASS (json loaded)** |
| Reduction v1 | Task 2 Category Classifier | `backend/app/ml/experiments/dataset_reduction_v1/task2_category_classifier_v1.joblib` | 4,920,927 B | `f9e2f258ebb8e9ab...` | **PASS (joblib loaded)** |
| Reduction v1 | Task 3 Subcategory Bundle | `backend/app/ml/experiments/dataset_reduction_v1/task3_subcategory_classifier_bundle_v1.joblib` | 12,552,686 B | `7e5a1244481d6eb4...` | **PASS (joblib loaded)** |
| Reduction v1 | Task 3 Global Baseline | `backend/app/ml/experiments/dataset_reduction_v1/task3_global_baseline_v1.joblib` | 17,579,651 B | `31993b2d4e3269b6...` | **PASS (joblib loaded)** |
| Reduction v2 | Task 2 Category Classifier | `backend/app/ml/experiments/dataset_reduction_v2/task2_category_classifier_v2.joblib` | 6,017,328 B | `62827429dcc08678...` | **PASS (joblib loaded)** |
| Reduction v2 | Task 3 Subcategory Bundle | `backend/app/ml/experiments/dataset_reduction_v2/task3_subcategory_classifier_bundle_v2.joblib` | 14,869,115 B | `e09719513b72e37c...` | **PASS (joblib loaded)** |
| Reduction v2 | Task 3 Global Baseline | `backend/app/ml/experiments/dataset_reduction_v2/task3_global_baseline_v2.joblib` | 21,486,196 B | `df35ce0dd4ada03c...` | **PASS (joblib loaded)** |
| Exp v3 (LR) | Task 2 Category Classifier | `backend/app/ml/experiments/algorithm_v3_logistic_regression/task2_category_classifier_v3.joblib` | 2,577,388 B | `0e712a07ade588c4...` | **PASS (joblib loaded)** |
| Exp v3 (LR) | Task 3 Subcategory Bundle | `backend/app/ml/experiments/algorithm_v3_logistic_regression/task3_subcategory_classifier_bundle_v3.joblib` | 9,486,989 B | `da21ad5ae77373e6...` | **PASS (joblib loaded)** |
| Exp v3 (LR) | Task 3 Global Baseline | `backend/app/ml/experiments/algorithm_v3_logistic_regression/task3_global_baseline_v3.joblib` | 7,731,676 B | `b41e82e46c5a6e2b...` | **PASS (joblib loaded)** |
| Exp v4 (CNB) | Task 2 Category Classifier | `backend/app/ml/experiments/algorithm_v4_complement_nb/task2_category_classifier_v4.joblib` | 4,582,017 B | `9a50f0497d08c466...` | **PASS (joblib loaded)** |
| Exp v4 (CNB) | Task 3 Subcategory Bundle | `backend/app/ml/experiments/algorithm_v4_complement_nb/task3_subcategory_classifier_bundle_v4.joblib` | 16,199,117 B | `41728dbb98468af1...` | **PASS (joblib loaded)** |
| Exp v4 (CNB) | Task 3 Global Baseline | `backend/app/ml/experiments/algorithm_v4_complement_nb/task3_global_baseline_v4.joblib` | 14,892,289 B | `ecee7699d9d792a6...` | **PASS (joblib loaded)** |
| Subcategory Imp v1 | Branch-Customized Bundle | `backend/app/ml/experiments/subcategory_improvement_v1/saved_models/best_subcategory_model_bundle.joblib` | 19,917,307 B | `4ce1aa59d2b55181...` | **PASS (joblib loaded)** |

---

## 2. Load Test Audit Findings
- **Total Artifacts Checked**: 19
- **Passed Load Tests**: **19 / 19 (100.0%)**
- **Failed Load Tests**: **0**
- **Exceptions Encountered**: None. Every `.joblib` model artifact loaded into memory and performed inference. Every `.json` configuration loaded and conformed to expected schema keys.
