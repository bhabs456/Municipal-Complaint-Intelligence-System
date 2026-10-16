# Evaluation Report: algorithm_v3_logistic_regression

## 1. Test Performance vs. v2 and Batch 3 (1,289 Test Records)

| Task | Metric | Batch 3 | Reduction v2 | **Candidate** | Delta vs v2 | 95% CI vs v2 | Delta vs B3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Task 2 (Category) | Accuracy | 0.8914 | 0.8922 | **0.8875** | -0.0047 | [-0.014, 0.0047] | -0.0039 |
| Task 2 (Category) | Macro-F1 | 0.8197 | 0.8272 | **0.8333** | +0.0061 | [-0.0132, 0.0243] | +0.0136 |
| Task 3 (Oracle) | Accuracy | 0.7890 | 0.7727 | **0.7611** | -0.0116 | N/A | -0.0279 |
| Task 3 (Oracle) | Macro-F1 | 0.6109 | 0.6210 | **0.5916** | -0.0294 | N/A | -0.0193 |
| Task 3 (E2E) | Accuracy | 0.7137 | 0.6974 | **0.6866** | -0.0109 | [-0.0248, 0.0031] | -0.0272 |
| Task 3 (E2E) | Macro-F1 | 0.4898 | 0.4900 | **0.4991** | +0.0092 | [-0.0306, 0.0409] | +0.0093 |
| Department | Accuracy | 0.8534 | 0.8534 | **0.8487** | -0.0047 | [-0.0163, 0.0062] | -0.0047 |
| Department | Macro-F1 | 0.7624 | 0.7695 | **0.7786** | +0.0091 | [-0.0107, 0.0288] | +0.0162 |

