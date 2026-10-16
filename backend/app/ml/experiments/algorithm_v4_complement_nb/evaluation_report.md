# Evaluation Report: algorithm_v4_complement_nb

## 1. Test Performance vs. v2 and Batch 3 (1,289 Test Records)

| Task | Metric | Batch 3 | Reduction v2 | **Candidate** | Delta vs v2 | 95% CI vs v2 | Delta vs B3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Task 2 (Category) | Accuracy | 0.8914 | 0.8922 | **0.8867** | -0.0054 | [-0.0178, 0.007] | -0.0047 |
| Task 2 (Category) | Macro-F1 | 0.8197 | 0.8272 | **0.7984** | -0.0288 | [-0.0566, -0.0032] | -0.0213 |
| Task 3 (Oracle) | Accuracy | 0.7890 | 0.7727 | **0.7324** | -0.0403 | N/A | -0.0566 |
| Task 3 (Oracle) | Macro-F1 | 0.6109 | 0.6210 | **0.5671** | -0.0538 | N/A | -0.0438 |
| Task 3 (E2E) | Accuracy | 0.7137 | 0.6974 | **0.6680** | -0.0295 | [-0.0504, -0.0093] | -0.0458 |
| Task 3 (E2E) | Macro-F1 | 0.4898 | 0.4900 | **0.4533** | -0.0366 | [-0.0839, 0.004] | -0.0365 |
| Department | Accuracy | 0.8534 | 0.8534 | **0.8285** | -0.0248 | [-0.0396, -0.0109] | -0.0248 |
| Department | Macro-F1 | 0.7624 | 0.7695 | **0.7302** | -0.0394 | [-0.0708, -0.0104] | -0.0322 |

