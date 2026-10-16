# Training Report: algorithm_v4_complement_nb

- **Algorithm**: TF-IDF + Complement Naive Bayes
- **Dataset**: Exact v2 soft-cap training dataset (7,224 train / 1,317 val / 1,289 test)
- **Tuning Protocol**: Grid search on validation split (1,317 records); test set untouched
- **Selected Task 2 Hyperparameters**: `{'norm': False, 'alpha': 0.5}`
- **Selected Task 3 Hyperparameters**: `{'norm': True, 'alpha': 0.05}`

### Validation Grid Search Results

#### Task 2 (Category)

| norm | alpha | val_accuracy | val_macro_f1 |
| --- | --- | --- | --- |
| False | 0.01 | 0.8603 | 0.7605 |
| False | 0.05 | 0.8648 | 0.7631 |
| False | 0.1 | 0.8671 | 0.7649 |
| False | 0.5 | 0.874 | 0.7727 |
| False | 1.0 | 0.8717 | 0.7657 |
| False | 2.0 | 0.8671 | 0.7421 |
| True | 0.01 | 0.8383 | 0.732 |
| True | 0.05 | 0.8633 | 0.769 |
| True | 0.1 | 0.8694 | 0.7701 |
| True | 0.5 | 0.8664 | 0.7526 |
| True | 1.0 | 0.8633 | 0.7329 |
| True | 2.0 | 0.8497 | 0.6844 |

#### Task 3 (Subcategory Oracle Branches)

| norm | alpha | val_oracle_accuracy | val_oracle_macro_f1 |
| --- | --- | --- | --- |
| False | 0.01 | 0.7175 | 0.5497 |
| False | 0.05 | 0.7373 | 0.5625 |
| False | 0.1 | 0.7426 | 0.563 |
| False | 0.5 | 0.7441 | 0.4976 |
| False | 1.0 | 0.7509 | 0.464 |
| False | 2.0 | 0.7441 | 0.4292 |
| True | 0.01 | 0.4966 | 0.5271 |
| True | 0.05 | 0.732 | 0.5681 |
| True | 0.1 | 0.7426 | 0.5308 |
| True | 0.5 | 0.7456 | 0.4768 |
| True | 1.0 | 0.7479 | 0.4501 |
| True | 2.0 | 0.738 | 0.4155 |

