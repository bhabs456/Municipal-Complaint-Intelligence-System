# Training Report: algorithm_v3_logistic_regression

- **Algorithm**: TF-IDF + Logistic Regression (balanced)
- **Dataset**: Exact v2 soft-cap training dataset (7,224 train / 1,317 val / 1,289 test)
- **Tuning Protocol**: Grid search on validation split (1,317 records); test set untouched
- **Selected Task 2 Hyperparameters**: `{'class_weight': 'balanced', 'C': 10.0}`
- **Selected Task 3 Hyperparameters**: `{'class_weight': 'balanced', 'C': 10.0}`

### Validation Grid Search Results

#### Task 2 (Category)

| class_weight | C | val_accuracy | val_macro_f1 |
| --- | --- | --- | --- |
| balanced | 0.1 | 0.8603 | 0.774 |
| balanced | 0.5 | 0.8778 | 0.8024 |
| balanced | 1.0 | 0.8785 | 0.8036 |
| balanced | 2.0 | 0.8815 | 0.8039 |
| balanced | 5.0 | 0.8846 | 0.8128 |
| balanced | 10.0 | 0.8876 | 0.8164 |
| None | 0.1 | 0.8018 | 0.458 |
| None | 0.5 | 0.8679 | 0.7494 |
| None | 1.0 | 0.8838 | 0.7918 |
| None | 2.0 | 0.8869 | 0.8015 |
| None | 5.0 | 0.8869 | 0.8057 |
| None | 10.0 | 0.8899 | 0.8137 |

#### Task 3 (Subcategory Oracle Branches)

| class_weight | C | val_oracle_accuracy | val_oracle_macro_f1 |
| --- | --- | --- | --- |
| balanced | 0.5 | 0.7472 | 0.5974 |
| balanced | 1.0 | 0.757 | 0.5924 |
| balanced | 2.0 | 0.7677 | 0.5985 |
| balanced | 5.0 | 0.776 | 0.6053 |
| balanced | 10.0 | 0.7752 | 0.6057 |
| None | 0.5 | 0.7578 | 0.4039 |
| None | 1.0 | 0.7684 | 0.4812 |
| None | 2.0 | 0.7775 | 0.5158 |
| None | 5.0 | 0.7813 | 0.5552 |
| None | 10.0 | 0.779 | 0.5562 |

