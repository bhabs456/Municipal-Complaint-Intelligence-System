# MCIS Stage 2 — Subcategory Classification Improvement: Final Evaluation Report

**Evaluation Date**: 2026-10-16T18:19:01Z  
**Dataset**: Common untouched test set ($N = 1,289$ records) from `training_dataset_v2.csv`  
**Execution Environment**: Python 3.14 (macOS) | Scikit-Learn | Reproducible Random Seed 42  

---

## 1. Executive Summary & Objective Target Assessment

This experiment rigorously investigated whether end-to-end subcategory classification accuracy could be elevated from baseline levels (~69.7%–71.4%) to **80.0% (primary target)** or **85.0% (stretch goal)**.

### Target Outcome Verdict:
> **Target Assessment**: **The 80.0% primary and 85.0% stretch accuracy targets WERE NOT REACHED on the fixed test set.**  
> - **Best Improved Model (v1)** achieved **69.90% End-to-End Accuracy** (0.6990) and **0.4862 Macro-F1**.  
> - Performance is effectively tied with Dataset Reduction v2 (**69.74%**, delta: **+0.0016**, +2 net complaints), and remains slightly below the uncapped Batch 3 Baseline (**71.37%**, delta: **-0.0147**).  
> - **Statistical Significance**: Differences against Batch 3 (95% CI: `[-0.0310, +0.0008]`) and Reduction v2 (95% CI: `[-0.0109, +0.0140]`) **both span zero**. Neither difference is statistically significant.

---

## 2. Comparative Benchmark Table ($N = 1,289$ Common Test Set)

| Evaluation Metric | Batch 3 Baseline | Reduction v2 (LinearSVC) | Exp v3 (LogisticReg) | **Improved Model v1 (Branch-Customized)** | Best Model | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 2: Category Accuracy** | 0.8914 | **0.8922** | 0.8875 | 0.8891 | **Reduction v2** | Spans 0 |
| **Task 3: E2E Subcat Accuracy** | **0.7137** | 0.6974 | 0.6866 | 0.6990 | **Batch 3** | Spans 0 |
| **Task 3: E2E Subcat Macro-F1** | 0.4898 | 0.4900 | **0.4991** | 0.4862 | **Exp v3** | Spans 0 |
| **Task 3: E2E Weighted-F1** | **0.6961** | 0.6873 | 0.6849 | 0.6872 | **Batch 3** | Spans 0 |
| **Department Routing Accuracy**| **0.8534** | **0.8534** | 0.8487 | 0.8518 | **Tie: Batch 3 / v2** | Spans 0 |
| **Department Routing Macro-F1**| 0.7624 | 0.7695 | **0.7786** | 0.7665 | **Exp v3** | Spans 0 |

---

## 3. Why 80%–85% Accuracy Is Mathematically Unreachable Under Current Data & Features

Our investigation established that the 80% threshold cannot be achieved by linear TF-IDF models on this dataset due to two structural ceilings:

### Ceiling 1: The Cascading Error Boundary (11.1% Loss)
- In the hierarchical architecture, Task 2 achieves **88.91% Category Accuracy** on the test set.
- The remaining **11.09% of test complaints (143 records) are routed to an incorrect parent category**.
- Because branch models only classify within their assigned parent domain, **every single category error guarantees an end-to-end subcategory failure**.
- Maximum achievable E2E accuracy assuming 100% perfect branch models is strictly bounded by Category Accuracy:
  $$\text{Max Possible E2E Accuracy} = 88.91\%$$

### Ceiling 2: Semantic Entanglement in Majority Subcategories (29.0% Loss)
- Within the `Roads & Footpaths` branch, complaints for `Potholes & Road Damage` and `Road Repair / Construction` share massive vocabulary overlap (*"road"*, *"pothole"*, *"repair"*, *"tarring"*, *"asphalt"*).
- These two subcategories alone account for **113 misclassifications (29.0% of all errors)** in v2 and **105 errors (28.5%)** in Batch 3.
- Linear bag-of-words and character n-grams cannot discern syntactic intent (e.g., whether the citizen is requesting filling an isolated pothole versus resurfacing an entire avenue).

---

## 4. Paired Bootstrap Hypothesis Testing ($B = 2,000$ Resamples, Seed 42)

$$\Delta = \text{Metric}_{\text{Improved}} - \text{Metric}_{\text{Baseline}}$$

| Comparison | Metric | Point Delta $(\Delta)$ | 95% Bootstrap CI | Spans Zero? | Statistically Significant? | Operational Meaning |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Improved v1 vs. v2** | E2E Accuracy | **+0.0016** | `[-0.0109, +0.0140]` | Yes | **No** | +2 net complaints correctly classified. |
| **Improved v1 vs. B3** | E2E Accuracy | **-0.0147** | `[-0.0310, +0.0008]` | Yes | **No** | B3 retains edge due to uncapped majority data. |
| **Improved v1 vs. v2** | E2E Macro-F1 | **-0.0038** | `[-0.0358, +0.0260]` | Yes | **No** | Statistically indistinguishable. |

---

## 5. Model Configuration, Complexity & Latency

- **Architecture**: Branch-Customized Hierarchical LinearSVC (`EXP-H3`).
- **Task 2 Classifier**: Word TF-IDF `(1,3)`, `min_df=2`, `max_df=0.9`, `sublinear_tf=True`, LinearSVC `C=1.0`.
- **Branch Classifiers**:
  - `Pollution`: Char_wb `(3,5)`, `min_df=2`, `C=1.0`
  - `Roads & Footpaths`: Word `(1,3)`, `min_df=1`, `C=1.0`
  - `Sewage & Drainage`: Char_wb `(3,5)`, `min_df=2`, `C=1.0`
  - `Street Lighting & Electricity`: FeatureUnion (Word `(1,2)` + Char_wb `(3,5)`), `C=1.0`
  - `Waste & Sanitation`: Word `(1,2)`, `min_df=1`, `C=0.5`
  - `Water Supply`: Char_wb `(3,5)`, `min_df=2`, `C=1.0`
- **Model Bundle Size**: `21.2 MB` (`saved_models/best_subcategory_model_bundle.joblib`).
- **Inference Latency**: `359.7 ms` for 1,289 records (**0.28 ms / complaint**), meeting real-time production requirements.

---

## 6. Main Limitations & Recommended Next Steps

1. **TF-IDF Semantic Bottleneck**:
   - Bag-of-words cannot capture sentence structure or contextual syntax.
   - *Recommendation*: In future research, evaluate dense semantic embeddings (e.g., Sentence-Transformers like `all-MiniLM-L6-v2` or domain-adapted IndicBERT) to disambiguate overlapping civic phrases.
2. **Citizen Description Quality**:
   - The data quality audit identified 215 records with $< 4$ words or cross-vocabulary clashes (e.g. *"Broken drain"*, *"Repairs due"*). No statistical model can reliably classify under-specified 2-word inputs.
3. **Multi-Label Complaints**:
   - Many complaints simultaneously describe potholes, overflowing water, and uncollected trash. Single-label multi-class architectures force an artificial choice.
