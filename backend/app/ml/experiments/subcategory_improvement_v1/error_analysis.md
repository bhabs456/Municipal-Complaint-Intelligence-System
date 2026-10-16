# MCIS Stage 2 — Comprehensive Error Analysis: Subcategory Classification
**Test Set**: 1,289 untouched common records  
**Audited Models**: Batch 3 Baseline, Dataset Reduction v2, Experiment v3 (Logistic Regression)  

---

## 1. Overall Error Decomposition & Theoretical Cascade Ceiling

| Model | Subcategory Errors | Overall Accuracy | Category Cascade Errors (Inter-Category) | Branch Confusion Errors (Intra-Category) |
| :--- | :---: | :---: | :---: | :---: |
| **Batch 3 Baseline** | **369** / 1,289 | **71.37%** | 140 (37.9% of errors) | 229 (62.1% of errors) |
| **Reduction v2** | **390** / 1,289 | **69.74%** | 139 (35.6% of errors) | 251 (64.4% of errors) |
| **Exp v3 (LogReg)** | **404** / 1,289 | **68.66%** | 145 (35.9% of errors) | 259 (64.1% of errors) |

### The Cascade Ceiling Formula
In the hierarchical architecture:
$$\text{Accuracy}_{\text{E2E}} = \text{Accuracy}_{\text{Task 2}} \times \text{Accuracy}_{\text{Oracle Branch | Correct Category}}$$

- Task 2 Category Accuracy is currently **89.2%**.
- This means **10.8% of complaints (139 records) are sent to the wrong branch**, guaranteeing a 0% subcategory accuracy on those complaints.
- Consequently, to achieve the target **80.0% End-to-End Accuracy**, oracle branch accuracy would need to reach:
  $$\text{Oracle Target} = \frac0.80.892 = \mathbf{89.69\%}$$
- However, actual oracle branch accuracy is currently **77.3%–78.9%**.
- The primary cause is **within-branch semantic entanglement** between majority subcategories.

---

## 2. Most Frequently Confused Subcategory Pairs

### Top Confusions in Dataset Reduction v2
- **True: `Potholes & Road Damage`** $\to$ **Pred: `Road Repair / Construction`**: **69 complaints** (17.7% of all errors)
- **True: `Road Repair / Construction`** $\to$ **Pred: `Potholes & Road Damage`**: **44 complaints** (11.3% of all errors)
- **True: `Garbage Dumping`** $\to$ **Pred: `Garbage Collection / Removal`**: **28 complaints** (7.2% of all errors)
- **True: `Street / Public Area Cleaning`** $\to$ **Pred: `Garbage Dumping`**: **17 complaints** (4.4% of all errors)
- **True: `Potholes & Road Damage`** $\to$ **Pred: `Garbage Dumping`**: **13 complaints** (3.3% of all errors)
- **True: `Garbage Dumping`** $\to$ **Pred: `Street / Public Area Cleaning`**: **11 complaints** (2.8% of all errors)
- **True: `Garbage Collection / Removal`** $\to$ **Pred: `Garbage Dumping`**: **9 complaints** (2.3% of all errors)
- **True: `Road Waterlogging`** $\to$ **Pred: `Road Repair / Construction`**: **8 complaints** (2.1% of all errors)
- **True: `Street Light Not Working / Repair`** $\to$ **Pred: `Street Light Installation`**: **6 complaints** (1.5% of all errors)
- **True: `Street Light Installation`** $\to$ **Pred: `Street Light Not Working / Repair`**: **6 complaints** (1.5% of all errors)

### Key Diagnostic Findings:
1. **The Pothole vs. Road Repair Knot**:
   - `Potholes & Road Damage` $\leftrightarrow$ `Road Repair / Construction` accounts for **113 errors (29.0% of all errors)** in v2 and **105 errors (28.5%)** in Batch 3.
   - Citizens frequently combine vocabulary (*"huge potholes, please asphalt and repair road"*).
2. **Sanitation vs. Dumping Ambiguity**:
   - `Street / Public Area Cleaning` $\leftrightarrow$ `Garbage Dumping` accounts for **28 errors**.
3. **Street Lighting Confusion**:
   - `Street Light Not Working / Repair` $\leftrightarrow$ `Street Light Installation` accounts for **12 errors**.

---

## 3. Representative Misclassification Case Studies (PII Redacted)

### A. Road Repair vs. Potholes Confusion (Lexical Overlap)
- **Row 192**: True: `Potholes & Road Damage` | Pred: `Road Repair / Construction`
  *Text*: "There is a leakage in the center of the road and it's not good fo... There is a leakage in the center of the road and it's not good for a dr..."
- **Row 253**: True: `Road Repair / Construction` | Pred: `Potholes & Road Damage`
  *Text*: "Dear BBMP, Road in front of elements mall was digged and never r... Dear BBMP, Road in front of elements mall was digged and never relaid ba..."
- **Row 299**: True: `Potholes & Road Damage` | Pred: `Road Repair / Construction`
  *Text*: "Kaveri water pipeline work pending, Road condition is very bad, During school timing to much traffic &amp; accident zone,. Request you to lo..."

### B. Both Models Incorrect (Challenging / Ambiguous Inputs)
- **Row 60**: True: `Footpath Damage / Repair` | B3: `Road Repair / Construction` | v2: `Road Repair / Construction`
  *Text*: "Large Water Pipes dumped in front of S-HIG-C, Block 7 apartment. It has been nearly 2 months and large water pipes have been dumped in front..."
- **Row 192**: True: `Potholes & Road Damage` | B3: `Road Repair / Construction` | v2: `Road Repair / Construction`
  *Text*: "There is a leakage in the center of the road and it's not good fo... There is a leakage in the center of the road and it's not good for a dr..."
- **Row 208**: True: `Garbage Dumping` | B3: `Footpath Damage / Repair` | v2: `Footpath Damage / Repair`
  *Text*: "Chicken shop stand is on foot path and blocking the traffic movem... Chicken shop stand is on foot path and blocking the traffic movement. P..."

### C. Records Corrected by v2 Soft-Cap (Minority Sensitivity)
- **Row 297**: True: `Road Repair / Construction` | B3 Pred: `Potholes & Road Damage` $\to$ Fixed by v2
  *Text*: "Vittasandra 6th cross road, Anjanniya circle roads towards Aksha... Vittasandra 6th cross road, Anjanniya circle roads towards Akshay nagar,..."
- **Row 690**: True: `Street Light Not Working / Repair` | B3 Pred: `Garbage Dumping` $\to$ Fixed by v2
  *Text*: "Near my house there is one old tree which is about to fall. Reque... Near my house there is one old tree which is about to fall. Request you..."
- **Row 2270**: True: `Road Repair / Construction` | B3 Pred: `Potholes & Road Damage` $\to$ Fixed by v2
  *Text*: "Pathetic road condition. Please repair the road in Alfa Garden. Pathetic condition now during rains. fully muddy road with pot holes...."

### D. Records Regressed in v2 (Majority Pruning Trade-off)
- **Row 22**: True: `Garbage Dumping` | B3 Correct $\to$ v2 Pred: `Garbage Collection / Removal`
  *Text*: "No garbage collection in vittasandra Main road Sri Balaji layout. No garbage collection in vittasandra Main road Sri Balaji layout..."
- **Row 183**: True: `Garbage Dumping` | B3 Correct $\to$ v2 Pred: `Street / Public Area Cleaning`
  *Text*: "Uncleared debris on pedestrian footpath. For more than two months there is debris that has been dumped on the footpath causing difficulty fo..."
- **Row 247**: True: `Garbage Dumping` | B3 Correct $\to$ v2 Pred: `Potholes & Road Damage`
  *Text*: "App crashing while selecting the image. App crashing while selecting the image..."
