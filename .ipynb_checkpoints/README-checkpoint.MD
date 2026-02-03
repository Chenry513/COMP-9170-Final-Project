# 30-Day Hospital Readmission Prediction  
Ensemble Models, Deep Learning, and Careful Clinical Evaluation  
(Diabetes 130-US Hospitals Dataset)

This project focuses on predicting **30-day hospital readmission** using the Diabetes 130-US Hospitals dataset (101,766 encounters).

Instead of just training one model and reporting accuracy, the goal here was to:

- Prevent patient-level leakage
- Handle class imbalance properly
- Compare classical and deep models fairly
- Build an ensemble (stacking)
- Evaluate performance with statistical confidence

> Given structured clinical and demographic features at discharge, predict whether a patient will be readmitted within 30 days.

All models use **patient-level GroupKFold cross-validation**, meaning encounters from the same patient never appear in both train and validation splits.

That design choice is one of the most important parts of this project.

---

## 1. Problem Setup

### 1.1 Why This Problem?

30-day readmission is a major healthcare quality metric. Hospitals are financially penalized for high readmission rates, and it often reflects discharge planning quality and patient risk.

But predicting readmission is hard:

- Only ~11% of cases are positive (imbalanced problem)
- Many real causes of readmission aren't fully captured in structured data
- Signals are subtle and noisy

This makes it a realistic and challenging applied ML problem.

---

## 2. Dataset and Target Engineering

We use the **Diabetes 130-US Hospitals dataset**, which includes:

- 101,766 hospital encounters  
- 50+ structured clinical and demographic features  
- A patient identifier (`patient_nbr`)  
- Original `readmitted` label: `NO`, `>30`, `<30`  

We convert it into a binary classification task:

- `readmit_30d = 1` if `readmitted == "<30"`  
- `readmit_30d = 0` otherwise  

Final positive rate: **~11.2%**

This immediately creates an **imbalanced classification problem**, which is why:

- Accuracy alone is misleading
- ROC-AUC and PR-AUC are more informative
- Class weighting is necessary

---

## 3. Data Cleaning and Feature Decisions

All cleaning logic lives in:

`utils/diabetes_utils.py`

I centralized preprocessing to:

- Keep models consistent
- Avoid accidental leakage
- Make experiments reproducible
- Easily adjust features without rewriting notebooks

### Cleaning Choices (and Why)

**Missing values**
- Replace `"?"` with NaN
- Drop columns with extreme missingness:
  - `weight`
  - `medical_specialty`
  - `payer_code`

These columns were too sparse and added instability.

**Race**
- Keep `race`
- Encode missing as `"Unknown"`

Dropping it entirely would remove potentially useful demographic signal.

**Diagnosis codes**
Raw ICD codes are too granular, so I grouped them into broader clinical categories like:

- Circulatory
- Respiratory
- Digestive
- Diabetes-related
- Injury
- Neoplasms
- Other

This reduces sparsity and improves generalization.

---

## 4. Cross-Validation Strategy (Very Important)

All models use:

**GroupKFold with `patient_nbr`**

Why?

If the same patient appears in both training and validation:

- The model sees almost identical feature patterns
- Validation metrics inflate artificially
- Results become clinically meaningless

Using patient-level splits ensures honest evaluation.

---

## 5. Model Implementations

Each model:

- Uses 5-fold GroupKFold
- Applies fold-specific scaling
- Handles imbalance via class weights
- Saves out-of-fold (OOF) predictions
- Tunes decision thresholds based on F1

---

### 5.1 LACE-style Logistic Regression

File: `01_lace.ipynb`

This acts as a clinical baseline.

Why start simple?

- Logistic regression is interpretable
- Clinicians understand linear risk models
- It gives us a realistic performance floor

This model is not expected to win — it sets context.

---

### 5.2 XGBoost

File: `02_xgboost.ipynb`

XGBoost is extremely strong on tabular data.

Reasons for including it:

- Captures nonlinear feature interactions
- Handles mixed data types well
- Strong performance in structured medical datasets

We use `scale_pos_weight` to account for imbalance.

This ended up being one of the strongest single models.

---

### 5.3 LSTM

File: `03_lstm.ipynb`

Even though the dataset isn't truly sequential, I experimented with an LSTM to see whether:

- Nonlinear representation learning
- Implicit feature interaction modeling

could improve performance.

Architecture:

- LSTM(64)
- Dropout
- Dense layers
- Sigmoid output

Scaling is done inside each fold to prevent leakage.

---

### 5.4 TabNet

File: `04_tabnet.ipynb`

TabNet is designed specifically for tabular data.

It uses attention mechanisms to learn which features matter most at each decision step.

This tests whether deep tabular architectures outperform boosting here.

---

### 5.5 TA-RNN (Time-Aware RNN)

File: `05_tarnn.ipynb`

This model separates:

- Longitudinal numeric features
- Demographic one-hot features
- Time channel input

Architecture:

Longitudinal → LSTM → Dense  
Concatenate demographics → Output

This is a more structured deep approach compared to the plain LSTM.

---

## 6. Stacking Ensemble

File: `06_stacking.ipynb`

Instead of choosing a single best model, I built a stacking ensemble.

Why stacking?

Different models make different mistakes:

- Trees capture nonlinear splits well
- Deep models smooth patterns differently
- Linear models generalize conservatively

Stacking lets a meta-model learn how to combine them.

Process:

1. Collect OOF predictions from:
   - LSTM
   - XGBoost
   - TabNet
2. Stack them into a new feature matrix
3. Train logistic regression as meta-learner
4. Tune decision threshold based on F1

### Stacking Results

AUC: 0.682  
AUPRC: 0.226  
F1 (tuned): 0.286  

The improvement is modest — not dramatic.

This suggests the base models already capture most of the available signal.

---

## 7. Evaluation and Statistical Confidence

File: `07_evaluation.ipynb`

Instead of reporting one metric, I included:

- ROC-AUC
- PR-AUC
- F1
- Brier score
- Calibration curve
- 95% bootstrap confidence intervals

Final stacking results:

AUC: 0.682  
95% CI: 0.677–0.687  

AUPRC: 0.226  
95% CI: 0.220–0.233  

Brier score: 0.2241  

What this means in plain terms:

- The model is meaningfully better than random
- Performance is stable (tight confidence intervals)
- Gains from stacking are real but small
- This is realistic performance for structured readmission prediction

---

## 8. Out-of-Fold Predictions

All OOF predictions are saved in:

`oof_predictions/`

Why save them?

- Enables leakage-free stacking
- Allows statistical analysis later
- Supports calibration plots
- Makes evaluation reproducible without retraining

---

## 9. Key Takeaways

- Proper validation matters more than model complexity.
- Tree-based models are extremely strong for structured healthcare data.
- Deep learning does not automatically outperform boosting.
- Stacking provides small but consistent gains.
- Clinical prediction tasks often have ceiling effects.

---

## 10. Limitations

- No external validation dataset
- Single institutional dataset
- Limited hyperparameter search
- No survival/time-to-event modeling
- No deployment simulation

---

## 11. Project Structure

```
COMP-9170-Final-Project/
│
├── data/
│   └── diabetic_data.csv
├── utils/
│   └── diabetes_utils.py
├── oof_predictions/
├── figures/
│
├── 01_lace.ipynb
├── 02_xgboost.ipynb
├── 03_lstm.ipynb
├── 04_tabnet.ipynb
├── 05_tarnn.ipynb
├── 06_stacking.ipynb
├── 07_evaluation.ipynb
└── README.md
```

---

## Final Thoughts

This project wasn’t about chasing the highest number possible.

It was about:

- Avoiding leakage
- Comparing models honestly
- Handling imbalance properly
- Quantifying uncertainty
- Building a clean, reproducible ML pipeline

The final performance (AUC ~0.68) is realistic for this dataset, and the methodology is the main contribution.
 