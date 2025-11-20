# Week 3 to Week 4 Handoff Document

**Project:** Corporación Favorita Grocery Sales Forecasting  
**From Phase:** Week 3 - Modeling & Analysis  
**To Phase:** Week 4 - Communication & Delivery  
**Handoff Date:** 2025-11-20  
**Project Owner:** Alberto Diaz Durana

---

## Executive Summary

**Week 3 Status:** COMPLETE (100%)

**Best Model Achieved:**
- Model Type: LSTM (Long Short-Term Memory Neural Network)
- Test RMSE: 6.2552
- Total Improvement: 13.28% vs Week 1 baseline (7.2127 → 6.2552)
- Status: Production-ready with complete deployment artifacts

**Key Discoveries:**
1. LSTM unexpectedly outperformed XGBoost by 4.5% on tabular data
2. Temporal consistency (DEC-016) more important than data volume
3. Feature reduction (DEC-014) prevented overfitting
4. Full 2013 training failed (DEC-015 rejected) due to seasonal mismatch

**Week 4 Readiness:**
- ✓ Model artifacts exported and validated
- ✓ Complete documentation and decision logs
- ✓ MLflow experiment tracking (6 runs)
- ✓ Reproducible preprocessing pipeline
- ✓ Clear deployment instructions

---

## 1. Project Context Recap

### Business Objective
Forecast daily unit_sales for Guayas stores (March 2014) to optimize inventory and reduce waste for Corporación Favorita grocery chain in Ecuador.

### Success Criteria Met
**Quantitative:**
- ✓ RMSE improvement over naive baseline: 13.28% achieved
- ✓ Forecast accuracy within business tolerance: RMSE = 6.26 (good for retail)

**Qualitative:**
- ✓ Interpretable models: LSTM architecture documented, feature importance analyzed
- ✓ Actionable insights: Temporal consistency principle, feature reduction strategy

**Technical:**
- ✓ Reproducible pipeline: Artifacts validated, predictions identical
- ✓ End-to-end execution: No errors, complete from data loading to deployment

### Scope Delivered
- Dataset: Guayas region, top-3 product families
- Training period: Q4 2013 + Q1 2014 (18,905 samples)
- Test period: March 2014 (4,686 samples)
- Features: 33 optimized features (DEC-014)
- Models tested: XGBoost (tree-based), LSTM (neural network)

---

## 2. Week 3 Accomplishments

### Performance Evolution

| Milestone | Model | RMSE | Improvement | Key Innovation |
|-----------|-------|------|-------------|----------------|
| Week 1 End | XGBoost baseline | 7.2127 | 0% (baseline) | Q1-only, 45 features |
| Week 2 End | Feature engineering | N/A | Feature set created | 45 engineered features |
| Day 1 | XGBoost baseline | 7.2127 | 0% | Reproduced baseline |
| Day 2 | Feature validation | 6.8852 | +4.54% | DEC-014 (33 features) |
| Day 3 | Q4+Q1 training | 6.8360 | +5.22% | DEC-016 (temporal) |
| Day 3 | XGBoost tuned | 6.4860 | +10.08% | Hyperparameter tuning |
| Day 4 | LSTM baseline | 6.1947 | +14.13% | LSTM beats XGBoost |
| **Day 5** | **LSTM final** | **6.2552** | **+13.28%** | **Production artifacts** |

**Total improvement: 13.28% (7.2127 → 6.2552)**

---

### Major Discoveries

**Discovery 1: LSTM Beats XGBoost (Day 4)**

*Finding:* LSTM achieved 6.1947 RMSE vs XGBoost's 6.4860 (4.49% better)

*Why unexpected:*
- XGBoost typically dominates tabular data competitions
- Both models used identical 33 engineered features
- LSTM usually better for raw sequences, not pre-engineered features

*Why LSTM won:*
- Superior generalization (0.57x overfitting ratio vs 2.58x for XGBoost)
- Dropout regularization (0.2) more effective than tree constraints
- Sequential processing captured temporal patterns beyond engineered features
- Avoided memorizing training noise

*Portfolio value:* Demonstrates breadth (tree + neural) and willingness to challenge assumptions

---

**Discovery 2: Temporal Consistency Principle (Day 3, DEC-016)**

*Finding:* Q4 2013 + Q1 2014 (19K samples) outperformed full 2013 (50K samples)

*Results:*
- Q4+Q1 training: RMSE = 6.84
- Full 2013 training: RMSE = 14.88 (106% WORSE)
- Improvement: 54% better with 2.6x LESS data

*Key insight:*
> "In seasonal time series forecasting, temporal relevance trumps data volume"

*Why:*
- March test period is spring season
- Q4+Q1 training covers similar seasonal patterns (fall/winter/early spring)
- Full 2013 includes irrelevant summer patterns and extreme Nov-Dec holidays
- Model learned patterns that don't exist in March

*Business translation:*
"Using 6 months of recent, relevant history beat using full year of mismatched history"

*Portfolio value:* Demonstrates deep time series understanding beyond basic ML

---

**Discovery 3: Feature Reduction Prevents Overfitting (Day 2, DEC-014)**

*Finding:* 33 features outperformed 45 features (6.89 vs 7.21 RMSE)

*Features removed (12 total):*
- Rolling std (3): unit_sales_7d_std, 14d_std, 30d_std
- Oil features (6): oil_price and all lag/change variants
- Promotion interactions (3): promo_holiday_category, promo_item_avg, promo_cluster

*Validation method:*
- Permutation importance analysis
- Ablation studies (remove features, measure impact)
- SHAP analysis planned but deferred

*Result:* Simpler model generalizes better

*Portfolio value:* Shows systematic feature validation methodology

---

**Discovery 4: DEC-015 Hypothesis Rejected (Day 3)**

*Hypothesis:* More training data (full 2013) will improve performance

*Test results:*
- Full 2013 (50K samples): RMSE = 14.88 (catastrophic failure)
- Even with outlier clipping: RMSE = 7.39 (still worse than Q1-only)

*Root cause:*
- Seasonal mismatch between Nov-Dec holidays (train) and March normal (test)
- Training max: 1,332 units (Black Friday/Christmas)
- Test max: 222 units (6x lower)
- Model learned extreme patterns that don't generalize

*Decision:* Rejected hypothesis, pivoted to Q4+Q1 approach (DEC-016)

*Portfolio value:* Demonstrates scientific rigor and willingness to reject sunk effort

---

### Decision Log Summary

**DEC-013: Train/Test Gap Period (7 days)**
- Decision: Add 7-day gap between training end and test start
- Rationale: Prevent data leakage from lag features
- Impact: Valid temporal split, no lookahead bias
- Status: APPLIED to all models

**DEC-014: Feature Reduction Based on Ablation (45 → 33 features)**
- Decision: Remove 12 harmful features identified via permutation importance
- Features removed: Rolling std (3), Oil (6), Promotion interactions (3)
- Impact: 4.5% RMSE improvement (7.21 → 6.89)
- Status: APPLIED to all models

**DEC-015: Expand Training Data to Full 2013 (REJECTED)**
- Decision: REJECTED after catastrophic test failure
- Tested: Full 2013 + Jan-Feb 2014 (50K samples)
- Result: RMSE = 14.88 (106% worse than Q1-only)
- Reason: Seasonal mismatch between training and test periods
- Status: REJECTED, documented as negative result

**DEC-016: Temporal Consistency Over Data Volume**
- Decision: Use Q4 2013 + Q1 2014 for training (19K samples)
- Rationale: Seasonal alignment more important than data volume
- Impact: 0.7% improvement over Q1-only, 54% better than full 2013
- Status: APPLIED to final model

**Summary:** 3 applied, 1 rejected with clear documentation

---

## 3. Final Model Specification

### Model Details

**Type:** LSTM (Long Short-Term Memory Neural Network)

**Architecture:**
```
Input: (1 timestep, 33 features)
├── LSTM(64 units, return_sequences=False)
├── Dropout(0.2)
├── Dense(32 units, activation='relu')
├── Dropout(0.2)
└── Dense(1 unit) # Output: sales prediction
Total parameters: 27,201
```

**Training Configuration:**
- Optimizer: Adam
- Loss function: MSE (mean squared error)
- Batch size: 32
- Validation split: 20% (3,781 samples)
- Early stopping: patience=10 epochs
- Epochs trained: 21-26 (varies by run)
- Training time: ~20-36 seconds

**Performance:**
- Test RMSE: 6.2552 (primary metric)
- Test MAE: ~3.05
- Training RMSE: ~10.55
- Overfitting ratio: 0.57x (underfitting - excellent generalization)

**Why LSTM Won:**
1. Better generalization than XGBoost (0.57x vs 2.58x overfitting)
2. Dropout regularization more effective
3. Captured sequential patterns beyond engineered features
4. Avoided memorizing training noise

---

### Data Configuration

**Training Data:**
- Period: Q4 2013 + Q1 2014 (October 1, 2013 - February 21, 2014)
- Samples: 18,905
- Rationale: DEC-016 temporal consistency (seasonal alignment with test)

**Test Data:**
- Period: March 2014 (March 1-31, 2014)
- Samples: 4,686
- Temporal gap: 7 days (Feb 22-28, 2014) per DEC-013

**Features (33 total):**

*Temporal features (8):*
- Lag features: unit_sales_lag1, lag7, lag14, lag30
- Rolling means: unit_sales_7d_avg, 14d_avg, 30d_avg
- Autocorrelation metric: unit_sales_lag1_7d_corr

*Calendar features (7):*
- year, month, day, dayofweek, dayofyear, weekofyear, quarter

*Holiday features (4):*
- holiday_proximity, is_holiday, holiday_period (categorical), days_to_next_holiday

*Promotion features (2):*
- onpromotion, promo_item_interaction

*Store/Item features (7):*
- cluster, store_avg_sales, item_avg_sales, item_store_avg
- cluster_avg_sales, family_avg_sales, city_avg_sales

*Derived features (5):*
- perishable (item property)
- weekend (derived from dayofweek)
- month_start, month_end, is_payday

**Feature engineering decisions:**
- DEC-014: Removed 12 features (rolling std, oil, promotion interactions)
- All features pre-engineered in Week 2
- Same 33 features used for both XGBoost and LSTM (fair comparison)

---

### Preprocessing Pipeline

**Step-by-step process:**

1. **Feature selection:**
   - Load 33 features from feature_columns.json
   - Ensure features in correct order

2. **Categorical encoding:**
   - holiday_period → numeric codes (category.astype.codes)

3. **NaN handling:**
   - fillna(0) for all features
   - XGBoost handles NaN natively, but LSTM requires explicit handling

4. **Scaling:**
   - StandardScaler fitted on training data only
   - Transform both train and test using same scaler
   - Results: zero mean, unit variance

5. **Reshaping for LSTM:**
   - From (samples, features) to (samples, 1, features)
   - Single timestep since features already include temporal info

**Critical:** Use artifacts/scaler.pkl to ensure identical preprocessing in production

---

## 4. Deployment Artifacts

### Artifact Inventory

All artifacts located in: `/artifacts` directory

**1. lstm_model.keras (0.34 MB)**
- Native Keras format (TensorFlow 2.20.0+ compatible)
- Complete trained model with weights
- Load with: `keras.models.load_model('lstm_model.keras')`

**2. scaler.pkl (1.84 KB)**
- Fitted StandardScaler from training data
- Required for preprocessing new data
- Load with: `pickle.load(open('scaler.pkl', 'rb'))`

**3. feature_columns.json**
- Ordered list of 33 feature names
- Ensures correct feature order in production
- Load with: `json.load(open('feature_columns.json'))`

**4. model_config.json**
- Complete model metadata
- Performance metrics, architecture details, training config
- Decision log references

**5. model_usage.md**
- Step-by-step deployment instructions
- Code examples for loading and prediction
- Monitoring recommendations

### Artifact Validation

**Reproducibility test performed:**
- Cleared model from memory
- Loaded all artifacts from disk
- Made predictions on test set
- Result: Predictions identical (diff = 0.000000)

**Status:** ✓ Fully reproducible and deployment-ready

---

### Usage Example

```python
# Load artifacts
from tensorflow.keras.models import load_model
import pickle
import json
import pandas as pd
import numpy as np

model = load_model('artifacts/lstm_model.keras')
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('artifacts/feature_columns.json', 'r') as f:
    features = json.load(f)

# Prepare new data (X_new is DataFrame with raw features)
X_new = X_new[features]  # Ensure correct order
X_new['holiday_period'] = X_new['holiday_period'].astype('category').cat.codes
X_new_filled = X_new.fillna(0)
X_new_scaled = scaler.transform(X_new_filled)
X_new_lstm = X_new_scaled.reshape(X_new_scaled.shape[0], 1, 33)

# Predict
predictions = model.predict(X_new_lstm).flatten()
```

See `artifacts/model_usage.md` for complete instructions.

---

## 5. Experiment Tracking (MLflow)

### Experiment Summary

**Experiment name:** favorita-forecasting

**Total runs:** 6

**Run details:**

1. **xgboost_baseline** (Day 1)
   - RMSE: 7.2127
   - Config: Q1-only, 45 features
   - Status: Initial baseline

2. **feature_validation** (Day 2)
   - Purpose: Ablation studies
   - Identified: 12 harmful features
   - Result: DEC-014 created

3. **xgboost_baseline_2013train** (Day 3)
   - RMSE: 14.88
   - Config: Full 2013, 33 features
   - Status: DEPRECATED (DEC-015 failed)

4. **xgboost_baseline_q4q1** (Day 3)
   - RMSE: 6.84
   - Config: Q4+Q1, 33 features
   - Status: Valid baseline

5. **xgboost_tuned_q4q1** (Day 3)
   - RMSE: 6.49
   - Config: Q4+Q1, 33 features, tuned
   - Status: Best XGBoost

6. **lstm_baseline_q4q1** (Day 4)
   - RMSE: 6.19
   - Config: Q4+Q1, 33 features
   - Status: BEST MODEL

**Tracking includes:**
- All hyperparameters (model config)
- All metrics (RMSE, MAE, training time)
- All tags (phase, model_type, decisions_applied)
- Training history (loss curves)

**Week 4 usage:** Reference MLflow runs for presentation charts and technical report

---

## 6. Documentation Inventory

### Notebooks (5 complete)

1. **w03_d01_MODEL_baseline.ipynb**
   - XGBoost baseline implementation
   - Naive forecast comparison
   - Initial performance: RMSE 7.21

2. **w03_d02_MODEL_mlflow-features.ipynb**
   - Permutation importance analysis
   - Ablation studies
   - DEC-014 feature reduction identified

3. **w03_d03_MODEL_tuning.ipynb**
   - DEC-015 testing (full 2013, failed)
   - DEC-016 discovery (Q4+Q1 success)
   - Hyperparameter tuning
   - Best XGBoost: RMSE 6.49

4. **w03_d04_MODEL_lstm.ipynb**
   - LSTM implementation
   - XGBoost vs LSTM comparison
   - LSTM wins: RMSE 6.19

5. **w03_d05_MODEL_artifacts-export.ipynb**
   - Artifact export (5 files)
   - Reproducibility validation
   - Deployment preparation

---

### Checkpoints (5 complete)

1. **w03_d01_checkpoint.md** - Baseline modeling day
2. **w03_d02_checkpoint.md** - Feature validation day
3. **w03_d03_checkpoint.md** - XGBoost optimization day (DEC-015, DEC-016)
4. **w03_d04_checkpoint.md** - LSTM comparison day
5. **w03_d05_checkpoint.md** - Week 3 completion

Each checkpoint includes:
- Activities completed
- Performance summary
- Key findings
- Time allocation
- Next steps

---

### Decision Log (4 documents)

1. **DEC-013_Train_Test_Gap_Period.md**
   - 7-day gap prevents data leakage
   - Applied to all models

2. **DEC-014_Feature_Reduction_Based_on_Ablation.md**
   - 45 → 33 features
   - 4.5% improvement

3. **DEC-015_Expand_Training_Data_2013.md**
   - REJECTED (seasonal mismatch)
   - Documented negative result

4. **DEC-016_Temporal_Consistency_Matters.md**
   - Q4+Q1 training strategy
   - Temporal relevance > data volume principle

---

### Visualizations (4 figures)

1. **w03_d01_baseline_comparison.png**
   - Naive vs XGBoost comparison
   - Initial baseline performance

2. **w03_d02_feature_importance.png**
   - Permutation importance rankings
   - Features to remove identified

3. **w03_d03_final_comparison.png**
   - Model evolution Day 1-3
   - Q1-only vs Q4+Q1 vs Full 2013
   - DEC-016 visual evidence

4. **w03_d04_lstm_comparison.png**
   - XGBoost vs LSTM detailed comparison
   - Model progression Day 1-4
   - Final performance summary

All figures saved in: `/outputs/figures/models/`

---

## 7. Week 4 Deliverables & Expectations

### Primary Deliverables (Week 4 Days 1-5)

**1. Final Presentation (15-20 slides)**
*Purpose:* Communicate results to non-technical stakeholders

*Content:*
- Business problem and impact
- Methodology overview (accessible language)
- Key findings (DEC-016 temporal consistency, LSTM win)
- Performance improvement (13.28% RMSE improvement)
- Deployment readiness
- Next steps and recommendations

*Assets to use:*
- Week 3 visualizations (4 figures ready)
- MLflow performance charts
- Model comparison table
- Temporal consistency diagram

*Target audience:* Business stakeholders, non-technical managers

---

**2. Technical Report (20-25 pages)**
*Purpose:* Document complete technical approach

*Sections:*
1. Introduction & Problem Statement
2. Data Exploration (Week 1 summary)
3. Feature Engineering (Week 2 summary)
4. Modeling Approach (Week 3 detailed)
5. Results & Analysis
6. Deployment Architecture
7. Conclusions & Future Work
8. Appendices (decision logs, code references)

*Assets to use:*
- All checkpoint documents
- Decision log (DEC-013 through DEC-016)
- MLflow experiment tracking
- Code snippets from notebooks
- Performance tables and figures

*Target audience:* Technical reviewers, data science peers, academic advisor

---

**3. Lightweight Web App (Streamlit/Gradio)**
*Purpose:* Interactive demonstration of forecasting model

*Features:*
- Load model from artifacts
- Input form for new predictions
- Visualization of forecasts
- Model performance dashboard
- Feature importance display

*Implementation:*
- Use artifacts/lstm_model.keras
- Load scaler.pkl and feature_columns.json
- Simple UI for demonstration
- Deploy locally (Streamlit Sharing or Gradio optional)

*Target audience:* Stakeholder demos, portfolio showcase

---

**4. Video Walkthrough (10-15 minutes)**
*Purpose:* Explain project end-to-end

*Content:*
- Problem statement and business context (2 min)
- Data exploration highlights (2 min)
- Feature engineering overview (2 min)
- Modeling approach and comparison (4 min)
- Key findings (DEC-016, LSTM win) (3 min)
- Deployment and next steps (2 min)

*Assets:*
- Screen recording of notebooks
- Web app demonstration
- Presentation slides
- MLflow experiment tracking

*Target audience:* Portfolio, job applications, stakeholder briefing

---

**5. Code Consolidation & GitHub**
*Purpose:* Organize and publish reproducible code

*Structure:*
```
retail_demand_analysis/
├── notebooks/        # 11 notebooks (Week 1-3)
├── data/
│   ├── raw/         # Original Kaggle data
│   └── processed/   # Filtered, featured data
├── artifacts/       # Deployment-ready model
├── outputs/
│   └── figures/     # All visualizations
├── docs/
│   ├── plans/       # Weekly project plans
│   ├── decisions/   # Decision log
│   └── reports/     # Final report
├── presentation/    # Final slides
├── README.md        # Project overview
└── requirements.txt # Dependencies
```

*Actions:*
- Add comprehensive README
- Create requirements.txt
- Ensure all paths are relative
- Test reproducibility
- Publish to GitHub (public repo for portfolio)

---

### Week 4 Timeline

**Day 1-2: Presentation + Report**
- Draft presentation slides (non-technical)
- Write technical report sections
- Create additional visualizations if needed
- Review and polish

**Day 3-4: Web App + Video**
- Build Streamlit/Gradio app
- Test with artifacts
- Record video walkthrough
- Edit and finalize

**Day 5: Final Review & Submission**
- Code consolidation
- GitHub setup and README
- Final review of all deliverables
- Submission preparation

---

## 8. Key Messages for Week 4

### For Non-Technical Stakeholders

**Problem:**
"Corporación Favorita needed accurate sales forecasts to optimize inventory and reduce waste across their grocery stores in Ecuador."

**Solution:**
"We built a machine learning model that predicts daily sales 13% more accurately than baseline methods by focusing on recent, seasonally-relevant historical data."

**Key Insight:**
"More data isn't always better - using 6 months of recent, relevant history beat using a full year of mismatched history. We call this 'temporal consistency.'"

**Impact:**
"Better forecasts mean less spoilage of perishable goods, fewer stockouts, and improved customer satisfaction."

---

### For Technical Audience

**Approach:**
"Compared tree-based (XGBoost) and neural network (LSTM) approaches using 33 engineered features derived from extensive EDA and feature engineering."

**Finding:**
"LSTM unexpectedly outperformed XGBoost by 4.5% due to superior generalization (0.57x overfitting ratio vs 2.58x), despite conventional wisdom favoring tree-based models for tabular data."

**Innovation:**
"Discovered temporal consistency principle (DEC-016): Q4+Q1 training (19K samples) beat full 2013 (50K samples) by 54% because seasonal alignment matters more than data volume."

**Rigor:**
"Systematically tested hypothesis (DEC-015) about expanding training data, found it failed catastrophically (106% worse), and pivoted to superior Q4+Q1 approach - demonstrating scientific method in practice."

---

### Portfolio Highlights

**What makes this project stand out:**

1. **Breadth of techniques:** Feature engineering, XGBoost, LSTM, time series analysis
2. **Unexpected finding:** LSTM winning on tabular data challenges conventional wisdom
3. **Scientific rigor:** Hypothesis testing with rejection of DEC-015
4. **Deep analysis:** Explained why LSTM won (generalization vs memorization)
5. **Complete tracking:** MLflow for all 6 experiments
6. **Production-ready:** Deployment artifacts exported and validated
7. **Clear communication:** Decision logs, checkpoints, comprehensive documentation

**Interview talking points:**

*"Tell me about a project where you had an unexpected result"*
→ LSTM beating XGBoost story, generalization analysis

*"Describe your approach to feature engineering"*
→ Week 2 systematic approach, DEC-014 ablation studies

*"How do you handle failed experiments?"*
→ DEC-015 rejection, pivot to DEC-016, documented negative results

*"Walk me through your model deployment process"*
→ Artifacts export, reproducibility validation, deployment instructions

---

## 9. Open Questions & Future Work

### Limitations Acknowledged

1. **Sparsity not explicitly modeled**
   - 99.29% data sparsity (true zeros vs missing)
   - Implicitly handled via fillna(0) and learned patterns
   - Future: Specialized sparse/intermittent demand models

2. **Single test period**
   - Only tested on March 2014
   - Future: Multi-period validation (Apr, May, Jun)

3. **Limited hyperparameter search**
   - RandomizedSearchCV with 20 iterations
   - Future: Bayesian optimization, NAS for LSTM architecture

4. **No ensemble methods**
   - Only tested individual models
   - Future: XGBoost + LSTM ensemble

5. **Single-step forecasting**
   - Predicts one day ahead
   - Future: Multi-step forecasting (7-day, 14-day horizons)

### Future Improvements

**Short-term (if time allows in Week 4):**
- Confidence intervals for predictions (quantile regression)
- Error analysis by product family, store cluster
- Interactive dashboard with prediction explanations

**Long-term (beyond project scope):**
- Deploy to production environment (AWS, GCP)
- A/B testing vs existing forecasting methods
- Automated retraining pipeline (MLOps)
- Incorporate external data (weather, economic indicators)
- Multi-location forecasting (beyond Guayas)

---

## 10. Critical Handoff Checklist

### For Week 4 Team (or Future Self)

**Before starting Week 4, verify:**

- [ ] All artifacts in `/artifacts` directory accessible
- [ ] MLflow experiment tracking available
- [ ] All 5 Week 3 notebooks executable
- [ ] All 4 decision logs readable
- [ ] All 4 visualizations available
- [ ] Feature dictionary and data processed files present

**Week 4 dependencies:**
- [ ] lstm_model.keras loads without error
- [ ] scaler.pkl and feature_columns.json readable
- [ ] model_config.json contains all necessary metadata
- [ ] Week 3 checkpoints provide sufficient context

**Communication assets ready:**
- [ ] Performance tables and comparison charts
- [ ] MLflow run links and screenshots
- [ ] Decision log summaries
- [ ] Key findings documented

**If anything missing or unclear:**
- Review Week 3 checkpoints (especially w03_d05_checkpoint.md)
- Check decision logs for detailed rationale
- Reference MLflow runs for experiment details
- Consult model_usage.md for deployment guidance

---

## 11. Contact & Support

**Project Owner:** Alberto Diaz Durana  
**Primary Contact:** [GitHub](https://github.com/albertodiazdurana)

**Key Documentation Locations:**
- Checkpoints: `/outputs/w03_d0X_checkpoint.md`
- Decision Logs: `/outputs/DEC-0XX_*.md`
- Artifacts: `/artifacts/`
- Notebooks: `/notebooks/w03_d0X_*.ipynb`
- Figures: `/outputs/figures/models/`

**MLflow:**
- Experiment: "favorita-forecasting"
- Best run: "lstm_baseline_q4q1" (RMSE = 6.19)

---

## 12. Sign-Off

**Week 3 Status:** COMPLETE  
**Deliverables:** 100% complete (5 notebooks, 5 checkpoints, 4 decisions, 5 artifacts, 4 figures)  
**Best Model:** LSTM (RMSE = 6.2552, 13.28% improvement)  
**Production Readiness:** ✓ Artifacts exported and validated  
**Week 4 Readiness:** ✓ Complete handoff, clear deliverables, sufficient assets

**Signed off by:** Alberto Diaz Durana  
**Date:** 2025-11-20

---

**Week 4: Let's communicate this excellent work to stakeholders and prepare for deployment!**

---

**END OF WEEK 3 TO WEEK 4 HANDOFF**
