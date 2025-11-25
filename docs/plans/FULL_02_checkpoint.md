# FULL_02 Checkpoint: Train Final Models (Production Pipeline)

**Date:** 2025-11-24  
**Notebook:** FULL_02_train_final_model.ipynb  
**Status:** COMPLETE  
**Time Spent:** ~3 hours (XGBoost: 92 sec, Feature importance: 482 sec, LSTM attempted: 12+ hours stopped)

---

## Environment

**System:**
- OS: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2 Ubuntu 22.04)
- CPU: Intel(R) Core(TM) i7-10875H @ 2.30GHz
- GPU: NVIDIA Quadro T1000 with Max-Q Design (4096 MiB, 2.2GB available)
- RAM: ~16GB
- Python: 3.11.0rc1
- TensorFlow: 2.20.0 (GPU-enabled)
- XGBoost: 3.1.2

**Project Path:** /home/berto/Demand-forecasting-in-retail

---

## Input Data

**File:** data/processed/full_featured_data.pkl (1.3 GB)

| Metric | Value |
|--------|-------|
| Total Rows | 4,801,160 |
| Features | 33 (per DEC-014) |
| Period | Oct 1, 2013 - Mar 31, 2014 |
| Training Rows | 3,798,720 (Q4 2013 + Q1 2014, per DEC-016) |
| Gap Rows | 184,660 (7 days, per DEC-013) |
| Test Rows | 817,780 (March 2014) |

---

## Activities Completed

### 1. Data Loading & Preparation (30 min)
- Loaded full_featured_data.pkl (2.7 sec)
- Fixed UInt32 dtype issue in weekofyear feature
- Converted all features to float32
- Applied DEC-016 split: Oct 1, 2013 - Feb 21, 2014 (training)
- Applied DEC-013 gap: Feb 22-28, 2014 (7 days excluded)
- Test period: Mar 1-31, 2014

**Validation:**
- X_train: (3,798,720, 33)
- y_train: (3,798,720,)
- X_test: (817,780, 33)
- y_test: (817,780,)
- NaN count: 0

### 2. XGBoost Training (92 sec)
- Used Week 3 tuned hyperparameters
- Trained on 3.8M samples
- 500 estimators, max_depth=6
- Logged to MLflow: xgboost_full_q4q1

**Results:**
- RMSE: 6.4008
- MAE: 1.7480
- Bias: -0.4368
- MAPE (non-zero): 56.67%
- Training time: 92 seconds

### 3. LSTM Training Attempt (12+ hours, stopped)
- Scaled features with StandardScaler
- Architecture: LSTM(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Dense(1)
- Total parameters: 27,201
- GPU memory: 2248 MB allocated

**Outcome: Did Not Converge**
- Epoch 1: val_loss 88.8 (RMSE ~9.42)
- Epoch 6: val_loss 87.8 (RMSE ~9.37) - best
- Epoch 14: val_loss increasing, stopped training
- Training time per epoch: ~20 minutes
- Total time before stop: ~12 hours

**Conclusion:** LSTM advantage from Week 3 (300K sample) did not hold at scale (4.8M rows)

### 4. Model Comparison Analysis (15 min)
Created comparison table:

| Dataset | XGBoost | LSTM | Winner |
|---------|---------|------|--------|
| Week 3 (300K) | 6.4860 | 6.2552 | LSTM (-4.5%) |
| Full (4.8M) | 6.4008 | ~9.37 | XGBoost (+46%) |

**Key Finding:** Tree models scale better with large tabular datasets

### 5. Visualizations Created (20 min)
1. **model_comparison_full.png** - Bar chart comparing models at both scales
2. **sample_vs_full_comparison.png** - Table visualization of all metrics
3. **feature_importance_full.png** - Top 15 features from permutation importance

Saved to: outputs/figures/full_pipeline/

### 6. Feature Importance Analysis (482 sec)
- Permutation importance on test set (817K rows)
- 5 repeats with random_state=42

**Top 5 Features:**
1. unit_sales_7d_avg: 6.428
2. unit_sales_lag1_7d_corr: 1.955
3. unit_sales_lag1: 1.643
4. item_avg_sales: 0.299
5. unit_sales_14d_avg: 0.228

**Validation:** DEC-014 feature selection confirmed at scale

### 7. Production Artifacts Export (5 min)
Exported to artifacts/:
- xgboost_model_full.pkl (2.09 MB)
- scaler_full.pkl (1.39 KB)
- feature_columns.json
- model_config_full.json

---

## Key Decisions

### DEC-017 (NEW): XGBoost Selected Over LSTM at Production Scale

**Context:** Week 3 showed LSTM beating XGBoost by 4.5% on 300K sample

**Hypothesis Tested:** Does LSTM advantage hold at 4.8M rows (16x scale)?

**Result:** NO - LSTM failed to converge, XGBoost 46% better

**Evidence:**
- LSTM val_loss oscillated between 87.8 - 95.1
- Best LSTM RMSE ~9.37 vs XGBoost 6.40
- LSTM training time: 12+ hours (stopped) vs XGBoost: 92 sec

**Decision:** Use XGBoost as production model

**Rationale:**
1. Tree models handle large tabular datasets better
2. XGBoost converged quickly and reliably
3. LSTM computational cost prohibitive at scale
4. Neural networks may need architecture redesign for this scale

**Impact:** Production deployment proceeds with XGBoost
**Status:** APPROVED - documented in FULL_02

---

## Decisions Applied

| ID | Decision | Application | Result |
|----|----------|-------------|--------|
| DEC-013 | 7-day gap | Feb 22-28 excluded | Applied successfully |
| DEC-014 | 33 features | Feature importance validated | Confirmed at scale |
| DEC-016 | Q4+Q1 training | Oct 1 - Feb 21 | Applied successfully |
| DEC-017 | XGBoost over LSTM | Production model | NEW - documented |

---

## Performance Summary

### XGBoost (Production Model)

| Metric | Week 3 (300K) | Full (4.8M) | Change |
|--------|---------------|-------------|--------|
| RMSE | 6.4860 | 6.4008 | -1.31% ✓ |
| MAE | ~3.2 | 1.7480 | Improved |
| Training Time | ~30 sec | 92 sec | Scalable |
| Samples | 18,905 | 3,798,720 | 200x |

### LSTM (Not Used)

| Metric | Week 3 (300K) | Full (4.8M) | Change |
|--------|---------------|-------------|--------|
| RMSE | 6.2552 | ~9.37 | +50% ✗ |
| Training Time | 36 sec (CPU) | 12+ hours (GPU) | Not scalable |

---

## Issues Encountered & Solutions

### Issue 1: UInt32 dtype in weekofyear
**Problem:** Pandas UInt32 (nullable integer) incompatible with numpy operations  
**Solution:** Converted all features to float64 before numpy conversion  
**Code:** `df_train[col].astype('float64')`

### Issue 2: LSTM Not Converging
**Problem:** Validation loss oscillating/increasing instead of decreasing  
**Root Cause:** LSTM architecture not suitable for 3.8M tabular samples  
**Solution:** Stopped training after 14 epochs, documented as DEC-017  
**Learning:** Tree models > Neural networks for large-scale tabular time series

### Issue 3: Feature Importance Calculation Time
**Problem:** Permutation importance took 482 seconds  
**Expected:** Standard on 817K test samples with 5 repeats  
**Solution:** No issue, completed successfully  
**Note:** Use n_jobs=-1 for parallelization

---

## Validation Checklist

- [x] Data loaded correctly (4.8M rows)
- [x] Train/test split applied correctly (DEC-016 + DEC-013)
- [x] XGBoost trained successfully
- [x] LSTM convergence tested (documented failure)
- [x] Metrics calculated for both models
- [x] MLflow runs logged
- [x] Visualizations created
- [x] Feature importance validated
- [x] Production artifacts exported
- [x] Model comparison documented

---

## Deliverables

### Notebooks
- [x] FULL_02_train_final_model.ipynb (complete with markdown summaries)

### Figures (outputs/figures/full_pipeline/)
- [x] model_comparison_full.png
- [x] sample_vs_full_comparison.png
- [x] feature_importance_full.png

### Artifacts (artifacts/)
- [x] xgboost_model_full.pkl (2.09 MB)
- [x] scaler_full.pkl (1.39 KB)
- [x] feature_columns.json
- [x] model_config_full.json

### MLflow
- [x] Experiment: full_pipeline_model_comparison
- [x] Run 1: xgboost_full_q4q1 (complete)
- [x] Run 2: lstm_full_q4q1 (incomplete - documented)

---

## Key Findings

### 1. Scale Reverses Model Advantage
- Small sample (300K): LSTM wins
- Large scale (4.8M): XGBoost wins decisively
- Lesson: Always validate at production scale

### 2. XGBoost Scalability
- More data → Better performance (-1.31% RMSE)
- Training time scales linearly (30s → 92s for 200x data)
- Reliable convergence

### 3. Feature Importance Stability
- Top features consistent from Week 3
- Rolling averages (7d_avg) dominate: 6.43 importance
- Lag features crucial: lag1, lag7 in top 10

### 4. Computational Efficiency
- XGBoost: 92 seconds for 3.8M samples
- LSTM: 12+ hours without convergence
- Decision: XGBoost is production-ready

---

## Next Steps (Week 4 Day 2)

### Immediate (Day 2 Morning)
- [x] Copy artifacts to Streamlit app repo
- [ ] Update app/config.py with XGBoost model paths
- [ ] Implement model loading in model/model_utils.py
- [ ] Test artifact loading locally

### Day 2 Afternoon
- [ ] Complete Streamlit app development
- [ ] Test full workflow locally
- [ ] Prepare for Day 3 deployment

### Documentation
- [ ] Create DEC-017 decision document
- [ ] Update Week4_ProjectPlan with actual results
- [ ] Add LSTM failure analysis to portfolio narrative

---

## Lessons Learned

1. **Validate at Scale:** Small sample winners may not hold at production scale
2. **Document Negative Results:** LSTM failure is valuable finding for portfolio
3. **Computational Cost Matters:** 92 sec vs 12+ hours is decisive for production
4. **Tree Models Excel:** XGBoost remains king for tabular time series at scale

---

**Checkpoint completed by:** Alberto Diaz Durana  
**FULL_02 Status:** COMPLETE  
**Production Model:** XGBoost (RMSE 6.4008)  
**Next Phase:** Week 4 Day 2 - Streamlit App Development

---

**END OF FULL_02 CHECKPOINT**
