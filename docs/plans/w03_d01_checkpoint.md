# Week 3 Day 1 Checkpoint - Baseline Modeling

**Project:** Corporación Favorita Grocery Sales Forecasting  
**Phase:** Week 3 - Modeling & Analysis  
**Day:** Day 1 of 5  
**Date:** 2025-11-18  
**Status:** COMPLETE

---

## Summary

**Day 1 Objective:** Establish XGBoost baseline model with comprehensive evaluation

**Status:** 100% Complete - All objectives achieved

**Key Achievement:** Baseline RMSE of 7.21 established with 41.75% improvement over naive forecast

---

## Completed Activities

### Part 1: Data Loading & Verification (30 min)
- Loaded w02_d05_FE_final.pkl (300,896 × 57 columns)
- Verified dataset structure and temporal order
- Confirmed 29 engineered features + 28 base features
- Memory usage: 219.9 MB

**Output:**
- Dataset shape: (300896, 57)
- Date range: 2013-01-02 to 2017-08-15
- Quality checks passed

### Part 2: Train/Test Split with Gap Period (45 min)
- Filtered to Q1 2014: 12,668 rows (4.2% of full dataset)
- Implemented 7-day gap to prevent lag7 leakage
- Train: Jan 1 - Feb 21, 2014 (7,050 rows, 52 days)
- Gap: Feb 22 - Feb 28, 2014 (932 rows EXCLUDED)
- Test: March 1-31, 2014 (4,686 rows, 31 days)

**Decision Made:**
- **DEC-013: 7-Day Gap Train/Test Split**
  - Context: Need to prevent lag feature leakage while maintaining training data
  - Decision: 7-day gap prevents lag7 leakage (strongest feature, r=0.40)
  - Trade-off: lag14/lag30 still use some training period (acceptable)
  - Impact: Balances data leakage prevention with sufficient training data (52 days)

**Output:**
- Train/test ratio: 60.1% / 39.9%
- Gap successfully prevents lag7 leakage
- No temporal overlap between sets

### Part 3: Feature & Target Separation (30 min)
- Excluded: id, date, store_nbr, item_nbr, unit_sales, categorical text (12 columns)
- Retained: 45 features across 6 categories
- Fixed 2 object dtype columns (holiday_period, promo_holiday_category) → category dtype
- Handled NaN in lag features (XGBoost native support per DEC-011)

**Feature Breakdown:**
- Base features: 3 (onpromotion, perishable, cluster)
- Lag features: 7 (unit_sales lag 1/7/14/30, oil price lags)
- Rolling features: 12 (7d/14d/30d avg/std)
- Oil features: 6 (price, lags, changes)
- Aggregation features: 12 (store/cluster/item baselines)
- Promotion features: 3 (interactions)
- Categorical features: 2 (holiday_period, promo_holiday_category)

**Output:**
- X_train: (7050, 45)
- X_test: (4686, 45)
- y_train: (7050,)
- y_test: (4686,)

### Part 4: XGBoost Baseline Training (30 min)
- Initialized XGBRegressor with default parameters
- Enabled categorical feature support
- Trained on 7,050 samples × 45 features
- Training time: 0.28 seconds

**Model Configuration:**
- n_estimators: 100
- learning_rate: 0.3 (default)
- max_depth: 6 (default)
- random_state: 42
- enable_categorical: True

**Training Performance:**
- Train RMSE: 1.21
- Train MAE: 0.84
- Expected overfitting (will evaluate on test)

**Output:**
- Model trained successfully
- 45 features captured
- Ready for test evaluation

### Part 5: Test Set Evaluation (45 min)
- Generated predictions on 4,686 test samples
- Calculated 6 comprehensive metrics
- Compared to naive baseline

**Comprehensive Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 3.10 | Average error magnitude |
| **RMSE** | 7.21 | Primary metric (std of errors) |
| **Bias** | 0.04 | Nearly unbiased (slight over-forecast) |
| **MAD** | 1.29 | Median error (robust) |
| **rMAD** | 43.11% | Median error as % of median sales |
| **MAPE** | 69.11% | High due to sparse low-volume items |

**Performance Context:**
- Mean actual sales: 7.26 units
- Mean predicted sales: 7.29 units
- RMSE as % of mean: 99.38%
- Naive baseline RMSE: 12.38
- **Improvement over naive: 41.75%**

**Output:**
- Baseline RMSE: 7.21 (target for Day 3 tuning)
- Model nearly unbiased (0.04 over-forecast)
- Strong improvement validates Week 2 feature engineering

### Part 6: Prediction Visualizations (30 min)
- Created 4-panel diagnostic visualization
- Saved as w03_d01_baseline_evaluation.png

**Visualization Panels:**
1. **Actual vs Predicted:** Points clustered near diagonal, good predictions
2. **Residuals Distribution:** Centered near zero, confirms unbiased model
3. **Residuals vs Predicted:** Random scatter with fan shape (heteroscedasticity expected)
4. **Error by Sales Magnitude:** Higher errors at higher volumes (larger absolute scale)

**Key Visual Insights:**
- No systematic bias detected
- Heteroscedasticity present (normal for retail sparse data)
- Prediction quality consistent across sales ranges
- Model captures main patterns effectively

**Output:**
- w03_d01_baseline_evaluation.png (4-panel, 150 DPI)

---

## Key Findings

### 1. Baseline Performance Strong
- **41.75% improvement** over naive baseline validates feature engineering effort
- RMSE of 7.21 provides clear target for hyperparameter tuning (Day 3)
- Nearly unbiased predictions (0.04 over-forecast) indicate well-calibrated model

### 2. Gap Period Successful
- 7-day gap prevents lag7 leakage (most important lag feature)
- Training data (52 days) sufficient for baseline model
- Trade-off between leakage prevention and data availability well-balanced

### 3. Feature Engineering Validated
- All 45 features accepted by XGBoost (including categorical)
- NaN handling strategy (DEC-011) works as expected
- Week 2 feature categories all represented

### 4. Sparse Data Handled Well
- XGBoost handles 99.1% sparsity naturally
- High MAPE (69%) expected in sparse retail with low-volume items
- MAD (1.29) and rMAD (43%) provide robust metrics less affected by outliers

### 5. Model Diagnostics Clean
- No systematic bias in residuals
- Heteroscedasticity expected in retail forecasting
- Prediction range reasonable (no extreme outliers predicted)

---

## Deliverables

### Completed
- [x] w03_d01_MODEL_baseline.ipynb (notebook with all cells executed)
- [x] Baseline XGBoost model (trained, in memory)
- [x] Comprehensive evaluation (6 metrics calculated)
- [x] w03_d01_baseline_evaluation.png (4-panel visualization)
- [x] w03_d01_checkpoint.md (this document)

### Ready for Day 2
- [x] Baseline RMSE established: 7.21
- [x] Model object available for MLflow logging
- [x] Test predictions available for comparison
- [x] Feature matrices prepared (X_train, X_test, y_train, y_test)
- [x] Feature list documented (45 features)

---

## Decision Log Updates

### DEC-013: Train/Test Split with 7-Day Gap
**Context:** Need to prevent lag feature leakage while maintaining sufficient training data for baseline model. Maximum lag feature is 30 days (unit_sales_lag30).

**Decision:** Implement 7-day gap between training and test sets
- Train: January 1 - February 21, 2014 (52 days)
- Gap: February 22 - February 28, 2014 (7 days, excluded)
- Test: March 1 - March 31, 2014 (31 days)

**Rationale:**
- Prevents lag7 leakage (strongest autocorrelation feature: r=0.40 from Week 2)
- Balances data availability (52 days) vs leakage prevention
- Pragmatic for academic project with limited Q1 2014 data
- Strict 30-day gap would leave only 30 days training (insufficient)

**Trade-offs:**
- lag1: Uses gap period data (acceptable - minimal leakage)
- lag7: No leakage (gap prevents)
- lag14: Partial overlap with training period
- lag30: Uses training period data
- Acceptable compromise for project scope

**Impact:**
- Successfully prevents most critical leakage (lag7)
- Sufficient training data for baseline model (7,050 samples)
- Documented limitation for project report
- Can revisit if performance suffers in Day 3 tuning

**Related:** DEC-011 (Lag NaN Strategy), Week 2 lag feature engineering

---

## Time Allocation

| Activity | Planned | Actual | Notes |
|----------|---------|--------|-------|
| Data Loading | 30min | 30min | Efficient load, quality checks passed |
| Train/Test Split | 30min | 45min | Added gap period analysis (+15min) |
| Feature Separation | 30min | 30min | Fixed dtype issue quickly |
| Model Training | 30min | 30min | Fast training (0.28 sec) |
| Evaluation | 45min | 45min | 6 metrics calculated |
| Visualizations | 30min | 30min | 4-panel created |
| Documentation | 30min | 20min | Efficient summary |
| **Total** | **3h 15min** | **3h 10min** | **On schedule** |

---

## Next Steps - Day 2 Preview

### Objectives
1. Set up MLflow experiment tracking
2. Log baseline run (params, metrics, visualizations)
3. Compute permutation importance (identify top 15 features)
4. Generate SHAP analysis (feature impact direction and magnitude)
5. Run ablation studies (validate feature groups, especially DEC-012 oil features)

### Prerequisites
- [x] Baseline model trained
- [x] Test predictions available
- [x] Baseline metrics documented
- [x] Feature matrices prepared

### Expected Outputs
- MLflow experiment configured
- Baseline run logged to MLflow
- Permutation importance plot (top 15 features)
- SHAP summary plot (beeswarm)
- SHAP dependence plots (top 5 features)
- Ablation study results (oil, rolling, aggregations)
- Feature validation report (markdown)

### Installation Requirements for Day 2
```bash
pip install mlflow shap --break-system-packages
```

### Time Estimate
- Day 2: ~4 hours
- MLflow setup: 45 min
- Baseline logging: 45 min
- Permutation importance: 1 hour
- SHAP analysis: 1.5 hours

---

## Blockers & Risks

### Current Blockers
- None

### Resolved Issues
1. **Object dtype columns** - Fixed by converting to category dtype
2. **Gap period definition** - Resolved with 7-day pragmatic approach

### Risks for Day 2
- **Risk:** MLflow setup issues (environment conflicts)
  - Mitigation: Test installation early, use local tracking
  - Contingency: Manual logging to CSV if MLflow fails

- **Risk:** SHAP computation slow on full dataset
  - Mitigation: Use sample (1000-2000 rows) for SHAP analysis
  - Contingency: Reduce feature count for SHAP if needed

---

## Notes & Observations

### What Went Well
- Progressive cell-by-cell execution maintained quality and validation
- Gap period discussion caught important leakage issue proactively
- Object dtype fix was quick and clean
- Training extremely fast (0.28 seconds)
- Baseline performance strong (41.75% improvement)
- Visualizations clearly show model behavior

### What Could Be Improved
- Could have anticipated object dtype issue earlier
- Initial split didn't include gap period (caught during review)
- MAPE interpretation needs context (high due to sparse data, not model failure)

### Lessons Learned
- Always consider lag feature leakage in time series splits
- XGBoost categorical support requires explicit dtype conversion
- Retail forecasting metrics need careful interpretation (sparsity matters)
- Gap period is essential best practice for time series with lag features
- Baseline establishment is critical foundation for tuning

### Business Insights
- 41.75% improvement validates Week 2 feature engineering investment
- Model nearly unbiased (good for inventory planning)
- Heteroscedasticity in retail forecasting is expected (not a flaw)
- Low-volume items drive high MAPE (consider separate models by velocity)

---

## Session Continuity

### For Next Session Start
1. Load w02_d05_FE_final.pkl
2. Run train/test split code (7-day gap)
3. Load trained model or retrain baseline
4. Install MLflow and SHAP: `pip install mlflow shap --break-system-packages`
5. Reference this checkpoint for baseline metrics

### Quick Start Code for Day 2
```python
# Load data and recreate split
df = pd.read_pickle('w02_d05_FE_final.pkl')
df_2014q1 = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2014-03-31')].copy()
train = df_2014q1[df_2014q1['date'] <= '2014-02-21'].copy()
test = df_2014q1[df_2014q1['date'] >= '2014-03-01'].copy()

# Feature separation (exclude same 12 columns)
exclude_cols = ['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 
                'city', 'state', 'type', 'family', 'class',
                'holiday_name', 'holiday_type']
feature_cols = [col for col in train.columns if col not in exclude_cols]

# Fix object dtypes
object_cols = ['holiday_period', 'promo_holiday_category']
for col in object_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Create X, y
X_train = train[feature_cols].copy()
y_train = train['unit_sales'].copy()
X_test = test[feature_cols].copy()
y_test = test['unit_sales'].copy()

# Retrain baseline model
model = xgb.XGBRegressor(random_state=42, enable_categorical=True)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

# Baseline metrics from Day 1
baseline_rmse = 7.2127
baseline_mae = 3.0957
```

### Key Context for Day 2
- Baseline RMSE: 7.21 (target for improvement)
- 45 features across 6 categories
- 7-day gap prevents lag7 leakage
- Object dtypes require category conversion
- Model object: XGBRegressor with enable_categorical=True

---

## Week 3 Overall Progress

**Days completed:** 1 / 5 (20%)  
**Time spent:** ~3h 10min / planned 20h  
**Status:** On schedule

**Deliverables completed:**
- [x] Day 1: Baseline modeling (XGBoost, 6 metrics, visualization)
- [ ] Day 2: MLflow + Feature validation
- [ ] Day 3: Hyperparameter tuning
- [ ] Day 4: LSTM model (optional)
- [ ] Day 5: Artifacts export + documentation

**Buffer status:**
- Week 1 buffer: +8.5 hours
- Week 2 buffer: +8-10 hours
- Week 3 Day 1: On schedule
- **Total accumulated buffer: ~19-21 hours**

---

**Checkpoint completed by:** Alberto Diaz Durana  
**Next checkpoint:** Day 2 (MLflow + Feature Validation)  
**Status:** Ready to proceed to Day 2

---

**END OF DAY 1 CHECKPOINT**
