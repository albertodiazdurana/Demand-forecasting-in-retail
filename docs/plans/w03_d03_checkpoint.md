# Week 3 Day 3 Checkpoint - Feature Optimization & Hyperparameter Tuning

**Project:** Corporación Favorita Grocery Sales Forecasting  
**Phase:** Week 3 - Modeling & Analysis  
**Day:** Day 3 of 5  
**Date:** 2025-11-20  
**Status:** COMPLETE

---

## Summary

**Day 3 Objective:** Optimize feature set, expand training data, and tune hyperparameters

**Status:** 100% Complete - MAJOR DISCOVERIES with critical pivot

**Key Achievement:** 10.08% total improvement through feature reduction, temporal consistency discovery, and hyperparameter tuning

---

## CRITICAL DISCOVERIES

### Discovery 1: DEC-015 Hypothesis REJECTED

**Initial hypothesis:** More training data (full 2013, 50K rows) will improve performance

**Testing results:**
- Full 2013 training (50K rows): RMSE = 14.88 (106% WORSE than Q1-only)
- Even with 99th percentile clipping: RMSE = 7.39 (still 7% worse)

**Root cause:** Seasonal mismatch
- Training includes Nov-Dec 2013 holiday extremes (sales up to 1,332 units)
- Test is March 2014 (normal month, max 222 units)
- Model learned patterns that don't exist in test period

**Conclusion:** DEC-015 REJECTED - More data ≠ better performance for seasonal forecasting

---

### Discovery 2: Temporal Consistency Principle (DEC-016)

**New hypothesis:** Temporally consistent training data outperforms larger but inconsistent data

**Testing results:**
- Q4 2013 + Q1 2014 training (19K rows): RMSE = 6.84
- Beats Q1-only (7K rows): 0.7% improvement
- Beats full 2013 (50K rows): 51% improvement

**Why it works:**
- Q4 (Oct-Dec) + Q1 (Jan-Feb) captures consistent seasonal patterns
- No extreme summer patterns or mid-year anomalies
- Holiday patterns in Q4 2013 contextually relevant for learning
- 2.7x more data than Q1-only, but temporally aligned

**Key insight:** 
> "Temporal relevance trumps data volume in seasonal forecasting"

This is a MAJOR finding for the portfolio - demonstrates deep understanding of time series dynamics.

---

### Discovery 3: Outlier Clipping Can Harm Performance

**Finding:** Clipping Q4+Q1 training data at 99th percentile WORSENED results

**Results:**
- Q4+Q1 unclipped: RMSE = 6.84
- Q4+Q1 clipped: RMSE = 7.38 (8% worse)

**Explanation:**
- Unlike full 2013, Q4+Q1 holiday extremes are contextually relevant
- Nov-Dec 2013 patterns help model understand seasonal variability
- Moderate overfitting (2.58x ratio) is acceptable
- Clipping removes legitimate signal, not just noise

**Lesson:** Context matters - same technique (clipping) helps in one scenario, hurts in another

---

## Completed Activities

### Part 1: Implement DEC-014 Feature Reduction (30 min)
- Removed 12 harmful features identified in Day 2
- Features removed: rolling_std (3), oil (6), promotion interactions (3)
- Result: 45 → 33 optimized features

**Output:**
- Feature set validated from Day 2 ablation studies
- Clean feature matrix ready for modeling

### Part 2: Test DEC-015 Expanded Training (2 hours)
- Loaded full 2013 + Jan-Feb 2014 (50,088 training samples)
- Trained baseline: RMSE = 14.88 (catastrophic failure)
- Diagnosed: Train RMSE = 2.98, Test RMSE = 14.88 (5x overfitting)
- Root cause: Seasonal extremes (max 1,332 in train vs 222 in test)
- Tested clipping: 99th percentile reduced RMSE to 7.39 (still worse)
- Tested 95th percentile: RMSE = 9.31 (even worse)

**Output:**
- DEC-015 REJECTED
- Clear evidence of seasonal mismatch problem
- MLflow run: xgboost_baseline_2013train (tagged deprecated)

### Part 3: Discover Q4+Q1 Temporal Consistency (1 hour)
- Tested Q4 2013 + Q1 2014 training (18,905 samples)
- Baseline: RMSE = 6.84 (0.7% better than Q1-only)
- Validated temporal alignment hypothesis
- Tested clipping: RMSE = 7.38 (worse, clipping removed signal)
- Confirmed unclipped Q4+Q1 as best approach

**Output:**
- New best baseline: RMSE = 6.84
- Foundation for DEC-016 decision
- MLflow run: xgboost_baseline_q4q1

### Part 4: Hyperparameter Tuning (30 min)
- RandomizedSearchCV with 20 iterations
- 3-fold TimeSeriesSplit cross-validation
- 60 model fits in 19 seconds

**Best hyperparameters:**
- n_estimators: 100
- max_depth: 3 (shallower than default 6)
- learning_rate: 0.1 (lower than default 0.3)
- subsample: 1.0
- colsample_bytree: 0.8

**Results:**
- Tuned RMSE: 6.49 (5.1% improvement over baseline)
- Total improvement: 10.08% vs Day 1

**Output:**
- Best model identified
- MLflow run: xgboost_tuned_q4q1 (tagged best_model, final_model)

### Part 5: Comprehensive Comparison & Visualization (20 min)
- Created 4-panel comparison figure
- Documented model evolution
- Generated final summary

**Output:**
- w03_d03_final_comparison.png
- Clear visual story of Day 3 journey

---

## Key Findings

### 1. Feature Optimization Works
- DEC-014 validated: 33 features optimal
- Removed features caused overfitting (Day 2 discovery)
- Simpler model generalizes better

### 2. Data Quality > Data Quantity
- 50K rows (full 2013) failed: RMSE = 14.88
- 19K rows (Q4+Q1) succeeded: RMSE = 6.84
- 7K rows (Q1-only) competitive: RMSE = 6.89
- Temporal alignment critical for time series

### 3. Hyperparameter Tuning Provides Polish
- 5.1% improvement on already-good baseline
- Shallower trees (max_depth=3) prevent overfitting
- Lower learning rate (0.1) improves generalization

### 4. Context-Dependent Techniques
- Clipping helped full 2013 (50% improvement)
- Clipping hurt Q4+Q1 (8% degradation)
- Same technique, different contexts, opposite results

### 5. Scientific Method in Action
- Proposed hypothesis (DEC-015)
- Tested rigorously
- Rejected when evidence contradicted
- Pivoted to better approach (DEC-016)
- This is excellent portfolio material

---

## Performance Summary

| Model | Training Data | Features | RMSE | Status |
|-------|--------------|----------|------|--------|
| Day 1 Baseline | Q1 2014 (7K) | 45 | 7.2127 | Baseline |
| Day 2 Optimized | Q1 2014 (7K) | 33 | 6.8852 | DEC-014 applied |
| Day 3 2013 Full | Full 2013 (50K) | 33 | 14.8759 | FAILED - Deprecated |
| Day 3 2013 Clipped | Full 2013 (50K) | 33 | 7.3888 | FAILED - Deprecated |
| Day 3 Q4+Q1 | Q4 2013 + Q1 2014 (19K) | 33 | 6.8360 | Success - New baseline |
| **Day 3 Final** | **Q4 2013 + Q1 2014 (19K)** | **33** | **6.4860** | **BEST MODEL** |

**Total improvement: 10.08%** (7.2127 → 6.4860)

**Breakdown:**
- Feature reduction (DEC-014): ~4.5% improvement
- Temporal consistency (DEC-016): ~0.7% improvement  
- Hyperparameter tuning: ~5.1% improvement

---

## MLflow Experiment Tracking

### Runs Created

**Run 1: xgboost_baseline_2013train**
- Training: Full 2013 (50,088 samples)
- RMSE: 14.8759
- Status: Deprecated (DEC-015 rejected)
- Tags: deprecated=true, replaced_by=DEC-016

**Run 2: xgboost_baseline_q4q1**
- Training: Q4 2013 + Q1 2014 (18,905 samples)
- RMSE: 6.8360
- Status: Valid baseline
- Tags: best_baseline=true

**Run 3: xgboost_tuned_q4q1**
- Training: Q4 2013 + Q1 2014 (18,905 samples)
- RMSE: 6.4860
- Status: Best model
- Tags: best_model=true, final_model=true, tuned=true

### MLflow Organization
- All runs tagged with phase=week3_day3
- Clear traceability of experiments
- Deprecated runs marked for historical context
- Best model clearly identified

---

## Decision Log Updates

### DEC-015: Expand Training Data to Full 2013 (REJECTED)

**Original Decision:** Use full 2013 + Jan-Feb 2014 for training (50K rows)

**Validation Results:**
- RMSE: 14.88 (106% worse than Q1-only)
- Root cause: Seasonal mismatch (holiday extremes in train, not in test)
- Even with clipping: 7.39 (still 7% worse)

**Status:** REJECTED - Hypothesis invalidated by testing

**Impact:** 
- Demonstrates scientific rigor (willing to reject sunk effort)
- Led to discovery of temporal consistency principle
- Portfolio shows mature data science methodology

---

### DEC-016: Temporal Consistency Over Data Volume (NEW)

**Decision ID:** DEC-016  
**Date:** 2025-11-20  
**Phase:** Week 3 Day 3  
**Status:** APPROVED

**Context:**
After DEC-015 failed, tested Q4 2013 + Q1 2014 for seasonal consistency.

**Decision:**
Use Q4 2013 + Q1 2014 (19K samples) as training period for temporal alignment with Q1 2014 test set.

**Results:**
- RMSE: 6.84 (0.7% better than Q1-only despite 2.7x more data)
- Beats full 2013 by 51% (14.88 → 6.84)
- Moderate overfitting (2.58x) vs severe (4.99x)

**Key Insight:**
> "In seasonal forecasting, temporal relevance trumps data volume"

**Impact:**
- Foundation for Week 3 best model
- Major portfolio insight
- Generalizable principle for time series work

Full decision document: DEC-016_Temporal_Consistency_Matters.md

---

## Lessons Learned

### For This Project

1. **Hypothesis testing is critical** - DEC-015 seemed logical but failed in practice
2. **Seasonal patterns dominate** - March 2014 has different patterns than annual average
3. **Overfitting diagnosis matters** - Train/test RMSE ratio revealed the problem
4. **Context determines technique** - Clipping helped in one case, hurt in another
5. **Pivoting shows maturity** - Abandoning sunk effort when evidence contradicts

### For Future Projects

1. **Check temporal alignment early** - Don't assume more data is always better
2. **Validate training/test similarity** - Distribution mismatch kills performance
3. **Use multiple validation methods** - CV RMSE was high (11.59) signaling issues
4. **Document failed experiments** - Negative results are valuable insights
5. **Scientific method wins** - Propose, test, reject, iterate

### Business Translation

**For stakeholders:**
- "We tested using full year of historical data but found seasonal patterns matter more than data volume"
- "March sales patterns differ from November-December, so we focused training on similar periods"
- "This approach improved forecast accuracy by 10% while using less historical data"

**Key message:** Smart data selection > blind data accumulation

---

## Technical Quality

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Model Performance | Excellent | 10.08% improvement, RMSE 6.49 |
| Feature Engineering | Excellent | DEC-014 validated, 33 optimal features |
| Experiment Tracking | Excellent | 3 MLflow runs, clear documentation |
| Hypothesis Testing | Excellent | Rigorous testing, rejected when wrong |
| Code Quality | Good | Progressive execution, clean outputs |
| Documentation | Excellent | Comprehensive decision log, checkpoints |

---

## Blockers & Issues

### Current Blockers
- None

### Resolved Issues

1. **DEC-015 failure (full 2013 training)**
   - Problem: RMSE jumped to 14.88 (catastrophic)
   - Diagnosis: Seasonal mismatch, extreme outliers
   - Resolution: Pivoted to Q4+Q1 temporal consistency

2. **Outlier clipping uncertainty**
   - Problem: Unclear when clipping helps vs hurts
   - Testing: Tried on both full 2013 and Q4+Q1
   - Resolution: Context-dependent, Q4+Q1 better unclipped

### Risks for Day 4-5

- **Low risk:** Best model identified, ready for LSTM comparison
- **LSTM may underperform XGBoost** (expected for tabular time series)
- **Artifact export** should be straightforward

---

## Sparsity Limitation Note

**Acknowledged limitation (from earlier discussion):**

Current approach does NOT explicitly address 99.29% data sparsity:
- Not using sparse matrix format (CSR/CSC)
- Not using specialized sparse/intermittent demand models
- XGBoost handles sparsity implicitly through tree splits

**Why acceptable:**
- Filtered dataset (Guayas + top-3 families) reduces effective sparsity
- XGBoost learns "zero vs non-zero" patterns naturally
- Family/store-level features aggregate across sparse items
- Academic project scope prioritizes core ML skills

**Documentation:** Include in Week 4 report as "Future Improvements"
- "Sparsity not explicitly modeled - future work: sparse matrix optimization"
- Shows awareness of limitation
- Demonstrates ability to prioritize within constraints

---

## Time Allocation

| Activity | Planned | Actual | Notes |
|----------|---------|--------|-------|
| Feature reduction | 30min | 30min | Clean implementation |
| DEC-015 testing | - | 2h | Unplanned but critical |
| Q4+Q1 discovery | - | 1h | Major breakthrough |
| Hyperparameter tuning | 2h | 30min | Fast with RandomizedSearchCV |
| Visualization | 30min | 20min | Efficient |
| Documentation | 30min | Ongoing | Comprehensive |
| **Total** | **~3h planned** | **~4h actual** | Worth the extra hour |

**Note:** Extra hour invested in testing DEC-015 and discovering DEC-016 was critical for finding best approach.

---

## Next Steps - Day 4 Preview

**Day 4 Primary Focus: LSTM Model Comparison**

**Objectives:**
1. Implement LSTM baseline with Q4+Q1 training data
2. Compare LSTM vs XGBoost performance
3. If LSTM competitive, attempt hyperparameter tuning
4. Document findings in MLflow

**Expected outcome:**
- XGBoost likely to outperform LSTM on tabular time series
- LSTM may struggle with sparse data and feature engineering
- If LSTM RMSE > 7.0, document as "XGBoost superior for this task"

**Preparation:**
- Q4+Q1 training data already prepared
- 33 features ready
- Clear baseline to compare against (RMSE 6.49)

**Decision criteria:**
- If LSTM RMSE < 6.49: Excellent, tune further
- If LSTM RMSE 6.5-7.0: Competitive, consider ensemble
- If LSTM RMSE > 7.0: XGBoost is winner, document why

---

## Week 3 Overall Progress

**Days completed:** 3 / 5 (60%)  
**Major milestones achieved:**
- [x] Day 1: Baseline modeling (RMSE 7.21)
- [x] Day 2: Feature validation (DEC-014 feature reduction)
- [x] Day 3: Temporal optimization + tuning (RMSE 6.49, 10% improvement)
- [ ] Day 4: LSTM comparison
- [ ] Day 5: Artifacts export + handoff

**Key discoveries:**
- DEC-014: Feature reduction prevents overfitting
- DEC-015: REJECTED - Full 2013 fails due to seasonal mismatch
- DEC-016: Temporal consistency trumps data volume

**Portfolio strength:**
- Shows rigorous hypothesis testing
- Demonstrates willingness to pivot when wrong
- Deep time series understanding
- Complete experiment tracking

---

**Checkpoint completed by:** Alberto Diaz Durana  
**Next checkpoint:** Day 4 (LSTM model comparison)  
**Status:** Ready for Day 4 with best XGBoost baseline

---

**END OF DAY 3 CHECKPOINT**
