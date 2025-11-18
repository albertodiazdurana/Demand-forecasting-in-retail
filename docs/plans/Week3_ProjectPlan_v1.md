# Corporación Favorita Grocery Sales Forecasting
## Week 3 Project Plan: Modeling & Analysis

**Prepared by:** Alberto Diaz Durana  
**Timeline:** Week 3 (5 working days)  
**Phase:** Phase 3 - Modeling & Analysis  
**Previous Phase:** Week 2 - Feature Engineering (COMPLETE, 29 features)  
**Next Phase:** Week 4 - Communication & Deployment  
**Plan Version:** 1.0

---

## 1. Purpose

**Objective:**  
Build, evaluate, and optimize forecasting models using the feature-engineered dataset from Week 2. Establish baseline performance, tune hyperparameters, validate feature importance, and track experiments systematically with MLflow.

**Business Value:**  
- Quantify forecast accuracy (MAE, RMSE, MAPE) for inventory planning decisions
- Identify most predictive features for future data collection priorities
- Compare XGBoost vs LSTM approaches for Favorita retail context
- Create reproducible experiment tracking for model governance
- Generate preprocessing artifacts for Week 4 deployment

**Resources:**
- Starting dataset: w02_d05_FE_final.pkl (300,896 × 57 columns, 110.4 MB)
- Accumulated buffer: 21.5 hours from Weeks 1-2
- Computing: XGBoost trains in minutes, LSTM may require 30-60 min

**Week 2 Foundation:**
- 29 engineered features: lags (4), rolling stats (6), oil (5), aggregations (8), promotions (6)
- All features validated (no data leakage, temporal order preserved)
- Strong autocorrelation captured (lag7: 0.40, lag14: 0.32)
- Quality status: Ready for modeling

---

## 2. Scope & Requirements

### Course Requirements (Week 3)
1. **Evaluate XGBoost baseline** with comprehensive metrics (MAE, RMSE, Bias, MAD, rMAD, MAPE)
2. **Set up MLflow** experiment tracking
3. **Log baseline run** (params, metrics, forecast plot)
4. **Tune XGBoost** hyperparameters (GridSearchCV or RandomizedSearchCV)
5. **Retrain and log best model** (second MLflow run)
6. **Optional: LSTM model** baseline + tuning
7. **Save preprocessing artifacts** (Scaler, feature columns) for Week 4 deployment
8. **GitHub commit** updated notebook

### Additional Requirements (Your Specifications)
- **Feature Validation Methods:** Permutation importance, SHAP values, ablation studies
- **Model interpretability:** Business-friendly explanations of predictions
- **Deployment readiness:** Artifacts properly packaged for Streamlit app

### Deliverables
- [ ] w03_d01_MODEL_baseline.ipynb (XGBoost baseline + evaluation)
- [ ] w03_d02_MODEL_mlflow-features.ipynb (MLflow + feature validation)
- [ ] w03_d03_MODEL_tuning.ipynb (Hyperparameter optimization)
- [ ] w03_d04_MODEL_lstm.ipynb (Optional LSTM comparison)
- [ ] w03_d05_MODEL_artifacts-export.ipynb (Preprocessing artifacts + final checks)
- [ ] MLflow experiment logs (2+ runs tracked)
- [ ] Preprocessing artifacts: scaler.pkl, feature_columns.json
- [ ] Week 3 summary checkpoint document
- [ ] Week3_to_Week4_Handoff.md
- [ ] GitHub commit with all updates

---

## 3. Daily Breakdown

### Day 1: Data Preparation & Baseline Modeling
**Goal:** Establish XGBoost baseline with comprehensive evaluation

**Notebook:** w03_d01_MODEL_baseline.ipynb

**Activities:**

**Part 0: Data Loading & Train/Test Split (45 min)**
- Load w02_d05_FE_final.pkl
- Verify temporal order (store_nbr, item_nbr, date ascending)
- Chronological split: Jan-Feb 2014 (train), March 2014 (test)
- NO random shuffling (time series principle)
- Document split strategy and dates

**Part 1: Feature & Target Separation (30 min)**
- Define feature columns (exclude: date, store_nbr, item_nbr, unit_sales)
- Target: unit_sales
- Create X_train, X_test, y_train, y_test
- Verify shapes and no data leakage

**Part 2: XGBoost Baseline Training (45 min)**
- Initialize XGBRegressor with default params
- Set random_state=42 for reproducibility
- Train on X_train, y_train
- Document training time and convergence

**Part 3: Comprehensive Evaluation (1.5 hours)**
- Predict on X_test
- Calculate metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Bias (mean(y_pred - y_test))
  - MAD (Median Absolute Deviation)
  - rMAD (relative MAD)
  - MAPE (Mean Absolute Percentage Error)
- Create visualizations:
  - Actual vs Predicted scatter plot
  - Residuals distribution histogram
  - Time series plot (y_test vs y_pred by date)
  - Error analysis by store/family
- Interpret results: Over/under-forecasting patterns?

**Deliverables:**
- Baseline model trained and evaluated
- 6 evaluation metrics computed
- 4+ visualizations created
- Initial insights documented

**Success Criteria:**
- [ ] Train/test split preserves temporal order
- [ ] No data leakage detected
- [ ] All 6 metrics calculated
- [ ] Visualizations clearly show model performance
- [ ] Baseline RMSE documented for comparison

---

### Day 2: MLflow Setup & Feature Validation
**Goal:** Set up experiment tracking and validate feature importance

**Notebook:** w03_d02_MODEL_mlflow-features.ipynb

**Activities:**

**Part 0: MLflow Setup (45 min)**
- Install MLflow: `pip install mlflow`
- Set experiment: `mlflow.set_experiment("favorita-forecasting")`
- Configure tracking URI (local or remote)
- Test logging with simple example
- Document MLflow UI access

**Part 1: Baseline Run Logging (45 min)**
- Start MLflow run
- Log baseline model parameters:
  - n_estimators, max_depth, learning_rate, etc.
- Log evaluation metrics:
  - MAE, RMSE, Bias, MAD, rMAD, MAPE
- Log artifacts:
  - Actual vs Predicted plot
  - Residuals plot
  - Feature importance plot (XGBoost built-in)
- End run
- Verify in MLflow UI

**Part 2: Permutation Importance (1 hour)**
- Compute permutation importance using sklearn
- Identify top 15 most important features
- Visualize importance scores (bar plot)
- Compare to XGBoost built-in importance
- Document findings: Which features matter most?

**Part 3: SHAP Values Analysis (1.5 hours)**
- Install shap: `pip install shap`
- Compute SHAP values for sample predictions
- Create SHAP summary plot (feature impact)
- Create SHAP dependence plots for top 5 features
- Interpret: How do features influence predictions?
- Business translation: What drives sales forecasts?

**Part 4: Ablation Study (45 min)**
- Retrain model without lag features → measure performance drop
- Retrain model without rolling features → measure performance drop
- Retrain model without oil features → measure performance drop
- Compare RMSE across ablation experiments
- Conclusion: Which feature groups are critical?

**Deliverables:**
- MLflow experiment configured and tested
- Baseline run logged with params, metrics, plots
- Permutation importance computed and visualized
- SHAP analysis completed with 5+ plots
- Ablation study results documented
- Feature validation report (markdown section in notebook)

**Success Criteria:**
- [ ] MLflow UI accessible and shows baseline run
- [ ] All metrics logged correctly
- [ ] Top 15 features identified via permutation importance
- [ ] SHAP summary plot clearly shows feature contributions
- [ ] Ablation study quantifies feature group importance
- [ ] Business interpretation provided for top features

---

### Day 3: Hyperparameter Tuning & Best Model
**Goal:** Optimize XGBoost and log best configuration

**Notebook:** w03_d03_MODEL_tuning.ipynb

**Activities:**

**Part 0: Hyperparameter Search Strategy (30 min)**
- Define parameter grid (constrained for speed):
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 5, 7]
  - learning_rate: [0.01, 0.05, 0.1]
  - subsample: [0.8, 1.0]
  - colsample_bytree: [0.8, 1.0]
- Choose search method: RandomizedSearchCV (faster) or GridSearchCV
- Set cv=3 (time series split: expanding window)
- Define scoring metric: neg_root_mean_squared_error

**Part 1: Hyperparameter Search Execution (1-2 hours)**
- Run RandomizedSearchCV with n_iter=20 (or GridSearchCV)
- Monitor progress (may take time - good break opportunity)
- Extract best parameters
- Document search time and iterations

**Part 2: Best Model Training & Evaluation (1 hour)**
- Retrain XGBoost with best_params on full training set
- Predict on test set
- Calculate all 6 metrics (compare to baseline)
- Visualize: Baseline vs Tuned performance comparison
- Compute improvement percentages

**Part 3: MLflow Best Model Logging (45 min)**
- Start new MLflow run (run_name="xgboost_tuned")
- Log best parameters
- Log tuned model metrics
- Log comparison plots (baseline vs tuned)
- Log model artifact (save trained model)
- End run
- Verify both runs visible in MLflow UI

**Deliverables:**
- Hyperparameter search completed
- Best parameters identified
- Tuned model trained and evaluated
- Performance improvement quantified
- Second MLflow run logged
- Comparison analysis documented

**Success Criteria:**
- [ ] Hyperparameter search completes successfully
- [ ] Best model shows improvement over baseline (RMSE reduction)
- [ ] All tuned metrics logged to MLflow
- [ ] Side-by-side comparison shows clear improvement
- [ ] Model artifact saved for deployment

---

### Day 4: LSTM Model (Optional)
**Goal:** Build and evaluate LSTM baseline, compare to XGBoost

**Notebook:** w03_d04_MODEL_lstm.ipynb

**Activities:**

**Part 0: Data Preparation for LSTM (1 hour)**
- Reshape data for LSTM: (samples, timesteps, features)
- Create sequences: Use past N days to predict next day
- Normalize features (StandardScaler or MinMaxScaler)
- Split into train/test preserving temporal order
- Document data shape transformations

**Part 1: LSTM Baseline Architecture (1 hour)**
- Import tensorflow/keras
- Define LSTM architecture:
  - LSTM layer (64-128 units)
  - Dropout (0.2-0.3)
  - Dense output layer
- Compile with optimizer (Adam), loss (MSE)
- Set callbacks: EarlyStopping, ModelCheckpoint

**Part 2: LSTM Training (30-60 min)**
- Train LSTM on prepared sequences
- Monitor training/validation loss
- Stop early if overfitting detected
- Document training time and epochs

**Part 3: LSTM Evaluation & Comparison (1 hour)**
- Predict on test set
- Inverse transform predictions (if normalized)
- Calculate same 6 metrics (MAE, RMSE, etc.)
- Create comparison table: XGBoost baseline vs XGBoost tuned vs LSTM
- Visualize: All three models on same plot
- Analysis: Which model performs best? Trade-offs?

**Part 4: Optional LSTM Tuning (if time permits)**
- Vary LSTM units, layers, dropout
- Log best LSTM run to MLflow
- Document LSTM-specific insights

**Deliverables:**
- LSTM model implemented and trained
- LSTM evaluation completed
- Three-way comparison: XGBoost baseline vs tuned vs LSTM
- MLflow run for LSTM (optional but recommended)
- Insights on model selection for Favorita forecasting

**Success Criteria:**
- [ ] LSTM trains without errors
- [ ] LSTM metrics comparable to XGBoost
- [ ] Clear comparison table shows all three models
- [ ] Recommendation documented: Best model for deployment?

**Note:** If LSTM training is slow or results are poor, document findings and proceed. XGBoost is sufficient for course requirements.

---

### Day 5: Artifacts Export & Week 3 Finalization
**Goal:** Package preprocessing artifacts, document findings, prepare Week 4 handoff

**Notebook:** w03_d05_MODEL_artifacts-export.ipynb

**Activities:**

**Part 0: Preprocessing Artifacts Export (1 hour)**
- Identify scaler used (if any) → save as scaler.pkl
- Document feature columns → save as feature_columns.json
- Save best model → best_model.pkl or best_model.json (XGBoost)
- Create artifacts/ directory structure
- Verify artifacts load correctly (test in clean environment)
- Document artifact usage instructions

**Part 1: Final Model Validation (1 hour)**
- Load best model from artifacts
- Reload test data
- Re-run predictions
- Verify metrics match previous results (reproducibility check)
- Document any discrepancies

**Part 2: Week 3 Summary Documentation (1 hour)**
- Create w03_d05_checkpoint.md:
  - Time spent summary
  - Models trained: XGBoost baseline, tuned, LSTM
  - Best model identified
  - Top 10 features by importance
  - Key findings and insights
  - Recommendations for Week 4
- Update feature_dictionary_v2.txt if needed

**Part 3: Week 3 → Week 4 Handoff Document (1.5 hours)**
- Create Week3_to_Week4_Handoff.md:
  - Executive summary
  - Model performance results
  - Artifacts location and usage
  - Deployment recommendations
  - Known limitations
  - Next steps for Streamlit app
  - MLflow experiment tracking summary
- Use Week2_to_Week3_Handoff.md as template

**Part 4: GitHub Commit (30 min)**
- Review all notebooks for cleanliness
- Add markdown documentation to cells
- Commit to GitHub with descriptive message:
  - "WEEK 3 COMPLETE: XGBoost tuning + LSTM comparison + MLflow tracking"
- Push changes
- Verify repository state

**Deliverables:**
- Preprocessing artifacts saved (scaler.pkl, feature_columns.json, best_model.pkl)
- Artifacts validated for reproducibility
- Week 3 checkpoint document
- Week3_to_Week4_Handoff.md
- GitHub repository updated
- Clean, documented notebooks ready for Week 4 consolidation

**Success Criteria:**
- [ ] All artifacts load and produce correct predictions
- [ ] Handoff document complete with clear next steps
- [ ] GitHub commit successful
- [ ] Week 3 deliverables checklist 100% complete
- [ ] Ready to start Week 4 (consolidation + deployment)

---

## 4. Success Criteria

### Quantitative

**Modeling Performance:**
- XGBoost baseline trained: RMSE baseline established
- XGBoost tuned: RMSE improvement over baseline (target: 5-15% reduction)
- LSTM evaluated: Performance compared to XGBoost
- All 6 metrics computed: MAE, RMSE, Bias, MAD, rMAD, MAPE
- Forecast accuracy: MAPE < 25% (retail standard)

**Feature Validation:**
- Permutation importance: Top 15 features identified
- SHAP analysis: 5+ dependence plots created
- Ablation study: 3+ feature groups tested
- Feature importance ranking documented

**Experiment Tracking:**
- MLflow experiments: 2+ runs logged (baseline + tuned)
- All params/metrics logged correctly
- Artifacts saved to MLflow
- MLflow UI accessible and functional

**Reproducibility:**
- Preprocessing artifacts saved and tested
- Model predictions reproducible from artifacts
- All notebooks run end-to-end without errors
- Random seeds set (random_state=42)

### Qualitative

**Model Interpretability:**
- Top features explained in business terms (e.g., "7-day lag captures weekly patterns")
- SHAP analysis provides clear feature impact direction
- Recommendations for data collection priorities based on feature importance
- Trade-offs between XGBoost and LSTM documented

**Documentation Quality:**
- Each notebook has clear markdown descriptions
- Decision rationale documented (e.g., why RandomizedSearchCV over GridSearchCV)
- Findings summarized in checkpoint documents
- Handoff document provides clear Week 4 starting point

**Deployment Readiness:**
- Artifacts properly packaged and validated
- Clear instructions for artifact usage
- Best model identified with confidence
- Known limitations documented

### Technical

**Code Quality:**
- All 5 notebooks execute without errors
- No data leakage (temporal order preserved)
- Proper train/test split (chronological, no shuffling)
- Reproducible results (seeds set, deterministic operations)
- Clean code with functions for repeated operations

**MLflow Integration:**
- Experiments properly organized
- Runs tagged with meaningful names
- Metrics and params logged consistently
- UI navigation documented for Week 4 reference

---

## 5. Phase 3 Readiness Checklist (for Week 4 Transition)

### Required Inputs (from Week 2)
- [x] Feature-engineered dataset (w02_d05_FE_final.pkl, 300,896 × 57 columns)
- [x] Feature dictionary v2 (feature_dictionary_v2_clean.txt)
- [x] Week 2 insights (29 features validated, no data leakage)

### Completion Criteria

**MUST Complete (Core Requirements):**
- [ ] XGBoost baseline trained and evaluated (6 metrics)
- [ ] MLflow setup complete (experiment tracking functional)
- [ ] Baseline run logged to MLflow (params + metrics + plots)
- [ ] Hyperparameter tuning completed (best params identified)
- [ ] Tuned model logged to MLflow (second run)
- [ ] Preprocessing artifacts saved (scaler.pkl, feature_columns.json)
- [ ] GitHub commit with updated notebooks

**SHOULD Complete (High Value):**
- [ ] Feature validation methods: permutation importance + SHAP
- [ ] Ablation study (3 feature groups tested)
- [ ] LSTM baseline trained and compared to XGBoost
- [ ] Week 3 summary checkpoint document
- [ ] Week3_to_Week4_Handoff.md created

**COULD Complete (Bonus):**
- [ ] LSTM hyperparameter tuning
- [ ] Advanced SHAP visualizations (force plots, waterfall plots)
- [ ] Cross-validation with time series split
- [ ] Prophet model comparison

### Quality Checks
- [ ] No data leakage in train/test split (verified manually)
- [ ] Temporal order preserved throughout pipeline
- [ ] All metrics calculated correctly (spot-check predictions)
- [ ] MLflow UI shows all expected runs
- [ ] Artifacts load successfully in clean environment
- [ ] Feature importance aligns with Week 1-2 insights (lag7, weekend, promo)
- [ ] Best model selected based on test set performance

### Deliverables Ready
- [ ] w03_d01_MODEL_baseline.ipynb (runs without errors)
- [ ] w03_d02_MODEL_mlflow-features.ipynb (runs without errors)
- [ ] w03_d03_MODEL_tuning.ipynb (runs without errors)
- [ ] w03_d04_MODEL_lstm.ipynb (optional, runs without errors)
- [ ] w03_d05_MODEL_artifacts-export.ipynb (runs without errors)
- [ ] artifacts/scaler.pkl (saved and tested)
- [ ] artifacts/feature_columns.json (saved and tested)
- [ ] artifacts/best_model.pkl (saved and tested)
- [ ] MLflow experiment logs (accessible via UI)
- [ ] w03_d05_checkpoint.md (Week 3 summary)
- [ ] Week3_to_Week4_Handoff.md (transition document)
- [ ] GitHub repository updated with Week 3 work

### Next Phase Readiness
After completing Week 3, you will have:
- Best forecasting model identified (XGBoost tuned or LSTM)
- Complete experiment tracking in MLflow (reproducible results)
- Deployment-ready artifacts (scaler, features, model)
- Feature importance insights for stakeholder communication
- Performance benchmarks for presentation (RMSE, MAPE, etc.)
- Clean notebooks ready for Week 4 consolidation
- Documented model limitations and recommendations

---

## 6. Communication Plan

### Daily Progress Updates
- **Frequency:** End of each day
- **Format:** Brief summary (5 minutes)
- **Content:** Models trained, metrics achieved, findings, blockers
- **Audience:** Self (tracking), advisor (if requested)

### Mid-Week Check-In
- **Timing:** Day 3 (after tuning complete)
- **Content:** XGBoost performance, feature importance insights, LSTM progress
- **Audience:** Advisor

### Week-End Summary
- **Timing:** Day 5 end-of-day
- **Format:** Week 3 checkpoint document (~2-3 pages)
- **Content:** 
  - Models trained summary
  - Best model recommendation
  - Feature validation findings
  - Week 4 preview (consolidation + Streamlit)
- **Audience:** Advisor, peer reviewers

### Week 3 → Week 4 Handoff
- **Timing:** Day 5 evening
- **Format:** Handoff document (similar to Week2_to_Week3_Handoff.md)
- **Content:**
  - Model artifacts location
  - Performance summary
  - Deployment recommendations
  - Known limitations
  - Week 4 tasks preview
- **Audience:** Next session (self), advisor

---

## 7. Risk Management

### Identified Risks

**Risk 1: Hyperparameter search takes too long**
- Likelihood: Medium
- Impact: Medium (delays Day 3)
- Mitigation: Use RandomizedSearchCV with n_iter=20 (vs full GridSearchCV)
- Contingency: Reduce parameter grid, use default params + manual tuning

**Risk 2: LSTM training is slow or unstable**
- Likelihood: Medium
- Impact: Low (optional task)
- Mitigation: Use smaller architecture (64 units), set max_epochs=50
- Contingency: Skip LSTM if Week 3 running behind, focus on XGBoost

**Risk 3: MLflow setup issues (environment conflicts)**
- Likelihood: Low
- Impact: Medium (blocks experiment tracking)
- Mitigation: Use pip install in isolated environment, test early on Day 2
- Contingency: Use manual logging (save metrics to CSV) if MLflow fails

**Risk 4: Artifacts don't load correctly in clean environment**
- Likelihood: Low
- Impact: High (blocks Week 4 deployment)
- Mitigation: Test artifacts in new notebook cell on Day 5
- Contingency: Debug serialization, use joblib/pickle alternatives

**Risk 5: Feature importance doesn't align with expectations**
- Likelihood: Low
- Impact: Low (surprising but valuable finding)
- Mitigation: No mitigation needed - document and investigate
- Contingency: N/A (not a blocker, just unexpected insight)

---

## 8. Decision Log Updates (Expected)

### DEC-013: Train/Test Split Strategy
- **Context:** Need to split data for model evaluation
- **Decision:** Jan-Feb 2014 (train), March 2014 (test), chronological split
- **Rationale:** Preserves temporal order, realistic forecast scenario
- **Impact:** No data leakage, valid performance estimates

### DEC-014: Hyperparameter Search Method
- **Context:** GridSearchCV vs RandomizedSearchCV trade-off
- **Decision:** RandomizedSearchCV with n_iter=20
- **Rationale:** 10x faster than full grid, sufficient for academic project
- **Impact:** Completes in 1-2 hours vs potential 10+ hours

### DEC-015: LSTM Architecture Choice
- **Context:** Many LSTM configurations possible
- **Decision:** Single LSTM layer (64 units) + dropout + dense
- **Rationale:** Simple baseline, interpretable, trains quickly
- **Impact:** Comparable to course examples, extendable if needed

### DEC-016: Best Model Selection Criteria
- **Context:** Multiple models trained (baseline, tuned, LSTM)
- **Decision:** Select based on test RMSE + business interpretability
- **Rationale:** RMSE aligns with business metric, interpretability aids adoption
- **Impact:** Clear recommendation for Week 4 deployment

---

## 9. Standards & Best Practices

### Code Standards
- Always print actual outputs (df.shape, metrics, not "Success!")
- Use docstrings for custom functions
- Set random_state=42 for reproducibility
- Save intermediate results (checkpoints)
- Clear markdown cells before each code section

### MLflow Standards
- Use descriptive run names (e.g., "xgboost_baseline", "xgboost_tuned")
- Log all hyperparameters (even defaults)
- Log both training and test metrics
- Save plots as artifacts (PNG format)
- Tag runs appropriately (e.g., "baseline", "tuned", "lstm")

### Visualization Standards
- Clear titles and axis labels
- Legend for multi-line plots
- Consistent color scheme across notebooks
- Save to outputs/figures/models/
- Reference in markdown cells

### Documentation Standards
- Markdown cell before each code cell (purpose + expected output)
- Interpret results after each visualization
- Business translation for technical findings
- Decision rationale documented inline
- No emojis, use text prefixes (OK:, WARNING:, ERROR:)

---

## 10. Final Notes

**Week 3 is about validation.** Week 1 explored data, Week 2 created features. Now we validate if those features actually improve forecasts. MLflow provides the discipline, feature validation provides the insights.

**Hyperparameter tuning is secondary.** A well-designed feature set with default params often outperforms a poorly-designed feature set with tuned params. Week 2's 29 features are the real work; tuning is polish.

**LSTM is optional for good reason.** XGBoost typically outperforms LSTM on tabular time series with engineered features. LSTM shines with raw sequences (e.g., text, audio). If LSTM underperforms XGBoost, that's a valuable finding, not a failure.

**Feature validation is critical for Week 4.** The presentation needs "Top 5 features that drive sales forecasts." SHAP provides the story. Permutation importance provides the proof. Ablation study provides the business case ("Without lag features, accuracy drops 15%").

**Artifacts are deployment prerequisites.** Week 4 Streamlit app needs scaler.pkl, feature_columns.json, best_model.pkl. No artifacts = no deployment. Test them religiously on Day 5.

**You're in an excellent position.** 21.5 hour buffer, clean dataset, validated features. Week 3 requirements are straightforward: train, tune, track, export. Focus on quality over speed.

**Trust the process.** Week 1 explored, Week 2 engineered, Week 3 validates, Week 4 communicates. You're exactly where you should be.

---

**WEEK 3 OBJECTIVES CLEAR. READY TO EXECUTE.**

**Good luck, Alberto!**

---

**Document version:** 1.0  
**Last updated:** 2025-11-18  
**Next review:** After Week 3 completion  

---

**End of Week 3 Project Plan**
