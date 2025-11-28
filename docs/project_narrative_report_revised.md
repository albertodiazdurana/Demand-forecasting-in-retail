# Corporación Favorita Grocery Sales Forecasting
## From Exploration to Production: A Complete Forecasting Pipeline

**Author:** Alberto Diaz Durana  
**Project Duration:** 4 Weeks (November 2025)  
**Live Application:** [demand-forecasting-in-retail-app.streamlit.app](https://demand-forecasting-in-retail-app.streamlit.app/)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Week 1: Discovery Through Discipline](#week-1-discovery-through-discipline)
3. [Week 2: Engineering Features with Purpose](#week-2-engineering-features-with-purpose)
4. [Week 3: Models, Experiments, and the Courage to Reject](#week-3-models-experiments-and-the-courage-to-reject)
5. [Full Pipeline: From Sample to Scale](#full-pipeline-from-sample-to-scale)
6. [WSL2 GPU Setup: Enabling Production Scale](#wsl2-gpu-setup-enabling-production-scale)
7. [MLflow and Streamlit: From Experiment to Application](#mlflow-and-streamlit-from-experiment-to-application)
8. [Production Hardening: Tests and Code Quality](#production-hardening-tests-and-code-quality)
9. [Reflections](#reflections)
10. [Conclusions](#conclusions)

---

# Introduction

Corporación Favorita operates over 200 grocery stores across Ecuador, managing inventory for thousands of products daily. The business challenge is straightforward but consequential: predict tomorrow's sales accurately enough to stock shelves without waste. Overestimate, and perishables spoil. Underestimate, and customers leave empty-handed. For a retailer processing millions of transactions, even small improvements in forecast accuracy translate to significant operational savings.

This project addresses that challenge using four years of historical sales data from a Kaggle competition. The dataset contains 125 million transactions across 54 stores and 4,000+ products, supplemented by oil prices, holiday calendars, and promotional flags. The objective: build a production-ready forecasting system for the Guayas region, Ecuador's commercial center, capable of generating daily unit sales predictions through an interactive web application.

The project progressed through four phases. Week 1 explored the data's structure and quality, revealing that retail sparsity—99.1% of potential store-item-date combinations having no records—represents normal business reality rather than missing data. Week 2 transformed these insights into 33 predictive features capturing temporal patterns, promotional effects, and macroeconomic indicators. Week 3 tested multiple models and discovered that temporal consistency in training data matters more than data volume—a finding that emerged only after a failed hypothesis forced deeper investigation. The final phase scaled the pipeline to 4.8 million rows using GPU acceleration and deployed a live forecasting application.

The result is a deployed XGBoost model achieving RMSE 6.40 on the test period, accessible through an interactive Streamlit interface where users can select stores and items, generate single or multi-day forecasts, and download predictions. This document traces that journey from raw data to deployed application.

---

# Week 1: Discovery Through Discipline

## Establishing Scope and Structure

The Corporación Favorita dataset presented an immediate scale challenge. The training file alone contained 125 million rows spanning January 2013 through August 2017. Processing this volume for exploratory analysis would be computationally prohibitive and analytically unnecessary. The first decision was strategic filtering: focus on Guayas, Ecuador's most populous region and commercial hub, representing the highest transaction volume and business impact.

Within Guayas, further scoping was required for development efficiency. Three product families—GROCERY I, BEVERAGES, and CLEANING—were selected based on item count representation (58.4% of all items) and business relevance (high-volume categories where forecast accuracy matters most). A 300,000-row sample enabled rapid iteration during exploration while maintaining statistical validity. This scoping was documented in decision logs DEC-001 and DEC-002, establishing traceability from the project's first day.

The week's structure followed a one-notebook-per-day approach: data inventory, filtering and loading, quality assessment, temporal pattern analysis, and context export. Each notebook had a single clear objective, preventing scope creep and ensuring completeness before progression.

## The Sparsity Discovery

Day 3's quality assessment revealed the dataset's most important characteristic. Computing the potential combinations of stores, items, and dates against actual records showed that only 0.9% of theoretical combinations existed in the data. Initial reaction suggested data quality concerns—was the dataset incomplete?

Investigation resolved this apparent problem. The calculation was straightforward: 10 Guayas stores × 2,638 items × 1,248 possible days yielded 32.9 million potential records. The actual dataset contained 300,896 records. The ratio (0.91%) meant 99.1% sparsity. But retail reality explains this pattern: most items don't sell every day at every store. A specialty product might sell weekly at one location but monthly at another. Attempting to fill these gaps with zeros would create 33 million rows—a 110x expansion that would be computationally catastrophic and conceptually wrong (absence of a sale record differs from a recorded zero sale).

This finding, documented as DEC-005, shaped all subsequent work. Feature engineering in Week 2 needed to handle sparse lag calculations gracefully. Model selection in Week 3 favored algorithms with native missing value support. The deployed application needed to accommodate items with intermittent sales history.

## Quantifying Data Quality

The three-method outlier detection implemented on Day 3 exemplified validation through triangulation. Rather than trusting any single statistical approach, three independent methods were applied to the same data:

The IQR method flagged values beyond 1.5 times the interquartile range from Q1 or Q3. For the sales distribution, this identified 12,847 samples (4.27%) as potential outliers on the high end. The Z-score method, using a threshold of 3 standard deviations from the mean, flagged 3,156 samples (1.05%). Isolation Forest, an unsupervised anomaly detection algorithm with contamination parameter set to 0.01, identified 3,009 samples (1.00%).

The intersection of all three methods—samples flagged by IQR, Z-score, and Isolation Forest simultaneously—contained only 846 records (0.28%). These high-confidence outliers weren't errors but legitimate business events: promotional spikes, bulk purchases for events, seasonal peaks. The decision (DEC-004) was to flag but retain these values, preserving signal that models should learn to predict.

## Temporal Patterns and Autocorrelation

Days 4 and 5 quantified the temporal structure that would drive feature engineering. Weekend effects showed a 33.9% lift in average daily sales compared to weekdays—a pattern consistent across product families and stores. Payday effects, measured on the first of each month, showed a 10.7% boost. December sales exceeded the annual average by 30.4%, driven by holiday shopping.

Autocorrelation analysis provided the statistical foundation for lag feature selection. The correlation between today's sales and yesterday's was 0.60—strong enough to justify a lag-1 feature. Weekly patterns showed lag-7 correlation of 0.59, nearly as strong. Lag-14 reached 0.63, suggesting a biweekly cycle perhaps tied to payroll schedules. By lag-30, correlation dropped to 0.36, still meaningful but diminishing. These values directly informed Week 2's lag feature decisions: include lags at 1, 7, 14, and 30 days.

Store-level analysis revealed substantial performance variation. Store 51 recorded 356,000 units sold across the analysis period, while Store 32 recorded only 84,000—a 4.25x gap. This heterogeneity suggested that store-level aggregation features might capture systematic differences in customer volume, product assortment, or operational efficiency.

## Week 1 Summary

By the end of Week 1, the exploratory phase had produced actionable intelligence: strong autocorrelation validating lag features, weekend and payday effects justifying calendar features, store heterogeneity motivating aggregation features, and the sparsity insight preventing a potentially catastrophic data expansion decision. Ten decisions were documented (DEC-001 through DEC-010), each with context, rationale, and impact. The handoff to Week 2 was clean: a validated 300,896-row dataset ready for feature engineering.

---

# Week 2: Engineering Features with Purpose

## From Insights to Implementation

Week 2 transformed Week 1's discoveries into predictive features. The handoff document had prioritized four categories: lag features (validated by autocorrelation analysis), rolling statistics (to smooth sparse data), calendar features (to capture weekend and payday effects), and aggregations (to leverage store and item-level patterns). Each day's work connected explicitly to these priorities.

The feature engineering approach respected a critical constraint: temporal causality. Every feature must use only information available at prediction time. A model predicting tomorrow's sales cannot use tomorrow's oil price or next week's promotion flag. This seemingly obvious principle has subtle implications for rolling statistics and aggregations, requiring careful implementation to prevent data leakage.

## Lag Features and Sparse Data Handling

Day 1 confronted the interaction between lag features and data sparsity. When calculating `unit_sales_lag7` (sales from seven days ago), what happens when that date has no record? Three approaches were considered: impute zero (assume no sales), impute the item's mean (assume typical demand), or preserve the missing value.

The decision (DEC-011) was to preserve NaN values in lag features. The reasoning involved understanding the downstream model: XGBoost, a tree-based algorithm, handles missing values natively through surrogate splits. When a tree encounters a missing value during splitting, it learns which child node produces better predictions for that case. Imputing zeros would introduce bias—assuming zero demand when the historical record simply doesn't exist is not the same as knowing demand was zero. Imputing means would smooth legitimate variation. Preserving NaN allowed the model to learn from missingness as a signal.

This decision affected 27,000 to 40,000 values per lag feature (9-13% of the dataset). The implementation used pandas' shift function with SQL-like window semantics, grouping by store-item combinations to ensure lags respected product identity.

## Rolling Statistics and the min_periods Decision

Rolling means and standard deviations presented a related challenge. Computing a 7-day rolling average requires seven days of history. For items with sparse sales, early observations might have insufficient history, producing NaN or unstable estimates.

The `min_periods` parameter controls this tradeoff. Setting `min_periods=1` produces an estimate even with a single observation—stable but potentially misleading. Setting `min_periods=7` requires a full window—accurate but excluding early observations. The implementation chose `min_periods=1` with documentation that early estimates might be less reliable. The model would learn to weight these features appropriately through training.

Three rolling windows were implemented (7, 14, and 30 days), each producing mean and standard deviation features. The standard deviations aimed to capture volatility—items with stable demand versus items with high variance might require different safety stock levels.

## The Oil Price Paradox

Day 3 introduced external economic data. Ecuador's economy depends heavily on oil exports, so oil prices might influence consumer purchasing power and retail demand. The correlation analysis produced a paradox: at the aggregate daily level (total sales across all stores and items versus daily oil price), correlation was -0.55, a moderate negative relationship suggesting higher oil prices associated with lower sales. But at the granular store-item-date level—the actual prediction target—correlation was only +0.01, essentially zero.

The decision (DEC-012) was to include oil features despite weak granular correlation. The reasoning: tree models can capture non-linear relationships invisible to linear correlation. Oil price might interact with specific product categories, store locations, or time periods in ways that aggregate correlation masks. Six oil features were engineered: current price, lagged values at 7, 14, and 30 days, and momentum indicators (price change over 7 and 14 days).

This decision was explicitly provisional—Week 3's ablation studies would validate or invalidate it through direct performance measurement. The methodology treated uncertainty as an opportunity for validation rather than a reason for arbitrary choices.

## Aggregation Features and Contextual Information

Day 4 built aggregation features capturing context beyond individual store-item observations. Store-level aggregations (average sales, median sales, standard deviation across all items) characterized location performance. Cluster-level aggregations (Corporación Favorita groups stores into clusters with similar characteristics) provided regional context. Item-level aggregations captured product popularity across all stores.

The implementation required careful temporal handling. Aggregations must use only historical data—computing a store's average sales using future observations would constitute leakage. The solution computed aggregations over the training period only, applying those fixed values to all observations.

Eleven aggregation features were created, including promotion interaction terms combining the binary promotion flag with store and item averages. The hypothesis was that promotions might have differential effects depending on the product's baseline popularity or the store's typical promotional responsiveness.

## Feature Validation and Week 2 Summary

Day 5 validated the complete feature set before handoff. The validation checklist confirmed no data leakage (all features used only past information), preserved temporal order (critical for time series), appropriate data types, and reasonable value ranges. The output—a 110.4 MB pickle file containing 300,896 rows and 45 features—was ready for modeling.

The feature dictionary documented each feature's definition, calculation method, and rationale. This documentation would prove valuable in Week 3 when ablation studies revealed that several features actually harmed performance—understanding what each feature represented enabled informed decisions about removal.

---

# Week 3: Models, Experiments, and the Courage to Reject

## Establishing the Baseline

Week 3 applied scientific method to model development. The approach was hypothesis-driven: propose configurations, test rigorously, accept or reject based on evidence. This methodology would lead to the project's most important discoveries—including the rejection of an initially promising hypothesis.

Day 1 established the baseline. XGBoost with default parameters and all 45 features achieved RMSE 7.21 on the March 2014 test set. The train/test split followed DEC-013: training on January-February 2014, testing on March 2014, with a 7-day gap (February 22-28) preventing data leakage from lag features. Without this gap, the model could use February 28's sales to predict March 1—information that wouldn't be available in production when forecasting the next day.

This baseline wasn't meant to be impressive. It was a reference point for measuring improvement. All subsequent experiments would be evaluated against RMSE 7.21.

## Ablation Studies: When Less Is More

Day 2 introduced systematic feature validation through ablation studies. The process was straightforward: remove feature groups, retrain the model, measure performance change. If removing features improves performance, those features are harming the model through overfitting or noise.

The results were counterintuitive. Removing rolling standard deviation features (3 features) improved RMSE by 3.82%, from 7.21 to 6.94. Removing oil features (6 features) improved RMSE by 3.14%, to 6.99. Removing promotion interaction features (3 features) had no impact—the model performed identically without them. Even aggregation features (12 features) showed negative value, with removal improving RMSE by 1.97%.

The pattern was clear: the model was overfitting with too many features. Three validation methods—permutation importance, SHAP values, and ablation—agreed on which features mattered. A single feature, `unit_sales_7d_avg` (7-day rolling average of sales), provided most of the predictive power, with importance 17 times higher than any other feature.

Decision DEC-014 formalized the reduction from 45 to 33 features. The removed features included all rolling standard deviations (redundant with rolling means), all oil features (the correlation paradox resolved—they were noise), and all promotion interactions (overfitting rather than generalizing). This simpler model achieved RMSE 6.89, a 4.5% improvement over the baseline.

DEC-012, which had included oil features provisionally pending validation, was formally invalidated. The original decision was reasonable given the available information—aggregate correlation existed. But rigorous testing proved the features harmful at the granular prediction level.

## The Failed Hypothesis: DEC-015

Day 3 began with an appealing hypothesis. The Q1-only training set contained only 7,050 samples—small by machine learning standards. What if expanding to full 2013 data (50,000+ samples) would improve performance? More data usually helps, and the calendar patterns (weekends, paydays, December lift) should be consistent across years.

DEC-015 documented this hypothesis and the test plan. The result was catastrophic. Training on January 2013 through February 2014 and testing on March 2014, the model achieved RMSE 14.88—a 106% degradation from the Q1-only baseline. Even with 99th percentile clipping to address extreme values, performance remained 7% worse than the smaller training set.

The investigation revealed the cause: seasonal mismatch. The expanded training set included November and December 2013, months dominated by holiday shopping with extreme promotional activity. Sales in those months reached 1,332 units for high-performing items. The March 2014 test period—a normal spring month—had maximum sales of 222 units, roughly 6x lower. The model learned holiday patterns and tried to apply them to non-holiday predictions, producing forecasts wildly inconsistent with actual demand.

DEC-015 was formally rejected. The hypothesis failed, but the investigation succeeded in revealing a fundamental principle: for seasonal data, temporal relevance matters more than data volume.

## The Pivot: DEC-016 and Temporal Consistency

From the DEC-015 failure emerged DEC-016. If seasonal mismatch was the problem, what training period would provide seasonal consistency with the March test period? The answer: Q4 2013 (October-December) plus Q1 2014 (January-February). This 5-month window captured fall and winter patterns temporally adjacent to the March test period, without introducing summer patterns from mid-2013.

The Q4+Q1 training set contained 18,905 samples—2.7 times more than Q1-only but without the seasonal mismatch of full 2013. Testing showed RMSE 6.84, slightly better than Q1-only despite having more data. More importantly, the overfitting ratio (train RMSE / test RMSE) improved from 4.99x with full 2013 to 2.58x with Q4+Q1.

With hyperparameter tuning—reducing max_depth to 3 to prevent overfitting, adjusting learning rate to 0.1—the final XGBoost model achieved RMSE 6.49. Total improvement from Day 1: 10.08%.

The principle established by DEC-016 extends beyond this project: when forecasting seasonal data, align training periods with the forecast target's seasonal characteristics. Data volume cannot compensate for distribution mismatch.

## The LSTM Surprise

Day 4 tested the assumption that XGBoost would dominate on tabular data. The LSTM architecture used identical features and train/test split: 64 units, dropout 0.2, early stopping with patience 10.

The result challenged conventional wisdom. LSTM achieved RMSE 6.26 versus XGBoost's 6.49—a 4.5% improvement. More striking was the generalization behavior. XGBoost's training RMSE was 2.51 versus test RMSE 6.49, an overfitting ratio of 2.58x. LSTM's training RMSE was 10.55 versus test RMSE 6.26, a ratio of 0.57x—the model slightly underfitted training data while generalizing better to the test set.

Why did LSTM win? Dropout regularization (randomly zeroing 20% of activations during training) proved more effective than tree-based constraints at preventing memorization. The sequential architecture may have captured temporal dependencies beyond the engineered features. Regardless of the mechanism, the empirical result was clear: for this dataset at this scale, LSTM outperformed XGBoost.

Day 5 exported production artifacts: the trained LSTM model, the fitted StandardScaler, the 33-feature column list, and configuration metadata. Week 3 ended with RMSE 6.26, a 13.28% improvement from the Day 1 baseline of 7.21.

---

# Full Pipeline: From Sample to Scale

## The Scale Test

Weeks 1-3 worked with a 300,000-row sample. The full Guayas dataset contained 4.8 million rows—16 times larger. Would findings hold at scale? The methodology required validation, not assumption.

Two production notebooks were created. FULL_01_data_to_features processed raw Kaggle CSVs through the complete feature engineering pipeline without sampling. All 32 product families were included (versus the top-3 in the sample), all 10 Guayas stores, all available dates. The output: full_featured_data.pkl at 1.3 GB, containing 4,801,160 rows with 33 features.

FULL_02_train_final_model applied the Week 3 approach to production data. DEC-016's training split (Q4 2013 + Q1 2014), DEC-014's 33 features, and DEC-013's 7-day gap were applied exactly as documented. Training samples: 3.8 million. Test samples: 818,000.

## The Scale Reversal

The production comparison inverted Week 3's finding. XGBoost achieved RMSE 6.40 on the full dataset. LSTM struggled to converge, eventually producing RMSE around 9.37—46% worse than XGBoost.

The model that won on 300K samples lost at 4.8M samples.

This wasn't failure—it was discovery. DEC-017 documented the finding: model selection is scale-dependent. LSTM's superior generalization on small data became irrelevant when XGBoost had sufficient samples to learn patterns without overfitting. With 3.8 million training observations, the tree model's tendency to memorize was mitigated by sheer data volume. Meanwhile, LSTM's regularization overhead—beneficial for small data—became a constraint preventing full capacity utilization.

The practical implication: always validate on production-scale data before committing to an architecture. Sample-based findings guide exploration but don't guarantee production performance.

## Production Artifacts

The production pipeline exported deployment-ready artifacts. The XGBoost model (2.1 MB) was trained on 3.8 million samples. The StandardScaler was fitted on production training data. Feature columns and configuration metadata enabled reproducible inference. Test RMSE of 6.40 represented an 11% improvement over the Day 1 baseline of 7.21.

These artifacts weren't endpoints but inputs to deployment. The model file would be loaded by the Streamlit application. The scaler would transform user-selected forecasts. The feature columns would ensure inference matched training. Production readiness meant more than training completion—it meant deployment enablement.

---

# WSL2 GPU Setup: Enabling Production Scale

## The Computational Challenge

Week 3's LSTM trained on CPU in approximately 36 seconds for 300K rows. Scaling linearly, 4.8M rows would require nearly 10 minutes per training run—acceptable for final production training but impractical for the iteration required during development. GPU acceleration was needed.

The development environment was Windows 11 with an NVIDIA Quadro T1000 GPU (4GB VRAM). TensorFlow GPU support on Windows has historically been problematic, with complex CUDA and cuDNN installation requirements. WSL2 (Windows Subsystem for Linux 2) offered a cleaner path: Linux-native TensorFlow with GPU passthrough from Windows.

## Installation and Configuration

WSL2 Ubuntu 22.04 was installed through the Windows Store. The NVIDIA drivers for WSL2 (version 570.176) were automatically configured through the Windows driver (573.57). The critical verification: `nvidia-smi` in WSL2 showing the GPU, and TensorFlow detecting the device.

The Python environment was configured with a virtual environment containing TensorFlow 2.20.0 installed via `pip install tensorflow[and-cuda]`. This package bundles CUDA libraries, avoiding manual CUDA toolkit installation. Verification command:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Challenges and Solutions

Initial GPU detection failed due to driver version mismatch. The solution was ensuring the Windows NVIDIA driver was current (570+), which automatically provides WSL2 compatibility.

Memory growth configuration was required for the 4GB VRAM device (2.2GB available to TensorFlow after system overhead). Without explicit configuration, TensorFlow attempts to allocate full GPU memory at startup, potentially failing or causing instability:

```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

Model serialization encountered format issues. The original .h5 format produced warnings about legacy compatibility. Switching to the native .keras format resolved serialization and improved loading reliability.

## The Ironic Outcome

After configuring GPU acceleration specifically for LSTM training at scale, XGBoost won the production comparison. The GPU setup wasn't wasted—it enabled rapid experimentation and confirmed the scale-dependent model selection finding. But the deployed production model runs on CPU, where XGBoost's inference speed makes GPU acceleration unnecessary.

This outcome reinforced the value of keeping architectural options open. Had the WSL2 setup been skipped under the assumption that "LSTM won, so we'll use LSTM," the scale reversal would have been a costly surprise rather than a documented finding.

---

# MLflow and Streamlit: From Experiment to Application

## MLflow: Experiment Memory

Across four weeks, 16 MLflow experiments accumulated—a complete record of the modeling journey. Each run captured parameters (feature counts, hyperparameters, training samples), metrics (RMSE, MAE, training time), and artifacts (trained models, visualizations, importance rankings).

The experiment history told the project's story. Runs 1-2 established the baseline with 45 features. Runs 3-6 conducted ablation studies, each removing a feature group and measuring impact. Run 7 tested the DEC-015 hypothesis (full 2013 training) and failed—this run is tagged `deprecated=true, deprecation_reason="seasonal_mismatch"`. Run 8 validated DEC-016's Q4+Q1 approach. Runs 10-12 compared XGBoost and LSTM on sample data. Runs 13-16 validated at production scale, with Run 16 becoming the deployed model.

The MLflow UI made comparison trivial. Sorting by RMSE immediately showed best performers. Filtering by tags isolated valid runs from deprecated experiments. Clicking any run revealed complete configuration—reproducibility was built into the process.

## Streamlit: Making Models Accessible

A model in a pickle file helps no one outside the data science team. The final deliverable was an interactive application where business users could generate forecasts without touching code.

The Streamlit application (https://demand-forecasting-in-retail-app.streamlit.app/) loads the production XGBoost model and exposes forecasting through an intuitive interface. Users select a store from 10 Guayas locations, then an item from that store's available products. They choose a forecast date and mode—single day or multi-day up to 30 days. The application displays historical sales, generates predictions, visualizes results on a time series plot, and enables CSV download.

The full 4.8M-row dataset exceeded Streamlit Cloud's memory limits. The solution was a representative sample: 20 store-item pairs (2 items per store—top seller plus medium seller) with at least 100 days of history each. This 1.1 MB sample enabled real forecasting while respecting deployment constraints.

Multi-day forecasts required autoregressive logic. The model uses lag features—yesterday's sales, 7-day average. For day 2+ predictions, these features must update with predicted values, not historical data. The `autoregressive_forecast()` function implements this: predict day 1, update lag features with the prediction, predict day 2, repeat. Without this logic, multi-day forecasts would show flat lines (same prediction repeated). With it, forecasts show realistic trajectories responding to their own predictions.

---

# Production Hardening: Tests and Code Quality

## Why Add Tests

Production code requires reliability guarantees beyond "it works on my machine." The Streamlit application would be accessed by users expecting consistent behavior. Tests provide that assurance and enable confident refactoring.

A test suite was implemented using pytest. The `tests/test_data_utils.py` module contains 15 tests covering data loading functions: `get_stores()` returns expected store IDs, `get_items_for_store()` filters correctly, `get_history()` returns proper date ranges, `generate_forecast_dates()` handles edge cases including month boundaries and leap years.

The `tests/test_model_utils.py` module contains 9 tests covering model functions: `load_feature_columns()` returns the correct 33 features in order, `load_config()` parses model metadata correctly, `predict()` produces outputs with expected shapes and reasonable value ranges.

Edge cases receive explicit coverage: empty data returns, invalid store IDs, dates outside the training range. These tests catch regressions before users encounter them.

## Code Quality

The ruff linter was applied to the codebase, automatically fixing style issues: unused imports, inconsistent formatting, line length violations. The result is cleaner, more maintainable code that follows Python conventions.

Running the test suite produces a clean report:

```bash
python -m pytest tests/ -v
# 24 tests passed
```

This test infrastructure enables future maintenance. When updating the model or modifying feature engineering, the test suite validates that existing functionality remains intact.

---

# Reflections

## Why Didn't I Identify Seasonality Earlier?

Week 1's EDA examined temporal patterns extensively—weekend effects, payday boosts, December lift, autocorrelation structure. Yet the analysis didn't explicitly compare training and test period distributions. The assumption that "more data helps" went unquestioned until DEC-015's catastrophic failure forced investigation.

In retrospect, Week 1 should have included train/test distribution comparison as standard time series EDA. Visualizing the target variable's distribution across proposed training and test periods would have revealed the November-December holiday extremes immediately. The lesson: for time series projects, seasonal decomposition and distribution alignment analysis belong in initial exploration, not as post-failure diagnostics.

## Sample vs. Production Scale Assumptions

LSTM won on 300K samples by 4.5%. The assumption that this finding would transfer to production scale seemed reasonable—surely a better model remains better with more data. The 4.8M-row comparison disproved this assumption. XGBoost won at scale by 46%.

The mechanism explains the reversal. LSTM's dropout regularization prevents overfitting on small datasets by randomly dropping connections during training. On small data, this generalization advantage outweighs capacity limitations. On large data, XGBoost has enough examples to learn patterns without overfitting, and its lack of regularization overhead becomes an advantage rather than a liability.

The lesson: model selection is conditional on data scale. An intermediate validation at 1M rows might have revealed the transition point. Future projects should include scale sensitivity analysis before committing to production architectures.

## Feature Engineering vs. Model Complexity

The initial assumption was that more features provide more information for models to exploit. Week 2's engineering produced 45 features representing every insight from Week 1's analysis. Week 3's ablation studies proved 12 of these features actively harmful—removing them improved performance.

Rolling standard deviations were redundant with rolling means (highly correlated, adding noise). Oil features, despite reasonable business intuition about macroeconomic effects, added noise at the granular prediction level. Promotion interactions overfit to training patterns without generalizing.

The lesson: systematic ablation studies should be standard practice, not optional validation. Simple models with clean features often outperform complex models with noisy features. The best feature count isn't "as many as possible" but "as few as sufficient."

## The Value of Documenting Failures

DEC-015's rejection—expanding training to full 2013—was a failure. The hypothesis seemed sound, the test was well-designed, and the outcome was catastrophic. The temptation was to quietly discard the result and move on.

Instead, DEC-015 was documented thoroughly: hypothesis, test procedure, results (RMSE 14.88, 106% degradation), root cause analysis (seasonal mismatch), and formal rejection. This documentation led directly to DEC-016, which established the temporal consistency principle that improved the final model.

Failure documentation often provides more value than success documentation. Successes confirm existing intuitions; failures challenge assumptions and force deeper understanding. A portfolio showing only successes suggests either limited experimentation or selective reporting. Documented failures with subsequent pivots demonstrate scientific thinking.

---

# Conclusions

## Technical Outcomes

The project delivered a production forecasting system. RMSE improved 11% from baseline (7.21) to production (6.40). The deployed XGBoost model processes 4.8 million transactions and generates forecasts through an interactive web interface. Users can select stores and items, generate single or multi-day predictions, visualize results, and download forecasts.

The technical stack spans the data science workflow: pandas for data manipulation, scikit-learn for preprocessing and tree models, TensorFlow for neural network experimentation, MLflow for experiment tracking, pytest for testing, and Streamlit for deployment. GPU acceleration through WSL2 enabled production-scale neural network training, though the final model runs efficiently on CPU.

## Key Decisions

Seven documented decisions shaped the project:

**DEC-011** (Keep NaN in lag features) preserved missing value semantics for tree model surrogate splits, avoiding imputation bias.

**DEC-012** (Include oil features) was later invalidated by DEC-014's ablation studies—demonstrating that documented hypotheses enable systematic validation.

**DEC-013** (7-day train/test gap) prevented data leakage that would have inflated apparent performance while degrading production accuracy.

**DEC-014** (33 features from 45) proved that complexity reduction improves generalization when features add noise rather than signal.

**DEC-015** (Expand to full 2013, REJECTED) established that hypothesis testing includes accepting negative results.

**DEC-016** (Temporal consistency) established the principle that seasonally aligned training data outperforms larger but misaligned datasets.

**DEC-017** (XGBoost at scale) demonstrated that model selection is context-dependent—LSTM won small, XGBoost won big.

## Lessons for Future Projects

Several principles emerged with applicability beyond this project:

**Validate at production scale.** Sample findings guide exploration but don't guarantee production performance. The LSTM-to-XGBoost reversal would have been a costly surprise without explicit scale validation.

**Document negative results.** DEC-015's failure taught more than its predecessors' successes. Rejected hypotheses are knowledge, not waste.

**Temporal relevance exceeds data volume.** For seasonal forecasting, more data isn't automatically better. Alignment matters more than volume.

**Feature reduction can improve performance.** Intuition suggests more features help. Empirical testing proved otherwise. Ablation studies should be standard practice.

**Test on deployment infrastructure.** The WSL2 GPU setup, Streamlit deployment, and pytest suite caught issues that notebook execution would have missed.

## Future Enhancements

The deployed application provides a foundation for extension:

**Expand coverage.** Current app covers 20 items. Full deployment could include all 2,638 items in the Guayas dataset.

**Add confidence intervals.** Point forecasts are useful; uncertainty quantification would be better. Bootstrap or ensemble methods could provide prediction intervals.

**Implement monitoring.** Production models drift. Automated comparison of predictions to actuals would detect degradation before it impacts decisions.

**Extend temporal scope.** Training on Q4 2013 + Q1 2014 worked for March 2014 predictions. Rolling window updates could enable ongoing forecasting.

These enhancements would build on the existing architecture, leveraging documented decisions and production-ready artifacts.

## Final Summary

The Corporación Favorita project transformed 125 million rows of grocery transaction data into a deployed forecasting application. The journey from raw data to production involved systematic exploration, feature engineering informed by domain patterns, model selection validated through rigorous experimentation, and deployment through modern infrastructure.

The technical outcome—a model achieving RMSE 6.40—matters less than the process that produced it. Each decision was documented. Failed hypotheses were investigated rather than discarded. Production scale was validated explicitly. The result is not just a working model but a reproducible methodology.

The application is live at https://demand-forecasting-in-retail-app.streamlit.app/. The code is available on GitHub. The documentation traces every decision from Day 1 through deployment. For Corporación Favorita—or any retailer facing similar forecasting challenges—this project demonstrates that accurate, deployable demand prediction is achievable through structured data science practice.

---

**Alberto Diaz Durana**  
**November 2025**

[GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)
