# Week 2 to Week 3 Handoff Document

**Project:** Corporación Favorita Grocery Sales Forecasting  
**Phase Completed:** Week 2 - Feature Engineering  
**Next Phase:** Week 3 - Modeling & Analysis  
**Handoff Date:** 2025-11-12  
**Prepared by:** Alberto Diaz Durana

---

## Executive Summary

**Week 2 Status: COMPLETE (100%)**

- **Allocated Time:** 20 hours (Days 1-5)
- **Actual Time:** ~10-12 hours (50% under budget)
- **Buffer Accumulated:** 21.5 hours (Week 1 + Week 2 combined)
- **Features Engineered:** 29 features (100% of target range)
- **Quality Status:** All validated, documented, ready for modeling

**Key Accomplishments:**
- Created 29 engineered features across 5 categories (temporal, oil, aggregations, promotions)
- Validated all features (no data leakage, minimal NaN)
- Made 2 critical decisions (lag NaN strategy, oil feature inclusion)
- Generated 8 visualizations and complete documentation
- Finished 8-10 hours ahead of schedule

**Week 3 Readiness:**
- Final dataset: w02_d05_FE_final.pkl (300,896 × 57 columns, 110.4 MB)
- Strong buffer position enables ambitious Week 3 scope
- Feature validation strategy defined (permutation, SHAP, ablation)
- All necessary context documented for next phase

---

## Week 2 Daily Progress Summary

### Day 1: Lag Features (w02_d01_FE_lags.ipynb)
**Status:** COMPLETE (-1.5h under budget)

**Features Created (4):**
- unit_sales_lag1 (yesterday's sales)
- unit_sales_lag7 (last week)
- unit_sales_lag14 (2 weeks ago)
- unit_sales_lag30 (1 month ago)

**Key Findings:**
- Strong autocorrelation: lag1 (0.26), lag7 (0.40), lag14 (0.32), lag30 (0.27)
- NaN counts: lag1 (27,372), lag7 (32,175), lag14 (35,063), lag30 (39,951)
- Decision: Keep NaN (tree models handle natively)

**Decision Log:**
- DEC-011: Lag NaN Strategy (keep NaN, don't impute)

**Output:**
- Dataset: w02_d01_FE_with-lags.pkl (32 columns)
- Visualization: w02_d01_FE_lag-validation.png

---

### Day 2: Rolling Features (w02_d02_FE_rolling.ipynb)
**Status:** COMPLETE (-1.5h under budget)

**Features Created (6):**
- unit_sales_7d/14d/30d_avg (moving averages)
- unit_sales_7d/14d/30d_std (rolling volatility)

**Key Findings:**
- Rolling averages smooth noise effectively
- min_periods=1 reduced NaN to <1% as planned
- Volatility features capture demand stability patterns

**Output:**
- Dataset: w02_d02_FE_with-rolling.pkl (38 columns)
- Visualization: w02_d02_FE_rolling-smoothing.png

---

### Day 3: Oil Features (w02_d03_FE_oil.ipynb)
**Status:** COMPLETE (-1.5h under budget, ENHANCED scope)

**Features Created (6, planned 5):**
- oil_price (daily WTI crude price)
- oil_price_lag7/14/30 (lags)
- oil_price_change7/14 (momentum derivatives)

**Key Findings:**
- Correlation sign flip: Week 1 (r=-0.55 aggregate) → Day 3 (r=+0.01 granular)
- Magnitude drop due to aggregation level and sparse data
- Decision: Keep all oil features despite weak linear correlation
- Enhancement: Added change14 for multi-scale momentum capture

**Decision Log:**
- DEC-012: Include oil features despite weak granular correlation
  - Rationale: Tree models find non-linear patterns, interactions with other features
  - Dual derivatives capture short-term vs medium-term momentum
  - Week 3 feature importance will validate utility

**Output:**
- Dataset: w02_d03_FE_with-oil.pkl (44 columns)
- Visualization: w02_d03_FE_oil-correlation.png

---

### Day 4: Aggregation Features (w02_d04_FE_aggregations.ipynb)
**Status:** COMPLETE (-2.0h under budget)

**Features Created (11):**
- Store aggregations (3): avg, median, std by store_nbr
- Cluster aggregations (3): avg, median, std by cluster
- Item aggregations (5): avg, median, std, count, total by item_nbr

**Key Findings:**
- No data leakage: All aggregations constant within entities
- Store range: 4.20-9.63 units avg (2.29x)
- Cluster gap: 2.01x (vs 4.25x Week 1, expected dilution)
- Item range: 1-21,424 total sales (popular vs niche spectrum)
- 20 NaN in item_std (single-transaction items, expected)

**Output:**
- Dataset: w02_d04_FE_with-aggregations.pkl (55 columns)
- Visualization: w02_d04_FE_aggregation-distributions.png

---

### Day 5: Promotion Features (w02_d05_FE_final.ipynb)
**Status:** COMPLETE (-2.5h under budget)

**Features Created (2):**
- promo_item_avg_interaction (onpromotion × item_avg_sales)
- promo_cluster_interaction (onpromotion × cluster_avg_sales)

**Key Findings:**
- Only 4.6% transactions promoted (13,905 / 300,896)
- +74% mean lift, +66.7% median lift (Week 1 validated)
- promo_item_avg_interaction: 2.4x better correlation (0.1549 vs 0.0653)
- Family differences: CLEANING (+71.7%), GROCERY I (+70.9%), BEVERAGES (+63.3%)

**Output:**
- Dataset: w02_d05_FE_final.pkl (57 columns) **← FINAL DATASET**
- Visualization: w02_d05_FE_promotion-effects.png

---

## Complete Feature Inventory (29 Engineered)

### Temporal Features (10)

| Feature | Type | Window | NaN % | Correlation | Description |
|---------|------|--------|-------|-------------|-------------|
| unit_sales_lag1 | Lag | 1 day | 9.1% | 0.2639 | Yesterday's sales |
| unit_sales_lag7 | Lag | 7 days | 10.7% | 0.4027 | Last week same day |
| unit_sales_lag14 | Lag | 14 days | 11.7% | 0.3194 | 2 weeks ago |
| unit_sales_lag30 | Lag | 30 days | 13.3% | 0.2654 | 1 month ago |
| unit_sales_7d_avg | Rolling | 7 days | 0.1% | Strong | 7-day moving average |
| unit_sales_14d_avg | Rolling | 14 days | 0.2% | Strong | 14-day moving average |
| unit_sales_30d_avg | Rolling | 30 days | 0.3% | Strong | 30-day moving average |
| unit_sales_7d_std | Rolling | 7 days | 0.1% | Moderate | 7-day volatility |
| unit_sales_14d_std | Rolling | 14 days | 0.2% | Moderate | 14-day volatility |
| unit_sales_30d_std | Rolling | 30 days | 0.3% | Moderate | 30-day volatility |

**Business Interpretation:**
- Lag features capture temporal dependencies (autocorrelation 0.26-0.40)
- Rolling averages smooth noise, identify trends
- Rolling std quantifies demand stability (high = erratic, low = stable)

---

### Oil Features (6)

| Feature | Type | Window | NaN % | Correlation | Description |
|---------|------|--------|-------|-------------|-------------|
| oil_price | External | - | 0.0% | +0.0124 | Daily WTI crude ($26-$111) |
| oil_price_lag7 | Lag | 7 days | 0.0% | +0.0122 | Oil price 1 week ago |
| oil_price_lag14 | Lag | 14 days | 0.0% | +0.0112 | Oil price 2 weeks ago |
| oil_price_lag30 | Lag | 30 days | 0.01% | +0.0113 | Oil price 1 month ago |
| oil_price_change7 | Derivative | 7 days | 0.0% | +0.0005 | 7-day momentum ($-79 to $+79) |
| oil_price_change14 | Derivative | 14 days | 0.0% | +0.0027 | 14-day momentum ($-77 to $+78) |

**Business Interpretation:**
- Weak linear correlation (r≈0.01) but non-linear patterns may exist
- Ecuador oil-dependent economy: transportation costs, purchasing power
- Dual derivatives: change7 = short-term volatility, change14 = sustained trends
- Week 3 feature importance will determine actual utility

**Critical Decision (DEC-012):**
- Keep despite weak correlation - tree models can find non-linear patterns
- Different products may respond to different timescales (elastic vs inelastic)
- Ablation study in Week 3 will validate inclusion

---

### Store Aggregations (3)

| Feature | Entities | Range | NaN % | Description |
|---------|----------|-------|-------|-------------|
| store_avg_sales | 11 stores | 4.20-9.63 | 0.0% | Historical avg sales per store |
| store_median_sales | 11 stores | 2-5 | 0.0% | Robust store baseline |
| store_std_sales | 11 stores | 7.37-20.68 | 0.0% | Store demand variability |

**Business Interpretation:**
- Store 24 example: avg=7.87, median=4.00, std=14.25 (right-skewed, volatile)
- Captures location, size, demographic differences
- High-volume stores: Higher avg, wider std

---

### Cluster Aggregations (3)

| Feature | Entities | Range | NaN % | Gap | Description |
|---------|----------|-------|-------|-----|-------------|
| cluster_avg_sales | 5 clusters | 4.78-9.63 | 0.0% | 2.01x | Historical avg sales per cluster |
| cluster_median_sales | 5 clusters | 2-5 | 0.0% | 2.5x | Robust cluster baseline |
| cluster_std_sales | 5 clusters | 13.15-17.86 | 0.0% | 1.36x | Cluster demand variability |

**Business Interpretation:**
- Corporate strategic groupings (pre-defined, not k-means)
- 2.01x gap vs 4.25x Week 1 (expected dilution at granular level)
- Captures regional/format patterns

---

### Item Aggregations (5)

| Feature | Entities | Range | NaN % | Description |
|---------|----------|-------|-------|-------------|
| item_avg_sales | 2,296 items | 1.00-74.09 | 0.0% | Historical avg sales per item |
| item_median_sales | 2,296 items | 0.56-66.00 | 0.0% | Typical transaction size |
| item_std_sales | 2,296 items | 0-237.56 | 0.01% | Item demand stability (20 NaN) |
| item_count | 2,296 items | 1-400 | 0.0% | Transaction frequency |
| item_total_sales | 2,296 items | 1-21,424 | 0.0% | Cumulative volume |

**Business Interpretation:**
- Most informative aggregations (2,296 unique items)
- Popular items: High count/total, low std/avg (stable demand)
- Niche items: Low count/total, high std/avg (erratic demand)
- Top item 257847: 21,424 total, avg 66.53 (high-volume staple)
- Bottom 10 items: Only 1 transaction each (ultra-rare)

---

### Promotion Features (2)

| Feature | Type | Non-zero % | Correlation | Description |
|---------|------|-----------|-------------|-------------|
| promo_item_avg_interaction | Interaction | 4.6% | 0.1549 | onpromotion × item_avg_sales |
| promo_cluster_interaction | Interaction | 4.6% | 0.0680 | onpromotion × cluster_avg_sales |

**Business Interpretation:**
- Only 4.6% promoted (sparse signal)
- +74% mean lift, +66.7% median lift
- promo_item_avg_interaction: 2.4x better correlation than raw onpromotion
- Captures differential effects: high-volume items respond differently than low-volume
- Family differences: CLEANING (+71.7%), GROCERY I (+70.9%), BEVERAGES (+63.3%)

---

## Data Quality Summary

**Final Dataset Specifications:**
- **File:** data/processed/w02_d05_FE_final.pkl
- **Shape:** 300,896 rows × 57 columns
- **Size:** 110.4 MB (file), 219.9 MB (memory)
- **Date Range:** 2013-01-02 to 2017-08-15 (1,686 days)
- **Entities:** 11 stores, 5 clusters, 2,296 items

**NaN Summary:**
- Lag features: 9-13% (expected, first observations)
- Rolling features: 0.1-0.3% (min_periods=1 strategy)
- Oil features: 0-0.01% (forward/back-fill successful)
- Aggregations: 0-0.01% (20 in item_std for single-transaction items)
- Promotions: 0% (interactions)
- **Overall:** Minimal NaN, tree models handle remaining

**Data Integrity:**
- No data leakage detected (aggregations constant within entities)
- All 300,896 rows preserved
- Temporal order maintained
- No future information used in features

---

## Critical Decisions & Rationale

### DEC-011: Lag NaN Strategy

**Decision:** Keep NaN in lag features, do not impute

**Context:**
- Lag features have 9-13% NaN (first observations per store-item group)
- Options: Drop rows, impute 0, forward-fill, keep NaN

**Rationale:**
- XGBoost/LightGBM handle NaN natively via learned splits
- Dropping rows loses valuable data (reduces 300K → ~270K)
- Imputing 0 creates false signal (0 ≠ missing)
- Forward-fill violates temporal causality (uses future info)
- Keeping NaN preserves uncertainty, models learn appropriate treatment

**Impact:**
- Week 3 models (XGBoost, LightGBM) will split on "missing" vs "present"
- Linear models (ARIMA) will require separate imputation strategy
- Feature engineering approach generalizes to production (no imputation logic needed)

---

### DEC-012: Oil Features Inclusion

**Decision:** Include 6 oil features despite weak granular correlation (r≈+0.01)

**Context:**
- Week 1 finding: r=-0.55 (aggregate daily totals)
- Day 3 finding: r≈+0.01 (granular store-item-date level)
- Sign flip and 97% magnitude drop due to aggregation level

**Rationale:**
1. **Non-linear patterns:** Tree models can find interactions correlation misses
2. **Multi-scale momentum:** Dual derivatives (change7, change14) capture different timescales
3. **Category effects:** BEVERAGES vs CLEANING may respond differently
4. **Interaction potential:** oil × promotion, oil × holiday
5. **Low cost:** 6 features, 0% NaN, 0.2s computation
6. **Week 3 validation:** Feature importance will determine actual utility

**Enhancement:**
- Added change14 (originally planned 5 features, created 6)
- Provides both short-term (7d) and medium-term (14d) momentum
- Different products respond to different timescales (elastic vs inelastic goods)

**Week 3 Validation Plan:**
- Permutation importance: Does oil matter to trained model?
- SHAP analysis: How do models use oil features?
- Ablation study: Performance with vs without oil features
- Category segmentation: BEVERAGES vs CLEANING sensitivity

---

## Week 3 Preparation: Feature Validation Strategy

### Objective
Validate which of 29 engineered features are actually useful for prediction, validate critical decisions (DEC-011, DEC-012), and identify any features to exclude from final model.

### Methods

#### 1. Permutation Importance (Week 3 Day 2)

**What:**
- Shuffle each feature independently after training
- Measure NWRMSLE performance drop
- Identifies features model actually uses (not just correlates with)

**Implementation:**
```python
from sklearn.inspection import permutation_importance

# After training XGBoost model
perm_importance = permutation_importance(
    model, X_val, y_val, 
    n_repeats=10,
    scoring='neg_mean_squared_log_error',
    random_state=42
)

# Rank features by importance
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)
```

**Expected Insights:**
- Which temporal features matter most (lag1 vs lag7 vs lag30)?
- Do oil features appear in top 20? (validates DEC-012)
- Are item aggregations more important than store/cluster?
- Does promo_item_avg_interaction outperform raw onpromotion?

---

#### 2. SHAP Values (Week 3 Day 3)

**What:**
- Game-theoretic approach to feature importance
- Provides both global and local explanations
- Shows feature interactions and non-linear effects

**Implementation:**
```python
import shap

# Install first: pip install shap --break-system-packages

# Create explainer for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Global importance
shap.summary_plot(shap_values, X_val, plot_type="bar")
shap.summary_plot(shap_values, X_val)  # Beeswarm plot

# Local explanation (single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], X_val.iloc[0])

# Dependence plots (feature interactions)
shap.dependence_plot("oil_price_change7", shap_values, X_val)
shap.dependence_plot("promo_item_avg_interaction", shap_values, X_val, 
                     interaction_index="item_avg_sales")
```

**Expected Insights:**
- How does oil_price_change7 interact with other features?
- Does promo_item_avg_interaction show non-linear effects?
- Are there interaction patterns (e.g., oil × holiday, promo × item_avg)?
- Which features drive predictions for high-volume vs low-volume items?

---

#### 3. Ablation Study (Week 3 Day 3)

**What:**
- Remove feature group, retrain model, compare NWRMSLE
- Tests if feature group improves prediction
- Computationally expensive (requires retraining)

**Implementation:**
```python
# Baseline model (all features)
baseline_score = train_and_evaluate(X_train_all, y_train, X_val_all, y_val)

# Test critical decisions
oil_features = ['oil_price', 'oil_price_lag7', 'oil_price_lag14', 
                'oil_price_lag30', 'oil_price_change7', 'oil_price_change14']
X_train_no_oil = X_train_all.drop(columns=oil_features)
X_val_no_oil = X_val_all.drop(columns=oil_features)
no_oil_score = train_and_evaluate(X_train_no_oil, y_train, X_val_no_oil, y_val)

print(f"Baseline NWRMSLE: {baseline_score:.4f}")
print(f"Without oil: {no_oil_score:.4f}")
print(f"Delta: {no_oil_score - baseline_score:+.4f}")
```

**Feature Groups to Test:**
1. **Oil features (6)** - Validates DEC-012
2. **Rolling std features (3)** - Do volatility features matter?
3. **Cluster aggregations (3)** - Do corporate groupings add value?
4. **Promotion interactions (2)** - Better than raw onpromotion?

**Expected Insights:**
- If removing oil features changes NWRMSLE by <0.001, exclude from final model
- If removing promo_item_avg_interaction hurts significantly, keep despite sparsity
- Quantitative validation of all critical decisions

---

#### 4. Correlation Analysis (Already Done, Extend in Week 3)

**What:**
- Pearson (linear relationships)
- Spearman (monotonic relationships)
- Compare to model-based importance

**Extension for Week 3:**
```python
# Correlation matrix of top 20 features by permutation importance
top_features = feature_importance_df.head(20)['feature'].tolist()
correlation_matrix = X_train[top_features + ['unit_sales']].corr()

# Identify multicollinear features (correlation > 0.9)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

# If highly correlated, keep one with higher permutation importance
```

**Expected Insights:**
- Are unit_sales_7d_avg and unit_sales_14d_avg redundant?
- Can we remove lag14 if lag7 and lag30 capture most information?
- Are item_count and item_total_sales highly correlated?

---

### Validation Timeline (Week 3)

**Day 1:** Baseline modeling (naive, ARIMA)
- Establish naive forecast baseline
- Quick ARIMA on aggregated data
- No feature validation yet

**Day 2:** XGBoost modeling + Permutation Importance
- Train XGBoost with all 29 features
- Run permutation importance (10 repeats, ~15 min computation)
- Identify top 20 features
- Document feature ranking

**Day 3:** SHAP Analysis + Ablation Study
- Install SHAP (`pip install shap --break-system-packages`)
- Generate SHAP summary plots (global importance)
- Create dependence plots for top 10 features
- Run ablation studies (oil, rolling std, cluster, promotion)
- Validate DEC-012 (oil inclusion)

**Day 4:** Advanced modeling (Prophet, LSTM)
- Use validated feature set from Days 2-3
- Compare feature importance across models
- Document model-specific feature preferences

**Day 5:** Model comparison + Final feature set
- Synthesize all validation results
- Create final feature importance report
- Decide which features to exclude (if any)
- Update feature dictionary with importance rankings

---

### Validation Deliverables

**Visualizations:**
- Permutation importance bar chart (all 29 features)
- SHAP summary plot (beeswarm)
- SHAP dependence plots (top 10 features)
- Ablation study results table

**Documentation:**
- Feature importance report (markdown)
- DEC-012 validation summary (did oil features help?)
- Recommendations for final feature set
- Feature exclusion rationale (if any)

**Decisions:**
- Which features to exclude from production model?
- Which interactions to add (if any)?
- Which feature groups most valuable per product category?

---

## Known Issues & Limitations

### Data Limitations
1. **99.1% sparsity** - Only 0.9% of potential store-item-date combinations have sales
   - Impact: Limits ability to learn from zeros vs true missing data
   - Mitigation: Working with sparse format (300K rows, not 33M dense)

2. **300K sample** - Filtered to Guayas region, top-3 families for speed
   - Impact: Patterns may differ in full 33M dataset
   - Mitigation: Clear documentation of scope, methodology generalizes

3. **Oil correlation weak** (r≈0.01 granular level)
   - Impact: Uncertain if oil features useful
   - Mitigation: Week 3 validation via feature importance and ablation

4. **Only 4.6% promoted** - Sparse promotion signal
   - Impact: Interaction features only non-zero for 13,905 rows
   - Mitigation: Created interactions to capture differential effects

### Feature Limitations
1. **Lag NaN (9-13%)** - First observations per store-item group
   - Impact: Reduces effective training data for some models
   - Mitigation: Tree models handle NaN natively (DEC-011)

2. **Aggregations assume stationarity** - Historical avg = future baseline
   - Impact: May not hold during regime changes (e.g., pandemic)
   - Mitigation: Rolling features capture recent trends

3. **No cross-feature interactions** - Only promo interactions created
   - Impact: May miss oil × holiday, lag × promotion patterns
   - Mitigation: Tree models learn interactions automatically

4. **Single-transaction items** (20 with NaN std)
   - Impact: 0.01% of data, can't compute volatility
   - Mitigation: Acceptable, models treat as missing

### Modeling Risks
1. **Oil feature utility uncertain** - DEC-012 hypothesis untested
   - Risk: Wasted 0.5 hours creating features that don't help
   - Mitigation: Week 3 ablation study will quantify impact

2. **Overfitting to sparse data** - Models may memorize rather than generalize
   - Risk: Low training error, high validation error
   - Mitigation: Time-series CV, proper train/val splits

3. **Temporal data leakage** - Ensure no future information in features
   - Risk: Artificially high performance, fails in production
   - Mitigation: All features use only past data, validated in EDA

---

## Buffer Analysis & Week 3 Recommendations

### Time Performance Summary

**Week 1 Performance:**
- Allocated: 20 hours
- Actual: ~8-10 hours
- Buffer gained: 10-12 hours

**Week 2 Performance:**
- Day 1: -1.5h (lag features)
- Day 2: -1.5h (rolling features)
- Day 3: -1.5h (oil features)
- Day 4: -2.0h (aggregations)
- Day 5: -2.5h (promotions)
- **Total: -9h (finished 45% under budget)**

**Cumulative Buffer: 21.5 hours** (Week 1: 12.5h + Week 2: 9h)

### Week 3 Recommendations

**Allocate Buffer to High-Value Activities:**

1. **Feature Validation (Days 2-3)** - Invest 4-5 hours
   - Permutation importance (thorough, 20+ repeats)
   - SHAP analysis (multiple visualizations)
   - Ablation studies (test all critical decisions)
   - Justification: Strong buffer enables rigorous validation

2. **Advanced Modeling (Day 4)** - Invest 3-4 hours
   - LSTM architecture tuning
   - Prophet parameter optimization
   - Ensemble methods exploration
   - Justification: 21.5h buffer allows experimentation

3. **Model Comparison (Day 5)** - Invest 2-3 hours
   - Category-level analysis (BEVERAGES vs CLEANING vs GROCERY)
   - Popular vs niche item strategies
   - Comprehensive report generation
   - Justification: Documentation critical for Week 4

**Contingency Plan:**
- If Week 3 runs behind, can reduce:
  - LSTM complexity (simpler architecture)
  - Prophet parameter grid (fewer combinations)
  - Ensemble methods (focus on best single model)
- Core deliverables (XGBoost, feature validation) protected by buffer

**Risk Mitigation:**
- Daily checkpoints continue (15 min each)
- Monitor buffer health (trigger contingency if <8h after Day 3)
- Preserve 8-10h buffer for Week 4 (presentation, report, code consolidation)

---

## Week 3 Quick-Start Guide

### Step 1: Environment Setup (15 minutes)

```bash
# Navigate to project directory
cd /path/to/retail_demand_analysis

# Activate environment
source venv/bin/activate  # or equivalent

# Install SHAP for feature validation
pip install shap --break-system-packages

# Verify data available
ls data/processed/w02_d05_FE_final.pkl
```

### Step 2: Load Final Dataset (5 minutes)

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load final feature set
PROJECT_ROOT = Path.cwd()
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

df = pd.read_pickle(DATA_PROCESSED / 'w02_d05_FE_final.pkl')

print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Verify all 29 engineered features present
engineered_features = [
    'unit_sales_lag1', 'unit_sales_lag7', 'unit_sales_lag14', 'unit_sales_lag30',
    'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg',
    'unit_sales_7d_std', 'unit_sales_14d_std', 'unit_sales_30d_std',
    'oil_price', 'oil_price_lag7', 'oil_price_lag14', 'oil_price_lag30',
    'oil_price_change7', 'oil_price_change14',
    'store_avg_sales', 'store_median_sales', 'store_std_sales',
    'cluster_avg_sales', 'cluster_median_sales', 'cluster_std_sales',
    'item_avg_sales', 'item_median_sales', 'item_std_sales', 'item_count', 'item_total_sales',
    'promo_item_avg_interaction', 'promo_cluster_interaction'
]

missing = [f for f in engineered_features if f not in df.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
else:
    print(f"OK: All 29 engineered features present")
```

### Step 3: Time-Series Train/Val Split (10 minutes)

```python
from sklearn.model_selection import TimeSeriesSplit

# Sort by date (critical for time-series)
df = df.sort_values('date').reset_index(drop=True)

# Define split date (80/20 split for simplicity)
split_date = df['date'].quantile(0.8)

train = df[df['date'] < split_date].copy()
val = df[df['date'] >= split_date].copy()

print(f"Train: {train.shape[0]:,} rows ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Val:   {val.shape[0]:,} rows ({val['date'].min().date()} to {val['date'].max().date()})")

# Prepare feature matrices
feature_cols = [col for col in df.columns if col not in ['id', 'date', 'unit_sales']]
target_col = 'unit_sales'

X_train = train[feature_cols]
y_train = train[target_col]
X_val = val[feature_cols]
y_val = val[target_col]

print(f"\nFeature count: {len(feature_cols)}")
print(f"Target range: {y_train.min():.2f} to {y_train.max():.2f}")
```

### Step 4: Define NWRMSLE Metric (10 minutes)

```python
from sklearn.metrics import mean_squared_log_error

def nwrmsle(y_true, y_pred, weights):
    """
    Normalized Weighted Root Mean Squared Logarithmic Error
    
    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    - weights: Sample weights (1.0 for non-perishable, 1.25 for perishable)
    
    Returns:
    - NWRMSLE score
    """
    # Clip negative predictions to 0
    y_pred = np.maximum(0, y_pred)
    
    # Clip negative actuals to 0 (shouldn't happen, but safety)
    y_true = np.maximum(0, y_true)
    
    # Add 1 to avoid log(0)
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    
    # Weighted squared log error
    squared_log_error = np.square(y_true_log - y_pred_log)
    weighted_squared_log_error = squared_log_error * weights
    
    # Normalize by sum of weights
    nwmsle = weighted_squared_log_error.sum() / weights.sum()
    
    # Take square root
    nwrmsle = np.sqrt(nwmsle)
    
    return nwrmsle

# Create weights (1.25 for perishable, 1.0 for non-perishable)
train_weights = train['perishable'].apply(lambda x: 1.25 if x == 1 else 1.0)
val_weights = val['perishable'].apply(lambda x: 1.25 if x == 1 else 1.0)

print(f"Train weights: {train_weights.value_counts()}")
print(f"Val weights: {val_weights.value_counts()}")
```

### Step 5: Begin Week 3 Day 1 - Baseline Models

**Objective:** Establish naive and ARIMA baselines

**Naive Forecast:**
- Forecast = Yesterday's sales (unit_sales_lag1)
- Simple, interpretable, fast
- Performance: NWRMSLE on validation set

**ARIMA:**
- Aggregate to daily totals (reduces 300K → 1,686 rows)
- Fit ARIMA(p,d,q) with auto-selection
- Expand back to store-item level
- Performance: NWRMSLE on validation set

**Deliverables:**
- Baseline NWRMSLE scores
- Notebook: w03_d01_MODEL_baseline.ipynb
- Figures: naive forecast, ARIMA forecast vs actual

---

## Documentation References

### Key Documents
1. **Week 1 Day 5:** w01_d05_EDA_context_export.ipynb (base features)
2. **Week 2 Plan:** Week2_ProjectPlan_v2_Expanded.md
3. **Decision Logs:** 
   - DEC-011_Lag_NaN_Strategy.md
   - DEC-012_Oil_Features_Inclusion.md
4. **Feature Dictionary:** feature_dictionary_v2.txt (29 entries)
5. **Daily Checkpoints:** w02_d01-d05_checkpoint.md

### Project Structure
```
retail_demand_analysis/
├── data/
│   ├── raw/                       # Original Kaggle CSVs
│   └── processed/
│       └── w02_d05_FE_final.pkl  # ← START HERE for Week 3
├── notebooks/
│   ├── w02_d01_FE_lags.ipynb
│   ├── w02_d02_FE_rolling.ipynb
│   ├── w02_d03_FE_oil.ipynb
│   ├── w02_d04_FE_aggregations.ipynb
│   └── w02_d05_FE_final.ipynb
├── outputs/
│   └── figures/
│       └── features/              # 8 visualizations from Week 2
├── docs/
│   ├── plans/
│   │   ├── Week2_ProjectPlan_v2_Expanded.md
│   │   └── w02_d01-d05_checkpoint.md
│   ├── decisions/
│   │   ├── DEC-011_Lag_NaN_Strategy.md
│   │   └── DEC-012_Oil_Features_Inclusion.md
│   └── feature_dictionary_v2.txt
└── presentation/                   # Week 4
```

---

## Success Criteria for Week 3

### Quantitative
- [ ] NWRMSLE < naive baseline (yesterday's sales)
- [ ] NWRMSLE < ARIMA baseline
- [ ] XGBoost converges without overfitting
- [ ] Feature importance documented for all 29 features
- [ ] Ablation study quantifies DEC-012 (oil features impact)

### Qualitative
- [ ] Feature validation strategy executed (permutation, SHAP, ablation)
- [ ] Model interpretability documented (SHAP plots, feature importance)
- [ ] Category-level analysis (BEVERAGES vs CLEANING vs GROCERY)
- [ ] Popular vs niche item strategies evaluated
- [ ] All Week 3 notebooks executable end-to-end

### Deliverables
- [ ] 5 notebooks (baseline, XGBoost, SHAP, advanced, comparison)
- [ ] 8-10 visualizations (predictions, feature importance, SHAP)
- [ ] Feature validation report (markdown)
- [ ] Model comparison table
- [ ] Week 3 summary (prepare for Week 4)

---

## Contact & Continuity

**Project Lead:** Alberto Diaz Durana  
**Session Completed:** Week 2 (2025-11-12)  
**Next Session:** Week 3 Day 1 - Baseline Modeling

**If starting new chat:**
1. Upload this handoff document to Project Knowledge
2. Reference: "Continue from Week 2 handoff, start Week 3 Day 1"
3. Load: data/processed/w02_d05_FE_final.pkl
4. Verify: 29 engineered features present

**Current Status:**
- Buffer: 21.5 hours (excellent)
- Features: 29/29 complete (100%)
- Dataset: 300,896 × 57 ready for modeling
- Documentation: Complete (checkpoints, decisions, dictionary)

---

**END OF WEEK 2 HANDOFF**

**Week 3 Status:** READY TO BEGIN  
**Confidence Level:** HIGH (strong buffer, complete features, clear plan)  
**Next Action:** Week 3 Day 1 - Baseline modeling notebook
