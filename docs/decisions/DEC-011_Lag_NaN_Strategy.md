# Decision Log Entry: DEC-011

**Decision ID:** DEC-011  
**Date:** 2025-11-12  
**Phase:** Week 2 Day 1 - Lag Feature Engineering  
**Author:** Alberto Diaz Durana

---

## Decision

**Keep NaN values in lag features; Accept lower correlations vs Week 1 expected values**

---

## Context

During Week 2 Day 1 lag feature engineering:

**Features Created:**
- `unit_sales_lag1` (1 day ago)
- `unit_sales_lag7` (7 days ago)
- `unit_sales_lag14` (14 days ago)
- `unit_sales_lag30` (30 days ago)

**NaN Counts Observed:**
- lag_1: 19,692 (6.54%)
- lag_7: 119,961 (39.87%)
- lag_14: 204,230 (67.87%)
- lag_30: 291,884 (97.00%)

**Correlation Discrepancy:**
- Week 1 expected (aggregated time series level):
  - lag_1: r = 0.602
  - lag_7: r = 0.585
  - lag_14: r = 0.625
  - lag_30: r = 0.360
  
- Week 2 observed (granular store-item level):
  - lag_1: r = 0.399
  - lag_7: r = 0.383
  - lag_14: r = 0.380
  - lag_30: r = 0.259

---

## Options Considered

### Option 1: Fill NaN with 0 (treat as "no sales")
**Pros:**
- Eliminates NaN values
- Simpler for some models

**Cons:**
- Creates false signal (0 ≠ "no history available")
- Assumes no sale = zero, not missing observation
- Biases models toward predicting zero

### Option 2: Fill NaN with mean/median (imputation)
**Pros:**
- Eliminates NaN values
- Maintains some statistical properties

**Cons:**
- Introduces artificial baseline assumption
- Loses information about missing patterns
- Creates false precision

### Option 3: Drop rows with NaN
**Pros:**
- Complete cases only
- No NaN to handle

**Cons:**
- Loses 97% of data (291,884 rows for lag_30)
- Introduces severe selection bias
- Unacceptable data loss

### Option 4: Keep NaN as-is (let models handle)
**Pros:**
- Preserves data integrity
- NaN correctly represents "no historical observation"
- XGBoost/LightGBM handle NaN natively
- Maintains all 300,896 rows

**Cons:**
- Incompatible with some models (e.g., linear regression)
- May require future adjustments

---

## Decision: Option 4 - Keep NaN Values

---

## Rationale

### Technical Justification:
1. **XGBoost and LightGBM handle NaN natively**
   - Treat NaN as a separate category during tree splits
   - Can learn patterns like "no lag_30 history → recent product"
   
2. **NaN preserves semantic meaning**
   - NaN = "no historical observation available"
   - 0 = "zero sales recorded" (different meaning)
   - Distinction is important for forecasting

3. **Data integrity maintained**
   - No artificial values introduced
   - All 300,896 rows preserved
   - Original temporal structure intact

4. **Sparse retail data reality**
   - 99.1% sparsity is inherent to retail (Week 1 finding)
   - Most store-item pairs don't sell daily
   - NaN correctly represents this sparsity

### Correlation Discrepancy Explanation:

**Week 1 autocorrelation (r = 0.60+):**
- Calculated on aggregated daily total sales
- Single time series (sum of all stores/items per day)
- Dense data (no gaps, continuous series)

**Week 2 correlations (r = 0.26-0.40):**
- Calculated at granular store-item level
- 300K sparse rows (most store-item pairs intermittent)
- Many NaN lags dilute correlation at row level

**Why this is CORRECT:**
- Different aggregation levels measure different patterns
- Sparse data inherently has weaker row-level correlations
- Positive correlations (0.26-0.40) confirm temporal signal exists
- Relative ordering preserved (lag_1 > lag_7 > lag_14 > lag_30)
- Models will learn from aggregate patterns during training

---

## Impact

### Week 3 Modeling:
- **XGBoost/LightGBM:** Will handle NaN correctly (primary models)
- **Linear models:** May require imputation if used (future decision point)
- **Feature importance:** NaN patterns may provide predictive signal (e.g., new products)

### Data Quality:
- All 300,896 rows preserved
- No artificial values introduced
- Temporal order maintained

### Future Flexibility:
- Can revisit imputation strategy in Week 3 if needed
- Can create "lag_available" binary flags if helpful
- Decision is reversible if model performance suffers

---

## Alternatives for Future Consideration

If NaN handling becomes problematic in Week 3:

1. **Binary availability flags:**
   - Create `has_lag1`, `has_lag7`, etc. (0/1)
   - Impute NaN with 0
   - Models learn both "availability" and "value"

2. **Forward-fill within groups:**
   - Fill NaN with last known value per store-item
   - Assumes "last sale persists until next sale"
   - May be reasonable for slow-moving items

3. **Separate models by data availability:**
   - Model 1: Items with lag_30 available (mature products)
   - Model 2: Items without lag_30 (new products)
   - Different feature sets per model

---

## Validation & Monitoring

**Week 2 Day 1 Validation:**
- Manual spot-checks: PASSED (lag calculations correct)
- Temporal sort: VERIFIED (dates ascending per store-item)
- Correlations: POSITIVE (0.26-0.40, temporal signal confirmed)
- Data integrity: MAINTAINED (300,896 rows preserved)

**Week 3 Monitoring:**
- Track XGBoost feature importance (do lag features rank high?)
- Monitor prediction errors for items with high NaN % vs low NaN %
- Compare model performance with/without lag features
- Re-evaluate if NaN handling becomes limiting factor

---

## Approval

**Approved by:** Alberto Diaz Durana  
**Date:** 2025-11-12  
**Phase:** Week 2 Day 1 Complete  

**Next review:** Week 3 Day 1 (Model Training)  
**Review criteria:** Feature importance, model performance, prediction accuracy

---

## References

- Week 1 Day 4: Autocorrelation analysis (r = 0.60+ for aggregated time series)
- Week 1 Day 3: Data sparsity finding (99.1% zero sales per store-item-day)
- Week 2 Project Plan v2: Section 4 (Lag Feature Creation)
- XGBoost documentation: Native missing value handling
- Feature dictionary v2: Lag feature definitions

---

**End of Decision Log Entry DEC-011**
