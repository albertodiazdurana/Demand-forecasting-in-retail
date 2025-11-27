# Decision Log - Week 1
## Corporación Favorita Grocery Sales Forecasting Project

**Project:** Demand Forecasting in Retail  
**Phase:** Week 1 - Exploration & Understanding  
**Date Range:** November 2025  
**Author:** Alberto Diaz Durana

---

## Decision Log Format

Each decision follows this structure:
- **ID:** Unique identifier (DEC-XXX)
- **Date:** When decision was made
- **Context:** What prompted the decision
- **Decision:** What was decided
- **Rationale:** Why this option was chosen
- **Alternatives Considered:** Other options evaluated
- **Impact:** Effect on project scope, timeline, or quality
- **Status:** Active / Superseded / Rejected

---

## DEC-001: Product Family Selection

**Date:** Week 1, Day 1  
**Notebook:** d01_w01_setup_inventory.ipynb  
**Status:** Active

**Context:**
Dataset contains 33 product families with varying item counts and sales patterns. Need to select manageable subset for analysis while maintaining representativeness.

**Decision:**
Select top-3 families by item count: GROCERY I, BEVERAGES, CLEANING

**Rationale:**
- Top-3 families contain 4,393 items (59% of catalog)
- Provides sufficient variety for pattern detection
- Reduces computational complexity for development
- All three are non-perishable (discovered later: simplifies forecasting)

**Alternatives Considered:**
1. All 33 families - Rejected (too complex for 4-week timeline)
2. Top-5 families - Rejected (marginal benefit, increased complexity)
3. Single family - Rejected (insufficient variety for pattern detection)
4. Sales-based selection - Rejected (item count better proxy for complexity)

**Impact:**
- Scope: 59% of items, ~70% of sales volume
- Timeline: Enables 300K sample (DEC-002)
- Quality: 0% perishables in scope (documented in DEC-010)

---

## DEC-002: Sample Size Selection

**Date:** Week 1, Day 2  
**Notebook:** d02_w01_sampling.ipynb  
**Status:** Active

**Context:**
Full Guayas dataset contains ~33M rows after filtering. Need representative sample for efficient development while maintaining statistical validity.

**Decision:**
Use 300K random sample with fixed seed (random_state=42)

**Rationale:**
- Sufficient coverage: All 11 stores, all 2,296 items, full date range
- Statistical validity: 0.9% sample rate provides narrow confidence intervals
- Development speed: Loads in <5 seconds, fits in 100 MB RAM
- Reproducibility: Fixed seed ensures identical sample across runs

**Alternatives Considered:**
1. Full 33M dataset - Rejected (hours per operation, memory issues)
2. 100K sample - Rejected (rare items underrepresented)
3. 1M sample - Rejected (marginal benefit, 3x slower)
4. Stratified sample - Rejected (random sampling preserved distributions naturally)

**Impact:**
- Timeline: 10x faster iteration → 8.5h buffer gained
- Quality: Verified representativeness (store, family, temporal distributions match)
- Risk: Validate on full dataset in Week 3 (optional)

---

## DEC-003: Missing Promotion Value Handling

**Date:** Week 1, Day 3  
**Notebook:** d03_w01_EDA_quality_preprocessing.ipynb  
**Status:** Active

**Context:**
`onpromotion` column has 18.57% missing values (55,706 records), concentrated in 2013-2014. Hypothesis: promotion tracking system not implemented until later.

**Decision:**
Fill missing `onpromotion` values with 0 (False = not promoted)

**Rationale:**
- Conservative assumption: If unknown, assume baseline (no promotion)
- Temporal context: Early years likely had fewer promotions
- Data preservation: Retain all rows (no deletion)
- Model safety: Underestimating promotion frequency is safer than overestimating

**Alternatives Considered:**
1. Delete rows with missing values - Rejected (18.57% data loss, temporal bias)
2. Fill with 1 (assume promoted) - Rejected (unrealistic, inflates promotion rate)
3. Fill with 0.5 (uncertain) - Rejected (promotion is binary, not continuous)
4. Impute from patterns - Rejected (complex, minimal benefit for EDA)

**Impact:**
- Data quality: 0% missing values after fill
- Promotion rate: 4.62% (may be slightly underestimated for 2013-2014)
- Feature engineering: Enables promotion history features in Week 2

---

## DEC-004: Outlier Detection and Handling

**Date:** Week 1, Day 3  
**Notebook:** d03_w01_EDA_quality_preprocessing.ipynb  
**Status:** Active

**Context:**
Retail sales data contains extreme values from promotions, holidays, and bulk purchases. Need to identify outliers without removing legitimate business events.

**Decision:**
Use 3-method triangulation (IQR, Z-score, Isolation Forest) and RETAIN outliers with flags

**Rationale:**
- Multiple methods reduce false positives (single method too aggressive)
- High-confidence outliers: Only 0.28% flagged by ALL three methods
- Business context: Sales spikes during promotions/holidays are LEGITIMATE
- Model flexibility: Outlier flags allow models to weight/exclude if needed

**Methods Applied:**
1. IQR (Interquartile Range): 4,956 outliers (1.65%)
2. Z-score (|z| > 3): 1,823 outliers (0.61%)
3. Isolation Forest: 2,147 outliers (0.72%)
4. Triangulated (all three): 846 outliers (0.28%) - HIGH CONFIDENCE

**Alternatives Considered:**
1. Remove all outliers - Rejected (loses legitimate promotional spikes)
2. Single method only - Rejected (high false positive rate)
3. Cap/winsorize outliers - Rejected (distorts true sales magnitude)
4. Ignore outliers - Rejected (no visibility for modeling)

**Impact:**
- Data integrity: All 300K rows retained
- Feature created: `outlier_score` (0-3 based on methods agreeing)
- Model options: Can exclude outliers if they hurt performance

---

## DEC-005: Sparse Data Format Retention

**Date:** Week 1, Day 3  
**Notebook:** d03_w01_EDA_quality_preprocessing.ipynb  
**Status:** Active

**Context:**
Dataset is 99.1% sparse (most store-item-date combinations have no sales). Options: keep sparse format OR fill calendar gaps with zeros.

**Decision:**
Keep sparse format (300K rows) - do NOT fill calendar gaps to create complete time series

**Rationale:**
- Memory efficiency: 300K rows vs 42.6M rows if gaps filled
- Retail reality: Sparsity is NORMAL (items don't sell every day)
- Model compatibility: Sparse time series models (Croston's, TSB) designed for this
- Information preservation: Implicit zeros ≠ explicit zeros (different meanings)

**Alternatives Considered:**
1. Fill all gaps with zeros - Rejected (42.6M rows, 32 GB memory, most rows = 0)
2. Fill gaps per item (weekly) - Rejected (still massive, unclear benefit)
3. Aggregate to weekly - Rejected (loses daily patterns needed for forecasting)

**Impact:**
- Memory: 153 MB (vs 32 GB if filled)
- Model selection: Must use sparse-aware models (Week 3)
- Feature engineering: "Days since last sale" requires explicit tracking

---

## DEC-006: Rolling Statistics Configuration

**Date:** Week 1, Day 4  
**Notebook:** d04_w01_EDA_temporal_patterns.ipynb  
**Status:** Active

**Context:**
Rolling statistics (7/14/30-day moving averages) require minimum observations. Sparse data means many store-item combinations have gaps.

**Decision:**
Use `min_periods=1` for rolling calculations

**Rationale:**
- Coverage: All rows receive rolling features (no NaN from insufficient data)
- Sparse data handling: Even items with few sales get smoothed values
- Model input: Ensures no missing features due to sparse history
- Validation: Visual inspection confirmed smoothing effect is reasonable

**Configuration:**
```python
df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
```

**Alternatives Considered:**
1. min_periods=window (e.g., 7 for 7-day) - Rejected (too many NaN for sparse items)
2. min_periods=3 (compromise) - Rejected (still creates gaps for slow movers)
3. Forward-fill NaN after rolling - Rejected (introduces bias, complex)

**Impact:**
- Feature completeness: 100% coverage for rolling features
- Accuracy trade-off: Early observations have less smoothing (acceptable)
- Week 2: Apply same approach to production rolling features

---

## DEC-007: Item Velocity Classification

**Date:** Week 1, Day 4  
**Notebook:** d04_w01_EDA_temporal_patterns.ipynb  
**Status:** Active

**Context:**
Items have vastly different sales velocities (0.1 to 70+ units/day). Need classification for differentiated forecasting strategies.

**Decision:**
Use 20/60/20 percentile split: Fast movers (top 20%), Medium movers (60%), Slow movers (bottom 20%)

**Rationale:**
- Pareto alignment: Top 20% generates 58.4% of sales (validates threshold)
- Actionable differentiation: Clear strategy per tier (accuracy targets, replenishment frequency)
- Industry standard: Similar to ABC analysis (A=20%, B=30%, C=50%)
- Symmetric interpretation: Easy to communicate (quintiles)

**Thresholds:**
- Fast movers: ≥7.8 units/day (460 items, 58.4% of sales)
- Slow movers: ≤2.27 units/day (460 items, 2.2% of sales)
- Medium movers: Between thresholds (1,376 items, 39.4% of sales)

**Alternatives Considered:**
1. 33/33/33 equal split - Rejected (dilutes top tier, includes medium performers in "fast")
2. 80/15/5 Pareto-strict - Rejected (tiny fast tier, hard to differentiate)
3. K-means clustering - Rejected (complex, non-standard thresholds)
4. Sales-based (not velocity) - Rejected (velocity normalizes for data availability)

**Impact:**
- Forecasting strategy: Different accuracy targets per tier
- Resource allocation: Focus effort on fast movers (58% of sales)
- Week 3: May create velocity tier features (one-hot encoding)

---

## DEC-008: Promotion and Holiday Combination Strategy

**Date:** Week 1, Day 5  
**Notebook:** d05_w01_EDA_context_export.ipynb  
**Status:** Active

**Context:**
Analysis revealed interaction between promotions and holidays. Expected synergy (amplification), but data shows opposite.

**Decision:**
Recommend AVOIDING promotion scheduling on holiday days

**Rationale:**
- Negative synergy discovered: -16.1% (combined lift < sum of individual lifts)
- Promotion alone: +76.4% lift (highly effective on normal days)
- Holiday alone: +12.9% lift (natural demand increase)
- Combined: +73.2% lift (LESS than promotion alone!)
- Business logic: Holidays already drive traffic; promotions add cost without incremental benefit

**Analysis Results:**
| Condition | Avg Sales | Lift vs Baseline |
|-----------|-----------|------------------|
| Normal (no promo, no holiday) | 6.49 | 0% (baseline) |
| Promotion only | 11.45 | +76.4% |
| Holiday only | 7.33 | +12.9% |
| Promotion + Holiday | 11.24 | +73.2% |

**Expected (additive):** 76.4% + 12.9% = +89.3%
**Actual:** +73.2%
**Synergy:** -16.1% (NEGATIVE)

**Alternatives Considered:**
1. Ignore interaction - Rejected (leaves money on table)
2. Heavy holiday promotions - Rejected (negative ROI demonstrated)
3. Test different holiday types - Future enhancement (Additional vs Event vs Holiday)

**Impact:**
- Promotional calendar: Schedule promotions on NORMAL days
- Budget optimization: Estimated 10-20% ROI improvement
- Feature engineering: Create `promo_x_holiday` interaction term (Week 2)

---

## DEC-009: Oil Price Feature Inclusion

**Date:** Week 1, Day 5  
**Notebook:** d05_w01_EDA_context_export.ipynb  
**Status:** Active

**Context:**
Ecuador is oil-dependent economy. Analyzed correlation between WTI oil price and daily sales.

**Decision:**
Include oil price as macroeconomic indicator feature in Week 2

**Rationale:**
- Moderate negative correlation: r = -0.55 (statistically significant, p < 0.001)
- Economic logic: High oil → inflation → reduced purchasing power (Ecuador context)
- Historical validation: 2014-2015 oil crash ($110→$26) coincides with sales increase
- Low cost: Oil price is freely available, easy to merge
- Comparison: -0.55 correlation stronger than many individual features

**Features to Create (Week 2):**
1. `oil_price` - Daily WTI oil price
2. `oil_price_lag7` - Price 1 week ago
3. `oil_price_lag14` - Price 2 weeks ago
4. `oil_price_lag30` - Price 1 month ago

**Alternatives Considered:**
1. Exclude oil (weak predictor) - Rejected (correlation is moderate, not weak)
2. Include more macro indicators (GDP, inflation) - Future enhancement (data availability)
3. Use oil price change instead of level - Include BOTH (change captures momentum)

**Impact:**
- Feature count: +4 features in Week 2
- Expected accuracy gain: 2-5% improvement (based on correlation strength)
- Risk: Monitor feature importance in Week 3 (drop if weight < 5%)

---

## DEC-010: Perishable Scope Limitation Documentation

**Date:** Week 1, Day 5  
**Notebook:** d05_w01_EDA_context_export.ipynb  
**Status:** Active

**Context:**
Discovered that top-3 families (GROCERY I, BEVERAGES, CLEANING) contain 0% perishable items. Perishables are concentrated in PRODUCE, DAIRY, MEATS, BREAD/BAKERY.

**Decision:**
Document scope limitation; proceed with non-perishable forecasting focus

**Rationale:**
- Scope is valid: Non-perishables represent significant business value
- Forecasting simplicity: Longer shelf life → more forgiving accuracy requirements
- Timeline protection: Adding perishables would expand scope beyond 4 weeks
- Transparency: Stakeholders informed of limitation

**Full Catalog Analysis:**
- Total items: 4,100
- Perishable items: 986 (24%)
- Top-3 families (our scope): 2,393 items, 0 perishable (0%)

**Perishable Families (NOT in scope):**
- BREAD/BAKERY: 100% perishable
- DAIRY: 100% perishable
- MEATS: 100% perishable
- PRODUCE: 100% perishable
- EGGS: 100% perishable

**Alternatives Considered:**
1. Add perishable families - Rejected (scope expansion, different forecasting requirements)
2. Replace one family with perishable - Rejected (changes established analysis)
3. Ignore limitation - Rejected (intellectual dishonesty)

**Impact:**
- Scope: Clear boundary (non-perishable forecasting only)
- Generalizability: Findings may not apply to PRODUCE, DAIRY, MEATS
- Future work: Phase 2 could expand to include perishables
- Communication: Include in final report limitations section

---

## Decision Summary Table

| ID | Decision | Status | Impact Level |
|----|----------|--------|--------------|
| DEC-001 | Top-3 families (GROCERY I, BEVERAGES, CLEANING) | Active | High |
| DEC-002 | 300K sample with fixed seed | Active | High |
| DEC-003 | Fill onpromotion NaN with False | Active | Medium |
| DEC-004 | 3-method outlier detection, retain with flags | Active | Medium |
| DEC-005 | Keep sparse format (no gap filling) | Active | High |
| DEC-006 | Rolling stats with min_periods=1 | Active | Low |
| DEC-007 | 20/60/20 velocity classification | Active | Medium |
| DEC-008 | Avoid promotions on holidays (-16% synergy) | Active | High |
| DEC-009 | Include oil price features | Active | Medium |
| DEC-010 | Document perishable scope limitation | Active | Medium |

---

## Decisions by Category

**Scope Decisions:**
- DEC-001: Family selection
- DEC-002: Sample size
- DEC-010: Perishable limitation

**Data Quality Decisions:**
- DEC-003: Missing value handling
- DEC-004: Outlier detection
- DEC-005: Sparse format retention

**Feature Engineering Decisions:**
- DEC-006: Rolling statistics configuration
- DEC-007: Velocity classification
- DEC-009: Oil price inclusion

**Business Strategy Decisions:**
- DEC-008: Promotion-holiday scheduling

---

## Week 2 Decision Preview

**Anticipated decisions for Week 2:**
- DEC-011: Lag feature NaN handling strategy
- DEC-012: Oil price forward-fill approach
- DEC-013: Aggregation granularity (store vs cluster vs family)
- DEC-014: Feature selection threshold (importance cutoff)

---

**End of Week 1 Decision Log**

**Next Update:** After Week 2 completion
