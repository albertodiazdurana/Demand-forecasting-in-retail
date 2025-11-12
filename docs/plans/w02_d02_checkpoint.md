# Day 2 Checkpoint - Week 2 (2025-11-12)

## Time Tracking
- **Allocated:** 4 hours (3.2h core + 0.8h buffer)
- **Actual:** 2.5 hours
- **Variance:** -1.5 hours (37.5% under budget)
- **Reason for variance:** Efficient rolling feature creation (30s actual vs 3-5min estimated), smooth execution, no debugging needed

## Scope Completion
- [X] Part 0: Load Data & Setup - Complete
- [X] Part 1: Create Rolling Average Features - Complete
- [X] Part 2: Create Rolling Standard Deviation Features - Complete
- [X] Part 3: Validation & Visualization - Complete
- [X] Part 4: Save Checkpoint & Documentation - Complete

**Completion Rate:** 4/4 parts complete = 100%

## Key Findings
1. **Most important finding:** BEVERAGES family has 2.5x higher volatility (6.96) than CLEANING (2.84), suggesting different forecasting strategies may be needed per family. Top volatile items are predominantly BEVERAGES (8/10).
2. **Second most important finding:** min_periods=1 for rolling averages eliminated all NaN (0%), while min_periods=2 for rolling std created expected 6.54% NaN (first observation per store-item). Strategy worked as planned.
3. **Unexpected discovery:** Rolling feature computation extremely fast (15.2s avg + 14.7s std = 30s total vs 3-5min estimated). Pandas rolling operations highly optimized.

## Quality Assessment
- **Feature quality:** Excellent - Smoothing visualizations confirm noise reduction, volatility metrics identify erratic items correctly
- **Validation results:** All passed - Longer windows smoother, shorter windows more responsive (as expected)
- **Computation time:** Within budget - 30 seconds actual vs 3-5 minutes estimated
- **Code quality:** Clean - Modular cells, clear documentation, feature dictionary exported

## Blockers & Issues
- **Technical blockers:** None
- **Data quality issues:** None - NaN counts expected and acceptable
- **Conceptual challenges:** None
- **Mitigation actions taken:** N/A - No issues encountered

## Buffer Status
- **Day 2 buffer allocated:** 0.8h
- **Day 2 buffer used:** 0h (finished 1.5h early)
- **Day 2 buffer remaining:** 0.8h unused
- **Cumulative buffer remaining (Week 1 + Week 2):** 15.5h / 12.5h (gained 1.5h today, +1.5h Day 1 = +3h total)
- **Buffer health:** Excellent (>8h threshold, significantly ahead)

## Feature Creation Status
**MUST Features (10 total):**
- [X] Lag 1/7/14/30 (4) - Complete (Day 1)
- [X] Rolling avg 7/14/30 (3) - Complete (Day 2)
- [X] Rolling std 7/14/30 (3) - Complete (Day 2)

**SHOULD Features (15 total):**
- [ ] Oil features (5) - Planned Day 3
- [ ] Store/cluster aggs (3) - Planned Day 4
- [ ] Item aggs (5) - Planned Day 4
- [ ] Promotion features (2) - Planned Day 5

**COULD Features (0-3):**
- [ ] Optional features - TBD based on Day 3-5 progress

**Total Features Created Today:** 6 features  
**Cumulative Features:** 10 / 26-29 target (38% complete)

## Adjustment Decisions for Day 3

**Scope Changes:**
- [X] Keep plan as-is
- [ ] Add analysis/feature: None
- [ ] Remove analysis/feature: None
- [ ] Simplify approach: None

**Time Reallocation:**
- [X] No changes needed
- [ ] Increase time for: N/A
- [ ] Decrease time for: N/A

**Priority Adjustment:**
- [X] Maintain MUST → SHOULD → COULD priority
- [ ] Focus only on MUST features (contingency triggered)
- [ ] Skip COULD features to preserve buffer

**Rationale:** Day 2 completed well ahead of schedule (1.5h under budget) with all quality checks passed. All MUST features now complete (10/10). Proceed with full Day 3 plan (oil price features) as designed. Strong buffer position (15.5h) allows confidence in completing all SHOULD features.

## Next Day Preview
**Day 3 Primary Objectives:**
1. Load guayas_with_rolling.pkl (38 columns)
2. Load and merge oil.csv (1,218 daily WTI prices)
3. Forward-fill missing oil prices (weekends/holidays)
4. Create oil_price_lag7, oil_price_lag14, oil_price_lag30
5. Create oil_price_change (derivative)
6. Validate oil correlation with unit_sales
7. Export guayas_with_oil.pkl (43 columns)

**Day 3 Success Criteria:**
- [ ] 5 oil features created (price + 3 lags + change)
- [ ] Oil prices merged correctly (date alignment verified)
- [ ] Forward-fill reduces oil NaN to ~0%
- [ ] Correlation analysis validates oil relevance (Week 1: r=-0.55)
- [ ] Dataset shape: 300,896 × 43 columns

**Day 3 Contingency Plan (if behind):**
- Reduce oil lags from 3 to 2 (keep 7/14, drop 30)
- Skip oil_price_change derivative
- Simplify correlation analysis

## Decision Log Updates
No new decisions required - Day 2 executed according to plan with no deviations or alternative approaches needed.

## Notes & Learnings
- **What worked well today:** 
  1. min_periods strategy balanced NaN reduction with feature quality
  2. Volatility analysis revealed family-level patterns (BEVERAGES most erratic)
  3. Visualization confirmed smoothing effectiveness
  
- **What could be improved:** 
  1. Time estimates still too conservative (30s vs 3-5min)
  2. Could explore coefficient of variation (CV = std/mean) for normalized volatility comparison
  
- **Insights for Week 3:** 
  - Different families may benefit from separate models (BEVERAGES volatility 2.5x CLEANING)
  - Volatility features (std) will help models quantify prediction uncertainty
  - Rolling features provide smooth baseline for detecting anomalies

---

**Checkpoint completed by:** Alberto Diaz Durana  
**Time spent on checkpoint:** 12 minutes (target: ≤15 min)  
**Next checkpoint:** Day 3, 2025-11-13

---

## Appendix: Outputs Created

**Datasets:**
- `data/processed/guayas_with_rolling.pkl` (300,896 rows × 38 columns, 66.8 MB)

**Visualizations:**
- `outputs/figures/features/rolling_smoothing_validation.png` (5 time series plots)
- `outputs/figures/features/volatility_distribution.png` (2-panel histogram)

**Documentation:**
- `docs/feature_dictionary_v2.txt` updated (6 new rolling features)

**Notebook:**
- `notebooks/w02_d02_FE_rolling.ipynb` (9 cells, ~250 lines)

---

## Feature Summary Statistics

**Rolling Averages (0% NaN):**
- 7-day avg: Smooth weekly trend
- 14-day avg: Bi-weekly baseline
- 30-day avg: Monthly reference

**Rolling Standard Deviations (6.54% NaN):**
- 7-day std: Mean 4.33, Median 2.32, Max 957.32
- 14-day std: Mean 4.54, Median 2.52
- 30-day std: Mean 4.63, Median 2.59

**Volatility by Family:**
- BEVERAGES: 6.96 (highest)
- GROCERY I: 3.88
- CLEANING: 2.84 (lowest)

**Top Volatile Items:**
1. Item 2060793 (BEVERAGES): std = 76.45
2. Item 2061214 (BEVERAGES): std = 67.01
3. Item 2084557 (CLEANING): std = 49.05

---

**End of Day 2 Checkpoint**
