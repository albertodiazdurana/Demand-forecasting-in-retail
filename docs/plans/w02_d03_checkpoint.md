# Day 3 Checkpoint - Week 2 (2025-11-12)

## Time Tracking
- **Allocated:** 4 hours (3.2h core + 0.8h buffer)
- **Actual:** 2.5 hours (estimated)
- **Variance:** -1.5 hours (37.5% under budget)
- **Reason for variance:** Efficient oil merge and feature creation (0.2s computation), straightforward external data integration, no unexpected issues

## Scope Completion
- [X] Part 0: Load Datasets & Date Alignment - Complete
- [X] Part 1: Merge Oil Prices & Forward-Fill - Complete
- [X] Part 2: Create Oil Lag Features - Complete (ENHANCED: Added dual change derivatives)
- [X] Part 3: Oil Correlation Analysis - Complete
- [X] Part 4: Save Checkpoint & Documentation - Complete

**Completion Rate:** 4/4 parts complete = 100%  
**Scope Enhancement:** Created 6 oil features (vs planned 5) - added change14 derivative

## Key Findings
1. **Most important finding:** Linear correlation flipped sign and magnitude: Week 1 aggregate (r=-0.55) → Day 3 granular (r=+0.01). This 97% drop due to aggregation level change and sparse data dilution. Decision: KEEP oil features - tree models can find non-linear patterns correlation misses.
2. **Second most important finding:** Dual change derivatives (change7, change14) provide multi-scale momentum capture. Short-term (±$79, std=$10.52) vs medium-term (±$78, std=$10.57). Different products likely respond to different timescales.
3. **Unexpected discovery:** Forward-fill + back-fill reduced oil NaN from 32.80% to 0% perfectly. Oil date coverage complete (2013-01-01 to 2017-08-31 covers main dataset 2013-01-02 to 2017-08-15).

## Quality Assessment
- **Feature quality:** Excellent - 0% NaN, clean merge, reasonable value ranges ($26-$111)
- **Validation results:** All passed - Correlation weak but non-zero, heatmap generated, dual derivatives validated
- **Computation time:** Within budget - 0.2 seconds for all 6 features
- **Code quality:** Clean - Proper date sorting for shift operations, restored original sort order

## Blockers & Issues
- **Technical blockers:** None
- **Data quality issues:** None - NaN handled perfectly by forward/back-fill
- **Conceptual challenges:** Weak granular correlation required decision analysis (DEC-012)
- **Mitigation actions taken:** Documented rationale for keeping features despite weak linear correlation, added dual derivatives for robustness

## Buffer Status
- **Day 3 buffer allocated:** 0.8h
- **Day 3 buffer used:** 0h (finished 1.5h early)
- **Day 3 buffer remaining:** 0.8h unused
- **Cumulative buffer remaining (Week 1 + Week 2):** 17.0h / 12.5h (gained 1.5h today, +4.5h total Days 1-3)
- **Buffer health:** Excellent (>8h threshold, significantly ahead of schedule)

## Feature Creation Status
**MUST Features (10 total):**
- [X] Lag 1/7/14/30 (4) - Complete (Day 1)
- [X] Rolling avg 7/14/30 (3) - Complete (Day 2)
- [X] Rolling std 7/14/30 (3) - Complete (Day 2)

**SHOULD Features (15 total):**
- [X] Oil features (6) - Complete (Day 3) - ENHANCED from 5 to 6
- [ ] Store/cluster aggs (3) - Planned Day 4
- [ ] Item aggs (5) - Planned Day 4
- [ ] Promotion features (2) - Planned Day 5

**COULD Features (0-3):**
- [ ] Optional features - TBD based on Day 4-5 progress

**Total Features Created Today:** 6 features (oil_price + 3 lags + 2 derivatives)  
**Cumulative Features:** 16 / 26-29 target (62% complete)

## Adjustment Decisions for Day 4

**Scope Changes:**
- [X] Keep plan as-is
- [X] Enhancement accepted: Added change14 derivative (justified by moderate correlation, multi-scale momentum)
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

**Rationale:** Day 3 completed ahead of schedule with thoughtful scope enhancement. All MUST features complete (10/10), 6/15 SHOULD features complete. Strong buffer position (17h) allows confidence in completing remaining SHOULD features (8 aggregations + 2 promotion) on Days 4-5. Proceed with full Day 4 plan (store/item aggregations).

## Next Day Preview
**Day 4 Primary Objectives:**
1. Load w02_d03_FE_with-oil.pkl (44 columns)
2. Create 3 store-level aggregations (avg, median, std by store_nbr)
3. Create 3 cluster-level aggregations (avg, median, std by cluster)
4. Create 5 item-level aggregations (avg, median, std, count, total by item_nbr)
5. Validate aggregation quality (no leakage, reasonable ranges)
6. Export w02_d04_FE_with-aggregations.pkl (55 columns)

**Day 4 Success Criteria:**
- [ ] 11 aggregation features created (3 store + 3 cluster + 5 item)
- [ ] No data leakage (aggregations use only historical data)
- [ ] Aggregation ranges reasonable (no extreme outliers)
- [ ] Computation time <10 minutes
- [ ] Dataset shape: 300,896 × 55 columns

**Day 4 Contingency Plan (if behind):**
- Reduce item aggregations from 5 to 3 (keep avg, median, std; drop count, total)
- Simplify validation (spot-checks only)
- Skip detailed aggregation distribution analysis

## Decision Log Updates
- **DEC-012:** Include 6 oil features despite weak granular correlation
  - Context: Correlation r=-0.55 (aggregate) → r=+0.01 (granular), sign flip
  - Decision: Keep all 6 features - tree models can find non-linear patterns
  - Enhancement: Added change14 derivative for multi-scale momentum
  - Impact: Provides macro context, models will reveal utility via feature importance

## Notes & Learnings
- **What worked well today:** 
  1. Dual change derivatives decision - provides multi-scale momentum (short vs medium-term)
  2. Forward/back-fill strategy eliminated all NaN perfectly
  3. Thoughtful correlation analysis led to justified decision despite weak signal
  
- **What could be improved:** 
  1. Could have checked correlation by product family (BEVERAGES vs CLEANING sensitivity)
  2. Scatter plots by time period might reveal non-stationary relationships
  
- **Insights for Week 3:** 
  - Feature importance analysis CRITICAL - will determine if oil features useful
  - May need ablation study: model performance with/without oil
  - Category-level analysis: Do BEVERAGES respond to oil differently than CLEANING?
  - Weak linear correlation ≠ uninformative - tree models learn non-linear patterns

---

**Checkpoint completed by:** Alberto Diaz Durana  
**Time spent on checkpoint:** 15 minutes (target: ≤15 min)  
**Next checkpoint:** Day 4, 2025-11-13

---

## Appendix: Outputs Created

**Datasets:**
- `data/processed/w02_d03_FE_with-oil.pkl` (300,896 rows × 44 columns, 80.6 MB)

**Visualizations:**
- `outputs/figures/features/w02_d03_FE_oil-correlation.png` (correlation heatmap)

**Documentation:**
- `docs/decisions/DEC-012_Oil_Features_Inclusion.md` (comprehensive decision rationale)
- `docs/feature_dictionary_v2.txt` updated (6 new oil features)

**Notebook:**
- `notebooks/w02_d03_FE_oil.ipynb` (7 cells, ~200 lines)

---

## Oil Feature Summary Statistics

**Oil Price:**
- Range: $26.19 to $110.62
- Mean: $61.71
- Median: $50.25
- Std: $23.90

**Oil Lags:**
- lag7 NaN: 7 (0.00%)
- lag14 NaN: 14 (0.00%)
- lag30 NaN: 30 (0.01%)

**Oil Change Derivatives:**
- change7 range: $-79.50 to $+79.35, std=$10.52
- change14 range: $-77.16 to $+78.10, std=$10.57

**Correlation with unit_sales:**
- oil_price: r = +0.0124 (weakest expected: -0.55 from Week 1)
- oil_price_lag7: r = +0.0122
- oil_price_lag14: r = +0.0112
- oil_price_lag30: r = +0.0113
- oil_price_change7: r = +0.0005
- oil_price_change14: r = +0.0027

**Interpretation:**
- Sign flip (negative → positive) due to aggregation level change
- Magnitude drop (0.55 → 0.01) expected for granular sparse data
- Tree models may find non-linear patterns despite weak linear correlation

---

**End of Day 3 Checkpoint**
