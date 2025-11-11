# Corporación Favorita Grocery Sales Forecasting
## Week 1 Project Plan: Exploration & Understanding (UPDATED)

**Prepared by:** Alberto Diaz Durana  
**Timeline:** Week 1 (5 working days, 20 hours total + 3 hours buffer available)  
**Phase:** Phase 1 - Exploration & Understanding  
**Next Phase:** Week 2 - Feature Development  
**Plan Version:** 2.0 (Updated after Day 2 completion)

---

## Plan Update Summary

**Changes from Version 1.0:**
- Days 1-2 completed ahead of schedule (5 hours vs 8 hours allocated)
- 3-hour buffer now available for expanded EDA (Days 3-5)
- Added 10 new analyses: store clustering, promotion effectiveness, Pareto analysis, etc.
- Established daily review checkpoints for scope adjustment
- Expanded from 8-step EDA to 18-step comprehensive analysis

**Status:**
- ✓ Day 1 Complete: Data inventory (11 Guayas stores, top-3 families identified)
- ✓ Day 2 Complete: Data filtering (300K representative sample created)
- → Day 3-5: Expanded EDA (quality + store + product + temporal + context)

---

## 1. Purpose

**Objective:**  
Conduct comprehensive exploratory data analysis on Guayas sample to understand data quality, temporal patterns, store performance, product dynamics, and external factors affecting sales.

**Business Value:**  
- Validate data quality for reliable forecasting
- Identify store-level performance patterns and opportunities
- Understand product sales dynamics (fast vs slow movers)
- Discover temporal patterns (seasonality, day-of-week effects, payday impacts)
- Assess promotion effectiveness and holiday impacts
- Build foundation for feature engineering and modeling phases

**Resources:**
- Time allocation: 15 hours (12 base + 3 buffer from Days 1-2 efficiency)
- Time buffer: Built into daily allocations (5 hours per day)
- Tools: Python, pandas, matplotlib, seaborn, statsmodels
- Data: guayas_sample_300k.pkl (300K rows, 11 stores, 2,296 items)

---

## 2. Inputs & Dependencies

### Primary Dataset
- **Source**: guayas_sample_300k.pkl (created in Day 2)
- **Characteristics**:
  - Rows: 300,000
  - Stores: 11 (Guayas region)
  - Families: 3 (GROCERY I, BEVERAGES, CLEANING)
  - Items: 2,296 unique
  - Date range: 2013-01-02 to 2017-08-15 (5 years)
  - Missing onpromotion: 18.57% (55,706 rows)
  - Negative sales: 13 rows (0.00%)

### Support Files
- stores.csv (54 stores, 5 columns)
- items.csv (4,100 items, 4 columns)
- oil.csv (1,218 records, 43 missing)
- holidays_events.csv (350 records)
- transactions.csv (83,488 records) - optional for Day 5

### Data Quality Assumptions
- Missing dates represent zero sales (not missing data)
- Negative sales are product returns (clip to zero)
- onpromotion NaN values assume no promotion (fill with False)
- Oil price gaps can be forward filled

---

## 3. Execution Timeline (Days 3-5)

| Day | Focus Area | Base Hours | Buffer | Total | Key Deliverables |
|-----|-----------|------------|--------|-------|------------------|
| 3 | Data Quality & Store Analysis | 4h | 1h | 5h | Clean data, store performance, item coverage |
| 4 | Temporal Patterns & Product Analysis | 4h | 1h | 5h | Features, seasonality, fast/slow movers |
| 5 | Context Analysis & Consolidation | 4h | 1h | 5h | Holidays, promotions, perishables, final export |
| **Total** | | **12h** | **3h** | **15h** | **guayas_prepared.csv + comprehensive EDA report** |

---

## 4. Detailed Deliverables

### Day 3 - Data Quality & Store Analysis
**Goal:** Clean data, analyze store performance, understand item coverage

**Total Time:** 5 hours (4h base + 1h buffer)

#### Part 0: Data Loading & Initial Checks (30 min)
**Objective:** Load filtered sample and verify characteristics
**Activities:**
- Load guayas_sample_300k.pkl
- Check shape, dtypes, memory usage
- Display first/last rows
- Summary statistics
**Deliverables:**
- Loaded DataFrame
- Initial quality report
- Baseline statistics

#### Part 1: Missing Value Analysis (45 min)
**Objective:** Detect and handle missing values across all columns
**Activities:**
- Count NaN per column
- Visualize missing patterns (heatmap if useful)
- Fill onpromotion NaN with False (0.0)
- Document handling decisions in decision log
**Deliverables:**
- Missing value report
- Cleaned onpromotion column
- Decision log entry (DEC-003: Fill onpromotion with False)

#### Part 2: Outlier Detection (1 hour)
**Objective:** Identify and handle outliers in unit_sales
**Activities:**
- Detect negative sales (13 rows identified)
- Clip negative values to 0
- Calculate Z-scores by store-item groups
- Flag extreme outliers (Z > 3.0)
- Visualize outlier distribution (boxplot, histogram)
- Document outlier handling decision
**Deliverables:**
- Outlier analysis report
- Cleaned unit_sales (negatives → 0)
- Outlier flags (for potential further investigation)
- Decision log entry (DEC-004: Clip negative sales to zero)

#### Part 3: Store-Level Performance Analysis (1.5 hours) ← NEW
**Objective:** Compare performance across 11 Guayas stores
**Activities:**
- Calculate total sales by store
- Sales by store type (A/B/C/D/E comparison)
- Sales by city (Guayaquil vs Daule/Playas/Libertad)
- Sales by cluster (1, 3, 6, 10, 17)
- Visualize store performance (bar charts, heatmap)
- Identify top/bottom performing stores
**Deliverables:**
- Store performance report
- Store type comparison chart
- City-level sales comparison
- Cluster analysis visualization

#### Part 4: Item Coverage Analysis (1 hour) ← NEW
**Objective:** Understand product availability across stores
**Activities:**
- Create store-item availability matrix (which items sold where)
- Calculate coverage per store (% of 2,296 items sold)
- Identify items sold in all stores vs few stores
- Zero-sales preliminary analysis (distinguish no-demand from stockouts)
**Deliverables:**
- Item coverage matrix
- Store coverage report
- Universal vs niche items list

#### Part 5: Calendar Gap Filling (30 min)
**Objective:** Create complete daily index for each store-item pair
**Activities:**
- Convert date to datetime
- Identify missing dates per (store_nbr, item_nbr) group
- Fill gaps with unit_sales=0 (asfreq daily)
- Validate no missing dates remain
- Compare row count before/after
**Deliverables:**
- Complete daily calendar DataFrame
- Gap filling validation report
- Before/after row count comparison

#### Part 6: Date Feature Extraction (30 min)
**Objective:** Extract basic temporal features
**Activities:**
- Create year, month, day columns
- Create day_of_week (0=Monday, 6=Sunday)
- Validate feature distributions
**Deliverables:**
- 4 new date-based features
- Feature distribution summary

---

#### **End-of-Day 3 Checkpoint** ⚠️

**Review Questions:**
1. Did we complete all Day 3 parts within 5 hours?
2. Are there any blockers or unexpected data issues?
3. Should we adjust Day 4-5 scope based on findings?
4. Do we need to add/remove analyses for remaining days?

**Adjustment Options:**
- If ahead of schedule → Add deeper analysis to Day 4-5
- If behind schedule → Deprioritize lower-value analyses
- If data issues found → Allocate more time to cleaning

**Document decisions in decision log and update Day 4 plan accordingly.**

---

### Day 4 - Temporal Patterns & Product Analysis
**Goal:** Engineer features, visualize temporal patterns, analyze products

**Total Time:** 5 hours (4h base + 1h buffer)

#### Part 0: Feature Engineering - Rolling Statistics (1 hour)
**Objective:** Calculate rolling averages for smoothing
**Activities:**
- Sort by (item_nbr, store_nbr, date)
- Calculate 7/14/30-day rolling means per group
- Handle edge cases (min_periods=1)
- Visualize raw vs smoothed for sample items
**Deliverables:**
- unit_sales_7d_avg, unit_sales_14d_avg, unit_sales_30d_avg columns
- Smoothing visualization (5-10 example items)
- Explanation of rolling window behavior

#### Part 1: Time Series Visualization (1.5 hours)
**Objective:** Visualize overall trends and seasonal patterns
**Activities:**
- Aggregate total sales by date
- Plot time series (2013-2017)
- Identify trend, seasonality, anomalies
- Create year-month heatmap
- Annotate major patterns (December peaks, Q1 lulls)
**Deliverables:**
- Total sales time series plot
- Year-month heatmap
- Pattern interpretation report
- Identified anomalies (e.g., August 2017 drop)

#### Part 2: Autocorrelation Analysis (30 min)
**Objective:** Assess temporal dependence for lag feature guidance
**Activities:**
- Aggregate daily sales
- Plot autocorrelation (pandas.plotting.autocorrelation_plot)
- Interpret lag significance
- Document findings for Week 2 feature engineering
**Deliverables:**
- Autocorrelation plot
- Lag analysis interpretation
- Recommendations for lag features (Week 2)

#### Part 3: Temporal Deep Dive (1.5 hours) ← NEW/EXPANDED
**Objective:** Uncover day-of-week, monthly, and payday patterns
**Activities:**
- Day-of-week patterns by family (do weekends differ?)
- Monthly seasonality by family (family-specific patterns?)
- Weekend vs weekday sales comparison
- Payday effects analysis (1st and 15th of month)
- End-of-month patterns (30th, 31st)
- Visualize with grouped bar charts and line plots
**Deliverables:**
- Day-of-week analysis by family
- Monthly seasonality comparison
- Payday effect report
- Temporal pattern summary

#### Part 4: Product Analysis (1 hour) ← NEW
**Objective:** Identify fast vs slow movers and sales concentration
**Activities:**
- Calculate sales velocity per item (total sales / days active)
- Classify items: fast movers (top 20%), slow movers (bottom 20%)
- Pareto analysis: 80% of sales from X% of items
- Sales concentration by family
- Identify hero products vs long-tail
**Deliverables:**
- Fast/slow mover classification
- Pareto chart (cumulative sales by item)
- Sales concentration report
- Hero products list per family

---

#### **End-of-Day 4 Checkpoint** ⚠️

**Review Questions:**
1. Did temporal patterns reveal unexpected insights?
2. Are fast/slow movers clearly distinguishable?
3. Should Day 5 focus more on promotions or external factors?
4. Do we have enough time for transaction analysis?

**Adjustment Options:**
- If rich temporal patterns found → Spend more time on seasonality
- If product analysis reveals issues → Investigate further
- If ahead of schedule → Add transaction analysis to Day 5

**Document decisions and adjust Day 5 priorities.**

---

### Day 5 - Context Analysis & Consolidation
**Goal:** Analyze holidays, promotions, perishables, oil; export final dataset

**Total Time:** 5 hours (4h base + 1h buffer)

#### Part 0: Holiday Impact Analysis (1 hour)
**Objective:** Assess how holidays affect sales
**Activities:**
- Merge holidays_events.csv with train on date
- Group by holiday type, calculate mean sales
- Plot bar chart (avg sales by holiday type)
- Analyze pre/post holiday effects (±3 days)
- Filter to Guayas-specific holidays (locale = Guayas)
- Identify significant impacts
**Deliverables:**
- Holiday analysis report
- Bar chart (sales by holiday type)
- Pre/post holiday pattern analysis
- Key findings (Work Day vs Holiday vs Transfer)

#### Part 1: Promotion Analysis (1.5 hours) ← NEW/EXPANDED
**Objective:** Measure promotion effectiveness and interactions
**Activities:**
- Compare sales: on promotion vs not on promotion
- Calculate sales lift (% increase when promoted)
- Promotion frequency by family and store
- Promotion × holiday interaction (are promoted holidays different?)
- Non-promoted baseline trends
- Visualize with grouped bar charts and line plots
**Deliverables:**
- Promotion effectiveness report (sales lift %)
- Promotion frequency by family/store
- Promotion × holiday interaction analysis
- Baseline trend (non-promoted sales)

#### Part 2: Perishable Deep Dive (1 hour) ← EXPANDED
**Objective:** Compare perishable vs non-perishable with waste indicators
**Activities:**
- Merge items.csv (perishable flag)
- Group by perishable, sum unit_sales
- Calculate perishable percentage by family
- Perishable sales velocity (faster turnover expected)
- Identify high-volatility perishables (waste risk)
- Plot comparison bar chart
- Interpret business implications (inventory risk)
**Deliverables:**
- Perishable vs non-perishable comparison
- Bar chart visualization
- High-volatility items list (waste risk)
- Business interpretation (inventory management)

#### Part 3: External Factors & Zero-Sales (1 hour)
**Objective:** Investigate oil prices and zero-sales patterns
**Activities:**
- Merge oil.csv with aggregated daily sales
- Plot dual-axis chart (oil price + sales over time)
- Calculate correlation coefficient
- Visual interpretation
- Document findings (likely weak correlation)
- Analyze zero-sales patterns (stockouts vs no-demand)
**Deliverables:**
- Oil price vs sales plot
- Correlation analysis
- Interpretation report
- Decision on oil feature inclusion (Week 2)
- Zero-sales pattern report

#### Part 4: Optional Transaction Analysis (30 min if time permits)
**Objective:** Understand traffic patterns using transactions.csv
**Activities:**
- Merge transactions.csv (Guayas stores only)
- Calculate sales per transaction
- Traffic patterns (transactions by day-of-week)
- Basket size trends over time
**Deliverables:**
- Transaction analysis report (if completed)
- Sales per transaction metric

#### Part 5: Export & Documentation (1 hour)
**Objective:** Save cleaned dataset and summarize Week 1
**Activities:**
- Export final DataFrame to data/processed/guayas_prepared.csv
- Export to pickle for Week 2
- Update decision log with all Week 1 decisions
- Create Week 1 summary report (key findings, visualizations)
- Document feature dictionary (all engineered features)
- Commit all notebooks and docs to Git
**Deliverables:**
- `guayas_prepared.csv` (cleaned, featured, calendar-filled)
- `guayas_prepared.pkl`
- Week 1 summary report
- Feature dictionary
- Updated decision log
- Git commit with message "Week 1: Comprehensive EDA complete"

---

#### **End-of-Day 5 Checkpoint** ⚠️

**Final Review Questions:**
1. Are all 18 EDA steps complete?
2. Is guayas_prepared.csv ready for Week 2?
3. Did we identify all major data quality issues?
4. Are key findings documented and actionable?

**Week 1 Completion Criteria:**
- [X] All data quality issues handled
- [X] Store, product, and temporal patterns documented
- [X] Holiday and promotion effects quantified
- [X] Final dataset exported with all features
- [X] Decision log complete with ≥4 entries
- [X] Summary report with key insights

**Prepare handoff to Week 2: Feature Development**

---

## 5. Readiness Checklist (for Week 2 Transition)

### Required Inputs
- [ ] Clean dataset (guayas_prepared.csv, ~300K+ rows after calendar fill)
- [ ] Validated cohort definition (11 Guayas stores, top-3 families)
- [ ] Data quality report showing <2% missing values post-cleaning

### Completion Criteria
- [ ] All missing values handled (onpromotion filled, negatives clipped)
- [ ] Calendar gaps filled (complete daily index per store-item)
- [ ] Features created (date components, 7/14/30-day rolling avgs)
- [ ] Temporal patterns documented (trend, seasonality, autocorrelation, day-of-week)
- [ ] Store performance analyzed (11 stores compared)
- [ ] Product dynamics understood (fast/slow movers, Pareto)
- [ ] External factors analyzed (holidays, promotions, perishables, oil)

### Quality Checks
- [ ] Data types validated (datetime for date, bool for onpromotion, int/float for numeric)
- [ ] Range checks passed (unit_sales ≥ 0 after cleaning)
- [ ] No duplicate (store_nbr, item_nbr, date) rows
- [ ] Row count matches expected (complete calendar × store-item combinations)
- [ ] All 11 stores present in final dataset
- [ ] All top-3 families present in final dataset

### Deliverables Ready
- [ ] All 3 EDA notebooks run without errors (Days 3-5)
- [ ] `guayas_prepared.csv` validated and exported
- [ ] Decision log populated (≥4 entries)
- [ ] Week 1 summary report complete with visualizations
- [ ] Feature dictionary documented

### Next Phase Readiness
After completing Week 1, you will have:
- Clean, gap-filled, featured dataset ready for advanced feature engineering
- Deep understanding of temporal patterns (seasonality, trend, autocorrelation, day-of-week)
- Store-level insights (performance, clustering opportunities)
- Product insights (fast/slow movers, sales concentration)
- Documented data quality decisions and outlier handling
- Identified external factors (holidays, promotions, perishables) for Week 2 feature creation
- Baseline visualizations for comparison in later phases

---

## 6. Success Criteria

### Quantitative
- Dataset: ~300K+ rows after calendar fill (complete daily coverage)
- Missing values: <1% in critical columns post-cleaning
- Calendar completeness: 100% daily coverage per store-item
- Outliers flagged: document count and % of total
- ≥4 major decisions logged with rationale
- 3 notebooks created (Days 3-5, ~400-500 lines each, 6-7 sections per notebook)
- ≥15 visualizations (time series, heatmaps, bar charts, autocorrelation, etc.)
- 18 EDA steps completed (original 8 + 10 new analyses)

### Qualitative
- Data quality issues understood and documented
- Temporal patterns clearly identified (trend, seasonality, day-of-week, payday effects)
- Store performance patterns interpretable by business stakeholders
- Product dynamics understood (fast/slow movers actionable for inventory)
- Holiday, promotion, and perishable impacts interpretable
- Decision rationale clear for future reference
- Notebooks readable by technical reviewer

### Technical
- All notebooks run end-to-end without errors
- Consistent file paths (relative, constants defined)
- Clean code (no debug/commented code in final version)
- Git repository organized (clear commit messages)
- Reproducibility: someone else can run notebooks and get same results

---

## 7. Documentation & Ownership

### Version Control
- **Repository**: retail_demand_analysis (GitHub)
- **Branch**: main
- **Commit frequency**: Daily (end of each day's work)
- **Commit message format**: "Day X: [accomplishment summary]"

### Key Documents
- `Week1_ProjectPlan.md` (this document - v2.0 updated)
- `decision_log.md` (track analytical choices - target ≥4 entries)
- `data_inventory.md` (dataset characteristics - completed Day 1)
- `Week1_Summary_Report.md` (to be created Day 5)
- `feature_dictionary.md` (to be created Day 5)
- `README.md` (project overview - completed Day 1)

### Assumptions
- 300K sample is representative of full Guayas dataset
- Top-3 families capture sufficient product diversity (58.4% of items)
- Zero-filling missing dates is appropriate (no stockouts, true zero demand)
- Clipping negative sales to zero is acceptable for forecasting (returns handled)
- Oil price correlation assumed weak (verify during Day 5)
- 3-hour buffer sufficient for expanded analyses

### Limitations
- Sample size may miss rare events (low-frequency items)
- Guayas focus excludes other regions (cannot generalize nationally)
- Top-3 families may not represent all product dynamics
- Historical data only (no forward-looking external signals)
- Transaction analysis optional (depends on Day 5 time availability)

### Ownership
- **Project Lead**: Alberto Diaz Durana
- **Stakeholders**: Academic advisor, peer reviewers
- **Timeline**: Week 1 of 4-week project

---

## 8. Risk Management

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Calendar fill creates too many rows (>1M) | Medium | Medium | Sample further if memory issues arise |
| Store analysis reveals data quality issues | Low | High | Allocate Day 3 buffer to investigate |
| Temporal patterns weak or unclear | Low | Medium | Focus on visible patterns, document limitations |
| Promotion analysis inconclusive | Low | Low | Document finding, still valuable for Week 2 |
| Time overrun with expanded scope | Medium | Medium | Use daily checkpoints to adjust, deprioritize low-value analyses |
| Transaction data not useful | Low | Low | Skip transaction analysis if time constrained |

---

## 9. Expected Outcomes

### Expected Outcomes Table

| Metric | Before Week 1 | After Days 1-2 | After Week 1 (Days 3-5) | Target Met |
|--------|---------------|----------------|-------------------------|------------|
| Dataset size | 125M rows | 300K rows (sample) | 300K-1M rows (after calendar fill) | Yes |
| Missing values | Unknown | 18.57% onpromotion | <1% in all columns | Yes |
| Calendar completeness | Gaps present | Gaps present | 100% daily coverage | Yes |
| Features | Raw columns | 9 columns | +5-7 engineered features | Yes |
| Visualizations | None | None | 15+ plots | Yes |
| Store insights | None | 11 stores identified | Performance, clustering, coverage | Yes |
| Product insights | None | 2,296 items | Fast/slow movers, Pareto | Yes |
| Temporal insights | None | Date range known | Seasonality, day-of-week, payday | Yes |
| Context analysis | None | None | Holidays, promotions, perishables | Yes |
| Documentation | Basic | Inventory | Decision log + summary + features | Yes |
| **Summary** | **Raw data** | **Filtered sample** | **Analysis-ready, fully understood** | **10/10 targets** |

### Key Benefits
- Validated dataset ready for advanced feature engineering
- Deep understanding of temporal dynamics (guides model selection)
- Store-level insights (actionable for business operations)
- Product dynamics (inventory optimization opportunities)
- Promotion effectiveness (ROI quantification)
- Documented decisions (avoid rework, enable peer review)
- Professional repository structure (portfolio-ready)

---

## 10. Communication Plan

### Daily Progress Updates
- **Frequency**: End of each day
- **Format**: Brief summary (5 minutes to write)
- **Content**: Accomplishments, blockers, next day plan, checkpoint decisions
- **Audience**: Self (for tracking), advisor (if requested)

### Daily Checkpoints (Days 3-5)
- **Timing**: End of each day (last 15 minutes)
- **Format**: Structured review using checkpoint questions
- **Content**: 
  - Time tracking (actual vs allocated)
  - Scope completion (all parts done?)
  - Findings quality (insights actionable?)
  - Adjustment decisions (add/remove analyses)
- **Output**: Update next day's plan if needed

### Week-End Summary
- **Timing**: Friday end-of-day (Day 5)
- **Format**: Email or document (~2 pages)
- **Content**: Week 1 accomplishments, key findings, Week 2 preview
- **Audience**: Advisor, peer reviewers

---

## 11. Daily Checkpoint Template

Use this template at the end of Days 3, 4, and 5:

```markdown
## Day X Checkpoint (YYYY-MM-DD)

### Time Tracking
- Allocated: X hours
- Actual: X hours
- Variance: +/- X hours
- Reason for variance: [brief explanation]

### Scope Completion
- [ ] Part 0: [Status]
- [ ] Part 1: [Status]
- [ ] Part 2: [Status]
- [ ] Part 3: [Status]
- [ ] Part 4: [Status]
- [ ] Part 5: [Status]

### Key Findings
1. [Most important finding]
2. [Second most important finding]
3. [Unexpected discovery]

### Quality Assessment
- Data quality: [Good/Fair/Needs work]
- Insights actionability: [High/Medium/Low]
- Visualizations clarity: [Excellent/Good/Needs improvement]

### Blockers & Issues
- [List any blockers encountered]
- [Data quality issues found]
- [Technical challenges]

### Adjustment Decisions
For Day X+1:
- [ ] Keep plan as-is
- [ ] Add analysis: [specify]
- [ ] Remove analysis: [specify]
- [ ] Reallocate time: [specify]

### Next Day Preview
Day X+1 focus:
1. [Primary objective]
2. [Secondary objective]
3. [Contingency plan]

### Decision Log Updates
- DEC-00X: [Decision made today]
```

---

## Summary

Week 1 (Updated) establishes comprehensive understanding of Guayas sales data by:
1. Validating data quality and handling missing values/outliers (Days 1-3)
2. Analyzing store performance, product dynamics, and item coverage (Day 3)
3. Engineering features and uncovering temporal patterns (Day 4)
4. Investigating external factors: holidays, promotions, perishables (Day 5)
5. Exporting analysis-ready dataset with complete documentation (Day 5)

**Expanded scope uses 3-hour buffer to add 10 high-value analyses:**
- Store-level performance comparison
- Store clustering and city-level patterns  
- Item coverage matrix
- Fast/slow mover classification
- Pareto (80/20) analysis
- Day-of-week patterns by family
- Payday effects analysis
- Promotion effectiveness measurement
- Promotion × holiday interactions
- Perishable waste risk indicators

**Daily checkpoints ensure agile adaptation** based on actual progress and findings, allowing intelligent reprioritization throughout Days 3-5.

Upon completion, Week 2 (Feature Development) can begin with:
- Clean, featured dataset (guayas_prepared.csv)
- Deep understanding of patterns (store, product, temporal, context)
- Clear direction for advanced feature engineering
- Documented decisions and actionable insights

**Week 1 deliverables enable Week 2-4 success.**

---

**End of Week 1 Project Plan (Version 2.0)**

**Last Updated:** 2025-11-11 (After Day 2 completion)
