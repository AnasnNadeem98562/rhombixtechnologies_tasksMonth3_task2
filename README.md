# Project 2: A/B Testing & Hypothesis Testing

Comparing two marketing campaigns with statistical tests (Chi-square, t-test and ANOVA) using SciPy and Statsmodels, to decide which campaign performed better on evidence rather than guesswork.

## Objective
Determine whether Campaign B truly outperforms Campaign A, or whether the observed difference could be due to random chance.

## Dataset
`campaign_data.csv`: synthetic data for 6,000 users, each shown Campaign A or B.
Columns: `campaign`, `date`, `device`, `region`, `age_group`, `clicked`, `converted`, `time_on_site_sec`, `order_value`

## Statistical Tests Performed
| Test | Purpose |
|---|---|
| Chi-square test | Does conversion rate depend on the campaign? |
| Two-proportion z-test | Size of the lift, with 95% confidence intervals |
| Welch's t-test | Compare time on site and average order value |
| One-way ANOVA + Tukey HSD | Does time on site differ across devices? |
| Two-way ANOVA | Interaction between campaign and device |
| Segmented Chi-square | Does Campaign B win on every device? |

Significance level: **α = 0.05**. If p < 0.05, the null hypothesis (no difference) is rejected.

## Key Results
| Metric | Campaign A | Campaign B | p-value | Significant? |
|---|---|---|---|---|
| Conversion rate | 9.54% | 13.01% | 0.00003 | ✅ Yes |
| Time on site | 184 s | 208 s | < 0.0001 | ✅ Yes |
| Avg order value | PKR 3,162 | PKR 3,445 | 0.00003 | ✅ Yes |

- Campaign B delivered a **36% relative lift** in conversions
- ANOVA showed **no significant difference** in time on site across devices (p = 0.60)
- B outperformed A on Mobile and Tablet. Desktop was higher but not significant (p = 0.09)

## Conclusion
Campaign B performs significantly better on conversion, engagement and order value. **Recommendation: roll out Campaign B.**

## Output Files
- `ab_test_results.png`: conversion rate with confidence intervals, time on site, conversion by device
- `hypothesis_test_results.csv`: summary of all test results

## Tools
Python · Pandas · SciPy · Statsmodels · Matplotlib · VS Code

## How to Run
```bash
pip install pandas numpy scipy statsmodels matplotlib
python project2_ab_testing.py
```
