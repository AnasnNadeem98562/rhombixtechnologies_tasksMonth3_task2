import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.multicomp import pairwise_tukeyhsd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

ALPHA = 0.05  # significance level: we accept a 5% chance of a false positive

df = pd.read_csv("campaign_data.csv", parse_dates=["date"])
print(df.shape)
print(df.head())

# %% STEP 2 - Check & understand the data
print("Missing values:\n", df.isna().sum())
print("\nUsers per campaign:\n", df["campaign"].value_counts())

summary = df.groupby("campaign").agg(
    users=("user_id", "count"),
    click_rate=("clicked", "mean"),
    conversion_rate=("converted", "mean"),
    avg_time_on_site=("time_on_site_sec", "mean"),
    total_revenue=("order_value", "sum"),
)
summary["revenue_per_user"] = summary["total_revenue"] / summary["users"]
print("\nCampaign summary:\n", summary.round(4))

# B looks better on paper - but is the difference REAL or just random chance?
# That is what hypothesis testing answers.


def verdict(p):
    return "SIGNIFICANT -> reject H0" if p < ALPHA else "NOT significant -> fail to reject H0"


# %% STEP 3 - Chi-square test: is conversion rate different between A and B?
# H0: Conversion rate does not depend on campaign
# H1: Conversion rate depends on campaign
table = pd.crosstab(df["campaign"], df["converted"])
print("Contingency table:\n", table)

chi2, p_chi, dof, expected = stats.chi2_contingency(table)
print(f"\nChi-square = {chi2:.3f}, dof = {dof}, p-value = {p_chi:.6f}")
print("Result:", verdict(p_chi))

# %% STEP 4 - Two-proportion z-test (Statsmodels) + confidence intervals
# Same question as chi-square, but tells us the direction and size of the lift.
conv = df.groupby("campaign")["converted"].agg(["sum", "count"])
z, p_z = proportions_ztest(conv["sum"], conv["count"])
low, high = proportion_confint(conv["sum"].values, conv["count"].values, alpha=ALPHA)

rate_a = conv.loc["A", "sum"] / conv.loc["A", "count"]
rate_b = conv.loc["B", "sum"] / conv.loc["B", "count"]
lift = (rate_b - rate_a) / rate_a * 100

print(f"Conversion A: {rate_a:.2%}  (95% CI {low[0]:.2%} - {high[0]:.2%})")
print(f"Conversion B: {rate_b:.2%}  (95% CI {low[1]:.2%} - {high[1]:.2%})")
print(f"Relative lift of B over A: {lift:.1f}%")
print(f"z = {z:.3f}, p-value = {p_z:.6f} -> {verdict(p_z)}")

# %% STEP 5 - Independent t-test: is time on site different?
# H0: Mean time on site is equal for A and B
# Welch's t-test (equal_var=False) is safer because it does not assume equal variances.
time_a = df.loc[df["campaign"] == "A", "time_on_site_sec"]
time_b = df.loc[df["campaign"] == "B", "time_on_site_sec"]

lev_stat, lev_p = stats.levene(time_a, time_b)
print(f"Levene's test for equal variances: p = {lev_p:.4f}")

t, p_t = stats.ttest_ind(time_a, time_b, equal_var=False)
print(f"Mean time A: {time_a.mean():.1f}s | Mean time B: {time_b.mean():.1f}s")
print(f"t = {t:.3f}, p-value = {p_t:.6f} -> {verdict(p_t)}")

# %% STEP 6 - t-test on order value (only users who bought)
# H0: Average order value is the same for A and B buyers
aov_a = df.loc[(df["campaign"] == "A") & (df["converted"] == 1), "order_value"]
aov_b = df.loc[(df["campaign"] == "B") & (df["converted"] == 1), "order_value"]

t2, p_t2 = stats.ttest_ind(aov_a, aov_b, equal_var=False)
print(f"Avg order value A: PKR {aov_a.mean():,.0f} | B: PKR {aov_b.mean():,.0f}")
print(f"t = {t2:.3f}, p-value = {p_t2:.6f} -> {verdict(p_t2)}")

# Effect size (Cohen's d) - how big is the difference in practical terms?
pooled_sd = np.sqrt((aov_a.var() + aov_b.var()) / 2)
print(f"Cohen's d = {(aov_b.mean() - aov_a.mean()) / pooled_sd:.2f} (0.2 small, 0.5 medium, 0.8 large)")

# %% STEP 7 - One-way ANOVA: does time on site differ across devices?
# ANOVA compares 3+ groups at once. H0: all device groups have the same mean.
groups = [g["time_on_site_sec"].values for _, g in df.groupby("device")]
f, p_anova = stats.f_oneway(*groups)
print(df.groupby("device")["time_on_site_sec"].mean().round(1))
print(f"\nOne-way ANOVA (SciPy): F = {f:.3f}, p-value = {p_anova:.6f} -> {verdict(p_anova)}")

# Same test with Statsmodels - gives a full ANOVA table
model = ols("time_on_site_sec ~ C(device)", data=df).fit()
print("\n", sm.stats.anova_lm(model, typ=2))

# %% STEP 8 - Post-hoc Tukey test: WHICH devices differ?
# ANOVA only says "at least one group is different". Tukey tells us which pairs.
tukey = pairwise_tukeyhsd(df["time_on_site_sec"], df["device"], alpha=ALPHA)
print(tukey)

# %% STEP 9 - Two-way ANOVA: campaign x device
# Does the campaign effect hold on every device, or only some? (interaction term)
model2 = ols("time_on_site_sec ~ C(campaign) * C(device)", data=df).fit()
print(sm.stats.anova_lm(model2, typ=2))

# %% STEP 10 - Chi-square by segment: does B win on every device?
rows = []
for dev, g in df.groupby("device"):
    _, p, _, _ = stats.chi2_contingency(pd.crosstab(g["campaign"], g["converted"]))
    r = g.groupby("campaign")["converted"].mean()
    rows.append({"device": dev, "conv_A": f"{r['A']:.2%}", "conv_B": f"{r['B']:.2%}",
                 "p_value": round(p, 4), "significant": p < ALPHA})
print(pd.DataFrame(rows))

# %% STEP 11 - Charts
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = ["#4C78A8", "#F58518"]

# Conversion rate with 95% CI error bars
rates = [rate_a, rate_b]
errors = [[rate_a - low[0], rate_b - low[1]], [high[0] - rate_a, high[1] - rate_b]]
axes[0].bar(["A", "B"], [r * 100 for r in rates], color=colors,
            yerr=np.array(errors) * 100, capsize=8)
axes[0].set_title("Conversion rate (95% CI)")
axes[0].set_ylabel("%")

# Time on site distribution
axes[1].boxplot([time_a, time_b], tick_labels=["A", "B"], showfliers=False)
axes[1].set_title("Time on site (seconds)")

# Conversion by device
dev_rates = df.pivot_table(index="device", columns="campaign", values="converted") * 100
dev_rates.plot(kind="bar", ax=axes[2], color=colors, rot=0)
axes[2].set_title("Conversion rate by device (%)")
axes[2].set_xlabel("")

plt.tight_layout()
plt.savefig("ab_test_results.png", dpi=150)
plt.show()
print("Chart saved: ab_test_results.png")

# %% STEP 12 - Final conclusion
results = pd.DataFrame([
    ["Chi-square", "Conversion rate", p_chi],
    ["Two-proportion z-test", "Conversion rate", p_z],
    ["Welch t-test", "Time on site", p_t],
    ["Welch t-test", "Order value (buyers)", p_t2],
    ["One-way ANOVA", "Time on site by device", p_anova],
], columns=["Test", "Metric", "p_value"])
results["Significant (p<0.05)"] = results["p_value"] < ALPHA
results["p_value"] = results["p_value"].map(lambda x: f"{x:.6f}")
print(results.to_string(index=False))
results.to_csv("hypothesis_test_results.csv", index=False)

winner = "B" if rate_b > rate_a and p_z < ALPHA else ("A" if rate_a > rate_b and p_z < ALPHA else "No clear winner")
print(f"\nRECOMMENDATION: Campaign {winner}")
if winner == "B":
    print(f"Campaign B converts {lift:.1f}% better than A, and the difference is statistically significant.")
    print("Business decision: roll out Campaign B.")
elif winner == "A":
    print("Campaign A converts significantly better. Business decision: keep Campaign A.")
else:
    print("The difference could be random chance. Collect more data before deciding.")
