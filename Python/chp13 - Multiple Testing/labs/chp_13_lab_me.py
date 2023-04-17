# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 13.6 Lab: Multiple Testing

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats
from statsmodels.stats.multitest import multipletests

import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

# %load_ext rpy2.ipython

# %% [markdown]
# ## 13.6.1 Review of Hypothesis Tests

# %% language="R"
# set.seed(6)
# x <- matrix(rnorm(10 * 100), 10, 100)
# x[, 1:50] <- x[, 1:50] + 0.5

# %%
data = robjects.r("""
set.seed(6)
x <- matrix(rnorm(10 * 100), 10, 100)
x[, 1:50] <- x[, 1:50] + 0.5
x <- x
""")

x_array = np.array(data)
#x_array[:, 0:50] = x_array[:,0:50] + 0.5

# %% language="R"
# t.test(x[, 1], mu = 0)

# %%
scipy.stats.ttest_1samp(x_array[:, i], popmean=0)

# %% language="R"
# p.values <- rep(0, 100)
# for (i in 1:100)
#     p.values[i] <- t.test(x[, i], mu = 0)$p.value
# decision <- rep("Do not reject H0", 100)
# decision[p.values <= 0.05] <- "Reject H0"
# table(decision,
#      c(rep("H0 is False", 50), rep("H0 is True", 50))
#      )

# %%
p_values = np.array([])
decisions = np.array([])

for i in range(x_array.shape[1]):
    p_val = scipy.stats.ttest_1samp(x_array[:, i], popmean=0).pvalue
    p_values = np.append(p_values, p_val)
    if p_val <= 0.05:
        decision = 'Reject H0'
    else:
        decision = 'Do not reject H0'
    decisions = np.append(decisions, decision)
    
decisions_df = pd.DataFrame({"Decisions":decisions})
decisions_df['H0_truth_val'] = 'H0 is False'
decisions_df['H0_truth_val'].iloc[50:,] = 'H0 is True'

pd.crosstab(decisions_df['Decisions'], decisions_df['H0_truth_val'], values = decisions_df['H0_truth_val'], aggfunc='count')

# %% language="R"
# x <- matrix(rnorm(10 * 100), 10, 100)
# x[, 1:50] <- x[, 1:50] + 1
# for (i in 1:100)
#     p.values[i] <- t.test(x[, i], mu=0)$p.value
# decision <- rep("Do no reject H0", 100)
# decision[p.values <= 0.05] <- 'Reject H0'
# table(decision,
#      c(rep("H0 is False", 50), rep("H0 is True", 50))
#      )

# %%
x_array = np.random.normal(loc=0, scale=1, size=1000).reshape(10, 100)
x_array[:, 0:50] = x_array[:, 0:50] + 1

p_values = np.array([])
decisions = np.array([])

for i in range(x_array.shape[1]):
    p_val = scipy.stats.ttest_1samp(x_array[:, i], popmean=0).pvalue
    p_values = np.append(p_values, p_val)
    if p_val <= 0.05:
        decision = 'Reject H0'
    else:
        decision = 'Do not reject H0'
    decisions = np.append(decisions, decision)
    
decisions_df = pd.DataFrame({"Decisions":decisions})
decisions_df['H0_truth_val'] = 'H0 is False'
decisions_df['H0_truth_val'].iloc[50:,] = 'H0 is True'

pd.crosstab(decisions_df['Decisions'], decisions_df['H0_truth_val'], values = decisions_df['H0_truth_val'], aggfunc='count')

# %% [markdown]
# ## 13.6.2 The Family-Wise Error Rate

# %% language="R"
# m <- 1:500
# fwe1 <- 1 - (1 - 0.05)^m
# fwe2 <- 1 - (1 - 0.01)^m
# fwe3 <- 1 - (1 - 0.001)^m

# %%
m = np.arange(1,501)
fwe1 = 1 - (1 - 0.05)**m
fwe2 = 1 - (1 - 0.01)**m
fwe3 = 1 - (1 - 0.001)**m

# %% language="R"
# par(mfrow = c(1, 1))
# plot(m, fwe1, type = 'l', log = 'x', ylim = c(0, 1), col = 2,
#     ylab = 'Family - Wise Error Rate',
#     xlab = 'Number of Hypotheses')
# lines(m, fwe2, col = 4)
# lines(m, fwe3, col = 3)
# abline(h = 0.05, lty = 2)

# %%
plt.figure(figsize=(8,8))
plt.plot(fwe1, color='tomato')
plt.plot(fwe2, color='lightblue')
plt.plot(fwe3, color='lightgreen')
plt.xscale('log')
plt.xlabel("Number of Hypotheses")
plt.ylabel("Family - Wise Error Rate")
plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500], ['1', '2', '5', '10', '20', '50', '100', '200', '500'])
plt.axhline(0.05, color='black', linestyle='--', lw=1);

# %% language="R"
# library(ISLR2)
# fund.mini <- Fund[, 1:5]
# t.test(fund.mini[, 1], mu = 0)

# %% language="R"
# fund.pvalue <- rep(0, 5)
# for (i in 1:5)
#     fund.pvalue[i] <- t.test(fund.mini[, i], mu = 0)$p.value
# fund.pvalue

# %%
data = robjects.r("""
library(ISLR2)
fund <- Fund
""")
with localconverter(robjects.default_converter + pandas2ri.converter):
    fund_df = robjects.conversion.rpy2py(data)

# %%
fund_mini = fund_df.iloc[:, 0:5]
scipy.stats.ttest_1samp(fund_mini.iloc[:, 0], popmean = 0)

# %%
fund_p_values = np.array([])

for i in range(5):
    p_val = scipy.stats.ttest_1samp(fund_mini.iloc[:, i], popmean = 0).pvalue
    fund_p_values = np.append(fund_p_values, p_val)
fund_p_values

# %% language="R"
# p.adjust(fund.pvalue, method = 'bonferroni')

# %%
multipletests(fund_p_values, method='bonferroni')[1]

# %%
a = fund_p_values * 5
a[a > 1] = 1
a

# %% language="R"
# p.adjust(fund.pvalue, method = 'holm')

# %%
multipletests(fund_p_values, method='holm')[1]

# %% language="R"
# apply(fund.mini, 2, mean)

# %%
fund_mini.mean(axis=0)

# %% language="R"
# t.test(fund.mini[, 1], fund.mini[, 2], paired=T)

# %%
scipy.stats.ttest_rel(fund_mini.iloc[:, 0], fund_mini.iloc[:, 1])

# %% language="R"
# returns <- as.vector(as.matrix(fund.mini))
# manager <- rep(c('1', '2', '3', '4', '5'), rep(50, 5))
# a1 <- aov(returns ~ manager)
# TukeyHSD(x = a1)

# %%
res = scipy.stats.tukey_hsd(fund_mini['Manager1'], fund_mini['Manager2'], fund_mini['Manager3'], fund_mini['Manager4'], fund_mini['Manager5'])
print(res)

# %% language="R"
# plot(TukeyHSD(x = a1))

# %%
ci = res.confidence_interval()

# %%
plt.figure(figsize=(8,8))


# %%
from statsmodels.stats.multicomp import MultiComparison

# %%
MultiComparison(fund_mini['Manager1'], fund_mini['Manager2'], fund_mini['Manager3'], fund_mini['Manager4'], fund_mini['Manager5'])

# %%
from statsmodels.examples.try_tukey_hsd import cylinders, cyl_labels

# %%
res.plot_simultaneous()

# %% [markdown]
# ## 13.6.3 The False Discovery Rate

# %%

# %%

# %%

# %%

# %% [markdown]
# ## 13.6.4 A Re-Sampling Approach

# %%
