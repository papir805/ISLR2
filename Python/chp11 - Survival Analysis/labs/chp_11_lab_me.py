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
# # 11.8 Lab: Survival Analysis

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import survive

from sksurv.compare import compare_survival

# %load_ext rpy2.ipython

# %% [markdown]
# ## 11.8.1 Brain Cancer Data

# %% language="R"
# library(ISLR2)
# names(BrainCancer)

# %%
data = robjects.r("""
library(ISLR2)
xdata <- BrainCancer
""")
with localconverter(robjects.default_converter + pandas2ri.converter):
    brain_cancer_df = robjects.conversion.rpy2py(data)
    
brain_cancer_df.columns.values.tolist()

# %% language="R"
# attach(BrainCancer)
# table(sex)

# %%
brain_cancer_df['sex'].value_counts().to_frame().T

# %% language="R"
# table(diagnosis)

# %%
brain_cancer_df['diagnosis'].value_counts().sort_index().to_frame().T

# %% language="R"
# table(status)

# %%
brain_cancer_df['status'].value_counts().to_frame().T

# %% [markdown]
# ### Kaplan Meier Survival Curve

# %% language="R"
# library(survival)
# fit.surv <- survfit(Surv(time, status) ~ 1)
# plot(fit.surv, xlab = 'Months',
#     ylab = 'Estimated Probability of Survival')

# %% [markdown]
# #### Different Python Libraries
# In Python, there are several libraries that can replicate parts of what R does in this lab, however I was unable to find any single library that did *all* of the things that R did.  The closest to an exact match was the `lifelines` library, as it could fit both KaplanMeier and CoxPH models, however there were a few areas that it fell short and some of the other libraries were better.
#
# Another Python library, `survive`, could only fit KaplanMeier models, but the use of this library felt very similar to that of R.  For instance, there is a `survive.SurvivalData()` object that one can create survival data, based on time and status information from your dataset, which is essentially the same survival data the `Surv()` function in R generates.  Furthermore, when fitting the model in `survive`, one can group by values of a predictor using the `group=` parameter, which will generate survival curves for each level of that group, much like one can do using the R style formula in the `survfit()` function.
#
# Lastly, there's a Python library I didn't explore as much as the other two called `scikit-survival`.  This library was able to mimic the behavior of R's `survdiff()` function for CoxPH models where `lifelines` and `survive` could not.

# %% [markdown]
# ##### Using Lifelines

# %%
kmf_ll = KaplanMeierFitter()
kmf_ll.fit(durations=brain_cancer_df['time'], event_observed=brain_cancer_df['status'])

fig, ax = plt.subplots(1,1, figsize=(8,6))

kmf_ll.plot_survival_function()
plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Estimated Probability of Survival')
ax.get_legend().remove();

# %% [markdown]
# By default, `lifelines` represents the uncertainty of the estimate using shading on the graph to represent the confidence interval for each timepoint, which is different behavior than in R.  By tinkering with `matplotlib` some, this can be overcome, but would be overkill and unncessary.

# %% [markdown]
# ##### Using Survive

# %%
brain_cancer_df['status'] = np.float64(brain_cancer_df['status'])

kmf_surv = survive.KaplanMeier()

surv_data = survive.SurvivalData(time=brain_cancer_df['time'], status=brain_cancer_df['status'])

kmf_surv_fit = kmf_surv.fit(surv_data)

fig, ax = plt.subplots(1,1, figsize=(8,6))

kmf_surv_fit.plot(ci_style='lines', ax=ax)
plt.xlabel('Months')
plt.ylabel('Estimated Probability of Survival');

# %% [markdown]
# `survive` represents the confidence intervals using dashed lines as opposed to the shading of `lifelines`.  By default, the curve also shows vertical bars representing censoring events, something that R doesn't do.

# %% [markdown]
# #### By Gender

# %% language="R"
# fit.sex <- survfit(Surv(time, status) ~ sex)
# plot(fit.sex, xlab = 'Months',
#     ylab = 'Estimated Probability of Survival', col = c(2,4))
# legend('bottomleft', levels(sex), col = c(2,4), lty = 1)

# %% [markdown]
# ##### Using Lifelines

# %%
fig, ax = plt.subplots(1,1, figsize=(8,6))

male = (brain_cancer_df['sex'] == 'Male')
female = (brain_cancer_df['sex'] == 'Female')

kmf_ll.fit(brain_cancer_df['time'][male], event_observed=brain_cancer_df['status'][male], label='Male')
kmf_ll.survival_function_.plot(ax=ax)

kmf_ll.fit(brain_cancer_df['time'][female], event_observed=brain_cancer_df['status'][female], label='Female')
kmf_ll.survival_function_.plot(ax=ax)

plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Estimated Probability of Survival');

# %% [markdown]
# With `lifelines` it's also possible to plot the survival curve for different levels of a predictor, such as gender in the case of the graph above, and without any confidence intervals.  Unfortunately, `lifelines` won't automatically generate a plot for each level of the predictor though.  Instead, we have to partition the dataset by level, fit the model once for each partition of the data, and then plot the survival curve on a shared axis.

# %% [markdown]
# ##### Using Survive

# %%
brain_cancer_df['status'] = np.float64(brain_cancer_df['status'])

kmf_surv = survive.KaplanMeier()

surv_data = survive.SurvivalData(time='time', status='status', group='sex', data=brain_cancer_df)

kmf_surv_fit = kmf_surv.fit(surv_data)

fig, ax = plt.subplots(1,1, figsize=(8,6))

kmf_surv.plot(ax=ax, ci=False)

plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Estimated Probability of Survival');

# %% [markdown]
# `survive` is a little easier to use when plotting a survival curve based on different levels of a predictor, as one can specify the `group=` parameter when creating the `survive.SurvivalData()` object and it will automatically fit a different survival curve for each level of the predictor.  Afterwards, it's easy to plot all the curves at once by using a single call of `kmf_surv.plot()`.

# %% [markdown]
# ### Log-Rank Test

# %% language="R"
# logrank.test <- survdiff(Surv(time, status) ~ sex)
# logrank.test

# %% [markdown]
# #### Using Lifelines

# %%
male = (brain_cancer_df['sex'] == 'Male')
female = (brain_cancer_df['sex'] == 'Female')

results = logrank_test(brain_cancer_df['time'][male], 
                       brain_cancer_df['time'][female],
                       event_observed_A=brain_cancer_df['status'][male],
                       event_observed_B=brain_cancer_df['status'][female]
                      )

results.print_summary()

# %% [markdown]
# `lifelines` was able to perform a Log-Rank test, however the output wasn't as verbose as the ouput found in R.  Both R and Python had the same chi-square test_statistic, degrees of freedom, p-value, and -log2(p), but R also showed the group sizes (N), as well as observed and expected counts.

# %% [markdown]
# #### Using survive
# I was unable to find any documentation for a Log-Rank test on the `survive` website.  I don't think the library can perform this kind of test.

# %% [markdown]
# #### Using scikit-survival
# `scikit-survival` was able to peform the Log-Rank test and generate output that most closely resembled R's output, however it was still missing a thing or two.

# %%
x = []

for item in zip(brain_cancer_df['status'].astype(bool), brain_cancer_df['time']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

test_stat, p_val, surv_counts, covar = compare_survival(x, brain_cancer_df['sex'], return_stats=True)

print(f'Scikit-survival Chi-Square Test Statistic: {test_stat:.3f}')
print(f'Scikit-survival P-Value: {p_val:.4f}')
surv_counts

# %% [markdown]
# ### Cox Proportional Hazards (CoxPH) model

# %% language="R"
# fit.cox <- coxph(Surv(time, status) ~ sex)
# summary(fit.cox)

# %% [markdown]
# #### Different Python Libraries
# I didn't play around with `scikit-survival` for CoxPH models, however the [documentation](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) indicates it should be possible.  At some point in the future I may investigate this libraries capabilites further.
#
# The `survive` library wasn't able to fix CoxPH models, however `lifelines` could, and will be used predominantly for the remainder of the lab.

# %% [markdown]
# ##### Using Lifelines and sex as a predictor

# %% tags=[]
cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(brain_cancer_df, 'time', 'status', formula='sex')

cph_ll_fit.print_summary()

# %% [markdown]
# `lifelines` matches R's output, for the most part.  There are some minor differences like R showing output for the Wald Test, or the score of the logrank test, while `lifelines` shows a confidence interval for both the coef and the exp(coef).  Overall, both are very similar though.

# %% language="R"
# summary(fit.cox)$logtest[1]

# %% language="R"
# summary(fit.cox)$waldtest[1]

# %% language="R"
# summary(fit.cox)$sctest[1]

# %% language="R"
# logrank.test$chisq

# %% [markdown]
# In `lifelines`, the `print_summary()` function shows several of the values shown above in R, however it shows them along with many others.  Unfortunately, if one wanted to access a single value, it isn't possible.  The `print_summary()` function returns a None type object when it's called and indexing into the output isn't possible like it is in R.  
#
# Regardless, the reasoning behind why the lab in R extracts the values individually was to show more decimals, which is possible using `lifelines`.  If we want to see more decimals, we can set the `decimals=` parameter.  

# %%
cph_ll_fit.print_summary(decimals=6)

# %% language="R"
# fit.all <- coxph(
# Surv(time, status) ~ sex + diagnosis + loc + ki + gtv + stereo)
# fit.all

# %% [markdown]
# #### Using Lifelines and all predictors

# %%
cph_all_ll = CoxPHFitter()
cph_all_ll_fit = cph_ll.fit(brain_cancer_df[brain_cancer_df['diagnosis'].isnull()==False], 'time', 'status', formula='sex + diagnosis + loc + ki + gtv + stereo')

cph_all_ll_fit.print_summary()

# %% language="R"
# modaldata <- data.frame(
#     diagnosis = levels(diagnosis),
#     sex = rep("Female", 4),
#     loc = rep('Supratentorial', 4),
#     ki = rep(mean(ki), 4),
#     gtv = rep(mean(gtv), 4),
#     stereo = rep("SRT", 4)
# )
#
# modaldata

# %% [markdown]
# While it's possible to recreate the R code for the `modaldata` `data.frame` in Python, with the `lifelines` library, it isn't necessary, as will soon be seen.  
#
# To use `lifelines`, all that's really needed is the first line of code that stores the unique diagnosis levels into a variable called `levels.`.  I've included the code below which would recreate the `modaldata`, but left it commented out.

# %%
levels = brain_cancer_df[brain_cancer_df['diagnosis'].isnull()==False]['diagnosis'].unique()[[0,2,1,3]]

# # This step is redundant when using lifelines
# modal_data_df = pd.DataFrame(data = [levels,
#                                      np.repeat("Female", 4),
#                                      np.repeat("Supratentorial", 4),
#                                      np.repeat(brain_cancer_df['ki'].mean(), 4),
#                                      np.repeat(brain_cancer_df['gtv'].mean(), 4),
#                                      np.repeat("SRT", 4)],
#                                      columns = ['diagnosis', 
#                                                 'sex', 
#                                                 'loc', 
#                                                 'ki', 
#                                                 'gtv', 
#                                                 'stereo']
#                             )

# modal_data_df

# %% [markdown]
# #### Survival Curve by Diagnosis

# %% language="R"
# survplots <- survfit(fit.all, newdata = modaldata)
# plot(survplots, xlab = 'Months',
#     ylab = 'Survival Probability', col = 2:5)
# legend('bottomleft', levels(diagnosis), col = 2:5, lty = 1)

# %% [markdown]
# ##### Using Lifelines

# %%
fig, ax = plt.subplots(1,1, figsize=(8,6))
cph_all_ll_fit.plot_partial_effects_on_outcome(covariates='diagnosis', 
                                               values=levels, 
                                               plot_baseline=False, 
                                               ax = ax)
plt.xlabel('Months')
plt.ylabel('Survival Probability');

# %% [markdown]
# `lifelines` is able to plot a survival curve based on diagnosis level automatically, as long as the `covariates=` and `values=` parameters are specified.

# %% [markdown]
# ## 11.8.2 Publication Data

# %% [markdown]
# ### Kaplan Meier Survival Curve by posres

# %% language="R"
# fit.posres <- survfit(
#     Surv(time,  status) ~ posres, data = Publication
# )
# plot(fit.posres, xlab='Months',
#     ylab = 'Probability of Not Being Published', col = 3:4)
# legend("topright", c("Negative Result", "Positive Result"),
#       col = 3:4, lty = 1)

# %%
data = robjects.r("""
library(ISLR2)
xdata <- Publication
""")
with localconverter(robjects.default_converter + pandas2ri.converter):
    publication_df = robjects.conversion.rpy2py(data)

# %% [markdown]
# #### Using Lifelines

# %%
fig, ax = plt.subplots(1,1, figsize=(8,6))

for level in publication_df['posres'].unique():

    idx = (publication_df['posres'] == level)

    if level == 1:
        label = "Positive Result"
    else:
        label = "Negative Result"
        
    kmf_ll.fit(publication_df['time'][idx], 
               event_observed=publication_df['status'][idx], 
               label=label)
    kmf_ll.survival_function_.plot(ax=ax)


plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Not Being Published');

# %% [markdown]
# ### CoxPH by posres

# %% language="R"
# fit.pub <- coxph(Surv(time, status) ~ posres,
#                 data = Publication)
# fit.pub

# %% [markdown]
# #### Using Lifelines and posres as a predictor

# %%
cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(publication_df, 'time', 'status', formula='posres')

cph_ll_fit.print_summary()

# %% [markdown]
# Like before, the output from fitting a CoxPH model using `lifelines` is very similar to that of R.

# %% language="R"
# logrank.test <- survdiff(Surv(time, status) ~ posres,
#                         data = Publication)
# logrank.test

# %% [markdown]
# #### Using Lifelines

# %%
neg = (publication_df['posres'] == 0)
pos = (publication_df['posres'] == 1)

results = logrank_test(publication_df['time'][neg], 
                       publication_df['time'][pos],
                       event_observed_A=publication_df['status'][neg],
                       event_observed_B=publication_df['status'][pos]
                      )

results.print_summary()

# %% [markdown]
# Again, `lifelines` can perform a Log-Rank test, however the output is less basic as compared to R.

# %% [markdown]
# #### Using scikit-survival

# %%
x = []

for item in zip(publication_df['status'].astype(bool), publication_df['time']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

test_stat, p_val, surv_counts, covar = compare_survival(x, publication_df['posres'], return_stats=True)

print(f'Scikit-survival Chi-Square Test Statistic: {test_stat:.3f}')
print(f'Scikit-survival P-Value: {p_val:.4f}')
surv_counts

# %% language="R"
# fit.pub2 <- coxph(Surv(time, status) ~ . - mech,
#                  data = Publication)
# fit.pub2

# %% [markdown]
# #### Using Lifelines and all predictors but mech

# %%
cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(publication_df, 'time', 'status', formula='posres + multi + clinend + sampsize + budget + impact')

cph_ll_fit.print_summary()

# %% [markdown]
# ## 11.8.3 Call Center Data

# %% language="R"
# set.seed(4)
# N <- 2000
# Operators <- sample(5:15, N, replace = T)
# Center <- sample(c("A", "B", "C"), N, replace = T)
# Time <- sample(c("Morn.", "After.", "Even."), N, replace = T)
# X <- model.matrix( ~ Operators + Center + Time)[, -1]

# %% language="R"
# X[1:5, ]

# %%
r_data = robjects.r("""
set.seed(4)
N <- 2000
Operators <- sample(5:15, N, replace = T)
Center <- sample(c("A", "B", "C"), N, replace = T)
Time <- sample(c("Morn.", "After.", "Even."), N, replace = T)
X_df = data.frame(Operators, Center, Time)
""")

with localconverter(robjects.default_converter + pandas2ri.converter):
    call_center_df = robjects.conversion.rpy2py(r_data)
    
call_center_df_dummies = pd.get_dummies(call_center_df, columns=['Center', 'Time'], drop_first=True)

call_center_df_dummies[0:5]

# %% language="R"
# true.beta <- c(0.04, -0.3, 0, 0.2, -0.2)
# h.fn <- function(x) return(0.00001 * x)

# %% language="R"
# library(coxed)

# %% language="R"
# queuing <- sim.survdata(N = N, T = 1000, X = X,
# beta = true.beta, hazard.fun = h.fn)
#
# names(queuing)

# %% [markdown]
# Because of the randomness when generating simulated survival data using `sim.survdata` in R, it would be difficult to reproduce in Python, however by accessing the data property of the simulated data, I can use the `robjects` library to get it into Python.

# %%
r_data = robjects.r("""
queuing_data <- queuing$data
""")

with localconverter(robjects.default_converter + pandas2ri.converter):
    queuing_df = robjects.conversion.rpy2py(r_data)

# %% language="R"
# head(queuing$data)

# %%
queuing_df.head()

# %% language="R"
# mean(queuing$data$failed)

# %%
queuing_df['failed'].mean()

# %% tags=[] language="R"
# par(mfrow = c(1,2))
# fit.Center <- survfit(Surv(y, failed) ~ Center,
#                       data = queuing$data)

# %% [markdown]
# #### Using Lifelines
# In `lifelines`, if we want to segment by different levels of a predictor in a KaplanMeier model, we have to fit a KaplanMeier curve separately for each level.  This gets a little tedious and is a downside compared to `survive`

# %%
center_b_idx = (queuing_df['CenterB'] == 1)
center_c_idx = (queuing_df['CenterC'] == 1)
center_a_idx = (~(center_b_idx) & ~(center_c_idx))

kmf_ll_a = KaplanMeierFitter(label='Call Center A')
kmf_ll_a.fit(durations=queuing_df['y'][center_a_idx], 
             event_observed=queuing_df['failed'][center_a_idx])

kmf_ll_b = KaplanMeierFitter(label='Call Center B')
kmf_ll_b.fit(durations=queuing_df['y'][center_b_idx], 
             event_observed=queuing_df['failed'][center_b_idx])

kmf_ll_c = KaplanMeierFitter(label='Call Center C')
kmf_ll_c.fit(durations=queuing_df['y'][center_c_idx],
             event_observed=queuing_df['failed'][center_c_idx])

# %% language="R"
# plot(fit.Center, xlab = 'Seconds',
#     ylab = 'Probability of Still Being on Hold',
#     col = c(2, 4, 5))
# legend("topright",
#       c("Call Center A", "Call Center B", "Call Center C"),
#       col = c(2, 4, 5), lty = 1)

# %%
fig, ax = plt.subplots(1,1, figsize=(8,6))

kmf_ll_a.survival_function_.plot(ax=ax)
plt.xlabel('Seconds')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Still Being on Hold')

kmf_ll_b.survival_function_.plot(ax=ax)
plt.xlabel('Seconds')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Still Being on Hold')


kmf_ll_c.survival_function_.plot(ax=ax)
plt.xlabel('Seconds')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Still Being on Hold');

# %% language="R"
# fit.Time <- survfit(Surv(y, failed) ~ Time,
#                     data = queuing$data)
# plot(fit.Time, xlab = "Seconds",
#     ylab = 'Probability of Still Being on Hold',
#     col = c(2, 4, 5))
# legend("topright", c("Morning", "Afternoon", "Evening"),
#       col = c(5, 2, 4), lty = 1)

# %% [markdown]
# #### Using Lifelines

# %%
time_morn_idx = (queuing_df['TimeMorn.'] == 1)
time_even_idx = (queuing_df['TimeEven.'] == 1)
time_after_idx = (~(time_morn_idx) & ~(time_even_idx))

kmf_ll_morn = KaplanMeierFitter(label='Morning')
kmf_ll_morn.fit(durations=queuing_df['y'][time_morn_idx], 
             event_observed=queuing_df['failed'][time_morn_idx])

kmf_ll_after = KaplanMeierFitter(label='Afternoon')
kmf_ll_after.fit(durations=queuing_df['y'][time_after_idx], 
             event_observed=queuing_df['failed'][time_after_idx])

kmf_ll_even = KaplanMeierFitter(label='Evening')
kmf_ll_even.fit(durations=queuing_df['y'][time_even_idx],
             event_observed=queuing_df['failed'][time_even_idx])

fig, ax = plt.subplots(1,1, figsize=(8,6))

kmf_ll_morn.survival_function_.plot(ax=ax)
plt.xlabel('Seconds')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Still Being on Hold')

kmf_ll_after.survival_function_.plot(ax=ax)
plt.xlabel('Seconds')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Still Being on Hold')


kmf_ll_even.survival_function_.plot(ax=ax)
plt.xlabel('Seconds')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Probability of Still Being on Hold');

# %% language="R"
# survdiff(Surv(y, failed) ~ Center, data = queuing$data)

# %% [markdown]
# #### Using scikit-survival
# Again, `scikit-survival` does a good job of recreating most of the output of R's `survdiff()` function.

# %%
centers = call_center_df['Center'].values
x = []

for item in zip(queuing_df['failed'].astype(bool), queuing_df['y']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

test_stat, p_val, surv_counts, covar = compare_survival(x, centers, return_stats=True)

print(f'Scikit-survival Chi-Square Test Statistic: {test_stat:.3f}')
print(f'Scikit-survival P-Value: {p_val:.4f}')
surv_counts

# %% language="R"
# survdiff(Surv(y, failed) ~ Time, data = queuing$data)

# %% [markdown]
# #### Using scikit-survival

# %%
times = call_center_df['Time']
x = []

for item in zip(queuing_df['failed'].astype(bool), queuing_df['y']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

test_stat, p_val, surv_counts, covar = compare_survival(x, times, return_stats=True)

print(f'Scikit-survival Chi-Square Test Statistic: {test_stat:.3f}')
print(f'Scikit-survival P-Value: {p_val}')
surv_counts

# %% language="R"
# fit.queuing <- coxph(Surv(y, failed) ~ .,
#                     data = queuing$data)
# fit.queuing

# %%
y = np.array(queuing_df['y'])
failed = np.array(queuing_df['failed'])

call_center_df['y'] = y
call_center_df['failed'] = failed

# %% [markdown]
# #### Using Lifelines

# %%
cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(call_center_df, 'y', 'failed', formula='Operators + Center + Time')

cph_ll_fit.print_summary()

# %% [markdown]
# The End
