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

import survive

from sksurv.compare import compare_survival
import sksurv

import patsy

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

# %%
# Using lifelines
kmf_ll = KaplanMeierFitter()
kmf_ll.fit(durations=brain_cancer_df['time'], event_observed=brain_cancer_df['status'])

fig, ax = plt.subplots(1,1)

kmf_ll.plot_survival_function()
plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Estimated Probability of Survival')
ax.get_legend().remove();

# %%
# Using survive
brain_cancer_df['status'] = np.float64(brain_cancer_df['status'])

kmf_surv = survive.KaplanMeier()

surv_data = survive.SurvivalData(time=brain_cancer_df['time'], status=brain_cancer_df['status'])

kmf_surv_fit = kmf_surv.fit(surv_data)

kmf_surv_fit.plot(ci_style='lines')
plt.xlabel('Months')
plt.ylabel('Estimated Probability of Survival');

# %% [markdown]
# #### By Gender

# %% language="R"
# fit.sex <- survfit(Surv(time, status) ~ sex)
# plot(fit.sex, xlab = 'Months',
#     ylab = 'Estimated Probability of Survival', col = c(2,4))
# legend('bottomleft', levels(sex), col = c(2,4), lty = 1)

# %%
# Using lifelines

fig, ax = plt.subplots(1,1)

male = (brain_cancer_df['sex'] == 'Male')
female = (brain_cancer_df['sex'] == 'Female')

kmf_ll.fit(brain_cancer_df['time'][male], event_observed=brain_cancer_df['status'][male], label='Male')
kmf_ll.survival_function_.plot(ax=ax)

kmf_ll.fit(brain_cancer_df['time'][female], event_observed=brain_cancer_df['status'][female], label='Female')
kmf_ll.survival_function_.plot(ax=ax)

plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Estimated Probability of Survival');

# %%
# Using survive

brain_cancer_df['status'] = np.float64(brain_cancer_df['status'])

kmf_surv = survive.KaplanMeier()

surv_data = survive.SurvivalData(time='time', status='status', group='sex', data=brain_cancer_df)

kmf_surv_fit = kmf_surv.fit(surv_data)

fig, ax = plt.subplots(1,1)

kmf_surv.plot(ax=ax, ci=False)

plt.xlabel('Months')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylabel('Estimated Probability of Survival');

# %% [markdown]
# ### Log-Rank Test

# %% language="R"
# logrank.test <- survdiff(Surv(time, status) ~ sex)
# logrank.test

# %%
# Using lifelines

from lifelines.statistics import logrank_test

male = (brain_cancer_df['sex'] == 'Male')
female = (brain_cancer_df['sex'] == 'Female')

results = logrank_test(brain_cancer_df['time'][male], 
                       brain_cancer_df['time'][female],
                       event_observed_A=brain_cancer_df['status'][male],
                       event_observed_B=brain_cancer_df['status'][female]
                      )

results.print_summary()

# %%
# Using survive - I don't think it has logrank testing functionality :(

# %%
# Using scikit-survival

x = []

for item in zip(brain_cancer_df['status'].astype(bool), brain_cancer_df['time']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

compare_survival(x, brain_cancer_df['sex'], return_stats=True)

# %% [markdown]
# ### Cox Proportional Hazards (CoxPH) model

# %% language="R"
#
# fit.cox <- coxph(Surv(time, status) ~ sex)
# summary(fit.cox)

# %% tags=[]
# Using lifelines

cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(brain_cancer_df, 'time', 'status', formula='sex')

# %% tags=[]
cph_ll_fit.print_summary()

# %%
cph_ll_fit.log_likelihood_ratio_test()

# %%
results.test_statistic, results.degrees_of_freedom, results.p_value

# %%
# Using survive - I don't think survive can do this

# %%
# Using scikit-survival - I don't think there's a nice summary like lifelines has.  I can get the coefficient of the model, the concordance index, but not much else.
import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

def gender_to_int(x):
    if x == 'Male':
        return 1
    else:
        return 0


estimator = sksurv.linear_model.CoxPHSurvivalAnalysis(verbose=True).fit(np.array(brain_cancer_df['sex'].apply(gender_to_int)).reshape(-1,1), x)

# %%
estimator.coef_

# %%
# test_df = brain_cancer_df.copy(deep=True)
# test_df['sex'] = test_df['sex'].apply(gender_to_int)



# sksurv.metrics.concordance_index_censored(test_df['status'].astype(bool), test_df['time'], prediction)

# %% language="R"
# summary(fit.cox)$logtest[1]

# %% language="R"
# summary(fit.cox)$waldtest[1]

# %% language="R"
# summary(fit.cox)$sctest[1]

# %% language="R"
# logrank.test$chisq

# %% language="R"
# fit.all <- coxph(
# Surv(time, status) ~ sex + diagnosis + loc + ki + gtv + stereo)
# fit.all

# %%
brain_cancer_df.info()

# %%
brain_cancer_df[brain_cancer_df['diagnosis'].isnull()==False]

# %%
# Using lifelines

cph_all_ll = CoxPHFitter()
cph_all_ll_fit = cph_ll.fit(brain_cancer_df[brain_cancer_df['diagnosis'].isnull()==False], 'time', 'status', formula='sex + diagnosis + loc + ki + gtv + stereo')

# %%
cph_all_ll_fit.print_summary()

# %%
cph_all_ll_fit.log_likelihood_ratio_test()

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

# %%
levels = brain_cancer_df[brain_cancer_df['diagnosis'].isnull()==False]['diagnosis'].unique()[[0,2,1,3]]

# This step is redundant when using lifelines
modal_data_df = pd.DataFrame(data = [levels,
                                     np.repeat("Female", 4),
                                     np.repeat("Supratentorial", 4),
                                     np.repeat(brain_cancer_df['ki'].mean(), 4),
                                     np.repeat(brain_cancer_df['gtv'].mean(), 4),
                                     np.repeat("SRT", 4)],
                                     columns = ['diagnosis', 
                                                'sex', 
                                                'loc', 
                                                'ki', 
                                                'gtv', 
                                                'stereo']
                            )

# %%
modal_data_df

# %% [markdown]
# #### Survival Curve by Diagnosis

# %% language="R"
# survplots <- survfit(fit.all, newdata = modaldata)
# plot(survplots, xlab = 'Months',
#     ylab = 'Survival Probability', col = 2:5)
# legend('bottomleft', levels(diagnosis), col = 2:5, lty = 1)

# %% language="R"
# survplots

# %%
# Using lifelines

cph_all_ll_fit.plot_partial_effects_on_outcome('diagnosis', levels)
plt.xlabel('Months')
plt.ylabel('Survival Probability');

# %%
# # Using lifelines
# new_df = brain_cancer_df.dropna()

# fig, ax = plt.subplots(1,1)

# for level in levels:
#     idx = (new_df['diagnosis'] == level)
    

#     cph_all_ll_fit = cph_all_ll.fit(df = new_df.dropna()[idx],
#                                     duration_col = 'time',
#                                     event_col = 'status',
#                                     formula='sex + loc + ki + gtv + stereo',
#                                     show_progress = True)
#     cph_all_ll_fit.survival_function_.plot(ax=ax)

# plt.xlabel('Months')
# plt.yticks(np.arange(0, 1.1, 0.2))
# plt.ylabel('Survival Probability');

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

# %%
publication_df.head()

# %%
publication_df['posres'].unique()

# %%
# Using lifelines

fig, ax = plt.subplots(1,1)

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

# %%
# Using lifelines

cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(publication_df, 'time', 'status', formula='posres')

# %%
cph_ll_fit.print_summary()

# %% language="R"
# logrank.test <- survdiff(Surv(time, status) ~ posres,
#                         data = Publication)
# logrank.test

# %%
# Using lifelines

from lifelines.statistics import logrank_test

neg = (publication_df['posres'] == 0)
pos = (publication_df['posres'] == 1)

results = logrank_test(publication_df['time'][neg], 
                       publication_df['time'][pos],
                       event_observed_A=publication_df['status'][neg],
                       event_observed_B=publication_df['status'][pos]
                      )

results.print_summary()

# %%
# Using scikit-survival

x = []

for item in zip(publication_df['status'].astype(bool), publication_df['time']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

compare_survival(x, publication_df['posres'], return_stats=True)


# %% language="R"
# fit.pub2 <- coxph(Surv(time, status) ~ . - mech,
#                  data = Publication)
# fit.pub2

# %%
def formula_from_cols(df, y, use_target, remove_intercept):
    formula_string = ' + '.join([col for col in df.columns if not col==y])
    if use_target == True:
        formula_string = y + ' ~ ' + formula_string
    if remove_intercept == True:
        formula_string = formula_string + ' - 1'
    return formula_string


# %%
formula_string = formula_from_cols(publication_df, 'mech', False, False)

# %%
formula_string

# %%
# Using lifelines

cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(publication_df, 'time', 'status', formula='posres + multi + clinend + sampsize + budget + impact')

# %%
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
# r_data = robjects.r("""
# set.seed(4)
# N <- 2000
# Operators <- sample(5:15, N, replace = T)
# Center <- sample(c("A", "B", "C"), N, replace = T)
# Time <- sample(c("Morn.", "After.", "Even."), N, replace = T)
# X_df = data.frame(Operators, Center, Time)
# """)

# with localconverter(robjects.default_converter + pandas2ri.converter):
#     call_center_df = robjects.conversion.rpy2py(r_data)

# %%
# X_dmatrix = patsy.dmatrix(formula_like = 'Operators + Center + Time', data = call_center_df)

# %%
# X_dmatrix[0:5, 1:]

# %% language="R"
# true.beta <- c(0.04, -0.3, 0, 0.2, -0.2)
# h.fn <- function(x) return(0.00001 * x)

# %%
# true_beta = np.array([0.04, -0.3, 0, 0.2, -0.2])

# def h_fn(x):
#     return 0.00001 * x

# %% language="R"
# library(coxed)

# %% language="R"
# set.seed(4)
# N <- 2000
# Operators <- sample(5:15, N, replace = T)
# Center <- sample(c("A", "B", "C"), N, replace = T)
# Time <- sample(c("Morn.", "After.", "Even."), N, replace = T)
# X <- model.matrix( ~ Operators + Center + Time)[, -1]
#
# queuing <- sim.survdata(N = N, T = 1000, X = X,
# beta = true.beta, hazard.fun = h.fn)
#
# names(queuing)

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

# %%
# Using lifelines

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
queuing_df.head()

# %%
fig, ax = plt.subplots(1,1)

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

# %%
# Using lifelines

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

fig, ax = plt.subplots(1,1)

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

# %%
r_data = robjects.r("""
center_data <- Center
""")

with localconverter(robjects.default_converter + pandas2ri.converter):
    centers = robjects.conversion.rpy2py(r_data)
    
centers = np.array(centers)

# %%
# Using scikit-survival

x = []

for item in zip(queuing_df['failed'].astype(bool), queuing_df['y']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

compare_survival(x, centers, return_stats=True)

# %% language="R"
# survdiff(Surv(y, failed) ~ Time, data = queuing$data)

# %%
r_data = robjects.r("""
time_data <- Time
""")

with localconverter(robjects.default_converter + pandas2ri.converter):
    times = robjects.conversion.rpy2py(r_data)
    
times = np.array(times)

# %%
# Using scikit-survival

x = []

for item in zip(queuing_df['failed'].astype(bool), queuing_df['y']):
    x.append(item)

x = np.array(x, dtype='bool,f').flatten()

compare_survival(x, times, return_stats=True)

# %% language="R"
# fit.queuing <- coxph(Surv(y, failed) ~ .,
#                     data = queuing$data)
# fit.queuing

# %%
queuing_df

# %%
r_data = robjects.r("""
operator_data <- Operators
""")

with localconverter(robjects.default_converter + pandas2ri.converter):
    operators = robjects.conversion.rpy2py(r_data)
    
operators = np.array(operators)

# %%
operators.reshape(-1,1).shape

# %%
y = np.array(queuing_df['y'])
failed = np.array(queuing_df['failed'])

un_1h_df = pd.DataFrame(data={"Operators":operators, "Center":centers, "Time":times, "y": y, "failed":failed})

# %%
un_1h_df

# %%
# Using lifelines

cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(un_1h_df, 'y', 'failed', formula='Operators + Center + Time')

# %%
cph_ll_fit.print_summary()

# %% [markdown]
# The End
