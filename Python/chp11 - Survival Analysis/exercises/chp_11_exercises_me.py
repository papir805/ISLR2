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
# # Chapter 11 Exercises

# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter, CoxPHFitter

import survive

import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

# %load_ext rpy2.ipython

# %% [markdown]
# ## Conceptual Exercises

# %% [markdown]
# ### Exercise 2
#
# We conduct a study with $n=4$ participants who have just purchased cell phones, in order to model the time unti phone replacement.  The first participant replaces her phone after 1.2 years.  The second participant still has nt replaced her phone at the end of the two-year study period.  The third participant changes her phone and is lost to follow up (but has not yet replaced her phone) 1.5 years into the study.  The fourth participant replaces her phone after 0.2 years.
#
# For each of the four participants $(i=1,...,4)$, answer the following questions using the notation introduced in Section 11.1:

# %% [markdown]
# #### 2a) Is the participant's cell phone replacement time censored?
# A participant's cell phone replacement time is not censored if they replaced their phone before the 2 year study period was over, anything else is considered censored.
#    * $i = 1$ is not censored because they replaced their phone in 1.2 years, which is before the end of the two year study period.
#    * $i = 2$ is censored because they haven't replaced their cell phone before the study was over
#    * $i = 3$ is censored because they haven't replaced their cell phone yet, despite being lost to follow up
#    * $i = 4$ is not censored because they replaced their phone before the study was over.

# %% [markdown]
# #### 2b) Is the value of $c_i$ known, and if so, then what is it?
# If a time is censored, then $c_i$ will exist and equal the time of censoring
#    * $c_1$ does not exist, observation 1 was not censored
#    * $c_2 = 2$ because observation 2 was censored by the study ending at 2 years
#    * $c_3 = 1.5$ because observation 3 was censored when they were lost to follow up at 1.5 years into the study
#    * $c_4$ does not exist, observation 4 was not censored

# %% [markdown]
# #### 2c) Is the value of $t_i$ known, and if so, then what is it?
# If a time is not censored, then $t_i$ will exist and equal the time when a participant replaced their phone.
#     
#    * $t_1 = 1.2$ because observation 1 replaced their phone after 1.2 years
#    * $t_2$ does not exist because observation 2 never replaced their phone during the study
#    * $t_3$ does not exist because observation 3 never replaced their phone
#    * $t_4 = 0.2$ because observation 4 replaced their phone after 0.2 years

# %% [markdown]
# #### 2d) Is the value of $y_i$ known, and if so, then what is it?
# $$y_i = min(c_i, t_i)$$
#    
#    * $y_1=min(c_1,t_1)=t_1=1.2$ since $c_1$ does not exist
#    * $y_2=min(c_2,t_2)=c_2=2$ since $t_2$ does not exist
#    * $y_3=min(c_3,t_3)=c_3=1.5$ since $t_3$ does not exist
#    * $y_4=min(c_4,t_4)=t_4=0.2$ since $c_4$ does not exist

# %% [markdown]
# #### 2e) Is the value of $\delta_i$ known, and if so, then what is it?
#
# $\delta_i$ indicates whether observation $i$ was censored or not.
#     
# $\begin{equation}
#     \delta =
#     \left\{
#     \begin{aligned}
#     \ 1\ \ \ \  & \text{if }  & T \leq &C \text{ then the observation is not censored}\\
#     \ 0 \ \ \ \ & \text{if }   & T >  &C  \text{ then the observation is censored}
#     \end{aligned}
#     \right.
# \end{equation}$
#
#    * $\delta_1 = 1$ because observation 1 was not censored
#    * $\delta_2 = 0$ because observation 2 was censored
#    * $\delta_3 = 0$ because observation 3 was censored
#    * $\delta_4 = 1$ because observation 4 was not censored.

# %% [markdown]
# ### Exercise 3
# For the example in Exercise 2, report the values of $K, d_1,...,d_K, r_1,...,r_K, and q_1,...q_K$, where this notation was defined in Section 11.3
#
#  * $K=2$ because there are 2 unique "death" times among the non-censored patients
#  * $d_1=0.2$ because this is the first unique death time among the non-censored patients
#  * $d_K=d_2=1.2$ because this is the last unique death time among the non-censored patients
#  * $r_1=4$ because there are 4 patients "alive" (in the study) just before $d_1$
#  * $r_K=r_2=3$ because there are 3 patients "alive" (in the study) just before $d_2$
#  * $q_1=1$ because one patient died at time $d_1$
#  * $q_K=q_2=1$ because one patient died at time $d_2$

# %% [markdown]
# ### Exercise 4
# This problem makes use of the Kaplan-Meier survival curve displayed in Figure 11.9.  The raw data that went into plotting this survival curve is given in Table 11.4.  The coviariate column of that table is not needed for this problem.

# %% [markdown]
# ![image](./images/fig_11_9.png)

# %% [markdown]
# #### 4a) What is the estimated probability of survival past 50 days?

# %% [markdown]
# Each point on a Kaplan Meier curve shows the estimated probability of surviving past the time indicated on the horizontal axis for that point.  
#
# Using figure 11.9 above, $P(T>50) = 0.6$.  In other words, there's a 60% chance of survival beyond 50 days.

# %% [markdown]
# #### 4b) Write out an analytical expression for the estimated survival function.

# %% [markdown]
# ![image](./images/tab_11_4.png)

# %% [markdown]
# Using the relationship $\hat{S}(t)=\hat{S}(d_k)$ and the equation below, with $k = 1, 2, 3$, because there are three unique "death" times for non-censored subjects:
# \begin{equation*}
#   \hat{S}(d_k)=\prod_{j=1}^{k} (\frac{r_j - q_j}{r_j})
# \end{equation*}
#
#
# \begin{equation*}
#   \hat{S}(d_1)=\prod_{j=1}^{1} (\frac{r_j - q_j}{r_j})=\frac{r_1 - q_1}{r_1}=\frac{5-1}{5}=0.8
# \end{equation*}
#
# \begin{equation*}
#   \hat{S}(d_2)=\prod_{j=1}^{2} (\frac{r_j - q_j}{r_j})=\frac{r_1 - q_1}{r_1}\cdot\frac{r_2 - q_2}{r_2}=\frac{5-1}{5}\cdot\frac{4-1}{4}=0.6
# \end{equation*}
#
# \begin{equation*}
#   \hat{S}(d_3)=\prod_{j=1}^{3} (\frac{r_j - q_j}{r_j})=\frac{r_1 - q_1}{r_1}\cdot\frac{r_2 - q_2}{r_2}\cdot\frac{r_3-q_3}{r_3}=\frac{5-1}{5}\cdot\frac{4-1}{4}\cdot\frac{3-1}{3}=0.4
# \end{equation*}
#

# %% [markdown]
# $\begin{equation}
#     \hat{S}(t) =
#     \left\{
#     \begin{aligned}
#     \ 1\ \ \ \  & \text{if }  & t < 26.5 \\
#     \ 0.8  \ \ \ \ & \text{if }   & 26.5 \leq t <  37.2 \\
#     \ 0.6  \ \ \ \ & \text{if }   & 37.2 \leq t <  57.3 \\
#     \ 0.4  \ \ \ \ & \text{if }   & 57.3 \leq  t \\
#     \end{aligned}
#     \right.
# \end{equation}$

# %% [markdown]
# ##### Confirming using the survive package in Python

# %%
obs_y = [26.5, 37.2, 57.3, 90.8, 20.2, 89.8]
cens_indicator = [1, 1, 1, 0, 0, 0]
cov_x = [0.1, 11, -0.3, 2.8, 1.8, 0.4]

ex_4_df = pd.DataFrame(data={"obs_y":obs_y, "cens_indicator":cens_indicator, "cov_x":cov_x})

# %%
ex_4_df = ex_4_df.sort_values(by=['obs_y']).reset_index(drop=True)
ex_4_df

# %%
surv = survive.SurvivalData('obs_y', status='cens_indicator', data=ex_4_df)

km = survive.KaplanMeier()
km.fit(surv)

# %%
km.plot(ci=False)
plt.xlabel('Time in Days')
plt.ylabel("Estimated Probability of Survival");

# %%
km.predict([50])

# %%
km.summary

# %% [markdown]
# ### Exercise 5
#
# Sketch the survival function given by the equation
#
# $\begin{equation}
#     \hat{S}(t) =
#     \left\{
#     \begin{aligned}
#     \ 0.8\ \ \ \  & \text{if }  & t < 31 \\
#     \ 0.5  \ \ \ \ & \text{if }   & 31 \leq t <  77 \\
#     \ 0.22  \ \ \ \ & \text{if }   & 77 \leq  t \\
#     \end{aligned}
#     \right.
# \end{equation}$

# %%
x_s = [0, 31, 77, 100]
y_s = [0.8, 0.5, 0.22, 0.22]
fig, ax = plt.subplots(1,1)
ax.step(x_s, y_s, where='post')
ax.set_xlabel('Time in Days')
ax.set_ylabel('Estimated Probability of Survival');

# %% [markdown]
# ### Exercise 6
# This problem makes use of the data in Figure 11.1.  In completing this problem, you can refer to the observaton times as $y_1,...,y_4$.  The ordering of these observation times can be seen from Figure 11.1; their exact values are not required.

# %% [markdown]
# ![image](./images/fig_11_1.png)

# %% [markdown]
# #### 6a) Report the values of $\delta_1,...,\delta_4, K, d_1,...,d_K, r_1,...r_K, \text{ and } q_1,...,q_K$
#
#  * $\delta_1 = \delta_3 = 1$ because patients 1 & 3 are not censored
#  * $\delta_2 = \delta_4 = 0$ because patients 2 and 4 are censored
#  * $K=2$ because there are two unique "death" times among non censored patients
#  * $d_1 \approx 150$ because the first unique "death" time is roughly 150 days
#  * $d_K = d_2 = 300$ because the second, or last, "death" time is 300 days
#  * $r_1 = 4$ because there are 4 patients at risk (still in the study) at $d_1$
#  * $r_K = r_2 = 2$ because there are 2 patients at risk (still in the study) at $d_2$
#  * $q_1 = 1$ because one patient "died" at $d_1$
#  * $q_K = q_2 = 1$ because one patient "died" at $d_2$

# %% [markdown]
# #### 6b) Sketch the Kaplan-Meier survival curve corresponding to this data set.  (You do not need to use any software to do this - you can sketch it by hand using the results obtained in (a).)

# %%
x_s = [0, 150, 300, 365]
y_s = [1, 0.75, 3/8, 3/8]
fig, ax = plt.subplots(1,1)
ax.step(x_s, y_s, where='post')
ax.set_xlabel('Time in Days')
ax.set_ylabel('Estimated Probability of Survival');

# %% [markdown]
# #### 6c) Based on the survival curve estimated in (b), what is the probability that the event occurs within 200 days?  What is the probability that the event does not occur within 310 days?
#
# $P(t\leq200) = 1 - P(t>200) = 1 - 0.75 = 0.25$
#
# $P((t\leq310)^{C}) = P(t>310) = 0.375$

# %% [markdown]
# #### 6d) Write out an expression for the estimated survival curve from (b).
#
# $\begin{equation}
#     \hat{S}(t) =
#     \left\{
#     \begin{aligned}
#     \ 1\ \ \ \  & \text{if }  & t < 150 \\
#     \ 0.75  \ \ \ \ & \text{if }   & 150 \leq t <  300 \\
#     \ 0.375  \ \ \ \ & \text{if }   & 300 \leq  t \\
#     \end{aligned}
#     \right.
# \end{equation}$

# %% [markdown]
# ### Exercise 7
# In this problem, we will derive (11.5) and (11.6), which are needed for the construction of the log-rank test statistic (11.8).  Recall the notation in Table 11.1

# %% [markdown]
# ![image](./images/table_11_1.png)

# %% [markdown]
# #### 7a) Assume that there is no difference between the survival functions of the two groups.  Then we can think of $q_{1k}$ as the number of failures if we draw $r_{1k}$ observations, without replacement, from a risk set of $r_k$  observations that contains a total of $q_k$ failures.  Argue that $q_{1k}$ follows a *hypergeometric distribution*.  Write the parameters of this distribution in terms of $r_{1k}, r_k,$ and $q_k$.

# %% [markdown]
# $q_{1k}$ follows a hypergeometric distribution because:
#
# * The population or set to be sampled is a finite population consisting of $N=r_{k}$ observations
# * Each observation selected is either a success or a failure, where success is defined as a death occuring, and there are $M=q_{k}$ successes in the population.
# * A sample of $n=r_{1k}$ observations are selected without replacement in such a way that each subset of size $n$ is equally likely to be chosen.
#
# $h(x; n, M, N)=h(q_{1k}; r_{1k}, q_k, r_k)$

# %% [markdown]
# #### 7b) Given your previous answer, and the properties of the hypergeometric distribution, what are the mean and variance of $q_{1k}$?  Compare your answers to (11.5) and (11.6)

# %% [markdown]
# ![image](./images/expected_value_q_1k.png)

# %% [markdown]
# ![image](./images/variance_q_1k.png)

# %% [markdown]
# $E(q_{1k})=n\cdot\frac{M}{N}=r_{1k}\cdot\frac{q_k}{r_k}=\frac{r_{1k}}{r_k}\cdot q_k$, which is the same as (11.5)
#
# $V(q_{1k})=(\frac{N-n}{N-1})\cdot n \cdot \frac{M}{N} \cdot (1-\frac{M}{N})$
#
# $=(\frac{r_k-r_{1k}}{r_k-1})\cdot r_{1k} \cdot \frac{q_k}{r_k} \cdot (1-\frac{q_k}{r_k})$
#
# $=(\frac{r_k-r_{1k}}{r_k-1})\cdot r_{1k} \cdot \frac{q_k}{r_k} \cdot (\frac{r_k - q_k}{r_k})$
#
# $=q_k(r_{1k}/r_k)(\frac{r_k - r_{1k}}{r_k-1})(\frac{r_k - q_k}{r_k})$
#
# $=q_k(r_{1k}/r_k)(1-\frac{r_{1k}}{r_k})(\frac{r_k - q_k}{r_k-1})$
#
# $=\frac{q_k(r_{1k}/r_k)(1-r_{1k}/r_k)(r_k - q_k)}{r_k-1}$, which is the same ast (11.6)

# %% [markdown]
# ### Exercise 8
# Recall that the survival function $S(t)$, the hazard function $h(t)$, and the density function $f(t)$ are defined in (11.2), (11.9), and (11.11), respectively.  Furthermore, define $F(t) = 1 - S(t)$.  Show that the following relationships hold:
#     
# $$f(t) = dF(t)/dt$$
# $$S(t) = exp \bigl( -\int_0^th(u)du \bigr)$$

# %% [markdown]
# ![image](./images/survival_function.png)

# %% [markdown]
# ![image](./images/hazard_function.png)

# %% [markdown]
# ![image](./images/density_function.png)

# %%

# %%

# %% [markdown]
# ### Exercise 9
# In this exercise, we will explore the consequences of assuming that the survival times follow an exponential distribution.

# %% [markdown]
# #### 9a) Suppose that survival time follows an $Exp(\lambda)$ distribution, so that its density function is $f(t) = \lambda exp(-\lambda t)$.  Using the relationships provided in Exercise 8, show that $S(t)=exp(-\lambda t)$.

# %%

# %% [markdown]
# #### 9b) Now suppose that each of $n$ independent surival times follows an $Exp(\lambda)$ distribution.  Write out an expression for the likelihood function (11.13).

# %% [markdown]
# ![image](./images/likelihood_function.png)

# %% [markdown]
# \begin{equation*}
#   L=\prod_{i=1}^{n} \lambda exp(-\lambda y_i)^{\delta_i} exp(-\lambda y_i)^{1-\delta_i} = \prod_{i=1}^{n} \lambda^{\delta_i}exp(-\lambda y_i)
# \end{equation*}
#
# \begin{equation*}
#   L=\lambda^{\sum_{i=1}^{n} \delta_i}\cdot exp\bigl(-\lambda \sum_{i=1}^{n} y_i\bigr)
# \end{equation*}

# %% [markdown]
# #### 9c) Show that the maximum likelihood estimator for $\lambda$ is 
#
# \begin{equation*}
#   \hat{\lambda} = \sum_{i=1}^{n}\delta_i / \sum_{i=1}^{n} y_i
# \end{equation*}
#
# Applying the natural logarithm to our likelihood function from 9b) allows us to use properties of logarithms to simplify the equation so that taking the derivative is easier later.
# \begin{equation*}
#   ln(L)=ln\Bigl(\lambda^{\sum_{i=1}^{n} \delta_i}\cdot exp\bigl(-\lambda \sum_{i=1}^{n} y_i\bigr)\Bigr)
# \end{equation*}
#
# \begin{equation*}
#   ln(L)=ln\lambda \sum_{i=1}^{n}\delta_i - \lambda \sum_{i=1}^{n}y_i
# \end{equation*}
#
# \begin{equation*}
#   \frac{d}{d\lambda}[ln(L)]=\frac{d}{d\lambda}\biggl[ln\lambda \sum_{i=1}^{n}\delta_i - \lambda \sum_{i=1}^{n}y_i\biggr]
# \end{equation*}
#
# \begin{equation*}
#   \frac{1}{L}\frac{dL}{d\lambda}=\frac{1}{\lambda}\sum_{i=1}^{n} \delta_i - \sum_{i=1}^{n}y_i
# \end{equation*}
#
# \begin{equation*}
#   \frac{dL}{d\lambda}=[\frac{1}{\lambda} \sum_{i=1}^{n} \delta_i - \sum_{i=1}^{n}y_i]\cdot L
# \end{equation*}
#
# \begin{equation*}
#   0=[\frac{1}{\lambda} \sum_{i=1}^{n} \delta_i - \sum_{i=1}^{n}y_i]\cdot L
# \end{equation*}
#
# \begin{equation*}
#   0=\frac{1}{\lambda} \sum_{i=1}^{n} \delta_i - \sum_{i=1}^{n}y_i
# \end{equation*}
#
# \begin{equation*}
#   \sum_{i=1}^{n}y_i = \frac{1}{\lambda} \sum_{i=1}^{n} \delta_i
# \end{equation*}
#
# \begin{equation*}
#   \lambda = \frac{\sum_{i=1}^{n} \delta_i}{\sum_{i=1}^{n}y_i}
# \end{equation*}

# %% [markdown]
# #### 9d) Use your answer to (c) to derive an estimator of the mean survival time.
#
# \begin{equation*}
#   E(y_i) = 1 / \lambda
# \end{equation*}
#
# \begin{equation*}
#   E(y_i) = \frac{1}{\frac{\sum_{i=1}^{n} \delta_i}{\sum_{i=1}^{n} y_i}}
# \end{equation*}
#
# \begin{equation*}
#   E(y_i) = \frac{\sum_{i=1}^{n} y_i}{\sum_{i=1}^{n} \delta_i}
# \end{equation*}

# %%

# %%

# %%

# %% [markdown]
# ## Applied Exercises

# %% [markdown]
# ### Exercise 10
# This exercises focuses on the brain tumor data, which is included in the ISLR2 R library

# %% [markdown]
# #### 10a) Plot the Kaplan-Meier survival curve with $\pm1$ standard error bands, using the `survfit()` function in the `survival` package.

# %% language="R"
# library(ISLR2)
# library(survival)
# attach(BrainCancer)
#
# kmf_fit_r <- survfit(Surv(time, status) ~ 1, conf.int = 0.68, conf.type='log-log')
#
# plot(kmf_fit_r, xlab='Months',
#     ylab='Estimated Probability of Survival')

# %%
brain_cancer_df = pd.read_csv("../datasets/BrainCancer.csv", index_col=0)

# %%
brain_cancer_df.sort_values(by='time')

# %%
a = 0.3173105191

kmf = KaplanMeierFitter(alpha = a)
kmf.fit(brain_cancer_df['time'], brain_cancer_df['status'])

# %%
kmf.plot_survival_function()
plt.xlabel('Months')
plt.ylabel('Estimated Probability of Survival')
plt.legend(labels=[]);

# %% [markdown]
# #### 10b) Draw a bootstrap sample of size $n=88$ from the pairs $(y_i,\delta_i)$, and compute the resulting Kaplan-Meier survival curve.  Repeat this process $B=200$ times.  Use the results to obtain an estimate of the standard error of the Kaplan-Meier survival curve at each timepoint.  Compare this to the standard errors obtained in (a).

# %% language="R"
# sort(Surv(time, status))

# %% language="R"
# kmf_fit_r <- survfit(Surv(time, status) ~ 1, conf.type='log-log', error='greenwood')
# kmf_fit_r

# %% language="R"
# names(kmf_fit_r)

# %% language="R"
# kmf_fit_r$std.err

# %% language="R"
# kmf_fit_r$n.risk

# %% language="R"
# kmf_fit_r$n.event

# %% language="R"
# kmf_fit_r$surv

# %% language="R"
# kmf_fit_r$upper

# %% language="R"
# kmf_fit_r$std.chaz

# %%
surv = survive.SurvivalData('time', status='status', data=brain_cancer_df)
surv

# %%
km = survive.KaplanMeier()
km.fit(surv, n_boot=500, var_type='greenwood')

# %%
km.

# %%
km.quantile()

# %%
from survive import datasets

# %%
datasets.leukemia()

# %%
survival_func.rename(columns={"KM_estimate":"session_1"}, inplace=True)

# %%
survival_func

# %%
B = 200
num_observations = brain_cancer_df.shape[0]

idx = kmf.survival_function_.index
bstrap_df = pd.DataFrame(data=[], index=idx)

kmf_bootstrap = KaplanMeierFitter(alpha = a)


for i, sample in enumerate(range(B), start = 1):
    new_sample = brain_cancer_df.sample(n = num_observations,
                                        replace = True)
    kmf_bootstrap.fit(new_sample['time'], new_sample['status'])
    
    survival_func = kmf_bootstrap.survival_function_
    survival_func.rename(columns={"KM_estimate":F"session_{i}"}, inplace=True)
    
    
    bstrap_df = bstrap_df.merge(survival_func, 
                                how='left', 
                                left_index=True,
                                right_index=True)

# %%
bstrap_df

# %%
bstrap_df['session_1'].std()

# %%
bstrap_df.std(axis=1)

# %%
bstrap_df.mean(axis=1)

# %%
brain_cancer_df.sort_values(by='time')

# %%
surv = survive.SurvivalData('time', status='status', data=brain_cancer_df)
surv

# %%
display(surv.describe)

# %%
surv.plot_lifetimes()

# %%
surv.plot_at_risk()

# %%
km = survive.KaplanMeier(conf_level= 1 - a, random_state=1)

# %%
km.fit(surv, n_boot=500, var_type='greenwood')

# %%
km.summary

# %%
km.plot()

# %%
kmf.confidence_interval_

# %%
kmf.alpha

# %%
kmf.conditional_time_to_event_

# %%
kmf.confidence_interval_

# %%
kmf.confidence_interval_cumulative_density_

# %%
kmf.confidence_interval_survival_function_

# %%
kmf.survival_function_

# %% [markdown]
# #### 10c) Fit a Cox proportional hazards model that uses all of the predictors to predict survival.  Summarize the main findings.

# %%
brain_cancer_df.head()

# %%
brain_cancer_df.dropna()

# %%
# Using lifelines

cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(brain_cancer_df.dropna(), 'time', 'status', formula='sex + diagnosis + loc + ki + gtv + stereo')

cph_ll_fit.print_summary()

# %% [markdown]
# #### 10d) Stratify the data by the value of `ki`.  (Since only one observation has `ki=40`, you can group that observation together with the observations that have `ki=60`.). Plot the Kaplan-Meier survival curves for each of the five strata, adjusted for the other predictors.

# %%
brain_cancer_df['ki'].unique()


# %%
def ki_grouper(x):
    if x in [40, 60]:
        return 'A'
    elif x == 70:
        return 'B'
    elif x == 80:
        return 'C'
    elif x == 90:
        return 'D'
    elif x == 100:
        return 'E'


# %%
brain_cancer_df['group'] = brain_cancer_df['ki'].apply(ki_grouper)

# %%
brain_cancer_df[brain_cancer_df['group'] == 'A']

# %%
surv_by_ki = survive.SurvivalData(time='time',
                                  status='status', 
                                  group='group',
                                  data=brain_cancer_df)
km = survive.KaplanMeier()
km.fit(surv_by_ki)

# %%
fig, ax = plt.subplots(1,1, figsize=(12,8))
km.plot(ci=False, ax=ax);


# %% [markdown]
# ## Exercise 11
# This example makes use of the data in Table 11.4
#
# ![image](./images/tab_11_4.png)

# %% [markdown]
# #### 11a) Create two groups of observations.  In Group 1, $X < 2$, whereas in Group 2, $X \geq 2$.  Plot the Kaplan-Meier survival curves corresponding to the two groups.  Be sure to label the curves so that it is clear which curve corresponds to which group.  By eye, does there appear to be a difference between the two groups' survival curves?

# %%
def grouper(x):
    if x < 2:
        return "group 1"
    else:
        return "group 2"


# %%
ex_4_df['group'] = ex_4_df['cov_x'].apply(grouper)

# %%
ex_4_df

# %%
surv = survive.SurvivalData(time='obs_y',
                            status='cens_indicator', 
                            group='group',
                            data=ex_4_df)
km = survive.KaplanMeier()
km.fit(surv)

# %%
fig, ax = plt.subplots(1,1, figsize=(12,8))
km.plot(ci=False, ax=ax);

# %% [markdown]
# Visually, there doesn't appear to be much difference between the survival curves of the two groups.

# %% [markdown]
# #### 11b) Fit Cox's proportional hazards model, using the group indicator as a covariate.  What is the estimated coefficient?  Write a sentence providing the interpretation of this coefficient, in terms of the hazard or the instantaneous probability of the event.  Is there evidence that the true coefficient value is non-zero?

# %%
# Using lifelines

cph_ll = CoxPHFitter()
cph_ll_fit = cph_ll.fit(ex_4_df, 'obs_y', 'cens_indicator', formula='group')

cph_ll_fit.print_summary()

# %% [markdown]
# The estimated coefficient is -0.34 for group 2, which means that the risk for those in group 2 is $\approx 0.71 \text{ times (i.e. } e^{-0.34})$ the risk for those in the baseline group (group 1).  Because the p-value is so high $(p=0.78)$, there isn't sufficient evidence to conclude that the true coefficient value is non-zero, indicating that there isn't a difference in risk between the two groups.

# %% [markdown]
# #### 11c) Recall from Section 11.5.2 that in the case of a single binary covariate, the log-rank test statistic should be identical to the score statistic from the Cox model.  Conduct a log-rank test to determine whether there is a difference between the survival curves for the two groups.  How does the *p*-value for the log-rank test statistic compare to the *p*-value for the score statistic for the Cox model from (b)?

# %%
# Using lifelines

from lifelines.statistics import logrank_test

group_1 = (ex_4_df['group'] == 'group 1')
group_2 = (ex_4_df['group'] == 'group 2')

results = logrank_test(ex_4_df['obs_y'][group_1], 
                       ex_4_df['obs_y'][group_2],
                       event_observed_A=ex_4_df['cens_indicator'][group_1],
                       event_observed_B=ex_4_df['cens_indicator'][group_2]
                      )

results.print_summary()

# %% [markdown]
# As expected, the test statistic from the log-rank test (0.08) is the same as the log-likelihood ratio test score from the Cox model.  Furthermore, the *p*-value from the log-rank test (0.78) is the same as the *p*-value for the coefficient from the Cox model.

# %% [markdown]
# The End