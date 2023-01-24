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
# # 5.3 Lab: Cross-Validation and the Bootstrap

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rpy2.robjects as robjects

import statsmodels.formula.api as smf

import sklearn.preprocessing
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression

# %load_ext rpy2.ipython

# %% [markdown]
# ## 5.3.1 The Validation Set Approach

# %% language="R"
# library(ISLR2)
# set.seed(1)
# train <- sort(sample(392, 196))

# %%
data = robjects.r("""
library(ISLR2)
set.seed(1)
train <- sample(392, 196)
""")

train_idx = np.array(data)
train_idx = np.sort(train_idx)

# %%
auto_df = pd.read_csv("../../../datasets/Auto.csv", na_values='?')

# Reset index labels to start at 1 to match R's behavior
auto_df = auto_df.set_index(keys=np.arange(1, len(auto_df) + 1))

# Drow rows that contain '?' values that represent na values
auto_df = auto_df.dropna()

# %%
## Since boolean masks work using integer labels for indexing, this mimics the behavior in R nicely.  It also makes it easyto get the testing indices by negating the training indices.

auto_df_no_gaps = auto_df.copy(deep=True)

auto_df_no_gaps= auto_df_no_gaps.set_index(np.arange(1, auto_df_no_gaps.shape[0] + 1))

auto_train_mask_no_gaps = auto_df_no_gaps.index.isin(train_idx)

auto_test_mask_no_gaps = ~auto_train_mask_no_gaps

# %% language="R"
# lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)

# %% language="R"
# summary(lm.fit)

# %%
lm_model = smf.ols(formula = 'mpg ~ horsepower', data = auto_df_no_gaps, subset=train_idx)
lm_fit = lm_model.fit()

lm_fit.summary()

# %% language="R"
# attach(Auto)
# mean((mpg - predict(lm.fit, Auto))[-train]^2)

# %%
pred = lm_fit.predict(auto_df_no_gaps[auto_test_mask_no_gaps]['horsepower'])
((auto_df_no_gaps[auto_test_mask_no_gaps]['mpg'] - pred)**2).mean()

# %% [markdown]
# ### Polynomial Fits

# %% [markdown]
# #### 2nd Degree Polynomial

# %% language="R"
# lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
# summary(lm.fit2)

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

lm_model2_poly_feats  = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(horsepower).reshape(-1,1))', data = auto_df_no_gaps, subset=train_idx)
lm_fit2_poly_feats = lm_model2_poly_feats.fit()

lm_fit2_poly_feats.summary()

# %% [markdown]
# **Using PolynomialFeatures with smf.ols doesn't produce the same model as poly and lm in R.  Although the models look different, that's because PolynomialFeatures returns an array where the vectors aren't orthogonalized, whereas poly in R does.  When the array isn't orthogonalized, the inputs are much larger and affect the coefficients for the model, hence the difference.**
#
# **If I generate the model in R using poly(horsepower, 2, raw=TRUE), then poly doesn't return an orthogonalized array and the coefficients of the model in R and Python match.  If I was able to orthogonalize the array from PolynomialFeatures in Python, then I think the models would generate the same coefficients.  I may look into this later, but it's a little unncessary and plan to proceed ahead for now.**

# %% language="R"
# mean((mpg - predict(lm.fit2, Auto))[-train]^2)

# %%
pred2_poly_feats = lm_fit2_poly_feats.predict(auto_df_no_gaps[auto_test_mask_no_gaps]['horsepower'])

((pred2_poly_feats - auto_df_no_gaps[auto_test_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **The MSE from using PolynomialFeatures with smf.ols matches in Python and R, despite the models between Python and R not entirely matching.  This is due to the difference in inputs where poly returns orthogonalized vectors while PolynomialFeatures doesn't.**

# %% [markdown]
# #### 3rd Degree Polynomial

# %% [markdown]
# ##### Checking that ortho_poly_fit and smf.ols produce same model as poly and lm in R

# %% language="R"
# lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data=Auto, subset=train)
# summary(lm.fit3)

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(3, include_bias=False)

lm_model3_poly_feats  = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(horsepower).reshape(-1,1))', data = auto_df_no_gaps, subset=train_idx)
lm_fit3_poly_feats = lm_model3_poly_feats.fit(method='qr')

lm_fit3_poly_feats.summary()

# %% language="R"
# mean((mpg - predict(lm.fit3, Auto))[-train]^2)

# %%
pred3_poly_feats = lm_fit3_poly_feats.predict(auto_df_no_gaps[~auto_train_mask_no_gaps])

((pred3_poly_feats - auto_df_no_gaps[~auto_train_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **The MSE from using PolynomialFeatures with smf.ols matches in Python and R, despite the models between Python and R not entirely matching.  This is due to the difference in inputs where poly returns orthogonalized vectors while PolynomialFeatures doesn't.**

# %% [markdown]
# ### Generating new training indices and reealuting MSE

# %% language="R"
# set.seed(2)
# train <- sample(392, 196)
# lm.fit <- lm(mpg ~ horsepower, subset = train)
# mean((mpg - predict(lm.fit, Auto))[-train]^2)

# %%
data = robjects.r("""
library(ISLR2)
set.seed(2)
train <- sample(392, 196)
""")

new_train_idx = np.array(data)
new_train_idx = np.sort(new_train_idx)

new_auto_train_mask_no_gaps = auto_df_no_gaps.index.isin(new_train_idx)
new_auto_test_mask_no_gaps = ~new_auto_train_mask_no_gaps

# %%
lm_model = smf.ols(formula='mpg ~ horsepower', data = auto_df_no_gaps, subset = new_train_idx)

lm_fit = lm_model.fit()

pred = lm_fit.predict(auto_df_no_gaps[~new_auto_train_mask_no_gaps]['horsepower'])

((pred - auto_df_no_gaps[~new_auto_train_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# #### Polynomial Fits

# %% [markdown]
# ##### 2nd Degree Polynomial

# %% language="R"
# lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
# mean((mpg - predict(lm.fit2, Auto))[-train]^2)

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

new_lm_model2_poly_feats  = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(horsepower).reshape(-1,1))', data = auto_df_no_gaps, subset=new_train_idx)
new_lm_fit2_poly_feats = new_lm_model2_poly_feats.fit()

new_pred2_poly_feats = new_lm_fit2_poly_feats.predict(auto_df_no_gaps[new_auto_test_mask_no_gaps]['horsepower'])

((new_pred2_poly_feats - auto_df_no_gaps[new_auto_test_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **Same MSE in R and Python despite models being slightly different due to input vectors not being orthogonalized in Python.**

# %% [markdown]
# ##### 3rd Degree Polynomial

# %% language="R"
# lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
# mean((mpg - predict(lm.fit3, Auto))[-train]^2)

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(3, include_bias=False)

new_lm_model3_poly_feats  = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(horsepower).reshape(-1,1))', data = auto_df_no_gaps, subset=new_train_idx)
new_lm_fit3_poly_feats = new_lm_model3_poly_feats.fit(method='qr')

new_pred3_poly_feats = new_lm_fit3_poly_feats.predict(auto_df_no_gaps[new_auto_test_mask_no_gaps]['horsepower'])

((new_pred3_poly_feats - auto_df_no_gaps[new_auto_test_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **Same MSE in R and Python despite models being slightly different due to input vectors not being orthogonalized in Pytho.**

# %% [markdown]
# ## 5.3.2 Leave-One-Out Cross-Validation

# %% language="R"
# glm.fit <- glm(mpg ~ horsepower, data = Auto)
# coef(glm.fit)

# %%
glm_model = smf.glm(formula='mpg ~ horsepower', data=auto_df_no_gaps)
glm_fit = glm_model.fit()
glm_fit.params

# %% language="R"
# lm.fit <- lm(mpg ~ horsepower, data = Auto)
# coef(lm.fit)

# %%
lm_model = smf.ols(formula = 'mpg ~ horsepower', data=auto_df_no_gaps)
lm_fit = lm_model.fit()
lm_fit.params

# %% language="R"
# library(boot)
# glm.fit <- glm(mpg ~ horsepower, data = Auto)
# cv.err <- cv.glm(Auto, glm.fit)
# cv.err$delta

# %%
X = auto_df_no_gaps['horsepower'].values.reshape(-1,1)
y = auto_df_no_gaps['mpg'].values.reshape(-1,1)

loo = LeaveOneOut()

skl_lm_model = sklearn.linear_model.LinearRegression()

scores = cross_val_score(skl_lm_model, X, y, cv = loo, scoring='neg_mean_squared_error')

np.mean(np.abs(scores))

# %% [markdown]
# ### Polynomial Fits

# %% language="R"
# cv.error <- rep(0, 10)
# for (i in 1:10) {
#     glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
#     cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
# }
# cv.error

# %%
cv_error = []

for i in range(1,11):
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(i, include_bias=False)

    scores = cross_val_score(skl_lm_model, polynomial_features.fit_transform(X), y, cv = loo, scoring='neg_mean_squared_error')
    
    mean_score = np.mean(np.abs(scores))
    
    cv_error.append(mean_score)
    
cv_error

# %% [markdown]
# **The MSEs match closely in Python and R.  All but the last 4 entries match exactly.**

# %% [markdown]
# ## 5.3.3 k-Fold Cross-Validation

# %% language="R"
# set.seed(17)
# cv.error.10 <- rep(0, 10)
# for (i in 1:10) {
#     glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
#     cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1]
# }
# cv.error.10

# %%
cv_error_10 = []

for i in range(1,11):
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(i, include_bias=False)

    scores = cross_val_score(skl_lm_model, polynomial_features.fit_transform(X), y, cv = 10, scoring='neg_mean_squared_error')
    
    mean_score = np.mean(np.abs(scores))
    
    cv_error_10.append(mean_score)
    
cv_error_10

# %% [markdown]
# **Because the folds are randomly generated, the folds generated from cv.glm in R are likely to be different than the folds generates from cross_val_score in Python and we shouldn't expect the bootstrap MSEs to match, but we'd expect them to be similar.** 

# %% [markdown]
# ## 5.3.4 The Bootstrap

# %% [markdown]
# ### Estimating the Accuracy of a Statistic of Interest

# %% language="R"
# alpha.fn <- function(data, index) {
#     X <- data$X[index]
#     Y <- data$Y[index]
#     (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
# }

# %% language="R"
# alpha.fn(Portfolio, 1:100)

# %%
pfolio_df = pd.read_csv("../../../datasets/Portfolio.csv")
pfolio_df = pfolio_df.set_index(np.arange(1, pfolio_df.shape[0] + 1))


# %%
def alpha_fn(data, index):
    X = data.loc[index]['X']
    Y = data.loc[index]['Y']
    var_x = np.var(X, ddof=1)
    var_y = np.var(Y, ddof=1)
    cov_x_y = np.cov(X, Y, ddof=1)[0][1]
    
    return (var_y - cov_x_y) / (var_x + var_y - 2 * cov_x_y)


# %%
alpha_fn(pfolio_df, np.arange(1,101))

# %% language="R"
# set.seed(7)
# alpha.fn(Portfolio, sample(100, 100, replace=T))

# %%
data = robjects.r("""
set.seed(7)
train <- sample(100, 100, replace=T)
""")

bstrap_idx = np.array(data)
bstrap_idx = np.sort(bstrap_idx)

# %%
alpha_fn(pfolio_df, bstrap_idx)

# %% language="R"
# boot(Portfolio, alpha.fn, R = 1000)

# %%
## Note: the boot package from R automatically creates bootstrap samples of length n, where n = the number of observations in the dataset you pass to boot.  In order to mimic that behavior in Python, we need np.random.choice to choose n numbers to ensure the bootstrap sample has the same number of observations as in the dataset.

alphas = []
n = pfolio_df.shape[0]

for _ in range(1000):
    idx = np.random.choice(np.arange(1, n+1), n)
    alpha = alpha_fn(pfolio_df, idx)
    alphas.append(alpha)
    
original_alpha = alpha_fn(pfolio_df, np.arange(1, n+1))
alpha_bstrap_mean = np.mean(alphas)
alpha_bstrap_std = np.std(alphas, ddof=1) ## When calculating the std dev, np.std divides by N-ddof.  To find the sample std dev, we set ddof=1

## Bias = bootstrap realization of the statistic - the original statistic from the original data
alpha_bias = alpha_bstrap_mean - original_alpha

print(f'Original Alpha: {original_alpha}')
print(f'Alpha Bias: {alpha_bias}')
print(f'Alpha Std: {alpha_bstrap_std}')


# %% [markdown]
# ### Estimating the Accuracy of a Linear Regression Model

# %% language="R"
# boot.fn <- function(data, index)
#     coef(lm(mpg ~ horsepower, data = data, subset = index))
# boot.fn(Auto, 1:392)

# %%
def boot_fn(data, index):
    model = smf.glm(formula = 'mpg ~ horsepower', data = data, subset = index)
    fit = model.fit()
    coefficients = fit.params
    
    return coefficients


# %%
boot_fn(auto_df_no_gaps, np.arange(1, 393))

# %% language="R"
# set.seed(1)
# boot.fn(Auto, sample(392, 392, replace = T))

# %%
data = robjects.r("""
set.seed(1)
samp <- sample(392, 392, replace=T)
""")

bstrap_idx = np.array(data)
bstrap_idx = np.sort(bstrap_idx)

# %%
boot_fn(auto_df_no_gaps, bstrap_idx)

# %% language="R"
# boot.fn(Auto, sample(392, 392, replace = T))

# %%
idx = np.random.choice(np.arange(1,393), 392)
boot_fn(auto_df_no_gaps, idx)

# %% language="R"
# boot(Auto, boot.fn, 1000)

# %%
intercepts = []
slopes = []

n = auto_df_no_gaps.shape[0]

for _ in range(1000):
    idx = np.random.choice(np.arange(1, n+1), n)
    param = boot_fn(auto_df_no_gaps, idx)
    intercept = param[0]
    slope = param[1]
    intercepts.append(intercept)
    slopes.append(slope)

original_intercept = boot_fn(auto_df_no_gaps, np.arange(1, 393))[0]
original_slope = boot_fn(auto_df_no_gaps, np.arange(1, 393))[1]
intercept_bstrap_mean = np.mean(intercepts)
intercept_bstrap_std = np.std(intercepts, ddof=1) ## When calculating the std dev, np.std divides by N-ddof.  To find the sample std dev, we set ddof=1
slope_bstrap_mean = np.mean(slopes)
slope_bstrap_std = np.std(slopes, ddof=1)

## Bias = bootstrap realization of the statistic - the original statistic from the original data
## bias for intercepts
intercept_bias = intercept_bstrap_mean - original_intercept

## bias for slopes
slope_bias = slope_bstrap_mean - original_slope

print(f'Original Intercept: {original_intercept}')
print(f'Bstrap Intercept Bias: {intercept_bias}')
print(f'Bstrap Intercept Std: {intercept_bstrap_std}')
print()
print(f'Original Slope: {original_slope}')
print(f'Bstrap Slope Bias: {slope_bias}')
print(f'Bstrap Slope Std: {slope_bstrap_std}')

# %% language="R"
# summary(lm(mpg ~ horsepower, data = Auto))$coef

# %%
## https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe

results = smf.ols(formula = 'mpg ~ horsepower', data = auto_df_no_gaps).fit().summary().tables[1]

results_as_html = results.as_html()
results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

results_df[['coef', 'std err', 't', 'P>|t|']]


# %% language="R"
# boot.fn <- function(data, index)
#     coef(
#       lm(mpg ~ horsepower + I(horsepower^2),
#         data = data, subset = index)
#     )
#
# set.seed(1)
# boot(Auto, boot.fn, 1000)

# %%
## What does I() do in the formula: https://stackoverflow.com/questions/24192428/what-does-the-capital-letter-i-in-r-linear-regression-formula-mean

def boot_fn(data, index):
    model = smf.ols(formula='mpg ~ horsepower + I(horsepower**2)', 
                    data=data, 
                    subset=index)
    fit = model.fit()
    params = fit.params
    
    return params


# %%
intercepts = []
hp_1s = []
hp_2s = []

n = auto_df_no_gaps.shape[0]

for _ in range(1000):
    idx = np.random.choice(np.arange(1, n+1), n)
    param = boot_fn(auto_df_no_gaps, idx)
    intercept = param[0]
    hp_1 = param[1]
    hp_2 = param[2]
    intercepts.append(intercept)
    hp_1s.append(hp_1)
    hp_2s.append(hp_2)

original_intercept = boot_fn(auto_df_no_gaps, np.arange(1, 393))[0]
original_hp_1 = boot_fn(auto_df_no_gaps, np.arange(1, 393))[1]
original_hp_2 = boot_fn(auto_df_no_gaps, np.arange(1, 393))[2]

intercept_bstrap_mean = np.mean(intercepts)
intercept_bstrap_std = np.std(intercepts, ddof=1) ## When calculating the std dev, np.std divides by N-ddof.  To find the sample std dev, we set ddof=1

hp_1_bstrap_mean = np.mean(hp_1s)
hp_1_bstrap_std = np.std(hp_1s, ddof=1)

hp_2_bstrap_mean = np.mean(hp_2s)
hp_2_bstrap_std = np.std(hp_2s, ddof=1)

## Bias = bootstrap realization of the statistic - the original statistic from the original data
## bias for intercepts
intercept_bias = intercept_bstrap_mean - original_intercept

## bias for horsepower
hp_1_bias = hp_1_bstrap_mean - original_hp_1

## bias for horsepower**2
hp_2_bias = hp_2_bstrap_mean - original_hp_2

# %%
print(f'Original Intercept: {original_intercept}')
print(f'Bstrap Intercept Bias: {intercept_bias}')
print(f'Bstrap Intercept Std: {intercept_bstrap_std}')
print()
print(f'Original Slope: {original_hp_1}')
print(f'Bstrap horsepower Bias: {hp_1_bias}')
print(f'Bstrap horsepower Std: {hp_1_bstrap_std}')
print()
print(f'Original Slope: {original_hp_2}')
print(f'Bstrap horsepower**2 Bias: {hp_2_bias}')
print(f'Bstrap horsepower**2 Std: {hp_2_bstrap_std}')

# %% language="R"
# summary(
#     lm(mpg ~ horsepower + I(horsepower^2), data = Auto)
# )$coef

# %%
model = smf.ols(formula = 'mpg ~ horsepower + I(horsepower**2)',
                data = auto_df_no_gaps)
fit = model.fit()

results = fit.summary().tables[1]

results_as_html = results.as_html()
results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

results_df[['coef', 'std err', 't', 'P>|t|']]

# %% [markdown]
# # The End
