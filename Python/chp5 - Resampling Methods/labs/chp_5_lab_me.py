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
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import pandas2ri

import statsmodels.formula.api as smf

import sklearn.preprocessing
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression

# %load_ext rpy2.ipython

# %% [markdown]
# ## 5.3.1 The Validation Set Approach

# %% [markdown]
# ### Checking that training indices match

# %% language="R"
# library(ISLR2)
# set.seed(1)
# train <- sort(sample(392, 196))

# %% jupyter={"source_hidden": true} tags=[] language="R"
# train

# %%
data = robjects.r("""
library(ISLR2)
set.seed(1)
train <- sample(392, 196)
""")

train_idx = np.array(data)
train_idx = np.sort(train_idx)

# %% jupyter={"source_hidden": true} tags=[]
train_idx

# %% [markdown]
# Complete: Training indices match

# %% [markdown]
# ### Checking that indexing in R and Python return same rows

# %% [markdown] tags=[]
# #### Using .iloc

# %% jupyter={"source_hidden": true} tags=[] language="R"
# Auto[sort(train), ]

# %%
auto_df = pd.read_csv("../../../datasets/Auto.csv", na_values='?')

# Reset index labels to start at 1 to match R's behavior
auto_df = auto_df.set_index(keys=np.arange(1, len(auto_df) + 1))

# Drow rows that contain '?' values that represent na values
auto_df = auto_df.dropna()

# %% jupyter={"source_hidden": true} tags=[]
auto_df[-1:]

# %% jupyter={"source_hidden": true} tags=[]
# .loc uses index labels to access rows from the dataframe.  Because auto_df originally had 397 rows, before dropping na values, the rows were labelled 1-397.
auto_df.loc[397]

# %% jupyter={"source_hidden": true} tags=[]
# .iloc uses integer labels to access rows from the dataframe.  After dropping the 5 na values from auto_df, there are only 392 rows remaining.  These rows have integer labels ranging from 0 (first row) to 391 (last row).  Indexing in R behaves like .iloc in that it uses integer labels.  This is why sample(392, 196) was used in generating the training index labels in R, there are 392 rows in the dataframe to sample from.
auto_df.iloc[392 - 1]

# %% jupyter={"source_hidden": true} tags=[]
# Using subtracting 1 from train_idx ensures the labels generated in R, which go from 1-392, now range from 0-391 so they work in P
auto_df.iloc[train_idx-1]

# %% [markdown]
# Complete: same rows are returned using the training indices using .iloc

# %% [markdown]
# #### Using a boolean mask

# %%
## Since boolean masks work using integer labels for indexing, this approach also can be used instead of .iloc, and it might be preferred since it's much easier to get the testing indices by negating the training indices.

auto_df_no_gaps = auto_df.copy(deep=True)

auto_df_no_gaps= auto_df_no_gaps.set_index(np.arange(1, auto_df_no_gaps.shape[0] + 1))

auto_train_mask_no_gaps = auto_df_no_gaps.index.isin(train_idx)

auto_test_maks_no_gaps = ~auto_train_mask_no_gaps

# %%
auto_df_no_gaps[auto_train_mask_no_gaps]

# %% [markdown]
# Complete: boolean index returns same rows in Python and R

# %% [markdown]
# ### Checking that lm and smf.ols produce same model

# %% language="R"
# lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)

# %% language="R"
# summary(lm.fit)

# %%
lm_model = smf.ols(formula = 'mpg ~ horsepower', data = auto_df.iloc[train_idx-1])
lm_fit = lm_model.fit()

# %%
lm_fit.summary()

# %% [markdown]
# Complete: R's lm function and Python's smf.ols return the same model

# %% [markdown]
# ### Creating test indices for Python and checking that same rows are returned in R and Python

# %% language="R"
# Auto[-train, c(1,9)]

# %% [markdown]
# #### Using .iloc

# %%
# because R using integer index labels when doing stuff like Auto[train, ], we need to use .iloc on our df in Python to copy the behavior.  Using .set_index, changes row labels, but not integer labels, and the integer labels always begin at 0, so we use np.arange(0, 392) to produce the indices 0 to 391, for the 392 entries in the auto_df.  Next, because train_idx is the list of integer row labels from R, which begins integer labels at 1, we need to subtract 1 from each entry to match the row integer labels in Python.  Using the set() function allows us to find the set difference, or the integer indices for rows not in our training set.  Because the set difference returns a set, which is treated as a single element, we can't use it with .iloc to get the rows we want, instead we convert the set to a list first.
test_idx = list(set(np.arange(0,392)) - set(train_idx-1))

# %%
auto_df.iloc[test_idx]

# %% [markdown]
# Complete: test indices return the same rows in R and Python using .iloc

# %% [markdown]
# #### Using boolean mask

# %%
auto_df[auto_test_maks_no_gaps]

# %% [markdown]
# Complete: test indices return the same rows in R and Python using boolean index

# %% [markdown]
# ### Checking that MSE of testing data matches in R and Python

# %% language="R"
# attach(Auto)
# mean((mpg - predict(lm.fit, Auto))[-train]^2)

# %%
pred = lm_fit.predict(auto_df.iloc[test_idx]['horsepower'])
((auto_df.iloc[test_idx]['mpg'] - pred)**2).mean()


# %% [markdown]
# Complete: MSE matches in R and Python

# %% [markdown]
# ### Polynomial Fits

# %% language="R"
# lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
# mean((mpg - predict(lm.fit2, Auto))[-train]^2)

# %% [markdown]
# #### 2nd Degree Polynomial

# %% [markdown]
# ##### Trying to using ortho_poly_fit with smf.ols in order to mimic R's lm function

# %%
## http://davmre.github.io/blog/python/2013/12/15/orthogonal_poly

def ortho_poly_fit(x, degree = 1):
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
            stop("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    return Z[:,1:], norm2, alpha


# %% [markdown]
# ##### Checking that ortho_poly_fit in Python produces same output as poly in R

# %% language="R"
# poly(horsepower[sort(train)], 2)

# %%
ortho_poly_fit(auto_df.iloc[train_idx-1]['horsepower'], 2)[0]

# %% [markdown]
# Complete: ortho_poly_fit in Python and poly in R produce same output

# %% [markdown]
# ##### Checking that using poly and lm in R produce same model as using ortho_poly_fit and smf.ols in Python

# %% language="R"
# summary(lm.fit2)

# %%
lm_model2_ortho_poly  = smf.ols(formula = 'mpg ~ ortho_poly_fit(horsepower, 2)[0]', data = auto_df.iloc[train_idx-1])
lm_fit2_ortho_poly = lm_model2_ortho_poly.fit()

# %%
lm_fit2_ortho_poly.summary()

# %% [markdown]
# **Incomplete: poly and lm in R produce a different model than ortho_poly_fit and smf.ols in Python.  As far as I can tell, the input to both functions is the same and am not sure why the outputs are different.  Perhaps I need to read the source code and try to understand how lm works compared to smf.ols in order to see how the inputs are used.  Since the inputs are the same, as far as I can tell, my best guess is that the outputs are different because lm and smf.ols work in different ways.**

# %% [markdown]
# ##### Checking if using poly and lm in R produce same model as PolynomialFeatures with smf.ols in Python

# %% language="R"
# summary(lm.fit2)

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

# %%
lm_model2_poly_feats  = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(horsepower).reshape(-1,1))', data = auto_df_no_gaps, subset=train_idx)
lm_fit2_poly_feats = lm_model2_poly_feats.fit(method='qr')

# %%
lm_fit2_poly_feats.summary()

# %% [markdown]
# **Incomplete: Using PolynomialFeatures with smf.ols doesn't produce the same model as poly and lm in R.  Although the models look different, that's because PolynomialFeatures returns an array where the vectors aren't orthogonalized, whereas poly in R does.  When the array isn't orthogonalized, the inputs are much larger and affect the coefficients for the model, hence the difference.**
#
# **If I generate the model in R using poly(horsepower, 2, raw=TRUE), then poly doesn't return and orthogonalized array and the coefficients of the model in R and Python match.  If I was able to orthogonalize the array from PolynomialFeatures in Pythonm, then I think the models would generate the same coefficients.  I may look into this later, but it's a little unncessary.**

# %% [markdown]
# ##### Checking MSEs

# %% language="R"
# mean((mpg - predict(lm.fit2, Auto))[-train]^2)

# %% [markdown]
# ###### Checking MSEs using ortho_poly_fit and smf.ols

# %%
pred2_ortho_poly = lm_fit2_ortho_poly.predict(auto_df_no_gaps[auto_test_maks_no_gaps]['horsepower'])

((pred2_ortho_poly - auto_df_no_gaps[auto_test_maks_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# ###### Checking MSEs using PolynomialFeatures and smf.ols

# %%
pred2_poly_feats = lm_fit2_poly_feats.predict(auto_df_no_gaps[auto_test_maks_no_gaps]['horsepower'])

((pred2_poly_feats - auto_df_no_gaps[auto_test_maks_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **Complete: The MSE from using PolynomialFeatures with smf.ols matches in Python and R, despite the models between Python and R not entirely matching, due to the difference in inputs where poly returns orthogonalized vectors while PolynomialFeatures doesn't**

# %% [markdown]
# #### 3rd Degree Polynomial

# %% [markdown]
# ##### Checking that ortho_poly_fit and smf.ols produce same model as poly and lm in R

# %% language="R"
# lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data=Auto, subset=train)
# summary(lm.fit3)

# %%
lm_model3_ortho_poly = smf.ols(formula='mpg ~ ortho_poly_fit(horsepower, 3)[0]', data=auto_df_no_gaps, subset=train_idx)

lm_fit3_ortho_poly = lm_model3_ortho_poly.fit()

lm_fit3_ortho_poly.summary()

# %% [markdown]
# **Incomplete: The models are not the same**

# %% [markdown]
# ##### Checking that PolynomialFeatures and smf.ols produce same model as poly and lm in R

# %%
## include_bias=False so that an intercept column is not returned
polynomial_features = sklearn.preprocessing.PolynomialFeatures(3, include_bias=False)

lm_model3_poly_feats  = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(horsepower).reshape(-1,1))', data = auto_df_no_gaps, subset=train_idx)
lm_fit3_poly_feats = lm_model3_poly_feats.fit(method='qr')

lm_fit3_poly_feats.summary()

# %% [markdown]
# **Incomplete: The models are not the same**

# %% [markdown]
# ##### Checking MSEs

# %% [markdown] tags=[]
# ###### Checking MSEs using ortho_poly_fit and smf.ols

# %% language="R"
# mean((mpg - predict(lm.fit3, Auto))[-train]^2)

# %%
pred3_ortho_poly = lm_fit3_ortho_poly.predict(auto_df_no_gaps[~auto_train_mask_no_gaps])

((pred3_ortho_poly - auto_df_no_gaps[~auto_train_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# ###### Checking MSEs using PolynomialFeatures and smf.ols

# %%
pred3_poly_feats = lm_fit3_poly_feats.predict(auto_df_no_gaps[~auto_train_mask_no_gaps])

((pred3_poly_feats - auto_df_no_gaps[~auto_train_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **Complete: The MSE from using PolynomialFeatures with smf.ols matches in Python and R, despite the models between Python and R not entirely matching, due to the difference in inputs where poly returns orthogonalized vectors while PolynomialFeatures doesn't**

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
new_lm_fit2_poly_feats = new_lm_model2_poly_feats.fit(method='qr')

new_pred2_poly_feats = new_lm_fit2_poly_feats.predict(auto_df_no_gaps[new_auto_test_mask_no_gaps]['horsepower'])

((new_pred2_poly_feats - auto_df_no_gaps[new_auto_test_mask_no_gaps]['mpg'])**2).mean()

# %% [markdown]
# **Complete: Same MSE in R and Python despite models being slightly different due to input vectors not being orthogonalized in Python**

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
# **Complete: Same MSE in R and Python despite models being slightly different due to input vectors not being orthogonalized in Python**

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

# %% [markdown]
# ### Checking that cv.glm in R and cross_val_score with smf.glm in Python produce same results 

# %% language="R"
# library(boot)
# glm.fit <- glm(mpg ~ horsepower, data = Auto)
# cv.err <- cv.glm(Auto, glm.fit)
# cv.err$delta

# %%
# glm_model = smf.glm(formula='mpg ~ horsepower', data=auto_df_no_gaps)
# glm_fit = glm_model.fit()

# %%
## Original Code and usage for the modified code in the cell below

# import statsmodels.api as sm
# from sklearn.base import BaseEstimator, RegressorMixin

# class SMWrapper(BaseEstimator, RegressorMixin):
#     """ A universal sklearn-style wrapper for statsmodels regressors """
#     def __init__(self, model_class, fit_intercept=True):
#         self.model_class = model_class
#         self.fit_intercept = fit_intercept
#     def fit(self, X, y):
#         if self.fit_intercept:
#             X = sm.add_constant(X)
#         self.model_ = self.model_class(y, X)
#         self.results_ = self.model_.fit()
#         return self
#     def predict(self, X):
#         if self.fit_intercept:
#             X = sm.add_constant(X)
#         return self.results_.predict(X)

# from sklearn.datasets import make_regression
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression

# X, y = make_regression(random_state=1, n_samples=300, noise=100)

# print(cross_val_score(SMWrapper(sm.OLS), X, y, scoring='r2'))
# print(cross_val_score(LinearRegression(), X, y, scoring='r2'))

# %% tags=[]
## The main code for SMWrapper was taken from: https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible, however I adapted it slightly to be able to work with the smf.glm command.  The changes are as follows:
## 1) I had to create a self.formula attribute for use in creating the model in the fit method.

## 2) I had to concatenate X and y into a pandas df to be passed when creating the model in the fit method.

## 3) When creating the model, I have to pass self.formula from step 1) and the pandas df from step 2)

## 4) I had to convert X into a pandas df when making predictions using the predict method.

import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, formula, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.formula = formula  ## 1)
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        data = pd.DataFrame(np.concatenate((X,y), axis=1), 
                            columns=['horsepower', 'mpg']) ## 2)
        self.model_ = self.model_class(self.formula, data) ## 3)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X):
        horsepower = pd.DataFrame(X, columns = ['horsepower']) ##4)
        if self.fit_intercept:
            X = sm.add_constant(horsepower)
        return self.results_.predict(horsepower)


X = auto_df_no_gaps['horsepower'].values.reshape(-1,1)
y = auto_df_no_gaps['mpg'].values.reshape(-1,1)

loo = LeaveOneOut()


scores = cross_val_score(estimator = SMWrapper(smf.glm, 
                                      formula='mpg ~ horsepower',
                                      fit_intercept=False), 
                            X=X, y=y, cv = loo, scoring = 'neg_mean_squared_error', error_score='raise')

# %%
np.mean(np.abs(scores))

# %% [markdown]
# **Complete: cv.glm in R and cross_val_score with smf.glm in Python produce same results, although it was somewhat difficult to replicate because cross_val_score is from sklearn and smf.glm is from statsmodels, which makes compatibility between them difficult.  Using the SMWrapper class overcomes this, but modifications need to be made to that class which don't scale well with my current solution.  It's probably easier to generate a model using the sklearn library, so that compatibility isn't an issue when using cross_val_score.**

# %% [markdown]
# ### Checking that cv.glm in R and cross_val_score with LinearRegression in Python produce same results 

# %%
## http://www.science.smith.edu/~jcrouser/SDS293/labs/lab7-py.html

X = auto_df_no_gaps['horsepower'].values.reshape(-1,1)
y = auto_df_no_gaps['mpg'].values.reshape(-1,1)

loo = LeaveOneOut()

skl_lm_model = sklearn.linear_model.LinearRegression()

scores = cross_val_score(skl_lm_model, X, y, cv = loo, scoring='neg_mean_squared_error')

np.mean(np.abs(scores))


# %% [markdown]
# **Complete: Same MSE and easier to implement since LinearRegression is a sklearn object, so it's compatible with cross_val_score, also from the sklearn library**

# %% [markdown]
# ### Polynomial Fits

# %% language="R"
# cv.error <- rep(0, 10)
# for (i in 1:10) {
#     glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
#     cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
# }
# cv.error

# %% [markdown]
# #### 2nd Degree Polynomial

# %% [markdown]
# ##### Checking that cv.glm in R and cross_val_score with smf.glm in Python produce same results 

# %%
def ortho_poly_predict(x, alpha, norm2, degree = 1):
    x = np.asarray(x).flatten()
    n = degree + 1
    Z = np.empty((len(x), n))
    Z[:,0] = 1
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    if degree > 1:
      for i in np.arange(1,degree):
          Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
    Z /= np.sqrt(norm2)
    return Z[:,1:]


# %%
## As much as I'd like to cross_validate using the smf.glm function to replicate the R code, it's going to be difficult to iterate over different powers for the polynomial fit unless I can figure out a way to use the ortho_poly_fit() in the formula= parameter when calling to SMWrapper, otherwise I have to manually type out the formula 'mpg ~ hp1 + hp2 + ...' and also modify the SWMrapper class accordingly in the SWMrapper fit and predict methods...It's probably better to switch to sklearn at this point and using LinearRegression and PolynomialFeatures instead.

import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, formula, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.formula = formula  ## 1)
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        X, self.norm2_, self.alpha_ = ortho_poly_fit(X, 2)
        
        
        data = pd.DataFrame(np.concatenate((X,y), axis=1), 
                            columns=['hp1', 'hp2', 'mpg']) ## 2)
        self.model_ = self.model_class(self.formula, data) ## 3)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X):
        X = ortho_poly_predict(X, self.alpha_, self.norm2_, 2)
        horsepower = pd.DataFrame(X, columns = ['hp1', 'hp2']) ##4)
        if self.fit_intercept:
            X = sm.add_constant(horsepower)
        return self.results_.predict(horsepower)


X = auto_df_no_gaps['horsepower'].values.reshape(-1,1)
y = auto_df_no_gaps['mpg'].values.reshape(-1,1)

loo = LeaveOneOut()


scores = cross_val_score(estimator = SMWrapper(smf.glm, 
                                      formula='mpg ~ hp1 + hp2',
                                      fit_intercept=False), 
                            X=X, y=y, cv = loo, scoring = 'neg_mean_squared_error', error_score='raise')

# %%
np.mean(np.abs(scores))

# %% [markdown]
# **Complete: With further modification to the SMWrapper class, I was able to get the same MSE as R, however the solution doesn't scale well when trying to iterate over polynomials with different degrees.  It's much easier to stay in the sklearn library with LinearRegression and cross_val_score.**

# %% [markdown]
# #### Checking MSE over polynomials with various degrees matches with R

# %%
cv_error = []

for i in range(1,11):
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(i, include_bias=False)

    scores = cross_val_score(skl_lm_model, polynomial_features.fit_transform(X), y, cv = loo, scoring='neg_mean_squared_error')
    
    mean_score = np.mean(np.abs(scores))
    
    cv_error.append(mean_score)

# %%
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

# %%
cv_error_10

# %% [markdown]
# **Because the folds are randomly generated, the folds generated from cv.glm in R are likely to be different than the folds generates from cross_val_score in Python and we shouldn't expect the bootstrap MSEs to match, but we'd expect them to be similar, which they are.** 

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

# %% [markdown]
# #### Figuring out how to match the variance and covariance calculated in R and Python

# %%
X = pfolio_df[1:10]['X']
Y = pfolio_df[1:10]['Y']

# %% language="R"
# var(Portfolio$X)

# %%
# Using ddof alters the formula when calculating the variance because the denominator = N - ddof.  When ddof=1, np.var calculates the unbiased estimate of the variance (the sample variance) as it divides by N - 1.  We need to do this in order to replicate the same values as R does.
np.var(pfolio_df['X'], ddof=1)

# %%
pfolio_df['X'].var()

# %% language="R"
# var(Portfolio$Y)

# %%
# Using ddof alters the formula when calculating the variance because the denominator = N - ddof.  When ddof=1, np.var calculates the unbiased estimate of the variance (the sample variance) as it divides by N - 1.  We need to do this in order to replicate the same values as R does.
np.var(pfolio_df['Y'], ddof=1)

# %%
pfolio_df['Y'].var()

# %% language="R"
# cov(Portfolio$X, Portfolio$Y)

# %% [markdown]
# Covariance Matrix:
# $$\Bigl(
# \begin{array}{rr}
# \sigma(x,x)&\sigma(x,y) \\
# \sigma(y, x)&\sigma(y,y)
# \end{array}
# \Bigr)$$

# %%
## np.cov returns the covariance matrix see above.  Where we only care about the covariance of x and y, so we have to index into the matrix to extract only that value
np.cov(pfolio_df['X'], pfolio_df['Y'])[0][1]

# %% language="R"
# set.seed(7)
# alpha.fn(Portfolio, sample(100, 100, replace=T))

# %% language="R"
# set.seed(7)
# sort(sample(100, 100, replace=T))

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
## Note: the boot package from R automatically creates bootstrap samples of length n, where n = the number of observations in the dataset you pass to boot.  In order to mimic that behavior in Python, we need to use np.random.choice n times.

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

# %% [markdown]
# #### Figuring out how the bias is calculated using boot in R

# %% language="R"
# myBootstrap <- boot(Auto, boot.fn, 1000)
#
# head(myBootstrap$t)

# %% language="R"
# myBootstrap$t0

# %% language="R"
# myBootstrap

# %% tags=[] language="R"
# myBootstrap$t0[1]

# %% language="R"
# # Intercept Bias
# mean(myBootstrap$t[,1]) - myBootstrap$t0[1]

# %% language="R"
# # Horsepower Bias
# mean(myBootstrap$t[,2]) - myBootstrap$t0[2]

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

# %%

# %%

# %%

# %% [markdown] tags=[]
# # Extra stuff

# %%
type(glm_model)

# %%
import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

# %%
import statsmodels.api as sm

# %%
X = sm.add_constant(auto_df_no_gaps['horsepower'].astype(np.float64).values)

glm_fit = glmnet(x = X, y = auto_df_no_gaps['mpg'].astype(np.float64).values, family='gaussian')

# %% tags=[]
glmnetCoef(glm_fit)

# %%
polynomial_features.fit_transform(np.array(auto_df['horsepower'][0:4]).reshape(-1, 1))[:,1:]


# %% language="R"
# horsepower[1:3]

# %%
def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]


# %%
poly(auto_df['horsepower'][0:4], 2)

# %%
auto_df[['horsepower_poly_1', 'horsepower_poly_2'] 

# %%
lm_model2 = smf.ols(formula = 'mpg ~ poly(horsepower,2)', data = auto_df.iloc[train_idx-1])
lm_fit2 = lm_model2.fit()
pred = lm_fit2.predict(auto_df.iloc[test_idx]['horsepower'])
((auto_df.iloc[test_idx]['mpg'] - pred)**2).mean()

# %%

# %%
lm_model2 = smf.ols(formula = 'mpg ~ poly(horsepower, 2)', data = auto_df.iloc[train_idx-1])

# %%
lm_fit2 = lm_model2.fit()

# %%
pred = lm_fit2.predict(auto_df.iloc[test_idx]['horsepower'])
((auto_df.iloc[test_idx]['mpg'] - pred)**2).mean()

# %%
lm_fit2.predict(auto_df['horsepower'][1])

# %%
lm_fit2.summary()

# %%
auto_df['horsepower'][0]

# %%
lm_model2 = smf.ols(formula = 'mpg ~ polynomial_features.fit_transform(np.array(auto_df.iloc[train_idx-1]["horsepower"]).reshape(-1,1))[:,1:]', data = auto_df.iloc[train_idx-1])

lm_fit2 = lm_model2.fit()

# poly.fit_transform(np.array(auto_df['horsepower'][0:4]).reshape(-1, 1))

# %%
lm_fit2.summary()

# %% tags=[]
polynomial_features.fit_transform(np.array(auto_df.iloc[train_idx-1]["horsepower"]).reshape(-1,1))[:,1:]


# %%
def ortho_poly_fit(x, degree = 1):
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
            stop("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    return Z[:,1:]##, norm2, alpha


# %%
X = ortho_poly_fit(auto_df['horsepower'][0:20], 2)
X

# %% tags=[]
lm_model2 = smf.ols(formula = 'mpg ~ ortho_poly_fit(horsepower, 2)', data = auto_df.iloc[train_idx-1])

lm_fit2 = lm_model2.fit(method='qr')

# poly.fit_transform(np.array(auto_df['horsepower'][0:4]).reshape(-1, 1))

# %%
lm_fit2.summary()

# %% tags=[]
ortho_poly_fit(auto_df.iloc[train_idx - 1]['horsepower'], 2)

# %%
X = ortho_poly_fit(auto_df.iloc[train_idx-1]['horsepower'], 2)

# %%
X

# %% tags=[]
X_df = pd.DataFrame(data={'h1':X[:,0], 'h2':X[:,1], 'mpg': auto_df['horsepower']})

# %%


lm_model2 = smf.ols(formula = 'mpg ~ h1 + h2', data = X_df.iloc[train_idx-1])

lm_fit2 = lm_model2.fit()
lm_fit2.summary()
# poly.fit_transform(np.array(auto_df['horsepower'][0:4]).reshape(-1, 1))

# %%
auto_df.set_index(1, auto_df.shape[0]+1)

# %%
test = auto_df.set_index(np.arange(1,auto_df.shape[0]+1))

# %%
test.loc[train_idx]

# %%
lm_model2 = smf.ols(formula = 'mpg ~ ortho_poly_fit(horsepower, 2)', data = test, subset=train_idx)

lm_fit2 = lm_model2.fit(method='qr')

lm_fit2.summary()

# %%
lm_fit2.fittedvalues

# %%
auto_df.iloc[train_idx-1]['mpg']

# %%
ortho_poly_fit(auto_df.iloc[train_idx-1]['horsepower'], 2)

# %%
