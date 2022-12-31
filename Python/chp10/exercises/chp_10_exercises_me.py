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
# # Chapter 10 Exercises

# %%
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)

# %%
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd

import patsy

from sklearn.model_selection import train_test_split

import os

import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

import statsmodels.formula.api as smf
import statsmodels.api as sm

# %% [markdown]
# ## Exercise 6
# Consider the simple function $R(\beta)=sin(\beta)+\beta/10$

# %% [markdown]
# ### 6a) 
# Draw a graph of this function over the range $\beta\in[-6,6]$

# %%
beta = np.linspace(-6, 6)

y = np.sin(beta) + beta * 1/10

plt.plot(beta, y)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$R(\beta)$');


# %% [markdown]
# ### 6b)  
# What is the derivative of this function?
#
# $$\nabla R(\beta) = \frac{\partial R(\beta)}{\partial \beta} = cos(\beta)+1/10$$

# %% [markdown]
# ### 6c) 
# Given $\beta^0 = 2.3$, run gradient descent to find a local minimum of $R(\beta)$ using a learning rate of $\rho=0.1$.  Show each of $\beta^0, \beta^1,...$ in your plot, as well as the final answer.

# %% [markdown]
# Below, I define a function `find_local_min()`, using the formula from section 10.7.1 on Backpropagation from page 435, that will run the gradient descent.  The formula is as follows:  
#
# $$\beta^{m+1}\leftarrow \beta^{m} - \rho\cdot\nabla R(\beta^{m})$$
#
# `find_local_min()` takes an initial condition $\beta^0$ and adjusts it based on the gradient evaluated at that point. The gradient will be calculated using the derivative from the previous exercise 6b), evaluated at $\beta^{m}$:
#
# $$\nabla R(\beta^{m}) = \frac{\partial R(\beta)}{\partial \beta}\Bigr|_{\beta=\beta^{m}} = cos(\beta^{m})+1/10$$
#
# Using the gradient and the learning rate $\rho$, The initial condition $\beta^0$ will be adjusted to get a new value $\beta^1$.  
#
# $$\beta^{m+1}\leftarrow \beta^{m} - \rho \cdot [cos(\beta^{m}) + 1/10]$$
# $$\beta^{1}\leftarrow \beta^{0} - \rho \cdot [cos(\beta^{0}) + 1/10]$$
#
# The process will be repeated using the new value $\beta^1$, which will be adjusted to get $\beta^2$, etc.  This process repeats until the gradient is less than a certain threshold and the function terminates, indicating a local minimum has been found:
#
# What a run looks like with $\rho=0.1$:
# $$\beta^{1}\leftarrow \beta^{0} - 0.1 \cdot [cos(\beta^{0}) + 1/10]$$
# $$\beta^{2}\leftarrow \beta^{1} - 0.1 \cdot [cos(\beta^{1}) + 1/10]$$
# $$\vdots$$

# %%
def find_local_min(b_0, p = 0.1):
    import numpy as np
    
    betas = []
    
    derivative = np.cos(b_0) + 1/10
    
    while not np.abs(derivative) < 0.000001 :
        #print(b_0, derivative)
        betas.append(b_0)
        b_0 -= p * derivative
        derivative = np.cos(b_0) + 1/10      
        
    betas.append(b_0)
    return betas


# %%
b = find_local_min(2.3)

# %%
first_betas = b[0:3]
optimal_beta = b[-1]

min_beta = min(b)
max_beta = max(b)

xt = np.append(np.arange(min_beta - 1, max_beta + 1, 1), first_betas)
xt = np.round(xt, decimals = 1)
xt = np.append(xt, optimal_beta)
xt = np.sort(xt)
xt = np.unique(xt)

# %%
num_betas = len(b)

xt_labels = []
i = 0

for item in xt:
    if b[0] <= item <= b[2]:
        xt_labels.append(fr'$\beta^{i}$')
        i += 1
    elif item == optimal_beta:
        xt_labels.append(fr'$\beta^{{{num_betas}}}$')
    else:
        xt_labels.append(item)

# %%
fig, ax = plt.subplots(1,1, figsize=(12,5))

x = np.linspace(min_beta - 1, max_beta + 1)

ax.plot(x, np.sin(x) + np.array(x) / 10)
ax.scatter(b, np.sin(b) + np.array(b) / 10, color='red')
ax.set_xticks(xt, labels = xt_labels)
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$R(\beta)$');

# %% [markdown]
# ### 6d)
# Repeat with $\beta^0 = 1.4$ 

# %%
c = find_local_min(1.4)

# %%
first_betas = c[0:3]
optimal_beta = c[-1]

min_beta = min(c)
max_beta = max(c)


xt = np.append(np.arange(min_beta - 1.3, max_beta + 0.7, 1), first_betas)
xt = np.round(xt, decimals=1)
xt = np.append(xt, optimal_beta)
xt = np.sort(xt)
xt = np.unique(xt)

# %%
num_betas = len(c)

xt_labels = []
i = 2

for item in xt:
    if c[2] <= item <= c[0]:
        xt_labels.append(fr'$\beta^{i}$')
        i -= 1
    elif item == optimal_beta:
        xt_labels.append(fr'$\beta^{{{num_betas}}}$')
    else:
        xt_labels.append(item)

# %%
fig, ax = plt.subplots(1,1, figsize=(12,5))

x = np.linspace(min_beta - 1, max_beta + 1)

ax.plot(x, np.sin(x) + np.array(x) / 10)
ax.scatter(c, np.sin(c) + np.array(c) / 10, color='orange')
ax.set_xticks(xt, labels = xt_labels)
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$R(\beta)$');

# %% [markdown]
# ## Exercise 7
# Fit a neural network to the `Default` data.  Use a single hidden layer with 10 units, and dropout regularization.  Have a look at Labs 10.9.1-10.9.2 for guidance.  Compare the classification performance of your model with that of logistic regression.

# %%
default = pd.read_csv("../../../datasets/Default.csv")

# %%
default.head()


# %%
def formula_from_cols(df, y):
    return y + ' ~ ' + ' + '.join([col for col in df.columns if not col==y])


# %%
formula_string = formula_from_cols(default, 'default')

# %%
formula_string


# %%
def convert_to_int(x):
    if x == 'Yes':
        return 1
    else:
        return 0


# %%
default['default_int'] = default['default'].apply(convert_to_int)

# %%
x = patsy.dmatrix(formula_like = 'student + balance + income - 1', data = default)

x_scale = patsy.scale(x, ddof=1)

y = np.array(default['default_int'])

# %%
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.2, random_state=42)

# %% [markdown]
# ### Neural Network

# %%
modnn = tf.keras.Sequential(
    [
            tf.keras.layers.Dense(units = 10,
                                  activation='relu'),
            tf.keras.layers.Dropout(rate=0.4),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
    ]
)

# %%
modnn.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics='accuracy'
             )

# %% tags=[]
history = modnn.fit(x_train, 
                    y_train, 
                    epochs = 30, 
                    batch_size = 128, 
                    validation_data = (x_test, y_test), 
                    verbose = 0)

# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))

ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].set_title('model loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend(['Train', 'Validation'])

ax[1].plot(history.history['accuracy'])
ax[1].plot(history.history['val_accuracy'])
ax[1].set_title('model accuracy')
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].legend(['Train', 'Validation'])

plt.tight_layout();

# %%
modnn.summary()

# %%
y_proba = modnn.predict(x_test)
y_pred_classes = y_proba > 0.5

# %%
modnn_acc = np.mean(y_pred_classes.flatten() == y_test)
modnn_acc

# %% [markdown]
# ### Logistic Regression

# %%
modlr = tf.keras.Sequential(
    [
            tf.keras.layers.Dense(units = 1,
                                  activation='sigmoid')
    ]
)

# %%
modlr.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics='accuracy'
             )

# %% tags=[]
history = modlr.fit(x_train, 
                    y_train, 
                    epochs = 30, 
                    batch_size = 128, 
                    validation_data = (x_test, y_test), 
                    verbose = 0)

# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))

ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].set_title('model loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend(['Train', 'Validation'])

ax[1].plot(history.history['accuracy'])
ax[1].plot(history.history['val_accuracy'])
ax[1].set_title('model accuracy')
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].legend(['Train', 'Validation'])

plt.tight_layout();

# %%
y_proba = modlr.predict(x_test)
y_pred_classes = y_proba > 0.5

# %%
modlr_acc = np.mean(y_pred_classes.flatten() == y_test)
modlr_acc

# %%
modlr.summary()

# %% [markdown]
# ### Comparison

# %% [markdown]
# The performance of the neural network and logistic regression is about the same.  Both produce accuracy of roughly 97%.

# %% [markdown]
# ## Exercise 8
# From your collection of personal photographs, pick 10 images of animals (such as dogs, cats, birds, farm animals, etc.).  If the subject does not occupy a reasonable part of the image, then crop the image.  Now use a pretrained image classification CNN as in Lab 10.9.4 to predict the class of each of your images, and report the probabilities for the top five predicted classes for each image.
#
# Note: I didn't have many personal photographs of pets other than my own, so to add some variety I used pictures of other items that I had in my photo library.  I also wanted to throw in some edge cases and see how the model handles them.

# %%
img_dir = './images'

image_names = os.listdir(img_dir)

num_images = len(image_names)

x = []
for img_name in image_names:
    if not img_name.startswith('.'):
        img_path = img_dir + '/' + img_name
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x.append(tf.keras.preprocessing.image.img_to_array(img))

x = np.array(x)

x = tf.keras.applications.imagenet_utils.preprocess_input(x)

# %% tags=[]
cnnmodel = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

# %%
cnnpred = cnnmodel.predict(x)
decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(cnnpred, top=5)
#decoded_preds

# %%
i=0

for img_name in image_names:
    if not img_name.startswith('.'):
        img_path = img_dir + '/' + img_name
        img = mpimg.imread(img_path)

        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        for pred in decoded_preds[i]:
            print(pred)
        
        i+=1

# %% [markdown]
# ### Summary

# %% [markdown]
# Above you can see the pictures I passed to the Convolutional Neural Network (CNN) and the classification with the top 5 highest probabilities.  
#
# The CNN assigned a low probability (25%) of being a minibus to the Volkswagon van, however that's a good classification and is basically correct.  The rest of the predictions are all in a similar vein, trailer truck, school bus, etc.  The CNN was sure it was a vehicle of sorts, so I wonder why its top guess had such a low probability.  If I had to guess, I think the probability would have been higher if there wasn't a surfboard attached to the van.  Since that's such a distinct feature, which isn't likely to be found on the types of pictures the CNN would have been trained on, it made the classification harder.
#
# The CNN assigned a very high (91%) probability that the cake was a cowboy hat, which isn't too bad of a guess to be honest.  With the cake stand, the profile is exactly that of a cowboy hat and is fairly distinct.  Given that the other guesses included sombrero, drum, and gong the profile must be a major factor in the CNN's classification.
#
# Next the CD was incorrectly classified as an odometer.  This is another case of where I probably should have cropped the image because I'd guess the steering wheel in the background negatively affected the classification.  The rest of the guesses aren't related to cars though, so perhaps the pattern on the face of the CD may have also made the classification harder.  Had the CD been flipped, so that the characteristic reflective side was showing, or if the picture had been cropped, the CNN might have made better predictions.
#
# None of the guesses really make a lot of sense for the motor oil picture and the guess with the highest probability (25%) was a carpenter's kit.  With the funnel slightly visible to the side, guesses related to tools make some sense though and the water bottle guess was on the right track, however ash can, vending machine, and pay-phone are way off.
#
# It correctly classified the cup of espresso and the car wheel, but guessed mouse on the Ferrero Rocher chocolate candy that was decorated to look like a golden snitch from Harry Potter, which seems reasonable to me.  The golden snitch's wings make the Ferrero Rocher appear to have whiskers or a tail that's attached to a small body, almost as if it was a mouse.
#
# A correct classification on the wooden spoon, with a reasonable classification of hotdog on what's actually a sandwich.  If you're of the mind that anything with two pieces of bread with something in between them is a sandwich, then a hotdog is a sandwich anyways; it seems that the CNN agrees.  
#  
# Lastly, the CNN assigned a low probability (27%) for its predictions of water jug on the ketchup bottle.  A ketchup bottle is a jug of sorts and ketchup does contain water, so this prediction makes some sense.
#
# There were 4 correct classifications (5 if you consider a hotdog and sandwich to be the same thing), with only two classifications that didn't really make any sense at all (the CD and motor oil).  While the rest of the predictions weren't correct, they were reasonable at least.  Overall, not too bad for an out of the box, pretrained image classification model.

# %% [markdown]
# ## Exercise 9
# Fit a lag-5 autoregressive model to the `NYSE` data, as described in the text and in Lab 10.9.6.  Refit the model with a 12-level factor representing the month.  Does this factor improve the performance of the model?

# %% [markdown]
# ### Lag-5 Linear Autoregressive (AR) logistic regression model with day_of_month variable

# %%
data = robjects.r("""
library(ISLR2)
xdata <- NYSE[, c("DJ_return", "log_volume", "log_volatility")]
""")
with localconverter(robjects.default_converter + pandas2ri.converter):
    xdata = robjects.conversion.rpy2py(data)

xdata = (xdata - xdata.mean()) / xdata.std()

# %%
data = robjects.r("""
library(ISLR2)
xdata <- NYSE[, "train"]
""")
istrain = np.array(data)

istrain = istrain[5:]

training_mask = istrain.astype(bool)

# %%
data = robjects.r("""
library(ISLR2)
xdata <- NYSE[, 'date']
""")

month_df = pd.DataFrame(np.array(data), columns=['date'])
month_df['date'] = pd.to_datetime(month_df['date'])
month_df['month'] = month_df['date'].dt.month
month_df['month_name'] = month_df['date'].dt.month_name()
month_df['day_of_week'] = month_df['date'].dt.day_name()


# %%
def lagm(df, prefix, k=1):
    num_rows, num_cols = df.shape
    
    null_df = pd.DataFrame(np.ones(shape=(k, num_cols)) * np.nan,
                           columns = df.columns)
    
    padded_df = pd.concat([null_df, df[:-k]], ignore_index=True)
    #padded_df.index += 1
    
    for column in padded_df.columns:
        padded_df = padded_df.rename(columns={column: prefix + '_' + column})
    
    return padded_df


# %%
arframe = pd.concat(
    [
        xdata[['log_volume']].reset_index(drop=True),
        lagm(xdata, 'L1', 1), 
        lagm(xdata, 'L2', 2),
        lagm(xdata, 'L3', 3),
        lagm(xdata, 'L4', 4),
        lagm(xdata, 'L5', 5)
    ], axis=1
)

arframe = arframe[5:]

arframemonth = pd.concat([month_df[['month_name', 'day_of_week']][5:], arframe], axis=1)

# %%
arframemonth.head()

# %%
formula_string = formula_from_cols(arframemonth, 'log_volume')
formula_string

# %%
armodelmonth = smf.ols(formula=formula_string, 
                       data = sm.add_constant(arframemonth[training_mask]))

arfitmonth = armodelmonth.fit()

arpredmonth = arfitmonth.predict(arframemonth[~training_mask])

V_0 = arframe[~training_mask]['log_volume'].var()
r_2 = 1 - np.mean((arpredmonth - arframemonth[~training_mask]['log_volume']) ** 2) / V_0
r_2

# %% [markdown]
# ### Comparison

# %% [markdown]
# The $R^2$ of the lag-5 AR model that has been refit with a 12-level factor for the month is 0.4630, which is slightly higher than the $R^2$ of 0.4599 from Lab 10.9.6.

# %% [markdown]
# ## Exercise 10
# In Section 10.9.6, we showed how to fit a linear AR model to the `NYSE` data using the `lm()` (`statsmodelsformulas.ols` in Python) function.  However, we also mentioned that we can "flatten" the short sequences produced for the RNN model in order to fit a linear AR model.  Use this latter approach to fit a linear AR model to the `NYSE` data.  Compare the test $R^2$ of this linear AR model to that of the linear AR model that we fit in the lab.  What are the advantages/disadvantages of each approach?

# %% [markdown]
# ### Lag-5 Linear Autoregressive (AR) neural network model without day_of_week variable

# %%
n = arframe.shape[0]
xrnn = pd.DataFrame(arframe.iloc[:,1:])
xrnn = np.array(xrnn)
xrnn = xrnn.reshape((n, 3, 5), order='F')
xrnn = xrnn[:,:,::-1]
xrnn = xrnn.transpose((0,2,1))
xrnn.shape

# %%
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 1)
    ]
)

model.compile(optimizer='rmsprop', loss='mse')

# %%
history = model.fit(
    x = xrnn[training_mask, :, :],
    y = arframe[training_mask]['log_volume'],
    batch_size = 64, epochs = 200,
    validation_data = (xrnn[~training_mask, :, :], 
                       arframe[~training_mask]['log_volume']),
    verbose = 0
)

kpred = model.predict(xrnn[~training_mask])
r_2 = 1 - np.mean((kpred.flatten() - arframe[~training_mask]['log_volume'])**2) / V_0
r_2

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation']);

# %% [markdown]
# The Linear AR model from the lab using `lm()` (`statsmodelsformulas.ols` in Python) has a test $R^2$ of 0.4132, which is very close to the $R^2$ above generated from using a neural network with a Flatten() layer.  Performance between the two models is very similar.
#
#
# Advantages:
# Need to do
#
# Disadvantanges:
# Need to do

# %% [markdown]
# ## Exercise 11
# Repeat the previous exercise, but now fit a nonlinear AR model by "flattening" the short sequences produced for the RNN model.

# %% [markdown]
# ### Lag-5 Non-Linear AR Recurrent Neural Network (RNN) model without day_of_week variable

# %%
n = arframe.shape[0]
xrnn = pd.DataFrame(arframe.iloc[:,1:])
xrnn = np.array(xrnn)
xrnn = xrnn.reshape((n, 3, 5), order='F')
xrnn = xrnn[:,:,::-1]
xrnn = xrnn.transpose((0,2,1))
xrnn.shape

# %%
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.SimpleRNN(units=15,
                         dropout=0.1,
                         recurrent_dropout=0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1)
    ]
)

model.compile(optimizer='rmsprop', loss='mse')

history = model.fit(
    x = xrnn[training_mask],
    y = arframe[training_mask][['log_volume']],
    epochs = 100, batch_size = 32, 
    validation_data = [xrnn[~training_mask], arframe[~training_mask]['log_volume']],
    verbose = 0
)

npred = model.predict(xrnn[~training_mask])
r_2 = 1 - np.mean((arframe[~training_mask]['log_volume'] - npred.flatten())**2) / V_0
r_2

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation']);

# %% [markdown]
# The $R^2$ of the nonlinear AR RNN model is very close to the linear AR model from the previous exercise.  Both of these are also similar to the $R^2$ generated from using `lm()` (`statsmodelsformulas.ols` in Python) in Lab 10.9.6.

# %% [markdown]
# ## Exercise 12
# Consider the RNN fit to the `NYSE` data in Section 10.9.6. Modify the code to allow inclusion of the variable `day_of_week`, and fit the RNN.  Compute the test $R^2$

# %% [markdown]
# ### Lag-5 Non-Linear AR RNN model with day_of_week variable

# %%
categorical_day_of_week = tf.keras.utils.to_categorical(month_df['date'].dt.dayofweek)

day_of_week_df = pd.DataFrame(categorical_day_of_week, columns=['day[mon]', 'day[tues]', 'day[wed]', 'day[thurs]', 'day[fri]'])

xdatad = pd.concat([xdata.reset_index(drop=True), day_of_week_df], axis=1)

# %%
arframed = pd.concat(
    [
        xdatad[['log_volume']].reset_index(drop=True),
        lagm(xdatad, 'L1', 1), 
        lagm(xdatad, 'L2', 2),
        lagm(xdatad, 'L3', 3),
        lagm(xdatad, 'L4', 4),
        lagm(xdatad, 'L5', 5)
    ], axis=1
)

arframed = arframed[5:]

# %%
arframed.head()

# %%
n = arframed.shape[0]
xrnnd = pd.DataFrame(arframed.iloc[:,1:])
xrnnd = np.array(xrnnd)
xrnnd = xrnnd.reshape((n, 8, 5), order='F')
xrnnd = xrnnd[:,:,::-1]
xrnnd = xrnnd.transpose((0,2,1))
xrnnd.shape

# %%
modeld = tf.keras.models.Sequential(
     [
         tf.keras.layers.SimpleRNN(units=12,
                          dropout=0.1,
                          recurrent_dropout=0.1),
         tf.keras.layers.Dense(units=1)
     ]
)

modeld.compile(optimizer='rmsprop', loss='mse')

# %%
history = modeld.fit(
    x = xrnnd[training_mask, :, :],
    y = arframed[training_mask]['log_volume'],
    batch_size = 64, epochs = 200,
    validation_data = (xrnnd[~training_mask, :, :], 
                       arframed[~training_mask]['log_volume']),
    verbose = 0
)

kpred = modeld.predict(xrnnd[~training_mask])
r_2 = 1 - np.mean((kpred.flatten() - arframed[~training_mask]['log_volume'])**2) / V_0
r_2


# %% [markdown]
# Including the day_of_week_variable using `lm()` (`statsmodelsformulas.ols` in Python) in Lab 10.9.6 slightly increased the $R^2$ to 0.4599 from 0.4132 and we see a similar performance increase when including day_of_week in the SimpleRNN.  Without day_of_week, the $R^2$ was 0.416 in Lab 10.9.6, compared to roughly 0.45 when it's included.

# %% [markdown]
# ## Exercise 13
# Repeat the analysis of Lab 10.9.5 on the `IMDb` data using a similarly structured neural network.  There we used a dictionary size of 10,000.  Conside the effects of varying dictionary size.  Try the values 1000, 3000, 5000, and 10,000, and compare the results.

# %%
def run_sentiment_analysis(max_features):

    import rpy2.robjects as robjects
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from scipy import sparse
    
    def one_hot(sequences, dimension):
        from scipy import sparse
        from itertools import chain

        seqlen = np.array([], dtype=np.int64)
        for seq in sequences:
            seqlen = np.append(seqlen, len(seq))

        n = len(seqlen)

        rowind = np.repeat(np.arange(0, n), repeats=seqlen)

        # Because R starts indexing at 1 and word_index has a minimum value of 1, everything matches up nicely in R.  However, Python starts indexing at 0 and we need to adjust the values of word_index accordingly.  By subtracting 1 from colind, the first word in our dataset will be in the first column (column 0), just like in R where the first word is in the first column (column 1).
        colind = np.array(list(chain(*sequences))) - 1  #taken from https://stackoverflow.com/questions/52573275/get-all-items-in-a-python-list

        sparse_array = np.zeros(shape=(n, dimension))
        sparse_array[rowind, colind] = 1

        sparse_matrix = sparse.coo_matrix(sparse_array)

        return sparse_matrix
    
    
    print(f'Performing analysis with {max_features} features')
    
    #max_features = 10_000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
    
    x_train_1h = one_hot(x_train, max_features)
    x_test_1h = one_hot(x_test, max_features)

    #x_train_1h.shape
    #x_train_1h.count_nonzero() / (25000 * num_features)
    
    data = robjects.r("""
    set.seed(3)
    ival <- sample(seq(1:25000), 2000)
    """)

    ival = np.sort(np.array(data) - 1)
    ival_mask = pd.DataFrame(x_train_1h.toarray()).index.isin(ival)
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ]
    )

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics='accuracy')

    history = model.fit(x=x_train_1h.tocsr()[~ival_mask], 
                        y=y_train[~ival_mask], 
                        epochs=20, batch_size=512, 
                        validation_data=(
                            x_train_1h.tocsr()[ival_mask], y_train[ival_mask]
                        ),
                        verbose = 0
                       )
    
    history2 = model.fit(x = x_train_1h.tocsr()[~ival_mask], 
                     y = y_train[~ival_mask], 
                     epochs=20, batch_size=512, 
                     validation_data=(x_test_1h.tocsr(), y_test),
                     verbose = 0
                    )
    
    fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))
    fig.suptitle('Training/Validation Data')
    
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['Train', 'Validation'])

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].set_title('model accuracy')
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['Train', 'Validation'])
    
    
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))
    fig.suptitle('Training/Test Data')

    ax[0].plot(history2.history['loss'])
    ax[0].plot(history2.history['val_loss'])
    ax[0].set_title('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['Train', 'Test'])

    ax[1].plot(history2.history['accuracy'])
    ax[1].plot(history2.history['val_accuracy'])
    ax[1].set_title('model accuracy')
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['Train', 'Test'])

    plt.tight_layout()
    plt.show()
    
    pred_proba = model.predict(x_test_1h.tocsr())
    pred_classes = pred_proba > 0.5
    acc = np.mean(pred_classes.flatten() == y_test)
    #accuracies.append(acc)
    print()
    print(f'num_features: {max_features}, accuracy: {acc}')
    
    return acc

# %% [markdown]
# ### Dictionary size: 1000

# %%
acc_1000 = run_sentiment_analysis(1000)

# %% [markdown]
# ### Dictionary size: 3000

# %%
acc_3000 = run_sentiment_analysis(3000)

# %% [markdown]
# ### Dictionary size: 5000

# %%
acc_5000 = run_sentiment_analysis(5000)

# %% [markdown]
# ### Dictionary size: 10000

# %%
acc_10000 = run_sentiment_analysis(10_000)

# %%
num_features = [1000, 3000, 5000, 10_000]
accuracies = [acc_1000, acc_3000, acc_5000, acc_1000]

plt.scatter(num_features, accuracies)
plt.xlabel('num_features')
plt.ylabel('accuracy')
plt.axhline(0.84, c='r', ls='--')
plt.axhline(0.85, c='r', ls='--');

# %% [markdown]
# Varying dictionary sizes didn't seem to change the classification accuracy much.  Accuracy seemed to stay within a 1% interval, from ~84% to ~85%.
