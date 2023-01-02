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

# %%
import pandas as pd
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

import statsmodels.formula.api as smf
import statsmodels.api as sm

import patsy

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import os

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPredict import glmnetPredict

from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict

# %% [markdown]
# # 10.9.1 A Single Layer Network on the Hitters Data

# %% [markdown]
# The lab starts off by reading baseball player data from Hitters.csv and then randomly generating a list of numbers, called testids.  Testids are indices referring to rows in our testing dataset.  This list of numbers will separate the training data from the testing data when creating a model and assessing its performance.
#
# In order to reproduce the results from the part of the lab in Python, I need to get the same list of testids in Python.  Unfortunately, there is no easy way to generate them in Python, however I can generate them in R and send them over to Python quite easily by using the [rpy2](https://rpy2.github.io/) module.
#
# After I have the testids, I adjust the index of the baseball data to start at 1.  Because R starts indexing at 1, but Python starts at 0, adjusting the indices ensures that I'm using the correct rows that were used in the lab.  Failure to do so would lead to different results.

# %%
data = robjects.r("""
library(ISLR2)
n <- nrow(na.omit(Hitters))
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)
""")

Gitters = pd.read_csv('../../../datasets/Hitters.csv')

Gitters = Gitters.dropna()

# R starts indexing at 1, Python starts indexing at 0, this is to make sure the indices of the dataframes match


Gitters = Gitters.set_index(keys=np.arange(1, len(Gitters) + 1)) 

testid = np.sort(np.array(data))
test_mask = Gitters.index.isin(testid)


# %% [markdown]
# In R, many models can be built using the `forumla=` parameter, which behaves like this:
#
# `formula = OutputFeature ~ InputFeature1 + InputFeature 2 + ...`
#
# When you want to use all of the features in your dataset to predict/approximate the `OutputFeature`, there's a special syntax that simplies the process, where a `.` can be used to refer to all columns in the dataset that aren't your `OutputFeature`:
#
# `formula = OutputFeature ~ .`
#
# To fit the model in R, we can use the `lm()` function, but in Python there's the `OLS()` function from the statsmodels library.  Unfortunately, `OLS()` doesn't have the same special syntax to refer to many columns at once and each individual column has to be explcitly typed into `formula=`.
#
# To simplify this process, I define a function `formula_from_cols()`, which will build the formulas for me.

# %%
def formula_from_cols(df, y, use_target, remove_intercept):
    formula_string = ' + '.join([col for col in df.columns if not col==y])
    if use_target == True:
        formula_string = y + ' ~ ' + formula_string
    if remove_intercept == True:
        formula_string = formula_string + ' - 1'
    return formula_string


# %%
formula_string = formula_from_cols(Gitters, 'Salary', use_target = True, remove_intercept=False)

lmodel = smf.ols(formula=formula_string, data = sm.add_constant(Gitters[~test_mask]))

lfit = lmodel.fit()

lpred = lfit.predict(Gitters[test_mask])

np.mean(abs(lpred - Gitters[test_mask]['Salary']))

# %% [markdown]
# Using this linear regression model to predict a baseball player's salary, the mean absolute error (MAE) is the same as the lab from the textbook.

# %% [markdown]
# Next, the lab goes on to fit a lasso regresson model on the same dataset, however the dataset needs some preprocessing beforehand.  In R, the lab uses the `model.matrix()` function to create what's called a [design matrix](https://www.statlect.com/glossary/design-matrix), but in Python we can use the [Patsy](https://patsy.readthedocs.io/en/latest/) library and the `dmatrix()` function.  
#
# Among other things, Patsy will take categorical variables, such as `League` or `Division` from the baseball dataset and convert them into dummy variables.  We can also use Patsy to scale the dataset, however we need to set `ddof=1` so that Patsy divides by the square root of the unbiased estimator of the variance as opposed to the maximum likelihood estimate.

# %%
formula_string = formula_from_cols(Gitters, 'Salary', use_target = False, remove_intercept=True)

x = patsy.dmatrix(formula_like = 'AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + League + Division + PutOuts + Assists + Errors + NewLeague - 1', data = Gitters)

x_scale = patsy.scale(x, ddof=1)

y = np.array(Gitters['Salary'])

# %%
x.design_info.column_names

# %%
X_train = x_scale[~test_mask]
X_test = x_scale[test_mask]

y_train = Gitters[~test_mask]['Salary']
y_test = Gitters[test_mask]['Salary']

cvfit = cvglmnet(x = x_scale[~test_mask], y = y[~test_mask], ptype='mae')

cpred = cvglmnetPredict(cvfit, newx = x_scale[test_mask], s = 'lambda_min')

# %%
np.mean(abs(y[test_mask] - cpred))

# %% [markdown]
# I'm getting a much higher MAE than the ISLR textbook does (they get ~253) and I'm not entirely sure why.  As far as I can tell, the inputs, `x_scale` and `y` are the same as in the textbook, however the predictions (`cpred`) are different, leading to a different MAE.  I need to investigate further.

# %% [markdown]
# Lastly, the lab uses the `keras` library in R to create a neural network to predict Salary from the baseball dataset.  The neural model has one hidden layer with 50 units that activate based on the `relu` activation function, then a dropout layer with a dropout rate of 40%, and finally an output layer with a single output, the predicted salary.
#
# The neural network will treat each row of our design matrix as an input vector.  Because the design matrix has 20 columns, each input vectors will have 20 features, one for each column.  Each feature of the input vectors will be fed into the 50 units in the first hidden layer.  The output of this first hidden layer will be either 0 or 1 based on the `relu` function, where an output of 1 means that the neuron "fired".  These outputs are then fed into the next layer, the dropout layer, where 40% of the outputs will be excluded and before being fed into the last layer, the output layer.

# %%
modnn = keras.Sequential(
    [
            layers.Dense(units = 50, activation='relu'),
            layers.Dropout(rate=0.4),
            layers.Dense(units=1)
    ]
)

# %% [markdown]
# After defining the neural network model, we compile and fit it to our dataset, then view the model's performance over time.

# %%
modnn.compile(
    loss=tf.keras.losses.MeanSquaredError(), 
    optimizer='rmsprop', 
    metrics=tf.keras.losses.MeanAbsoluteError()
)

# %% tags=[]
history = modnn.fit(X_train, 
                    y_train, 
                    epochs=1500, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    verbose=0)

# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))

ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].set_title('model loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend(['Train', 'Validation'])

ax[1].plot(history.history['mean_absolute_error'])
ax[1].plot(history.history['val_mean_absolute_error'])
ax[1].set_title('model mean_abs_error')
ax[1].set_ylabel('mean_abs_error')
ax[1].set_xlabel('epoch')
ax[1].legend(['Train', 'Validation'])

plt.tight_layout();

# %%
npred = modnn.predict(X_test)

# %% tags=[]
np.mean(abs(y_test - npred.flatten()))

# %% [markdown]
# The graph above shows that after about 800 training epochs, the validation MAE settles down and remains constant roughly at 250.  This is close to what the book gets (they get 257.43) and is close to what our linear regression model in the first part of this lab got (254.6687).  
#
# Because the dropout layer drops at random, I don't think it's possible to reproduce the results from the lab in R exactly, however the MAE is close enough and the plots above are quite similar to the plots in R, such that I'm confident I've reproduced the lab's results as much as possible.

# %% [markdown]
# # 10.9.2 A Multilayer Network on the MNIST Digit Data

# %% [markdown]
# This lab demonstrates the ability of a neural network to recognize and classify digits 0-9 from pictures of hand written digits.  The mnist dataset, which can be found in the `keras` library contains many such images and is first loaded into a training and testing dataset

# %%
(x_train, g_train), (x_test, g_test) = keras.datasets.mnist.load_data()

# %%
x_train.shape

# %%
x_test.shape

# %%
x_train = np.reshape(x_train, newshape=(x_train.shape[0], 784))
x_test = np.reshape(x_test, newshape=(x_test.shape[0], 784))

y_train = keras.utils.to_categorical(g_train, num_classes=10)
y_test = keras.utils.to_categorical(g_test, num_classes=10)

# %% [markdown]
# The lab describes how neural networks are sensitive to the scale of the inputs.  Since our dataset consists of grayscale images, each pixel in the image will have a value between 0 and 255.  Dividing by 255 will scale all values down such that they are now between 0 and 1, which is equivalent to what's known as [Min-Max scaling](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).

# %%
x_train = x_train / 255
x_test = x_test / 255

# %% [markdown]
# The neural network in this lab increases in complexity by adding a second hidden layer.  
#
# Each feature of an input vector will first be fed into each of the 256 neurons in the first hidden layer, where 40% of the outputs will be dropped, before feeding the remainder into each of the 128 neurons in the second hidden layer.  Finally, 30% of the outputs from the second hidden layer will be dropped before being passed to the 10 units in the final output layer.
#
# The output layer uses the 'softmax' activation function, which will compute the probability for each of the 10 units in the output layer, which will correspond to the probability of each digit 0-9 being written in a given picture.

# %%
modelnn = keras.Sequential(
    [
            layers.Dense(units=256, activation='relu'),
            layers.Dropout(rate=0.4),
            layers.Dense(units=128, activation='relu'),
            layers.Dropout(rate=0.3),
            layers.Dense(units=10, activation='softmax')
    ]
)

# %%
modelnn.compile(
    loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics='accuracy'
)

# %% tags=[]
history = modelnn.fit(x_train, 
                      y_train, 
                      epochs=30, 
                      batch_size=128, 
                      validation_split=0.2,
                      verbose=0)

# %%
modelnn.summary()

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

# %% [markdown]
# The `softmax` activation function will output the probability of each of the 10 digits being written in the photo, like so:
#     
# `[prob_of_digit_0, prob_of_digit_1, ..., prob_of_digit9]`
#
# Using these probabilities, we can base the prediction on which digit has the highest probability.
#
# The `np.argmax()` function with `axis=1` will return the index of the maximum value (probability) in each row.  Since NumPy arrays begin indexing at 0, the index of the row will correspond to digit with the highest probability, the one that we want to use as the prediction.

# %%
y_proba = modelnn.predict(x_test)
y_pred_classes = np.argmax(y_proba, axis=1)

# %%
y_pred_classes

# %%
modelnn_acc = np.mean(y_pred_classes == g_test)
modelnn_acc

# %% [markdown]
# The neural network does well with 98% accuracy, close to the accuracy from the textbook (0.9813).  Again, because the dropout layers drop randomly, I don't think it's possible to recreate the results from the lab exactly, but these are very close.

# %% [markdown]
# ## Multiclass Logistic Regression

# %% [markdown]
# This part of the lab shows how `keras` can perform multiclass logistic regression.  The lab mentions that `glmnet` library, which was used in lab 10.9.1, can also perform multiclass logistic regression, `keras` does it more quickly.  
#
# When constructing the `keras` model there is just a single output layer with 10, which uses the `softmax` activation function, similar to before.

# %%
modellr = keras.models.Sequential(
    [
            layers.Dense(units=10, activation='softmax')
    ]
)

# %%
modellr.compile(
    loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics='accuracy')

# %%
history = modellr.fit(x_train, 
            y_train, 
            epochs=30, 
            batch_size=128, 
            validation_split=0.2,
            verbose=0)

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

# %% [markdown]
# The `np.argmax()` function needs to be used again to determine which digit has the highest probability, in order to make the prediction,

# %%
y_proba = modellr.predict(x_test)
y_pred_classes = np.argmax(y_proba, axis=1)

# %%
modellr_acc = np.mean(y_pred_classes == g_test)
modellr_acc

# %% [markdown]
# The accuracy for this logistic regression model is quite close to what was found in the lab (0.9286).  It's a little lower when compared to ~98% in the neural network, but the model is far simpler and if interpretability matters, would be way easier to interpret.

# %% [markdown]
# # 10.9.3 Convolutional Neural Networks

# %%
(x_train, g_train), (x_test, g_test) = keras.datasets.cifar100.load_data()

x_train.shape

# %%
x_train[0, :, :, 0].min(), x_train[0, :, :, 0].max()

# %%
x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.to_categorical(g_train, num_classes=100)

y_train.shape

# %% [markdown]
# Some code below borrowed from: https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html#cifar100-single

# %%
rows, columns = 5,5
rand_idx = np.random.randint(0, 50000, rows * columns)
rand_images = x_train[rand_idx]
fig = plt.figure(figsize=(8,10))
for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(rand_images[i-1])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# %%
model = keras.models.Sequential(
    [
        layers.Conv2D(filters=32, kernel_size=(3,3),
                     padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=64, kernel_size=(3,3),
                     padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=128, kernel_size=(3,3),
                     padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(rate=0.5),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=100, activation='softmax')
    ]
)

# %%
model.compile(
    loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics='accuracy'
)

history = model.fit(x_train, 
                    y_train, 
                    epochs=30, 
                    batch_size=128, 
                    validation_split=0.2,
                    verbose=0)

y_proba = model.predict(x_test)
y_pred_classes = np.argmax(y_proba, axis=-1)

np.mean(y_pred_classes == g_test.flatten())

# %%
model.summary()

# %% [markdown]
# # 10.9.4 Using Pretrained CNN Models

# %% [markdown]
# Help with keras preprocessing helper functions in Python found here: https://pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/

# %%
# Help with keras preprocessing helper functions found here: https://pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/

img_dir = './book_images'

image_names = os.listdir(img_dir)

num_images = len(image_names)

x = []
for img_name in image_names:
    if not img_name.startswith('.'):
        img_path = img_dir + '/' + img_name
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x.append(keras.preprocessing.image.img_to_array(img))

x = np.array(x)

x = preprocess_input(x)

# %% tags=[]
model = keras.applications.resnet50.ResNet50(weights='imagenet')

model.summary()

# %%
pred6 = model.predict(x)
keras.applications.imagenet_utils.decode_predictions(pred6, top=3)

# %% [markdown]
# # 10.9.5 IMDb Document Classification

# %%
max_features = 10_000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# %%
x_train[0][0:12]

# %%
word_index = keras.datasets.imdb.get_word_index()

def decode_review(text: list, word_index: dict):
    new_strings = ["<PAD>", "<START>", "<UNK>", "<UNUSED>"]
    
    idx = list(word_index.values())
    idx_plus_3 = np.array(idx) + 3
    new_idx = np.append(np.arange(0,4), idx_plus_3)
    new_idx = list(new_idx)
    
    word = list(word_index.keys())
    word = new_strings + word
    
    words = []
    
    for word_val in text:
        if word_val not in new_idx:
            word_val = 2 #use idx 2 to return "<UNK>" when idx of word can't be found
        words.append(word[new_idx.index(word_val)])

    print(" ".join(words))
    
decode_review(x_train[0][0:12], word_index)


# %%
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


# %%
x_train_1h = one_hot(x_train, 10000)
x_test_1h = one_hot(x_test, 10000)

x_train_1h.shape

# %%
x_train_1h.count_nonzero() / (25000 * 10000)

# %%
data = robjects.r("""
set.seed(3)
ival <- sample(seq(1:25000), 2000)
""")

ival = np.sort(np.array(data) - 1)
ival_mask = pd.DataFrame(x_train_1h.toarray()).index.isin(ival)

# %% [markdown]
# This article was used as a reference for how to select rows from a sparse matrix so that I could filter using the ival_mask to separate training and testing observations: 
#
# https://cmdlinetips.com/2019/07/how-to-slice-rows-and-columns-of-sparse-matrix-in-python/

# %%
fitlm = glmnet(x=x_train_1h.tocsr()[~ival_mask].toarray(),
               y=y_train[~ival_mask].astype(np.float64), 
               family='binomial', 
               standardize=True)

classlmv = glmnetPredict(fitlm, x_train_1h.tocsr()[ival_mask]) > 0

acclmv = []
for i in range(100):
    acc = np.mean(np.array(classlmv[:, i]).flatten() == (y_train[ival_mask] > 0))
    acclmv.append(acc)

# %%
plt.scatter(-np.log(fitlm['lambdau']), acclmv)
plt.xlabel("$-log(fitlm(\lambda u))$")
plt.ylabel("acclmv");

# %%
glmnetPlot(fitlm, xvar='lambda');

# %%
model = keras.models.Sequential(
    [
        layers.Dense(units=16, activation='relu'),
        layers.Dense(units=16, activation='relu'),
        layers.Dense(units=1, activation='sigmoid')
    ]
)

model.compile(
    loss='binary_crossentropy', 
    optimizer='rmsprop', 
    metrics='accuracy'
)

history = model.fit(x=x_train_1h.tocsr()[~ival_mask], 
                    y=y_train[~ival_mask], 
                    epochs=20, 
                    batch_size=512, 
                    validation_data=(x_train_1h.tocsr()[ival_mask], 
                                     y_train[ival_mask]),
                    verbose=0)

# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))

fig.suptitle("Training/Validation Data")

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

# %% tags=[]
history2 = model.fit(x = x_train_1h.tocsr()[~ival_mask], 
                     y = y_train[~ival_mask], 
                     epochs=20, 
                     batch_size=512, 
                     validation_data=(x_test_1h.tocsr(), y_test),
                     verbose=0)

# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,6))

fig.suptitle("Training/Testing Data")

ax[0].plot(history2.history['loss'])
ax[0].plot(history2.history['val_loss'])
ax[0].set_title('model loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend(['Train', 'Validation'])

ax[1].plot(history2.history['accuracy'])
ax[1].plot(history2.history['val_accuracy'])
ax[1].set_title('model accuracy')
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].legend(['Train', 'Validation'])

plt.tight_layout();

# %% [markdown]
# # 10.9.6 Recurrent Neural Networks

# %%
max_features = 10_000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# %%
wc = []
for seq in x_train:
    seq_len = len(seq)
    wc.append(seq_len)
    
wc = np.array(wc)

np.median(wc)

# %%
sum(wc <= 500) / len(wc)

# %%
maxlen = 500
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)


# %%
def pad_sequences(data, maxlen):

    padded = []

    for row in data:
        row_len = len(row)
        abs_diff = abs(maxlen - row_len)
        if row_len < maxlen:
            padded.append(np.pad(array=row, pad_width=(abs_diff, 0)))
        else:
            padded.append(np.array(row[abs_diff:]))
           
    padded_array = np.array(padded)
    
    return padded_array


# %%
x_train = pad_sequences(x_train, maxlen)
x_test = pad_sequences(x_test, maxlen)

x_train.shape

# %%
x_test.shape

# %%
x_train[0, 489:500]

# %%
model = keras.models.Sequential(
    [
        layers.Embedding(input_dim = 10000, output_dim = 32),
        layers.LSTM(units = 32),
        layers.Dense(units = 1, activation = 'sigmoid')
    ]
)

# %%
model.compile(
    optimizer = 'rmsprop', 
    loss='binary_crossentropy', 
    metrics='accuracy'
)

history = model.fit(x = x_train, 
                    y = y_train,
                    epochs = 10, 
                    batch_size = 128,
                    validation_data = (x_test, y_test),
                    verbose=0)

# %%
predy = model.predict(x_test) > 0.5

np.mean(y_test == predy.flatten())

# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize = (8,6))

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

# %% [markdown]
# ## Time Series Prediction

# %% tags=[]
data = robjects.r("""
library(ISLR2)
xdata <- NYSE[, c("DJ_return", "log_volume", "log_volatility")]
""")
with localconverter(robjects.default_converter + pandas2ri.converter):
    xdata = robjects.conversion.rpy2py(data)
    
data = robjects.r("""
library(ISLR2)
xdata <- NYSE[, "train"]
""")
istrain = np.array(data)

xdata = (xdata - xdata.mean()) / xdata.std()


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

# %%
arframe = arframe[5:]
istrain = istrain[5:]

# %%
training_mask = istrain.astype(bool)

formula_string = formula_from_cols(arframe, 'log_volume')

armodel = smf.ols(formula=formula_string, data = sm.add_constant(arframe[training_mask]))

arfit = armodel.fit()

arpred = arfit.predict(arframe[~training_mask])
V_0 = arframe[~training_mask]['log_volume'].var()
1 - np.mean((arpred - arframe[~training_mask]['log_volume'])**2) / V_0

# %% tags=[]
data = robjects.r("""
library(ISLR2)
xdata <- NYSE[, 'day_of_week']
""")
day_df = pd.DataFrame(np.array(data), columns=['day'])

arframed = pd.concat([day_df[5:], arframe], axis=1)

formula_string = formula_from_cols(arframed, 'log_volume')

armodeld = smf.ols(formula=formula_string, data = sm.add_constant(arframed[training_mask]))

arfitd = armodeld.fit()

arpredd = arfitd.predict(arframed[~training_mask])
1 - np.mean((arpredd - arframed[~training_mask]['log_volume']) ** 2) / V_0

# %%
n = arframe.shape[0]
xrnn = pd.DataFrame(arframe.iloc[:,1:])
xrnn = np.array(xrnn)
xrnn = xrnn.reshape((n, 3, 5), order='F')
xrnn = xrnn[:,:,::-1]
xrnn = xrnn.transpose((0,2,1))
xrnn.shape

# %%
model = keras.models.Sequential(
    [
        layers.SimpleRNN(units=12,
                         dropout=0.1,
                         recurrent_dropout=0.1),
        layers.Dense(units=1)
    ]
)

model.compile(optimizer='rmsprop', loss='mse')

# %% tags=[]
history = model.fit(
    x = xrnn[training_mask, :, :],
    y = arframe[training_mask]['log_volume'],
    batch_size = 64, epochs = 200,
    validation_data = (xrnn[~training_mask, :, :], 
                       arframe[~training_mask]['log_volume']),
    verbose=0
)

kpred = model.predict(xrnn[~training_mask])
1 - np.mean((kpred.flatten() - arframe[~training_mask]['log_volume'])**2) / V_0

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation']);

# %%
x = patsy.dmatrix(formula_like = 'day + L1_DJ_return + L1_log_volume + L1_log_volatility + L2_DJ_return + L2_log_volume + L2_log_volatility + L3_DJ_return + L3_log_volume + L3_log_volatility + L4_DJ_return + L4_log_volume + L4_log_volatility + L5_DJ_return + L5_log_volume + L5_log_volatility - 1', data = arframed)

x.design_info.column_names

# %%
arnnd = keras.models.Sequential(
    [
        layers.Dense(units = 32, activation = 'relu'),
        layers.Dropout(rate = 0.5),
        layers.Dense(units = 1)
    ]
)

arnnd.compile(loss = 'mse', optimizer = 'rmsprop')

history = arnnd.fit(
    x = x[training_mask],
    y = arframe[training_mask]['log_volume'],
    epochs = 100, batch_size = 32, 
    validation_data = [x[~training_mask], arframe[~training_mask]['log_volume']],
    verbose = 0
)

npred = arnnd.predict(x[~training_mask])
1 - np.mean((arframe[~training_mask]['log_volume'] - npred.flatten())**2) / V_0

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation']);

# %% [markdown]
# The End
