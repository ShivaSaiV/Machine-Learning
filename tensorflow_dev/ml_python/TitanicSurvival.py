from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
# print(dftrain.head())
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# print(dftrain.head())

# print(y_train)
# print(dftrain.age.hist(bins=20))
# plt.show()

categorical_d = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
numerical_d = ['age', 'fare']

feature_column = []

for feature in categorical_d:
    vocab = dftrain[feature].unique()
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab))

for feature in numerical_d:
    feature_column.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_model = tf.estimator.LinearClassifier(feature_columns=feature_column) # Creating the model
linear_model.train(train_input_fn) # First training the model

result = linear_model.evaluate(eval_input_fn) # Result of the model trained

clear_output()
print("Accuracy = " + str(result["accuracy"]))

result = list(linear_model.predict(eval_input_fn))
print(result[0])
