#!/usr/bin/python
import csv
import logging
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.models import load_model
from keras.objectives import categorical_crossentropy
from keras.utils import to_categorical

import TensorFI as ti

# Setup
# Random
SEED = 0
np.random.seed(SEED)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(FILE_PATH, "DNN-model/Iris/iris-data-set.csv")
MODEL_PATH = os.path.join(FILE_PATH, "DNN-model/Iris/iris_neural_network.h5")
CONFIG_PATH = os.path.join(FILE_PATH, "DNN-model/Iris/config/default.yaml")
LOGS_PATH = os.path.join(FILE_PATH, "DNN-model/Iris/")

# Load Iris dataset.
with open(DATASET_PATH, 'r') as csvfile:
    iris = list(csv.reader(csvfile))[1:]

num_input = 4
num_classes = 3

# The inputs are four floats: sepal length, sepal width, petal length, petal width.
inputs = np.array(iris)[:, :num_input].astype(
    np.float)  # We select the first three columns.

# Outputs are initially individual strings: setosa, versicolor or virginica.
outputs = np.array(iris)[:, 4]  # We select the 4th column.

# Convert the output strings to ints.
outputs_vals, outputs_ints = np.unique(outputs, return_inverse=True)

# Encode the category integers as binary categorical vairables.
outputs_cats = to_categorical(outputs_ints)

# Split the input and output data sets into training and test subsets.
inds = np.random.permutation(len(inputs))

train_inds, test_inds = np.array_split(inds, 2)
inputs_train, outputs_train = inputs[train_inds], outputs_cats[train_inds]
inputs_test, outputs_test = inputs[test_inds], outputs_cats[test_inds]

# TF-Keras Session
sess = tf.Session()
K.set_session(sess)

init = tf.global_variables_initializer()
sess.run(init)

# Model
K.set_learning_phase(0)
model = load_model(MODEL_PATH, custom_objects=None, compile=True)
model.summary()
x = model.input
y = model.output

# Evaluate model
with sess.as_default():
    output_value = sess.run([y], feed_dict={x: inputs_test})
    accuracy = tf.reduce_mean(
        categorical_accuracy(outputs_test, output_value[0])).eval()
    loss = tf.reduce_mean(
        categorical_crossentropy(outputs_test, output_value[0])).eval()
    print("Loss/Accuracy (W/O FI) = ", [loss, accuracy])

# Run the model and print the accuracy
fi = ti.TensorFI(sess,
                 configFileName=CONFIG_PATH,
                 logDir=LOGS_PATH,
                 logLevel=logging.WARNING,
                 disableInjections=True,
                 name="IrisInjection",
                 fiPrefix="fi_")
fi.turnOnInjections()

# writer used to save graph. Use tensorboard to view.
writer = tf.compat.v1.summary.FileWriter(LOGS_PATH, sess.graph)

with sess.as_default():
    output_value = sess.run([y], feed_dict={x: inputs_test})
    accuracy = tf.reduce_mean(
        categorical_accuracy(outputs_test, output_value[0])).eval()
    loss = tf.reduce_mean(
        categorical_crossentropy(outputs_test, output_value[0])).eval()
    print("Loss/Accuracy (FI) = ", [loss, accuracy])

writer.close()
print("Done running instrumented model")
