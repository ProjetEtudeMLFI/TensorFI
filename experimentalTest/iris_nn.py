#!/usr/bin/python
import csv
import numpy as np
import tensorflow as tf
import TensorFI as ti

from keras.utils import to_categorical
from keras import backend as K
from keras.models import load_model
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy

# Setup
# Remove deprecated warnings from TensorFlow
tf.logging.set_verbosity(tf.logging.FATAL)

# Random
SEED = 0
np.random.seed(SEED)

# Load Iris dataset.
iris = list(csv.reader(open('iris-data-set.csv')))[1:]

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
x = tf.placeholder(tf.float32, shape=(None, 4))
model = load_model('iris_neural_network.h5', custom_objects=None, compile=True)
model.summary()
y = model(x)

# Evaluate model
with sess.as_default():
    output_value = sess.run([y], feed_dict={x: inputs_test})
    # print("Predictions (W/O FI) = ", output_value)
    # print("Expected = ", outputs_test)
    accuracy = tf.reduce_mean(
        categorical_accuracy(outputs_test, output_value[0])).eval()
    loss = tf.reduce_mean(
        categorical_crossentropy(outputs_test, output_value[0])).eval()
    print("Loss/Accuracy (W/O FI) = ", [loss, accuracy])

# Run the model and print the accuracy
fi = ti.TensorFI(sess,
                 configFileName="confFiles/default.yaml",
                 logDir="faultLogs/",
                 logLevel=100,
                 disableInjections=True,
                 name="IrisInjection",
                 fiPrefix="fi_")
fi.turnOnInjections()

# Save graph
writer = tf.summary.FileWriter("logs", sess.graph)
with sess.as_default():
    output_value = sess.run([y], feed_dict={x: inputs_test})
    # print("Predictions (FI) = ", output_value)
    # print("Expected = ", outputs_test)
    try:
        accuracy = tf.reduce_mean(
            categorical_accuracy(outputs_test, output_value[0])).eval()
        loss = tf.reduce_mean(
            categorical_crossentropy(outputs_test, output_value[0])).eval()
        print("Loss/Accuracy (W/O FI) = ", [loss, accuracy])
    except ValueError as e:
        print("Loss/Accuracy (W/O FI) = ", e)

writer.close()
print("Done running instrumented model")
