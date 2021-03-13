#!/usr/bin/python
# MNIST dataset recognition using Keras - example taken from the Keras tutorial
#  https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#calling-keras-layers-on-tensorflow-tensors

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data

import TensorFI as ti

sess = tf.compat.v1.Session()

K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(
    img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(
    x)  # output layer with 10 units and a softmax activation

# Place-holder for the labels and loss function
labels = tf.placeholder(tf.float32, shape=(None, 10))

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0], labels: batch[1]})

with sess.as_default():
    y_preds = preds.eval(feed_dict={img: mnist_data.test.images})
    accuracy = tf.reduce_mean(
        categorical_accuracy(mnist_data.test.labels, y_preds)).eval()
    loss = tf.reduce_mean(
        categorical_crossentropy(mnist_data.test.labels, y_preds)).eval()
    print("Loss/Accuracy (W/O FI) = ", [loss, accuracy])
print("Done running model")

# Instrument the graph with TensorFI
fi = ti.TensorFI(sess, logLevel=100)
fi.turnOnInjections()
with sess.as_default():
    y_preds = preds.eval(feed_dict={img: mnist_data.test.images})
    accuracy = tf.reduce_mean(
        categorical_accuracy(mnist_data.test.labels, y_preds)).eval()
    loss = tf.reduce_mean(
        categorical_crossentropy(mnist_data.test.labels, y_preds)).eval()
    print("Loss/Accuracy (FI) = ", [loss, accuracy])
print("Done running instrumented model")
