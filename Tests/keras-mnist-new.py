#!/usr/bin/python
# MNIST dataset recognition using Keras - example taken from the Keras tutorial
#  https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#calling-keras-layers-on-tensorflow-tensors

from __future__ import print_function
import sys
import imp

import tensorflow as tf
import TensorFI as ti

from tensorflow.keras.layers import Dense

# Keras layers can be called on TensorFlow tensors:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Place-holder for the labels and loss function
labels = tf.placeholder(tf.float32, shape=(None, 10))

from tensorflow.keras.losses import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Model training with MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

# Run the model and print the accuracy
from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels, preds)
with sess.as_default():
    print( "Accuracy = ",  acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}) )
print("Done running model");

# Instrument the graph with TensorFI
fi = ti.TensorFI(sess, logLevel = 100)
fi.turnOnInjections()
with sess.as_default():
    print( "Accuracy = ",  acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}) )
print("Done running instrumented model");
