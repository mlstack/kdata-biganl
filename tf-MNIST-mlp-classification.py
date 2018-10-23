#!/usr/bin/env /c/Apps/Anaconda3/python
"""
[Title] MLP - Tensorflow Application - Mnist
[Code] tf-MNIST-mlp-classificaton.py
[Author] 이이백(Yibeck.Lee@gmail.com)
"""
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data",one_hot=True)
num_obs_train = mnist.train.num_examples
num_obs_test = mnist.test.num_examples
print(num_obs_train, num_obs_test)

X = tf.placeholder(tf.float32, [None, 784], name="X-feature")
Y = tf.placeholder(tf.float32, [None, 10], name="Y-label")
H1W = tf.Variable(tf.random_normal([784,256]))
H1B = tf.Variable(tf.random_normal([256]))
H1XWB = tf.sigmoid(tf.matmul(X,H1W) + H1B)

H2W = tf.Variable(tf.random_normal([256,10]))
H2B = tf.Variable(tf.random_normal([10]))
Hypothesis = tf.nn.softmax(tf.matmul(H1XWB,H2W) + H2B)
print(Hypothesis)
learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Hypothesis))
print(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
print(optimizer)
correctHypothesis = tf.equal(tf.argmax(Hypothesis,1),tf.argmax(Y,1))
accuracyRate = tf.reduce_mean(tf.cast(correctHypothesis,tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(50000): # 200 -> 500 -> 1000 -> ...
        mini_batch_x, mini_batch_y = mnist.train.next_batch(batch_size=128) # batch_size
        sess.run(optimizer, feed_dict = {X: mini_batch_x, Y: mini_batch_y})
        if step % 1000 == 0:
            step_accuracy_rate = sess.run(accuracyRate, feed_dict = {X:mini_batch_x, Y:mini_batch_y})
            step_cost = sess.run(cost, feed_dict = {X:mini_batch_x, Y:mini_batch_y})
            print(
            	'[step]{:3d}'.format(step)
            ,	'[cost] {:6.5f}'.format(step_cost)
            ,	'[accuracy_rate] {:6.6}'.format(step_accuracy_rate)
            	)
    # test_data = mnist.test.images[:10000].reshape((10000, 28 , 28)) # (batch_size,time_steps,n_input)
    # test_label = mnist.test.labels[:10000]
    num_train = mnist.train.num_examples
    print('num_train : ', num_train)
    num_test = mnist.test.num_examples
    print('num_test : ', num_test)
    train_feature = mnist.train.images[:55000]
    train_label = mnist.train.labels[:55000]
    test_feature = mnist.test.images[:10000]
    test_label = mnist.test.labels[:10000]
    print("Accuracy Rate of Train Data(55000) : ", sess.run(accuracyRate, feed_dict={X: train_feature, Y: train_label}))
    print("Accuracy Rate of Test Data(10000) : ", sess.run(accuracyRate, feed_dict={X: test_feature, Y: test_label}))
    pred_digit = sess.run(
    		tf.argmax(Hypothesis, 1)
    	,	feed_dict={X:test_feature}
    	)
    for i in range(10):
    	print(
    		'{:3d}'.format(i)
    	,	'[pred-acutal] {}-{}'.format(pred_digit[i], sess.run(tf.argmax(test_label[i]))) 
    	)
import numpy as np 
import matplotlib.pyplot as plt
def digit_show(one_dim_arr):
    two_dim_arr = (np.reshape(one_dim_arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_dim_arr, interpolation='nearest')
    return plt
digit_show(one_dim_arr = test_feature[0:1]).show()
digit_show(one_dim_arr = test_feature[1:2]).show()
