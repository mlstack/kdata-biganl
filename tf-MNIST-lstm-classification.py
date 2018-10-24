#!/usr/bin/env /c/Apps/Anaconda3/python
"""
[Title] LSTM - Tensorflow Application - Mnist
[Code] tf-MNIST-lstm-classificaton.py
[Author] 이이백(Yibeck.Lee@gmail.com)
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data",one_hot=True)
X = tf.placeholder("float",[None,28,28]) # [None,time_steps,n_input]
Y = tf.placeholder("float",[None,10]) # [None,n_classes]
Xrnninput = tf.unstack(X ,28,1) # time_steps
LstmCell = tf.nn.rnn_cell.BasicLSTMCell(128,forget_bias=1) # num_units
LstmCellOutputs, _ = tf.nn.static_rnn(LstmCell, Xrnninput, dtype="float32")
LstmCellWOutW = tf.Variable(tf.random_normal([128,10])) # [num_units,n_classes]
LstmCellWOutB = tf.Variable(tf.random_normal([10])) # [n_classes]
Hypothesis = tf.matmul(LstmCellOutputs[-1],LstmCellWOutW) + LstmCellWOutB
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Hypothesis,labels=Y))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correctHypothesis = tf.equal(tf.argmax(Hypothesis,1),tf.argmax(Y,1))
accuracyRate = tf.reduce_mean(tf.cast(correctHypothesis,tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000): # 200 -> 500 -> 1000
        mini_batch_x, mini_batch_y = mnist.train.next_batch(batch_size=128) # batch_size
        mini_batch_x = mini_batch_x.reshape((128, 28, 28)) # (batch_size,time_steps,n_input)
        sess.run(optimizer, feed_dict = {X: mini_batch_x, Y: mini_batch_y})
        if step % 100 == 0:
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
    train_feature = mnist.train.images[:55000].reshape((55000, 28 , 28)) # (batch_size,time_steps,n_input)
    train_label = mnist.train.labels[:55000]
    test_feature = mnist.test.images[:10000].reshape((10000, 28 , 28)) # (batch_size,time_steps,n_input)
    test_label = mnist.test.labels[:10000]
    print("Accuracy Rate of Train Data(55000) : ", sess.run(accuracyRate, feed_dict={X: train_feature, Y: train_label}))
    print("Accuracy Rate of Test Data(10000) : ", sess.run(accuracyRate, feed_dict={X: test_feature, Y: test_label}))
    pred_digit = sess.run(
    		tf.argmax(Hypothesis, 1)
    	,	feed_dict={X:test_feature}
    	)
    for i in range(2):
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
