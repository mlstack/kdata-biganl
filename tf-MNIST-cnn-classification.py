#!/usr/bin/env /c/Apps/Anaconda3/python
"""
[Title] CNN - Tensorflow Application - Mnist
[Code] tf-MNIST-cnn-dropout-classificaton.py
[Author] 이이백(Yibeck.Lee@gmail.com)
"""
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data",one_hot=True)
num_obs_train = mnist.train.num_examples
num_obs_test = mnist.test.num_examples
print(num_obs_train, num_obs_test)

X = tf.placeholder(tf.float32, [None, 784], name="X-feature")
print('[X] ', X)
Y = tf.placeholder(tf.float32, [None, 10], name="Y-label")
Ximage = tf.reshape(
		tensor = X
	,	shape = [-1,28,28,1]
	)
print('[Ximage] ', Ximage)
keep_prob= tf.placeholder(tf.float32)

H1ConvW = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
H1ConvB = tf.Variable(tf.zeros([32]))
H1ConvXreshapeWpB = tf.nn.relu(
	tf.nn.conv2d(
		input = Ximage
	,	filter = H1ConvW
	,	strides = [1,1,1,1]
	,	padding = 'SAME'
		)
	+ H1ConvB
	)
print('[H1ConvXreshapeWpB] ', H1ConvXreshapeWpB)
H1ConvMaxpool = tf.nn.max_pool(
		value = H1ConvXreshapeWpB
	,	ksize = [1,2,2,1]
	,	strides = [1,2,2,1]
	,	padding = 'SAME'
	)
print('[H1ConvXreshapeWpB] ', H1ConvMaxpool)
H2ConvW = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
H2ConvB = tf.Variable(tf.zeros([64]))
H2ConvH1ConvMaxpoolWpB = tf.nn.relu(
	tf.nn.conv2d(
		input = H1ConvMaxpool
	,	filter = H2ConvW
	,	strides = [1,1,1,1]
	,	padding = 'SAME'
		)
	+ H2ConvB
	)
print(H2ConvH1ConvMaxpoolWpB)
H2ConvMaxpool = tf.nn.max_pool(
		value = H2ConvH1ConvMaxpoolWpB
	,	ksize = [1,2,2,1]
	,	strides = [1,2,2,1]
	,	padding = 'SAME'

	)
print('[H2ConvMaxpool] ', H2ConvMaxpool)
H2ConvMaxpoolFlat = tf.reshape(H2ConvMaxpool,[-1,7*7*64])
print('[H2ConvMaxpoolFlat] ', H2ConvMaxpoolFlat)
Fc1W = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
Fc1B = tf.Variable(tf.zeros([1024]))
Fc1out_ = tf.nn.relu(tf.matmul(H2ConvMaxpoolFlat,Fc1W) + Fc1B)
Fc1out = tf.nn.dropout(Fc1out_, keep_prob)
print('[Fc1out] ', Fc1out)
Fc2W = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
Fc2B = tf.Variable(tf.zeros([10]))

Hypothesis = tf.nn.softmax(tf.matmul(Fc1out,Fc2W) + Fc2B)
print('[Hypothesis] ',Hypothesis)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Hypothesis),reduction_indices=[1]))
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Hypothesis)))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Hypothesis))
print('[cost] ',cost)
learning_rate = 0.0001
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
print('[optimizer] ',optimizer)

correctHypothesis  = tf.equal(tf.argmax(Hypothesis, 1), tf.argmax(Y, 1))
accuracyRate =  tf.reduce_mean(tf.cast(correctHypothesis, tf.float32))
init = tf.global_variables_initializer()
print('[init] ',init)
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000): # 200 -> 500 -> 1000 -> ...
        mini_batch_x, mini_batch_y = mnist.train.next_batch(batch_size=100) # batch_size
        sess.run(
        	optimizer
        , 	feed_dict = {X: mini_batch_x, Y: mini_batch_y,	keep_prob: 0.5}
        )
        if step % 100 == 0:
            step_accuracy_rate = sess.run(accuracyRate, feed_dict = {X:mini_batch_x, Y:mini_batch_y, keep_prob: 0.5})
            step_cost = sess.run(cost, feed_dict = {X:mini_batch_x, Y:mini_batch_y, keep_prob: 0.5})
            print(
            	'[step]{:3d}'.format(step)
            ,	'[cost] {:6.5f}'.format(step_cost)
            ,	'[accuracy_rate] {:6.6}'.format(step_accuracy_rate)
            	)
    num_train = mnist.train.num_examples
    print('num_train : ', num_train)
    num_test = mnist.test.num_examples
    print('num_test : ', num_test)
    # train_feature = mnist.train.images[:55000]
    # train_label = mnist.train.labels[:55000]
    # print("Accuracy Rate of Train Data(55000) : ", sess.run(accuracyRate, feed_dict={X: train_feature, Y: train_label, keep_prob: 0.5}))
    test_feature = mnist.test.images[:10000]
    test_label = mnist.test.labels[:10000]
    print("Accuracy Rate of Test Data(10000) : ", sess.run(accuracyRate, feed_dict={X: test_feature, Y: test_label, keep_prob: 0.5}))
    pred_digit = sess.run(
    		tf.argmax(Hypothesis, 1)
    	,	feed_dict={X:test_feature, keep_prob: 0.5}
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
