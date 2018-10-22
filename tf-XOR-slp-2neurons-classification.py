#!/usr/bin/env /c/Apps/Anaconda3/python
"""
[Title] CNN - Tensorflow Application - Mnist
[Code] tf-MNIST-cnn-classificaton.py
[Author] 이이백(Yibeck.Lee@gmail.com)
"""
import tensorflow as tf 
import math
feature = [
	[0,0]
,	[0,1]
,	[1,0]
,	[1,1]
]
label = [
	[0]
,	[1]
,	[1]
,	[0]
]
X = tf.placeholder(tf.float32, shape=[4, 2], name="X-feature")
Y = tf.placeholder(tf.float32, shape=[4, 1], name="Y-label")
# W1 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0), name="Weight1")
W1 = tf.Variable(tf.random_normal([2,1]), name="Weight1")
# B1 = tf.Variable(tf.random_uniform([1],-1.0,1.0), name="Bias1")
B1 = tf.Variable(tf.random_normal([1]), name="Bias1")

Hypothesis = tf.sigmoid(tf.matmul(X,W1)+B1)
print(Hypothesis)
learning_rate = 0.001
cost = tf.reduce_mean(
	-Y*tf.log(Hypothesis) - (1-Y)*tf.log(1-Hypothesis)
	)
print(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print(optimizer)
init = tf.global_variables_initializer()
print(init)
with tf.Session() as sess:
	sess.run(init)
	avg_cost = 0
	for epoch in range(15):
		_, epoch_cost = sess.run(
				[optimizer, cost]
			,	feed_dict={X:feature, Y:label}
			)
		avg_cost += epoch_cost / 4
		if epoch % 1 == 0:
			print("[Epoch-Cost] ", epoch+1, epoch_cost, avg_cost)
			Weight1 = W1.eval(sess)
			Bias1 = B1.eval(sess)
		Pred = sess.run(Hypothesis,feed_dict={X:feature})
		print(Pred)
		for i in range(4):
			print(
				epoch+1
			,	Weight1[0][0]
			,	Weight1[1][0]
			,	Bias1[0]
			,	feature[i][0]
			,	feature[i][1]
			,	1/(1+math.exp(-(Weight1[0][0]*feature[i][0] + Weight1[1][0]*feature[i][1] + Bias1[0])))
			,	round(1/(1+math.exp(-(Weight1[0][0]*feature[i][0] + Weight1[1][0]*feature[i][1] + Bias1[0]))))
			,	label[i]
				)
		correct_prediction = tf.equal(tf.round(Hypothesis),label)
		accuracy_rate = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		print('accuracy rate : ', sess.run(accuracy_rate, feed_dict={X:feature, Y:label}))

	print('[0,0] ',sess.run(Hypothesis,feed_dict={X:[[0,0],[0,1],[1,0],[1,1]]}))

