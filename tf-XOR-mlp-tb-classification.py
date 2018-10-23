#!/usr/bin/env /c/Apps/Anaconda3/python
#_*_coding:utf-8_*_
"""
[Title] XOR Multi Layer Perceptron - Tensorflow/Tensorboard Application
[Code] tf-XOR-mlp-classificaton.py
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
W1 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0), name="Weight1")
# W1 = tf.Variable(tf.random_normal([2,2]), name="Weight1")
# B1 = tf.Variable(tf.random_uniform([1],-1.0,1.0), name="Bias1")
B1 = tf.Variable(tf.zeros([2]), name="Bias1")
H1WpB = tf.nn.sigmoid(tf.matmul(X,W1) + B1)
W2 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0), name="Weight2")
B2 = tf.Variable(tf.zeros([1]), name="Bias2")
Hypothesis = tf.sigmoid(tf.matmul(H1WpB,W2)+B2)
cost = tf.reduce_mean(
	-Y*tf.log(Hypothesis) - (1-Y)*tf.log(1-Hypothesis)
	)
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.round(Hypothesis),label)
accuracy_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
print(init)
with tf.Session() as sess:
	
	tf.summary.histogram("W1", W1)
	tf.summary.histogram("B1", B1)
	tf.summary.scalar("Cost", cost)
	tf.summary.scalar("Accuracy Rate", accuracy_rate)
	summary = tf.summary.merge_all()
	tb_log = tf.summary.FileWriter(r"./tb-XOR-mlp", sess.graph)
	sess.run(init)
	for epoch in range(10000):
		_, epoch_cost = sess.run(
				[optimizer, cost]
			,	feed_dict={X:feature, Y:label}
			)
		if epoch % 100 == 0:
			# print("[Epoch-Cost] ", epoch+1, epoch_cost)
			# Weight1 = W1.eval(sess)
			# Bias1 = B1.eval(sess)
			print(epoch+1,"epoch-cost=", epoch_cost, "accuracy rate=", sess.run(accuracy_rate, feed_dict={X:feature, Y:label}))
			summ = sess.run(summary, feed_dict={X:feature, Y:label})
			tb_log.add_summary(summ, epoch)			
	print(epoch+1,"epoch-cost=", epoch_cost, "accuracy rate=", sess.run(accuracy_rate, feed_dict={X:feature, Y:label}))
	pred = sess.run(Hypothesis, feed_dict={X:[[0,0],[0,1],[1,0],[1,1]]})
	for i in range(4):
		print(feature[i],label[i],pred[i])	
	# tb_log.close()
