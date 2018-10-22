#!/usr/bin/env /c/Apps/Anaconda3/python
"""
[Title] RNN/LSTM - Tensorflow Application - Mnist
[Code] tf-MNIST-lstm-classificaton.py
[Author] 이이백(Yibeck.Lee@gmail.com)
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)
mnist = input_data.read_data_sets("data",one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in' : tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
  , 'out' : tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in' : tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
  , 'out' : tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, bias):
	X = tf.reshape(X, [-1, n_inputs])
	X_in = tf.matmul(X, weights['in']) + biases['in']
	X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	_init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = _init_state, time_major=False)
	results = tf.matmul(final_state[1], weights['out']) + biases['out']
	# outputs = tf.unpack(tf.transpose(outputs, [1,0,2]))
	# results = tf.matmul(outputs[-1], weight['out']) + biases['out']
	return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))


optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
# with tf.Session() as sess:
# 	sess.run(init)
# 	step = 0
# 	while step * batch_size < training_iters:
# 		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# 		batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
# 		sess.run([train_op], feed_dict={
# 		    x:batch_xs
# 		  , y:batch_ys
# 		})
# 		step += 1
# 		if step % 20 == 0:
# 			print(step,sess.run(accuracy, feed_dict={
# 			  x : batch_xs
# 			, y : batch_ys
# 			}))

with tf.Session() as sess:
	sess.run(init)
	for step in range(15):
		avg_cost = 0
		num_tot_batch = int(mnist.train.num_examples/50)
		for mini_step in range(num_tot_batch):
			mini_batch_feature_, mini_batch_label = mnist.train.next_batch(128)
			mini_batch_feature = mini_batch_feature_.reshape([batch_size, n_steps, n_inputs])
			_, step_cost = sess.run(
					[pred, cost]
				,	feed_dict={x:mini_batch_feature, y:mini_batch_label}
				)
			avg_cost += step_cost / num_tot_batch
			if mini_step % 100 == 0:
				predicted_label = tf.nn.softmax(pred)
				correct_prediction = tf.equal(tf.argmax(predicted_label,1),tf.argmax(y,1))
				accuracy_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				# print(
				# 	'mini-batch-step='
				# ,	mini_step + 1
				# ,	step_cost
				# ,	avg_cost
				# ,	"accuracy rate[mini_batch] :", sess.run(accuracy_rate, feed_dict={x:mini_batch_feature, y:mini_batch_label})
				# 	)
		print(
			'step '
		,	step + 1
		,	step_cost
		,	avg_cost
		,	"accuracy rate[mini_batch] :", sess.run(accuracy_rate, feed_dict={x:mini_batch_feature, y:mini_batch_label})
			)
	# predicted_label = tf.nn.softmax(Hypothesis)
	# correct_prediction = tf.equal(tf.argmax(predicted_label,1),tf.argmax(Y,1))
	# accuracy_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# print("accuracy rate[train] :", sess.run(accuracy_rate, feed_dict={X:mnist.train.images, Y:mnist.train.labels}))
	# print("accuracy rate[test] :", sess.run(accuracy_rate, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

	# pred_digit = sess.run(tf.argmax(Hypothesis,1),feed_dict={X:mnist.test.images})
	# actual_digit = sess.run(tf.argmax(mnist.test.labels,1))
	# for i in range(9000,9001):
	# 	print(
	# 		pred_digit[i]
	# 	,	actual_digit[i]
	# 	)

