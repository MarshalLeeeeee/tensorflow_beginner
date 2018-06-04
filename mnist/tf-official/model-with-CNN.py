import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
	# tf.truncated_normal() is to make some noise around 0(?) following normalized distribution

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape = shape))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
	# tf.nn.conv2d(input, filter, strides, padding, ...)
	# the target is the picture normally which is type-tensor, whose dimension is 4, [batch, in_height, in_width, in_channels]
	# the filter is the kernel which is type-tensor, whose dimension is 4, [filter_height, filter_width, in_channels, out_channels]
	# the strides is a 4-dimensional tensor, the first and last element are usually 1
	# the padding is either 'VALID' or 'SAME'. 
	### If 'VALID', then the convolution start from the leftmost and may ignore some rightmost
	### If 'SAME' , then the output will be exactly the upper-bound of w/s, with the equal padding at both sides

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  # tf.nn.max_pool(value, ksize, strides, padding, name=None)
  # the value is the input pis which is a 4-dimensional tensor
  # the ksize is the pooling kernel which is a 4-dimensional tensor, the first and the last element are usually 1
  # the stride is a 4-dimensional tensor, the first and the last element are usually 1
  # the padding is either 'VALID' or 'SAME'

if __name__ == '__main__':
	# load data
	mnist = input_data.read_data_sets("./", one_hot=True)
	x  = tf.placeholder("float", [None, 784])
	y_ = tf.placeholder("float", [None, 10])
	sess = tf.InteractiveSession()

	# reshape the data in to the 4-dimensional tensor
	x_image = tf.reshape(x, [-1,28,28,1])

	# the start kernel and bias of the first convolution level
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	# put them into the first convolution level
	# use reLu as activation
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# the start kernel and bias of the second convolution level
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	# put them into the second convolution level
	# use reLu as activation
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# the weight of the fully-connected level
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	# reshape it into a vector as the input of the fully-connected level
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# set the dropout
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# softmax level
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	sess.run(tf.initialize_all_variables())

	for i in range(20000):
	  batch = mnist.train.next_batch(50)
	  if i%100 == 0:
	  	# when running we should not use dropout, so we set the prob to 1
	    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
	    print("step %d, training accuracy %g"%(i, train_accuracy))
	  # equal to sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))