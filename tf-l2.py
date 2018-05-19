import tensorflow as tf
import numpy as np

s = np.array([[1,2,3,4,5],[12,13,14,15,16]])
print(np.power(np.std(s.T,0),2))

with tf.Session() as sess:
	a = tf.constant(3,shape=[1,10],dtype='float32')
	b = tf.constant(0,shape=[1,10],dtype='float32')
	x = tf.transpose(tf.constant([[1,2,3,4,5],[12,13,14,15,16]],dtype='float32'))
	mean = tf.reduce_mean(x,0)
	var = tf.reduce_mean(tf.pow(tf.subtract(x,mean),2),0)
	c = tf.nn.l2_loss(a-b)
	sess.run(tf.global_variables_initializer())
	print(x.shape)
	print(sess.run(mean))
	print(sess.run(var))