import tensorflow as tf

with tf.Session() as sess:
	'''
	a = tf.constant([[[[1],[3],[2],[4]],[[5],[6],[8],[7]],[[15],[11],[13],[12]],[[18],[14],[17],[16]]]], dtype='float32')
	#a = tf.expand_dims(a,0)
	#a_input = tf.expand_dims(a,-1)
	b = tf.nn.conv2d(a, tf.constant([2,2,1,1],dtype='float32'), strides=[1, 1, 1, 1], padding = 'VALID')
	'''


	input = tf.Variable(tf.ones([1,4,4,1]))  
	filter = tf.Variable(tf.ones([2,2,1,1]))  
	  
	op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  

	sess.run(tf.global_variables_initializer())
	#print(sess.run(a))
	#print(sess.run(b))
	print(sess.run(op))