import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image

train_path = 'instruction/mnist_train/'
test_path = 'instruction/mnist_test/'

def tensorflow_load():
	return input_data.read_data_sets("data/", one_hot=True)

def toHotOne(label):
	return (np.arange(10)==label[:,None]).astype(np.int32)

def numpyFile_load(path,data_num = 60000, fig_w = 45,mode = 'train'):
	data = np.fromfile(path+"mnist_"+mode+"_data",dtype=np.uint8).astype(np.float32)
	label = np.fromfile(path+"mnist_"+mode+"_label",dtype=np.uint8)
	data = data.reshape(data_num,fig_w*fig_w)
	label = toHotOne(label).astype(np.float32)
	return (data,label)

def batch(data, label, num, data_num = 60000):
	arr = np.arange(data_num)
	np.random.shuffle(arr)
	data_batch = []
	label_batch = []
	for i in arr:
		data_batch.append(data[i])
		label_batch.append(label[i])
	return (data_batch,label_batch)


if __name__ == '__main__':
	'''
	aaa = np.arange(10)
	np.random.seed(1)
	np.random.shuffle(aaa)
	bbb = np.arange(10)
	np.random.seed(1)
	np.random.shuffle(bbb)
	print(aaa)
	print(bbb)
	exit(0)
	'''

	# 'x' and 'y_' are placeholders where the value of them are not sure 
	# until we assign some value to it
	# placeholder is always used with feed_dict when using sess.run()
	x  = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	# tf.Variable() defines the node with initialized value but can also be changed with time varying
	# initialize the value of the parameter as constant 0
	'''W = tf.Variable(tf.random_normal([784, 10],stddev=1))
				b = tf.Variable(tf.random_normal([10],stddev=1))'''
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	'''
	# actually assigning it with some uniformed variables is a better choice
	w = 
	b = 
	'''

	# the output of this layer
	# using softmax as the activation function of this layer
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	# decide the loss function
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

	# decide the optimization algorithm and the learning rate
	# the target of the training is to minimize the loss function
	train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

	# be ready for the initialization of the parameter
	init = tf.initialize_all_variables()

	with tf.Session() as sess: 
		'''
		# if we use sess = tf.InteractiveSession() then we can construct the graph at any time
		# however if we use sess = tf.Seesion() then we must call this after we construct the whole graph
		# furthermore we cannot alter the graph as long as we constrcut the graph
		'''

		# execute the initialization
		sess.run(init)

		
		######### TENSORFLOW LOAD ########
		mnist = tensorflow_load()
		print(mnist.train.images.shape)
		print(mnist.test.images.shape)
		# execute the training
		for i in range(3000):
			# make the batch randomly
			batch_xs, batch_ys = mnist.train.next_batch(100)

			# execute one step of training
			sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
		print(sess.run(cross_entropy,feed_dict = {x: batch_xs, y_: batch_ys}))

		### the following is running the model over the test dataset ###
		# tf.argmax() returns the index of the maximal value in the vector
		# as the model select the maximal output probablity as the answer, tf.argmax() means the output
		# tf.equal() returns a vector whose item is type-boolean corresponding to each position
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

		# calculate the mean of the above vector, which is the accuracy of the model
		# tf.cast() is function to change the type of the value explicitly
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		# run the model again
		# notice that this time the parameter will not be changed, because the ending node will not pass the optimizer node
		print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
		

		'''
		########## NUMPY LOAD ###########
		train_data,train_label = numpyFile_load(train_path,mode='train')
		test_data,test_label = numpyFile_load(test_path,mode='test',data_num=10000)
		for i in range(1000):
			if(i % 50 == 0):
				np.random.seed(1)
				np.random.shuffle(train_data)
				np.random.seed(1)
				np.random.shuffle(train_label)
			start = 1200*(i%50)
			end = 1200 *(i%50+1)
			print('cnt start end',i,start,end)
			sess.run(train_step,feed_dict={x:train_data[start:end],y_:train_label[start:end]})
			#print('cnt: ',i,sess.run(y,feed_dict={x:train_data[:100],y_:train_label[:100]}))
			#print('----------')
			#print('cnt: ',i,sess.run(y_,feed_dict={x:train_data[:100],y_:train_label[:100]}))
			#print('------------')
			#print('cnt: ',i,sess.run(tf.matmul(x,W), feed_dict={x:train_data[:100],y_:train_label[:100]}))
			print('cnt: ',i,sess.run(cross_entropy,feed_dict={x:train_data[:100],y_:train_label[:100]}))
		print('train over,,,')
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))
		'''		
