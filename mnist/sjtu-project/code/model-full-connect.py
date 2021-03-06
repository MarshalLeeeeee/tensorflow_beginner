import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# the path to save the file
# you should modify here to run the program on yout own platform
train_path = 'instruction/mnist_train/'
test_path = 'instruction/mnist_test/'


def toHotOne(label):
	# given the input as the output vector , ranging from 0 to 9
	# return the matrix containing Hot one vactor
	return (np.arange(10)==label[:,None]).astype(np.int32)

def norm(data):
	# normalize the data
	mean = np.mean(data,axis=0)
	stddev = np.std(data,axis=0)
	# if the stddev is 0, make it 1
	stddev = stddev + (stddev == 0).astype(np.float32)
	return (data-mean)/stddev,mean,stddev

def norm2(data,mean,stddev):
	# normalize with known mean and sigma
	return (data-mean)/stddev

def PCA(data):
	# return the data in the new space
	n = data.shape[0]
	d = data.shape[1]
	Sigma = np.dot(data.T,data)
	eigValue, eigMatrix = np.linalg.eig(Sigma)
	data_new = np.dot(data,eigMatrix)
	return (data_new,eigValue,eigMatrix)

def numpyFile_load(path,data_num = 60000, fig_w = 45,mode = 'train'):
	#data = np.fromfile(path+"mnist_"+mode+"_data",dtype=np.uint8).astype(np.float32)
	data = np.load(path+mode+'_data_3.npy')
	label = np.fromfile(path+"mnist_"+mode+"_label",dtype=np.uint8)
	#data = data.reshape(data_num,fig_w,fig_w)
	label = toHotOne(label).astype(np.float32)
	return (data,label)

if __name__ == '__main__':

	# PCA demo
	# to see the if the reconstructed image is similar enough to the original data
	# so as to exam if the PCA is done correctly and to see if the dimension determined is enough
	# the following script is implemented on the original data whise size is 45*45

	'''
	train_data,train_label = numpyFile_load(train_path)
	plt.imshow(np.reshape(train_data[0],[45,45]))
	plt.axis('off')
	plt.show()
	train_data, mean, stddev = norm(train_data)
	train_data,eigV,eigM = PCA(train_data)
	eigM_inv = np.linalg.inv(eigM)
	print(train_data.shape)
	print(train_data[0].shape)
	train_data0_origin = np.reshape(np.dot(train_data[0],eigM_inv)*stddev+mean,[45,45])
	plt.imshow(train_data0_origin)
	plt.axis('off')
	plt.show()
	train_data_cut = train_data[:,:255]
	print(train_data[0])
	print(train_data_cut[0])
	test_data, test_label = numpyFile_load(test_path,mode='test',data_num=10000)
	plt.imshow(np.reshape(test_data[33],[45,45]))
	plt.axis('off')
	plt.show()
	test_data = norm2(test_data,mean,stddev)
	test_data_cut = np.dot(test_data,eigM)[:,:255]
	print(train_data_cut.shape)
	print(train_data_cut[0].shape)
	#print(np.pad(train_data_cut[0],(0,2025-255),'constant'))
	train_data0 = np.pad(train_data_cut[0],(0,2025-255),'constant')
	print(train_data0.shape)
	train_data0_back = np.reshape(np.dot(train_data0,eigM_inv)*stddev+mean,[45,45])
	print(train_data0_back.shape)
	plt.imshow(train_data0_back)
	plt.axis('off')
	plt.show()

	test_data0 = np.pad(test_data_cut[33],(0,2025-255),'constant')
	print(test_data0.shape)
	test_data0_back = np.reshape(np.dot(test_data0,eigM_inv)*stddev+mean,[45,45])
	print(test_data0_back.shape)
	plt.imshow(test_data0_back)
	plt.axis('off')
	plt.show()
	'''
	
	# load the data
	train_data,train_label = numpyFile_load(train_path)
	test_data, test_label = numpyFile_load(test_path,mode='test',data_num=10000)
	print('train_data shape: ', train_data.shape)
	print('test_data shape: ', test_data.shape)
	print('train_label shape: ', train_label.shape)
	print('test_label shape: ', test_label.shape)

	# do the PCA
	train_data, mean, stddev = norm(train_data)
	train_data,eigV,eigM = PCA(train_data)
	eigM_inv = np.linalg.inv(eigM)
	test_data = np.dot(norm2(test_data,mean,stddev),eigM)

	# determine the dimension we want to reserve in PCA
	# still we output to see how many information is reserves given the determined dimension
	cut = 256
	train_data_cut = train_data[:,:cut]
	test_data_cut = test_data[:,:cut]
	print(np.sum(eigV[:256])/np.sum(eigV))

	# initialize some constants
	layer1 = 1024
	layer2 = 128
	layer3 = 10
	fig_w = 45
	learning_rate = 3.1e-1
	learning_rate_decay = 0.9999
	regularize_rate = 0.0001
	train_num = train_data_cut.shape[0]
	batch_size = 100
	training_step = 120000
	prob = 0.9

	with tf.Session() as sess:
		X  = tf.placeholder(tf.float32,shape=[None,cut],name='figure-input')
		y_ = tf.placeholder(tf.float32,shape=[None,10],name='true-label')
		keep_prob = tf.placeholder("float")
		
		# define the variables of the three layers
		W1 = tf.Variable(tf.truncated_normal([cut, layer1], stddev=0.1),name='W1')
		b1 = tf.Variable(tf.constant(0.1, shape=[layer1]),name='b1')
		W2 = tf.Variable(tf.truncated_normal([layer1,layer2], stddev=0.1),name='W2')
		b2 = tf.Variable(tf.constant(0.1, shape=[layer2]),name='b2')
		W3 = tf.Variable(tf.truncated_normal([layer2, layer3], stddev=0.1),name='W3')
		b3 = tf.Variable(tf.constant(0.1, shape=[layer3]),name='b3')

		# define the ops
		z1 = tf.add(tf.matmul(X,W1),b1,name='z1')
		h1 = tf.nn.dropout(tf.nn.relu(z1),keep_prob,name='h1')
		z2 = tf.add(tf.matmul(h1,W2),b2,name='z2')
		h2 = tf.nn.dropout(tf.nn.relu(z2),keep_prob,name='h2')
		z3 = tf.add(tf.matmul(h2,W3),b3,name='z3')
		h3 = tf.nn.relu(z3,name='h3')
		y  = tf.nn.softmax(h3,name='y')

		# regularization
		regularizer = tf.contrib.layers.l2_regularizer(regularize_rate)
		regularization = regularizer(W1) + regularizer(W2) + regularizer(W3)

		# the loss function
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		loss = cross_entropy_mean + regularization
		lr = tf.train.exponential_decay(learning_rate, global_step, train_num / batch_size,learning_rate_decay)

		# train step
		global_step = tf.Variable(0, trainable=False)
		train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

		sess.run(tf.global_variables_initializer())

		# start train...
		for n in range(training_step):
			start = int(batch_size*(n%(train_num/batch_size)))
			end = int(batch_size*(1+n%(train_num/batch_size)))
			sess.run(train_step,feed_dict={X:train_data_cut[start:end],y_:train_label[start:end],keep_prob:prob})
			if(n % 6000 == 5999):
				print('epoch: lr: ',str(int(n/600)),sess.run([loss,lr],feed_dict={X:train_data_cut[start:end],y_:train_label[start:end],keep_prob:1.0}))

		# output the accuracy
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(sess.run(accuracy, feed_dict={X: test_data_cut, y_: test_label,keep_prob:1.0}))
		print(sess.run(accuracy, feed_dict={X: train_data_cut, y_: train_label,keep_prob:1.0}))
