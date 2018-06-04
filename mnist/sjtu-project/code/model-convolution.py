import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image

# the path to save the file
# you should modify here to run the program on yout own platform
train_path = 'instruction/mnist_train/'
test_path = 'instruction/mnist_test/'


def toHotOne(label):
	# given the input as the output vector , ranging from 0 to 9
	# return the matrix containing Hot one vactor
	return (np.arange(10)==label[:,None]).astype(np.int32)

def PCA(data):
	# return the data in the new space
	n = data.shape[0]
	d = data.shape[1]
	Sigma = np.dot(data.T,data)
	eigValue, eigMatrix = np.linalg.eig(Sigma)
	return np.dot(data,eigMatrix[:][:255].T)

def numpyFile_load(path,data_num = 60000, fig_w = 24,mode = 'train'):
	#data = np.fromfile(path+"mnist_"+mode+"_data",dtype=np.uint8).astype(np.float32)
	data = np.load(path+mode+'_data_3.npy').astype(np.float32)
	label = np.fromfile(path+"mnist_"+mode+"_label",dtype=np.uint8)
	data = data.reshape(data_num,fig_w,fig_w,1)
	label = toHotOne(label).astype(np.float32)
	return (data,label)

if __name__ == '__main__':

	# load data
	train_data, train_label = numpyFile_load(train_path)
	test_data,test_label = numpyFile_load(test_path,mode='test',data_num=10000)

	# initialize some constants
	train_num = train_data.shape[0]
	test_num = test_data.shape[0]
	fig_w = train_data.shape[1]
	learning_rate = 1e-3
	batch_size = 50
	epochs = 20
	batch_num = int(train_num/batch_size)
	prob = 0.9

	with tf.Session() as sess:
		X  = tf.placeholder(tf.float32,shape=[None,fig_w,fig_w,1])
		y_ = tf.placeholder(tf.float32,shape=[None,10])
		keep_prob = tf.placeholder("float")

		# convolution layer 1
		W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1),name='W_conv1')
		b_conv1 = tf.Variable(tf.random_normal([32]),name='b_conv1')
		z_conv1 = tf.add(tf.nn.conv2d(X, W_conv1, strides = [1,1,1,1], padding = 'SAME'),b_conv1,name='z_conv1')
		h_conv1 = tf.nn.sigmoid(tf.nn.max_pool(z_conv1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),name='h_conv1')

		# convolution layer 2
		W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1),name='W_conv2')
		b_conv2 = tf.Variable(tf.random_normal([64]),name='b_conv2')
		z_conv2 = tf.add(tf.nn.conv2d(h_conv1, W_conv2, strides = [1,1,1,1], padding = 'SAME'),b_conv2,name='z_conv2')
		h_conv2 = tf.nn.sigmoid(tf.nn.max_pool(z_conv2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),name='h_conv2')

		# reshape the data to be the input of fcn
		H = tf.reshape(h_conv2,[-1,6*6*64],name='fully-connected-input')

		# fcn layer 1
		W_fc1 = tf.Variable(tf.truncated_normal([6*6*64,1024], stddev = 0.1),name='W_fc1')
		b_fc1 = tf.Variable(tf.random_normal([1024]),name='b_fc1')
		z_fc1 = tf.add(tf.matmul(H,W_fc1),b_fc1,name='z_fc1')
		h_fc1 = tf.nn.dropout(tf.nn.sigmoid(z_fc1),keep_prob,name='h_fc1')

		# fcn layer 2
		W_fc2 = tf.Variable(tf.truncated_normal([1024,256], stddev = 0.1),name='W_fc2')
		b_fc2 = tf.Variable(tf.random_normal([256]),name='b_fc2')
		z_fc2 = tf.add(tf.matmul(h_fc1,W_fc2),b_fc2,name='z_fc2')
		h_fc2 = tf.nn.dropout(tf.nn.sigmoid(z_fc2),keep_prob,name='h_fc2')

		# fcn layer 3
		W_fc3 = tf.Variable(tf.truncated_normal([256,10], stddev = 0.1),name='W_fc3')
		b_fc3 = tf.Variable(tf.random_normal([10]),name='b_fc3')
		z_fc3 = tf.add(tf.matmul(h_fc2,W_fc3),b_fc3,name='z_fc3')
		h_fc3 = tf.nn.sigmoid(z_fc3,name='h_fc3')
		y = tf.nn.softmax(h_fc3,name='pred_y')

		# loss function
		cross_entropy = -tf.reduce_sum(y_*tf.log(y))
		train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		sess.run(tf.global_variables_initializer())

		# start training...
		print('start training...')
		for epoch in range(epochs):
			for num in range(batch_num):
				train_data_batch = train_data[num*batch_size:(num+1)*batch_size]
				train_label_batch = train_label[num*batch_size:(num+1)*batch_size]
				feed_dict = {X:train_data_batch,y_:train_label_batch,keep_prob:prob}
				sess.run(train,feed_dict=feed_dict)
			print('epoch: %d, loss: %f' % (epoch,sess.run(cross_entropy,feed_dict={X:train_data_batch,y_:train_label_batch,keep_prob:1.0})))

		# calculate anf output the accuracy
		train_accu = 0.0
		test_accu = 0.0
		for num in range(batch_num):
			train_accu += sess.run(accuracy,feed_dict={X:train_data[num*batch_size:(num+1)*batch_size],y_:train_label[num*batch_size:(num+1)*batch_size],keep_prob:1.0})
		print('accuracy on train set: ',train_accu/batch_num)

		for num in range(200):
			test_accu += sess.run(accuracy,feed_dict={X:test_data[num*batch_size:(num+1)*batch_size],y_:test_label[num*batch_size:(num+1)*batch_size],keep_prob:1.0})
		print('accuracy on test set: ',test_accu/200)
