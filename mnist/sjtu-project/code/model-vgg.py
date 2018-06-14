import tensorflow as tf
import numpy as np
import vgg

train_path = 'instruction/mnist_train/'
test_path = 'instruction/mnist_test/'
vgg_path = '/home/marshallee/Documents/pretrained-models/vgg/vgg19/imagenet-vgg-verydeep-19.mat'

def numpyFile_load(path,data_num = 60000, fig_w = 24,mode = 'train'):
	#data = np.fromfile(path+"mnist_"+mode+"_data",dtype=np.uint8).astype(np.float32)
	data = np.load(path+mode+'_data_3.npy').astype(np.float32)
	data = data.reshape(data_num,fig_w,fig_w,1)
	return data

if __name__ == '__main__':
	train_data = numpyFile_load(train_path)
	test_data = numpyFile_load(test_path,mode='test',data_num=10000)

	train_data = np.concatenate((train_data,train_data,train_data),axis=3)
	test_data = np.concatenate((test_data,test_data,test_data),axis=3)

	target_layer = 'relu4_4'
	batch_size = 50
	train_batch_num = 1200
	test_batch_num = 200
	image_shape = [batch_size,24,24,3]

	preprocess_train = np.ndarray([60000,3*3*512], dtype=np.float32)
	preprocess_test = np.ndarray([10000,3*3*512], dtype=np.float32)

	with tf.Session() as sess:
		image = tf.placeholder(tf.float32, shape=image_shape, name='mnist_image')
		net = vgg.net(vgg_path, image)

		print('start process train data...')
		for i in range(train_batch_num):
			feed_dict = {image: train_data[i*batch_size:(i+1)*batch_size]}
			out = net[target_layer].eval(feed_dict=feed_dict)
			preprocess_train[i*batch_size:(i+1)*batch_size] = out.reshape([batch_size,3*3*512])
		np.save('train_vgg_4608d.npy',preprocess_train)
		print('processed train shape: ', preprocess_train.shape)
		print('end process train data...')

		print('start process test data...')
		for i in range(test_batch_num):
			feed_dict = {image: test_data[i*batch_size:(i+1)*batch_size]}
			out = net[target_layer].eval(feed_dict=feed_dict)
			preprocess_test[i*batch_size:(i+1)*batch_size] = out.reshape([batch_size,3*3*512])
		np.save('test_vgg_4608d.npy',preprocess_test)
		print('processed test shape: ', preprocess_test.shape)
		print('end process test data...')