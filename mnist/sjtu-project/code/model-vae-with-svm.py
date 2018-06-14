import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# the path to save the file
# you should modify here to run the program on yout own platform
train_path = '/home/marshallee/Documents/mnist/instruction/mnist_train/'
test_path = '/home/marshallee/Documents/mnist/instruction/mnist_test/'

def numpyFile_load(path,data_num = 60000, fig_w = 24,mode = 'train'):
	#data = np.fromfile(path+"mnist_"+mode+"_data",dtype=np.uint8).astype(np.float32)
	data = np.load(path+mode+'-latent-2d.npy')
	label = np.fromfile(path+"mnist_"+mode+"_label",dtype=np.uint8)
	#data = data.reshape(data_num,fig_w*fig_w)
	return (data,label)

if __name__ == '__main__':
	
	# laod the data
	train_data, train_label = numpyFile_load(train_path)
	test_data,test_label = numpyFile_load(test_path,mode='test',data_num=10000)
	print('train data: ', train_data.shape)
	print('test data: ', test_data.shape)

	# here was originally a list saving lots of value
	# but here we only reserve the best parameters
	Cs = [6.0]
	kernels = ['rbf']

	for C in Cs:
		for kernel in kernels:
			print('C: %f, kernel: %s' % (C,kernel))
			print('start fitting...')
			clf = SVC(kernel = kernel,C=C) # initialize the model
			clf.fit(train_data,train_label)
			print('end fitting...')
			pred = clf.predict(test_data)
			accuracy = np.mean((pred == test_label).astype(np.float32)) # accuracy on the test set
			print('test set accuracy', accuracy)
			pred = clf.predict(train_data)
			accuracy = np.mean((pred == train_label).astype(np.float32)) # accuracy on the train set
			print('train set accuracy', accuracy)
		


	
