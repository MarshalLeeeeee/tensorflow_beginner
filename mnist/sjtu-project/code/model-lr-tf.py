import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import time

"""def next_batch(X,Y,batch_size):
	perm = np.arange(len(X))
	np.random.shuffle(perm)
	_X = X[perm[:batch_size]].tolist()
	_Y = Y[perm[:batch_size]].tolist()
	return _X,_Y"""



def next_batch(X,Y,batch_size,idx):
    #num = len(X)
    #perm = np.arange(num)
    #np.random.shuffle(perm)
    #X = X[perm]
    #Y = Y[perm]
    X_next=X[batch_size*idx:batch_size*idx+batch_size]
    Y_next=Y[batch_size*idx:batch_size*idx+batch_size]

    _X=[]
    _Y=[]
    for i in range(batch_size):
        _X.append(X_next[i])
        _Y.append(Y_next[i])
    return _X,_Y

def pca(X_full):
	pca256c=PCA(n_components=150)
	data_pca256c = pca256c.fit_transform(X_full)
	print(data_pca256c.shape)
	print(pca256c.explained_variance_ratio_.sum())
	return data_pca256c

def one_hot(Y):
	return (np.arange(10)==Y[:,None]).astype(np.int32)

train_num=60000
test_num=10000
img_size=480

X_train = np.load('train_data_2.npy').reshape([-1,img_size])
Y_train = np.fromfile("./mnist_train/mnist_train_label",dtype=np.uint8)
Y_train=one_hot(Y_train)

perm=np.arange(len(Y_train))
np.random.shuffle(perm)
X_train=X_train[perm][:train_num]
Y_train=Y_train[perm][:train_num]

X_test=np.load('test_data_2.npy').reshape([-1,img_size])
Y_test = np.fromfile("./mnist_test/mnist_test_label",dtype=np.uint8)
Y_test=one_hot(Y_test)

perm=np.arange(len(Y_test))
np.random.shuffle(perm)
X_test=X_test[perm][:test_num]
Y_test=Y_test[perm][:test_num]


learning_rate = 0.001
learning_rate_decay=0.999
batch_size = 200
n_epochs = 200

X = tf.placeholder(tf.float32,[batch_size,img_size],name='X')
Y = tf.placeholder(tf.int64,[batch_size,10],name='Y')

w = tf.Variable(tf.zeros([img_size,10]),name='weight')
b = tf.Variable(tf.zeros([1,10]),name='bias')

logits = tf.matmul(X,w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
loss = tf.reduce_mean(entropy)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learning_rate, global_step, train_num / batch_size,learning_rate_decay)

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(train_num/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        total_correct_preds = 0      
        for j in range(n_batches):           
            X_batch, Y_batch = next_batch(X_train,Y_train,batch_size,j)   
            _, loss_batch = sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss += loss_batch
            #accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch})
            #for ab in accuracy_batch:
            #    total_correct_preds += ab

        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        #print('Accuracy {0}'.format(total_correct_preds/train_num))       

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')

    n_batches = int(test_num/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = next_batch(X_test,Y_test,batch_size,i) 
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch})
        for ab in accuracy_batch:
            total_correct_preds += ab

    print('Accuracy {0}'.format(total_correct_preds/test_num))