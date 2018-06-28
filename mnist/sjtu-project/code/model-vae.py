import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# the path to save the file
# you should modify here to run the program on yout own platform
train_path = '/home/marshallee/Documents/mnist/instruction/mnist_train/'
test_path = '/home/marshallee/Documents/mnist/instruction/mnist_test/'

def numpyFile_load(path,data_num = 60000, fig_w = 24,mode = 'train'):
    '''
    return the data and label saved as numpy array, given the data file path
    '''

    #data = np.fromfile(path+"mnist_"+mode+"_data",dtype=np.uint8).astype(np.float32)
    data = np.load(path+mode+'_data_3.npy')
    label = np.fromfile(path+"mnist_"+mode+"_label",dtype=np.uint8)
    data = data.reshape(data_num,fig_w*fig_w)
    data = data / 255.0 # normalize the data to [0,1]
    #label = toHotOne(label).astype(np.float32)
    return (data,label)


def encoder(X,keep_prob,layer0=480,layer1=1024,layer2=128,layer3=20):
    '''
    Use a 2-hidden-layer fcn structure to return mean and Sigma
    X is the input data
    '''
    with tf.variable_scope("encoder"):
        W1 = tf.Variable(tf.truncated_normal([layer0, layer1], stddev=0.1),name='W1')
        b1 = tf.Variable(tf.constant(0.1, shape=[layer1]),name='b1')
        W2 = tf.Variable(tf.truncated_normal([layer1,layer2], stddev=0.1),name='W2')
        b2 = tf.Variable(tf.constant(0.1, shape=[layer2]),name='b2')
        W3 = tf.Variable(tf.truncated_normal([layer2, layer3*2], stddev=0.1),name='W3')
        b3 = tf.Variable(tf.constant(0.1, shape=[layer3*2]),name='b3')

        z1 = tf.add(tf.matmul(X,W1),b1,name='z1')
        h1 = tf.nn.dropout(tf.nn.relu(z1),keep_prob,name='h1')
        z2 = tf.add(tf.matmul(h1,W2),b2,name='z2')
        h2 = tf.nn.dropout(tf.nn.tanh(z2),keep_prob,name='h2')
        z3 = tf.add(tf.matmul(h2,W3),b3,name='z3')

        # the output of 'mean' and 'Sigma
        # with reference to https://github.com/altosaar/variational-autoencoder/blob/master/vae.py
        mean = z3[:,:layer3]
        Sigma = tf.sqrt(tf.exp(z3[:, layer3:]))

    return mean, Sigma

def decoder(z,keep_prob,layer0=20,layer1=128,layer2=1024,layer3=480):
    '''
    Use a 2-hidden-layer fcn structure to return the reconstrcuted data
    z is the latent code
    '''

    with tf.variable_scope("decoder"):
        W1 = tf.Variable(tf.truncated_normal([layer0, layer1], stddev=0.1),name='W1')
        b1 = tf.Variable(tf.constant(0.1, shape=[layer1]),name='b1')
        W2 = tf.Variable(tf.truncated_normal([layer1,layer2], stddev=0.1),name='W2')
        b2 = tf.Variable(tf.constant(0.1, shape=[layer2]),name='b2')
        W3 = tf.Variable(tf.truncated_normal([layer2, layer3], stddev=0.1),name='W3')
        b3 = tf.Variable(tf.constant(0.1, shape=[layer3]),name='b3')

        z1 = tf.add(tf.matmul(z,W1),b1,name='z1')
        h1 = tf.nn.dropout(tf.nn.tanh(z1),keep_prob,name='h1')
        z2 = tf.add(tf.matmul(h1,W2),b2,name='z2')
        h2 = tf.nn.dropout(tf.nn.relu(z2),keep_prob,name='h2')
        z3 = tf.add(tf.matmul(h2,W3),b3,name='z3')
        y = tf.sigmoid(z3,name='reconstruction')

    return y

def vae(X,X_target,keep_prob,layer1=1024,layer2=128,layer3=20,d=480):

    # get miu and Sigma
    miu, Sigma = encoder(X,keep_prob,d,layer1,layer2,layer3)

    # generate the noise following the Gaussian distribution
    # tf.random_normal is a constant, which is not trainable
    z = miu + Sigma * tf.random_normal(tf.shape(miu), 0, 1, dtype=tf.float32)

    # get the reconstructed data through decode
    # restrict the lower and upper boudn to avoid computational error
    y = decoder(z,keep_prob,layer3,layer2,layer1,d)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    # marginal_likelihood evaluate the loss between the recontructed data and the original data
    # KL_divergence evaluate the difference between the distrubution with N(miu,Sigma), can also be deemed as the regularization term
    marginal_likelihood = tf.reduce_sum(X_target * tf.log(y) + (1 - X_target) * tf.log(1 - y), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(miu) + tf.square(Sigma) - tf.log(1e-8 + tf.square(Sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    # the overal loss function
    loss = - marginal_likelihood + KL_divergence

    return y, z, loss, -marginal_likelihood, KL_divergence

if __name__ == '__main__':

    # load data
    train_data, train_label = numpyFile_load(train_path)
    test_data,test_label = numpyFile_load(test_path,mode='test',data_num=10000)

    # determine some constants
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]
    d = train_data.shape[1]
    d_z = 2
    n_hidden = 500
    epochs = 100
    batch = 100
    prob = 0.9
    learning_rate = 1e-3
    iterates = int(train_num / batch)

    with tf.Session() as sess:

        X = tf.placeholder(tf.float32,shape=[None,d],name='input-image')
        X_target = tf.placeholder(tf.float32,shape=[None,d],name='target-image')
        keep_prob = tf.placeholder("float")

        y, z, loss, neg_marginal_likelihood, KL_divergence = vae(X, X_target, keep_prob,d=d,layer3=d_z)
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # starts training...
        for epoch in range(epochs):
            for i in range(iterates):
                train_batch = train_data[i*batch:(i+1)*batch,:]
                feed_dict = {X:train_batch,X_target:train_batch,keep_prob:prob}
                _, tot_loss, loss_likelihood, loss_divergence = sess.run((train, loss, neg_marginal_likelihood, KL_divergence),feed_dict=feed_dict)
            print('epoch: %d, total loss: %f, reconstruction error: %f, KL: %f' % (epoch,tot_loss,loss_likelihood,loss_divergence))


        train_z = sess.run(z,feed_dict = {X:train_data,X_target:train_data,keep_prob:1.0})
        test_z = sess.run(z,feed_dict = {X:test_data,X_target:test_data,keep_prob:1.0})

        # to see if the reconstruction is similar enough to the original data 
        train_data_0 = train_data[0].reshape((24,24))
        print('train_data shape: ', train_data_0.shape)
        print('train_data shape: ', train_data.shape)
        train_data_reconstruction = sess.run(y,feed_dict={X:train_data[0:1,:],X_target:train_data[0:1,:],keep_prob:1.0})
        print('train_data_reconstruction shape: ', train_data_reconstruction.shape)
        train_data_reconstruction = train_data_reconstruction.reshape((24,24))
        print('train_data_reconstruction shape: ', train_data_reconstruction.shape)
        plt.axis('off')
        plt.subplot(2,2,1)
        plt.imshow(train_data_0, cmap='Greys_r')
        plt.subplot(2,2,2)
        plt.imshow(train_data_reconstruction, cmap='Greys_r')
        plt.show()
        
        # save the latent space to npy file
        np.save('train-latent.npy',train_z)
        np.save('test-latnet.npy',test_z)
        print('shape of train data mapping to latent space: ', train_z.shape)
        print('shape of test data mapping to latent space: ', test_z.shape)