import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# load data
mnist = input_data.read_data_sets("./", one_hot=True)

# 'x' and 'y_' are placeholders where the value of them are not sure 
# until we assign some value to it
# placeholder is always used with feed_dict when using sess.run()
x  = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

# tf.Variable() defines the node with initialized value but can also be changed with time varying
# initialize the value of the parameter as constant 0
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
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# be ready for the initialization of the parameter
init = tf.initialize_all_variables()

# start a session which is to control the execution of the training model
sess = tf.Session()
'''
# if we use sess = tf.InteractiveSession() then we can construct the graph at any time
# however if we use sess = tf.Seesion() then we must call this after we construct the whole graph
# furthermore we cannot alter the graph as long as we constrcut the graph
'''

# execute the initialization
sess.run(init)

# execute the training
for i in range(1000):
	# make the batch randomly
	batch_xs, batch_ys = mnist.train.next_batch(100)

	# execute one step of training
	sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})


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
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
