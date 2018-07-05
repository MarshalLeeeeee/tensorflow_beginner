import numpy as np
import tensorflow as tf
from scipy.fftpack import fft, dct

a = np.arange(10)
b = np.reshape(np.arange(12),(3,4))
a_dct = dct(a)
b_dct = dct(b)
print(a)
print(b)
print(a_dct)
print(b_dct)
A = tf.placeholder(tf.float32, shape=a.shape)
B = tf.placeholder(tf.float32, shape=b.shape)
A_dct = tf.spectral.dct(A)
B_dct = tf.spectral.dct(B)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(A_dct,feed_dict={A:a}))
print(sess.run(B_dct,feed_dict={B:b}))