import tensorflow as tf
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = 'rain_princess.jpg'

'''img = mpimg.imread(filename)
print(img.shape)
plt.imshow(img)
plt.axis('off')
plt.show()'''

img_man = cv2.imread(filename,0)
print(img_man.shape)
plt.imshow(img_man, cmap='Greys_r')
plt.axis('off')
plt.show()

w = 800
rows = int(img_man.shape[0])
cols = int(img_man.shape[1])
mask = np.ones(img_man.shape,np.uint8)
mask[int(rows/2)-30:int(rows/2)+30,int(cols/2)-30:int(cols/2)+30] = 0

with tf.Session() as sess:
	X = tf.placeholder(tf.float32,shape=img_man.shape)
	Y = tf.zeros(shape=img_man.shape,dtype=tf.float32)
	Z = tf.complex(X,Y)
	X_fft = tf.fft(Z)
	X_fft2 = tf.fft2d(Z)
	sess.run(tf.global_variables_initializer())
	feed_dict = {X:img_man}

	X_ifft = tf.real(tf.ifft(X_fft))
	X_ifft2 = tf.real(tf.ifft2d(X_fft2))
	X_fre = sess.run(X_fft,feed_dict=feed_dict)
	X_fre2 = sess.run(X_fft2,feed_dict=feed_dict)
	X_recon = sess.run(X_ifft,feed_dict=feed_dict)
	X_recon2 = sess.run(X_ifft2,feed_dict=feed_dict)

	'''
	print(X_fre)
	print(X_fre.shape)
	print(X_fre2)
	print(X_fre2.shape)
	print(X_recon)
	print(X_recon.shape)
	print(X_recon2)
	print(X_recon2.shape)
	'''

	#X_fftshift = np.fft.fftshift(X_fre)
	X_fftshift2 = np.fft.fftshift(X_fre2)
	#X_fftshift_mask = X_fftshift*mask
	X_fftshift_mask2 = X_fftshift2*mask

	#X_ifftshift = np.fft.ifftshift(X_fftshift_mask) 
	#X_re = np.abs(np.fft.ifft2(X_ifftshift))
	X_ifftshift2 = np.fft.ifftshift(X_fftshift_mask2) 
	X_re2 = np.abs(np.fft.ifft2(X_ifftshift2))

	#mask = np.ones(img_man.shape,np.uint8)
	#mask[int(rows/2)-30:int(rows/2)+30,int(cols/2)-30:int(cols/2)+30] = 0
	#--------------------------------
	f1 = np.fft.fft2(img_man)
	f1shift = np.fft.fftshift(f1)
	f1shift = f1shift*mask
	f2shift = np.fft.ifftshift(f1shift) 
	img_new = np.fft.ifft2(f2shift)

	img_new = np.abs(img_new)

	img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))

	print('numpy fft: ',f1)
	print('tf fft:', X_fre2)

	'''	plt.imshow(X_re, cmap='Greys_r')
	plt.axis('off')
	plt.show()'''

	
	plt.imshow(X_re2, cmap='Greys_r')
	plt.axis('off')
	plt.show()

	plt.imshow(img_new, cmap='Greys_r')
	plt.axis('off')
	plt.show()
	

	'''print(X_re2.shape)
				print(img_new.shape)
				cv2.imshow('tf',X_re2)
				cv2.imshow('numpy',img_new)
				k=cv2.waitKey(0)'''

