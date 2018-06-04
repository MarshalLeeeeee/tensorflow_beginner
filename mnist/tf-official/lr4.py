import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

np.set_printoptions(suppress=True)

train_num=60000
test_num=10000

def pca(X_full):
	pca256c=PCA(n_components=150)
	data_pca256c = pca256c.fit_transform(X_full)
	print(data_pca256c.shape)
	return data_pca256c

def del_black(data):
	res=[]
	for i in range(len(data)):
		pic=data[i]
		row_del=[]
		col_del=[]
		perm=np.arange(45)
		np.random.shuffle(perm)
		for i in perm:
			if(len(np.nonzero(pic[i])[0])<=3):
				row_del.append(i)
			if(len(row_del)==21):
				break
		for i in perm:
			if(len(np.nonzero(pic[:,i])[0])<=3):
				col_del.append(i)
			if(len(col_del)==21):
				break
		pic=np.delete(pic,row_del,axis=0)
		pic=np.delete(pic,col_del,axis=1)
		if(len(row_del)<21):
			perm=np.arange(len(pic))
			np.random.shuffle(perm)
			add_row_del=perm[:21-len(row_del)]
			pic=np.delete(pic,add_row_del,axis=0)
		if(len(col_del)<21):
			perm=np.arange(45-len(col_del))
			np.random.shuffle(perm)
			add_col_del=perm[:21-len(col_del)]
			pic=np.delete(pic,add_col_del,axis=1)
		res.append(pic)
	
	print(np.array(res).shape)
	return np.array(res)

def get_train_data(train_data,train_label):
	perm=np.arange(len(train_label))
	np.random.shuffle(perm)
	train_data=train_data[perm]
	train_label=train_label[perm]
	train_data=del_black(train_data)

	perm=np.arange(len(train_label))
	np.random.shuffle(perm)
	_X=train_data[perm[:train_num]]
	_Y=train_label[perm[:train_num]]
	print(_X.shape,_Y.shape)
	return _X,_Y

def get_test_data(test_data,test_label):
	perm=np.arange(len(test_label))
	np.random.shuffle(perm)
	test_data=test_data[perm]
	test_label=test_label[perm]
	test_data=del_black(test_data)

	perm=np.arange(len(test_label))
	np.random.shuffle(perm)
	_X=test_data[perm[:test_num]]
	_Y=test_label[perm[:test_num]]
	print(_X.shape,_Y.shape)
	return _X,_Y


train_data = np.fromfile("./mnist_train/mnist_train_data",dtype=np.uint8).reshape([-1,45,45])
train_label = np.fromfile("./mnist_train/mnist_train_label",dtype=np.uint8)

test_data=np.fromfile("./mnist_test/mnist_test_data",dtype=np.uint8).reshape([-1,45,45])
test_label = np.fromfile("./mnist_test/mnist_test_label",dtype=np.uint8)

X_train,Y_train=get_train_data(train_data,train_label)
X_test,Y_test=get_test_data(test_data,test_label)
X_train=X_train.reshape([-1,576])
X_test=X_test.reshape([-1,576])
#X_train=pca(X_train)
#X_test=pca(X_test)

model = LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print(classification_report(Y_test,y_pred))
