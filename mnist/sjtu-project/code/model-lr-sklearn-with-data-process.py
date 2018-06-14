import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

np.set_printoptions(suppress=True)

def pca(X_full):
	pca256c=PCA(n_components=256)
	data_pca256c = pca256c.fit_transform(X_full)
	print(data_pca256c.shape)
	return data_pca256c

train_num=60000
test_num=10000

"""train_data = np.fromfile("./mnist_train/mnist_train_data",dtype=np.uint8).reshape([-1,2025])
train_label = np.fromfile("./mnist_train/mnist_train_label",dtype=np.uint8)


#shuffle the training data
perm=np.arange(len(train_label))
np.random.shuffle(perm)
train_data=train_data[perm]
train_label=train_label[perm]

#select train_num samples for training
perm=np.arange(len(train_label))
np.random.shuffle(perm)
X_train=train_data[perm[:train_num]]
Y_train=train_label[perm[:train_num]]
print((Y_train))
for i in range(10):
	print(len(np.where(Y_train==i)[0]))
#Y_train=one_hot(Y_train)
print(len(Y_train))

test_data=np.fromfile("./mnist_test/mnist_test_data",dtype=np.uint8).reshape([-1,2025])
test_label = np.fromfile("./mnist_test/mnist_test_label",dtype=np.uint8)
#test_data = pca(test_data)

#shuffle test data
perm=np.arange(len(test_label))
np.random.shuffle(perm)
test_data=test_data[perm]
test_label=test_label[perm]

#select test_num samples for testing
perm=np.arange(len(test_label))
np.random.shuffle(perm)
X_test=test_data[perm[:test_num]]
Y_test=test_label[perm[:test_num]]
print(np.unique(Y_test))
for i in range(10):
	print(len(np.where(Y_test==i)[0]))
#Y_test=one_hot(Y_test)
print(len(Y_test))"""

"""def del_point(pic):
	max_size=6
	origin_size=45
	for _ in range(1):
		x=len(pic)*[0]
		pic=np.insert(pic,0,values=x,axis=1)
		pic=np.insert(pic,-1,values=x,axis=1)
	for _ in range(1):
		y=len(pic.T)*[0]
		pic=np.insert(pic,0,values=y,axis=0)
		pic=np.insert(pic,-1,values=y,axis=0)
	
	#print(pic.shape)
	cnt=0
	cnt_tr=0
	for i in range(1,origin_size-max_size+2):
		for j in range(1,origin_size-max_size+2):
			flag=False
			if(len(np.nonzero(pic[i-1][j:j+max_size])[0])==0 
				and len(np.nonzero(pic[i+6][j:j+max_size])[0])==0 
				and len(np.nonzero(pic[:,j-1][i:i+max_size])[0])==0 
				and len(np.nonzero(pic[:,j+6][i:i+max_size])[0])==0):
				flag=True
			cnt+=1
			if(flag):
				cnt_tr+=1
				for r in range(i,i+6):
					for c in range(j,j+6):
						pic[r][c]=0
	#print(cnt,cnt_tr)
	return pic[1:46][:,1:46]
			
	
def del_black(data):
	res=[]
	cut_size_col=25
	cut_size_row=21
	for idx in range(len(data)):
		pic=data[idx]
		pic=del_point(pic)
		row_del=[]
		col_del=[]
		perm=np.arange(45)
		np.random.shuffle(perm)
		# print(pic.shape)
		for i in perm:
			
			if(len(np.nonzero(pic[i])[0])==0):
				row_del.append(i)
			if(len(row_del)==cut_size_row):
				break
		
		for i in perm:
			
			if(len(np.nonzero(pic[:,i])[0])==0):
				col_del.append(i)
			if(len(col_del)==cut_size_col):
				break
		pic=np.delete(pic,row_del,axis=0)
		pic=np.delete(pic,col_del,axis=1)
		
		if(len(row_del)<cut_size_row):
			perm=np.arange(len(pic))
			np.random.shuffle(perm)
			add_row_del=perm[:cut_size_row-len(row_del)]
			pic=np.delete(pic,add_row_del,axis=0)
		if(len(col_del)<cut_size_col):
			perm=np.arange(45-len(col_del))
			np.random.shuffle(perm)
			add_col_del=perm[:cut_size_col-len(col_del)]
			pic=np.delete(pic,add_col_del,axis=1)
		res.append(pic)
	
	print(np.array(res).shape)
	return np.array(res)"""

def del_point(pic):
	row=len(pic)
	col=len(pic.T)
	visited=np.zeros([row,col])
	
	drc=[[0,1],[1,0],[0,-1],[-1,0]]
	arr=[]
	def dfs(i,j):
		arr.append([i,j])
		visited[i][j]=1
		for d in range(4):
			new_i=i+drc[d][0]
			new_j=j+drc[d][1]
			if(new_i>=0 and new_j>=0 and new_i<row and new_j<col and visited[new_i][new_j]==0 and pic[new_i][new_j]!=0):
				dfs(new_i,new_j)
	
	for r in range(row):
		for c in range(col):
			if(pic[r][c]!=0 and visited[r][c]==0):
				dfs(r,c)
				if(len(arr)<40):
					for i in arr:
						pic[i[0]][i[1]]=0
				arr=[]

	return pic
	
def del_black(data):
	res=[]
	cut_size_col=21
	cut_size_row=21
	for idx in range(len(data)):
		pic=data[idx]
		pic=del_point(pic)
		row_del=[]
		col_del=[]
		perm=np.arange(45)
		np.random.shuffle(perm)
		
		for i in perm:
			if(len(np.nonzero(pic[i])[0])==0):
				row_del.append(i)
		
		for i in perm:
			if(len(np.nonzero(pic[:,i])[0])==0):
				col_del.append(i)
		pic=np.delete(pic,row_del,axis=0)
		pic=np.delete(pic,col_del,axis=1)
		
		if(len(row_del)<cut_size_row):
			perm=np.arange(len(pic))
			np.random.shuffle(perm)
			add_row_del=perm[:cut_size_row-len(row_del)]
			pic=np.delete(pic,add_row_del,axis=0)
		else:
			up=int((len(row_del)-cut_size_row)/2)
			down=len(row_del)-cut_size_row-up
			for _ in range(up):
				x=np.array(len(pic.T)*[0]).astype(np.uint8)
				pic=np.insert(pic,0,values=x,axis=0)
			for _ in range(down):
				x=np.array(len(pic.T)*[0]).astype(np.uint8)
				pic=np.row_stack((pic,np.array(x)))
			
		if(len(col_del)<cut_size_col):
			perm=np.arange(45-len(col_del))
			np.random.shuffle(perm)
			add_col_del=perm[:cut_size_col-len(col_del)]
			pic=np.delete(pic,add_col_del,axis=1)
		else:
			left=int((len(col_del)-cut_size_col)/2)
			right=len(col_del)-cut_size_col-left
			for _ in range(left):
				x=np.array(len(pic)*[0]).astype(np.uint8)
				pic=np.insert(pic,0,values=x,axis=1)
			for _ in range(right):
				x=np.array(len(pic)*[0]).astype(np.uint8)
				pic=np.column_stack((pic,np.array(x)))

		res.append(np.array(pic))
	
	print(np.array(res).shape)
	return np.array(res)

X_train =np.fromfile("./mnist_train/mnist_train_data",dtype=np.uint8).reshape([-1,2025])#np.load('train_data_cut.npy').reshape([-1,45,45])
X_test=np.fromfile("./mnist_test/mnist_test_data",dtype=np.uint8).reshape([-1,2025])#np.load('test_data_cut.npy').reshape([-1,45,45])
#X_train=np.load('train_data_2.npy')
#X_test=np.load('test_data_2.npy')
Y_train=np.fromfile("./mnist_train/mnist_train_label",dtype=np.uint8)
Y_test=np.fromfile("./mnist_test/mnist_test_label",dtype=np.uint8)


# X_train=del_black(X_train).reshape([-1,576])
# np.save('train_data_3.npy',X_train)
# X_test=del_black(X_test).reshape([-1,576])
# np.save('test_data_3.npy',X_test)

print(X_train.shape)

perm=np.arange(len(Y_train))
np.random.shuffle(perm)
X_train=X_train[perm]
Y_train=Y_train[perm]

perm=np.arange(len(Y_test))
np.random.shuffle(perm)
X_test=X_test[perm]
Y_test=Y_test[perm]



model = LogisticRegression(max_iter=150)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print(classification_report(Y_test,y_pred))