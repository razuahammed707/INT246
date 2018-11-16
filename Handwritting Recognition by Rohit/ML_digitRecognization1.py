
# coding: utf-8

# In[1]:

import os
import pandas as p
import numpy as np
import matplotlib.pyplot as plt
Df=p.read_csv("mnist_train.csv")
data = Df.values.astype(np.float32)
#Copy of the array, cast to a specified type.



# In[2]:


sample = np.random.choice(data.shape[0])
#Generates a random sample from a given 1-D array
plt.imshow(data[sample,1:].reshape(28,28))


# In[3]:


fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten() 
'''flatten() is method of the numpy array. Instead of an iterator, 
It returns a flattened version of the array:
The advantage that you will save one loop'''

'''To get a idea what the images in MNIST look like, 
let's visualize examples of the digits 0-9 after reshaping the 784-pixel vectors
 from our feature matrix into the original 28 × 28 image that we can plot 
 via matplotlib'''

X_train=data[:,1:]
y_train=data[:,0]
for i in range(10): 
    img = X_train[y_train == i][0].reshape(28, 28) 
    ax[i].imshow(img, cmap='Greys', interpolation='nearest') 
    ax[0].set_xticks([]) 
    ax[0].set_yticks([]) 
plt.tight_layout() 
'''tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.'''
plt.show() 
#process of generating data points beween already existing data points

# In[4]:

'''In addition, let's also plot multiple examples of the same digit to 
see how different those handwriting examples really are:'''

fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,  sharey=True,)
ax = ax.flatten() 
for i in range(25): 
    img = X_train[y_train == 7][i].reshape(28, 28) 
    ax[i].imshow(img, cmap='Greys', interpolation='nearest') 
    ax[0].set_xticks([]) 
    ax[0].set_yticks([]) 
plt.tight_layout() 
plt.show()


# In[5]:


from sklearn.utils import shuffle 
'''Shuffling the training set prior to every epoch to prevent the algorithm from getting stuck in cycles'''
data = data[:20000]
    
data = shuffle(data)

X = data[:, 1:]
Y = data[:, 0].astype(np.int32)

Xtrain = X[:-2000]
Ytrain = Y[:-2000]
Xtest  = X[-2000:]
Ytest  = Y[-2000:]


# In[6]:


from sklearn.neural_network import MLPClassifier 

classifier = MLPClassifier()
'''it is a class of feedforward artificial neural network. ... 
MLP utilizes a supervised learning technique called backpropagation for training. '''
classifier.fit(Xtrain , Ytrain)

print(" Training Accuracy : " , classifier.score(Xtrain , Ytrain))
print(" Testing Accuracy : " , classifier.score(Xtest , Ytest))


# In[7]:


n=int(input("Enter any number between 1 to 2000 "))

plt.imshow(Xtest[n,:].reshape(28,28))
print(classifier.predict((Xtest[n,:].reshape(1,784))))

