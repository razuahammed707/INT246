# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:21:40 2018

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:08:07 2018

@author: Pavilion
"""


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


X_train=data[:,1:]
y_train=data[:,0]
for i in range(10): 
    img = X_train[y_train == i][0].reshape(28, 28) 
    ax[i].imshow(img, cmap='Greys', interpolation='nearest') 
    ax[0].set_xticks([]) 
    ax[0].set_yticks([]) 
plt.tight_layout() 
plt.show() 


# In[4]:



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

classifier.fit(Xtrain , Ytrain)

print(" Training Accuracy : " , classifier.score(Xtrain , Ytrain))
print(" Testing Accuracy : " , classifier.score(Xtest , Ytest))


# In[7]:


n=int(input("Enter any number between 1 to 2000 "))

plt.imshow(Xtest[n,:].reshape(28,28))
print(classifier.predict((Xtest[n,:].reshape(1,784))))

