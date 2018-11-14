
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import randint

# In[4]:


df = pd.read_csv('seeds.csv')
#df.head(7)


# In[6]:


x = np.array(df.loc[:,['AREA','PERIMETER','COMPACTNESS','LENGTH','WIDTH','ASSYMMETRY_COEFFICIENT','GROOVE_LENGTH',]])
y = np.array(df['TYPE'])


# In[8]:


lr = 0.6   #learning rate
e =0
D = [0,0,0]  #size depending on the number of neurons
epoch = 5    # number of iterations
#Assigning the weights
w=np.random.rand(7,3)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.28,random_state=7)
[r,c] = X_train.shape
#print(w)
while(e<epoch):
    for i in range(r):
        for j in range(3): #size of neurons
            temp =0
            for k in range(c):
                temp = temp + ((w[k,j]-X_train[i,k])**2)
            D[j]=temp
            min = D[0]
            J = 0
            for i in range(3):
                if(D[i] <= min):
                    min = D[i]
                    J = i
            #print('winning unit is',J+1)
            #print('weight updation ...')
            for m in range(c):
                 w[m,J]=w[m,J] + (lr*(x[i,m]-w[m,J]))
            #print('Updated weights',w)
    e=e+1
    lr = lr*0.5


#print(w)
out = []
for i in range(r):
        for j in range(3): #size of neurons
            temp =0
            for k in range(c):
                temp = temp + ((w[k,j]-X_train[i,k])**2)
            D[j]=temp
            min = D[0]
            J = 0
            for i in range(3):
                if(D[i] <= min):
                    min = D[i]
                    J = i
        #out.append(np.random.randint(1,4))
out.append(y)
print(out)
