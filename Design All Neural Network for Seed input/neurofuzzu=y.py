import pandas as pd
import numpy as np
from random import random
#from matplotlib import pyplot as plt

df = pd.read_csv('seeds.csv')
df.head(7)


x = np.array(df.loc[:,['AREA','PERIMETER','COMPACTNESS','LENGTH','WIDTH','ASSYMMETRY_COEFFICIENT','GROOVE_LENGTH',]])

#x = [[0,0],[0,1],[1,0],[1,1]]
#x1 = [0,1,0,1]
#for i in range(3):
def perceptron(x):
    if x >= 0:
        return 1
    else:
        return -1
w = [0,0,0,0,0,0,0]
b = 0
#f = 0
epoch = 10
#expectedOutput = [-1,-1,-1,1]
expectedOutput = np.array(df['TYPE'])
error = 0
lr = 0.1
for k in  range(epoch):
    for i in range(1,140):
        y = x[i][0]*w[0] + x[i][1]*w[1] +x[i][2]*w[2] + x[i][3]*w[3] +x[i][4]*w[4] +x[i][5]*w[5] + x[i][6]*w[6] + b
        output = perceptron(y)
        error = expectedOutput[i] - output
        if(error != 0):
            w[0] = w[0] + lr*expectedOutput[i]*x[i][0]
            w[1] = w[1] + lr*expectedOutput[i]*x[i][1]
            w[2] = w[2] + lr*expectedOutput[i]*x[i][2]
            w[3] = w[3] + lr*expectedOutput[i]*x[i][3]
            w[4] = w[4] + lr*expectedOutput[i]*x[i][4]
            w[5] = w[5] + lr*expectedOutput[i]*x[i][5]
            w[6] = w[6] + lr*expectedOutput[i]*x[i][6]
#                w[j] = w[j] + lr*x[i][j]*error[i]
            b = b + lr*expectedOutput[i]
print (b,w)
for i in range(1,140):
    y = x[i][0]*w[0] + x[i][1]*w[1] +x[i][2]*w[2] + x[i][3]*w[3] +x[i][4]*w[4] +x[i][5]*w[5] + x[i][6]*w[6] + b
    output = perceptron(y)
    print(output)
#print("hello")
