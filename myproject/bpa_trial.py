import pandas as pd
import numpy as np

df = pd.read_csv('seeds.csv')

#Input array
y = []
y = np.array([[(x - 1) / (2 - 1)] for x in df['TYPE'].tolist()])

X = np.array(df.loc[:,['AREA','PERIMETER','COMPACTNESS','LENGTH','WIDTH','ASSYMMETRY_COEFFICIENT','GROOVE_LENGTH']])
#Output

x_md = []
for i in range(X.shape[1]):
    x_md.append([min(X[:,i]), max(X[:,i])])

for i in range(X.shape[1]):
    X[:,i] = [(n - x_md[i][0]) / (x_md[i][1] - x_md[i][0]) for n in X[:,i]]

ips = [[a,b,c,d,e,f,g] for (a,b,c,d,e,f,g) in zip(X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5], X[:,6])]

X = np.array(ips)

#X[i] = (X[i] - min(X)) / (max(X) - min(X))

#y=np.array([[1],[1],[0]])
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
#Variable initialization
epoch=500 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):

#Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #Backpropagation

    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

output = [[round(n[0]) + 1] for n in output]

print ("Output: \n", output)
#print("Error: \n", E)
