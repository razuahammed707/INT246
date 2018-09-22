import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from matplotlib.ticker import FormatStrFormatter
import json
import PIL
from PIL import Image
import argparse
import just
#from train import model as model
parser=argparse.ArgumentParser(description="predict.py")
parser.add_argument('data_dir',action="store",nargs='*',default="./flowers/")
parser.add_argument('image_path',action="store", nargs='*',default="./flowers/test/22/image_05360.jpg")
parser.add_argument('checkpoint',action="store", nargs='*',default='./checkpoint.pth')
parser.add_argument('--top_k',action="store",dest="top_k",default=5,type=int)
parser.add_argument('--gpu',action="store",type=str,dest="gpu",default='on')
parser.add_argument('--jfile_path',action="store",dest="jfile_path",default='cat_to_name.json')

mango=parser.parse_args()

flower=mango.data_dir
image_path=mango.image_path
topk=mango.top_k
filepath=mango.checkpoint
gpu=mango.gpu
jfile_path=mango.jfile_path

#trainloader , validloader, testloader=just.loading()
model=just.load_checkpoint(filepath)

with open(jfile_path, 'r') as json_file:
    cat_to_name = json.load(json_file)
	
probabilities=just.predict(image_path,model,jfile_path,topk,gpu)
#print(probabilities)
probability= np.array(probabilities[0])#converting values(tensor) into array
#print(probability)
lables= [jfile_path[str(index)] for index in np.array(probabilities[1])]#u will get names of top k

#printing the lables with its probabilities
i=0
while i < topk:
    print("{} with a probability of {:.4f}".format(lables[i],probability[i]))
    i += 1