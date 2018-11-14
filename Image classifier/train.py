import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import argparse

import just

parser=argparse.ArgumentParser(description="train.py")

parser.add_argument('data_dir',action="store",nargs='*',default="./flowers/")

parser.add_argument('--hidden_size',action="store",type=int,dest="hidden_size",default=[1000,200])
parser.add_argument('--nmodel',action="store",type=str,dest="nmodel",default='vgg16')
parser.add_argument('--output_size',action="store",type=int,dest="output_size",default=102)
parser.add_argument('--dropout', action = "store",type=float,dest ="dropout", default = 0.5)
parser.add_argument('--gpu',action="store",type=str,dest="gpu",default='on')

parser.add_argument('--epochs',action="store",type=int,dest="epochs",default=3)
parser.add_argument('--print_every',action="store",type=int,dest="print_every",default=40)

parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")

mango=parser.parse_args()

flower=mango.data_dir
hidden_size=mango.hidden_size
nmodel=mango.nmodel
output_size=mango.output_size
dropout=mango.dropout 
gpu=mango.gpu

epochs=mango.epochs
print_every=mango.print_every

path=mango.save_dir

trainloader , validloader, testloader,train_data=just.loading(flower)

model, criterion, optimizer=just.setup(nmodel,hidden_size,output_size,dropout,gpu)

just.training(model,criterion,optimizer,trainloader,validloader,epochs,print_every,gpu)

just.save_checkpoint(train_data,model,nmodel,path)#,hidden_size,epochs,dropout,print_every,output_size