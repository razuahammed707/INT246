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
from collections import OrderedDict


def loading(flower= "./flowers"):
    data_dir = flower
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)

    return trainloader , validloader, testloader,train_data

stru={'vgg16':25088,"densenet121":1024}

def setup(nmodel='vgg16',hidden_size= [1000, 200],output_size = 102,dropout=0.5,gpu='on'):#added gpu for user
    if nmodel=='vgg16':
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('fc1', nn.Linear(stru[nmodel], hidden_size[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hidden_size[1],output_size)),
            ('output', nn.LogSoftmax(dim=1))]))


        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        if gpu=='on':
            model.cuda()
            print("just passed setup!!")
        return model, criterion, optimizer
    

def training(model,criterion,optimizer,trainloader,validloader,epochs = 3,print_every = 40,gpu='on'):#might need validloader
    
    steps = 0
    if gpu=='on':
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu=='on':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()  
            #will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval model instead of training mode.
                validloss=0
                accuracy=0
                print("inside training")
                for ii, (inputs1,labels1) in enumerate(validloader):
                    optimizer.zero_grad()
                    if gpu=='on':
                        inputs1, labels1 = inputs1.to('cuda:0') , labels1.to('cuda:0')
                        model.to('cuda:0')
                    with torch.no_grad():
                        #validloss,accuracy=validation_fun(model,testloader,criterion):
                        outputs = model.forward(inputs1)
                        validloss = criterion(outputs,labels1)
                        ps = torch.exp(outputs).data
                        equality = (labels1.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                print("Epoch: {}/{}  |".format(e+1, epochs),
                      "Training Loss: {:.3f} |".format(running_loss/print_every),
                      "Validation Lost {:.3f} |".format(validloss / len(validloader)),
                      "Valid Accuracy: {:.3f} ".format(accuracy /len(validloader)))
            
                running_loss = 0
    print("your network has been trained")

def save_checkpoint(train_data,model,nmodel='vgg16',path='checkpoint.pth'):
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    checkpoint={'nmodel':'vgg16',
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx}
    torch.save(checkpoint,path)
    print("your check point has been saved!!!")
    
    
'''def save_checkpoint(train_data,model,nmodel='vgg16',path='checkpoint.pth',hidden_size= [1000, 200],epochs = 3,dropout=0.5,print_every = 40,output_size=102):
        # TODO: Save the checkpoint 
        model.class_to_idx = train_data.class_to_idx
        checkpoint={'output_size':output_size,
                    'nmodel':nmodel,
                    'hidden_size': hidden_size,
                    'dropout':dropout,
                    'epochs':epochs,
                    'print_every':print_every,
                    #'hidden_size':[each.out_features for each in model.hidden_size],
                    'state_dict':model.state_dict(),
                    'class_to_idx':model.class_to_idx}
        torch.save(checkpoint,path)#saving checkpoint in 'checkpoint.pth'
        print("your check point has been saved!!!")'''  

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath='checkpoint.pth'):
    checkpoint=torch.load(filepath)
    
    if checkpoint['nmodel']=='vgg16':
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(0.5)),
        ('fc1', nn.Linear(25088,1000)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(1000,200)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(200,102)),
        ('output', nn.LogSoftmax(dim=1))]))
            
    model.classifier=classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
'''def load_checkpoint(filepath='checkpoint.pth'):
    checkpoint=torch.load(filepath)
    
    hidden_size=checkpoint['hidden_size']
    nmodel=checkpoint['nmodel']
    output_size=checkpoint['output_size']
    dropout=checkpoint['dropout']

    model,_,_ = setup(nmodel,hidden_size,output_size,dropout)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model'''
        
# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):#insert path over here
    im=Image.open(image)
    
    image_fix= transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    
    apply=image_fix(im)
    return apply

def predict(image_path, model,jfile_path='cat_to_name.json',topk=5,gpu='on'):
    if gpu=='on':
        model.to('cuda:0')
    img_torch = process_image(image_path)
    
    #img_torch=torch.from_numpy(img_torch).type(torch.FloatTensor) 
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if gpu=='on':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
  
    probs=F.softmax(output.data,dim=1)
    top_probs,top_labs = probs.topk(topk)
    
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    print(top_probs)
    print(top_labs)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    print(top_labels)
    top_flowers = [jfile_path[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels,top_flowers
'''def predict(image_path, model, topk=5,gpu='on'):
    if gpu=='on':
        model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if gpu=='on':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)'''
