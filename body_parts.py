import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim,tensor
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import json
from collections import OrderedDict

import argparse

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

def load_data(filepath  = "./flowers" ):
    '''
    Arguments : the datas path
    Returns : The loaders for datasets
    This function receives the location of the image files, applies data augmentation and converts the images to tensor to be able to be used in neural network
    '''
    data_dir = filepath
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #transfroms applied
    # DONE: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    test_transforms=transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    validation_transforms=transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data =datasets.ImageFolder(test_dir,transform=test_transforms)
    validation_data= datasets.ImageFolder(valid_dir,transform=validation_transforms)
    
    # DONE: Using the image datasets and the trainforms, define the dataloaders
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    
    return train_loader , test_loader, validation_loader

  
def pre_trained_NN(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid structure name.Choose from vgg16/densenet121/alexnet".format(structure))
        
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        
        if torch.cuda.is_available() and power = 'gpu':
            model.cuda()
        
        return model, criterion, optimizer
    
def train_network(model, criterion, optimizer, epochs = 12, print_every=40, loader=trainloader, power='gpu'):
    steps = 0
    model.to('cuda')
    print("<<<<<<<<<<<<<<<<<<<<Training Of the Network is Started>>>>>>>>>>>>>>>>>>>>")
    for e in range(epochs):
        running_loss = 0
        if torch.cuda.is_available() and power='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs,labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
                if steps % print_every == 0:
                    model.eval()
                    validation_lost = 0
                    accuracy=0
            
            
                    for ii, (inputs2,labels2) in enumerate(validation_loader):
                        optimizer.zero_grad()
                
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            validation_lost = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                    validation_lost = validation_lost / len(validation_loader)
                    accuracy = accuracy /len(validation_loader)
            
                    
            
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                        "Loss: {:.4f}".format(running_loss/print_every),
                        "Validation Lost {:.4f}".format(validation_lost),
                        "Accuracy: {:.4f}".format(accuracy))
            
            
                    running_loss = 0
    print("-------------- Finished training -----------------------")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))
    
def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12):
    model.class_to_idx = train_data.class_to_idx

    model.cpu

    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)


def load_checkpoint(path='checkpoint.pth'):
    
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = pre_trained_NN(structure , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    for i in image_path:
        path = str(i)
        pil_image = Image.open(i) 
   
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image
  

def predict(image_path, model, labels='', topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and power='gpu':
        model.to('cuda:0')
    model.eval()

    
    image = process_image(image_path)
    
    
    image = torch.from_numpy(np.array([image])).float()
    
    
    image = Variable(image)
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label