#main .py
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from torch import nn,tensor,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import body_parts

ap = argparse.ArgumentParser(description='Train.py')
# Initializing Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_layers', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
filepath = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_layers
power = pa.gpu
epochs = pa.epochs


train_loader, test_loader, validation_loader = body_parts.load_data(filepath)
model, optimizer, criterion = body_parts.nn_setup(structure,dropout,hidden_layer1,lr,power)
body_parts.train_network(model, optimizer, criterion, epochs, 40, train_loader, power)
body_parts.save_checkpoint(path,structure,hidden_layer1,dropout,lr)


print("Model Trained!!!")