import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import model_functions
import processing_functions
import json
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim


parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 10000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 20, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arguments = parser.parse_args()

# Image data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for the training, validation, and testing sets
train_transforms, valid_transforms, test_transforms = processing_functions.data_transforms()

# Load the datasets with ImageFolder
train_datasets, valid_datasets, test_datasets = processing_functions.load_datasets(train_dir, train_transforms, valid_dir, valid_transforms, test_dir, test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)

# Build and train the neural network (Transfer Learning)
if arguments.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif arguments.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
    
print(model)

# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

# Build custom classifier
 classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, 2048)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (2048, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))

model.classifier = classifier

# Loss function (since the output is LogSoftmax, we use NLLLoss)
criterion = nn.NLLLoss()

# Gradient descent optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
    
model_functions.train_classifier(model, optimizer, criterion, arguments.epochs, train_loader, valid_loader, arguments.gpu)
    
model_functions.test_accuracy(model, test_loader, arguments.gpu)

model_functions.save_checkpoint(model, train_datasets, arguments.arch, arguments.epochs, arguments.learning_rate, arguments.hidden_units, input_size)  
