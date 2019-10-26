import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import processing_functions

from collections import OrderedDict



# to save the checkpoints
def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):

    model.class_to_idx = train_datasets.class_to_idx 

    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 102,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}

    torch.save(checkpoint, 'project_checkpoint.pth')
    
# function to load the checkpoint 
def loading_model (file_path):
    checkpoint = torch.load (file_path) #loading checkpoint from a file
    
    model = models.alexnet (pretrained = True) #function works solely for Alexnet
    #you can use the arch from the checkpoint and choose the model architecture in a more generic way:
    #model = getattr(models, checkpoint['arch']
        
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    
    for param in model.parameters(): 
        param.requires_grad = False #turning off tuning of the model
    
    return model

def valid(model, valid_loader, criterion, gpu='cuda'):
    model.to ('cuda')
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        p = torch.exp(output)
        equality = (labels.data == p.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


def test_accuracy(model, test_loader):
    test_correct = 0
    test_total = 0

       with torch.no_grad ():
       for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model (inputs)
        _, predicted = torch.max (outputs.data,1)
        test_total += labels.size (0)
        test_correct += (predicted == labels).sum().item()

       print('Accuracy of the network on test images: %d %%' % (100 * test_correct / test_total))

def train_classifier(model, optimizer, criterion, epochs, train_loader, valid_loader, gpu='cuda'):

 #change to cuda if enabled
  model.to ('cuda')
  epochs = 8
  ForPrinting = 40
  steps = 0


  for e in range (epochs): 
              running_loss = 0
      for ii, (inputs, labels) in enumerate (train_loader):
          steps += 1
    
          inputs, labels = inputs.to('cuda'), labels.to('cuda')
    
          optimizer.zero_grad () #optimizer is working on classifier paramters only
    
        # Forward and backward passes
          outputs = model.forward (inputs) #calculating output
          loss = criterion (outputs, labels) #calculating loss
          loss.backward () 
          optimizer.step ()  
    
          running_loss += loss.item () 
    
          if steps % ForPrinting == 0:
            model.eval () 
            with torch.no_grad():
                valid_loss, accuracy = valid(model, valid_loader, criterion)
            
              print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/ForPrinting),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))
            
                running_loss = 0
            
            # Make sure training is back on
             model.train()
                
                
def predict(image_path, model, topk=5, gpu='cuda'):   
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

     img = (data_dir + '/test' + '/10/' + 'image_07104.jpg')
     probs, classes = predict(img, model)
     print(probs)
     print(classes)