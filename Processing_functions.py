  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from PIL import Image
import json

import torch
from torch import nn
from torch import optim
from matplotlib.ticker import FormatStrFormatter
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def dataTransform():
    
    train_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                             transforms.RandomResizedCrop (224),
                                             transforms.RandomHorizontalFlip (),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

    valid_transforms = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

    test_transforms = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])
    
    return train_transforms, valid_transforms, test_transforms

def load_datasets(train_dir, train_transforms, valid_dir, valid_transforms, test_dir, test_dir):
    train_datasets = datasets.ImageFolder (train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder (valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder (test_dir, transform = test_dir)
    
    return train_datasets, valid_datasets, test_datasets


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
image_path = 'flowers/train/10/image_07086.jpg'
img = process_image(image_path)
#img.shape
imshow(img)

def process_image(image):
  
    img = Image.open (image) #loading image
    
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    
    img_tensor = adjustments(img)
    
    return img_tensor

def load_json(json_file):
    with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

def check_sanity():
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(211)
    
    index = 1
    path = test_dir + '/1/image_06743.jpg'

    prob = predict(path, model)
    img = process_image(path)
    
    

    axs = imshow(img, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(index)])
    axs.show()
    
    
    a = np.array(prob[0][0])
    b = [cat_to_name[str(index + 1)] for index in np.array(prob[1][0])]
    
    
    N=float(len(b))
    fig,ax = plt.subplots(figsize=(8,3))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(b)
    ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
    ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()