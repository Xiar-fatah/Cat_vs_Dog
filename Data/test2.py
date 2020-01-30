import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models
import torch.nn as nn
import torch.nn.functional as F
####################### Prepossessing Data #######################
train_dir = './dataset'
transforms= transforms.Compose([transforms.Resize(50),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
           [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform=transforms)

trainloader = torch.utils.data.DataLoader(train_dataset , batch_size=4,
                                          shuffle=True, num_workers=2)
####################### CNN #######################

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        


    def forward(self, t):

        return t


