import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

####################### CNN #######################
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels =  64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(179776, 64)
        self.fc2 = nn.Linear(64, 2)


    def forward(self, t):
        t.cuda(device)
#        print("Before sending to first conv2d layer")
#        print(t.shape)
        t = self.layer1(t)
#        print("First conv2d layer")
#        print(t.shape)
        t = self.layer2(t)
#        print("Second conv2d layer")
#        print(t.shape)
        t = t.reshape(t.size(0), -1)
#        print("Reshape")
#        print(t.shape)
        t = self.drop_out(t)
#        print("Drop_out")
#        print(t.shape)
        t = self.fc1(t)
#        print("fc1")
#        print(t.shape)
        t = self.fc2(t)
#        print("fc2")
#        print(t.shape)
        return t
