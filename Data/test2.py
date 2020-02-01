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
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transforms= transforms.Compose([
        transforms.Resize((50,50)),
        transforms.RandomRotation(30),
#        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
#        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

train_dataset = datasets.ImageFolder(train_dir, transform=transforms)
print(train_dataset[0][0][0])
print(len(train_dataset[0][0][0]))
trainloader = torch.utils.data.DataLoader(train_dataset , batch_size=4,
                                          shuffle=True, num_workers=4)
####################### CNN #######################

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels =  64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)


    def forward(self, t):
        t = self.layer1(t)
        t = self.layer2(t)
        t = t.reshape(t.size(0), -1)
        t = self.drop_out(t)
        t = self.fc1(t)
        t = self.fc2(t)
        return t



























