import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import train
import matplotlib.image as mpimg

#import cat as c

#We need to 1. grayscale 2. resize 3. label 4. put everything in a tensor
train_dir = './train'
resize = 50
transforms = transforms.Compose(
    [transforms.Resize(resize),
     transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img2 = mpimg.imread('cat.0.jpg')
imgplot = plt.imshow(img2)
plt.show()


print(img2)