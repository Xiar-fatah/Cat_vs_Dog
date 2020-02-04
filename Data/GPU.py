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
        
        self.fc1 = nn.Linear(179776, 1000)
        self.fc2 = nn.Linear(1000, 2)


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
        print("Drop_out")
        print(t.shape)
        t = self.fc1(t)
#        print("fc1")
#        print(t.shape)
        t = self.fc2(t)
#        print("fc2")
#        print(t.shape)
        return t



if __name__ == "__main__":
    ####################### Prepossessing Data #######################
    train_dir = './dataset'
    test_dir = './testset'
    mean = [0.485, 0.456, 0.406] 
    std  = [0.229, 0.224, 0.225]
    transforms= transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
    #        transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset , batch_size=100,
                                              shuffle=True, num_workers=4)
    
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms)
    
    testloader = torch.utils.data.DataLoader(train_dataset , batch_size=100,
                                              shuffle=False, num_workers=4)
    
    model = CNN()
    model.cuda(device)
    learning_rate = 0.01
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(trainloader)
    num_epochs = 2
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)  # missing line from original code
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
    
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
    
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
                
    images = np.linspace(0,len(loss_list)-1,len(loss_list))
    plt.figure()  
    plt.plot(images, acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(("Accuracy"))
    plt.figure()
    plt.plot(images, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(("Loss"))
    
#    model.eval()
#    with torch.no_grad():
#        correct = 0
#        total = 0
#        for images, labels in testloader:
#            outputs = model(images)
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
#
#
#     










