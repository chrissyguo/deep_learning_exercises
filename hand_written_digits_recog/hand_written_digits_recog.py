#!/usr/bin/env python
# coding: utf-8

# In[28]:


# ENVIRONMENT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms


from torch.autograd import Variable


# In[91]:


# PREPARING THE DATASET
## Hyper parameters

n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enable = False
torch.manual_seed(random_seed)


# In[92]:


## Load MNIST dataset in a handy way

train_loader = torch.utils.data.DataLoader(
    dsets.MNIST('/home/bdggj/Documents/Deep_learning_exercise/hand_written_digits_recog/files/', train=True, download=True, 
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                             ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dsets.MNIST('/home/bdggj/Documents/Deep_learning_exercise/hand_written_digits_recog/files/', train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                             ])),
    batch_size=batch_size_test, shuffle=True)


# In[93]:


## examples

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# In[94]:


example_data.shape


# In[95]:


## show some pics

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
    
fig


# In[96]:


# BUILDING THE NETWORK

# 3-D convolutional layers followed by two fully-connected layers

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  #flat the features from (20, 4, 4) to (1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    


# In[97]:


network = Net()
network.cuda() # use a GPU ofr training
print(network)

optimizer = optim.SGD(network.parameters(), lr=learning_rate, 
                     momentum=momentum)


# In[98]:


# TRAINING THE MODEL

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


# In[99]:


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)  #calculate the negative log likeligood loss
        loss.backward()
        optimizer.step() #renew the parameters
        if batch_idx % log_interval == 0:
             print ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
             epoch, batch_idx * len(data), len(train_loader.dataset),
             100. *batch_idx / len(train_loader), loss.item()))
        
             train_losses.append(loss.item())
             train_counter.append(
                 (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
             torch.save(network.state_dict(), 
                        '/home/bdggj/Documents/Deep_learning_exercise/hand_written_digits_recog/results/model.pth')
             torch.save(optimizer.state_dict(), 
                        '/home/bdggj/Documents/Deep_learning_exercise/hand_written_digits_recog/results/optimizer.pth')
        


# In[100]:


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.cuda()
            target=target.cuda()
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1].cuda()
            correct += pred.eq(target.data.view_as(pred)).sum()
        
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
        


# In[101]:


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()


# In[ ]:




