from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch
from torch import optim
from torch import nn
import numpy as np
import tensorflow as tf
import wandb

_tasks = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    
mnist = MNIST("data", download=True, train=True, transform=_tasks)

split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]
## create sampler objects using SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)
## create iterator objects for train and valid datasets
trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)
  
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x

model = Model()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                      lr=0.01, 
                      weight_decay= 1e-6, 
                      momentum = 0.9, 
                      nesterov = True)

for epoch in range(1, 21): 
    train_loss, valid_loss = [], []
    model.train()

    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)# 1. forward propagation
        loss = loss_function(output, target)#2. loss calculation
        loss.backward()#3. backward propagation-计算损失函数对各权重的偏导（即斜率）
        optimizer.step()#4. weight optimization-利用3算出来的结果进行权重更新
        
        train_loss.append(loss.item())
        
    model.eval()
    for data, target in validloader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))