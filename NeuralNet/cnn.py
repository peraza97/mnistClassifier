import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Dataset import MnistDataset, viewImage
from matplotlib import pyplot as plt
import argparse

def get_default_device():
    if torch.cuda.is_available():
        print("GPU accelerated")
        return torch.device('cuda')
    else:
        print("CPU garbage")
        return torch.device('cpu')

device = get_default_device()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Linear(input output)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3) 
        self.fc1 = nn.Linear(64*5*5,125)
        self.fc2 = nn.Linear(125,10)

    def forward(self, input):
        #first conv layer
        input = self.conv1(input)
        input = F.relu(input)
        input = F.max_pool2d(input,kernel_size=2,stride=2)
        #second conv layer
        input = self.conv2(input)
        input = F.relu(input)
        input = F.max_pool2d(input,kernel_size=2,stride=2) 
        #reshape input
        input = input.view(-1,64*5*5)
        #first linear layer
        input = self.fc1(input)      
        input = F.relu(input)
        #second linear layer
        output = self.fc2(input)

        return output

def train(model, dataLoader, optimizer, lossFunction):
    model.train()
    runningLoss = 0.0
    #training loop
    for i, data in enumerate(dataLoader):
        #get the batch + labels
        batch,labels = data
        batch, labels = batch.view(-1,1,28,28).to(device), labels.view(-1).to(device)
        #zero out the gadients
        optimizer.zero_grad()
        #forward + optimization
        output = model(batch)
        loss = lossFunction(output, labels.long())
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    epochLoss = runningLoss / len(dataLoader)
    return epochLoss

def test(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    #validate the model on validationSet
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            batch,labels = data
            batch = batch.view(-1,1,28,28).to(device)
            labels = labels.to(device)
            output = model(batch)
            outputMaxes = [torch.argmax(i) for i in output]
            actualMaxes = [torch.max(i) for i in labels]
            for i,j in zip(outputMaxes,actualMaxes):
                print(i, j)
                if i == j:
                    correct += 1
                total += 1
    try:
        accuracy = float(correct)/total
        return accuracy
    except ZeroDivisionError:
        return 0

def main():

    parser = argparse.ArgumentParser(description='Train or evaluate a dataset')
    parser.add_argument('--train', action='store_true', help="training mode")
    parser.add_argument('--eval', action='store_true', help="evaluation mode")
    parser.add_argument('-w','--weightsPath', help="path that will save/load weights")
    parser.add_argument('-d', '--dataset',required=True,help="name of dataset that will be loaded. Must be located in ./Data folder")
    args = parser.parse_args()

    if args.train:
        #define training hyper parameters
        learningRate = .001
        epoch = 50
        batchSize = 2

        model = Net().to(device)
        #loss/ optimization
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=learningRate)

        #load datasets
        trainDataset = MnistDataset('dummy')
        #testDataset = MnistDataset('test')
        trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        #testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

        for e in range(1, epoch + 1):
            runningLoss = train(model, trainDataLoader, optimizer, lossFunction)
            if e % 10 == 9:
                print("Epoch {0} - loss: {1:4f}".format(e, runningLoss))


    elif args.eval:
        pass

if __name__ == '__main__':
    main()