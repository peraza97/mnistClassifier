import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Dataset import MnistDataset, SplitDataSet, viewImage
from matplotlib import pyplot as plt
import argparse
import csv

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
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3) 
        self.fc1 = nn.Linear(128*5*5,125)
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
        input = input.view(-1,128*5*5)
        #first linear layer
        input = F.dropout(self.fc1(input), .35)    
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
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('-p', '--showPlot', action='store_true', help="plot metrics after training model")
    parser.add_argument('-s', '--saveModel', action='store_true', help="save the weights")   
    parser.add_argument('-w', '--weights', help="load weights file in Weights foler")  
    parser.add_argument('-d', '--dataset', help="name of dataset that will be loaded. Must be located in ./Data folder")
    args = parser.parse_args()

    if args.train:
        #define training hyper parameters
        learningRate = .001
        epoch = 50
        batchSize = 250
        split = .2

        model = Net().to(device)
        #loss/ optimization
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=learningRate)

        #load datasets
        dataset = MnistDataset('train')
        trainDataset, testDataset = SplitDataSet(dataset,split)
        trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

        losses = []
        #epoch loop
        for e in range(1, epoch + 1):
            runningLoss = train(model, trainDataLoader, optimizer, lossFunction)
            losses.append(runningLoss)
            if e % 10 == 9:
                print("Epoch {0} - loss: {1:4f}".format(e + 1, runningLoss))

        accuracy = test(model, testDataLoader)
        print("Accuracy: {0:4f}".format(accuracy))

        if args.showPlot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(np.array(losses), 'r')
            ax1.set(xlabel='Epoch')
            ax1.set_title('Loss')
            plt.show()
        
        if args.saveModel:
            weightsPath = "../Weights/weights{0:.0f}.pth".format(int(accuracy*100))
            torch.save(model.state_dict(),weightsPath)


    elif args.test:
        weightsPath = "../Weights/{0}.pth".format(args.weights) if args.weights is not None else '../Weights/weights.pth'
        resultsPath = "../Results/submission.csv"
        #load the model
        model = Net().to(device)
        model.load_state_dict(torch.load(weightsPath))
        model.eval()
        #load dataset to evaluate
        dataset = MnistDataset(args.dataset)

        predictions = []

        #get all predictions
        for (data, _) in dataset:
            img = data.view(-1,1,28,28).to(device)
            output = model(img)
            predictions.append(torch.argmax(output).item())
        #save predictions to csv file

        with open(resultsPath, mode='w',newline='') as file :
            writer = csv.writer(file,delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['ImageId','Label'])
            for i, label in enumerate(predictions):
                writer.writerow([i+1,label])



if __name__ == '__main__':
    main()