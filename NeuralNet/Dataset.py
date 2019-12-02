import csv
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from pathlib import Path

DATA_PATH = '../Data/'

def viewImage(img):
        plt.imshow(img.view(28,28))
        plt.show()

def loadPickle(fileName):
    with open(fileName, 'rb') as f:
        pickleObj = pickle.load(f)
        return pickleObj["labels"], pickleObj["data"]

def loadCSV(fileName,save):
    #generate numpy array from csv file
    rawData = np.genfromtxt(fileName, dtype= 'unicode', delimiter=',')
    #features is the first row
    features = rawData[0,:]
    #get index of label column(just in case not in first column)
    labelIdx = np.where(features == 'label')
    labels = rawData[1:,labelIdx].astype(float)
    #turn the data into floats
    data = rawData[1:,:].astype(float)
    #delete the column that corresponds to the label(if exists)
    data = np.delete(data,labelIdx,1)
    #do we want to pikle this data
    if save:
        filename = Path(fileName)
        pickleFileName = filename.with_suffix('.pkl')
        filehandler = open(pickleFileName, 'wb') 
        pickle.dump({"data":data, "labels":labels}, filehandler)

    return labels, data

class MnistDataset(Dataset):
    def __init__(self, dataFile): 
        self.data = None
        self.labels = None
        #if pkl file exists for the dataset, saves like 30 seconds
        if os.path.isfile(DATA_PATH + dataFile + '.pkl'):
            self.labels, self.data = loadPickle(DATA_PATH + dataFile + '.pkl')
            print("loaded {0} dataset from .pkl file".format(dataFile))
        #no pkl file exists, so load the csv and create a pkl file
        else:
            self.labels, self.data = loadCSV(DATA_PATH + dataFile + '.csv', False)
            print("loaded {0} dataset from .csv file".format(dataFile))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sample = self.data[idx,:]
        label = self.labels[idx]
        return torch.Tensor(sample), torch.Tensor(label).view(-1,1)