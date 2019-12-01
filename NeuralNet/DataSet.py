import csv
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = '../Data/'

def showNumber(image):
    #assume image is numpy array
    image = image.reshape(28,28)
    plt.imshow(image)
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
    data = rawData[1:,1:].astype(float)

    #do we want to pikle this data
    if save:
        filename = Path(fileName)
        pickleFileName = filename.with_suffix('.pkl')
        filehandler = open(pickleFileName, 'wb') 
        pickle.dump({"data":data, "labels":labels}, filehandler)

    return labels, data

name = 'test'

labels = None
data = None
if os.path.isfile(DATA_PATH + name + '.pkl'):
    labels, data = loadPickle(DATA_PATH + name + '.pkl')
    print("from pickle")
else:
    labels, data = loadCSV(DATA_PATH + name + '.csv',True)
    print("from csv")

for i, image in enumerate(data):
    if i > 2:
        quit()
    print('label: {0}'.format(labels[i].item()))
    showNumber(image)
