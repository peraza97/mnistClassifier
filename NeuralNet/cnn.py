import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from Dataset import MnistDataset
from matplotlib import pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a dataset')
    parser.add_argument('--train', action='store_true', help="training mode")
    parser.add_argument('--eval', action='store_true', help="evaluation mode")
    parser.add_argument('-w','--weightsPath', help="path that will save/load weights")
    parser.add_argument('-d', '--dataset',required=True,help="name of dataset that will be loaded. Must be located in ./Data folder")
    args = parser.parse_args()

    trainDataset = MnistDataset(args.dataset)
    i = 0
    for data,label in trainDataset:
        plt.imshow(data.view(28,28))
        plt.show()
        i += 1
        if i > 3:
            quit()

if __name__ == '__main__':
    main()