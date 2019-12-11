from Dataset import loadDataset
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def PCAanalysis(k):
    colors = ['b', 'g', '#1a3335', 'c', 'm', 'y', 'k', 'y', '#911FF1', 'r']
    shapes = ['.', '8', 'o', 'X', '+', '<', '>', '^', 'h', '*']

    data, labels = loadDataset('train')
    pca = PCA(n_components=2)
    pca.fit(data)
    dataPca = pca.transform(data)

    for num in range(0,10):
        idxs = [i for i, label in enumerate(labels) if num == label] 
        plt.scatter(dataPca[idxs, 0], dataPca[idxs,1], label=num, alpha=.7, marker=shapes[num], color=colors[num])
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Component 2")       
    plt.legend()
    plt.show()

def main():
    PCAanalysis(2)

if __name__ == '__main__':
    main()