# MnistClassifier

## Installation
```sh
$ git clone https://github.com/peraza97/mnistClassifier.git
$ mkdir Weights; mkdir Results; mkdir Data
$ cd NeuralNet
$ pip install -r requirements.txt
```

## Usage

### Display first 2 Principal components
 run the following command
```sh
$ python PCA.py
```

### Training the Neural Network
run the following command
```sh
$ python cnn.py --train
```
Optional parameters:
* -s : save the weights after training the model. Will save to the Weights folder
* -p : show epoch loss plot after training completes

### Testing the Neural Network
run the following command
```sh
$ python cnn.py --test --dataset="datasetName"
```
Optional parameters: 
* -w : pass weights file(not including extension) in Weights folder. If you do not pass in this arguments it will look for default weights file 

#### Example
```sh
$ python cnn.py --test --dataset=test --weights=bestWeights
```

#### Notes
* Training the model expects a file called train.csv in Data folder
