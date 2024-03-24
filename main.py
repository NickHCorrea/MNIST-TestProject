import time

import numpy as np
import matplotlib.pyplot

train_x = "Dataset/train-images.idx3-ubyte"
train_y = "Dataset/train-labels.idx1-ubyte"

test_x = "Dataset/t10k-images.idx3-ubyte"
test_y = "Dataset/t10k-labels.idx1-ubyte"


def convertToCSV(imgs, labels, outFile, n):
    imgs_f = open(imgs, "rb")
    label_f = open(labels, 'rb')
    csv_f = open(outFile, 'w')

    imgs_f.read(16)
    label_f.read(8)
    images = []

    for i in range(n):
        image = [ord(label_f.read(1))]
        for j in range(28 * 28):
            image.append(ord(imgs_f.read(1)))
        images.append(image)

    for image in images:
        csv_f.write(",".join(str(pix) for pix in image) + "\n")

    imgs_f.close()
    label_f.close()
    csv_f.close()


def convertTrainData(outputFile, n):
    convertToCSV(train_x, train_y, outputFile, n)


def convertTestData(outputFile, n):
    convertToCSV(test_x, test_y, outputFile, n)


def readFile(inputFile):
    f = open(inputFile, 'r')
    resultList = f.readlines()
    f.close()
    return resultList


class DNN:
    def __init__(self, sizes, epochs, learningRate):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = learningRate

        inputLayer = sizes[0]
        hidden1Layer = sizes[1]
        hidden2Layer = sizes[2]
        outputLayer = sizes[3]

        self.params = {
            'W1': np.random.randn(hidden1Layer, inputLayer) * np.sqrt(1. / hidden1Layer),
            'W2': np.random.randn(hidden2Layer, hidden1Layer) * np.sqrt(1. / hidden2Layer),
            'W3': np.random.randn(outputLayer, hidden2Layer) * np.sqrt(1. / outputLayer)
        }

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forwardPass(self, trainImage):
        params = self.params
        params['A0'] = trainImage  # Activation function for the input layer

        # input layer to first hidden layer

        params['Z1'] = np.dot(params['W1'], params['A0'])  # Input activation x weights between 1Hidden and Input
        params['A1'] = self.sigmoid(params['Z1'])  # Activation function for the first hidden layer

        params['Z2'] = np.dot(params['W2'], params['A1'])  # 1Hidden activation x weights between 2Hidden and 1Hidden
        params['A2'] = self.sigmoid(params['Z2'])  # Activation function for the second hidden layer

        params['Z3'] = np.dot(params['W3'], params['A2'])  # 2Hidden activation x weights between Output and 2Hidden
        params['A3'] = self.softmax(params['Z3'])

        return params['Z3']

    def backwardPass(self, trainLabels, output):
        params = self.params
        changeW = {}
        #  Calculate change in weights

        error = 2 * (output - trainLabels) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        changeW['W3'] = np.outer(error, params['A2'])

        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        changeW['W2'] = np.outer(error, params['A1'])

        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        changeW['W1'] = np.outer(error, params['A0'])

        return changeW

    def updateWeights(self, changeDict):
        for key, val in changeDict.items():
            self.params[key] -= self.lr * val

    def computeAccuracy(self, testImage):
        predictions = []
        for x in testImage:
            values = x.split(",")
            inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.forwardPass(inputs)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(targets))
        return np.mean(predictions)

    def train(self, trainList, testList):
        startTime = time.time()
        for i in range(self.epochs):
            for x in trainList:
                values = x.split(",")
                inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99

                output = self.forwardPass(inputs)
                changeW = self.backwardPass(targets, output)
                self.updateWeights(changeW)
            accuracy = self.computeAccuracy(testList)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                i + 1, time.time() - startTime, accuracy * 100))


def main():
    nTrain = 60000
    nTest = 10000
    trainCSVFile = "Dataset/train.csv"
    testCSVFile = "Dataset/test.csv"
    convertTrainData(outputFile=trainCSVFile, n=nTrain)
    convertTestData(outputFile=testCSVFile, n=nTest)
    trainList = readFile(inputFile=trainCSVFile)
    testList = readFile(inputFile=testCSVFile)
    dnn = DNN(sizes=[784, 128, 64, 10], epochs=10, learningRate=1)
    dnn.train(trainList, testList)


main()
