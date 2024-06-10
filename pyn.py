import pickle
import numpy as np

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.output = []
        self.dataset = []
        self.gradientNN = []
        self.nn = []

    def load_dataset(self, dataset):
        self.dataset = dataset

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __cost(self, data):
        self.prediction(data[0])
        cost = np.sum((self.output - data[1]) ** 2)
        return cost

    def __update_values(self, learnRate):
        for index, (weights, biases) in enumerate(self.nn):
            weights -= self.gradientNN[index][0] * learnRate
            biases -= self.gradientNN[index][1] * learnRate

    def train(self, epochs, learnRate, printdata=False):
        for i in range(len(self.layers) - 1):
            self.gradientNN.append([
                np.zeros((self.layers[i], self.layers[i + 1])), 
                np.zeros(self.layers[i + 1])
            ])
        for epoch in range(epochs):
            for j in self.dataset:
                oldCost = self.__cost(j)
                for i in range(len(self.nn)):
                    h = 0.0001
                    for y in range(len(self.nn[i][0])):
                        for x in range(len(self.nn[i][0][y])):
                            self.nn[i][0][y][x] += h
                            difference = self.__cost(j) - oldCost
                            self.nn[i][0][y][x] -= h
                            self.gradientNN[i][0][y][x] = difference / h
                        
                    for y in range(len(self.nn[i][1])):
                        self.nn[i][1][y] += h
                        difference = self.__cost(j) - oldCost
                        self.nn[i][1][y] -= h
                        self.gradientNN[i][1][y] = difference / h
                
                self.__update_values(learnRate)
            if(printdata):
                print(f"Epoch: {epoch}")
                print(f"oldCost; {oldCost}")
                print(f"newCost; {self.__cost(j)}")
    def prediction(self, inputs):
        activations = np.array(inputs)
        for weights, biases in self.nn:
            activations = self.__sigmoid(np.dot(activations, weights) + biases)
        self.output = activations

    def randomize_values(self):
        for i in range(len(self.layers) - 1):
            self.nn.append([
                np.random.randn(self.layers[i], self.layers[i + 1]), 
                np.random.randn(self.layers[i + 1])
            ])

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.nn, f)

    def load(self, name):
        with open(name, 'rb') as f:
            self.nn = pickle.load(f)

    def print_output(self):
        print(self.output)

    def get_output(self):
        return self.output
