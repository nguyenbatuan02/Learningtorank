import math
import random
import copy
import time
import DataProcess
import matplotlib.pyplot as plt

LearningRate = 0.001
HiddenLayer = 20
Iteration = 10


def plotErrorRate(errorRate):

    plt.plot(errorRate)
    plt.ylabel('Error Rate')
    plt.show()


# The transfer function of neurons, g(x)
def logFunc(x):
    # print(f'x = {x}')
    return 1 / (1 + math.exp(-x))


# The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return logFunc(x) * (1 - logFunc(x))


def random_float(low, high):
    return random.random() * (high - low) + low


# Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0] * J)
    return m


class NN:  # Neural Network
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        # Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden
        self.numOutput = 1

        # Current activation levels for nodes (in other words, the nodes output value)
        self.activations_input = [1.0] * self.numInputs
        self.activations_hidden = [1.0] * self.numHidden
        self.activation_output = 1.0
        self.learning_rate = learningRate

        # create weights
        # A matrix with all weights from input layer to hidden layer
        self.weights_input = makeMatrix(self.numInputs, self.numHidden)
        # A list with all weights from hidden layer to the single output neuron.
        # self.weights_output = [0 for i in range(self.numHidden)]
        self.weights_output = makeMatrix(self.numHidden, self.numOutput)
        # set them to random values
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weights_input[i][j] = random_float(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weights_output[j] = random_float(-0.5, 0.5)

        # Data for the backpropagation step in RankNets.
        # For storing the previous activation levels of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0
        # For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = [0 for i in range(self.numHidden)]
        # For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = [0 for i in range(self.numHidden)]

    def propagate(self, inputs):
        # print('Propagating input...')
        if len(inputs) != self.numInputs - 1:
            # print(self.numInputs-1)
            raise ValueError('wrong number of inputs')

        # input activations
        self.prevInputActivations = copy.deepcopy(self.activations_input)
        for i in range(self.numInputs - 1):
            self.activations_input[i] = inputs[i]
        self.activations_input[-1] = 1  # Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations = copy.deepcopy(self.activations_hidden)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                # print self.ai[i] ," * " , self.wi[i][j]
                sum = sum + self.activations_input[i] * self.weights_input[i][j]
            self.activations_hidden[j] = logFunc(sum)

        # output activations
        self.prevOutputActivation = self.activation_output
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.activations_hidden[j] * self.weights_output[j]
        self.activation_output = logFunc(sum)
        return self.activation_output

    def computeOutputDelta(self):
        # Updating the delta in the output layer

        Pab = 1 / (1 + math.exp(-(self.prevOutputActivation - self.activation_output)))
        self.prevDeltaOutput = logFuncDerivative(self.prevOutputActivation) * (1.0 - Pab)
        self.deltaOutput = logFuncDerivative(self.activation_output) * (1.0 - Pab)

    def computeHiddenDelta(self):
        # Updating the delta values in the hidden layer

        # Update delta_{A}
        for i in range(self.numHidden):
            self.prevDeltaHidden[i] = logFuncDerivative(self.prevHiddenActivations[i]) * self.weights_output[i] * (
                    self.prevDeltaOutput - self.deltaOutput)
        # Update delta_{B}
        for j in range(self.numHidden):
            self.deltaHidden[j] = logFuncDerivative(self.activations_hidden[j]) * self.weights_output[j] * (
                    self.prevDeltaOutput - self.deltaOutput)

    def updateWeights(self):
        # Update the weights of the NN

        # Update weights going from the input layer to the output layer
        # Each input node is connected with all nodes in the hidden layer
        for j in range(self.numHidden):
            for i in range(self.numInputs):
                self.weights_input[i][j] = self.weights_input[i][j] + self.learning_rate * (
                        self.prevDeltaHidden[j] * self.prevInputActivations[i] - self.deltaHidden[j] *
                        self.activations_input[i])

        # Update weights going from the hidden layer (i) to the output layer (j)
        for i in range(self.numHidden):
            self.weights_output[i] = self.weights_output[i] + self.learning_rate * (
                    self.prevDeltaOutput * self.prevHiddenActivations[i] - self.deltaOutput *
                    self.activations_hidden[i])

    # Removed target value
    def backPropagate(self):
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    def weights(self):
        print('Input weights:')
        for i in range(self.numInputs):
            print(self.weights_input[i])
        print()
        print('Output weights:')
        print(self.weights_output)

    def train(self, X_train, pairs, iterations):

        errorRate = []
        start = time.time()

        print('Training the neural network...')
        for epoch in range(iterations):
            print('Epoch %d' % (epoch + 1))
            for pair in pairs:
                self.propagate(X_train[pair[0]])
                self.propagate(X_train[pair[1]])
                self.backPropagate()
            errorRate.append(self.countDisorderedPairs(X_train, pairs))
            print('Error rate: %.2f' % errorRate[epoch])
            # self.weights()
        m, s = divmod(time.time() - start, 60)
        print('Training took %dm %.1fs' % (m, s))
        plotErrorRate(errorRate)

    def countDisorderedPairs(self, X_train, pairs):
        """
        errorRate = numWrong/(Total)
        """

        DisorderedPairs = 0

        for pair in pairs:
            self.propagate(X_train[pair[0]])
            self.propagate(X_train[pair[1]])
            if self.prevOutputActivation <= self.activation_output:
                DisorderedPairs += 1

        return DisorderedPairs / float(len(pairs))


if __name__ == '__main__':
    # Read training data
    X_train, Y_train, Query = DataProcess.readDataset(r'C:\ProjectI\Data\MSLR-WEB10K\Fold1\train.txt')
    # Extract document pairs
    pairs = DataProcess.extractPairsOfRatedSites(Y_train, Query)
    # Initialize Neural Network
    rankNet = NN(136, HiddenLayer, LearningRate)
    # Train the Neural Network
    rankNet.train(X_train, pairs, Iteration)
    # Read test set
    X_train, Y_train, Query = DataProcess.readDataset(r'C:\ProjectI\Data\MSLR-WEB10K\Fold1\test.txt')
    # Extract document pairs
    pairs = DataProcess.extractPairsOfRatedSites(Y_train, Query)
    print('Test set errorRate: ' + str(rankNet.countDisorderedPairs(X_train, pairs)))
