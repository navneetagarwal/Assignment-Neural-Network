import math
import random
import numpy as np

# sigmoid transfer function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)


class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output, iterations, learningrate, constant, decay_rate):
        self.iterations = iterations
        self.learningrate = learningrate
        self.constant = constant
        self.decay_rate = decay_rate
        self.input = input + 1
        self.output = output
        self.ai = [1.0] * self.input
        self.ao = [1.0] * self.output
        input_range = 1.0 / self.input ** (1/2)
        self.wio = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.output))
        self.cio = np.zeros((self.input, self.output))
    def feedForwardAlgo(self, inputs):
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs')
        for i in range(self.input -1):
            self.ai[i] = inputs[i]
        for k in range(self.output):
            sum = 0.0
            for j in range(self.input):
                sum += self.ai[j] * self.wio[j][k]
            self.ao[k] = sigmoid(sum)
        return self.ao[:]
    def backPropagate(self, targets):
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        for i in range(self.input):
            for j in range(self.output):
                change = output_deltas[j] * self.ai[i]
                self.wio[i][j] -= self.learningrate * change + self.cio[i][j] * self.constant
                self.cio[i][j] = change
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error
    def test(self, patterns):
        out = []
        i = 0
        f = open("Final.csv",'w')
        f.write("id,label\n")
        for p in patterns:
            if ( self.feedForwardAlgo(p[0])[0] > 0.5):
                out.append(1.0)
                f.write(str(i)+','+str(1)+"\n")
            else:
                f.write(str(i)+','+str(0)+"\n")
                out.append(0.0)
            i = i+1
    def train(self, patterns):
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForwardAlgo(inputs)
                error += self.backPropagate(targets)
            self.learningrate = self.learningrate * (self.learningrate / (self.learningrate + (self.learningrate * self.decay_rate)))
    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(self.feedForwardAlgo(p))
        return predictions
def demo():
    def load_data_Train():
        data = np.loadtxt('Train.csv', delimiter = ',')
        y = data[:,-1]
        data = data[:,0:57]
        data -= data.min()
        data /= data.max()
        out = []
        for i in range(data.shape[0]):
            list_u = []
            list_u.append(y[i])
            temp = list((data[i,:].tolist(), list_u))
            out.append(temp)
        return out

    def load_data_Test():
        data = np.loadtxt('TestX.csv', delimiter = ',')
        data = data[:,0:57]
        data -= data.min()
        data /= data.max()
        out = []
        for i in range(data.shape[0]):
            list_u = []
            temp = list((data[i,:].tolist(), list_u))
            out.append(temp)
        return out
    X = load_data_Train()
    Y = load_data_Test()
    Neural_Network = MLP_NeuralNetwork(57, 10, 1, iterations = 100, learningrate = 0.4, constant = 0.5, decay_rate = 0.01)#50 0.3
    Neural_Network.train(X)
    Neural_Network.test(Y)
if __name__ == '__main__':
    demo()
