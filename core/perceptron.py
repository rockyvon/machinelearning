from functools import reduce

class Perceptron(object):
    def __init__(self, count, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(count)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' %(self.weights, self.bias)

    def predict(self, values):
        samples = list(zip(values, self.weights))
        mutiList = list( map(lambda x: x[0] * x[1], samples))
        sums = reduce(lambda a, b : a + b, mutiList, 0.0) + self.bias
        return self.activator(sums )
        
    def train(self, valuesList, labels, iteration, rate):
        for _ in range(iteration):
            self.trainPerIteration(valuesList, labels, rate)

    def trainPerIteration(self, valuesList, labels, rate):
        samples = zip(valuesList, labels)

        for(values, label) in samples:
            output = self.predict(values)
            self.updateWeights(values, output, label, rate)

    def updateWeights(self, values, output, label, rate):
        delta = label - output
        self.weights = list(map(
            lambda x: x[1] + delta * rate * x[0],
            list(zip(values, self.weights))
        ))
        self.bias += delta * rate