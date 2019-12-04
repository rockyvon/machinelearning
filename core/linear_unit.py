from core.perceptron import Perceptron

f = lambda x : x

class LinearUnit(Perceptron):
    def __init__(self, count):
        super().__init__(count, f)