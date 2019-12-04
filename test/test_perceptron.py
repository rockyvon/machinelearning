from test.test_base import TestBase
from core.perceptron import Perceptron

f = lambda x : 1 if x > 0 else 0

class TestPerceptron(TestBase):

    def getSamples(self):
        valuesList = [
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0]
        ]
        labels = [
            0,
            1,
            0,
            0
        ]
        return valuesList, labels

    def test(self):
        weightCount = 2
        iteration = 10
        rate = 0.01
        p = Perceptron(weightCount, f)

        valuesList, labels = self.getSamples()

        p.train(valuesList, labels, iteration, rate)

        print('the perceptron is:\n' + str(p))
        print('input:[1,1],output:' + str(p.predict([1,1])))
        print('input:[0,0],output:' + str(p.predict([0,0])))
        print('input:[0,1],output:' + str(p.predict([0,1])))
        print('input:[1,0],output:' + str(p.predict([1,0])))