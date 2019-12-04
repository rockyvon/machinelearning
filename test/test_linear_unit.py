from test.test_base import TestBase
from core.linear_unit import LinearUnit

class TestLinearUnit(TestBase):
    def getSamples(self):
        '''
        数据为5个人的收入数据和工作年限
        '''
        #工作年限
        valuesList = [[5], [3], [8], [1.4], [10.1]]
        #收入
        labels = [5500, 2300, 7600, 1800, 11400]
        return valuesList, labels

    def test(self):
        weightCount = 1
        iteration = 10
        rate = 0.1

        lu = LinearUnit(weightCount)
        #训练模型
        valuesList, labels = self.getSamples()
        lu.train(valuesList, labels, iteration, rate)
        #预测并输出结果
        print('the LinearUnit is:\n' + str(lu))
        print('input:[3.4],output:' + str(lu.predict([3.4])))
        print('input:[15],output:' + str(lu.predict([15])))
        print('input:[1.5],output:' + str(lu.predict([1.5])))
        print('input:[6.3],output:' + str(lu.predict([6.3])))