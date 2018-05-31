import unittest
import neural_network_3
import math 

class HidenNeuronTestCase(unittest.TestCase):
    def test_value(self):
        weights = [1]
        prev_neuron = [neural_network_3.Neuron(1)]
        neuron = neural_network_3.HiddenNeuron()
        weighted_sum = neuron.getWeightedSum(weights, prev_neuron)
        self.assertTrue(weighted_sum == 1)
   
    def test_activation(self):
        neuron = neural_network_3.HiddenNeuron()
        neuron.value = 1
        neuron.activate()
        self.assertTrue(math.ceil(neuron.activation*100)/100 == 0.74)

    def test_forward(self):
        weights = [1]
        prev_neuron = [neural_network_3.Neuron(2)]
        neuron = neural_network_3.HiddenNeuron()
        neuron.bias = 0
        neuron.forward(weights,prev_neuron)
        self.assertTrue(math.ceil(neuron.activation*100)/100 == 0.89)

"""
class InputLayerTestCase(unittest.TestCase):
    def test_one_neuron(self):
        layer = neural_network_3.InputLayer(1)
        layer.forward()
        layer.backprop()

        weights = [[]]
        self.assertTrue(layer.weights == weights)

    def test_multiple_neuron(self):
        layer = neural_network_3.InputLayer(1,2)
        layer.forward()
        layer.backprop()

        weights = [[]]
        self.assertTrue(layer.weights == weights)

class HiddenLayerTestCase(unittest.TestCase):
    def test_one_neuron(self):
        layer = neural_network_3.HiddenLayer(1,1)
        layer.forward()
        layer.backprop()

        weights = [[]]
        self.assertTrue(layer.weights == weights)

    def test_multiple_neuron(self):
        layer = neural_network_3.HiddenLayer(5,1)
        layer.forward()
        layer.backprop()

        weights = [[]]
        self.assertTrue(layer.weights == weights)

class OuputLayerTestCase(unittest.TestCase):
    def test_one_neuron(self):
        layer = neural_network_3.OutputLayer(1,1,1)
        layer.forward()
        layer.backprop()

        weights = [[]]
        self.assertTrue(layer.weights == weights)

    def test_multiple_neuron(self):
        layer = neural_network_3.OutputLayer(5,5)
        layer.forward()
        layer.backprop()

        weights = [[]]
        self.assertTrue(layer.weights == weights)

class NetworkTestCase(unittest.TestCase):
    def test_one_hidden_layer(self):
        pass

    def test_multiple_hidden_layer(self):
        pass
"""
if __name__ == '__main__':
    unittest.main()

