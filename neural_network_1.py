import random 
import numpy 
import math 

class Neuron():
    def __init__(self, value=None):
        if value:
            self.value = value
        else:
            self.value = random.random()
        self.bias = random.random() 
        self.active = True

    def activate(self):
        # sigmoid actvition
        activation = 1 / (1 + math.exp(self.value * -1))
        self.active = activation > 0

class HiddenNeuron(Neuron):
    def forward(self, weights, prev_neurons):
        self.value = self.getWeightedSum(weights, prev_neurons) + self.bias
        self.activate()
    
    def getWeightedSum(self,weights,prev_neurons):
        value = 0
        for i,neuron in enumerate(prev_neurons):
            if neuron.active:
                value += weights[i] * neuron.value
        return value

class OutputNeuron(Neuron):
    # Softmax neuron to get propability if its belongs to specific label
    def __init__(self, reference_value):
        super().__init__()
        self.reference = reference_value

class Layer():
    def __init__(self):
        self.neurons = []
        self.weights = []
    
    def forward(self,previous_layer):
        for i,next_neuron in enumerate(self.neurons):
            next_neuron.forward(self.weights[i], previous_layer.neurons)

class InputLayer(Layer):
    def __init__(self, values):
        super().__init__()
        for value in values:
            self.neurons.append(Neuron(value))

class OutputLayer(Layer):
    def __init__(self,number_neurons,number_input_weights,values):
        super().__init__()
        self.weights = numpy.random.ranf((number_neurons,number_input_weights))
        for l in range(0,number_neurons):
            self.neurons.append(OutputNeuron(values[l])) 
    
    def backprop(self, value, desired_value):
        for i,neuron in enumerate(self.neurons):
            cost_derivative = 2 * (value - desired_value)

            cost_derivative_to_bias = 1
            change_of_bias = cost_derivative_to_bias * 1
            change_of_bias = 2 * change_of_bias
            neuron.bias = neuron.bias + change_of_bias

            cost_derivative_to_weight = value_of_last_neuron
            change_of_weight = cost_derivative_to_weight * 22
            change_of_weight = 2 * change_of_weight
            self.weights[i] = self.weights[i] + change_of_weight

class HiddenLayer(Layer):
    def __init__(self,number_neurons,number_input_weights):
        super().__init__()
        self.weights = numpy.random.ranf((number_neurons,number_input_weights))
        for l in range(0,number_neurons):
            self.neurons.append(HiddenNeuron()) 

    def backprop(self, next_layer):
        

class NeuralNetwork():
    def __init__(self):
        self.hidden_layer = HiddenLayer(20,1)
        self.output_layer = OutputLayer(10,20,[0,1,2,3,4,5,6,7,8,9])

    def train(self):
        print("start training")
        # Iterate through train data
        for i,item in enumerate(self.train_data):
            print("train image " + str(i) + " of " + str(len(self.train_data)))
            # Forwarding throug all three layers 
            
            # First Layer
            self.input_layer = InputLayer([item[0]])
            # Hidden Layer
            self.hidden_layer.forward(self.input_layer)
            # Output Layer
            self.output_layer.forward(self.hidden_layer)

            # Check output value of each output neuron and adjust weights and biases by backpropagation
            complete_cost = 0
            for i,neuron in enumerate(self.output_layer.neurons):
                desired_output = item[1]
                desired_prob = 0
                if desired_output == neuron.reference:
                    desired_prob = 1
                else:
                    desired_prob = 0

                neuron_cost = self.cost(neuron.value, desired_prob)
                complete_cost += neuron_cost
            print("Complete Cost: " + complete_cost)
            print("Backpropagate to minimize Cost")
            self.output_layer.backprop()
            self.hidden_layer.backprop()

    def cost(self, value, desired_output):
        return (value - desired_output) * (value - desired_output)
        
    def test(self):
        print("start testing")
        errors = 0
        for item in self.test_data:
            prediction = self.predict(item[0])
            if prediction is not item[1]:
                errors = errors + 1
        print(errors / len(self.test_data) * 100)

    def load_data(self):
        print("loading data")
        self.train_data = [
            [1,1],
            [2,1],
            [4,1],
            [5,1],
            [1,1],
            [2,1],
            [4,1],
            [5,1],
            [6,0],
            [7,0],
            [9,0],
            [6,0],
            [7,0],
            [9,0],
            [6,0],
            [7,0],
            [1,1],
            [2,1],
            [4,1],
            [5,1],
            [1,1],
            [2,1],
            [4,1],
            [5,1],
            [6,0],
            [7,0],
            [9,0],
            [6,0],
            [7,0],
            [9,0],
            [6,0],
            [7,0]
        ]
        self.test_data = [[3,1], [8,0], [20,0], [500,0]]

    def predict(self,item):
        print("start prediction")
        # Forwarding throug all three layers 
        # First Layer
        self.input_layer = InputLayer([item])

        # Hidden Layer
        self.hidden_layer.forward(self.input_layer)
        
        # Output Layer
        self.output_layer.forward(self.hidden_layer)

        # Prediction results 
        prediction = None
        prediction_prob = 0
        for i,neuron in enumerate(self.output_layer.neurons):
            print(str(neuron.value) + "->" + str(neuron.value))
            if neuron.value > prediction_prob:
                prediction_prob = neuron.value
                prediction = neuron.reference
        return prediction

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.load_data()
    neural_network.train()
    neural_network.test()

