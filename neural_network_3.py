# Multiple neurons per layer 
# static amount of neurons layer and static amount of layers
# no use of chain rule 

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
        self.activation = 0
        self.derivative = None

    def activate(self):
        # sigmoid actvition
        self.activation = 1 / (1 + math.exp(self.value * -1))

class HiddenNeuron(Neuron):
    def forward(self, weights, prev_neurons):
        self.value = self.getWeightedSum(weights, prev_neurons) + self.bias
        self.activate()
    
    def getWeightedSum(self,weights,prev_neurons):
        value = 0
        for i,neuron in enumerate(prev_neurons):
            value += weights[i] * neuron.value # use value instead of activation here becuase input neurons do not have activation
        return value

class OutputNeuron(Neuron):
    def __init__(self, reference_value):
        super().__init__()
        self.reference = reference_value

    def forward(self, weights, prev_neurons):
        self.value = self.getWeightedSum(weights, prev_neurons) + self.bias

    
    def getWeightedSum(self,weights,prev_neurons):
        value = 0
        for i,neuron in enumerate(prev_neurons):
            value += weights[i] * neuron.activation
        return value

    def activate(self, output_neurons):
        # softmax activation
        sum_of_exp_values_of_previos_neurons = 0
        for neuron in output_neurons:
            sum_of_exp_values_of_previos_neurons += math.exp(neuron.value)

        self.activation = math.exp(self.value) / (sum_of_exp_values_of_previos_neurons)
        print("Output neuron with value: " + str(self.value))
        print("Output neuron with reference value: " + str(self.reference) + " got prob: " + str(self.activation))

class Layer():
    def __init__(self):
        self.neurons = []
        self.weights = []
    
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
    
    def backprop(self, previous_layer, desired_outputs):
        for i,neuron in enumerate(self.neurons):
            cost_derivative = 2 * (neuron.activation - desired_outputs[neuron.reference])
            sigmoid = 1 / (1 + math.exp(neuron.value * -1))
            activation_derivative = sigmoid * (1 - sigmoid)

            neuron.derivative = activation_derivative * cost_derivative

            cost_derivative_to_bias = 1
            change_of_bias = 0.8 * cost_derivative_to_bias * activation_derivative * cost_derivative
            neuron.bias = neuron.bias + change_of_bias

            for l,weight in enumerate(self.weights[i]):
                value_of_last_neuron = previous_layer.neurons[l].value
                cost_derivative_to_weight = value_of_last_neuron
                change_of_weight = 0.8 * cost_derivative_to_weight * activation_derivative * cost_derivative
                self.weights[i][l] = weight + change_of_weight   

    def forward(self,previous_layer):
        for i,next_neuron in enumerate(self.neurons):
            next_neuron.forward(self.weights[i], previous_layer.neurons)
        
        for i,next_neuron in enumerate(self.neurons):
            next_neuron.activate(self.neurons)
     

# todo bias auch mit in weights matrix 
class HiddenLayer(Layer):
    def __init__(self,number_neurons,number_input_weights):
        super().__init__()
        self.weights = numpy.random.ranf((number_neurons,number_input_weights))
        for l in range(0,number_neurons):
            self.neurons.append(HiddenNeuron()) 

    def backprop(self, previous_layer, next_layer):
        for i,neuron in enumerate(self.neurons):
            for next_neuron in next_layer.neurons:
                derivative_of_next_neuron = next_neuron.derivative
                sigmoid = 1 / (1 + math.exp(neuron.value * -1))
                activation_derivative = sigmoid * (1 - sigmoid) 

                neuron.derivative = activation_derivative * derivative_of_next_neuron

                derivative_of_neuron_wrt_bias = 1
                change_of_bias = 0.8 * derivative_of_neuron_wrt_bias * activation_derivative * derivative_of_next_neuron
                neuron.bias = neuron.bias + change_of_bias

                for l,weight in enumerate(self.weights[i]):
                    derivative_of_neuron_wrt_weight = previous_layer.neurons[l].value
                    change_of_weight = 0.8 * derivative_of_neuron_wrt_weight * activation_derivative * derivative_of_next_neuron
                    self.weights[i][l] += change_of_weight

    def forward(self,previous_layer):
        for i,next_neuron in enumerate(self.neurons):
            next_neuron.forward(self.weights[i], previous_layer.neurons)

class NeuralNetwork():
    def __init__(self):
        # todo stepsize, number neurons = numer input weights
        self.hidden_layer = HiddenLayer(5,2)
        self.hidden_layer2 = HiddenLayer(5,5)
        self.output_layer = OutputLayer(2,5,[0,1])

    def train(self):
        print("start training")
        # Iterate through train data
        errors = 0
        index = 0
        print(self.output_layer.weights)
        for i,item in enumerate(self.train_data):
            print("train image " + str(i) + " of " + str(len(self.train_data)))
            # Forwarding throug all three layers 
            
            # First Layer
            self.input_layer = InputLayer([item[0], item[1]])
            # Hidden Layer
            self.hidden_layer.forward(self.input_layer)
            self.hidden_layer2.forward(self.hidden_layer)

            # Output Layer
            self.output_layer.forward(self.hidden_layer2)

            # Check output value of each output neuron and adjust weights and biases by backpropagation
            prediction = None
            prediction_prob = 0
            desired_probs = {}
            # Set prediction as reference value from outputlayer with the highest activation
            for i,neuron in enumerate(self.output_layer.neurons):
                desired_output = item[2]
                if desired_output == neuron.reference:
                    desired_probs[neuron.reference] = 1
                else:
                    desired_probs[neuron.reference] = 0
                        
                if neuron.activation > prediction_prob:
                    prediction_prob = neuron.activation
                    prediction = neuron.reference
            
            if prediction != desired_output:
                errors += 1 
            index += 1
            print("Average wrong results: " + str(errors/index*20))
            
            self.output_layer.backprop(self.hidden_layer2,desired_probs)
            self.hidden_layer2.backprop(self.hidden_layer,self.output_layer)
            self.hidden_layer.backprop(self.input_layer,self.hidden_layer2)
        print(self.output_layer.weights)


    def cost(self, value, desired_output):
        # squared error
        return (value - desired_output) * (value - desired_output)
        
    def test(self):
        print("start testing")
        errors = 0
        for item in self.test_data:
            print("Predict: " + str(item[0]) + "," + str(item[1]) + " should be " + str(item[2]))
            prediction = self.predict(item[0], item[1])
            print("Predicted: " + str(prediction))
            if prediction is not item[2]:
                errors = errors + 1
        print(errors / len(self.test_data) * 20)

    def load_data(self, train_data, test_data):
        print("loading data")
        self.train_data = train_data
        self.test_data = test_data
        
    def predict(self,item1, item2):
        # Forwarding throug all three layers 
        # First Layer
        self.input_layer = InputLayer([item1, item2])

        # Hidden Layer
        self.hidden_layer.forward(self.input_layer)
        self.hidden_layer2.forward(self.hidden_layer)

        # Output Layer
        self.output_layer.forward(self.hidden_layer2)

        # Prediction results 
        prediction = None
        prediction_prob = 0
        for i,neuron in enumerate(self.output_layer.neurons):
            if neuron.activation > prediction_prob:
                prediction_prob = neuron.activation
                prediction = neuron.reference
        return prediction

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.load_data(train_data, test_data)
    neural_network.train()
    neural_network.test()

