# One Neuron per layer 

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
