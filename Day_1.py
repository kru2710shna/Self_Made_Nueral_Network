import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
np.random.seed(0)
# X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

# Threshold example
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []
for i in inputs:
    outputs.append(i if i > 0 else 0)

# Hidden Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# RELU Function
class Activation_Function:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

if __name__ == "__main__":
    layer1 = Layer_Dense(2, 5)
    activation1 = Activation_Function()
    
    # Forward pass through first layer
    layer1.forward(X)
    # Applying activation function
    activation1.forward(layer1.output)
    
    # Print the output of the activation function
    print("Activation output:\n", activation1.output)