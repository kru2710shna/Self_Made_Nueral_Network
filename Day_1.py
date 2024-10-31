import numpy as np

X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

# Hidden Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

if __name__ == "__main__":
    layer1 = Layer_Dense(4, 5)
    layer2 = Layer_Dense(5, 2)
    
    print("Layer 1")
    layer1.forward(X)
    print("Layer1 output shape:", layer1.output.shape)
    print(layer1.output)
    
    print("Layer 2")
    layer2.forward(layer1.output)
    print("Layer2 output shape:", layer2.output.shape)
    print(layer2.output)
