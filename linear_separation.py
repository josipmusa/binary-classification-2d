import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, num_inputs):
        self.inputs = None
        self.output = None
        self.weights = np.random.uniform(size=(num_inputs))
        self.bias = np.random.uniform()

    def activate(self, inputs):
        self.inputs = inputs
        z = np.dot(inputs , self.weights) + self.bias
        self.output = _sigmoid(z)
        return self.output

    def activation_derivative(self):
        return self.output * (1 - self.output)

    def update_weights(self, negative_delta, learning_rate):
        self.weights += learning_rate * negative_delta * self.inputs
        self.bias += learning_rate * negative_delta

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate):
        deltas = []
        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * neuron.activation_derivative()
            neuron.update_weights(delta, learning_rate)
            deltas.append(delta)
        return self._calculate_error_loss(deltas)

    def _calculate_error_loss(self, deltas):
        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []

        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i+1], layers[i]))

    def train(self, inputs, outputs):
        for epoch in range(self.epochs):
            total_error = 0
            for x,y in zip(inputs, outputs):

                activations = self._forward(x)

                output_error_loss = y - activations
                total_error += np.sum(output_error_loss ** 2)

                self._backward(output_error_loss)

            if epoch % 1000 == 0:
                mse = total_error / len(inputs)
                print(f'Epoch {epoch}, MSE: {mse}')

    def _forward(self, initial_input):
        activations = initial_input
        for layer in self.layers:
            activations = layer.forward(activations)

        return activations

    def _backward(self, output_error_loss):
        loss_error = output_error_loss
        for i in reversed(range(len(self.layers))):
            loss_error = self.layers[i].backward(loss_error, self.learning_rate)


    def predict(self, inputs):
        activations = []
        for x in inputs:
            activations.append(self._forward(x))

        return np.array(activations)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))



def linear_separation():
    x = np.random.rand(200, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    y = (x2 > 0.5 * x1 + 0.2).astype(int)

    #input layer 2 neurons, hidden layer 4 neurons, output layer 1 neuron
    layers = [2,4,1]
    neural_network = NeuralNetwork(layers)
    neural_network.train(x, y)

    neural_network.predict(x)

    #points below the line → expected class 0
    test_points_0 = np.array([
        [0.1, 0.1],
        [0.3, 0.2],
        [0.5, 0.3]
    ])

    #points above the line → expected class 1
    test_points_1 = np.array([
        [0.1, 0.5],
        [0.4, 0.6],
        [0.6, 0.7]
    ])

    X_test = np.vstack((test_points_0, test_points_1))
    y_test = (X_test[:, 1] > 0.5 * X_test[:, 0] + 0.2).astype(int)
    predictions = neural_network.predict(X_test)
    predicted_classes = (predictions.flatten() > 0.5).astype(int)

    print(" x1     x2   | Expected  Predicted")
    print("---------------------------------")
    for i in range(len(X_test)):
        print(f"{X_test[i, 0]:.2f}  {X_test[i, 1]:.2f}  |    {y_test[i]}        {predicted_classes[i]}")

if __name__ == '__main__':
    linear_separation()