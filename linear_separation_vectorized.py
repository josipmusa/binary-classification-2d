import time
import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, num_neurons, num_input_per_neuron):
        self.weights = np.random.uniform(size=(num_input_per_neuron, num_neurons))
        self.biases = np.random.uniform(size=(1, num_neurons))
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(inputs, self.weights) + self.biases
        a = _sigmoid(z)
        self.outputs = a
        return self.outputs

    # mental note for delta(or how a weight affected final loss AKA partial derivative) ->
    # 1. calculate how much the weight affected the product of a neuron (AKA a change in w to z) -> this results to just the input
    # 2. calculate how much the product affected the activation (AKA a change in z to a) -> this results to derivative of activation function (in this case sigmoid)
    # 3. calculate how much the activation affected the next layer (this can be computed by passing the error rate of the current layer to the previous layer)
    # 4. this all comes out to previous_error * activation derivation * input -> all of this computes the gradient descent for the weights - process is same for bias except no need for input multiplication
    def backward(self, loss_errors, learning_rate):
        activations_derivatives = self._activations_derivatives()
        deltas = loss_errors * activations_derivatives

        weight_gradient = np.dot(self.inputs.T, deltas)
        bias_gradient = np.sum(deltas, axis=0, keepdims=True)

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return self._calculate_error_loss(deltas)

    def _calculate_error_loss(self, deltas):
        return np.dot(deltas, self.weights.T)

    def _activations_derivatives(self):
        return self.outputs * (1 - self.outputs)


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=10000):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_history = []

        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i + 1], layers[i]))

    def train(self, X, Y, batch_size=32):
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            # Shuffle dataset at the start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            total_error = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                activations = self._forward_pass(X_batch)

                output_error_loss = activations - Y_batch
                total_error += np.sum(output_error_loss ** 2)

                self._backward_pass(output_error_loss)

            if epoch % 1000 == 0:
                epoch_mse = total_error / n_samples
                self.loss_history.append(epoch_mse)
                print(f"Epoch {epoch}, MSE: {epoch_mse}")

    def predict(self, X):
        return self._forward_pass(X)

    def _forward_pass(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)

        return activations

    def _backward_pass(self, output_error_loss):
        error_loss = output_error_loss
        for i in reversed(range(len(self.layers))):
            error_loss = self.layers[i].backward(error_loss, self.learning_rate)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualize(neural_network, x, y):
    # --- Plot loss curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, neural_network.epochs, 1000), neural_network.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    # --- Plot decision boundary ---
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = neural_network.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.6)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k", s=40)
    plt.title("Decision Boundary")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.tight_layout()
    plt.savefig("decision_boundary.png")
    plt.close()


def linear_separation_vectorized():
    x = np.random.rand(200, 2)
    y = (x[:, 1] > 0.5 * x[:, 0] + 0.2).astype(int).reshape(-1, 1)  # reshape so it matches activations

    # input layer 2 neurons, hidden layer 4 neurons, output layer 1 neuron
    layers = [2, 4, 1]
    neural_network = NeuralNetwork(layers)
    neural_network.train(x, y)

    neural_network.predict(x)

    # points below the line → expected class 0
    test_points_0 = np.array([
        [0.1, 0.1],
        [0.3, 0.2],
        [0.5, 0.3]
    ])

    # points above the line → expected class 1
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

    visualize(neural_network, x, y)


if __name__ == '__main__':
    start_time = time.time()
    linear_separation_vectorized()
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")
