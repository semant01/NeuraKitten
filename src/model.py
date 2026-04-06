import numpy as np


class DeepNeuralNetwork:
    """A flexible Deep Neural Network implementation from scratch using NumPy.

    Supports arbitrary hidden layer architectures and vectorized batch training.
    """

    def __init__(self, layers_size: list) -> None:
        """Initiate parameters for the Neuaral Network."""
        self.layers_size = layers_size
        self.weights = []
        self.biases = []

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        # He Initialization: optimized for layers using ReLU/Leaky ReLU
        for i in range(len(layers_size) - 1):
            n_in = layers_size[i]  # nodes at this layer
            n_out = layers_size[i + 1]  # nodes at next layer

            # Weights: normal distribution with variance scaled by input size
            std = np.sqrt(2.0 / n_in)
            w = np.random.normal(0.0, std, (n_out, n_in))
            self.weights.append(w)

            # Biases: initialized to zero to ensure neutral starting state
            b = np.zeros((n_out, 1))
            self.biases.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Return result of sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, output: np.ndarray) -> np.ndarray:
        """Return result of sigmoid detivative."""
        # Expected 'x' is the output of the sigmoid function
        return output * (1.0 - output)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Return result of Leaky Re-LU function."""
        # Expected 'x' is Z after activation
        return np.where(x > 0, x, x * alpha)

    def leaky_relu_deriv(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Return result of Leaky Re-LU derivative."""
        # Expected 'x' is Z before activation
        return np.where(x > 0, 1, alpha)

    def predict(self, inputs_list: np.ndarray) -> np.ndarray:
        """Forward pass to generate predictions."""
        inputs = np.array(inputs_list, ndmin=2).T
        activation = inputs

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            if i == len(self.weights) - 1:
                activation = self.sigmoid(z)
            else:
                activation = self.leaky_relu(z)

        return activation.T

    def train(
        self, inputs_list: np.ndarray, targets_list: np.ndarray, lr: float
    ) -> float:
        """Run Standard Backpropagation algorithm on a data batch and return loss."""
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        batch_size = inputs.shape[1]

        self.t += 1

        # 1. Forward propagation
        zs = []
        activations = [inputs]

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)

            # The last layer is different activation function (sigmoind)
            if i == len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self.leaky_relu(z)
            activations.append(a)

        # 2. Error and Loss calculation (MSE - Mean Squared Error)
        errors = activations[-1] - targets
        loss = np.mean(np.square(errors))

        # 3. Backward propagation
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                # Output layer delta (Sigmoid)
                delta = errors * self.sigmoid_deriv(activations[i + 1])
            else:
                # Hidden layer delta (Leaky ReLU)
                delta = errors * self.leaky_relu_deriv(zs[i])

            # Recalculate error for the previous layer
            if i > 0:
                errors = np.dot(self.weights[i].T, delta)

            grad_w = np.dot(delta, activations[i].T) / batch_size
            grad_b = np.sum(delta, axis=1, keepdims=True) / batch_size

            # Weights moments
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w**2)

            # Biases moments
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b**2)

            # Bias correction
            m_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat = self.v_w[i] / (1 - self.beta2**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # Apply updates
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return float(loss)
