from typing import Any

import numpy as np

from src.structures import NeuraConfig


class DeepNeuralNetwork:
    """Multilayer Perceptron (MLP) implementation with ADAM optimizer.

    Supports custom hidden layer architectures, He initialization,
    and categorical cross-entropy loss for multi-class classification.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def __init__(self, config: NeuraConfig, layer_sizes: list[int]) -> None:
        """Initialize the network architecture and optimization parameters.

        Args:
            config (NeuraConfig): Centralized configuration object.
            layer_sizes (list[int]): List containing the number of neurons
                for each layer (input, hidden, output).

        """
        self.cfg = config
        self.training = True

        # Reproducibility anchor.
        self.rng = np.random.default_rng(seed=self.cfg.seed)

        self.layer_sizes = layer_sizes

        self._initialize_parameters()

    def train(self) -> None:
        """Set the model to training mode."""
        self.training = True

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.training = False

    def _initialize_parameters(self) -> None:
        """Allocates memory and initializes weights, biases, and ADAM optimizer moments.

        Uses He initialization for weights to suit Leaky ReLU activation and
        initializes all biases and ADAM moments to zero.

        Attributes initialized:
            weights (list[np.ndarray]): List of weight matrices for each layer.
            biases (list[np.ndarray]): List of bias vectors for each layer.
            m_w (list[np.ndarray]): First moment vectors for weights (ADAM).
            v_w (list[np.ndarray]): Second moment vectors for weights (ADAM).
            m_b (list[np.ndarray]): First moment vectors for biases (ADAM).
            v_b (list[np.ndarray]): Second moment vectors for biases (ADAM).
        """
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # He Initialization: optimized for layers using ReLU/Leaky ReLU
            std = np.sqrt(2.0 / n_in)
            w = self.rng.normal(0.0, std, (n_out, n_in))
            self.weights.append(w)

            # Biases initialization
            b = np.zeros((n_out, 1))
            self.biases.append(b)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Apply the Leaky Rectified Linear Unit activation function.

        Args:
            x (np.ndarray): Input tensor (pre-activation values Z).
            alpha (float): Slope of the activation function for x < 0.
                Defaults to 0.01.

        Returns:
            np.ndarray: Activated values of the same shape as input.

        """
        return np.where(x > 0, x, x * alpha)

    def leaky_relu_deriv(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Compute the derivative of the Leaky ReLU activation function.

        Args:
            x (np.ndarray): Input tensor (pre-activation values Z).
            alpha (float): Slope for the negative gradient.
                Defaults to 0.01.

        Returns:
            np.ndarray: Gradient of the activation function.

        """
        return np.where(x > 0, 1, alpha)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute the Softmax activation for multi-class classification.

        Includes a shift (max subtraction) for numerical stability to
        prevent overflow during exponentiation.

        Args:
            x (np.ndarray): Input logit tensor of shape (n_classes, batch_size).

        Returns:
            np.ndarray: Normalized probability distribution where the sum
                of each column equals 1.

        """
        shift_x = x - np.max(x, axis=0, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def predict(self, inputs_list: np.ndarray) -> np.ndarray:
        """Perform a forward pass through the network to generate predictions.

        Args:
            inputs_list (np.ndarray): Input data matrix
                of shape (n_samples, n_features).

        Returns:
            np.ndarray: Output probability matrix of shape (n_samples, n_classes).

        """
        inputs = np.array(inputs_list, ndmin=2).T
        _, activations = self._forward(inputs)

        return activations[-1].T

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy percentage.

        Args:
            X: Input features.
            y: One-hot encoded target labels.

        """
        was_training = self.training
        self.eval()

        predictions = self.predict(X)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(pred_labels == true_labels) * 100.0

        if was_training:
            self.train()
        return float(accuracy)

    def train_step(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Perform forward and backward passes to compute gradients.

        Returns:
            A tuple of (loss, grads_w, grads_b).

        """
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        batch_size = inputs.shape[1]

        # 1. Forward
        z_steps, activations = self._forward(inputs)

        # 2. Loss & Initial Error
        predictions = np.clip(activations[-1], self.cfg.epsilon, 1.0 - self.cfg.epsilon)
        loss = -np.sum(targets * np.log(predictions)) / batch_size
        errors = activations[-1] - targets

        # 3. Backward
        grads_w: list[np.ndarray] = [np.empty(0)] * len(self.weights)
        grads_b: list[np.ndarray] = [np.empty(0)] * len(self.biases)

        for i in reversed(range(len(self.weights))):
            delta = (
                errors
                if i == len(self.weights) - 1
                else errors * self.leaky_relu_deriv(z_steps[i])
            )

            if i > 0:
                errors = np.dot(self.weights[i].T, delta)

            grads_w[i] = np.dot(delta, activations[i].T) / batch_size
            grads_b[i] = np.sum(delta, axis=1, keepdims=True) / batch_size

        return float(loss), grads_w, grads_b

    def get_state_dict(self) -> dict[str, Any]:
        """Collect all trainable parameters (weights and biases) from the network.

        Returns:
            dict[str, list[np.ndarray]]: A dictionary where keys are parameter names
                (e.g., 'weights', 'biases') and values are lists with NumPy arrays.

        """
        return {
            "weights": np.array(self.weights, dtype=object),
            "biases": np.array(self.biases, dtype=object),
        }

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Perform a full forward pass through the network layers.

        Computes pre-activation values (Z) and activated values (A) for
        each layer, handling the transition between hidden layer
        activation (Leaky ReLU) and output activation (Softmax).

        Args:
            X (np.ndarray): Input feature matrix of shape (n_features, batch_size).

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing:
                - z_steps: List of pre-activation values for each layer.
                - activations: List of activated values, including the
                  original input as the first element.

        """
        z_steps: list[np.ndarray] = []
        activations: list[np.ndarray] = [X]

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_steps.append(z)

            a = self.softmax(z) if i == len(self.weights) - 1 else self.leaky_relu(z)
            activations.append(a)

        return z_steps, activations
