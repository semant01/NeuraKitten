import numpy as np

from src.structures import NeuraConfig


class DeepNeuralNetwork:
    """Multilayer Perceptron (MLP) implementation with ADAM optimizer.

    Supports custom hidden layer architectures, He initialization,
    and categorical cross-entropy loss for multi-class classification.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    m_w: list[np.ndarray]
    v_w: list[np.ndarray]
    m_b: list[np.ndarray]
    v_b: list[np.ndarray]

    def __init__(self, config: NeuraConfig, layer_sizes: list[int]) -> None:
        """Initialize the network architecture and optimization parameters.

        Args:
            config (NeuraConfig): Centralized configuration object.
            layer_sizes (list[int]): List containing the number of neurons
                for each layer (input, hidden, output).

        """
        self.cfg = config

        # Reproducibility anchor.
        self.rng = np.random.default_rng(seed=self.cfg.seed)

        self.layer_sizes = layer_sizes
        self.t = 0

        self._initialize_parameters()

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

        self.m_w: list[np.ndarray] = []
        self.v_w: list[np.ndarray] = []
        self.m_b: list[np.ndarray] = []
        self.v_b: list[np.ndarray] = []

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

            # ADAM moments initialization
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

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

    def train(self, inputs: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """Perform one training step using backpropagation and ADAM.

        Args:
            inputs (np.ndarray): Batch of input features.
            targets (np.ndarray): One-hot encoded target labels.
            lr (float): Current learning rate.

        Returns:
            float: The categorical cross-entropy loss for the current batch.

        """
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        batch_size = inputs.shape[1]

        self.t += 1

        # 1.--- Forward propagation ---
        z_steps: list[np.ndarray] = []
        activations: list[np.ndarray] = [inputs]

        z_steps, activations = self._forward(inputs)

        # 2. Error and Loss calculation
        # Categorical Cross-Entropy for multi-class
        predictions: np.ndarray = np.clip(
            activations[-1], self.cfg.epsilon, 1.0 - self.cfg.epsilon
        )
        loss = -np.sum(targets * np.log(predictions)) / batch_size
        errors = activations[-1] - targets

        # 3. Backward propagation
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                delta = errors
            else:
                # Hidden layer delta (Leaky ReLU)
                delta = errors * self.leaky_relu_deriv(z_steps[i])

            # Recalculate error for the previous layer
            if i > 0:
                errors = np.dot(self.weights[i].T, delta)

            grad_w = np.dot(delta, activations[i].T) / batch_size
            grad_b = np.sum(delta, axis=1, keepdims=True) / batch_size

            # Weights moments
            self.m_w[i], self.v_w[i] = self._update_moments(
                self.m_w[i], self.v_w[i], grad_w
            )

            # Biases moments
            self.m_b[i], self.v_b[i] = self._update_moments(
                self.m_b[i], self.v_b[i], grad_b
            )

            # Bias correction and apply updates
            self.weights[i] -= lr * self._get_adam_update(self.m_w[i], self.v_w[i])
            self.biases[i] -= lr * self._get_adam_update(self.m_b[i], self.v_b[i])

        return float(loss)

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

    def _update_moments(
        self, m: np.ndarray, v: np.ndarray, grad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update ADAM first and second moments for a given gradient.

        Args:
            m (np.ndarray): Current first moment.
            v (np.ndarray): Current second moment.
            grad (np.ndarray): Calculated gradient for the parameter.

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated (m, v) moments.

        """
        beta1 = self.cfg.beta1
        beta2 = self.cfg.beta2

        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * (grad**2)

        return m_new, v_new

    def _get_adam_update(self, m: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the ADAM gradient correction term.

        Calculates bias-corrected first and second moments and returns
        the final update component: m_hat / (sqrt(v_hat) + epsilon).

        Args:
            m (np.ndarray): The first moment vector (mean of gradients).
            v (np.ndarray): The second moment vector (uncentered variance of gradients).

        Returns:
            np.ndarray: The computed update vector to be subtracted from
                parameters, of the same shape as input moments.

        """
        m_hat = m / (1 - self.cfg.beta1**self.t)
        v_hat = v / (1 - self.cfg.beta2**self.t)

        return m_hat / (np.sqrt(v_hat) + self.cfg.epsilon)
