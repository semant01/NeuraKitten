import numpy as np

from .structures import NeuraConfig


class AdamOptimizer:
    """ADAM optimizer with integrated learning rate decay.

    Manages first and second moments for all model parameters and
    updates them using the ADAM update rule.
    """

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initialize the Adam Optimizer class with config parameters.

        Args:
        cfg (NeuraConfig): Centralized configuration object.

        """
        self.cfg = cfg
        self.lr = cfg.initial_lr
        self.t = 0

        self.m_w: list[np.ndarray] = []
        self.v_w: list[np.ndarray] = []
        self.m_b: list[np.ndarray] = []
        self.v_b: list[np.ndarray] = []

    def initialize(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """Allocate memory for moments based on model parameters."""
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]

    def step(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        grads_w: list[np.ndarray],
        grads_b: list[np.ndarray],
    ) -> None:
        """Perform one update step for all parameters."""
        self.t += 1

        # Learning Rate Decay (Time-based decay)
        self.lr = max(
            self.cfg.initial_lr * (1.0 / (1.0 + self.cfg.decay_rate * self.t)),
            self.cfg.min_lr,
        )

        for i in range(len(weights)):
            # Update moments for weights
            self.m_w[i] = (
                self.cfg.beta1 * self.m_w[i] + (1 - self.cfg.beta1) * grads_w[i]
            )
            self.v_w[i] = self.cfg.beta2 * self.v_w[i] + (1 - self.cfg.beta2) * (
                grads_w[i] ** 2
            )

            # Update moments for biases
            self.m_b[i] = (
                self.cfg.beta1 * self.m_b[i] + (1 - self.cfg.beta1) * grads_b[i]
            )
            self.v_b[i] = self.cfg.beta2 * self.v_b[i] + (1 - self.cfg.beta2) * (
                grads_b[i] ** 2
            )

            # Bias correction & Application
            weights[i] -= self.lr * self._get_update(self.m_w[i], self.v_w[i])
            biases[i] -= self.lr * self._get_update(self.m_b[i], self.v_b[i])

    def _get_update(self, m: np.ndarray, v: np.ndarray) -> np.ndarray:
        m_hat = m / (1 - self.cfg.beta1**self.t)
        v_hat = v / (1 - self.cfg.beta2**self.t)
        return m_hat / (np.sqrt(v_hat) + self.cfg.epsilon)
