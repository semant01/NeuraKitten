from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .callbacks import BaseCallback
    from .model import DeepNeuralNetwork
    from .optimizer import AdamOptimizer
    from .structures import ExperimentContext, NeuraConfig


class Trainer:
    """Orchestrator for the neural network training process.

    This class binds the model, optimizer, and configuration together,
    managing the execution of the training loop and callback notifications.
    """

    def __init__(
        self, model: DeepNeuralNetwork, optimizer: AdamOptimizer, cfg: NeuraConfig
    ) -> None:
        """Initialize the trainer with the core components.

        Args:
            model: The neural network instance.
            optimizer: The optimizer (e.g., Adam).
            cfg: Centralized configuration object.

        """
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ctx: ExperimentContext,
        callbacks: list[BaseCallback] | None = None,
    ) -> None:
        """Execute the training loop and notify observers of progress.

        Args:
            X_train: Preprocessed input features.
            y_train: One-hot encoded labels.
            ctx: Centralized context for storing metrics and history.
            callbacks: Optional list of hooks for logging, saving, and UI.

        """
        callbacks = callbacks or []
        if hasattr(self.model, "train"):
            self.model.train()

        # 1. Signal training start
        for cb in callbacks:
            cb.on_train_begin(self.model, ctx)

        for epoch in range(1, self.cfg.epochs + 1):
            # 2. Mathematical core: Training step
            loss, g_w, g_b = self.model.train_step(X_train, y_train)
            self.optimizer.step(self.model.weights, self.model.biases, g_w, g_b)

            # 3. State update: Metrics calculation
            accuracy = self.model.calculate_accuracy(X_train, y_train)
            ctx.update_metrics(epoch, loss, accuracy, self.optimizer.lr)

            # 4. Observer notification: Epoch end
            for cb in callbacks:
                cb.on_epoch_end(epoch, self.model, ctx)

            # 5. Control flow: Check for early stopping (e.g., ESC key)
            if any(cb.should_stop for cb in callbacks):
                break

        # 6. Signal training end
        for cb in callbacks:
            cb.on_train_end(self.model, ctx)
