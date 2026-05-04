from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import DeepNeuralNetwork
    from .storage import ExperimentManager
    from .structures import ExperimentContext, NeuraConfig
    from .visualization import VisualizerEngine


class BaseCallback(ABC):
    """Abstract base class for building training callbacks.

    Callbacks allow for custom logic to be executed at specific stages
    of the training loop (e.g., logging, saving, visualization).
    """

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initialize the callback class with config parameters.

        Args:
        cfg (NeuraConfig): Centralized configuration object.

        """
        self.cfg = cfg

    def on_train_begin(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Call once before the training loop starts.

        Args:
            model: The neural network instance being trained.
            ctx: Current experiment context.

        """
        pass

    def on_epoch_end(
        self, epoch: int, model: DeepNeuralNetwork, ctx: ExperimentContext
    ) -> None:
        """Call at the end of every epoch.

        Args:
            epoch: The index of the epoch just completed.
            model: The neural network instance with updated weights.
            ctx: Experiment context with updated metrics.

        """
        pass

    def on_train_end(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Call once after the training loop completes.

        Args:
            model: The final trained model.
            ctx: Final experiment context with full history.

        """
        pass

    @property
    def should_stop(self) -> bool:
        """Indicates whether the training loop should be terminated early.

        Returns:
            bool: True if training should stop, False otherwise.

        """
        return False


class ConsoleLoggerCallback(BaseCallback):
    """Callback that logs training progress to the console.

    This replaces the manual logging previously found inside the fit loop.
    """

    def on_train_begin(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Log the start of the training process.

        Args:
            model: The neural network instance.
            ctx: Context with the experiment name.

        """
        logging.info(f"--- Starting training: {ctx.experiment_name} ---")
        logging.info(f"Total epochs: {self.cfg.epochs}")

    def on_epoch_end(
        self, epoch: int, model: DeepNeuralNetwork, ctx: ExperimentContext
    ) -> None:
        """Log metrics at specified intervals.

        Args:
            epoch: Current epoch index.
            model: The model (not used here, but required by signature).
            ctx: Context containing the latest metrics (loss, accuracy, lr).

        """
        if epoch % self.cfg.frame_log == 0:
            logging.info(
                f"Epoch: {epoch:4d} | Loss: {ctx.loss:.6f} | "
                f"Acc: {ctx.accuracy:6.2f}% | LR: {ctx.lr:.6f}"
            )

    def on_train_end(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Log the completion of training.

        Args:
            model: The final model.
            ctx: Context with final metrics.

        """
        logging.info("--- Training complete ---")
        logging.info(
            f"Epoch {ctx.epoch}: "
            f"Final Loss: {ctx.loss:.6f}, Final Acc: {ctx.accuracy:.2f}%"
        )


class StorageCallback(BaseCallback):
    """Handles all disk I/O operations: checkpoints and final metrics."""

    def __init__(
        self,
        cfg: NeuraConfig,
        storage: ExperimentManager,
        experiment_name: str,
        X_raw: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Initialize the callback class with config parameters.

        Args:
        cfg (NeuraConfig): Centralized configuration object.
        storage (ExperimentManager): Filesystem infrastructure manager.
        experiment_name (str): Name used in saved files.
        X_raw (np.ndarray): Input data to be saved in the experiment folder.
        targets (np.ndarray): Target data to be saved in the experiment folder.

        """
        super().__init__(cfg)
        self.storage = storage
        self.X_raw = X_raw
        self.targets = targets

        if self.cfg.save_to_file:
            self.storage.create_session(experiment_name)

    def on_train_begin(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Save initial config and dataset."""
        if self.cfg.save_to_file:
            self.storage.save_config(self.cfg)
            self.storage.save_dataset(self.X_raw, self.targets)

    def on_epoch_end(
        self, epoch: int, model: DeepNeuralNetwork, ctx: ExperimentContext
    ) -> None:
        """Save parameters at currect epoch.

        Args:
            epoch: current epoch
            model: The final model.
            ctx: Context with final metrics.

        """
        if self.cfg.save_to_file and epoch % self.cfg.checkpoint_interval == 0:
            self.storage.save_checkpoint(epoch, model.get_state_dict())

    def on_train_end(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Save the final parameters.

        Args:
            epoch: current epoch
            model: The final model.
            ctx: Context with final metrics.

        """
        if self.cfg.save_to_file:
            self.storage.save_checkpoint(ctx.epoch, model.get_state_dict())
            self.storage.save_metrics(ctx)


class VisualizerCallback(BaseCallback):
    """Bridge between the training loop and the graphic rendering engine.

    This callback encapsulates the visualization logic, ensuring the training
    loop remains decoupled from matplotlib and UI event handling.
    """

    def __init__(
        self,
        cfg: NeuraConfig,
        viz_engine: VisualizerEngine,
    ) -> None:
        """Initialize the visualization engine and pre-calculate the decision mesh.

        Args:
            cfg: Centralized configuration object.
            viz_engine: ,

        """
        super().__init__(cfg)
        self.viz = viz_engine
        self._stop_requested: bool = False

    def on_train_begin(self, model: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Prepare the figure and display the initial state of the model.

        Args:
            model: The neural network instance.
            ctx: Current experiment context.

        """
        self.viz.render(model, ctx)

    def on_epoch_end(
        self, epoch: int, model: DeepNeuralNetwork, ctx: ExperimentContext
    ) -> None:
        """Update the visualization at intervals defined in the configuration.

        Also monitors the UI for termination signals (e.g., ESC key).

        Args:
            epoch: Current training epoch.
            model: The neural network with updated weights.
            ctx: Context containing learning history and current metrics.

        """
        # Sync the pause/continue signal from the UI engine
        while self.viz.paused and not self._stop_requested:
            self.viz.fig.canvas.start_event_loop(0.1)
            if self.viz.stop_requested:
                self._stop_requested = True

        # Sync the stop signal from the UI engine
        if self.viz.stop_requested:
            self._stop_requested = True
            logging.info("Interruption detected: Stop signal received from Visualizer.")

        # Render frame based on configuration interval
        if epoch % self.cfg.frame_visual == 0:
            self.viz.render(model, ctx)

    @property
    def should_stop(self) -> bool:
        """Return True if the user has requested to stop the experiment via UI.

        Returns:
            bool: Termination flag for the training loop.

        """
        return self._stop_requested
