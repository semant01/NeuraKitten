import logging
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event, KeyEvent

from .data_utils import NeuraDataLoader
from .visualization import live_plot

matplotlib.use("TkAgg")

if TYPE_CHECKING:
    from src.config import NeuraConfig

    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def _calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute the accuracy percentage between predictions and ground truth.

    Args:
        predictions (np.ndarray): Probability matrix from the model.
        targets (np.ndarray): One-hot encoded target labels.

    Returns:
        float: Accuracy percentage (0.0 to 100.0).

    """
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    return float(np.mean(pred_labels == true_labels) * 100)


def _get_current_lr(
    initial_lr: float, decay_rate: float, epoch: int, min_lr: float
) -> float:
    """Calculate the decayed learning rate for the current epoch.

    Args:
        initial_lr (float): Starting learning rate.
        decay_rate (float): Rate of decay per epoch.
        epoch (int): Current epoch number.
        min_lr (float): Minimum allowed learning rate.

    Returns:
        float: Adjusted learning rate.

    """
    lr = initial_lr / (1 + decay_rate * epoch)
    return max(lr, min_lr)


def fit(
    model: "DeepNeuralNetwork",
    inputs: np.ndarray,
    targets: np.ndarray,
    cfg: "NeuraConfig",
    X_raw: np.ndarray,
    scaler: "DataScaler",
    engine: "FeatureEngine",
) -> "DeepNeuralNetwork":
    """Trains the Deep Neural Network using Mini-batch Gradient Descent and ADAM.

    Handles the training loop, learning rate scheduling, logging, and
    interactive visualization (pause/stop via keyboard).

    Args:
        model (DeepNeuralNetwork): The neural network instance to train.
        inputs (np.ndarray): Preprocessed training features.
        targets (np.ndarray): Target labels (one-hot encoded).
        cfg (NeuraConfig): Configuration object with hyperparameters.
        X_raw (np.ndarray): Original features for visualization purposes.
        scaler (DataScaler): Scaler used for data normalization.
        engine (FeatureEngine): Engine used for feature mapping.

    Returns:
        DeepNeuralNetwork: The trained model instance.

    """
    state: dict[str, bool] = {"paused": False, "stop": False}

    fig, ax = None, None

    if cfg.visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

        def on_press(event: Event) -> None:
            if not isinstance(event, KeyEvent):
                return
            if event.key == "p":
                state["paused"] = not state["paused"]
                status = "PAUSED" if state["paused"] else "RESUMED"
                print(f"\n[Status] {status}")
            if event.key == "escape":
                state["stop"] = True
                print("\n[Status] STOPPING...")

        fig.canvas.mpl_connect("key_press_event", on_press)

    loader: NeuraDataLoader = NeuraDataLoader(
        inputs, targets, cfg.batch_size, balanced=cfg.balanced_batches, seed=cfg.seed
    )

    for epoch in range(cfg.epochs):
        if cfg.visualize and state["stop"]:
            break
        while cfg.visualize and state["paused"]:
            plt.pause(0.1)
            if state["stop"]:
                break

        # Update Learning Rate
        lr = _get_current_lr(cfg.initial_lr, cfg.decay_rate, epoch, cfg.min_lr)

        epoch_losses: list[float] = []

        # Mini-batch training
        for batch_indices in loader:
            loss = model.train(inputs[batch_indices], targets[batch_indices], lr)
            epoch_losses.append(loss)

        current_loss = float(np.mean(epoch_losses))

        # Logging and plot visualization on demand and at the last epoch
        is_log_frame = epoch % cfg.frame_log == 0 or epoch == cfg.epochs - 1
        is_vis_frame = epoch % cfg.frame_visual == 0 or epoch == cfg.epochs - 1

        if is_log_frame or is_vis_frame:
            predictions = model.predict(inputs)
            accuracy = _calculate_accuracy(predictions, targets)

        if is_log_frame:
            logging.info(
                f"Epoch: {epoch:4d} | Loss: {current_loss:.6f} | "
                f"Acc: {accuracy:6.2f}% | LR: {lr:.6f}"
            )

        if cfg.visualize and ax is not None and is_vis_frame:
            live_plot(
                brain=model,
                cfg=cfg,
                engine=engine,
                scaler=scaler,
                X_raw=X_raw,
                targets=targets,
                epoch=epoch,
                lr=lr,
                current_loss=current_loss,
                accuracy=accuracy,
                ax=ax,
            )

    if cfg.visualize:
        plt.ioff()
        plt.show()

    return model
