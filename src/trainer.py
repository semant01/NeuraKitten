import logging

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event, KeyEvent

from .visualization import live_plot

if TYPE_CHECKING:
    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def fit(
    model: "DeepNeuralNetwork",
    engine: "FeatureEngine",
    inputs: np.ndarray,
    targets: np.ndarray,
    scaler: "DataScaler",
    X_raw: np.ndarray,
    epochs: int = 5001,
    batch_size: int = 256,
    base_lr: float = 0.01,
    lr_gradient: float = 0.9998,
    visualize: bool = True,
    color_gradient: bool = False,
    show_dataset_points: bool = False,
    show_levels: bool = False,
) -> "DeepNeuralNetwork":
    """Train Deep Neural Network."""
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        state = {"paused": False, "stop": False}

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

    indices = np.arange(len(inputs))
    history_loss = []

    # Smart LR adjusting, baseline batch size is assumed 32
    initial_lr = base_lr * (batch_size / 32)
    lr = initial_lr
    lr_grad = lr_gradient

    for epoch in range(epochs):
        if visualize and state["stop"]:
            break

        while visualize and state["paused"]:
            plt.pause(0.1)
            if state["stop"]:
                break

        # LR Decay
        lr = initial_lr * (lr_grad**epoch)
        lr = max(lr, 0.0001)

        epoch_losses = []

        np.random.shuffle(indices)

        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_loss = model.train(inputs[batch_indices], targets[batch_indices], lr)
            epoch_losses.append(batch_loss)

        current_loss = float(np.mean(epoch_losses))
        history_loss.append(current_loss)

        # Log reporting
        if epoch % 1000 == 0:
            logging.info(f"Epoch: {epoch} | Loss: {current_loss:.6f} | LR: {lr:.4f}")

        # Visual plot
        if visualize and epoch % 10 == 0:
            live_plot(
                model,
                engine,
                scaler,
                X_raw,
                targets,
                epoch,
                lr,
                current_loss,
                ax,
                color_gradient,
                show_dataset_points,
                show_levels,
            )

    if visualize:
        plt.ioff()
        plt.show()

    return model
