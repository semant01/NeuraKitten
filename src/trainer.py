import logging

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event, KeyEvent

from src.config import NeuraConfig

from .visualization import live_plot

if TYPE_CHECKING:
    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def fit(
    model: "DeepNeuralNetwork",
    inputs: np.ndarray,
    targets: np.ndarray,
    cfg: NeuraConfig,
    X_raw: np.ndarray,
    scaler: "DataScaler",
    engine: "FeatureEngine",
) -> "DeepNeuralNetwork":
    """Train Deep Neural Network."""
    # Reproducibility anchor.
    rng = np.random.default_rng(seed=cfg.seed)
    if cfg.visualize:
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

    # LR adjusting
    initial_lr = cfg.initial_lr
    decay_rate = cfg.decay_rate

    for epoch in range(cfg.epochs):
        if cfg.visualize and state["stop"]:
            break

        while cfg.visualize and state["paused"]:
            plt.pause(0.1)
            if state["stop"]:
                break

        # LR Decay
        lr = initial_lr / (1 + decay_rate * epoch)
        lr = max(lr, cfg.min_lr)

        epoch_losses = []

        rng.shuffle(indices)

        for start_idx in range(0, len(indices), cfg.batch_size):
            batch_indices = indices[start_idx : start_idx + cfg.batch_size]
            batch_loss = model.train(inputs[batch_indices], targets[batch_indices], lr)
            epoch_losses.append(batch_loss)

        current_loss = float(np.mean(epoch_losses))
        history_loss.append(current_loss)

        # Log reporting
        if epoch % cfg.frame_log == 0:
            logging.info(f"Epoch: {epoch} | Loss: {current_loss:.6f} | LR: {lr:.6f}")

        # Visual plot
        if cfg.visualize and epoch % cfg.frame_skip == 0:
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
                ax=ax,
            )

    if cfg.visualize:
        plt.ioff()
        plt.show()

    return model
