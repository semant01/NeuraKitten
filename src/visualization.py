from typing import TYPE_CHECKING, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

matplotlib.use("TkAgg")

if TYPE_CHECKING:
    from src.config import NeuraConfig

    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def _create_decision_mesh(
    cfg: "NeuraConfig", engine: "FeatureEngine", scaler: "DataScaler"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a transformed grid for decision boundary plotting.

    Args:
        cfg: Configuration object for range and resolution.
        engine: Feature engine for data transformation.
        scaler: Scaler for data normalization.

    Returns:
        A tuple of (xx, yy, scaled_grid) where xx, yy are meshgrid matrices.

    """
    view_range = cfg.view_range
    res = cfg.resolution

    x_vals = np.linspace(-view_range, view_range, res)
    y_vals = np.linspace(-view_range, view_range, res)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Prepare grid for prediction
    raw_grid = np.c_[xx.ravel(), yy.ravel()]
    extended_grid = engine.transform(raw_grid)
    scaled_grid = scaler.transform(extended_grid)

    return xx, yy, scaled_grid


def live_plot(
    brain: "DeepNeuralNetwork",
    cfg: "NeuraConfig",
    engine: "FeatureEngine",
    scaler: "DataScaler",
    X_raw: np.ndarray,
    targets: np.ndarray,
    epoch: int,
    lr: float,
    current_loss: float,
    accuracy: float,
    ax: Axes,
) -> None:
    """Update the live visualization of the neural network's decision boundary.

    Args:
        brain: The trained neural network model instance.
        cfg: Centralized configuration object.
        engine: Feature engineering engine for data transformation.
        scaler: Scaler used for data normalization.
        X_raw: Original input features (Cartesian), shape (N, 2).
        targets: One-hot encoded labels for the dataset.
        epoch: Current training epoch.
        lr: Current learning rate.
        current_loss: Current loss value.
        accuracy: Current prediction accuracy.
        ax: Matplotlib axes object to draw on.

    """
    ax.clear()

    # 1. Prepare grid and prediction
    xx, yy, scaled_grid = _create_decision_mesh(cfg, engine, scaler)

    # 2. Get prediction
    preds = brain.predict(scaled_grid)
    zz = np.argmax(preds, axis=1).reshape(xx.shape)

    num_classes = preds.shape[1]
    plot_targets = np.argmax(targets, axis=1)

    # 3. Plot Decision Boundaries
    levels = np.arange(num_classes + 1) - 0.5
    ax.contourf(xx, yy, zz, levels=levels, cmap=cfg.cmap, alpha=0.4)

    # 4. Plot dataset
    if cfg.show_dataset_points and X_raw is not None:
        ax.scatter(
            X_raw[:, 0],
            X_raw[:, 1],
            c=plot_targets,
            s=15,
            cmap=cfg.cmap,
            edgecolors="white",
            linewidth=0.5,
            alpha=0.7,
        )

    # 5. Configure plot
    view_range = cfg.view_range
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")

    ax.set_title(
        f"Epoch: {epoch:4d} | Loss: {current_loss:.6f} | "
        f"Acc: {accuracy:.2f}% | LR: {lr:.6f}"
    )

    # Update screen
    plt.draw()
    plt.pause(0.001)
