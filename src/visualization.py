import matplotlib
import numpy as np
from matplotlib.axes import Axes

matplotlib.use("TkAgg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from src.config import NeuraConfig

if TYPE_CHECKING:
    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def live_plot(
    brain: "DeepNeuralNetwork",
    cfg: NeuraConfig,
    engine: "FeatureEngine",
    scaler: "DataScaler",
    X_raw: np.ndarray,
    targets: np.ndarray,
    epoch: int,
    lr: float,
    current_loss: float,
    ax: Axes,
) -> None:
    """Update the live visualization of the neural network's decision boundary.

    Args:
        brain: The neural network model instance.
        cfg: Configuration dataset
        engine: Feature engineering engine for data transformation.
        scaler: Scaler used for data normalization.
        X_raw: Original input features (Cartesian).
        targets: Labels for the dataset.
        epoch: Current training epoch.
        lr: Current learning rate.
        current_loss: Current loss value.
        ax: Matplotlib axes object to draw on.

    """
    ax.clear()  # Clear previous frame
    view_range = cfg.view_range
    res = 100  # grid-map resolution

    # 1.Create grid cells
    x_vals = np.linspace(-view_range, view_range, res)
    y_vals = np.linspace(-view_range, view_range, res)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # 2. Predict the entire grid cell area
    raw_grid = np.c_[xx.ravel(), yy.ravel()]
    extended_grid = engine.transform(raw_grid)

    # Scaling
    scaled_grid = scaler.transform(extended_grid)

    # Predict through vectors
    zz = brain.predict(scaled_grid).reshape(xx.shape)

    # 3. Plot gradient map
    if cfg.color_gradient:
        ax.contourf(xx, yy, zz, 256, cmap="YlOrRd", alpha=0.8)
    else:
        ax.contourf(xx, yy, zz, levels=[0.0, 0.5, 1.0], colors=["white", "#7f7deb"])

    if cfg.show_levels:
        ax.contour(xx, yy, zz, levels=10, colors="black", linewidths=0.2, alpha=0.9)

    if cfg.show_dataset_points:
        if X_raw is not None:
            ax.scatter(
                X_raw[:, 0],
                X_raw[:, 1],
                c=targets.ravel(),
                s=15,  # points size
                cmap="RdYlGn",
                edgecolors="white",
                linewidth=0.5,
                alpha=0.7,
            )

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)

    ax.set_title(f"Epoch: {epoch} | Loss: {current_loss:.6f} | LR: {lr:.6f}")
    ax.set_aspect("equal")

    # Update frames
    plt.draw()
    plt.pause(0.001)  # pause to draw window
