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
    accuracy: float,
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
        accuracy: Current prediction accuracy.
        ax: Matplotlib axes object to draw on.

    """
    ax.clear()  # Clear previous frame
    view_range = cfg.view_range
    res = cfg.resolution  # grid-map resolution

    # 1.Create grid cells
    x_vals = np.linspace(-view_range, view_range, res)
    y_vals = np.linspace(-view_range, view_range, res)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # 2. Predict the entire grid cell area
    raw_grid = np.c_[xx.ravel(), yy.ravel()]
    extended_grid = engine.transform(raw_grid)
    scaled_grid = scaler.transform(extended_grid)

    # 3. Predict through vectors
    preds = brain.predict(scaled_grid)  # (10000, num_classes) or (10000, 1)
    zz = np.argmax(preds, axis=1).reshape(xx.shape)
    num_classes = preds.shape[1]
    plot_targets = np.argmax(targets, axis=1)

    # 4. Plot Decision Boundaries
    levels = np.arange(num_classes + 1) - 0.5
    ax.contourf(xx, yy, zz, levels=levels, cmap=cfg.cmap, alpha=0.4)

    # 5. Plot dataset points
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

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)

    ax.set_title(
        f"Epoch: {epoch} | "
        f"Loss: {current_loss:.6f} | "
        f"Acc: {accuracy:.2f}% | "
        f"LR: {lr:.6f}"
    )

    ax.set_aspect("equal")

    # Update frames
    plt.draw()
    plt.pause(0.001)  # pause to draw window
