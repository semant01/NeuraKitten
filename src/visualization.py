import matplotlib
import numpy as np
from matplotlib.axes import Axes

matplotlib.use("TkAgg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def live_plot(
    brain: "DeepNeuralNetwork",
    engine: "FeatureEngine",
    scaler: "DataScaler",
    X_raw: np.ndarray,
    targets: np.ndarray,
    epoch: int,
    LR: float,
    loss: float,
    ax: Axes,
    color_gradient: bool = False,
    show_dataset_points: bool = False,
    show_levels: bool = False,
) -> None:
    """Update the live visualization of the neural network's decision boundary.

    Args:
        brain: The neural network model instance.
        engine: Feature engineering engine for data transformation.
        scaler: Scaler used for data normalization.
        X_raw: Original input features (Cartesian).
        targets: Labels for the dataset.
        epoch: Current training epoch.
        LR: Current learning rate.
        loss: Current loss value.
        ax: Matplotlib axes object to draw on.
        color_gradient: Whether to show smooth probability gradients.
        show_dataset_points: Whether to overlay data points on the plot.
        show_levels: Whether to draw contour lines.

    """
    ax.clear()  # Clear previous frame
    view_range = 1.5
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
    if color_gradient:
        ax.contourf(xx, yy, zz, 256, cmap="YlOrRd", alpha=0.8)
    else:
        ax.contourf(xx, yy, zz, levels=[0.0, 0.5, 1.0], colors=["white", "#7f7deb"])

    if show_levels:
        ax.contour(xx, yy, zz, levels=10, colors="black", linewidths=0.2, alpha=0.9)

    if show_dataset_points:
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

    ax.set_title(f"Epoch: {epoch} | Loss: {loss:.6f} | LR: {LR:.4f}")
    ax.set_aspect("equal")

    # Update frames
    plt.draw()
    plt.pause(0.001)  # pause to draw window
