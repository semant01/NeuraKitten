from typing import TYPE_CHECKING, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

matplotlib.use("TkAgg")

if TYPE_CHECKING:
    from src.structures import ExperimentContext, NeuraConfig

    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork


def _create_decision_mesh(
    X_raw: np.ndarray, cfg: "NeuraConfig", engine: "FeatureEngine", scaler: "DataScaler"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a transformed grid for decision boundary plotting.

    Args:
        X_raw: Original input features.
        cfg: Configuration object for range and resolution.
        engine: Feature engine for data transformation.
        scaler: Scaler for data normalization.

    Returns:
        A tuple of (xx, yy, scaled_grid) where xx, yy are meshgrid matrices.

    """
    x_vals = np.linspace(cfg.x_min, cfg.x_max, cfg.resolution)
    y_vals = np.linspace(cfg.y_min, cfg.y_max, cfg.resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Prepare grid for prediction
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    ax_x, ax_y = cfg.vis_axes

    if X_raw.shape[1] > 2:
        # 1. Use mean data if more than 2 feature columns
        X_mean = np.mean(X_raw, axis=0)

        # 2. (N, total_dims)
        full_grid = np.tile(X_mean, (grid_2d.shape[0], 1))

        # 3. replace first two columns with 2D grid
        full_grid[:, ax_x] = grid_2d[:, 0]
        full_grid[:, ax_y] = grid_2d[:, 1]
        raw_grid = full_grid
    else:
        raw_grid = grid_2d

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
    ctx: "ExperimentContext",
    ax_main: Axes,
    ax_loss: Axes,
    ax_acc: Axes,
    ax_info: Axes,
) -> None:
    """Update the live visualization of the neural network's decision boundary.

    Args:
        brain: The trained neural network model instance.
        cfg: Centralized configuration object.
        engine: Feature engineering engine for data transformation.
        scaler: Scaler used for data normalization.
        X_raw: Original input features (Cartesian).
        targets: One-hot encoded labels for the dataset.
        ctx (ExperimentContext): Experiment parameters to be used for logging
        ax_main: Matplotlib axes object to show visualization.
        ax_loss: Matplotlib axes object to show loss chart.
        ax_acc: Matplotlib axes object to show accuracy chart.
        ax_info: Matplotlib axes object to show text info.

    """
    ax_main.clear()

    # 1. Prepare grid and prediction
    xx, yy, scaled_grid = _create_decision_mesh(X_raw, cfg, engine, scaler)

    # 2. Get prediction
    preds = brain.predict(scaled_grid)
    zz = np.argmax(preds, axis=1).reshape(xx.shape)

    num_classes = preds.shape[1]
    plot_targets = np.argmax(targets, axis=1)

    # 3. Plot Decision Boundaries
    levels = np.arange(num_classes + 1) - 0.5
    ax_main.contourf(xx, yy, zz, levels=levels, cmap=cfg.cmap, alpha=0.4)

    # 4. Plot dataset
    ax_x, ax_y = cfg.vis_axes
    if cfg.show_dataset_points and X_raw is not None:
        ax_main.scatter(
            X_raw[:, ax_x],
            X_raw[:, ax_y],
            c=plot_targets,
            s=15,
            cmap=cfg.cmap,
            edgecolors="white",
            linewidth=0.5,
            alpha=1,
        )

    # 5. Configure plot
    ax_main.set_xlim(cfg.x_min, cfg.x_max)
    ax_main.set_ylim(cfg.y_min, cfg.y_max)
    ax_main.set_aspect("auto")  # "auto" "equal"

    # 6. Info text box
    ax_info.clear()
    ax_info.axis("off")
    info_text = (
        f"METRICS\n{'-' * 42}\n"
        f"Epoch:      {ctx.epoch:4d} /{cfg.epochs:4d}\n"
        f"Loss:       {ctx.loss:.6f}\n"
        f"Accuracy:   {ctx.accuracy:.2f}%\n"
        f"LR:         {ctx.lr:.6f}\n\n"
        f"CONFIG\n{'-' * 42}\n"
        f"Model:      {ctx.architecture_log}\n"
        f"Mode:       {cfg.data_mode}\n"
        f"Features:   {cfg.feature_mode}\n"
        f"Noise:      {cfg.noise}"
    )
    ax_info.text(
        0.05,
        0.95,
        info_text,
        transform=ax_info.transAxes,
        verticalalignment="top",
        family="monospace",
        fontsize=9,
    )

    # 7. Loss and Accuracy charts
    ax_loss.clear()

    line_loss = ax_loss.plot(
        ctx.loss_history, color="#e74c3c", linewidth=1.5, label="Loss"
    )
    ax_loss.set_ylabel("Loss", color="#e74c3c")
    ax_loss.tick_params(axis="y", labelcolor="#e74c3c")
    ax_loss.set_ylim(0, max(ctx.loss_history) * 1.1 if ctx.loss_history else 1.0)
    ax_loss.set_yscale("linear")

    ax_acc.clear()
    line_acc = ax_acc.plot(
        ctx.acc_history, color="#3498db", linewidth=1.5, label="Accuracy"
    )
    ax_acc.set_ylabel("Accuracy (%)", color="#3498db")
    ax_acc.yaxis.set_label_position("right")
    ax_acc.tick_params(axis="y", labelcolor="#3498db")
    ax_acc.set_ylim(0, 105)
    ax_loss.set_yscale("linear")

    lines = line_loss + line_acc
    labels = [str(label.get_label()) for label in lines]
    ax_loss.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize="small",
        frameon=False,
    )

    ax_loss.set_title("Learning Progress", fontsize=10)
    ax_loss.grid(True, alpha=0.2)

    # Update screen
    plt.draw()
    plt.pause(0.001)
