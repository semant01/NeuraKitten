from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

if TYPE_CHECKING:
    import matplotlib.backend_bases
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .data_utils import DataScaler, FeatureEngine
    from .model import DeepNeuralNetwork
    from .structures import ExperimentContext, NeuraConfig


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

    This is a heavy operation that should be called once per experiment.

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


class VisualizerEngine:
    """High-performance rendering engine for neural network training progress.

    It pre-calculates the decision mesh and manages the matplotlib figure state,
    avoiding redundant computations during the training loop.
    """

    def __init__(
        self,
        cfg: NeuraConfig,
        engine: FeatureEngine,
        scaler: DataScaler,
        X_raw: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Initialize the engine and prepare static data.

        Args:
            cfg: Centralized configuration object.
            engine: Feature engineering component.
            scaler: Data normalization component.
            X_raw: Original input features for scatter plotting.
            targets: One-hot encoded labels.

        """
        self.cfg = cfg
        self.X_raw = X_raw
        self.targets = targets
        self.plot_targets = np.argmax(targets, axis=1)

        # 1. Pre-calculate the mesh
        self.xx, self.yy, self.scaled_grid = _create_decision_mesh(
            X_raw, cfg, engine, scaler
        )

        # 2. Setup Figure and Axes
        self.fig: Figure = plt.figure(figsize=(14, 8))
        self.gs = self.fig.add_gridspec(
            2, 3, wspace=0.3, hspace=0.3, width_ratios=[1, 1, 1.5]
        )

        self.ax_main: Axes = self.fig.add_subplot(self.gs[:, :2])
        self.ax_loss: Axes = self.fig.add_subplot(self.gs[0, 2])
        self.ax_acc: Axes = self.ax_loss.twinx()
        self.ax_info: Axes = self.fig.add_subplot(self.gs[1, 2])
        self.ax_info.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        self.ax_info.axis("off")

        # 3. Handle Interactive Events
        self.stop_requested: bool = False
        self.paused: bool = False

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    def _on_close(self, event: matplotlib.backend_bases.Event) -> None:
        """Handle window close event."""
        self.stop_requested = True

    def _on_key(self, event: matplotlib.backend_bases.Event) -> None:
        """Handle key press events for manual interruption."""
        if isinstance(event, matplotlib.backend_bases.KeyEvent):
            if event.key == "escape":
                self.stop_requested = True
            elif event.key in [" ", "p", "P"]:
                self.paused = not self.paused

    def render(self, brain: DeepNeuralNetwork, ctx: ExperimentContext) -> None:
        """Perform a single rendering pass of the current model state.

        Args:
            brain: The neural network instance for inference.
            ctx: Current experiment context with metrics and history.

        """
        # --- 1. Decision Boundary ---
        self.ax_main.clear()

        preds = brain.predict(self.scaled_grid)
        zz = np.argmax(preds, axis=1).reshape(self.xx.shape)
        num_classes = preds.shape[1]

        levels = np.arange(num_classes + 1) - 0.5
        self.ax_main.contourf(
            self.xx, self.yy, zz, levels=levels, cmap=self.cfg.cmap, alpha=0.4
        )

        # --- 2. Dataset Points ---
        if self.cfg.show_dataset_points:
            ax_x, ax_y = self.cfg.vis_axes
            self.ax_main.scatter(
                self.X_raw[:, ax_x],
                self.X_raw[:, ax_y],
                c=self.plot_targets,
                s=15,
                cmap=self.cfg.cmap,
                edgecolors="white",
                linewidth=0.5,
            )
        self.ax_main.set_xlim(self.cfg.x_min, self.cfg.x_max)
        self.ax_main.set_ylim(self.cfg.y_min, self.cfg.y_max)
        self.ax_main.set_title(f"Decision Boundary (Epoch {ctx.epoch})")

        # --- 3. Info Panel ---
        self.ax_info.clear()
        # self.ax_info.axis("off")
        info_text = (
            f"METRICS\n{'-' * 30}\n"
            f"Epoch:    {ctx.epoch:4d} / {self.cfg.epochs}\n"
            f"Loss:     {ctx.loss:.6f}\n"
            f"Accuracy: {ctx.accuracy:.2f}%\n"
            f"LR:       {ctx.lr:.6f}\n\n"
            f"CONFIG\n{'-' * 30}\n"
            f"Data:     {self.cfg.data_mode}\n"
            f"Features: {self.cfg.feature_mode}\n"
            f"Arch:     {ctx.architecture_log}\n"
            f"Seed:     {self.cfg.seed}"
        )
        self.ax_info.text(
            0.05,
            0.95,
            info_text,
            transform=self.ax_info.transAxes,
            va="top",
            family="monospace",
            fontsize=9,
        )

        # --- 4. Learning Curves ---
        self._render_metrics(ctx)

        # --- 5. Update Screen ---
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

    def _render_metrics(self, ctx: ExperimentContext) -> None:
        """Render loss and accuracy charts filtered by current context epoch."""
        if not ctx.metrics:
            return

        self.ax_loss.clear()
        self.ax_acc.clear()

        # Filter history to show only frames up to the current epoch
        visible_history = [m for m in ctx.metrics if m.epoch <= ctx.epoch]
        if not visible_history:
            return

        epochs = [frame.epoch for frame in visible_history]
        losses = [frame.loss for frame in visible_history]
        accuracies = [frame.accuracy for frame in visible_history]

        (line1,) = self.ax_loss.plot(
            epochs, losses, color="#e74c3c", label="Loss", linewidth=1.5
        )
        self.ax_loss.set_ylabel("Loss", color="#e74c3c", fontsize=10, labelpad=5)
        self.ax_loss.tick_params(axis="y", labelcolor="#e74c3c")
        self.ax_loss.grid(True, alpha=0.3)

        (line2,) = self.ax_acc.plot(
            epochs, accuracies, color="#2ecc71", label="Accuracy", linewidth=1.5
        )
        self.ax_acc.set_ylabel("Accuracy %", color="#2ecc71", fontsize=10, labelpad=5)
        self.ax_acc.tick_params(axis="y", labelcolor="#2ecc71")
        self.ax_acc.yaxis.set_label_position("right")
        self.ax_acc.yaxis.tick_right()

        self.ax_acc.set_ylim(0, 105)

        self.ax_loss.set_title("Training Progress", fontsize=12, pad=10)

    def close(self) -> None:
        """Close the visualization window."""
        plt.close(self.fig)
