from pathlib import Path

import numpy as np

from src.callbacks import ConsoleLoggerCallback, StorageCallback, VisualizerCallback
from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.optimizer import AdamOptimizer
from src.storage import ExperimentManager
from src.structures import ExperimentContext, MetricFrame, NeuraConfig
from src.trainer import Trainer
from src.visualization import VisualizerEngine


class NeuraPipeline:
    """Orchestrator for the NeuraKitten machine learning workflow.

    This class encapsulates the entire lifecycle of an experiment, managing
    the flow between data generation, feature transformation, scaling,
    and model training. It acts as the high-level API for the project,
    allowing for reproducible and clean experiment execution.

    Attributes:
        cfg (NeuraConfig): The configuration object containing all
            hyperparameters and environment settings.
        factory (DataFactory): Utility for synthetic dataset generation.
        engine (FeatureEngine): Component for coordinate transformations.
        scaler (DataScaler): Normalizer for input features.
        model (DeepNeuralNetwork): The core MLP instance (initialized during run).

    """

    def __init__(self, experiment_name: str, cfg: NeuraConfig) -> None:
        """Initialize the pipeline with a specific configuration.

        Args:
            experiment_name (str): Experiment name for visualization and logging
            cfg (NeuraConfig): An instance of NeuraConfig holding
                all necessary parameters for the experiment.

        """
        self.experiment_name = experiment_name
        self.cfg = cfg
        self.factory = DataFactory(cfg)
        self.engine = FeatureEngine(cfg)
        self.scaler = DataScaler(cfg)
        self.model = None

    def _adjust_visual_range(self, X_raw: np.ndarray) -> None:
        ax_x, ax_y = self.cfg.vis_axes
        padding = self.cfg.padding
        self.cfg.x_min = float(X_raw[:, ax_x].min() - padding)
        self.cfg.x_max = float(X_raw[:, ax_x].max() + padding)
        self.cfg.y_min = float(X_raw[:, ax_y].min() - padding)
        self.cfg.y_max = float(X_raw[:, ax_y].max() + padding)

    def _get_arch_string(self, input_dim: int, output_dim: int) -> str:
        """Help to create a standardized architecture string."""
        return f"[{input_dim}] --> {self.cfg.hidden_layers} --> [{output_dim}]"

    def run(self, mode: str = "train", experiment_path: str | None = None) -> None:
        """Orchestrate the pipeline based on the selected mode.

        Args:
            mode (str): Execution mode ('train' or 'replay').
            experiment_path (str): Path to the experiment folder for replay.

        """
        self.manager = ExperimentManager(base_path=self.cfg.output_dir)

        if mode == "train":
            self._run_train()
        elif mode == "replay":
            if not experiment_path:
                raise ValueError("experiment_path is required for replay mode.")
            self._run_replay(Path(experiment_path))

    def _run_train(self) -> None:
        """Execute the complete pipeline for Training.

        This method coordinates the sequence of operations required to
        train the model and trigger the live visualization.
        """
        # 1. Data Generation
        X_raw, targets = self.factory.generate()

        # 2. Auto-adjust visual range
        if self.cfg.visual_range_auto:
            self._adjust_visual_range(X_raw)

        # 3. Transformation & Scaling
        X_featured = self.engine.transform(X_raw)
        X_transformed = self.scaler.fit_transform(X_featured)

        # 4. Core Components Initialization
        input_dim = X_transformed.shape[1]
        output_dim = targets.shape[1]
        layer_sizes = [input_dim] + self.cfg.hidden_layers + [output_dim]

        self.model = DeepNeuralNetwork(config=self.cfg, layer_sizes=layer_sizes)

        optimizer = AdamOptimizer(self.cfg)
        optimizer.initialize(self.model.weights, self.model.biases)

        # 5. Context & Callbacks
        ctx = ExperimentContext(
            experiment_name=self.experiment_name,
            architecture_log=self._get_arch_string(input_dim, output_dim),
        )

        callbacks = [
            ConsoleLoggerCallback(self.cfg),
            StorageCallback(
                self.cfg,
                self.manager,
                experiment_name=self.experiment_name,
                X_raw=X_raw,
                targets=targets,
            ),
        ]

        if self.cfg.visualize:
            viz_engine = VisualizerEngine(
                cfg=self.cfg,
                engine=self.engine,
                scaler=self.scaler,
                X_raw=X_raw,
                targets=targets,
            )
            callbacks.append(VisualizerCallback(self.cfg, viz_engine))

        # 6. Training
        trainer = Trainer(self.model, optimizer, self.cfg)
        trainer.fit(X_transformed, targets, ctx, callbacks=callbacks)

    def _run_replay(self, exp_dir: Path) -> None:
        """Replay a previously saved experiment from disk."""
        # 1. Load data and restore config
        raw_config = self.manager.load_config(exp_dir)

        self.cfg = NeuraConfig(**raw_config)

        X_raw, targets = self.manager.load_dataset(exp_dir)
        raw_metrics = self.manager.load_metrics(exp_dir)

        metrics_history = [MetricFrame(**m) for m in raw_metrics]

        # 2. Reconstruct components
        self.scaler.fit(self.engine.transform(X_raw))
        X_transformed = self.scaler.transform(self.engine.transform(X_raw))

        input_dim = X_transformed.shape[1]
        output_dim = targets.shape[1]
        layer_sizes = [input_dim] + self.cfg.hidden_layers + [output_dim]

        self.model = DeepNeuralNetwork(config=self.cfg, layer_sizes=layer_sizes)

        viz_engine = VisualizerEngine(
            self.cfg, self.engine, self.scaler, X_raw, targets
        )
        ctx = ExperimentContext(
            experiment_name=f"REPLAY: {exp_dir.name}",
            architecture_log=self._get_arch_string(input_dim, output_dim),
        )

        ctx.metrics = metrics_history

        # 3. Replay loop
        for frame in metrics_history:
            if viz_engine.stop_requested:
                break

            while viz_engine.paused and not viz_engine.stop_requested:
                viz_engine.fig.canvas.start_event_loop(0.1)

            if frame.epoch % self.cfg.checkpoint_interval == 0:
                try:
                    state = self.manager.load_checkpoint(exp_dir, frame.epoch)
                    self.model.weights = list(state["weights"])
                    self.model.biases = list(state["biases"])
                except FileNotFoundError:
                    pass

            # Update context from MetricFrame object attributes
            ctx.epoch = frame.epoch
            ctx.loss = frame.loss
            ctx.accuracy = frame.accuracy
            ctx.lr = frame.lr

            viz_engine.render(self.model, ctx)

            # time.sleep(self.cfg.time_sleep)
