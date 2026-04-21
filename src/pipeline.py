import logging

from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.structures import ExperimentContext, NeuraConfig
from src.trainer import fit


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

    def _log_setup(self, ctx: ExperimentContext) -> None:
        """Log the experiment parameters to the console for tracking.

        Args:
            ctx (ExperimentContext): Experiment parameters to be logged

        """
        logging.info(f"Run for {self.cfg.epochs} epochs")
        logging.info(
            f"Samples: {self.cfg.samples} ; Batch size: {self.cfg.batch_size}\n"
        )
        logging.info(
            f"Data mode: {self.cfg.data_mode} | Feature mode: {self.cfg.feature_mode}\n"
        )
        logging.info(f"Neural Network: {ctx.architecture_log}\n")

    def _get_arch_string(self, input_dim: int, output_dim: int) -> str:
        """Help to create a standardized architecture string."""
        return f"[{input_dim}] --> {self.cfg.hidden_layers} --> [{output_dim}]"

    def run(self) -> None:
        """Execute the complete pipeline: Generation -> Transformation -> Training.

        This method coordinates the sequence of operations required to
        train the model and trigger the live visualization.
        """
        # 1. Generation
        X_raw, targets = self.factory.generate()

        # 2. Transformation & Scaling
        X_featured = self.engine.transform(X_raw)
        X_transformed = self.scaler.fit_transform(X_featured)

        # 3. Model Initialization
        input_dim = X_transformed.shape[1]
        output_dim = targets.shape[1]

        self.model = DeepNeuralNetwork(
            config=self.cfg,
            layer_sizes=[input_dim] + self.cfg.hidden_layers + [output_dim],
        )

        arch_log = self._get_arch_string(input_dim, output_dim)

        ctx = ExperimentContext(
            experiment_name=self.experiment_name,
            architecture_log=arch_log,
        )

        self._log_setup(ctx)

        # 4. Training
        fit(
            model=self.model,
            inputs=X_transformed,
            targets=targets,
            cfg=self.cfg,
            ctx=ctx,
            X_raw=X_raw,
            scaler=self.scaler,
            engine=self.engine,
        )
