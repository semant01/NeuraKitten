from dataclasses import dataclass, field


@dataclass
class NeuraConfig:
    """Configuration dataset.

    Centralized repository for all hyperparameters, architecture settings,
    and data generation parameters of the NeuraKitten project.

    This class serves as a single source of truth for the entire pipeline,
    ensuring consistency across data factory, model initialization,
    and training processes.
    """

    # --- Network Architecture ---
    hidden_layers: list[int] = field(default_factory=lambda: [12, 12])

    # --- Hyperparameters & Training ---
    epochs: int = 501
    batch_size: int = 32
    balanced_batches: bool = True
    initial_lr: float = 0.001
    decay_rate: float = 0.01
    min_lr: float = 1e-5

    # --- ADAM Optimizer ---
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8  # also used for Categorical Cross-Entropy

    # --- Data Generation ---
    samples: int = 2048
    noise: float = 0
    data_mode: str = "multidonut"  # "multidonut", "spirals", "rhodonea"

    #       Multi-donut
    mdonut_radii: list[float] = field(
        default_factory=lambda: [
            1.2,
            0.7,
            0.5,
            0.3,
        ]
    )
    mdonut_r_evenly_dist: bool = False  # distribute evenly along the radius
    #                            or closer to the center

    #       Donut
    donut_r_inner: float = 0.3  # Inner radius
    donut_r_outer: float = 0.7  # Outer radius
    donut_r_evenly_dist: bool = False  # distribute evenly along the radius
    #                            or closer to the center

    #       Spiral
    spiral_num_classes: int = 3
    spiral_turns: float = 2.5  # number of semi-turns
    spiral_max_radius: float = 1.0

    #       Rhodonea
    rhodonea_k: float = 3.0  # 2k petals for rhodonea
    rhodonea_max_radius: float = 1.0  # Maximum radius of rhodonea
    rhodonea_num_classes: int = 1  # Number of classes in addition to background class
    rhodonea_r_evenly_dist: bool = False

    #       Iris
    iris_pca: bool = True  # Principal Component Analysis

    # --- Feature Engineering ---
    feature_mode: str = "polar"  # "cartesian", "polar"
    use_squares: bool = False
    use_interaction: bool = False
    use_trig: bool = False

    # --- Data Scaling ---
    feature_range: tuple[float, float] = (-1, 1)

    # --- UI / UX / Visualization ---
    visualize: bool = True
    resolution: int = 300
    vis_axes: tuple[int, int] = (0, 1)  # Feature indices for drawing on X and Y axes
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5

    cmap: str = "CMRmap"
    #     cmap options:  "gist_stern", "tab20c", "CMRmap", "nipy_spectral" ,"gnuplot2"
    show_dataset_points: bool = True
    frame_visual: int = 1
    frame_log: int = 50

    # --- Reproducibility ---
    # Reproducibility anchor and the Answer to the Ultimate Question of Life.
    seed: int | None = 42


@dataclass
class ExperimentContext:
    """Encapsulates the full state and configuration of a training experiment.

    This class acts as a centralized data hub, carrying both static configuration
    metadata and dynamic training metrics to be used by loggers and visualizers.

    Attributes:
        architecture_log (str): String representation of the NN structure.
        epoch (int): Current training epoch.
        loss (float): Current loss value.
        accuracy (float): Current model accuracy (0.0 to 1.0).
        lr (float): Current learning rate.

    """

    # Static Metadata
    experiment_name: str
    architecture_log: str

    # Dynamic Metrics (initialized with defaults)
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    lr: float = 0.0

    loss_history: list[float] = field(default_factory=list)
    acc_history: list[float] = field(default_factory=list)

    def update_metrics(
        self, epoch: int, loss: float, accuracy: float, lr: float
    ) -> None:
        """Update the running metrics and history of the experiment."""
        self.epoch = epoch
        self.loss = loss
        self.accuracy = accuracy
        self.lr = lr
        self.loss_history.append(loss)
        self.acc_history.append(accuracy)
